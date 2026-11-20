"""colab_core.agent.drivers —— 原生轻量 ReAct 驱动与 Claude Code CLI 外部引擎驱动。"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from colab_core.agent.protocols import (
    SUPPORTED_API_FORMATS,
    AnthropicAdapter,
    GeminiAdapter,
    OpenAIChatAdapter,
    OpenAIResponsesAdapter,
    ToolExecRecord,
    _emit_event,
    _normalize_reasoning_effort,
    _openai_effort_candidates,
    _strip_mcp_prefix,
    get_protocol_adapter,
    open_http_stream,
    open_stream_with_effort_fallback,
)
from colab_core.agent.safety import _redact_obj, _safe_json_dumps, _summarize_tool_result
from colab_core.agent.specs import AGENT_TOOL_SPECS, DEFAULT_SYSTEM_PROMPT, get_tool_spec
from colab_core.runtime import (
    CREATE_NO_WINDOW,
    CancelToken,
    clean_subprocess_env,
    kill_process_tree,
    suppress_logged,
)
from colab_security import redact_secrets, register_ephemeral_secret

_PKG_DIR = Path(__file__).resolve().parents[2]


# =========================================================================
# 1. 原生轻量 ReAct 驱动（通过协议适配器策略表统一驱动四协议）
# =========================================================================


class NativeLightDriver:
    """纯标准库轻量原生 ReAct Tool-Calling 驱动，支持 OpenAI / Responses / Anthropic / Gemini SSE 流。"""

    def __init__(
        self,
        base_url: str = "",
        api_key: str = "",
        model: str = "",
        api_format: str = "openai",
        messages: list[dict[str, Any]] | None = None,
        tool_specs: Sequence[dict[str, Any]] | None = None,
        tool_executor: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
        on_event: Callable[[dict[str, Any]], None] | None = None,
        cancel_token: CancelToken | None = None,
        system_prompt: str | None = None,
        extra_headers: dict[str, str] | None = None,
        max_turns: int = 8,
        timeout_sec: float = 120.0,
        reasoning_effort: str = "auto",
    ) -> None:
        self.base_url = str(base_url or "").strip()
        self.api_key = str(api_key or "").strip()
        self.model = str(model or "").strip()
        fmt = str(api_format or "openai").strip().lower()
        self.api_format = fmt if fmt in SUPPORTED_API_FORMATS else "openai"
        self.reasoning_effort = _normalize_reasoning_effort(reasoning_effort)
        self.messages = list(messages) if messages is not None else []
        self.tool_specs = tuple(tool_specs) if tool_specs is not None else AGENT_TOOL_SPECS
        self.tool_executor = tool_executor
        self.on_event = on_event
        self.cancel_token = cancel_token
        self.system_prompt = system_prompt if system_prompt is not None else DEFAULT_SYSTEM_PROMPT
        self.extra_headers = dict(extra_headers or {})
        self.max_turns = max(1, min(16, int(max_turns or 8)))
        self.timeout_sec = max(5.0, float(timeout_sec or 120.0))
        if self.api_key:
            register_ephemeral_secret(self.api_key)

    def _emit(self, on_event: Callable[[dict[str, Any]], None] | None, evt: dict[str, Any]) -> None:
        _emit_event(on_event, evt)

    def _build_openai_messages(self, raw_messages: Sequence[dict[str, Any]], sys_prompt: str) -> list[dict[str, Any]]:
        return OpenAIChatAdapter.build_messages(raw_messages, sys_prompt)

    def _build_anthropic_messages(
        self, raw_messages: Sequence[dict[str, Any]], sys_prompt: str
    ) -> tuple[str, list[dict[str, Any]]]:
        return AnthropicAdapter.build_messages(raw_messages, sys_prompt)

    def _build_responses_input(
        self, raw_messages: Sequence[dict[str, Any]], sys_prompt: str
    ) -> tuple[str, list[dict[str, Any]]]:
        return OpenAIResponsesAdapter.build_input(raw_messages, sys_prompt)

    def _build_gemini_contents(
        self, raw_messages: Sequence[dict[str, Any]], sys_prompt: str
    ) -> tuple[str, list[dict[str, Any]]]:
        return GeminiAdapter.build_contents(raw_messages, sys_prompt)

    def _open_stream(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        cancel_token: CancelToken | None,
        secret_tuple: tuple[str, ...],
    ) -> Any:
        return open_http_stream(
            url,
            headers,
            payload,
            base_url=self.base_url,
            timeout_sec=self.timeout_sec,
            cancel_token=cancel_token,
            secret_tuple=secret_tuple,
        )

    def _open_stream_with_effort_fallback(
        self,
        url: str,
        headers: dict[str, str],
        base_payload: dict[str, Any],
        effort_candidates: Sequence[str],
        apply_effort: Callable[[dict[str, Any], str], None],
        cancel_token: CancelToken | None,
        secret_tuple: tuple[str, ...],
    ) -> Any:
        return open_stream_with_effort_fallback(
            url,
            headers,
            base_payload,
            effort_candidates,
            apply_effort,
            base_url=self.base_url,
            timeout_sec=self.timeout_sec,
            cancel_token=cancel_token,
            secret_tuple=secret_tuple,
        )

    @staticmethod
    def _openai_effort_candidates(effort: str) -> list[str]:
        return _openai_effort_candidates(effort)

    @staticmethod
    def _parse_tool_arguments(raw_args: Any) -> tuple[dict[str, Any], str]:
        """解析工具调用参数为字典；遇到损坏 JSON 时返回友好错误信息。"""
        if isinstance(raw_args, dict):
            return dict(raw_args), ""
        arg_txt = str(raw_args or "").strip()
        if not arg_txt:
            return {}, ""
        try:
            parsed = json.loads(arg_txt)
            return (parsed if isinstance(parsed, dict) else {"value": parsed}), ""
        except json.JSONDecodeError as exc:
            return {}, f"工具参数 JSON 解析失败: {exc}"

    def _execute_tool_calls(
        self,
        tool_calls: Sequence[dict[str, Any]],
        *,
        turn_idx: int,
        tier_by_name: dict[str, str],
        tool_executor: Callable[[str, dict[str, Any]], dict[str, Any]] | None,
        on_event: Callable[[dict[str, Any]], None] | None,
        cancel_token: CancelToken | None,
        secret_tuple: tuple[str, ...],
    ) -> tuple[list[ToolExecRecord], str | None]:
        """顺序执行本轮全部工具调用并广播事件，返回 `(执行记录列表, 取消原因或None)`。"""
        records: list[ToolExecRecord] = []
        for idx, tc in enumerate(tool_calls):
            tc_id = str(tc.get("id") or f"call_{turn_idx}_{idx}")
            tc_name = str(tc["name"]).strip()
            raw_args = tc.get("arguments")
            args_dict, parse_err = self._parse_tool_arguments(raw_args)

            tier = tier_by_name.get(tc_name, "exec")
            self._emit(on_event, {"type": "tool_start", "name": tc_name, "args": _redact_obj(args_dict, secret_tuple), "tier": tier})

            t0 = time.monotonic()
            if parse_err:
                res_obj: dict[str, Any] = {"ok": False, "error": parse_err}
            elif tool_executor is None:
                res_obj = {"ok": False, "error": f"未注册工具执行器，无法调用 {tc_name}"}
            else:
                try:
                    raw_res = tool_executor(tc_name, args_dict)
                    res_obj = raw_res if isinstance(raw_res, dict) else {"ok": True, "result": raw_res}
                except Exception as exc:
                    res_obj = {"ok": False, "error": redact_secrets(str(exc) or type(exc).__name__, secret_tuple)}

            elapsed_ms = int(round((time.monotonic() - t0) * 1000))
            ok_flag, summary = _summarize_tool_result(res_obj, secret_tuple)
            self._emit(
                on_event,
                {"type": "tool_end", "name": tc_name, "ok": ok_flag, "summary": summary, "elapsed_ms": elapsed_ms},
            )
            res_json = _safe_json_dumps(res_obj, extra_secrets=secret_tuple)
            meta = {"thought_signature": tc["thought_signature"]} if tc.get("thought_signature") else {}
            records.append(
                ToolExecRecord(
                    call_id=tc_id,
                    name=tc_name,
                    raw_args=raw_args,
                    args_dict=args_dict,
                    res_json=res_json,
                    ok=ok_flag,
                    meta=meta,
                )
            )
            if (cancel_token is not None and cancel_token.is_cancelled) or bool(res_obj.get("cancelled")):
                reason = redact_secrets(str(res_obj.get("reason") or res_obj.get("error") or "操作已取消"), secret_tuple)
                return records, reason
        return records, None

    @staticmethod
    def _build_run_result(
        *,
        ok: bool,
        all_texts: Sequence[str],
        all_reasonings: Sequence[str],
        turns: int,
        cancelled: bool,
        error: str,
        total_tools: int,
    ) -> dict[str, Any]:
        """构造统一格式的 ReAct 轮次返回字典。"""
        return {
            "ok": ok,
            "text": "\n".join(t for t in all_texts if t),
            "reasoning": "\n".join(r for r in all_reasonings if r),
            "turns": turns,
            "cancelled": cancelled,
            "error": error,
            "tool_calls_count": total_tools,
        }

    def _resolve_run_config(self, messages: list[dict[str, Any]] | None, overrides: dict[str, Any]) -> dict[str, Any]:
        """合并构造默认值与单次 `run()` 覆盖参数，完成 API 密钥脱敏注册。"""
        api_key = str(overrides.get("api_key") if "api_key" in overrides else self.api_key).strip()
        if api_key:
            register_ephemeral_secret(api_key)
        fmt = str(overrides.get("api_format") or self.api_format).strip().lower()
        extra_headers = dict(self.extra_headers)
        if isinstance(overrides.get("extra_headers"), dict):
            extra_headers.update(overrides["extra_headers"])
        return {
            "base_url": str(overrides.get("base_url") or self.base_url).strip(),
            "api_key": api_key,
            "secret_tuple": (api_key,) if api_key else (),
            "model": str(overrides.get("model") if "model" in overrides else self.model).strip(),
            "api_format": fmt if fmt in SUPPORTED_API_FORMATS else "openai",
            "reasoning_effort": _normalize_reasoning_effort(
                overrides["reasoning_effort"] if "reasoning_effort" in overrides else self.reasoning_effort
            ),
            "tool_specs": tuple(overrides.get("tool_specs") or self.tool_specs),
            "tool_executor": overrides.get("tool_executor") or self.tool_executor,
            "on_event": overrides.get("on_event") or self.on_event,
            "cancel_token": overrides.get("cancel_token") or self.cancel_token,
            "sys_prompt": str((overrides["system_prompt"] if "system_prompt" in overrides else self.system_prompt) or ""),
            "extra_headers": extra_headers,
            "raw_msgs": list(messages if messages is not None else self.messages),
        }

    def run(self, messages: list[dict[str, Any]] | None = None, **overrides: Any) -> dict[str, Any]:
        """执行原生 ReAct 多轮对话与工具调用闭环（支持四协议思考强度注入与流式推理提取）。"""
        cfg = self._resolve_run_config(messages, overrides)
        cancel_token: CancelToken | None = cfg["cancel_token"]
        if cancel_token is not None and cancel_token.is_cancelled:
            return self._build_run_result(
                ok=False, all_texts=(), all_reasonings=(), turns=0, cancelled=True, error="任务在启动前已取消", total_tools=0
            )

        tier_by_name = {str(s["name"]): str(s.get("tier") or "exec") for s in cfg["tool_specs"] if isinstance(s, dict) and s.get("name")}
        adapter = get_protocol_adapter(cfg["api_format"])
        endpoint = adapter.build_endpoint(cfg["base_url"], cfg["model"])
        headers = adapter.build_headers(cfg["api_key"], cfg["extra_headers"])
        tools_payload = adapter.build_tools(cfg["tool_specs"])
        sys_text, conv_msgs, proto_state = adapter.init_conversation(cfg["raw_msgs"], cfg["sys_prompt"])

        all_texts: list[str] = []
        all_reasonings: list[str] = []
        total_tools = 0
        turns_done = 0

        try:
            for turn_idx in range(self.max_turns):
                if cancel_token is not None and cancel_token.is_cancelled:
                    return self._build_run_result(
                        ok=False, all_texts=all_texts, all_reasonings=all_reasonings, turns=turns_done,
                        cancelled=True, error="操作已手动取消", total_tools=total_tools,
                    )

                turns_done = turn_idx + 1
                turn_res = adapter.stream_turn(
                    endpoint, headers, sys_text, conv_msgs, tools_payload,
                    model=cfg["model"], base_url=cfg["base_url"], timeout_sec=self.timeout_sec,
                    reasoning_effort=cfg["reasoning_effort"], state=proto_state, turn_idx=turn_idx,
                    on_event=cfg["on_event"], cancel_token=cancel_token, secret_tuple=cfg["secret_tuple"],
                )
                if turn_res.text:
                    all_texts.append(turn_res.text)
                if turn_res.reasoning:
                    all_reasonings.append(turn_res.reasoning)

                if cancel_token is not None and cancel_token.is_cancelled:
                    return self._build_run_result(
                        ok=False, all_texts=all_texts, all_reasonings=all_reasonings, turns=turns_done,
                        cancelled=True, error="操作已手动取消", total_tools=total_tools,
                    )
                if not turn_res.tool_calls:
                    return self._build_run_result(
                        ok=True, all_texts=all_texts, all_reasonings=all_reasonings, turns=turns_done,
                        cancelled=False, error="", total_tools=total_tools,
                    )

                records, cancel_reason = self._execute_tool_calls(
                    turn_res.tool_calls,
                    turn_idx=turn_idx,
                    tier_by_name=tier_by_name,
                    tool_executor=cfg["tool_executor"],
                    on_event=cfg["on_event"],
                    cancel_token=cancel_token,
                    secret_tuple=cfg["secret_tuple"],
                )
                total_tools += len(records)
                if cancel_reason is not None:
                    return self._build_run_result(
                        ok=False, all_texts=all_texts, all_reasonings=all_reasonings, turns=turns_done,
                        cancelled=True, error=cancel_reason, total_tools=total_tools,
                    )
                adapter.append_tool_round(conv_msgs, turn_res, records, proto_state, turn_idx)

            if not any(all_texts):
                hint = "已达到单轮最大工具调用步数上限 (8 步)，请根据已执行结果继续。"
                self._emit(cfg["on_event"], {"type": "delta", "text": hint})
                all_texts.append(hint)
            return self._build_run_result(
                ok=True, all_texts=all_texts, all_reasonings=all_reasonings, turns=turns_done,
                cancelled=False, error="", total_tools=total_tools,
            )
        except Exception as exc:
            cancelled_now = bool(cancel_token is not None and cancel_token.is_cancelled)
            err_msg = "操作已手动取消" if cancelled_now else redact_secrets(str(exc) or type(exc).__name__, cfg["secret_tuple"])
            return self._build_run_result(
                ok=False, all_texts=all_texts, all_reasonings=all_reasonings, turns=turns_done,
                cancelled=cancelled_now, error=err_msg, total_tools=total_tools,
            )


# =========================================================================
# 2. Claude Code CLI 外部引擎驱动
# =========================================================================


@dataclass
class _ClaudeStreamState:
    """单次 Claude Code CLI 流式执行期的增量状态容器。"""

    text_parts: list[str] = field(default_factory=list)
    reasoning_parts: list[str] = field(default_factory=list)
    pending_tools: dict[str, tuple[str, float]] = field(default_factory=dict)
    stderr_lines: list[str] = field(default_factory=list)
    emitted_delta: bool = False
    emitted_thinking: bool = False
    done_emitted: bool = False


_CLAUDE_EFFORT_ENV_MAP: dict[str, str] = {
    "minimal": "low",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "max",
    "max": "max",
}


class ClaudeCodeDriver:
    """基于本地 `claude` CLI 与临时 `--mcp-config` 挂载 Colab MCP 工具的外部引擎驱动。"""

    def __init__(
        self,
        ctl_script: Path | None = None,
        model: str = "",
        on_event: Callable[[dict[str, Any]], None] | None = None,
        cancel_token: CancelToken | None = None,
        system_prompt: str | None = None,
        ctl_prefix: list[str] | None = None,
        timeout_sec: float = 300.0,
        claude_bin: str | None = None,
        reasoning_effort: str = "auto",
        allow_mcp_mutations: bool = False,
    ) -> None:
        self.ctl_script = Path(ctl_script) if ctl_script is not None else (_PKG_DIR / "colab_ctl.py")
        self.model = str(model or "").strip()
        self.on_event = on_event
        self.cancel_token = cancel_token
        self.system_prompt = system_prompt if system_prompt is not None else DEFAULT_SYSTEM_PROMPT
        self.ctl_prefix = list(ctl_prefix) if ctl_prefix else None
        self.timeout_sec = max(10.0, float(timeout_sec or 300.0))
        self.claude_bin = str(claude_bin or "").strip()
        self.reasoning_effort = _normalize_reasoning_effort(reasoning_effort)
        self.allow_mcp_mutations = bool(allow_mcp_mutations)

    @staticmethod
    def is_available() -> bool:
        """检测系统 PATH 中是否已安装 `claude` CLI。"""
        return bool(shutil.which("claude"))

    @staticmethod
    def _format_prompt(messages: list[dict[str, Any]] | str) -> str:
        if isinstance(messages, str):
            return messages.strip()
        lines: list[str] = []
        recent = [m for m in messages if isinstance(m, dict)][-12:]
        if len(recent) == 1 and str(recent[0].get("role") or "user") == "user":
            return str(recent[0].get("content") or "").strip()
        for m in recent:
            role = str(m.get("role") or "user").strip().lower()
            content = str(m.get("content") or "").strip()
            if not content:
                continue
            tag = "用户" if role == "user" else ("助手" if role == "assistant" else "系统")
            lines.append(f"[{tag}] {content}")
        return "\n\n".join(lines).strip()

    def _emit(self, on_event: Callable[[dict[str, Any]], None] | None, evt: dict[str, Any]) -> None:
        _emit_event(on_event, evt)

    def _prepare_mcp_and_cmd(
        self,
        cfg_path: Path,
        claude_exe: str,
        model: str,
        sys_prompt: str,
        prompt_txt: str,
    ) -> tuple[list[str], bool]:
        """写入临时 `--mcp-config` 并构造 `claude -p --output-format stream-json` 命令参数。"""
        mcp_cmd = list(self.ctl_prefix) if self.ctl_prefix else [sys.executable, str(self.ctl_script)]
        server_entry: dict[str, Any] = {
            "command": mcp_cmd[0],
            "args": [*mcp_cmd[1:], "mcp-stdio"],
        }
        if self.allow_mcp_mutations:
            # 仅在显式授权无人值守变更时，才向无头 MCP 子进程下发变更许可开关
            server_entry["env"] = {"COLAB_MCP_ALLOW_MUTATIONS": "1"}
        mcp_cfg = {"mcpServers": {"colab-studio": server_entry}}
        cfg_path.write_text(json.dumps(mcp_cfg, ensure_ascii=False, indent=2), encoding="utf-8")

        cmd = [
            claude_exe,
            "-p",
            "--output-format",
            "stream-json",
            "--verbose",
            "--mcp-config",
            str(cfg_path),
        ]
        if model:
            cmd.extend(["--model", model])
        if sys_prompt:
            # Windows .cmd/.bat 经由 cmd.exe 解析时会因换行符截断整条命令行，统一折叠为空格
            sys_one_line = re.sub(r"\s*[\r\n]+\s*", " ", sys_prompt).strip()
            if sys_one_line:
                cmd.extend(["--append-system-prompt", sys_one_line])

        use_stdin = ("\n" in prompt_txt) or ("\r" in prompt_txt) or (len(prompt_txt) > 1500)
        if not use_stdin:
            cmd.append(prompt_txt)
        return cmd, use_stdin

    def _handle_stream_delta(
        self, obj: dict[str, Any], cb: Callable[[dict[str, Any]], None] | None, st: _ClaudeStreamState
    ) -> None:
        """处理 `stream_event` 内部 `content_block_delta` 文本与思考增量。"""
        inner = obj.get("event") or {}
        if not (isinstance(inner, dict) and inner.get("type") == "content_block_delta"):
            return
        delta = inner.get("delta") or {}
        if not isinstance(delta, dict):
            return
        dtype = delta.get("type")
        if dtype == "text_delta":
            txt = redact_secrets(str(delta.get("text") or ""))
            if txt:
                st.emitted_delta = True
                st.text_parts.append(txt)
                self._emit(cb, {"type": "delta", "text": txt})
        elif dtype == "thinking_delta":
            t_txt = redact_secrets(str(delta.get("thinking") or ""))
            if t_txt:
                st.emitted_thinking = True
                st.reasoning_parts.append(t_txt)
                self._emit(cb, {"type": "reasoning_delta", "text": t_txt})

    def _handle_assistant_event(
        self, obj: dict[str, Any], cb: Callable[[dict[str, Any]], None] | None, st: _ClaudeStreamState
    ) -> None:
        """处理 `assistant` 聚合消息块（兜底补发未增量流出的文本/思考与 tool_use 启动事件）。"""
        for item in (obj.get("message") or {}).get("content") or []:
            if not isinstance(item, dict):
                continue
            itype = item.get("type")
            if itype == "thinking" and item.get("thinking") and not st.emitted_thinking:
                t_txt = redact_secrets(str(item["thinking"]))
                st.reasoning_parts.append(t_txt)
                self._emit(cb, {"type": "reasoning_delta", "text": t_txt})
            elif itype == "text" and item.get("text") and not st.emitted_delta:
                txt = redact_secrets(str(item["text"]))
                st.text_parts.append(txt)
                self._emit(cb, {"type": "delta", "text": txt})
            elif itype == "tool_use":
                tid = str(item.get("id") or "")
                tname = _strip_mcp_prefix(str(item.get("name") or ""))
                spec = get_tool_spec(tname)
                tier = spec["tier"] if spec else "exec"
                targs = _redact_obj(item.get("input") if isinstance(item.get("input"), dict) else {})
                if tid:
                    st.pending_tools[tid] = (tname, time.monotonic())
                self._emit(cb, {"type": "tool_start", "name": tname, "args": targs, "tier": tier})

    def _handle_user_tool_result(
        self, obj: dict[str, Any], cb: Callable[[dict[str, Any]], None] | None, st: _ClaudeStreamState
    ) -> None:
        """处理 `user` 回灌的 `tool_result` 块，计算耗时并广播 `tool_end` 事件。"""
        for item in (obj.get("message") or {}).get("content") or []:
            if not isinstance(item, dict) or item.get("type") != "tool_result":
                continue
            tid = str(item.get("tool_use_id") or "")
            tname, t0 = st.pending_tools.pop(tid, ("tool", time.monotonic()))
            elapsed_ms = int(round((time.monotonic() - t0) * 1000))
            raw_c = item.get("content")
            if isinstance(raw_c, list):
                summary = " ".join(str(x.get("text") or "") for x in raw_c if isinstance(x, dict)).strip()
            else:
                summary = str(raw_c or "").strip()
            self._emit(
                cb,
                {
                    "type": "tool_end",
                    "name": tname,
                    "ok": not bool(item.get("is_error")),
                    "summary": redact_secrets(summary)[:220],
                    "elapsed_ms": elapsed_ms,
                },
            )
        st.emitted_delta = False
        st.emitted_thinking = False

    def _handle_stream_json_event(
        self, obj: dict[str, Any], cb: Callable[[dict[str, Any]], None] | None, st: _ClaudeStreamState
    ) -> None:
        """分发并转换单条 `claude --output-format stream-json` 事件对象。"""
        evt_type = str(obj.get("type") or "").strip()
        if evt_type == "stream_event":
            self._handle_stream_delta(obj, cb, st)
        elif evt_type == "delta":
            txt = redact_secrets(str(obj.get("text") or ""))
            if txt:
                st.emitted_delta = True
                st.text_parts.append(txt)
                self._emit(cb, {"type": "delta", "text": txt})
        elif evt_type == "tool_start":
            tname = _strip_mcp_prefix(str(obj.get("name") or ""))
            spec = get_tool_spec(tname)
            tier = str(obj.get("tier") or (spec["tier"] if spec else "exec"))
            self._emit(cb, {"type": "tool_start", "name": tname, "args": _redact_obj(obj.get("args") or {}), "tier": tier})
        elif evt_type == "tool_end":
            tname = _strip_mcp_prefix(str(obj.get("name") or ""))
            self._emit(
                cb,
                {
                    "type": "tool_end",
                    "name": tname,
                    "ok": bool(obj.get("ok", True)),
                    "summary": redact_secrets(str(obj.get("summary") or "")),
                    "elapsed_ms": int(obj.get("elapsed_ms") or 0),
                },
            )
        elif evt_type == "assistant":
            self._handle_assistant_event(obj, cb, st)
        elif evt_type == "user":
            self._handle_user_tool_result(obj, cb, st)
        elif evt_type == "result":
            res_txt = redact_secrets(str(obj.get("result") or ""))
            if res_txt and not st.text_parts:
                st.text_parts.append(res_txt)
                self._emit(cb, {"type": "delta", "text": res_txt})
            is_err = bool(obj.get("is_error"))
            st.done_emitted = True
            self._emit(
                cb,
                {
                    "type": "turn_done",
                    "ok": not is_err,
                    "text": "".join(st.text_parts),
                    "reasoning": "".join(st.reasoning_parts),
                    "cancelled": False,
                    "error": res_txt if is_err else "",
                },
            )

    def _spawn_and_monitor_process(
        self,
        cmd: list[str],
        prompt_txt: str,
        use_stdin: bool,
        effort: str,
        token: CancelToken | None,
        cb: Callable[[dict[str, Any]], None] | None,
        st: _ClaudeStreamState,
    ) -> tuple[int, bool]:
        """拉起并监控 `claude` 子进程（含 stdin 喂入、超时看门狗、取消杀树与 JSONL 解析）。"""
        extra_env = {"CLAUDE_CODE_EFFORT_LEVEL": _CLAUDE_EFFORT_ENV_MAP[effort]} if effort in _CLAUDE_EFFORT_ENV_MAP else None
        timed_out_flag = threading.Event()
        watchdog: threading.Timer | None = None
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE if use_stdin else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=clean_subprocess_env(extra_env),
            creationflags=CREATE_NO_WINDOW,
        )
        try:
            if use_stdin and proc.stdin is not None:
                try:
                    proc.stdin.write(prompt_txt)
                    proc.stdin.flush()
                    proc.stdin.close()
                except (BrokenPipeError, OSError):
                    pass
            if token is not None:
                token.register_callback(lambda: kill_process_tree(proc))

            def _on_timeout() -> None:
                timed_out_flag.set()
                kill_process_tree(proc)

            watchdog = threading.Timer(self.timeout_sec, _on_timeout)
            watchdog.daemon = True
            watchdog.start()

            def _drain_stderr() -> None:
                if proc.stderr is None:
                    return
                with suppress_logged("agent_driver", throttled_key="claude_stderr_drain", interval_sec=30.0):
                    for ln in proc.stderr:
                        s = redact_secrets(ln.strip())
                        if s:
                            st.stderr_lines.append(s)

            err_th = threading.Thread(target=_drain_stderr, daemon=True)
            err_th.start()

            if proc.stdout is not None:
                for raw_ln in proc.stdout:
                    if token is not None and token.is_cancelled:
                        break
                    line = raw_ln.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(obj, dict):
                        self._handle_stream_json_event(obj, cb, st)

            rc = int(proc.wait())
            err_th.join(timeout=1.5)
            return rc, timed_out_flag.is_set()
        finally:
            if watchdog is not None:
                watchdog.cancel()

    def run(
        self,
        messages: list[dict[str, Any]] | str,
        *,
        model: str | None = None,
        on_event: Callable[[dict[str, Any]], None] | None = None,
        cancel_token: CancelToken | None = None,
        system_prompt: str | None = None,
        reasoning_effort: str | None = None,
    ) -> dict[str, Any]:
        """执行 `claude -p --output-format stream-json --verbose` 并转为统一事件流。"""
        cb = on_event or self.on_event
        token = cancel_token or self.cancel_token
        mdl = str(model if model is not None else self.model).strip()
        sys_p = str(system_prompt if system_prompt is not None else self.system_prompt).strip()
        eff = _normalize_reasoning_effort(reasoning_effort if reasoning_effort is not None else self.reasoning_effort)
        claude_exe = self.claude_bin or shutil.which("claude") or ""

        if not claude_exe:
            err = "未检测到 claude CLI，请先安装 Claude Code CLI 或切换至 Native 轻量引擎"
            self._emit(cb, {"type": "turn_done", "ok": False, "text": "", "reasoning": "", "cancelled": False, "error": err})
            return {"ok": False, "text": "", "reasoning": "", "cancelled": False, "error": err}
        if token is not None and token.is_cancelled:
            self._emit(cb, {"type": "turn_done", "ok": False, "text": "", "reasoning": "", "cancelled": True, "error": "任务已取消"})
            return {"ok": False, "text": "", "reasoning": "", "cancelled": True, "error": "任务已取消"}

        prompt_txt = self._format_prompt(messages)
        if not prompt_txt:
            err = "用户输入为空"
            self._emit(cb, {"type": "turn_done", "ok": False, "text": "", "reasoning": "", "cancelled": False, "error": err})
            return {"ok": False, "text": "", "reasoning": "", "cancelled": False, "error": err}
        if prompt_txt.startswith("-"):
            prompt_txt = " " + prompt_txt

        st = _ClaudeStreamState()
        with tempfile.TemporaryDirectory(prefix="colab_claude_mcp_") as tmp_dir:
            cmd, use_stdin = self._prepare_mcp_and_cmd(Path(tmp_dir) / "mcp_config.json", claude_exe, mdl, sys_p, prompt_txt)
            try:
                rc, timed_out = self._spawn_and_monitor_process(cmd, prompt_txt, use_stdin, eff, token, cb, st)
            except Exception as exc:
                err_txt = redact_secrets(str(exc) or type(exc).__name__)
                out_err = {"ok": False, "text": "".join(st.text_parts), "reasoning": "".join(st.reasoning_parts), "cancelled": False, "error": err_txt}
                self._emit(cb, {"type": "turn_done", **out_err})
                return out_err

        cancelled = bool(token is not None and token.is_cancelled)
        final_text = "".join(st.text_parts)
        final_reasoning = "".join(st.reasoning_parts)
        if not st.done_emitted:
            if cancelled:
                ok_final, err_str = False, "操作已手动取消"
            elif timed_out:
                ok_final, err_str = False, f"Claude Code 执行超时 ({int(self.timeout_sec)}s)"
            elif rc != 0:
                ok_final = False
                err_str = redact_secrets(st.stderr_lines[-1] if st.stderr_lines else f"claude CLI 退出码 {rc}")
            else:
                ok_final, err_str = True, ""
            out = {"ok": ok_final, "text": final_text, "reasoning": final_reasoning, "cancelled": cancelled, "error": err_str}
            self._emit(cb, {"type": "turn_done", **out})
            return out

        return {
            "ok": rc == 0 and not cancelled and not timed_out,
            "text": final_text,
            "reasoning": final_reasoning,
            "cancelled": cancelled,
            "error": "",
        }


__all__ = [
    "ClaudeCodeDriver",
    "NativeLightDriver",
]
