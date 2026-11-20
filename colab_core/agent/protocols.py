"""colab_core.agent.protocols —— 四协议端点归一化、公共 SSE 传输层与模板/策略模式协议适配器。

支持协议：
- `openai`（OpenAI Chat Completions，含 `reasoning_content` 与 `<think>...</think>` 流式剥离）
- `openai_responses`（OpenAI Responses，含 `reasoning` 摘要与 `function_call` 增量拼接）
- `anthropic`（Anthropic Messages，含 `thinking` 签名多轮回灌与无签名自动降级）
- `gemini`（Gemini generateContent/streamGenerateContent，含 `thinkingLevel` / `thinkingBudget` 与 `thought` 分离）
"""

from __future__ import annotations

import contextlib
import json
import re
import socket
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from colab_core.runtime import CancelToken, normalize_reasoning_effort_value, suppress_logged
from colab_security import redact_secrets, urlopen_safe, url_origin_key

SUPPORTED_API_FORMATS: tuple[str, ...] = ("openai", "openai_responses", "anthropic", "gemini")

_ANTHROPIC_THINKING_BUDGETS: dict[str, int] = {
    "minimal": 1024, "low": 2048, "medium": 8192, "high": 16000, "xhigh": 31999, "max": 31999,
}
_GEMINI_THINKING_BUDGETS: dict[str, int] = {
    "none": 0, "minimal": 1024, "low": 2048, "medium": 8192, "high": 16384, "xhigh": 24576, "max": 24576,
}


def _normalize_reasoning_effort(raw: Any) -> str:
    """规范化思考强度枚举，非法值平滑回退为 auto。"""
    return normalize_reasoning_effort_value(raw, default="auto")


def _gemini_uses_thinking_level(model: str) -> bool:
    """Gemini 3/4 与 Hydrogen/Carbon/Argon 使用 thinkingLevel；2.5 系列沿用 thinkingBudget。"""
    tail = str(model or "").strip().lower().rsplit("/", 1)[-1]
    return any(tok in tail for tok in ("gemini-3", "gemini-4", "hydrogen", "carbon", "argon"))


def _openai_effort_candidates(effort: str) -> list[str]:
    """按 OpenAI / Responses 协议生成思考强度逐级降级重试序列。"""
    eff = _normalize_reasoning_effort(effort)
    if eff == "xhigh":
        return ["xhigh", "high", ""]
    if eff == "max":
        return ["max", "xhigh", "high", ""]
    if eff == "minimal":
        return ["minimal", "low", ""]
    return [eff, ""] if eff in ("none", "low", "medium", "high") else [""]


class _LLMHttpError(RuntimeError):
    """携带原始 HTTP 状态码的 LLM 请求异常，便于自动剥离不兼容思考参数并降级重试。"""

    def __init__(self, code: int, detail: str) -> None:
        super().__init__(f"LLM 接口返回 HTTP {code}: {detail}")
        self.code = int(code)
        self.detail = detail


class _ThinkTagSplitter:
    """增量剥离文本首部内联 `<think>...</think>` 思考块，非思考流零缓冲直通。"""

    _OPEN_TAG = "<think>"
    _CLOSE_TAG = "</think>"

    def __init__(self) -> None:
        self._state = "init"  # init | think | plain
        self._buf = ""

    @staticmethod
    def _strip_leading_newline(text: str) -> str:
        if text.startswith("\r\n"):
            return text[2:]
        return text[1:] if text.startswith("\n") else text

    def _step_init(self, out: list[tuple[str, str]]) -> bool:
        stripped = self._buf.lstrip()
        if not stripped:
            if len(self._buf) > 16:
                self._state = "plain"
                out.append(("delta", self._buf))
                self._buf = ""
            return False
        low = stripped.lower()
        if low.startswith(self._OPEN_TAG):
            idx = self._buf.lower().find(self._OPEN_TAG)
            self._buf = self._strip_leading_newline(self._buf[idx + len(self._OPEN_TAG) :])
            self._state = "think"
            return True
        if len(low) < len(self._OPEN_TAG) and self._OPEN_TAG.startswith(low):
            return False
        self._state = "plain"
        out.append(("delta", self._buf))
        self._buf = ""
        return False

    def _step_think(self, out: list[tuple[str, str]]) -> None:
        end_idx = self._buf.lower().find(self._CLOSE_TAG)
        if end_idx >= 0:
            if end_idx > 0:
                out.append(("reasoning", self._buf[:end_idx]))
            after = self._strip_leading_newline(self._buf[end_idx + len(self._CLOSE_TAG) :])
            self._buf = ""
            self._state = "plain"
            if after:
                out.append(("delta", after))
            return
        hold = len(self._CLOSE_TAG) - 1
        if len(self._buf) > hold:
            out.append(("reasoning", self._buf[:-hold]))
            self._buf = self._buf[-hold:]

    def feed(self, text: str) -> list[tuple[str, str]]:
        """喂入增量文本并返回切分后的 `("reasoning" | "delta", chunk)` 列表。"""
        if not text:
            return []
        if self._state == "plain":
            return [("delta", text)]
        self._buf += text
        out: list[tuple[str, str]] = []
        while self._buf:
            if self._state == "init":
                if not self._step_init(out):
                    break
            elif self._state == "think":
                self._step_think(out)
                break
            else:
                out.append(("delta", self._buf))
                self._buf = ""
        return out

    def flush(self) -> list[tuple[str, str]]:
        """流结束时刷出残留缓冲内容。"""
        if not self._buf:
            return []
        rem = self._buf
        self._buf = ""
        if self._state == "think":
            return [("reasoning", rem)]
        self._state = "plain"
        return [("delta", rem)]


# =========================================================================
# 1. URL 校验与四协议端点归一化
# =========================================================================


def _validate_http_base_url(base_url: str) -> str:
    """校验并规范化外部网关 Base URL，仅允许 HTTP/HTTPS 协议与合法主机名。"""
    u = str(base_url or "").strip().rstrip("/")
    if not u:
        raise ValueError("未配置 API Base URL")
    parsed = urllib.parse.urlparse(u)
    if parsed.scheme.lower() not in ("http", "https") or not parsed.hostname:
        raise ValueError("Base URL 必须以 http:// 或 https:// 开头并包含合法主机名")
    return u


def is_loopback_base_url(base_url: str) -> bool:
    """判断给定 Base URL 是否指向本机回环网关（如 Magpie 127.0.0.1:3425 / localhost）。"""
    u = str(base_url or "").strip()
    if not u:
        return False
    try:
        parsed = urllib.parse.urlparse(u)
    except Exception:
        return False
    return parsed.scheme.lower() in ("http", "https") and (parsed.hostname or "").strip().lower() in ("127.0.0.1", "localhost", "::1")


def _strip_gateway_suffix(base_url: str) -> str:
    """剥离常见接口后缀，提取用于推导 `/v1/...` 或 `/v1beta/...` 的网关根地址。"""
    u = _validate_http_base_url(base_url)
    for suffix in (
        "/v1/chat/completions", "/chat/completions", "/v1/responses", "/responses",
        "/v1/messages", "/messages", "/v1/models", "/v1beta/models", "/models", "/v1beta", "/v1",
    ):
        if u.lower().endswith(suffix):
            return u[: len(u) - len(suffix)].rstrip("/") or u
    return u


def _normalize_openai_endpoint(base_url: str) -> str:
    """规范化 OpenAI Chat Completions 接口地址。"""
    raw = str(base_url or "").strip().rstrip("/")
    if not raw:
        raise ValueError("未配置 OpenAI 兼容 API Base URL")
    u = _validate_http_base_url(raw)
    if u.endswith("/chat/completions"):
        return u
    return f"{u}/chat/completions" if u.endswith("/v1") else f"{u}/v1/chat/completions"


def _normalize_openai_responses_endpoint(base_url: str) -> str:
    """规范化 OpenAI Responses 接口地址。"""
    raw = str(base_url or "").strip().rstrip("/")
    if not raw:
        raise ValueError("未配置 OpenAI Responses API Base URL")
    u = _validate_http_base_url(raw)
    if u.endswith("/responses"):
        return u
    if u.endswith("/chat/completions"):
        return f"{u[:-len('/chat/completions')]}/responses"
    return f"{u}/responses" if u.endswith("/v1") else f"{u}/v1/responses"


def _normalize_anthropic_endpoint(base_url: str) -> str:
    """规范化 Anthropic Messages 接口地址。"""
    raw = str(base_url or "").strip().rstrip("/")
    if not raw:
        raise ValueError("未配置 Anthropic API Base URL")
    u = _validate_http_base_url(raw)
    if u.endswith("/messages"):
        return u
    return f"{u}/messages" if u.endswith("/v1") else f"{u}/v1/messages"


def _normalize_gemini_endpoint(base_url: str, model: str, *, stream: bool = True) -> str:
    """规范化 Gemini API generateContent / streamGenerateContent 接口地址。"""
    raw = str(base_url or "").strip().rstrip("/")
    if not raw:
        raise ValueError("未配置 Gemini 兼容 API Base URL")
    u = _validate_http_base_url(raw)
    clean_model = str(model or "gemini-2.5-flash").strip()
    if clean_model.startswith("models/"):
        clean_model = clean_model[len("models/") :]
    encoded_model = urllib.parse.quote(clean_model or "gemini-2.5-flash", safe="-._")
    root = _strip_gateway_suffix(u)
    action = f"{encoded_model}:streamGenerateContent?alt=sse" if stream else f"{encoded_model}:generateContent"
    if u.lower().endswith("/v1beta/models"):
        return f"{u}/{action}"
    if u.lower().endswith("/v1beta"):
        return f"{u}/models/{action}"
    return f"{root}/v1beta/models/{action}"


def _strip_mcp_prefix(raw_name: str) -> str:
    """剥离 Claude Code MCP 工具名中的 `mcp__<server>__` 前缀。"""
    return re.sub(r"^mcp__[^_]+(?:-[^_]+)*__", "", str(raw_name or "").strip())


# =========================================================================
# 2. 公共 HTTP 流打开、取消打断、思考降级重试与 SSE 行迭代辅助
# =========================================================================


def _emit_event(on_event: Callable[[dict[str, Any]], None] | None, evt: dict[str, Any]) -> None:
    """安全触发流式事件回调，隔离调用方异常并节流记录脱敏诊断日志。"""
    if on_event is not None:
        with suppress_logged("agent_proto", throttled_key="emit_event_err", interval_sec=30.0):
            on_event(evt)


def _close_response_quietly(resp: Any) -> None:
    """安静关闭流式 HTTP 响应对象，记录非预期关闭异常但不向外抛出。"""
    with suppress_logged("agent_proto", throttled_key="close_resp_err", interval_sec=60.0):
        resp.close()


def _register_stream_abort(resp: Any, cancel_token: CancelToken | None) -> None:
    """为响应流绑定即时取消回调，在 Windows 上强关底层 socket 立即打断阻塞读。"""
    if cancel_token is None:
        return

    def _abort_resp() -> None:
        fp = getattr(resp, "fp", None)
        raw = getattr(fp, "raw", None)
        sock = getattr(raw, "_sock", None) or getattr(fp, "_sock", None) or getattr(resp, "_sock", None)
        if sock is not None:
            for fn in (lambda: sock.shutdown(socket.SHUT_RDWR), sock.close):
                # 取消流式读取时底层 socket 可能已关闭或复位，安全忽略清理异常
                with contextlib.suppress(OSError, RuntimeError, ValueError, AttributeError):
                    fn()
        if raw is not None:
            # 底层缓冲流在连接中止后关闭异常可安全忽略
            with contextlib.suppress(OSError, RuntimeError, ValueError, AttributeError):
                raw.close()
        _close_response_quietly(resp)

    cancel_token.register_callback(_abort_resp)


def open_http_stream(
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    *,
    base_url: str,
    timeout_sec: float,
    cancel_token: CancelToken | None,
    secret_tuple: tuple[str, ...],
) -> Any:
    """向同源网关发起 POST 流式请求，并在取消令牌触发时立即中断底层连接。"""
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        resp = urlopen_safe(req, timeout=timeout_sec, expected_origin=url_origin_key(base_url))
    except urllib.error.HTTPError as exc:
        err_body = ""
        # 读取可选 HTTP 错误体失败时回退使用状态原因字符串，不遮蔽主 HTTP 状态码
        with contextlib.suppress(OSError, ValueError):
            err_body = exc.read(512).decode("utf-8", errors="replace").strip()
        raise _LLMHttpError(exc.code, redact_secrets(err_body or exc.reason or str(exc), secret_tuple)) from None
    except Exception as exc:
        raise RuntimeError(f"LLM 网络请求失败: {redact_secrets(str(exc) or type(exc).__name__, secret_tuple)}") from None
    _register_stream_abort(resp, cancel_token)
    return resp


def open_stream_with_effort_fallback(
    url: str,
    headers: dict[str, str],
    base_payload: dict[str, Any],
    effort_candidates: Sequence[str],
    apply_effort: Callable[[dict[str, Any], str], None],
    *,
    base_url: str,
    timeout_sec: float,
    cancel_token: CancelToken | None,
    secret_tuple: tuple[str, ...],
) -> Any:
    """按候选思考强度依次请求；若非推理模型返回 HTTP 400/422 则自动降级或剥离重试。"""
    cands = list(effort_candidates) or [""]
    for idx, cand in enumerate(cands):
        payload = dict(base_payload)
        if cand:
            apply_effort(payload, cand)
        try:
            return open_http_stream(
                url, headers, payload, base_url=base_url, timeout_sec=timeout_sec, cancel_token=cancel_token, secret_tuple=secret_tuple
            )
        except _LLMHttpError as exc:
            if exc.code in (400, 422) and idx + 1 < len(cands):
                continue
            raise


def _iter_sse_lines(resp: Any, cancel_token: CancelToken | None) -> Iterator[tuple[str, str]]:
    """统一迭代 SSE 响应中的 `(event_name, data_str)` 帧，自动处理取消、注释与 `[DONE]`。"""
    cur_event = ""
    while True:
        if cancel_token is not None and cancel_token.is_cancelled:
            break
        try:
            raw_line = resp.readline()
        except Exception:
            if cancel_token is not None and cancel_token.is_cancelled:
                break
            raise
        if not raw_line:
            break
        line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
        if not line:
            cur_event = ""
            continue
        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            cur_event = line[6:].strip()
            continue
        if not line.startswith("data:"):
            continue
        data_str = line[5:].strip()
        if data_str == "[DONE]":
            break
        yield cur_event, data_str


def _extract_error_message(err_obj: Any, default_msg: str, secret_tuple: tuple[str, ...]) -> str:
    msg = err_obj.get("message") if isinstance(err_obj, dict) else str(err_obj)
    return redact_secrets(str(msg or default_msg), secret_tuple)


# =========================================================================
# 3. 模板 + 策略模式四协议适配器
# =========================================================================


@dataclass
class TurnStreamResult:
    """单轮模型流式/非流式响应的结构化产出。"""

    text: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    reasoning: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolExecRecord:
    """单次工具执行完成后的标准化回灌记录。"""

    call_id: str
    name: str
    raw_args: Any
    args_dict: dict[str, Any]
    res_json: str
    ok: bool
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class _StreamTurnContext:
    """单轮流式响应公共累加器与脱敏事件发射上下文。"""

    turn_idx: int
    on_event: Callable[[dict[str, Any]], None] | None
    secret_tuple: tuple[str, ...]
    text_parts: list[str] = field(default_factory=list)
    reasoning_parts: list[str] = field(default_factory=list)

    def emit_text(self, raw_text: str) -> None:
        if not raw_text:
            return
        safe = redact_secrets(raw_text, self.secret_tuple)
        self.text_parts.append(safe)
        _emit_event(self.on_event, {"type": "delta", "text": safe})

    def emit_reasoning(self, raw_text: str) -> None:
        if not raw_text:
            return
        safe = redact_secrets(raw_text, self.secret_tuple)
        self.reasoning_parts.append(safe)
        _emit_event(self.on_event, {"type": "reasoning_delta", "text": safe})


class BaseProtocolAdapter:
    """四协议策略适配器基类：以模板方法统一请求头、HTTP 流打开、JSON/SSE 读取与清理。"""

    format_name: str = "openai"
    _auth_header: str = "Authorization"
    _auth_prefix: str = "Bearer "
    _static_headers: dict[str, str] = {}

    def build_endpoint(self, base_url: str, model: str) -> str:
        raise NotImplementedError

    def build_headers(self, api_key: str, extra_headers: dict[str, str]) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "User-Agent": "ColabPet-Agent/1.0",
            **self._static_headers,
        }
        if api_key:
            headers[self._auth_header] = f"{self._auth_prefix}{api_key}"
        headers.update({str(k): str(v) for k, v in extra_headers.items()})
        return headers

    def build_tools(self, tool_specs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        raise NotImplementedError

    def init_conversation(
        self, raw_messages: Sequence[dict[str, Any]], sys_prompt: str
    ) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
        raise NotImplementedError

    def _prepare_request(
        self,
        sys_text: str,
        conv_msgs: list[dict[str, Any]],
        tools_payload: list[dict[str, Any]],
        model: str,
        reasoning_effort: str,
        state: dict[str, Any],
        endpoint: str,
    ) -> tuple[dict[str, Any], list[str], Callable[[dict[str, Any], str], None]]:
        raise NotImplementedError

    def _new_accumulator(self) -> dict[str, Any]:
        return {}

    def _consume_json(self, raw_obj: Any, ctx: _StreamTurnContext, acc: dict[str, Any]) -> None:
        raise NotImplementedError

    def _consume_sse(self, cur_event: str, chunk: Any, ctx: _StreamTurnContext, acc: dict[str, Any]) -> bool:
        raise NotImplementedError

    def _finalize(self, ctx: _StreamTurnContext, acc: dict[str, Any]) -> TurnStreamResult:
        raise NotImplementedError

    def stream_turn(
        self,
        endpoint: str,
        headers: dict[str, str],
        sys_text: str,
        conv_msgs: list[dict[str, Any]],
        tools_payload: list[dict[str, Any]],
        *,
        model: str,
        base_url: str,
        timeout_sec: float,
        reasoning_effort: str,
        state: dict[str, Any],
        turn_idx: int,
        on_event: Callable[[dict[str, Any]], None] | None,
        cancel_token: CancelToken | None,
        secret_tuple: tuple[str, ...],
    ) -> TurnStreamResult:
        """统一模板流程：构造载荷 -> 思考强度降级打开连接 -> 消费非流式/SSE -> 收束产出。"""
        base_payload, effort_cands, apply_effort = self._prepare_request(
            sys_text, conv_msgs, tools_payload, model, reasoning_effort, state, endpoint
        )
        ctx = _StreamTurnContext(turn_idx=turn_idx, on_event=on_event, secret_tuple=secret_tuple)
        acc = self._new_accumulator()

        resp = open_stream_with_effort_fallback(
            endpoint, headers, base_payload, effort_cands, apply_effort,
            base_url=base_url, timeout_sec=timeout_sec, cancel_token=cancel_token, secret_tuple=secret_tuple,
        )
        try:
            ctype = str(resp.headers.get("Content-Type") or "").lower()
            if "application/json" in ctype and "text/event-stream" not in ctype:
                raw_obj = json.loads(resp.read().decode("utf-8", errors="replace") or "{}")
                self._consume_json(raw_obj, ctx, acc)
            else:
                for cur_event, data_str in _iter_sse_lines(resp, cancel_token):
                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    if self._consume_sse(cur_event, chunk, ctx, acc):
                        break
        finally:
            _close_response_quietly(resp)

        return self._finalize(ctx, acc)

    def append_tool_round(
        self,
        conv_msgs: list[dict[str, Any]],
        turn_res: TurnStreamResult,
        tool_records: Sequence[ToolExecRecord],
        state: dict[str, Any],
        turn_idx: int,
    ) -> None:
        raise NotImplementedError


class OpenAIChatAdapter(BaseProtocolAdapter):
    """OpenAI Chat Completions (`/v1/chat/completions`) 协议适配器。"""

    format_name = "openai"

    def build_endpoint(self, base_url: str, model: str) -> str:
        _ = model
        return _normalize_openai_endpoint(base_url)

    def build_tools(self, tool_specs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {"type": "function", "function": {"name": s["name"], "description": s["description"], "parameters": s["parameters"]}}
            for s in tool_specs
        ]

    @staticmethod
    def build_messages(raw_messages: Sequence[dict[str, Any]], sys_prompt: str) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = [{"role": "system", "content": sys_prompt.strip()}] if sys_prompt.strip() else []
        for msg in raw_messages:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role") or "user").strip().lower()
            role = role if role in ("system", "user", "assistant", "tool") else "user"
            content = msg.get("content")
            if role == "system" and out and out[0]["role"] == "system":
                out[0]["content"] = f"{out[0]['content']}\n\n{content or ''}".strip()
            else:
                copied = dict(msg)
                copied["role"] = role
                if content is None and role != "assistant":
                    copied["content"] = ""
                out.append(copied)
        return out

    def init_conversation(
        self, raw_messages: Sequence[dict[str, Any]], sys_prompt: str
    ) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
        return "", self.build_messages(raw_messages, sys_prompt), {}

    def _prepare_request(self, sys_text, conv_msgs, tools_payload, model, reasoning_effort, state, endpoint):
        _ = (sys_text, state, endpoint)
        payload: dict[str, Any] = {"model": model or "default", "messages": conv_msgs, "stream": True}
        if tools_payload:
            payload["tools"] = tools_payload
        return payload, _openai_effort_candidates(reasoning_effort), lambda pl, eff: pl.__setitem__("reasoning_effort", eff)

    def _new_accumulator(self) -> dict[str, Any]:
        return {"tc_acc": {}, "splitter": _ThinkTagSplitter()}

    @staticmethod
    def _emit_split(pieces: list[tuple[str, str]], ctx: _StreamTurnContext) -> None:
        for kind, chunk in pieces:
            if kind == "reasoning":
                ctx.emit_reasoning(chunk)
            else:
                ctx.emit_text(chunk)

    def _consume_json(self, raw_obj: Any, ctx: _StreamTurnContext, acc: dict[str, Any]) -> None:
        if isinstance(raw_obj, dict) and raw_obj.get("error"):
            raise RuntimeError(_extract_error_message(raw_obj["error"], "LLM 返回错误", ctx.secret_tuple))
        msg_obj = ((raw_obj.get("choices") or [{}])[0].get("message") or {}) if isinstance(raw_obj, dict) else {}
        ctx.emit_reasoning(str(msg_obj.get("reasoning_content") or msg_obj.get("reasoning") or ""))
        txt = str(msg_obj.get("content") or "")
        if txt:
            self._emit_split(acc["splitter"].feed(txt), ctx)
        for idx, tc in enumerate(msg_obj.get("tool_calls") or []):
            fn = tc.get("function") or {}
            raw_args = fn.get("arguments")
            if isinstance(raw_args, dict):
                raw_args = json.dumps(raw_args, ensure_ascii=False)
            acc["tc_acc"][idx] = {"id": str(tc.get("id") or f"call_{idx}"), "name": str(fn.get("name") or ""), "arguments": str(raw_args or "{}")}

    def _consume_sse(self, cur_event: str, chunk: Any, ctx: _StreamTurnContext, acc: dict[str, Any]) -> bool:
        _ = cur_event
        if not isinstance(chunk, dict):
            return False
        if chunk.get("error"):
            raise RuntimeError(_extract_error_message(chunk["error"], "LLM 流式错误", ctx.secret_tuple))
        choices = chunk.get("choices")
        if not isinstance(choices, list) or not choices:
            return False
        delta = choices[0].get("delta") or {}
        r_piece = delta.get("reasoning_content")
        if r_piece is None:
            r_piece = delta.get("reasoning")
        if isinstance(r_piece, str) and r_piece:
            ctx.emit_reasoning(r_piece)
        c_piece = delta.get("content")
        if isinstance(c_piece, str) and c_piece:
            self._emit_split(acc["splitter"].feed(c_piece), ctx)
        tc_acc: dict[int, dict[str, str]] = acc["tc_acc"]
        for tc in delta.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            entry = tc_acc.setdefault(int(tc.get("index", 0) or 0), {"id": "", "name": "", "arguments": ""})
            if tc.get("id"):
                entry["id"] = str(tc["id"])
            fn = tc.get("function") or {}
            if isinstance(fn, dict):
                if fn.get("name"):
                    entry["name"] += str(fn["name"])
                if fn.get("arguments") is not None:
                    entry["arguments"] += str(fn["arguments"])
        return False

    def _finalize(self, ctx: _StreamTurnContext, acc: dict[str, Any]) -> TurnStreamResult:
        self._emit_split(acc["splitter"].flush(), ctx)
        tc_acc = acc["tc_acc"]
        tool_calls = [tc_acc[k] for k in sorted(tc_acc.keys()) if tc_acc[k].get("name")]
        return TurnStreamResult(text="".join(ctx.text_parts), tool_calls=tool_calls, reasoning="".join(ctx.reasoning_parts))

    def append_tool_round(self, conv_msgs, turn_res, tool_records, state, turn_idx) -> None:
        _ = state
        conv_msgs.append(
            {
                "role": "assistant",
                "content": turn_res.text or None,
                "tool_calls": [
                    {
                        "id": tc.get("id") or f"call_{turn_idx}_{i}",
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": tc["arguments"] if isinstance(tc["arguments"], str) else json.dumps(tc["arguments"], ensure_ascii=False),
                        },
                    }
                    for i, tc in enumerate(turn_res.tool_calls)
                ],
            }
        )
        for rec in tool_records:
            conv_msgs.append({"role": "tool", "tool_call_id": rec.call_id, "name": rec.name, "content": rec.res_json})


class OpenAIResponsesAdapter(BaseProtocolAdapter):
    """OpenAI Responses (`/v1/responses`) 协议适配器。"""

    format_name = "openai_responses"

    def build_endpoint(self, base_url: str, model: str) -> str:
        _ = model
        return _normalize_openai_responses_endpoint(base_url)

    def build_tools(self, tool_specs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        return [{"type": "function", "name": s["name"], "description": s["description"], "parameters": s["parameters"]} for s in tool_specs]

    @staticmethod
    def build_input(raw_messages: Sequence[dict[str, Any]], sys_prompt: str) -> tuple[str, list[dict[str, Any]]]:
        sys_parts = [sys_prompt.strip()] if sys_prompt.strip() else []
        items: list[dict[str, Any]] = []
        for msg in raw_messages:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role") or "user").strip().lower()
            content = str(msg.get("content") or "").strip()
            if role == "system":
                if content:
                    sys_parts.append(content)
                continue
            if content:
                items.append({"role": "assistant" if role == "assistant" else "user", "content": content})
        return "\n\n".join(p for p in sys_parts if p), items

    def init_conversation(self, raw_messages, sys_prompt):
        sys_text, items = self.build_input(raw_messages, sys_prompt)
        return sys_text, items, {}

    def _prepare_request(self, sys_text, conv_msgs, tools_payload, model, reasoning_effort, state, endpoint):
        _ = (state, endpoint)
        payload: dict[str, Any] = {"model": model or "default", "input": conv_msgs, "stream": True}
        if sys_text:
            payload["instructions"] = sys_text
        if tools_payload:
            payload["tools"] = tools_payload

        def _apply(pl: dict[str, Any], cand: str) -> None:
            pl["reasoning"] = {"effort": "none"} if cand == "none" else {"effort": cand, "summary": "auto"}

        return payload, _openai_effort_candidates(reasoning_effort), _apply

    def _new_accumulator(self) -> dict[str, Any]:
        return {"calls_by_key": {}, "order_keys": []}

    @staticmethod
    def _norm_arg_str(raw_v: Any) -> str:
        return raw_v if isinstance(raw_v, str) else json.dumps(raw_v or {}, ensure_ascii=False)

    @staticmethod
    def _ensure_slot(acc: dict[str, Any], key: str, default_id: str) -> dict[str, Any]:
        if key not in acc["calls_by_key"]:
            acc["order_keys"].append(key)
            acc["calls_by_key"][key] = {"call_id": default_id, "name": "", "arguments": ""}
        return acc["calls_by_key"][key]

    @staticmethod
    def _find_key(acc: dict[str, Any], evt: dict[str, Any], item: dict[str, Any] | None = None) -> str:
        for cand in ((item or {}).get("id"), evt.get("item_id"), evt.get("output_index")):
            if cand is not None and str(cand) in acc["calls_by_key"]:
                return str(cand)
        first = (item or {}).get("id") or evt.get("item_id") or evt.get("output_index")
        return str(first if first is not None else len(acc["order_keys"]))

    def _consume_json(self, raw_obj: Any, ctx: _StreamTurnContext, acc: dict[str, Any]) -> None:
        if isinstance(raw_obj, dict) and raw_obj.get("error"):
            raise RuntimeError(_extract_error_message(raw_obj["error"], "Responses 接口返回错误", ctx.secret_tuple))
        for idx, item in enumerate((raw_obj or {}).get("output") or []):
            if not isinstance(item, dict):
                continue
            itype = str(item.get("type") or "")
            if itype == "reasoning":
                for sp in item.get("summary") or []:
                    if isinstance(sp, dict) and sp.get("text"):
                        ctx.emit_reasoning(str(sp["text"]))
            elif itype == "message":
                for part in item.get("content") or []:
                    if isinstance(part, dict) and part.get("text"):
                        ctx.emit_text(str(part["text"]))
            elif itype == "function_call":
                cid = str(item.get("call_id") or item.get("id") or f"call_{ctx.turn_idx}_{idx}")
                slot = self._ensure_slot(acc, cid, cid)
                slot["name"] = str(item.get("name") or "")
                slot["arguments"] = self._norm_arg_str(item.get("arguments"))

    def _consume_item_event(self, etype: str, evt: dict[str, Any], ctx: _StreamTurnContext, acc: dict[str, Any]) -> None:
        item = evt.get("item") or {}
        if not isinstance(item, dict):
            return
        if item.get("type") == "function_call":
            k = self._find_key(acc, evt, item)
            cid = str(item.get("call_id") or item.get("id") or f"call_{ctx.turn_idx}_{len(acc['order_keys'])}")
            slot = self._ensure_slot(acc, k, cid)
            if item.get("call_id"):
                slot["call_id"] = str(item["call_id"])
            if item.get("name"):
                slot["name"] = str(item["name"])
            if etype == "response.output_item.done" and item.get("arguments") is not None and not slot["arguments"]:
                slot["arguments"] = self._norm_arg_str(item.get("arguments"))
        elif etype == "response.output_item.done" and item.get("type") == "message" and not ctx.text_parts:
            for part in item.get("content") or []:
                if isinstance(part, dict) and part.get("text"):
                    ctx.emit_text(str(part["text"]))

    def _consume_sse(self, cur_event: str, chunk: Any, ctx: _StreamTurnContext, acc: dict[str, Any]) -> bool:
        if not isinstance(chunk, dict):
            return False
        etype = str(chunk.get("type") or cur_event)
        if etype in ("error", "response.failed"):
            err_obj = chunk.get("error") or (chunk.get("response") or {}).get("error") or {}
            raise RuntimeError(_extract_error_message(err_obj, "Responses 流式错误", ctx.secret_tuple))
        if etype in ("response.reasoning_summary_text.delta", "response.reasoning_text.delta"):
            ctx.emit_reasoning(str(chunk.get("delta") or ""))
        elif etype == "response.output_text.delta":
            ctx.emit_text(str(chunk.get("delta") or ""))
        elif etype in ("response.output_item.added", "response.output_item.done"):
            self._consume_item_event(etype, chunk, ctx, acc)
        elif etype == "response.function_call_arguments.delta":
            slot = self._ensure_slot(acc, self._find_key(acc, chunk), f"call_{ctx.turn_idx}_{len(acc['order_keys'])}")
            slot["arguments"] += str(chunk.get("delta") or "")
        elif etype == "response.function_call_arguments.done":
            slot = self._ensure_slot(acc, self._find_key(acc, chunk), f"call_{ctx.turn_idx}_{len(acc['order_keys'])}")
            if chunk.get("name") and not slot["name"]:
                slot["name"] = str(chunk["name"])
            if chunk.get("arguments") is not None and not slot["arguments"]:
                slot["arguments"] = self._norm_arg_str(chunk.get("arguments"))
        elif etype == "response.completed":
            return True
        return False

    def _finalize(self, ctx: _StreamTurnContext, acc: dict[str, Any]) -> TurnStreamResult:
        tool_calls = [
            {"id": acc["calls_by_key"][k]["call_id"], "name": acc["calls_by_key"][k]["name"], "arguments": acc["calls_by_key"][k]["arguments"] or "{}"}
            for k in acc["order_keys"]
            if acc["calls_by_key"][k].get("name")
        ]
        return TurnStreamResult(text="".join(ctx.text_parts), tool_calls=tool_calls, reasoning="".join(ctx.reasoning_parts))

    def append_tool_round(self, conv_msgs, turn_res, tool_records, state, turn_idx) -> None:
        _ = (turn_res, state, turn_idx)
        for rec in tool_records:
            arg_str = rec.raw_args if isinstance(rec.raw_args, str) else json.dumps(rec.args_dict, ensure_ascii=False)
            conv_msgs.append({"type": "function_call", "call_id": rec.call_id, "name": rec.name, "arguments": arg_str or "{}"})
            conv_msgs.append({"type": "function_call_output", "call_id": rec.call_id, "output": rec.res_json})


class AnthropicAdapter(BaseProtocolAdapter):
    """Anthropic Messages (`/v1/messages`) 协议适配器。"""

    format_name = "anthropic"
    _auth_header = "x-api-key"
    _auth_prefix = ""
    _static_headers = {"anthropic-version": "2023-06-01"}

    def build_endpoint(self, base_url: str, model: str) -> str:
        _ = model
        return _normalize_anthropic_endpoint(base_url)

    def build_tools(self, tool_specs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        return [{"name": s["name"], "description": s["description"], "input_schema": s["parameters"]} for s in tool_specs]

    @staticmethod
    def build_messages(raw_messages: Sequence[dict[str, Any]], sys_prompt: str) -> tuple[str, list[dict[str, Any]]]:
        sys_parts = [sys_prompt.strip()] if sys_prompt.strip() else []
        out: list[dict[str, Any]] = []
        for msg in raw_messages:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role") or "user").strip().lower()
            content = msg.get("content")
            if role == "system":
                if content:
                    sys_parts.append(str(content).strip())
                continue
            norm_role = "assistant" if role == "assistant" else "user"
            norm_content: Any = content if isinstance(content, list) else str(content or "")
            if out and out[-1]["role"] == norm_role and isinstance(out[-1]["content"], str) and isinstance(norm_content, str):
                out[-1]["content"] = f"{out[-1]['content']}\n{norm_content}".strip()
            else:
                out.append({"role": norm_role, "content": norm_content or " "})
        return "\n\n".join(p for p in sys_parts if p), out

    def init_conversation(self, raw_messages, sys_prompt):
        sys_text, msgs = self.build_messages(raw_messages, sys_prompt)
        return sys_text, msgs, {"allow_thinking": True}

    def _prepare_request(self, sys_text, conv_msgs, tools_payload, model, reasoning_effort, state, endpoint):
        _ = endpoint
        eff = _normalize_reasoning_effort(reasoning_effort)
        payload: dict[str, Any] = {"model": model or "claude-sonnet-4-20250514", "max_tokens": 4096, "messages": conv_msgs, "stream": True}
        if sys_text:
            payload["system"] = sys_text
        if tools_payload:
            payload["tools"] = tools_payload
        cands = [eff, ""] if (bool(state.get("allow_thinking", True)) and eff in _ANTHROPIC_THINKING_BUDGETS) else [""]

        def _apply(pl: dict[str, Any], cand: str) -> None:
            budget = _ANTHROPIC_THINKING_BUDGETS.get(cand)
            if budget:
                pl["thinking"] = {"type": "enabled", "budget_tokens": budget}
                pl["max_tokens"] = max(8192, budget + 4096)

        return payload, cands, _apply

    def _new_accumulator(self) -> dict[str, Any]:
        return {"blocks": {}, "thinking_blocks": {}}

    def _consume_json(self, raw_obj: Any, ctx: _StreamTurnContext, acc: dict[str, Any]) -> None:
        if isinstance(raw_obj, dict) and raw_obj.get("type") == "error":
            raise RuntimeError(_extract_error_message(raw_obj.get("error") or {}, "Anthropic 接口错误", ctx.secret_tuple))
        for idx, blk in enumerate((raw_obj or {}).get("content") or []):
            if not isinstance(blk, dict):
                continue
            btype = blk.get("type")
            if btype == "thinking":
                t_txt = str(blk.get("thinking") or "")
                acc["thinking_blocks"][idx] = {"thinking": t_txt, "signature": str(blk.get("signature") or "")}
                ctx.emit_reasoning(t_txt)
            elif btype == "text":
                ctx.emit_text(str(blk.get("text") or ""))
            elif btype == "tool_use":
                acc["blocks"][idx] = {
                    "id": str(blk.get("id") or f"toolu_{ctx.turn_idx}_{idx}"),
                    "name": str(blk.get("name") or ""),
                    "input_json": "",
                    "input_obj": blk.get("input") if isinstance(blk.get("input"), dict) else {},
                }

    def _on_block_start(self, chunk: dict[str, Any], ctx: _StreamTurnContext, acc: dict[str, Any]) -> None:
        idx = int(chunk.get("index", len(acc["blocks"])) or 0)
        cb = chunk.get("content_block") or {}
        cb_type = cb.get("type")
        if cb_type == "thinking":
            init_t = str(cb.get("thinking") or "")
            acc["thinking_blocks"][idx] = {"thinking": init_t, "signature": str(cb.get("signature") or "")}
            ctx.emit_reasoning(init_t)
        elif cb_type == "tool_use":
            init_in = cb.get("input")
            acc["blocks"][idx] = {
                "id": str(cb.get("id") or f"toolu_{ctx.turn_idx}_{idx}"),
                "name": str(cb.get("name") or ""),
                "input_json": "",
                "input_obj": init_in if isinstance(init_in, dict) and init_in else None,
            }
        elif cb_type == "text":
            ctx.emit_text(str(cb.get("text") or ""))

    def _on_block_delta(self, chunk: dict[str, Any], ctx: _StreamTurnContext, acc: dict[str, Any]) -> None:
        idx = int(chunk.get("index", 0) or 0)
        delta = chunk.get("delta") or {}
        dtype = str(delta.get("type") or "")
        if dtype == "thinking_delta":
            t_piece = str(delta.get("thinking") or "")
            if t_piece:
                acc["thinking_blocks"].setdefault(idx, {"thinking": "", "signature": ""})["thinking"] += t_piece
                ctx.emit_reasoning(t_piece)
        elif dtype == "signature_delta" and delta.get("signature"):
            acc["thinking_blocks"].setdefault(idx, {"thinking": "", "signature": ""})["signature"] += str(delta["signature"])
        elif dtype == "text_delta":
            ctx.emit_text(str(delta.get("text") or ""))
        elif dtype == "input_json_delta" and idx in acc["blocks"] and delta.get("partial_json"):
            acc["blocks"][idx]["input_json"] += str(delta["partial_json"])

    def _consume_sse(self, cur_event: str, chunk: Any, ctx: _StreamTurnContext, acc: dict[str, Any]) -> bool:
        if not isinstance(chunk, dict):
            return False
        etype = str(chunk.get("type") or cur_event)
        if etype == "error":
            raise RuntimeError(_extract_error_message(chunk.get("error") or {}, "Anthropic 流式错误", ctx.secret_tuple))
        if etype == "content_block_start":
            self._on_block_start(chunk, ctx, acc)
        elif etype == "content_block_delta":
            self._on_block_delta(chunk, ctx, acc)
        elif etype == "message_stop":
            return True
        return False

    def _finalize(self, ctx: _StreamTurnContext, acc: dict[str, Any]) -> TurnStreamResult:
        tool_calls = [
            {"id": blk["id"], "name": blk["name"], "arguments": str(blk.get("input_json") or "").strip() or blk.get("input_obj") or {}}
            for _, blk in sorted(acc["blocks"].items())
            if blk.get("name")
        ]
        signed_thinking = [
            {"type": "thinking", "thinking": tb["thinking"], "signature": tb["signature"]}
            for _, tb in sorted(acc["thinking_blocks"].items())
            if tb.get("thinking") and tb.get("signature")
        ]
        return TurnStreamResult(
            text="".join(ctx.text_parts), tool_calls=tool_calls, reasoning="".join(ctx.reasoning_parts), extra={"signed_thinking": signed_thinking}
        )

    def append_tool_round(self, conv_msgs, turn_res, tool_records, state, turn_idx) -> None:
        _ = turn_idx
        assistant_blocks: list[dict[str, Any]] = []
        signed = turn_res.extra.get("signed_thinking") or []
        if signed:
            assistant_blocks.extend(signed)
        elif turn_res.reasoning:
            # 网关未透传 thinking signature 时，后续工具回灌轮次自动停用 thinking 防校验报错
            state["allow_thinking"] = False
        if turn_res.text:
            assistant_blocks.append({"type": "text", "text": turn_res.text})
        tool_result_blocks: list[dict[str, Any]] = []
        for rec in tool_records:
            assistant_blocks.append({"type": "tool_use", "id": rec.call_id, "name": rec.name, "input": rec.args_dict})
            tool_result_blocks.append({"type": "tool_result", "tool_use_id": rec.call_id, "content": rec.res_json, "is_error": not rec.ok})
        conv_msgs.append({"role": "assistant", "content": assistant_blocks})
        conv_msgs.append({"role": "user", "content": tool_result_blocks})


class GeminiAdapter(BaseProtocolAdapter):
    """Gemini API (`/v1beta/models/...:streamGenerateContent`) 协议适配器。"""

    format_name = "gemini"
    _auth_header = "x-goog-api-key"
    _auth_prefix = ""

    def build_endpoint(self, base_url: str, model: str) -> str:
        return _normalize_gemini_endpoint(base_url, model, stream=True)

    def build_tools(self, tool_specs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        return [{"name": s["name"], "description": s["description"], "parameters": s["parameters"]} for s in tool_specs]

    @staticmethod
    def build_contents(raw_messages: Sequence[dict[str, Any]], sys_prompt: str) -> tuple[str, list[dict[str, Any]]]:
        sys_parts = [sys_prompt.strip()] if sys_prompt.strip() else []
        out: list[dict[str, Any]] = []
        for msg in raw_messages:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role") or "user").strip().lower()
            content = str(msg.get("content") or "").strip()
            if role == "system":
                if content:
                    sys_parts.append(content)
                continue
            norm_role = "model" if role == "assistant" else "user"
            text_val = content or " "
            if out and out[-1]["role"] == norm_role:
                out[-1]["parts"].append({"text": text_val})
            else:
                out.append({"role": norm_role, "parts": [{"text": text_val}]})
        return "\n\n".join(p for p in sys_parts if p), out

    def init_conversation(self, raw_messages, sys_prompt):
        sys_text, contents = self.build_contents(raw_messages, sys_prompt)
        return sys_text, contents, {}

    @staticmethod
    def _effort_candidates(eff: str, use_level: bool) -> list[str]:
        if not use_level:
            return [eff, ""] if eff in _GEMINI_THINKING_BUDGETS else [""]
        if eff in ("xhigh", "max"):
            return ["xhigh", "high", ""]
        if eff in ("low", "medium", "high"):
            return [eff, ""]
        return ["minimal", ""] if eff in ("minimal", "none") else [""]

    def _prepare_request(self, sys_text, conv_msgs, tools_payload, model, reasoning_effort, state, endpoint):
        _ = state
        eff = _normalize_reasoning_effort(reasoning_effort)
        use_model = str(model or "").strip()
        if not use_model:
            m_url = re.search(r"/models/([^/:?#]+)", str(endpoint or ""))
            use_model = m_url.group(1) if m_url else ""
        use_level = _gemini_uses_thinking_level(use_model)
        payload: dict[str, Any] = {"contents": conv_msgs}
        if sys_text:
            payload["systemInstruction"] = {"parts": [{"text": sys_text}]}
        if tools_payload:
            payload["tools"] = [{"functionDeclarations": tools_payload}]

        def _apply(pl: dict[str, Any], cand: str) -> None:
            if use_level:
                if cand:
                    pl["generationConfig"] = {"thinkingConfig": {"thinkingLevel": cand, "includeThoughts": True}}
                return
            budget = _GEMINI_THINKING_BUDGETS.get(cand)
            if budget is not None:
                tc_cfg: dict[str, Any] = {"thinkingBudget": budget}
                if budget > 0:
                    tc_cfg["includeThoughts"] = True
                pl["generationConfig"] = {"thinkingConfig": tc_cfg}

        return payload, self._effort_candidates(eff, use_level), _apply

    def _new_accumulator(self) -> dict[str, Any]:
        return {"tool_calls": []}

    def _consume_chunk(self, obj: Any, ctx: _StreamTurnContext, acc: dict[str, Any]) -> None:
        if isinstance(obj, list):
            for sub in obj:
                self._consume_chunk(sub, ctx, acc)
            return
        if not isinstance(obj, dict):
            return
        if obj.get("error"):
            raise RuntimeError(_extract_error_message(obj["error"], "Gemini 接口返回错误", ctx.secret_tuple))
        for cand in obj.get("candidates") or []:
            if not isinstance(cand, dict):
                continue
            for part in (cand.get("content") or {}).get("parts") or []:
                if not isinstance(part, dict):
                    continue
                if isinstance(part.get("text"), str) and part["text"]:
                    (ctx.emit_reasoning if bool(part.get("thought")) else ctx.emit_text)(part["text"])
                fc = part.get("functionCall")
                if isinstance(fc, dict) and fc.get("name"):
                    tc_entry: dict[str, Any] = {
                        "id": f"call_gem_{ctx.turn_idx}_{len(acc['tool_calls'])}",
                        "name": str(fc["name"]),
                        "arguments": fc.get("args") if isinstance(fc.get("args"), dict) else {},
                    }
                    if part.get("thoughtSignature"):
                        tc_entry["thought_signature"] = str(part["thoughtSignature"])
                    acc["tool_calls"].append(tc_entry)

    def _consume_json(self, raw_obj: Any, ctx: _StreamTurnContext, acc: dict[str, Any]) -> None:
        self._consume_chunk(raw_obj, ctx, acc)

    def _consume_sse(self, cur_event: str, chunk: Any, ctx: _StreamTurnContext, acc: dict[str, Any]) -> bool:
        _ = cur_event
        self._consume_chunk(chunk, ctx, acc)
        return False

    def _finalize(self, ctx: _StreamTurnContext, acc: dict[str, Any]) -> TurnStreamResult:
        return TurnStreamResult(text="".join(ctx.text_parts), tool_calls=list(acc["tool_calls"]), reasoning="".join(ctx.reasoning_parts))

    def append_tool_round(self, conv_msgs, turn_res, tool_records, state, turn_idx) -> None:
        _ = (state, turn_idx)
        model_parts: list[dict[str, Any]] = [{"text": turn_res.text}] if turn_res.text else []
        resp_parts: list[dict[str, Any]] = []
        for rec in tool_records:
            fc_part: dict[str, Any] = {"functionCall": {"name": rec.name, "args": rec.args_dict}}
            if rec.meta.get("thought_signature"):
                fc_part["thoughtSignature"] = str(rec.meta["thought_signature"])
            model_parts.append(fc_part)
            resp_parts.append({"functionResponse": {"name": rec.name, "response": {"result": rec.res_json}}})
        conv_msgs.append({"role": "model", "parts": model_parts})
        conv_msgs.append({"role": "user", "parts": resp_parts})


PROTOCOL_ADAPTERS: dict[str, BaseProtocolAdapter] = {
    "openai": OpenAIChatAdapter(),
    "openai_responses": OpenAIResponsesAdapter(),
    "anthropic": AnthropicAdapter(),
    "gemini": GeminiAdapter(),
}


def get_protocol_adapter(api_format: str) -> BaseProtocolAdapter:
    """按协议标识返回对应策略适配器，未知格式平滑回退为 OpenAI Chat。"""
    return PROTOCOL_ADAPTERS.get(str(api_format or "openai").strip().lower(), PROTOCOL_ADAPTERS["openai"])


__all__ = [
    "AnthropicAdapter", "BaseProtocolAdapter", "GeminiAdapter", "OpenAIChatAdapter", "OpenAIResponsesAdapter",
    "PROTOCOL_ADAPTERS", "SUPPORTED_API_FORMATS", "ToolExecRecord", "TurnStreamResult",
    "_ANTHROPIC_THINKING_BUDGETS", "_GEMINI_THINKING_BUDGETS", "_LLMHttpError", "_ThinkTagSplitter",
    "_emit_event", "_gemini_uses_thinking_level", "_iter_sse_lines", "_normalize_anthropic_endpoint",
    "_normalize_gemini_endpoint", "_normalize_openai_endpoint", "_normalize_openai_responses_endpoint",
    "_normalize_reasoning_effort", "_openai_effort_candidates", "_strip_gateway_suffix", "_strip_mcp_prefix",
    "_validate_http_base_url", "get_protocol_adapter", "is_loopback_base_url",
    "open_http_stream", "open_stream_with_effort_fallback",
]
