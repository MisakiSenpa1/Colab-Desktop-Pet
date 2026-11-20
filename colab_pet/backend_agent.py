"""Colab Studio 同进程轻量 AI 副驾服务模块。

负责维护副驾多轮对话历史、执行大脑自动路由（在线 Colab Qwen3.8 vs 外部 OpenAI/Anthropic/Gemini API）、
编排 NativeLightDriver 与 ClaudeCodeDriver 双引擎、实现 10 个核心工具在同进程 ColabPetBackend
上的本地+远端执行器，并支持高风险/计费操作的线程安全挂起审批与全链路凭据脱敏。
"""

from __future__ import annotations

import copy
import json
import os
import re
import subprocess
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from colab_core.agent_tools import (
    AGENT_TOOL_SPECS,
    SUPPORTED_API_FORMATS,
    ClaudeCodeDriver,
    NativeLightDriver,
    check_remote_code_safety,
    detect_provider_protocol,
    fetch_provider_models,
    is_loopback_base_url,
    is_readonly_safe_remote_cmd,
    probe_provider_models,
    url_origin_key,
)
from colab_core.file_sync import _read_png_text_chunks, is_api_prompt_dict
from colab_core.runtime import (
    CREATE_NO_WINDOW,
    DEFAULT_PREFS,
    CancelToken,
    clean_subprocess_env,
    infer_model_reasoning_spec,
    kill_process_tree,
    normalize_reasoning_effort_value,
    normalize_reasoning_level_list,
    sanitize_agent_providers,
    sanitize_model_capabilities,
    suppress_logged,
)
from colab_security import (
    AGENT_VAULT_API_KEY,
    VaultStore,
    agent_provider_vault_key,
    load_cf_tunnel_settings,
    redact_secrets,
    register_ephemeral_secret,
)

if TYPE_CHECKING:
    from colab_backend import ColabPetBackend

MAX_AGENT_MESSAGES = 40
MAX_AGENT_SESSIONS = 30
CONFIRM_TIMEOUT_SEC = 180.0
_MAX_EXEC_OUTPUT_CHARS = 8000
_MAX_PROMPT_DEPTH = 10
_CJK_CHAR_RE = re.compile(r"[\u3000-\u303f\u4e00-\uffef]")

_PREF_KEY_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("agent_driver", ("agent_driver", "driver")),
    ("agent_brain_source", ("agent_brain_source", "brain_source")),
    ("agent_api_format", ("agent_api_format", "api_format")),
    ("agent_api_base_url", ("agent_api_base_url", "api_base_url", "base_url")),
    ("agent_model", ("agent_model", "model")),
    ("agent_reasoning_effort", ("agent_reasoning_effort", "reasoning_effort")),
    ("agent_auto_approve_billing", ("agent_auto_approve_billing", "auto_approve_billing")),
)

_REASONING_EFFORT_LABELS: dict[str, str] = {
    "auto": "自动",
    "none": "关闭思考",
    "minimal": "极简",
    "low": "低",
    "medium": "中",
    "high": "高",
    "xhigh": "超高 (Extra)",
    "max": "极高 (Max)",
}


def _origin_vault_key(vkey: str) -> str:
    """生成指定凭据保险箱键名对应的源站硬绑定伴随键名。"""
    return f"{vkey}__origin"


def _parse_str_list(raw_val: Any) -> list[str]:
    """将逗号分隔字符串或序列统一解析为去空白后的非空字符串列表。"""
    if isinstance(raw_val, str):
        return [x.strip() for x in raw_val.split(",") if x.strip()]
    if isinstance(raw_val, (list, tuple)):
        return [str(x or "").strip() for x in raw_val if str(x or "").strip()]
    return []


def _collect_conditioning_node_ids(
    graph: dict[str, dict[str, Any]],
    link_val: Any,
    visited: set[str] | None = None,
    depth: int = 0,
) -> list[str]:
    """沿 ComfyUI API prompt 图中的 Conditioning 链接逆向收集文本编码节点 ID。"""
    if depth > _MAX_PROMPT_DEPTH or not isinstance(link_val, (list, tuple)) or not link_val:
        return []
    node_id = str(link_val[0])
    seen = visited if visited is not None else set()
    if node_id in seen:
        return []
    seen.add(node_id)

    node = graph.get(node_id)
    if not isinstance(node, dict):
        return []
    inputs = node.get("inputs")
    if not isinstance(inputs, dict):
        return []

    found: list[str] = []
    for txt_key in ("text", "prompt", "t5xxl", "clip_l", "positive", "string"):
        if txt_key in inputs:
            raw_v = inputs[txt_key]
            if isinstance(raw_v, str):
                found.append(node_id)
                break
            if isinstance(raw_v, (list, tuple)) and raw_v:
                up_id = str(raw_v[0])
                if up_id in graph and up_id not in found:
                    found.append(up_id)
                    break

    for next_key in (
        "conditioning",
        "conditioning_1",
        "conditioning_2",
        "conditioning_to",
        "conditioning_from",
        "positive",
        "negative",
    ):
        next_val = inputs.get(next_key)
        if isinstance(next_val, (list, tuple)):
            found.extend(_collect_conditioning_node_ids(graph, next_val, seen, depth + 1))
    return list(dict.fromkeys(found))


def _set_node_prompt_text(graph: dict[str, dict[str, Any]], node_id: str, new_text: str) -> bool:
    """将指定 ComfyUI 节点中的提示词字符串字段替换为新文本。"""
    node = graph.get(node_id)
    if not isinstance(node, dict):
        return False
    inputs = node.get("inputs")
    if not isinstance(inputs, dict):
        return False
    for key in ("text", "prompt", "t5xxl", "clip_l", "positive", "string", "value"):
        if key in inputs and isinstance(inputs[key], str):
            inputs[key] = new_text
            return True
    return False


def _collect_guider_conditioning_nodes(
    graph: dict[str, dict[str, Any]],
    guider_ref: Any,
    pos_nodes: list[str],
    neg_nodes: list[str],
) -> None:
    """沿采样器的 guider 上游引用提取正反向文本编码节点 ID。"""
    if not isinstance(guider_ref, (list, tuple)) or not guider_ref:
        return
    g_node = graph.get(str(guider_ref[0]))
    if not isinstance(g_node, dict) or not isinstance(g_node.get("inputs"), dict):
        return
    g_in = g_node["inputs"]
    if "positive" in g_in:
        pos_nodes.extend(_collect_conditioning_node_ids(graph, g_in.get("positive")))
    elif "conditioning" in g_in:
        pos_nodes.extend(_collect_conditioning_node_ids(graph, g_in.get("conditioning")))
    if "negative" in g_in:
        neg_nodes.extend(_collect_conditioning_node_ids(graph, g_in.get("negative")))


def _find_sampler_and_prompt_nodes(
    graph: dict[str, dict[str, Any]],
) -> tuple[list[str], list[str], list[str]]:
    """扫描 ComfyUI API prompt 图，定位采样器及关联的正反向提示词节点。"""
    pos_nodes: list[str] = []
    neg_nodes: list[str] = []
    sampler_ids: list[str] = []
    for nid, node in graph.items():
        cls_low = str(node.get("class_type") or "").lower()
        inputs = node["inputs"]
        is_sampler = ("sampler" in cls_low) or (
            "positive" in inputs
            and any(k in inputs for k in ("steps", "seed", "noise_seed", "guider", "sigmas"))
        )
        if not is_sampler:
            continue
        sampler_ids.append(nid)
        if "positive" in inputs:
            pos_nodes.extend(_collect_conditioning_node_ids(graph, inputs.get("positive")))
        if "negative" in inputs:
            neg_nodes.extend(_collect_conditioning_node_ids(graph, inputs.get("negative")))
        _collect_guider_conditioning_nodes(graph, inputs.get("guider"), pos_nodes, neg_nodes)
    return sampler_ids, pos_nodes, neg_nodes


def _fill_fallback_text_nodes(
    graph: dict[str, dict[str, Any]],
    pos_nodes: list[str],
    neg_nodes: list[str],
) -> None:
    """当采样器连线未定位到正反向编码节点时，按节点类名与标题兜底匹配。"""
    fallback_nodes: list[str] = []
    for nid, node in graph.items():
        cls_low = str(node.get("class_type") or "").lower()
        if not any(h in cls_low for h in ("cliptextencode", "textencode", "prompt")):
            continue
        title = str((node.get("_meta") or {}).get("title") or "").lower()
        if any(w in title for w in ("neg", "负面", "反向")):
            if nid not in neg_nodes:
                neg_nodes.append(nid)
        elif any(w in title for w in ("pos", "正面", "正向")):
            if nid not in pos_nodes:
                pos_nodes.append(nid)
        else:
            fallback_nodes.append(nid)
    if not pos_nodes and fallback_nodes:
        pos_nodes.append(fallback_nodes[0])
    if not neg_nodes and len(fallback_nodes) > 1:
        neg_nodes.append(fallback_nodes[1])


def _apply_linked_scalar(
    graph: dict[str, dict[str, Any]],
    ref_val: Any,
    field_name: str,
    new_val: int,
) -> bool:
    """沿节点引用覆写上游标量参数（跳过已连线列表值）。"""
    if not isinstance(ref_val, (list, tuple)) or not ref_val:
        return False
    up_node = graph.get(str(ref_val[0]))
    if not isinstance(up_node, dict) or not isinstance(up_node.get("inputs"), dict):
        return False
    up_in = up_node["inputs"]
    if field_name in up_in and not isinstance(up_in[field_name], list):
        up_in[field_name] = new_val
        return True
    return False


def _apply_seed_to_samplers(
    graph: dict[str, dict[str, Any]],
    sampler_ids: list[str],
    seed_val: int,
    applied: dict[str, Any],
) -> None:
    """将随机种子写回所有采样器及其上游噪声节点。"""
    for sid in sampler_ids:
        s_in = graph[sid]["inputs"]
        for key in ("seed", "noise_seed"):
            if key in s_in and not isinstance(s_in.get(key), list):
                s_in[key] = seed_val
                applied["seed"] = seed_val
        if _apply_linked_scalar(graph, s_in.get("noise"), "noise_seed", seed_val):
            applied["seed"] = seed_val


def _apply_steps_to_samplers(
    graph: dict[str, dict[str, Any]],
    sampler_ids: list[str],
    steps_val: int,
    applied: dict[str, Any],
) -> None:
    """将采样步数写回所有采样器及其上游调度器（sigmas）节点。"""
    for sid in sampler_ids:
        s_in = graph[sid]["inputs"]
        if "steps" in s_in and not isinstance(s_in["steps"], list):
            s_in["steps"] = steps_val
            applied["steps"] = steps_val
        if _apply_linked_scalar(graph, s_in.get("sigmas"), "steps", steps_val):
            applied["steps"] = steps_val


def _apply_batch_size_to_latents(
    graph: dict[str, dict[str, Any]],
    bs_val: int,
    applied: dict[str, Any],
) -> None:
    """将出图张数写回潜空间（Latent）节点的 batch_size 字段。"""
    for node in graph.values():
        cls_low = str(node.get("class_type") or "").lower()
        inputs = node["inputs"]
        if ("latent" in cls_low or "batch_size" in inputs) and "batch_size" in inputs:
            if not isinstance(inputs["batch_size"], list):
                inputs["batch_size"] = bs_val
                applied["batch_size"] = bs_val


def _patch_comfy_api_graph(
    prompt_graph: dict[str, Any],
    *,
    positive_prompt: str = "",
    negative_prompt: str = "",
    seed: int | None = None,
    steps: int | None = None,
    batch_size: int | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """深拷贝并覆写 ComfyUI API prompt 图的正反向提示词、随机种子、步数与批次大小。"""
    graph: dict[str, dict[str, Any]] = {
        str(k): copy.deepcopy(v)
        for k, v in prompt_graph.items()
        if isinstance(v, dict) and isinstance(v.get("inputs"), dict)
    }
    pos_text = str(positive_prompt or "").strip()
    neg_text = str(negative_prompt or "").strip()

    sampler_ids, pos_nodes, neg_nodes = _find_sampler_and_prompt_nodes(graph)
    if (pos_text and not pos_nodes) or (neg_text and not neg_nodes):
        _fill_fallback_text_nodes(graph, pos_nodes, neg_nodes)

    applied: dict[str, Any] = {}
    if pos_text and any(_set_node_prompt_text(graph, pid, pos_text) for pid in dict.fromkeys(pos_nodes)):
        applied["positive_prompt"] = pos_text
    if neg_text and any(_set_node_prompt_text(graph, nid, neg_text) for nid in dict.fromkeys(neg_nodes)):
        applied["negative_prompt"] = neg_text
    if seed is not None:
        _apply_seed_to_samplers(graph, sampler_ids, int(seed), applied)
    if steps is not None:
        _apply_steps_to_samplers(graph, sampler_ids, max(1, min(150, int(steps))), applied)
    if batch_size is not None:
        _apply_batch_size_to_latents(graph, max(1, min(8, int(batch_size))), applied)
    return graph, applied


class AgentService:
    """Colab Studio 同进程轻量 AI 副驾编排服务（多会话历史持久化、双引擎调度、审批挂起与全量脱敏）。"""

    def __init__(self, backend: ColabPetBackend) -> None:
        self._b = backend
        self._store = backend._store
        self._secrets = backend._secrets
        self._bus = backend._bus
        self._ctx = backend._ctx
        self._lock = threading.RLock()
        self._busy_lock = threading.Lock()
        state_p = Path(str(self._store.state_path))
        sess_fname = "agent_sessions.json" if state_p.stem == "pet_state" else f"{state_p.stem}_agent_sessions.json"
        self._sessions_path: Path = state_p.with_name(sess_fname)
        self._sessions: list[dict[str, Any]] = []
        self._active_session_id: str = ""
        self._messages: list[dict[str, Any]] = []
        self._running: bool = False
        self._active_turn_id: str = ""
        self._cancel_token: CancelToken | None = None
        self._turn_thread: threading.Thread | None = None
        self._pending_confirm: dict[str, Any] | None = None
        self._confirm_event: threading.Event | None = None
        self._confirm_result: bool | None = None
        self._load_sessions_from_disk()

    def _get_merged_prefs(self) -> dict[str, Any]:
        """从状态容器安全读取合并默认键后的用户偏好字典副本。"""
        prefs = dict(DEFAULT_PREFS)
        prefs.update(self._store.get_prefs())
        return prefs

    # ------------------------------------------------------------------
    # 多会话历史轻量持久化与上下文统计
    # ------------------------------------------------------------------

    @staticmethod
    def _new_empty_session(title: str = "新对话", now_ts: int | None = None) -> dict[str, Any]:
        ts = int(now_ts if now_ts is not None else time.time())
        return {
            "id": f"sess_{uuid.uuid4().hex[:8]}",
            "title": str(title or "新对话")[:40] or "新对话",
            "created_ts": ts,
            "updated_ts": ts,
            "messages": [],
        }

    @staticmethod
    def _derive_session_title(text: str) -> str:
        cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
        if not cleaned:
            return "新对话"
        return f"{cleaned[:24]}..." if len(cleaned) > 24 else cleaned

    @staticmethod
    def _compute_context_stats(messages: list[dict[str, Any]]) -> dict[str, int]:
        valid = [
            m
            for m in messages
            if isinstance(m, dict) and m.get("role") in ("user", "assistant") and str(m.get("content") or "").strip()
        ]
        approx_chars = 0
        est_tokens = 0.0
        for m in valid:
            txt = str(m.get("content") or "")
            approx_chars += len(txt)
            cjk_cnt = len(_CJK_CHAR_RE.findall(txt))
            other_cnt = max(0, len(txt) - cjk_cnt)
            est_tokens += cjk_cnt * 0.7 + other_cnt * 0.3
        return {
            "message_count": len(valid),
            "max_messages": MAX_AGENT_MESSAGES,
            "approx_chars": approx_chars,
            "approx_tokens": int(round(est_tokens)),
        }

    def _trim_sessions_locked(self) -> None:
        if len(self._sessions) <= MAX_AGENT_SESSIONS:
            return
        active_id = self._active_session_id
        active_list = [s for s in self._sessions if s["id"] == active_id]
        others = [s for s in self._sessions if s["id"] != active_id]
        others.sort(key=lambda s: (-int(s.get("updated_ts") or 0), -int(s.get("created_ts") or 0)))
        kept_others = others[: max(0, MAX_AGENT_SESSIONS - len(active_list))]
        self._sessions = active_list + kept_others

    def _get_active_session_locked(self) -> dict[str, Any]:
        for s in self._sessions:
            if s.get("id") == self._active_session_id:
                if not isinstance(s.get("messages"), list):
                    s["messages"] = []
                self._messages = s["messages"]
                return s
        if self._sessions:
            picked = self._sessions[0]
            if not isinstance(picked.get("messages"), list):
                picked["messages"] = []
            self._active_session_id = str(picked["id"])
            self._messages = picked["messages"]
            return picked
        fresh = self._new_empty_session()
        self._sessions = [fresh]
        self._active_session_id = fresh["id"]
        self._messages = fresh["messages"]
        return fresh

    def _sanitize_loaded_session(
        self,
        idx: int,
        item: Any,
        seen_ids: set[str],
        now_ts: int,
    ) -> dict[str, Any] | None:
        """清洗单条磁盘会话条目并脱敏历史消息，无效项返回 None。"""
        if not isinstance(item, dict):
            return None
        sid = re.sub(r"[^a-zA-Z0-9_-]", "_", str(item.get("id") or f"sess_{idx + 1}").strip())[:32]
        if not sid or sid in seen_ids:
            return None
        seen_ids.add(sid)
        msgs_raw = item.get("messages")
        clean_msgs = [
            self._scrub_obj(dict(m))
            for m in (msgs_raw[-MAX_AGENT_MESSAGES:] if isinstance(msgs_raw, list) else [])
            if isinstance(m, dict) and m.get("role") in ("user", "assistant")
        ]
        c_ts = int(item.get("created_ts") or now_ts)
        u_ts = int(item.get("updated_ts") or c_ts)
        title = str(item.get("title") or "").strip()[:40] or "新对话"
        return {
            "id": sid,
            "title": title,
            "created_ts": c_ts,
            "updated_ts": u_ts,
            "messages": clean_msgs,
        }

    def _load_sessions_from_disk(self) -> None:
        raw_obj: Any = None
        try:
            if self._sessions_path.is_file():
                raw_txt = self._sessions_path.read_text(encoding="utf-8", errors="replace")
                raw_obj = json.loads(raw_txt or "{}")
        except Exception:
            raw_obj = None

        loaded: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        now_ts = int(time.time())
        if isinstance(raw_obj, dict) and isinstance(raw_obj.get("sessions"), list):
            for idx, item in enumerate(raw_obj["sessions"]):
                entry = self._sanitize_loaded_session(idx, item, seen_ids, now_ts)
                if entry is not None:
                    loaded.append(entry)

        with self._lock:
            if loaded:
                self._sessions = loaded
                req_active = str((raw_obj or {}).get("active_session_id") or "").strip() if isinstance(raw_obj, dict) else ""
                if any(s["id"] == req_active for s in self._sessions):
                    self._active_session_id = req_active
                else:
                    best = max(self._sessions, key=lambda s: int(s.get("updated_ts") or 0))
                    self._active_session_id = str(best["id"])
                self._trim_sessions_locked()
                self._get_active_session_locked()
            else:
                fresh = self._new_empty_session(now_ts=now_ts)
                self._sessions = [fresh]
                self._active_session_id = fresh["id"]
                self._messages = fresh["messages"]

    def _save_sessions_to_disk(self, secrets_tuple: tuple[str, ...] = ()) -> None:
        with self._lock:
            self._trim_sessions_locked()
            payload = self._scrub_obj(
                {
                    "active_session_id": self._active_session_id,
                    "sessions": copy.deepcopy(self._sessions),
                },
                secrets_tuple,
            )
        try:
            self._sessions_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_p = self._sessions_path.with_suffix(".tmp")
            tmp_p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(tmp_p, self._sessions_path)
        except OSError:
            pass

    def _build_sessions_summary(self) -> list[dict[str, Any]]:
        with self._lock:
            active_id = self._active_session_id
            items = list(self._sessions)
            items.sort(
                key=lambda s: (
                    0 if (s["id"] == active_id and len(s.get("messages") or []) == 0) else 1,
                    -int(s.get("updated_ts") or 0),
                    -int(s.get("created_ts") or 0),
                )
            )
            out: list[dict[str, Any]] = []
            for s in items:
                msgs = s.get("messages") or []
                preview = ""
                for m in reversed(msgs):
                    if isinstance(m, dict):
                        txt = re.sub(r"\s+", " ", str(m.get("content") or "")).strip()
                        if txt:
                            preview = f"{txt[:44]}..." if len(txt) > 44 else txt
                            break
                out.append(
                    {
                        "id": str(s["id"]),
                        "title": str(s.get("title") or "新对话"),
                        "created_ts": int(s.get("created_ts") or 0),
                        "updated_ts": int(s.get("updated_ts") or 0),
                        "message_count": len(msgs),
                        "preview": preview,
                    }
                )
            return out

    # ------------------------------------------------------------------
    # 密钥收集与深度脱敏辅助
    # ------------------------------------------------------------------

    def _collect_known_secrets(self, *extra: str) -> tuple[str, ...]:
        """汇总内存中所有需防泄露的密钥（含外部 API Key、LLM 内存密钥与 Comfy 访问口令）。"""
        items: list[str] = [str(s or "").strip() for s in extra if str(s or "").strip()]
        with self._store.lock:
            for val in self._secrets.runtime_secrets.values():
                s = str(val or "").strip()
                if s:
                    items.append(s)
            for val in self._secrets.comfy_secrets_by_ep.values():
                s = str(val or "").strip()
                if s:
                    items.append(s)
        with suppress_logged("agent", throttled_key="collect_vault_secrets", interval_sec=60.0):
            vault = VaultStore()
            colab_py = self._ctx.get_colab_py()
            vault_key = str(vault.get_secret(AGENT_VAULT_API_KEY, colab_py=colab_py) or "").strip()
            if vault_key:
                register_ephemeral_secret(vault_key)
                items.append(vault_key)
            raw_provs = list(self._store.get_prefs().get("agent_providers") or [])
            for prov in sanitize_agent_providers(raw_provs):
                pkey = str(vault.get_secret(agent_provider_vault_key(prov["id"]), colab_py=colab_py) or "").strip()
                if pkey:
                    register_ephemeral_secret(pkey)
                    items.append(pkey)
        # 去重且保持长度从长到短排序，避免短子串提前替换导致残留尾缀
        unique = sorted(set(items), key=len, reverse=True)
        return tuple(unique)

    def _scrub_obj(self, value: Any, secrets_tuple: tuple[str, ...] = ()) -> Any:
        """递归脱敏字典、列表与字符串中的敏感凭据与运行时密钥。"""
        if isinstance(value, str):
            out = redact_secrets(value, secrets_tuple)
            for sec in secrets_tuple:
                if len(sec) >= 4 and sec in out:
                    out = out.replace(sec, "[REDACTED]")
            return out
        if isinstance(value, (list, tuple)):
            return [self._scrub_obj(x, secrets_tuple) for x in value]
        if isinstance(value, dict):
            return {
                k: ("[REDACTED]" if k in ("api_key", "agent_api_key") and isinstance(v, str) and v else self._scrub_obj(v, secrets_tuple))
                for k, v in value.items()
            }
        return value

    def _emit_agent_event(self, payload: dict[str, Any], secrets_tuple: tuple[str, ...] = ()) -> None:
        """通过事件总线广播副驾实时事件，发送前执行深度脱敏。"""
        safe_payload = self._scrub_obj(payload, secrets_tuple)
        if isinstance(safe_payload, dict):
            safe_payload["type"] = "agent_event"
            self._bus.emit(safe_payload)

    def _append_message(self, msg: dict[str, Any], secrets_tuple: tuple[str, ...] = ()) -> dict[str, Any]:
        """脱敏并写入当前活跃会话的对话历史队列（严格保留最近 MAX_AGENT_MESSAGES 条并原子落盘）。"""
        safe_msg = self._scrub_obj(msg, secrets_tuple)
        with self._lock:
            sess = self._get_active_session_locked()
            sess["messages"].append(safe_msg)
            if len(sess["messages"]) > MAX_AGENT_MESSAGES:
                sess["messages"] = sess["messages"][-MAX_AGENT_MESSAGES:]
            self._messages = sess["messages"]
            sess["updated_ts"] = int(time.time())
            if safe_msg.get("role") == "user" and (not sess.get("title") or sess.get("title") == "新对话"):
                sess["title"] = self._derive_session_title(str(safe_msg.get("content") or ""))
        self._save_sessions_to_disk(secrets_tuple)
        return safe_msg

    # ------------------------------------------------------------------
    # 大脑自动路由（Colab 自部署 Qwen3.8 vs 外部 OpenAI / Anthropic / Gemini API）
    # ------------------------------------------------------------------

    def _collect_colab_llm_context(self) -> dict[str, Any]:
        """收集探测在线 Colab LLM 所需的槽位、元数据、隧道缓存与内存密钥快照。"""
        b = self._b
        cstate = b.get_comfy_state()
        llm_slot = b._cluster_svc.get_slot("llm") if b._cluster_svc.has_active_slots() else None

        llm_ep = str(llm_slot.get("endpoint") or "").strip() if isinstance(llm_slot, dict) else ""
        slot_acct = str(llm_slot.get("account_id") or "").strip() if isinstance(llm_slot, dict) else ""
        slot_url = str(llm_slot.get("tunnel_url") or "").strip().rstrip("/") if isinstance(llm_slot, dict) else ""
        slot_online = bool(llm_slot.get("online")) if isinstance(llm_slot, dict) else False

        if not llm_ep:
            sess = self._store.pick_target_session("comfy")
            if isinstance(sess, dict) and sess.get("endpoint"):
                llm_ep = str(sess.get("endpoint") or "").strip()

        with self._store.lock:
            meta_snapshot = {
                k: dict(v) for k, v in self._secrets.service_meta_by_ep.items() if isinstance(v, dict)
            }
            tunnel_cache = self._store.get_tunnel_cache()
            runtime_secrets = dict(self._secrets.runtime_secrets)

        if not llm_ep:
            for ep_k, m_v in meta_snapshot.items():
                if str(m_v.get("kind") or "").lower() in ("llm", "dual"):
                    llm_ep = ep_k
                    break

        return {
            "cstate": cstate,
            "llm_slot": llm_slot,
            "llm_ep": llm_ep,
            "slot_acct": slot_acct,
            "slot_url": slot_url,
            "slot_online": slot_online,
            "meta_snapshot": meta_snapshot,
            "tunnel_cache": tunnel_cache,
            "runtime_secrets": runtime_secrets,
        }

    @staticmethod
    def _extract_llm_url_from_dict(src: dict[str, Any], kind_key: str) -> str:
        """从运行态或元数据字典提取 LLM 隧道地址。"""
        kind = str(src.get(kind_key) or "").lower()
        raw = src.get("llm_tunnel_url") or (src.get("tunnel_url") if kind == "llm" else "") or ""
        return str(raw).strip().rstrip("/")

    def _resolve_colab_llm_url(
        self,
        ctx_info: dict[str, Any],
        meta: dict[str, Any],
        *,
        probe_network: bool,
    ) -> tuple[str, bool]:
        """按「分槽 -> 运行态 -> 元数据 -> 缓存 -> 按需发现」优先级解析 Colab LLM 隧道地址。"""
        b = self._b
        cstate = ctx_info["cstate"]
        cs_llm_online = bool(cstate.get("llm_online"))
        cs_sk = str(cstate.get("service_kind") or "").lower()
        cs_llm_url = self._extract_llm_url_from_dict(cstate, "service_kind")
        meta_url = self._extract_llm_url_from_dict(meta, "kind")
        has_running = self._store.has_running_session()
        llm_ep = ctx_info["llm_ep"]

        if ctx_info["slot_url"] and (ctx_info["slot_online"] or cs_llm_online):
            return ctx_info["slot_url"], False
        if cs_llm_url and cs_llm_online:
            return cs_llm_url, False
        if meta_url and (cs_llm_online or (has_running and b.has_llm_service_active())):
            return meta_url, False
        if llm_ep and llm_ep in ctx_info["tunnel_cache"] and cs_llm_online and cs_sk == "llm":
            return str(ctx_info["tunnel_cache"].get(llm_ep) or "").strip().rstrip("/"), False

        should_fast = probe_network and (has_running or ctx_info["llm_slot"] is not None or cs_llm_online)
        if should_fast or (not has_running):
            with suppress_logged("agent", throttled_key="resolve_colab_llm_url", interval_sec=30.0):
                u = b.resolve_active_llm_tunnel_url(allow_fast_discover=should_fast)
                if u:
                    return str(u).strip().rstrip("/"), True
        return "", False

    @staticmethod
    def _match_colab_key_from_memory(
        ctx_info: dict[str, Any],
        meta: dict[str, Any],
        colab_key: str,
        resolved_url: str,
    ) -> tuple[str, str, dict[str, Any]]:
        """优先从内存元数据同源匹配、端点直查与单密钥兜底中解析 Colab LLM 密钥。"""
        llm_ep = ctx_info["llm_ep"]
        runtime_secrets = ctx_info["runtime_secrets"]
        if not colab_key and resolved_url:
            for ep_k, m_v in ctx_info["meta_snapshot"].items():
                m_url = str(m_v.get("llm_tunnel_url") or m_v.get("tunnel_url") or "").strip().rstrip("/")
                if m_url == resolved_url and runtime_secrets.get(ep_k):
                    return str(runtime_secrets[ep_k]).strip(), ep_k, m_v
        if not colab_key and llm_ep and runtime_secrets.get(llm_ep):
            colab_key = str(runtime_secrets[llm_ep]).strip()
        if not colab_key and len(runtime_secrets) == 1:
            only_ep, only_k = next(iter(runtime_secrets.items()))
            if str(only_k or "").strip():
                return str(only_k).strip(), (llm_ep or only_ep), meta
        return colab_key, llm_ep, meta

    def _hydrate_colab_key_from_cf(self, resolved_url: str) -> str:
        """从固定 Cloudflare 隧道配置中按同源校验补齐 LLM API 密钥。"""
        if not resolved_url:
            return ""
        with suppress_logged("agent", throttled_key="hydrate_cf_llm_key", interval_sec=60.0):
            cf_cfg = load_cf_tunnel_settings(
                self._ctx.resolve_cf_config_path(),
                include_secrets=True,
                colab_py=self._ctx.get_colab_py(),
            )
            cf_llm_u = str(cf_cfg.get("llm_url") or "").strip().rstrip("/")
            cf_llm_k = str(cf_cfg.get("llm_api_key") or "").strip()
            if cf_llm_u and cf_llm_u == resolved_url and cf_llm_k:
                return cf_llm_k
        return ""

    def _resolve_colab_llm_key(
        self,
        ctx_info: dict[str, Any],
        meta: dict[str, Any],
        colab_key: str,
        resolved_url: str,
        *,
        probe_network: bool,
    ) -> tuple[str, str, dict[str, Any]]:
        """按内存元数据同源匹配、单密钥、快速回读与 CF 固定隧道配置顺序补齐 Colab LLM 密钥。"""
        colab_key, llm_ep, meta = self._match_colab_key_from_memory(ctx_info, meta, colab_key, resolved_url)
        if not colab_key and probe_network and (llm_ep or self._store.has_running_session()):
            with suppress_logged("agent", throttled_key="fast_read_llm_tunnel", interval_sec=30.0):
                self._b._tunnel_svc.fast_read_slot_tunnel_url(llm_ep, "comfy", account_id=ctx_info["slot_acct"])
                meta2, key2 = self._secrets.get_ep_meta_and_key(llm_ep)
                meta = meta2 or meta
                colab_key = key2 or colab_key
        if not colab_key:
            colab_key = self._hydrate_colab_key_from_cf(resolved_url)
        return colab_key, llm_ep, meta

    def _discover_colab_llm(self, *, probe_network: bool = False) -> dict[str, Any]:
        """探测当前在线 Colab Qwen3.8 实例隧道 URL、服务模型与内存密钥。"""
        ctx_info = self._collect_colab_llm_context()
        meta, colab_key = self._secrets.get_ep_meta_and_key(ctx_info["llm_ep"])
        resolved_url, probed_online = self._resolve_colab_llm_url(ctx_info, meta, probe_network=probe_network)
        colab_key, llm_ep, meta = self._resolve_colab_llm_key(
            ctx_info,
            meta,
            colab_key,
            resolved_url,
            probe_network=probe_network,
        )
        if colab_key:
            register_ephemeral_secret(colab_key)

        cstate = ctx_info["cstate"]
        cs_llm_online = bool(cstate.get("llm_online"))
        colab_online = bool(
            resolved_url
            and (
                probed_online
                or cs_llm_online
                or ctx_info["slot_online"]
                or bool(colab_key and self._store.has_running_session())
            )
        )
        colab_model = str(cstate.get("model_name") or "").strip() or self._b._infer_default_llm_model_name(meta)
        api_base = resolved_url if (not resolved_url or resolved_url.endswith("/v1")) else f"{resolved_url}/v1"
        return {
            "online": colab_online,
            "tunnel_url": resolved_url,
            "api_base_url": api_base,
            "api_key": colab_key,
            "model": colab_model,
            "endpoint": llm_ep,
        }

    def _infer_legacy_provider_name(self, base_url: str) -> str:
        if ":3425" in str(base_url or ""):
            return "Magpie 本机网关"
        if is_loopback_base_url(base_url):
            return "本机网关"
        return "外部 API"

    def _write_vault_secret_with_origin(
        self,
        vault: VaultStore,
        vkey: str,
        secret_val: str,
        base_url: str,
        colab_py: Path | None,
    ) -> None:
        """写入凭据保险箱密钥并同步绑定当前 base_url 的标准源站标识。"""
        vault.set_secret(vkey, secret_val, colab_py=colab_py)
        origin_val = url_origin_key(base_url) if str(base_url or "").strip() else ""
        okey = _origin_vault_key(vkey)
        if origin_val:
            vault.set_secret(okey, origin_val, colab_py=colab_py)
        else:
            with suppress_logged("agent", throttled_key="delete_origin_vault_key", interval_sec=30.0):
                vault.delete_secret(okey, colab_py=colab_py)

    def _delete_vault_secret_with_origin(
        self,
        vault: VaultStore,
        vkey: str,
        colab_py: Path | None,
    ) -> None:
        """删除凭据保险箱密钥并同步移除源站硬绑定伴随键。"""
        vault.delete_secret(vkey, colab_py=colab_py)
        vault.delete_secret(_origin_vault_key(vkey), colab_py=colab_py)

    @staticmethod
    def _read_origin_pinned_vault_key(
        vault: VaultStore,
        vkey: str,
        base_url: str,
        origin_cur: str,
        colab_py: Path | None,
    ) -> tuple[str, bool]:
        """读取单条带 `__origin` 硬绑定的密钥，返回 `(secret, had_candidate_but_rejected)`。"""
        try:
            cand = str(vault.get_secret(vkey, colab_py=colab_py) or "").strip()
            if not cand:
                return "", False
            pinned = str(vault.get_secret(_origin_vault_key(vkey), colab_py=colab_py) or "").strip()
            if base_url and origin_cur and pinned and origin_cur == pinned:
                return cand, False
            return "", True
        except Exception:
            return "", False

    def _read_provider_secret(
        self,
        provider_id: str,
        base_url: str = "",
        legacy_base_url: str = "",
    ) -> str:
        """从 VaultStore 严格按源站硬绑定读取指定供应商 API Key（缺少 `__origin` 绑定一律拒读）。"""
        _ = legacy_base_url
        vault = VaultStore()
        colab_py = self._ctx.get_colab_py()
        pid = str(provider_id or "").strip()
        origin_cur = url_origin_key(base_url)
        val = ""
        named_pin_rejected = False
        if pid and pid != "__legacy__":
            val, named_pin_rejected = self._read_origin_pinned_vault_key(
                vault, agent_provider_vault_key(pid), base_url, origin_cur, colab_py
            )
        if not val and not named_pin_rejected:
            val, _ = self._read_origin_pinned_vault_key(
                vault, AGENT_VAULT_API_KEY, base_url, origin_cur, colab_py
            )
        if val:
            register_ephemeral_secret(val)
        return val

    def _load_provider_catalog(self, prefs: dict[str, Any]) -> tuple[list[dict[str, Any]], str, bool]:
        """加载多供应商清单并返回 `(providers, active_provider_id, is_legacy_implicit)`。"""
        saved = sanitize_agent_providers(prefs.get("agent_providers"))
        active_id = str(prefs.get("agent_active_provider_id") or "").strip()
        legacy_url = str(prefs.get("agent_api_base_url") or "").strip().rstrip("/")
        legacy_fmt = str(prefs.get("agent_api_format") or "openai").strip().lower()
        if legacy_fmt not in SUPPORTED_API_FORMATS:
            legacy_fmt = "openai"
        legacy_model = str(prefs.get("agent_model") or "").strip()

        if saved:
            matched = next((p for p in saved if p["id"] == active_id), None)
            if matched is None and legacy_url:
                matched = next((p for p in saved if p["base_url"] == legacy_url), None)
            return saved, (matched["id"] if matched else ""), False

        if legacy_url:
            models = [legacy_model] if legacy_model else []
            virtual = {
                "id": "__legacy__",
                "name": self._infer_legacy_provider_name(legacy_url),
                "api_format": legacy_fmt,
                "base_url": legacy_url,
                "models": models,
                "default_model": legacy_model,
                "model_capabilities": {},
            }
            return [virtual], "__legacy__", True
        return [], "", False

    @staticmethod
    def _clamp_reasoning_effort(
        raw_effort: Any,
        *,
        reasoning_supported: bool,
        supported_levels: list[str],
    ) -> str:
        """将思考强度枚举收敛到模型真实支持的档位集合，避免界面或请求残留非法档位。"""
        eff = normalize_reasoning_effort_value(raw_effort, default="auto")
        if not reasoning_supported:
            return "auto"
        if eff == "auto" or not supported_levels or eff in supported_levels:
            return eff
        if eff == "xhigh" and "max" in supported_levels:
            return "max"
        if eff == "max" and "xhigh" in supported_levels:
            return "xhigh"
        return "auto"

    @staticmethod
    def _resolve_active_reasoning_spec(
        *,
        driver: str,
        resolved_source: str,
        use_model: str,
        use_api_format: str,
        active_prov: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """综合网关 /v1/models 元数据与模型家族推断当前活跃大脑的思考能力规格。"""
        if driver == "claude_code":
            return {
                "reasoning": True,
                "levels": ["low", "medium", "high", "max"],
                "default_level": "",
                "source": "claude_code",
            }
        if resolved_source == "colab_llm":
            return infer_model_reasoning_spec(use_model or "qwen3", "openai")
        caps = (active_prov or {}).get("model_capabilities") or {}
        gw_cap = caps.get(use_model)
        if not isinstance(gw_cap, dict) and use_model.startswith("models/"):
            gw_cap = caps.get(use_model[len("models/") :])
        inferred = infer_model_reasoning_spec(use_model, use_api_format)
        if isinstance(gw_cap, dict) and "reasoning" in gw_cap:
            if not bool(gw_cap.get("reasoning")):
                return {
                    "reasoning": False,
                    "levels": [],
                    "default_level": "",
                    "source": "gateway",
                }
            gw_levels = normalize_reasoning_level_list(gw_cap.get("levels"))
            use_levels = gw_levels if gw_levels else list(inferred.get("levels") or [])
            def_lvl = str(gw_cap.get("default_level") or inferred.get("default_level") or "").strip()
            if def_lvl and use_levels and def_lvl not in use_levels:
                def_lvl = ""
            return {
                "reasoning": True,
                "levels": use_levels,
                "default_level": def_lvl,
                "source": "gateway",
            }
        return inferred

    @staticmethod
    def _default_ext_model(fmt: str) -> str:
        """按协议格式返回外部 API 的默认兜底模型名。"""
        if fmt == "anthropic":
            return "claude-sonnet-4-20250514"
        if fmt == "gemini":
            return "gemini-2.5-flash"
        return "gpt-4o-mini"

    def _normalize_brain_prefs(self, prefs: dict[str, Any]) -> dict[str, Any]:
        """规范化副驾偏好中的驱动、外部供应商、模型及对应同源密钥。"""
        driver = str(prefs.get("agent_driver") or "native").strip().lower()
        if driver not in ("native", "claude_code"):
            driver = "native"
        brain_source = str(prefs.get("agent_brain_source") or "auto").strip().lower()
        if brain_source not in ("auto", "colab_llm", "external_api"):
            brain_source = "auto"
        ext_format = str(prefs.get("agent_api_format") or "openai").strip().lower()
        if ext_format not in SUPPORTED_API_FORMATS:
            ext_format = "openai"
        ext_base_url = str(prefs.get("agent_api_base_url") or "").strip().rstrip("/")
        ext_model = str(prefs.get("agent_model") or "").strip()

        providers, active_prov_id, is_legacy_implicit = self._load_provider_catalog(prefs)
        active_prov = next((p for p in providers if p["id"] == active_prov_id), None)
        active_prov_name = ""
        if active_prov is not None:
            active_prov_name = str(active_prov.get("name") or "")
            if active_prov.get("base_url"):
                ext_base_url = str(active_prov["base_url"]).strip().rstrip("/")
            if active_prov.get("api_format") in SUPPORTED_API_FORMATS:
                ext_format = str(active_prov["api_format"])
            if not ext_model:
                ext_model = str(active_prov.get("default_model") or (active_prov.get("models") or [""])[0] or "").strip()

        ext_api_key = self._read_provider_secret(
            active_prov_id,
            base_url=ext_base_url,
            legacy_base_url=str(prefs.get("agent_api_base_url") or "").strip().rstrip("/"),
        )
        ext_is_loopback = is_loopback_base_url(ext_base_url)
        return {
            "driver": driver,
            "brain_source": brain_source,
            "ext_format": ext_format,
            "ext_base_url": ext_base_url,
            "ext_model": ext_model,
            "raw_reasoning_effort": normalize_reasoning_effort_value(prefs.get("agent_reasoning_effort"), default="auto"),
            "auto_approve": bool(prefs.get("agent_auto_approve_billing", False)),
            "active_prov": active_prov,
            "active_prov_id": active_prov_id,
            "active_prov_name": active_prov_name,
            "ext_label_prefix": active_prov_name if (active_prov_name and not is_legacy_implicit) else "外部 API",
            "ext_api_key": ext_api_key,
            "ext_is_loopback": ext_is_loopback,
            "has_ext_ready": bool(ext_base_url and (ext_api_key or ext_is_loopback)),
        }

    def _select_brain_route(
        self,
        norm: dict[str, Any],
        colab_info: dict[str, Any],
        claude_avail: bool,
    ) -> dict[str, Any]:
        """根据驱动与路由偏好决定最终生效的大脑来源、连接参数与状态提示。"""
        driver = norm["driver"]
        brain_source = norm["brain_source"]
        ext_format = norm["ext_format"]
        ext_base_url = norm["ext_base_url"]
        ext_api_key = norm["ext_api_key"]
        ext_model = norm["ext_model"]
        ext_label_prefix = norm["ext_label_prefix"]
        has_ext_ready = norm["has_ext_ready"]
        colab_online = bool(colab_info["online"])

        if driver == "claude_code":
            return {
                "resolved_source": "claude_code",
                "resolved_label": "Claude Code CLI" if claude_avail else "Claude Code CLI (未安装)",
                "ready": bool(claude_avail),
                "reason": "" if claude_avail else "未检测到本地 claude CLI，请安装 Claude Code 或切换至 Native 轻量引擎",
                "use_base_url": "",
                "use_api_key": "",
                "use_api_format": ext_format,
                "use_model": ext_model,
            }

        if brain_source == "colab_llm":
            use_base_url = str(colab_info["api_base_url"])
            use_model = str(colab_info["model"])
            ready = bool(colab_online and use_base_url)
            return {
                "resolved_source": "colab_llm",
                "resolved_label": f"Colab {use_model}" if ready else "Colab Qwen3.8 (离线)",
                "ready": ready,
                "reason": "" if ready else "当前未检测到在线 Colab Qwen3.8 服务，请先开机或切回自动/外部 API",
                "use_base_url": use_base_url,
                "use_api_key": str(colab_info["api_key"]),
                "use_api_format": "openai",
                "use_model": use_model,
            }

        if brain_source == "external_api":
            use_model = ext_model or self._default_ext_model(ext_format)
            reason = ""
            if not has_ext_ready:
                reason = (
                    "请先点击「管理与导入模型」填写网关 Base URL（如 Magpie 本机 127.0.0.1:3425）"
                    if not ext_base_url
                    else "请为当前远程供应商填写 API Key（本机回环网关可留空）"
                )
            return {
                "resolved_source": "external_api",
                "resolved_label": f"{ext_label_prefix} · {use_model}" if has_ext_ready else "外部 API (未配置)",
                "ready": has_ext_ready,
                "reason": reason,
                "use_base_url": ext_base_url,
                "use_api_key": ext_api_key,
                "use_api_format": ext_format,
                "use_model": use_model,
            }

        # auto 自动路由：优先已完整配置的外部 API/本机网关，否则自动接管在线 Colab Qwen3.8
        if has_ext_ready:
            use_model = ext_model or self._default_ext_model(ext_format)
            return {
                "resolved_source": "external_api",
                "resolved_label": f"{ext_label_prefix} · {use_model}",
                "ready": True,
                "reason": "",
                "use_base_url": ext_base_url,
                "use_api_key": ext_api_key,
                "use_api_format": ext_format,
                "use_model": use_model,
            }
        if colab_online and colab_info["api_base_url"]:
            use_model = str(colab_info["model"])
            return {
                "resolved_source": "colab_llm",
                "resolved_label": f"Colab {use_model}",
                "ready": True,
                "reason": "",
                "use_base_url": str(colab_info["api_base_url"]),
                "use_api_key": str(colab_info["api_key"]),
                "use_api_format": "openai",
                "use_model": use_model,
            }
        return {
            "resolved_source": "none",
            "resolved_label": "未就绪 (待开机或配 API)",
            "ready": False,
            "reason": "未检测到在线 Colab Qwen3.8 服务，且尚未配置外部 API 或本机 Magpie 网关",
            "use_base_url": "",
            "use_api_key": "",
            "use_api_format": ext_format,
            "use_model": "",
        }

    def _resolve_brain_runtime(self, *, probe_network: bool = False) -> tuple[dict[str, Any], dict[str, Any]]:
        """解析当前生效的大脑路由（返回不含密钥的公开描述与仅供内部调用的运行时凭据）。"""
        norm = self._normalize_brain_prefs(self._get_merged_prefs())
        colab_info = self._discover_colab_llm(probe_network=probe_network)
        claude_avail = ClaudeCodeDriver.is_available()
        route = self._select_brain_route(norm, colab_info, claude_avail)

        use_api_format = route["use_api_format"]
        use_model = route["use_model"]
        spec_model = use_model or norm["ext_model"] or self._default_ext_model(use_api_format or norm["ext_format"])
        reasoning_spec = self._resolve_active_reasoning_spec(
            driver=norm["driver"],
            resolved_source=route["resolved_source"],
            use_model=spec_model,
            use_api_format=use_api_format or norm["ext_format"],
            active_prov=norm["active_prov"],
        )
        supported_levels = list(reasoning_spec.get("levels") or [])
        reasoning_supported = bool(reasoning_spec.get("reasoning", True))
        reasoning_effort = self._clamp_reasoning_effort(
            norm["raw_reasoning_effort"],
            reasoning_supported=reasoning_supported,
            supported_levels=supported_levels,
        )

        public_brain = {
            "driver": norm["driver"],
            "brain_source": norm["brain_source"],
            "resolved_source": route["resolved_source"],
            "resolved_brain_label": route["resolved_label"],
            "colab_llm_online": bool(colab_info["online"]),
            "colab_tunnel_url": str(colab_info["tunnel_url"] or ""),
            "colab_model": str(colab_info["model"] or ""),
            "claude_code_available": claude_avail,
            "has_external_key": bool(norm["ext_api_key"]),
            "external_ready": norm["has_ext_ready"],
            "external_is_loopback": norm["ext_is_loopback"],
            "active_provider_id": norm["active_prov_id"],
            "active_provider_name": norm["active_prov_name"],
            "auto_approve_billing": norm["auto_approve"],
            "active_format": use_api_format,
            "active_model": use_model,
            "reasoning_effort": reasoning_effort,
            "supported_reasoning_efforts": supported_levels,
            "reasoning_supported": reasoning_supported,
            "default_reasoning_effort": str(reasoning_spec.get("default_level") or ""),
            "reasoning_spec_source": str(reasoning_spec.get("source") or "fallback"),
            "ready": route["ready"],
            "reason": route["reason"],
        }
        runtime_cfg = {
            "driver": norm["driver"],
            "resolved_source": route["resolved_source"],
            "base_url": route["use_base_url"],
            "api_key": route["use_api_key"],
            "api_format": use_api_format,
            "model": use_model,
            "reasoning_effort": reasoning_effort,
            "supported_reasoning_efforts": supported_levels,
            "reasoning_supported": reasoning_supported,
            "ext_api_key": norm["ext_api_key"],
            "colab_api_key": str(colab_info["api_key"] or ""),
            "ready": route["ready"],
            "reason": route["reason"],
        }
        return public_brain, runtime_cfg

    def _resolve_brain_config(self, *, probe_network: bool = False) -> dict[str, Any]:
        """解析并返回不含任何明文密钥的大脑路由状态描述。"""
        public_brain, _ = self._resolve_brain_runtime(probe_network=probe_network)
        return public_brain

    # ------------------------------------------------------------------
    # 10 工具本地 + 远端同进程执行器
    # ------------------------------------------------------------------

    def _tool_get_cluster_and_pool_status(self, args: dict[str, Any], _cancel: CancelToken | None) -> dict[str, Any]:
        b = self._b
        if bool(args.get("refresh_pool")):
            pool_res = b.refresh_account_pool()
            snap = pool_res.get("snapshot") if isinstance(pool_res.get("snapshot"), dict) else b.get_cached_snapshot()
        else:
            snap = b.get_cached_snapshot()
        cstate = b.get_comfy_state()
        dstate = b.get_deploy_state()
        ob = snap.get("onboarding") if isinstance(snap.get("onboarding"), dict) else {}
        accts_out = [
            {
                "id": a.get("id"),
                "label": a.get("label"),
                "balance": a.get("last_balance"),
                "is_running": bool(a.get("last_is_running")),
                "eligible_gpus": a.get("eligible_gpus"),
                "cluster_role": a.get("cluster_role"),
                "is_active": bool(a.get("is_active")),
            }
            for a in (ob.get("accounts") or [])
            if isinstance(a, dict)
        ]
        return {
            "ok": True,
            "is_running": bool(snap.get("is_running")),
            "balance": snap.get("balance"),
            "rate_hourly": snap.get("rate_hourly"),
            "assignments_count": snap.get("assignments_count"),
            "active_account_id": (snap.get("prefs") or {}).get("active_account_id", ""),
            "sessions": list(snap.get("sessions") or []),
            "cluster": snap.get("cluster") or {},
            "accounts": accts_out,
            "comfy": {
                "online": bool(cstate.get("online")),
                "service_kind": cstate.get("service_kind"),
                "comfy_online": bool(cstate.get("comfy_online")),
                "llm_online": bool(cstate.get("llm_online")),
                "tunnel_url": cstate.get("tunnel_url"),
                "llm_tunnel_url": cstate.get("llm_tunnel_url"),
                "running_count": cstate.get("running_count"),
                "pending_count": cstate.get("pending_count"),
                "gpu_name": cstate.get("gpu_name"),
                "vram_used_gb": cstate.get("vram_used_gb"),
                "vram_total_gb": cstate.get("vram_total_gb"),
                "has_error": bool(cstate.get("has_error")),
                "error_summary": cstate.get("error_summary"),
                "model_name": cstate.get("model_name"),
                "quant": cstate.get("quant"),
                "ctx_size": cstate.get("ctx_size"),
                "tps": cstate.get("tps"),
            },
            "deploy": dstate,
        }

    def _tool_tail_instance_logs(self, args: dict[str, Any], _cancel: CancelToken | None) -> dict[str, Any]:
        source = str(args.get("source") or "local_diag").strip().lower()
        try:
            lines = max(10, min(500, int(args.get("lines") or 100)))
        except (TypeError, ValueError):
            lines = 100
        return self._b.tail_live_logs({"source": source, "lines": lines})

    def _tool_list_gallery_and_inspect(self, args: dict[str, Any], _cancel: CancelToken | None) -> dict[str, Any]:
        rel_path = str(args.get("rel_path") or "").strip()
        if rel_path:
            return self._b.inspect_gallery_item(rel_path)
        try:
            limit = max(1, min(100, int(args.get("limit") or 20)))
        except (TypeError, ValueError):
            limit = 20
        return self._b.list_gallery_items({"kind": "all", "limit": limit})

    def _tool_list_gdrive_assets(self, _args: dict[str, Any], _cancel: CancelToken | None) -> dict[str, Any]:
        return self._b.list_cloud_assets()

    def _tool_run_environment_doctor(self, args: dict[str, Any], _cancel: CancelToken | None) -> dict[str, Any]:
        prof = str(args.get("profile") or "").strip().lower()
        gpu = str(args.get("gpu") or "").strip().upper() or None
        return self._b.run_preflight(
            deep=True,
            profiles=[prof] if prof else None,
            gpu=gpu,
        )

    def _resolve_exec_target(self, slot: str) -> tuple[str, str, str, str | None]:
        """按 slot 解析远端执行目标的 `(session, account_id, endpoint, error_msg)`。"""
        b = self._b
        slot_norm = str(slot or "auto").strip().lower()
        if slot_norm not in ("auto", "comfy", "llm"):
            slot_norm = "auto"

        if b._cluster_svc.has_active_slots():
            slots = b._cluster_svc.get_cluster_slots()
            picked = (
                slots.get("llm")
                if slot_norm == "llm"
                else (slots.get("comfy") if slot_norm == "comfy" else (slots.get("comfy") or slots.get("llm")))
            )
            if not isinstance(picked, dict):
                return "", "", "", f"当前异构集群未运行 {slot_norm} 槽位"
            return (
                str(picked.get("session") or "comfy").strip() or "comfy",
                str(picked.get("account_id") or "").strip(),
                str(picked.get("endpoint") or "").strip(),
                None,
            )

        sname, ep, _ = b._resolve_target_binding("comfy")
        acct_id = b._current_account_key()
        return sname or "comfy", acct_id, ep, None

    def _request_user_confirm(
        self,
        tool: str,
        title: str,
        detail: str,
        args: dict[str, Any],
        cancel_token: CancelToken | None,
    ) -> tuple[bool, str]:
        """向前端广播人工审批卡片并线程安全挂起等待批准、拒绝、手动取消或超时。"""
        if cancel_token is not None and cancel_token.is_cancelled:
            return False, "操作已手动取消"

        confirm_id = f"cfm_{uuid.uuid4().hex[:10]}"
        wait_evt = threading.Event()
        confirm_obj = {
            "id": confirm_id,
            "tool": str(tool or "").strip(),
            "title": str(title or "确认执行敏感操作"),
            "detail": str(detail or ""),
            "args": dict(args or {}),
            "created_ts": int(time.time()),
        }
        with self._lock:
            turn_id = self._active_turn_id
            self._pending_confirm = confirm_obj
            self._confirm_event = wait_evt
            self._confirm_result = None

        if cancel_token is not None:
            cancel_token.register_callback(wait_evt.set)

        self._emit_agent_event({"phase": "confirm_required", "turn_id": turn_id, "confirm": confirm_obj})
        signaled = wait_evt.wait(timeout=CONFIRM_TIMEOUT_SEC)

        with self._lock:
            res_flag = self._confirm_result
            self._pending_confirm = None
            self._confirm_event = None
            self._confirm_result = None

        cancelled_now = bool(cancel_token is not None and cancel_token.is_cancelled)
        approved = bool(res_flag is True) and not cancelled_now
        self._emit_agent_event(
            {
                "phase": "confirm_resolved",
                "turn_id": turn_id,
                "confirm_id": confirm_id,
                "approved": approved,
            }
        )
        if approved:
            return True, ""
        if cancelled_now:
            return False, "操作已手动取消"
        if not signaled:
            return False, "等待用户审批超时，已自动取消"
        return False, "用户拒绝执行该操作"

    def _is_remote_exec_auto_approved(self) -> bool:
        """判断是否已显式开启非只读远端代码自动放行策略。

        桌面 GUI 模式彻底忽略环境变量 `COLAB_AGENT_AUTO_APPROVE_EXEC`，防止本机遗留
        环境变量静默降级为仅剩正则黑名单；非只读远端执行默认且始终强制人工审批。
        """
        if str(os.environ.get("COLAB_AGENT_AUTO_APPROVE_EXEC") or "").strip():
            self._b._diag.debug_throttled(
                "agent",
                "ignore_env_auto_approve_exec",
                "GUI 模式下已安全忽略环境变量 COLAB_AGENT_AUTO_APPROVE_EXEC，非只读远端执行仍强制人工审批",
            )
        prefs = self._store.get_prefs()
        raw_pref = prefs.get("agent_auto_approve_remote_exec", False)
        if isinstance(raw_pref, str):
            return raw_pref.strip().lower() not in ("", "0", "false", "no", "off")
        return bool(raw_pref)

    def _confirm_remote_exec_if_needed(
        self,
        *,
        code: str,
        mode: str,
        slot: str,
        sname: str,
        acct_id: str,
        ep: str,
        timeout_sec: int,
        cancel_token: CancelToken | None,
    ) -> dict[str, Any] | None:
        """对非只读远端代码执行第二道人工审批门禁；放行返回 None，拒绝返回拦截结果字典。"""
        readonly_ok, ro_reason = is_readonly_safe_remote_cmd(code, mode)
        if readonly_ok or self._is_remote_exec_auto_approved():
            return None
        target_desc = f"槽位: {slot} | 会话: {sname}" + (f" | 账号: {acct_id}" if acct_id else "")
        if ep:
            target_desc += f" | 端点: {ep}"
        title = f"确认在远端 Colab 执行 {mode.upper()} 代码"
        detail = f"{target_desc} | 模式: {mode}\n原因: {ro_reason}\n待执行代码:\n{code}"
        confirm_args = {
            "code": code,
            "mode": mode,
            "slot": slot,
            "session": sname,
            "account_id": acct_id,
            "timeout_sec": timeout_sec,
        }
        approved, cfm_reason = self._request_user_confirm("exec_remote_colab", title, detail, confirm_args, cancel_token)
        if approved:
            return None
        return {
            "ok": False,
            "cancelled": True,
            "approval_required": True,
            "reason": cfm_reason or "用户取消远端代码执行",
        }

    @staticmethod
    def _format_remote_exec_result(
        *,
        out_raw: Any,
        rc: int,
        slot: str,
        sname: str,
        acct_id: str,
        ep: str,
        cancel_token: CancelToken | None,
    ) -> dict[str, Any]:
        """整理并脱敏截断远端执行子进程的输出与返回字典。"""
        safe_out = redact_secrets(str(out_raw or "")).strip()
        if len(safe_out) > _MAX_EXEC_OUTPUT_CHARS:
            safe_out = "...(已截断前文)\n" + safe_out[-_MAX_EXEC_OUTPUT_CHARS:]
        if cancel_token is not None and cancel_token.is_cancelled:
            return {"ok": False, "cancelled": True, "output": safe_out, "error": "远端执行已取消"}
        res: dict[str, Any] = {
            "ok": rc == 0,
            "returncode": rc,
            "slot": slot,
            "session": sname,
            "account_id": acct_id,
            "endpoint": ep,
            "output": safe_out,
        }
        if rc != 0:
            res["error"] = safe_out.splitlines()[-1] if safe_out else f"退出码 {rc}"
        return res

    def _tool_exec_remote_colab(self, args: dict[str, Any], cancel_token: CancelToken | None) -> dict[str, Any]:
        code = str(args.get("code") or "")
        mode = str(args.get("mode") or "bash").strip().lower()
        slot = str(args.get("slot") or "auto").strip().lower()
        timeout_sec = max(5, min(600, self._parse_optional_int(args.get("timeout_sec"), default=180) or 180))

        # 第一道防线：硬黑名单安全护栏（命中破坏性操作或凭据窥探立即拒绝，绝不弹审批）
        safe_ok, reason = check_remote_code_safety(code, mode)
        if not safe_ok:
            return {"ok": False, "blocked": True, "error": reason or "安全策略拦截：禁止执行高危远端指令"}
        if not code.strip():
            return {"ok": False, "error": "exec_remote_colab 的 code 参数不能为空"}
        if not self._ctx.ctl_available():
            return {"ok": False, "error": f"未找到控制脚本: {self._ctx.ctl_script.name}"}

        sname, acct_id, ep, slot_err = self._resolve_exec_target(slot)
        if slot_err:
            return {"ok": False, "error": slot_err}

        # 第二道防线：严格只读诊断白名单直通，其余非只读 Bash 或任意 Python 一律触发人工审批弹窗
        blocked_res = self._confirm_remote_exec_if_needed(
            code=code, mode=mode, slot=slot, sname=sname, acct_id=acct_id, ep=ep,
            timeout_sec=timeout_sec, cancel_token=cancel_token,
        )
        if blocked_res is not None:
            return blocked_res

        cmd = [*self._ctx.ctl_cmd_prefix(unbuffered=True), "exec", "-s", sname, "--mode", mode, "--timeout", str(timeout_sec)]
        if acct_id:
            cmd.extend(["--account", acct_id])
        if cancel_token is not None and cancel_token.is_cancelled:
            return {"ok": False, "cancelled": True, "error": "操作已手动取消"}

        proc: subprocess.Popen[str] | None = None
        try:
            # 必须保留模块级 subprocess.Popen 调用，确保白盒安全单测对 ba_mod.subprocess.Popen 的补丁生效
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=clean_subprocess_env(self._ctx.build_ctl_env()),
                creationflags=CREATE_NO_WINDOW,
            )
            if cancel_token is not None:
                cancel_token.register_callback(lambda: kill_process_tree(proc))

            out_raw, _ = proc.communicate(input=code, timeout=timeout_sec)
            return self._format_remote_exec_result(
                out_raw=out_raw,
                rc=int(proc.returncode or 0),
                slot=slot,
                sname=sname,
                acct_id=acct_id,
                ep=ep,
                cancel_token=cancel_token,
            )
        except subprocess.TimeoutExpired:
            kill_process_tree(proc)
            return {"ok": False, "timed_out": True, "error": f"远端指令执行超时 ({timeout_sec}s)"}
        except Exception as exc:
            kill_process_tree(proc)
            return {"ok": False, "error": f"远端执行异常: {redact_secrets(str(exc) or type(exc).__name__)}"}

    def _resolve_online_comfy_target(self) -> tuple[str, str]:
        """解析当前在线 ComfyUI 的公网地址与关联 endpoint。"""
        b = self._b
        comfy_ep = ""
        if b._cluster_svc.has_active_slots():
            cslot = b._cluster_svc.get_slot("comfy")
            if isinstance(cslot, dict):
                comfy_ep = str(cslot.get("endpoint") or "").strip()
        if not comfy_ep:
            sess = self._store.pick_target_session("comfy")
            if isinstance(sess, dict) and sess.get("endpoint"):
                comfy_ep = str(sess.get("endpoint") or "").strip()

        turl = ""
        cstate = b.get_comfy_state()
        if cstate.get("comfy_online") or (cstate.get("online") and cstate.get("service_kind") in ("comfy", "dual")):
            turl = str(cstate.get("tunnel_url") or "").strip().rstrip("/")
        if not turl and b._cluster_svc.has_active_slots():
            cslot = b._cluster_svc.get_slot("comfy")
            if isinstance(cslot, dict) and cslot.get("online") and cslot.get("tunnel_url"):
                turl = str(cslot["tunnel_url"]).strip().rstrip("/")
        if not turl:
            try:
                u = b.resolve_active_tunnel_url(session="comfy", allow_fast_discover=True)
                if u:
                    turl = str(u).strip().rstrip("/")
            except Exception:
                turl = ""
        return turl, comfy_ep

    @staticmethod
    def _extract_png_api_prompt(png_path: Path) -> dict[str, Any] | None:
        """从 PNG 文本块中提取合法 ComfyUI API prompt 字典。"""
        chunks = _read_png_text_chunks(png_path)
        raw_prompt = chunks.get("prompt")
        if not isinstance(raw_prompt, str) or not raw_prompt.strip():
            return None
        try:
            parsed = json.loads(raw_prompt)
            return parsed if is_api_prompt_dict(parsed) else None
        except Exception:
            return None

    def _load_prompt_from_file(self, base_rel_path: str) -> tuple[dict[str, Any] | None, str, str]:
        """从用户指定的本地 PNG/JSON 文件加载 ComfyUI API prompt 模板。"""
        try:
            _, target, rel_posix = self._b._studio_svc._resolve_media_item_path(base_rel_path)
        except PermissionError as exc:
            return None, "", str(exc)
        if not target.is_file():
            return None, "", f"基础作品文件不存在: {base_rel_path}"
        suf = target.suffix.lower()
        if suf == ".png":
            parsed = self._extract_png_api_prompt(target)
            if parsed is not None:
                return parsed, rel_posix, ""
            return None, rel_posix, f"{target.name} 未包含可提交的 ComfyUI API prompt 元数据"
        if suf == ".json":
            try:
                parsed = json.loads(target.read_text(encoding="utf-8"))
                if is_api_prompt_dict(parsed):
                    return parsed, rel_posix, ""
            except Exception as exc:
                return None, rel_posix, f"解析工作流 JSON 失败: {redact_secrets(str(exc))}"
            return None, rel_posix, "该 JSON 为前端 UI 格式而非 API prompt 拓扑，请选择 ComfyUI 生成的 PNG 作品"
        return None, rel_posix, f"不支持作为工作流基础模板的文件类型: {suf}"

    def _load_prompt_from_gallery(self) -> tuple[dict[str, Any] | None, str]:
        """从本地画廊中挑选最新一张含 API prompt 的 PNG 作品。"""
        b = self._b
        gal = b.list_gallery_items({"kind": "image", "limit": 20})
        root = b.download_root.resolve()
        for item in gal.get("items") or []:
            rp = str(item.get("rel_path") or "")
            if not rp.lower().endswith(".png"):
                continue
            cand = (root / rp).resolve()
            if not cand.is_file():
                continue
            parsed = self._extract_png_api_prompt(cand)
            if parsed is not None:
                return parsed, rp
        return None, ""

    def _load_prompt_from_remote_history(self, comfy_url: str, comfy_ep: str) -> tuple[dict[str, Any] | None, str]:
        """读取云端 ComfyUI 最近一次成功历史执行图作为工作流模板。"""
        with suppress_logged("agent", throttled_key="load_remote_history_prompt", interval_sec=30.0):
            hr = self._b._comfy_http_get(f"{comfy_url}/history?max_items=5", base_url=comfy_url, ep=comfy_ep, timeout=8)
            if hr.status_code != 200:
                return None, ""
            hdata = hr.json()
            if not isinstance(hdata, dict):
                return None, ""
            for pid, hitem in hdata.items():
                prompt_entry = hitem.get("prompt") if isinstance(hitem, dict) else None
                if isinstance(prompt_entry, list) and len(prompt_entry) >= 3 and is_api_prompt_dict(prompt_entry[2]):
                    return prompt_entry[2], f"remote_history:{pid}"
        return None, ""

    def _build_fallback_ckpt_prompt(self, comfy_url: str, comfy_ep: str) -> tuple[dict[str, Any] | None, str]:
        """通过远端 CheckpointLoaderSimple 列表构造标准文生图兜底模板。"""
        try:
            oir = self._b._comfy_http_get(
                f"{comfy_url}/object_info/CheckpointLoaderSimple",
                base_url=comfy_url,
                ep=comfy_ep,
                timeout=8,
            )
            if oir.status_code != 200:
                return None, ""
            odata = oir.json()
            ckpt_spec = (
                ((odata.get("CheckpointLoaderSimple") or {}).get("input") or {}).get("required") or {}
            ).get("ckpt_name")
            ckpt_list = ckpt_spec[0] if isinstance(ckpt_spec, list) and ckpt_spec and isinstance(ckpt_spec[0], list) else []
            if not ckpt_list:
                return None, ""
            ckpt_name = str(ckpt_list[0])
            fallback_graph = {
                "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt_name}},
                "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "masterpiece, best quality", "clip": ["1", 1]}},
                "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "lowres, bad anatomy, blurry", "clip": ["1", 1]}},
                "4": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
                "5": {
                    "class_type": "KSampler",
                    "inputs": {
                        "model": ["1", 0],
                        "positive": ["2", 0],
                        "negative": ["3", 0],
                        "latent_image": ["4", 0],
                        "seed": 42,
                        "steps": 24,
                        "cfg": 5.0,
                        "sampler_name": "euler",
                        "scheduler": "normal",
                        "denoise": 1.0,
                    },
                },
                "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
                "7": {"class_type": "SaveImage", "inputs": {"images": ["6", 0], "filename_prefix": "ColabAgent"}},
            }
            return fallback_graph, f"cloud_template:{ckpt_name}"
        except Exception:
            return None, ""

    def _load_prompt_graph_template(
        self, base_rel_path: str, comfy_url: str, comfy_ep: str
    ) -> tuple[dict[str, Any] | None, str, str]:
        """按给定 rel_path 或画廊最新 PNG / 云端历史记录 / 云端默认检查点加载 ComfyUI API prompt 图。"""
        if base_rel_path:
            return self._load_prompt_from_file(base_rel_path)

        for loader in (
            self._load_prompt_from_gallery,
            lambda: self._load_prompt_from_remote_history(comfy_url, comfy_ep),
            lambda: self._build_fallback_ckpt_prompt(comfy_url, comfy_ep),
        ):
            graph, src_label = loader()
            if graph is not None:
                return graph, src_label, ""

        return None, "", "未找到可用的基础工作流（请先在画廊保留一张 ComfyUI 生成的 PNG 或指定 base_rel_path）"

    @staticmethod
    def _parse_optional_int(raw_val: Any, default: int | None = None) -> int | None:
        """安全解析可选整型参数，失败时返回默认值。"""
        if raw_val is None:
            return default
        try:
            return int(raw_val)
        except (TypeError, ValueError):
            return default

    def _tool_queue_comfy_workflow(self, args: dict[str, Any], _cancel: CancelToken | None) -> dict[str, Any]:
        b = self._b
        comfy_url, comfy_ep = self._resolve_online_comfy_target()
        if not comfy_url:
            return {"ok": False, "error": "当前无在线 ComfyUI 实例，请先开机部署画布服务"}

        base_rel_path = str(args.get("base_rel_path") or "").strip()
        base_graph, used_source, load_err = self._load_prompt_graph_template(base_rel_path, comfy_url, comfy_ep)
        if base_graph is None:
            return {"ok": False, "error": load_err}

        patched_graph, applied = _patch_comfy_api_graph(
            base_graph,
            positive_prompt=str(args.get("positive_prompt") or ""),
            negative_prompt=str(args.get("negative_prompt") or ""),
            seed=self._parse_optional_int(args.get("seed")),
            steps=self._parse_optional_int(args.get("steps")),
            batch_size=self._parse_optional_int(args.get("batch_size", 1), default=1),
        )

        client_id = f"colab_agent_{uuid.uuid4().hex[:8]}"
        try:
            resp = b._comfy_http_post(
                f"{comfy_url}/prompt",
                base_url=comfy_url,
                ep=comfy_ep,
                json={"prompt": patched_graph, "client_id": client_id},
                timeout=20,
            )
        except Exception as exc:
            return {"ok": False, "error": f"提交 ComfyUI /prompt 失败: {redact_secrets(str(exc))}"}

        try:
            data = resp.json() if resp.content else {}
        except Exception:
            data = {}
        if resp.status_code != 200 or not isinstance(data, dict) or data.get("error") or data.get("node_errors") or not data.get("prompt_id"):
            err_detail = redact_secrets(json.dumps(data, ensure_ascii=False)[:400]) if data else f"HTTP {resp.status_code}"
            return {"ok": False, "status_code": resp.status_code, "error": f"ComfyUI 拒绝入队: {err_detail}"}

        b.touch_activity("comfy")
        prompt_id = str(data["prompt_id"])
        return {
            "ok": True,
            "prompt_id": prompt_id,
            "number": data.get("number"),
            "base_source": used_source,
            "applied": applied,
            "message": f"任务已成功入队 ComfyUI (prompt_id={prompt_id})，出图完成后将自动拉回或提示（无需同步死等）。",
        }

    def _tool_sync_and_pull_outputs(self, _args: dict[str, Any], _cancel: CancelToken | None) -> dict[str, Any]:
        return self._b.sync_outputs()

    def _tool_reconfigure_llm_engine(self, args: dict[str, Any], _cancel: CancelToken | None) -> dict[str, Any]:
        cfg: dict[str, Any] = {}
        if args.get("ctx") is not None:
            cfg["ctx"] = args["ctx"]
        if args.get("quant") is not None:
            cfg["quant"] = args["quant"]
        if args.get("mmproj") is not None:
            cfg["mmproj"] = bool(args["mmproj"])
        return self._b.reconfigure_running_llm(cfg)

    @staticmethod
    def _build_compute_confirm_copy(
        action: str,
        profiles: list[str],
        gpu: str | None,
        cluster_mode: str,
        slot: str,
        prefs: dict[str, Any],
    ) -> tuple[str, str]:
        """构造开关机与槽位释放的人工审批弹窗标题与说明文案。"""
        if action == "start":
            prof_txt = ", ".join(profiles) if profiles else str(prefs.get("boot_profile") or "默认工作流")
            gpu_txt = gpu or str(prefs.get("boot_gpu") or "自动选卡")
            return (
                "确认开机部署 Colab 计算实例",
                f"操作: 开机 ({prof_txt}) | 卡型: {gpu_txt} | 模式: {cluster_mode}（将触发 CU 计费）",
            )
        if action == "stop_slot":
            return f"确认释放集群槽位 ({slot})", f"操作: 停止 {slot} 槽位实例并停止该槽位计费"
        return "确认停止全部 Colab 实例", "操作: 关闭当前账号/集群下全部活跃实例并停止计费"

    def _tool_manage_compute_instance(self, args: dict[str, Any], cancel_token: CancelToken | None) -> dict[str, Any]:
        b = self._b
        action = str(args.get("action") or "start").strip().lower()
        if action not in ("start", "stop_slot", "stop_all"):
            return {"ok": False, "error": f"不支持的实例操作类型: {action}"}

        profiles = _parse_str_list(args.get("profiles"))
        gpu = str(args.get("gpu") or "").strip().upper() or None
        cluster_mode = str(args.get("cluster_mode") or "auto").strip().lower() or "auto"
        slot = str(args.get("slot") or "comfy").strip().lower() or "comfy"

        prefs = self._store.get_prefs()
        if not bool(prefs.get("agent_auto_approve_billing", False)):
            title, detail = self._build_compute_confirm_copy(action, profiles, gpu, cluster_mode, slot, prefs)
            approved, _ = self._request_user_confirm("manage_compute_instance", title, detail, dict(args), cancel_token)
            if not approved:
                return {"ok": False, "cancelled": True, "reason": "用户取消计费操作"}

        if action == "start":
            start_cfg: dict[str, Any] = {"cluster_mode": cluster_mode}
            if profiles:
                start_cfg["profiles"] = profiles
            if gpu:
                start_cfg["gpu"] = gpu
            return b.start_instance(start_cfg)
        if action == "stop_slot":
            if b._cluster_svc.has_active_slots() or slot == "all":
                return b.stop_cluster_slot(slot)
            return b.stop_session_instance(session=slot)
        return b.stop_all_instances()

    def _execute_tool(
        self, name: str, args: dict[str, Any], cancel_token: CancelToken | None, secrets_tuple: tuple[str, ...]
    ) -> dict[str, Any]:
        """按标准化工具名路由并执行对应操作，统一拦截异常并脱敏返回。"""
        dispatch = {
            "get_cluster_and_pool_status": self._tool_get_cluster_and_pool_status,
            "tail_instance_logs": self._tool_tail_instance_logs,
            "list_gallery_and_inspect": self._tool_list_gallery_and_inspect,
            "list_gdrive_assets": self._tool_list_gdrive_assets,
            "run_environment_doctor": self._tool_run_environment_doctor,
            "exec_remote_colab": self._tool_exec_remote_colab,
            "queue_comfy_workflow": self._tool_queue_comfy_workflow,
            "sync_and_pull_outputs": self._tool_sync_and_pull_outputs,
            "reconfigure_llm_engine": self._tool_reconfigure_llm_engine,
            "manage_compute_instance": self._tool_manage_compute_instance,
        }
        handler = dispatch.get(str(name or "").strip())
        if handler is None:
            return {"ok": False, "error": f"未知工具: {name}"}
        try:
            raw = handler(args if isinstance(args, dict) else {}, cancel_token)
            res_dict = raw if isinstance(raw, dict) else {"ok": True, "result": raw}
        except Exception as exc:
            res_dict = {"ok": False, "error": redact_secrets(str(exc) or type(exc).__name__, secrets_tuple)}
        return self._scrub_obj(res_dict, secrets_tuple)

    # ------------------------------------------------------------------
    # 对外公开标准控制与多供应商发现探测方法
    # ------------------------------------------------------------------

    def get_agent_state(self) -> dict[str, Any]:
        """返回副驾当前配置、多供应商模型目录、大脑路由状态、忙碌状态、待审批卡及最近 40 条历史。"""
        prefs = self._get_merged_prefs()
        brain = self._resolve_brain_config(probe_network=False)
        secrets_tuple = self._collect_known_secrets()
        providers_raw, active_prov_id, _ = self._load_provider_catalog(prefs)
        legacy_url = str(prefs.get("agent_api_base_url") or "").strip().rstrip("/")

        providers_out: list[dict[str, Any]] = []
        for p in providers_raw:
            pkey = self._read_provider_secret(p["id"], base_url=p["base_url"], legacy_base_url=legacy_url)
            is_lb = is_loopback_base_url(p["base_url"])
            has_k = bool(pkey)
            providers_out.append(
                {
                    "id": p["id"],
                    "name": p["name"],
                    "api_format": p["api_format"],
                    "base_url": p["base_url"],
                    "models": list(p.get("models") or []),
                    "default_model": str(p.get("default_model") or ""),
                    "model_capabilities": dict(p.get("model_capabilities") or {}),
                    "has_api_key": has_k,
                    "is_loopback": is_lb,
                    "ready": bool(p["base_url"] and (has_k or is_lb)),
                }
            )

        with self._lock:
            sess = self._get_active_session_locked()
            active_sid = str(sess["id"])
            msgs_copy = self._scrub_obj(list(sess["messages"][-MAX_AGENT_MESSAGES:]), secrets_tuple)
            pending_copy = self._scrub_obj(dict(self._pending_confirm), secrets_tuple) if self._pending_confirm else None
            busy = bool(self._running)
            turn_id = str(self._active_turn_id if busy else "")
        sessions_summary = self._build_sessions_summary()
        ctx_stats = self._compute_context_stats(msgs_copy)

        cfg_out = {
            "agent_driver": str(prefs.get("agent_driver") or "native"),
            "agent_brain_source": str(prefs.get("agent_brain_source") or "auto"),
            "agent_api_format": str(prefs.get("agent_api_format") or "openai"),
            "agent_api_base_url": str(prefs.get("agent_api_base_url") or ""),
            "agent_model": str(prefs.get("agent_model") or ""),
            "agent_reasoning_effort": str(brain.get("reasoning_effort") or prefs.get("agent_reasoning_effort") or "auto"),
            "agent_auto_approve_billing": bool(prefs.get("agent_auto_approve_billing", False)),
            "agent_active_provider_id": active_prov_id,
            "providers": providers_out,
            "has_api_key": bool(brain.get("has_external_key", False)),
        }
        return self._scrub_obj(
            {
                "ok": True,
                "busy": busy,
                "turn_id": turn_id,
                "active_session_id": active_sid,
                "sessions": sessions_summary,
                "context_stats": ctx_stats,
                "config": cfg_out,
                "brain": brain,
                "messages": msgs_copy,
                "pending_confirm": pending_copy,
            },
            secrets_tuple,
        )

    @staticmethod
    def _extract_pref_updates_and_key(cfg: dict[str, Any]) -> tuple[dict[str, Any], bool, str | None]:
        """提取通用偏好更新项、清密标记与显式 API Key（自动登记临时脱敏）。"""
        pref_updates: dict[str, Any] = {}
        for target_k, aliases in _PREF_KEY_ALIASES:
            for ak in aliases:
                if ak in cfg and cfg[ak] is not None:
                    pref_updates[target_k] = cfg[ak]
                    break
        if "agent_reasoning_effort" in pref_updates:
            pref_updates["agent_reasoning_effort"] = normalize_reasoning_effort_value(
                pref_updates["agent_reasoning_effort"],
                default="auto",
            )
        clear_flag = bool(cfg.get("clear_api_key", False))
        new_key_raw: str | None = None
        for kk in ("api_key", "agent_api_key"):
            if kk in cfg and cfg[kk] is not None:
                new_key_raw = str(cfg[kk]).strip()
                break
        if new_key_raw:
            register_ephemeral_secret(new_key_raw)
        return pref_updates, clear_flag, new_key_raw

    def _normalize_upsert_provider_payload(
        self,
        up: dict[str, Any],
        new_key_raw: str | None,
        clear_flag: bool,
    ) -> tuple[dict[str, Any] | None, str | None]:
        """校验并规范化命名供应商新增/编辑载荷，无效时返回错误文案。"""
        raw_pid = str(up.get("id") or "").strip()
        is_from_legacy = raw_pid == "__legacy__"
        pid = f"prov_{uuid.uuid4().hex[:8]}" if (not raw_pid or is_from_legacy) else raw_pid[:48]
        up_fmt = str(up.get("api_format") or "openai").strip().lower()
        if up_fmt not in SUPPORTED_API_FORMATS:
            up_fmt = "openai"
        up_url = str(up.get("base_url") or "").strip().rstrip("/")
        if not up_url:
            return None, "请填写供应商 Base URL"
        up_name = str(up.get("name") or "").strip()[:64] or self._infer_legacy_provider_name(up_url)
        dedup_models = list(dict.fromkeys(_parse_str_list(up.get("models"))))[:200]
        def_model = str(up.get("default_model") or "").strip()
        if not def_model and dedup_models:
            def_model = dedup_models[0]
        elif def_model and def_model not in dedup_models:
            dedup_models.insert(0, def_model)
        up_key = str(up.get("api_key") if up.get("api_key") is not None else (new_key_raw or "")).strip()
        if up_key:
            register_ephemeral_secret(up_key)
        return {
            "pid": pid,
            "is_from_legacy": is_from_legacy,
            "api_format": up_fmt,
            "base_url": up_url,
            "name": up_name,
            "models": dedup_models,
            "default_model": def_model,
            "api_key": up_key,
            "clear_api_key": bool(up.get("clear_api_key", clear_flag)),
            "activate": bool(up.get("activate", True)),
            "switch_brain": bool(up.get("switch_brain", True)),
            "model_capabilities": up.get("model_capabilities"),
        }, None

    def _try_vault_op(
        self,
        fn: Callable[[], None],
        err_prefix: str,
        extra_secrets: tuple[str, ...] = (),
    ) -> str | None:
        """执行单次 VaultStore 变更并捕获异常返回脱敏错误文案。"""
        try:
            fn()
            return None
        except Exception as exc:
            return f"{err_prefix}: {redact_secrets(str(exc), extra_secrets)}"

    def _sync_upsert_provider_vault(
        self,
        *,
        vault: VaultStore,
        colab_py: Path | None,
        norm_up: dict[str, Any],
        prev_item: dict[str, Any] | None,
        active_id: str,
        legacy_url: str,
    ) -> str | None:
        """同步命名供应商与旧单键的 VaultStore 源站硬绑定凭据；失败返回脱敏错误消息。"""
        pid = norm_up["pid"]
        up_url = norm_up["base_url"]
        up_key = norm_up["api_key"]
        up_clear = norm_up["clear_api_key"]
        is_from_legacy = norm_up["is_from_legacy"]
        will_activate = norm_up["activate"]

        prev_origin = url_origin_key(str((prev_item or {}).get("base_url") or "").strip().rstrip("/"))
        up_origin = url_origin_key(up_url)
        leg_origin = url_origin_key(legacy_url)
        origin_changed = prev_item is not None and prev_origin != up_origin
        same_prev_leg = bool(prev_origin and prev_origin == leg_origin)
        same_up_leg = bool(up_origin and up_origin == leg_origin)
        touches_active = active_id == pid or will_activate

        # 若从旧版单配置平滑升级且用户未重填密钥，同源下自动继承迁移原 Vault 密钥
        if not up_key and not up_clear and is_from_legacy and same_up_leg:
            up_key = self._read_provider_secret("__legacy__", base_url=up_url, legacy_base_url=legacy_url) or ""

        vkey = agent_provider_vault_key(pid)
        if up_clear:
            clear_leg = is_from_legacy or touches_active or same_prev_leg or same_up_leg
            return self._try_vault_op(
                lambda: (
                    self._delete_vault_secret_with_origin(vault, vkey, colab_py),
                    self._delete_vault_secret_with_origin(vault, AGENT_VAULT_API_KEY, colab_py) if clear_leg else None,
                ),
                "清除供应商密钥失败",
            )
        if up_key:
            clear_leg = touches_active or is_from_legacy
            return self._try_vault_op(
                lambda: (
                    self._write_vault_secret_with_origin(vault, vkey, up_key, up_url, colab_py),
                    self._delete_vault_secret_with_origin(vault, AGENT_VAULT_API_KEY, colab_py) if clear_leg else None,
                ),
                "写入凭据保险箱失败",
                (up_key,),
            )
        if origin_changed:
            clear_leg = touches_active or same_prev_leg
            return self._try_vault_op(
                lambda: (
                    self._delete_vault_secret_with_origin(vault, vkey, colab_py),
                    self._delete_vault_secret_with_origin(vault, AGENT_VAULT_API_KEY, colab_py) if clear_leg else None,
                ),
                "清除换源旧密钥失败",
            )
        if will_activate and (is_from_legacy or leg_origin != up_origin):
            return self._try_vault_op(
                lambda: self._delete_vault_secret_with_origin(vault, AGENT_VAULT_API_KEY, colab_py),
                "清理旧源站单键失败",
            )
        return None

    def _save_upsert_provider(
        self,
        up_raw: dict[str, Any],
        saved_provs: list[dict[str, Any]],
        active_id: str,
        legacy_url: str,
        pref_updates: dict[str, Any],
        new_key_raw: str | None,
        clear_flag: bool,
        vault: VaultStore,
        colab_py: Path | None,
    ) -> tuple[str | None, str | None]:
        """执行命名供应商新增/编辑分支，返回 `(toast_msg, error_msg)`。"""
        norm_up, err_msg = self._normalize_upsert_provider_payload(up_raw, new_key_raw, clear_flag)
        if err_msg or norm_up is None:
            return None, err_msg

        pid = norm_up["pid"]
        prev_item = next((p for p in saved_provs if p["id"] == pid), None)
        vault_err = self._sync_upsert_provider_vault(
            vault=vault,
            colab_py=colab_py,
            norm_up=norm_up,
            prev_item=prev_item,
            active_id=active_id,
            legacy_url=legacy_url,
        )
        if vault_err:
            return None, vault_err

        merged_caps = dict((prev_item or {}).get("model_capabilities") or {})
        if prev_item is not None and prev_item.get("base_url") != norm_up["base_url"]:
            merged_caps = {}
        incoming_caps = sanitize_model_capabilities(norm_up["model_capabilities"])
        if incoming_caps:
            merged_caps.update(incoming_caps)

        entry = {
            "id": pid,
            "name": norm_up["name"],
            "api_format": norm_up["api_format"],
            "base_url": norm_up["base_url"],
            "models": norm_up["models"],
            "default_model": norm_up["default_model"],
            "model_capabilities": merged_caps,
        }
        replaced = False
        next_provs: list[dict[str, Any]] = []
        for p in saved_provs:
            if p["id"] == pid:
                next_provs.append(entry)
                replaced = True
            else:
                next_provs.append(p)
        if not replaced:
            next_provs.append(entry)

        pref_updates["agent_providers"] = next_provs
        if norm_up["activate"]:
            pref_updates["agent_active_provider_id"] = pid
            pref_updates["agent_api_format"] = norm_up["api_format"]
            pref_updates["agent_api_base_url"] = norm_up["base_url"]
            if norm_up["default_model"]:
                pref_updates["agent_model"] = norm_up["default_model"]
            elif not norm_up["models"]:
                pref_updates["agent_model"] = ""
            if "agent_brain_source" not in pref_updates and norm_up["switch_brain"]:
                pref_updates["agent_brain_source"] = "external_api"
        self._b.update_prefs(pref_updates)
        return f"已保存供应商「{norm_up['name']}」", None

    def _save_delete_provider(
        self,
        del_id_raw: Any,
        saved_provs: list[dict[str, Any]],
        active_id: str,
        pref_updates: dict[str, Any],
        vault: VaultStore,
        colab_py: Path | None,
    ) -> str:
        """执行删除指定供应商及其凭据分支并返回提示文案。"""
        del_id = str(del_id_raw or "").strip()
        if del_id == "__legacy__":
            with suppress_logged("agent", throttled_key="delete_legacy_vault_key", interval_sec=30.0):
                self._delete_vault_secret_with_origin(vault, AGENT_VAULT_API_KEY, colab_py)
            pref_updates.update({"agent_active_provider_id": "", "agent_api_base_url": "", "agent_model": ""})
        else:
            next_provs = [p for p in saved_provs if p["id"] != del_id]
            with suppress_logged("agent", throttled_key="delete_provider_vault_key", interval_sec=30.0):
                self._delete_vault_secret_with_origin(vault, agent_provider_vault_key(del_id), colab_py)
                if active_id == del_id:
                    self._delete_vault_secret_with_origin(vault, AGENT_VAULT_API_KEY, colab_py)
            pref_updates["agent_providers"] = next_provs
            if active_id == del_id:
                if next_provs:
                    fb = next_provs[0]
                    pref_updates["agent_active_provider_id"] = fb["id"]
                    pref_updates["agent_api_format"] = fb["api_format"]
                    pref_updates["agent_api_base_url"] = fb["base_url"]
                    pref_updates["agent_model"] = fb.get("default_model") or (fb.get("models") or [""])[0]
                else:
                    pref_updates["agent_active_provider_id"] = ""
                    pref_updates["agent_api_base_url"] = ""
                    pref_updates["agent_model"] = ""
        self._b.update_prefs(pref_updates)
        return "已移除该模型供应商"

    def _save_select_provider_model(
        self,
        sel_raw: dict[str, Any],
        saved_provs: list[dict[str, Any]],
        pref_updates: dict[str, Any],
    ) -> tuple[str | None, str | None]:
        """执行一键切换激活供应商与模型分支，返回 `(toast_msg, error_msg)`。"""
        pid = str(sel_raw.get("provider_id") or "").strip()
        sel_model = str(sel_raw.get("model") or "").strip()
        if pid == "__legacy__" or not pid:
            if sel_model:
                pref_updates["agent_model"] = sel_model
            pref_updates.setdefault("agent_brain_source", "external_api")
        else:
            target = next((p for p in saved_provs if p["id"] == pid), None)
            if target is None:
                return None, "未找到目标模型供应商"
            models = list(target.get("models") or [])
            if sel_model:
                if sel_model not in models:
                    models.append(sel_model)
                target["default_model"] = sel_model
            else:
                sel_model = str(target.get("default_model") or (models[0] if models else ""))
            target["models"] = models
            pref_updates["agent_providers"] = saved_provs
            pref_updates["agent_active_provider_id"] = pid
            pref_updates["agent_api_format"] = target["api_format"]
            pref_updates["agent_api_base_url"] = target["base_url"]
            pref_updates["agent_model"] = sel_model
            pref_updates.setdefault("agent_brain_source", "external_api")
        self._b.update_prefs(pref_updates)
        return (f"已切换至 {sel_model}" if sel_model else "已切换模型供应商"), None

    def _sync_legacy_vault_key(
        self,
        *,
        vault: VaultStore,
        colab_py: Path | None,
        clear_flag: bool,
        new_key_raw: str | None,
        url_origin_changed: bool,
        eff_active_id: str,
        target_base_url: str,
        eff_prov_url: str,
    ) -> str | None:
        """同步传统单表单路径下的 VaultStore 密钥与源站绑定；失败返回脱敏错误消息。"""
        if clear_flag:
            try:
                self._delete_vault_secret_with_origin(vault, AGENT_VAULT_API_KEY, colab_py)
                if eff_active_id and eff_active_id != "__legacy__":
                    self._delete_vault_secret_with_origin(vault, agent_provider_vault_key(eff_active_id), colab_py)
            except Exception as exc:
                return f"清除 API Key 失败: {redact_secrets(str(exc))}"
            return None
        if new_key_raw:
            try:
                self._write_vault_secret_with_origin(vault, AGENT_VAULT_API_KEY, new_key_raw, target_base_url, colab_py)
                if eff_active_id and eff_active_id != "__legacy__":
                    self._write_vault_secret_with_origin(
                        vault, agent_provider_vault_key(eff_active_id), new_key_raw, eff_prov_url, colab_py
                    )
            except Exception as exc:
                return f"写入凭据保险箱失败: {redact_secrets(str(exc), (new_key_raw,))}"
            return None
        if url_origin_changed:
            try:
                # 更换域名/端口且未提供新密钥时，立即销毁旧单键及其源站绑定（保留已解绑的命名供应商原密钥）
                self._delete_vault_secret_with_origin(vault, AGENT_VAULT_API_KEY, colab_py)
            except Exception as exc:
                return f"清除换源旧密钥失败: {redact_secrets(str(exc))}"
        return None

    def _save_legacy_agent_prefs(
        self,
        cur_prefs: dict[str, Any],
        saved_provs: list[dict[str, Any]],
        active_id: str,
        legacy_url: str,
        pref_updates: dict[str, Any],
        clear_flag: bool,
        new_key_raw: str | None,
        vault: VaultStore,
        colab_py: Path | None,
    ) -> tuple[str | None, str | None]:
        """执行传统单表单/开关更新路径，返回 `(toast_msg, error_msg)`。"""
        url_origin_changed = False
        if "agent_api_base_url" in pref_updates:
            new_u = str(pref_updates["agent_api_base_url"] or "").strip().rstrip("/")
            if legacy_url and url_origin_key(new_u) != url_origin_key(legacy_url):
                url_origin_changed = True
            if active_id:
                cur_act = next((p for p in saved_provs if p["id"] == active_id), None)
                if cur_act is not None and new_u != cur_act["base_url"]:
                    pref_updates["agent_active_provider_id"] = ""

        toast_msg = "已保存副驾配置"
        if pref_updates:
            self._b.update_prefs(pref_updates)
            if list(pref_updates.keys()) == ["agent_reasoning_effort"] and not clear_flag and not new_key_raw:
                eff_val = normalize_reasoning_effort_value(pref_updates["agent_reasoning_effort"], default="auto")
                toast_msg = f"思考强度已设为「{_REASONING_EFFORT_LABELS.get(eff_val, eff_val)}」"

        eff_active_id = str(pref_updates.get("agent_active_provider_id", active_id) or "").strip()
        target_base_url = str(
            pref_updates["agent_api_base_url"]
            if "agent_api_base_url" in pref_updates
            else cur_prefs.get("agent_api_base_url") or ""
        ).strip().rstrip("/")
        eff_act_prov = next((p for p in saved_provs if p["id"] == eff_active_id), None) if eff_active_id else None
        eff_prov_url = str((eff_act_prov or {}).get("base_url") or target_base_url).strip().rstrip("/")

        vault_err = self._sync_legacy_vault_key(
            vault=vault,
            colab_py=colab_py,
            clear_flag=clear_flag,
            new_key_raw=new_key_raw,
            url_origin_changed=url_origin_changed,
            eff_active_id=eff_active_id,
            target_base_url=target_base_url,
            eff_prov_url=eff_prov_url,
        )
        if vault_err:
            return None, vault_err
        return toast_msg, None

    def save_agent_config(self, config: dict[str, Any] | None = None) -> dict[str, Any]:
        """保存副驾引擎、多供应商模型目录与路由偏好；API Key 仅写入系统 VaultStore 绝不落盘。"""
        cfg = dict(config) if isinstance(config, dict) else {}
        vault = VaultStore()
        colab_py = self._ctx.get_colab_py()
        cur_prefs = self._get_merged_prefs()
        pref_updates, clear_flag, new_key_raw = self._extract_pref_updates_and_key(cfg)

        saved_provs = sanitize_agent_providers(cur_prefs.get("agent_providers"))
        active_id = str(cur_prefs.get("agent_active_provider_id") or "").strip()
        legacy_url = str(cur_prefs.get("agent_api_base_url") or "").strip().rstrip("/")

        err_msg: str | None = None
        if isinstance(cfg.get("upsert_provider"), dict):
            toast_msg, err_msg = self._save_upsert_provider(
                dict(cfg["upsert_provider"]),
                saved_provs,
                active_id,
                legacy_url,
                pref_updates,
                new_key_raw,
                clear_flag,
                vault,
                colab_py,
            )
        elif cfg.get("delete_provider_id") is not None:
            toast_msg = self._save_delete_provider(
                cfg.get("delete_provider_id"), saved_provs, active_id, pref_updates, vault, colab_py
            )
        elif isinstance(cfg.get("select_provider_model"), dict):
            toast_msg, err_msg = self._save_select_provider_model(
                dict(cfg["select_provider_model"]), saved_provs, pref_updates
            )
        else:
            toast_msg, err_msg = self._save_legacy_agent_prefs(
                cur_prefs, saved_provs, active_id, legacy_url, pref_updates, clear_flag, new_key_raw, vault, colab_py
            )

        if err_msg:
            return {"ok": False, "message": err_msg}

        state_out = self.get_agent_state()
        self._emit_agent_event({"phase": "config_updated", "config": state_out["config"], "brain": state_out["brain"]})
        return {
            "ok": True,
            "message": toast_msg or "已保存副驾配置",
            "config": state_out["config"],
            "brain": state_out["brain"],
            "state": state_out,
        }

    def _resolve_probe_credentials(self, cfg: dict[str, Any]) -> tuple[str, str, str, tuple[str, ...]]:
        """解析模型探测请求的 URL、协议与同源受保护 API Key（防跨站点泄露密钥）。"""
        prefs = self._get_merged_prefs()
        req_url = str(cfg.get("base_url") or cfg.get("api_base_url") or prefs.get("agent_api_base_url") or "").strip().rstrip("/")
        fmt = str(cfg.get("api_format") or prefs.get("agent_api_format") or "openai").strip().lower()
        if fmt not in SUPPORTED_API_FORMATS:
            fmt = "openai"

        resolved_key = ""
        for kk in ("api_key", "agent_api_key"):
            if cfg.get(kk):
                resolved_key = str(cfg[kk]).strip()
                break

        if not resolved_key and req_url:
            target_origin = url_origin_key(req_url)
            req_pid = str(cfg.get("provider_id") or prefs.get("agent_active_provider_id") or "").strip()
            providers = sanitize_agent_providers(prefs.get("agent_providers"))
            legacy_url = str(prefs.get("agent_api_base_url") or "").strip().rstrip("/")

            # 仅当目标 URL 与对应已存供应商同源时，才回退使用 Vault 密钥，防止跨域名外泄
            candidate_ids: list[str] = []
            for p in providers:
                if p["id"] == req_pid and url_origin_key(p["base_url"]) == target_origin:
                    candidate_ids.insert(0, p["id"])
                elif url_origin_key(p["base_url"]) == target_origin:
                    candidate_ids.append(p["id"])
            for cid in candidate_ids:
                k = self._read_provider_secret(cid, base_url=req_url, legacy_base_url=legacy_url)
                if k:
                    resolved_key = k
                    break
            if not resolved_key and target_origin and target_origin == url_origin_key(legacy_url):
                resolved_key = self._read_provider_secret("__legacy__", base_url=req_url, legacy_base_url=legacy_url)

        secrets_tuple = self._collect_known_secrets(resolved_key)
        return req_url, fmt, resolved_key, secrets_tuple

    def fetch_agent_models(self, config: dict[str, Any] | None = None) -> dict[str, Any]:
        """向指定 Base URL（如 Magpie 本机网关 http://127.0.0.1:3425/v1）实时获取可用模型清单并同步思考能力缓存。"""
        cfg = dict(config) if isinstance(config, dict) else {}
        base_url, fmt, api_key, secrets_tuple = self._resolve_probe_credentials(cfg)
        res = fetch_provider_models(base_url, api_key=api_key, api_format=fmt)
        caps = sanitize_model_capabilities(res.get("model_capabilities"))
        if res.get("ok") and caps:
            target_pid = str(cfg.get("provider_id") or "").strip()
            norm_url = str(base_url or "").strip().rstrip("/")
            saved_provs = sanitize_agent_providers(self._get_merged_prefs().get("agent_providers"))
            changed = False
            next_provs: list[dict[str, Any]] = []
            for p in saved_provs:
                same_url = not norm_url or p.get("base_url") == norm_url
                matched = same_url and (
                    (bool(target_pid) and p["id"] == target_pid) or (not target_pid and bool(norm_url))
                )
                if matched:
                    old_caps = dict(p.get("model_capabilities") or {})
                    new_caps = dict(old_caps)
                    new_caps.update(caps)
                    if new_caps != old_caps:
                        p = dict(p)
                        p["model_capabilities"] = new_caps
                        changed = True
                next_provs.append(p)
            if changed:
                self._b.update_prefs({"agent_providers": next_provs})
        return self._scrub_obj(res, secrets_tuple)

    def probe_agent_provider(self, config: dict[str, Any] | None = None) -> dict[str, Any]:
        """执行网关协议自动识别（detect_protocol）或所选模型并发连通性测速（test_models）。"""
        cfg = dict(config) if isinstance(config, dict) else {}
        mode = str(cfg.get("mode") or "test_models").strip().lower()
        base_url, fmt, api_key, secrets_tuple = self._resolve_probe_credentials(cfg)
        if mode == "detect_protocol":
            hint_model = str(cfg.get("probe_model") or cfg.get("model") or "").strip()
            res = detect_provider_protocol(base_url, api_key=api_key, hint_model=hint_model)
            return self._scrub_obj(res, secrets_tuple)

        models = _parse_str_list(cfg.get("models"))
        if not models and cfg.get("probe_model"):
            models = [str(cfg["probe_model"]).strip()]
        res = probe_provider_models(base_url, models, api_key=api_key, api_format=fmt)
        return self._scrub_obj(res, secrets_tuple)

    # ------------------------------------------------------------------
    # 对话轮次编排与流式后台执行
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_context_tag(cfg: dict[str, Any], ctx_extra: Any) -> str:
        """从请求显式标签或上下文字典中提取展示用的简短上下文标签。"""
        ctx_tag_val = str(cfg.get("ctx_tag") or "").strip()[:120]
        if ctx_tag_val or not isinstance(ctx_extra, dict) or not ctx_extra:
            return ctx_tag_val
        for key in ("rel_path", "source", "name", "view"):
            raw_ref = str(ctx_extra.get(key) or "").strip()
            if raw_ref:
                return raw_ref.rsplit("/", 1)[-1][:120]
        return ""

    def _attach_extra_context(
        self,
        history_for_driver: list[dict[str, Any]],
        ctx_extra: Any,
        secrets_tuple: tuple[str, ...],
    ) -> None:
        """将脱敏后的附加上下文参考拼入本轮最后一条用户消息。"""
        if not history_for_driver:
            return
        if isinstance(ctx_extra, dict) and ctx_extra:
            ctx_txt = json.dumps(self._scrub_obj(ctx_extra, secrets_tuple), ensure_ascii=False)
            history_for_driver[-1]["content"] = f"{history_for_driver[-1]['content']}\n\n[上下文参考] {ctx_txt}"
        elif isinstance(ctx_extra, str) and ctx_extra.strip():
            history_for_driver[-1]["content"] = f"{history_for_driver[-1]['content']}\n\n[上下文参考] {ctx_extra.strip()}"

    def _prepare_turn_context(
        self,
        cfg: dict[str, Any],
        user_text: str,
        runtime_cfg: dict[str, Any],
        secrets_tuple: tuple[str, ...],
    ) -> dict[str, Any]:
        """构造并登记新一轮对话的思考强度、用户消息、上下文参考与取消令牌。"""
        raw_override = cfg.get("reasoning_effort") or cfg.get("agent_reasoning_effort")
        norm_override = normalize_reasoning_effort_value(raw_override, default="") if raw_override is not None else ""
        use_effort = (
            self._clamp_reasoning_effort(
                norm_override,
                reasoning_supported=bool(runtime_cfg.get("reasoning_supported", True)),
                supported_levels=list(runtime_cfg.get("supported_reasoning_efforts") or []),
            )
            if norm_override
            else str(runtime_cfg.get("reasoning_effort") or "auto")
        )
        turn_id = f"turn_{uuid.uuid4().hex[:10]}"
        cancel_token = CancelToken()
        ctx_extra = cfg.get("context")
        ctx_tag_val = self._extract_context_tag(cfg, ctx_extra)

        user_entry: dict[str, Any] = {
            "id": f"msg_{uuid.uuid4().hex[:10]}",
            "role": "user",
            "content": user_text,
            "ts": int(time.time()),
        }
        if ctx_tag_val:
            user_entry["ctx_tag"] = ctx_tag_val
        user_msg = self._append_message(user_entry, secrets_tuple)

        with self._lock:
            sess_cur = self._get_active_session_locked()
            history_for_driver = [
                {"role": m["role"], "content": m["content"]}
                for m in sess_cur["messages"][-MAX_AGENT_MESSAGES:]
                if m.get("role") in ("user", "assistant") and str(m.get("content") or "").strip()
            ]
            self._running = True
            self._active_turn_id = turn_id
            self._cancel_token = cancel_token

        self._attach_extra_context(history_for_driver, ctx_extra, secrets_tuple)
        return {
            "turn_id": turn_id,
            "cancel_token": cancel_token,
            "use_effort": use_effort,
            "user_msg": user_msg,
            "history_for_driver": history_for_driver,
        }

    def _handle_driver_event(
        self,
        evt: dict[str, Any],
        *,
        turn_id: str,
        secrets_tuple: tuple[str, ...],
        text_chunks: list[str],
        reasoning_chunks: list[str],
        tool_records: list[dict[str, Any]],
    ) -> None:
        """处理底层推理引擎产生的增量文本、思考过程与工具起止事件并实时广播。"""
        if not isinstance(evt, dict):
            return
        etype = str(evt.get("type") or "").strip()
        if etype == "reasoning_delta":
            r_chunk = str(evt.get("text") or "")
            if r_chunk:
                reasoning_chunks.append(r_chunk)
                self._emit_agent_event({"phase": "reasoning_delta", "turn_id": turn_id, "text": r_chunk}, secrets_tuple)
            return
        if etype == "delta":
            chunk = str(evt.get("text") or "")
            if chunk:
                text_chunks.append(chunk)
                self._emit_agent_event({"phase": "delta", "turn_id": turn_id, "text": chunk}, secrets_tuple)
            return
        if etype == "tool_start":
            rec = {
                "name": str(evt.get("name") or ""),
                "args": evt.get("args") if isinstance(evt.get("args"), dict) else {},
                "tier": str(evt.get("tier") or "exec"),
                "status": "running",
            }
            tool_records.append(rec)
            self._emit_agent_event(
                {
                    "phase": "tool_start",
                    "turn_id": turn_id,
                    "name": rec["name"],
                    "args": rec["args"],
                    "tier": rec["tier"],
                },
                secrets_tuple,
            )
            return
        if etype == "tool_end":
            tname = str(evt.get("name") or "")
            ok_flag = bool(evt.get("ok", True))
            summary = str(evt.get("summary") or "")
            elapsed_ms = int(evt.get("elapsed_ms") or 0)
            for rec in reversed(tool_records):
                if rec.get("name") == tname and rec.get("status") == "running":
                    rec["status"] = "ok" if ok_flag else "error"
                    rec["summary"] = summary
                    rec["elapsed_ms"] = elapsed_ms
                    break
            self._emit_agent_event(
                {
                    "phase": "tool_end",
                    "turn_id": turn_id,
                    "name": tname,
                    "ok": ok_flag,
                    "summary": summary,
                    "elapsed_ms": elapsed_ms,
                },
                secrets_tuple,
            )

    def _run_driver_for_turn(
        self,
        *,
        runtime_cfg: dict[str, Any],
        history_for_driver: list[dict[str, Any]],
        use_effort: str,
        on_event: Callable[[dict[str, Any]], None],
        cancel_token: CancelToken,
        secrets_tuple: tuple[str, ...],
    ) -> dict[str, Any]:
        """根据配置实例化并运行 ClaudeCodeDriver 或 NativeLightDriver。"""
        if runtime_cfg["driver"] == "claude_code":
            drv = ClaudeCodeDriver(
                ctl_script=self._ctx.ctl_script,
                model=str(runtime_cfg.get("model") or ""),
                on_event=on_event,
                cancel_token=cancel_token,
                ctl_prefix=self._ctx.ctl_cmd_prefix(unbuffered=True),
                reasoning_effort=use_effort,
            )
            return drv.run(history_for_driver)
        drv = NativeLightDriver(
            base_url=str(runtime_cfg["base_url"]),
            api_key=str(runtime_cfg["api_key"]),
            model=str(runtime_cfg["model"]),
            api_format=str(runtime_cfg["api_format"]),
            messages=history_for_driver,
            tool_specs=AGENT_TOOL_SPECS,
            tool_executor=lambda n, a: self._execute_tool(n, a, cancel_token, secrets_tuple),
            on_event=on_event,
            cancel_token=cancel_token,
            reasoning_effort=use_effort,
        )
        return drv.run()

    def _finalize_turn_outcome(
        self,
        *,
        turn_id: str,
        drv_res: dict[str, Any],
        cancel_token: CancelToken,
        text_chunks: list[str],
        reasoning_chunks: list[str],
        tool_records: list[dict[str, Any]],
        secrets_tuple: tuple[str, ...],
    ) -> None:
        """按取消、成功与失败三种结果落盘助手消息并广播最终轮次事件。"""
        cancelled = bool(drv_res.get("cancelled") or cancel_token.is_cancelled)
        ok_final = bool(drv_res.get("ok")) and not cancelled
        err_txt = str(drv_res.get("error") or "").strip()
        final_text = str(drv_res.get("text") or "".join(text_chunks)).strip()
        final_reasoning = str(drv_res.get("reasoning") or "".join(reasoning_chunks)).strip()

        if cancelled or ok_final:
            stored_text = (final_text or f"（{err_txt or '已停止生成'}）") if cancelled else final_text
            asst_msg = self._append_message(
                {
                    "id": f"msg_{uuid.uuid4().hex[:10]}",
                    "role": "assistant",
                    "content": stored_text,
                    "reasoning": final_reasoning,
                    "ts": int(time.time()),
                    "cancelled": cancelled,
                    "tool_calls": tool_records,
                },
                secrets_tuple,
            )
            done_evt: dict[str, Any] = {
                "phase": "turn_done",
                "turn_id": turn_id,
                "ok": not cancelled,
                "cancelled": cancelled,
                "text": stored_text,
                "reasoning": final_reasoning,
                "message": asst_msg,
            }
            if cancelled:
                done_evt["reason"] = err_txt or "用户已停止生成"
            self._emit_agent_event(done_evt, secrets_tuple)
            return

        err_msg = err_txt or "副驾推理执行失败"
        if final_text or final_reasoning or tool_records:
            self._append_message(
                {
                    "id": f"msg_{uuid.uuid4().hex[:10]}",
                    "role": "assistant",
                    "content": final_text or f"执行出错: {err_msg}",
                    "reasoning": final_reasoning,
                    "ts": int(time.time()),
                    "error": err_msg,
                    "tool_calls": tool_records,
                },
                secrets_tuple,
            )
        self._emit_agent_event(
            {
                "phase": "turn_error",
                "turn_id": turn_id,
                "error": err_msg,
                "text": final_text,
                "reasoning": final_reasoning,
            },
            secrets_tuple,
        )

    def _execute_turn_worker(
        self,
        *,
        turn_id: str,
        runtime_cfg: dict[str, Any],
        history_for_driver: list[dict[str, Any]],
        use_effort: str,
        cancel_token: CancelToken,
        secrets_tuple: tuple[str, ...],
    ) -> None:
        """后台工作线程主函数：驱动执行、异常兜底并在 finally 中原子释放运行态与单飞锁。"""
        tool_records: list[dict[str, Any]] = []
        text_chunks: list[str] = []
        reasoning_chunks: list[str] = []
        try:
            drv_res = self._run_driver_for_turn(
                runtime_cfg=runtime_cfg,
                history_for_driver=history_for_driver,
                use_effort=use_effort,
                on_event=lambda evt: self._handle_driver_event(
                    evt,
                    turn_id=turn_id,
                    secrets_tuple=secrets_tuple,
                    text_chunks=text_chunks,
                    reasoning_chunks=reasoning_chunks,
                    tool_records=tool_records,
                ),
                cancel_token=cancel_token,
                secrets_tuple=secrets_tuple,
            )
            self._finalize_turn_outcome(
                turn_id=turn_id,
                drv_res=drv_res,
                cancel_token=cancel_token,
                text_chunks=text_chunks,
                reasoning_chunks=reasoning_chunks,
                tool_records=tool_records,
                secrets_tuple=secrets_tuple,
            )
        except Exception as exc:
            safe_err = redact_secrets(str(exc) or type(exc).__name__, secrets_tuple)
            if cancel_token.is_cancelled:
                self._emit_agent_event(
                    {
                        "phase": "turn_done",
                        "turn_id": turn_id,
                        "ok": False,
                        "cancelled": True,
                        "reason": "操作已手动取消",
                        "text": "".join(text_chunks),
                    },
                    secrets_tuple,
                )
            else:
                self._b._diag.error("agent", f"副驾执行异常: {safe_err}")
                self._emit_agent_event(
                    {
                        "phase": "turn_error",
                        "turn_id": turn_id,
                        "error": safe_err,
                        "text": "".join(text_chunks),
                    },
                    secrets_tuple,
                )
        finally:
            with self._lock:
                if self._confirm_event is not None:
                    self._confirm_result = False
                    self._confirm_event.set()
                self._pending_confirm = None
                self._confirm_event = None
                self._confirm_result = None
                self._running = False
                self._active_turn_id = ""
                self._cancel_token = None
                self._turn_thread = None
            self._busy_lock.release()

    def send_agent_message(self, config: dict[str, Any] | None = None) -> dict[str, Any]:
        """接收用户消息并启动同进程流式推理与工具调用后台轮次。"""
        cfg = dict(config) if isinstance(config, dict) else {}
        user_text = str(cfg.get("text") or cfg.get("message") or cfg.get("prompt") or "").strip()
        if not user_text:
            return {"ok": False, "message": "请输入指令内容"}

        if not self._busy_lock.acquire(blocking=False):
            return {"ok": False, "busy": True, "message": "副驾正在执行上一轮任务，请等待完成或先点击停止"}

        try:
            public_brain, runtime_cfg = self._resolve_brain_runtime(probe_network=True)
            if not runtime_cfg["ready"]:
                self._busy_lock.release()
                return {
                    "ok": False,
                    "needs_config": True,
                    "brain": public_brain,
                    "message": runtime_cfg["reason"] or "副驾大脑尚未就绪，请先开机或配置外部 API",
                }

            secrets_tuple = self._collect_known_secrets(
                str(runtime_cfg.get("api_key") or ""),
                str(runtime_cfg.get("ext_api_key") or ""),
                str(runtime_cfg.get("colab_api_key") or ""),
            )
            turn_ctx = self._prepare_turn_context(cfg, user_text, runtime_cfg, secrets_tuple)
            self._emit_agent_event(
                {
                    "phase": "turn_start",
                    "turn_id": turn_ctx["turn_id"],
                    "brain": public_brain,
                    "user_message": turn_ctx["user_msg"],
                },
                secrets_tuple,
            )
        except Exception:
            with self._lock:
                self._running = False
                self._active_turn_id = ""
                self._cancel_token = None
            self._busy_lock.release()
            raise

        th = threading.Thread(
            target=self._execute_turn_worker,
            kwargs={
                "turn_id": turn_ctx["turn_id"],
                "runtime_cfg": runtime_cfg,
                "history_for_driver": turn_ctx["history_for_driver"],
                "use_effort": turn_ctx["use_effort"],
                "cancel_token": turn_ctx["cancel_token"],
                "secrets_tuple": secrets_tuple,
            },
            name=f"colab-agent-{turn_ctx['turn_id']}",
            daemon=True,
        )
        with self._lock:
            self._turn_thread = th
        th.start()
        return self._scrub_obj(
            {
                "ok": True,
                "started": True,
                "turn_id": turn_ctx["turn_id"],
                "brain": public_brain,
                "user_message": turn_ctx["user_msg"],
            },
            secrets_tuple,
        )

    def resolve_agent_confirm(self, config: dict[str, Any] | None = None) -> dict[str, Any]:
        """响应前端审批卡操作（批准或拒绝待确认的计费/远端执行工具），立即唤醒挂起工作线程。"""
        cfg = dict(config) if isinstance(config, dict) else {}
        confirm_id = str(cfg.get("confirm_id") or cfg.get("id") or "").strip()
        approved = bool(cfg.get("approved", False))

        with self._lock:
            pending = self._pending_confirm
            ev = self._confirm_event
            if pending is None or ev is None:
                return {"ok": False, "message": "当前没有待审批的操作"}
            cur_id = str(pending.get("id") or "")
            cur_tool = str(pending.get("tool") or "")
            if confirm_id and confirm_id != cur_id:
                return {"ok": False, "message": "审批单号不匹配或已过期"}
            self._confirm_result = approved
            ev.set()

        if cur_tool == "exec_remote_colab":
            msg = "已批准执行远端代码" if approved else "已取消远端代码执行"
        else:
            msg = "已批准执行计费操作" if approved else "已取消计费操作"
        return {
            "ok": True,
            "confirm_id": cur_id,
            "approved": approved,
            "message": msg,
        }

    def cancel_agent_turn(self) -> dict[str, Any]:
        """立即中止当前流式生成、终止远端/CLI 子进程树并唤醒任何挂起中的计费审批。"""
        with self._lock:
            tok = self._cancel_token
            ev = self._confirm_event
            had_pending = self._pending_confirm is not None
            was_running = self._running
            if had_pending:
                self._confirm_result = False
        if tok is not None:
            tok.cancel()
        if ev is not None:
            ev.set()
        return {
            "ok": True,
            "was_running": bool(was_running),
            "cancelled_confirm": bool(had_pending),
            "message": "已停止副驾当前生成" if was_running else "当前无正在运行的副驾任务",
        }

    def clear_agent_history(self) -> dict[str, Any]:
        """清空当前会话对话历史并更新本地持久化（运行中需先停止）。"""
        with self._lock:
            if self._running:
                return {"ok": False, "busy": True, "message": "副驾正在运行中，请先停止后再清空对话"}
            sess = self._get_active_session_locked()
            cleared_cnt = len(sess["messages"])
            sess["messages"].clear()
            self._messages = sess["messages"]
            sess["title"] = "新对话"
            sess["updated_ts"] = int(time.time())
            active_sid = str(sess["id"])
        self._save_sessions_to_disk(self._collect_known_secrets())
        self._emit_agent_event({"phase": "history_cleared", "active_session_id": active_sid})
        return {
            "ok": True,
            "cleared": cleared_cnt,
            "active_session_id": active_sid,
            "state": self.get_agent_state(),
            "message": "已清空副驾对话历史",
        }

    # ------------------------------------------------------------------
    # 多会话历史管理动作分解（list / new / switch / delete / rename / clear_all）
    # ------------------------------------------------------------------

    def _session_action_new(self, cfg: dict[str, Any], secrets_tuple: tuple[str, ...]) -> dict[str, Any]:
        """新建空白对话并回收其余未发过消息的空白非活跃会话。"""
        reuse_id = ""
        with self._lock:
            if self._running:
                return {"ok": False, "busy": True, "message": "副驾正在生成回复，请等待完成或先停止"}
            cur = self._get_active_session_locked()
            if len(cur["messages"]) == 0:
                reuse_id = str(cur["id"])
            else:
                self._sessions = [
                    s for s in self._sessions if s["id"] == cur["id"] or len(s.get("messages") or []) > 0
                ]
                fresh = self._new_empty_session(title=str(cfg.get("title") or "新对话"))
                self._sessions.insert(0, fresh)
                self._active_session_id = fresh["id"]
                self._messages = fresh["messages"]
                self._pending_confirm = None
                self._trim_sessions_locked()
                new_id = fresh["id"]
        if reuse_id:
            return {
                "ok": True,
                "created": False,
                "active_session_id": reuse_id,
                "state": self.get_agent_state(),
                "message": "当前已是空白新对话",
            }
        self._save_sessions_to_disk(secrets_tuple)
        self._emit_agent_event({"phase": "session_changed", "active_session_id": new_id}, secrets_tuple)
        return {
            "ok": True,
            "created": True,
            "active_session_id": new_id,
            "state": self.get_agent_state(),
            "message": "已新建副驾对话",
        }

    def _session_action_switch(self, cfg: dict[str, Any], secrets_tuple: tuple[str, ...]) -> dict[str, Any]:
        """切换活跃历史会话，并在离开空白旧会话时自动清理。"""
        req_id = str(cfg.get("session_id") or cfg.get("id") or "").strip()
        if not req_id:
            return {"ok": False, "message": "缺少目标会话 ID"}
        already_title: str | None = None
        with self._lock:
            if self._running:
                return {"ok": False, "busy": True, "message": "副驾正在生成回复，请等待完成或先停止"}
            target = next((s for s in self._sessions if s["id"] == req_id), None)
            if target is None:
                return {"ok": False, "message": "未找到目标历史会话"}
            if req_id == self._active_session_id:
                already_title = str(target.get("title") or "当前会话")
            else:
                prev_id = self._active_session_id
                prev_sess = next((s for s in self._sessions if s["id"] == prev_id), None)
                if prev_sess is not None and len(prev_sess.get("messages") or []) == 0 and len(self._sessions) > 1:
                    self._sessions = [s for s in self._sessions if s["id"] != prev_id]
                self._active_session_id = str(target["id"])
                if not isinstance(target.get("messages"), list):
                    target["messages"] = []
                self._messages = target["messages"]
                self._pending_confirm = None
                target_title = str(target.get("title") or "新对话")
        if already_title is not None:
            return {
                "ok": True,
                "active_session_id": req_id,
                "state": self.get_agent_state(),
                "message": f"已在「{already_title}」",
            }
        self._save_sessions_to_disk(secrets_tuple)
        self._emit_agent_event({"phase": "session_changed", "active_session_id": req_id}, secrets_tuple)
        return {
            "ok": True,
            "active_session_id": req_id,
            "state": self.get_agent_state(),
            "message": f"已切换至「{target_title}」",
        }

    def _session_action_delete(self, cfg: dict[str, Any], secrets_tuple: tuple[str, ...]) -> dict[str, Any]:
        """删除指定历史会话；若删空则自动重建空白会话。"""
        with self._lock:
            req_id = str(cfg.get("session_id") or cfg.get("id") or self._active_session_id).strip()
            deleting_active = req_id == self._active_session_id
            if deleting_active and self._running:
                return {"ok": False, "busy": True, "message": "当前会话正在生成回复，请先停止后再删除"}
            target = next((s for s in self._sessions if s["id"] == req_id), None)
            if target is None:
                return {"ok": False, "message": "未找到要删除的历史会话"}
            self._sessions = [s for s in self._sessions if s["id"] != req_id]
            if not self._sessions:
                fresh = self._new_empty_session()
                self._sessions = [fresh]
                self._active_session_id = fresh["id"]
                self._messages = fresh["messages"]
                if deleting_active:
                    self._pending_confirm = None
            elif deleting_active:
                best = max(self._sessions, key=lambda s: (int(s.get("updated_ts") or 0), int(s.get("created_ts") or 0)))
                self._active_session_id = str(best["id"])
                self._messages = best.setdefault("messages", [])
                self._pending_confirm = None
            new_active = self._active_session_id
        self._save_sessions_to_disk(secrets_tuple)
        self._emit_agent_event({"phase": "session_changed", "active_session_id": new_active}, secrets_tuple)
        return {
            "ok": True,
            "deleted_id": req_id,
            "active_session_id": new_active,
            "state": self.get_agent_state(),
            "message": "已删除历史会话",
        }

    def _session_action_rename(self, cfg: dict[str, Any], secrets_tuple: tuple[str, ...]) -> dict[str, Any]:
        """重命名指定历史会话标题。"""
        new_title = re.sub(r"\s+", " ", str(cfg.get("title") or "")).strip()[:40]
        if not new_title:
            return {"ok": False, "message": "会话标题不能为空"}
        with self._lock:
            req_id = str(cfg.get("session_id") or cfg.get("id") or self._active_session_id).strip()
            target = next((s for s in self._sessions if s["id"] == req_id), None)
            if target is None:
                return {"ok": False, "message": "未找到目标会话"}
            target["title"] = new_title
            target["updated_ts"] = int(time.time())
            cur_active = self._active_session_id
        self._save_sessions_to_disk(secrets_tuple)
        self._emit_agent_event({"phase": "session_changed", "active_session_id": cur_active}, secrets_tuple)
        return {
            "ok": True,
            "active_session_id": cur_active,
            "state": self.get_agent_state(),
            "message": f"已重命名为「{new_title}」",
        }

    def _session_action_clear_all(self, _cfg: dict[str, Any], secrets_tuple: tuple[str, ...]) -> dict[str, Any]:
        """清空全部历史会话并重置为一个全新空白对话。"""
        with self._lock:
            if self._running:
                return {"ok": False, "busy": True, "message": "副驾正在运行中，请先停止后再清空全部会话"}
            fresh = self._new_empty_session()
            self._sessions = [fresh]
            self._active_session_id = fresh["id"]
            self._messages = fresh["messages"]
            self._pending_confirm = None
            new_active = fresh["id"]
        self._save_sessions_to_disk(secrets_tuple)
        self._emit_agent_event({"phase": "session_changed", "active_session_id": new_active}, secrets_tuple)
        return {
            "ok": True,
            "active_session_id": new_active,
            "state": self.get_agent_state(),
            "message": "已清空全部历史会话",
        }

    def manage_agent_session(self, config: dict[str, Any] | None = None) -> dict[str, Any]:
        """轻量多会话历史管理（list / new / switch / delete / rename / clear_all）。"""
        cfg = dict(config) if isinstance(config, dict) else {}
        action = str(cfg.get("action") or "list").strip().lower()
        if action == "list":
            return self.get_agent_state()
        handlers = {
            "new": self._session_action_new,
            "switch": self._session_action_switch,
            "delete": self._session_action_delete,
            "rename": self._session_action_rename,
            "clear_all": self._session_action_clear_all,
        }
        handler = handlers.get(action)
        if handler is None:
            return {"ok": False, "message": f"未知会话操作: {action}"}
        return handler(cfg, self._collect_known_secrets())
