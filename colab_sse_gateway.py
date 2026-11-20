#!/usr/bin/env python3
"""本地零依赖 SSE 保活反向代理网关（127.0.0.1:18080 -> 127.0.0.1:18081）。

针对流式推理 POST 请求（stream=true），在 llama-server 长上下文 Prefill 期间，
按 SSE_HEARTBEAT_SEC 间隔向下游发送合法 SSE 注释帧 `: keepalive\n\n`（Chunked 编码），
防止 Cloudflare 代理因约 100 秒无首字节响应而触发超时断流；
其余请求（/health、/v1/models、非流式、401 鉴权错误等）保持零修改原样透传。
"""
import asyncio
import contextlib
import hmac
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import zlib
from pathlib import Path
from typing import NamedTuple

SSE_GATEWAY_VERSION = 2
LISTEN_HOST = os.environ.get("SSE_GW_HOST", "127.0.0.1")
LISTEN_PORT = int(os.environ.get("SSE_GW_PORT", "18080"))
UPSTREAM_HOST = os.environ.get("SSE_UPSTREAM_HOST", "127.0.0.1")
UPSTREAM_PORT = int(os.environ.get("SSE_UPSTREAM_PORT", "18081"))
HEARTBEAT_SEC = float(os.environ.get("SSE_HEARTBEAT_SEC", "12.0"))
BASE_MODEL = (os.environ.get("SSE_BASE_MODEL") or "Qwen3.8-Flash-Next-abliterated").strip() or "Qwen3.8-Flash-Next-abliterated"
DEFAULT_EFFORT = (os.environ.get("SSE_DEFAULT_EFFORT") or "medium").strip().lower()
if DEFAULT_EFFORT not in {"none", "low", "medium", "high", "xhigh"}:
    DEFAULT_EFFORT = "medium"
MAX_STREAM_INSPECT_BYTES = 128 * 1024 * 1024
MAX_DECOMPRESSED_BYTES = int(os.environ.get("SSE_MAX_DECOMPRESSED_BYTES", str(32 * 1024 * 1024)))
_MAX_MODELS_BODY_BYTES = 16 * 1024 * 1024

STREAM_PATHS = frozenset({
    "/v1/chat/completions", "/chat/completions",
    "/v1/completions", "/completions",
    "/v1/messages", "/messages",
})
MODELS_PATHS = frozenset({"/v1/models", "/models"})
_PROTECTED_EXTRA_PATHS = frozenset({
    "/slots", "/props", "/tokenize", "/detokenize",
    "/embedding", "/embeddings", "/rerank", "/infill", "/__pet_metrics",
})

_KEEPALIVE_PAYLOAD = b": keepalive\n\n"
_KEEPALIVE_CHUNK = f"{len(_KEEPALIVE_PAYLOAD):x}\r\n".encode("ascii") + _KEEPALIVE_PAYLOAD + b"\r\n"

_SUFFIX_TO_EFFORT = (
    ("-nothink", "none"), ("-minimal", "low"), ("-medium", "medium"),
    ("-xhigh", "xhigh"), ("-none", "none"), ("-fast", "low"),
    ("-off", "none"), ("-high", "xhigh"), ("-max", "xhigh"), ("-low", "low"),
)
_SUPPORTED_PARAMS = ["reasoning_effort", "thinking", "enable_thinking", "tools"]
_ALL_REASONING_LEVELS = ["none", "low", "medium", "high", "xhigh"]
_VIRTUAL_REASONING_VARIANTS = (
    ("-nothink", ("none",)),
    ("-low", ("low",)),
    ("-medium", ("medium",)),
    ("-xhigh", ("xhigh",)),
)


class _ParsedHttpRequest(NamedTuple):
    """已解析且通过鉴权的入站 HTTP 请求头上下文。"""

    raw_head: bytes
    start_line: str
    hdrs: list[tuple[str, str]]
    hdr_map: dict[str, str]
    method: str
    path: str
    is_upgrade: bool


class _InspectedRequestBody(NamedTuple):
    """流式推理请求体探测与思考参数规范化结果。"""

    body_bytes: bytes | None
    is_stream: bool
    was_rewritten: bool


def _map_effort_str(val: object) -> str | None:
    """将各类协议的思考强度字符串安全映射为 Jinja 模板合法集合（none/low/medium/xhigh）。"""
    if not isinstance(val, str):
        return None
    s = val.strip().lower()
    if s in {"none", "off", "disable", "disabled", "false", "0", "nothink"}:
        return "none"
    if s in {"minimal", "low", "fast"}:
        return "low"
    if s in {"medium", "med", "balanced", "default"}:
        return "medium"
    if s in {"high", "xhigh", "max", "heavy"}:
        return "xhigh"
    return None


def _extract_item_model_id(item: object, id_keys: tuple[str, ...]) -> str:
    """按优先级提取模型列表条目中的模型标识字段。"""
    if not isinstance(item, dict):
        return ""
    for key in id_keys:
        val = item.get(key)
        if val:
            return str(val)
    return ""


def _find_base_model_item(items: list, id_keys: tuple[str, ...]) -> dict | None:
    """优先匹配 BASE_MODEL 条目，否则回退到首条字典项。"""
    for item in items:
        if isinstance(item, dict) and any(item.get(k) == BASE_MODEL for k in id_keys):
            return item
    return items[0] if isinstance(items[0], dict) else None


def _append_virtual_reasoning_variants(
    items: object,
    *,
    id_keys: tuple[str, ...],
    assign_keys: tuple[str, ...],
    include_supported_params: bool = False,
    update_aliases: bool = False,
) -> None:
    """向模型列表原地注入 reasoning_levels 与分档虚拟模型（自动按已有 ID 去重）。"""
    if not isinstance(items, list) or not items:
        return
    base_item = _find_base_model_item(items, id_keys)
    if base_item is None:
        return

    base_id = _extract_item_model_id(base_item, id_keys) or BASE_MODEL
    base_item["reasoning_levels"] = list(_ALL_REASONING_LEVELS)
    if include_supported_params:
        base_item["supported_parameters"] = list(_SUPPORTED_PARAMS)

    existing_ids = {mid for it in items if (mid := _extract_item_model_id(it, id_keys))}
    for suf, lvls in _VIRTUAL_REASONING_VARIANTS:
        vid = f"{base_id}{suf}"
        if vid in existing_ids:
            continue
        clone = dict(base_item)
        for ak in assign_keys:
            clone[ak] = vid
        if update_aliases and "aliases" in clone:
            clone["aliases"] = [vid]
        clone["reasoning_levels"] = list(lvls)
        if include_supported_params:
            clone["supported_parameters"] = list(_SUPPORTED_PARAMS)
        items.append(clone)
        existing_ids.add(vid)


def _augment_models_catalog(obj: dict) -> dict:
    """为 /v1/models 注入 reasoning_levels 与分档虚拟模型供 Magpie 一键识别。"""
    if not isinstance(obj, dict):
        return obj
    _append_virtual_reasoning_variants(
        obj.get("data"), id_keys=("id",), assign_keys=("id",),
        include_supported_params=True, update_aliases=True,
    )
    _append_virtual_reasoning_variants(
        obj.get("models"), id_keys=("model", "name"), assign_keys=("name", "model"),
    )
    return obj


def _budget_tokens_to_effort(bt: object) -> str | None:
    """将 Anthropic thinking.budget_tokens 映射为网关规范思考档位。"""
    if not isinstance(bt, (int, float)):
        return None
    if bt <= 0:
        return "none"
    if bt <= 1024:
        return "low"
    if bt <= 4096:
        return "medium"
    return "xhigh"


def _extract_thinking_hints(payload: dict, ct_kwargs: dict) -> tuple[bool, bool, str | None]:
    """提取请求中显式思考开关与思考档位提示（兼容 OpenAI/Anthropic/Jinja 字段）。"""
    explicit_disable = False
    explicit_enable = False
    for flag_val in (ct_kwargs.get("enable_thinking"), payload.get("enable_thinking")):
        if isinstance(flag_val, bool):
            explicit_enable = explicit_enable or flag_val
            explicit_disable = explicit_disable or (not flag_val)

    thinking_cfg = payload.get("thinking")
    budget_effort: str | None = None
    if isinstance(thinking_cfg, dict):
        t_type = str(thinking_cfg.get("type") or "").strip().lower()
        if t_type == "disabled":
            explicit_disable = True
        elif t_type == "enabled":
            explicit_enable = True
            budget_effort = _budget_tokens_to_effort(thinking_cfg.get("budget_tokens"))

    reasoning_cfg = payload.get("reasoning")
    output_cfg = payload.get("output_config")
    candidates = (
        ct_kwargs.get("reasoning_effort"),
        payload.get("reasoning_effort"),
        reasoning_cfg.get("effort") if isinstance(reasoning_cfg, dict) else None,
        output_cfg.get("effort") if isinstance(output_cfg, dict) else None,
    )
    explicit_level = next((mapped for raw in candidates if (mapped := _map_effort_str(raw))), None) or budget_effort
    return explicit_disable, explicit_enable, explicit_level


def _resolve_reasoning_effort(
    *, explicit_disable: bool, explicit_enable: bool, explicit_level: str | None, suffix_effort: str | None
) -> str:
    """综合显式开关、请求档位、模型名后缀与网关默认值，解析最终思考强度。"""
    default_mapped = _map_effort_str(DEFAULT_EFFORT) or "medium"
    if explicit_disable or explicit_level == "none":
        return "none"
    if explicit_level in {"low", "medium", "xhigh"}:
        return explicit_level
    if explicit_enable:
        if suffix_effort in {"low", "medium", "xhigh"}:
            return suffix_effort
        return default_mapped if default_mapped in {"low", "medium", "xhigh"} else "medium"
    return suffix_effort if suffix_effort is not None else default_mapped


def _normalize_inference_payload(payload: dict, path: str) -> dict:
    """解析虚拟模型后缀与多协议思考参数，统一规范化为 Jinja 安全参数并防报错。"""
    if not isinstance(payload, dict):
        return payload

    suffix_effort: str | None = None
    model_val = payload.get("model")
    if isinstance(model_val, str):
        lower_val = model_val.strip().lower()
        for suf, eff in _SUFFIX_TO_EFFORT:
            if lower_val.endswith(suf):
                suffix_effort = eff
                payload["model"] = BASE_MODEL
                break

    raw_ct_kwargs = payload.get("chat_template_kwargs")
    ct_kwargs = dict(raw_ct_kwargs) if isinstance(raw_ct_kwargs, dict) else {}
    explicit_disable, explicit_enable, explicit_level = _extract_thinking_hints(payload, ct_kwargs)
    resolved = _resolve_reasoning_effort(
        explicit_disable=explicit_disable,
        explicit_enable=explicit_enable,
        explicit_level=explicit_level,
        suffix_effort=suffix_effort,
    )

    # 清理可能透传进 llama-server Jinja 模板引发异常的顶层非标准字段
    for key in ("reasoning_effort", "enable_thinking", "reasoning", "output_config"):
        payload.pop(key, None)

    if resolved == "none":
        ct_kwargs.pop("reasoning_effort", None)
        ct_kwargs["enable_thinking"] = False
        if path in {"/v1/messages", "/messages"}:
            payload["thinking"] = {"type": "disabled"}
        else:
            payload.pop("thinking", None)
    else:
        ct_kwargs["enable_thinking"] = True
        ct_kwargs["reasoning_effort"] = resolved

    payload["chat_template_kwargs"] = ct_kwargs
    return payload


def _parse_headers(raw_head: bytes) -> tuple[str, list[tuple[str, str]], dict[str, str]]:
    """解析 HTTP 起始行与大小写不敏感头字段映射。"""
    lines = raw_head[:-4].decode("latin-1").split("\r\n")
    start_line = lines[0]
    hdrs: list[tuple[str, str]] = []
    hdr_map: dict[str, str] = {}
    for line in lines[1:]:
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        k_s, v_s = k.strip(), v.strip()
        hdrs.append((k_s, v_s))
        kl = k_s.lower()
        hdr_map[kl] = f"{hdr_map[kl]}, {v_s}" if kl in hdr_map else v_s
    return start_line, hdrs, hdr_map


def _parse_http_status(start_line: str, default: int = 502) -> int:
    """从 HTTP 响应起始行解析状态码，异常格式回退 default。"""
    parts = start_line.split(" ", 2)
    if len(parts) < 2:
        return default
    try:
        return int(parts[1])
    except ValueError:
        return default


async def _read_chunked_body(reader: asyncio.StreamReader, limit: int) -> bytes:
    """解码客户端 Chunked 请求体，限制最大缓冲大小防止内存溢出。"""
    buf = bytearray()
    while True:
        size_line = await reader.readline()
        if not size_line:
            raise EOFError("unexpected EOF reading chunk size")
        chunk_size = int(size_line.split(b";", 1)[0].strip(), 16)
        if chunk_size == 0:
            while (trailer := await reader.readline()) and trailer not in (b"\r\n", b"\n"):
                pass
            break
        if len(buf) + chunk_size > limit:
            raise ValueError("payload_too_large")
        buf.extend(await reader.readexactly(chunk_size))
        if await reader.readexactly(2) != b"\r\n":
            raise ValueError("invalid chunk terminator")
    return bytes(buf)


def _rebuild_headers(
    start_line: str,
    hdrs: list[tuple[str, str]],
    *,
    force_close: bool = True,
    content_length: int | None = None,
    strip_expect: bool = False,
    strip_accept_encoding: bool = False,
    strip_content_encoding: bool = False,
) -> bytes:
    """重组 HTTP 头：统一注入 Connection: close 以保证每次请求均经过网关调度。"""
    drop = {"connection", "keep-alive", "proxy-connection"} if force_close else set()
    if strip_expect:
        drop.add("expect")
    if strip_accept_encoding:
        drop.add("accept-encoding")
    if strip_content_encoding:
        drop.add("content-encoding")
    if content_length is not None:
        drop.update({"transfer-encoding", "content-length"})
    out = [start_line] + [f"{k}: {v}" for k, v in hdrs if k.lower() not in drop]
    if force_close:
        out.append("Connection: close")
    if content_length is not None:
        out.append(f"Content-Length: {content_length}")
    return ("\r\n".join(out) + "\r\n\r\n").encode("latin-1")


async def _pump_stream(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    *,
    as_chunked: bool = False,
    emit_last_chunk: bool = False,
) -> None:
    """将输入流单向转发至输出流，可选封装为 HTTP/1.1 Chunked 帧。"""
    while data := await reader.read(65536):
        if as_chunked:
            writer.write(f"{len(data):x}\r\n".encode("ascii") + data + b"\r\n")
        else:
            writer.write(data)
        await writer.drain()
    if as_chunked and emit_last_chunk:
        writer.write(b"0\r\n\r\n")
        await writer.drain()


async def _wait_client_disconnect(reader: asyncio.StreamReader) -> None:
    """监听客户端关闭信号（read 返回 EOF 即视为断开）。"""
    while await reader.read(1024):
        pass


def _parse_allowed_cors_origins() -> frozenset[str]:
    """解析环境变量 `SSE_ALLOWED_ORIGINS` 中的逗号分隔精确 Origin 白名单（过滤非法项与通配符）。"""
    raw = str(os.environ.get("SSE_ALLOWED_ORIGINS") or "").strip()
    if not raw:
        return frozenset()
    allowed: set[str] = set()
    for item in raw.split(","):
        cand = item.strip().rstrip("/")
        if cand and cand != "*" and not any(ch in cand for ch in ("\r", "\n", " ", "\t")):
            allowed.add(cand)
    return frozenset(allowed)


def _cors_header_block(hdr_map: dict[str, str]) -> str:
    """生成安全收紧后的 CORS 响应头片段（严禁无条件反射未知 Origin 或附带 Credentials）。"""
    raw_origin = str(hdr_map.get("origin") or "").strip()
    origin = "" if ("\r" in raw_origin or "\n" in raw_origin) else raw_origin
    allowed_origins = _parse_allowed_cors_origins()

    if not allowed_origins:
        # 未配置显式白名单时仅返回通配 `*`，绝不反射任意未知 Origin 且绝不附带 Allow-Credentials
        lines = ["Access-Control-Allow-Origin: *"]
    elif origin and origin.rstrip("/") in allowed_origins:
        lines = [f"Access-Control-Allow-Origin: {origin}", "Vary: Origin"]
        if str(os.environ.get("SSE_CORS_ALLOW_CREDENTIALS") or "").strip().lower() in ("1", "true", "yes", "on"):
            lines.append("Access-Control-Allow-Credentials: true")
    else:
        # 已启用白名单但请求未命中时，仅返回 Vary: Origin，不授予跨域读取权限
        lines = ["Vary: Origin"]

    lines.extend([
        "Access-Control-Allow-Methods: GET, POST, OPTIONS",
        "Access-Control-Allow-Headers: Authorization, Content-Type, X-Api-Key, Anthropic-Version, X-Requested-With",
    ])
    return "\r\n".join(lines) + "\r\n"


def _load_expected_api_key() -> str:
    """加载网关期待的 API Key（优先环境变量 `SSE_API_KEY`，其次 `SSE_API_KEY_FILE`）。"""
    key = str(os.environ.get("SSE_API_KEY") or "").strip()
    if key:
        return key
    key_file = str(os.environ.get("SSE_API_KEY_FILE") or "").strip()
    if key_file:
        try:
            raw = Path(key_file).read_text(encoding="utf-8", errors="ignore").strip()
            if raw:
                return raw.splitlines()[0].strip()
        except OSError:
            pass
    return ""


def _extract_request_api_key(hdr_map: dict[str, str]) -> str:
    """从请求头中提取 `Authorization: Bearer ...` 或 `x-api-key`。"""
    auth = str(hdr_map.get("authorization") or "").strip()
    if auth.lower().startswith("bearer "):
        tok = auth[7:].strip()
        if tok:
            return tok
    return str(hdr_map.get("x-api-key") or "").strip()


_METRICS_CACHE: dict[str, object] = {"ts": 0.0, "data": None}
_METRICS_CACHE_LOCK = threading.Lock()


def _probe_gpu_metrics() -> tuple[str, float, float, float]:
    """通过 nvidia-smi 采集单卡名称与显存使用（GB 与百分比），失败安全归零。"""
    try:
        nvsmi = shutil.which("nvidia-smi")
        if not nvsmi:
            return "", 0.0, 0.0, 0.0
        out = subprocess.check_output(
            [nvsmi, "--query-gpu=name,memory.used,memory.total", "--format=csv,noheader,nounits"],
            stdin=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True, timeout=1.5,
        ).strip()
        parts = [p.strip() for p in (out.splitlines() or [""])[0].split(",")]
        if len(parts) < 3:
            return "", 0.0, 0.0, 0.0
        u_mib, t_mib = float(parts[1]), float(parts[2])
        if t_mib <= 0:
            return parts[0], 0.0, 0.0, 0.0
        return parts[0], round(u_mib / 1024.0, 1), round(t_mib / 1024.0, 1), round(u_mib * 100.0 / t_mib, 1)
    except Exception:
        return "", 0.0, 0.0, 0.0


def _probe_ram_metrics() -> tuple[float, float, float, float]:
    """解析 Linux /proc/meminfo 获取系统内存（已用/总量/可用 GB 与占用百分比）。"""
    meminfo_p = Path("/proc/meminfo")
    if not meminfo_p.is_file():
        return 0.0, 0.0, 0.0, 0.0
    try:
        vals: dict[str, int] = {}
        for line in meminfo_p.read_text(encoding="utf-8", errors="ignore").splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                parts = v.strip().split()
                if parts and parts[0].isdigit():
                    vals[k.strip()] = int(parts[0])
        tot_kb = vals.get("MemTotal", 0)
        avail_kb = vals.get("MemAvailable")
        if avail_kb is None and tot_kb > 0:
            avail_kb = vals.get("MemFree", 0) + vals.get("Buffers", 0) + vals.get("Cached", 0)
        if tot_kb <= 0 or avail_kb is None:
            return 0.0, 0.0, 0.0, 0.0
        avail_kb = max(0, min(tot_kb, avail_kb))
        used_kb = max(0, tot_kb - avail_kb)
        gb_div = 1024.0 * 1024.0
        return (
            round(used_kb / gb_div, 1), round(tot_kb / gb_div, 1),
            round(avail_kb / gb_div, 1), round(used_kb * 100.0 / tot_kb, 1),
        )
    except Exception:
        return 0.0, 0.0, 0.0, 0.0


def _probe_disk_metrics(path_str: str) -> tuple[float, float, float, float]:
    """采集指定挂载路径的磁盘容量（已用/总量/可用 GB 与占用百分比）。"""
    try:
        du = shutil.disk_usage(path_str)
        if du.total <= 0:
            return 0.0, 0.0, 0.0, 0.0
        used_b = max(0, du.total - du.free)
        gb_div = float(1024 ** 3)
        return (
            round(used_b / gb_div, 1), round(du.total / gb_div, 1),
            round(du.free / gb_div, 1), round(used_b * 100.0 / du.total, 1),
        )
    except Exception:
        return 0.0, 0.0, 0.0, 0.0


def _probe_drive_metrics() -> tuple[bool, float, float]:
    """探测 Google Drive 挂载或 rclone 持久化配置及可用空间。"""
    # FUSE 挂载未就绪或 statvfs 受限时按未挂载降级返回 0
    with contextlib.suppress(OSError):
        drive_p = Path("/content/drive/MyDrive")
        if drive_p.is_dir():
            _, total_gb, free_gb, _ = _probe_disk_metrics(str(drive_p))
            return True, free_gb, total_gb
        if Path("/tmp/rclone.conf").is_file():
            return True, 0.0, 0.0
    return False, 0.0, 0.0


def _collect_gateway_host_metrics() -> dict:
    """采集云端 LLM 实例 GPU 显存、内存 RAM、磁盘、/dev/shm 与 Google Drive 挂载空间（受鉴权保护）。"""
    now_m = time.monotonic()
    with _METRICS_CACHE_LOCK:
        cached = _METRICS_CACHE.get("data")
        cached_ts = float(_METRICS_CACHE.get("ts") or 0.0)
        if isinstance(cached, dict) and (now_m - cached_ts) < 3.0:
            return dict(cached)

    gpu_full_name, vram_used_gb, vram_total_gb, vram_pct = _probe_gpu_metrics()
    ram_used_gb, ram_total_gb, ram_free_gb, ram_pct = _probe_ram_metrics()
    disk_target = "/content" if Path("/content").exists() else "/"
    disk_used_gb, disk_total_gb, disk_free_gb, disk_pct = _probe_disk_metrics(disk_target)
    shm_p = Path("/dev/shm")
    shm_used_gb, shm_total_gb, shm_free_gb, _ = (
        _probe_disk_metrics(str(shm_p)) if shm_p.exists() else (0.0, 0.0, 0.0, 0.0)
    )
    drive_persisted, drive_free_gb, drive_total_gb = _probe_drive_metrics()

    payload = {
        "ok": True,
        "gpu_full_name": gpu_full_name, "vram_used_gb": vram_used_gb,
        "vram_total_gb": vram_total_gb, "vram_pct": vram_pct,
        "ram_used_gb": ram_used_gb, "ram_total_gb": ram_total_gb,
        "ram_free_gb": ram_free_gb, "ram_pct": ram_pct,
        "disk_used_gb": disk_used_gb, "disk_total_gb": disk_total_gb,
        "disk_free_gb": disk_free_gb, "disk_pct": disk_pct,
        "shm_used_gb": shm_used_gb, "shm_total_gb": shm_total_gb, "shm_free_gb": shm_free_gb,
        "drive_persisted": drive_persisted, "drive_free_gb": drive_free_gb, "drive_total_gb": drive_total_gb,
    }
    done_m = time.monotonic()
    with _METRICS_CACHE_LOCK:
        _METRICS_CACHE["ts"] = done_m
        _METRICS_CACHE["data"] = dict(payload)
    return payload


def _is_protected_api_path(path: str) -> bool:
    """判断路径是否属于受保护的推理与模型接口（`/health` 与 `OPTIONS` 免密）。"""
    return path in STREAM_PATHS or path in MODELS_PATHS or path.startswith("/v1/") or path in _PROTECTED_EXTRA_PATHS


def _safe_decompress(data: bytes, encoding: str, max_bytes: int = MAX_DECOMPRESSED_BYTES) -> bytes:
    """带严格输出体积上限的流式解压（防御高压缩比 gzip/deflate 内存膨胀攻击）。"""
    enc = str(encoding or "").strip().lower()
    if "gzip" in enc:
        wbits_candidates: tuple[int, ...] = (16 + zlib.MAX_WBITS,)
    elif "deflate" in enc:
        wbits_candidates = (zlib.MAX_WBITS, -zlib.MAX_WBITS)
    else:
        return data

    limit = max(1, int(max_bytes))
    last_err: zlib.error | None = None
    for wbits in wbits_candidates:
        try:
            decomp = zlib.decompressobj(wbits)
            out = decomp.decompress(data, limit + 1)
            if len(out) > limit or decomp.unconsumed_tail:
                raise OverflowError("decompressed_too_large")
            total = out + decomp.flush(limit + 1 - len(out))
            if len(total) > limit:
                raise OverflowError("decompressed_too_large")
            return total
        except zlib.error as exc:
            last_err = exc
    if last_err is not None:
        raise last_err
    return data


async def _send_json_response(
    writer: asyncio.StreamWriter,
    status: int,
    reason: str,
    body: bytes,
    hdr_map: dict[str, str] | None = None,
    extra_headers: dict[str, str] | None = None,
) -> None:
    """向客户端写出带统一 CORS 与关闭连接语义的 JSON 响应。"""
    cors_blk = _cors_header_block(hdr_map or {})
    extra_blk = "".join(f"{k}: {v}\r\n" for k, v in (extra_headers or {}).items())
    hdr = (
        f"HTTP/1.1 {status} {reason}\r\n"
        f"{cors_blk}{extra_blk}"
        f"Content-Type: application/json; charset=utf-8\r\n"
        f"Content-Length: {len(body)}\r\n"
        f"Connection: close\r\n\r\n"
    ).encode("latin-1")
    writer.write(hdr + body)
    await writer.drain()


async def _send_simple_error(
    writer: asyncio.StreamWriter,
    status: int,
    reason: str,
    message: str,
    hdr_map: dict[str, str] | None = None,
    extra_headers: dict[str, str] | None = None,
) -> None:
    """向下游返回标准 OpenAI 风格 JSON 错误响应（附带 CORS 头防浏览器跨域拦截）。"""
    body = json.dumps({"error": {"message": message, "type": "gateway_error"}}, ensure_ascii=False).encode("utf-8")
    await _send_json_response(writer, status, reason, body, hdr_map, extra_headers)


def _format_sse_error_frame(path: str, status: int, raw_text: str) -> bytes:
    """按请求端点协议格式化已进入 SSE 流后的终端错误帧（兼容 Anthropic /v1/messages 与 OpenAI）。"""
    clean_msg = str(raw_text or f"Upstream HTTP {status}").strip()
    err_type = "api_error"
    # 上游返回非 JSON 纯文本错误体时保留原文本透传
    with contextlib.suppress(ValueError, TypeError):
        parsed = json.loads(clean_msg)
        if isinstance(parsed, dict):
            inner = parsed.get("error") if isinstance(parsed.get("error"), dict) else parsed
            if isinstance(inner, dict):
                clean_msg = str(inner["message"]) if inner.get("message") else clean_msg
                err_type = str(inner["type"]) if inner.get("type") else err_type

    if path in {"/v1/messages", "/messages"}:
        if status == 400 or "context size" in clean_msg.lower():
            anth_type = "invalid_request_error"
        elif status in (401, 403):
            anth_type = "authentication_error"
        elif status == 429:
            anth_type = "rate_limit_error"
        else:
            anth_type = "api_error"
        frame_obj = {"type": "error", "error": {"type": anth_type, "message": clean_msg}}
        return f"event: error\ndata: {json.dumps(frame_obj, ensure_ascii=False)}\n\n".encode()

    err_obj = {"error": {"status": status, "type": err_type, "message": clean_msg}}
    return f"data: {json.dumps(err_obj, ensure_ascii=False)}\n\ndata: [DONE]\n\n".encode()


async def _send_chunked_sse_error(client_writer: asyncio.StreamWriter, path: str, status: int, raw_text: str) -> None:
    """在已发送 SSE 响应头后，以 Chunked 形式写出终端错误帧并关闭分块流。"""
    err_evt = _format_sse_error_frame(path, status, raw_text)
    client_writer.write(f"{len(err_evt):x}\r\n".encode("ascii") + err_evt + b"\r\n0\r\n\r\n")
    await client_writer.drain()


async def _close_writer(writer: asyncio.StreamWriter | None) -> None:
    """安全关闭 StreamWriter，忽略已断开连接异常。"""
    if writer is None:
        return
    # 关闭已断开或半关闭的传输流时忽略连接重置与取消异常
    with contextlib.suppress(OSError, RuntimeError, asyncio.CancelledError):
        if not writer.is_closing():
            writer.close()
        await writer.wait_closed()


async def _read_and_authenticate_request(
    client_reader: asyncio.StreamReader,
    client_writer: asyncio.StreamWriter,
) -> _ParsedHttpRequest | None:
    """读取 HTTP 请求头、校验请求行、处理 OPTIONS 预检，并执行前置 API Key 恒时鉴权。"""
    try:
        raw_req_head = await client_reader.readuntil(b"\r\n\r\n")
    except (asyncio.IncompleteReadError, asyncio.LimitOverrunError, ConnectionError):
        return None

    start_line, hdrs, hdr_map = _parse_headers(raw_req_head)
    parts = start_line.split(" ", 2)
    if len(parts) < 2:
        await _send_simple_error(client_writer, 400, "Bad Request", "Malformed HTTP request line", hdr_map)
        return None

    method, target = parts[0].upper(), parts[1]
    path = target.split("?", 1)[0]
    if method == "OPTIONS":
        preflight_hdr = (
            f"HTTP/1.1 204 No Content\r\n"
            f"{_cors_header_block(hdr_map)}"
            f"Access-Control-Max-Age: 86400\r\n"
            f"Connection: close\r\n\r\n"
        ).encode("latin-1")
        client_writer.write(preflight_hdr)
        await client_writer.drain()
        return None

    expected_key = _load_expected_api_key()
    auth_active = (os.environ.get("SSE_REQUIRE_AUTH") == "1") or bool(expected_key)
    if auth_active and _is_protected_api_path(path):
        if not expected_key:
            await _send_simple_error(client_writer, 503, "Service Unavailable", "Gateway API key not configured", hdr_map)
            return None
        req_key = _extract_request_api_key(hdr_map)
        if not req_key or not hmac.compare_digest(req_key, expected_key):
            await _send_simple_error(
                client_writer, 401, "Unauthorized", "Invalid or missing API key",
                hdr_map, extra_headers={"WWW-Authenticate": "Bearer"},
            )
            return None

    return _ParsedHttpRequest(
        raw_head=raw_req_head, start_line=start_line, hdrs=hdrs,
        hdr_map=hdr_map, method=method, path=path, is_upgrade=("upgrade" in hdr_map),
    )


async def _handle_metrics_request(client_writer: asyncio.StreamWriter, hdr_map: dict[str, str]) -> None:
    """响应 `GET /__pet_metrics`：在工作线程采集主机遥测指标，避免阻塞事件循环。"""
    metrics_payload = await asyncio.to_thread(_collect_gateway_host_metrics)
    metrics_bytes = json.dumps(metrics_payload, ensure_ascii=False).encode("utf-8")
    await _send_json_response(client_writer, 200, "OK", metrics_bytes, hdr_map, {"Cache-Control": "no-store"})


def _parse_inspect_body_mode(req: _ParsedHttpRequest) -> tuple[bool, bool, int]:
    """判定当前请求是否需要拦截探测，返回 `(can_inspect, is_chunked, content_len)`。"""
    if req.is_upgrade or req.method != "POST" or req.path not in STREAM_PATHS:
        return False, False, 0
    is_chunked = "chunked" in req.hdr_map.get("transfer-encoding", "").lower()
    if is_chunked:
        return True, True, 0
    cl_raw = req.hdr_map.get("content-length")
    if cl_raw is None:
        return False, False, 0
    try:
        content_len = int(cl_raw.strip())
    except ValueError:
        return False, False, 0
    return (0 < content_len <= MAX_STREAM_INSPECT_BYTES), False, content_len


def _rewrite_inspected_payload(body_bytes: bytes, req: _ParsedHttpRequest) -> _InspectedRequestBody:
    """解压（若启用 gzip/deflate）并规范化 JSON 推理载荷；解压超限时抛出 OverflowError。"""
    try:
        inspect_bytes = body_bytes
        enc = req.hdr_map.get("content-encoding", "").lower()
        if "gzip" in enc or "deflate" in enc:
            inspect_bytes = _safe_decompress(body_bytes, enc, MAX_DECOMPRESSED_BYTES)
        payload_obj = json.loads(inspect_bytes.decode("utf-8"))
        if isinstance(payload_obj, dict):
            is_stream = bool(payload_obj.get("stream"))
            normalized = _normalize_inference_payload(payload_obj, req.path)
            rewritten = json.dumps(normalized, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            return _InspectedRequestBody(body_bytes=rewritten, is_stream=is_stream, was_rewritten=True)
        return _InspectedRequestBody(body_bytes=body_bytes, is_stream=False, was_rewritten=False)
    except OverflowError:
        raise
    except Exception:
        # 若外层带非标准编码或微格式异常，至少按明文切片探测 stream 意图，绝不误关 SSE 保活
        is_stream = bool(re.search(rb'"stream"\s*:\s*true\b', body_bytes or b""))
        return _InspectedRequestBody(body_bytes=body_bytes, is_stream=is_stream, was_rewritten=False)


async def _inspect_and_rewrite_stream_body(
    req: _ParsedHttpRequest,
    client_reader: asyncio.StreamReader,
    client_writer: asyncio.StreamWriter,
) -> _InspectedRequestBody | None:
    """对推理 POST 请求处理 100-continue、读取载荷、防炸弹安全解压并规范化思考参数。"""
    can_inspect, is_chunked_req, content_len = _parse_inspect_body_mode(req)
    if not can_inspect:
        return _InspectedRequestBody(body_bytes=None, is_stream=False, was_rewritten=False)

    if "100-continue" in req.hdr_map.get("expect", "").lower():
        client_writer.write(b"HTTP/1.1 100 Continue\r\n\r\n")
        await client_writer.drain()

    try:
        body_bytes = (
            await _read_chunked_body(client_reader, MAX_STREAM_INSPECT_BYTES)
            if is_chunked_req
            else await client_reader.readexactly(content_len)
        )
    except ValueError as exc:
        if str(exc) == "payload_too_large":
            await _send_simple_error(client_writer, 413, "Payload Too Large", "Request body exceeds 128 MiB limit", req.hdr_map)
        else:
            await _send_simple_error(client_writer, 400, "Bad Request", "Invalid chunked request body", req.hdr_map)
        return None
    except (EOFError, asyncio.IncompleteReadError):
        return None

    try:
        return _rewrite_inspected_payload(body_bytes, req)
    except OverflowError:
        await _send_simple_error(
            client_writer, 413, "Payload Too Large", "Decompressed request body exceeds size limit", req.hdr_map
        )
        return None


async def _read_upstream_body(up_reader: asyncio.StreamReader, up_hdr_map: dict[str, str], limit: int) -> bytes:
    """按上游响应头（Chunked / Content-Length / EOF）读取完整响应体。"""
    if "chunked" in up_hdr_map.get("transfer-encoding", "").lower():
        return await _read_chunked_body(up_reader, limit)
    if "content-length" in up_hdr_map:
        try:
            return await up_reader.readexactly(int(up_hdr_map["content-length"].strip()))
        except Exception:
            return await up_reader.read()
    return await up_reader.read()


async def _proxy_models_catalog(
    req: _ParsedHttpRequest,
    up_reader: asyncio.StreamReader,
    up_writer: asyncio.StreamWriter,
    client_writer: asyncio.StreamWriter,
) -> None:
    """转发 `GET /v1/models` 至上游，并对 200 JSON 响应注入思考强度虚拟模型目录。"""
    up_writer.write(_rebuild_headers(req.start_line, req.hdrs, force_close=True, strip_accept_encoding=True))
    await up_writer.drain()

    try:
        up_hdr_raw = await up_reader.readuntil(b"\r\n\r\n")
    except (asyncio.IncompleteReadError, asyncio.LimitOverrunError, ConnectionError) as exc:
        await _send_simple_error(client_writer, 502, "Bad Gateway", f"Upstream closed early: {exc}", req.hdr_map)
        return

    up_start, _, up_hdr_map = _parse_headers(up_hdr_raw)
    if _parse_http_status(up_start) != 200:
        client_writer.write(up_hdr_raw)
        await client_writer.drain()
        await _pump_stream(up_reader, client_writer)
        return

    raw_models_body = await _read_upstream_body(up_reader, up_hdr_map, _MAX_MODELS_BODY_BYTES)
    up_enc = up_hdr_map.get("content-encoding", "").lower()
    if "gzip" in up_enc or "deflate" in up_enc:
        try:
            raw_models_body = _safe_decompress(raw_models_body, up_enc, _MAX_MODELS_BODY_BYTES)
        except OverflowError:
            await _send_simple_error(client_writer, 502, "Bad Gateway", "Upstream /models response too large", req.hdr_map)
            return

    try:
        models_obj = json.loads(raw_models_body.decode("utf-8"))
        out_models_bytes = json.dumps(_augment_models_catalog(models_obj), ensure_ascii=False).encode("utf-8")
    except Exception:
        out_models_bytes = raw_models_body

    await _send_json_response(client_writer, 200, "OK", out_models_bytes, req.hdr_map)


async def _forward_request_to_upstream(
    req: _ParsedHttpRequest,
    inspected: _InspectedRequestBody,
    up_writer: asyncio.StreamWriter,
) -> None:
    """将重写后或原始请求头及可选请求体写入上游连接。"""
    if inspected.body_bytes is not None:
        fwd_head = _rebuild_headers(
            req.start_line, req.hdrs, force_close=True,
            content_length=len(inspected.body_bytes), strip_expect=True,
            strip_accept_encoding=inspected.is_stream, strip_content_encoding=inspected.was_rewritten,
        )
        up_writer.write(fwd_head + inspected.body_bytes)
    elif not req.is_upgrade:
        up_writer.write(_rebuild_headers(req.start_line, req.hdrs, force_close=True))
    else:
        up_writer.write(req.raw_head)
    await up_writer.drain()


async def _proxy_passthrough(
    req: _ParsedHttpRequest,
    body_bytes: bytes | None,
    client_reader: asyncio.StreamReader,
    client_writer: asyncio.StreamWriter,
    up_reader: asyncio.StreamReader,
    up_writer: asyncio.StreamWriter,
) -> None:
    """执行常规 HTTP、非流式推理或 Upgrade 连接的双向零改动转发。"""
    if body_bytes is not None:
        u2c = asyncio.create_task(_pump_stream(up_reader, client_writer))
        disc = asyncio.create_task(_wait_client_disconnect(client_reader))
        done, pending = await asyncio.wait({u2c, disc}, return_when=asyncio.FIRST_COMPLETED)
        for t in pending:
            t.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        for t in done:
            if not t.cancelled():
                exc = t.exception()
                if exc and not isinstance(exc, ConnectionError):
                    raise exc
        return

    c2u = asyncio.create_task(_pump_stream(client_reader, up_writer))
    u2c = asyncio.create_task(_pump_stream(up_reader, client_writer))
    try:
        done, pending = await asyncio.wait({c2u, u2c}, return_when=asyncio.FIRST_COMPLETED)
        if c2u in done and u2c in pending:
            if not req.is_upgrade:
                # 向上游转发半关闭 EOF 失败（对端已关闭）时安全忽略
                with contextlib.suppress(OSError, RuntimeError):
                    if up_writer.can_write_eof():
                        up_writer.write_eof()
                await u2c
        elif u2c in done and c2u in pending:
            c2u.cancel()
            await asyncio.gather(c2u, return_exceptions=True)
    finally:
        for t in (c2u, u2c):
            if not t.done():
                t.cancel()
        await asyncio.gather(c2u, u2c, return_exceptions=True)


async def _wait_upstream_headers_with_heartbeat(
    req: _ParsedHttpRequest,
    client_writer: asyncio.StreamWriter,
    hdr_task: asyncio.Task[bytes],
    disc_task: asyncio.Task[None],
) -> tuple[bool, bool]:
    """在等待上游响应头期间按 HEARTBEAT_SEC 发送 SSE 心跳；返回 `(client_disconnected, sent_sse_headers)`。"""
    sent_sse_headers = False
    while not hdr_task.done():
        done, _ = await asyncio.wait({hdr_task, disc_task}, timeout=HEARTBEAT_SEC, return_when=asyncio.FIRST_COMPLETED)
        if disc_task in done:
            return True, sent_sse_headers
        if hdr_task in done:
            break
        if not sent_sse_headers:
            client_writer.write(
                (
                    f"HTTP/1.1 200 OK\r\n"
                    f"{_cors_header_block(req.hdr_map)}"
                    f"Content-Type: text/event-stream; charset=utf-8\r\n"
                    f"Cache-Control: no-cache, no-transform\r\n"
                    f"Connection: close\r\n"
                    f"Transfer-Encoding: chunked\r\n"
                    f"X-Accel-Buffering: no\r\n\r\n"
                ).encode("latin-1")
            )
            sent_sse_headers = True
        client_writer.write(_KEEPALIVE_CHUNK)
        await client_writer.drain()
    return False, sent_sse_headers


async def _pump_until_disconnect(
    up_reader: asyncio.StreamReader,
    client_writer: asyncio.StreamWriter,
    disc_task: asyncio.Task[None],
    *,
    as_chunked: bool = False,
    emit_last_chunk: bool = False,
) -> None:
    """将上游响应流转发至下游，同时监听客户端断连并及时取消转发任务。"""
    u2c = asyncio.create_task(
        _pump_stream(up_reader, client_writer, as_chunked=as_chunked, emit_last_chunk=emit_last_chunk)
    )
    _, pending = await asyncio.wait({u2c, disc_task}, return_when=asyncio.FIRST_COMPLETED)
    for t in pending:
        t.cancel()
    await asyncio.gather(*pending, return_exceptions=True)


async def _proxy_sse_with_heartbeat(
    req: _ParsedHttpRequest,
    client_reader: asyncio.StreamReader,
    client_writer: asyncio.StreamWriter,
    up_reader: asyncio.StreamReader,
) -> None:
    """处理 `stream=true` 的 SSE 流式推理：Prefill 心跳保活、断连监听与非 200 终端错误帧封装。"""
    hdr_task = asyncio.create_task(up_reader.readuntil(b"\r\n\r\n"))
    disc_task = asyncio.create_task(_wait_client_disconnect(client_reader))
    try:
        disconnected, sent_sse_headers = await _wait_upstream_headers_with_heartbeat(
            req, client_writer, hdr_task, disc_task
        )
        if disconnected:
            return

        try:
            up_hdr_raw = hdr_task.result()
        except (asyncio.IncompleteReadError, asyncio.LimitOverrunError, ConnectionError) as exc:
            if not sent_sse_headers:
                await _send_simple_error(client_writer, 502, "Bad Gateway", f"Upstream closed early: {exc}", req.hdr_map)
            else:
                await _send_chunked_sse_error(client_writer, req.path, 502, f"Upstream closed early: {exc}")
            return

        if not sent_sse_headers:
            client_writer.write(up_hdr_raw)
            await client_writer.drain()
            await _pump_until_disconnect(up_reader, client_writer, disc_task)
            return

        up_start, _, up_hdr_map = _parse_headers(up_hdr_raw)
        up_status = _parse_http_status(up_start)
        if up_status != 200:
            err_txt = (await up_reader.read(65536)).decode("utf-8", errors="ignore").strip() or f"Upstream HTTP {up_status}"
            await _send_chunked_sse_error(client_writer, req.path, up_status, err_txt)
            return

        up_is_chunked = "chunked" in up_hdr_map.get("transfer-encoding", "").lower()
        await _pump_until_disconnect(
            up_reader, client_writer, disc_task,
            as_chunked=(not up_is_chunked), emit_last_chunk=(not up_is_chunked),
        )
    finally:
        for t in (hdr_task, disc_task):
            if not t.done():
                t.cancel()
        await asyncio.gather(hdr_task, disc_task, return_exceptions=True)


async def _handle_client(client_reader: asyncio.StreamReader, client_writer: asyncio.StreamWriter) -> None:
    """处理单条入站 TCP HTTP 请求，执行鉴权、遥测、SSE 保活调度或零修改透传。"""
    up_writer: asyncio.StreamWriter | None = None
    try:
        req = await _read_and_authenticate_request(client_reader, client_writer)
        if req is None:
            return
        if (not req.is_upgrade) and (req.method == "GET") and (req.path == "/__pet_metrics"):
            await _handle_metrics_request(client_writer, req.hdr_map)
            return

        inspected = await _inspect_and_rewrite_stream_body(req, client_reader, client_writer)
        if inspected is None:
            return

        try:
            up_reader, up_writer = await asyncio.open_connection(UPSTREAM_HOST, UPSTREAM_PORT, limit=1024 * 1024)
        except Exception as exc:
            await _send_simple_error(client_writer, 502, "Bad Gateway", f"Upstream unavailable: {exc}", req.hdr_map)
            return

        if (not req.is_upgrade) and (req.method == "GET") and (req.path in MODELS_PATHS):
            await _proxy_models_catalog(req, up_reader, up_writer, client_writer)
            return

        await _forward_request_to_upstream(req, inspected, up_writer)
        if not inspected.is_stream:
            await _proxy_passthrough(req, inspected.body_bytes, client_reader, client_writer, up_reader, up_writer)
            return

        await _proxy_sse_with_heartbeat(req, client_reader, client_writer, up_reader)
    except ConnectionError:
        pass
    except Exception as exc:
        print(f"[sse-gw] unexpected error: {exc}", file=sys.stderr, flush=True)
    finally:
        await _close_writer(up_writer)
        await _close_writer(client_writer)


async def _amain() -> None:
    server = await asyncio.start_server(_handle_client, host=LISTEN_HOST, port=LISTEN_PORT, limit=1024 * 1024)
    print(
        f"[sse-gw] listening on {LISTEN_HOST}:{LISTEN_PORT} -> {UPSTREAM_HOST}:{UPSTREAM_PORT} "
        f"(heartbeat={HEARTBEAT_SEC}s)",
        flush=True,
    )
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    try:
        asyncio.run(_amain())
    except KeyboardInterrupt:
        pass
