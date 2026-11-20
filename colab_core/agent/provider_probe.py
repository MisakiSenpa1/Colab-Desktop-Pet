"""colab_core.agent.provider_probe —— 模型供应商目录发现、协议自动识别与并发连通性测速探针。"""

from __future__ import annotations

import concurrent.futures
import contextlib
import json
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from typing import Any

from colab_core.agent.protocols import (
    SUPPORTED_API_FORMATS,
    _normalize_anthropic_endpoint,
    _normalize_gemini_endpoint,
    _normalize_openai_endpoint,
    _normalize_openai_responses_endpoint,
    _strip_gateway_suffix,
    _validate_http_base_url,
)
from colab_core.runtime import (
    normalize_reasoning_effort_value,
    normalize_reasoning_level_list,
    sanitize_model_capabilities,
)
from colab_security import redact_secrets, register_ephemeral_secret, urlopen_safe, url_origin_key


def _build_probe_auth_headers(
    api_key: str, *, include_anthropic: bool = True, include_gemini: bool = True
) -> dict[str, str]:
    """统一构建模型列表与协议探针请求头，空密钥不发送空认证头以兼容本机免密网关。"""
    headers: dict[str, str] = {
        "Accept": "application/json",
        "User-Agent": "ColabPet-Agent/1.0",
    }
    key = str(api_key or "").strip()
    if key:
        register_ephemeral_secret(key)
        headers["Authorization"] = f"Bearer {key}"
        if include_anthropic:
            headers["x-api-key"] = key
        if include_gemini:
            headers["x-goog-api-key"] = key
    if include_anthropic:
        headers["anthropic-version"] = "2023-06-01"
    return headers


def _parse_model_capability(entry: dict[str, Any]) -> dict[str, Any] | None:
    """从单个 `/v1/models` 模型条目中提取思考能力与档位元数据。"""
    sub_caps = entry.get("capabilities") if isinstance(entry.get("capabilities"), dict) else {}
    has_reasoning = ("reasoning" in entry) or ("reasoning" in sub_caps)
    raw_levels = next(
        (
            val
            for val in (
                entry.get("reasoning_levels"),
                entry.get("supported_reasoning_levels"),
                entry.get("thinking_levels"),
                sub_caps.get("reasoning_levels"),
                sub_caps.get("supported_reasoning_levels"),
            )
            if val is not None
        ),
        None,
    )
    raw_def = (
        entry.get("default_reasoning_level")
        or entry.get("default_reasoning_effort")
        or entry.get("default_effort")
        or sub_caps.get("default_reasoning_level")
        or ""
    )
    def_lvl = normalize_reasoning_effort_value(raw_def, default="")
    if def_lvl == "auto":
        def_lvl = ""
    if not (has_reasoning or (raw_levels is not None) or bool(def_lvl)):
        return None
    if "reasoning" in entry:
        reasoning_flag = bool(entry.get("reasoning"))
    elif "reasoning" in sub_caps:
        reasoning_flag = bool(sub_caps.get("reasoning"))
    else:
        reasoning_flag = True
    return {
        "reasoning": reasoning_flag,
        "levels": normalize_reasoning_level_list(raw_levels) if reasoning_flag else [],
        "default_level": def_lvl if reasoning_flag else "",
    }


def _extract_models_and_capabilities(
    payload: Any,
) -> tuple[list[dict[str, str]], dict[str, dict[str, Any]]]:
    """提取模型清单并同步解析 Magpie (/v1/models) 下发的思考能力与档位元数据。"""
    raw_items: list[Any] = []
    if isinstance(payload, dict):
        if isinstance(payload.get("data"), list):
            raw_items = payload["data"]
        elif isinstance(payload.get("models"), list):
            raw_items = payload["models"]
    elif isinstance(payload, list):
        raw_items = payload

    out: list[dict[str, str]] = []
    raw_caps: dict[str, dict[str, Any]] = {}
    seen: set[str] = set()
    for entry in raw_items:
        if isinstance(entry, str):
            mid, owner = entry.strip(), ""
        elif isinstance(entry, dict):
            raw_id = entry.get("id") or entry.get("name") or entry.get("model") or ""
            mid = str(raw_id).strip()
            if mid.startswith("models/"):
                mid = mid[len("models/") :]
            owner = str(entry.get("owned_by") or entry.get("provider") or "").strip()
        else:
            continue
        if not mid or mid in seen:
            continue
        seen.add(mid)
        out.append({"id": mid, "owned_by": owner})
        if isinstance(entry, dict):
            cap = _parse_model_capability(entry)
            if cap is not None:
                raw_caps[mid] = cap
    return out, sanitize_model_capabilities(raw_caps)


def _extract_models_from_payload(payload: Any) -> list[dict[str, str]]:
    """兼容提取 OpenAI (/v1/models)、Anthropic、Magpie 与 Gemini (/v1beta/models) 返回的模型清单。"""
    models, _ = _extract_models_and_capabilities(payload)
    return models


def _candidate_model_endpoints(base_url: str, api_format: str) -> list[str]:
    """按协议与路径后缀推导模型列表探测候选地址清单。"""
    root = _strip_gateway_suffix(base_url)
    low = base_url.lower()
    if api_format == "gemini":
        cands = [f"{root}/v1beta/models", f"{base_url}/models", f"{root}/v1/models"]
    elif low.endswith("/models"):
        cands = [base_url]
    elif low.endswith(("/v1", "/v1beta")):
        cands = [f"{base_url}/models", f"{root}/v1/models"]
    else:
        cands = [f"{base_url}/v1/models", f"{base_url}/models", f"{root}/v1beta/models"]
    return list(dict.fromkeys(cands))


def fetch_provider_models(
    base_url: str,
    *,
    api_key: str = "",
    api_format: str = "openai",
    timeout_sec: float = 8.0,
) -> dict[str, Any]:
    """从指定网关 / 供应商拉取可用模型列表（兼容 Magpie 网关与 OpenAI / Anthropic / Gemini）。"""
    key = str(api_key or "").strip()
    secret_tuple: tuple[str, ...] = (key,) if key else ()
    try:
        u = _validate_http_base_url(base_url)
    except ValueError as exc:
        return {"ok": False, "models": [], "model_ids": [], "model_capabilities": {}, "count": 0, "message": str(exc)}

    fmt = str(api_format or "openai").strip().lower()
    dedup_urls = _candidate_model_endpoints(u, fmt)
    headers = _build_probe_auth_headers(key)
    exp_origin = url_origin_key(u)
    last_err = ""
    for ep in dedup_urls:
        req = urllib.request.Request(ep, headers=headers, method="GET")
        try:
            with urlopen_safe(req, timeout=max(2.0, float(timeout_sec)), expected_origin=exp_origin) as resp:
                raw_txt = resp.read(2 * 1024 * 1024).decode("utf-8", errors="replace")
            parsed = json.loads(raw_txt or "{}")
            models, model_caps = _extract_models_and_capabilities(parsed)
            model_ids = [m["id"] for m in models]
            return {
                "ok": True,
                "models": models,
                "model_ids": model_ids,
                "model_capabilities": model_caps,
                "count": len(model_ids),
                "endpoint": ep,
                "message": f"已获取 {len(model_ids)} 个可用模型" if model_ids else "接口连通但未返回可用模型",
            }
        except urllib.error.HTTPError as exc:
            err_body = ""
            # 读取可选 HTTP 错误响应体失败时回退至状态原因短语，不遮蔽原始状态码
            with contextlib.suppress(OSError, ValueError):
                err_body = exc.read(256).decode("utf-8", errors="replace").strip()
            last_err = redact_secrets(f"HTTP {exc.code}: {err_body or exc.reason}", secret_tuple)
            if exc.code in (401, 403):
                break
        except Exception as exc:
            last_err = redact_secrets(str(exc) or type(exc).__name__, secret_tuple)

    return {
        "ok": False,
        "models": [],
        "model_ids": [],
        "model_capabilities": {},
        "count": 0,
        "endpoint": dedup_urls[0] if dedup_urls else "",
        "message": f"获取模型列表失败：{last_err or '无法连接目标地址'}",
    }


def _build_probe_request(
    base_url: str, fmt: str, model: str, api_key: str
) -> tuple[str, dict[str, str], dict[str, Any]]:
    """为指定协议与模型构造最小化的轻量连通性探测请求。"""
    key = str(api_key or "").strip()
    mdl = str(model or "gpt-4o-mini").strip() or "gpt-4o-mini"
    headers = {"Content-Type": "application/json", "Accept": "application/json", "User-Agent": "ColabPet-Agent/1.0"}
    if fmt == "anthropic":
        ep = _normalize_anthropic_endpoint(base_url)
        headers["anthropic-version"] = "2023-06-01"
        if key:
            headers["x-api-key"] = key
            headers["Authorization"] = f"Bearer {key}"
        payload = {"model": mdl, "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}], "stream": False}
        return ep, headers, payload
    if fmt == "openai_responses":
        ep = _normalize_openai_responses_endpoint(base_url)
        if key:
            headers["Authorization"] = f"Bearer {key}"
        payload = {"model": mdl, "input": [{"role": "user", "content": "hi"}], "max_output_tokens": 16, "stream": False}
        return ep, headers, payload
    if fmt == "gemini":
        ep = _normalize_gemini_endpoint(base_url, mdl, stream=False)
        if key:
            headers["x-goog-api-key"] = key
        payload = {"contents": [{"role": "user", "parts": [{"text": "hi"}]}], "generationConfig": {"maxOutputTokens": 8}}
        return ep, headers, payload
    ep = _normalize_openai_endpoint(base_url)
    if key:
        headers["Authorization"] = f"Bearer {key}"
    payload = {"model": mdl, "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}], "stream": False}
    return ep, headers, payload


def _is_valid_protocol_response(fmt: str, body_obj: Any) -> bool:
    """校验返回 JSON 是否符合对应协议的最小结构特征。"""
    if not isinstance(body_obj, dict) or body_obj.get("error") or body_obj.get("type") == "error":
        return False
    if fmt == "anthropic":
        return body_obj.get("type") == "message" or isinstance(body_obj.get("content"), list)
    if fmt == "openai_responses":
        return body_obj.get("object") == "response" or isinstance(body_obj.get("output"), list) or "output_text" in body_obj
    if fmt == "gemini":
        return isinstance(body_obj.get("candidates"), list)
    return isinstance(body_obj.get("choices"), list)


def _probe_single_format(
    base_url: str,
    fmt: str,
    model: str,
    *,
    api_key: str = "",
    timeout_sec: float = 6.0,
) -> dict[str, Any]:
    """针对单个 (协议, 模型) 发送一次极轻探针并测量毫秒级响应延迟。"""
    key = str(api_key or "").strip()
    secret_tuple: tuple[str, ...] = (key,) if key else ()
    if key:
        register_ephemeral_secret(key)
    t0 = time.monotonic()
    try:
        ep, headers, payload = _build_probe_request(base_url, fmt, model, key)
    except ValueError as exc:
        return {"ok": False, "api_format": fmt, "model": model, "latency_ms": 0, "status": 0, "error": str(exc)}

    raw_data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(ep, data=raw_data, headers=headers, method="POST")
    try:
        with urlopen_safe(req, timeout=max(2.0, float(timeout_sec)), expected_origin=url_origin_key(base_url)) as resp:
            status = int(getattr(resp, "status", 200) or 200)
            ctype = str(resp.headers.get("Content-Type") or "").lower()
            raw_txt = resp.read(256 * 1024).decode("utf-8", errors="replace")
        latency_ms = int(round((time.monotonic() - t0) * 1000))
        if "text/event-stream" in ctype:
            has_sse = "data:" in raw_txt and '"error"' not in raw_txt[:256]
            return {
                "ok": has_sse,
                "api_format": fmt,
                "model": model,
                "latency_ms": latency_ms,
                "status": status,
                "error": "" if has_sse else "SSE 响应异常",
            }
        parsed = json.loads(raw_txt or "{}")
        if isinstance(parsed, dict) and parsed.get("error"):
            err_o = parsed["error"]
            emsg = err_o.get("message") if isinstance(err_o, dict) else str(err_o)
            return {
                "ok": False,
                "api_format": fmt,
                "model": model,
                "latency_ms": latency_ms,
                "status": status,
                "error": redact_secrets(str(emsg or "接口返回错误"), secret_tuple)[:120],
            }
        valid = _is_valid_protocol_response(fmt, parsed)
        return {
            "ok": valid,
            "api_format": fmt,
            "model": model,
            "latency_ms": latency_ms,
            "status": status,
            "error": "" if valid else "响应结构与目标协议不匹配",
        }
    except urllib.error.HTTPError as exc:
        latency_ms = int(round((time.monotonic() - t0) * 1000))
        err_body = ""
        # 读取可选 HTTP 错误响应体失败时回退至状态原因短语，不遮蔽原始状态码
        with contextlib.suppress(OSError, ValueError):
            err_body = exc.read(256).decode("utf-8", errors="replace").strip()
        detail = redact_secrets(f"HTTP {exc.code}: {err_body or exc.reason}", secret_tuple)
        return {"ok": False, "api_format": fmt, "model": model, "latency_ms": latency_ms, "status": exc.code, "error": detail[:140]}
    except Exception as exc:
        latency_ms = int(round((time.monotonic() - t0) * 1000))
        detail = redact_secrets(str(exc) or type(exc).__name__, secret_tuple)
        return {"ok": False, "api_format": fmt, "model": model, "latency_ms": latency_ms, "status": 0, "error": detail[:140]}


def detect_provider_protocol(
    base_url: str,
    *,
    api_key: str = "",
    hint_model: str = "",
    timeout_sec: float = 6.0,
) -> dict[str, Any]:
    """自动检测目标 Base URL 支持的 API 协议（OpenAI / Responses / Anthropic / Gemini）。"""
    try:
        u = _validate_http_base_url(base_url)
    except ValueError as exc:
        return {"ok": False, "api_format": "openai", "probe_model": "", "results": [], "message": str(exc)}

    probe_model = str(hint_model or "").strip()
    if not probe_model:
        fetched = fetch_provider_models(u, api_key=api_key, timeout_sec=min(4.0, timeout_sec))
        if fetched.get("ok") and fetched.get("model_ids"):
            probe_model = str(fetched["model_ids"][0])
    probe_model = probe_model or "gpt-4o-mini"

    results_by_fmt: dict[str, dict[str, Any]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futs = {
            pool.submit(_probe_single_format, u, fmt, probe_model, api_key=api_key, timeout_sec=timeout_sec): fmt
            for fmt in SUPPORTED_API_FORMATS
        }
        for fut in concurrent.futures.as_completed(futs):
            fmt = futs[fut]
            try:
                results_by_fmt[fmt] = fut.result()
            except Exception as exc:
                results_by_fmt[fmt] = {"ok": False, "api_format": fmt, "model": probe_model, "latency_ms": 0, "status": 0, "error": str(exc)}

    ordered_results = [results_by_fmt[f] for f in SUPPORTED_API_FORMATS]
    for res in ordered_results:
        if res["ok"]:
            fmt = res["api_format"]
            return {
                "ok": True,
                "api_format": fmt,
                "probe_model": probe_model,
                "latency_ms": res["latency_ms"],
                "results": ordered_results,
                "message": f"检测到兼容协议：{fmt}（模型 {probe_model} · {res['latency_ms']} ms）",
            }

    return {
        "ok": False,
        "api_format": "openai",
        "probe_model": probe_model,
        "results": ordered_results,
        "message": f"未匹配到返回 200 的协议（测试模型：{probe_model}），请手动选择协议或先获取模型后指定测试模型",
    }


def probe_provider_models(
    base_url: str,
    models: Sequence[str],
    *,
    api_key: str = "",
    api_format: str = "openai",
    timeout_sec: float = 8.0,
) -> dict[str, Any]:
    """并发逐个检测所选模型列表在指定协议与 Base URL 下的真实连通性与响应延迟。"""
    fmt = str(api_format or "openai").strip().lower()
    if fmt not in SUPPORTED_API_FORMATS:
        fmt = "openai"
    try:
        u = _validate_http_base_url(base_url)
    except ValueError as exc:
        return {"ok": False, "api_format": fmt, "results": [], "passed": 0, "total": 0, "message": str(exc)}

    cleaned_models: list[str] = []
    for item in models:
        ms = str(item or "").strip()
        if ms and ms not in cleaned_models:
            cleaned_models.append(ms)
        if len(cleaned_models) >= 16:
            break
    if not cleaned_models:
        return {"ok": False, "api_format": fmt, "results": [], "passed": 0, "total": 0, "message": "请先选择或填写至少一个模型 ID"}

    results_map: dict[str, dict[str, Any]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(cleaned_models))) as pool:
        futs = {
            pool.submit(_probe_single_format, u, fmt, mdl, api_key=api_key, timeout_sec=timeout_sec): mdl
            for mdl in cleaned_models
        }
        for fut in concurrent.futures.as_completed(futs):
            mdl = futs[fut]
            try:
                results_map[mdl] = fut.result()
            except Exception as exc:
                results_map[mdl] = {"ok": False, "api_format": fmt, "model": mdl, "latency_ms": 0, "status": 0, "error": str(exc)}

    ordered = [results_map[m] for m in cleaned_models]
    passed = sum(1 for r in ordered if r["ok"])
    return {
        "ok": passed > 0,
        "api_format": fmt,
        "results": ordered,
        "passed": passed,
        "total": len(ordered),
        "message": f"模型检测完成：{passed}/{len(ordered)} 个可用",
    }


__all__ = [
    "_build_probe_auth_headers",
    "_build_probe_request",
    "_extract_models_and_capabilities",
    "_extract_models_from_payload",
    "_is_valid_protocol_response",
    "_probe_single_format",
    "detect_provider_protocol",
    "fetch_provider_models",
    "probe_provider_models",
]
