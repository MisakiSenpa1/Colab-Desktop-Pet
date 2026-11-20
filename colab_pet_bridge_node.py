"""Colab 桌宠直铺画布服务端桥接与公网访问认证网关扩展。

功能说明：
1. 公网入口认证网关（aiohttp 中间件）：
   - 纯本机回环直连（127.0.0.1 / localhost 且无任何反向代理转发头）免密放行，确保云端脚本零改动运行；
   - 经由 cloudflared 等公网隧道进入的所有 HTTP / WebSocket 流量强制校验 Bearer 密钥或 HttpOnly 会话 Cookie；
   - 支持 `/__pet_auth?ticket=...` 桌宠一次性 HMAC 票据免输直进（防重放、防开放重定向），也支持 `/__pet_auth` 中文口令登录与连续输错限流；
   - `/__pet_health` 仅提供最小匿名健康状态探测，不泄露硬件或目录信息。
2. 直铺画布原生桥接路由：
   - 提供 `/colab_pet/ping`（version=3）、`/colab_pet/open_workflow` 与 `/colab_pet/ack_workflow`。
"""
from __future__ import annotations

import asyncio
import contextlib
import hashlib
import hmac
import html
import json
import os
import re
import shutil
import sys
import time
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path

from aiohttp import web

try:
    import server
except ImportError:
    server = None

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
WEB_DIRECTORY = "./js"

_ACK_MAP: OrderedDict[str, dict] = OrderedDict()
_SERVICE_META_PATH = Path("/root/colab_service_meta.json")
_DEFAULT_AUTH_KEY_PATH = Path("/root/comfy_auth_key")
_PROXY_FORWARD_HEADERS = (
    "Cf-Connecting-Ip",
    "Cf-Ray",
    "Cf-Visitor",
    "X-Forwarded-For",
    "X-Forwarded-Host",
    "X-Forwarded-Proto",
    "Forwarded",
    "True-Client-Ip",
    "Cdn-Loop",
)
_LOOPBACK_REMOTE_IPS = {"127.0.0.1", "::1", "::ffff:127.0.0.1"}
_LOOPBACK_HOST_RE = re.compile(r"^(?:127\.0\.0\.1|localhost|\[::1\]|::1)(?::\d{1,5})?$", re.IGNORECASE)
_API_PATH_PREFIXES = (
    "/system_stats",
    "/queue",
    "/history",
    "/view",
    "/prompt",
    "/object_info",
    "/upload/",
    "/userdata/",
    "/colab_pet/",
    "/api/",
    "/ws",
)
_TICKET_TTL_SEC = 120
_COOKIE_TTL_SEC = 86400
_COOKIE_NAME = "colab_pet_comfy_auth"
_MAX_USED_NONCES = 512
_RATE_WINDOW_SEC = 60.0
_RATE_MAX_FAILURES = 5


def _load_comfy_auth_key() -> str:
    """从私有目录加载 ComfyUI 公网访问密钥（绝不从 ComfyUI 静态目录读取）。"""
    key_file = Path(os.environ.get("COMFY_AUTH_KEY_FILE") or str(_DEFAULT_AUTH_KEY_PATH))
    if key_file.is_file():
        try:
            raw = key_file.read_text(encoding="utf-8", errors="ignore").strip()
            if raw:
                return raw.splitlines()[0].strip()
        except OSError:
            pass
    if _SERVICE_META_PATH.is_file():
        # 读取私有元数据文件失败时回退空密钥（公网将 fail-closed 返回 503）
        with contextlib.suppress(OSError, ValueError, TypeError):
            meta = json.loads(_SERVICE_META_PATH.read_text(encoding="utf-8"))
            if isinstance(meta, dict):
                val = str(meta.get("comfy_api_key") or "").strip()
                if val:
                    return val
    return ""


def _sanitize_next_path(raw_next: str | None) -> str:
    """限制登录后重定向目标仅为站内相对路径，防止开放重定向。"""
    s = str(raw_next or "").strip()
    if not s:
        return "/"
    if any(ch in s for ch in ("\r", "\n", "\x00", "\\")):
        return "/"
    if not s.startswith("/") or s.startswith("//"):
        return "/"
    return s


def _is_trusted_loopback(request: web.Request) -> bool:
    """仅当对端 IP 为本机回环、Host 为回环地址且无代理转发头时才免密放行。"""
    remote = str(getattr(request, "remote", "") or "").strip()
    if remote not in _LOOPBACK_REMOTE_IPS:
        return False
    headers = getattr(request, "headers", None)
    if headers is not None:
        for hdr in _PROXY_FORWARD_HEADERS:
            if hdr in headers:
                return False
    host = str(getattr(request, "host", "") or "").strip()
    if not host or not _LOOPBACK_HOST_RE.match(host):
        return False
    return True


def _is_https_request(request: web.Request) -> bool:
    if getattr(request, "secure", False):
        return True
    headers = getattr(request, "headers", None)
    if headers is None:
        return False
    if str(headers.get("X-Forwarded-Proto") or "").strip().lower() == "https":
        return True
    cf_vis = str(headers.get("Cf-Visitor") or "").lower()
    return '"scheme":"https"' in cf_vis or '"scheme": "https"' in cf_vis


def _sign_session_cookie(secret_key: str, now_ts: int | None = None) -> str:
    ts = int(now_ts if now_ts is not None else time.time())
    sig = hmac.new(
        secret_key.encode(),
        f"session:{ts}".encode(),
        hashlib.sha256,
    ).hexdigest()
    return f"{ts}.{sig}"


def _verify_session_cookie(secret_key: str, cookie_val: str | None) -> bool:
    raw = str(cookie_val or "").strip()
    if not raw or "." not in raw:
        return False
    ts_str, sig = raw.split(".", 1)
    if not ts_str.isdigit() or not sig:
        return False
    ts = int(ts_str)
    now_ts = int(time.time())
    if (now_ts - ts) > _COOKIE_TTL_SEC or (ts - now_ts) > 60:
        return False
    expected = hmac.new(
        secret_key.encode(),
        f"session:{ts}".encode(),
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(sig, expected)


def _client_ip_for_ratelimit(request: web.Request) -> str:
    headers = getattr(request, "headers", None)
    if headers is not None:
        cf_ip = str(headers.get("Cf-Connecting-Ip") or "").strip()
        if cf_ip:
            return cf_ip
        xff = str(headers.get("X-Forwarded-For") or "").strip()
        if xff:
            return xff.split(",", 1)[0].strip() or "unknown"
    return str(getattr(request, "remote", "") or "unknown").strip()


def _render_login_html(next_path: str, error_msg: str = "") -> str:
    escaped_next = html.escape(_sanitize_next_path(next_path), quote=True)
    err_block = ""
    if error_msg:
        err_block = f'<div class="err">{html.escape(error_msg)}</div>'
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>ComfyUI 公网访问验证</title>
  <style>
    body{{margin:0;min-height:100vh;display:flex;align-items:center;justify-content:center;
      font-family:"Microsoft YaHei UI",system-ui,-apple-system,sans-serif;
      background:linear-gradient(145deg,#f5f0ff,#ebe5ff);color:#2e1065;}}
    .card{{width:min(360px,90vw);background:rgba(255,255,255,.92);border:1.5px solid #c4b5fd;
      border-radius:16px;padding:24px 22px;box-shadow:0 16px 40px rgba(76,29,149,.16);}}
    h1{{margin:0 0 8px;font-size:18px;font-weight:800;}}
    p{{margin:0 0 14px;font-size:13px;line-height:1.55;color:#5b21b6;}}
    .err{{margin:0 0 12px;padding:8px 10px;border-radius:8px;background:#fee2e2;
      border:1px solid #fca5a5;color:#991b1b;font-size:12.5px;font-weight:700;}}
    input{{width:100%;box-sizing:border-box;padding:10px 12px;border-radius:10px;
      border:1.5px solid #c4b5fd;font-size:13.5px;outline:none;margin-bottom:12px;}}
    input:focus{{border-color:#7c3aed;box-shadow:0 0 0 3px rgba(124,58,237,.16);}}
    button{{width:100%;padding:10px 14px;border:none;border-radius:10px;cursor:pointer;
      background:linear-gradient(135deg,#7c3aed,#6366f1);color:#fff;font-size:14px;font-weight:700;}}
  </style>
</head>
<body>
  <form class="card" method="post" action="/__pet_auth">
    <h1>ComfyUI 工作台已受保护</h1>
    <p>请输入 Colab 桌宠中配置的访问口令，或直接从桌宠内点击「打开 ComfyUI」免密直进。</p>
    {err_block}
    <input type="hidden" name="next" value="{escaped_next}">
    <input type="password" name="token" placeholder="粘贴访问密钥 (cpk-...)" autocomplete="current-password" required autofocus>
    <button type="submit">验证并进入画布</button>
  </form>
</body>
</html>"""


def _consume_auth_ticket(
    used_nonces: OrderedDict[str, int],
    secret_key: str,
    ticket: str,
    raw_next: str | None,
) -> tuple[bool, str]:
    """校验并一次性核销桌宠免密 HMAC 票据，防开放重定向与防重放。"""
    if raw_next is not None and str(raw_next).strip():
        stripped_raw = str(raw_next).strip()
        clean_next = _sanitize_next_path(stripped_raw)
        if clean_next != stripped_raw:
            return False, "/"
    else:
        clean_next = "/"

    parts = str(ticket or "").strip().split(".")
    if len(parts) != 3:
        return False, clean_next
    ts_str, nonce, sig = parts
    if not ts_str.isdigit() or not nonce or not sig:
        return False, clean_next
    ts = int(ts_str)
    now_ts = int(time.time())
    if abs(now_ts - ts) > _TICKET_TTL_SEC:
        return False, clean_next

    expected = hmac.new(
        secret_key.encode(),
        f"ticket:{ts}:{nonce}:{clean_next}".encode(),
        hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(sig, expected):
        return False, clean_next

    # 清理过期 nonce 并强制防重放（同票据仅允许兑换一次）
    expire_before = now_ts - (_TICKET_TTL_SEC + 60)
    for old_n, old_ts in list(used_nonces.items()):
        if old_ts < expire_before:
            used_nonces.pop(old_n, None)
        else:
            break
    if nonce in used_nonces:
        return False, clean_next
    used_nonces[nonce] = ts
    used_nonces.move_to_end(nonce)
    while len(used_nonces) > _MAX_USED_NONCES:
        used_nonces.popitem(last=False)
    return True, clean_next


def _set_auth_cookie_response(
    request: web.Request,
    secret_key: str,
    status: int,
    location: str,
) -> web.Response:
    """构造下发 HttpOnly 会话 Cookie 的站内安全重定向响应。"""
    resp = web.Response(
        status=status,
        headers={
            "Location": _sanitize_next_path(location),
            "Cache-Control": "no-store",
            "Referrer-Policy": "no-referrer",
        },
    )
    resp.set_cookie(
        _COOKIE_NAME,
        _sign_session_cookie(secret_key),
        max_age=_COOKIE_TTL_SEC,
        path="/",
        httponly=True,
        samesite="Lax",
        secure=_is_https_request(request),
    )
    return resp


def _prune_and_check_ratelimit(
    auth_failures: dict[str, list[float]],
    client_ip: str,
    now_m: float,
) -> tuple[bool, list[float]]:
    """按窗口淘汰过期失败记录并返回 `(是否触发限流, 当前窗口失败时间列表)`。"""
    if len(auth_failures) > 256:
        for ip_k in list(auth_failures.keys()):
            kept = [t for t in auth_failures[ip_k] if (now_m - t) <= _RATE_WINDOW_SEC]
            if kept:
                auth_failures[ip_k] = kept
            else:
                auth_failures.pop(ip_k, None)
        while len(auth_failures) > 512:
            auth_failures.pop(next(iter(auth_failures)), None)
    hist = [t for t in auth_failures.get(client_ip, []) if (now_m - t) <= _RATE_WINDOW_SEC]
    auth_failures[client_ip] = hist
    return len(hist) >= _RATE_MAX_FAILURES, hist


async def _handle_pet_auth_request(
    request: web.Request,
    secret_key: str,
    used_nonces: OrderedDict[str, int],
    auth_failures: dict[str, list[float]],
) -> web.Response | None:
    """处理 `/__pet_auth` 的 GET 票据兑换与 POST 口令表单登录请求。"""
    if request.method == "GET":
        ticket = request.query.get("ticket", "")
        raw_next = request.query.get("next")
        ok, clean_next = _consume_auth_ticket(used_nonces, secret_key, ticket, raw_next)
        if ok:
            return _set_auth_cookie_response(request, secret_key, 302, clean_next)
        return web.Response(
            status=401,
            text=_render_login_html(clean_next, "免密票据无效或已过期，请手动输入口令"),
            content_type="text/html",
            charset="utf-8",
            headers={"Cache-Control": "no-store", "Referrer-Policy": "no-referrer"},
        )

    if request.method != "POST":
        return None

    client_ip = _client_ip_for_ratelimit(request)
    now_m = time.monotonic()
    limited, hist = _prune_and_check_ratelimit(auth_failures, client_ip, now_m)
    if limited:
        return web.json_response(
            {"error": "too_many_requests", "message": "口令错误次数过多，请 1 分钟后再试"},
            status=429,
            headers={"Cache-Control": "no-store", "Retry-After": "60"},
        )

    raw_next = request.query.get("next", "/")
    token_in = ""
    try:
        post_data = await request.post()
        token_in = str(post_data.get("token") or "").strip()
        if "next" in post_data:
            raw_next = str(post_data.get("next") or "/")
    except Exception:
        token_in = ""
    clean_next = _sanitize_next_path(raw_next)

    if token_in and hmac.compare_digest(token_in, secret_key):
        auth_failures.pop(client_ip, None)
        return _set_auth_cookie_response(request, secret_key, 303, clean_next)

    hist.append(now_m)
    auth_failures[client_ip] = hist
    await asyncio.sleep(0.35)
    return web.Response(
        status=401,
        text=_render_login_html(clean_next, "访问密钥错误，请核对后重试"),
        content_type="text/html",
        charset="utf-8",
        headers={"Cache-Control": "no-store", "Referrer-Policy": "no-referrer"},
    )


def _build_unauthorized_response(request: web.Request, path: str) -> web.Response:
    """按请求类型返回浏览器登录页或 API/WebSocket 401 JSON 响应。"""
    accept_hdr = str(request.headers.get("Accept") or "").lower()
    upgrade_hdr = str(request.headers.get("Upgrade") or "").lower()
    is_api_or_ws = bool(upgrade_hdr == "websocket") or any(
        path == p.rstrip("/") or path.startswith(p) for p in _API_PATH_PREFIXES
    )
    if request.method == "GET" and not is_api_or_ws and "text/html" in accept_hdr:
        return web.Response(
            status=401,
            text=_render_login_html(request.rel_url.path_qs if hasattr(request, "rel_url") else path),
            content_type="text/html",
            charset="utf-8",
            headers={"Cache-Control": "no-store", "Referrer-Policy": "no-referrer"},
        )
    return web.json_response(
        {"error": "unauthorized", "message": "ComfyUI 公网入口已启用访问保护"},
        status=401,
        headers={"Cache-Control": "no-store"},
    )


def create_comfy_auth_middleware(key_getter: Callable[[], str] | None = None):
    """创建 ComfyUI 公网访问鉴权中间件（支持传入自定义密钥获取器用于离线测试）。"""
    get_key = key_getter or _load_comfy_auth_key
    used_nonces: OrderedDict[str, int] = OrderedDict()
    auth_failures: dict[str, list[float]] = {}

    @web.middleware
    async def comfy_auth_middleware(request: web.Request, handler):
        key = str(get_key() or "").strip()
        path = str(request.path or "/")

        if path == "/__pet_health":
            return web.json_response(
                {"status": "ok", "service": "comfy", "auth_required": bool(key)},
                headers={"Cache-Control": "no-store"},
            )

        if _is_trusted_loopback(request):
            return await handler(request)

        if not key:
            return web.json_response(
                {"error": "auth_not_configured", "message": "ComfyUI 公网访问密钥未配置，拒绝公网免密访问"},
                status=503,
                headers={"Cache-Control": "no-store"},
            )

        if path == "/__pet_auth":
            auth_resp = await _handle_pet_auth_request(request, key, used_nonces, auth_failures)
            if auth_resp is not None:
                return auth_resp

        auth_hdr = str(request.headers.get("Authorization") or "").strip()
        if auth_hdr.lower().startswith("bearer "):
            bearer_val = auth_hdr[7:].strip()
            if bearer_val and hmac.compare_digest(bearer_val, key):
                return await handler(request)

        if _verify_session_cookie(key, request.cookies.get(_COOKIE_NAME)):
            return await handler(request)

        return _build_unauthorized_response(request, path)

    return comfy_auth_middleware


def _record_ack(req_id: str, name: str = "") -> None:
    if not req_id:
        return
    _ACK_MAP[req_id] = {"id": req_id, "name": name, "ts": time.time()}
    _ACK_MAP.move_to_end(req_id)
    while len(_ACK_MAP) > 64:
        _ACK_MAP.popitem(last=False)


def _count_browser_clients(ps) -> int:
    sockets = getattr(ps, "sockets", None) or {}
    count = 0
    for cid, ws in list(sockets.items()):
        if str(cid).startswith("colab_pet_"):
            continue
        if ws is None or getattr(ws, "closed", False):
            continue
        count += 1
    return count


def _collect_host_metrics() -> dict:
    """采集实例内存 RAM、本地磁盘 Disk、共享内存盘 /dev/shm 与 Google Drive 挂载空间。"""
    ram_total_gb = 0.0
    ram_free_gb = 0.0
    ram_used_gb = 0.0
    ram_pct = 0.0
    meminfo_p = Path("/proc/meminfo")
    if meminfo_p.is_file():
        # 非 Linux 或 /proc 不可读时按 0.0 降级，不阻断其余磁盘指标采集
        with contextlib.suppress(OSError, ValueError):
            vals: dict[str, int] = {}
            for line in meminfo_p.read_text(encoding="utf-8", errors="ignore").splitlines():
                if ":" not in line:
                    continue
                k, v = line.split(":", 1)
                parts = v.strip().split()
                if parts and parts[0].isdigit():
                    vals[k.strip()] = int(parts[0])
            tot_kb = vals.get("MemTotal", 0)
            avail_kb = vals.get("MemAvailable")
            if avail_kb is None and tot_kb > 0:
                avail_kb = vals.get("MemFree", 0) + vals.get("Buffers", 0) + vals.get("Cached", 0)
            if tot_kb > 0 and avail_kb is not None:
                avail_kb = max(0, min(tot_kb, avail_kb))
                used_kb = max(0, tot_kb - avail_kb)
                ram_total_gb = round(tot_kb / (1024.0 * 1024.0), 1)
                ram_free_gb = round(avail_kb / (1024.0 * 1024.0), 1)
                ram_used_gb = round(used_kb / (1024.0 * 1024.0), 1)
                ram_pct = round(used_kb * 100.0 / tot_kb, 1)

    disk_total_gb = 0.0
    disk_free_gb = 0.0
    disk_used_gb = 0.0
    disk_pct = 0.0
    # 磁盘空间探测失败时保留 0.0 默认值
    with contextlib.suppress(OSError):
        disk_target = "/content" if Path("/content").exists() else "/"
        du = shutil.disk_usage(disk_target)
        if du.total > 0:
            used_b = max(0, du.total - du.free)
            disk_total_gb = round(du.total / (1024 ** 3), 1)
            disk_free_gb = round(du.free / (1024 ** 3), 1)
            disk_used_gb = round(used_b / (1024 ** 3), 1)
            disk_pct = round(used_b * 100.0 / du.total, 1)

    shm_total_gb = 0.0
    shm_free_gb = 0.0
    shm_used_gb = 0.0
    # 共享内存盘不存在或不可读时保留 0.0 默认值
    with contextlib.suppress(OSError):
        shm_p = Path("/dev/shm")
        if shm_p.exists():
            su = shutil.disk_usage(str(shm_p))
            if su.total > 0:
                shm_total_gb = round(su.total / (1024 ** 3), 1)
                shm_free_gb = round(su.free / (1024 ** 3), 1)
                shm_used_gb = round(max(0, su.total - su.free) / (1024 ** 3), 1)

    drive_persisted = False
    drive_total_gb = 0.0
    drive_free_gb = 0.0
    # Google Drive FUSE 挂载未就绪或 statvfs 异常时按未挂载降级
    with contextlib.suppress(OSError):
        drive_p = Path("/content/drive/MyDrive")
        rclone_p = Path("/tmp/rclone.conf")
        if drive_p.is_dir():
            drive_persisted = True
            drv = shutil.disk_usage(str(drive_p))
            if drv.total > 0:
                drive_total_gb = round(drv.total / (1024 ** 3), 1)
                drive_free_gb = round(drv.free / (1024 ** 3), 1)
        elif rclone_p.is_file():
            drive_persisted = True

    return {
        "ok": True,
        "ram_used_gb": ram_used_gb,
        "ram_total_gb": ram_total_gb,
        "ram_free_gb": ram_free_gb,
        "ram_pct": ram_pct,
        "disk_used_gb": disk_used_gb,
        "disk_total_gb": disk_total_gb,
        "disk_free_gb": disk_free_gb,
        "disk_pct": disk_pct,
        "shm_used_gb": shm_used_gb,
        "shm_total_gb": shm_total_gb,
        "shm_free_gb": shm_free_gb,
        "drive_persisted": drive_persisted,
        "drive_free_gb": drive_free_gb,
        "drive_total_gb": drive_total_gb,
    }


def _register_routes() -> None:
    if server is None:
        return
    ps = getattr(server.PromptServer, "instance", None)
    if ps is None:
        return

    app = getattr(ps, "app", None)
    if app is not None and not getattr(ps, "_colab_pet_auth_mw_ready", False):
        ps._colab_pet_auth_mw_ready = True
        app.middlewares.insert(0, create_comfy_auth_middleware())

    if getattr(ps, "_colab_pet_routes_ready", False):
        return
    ps._colab_pet_routes_ready = True
    routes = ps.routes

    @routes.get("/colab_pet/ping")
    async def colab_pet_ping(_request):
        return web.json_response({
            "ok": True,
            "version": 3,
            "auth_enabled": bool(_load_comfy_auth_key()),
            "browser_clients": _count_browser_clients(ps),
        })

    @routes.get("/colab_pet/host_metrics")
    async def colab_pet_host_metrics(_request):
        # 将含 /proc/meminfo 与 FUSE 挂载空间读取的采集放入线程池，避免阻塞 aiohttp 事件循环
        payload = await asyncio.to_thread(_collect_host_metrics)
        return web.json_response(payload)

    @routes.post("/colab_pet/open_workflow")
    async def colab_pet_open_workflow(request):
        try:
            data = await request.json()
        except Exception:
            data = {}
        req_id = str(data.get("id") or "")
        name = str(data.get("name") or "")
        extra_names = data.get("extra_names") if isinstance(data.get("extra_names"), list) else []
        clients = _count_browser_clients(ps)
        if name and clients > 0:
            ps.send_sync(
                "colab_pet.open_workflow",
                {"id": req_id, "name": name, "extra_names": extra_names},
            )
        return web.json_response({
            "ok": True,
            "id": req_id,
            "browser_clients": clients,
        })

    @routes.post("/colab_pet/ack_workflow")
    async def colab_pet_ack_post(request):
        try:
            data = await request.json()
        except Exception:
            data = {}
        req_id = str(data.get("id") or "")
        name = str(data.get("name") or "")
        _record_ack(req_id, name)
        return web.json_response({"ok": True, "id": req_id})

    @routes.get("/colab_pet/ack_workflow")
    async def colab_pet_ack_get(request):
        req_id = str(request.query.get("id") or "")
        info = _ACK_MAP.get(req_id)
        return web.json_response({
            "ok": True,
            "acked": info is not None,
            "info": info,
        })


try:
    _register_routes()
except Exception as _reg_exc:
    if sys.stderr is not None:
        with contextlib.suppress(OSError):
            sys.stderr.write(f"[colab_pet_bridge] 跳过自动路由注册: {_reg_exc}\n")
