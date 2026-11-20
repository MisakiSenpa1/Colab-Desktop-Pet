"""Colab 桌宠本地只读 HTTP 资产与安全媒体流服务模块（纯 Python 标准库，零 GUI/PIL 依赖）。

负责提供：
1. 桌宠 `ui.html` 与 Studio 前端静态页（`studio.html` / `studio.css` / `studio.js`）；
2. `/assets/` 白名单静态资源（严禁隐藏点文件、路径穿越与非白名单扩展名，支持 `.png <-> .webp` 双向回退）；
3. `/__media/` 沙箱内本地创作媒体流（支持 HTTP Range 206 分段拖动播放）。
"""

from __future__ import annotations

import http.server
import mimetypes
import re
import socketserver
import sys
import threading
import urllib.parse
from pathlib import Path
from typing import Any

mimetypes.add_type("image/webp", ".webp")
mimetypes.add_type("text/css", ".css")
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("video/mp4", ".mp4")
mimetypes.add_type("video/webm", ".webm")
mimetypes.add_type("audio/wav", ".wav")
mimetypes.add_type("audio/mpeg", ".mp3")
mimetypes.add_type("audio/flac", ".flac")

PET_DIR: Path = Path(__file__).resolve().parent
ASSETS_DIR: Path = PET_DIR / "assets"
_DEFAULT_PET_DIR: Path = PET_DIR
_DEFAULT_ASSETS_DIR: Path = ASSETS_DIR

_ASSET_CACHE_LOCK = threading.Lock()
_ASSET_MEM_CACHE: dict[Path, tuple[int, bytes, str]] = {}

_MEDIA_ROOT_LOCK = threading.Lock()
_ACTIVE_MEDIA_ROOT: Path | None = None

_MEDIA_MIME_MAP: dict[str, str] = {
    ".png": "image/png",
    ".webp": "image/webp",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".flac": "audio/flac",
}
_STUDIO_STATIC_FILES: dict[str, str] = {
    "/studio.html": "text/html; charset=utf-8",
    "/studio.css": "text/css; charset=utf-8",
    "/studio.js": "application/javascript; charset=utf-8",
}
_ASSET_FALLBACK_MIME: dict[str, str] = {
    ".webp": "image/webp",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
    ".json": "application/json; charset=utf-8",
}
_ALLOWED_ASSET_EXTS: frozenset[str] = frozenset({".webp", ".png", ".ico", ".svg", ".json"})
_ALLOWED_LOOPBACK_HOSTS: frozenset[str] = frozenset({"127.0.0.1", "localhost", "::1"})
_MEDIA_STREAM_CHUNK_BYTES = 64 * 1024
_RANGE_HEADER_RE = re.compile(r"^bytes=(\d*)-(\d*)$", re.IGNORECASE)
_WIN_DRIVE_PREFIX_RE = re.compile(r"^[A-Za-z]:")
_ASCII_PORT_RE = re.compile(r"^[1-9][0-9]{0,4}$")


def _parse_host_port_str(port_str: str) -> int | None:
    """校验并解析纯 ASCII 十进制端口号（1~65535），非法或含前导零时返回 None。"""
    if not _ASCII_PORT_RE.fullmatch(port_str):
        return None
    port_val = int(port_str)
    return port_val if 1 <= port_val <= 65535 else None


def _is_allowed_loopback_host(raw_host: str, server_port: int) -> bool:
    """校验 HTTP `Host` 请求头是否为合法本地环回源，阻断 DNS Rebinding 与伪造 Host 攻击。

    规则：
    1. 空白、控制字符、空白符、多值逗号或路径/凭据分隔符一律拒绝；
    2. 仅允许环回主机名 `127.0.0.1`、`localhost`、`[::1]`（不区分大小写）；
    3. 若携带 `:port`，端口必须合法且严格等于当前服务监听端口 `server_port`。
    """
    if not isinstance(raw_host, str):
        return False
    host_val = raw_host.strip()
    if not host_val:
        return False
    if any(ord(ch) <= 0x20 or ord(ch) == 0x7F or ch in "/\\@?#," for ch in host_val):
        return False

    hostname = ""
    port: int | None = None
    if host_val.startswith("["):
        end_idx = host_val.find("]")
        if end_idx <= 1:
            return False
        hostname = host_val[1:end_idx].lower()
        if hostname != "::1":
            return False
        rest = host_val[end_idx + 1 :]
        if rest:
            if not rest.startswith(":"):
                return False
            port = _parse_host_port_str(rest[1:])
            if port is None:
                return False
    else:
        if "[" in host_val or "]" in host_val:
            return False
        colon_cnt = host_val.count(":")
        if colon_cnt > 1:
            return False
        if colon_cnt == 1:
            host_part, port_str = host_val.split(":", 1)
            port = _parse_host_port_str(port_str)
            if port is None:
                return False
            hostname = host_part.lower()
        else:
            hostname = host_val.lower()
        if hostname not in ("127.0.0.1", "localhost"):
            return False

    if hostname not in _ALLOWED_LOOPBACK_HOSTS:
        return False
    if port is not None:
        return int(server_port) > 0 and port == int(server_port)
    return True


def configure_asset_roots(pet_dir: Path, assets_dir: Path | None = None) -> tuple[Path, Path]:
    """同步配置静态前端与 `/assets/` 默认根目录（供源码态与 PyInstaller 冻结态统一调用）。"""
    global PET_DIR, ASSETS_DIR, _DEFAULT_PET_DIR, _DEFAULT_ASSETS_DIR
    resolved_pet = Path(pet_dir).resolve()
    resolved_assets = Path(assets_dir).resolve() if assets_dir is not None else (resolved_pet / "assets")
    PET_DIR = resolved_pet
    ASSETS_DIR = resolved_assets
    _DEFAULT_PET_DIR = resolved_pet
    _DEFAULT_ASSETS_DIR = resolved_assets
    return PET_DIR, ASSETS_DIR


def _safe_file_mtime_ns(path: Path) -> int | None:
    """安全读取非空文件的纳秒级修改时间，忽略编辑器瞬时写入竞争或文件缺失。"""
    try:
        st = path.stat()
        return int(st.st_mtime_ns) if st.st_size > 0 else None
    except OSError:
        return None


def _resolve_handler_pet_root() -> Path:
    """解析当前静态前端根目录（兼容测试临时覆写 `pet_app.PET_DIR`）。"""
    if PET_DIR != _DEFAULT_PET_DIR:
        return Path(PET_DIR).resolve()
    pa = sys.modules.get("pet_app")
    if pa is not None:
        pa_pet = getattr(pa, "PET_DIR", None)
        if pa_pet is not None and Path(pa_pet) != _DEFAULT_PET_DIR:
            return Path(pa_pet).resolve()
    return Path(PET_DIR).resolve()


def _resolve_handler_assets_root(handler: http.server.BaseHTTPRequestHandler) -> Path:
    """按 server 实例 -> handler 类属性 -> 模块级 ASSETS_DIR / PET_DIR 顺序解析静态资产根目录。"""
    srv_root = getattr(handler.server, "assets_root", None)
    if srv_root is not None:
        return Path(srv_root).resolve()
    cls_root = getattr(type(handler), "assets_root", None)
    if cls_root is not None:
        return Path(cls_root).resolve()
    if ASSETS_DIR != _DEFAULT_ASSETS_DIR:
        return Path(ASSETS_DIR).resolve()
    if PET_DIR != _DEFAULT_PET_DIR:
        return (Path(PET_DIR) / "assets").resolve()
    pa = sys.modules.get("pet_app")
    if pa is not None:
        pa_assets = getattr(pa, "ASSETS_DIR", None)
        if pa_assets is not None and Path(pa_assets) != _DEFAULT_ASSETS_DIR:
            return Path(pa_assets).resolve()
        pa_pet = getattr(pa, "PET_DIR", None)
        if pa_pet is not None and Path(pa_pet) != _DEFAULT_PET_DIR:
            return (Path(pa_pet) / "assets").resolve()
    return Path(ASSETS_DIR).resolve()


def _is_within_allowed_asset_root(candidate: Path, assets_base: Path) -> bool:
    """核验规范化路径严格位于 `assets_base` 下且非隐藏文件、属于扩展名白名单。"""
    try:
        candidate.relative_to(assets_base)
    except ValueError:
        return False
    return (
        candidate != assets_base
        and not candidate.name.startswith(".")
        and candidate.suffix.lower() in _ALLOWED_ASSET_EXTS
    )


def _resolve_safe_asset_path(assets_root: Path, asset_rel_raw: str) -> tuple[Path | None, int]:
    """严格校验 `/assets/` 相对路径与扩展名白名单，禁止隐藏点文件、穿越与非白名单后缀。"""
    if (
        not asset_rel_raw
        or "\x00" in asset_rel_raw
        or "\\" in asset_rel_raw
        or asset_rel_raw.startswith("/")
        or _WIN_DRIVE_PREFIX_RE.match(asset_rel_raw)
    ):
        return None, 403

    parts = asset_rel_raw.split("/")
    if any(not seg or seg in (".", "..") or seg.startswith(".") for seg in parts):
        return None, 403
    if Path(parts[-1]).suffix.lower() not in _ALLOWED_ASSET_EXTS:
        return None, 403

    assets_base = assets_root.resolve()
    target = assets_base.joinpath(*parts).resolve()
    if not _is_within_allowed_asset_root(target, assets_base):
        return None, 403

    if not target.is_file():
        suf = target.suffix.lower()
        alt_suf = ".webp" if suf == ".png" else (".png" if suf == ".webp" else "")
        if alt_suf:
            alt = target.with_suffix(alt_suf).resolve()
            if not _is_within_allowed_asset_root(alt, assets_base):
                return None, 403
            if alt.is_file():
                target = alt

    return (target, 200) if target.is_file() else (None, 404)


def _set_active_media_root(media_root: Path | None) -> None:
    """设置全局默认本地媒体沙箱根目录，便于测试直起 HTTP 服务时自动共享沙箱路径。"""
    global _ACTIVE_MEDIA_ROOT
    with _MEDIA_ROOT_LOCK:
        _ACTIVE_MEDIA_ROOT = Path(media_root).resolve() if media_root is not None else None


def _resolve_handler_media_root(handler: http.server.BaseHTTPRequestHandler) -> Path:
    """按 server 实例 -> handler 类属性 -> 模块全局活跃路径顺序解析媒体沙箱根目录。"""
    srv_root = getattr(handler.server, "media_root", None)
    if srv_root is not None:
        return Path(srv_root).resolve()
    cls_root = getattr(type(handler), "media_root", None)
    if cls_root is not None:
        return Path(cls_root).resolve()
    with _MEDIA_ROOT_LOCK:
        if _ACTIVE_MEDIA_ROOT is not None:
            return _ACTIVE_MEDIA_ROOT
    return (PET_DIR.parent.parent / "_downloads" / "all").resolve()


def _parse_byte_range(range_hdr: str, file_size: int) -> tuple[int, int, bool] | None:
    """解析单个 HTTP Range 字节区间，非法或越界返回 None（由调用方响应 416）。"""
    if not range_hdr:
        return 0, max(0, file_size - 1), False
    first_spec = range_hdr.split(",", 1)[0].strip()
    m = _RANGE_HEADER_RE.match(first_spec)
    if not m or file_size <= 0:
        return None
    s_raw, e_raw = m.group(1), m.group(2)
    if not s_raw and not e_raw:
        return None
    if not s_raw:
        suffix_len = int(e_raw)
        if suffix_len <= 0:
            return None
        start, end = max(0, file_size - suffix_len), file_size - 1
    elif not e_raw:
        start, end = int(s_raw), file_size - 1
    else:
        start, end = int(s_raw), min(int(e_raw), file_size - 1)
    if start >= file_size or start > end:
        return None
    return start, end, True


def _read_cached_asset(target: Path) -> tuple[bytes, str]:
    """按文件纳秒级 mtime 读写 `/assets/` 内存缓存，避免重复磁盘 IO。"""
    mtime_ns = _safe_file_mtime_ns(target) or 0
    with _ASSET_CACHE_LOCK:
        cached = _ASSET_MEM_CACHE.get(target)
        if cached is not None and cached[0] == mtime_ns:
            return cached[1], cached[2]
        data = target.read_bytes()
        guessed, _ = mimetypes.guess_type(str(target))
        ctype = guessed or _ASSET_FALLBACK_MIME.get(target.suffix.lower()) or "application/octet-stream"
        _ASSET_MEM_CACHE[target] = (mtime_ns, data, ctype)
        return data, ctype


class _PetAssetHandler(http.server.BaseHTTPRequestHandler):
    """只读提供 ui.html、Studio 前端静态文件、assets 资产与安全沙箱 /__media/ 媒体流。"""

    media_root: Path | None = None

    def _resolve_media_target(self, encoded_rel: str) -> tuple[Path | None, int]:
        """校验 `/__media/` 相对路径严格落在媒体沙箱根目录内。"""
        raw_rel = urllib.parse.unquote(encoded_rel).replace("\\", "/").lstrip("/")
        if (
            not raw_rel
            or "\x00" in raw_rel
            or _WIN_DRIVE_PREFIX_RE.match(raw_rel)
            or Path(raw_rel).is_absolute()
        ):
            return None, 403
        media_root = _resolve_handler_media_root(self)
        target = (media_root / raw_rel).resolve()
        try:
            target.relative_to(media_root)
        except ValueError:
            return None, 403
        if target == media_root:
            return None, 403
        return (target, 200) if target.is_file() else (None, 404)

    def _serve_media(self, encoded_rel: str, *, send_body: bool = True) -> None:
        """安全提供 download_root 沙箱内媒体文件，支持 HTTP Range 206 分段拖动播放。"""
        target, status = self._resolve_media_target(encoded_rel)
        if target is None or status != 200:
            self.send_error(status)
            return

        ext = target.suffix.lower()
        ctype = _MEDIA_MIME_MAP.get(ext) or mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        file_size = int(target.stat().st_size)
        parsed_range = _parse_byte_range(str(self.headers.get("Range") or "").strip(), file_size)
        if parsed_range is None:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{file_size}")
            self.end_headers()
            return

        start, end, is_partial = parsed_range
        content_len = (end - start + 1) if file_size > 0 else 0
        self.send_response(206 if is_partial else 200)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(content_len))
        if is_partial:
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

        if not send_body or content_len <= 0:
            return
        self._stream_file_slice(target, start, content_len)

    def _stream_file_slice(self, target: Path, start: int, length: int) -> None:
        """分块流式发送文件指定字节区间，忽略客户端提前断开。"""
        remaining = length
        try:
            with open(target, "rb") as fp:
                fp.seek(start)
                while remaining > 0:
                    buf = fp.read(min(_MEDIA_STREAM_CHUNK_BYTES, remaining))
                    if not buf:
                        break
                    self.wfile.write(buf)
                    remaining -= len(buf)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def _resolve_static_target(self, clean_path: str) -> tuple[Path | None, int, bool, str | None]:
        """解析静态页面或 `/assets/` 目标文件，返回 `(path, status, is_no_store, forced_ctype)`。"""
        is_ui_page = clean_path in ("/", "/ui.html")
        studio_ctype = _STUDIO_STATIC_FILES.get(clean_path)
        pet_root = _resolve_handler_pet_root()
        if is_ui_page or studio_ctype is not None:
            rel_name = "ui.html" if is_ui_page else clean_path.lstrip("/")
            target = (pet_root / rel_name).resolve()
            try:
                target.relative_to(pet_root)
            except ValueError:
                return None, 403, True, None
            return (target, 200, True, studio_ctype) if target.is_file() else (None, 404, True, None)
        if clean_path.startswith("/assets/"):
            assets_root = _resolve_handler_assets_root(self)
            safe_target, asset_status = _resolve_safe_asset_path(assets_root, clean_path[len("/assets/") :])
            return safe_target, asset_status, False, None
        return None, 404, False, None

    def _handle_request(self, *, send_body: bool) -> None:
        host_headers = self.headers.get_all("Host") or []
        raw_host = host_headers[0] if len(host_headers) == 1 else ""
        srv_addr = getattr(self.server, "server_address", None)
        server_port = (
            int(srv_addr[1])
            if isinstance(srv_addr, (tuple, list)) and len(srv_addr) >= 2
            else 0
        )
        if not _is_allowed_loopback_host(raw_host, server_port):
            self.send_error(403, "Forbidden Host")
            return

        raw_path = self.path.split("?", 1)[0]
        if raw_path.startswith("/__media/"):
            self._serve_media(raw_path[len("/__media/") :], send_body=send_body)
            return

        target, status, is_no_store_page, studio_ctype = self._resolve_static_target(urllib.parse.unquote(raw_path))
        if target is None or status != 200:
            self.send_error(status)
            return

        if is_no_store_page:
            ctype = studio_ctype or (mimetypes.guess_type(str(target))[0] or "text/html; charset=utf-8")
            data = target.read_bytes()
            cache_ctrl = "no-store"
        else:
            data, ctype = _read_cached_asset(target)
            cache_ctrl = "public, max-age=31536000, immutable"

        self.send_response(200)
        self.send_header("Content-Type", ctype or "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", cache_ctrl)
        self.end_headers()
        if send_body:
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass

    def do_GET(self) -> None:  # noqa: N802
        self._handle_request(send_body=True)

    def do_HEAD(self) -> None:  # noqa: N802
        self._handle_request(send_body=False)

    def log_message(self, format: str, *args: Any) -> None:
        """静默 HTTP 访问日志。"""


class _ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


def start_asset_http_server(media_root: Path) -> tuple[_ThreadingHTTPServer, int]:
    """在本地回环随机端口启动后台静态资源与媒体流 HTTP 服务器。"""
    httpd = _ThreadingHTTPServer(("127.0.0.1", 0), _PetAssetHandler)
    httpd.media_root = Path(media_root).resolve()  # type: ignore[attr-defined]
    port = int(httpd.server_address[1])
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, port
