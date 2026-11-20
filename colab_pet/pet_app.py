"""Colab 桌宠主程序入口（基于 pywebview + Edge WebView2 + Win32 区域裁剪无边框悬浮窗）。

核心模块职责：
1. `asset_server.py`：纯标准库本地静态页、白名单 `/assets/` 与 `/__media/` Range 206 流服务；
2. `win32_window.py`：Win32 ctypes 句柄、单实例互斥热替换、Alpha/圆角 Region 裁剪与 GDI+ 截图；
3. `bridge_api.py`：`PetBridgeApi` JS 桥接接口、表驱动透传器与双窗状态多播推送；
4. `pet_app.py`（本文件）：无控制台流兜底、`--internal-ctl` 极速无 GUI 转发、应用生命周期编排与兼容重导出。
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import math
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, TextIO

# 持有空流/控制台兜底句柄，防止 pythonw.exe 或 console=False 冻结包运行 pet_app.py 时 sys.stdout/sys.stderr 为 None
_DEVNULL_HANDLES: list[TextIO] = []
_CLI_CONSOLE_FLAGS = frozenset({"--doctor", "--export-diag", "--internal-ctl", "-h", "--help"})


def _ensure_std_streams() -> None:
    """在无控制台 GUI 进程中确保 stdout/stderr 可安全写入，CLI 诊断模式优先挂载父控制台。"""
    missing = [attr for attr in ("stdout", "stderr") if getattr(sys, attr, None) is None]
    if not missing:
        return

    attach_parent = (
        os.name == "nt"
        and getattr(sys, "stdout", None) is None
        and any(arg in _CLI_CONSOLE_FLAGS for arg in sys.argv)
    )
    use_conout = False
    if attach_parent:
        try:
            # ATTACH_PARENT_PROCESS = (DWORD)-1，在 console=False 冻结包执行 CLI 诊断时将输出回写到父控制台
            use_conout = bool(ctypes.windll.kernel32.AttachConsole(ctypes.c_uint(-1).value))
        except (OSError, AttributeError, ValueError, TypeError, RuntimeError):
            use_conout = False

    targets = ("CONOUT$", os.devnull) if use_conout else (os.devnull,)
    for attr in missing:
        handle: TextIO | None = None
        for stream_target in targets:
            try:
                handle = open(stream_target, "w", encoding="utf-8", errors="replace")
                break
            except OSError:
                continue
        if handle is not None:
            _DEVNULL_HANDLES.append(handle)
            setattr(sys, attr, handle)


_ensure_std_streams()

# 内部 CLI 转发（专供 PyInstaller 冻结单目录版后台无窗口调用 colab_ctl 子命令，零 GUI 开销）
if sys.argv[1:2] == ["--internal-ctl"]:
    _meipass_early = getattr(sys, "_MEIPASS", None)
    if getattr(sys, "frozen", False) and _meipass_early:
        _meipass_early_str = str(Path(_meipass_early).resolve())
        if _meipass_early_str not in sys.path:
            sys.path.insert(0, _meipass_early_str)
    _pet_dir_early = Path(__file__).resolve().parent
    for _p in (str(_pet_dir_early.parent), str(_pet_dir_early)):
        if _p not in sys.path:
            sys.path.insert(0, _p)
    import colab_ctl  # noqa: E402

    raise SystemExit(colab_ctl.main(sys.argv[2:]))


def _resolve_pet_dir() -> Path:
    """解析桌宠前端静态资源根目录（PyInstaller 冻结态优先定位 `_MEIPASS/colab_pet`）。"""
    meipass_raw = getattr(sys, "_MEIPASS", None)
    if getattr(sys, "frozen", False) and meipass_raw:
        frozen_pet = (Path(meipass_raw) / "colab_pet").resolve()
        if (frozen_pet / "ui.html").is_file():
            return frozen_pet
    return Path(__file__).resolve().parent


PET_DIR = _resolve_pet_dir()
ASSETS_DIR = PET_DIR / "assets"
_meipass_root = getattr(sys, "_MEIPASS", None)
if getattr(sys, "frozen", False) and _meipass_root:
    _meipass_str = str(Path(_meipass_root).resolve())
    if _meipass_str not in sys.path:
        sys.path.insert(0, _meipass_str)
if str(PET_DIR) not in sys.path:
    sys.path.insert(0, str(PET_DIR))

import webview  # noqa: E402
from webview.dom import DOMEventHandler  # noqa: E402

from asset_server import (  # noqa: E402
    _ALLOWED_ASSET_EXTS, _PetAssetHandler, _ThreadingHTTPServer,
    _resolve_handler_assets_root, _resolve_handler_media_root, _resolve_safe_asset_path,
    _safe_file_mtime_ns, _set_active_media_root, configure_asset_roots, start_asset_http_server,
)
from colab_backend import ColabPetBackend  # noqa: E402
from colab_core.exec_runner import suppress_logged  # noqa: E402
from win32_window import (  # noqa: E402
    BASE_H, BASE_W, EDGE_BG_HEX, EDGE_BG_RGB, DotNetAction,
    _acquire_single_instance, _capture_hwnd_png, _compute_edge_x_pair, _compute_initial_pos,
    _compute_pet_opaque_rects, _init_pet_form_frame, _round_box_px, _rounded_rect_runs,
    configure_window_paths, dwmapi, kernel32_api, user32,
)
from bridge_api import (  # noqa: E402
    PetBridgeApi, _push_event_to_studio, _push_event_to_ui, _push_to_studio, _push_to_ui,
)

configure_asset_roots(PET_DIR, ASSETS_DIR)
configure_window_paths(PET_DIR, ASSETS_DIR)

__all__ = [
    "BASE_W", "BASE_H", "EDGE_BG_HEX", "EDGE_BG_RGB", "PET_DIR", "ASSETS_DIR",
    "DotNetAction", "PetBridgeApi", "_ALLOWED_ASSET_EXTS", "_PetAssetHandler",
    "_ThreadingHTTPServer", "_capture_hwnd_png", "_compute_edge_x_pair",
    "_compute_pet_opaque_rects", "_push_event_to_ui", "_push_to_ui",
    "_push_to_studio", "_push_event_to_studio", "_resolve_handler_assets_root",
    "_resolve_handler_media_root", "_resolve_safe_asset_path", "_round_box_px",
    "_rounded_rect_runs", "_set_active_media_root",
    "user32", "dwmapi", "kernel32_api", "run_app", "main",
]


def _start_pet_asset_server(backend: ColabPetBackend, api: PetBridgeApi) -> tuple[_ThreadingHTTPServer, int]:
    """启动桌宠本地回环资产与媒体 HTTP 服务，并将基础地址写入 API 实例。"""
    httpd, port = start_asset_http_server(backend.download_root.resolve())
    api._asset_base_url = f"http://127.0.0.1:{port}"
    return httpd, port


def _attach_pet_window_events(
    window: webview.Window, api: PetBridgeApi, backend: ColabPetBackend, stop_event: threading.Event
) -> None:
    """为主桌宠窗口挂载 DOM 拖拽投递与退出清理事件。"""

    def _on_dom_loaded() -> None:
        with suppress_logged("pet_ui", throttled_key="dom_drop_bind"):
            doc = window.dom.document
            doc.events.dragover += DOMEventHandler(lambda _e: None, prevent_default=True, stop_propagation=False)
            doc.events.drop += DOMEventHandler(api._on_dom_drop, prevent_default=True, stop_propagation=False)

    def _on_closed() -> None:
        stop_event.set()
        api._wake_event.set()
        api._close_studio_window_internal()
        backend.stop_monitor()

    window.events.loaded += _on_dom_loaded
    window.events.closed += _on_closed


def _reload_ui_if_modified(
    window: webview.Window, backend: ColabPetBackend, stop_event: threading.Event,
    port: int, ui_html_path: Path, last_mtime_ns: int | None,
) -> int | None:
    """检测 `ui.html` 磁盘变更并免重启无感热重载前端页面与状态快照。"""
    cur_mtime = _safe_file_mtime_ns(ui_html_path)
    if cur_mtime is None or last_mtime_ns is None or cur_mtime == last_mtime_ns:
        return last_mtime_ns if cur_mtime is None else cur_mtime
    reloaded = threading.Event()
    mark_loaded = reloaded.set
    try:
        window.events.loaded += mark_loaded
        window.load_url(f"http://127.0.0.1:{port}/ui.html?v={cur_mtime}")
        reloaded.wait(timeout=1.5)
    except Exception:
        time.sleep(0.45)
    finally:
        # 热重载结束后忽略已失效窗口的事件解绑异常
        with contextlib.suppress(AttributeError, KeyError, ValueError, RuntimeError):
            window.events.loaded -= mark_loaded
    time.sleep(0.12)
    if not stop_event.is_set():
        _push_to_ui(window, backend.get_cached_snapshot())
    return cur_mtime


def _run_test_render_capture(
    window: webview.Window, stop_event: threading.Event, test_render_sec: float, capture_path: str | None
) -> None:
    """测试渲染模式：等待指定秒数、可选截图保存后安全销毁窗口退出。"""
    time.sleep(max(1.0, float(test_render_sec)))
    if capture_path and window.native is not None:
        try:
            _capture_hwnd_png(int(window.native.Handle.ToInt64()), Path(capture_path))
        except Exception as exc:
            print(f"CAPTURE_ERROR: {exc}", file=sys.stderr)
    stop_event.set()
    # 测试渲染收尾阶段安全忽略已销毁窗口异常
    with suppress_logged("pet_ui", throttled_key="test_render_destroy"):
        window.destroy()


def _run_pet_background_loop(
    window: webview.Window, api: PetBridgeApi, backend: ColabPetBackend, stop_event: threading.Event,
    *, prefs: dict[str, Any], win_w: int, win_h: int, init_x: int, init_y: int,
    port: int, test_render_sec: float | None, capture_path: str | None,
) -> None:
    """主窗口后台工作线程：完成窗体初始化、首帧状态推送、边缘巡检与周期轮询。"""
    _init_pet_form_frame(window, prefs, win_w, win_h, init_x, init_y)
    time.sleep(0.45)
    snap = backend.fetch_live_status()
    if not stop_event.is_set():
        _push_to_ui(window, snap)
    if test_render_sec is not None:
        _run_test_render_capture(window, stop_event, test_render_sec, capture_path)
        return

    api._sync_initial_dock_edge()
    threading.Thread(target=api._edge_watch_loop, daemon=True).start()
    backend.start_monitor(lambda evt: None if stop_event.is_set() else _push_event_to_ui(window, api, evt))

    ui_html_path = PET_DIR / "ui.html"
    last_mtime_ns = _safe_file_mtime_ns(ui_html_path)
    dev_reload = (os.environ.get("COLAB_PET_DEV_RELOAD") == "1") and not bool(getattr(sys, "frozen", False))
    sec_ticks = 0
    while not stop_event.is_set():
        interval = max(15, min(600, int((backend.state.get("prefs") or {}).get("poll_interval_sec") or 60)))
        if dev_reload:
            for _ in range(interval):
                if stop_event.wait(timeout=1.0):
                    return
                sec_ticks += 1
                if sec_ticks % 2 == 0 and not stop_event.is_set():
                    last_mtime_ns = _reload_ui_if_modified(window, backend, stop_event, port, ui_html_path, last_mtime_ns)
        elif stop_event.wait(timeout=float(interval)):
            return
        snap = backend.fetch_live_status()
        if not stop_event.is_set():
            _push_to_ui(window, snap)


def _cleanup_pet_runtime(
    stop_event: threading.Event, api: PetBridgeApi, backend: ColabPetBackend, httpd: _ThreadingHTTPServer
) -> None:
    """退出收尾：置位停止信号、销毁 Studio 子窗、停监控并关闭静态 HTTP 服务。"""
    stop_event.set()
    api._close_studio_window_internal()
    backend.stop_monitor()
    # 退出收尾忽略本地静态 HTTP 服务重复关闭或套接字关闭异常
    with contextlib.suppress(OSError, RuntimeError, AttributeError):
        httpd.shutdown()
        httpd.server_close()


def run_app(
    test_render_sec: float | None = None,
    capture_path: str | None = None,
    force_restart: bool = False,
) -> int:
    """启动 Colab 桌宠桌面挂件主循环。"""
    if test_render_sec is None and not _acquire_single_instance(force_restart=force_restart):
        return 0

    backend = ColabPetBackend()
    backend._diag.info("pet", "Colab 桌宠界面进程已启动", frozen=bool(getattr(sys, "frozen", False)))
    prefs = backend.state.get("prefs") or {}
    scale = max(0.75, min(1.35, float(prefs.get("scale", 1.0))))
    win_w, win_h = int(math.ceil(BASE_W * scale)), int(math.ceil(BASE_H * scale))
    init_x, init_y = _compute_initial_pos(prefs, win_w, win_h)

    stop_event = threading.Event()
    api = PetBridgeApi(backend, stop_event)
    httpd, port = _start_pet_asset_server(backend, api)

    # 关闭 WebView2 Layered TransparencyKey，改用 Form.Region 实现真透明 + 100% 稳定鼠标交互
    window = webview.create_window(
        title="Colab 桌宠", url=f"http://127.0.0.1:{port}/ui.html", js_api=api,
        width=win_w, height=win_h, x=init_x, y=init_y,
        resizable=False, frameless=True, easy_drag=False, shadow=False,
        transparent=False, background_color=EDGE_BG_HEX, focus=True,
        on_top=bool(prefs.get("on_top", True)),
    )
    api._bind_window(window)
    _attach_pet_window_events(window, api, backend, stop_event)

    try:
        webview.start(
            func=lambda: _run_pet_background_loop(
                window, api, backend, stop_event,
                prefs=prefs, win_w=win_w, win_h=win_h, init_x=init_x, init_y=init_y,
                port=port, test_render_sec=test_render_sec, capture_path=capture_path,
            )
        )
    finally:
        _cleanup_pet_runtime(stop_event, api, backend, httpd)
    return 0


def main() -> int:
    """解析命令行参数并调度体检、诊断导出或桌宠 GUI 主程序。"""
    parser = argparse.ArgumentParser(description="Colab Q 版桌宠悬浮挂件")
    parser.add_argument("--test-render", type=float, default=None, help="测试渲染指定秒数后自动退出")
    parser.add_argument("--capture", type=str, default=None, help="配合 --test-render 将窗口渲染结果保存为 PNG")
    parser.add_argument("--restart", action="store_true", help="若已有桌宠实例运行，优雅关闭旧窗口并以最新代码重启")
    parser.add_argument("--doctor", action="store_true", help="执行启动前全维度环境体检并打印报告后退出")
    parser.add_argument("--export-diag", action="store_true", help="立即导出脱敏排障诊断包 (.zip) 并打印路径后退出")
    args = parser.parse_args()
    if args.doctor or args.export_diag:
        import colab_ctl

        return colab_ctl.main(["doctor"] if args.doctor else ["diag"])
    return run_app(test_render_sec=args.test_render, capture_path=args.capture, force_restart=bool(args.restart))


if __name__ == "__main__":
    raise SystemExit(main())
