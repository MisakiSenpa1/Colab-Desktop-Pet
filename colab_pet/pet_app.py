"""Colab 桌宠主程序（基于 pywebview + Edge WebView2 + Win32 区域裁剪无边框悬浮窗）。

核心机制：
1. 使用 Form.Region (SetWindowRgn) 将窗口精确裁剪为“气泡 + 思维圆点 + 小人轮廓 + 悬浮栏”，
   彻底避开 WinForms TransparencyKey 与 WebView2 DirectComposition 冲突导致的鼠标穿透失效；
2. 空白裁剪区由系统内核直接丢弃并穿透到底层桌面，窗口内部全部事件 100% 正常响应；
3. 拖拽超出 4px 后移交 Win32 GetCursorPos + SetWindowPos 原生高帧率跟踪，跨屏快拖不脱手。
"""

from __future__ import annotations

import argparse
import ctypes
import http.server
import io
import json
import math
import mimetypes
import os
import re
import socketserver
import sys
import threading
import time
import urllib.parse
from collections.abc import Callable
from ctypes import wintypes
from pathlib import Path
from typing import Any, TextIO

# 持有空流兜底句柄，防止 pythonw.exe 直接运行 pet_app.py 时 sys.stdout/sys.stderr 为 None
_DEVNULL_HANDLES: list[TextIO] = []


def _ensure_std_streams() -> None:
    """在 pythonw.exe 无控制台环境下确保 stdout/stderr 可安全写入。"""
    for attr in ("stdout", "stderr"):
        if getattr(sys, attr, None) is None:
            handle = open(os.devnull, "w", encoding="utf-8", errors="replace")
            _DEVNULL_HANDLES.append(handle)
            setattr(sys, attr, handle)


_ensure_std_streams()

import clr
from PIL import Image, ImageDraw

clr.AddReference("System.Drawing")
clr.AddReference("System.Windows.Forms")
from System import Action as DotNetAction, IntPtr  # type: ignore
from System.Drawing import Color, Rectangle, Region  # type: ignore
from System.Drawing.Drawing2D import GraphicsPath  # type: ignore
from System.Windows.Forms import Clipboard, Screen  # type: ignore
import webview
from webview.dom import DOMEventHandler, _dnd_state

PET_DIR = Path(__file__).resolve().parent
ASSETS_DIR = PET_DIR / "assets"
if str(PET_DIR) not in sys.path:
    sys.path.insert(0, str(PET_DIR))

from colab_backend import ColabPetBackend  # noqa: E402

BASE_W = 392
BASE_H = 548
# 与人物立绘深靛蓝描边一致，在 Region 像素轮廓边缘无缝融合
EDGE_BG_HEX = "#221F52"
EDGE_BG_RGB = (34, 31, 82)

# 模块级独立 user32 实例与 64 位句柄安全参数签名，避免污染全局 ctypes.windll.user32 影响 pywebview
user32 = ctypes.WinDLL("user32")
user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
user32.GetWindowRect.restype = wintypes.BOOL
user32.SetWindowPos.argtypes = [
    wintypes.HWND,
    wintypes.HWND,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    wintypes.UINT,
]
user32.SetWindowPos.restype = wintypes.BOOL
user32.GetCursorPos.argtypes = [ctypes.POINTER(wintypes.POINT)]
user32.GetCursorPos.restype = wintypes.BOOL
user32.GetDpiForWindow.argtypes = [wintypes.HWND]
user32.GetDpiForWindow.restype = wintypes.UINT
user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
user32.GetWindowThreadProcessId.restype = wintypes.DWORD
user32.PostMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
user32.PostMessageW.restype = wintypes.BOOL
user32.IsWindow.argtypes = [wintypes.HWND]
user32.IsWindow.restype = wintypes.BOOL

_FILETIME_UNIX_EPOCH_100NS = 116444736000000000
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_WM_CLOSE = 0x0010


class _FILETIME(ctypes.Structure):
    """Win32 FILETIME 64 位时间戳结构体（100ns 粒度，自 1601-01-01 UTC 起算）。"""

    _fields_ = [
        ("dwLowDateTime", wintypes.DWORD),
        ("dwHighDateTime", wintypes.DWORD),
    ]


kernel32_api = ctypes.WinDLL("kernel32", use_last_error=True)
kernel32_api.CreateMutexW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR]
kernel32_api.CreateMutexW.restype = wintypes.HANDLE
kernel32_api.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
kernel32_api.OpenProcess.restype = wintypes.HANDLE
kernel32_api.GetProcessTimes.argtypes = [
    wintypes.HANDLE,
    ctypes.POINTER(_FILETIME),
    ctypes.POINTER(_FILETIME),
    ctypes.POINTER(_FILETIME),
    ctypes.POINTER(_FILETIME),
]
kernel32_api.GetProcessTimes.restype = wintypes.BOOL
kernel32_api.CloseHandle.argtypes = [wintypes.HANDLE]
kernel32_api.CloseHandle.restype = wintypes.BOOL

_MUTEX_HANDLE = None
_ALPHA_THRESHOLD = 152
_ALPHA_LUT = [0] * _ALPHA_THRESHOLD + [255] * (256 - _ALPHA_THRESHOLD)
_OPAQUE_RUN_RE = re.compile(b"\xff+")
_SPRITE_ALPHA_CACHE: dict[str, Image.Image] = {}
_FALLBACK_SPRITE_CACHE: dict[str, tuple[Image.Image, bytes]] = {}
_MASK_RUNS_CACHE: dict[tuple[int, int, bool, bool], list[tuple[int, int, int, int]]] = {}


def _render_fallback_sprite(mood: str) -> tuple[Image.Image, bytes]:
    """当未放入自定义立绘 PNG 时，生成 500x418 原创极简矢量徽章兜底图及其 PNG 字节。"""
    key = mood if mood in ("idle", "blink", "proud", "sad") else "idle"
    cached = _FALLBACK_SPRITE_CACHE.get(key)
    if cached is not None:
        return cached

    img = Image.new("RGBA", (500, 418), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    outline = (*EDGE_BG_RGB, 255)
    body_fill = (242, 236, 255, 255) if key != "sad" else (228, 224, 245, 255)
    inner_fill = (255, 255, 255, 255)

    # 双猫耳外轮廓与头部主体
    left_ear = [(136, 178), (112, 76), (206, 128)]
    right_ear = [(364, 178), (388, 76), (294, 128)]
    draw.polygon(left_ear, fill=body_fill, outline=outline)
    draw.polygon(right_ear, fill=body_fill, outline=outline)
    draw.line([left_ear[0], left_ear[1], left_ear[2]], fill=outline, width=14, joint="curve")
    draw.line([right_ear[0], right_ear[1], right_ear[2]], fill=outline, width=14, joint="curve")
    draw.polygon([(142, 162), (128, 102), (186, 134)], fill=(214, 193, 252, 255))
    draw.polygon([(358, 162), (372, 102), (314, 134)], fill=(214, 193, 252, 255))

    # 主体圆润躯干 + 头顶小星标
    draw.ellipse((116, 108, 384, 344), fill=body_fill, outline=outline, width=14)
    draw.ellipse((134, 294, 366, 408), fill=body_fill, outline=outline, width=14)
    draw.ellipse((160, 314, 340, 396), fill=inner_fill)
    draw.ellipse((234, 72, 266, 104), fill=(253, 224, 71, 255), outline=outline, width=7)

    # 腮红与面部表情
    cheek = (244, 164, 196, 200) if key != "sad" else (186, 192, 224, 180)
    draw.ellipse((164, 232, 198, 254), fill=cheek)
    draw.ellipse((302, 232, 336, 254), fill=cheek)

    eye_col = (56, 38, 104, 255)
    if key == "blink":
        draw.arc((184, 206, 224, 230), 0, 180, fill=eye_col, width=7)
        draw.arc((276, 206, 316, 230), 0, 180, fill=eye_col, width=7)
        draw.arc((236, 238, 264, 256), 15, 165, fill=eye_col, width=5)
    elif key == "proud":
        draw.arc((184, 196, 224, 226), 190, 350, fill=eye_col, width=7)
        draw.arc((276, 196, 316, 226), 190, 350, fill=eye_col, width=7)
        draw.arc((230, 236, 270, 262), 10, 170, fill=eye_col, width=6)
    elif key == "sad":
        draw.line([(186, 212), (222, 222)], fill=eye_col, width=7)
        draw.line([(314, 212), (278, 222)], fill=eye_col, width=7)
        draw.arc((232, 248, 268, 272), 195, 345, fill=eye_col, width=6)
    else:
        draw.ellipse((188, 194, 218, 228), fill=eye_col)
        draw.ellipse((282, 194, 312, 228), fill=eye_col)
        draw.ellipse((194, 200, 204, 210), fill=(255, 255, 255, 255))
        draw.ellipse((288, 200, 298, 210), fill=(255, 255, 255, 255))
        draw.arc((234, 236, 266, 256), 15, 165, fill=eye_col, width=5)

    # 胸前 Colab 算力菱形小徽标
    diamond = [(250, 330), (270, 350), (250, 370), (230, 350)]
    draw.polygon(diamond, fill=(124, 58, 237, 255), outline=outline)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    res = (img, buf.getvalue())
    _FALLBACK_SPRITE_CACHE[key] = res
    return res


def _current_desktop_tag() -> str:
    """获取当前线程所属 Win32 桌面名称并清洗为互斥量后缀，避免跨桌面误锁。"""
    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32
    raw_name = ""
    try:
        h_desk = user32.GetThreadDesktop(kernel32.GetCurrentThreadId())
        if h_desk:
            buf = ctypes.create_unicode_buffer(256)
            needed = wintypes.DWORD()
            if user32.GetUserObjectInformationW(
                h_desk, 2, buf, ctypes.sizeof(buf), ctypes.byref(needed)
            ):
                raw_name = buf.value or ""
    except Exception:
        raw_name = ""
    cleaned = re.sub(r"[^A-Za-z0-9_-]", "_", raw_name).strip("_")
    return cleaned or "Default"


def _find_existing_pet_hwnd(wait_sec: float = 2.2) -> int:
    """轮询寻找当前桌面中已运行的 Colab 桌宠窗口句柄，未找到返回 0。"""
    u32 = ctypes.windll.user32
    deadline = time.monotonic() + max(0.2, wait_sec)
    while True:
        hwnd = int(u32.FindWindowW(None, "Colab 桌宠") or 0)
        if hwnd:
            return hwnd
        if time.monotonic() >= deadline:
            return 0
        time.sleep(0.15)


def _get_hwnd_process_start_epoch(hwnd: int) -> float | None:
    """查询窗口所属进程的启动时间（UNIX 秒），用于判定磁盘源码是否比运行中实例更新。"""
    if not hwnd:
        return None
    pid = wintypes.DWORD(0)
    try:
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    except Exception:
        return None
    if not pid.value or int(pid.value) == os.getpid():
        return None

    h_proc = kernel32_api.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid.value)
    if not h_proc:
        return None
    try:
        ft_create = _FILETIME()
        ft_exit = _FILETIME()
        ft_kernel = _FILETIME()
        ft_user = _FILETIME()
        ok = bool(
            kernel32_api.GetProcessTimes(
                h_proc,
                ctypes.byref(ft_create),
                ctypes.byref(ft_exit),
                ctypes.byref(ft_kernel),
                ctypes.byref(ft_user),
            )
        )
        if not ok:
            return None
        ticks_100ns = (int(ft_create.dwHighDateTime) << 32) | int(ft_create.dwLowDateTime)
        if ticks_100ns <= _FILETIME_UNIX_EPOCH_100NS:
            return None
        return (ticks_100ns - _FILETIME_UNIX_EPOCH_100NS) / 10_000_000.0
    except Exception:
        return None
    finally:
        kernel32_api.CloseHandle(h_proc)


def _latest_pet_source_mtime() -> float:
    """获取 pet_app.py 与 ui.html 的最新磁盘修改时间戳（秒）。"""
    latest = 0.0
    for fname in ("pet_app.py", "ui.html"):
        try:
            latest = max(latest, float((PET_DIR / fname).stat().st_mtime))
        except OSError:
            pass
    return latest


def _safe_file_mtime_ns(path: Path) -> int | None:
    """安全读取非空文件的纳秒级修改时间，忽略编辑器瞬时写入竞争或文件缺失。"""
    try:
        st = path.stat()
        return int(st.st_mtime_ns) if st.st_size > 0 else None
    except OSError:
        return None


def _restore_and_focus_pet_hwnd(hwnd: int) -> bool:
    """将已存在的桌宠窗口恢复显示、校正屏幕工作区越界并激活到前台。"""
    if not hwnd:
        return False
    u32 = ctypes.windll.user32
    if u32.IsIconic(hwnd):
        u32.ShowWindow(hwnd, 9)  # SW_RESTORE
    else:
        u32.ShowWindow(hwnd, 5)  # SW_SHOW

    # 检查窗口是否在任一显示器工作区内可见；若跑出屏幕或右侧/底部越界过大则拉回安全区
    rect = wintypes.RECT()
    if user32.GetWindowRect(hwnd, ctypes.byref(rect)):
        win_w = max(BASE_W, int(rect.right - rect.left))
        win_h = max(BASE_H, int(rect.bottom - rect.top))
        in_view = False
        try:
            for scr in Screen.AllScreens:
                b = scr.WorkingArea
                inter_w = min(rect.right, b.Right) - max(rect.left, b.Left)
                inter_h = min(rect.bottom, b.Bottom) - max(rect.top, b.Top)
                if inter_w >= 80 and inter_h >= 80:
                    in_view = True
                    break
            wa = None
            try:
                wa = Screen.FromHandle(IntPtr(hwnd)).WorkingArea
                if wa.Width <= 0 or wa.Height <= 0:
                    wa = None
            except Exception:
                wa = None
            if wa is None:
                wa = Screen.PrimaryScreen.WorkingArea
            if not in_view:
                wa = Screen.PrimaryScreen.WorkingArea
                safe_x = max(wa.Left + 4, wa.Right - win_w - 16)
                safe_y = max(wa.Top + 4, wa.Bottom - win_h - 10)
                # SWP_NOSIZE (0x0001) | SWP_NOZORDER (0x0004) | SWP_SHOWWINDOW (0x0040)
                user32.SetWindowPos(hwnd, 0, safe_x, safe_y, 0, 0, 0x0001 | 0x0004 | 0x0040)
            else:
                overflow_right = rect.right - wa.Right
                overflow_bottom = rect.bottom - wa.Bottom
                if (
                    overflow_right > 16
                    or overflow_bottom > 16
                    or rect.left < wa.Left - 24
                    or rect.top < wa.Top - 16
                ):
                    safe_x = max(wa.Left + 4, min(int(rect.left), wa.Right - win_w - 4))
                    safe_y = max(wa.Top + 4, min(int(rect.top), wa.Bottom - win_h + 4))
                    user32.SetWindowPos(hwnd, 0, safe_x, safe_y, 0, 0, 0x0001 | 0x0004 | 0x0040)
        except Exception:
            pass

    u32.SetForegroundWindow(hwnd)
    return True


def _activate_existing_pet_window(wait_sec: float = 2.2) -> bool:
    """寻找当前桌面中已运行的 Colab 桌宠窗口，将其恢复显示并拉回可见桌面。"""
    hwnd = _find_existing_pet_hwnd(wait_sec=wait_sec)
    if not hwnd:
        return False
    return _restore_and_focus_pet_hwnd(hwnd)


def _acquire_single_instance(
    name: str = "Local\\ColabDesktopPetSingletonMutex",
    force_restart: bool = False,
) -> bool:
    """使用按桌面隔离的 Win32 互斥量确保单实例，支持代码更新后重双击优雅热替换重启。"""
    global _MUTEX_HANDLE
    scoped_name = f"{name}_{_current_desktop_tag()}"
    ctypes.set_last_error(0)
    handle = kernel32_api.CreateMutexW(None, False, scoped_name)
    err = ctypes.get_last_error()
    if err != 183:  # ERROR_ALREADY_EXISTS
        _MUTEX_HANDLE = handle
        return True

    hwnd = _find_existing_pet_hwnd(wait_sec=2.2)
    if not hwnd:
        # 超时仍未找到窗口（旧进程无窗僵死残留），保留互斥句柄并允许新实例接管启动
        _MUTEX_HANDLE = handle
        return True

    should_replace = bool(force_restart)
    if not should_replace:
        proc_start_ts = _get_hwnd_process_start_epoch(hwnd)
        latest_mtime = _latest_pet_source_mtime()
        if proc_start_ts is not None and latest_mtime > (proc_start_ts + 1.5):
            should_replace = True

    if should_replace:
        # 先关闭当前进程持有的冲突句柄，以便旧实例退出后内核立即销毁旧命名互斥量
        if handle:
            kernel32_api.CloseHandle(handle)
            handle = None
        user32.PostMessageW(hwnd, _WM_CLOSE, 0, 0)
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            time.sleep(0.08)
            ctypes.set_last_error(0)
            new_h = kernel32_api.CreateMutexW(None, False, scoped_name)
            new_err = ctypes.get_last_error()
            if new_h and new_err != 183:
                _MUTEX_HANDLE = new_h
                return True
            if new_h:
                kernel32_api.CloseHandle(new_h)
        if not user32.IsWindow(hwnd):
            ctypes.set_last_error(0)
            fallback_h = kernel32_api.CreateMutexW(None, False, scoped_name)
            _MUTEX_HANDLE = fallback_h
            return True
        _restore_and_focus_pet_hwnd(hwnd)
        return False

    # 代码未发生更新且未传 --restart：直接唤醒并拉回现有桌宠窗口
    _restore_and_focus_pet_hwnd(hwnd)
    if handle:
        kernel32_api.CloseHandle(handle)
    return False


def _get_alpha_template(is_blink: bool) -> Image.Image | None:
    """懒加载并缓存预缩放 Alpha 模板，分离常驻帧与眨眼帧以避免多余暗边。"""
    key = "blink" if is_blink else "base"
    if key in _SPRITE_ALPHA_CACHE:
        return _SPRITE_ALPHA_CACHE[key]

    candidates = ("blink.png", "idle.png") if is_blink else ("idle.png",)
    for fname in candidates:
        p = ASSETS_DIR / fname
        if p.exists():
            alpha = (
                Image.open(p)
                .convert("RGBA")
                .getchannel("A")
                .resize((500, 418), Image.Resampling.LANCZOS)
            )
            _SPRITE_ALPHA_CACHE[key] = alpha
            return alpha

    # 未放置自定义立绘时，提取内置兜底矢量立绘的 Alpha 轮廓以确保边缘裁剪完整
    fb_img, _ = _render_fallback_sprite("blink" if is_blink else "idle")
    alpha = fb_img.getchannel("A")
    _SPRITE_ALPHA_CACHE[key] = alpha
    return alpha


def _sprite_mask_runs(
    w: int, h: int, flipped: bool, is_blink: bool = False
) -> list[tuple[int, int, int, int]]:
    """计算指定物理宽高下的立绘轮廓扫描行矩形 (x, y, rw, rh)，单帧耗时 <1.5ms。"""
    key = (w, h, flipped, is_blink)
    if key in _MASK_RUNS_CACHE:
        return _MASK_RUNS_CACHE[key]

    tmpl = _get_alpha_template(is_blink)
    if tmpl is None or w <= 0 or h <= 0:
        return [(0, 0, max(1, w), max(1, h))]

    alpha = tmpl.resize((w, h), Image.Resampling.BILINEAR)
    if flipped:
        alpha = alpha.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    bw = alpha.point(_ALPHA_LUT).tobytes()

    # 利用 C 级字节正则扫出每行连续不透明区间，并合并垂直方向完全相同的矩形
    rects: list[tuple[int, int, int, int]] = []
    active: dict[tuple[int, int], int] = {}
    for yy in range(h):
        row = bw[yy * w : (yy + 1) * w]
        runs = [(m.start(), m.end()) for m in _OPAQUE_RUN_RE.finditer(row)]
        cur_set = set(runs)
        for seg in list(active.keys()):
            if seg not in cur_set:
                y0 = active.pop(seg)
                rects.append((seg[0], y0, seg[1] - seg[0], yy - y0))
        for seg in runs:
            if seg not in active:
                active[seg] = yy
    for seg, y0 in active.items():
        rects.append((seg[0], y0, seg[1] - seg[0], h - y0))

    if len(_MASK_RUNS_CACHE) >= 256:
        _MASK_RUNS_CACHE.clear()
    _MASK_RUNS_CACHE[key] = rects
    return rects


def _round_box_px(box: dict[str, Any], dpi: float) -> tuple[int, int, int, int]:
    """将 CSS 浮点矩形严格四舍五入对齐到物理整数像素 (ix0, iy0, iw, ih)，消除半像素切点台阶。"""
    ix0 = int(round(float(box["x"]) * dpi))
    iy0 = int(round(float(box["y"]) * dpi))
    ix1 = int(round((float(box["x"]) + float(box["w"])) * dpi))
    iy1 = int(round((float(box["y"]) + float(box["h"])) * dpi))
    return ix0, iy0, max(0, ix1 - ix0), max(0, iy1 - iy0)


def _rounded_rect_runs(
    fx0: float, fy0: float, fx1: float, fy1: float, radius: float
) -> list[tuple[int, int, int, int]]:
    """按像素中心采样精确计算圆角矩形的水平扫描条，并合并相邻同宽行以消除 GDI+ 圆弧弦近似锯齿。"""
    if fx1 <= fx0 or fy1 <= fy0:
        return []
    w = fx1 - fx0
    h = fy1 - fy0
    r = max(0.0, min(float(radius), w * 0.5, h * 0.5))
    y_start = int(math.floor(fy0))
    y_end = int(math.ceil(fy1))
    runs: list[tuple[int, int, int, int]] = []
    for py in range(y_start, y_end + 1):
        yc = py + 0.5
        if yc < fy0 or yc > fy1:
            continue
        if r > 0.0 and yc < fy0 + r:
            dy = (fy0 + r) - yc
            dx = math.sqrt(max(0.0, r * r - dy * dy))
            lx = fx0 + r - dx
            rx = fx1 - r + dx
        elif r > 0.0 and yc > fy1 - r:
            dy = yc - (fy1 - r)
            dx = math.sqrt(max(0.0, r * r - dy * dy))
            lx = fx0 + r - dx
            rx = fx1 - r + dx
        else:
            lx = fx0
            rx = fx1
        px0 = int(math.ceil(lx - 0.5))
        px1 = int(math.floor(rx - 0.5))
        if px1 < px0:
            continue
        rw = px1 - px0 + 1
        if runs and runs[-1][0] == px0 and runs[-1][2] == rw and (runs[-1][1] + runs[-1][3]) == py:
            prev = runs[-1]
            runs[-1] = (prev[0], prev[1], prev[2], prev[3] + 1)
        else:
            runs.append((px0, py, rw, 1))
    return runs


def _add_smooth_rounded_rect(
    path: GraphicsPath, fx0: float, fy0: float, fw: float, fh: float, radius: float
) -> None:
    """将高精度像素对齐的圆角矩形扫描条写入 .NET GraphicsPath。"""
    if fw <= 0 or fh <= 0:
        return
    from System.Drawing import Rectangle

    for rx, ry, rw, rh in _rounded_rect_runs(fx0, fy0, fx0 + fw, fy0 + fh, radius):
        path.AddRectangle(Rectangle(int(rx), int(ry), int(rw), int(rh)))


def _compute_edge_x_pair(
    dock_edge: str | None,
    wa_left: int,
    wa_right: int,
    w_phys: int,
    ui_scale: float,
    dpi: float,
) -> tuple[int, int]:
    """计算左右停靠边的全显水平坐标与贴边半隐藏（保留约 68px 头像露头）坐标。"""
    edge = str(dock_edge or "").strip().lower()
    factor = max(0.65, min(1.35, float(ui_scale or 1.0))) * max(0.5, float(dpi or 1.0))
    pad = int(round(4.0 * max(0.5, float(dpi or 1.0))))
    if edge == "right":
        x_visible = int(wa_right) - int(w_phys) - pad
        x_peek = int(wa_right) - int(round(140.0 * factor))
        return x_visible, x_peek
    x_visible = int(wa_left) + pad
    x_peek = int(wa_left) - int(round(251.0 * factor))
    return x_visible, x_peek


class _PetAssetHandler(http.server.BaseHTTPRequestHandler):
    """只读提供 ui.html 与 assets/ 下静态图片资源。"""

    def do_GET(self) -> None:  # noqa: N802
        clean = self.path.split("?", 1)[0]
        if clean in ("/", "/ui.html"):
            target = (PET_DIR / "ui.html").resolve()
        elif clean.startswith("/assets/"):
            rel = clean.lstrip("/")
            target = (PET_DIR / rel).resolve()
        else:
            self.send_error(404)
            return

        try:
            target.relative_to(PET_DIR.resolve())
        except ValueError:
            self.send_error(403)
            return

        if not target.is_file():
            if clean in ("/assets/idle.png", "/assets/blink.png", "/assets/proud.png", "/assets/sad.png"):
                mood = Path(clean).stem
                _, data = _render_fallback_sprite(mood)
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)
                return
            self.send_error(404)
            return

        ctype, _ = mimetypes.guess_type(str(target))
        data = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype or "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: Any) -> None:
        """静默 HTTP 访问日志。"""


class _ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


def _capture_hwnd_png(hwnd: int, out_path: Path) -> None:
    """截取窗口当前可见渲染区域，并按 GetWindowRgn 真实裁剪区域合成为白底 PNG。"""
    u32 = ctypes.WinDLL("user32")
    g32 = ctypes.WinDLL("gdi32")
    u32.GetWindowDC.argtypes = [wintypes.HWND]
    u32.GetWindowDC.restype = wintypes.HDC
    u32.ReleaseDC.argtypes = [wintypes.HWND, wintypes.HDC]
    u32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    u32.PrintWindow.argtypes = [wintypes.HWND, wintypes.HDC, wintypes.UINT]
    u32.GetWindowRgn.argtypes = [wintypes.HWND, wintypes.HRGN]
    g32.CreateCompatibleDC.argtypes = [wintypes.HDC]
    g32.CreateCompatibleDC.restype = wintypes.HDC
    g32.CreateCompatibleBitmap.argtypes = [wintypes.HDC, ctypes.c_int, ctypes.c_int]
    g32.CreateCompatibleBitmap.restype = wintypes.HBITMAP
    g32.SelectObject.argtypes = [wintypes.HDC, wintypes.HGDIOBJ]
    g32.SelectObject.restype = wintypes.HGDIOBJ
    g32.DeleteObject.argtypes = [wintypes.HGDIOBJ]
    g32.DeleteDC.argtypes = [wintypes.HDC]
    g32.CreateRectRgn.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    g32.CreateRectRgn.restype = wintypes.HRGN
    g32.CreateSolidBrush.argtypes = [wintypes.COLORREF]
    g32.CreateSolidBrush.restype = wintypes.HBRUSH
    g32.FillRgn.argtypes = [wintypes.HDC, wintypes.HRGN, wintypes.HBRUSH]
    g32.PatBlt.argtypes = [wintypes.HDC, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, wintypes.DWORD]
    g32.GetDIBits.argtypes = [
        wintypes.HDC,
        wintypes.HBITMAP,
        wintypes.UINT,
        wintypes.UINT,
        ctypes.c_void_p,
        ctypes.c_void_p,
        wintypes.UINT,
    ]
    g32.GetDIBits.restype = ctypes.c_int

    rect = wintypes.RECT()
    u32.GetWindowRect(hwnd, ctypes.byref(rect))
    w = max(1, rect.right - rect.left)
    h = max(1, rect.bottom - rect.top)

    hdc_win = u32.GetWindowDC(hwnd)
    hdc_mem = g32.CreateCompatibleDC(hdc_win)
    hbmp = g32.CreateCompatibleBitmap(hdc_win, w, h)
    old = g32.SelectObject(hdc_mem, hbmp)
    u32.PrintWindow(hwnd, hdc_mem, 2)

    class BITMAPINFOHEADER(ctypes.Structure):
        _fields_ = [
            ("biSize", wintypes.DWORD),
            ("biWidth", wintypes.LONG),
            ("biHeight", wintypes.LONG),
            ("biPlanes", wintypes.WORD),
            ("biBitCount", wintypes.WORD),
            ("biCompression", wintypes.DWORD),
            ("biSizeImage", wintypes.DWORD),
            ("biXPelsPerMeter", wintypes.LONG),
            ("biYPelsPerMeter", wintypes.LONG),
            ("biClrUsed", wintypes.DWORD),
            ("biClrImportant", wintypes.DWORD),
        ]

    bmi = BITMAPINFOHEADER()
    bmi.biSize = ctypes.sizeof(BITMAPINFOHEADER)
    bmi.biWidth = w
    bmi.biHeight = -h
    bmi.biPlanes = 1
    bmi.biBitCount = 32
    buf = ctypes.create_string_buffer(w * h * 4)
    g32.GetDIBits(hdc_mem, hbmp, 0, h, buf, ctypes.byref(bmi), 0)

    win_img = Image.frombuffer("RGBA", (w, h), buf, "raw", "BGRA", 0, 1).copy()

    # 读取窗口当前 Form.Region 区域，将裁剪区外部合成纯白以 1:1 复现桌面显示边缘
    hrgn = g32.CreateRectRgn(0, 0, 0, 0)
    rgn_type = u32.GetWindowRgn(hwnd, hrgn)
    if rgn_type in (2, 3):  # SIMPLEREGION=2, COMPLEXREGION=3
        hdc_mask = g32.CreateCompatibleDC(hdc_win)
        hbmp_mask = g32.CreateCompatibleBitmap(hdc_win, w, h)
        old_mask = g32.SelectObject(hdc_mask, hbmp_mask)
        g32.PatBlt(hdc_mask, 0, 0, w, h, 0x00000042)  # BLACKNESS
        hbrush_white = g32.CreateSolidBrush(0x00FFFFFF)
        g32.FillRgn(hdc_mask, hrgn, hbrush_white)
        mask_buf = ctypes.create_string_buffer(w * h * 4)
        g32.GetDIBits(hdc_mask, hbmp_mask, 0, h, mask_buf, ctypes.byref(bmi), 0)
        g32.SelectObject(hdc_mask, old_mask)
        g32.DeleteObject(hbrush_white)
        g32.DeleteObject(hbmp_mask)
        g32.DeleteDC(hdc_mask)

        mask_l = Image.frombuffer("RGBA", (w, h), mask_buf, "raw", "BGRA", 0, 1).getchannel("R")
        white_bg = Image.new("RGBA", (w, h), (255, 255, 255, 255))
        r_ch, g_ch, b_ch, _ = win_img.split()
        solid_win = Image.merge("RGBA", (r_ch, g_ch, b_ch, Image.new("L", (w, h), 255)))
        win_img = Image.composite(solid_win, white_bg, mask_l)

    g32.DeleteObject(hrgn)
    g32.SelectObject(hdc_mem, old)
    g32.DeleteObject(hbmp)
    g32.DeleteDC(hdc_mem)
    u32.ReleaseDC(hwnd, hdc_win)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    win_img.save(out_path)


class PetBridgeApi:
    """暴露给前端页面的 window.pywebview.api 控制接口。

    注意：所有内部成员与私有辅助方法均以下划线开头，防止 pywebview 递归遍历
    WinForms 对象属性导致栈溢出。
    """

    def __init__(self, backend: ColabPetBackend, stop_event: threading.Event) -> None:
        self._backend = backend
        self._stop_event = stop_event
        self._window: webview.Window | None = None
        self._dragging = False
        self._in_dialog = False
        self._test_mode = False
        self._edge_peeking = False
        self._edge_animating = False
        self._hover_out_since: float | None = None
        self._peek_wake_until: float = time.monotonic() + 2.5
        self._edge_lock = threading.Lock()
        self._wake_event = threading.Event()
        self._visible_boxes_px: list[tuple[int, int, int, int]] = []

    def _bind_window(self, window: webview.Window) -> None:
        self._window = window

    def _dpi_scale(self) -> float:
        hwnd = self._native_hwnd()
        if hwnd:
            try:
                dpi_val = int(user32.GetDpiForWindow(hwnd) or 0)
                if dpi_val > 0:
                    return max(1.0, float(dpi_val) / 96.0)
            except Exception:
                pass
        return 1.0

    def get_state(self) -> dict[str, Any]:
        return self._backend.get_cached_snapshot()

    def refresh_now(self) -> dict[str, Any]:
        self._backend.touch_activity()
        return self._backend.fetch_live_status()

    def _apply_clipboard_and_redact(self, res: dict[str, Any]) -> dict[str, Any]:
        """若后端结果含非空剪贴板文本则写入系统剪贴板，并剥离原文防止敏感密钥进入前端日志。"""
        if not isinstance(res, dict):
            return res
        out = dict(res)
        clip_text = out.pop("clipboard_text", None)
        if isinstance(clip_text, str) and clip_text:
            self._set_clipboard(clip_text)
        return out

    def open_comfy_tunnel(self) -> dict[str, Any]:
        return self._apply_clipboard_and_redact(self._backend.open_comfy_tunnel())

    def open_comfy_workflow(self, workflow_name: str) -> dict[str, Any]:
        """按文件名直铺指定工作流到云端 ComfyUI 画布。"""
        return self._apply_clipboard_and_redact(
            self._backend.open_comfy_tunnel(workflow_name=workflow_name)
        )

    def open_colab_url(self) -> dict[str, Any]:
        return self._backend.open_colab_url()

    def sync_outputs(self) -> dict[str, Any]:
        return self._apply_clipboard_and_redact(self._backend.sync_outputs())

    def copy_llm_api_config(self) -> dict[str, Any]:
        """一键复制当前在线 Qwen3.8 实例的 OpenAI 接口配置并脱敏返回前端。"""
        return self._apply_clipboard_and_redact(self._backend.get_llm_api_bundle("comfy"))

    def stop_all_instances(self) -> dict[str, Any]:
        return self._backend.stop_all_instances()

    def start_instance(self, config: dict[str, Any] | None = None) -> dict[str, Any]:
        """按所选生态与显卡一键开机部署并直播 7 步进度。"""
        return self._backend.start_instance(config or {})

    def cancel_deploy(self) -> dict[str, Any]:
        """主动中止当前正在执行的一键开机部署任务。"""
        return self._backend.cancel_deploy()

    def _set_clipboard(self, text: str) -> None:
        """在 WinForms UI 线程安全写入剪贴板文本。"""
        if not text or not self._window or not self._window.native:
            return
        form = self._window.native
        try:
            form.Invoke(DotNetAction(lambda: Clipboard.SetText(str(text))))
        except Exception:
            pass

    def _on_dom_drop(self, event: Any) -> None:
        """响应 Edge WebView2 DOM 原生 drop 事件，提取真实磁盘路径并投递后端。"""
        if not isinstance(event, dict):
            return
        files = (event.get("dataTransfer") or {}).get("files") or []
        paths: list[str] = []
        dnd_paths = _dnd_state.get("paths") or []
        for fobj in files:
            if not isinstance(fobj, dict):
                continue
            full_path = fobj.get("pywebviewFullPath")
            if full_path:
                paths.append(str(full_path))
                continue
            fname = str(fobj.get("name") or "")
            matched = [
                item
                for item in dnd_paths
                if item and (item[0] == fname or urllib.parse.unquote(item[0]) == fname)
            ]
            if matched:
                paths.append(urllib.parse.unquote(str(matched[0][1])))
                try:
                    dnd_paths.remove(matched[0])
                except ValueError:
                    pass
        if paths:
            self._backend.submit_dropped_files(paths)

    def pick_and_upload_files(self) -> dict[str, Any]:
        """打开本地文件选择器：按当前活跃服务智能切换 ComfyUI 素材或 LLM 文本/图片后缀。"""
        if not self._window:
            return {"ok": False, "message": "窗口未就绪"}
        snap = self._backend.get_cached_snapshot()
        is_llm = (
            str((snap.get("comfy") or {}).get("service_kind") or "") == "llm"
            if snap.get("is_running")
            else self._backend.is_llm_mode_active()
        )
        if is_llm:
            dialog_file_types = (
                "文本与识图文件 (*.txt;*.md;*.json;*.py;*.yaml;*.yml;*.csv;*.png;*.jpg;*.jpeg;*.webp)",
                "纯文本与代码 (*.txt;*.md;*.json;*.py;*.yaml;*.yml;*.csv)",
                "识图素材 (*.png;*.jpg;*.jpeg;*.webp;*.gif;*.bmp)",
                "所有文件 (*.*)",
            )
        else:
            dialog_file_types = (
                "常用投递文件 (*.png;*.jpg;*.jpeg;*.webp;*.mp4;*.json;*.safetensors;*.pt)",
                "所有文件 (*.*)",
            )
        self._in_dialog = True
        self._peek_wake_until = max(self._peek_wake_until, time.monotonic() + 4.0)
        try:
            selected = self._window.create_file_dialog(
                webview.FileDialog.OPEN,
                allow_multiple=True,
                file_types=dialog_file_types,
            )
        finally:
            self._in_dialog = False
            self._peek_wake_until = max(self._peek_wake_until, time.monotonic() + 2.0)
            self._wake_event.set()
        if not selected:
            return {"ok": False, "cancelled": True, "message": "已取消选择文件"}
        res = self._backend.submit_dropped_files([str(p) for p in selected])
        return self._apply_clipboard_and_redact(res)

    def confirm_png_drop(self, path: str, mode: str) -> dict[str, Any]:
        """确认含工作流 PNG 的投递方式（image 作为参考图 / workflow 提取导入）。"""
        return self._apply_clipboard_and_redact(
            self._backend.submit_dropped_files([path], png_mode=mode)
        )

    def handle_toast_action(self, action: str) -> dict[str, Any]:
        """响应顶部互动提醒气泡点击动作（查看成图 / 排错 / 续期 / 打开隧道或对话页 / 复制配置）。"""
        self._backend.touch_activity()
        if action == "keep_working":
            return {"ok": True, "message": "✅ 已重置闲置计时，继续保留实例"}
        if action == "outputs":
            if self._backend.is_llm_mode_active():
                return self._apply_clipboard_and_redact(self._backend.get_llm_api_bundle("comfy"))
            return self._apply_clipboard_and_redact(self._backend.pull_or_open_latest_outputs())
        if action == "error":
            self._backend.acknowledge_comfy_error()
            return self._apply_clipboard_and_redact(self._backend.open_comfy_tunnel())
        if action in ("open_comfy", "open_llm_webui"):
            return self._apply_clipboard_and_redact(self._backend.open_comfy_tunnel())
        if action == "copy_llm_api":
            return self._apply_clipboard_and_redact(self._backend.get_llm_api_bundle("comfy"))
        if isinstance(action, str) and action.startswith("open_comfy_wf:"):
            wf_name = action.split(":", 1)[1].strip()
            return self._apply_clipboard_and_redact(
                self._backend.open_comfy_tunnel(workflow_name=wf_name or None)
            )
        return {"ok": False, "message": "未定义的快捷动作"}

    def save_prefs(self, updates: dict[str, Any]) -> dict[str, Any]:
        return self._backend.update_prefs(updates or {})

    def set_topmost(self, flag: bool) -> dict[str, Any]:
        flag_bool = bool(flag)
        if self._window and self._window.native:
            form = self._window.native
            form.Invoke(DotNetAction(lambda: setattr(form, "TopMost", flag_bool)))
        return self._backend.update_prefs({"on_top": flag_bool})

    def set_scale(self, scale_val: float) -> dict[str, Any]:
        """缩放桌宠窗口并确保缩放后不越出屏幕右/下工作区边缘。"""
        sc = max(0.75, min(1.35, float(scale_val)))
        updates: dict[str, Any] = {"scale": sc}
        if self._window:
            new_w = int(math.ceil(BASE_W * sc))
            new_h = int(math.ceil(BASE_H * sc))
            self._window.resize(new_w, new_h)
            if self._window.native:
                try:
                    hwnd = self._native_hwnd()
                    rect = wintypes.RECT()
                    if hwnd and user32.GetWindowRect(hwnd, ctypes.byref(rect)):
                        wa = Screen.FromHandle(self._window.native.Handle).WorkingArea
                        dpi = self._dpi_scale()
                        w_phys = max(1, int(rect.right - rect.left))
                        h_phys = max(1, int(rect.bottom - rect.top))
                        fx = int(rect.left)
                        fy = int(rect.top)
                        pad = int(round(4 * dpi))
                        dock_edge = str(
                            (self._backend.state.get("prefs") or {}).get("dock_edge") or ""
                        ).strip().lower()
                        if dock_edge == "right":
                            clamped_x = int(wa.Right) - w_phys - pad
                        elif dock_edge == "left":
                            clamped_x = int(wa.Left) + pad
                        else:
                            clamped_x = max(
                                int(wa.Left) + pad, min(fx, int(wa.Right) - w_phys - pad)
                            )
                        clamped_y = max(int(wa.Top) + pad, min(fy, int(wa.Bottom) - h_phys + pad))
                        if clamped_x != fx or clamped_y != fy:
                            # SWP_NOSIZE (0x0001) | SWP_NOZORDER (0x0004) | SWP_NOACTIVATE (0x0010)
                            user32.SetWindowPos(
                                hwnd, 0, clamped_x, clamped_y, 0, 0, 0x0001 | 0x0004 | 0x0010
                            )
                            updates["x"] = int(round(clamped_x / dpi))
                            updates["y"] = int(round(clamped_y / dpi))
                except Exception:
                    pass
        self._edge_peeking = False
        self._hover_out_since = None
        self._peek_wake_until = max(self._peek_wake_until, time.monotonic() + 2.5)
        self._wake_event.set()
        return self._backend.update_prefs(updates)

    def sync_region(self, layout: dict[str, Any]) -> None:
        """根据前端各可见元素精确边界，实时更新 WinForms Form.Region 裁剪轮廓并记录可见外接框。"""
        if not self._window or not self._window.native:
            return
        form = self._window.native
        dpi = self._dpi_scale()
        flipped = bool(layout.get("left_flipped", False))
        ui_scale = float((self._backend.state.get("prefs") or {}).get("scale", 1.0))

        path = GraphicsPath()
        has_any = False
        boxes: list[tuple[int, int, int, int]] = []

        # 1) 顶部思维气泡圆角矩形（整数边界对齐同心外环，彻底消除半像素台阶与角部磨损）
        bubble = layout.get("bubble")
        if isinstance(bubble, dict) and bubble.get("w", 0) > 4 and bubble.get("h", 0) > 4:
            bx, by, bw, bh = _round_box_px(bubble, dpi)
            if bw > 4 and bh > 4:
                anim_scale = max(0.2, min(1.2, float(bubble.get("scale", 1.0))))
                br = float(bubble.get("r", 16.0 * ui_scale * anim_scale)) * dpi
                _add_smooth_rounded_rect(path, float(bx), float(by), float(bw), float(bh), br)
                boxes.append((bx, by, bx + bw, by + bh))
                has_any = True

        # 2) 思维圆点（整数像素外框裁剪内嵌圆角纯色底板）
        for dot in layout.get("dots") or []:
            if isinstance(dot, dict) and dot.get("w", 0) > 1 and dot.get("h", 0) > 1:
                dx, dy, dw, dh = _round_box_px(dot, dpi)
                if dw > 1 and dh > 1:
                    _add_smooth_rounded_rect(
                        path,
                        float(dx),
                        float(dy),
                        float(dw),
                        float(dh),
                        min(dw, dh) * 0.5,
                    )
                    boxes.append((dx, dy, dx + dw, dy + dh))
                    has_any = True

        # 3) 右侧（或左侧）常驻迷你操作胶囊、迷你余额条 & 互动台词小气泡
        for pill_key in ("tools", "miniPill", "toast"):
            box = layout.get(pill_key)
            if isinstance(box, dict) and box.get("w", 0) > 2 and box.get("h", 0) > 2:
                px0, py0, pw0, ph0 = _round_box_px(box, dpi)
                if pw0 > 2 and ph0 > 2:
                    default_r = min(pw0, ph0) * 0.5
                    pr = (
                        float(box["r"]) * dpi
                        if isinstance(box.get("r"), (int, float))
                        else default_r
                    )
                    _add_smooth_rounded_rect(
                        path,
                        float(px0),
                        float(py0),
                        float(pw0),
                        float(ph0),
                        pr,
                    )
                    boxes.append((px0, py0, px0 + pw0, py0 + ph0))
                    has_any = True

        # 4) 人物立绘精准不透明像素轮廓（跟随动画尺寸实时缩放/位移，并区分眨眼帧）
        pet = layout.get("pet")
        if isinstance(pet, dict) and pet.get("w", 0) > 8 and pet.get("h", 0) > 8:
            px = int(round(pet["x"] * dpi))
            py = int(round(pet["y"] * dpi))
            pw = max(1, int(round(pet["w"] * dpi)))
            ph = max(1, int(round(pet["h"] * dpi)))
            is_blink = bool(pet.get("blink", False))
            for rx, ry, rw, rh in _sprite_mask_runs(pw, ph, flipped, is_blink):
                path.AddRectangle(Rectangle(px + rx, py + ry, rw, rh))
            boxes.append((px, py, px + pw, py + ph))
            has_any = True

        if not has_any:
            path.Dispose()
            return

        # 原子替换窗口内真实可见元素像素外接框列表，消除气泡收起后头顶透明区误判悬停
        self._visible_boxes_px = boxes

        new_reg = Region(path)
        path.Dispose()

        def _apply() -> None:
            old_reg = form.Region
            form.Region = new_reg
            if old_reg is not None:
                old_reg.Dispose()

        try:
            form.Invoke(DotNetAction(_apply))
        except Exception:
            new_reg.Dispose()

    def begin_drag(self) -> None:
        """接管鼠标左键拖拽：脱离浏览器视口束缚，原生 Win32 跟随并贴边吸附。"""
        if self._dragging or not self._window or not self._window.native:
            return
        hwnd = self._native_hwnd()
        if not hwnd:
            return
        self._dragging = True
        vk = 0x02 if user32.GetSystemMetrics(23) else 0x01  # 兼容左右手鼠标主按键

        pt0 = wintypes.POINT()
        rect0 = wintypes.RECT()
        user32.GetCursorPos(ctypes.byref(pt0))
        user32.GetWindowRect(hwnd, ctypes.byref(rect0))

        def _drag_worker() -> None:
            try:
                cur = wintypes.POINT()
                while not self._stop_event.is_set() and (user32.GetAsyncKeyState(vk) & 0x8000):
                    user32.GetCursorPos(ctypes.byref(cur))
                    nx = rect0.left + (cur.x - pt0.x)
                    ny = rect0.top + (cur.y - pt0.y)
                    user32.SetWindowPos(hwnd, 0, nx, ny, 0, 0, 0x0001 | 0x0004 | 0x0010)
                    time.sleep(0.008)

                # 松手时结合真实可见边界判定贴边磁吸（64*dpi 磁吸距，拖过边缘一律锁定吸附）
                final_rect = wintypes.RECT()
                user32.GetWindowRect(hwnd, ctypes.byref(final_rect))
                w_phys = final_rect.right - final_rect.left
                h_phys = final_rect.bottom - final_rect.top
                wa = Screen.FromHandle(self._window.native.Handle).WorkingArea
                dpi = self._dpi_scale()

                prefs = self._backend.state.get("prefs") or {}
                ui_scale = max(0.65, min(1.35, float(prefs.get("scale", 1.0) or 1.0)))
                bubble_open = bool(prefs.get("bubble_open", True))

                fx_before = int(final_rect.left)
                fx = fx_before
                fy = int(final_rect.top)
                vis_left = fx_before + int(round((18.0 if bubble_open else 71.0) * ui_scale * dpi))
                vis_right = fx_before + int(round((374.0 if bubble_open else 356.0) * ui_scale * dpi))

                snap_phys = int(round(64.0 * dpi))
                wa_l = int(wa.Left)
                wa_r = int(wa.Right)
                dock_edge: str | None = None
                if fx_before <= wa_l + snap_phys or vis_left <= wa_l + snap_phys:
                    fx = wa_l + int(round(4.0 * dpi))
                    dock_edge = "left"
                elif (fx_before + w_phys) >= wa_r - snap_phys or vis_right >= wa_r - snap_phys:
                    fx = wa_r - w_phys - int(round(4.0 * dpi))
                    dock_edge = "right"

                push_tol = int(round(20.0 * dpi))
                pushed_past = (
                    fx_before < wa_l - push_tol
                    or vis_left < wa_l - push_tol
                    or (fx_before + w_phys) > wa_r + push_tol
                    or vis_right > wa_r + push_tol
                )

                fy = max(
                    int(wa.Top) - int(round(4.0 * dpi)),
                    min(fy, int(wa.Bottom) - h_phys + int(round(10.0 * dpi))),
                )
                user32.SetWindowPos(hwnd, 0, fx, fy, 0, 0, 0x0001 | 0x0004 | 0x0010)

                logical_x = int(round(fx / dpi))
                logical_y = int(round(fy / dpi))
                mid_wa = (wa_l + wa_r) / 2.0
                left_flipped = (
                    (dock_edge == "left")
                    if dock_edge
                    else ((fx + w_phys / 2.0) < mid_wa)
                )
                hold_sec = 0.25 if (dock_edge and pushed_past) else 0.55
                self._edge_peeking = False
                self._hover_out_since = None
                self._peek_wake_until = max(self._peek_wake_until, time.monotonic() + hold_sec)
                self._backend.update_prefs(
                    {
                        "x": logical_x,
                        "y": logical_y,
                        "left_flipped": left_flipped,
                        "dock_edge": dock_edge,
                    }
                )
                self._wake_event.set()
                if not self._stop_event.is_set():
                    flag_js = "true" if left_flipped else "false"
                    dock_js = json.dumps(dock_edge)
                    try:
                        self._window.evaluate_js(
                            f"window.onDragFinished && window.onDragFinished({flag_js}, {dock_js});"
                        )
                    except Exception:
                        pass
            finally:
                self._dragging = False

        self._edge_peeking = False
        self._hover_out_since = None
        threading.Thread(target=_drag_worker, daemon=True).start()

    def _native_hwnd(self) -> int:
        if not self._window or not self._window.native:
            return 0
        try:
            return int(self._window.native.Handle.ToInt64())
        except Exception:
            try:
                return int(self._window.native.Handle)
            except Exception:
                return 0

    def set_edge_auto_hide(self, enabled: bool) -> dict[str, Any]:
        """开关贴边自动半隐藏；关闭时立即唤醒滑回桌面可见区域。"""
        flag = bool(enabled)
        res = self._backend.update_prefs({"edge_auto_hide": flag})
        if not flag:
            self.wake_from_edge_peek(hold_sec=2.0)
        else:
            self._hover_out_since = None
            self._peek_wake_until = max(self._peek_wake_until, time.monotonic() + 0.6)
            self._wake_event.set()
        return res

    def wake_from_edge_peek(self, hold_sec: float = 4.0) -> None:
        """从贴边半隐藏状态平顺唤醒回屏幕可见区并延长防隐藏窗口。"""
        self._peek_wake_until = max(self._peek_wake_until, time.monotonic() + max(0.5, float(hold_sec)))
        self._hover_out_since = None
        self._wake_event.set()
        if self._edge_peeking:
            threading.Thread(
                target=self._set_peek_state,
                args=(False, "wake"),
                daemon=True,
            ).start()

    def _is_cursor_on_visible_ui(
        self, pt_x: int, pt_y: int, rect: wintypes.RECT, dpi: float
    ) -> bool:
        """基于 sync_region 真实可见元素外接框判定光标是否悬停在桌宠可见部分上。"""
        boxes = self._visible_boxes_px
        if boxes:
            rx = int(pt_x) - int(rect.left)
            ry = int(pt_y) - int(rect.top)
            pad = int(round(14.0 * dpi))
            for x0, y0, x1, y1 in boxes:
                if (x0 - pad) <= rx <= (x1 + pad) and (y0 - pad) <= ry <= (y1 + pad):
                    return True
            return False

        margin = int(round(12.0 * dpi))
        return (
            (int(rect.left) - margin) <= int(pt_x) <= (int(rect.right) + margin)
            and (int(rect.top) - margin) <= int(pt_y) <= (int(rect.bottom) + margin)
        )

    def _is_cursor_in_wake_zone(
        self,
        pt_x: int,
        pt_y: int,
        rect: wintypes.RECT,
        wa: Any,
        dock_edge: str,
        ui_scale: float,
        dpi: float,
    ) -> bool:
        """判定光标是否进入半隐藏露头角色热区或 Fitts 全高屏幕边缘唤醒窄条。"""
        h_phys = max(1, int(rect.bottom) - int(rect.top))
        out_tol = int(round(24.0 * dpi))
        wa_l = int(wa.Left)
        wa_r = int(wa.Right)

        # 1) 露头角色宽幅热区
        hot_w = int(round(96.0 * ui_scale * dpi))
        char_y_top = int(rect.top) + int(0.45 * h_phys) - int(round(16.0 * dpi))
        char_y_bot = int(rect.bottom) + int(round(16.0 * dpi))
        if dock_edge == "right":
            in_char_x = (wa_r - hot_w) <= pt_x <= (wa_r + out_tol)
        else:
            in_char_x = (wa_l - out_tol) <= pt_x <= (wa_l + hot_w)
        if in_char_x and (char_y_top <= pt_y <= char_y_bot):
            return True

        # 2) 全高贴边唤醒窄条（鼠标甩到气泡原位所在屏幕边缘也能秒唤醒）
        edge_strip_w = int(round(18.0 * dpi))
        strip_y_top = int(rect.top) - int(round(8.0 * dpi))
        strip_y_bot = int(rect.bottom) + int(round(12.0 * dpi))
        if dock_edge == "right":
            in_strip_x = (wa_r - edge_strip_w) <= pt_x <= (wa_r + out_tol)
        else:
            in_strip_x = (wa_l - out_tol) <= pt_x <= (wa_l + edge_strip_w)
        return bool(in_strip_x and (strip_y_top <= pt_y <= strip_y_bot))

    def _is_cursor_keeping_awake(
        self,
        pt_x: int,
        pt_y: int,
        rect: wintypes.RECT,
        wa: Any,
        dock_edge: str,
        ui_scale: float,
        dpi: float,
    ) -> bool:
        """判定光标是否处于悬停保活包络内（严格覆盖唤醒热区并桥接多组件间凹陷过渡区）。"""
        if self._is_cursor_on_visible_ui(pt_x, pt_y, rect, dpi):
            return True
        if dock_edge in ("left", "right") and self._is_cursor_in_wake_zone(
            pt_x, pt_y, rect, wa, dock_edge, ui_scale, dpi
        ):
            return True
        boxes = self._visible_boxes_px
        if len(boxes) >= 2:
            pad = int(round(14.0 * dpi))
            min_x = min(b[0] for b in boxes) - pad
            min_y = min(b[1] for b in boxes) - pad
            max_x = max(b[2] for b in boxes) + pad
            max_y = max(b[3] for b in boxes) + pad
            rx = int(pt_x) - int(rect.left)
            ry = int(pt_y) - int(rect.top)
            if min_x <= rx <= max_x and min_y <= ry <= max_y:
                return True
        return False

    def _animate_window_x(
        self,
        hwnd: int,
        start_x: int,
        end_x: int,
        y_phys: int,
        duration: float = 0.16,
        abort_on_wake: bool = False,
        wake_check: Callable[[], bool] | None = None,
    ) -> bool:
        """约 120Hz 缓动插值水平移动窗口，支持收起中途光标移回即时打断回弹。"""
        if start_x == end_x:
            return True
        t0 = time.perf_counter()
        dur = max(0.06, float(duration))
        flags = 0x0001 | 0x0004 | 0x0010
        while not self._stop_event.is_set() and not self._dragging:
            if abort_on_wake:
                if self._in_dialog or time.monotonic() < self._peek_wake_until:
                    return False
                if wake_check is not None:
                    try:
                        if wake_check():
                            return False
                    except Exception:
                        pass
            elapsed = time.perf_counter() - t0
            p = min(1.0, elapsed / dur)
            # ease-out cubic：起始略快、落位轻柔减速
            eased = 1.0 - (1.0 - p) ** 3
            cx = int(round(start_x + (end_x - start_x) * eased))
            user32.SetWindowPos(hwnd, 0, cx, int(y_phys), 0, 0, flags)
            if p >= 1.0:
                return True
            time.sleep(0.008)
        return False

    def _can_auto_hide_ui(self) -> bool:
        if not self._window or self._stop_event.is_set():
            return False
        try:
            res = self._window.evaluate_js("window.canEdgeAutoHide ? window.canEdgeAutoHide() : true;")
            return bool(res) if res is not None else True
        except Exception:
            return True

    def _set_peek_state(self, peeking: bool, reason: str = "") -> bool:
        """切换贴边半隐藏状态：支持收起中途鼠标移回即时回弹与唤醒时气泡/窗口并行展开。"""
        hwnd = self._native_hwnd()
        if not hwnd or not self._window or not self._window.native or self._stop_event.is_set():
            return False
        with self._edge_lock:
            prefs = self._backend.state.get("prefs") or {}
            dock_edge = str(prefs.get("dock_edge") or "").strip().lower()
            if peeking and (
                dock_edge not in ("left", "right")
                or not bool(prefs.get("edge_auto_hide", True))
                or self._dragging
                or self._in_dialog
                or time.monotonic() < self._peek_wake_until
            ):
                return False

            rect = wintypes.RECT()
            if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
                return False
            w_phys = rect.right - rect.left
            wa = Screen.FromHandle(self._window.native.Handle).WorkingArea
            dpi = self._dpi_scale()
            ui_scale = max(0.65, min(1.35, float(prefs.get("scale", 1.0) or 1.0)))
            edge_key = dock_edge if dock_edge in ("left", "right") else "right"
            x_visible, x_peek = _compute_edge_x_pair(
                edge_key,
                int(wa.Left),
                int(wa.Right),
                w_phys,
                ui_scale,
                dpi,
            )

            self._edge_animating = True
            try:
                if peeking:
                    if not self._can_auto_hide_ui():
                        return False
                    bubble_was_open = bool(prefs.get("bubble_open", True))
                    init_pt = wintypes.POINT()
                    has_init_pt = bool(user32.GetCursorPos(ctypes.byref(init_pt)))
                    move_thresh_sq = float(int(round(4.0 * dpi)) ** 2)

                    def _cursor_returned_to_pet() -> bool:
                        cur_pt = wintypes.POINT()
                        cur_r = wintypes.RECT()
                        try:
                            if not user32.GetCursorPos(ctypes.byref(cur_pt)):
                                return False
                            if has_init_pt:
                                dx = float(int(cur_pt.x) - int(init_pt.x))
                                dy = float(int(cur_pt.y) - int(init_pt.y))
                                if (dx * dx + dy * dy) < move_thresh_sq:
                                    return False
                            if not user32.GetWindowRect(hwnd, ctypes.byref(cur_r)):
                                return False
                        except Exception:
                            return False
                        return self._is_cursor_in_wake_zone(
                            int(cur_pt.x), int(cur_pt.y), cur_r, wa, edge_key, ui_scale, dpi
                        ) or self._is_cursor_on_visible_ui(
                            int(cur_pt.x), int(cur_pt.y), cur_r, dpi
                        )

                    try:
                        self._window.evaluate_js(
                            "window.onEdgePeekChanged && window.onEdgePeekChanged(true);"
                        )
                    except Exception:
                        pass

                    # 仅在气泡原本展开时短暂等待收起动画，气泡已收起时极速起滑
                    wait_sec = 0.10 if bubble_was_open else 0.02
                    deadline = time.monotonic() + wait_sec
                    while time.monotonic() < deadline:
                        if (
                            self._stop_event.is_set()
                            or self._dragging
                            or self._in_dialog
                            or time.monotonic() < self._peek_wake_until
                            or _cursor_returned_to_pet()
                        ):
                            self._peek_wake_until = max(
                                self._peek_wake_until, time.monotonic() + 0.55
                            )
                            self._hover_out_since = None
                            try:
                                self._window.evaluate_js(
                                    "window.onEdgePeekChanged && window.onEdgePeekChanged(false);"
                                )
                            except Exception:
                                pass
                            return False
                        time.sleep(0.015)

                    user32.GetWindowRect(hwnd, ctypes.byref(rect))
                    ok = self._animate_window_x(
                        hwnd,
                        rect.left,
                        x_peek,
                        rect.top,
                        duration=0.16,
                        abort_on_wake=True,
                        wake_check=_cursor_returned_to_pet,
                    )
                    if not ok:
                        self._peek_wake_until = max(
                            self._peek_wake_until, time.monotonic() + 0.55
                        )
                        self._hover_out_since = None
                        self._edge_peeking = False
                        try:
                            self._window.evaluate_js(
                                "window.onEdgePeekChanged && window.onEdgePeekChanged(false);"
                            )
                        except Exception:
                            pass
                        user32.GetWindowRect(hwnd, ctypes.byref(rect))
                        self._animate_window_x(
                            hwnd,
                            rect.left,
                            x_visible,
                            rect.top,
                            duration=0.14,
                            abort_on_wake=False,
                        )
                        return False
                    self._edge_peeking = True
                    return True

                # 并行唤醒：先立即触发前端气泡展开，同时滑回全显停靠坐标，消除两段式串行卡顿
                self._edge_peeking = False
                self._hover_out_since = None
                try:
                    self._window.evaluate_js(
                        "window.onEdgePeekChanged && window.onEdgePeekChanged(false);"
                    )
                except Exception:
                    pass
                cur_rect = wintypes.RECT()
                if user32.GetWindowRect(hwnd, ctypes.byref(cur_rect)):
                    if dock_edge in ("left", "right"):
                        self._animate_window_x(
                            hwnd,
                            cur_rect.left,
                            x_visible,
                            cur_rect.top,
                            duration=0.15,
                            abort_on_wake=False,
                        )
                        self._backend.update_prefs({"x": int(round(x_visible / dpi))})
                return True
            finally:
                self._edge_animating = False

    def _edge_watch_tick(self, now: float | None = None) -> None:
        """单次巡检鼠标与停靠边位置，触发贴边自动半隐藏或悬停唤醒。"""
        if self._dragging or self._in_dialog or self._edge_animating or self._stop_event.is_set():
            return
        hwnd = self._native_hwnd()
        if not hwnd or not self._window or not self._window.native:
            return
        prefs = self._backend.state.get("prefs") or {}
        dock_edge = str(prefs.get("dock_edge") or "").strip().lower()
        auto_hide = bool(prefs.get("edge_auto_hide", True))
        if dock_edge not in ("left", "right") or not auto_hide:
            if self._edge_peeking:
                self._set_peek_state(False, reason="pref_off")
            self._hover_out_since = None
            return

        pt = wintypes.POINT()
        rect = wintypes.RECT()
        if not user32.GetCursorPos(ctypes.byref(pt)) or not user32.GetWindowRect(
            hwnd, ctypes.byref(rect)
        ):
            return
        wa = Screen.FromHandle(self._window.native.Handle).WorkingArea
        dpi = self._dpi_scale()
        ui_scale = max(0.65, min(1.35, float(prefs.get("scale", 1.0) or 1.0)))
        ts = time.monotonic() if now is None else float(now)

        if self._edge_peeking:
            w_phys = max(1, rect.right - rect.left)
            x_visible, _ = _compute_edge_x_pair(
                dock_edge, int(wa.Left), int(wa.Right), w_phys, ui_scale, dpi
            )
            if abs(int(rect.left) - x_visible) <= int(round(8.0 * dpi)):
                # 窗口已被外部激活拉回可见区：自愈复位半隐藏状态并恢复气泡
                self._edge_peeking = False
                self._hover_out_since = None
                self._peek_wake_until = max(self._peek_wake_until, ts + 0.8)
                try:
                    self._window.evaluate_js(
                        "window.onEdgePeekChanged && window.onEdgePeekChanged(false);"
                    )
                except Exception:
                    pass
                return
            if self._is_cursor_in_wake_zone(
                int(pt.x), int(pt.y), rect, wa, dock_edge, ui_scale, dpi
            ):
                self._peek_wake_until = max(self._peek_wake_until, ts + 0.50)
                self._set_peek_state(False, reason="hover")
            return

        inside = self._is_cursor_keeping_awake(
            int(pt.x), int(pt.y), rect, wa, dock_edge, ui_scale, dpi
        )
        if inside:
            self._hover_out_since = None
            return
        if self._hover_out_since is None:
            self._hover_out_since = ts
        if ts < self._peek_wake_until:
            return
        if (ts - self._hover_out_since) >= 0.26:
            self._set_peek_state(True, reason="idle_leave")

    def _sync_initial_dock_edge(self) -> None:
        """启动停靠自检：若未保存 dock_edge 但位置已在屏幕左右边 16*dpi 内，则自动补全停靠边。"""
        prefs = self._backend.state.get("prefs") or {}
        if prefs.get("dock_edge") in ("left", "right"):
            return
        hwnd = self._native_hwnd()
        if not hwnd or not self._window or not self._window.native:
            return
        try:
            rect = wintypes.RECT()
            if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
                return
            wa = Screen.FromHandle(self._window.native.Handle).WorkingArea
            dpi = self._dpi_scale()
            edge_thresh = int(round(16.0 * dpi))
            if abs(int(rect.left) - int(wa.Left)) <= edge_thresh:
                self._backend.update_prefs({"dock_edge": "left", "left_flipped": True})
            elif abs(int(wa.Right) - int(rect.right)) <= edge_thresh:
                self._backend.update_prefs({"dock_edge": "right", "left_flipped": False})
        except Exception:
            pass

    def _edge_watch_loop(self) -> None:
        """后台高频检测光标停靠状态（测试渲染模式下自动跳过）。"""
        if "--test-render" in sys.argv:
            return
        if self._window:
            try:
                self._window.events.loaded.wait(timeout=5.0)
            except Exception:
                pass
        last_err_ts = 0.0
        while not self._stop_event.is_set():
            try:
                self._edge_watch_tick()
            except Exception as exc:
                now = time.monotonic()
                if now - last_err_ts >= 10.0:
                    last_err_ts = now
                    try:
                        print(f"[edge_watch] 巡检异常: {exc!r}", file=sys.stderr)
                    except Exception:
                        pass
            self._wake_event.wait(0.035)
            self._wake_event.clear()

    def quit_app(self) -> None:
        self._stop_event.set()
        if self._window:
            try:
                self._window.destroy()
            except Exception:
                pass


def _compute_initial_pos(prefs: dict[str, Any], win_w: int, win_h: int) -> tuple[int, int]:
    """计算初始合法窗口坐标；若未保存或超出桌面范围则落回右下角。"""
    wa = Screen.PrimaryScreen.WorkingArea
    dpi_scale = 1.0
    try:
        hdc = ctypes.windll.user32.GetDC(0)
        dpi = ctypes.windll.gdi32.GetDeviceCaps(hdc, 88)
        ctypes.windll.user32.ReleaseDC(0, hdc)
        if dpi > 0:
            dpi_scale = dpi / 96.0
    except Exception:
        pass

    wa_l = int(wa.Left / dpi_scale)
    wa_t = int(wa.Top / dpi_scale)
    wa_r = int(wa.Right / dpi_scale)
    wa_b = int(wa.Bottom / dpi_scale)

    saved_x = prefs.get("x")
    saved_y = prefs.get("y")
    if isinstance(saved_x, (int, float)) and isinstance(saved_y, (int, float)):
        sx, sy = int(saved_x), int(saved_y)
        if (wa_l - 80) <= sx <= (wa_r - 80) and (wa_t - 40) <= sy <= (wa_b - 80):
            sx = max(wa_l + 4, min(sx, wa_r - win_w - 4))
            sy = max(wa_t + 4, min(sy, wa_b - win_h + 4))
            return sx, sy

    return max(wa_l + 8, wa_r - win_w - 24), max(wa_t + 8, wa_b - win_h - 10)


def _push_to_ui(win: webview.Window, snap: dict[str, Any]) -> None:
    """将状态快照安全推送到前端 JS 渲染函数。"""
    payload = json.dumps(snap, ensure_ascii=False)
    try:
        win.evaluate_js(f"window.renderState && window.renderState({payload});")
    except Exception:
        pass


def _push_event_to_ui(
    win: webview.Window, api: PetBridgeApi, evt: dict[str, Any]
) -> None:
    """将后端实时事件（推理进度 / 出图提醒 / 上传进度）脱敏后推送至前端。"""
    clip_text = evt.get("clipboard_text")
    if isinstance(clip_text, str) and clip_text:
        api._set_clipboard(clip_text)
    evt_type = str(evt.get("type") or "")
    if evt_type in {
        "outputs_ready",
        "comfy_error",
        "deploy_done",
        "deploy_error",
        "png_workflow_choice",
        "idle_warning",
        "idle_stopped",
    }:
        api.wake_from_edge_peek(hold_sec=8.5)
    safe_evt = {k: v for k, v in evt.items() if k != "clipboard_text"}
    payload = json.dumps(safe_evt, ensure_ascii=False)
    try:
        win.evaluate_js(f"window.onBackendEvent && window.onBackendEvent({payload});")
    except Exception:
        pass


def run_app(
    test_render_sec: float | None = None,
    capture_path: str | None = None,
    force_restart: bool = False,
) -> int:
    if test_render_sec is None and not _acquire_single_instance(force_restart=force_restart):
        return 0

    backend = ColabPetBackend()
    prefs = backend.state.get("prefs") or {}
    scale = max(0.75, min(1.35, float(prefs.get("scale", 1.0))))
    win_w = int(math.ceil(BASE_W * scale))
    win_h = int(math.ceil(BASE_H * scale))
    init_x, init_y = _compute_initial_pos(prefs, win_w, win_h)

    stop_event = threading.Event()
    api = PetBridgeApi(backend, stop_event)
    ui_html_path = PET_DIR / "ui.html"
    last_ui_mtime_ns = _safe_file_mtime_ns(ui_html_path)

    httpd = _ThreadingHTTPServer(("127.0.0.1", 0), _PetAssetHandler)
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()

    # 关闭 WebView2 Layered TransparencyKey，改用 Form.Region 实现真透明 + 100% 稳定鼠标交互
    window = webview.create_window(
        title="Colab 桌宠",
        url=f"http://127.0.0.1:{port}/ui.html",
        js_api=api,
        width=win_w,
        height=win_h,
        x=init_x,
        y=init_y,
        resizable=False,
        frameless=True,
        easy_drag=False,
        shadow=False,
        transparent=False,
        background_color=EDGE_BG_HEX,
        focus=True,
        on_top=bool(prefs.get("on_top", True)),
    )
    api._bind_window(window)

    def _on_dom_loaded() -> None:
        try:
            doc = window.dom.document
            doc.events.dragover += DOMEventHandler(
                lambda _e: None, prevent_default=True, stop_propagation=False
            )
            doc.events.drop += DOMEventHandler(
                api._on_dom_drop, prevent_default=True, stop_propagation=False
            )
        except Exception:
            pass

    window.events.loaded += _on_dom_loaded

    def _init_form_frame() -> None:
        for _ in range(40):
            if window.native is not None:
                break
            time.sleep(0.05)
        if window.native is None:
            return
        form = window.native

        def _configure() -> None:
            form.BackColor = Color.FromArgb(*EDGE_BG_RGB)
            form.TopMost = bool(prefs.get("on_top", True))
            form.Show()
            form.Activate()
            try:
                hwnd = int(form.Handle.ToInt64())
                ctypes.windll.user32.ShowWindow(hwnd, 5)  # SW_SHOW
            except Exception:
                pass

        try:
            form.Invoke(DotNetAction(_configure))
            window.resize(win_w, win_h)
            window.move(init_x, init_y)
        except Exception:
            pass

    def _reload_ui_if_modified() -> None:
        """检测 ui.html 磁盘变更并免重启无感热重载前端页面与状态快照。"""
        nonlocal last_ui_mtime_ns
        cur_mtime = _safe_file_mtime_ns(ui_html_path)
        if cur_mtime is None:
            return
        if last_ui_mtime_ns is None:
            last_ui_mtime_ns = cur_mtime
            return
        if cur_mtime == last_ui_mtime_ns:
            return
        last_ui_mtime_ns = cur_mtime
        reloaded = threading.Event()

        def _mark_loaded() -> None:
            reloaded.set()

        try:
            window.events.loaded += _mark_loaded
            window.load_url(f"http://127.0.0.1:{port}/ui.html?v={cur_mtime}")
            reloaded.wait(timeout=1.5)
        except Exception:
            time.sleep(0.45)
        finally:
            try:
                window.events.loaded -= _mark_loaded
            except Exception:
                pass
        time.sleep(0.12)
        if not stop_event.is_set():
            _push_to_ui(window, backend.get_cached_snapshot())

    def _background_loop() -> None:
        _init_form_frame()
        time.sleep(0.45)
        snap = backend.fetch_live_status()
        if not stop_event.is_set():
            _push_to_ui(window, snap)

        if test_render_sec is not None:
            time.sleep(max(1.0, float(test_render_sec)))
            if capture_path and window.native is not None:
                try:
                    hwnd = int(window.native.Handle.ToInt64())
                    _capture_hwnd_png(hwnd, Path(capture_path))
                except Exception as exc:
                    print(f"CAPTURE_ERROR: {exc}", file=sys.stderr)
            stop_event.set()
            try:
                window.destroy()
            except Exception:
                pass
            return

        api._sync_initial_dock_edge()
        threading.Thread(target=api._edge_watch_loop, daemon=True).start()
        backend.start_monitor(
            lambda evt: None if stop_event.is_set() else _push_event_to_ui(window, api, evt)
        )

        sec_ticks = 0
        while not stop_event.is_set():
            interval = int((backend.state.get("prefs") or {}).get("poll_interval_sec") or 60)
            interval = max(15, min(600, interval))
            for _ in range(interval):
                if stop_event.is_set():
                    return
                time.sleep(1.0)
                sec_ticks += 1
                if sec_ticks % 2 == 0 and not stop_event.is_set():
                    _reload_ui_if_modified()
            snap = backend.fetch_live_status()
            if not stop_event.is_set():
                _push_to_ui(window, snap)

    def _on_closed() -> None:
        stop_event.set()
        backend.stop_monitor()

    window.events.closed += _on_closed

    try:
        webview.start(func=_background_loop)
    finally:
        stop_event.set()
        backend.stop_monitor()
        try:
            httpd.shutdown()
            httpd.server_close()
        except Exception:
            pass

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Colab Q 版桌宠悬浮挂件")
    parser.add_argument(
        "--test-render",
        type=float,
        default=None,
        help="测试渲染指定秒数后自动退出",
    )
    parser.add_argument(
        "--capture",
        type=str,
        default=None,
        help="配合 --test-render 将窗口渲染结果保存为 PNG",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="若已有桌宠实例运行，优雅关闭旧窗口并以最新代码重启",
    )
    args = parser.parse_args()
    return run_app(
        test_render_sec=args.test_render,
        capture_path=args.capture,
        force_restart=bool(args.restart),
    )


if __name__ == "__main__":
    raise SystemExit(main())
