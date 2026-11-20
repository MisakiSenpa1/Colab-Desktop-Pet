"""Colab 桌宠 Win32 原生窗口、单实例互斥、Alpha/圆角 Region 裁剪与 GDI+ 截图模块。

说明：本模块封装 Win32 ctypes（`user32` / `dwmapi` / `kernel32_api`）与 .NET WinForms/System.Drawing
窗口裁剪、单实例唤醒热替换及无 PIL 依赖 PNG 截图。严禁导入 `PIL`。
"""

from __future__ import annotations

import base64
import ctypes
import json
import math
import os
import re
import sys
import time
import zlib
from ctypes import wintypes
from pathlib import Path
from typing import Any

import clr

PET_DIR: Path = Path(__file__).resolve().parent
_PACKAGE_DIR: Path = PET_DIR.parent
if str(_PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_DIR))

from colab_core.exec_runner import suppress_logged  # noqa: E402

clr.AddReference("System.Drawing")
clr.AddReference("System.Windows.Forms")
from System import Action as DotNetAction, Int64, IntPtr  # type: ignore  # noqa: E402
from System.Drawing import Bitmap, Color, Graphics, Image as DotNetImage, Rectangle, Region, Size as Size  # type: ignore  # noqa: E402
from System.Drawing.Drawing2D import GraphicsPath  # type: ignore  # noqa: E402
from System.Drawing.Imaging import ImageFormat, PixelFormat  # type: ignore  # noqa: E402
from System.Windows.Forms import Clipboard as Clipboard, FormWindowState as FormWindowState, Screen  # type: ignore  # noqa: E402

ASSETS_DIR: Path = PET_DIR / "assets"

BASE_W = 392
BASE_H = 548
# 与人物立绘深靛蓝描边一致，在 Region 像素轮廓边缘无缝融合
EDGE_BG_HEX = "#221F52"
EDGE_BG_RGB = (34, 31, 82)

# 模块级唯一 user32 实例与 64 位句柄安全参数签名，避免污染全局 ctypes.windll.user32 影响 pywebview
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
user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
user32.ShowWindow.restype = wintypes.BOOL
user32.SetForegroundWindow.argtypes = [wintypes.HWND]
user32.SetForegroundWindow.restype = wintypes.BOOL
user32.IsIconic.argtypes = [wintypes.HWND]
user32.IsIconic.restype = wintypes.BOOL
user32.GetForegroundWindow.argtypes = []
user32.GetForegroundWindow.restype = wintypes.HWND
user32.IsWindowVisible.argtypes = [wintypes.HWND]
user32.IsWindowVisible.restype = wintypes.BOOL

try:
    dwmapi = ctypes.WinDLL("dwmapi")
    dwmapi.DwmSetWindowAttribute.argtypes = [
        wintypes.HWND,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
    ]
    dwmapi.DwmSetWindowAttribute.restype = ctypes.c_long
except Exception:
    dwmapi = None  # type: ignore[assignment]

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
# 将 0/1 二值字节快速映射为 0x00 / 0xff，配合 _OPAQUE_RUN_RE 扫描水平连续不透明区间
_MASK_BYTE_TRANS = bytes([0] + [255] * 255)
_OPAQUE_RUN_RE = re.compile(b"\xff+")
_SPRITE_ALPHA_CACHE: dict[str, tuple[int, int, bytes]] = {}
_MASK_RUNS_CACHE: dict[tuple[int, int, bool, bool], list[tuple[int, int, int, int]]] = {}


def configure_window_paths(pet_dir: Path, assets_dir: Path | None = None) -> tuple[Path, Path]:
    """同步配置桌宠源码目录与资产目录（兼容 PyInstaller 冻结态 `_MEIPASS/colab_pet`）。"""
    global PET_DIR, ASSETS_DIR
    PET_DIR = Path(pet_dir).resolve()
    ASSETS_DIR = Path(assets_dir).resolve() if assets_dir is not None else (PET_DIR / "assets")
    return PET_DIR, ASSETS_DIR


def get_window_dpi_scale(hwnd: int) -> float:
    """查询指定窗口句柄的物理 DPI 缩放倍率。"""
    if hwnd:
        with suppress_logged("win32", throttled_key="dpi_scale"):
            dpi_val = int(user32.GetDpiForWindow(hwnd) or 0)
            if dpi_val > 0:
                return max(1.0, float(dpi_val) / 96.0)
    return 1.0


def _current_desktop_tag() -> str:
    """获取当前线程所属 Win32 桌面名称并清洗为互斥量后缀，避免跨桌面误锁。"""
    u32 = ctypes.windll.user32
    k32 = ctypes.windll.kernel32
    raw_name = ""
    try:
        h_desk = u32.GetThreadDesktop(k32.GetCurrentThreadId())
        if h_desk:
            buf = ctypes.create_unicode_buffer(256)
            needed = wintypes.DWORD()
            if u32.GetUserObjectInformationW(h_desk, 2, buf, ctypes.sizeof(buf), ctypes.byref(needed)):
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
        ft_create, ft_exit, ft_kernel, ft_user = _FILETIME(), _FILETIME(), _FILETIME(), _FILETIME()
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
    """获取桌宠前后端源码（自动涵盖 `PET_DIR/*.py`、前端静态页与 `colab_core`）最新修改时间戳。"""
    latest = 0.0
    candidates: list[Path] = [
        PET_DIR / "ui.html",
        PET_DIR / "studio.html",
        PET_DIR / "studio.css",
        PET_DIR / "studio.js",
        PET_DIR.parent / "colab_security.py",
    ]
    if PET_DIR.is_dir():
        candidates.extend(PET_DIR.glob("*.py"))
    core_dir = PET_DIR.parent / "colab_core"
    if core_dir.is_dir():
        candidates.extend(core_dir.rglob("*.py"))
    for p in candidates:
        try:
            latest = max(latest, float(p.stat().st_mtime))
        except OSError:
            pass
    return latest


def _clamp_hwnd_to_workarea(hwnd: int, rect: wintypes.RECT) -> None:
    """检查桌宠窗口是否越出显示器工作区，必要时拉回安全可见范围。"""
    win_w = max(BASE_W, int(rect.right - rect.left))
    win_h = max(BASE_H, int(rect.bottom - rect.top))
    in_view = False
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

    # SWP_NOSIZE (0x0001) | SWP_NOZORDER (0x0004) | SWP_SHOWWINDOW (0x0040)
    swp_flags = 0x0001 | 0x0004 | 0x0040
    if not in_view:
        primary_wa = Screen.PrimaryScreen.WorkingArea
        safe_x = max(primary_wa.Left + 4, primary_wa.Right - win_w - 16)
        safe_y = max(primary_wa.Top + 4, primary_wa.Bottom - win_h - 10)
        user32.SetWindowPos(hwnd, 0, safe_x, safe_y, 0, 0, swp_flags)
        return

    if (
        (rect.right - wa.Right) > 16
        or (rect.bottom - wa.Bottom) > 16
        or rect.left < wa.Left - 24
        or rect.top < wa.Top - 16
    ):
        safe_x = max(wa.Left + 4, min(int(rect.left), wa.Right - win_w - 4))
        safe_y = max(wa.Top + 4, min(int(rect.top), wa.Bottom - win_h + 4))
        user32.SetWindowPos(hwnd, 0, safe_x, safe_y, 0, 0, swp_flags)


def _restore_and_focus_pet_hwnd(hwnd: int) -> bool:
    """将已存在的桌宠窗口恢复显示、校正屏幕工作区越界并激活到前台。"""
    if not hwnd:
        return False
    u32 = ctypes.windll.user32
    u32.ShowWindow(hwnd, 9 if u32.IsIconic(hwnd) else 5)
    rect = wintypes.RECT()
    if user32.GetWindowRect(hwnd, ctypes.byref(rect)):
        with suppress_logged("win32", throttled_key="clamp_workarea"):
            _clamp_hwnd_to_workarea(hwnd, rect)
    u32.SetForegroundWindow(hwnd)
    return True


def _replace_existing_pet_instance(hwnd: int, scoped_name: str, old_handle: Any) -> bool:
    """优雅关闭旧实例并重新获取命名互斥量以实现热替换重启。"""
    global _MUTEX_HANDLE
    if old_handle:
        kernel32_api.CloseHandle(old_handle)
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
        _MUTEX_HANDLE = kernel32_api.CreateMutexW(None, False, scoped_name)
        return True
    _restore_and_focus_pet_hwnd(hwnd)
    return False


def _acquire_single_instance(
    name: str = "Local\\ColabDesktopPetSingletonMutex",
    force_restart: bool = False,
) -> bool:
    """使用按桌面隔离的 Win32 互斥量确保单实例，支持代码更新后重双击优雅热替换重启。"""
    global _MUTEX_HANDLE
    scoped_name = f"{name}_{_current_desktop_tag()}"
    ctypes.set_last_error(0)
    handle = kernel32_api.CreateMutexW(None, False, scoped_name)
    if ctypes.get_last_error() != 183:  # ERROR_ALREADY_EXISTS
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
        if proc_start_ts is not None and _latest_pet_source_mtime() > (proc_start_ts + 1.5):
            should_replace = True

    if should_replace:
        return _replace_existing_pet_instance(hwnd, scoped_name, handle)

    _restore_and_focus_pet_hwnd(hwnd)
    if handle:
        kernel32_api.CloseHandle(handle)
    return False


def _get_sprite_alpha_template(is_blink: bool) -> tuple[int, int, bytes] | None:
    """懒加载并缓存预置二值 Alpha 网格（源自 assets/alpha_masks.json），分离常驻帧与眨眼帧。"""
    key = "blink" if is_blink else "base"
    cached = _SPRITE_ALPHA_CACHE.get(key)
    if cached is not None:
        return cached

    masks_path = ASSETS_DIR / "alpha_masks.json"
    if not masks_path.is_file():
        return None
    try:
        raw_obj = json.loads(masks_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(raw_obj, dict):
        return None

    tmpl_w = int(raw_obj.get("width") or 0)
    tmpl_h = int(raw_obj.get("height") or 0)
    if tmpl_w <= 0 or tmpl_h <= 0:
        return None
    expected_len = tmpl_w * tmpl_h

    for field in ("base", "blink"):
        b64_val = raw_obj.get(field)
        if not isinstance(b64_val, str) or not b64_val:
            continue
        try:
            raw_grid = zlib.decompress(base64.b64decode(b64_val))
        except Exception:
            continue
        if len(raw_grid) == expected_len:
            _SPRITE_ALPHA_CACHE[field] = (tmpl_w, tmpl_h, raw_grid.translate(_MASK_BYTE_TRANS))

    if key in _SPRITE_ALPHA_CACHE:
        return _SPRITE_ALPHA_CACHE[key]
    if is_blink and "base" in _SPRITE_ALPHA_CACHE:
        # 若文件存在但缺少单独的 blink 字段，仅在内存兜底回退 base，不误缓存 blink
        return _SPRITE_ALPHA_CACHE["base"]
    return None


def _compute_pet_opaque_rects(
    w: int, h: int, flipped: bool, is_blink: bool = False
) -> list[tuple[int, int, int, int]]:
    """基于预置 500x418 二值模板纯 Python 索引缩放计算不透明扫描矩形 (x, y, rw, rh)。"""
    key = (w, h, flipped, is_blink)
    if key in _MASK_RUNS_CACHE:
        return _MASK_RUNS_CACHE[key]

    tmpl = _get_sprite_alpha_template(is_blink)
    if tmpl is None or w <= 0 or h <= 0:
        return [(0, 0, max(1, w), max(1, h))]

    tw, th, grid = tmpl
    x_map = [min(tw - 1, (x * tw) // w) for x in range(w)]
    if flipped:
        x_map.reverse()

    row_cache: dict[int, bytes] = {}
    rects: list[tuple[int, int, int, int]] = []
    active: dict[tuple[int, int], int] = {}
    for yy in range(h):
        sy = min(th - 1, (yy * th) // h)
        row = row_cache.get(sy)
        if row is None:
            off = sy * tw
            row = bytes(grid[off + sx] for sx in x_map)
            row_cache[sy] = row
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
    r = max(0.0, min(float(radius), (fx1 - fx0) * 0.5, (fy1 - fy0) * 0.5))
    runs: list[tuple[int, int, int, int]] = []
    for py in range(int(math.floor(fy0)), int(math.ceil(fy1)) + 1):
        yc = py + 0.5
        if yc < fy0 or yc > fy1:
            continue
        if r > 0.0 and (yc < fy0 + r or yc > fy1 - r):
            dy = (fy0 + r - yc) if yc < fy0 + r else (yc - (fy1 - r))
            dx = math.sqrt(max(0.0, r * r - dy * dy))
            lx, rx = fx0 + r - dx, fx1 - r + dx
        else:
            lx, rx = fx0, fx1
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
        return int(wa_right) - int(w_phys) - pad, int(wa_right) - int(round(140.0 * factor))
    return int(wa_left) + pad, int(wa_left) - int(round(251.0 * factor))


def _to_intptr(handle: Any) -> IntPtr:
    """将 Win32 64 位无符号句柄安全转换为 .NET IntPtr，防止符号位溢出。"""
    return IntPtr(Int64(ctypes.c_int64(int(handle or 0)).value))


def _capture_hwnd_png(hwnd: int, out_path: Path) -> None:
    """截取窗口当前可见渲染区域，并使用 .NET System.Drawing 按 GetWindowRgn 合成为白底 PNG。"""
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

    rect = wintypes.RECT()
    u32.GetWindowRect(hwnd, ctypes.byref(rect))
    w = max(1, rect.right - rect.left)
    h = max(1, rect.bottom - rect.top)

    hdc_win = u32.GetWindowDC(hwnd)
    hdc_mem = g32.CreateCompatibleDC(hdc_win)
    hbmp = g32.CreateCompatibleBitmap(hdc_win, w, h)
    old = g32.SelectObject(hdc_mem, hbmp)
    u32.PrintWindow(hwnd, hdc_mem, 2)
    # 必须先将 hbmp 从内存 DC 选出，再交给 GDI+ FromHbitmap 读取像素
    g32.SelectObject(hdc_mem, old)

    hrgn = g32.CreateRectRgn(0, 0, 0, 0)
    rgn_type = u32.GetWindowRgn(hwnd, hrgn)

    src_bmp = DotNetImage.FromHbitmap(_to_intptr(hbmp))
    dst_bmp = Bitmap(w, h, PixelFormat.Format32bppArgb)
    gfx = Graphics.FromImage(dst_bmp)
    clip_reg = None
    try:
        gfx.Clear(Color.White)
        if rgn_type in (2, 3):  # SIMPLEREGION=2, COMPLEXREGION=3
            clip_reg = Region.FromHrgn(_to_intptr(hrgn))
            gfx.Clip = clip_reg
        gfx.DrawImageUnscaled(src_bmp, 0, 0)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        dst_bmp.Save(str(out_path), ImageFormat.Png)
    finally:
        if clip_reg is not None:
            clip_reg.Dispose()
        gfx.Dispose()
        dst_bmp.Dispose()
        src_bmp.Dispose()
        g32.DeleteObject(hrgn)
        g32.DeleteObject(hbmp)
        g32.DeleteDC(hdc_mem)
        u32.ReleaseDC(hwnd, hdc_win)


def _compute_initial_pos(prefs: dict[str, Any], win_w: int, win_h: int) -> tuple[int, int]:
    """计算初始合法窗口坐标；若未保存或超出桌面范围则落回右下角。"""
    wa = Screen.PrimaryScreen.WorkingArea
    dpi_scale = 1.0
    with suppress_logged("win32", throttled_key="init_pos_dpi"):
        hdc = ctypes.windll.user32.GetDC(0)
        dpi = ctypes.windll.gdi32.GetDeviceCaps(hdc, 88)
        ctypes.windll.user32.ReleaseDC(0, hdc)
        if dpi > 0:
            dpi_scale = dpi / 96.0

    wa_l, wa_t = int(wa.Left / dpi_scale), int(wa.Top / dpi_scale)
    wa_r, wa_b = int(wa.Right / dpi_scale), int(wa.Bottom / dpi_scale)

    saved_x = prefs.get("x")
    saved_y = prefs.get("y")
    if isinstance(saved_x, (int, float)) and isinstance(saved_y, (int, float)):
        sx, sy = int(saved_x), int(saved_y)
        if (wa_l - 80) <= sx <= (wa_r - 80) and (wa_t - 40) <= sy <= (wa_b - 80):
            return (
                max(wa_l + 4, min(sx, wa_r - win_w - 4)),
                max(wa_t + 4, min(sy, wa_b - win_h + 4)),
            )

    return max(wa_l + 8, wa_r - win_w - 24), max(wa_t + 8, wa_b - win_h - 10)


def _init_pet_form_frame(
    window: Any, prefs: dict[str, Any], win_w: int, win_h: int, init_x: int, init_y: int
) -> None:
    """等待原生 WinForms 窗体就绪并设置背景色、置顶状态及初始物理位置尺寸。"""
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
        with suppress_logged("win32", throttled_key="init_show_window"):
            hwnd = int(form.Handle.ToInt64())
            ctypes.windll.user32.ShowWindow(hwnd, 5)  # SW_SHOW

    with suppress_logged("win32", throttled_key="init_form_frame"):
        form.Invoke(DotNetAction(_configure))
        window.resize(win_w, win_h)
        window.move(init_x, init_y)
