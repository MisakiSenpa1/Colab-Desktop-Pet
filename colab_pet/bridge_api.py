"""Colab 桌宠与 Studio 双窗 JS Bridge API、表驱动转发器与跨窗状态推送模块。

说明：
- 所有暴露给 `window.pywebview.api` 的公开方法严格保持原始形参签名（无 `_` 参数污染反射）；
- 复用 `win32_window.user32` 同一实例对象，确保跨模块与单测对 `user32` 的 patch 完全生效；
- 严禁导入 `PIL`。
"""

from __future__ import annotations

import contextlib
import ctypes
import json
import math
import os
import sys
import threading
import time
import urllib.parse
from collections.abc import Callable
from ctypes import wintypes
from typing import Any

import webview
from webview.dom import DOMEventHandler, _dnd_state

from asset_server import _set_active_media_root, start_asset_http_server
from colab_backend import ColabPetBackend, _redact_payload
from colab_core.exec_runner import suppress_logged
from win32_window import (
    BASE_H,
    BASE_W,
    Clipboard,
    DotNetAction,
    FormWindowState,
    GraphicsPath,
    Rectangle,
    Region,
    Screen,
    Size,
    _add_smooth_rounded_rect,
    _compute_edge_x_pair,
    _compute_pet_opaque_rects,
    _round_box_px,
    dwmapi,
    get_window_dpi_scale,
    user32,
)

_WAKE_ON_EVENT_TYPES = frozenset({
    "outputs_ready",
    "comfy_error",
    "deploy_done",
    "deploy_error",
    "png_workflow_choice",
    "dual_image_choice",
    "idle_warning",
    "idle_stopped",
    "idle_archive_failed",
    "onboarding_event",
})


class PetBridgeApi:
    """暴露给前端页面的 window.pywebview.api 控制接口。

    注意：所有内部成员与私有辅助方法均以下划线开头，防止 pywebview 递归遍历
    WinForms 对象属性导致栈溢出或污染反射参数签名。
    """

    def __init__(self, backend: ColabPetBackend, stop_event: threading.Event) -> None:
        self._backend = backend
        self._stop_event = stop_event
        self._window: webview.Window | None = None
        self._studio_window: webview.Window | None = None
        self._studio_unsub: Callable[[], None] | None = None
        self._studio_lock = threading.RLock()
        self._studio_create_lock = threading.Lock()
        self._studio_dragging = False
        self._studio_resizing = False
        self._asset_base_url: str = ""
        self._fallback_httpd: Any = None
        self._dragging = False
        self._in_dialog = False
        self._edge_peeking = False
        self._edge_animating = False
        self._hover_out_since: float | None = None
        self._peek_wake_until: float = time.monotonic() + 2.5
        self._edge_lock = threading.Lock()
        self._wake_event = threading.Event()
        self._visible_boxes_px: list[tuple[int, int, int, int]] = []
        _set_active_media_root(backend.download_root.resolve())

    def _bind_window(self, window: webview.Window) -> None:
        self._window = window

    def _dpi_scale(self) -> float:
        return get_window_dpi_scale(self._native_hwnd())

    def _apply_clipboard_and_redact(self, res: dict[str, Any]) -> dict[str, Any]:
        """若后端结果含非空剪贴板文本则写入系统剪贴板，并剥离原文与脱敏返回给前端。"""
        if not isinstance(res, dict):
            return res
        out = dict(res)
        clip_text = out.pop("clipboard_text", None)
        if isinstance(clip_text, str) and clip_text:
            self._set_clipboard(clip_text)
        return _redact_payload(out)

    def stop_session_instance(self, session: str = "comfy") -> dict[str, Any]:
        """精准释放指定绑定会话的 Colab 实例（绝不误杀同账号其余会话）。"""
        return self._apply_clipboard_and_redact(self._backend.stop_session_instance(session=session))

    def complete_colab_oauth(self, code: str = "", label: str = "") -> dict[str, Any]:
        """提交 Google 授权码完成绑定或新增 Ultra 账号；留空时安全读取本地剪贴板并校验。"""
        raw_code = str(code or "").strip()
        if not raw_code:
            raw_code = str(self._get_clipboard_text() or "").strip()
            if not raw_code:
                return {"ok": False, "message": "剪贴板为空，请先在浏览器复制 Google 授权码"}
        return self._backend.complete_colab_oauth(raw_code, label=label)

    def rename_colab_account(self, acct_id: str, label: str) -> dict[str, Any]:
        """修改指定已绑定账号的备注名称。"""
        return self._backend.rename_colab_account(acct_id, label)

    def save_prefs(self, updates: dict[str, Any]) -> dict[str, Any]:
        upd = dict(updates or {})
        prefs = self._backend.update_prefs(upd)
        # 仅当同进程 Studio 窗口开启且改动涉及功能偏好时，跨窗增量同步 prefs 避免覆盖实时快照
        non_pos_keys = set(upd.keys()) - {"x", "y", "left_flipped", "dock_edge", "bubble_open"}
        if non_pos_keys:
            with self._studio_lock:
                swin = self._studio_window
            if swin is not None:
                snap = self._backend.get_cached_snapshot()
                mwin = self._window
                prefs_json = json.dumps(prefs, ensure_ascii=False)
                snap_json = json.dumps(snap, ensure_ascii=False)

                def _sync() -> None:
                    if mwin is not None:
                        with suppress_logged("pet_ui", throttled_key="sync_prefs_js"):
                            mwin.evaluate_js(
                                f"(() => {{ if (typeof latestSnap !== 'undefined' && latestSnap) {{ latestSnap.prefs = Object.assign({{}}, latestSnap.prefs || {{}}, {prefs_json}); window.renderState && window.renderState(latestSnap); }} else if (window.renderState) {{ window.renderState({snap_json}); }} }})();"
                            )
                    if swin is not None:
                        _push_to_studio(swin, snap)

                threading.Thread(target=_sync, daemon=True).start()
        return prefs

    def _set_clipboard(self, text: str) -> None:
        """在 WinForms UI 线程安全写入剪贴板文本。"""
        if not text or not self._window or not self._window.native:
            return
        form = self._window.native
        with suppress_logged("pet_ui", throttled_key="set_clipboard"):
            form.Invoke(DotNetAction(lambda: Clipboard.SetText(str(text))))

    def _get_clipboard_text(self) -> str:
        """在 WinForms UI 线程安全读取系统剪贴板文本。"""
        if not self._window or not self._window.native:
            return ""
        form = self._window.native
        holder = [""]

        def _read() -> None:
            try:
                if Clipboard.ContainsText():
                    holder[0] = str(Clipboard.GetText() or "")
            except Exception:
                holder[0] = ""

        try:
            form.Invoke(DotNetAction(_read))
        except Exception:
            return ""
        return holder[0]

    def copy_text(self, text: str) -> dict[str, Any]:
        """将受限长度内（<= 512 字符）的命令或文本写入系统剪贴板。"""
        raw = str(text or "")
        if not raw:
            return {"ok": False, "message": "内容为空"}
        if len(raw) > 512:
            return {"ok": False, "message": "文本过长，已拒绝写入剪贴板"}
        self._set_clipboard(raw)
        return {"ok": True}

    def _resolve_dialog_window(self) -> webview.Window | None:
        """优先选择当前活跃且可见的窗口作为原生文件选择框宿主，避免 Studio 弹窗掉到窗后。"""
        with self._studio_lock:
            swin = self._studio_window
            shwnd = self._studio_hwnd()
        pet_win = self._window
        pet_hwnd = self._native_hwnd()
        if swin is not None and shwnd and not bool(user32.IsIconic(shwnd)):
            try:
                fg_hwnd = int(user32.GetForegroundWindow() or 0)
            except Exception:
                fg_hwnd = 0
            if fg_hwnd == shwnd or (fg_hwnd != pet_hwnd and bool(user32.IsWindowVisible(shwnd))):
                return swin
        return pet_win or swin

    def _open_single_file_dialog(self, file_types: tuple[str, ...]) -> str | None:
        dlg_win = self._resolve_dialog_window()
        if not dlg_win:
            return None
        self._in_dialog = True
        self._peek_wake_until = max(self._peek_wake_until, time.monotonic() + 4.0)
        try:
            selected = dlg_win.create_file_dialog(
                webview.FileDialog.OPEN, allow_multiple=False, file_types=file_types
            )
            return str(selected[0]) if selected else None
        finally:
            self._in_dialog = False
            self._peek_wake_until = max(self._peek_wake_until, time.monotonic() + 2.0)
            self._wake_event.set()

    def pick_rclone_conf(self) -> dict[str, Any]:
        if not self._resolve_dialog_window():
            return {"ok": False, "message": "窗口未就绪"}
        picked = self._open_single_file_dialog(("rclone 配置文件 (*.conf)", "所有文件 (*.*)"))
        return self._backend.import_rclone_conf(picked) if picked else {"ok": False, "cancelled": True, "message": "已取消选择"}

    def pick_tool_binary(self, kind: str) -> dict[str, Any]:
        if str(kind or "").strip().lower() == "rclone_conf":
            return self.pick_rclone_conf()
        if not self._resolve_dialog_window():
            return {"ok": False, "message": "窗口未就绪"}
        picked = self._open_single_file_dialog(("可执行程序 (*.exe)", "所有文件 (*.*)"))
        return self._backend.set_tool_path(kind, picked) if picked else {"ok": False, "cancelled": True, "message": "已取消选择"}

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

    def _resolve_upload_dialog_filters(self) -> tuple[str, ...]:
        """按当前活跃服务模式（dual/llm/comfy）返回本地文件选择器的过滤器元组。"""
        snap = self._backend.get_cached_snapshot()
        if snap.get("is_running"):
            sk = str((snap.get("comfy") or {}).get("service_kind") or "")
        elif self._backend.is_llm_mode_active():
            sk = "llm"
        elif self._backend.is_dual_mode_active():
            sk = "dual"
        else:
            sk = "comfy"

        if sk == "dual":
            return (
                "全能投递 (*.png;*.jpg;*.jpeg;*.webp;*.mp4;*.json;*.safetensors;*.txt;*.md;*.py)",
                "图片与视频素材 (*.png;*.jpg;*.jpeg;*.webp;*.mp4;*.json;*.safetensors)",
                "对话文本与脚本 (*.txt;*.md;*.py;*.yaml;*.yml;*.csv)",
                "所有文件 (*.*)",
            )
        if sk == "llm":
            return (
                "文本与识图文件 (*.txt;*.md;*.json;*.py;*.yaml;*.yml;*.csv;*.png;*.jpg;*.jpeg;*.webp)",
                "纯文本与代码 (*.txt;*.md;*.json;*.py;*.yaml;*.yml;*.csv)",
                "识图素材 (*.png;*.jpg;*.jpeg;*.webp;*.gif;*.bmp)",
                "所有文件 (*.*)",
            )
        return (
            "常用投递文件 (*.png;*.jpg;*.jpeg;*.webp;*.mp4;*.json;*.safetensors;*.pt)",
            "所有文件 (*.*)",
        )

    def pick_and_upload_files(self) -> dict[str, Any]:
        """打开本地文件选择器：按当前活跃服务智能切换 ComfyUI 素材或 LLM 文本/图片后缀。"""
        dlg_win = self._resolve_dialog_window()
        if not dlg_win:
            return {"ok": False, "message": "窗口未就绪"}
        dialog_file_types = self._resolve_upload_dialog_filters()
        self._in_dialog = True
        self._peek_wake_until = max(self._peek_wake_until, time.monotonic() + 4.0)
        try:
            selected = dlg_win.create_file_dialog(
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
        return self._apply_clipboard_and_redact(self._backend.submit_dropped_files([path], png_mode=mode))

    def confirm_dual_image_drop(self, paths: list[str], target: str) -> dict[str, Any]:
        """确认双栖模式下纯图片的投递目标（comfy 传参考图 / llm 打开识图对话页）。"""
        path_list = [str(p) for p in (paths or []) if p]
        if target == "llm":
            return self._apply_clipboard_and_redact(self._backend.open_llm_tunnel())
        if not path_list:
            return {"ok": False, "message": "未提供有效图片路径"}
        return self._apply_clipboard_and_redact(self._backend.submit_dropped_files(path_list, png_mode="image"))

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
        simple_actions: dict[str, Callable[[], dict[str, Any]]] = {
            "sync_outputs": self._backend.sync_outputs,
            "open_comfy": self._backend.open_comfy_tunnel,
            "open_llm_webui": self._backend.open_llm_tunnel,
            "copy_llm_api": lambda: self._backend.get_llm_api_bundle("comfy"),
        }
        handler = simple_actions.get(str(action or ""))
        if handler is not None:
            return self._apply_clipboard_and_redact(handler())
        if isinstance(action, str) and action.startswith("open_comfy_wf:"):
            wf_name = action.split(":", 1)[1].strip()
            return self._apply_clipboard_and_redact(
                self._backend.open_comfy_tunnel(workflow_name=wf_name or None)
            )
        return {"ok": False, "message": "未定义的快捷动作"}

    def set_topmost(self, flag: bool) -> dict[str, Any]:
        flag_bool = bool(flag)
        if self._window and self._window.native:
            form = self._window.native
            form.Invoke(DotNetAction(lambda: setattr(form, "TopMost", flag_bool)))
        return self._backend.update_prefs({"on_top": flag_bool})

    def _clamp_window_after_scale(self, updates: dict[str, Any]) -> None:
        """缩放后校验并回拉窗口至显示器安全工作区。"""
        if not self._window or not self._window.native:
            return
        with suppress_logged("pet_ui", throttled_key="clamp_after_scale"):
            hwnd = self._native_hwnd()
            rect = wintypes.RECT()
            if not (hwnd and user32.GetWindowRect(hwnd, ctypes.byref(rect))):
                return
            wa = Screen.FromHandle(self._window.native.Handle).WorkingArea
            dpi = self._dpi_scale()
            w_phys = max(1, int(rect.right - rect.left))
            h_phys = max(1, int(rect.bottom - rect.top))
            fx, fy = int(rect.left), int(rect.top)
            pad = int(round(4 * dpi))
            dock_edge = str((self._backend.state.get("prefs") or {}).get("dock_edge") or "").strip().lower()
            if dock_edge == "right":
                clamped_x = int(wa.Right) - w_phys - pad
            elif dock_edge == "left":
                clamped_x = int(wa.Left) + pad
            else:
                clamped_x = max(int(wa.Left) + pad, min(fx, int(wa.Right) - w_phys - pad))
            clamped_y = max(int(wa.Top) + pad, min(fy, int(wa.Bottom) - h_phys + pad))
            if clamped_x != fx or clamped_y != fy:
                # SWP_NOSIZE (0x0001) | SWP_NOZORDER (0x0004) | SWP_NOACTIVATE (0x0010)
                user32.SetWindowPos(hwnd, 0, clamped_x, clamped_y, 0, 0, 0x0001 | 0x0004 | 0x0010)
                updates["x"] = int(round(clamped_x / dpi))
                updates["y"] = int(round(clamped_y / dpi))

    def set_scale(self, scale_val: float) -> dict[str, Any]:
        """缩放桌宠窗口并确保缩放后不越出屏幕右/下工作区边缘。"""
        sc = max(0.75, min(1.35, float(scale_val)))
        updates: dict[str, Any] = {"scale": sc}
        if self._window:
            self._window.resize(int(math.ceil(BASE_W * sc)), int(math.ceil(BASE_H * sc)))
            self._clamp_window_after_scale(updates)
        self._edge_peeking = False
        self._hover_out_since = None
        self._peek_wake_until = max(self._peek_wake_until, time.monotonic() + 2.5)
        self._wake_event.set()
        return self._backend.update_prefs(updates)

    def _append_rounded_box_region(
        self,
        path: GraphicsPath,
        boxes: list[tuple[int, int, int, int]],
        box: Any,
        *,
        dpi: float,
        min_dim: int,
        radius_fn: Callable[[dict[str, Any], int, int], float],
    ) -> bool:
        """将单个 CSS 圆角盒转换为整数像素扫描条加入 GraphicsPath 并登记外接框。"""
        if not isinstance(box, dict) or box.get("w", 0) <= min_dim or box.get("h", 0) <= min_dim:
            return False
        x0, y0, w0, h0 = _round_box_px(box, dpi)
        if w0 <= min_dim or h0 <= min_dim:
            return False
        _add_smooth_rounded_rect(path, float(x0), float(y0), float(w0), float(h0), radius_fn(box, w0, h0))
        boxes.append((x0, y0, x0 + w0, y0 + h0))
        return True

    def _build_region_from_layout(
        self, layout: dict[str, Any], dpi: float, ui_scale: float, flipped: bool
    ) -> tuple[GraphicsPath, list[tuple[int, int, int, int]]]:
        """将前端气泡、圆点、悬浮胶囊与立绘不透明轮廓组装为 .NET GraphicsPath。"""
        path = GraphicsPath()
        boxes: list[tuple[int, int, int, int]] = []

        # 1) 顶部思维气泡圆角矩形
        self._append_rounded_box_region(
            path,
            boxes,
            layout.get("bubble"),
            dpi=dpi,
            min_dim=4,
            radius_fn=lambda b, _w, _h: float(
                b.get("r", 16.0 * ui_scale * max(0.2, min(1.2, float(b.get("scale", 1.0)))))
            ) * dpi,
        )
        # 2) 思维圆点
        for dot in layout.get("dots") or []:
            self._append_rounded_box_region(
                path, boxes, dot, dpi=dpi, min_dim=1, radius_fn=lambda _b, w, h: min(w, h) * 0.5
            )
        # 3) 常驻迷你操作胶囊、迷你余额条与互动台词气泡
        for pill_key in ("tools", "miniPill", "toast"):
            self._append_rounded_box_region(
                path,
                boxes,
                layout.get(pill_key),
                dpi=dpi,
                min_dim=2,
                radius_fn=lambda b, w, h: float(b["r"]) * dpi if isinstance(b.get("r"), (int, float)) else min(w, h) * 0.5,
            )
        # 4) 人物立绘精准不透明像素轮廓（区分眨眼帧与水平镜像）
        pet = layout.get("pet")
        if isinstance(pet, dict) and pet.get("w", 0) > 8 and pet.get("h", 0) > 8:
            px, py = int(round(pet["x"] * dpi)), int(round(pet["y"] * dpi))
            pw, ph = max(1, int(round(pet["w"] * dpi))), max(1, int(round(pet["h"] * dpi)))
            for rx, ry, rw, rh in _compute_pet_opaque_rects(pw, ph, flipped, bool(pet.get("blink", False))):
                path.AddRectangle(Rectangle(px + rx, py + ry, rw, rh))
            boxes.append((px, py, px + pw, py + ph))
        return path, boxes

    def sync_region(self, layout: dict[str, Any]) -> None:
        """根据前端各可见元素精确边界，实时更新 WinForms Form.Region 裁剪轮廓并记录可见外接框。"""
        if not self._window or not self._window.native:
            return
        form = self._window.native
        dpi = self._dpi_scale()
        flipped = bool(layout.get("left_flipped", False))
        ui_scale = float((self._backend.state.get("prefs") or {}).get("scale", 1.0))

        path, boxes = self._build_region_from_layout(layout, dpi, ui_scale, flipped)
        if not boxes:
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

    def _compute_drag_dock_snap(
        self, final_rect: wintypes.RECT, wa: Any, dpi: float, prefs: dict[str, Any]
    ) -> tuple[int, int, str | None, bool, bool]:
        """结合真实可见边界计算拖拽松手后的磁吸位置、停靠边、镜像翻转与过界推边标记。"""
        w_phys = final_rect.right - final_rect.left
        h_phys = final_rect.bottom - final_rect.top
        ui_scale = max(0.65, min(1.35, float(prefs.get("scale", 1.0) or 1.0)))
        bubble_open = bool(prefs.get("bubble_open", True))

        fx_before = int(final_rect.left)
        fx, fy = fx_before, int(final_rect.top)
        vis_left = fx_before + int(round((18.0 if bubble_open else 71.0) * ui_scale * dpi))
        vis_right = fx_before + int(round((374.0 if bubble_open else 356.0) * ui_scale * dpi))

        snap_phys = int(round(64.0 * dpi))
        wa_l, wa_r = int(wa.Left), int(wa.Right)
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
        mid_wa = (wa_l + wa_r) / 2.0
        left_flipped = (dock_edge == "left") if dock_edge else ((fx + w_phys / 2.0) < mid_wa)
        return fx, fy, dock_edge, left_flipped, pushed_past

    def _finalize_pet_drag(self, hwnd: int) -> None:
        """拖拽松手后执行磁吸落位、更新持久化坐标并通知前端刷新镜像状态。"""
        if not self._window or not self._window.native:
            return
        final_rect = wintypes.RECT()
        user32.GetWindowRect(hwnd, ctypes.byref(final_rect))
        wa = Screen.FromHandle(self._window.native.Handle).WorkingArea
        dpi = self._dpi_scale()
        prefs = self._backend.state.get("prefs") or {}

        fx, fy, dock_edge, left_flipped, pushed_past = self._compute_drag_dock_snap(final_rect, wa, dpi, prefs)
        user32.SetWindowPos(hwnd, 0, fx, fy, 0, 0, 0x0001 | 0x0004 | 0x0010)

        hold_sec = 0.25 if (dock_edge and pushed_past) else 0.55
        self._edge_peeking = False
        self._hover_out_since = None
        self._peek_wake_until = max(self._peek_wake_until, time.monotonic() + hold_sec)
        self._backend.update_prefs(
            {
                "x": int(round(fx / dpi)),
                "y": int(round(fy / dpi)),
                "left_flipped": left_flipped,
                "dock_edge": dock_edge,
            }
        )
        self._wake_event.set()
        if not self._stop_event.is_set():
            flag_js = "true" if left_flipped else "false"
            dock_js = json.dumps(dock_edge)
            with suppress_logged("pet_ui", throttled_key="drag_finished_js"):
                self._window.evaluate_js(f"window.onDragFinished && window.onDragFinished({flag_js}, {dock_js});")

    def begin_drag(self) -> None:
        """接管鼠标左键拖拽：脱离浏览器视口束缚，原生 Win32 跟随并贴边吸附。"""
        if self._dragging or not self._window or not self._window.native:
            return
        hwnd = self._native_hwnd()
        if not hwnd:
            return
        self._dragging = True
        vk = 0x02 if user32.GetSystemMetrics(23) else 0x01  # 兼容左右手鼠标主按键

        pt0, rect0 = wintypes.POINT(), wintypes.RECT()
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
                self._finalize_pet_drag(hwnd)
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
            self._wake_from_edge_peek(hold_sec=2.0)
        else:
            self._hover_out_since = None
            self._peek_wake_until = max(self._peek_wake_until, time.monotonic() + 0.6)
            self._wake_event.set()
        return res

    def _wake_from_edge_peek(self, hold_sec: float = 4.0) -> None:
        """从贴边半隐藏状态平顺唤醒回屏幕可见区并延长防隐藏窗口。"""
        self._peek_wake_until = max(self._peek_wake_until, time.monotonic() + max(0.5, float(hold_sec)))
        self._hover_out_since = None
        self._wake_event.set()
        if self._edge_peeking and not self._edge_animating:
            threading.Thread(target=self._set_peek_state, args=(False, "wake"), daemon=True).start()

    def _is_cursor_on_visible_ui(self, pt_x: int, pt_y: int, rect: wintypes.RECT, dpi: float) -> bool:
        """基于 sync_region 真实可见元素外接框判定光标是否悬停在桌宠可见部分上。"""
        boxes = self._visible_boxes_px
        if boxes:
            rx, ry = int(pt_x) - int(rect.left), int(pt_y) - int(rect.top)
            pad = int(round(14.0 * dpi))
            return any((x0 - pad) <= rx <= (x1 + pad) and (y0 - pad) <= ry <= (y1 + pad) for x0, y0, x1, y1 in boxes)
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
        wa_l, wa_r = int(wa.Left), int(wa.Right)

        # 1) 露头角色宽幅热区
        hot_w = int(round(96.0 * ui_scale * dpi))
        char_y_top = int(rect.top) + int(0.45 * h_phys) - int(round(16.0 * dpi))
        char_y_bot = int(rect.bottom) + int(round(16.0 * dpi))
        in_char_x = ((wa_r - hot_w) <= pt_x <= (wa_r + out_tol)) if dock_edge == "right" else ((wa_l - out_tol) <= pt_x <= (wa_l + hot_w))
        if in_char_x and (char_y_top <= pt_y <= char_y_bot):
            return True

        # 2) 全高贴边唤醒窄条（鼠标甩到气泡原位所在屏幕边缘也能秒唤醒）
        edge_strip_w = int(round(18.0 * dpi))
        strip_y_top = int(rect.top) - int(round(8.0 * dpi))
        strip_y_bot = int(rect.bottom) + int(round(12.0 * dpi))
        in_strip_x = ((wa_r - edge_strip_w) <= pt_x <= (wa_r + out_tol)) if dock_edge == "right" else ((wa_l - out_tol) <= pt_x <= (wa_l + edge_strip_w))
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
        if dock_edge in ("left", "right") and self._is_cursor_in_wake_zone(pt_x, pt_y, rect, wa, dock_edge, ui_scale, dpi):
            return True
        boxes = self._visible_boxes_px
        if len(boxes) >= 2:
            pad = int(round(14.0 * dpi))
            rx, ry = int(pt_x) - int(rect.left), int(pt_y) - int(rect.top)
            return (
                (min(b[0] for b in boxes) - pad) <= rx <= (max(b[2] for b in boxes) + pad)
                and (min(b[1] for b in boxes) - pad) <= ry <= (max(b[3] for b in boxes) + pad)
            )
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
            if abort_on_wake and self._should_abort_peek_animation(wake_check):
                return False
            p = min(1.0, (time.perf_counter() - t0) / dur)
            # ease-out cubic：起始略快、落位轻柔减速
            eased = 1.0 - (1.0 - p) ** 3
            cx = int(round(start_x + (end_x - start_x) * eased))
            user32.SetWindowPos(hwnd, 0, cx, int(y_phys), 0, 0, flags)
            if p >= 1.0:
                return True
            time.sleep(0.008)
        return False

    def _should_abort_peek_animation(self, wake_check: Callable[[], bool] | None) -> bool:
        """判定贴边隐藏动画期间是否应立即打断并回弹。"""
        if self._in_dialog or time.monotonic() < self._peek_wake_until:
            return True
        if wake_check is None:
            return False
        try:
            return bool(wake_check())
        except Exception:
            return False

    def _can_auto_hide_ui(self) -> bool:
        if not self._window or self._stop_event.is_set():
            return False
        try:
            res = self._window.evaluate_js("window.canEdgeAutoHide ? window.canEdgeAutoHide() : true;")
            return bool(res) if res is not None else True
        except Exception:
            return True

    def _notify_edge_peek_ui(self, peeking: bool) -> None:
        """安全通知桌宠前端切换半隐藏视觉状态。"""
        if not self._window:
            return
        flag_js = "true" if peeking else "false"
        with suppress_logged("pet_ui", throttled_key="edge_peek_js"):
            self._window.evaluate_js(f"window.onEdgePeekChanged && window.onEdgePeekChanged({flag_js});")

    def _make_cursor_return_checker(
        self, hwnd: int, wa: Any, edge_key: str, ui_scale: float, dpi: float
    ) -> Callable[[], bool]:
        """构造检测鼠标在收起动画期间是否重新回到桌宠热区的回调。"""
        init_pt = wintypes.POINT()
        has_init_pt = bool(user32.GetCursorPos(ctypes.byref(init_pt)))
        move_thresh_sq = float(int(round(4.0 * dpi)) ** 2)

        def _cursor_returned_to_pet() -> bool:
            cur_pt, cur_r = wintypes.POINT(), wintypes.RECT()
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
            ) or self._is_cursor_on_visible_ui(int(cur_pt.x), int(cur_pt.y), cur_r, dpi)

        return _cursor_returned_to_pet

    def _wait_before_peek_slide(self, wait_sec: float, returned_check: Callable[[], bool]) -> bool:
        """气泡收起缓冲等待，期间如发生用户交互或鼠标回移则返回 False 终止贴边隐藏。"""
        deadline = time.monotonic() + wait_sec
        while time.monotonic() < deadline:
            if (
                self._stop_event.is_set()
                or self._dragging
                or self._in_dialog
                or time.monotonic() < self._peek_wake_until
                or returned_check()
            ):
                self._peek_wake_until = max(self._peek_wake_until, time.monotonic() + 0.55)
                self._hover_out_since = None
                self._notify_edge_peek_ui(False)
                return False
            time.sleep(0.015)
        return True

    def _perform_peek_hide(
        self,
        hwnd: int,
        wa: Any,
        edge_key: str,
        ui_scale: float,
        dpi: float,
        x_visible: int,
        x_peek: int,
        bubble_was_open: bool,
    ) -> bool:
        """执行贴边半隐藏平滑滑出，若中途被打断则立即平滑回弹至可见停靠位。"""
        if not self._can_auto_hide_ui():
            return False
        returned_check = self._make_cursor_return_checker(hwnd, wa, edge_key, ui_scale, dpi)
        self._notify_edge_peek_ui(True)

        # 仅在气泡原本展开时短暂等待收起动画，气泡已收起时极速起滑
        if not self._wait_before_peek_slide(0.10 if bubble_was_open else 0.02, returned_check):
            return False

        rect = wintypes.RECT()
        user32.GetWindowRect(hwnd, ctypes.byref(rect))
        ok = self._animate_window_x(
            hwnd, rect.left, x_peek, rect.top, duration=0.16, abort_on_wake=True, wake_check=returned_check
        )
        if not ok:
            self._peek_wake_until = max(self._peek_wake_until, time.monotonic() + 0.55)
            self._hover_out_since = None
            self._edge_peeking = False
            self._notify_edge_peek_ui(False)
            user32.GetWindowRect(hwnd, ctypes.byref(rect))
            self._animate_window_x(hwnd, rect.left, x_visible, rect.top, duration=0.14, abort_on_wake=False)
            return False
        self._edge_peeking = True
        return True

    def _perform_peek_wake(self, hwnd: int, dock_edge: str, x_visible: int, dpi: float) -> bool:
        """并行唤醒：先通知前端展开气泡，同时平滑滑回全显停靠坐标。"""
        self._edge_peeking = False
        self._hover_out_since = None
        self._notify_edge_peek_ui(False)
        cur_rect = wintypes.RECT()
        if user32.GetWindowRect(hwnd, ctypes.byref(cur_rect)) and dock_edge in ("left", "right"):
            self._animate_window_x(hwnd, cur_rect.left, x_visible, cur_rect.top, duration=0.15, abort_on_wake=False)
            self._backend.update_prefs({"x": int(round(x_visible / dpi))})
        return True

    def _set_peek_state(self, peeking: bool, reason: str = "") -> bool:
        """切换贴边半隐藏状态：支持收起中途鼠标移回即时回弹与唤醒时气泡/窗口并行展开。"""
        _ = reason
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
            wa = Screen.FromHandle(self._window.native.Handle).WorkingArea
            dpi = self._dpi_scale()
            ui_scale = max(0.65, min(1.35, float(prefs.get("scale", 1.0) or 1.0)))
            edge_key = dock_edge if dock_edge in ("left", "right") else "right"
            x_visible, x_peek = _compute_edge_x_pair(
                edge_key, int(wa.Left), int(wa.Right), rect.right - rect.left, ui_scale, dpi
            )

            self._edge_animating = True
            try:
                if peeking:
                    return self._perform_peek_hide(
                        hwnd, wa, edge_key, ui_scale, dpi, x_visible, x_peek, bool(prefs.get("bubble_open", True))
                    )
                return self._perform_peek_wake(hwnd, dock_edge, x_visible, dpi)
            finally:
                self._edge_animating = False

    def _handle_peeking_watch_tick(
        self, pt: wintypes.POINT, rect: wintypes.RECT, wa: Any, dock_edge: str, ui_scale: float, dpi: float, ts: float
    ) -> None:
        """半隐藏状态下的单次巡检：若被外部拉回则自愈复位，若进入唤醒热区则立即滑出。"""
        x_visible, _ = _compute_edge_x_pair(
            dock_edge, int(wa.Left), int(wa.Right), max(1, rect.right - rect.left), ui_scale, dpi
        )
        if abs(int(rect.left) - x_visible) <= int(round(8.0 * dpi)):
            self._edge_peeking = False
            self._hover_out_since = None
            self._peek_wake_until = max(self._peek_wake_until, ts + 0.8)
            self._notify_edge_peek_ui(False)
            return
        if self._is_cursor_in_wake_zone(int(pt.x), int(pt.y), rect, wa, dock_edge, ui_scale, dpi):
            self._peek_wake_until = max(self._peek_wake_until, ts + 0.50)
            self._set_peek_state(False, reason="hover")

    def _is_edge_watch_paused(self) -> bool:
        """判定当前是否处于拖拽、弹窗、动画或停止期，应跳过贴边巡检。"""
        return bool(self._dragging or self._in_dialog or self._edge_animating or self._stop_event.is_set())

    def _edge_watch_tick(self, now: float | None = None) -> None:
        """单次巡检鼠标与停靠边位置，触发贴边自动半隐藏或悬停唤醒。"""
        if self._is_edge_watch_paused():
            return
        hwnd = self._native_hwnd()
        if not hwnd or not self._window or not self._window.native:
            return
        prefs = self._backend.state.get("prefs") or {}
        dock_edge = str(prefs.get("dock_edge") or "").strip().lower()
        if dock_edge not in ("left", "right") or not bool(prefs.get("edge_auto_hide", True)):
            if self._edge_peeking:
                self._set_peek_state(False, reason="pref_off")
            self._hover_out_since = None
            return

        pt, rect = wintypes.POINT(), wintypes.RECT()
        if not user32.GetCursorPos(ctypes.byref(pt)) or not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
            return
        wa = Screen.FromHandle(self._window.native.Handle).WorkingArea
        dpi = self._dpi_scale()
        ui_scale = max(0.65, min(1.35, float(prefs.get("scale", 1.0) or 1.0)))
        ts = time.monotonic() if now is None else float(now)

        if self._edge_peeking:
            self._handle_peeking_watch_tick(pt, rect, wa, dock_edge, ui_scale, dpi, ts)
            return

        if self._is_cursor_keeping_awake(int(pt.x), int(pt.y), rect, wa, dock_edge, ui_scale, dpi):
            self._hover_out_since = None
            return
        if self._hover_out_since is None:
            self._hover_out_since = ts
        if ts >= self._peek_wake_until and (ts - self._hover_out_since) >= 0.26:
            self._set_peek_state(True, reason="idle_leave")

    def _sync_initial_dock_edge(self) -> None:
        """启动停靠自检：若未保存 dock_edge 但位置已在屏幕左右边 16*dpi 内，则自动补全停靠边。"""
        prefs = self._backend.state.get("prefs") or {}
        if prefs.get("dock_edge") in ("left", "right"):
            return
        hwnd = self._native_hwnd()
        if not hwnd or not self._window or not self._window.native:
            return
        with suppress_logged("pet_ui", throttled_key="sync_initial_dock"):
            rect = wintypes.RECT()
            if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
                return
            wa = Screen.FromHandle(self._window.native.Handle).WorkingArea
            edge_thresh = int(round(16.0 * self._dpi_scale()))
            if abs(int(rect.left) - int(wa.Left)) <= edge_thresh:
                self._backend.update_prefs({"dock_edge": "left", "left_flipped": True})
            elif abs(int(wa.Right) - int(rect.right)) <= edge_thresh:
                self._backend.update_prefs({"dock_edge": "right", "left_flipped": False})

    def _edge_watch_loop(self) -> None:
        """后台高频检测光标停靠状态（测试渲染模式下自动跳过）。"""
        if "--test-render" in sys.argv:
            return
        if self._window:
            with suppress_logged("pet_ui", throttled_key="edge_watch_wait_loaded"):
                self._window.events.loaded.wait(timeout=5.0)
        last_err_ts = 0.0
        while not self._stop_event.is_set():
            try:
                self._edge_watch_tick()
            except Exception as exc:
                now = time.monotonic()
                if now - last_err_ts >= 10.0:
                    last_err_ts = now
                    # 无控制台 GUI 场景忽略 stderr 写入失败
                    with contextlib.suppress(OSError, ValueError):
                        print(f"[edge_watch] 巡检异常: {exc!r}", file=sys.stderr)
            prefs = self._backend.state.get("prefs") or {}
            dock_edge = str(prefs.get("dock_edge") or "").strip().lower()
            auto_hide = bool(prefs.get("edge_auto_hide", True))
            wait_sec = 2.0 if (not auto_hide or dock_edge not in ("left", "right")) and not (self._edge_peeking or self._dragging) else 0.085
            self._wake_event.wait(timeout=wait_sec)
            self._wake_event.clear()

    @staticmethod
    def _window_dpi_scale(hwnd: int) -> float:
        """查询指定窗口句柄的物理 DPI 缩放倍率。"""
        return get_window_dpi_scale(hwnd)

    def _studio_hwnd(self) -> int:
        """获取当前 Studio 窗口原生 HWND 并验证有效性，失效则返回 0。"""
        win = self._studio_window
        if not win or not win.native:
            return 0
        try:
            hwnd = int(win.native.Handle.ToInt64())
        except Exception:
            try:
                hwnd = int(win.native.Handle)
            except Exception:
                return 0
        return hwnd if hwnd and bool(user32.IsWindow(hwnd)) else 0

    def _resolve_asset_base_url(self) -> str:
        """解析本地资产 HTTP 服务基础地址（优先复用桌宠同源端口，必要时自动补建）。"""
        if self._asset_base_url:
            return self._asset_base_url
        if self._window is not None:
            for attr in ("original_url", "real_url"):
                raw_u = str(getattr(self._window, attr, "") or "").strip()
                if raw_u.startswith(("http://", "https://")):
                    parsed = urllib.parse.urlsplit(raw_u)
                    if parsed.scheme and parsed.netloc:
                        self._asset_base_url = f"{parsed.scheme}://{parsed.netloc}"
                        return self._asset_base_url
        httpd, port = start_asset_http_server(self._backend.download_root.resolve())
        self._fallback_httpd = httpd
        self._asset_base_url = f"http://127.0.0.1:{port}"
        return self._asset_base_url

    @staticmethod
    def _apply_studio_dwm_style(studio_win: webview.Window, *, ensure_initial_size: bool = False) -> None:
        """为无边框 Studio 窗体启用 Win11 原生圆角（严禁设置 Form.Region）、校准初始尺寸并约束最大化工作区。"""
        if not studio_win:
            return
        for _ in range(40):
            if studio_win.native is not None:
                break
            time.sleep(0.025)
        if not studio_win.native:
            return
        form = studio_win.native

        def _cfg() -> None:
            try:
                hwnd = int(form.Handle.ToInt64())
            except Exception:
                hwnd = 0
            working_area = None
            with suppress_logged("studio_ui", throttled_key="studio_max_bounds"):
                working_area = Screen.FromHandle(form.Handle).WorkingArea
                form.MaximizedBounds = working_area
            dpi = PetBridgeApi._window_dpi_scale(hwnd)
            min_w_phys, min_h_phys = int(round(920 * dpi)), int(round(600 * dpi))
            with suppress_logged("studio_ui", throttled_key="studio_min_size"):
                if form.MinimumSize.Width < min_w_phys or form.MinimumSize.Height < min_h_phys:
                    form.MinimumSize = Size(min_w_phys, min_h_phys)
            if ensure_initial_size and hwnd:
                PetBridgeApi._center_studio_initial_bounds(form, hwnd, working_area, dpi, min_w_phys, min_h_phys)
            if hwnd and dwmapi is not None:
                with suppress_logged("studio_ui", throttled_key="studio_dwm_round"):
                    corner_pref = ctypes.c_int(2)  # DWMWCP_ROUND
                    dwmapi.DwmSetWindowAttribute(hwnd, 33, ctypes.byref(corner_pref), 4)

        with suppress_logged("studio_ui", throttled_key="studio_dwm_invoke"):
            form.Invoke(DotNetAction(_cfg))

    @staticmethod
    def _center_studio_initial_bounds(
        form: Any, hwnd: int, working_area: Any, dpi: float, min_w_phys: int, min_h_phys: int
    ) -> None:
        """补偿 WinForms 无边框建窗尺寸缩水并在当前屏幕工作区居中显示。"""
        with suppress_logged("studio_ui", throttled_key="studio_center_bounds"):
            if form.WindowState != FormWindowState.Normal:
                return
            target_w, target_h = int(round(1360 * dpi)), int(round(840 * dpi))
            if working_area is not None:
                target_w = min(target_w, max(min_w_phys, int(working_area.Width) - 24))
                target_h = min(target_h, max(min_h_phys, int(working_area.Height) - 24))
                pos_x = int(working_area.X) + max(0, (int(working_area.Width) - target_w) // 2)
                pos_y = int(working_area.Y) + max(0, (int(working_area.Height) - target_h) // 2)
                user32.SetWindowPos(hwnd, 0, pos_x, pos_y, target_w, target_h, 0x0004 | 0x0010)
            else:
                user32.SetWindowPos(hwnd, 0, 0, 0, target_w, target_h, 0x0002 | 0x0004 | 0x0010)

    def _close_studio_window_internal(self) -> None:
        """随主桌宠退出时联动销毁 Studio 窗体并清理事件订阅与临时服务。"""
        with self._studio_lock:
            unsub = self._studio_unsub
            self._studio_unsub = None
            win = self._studio_window
            self._studio_window = None
        if unsub is not None:
            with suppress_logged("studio_ui", throttled_key="studio_unsub_close"):
                unsub()
        if win is not None:
            with suppress_logged("studio_ui", throttled_key="studio_destroy_close"):
                win.destroy()
        if self._fallback_httpd is not None:
            # 退出清理忽略本地回环 HTTP 兜底服务重复关闭异常
            with contextlib.suppress(OSError, RuntimeError, AttributeError):
                self._fallback_httpd.shutdown()
                self._fallback_httpd.server_close()
            self._fallback_httpd = None

    def _bring_studio_to_front(self, existing_win: webview.Window, existing_hwnd: int, clean_tab: str) -> None:
        """唤醒并置前已有 Studio 窗口，按需切换到目标页签或展开副驾。"""
        if existing_win.native:
            form = existing_win.native

            def _bring_front() -> None:
                with suppress_logged("studio_ui", throttled_key="studio_bring_front"):
                    user32.ShowWindow(existing_hwnd, 9 if user32.IsIconic(existing_hwnd) else 5)
                    form.Show()
                    form.Activate()
                    user32.SetForegroundWindow(existing_hwnd)

            with suppress_logged("studio_ui", throttled_key="studio_bring_front_invoke"):
                form.Invoke(DotNetAction(_bring_front))
        if not clean_tab:
            return
        with suppress_logged("studio_ui", throttled_key="studio_switch_tab_js"):
            if clean_tab.lower() == "copilot":
                existing_win.evaluate_js("window.openStudioCopilot && window.openStudioCopilot();")
            else:
                tab_js = json.dumps(clean_tab, ensure_ascii=False)
                existing_win.evaluate_js(f"window.switchStudioTab && window.switchStudioTab({tab_js});")

    def _create_studio_webview_window(self, studio_url: str) -> tuple[webview.Window | None, str | None]:
        """创建无边框 Studio PyWebView 窗口（若位于主线程则经子线程安全触发运行期建窗）。"""
        win_box: list[webview.Window | None] = [None]
        err_box: list[BaseException] = []

        def _do_create() -> None:
            try:
                win_box[0] = webview.create_window(
                    title="Colab Studio",
                    url=studio_url,
                    js_api=self,
                    width=1360,
                    height=840,
                    min_size=(920, 600),
                    frameless=True,
                    easy_drag=False,
                    background_color="#F6F8FB",
                    focus=True,
                )
            except BaseException as exc:
                err_box.append(exc)

        if threading.current_thread().name == "MainThread":
            t = threading.Thread(target=_do_create, daemon=True)
            t.start()
            t.join(timeout=15.0)
        else:
            _do_create()

        if err_box:
            return None, f"创建 Studio 窗口失败: {err_box[0]}"
        if win_box[0] is None:
            return None, "创建 Studio 窗口超时"
        return win_box[0], None

    def _attach_studio_window_lifecycle(self, studio_win: webview.Window) -> None:
        """为新建 Studio 窗口挂载后端事件多播订阅、DOM 拖拽与独立关闭生命周期（关闭不终止桌宠）。"""

        def _on_studio_event(evt: dict[str, Any]) -> None:
            if self._stop_event.is_set():
                return
            with self._studio_lock:
                cur_win = self._studio_window
            if cur_win is not None:
                _push_event_to_studio(cur_win, evt)

        studio_loaded_once = threading.Event()

        def _on_studio_loaded() -> None:
            if studio_loaded_once.is_set():
                return
            studio_loaded_once.set()
            self._apply_studio_dwm_style(studio_win, ensure_initial_size=True)
            with suppress_logged("studio_ui", throttled_key="studio_dom_drop_bind"):
                doc = studio_win.dom.document
                doc.events.dragover += DOMEventHandler(lambda _e: None, prevent_default=True, stop_propagation=False)
                doc.events.drop += DOMEventHandler(self._on_dom_drop, prevent_default=True, stop_propagation=False)
            with suppress_logged("studio_ui", throttled_key="studio_initial_push"):
                _push_to_studio(studio_win, self._backend.get_cached_snapshot())

        def _on_studio_closed() -> None:
            with self._studio_lock:
                unsub_fn = self._studio_unsub
                self._studio_unsub = None
                if self._studio_window is studio_win:
                    self._studio_window = None
            if unsub_fn is not None:
                with suppress_logged("studio_ui", throttled_key="studio_unsub_on_closed"):
                    unsub_fn()

        with self._studio_lock:
            self._studio_window = studio_win
            self._studio_unsub = self._backend.subscribe_events(_on_studio_event)

        studio_win.events.loaded += _on_studio_loaded
        studio_win.events.closed += _on_studio_closed
        self._apply_studio_dwm_style(studio_win, ensure_initial_size=True)
        if studio_win.events.loaded.is_set():
            _on_studio_loaded()

    def open_studio_window(self, tab: str = "") -> dict[str, Any]:
        """在同进程单 Backend 下打开或置前唤醒无边框 Colab Studio 工作台窗口。"""
        clean_tab = str(tab or "").strip()
        with self._studio_create_lock:
            with self._studio_lock:
                existing_win = self._studio_window
                existing_hwnd = self._studio_hwnd()
                if existing_win is not None and existing_hwnd:
                    self._bring_studio_to_front(existing_win, existing_hwnd, clean_tab)
                    return {
                        "ok": True,
                        "reused": True,
                        "pid": os.getpid(),
                        "backend_id": id(self._backend),
                        "tab": clean_tab,
                    }
                if self._studio_unsub is not None:
                    with suppress_logged("studio_ui", throttled_key="studio_unsub_reopen"):
                        self._studio_unsub()
                    self._studio_unsub = None
                self._studio_window = None

            base_url = self._resolve_asset_base_url().rstrip("/")
            studio_url = f"{base_url}/studio.html" + (f"?tab={urllib.parse.quote(clean_tab)}" if clean_tab else "")
            studio_win, err_msg = self._create_studio_webview_window(studio_url)
            if studio_win is None:
                return {"ok": False, "message": err_msg or "创建 Studio 窗口失败"}

            self._attach_studio_window_lifecycle(studio_win)
            return {
                "ok": True,
                "reused": False,
                "pid": os.getpid(),
                "backend_id": id(self._backend),
                "tab": clean_tab,
            }

    def _studio_ctl_close(self, win: webview.Window, form: Any) -> dict[str, Any]:
        """关闭 Studio 窗口并注销事件订阅，保持主桌宠与后端继续在线。"""
        with self._studio_lock:
            unsub_fn = self._studio_unsub
            self._studio_unsub = None
            self._studio_window = None
        if unsub_fn is not None:
            with suppress_logged("studio_ui", throttled_key="studio_ctl_unsub"):
                unsub_fn()
        try:
            win.destroy()
        except Exception:
            with suppress_logged("studio_ui", throttled_key="studio_ctl_form_close"):
                form.Invoke(DotNetAction(form.Close))
        return {"ok": True, "action": "close", "closed": True}

    def _studio_ctl_toggle_maximize(self, form: Any) -> dict[str, Any]:
        """切换 Studio 窗口最大化与还原状态（遵循当前屏工作区边界）。"""
        state_box = [False]

        def _toggle() -> None:
            if form.WindowState == FormWindowState.Maximized:
                form.WindowState = FormWindowState.Normal
                state_box[0] = False
            else:
                with suppress_logged("studio_ui", throttled_key="studio_max_bounds_toggle"):
                    form.MaximizedBounds = Screen.FromHandle(form.Handle).WorkingArea
                form.WindowState = FormWindowState.Maximized
                state_box[0] = True

        with suppress_logged("studio_ui", throttled_key="studio_toggle_max_invoke"):
            form.Invoke(DotNetAction(_toggle))
        return {"ok": True, "action": "toggle_maximize", "maximized": state_box[0], "minimized": False}

    def _studio_ctl_read_state(self, form: Any, hwnd: int) -> dict[str, Any]:
        """读取 Studio 窗口当前开关、最大化/最小化与逻辑宽高状态。"""
        st_box: dict[str, Any] = {"maximized": False, "minimized": False}

        def _read_state() -> None:
            st_box["maximized"] = form.WindowState == FormWindowState.Maximized
            st_box["minimized"] = form.WindowState == FormWindowState.Minimized

        with suppress_logged("studio_ui", throttled_key="studio_read_state"):
            form.Invoke(DotNetAction(_read_state))
        rect = wintypes.RECT()
        w_px, h_px = 0, 0
        if user32.GetWindowRect(hwnd, ctypes.byref(rect)):
            dpi = self._window_dpi_scale(hwnd)
            w_px = int(round((rect.right - rect.left) / dpi))
            h_px = int(round((rect.bottom - rect.top) / dpi))
        return {
            "ok": True,
            "open": True,
            "maximized": st_box["maximized"],
            "minimized": st_box["minimized"],
            "width": w_px,
            "height": h_px,
        }

    def studio_window_ctl(self, action: str) -> dict[str, Any]:
        """控制 Studio 窗口生命周期与几何状态（close / minimize / toggle_maximize / get_window_state）。"""
        act = str(action or "").strip().lower()
        with self._studio_lock:
            win = self._studio_window
            hwnd = self._studio_hwnd()
        if win is None or not hwnd or not win.native:
            if act == "get_window_state":
                return {"ok": True, "open": False, "maximized": False, "minimized": False}
            return {"ok": False, "open": False, "message": "Studio 窗口未打开"}

        form = win.native
        if act == "close":
            return self._studio_ctl_close(win, form)
        if act == "minimize":
            try:
                win.minimize()
            except Exception:
                with suppress_logged("studio_ui", throttled_key="studio_minimize_fallback"):
                    form.Invoke(DotNetAction(lambda: setattr(form, "WindowState", FormWindowState.Minimized)))
            return {"ok": True, "action": "minimize", "minimized": True, "maximized": False}
        if act == "toggle_maximize":
            return self._studio_ctl_toggle_maximize(form)
        if act == "get_window_state":
            return self._studio_ctl_read_state(form, hwnd)
        return {"ok": False, "message": f"未知窗口控制动作: {act}"}

    def begin_studio_drag(self) -> None:
        """接管 Studio 标题栏拖拽：使用原生 Win32 GetCursorPos + SetWindowPos 高帧率跟随。"""
        with self._studio_lock:
            win = self._studio_window
            hwnd = self._studio_hwnd()
        if self._studio_dragging or win is None or not hwnd or not win.native:
            return
        self._studio_dragging = True
        form = win.native
        vk = 0x02 if user32.GetSystemMetrics(23) else 0x01
        pt0, rect0 = wintypes.POINT(), wintypes.RECT()
        user32.GetCursorPos(ctypes.byref(pt0))
        user32.GetWindowRect(hwnd, ctypes.byref(rect0))

        # 若最大化状态下拖动标题栏，先还原为正常窗口再跟随光标
        with suppress_logged("studio_ui", throttled_key="studio_drag_restore"):
            is_max_box = [False]
            form.Invoke(DotNetAction(lambda: is_max_box.__setitem__(0, form.WindowState == FormWindowState.Maximized)))
            if is_max_box[0]:
                form.Invoke(DotNetAction(lambda: setattr(form, "WindowState", FormWindowState.Normal)))
                user32.GetWindowRect(hwnd, ctypes.byref(rect0))
                norm_w = max(1, rect0.right - rect0.left)
                user32.SetWindowPos(hwnd, 0, int(pt0.x - norm_w // 2), int(pt0.y - 18), 0, 0, 0x0001 | 0x0004 | 0x0010)
                user32.GetWindowRect(hwnd, ctypes.byref(rect0))

        def _drag_worker() -> None:
            try:
                cur = wintypes.POINT()
                while not self._stop_event.is_set() and user32.IsWindow(hwnd) and (user32.GetAsyncKeyState(vk) & 0x8000):
                    user32.GetCursorPos(ctypes.byref(cur))
                    nx, ny = rect0.left + (cur.x - pt0.x), rect0.top + (cur.y - pt0.y)
                    user32.SetWindowPos(hwnd, 0, nx, ny, 0, 0, 0x0001 | 0x0004 | 0x0010)
                    time.sleep(0.008)
            finally:
                self._studio_dragging = False

        threading.Thread(target=_drag_worker, daemon=True).start()

    def begin_studio_resize(self, edge: str) -> None:
        """支持 n/s/e/w/nw/ne/sw/se 8 方向原生平滑缩放 Studio 窗口（受最小尺寸 920x600 约束）。"""
        clean_edge = str(edge or "").strip().lower()
        if clean_edge not in ("n", "s", "e", "w", "nw", "ne", "sw", "se"):
            return
        with self._studio_lock:
            win = self._studio_window
            hwnd = self._studio_hwnd()
        if self._studio_resizing or win is None or not hwnd or not win.native:
            return
        form = win.native
        with suppress_logged("studio_ui", throttled_key="studio_resize_check_max"):
            is_max_box = [False]
            form.Invoke(DotNetAction(lambda: is_max_box.__setitem__(0, form.WindowState == FormWindowState.Maximized)))
            if is_max_box[0]:
                return

        self._studio_resizing = True
        vk = 0x02 if user32.GetSystemMetrics(23) else 0x01
        pt0, rect0 = wintypes.POINT(), wintypes.RECT()
        user32.GetCursorPos(ctypes.byref(pt0))
        user32.GetWindowRect(hwnd, ctypes.byref(rect0))
        dpi = self._window_dpi_scale(hwnd)
        min_w, min_h = int(round(920 * dpi)), int(round(600 * dpi))
        orig_w, orig_h = max(min_w, int(rect0.right - rect0.left)), max(min_h, int(rect0.bottom - rect0.top))

        def _resize_worker() -> None:
            try:
                cur = wintypes.POINT()
                while not self._stop_event.is_set() and user32.IsWindow(hwnd) and (user32.GetAsyncKeyState(vk) & 0x8000):
                    user32.GetCursorPos(ctypes.byref(cur))
                    dx, dy = int(cur.x - pt0.x), int(cur.y - pt0.y)
                    new_left, new_top, new_w, new_h = int(rect0.left), int(rect0.top), orig_w, orig_h
                    if "e" in clean_edge:
                        new_w = max(min_w, orig_w + dx)
                    elif "w" in clean_edge:
                        new_w = max(min_w, orig_w - dx)
                        new_left = int(rect0.right) - new_w
                    if "s" in clean_edge:
                        new_h = max(min_h, orig_h + dy)
                    elif "n" in clean_edge:
                        new_h = max(min_h, orig_h - dy)
                        new_top = int(rect0.bottom) - new_h
                    # SWP_NOZORDER (0x0004) | SWP_NOACTIVATE (0x0010)
                    user32.SetWindowPos(hwnd, 0, new_left, new_top, new_w, new_h, 0x0004 | 0x0010)
                    time.sleep(0.008)
            finally:
                self._studio_resizing = False

        threading.Thread(target=_resize_worker, daemon=True).start()

    def run_preflight(self, deep: bool = True) -> dict[str, Any]:
        """执行环境深度或快速自检并返回已脱敏的结构化体检报告。"""
        deep_flag = bool(deep.get("deep", True)) if isinstance(deep, dict) else bool(deep)
        return _redact_payload(self._backend.run_preflight(deep=deep_flag))

    def quit_app(self) -> None:
        self._stop_event.set()
        self._close_studio_window_internal()
        if self._window:
            with suppress_logged("pet_ui", throttled_key="pet_destroy_on_quit"):
                self._window.destroy()


# 表驱动绑定机械透传方法：保留与原始手写 def 完全一致的函数名及形参（供 pywebview getfullargspec 精确反射）
_BACKEND_FORWARDER_SPECS: tuple[tuple[str, str, str, dict[str, Any]], ...] = (
    ("get_state", "get_cached_snapshot", "0", {}),
    ("refresh_now", "fetch_live_status", "0", {"touch": True}),
    ("open_comfy_tunnel", "open_comfy_tunnel", "0", {"redact": True}),
    ("copy_comfy_access_config", "get_comfy_access_bundle", "0", {"redact": True}),
    ("open_llm_tunnel", "open_llm_tunnel", "0", {"redact": True}),
    ("open_colab_url", "open_colab_url", "0", {}),
    ("sync_outputs", "sync_outputs", "0", {"redact": True}),
    ("copy_llm_api_config", "get_llm_api_bundle", "0", {"redact": True}),
    ("stop_all_instances", "stop_all_instances", "0", {"redact": True}),
    ("start_instance", "start_instance", "cfg", {}),
    ("reconfigure_running_llm", "reconfigure_running_llm", "cfg", {}),
    ("cancel_deploy", "cancel_deploy", "0", {}),
    ("start_colab_oauth", "start_colab_oauth", "0", {"wake": 30.0}),
    ("cancel_colab_oauth", "cancel_colab_oauth", "0", {}),
    ("switch_colab_account", "switch_colab_account", "acct", {}),
    ("remove_colab_account", "remove_colab_account", "acct", {}),
    ("refresh_account_pool", "refresh_account_pool", "0", {}),
    # 单字符串参数 slot ("comfy" | "llm" | "all")，复用单参透传器
    ("stop_cluster_slot", "stop_cluster_slot", "acct", {}),
    ("start_gdrive_bind", "start_gdrive_bind", "0", {"wake": 30.0}),
    ("cancel_gdrive_bind", "cancel_gdrive_bind", "0", {}),
    ("probe_environment_deep", "probe_environment_deep", "0", {}),
    ("save_cf_tunnel_config", "save_cf_tunnel_config", "cfg", {}),
    ("clear_cf_tunnel_config", "clear_cf_tunnel_config", "0", {}),
    ("reset_tool_paths", "reset_tool_paths", "0", {}),
    ("open_local_outputs_dir", "open_local_outputs_dir", "0", {}),
    ("export_diagnostics", "export_diagnostics", "0", {"redact": True}),
    ("open_logs_dir", "open_logs_dir", "0", {"redact": True}),
    ("list_gallery_items", "list_gallery_items", "cfg", {}),
    ("inspect_gallery_item", "inspect_gallery_item", "rel", {}),
    ("push_gallery_workflow_to_canvas", "push_gallery_workflow_to_canvas", "rel", {"redact": True}),
    ("reveal_gallery_item", "reveal_gallery_item", "rel", {}),
    ("list_cloud_assets", "list_cloud_assets", "0", {}),
    ("tail_live_logs", "tail_live_logs", "cfg", {}),
    ("get_agent_state", "get_agent_state", "0", {"redact": True}),
    ("save_agent_config", "save_agent_config", "cfg", {"redact": True}),
    ("fetch_agent_models", "fetch_agent_models", "cfg", {"redact": True}),
    ("probe_agent_provider", "probe_agent_provider", "cfg", {"redact": True}),
    ("send_agent_message", "send_agent_message", "cfg", {"redact": True}),
    ("resolve_agent_confirm", "resolve_agent_confirm", "cfg", {"redact": True}),
    ("cancel_agent_turn", "cancel_agent_turn", "0", {"redact": True}),
    ("clear_agent_history", "clear_agent_history", "0", {"redact": True}),
    ("manage_agent_session", "manage_agent_session", "cfg", {"redact": True}),
)


def _make_forwarder_0(target_name: str, redact: bool, wake: float, touch: bool):
    def _fn(self: Any) -> dict[str, Any]:
        if touch:
            self._backend.touch_activity()
        if wake > 0.0:
            self._peek_wake_until = max(self._peek_wake_until, time.monotonic() + wake)
        res = getattr(self._backend, target_name)()
        return self._apply_clipboard_and_redact(res) if redact else res

    return _fn


def _make_forwarder_cfg(target_name: str, redact: bool = False):
    def _fn(self: Any, config: dict[str, Any] | None = None) -> dict[str, Any]:
        res = getattr(self._backend, target_name)(config or {})
        return self._apply_clipboard_and_redact(res) if redact else res

    return _fn


def _make_forwarder_acct(target_name: str):
    def _fn(self: Any, acct_id: str) -> dict[str, Any]:
        return getattr(self._backend, target_name)(acct_id)

    return _fn


def _make_forwarder_rel(target_name: str, redact: bool):
    def _fn(self: Any, rel_path: str) -> dict[str, Any]:
        res = getattr(self._backend, target_name)(rel_path)
        return self._apply_clipboard_and_redact(res) if redact else res

    return _fn


def _bind_backend_forwarders(cls: type) -> None:
    for api_name, target_name, kind, opts in _BACKEND_FORWARDER_SPECS:
        if kind == "0":
            fn = _make_forwarder_0(
                target_name,
                bool(opts.get("redact", False)),
                float(opts.get("wake", 0.0)),
                bool(opts.get("touch", False)),
            )
        elif kind == "cfg":
            fn = _make_forwarder_cfg(target_name, bool(opts.get("redact", False)))
        elif kind == "rel":
            fn = _make_forwarder_rel(target_name, bool(opts.get("redact", False)))
        else:
            fn = _make_forwarder_acct(target_name)
        fn.__name__ = api_name
        fn.__qualname__ = f"{cls.__name__}.{api_name}"
        setattr(cls, api_name, fn)


_bind_backend_forwarders(PetBridgeApi)


def _push_to_ui(win: webview.Window, snap: dict[str, Any]) -> None:
    """将状态快照安全推送到前端 JS 渲染函数。"""
    payload = json.dumps(snap, ensure_ascii=False)
    with suppress_logged("pet_ui", throttled_key="push_to_ui"):
        win.evaluate_js(f"window.renderState && window.renderState({payload});")


def _push_to_studio(win: webview.Window, snap: dict[str, Any]) -> None:
    """将状态快照安全推送到 Studio 窗口前端渲染函数。"""
    payload = json.dumps(snap, ensure_ascii=False)
    with suppress_logged("studio_ui", throttled_key="push_to_studio"):
        win.evaluate_js(f"window.renderStudioState && window.renderStudioState({payload});")


def _push_event_to_studio(win: webview.Window, evt: dict[str, Any]) -> None:
    """将后端脱敏事件实时推送到 Studio 工作台窗口，并在 state_sync 时同步刷新视图状态。"""
    if not isinstance(evt, dict):
        return
    safe_evt = _redact_payload({k: v for k, v in evt.items() if k != "clipboard_text"})
    payload = json.dumps(safe_evt, ensure_ascii=False)
    js_parts = [f"window.onStudioBackendEvent && window.onStudioBackendEvent({payload});"]
    if safe_evt.get("type") == "state_sync" and isinstance(safe_evt.get("snapshot"), dict):
        snap_payload = json.dumps(safe_evt["snapshot"], ensure_ascii=False)
        js_parts.append(f"window.renderStudioState && window.renderStudioState({snap_payload});")
    with suppress_logged("studio_ui", throttled_key="push_event_to_studio"):
        win.evaluate_js(" ".join(js_parts))


def _push_event_to_ui(win: webview.Window, api: PetBridgeApi, evt: dict[str, Any]) -> None:
    """将后端实时事件（推理进度 / 出图提醒 / 上传进度）脱敏后推送至前端。"""
    clip_text = evt.get("clipboard_text")
    if isinstance(clip_text, str) and clip_text:
        api._set_clipboard(clip_text)
    if str(evt.get("type") or "") in _WAKE_ON_EVENT_TYPES:
        api._wake_from_edge_peek(hold_sec=8.5)
    safe_evt = _redact_payload({k: v for k, v in evt.items() if k != "clipboard_text"})
    payload = json.dumps(safe_evt, ensure_ascii=False)
    with suppress_logged("pet_ui", throttled_key="push_event_to_ui"):
        win.evaluate_js(f"window.onBackendEvent && window.onBackendEvent({payload});")
