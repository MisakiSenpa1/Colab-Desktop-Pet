"""Phase 3 Wave 3A 路 2：`pet_app.py` 四模块拆解（asset_server / win32_window / bridge_api / pet_app）回归测试。

说明：全量基于隔离临时沙箱目录与环境变量（COLAB_PET_MOCK_VAULT_PATH、
COLAB_PET_CF_CONFIG_PATH、COLAB_PET_DIAG_LOG_PATH），绝不读取或修改宿主机真实凭据库或配置文件。
"""

from __future__ import annotations

import ast
import http.client
import inspect
import json
import os
import re
import subprocess
import sys
import threading
from ctypes import wintypes
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

PKG_DIR = Path(__file__).resolve().parents[1]
PET_DIR = PKG_DIR / "colab_pet"
for _p in (str(PKG_DIR), str(PET_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import asset_server  # noqa: E402
import bridge_api  # noqa: E402
import build_windows_package as win_pkg  # noqa: E402
import colab_backend as cb  # noqa: E402
import colab_security as csec  # noqa: E402
import pet_app  # noqa: E402
import win32_window  # noqa: E402


@pytest.fixture()
def isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """构建完全隔离的沙箱凭据库、隧道配置与诊断日志目录。"""
    csec.clear_ephemeral_secrets()
    vault_file = tmp_path / "mock_vault.json"
    cf_file = tmp_path / ".cf_tunnel_secret.json"
    diag_log = tmp_path / "logs" / "colab_diag.log"
    monkeypatch.setenv("COLAB_PET_MOCK_VAULT_PATH", str(vault_file))
    monkeypatch.setenv("COLAB_PET_CF_CONFIG_PATH", str(cf_file))
    monkeypatch.setenv("COLAB_PET_DIAG_LOG_PATH", str(diag_log))
    monkeypatch.delenv("COLAB_PET_ACCOUNT_ID", raising=False)
    yield SimpleNamespace(tmp=tmp_path, vault_file=vault_file, cf_file=cf_file, diag_log=diag_log)
    pet_app._set_active_media_root(None)
    csec.clear_ephemeral_secrets()


def _make_backend(env_ns: SimpleNamespace, name: str = "phase3_pet") -> cb.ColabPetBackend:
    return cb.ColabPetBackend(
        state_path=env_ns.tmp / f"{name}.json",
        download_root=env_ns.tmp / f"{name}_dl",
        cf_config_path=env_ns.cf_file,
    )


def _collect_top_imports(py_path: Path) -> set[str]:
    tree = ast.parse(py_path.read_text(encoding="utf-8"), filename=py_path.name)
    mods: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mods.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module.split(".")[0])
    return mods


# =========================================================================
# 1. 模块依赖边界、纯标准库 asset_server、零 PIL 与同对象重导出校验
# =========================================================================


def test_pet_app_four_module_boundaries_zero_pil_and_pure_stdlib_asset_server() -> None:
    """校验 4 模块零 PIL、`asset_server.py` 纯标准库零 clr/webview，以及 pet_app 重导出同对象契约。"""
    # 1) asset_server.py 仅使用 Python 标准库（严禁 clr / webview / PIL / System）
    asset_imports = _collect_top_imports(PET_DIR / "asset_server.py")
    non_stdlib = sorted(m for m in asset_imports if m not in sys.stdlib_module_names)
    assert not non_stdlib, f"asset_server.py 存在非标准库依赖: {non_stdlib}"
    assert not (asset_imports & {"clr", "webview", "PIL", "System"})

    # 独立子进程导入 asset_server 不加载 clr / webview
    sub = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                f"import sys; sys.path.insert(0, {str(PET_DIR)!r}); "
                "import asset_server; "
                "assert 'clr' not in sys.modules and 'webview' not in sys.modules; "
                "print('ASSET_SERVER_CLEAN_IMPORT_OK')"
            ),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=15,
    )
    assert sub.returncode == 0 and "ASSET_SERVER_CLEAN_IMPORT_OK" in sub.stdout, (sub.stdout, sub.stderr)

    # 2) 全部 4 模块严禁导入 PIL
    for fname in ("asset_server.py", "win32_window.py", "bridge_api.py", "pet_app.py"):
        mods = _collect_top_imports(PET_DIR / fname)
        assert "PIL" not in mods, f"{fname} 禁止导入 PIL"

    # 3) pet_app.py 瘦身行数 <= 320 且显式重导出全部外部/测试依赖符号（同对象身份一致）
    pet_lines = len((PET_DIR / "pet_app.py").read_text(encoding="utf-8").splitlines())
    assert pet_lines <= 320, f"pet_app.py 行数超出 320 行目标: {pet_lines}"

    required_exports = (
        "BASE_W", "BASE_H", "EDGE_BG_HEX", "EDGE_BG_RGB", "PET_DIR", "ASSETS_DIR",
        "DotNetAction", "PetBridgeApi", "_ALLOWED_ASSET_EXTS", "_PetAssetHandler",
        "_ThreadingHTTPServer", "_capture_hwnd_png", "_push_event_to_ui", "_push_to_ui",
        "_push_to_studio", "_push_event_to_studio", "_set_active_media_root",
        "user32", "dwmapi", "kernel32_api",
    )
    for sym in required_exports:
        assert hasattr(pet_app, sym), f"pet_app 缺少重导出符号: {sym}"

    assert pet_app.user32 is win32_window.user32 is bridge_api.user32
    assert pet_app.dwmapi is win32_window.dwmapi is bridge_api.dwmapi
    assert pet_app.kernel32_api is win32_window.kernel32_api
    assert pet_app.PetBridgeApi is bridge_api.PetBridgeApi
    assert pet_app._PetAssetHandler is asset_server._PetAssetHandler
    assert pet_app._ThreadingHTTPServer is asset_server._ThreadingHTTPServer
    assert pet_app._ALLOWED_ASSET_EXTS is asset_server._ALLOWED_ASSET_EXTS


# =========================================================================
# 2. PetBridgeApi 全部 64 个公开方法签名零漂移校验
# =========================================================================


def test_pet_bridge_api_all_64_public_methods_and_zero_underscore_arg_pollution(
    isolated_env: SimpleNamespace,
) -> None:
    """校验 PetBridgeApi 全部 64 个公开方法名与参数列表 100% 保持原契约且无下划线参数污染。"""
    expected_public_args: dict[str, list[str]] = {
        "begin_drag": ["self"],
        "begin_studio_drag": ["self"],
        "begin_studio_resize": ["self", "edge"],
        "cancel_agent_turn": ["self"],
        "cancel_colab_oauth": ["self"],
        "cancel_deploy": ["self"],
        "cancel_gdrive_bind": ["self"],
        "clear_agent_history": ["self"],
        "clear_cf_tunnel_config": ["self"],
        "complete_colab_oauth": ["self", "code", "label"],
        "confirm_dual_image_drop": ["self", "paths", "target"],
        "confirm_png_drop": ["self", "path", "mode"],
        "copy_comfy_access_config": ["self"],
        "copy_llm_api_config": ["self"],
        "copy_text": ["self", "text"],
        "export_diagnostics": ["self"],
        "fetch_agent_models": ["self", "config"],
        "get_agent_state": ["self"],
        "get_state": ["self"],
        "handle_toast_action": ["self", "action"],
        "inspect_gallery_item": ["self", "rel_path"],
        "list_cloud_assets": ["self"],
        "list_gallery_items": ["self", "config"],
        "manage_agent_session": ["self", "config"],
        "open_colab_url": ["self"],
        "open_comfy_tunnel": ["self"],
        "open_llm_tunnel": ["self"],
        "open_local_outputs_dir": ["self"],
        "open_logs_dir": ["self"],
        "open_studio_window": ["self", "tab"],
        "pick_and_upload_files": ["self"],
        "pick_rclone_conf": ["self"],
        "pick_tool_binary": ["self", "kind"],
        "probe_agent_provider": ["self", "config"],
        "probe_environment_deep": ["self"],
        "push_gallery_workflow_to_canvas": ["self", "rel_path"],
        "quit_app": ["self"],
        "reconfigure_running_llm": ["self", "config"],
        "refresh_account_pool": ["self"],
        "refresh_now": ["self"],
        "remove_colab_account": ["self", "acct_id"],
        "rename_colab_account": ["self", "acct_id", "label"],
        "reset_tool_paths": ["self"],
        "resolve_agent_confirm": ["self", "config"],
        "reveal_gallery_item": ["self", "rel_path"],
        "run_preflight": ["self", "deep"],
        "save_agent_config": ["self", "config"],
        "save_cf_tunnel_config": ["self", "config"],
        "save_prefs": ["self", "updates"],
        "send_agent_message": ["self", "config"],
        "set_edge_auto_hide": ["self", "enabled"],
        "set_scale": ["self", "scale_val"],
        "set_topmost": ["self", "flag"],
        "start_colab_oauth": ["self"],
        "start_gdrive_bind": ["self"],
        "start_instance": ["self", "config"],
        "stop_all_instances": ["self"],
        "stop_cluster_slot": ["self", "acct_id"],
        "stop_session_instance": ["self", "session"],
        "studio_window_ctl": ["self", "action"],
        "switch_colab_account": ["self", "acct_id"],
        "sync_outputs": ["self"],
        "sync_region": ["self", "layout"],
        "tail_live_logs": ["self", "config"],
    }
    assert len(expected_public_args) == 64

    actual_public: dict[str, list[str]] = {}
    for name, fn in sorted(inspect.getmembers(pet_app.PetBridgeApi, predicate=inspect.isfunction)):
        if name.startswith("_"):
            continue
        spec = inspect.getfullargspec(fn)
        assert spec.varargs is None and spec.varkw is None, f"{name} 禁止含可变参数: {spec}"
        assert spec.args and spec.args[0] == "self", f"{name} 首参必须为 self"
        assert not any(a.startswith("_") for a in spec.args[1:]), f"{name} 参数受下划线内部参数污染: {spec.args}"
        actual_public[name] = list(spec.args)

    assert actual_public == expected_public_args

    # 校验 ui.html 与 studio.js 中所有 window.pywebview.api.* 调用均被完整覆盖
    backend = _make_backend(isolated_env, "api_sig_check")
    bridge = pet_app.PetBridgeApi(backend, threading.Event())
    for frontend_file in ("ui.html", "studio.js"):
        txt = (PET_DIR / frontend_file).read_text(encoding="utf-8")
        for m_name in sorted(set(re.findall(r"window\.pywebview\.api\.([a-zA-Z0-9_]+)", txt))):
            attr = getattr(bridge, m_name, None)
            assert callable(attr) and inspect.ismethod(attr), f"{frontend_file} 调用的方法缺失: {m_name}"


# =========================================================================
# 3. asset_server HTTP 静态资产、白名单与 Range 206 / 416 / 403 校验
# =========================================================================


def test_asset_server_whitelist_range_206_416_and_root_override(isolated_env: SimpleNamespace) -> None:
    """验证 asset_server 的 Range 解析、/assets/ 白名单与 /__media/ 206/416/403 边界。"""
    assert asset_server._parse_byte_range("", 100) == (0, 99, False)
    assert asset_server._parse_byte_range("bytes=10-19", 100) == (10, 19, True)
    assert asset_server._parse_byte_range("bytes=90-", 100) == (90, 99, True)
    assert asset_server._parse_byte_range("bytes=-15", 100) == (85, 99, True)
    assert asset_server._parse_byte_range("bytes=200-300", 100) is None
    assert asset_server._parse_byte_range("bytes=-", 100) is None
    assert asset_server._parse_byte_range("bytes=-0", 100) is None

    media_root = isolated_env.tmp / "media_box"
    media_root.mkdir(parents=True, exist_ok=True)
    sample_mp4 = media_root / "clip.mp4"
    sample_mp4.write_bytes(b"0123456789ABCDEF")
    (isolated_env.tmp / "outside_secret.txt").write_text("NEVER_LEAK", encoding="utf-8")

    fake_assets = isolated_env.tmp / "fake_assets"
    fake_assets.mkdir(parents=True, exist_ok=True)
    (fake_assets / "demo.webp").write_bytes(b"RIFF_DEMO_WEBP")
    (fake_assets / "forbidden.py").write_text("SECRET=1\n", encoding="utf-8")

    httpd, port = asset_server.start_asset_http_server(media_root)
    httpd.assets_root = fake_assets  # type: ignore[attr-defined]

    def _req(method: str, path: str, headers: dict[str, str] | None = None) -> tuple[int, dict[str, str], bytes]:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        try:
            conn.request(method, path, headers=headers or {})
            resp = conn.getresponse()
            return resp.status, {k.lower(): v for k, v in resp.getheaders()}, resp.read()
        finally:
            conn.close()

    try:
        st_ui, h_ui, body_ui = _req("GET", "/ui.html")
        assert st_ui == 200 and "no-store" in h_ui.get("cache-control", "") and len(body_ui) > 200

        # /assets/ .png -> .webp 自动回退 + 非白名单 .py 拦截 403
        st_ok, _, body_ok = _req("GET", "/assets/demo.png")
        assert st_ok == 200 and body_ok == b"RIFF_DEMO_WEBP"
        st_forb, _, _ = _req("GET", "/assets/forbidden.py")
        assert st_forb == 403

        # /__media/ Range 206、HEAD、越界 416、穿越 403
        st_206, h_206, body_206 = _req("GET", "/__media/clip.mp4", {"Range": "bytes=4-7"})
        assert st_206 == 206 and body_206 == b"4567" and h_206.get("content-range") == "bytes 4-7/16"

        st_head, h_head, body_head = _req("HEAD", "/__media/clip.mp4")
        assert st_head == 200 and body_head == b"" and h_head.get("content-length") == "16"

        st_416, h_416, _ = _req("GET", "/__media/clip.mp4", {"Range": "bytes=99-120"})
        assert st_416 == 416 and h_416.get("content-range") == "bytes */16"

        st_trav, _, _ = _req("GET", "/__media/../outside_secret.txt")
        assert st_trav == 403
    finally:
        httpd.shutdown()
        httpd.server_close()


# =========================================================================
# 4. win32_window 几何、源码 mtime、user32 跨模块 patch 与 bridge 离线辅助校验
# =========================================================================


def test_win32_window_geometry_source_mtime_and_bridge_offline_helpers(isolated_env: SimpleNamespace) -> None:
    """校验几何/停靠计算、_latest_pet_source_mtime 覆盖新模块、user32 同实例 patch 透传及拖拽/窗口控制逻辑。"""
    assert win32_window._round_box_px({"x": 10.4, "y": 20.6, "w": 30.2, "h": 40.4}, 1.0) == (10, 21, 31, 40)
    runs = win32_window._rounded_rect_runs(0.0, 0.0, 20.0, 10.0, 4.0)
    assert runs and sum(rh for _, _, _, rh in runs) == 10
    assert win32_window._rounded_rect_runs(5.0, 5.0, 5.0, 10.0, 2.0) == []

    x_vis_r, x_peek_r = win32_window._compute_edge_x_pair("right", 0, 1920, 392, 1.0, 1.0)
    assert x_vis_r == 1920 - 392 - 4 and x_peek_r == 1920 - 140
    x_vis_l, x_peek_l = win32_window._compute_edge_x_pair("left", 0, 1920, 392, 1.0, 1.0)
    assert x_vis_l == 4 and x_peek_l == -251

    # _latest_pet_source_mtime 应涵盖 PET_DIR/*.py 新拆分子模块
    latest_m = win32_window._latest_pet_source_mtime()
    for sub_py in ("win32_window.py", "asset_server.py", "bridge_api.py", "pet_app.py"):
        assert latest_m >= (PET_DIR / sub_py).stat().st_mtime

    # 校验 patch.object(pet_app.user32, ...) 对 bridge_api 内部调用 100% 生效
    backend = _make_backend(isolated_env, "bridge_helpers")
    api = pet_app.PetBridgeApi(backend, threading.Event())
    fake_pet_win = SimpleNamespace(native=SimpleNamespace(Handle=SimpleNamespace(ToInt64=lambda: 301)))
    fake_stu_win = SimpleNamespace(native=SimpleNamespace(Handle=SimpleNamespace(ToInt64=lambda: 402)))
    api._bind_window(fake_pet_win)
    api._studio_window = fake_stu_win

    with (
        patch.object(pet_app.user32, "IsWindow", side_effect=lambda h: int(h) == 402),
        patch.object(pet_app.user32, "IsIconic", return_value=False),
        patch.object(pet_app.user32, "GetForegroundWindow", return_value=402),
    ):
        assert api._studio_hwnd() == 402
        assert api._resolve_dialog_window() is fake_stu_win

    # 校验 _compute_drag_dock_snap（左侧吸附 / 右侧吸附 / 中间自由落位）
    wa = SimpleNamespace(Left=0, Top=0, Right=1920, Bottom=1040)
    r_left = wintypes.RECT(10, 200, 402, 748)
    fx_l, _, edge_l, flip_l, _ = api._compute_drag_dock_snap(r_left, wa, 1.0, {"scale": 1.0, "bubble_open": True})
    assert edge_l == "left" and flip_l is True and fx_l == 4

    r_right = wintypes.RECT(1520, 200, 1912, 748)
    fx_r, _, edge_r, flip_r, _ = api._compute_drag_dock_snap(r_right, wa, 1.0, {"scale": 1.0, "bubble_open": True})
    assert edge_r == "right" and flip_r is False and fx_r == 1920 - 392 - 4

    r_mid = wintypes.RECT(760, 200, 1152, 748)
    _, _, edge_m, _, _ = api._compute_drag_dock_snap(r_mid, wa, 1.0, {"scale": 1.0, "bubble_open": True})
    assert edge_m is None

    # 校验 studio_window_ctl 无效句柄回退与未知动作安全返回
    api._studio_window = None
    assert api.studio_window_ctl("get_window_state") == {
        "ok": True, "open": False, "maximized": False, "minimized": False
    }
    assert api.studio_window_ctl("close")["ok"] is False


# =========================================================================
# 5. --internal-ctl 无 GUI 极速转发与打包清单同步核验
# =========================================================================


def test_pet_app_internal_ctl_fast_forward_and_packaging_manifests(isolated_env: SimpleNamespace) -> None:
    """校验 pet_app.py --internal-ctl doctor --json 无 GUI 转发以及便携包/PyInstaller 清单覆盖全量新模块。"""
    env = dict(os.environ)
    env["COLAB_PET_EDITION"] = "public"
    proc = subprocess.run(
        [sys.executable, str(PET_DIR / "pet_app.py"), "--internal-ctl", "doctor", "--json"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=25,
        env=env,
    )
    assert proc.stdout.strip(), f"--internal-ctl doctor --json 无标准输出: stderr={proc.stderr}"
    payload = json.loads(proc.stdout)
    assert "items" in payload and "ok" in payload

    # 校验 PORTABLE_PET_FILES 涵盖 Wave 3A 全部 7 个新增模块
    for req_pet_file in (
        "__init__.py",
        "state_store.py",
        "secret_store.py",
        "context.py",
        "win32_window.py",
        "asset_server.py",
        "bridge_api.py",
    ):
        assert req_pet_file in win_pkg.PORTABLE_PET_FILES, f"PORTABLE_PET_FILES 缺少: {req_pet_file}"

    # 校验 colab_pet.spec 已补充 sys.path.insert 与全部 6 个新模块 hiddenimports
    spec_txt = (PKG_DIR / "colab_pet.spec").read_text(encoding="utf-8")
    assert "sys.path.insert(0, str(SPEC_DIR))" in spec_txt
    for req_mod in ("state_store", "secret_store", "context", "win32_window", "asset_server", "bridge_api"):
        assert f'"{req_mod}"' in spec_txt, f"colab_pet.spec hiddenimports 缺少: {req_mod}"
