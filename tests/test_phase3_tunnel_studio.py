"""Phase 3 Wave 3B 专项回归测试：TunnelService 与 StudioService 去 `_cb_mod()` 及热函数拆解验收。

测试严格在临时沙箱目录与 `COLAB_PET_MOCK_VAULT_PATH` 隔离环境下运行，绝不触碰宿主机真实状态或凭据库。
"""

from __future__ import annotations

import ast
import json
import os
import struct
import sys
import zlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

PKG_DIR = Path(__file__).resolve().parents[1]
PET_DIR = PKG_DIR / "colab_pet"
for _p in (str(PKG_DIR), str(PET_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import backend_studio as bs_mod  # noqa: E402
import backend_tunnel as bt_mod  # noqa: E402
import colab_backend as cb  # noqa: E402
from colab_core.runtime import CommandResult  # noqa: E402
from context import BackendContext  # noqa: E402


def _calc_ast_cc(node: ast.AST) -> int:
    """计算单个函数 AST 节点的圈复杂度（CC）。"""
    cc = 1
    for child in ast.walk(node):
        if isinstance(
            child,
            (
                ast.If,
                ast.While,
                ast.For,
                ast.AsyncFor,
                ast.ExceptHandler,
                ast.With,
                ast.AsyncWith,
                ast.IfExp,
                ast.Assert,
            ),
        ):
            cc += 1
        elif isinstance(child, ast.BoolOp):
            cc += len(child.values) - 1
        elif isinstance(child, ast.comprehension):
            cc += 1 + len(child.ifs)
    return cc


def _make_isolated_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    ctx_overrides: dict[str, Any] | None = None,
) -> cb.ColabPetBackend:
    """构造完全隔离的 `ColabPetBackend` 实例，可选注入 `BackendContext` 覆写项。"""
    monkeypatch.setenv("COLAB_PET_MOCK_VAULT_PATH", str(tmp_path / "mock_vault.json"))
    state_file = tmp_path / "pet_state.json"
    dl_root = tmp_path / "downloads"
    dl_root.mkdir(parents=True, exist_ok=True)
    cf_file = tmp_path / ".cf_tunnel_secret.json"
    backend = cb.ColabPetBackend(
        state_path=state_file,
        download_root=dl_root,
        cf_config_path=cf_file,
    )
    if ctx_overrides:
        new_ctx = BackendContext.from_module_compat(cb, backend=backend, **ctx_overrides)
        backend._ctx = new_ctx
        backend._tunnel_svc._ctx = new_ctx
        backend._studio_svc._ctx = new_ctx
    return backend


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    """构造单个带 CRC32 的 PNG 数据块。"""
    crc = zlib.crc32(chunk_type + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", crc)


def _write_minimal_png(path: Path, width: int, height: int, texts: dict[str, str] | None = None) -> None:
    """使用纯标准库写入带可选 tEXt 块的合法最小 PNG 文件。"""
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    raw_scanline = b"\x00" + b"\x00\x00\x00" * width
    idat = zlib.compress(raw_scanline * height)
    chunks = [b"\x89PNG\r\n\x1a\n", _png_chunk(b"IHDR", ihdr)]
    for k, v in (texts or {}).items():
        chunks.append(_png_chunk(b"tEXt", k.encode("latin-1") + b"\x00" + v.encode("utf-8")))
    chunks.append(_png_chunk(b"IDAT", idat))
    chunks.append(_png_chunk(b"IEND", b""))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(chunks))


def _write_minimal_webp_vp8x(path: Path, width: int, height: int) -> None:
    """构造 VP8X 扩展头格式的 WebP 文件头。"""
    payload = b"\x00\x00\x00\x00" + (width - 1).to_bytes(3, "little") + (height - 1).to_bytes(3, "little")
    chunk = b"VP8X" + struct.pack("<I", len(payload)) + payload
    header = b"RIFF" + struct.pack("<I", 4 + len(chunk)) + b"WEBP" + chunk
    path.write_bytes(header)


def _write_minimal_webp_vp8(path: Path, width: int, height: int) -> None:
    """构造有损 VP8 关键帧格式的 WebP 文件头。"""
    frame_hdr = b"\x00\x00\x00" + b"\x9d\x01\x2a" + struct.pack("<HH", width, height)
    chunk = b"VP8 " + struct.pack("<I", len(frame_hdr)) + frame_hdr
    header = b"RIFF" + struct.pack("<I", 4 + len(chunk)) + b"WEBP" + chunk
    path.write_bytes(header)


def _write_minimal_webp_vp8l(path: Path, width: int, height: int) -> None:
    """构造无损 VP8L 格式的 WebP 文件头。"""
    bits = ((width - 1) & 0x3FFF) | (((height - 1) & 0x3FFF) << 14)
    payload = b"\x2f" + struct.pack("<I", bits)
    chunk = b"VP8L" + struct.pack("<I", len(payload)) + payload
    header = b"RIFF" + struct.pack("<I", 4 + len(chunk)) + b"WEBP" + chunk
    path.write_bytes(header)


def _write_minimal_jpeg(path: Path, width: int, height: int) -> None:
    """构造带 APP0 与 SOF0 段的最小 JPEG 文件头。"""
    app0_body = b"JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    app0 = b"\xff\xe0" + struct.pack(">H", len(app0_body) + 2) + app0_body
    sof0_body = struct.pack(">BHHB", 8, height, width, 3) + b"\x01\x11\x00\x02\x11\x01\x03\x11\x01"
    sof0 = b"\xff\xc0" + struct.pack(">H", len(sof0_body) + 2) + sof0_body
    path.write_bytes(b"\xff\xd8" + app0 + b"\xff\xff" + sof0 + b"\xff\xd9")


# ======================================================================
# 1. 静态架构门禁：容器绑定、零 `_cb_mod()`、零重依赖与低圈复杂度
# ======================================================================


def test_phase3_tunnel_and_studio_architecture_guardrails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """验证两模块均已彻底去 `_cb_mod()`、零禁引依赖、容器完整绑定且全局最高 CC <= 25。"""
    forbidden = {"PIL", "Pillow", "aiohttp", "pywin32", "win32api", "win32con", "win32gui"}
    for fname in ("backend_tunnel.py", "backend_studio.py"):
        fpath = PET_DIR / fname
        src = fpath.read_text(encoding="utf-8")
        assert "_cb_mod(" not in src, f"{fname} 仍残留 _cb_mod() 调用"
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".")[0] not in forbidden, f"{fname} 禁止导入 {alias.name}"
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".")[0] not in forbidden, f"{fname} 禁止导入 {node.module}"
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                cc = _calc_ast_cc(node)
                assert cc <= 25, f"{fname}:{node.name} 圈复杂度过高: CC={cc}"

    assert bt_mod.TunnelService is not None
    backend = _make_isolated_backend(tmp_path, monkeypatch)
    for svc in (backend._tunnel_svc, backend._studio_svc):
        assert svc._b is backend
        assert svc._store is backend._store
        assert svc._secrets is backend._secrets
        assert svc._bus is backend._bus
        assert svc._ctx is backend._ctx


# ======================================================================
# 2. TunnelService 纯 DI 与核心场景回归测试
# ======================================================================


def test_tunnel_fast_read_metadata_secrets_and_no_session_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """验证 fast_read_tunnel_url 剥离明文写入 SecretStore，以及 no_session 原子清理行为。"""
    fake_py = tmp_path / "colab_py.exe"
    fake_py.write_bytes(b"MZ")
    responses: list[CommandResult] = []
    captured_cmds: list[list[str]] = []

    def _fake_run(cmd: list[str], **_kw: Any) -> CommandResult:
        captured_cmds.append(list(cmd))
        return responses.pop(0)

    backend = _make_isolated_backend(
        tmp_path,
        monkeypatch,
        ctx_overrides={
            "get_colab_py": lambda: fake_py,
            "run_hidden": _fake_run,
            "load_cf_secret_raw": lambda **_kw: {},
        },
    )
    backend._store.state["prefs"]["active_account_id"] = "acct_alpha"

    payload_ok = {
        "status": "ok",
        "endpoint": "ep-dual-1",
        "url": "https://cfy-dual.trycloudflare.com",
        "api_key": "sk-llm-secret-0001",
        "comfy_api_key": "cpk-comfy-secret-0002",
        "meta": {
            "kind": "dual",
            "model": "Qwen3.8-27B-abliterated",
            "llm_tunnel_url": "https://llm-dual.trycloudflare.com",
            "api_key": "sk-llm-secret-0001",
            "comfy_api_key": "cpk-comfy-secret-0002",
        },
    }
    responses.append(
        CommandResult(["bridge"], 0, stdout=f"<<<COLAB_TUNNEL>>>{json.dumps(payload_ok)}<<<END_COLAB_TUNNEL>>>", stderr="")
    )
    tstatus, rep, fast_url = backend._tunnel_svc.fast_read_tunnel_url("ep-dual-1", "comfy")
    assert (tstatus, rep, fast_url) == ("ok", "ep-dual-1", "https://cfy-dual.trycloudflare.com")
    assert captured_cmds[0][2] == "fast-tunnel"

    meta, rt_key = backend._secrets.get_ep_meta_and_key("ep-dual-1")
    assert meta["kind"] == "dual"
    assert meta["comfy_tunnel_url"] == "https://cfy-dual.trycloudflare.com"
    assert "api_key" not in meta and "comfy_api_key" not in meta
    assert rt_key == "sk-llm-secret-0001"
    assert "ep-dual-1" in backend._secrets.comfy_key_checked_eps
    assert backend._get_comfy_key_for_url("https://cfy-dual.trycloudflare.com", "ep-dual-1") == "cpk-comfy-secret-0002"

    # no_session 应清空单号会话缓存与非集群临时凭据
    backend._store.state["last_sessions"] = [{"name": "comfy", "endpoint": "ep-dual-1"}]
    backend._store.state["last_assignments_count"] = 1
    backend._store.set_tunnel_cache("ep-dual-1", "https://cfy-dual.trycloudflare.com")
    responses.append(
        CommandResult(["bridge"], 0, stdout='<<<COLAB_TUNNEL>>>{"status":"no_session"}<<<END_COLAB_TUNNEL>>>', stderr="")
    )
    st2, _, u2 = backend._tunnel_svc.fast_read_tunnel_url("ep-dual-1", "comfy")
    assert (st2, u2) == ("no_session", None)
    assert backend._store.get_last_sessions() == []
    assert backend._store.get_tunnel_cache() == {}
    assert backend._secrets.get_ep_meta_and_key("ep-dual-1") == ({}, "")


def test_tunnel_open_comfy_three_tiers_and_restart_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """验证 open_comfy_tunnel 第 1 级缓存秒开、第 2 级速读直开与第 3 级断线重启兜底。"""
    opened_urls: list[str] = []
    alive_set: set[str] = set()
    ctl_calls: list[list[str]] = []

    def _fake_probe(url: str, **_kw: Any) -> bool:
        return url.rstrip("/") in alive_set

    def _fake_detect(url: str, **_kw: Any) -> str | None:
        return "comfy" if url.rstrip("/") in alive_set else None

    def _fake_run_hidden(cmd: list[str], **_kw: Any) -> CommandResult:
        ctl_calls.append(list(cmd))
        return CommandResult(
            cmd,
            0,
            stdout="COMFYUI_TUNNEL_URL: https://restarted-cfy.trycloudflare.com\n",
            stderr="",
        )

    backend = _make_isolated_backend(
        tmp_path,
        monkeypatch,
        ctx_overrides={
            "open_browser": lambda u: opened_urls.append(u) or True,
            "probe_tunnel_alive": _fake_probe,
            "detect_tunnel_kind": _fake_detect,
            "run_hidden": _fake_run_hidden,
            "ctl_available": lambda: True,
            "ctl_cmd_prefix": lambda unbuffered=False: ["python", "colab_ctl.py"],
            "load_cf_secret_raw": lambda **_kw: {},
            "now": lambda: 1700000000.0,
        },
    )
    backend._store.state["last_sessions"] = [{"name": "comfy", "endpoint": "ep-1"}]
    backend._store.state["last_assignments_count"] = 1
    backend._store.set_tunnel_cache("ep-1", "https://cached-cfy.trycloudflare.com")
    backend._secrets.set_ep_meta("ep-1", {"kind": "comfy", "port": 8188})
    backend._secrets.comfy_key_checked_eps.add("ep-1")

    # Tier 1: 缓存命中秒开
    alive_set.add("https://cached-cfy.trycloudflare.com")
    r1 = backend.open_comfy_tunnel()
    assert r1["ok"] is True
    assert "缓存命中" in r1["message"]
    assert opened_urls[-1] == "https://cached-cfy.trycloudflare.com"

    # Tier 2: 缓存失效 -> 速读命中并更新缓存
    alive_set.clear()
    alive_set.add("https://fast-cfy.trycloudflare.com")
    with patch.object(
        backend, "_fast_read_tunnel_url", return_value=("ok", "ep-1", "https://fast-cfy.trycloudflare.com")
    ):
        r2 = backend.open_comfy_tunnel()
    assert r2["ok"] is True
    assert backend._store.get_tunnel_cache()["ep-1"] == "https://fast-cfy.trycloudflare.com"
    assert opened_urls[-1] == "https://fast-cfy.trycloudflare.com"

    # Tier 3: 缓存与速读均失效 -> 自动重启隧道并更新缓存
    alive_set.clear()
    alive_set.add("https://restarted-cfy.trycloudflare.com")
    with patch.object(backend, "_fast_read_tunnel_url", return_value=("missing", "ep-1", None)):
        r3 = backend.open_comfy_tunnel()
    assert r3["ok"] is True
    assert "已重启并打开" in r3["message"]
    assert any("tunnel" in c and "--restart" in c for c in ctl_calls)
    assert backend._store.get_tunnel_cache()["ep-1"] == "https://restarted-cfy.trycloudflare.com"


def test_tunnel_ensure_direct_canvas_support_and_live_push(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """验证直铺画布扩展热检测/热注入以及原生接口实时推送工作流与回退打开浏览器。"""
    fake_py = tmp_path / "colab_py.exe"
    fake_py.write_bytes(b"MZ")
    opened_urls: list[str] = []
    installed_payloads: list[dict[str, Any]] = []
    state_flags = {"installed": False, "acked": True}

    def _fake_subproc_run(cmd: list[str], **kwargs: Any) -> CommandResult:
        installed_payloads.append(json.loads(kwargs.get("input") or "{}"))
        state_flags["installed"] = True
        return CommandResult(cmd, 0, stdout="OK", stderr="")

    fake_subproc = SimpleNamespace(run=_fake_subproc_run)
    mono_val = {"t": 100.0}

    def _fake_mono() -> float:
        mono_val["t"] += 0.2
        return mono_val["t"]

    backend = _make_isolated_backend(
        tmp_path,
        monkeypatch,
        ctx_overrides={
            "get_colab_py": lambda: fake_py,
            "subprocess_mod": fake_subproc,
            "open_browser": lambda u: opened_urls.append(u) or True,
            "sleep": lambda _s: None,
            "now": lambda: 1700000000.0,
            "monotonic": _fake_mono,
        },
    )
    backend._store.state["last_sessions"] = [{"name": "comfy", "endpoint": "ep-ext"}]
    backend._store.state["last_assignments_count"] = 1

    def _fake_http_get(url: str, **_kw: Any) -> Any:
        if url.endswith("/colab_pet/ping"):
            return SimpleNamespace(status_code=200, json=lambda: {"ok": True, "version": 1}, text="")
        if url.endswith("/direct_canvas.js"):
            body = "// colab_pet_direct_canvas v2" if state_flags["installed"] else "// old v1"
            return SimpleNamespace(status_code=200, text=body, json=lambda: {})
        if url.endswith("/colab_pet/ack_workflow"):
            return SimpleNamespace(status_code=200, json=lambda: {"acked": state_flags["acked"]}, text="")
        return SimpleNamespace(status_code=404, text="", json=lambda: {})

    def _fake_http_post(url: str, **_kw: Any) -> Any:
        if url.endswith("/colab_pet/open_workflow"):
            return SimpleNamespace(status_code=200, json=lambda: {"browser_clients": 1})
        return SimpleNamespace(status_code=200, json=lambda: {})

    with patch.object(backend, "_comfy_http_get", side_effect=_fake_http_get), patch.object(
        backend, "_comfy_http_post", side_effect=_fake_http_post
    ):
        # 首次检测到 v1 -> 热注入 v2 并缓存 native
        mode1, just1 = backend._tunnel_svc.ensure_direct_canvas_support(
            "https://ext-cfy.trycloudflare.com", ep="ep-ext"
        )
        assert (mode1, just1) == ("native", True)
        assert len(installed_payloads) == 1
        assert "js_b64" in installed_payloads[0] and "py_b64" in installed_payloads[0]

        # 再次调用直接命中缓存，不重复安装
        mode2, just2 = backend._tunnel_svc.ensure_direct_canvas_support(
            "https://ext-cfy.trycloudflare.com", ep="ep-ext"
        )
        assert (mode2, just2) == ("native", False)

        # 实时推送命中浏览器回执，不拉起新标签页
        res_live = backend._tunnel_svc.push_or_open_workflow_on_canvas(
            "https://ext-cfy.trycloudflare.com", "demo.json"
        )
        assert res_live["pushed_live"] is True
        assert res_live["opened_browser"] is False
        assert opened_urls == []

        # 无回执时自动回退拉起浏览器并携带一次性票据
        state_flags["acked"] = False
        backend._secrets.set_comfy_secret(
            "ep-ext",
            "cpk-handoff-999",
            account_id="",
            origin="https://ext-cfy.trycloudflare.com",
        )
        res_open = backend._tunnel_svc.push_or_open_workflow_on_canvas(
            "https://ext-cfy.trycloudflare.com", "demo.json"
        )
        assert res_open["pushed_live"] is False
        assert res_open["opened_browser"] is True
        assert len(opened_urls) == 1
        assert "/__pet_auth?ticket=" in opened_urls[0]


def test_tunnel_get_llm_api_bundle_dual_and_pure_comfy_rejection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """验证 get_llm_api_bundle 在双栖/纯 LLM 模式组装接口配置，纯 ComfyUI 时明确拒绝。"""
    backend = _make_isolated_backend(
        tmp_path,
        monkeypatch,
        ctx_overrides={
            "probe_tunnel_alive": lambda u, **_kw: u.startswith("https://llm-live"),
            "load_cf_secret_raw": lambda **_kw: {},
        },
    )
    backend._store.state["last_sessions"] = [{"name": "comfy", "endpoint": "ep-llm"}]
    backend._store.state["last_assignments_count"] = 1
    backend._store.state["prefs"]["boot_profiles"] = ["anima", "qwen_27b"]
    backend._secrets.set_ep_meta(
        "ep-llm",
        {"kind": "dual", "llm_tunnel_url": "https://llm-live.trycloudflare.com"},
        runtime_secret="sk-test-llm-key-888",
    )

    bundle = backend.get_llm_api_bundle()
    assert bundle["ok"] is True
    assert bundle["base_url"] == "https://llm-live.trycloudflare.com/v1"
    assert bundle["model"] == "Qwen3.8-27B-abliterated"
    assert bundle["has_key"] is True
    assert "sk-test-llm-key-888" in bundle["clipboard_text"]

    # 纯 ComfyUI 实例调用应被拒绝
    backend._store.state["prefs"]["boot_profiles"] = ["anima"]
    backend._secrets.set_ep_meta("ep-llm", {"kind": "comfy", "port": 8188}, runtime_secret="")
    with patch.object(backend, "_fast_read_tunnel_url", return_value=("ok", "ep-llm", "https://cfy.trycloudflare.com")):
        rej = backend.get_llm_api_bundle()
    assert rej["ok"] is False
    assert "非 LLM 对话服务" in rej["message"]


def test_tunnel_resolve_active_url_cooldown_and_open_colab_url_fallbacks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """验证 resolve_active_tunnel_url 45s 冷却门控与 open_colab_url 会话/主页回退。"""
    opened: list[str] = []
    clock = {"m": 100.0}
    fast_calls = {"n": 0}

    backend = _make_isolated_backend(
        tmp_path,
        monkeypatch,
        ctx_overrides={
            "monotonic": lambda: clock["m"],
            "open_browser": lambda u: opened.append(u) or True,
            "probe_tunnel_alive": lambda u, **_kw: u == "https://fresh.trycloudflare.com",
            "load_cf_secret_raw": lambda **_kw: {},
            "get_colab_exe": lambda: None,
        },
    )
    backend._store.state["last_sessions"] = [{"name": "comfy", "endpoint": "ep-cool"}]
    backend._store.state["last_assignments_count"] = 1

    def _fake_fast(ep: str, sname: str) -> tuple[str, str, str | None]:
        fast_calls["n"] += 1
        return ("missing", ep, None) if fast_calls["n"] == 1 else ("ok", ep, "https://fresh.trycloudflare.com")

    with patch.object(backend, "_fast_read_tunnel_url", side_effect=_fake_fast):
        assert backend.resolve_active_tunnel_url() is None
        assert fast_calls["n"] == 1
        # 冷却期内（<45s）不再重复 fast_read
        clock["m"] = 120.0
        assert backend.resolve_active_tunnel_url() is None
        assert fast_calls["n"] == 1
        # 超过 45s 后恢复速读并命中新地址
        clock["m"] = 150.0
        assert backend.resolve_active_tunnel_url() == "https://fresh.trycloudflare.com"
        assert fast_calls["n"] == 2

    # open_colab_url：存在会话打开运行时 URL，清空会话后打开 Colab 首页
    r_sess = backend.open_colab_url()
    assert r_sess["ok"] is True and "ep-cool" in r_sess["url"]
    backend._store.state["last_sessions"] = []
    r_home = backend.open_colab_url()
    assert r_home["ok"] is True and r_home["url"] == "https://colab.research.google.com/"
    assert len(opened) == 2


# ======================================================================
# 3. StudioService 纯标准库解析、画廊筛选排序与多端日志测试
# ======================================================================


def test_studio_read_image_dimensions_png_webp_jpeg_pure_stdlib(tmp_path: Path) -> None:
    """验证纯标准库文件头解析精准支持 PNG / WebP(VP8X, VP8, VP8L) / JPEG 及坏文件兜底。"""
    p_png = tmp_path / "a.png"
    p_vp8x = tmp_path / "b_vp8x.webp"
    p_vp8 = tmp_path / "c_vp8.webp"
    p_vp8l = tmp_path / "d_vp8l.webp"
    p_jpg = tmp_path / "e.jpg"
    p_bad = tmp_path / "broken.bin"

    _write_minimal_png(p_png, 832, 1440)
    _write_minimal_webp_vp8x(p_vp8x, 1024, 768)
    _write_minimal_webp_vp8(p_vp8, 640, 480)
    _write_minimal_webp_vp8l(p_vp8l, 512, 384)
    _write_minimal_jpeg(p_jpg, 1280, 720)
    p_bad.write_bytes(b"NOT_AN_IMAGE_HEADER")

    assert bs_mod._read_image_dimensions(p_png) == (832, 1440)
    assert bs_mod._read_image_dimensions(p_vp8x) == (1024, 768)
    assert bs_mod._read_image_dimensions(p_vp8) == (640, 480)
    assert bs_mod._read_image_dimensions(p_vp8l) == (512, 384)
    assert bs_mod._read_image_dimensions(p_jpg) == (1280, 720)
    assert bs_mod._read_image_dimensions(p_bad) == (None, None)


def test_studio_parse_comfy_prompt_graph_guider_loras_and_title_fallback() -> None:
    """验证 _parse_comfy_prompt_graph 支持 guider 链路、多种 LoRA 格式及标题兜底扫描。"""
    # 1) SamplerCustom + Guider + Power Lora Loader + 编号槽位 LoRA
    graph_guider = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "models\\anima_base.safetensors"}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "masterpiece, blue sky"}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "lowres, bad hands"}},
        "4": {"class_type": "CFGGuider", "inputs": {"cfg": 4.5, "positive": ["2", 0], "negative": ["3", 0]}},
        "5": {"class_type": "RandomNoise", "inputs": {"noise_seed": 424242}},
        "6": {"class_type": "BasicScheduler", "inputs": {"steps": 28, "scheduler": "simple"}},
        "7": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "er_sde"}},
        "8": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["5", 0],
                "guider": ["4", 0],
                "sampler": ["7", 0],
                "sigmas": ["6", 0],
            },
        },
        "9": {
            "class_type": "Power Lora Loader (rgthree)",
            "inputs": {
                "lora_1": {"lora": "dir/style_a.safetensors", "strength": 0.85, "on": True},
                "lora_2": {"lora": "disabled_lora.safetensors", "strength": 1.0, "on": False},
                "lora_name_3": "char_b.safetensors",
                "lora_wt_3": 0.7,
            },
        },
        "10": {"class_type": "EmptyLatentImage", "inputs": {"width": 832, "height": 1440}},
    }
    meta = bs_mod._parse_comfy_prompt_graph(graph_guider)
    assert meta["positive_prompt"] == "masterpiece, blue sky"
    assert meta["negative_prompt"] == "lowres, bad hands"
    assert meta["seed"] == 424242
    assert meta["steps"] == 28
    assert meta["cfg"] == pytest.approx(4.5)
    assert meta["sampler"] == "er_sde"
    assert meta["scheduler"] == "simple"
    assert meta["model"] == "anima_base.safetensors"
    assert meta["width"] == 832 and meta["height"] == 1440
    assert meta["loras"] == [
        {"name": "style_a.safetensors", "strength": 0.85},
        {"name": "char_b.safetensors", "strength": 0.7},
    ]

    # 2) 无采样器连线时的节点标题兜底扫描
    graph_fallback = {
        "a": {"class_type": "CLIPTextEncode", "_meta": {"title": "正面提示词"}, "inputs": {"text": "sunlit forest"}},
        "b": {"class_type": "CLIPTextEncode", "_meta": {"title": "负面过滤"}, "inputs": {"text": "blurry"}},
    }
    fb = bs_mod._parse_comfy_prompt_graph(graph_fallback)
    assert fb["positive_prompt"] == "sunlit forest"
    assert fb["negative_prompt"] == "blurry"


def test_studio_gallery_date_range_natural_sort_and_webui_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """验证 list_gallery_items 的时间窗口过滤、自然排序及 inspect_gallery_item 的 WebUI 参数回退脱敏。"""
    fixed_now = 1700500000.0
    revealed: list[Path] = []
    backend = _make_isolated_backend(
        tmp_path,
        monkeypatch,
        ctx_overrides={
            "now": lambda: fixed_now,
            "reveal_path": lambda p: revealed.append(Path(p)),
        },
    )
    dl = backend.download_root

    p2 = dl / "img_2.png"
    p10 = dl / "img_10.png"
    p_old = dl / "archive" / "img_1.webp"
    _write_minimal_png(
        p2,
        640,
        480,
        texts={"parameters": "hero portrait token=sk-pet-FakeLeak1234567890abcd\nNegative prompt: ugly\nSteps: 20"},
    )
    _write_minimal_png(p10, 320, 240)
    p_old.parent.mkdir(parents=True, exist_ok=True)
    _write_minimal_webp_vp8x(p_old, 512, 512)

    os.utime(p2, (fixed_now - 60, fixed_now - 60))
    os.utime(p10, (fixed_now - 3 * 86400, fixed_now - 3 * 86400))
    os.utime(p_old, (fixed_now - 20 * 86400, fixed_now - 20 * 86400))

    # 自然排序：img_1.webp < img_2.png < img_10.png
    asc = backend.list_gallery_items({"sort": "name_asc"})
    assert [it["name"] for it in asc["items"]] == ["img_1.webp", "img_2.png", "img_10.png"]

    # 时间范围过滤：today / 7d / 30d
    assert [it["name"] for it in backend.list_gallery_items({"date_range": "today"})["items"]] == ["img_2.png"]
    assert [it["name"] for it in backend.list_gallery_items({"date_range": "7d"})["items"]] == ["img_2.png", "img_10.png"]
    assert len(backend.list_gallery_items({"date_range": "30d"})["items"]) == 3

    # WebUI parameters 解析且强制脱敏
    insp = backend.inspect_gallery_item("img_2.png")
    assert insp["ok"] is True
    assert "hero portrait" in insp["meta"]["positive_prompt"]
    assert "sk-pet-FakeLeak1234567890abcd" not in insp["meta"]["positive_prompt"]
    assert insp["meta"]["negative_prompt"] == "ugly"

    # 非 PNG WebP 文件同样解析纯标准库宽高
    insp_webp = backend.inspect_gallery_item("archive/img_1.webp")
    assert insp_webp["ok"] is True
    assert insp_webp["meta"]["width"] == 512 and insp_webp["meta"]["height"] == 512

    # reveal_gallery_item 经由 ctx.reveal_path 定位
    rev = backend.reveal_gallery_item("img_2.png")
    assert rev["ok"] is True
    assert revealed == [p2.resolve()]


def test_studio_tail_live_logs_llama_cluster_slot_and_cloud_assets_partial_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """验证 tail_live_logs 的 llama 槽位解析与脱敏，以及 list_cloud_assets 并发容错。"""
    fake_py = tmp_path / "colab_py.exe"
    fake_py.write_bytes(b"MZ")
    fake_rclone = tmp_path / "rclone.exe"
    fake_rclone.write_bytes(b"MZ")
    rclone_conf = tmp_path / "rclone.conf"
    rclone_conf.write_text("[gdrive_save]\ntype = drive\n", encoding="utf-8")

    seen_cmds: list[list[str]] = []

    def _fake_run(cmd: list[str], **_kw: Any) -> CommandResult:
        seen_cmds.append(list(cmd))
        if "contents-tail" in cmd:
            frame = {
                "ok": True,
                "path": "root/llama_server.log",
                "lines": ["llama ready bearer sk-pet-SecretLogToken9999999999"],
            }
            return CommandResult(
                cmd,
                0,
                stdout=f"<<<COLAB_CONTENTS_TAIL>>>{json.dumps(frame)}<<<END_COLAB_CONTENTS_TAIL>>>",
                stderr="",
            )
        if "lsjson" in cmd:
            target = cmd[-1]
            if "loras" in target.lower():
                return CommandResult(
                    cmd,
                    0,
                    stdout=json.dumps([{"Name": "a.safetensors", "Path": "a.safetensors", "Size": 123, "IsDir": False}]),
                    stderr="",
                )
            return CommandResult(cmd, 1, stdout="", stderr="directory not found")
        return CommandResult(cmd, 0, stdout="", stderr="")

    backend = _make_isolated_backend(
        tmp_path,
        monkeypatch,
        ctx_overrides={
            "get_colab_py": lambda: fake_py,
            "get_rclone_exe": lambda: fake_rclone,
            "get_rclone_conf": lambda: rclone_conf,
            "run_hidden": _fake_run,
            "now": lambda: 1700500000.0,
        },
    )
    backend._store.state["cluster_slots"] = {
        "llm": {"account_id": "acct_llm_1", "endpoint": "ep-llm-9", "online": True},
    }

    res_log = backend.tail_live_logs({"source": "llama", "lines": 30})
    assert res_log["ok"] is True
    assert "sk-pet-SecretLogToken9999999999" not in res_log["lines"][0]
    tail_cmd = seen_cmds[0]
    assert "--ep=ep-llm-9" in tail_cmd
    assert "--acct=acct_llm_1" in tail_cmd
    assert "--path=root/llama_server.log" in tail_cmd

    cloud_res = backend.list_cloud_assets()
    assert cloud_res["ok"] is True
    assert len(cloud_res["loras"]) == 1 and cloud_res["loras"][0]["name"] == "a.safetensors"
    assert cloud_res["workflows"] == [] and cloud_res["refs"] == []


def test_single_node_multipool_account_propagation_and_right_log_banner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """验证多号池单机重启隧道与拉回成图必须透传 --account，而单号模式保持无 --account，且日志顶栏不误报。"""
    ctl_calls: list[list[str]] = []
    alive_set: set[str] = set()

    def _fake_probe(url: str, **_kw: Any) -> bool:
        return url.rstrip("/") in alive_set

    def _fake_detect(url: str, **_kw: Any) -> str | None:
        return "comfy" if url.rstrip("/") in alive_set else None

    def _fake_run_hidden(cmd: list[str], **_kw: Any) -> CommandResult:
        ctl_calls.append(list(cmd))
        alive_set.add("https://restarted-acct.trycloudflare.com")
        return CommandResult(
            cmd,
            0,
            stdout="COMFYUI_TUNNEL_URL: https://restarted-acct.trycloudflare.com\n<<<SYNC_VERIFIED>>>\n",
            stderr="",
        )

    backend = _make_isolated_backend(
        tmp_path,
        monkeypatch,
        ctx_overrides={
            "open_browser": lambda _u: True,
            "reveal_path": lambda _p: True,
            "probe_tunnel_alive": _fake_probe,
            "detect_tunnel_kind": _fake_detect,
            "run_hidden": _fake_run_hidden,
            "ctl_available": lambda: True,
            "ctl_cmd_prefix": lambda unbuffered=False: ["python", "colab_ctl.py"],
            "load_cf_secret_raw": lambda **_kw: {},
            "now": lambda: 1700000000.0,
        },
    )
    backend._comfy_svc._ctx = backend._ctx
    backend._store.state["last_sessions"] = [{"name": "comfy", "endpoint": "ep-single"}]
    backend._store.state["last_assignments_count"] = 1
    backend._store.state["prefs"]["active_account_id"] = "acct_only_1"
    backend._store.state["account_profiles"] = [{"id": "acct_only_1", "label": "Single"}]

    # 单号默认模式：重启隧道与同步均不传递多余 --account
    with patch.object(backend, "_fast_read_tunnel_url", return_value=("missing", "ep-single", None)):
        res_single = backend.open_comfy_tunnel()
    assert res_single["ok"] is True
    assert len(ctl_calls) == 1 and "tunnel" in ctl_calls[-1] and "--account" not in ctl_calls[-1]

    with patch.object(backend, "_fast_read_tunnel_url", return_value=("missing", "ep-single", None)):
        sync_single = backend.sync_outputs()
    assert sync_single["ok"] is True
    assert len(ctl_calls) == 2 and "sync" in ctl_calls[-1] and "--account" not in ctl_calls[-1]

    # 多号池但部署时未带 --account（未开智能寻优，会话在全局 sessions.json）：绝不能擅自补 --account
    alive_set.clear()
    ctl_calls.clear()
    backend._store.state["tunnel_cache"] = {}
    backend._store.state["prefs"]["active_account_id"] = "acct_pool_1"
    backend._store.state["account_profiles"] = [
        {"id": "acct_pool_1", "label": "Ultra1"},
        {"id": "acct_pool_2", "label": "Pro2"},
    ]
    backend._store.record_managed_binding("comfy", "ep-single", persist=False, cli_scope="")
    with patch.object(backend, "_fast_read_tunnel_url", return_value=("missing", "ep-single", None)):
        res_global = backend.open_comfy_tunnel()
    assert res_global["ok"] is True
    assert len(ctl_calls) == 1 and "--account" not in ctl_calls[0]

    # 多号池且部署时带了 --account（会话在账号隔离文件）：重启隧道与拉回成图均沿用该作用域
    alive_set.clear()
    ctl_calls.clear()
    backend._store.state["tunnel_cache"] = {}
    backend._store.record_managed_binding("comfy", "ep-single", persist=False, cli_scope="acct_pool_1")

    with patch.object(backend, "_fast_read_tunnel_url", return_value=("missing", "ep-single", None)):
        res_pool = backend.open_comfy_tunnel()
    assert res_pool["ok"] is True
    assert len(ctl_calls) == 1
    restart_cmd = ctl_calls[0]
    assert "tunnel" in restart_cmd and "--restart" in restart_cmd
    assert "--account" in restart_cmd and restart_cmd[restart_cmd.index("--account") + 1] == "acct_pool_1"

    # sync 内部重写绑定（未传 cli_scope）时必须沿用同端点已记录的作用域
    with patch.object(backend, "_fast_read_tunnel_url", return_value=("missing", "ep-single", None)):
        sync_pool = backend.sync_outputs()
    assert sync_pool["ok"] is True
    assert len(ctl_calls) == 2
    sync_cmd = ctl_calls[-1]
    assert "sync" in sync_cmd
    assert "--account" in sync_cmd and sync_cmd[sync_cmd.index("--account") + 1] == "acct_pool_1"
    assert backend._store.state["managed_binding"]["cli_scope"] == "acct_pool_1"

    # 端点变化（换了新实例）后作用域视为未知，回退为不传 --account
    backend._store.record_managed_binding("comfy", "ep-other", persist=False)
    assert backend._store.resolve_single_node_account_id("ep-other") == ""
    assert backend._store.resolve_single_node_account_id("ep-single") == ""

    # 部署命令行作用域提取与状态文件清洗往返保留 cli_scope
    import backend_deploy as bd_mod
    from colab_core import ledger as ledger_mod

    assert bd_mod._extract_cmd_account(["py", "colab_ctl.py", "up", "-s", "comfy", "--account", "acct_x"]) == "acct_x"
    assert bd_mod._extract_cmd_account(["py", "colab_ctl.py", "up", "-s", "comfy"]) == ""
    assert ledger_mod._sanitize_managed_binding(
        {"account_id": "a1", "session": "comfy", "endpoint": "ep1", "cli_scope": "acct_x"}
    )["cli_scope"] == "acct_x"
    assert "cli_scope" not in ledger_mod._sanitize_managed_binding(
        {"account_id": "a1", "session": "comfy", "endpoint": "ep1"}
    )

    studio_js = (PKG_DIR / "colab_pet" / "studio.js").read_text(encoding="utf-8")
    assert "实例在线 · 等待隧道连通" in studio_js
    assert "隧道已连通 · 等待任务触发" in studio_js
