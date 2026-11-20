"""Phase 0B 隧道鉴权与域名解耦 + 公开基础版开箱能力离线回归测试。

全量使用本地内存 HTTP 桩与 `tmp_path` 沙箱环境变量（`COLAB_PET_MOCK_VAULT_PATH`、
`COLAB_PET_CF_CONFIG_PATH`、`COLAB_PET_DIAG_LOG_PATH`），严禁读写宿主机真实凭据库或发起外网请求。
"""

from __future__ import annotations

import argparse
import contextlib
import http.server
import io
import json
import re
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

PKG_DIR = Path(__file__).resolve().parents[1]
if str(PKG_DIR) not in sys.path:
    sys.path.insert(0, str(PKG_DIR))

import colab_bootstrap_anima as boot_anima  # noqa: E402
import colab_bootstrap_drama3 as boot_drama3  # noqa: E402
import colab_ctl  # noqa: E402
import colab_remote_common as rcom  # noqa: E402
import colab_security as csec  # noqa: E402
import verify_smoke  # noqa: E402
from colab_core import colab_client, health, runtime  # noqa: E402


@pytest.fixture()
def isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """为每条用例构造隔离的凭据沙箱、隧道配置与诊断日志路径。"""
    csec.clear_ephemeral_secrets()
    vault_file = tmp_path / "mock_vault.json"
    cf_file = tmp_path / ".cf_tunnel_secret.json"
    diag_log = tmp_path / "logs" / "colab_diag.log"
    monkeypatch.setenv("COLAB_PET_MOCK_VAULT_PATH", str(vault_file))
    monkeypatch.setenv("COLAB_PET_CF_CONFIG_PATH", str(cf_file))
    monkeypatch.setenv("COLAB_PET_DIAG_LOG_PATH", str(diag_log))
    monkeypatch.delenv("COLAB_PET_EDITION", raising=False)
    monkeypatch.delenv("ANIMA_STRICT_LORAS", raising=False)
    monkeypatch.delenv("ANIMA_SMOKE_MODE", raising=False)
    monkeypatch.delenv("DRAMA3_DEFAULT_REF_WAV", raising=False)
    monkeypatch.delenv("COLAB_DRAMA3_REFS_DIR", raising=False)
    yield SimpleNamespace(tmp=tmp_path, vault_file=vault_file, cf_file=cf_file, diag_log=diag_log)
    csec.clear_ephemeral_secrets()


@contextlib.contextmanager
def _start_stub_server(handler_cls: type[http.server.BaseHTTPRequestHandler]):
    """启动绑定到 127.0.0.1 随机端口的本地 HTTP 测试桩。"""
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    port = int(server.server_address[1])
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield port
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def _make_llm_public_stub(state: dict[str, Any]) -> type[http.server.BaseHTTPRequestHandler]:
    """构造可控制鉴权与错误状态的公网 LLM 端点桩。"""

    class _LlmStub(http.server.BaseHTTPRequestHandler):
        def log_message(self, _fmt: str, *_args: object) -> None:
            return

        def _json(self, code: int, payload: dict[str, Any]) -> None:
            raw = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                if state.get("health_ok", True):
                    self._json(200, {"status": "ok"})
                else:
                    self._json(503, {"status": "loading"})
                return
            if self.path == "/v1/models":
                if state.get("auth_enforced", False):
                    expected = f"Bearer {state.get('api_key', '')}"
                    if self.headers.get("Authorization") != expected:
                        self._json(int(state.get("reject_status", 401)), {"error": "unauthorized"})
                        return
                self._json(200, {"object": "list", "data": [{"id": "Qwen3.8-27B-abliterated"}]})
                return
            self._json(404, {"error": "not_found"})

    return _LlmStub


def _make_comfy_public_stub(state: dict[str, Any]) -> type[http.server.BaseHTTPRequestHandler]:
    """构造可控制公网鉴权是否生效的 ComfyUI 端点桩。"""

    class _ComfyStub(http.server.BaseHTTPRequestHandler):
        def log_message(self, _fmt: str, *_args: object) -> None:
            return

        def _json(self, code: int, payload: dict[str, Any]) -> None:
            raw = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/__pet_health":
                self._json(
                    200,
                    {
                        "status": "ok",
                        "service": "comfy",
                        "auth_required": bool(state.get("auth_enforced", False)),
                    },
                )
                return
            if self.path == "/system_stats":
                if state.get("auth_enforced", False):
                    expected = f"Bearer {state.get('comfy_key', '')}"
                    if self.headers.get("Authorization") != expected:
                        self._json(int(state.get("reject_status", 401)), {"error": "unauthorized"})
                        return
                self._json(
                    200,
                    {
                        "system": {"os": "posix"},
                        "devices": [{"name": "NVIDIA L4", "vram_total": 24 * (1024**3)}],
                    },
                )
                return
            self._json(404, {"error": "not_found"})

    return _ComfyStub


# =========================================================================
# 1. 清除私人域名硬编码 & 域名提取 vs 凭据发送源站授权解耦
# =========================================================================


def test_domain_decoupling_extract_and_credential_origin_isolation(isolated_env: SimpleNamespace) -> None:
    """验证私人域名已彻底清除，且 URL 提取正则与凭据源站授权严格解耦。"""
    # 1a. 4 份隧道核心源码不再残留硬编码私人域名
    for rel in (
        "colab_core/colab_client.py",
        "colab_core/runtime.py",
        "colab_cli_bridge.py",
        "colab_remote_common.py",
    ):
        src = (PKG_DIR / rel).read_text(encoding="utf-8")
        assert "private.example.com" not in src, f"{rel} 仍残留硬编码私人域名"

    # 1b. 默认提取正则仅匹配官方 trycloudflare.com，拒绝未授权私人域名、api.trycloudflare 与后缀伪造
    default_pat = colab_client.build_tunnel_url_pattern({})
    default_re = re.compile(default_pat)
    assert default_re.search("COMFYUI_TUNNEL_URL: https://demo-node-1.trycloudflare.com").group(0) == (
        "https://demo-node-1.trycloudflare.com"
    )
    assert default_re.search("COMFYUI_TUNNEL_URL: https://api.trycloudflare.com") is None
    assert default_re.search("COMFYUI_TUNNEL_URL: https://comfy.private.example.com") is None
    assert default_re.search("COMFYUI_TUNNEL_URL: https://demo-node-1.trycloudflare.com.evil.com") is None
    assert rcom.DEFAULT_TUNNEL_URL_RE.search("https://comfy.private.example.com") is None

    # 1c. 自定义任意域名经 build_tunnel_url_pattern + consume_deploy_line 动态解析
    custom_cf = isolated_env.cf_file
    custom_cf.write_text(
        json.dumps(
            {
                "comfy_url": "https://draw.my-custom-studio.example",
                "llm_url": "https://chat.my-custom-studio.example",
            }
        ),
        encoding="utf-8",
    )
    custom_pat = colab_client.build_tunnel_url_pattern(cf_config_path=custom_cf)
    st_default = runtime.default_deploy_state()
    ctx_default: dict[str, Any] = {"phase_idx": 1, "phase_total": 1}
    # 未传动态 pattern 时，自定义域名与旧私人域名均不会被默认规则误捕
    assert runtime.consume_deploy_line("COMFYUI_TUNNEL_URL: https://comfy.private.example.com", st_default, ctx_default, {}) == (None, False)
    assert runtime.consume_deploy_line("COMFYUI_TUNNEL_URL: https://draw.my-custom-studio.example", st_default, ctx_default, {}) == (None, False)
    # 传入动态 tunnel_pattern 后，支持任意配置域名双阶段提取
    st_dyn = runtime.default_deploy_state()
    ctx_dyn: dict[str, Any] = {"phase_idx": 1, "phase_total": 2}
    runtime.consume_deploy_line("DEPLOY_PHASE 1/2: qwen_27b", st_dyn, ctx_dyn, {}, tunnel_pattern=custom_pat)
    runtime.consume_deploy_line(
        "LLM_TUNNEL_URL: https://chat.my-custom-studio.example",
        st_dyn,
        ctx_dyn,
        {},
        tunnel_pattern=custom_pat,
    )
    runtime.consume_deploy_line("DEPLOY_PHASE 2/2: anima", st_dyn, ctx_dyn, {}, tunnel_pattern=custom_pat)
    got_url, should_warm = runtime.consume_deploy_line(
        "COMFYUI_TUNNEL_URL: https://draw.my-custom-studio.example",
        st_dyn,
        ctx_dyn,
        {},
        tunnel_pattern=re.compile(custom_pat),
    )
    assert got_url == "https://draw.my-custom-studio.example"
    assert should_warm is True
    assert ctx_dyn["captured_llm_tunnel"] == "https://chat.my-custom-studio.example"

    # 1d. 凭据发送源站授权（is_authorized_credential_origin）严格杜绝宽泛正则误发密钥
    bound_trycf = "https://bound-session-abc.trycloudflare.com"
    stranger_trycf = "https://attacker-xyz.trycloudflare.com"
    assert colab_client.is_authorized_credential_origin(stranger_trycf) is False
    assert colab_client.is_authorized_credential_origin(stranger_trycf, config_path=custom_cf) is False
    assert colab_client.is_authorized_credential_origin(
        stranger_trycf, allowed_origins=[bound_trycf]
    ) is False
    assert colab_client.is_authorized_credential_origin(
        f"{bound_trycf}/system_stats", allowed_origins=[bound_trycf]
    ) is True
    assert colab_client.is_authorized_credential_origin(
        "https://draw.my-custom-studio.example/system_stats", config_path=custom_cf
    ) is True
    assert colab_client.is_authorized_credential_origin(
        "https://chat.my-custom-studio.example/v1/models", config_path=custom_cf
    ) is True
    # 拒绝非 HTTPS、端口不匹配、后缀伪造、userinfo 欺骗与未配置私人域名
    assert colab_client.is_authorized_credential_origin(
        "http://draw.my-custom-studio.example",
        allowed_origins=["http://draw.my-custom-studio.example"],
    ) is False
    assert colab_client.is_authorized_credential_origin(
        "https://draw.my-custom-studio.example:8443", config_path=custom_cf
    ) is False
    assert colab_client.is_authorized_credential_origin(
        "https://draw.my-custom-studio.example.evil.com", config_path=custom_cf
    ) is False
    assert colab_client.is_authorized_credential_origin(
        "https://draw.my-custom-studio.example@evil.com", config_path=custom_cf
    ) is False
    assert colab_client.is_authorized_credential_origin("https://comfy.private.example.com", config_path=custom_cf) is False

    # 1e. is_allowed_tunnel_url 向后兼容：放行官方 trycloudflare 与配置域名，阻断伪造与未配置私人域名
    assert colab_client.is_allowed_tunnel_url(bound_trycf) is True
    assert colab_client.is_allowed_tunnel_url("https://api.trycloudflare.com") is False
    assert colab_client.is_allowed_tunnel_url("https://bound-session-abc.trycloudflare.com.evil.com") is False
    assert colab_client.is_allowed_tunnel_url("https://comfy.private.example.com") is False
    assert colab_client.is_allowed_tunnel_url("https://draw.my-custom-studio.example", config_path=custom_cf) is True


# =========================================================================
# 2. cmd_tunnel 快速复用通道双向鉴权拦截 & Drama 3 本地路径解耦
# =========================================================================


def test_cmd_tunnel_fast_reuse_enforces_two_way_auth_for_llm_dual_and_comfy(
    isolated_env: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 cmd_tunnel 快速复用严禁放行未鉴权公网 LLM/Comfy 或缺密钥端点，并校验路径环境变量化。"""
    tmp = isolated_env.tmp
    fake_py = tmp / "colab_py.exe"
    fake_bridge = tmp / "colab_cli_bridge.py"
    fake_py.write_text("", encoding="utf-8")
    fake_bridge.write_text("", encoding="utf-8")

    # 2a. colab_ctl.py 无个人路径硬编码，且 COLAB_DRAMA3_REFS_DIR 优先、默认回退 WORKSPACE_ROOT/drama3_refs
    ctl_src = (PKG_DIR / "colab_ctl.py").read_text(encoding="utf-8")
    assert "prepared_wav" not in ctl_src
    assert colab_ctl._resolve_drama3_refs_dir() == colab_ctl.WORKSPACE_ROOT / "drama3_refs"
    custom_refs = tmp / "my_drama3_refs"
    monkeypatch.setenv("COLAB_DRAMA3_REFS_DIR", str(custom_refs))
    assert colab_ctl._resolve_drama3_refs_dir() == custom_refs

    captured_sync_dirs: list[Path] = []
    monkeypatch.setattr(
        colab_ctl,
        "sync_drama3_refs_cli",
        lambda *, src_dir, **_kw: captured_sync_dirs.append(Path(src_dir)),
    )
    colab_ctl.cmd_sync_drama3_refs(argparse.Namespace(src=None))
    assert captured_sync_dirs == [custom_refs]

    # 2b. 启动真实本地 HTTP 桩验证双向鉴权探针与 cmd_tunnel 快速复用分流
    llm_key = "sk-pet-fast-reuse-verify-123456"
    comfy_key = "cpk-fast-reuse-verify-abcdef123"
    llm_state: dict[str, Any] = {"health_ok": True, "auth_enforced": False, "api_key": llm_key, "reject_status": 401}
    comfy_state: dict[str, Any] = {"auth_enforced": False, "comfy_key": comfy_key, "reject_status": 401}

    with (
        _start_stub_server(_make_llm_public_stub(llm_state)) as llm_port,
        _start_stub_server(_make_comfy_public_stub(comfy_state)) as comfy_port,
    ):
        llm_url = f"http://127.0.0.1:{llm_port}"
        comfy_url = f"http://127.0.0.1:{comfy_port}"

        # 探针单测：未鉴权裸奔 LLM / 缺密钥 / 错密钥均必拒；双向鉴权生效才放行
        assert colab_client.probe_llm_endpoint_auth_ok(llm_url, "")[0] is False
        assert colab_client.probe_llm_endpoint_auth_ok(llm_url, llm_key)[0] is False
        llm_state["auth_enforced"] = True
        assert colab_client.probe_llm_endpoint_auth_ok(llm_url, "sk-wrong-key")[0] is False
        assert colab_client.probe_llm_endpoint_auth_ok(llm_url, llm_key)[0] is True
        llm_state["reject_status"] = 403
        assert colab_client.probe_llm_endpoint_auth_ok(llm_url, llm_key)[0] is True
        llm_state["reject_status"] = 401

        # 探针单测：无保护公网 ComfyUI（匿名 200）严禁放行
        assert colab_client.probe_comfy_endpoint_ok(comfy_url, comfy_key=comfy_key)[0] is False
        comfy_state["auth_enforced"] = True
        assert colab_client.probe_comfy_endpoint_ok(comfy_url, comfy_key="cpk-wrong")[0] is False
        assert colab_client.probe_comfy_endpoint_ok(comfy_url, comfy_key="", require_key=True)[0] is False
        assert colab_client.probe_comfy_endpoint_ok(comfy_url, comfy_key="", require_key=False)[0] is True
        assert colab_client.probe_comfy_endpoint_ok(comfy_url, comfy_key=comfy_key, require_key=True)[0] is True

        # 端到端验证 colab_ctl.cmd_tunnel 快速复用通道
        monkeypatch.setattr(colab_ctl, "COLAB_PY", fake_py)
        monkeypatch.setattr(colab_ctl, "CLI_BRIDGE_SCRIPT", fake_bridge)
        monkeypatch.setattr(colab_ctl, "_build_tunnel_url_pattern", lambda _cfg=None: r"http://127\.0\.0\.1:\d+")
        monkeypatch.setattr(colab_ctl.time, "sleep", lambda *_a: None)

        cf_cfg_box: dict[str, Any] = {"tunnel_token": "", "llm_url": "", "comfy_url": "", "llm_api_key": ""}
        monkeypatch.setattr(colab_ctl, "_load_cf_tunnel_secret", lambda: dict(cf_cfg_box))

        fast_pkt_box: dict[str, Any] = {}
        remote_calls: list[str] = []

        def _fake_subprocess_run(_cmd, **_kw):
            out = "<<<COLAB_TUNNEL>>>" + json.dumps(fast_pkt_box, ensure_ascii=False) + "<<<END_COLAB_TUNNEL>>>"
            return subprocess.CompletedProcess([], 0, stdout=out, stderr="")

        def _fake_exec_remote(_sess: str, code: str, **_kw):
            remote_calls.append(code)
            # 模拟远端修复/重建完成并开启鉴权后返回地址
            llm_state["auth_enforced"] = True
            comfy_state["auth_enforced"] = True
            kind = str((fast_pkt_box.get("meta") or {}).get("kind") or "comfy")
            if kind == "llm":
                stdout = f"SERVICE_KIND: llm\nSERVICE_TUNNEL_URL: {llm_url}\nCOMFYUI_TUNNEL_URL: {llm_url}\n"
            elif kind == "dual":
                stdout = (
                    f"SERVICE_KIND: dual\nLLM_TUNNEL_URL: {llm_url}\n"
                    f"SERVICE_TUNNEL_URL: {comfy_url}\nCOMFYUI_TUNNEL_URL: {comfy_url}\n"
                )
            else:
                stdout = f"SERVICE_KIND: comfy\nCOMFYUI_TUNNEL_URL: {comfy_url}\n"
            return subprocess.CompletedProcess([], 0, stdout=stdout, stderr="")

        monkeypatch.setattr(colab_ctl.subprocess, "run", _fake_subprocess_run)
        monkeypatch.setattr(colab_ctl, "_exec_remote_py", _fake_exec_remote)

        args = argparse.Namespace(session="comfy", restart=False, account="")

        # 场景 1：fast_kind == "llm" 且公网 LLM 未鉴权（/health 200 但 /v1/models 裸奔 200）-> 严禁快速复用，必须走远端重建
        llm_state["auth_enforced"] = False
        remote_calls.clear()
        fast_pkt_box.clear()
        fast_pkt_box.update(
            {
                "status": "ok",
                "url": llm_url,
                "api_key": llm_key,
                "meta": {"kind": "llm", "model": "Qwen3.8-27B"},
            }
        )
        colab_ctl.cmd_tunnel(args)
        assert len(remote_calls) == 1, "未鉴权公网 LLM 绝不能走快速复用返回"

        # 场景 2：fast_kind == "llm" 但缺少 api_key -> 严禁快速复用，必须走远端流程
        llm_state["auth_enforced"] = True
        remote_calls.clear()
        fast_pkt_box.clear()
        fast_pkt_box.update(
            {
                "status": "ok",
                "url": llm_url,
                "api_key": "",
                "meta": {"kind": "llm", "model": "Qwen3.8-27B"},
            }
        )
        colab_ctl.cmd_tunnel(args)
        assert len(remote_calls) == 1, "缺少 LLM 密钥时绝不能走快速复用返回"

        # 场景 3：fast_kind == "llm" 且通过双向鉴权（密钥来自固定配置 cf_cfg.llm_api_key）-> 快速复用直接返回
        llm_state["auth_enforced"] = True
        cf_cfg_box["llm_api_key"] = llm_key
        remote_calls.clear()
        colab_ctl.cmd_tunnel(args)
        assert remote_calls == [], "双向鉴权通过的公网 LLM 应允许快速复用"
        cf_cfg_box["llm_api_key"] = ""

        # 场景 4：fast_kind == "dual"，Comfy 已鉴权但 LLM 裸奔 -> 严禁快速复用
        comfy_state["auth_enforced"] = True
        llm_state["auth_enforced"] = False
        remote_calls.clear()
        fast_pkt_box.clear()
        fast_pkt_box.update(
            {
                "status": "ok",
                "url": comfy_url,
                "comfy_api_key": comfy_key,
                "meta": {
                    "kind": "dual",
                    "llm_tunnel_url": llm_url,
                    "api_key": llm_key,
                    "model": "Qwen3.8-27B",
                },
            }
        )
        colab_ctl.cmd_tunnel(args)
        assert len(remote_calls) == 1, "双栖模式下 LLM 未鉴权绝不能快速复用"

        # 场景 5：fast_kind == "dual"，LLM 已鉴权但 Comfy 裸奔 -> 严禁快速复用
        comfy_state["auth_enforced"] = False
        llm_state["auth_enforced"] = True
        remote_calls.clear()
        colab_ctl.cmd_tunnel(args)
        assert len(remote_calls) == 1, "双栖模式下 Comfy 无保护公网入口绝不能快速复用"

        # 场景 6：fast_kind == "dual"，双端均通过双向鉴权 -> 快速复用直接返回
        comfy_state["auth_enforced"] = True
        llm_state["auth_enforced"] = True
        remote_calls.clear()
        colab_ctl.cmd_tunnel(args)
        assert remote_calls == [], "双栖双端双向鉴权均通过时应允许快速复用"

        # 场景 7：fast_kind == "comfy"，公网 Comfy 无保护（匿名 200）-> 严禁快速复用
        comfy_state["auth_enforced"] = False
        remote_calls.clear()
        fast_pkt_box.clear()
        fast_pkt_box.update(
            {
                "status": "ok",
                "url": comfy_url,
                "comfy_api_key": comfy_key,
                "meta": {"kind": "comfy"},
            }
        )
        colab_ctl.cmd_tunnel(args)
        assert len(remote_calls) == 1, "无保护公网 Comfy 入口绝不能快速复用"


# =========================================================================
# 3. 公开基础版 vs 个人增强版：Preflight / Bootstrap / Drama3 / Smoke
# =========================================================================


def test_public_vs_personal_edition_preflight_bootstrap_and_smoke(
    isolated_env: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证干净配置下公开基础版可完整跑通 preflight/bootstrap/drama3/smoke，且个人增强版强校验保留。"""
    tmp = isolated_env.tmp
    fake_colab_exe = tmp / "colab.exe"
    fake_colab_py = tmp / "colab_py.exe"
    fake_colab_exe.write_text("", encoding="utf-8")
    fake_colab_py.write_text("", encoding="utf-8")

    # 3a. Preflight：干净新机（无 rclone.exe 且无 rclone.conf / [gdrive_save]）
    clean_tp = health.ToolPaths(
        colab_exe=fake_colab_exe,
        colab_py=fake_colab_py,
        rclone_exe=tmp / "missing_rclone.exe",
        rclone_conf=tmp / "missing_rclone.conf",
    )

    # 公开基础版（edition="public" / public_mode=True / 环境变量 / require_local_rclone=False）均降级为 warn 且 ok=True
    pub_rep = health.run_preflight_checks(
        mode="fast",
        profiles=["anima"],
        gpu="L4",
        tool_paths=clean_tp,
        download_dir=tmp,
        cf_config_path=isolated_env.cf_file,
        pkg_dir=PKG_DIR,
        edition="public",
    )
    assert pub_rep.ok is True, pub_rep.format_text()
    assert pub_rep.versions.get("edition") == "public"
    gd_item = next(it for it in pub_rep.items if it.name == "Google 云盘配置")
    rc_item = next(it for it in pub_rep.items if it.name == "rclone")
    assert gd_item.status == "warn" and "公开基础版" in gd_item.summary
    assert rc_item.status == "warn"

    pub_bool_rep = health.run_preflight_checks(
        mode="fast",
        profiles=["anima"],
        gpu="L4",
        tool_paths=clean_tp,
        download_dir=tmp,
        cf_config_path=isolated_env.cf_file,
        pkg_dir=PKG_DIR,
        public_mode=True,
    )
    assert pub_bool_rep.ok is True

    relax_rclone_rep = health.run_preflight_checks(
        mode="fast",
        profiles=["anima"],
        gpu="L4",
        tool_paths=clean_tp,
        download_dir=tmp,
        cf_config_path=isolated_env.cf_file,
        pkg_dir=PKG_DIR,
        require_local_rclone=False,
    )
    assert relax_rclone_rep.ok is True
    assert next(it for it in relax_rclone_rep.items if it.name == "Google 云盘配置").status == "warn"

    monkeypatch.setenv("COLAB_PET_EDITION", "public")
    env_pub_rep = health.run_preflight_checks(
        mode="fast",
        profiles=["anima"],
        gpu="L4",
        tool_paths=clean_tp,
        download_dir=tmp,
        cf_config_path=isolated_env.cf_file,
        pkg_dir=PKG_DIR,
    )
    assert env_pub_rep.ok is True
    # 即使在 public 环境下，显式 require_local_rclone=True 仍执行强校验
    explicit_strict_rep = health.run_preflight_checks(
        mode="fast",
        profiles=["anima"],
        gpu="L4",
        tool_paths=clean_tp,
        download_dir=tmp,
        cf_config_path=isolated_env.cf_file,
        pkg_dir=PKG_DIR,
        require_local_rclone=True,
    )
    assert explicit_strict_rep.ok is False
    monkeypatch.delenv("COLAB_PET_EDITION", raising=False)

    # 个人增强版（默认不传参数）保持原有强校验：缺失 [gdrive_save] 必须 fail
    personal_rep = health.run_preflight_checks(
        mode="fast",
        profiles=["anima"],
        gpu="L4",
        tool_paths=clean_tp,
        download_dir=tmp,
        cf_config_path=isolated_env.cf_file,
        pkg_dir=PKG_DIR,
    )
    assert personal_rep.ok is False
    assert next(it for it in personal_rep.items if it.name == "Google 云盘配置").status == "fail"

    # CLI doctor：无 rclone.conf 且未指定 --edition / COLAB_PET_EDITION 时默认按公开基础版通过
    monkeypatch.setattr(colab_ctl, "COLAB_EXE", fake_colab_exe)
    monkeypatch.setattr(colab_ctl, "COLAB_PY", fake_colab_py)
    monkeypatch.setattr(colab_ctl, "RCLONE_EXE", tmp / "missing_rclone.exe")
    monkeypatch.setattr(colab_ctl, "RCLONE_CONF", tmp / "missing_rclone.conf")

    buf_default = io.StringIO()
    with patch("sys.stdout", buf_default):
        rc_default = colab_ctl.main(["doctor", "--profile", "anima", "--gpu", "L4", "--json"])
    assert rc_default == 0
    doc_default = json.loads(buf_default.getvalue())
    assert doc_default["ok"] is True and doc_default["versions"]["edition"] == "public"
    assert any(it["name"] == "Google 云盘配置" and it["status"] == "warn" for it in doc_default["items"])

    buf = io.StringIO()
    with patch("sys.stdout", buf):
        rc_pub = colab_ctl.main(["doctor", "--profile", "anima", "--gpu", "L4", "--edition", "public", "--json"])
    assert rc_pub == 0
    doc_json = json.loads(buf.getvalue())
    assert doc_json["ok"] is True and doc_json["versions"]["edition"] == "public"

    # CLI doctor：显式 --edition personal 或环境变量 COLAB_PET_EDITION=personal 时缺 [gdrive_save] 按 SystemExit(1) 拦阻
    buf_personal_cli = io.StringIO()
    with patch("sys.stdout", buf_personal_cli), pytest.raises(SystemExit) as exc_cli:
        colab_ctl.main(["doctor", "--profile", "anima", "--gpu", "L4", "--edition", "personal", "--json"])
    assert exc_cli.value.code == 1
    doc_personal_cli = json.loads(buf_personal_cli.getvalue())
    assert doc_personal_cli["ok"] is False and doc_personal_cli["versions"]["edition"] == "personal"

    monkeypatch.setenv("COLAB_PET_EDITION", "personal")
    buf_personal_env = io.StringIO()
    with patch("sys.stdout", buf_personal_env), pytest.raises(SystemExit) as exc_env:
        colab_ctl.main(["doctor", "--profile", "anima", "--gpu", "L4", "--json"])
    assert exc_env.value.code == 1
    doc_personal_env = json.loads(buf_personal_env.getvalue())
    assert doc_personal_env["ok"] is False and doc_personal_env["versions"]["edition"] == "personal"
    monkeypatch.delenv("COLAB_PET_EDITION", raising=False)

    # 3b. colab_bootstrap_anima.py：公开基础版无 rclone.conf 平滑跳过私训 LoRA，个人增强版强制校验
    fake_comfy_dir = tmp / "ComfyUI"
    fake_comfy_db = tmp / "comfy_db"
    missing_rc = tmp / "no_rclone.conf"
    monkeypatch.setattr(boot_anima, "COMFY_DIR", fake_comfy_dir)
    monkeypatch.setattr(boot_anima, "COMFY_DB", fake_comfy_db)
    monkeypatch.setattr(boot_anima, "RCLONE_CONF", missing_rc)

    # 公开基础版：无 rclone.conf 不再抛出 FileNotFoundError，且完整 main()（mock）可顺畅走完纯底座流程
    boot_anima.pull_loras_from_gdrive()
    assert (fake_comfy_dir / "models" / "loras").is_dir()

    called_steps: list[str] = []
    monkeypatch.setattr(boot_anima, "check_gpu_and_sdpa", lambda: called_steps.append("gpu"))
    monkeypatch.setattr(boot_anima, "ensure_tools", lambda: called_steps.append("tools"))
    monkeypatch.setattr(boot_anima, "ensure_comfyui_repo", lambda *_a, **_kw: called_steps.append("repo"))
    monkeypatch.setattr(boot_anima, "download_anima_base", lambda: called_steps.append("base"))
    monkeypatch.setattr(boot_anima, "start_comfyui_and_wait", lambda **_kw: called_steps.append("comfy"))
    monkeypatch.setattr(boot_anima, "start_comfy_tunnel_standard", lambda: called_steps.append("tunnel") or "https://pub.trycloudflare.com")
    monkeypatch.setattr(boot_anima, "write_runtime_versions_record", lambda *_a, **_kw: called_steps.append("versions"))
    boot_anima.main()
    assert called_steps == ["gpu", "tools", "repo", "base", "comfy", "tunnel", "versions"]

    # 公开基础版有 rclone.conf 但云盘里无私训 LoRA 时不报错
    present_rc = tmp / "present_rclone.conf"
    present_rc.write_text("[gdrive_save]\ntype = drive\n", encoding="utf-8")
    monkeypatch.setattr(boot_anima, "RCLONE_CONF", present_rc)
    monkeypatch.setattr(boot_anima, "run_cmd", lambda cmd, check=True, **_kw: subprocess.CompletedProcess(cmd, 0, "", ""))
    boot_anima.pull_loras_from_gdrive()

    # 个人增强版（ANIMA_STRICT_LORAS=1）：无 rclone.conf 必报错；有 rclone.conf 缺主力 LoRA 必报错
    monkeypatch.setenv("ANIMA_STRICT_LORAS", "1")
    monkeypatch.setattr(boot_anima, "RCLONE_CONF", missing_rc)
    with pytest.raises(FileNotFoundError, match="个人增强模式"):
        boot_anima.pull_loras_from_gdrive()

    monkeypatch.setattr(boot_anima, "RCLONE_CONF", present_rc)
    monkeypatch.setattr(boot_anima, "REQUIRED_LORAS", {"sample_char.safetensors": 1234})
    with pytest.raises(FileNotFoundError, match="主力 LoRA 缺失"):
        boot_anima.pull_loras_from_gdrive()
    monkeypatch.delenv("ANIMA_STRICT_LORAS", raising=False)

    # 3c. colab_bootstrap_drama3.py：删除个人不雅音频名硬编码，优先 DRAMA3_DEFAULT_REF_WAV 否则 wavs[0]
    d3_src = (PKG_DIR / "colab_bootstrap_drama3.py").read_text(encoding="utf-8")
    assert "我操" not in d3_src
    d3_comfy = tmp / "d3_comfy"
    d3_out = tmp / "d3_out"
    (d3_comfy / "input").mkdir(parents=True, exist_ok=True)
    (d3_comfy / "input" / "a_first.wav").write_bytes(b"RIFF")
    (d3_comfy / "input" / "b_preferred.wav").write_bytes(b"RIFF")
    monkeypatch.setattr(boot_drama3, "COMFY_DIR", d3_comfy)
    monkeypatch.setattr(boot_drama3, "COMFY_OUT", d3_out)
    monkeypatch.setattr(boot_drama3, "RCLONE_CONF", missing_rc)

    # 未设环境变量 -> 自动选排序第一个 a_first.wav
    boot_drama3.install_workflows_and_refs()
    wf_path = d3_comfy / "user" / "default" / "workflows" / "drama3_voice_clone.json"
    ref_n1 = next(n for n in json.loads(wf_path.read_text(encoding="utf-8"))["nodes"] if n.get("type") == "Drama3LoadReferenceAudio")
    assert ref_n1["widgets_values"][0] == "a_first.wav"
    assert ref_n1["widgets_values_named"]["audio"] == "a_first.wav"

    # 设置 DRAMA3_DEFAULT_REF_WAV -> 优先选 b_preferred.wav
    monkeypatch.setenv("DRAMA3_DEFAULT_REF_WAV", "b_preferred.wav")
    boot_drama3.install_workflows_and_refs()
    ref_n2 = next(n for n in json.loads(wf_path.read_text(encoding="utf-8"))["nodes"] if n.get("type") == "Drama3LoadReferenceAudio")
    assert ref_n2["widgets_values"][0] == "b_preferred.wav"

    # DRAMA3_DEFAULT_REF_WAV 指定的文件不存在于 wavs 时自动回退 wavs[0]
    monkeypatch.setenv("DRAMA3_DEFAULT_REF_WAV", "nonexistent.wav")
    boot_drama3.install_workflows_and_refs()
    ref_n3 = next(n for n in json.loads(wf_path.read_text(encoding="utf-8"))["nodes"] if n.get("type") == "Drama3LoadReferenceAudio")
    assert ref_n3["widgets_values"][0] == "a_first.wav"
    monkeypatch.delenv("DRAMA3_DEFAULT_REF_WAV", raising=False)

    # 3d. verify_smoke.py：缺私训 LoRA 自动走纯底座图，存在私训 LoRA 保留完整 LoRA 链路，支持 ANIMA_SMOKE_MODE 控制
    smoke_comfy = tmp / "smoke_comfy"
    smoke_loras = smoke_comfy / "models" / "loras"
    smoke_loras.mkdir(parents=True, exist_ok=True)
    smoke_out = tmp / "smoke_out"
    monkeypatch.setattr(verify_smoke, "COMFY_DIR", smoke_comfy)
    monkeypatch.setattr(verify_smoke, "COMFY_OUT", smoke_out)
    monkeypatch.setattr(verify_smoke, "OUT_DIR", smoke_out / "smoke_verify")
    monkeypatch.setattr(verify_smoke, "LORA_CHAR", "sample_char.safetensors")
    monkeypatch.setattr(verify_smoke, "LORA_STYLE", "sample_style.safetensors")

    # 缺失私训 LoRA -> 自动 base 模式（无 LoraLoaderModelOnly，UNET 直接连 KSampler）
    assert verify_smoke.resolve_smoke_mode() == "base"
    base_graph, base_hint = verify_smoke.build_smoke_graph("base")
    assert not any(n.get("class_type") == "LoraLoaderModelOnly" for n in base_graph.values())
    assert base_graph["ks"]["inputs"]["model"] == ["u", 0]
    assert "公开基础版" in base_hint

    # 强制 ANIMA_SMOKE_MODE=personal 但缺失 LoRA -> 明确报错
    monkeypatch.setenv("ANIMA_SMOKE_MODE", "personal")
    with pytest.raises(FileNotFoundError, match="私训 LoRA"):
        verify_smoke.resolve_smoke_mode()
    monkeypatch.delenv("ANIMA_SMOKE_MODE", raising=False)

    # 端到端 mock verify_smoke.main() 跑通公开纯底座冒烟
    submitted_graphs: list[dict[str, Any]] = []
    out_img = smoke_out / "smoke_verify" / "smoke_00001_.png"
    out_img.parent.mkdir(parents=True, exist_ok=True)
    out_img.write_bytes(b"\x89PNG\r\n\x1a\nMOCK_IMAGE")

    def _fake_submit(_label: str, graph: dict[str, Any], **_kw):
        submitted_graphs.append(graph)
        return (
            {
                "outputs": {
                    "save": {"images": [{"filename": "smoke_00001_.png", "subfolder": "smoke_verify", "type": "output"}]}
                }
            },
            1.2,
        )

    monkeypatch.setattr(verify_smoke, "comfy_submit_and_wait", _fake_submit)
    out_buf = io.StringIO()
    with patch("sys.stdout", out_buf):
        verify_smoke.main()
    assert len(submitted_graphs) == 1
    assert submitted_graphs[0]["ks"]["inputs"]["model"] == ["u", 0]
    assert "公开基础版" in out_buf.getvalue()

    # 放入个人 LoRA 后自动切回 personal 完整 LoRA 冒烟链路；ANIMA_SMOKE_MODE=base 可显式强制纯底座
    (smoke_loras / verify_smoke.LORA_CHAR).write_bytes(b"LORA1")
    (smoke_loras / verify_smoke.LORA_STYLE).write_bytes(b"LORA2")
    assert verify_smoke.resolve_smoke_mode() == "personal"
    personal_graph, personal_hint = verify_smoke.build_smoke_graph("personal")
    assert personal_graph["lc"]["class_type"] == "LoraLoaderModelOnly"
    assert personal_graph["ls"]["class_type"] == "LoraLoaderModelOnly"
    assert personal_graph["ks"]["inputs"]["model"] == ["ls", 0]
    assert "LoRA" in personal_hint

    monkeypatch.setenv("ANIMA_SMOKE_MODE", "base")
    assert verify_smoke.resolve_smoke_mode() == "base"
