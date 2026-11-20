"""S1a / S1b 公网鉴权强制拦截闸门与 S2 定向停卡隔离离线单元测试。

全量使用本地内存 HTTP 桩服务与 `tmp_path` 沙箱环境变量（COLAB_PET_MOCK_VAULT_PATH、
COLAB_PET_CF_CONFIG_PATH、COLAB_PET_DIAG_LOG_PATH），严禁执行真实 `colab stop` /
`colab_ctl.py` 外部停开机命令，严禁触碰宿主机真实凭据库或隧道配置文件。
"""

from __future__ import annotations

import contextlib
import http.server
import importlib
import json
import shutil
import subprocess
import sys
import threading
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

PKG_DIR = Path(__file__).resolve().parents[1]
if str(PKG_DIR) not in sys.path:
    sys.path.insert(0, str(PKG_DIR))

boot_llm = importlib.import_module("colab_bootstrap_qwen_llm")
rcom = importlib.import_module("colab_remote_common")
colab_client = importlib.import_module("colab_core.colab_client")


@pytest.fixture()
def isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """为每条用例创建隔离的凭据、隧道、日志与远端路径沙箱。"""
    vault_file = tmp_path / "mock_vault.json"
    cf_file = tmp_path / ".cf_tunnel_secret.json"
    diag_log = tmp_path / "logs" / "colab_diag.log"
    monkeypatch.setenv("COLAB_PET_MOCK_VAULT_PATH", str(vault_file))
    monkeypatch.setenv("COLAB_PET_CF_CONFIG_PATH", str(cf_file))
    monkeypatch.setenv("COLAB_PET_DIAG_LOG_PATH", str(diag_log))
    monkeypatch.setenv("COMFY_AUTH_KEY_FILE", str(tmp_path / "comfy_auth_key"))
    return SimpleNamespace(
        tmp=tmp_path,
        vault_file=vault_file,
        cf_file=cf_file,
        diag_log=diag_log,
    )


class _FakeProc:
    """模拟后台子进程对象，记录 terminate / kill 调用状态。"""

    def __init__(self, pid: int = 42424) -> None:
        self.pid = pid
        self.returncode: int | None = None
        self.terminated = False
        self.killed = False

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = -15

    def wait(self, timeout: float | None = None) -> int:
        _ = timeout
        if self.returncode is None:
            self.returncode = 0
        return self.returncode

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9


@contextlib.contextmanager
def _start_stub_http_server(handler_cls: type[http.server.BaseHTTPRequestHandler]):
    """启动绑定到 127.0.0.1 临时端口的本地 HTTP 桩服务。"""
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


def _make_comfy_stub_handler(state: dict[str, Any]) -> type[http.server.BaseHTTPRequestHandler]:
    """构造可切换是否挂载 v3 公网鉴权网关的 ComfyUI 模拟 HTTP 处理器。"""

    class _ComfyStubHandler(http.server.BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            _ = (format, args)

        def _send_json(self, status: int, payload: dict[str, Any]) -> None:
            raw = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:
            if not state.get("alive", True):
                self.send_error(503, "down")
                return
            if self.path == "/colab_pet/ping":
                ver = int(state.get("bridge_version", 3))
                self._send_json(200, {"ok": True, "version": ver})
                return
            if self.path == "/system_stats":
                has_proxy_hdr = any(
                    self.headers.get(h)
                    for h in ("X-Forwarded-For", "Cf-Connecting-Ip", "Forwarded")
                )
                if state.get("auth_enforced", False) and has_proxy_hdr:
                    expected = f"Bearer {state.get('auth_key', '')}"
                    if self.headers.get("Authorization") != expected:
                        status = int(state.get("reject_status", 401))
                        self._send_json(status, {"error": "unauthorized"})
                        return
                self._send_json(200, {"system": {"os": "posix"}, "devices": []})
                return
            self._send_json(404, {"error": "not_found"})

    return _ComfyStubHandler


def _make_llm_stub_handler(state: dict[str, Any]) -> type[http.server.BaseHTTPRequestHandler]:
    """构造可切换 API Key 鉴权是否生效的 LLM 网关模拟 HTTP 处理器。"""

    class _LlmStubHandler(http.server.BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            _ = (format, args)

        def _send_json(self, status: int, payload: dict[str, Any]) -> None:
            raw = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:
            if self.path == "/health":
                self._send_json(200, {"status": "ok"})
                return
            if self.path == "/v1/models":
                if state.get("auth_enforced", False):
                    expected = f"Bearer {state.get('api_key', '')}"
                    if self.headers.get("Authorization") != expected:
                        status = int(state.get("reject_status", 401))
                        self._send_json(status, {"error": "invalid_api_key"})
                        return
                self._send_json(200, {"data": [{"id": "Qwen3.8-27B-abliterated"}]})
                return
            self._send_json(404, {"error": "not_found"})

    return _LlmStubHandler


# =========================================================================
# 1. S1a：ComfyUI 公网鉴权网关强制校验与拦截测试
# =========================================================================


def test_s1a_comfy_auth_gate_blocks_unprotected_and_allows_v3(
    isolated_env: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 ComfyUI 未生效鉴权时双处拦截且绝不启动 cloudflared；v3 鉴权生效时正常放行。"""
    tmp = isolated_env.tmp
    comfy_dir = tmp / "ComfyUI"
    comfy_dir.mkdir(parents=True, exist_ok=True)
    auth_key = "cpk-test-secret-comfy-123456"
    key_file = tmp / "comfy_auth_key"
    key_file.write_text(auth_key + "\n", encoding="utf-8")

    monkeypatch.setattr(rcom, "COMFY_DIR", comfy_dir)
    monkeypatch.setattr(rcom, "COMFY_OUT", tmp / "comfy_out")
    monkeypatch.setattr(rcom, "COMFY_LOG", tmp / "comfy.log")
    monkeypatch.setattr(rcom, "AUTH_KEY_FILE_DEFAULT", key_file)
    monkeypatch.setattr(rcom, "SERVICE_META_FILE", tmp / "service_meta.json")
    monkeypatch.setattr(rcom, "TUNNEL_URL_FILE", tmp / "comfy_tunnel_url.txt")
    monkeypatch.setattr(rcom.time, "sleep", lambda *_a: None)

    stub_state: dict[str, Any] = {
        "alive": True,
        "bridge_version": 3,
        "auth_enforced": False,
        "auth_key": auth_key,
        "reject_status": 401,
    }

    with _start_stub_http_server(_make_comfy_stub_handler(stub_state)) as port:
        monkeypatch.setattr(rcom, "COMFY_API", f"http://127.0.0.1:{port}")

        # 1a-1. /system_stats 对伪造 X-Forwarded-For 返回 200 时，校验必须失败
        ok, reason = rcom.verify_local_comfy_auth_enforced(auth_key)
        assert ok is False
        assert "200" in reason

        # 1a-2. start_comfy_tunnel_standard 入口拦截，绝不能拉起 cloudflared
        tunnel_called = False

        def _forbid_tunnel(**_kw: Any) -> str:
            nonlocal tunnel_called
            tunnel_called = True
            return "https://should-not-start.trycloudflare.com"

        monkeypatch.setattr(rcom, "start_cloudflared_tunnel", _forbid_tunnel)
        with pytest.raises(RuntimeError, match="ComfyUI 公网鉴权网关未生效，拒绝暴露公网隧道"):
            rcom.start_comfy_tunnel_standard()
        assert tunnel_called is False

        # 1a-3. start_comfyui_and_wait：存量不放行，重启拉起新 proc 后连续 3 次仍未生效时终止 proc 并报错
        pkill_cmds: list[list[str]] = []
        spawned_procs: list[_FakeProc] = []

        def _fake_run(cmd: list[Any], **_kw: Any) -> subprocess.CompletedProcess[str]:
            cmd_s = [str(x) for x in cmd]
            pkill_cmds.append(cmd_s)
            if cmd_s and cmd_s[0] == "pkill":
                # 模拟 pkill 后短暂离线再被新拉起
                stub_state["alive"] = False
            return subprocess.CompletedProcess(cmd_s, 0, stdout="", stderr="")

        def _fake_popen(*_a: Any, **_kw: Any) -> _FakeProc:
            stub_state["alive"] = True
            proc = _FakeProc(pid=51001)
            spawned_procs.append(proc)
            return proc

        monkeypatch.setattr(rcom.subprocess, "run", _fake_run)
        monkeypatch.setattr(rcom.subprocess, "Popen", _fake_popen)

        with pytest.raises(RuntimeError, match="ComfyUI 公网鉴权网关未生效，拒绝暴露公网隧道"):
            rcom.start_comfyui_and_wait(wait_timeout_sec=12)

        assert any("pkill" in c for c in pkill_cmds), "存量鉴权未生效必须触发平滑重启"
        assert len(spawned_procs) == 1 and spawned_procs[0].terminated is True

        # 1a-3b. 短超时窗口 (< 3 次轮询) 下 /system_stats 已就绪但鉴权仍失败同样终止 proc 并抛 RuntimeError
        spawned_procs.clear()
        with pytest.raises(RuntimeError, match="ComfyUI 公网鉴权网关未生效，拒绝暴露公网隧道"):
            rcom.start_comfyui_and_wait(wait_timeout_sec=4)
        assert len(spawned_procs) == 1 and spawned_procs[0].terminated is True

        # 1a-4. 当 bridge_version < 3 时也必须拒绝放行
        stub_state["auth_enforced"] = True
        stub_state["bridge_version"] = 2
        ok_v2, reason_v2 = rcom.verify_local_comfy_auth_enforced(auth_key)
        assert ok_v2 is False and "version=2" in reason_v2
        with pytest.raises(RuntimeError, match="ComfyUI 公网鉴权网关未生效"):
            rcom.start_comfy_tunnel_standard()
        assert tunnel_called is False

        # 1a-5. 挂载完整 v3 鉴权响应（无口令 401/403，带口令 200）时正常放行；错误密钥必须被拒
        stub_state["bridge_version"] = 3
        stub_state["auth_enforced"] = True
        stub_state["reject_status"] = 401
        ok_wrong, _ = rcom.verify_local_comfy_auth_enforced("cpk-wrong-key")
        assert ok_wrong is False

        ok_default_key, _ = rcom.verify_local_comfy_auth_enforced()
        assert ok_default_key is True

        stub_state["reject_status"] = 403
        ok_403, _ = rcom.verify_local_comfy_auth_enforced(auth_key)
        assert ok_403 is True

        # 复用分支与隧道启动均放行
        pkill_cmds.clear()
        rcom.start_comfyui_and_wait(wait_timeout_sec=10)
        assert pkill_cmds == [], "鉴权已生效的存量 v3 服务应直接复用，不应重启"

        monkeypatch.setattr(
            rcom,
            "start_cloudflared_tunnel",
            lambda **_kw: "https://verified-comfy.trycloudflare.com",
        )
        got_url = rcom.start_comfy_tunnel_standard()
        assert got_url == "https://verified-comfy.trycloudflare.com"


# =========================================================================
# 2. S1b：Qwen LLM 网关 API Key 鉴权强制校验与拦截测试
# =========================================================================


def test_s1b_llm_gateway_auth_gate_blocks_unprotected_and_allows_valid_key(
    isolated_env: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 Qwen LLM 网关无口令返回 200 时双处拦截并清理子进程；401+Bearer 200 时放行。"""
    tmp = isolated_env.tmp
    api_key = "sk-pet-test-llm-secret-987654"
    gw_script = tmp / "colab_sse_gateway.py"
    gw_script.write_text("# stub gateway\n", encoding="utf-8")

    monkeypatch.setattr(boot_llm, "LLAMA_LOG", tmp / "llama_server.log")
    monkeypatch.setattr(boot_llm, "SSE_GATEWAY_FILE", tmp / "llm_sse_gateway.py")
    monkeypatch.setattr(boot_llm, "SSE_GATEWAY_LOG", tmp / "llm_sse_gateway.log")
    monkeypatch.setattr(boot_llm, "SERVICE_META_FILE", tmp / "colab_service_meta.json")
    monkeypatch.setattr(boot_llm, "TUNNEL_URL_FILE", tmp / "tunnel_url.txt")
    monkeypatch.setattr(boot_llm, "LLM_TUNNEL_URL_FILE", tmp / "llm_tunnel_url.txt")
    monkeypatch.setattr(boot_llm, "load_companion_text", lambda _name: "# stub sse gw")
    monkeypatch.setattr(boot_llm, "_query_steady_vram", lambda: ("L4", 12.0, 24.0))
    monkeypatch.setattr(boot_llm.time, "sleep", lambda *_a: None)

    stub_state: dict[str, Any] = {
        "auth_enforced": False,
        "api_key": api_key,
        "reject_status": 401,
    }

    with _start_stub_http_server(_make_llm_stub_handler(stub_state)) as port:
        monkeypatch.setattr(boot_llm, "LLAMA_PORT", port)
        spawned: list[_FakeProc] = []
        monkeypatch.setattr(boot_llm.shutil, "which", lambda *_a: None)
        monkeypatch.setattr(
            boot_llm.subprocess,
            "run",
            lambda cmd, **_kw: subprocess.CompletedProcess([str(x) for x in cmd], 0, "", ""),
        )

        # 2a-1. 空 key 或 /v1/models 无口令返回 200 均必须失败
        ok_empty, _ = boot_llm._verify_llm_gateway_auth_enforced(port, "")
        assert ok_empty is False
        ok_open, reason_open = boot_llm._verify_llm_gateway_auth_enforced(port, api_key)
        assert ok_open is False and "200" in reason_open

        # 2a-2. start_tunnel_and_write_contract 在校验失败时拒绝开隧道（且已在调用前隔离 subprocess.run/pkill）
        tunnel_invoked = False

        def _forbid_cf(**_kw: Any) -> str:
            nonlocal tunnel_invoked
            tunnel_invoked = True
            return "https://should-not-open.trycloudflare.com"

        monkeypatch.setattr(boot_llm, "start_cloudflared_tunnel", _forbid_cf)
        monkeypatch.setattr(boot_llm, "_probe_local_comfy_ready", lambda timeout=2.0: False)
        with pytest.raises(RuntimeError, match="LLM 网关 API Key 鉴权未生效，拒绝暴露公网隧道"):
            boot_llm.start_tunnel_and_write_contract("UD-Q4_K_XL", api_key, "L4", 12.0, 24.0)
        assert tunnel_invoked is False

        # 2a-3. start_llama_and_wait：/health 通过但 /v1/models 裸露 200 时，清理子进程并抛 RuntimeError
        monkeypatch.setattr(
            boot_llm,
            "_probe_http_health",
            lambda p, timeout=3.0: p in (boot_llm.LLAMA_INTERNAL_PORT, port),
        )

        def _fake_popen(*_a: Any, **_kw: Any) -> _FakeProc:
            proc = _FakeProc(pid=62000 + len(spawned))
            spawned.append(proc)
            return proc

        monkeypatch.setattr(boot_llm.subprocess, "Popen", _fake_popen)
        with pytest.raises(RuntimeError, match="LLM 网关 API Key 鉴权未生效，拒绝暴露公网隧道"):
            boot_llm.start_llama_and_wait(["llama-server"], "", "UD-Q4_K_XL", api_key)
        assert len(spawned) == 2
        assert all(p.terminated is True for p in spawned), "鉴权失败必须清理 llama-server 与 SSE 网关子进程"

        # 2a-3b. 存量配置一致时若鉴权未生效，严禁直接复用，必须进入重启并拦截
        (tmp / "colab_service_meta.json").write_text(
            json.dumps({
                "kind": "llm",
                "model_variant": boot_llm.MODEL_VARIANT,
                "model": boot_llm.MODEL_ALIAS,
                "quant": "UD-Q4_K_XL",
                "ctx_size": boot_llm.CTX_SIZE,
                "mmproj_enabled": boot_llm.WITH_MMPROJ,
                "webui_enabled": boot_llm.WITH_WEBUI,
                "sse_keepalive": True,
                "kv_unified": True,
                "n_parallel": 2,
                "default_effort": boot_llm.DEFAULT_EFFORT,
                "api_key": api_key,
            }),
            encoding="utf-8",
        )
        spawned.clear()
        with pytest.raises(RuntimeError, match="LLM 网关 API Key 鉴权未生效，拒绝暴露公网隧道"):
            boot_llm.start_llama_and_wait(["llama-server"], "", "UD-Q4_K_XL", api_key)
        assert len(spawned) == 2

        # 2b. 鉴权生效（无口令 401/403，带 Bearer 200）时通过并正常放行隧道
        stub_state["auth_enforced"] = True
        ok_good, _ = boot_llm._verify_llm_gateway_auth_enforced(port, api_key)
        assert ok_good is True
        stub_state["reject_status"] = 403
        ok_good_403, _ = boot_llm._verify_llm_gateway_auth_enforced(port, api_key)
        assert ok_good_403 is True
        stub_state["reject_status"] = 401

        spawned.clear()
        gpu, used_g, tot_g = boot_llm.start_llama_and_wait(
            ["llama-server"],
            "",
            "UD-Q4_K_XL",
            api_key,
        )
        assert spawned == [], "配置一致且鉴权已生效时存量服务应直接复用"
        assert (gpu, used_g, tot_g) == ("L4", 12.0, 24.0)

        monkeypatch.setattr(
            boot_llm,
            "start_cloudflared_tunnel",
            lambda **_kw: "https://verified-llm.trycloudflare.com",
        )
        llm_url = boot_llm.start_tunnel_and_write_contract("UD-Q4_K_XL", api_key, gpu, used_g, tot_g)
        assert llm_url == "https://verified-llm.trycloudflare.com"


# =========================================================================
# 3. S2：显式指定 target_endpoint 时严禁按旧会话名盲停并精准清理 store
# =========================================================================


def _exec_colab_py_snippet(
    code: str,
    store_items: dict[str, Any],
    active_eps: list[str],
    unassigned: list[str],
) -> subprocess.CompletedProcess[str]:
    """在内存隔离的 fake `colab_cli.common.state` 沙箱中真实执行内联脚本。"""
    import io

    class _FakeStore:
        def list(self) -> dict[str, Any]:
            return dict(store_items)

        def remove(self, key: str) -> None:
            store_items.pop(key, None)

    class _FakeClient:
        def list_assignments(self) -> list[SimpleNamespace]:
            return [SimpleNamespace(endpoint=ep) for ep in active_eps]

        def unassign(self, ep: str) -> None:
            unassigned.append(ep)
            if ep in active_eps:
                active_eps.remove(ep)

    fake_state = SimpleNamespace(
        store=_FakeStore(),
        client=_FakeClient(),
        get_session=lambda name, ignore_missing_session=True: store_items.get(name),
    )

    colab_cli_pkg = types.ModuleType("colab_cli")
    colab_cli_common = types.ModuleType("colab_cli.common")
    colab_cli_common.state = fake_state
    colab_cli_pkg.common = colab_cli_common

    saved = {
        name: sys.modules.get(name)
        for name in ("colab_cli", "colab_cli.common")
    }
    sys.modules["colab_cli"] = colab_cli_pkg
    sys.modules["colab_cli.common"] = colab_cli_common
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(compile(code, "<colab_inline_test>", "exec"), {"__name__": "__main__"})  # noqa: S102
    finally:
        for name, mod in saved.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod
    return subprocess.CompletedProcess(["colab_py", "-c", code], 0, stdout=buf.getvalue(), stderr="")


def test_s2_stop_session_explicit_target_endpoint_skips_blind_stop_and_preserves_other(
    isolated_env: SimpleNamespace,
) -> None:
    """验证显式传 target_endpoint 时跳过 colab stop -s，仅精准 unassign 并保留其他映射。"""
    tmp = isolated_env.tmp
    fake_exe = tmp / "colab.exe"
    fake_py = tmp / "colab_py.exe"
    fake_exe.write_text("", encoding="utf-8")
    fake_py.write_text("", encoding="utf-8")

    store_items: dict[str, Any] = {
        "comfy": SimpleNamespace(endpoint="ep-other"),
        "target_sess": SimpleNamespace(endpoint="ep-target"),
        "dict_target": {"endpoint": "ep-target"},
        "dict_keep": {"endpoint": "ep-other"},
        "str_target": "ep-target",
        "str_keep": "ep-other",
    }
    active_eps = ["ep-other", "ep-target"]
    unassigned: list[str] = []
    exe_stop_calls: list[list[str]] = []
    probe_calls: list[str] = []

    def _mock_run(cmd: list[Any], check: bool = False, capture: bool = False, timeout: float | None = None):
        _ = (check, capture, timeout)
        cmd_s = [str(x) for x in cmd]
        if cmd_s[0] == str(fake_exe):
            exe_stop_calls.append(cmd_s)
            return subprocess.CompletedProcess(cmd_s, 0, stdout="should-not-run", stderr="")
        if cmd_s[0] == str(fake_py) and len(cmd_s) >= 3 and cmd_s[1] == "-c":
            code = cmd_s[2]
            if "get_session" in code:
                probe_calls.append(code)
            return _exec_colab_py_snippet(code, store_items, active_eps, unassigned)
        return subprocess.CompletedProcess(cmd_s, 0, stdout="", stderr="")

    summary = colab_client.stop_session_cli(
        session="comfy",
        clean_all=False,
        target_endpoint="ep-target",
        verify_release=True,
        colab_exe=fake_exe,
        colab_py=fake_py,
        run_fn=_mock_run,
    )

    assert exe_stop_calls == [], f"显式传 target_endpoint 时严禁先调用 colab.exe stop -s: {exe_stop_calls}"
    assert probe_calls == [], "显式传 target_endpoint 时不应再盲查旧会话 Endpoint"
    assert unassigned == ["ep-target"], f"只能精准 unassign ep-target: {unassigned}"
    assert active_eps == ["ep-other"], f"同账号另一台活跃实例 ep-other 必须保留: {active_eps}"
    assert "comfy" in store_items and store_items["comfy"].endpoint == "ep-other"
    assert "dict_keep" in store_items and "str_keep" in store_items
    assert "target_sess" not in store_items
    assert "dict_target" not in store_items
    assert "str_target" not in store_items
    assert summary["ok"] is True
    assert summary["released_endpoint"] == "ep-target"
    assert summary["remaining_count"] == 1


def test_s2_stop_fallback_cleans_only_empty_endpoint_for_matching_session_and_rejects_missing_colab_py(
    isolated_env: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证无 endpoint 字段仅兜底删除当前 session_name，且缺失 colab_py 时严格 Fail-Closed 0 次停卡。"""
    tmp = isolated_env.tmp
    fake_exe = tmp / "colab.exe"
    fake_py = tmp / "colab_py.exe"
    fake_exe.write_text("", encoding="utf-8")
    fake_py.write_text("", encoding="utf-8")
    monkeypatch.setattr(colab_client, "CLI_BRIDGE_SCRIPT", PKG_DIR / "colab_cli_bridge.py")

    # 3b-1. 当前 session_name 无 endpoint 字段时才兜底移除，其他无 endpoint 记录不误删
    store_items: dict[str, Any] = {
        "comfy": {"accelerator": "L4"},
        "other_no_ep": {"accelerator": "T4"},
        "other_bound": SimpleNamespace(endpoint="ep-other"),
    }
    active_eps = ["ep-target", "ep-other"]
    unassigned: list[str] = []

    def _mock_run_no_ep(cmd: list[Any], check: bool = False, capture: bool = False, timeout: float | None = None):
        _ = (check, capture, timeout)
        cmd_s = [str(x) for x in cmd]
        if cmd_s[0] == str(fake_exe):
            raise AssertionError("具备 colab_py 且显式传 target_endpoint 时绝不应调用 colab.exe stop")
        return _exec_colab_py_snippet(cmd_s[2], store_items, active_eps, unassigned)

    summary = colab_client.stop_session_cli(
        session="comfy",
        clean_all=False,
        target_endpoint="ep-target",
        verify_release=True,
        colab_exe=fake_exe,
        colab_py=fake_py,
        run_fn=_mock_run_no_ep,
    )
    assert summary["ok"] is True
    assert "comfy" not in store_items
    assert "other_no_ep" in store_items
    assert "other_bound" in store_items

    # 3b-2. 缺失 colab_py 时，严禁回退 colab stop -s 或跳过释放核验：停卡调用次数严格为 0
    missing_py = tmp / "missing_colab_py.exe"
    call_count = {"n": 0}

    def _forbid_run(*_a: Any, **_kw: Any):
        call_count["n"] += 1
        return subprocess.CompletedProcess([], 0, stdout="", stderr="")

    with pytest.raises(FileNotFoundError, match="拒绝停卡"):
        colab_client.stop_session_cli(
            session="comfy",
            clean_all=False,
            target_endpoint="ep-target",
            verify_release=False,
            colab_exe=fake_exe,
            colab_py=missing_py,
            run_fn=_forbid_run,
        )
    assert call_count["n"] == 0

    with pytest.raises(FileNotFoundError, match="拒绝停卡"):
        colab_client.stop_session_cli(
            session="comfy",
            clean_all=False,
            target_endpoint=None,
            verify_release=True,
            colab_exe=fake_exe,
            colab_py=missing_py,
            run_fn=_forbid_run,
        )
    assert call_count["n"] == 0


def test_s1_start_cloudflared_tunnel_gate_blocks_cli_restart_and_auto_recover(
    isolated_env: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证鉴权闸门已下沉进 start_cloudflared_tunnel，拦截 CLI 重启/桌宠恢复等全部隧道发布入口。"""
    tmp = isolated_env.tmp
    comfy_key = "cpk-tunnel-gate-secret-1122"
    llm_key = "sk-pet-tunnel-gate-llm-3344"
    key_file = tmp / "comfy_auth_key"
    key_file.write_text(comfy_key + "\n", encoding="utf-8")
    llm_key_file = tmp / "llama_api_key"
    llm_key_file.write_text(llm_key + "\n", encoding="utf-8")
    cf_bin = tmp / "cloudflared"
    cf_bin.write_text("", encoding="utf-8")

    monkeypatch.setattr(rcom, "AUTH_KEY_FILE_DEFAULT", key_file)
    monkeypatch.setattr(rcom, "LLAMA_API_KEY_FILE", llm_key_file)
    monkeypatch.setattr(rcom, "SERVICE_META_FILE", tmp / "colab_service_meta.json")
    monkeypatch.setattr(rcom, "TUNNEL_URL_FILE", tmp / "comfy_tunnel_url.txt")
    monkeypatch.setattr(rcom, "LLM_TUNNEL_URL_FILE", tmp / "llm_tunnel_url.txt")
    monkeypatch.setattr(rcom, "CLOUDFLARED_LOG", tmp / "cloudflared.log")
    monkeypatch.setattr(rcom, "CLOUDFLARED_BIN", cf_bin)
    monkeypatch.setattr(rcom.time, "sleep", lambda *_a: None)
    monkeypatch.setattr(shutil, "which", lambda name: str(cf_bin) if name == "cloudflared" else None)

    pkill_calls: list[list[str]] = []
    popen_calls: list[list[str]] = []

    def _mock_run(cmd: list[Any], **_kw: Any) -> subprocess.CompletedProcess[str]:
        cmd_s = [str(x) for x in cmd]
        if cmd_s and cmd_s[0] == "pkill":
            pkill_calls.append(cmd_s)
        if cmd_s and cmd_s[0] == "pgrep":
            return subprocess.CompletedProcess(cmd_s, 1, stdout="", stderr="")
        return subprocess.CompletedProcess(cmd_s, 0, stdout="", stderr="")

    def _forbid_popen(cmd: list[Any], **_kw: Any) -> _FakeProc:
        popen_calls.append([str(x) for x in cmd])
        return _FakeProc(pid=55555)

    monkeypatch.setattr(rcom.subprocess, "run", _mock_run)
    monkeypatch.setattr(rcom.subprocess, "Popen", _forbid_popen)

    comfy_state: dict[str, Any] = {
        "alive": True,
        "bridge_version": 3,
        "auth_enforced": False,
        "auth_key": comfy_key,
        "reject_status": 401,
    }
    llm_state: dict[str, Any] = {
        "auth_enforced": False,
        "api_key": llm_key,
        "reject_status": 401,
    }

    with (
        _start_stub_http_server(_make_comfy_stub_handler(comfy_state)) as comfy_port,
        _start_stub_http_server(_make_llm_stub_handler(llm_state)) as llm_port,
    ):
        monkeypatch.setattr(rcom, "COMFY_API", f"http://127.0.0.1:{comfy_port}")

        # ComfyUI 未生效鉴权时，直接调 start_cloudflared_tunnel（模拟 colab tunnel --restart）必拦截
        with pytest.raises(RuntimeError, match="ComfyUI 公网鉴权网关未生效"):
            rcom.start_cloudflared_tunnel(
                local_port=comfy_port,
                service_kind="comfy",
                force_restart=True,
                auth_key=comfy_key,
            )
        assert popen_calls == []
        assert len(pkill_calls) == 1

        # LLM 未生效鉴权时，直接调 start_cloudflared_tunnel(service_kind="llm") 必拦截
        pkill_calls.clear()
        with pytest.raises(RuntimeError, match="LLM 网关 API Key 鉴权未生效"):
            rcom.start_cloudflared_tunnel(
                local_port=llm_port,
                service_kind="llm",
                force_restart=True,
                auth_key=llm_key,
            )
        assert popen_calls == []
        assert len(pkill_calls) == 1

        # 双栖命名隧道场景：若 LLM 已加密但 ComfyUI 未加密，同样整条拒绝
        llm_state["auth_enforced"] = True
        with pytest.raises(RuntimeError, match="ComfyUI 公网鉴权网关未生效"):
            rcom.start_cloudflared_tunnel(
                local_port=llm_port,
                service_kind="llm",
                cf_token="cf-tok-dummy",
                cf_url="https://llm.example.com",
                cohost_other=True,
                auth_key=llm_key,
            )
        assert popen_calls == []


def test_s2_fail_closed_when_account_bridge_missing_or_session_unresolved(
    isolated_env: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证缺少多账号隔离桥接组件或目标会话无法锁定时，停卡操作严格 Fail-Closed（调用次数为 0）。"""
    tmp = isolated_env.tmp
    fake_exe = tmp / "colab.exe"
    fake_py = tmp / "colab_py.exe"
    fake_exe.write_text("", encoding="utf-8")
    fake_py.write_text("", encoding="utf-8")

    # 1) 指定 account_id 但缺少 colab_cli_bridge.py 时，stop_session_cli / build_scoped_colab_cmd 均 0 调用拒绝
    monkeypatch.setattr(colab_client, "CLI_BRIDGE_SCRIPT", tmp / "missing_bridge.py")
    stop_calls = {"n": 0}

    def _forbid_stop(*_a: Any, **_kw: Any):
        stop_calls["n"] += 1
        return subprocess.CompletedProcess([], 0, "", "")

    with pytest.raises(FileNotFoundError, match="拒绝在未隔离状态下执行"):
        colab_client.build_scoped_colab_cmd(fake_exe, fake_py, ["sessions"], account_id="acct_a")
    with pytest.raises(FileNotFoundError, match="拒绝停卡"):
        colab_client.stop_session_cli(
            session="comfy",
            clean_all=False,
            target_endpoint="ep-target",
            verify_release=True,
            colab_exe=fake_exe,
            colab_py=fake_py,
            run_fn=_forbid_stop,
            account_id="acct_a",
        )
    assert stop_calls["n"] == 0

    # 2) resolve_target_binding 找不到指定会话名时绝不跨会话选中另一名字的实例
    sname, ep, is_orphan = colab_client.resolve_target_binding(
        sessions=[{"name": "other", "endpoint": "ep-other", "orphan": False}],
        binding={"account_id": "acct1", "session": "other", "endpoint": "ep-other"},
        active_account_id="acct1",
        session="comfy",
    )
    assert sname == "comfy" and ep == "" and is_orphan is False

    # 3) 桌宠 backend.stop_session_instance("comfy") 遇到仅存在其他会话 "other" 时，停卡次数为 0
    pet_dir = PKG_DIR / "colab_pet"
    if str(pet_dir) not in sys.path:
        sys.path.insert(0, str(pet_dir))
    cb = importlib.import_module("colab_backend")
    backend = cb.ColabPetBackend(
        state_path=tmp / "pet_fail_closed.json",
        download_root=tmp / "dl_fail_closed",
        cf_config_path=isolated_env.cf_file,
    )
    with backend._lock:
        backend.state["last_assignments_count"] = 1
        backend.state["last_sessions"] = [
            {"name": "other", "endpoint": "ep-other", "accelerator": "T4", "orphan": False}
        ]

    hidden_calls: list[list[str]] = []
    monkeypatch.setattr(
        cb,
        "_run_hidden",
        lambda cmd, **_kw: (hidden_calls.append([str(x) for x in cmd]), subprocess.CompletedProcess([], 0, "", ""))[1],
    )
    monkeypatch.setattr(
        backend,
        "fetch_live_status",
        lambda: {
            "ok": True,
            "assignments_count": 1,
            "sessions": [{"name": "other", "endpoint": "ep-other", "accelerator": "T4", "orphan": False}],
        },
    )
    res = backend.stop_session_instance("comfy")
    assert res["ok"] is False and res.get("target_unresolved") is True
    assert hidden_calls == []


def test_exec_sandbox_without_dunder_file_imports_cleanly(isolated_env: SimpleNamespace) -> None:
    """验证 colab_remote_common 与 colab_bootstrap_qwen_llm 在无 __file__ 的 exec() 沙箱可执行。"""
    _ = isolated_env
    saved_rcm = sys.modules.pop("colab_remote_common", None)
    try:
        for filename in ("colab_remote_common.py", "colab_bootstrap_qwen_llm.py"):
            src = (PKG_DIR / filename).read_text(encoding="utf-8")
            ns: dict[str, Any] = {"__name__": "__sandbox_test__"}
            exec(compile(src, filename, "exec"), ns)  # noqa: S102
            if filename == "colab_remote_common.py":
                assert callable(ns["verify_local_comfy_auth_enforced"])
                sys.modules["colab_remote_common"] = types.ModuleType("colab_remote_common")
                sys.modules["colab_remote_common"].__dict__.update(ns)
            else:
                assert callable(ns["_verify_llm_gateway_auth_enforced"])
    finally:
        sys.modules.pop("colab_remote_common", None)
        if saved_rcm is not None:
            sys.modules["colab_remote_common"] = saved_rcm


def test_dual_cold_boot_defers_tunnel_checks_gateway_version_and_public_two_way_auth(
    isolated_env: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证双栖冷启动分阶段发布、SSE 网关热升级版本检查与双公网域双向鉴权验收。"""
    import argparse
    import re as _re
    import colab_ctl

    tmp = isolated_env.tmp
    llm_key = "sk-pet-dual-cold-boot-secret-123"
    comfy_key = "cpk-dual-cold-boot-secret-456"
    key_file = tmp / "comfy_auth_key"
    llm_key_file = tmp / "llama_api_key"
    meta_file = tmp / "colab_service_meta.json"
    comfy_url_file = tmp / "comfy_tunnel_url.txt"
    llm_url_file = tmp / "llm_tunnel_url.txt"
    gw_file = tmp / "llm_sse_gateway.py"
    cf_log = tmp / "cloudflared.log"
    cf_bin = tmp / ("cloudflared.exe" if sys.platform.startswith("win") else "cloudflared")
    cf_bin.write_text("", encoding="utf-8")
    key_file.write_text(comfy_key + "\n", encoding="utf-8")
    llm_key_file.write_text(llm_key + "\n", encoding="utf-8")

    # 1) cmd_up 双栖命名隧道阶段 1 必须注入 LLM_DEFER_TUNNEL=1，阶段 2 才带 finalize_service_contract
    fake_exe = tmp / "colab.exe"
    fake_py = tmp / "colab_py.exe"
    fake_rclone = tmp / "rclone.conf"
    fake_llm_boot = tmp / "fake_boot_llm.py"
    fake_mm_boot = tmp / "fake_boot_mm.py"
    for fp in (fake_exe, fake_py, fake_rclone, fake_llm_boot, fake_mm_boot):
        fp.write_text("# stub\n", encoding="utf-8")

    executed_bodies: list[str] = []
    monkeypatch.setattr(colab_ctl, "COLAB_EXE", fake_exe)
    monkeypatch.setattr(colab_ctl, "COLAB_PY", fake_py)
    monkeypatch.setattr(colab_ctl, "RCLONE_CONF", fake_rclone)
    monkeypatch.setattr(colab_ctl, "QWEN_LLM_BOOTSTRAP_SCRIPT", fake_llm_boot)
    monkeypatch.setattr(colab_ctl, "MINIMAX_BOOTSTRAP_SCRIPT", fake_mm_boot)
    monkeypatch.setattr(
        colab_ctl,
        "_load_cf_tunnel_secret",
        lambda: {
            "tunnel_token": "cf-tok-dual",
            "llm_url": "https://llm.fixed.example",
            "comfy_url": "https://comfy.fixed.example",
            "llm_api_key": llm_key,
        },
    )
    monkeypatch.setattr(colab_ctl, "run_preflight_checks", lambda **_kw: SimpleNamespace(ok=True))
    monkeypatch.setattr(colab_ctl, "_core_ensure_or_adopt_session", lambda **_kw: True)
    monkeypatch.setattr(colab_ctl, "_run", lambda *_a, **_kw: subprocess.CompletedProcess([], 0, "", ""))
    monkeypatch.setattr(
        colab_ctl,
        "_colab_exec_file",
        lambda _s, script_path, **_kw: (
            executed_bodies.append(Path(script_path).read_text(encoding="utf-8")),
            subprocess.CompletedProcess([], 0, "", ""),
        )[1],
    )
    up_ns = argparse.Namespace(
        session="comfy", endpoint=None, profile="minimax", profiles="minimax,qwen_27b",
        gpu="G4", high_mem=False, recreate=False, skip_preflight=True,
        minimax_lite=False, no_r2v=False, no_pe=False, llm_quant="UD-Q4_K_XL",
        llm_ctx=65536, no_mmproj=False, no_webui=False, gdrive_cache=False, llm_api_key="auto",
    )
    colab_ctl.cmd_up(up_ns)
    assert len(executed_bodies) == 2
    assert "os.environ['LLM_DEFER_TUNNEL'] = '1'" in executed_bodies[0]
    assert "LLM_DEFER_TUNNEL" not in executed_bodies[1]
    assert "finalize_service_contract" in executed_bodies[1]

    # 2) & 3) & 4) 空实例下阶段 1 延后发布、网关版本检查强制升级重启、阶段 2 双向公网验收
    local_comfy_state: dict[str, Any] = {"alive": False, "bridge_version": 3, "auth_enforced": True, "auth_key": comfy_key, "reject_status": 401}
    local_llm_state: dict[str, Any] = {"auth_enforced": True, "api_key": llm_key, "reject_status": 401}
    pub_comfy_state: dict[str, Any] = {"auth_enforced": True, "auth_key": comfy_key}
    pub_llm_state: dict[str, Any] = {"auth_enforced": True, "api_key": llm_key}

    def _make_public_proxy_handler(c_st: dict[str, Any], l_st: dict[str, Any]):
        class _PubProxy(http.server.BaseHTTPRequestHandler):
            def log_message(self, *_a: object) -> None:
                return

            def _json(self, code: int, payload: dict[str, Any]) -> None:
                raw = json.dumps(payload).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self) -> None:  # noqa: N802
                if self.path == "/system_stats":
                    if c_st.get("auth_enforced", True) and self.headers.get("Authorization") != f"Bearer {c_st['auth_key']}":
                        self._json(401, {"error": "unauthorized"})
                        return
                    self._json(200, {"system": {"os": "posix"}, "devices": []})
                    return
                if self.path == "/v1/models":
                    if l_st.get("auth_enforced", True) and self.headers.get("Authorization") != f"Bearer {l_st['api_key']}":
                        self._json(401, {"error": "unauthorized"})
                        return
                    self._json(200, {"data": [{"id": "Qwen3.8-27B-abliterated"}]})
                    return
                self._json(404, {"error": "not_found"})

        return _PubProxy

    with (
        _start_stub_http_server(_make_comfy_stub_handler(local_comfy_state)) as comfy_port,
        _start_stub_http_server(_make_llm_stub_handler(local_llm_state)) as llm_port,
        _start_stub_http_server(_make_public_proxy_handler(pub_comfy_state, pub_llm_state)) as pub_comfy_port,
        _start_stub_http_server(_make_public_proxy_handler(pub_comfy_state, pub_llm_state)) as pub_llm_port,
    ):
        pub_comfy_url = f"http://127.0.0.1:{pub_comfy_port}"
        pub_llm_url = f"http://127.0.0.1:{pub_llm_port}"
        loop_pat = r"http://127\.0\.0\.1:\d+"

        monkeypatch.setattr(rcom, "COMFY_API", f"http://127.0.0.1:{comfy_port}")
        monkeypatch.setattr(rcom, "AUTH_KEY_FILE_DEFAULT", key_file)
        monkeypatch.setattr(rcom, "LLAMA_API_KEY_FILE", llm_key_file)
        monkeypatch.setattr(rcom, "SERVICE_META_FILE", meta_file)
        monkeypatch.setattr(rcom, "TUNNEL_URL_FILE", comfy_url_file)
        monkeypatch.setattr(rcom, "LLM_TUNNEL_URL_FILE", llm_url_file)
        monkeypatch.setattr(rcom, "CLOUDFLARED_LOG", cf_log)
        monkeypatch.setattr(rcom, "CLOUDFLARED_BIN", cf_bin)
        monkeypatch.setattr(rcom.time, "sleep", lambda *_a: None)

        monkeypatch.setattr(boot_llm, "LLAMA_PORT", llm_port)
        monkeypatch.setattr(boot_llm, "LLAMA_INTERNAL_PORT", llm_port)
        monkeypatch.setattr(boot_llm, "SERVICE_META_FILE", meta_file)
        monkeypatch.setattr(boot_llm, "TUNNEL_URL_FILE", comfy_url_file)
        monkeypatch.setattr(boot_llm, "LLM_TUNNEL_URL_FILE", llm_url_file)
        monkeypatch.setattr(boot_llm, "SSE_GATEWAY_FILE", gw_file)
        monkeypatch.setattr(boot_llm, "LLAMA_LOG", tmp / "llama.log")
        monkeypatch.setattr(boot_llm, "SSE_GATEWAY_LOG", tmp / "gw.log")
        monkeypatch.setattr(boot_llm, "COHOST_COMFY", True)
        monkeypatch.setattr(boot_llm, "CF_TUNNEL_TOKEN", "cf-tok-dual")
        monkeypatch.setattr(boot_llm, "CF_TUNNEL_URL", pub_llm_url)
        monkeypatch.setattr(boot_llm, "CF_COMFY_TUNNEL_URL", pub_comfy_url)
        monkeypatch.setattr(boot_llm, "load_companion_text", lambda _n: "# gw v2 clean")
        monkeypatch.setattr(boot_llm, "_query_steady_vram", lambda: ("G4", 20.0, 96.0))
        monkeypatch.setattr(boot_llm.time, "sleep", lambda *_a: None)
        monkeypatch.setattr(boot_llm.shutil, "which", lambda *_a: None)
        monkeypatch.setattr(
            boot_llm.subprocess,
            "run",
            lambda cmd, **_kw: subprocess.CompletedProcess([str(x) for x in cmd], 0, "", ""),
        )
        monkeypatch.setenv("LLM_DEFER_TUNNEL", "1")
        monkeypatch.setenv("CF_TUNNEL_TOKEN", "cf-tok-dual")

        # 阶段 1：ComfyUI 未开机，LLM_DEFER_TUNNEL=1 延后隧道，不被双栖闸门卡住
        deferred_calls: list[str] = []
        monkeypatch.setattr(boot_llm, "start_cloudflared_tunnel", lambda **_kw: deferred_calls.append("called") or "")
        ret_defer = boot_llm.start_tunnel_and_write_contract("UD-Q4_K_XL", llm_key, "G4", 20.0, 96.0)
        assert ret_defer == "" and deferred_calls == []
        meta_p1 = json.loads(meta_file.read_text(encoding="utf-8"))
        assert meta_p1["kind"] == "dual"
        assert meta_p1["sse_gateway_version"] == boot_llm.SSE_GATEWAY_VERSION
        assert meta_p1["llm_port"] == llm_port

        # 热升级：存量网关文件不同或 sse_gateway_version 为旧版时，拒绝复用必须重启并覆盖新网关
        gw_file.write_text("# legacy sse gw v1\n", encoding="utf-8")
        spawned_gw: list[_FakeProc] = []
        monkeypatch.setattr(boot_llm.subprocess, "Popen", lambda *_a, **_kw: (spawned_gw.append(_FakeProc(63000 + len(spawned_gw))), spawned_gw[-1])[1])
        boot_llm.start_llama_and_wait(["llama-server"], "", "UD-Q4_K_XL", llm_key)
        assert len(spawned_gw) == 2 and gw_file.read_text(encoding="utf-8").strip() == "# gw v2 clean"

        meta_v1 = dict(meta_p1, sse_gateway_version=1)
        meta_file.write_text(json.dumps(meta_v1), encoding="utf-8")
        spawned_gw.clear()
        boot_llm.start_llama_and_wait(["llama-server"], "", "UD-Q4_K_XL", llm_key)
        assert len(spawned_gw) == 2

        # 阶段 2：ComfyUI 启动就绪，发布共享命名隧道并对双公网入口执行双向鉴权验收
        meta_file.write_text(json.dumps(meta_p1), encoding="utf-8")
        local_comfy_state["alive"] = True
        pkill_hits: list[list[str]] = []
        monkeypatch.setattr(shutil, "which", lambda name: str(cf_bin) if name == "cloudflared" else None)

        def _rcom_run(cmd: list[Any], **_kw: Any) -> subprocess.CompletedProcess[str]:
            cs = [str(x) for x in cmd]
            if cs and cs[0] == "pkill":
                pkill_hits.append(cs)
            if cs and cs[0] == "pgrep":
                return subprocess.CompletedProcess(cs, 1, stdout="", stderr="")
            return subprocess.CompletedProcess(cs, 0, stdout="", stderr="")

        def _rcom_popen(_cmd: list[Any], **_kw: Any) -> _FakeProc:
            cf_log.write_text("INF Registered tunnel connection connIndex=0\n", encoding="utf-8")
            return _FakeProc(pid=64001)

        monkeypatch.setattr(rcom.subprocess, "run", _rcom_run)
        monkeypatch.setattr(rcom.subprocess, "Popen", _rcom_popen)

        # 若公网 LLM 对匿名请求返回 HTTP 200（未生效鉴权），finalize_service_contract 必须立即 pkill 并清理
        pub_llm_state["auth_enforced"] = False
        rcom.start_cloudflared_tunnel(
            local_port=comfy_port, service_kind="comfy", cf_token="cf-tok-dual",
            cf_url=pub_comfy_url, cohost_other=True, tunnel_re=_re.compile(loop_pat), auth_key=comfy_key,
        )
        assert comfy_url_file.is_file()
        with pytest.raises(RuntimeError, match="HTTP 200"):
            rcom.finalize_service_contract(
                is_dual=True, profiles=["minimax", "qwen_27b"], comfy_profiles=["minimax"],
                llm_profile="qwen_27b", cf_comfy_url=pub_comfy_url, cf_llm_url=pub_llm_url, tunnel_pattern=loop_pat,
            )
        assert pkill_hits and not comfy_url_file.exists() and not llm_url_file.exists()

        # 公网双入口双向鉴权均正常时，完整发布双栖契约
        pub_llm_state["auth_enforced"] = True
        pub_u = rcom.start_cloudflared_tunnel(
            local_port=comfy_port, service_kind="comfy", cf_token="cf-tok-dual",
            cf_url=pub_comfy_url, cohost_other=True, tunnel_re=_re.compile(loop_pat), auth_key=comfy_key,
        )
        final_meta = rcom.finalize_service_contract(
            is_dual=True, profiles=["minimax", "qwen_27b"], comfy_profiles=["minimax"],
            llm_profile="qwen_27b", cf_comfy_url=pub_comfy_url, cf_llm_url=pub_llm_url, tunnel_pattern=loop_pat,
        )
        assert pub_u == pub_comfy_url
        assert final_meta["comfy_tunnel_url"] == pub_comfy_url
        assert final_meta["llm_tunnel_url"] == pub_llm_url
        assert comfy_url_file.read_text(encoding="utf-8").strip() == pub_comfy_url
        assert llm_url_file.read_text(encoding="utf-8").strip() == pub_llm_url


def test_multi_comfy_cmd_up_resets_defer_tunnel_and_finalize_starts_missing_tunnel(
    isolated_env: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证多 ComfyUI 工作流部署显式清零 COMFY_DEFER_TUNNEL，且 finalize 禁止盲写未验收域名。"""
    import argparse
    import colab_ctl

    tmp = isolated_env.tmp
    fake_exe = tmp / "colab.exe"
    fake_py = tmp / "colab_py.exe"
    fake_rclone = tmp / "rclone.conf"
    fake_anima_boot = tmp / "fake_boot_anima.py"
    fake_mm_boot = tmp / "fake_boot_minimax.py"
    for fp in (fake_exe, fake_py, fake_rclone, fake_anima_boot, fake_mm_boot):
        fp.write_text("# stub\n", encoding="utf-8")

    executed_bodies: list[str] = []
    monkeypatch.setattr(colab_ctl, "COLAB_EXE", fake_exe)
    monkeypatch.setattr(colab_ctl, "COLAB_PY", fake_py)
    monkeypatch.setattr(colab_ctl, "RCLONE_CONF", fake_rclone)
    monkeypatch.setattr(colab_ctl, "BOOTSTRAP_SCRIPT", fake_anima_boot)
    monkeypatch.setattr(colab_ctl, "MINIMAX_BOOTSTRAP_SCRIPT", fake_mm_boot)
    monkeypatch.setattr(
        colab_ctl,
        "_load_cf_tunnel_secret",
        lambda: {
            "tunnel_token": "cf-tok-multi-comfy",
            "comfy_url": "https://comfy.fixed.example",
            "llm_url": "https://llm.fixed.example",
        },
    )
    monkeypatch.setattr(colab_ctl, "run_preflight_checks", lambda **_kw: SimpleNamespace(ok=True))
    monkeypatch.setattr(colab_ctl, "_core_ensure_or_adopt_session", lambda **_kw: True)
    monkeypatch.setattr(colab_ctl, "_run", lambda *_a, **_kw: subprocess.CompletedProcess([], 0, "", ""))
    monkeypatch.setattr(
        colab_ctl,
        "_colab_exec_file",
        lambda _s, script_path, **_kw: (
            executed_bodies.append(Path(script_path).read_text(encoding="utf-8")),
            subprocess.CompletedProcess([], 0, "", ""),
        )[1],
    )

    up_ns = argparse.Namespace(
        session="comfy",
        endpoint=None,
        profile="anima",
        profiles="anima,minimax",
        gpu="G4",
        high_mem=False,
        recreate=False,
        skip_preflight=True,
        minimax_lite=False,
        no_r2v=False,
        no_pe=False,
        no_named_tunnel=False,
    )
    colab_ctl.cmd_up(up_ns)
    assert len(executed_bodies) == 2
    assert "os.environ['COMFY_DEFER_TUNNEL'] = '1'" in executed_bodies[0]
    assert "os.environ['COMFY_DEFER_TUNNEL'] = '0'" in executed_bodies[1]
    assert "finalize_service_contract" in executed_bodies[1]
    # 复用 Jupyter 内核时必须显式复位双栖共存参数与 MiniMax 精简开关，杜绝跨次部署残留
    for body in executed_bodies:
        assert "os.environ['COMFY_COHOST_LLM'] = '0'" in body
        assert "os.environ['COMFY_RESERVE_VRAM'] = ''" in body
    assert "os.environ['MINIMAX_LITE'] = '0'" in executed_bodies[1]
    assert "os.environ['MINIMAX_WITH_R2V'] = '1'" in executed_bodies[1]
    assert "os.environ['MINIMAX_WITH_PE'] = '1'" in executed_bodies[1]

    # 仅配置 LLM 固定域名（无 comfy_url）时，ComfyUI 阶段必须清空 CF_TUNNEL_URL，不得继承 LLM 域名
    executed_bodies.clear()
    monkeypatch.setattr(
        colab_ctl,
        "_load_cf_tunnel_secret",
        lambda: {"tunnel_token": "cf-tok-llm-only", "llm_url": "https://llm.fixed.example"},
    )
    colab_ctl.cmd_up(up_ns)
    assert len(executed_bodies) == 2
    for body in executed_bodies:
        assert "os.environ['CF_TUNNEL_TOKEN'] = ''" in body
        assert "os.environ['CF_TUNNEL_URL'] = ''" in body
        assert "llm.fixed.example" not in body.split("\n# stub")[0]

    # TUNNEL_URL_FILE 缺失时，finalize_service_contract 必须调用 start_cloudflared_tunnel 验收，禁止盲写静态域名
    key_file = tmp / "comfy_auth_key"
    comfy_url_file = tmp / "comfy_tunnel_url.txt"
    meta_file = tmp / "colab_service_meta.json"
    key_file.write_text("cpk-multi-comfy-key-123\n", encoding="utf-8")
    monkeypatch.setattr(rcom, "AUTH_KEY_FILE_DEFAULT", key_file)
    monkeypatch.setattr(rcom, "TUNNEL_URL_FILE", comfy_url_file)
    monkeypatch.setattr(rcom, "SERVICE_META_FILE", meta_file)
    monkeypatch.setenv("CF_TUNNEL_TOKEN", "cf-tok-multi-comfy")

    tunnel_calls: list[dict[str, Any]] = []

    def _fail_start_tunnel(**kw: Any) -> str:
        tunnel_calls.append(dict(kw))
        raise RuntimeError("cloudflared down")

    monkeypatch.setattr(rcom, "start_cloudflared_tunnel", _fail_start_tunnel)
    with pytest.raises(RuntimeError, match="cloudflared down"):
        rcom.finalize_service_contract(
            is_dual=False,
            profiles=["anima", "minimax"],
            comfy_profiles=["anima", "minimax"],
            cf_comfy_url="https://comfy.fixed.example",
            tunnel_pattern=r"https://comfy\.fixed\.example",
        )
    assert len(tunnel_calls) == 1
    assert tunnel_calls[0]["cf_token"] == "cf-tok-multi-comfy"
    assert tunnel_calls[0]["cf_url"] == "https://comfy.fixed.example"
    assert not meta_file.exists()

    def _ok_start_tunnel(**kw: Any) -> str:
        tunnel_calls.append(dict(kw))
        return "https://comfy.fixed.example"

    monkeypatch.setattr(rcom, "start_cloudflared_tunnel", _ok_start_tunnel)
    meta_ok = rcom.finalize_service_contract(
        is_dual=False,
        profiles=["anima", "minimax"],
        comfy_profiles=["anima", "minimax"],
        cf_comfy_url="https://comfy.fixed.example",
        tunnel_pattern=r"https://comfy\.fixed\.example",
    )
    assert meta_ok["tunnel_url"] == "https://comfy.fixed.example"
    assert meta_file.is_file()
    assert json.loads(meta_file.read_text(encoding="utf-8"))["comfy_tunnel_url"] == "https://comfy.fixed.example"

    # 隧道地址文件为上次部署遗留（公网实测不可达）时，不得直接信任，必须重新拉起
    comfy_url_file.write_text("https://comfy.fixed.example\n", encoding="utf-8")
    monkeypatch.setattr(rcom, "probe_public_tunnel_url", lambda *_a, **_kw: False)
    calls_before = len(tunnel_calls)
    rcom.finalize_service_contract(
        is_dual=False,
        profiles=["anima"],
        comfy_profiles=["anima"],
        cf_comfy_url="https://comfy.fixed.example",
        tunnel_pattern=r"https://comfy\.fixed\.example",
    )
    assert len(tunnel_calls) == calls_before + 1

    # 文件地址公网实测可达时直接复用，不重复拉起
    monkeypatch.setattr(rcom, "probe_public_tunnel_url", lambda *_a, **_kw: True)
    calls_before = len(tunnel_calls)
    rcom.finalize_service_contract(
        is_dual=False,
        profiles=["anima"],
        comfy_profiles=["anima"],
        cf_comfy_url="https://comfy.fixed.example",
        tunnel_pattern=r"https://comfy\.fixed\.example",
    )
    assert len(tunnel_calls) == calls_before
