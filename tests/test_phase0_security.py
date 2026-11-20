"""Phase 0B 安全边界补强离线回归测试套件。

严格使用 `COLAB_PET_MOCK_VAULT_PATH` 等沙箱环境变量隔离，严禁触碰宿主机真实 Windows 凭据库、
真实 `.cf_tunnel_secret.json` 或真实远端 Colab 实例。

覆盖 4 项安全边界与典型绕过尝试：
1. `is_readonly_safe_remote_cmd` 严格只读诊断白名单（放行合法诊断，阻断写/替换/链式/通配符/解释器/Python）；
2. 无头 MCP `_default_mcp_executor` 与 `run_mcp_stdio_server` 默认 Fail-Closed（仅放行 safe 与只读诊断，
   拦截非只读 exec / sync / 计费开关机；支持显式参数与 `COLAB_MCP_ALLOW_MUTATIONS=1` 放行且不可绕过硬黑名单）；
3. `AgentService._tool_exec_remote_colab` 经 `_request_user_confirm` 交互审批（硬黑名单直拒、只读直通、
   非只读挂起弹窗展示完整代码与槽位、同意/拒绝/取消与自动放行策略隔离）；
4. `colab_sse_gateway._cors_header_block` CORS 收紧与 `pet_app._PetAssetHandler` `/assets/` 扩展名与路径白名单。
"""

from __future__ import annotations

import asyncio
import http.client
import io
import json
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
from aiohttp import web

PKG_DIR = Path(__file__).resolve().parents[1]
PET_DIR = PKG_DIR / "colab_pet"
for _p in (str(PKG_DIR), str(PET_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import colab_backend as cb  # noqa: E402
import colab_security as csec  # noqa: E402
import colab_sse_gateway as sse_gw  # noqa: E402
import pet_app  # noqa: E402
from colab_core import agent_tools  # noqa: E402
from colab_core.runtime import CancelToken  # noqa: E402


@pytest.fixture()
def isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """为每条用例创建完全隔离的凭据保险箱、状态文件、日志与策略环境变量沙箱。"""
    csec.clear_ephemeral_secrets()
    vault_file = tmp_path / "mock_vault.json"
    cf_file = tmp_path / ".cf_tunnel_secret.json"
    diag_log = tmp_path / "logs" / "colab_diag.log"
    state_file = tmp_path / "pet_state.json"

    monkeypatch.setenv("COLAB_PET_MOCK_VAULT_PATH", str(vault_file))
    monkeypatch.setenv("COLAB_PET_CF_CONFIG_PATH", str(cf_file))
    monkeypatch.setenv("COLAB_PET_DIAG_LOG_PATH", str(diag_log))
    monkeypatch.setenv("COLAB_PET_MOCK_STATE_PATH", str(state_file))
    monkeypatch.delenv("COLAB_MCP_ALLOW_MUTATIONS", raising=False)
    monkeypatch.delenv("COLAB_AGENT_AUTO_APPROVE_EXEC", raising=False)
    monkeypatch.delenv("SSE_ALLOWED_ORIGINS", raising=False)
    monkeypatch.delenv("SSE_CORS_ALLOW_CREDENTIALS", raising=False)

    yield SimpleNamespace(
        tmp=tmp_path,
        vault_file=vault_file,
        cf_file=cf_file,
        diag_log=diag_log,
        state_file=state_file,
    )
    pet_app._set_active_media_root(None)
    csec.clear_ephemeral_secrets()


def _make_backend(env_ns: SimpleNamespace, name: str = "phase0_sec") -> cb.ColabPetBackend:
    """实例化绑定到隔离临时目录的 ColabPetBackend。"""
    return cb.ColabPetBackend(
        state_path=env_ns.tmp / f"{name}.json",
        download_root=env_ns.tmp / f"{name}_dl",
        cf_config_path=env_ns.cf_file,
    )


def _wait_until(predicate, *, timeout: float = 5.0, interval: float = 0.02, err_msg: str = "等待超时"):
    """轮询等待条件满足。"""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        val = predicate()
        if val:
            return val
        time.sleep(interval)
    raise AssertionError(err_msg)


# =========================================================================
# 1. is_readonly_safe_remote_cmd 严格只读诊断白名单与绕过测试
# =========================================================================


def test_readonly_remote_cmd_whitelist_and_bypass_attempts(isolated_env: SimpleNamespace) -> None:
    """校验仅明确无副作用的只读 Bash 诊断命令放行，一切写操作/链式/通配符/解释器/Python 均被拒绝。"""
    _ = isolated_env

    allowed_bash_cmds = [
        "nvidia-smi",
        "nvidia-smi -L",
        "nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader",
        "free -h",
        "df -h /",
        "df -h /content",
        "uptime",
        "uname -a",
        "ps aux",
        "ps -ef",
        "ps -e -o pid,comm",
        "ps -eo pid,comm,pcpu,pmem",
        "ls -lh /content/ComfyUI/models/loras",
        "ls -lh /root/comfy_colab.log",
        "tail -n 100 /root/comfy_colab.log",
        "tail -n 50 /root/comfy_anima.log",
        "head -n 80 /root/llama_server.log",
        "tail -n 40 /root/llm_sse_gateway.log",
        "nvidia-smi && df -h /",
        "free -h && uptime",
    ]
    for cmd in allowed_bash_cmds:
        ok, reason = agent_tools.is_readonly_safe_remote_cmd(cmd, "bash")
        assert ok is True and reason == "", f"合法只读诊断命令被误拦: {cmd!r} -> {reason}"

    rejected_cases = [
        # 所有 Python 代码一律视为非只读
        ("print('hello')", "python"),
        ("import torch; print(torch.cuda.is_available())", "python"),
        # 重定向写入与参数写文件
        ("nvidia-smi > /tmp/out.txt", "bash"),
        ("uptime >> /tmp/out.txt", "bash"),
        ("nvidia-smi -f /tmp/gpu.log", "bash"),
        ("ls /content 2>/tmp/err.log", "bash"),
        ("df -h < /etc/hosts", "bash"),
        # 命令替换与变量展开
        ("ls $(whoami)", "bash"),
        ("ls `id`", "bash"),
        ("df -h $HOME", "bash"),
        ("uname ${IFS}-a", "bash"),
        # 分号 / 管道 / 逻辑或 / 后台 / 多行 / 链式写操作
        ("nvidia-smi; uptime", "bash"),
        ("nvidia-smi || id", "bash"),
        ("ps aux | sh", "bash"),
        ("nvidia-smi | head -n 5", "bash"),
        ("nvidia-smi &", "bash"),
        ("nvidia-smi && touch /tmp/pwn", "bash"),
        ("nvidia-smi && rm -f /tmp/a", "bash"),
        ("nvidia-smi\ntouch /tmp/pwn", "bash"),
        # 任意通配符探测敏感文件
        ("ls /tmp/rc*", "bash"),
        ("ls /root/*auth*", "bash"),
        ("ls /content/*.json", "bash"),
        ("ls /root/?*", "bash"),
        ("ls /root/[a-z]*", "bash"),
        # 隐藏文件、敏感文件与越界路径探测
        ("ls -la", "bash"),
        ("ls -la /root", "bash"),
        ("ls /tmp/rclone.conf", "bash"),
        ("ls /root/.comfy_auth_key", "bash"),
        ("tail -n 20 /etc/passwd", "bash"),
        ("tail -n 20 /tmp/rclone.conf", "bash"),
        ("head -n 20 /root/comfy_../.comfy_auth_key", "bash"),
        ("tail -f /root/comfy_colab.log", "bash"),
        ("tail -n 5000 /root/comfy_colab.log", "bash"),
        ("tail -n 10 /root/comfy_colab.log /etc/passwd", "bash"),
        # 解释器、下载与编码工具
        ("python -c 'print(1)'", "bash"),
        ("python3 -c 'print(1)'", "bash"),
        ("sh -c 'uptime'", "bash"),
        ("bash -c 'nvidia-smi'", "bash"),
        ("curl https://example.com", "bash"),
        ("wget https://example.com", "bash"),
        ("base64 /root/comfy_colab.log", "bash"),
        ("/usr/bin/nvidia-smi", "bash"),
        # 进程环境变量泄露（含 procps 混合短选项回退 BSD 转储 environ）与 GPU 参数修改
        ("ps auxe", "bash"),
        ("ps e", "bash"),
        ("ps -auxe", "bash"),
        ("ps -auxwwe", "bash"),
        ("ps -xe", "bash"),
        ("ps -ex", "bash"),
        ("ps -eux", "bash"),
        ("ps -e -x", "bash"),
        ("ps aux -e", "bash"),
        ("ps -e aux", "bash"),
        ("ps aux -eo pid,comm", "bash"),
        ("ps -eo pid,environ", "bash"),
        ("nvidia-smi --power-limit=100", "bash"),
        ("nvidia-smi -pm 1", "bash"),
        # 硬黑名单危险命令
        ("rm -rf /", "bash"),
        ("cat ~/.cf_tunnel_secret.json", "bash"),
        # 非法 mode
        ("nvidia-smi", "powershell"),
    ]
    for code, mode in rejected_cases:
        ok, reason = agent_tools.is_readonly_safe_remote_cmd(code, mode)
        assert ok is False and reason, f"非只读/绕过命令未被拦截: ({mode}) {code!r}"


# =========================================================================
# 2. 无头 MCP 默认 Fail-Closed 与显式受控放行校验
# =========================================================================


def test_headless_mcp_executor_default_fail_closed_and_opt_in(
    isolated_env: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证无头 MCP 默认仅开放 safe 工具与只读诊断，拒绝未授权写/计费操作，且硬黑名单永不可绕过。"""
    fake_ctl = isolated_env.tmp / "fake_colab_ctl.py"
    fake_ctl.write_text("# stub ctl\n", encoding="utf-8")

    spawned_cmds: list[tuple[list[str], str | None]] = []

    def _mock_subproc_run(cmd, input=None, **_kwargs):
        cmd_list = [str(x) for x in cmd]
        spawned_cmds.append((cmd_list, input))
        if "doctor" in cmd_list:
            return subprocess.CompletedProcess(cmd_list, 0, stdout='{"ok": true}', stderr="")
        return subprocess.CompletedProcess(cmd_list, 0, stdout="MOCK_OK", stderr="")

    with patch.object(agent_tools.subprocess, "run", side_effect=_mock_subproc_run):
        # 2a. 默认无头执行器：严格 Fail-Closed
        exec_default = agent_tools._default_mcp_executor(fake_ctl)

        # safe 工具正常放行
        r_doc = exec_default("run_environment_doctor", {"profile": "anima"})
        assert r_doc["ok"] is True
        r_st = exec_default("get_cluster_and_pool_status", {})
        assert r_st["ok"] is True
        r_log = exec_default("tail_instance_logs", {"source": "local_diag", "lines": 20})
        assert r_log["ok"] is True

        # 只读白名单诊断命令放行
        r_ro = exec_default("exec_remote_colab", {"code": "nvidia-smi && free -h", "mode": "bash"})
        assert r_ro["ok"] is True and r_ro["stdout"] == "MOCK_OK"

        # 非只读 exec、sync、manage_compute_instance 默认全部拦截，零子进程触发
        spawned_cmds.clear()
        blocked_calls = [
            ("exec_remote_colab", {"code": "touch /tmp/side_effect", "mode": "bash"}),
            ("exec_remote_colab", {"code": "nvidia-smi > /tmp/out.txt", "mode": "bash"}),
            ("exec_remote_colab", {"code": "ls /tmp/rc*", "mode": "bash"}),
            ("exec_remote_colab", {"code": "import torch; print(torch.cuda.is_available())", "mode": "python"}),
            ("sync_and_pull_outputs", {}),
            ("manage_compute_instance", {"action": "start", "profiles": ["anima"], "gpu": "L4"}),
            ("manage_compute_instance", {"action": "stop_slot", "slot": "comfy"}),
            ("manage_compute_instance", {"action": "stop_all"}),
        ]
        for tool_name, tool_args in blocked_calls:
            res = exec_default(tool_name, tool_args)
            assert res["ok"] is False, (tool_name, res)
            assert res.get("blocked") is True, (tool_name, res)
            assert res.get("approval_required") is True, (tool_name, res)
            assert res.get("error"), (tool_name, res)
        assert spawned_cmds == [], f"默认只读模式下不应派生任何子进程: {spawned_cmds}"

        # 硬黑名单命令直接阻断（不提示 approval_required）
        r_hard = exec_default("exec_remote_colab", {"code": "rm -rf /", "mode": "bash"})
        assert r_hard["ok"] is False and r_hard.get("blocked") is True
        assert not r_hard.get("approval_required")
        assert spawned_cmds == []

        # 2b. 通过显式参数 allow_unattended_mutations=True 放行受控变更，但硬黑名单仍不可绕过
        exec_opt_in = agent_tools._default_mcp_executor(fake_ctl, allow_unattended_mutations=True)
        assert exec_opt_in("exec_remote_colab", {"code": "print(123)", "mode": "python"})["ok"] is True
        assert exec_opt_in("sync_and_pull_outputs", {})["ok"] is True
        assert exec_opt_in("manage_compute_instance", {"action": "stop_all"})["ok"] is True
        assert len(spawned_cmds) == 3

        spawned_cmds.clear()
        r_still_hard = exec_opt_in("exec_remote_colab", {"code": "cat ~/.cf_tunnel_secret.json", "mode": "bash"})
        assert r_still_hard["ok"] is False and r_still_hard.get("blocked") is True
        assert spawned_cmds == []

        # 2c. 通过受控环境变量 COLAB_MCP_ALLOW_MUTATIONS=1 放行
        monkeypatch.setenv("COLAB_MCP_ALLOW_MUTATIONS", "1")
        assert exec_default("sync_and_pull_outputs", {})["ok"] is True
        assert exec_default("manage_compute_instance", {"action": "stop_slot", "slot": "llm"})["ok"] is True
        assert len(spawned_cmds) == 2

        # 关闭环境变量后立即恢复 Fail-Closed
        monkeypatch.delenv("COLAB_MCP_ALLOW_MUTATIONS", raising=False)
        spawned_cmds.clear()
        r_back = exec_default("manage_compute_instance", {"action": "stop_all"})
        assert r_back["ok"] is False and r_back.get("approval_required") is True
        assert spawned_cmds == []

        # 2d. JSON-RPC 2.0 Stdio 管道验证默认拦截结构化回传
        rpc_in = io.StringIO(
            "\n".join(
                [
                    json.dumps(
                        {
                            "jsonrpc": "2.0",
                            "id": 10,
                            "method": "tools/call",
                            "params": {"name": "manage_compute_instance", "arguments": {"action": "start"}},
                        }
                    ),
                    json.dumps(
                        {
                            "jsonrpc": "2.0",
                            "id": 11,
                            "method": "tools/call",
                            "params": {"name": "exec_remote_colab", "arguments": {"code": "print(1)", "mode": "python"}},
                        }
                    ),
                ]
            )
            + "\n"
        )
        rpc_out = io.StringIO()
        assert agent_tools.run_mcp_stdio_server(fake_ctl, stdin=rpc_in, stdout=rpc_out) == 0
        lines = [json.loads(ln) for ln in rpc_out.getvalue().splitlines() if ln.strip()]
        assert len(lines) == 2
        for resp in lines:
            assert resp["result"]["isError"] is True
            body = json.loads(resp["result"]["content"][0]["text"])
            assert body["ok"] is False and body["blocked"] is True and body["approval_required"] is True

    # 2e. 验证 ClaudeCodeDriver 默认不在 mcp-config 授予变更环境变量，仅显式 allow_mcp_mutations=True 时注入
    seen_mcp_cfgs: list[dict[str, Any]] = []
    fake_claude_py = isolated_env.tmp / "fake_claude_check.py"
    cfg_dump_file = isolated_env.tmp / "dumped_mcp_cfg.json"
    fake_claude_py.write_text(
        f"""import json, sys
args = sys.argv[1:]
cfg_p = args[args.index("--mcp-config") + 1]
open({str(cfg_dump_file)!r}, "w", encoding="utf-8").write(open(cfg_p, "r", encoding="utf-8").read())
print(json.dumps({{"type": "result", "is_error": False, "result": "ok"}}), flush=True)
""",
        encoding="utf-8",
    )
    fake_claude_cmd = isolated_env.tmp / "fake_claude_check.cmd"
    fake_claude_cmd.write_text(f'@echo off\r\n"{sys.executable}" "{fake_claude_py}" %*\r\n', encoding="utf-8")

    drv_default = agent_tools.ClaudeCodeDriver(ctl_script=fake_ctl, claude_bin=str(fake_claude_cmd))
    assert drv_default.run("检查状态")["ok"] is True
    seen_mcp_cfgs.append(json.loads(cfg_dump_file.read_text(encoding="utf-8")))
    assert "env" not in seen_mcp_cfgs[-1]["mcpServers"]["colab-studio"]

    drv_mut = agent_tools.ClaudeCodeDriver(
        ctl_script=fake_ctl,
        claude_bin=str(fake_claude_cmd),
        allow_mcp_mutations=True,
    )
    assert drv_mut.run("允许变更")["ok"] is True
    seen_mcp_cfgs.append(json.loads(cfg_dump_file.read_text(encoding="utf-8")))
    assert seen_mcp_cfgs[-1]["mcpServers"]["colab-studio"]["env"] == {"COLAB_MCP_ALLOW_MUTATIONS": "1"}


# =========================================================================
# 3. backend_agent._tool_exec_remote_colab 与 _request_user_confirm 审批闭环
# =========================================================================


def test_backend_agent_exec_remote_colab_approval_gate(
    isolated_env: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证后端 exec_remote_colab 的硬拦截直拒、只读直通、非只读审批弹窗（批准/拒绝/取消）及策略隔离。"""
    backend = _make_backend(isolated_env, "exec_confirm_state")
    svc = backend._agent_svc
    events: list[dict[str, Any]] = []
    unsub = backend.subscribe_events(lambda e: events.append(dict(e)))

    spawned: list[tuple[list[str], str]] = []

    class _MockPopen:
        def __init__(self, cmd, **_kwargs) -> None:
            self._cmd = [str(x) for x in cmd]
            self.returncode = 0
            self.pid = 99991

        def communicate(self, input: str = "", timeout: int = 180) -> tuple[str, str]:
            _ = timeout
            spawned.append((self._cmd, input))
            return "REMOTE_EXEC_OK\n", ""

    import backend_agent as ba_mod

    try:
        with patch.object(ba_mod.subprocess, "Popen", side_effect=_MockPopen):
            # 3a. 硬黑名单直接拒绝，不弹审批且零子进程
            events.clear()
            r_blk = svc._execute_tool("exec_remote_colab", {"code": "rm -rf /*", "mode": "bash"}, CancelToken(), ())
            assert r_blk["ok"] is False and r_blk.get("blocked") is True
            assert not any(e.get("phase") == "confirm_required" for e in events)
            assert spawned == []

            # 3b. 只读白名单命令免弹窗直通
            events.clear()
            r_ro = svc._execute_tool(
                "exec_remote_colab",
                {"code": "nvidia-smi && df -h /", "mode": "bash", "slot": "auto"},
                CancelToken(),
                (),
            )
            assert r_ro["ok"] is True and r_ro["output"] == "REMOTE_EXEC_OK"
            assert not any(e.get("phase") == "confirm_required" for e in events)
            assert len(spawned) == 1

            # 3c. 仅开启计费自动放行 (agent_auto_approve_billing=True) 不得放行非只读远端代码！
            backend.update_prefs({"agent_auto_approve_billing": True})

            # 非只读 Python 代码 -> 挂起弹窗展示完整代码与槽位 -> 用户拒绝 -> 0 子进程
            events.clear()
            spawned.clear()
            py_code = "import torch\nprint(torch.cuda.is_available())"
            rej_box: list[dict[str, Any]] = []
            th_rej = threading.Thread(
                target=lambda: rej_box.append(
                    svc._execute_tool(
                        "exec_remote_colab",
                        {"code": py_code, "mode": "python", "slot": "comfy"},
                        CancelToken(),
                        (),
                    )
                ),
                daemon=True,
            )
            th_rej.start()
            cfm_evt = _wait_until(
                lambda: next((e for e in events if e.get("phase") == "confirm_required"), None),
                err_msg="非只读 Python 代码未触发 confirm_required 审批弹窗",
            )
            cfm = cfm_evt["confirm"]
            assert cfm["tool"] == "exec_remote_colab"
            assert py_code in cfm["detail"]
            assert "comfy" in cfm["detail"]
            assert spawned == []

            res_rej_call = backend.resolve_agent_confirm({"confirm_id": cfm["id"], "approved": False})
            assert res_rej_call["ok"] is True and res_rej_call["approved"] is False
            th_rej.join(timeout=5.0)
            assert not th_rej.is_alive() and rej_box
            assert rej_box[0]["ok"] is False and rej_box[0].get("cancelled") is True
            assert rej_box[0].get("approval_required") is True
            assert spawned == []

            # 3d. 非只读 Bash（写文件）-> 挂起弹窗 -> 用户批准 -> 放行执行
            events.clear()
            spawned.clear()
            bash_write = "touch /tmp/test_marker.txt"
            ok_box: list[dict[str, Any]] = []
            th_ok = threading.Thread(
                target=lambda: ok_box.append(
                    svc._execute_tool(
                        "exec_remote_colab",
                        {"code": bash_write, "mode": "bash", "slot": "auto"},
                        CancelToken(),
                        (),
                    )
                ),
                daemon=True,
            )
            th_ok.start()
            cfm_evt2 = _wait_until(
                lambda: next((e for e in events if e.get("phase") == "confirm_required"), None),
                err_msg="非只读 Bash 写命令未触发审批弹窗",
            )
            assert bash_write in cfm_evt2["confirm"]["detail"]
            res_ok_call = backend.resolve_agent_confirm({"confirm_id": cfm_evt2["confirm"]["id"], "approved": True})
            assert res_ok_call["ok"] is True and res_ok_call["approved"] is True
            th_ok.join(timeout=5.0)
            assert not th_ok.is_alive() and ok_box and ok_box[0]["ok"] is True
            assert len(spawned) == 1 and spawned[0][1] == bash_write

            # 3e. 挂起期间通过 CancelToken 取消 -> 立即唤醒且不执行子进程
            events.clear()
            spawned.clear()
            tok = CancelToken()
            can_box: list[dict[str, Any]] = []
            th_can = threading.Thread(
                target=lambda: can_box.append(
                    svc._execute_tool(
                        "exec_remote_colab",
                        {"code": "ls /tmp/rc*", "mode": "bash"},
                        tok,
                        (),
                    )
                ),
                daemon=True,
            )
            th_can.start()
            _wait_until(
                lambda: next((e for e in events if e.get("phase") == "confirm_required"), None),
                err_msg="通配符命令未触发审批弹窗",
            )
            tok.cancel()
            th_can.join(timeout=3.0)
            assert not th_can.is_alive() and can_box
            assert can_box[0]["ok"] is False and can_box[0].get("cancelled") is True
            assert spawned == []

            # 3f. GUI 模式下必须彻底忽略 COLAB_AGENT_AUTO_APPROVE_EXEC 环境变量；即便内部偏好开启，硬黑名单仍拦截
            events.clear()
            spawned.clear()
            monkeypatch.setenv("COLAB_AGENT_AUTO_APPROVE_EXEC", "1")
            env_rej_box: list[dict[str, Any]] = []
            th_env_rej = threading.Thread(
                target=lambda: env_rej_box.append(
                    svc._execute_tool("exec_remote_colab", {"code": "print(42)", "mode": "python"}, CancelToken(), ())
                ),
                daemon=True,
            )
            th_env_rej.start()
            cfm_env = _wait_until(
                lambda: next((e for e in events if e.get("phase") == "confirm_required"), None),
                err_msg="设置 COLAB_AGENT_AUTO_APPROVE_EXEC 后 GUI 模式仍未触发审批弹窗",
            )
            backend.resolve_agent_confirm({"confirm_id": cfm_env["confirm"]["id"], "approved": False})
            th_env_rej.join(timeout=5.0)
            assert not th_env_rej.is_alive() and env_rej_box
            assert env_rej_box[0]["ok"] is False
            assert env_rej_box[0].get("cancelled") is True
            assert env_rej_box[0].get("approval_required") is True
            assert spawned == []

            monkeypatch.delenv("COLAB_AGENT_AUTO_APPROVE_EXEC", raising=False)
            with backend._store.lock:
                backend.state.setdefault("prefs", {})["agent_auto_approve_remote_exec"] = True
            try:
                spawned.clear()
                r_hard2 = svc._execute_tool("exec_remote_colab", {"code": "cat ~/.cf_tunnel_secret.json", "mode": "bash"}, CancelToken(), ())
                assert r_hard2["ok"] is False and r_hard2.get("blocked") is True
                assert spawned == []
            finally:
                with backend._store.lock:
                    backend.state.setdefault("prefs", {}).pop("agent_auto_approve_remote_exec", None)

            # 3g. 确认重构后的 manage_compute_instance 复用 _request_user_confirm 审批闭环正常
            monkeypatch.delenv("COLAB_AGENT_AUTO_APPROVE_EXEC", raising=False)
            backend.update_prefs({"agent_auto_approve_billing": False})
            stop_hits: list[int] = []
            backend.stop_all_instances = lambda: (stop_hits.append(1), {"ok": True})[1]  # type: ignore[method-assign]
            events.clear()
            m_box: list[dict[str, Any]] = []
            th_m = threading.Thread(
                target=lambda: m_box.append(
                    svc._execute_tool("manage_compute_instance", {"action": "stop_all"}, CancelToken(), ())
                ),
                daemon=True,
            )
            th_m.start()
            cfm_m = _wait_until(
                lambda: next((e for e in events if e.get("phase") == "confirm_required"), None),
                err_msg="manage_compute_instance 未触发审批弹窗",
            )
            assert cfm_m["confirm"]["tool"] == "manage_compute_instance"
            backend.resolve_agent_confirm({"confirm_id": cfm_m["confirm"]["id"], "approved": False})
            th_m.join(timeout=3.0)
            assert m_box and m_box[0]["ok"] is False and m_box[0].get("cancelled") is True
            assert stop_hits == []
    finally:
        unsub()


# =========================================================================
# 4. SSE 网关 CORS 策略收紧测试（含真实 TCP/HTTP 端到端校验）
# =========================================================================


def test_sse_gateway_cors_hardening(
    isolated_env: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证默认不反射未知 Origin 且不发 Credentials，白名单精确匹配时才回显 Origin + Vary: Origin。"""
    _ = isolated_env

    # 4a. 默认无 SSE_ALLOWED_ORIGINS：始终返回 `*`，严禁反射攻击者 Origin，严禁发 Allow-Credentials
    blk_default = sse_gw._cors_header_block({"origin": "https://evil.attacker.example"})
    assert "Access-Control-Allow-Origin: *\r\n" in blk_default
    assert "evil.attacker.example" not in blk_default
    assert "Access-Control-Allow-Credentials" not in blk_default

    # 4b. 配置白名单后：精确命中才回显 Origin + Vary: Origin，默认不发 Credentials
    monkeypatch.setenv("SSE_ALLOWED_ORIGINS", "https://app.example.com, http://127.0.0.1:3000 ")
    blk_hit = sse_gw._cors_header_block({"origin": "https://app.example.com"})
    assert "Access-Control-Allow-Origin: https://app.example.com\r\n" in blk_hit
    assert "Vary: Origin\r\n" in blk_hit
    assert "Access-Control-Allow-Credentials" not in blk_hit

    blk_hit2 = sse_gw._cors_header_block({"origin": "http://127.0.0.1:3000"})
    assert "Access-Control-Allow-Origin: http://127.0.0.1:3000\r\n" in blk_hit2
    assert "Vary: Origin\r\n" in blk_hit2
    assert "Access-Control-Allow-Credentials" not in blk_hit2

    # 未命中（未知域名、前缀/子串伪造、缺失 Origin）均不发 Access-Control-Allow-Origin
    for bad_origin in (
        "https://evil.example",
        "https://app.example.com.evil.com",
        "http://app.example.com",
        "https://app.example.com\r\nX-Injected: yes",
        "",
    ):
        blk_miss = sse_gw._cors_header_block({"origin": bad_origin})
        assert "Access-Control-Allow-Origin" not in blk_miss, f"未授权 Origin 不应返回 ACAO: {bad_origin!r}"
        assert "Access-Control-Allow-Credentials" not in blk_miss
        assert "X-Injected" not in blk_miss
        assert "Vary: Origin\r\n" in blk_miss

    # 误配置通配符 `*` 到白名单时，忽略 `*` 并回退无凭证 `*` 模式，绝不反射任意源
    monkeypatch.setenv("SSE_ALLOWED_ORIGINS", "*")
    blk_star = sse_gw._cors_header_block({"origin": "https://evil.example"})
    assert "Access-Control-Allow-Origin: *\r\n" in blk_star
    assert "evil.example" not in blk_star
    assert "Access-Control-Allow-Credentials" not in blk_star

    # 4c. 真实本地 TCP HTTP OPTIONS / GET /v1/models / POST SSE 端到端核验
    async def _run_http_checks() -> None:
        def _free_port() -> int:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", 0))
                return int(s.getsockname()[1])

        up_port = _free_port()
        gw_port = _free_port()

        async def _up_models(_r: web.Request) -> web.Response:
            return web.json_response({"data": [{"id": "Qwen3.8-27B-abliterated"}]})

        async def _up_slow_sse(_r: web.Request) -> web.StreamResponse:
            await asyncio.sleep(0.18)
            resp = web.StreamResponse(status=200, headers={"Content-Type": "text/event-stream"})
            await resp.prepare(_r)
            await resp.write(b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n')
            await resp.write_eof()
            return resp

        up_app = web.Application()
        up_app.router.add_get("/v1/models", _up_models)
        up_app.router.add_post("/v1/chat/completions", _up_slow_sse)
        runner = web.AppRunner(up_app)
        await runner.setup()
        await web.TCPSite(runner, "127.0.0.1", up_port).start()

        monkeypatch.setattr(sse_gw, "LISTEN_PORT", gw_port)
        monkeypatch.setattr(sse_gw, "UPSTREAM_PORT", up_port)
        monkeypatch.setattr(sse_gw, "BASE_MODEL", "Qwen3.8-27B-abliterated")
        monkeypatch.setattr(sse_gw, "HEARTBEAT_SEC", 0.06)
        gw_srv = await asyncio.start_server(sse_gw._handle_client, "127.0.0.1", gw_port)
        try:
            def _req(method: str, path: str, origin: str, data: bytes | None = None) -> tuple[int, dict[str, str], str]:
                hdrs = {"Origin": origin}
                if data is not None:
                    hdrs["Content-Type"] = "application/json"
                req = urllib.request.Request(f"http://127.0.0.1:{gw_port}{path}", data=data, method=method, headers=hdrs)
                with urllib.request.urlopen(req, timeout=4.0) as resp:
                    return resp.status, {k.lower(): v for k, v in resp.headers.items()}, resp.read().decode("utf-8", errors="replace")

            # 默认模式：未知源返回 `*` 且无 Credentials
            monkeypatch.delenv("SSE_ALLOWED_ORIGINS", raising=False)
            st1, h1, _ = await asyncio.to_thread(_req, "OPTIONS", "/v1/chat/completions", "https://evil.example")
            assert st1 == 204
            assert h1.get("access-control-allow-origin") == "*"
            assert "access-control-allow-credentials" not in h1

            # 白名单模式：命中返回精确 Origin + Vary，未命中不返回 ACAO
            monkeypatch.setenv("SSE_ALLOWED_ORIGINS", "https://app.example.com,http://127.0.0.1:3000")
            st2, h2, _ = await asyncio.to_thread(_req, "GET", "/v1/models", "https://app.example.com")
            assert st2 == 200
            assert h2.get("access-control-allow-origin") == "https://app.example.com"
            assert h2.get("vary") == "Origin"
            assert "access-control-allow-credentials" not in h2

            st3, h3, _ = await asyncio.to_thread(_req, "GET", "/v1/models", "https://evil.example")
            assert st3 == 200
            assert "access-control-allow-origin" not in h3
            assert h3.get("vary") == "Origin"
            assert "access-control-allow-credentials" not in h3

            sse_body = json.dumps({"model": "Qwen3.8-27B-abliterated", "stream": True, "messages": [{"role": "user", "content": "hi"}]}).encode("utf-8")
            st4, h4, b4 = await asyncio.to_thread(_req, "POST", "/v1/chat/completions", "http://127.0.0.1:3000", sse_body)
            assert st4 == 200 and ": keepalive" in b4
            assert h4.get("access-control-allow-origin") == "http://127.0.0.1:3000"
            assert h4.get("vary") == "Origin"
            assert "access-control-allow-credentials" not in h4
        finally:
            gw_srv.close()
            await gw_srv.wait_closed()
            await runner.cleanup()

    asyncio.run(_run_http_checks())


# =========================================================================
# 5. 本地 /assets/ 静态扩展名白名单与路径穿越阻断测试
# =========================================================================


def test_pet_asset_handler_assets_whitelist_and_traversal_block(isolated_env: SimpleNamespace) -> None:
    """验证 /assets/ 仅开放白名单扩展名，兼容 png<->webp 回退与 alpha_masks.json，拦截点文件/非白名单/穿越。"""
    # 验证模块常量白名单定义
    assert pet_app._ALLOWED_ASSET_EXTS == frozenset({".webp", ".png", ".ico", ".svg", ".json"})

    fake_pet_root = isolated_env.tmp / "fake_pet"
    fake_assets = fake_pet_root / "assets"
    (fake_assets / "sub").mkdir(parents=True, exist_ok=True)

    # 在父目录放入敏感诱饵（模拟 pet_state.json 与源码文件，绝不能经由 /assets/../ 读出）
    (fake_pet_root / "pet_state.json").write_text('{"secret_state": true}', encoding="utf-8")
    (fake_pet_root / "colab_backend.py").write_text("# secret python source\n", encoding="utf-8")

    # 在 assets 下放入白名单资源与禁止访问的非白名单/隐藏诱饵文件
    (fake_assets / "idle.webp").write_bytes(b"RIFF_FAKE_WEBP_BYTES")
    (fake_assets / "only_png.png").write_bytes(b"\x89PNG_FAKE_PNG_BYTES")
    (fake_assets / "pet.ico").write_bytes(b"\x00\x00\x01\x00FAKEICO")
    (fake_assets / "icon.svg").write_text("<svg></svg>", encoding="utf-8")
    (fake_assets / "alpha_masks.json").write_text('{"width": 1, "height": 1}', encoding="utf-8")

    (fake_assets / "leak.py").write_text("SECRET_PY = 1\n", encoding="utf-8")
    (fake_assets / "debug.log").write_text("SECRET_LOG_LINE\n", encoding="utf-8")
    (fake_assets / "notes.txt").write_text("SECRET_NOTES\n", encoding="utf-8")
    (fake_assets / ".env").write_text("API_KEY=leak\n", encoding="utf-8")
    (fake_assets / ".hidden.json").write_text('{"hidden": true}', encoding="utf-8")
    (fake_assets / "sub" / ".secret.png").write_bytes(b"\x89PNG_HIDDEN")
    (fake_assets / ".dotdir").mkdir(exist_ok=True)
    (fake_assets / ".dotdir" / "ok.png").write_bytes(b"\x89PNG_IN_DOTDIR")

    httpd = pet_app._ThreadingHTTPServer(("127.0.0.1", 0), pet_app._PetAssetHandler)
    httpd.assets_root = fake_assets  # type: ignore[attr-defined]
    port = httpd.server_address[1]
    th = threading.Thread(target=httpd.serve_forever, daemon=True)
    th.start()

    def _get(path: str) -> tuple[int, str, bytes]:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        try:
            conn.request("GET", path)
            resp = conn.getresponse()
            body = resp.read()
            return resp.status, str(resp.getheader("Content-Type") or ""), body
        finally:
            conn.close()

    try:
        # 5a. 白名单静态资源与 .png <-> .webp 双向回退全部 200
        st, ctype, data = _get("/assets/idle.webp")
        assert st == 200 and data == b"RIFF_FAKE_WEBP_BYTES" and "webp" in ctype

        st, ctype, data = _get("/assets/idle.png")
        assert st == 200 and data == b"RIFF_FAKE_WEBP_BYTES" and "webp" in ctype

        st, _, data = _get("/assets/only_png.webp")
        assert st == 200 and data == b"\x89PNG_FAKE_PNG_BYTES"

        st, _, data = _get("/assets/pet.ico")
        assert st == 200 and data.startswith(b"\x00\x00\x01\x00")

        st, _, data = _get("/assets/icon.svg")
        assert st == 200 and data == b"<svg></svg>"

        st, _, data = _get("/assets/alpha_masks.json")
        assert st == 200 and json.loads(data.decode("utf-8"))["width"] == 1

        # 5b. 非白名单扩展名、隐藏点文件/点目录与路径穿越全部 403 且不泄露任何敏感内容
        forbidden_paths = [
            "/assets/leak.py",
            "/assets/debug.log",
            "/assets/notes.txt",
            "/assets/.env",
            "/assets/.hidden.json",
            "/assets/sub/.secret.png",
            "/assets/.dotdir/ok.png",
            "/assets/../pet_state.json",
            "/assets/%2e%2e/pet_state.json",
            "/assets/..\\pet_state.json",
            "/assets/%5c..%5ccolab_backend.py",
            "/assets/",
            "/assets/sub/",
            "/assets/idle.webp%00.png",
        ]
        for bad_path in forbidden_paths:
            st, _, body = _get(bad_path)
            assert st in (403, 404), f"非法资产路径未被拒绝: {bad_path} -> HTTP {st}"
            assert st == 403, f"预期返回 403 拒绝非法/越界路径: {bad_path} -> HTTP {st}"
            assert b"SECRET_" not in body and b"secret_state" not in body and b"API_KEY" not in body

        # 5c. 合法后缀但文件不存在 -> 404
        st_missing, _, _ = _get("/assets/not_exist.png")
        assert st_missing == 404
    finally:
        httpd.shutdown()
        httpd.server_close()
        th.join(timeout=2.0)

    # 5d. 真实包内 ASSETS_DIR 兼容性验收（idle.webp / idle.png 回退 / pet.ico / alpha_masks.json）
    httpd_real = pet_app._ThreadingHTTPServer(("127.0.0.1", 0), pet_app._PetAssetHandler)
    real_port = httpd_real.server_address[1]
    th_real = threading.Thread(target=httpd_real.serve_forever, daemon=True)
    th_real.start()
    try:
        for ok_path in ("/assets/idle.webp", "/assets/idle.png", "/assets/pet.ico", "/assets/alpha_masks.json"):
            conn = http.client.HTTPConnection("127.0.0.1", real_port, timeout=5)
            conn.request("GET", ok_path)
            resp = conn.getresponse()
            body = resp.read()
            conn.close()
            assert resp.status == 200 and len(body) > 0, ok_path

        conn = http.client.HTTPConnection("127.0.0.1", real_port, timeout=5)
        conn.request("GET", "/assets/../colab_backend.py")
        resp = conn.getresponse()
        resp.read()
        conn.close()
        assert resp.status == 403
    finally:
        httpd_real.shutdown()
        httpd_real.server_close()
        th_real.join(timeout=2.0)


# =========================================================================
# 6. 并发非阻塞遥测与桥接采样锁异常安全回归测试
# =========================================================================


def test_sse_gateway_metrics_nonblocking_and_cache_lock(
    isolated_env: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 SSE 网关 `/__pet_metrics` 走 asyncio.to_thread 非阻塞且缓存锁绝不包住子进程 IO。"""
    _ = isolated_env

    # 6a. 校验 _METRICS_CACHE_LOCK 保护缓存读写，且执行 nvidia-smi 子进程期间绝不持锁
    with sse_gw._METRICS_CACHE_LOCK:
        sse_gw._METRICS_CACHE["ts"] = 0.0
        sse_gw._METRICS_CACHE["data"] = None

    subproc_calls: list[bool] = []

    def _fake_check_output(*_args: Any, **_kwargs: Any) -> str:
        subproc_calls.append(sse_gw._METRICS_CACHE_LOCK.locked())
        return "NVIDIA L4, 4096, 24576\n"

    monkeypatch.setattr(sse_gw.shutil, "which", lambda name: "/usr/bin/nvidia-smi" if name == "nvidia-smi" else None)
    monkeypatch.setattr(sse_gw.subprocess, "check_output", _fake_check_output)

    m1 = sse_gw._collect_gateway_host_metrics()
    assert m1["ok"] is True and m1["gpu_full_name"] == "NVIDIA L4" and m1["vram_used_gb"] == 4.0
    assert subproc_calls == [False], "严禁在持有 _METRICS_CACHE_LOCK 期间调用 subprocess.check_output"
    assert sse_gw._METRICS_CACHE_LOCK.locked() is False

    # 3 秒缓存窗口内再次调用应直接命中缓存，不再触发子进程
    m2 = sse_gw._collect_gateway_host_metrics()
    assert m2["vram_used_gb"] == 4.0
    assert len(subproc_calls) == 1

    # 6b. 真实异步网关端到端：/__pet_metrics 慢采集 (0.15s) 期间并发 /health 必须快速返回不被阻塞
    entered_evt = threading.Event()

    def _slow_metrics() -> dict[str, Any]:
        entered_evt.set()
        time.sleep(0.15)
        return {"ok": True, "gpu_full_name": "NVIDIA L4", "ram_pct": 33.5}

    monkeypatch.setattr(sse_gw, "_collect_gateway_host_metrics", _slow_metrics)
    monkeypatch.setenv("SSE_API_KEY", "sk-metrics-nonblocking-test")

    async def _run_concurrent_check() -> None:
        def _free_port() -> int:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", 0))
                return int(s.getsockname()[1])

        up_port = _free_port()
        gw_port = _free_port()

        async def _up_health(_r: web.Request) -> web.Response:
            return web.json_response({"status": "ok"})

        up_app = web.Application()
        up_app.router.add_get("/health", _up_health)
        runner = web.AppRunner(up_app)
        await runner.setup()
        await web.TCPSite(runner, "127.0.0.1", up_port).start()

        monkeypatch.setattr(sse_gw, "LISTEN_PORT", gw_port)
        monkeypatch.setattr(sse_gw, "UPSTREAM_PORT", up_port)
        gw_srv = await asyncio.start_server(sse_gw._handle_client, "127.0.0.1", gw_port)
        try:
            async def _raw_http_get(path: str, auth: str = "") -> bytes:
                reader, writer = await asyncio.open_connection("127.0.0.1", gw_port)
                try:
                    auth_line = f"Authorization: Bearer {auth}\r\n" if auth else ""
                    req = f"GET {path} HTTP/1.1\r\nHost: 127.0.0.1\r\n{auth_line}Connection: close\r\n\r\n"
                    writer.write(req.encode("ascii"))
                    await writer.drain()
                    return await reader.read()
                finally:
                    writer.close()
                    await writer.wait_closed()

            metrics_task = asyncio.create_task(_raw_http_get("/__pet_metrics", auth="sk-metrics-nonblocking-test"))
            deadline = time.monotonic() + 2.0
            while not entered_evt.is_set() and time.monotonic() < deadline:
                await asyncio.sleep(0.005)
            assert entered_evt.is_set(), "/__pet_metrics 未进入工作线程采集桩"

            t0 = time.monotonic()
            health_raw = await _raw_http_get("/health")
            health_dt = time.monotonic() - t0
            assert b"200 OK" in health_raw and b'"ok"' in health_raw
            assert metrics_task.done() is False, "慢遥测阻塞了事件循环导致 /health 无法并发先行完成"
            assert health_dt < 0.12, f"/health 被慢遥测阻塞，耗时异常偏高: {health_dt:.3f}s"

            metrics_raw = await metrics_task
            assert b"200 OK" in metrics_raw
            body_txt = metrics_raw.split(b"\r\n\r\n", 1)[1].decode("utf-8")
            assert json.loads(body_txt)["ram_pct"] == 33.5
        finally:
            gw_srv.close()
            await gw_srv.wait_closed()
            await runner.cleanup()

    asyncio.run(_run_concurrent_check())


def test_bridge_node_host_metrics_nonblocking(
    isolated_env: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 colab_pet_bridge_node 的 `/colab_pet/host_metrics` 走 asyncio.to_thread 不阻塞 `/__pet_health`。"""
    from aiohttp.test_utils import TestClient, TestServer

    import colab_pet_bridge_node as bridge_node

    _ = isolated_env
    auth_key = "cpk-bridge-nonblocking-key-1234"
    entered_evt = threading.Event()

    def _slow_host_metrics() -> dict[str, Any]:
        entered_evt.set()
        time.sleep(0.15)
        return {"ok": True, "ram_pct": 41.2, "disk_pct": 18.5}

    monkeypatch.setattr(bridge_node, "_collect_host_metrics", _slow_host_metrics)
    monkeypatch.setattr(bridge_node, "_load_comfy_auth_key", lambda: auth_key)

    async def _run_bridge_check() -> None:
        app = web.Application()
        routes = web.RouteTableDef()
        fake_ps = SimpleNamespace(app=app, routes=routes, sockets={})
        fake_server_mod = SimpleNamespace(PromptServer=SimpleNamespace(instance=fake_ps))
        monkeypatch.setattr(bridge_node, "server", fake_server_mod)
        bridge_node._register_routes()
        app.add_routes(routes)

        async with TestClient(TestServer(app)) as client:
            pub_hdrs = {"Host": "demo-pet.trycloudflare.com", "X-Forwarded-Proto": "https"}
            auth_hdrs = {**pub_hdrs, "Authorization": f"Bearer {auth_key}"}

            metrics_task = asyncio.create_task(client.get("/colab_pet/host_metrics", headers=auth_hdrs))
            deadline = time.monotonic() + 2.0
            while not entered_evt.is_set() and time.monotonic() < deadline:
                await asyncio.sleep(0.005)
            assert entered_evt.is_set(), "/colab_pet/host_metrics 未进入工作线程采集桩"

            t0 = time.monotonic()
            r_health = await client.get("/__pet_health", headers=pub_hdrs)
            health_dt = time.monotonic() - t0
            assert r_health.status == 200
            assert (await r_health.json())["status"] == "ok"
            assert metrics_task.done() is False, "慢 /colab_pet/host_metrics 阻塞了 aiohttp 事件循环"
            assert health_dt < 0.12, f"/__pet_health 响应被阻塞，耗时异常偏高: {health_dt:.3f}s"

            r_metrics = await metrics_task
            assert r_metrics.status == 200
            m_data = await r_metrics.json()
            assert m_data["ok"] is True and m_data["ram_pct"] == 41.2

    asyncio.run(_run_bridge_check())


def test_comfy_bridge_host_metrics_thread_start_error_unlocks_and_throttles(
    isolated_env: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证桥接遥测线程启动失败时 `_bridge_hm_lock` 必释放且不阻塞后续触发，30s 节流期内不重复起线程。"""
    import backend_comfy as bc_mod

    backend = _make_backend(isolated_env, "bridge_hm_lock_state")
    svc = backend._comfy_svc

    fake_py = isolated_env.tmp / "fake_colab_py.exe"
    fake_py.write_bytes(b"MZ")
    backend._get_colab_py = lambda: fake_py  # type: ignore[method-assign]
    with backend._lock:
        backend.state["prefs"]["active_account_id"] = "acct_hm"
        backend.state["last_sessions"] = [{"name": "comfy", "endpoint": "ep-hm-001", "accelerator": "L4"}]
        backend.state["active_endpoint"] = "ep-hm-001"

    real_thread_cls = bc_mod.threading.Thread
    start_attempts: list[int] = []
    fail_once = {"flag": True}

    class _ControlledThread:
        def __init__(self, target=None, daemon=None, **kwargs: Any) -> None:
            self._inner = real_thread_cls(target=target, daemon=daemon, **kwargs)

        def start(self) -> None:
            start_attempts.append(1)
            if fail_once["flag"]:
                fail_once["flag"] = False
                raise RuntimeError("simulated thread start failure")
            self._inner.start()

    monkeypatch.setattr(bc_mod.threading, "Thread", _ControlledThread)

    # 1) Thread.start() 抛 RuntimeError 时，必须释放 _bridge_hm_lock 且回滚未生效节流时间戳
    with pytest.raises(RuntimeError, match="simulated thread start failure"):
        svc._maybe_trigger_bridge_host_metrics_async("https://demo-hm.trycloudflare.com", "comfy")
    assert svc._bridge_hm_lock.locked() is False, "Thread.start() 抛异常后 _bridge_hm_lock 未释放导致死锁"
    assert len(start_attempts) == 1

    # 2) 后续立刻再次触发应能正常启动工作线程并写入指标，结束后锁被 _worker 唯一释放
    worker_done = threading.Event()
    subproc_calls: list[list[str]] = []

    def _fake_bridge_run(cmd, **_kwargs):
        subproc_calls.append([str(x) for x in cmd])
        payload = json.dumps(
            {
                "ok": True,
                "disk_used_gb": 25.0,
                "disk_total_gb": 100.0,
                "disk_free_gb": 75.0,
                "disk_pct": 25.0,
                "ram_used_gb": 8.0,
                "ram_total_gb": 32.0,
                "ram_free_gb": 24.0,
                "ram_pct": 25.0,
            }
        )
        worker_done.set()
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout=f"<<<COLAB_HOST_METRICS>>>{payload}<<<END_COLAB_HOST_METRICS>>>",
            stderr="",
        )

    with patch.object(cb.subprocess, "run", side_effect=_fake_bridge_run):
        svc._maybe_trigger_bridge_host_metrics_async("https://demo-hm.trycloudflare.com", "comfy")
        assert worker_done.wait(timeout=3.0), "失败恢复后的桥接遥测工作线程未运行"
        _wait_until(
            lambda: (not svc._bridge_hm_lock.locked()) and backend.get_comfy_state().get("disk_used_gb") == 25.0,
            timeout=3.0,
            err_msg="工作线程完成后未释放 _bridge_hm_lock 或未写入指标",
        )
        assert len(start_attempts) == 2
        assert len(subproc_calls) == 1

        # 3) 30s 节流期内连续多次调用：不得重复拉起线程，且每次提前返回均正确释放锁
        for _ in range(3):
            svc._maybe_trigger_bridge_host_metrics_async("https://demo-hm.trycloudflare.com", "comfy")
            assert svc._bridge_hm_lock.locked() is False
        assert len(start_attempts) == 2
        assert len(subproc_calls) == 1
