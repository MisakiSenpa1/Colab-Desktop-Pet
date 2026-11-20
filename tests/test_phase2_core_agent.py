"""Phase 2 路 1：`colab_core/agent/` 子包拆解与 `colab_core/agent_tools.py` 门面回归单测。

说明：全量基于 `COLAB_PET_MOCK_VAULT_PATH` 与本地回环 HTTP 桩隔离，严禁触碰宿主机真实 Windows 凭据库或外网。
"""

from __future__ import annotations

import http.server
import io
import json
import socketserver
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

import colab_security as csec  # noqa: E402
from colab_core import agent as core_agent  # noqa: E402
from colab_core import agent_tools  # noqa: E402
from colab_core.agent import drivers, mcp_server, protocols, provider_probe, safety, specs  # noqa: E402


class _MockHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


@pytest.fixture()
def isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """为每条用例创建隔离凭据保险箱与干净环境变量沙箱。"""
    csec.clear_ephemeral_secrets()
    vault_file = tmp_path / "mock_vault.json"
    cf_file = tmp_path / ".cf_tunnel_secret.json"
    diag_log = tmp_path / "logs" / "colab_diag.log"
    monkeypatch.setenv("COLAB_PET_MOCK_VAULT_PATH", str(vault_file))
    monkeypatch.setenv("COLAB_PET_CF_CONFIG_PATH", str(cf_file))
    monkeypatch.setenv("COLAB_PET_DIAG_LOG_PATH", str(diag_log))
    monkeypatch.delenv("COLAB_MCP_ALLOW_MUTATIONS", raising=False)
    yield SimpleNamespace(tmp=tmp_path, vault_file=vault_file, cf_file=cf_file, diag_log=diag_log)
    csec.clear_ephemeral_secrets()


def _start_sse_stub(handler_fn):
    """启动本地临时端口 HTTP 桩用于验证四协议流式响应解析。"""
    recorded: list[dict[str, Any]] = []

    class _Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: Any) -> None:
            _ = (format, args)

        def do_POST(self) -> None:  # noqa: N802
            ln = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(ln) if ln > 0 else b"{}"
            req_info = {
                "path": self.path,
                "headers": {k.lower(): v for k, v in self.headers.items()},
                "body": json.loads(raw.decode("utf-8", errors="replace") or "{}"),
            }
            recorded.append(req_info)
            handler_fn(self, req_info)

    srv = _MockHTTPServer(("127.0.0.1", 0), _Handler)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()

    def _close() -> None:
        srv.shutdown()
        srv.server_close()
        th.join(timeout=2.0)

    return srv.server_address[1], recorded, _close


def _write_sse(handler: http.server.BaseHTTPRequestHandler, frames: list[str]) -> None:
    handler.send_response(200)
    handler.send_header("Content-Type", "text/event-stream; charset=utf-8")
    handler.send_header("Connection", "close")
    handler.end_headers()
    for frame in frames:
        handler.wfile.write(frame.encode("utf-8"))
        handler.wfile.flush()


# =========================================================================
# 1. 门面符号完整性与 subprocess patch 双向兼容性
# =========================================================================


def test_agent_tools_facade_symbols_and_subprocess_patch_compat(isolated_env: SimpleNamespace) -> None:
    """校验 agent_tools 门面导出全部必需符号，且 patch agent_tools.subprocess 仍能精准拦截 MCP 执行器。"""
    required_symbols = (
        "AGENT_TOOL_SPECS",
        "AGENT_TOOL_MAP",
        "ClaudeCodeDriver",
        "DEFAULT_SYSTEM_PROMPT",
        "NativeLightDriver",
        "SUPPORTED_API_FORMATS",
        "_default_mcp_executor",
        "check_remote_code_safety",
        "detect_provider_protocol",
        "fetch_provider_models",
        "get_tool_spec",
        "is_loopback_base_url",
        "is_readonly_safe_remote_cmd",
        "probe_provider_models",
        "run_mcp_stdio_server",
        "subprocess",
        "url_origin_key",
    )
    for sym in required_symbols:
        assert hasattr(agent_tools, sym), f"agent_tools 门面缺少符号: {sym}"
        assert sym in agent_tools.__all__, f"agent_tools.__all__ 缺少导出项: {sym}"

    # 校验门面与子包模块对象严格同一引用
    assert agent_tools.AGENT_TOOL_SPECS is specs.AGENT_TOOL_SPECS
    assert agent_tools.check_remote_code_safety is safety.check_remote_code_safety
    assert agent_tools.is_readonly_safe_remote_cmd is safety.is_readonly_safe_remote_cmd
    assert agent_tools.is_loopback_base_url is protocols.is_loopback_base_url
    assert agent_tools.fetch_provider_models is provider_probe.fetch_provider_models
    assert agent_tools.NativeLightDriver is drivers.NativeLightDriver
    assert agent_tools.ClaudeCodeDriver is drivers.ClaudeCodeDriver
    assert agent_tools._default_mcp_executor is mcp_server._default_mcp_executor
    assert agent_tools.run_mcp_stdio_server is mcp_server.run_mcp_stdio_server
    assert core_agent.PROTOCOL_ADAPTERS is protocols.PROTOCOL_ADAPTERS

    # 1a. patch.object(agent_tools.subprocess, "run", ...) 兼容验证
    fake_ctl = isolated_env.tmp / "fake_ctl.py"
    fake_ctl.write_text("# stub\n", encoding="utf-8")
    seen_cmds: list[list[str]] = []

    def _mock_run(cmd, **_kwargs):
        seen_cmds.append([str(x) for x in cmd])
        return subprocess.CompletedProcess(cmd, 0, stdout="FACADE_PATCH_OK", stderr="")

    with patch.object(agent_tools.subprocess, "run", side_effect=_mock_run):
        exec_fn = agent_tools._default_mcp_executor(fake_ctl)
        res = exec_fn("exec_remote_colab", {"code": "nvidia-smi", "mode": "bash"})
    assert res["ok"] is True and res["stdout"] == "FACADE_PATCH_OK"
    assert len(seen_cmds) == 1 and "exec" in seen_cmds[0]

    # 1b. patch.object(agent_tools, "subprocess", fake_module) 整模块替换兼容验证
    mod_cmds: list[list[str]] = []
    fake_sub_mod = SimpleNamespace(
        run=lambda cmd, **_kw: (
            mod_cmds.append([str(x) for x in cmd]),
            subprocess.CompletedProcess(cmd, 0, stdout="MOD_REPLACE_OK", stderr=""),
        )[1]
    )
    with patch.object(agent_tools, "subprocess", fake_sub_mod):
        exec_fn2 = agent_tools._default_mcp_executor(fake_ctl)
        res2 = exec_fn2("get_cluster_and_pool_status", {})
    assert res2["ok"] is True and res2["stdout"] == "MOD_REPLACE_OK"
    assert len(mod_cmds) == 1 and "status" in mod_cmds[0]


# =========================================================================
# 2. 协议适配器策略表、工具回灌与 ThinkTagSplitter 单元验证
# =========================================================================


def test_protocol_adapters_strategy_and_think_splitter(isolated_env: SimpleNamespace) -> None:
    """校验 4 套协议适配器的端点、头、工具 Schema、思考切分与多轮回灌逻辑。"""
    _ = isolated_env
    assert set(protocols.PROTOCOL_ADAPTERS.keys()) == set(protocols.SUPPORTED_API_FORMATS)
    assert protocols.get_protocol_adapter("unknown_fmt") is protocols.PROTOCOL_ADAPTERS["openai"]

    sample_specs = (specs.AGENT_TOOL_SPECS[0],)
    raw_msgs = [{"role": "system", "content": "补充指令"}, {"role": "user", "content": "查状态"}]

    # OpenAI Chat
    oai = protocols.get_protocol_adapter("openai")
    assert oai.build_endpoint("http://127.0.0.1:3425", "gpt-4o") == "http://127.0.0.1:3425/v1/chat/completions"
    assert oai.build_headers("sk-oai", {})["Authorization"] == "Bearer sk-oai"
    assert oai.build_tools(sample_specs)[0]["type"] == "function"
    _, oai_msgs, _ = oai.init_conversation(raw_msgs, "主提示")
    assert oai_msgs[0]["role"] == "system" and "主提示" in oai_msgs[0]["content"] and "补充指令" in oai_msgs[0]["content"]

    # OpenAI Responses
    resp_adp = protocols.get_protocol_adapter("openai_responses")
    assert resp_adp.build_endpoint("http://127.0.0.1:3425/v1/chat/completions", "gpt-5") == "http://127.0.0.1:3425/v1/responses"
    sys_r, r_items, _ = resp_adp.init_conversation(raw_msgs, "主提示")
    assert "主提示" in sys_r and "补充指令" in sys_r and r_items == [{"role": "user", "content": "查状态"}]

    # Anthropic：验证带 signature 保留与缺 signature 自动降级 allow_thinking=False
    anth = protocols.get_protocol_adapter("anthropic")
    assert anth.build_endpoint("http://127.0.0.1:3425", "claude-sonnet-4-5") == "http://127.0.0.1:3425/v1/messages"
    anth_hdrs = anth.build_headers("sk-ant", {})
    assert anth_hdrs["x-api-key"] == "sk-ant" and anth_hdrs["anthropic-version"] == "2023-06-01"
    _, anth_msgs, anth_state = anth.init_conversation(raw_msgs, "主提示")
    assert anth_state["allow_thinking"] is True
    rec = protocols.ToolExecRecord(
        call_id="tu_1",
        name="get_cluster_and_pool_status",
        raw_args="{}",
        args_dict={},
        res_json='{"ok":true}',
        ok=True,
    )
    anth.append_tool_round(
        anth_msgs,
        protocols.TurnStreamResult(text="", tool_calls=[{"id": "tu_1", "name": "get_cluster_and_pool_status", "arguments": {}}], reasoning="纯思考但无签名"),
        [rec],
        anth_state,
        0,
    )
    assert anth_state["allow_thinking"] is False
    assert anth_msgs[-2]["role"] == "assistant" and anth_msgs[-1]["role"] == "user"

    # Gemini：验证端点、thinkingLevel 判定与 thoughtSignature 回灌
    gem = protocols.get_protocol_adapter("gemini")
    assert ":streamGenerateContent?alt=sse" in gem.build_endpoint("http://127.0.0.1:3425", "gemini-4-hydrogen")
    assert gem.build_headers("AIzaTest", {})["x-goog-api-key"] == "AIzaTest"
    _, gem_msgs, gem_state = gem.init_conversation(raw_msgs, "主提示")
    rec_gem = protocols.ToolExecRecord(
        call_id="call_gem_0_0",
        name="get_cluster_and_pool_status",
        raw_args={},
        args_dict={},
        res_json='{"ok":true}',
        ok=True,
        meta={"thought_signature": "tsig_xyz"},
    )
    gem.append_tool_round(gem_msgs, protocols.TurnStreamResult(text="说明"), [rec_gem], gem_state, 0)
    assert gem_msgs[-2]["role"] == "model" and gem_msgs[-2]["parts"][1]["thoughtSignature"] == "tsig_xyz"
    assert gem_msgs[-1]["role"] == "user" and "functionResponse" in gem_msgs[-1]["parts"][0]

    # _ThinkTagSplitter 分片、直通与未闭合 flush 验证
    sp1 = protocols._ThinkTagSplitter()
    out1 = sp1.feed("<thi") + sp1.feed("nk>\n思考A") + sp1.feed("</thi") + sp1.feed("nk>\n正文B") + sp1.flush()
    assert "".join(v for k, v in out1 if k == "reasoning") == "思考A"
    assert "".join(v for k, v in out1 if k == "delta") == "正文B"

    sp2 = protocols._ThinkTagSplitter()
    assert sp2.feed("普通正文") == [("delta", "普通正文")]
    assert sp2.flush() == []

    sp3 = protocols._ThinkTagSplitter()
    out3 = sp3.feed("<think>未闭合思考") + sp3.flush()
    assert "".join(v for k, v in out3 if k == "reasoning") == "未闭合思考"

    # 真实回环 SSE 验证 NativeLightDriver 走适配器完整跑通工具调用闭环
    turn_cnt = {"n": 0}

    def _sse_handler(handler, req_info):
        turn_cnt["n"] += 1
        if turn_cnt["n"] == 1:
            _write_sse(
                handler,
                [
                    'data: {"choices":[{"delta":{"reasoning_content":"先查一下"}}]}\n\n',
                    'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","type":"function","function":{"name":"get_cluster_and_pool_status","arguments":"{}"}}]}}]}\n\n',
                    "data: [DONE]\n\n",
                ],
            )
            return
        assert any(m.get("role") == "tool" and m.get("tool_call_id") == "c1" for m in req_info["body"]["messages"])
        _write_sse(
            handler,
            [
                'data: {"choices":[{"delta":{"content":"一切正常。"}}]}\n\n',
                "data: [DONE]\n\n",
            ],
        )

    port, _, close_srv = _start_sse_stub(_sse_handler)
    try:
        evts: list[dict[str, Any]] = []
        drv = drivers.NativeLightDriver(
            base_url=f"http://127.0.0.1:{port}/v1",
            model="gpt-4o-mini",
            api_format="openai",
            tool_executor=lambda n, _a: {"ok": True, "tool": n},
            on_event=evts.append,
        )
        res = drv.run([{"role": "user", "content": "检查集群"}])
        assert res["ok"] is True and res["tool_calls_count"] == 1
        assert res["reasoning"] == "先查一下" and res["text"] == "一切正常。"
        assert any(e["type"] == "tool_end" and e["ok"] is True for e in evts)
    finally:
        close_srv()


# =========================================================================
# 3. 表驱动只读白名单校验器完整性与绕过拦截测试
# =========================================================================


def test_table_driven_readonly_cmd_validators(isolated_env: SimpleNamespace) -> None:
    """校验 _READONLY_CMD_VALIDATORS 分派表覆盖全部只读命令并保持严格绕过拦截。"""
    _ = isolated_env
    expected_cmds = {
        "nvidia-smi",
        "free",
        "df",
        "uptime",
        "uname",
        "ps",
        "ls",
        "head",
        "tail",
        "du",
        "pwd",
        "whoami",
        "nproc",
        "lscpu",
        "date",
    }
    assert expected_cmds.issubset(set(safety._READONLY_CMD_VALIDATORS.keys()))

    for ok_cmd in (
        "nvidia-smi --query-gpu=name --format=csv,noheader",
        "free -m",
        "df -h /content",
        "uptime -p",
        "uname -sr",
        "ps -ef",
        "ps aux",
        "ps -e -o pid,comm",
        "ls -lh /content/ComfyUI",
        "head -n 20 /root/llama_server.log",
        "tail -n 30 /root/comfy_colab.log",
        "du -sh /content/ComfyUI",
        "pwd && whoami && nproc && date -u",
    ):
        ok, reason = safety.is_readonly_safe_remote_cmd(ok_cmd, "bash")
        assert ok is True and reason == "", f"表驱动校验误拦合法只读命令: {ok_cmd!r} -> {reason}"

    for bad_cmd, bad_mode in (
        ("ps -auxe", "bash"),
        ("ps -xe", "bash"),
        ("ps -eux", "bash"),
        ("ps aux -e", "bash"),
        ("ps -e aux", "bash"),
        ("ps aux -eo pid,comm", "bash"),
        ("nvidia-smi -f /tmp/out.txt", "bash"),
        ("nvidia-smi --power-limit=150", "bash"),
        ("ls -la", "bash"),
        ("ls /tmp/rc*", "bash"),
        ("tail -n 999 /root/comfy_colab.log", "bash"),
        ("du -sh /etc", "bash"),
        ("pwd /tmp", "bash"),
        ("date +%s; id", "bash"),
        ("print(1)", "python"),
    ):
        ok, reason = safety.is_readonly_safe_remote_cmd(bad_cmd, bad_mode)
        assert ok is False and reason, f"表驱动校验未拦截违规命令: ({bad_mode}) {bad_cmd!r}"


# =========================================================================
# 4. MCP JSON-RPC 2.0 错误分支与 ClaudeCodeDriver 配置构造测试
# =========================================================================


def test_mcp_server_jsonrpc_and_claude_driver_helpers(isolated_env: SimpleNamespace) -> None:
    """校验 MCP Stdio JSON-RPC 2.0 协议分发与 ClaudeCodeDriver 命令/MCP 配置拆分方法。"""
    fake_ctl = isolated_env.tmp / "fake_ctl.py"
    fake_ctl.write_text("# stub\n", encoding="utf-8")

    rpc_in = io.StringIO(
        "\n".join(
            [
                "{not-json",
                json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05"}}),
                json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
                json.dumps({"jsonrpc": "2.0", "id": 2, "method": "ping"}),
                json.dumps({"jsonrpc": "2.0", "id": 3, "method": "tools/list"}),
                json.dumps({"jsonrpc": "2.0", "id": 4, "method": "unknown/method"}),
                json.dumps({"jsonrpc": "2.0", "id": 5, "method": "tools/call", "params": {"name": "no_such_tool"}}),
                json.dumps({"jsonrpc": "2.0", "id": 6, "method": "tools/call", "params": {"name": "sync_and_pull_outputs"}}),
            ]
        )
        + "\n"
    )
    rpc_out = io.StringIO()
    assert mcp_server.run_mcp_stdio_server(fake_ctl, stdin=rpc_in, stdout=rpc_out) == 0

    replies = [json.loads(ln) for ln in rpc_out.getvalue().splitlines() if ln.strip()]
    assert len(replies) == 7
    assert replies[0]["error"]["code"] == -32700
    assert replies[1]["result"]["serverInfo"]["name"] == "colab-studio-mcp"
    assert replies[2]["result"] == {}
    assert len(replies[3]["result"]["tools"]) == 10
    assert replies[4]["error"]["code"] == -32601
    assert replies[5]["error"]["code"] == -32602
    assert replies[6]["result"]["isError"] is True
    blk_body = json.loads(replies[6]["result"]["content"][0]["text"])
    assert blk_body["blocked"] is True and blk_body["approval_required"] is True

    # ClaudeCodeDriver._prepare_mcp_and_cmd 校验
    cfg_default = isolated_env.tmp / "mcp_default.json"
    drv_default = drivers.ClaudeCodeDriver(ctl_script=fake_ctl, claude_bin="fake-claude")
    cmd_1, use_stdin_1 = drv_default._prepare_mcp_and_cmd(cfg_default, "fake-claude", "sonnet", "第一行\n第二行", "单行提示")
    assert use_stdin_1 is False and "单行提示" in cmd_1
    assert "第一行 第二行" in cmd_1
    assert "env" not in json.loads(cfg_default.read_text(encoding="utf-8"))["mcpServers"]["colab-studio"]

    cfg_mut = isolated_env.tmp / "mcp_mut.json"
    drv_mut = drivers.ClaudeCodeDriver(ctl_script=fake_ctl, claude_bin="fake-claude", allow_mcp_mutations=True)
    cmd_2, use_stdin_2 = drv_mut._prepare_mcp_and_cmd(cfg_mut, "fake-claude", "", "", "第一行\n第二行")
    assert use_stdin_2 is True and "第一行\n第二行" not in cmd_2
    assert json.loads(cfg_mut.read_text(encoding="utf-8"))["mcpServers"]["colab-studio"]["env"] == {
        "COLAB_MCP_ALLOW_MUTATIONS": "1"
    }
