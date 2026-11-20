"""Colab Studio「同进程轻量双引擎 AI 副驾」全链路安全、协议、MCP 与真渲染验收套件。

说明：全量基于隔离临时沙箱目录与环境变量（COLAB_PET_MOCK_VAULT_PATH、
COLAB_PET_CF_CONFIG_PATH、COLAB_PET_DIAG_LOG_PATH、COLAB_PET_MOCK_STATE_PATH），
绝不读取或修改宿主机真实凭据库、`.cf_tunnel_secret.json` 或 `~/.config/colab-cli/sessions.json`。
"""

from __future__ import annotations

import argparse
import base64
import http.server
import inspect
import json
import os
import re
import socketserver
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence
from unittest.mock import patch

import pytest
from PIL import Image, ImageDraw
from PIL.PngImagePlugin import PngInfo

PKG_DIR = Path(__file__).resolve().parents[1]
PET_DIR = PKG_DIR / "colab_pet"
for _p in (str(PKG_DIR), str(PET_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import build_windows_package as win_pkg  # noqa: E402
import colab_backend as cb  # noqa: E402
import colab_security as csec  # noqa: E402
import pet_app  # noqa: E402
from colab_core import agent_tools  # noqa: E402
from colab_core.runtime import (  # noqa: E402
    DEFAULT_PREFS,
    CancelToken,
    DiagLogger,
    infer_model_reasoning_spec,
    sanitize_prefs_dict,
)

_EMOJI_RE = re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]")


class _ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    """多线程 HTTP 测试服务器（支持并发 SSE 流与快速关闭）。"""

    daemon_threads = True
    allow_reuse_address = True


@pytest.fixture()
def isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """为每条测试用例构建完全隔离的凭据保险箱、状态文件、隧道配置与诊断日志沙箱。"""
    csec.clear_ephemeral_secrets()
    vault_file = tmp_path / "mock_vault.json"
    cf_file = tmp_path / ".cf_tunnel_secret.json"
    diag_log = tmp_path / "logs" / "colab_diag.log"
    state_file = tmp_path / "pet_state.json"

    fake_rclone_exe = tmp_path / "sandbox_rclone.exe"
    fake_rclone_exe.write_bytes(b"MZ")
    unconf_rclone = tmp_path / "sandbox_rclone.conf"
    unconf_rclone.write_text("[other_remote]\ntype = local\n", encoding="utf-8")
    fake_colab_exe = tmp_path / "sandbox_colab.exe"
    fake_colab_exe.write_bytes(b"MZ")
    fake_colab_py = tmp_path / "sandbox_colab_py.exe"
    fake_colab_py.write_bytes(b"MZ")

    monkeypatch.setenv("COLAB_PET_MOCK_VAULT_PATH", str(vault_file))
    monkeypatch.setenv("COLAB_PET_CF_CONFIG_PATH", str(cf_file))
    monkeypatch.setenv("COLAB_PET_DIAG_LOG_PATH", str(diag_log))
    monkeypatch.setenv("COLAB_PET_MOCK_STATE_PATH", str(state_file))
    monkeypatch.setenv("COLAB_PET_RCLONE_EXE", str(fake_rclone_exe))
    monkeypatch.setenv("COLAB_PET_RCLONE_CONF", str(unconf_rclone))
    monkeypatch.setenv("COLAB_PET_COLAB_EXE", str(fake_colab_exe))
    monkeypatch.setenv("COLAB_PET_COLAB_PY", str(fake_colab_py))
    monkeypatch.delenv("COLAB_PET_ACCOUNT_ID", raising=False)

    yield SimpleNamespace(
        tmp=tmp_path,
        vault_file=vault_file,
        cf_file=cf_file,
        diag_log=diag_log,
        state_file=state_file,
        fake_colab_exe=fake_colab_exe,
        fake_colab_py=fake_colab_py,
        fake_rclone_exe=fake_rclone_exe,
        unconf_rclone=unconf_rclone,
    )
    pet_app._set_active_media_root(None)
    csec.clear_ephemeral_secrets()


def _make_backend(env_ns: SimpleNamespace, name: str = "agent_state") -> cb.ColabPetBackend:
    """实例化绑定到沙箱目录的 ColabPetBackend。"""
    return cb.ColabPetBackend(
        state_path=env_ns.tmp / f"{name}.json",
        download_root=env_ns.tmp / f"{name}_dl",
        cf_config_path=env_ns.cf_file,
    )


def _wait_until(predicate, *, timeout: float = 10.0, interval: float = 0.03, err_msg: str = "等待条件超时"):
    """轮询等待断言条件成立。"""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        val = predicate()
        if val:
            return val
        time.sleep(interval)
    raise AssertionError(err_msg)


def _create_comfy_artwork_png(
    out_path: Path,
    *,
    width: int = 768,
    height: int = 512,
    positive: str = "1girl, anime cel shading, lavender eyes, sunlit atelier, masterpiece",
    negative: str = "lowres, blurry, extra fingers, watermark",
    seed: int = 88991234,
    steps: int = 28,
) -> Path:
    """生成带完整 ComfyUI API prompt 元数据的测试 PNG 作品。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (width, height), (236, 229, 255))
    draw = ImageDraw.Draw(img)
    for y in range(height):
        t = y / max(1, height - 1)
        r = int(124 * (1.0 - t) + 236 * t)
        g = int(58 * (1.0 - t) + 229 * t)
        b = int(237 * (1.0 - t) + 255 * t)
        draw.line([(0, y), (width, y)], fill=(r, g, b))
    draw.ellipse([120, 90, 440, 390], fill=(255, 255, 255), outline=(124, 58, 237), width=4)

    prompt_graph = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "anima_v2_fp16.safetensors"}},
        "5": {"class_type": "EmptyLatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
        "20": {"class_type": "CLIPTextEncode", "inputs": {"text": positive, "clip": ["1", 1]}, "_meta": {"title": "Positive"}},
        "21": {"class_type": "CLIPTextEncode", "inputs": {"text": negative, "clip": ["1", 1]}, "_meta": {"title": "Negative"}},
        "30": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0],
                "positive": ["20", 0],
                "negative": ["21", 0],
                "latent_image": ["5", 0],
                "seed": seed,
                "steps": steps,
                "cfg": 4.5,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 1.0,
            },
        },
        "31": {"class_type": "VAEDecode", "inputs": {"samples": ["30", 0], "vae": ["1", 2]}},
        "32": {"class_type": "SaveImage", "inputs": {"images": ["31", 0], "filename_prefix": "CopilotTest"}},
    }
    png_info = PngInfo()
    png_info.add_text("prompt", json.dumps(prompt_graph, ensure_ascii=False))
    img.save(out_path, format="PNG", pnginfo=png_info)
    return out_path


def _start_dual_protocol_sse_server(scripted_handler):
    """启动支持 OpenAI /v1/chat/completions 与 Anthropic /v1/messages 的本地测试 SSE 服务。"""
    recorded_requests: list[dict[str, Any]] = []
    active_conns: list[Any] = []
    conns_lock = threading.Lock()

    class _Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: Any) -> None:
            return

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            raw_body = self.rfile.read(length) if length > 0 else b"{}"
            body = json.loads(raw_body.decode("utf-8", errors="replace") or "{}")
            req_info = {
                "path": self.path,
                "headers": {k.lower(): v for k, v in self.headers.items()},
                "body": body,
            }
            recorded_requests.append(req_info)
            with conns_lock:
                active_conns.append(self.connection)
            try:
                scripted_handler(self, req_info)
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            finally:
                with conns_lock:
                    if self.connection in active_conns:
                        active_conns.remove(self.connection)

    srv = _ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()

    def _close() -> None:
        with conns_lock:
            for c in list(active_conns):
                try:
                    c.close()
                except Exception:
                    pass
        try:
            srv.shutdown()
            srv.server_close()
        except Exception:
            pass

    return srv.server_address[1], recorded_requests, _close


def _send_sse_frames(handler: http.server.BaseHTTPRequestHandler, frames: list[str], delay_sec: float = 0.0) -> None:
    """按帧向客户端写入 HTTP 200 SSE 事件流。"""
    handler.send_response(200)
    handler.send_header("Content-Type", "text/event-stream; charset=utf-8")
    handler.send_header("Cache-Control", "no-cache")
    handler.send_header("Connection", "close")
    handler.end_headers()
    for frame in frames:
        handler.wfile.write(frame.encode("utf-8"))
        handler.wfile.flush()
        if delay_sec > 0:
            time.sleep(delay_sec)


def _send_json_resp(handler: http.server.BaseHTTPRequestHandler, code: int, obj: Any) -> None:
    """向客户端写入 HTTP JSON 响应。"""
    raw = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(raw)))
    handler.send_header("Connection", "close")
    handler.end_headers()
    handler.wfile.write(raw)


def _start_magpie_mock_gateway(
    *,
    models: Sequence[str] = ("gpt-4.1", "claude-sonnet-4-5", "gemini-2.5-pro", "qwen3-coder-plus", "broken-canary"),
    model_meta: dict[str, dict[str, Any]] | None = None,
    broken_models: set[str] | None = None,
    supported_post_formats: tuple[str, ...] = ("openai", "openai_responses", "anthropic", "gemini"),
    post_handler=None,
):
    """启动模拟本机 Magpie 网关的 HTTP 服务（支持免密 GET /v1/models 能力元数据及四协议 POST）。"""
    recorded: list[dict[str, Any]] = []
    bad_set = set(broken_models or ())
    meta_map = {str(k): dict(v) for k, v in (model_meta or {}).items() if isinstance(v, dict)}

    class _GatewayHandler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: Any) -> None:
            return

        def do_GET(self) -> None:
            hdrs = {k.lower(): v for k, v in self.headers.items()}
            recorded.append({"method": "GET", "path": self.path, "headers": hdrs, "body": None})
            if self.path.endswith("/v1/models") or self.path.endswith("/models"):
                data_rows = []
                for m in models:
                    row: dict[str, Any] = {"id": m, "object": "model", "owned_by": "magpie-local"}
                    if m in meta_map:
                        row.update(meta_map[m])
                    data_rows.append(row)
                _send_json_resp(
                    self,
                    200,
                    {
                        "object": "list",
                        "data": data_rows,
                    },
                )
                return
            if self.path.endswith("/v1beta/models"):
                _send_json_resp(
                    self,
                    200,
                    {"models": [{"name": f"models/{m}", "displayName": m} for m in models]},
                )
                return
            _send_json_resp(self, 404, {"error": {"message": f"Unknown GET path: {self.path}"}})

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            raw_body = self.rfile.read(length) if length > 0 else b"{}"
            body = json.loads(raw_body.decode("utf-8", errors="replace") or "{}")
            req_info = {
                "method": "POST",
                "path": self.path,
                "headers": {k.lower(): v for k, v in self.headers.items()},
                "body": body,
            }
            recorded.append(req_info)
            if post_handler is not None:
                post_handler(self, req_info)
                return

            mdl = str(body.get("model") or "").strip()
            if ":generateContent" in self.path or ":streamGenerateContent" in self.path:
                m_part = self.path.split("/models/", 1)[-1].split(":", 1)[0]
                mdl = mdl or m_part
            if mdl in bad_set:
                _send_json_resp(self, 404, {"error": {"message": f"Model {mdl} is offline in gateway"}})
                return

            if self.path.endswith("/v1/chat/completions") and "openai" in supported_post_formats:
                _send_json_resp(
                    self,
                    200,
                    {"id": "cmpl_magpie", "choices": [{"index": 0, "message": {"role": "assistant", "content": f"Magpie-{mdl}-OK"}}]},
                )
            elif self.path.endswith("/v1/responses") and "openai_responses" in supported_post_formats:
                _send_json_resp(
                    self,
                    200,
                    {
                        "id": "resp_magpie",
                        "object": "response",
                        "output": [{"type": "message", "content": [{"type": "output_text", "text": f"Resp-{mdl}-OK"}]}],
                    },
                )
            elif self.path.endswith("/v1/messages") and "anthropic" in supported_post_formats:
                _send_json_resp(
                    self,
                    200,
                    {"id": "msg_magpie", "type": "message", "role": "assistant", "content": [{"type": "text", "text": f"Anth-{mdl}-OK"}]},
                )
            elif (":generateContent" in self.path or ":streamGenerateContent" in self.path) and "gemini" in supported_post_formats:
                _send_json_resp(
                    self,
                    200,
                    {"candidates": [{"content": {"role": "model", "parts": [{"text": f"Gem-{mdl}-OK"}]}}]},
                )
            else:
                _send_json_resp(self, 404, {"error": {"message": f"Protocol mismatch at {self.path}"}})

    srv = _ThreadingHTTPServer(("127.0.0.1", 0), _GatewayHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    def _close() -> None:
        try:
            srv.shutdown()
            srv.server_close()
        except Exception:
            pass

    return srv.server_address[1], recorded, _close


# =========================================================================
# 1. 安全隔离、零密钥泄露与远端代码护栏校验
# =========================================================================


def test_security_vault_isolation_zero_leak_and_code_guardrails(isolated_env: SimpleNamespace) -> None:
    """校验 API Key 仅入 VaultStore 隔离、零落盘零回显，以及 check_remote_code_safety 强拦截危险脚本。"""
    backend = _make_backend(isolated_env, "sec_state")
    secret_a = "sk-proj-AgentSuperSecret9876543210XYZ"
    secret_b = "custom-nonprefix-agent-token-20261118"

    # 验证 prefs 清洗严禁保留 agent_api_key 且对枚举与布尔字段做严格归一化
    sanitized = sanitize_prefs_dict(
        {
            "agent_driver": "invalid_drv",
            "agent_brain_source": "colab_llm",
            "agent_api_format": "anthropic",
            "agent_api_base_url": "  https://api.example.com/v1  ",
            "agent_model": "  claude-sonnet-4  ",
            "agent_auto_approve_billing": "true",
            "agent_api_key": secret_a,
        }
    )
    assert "agent_api_key" not in sanitized
    assert sanitized["agent_driver"] == "native"
    assert sanitized["agent_brain_source"] == "colab_llm"
    assert sanitized["agent_api_format"] == "anthropic"
    assert sanitized["agent_api_base_url"] == "https://api.example.com/v1"
    assert sanitized["agent_model"] == "claude-sonnet-4"
    assert sanitized["agent_auto_approve_billing"] is True
    assert "agent_api_key" not in DEFAULT_PREFS

    # 1) save_agent_config 将明文密钥隔离写入 Mock VaultStore，绝不写入 pet_state.json
    for test_secret in (secret_a, secret_b):
        res = backend.save_agent_config(
            {
                "agent_driver": "native",
                "agent_brain_source": "external_api",
                "agent_api_format": "openai",
                "agent_api_base_url": "https://llm.sandbox.local/v1",
                "agent_model": "gpt-4o-mini",
                "api_key": test_secret,
            }
        )
        assert res["ok"] is True
        assert isolated_env.vault_file.is_file()
        assert csec.VaultStore().get_secret(csec.AGENT_VAULT_API_KEY) == test_secret

        state_disk_raw = backend.state_path.read_text(encoding="utf-8")
        assert test_secret not in state_disk_raw
        assert "agent_api_key" not in state_disk_raw

        agent_state = backend.get_agent_state()
        assert agent_state["config"]["has_api_key"] is True
        state_json = json.dumps(agent_state, ensure_ascii=False)
        assert test_secret not in state_json
        assert test_secret not in json.dumps(res, ensure_ascii=False)

        # 校验 diag_log 与工具返回均绝不泄露明文密钥及 JSON secret 字段
        DiagLogger().info("agent", f"test leak check token={test_secret}")
        log_txt = isolated_env.diag_log.read_text(encoding="utf-8")
        assert test_secret not in log_txt

        # 校验 _execute_tool 输出脱敏（含非前缀密钥与 JSON 格式 agent_api_key / x-api-key）
        leaky_payload = {
            "ok": True,
            "raw": f"Authorization: Bearer {test_secret}",
            "extra": test_secret,
            "json_str": json.dumps({"agent_api_key": "raw-inline-secret-value-1234", "x-api-key": "raw-x-api-key-5678"}),
        }
        with patch.object(backend._agent_svc, "_tool_list_gdrive_assets", return_value=leaky_payload):
            tool_out = backend._agent_svc._execute_tool(
                "list_gdrive_assets",
                {},
                CancelToken(),
                backend._agent_svc._collect_known_secrets(),
            )
        tool_out_txt = json.dumps(tool_out, ensure_ascii=False)
        assert test_secret not in tool_out_txt
        assert "raw-inline-secret-value-1234" not in tool_out_txt
        assert "raw-x-api-key-5678" not in tool_out_txt

    # 清空密钥后验证 has_api_key 回落为 False
    clr_res = backend.save_agent_config({"clear_api_key": True})
    assert clr_res["ok"] is True
    assert clr_res["config"]["has_api_key"] is False
    assert csec.VaultStore().get_secret(csec.AGENT_VAULT_API_KEY) is None

    # 2) check_remote_code_safety 强制拦截危险脚本
    dangerous_cases = [
        ("rm -rf /", "bash"),
        ("rm -rf /*", "bash"),
        ("echo ok && rm -rf --no-preserve-root /", "bash"),
        ("cat ~/.cf_tunnel_secret.json", "bash"),
        ("cat ~/.config/colab-cli/sessions.json", "bash"),
        ("cat mock_vault.json", "bash"),
        ("mkfs.ext4 /dev/sda1", "bash"),
        ("dd if=/dev/zero of=/dev/sda bs=1M", "bash"),
        ("shutdown -h now", "bash"),
        (":(){ :|:& };:", "bash"),
        ("import shutil; shutil.rmtree('/')", "python"),
        ("print(open('.cf_tunnel_secret.json').read())", "python"),
    ]
    for code_str, mode_str in dangerous_cases:
        safe_ok, reason = agent_tools.check_remote_code_safety(code_str, mode_str)
        assert safe_ok is False, f"危险命令未被拦截: {code_str}"
        assert reason

    # 合法诊断指令应允许通过
    assert agent_tools.check_remote_code_safety("nvidia-smi && df -h /", "bash") == (True, "")
    assert agent_tools.check_remote_code_safety("import torch; print(torch.cuda.is_available())", "python") == (True, "")

    # 验证 exec_remote_colab 工具在后端层同样拦截危险命令且不触发子进程
    blocked = backend._agent_svc._execute_tool(
        "exec_remote_colab",
        {"code": "rm -rf /*", "mode": "bash"},
        CancelToken(),
        (),
    )
    assert blocked["ok"] is False and blocked.get("blocked") is True


# =========================================================================
# 2. NativeLightDriver 双协议 SSE 工具闭环 + 计费审批挂起恢复 + 路由与取消
# =========================================================================


def test_native_driver_openai_tool_loop_confirm_approve_and_reject(isolated_env: SimpleNamespace) -> None:
    """校验 OpenAI 流式 tool_calls 分片、safe 直跑、confirm 审批挂起放行/拒绝全闭环。"""
    backend = _make_backend(isolated_env, "openai_loop")
    events: list[dict[str, Any]] = []
    unsub = backend.subscribe_events(lambda e: events.append(e))

    turn_counter = {"n": 0}
    start_calls: list[dict[str, Any]] = []
    stop_calls: list[int] = []
    backend.start_instance = lambda cfg=None: (start_calls.append(dict(cfg or {})), {"ok": True, "started": True, "gpu": "L4"})[1]  # type: ignore[method-assign]
    backend.stop_all_instances = lambda: (stop_calls.append(1), {"ok": True})[1]  # type: ignore[method-assign]

    def _oai_tc_sse(idx: int, tc_id: str | None, name_part: str | None, arg_part: str | None) -> str:
        fn_obj: dict[str, Any] = {}
        if name_part is not None:
            fn_obj["name"] = name_part
        if arg_part is not None:
            fn_obj["arguments"] = arg_part
        tc_obj: dict[str, Any] = {"index": idx, "function": fn_obj}
        if tc_id is not None:
            tc_obj["id"] = tc_id
            tc_obj["type"] = "function"
        return "data: " + json.dumps({"choices": [{"delta": {"tool_calls": [tc_obj]}}]}, ensure_ascii=False) + "\n\n"

    def _sse_handler(handler, req_info):
        turn_counter["n"] += 1
        msgs = req_info["body"].get("messages") or []
        if turn_counter["n"] == 1:
            # 第 1 轮：流式分片调用 safe 工具 get_cluster_and_pool_status
            _send_sse_frames(
                handler,
                [
                    _oai_tc_sse(0, "call_oai_1", "get_cluster_", '{"refresh'),
                    _oai_tc_sse(0, None, "and_pool_status", '_pool": false}'),
                    "data: [DONE]\n\n",
                ],
            )
            return
        if turn_counter["n"] == 2:
            # 确认第 1 轮工具结果已回灌，再发起 Tier-confirm 计费工具调用
            assert any(m.get("role") == "tool" and m.get("name") == "get_cluster_and_pool_status" for m in msgs)
            _send_sse_frames(
                handler,
                [
                    _oai_tc_sse(0, "call_oai_2", "manage_compute_instance", '{"action": "start", '),
                    _oai_tc_sse(0, None, None, '"profiles": ["anima"], "gpu": "L4"}'),
                    "data: [DONE]\n\n",
                ],
            )
            return
        # 第 3 轮：审批放行且 start_instance 成功后流式回复最终结论
        assert any(m.get("role") == "tool" and m.get("name") == "manage_compute_instance" for m in msgs)
        _send_sse_frames(
            handler,
            [
                "data: " + json.dumps({"choices": [{"delta": {"content": "已完成核验，"}}]}, ensure_ascii=False) + "\n\n",
                "data: " + json.dumps({"choices": [{"delta": {"content": "L4 实例已批准开机。"}}]}, ensure_ascii=False) + "\n\n",
                "data: [DONE]\n\n",
            ],
        )

    port, reqs, close_srv = _start_dual_protocol_sse_server(_sse_handler)
    try:
        api_key = "sk-openai-sse-secret-11223344"
        backend.save_agent_config(
            {
                "agent_driver": "native",
                "agent_brain_source": "external_api",
                "agent_api_format": "openai",
                "agent_api_base_url": f"http://127.0.0.1:{port}/v1",
                "agent_model": "gpt-4o-mini",
                "agent_auto_approve_billing": False,
                "api_key": api_key,
            }
        )

        # 场景 A：第 1 轮 safe 直跑 + 第 2 轮 confirm 挂起 -> 批准放行 -> turn_done
        send_res = backend.send_agent_message({"text": "帮我检查号池并启动 L4 Anima 实例"})
        assert send_res["ok"] is True and send_res["started"] is True

        confirm_evt = _wait_until(
            lambda: next((e for e in events if e.get("type") == "agent_event" and e.get("phase") == "confirm_required"), None),
            err_msg="未收到 manage_compute_instance 的 confirm_required 挂起事件",
        )
        confirm_id = confirm_evt["confirm"]["id"]
        assert confirm_evt["confirm"]["tool"] == "manage_compute_instance"
        assert backend.get_agent_state()["pending_confirm"]["id"] == confirm_id
        assert len(start_calls) == 0  # 批准前严禁真正触发开机

        resolve_res = backend.resolve_agent_confirm({"confirm_id": confirm_id, "approved": True})
        assert resolve_res["ok"] is True and resolve_res["approved"] is True

        done_evt = _wait_until(
            lambda: next((e for e in events if e.get("type") == "agent_event" and e.get("phase") == "turn_done"), None),
            err_msg="批准执行后未等待到 turn_done",
        )
        assert done_evt["ok"] is True and done_evt["cancelled"] is False
        assert "L4 实例已批准开机" in done_evt["text"]
        assert len(start_calls) == 1 and start_calls[0]["gpu"] == "L4"
        assert any(e.get("phase") == "tool_start" and e.get("name") == "get_cluster_and_pool_status" and e.get("tier") == "safe" for e in events)
        assert any(e.get("phase") == "tool_end" and e.get("name") == "manage_compute_instance" and e.get("ok") is True for e in events)
        assert reqs[0]["headers"].get("authorization") == f"Bearer {api_key}"

        # 场景 B：confirm 挂起 -> 拒绝 (approved=False) -> 取消执行且不调用 stop_all_instances
        events.clear()
        turn_counter["n"] = 0

        def _reject_handler(handler, _req_info):
            _send_sse_frames(
                handler,
                [
                    'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_stop_1","type":"function","function":{"name":"manage_compute_instance","arguments":"{\\"action\\": \\"stop_all\\"}"}}]}}]}\n\n',
                    "data: [DONE]\n\n",
                ],
            )

        # 替换为仅触发 stop_all 的 handler
        port_rej, _reqs_rej, close_rej = _start_dual_protocol_sse_server(_reject_handler)
        try:
            backend.save_agent_config({"agent_api_base_url": f"http://127.0.0.1:{port_rej}/v1"})
            events.clear()
            res2 = backend.send_agent_message({"text": "停止全部实例"})
            assert res2["ok"] is True

            cfm2 = _wait_until(
                lambda: next((e for e in events if e.get("type") == "agent_event" and e.get("phase") == "confirm_required"), None),
                err_msg="拒绝分支未收到 confirm_required",
            )
            rej_res = backend.resolve_agent_confirm({"confirm_id": cfm2["confirm"]["id"], "approved": False})
            assert rej_res["ok"] is True and rej_res["approved"] is False

            done2 = _wait_until(
                lambda: next((e for e in events if e.get("type") == "agent_event" and e.get("phase") == "turn_done"), None),
                err_msg="拒绝后未等待到 turn_done",
            )
            assert done2["ok"] is False and done2["cancelled"] is True
            assert len(stop_calls) == 0
            assert backend.get_agent_state()["busy"] is False

            # 场景 C：confirm 挂起期间主动 cancel_agent_turn() 立即唤醒并解除挂起
            events.clear()
            res3 = backend.send_agent_message({"text": "再次尝试停卡"})
            assert res3["ok"] is True
            _wait_until(
                lambda: next((e for e in events if e.get("type") == "agent_event" and e.get("phase") == "confirm_required"), None),
                err_msg="场景 C 未收到 confirm_required",
            )
            cancel_cfm = backend.cancel_agent_turn()
            assert cancel_cfm["ok"] is True and cancel_cfm["cancelled_confirm"] is True
            done3 = _wait_until(
                lambda: next((e for e in events if e.get("type") == "agent_event" and e.get("phase") == "turn_done"), None),
                timeout=3.0,
                err_msg="挂起期间 cancel_agent_turn 未立即唤醒",
            )
            assert done3["cancelled"] is True
            _wait_until(lambda: not backend.get_agent_state()["busy"], timeout=3.0)

            # 场景 D：clear_agent_history() 正确清空历史并广播 history_cleared
            clr = backend.clear_agent_history()
            assert clr["ok"] is True and clr["cleared"] >= 1
            assert backend.get_agent_state()["messages"] == []
            assert any(e.get("type") == "agent_event" and e.get("phase") == "history_cleared" for e in events)
        finally:
            close_rej()
    finally:
        unsub()
        close_srv()


def test_native_driver_anthropic_sse_and_colab_llm_routing_and_cancel(isolated_env: SimpleNamespace) -> None:
    """校验 Anthropic SSE 协议 tool_use 解析、Colab LLM 内存密钥自动路由与 cancel_agent_turn 立即中止。"""
    backend = _make_backend(isolated_env, "anth_and_colab")
    events: list[dict[str, Any]] = []
    unsub = backend.subscribe_events(lambda e: events.append(e))

    # 1) Anthropic /v1/messages 流式 content_block_start / input_json_delta / text_delta
    anth_turns = {"n": 0}

    def _anth_handler(handler, req_info):
        anth_turns["n"] += 1
        assert req_info["path"].endswith("/v1/messages")
        if anth_turns["n"] == 1:
            _send_sse_frames(
                handler,
                [
                    'event: message_start\ndata: {"type":"message_start"}\n\n',
                    'event: content_block_start\ndata: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_01","name":"get_cluster_and_pool_status","input":{}}}\n\n',
                    'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\\"refresh_"}}\n\n',
                    'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"pool\\": false}"}}\n\n',
                    'event: content_block_stop\ndata: {"type":"content_block_stop","index":0}\n\n',
                    'event: message_stop\ndata: {"type":"message_stop"}\n\n',
                ],
            )
            return
        msgs = req_info["body"].get("messages") or []
        assert any(
            m.get("role") == "user"
            and isinstance(m.get("content"), list)
            and any(b.get("type") == "tool_result" and b.get("tool_use_id") == "toolu_01" for b in m["content"])
            for m in msgs
        )
        _send_sse_frames(
            handler,
            [
                'event: content_block_start\ndata: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n\n',
                'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Anthropic 协议"}}\n\n',
                'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"工具调用已完成。"}}\n\n',
                'event: message_stop\ndata: {"type":"message_stop"}\n\n',
            ],
        )

    anth_port, anth_reqs, close_anth = _start_dual_protocol_sse_server(_anth_handler)
    try:
        anth_key = "sk-ant-api03-test-key-abcdef9999"
        backend.save_agent_config(
            {
                "agent_driver": "native",
                "agent_brain_source": "external_api",
                "agent_api_format": "anthropic",
                "agent_api_base_url": f"http://127.0.0.1:{anth_port}",
                "agent_model": "claude-sonnet-4-20250514",
                "api_key": anth_key,
            }
        )
        r1 = backend.send_agent_message({"text": "查看算力状态"})
        assert r1["ok"] is True
        done_anth = _wait_until(
            lambda: next((e for e in events if e.get("type") == "agent_event" and e.get("phase") == "turn_done"), None),
            err_msg="Anthropic SSE 对话未完成",
        )
        assert done_anth["ok"] is True
        assert "Anthropic 协议工具调用已完成" in done_anth["text"]
        assert anth_reqs[0]["headers"].get("x-api-key") == anth_key
        assert anth_reqs[0]["headers"].get("anthropic-version") == "2023-06-01"
    finally:
        close_anth()

    # 2) 自动路由与显式切换至在线 Colab Qwen (colab_llm + 内存 _runtime_secrets)
    events.clear()
    colab_runtime_key = "cpk-colab-qwen-runtime-secret-88776655"
    slow_gate = threading.Event()
    colab_mode = {"hang": False}

    def _colab_llm_handler(handler, _req_info):
        if colab_mode["hang"]:
            _send_sse_frames(
                handler,
                ['data: {"choices":[{"delta":{"content":"正在处理..."}}]}\n\n'],
            )
            slow_gate.wait(timeout=10.0)
            return
        _send_sse_frames(
            handler,
            [
                f'data: {{"choices":[{{"delta":{{"content":"在线 Colab Qwen 已就绪（泄露测试:{colab_runtime_key}）。"}}}}]}}\n\n',
                "data: [DONE]\n\n",
            ],
        )

    llm_port, llm_reqs, close_llm = _start_dual_protocol_sse_server(_colab_llm_handler)
    try:
        # 清空外部密钥并注入在线 Colab LLM 会话与内存密钥
        backend.save_agent_config(
            {
                "agent_brain_source": "auto",
                "agent_api_base_url": "",
                "clear_api_key": True,
            }
        )
        llm_ep = "ep-qwen-llm-sandbox"
        with backend._lock:
            backend.state["last_sessions"] = [{"name": "comfy", "endpoint": llm_ep}]
            backend._service_meta_by_ep[llm_ep] = {
                "kind": "llm",
                "llm_tunnel_url": f"http://127.0.0.1:{llm_port}",
                "model_name": "Qwen3.8-27B",
            }
            backend._runtime_secrets[llm_ep] = colab_runtime_key
        with backend._comfy_lock:
            backend._comfy_state.update(
                {
                    "online": True,
                    "service_kind": "llm",
                    "llm_online": True,
                    "llm_tunnel_url": f"http://127.0.0.1:{llm_port}",
                    "model_name": "Qwen3.8-27B",
                }
            )

        # 验证 auto 自动路由命中 colab_llm，且内存密钥绝不落盘或暴露
        st_auto = backend.get_agent_state()
        assert st_auto["brain"]["resolved_source"] == "colab_llm"
        assert st_auto["brain"]["colab_llm_online"] is True
        assert st_auto["brain"]["ready"] is True
        assert colab_runtime_key not in json.dumps(st_auto, ensure_ascii=False)
        assert colab_runtime_key not in backend.state_path.read_text(encoding="utf-8")

        # 显式切换到 colab_llm 并发送消息，验证携带内存密钥请求并脱敏输出
        backend.save_agent_config({"agent_brain_source": "colab_llm"})
        res_qwen = backend.send_agent_message({"text": "测试 Colab Qwen 路由"})
        assert res_qwen["ok"] is True
        done_qwen = _wait_until(
            lambda: next((e for e in events if e.get("type") == "agent_event" and e.get("phase") == "turn_done"), None),
            err_msg="Colab Qwen 对话未完成",
        )
        assert done_qwen["ok"] is True
        assert llm_reqs[-1]["headers"].get("authorization") == f"Bearer {colab_runtime_key}"
        assert colab_runtime_key not in done_qwen["text"]
        assert "[REDACTED" in done_qwen["text"]

        # 3) 用户主动 cancel_agent_turn() 立即终止流式生成
        events.clear()
        colab_mode["hang"] = True
        slow_gate.clear()
        res_hang = backend.send_agent_message({"text": "模拟长输出中断"})
        assert res_hang["ok"] is True
        _wait_until(
            lambda: any(e.get("type") == "agent_event" and e.get("phase") == "delta" for e in events),
            err_msg="未收到首段流式 delta",
        )
        t0 = time.monotonic()
        cancel_res = backend.cancel_agent_turn()
        assert cancel_res["ok"] is True and cancel_res["was_running"] is True
        done_cancel = _wait_until(
            lambda: next((e for e in events if e.get("type") == "agent_event" and e.get("phase") == "turn_done"), None),
            timeout=4.0,
            err_msg="cancel_agent_turn 后未及时退出",
        )
        assert (time.monotonic() - t0) < 3.5
        assert done_cancel["cancelled"] is True
        _wait_until(lambda: not backend.get_agent_state()["busy"], timeout=3.0, err_msg="取消后 busy 未复位")
    finally:
        slow_gate.set()
        unsub()
        close_llm()


def test_magpie_local_gateway_models_fetch_probe_and_multi_provider_vault(isolated_env: SimpleNamespace) -> None:
    """校验 Magpie 本机免密网关模型拉取、协议探测、逐模型测速、多供应商秒切与 SSRF 密钥保护。"""
    backend = _make_backend(isolated_env, "magpie_state")
    events: list[dict[str, Any]] = []
    unsub = backend.subscribe_events(lambda e: events.append(e))

    # 1) 启动仅支持 OpenAI Responses + OpenAI Models 的本机免密网关，并模拟一个坏模型
    gw_port, gw_reqs, close_gw = _start_magpie_mock_gateway(
        models=("gpt-4.1", "claude-sonnet-4-5", "broken-canary"),
        broken_models={"broken-canary"},
        supported_post_formats=("openai_responses",),
    )
    # 2) 启动另一个带严格密钥的多协议远端供应商
    ext_port, ext_reqs, close_ext = _start_magpie_mock_gateway(
        models=("deepseek-r1", "qwen3-coder"),
        supported_post_formats=("openai",),
    )
    ext_secret = "sk-remote-relay-vault-secret-20261118"
    try:
        gw_url = f"http://127.0.0.1:{gw_port}/v1"

        # 1a. 本机免密获取模型清单
        fetched = backend.fetch_agent_models({"base_url": gw_url, "api_format": "openai"})
        assert fetched["ok"] is True, fetched
        assert fetched["model_ids"] == ["gpt-4.1", "claude-sonnet-4-5", "broken-canary"]
        assert "authorization" not in gw_reqs[-1]["headers"]

        # 1b. 自动检测协议 -> 识别为 openai_responses
        proto_res = backend.probe_agent_provider({"mode": "detect_protocol", "base_url": gw_url})
        assert proto_res["ok"] is True, proto_res
        assert proto_res["api_format"] == "openai_responses"

        # 1c. 并发逐个检测所选模型 -> 2 可用 + 1 失败
        test_res = backend.probe_agent_provider(
            {
                "mode": "test_models",
                "base_url": gw_url,
                "api_format": "openai_responses",
                "models": ["gpt-4.1", "claude-sonnet-4-5", "broken-canary"],
            }
        )
        assert test_res["ok"] is True and test_res["passed"] == 2 and test_res["total"] == 3
        by_mdl = {r["model"]: r for r in test_res["results"]}
        assert by_mdl["gpt-4.1"]["ok"] is True and by_mdl["gpt-4.1"]["latency_ms"] >= 0
        assert by_mdl["claude-sonnet-4-5"]["ok"] is True
        assert by_mdl["broken-canary"]["ok"] is False and "404" in by_mdl["broken-canary"]["error"]

        # 1d. 保存无密钥 Magpie 本机网关 -> 立即判定 ready=True 并可免密发送对话
        save_gw = backend.save_agent_config(
            {
                "upsert_provider": {
                    "name": "Magpie 本机网关",
                    "api_format": "openai_responses",
                    "base_url": gw_url,
                    "models": ["gpt-4.1", "claude-sonnet-4-5"],
                    "default_model": "gpt-4.1",
                }
            }
        )
        assert save_gw["ok"] is True
        assert save_gw["brain"]["ready"] is True
        assert save_gw["brain"]["external_is_loopback"] is True
        assert "Magpie 本机网关" in save_gw["brain"]["resolved_brain_label"]
        prov_gw_id = save_gw["config"]["agent_active_provider_id"]
        assert prov_gw_id

        send_ok = backend.send_agent_message({"text": "你好 Magpie"})
        assert send_ok["ok"] is True
        done_gw = _wait_until(
            lambda: next((e for e in events if e.get("type") == "agent_event" and e.get("phase") == "turn_done"), None),
            err_msg="Magpie 本机免密网关未完成对话",
        )
        assert done_gw["ok"] is True and "Magpie-gpt-4.1-OK" in done_gw["text"] or "Resp-gpt-4.1-OK" in done_gw["text"]

        # 2. 保存带密钥的第二个供应商 -> 密钥独立入 Vault、零落盘、且 SSRF 跨端口改 URL 绝不携带旧密钥
        ext_url = f"http://127.0.0.1:{ext_port}/v1"
        save_ext = backend.save_agent_config(
            {
                "upsert_provider": {
                    "name": "远端多模型",
                    "api_format": "openai",
                    "base_url": ext_url,
                    "models": ["deepseek-r1", "qwen3-coder"],
                    "default_model": "deepseek-r1",
                    "api_key": ext_secret,
                }
            }
        )
        assert save_ext["ok"] is True
        prov_ext_id = save_ext["config"]["agent_active_provider_id"]
        assert prov_ext_id != prov_gw_id
        assert csec.VaultStore().get_secret(csec.agent_provider_vault_key(prov_ext_id)) == ext_secret
        assert ext_secret not in backend.state_path.read_text(encoding="utf-8")
        assert ext_secret not in json.dumps(save_ext, ensure_ascii=False)

        # 关键 SSRF 防护验证：传入 prov_ext_id 但将 base_url 指向另一端口(gw_url)，绝不能发送 ext_secret！
        gw_reqs.clear()
        ssrf_check = backend.fetch_agent_models({"provider_id": prov_ext_id, "base_url": gw_url, "api_format": "openai"})
        assert ssrf_check["ok"] is True
        assert gw_reqs[-1]["headers"].get("authorization") is None

        # 同源情况下允许免输回退读取已存密钥
        ext_reqs.clear()
        same_origin = backend.fetch_agent_models({"provider_id": prov_ext_id, "base_url": ext_url, "api_format": "openai"})
        assert same_origin["ok"] is True
        assert ext_reqs[-1]["headers"].get("authorization") == f"Bearer {ext_secret}"

        # 3. 1 键秒切供应商下的特定模型 + 删除供应商同步清理 Vault 密钥
        events.clear()
        sw_res = backend.save_agent_config(
            {"select_provider_model": {"provider_id": prov_gw_id, "model": "claude-sonnet-4-5"}}
        )
        assert sw_res["ok"] is True
        assert sw_res["config"]["agent_active_provider_id"] == prov_gw_id
        assert sw_res["config"]["agent_model"] == "claude-sonnet-4-5"

        del_res = backend.save_agent_config({"delete_provider_id": prov_ext_id})
        assert del_res["ok"] is True
        assert csec.VaultStore().get_secret(csec.agent_provider_vault_key(prov_ext_id)) is None
        assert len(del_res["config"]["providers"]) == 1
    finally:
        unsub()
        close_gw()
        close_ext()


def test_provider_origin_change_invalidates_old_key_and_vault_origin_pinning(
    isolated_env: SimpleNamespace,
) -> None:
    """校验供应商换源自动清除旧密钥、Vault __origin 硬绑定防跨站外发，以及磁盘篡改重启拦截。"""
    port_a, reqs_a, close_a = _start_magpie_mock_gateway(models=("gpt-4o-mini",))
    port_b, reqs_b, close_b = _start_magpie_mock_gateway(models=("gpt-4o-mini",))
    try:
        vault = csec.VaultStore()
        secret_a = "sk-origin-pin-site-a-99887766"
        secret_b = "sk-origin-pin-site-b-55443322"

        # ------------------------------------------------------------------
        # ① 编辑已有命名供应商（同 id）从 api-a 换到 api-b 且不填 api_key
        # ------------------------------------------------------------------
        backend = _make_backend(isolated_env, "origin_pin_state")
        ba_mod = sys.modules[type(backend._agent_svc).__module__]
        save_a = backend.save_agent_config(
            {
                "upsert_provider": {
                    "name": "站点A供应商",
                    "api_format": "openai",
                    "base_url": "https://api-a.example/v1",
                    "models": ["gpt-4o-mini"],
                    "default_model": "gpt-4o-mini",
                    "api_key": secret_a,
                    "activate": True,
                }
            }
        )
        assert save_a["ok"] is True
        pid = save_a["config"]["agent_active_provider_id"]
        vkey = csec.agent_provider_vault_key(pid)
        okey = f"{vkey}__origin"
        assert save_a["config"]["has_api_key"] is True
        assert save_a["config"]["providers"][0]["has_api_key"] is True
        assert vault.get_secret(vkey) == secret_a
        assert vault.get_secret(okey) == "https://api-a.example:443"
        # 脱敏词表不得把 __origin 源站 URL 纳入敏感密钥
        known_secs = backend._agent_svc._collect_known_secrets()
        assert secret_a in known_secs
        assert "https://api-a.example:443" not in known_secs

        # 同源修改路径（api-a/v1 -> api-a/v2）且不填密钥，应保留原密钥
        same_origin_edit = backend.save_agent_config(
            {
                "upsert_provider": {
                    "id": pid,
                    "name": "站点A供应商",
                    "api_format": "openai",
                    "base_url": "https://api-a.example/v2",
                    "models": ["gpt-4o-mini"],
                    "default_model": "gpt-4o-mini",
                    "api_key": "",
                    "activate": True,
                }
            }
        )
        assert same_origin_edit["ok"] is True
        assert same_origin_edit["config"]["has_api_key"] is True
        assert vault.get_secret(vkey) == secret_a

        # 跨源改为 https://api-b.example/v1 且不填 api_key -> 立即失效并清除 Vault 旧密钥及 __origin
        cross_edit = backend.save_agent_config(
            {
                "upsert_provider": {
                    "id": pid,
                    "name": "站点A供应商",
                    "api_format": "openai",
                    "base_url": "https://api-b.example/v1",
                    "models": ["gpt-4o-mini"],
                    "default_model": "gpt-4o-mini",
                    "activate": True,
                }
            }
        )
        assert cross_edit["ok"] is True
        assert cross_edit["config"]["has_api_key"] is False
        assert cross_edit["config"]["providers"][0]["has_api_key"] is False
        assert cross_edit["brain"]["has_external_key"] is False
        assert cross_edit["brain"]["ready"] is False
        assert vault.get_secret(vkey) is None
        assert vault.get_secret(okey) is None
        assert vault.get_secret(csec.AGENT_VAULT_API_KEY) is None
        assert vault.get_secret(f"{csec.AGENT_VAULT_API_KEY}__origin") is None

        # 对话与探测均绝不向 api-b 发送旧密钥
        send_blocked = backend.send_agent_message({"text": "测试切源后禁发旧密钥"})
        assert send_blocked["ok"] is False and send_blocked.get("needs_config") is True

        seen_probe_keys: list[str] = []
        with patch.object(
            ba_mod,
            "fetch_provider_models",
            side_effect=lambda url, api_key="", api_format="openai": (
                seen_probe_keys.append(str(api_key or "")),
                {"ok": True, "model_ids": ["gpt-4o-mini"], "model_capabilities": {}},
            )[1],
        ), patch.object(
            ba_mod,
            "probe_provider_models",
            side_effect=lambda url, models, api_key="", api_format="openai": (
                seen_probe_keys.append(str(api_key or "")),
                {"ok": True, "passed": 1, "total": 1, "results": []},
            )[1],
        ):
            f_res = backend.fetch_agent_models({"provider_id": pid, "base_url": "https://api-b.example/v1"})
            p_res = backend.probe_agent_provider(
                {"mode": "test_models", "provider_id": pid, "base_url": "https://api-b.example/v1", "models": ["gpt-4o-mini"]}
            )
            assert f_res["ok"] is True and p_res["ok"] is True
        assert seen_probe_keys == ["", ""]

        # 真实本地双端口验证：先绑定 port_a + secret_a，再编辑同 id 改为 port_b 且不填 key，绝不向 port_b 发 Bearer Key
        backend.save_agent_config(
            {
                "upsert_provider": {
                    "id": pid,
                    "name": "回环换源校验",
                    "api_format": "openai",
                    "base_url": f"http://127.0.0.1:{port_a}/v1",
                    "models": ["gpt-4o-mini"],
                    "default_model": "gpt-4o-mini",
                    "api_key": secret_a,
                    "activate": True,
                }
            }
        )
        assert vault.get_secret(okey) == f"http://127.0.0.1:{port_a}"
        reqs_a.clear()
        assert backend.fetch_agent_models({"provider_id": pid})["ok"] is True
        assert reqs_a[-1]["headers"].get("authorization") == f"Bearer {secret_a}"

        edit_to_b = backend.save_agent_config(
            {
                "upsert_provider": {
                    "id": pid,
                    "name": "回环换源校验",
                    "api_format": "openai",
                    "base_url": f"http://127.0.0.1:{port_b}/v1",
                    "models": ["gpt-4o-mini"],
                    "default_model": "gpt-4o-mini",
                    "api_key": "",
                    "activate": True,
                }
            }
        )
        assert edit_to_b["config"]["has_api_key"] is False
        assert vault.get_secret(vkey) is None
        assert vault.get_secret(okey) is None

        reqs_b.clear()
        ev_b: list[dict[str, Any]] = []
        unsub_b = backend.subscribe_events(lambda e: ev_b.append(e))
        try:
            assert backend.fetch_agent_models({"provider_id": pid})["ok"] is True
            assert backend.probe_agent_provider({"mode": "test_models", "provider_id": pid, "models": ["gpt-4o-mini"]})["ok"] is True
            msg_b = backend.send_agent_message({"text": "验证切端口后无密钥外发"})
            assert msg_b["ok"] is True
            _wait_until(
                lambda: any(e.get("type") == "agent_event" and e.get("phase") == "turn_done" for e in ev_b),
                err_msg="port_b 对话未完成",
            )
            assert len(reqs_b) >= 3
            for r in reqs_b:
                assert r["headers"].get("authorization") is None
                assert r["headers"].get("x-api-key") is None
                assert secret_a not in json.dumps(r, ensure_ascii=False)
        finally:
            unsub_b()

        # 编辑换源同时传入新 up_key 时，正常写入新密钥并更新 __origin
        save_new_b = backend.save_agent_config(
            {
                "upsert_provider": {
                    "id": pid,
                    "name": "站点B新密",
                    "api_format": "openai",
                    "base_url": "https://api-b.example/v1",
                    "models": ["gpt-4o-mini"],
                    "default_model": "gpt-4o-mini",
                    "api_key": secret_b,
                    "activate": True,
                }
            }
        )
        assert save_new_b["config"]["has_api_key"] is True
        assert vault.get_secret(vkey) == secret_b
        assert vault.get_secret(okey) == "https://api-b.example:443"

        # ------------------------------------------------------------------
        # ② 传统 save_agent_config({"agent_api_base_url": ...}) 切换域名/端口且不传新 Key
        # ------------------------------------------------------------------
        b_leg = _make_backend(isolated_env, "origin_legacy_form")
        vault.delete_secret(vkey)
        vault.delete_secret(okey)
        vault.delete_secret(csec.AGENT_VAULT_API_KEY)
        vault.delete_secret(f"{csec.AGENT_VAULT_API_KEY}__origin")

        res_leg1 = b_leg.save_agent_config(
            {
                "agent_brain_source": "external_api",
                "agent_api_format": "openai",
                "agent_api_base_url": f"http://127.0.0.1:{port_a}/v1",
                "agent_model": "gpt-4o-mini",
                "api_key": secret_a,
            }
        )
        assert res_leg1["config"]["has_api_key"] is True
        assert vault.get_secret(csec.AGENT_VAULT_API_KEY) == secret_a
        assert vault.get_secret(f"{csec.AGENT_VAULT_API_KEY}__origin") == f"http://127.0.0.1:{port_a}"

        # 切换为外部异域名且不传新 Key -> 旧 AGENT_VAULT_API_KEY 及其 __origin 立即清除
        res_leg_ext = b_leg.save_agent_config({"agent_api_base_url": "https://api-b.example/v1"})
        assert res_leg_ext["config"]["has_api_key"] is False
        assert res_leg_ext["brain"]["ready"] is False
        assert vault.get_secret(csec.AGENT_VAULT_API_KEY) is None
        assert vault.get_secret(f"{csec.AGENT_VAULT_API_KEY}__origin") is None
        assert b_leg.send_agent_message({"text": "不应发送旧密钥"})["ok"] is False

        # 再测试 127.0.0.1:portA 切换到 127.0.0.1:portB：清旧 Key、loopback 免密放行且绝不发送 portA 密钥
        b_leg.save_agent_config(
            {
                "agent_api_base_url": f"http://127.0.0.1:{port_a}/v1",
                "api_key": secret_a,
            }
        )
        assert vault.get_secret(csec.AGENT_VAULT_API_KEY) == secret_a
        res_leg_portb = b_leg.save_agent_config({"agent_api_base_url": f"http://127.0.0.1:{port_b}/v1"})
        assert res_leg_portb["config"]["has_api_key"] is False
        assert res_leg_portb["brain"]["external_is_loopback"] is True
        assert res_leg_portb["brain"]["ready"] is True
        assert vault.get_secret(csec.AGENT_VAULT_API_KEY) is None
        assert vault.get_secret(f"{csec.AGENT_VAULT_API_KEY}__origin") is None

        reqs_b.clear()
        ev_lb: list[dict[str, Any]] = []
        unsub_lb = b_leg.subscribe_events(lambda e: ev_lb.append(e))
        try:
            assert b_leg.fetch_agent_models({})["ok"] is True
            assert b_leg.send_agent_message({"text": "验证 portB 免密且无 portA 密钥"})["ok"] is True
            _wait_until(
                lambda: any(e.get("type") == "agent_event" and e.get("phase") == "turn_done" for e in ev_lb),
                err_msg="传统表单切端口对话未完成",
            )
            assert len(reqs_b) >= 2
            for r in reqs_b:
                assert r["headers"].get("authorization") is None
                assert secret_a not in json.dumps(r, ensure_ascii=False)
        finally:
            unsub_lb()

        # 验证激活了命名供应商后，若通过传统表单改用异源 URL 解绑，绝不误删原命名供应商自己的密钥
        b_unbind = _make_backend(isolated_env, "origin_unbind_named")
        saved_named = b_unbind.save_agent_config(
            {
                "upsert_provider": {
                    "name": "命名供应商A",
                    "api_format": "openai",
                    "base_url": "https://api-a.example/v1",
                    "models": ["gpt-4o-mini"],
                    "default_model": "gpt-4o-mini",
                    "api_key": secret_a,
                    "activate": True,
                }
            }
        )
        named_pid = saved_named["config"]["agent_active_provider_id"]
        named_vkey = csec.agent_provider_vault_key(named_pid)
        # 模拟曾存在同源单键，再通过传统表单切换至另一 URL
        b_unbind.save_agent_config({"api_key": secret_a})
        assert vault.get_secret(csec.AGENT_VAULT_API_KEY) == secret_a
        res_unbind = b_unbind.save_agent_config({"agent_api_base_url": "https://api-b.example/v1"})
        assert res_unbind["config"]["agent_active_provider_id"] == ""
        assert res_unbind["config"]["has_api_key"] is False
        assert vault.get_secret(csec.AGENT_VAULT_API_KEY) is None
        assert vault.get_secret(f"{csec.AGENT_VAULT_API_KEY}__origin") is None
        assert vault.get_secret(named_vkey) == secret_a
        assert vault.get_secret(f"{named_vkey}__origin") == "https://api-a.example:443"
        # 切回原命名供应商后密钥依然完整可用
        res_reselect = b_unbind.save_agent_config(
            {"select_provider_model": {"provider_id": named_pid, "model": "gpt-4o-mini"}}
        )
        assert res_reselect["config"]["has_api_key"] is True

        # ------------------------------------------------------------------
        # ③ 直接篡改磁盘 pet_state.json 的 base_url 为异源地址并重启 ColabPetBackend
        # ------------------------------------------------------------------
        # 3a) 命名供应商被磁盘篡改
        b_tamper_named = _make_backend(isolated_env, "tamper_named_state")
        st_tn = b_tamper_named.save_agent_config(
            {
                "upsert_provider": {
                    "name": "防篡改供应商",
                    "api_format": "openai",
                    "base_url": "https://api-a.example/v1",
                    "models": ["gpt-4o-mini"],
                    "default_model": "gpt-4o-mini",
                    "api_key": secret_a,
                    "activate": True,
                }
            }
        )
        tn_pid = st_tn["config"]["agent_active_provider_id"]
        disk_obj = json.loads(b_tamper_named.state_path.read_text(encoding="utf-8"))
        disk_obj["prefs"]["agent_providers"][0]["base_url"] = f"http://127.0.0.1:{port_b}/v1"
        disk_obj["prefs"]["agent_api_base_url"] = f"http://127.0.0.1:{port_b}/v1"
        b_tamper_named.state_path.write_text(json.dumps(disk_obj, ensure_ascii=False, indent=2), encoding="utf-8")

        b_restarted_named = _make_backend(isolated_env, "tamper_named_state")
        pub_tn, rt_tn = b_restarted_named._agent_svc._resolve_brain_runtime(probe_network=False)
        assert pub_tn["has_external_key"] is False
        assert rt_tn["api_key"] == ""
        assert rt_tn["ext_api_key"] == ""
        st_after_tn = b_restarted_named.get_agent_state()
        assert st_after_tn["config"]["has_api_key"] is False
        assert st_after_tn["config"]["providers"][0]["has_api_key"] is False

        reqs_b.clear()
        assert b_restarted_named.fetch_agent_models({"provider_id": tn_pid})["ok"] is True
        assert reqs_b[-1]["headers"].get("authorization") is None
        assert secret_a not in json.dumps(reqs_b[-1], ensure_ascii=False)

        # 3b) 传统单键 AGENT_VAULT_API_KEY 被磁盘篡改（异源与空串均应拒绝）
        b_tamper_leg = _make_backend(isolated_env, "tamper_legacy_state")
        b_tamper_leg.save_agent_config(
            {
                "agent_brain_source": "external_api",
                "agent_api_format": "openai",
                "agent_api_base_url": "https://api-a.example/v1",
                "agent_model": "gpt-4o-mini",
                "api_key": secret_b,
            }
        )
        assert vault.get_secret(csec.AGENT_VAULT_API_KEY) == secret_b
        assert vault.get_secret(f"{csec.AGENT_VAULT_API_KEY}__origin") == "https://api-a.example:443"

        leg_disk = json.loads(b_tamper_leg.state_path.read_text(encoding="utf-8"))
        leg_disk["prefs"]["agent_api_base_url"] = "https://evil-tamper.example/v1"
        b_tamper_leg.state_path.write_text(json.dumps(leg_disk, ensure_ascii=False, indent=2), encoding="utf-8")
        b_restarted_leg = _make_backend(isolated_env, "tamper_legacy_state")
        pub_tl, rt_tl = b_restarted_leg._agent_svc._resolve_brain_runtime(probe_network=False)
        assert pub_tl["has_external_key"] is False
        assert rt_tl["api_key"] == ""
        assert rt_tl["ext_api_key"] == ""
        assert b_restarted_leg.get_agent_state()["config"]["has_api_key"] is False

        leg_disk["prefs"]["agent_api_base_url"] = ""
        b_tamper_leg.state_path.write_text(json.dumps(leg_disk, ensure_ascii=False, indent=2), encoding="utf-8")
        b_restarted_empty = _make_backend(isolated_env, "tamper_legacy_state")
        pub_te, rt_te = b_restarted_empty._agent_svc._resolve_brain_runtime(probe_network=False)
        assert pub_te["has_external_key"] is False
        assert rt_te["ext_api_key"] == ""

        # 3c) 无 __origin 伴随键的旧单键记录不再允许隐式回退读取（严格 Fail-Closed）
        vault.delete_secret(f"{csec.AGENT_VAULT_API_KEY}__origin")
        assert b_restarted_empty._agent_svc._read_provider_secret(
            "__legacy__",
            base_url="https://api-a.example/v1",
            legacy_base_url="https://api-a.example/v1",
        ) == ""
        assert b_restarted_empty._agent_svc._read_provider_secret(
            "__legacy__",
            base_url="https://api-b.example/v1",
            legacy_base_url="https://api-a.example/v1",
        ) == ""
    finally:
        close_a()
        close_b()


def test_agent_cross_origin_redirect_never_leaks_credentials(isolated_env: SimpleNamespace) -> None:
    """验证供应商/探针/对话遇 HTTP 302 跨源重定向时立即阻断，绝不向异源目标发送认证头。"""
    evil_reqs: list[dict[str, Any]] = []

    class _EvilHandler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *_a: Any) -> None:
            return

        def do_GET(self) -> None:  # noqa: N802
            evil_reqs.append({"method": "GET", "path": self.path, "headers": dict(self.headers)})
            self.send_response(200)
            self.end_headers()

        def do_POST(self) -> None:  # noqa: N802
            evil_reqs.append({"method": "POST", "path": self.path, "headers": dict(self.headers)})
            self.send_response(200)
            self.end_headers()

    evil_srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _EvilHandler)
    evil_port = int(evil_srv.server_address[1])
    evil_t = threading.Thread(target=evil_srv.serve_forever, daemon=True)
    evil_t.start()

    class _RedirectHandler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *_a: Any) -> None:
            return

        def do_GET(self) -> None:  # noqa: N802
            self.send_response(302)
            self.send_header("Location", f"http://127.0.0.1:{evil_port}{self.path}")
            self.end_headers()

        def do_POST(self) -> None:  # noqa: N802
            ln = int(self.headers.get("Content-Length") or 0)
            if ln > 0:
                self.rfile.read(ln)
            self.send_response(302)
            self.send_header("Location", f"http://127.0.0.1:{evil_port}{self.path}")
            self.end_headers()

    orig_srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _RedirectHandler)
    orig_port = int(orig_srv.server_address[1])
    orig_t = threading.Thread(target=orig_srv.serve_forever, daemon=True)
    orig_t.start()

    try:
        leak_key = "sk-redirect-must-never-leak-8877"
        base_orig = f"http://127.0.0.1:{orig_port}/v1"

        m_res = agent_tools.fetch_provider_models(base_orig, api_key=leak_key, api_format="openai")
        assert m_res["ok"] is False and "跨源重定向" in m_res["message"]
        assert evil_reqs == []

        p_res = agent_tools.probe_provider_models(
            base_orig,
            ["gpt-4o-mini"],
            api_key=leak_key,
            api_format="openai",
        )
        assert p_res["ok"] is False and p_res["passed"] == 0
        assert "跨源重定向" in p_res["results"][0]["error"]
        assert evil_reqs == []

        drv = agent_tools.NativeLightDriver(
            base_url=base_orig,
            api_key=leak_key,
            model="gpt-4o-mini",
            api_format="openai",
        )
        run_res = drv.run([{"role": "user", "content": "hi"}])
        assert run_res["ok"] is False
        assert evil_reqs == []
    finally:
        orig_srv.shutdown()
        orig_srv.server_close()
        orig_t.join(timeout=2.0)
        evil_srv.shutdown()
        evil_srv.server_close()
        evil_t.join(timeout=2.0)


def test_native_driver_openai_responses_and_gemini_tool_loops(isolated_env: SimpleNamespace) -> None:
    """校验 NativeLightDriver 对 openai_responses 与 gemini 协议的 SSE 工具调用多轮闭环。"""
    # 1) OpenAI Responses SSE 工具流
    resp_turns = {"n": 0}

    def _responses_sse(handler, req_info):
        resp_turns["n"] += 1
        assert req_info["path"].endswith("/v1/responses")
        inp = req_info["body"].get("input") or []
        if resp_turns["n"] == 1:
            _send_sse_frames(
                handler,
                [
                    'event: response.output_item.added\ndata: {"type":"response.output_item.added","output_index":0,"item":{"id":"fc_1","type":"function_call","call_id":"call_r_1","name":"get_cluster_and_pool_status"}}\n\n',
                    'event: response.function_call_arguments.delta\ndata: {"type":"response.function_call_arguments.delta","item_id":"fc_1","delta":"{\\"refresh_pool\\": false}"}\n\n',
                    'event: response.function_call_arguments.done\ndata: {"type":"response.function_call_arguments.done","item_id":"fc_1"}\n\n',
                    'event: response.completed\ndata: {"type":"response.completed"}\n\n',
                ],
            )
            return
        assert any(isinstance(x, dict) and x.get("type") == "function_call_output" and x.get("call_id") == "call_r_1" for x in inp)
        _send_sse_frames(
            handler,
            [
                'event: response.output_text.delta\ndata: {"type":"response.output_text.delta","delta":"Responses 工具链路已通。"}\n\n',
                'event: response.completed\ndata: {"type":"response.completed"}\n\n',
            ],
        )

    r_port, _, close_r = _start_magpie_mock_gateway(post_handler=_responses_sse)
    try:
        evts_r: list[dict[str, Any]] = []
        drv_r = agent_tools.NativeLightDriver(
            base_url=f"http://127.0.0.1:{r_port}/v1",
            model="gpt-5-codex",
            api_format="openai_responses",
            tool_executor=lambda n, _a: {"ok": True, "tool": n, "balance": 200.0},
            on_event=evts_r.append,
        )
        res_r = drv_r.run([{"role": "user", "content": "查一下状态"}])
        assert res_r["ok"] is True and res_r["tool_calls_count"] == 1
        assert "Responses 工具链路已通" in res_r["text"]
    finally:
        close_r()

    # 2) Gemini SSE functionCall + functionResponse 闭环
    gem_turns = {"n": 0}

    def _gemini_sse(handler, req_info):
        gem_turns["n"] += 1
        assert "/v1beta/models/gemini-2.5-pro:streamGenerateContent" in req_info["path"]
        assert req_info["headers"].get("x-goog-api-key") == "AIzaSyGeminiTestKey123456"
        contents = req_info["body"].get("contents") or []
        if gem_turns["n"] == 1:
            _send_sse_frames(
                handler,
                [
                    'data: {"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"name":"get_cluster_and_pool_status","args":{"refresh_pool":false}}}]}}]}\n\n',
                ],
            )
            return
        assert any(
            isinstance(c, dict)
            and any(isinstance(p, dict) and "functionResponse" in p for p in (c.get("parts") or []))
            for c in contents
        )
        _send_sse_frames(
            handler,
            [
                'data: {"candidates":[{"content":{"role":"model","parts":[{"text":"Gemini 工具链路已通。"}]}}]}\n\n',
            ],
        )

    g_port, _, close_g = _start_magpie_mock_gateway(post_handler=_gemini_sse)
    try:
        drv_g = agent_tools.NativeLightDriver(
            base_url=f"http://127.0.0.1:{g_port}",
            api_key="AIzaSyGeminiTestKey123456",
            model="gemini-2.5-pro",
            api_format="gemini",
            tool_executor=lambda n, _a: {"ok": True, "tool": n, "balance": 320.0},
        )
        res_g = drv_g.run([{"role": "user", "content": "查看 Gemini 链路"}])
        assert res_g["ok"] is True and res_g["tool_calls_count"] == 1
        assert "Gemini 工具链路已通" in res_g["text"]
    finally:
        close_g()


def test_reasoning_effort_wire_mapping_streaming_trace_and_fallback(isolated_env: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    """校验模型思考强度配置持久化、四协议请求映射、流式思考提取及 HTTP 400 自动降级重试。"""
    # 1) sanitize_prefs_dict 与后端配置持久化校验
    assert sanitize_prefs_dict({"agent_reasoning_effort": "HIGH"})["agent_reasoning_effort"] == "high"
    assert sanitize_prefs_dict({"agent_reasoning_effort": "bogus"})["agent_reasoning_effort"] == "auto"

    backend = _make_backend(isolated_env, "reasoning_state")
    events: list[dict[str, Any]] = []
    unsub = backend.subscribe_events(lambda e: events.append(e))

    # 2) OpenAI Chat Completions：xhigh -> HTTP 400 降级为 high，流式 reasoning_content 与 <think> 内联切分
    oai_bodies: list[dict[str, Any]] = []

    def _oai_reasoning_handler(handler, req_info):
        oai_bodies.append(req_info["body"])
        eff = req_info["body"].get("reasoning_effort")
        if eff == "xhigh":
            _send_json_resp(handler, 400, {"error": {"message": "Unsupported reasoning_effort: xhigh"}})
            return
        if any("THINK_TAG" in str(m.get("content") or "") for m in (req_info["body"].get("messages") or [])):
            _send_sse_frames(
                handler,
                [
                    'data: {"choices":[{"delta":{"content":"<thi"}}]}\n\n',
                    'data: {"choices":[{"delta":{"content":"nk>第一步分析\\n第二步收敛</thi"}}]}\n\n',
                    'data: {"choices":[{"delta":{"content":"nk>\\n最终清晰结论。"}}]}\n\n',
                    "data: [DONE]\n\n",
                ],
            )
            return
        _send_sse_frames(
            handler,
            [
                'data: {"choices":[{"delta":{"reasoning_content":"先检查号池余额，"}}]}\n\n',
                'data: {"choices":[{"delta":{"reasoning_content":"再评估 L4 成本。"}}]}\n\n',
                'data: {"choices":[{"delta":{"content":"推荐优先选用 L4 节点。"}}]}\n\n',
                "data: [DONE]\n\n",
            ],
        )

    o_port, _, close_o = _start_magpie_mock_gateway(post_handler=_oai_reasoning_handler)
    try:
        save_res = backend.save_agent_config(
            {
                "agent_driver": "native",
                "agent_brain_source": "external_api",
                "agent_api_format": "openai",
                "agent_api_base_url": f"http://127.0.0.1:{o_port}/v1",
                "agent_model": "deepseek-v4.1-flash",
                "agent_reasoning_effort": "xhigh",
            }
        )
        assert save_res["ok"] is True and save_res["config"]["agent_reasoning_effort"] == "xhigh"

        # 触发 send_agent_message -> 先尝试 xhigh 吃 400，自动降级为 high 后成功返回 reasoning_delta + 落库
        send_ok = backend.send_agent_message({"text": "帮我分析选卡策略"})
        assert send_ok["ok"] is True
        done_evt = _wait_until(
            lambda: next((e for e in events if e.get("type") == "agent_event" and e.get("phase") == "turn_done"), None),
            err_msg="OpenAI 思考链对话未完成",
        )
        assert done_evt["ok"] is True
        assert oai_bodies[0].get("reasoning_effort") == "xhigh"
        assert oai_bodies[1].get("reasoning_effort") == "high"
        assert "先检查号池余额，再评估 L4 成本。" in done_evt["reasoning"]
        assert "推荐优先选用 L4 节点" in done_evt["text"]
        assert any(e.get("phase") == "reasoning_delta" for e in events)
        hist = backend.get_agent_state()["messages"]
        assert hist[-1]["reasoning"] == "先检查号池余额，再评估 L4 成本。"

        # 校验内联 <think>...</think> 流式标签剥离
        oai_bodies.clear()
        evts_tag: list[dict[str, Any]] = []
        drv_tag = agent_tools.NativeLightDriver(
            base_url=f"http://127.0.0.1:{o_port}/v1",
            model="qwen3-thinking",
            api_format="openai",
            reasoning_effort="medium",
            on_event=evts_tag.append,
        )
        res_tag = drv_tag.run([{"role": "user", "content": "THINK_TAG test"}])
        assert res_tag["ok"] is True
        assert res_tag["reasoning"] == "第一步分析\n第二步收敛"
        assert res_tag["text"] == "最终清晰结论。"
        assert any(e["type"] == "reasoning_delta" for e in evts_tag)
    finally:
        close_o()

    # 3) 测试非推理模型对任意 reasoning_effort 返回 400 -> 自动剥离参数重试成功
    no_think_bodies: list[dict[str, Any]] = []

    def _no_think_handler(handler, req_info):
        no_think_bodies.append(req_info["body"])
        if "reasoning_effort" in req_info["body"]:
            _send_json_resp(handler, 400, {"error": {"message": "Unknown parameter: reasoning_effort"}})
            return
        _send_sse_frames(
            handler,
            ['data: {"choices":[{"delta":{"content":"普通模型降级兼容成功。"}}]}\n\n', "data: [DONE]\n\n"],
        )

    nt_port, _, close_nt = _start_magpie_mock_gateway(post_handler=_no_think_handler)
    try:
        drv_nt = agent_tools.NativeLightDriver(
            base_url=f"http://127.0.0.1:{nt_port}/v1",
            model="legacy-chat",
            api_format="openai",
            reasoning_effort="low",
        )
        res_nt = drv_nt.run([{"role": "user", "content": "普通模型测试"}])
        assert res_nt["ok"] is True and "普通模型降级兼容成功" in res_nt["text"]
        assert "reasoning_effort" in no_think_bodies[0]
        assert "reasoning_effort" not in no_think_bodies[1]
    finally:
        close_nt()

    # 4) OpenAI Responses 协议 reasoning 参数与 reasoning_summary_text.delta 提取
    resp_bodies: list[dict[str, Any]] = []

    def _resp_reason_handler(handler, req_info):
        resp_bodies.append(req_info["body"])
        _send_sse_frames(
            handler,
            [
                'event: response.reasoning_summary_text.delta\ndata: {"type":"response.reasoning_summary_text.delta","delta":"摘要推演中..."}\n\n',
                'event: response.output_text.delta\ndata: {"type":"response.output_text.delta","delta":"Responses 推理回答。"}\n\n',
                'event: response.completed\ndata: {"type":"response.completed"}\n\n',
            ],
        )

    r_port, _, close_r = _start_magpie_mock_gateway(post_handler=_resp_reason_handler)
    try:
        drv_r = agent_tools.NativeLightDriver(
            base_url=f"http://127.0.0.1:{r_port}/v1",
            model="gpt-5",
            api_format="openai_responses",
            reasoning_effort="medium",
        )
        res_r = drv_r.run([{"role": "user", "content": "推理"}])
        assert res_r["ok"] is True
        assert resp_bodies[0]["reasoning"] == {"effort": "medium", "summary": "auto"}
        assert res_r["reasoning"] == "摘要推演中..."
        assert res_r["text"] == "Responses 推理回答。"
    finally:
        close_r()

    # 5) Anthropic 协议 thinking + budget_tokens + 签名多轮工具回灌 + Gemini thought 分离
    anth_bodies: list[dict[str, Any]] = []
    anth_turn = {"n": 0}

    def _anth_think_handler(handler, req_info):
        anth_bodies.append(req_info["body"])
        anth_turn["n"] += 1
        if anth_turn["n"] == 1:
            _send_sse_frames(
                handler,
                [
                    'event: content_block_start\ndata: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":""}}\n\n',
                    'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"先查状态表"}}\n\n',
                    'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig_abc_123"}}\n\n',
                    'event: content_block_start\ndata: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"tu_t1","name":"get_cluster_and_pool_status","input":{"refresh_pool":false}}}\n\n',
                    'event: message_stop\ndata: {"type":"message_stop"}\n\n',
                ],
            )
            return
        _send_sse_frames(
            handler,
            [
                'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Claude 思考链完成。"}}\n\n',
                'event: message_stop\ndata: {"type":"message_stop"}\n\n',
            ],
        )

    a_port, _, close_a = _start_magpie_mock_gateway(post_handler=_anth_think_handler)
    try:
        drv_a = agent_tools.NativeLightDriver(
            base_url=f"http://127.0.0.1:{a_port}",
            api_key="sk-ant-test-999",
            model="claude-sonnet-4-5",
            api_format="anthropic",
            reasoning_effort="high",
            tool_executor=lambda n, _a: {"ok": True, "tool": n},
        )
        res_a = drv_a.run([{"role": "user", "content": "思考并查集群"}])
        assert res_a["ok"] is True and res_a["tool_calls_count"] == 1
        assert anth_bodies[0]["thinking"] == {"type": "enabled", "budget_tokens": 16000}
        assert anth_bodies[0]["max_tokens"] >= 20096
        # 第二轮 assistant 消息首块必须带回 signature thinking block
        asst_turn_msg = next(m for m in anth_bodies[1]["messages"] if m.get("role") == "assistant")
        assert asst_turn_msg["content"][0] == {"type": "thinking", "thinking": "先查状态表", "signature": "sig_abc_123"}
        assert res_a["reasoning"] == "先查状态表"
        assert "Claude 思考链完成" in res_a["text"]
    finally:
        close_a()

    # Gemini thought=True 分离验证
    gem_bodies: list[dict[str, Any]] = []

    def _gem_think_handler(handler, req_info):
        gem_bodies.append(req_info["body"])
        _send_sse_frames(
            handler,
            [
                'data: {"candidates":[{"content":{"role":"model","parts":[{"thought":true,"text":"Gemini 内部思考过程"}]}}]}\n\n',
                'data: {"candidates":[{"content":{"role":"model","parts":[{"text":"Gemini 正式答复。"}]}}]}\n\n',
            ],
        )

    g_port, _, close_g2 = _start_magpie_mock_gateway(post_handler=_gem_think_handler)
    try:
        drv_g2 = agent_tools.NativeLightDriver(
            base_url=f"http://127.0.0.1:{g_port}",
            api_key="AIzaSyGemThink123",
            model="gemini-2.5-pro",
            api_format="gemini",
            reasoning_effort="low",
        )
        res_g2 = drv_g2.run([{"role": "user", "content": "Gemini 思考"}])
        assert res_g2["ok"] is True
        assert gem_bodies[0]["generationConfig"]["thinkingConfig"] == {"thinkingBudget": 2048, "includeThoughts": True}
        assert res_g2["reasoning"] == "Gemini 内部思考过程"
        assert res_g2["text"] == "Gemini 正式答复。"
    finally:
        close_g2()
        unsub()

    # 3) Magpie /v1/models 能力元数据解析、自动收敛与 Gemini 4 Hydrogen thinkingLevel 校验
    h_spec = infer_model_reasoning_spec("ai-studio/gemini-4-hydrogen")
    assert h_spec["reasoning"] is True
    assert h_spec["levels"] == ["low", "medium", "high", "xhigh"]
    assert h_spec["default_level"] == "high"
    assert sanitize_prefs_dict({"agent_reasoning_effort": "extra"})["agent_reasoning_effort"] == "xhigh"

    h_gem_bodies: list[dict[str, Any]] = []

    def _h_gem_handler(handler, req_info):
        h_gem_bodies.append(req_info["body"])
        _send_sse_frames(
            handler,
            [
                'data: {"candidates":[{"content":{"role":"model","parts":[{"thought":true,"text":"Hydrogen 深度思考"}]}}]}\n\n',
                'data: {"candidates":[{"content":{"role":"model","parts":[{"text":"Hydrogen 完成验证。"}]}}]}\n\n',
            ],
        )

    m_port, _, close_m = _start_magpie_mock_gateway(
        models=("ai-studio/gemini-4-hydrogen", "plain-chat-only"),
        model_meta={
            "ai-studio/gemini-4-hydrogen": {
                "reasoning": True,
                "reasoning_levels": ["low", "medium", "high", "xhigh"],
                "default_reasoning_level": "high",
            },
            "plain-chat-only": {"reasoning": False},
        },
        post_handler=_h_gem_handler,
    )
    try:
        saved_m = backend.save_agent_config(
            {
                "agent_reasoning_effort": "none",
                "upsert_provider": {
                    "name": "Magpie Hydrogen Sync",
                    "api_format": "openai",
                    "base_url": f"http://127.0.0.1:{m_port}/v1",
                    "models": ["ai-studio/gemini-4-hydrogen", "plain-chat-only"],
                    "default_model": "ai-studio/gemini-4-hydrogen",
                    "activate": True,
                },
            }
        )
        m_pid = saved_m["config"]["agent_active_provider_id"]
        fetched_m = backend.fetch_agent_models({"provider_id": m_pid})
        assert fetched_m["ok"] is True
        assert fetched_m["model_capabilities"]["ai-studio/gemini-4-hydrogen"] == {
            "reasoning": True,
            "levels": ["low", "medium", "high", "xhigh"],
            "default_level": "high",
        }
        assert fetched_m["model_capabilities"]["plain-chat-only"] == {
            "reasoning": False,
            "levels": [],
            "default_level": "",
        }
        st_h = backend.get_agent_state()
        assert st_h["brain"]["reasoning_spec_source"] == "gateway"
        assert st_h["brain"]["supported_reasoning_efforts"] == ["low", "medium", "high", "xhigh"]
        assert st_h["brain"]["default_reasoning_effort"] == "high"
        # 原保存为 none，但 Hydrogen 不支持关闭思考，应自动收敛为 auto
        assert st_h["brain"]["reasoning_effort"] == "auto"
        assert st_h["brain"]["reasoning_supported"] is True

        st_plain = backend.save_agent_config(
            {"select_provider_model": {"provider_id": m_pid, "model": "plain-chat-only"}}
        )
        assert st_plain["brain"]["reasoning_supported"] is False
        assert st_plain["brain"]["supported_reasoning_efforts"] == []

        drv_h = agent_tools.NativeLightDriver(
            base_url=f"http://127.0.0.1:{m_port}",
            api_key="AIzaSyHydrogen123",
            model="ai-studio/gemini-4-hydrogen",
            api_format="gemini",
            reasoning_effort="xhigh",
        )
        res_h = drv_h.run([{"role": "user", "content": "测试 Hydrogen thinkingLevel"}])
        assert res_h["ok"] is True
        assert h_gem_bodies[0]["generationConfig"]["thinkingConfig"] == {
            "thinkingLevel": "xhigh",
            "includeThoughts": True,
        }
    finally:
        close_m()


def test_multi_session_history_persistence_scrub_and_switch(isolated_env: SimpleNamespace) -> None:
    """校验多会话新建/自动标题/切换隔离/删除/超限裁剪、上下文统计及 agent_sessions.json 零密钥泄露。"""
    sess_secret = "sk-sess-history-secret-9988776655"
    sess_port, _, close_s = _start_magpie_mock_gateway(
        post_handler=lambda handler, req_info: _send_sse_frames(
            handler,
            [
                f'data: {{"choices":[{{"delta":{{"content":"收到问题：{(req_info["body"].get("messages") or [])[-1].get("content", "")[:20]} · 含密钥引用 {sess_secret}"}}}}]}}\n\n',
                "data: [DONE]\n\n",
            ],
        )
    )
    try:
        b1 = _make_backend(isolated_env, "multi_sess_state")
        b1.save_agent_config(
            {
                "agent_driver": "native",
                "agent_brain_source": "external_api",
                "agent_api_format": "openai",
                "agent_api_base_url": f"http://127.0.0.1:{sess_port}/v1",
                "agent_model": "gpt-5",
                "api_key": sess_secret,
            }
        )
        ev1: list[dict[str, Any]] = []
        unsub1 = b1.subscribe_events(lambda e: ev1.append(e))
        try:
            st0 = b1.get_agent_state()
            sid1 = st0["active_session_id"]
            assert sid1 and len(st0["sessions"]) == 1
            assert st0["context_stats"]["max_messages"] == 40

            # 第一会话发送 1 条消息（应自动从用户首句派生标题并脱敏写入磁盘）
            b1.send_agent_message({"text": "排查 L4 节点 ComfyUI 显存溢出问题的步骤有哪些？", "ctx_tag": "local_diag"})
            done1 = _wait_until(
                lambda: next((e for e in ev1 if e.get("phase") == "turn_done"), None),
                err_msg="第一会话未完成",
            )
            assert sess_secret not in json.dumps(done1, ensure_ascii=False)

            st1 = b1.get_agent_state()
            assert st1["sessions"][0]["id"] == sid1
            assert "排查 L4 节点" in st1["sessions"][0]["title"]
            assert st1["context_stats"]["message_count"] == 2
            assert st1["context_stats"]["approx_tokens"] > 0

            # 新建第二会话并发送不同消息
            new_res = b1.manage_agent_session({"action": "new"})
            assert new_res["ok"] is True and new_res["created"] is True
            sid2 = new_res["active_session_id"]
            assert sid2 != sid1
            assert len(new_res["state"]["messages"]) == 0

            ev1.clear()
            b1.send_agent_message({"text": "查看云盘中可用的 LoRA 资产清单"})
            _wait_until(
                lambda: next((e for e in ev1 if e.get("phase") == "turn_done"), None),
                err_msg="第二会话未完成",
            )
            st2 = b1.get_agent_state()
            assert len(st2["sessions"]) == 2
            assert len(st2["messages"]) == 2
            assert "LoRA" in st2["messages"][0]["content"]

            # 切回第一会话应完整恢复第一会话原消息与 ctx_tag
            sw1 = b1.manage_agent_session({"action": "switch", "session_id": sid1})
            assert sw1["ok"] is True and sw1["active_session_id"] == sid1
            assert sw1["state"]["messages"][0]["ctx_tag"] == "local_diag"
            assert "显存溢出" in sw1["state"]["messages"][0]["content"]
        finally:
            unsub1()

        # 验证磁盘文件绝无明文密钥，且新实例化后端可自动从磁盘恢复两段历史会话
        sess_file = isolated_env.tmp / "multi_sess_state_agent_sessions.json"
        assert sess_file.is_file()
        raw_disk = sess_file.read_text(encoding="utf-8")
        assert sess_secret not in raw_disk
        assert "agent_api_key" not in raw_disk

        b2 = _make_backend(isolated_env, "multi_sess_state")
        st_reboot = b2.get_agent_state()
        assert st_reboot["active_session_id"] == sid1
        assert len(st_reboot["sessions"]) == 2
        assert "显存溢出" in st_reboot["messages"][0]["content"]

        # 验证删除非活跃会话与重命名及超限裁剪 (MAX_AGENT_SESSIONS = 30)
        ren = b2.manage_agent_session({"action": "rename", "session_id": sid2, "title": "云盘 LoRA 核验"})
        assert ren["ok"] is True and any(s["title"] == "云盘 LoRA 核验" for s in ren["state"]["sessions"])
        dele = b2.manage_agent_session({"action": "delete", "session_id": sid2})
        assert dele["ok"] is True and len(dele["state"]["sessions"]) == 1

        # 批量写入超 30 个会话验证自动裁剪与活跃会话保留
        svc = b2._agent_svc
        with svc._lock:
            for idx in range(35):
                s_item = svc._new_empty_session(title=f"归档#{idx}", now_ts=1700000000 + idx)
                s_item["messages"].append({"id": f"m_{idx}", "role": "user", "content": f"ping {idx}", "ts": 1700000000 + idx})
                svc._sessions.append(s_item)
        svc._save_sessions_to_disk()
        st_trim = b2.get_agent_state()
        assert len(st_trim["sessions"]) == 30
        assert any(s["id"] == sid1 for s in st_trim["sessions"])

        # clear_all 重置为单条空白新对话
        clr = b2.manage_agent_session({"action": "clear_all"})
        assert clr["ok"] is True and len(clr["state"]["sessions"]) == 1 and clr["state"]["messages"] == []
    finally:
        close_s()


# =========================================================================
# 3. ClaudeCodeDriver stream-json 解析与 MCP Stdio 服务 (JSON-RPC 2.0)
# =========================================================================


def test_claude_code_driver_stream_json_and_cancel(isolated_env: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    """模拟假 claude CLI 输出标准 stream-json JSONL 验证事件解析与取消杀进程树。"""
    fake_py = isolated_env.tmp / "fake_claude.py"
    fake_py.write_text(
        """import json, sys, time
args = sys.argv[1:]
stdin_txt = "" if sys.stdin.isatty() else sys.stdin.read()
cfg_path = ""
for i, a in enumerate(args):
    if a == "--mcp-config" and i + 1 < len(args):
        cfg_path = args[i + 1]
cfg = json.loads(open(cfg_path, "r", encoding="utf-8").read()) if cfg_path else {}
mcp_args = cfg.get("mcpServers", {}).get("colab-studio", {}).get("args", [])
assert "mcp-stdio" in mcp_args, f"缺少 mcp-stdio 参数: {cfg}"

if any("HANG_TEST" in a for a in args):
    print(json.dumps({"type": "stream_event", "event": {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "开始长任务..."}}}), flush=True)
    time.sleep(30)
    sys.exit(0)

if "MULTILINE_STDIN" in stdin_txt:
    print(json.dumps({"type": "result", "is_error": False, "result": "多行 stdin 透传成功。"}, ensure_ascii=False), flush=True)
    sys.exit(0)

lines = [
    {"type": "stream_event", "event": {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "正在排查集群，"}}},
    {"type": "assistant", "message": {"content": [{"type": "tool_use", "id": "tu_1", "name": "mcp__colab-studio__get_cluster_and_pool_status", "input": {"refresh_pool": False}}]}},
    {"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": "tu_1", "is_error": False, "content": [{"type": "text", "text": "{\\"ok\\": true, \\"balance\\": 120.5}"}]}]}},
    {"type": "stream_event", "event": {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "当前余额充足。"}}},
    {"type": "result", "is_error": False, "result": "正在排查集群，当前余额充足。"},
]
for obj in lines:
    print(json.dumps(obj, ensure_ascii=False), flush=True)
""",
        encoding="utf-8",
    )
    fake_cmd = isolated_env.tmp / "fake_claude.cmd"
    fake_cmd.write_text(f'@echo off\r\n"{sys.executable}" "{fake_py}" %*\r\n', encoding="utf-8")

    # 校验 is_available 跟随 PATH 检测
    monkeypatch.setattr("shutil.which", lambda name: str(fake_cmd) if name == "claude" else None)
    assert agent_tools.ClaudeCodeDriver.is_available() is True

    captured_evts: list[dict[str, Any]] = []
    drv = agent_tools.ClaudeCodeDriver(
        ctl_script=PKG_DIR / "colab_ctl.py",
        model="claude-sonnet-4-20250514",
        on_event=captured_evts.append,
        claude_bin=str(fake_cmd),
    )
    res = drv.run([{"role": "user", "content": "查一下集群状态"}])
    assert res["ok"] is True and res["cancelled"] is False
    assert "当前余额充足" in res["text"]
    assert any(e["type"] == "tool_start" and e["name"] == "get_cluster_and_pool_status" and e["tier"] == "safe" for e in captured_evts)
    assert any(e["type"] == "tool_end" and e["name"] == "get_cluster_and_pool_status" and e["ok"] is True for e in captured_evts)
    assert any(e["type"] == "turn_done" and e["ok"] is True for e in captured_evts)

    # 验证多行提示词通过 .cmd 包装脚本经 stdin 完整透传（不被 cmd.exe 换行截断）
    res_ml = drv.run("MULTILINE_STDIN\n第二行测试内容")
    assert res_ml["ok"] is True and "多行 stdin 透传成功" in res_ml["text"]

    # 验证取消时能杀进程树并快速退出
    tok = CancelToken()
    hang_evts: list[dict[str, Any]] = []
    drv_hang = agent_tools.ClaudeCodeDriver(
        ctl_script=PKG_DIR / "colab_ctl.py",
        on_event=hang_evts.append,
        cancel_token=tok,
        claude_bin=str(fake_cmd),
    )
    out_box: list[dict[str, Any]] = []
    th = threading.Thread(target=lambda: out_box.append(drv_hang.run("HANG_TEST")), daemon=True)
    th.start()
    _wait_until(lambda: any(e["type"] == "delta" for e in hang_evts), timeout=5.0, err_msg="假 claude 未输出首帧")
    tok.cancel()
    th.join(timeout=5.0)
    assert not th.is_alive() and out_box and out_box[0]["cancelled"] is True


def test_mcp_stdio_server_subprocess_jsonrpc_and_10_tools(isolated_env: SimpleNamespace) -> None:
    """真实通过子进程 stdin/stdout 管道向 colab_ctl.py mcp-stdio 发送 JSON-RPC 2.0 验证 MCP 规范与 10 工具清单。"""
    fake_home = isolated_env.tmp / "fake_home"
    fake_home.mkdir(parents=True, exist_ok=True)
    env_sub = dict(os.environ)
    env_sub["HOME"] = str(fake_home)
    env_sub["USERPROFILE"] = str(fake_home)
    env_sub["APPDATA"] = str(fake_home / "AppData" / "Roaming")
    env_sub["LOCALAPPDATA"] = str(fake_home / "AppData" / "Local")
    env_sub["PYTHONUNBUFFERED"] = "1"

    proc = subprocess.Popen(
        [sys.executable, str(PKG_DIR / "colab_ctl.py"), "mcp-stdio"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env_sub,
    )
    assert proc.stdin is not None and proc.stdout is not None

    def _rpc(req: dict[str, Any], *, expect_reply: bool = True) -> dict[str, Any] | None:
        proc.stdin.write(json.dumps(req, ensure_ascii=False) + "\n")
        proc.stdin.flush()
        if not expect_reply:
            return None
        line = proc.stdout.readline()
        assert line, f"MCP 服务未返回响应: {proc.stderr.read() if proc.stderr else ''}"
        return json.loads(line)

    try:
        # 1) initialize
        init_resp = _rpc(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {"protocolVersion": "2024-11-05", "clientInfo": {"name": "pytest-mcp", "version": "1.0"}},
            }
        )
        assert init_resp is not None and init_resp["id"] == 1
        assert init_resp["result"]["protocolVersion"] == "2024-11-05"
        assert "tools" in init_resp["result"]["capabilities"]
        assert init_resp["result"]["serverInfo"]["name"] == "colab-studio-mcp"

        # 2) notifications/initialized（无响应）
        _rpc({"jsonrpc": "2.0", "method": "notifications/initialized"}, expect_reply=False)

        # 3) tools/list（必须精确返回 10 个工具）
        list_resp = _rpc({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        assert list_resp is not None and list_resp["id"] == 2
        tools = list_resp["result"]["tools"]
        assert len(tools) == 10
        expected_names = [s["name"] for s in agent_tools.AGENT_TOOL_SPECS]
        assert [t["name"] for t in tools] == expected_names
        for t in tools:
            assert isinstance(t.get("inputSchema"), dict)
            assert t["inputSchema"].get("type") == "object"

        # 4) tools/call -> run_environment_doctor（真实走 colab_ctl.py doctor --json 隔离沙箱执行）
        doc_resp = _rpc(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {"name": "run_environment_doctor", "arguments": {"profile": "anima"}},
            }
        )
        assert doc_resp is not None and doc_resp["id"] == 3
        content_items = doc_resp["result"]["content"]
        assert isinstance(content_items, list) and content_items[0]["type"] == "text"
        doc_payload = json.loads(content_items[0]["text"])
        assert "report" in doc_payload or "ok" in doc_payload

        # 5) tools/call -> exec_remote_colab 拦截危险命令
        blk_resp = _rpc(
            {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": {"name": "exec_remote_colab", "arguments": {"code": "rm -rf /", "mode": "bash"}},
            }
        )
        assert blk_resp is not None and blk_resp["result"]["isError"] is True
        blk_payload = json.loads(blk_resp["result"]["content"][0]["text"])
        assert blk_payload.get("blocked") is True
    finally:
        try:
            proc.stdin.close()
        except Exception:
            pass
        rc = proc.wait(timeout=10)
        assert rc == 0


# =========================================================================
# 4. 真实 PyWebView 双窗真渲染 + 副驾零溢出、联动与 3 张高清截图验收
# =========================================================================


def _run_pywebview_copilot_child(work_dir: Path, shot_dir: Path) -> int:
    """在独立子进程中拉起 392×548 桌宠 + 1080×700 / 920×600 Studio 双窗完成副驾全链路真渲染验收。"""
    import shutil
    import webview

    # 强制在真渲染验收中校验未安装 claude CLI 时前端正确灰置按钮（与宿主机 PATH 隔离）
    orig_which = shutil.which
    shutil.which = lambda name, *a, **kw: None if name == "claude" else orig_which(name, *a, **kw)  # type: ignore[assignment]

    shot_dir.mkdir(parents=True, exist_ok=True)
    state_file = work_dir / "copilot_pet_state.json"
    dl_root = work_dir / "downloads"
    cf_file = work_dir / ".cf_tunnel_secret.json"
    dl_root.mkdir(parents=True, exist_ok=True)

    # 准备 2 张真实画廊 PNG 作品以供画廊 Inspector 与「交由副驾改词重抽」测试
    _create_comfy_artwork_png(
        dl_root / "atelier_girl_01.png",
        positive="1girl, lavender eyes, sunlit atelier, cel shading, crisp lineart",
        seed=11223344,
        steps=28,
    )
    _create_comfy_artwork_png(
        dl_root / "mecha_dock_02.png",
        positive="cyberpunk hangar, volumetric dusk light, sharp detail",
        seed=55667788,
        steps=30,
    )

    backend = cb.ColabPetBackend(
        state_path=state_file,
        download_root=dl_root,
        cf_config_path=cf_file,
    )
    backend.run_preflight = lambda deep=True, **_kw: {  # type: ignore[method-assign]
        "ok": True,
        "ok_count": 4,
        "warn_count": 1,
        "fail_count": 0,
        "versions": {"Python": "3.14", "PyWebView": "5.x"},
        "items": [
            {"name": "google-colab-cli", "category": "CLI", "status": "ok", "summary": "已就绪", "detail": "0.4.2"},
            {"name": "Google Ultra 号池", "category": "账号", "status": "ok", "summary": "2 个账号在线", "detail": "210 CU"},
            {"name": "Cloudflare Tunnel", "category": "网络", "status": "ok", "summary": "固定域名已配置", "detail": "ok"},
            {"name": "Windows 凭据保险箱", "category": "安全", "status": "ok", "summary": "DPAPI / Mock Vault 已启用", "detail": "ok"},
            {"name": "Claude Code CLI", "category": "可选", "status": "warn", "summary": "未安装（当前由 Native 引擎接管）", "detail": "Native Ready"},
        ],
    }
    backend.start_instance = lambda cfg=None: {"ok": True, "started": True, "gpu": (cfg or {}).get("gpu", "L4"), "message": "实例开机流程已触发"}  # type: ignore[method-assign]

    # 写入多条含 ERROR 的沙箱诊断日志以供控制台「副驾诊断报错」测试
    diag = DiagLogger()
    diag.info("studio", "Colab Studio 同进程服务启动完成")
    diag.warn("tunnel", "LLM 隧道首次心跳延迟 420ms，已自动重试成功")
    diag.error("comfy", "KSampler CUDA out of memory: tried to allocate 2.40 GiB on L4")
    diag.info("monitor", "算力池巡检正常，剩余余额 218.40 CU")

    # 注入活跃号池快照（含 Ultra 896.25 CU + Pro 200.0 CU 混合账号）以便前端完整渲染
    snap = backend.get_cached_snapshot()
    snap.update(
        {
            "ok": True,
            "is_running": True,
            "balance": 896.25,
            "baseline_balance": 1000.0,
            "balance_pct": 89.63,
            "rate_hourly": 2.46,
            "remaining_hours": 364.3,
            "day_consumed_cu": 6.80,
            "status_level": "running",
            "status_label": "双节点运行中",
            "updated_str": "04:40:00",
            "cluster": {
                "active": False,
                "can_split": True,
                "total_pool_balance": 1096.25,
                "total_pool_baseline": 1200.0,
                "pool_balance_pct": 91.35,
                "pool_tier_summary": "1 Ultra + 1 Pro",
                "pool_healthy_count": 2,
            },
            "onboarding": {
                "auth_missing": False,
                "active_account_id": "acct_main",
                "active_account_label": "Ultra主力",
                "accounts": [
                    {
                        "id": "acct_main",
                        "label": "Ultra主力",
                        "email_masked": "ul***08@gmail.com",
                        "last_balance": 896.25,
                        "last_is_running": True,
                        "is_active": True,
                        "tier": "ultra",
                        "tier_label": "Ultra",
                        "baseline_balance": 1000.0,
                        "eligible_gpus": ["H100", "G4", "A100", "L4", "T4"],
                        "cluster_role": "comfy",
                    },
                    {
                        "id": "acct_pro",
                        "label": "demo_acct_01",
                        "email_masked": "de***23@gmail.com",
                        "last_balance": 200.00,
                        "last_is_running": False,
                        "is_active": False,
                        "tier": "pro",
                        "tier_label": "Pro",
                        "baseline_balance": 200.0,
                        "eligible_gpus": ["G4", "A100", "L4", "T4"],
                        "ineligible_gpus": ["H100"],
                        "cluster_role": None,
                    },
                ],
            },
            "comfy": {
                "online": True,
                "comfy_online": True,
                "llm_online": False,
                "service_kind": "comfy",
                "active_profile": "anima",
                "gpu_name": "NVIDIA L4",
                "vram_used_gb": 11.4,
                "vram_total_gb": 24.0,
                "vram_pct": 47.5,
                "running_count": 0,
                "pending_count": 0,
            },
        }
    )
    backend.last_snapshot = json.loads(json.dumps(snap))

    # 启动本地 OpenAI SSE 模拟副驾后端：
    # - 常规指令：第 1 轮调用 get_cluster_and_pool_status -> 第 2 轮调用 manage_compute_instance (触发红卡审批) -> 第 3 轮流式输出 Markdown 结论
    # - 诊断指令：直接调用 tail_instance_logs 并输出根因分析
    sse_state = {"turn": 0}

    def _ui_sse_handler(handler, req_info):
        msgs = req_info["body"].get("messages") or []
        last_user = next((str(m.get("content") or "") for m in reversed(msgs) if m.get("role") == "user"), "")
        has_tool_cluster = any(m.get("role") == "tool" and m.get("name") == "get_cluster_and_pool_status" for m in msgs)
        has_tool_manage = any(m.get("role") == "tool" and m.get("name") == "manage_compute_instance" for m in msgs)
        has_tool_logs = any(m.get("role") == "tool" and m.get("name") == "tail_instance_logs" for m in msgs)

        if "诊断" in last_user or "日志" in last_user:
            if not has_tool_logs:
                _send_sse_frames(
                    handler,
                    [
                        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_ui_log","type":"function","function":{"name":"tail_instance_logs","arguments":"{\\"source\\": \\"local_diag\\", \\"lines\\": 50}"}}]}}]}\n\n',
                        "data: [DONE]\n\n",
                    ],
                )
                return
            _send_sse_frames(
                handler,
                [
                    'data: {"choices":[{"delta":{"content":"日志根因定位：检测到 `KSampler CUDA out of memory`（L4 显存 24G 不足），建议降低 `batch_size` 或切换至 G4 节点。"}}]}\n\n',
                    "data: [DONE]\n\n",
                ],
            )
            return

        if not has_tool_cluster:
            sse_state["turn"] += 1
            _send_sse_frames(
                handler,
                [
                    'data: {"choices":[{"delta":{"content":"先为你核验当前号池与算力状态。"}}]}\n\n',
                    'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_ui_1","type":"function","function":{"name":"get_cluster_and_pool_status","arguments":"{\\"refresh_pool\\": false}"}}]}}]}\n\n',
                    "data: [DONE]\n\n",
                ],
                delay_sec=0.02,
            )
            return

        if not has_tool_manage:
            _send_sse_frames(
                handler,
                [
                    'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_ui_2","type":"function","function":{"name":"manage_compute_instance","arguments":"{\\"action\\": \\"start\\", \\"profiles\\": [\\"anima\\"], \\"gpu\\": \\"L4\\"}"}}]}}]}\n\n',
                    "data: [DONE]\n\n",
                ],
                delay_sec=0.02,
            )
            return

        _send_sse_frames(
            handler,
            [
                'data: {"choices":[{"delta":{"reasoning_content":"1. 已完成号池额度核验与 L4 节点审批；\\n2. 整理后续显存监控建议与画廊重绘入口。"}}]}\n\n',
                'data: {"choices":[{"delta":{"content":"已完成部署放行，核心结论：\\n- 当前集群：`L4` 节点在线，余额 `128.40 CU`\\n- 建议：可在画廊选中作品直接改词重抽\\n```bash\\nnvidia-smi --query-gpu=memory.used --format=csv\\n```"}}]}\n\n',
                "data: [DONE]\n\n",
            ],
            delay_sec=0.02,
        )

    sse_port, _sse_reqs, close_sse = _start_dual_protocol_sse_server(_ui_sse_handler)
    magpie_port, _magpie_reqs, close_magpie = _start_magpie_mock_gateway(
        models=(
            "ai-studio/gemini-4-hydrogen",
            "gpt-5",
            "gpt-4.1",
            "claude-sonnet-4-5",
            "gemini-2.5-pro",
            "qwen3-coder-plus",
            "deepseek-r1",
            "broken-canary",
        ),
        model_meta={
            "ai-studio/gemini-4-hydrogen": {
                "reasoning": True,
                "reasoning_levels": ["low", "medium", "high", "xhigh"],
                "default_reasoning_level": "high",
            }
        },
        broken_models={"broken-canary"},
    )

    # 配置真实 Vault 隔离密钥与本地 SSE 大脑地址
    ui_secret = "sk-ui-copilot-vault-key-77889900"
    backend.save_agent_config(
        {
            "agent_driver": "native",
            "agent_brain_source": "external_api",
            "agent_api_format": "openai",
            "agent_api_base_url": f"http://127.0.0.1:{sse_port}/v1",
            "agent_model": "gpt-4o-mini",
            "agent_auto_approve_billing": False,
            "api_key": ui_secret,
        }
    )

    stop_event = threading.Event()
    api = pet_app.PetBridgeApi(backend, stop_event)

    # 验证 8 个副驾及多供应商发现方法均暴露在 PetBridgeApi 且无 *args/**kwargs
    for m_name in (
        "get_agent_state",
        "save_agent_config",
        "fetch_agent_models",
        "probe_agent_provider",
        "send_agent_message",
        "resolve_agent_confirm",
        "cancel_agent_turn",
        "clear_agent_history",
        "manage_agent_session",
    ):
        fn = getattr(api, m_name, None)
        assert callable(fn), f"PetBridgeApi 缺少副驾方法: {m_name}"
        spec = inspect.getfullargspec(fn.__func__)
        assert spec.varargs is None and spec.varkw is None, f"副驾方法参数含可变参数: {m_name}"

    httpd = pet_app._ThreadingHTTPServer(("127.0.0.1", 0), pet_app._PetAssetHandler)
    httpd.media_root = dl_root  # type: ignore[attr-defined]
    asset_port = httpd.server_address[1]
    api._asset_base_url = f"http://127.0.0.1:{asset_port}"
    threading.Thread(target=httpd.serve_forever, daemon=True).start()

    pet_win = webview.create_window(
        title="Colab 桌宠 · 副驾双窗验收",
        url=f"http://127.0.0.1:{asset_port}/ui.html",
        js_api=api,
        width=pet_app.BASE_W,
        height=pet_app.BASE_H,
        x=60,
        y=60,
        resizable=False,
        frameless=True,
        easy_drag=False,
        shadow=False,
        transparent=False,
        background_color=pet_app.EDGE_BG_HEX,
        focus=True,
        on_top=True,
    )
    api._bind_window(pet_win)
    backend._event_cb = lambda evt: None if stop_event.is_set() else pet_app._push_event_to_ui(pet_win, api, evt)
    errs: list[BaseException] = []

    def _cdp_call(win: webview.Window, method: str, params: dict[str, Any], *, allow_dispose: bool = False) -> Any:
        form = win.native
        assert form is not None
        t_box: list[Any] = []
        e_box: list[BaseException] = []
        p_str = json.dumps(params, ensure_ascii=False)

        def _start() -> None:
            try:
                wv = next(c for c in form.Controls if type(c).__name__ == "WebView2")
                t_box.append(wv.CoreWebView2.CallDevToolsProtocolMethodAsync(method, p_str))
            except BaseException as ex:
                e_box.append(ex)

        try:
            form.Invoke(pet_app.DotNetAction(_start))
        except BaseException as exc:
            if allow_dispose and ("ObjectDisposed" in repr(exc) or getattr(form, "IsDisposed", False)):
                return ""
            raise
        if e_box:
            raise e_box[0]
        task = t_box[0]
        deadline = time.monotonic() + 8.0
        while not task.IsCompleted:
            if allow_dispose and getattr(form, "IsDisposed", False):
                return ""
            if time.monotonic() > deadline:
                raise TimeoutError(f"CDP 调用超时: {method}")
            time.sleep(0.005)
        res_box: list[Any] = []

        def _read_res() -> None:
            if task.IsFaulted:
                e_box.append(RuntimeError(str(task.Exception)))
            else:
                res_box.append(task.Result)

        if allow_dispose and getattr(form, "IsDisposed", False):
            return ""
        try:
            form.Invoke(pet_app.DotNetAction(_read_res))
        except BaseException as exc:
            if allow_dispose and ("ObjectDisposed" in repr(exc) or getattr(form, "IsDisposed", False)):
                return ""
            raise
        if e_box:
            raise e_box[0]
        return res_box[0] if res_box else ""

    def _click_sel(win: webview.Window, sel: str, *, button: str = "left", allow_dispose: bool = False) -> None:
        pt = win.evaluate_js(
            f"(() => {{ const el = document.querySelector({sel!r});"
            " if (!el) return null;"
            " const r = el.getBoundingClientRect();"
            " return {x: r.left + r.width / 2, y: r.top + r.height / 2, w: r.width, h: r.height}; })()"
        )
        assert pt and pt["w"] > 0 and pt["h"] > 0, f"目标元素不可见或未找到: {sel}"
        fx, fy = float(pt["x"]), float(pt["y"])
        btn_mask = 2 if button == "right" else 1
        _cdp_call(win, "Input.dispatchMouseEvent", {"type": "mouseMoved", "x": fx, "y": fy, "button": "none", "buttons": 0})
        _cdp_call(win, "Input.dispatchMouseEvent", {"type": "mousePressed", "x": fx, "y": fy, "button": button, "buttons": btn_mask, "clickCount": 1})
        time.sleep(0.025)
        _cdp_call(
            win,
            "Input.dispatchMouseEvent",
            {"type": "mouseReleased", "x": fx, "y": fy, "button": button, "buttons": 0, "clickCount": 1},
            allow_dispose=allow_dispose,
        )

    def _send_ctrl_k(win: webview.Window) -> None:
        for ev_type in ("rawKeyDown", "keyUp"):
            _cdp_call(
                win,
                "Input.dispatchKeyEvent",
                {
                    "type": ev_type,
                    "modifiers": 2,
                    "windowsVirtualKeyCode": 75,
                    "nativeVirtualKeyCode": 75,
                    "key": "k",
                    "code": "KeyK",
                },
            )
            time.sleep(0.02)

    def _save_cdp_screenshot(win: webview.Window, filename: str) -> Path:
        raw_json = _cdp_call(win, "Page.captureScreenshot", {"format": "png"})
        parsed = json.loads(raw_json)
        png_bytes = base64.b64decode(parsed["data"])
        target = shot_dir / filename
        target.write_bytes(png_bytes)
        return target

    def _audit_studio_copilot(win: webview.Window, label: str, *, expect_copilot_open: bool = True) -> None:
        diag = win.evaluate_js(
            "(() => {"
            "  const shell = document.getElementById('studio-shell');"
            "  const drawer = document.getElementById('copilot-drawer');"
            "  const ws = document.querySelector('.studio-workspace');"
            "  const pane = document.querySelector('.view-pane.active');"
            "  const scroller = pane ? pane.querySelector('.view-scroll') : null;"
            "  const docOverflow = document.documentElement.scrollWidth - window.innerWidth;"
            "  const wsOverflow = ws ? (ws.scrollWidth - ws.clientWidth) : 0;"
            "  const paneOverflow = pane ? (pane.scrollWidth - pane.clientWidth) : 0;"
            "  const scOverflow = scroller ? (scroller.scrollWidth - scroller.clientWidth) : 0;"
            "  const drRect = drawer ? drawer.getBoundingClientRect() : {width: 0, right: 0};"
            "  const wsRect = ws ? ws.getBoundingClientRect() : {width: 0, right: 0};"
            "  const bodyText = document.body ? (document.body.innerText || '') : '';"
            "  const clippedButtons = [];"
            "  document.querySelectorAll('.sidebar-nav .nav-item, .titlebar-actions button, .view-pane.active button:not(.hidden), .copilot-drawer button:not(.hidden)').forEach(b => {"
            "    if (b.closest('.hidden') || b.closest('[aria-hidden=\"true\"]')) return;"
            "    const r = b.getBoundingClientRect();"
            "    if (r.width === 0 || r.height === 0) return;"
            "    if (b.scrollWidth > b.clientWidth + 2 || r.right > window.innerWidth + 1) {"
            "      clippedButtons.push({id: b.id || '', cls: b.className, txt: (b.textContent || '').trim(), sw: b.scrollWidth, cw: b.clientWidth, r: r.right, iw: window.innerWidth});"
            "    }"
            "  });"
            "  const navTrunc = [];"
            "  document.querySelectorAll('.nav-label').forEach(l => {"
            "    if (l.scrollWidth > l.clientWidth + 1) navTrunc.push({txt: l.textContent, sw: l.scrollWidth, cw: l.clientWidth});"
            "  });"
            "  return {"
            "    copilotOpen: Boolean(shell && shell.classList.contains('copilot-open')),"
            "    drawerW: drRect.width,"
            "    drawerRight: drRect.right,"
            "    wsW: wsRect.width,"
            "    innerW: window.innerWidth,"
            "    docOverflow, wsOverflow, paneOverflow, scOverflow,"
            "    bodyText,"
            "    clippedButtons, navTrunc,"
            "    theme: document.documentElement.dataset.theme"
            "  };"
            "})()"
        )
        assert diag["copilotOpen"] is expect_copilot_open, f"[{label}] 副驾展开态不符: {diag}"
        assert diag["docOverflow"] <= 0, f"[{label}] 窗口横向溢出: {diag}"
        assert diag["wsOverflow"] <= 1, f"[{label}] 主工作区横向溢出: {diag}"
        assert diag["paneOverflow"] <= 1, f"[{label}] 当前视图横向溢出: {diag}"
        assert diag["scOverflow"] <= 1, f"[{label}] 滚动区横向溢出: {diag}"
        assert diag["clippedButtons"] == [], f"[{label}] 按钮文本截断或越界: {diag['clippedButtons']}"
        assert diag["navTrunc"] == [], f"[{label}] 侧栏导航截断: {diag['navTrunc']}"
        if expect_copilot_open:
            assert diag["drawerW"] >= 330 and diag["drawerRight"] <= diag["innerW"] + 1, diag
            assert diag["wsW"] >= 320, f"[{label}] 主工作区被副驾过度挤压: {diag}"
        emoji_hits = _EMOJI_RE.findall(diag["bodyText"])
        assert not emoji_hits, f"[{label}] Studio DOM 检测到禁止出现的 Emoji 字符: {emoji_hits}"

    def _worker() -> None:
        try:
            pet_win.events.loaded.wait(timeout=10.0)
            _wait_until(lambda: pet_win.native is not None, timeout=6.0, err_msg="pet_win.native 未就绪")
            pet_form = pet_win.native
            pet_form.Invoke(pet_app.DotNetAction(lambda: (setattr(pet_form, "TopMost", True), pet_form.Show(), pet_form.Activate())))
            pet_win.resize(pet_app.BASE_W, pet_app.BASE_H)

            _wait_until(
                lambda: pet_win.evaluate_js("Boolean(window.pywebview && window.pywebview.api && typeof window.renderState === 'function' && latestSnap)"),
                timeout=8.0,
                err_msg="桌宠 pywebviewready 超时",
            )
            pet_app._push_to_ui(pet_win, snap)
            pet_win.evaluate_js("setPage(0); toggleBubble(true);")
            time.sleep(0.35)

            # 1) 严格断言桌宠第 1 页全池总积分 1096.25 + Σ标 与 5 页 oy == 0，并留存 06_pet_pool_total_cu.png
            _wait_until(
                lambda: pet_win.evaluate_js("el('cu-balance').textContent === '1096.25' && !el('cu-sigma').classList.contains('hidden')"),
                timeout=4.0,
                err_msg="桌宠第 1 页未刷新为全池总积分 1096.25",
            )
            time.sleep(0.15)
            pet_p1_diag = pet_win.evaluate_js(
                "({cu: el('cu-balance').textContent, sigmaHidden: el('cu-sigma').classList.contains('hidden'), miniCu: el('mini-cu').textContent, lbl: el('balance-label-k').textContent, pct: el('balance-pct-text').textContent})"
            )
            assert pet_p1_diag["cu"] == "1096.25" and pet_p1_diag["sigmaHidden"] is False, pet_p1_diag
            assert pet_p1_diag["miniCu"] == "Σ 1096.25 CU" and "全池水位" in pet_p1_diag["lbl"], pet_p1_diag
            _save_cdp_screenshot(pet_win, "06_pet_pool_total_cu.png")

            for pg_idx in range(5):
                pet_win.evaluate_js(f"setPage({pg_idx});")
                time.sleep(0.18)
                pet_diag = pet_win.evaluate_js(
                    "({stageSh: el('stage').scrollHeight, pages: [...document.querySelectorAll('#pages-track > .page-body')].map((p, i) => ({i, oy: p.scrollHeight - p.clientHeight}))})"
                )
                assert pet_diag["stageSh"] <= 548, pet_diag
                for pg in pet_diag["pages"]:
                    assert pg["oy"] == 0, f"桌宠 Page {pg['i'] + 1} 纵向溢出 oy={pg['oy']}"
            pet_win.evaluate_js("setPage(0);")
            time.sleep(0.15)

            # 2) CDP 右键点击桌宠 #tool-studio -> 秒开 Studio 并自动展开 #copilot-drawer
            _click_sel(pet_win, "#tool-studio", button="right")
            _wait_until(
                lambda: api._studio_window is not None and api._studio_hwnd(),
                timeout=10.0,
                err_msg="右键 #tool-studio 未成功拉起 Studio 窗口",
            )
            studio_win = api._studio_window
            assert studio_win is not None
            _wait_until(lambda: studio_win.native is not None, timeout=6.0, err_msg="studio_win.native 未就绪")
            _wait_until(
                lambda: studio_win.evaluate_js(
                    "Boolean(window.pywebview && window.pywebview.api && typeof window.openStudioCopilot === 'function' && document.getElementById('studio-shell').classList.contains('copilot-open'))"
                ),
                timeout=10.0,
                err_msg="右键 #tool-studio 打开后未自动展开 #copilot-drawer",
            )
            time.sleep(0.4)

            # 检查配置 Popover 中外部 API Key 绝不回显明文且 Claude 按钮灰置
            _click_sel(studio_win, "#copilot-brain-pill")
            time.sleep(0.2)
            pop_diag = studio_win.evaluate_js(
                "({open: !document.getElementById('copilot-config-popover').classList.contains('hidden'),"
                " keyVal: document.getElementById('inp-agent-api-key').value,"
                " clearVis: !document.getElementById('btn-agent-clear-key').classList.contains('hidden'),"
                " claudeDis: document.getElementById('btn-cfg-driver-claude').disabled})"
            )
            assert pop_diag["open"] is True and pop_diag["keyVal"] == "" and pop_diag["clearVis"] is True, pop_diag
            assert pop_diag["claudeDis"] is True, pop_diag
            _click_sel(studio_win, "#btn-agent-cfg-close")
            time.sleep(0.2)

            # 2.5) 验收底部思考强度 Popover：打开并切换到「高 (High)」，验证持久化与界面状态同步
            _click_sel(studio_win, "#btn-copilot-effort")
            _wait_until(
                lambda: not studio_win.evaluate_js("document.getElementById('copilot-effort-popover').classList.contains('hidden')"),
                timeout=3.0,
                err_msg="打开模型思考强度浮层失败",
            )
            time.sleep(0.2)
            _audit_studio_copilot(studio_win, "effort-popover-light", expect_copilot_open=True)
            _click_sel(studio_win, '#effort-options-list .effort-option[data-effort="high"]')
            _wait_until(
                lambda: studio_win.evaluate_js(
                    "document.getElementById('copilot-effort-popover').classList.contains('hidden')"
                    " && (document.getElementById('copilot-effort-label').textContent || '').includes('高')"
                ),
                timeout=4.0,
                err_msg="选择高思考强度后未自动收起或未更新按钮标签",
            )
            assert backend.get_agent_state()["config"]["agent_reasoning_effort"] == "high"

            # 3) 通过输入框发起对话 -> 工具微卡片 + 红色计费审批卡 -> 切深色主题截图 02_copilot_confirm_card_dark.png 与 08_antigravity_copilot_dark.png
            studio_win.evaluate_js(
                "(() => { const inp = document.getElementById('copilot-input'); inp.value = '检查算力状态并开机部署 L4'; inp.dispatchEvent(new Event('input')); })()"
            )
            _click_sel(studio_win, "#btn-copilot-send")
            _wait_until(
                lambda: studio_win.evaluate_js(
                    "Boolean(document.querySelector('.copilot-tool-card') && document.querySelector('.copilot-confirm-card .btn-confirm-approve'))"
                ),
                timeout=10.0,
                err_msg="UI 未渲染出工具微卡片或红色计费审批卡",
            )
            time.sleep(0.35)
            assert any(r.get("body", {}).get("reasoning_effort") == "high" for r in _sse_reqs), _sse_reqs

            # 切换深空暗房主题并保存审批卡高清截图
            _click_sel(studio_win, "#btn-theme-toggle")
            time.sleep(0.4)
            assert studio_win.evaluate_js("document.documentElement.dataset.theme") == "dark"
            _audit_studio_copilot(studio_win, "confirm-dark-1080", expect_copilot_open=True)
            _save_cdp_screenshot(studio_win, "02_copilot_confirm_card_dark.png")

            # 深色模式下展开思考强度浮层截图 08_antigravity_copilot_dark.png，随后收起
            _click_sel(studio_win, "#btn-copilot-effort")
            _wait_until(
                lambda: not studio_win.evaluate_js("document.getElementById('copilot-effort-popover').classList.contains('hidden')"),
                timeout=3.0,
                err_msg="深色模式下打开思考强度浮层失败",
            )
            time.sleep(0.25)
            _audit_studio_copilot(studio_win, "effort-popover-dark", expect_copilot_open=True)
            _save_cdp_screenshot(studio_win, "08_antigravity_copilot_dark.png")
            _click_sel(studio_win, "#btn-copilot-effort")
            _wait_until(
                lambda: studio_win.evaluate_js("document.getElementById('copilot-effort-popover').classList.contains('hidden')"),
                timeout=3.0,
                err_msg="再次点击未能收起思考强度浮层",
            )
            time.sleep(0.2)

            # 4) CDP 点击审批卡 [批准执行] -> 等待流式思考链 + Markdown 回答收束 -> 切回浅色主题保存 01_copilot_light_1080.png
            _click_sel(studio_win, ".copilot-confirm-card .btn-confirm-approve")
            _wait_until(
                lambda: studio_win.evaluate_js(
                    "Boolean(document.querySelector('.copilot-confirm-card.resolved-approved'))"
                    " && Boolean(document.querySelector('.copilot-think-card'))"
                    " && document.querySelectorAll('.copilot-msg.assistant .copilot-code-block').length >= 1"
                    " && !document.getElementById('btn-copilot-send').classList.contains('is-stop')"
                ),
                timeout=10.0,
                err_msg="点击 [批准执行] 后未完成审批收敛、思考卡片或最终 Markdown 流式回复",
            )
            think_open_after_done = studio_win.evaluate_js("Boolean(document.querySelector('.copilot-think-card').open)")
            assert think_open_after_done is False, "流式正文输出完毕后思考过程卡片未自动收起"
            time.sleep(0.35)
            _click_sel(studio_win, "#btn-theme-toggle")
            time.sleep(0.4)
            assert studio_win.evaluate_js("document.documentElement.dataset.theme") == "light"
            _audit_studio_copilot(studio_win, "copilot-light-1080", expect_copilot_open=True)
            _save_cdp_screenshot(studio_win, "01_copilot_light_1080.png")

            # 5) 验证 Ctrl+K 收起 / 展开，以及 open_studio_window("copilot") 复用直达
            _send_ctrl_k(studio_win)
            _wait_until(
                lambda: not studio_win.evaluate_js("document.getElementById('studio-shell').classList.contains('copilot-open')"),
                timeout=3.0,
                err_msg="Ctrl+K 未收起副驾抽屉",
            )
            time.sleep(0.3)
            _send_ctrl_k(studio_win)
            _wait_until(
                lambda: studio_win.evaluate_js("document.getElementById('studio-shell').classList.contains('copilot-open')"),
                timeout=3.0,
                err_msg="再次 Ctrl+K 未重新展开副驾抽屉",
            )
            time.sleep(0.25)

            # 再验证关闭后调 open_studio_window("copilot") 也能复用窗口并重新展开
            _send_ctrl_k(studio_win)
            time.sleep(0.25)
            reuse_ret = api.open_studio_window("copilot")
            assert reuse_ret["ok"] is True and reuse_ret["reused"] is True
            _wait_until(
                lambda: studio_win.evaluate_js("document.getElementById('studio-shell').classList.contains('copilot-open')"),
                timeout=3.0,
                err_msg="open_studio_window('copilot') 未重新展开副驾抽屉",
            )

            # 6) 跨视图联动 1：画廊选中作品 -> 点击「交由副驾改词重抽」(#btn-gal-copilot)
            _click_sel(studio_win, "#nav-gallery")
            _wait_until(
                lambda: studio_win.evaluate_js("document.getElementById('view-gallery').classList.contains('active') && document.querySelectorAll('#gallery-masonry .gallery-card').length >= 2"),
                timeout=5.0,
                err_msg="画廊列表未加载",
            )
            # 等待 View Transition 动画与缩略图布局稳定后再点击作品卡片
            time.sleep(0.5)
            _click_sel(studio_win, '.gallery-card[data-rel-path="atelier_girl_01.png"]')
            _wait_until(
                lambda: studio_win.evaluate_js("document.getElementById('gallery-inspector').classList.contains('open')"),
                timeout=5.0,
                err_msg="画廊 Inspector 未展开",
            )
            time.sleep(0.3)
            _click_sel(studio_win, "#btn-gal-copilot")
            time.sleep(0.3)
            ctx_diag = studio_win.evaluate_js(
                "({val: document.getElementById('copilot-input').value,"
                " stripHidden: document.getElementById('copilot-ctx-strip').classList.contains('hidden'),"
                " hasThumb: Boolean(document.querySelector('#copilot-ctx-strip img.copilot-ctx-thumb')),"
                " chipTxt: (document.getElementById('copilot-ctx-strip').textContent || '').trim()})"
            )
            assert "atelier_girl_01.png" in (ctx_diag["val"] or ""), f"交由副驾改词重抽未代入作品路径: {ctx_diag}"
            assert ctx_diag["stripHidden"] is False and ctx_diag["hasThumb"] is True and "atelier_girl_01.png" in ctx_diag["chipTxt"], ctx_diag
            _audit_studio_copilot(studio_win, "antigravity-ctx-light", expect_copilot_open=True)
            _save_cdp_screenshot(studio_win, "07_antigravity_copilot_light.png")

            # 7) 跨视图联动 2：控制台点击「副驾诊断报错」(#btn-console-copilot)
            _click_sel(studio_win, "#nav-console")
            _wait_until(
                lambda: studio_win.evaluate_js(
                    "document.getElementById('view-console').classList.contains('active') && document.querySelectorAll('#console-log-view .log-line').length >= 3"
                ),
                timeout=5.0,
                err_msg="控制台日志未加载",
            )
            time.sleep(0.5)
            before_msg_cnt = int(studio_win.evaluate_js("document.querySelectorAll('.copilot-msg.assistant').length") or 0)
            _click_sel(studio_win, "#btn-console-copilot")
            _wait_until(
                lambda: int(studio_win.evaluate_js("document.querySelectorAll('.copilot-msg.assistant').length") or 0) > before_msg_cnt
                and not studio_win.evaluate_js("document.getElementById('btn-copilot-send').classList.contains('is-stop')"),
                timeout=10.0,
                err_msg="点击「副驾诊断报错」后未完成日志诊断输出",
            )
            time.sleep(0.25)

            # 7.5) 验收 Magpie 风格「自定义供应商与模型导入」模态弹窗与多模型快切交互，并截图 04_provider_modal_magpie.png
            _click_sel(studio_win, "#copilot-brain-pill")
            _wait_until(
                lambda: not studio_win.evaluate_js("document.getElementById('copilot-config-popover').classList.contains('hidden')"),
                timeout=3.0,
                err_msg="打开快切浮层失败",
            )
            _click_sel(studio_win, "#btn-open-provider-modal")
            _wait_until(
                lambda: not studio_win.evaluate_js("document.getElementById('provider-modal').classList.contains('hidden')"),
                timeout=3.0,
                err_msg="未打开供应商与模型导入弹窗",
            )
            time.sleep(0.25)
            _click_sel(studio_win, "#btn-prov-tab-new")
            time.sleep(0.15)
            magpie_gw_url = f"http://127.0.0.1:{magpie_port}/v1"
            studio_win.evaluate_js(
                f"(() => {{"
                f"  const n = document.getElementById('inp-prov-name');"
                f"  const u = document.getElementById('inp-prov-base-url');"
                f"  n.value = 'Magpie 本机网关'; n.dispatchEvent(new Event('input'));"
                f"  u.value = {magpie_gw_url!r}; u.dispatchEvent(new Event('input'));"
                f"}})()"
            )
            _click_sel(studio_win, "#btn-prov-fetch-models")
            _wait_until(
                lambda: int(studio_win.evaluate_js("document.querySelectorAll('#prov-models-grid .prov-model-chip').length") or 0) == 8,
                timeout=5.0,
                err_msg="点击 [获取模型] 后未加载出 8 个网关模型胶囊",
            )
            _click_sel(studio_win, '.prov-model-chip[data-model="broken-canary"] .prov-model-toggle')
            time.sleep(0.1)
            _click_sel(studio_win, "#btn-prov-test-models")
            _wait_until(
                lambda: studio_win.evaluate_js(
                    "document.querySelectorAll('#prov-models-grid .latency-pill.ok').length >= 1"
                    " && document.querySelectorAll('#prov-models-grid .latency-pill.err').length >= 1"
                ),
                timeout=8.0,
                err_msg="点击 [逐个检测已选模型] 后未渲染成功与失败延迟徽章",
            )
            time.sleep(0.2)
            modal_diag = studio_win.evaluate_js(
                "(() => {"
                "  const sheet = document.querySelector('.provider-modal-sheet');"
                "  const body = document.querySelector('.provider-modal-body');"
                "  const r = sheet.getBoundingClientRect();"
                "  const clipBtns = [];"
                "  sheet.querySelectorAll('button:not(.hidden)').forEach(b => {"
                "    if (b.classList.contains('prov-model-toggle') || b.classList.contains('prov-star-btn')) return;"
                "    const br = b.getBoundingClientRect();"
                "    if (br.width > 0 && (b.scrollWidth > b.clientWidth + 2 || br.right > r.right + 1)) clipBtns.push(b.id || b.textContent);"
                "  });"
                "  return {w: r.width, h: r.height, iw: window.innerWidth, ih: window.innerHeight, ovX: sheet.scrollWidth - sheet.clientWidth, bodyOvX: body.scrollWidth - body.clientWidth, keyVal: document.getElementById('inp-agent-api-key').value, clipBtns, text: sheet.innerText};"
                "})()"
            )
            assert modal_diag["w"] <= modal_diag["iw"] and modal_diag["h"] <= modal_diag["ih"], modal_diag
            assert modal_diag["ovX"] <= 1 and modal_diag["bodyOvX"] <= 1, modal_diag
            assert modal_diag["clipBtns"] == [], f"供应商弹窗按钮截断: {modal_diag['clipBtns']}"
            assert modal_diag["keyVal"] == "", modal_diag
            assert not _EMOJI_RE.findall(modal_diag["text"]), "供应商弹窗包含禁止的 Emoji 字符"
            _save_cdp_screenshot(studio_win, "04_provider_modal_magpie.png")

            _click_sel(studio_win, "#btn-prov-save")
            _wait_until(
                lambda: studio_win.evaluate_js("document.getElementById('provider-modal').classList.contains('hidden')"),
                timeout=4.0,
                err_msg="保存供应商后未自动关闭弹窗",
            )
            time.sleep(0.2)
            _click_sel(studio_win, "#copilot-brain-pill")
            _wait_until(
                lambda: int(studio_win.evaluate_js("document.querySelectorAll('#cfg-quick-model-list .quick-model-btn').length") or 0) >= 6,
                timeout=4.0,
                err_msg="快切浮层未展示刚刚导入的多模型列表",
            )
            _click_sel(studio_win, '#cfg-quick-model-list .quick-model-btn[data-model="claude-sonnet-4-5"]')
            _wait_until(
                lambda: studio_win.evaluate_js(
                    "document.getElementById('copilot-config-popover').classList.contains('hidden')"
                    " && (document.getElementById('copilot-brain-label').textContent || '').includes('claude-sonnet-4-5')"
                ),
                timeout=4.0,
                err_msg="快切点击后顶栏大脑标签未更新为 claude-sonnet-4-5",
            )
            time.sleep(0.1)

            # 7.6) 切换为 Magpie 下发的 ai-studio/gemini-4-hydrogen，验证思考强度自动对齐 4 档 (无 Off) 并截图 09_hydrogen_effort_magpie_sync.png
            _click_sel(studio_win, "#copilot-brain-pill")
            _wait_until(
                lambda: studio_win.evaluate_js(
                    "!document.getElementById('copilot-config-popover').classList.contains('hidden')"
                    " && Boolean(document.querySelector('#cfg-quick-model-list .quick-model-btn[data-model=\"ai-studio/gemini-4-hydrogen\"]'))"
                ),
                timeout=3.0,
                err_msg="重新打开快切浮层失败",
            )
            time.sleep(0.1)
            _click_sel(studio_win, '#cfg-quick-model-list .quick-model-btn[data-model="ai-studio/gemini-4-hydrogen"]')
            _wait_until(
                lambda: "gemini-4-hydrogen" in str(studio_win.evaluate_js("document.getElementById('copilot-brain-label').textContent") or ""),
                timeout=4.0,
                err_msg="切换至 gemini-4-hydrogen 失败",
            )
            _click_sel(studio_win, "#btn-copilot-effort")
            _wait_until(
                lambda: not studio_win.evaluate_js("document.getElementById('copilot-effort-popover').classList.contains('hidden')"),
                timeout=3.0,
                err_msg="切换 Hydrogen 后展开思考强度浮层失败",
            )
            time.sleep(0.2)
            h_eff_diag = studio_win.evaluate_js(
                "(() => {"
                "  const opts = [...document.querySelectorAll('#effort-options-list .effort-option')].map(b => b.dataset.effort);"
                "  const pop = document.getElementById('copilot-effort-popover');"
                "  return {opts, text: pop ? pop.innerText : ''};"
                "})()"
            )
            assert h_eff_diag["opts"] == ["auto", "low", "medium", "high", "xhigh"], h_eff_diag
            assert "超高 (Extra)" in h_eff_diag["text"] and "推荐: 高" in h_eff_diag["text"], h_eff_diag
            _audit_studio_copilot(studio_win, "hydrogen-effort-sync", expect_copilot_open=True)
            _save_cdp_screenshot(studio_win, "09_hydrogen_effort_magpie_sync.png")
            _click_sel(studio_win, "#btn-copilot-effort")
            _wait_until(
                lambda: studio_win.evaluate_js("document.getElementById('copilot-effort-popover').classList.contains('hidden')"),
                timeout=3.0,
                err_msg="Hydrogen 验收后未收起思考强度浮层",
            )
            time.sleep(0.15)

            # 7.7) 长对话防 Flexbox 压缩遮挡 & 滚轮触底回滚浮标验证 -> 轻量多会话新建、历史浮层验收与截图 10_copilot_sessions_and_scroll.png
            studio_win.evaluate_js(
                "(() => {"
                "  const box = document.getElementById('copilot-messages');"
                "  const padItem = document.createElement('div');"
                "  padItem.className = 'copilot-msg assistant';"
                "  padItem.id = 'test-long-scroll-probe';"
                "  const rich = document.createElement('div');"
                "  rich.className = 'copilot-rich';"
                "  for (let i = 1; i <= 18; i++) {"
                "    const p = document.createElement('p');"
                "    p.className = 'copilot-p';"
                "    p.textContent = `长文本校验段落 #${i}：检查 G4/H100 实例显存水位与队列积压情况，确保末尾行不被底部输入舱遮挡。`;"
                "    rich.appendChild(p);"
                "  }"
                "  padItem.appendChild(rich);"
                "  box.appendChild(padItem);"
                "  box.scrollTop = box.scrollHeight;"
                "})()"
            )
            time.sleep(0.2)
            scroll_diag = studio_win.evaluate_js(
                "(() => {"
                "  const box = document.getElementById('copilot-messages');"
                "  const composer = document.querySelector('.copilot-composer');"
                "  const probe = document.getElementById('test-long-scroll-probe');"
                "  const msgs = [...box.querySelectorAll('.copilot-msg')];"
                "  const squashed = msgs.filter(m => m.getBoundingClientRect().height + 2 < m.scrollHeight).map(m => ({h: m.getBoundingClientRect().height, sh: m.scrollHeight}));"
                "  const pr = probe.getBoundingClientRect();"
                "  const cr = composer.getBoundingClientRect();"
                "  return {scrollH: box.scrollHeight, clientH: box.clientHeight, scrollTop: box.scrollTop, maxScrollTop: box.scrollHeight - box.clientHeight, squashed, probeBottom: pr.bottom, composerTop: cr.top};"
                "})()"
            )
            assert scroll_diag["scrollH"] > scroll_diag["clientH"] + 150, f"长对话未撑开纵向滚动区: {scroll_diag}"
            assert scroll_diag["squashed"] == [], f"存在被 Flexbox 压缩导致内容溢出的消息节点: {scroll_diag}"
            assert abs(scroll_diag["scrollTop"] - scroll_diag["maxScrollTop"]) <= 4, f"无法滚动到对话最底部: {scroll_diag}"
            assert scroll_diag["probeBottom"] <= scroll_diag["composerTop"] + 2, f"滚到底部后末尾正文仍被输入舱遮挡: {scroll_diag}"

            # 向上滚动离开底部 -> 验证「最新回复」悬浮钮自动显现 -> 点击一键回到底部并隐藏
            studio_win.evaluate_js(
                "(() => { const b = document.getElementById('copilot-messages'); b.scrollTop = 0; b.dispatchEvent(new Event('scroll')); })()"
            )
            _wait_until(
                lambda: not studio_win.evaluate_js("document.getElementById('btn-copilot-scroll-bottom').classList.contains('hidden')"),
                timeout=3.0,
                err_msg="上滚浏览历史时未弹出「最新回复」回到底部按钮",
            )
            _audit_studio_copilot(studio_win, "scroll-bottom-btn-visible", expect_copilot_open=True)
            _click_sel(studio_win, "#btn-copilot-scroll-bottom")
            _wait_until(
                lambda: studio_win.evaluate_js(
                    "(() => { const b = document.getElementById('copilot-messages'); const btn = document.getElementById('btn-copilot-scroll-bottom'); return btn.classList.contains('hidden') && (b.scrollHeight - b.scrollTop - b.clientHeight) <= 8; })()"
                ),
                timeout=3.0,
                err_msg="点击「最新回复」后未自动滚到底部并隐藏",
            )

            # 记录第一会话 ID -> 点击顶栏新建对话 -> 发一条消息 -> 展开历史会话管理浮层截图并切回原会话
            sess1_id = backend.get_agent_state()["active_session_id"]
            _click_sel(studio_win, "#btn-copilot-new-chat")
            _wait_until(
                lambda: studio_win.evaluate_js(
                    "document.querySelectorAll('#copilot-messages .copilot-msg').length === 0 && Boolean(document.querySelector('.copilot-empty'))"
                ),
                timeout=4.0,
                err_msg="点击新建对话后未切入空白新会话",
            )
            studio_win.evaluate_js(
                "(() => { const inp = document.getElementById('copilot-input'); inp.value = '第二会话测试：检查日志与显存状态'; inp.dispatchEvent(new Event('input')); })()"
            )
            _click_sel(studio_win, "#btn-copilot-send")
            _wait_until(
                lambda: studio_win.evaluate_js(
                    "document.querySelectorAll('.copilot-msg.assistant').length >= 1 && !document.getElementById('btn-copilot-send').classList.contains('is-stop')"
                ),
                timeout=10.0,
                err_msg="第二会话消息未完成响应",
            )
            _click_sel(studio_win, "#btn-copilot-history")
            _wait_until(
                lambda: not studio_win.evaluate_js("document.getElementById('copilot-history-popover').classList.contains('hidden')")
                and int(studio_win.evaluate_js("document.querySelectorAll('#copilot-history-list .hist-session-item').length") or 0) == 2,
                timeout=4.0,
                err_msg="历史会话浮层未列出 2 条会话",
            )
            time.sleep(0.2)
            _audit_studio_copilot(studio_win, "history-popover-sync", expect_copilot_open=True)
            _save_cdp_screenshot(studio_win, "10_copilot_sessions_and_scroll.png")

            # 点击切回第一条历史会话并核验原工具卡与对话完整恢复
            _click_sel(studio_win, f'#copilot-history-list .hist-session-main[data-session-id="{sess1_id}"]')
            _wait_until(
                lambda: studio_win.evaluate_js(
                    "document.getElementById('copilot-history-popover').classList.contains('hidden')"
                    " && document.querySelectorAll('#copilot-messages .copilot-tool-card').length >= 1"
                    " && (document.getElementById('copilot-messages').innerText || '').includes('128.40 CU')"
                ),
                timeout=5.0,
                err_msg="切回第一条历史会话后未完整恢复原对话与工具卡",
            )
            time.sleep(0.2)

            # 8) 默认 1080×700 与 920×600 下校验 Ultra+Pro 权益文案与号池防错位排版 -> 截图 05_pool_pro_ultra_layout.png 与 03_copilot_narrow_920.png
            _click_sel(studio_win, "#nav-overview")
            time.sleep(0.35)

            def _audit_pool_rows_no_overlap(tag: str) -> None:
                p_diag = studio_win.evaluate_js(
                    "(() => {"
                    "  const rows = [...document.querySelectorAll('#ov-account-list .inset-row')];"
                    "  const info = rows.map(r => {"
                    "    const left = r.querySelector('.acct-row-left').getBoundingClientRect();"
                    "    const right = r.querySelector('.acct-row-right').getBoundingClientRect();"
                    "    const name = r.querySelector('.acct-row-name').getBoundingClientRect();"
                    "    const btn = r.querySelector('.acct-row-edit-btn').getBoundingClientRect();"
                    "    return {"
                    "      txt: r.innerText,"
                    "      nameW: name.width,"
                    "      btnX: btn.left,"
                    "      rightLeft: right.left,"
                    "      leftBottom: left.bottom,"
                    "      rightTop: right.top,"
                    "      leftRight: left.right,"
                    "      overlap: (left.right > right.left + 2) && (left.bottom > right.top + 2) && (right.bottom > left.top + 2),"
                    "      btnOverlap: (btn.right > right.left + 2) && (btn.bottom > right.top + 2) && (right.bottom > btn.top + 2)"
                    "    };"
                    "  });"
                    "  return {"
                    "    poolSub: document.getElementById('ov-pool-sub').textContent,"
                    "    activeAcct: document.getElementById('ov-active-acct').textContent,"
                    "    info"
                    "  };"
                    "})()"
                )
                assert "1 Ultra + 1 Pro" in p_diag["poolSub"], p_diag
                assert "Ultra主力" in p_diag["activeAcct"], p_diag
                assert len(p_diag["info"]) == 2, p_diag
                assert "Ultra" in p_diag["info"][0]["txt"] and "H100 / G4" in p_diag["info"][0]["txt"], p_diag
                assert "Pro" in p_diag["info"][1]["txt"] and "G4 / A100" in p_diag["info"][1]["txt"], p_diag
                for row_i in p_diag["info"]:
                    assert not row_i["overlap"], f"[{tag}] 号池左右区域发生重叠: {row_i}"
                    assert not row_i["btnOverlap"], f"[{tag}] 编辑铅笔按钮与右侧徽章重叠: {row_i}"
                    assert row_i["nameW"] >= 50, f"[{tag}] 账号名称列被过度压缩: {row_i}"

            _audit_pool_rows_no_overlap("default-1080")
            studio_win.evaluate_js(
                "document.querySelector('#view-overview .view-scroll').scrollTop = document.querySelector('#view-overview .view-scroll').scrollHeight"
            )
            time.sleep(0.25)
            _save_cdp_screenshot(studio_win, "05_pool_pro_ultra_layout.png")

            studio_win.resize(920, 600)
            time.sleep(0.45)
            for tab_name in ("overview", "gallery", "tuner", "console"):
                _click_sel(studio_win, f"#nav-{tab_name}")
                time.sleep(0.35)
                _audit_studio_copilot(studio_win, f"narrow-920-{tab_name}", expect_copilot_open=True)
            _click_sel(studio_win, "#nav-overview")
            time.sleep(0.4)
            _audit_pool_rows_no_overlap("narrow-920")
            _audit_studio_copilot(studio_win, "narrow-920-final", expect_copilot_open=True)
            _save_cdp_screenshot(studio_win, "03_copilot_narrow_920.png")

            # 关闭 Studio 验证桌宠保活
            close_res = api.studio_window_ctl("close")
            assert close_res["ok"] is True and close_res["closed"] is True
            assert stop_event.is_set() is False
            assert ui_secret not in json.dumps(backend.get_agent_state(), ensure_ascii=False)
        except BaseException as exc:
            errs.append(exc)
        finally:
            stop_event.set()
            try:
                api._close_studio_window_internal()
            except Exception:
                pass
            try:
                pet_win.destroy()
            except Exception:
                pass
            try:
                httpd.shutdown()
                httpd.server_close()
            except Exception:
                pass
            close_sse()
            close_magpie()

    webview.start(func=_worker)
    if errs:
        raise errs[0]
    print("PYWEBVIEW_COPILOT_ALL_OK", flush=True)
    return 0


@pytest.mark.gui
def test_pywebview_copilot_dual_window_and_visual_audit(isolated_env: SimpleNamespace) -> None:
    """真拉起 392×548 桌宠 + 1080×700 / 920×600 Studio 双窗，全链路验收副驾交互、零溢出零 Emoji 与 3 张高清截图。"""
    work_dir = isolated_env.tmp / "copilot_webview_box"
    work_dir.mkdir(parents=True, exist_ok=True)
    env_shot = str(os.environ.get("COLAB_PET_COPILOT_SCREENSHOT_DIR") or "").strip()
    shot_dir = Path(env_shot) if env_shot else (isolated_env.tmp / "screenshots")
    shot_dir.mkdir(parents=True, exist_ok=True)

    proc = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--copilot-webview-child",
            str(work_dir),
            "--shot-dir",
            str(shot_dir),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=90,
        env=dict(os.environ),
    )
    assert proc.returncode == 0 and "PYWEBVIEW_COPILOT_ALL_OK" in proc.stdout, (
        f"PyWebView 副驾双窗子进程验收失败 (rc={proc.returncode}):\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )

    for shot_name, min_w, min_h in (
        ("01_copilot_light_1080.png", 1070, 690),
        ("02_copilot_confirm_card_dark.png", 1070, 690),
        ("03_copilot_narrow_920.png", 910, 590),
        ("04_provider_modal_magpie.png", 1070, 690),
        ("05_pool_pro_ultra_layout.png", 1070, 690),
        ("06_pet_pool_total_cu.png", 380, 530),
        ("07_antigravity_copilot_light.png", 1070, 690),
        ("08_antigravity_copilot_dark.png", 1070, 690),
        ("09_hydrogen_effort_magpie_sync.png", 1070, 690),
        ("10_copilot_sessions_and_scroll.png", 1070, 690),
    ):
        p = shot_dir / shot_name
        assert p.is_file() and p.stat().st_size > 10_000, f"验收截图缺失或异常过小: {p}"
        with Image.open(p) as im:
            w, h = im.size
            assert w >= min_w and h >= min_h, f"截图尺寸异常 {shot_name}: {im.size}"


# =========================================================================
# 5. Windows 便携打包安全审计
# =========================================================================


def test_portable_package_includes_agent_modules_and_passes_gate(isolated_env: SimpleNamespace) -> None:
    """验证 build_portable_package 产出的发行 zip 包含 agent_tools.py 与 backend_agent.py 且通过安全闸门。"""
    out_dir = isolated_env.tmp / "dist_agent"
    bundle_dir, zip_path = win_pkg.build_portable_package(PKG_DIR, out_dir)
    assert bundle_dir.is_dir() and zip_path.is_file()

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = set(zf.namelist())
        for rel in (
            "Colab桌宠/colab_core/agent_tools.py",
            "Colab桌宠/colab_pet/backend_agent.py",
            "Colab桌宠/colab_ctl.py",
            "Colab桌宠/colab_security.py",
            "Colab桌宠/colab_pet/studio.html",
            "Colab桌宠/colab_pet/studio.css",
            "Colab桌宠/colab_pet/studio.js",
            "Colab桌宠/colab_pet/ui.html",
            "Colab桌宠/Colab桌宠.exe",
        ):
            assert rel in names, f"便携发行包缺少核心文件: {rel}"

        for forbidden in win_pkg.FORBIDDEN_FILENAMES:
            assert not any(n.endswith(f"/{forbidden}") for n in names), f"发行包夹带禁止文件: {forbidden}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--copilot-webview-child", default="")
    ap.add_argument("--shot-dir", default="")
    ns, _ = ap.parse_known_args()
    if ns.copilot_webview_child:
        w_dir = Path(ns.copilot_webview_child)
        s_dir = Path(ns.shot_dir) if ns.shot_dir else (w_dir / "screenshots")
        rc = int(_run_pywebview_copilot_child(w_dir, s_dir))
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(rc)
    raise SystemExit(pytest.main([__file__, "-v"]))
