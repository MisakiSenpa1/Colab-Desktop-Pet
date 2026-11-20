"""Phase 2 路 3：`colab_sse_gateway.py` 拆解与去重专项离线回归测试套件。

覆盖范围：
1. `_augment_models_catalog` 对 OpenAI `data` 与 llama.cpp/Ollama `models` 双格式注入及幂等去重；
2. `_normalize_inference_payload` 与 `_format_sse_error_frame` 多协议思考参数与错误帧边界；
3. 真实本地 TCP 端到端验证：OPTIONS 预检、401/503 鉴权拦截、`/__pet_metrics`、`/v1/models`、
   非流式透传、上游不可达 502、SSE Prefill 心跳保活及 Anthropic/OpenAI 终端错误帧封装。

说明：仅绑定 `127.0.0.1:0` 随机端口与隔离临时沙箱，绝不触碰宿主机真实配置与凭据库。
"""

from __future__ import annotations

import asyncio
import json
import socket
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pytest
from aiohttp import web

PKG_DIR = Path(__file__).resolve().parents[1]
if str(PKG_DIR) not in sys.path:
    sys.path.insert(0, str(PKG_DIR))

import colab_sse_gateway as sse_gw  # noqa: E402


def _free_port() -> int:
    """分配一个本机回环可用 TCP 端口。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _http_request(
    url: str,
    *,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    body: bytes | None = None,
    timeout: float = 5.0,
) -> tuple[int, dict[str, str], bytes]:
    """发起同步 HTTP 请求并返回 `(status, lower_headers, body_bytes)`（兼容 4xx/5xx）。"""
    req = urllib.request.Request(url, data=body, headers=dict(headers or {}), method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, {k.lower(): v for k, v in resp.headers.items()}, resp.read()
    except urllib.error.HTTPError as err:
        raw = err.read()
        return err.code, {k.lower(): v for k, v in err.headers.items()}, raw


# =========================================================================
# 1. 模型目录双格式去重与思考参数/错误帧纯函数测试
# =========================================================================


def test_augment_models_catalog_dual_formats_and_dedup(monkeypatch: pytest.MonkeyPatch) -> None:
    """校验 `_augment_models_catalog` 支持 data/models 双格式、别名重写及重复调用幂等去重。"""
    monkeypatch.setattr(sse_gw, "BASE_MODEL", "Qwen3.8-27B-abliterated")

    # 1) 同时包含 data 与 models 且各自已预置一个虚拟档位（验证精确去重）
    catalog: dict[str, Any] = {
        "object": "list",
        "data": [
            {"id": "other-model", "object": "model"},
            {
                "id": "Qwen3.8-27B-abliterated",
                "object": "model",
                "aliases": ["Qwen3.8-27B-abliterated"],
            },
            {
                "id": "Qwen3.8-27B-abliterated-low",
                "object": "model",
                "custom_keep": True,
            },
        ],
        "models": [
            {"name": "Qwen3.8-27B-abliterated", "model": "Qwen3.8-27B-abliterated"},
            {"name": "Qwen3.8-27B-abliterated-nothink", "model": "Qwen3.8-27B-abliterated-nothink", "kept": 1},
        ],
    }

    out = sse_gw._augment_models_catalog(catalog)
    data_ids = [item["id"] for item in out["data"]]
    assert data_ids == [
        "other-model",
        "Qwen3.8-27B-abliterated",
        "Qwen3.8-27B-abliterated-low",
        "Qwen3.8-27B-abliterated-nothink",
        "Qwen3.8-27B-abliterated-medium",
        "Qwen3.8-27B-abliterated-xhigh",
    ]
    base_data = out["data"][1]
    assert base_data["reasoning_levels"] == ["none", "low", "medium", "high", "xhigh"]
    assert base_data["supported_parameters"] == ["reasoning_effort", "thinking", "enable_thinking", "tools"]
    # 预存的 -low 保持原样未被覆写；新增克隆的 aliases 与 reasoning_levels 精准对齐
    assert out["data"][2].get("custom_keep") is True
    nothink_item = out["data"][3]
    assert nothink_item["aliases"] == ["Qwen3.8-27B-abliterated-nothink"]
    assert nothink_item["reasoning_levels"] == ["none"]
    assert nothink_item["supported_parameters"] == ["reasoning_effort", "thinking", "enable_thinking", "tools"]

    model_names = [item["model"] for item in out["models"]]
    assert model_names == [
        "Qwen3.8-27B-abliterated",
        "Qwen3.8-27B-abliterated-nothink",
        "Qwen3.8-27B-abliterated-low",
        "Qwen3.8-27B-abliterated-medium",
        "Qwen3.8-27B-abliterated-xhigh",
    ]
    assert out["models"][1].get("kept") == 1
    assert out["models"][2]["name"] == "Qwen3.8-27B-abliterated-low"
    assert out["models"][2]["reasoning_levels"] == ["low"]
    assert "supported_parameters" not in out["models"][2]

    # 2) 二次调用必须完全幂等（零重复条目）
    count_data_before = len(out["data"])
    count_models_before = len(out["models"])
    sse_gw._augment_models_catalog(out)
    assert len(out["data"]) == count_data_before
    assert len(out["models"]) == count_models_before

    # 3) 未命中 BASE_MODEL 时回退到首条字典项；非字典输入原样透传
    fallback_obj = {"data": [{"id": "custom-gguf"}]}
    sse_gw._augment_models_catalog(fallback_obj)
    assert [x["id"] for x in fallback_obj["data"]] == [
        "custom-gguf",
        "custom-gguf-nothink",
        "custom-gguf-low",
        "custom-gguf-medium",
        "custom-gguf-xhigh",
    ]
    assert sse_gw._augment_models_catalog("not-a-dict") == "not-a-dict"  # type: ignore[arg-type]


def test_normalize_inference_payload_and_error_frame_formatting(monkeypatch: pytest.MonkeyPatch) -> None:
    """校验虚拟模型后缀、多协议思考参数规范化与 Anthropic/OpenAI SSE 终端错误帧格式。"""
    monkeypatch.setattr(sse_gw, "BASE_MODEL", "Qwen3.8-27B-abliterated")
    monkeypatch.setattr(sse_gw, "DEFAULT_EFFORT", "medium")

    # 1) 虚拟模型后缀 -xhigh 重写并清理顶层非标准键
    p1 = sse_gw._normalize_inference_payload(
        {"model": "Qwen3.8-27B-abliterated-xhigh", "reasoning_effort": "low", "enable_thinking": True},
        "/v1/chat/completions",
    )
    assert p1["model"] == "Qwen3.8-27B-abliterated"
    assert "reasoning_effort" not in p1 and "enable_thinking" not in p1
    assert p1["chat_template_kwargs"] == {"enable_thinking": True, "reasoning_effort": "low"}

    # 2) Anthropic /v1/messages 显式禁用思考时保留 thinking.type == disabled
    p2 = sse_gw._normalize_inference_payload(
        {"model": "Qwen3.8-27B-abliterated-nothink", "thinking": {"type": "disabled"}},
        "/v1/messages",
    )
    assert p2["model"] == "Qwen3.8-27B-abliterated"
    assert p2["chat_template_kwargs"] == {"enable_thinking": False}
    assert p2["thinking"] == {"type": "disabled"}

    # 3) Anthropic budget_tokens 分档映射
    p3 = sse_gw._normalize_inference_payload(
        {"model": "Qwen3.8-27B-abliterated", "thinking": {"type": "enabled", "budget_tokens": 8192}},
        "/v1/messages",
    )
    assert p3["chat_template_kwargs"] == {"enable_thinking": True, "reasoning_effort": "xhigh"}

    # 4) Anthropic vs OpenAI SSE 终端错误帧协议边界
    anth_ctx = sse_gw._format_sse_error_frame(
        "/v1/messages", 500, json.dumps({"error": {"message": "Context size exceeded", "type": "server_error"}})
    ).decode("utf-8")
    assert anth_ctx.startswith("event: error\ndata: ")
    anth_ctx_obj = json.loads(anth_ctx.split("data: ", 1)[1].strip())
    assert anth_ctx_obj == {
        "type": "error",
        "error": {"type": "invalid_request_error", "message": "Context size exceeded"},
    }

    anth_401 = sse_gw._format_sse_error_frame("/messages", 401, "Bad token").decode("utf-8")
    assert json.loads(anth_401.split("data: ", 1)[1].strip())["error"]["type"] == "authentication_error"

    anth_429 = sse_gw._format_sse_error_frame("/v1/messages", 429, "Too many requests").decode("utf-8")
    assert json.loads(anth_429.split("data: ", 1)[1].strip())["error"]["type"] == "rate_limit_error"

    oai_err = sse_gw._format_sse_error_frame(
        "/v1/chat/completions", 502, json.dumps({"error": {"type": "upstream_err", "message": "Upstream closed"}})
    ).decode("utf-8")
    assert "data: [DONE]\n\n" in oai_err
    first_data = oai_err.splitlines()[0].split("data: ", 1)[1]
    assert json.loads(first_data) == {
        "error": {"status": 502, "type": "upstream_err", "message": "Upstream closed"}
    }


# =========================================================================
# 2. OPTIONS 预检与 401/503 鉴权拦截端到端测试
# =========================================================================


def test_gateway_options_preflight_and_auth_401_503_guards(monkeypatch: pytest.MonkeyPatch) -> None:
    """校验 OPTIONS 免密预检、SSE_REQUIRE_AUTH=1 无密钥 503、错误密钥 401、Bearer/x-api-key 放行与 /health 免密。"""

    async def _run() -> None:
        up_port = _free_port()
        gw_port = _free_port()

        async def _health(_r: web.Request) -> web.Response:
            return web.json_response({"status": "ok"})

        async def _models(_r: web.Request) -> web.Response:
            return web.json_response({"data": [{"id": "Qwen3.8-27B-abliterated"}]})

        up_app = web.Application()
        up_app.router.add_get("/health", _health)
        up_app.router.add_get("/v1/models", _models)
        runner = web.AppRunner(up_app)
        await runner.setup()
        await web.TCPSite(runner, "127.0.0.1", up_port).start()

        monkeypatch.setattr(sse_gw, "LISTEN_PORT", gw_port)
        monkeypatch.setattr(sse_gw, "UPSTREAM_PORT", up_port)
        monkeypatch.setattr(sse_gw, "BASE_MODEL", "Qwen3.8-27B-abliterated")
        monkeypatch.delenv("SSE_ALLOWED_ORIGINS", raising=False)
        monkeypatch.delenv("SSE_CORS_ALLOW_CREDENTIALS", raising=False)
        monkeypatch.delenv("SSE_API_KEY", raising=False)
        monkeypatch.delenv("SSE_API_KEY_FILE", raising=False)
        monkeypatch.setenv("SSE_REQUIRE_AUTH", "1")

        gw_srv = await asyncio.start_server(sse_gw._handle_client, "127.0.0.1", gw_port)
        base = f"http://127.0.0.1:{gw_port}"
        try:
            # 1) OPTIONS 预检免密返回 204，默认返回通配 * 且不附带 Credentials
            st_opt, h_opt, _ = await asyncio.to_thread(
                _http_request, f"{base}/v1/chat/completions", method="OPTIONS", headers={"Origin": "https://app.example.com"}
            )
            assert st_opt == 204
            assert h_opt.get("access-control-allow-origin") == "*"
            assert "access-control-allow-credentials" not in h_opt

            # 启用白名单时精确回显 Origin + Vary: Origin
            monkeypatch.setenv("SSE_ALLOWED_ORIGINS", "https://app.example.com")
            st_opt2, h_opt2, _ = await asyncio.to_thread(
                _http_request, f"{base}/v1/chat/completions", method="OPTIONS", headers={"Origin": "https://app.example.com"}
            )
            assert st_opt2 == 204
            assert h_opt2.get("access-control-allow-origin") == "https://app.example.com"
            assert h_opt2.get("vary") == "Origin"
            monkeypatch.delenv("SSE_ALLOWED_ORIGINS", raising=False)

            # 2) SSE_REQUIRE_AUTH=1 但未配置密钥 -> 受保护路径全部 Fail-Closed 返回 503
            for protected_path in ("/v1/models", "/__pet_metrics"):
                st_503, _, raw_503 = await asyncio.to_thread(_http_request, f"{base}{protected_path}")
                assert st_503 == 503
                assert "not configured" in raw_503.decode("utf-8").lower()

            # /health 免密放行
            st_h, _, raw_h = await asyncio.to_thread(_http_request, f"{base}/health")
            assert st_h == 200 and json.loads(raw_h.decode("utf-8"))["status"] == "ok"

            # 3) 配置密钥后：匿名与错误密钥返回 401 + WWW-Authenticate: Bearer
            secret = "sk-phase2-sse-gateway-test-key-01"
            monkeypatch.setenv("SSE_API_KEY", secret)

            st_no_key, h_no_key, _ = await asyncio.to_thread(_http_request, f"{base}/v1/models")
            assert st_no_key == 401 and h_no_key.get("www-authenticate") == "Bearer"

            st_bad_key, _, _ = await asyncio.to_thread(
                _http_request, f"{base}/__pet_metrics", headers={"Authorization": "Bearer wrong-key"}
            )
            assert st_bad_key == 401

            # 4) 有效 Bearer 与 x-api-key 均通过鉴权
            st_bearer, _, raw_m = await asyncio.to_thread(
                _http_request, f"{base}/v1/models", headers={"Authorization": f"Bearer {secret}"}
            )
            assert st_bearer == 200
            assert any(m["id"].endswith("-xhigh") for m in json.loads(raw_m.decode("utf-8"))["data"])

            st_xkey, _, _ = await asyncio.to_thread(
                _http_request, f"{base}/v1/models", headers={"x-api-key": secret}
            )
            assert st_xkey == 200
        finally:
            gw_srv.close()
            await gw_srv.wait_closed()
            await runner.cleanup()

    asyncio.run(_run())


# =========================================================================
# 3. /__pet_metrics、/v1/models 与非流式透传端到端测试
# =========================================================================


def test_gateway_metrics_models_and_non_stream_passthrough(monkeypatch: pytest.MonkeyPatch) -> None:
    """校验 `/__pet_metrics`、`/v1/models` 增强与错误透传、非流式推理改写透传及上游离线 502。"""

    async def _run() -> None:
        up_port = _free_port()
        gw_port = _free_port()
        received_chat_bodies: list[dict[str, Any]] = []
        models_fail_mode = {"fail": False}

        async def _up_models(_r: web.Request) -> web.Response:
            if models_fail_mode["fail"]:
                return web.json_response({"error": {"message": "models offline"}}, status=503)
            return web.json_response({"data": [{"id": "Qwen3.8-27B-abliterated"}]})

        async def _up_chat(request: web.Request) -> web.Response:
            payload = await request.json()
            received_chat_bodies.append(payload)
            return web.json_response(
                {
                    "id": "chatcmpl-nonstream",
                    "choices": [{"message": {"role": "assistant", "content": "NONSTREAM_OK"}}],
                }
            )

        up_app = web.Application()
        up_app.router.add_get("/v1/models", _up_models)
        up_app.router.add_post("/v1/chat/completions", _up_chat)
        runner = web.AppRunner(up_app)
        await runner.setup()
        await web.TCPSite(runner, "127.0.0.1", up_port).start()

        monkeypatch.setattr(sse_gw, "LISTEN_PORT", gw_port)
        monkeypatch.setattr(sse_gw, "UPSTREAM_PORT", up_port)
        monkeypatch.setattr(sse_gw, "BASE_MODEL", "Qwen3.8-27B-abliterated")
        monkeypatch.delenv("SSE_API_KEY", raising=False)
        monkeypatch.delenv("SSE_API_KEY_FILE", raising=False)
        monkeypatch.delenv("SSE_REQUIRE_AUTH", raising=False)
        monkeypatch.delenv("SSE_ALLOWED_ORIGINS", raising=False)

        fake_metrics = {
            "ok": True,
            "gpu_full_name": "NVIDIA L4",
            "vram_used_gb": 14.2,
            "vram_total_gb": 24.0,
            "vram_pct": 59.2,
            "ram_used_gb": 12.0,
            "ram_total_gb": 53.0,
            "ram_free_gb": 41.0,
            "ram_pct": 22.6,
            "disk_used_gb": 40.0,
            "disk_total_gb": 200.0,
            "disk_free_gb": 160.0,
            "disk_pct": 20.0,
            "shm_used_gb": 0.1,
            "shm_total_gb": 16.0,
            "shm_free_gb": 15.9,
            "drive_persisted": True,
            "drive_free_gb": 80.0,
            "drive_total_gb": 100.0,
        }
        monkeypatch.setattr(sse_gw, "_collect_gateway_host_metrics", lambda: dict(fake_metrics))

        gw_srv = await asyncio.start_server(sse_gw._handle_client, "127.0.0.1", gw_port)
        base = f"http://127.0.0.1:{gw_port}"
        try:
            # 1) GET /__pet_metrics 返回 200 + Cache-Control: no-store
            st_met, h_met, raw_met = await asyncio.to_thread(_http_request, f"{base}/__pet_metrics")
            assert st_met == 200
            assert h_met.get("cache-control") == "no-store"
            assert json.loads(raw_met.decode("utf-8"))["gpu_full_name"] == "NVIDIA L4"

            # 2) GET /v1/models 正常增强
            st_m, _, raw_m = await asyncio.to_thread(_http_request, f"{base}/v1/models")
            assert st_m == 200
            m_ids = [item["id"] for item in json.loads(raw_m.decode("utf-8"))["data"]]
            assert "Qwen3.8-27B-abliterated-medium" in m_ids

            # 上游 /v1/models 非 200 时原样透传错误状态码
            models_fail_mode["fail"] = True
            st_m_err, _, raw_m_err = await asyncio.to_thread(_http_request, f"{base}/v1/models")
            assert st_m_err == 503 and "models offline" in raw_m_err.decode("utf-8")
            models_fail_mode["fail"] = False

            # 3) 非流式 POST /v1/chat/completions 透传且规范化思考参数
            req_body = json.dumps(
                {
                    "model": "Qwen3.8-27B-abliterated-low",
                    "stream": False,
                    "messages": [{"role": "user", "content": "ping"}],
                }
            ).encode("utf-8")
            st_chat, _, raw_chat = await asyncio.to_thread(
                _http_request,
                f"{base}/v1/chat/completions",
                method="POST",
                headers={"Content-Type": "application/json"},
                body=req_body,
            )
            assert st_chat == 200
            assert json.loads(raw_chat.decode("utf-8"))["choices"][0]["message"]["content"] == "NONSTREAM_OK"
            assert len(received_chat_bodies) == 1
            assert received_chat_bodies[0]["model"] == "Qwen3.8-27B-abliterated"
            assert received_chat_bodies[0]["chat_template_kwargs"] == {
                "enable_thinking": True,
                "reasoning_effort": "low",
            }
        finally:
            gw_srv.close()
            await gw_srv.wait_closed()
            await runner.cleanup()

        # 4) 上游关闭后访问返回 502 Bad Gateway
        gw_srv2 = await asyncio.start_server(sse_gw._handle_client, "127.0.0.1", gw_port)
        try:
            st_502, _, raw_502 = await asyncio.to_thread(_http_request, f"{base}/v1/models")
            assert st_502 == 502 and "upstream unavailable" in raw_502.decode("utf-8").lower()
        finally:
            gw_srv2.close()
            await gw_srv2.wait_closed()

    asyncio.run(_run())


# =========================================================================
# 4. SSE 心跳保活、快响应零修改透传与 Anthropic/OpenAI 错误帧边界测试
# =========================================================================


def test_gateway_sse_heartbeat_and_error_frame_boundaries(monkeypatch: pytest.MonkeyPatch) -> None:
    """校验慢 Prefill 心跳注入、快响应直通、Anthropic/OpenAI 延迟非 200 封装及上游提前断连 502 帧。"""

    async def _run() -> None:
        up_port = _free_port()
        gw_port = _free_port()

        async def _up_openai_sse(request: web.Request) -> web.StreamResponse:
            payload = await request.json()
            prompt = str((payload.get("messages") or [{}])[-1].get("content") or "")
            if prompt == "fast":
                resp = web.StreamResponse(status=200, headers={"Content-Type": "text/event-stream"})
                await resp.prepare(request)
                await resp.write(b'data: {"choices":[{"delta":{"content":"FAST_OK"}}]}\n\ndata: [DONE]\n\n')
                await resp.write_eof()
                return resp
            if prompt == "slow_400":
                await asyncio.sleep(0.14)
                return web.json_response({"error": {"message": "Context size has been exceeded.", "type": "exceed_context"}}, status=400)
            await asyncio.sleep(0.14)
            resp = web.StreamResponse(status=200, headers={"Content-Type": "text/event-stream"})
            await resp.prepare(request)
            await resp.write(b'data: {"choices":[{"delta":{"content":"SLOW_OK"}}]}\n\ndata: [DONE]\n\n')
            await resp.write_eof()
            return resp

        async def _up_anthropic_sse(request: web.Request) -> web.StreamResponse:
            payload = await request.json()
            prompt = str((payload.get("messages") or [{}])[-1].get("content") or "")
            await asyncio.sleep(0.14)
            if prompt == "slow_429":
                return web.json_response({"error": {"message": "Rate limited by upstream"}}, status=429)
            return web.json_response({"error": {"message": "Context size limit reached"}}, status=400)

        up_app = web.Application()
        up_app.router.add_post("/v1/chat/completions", _up_openai_sse)
        up_app.router.add_post("/v1/messages", _up_anthropic_sse)
        runner = web.AppRunner(up_app)
        await runner.setup()
        await web.TCPSite(runner, "127.0.0.1", up_port).start()

        monkeypatch.setattr(sse_gw, "LISTEN_PORT", gw_port)
        monkeypatch.setattr(sse_gw, "UPSTREAM_PORT", up_port)
        monkeypatch.setattr(sse_gw, "BASE_MODEL", "Qwen3.8-27B-abliterated")
        monkeypatch.setattr(sse_gw, "HEARTBEAT_SEC", 0.05)
        monkeypatch.delenv("SSE_API_KEY", raising=False)
        monkeypatch.delenv("SSE_API_KEY_FILE", raising=False)
        monkeypatch.delenv("SSE_REQUIRE_AUTH", raising=False)

        gw_srv = await asyncio.start_server(sse_gw._handle_client, "127.0.0.1", gw_port)
        base = f"http://127.0.0.1:{gw_port}"
        try:
            # 1) 慢 Prefill 200 SSE：先推送 : keepalive，再推送正文与 [DONE]
            st_slow, _, raw_slow = await asyncio.to_thread(
                _http_request,
                f"{base}/v1/chat/completions",
                method="POST",
                headers={"Content-Type": "application/json"},
                body=json.dumps({"model": "Qwen3.8-27B-abliterated", "stream": True, "messages": [{"role": "user", "content": "slow"}]}).encode("utf-8"),
            )
            txt_slow = raw_slow.decode("utf-8")
            assert st_slow == 200
            assert ": keepalive" in txt_slow and "SLOW_OK" in txt_slow and "data: [DONE]" in txt_slow

            # 2) 快响应（首字节 < HEARTBEAT_SEC）：零心跳直通
            st_fast, _, raw_fast = await asyncio.to_thread(
                _http_request,
                f"{base}/v1/chat/completions",
                method="POST",
                headers={"Content-Type": "application/json"},
                body=json.dumps({"model": "Qwen3.8-27B-abliterated", "stream": True, "messages": [{"role": "user", "content": "fast"}]}).encode("utf-8"),
            )
            txt_fast = raw_fast.decode("utf-8")
            assert st_fast == 200
            assert ": keepalive" not in txt_fast and "FAST_OK" in txt_fast

            # 3) OpenAI 慢 400：已发 SSE 头后封装为 OpenAI 错误帧 + [DONE]
            st_oai_err, _, raw_oai_err = await asyncio.to_thread(
                _http_request,
                f"{base}/v1/chat/completions",
                method="POST",
                headers={"Content-Type": "application/json"},
                body=json.dumps({"model": "Qwen3.8-27B-abliterated", "stream": True, "messages": [{"role": "user", "content": "slow_400"}]}).encode("utf-8"),
            )
            txt_oai_err = raw_oai_err.decode("utf-8")
            assert st_oai_err == 200
            assert ": keepalive" in txt_oai_err
            assert "Context size has been exceeded." in txt_oai_err
            assert "data: [DONE]" in txt_oai_err

            # 4) Anthropic /v1/messages 慢 400 & 慢 429：封装为 Anthropic `event: error` 帧
            st_anth_400, _, raw_anth_400 = await asyncio.to_thread(
                _http_request,
                f"{base}/v1/messages",
                method="POST",
                headers={"Content-Type": "application/json"},
                body=json.dumps({"model": "Qwen3.8-27B-abliterated", "stream": True, "messages": [{"role": "user", "content": "slow_400"}]}).encode("utf-8"),
            )
            txt_anth_400 = raw_anth_400.decode("utf-8")
            assert st_anth_400 == 200
            assert ": keepalive" in txt_anth_400 and "event: error" in txt_anth_400
            assert '"invalid_request_error"' in txt_anth_400 and "Context size limit reached" in txt_anth_400

            st_anth_429, _, raw_anth_429 = await asyncio.to_thread(
                _http_request,
                f"{base}/v1/messages",
                method="POST",
                headers={"Content-Type": "application/json"},
                body=json.dumps({"model": "Qwen3.8-27B-abliterated", "stream": True, "messages": [{"role": "user", "content": "slow_429"}]}).encode("utf-8"),
            )
            txt_anth_429 = raw_anth_429.decode("utf-8")
            assert st_anth_429 == 200
            assert ": keepalive" in txt_anth_429 and "event: error" in txt_anth_429
            assert '"rate_limit_error"' in txt_anth_429
        finally:
            gw_srv.close()
            await gw_srv.wait_closed()
            await runner.cleanup()

        # 5) 慢 Prefill 期间上游不返回响应头直接断连 -> 在 SSE 流中输出 502 错误帧而非裸断
        drop_port = _free_port()

        async def _drop_upstream(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            try:
                await reader.readuntil(b"\r\n\r\n")
                await asyncio.sleep(0.12)
            finally:
                writer.close()
                await writer.wait_closed()

        drop_srv = await asyncio.start_server(_drop_upstream, "127.0.0.1", drop_port)
        monkeypatch.setattr(sse_gw, "UPSTREAM_PORT", drop_port)
        gw_srv3 = await asyncio.start_server(sse_gw._handle_client, "127.0.0.1", gw_port)
        try:
            st_drop, _, raw_drop = await asyncio.to_thread(
                _http_request,
                f"{base}/v1/chat/completions",
                method="POST",
                headers={"Content-Type": "application/json"},
                body=json.dumps({"model": "Qwen3.8-27B-abliterated", "stream": True, "messages": [{"role": "user", "content": "drop"}]}).encode("utf-8"),
            )
            txt_drop = raw_drop.decode("utf-8")
            assert st_drop == 200
            assert ": keepalive" in txt_drop
            assert "Upstream closed early" in txt_drop
            assert "data: [DONE]" in txt_drop
        finally:
            gw_srv3.close()
            await gw_srv3.wait_closed()
            drop_srv.close()
            await drop_srv.wait_closed()

    asyncio.run(_run())
