"""verify_qwen_llm.py —— Qwen3.8-Flash-Next-abliterated (llama-server) 端到端冒烟验证脚本

用法（在 Colab 实例内或通过 `python colab_ctl.py smoke --profile qwen_llm` 执行）：
    python verify_qwen_llm.py

成功标志：
    1. 读取 `/root/colab_service_meta.json` 获取服务端口与 Bearer 鉴权密钥；
    2. 校验 `GET :18080/health` 免鉴权返回 HTTP 200 且 `status == "ok"`；
    3. 校验无 Header 请求 `GET :18080/v1/models` 被鉴权拦截返回 HTTP 401/403；
    4. 校验携带 Bearer Key 流式请求 `POST :18080/v1/chat/completions` (`max_tokens=64`)，
       打印首字延迟 TTFT、生成吞吐速率 `tok/s` 与回复文本摘要，最后输出 `QWEN_LLM_SMOKE_OK`。
"""
import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

SERVICE_META_FILE = Path("/root/colab_service_meta.json")


def load_service_meta() -> dict:
    """读取并校验远端 LLM 服务元数据契约文件。"""
    if not SERVICE_META_FILE.is_file():
        raise FileNotFoundError(
            f"未找到服务契约文件 {SERVICE_META_FILE}，请先运行 colab_bootstrap_qwen_llm.py 完成部署。"
        )
    data = json.loads(SERVICE_META_FILE.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("kind") != "llm":
        raise RuntimeError(f"当前服务契约类型不是 llm: {data.get('kind')!r}")
    return data


def verify_health(base_url: str):
    """验证 /health 探针返回 HTTP 200 且 status == 'ok'。"""
    print(f"=== [1/3] 校验免鉴权健康探针 {base_url}/health ===", flush=True)
    with urllib.request.urlopen(f"{base_url}/health", timeout=15) as resp:
        status = resp.status
        body = json.loads(resp.read().decode("utf-8", errors="ignore") or "{}")
    if status != 200 or body.get("status") != "ok":
        raise RuntimeError(f"/health 返回异常: status={status}, body={body}")
    print(f"[通过] /health HTTP {status}: {body}", flush=True)


def verify_unauthorized_blocked(base_url: str):
    """验证未携带 Authorization 请求头的 /v1/models 调用被拒绝 (401/403)。"""
    print(f"=== [2/3] 校验未授权拦截保护 {base_url}/v1/models ===", flush=True)
    req = urllib.request.Request(f"{base_url}/v1/models", method="GET")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            raise RuntimeError(
                f"安全校验失败：无鉴权请求 /v1/models 未被拦截，意外返回 HTTP {resp.status}！"
            )
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            print(f"[通过] 无密钥请求已被正确拦截 (HTTP {exc.code})", flush=True)
            return
        raise RuntimeError(f"无密钥请求返回了非预期 HTTP 错误码: {exc.code}") from exc


def verify_streaming_chat(base_url: str, api_key: str, model_name: str):
    """验证带 Bearer 鉴权的流式对话生成，测算 TTFT 与解码 tok/s。"""
    print(f"=== [3/3] 校验 Bearer 流式推理 {base_url}/v1/chat/completions ===", flush=True)
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "请用简体中文简短回答。"},
            {"role": "user", "content": "用一句话介绍量子隧穿效应的核心物理直觉。"},
        ],
        "max_tokens": 64,
        "temperature": 0.6,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    t0 = time.monotonic()
    first_token_ts: float | None = None
    content_chunks: list[str] = []
    reasoning_chunks: list[str] = []
    chunk_events = 0
    completion_tokens = 0
    server_tps: float | None = None

    with urllib.request.urlopen(req, timeout=180) as resp:
        if resp.status != 200:
            raise RuntimeError(f"/v1/chat/completions 返回非 200 状态码: {resp.status}")
        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line or not line.startswith("data:"):
                continue
            data_str = line[5:].strip()
            if data_str == "[DONE]":
                break
            try:
                pkt = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            timings = pkt.get("timings")
            if isinstance(timings, dict) and timings.get("predicted_per_second"):
                server_tps = float(timings["predicted_per_second"])

            usage = pkt.get("usage")
            if isinstance(usage, dict) and usage.get("completion_tokens"):
                completion_tokens = int(usage["completion_tokens"])

            choices = pkt.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            c_piece = delta.get("content") or ""
            r_piece = delta.get("reasoning_content") or ""
            if c_piece or r_piece:
                if first_token_ts is None:
                    first_token_ts = time.monotonic()
                chunk_events += 1
                if c_piece:
                    content_chunks.append(c_piece)
                if r_piece:
                    reasoning_chunks.append(r_piece)

    t_end = time.monotonic()
    if first_token_ts is None:
        raise RuntimeError("流式响应未产出任何有效 token 内容！")

    ttft_s = first_token_ts - t0
    gen_window_s = max(0.001, t_end - first_token_ts)
    eff_tokens = completion_tokens if completion_tokens > 0 else chunk_events
    calc_tps = server_tps if server_tps and server_tps > 0 else (eff_tokens / gen_window_s)

    reply_text = "".join(content_chunks).strip()
    think_text = "".join(reasoning_chunks).strip()
    shown_reply = reply_text if reply_text else f"[思考链片段] {think_text}"

    print(
        f"[性能指标] TTFT={ttft_s:.2f}s | 吞吐={calc_tps:.1f} tok/s | 生成Token数={eff_tokens} | 总耗时={t_end - t0:.2f}s",
        flush=True,
    )
    print(f"[模型回复] {shown_reply[:240]}", flush=True)


def main():
    """执行完整 LLM 冒烟测试并输出成功标记。"""
    parser = argparse.ArgumentParser(description="Qwen3.8-Flash-Next-abliterated 端到端冒烟验证")
    parser.parse_known_args()

    meta = load_service_meta()
    port = int(meta.get("port") or 18080)
    api_key = str(meta.get("api_key") or "").strip()
    model_name = str(meta.get("model") or "Qwen3.8-Flash-Next-abliterated")
    if not api_key:
        raise RuntimeError("服务契约中缺少有效的 api_key！")

    base_url = f"http://127.0.0.1:{port}"
    print(
        f"加载服务契约成功: model={model_name} | quant={meta.get('quant')} | ctx={meta.get('ctx_size')} | port={port}",
        flush=True,
    )
    verify_health(base_url)
    verify_unauthorized_blocked(base_url)
    verify_streaming_chat(base_url, api_key, model_name)
    print("\nQWEN_LLM_SMOKE_OK", flush=True)


if __name__ == "__main__":
    main()
