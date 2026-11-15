"""verify_drama3.py —— Fish Audio Drama 3 端到端冒烟验证（参考音频克隆 + Qwen3-ASR 转写）

用法（在 Colab 实例内或通过 colab_ctl.py smoke --profile drama3 执行）：
    python verify_drama3.py
成功标志：
    1. 加载 drama3_fp8experts_bf16 + drama3_codec + qwen3_asr_1.7b_bf16；
    2. 自动选中预置参考音频，经 Qwen3-ASR 转写后合成带情绪指令的中文语音；
    3. 校验输出音轨有效且时长 > 1.0 秒。
"""
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

API = os.environ.get("COMFY", "http://127.0.0.1:8188").rstrip("/")
COMFY_DIR = Path(os.environ.get("COMFY_DIR", "/content/ComfyUI"))
COMFY_INPUT = COMFY_DIR / "input"
COMFY_OUT_ROOT = Path(os.environ.get("COMFY_OUT", "/root/comfy_out"))

DRAMA3_MODEL = "drama3_fp8experts_bf16.safetensors"
DRAMA3_CODEC = "drama3_codec.safetensors"
ASR_MODEL = "qwen3_asr_1.7b_bf16.safetensors"
PREFERRED_REF = "main_01_我操宝宝原来你长这样.wav"


def get_json(path: str) -> dict:
    with urllib.request.urlopen(f"{API}{path}", timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8") or "{}")


def post_json(path: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{API}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8") or "{}")


def pick_reference_audio() -> str:
    if (COMFY_INPUT / PREFERRED_REF).is_file():
        return PREFERRED_REF
    wavs = sorted(p.name for p in COMFY_INPUT.glob("*.wav"))
    if wavs:
        return wavs[0]
    raise FileNotFoundError(f"{COMFY_INPUT} 下未找到任何 .wav 参考音频，请先运行 sync-drama3-refs")


def build_drama3_clone_graph(ref_filename: str, text: str, seed: int = 20261115) -> dict:
    return {
        "m": {
            "class_type": "Drama3Loader",
            "inputs": {
                "model_name": DRAMA3_MODEL,
                "experts": "in RAM",
                "expert_cache_gb": 0.0,
                "second_gpu": "none",
                "split_layer": 0,
                "pin_experts": True,
            },
        },
        "c": {
            "class_type": "Drama3CodecLoader",
            "inputs": {"codec_name": DRAMA3_CODEC},
        },
        "asr_m": {
            "class_type": "Qwen3ASRLoader",
            "inputs": {"model_name": ASR_MODEL},
        },
        "ref": {
            "class_type": "Drama3LoadReferenceAudio",
            "inputs": {
                "audio": ref_filename,
                "trim": True,
                "length": "20 s",
            },
        },
        "asr": {
            "class_type": "Qwen3ASRTranscribe",
            "inputs": {
                "model": ["asr_m", 0],
                "audio": ["ref", 0],
                "language": "Chinese",
                "context": "",
                "max_new_tokens": 4096,
            },
        },
        "enc": {
            "class_type": "Drama3EncodeVoice",
            "inputs": {
                "codec": ["c", 0],
                "audio": ["ref", 0],
                "transcript": ["asr", 0],
            },
        },
        "gen": {
            "class_type": "Drama3Generate",
            "inputs": {
                "model": ["m", 0],
                "voices.voice0": ["enc", 0],
                "text": text,
                "seed": seed,
                "temperature": 1.0,
                "top_p": 0.9,
                "top_k": 0,
                "max_frames": 0,
                "split_long_text": True,
                "speakers": "one take per line",
                "voice_guidance": 0.0,
            },
        },
        "dec": {
            "class_type": "Drama3Decode",
            "inputs": {
                "codec": ["c", 0],
                "codes": ["gen", 0],
                "shorten_long_pauses": True,
            },
        },
        "save": {
            "class_type": "SaveAudio",
            "inputs": {
                "audio": ["dec", 0],
                "filename_prefix": "drama3_verify/smoke",
            },
        },
    }


def main() -> int:
    ref_audio = pick_reference_audio()
    print(f"使用参考音频: {ref_audio}", flush=True)
    text = "[sighs] 喂，你到底听没听清楚啊？[laughs] 不过算了，这次表现还算过得去。"
    graph = build_drama3_clone_graph(ref_audio, text)

    t0 = time.monotonic()
    res = post_json("/prompt", {"prompt": graph})
    if not isinstance(res, dict) or res.get("node_errors") or res.get("error") or not res.get("prompt_id"):
        raise RuntimeError(f"提交失败: {json.dumps(res, ensure_ascii=False)[:800]}")

    pid = res["prompt_id"]
    print(f"prompt_id={pid}，等待 Drama 3 推理完成...", flush=True)

    hist_item = None
    deadline = time.monotonic() + 900
    last_log = 0.0
    while time.monotonic() < deadline:
        hh = get_json(f"/history/{pid}")
        if isinstance(hh, dict) and pid in hh:
            hist_item = hh[pid]
            break
        now = time.monotonic()
        if now - last_log >= 10.0:
            print(f"运行中... 已等待 {now - t0:.0f}s", flush=True)
            last_log = now
        time.sleep(2)

    if hist_item is None:
        raise TimeoutError(f"等待执行超时 (prompt_id={pid})")

    status = hist_item.get("status") or {}
    if status.get("status_str") != "success" or status.get("completed") is False:
        raise RuntimeError(f"执行失败: {status}")

    outputs = []
    for node_out in (hist_item.get("outputs") or {}).values():
        if not isinstance(node_out, dict):
            continue
        for item in node_out.get("audio") or []:
            fname = item.get("filename")
            if not fname:
                continue
            sub = item.get("subfolder") or ""
            p = (COMFY_OUT_ROOT / sub / fname) if sub else (COMFY_OUT_ROOT / fname)
            if p.is_file():
                outputs.append(p)

    if not outputs:
        raise RuntimeError("任务成功但未找到生成的音频文件")

    elapsed = time.monotonic() - t0
    for p in outputs:
        dur_info = ""
        if shutil.which("ffprobe"):
            raw = subprocess.check_output(
                [
                    "ffprobe", "-v", "error",
                    "-show_entries", "format=duration",
                    "-show_entries", "stream=sample_rate,channels,codec_name",
                    "-of", "json", str(p),
                ],
                text=True,
            )
            info = json.loads(raw)
            dur = float((info.get("format") or {}).get("duration") or 0.0)
            st = (info.get("streams") or [{}])[0]
            dur_info = f" | {dur:.2f}s | {st.get('sample_rate')}Hz {st.get('channels')}ch {st.get('codec_name')}"
            if dur < 1.0:
                raise RuntimeError(f"输出音频异常过短 ({dur:.2f}s): {p}")
        print(f"OK 生成成功 ({elapsed:.1f}s): {p} ({p.stat().st_size} B{dur_info})", flush=True)
    return 0


if __name__ == "__main__":
    main()
