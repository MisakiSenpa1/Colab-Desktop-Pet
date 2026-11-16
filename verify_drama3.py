"""verify_drama3.py —— Fish Audio Drama 3 端到端冒烟验证。"""

from __future__ import annotations

import json
import shutil
import subprocess

from colab_remote_common import COMFY_DIR, COMFY_OUT, comfy_extract_outputs, comfy_submit_and_wait

COMFY_INPUT = COMFY_DIR / "input"
DRAMA3_MODEL = "drama3_fp8experts_bf16.safetensors"
DRAMA3_CODEC = "drama3_codec.safetensors"
ASR_MODEL = "qwen3_asr_1.7b_bf16.safetensors"
PREFERRED_REF = "main_01_我操宝宝原来你长这样.wav"


def pick_reference_audio() -> str:
    if (COMFY_INPUT / PREFERRED_REF).is_file():
        return PREFERRED_REF
    wavs = sorted(p.name for p in COMFY_INPUT.rglob("*.wav"))
    if wavs:
        return wavs[0]
    raise FileNotFoundError(f"{COMFY_INPUT} 下未找到任何 .wav 参考音频，请先运行 sync-drama3-refs")


def build_drama3_clone_graph(ref_filename: str, text: str, seed: int = 20261115) -> dict:
    return {
        "m": {"class_type": "Drama3Loader", "inputs": {"model_name": DRAMA3_MODEL, "experts": "in RAM", "expert_cache_gb": 0.0, "second_gpu": "none", "split_layer": 0, "pin_experts": True}},
        "c": {"class_type": "Drama3CodecLoader", "inputs": {"codec_name": DRAMA3_CODEC}},
        "asr_m": {"class_type": "Qwen3ASRLoader", "inputs": {"model_name": ASR_MODEL}},
        "ref": {"class_type": "Drama3LoadReferenceAudio", "inputs": {"audio": ref_filename, "trim": True, "length": "20 s"}},
        "asr": {"class_type": "Qwen3ASRTranscribe", "inputs": {"model": ["asr_m", 0], "audio": ["ref", 0], "language": "Chinese", "context": "", "max_new_tokens": 4096}},
        "enc": {"class_type": "Drama3EncodeVoice", "inputs": {"codec": ["c", 0], "audio": ["ref", 0], "transcript": ["asr", 0]}},
        "gen": {"class_type": "Drama3Generate", "inputs": {"model": ["m", 0], "voices.voice0": ["enc", 0], "text": text, "seed": seed, "temperature": 1.0, "top_p": 0.9, "top_k": 0, "max_frames": 0, "split_long_text": True, "speakers": "one take per line", "voice_guidance": 0.0}},
        "dec": {"class_type": "Drama3Decode", "inputs": {"codec": ["c", 0], "codes": ["gen", 0], "shorten_long_pauses": True}},
        "save": {"class_type": "SaveAudio", "inputs": {"audio": ["dec", 0], "filename_prefix": "drama3_verify/smoke"}},
    }


def main() -> int:
    ref_audio = pick_reference_audio()
    print(f"使用参考音频: {ref_audio}", flush=True)
    graph = build_drama3_clone_graph(ref_audio, "[sighs] 喂，你到底听没听清楚啊？[laughs] 不过算了，这次表现还算过得去。")
    hist_item, elapsed = comfy_submit_and_wait("Drama 3 Clone", graph, timeout_s=900, poll_sec=2.0, log_interval=10.0)
    outputs = comfy_extract_outputs(hist_item, COMFY_OUT, keys=("audio",))
    if not outputs:
        raise RuntimeError("任务成功但未找到生成的音频文件")

    for p in outputs:
        dur_info = ""
        if shutil.which("ffprobe"):
            raw = subprocess.check_output(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-show_entries", "stream=sample_rate,channels,codec_name", "-of", "json", str(p)],
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
    raise SystemExit(main())
