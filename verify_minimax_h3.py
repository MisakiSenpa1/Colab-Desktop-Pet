"""verify_minimax_h3.py —— MiniMax-H3 全生态（生图 + 编辑 + 视频音频）端到端冒烟验证。"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

from colab_remote_common import (
    COMFY_DIR,
    COMFY_OUT,
    comfy_extract_outputs,
    comfy_submit_and_wait,
)

COMFY_INPUT = COMFY_DIR / "input"
VERIFY_DIR = COMFY_OUT / "minimax_verify"

IMAGE_UNET = "minimax_h3_image_pruned_int8_convrot.safetensors"
SHARED_CLIP = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
IMAGE_VAE = "minimax_h3_image_vae_fp16.safetensors"
VIDEO_FL2VA_UNET = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
VIDEO_VAE = "minimax_h3_video_vae_int8_convrot.safetensors"
AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"
FL2V_TURBO_LORA = "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"


def build_h3_image_t2i(prompt: str, width: int = 2048, height: int = 2048, seed: int = 202611) -> dict:
    return {
        "u": {"class_type": "UNETLoader", "inputs": {"unet_name": IMAGE_UNET, "weight_dtype": "default"}},
        "c": {"class_type": "CLIPLoader", "inputs": {"clip_name": SHARED_CLIP, "type": "minimax", "device": "default"}},
        "v": {"class_type": "VAELoader", "inputs": {"vae_name": IMAGE_VAE}},
        "cond": {"class_type": "MiniMaxH3ImageConditioning", "inputs": {"clip": ["c", 0], "vae": ["v", 0], "prompt": prompt, "width": width, "height": height}},
        "neg": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["cond", 0]}},
        "ks": {"class_type": "KSampler", "inputs": {"model": ["u", 0], "positive": ["cond", 0], "negative": ["neg", 0], "latent_image": ["cond", 1], "seed": seed, "steps": 6, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "dec": {"class_type": "VAEDecode", "inputs": {"samples": ["ks", 0], "vae": ["v", 0]}},
        "save": {"class_type": "SaveImage", "inputs": {"images": ["dec", 0], "filename_prefix": "minimax_verify/h3_image_t2i"}},
    }


def build_h3_image_edit(input_filename: str, prompt: str, width: int = 2048, height: int = 2048, seed: int = 202612) -> dict:
    return {
        "u": {"class_type": "UNETLoader", "inputs": {"unet_name": IMAGE_UNET, "weight_dtype": "default"}},
        "c": {"class_type": "CLIPLoader", "inputs": {"clip_name": SHARED_CLIP, "type": "minimax", "device": "default"}},
        "v": {"class_type": "VAELoader", "inputs": {"vae_name": IMAGE_VAE}},
        "img1": {"class_type": "LoadImage", "inputs": {"image": input_filename}},
        "cond": {"class_type": "MiniMaxH3ImageConditioning", "inputs": {"clip": ["c", 0], "vae": ["v", 0], "prompt": prompt, "width": width, "height": height, "images.image_1": ["img1", 0]}},
        "neg": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["cond", 0]}},
        "ks": {"class_type": "KSampler", "inputs": {"model": ["u", 0], "positive": ["cond", 0], "negative": ["neg", 0], "latent_image": ["cond", 1], "seed": seed, "steps": 6, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "dec": {"class_type": "VAEDecode", "inputs": {"samples": ["ks", 0], "vae": ["v", 0]}},
        "save": {"class_type": "SaveImage", "inputs": {"images": ["dec", 0], "filename_prefix": "minimax_verify/h3_image_edit"}},
    }


def build_h3_video_i2v(first_frame_filename: str, prompt: str, width: int = 768, height: int = 768, length: int = 73, steps: int = 8, seed: int = 202613) -> dict:
    return {
        "u": {"class_type": "UNETLoader", "inputs": {"unet_name": VIDEO_FL2VA_UNET, "weight_dtype": "default"}},
        "lora": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["u", 0], "lora_name": FL2V_TURBO_LORA, "strength_model": 1.0}},
        "c": {"class_type": "CLIPLoader", "inputs": {"clip_name": SHARED_CLIP, "type": "minimax", "device": "default"}},
        "vv": {"class_type": "VAELoader", "inputs": {"vae_name": VIDEO_VAE}},
        "av": {"class_type": "VAELoader", "inputs": {"vae_name": AUDIO_VAE}},
        "img": {"class_type": "LoadImage", "inputs": {"image": first_frame_filename}},
        "i2v": {"class_type": "MiniMaxH3ImageToVideo", "inputs": {"clip": ["c", 0], "vae": ["vv", 0], "prompt": prompt, "width": width, "height": height, "length": length, "first_frame": ["img", 0]}},
        "noise": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "guider": {"class_type": "BasicGuider", "inputs": {"model": ["lora", 0], "conditioning": ["i2v", 0]}},
        "sampler": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}},
        "sched": {"class_type": "BasicScheduler", "inputs": {"model": ["lora", 0], "scheduler": "simple", "steps": steps, "denoise": 1.0}},
        "sample": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["noise", 0], "guider": ["guider", 0], "sampler": ["sampler", 0], "sigmas": ["sched", 0], "latent_image": ["i2v", 1]}},
        "vdec": {"class_type": "VAEDecode", "inputs": {"samples": ["sample", 0], "vae": ["vv", 0]}},
        "adec": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["sample", 0], "vae": ["av", 0]}},
        "cv": {"class_type": "CreateVideo", "inputs": {"images": ["vdec", 0], "audio": ["adec", 0], "fps": 24.0}},
        "sv": {"class_type": "SaveVideo", "inputs": {"video": ["cv", 0], "filename_prefix": "minimax_verify/h3_video_i2v", "format": "auto"}},
    }


def probe_and_snapshot_video(video_path: Path) -> None:
    ffprobe, ffmpeg = shutil.which("ffprobe"), shutil.which("ffmpeg")
    if ffprobe:
        out = subprocess.check_output(
            [ffprobe, "-v", "error", "-show_entries", "stream=codec_type,codec_name,width,height,r_frame_rate,channels,sample_rate,duration", "-of", "json", str(video_path)],
            stdin=subprocess.DEVNULL, text=True,
        )
        print(f"[ffprobe 媒体轨校验] {video_path.name}:\n{out.strip()}", flush=True)
        streams = json.loads(out).get("streams") or []
        if not any(s.get("codec_type") == "video" for s in streams) or not any(s.get("codec_type") == "audio" for s in streams):
            raise RuntimeError(f"视频流或音频流缺失: {video_path.name}")
    if ffmpeg:
        for idx, ts in enumerate(("0.1", "1.5", "2.8"), start=1):
            subprocess.run(
                [ffmpeg, "-y", "-ss", ts, "-i", str(video_path), "-frames:v", "1", "-q:v", "2", str(VERIFY_DIR / f"video_frame_{idx}.jpg")],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="MiniMax-H3 全生态端到端验证脚本")
    parser.add_argument("--skip-edit", action="store_true", help="跳过图生图编辑验证")
    args, _ = parser.parse_known_args()
    skip_edit = args.skip_edit or os.environ.get("MINIMAX_SKIP_EDIT", "").strip() in ("1", "true", "yes")

    VERIFY_DIR.mkdir(parents=True, exist_ok=True)
    COMFY_INPUT.mkdir(parents=True, exist_ok=True)

    t2i_hist, t2i_s = comfy_submit_and_wait(
        "1/3 MiniMax-H3-Image T2I (2048x2048)",
        build_h3_image_t2i(
            "Studio product photography on a warm minimalist oak desk by a sunlit window: "
            "a cute cream-and-coral ceramic desk clock-work robot figurine holding a tiny brass lantern "
            "with a glowing warm amber core. Clean typography plaque at the bottom reading 'MINIMAX H3'."
        ),
    )
    t2i_files = comfy_extract_outputs(t2i_hist, COMFY_OUT)
    if not t2i_files:
        raise RuntimeError("MiniMax-H3-Image T2I 未生成任何输出文件")
    t2i_img = t2i_files[0]
    ref_input_name = "h3_smoke_robot_ref.png"
    shutil.copy2(t2i_img, COMFY_INPUT / ref_input_name)

    edit_s, edit_img = 0.0, None
    if not skip_edit:
        edit_hist, edit_s = comfy_submit_and_wait(
            "2/3 MiniMax-H3-Image Edit (2048x2048)",
            build_h3_image_edit(
                ref_input_name,
                "Keep the exact desk, sunlit window, brass lantern, pose and composition of <Picture 1>, "
                "change the ceramic robot figurine color to deep sapphire blue porcelain and make lantern glow emerald cyan.",
            ),
        )
        edit_files = comfy_extract_outputs(edit_hist, COMFY_OUT)
        if not edit_files:
            raise RuntimeError("MiniMax-H3-Image Edit 未生成任何输出文件")
        edit_img = edit_files[0]

    vid_hist, vid_s = comfy_submit_and_wait(
        "3/3 MiniMax-H3 I2V+Audio (768x768, 73frames, 8steps)",
        build_h3_video_i2v(
            ref_input_name,
            "Realistic studio macro shot, smooth gentle camera push-in: "
            "the cute ceramic robot from <Picture 1> blinks its warm eyes, raises the tiny brass lantern slightly. "
            "Audio: delicate mechanical gear clicks, soft chime ring, gentle warm ambient room tone.",
        ),
    )
    mp4_files = [p for p in comfy_extract_outputs(vid_hist, COMFY_OUT) if p.suffix.lower() in (".mp4", ".webm", ".mkv", ".mov")]
    if not mp4_files:
        raise RuntimeError("MiniMax-H3 视频任务未产出视频文件")
    mp4_path = mp4_files[0]
    probe_and_snapshot_video(mp4_path)

    report = {
        "t2i_seconds": round(t2i_s, 2),
        "t2i_file": t2i_img.name,
        "edit_seconds": round(edit_s, 2) if edit_img else None,
        "edit_file": edit_img.name if edit_img else None,
        "video_seconds": round(vid_s, 2),
        "video_file": mp4_path.name,
    }
    (VERIFY_DIR / "minimax_verify_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("\nOK MiniMax-H3 全生态验证全部通过!", json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
