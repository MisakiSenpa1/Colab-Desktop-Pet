"""verify_minimax_h3.py —— MiniMax-H3 全生态（生图 + 编辑 + 视频音频）端到端冒烟验证

用法（在 Colab 实例内或通过 colab_ctl.py smoke --profile minimax 执行）：
    python verify_minimax_h3.py
成功标志：
    1. MiniMax-H3-Image 文生图（2048x2048, 6步, CFG=1）成功出图；
    2. MiniMax-H3-Image 图生图编辑（参考 <Picture 1> 修改主色调）成功出图；
    3. MiniMax-H3 FL2VA 视频+原生立体声（768x768, 73帧/3s, 8步 Turbo LoRA）成功导出 MP4，
       使用 ffprobe 校验视频轨与音频轨均有效，并提取首/中/尾帧截图供直观核验。
"""
import argparse
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
VERIFY_DIR = COMFY_OUT_ROOT / "minimax_verify"

IMAGE_UNET = "minimax_h3_image_pruned_int8_convrot.safetensors"
SHARED_CLIP = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
IMAGE_VAE = "minimax_h3_image_vae_fp16.safetensors"
VIDEO_FL2VA_UNET = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
VIDEO_VAE = "minimax_h3_video_vae_int8_convrot.safetensors"
AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"
FL2V_TURBO_LORA = "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"


def get_json(path: str) -> dict:
    """请求 ComfyUI GET 接口并解析 JSON。"""
    with urllib.request.urlopen(f"{API}{path}", timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8") or "{}")


def post_json(path: str, payload: dict) -> dict:
    """提交 JSON POST 请求至 ComfyUI。"""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{API}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8") or "{}")


def submit_and_wait(label: str, graph: dict, timeout_s: int = 900) -> tuple[dict, float]:
    """提交工作流并等待执行完成，返回 history 详情与实际耗时。"""
    print(f"\n=== [{label}] 提交任务 ===", flush=True)
    t0 = time.monotonic()
    res = post_json("/prompt", {"prompt": graph})
    if not isinstance(res, dict) or res.get("node_errors") or res.get("error") or not res.get("prompt_id"):
        raise RuntimeError(f"[{label}] 提交被 ComfyUI 拒绝: {json.dumps(res, ensure_ascii=False)[:800]}")

    pid = res["prompt_id"]
    print(f"[{label}] prompt_id={pid}，等待执行...", flush=True)

    hist_item = None
    last_log = 0.0
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        hh = get_json(f"/history/{pid}")
        if isinstance(hh, dict) and pid in hh:
            hist_item = hh[pid]
            break
        now = time.monotonic()
        if now - last_log >= 15.0:
            print(f"[{label}] 运行中... 已等待 {now - t0:.0f}s", flush=True)
            last_log = now
        time.sleep(3)

    if hist_item is None:
        raise TimeoutError(f"[{label}] 等待执行超时 ({timeout_s}s)")

    status = hist_item.get("status") or {}
    if status.get("status_str") != "success" or status.get("completed") is False:
        raise RuntimeError(
            f"[{label}] 执行失败: status={status.get('status_str')}, messages={status.get('messages')}"
        )

    elapsed = time.monotonic() - t0
    print(f"[{label}] 执行成功（耗时 {elapsed:.1f}s）", flush=True)
    return hist_item, elapsed


def extract_output_files(hist_item: dict) -> list[Path]:
    """从 ComfyUI history 输出条目提取真实生成文件绝对路径。"""
    files: list[Path] = []
    for node_out in (hist_item.get("outputs") or {}).values():
        if not isinstance(node_out, dict):
            continue
        for key in ("images", "videos", "gifs", "audio"):
            for item in node_out.get(key) or []:
                fname = item.get("filename")
                if not fname:
                    continue
                sub = item.get("subfolder") or ""
                p = (COMFY_OUT_ROOT / sub / fname) if sub else (COMFY_OUT_ROOT / fname)
                if p.is_file() and p not in files:
                    files.append(p)
    return files


def build_h3_image_t2i(prompt: str, width: int = 2048, height: int = 2048, seed: int = 202611) -> dict:
    """构造 MiniMax-H3-Image 文生图 API 工作流。"""
    return {
        "u": {"class_type": "UNETLoader", "inputs": {"unet_name": IMAGE_UNET, "weight_dtype": "default"}},
        "c": {"class_type": "CLIPLoader", "inputs": {"clip_name": SHARED_CLIP, "type": "minimax", "device": "default"}},
        "v": {"class_type": "VAELoader", "inputs": {"vae_name": IMAGE_VAE}},
        "cond": {
            "class_type": "MiniMaxH3ImageConditioning",
            "inputs": {
                "clip": ["c", 0],
                "vae": ["v", 0],
                "prompt": prompt,
                "width": width,
                "height": height,
            },
        },
        "neg": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["cond", 0]}},
        "ks": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["u", 0],
                "positive": ["cond", 0],
                "negative": ["neg", 0],
                "latent_image": ["cond", 1],
                "seed": seed,
                "steps": 6,
                "cfg": 1.0,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": 1.0,
            },
        },
        "dec": {"class_type": "VAEDecode", "inputs": {"samples": ["ks", 0], "vae": ["v", 0]}},
        "save": {"class_type": "SaveImage", "inputs": {"images": ["dec", 0], "filename_prefix": "minimax_verify/h3_image_t2i"}},
    }


def build_h3_image_edit(input_filename: str, prompt: str, width: int = 2048, height: int = 2048, seed: int = 202612) -> dict:
    """构造 MiniMax-H3-Image 图像编辑 API 工作流（参考 <Picture 1>）。"""
    return {
        "u": {"class_type": "UNETLoader", "inputs": {"unet_name": IMAGE_UNET, "weight_dtype": "default"}},
        "c": {"class_type": "CLIPLoader", "inputs": {"clip_name": SHARED_CLIP, "type": "minimax", "device": "default"}},
        "v": {"class_type": "VAELoader", "inputs": {"vae_name": IMAGE_VAE}},
        "img1": {"class_type": "LoadImage", "inputs": {"image": input_filename}},
        "cond": {
            "class_type": "MiniMaxH3ImageConditioning",
            "inputs": {
                "clip": ["c", 0],
                "vae": ["v", 0],
                "prompt": prompt,
                "width": width,
                "height": height,
                "images.image_1": ["img1", 0],
            },
        },
        "neg": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["cond", 0]}},
        "ks": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["u", 0],
                "positive": ["cond", 0],
                "negative": ["neg", 0],
                "latent_image": ["cond", 1],
                "seed": seed,
                "steps": 6,
                "cfg": 1.0,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": 1.0,
            },
        },
        "dec": {"class_type": "VAEDecode", "inputs": {"samples": ["ks", 0], "vae": ["v", 0]}},
        "save": {"class_type": "SaveImage", "inputs": {"images": ["dec", 0], "filename_prefix": "minimax_verify/h3_image_edit"}},
    }


def build_h3_video_i2v(
    first_frame_filename: str,
    prompt: str,
    width: int = 768,
    height: int = 768,
    length: int = 73,
    steps: int = 8,
    seed: int = 202613,
) -> dict:
    """构造 MiniMax-H3 图生视频 + 原生立体声音频 API 工作流（默认启用 8 步 Turbo LoRA）。"""
    return {
        "u": {"class_type": "UNETLoader", "inputs": {"unet_name": VIDEO_FL2VA_UNET, "weight_dtype": "default"}},
        "lora": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {"model": ["u", 0], "lora_name": FL2V_TURBO_LORA, "strength_model": 1.0},
        },
        "c": {"class_type": "CLIPLoader", "inputs": {"clip_name": SHARED_CLIP, "type": "minimax", "device": "default"}},
        "vv": {"class_type": "VAELoader", "inputs": {"vae_name": VIDEO_VAE}},
        "av": {"class_type": "VAELoader", "inputs": {"vae_name": AUDIO_VAE}},
        "img": {"class_type": "LoadImage", "inputs": {"image": first_frame_filename}},
        "i2v": {
            "class_type": "MiniMaxH3ImageToVideo",
            "inputs": {
                "clip": ["c", 0],
                "vae": ["vv", 0],
                "prompt": prompt,
                "width": width,
                "height": height,
                "length": length,
                "first_frame": ["img", 0],
            },
        },
        "noise": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "guider": {"class_type": "BasicGuider", "inputs": {"model": ["lora", 0], "conditioning": ["i2v", 0]}},
        "sampler": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}},
        "sched": {
            "class_type": "BasicScheduler",
            "inputs": {"model": ["lora", 0], "scheduler": "simple", "steps": steps, "denoise": 1.0},
        },
        "sample": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["noise", 0],
                "guider": ["guider", 0],
                "sampler": ["sampler", 0],
                "sigmas": ["sched", 0],
                "latent_image": ["i2v", 1],
            },
        },
        "vdec": {"class_type": "VAEDecode", "inputs": {"samples": ["sample", 0], "vae": ["vv", 0]}},
        "adec": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["sample", 0], "vae": ["av", 0]}},
        "cv": {
            "class_type": "CreateVideo",
            "inputs": {"images": ["vdec", 0], "audio": ["adec", 0], "fps": 24.0},
        },
        "sv": {
            "class_type": "SaveVideo",
            "inputs": {"video": ["cv", 0], "filename_prefix": "minimax_verify/h3_video_i2v", "format": "auto"},
        },
    }


def probe_and_snapshot_video(video_path: Path):
    """使用 ffprobe 校验生成的 MP4 视频与音轨信息，并提取关键帧 JPG 方便直观查验。"""
    ffprobe = shutil.which("ffprobe")
    ffmpeg = shutil.which("ffmpeg")
    if ffprobe:
        out = subprocess.check_output(
            [
                ffprobe,
                "-v", "error",
                "-show_entries", "stream=codec_type,codec_name,width,height,r_frame_rate,channels,sample_rate,duration",
                "-of", "json",
                str(video_path),
            ],
            stdin=subprocess.DEVNULL,
            text=True,
        )
        print(f"[ffprobe 媒体轨校验] {video_path.name}:\n{out.strip()}", flush=True)
        info = json.loads(out)
        streams = info.get("streams") or []
        has_v = any(s.get("codec_type") == "video" for s in streams)
        has_a = any(s.get("codec_type") == "audio" for s in streams)
        if not has_v or not has_a:
            raise RuntimeError(f"视频流或音频流缺失: has_video={has_v}, has_audio={has_a}")

    if ffmpeg:
        for idx, ts in enumerate(("0.1", "1.5", "2.8"), start=1):
            snap = VERIFY_DIR / f"video_frame_{idx}.jpg"
            subprocess.run(
                [ffmpeg, "-y", "-ss", ts, "-i", str(video_path), "-frames:v", "1", "-q:v", "2", str(snap)],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        print(f"已提取视频首/中/尾关键帧至 {VERIFY_DIR}", flush=True)


def main():
    """串联运行 MiniMax-H3 生图、编辑与图生视频音频完整验证。"""
    parser = argparse.ArgumentParser(description="MiniMax-H3 全生态端到端验证脚本")
    parser.add_argument("--skip-edit", action="store_true", help="跳过图生图编辑验证")
    # 兼容 Colab Jupyter 内核自带的 -f <kernel.json> 启动参数
    args, _ = parser.parse_known_args()
    skip_edit = args.skip_edit or os.environ.get("MINIMAX_SKIP_EDIT", "").strip() in ("1", "true", "yes")

    VERIFY_DIR.mkdir(parents=True, exist_ok=True)
    COMFY_INPUT.mkdir(parents=True, exist_ok=True)

    # 1. 文生图（MiniMax-H3-Image）
    t2i_prompt = (
        "Studio product photography on a warm minimalist oak desk by a sunlit window: "
        "a cute cream-and-coral ceramic desk clock-work robot figurine holding a tiny brass lantern "
        "with a glowing warm amber core. Crisp macro textures on matte glazed ceramic, brushed brass, "
        "soft afternoon shadows, clean typography plaque at the bottom reading 'MINIMAX H3'. "
        "Photorealistic, crisp focus, rich surface detail."
    )
    t2i_hist, t2i_s = submit_and_wait("1/3 MiniMax-H3-Image T2I (2048x2048)", build_h3_image_t2i(t2i_prompt))
    t2i_files = extract_output_files(t2i_hist)
    if not t2i_files:
        raise RuntimeError("MiniMax-H3-Image T2I 未生成任何输出文件")
    t2i_img = t2i_files[0]
    print(f"T2I 产物: {t2i_img} ({t2i_img.stat().st_size / 1024:.1f} KiB)", flush=True)

    # 复制一份到 input 目录用于后续编辑和图生视频测试
    ref_input_name = "h3_smoke_robot_ref.png"
    shutil.copy2(t2i_img, COMFY_INPUT / ref_input_name)

    # 2. 图生图编辑（MiniMax-H3-Image Edit）
    edit_s = 0.0
    edit_img = None
    if not skip_edit:
        edit_prompt = (
            "Keep the exact desk, sunlit window, brass lantern, pose and composition of <Picture 1>, "
            "but change the ceramic robot figurine color from cream-and-coral to deep sapphire blue "
            "porcelain with subtle silver starlight glaze patterns, and make the lantern glow emerald cyan."
        )
        edit_hist, edit_s = submit_and_wait(
            "2/3 MiniMax-H3-Image Edit (2048x2048)",
            build_h3_image_edit(ref_input_name, edit_prompt),
        )
        edit_files = extract_output_files(edit_hist)
        if not edit_files:
            raise RuntimeError("MiniMax-H3-Image Edit 未生成任何输出文件")
        edit_img = edit_files[0]
        print(f"Edit 产物: {edit_img} ({edit_img.stat().st_size / 1024:.1f} KiB)", flush=True)

    # 3. 图生视频 + 原生立体声（MiniMax-H3 FL2VA + 8步 Turbo LoRA）
    vid_prompt = (
        "Realistic studio macro shot, smooth gentle camera push-in: "
        "the cute ceramic robot from <Picture 1> blinks its warm eyes, raises the tiny brass lantern slightly "
        "as warm amber light flickers onto the oak desk, soft sunlit dust motes drift slowly in the air. "
        "Audio: delicate mechanical gear clicks, soft chime ring, gentle warm ambient room tone."
    )
    vid_hist, vid_s = submit_and_wait(
        "3/3 MiniMax-H3 I2V+Audio (768x768, 73frames, 8steps)",
        build_h3_video_i2v(ref_input_name, vid_prompt, width=768, height=768, length=73, steps=8),
        timeout_s=900,
    )
    vid_files = extract_output_files(vid_hist)
    mp4_files = [p for p in vid_files if p.suffix.lower() in (".mp4", ".webm", ".mkv", ".mov")]
    if not mp4_files:
        raise RuntimeError(f"MiniMax-H3 视频任务未产出视频文件，outputs={vid_files}")
    mp4_path = mp4_files[0]
    print(f"Video 产物: {mp4_path} ({mp4_path.stat().st_size / (1024**2):.2f} MiB)", flush=True)
    probe_and_snapshot_video(mp4_path)

    report = {
        "t2i_seconds": round(t2i_s, 2),
        "t2i_file": t2i_img.name,
        "edit_seconds": round(edit_s, 2) if edit_img else None,
        "edit_file": edit_img.name if edit_img else None,
        "video_seconds": round(vid_s, 2),
        "video_file": mp4_path.name,
    }
    (VERIFY_DIR / "minimax_verify_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("\nOK MiniMax-H3 全生态验证全部通过!", json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
