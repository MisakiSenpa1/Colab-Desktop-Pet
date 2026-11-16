"""colab_bootstrap_minimax_h3.py —— Google Colab 环境部署 ComfyUI + MiniMax-H3 全生态（生图 + 视频音频）。"""

from __future__ import annotations

import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from colab_remote_common import (
    COMFY_DIR,
    RCLONE_CONF,
    download_and_verify,
    download_with_urllib,
    ensure_comfyui_repo,
    ensure_tools,
    env_flag,
    load_pet_bridge_init_py,
    run_cmd,
    safe_cmd_line,
    start_comfy_tunnel_standard,
    start_comfyui_and_wait,
    write_runtime_versions_record,
)

GDRIVE_WF_REMOTE = "gdrive_save:Colab_ComfyUI_Models/workflows"
_BRIDGE_BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else None
_PET_BRIDGE_INIT_PY = load_pet_bridge_init_py(_BRIDGE_BASE_DIR)

LITE_MODE = env_flag("MINIMAX_LITE", False)
WITH_R2V = False if LITE_MODE else env_flag("MINIMAX_WITH_R2V", True)
WITH_PE = False if LITE_MODE else env_flag("MINIMAX_WITH_PE", True)

CORE_MODELS = [
    ("https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", COMFY_DIR / "models" / "text_encoders" / "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", 10_000_000_000),
    ("https://huggingface.co/Comfy-Org/MiniMax-H3-Image/resolve/main/diffusion_models/minimax_h3_image_pruned_int8_convrot.safetensors", COMFY_DIR / "models" / "diffusion_models" / "minimax_h3_image_pruned_int8_convrot.safetensors", 15_000_000_000),
    ("https://huggingface.co/Comfy-Org/MiniMax-H3-Image/resolve/main/vae/minimax_h3_image_vae_fp16.safetensors", COMFY_DIR / "models" / "vae" / "minimax_h3_image_vae_fp16.safetensors", 900_000_000),
    ("https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors", COMFY_DIR / "models" / "diffusion_models" / "minimax_h3_fl2va_pruned_int8_convrot.safetensors", 15_000_000_000),
    ("https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_video_vae_int8_convrot.safetensors", COMFY_DIR / "models" / "vae" / "minimax_h3_video_vae_int8_convrot.safetensors", 2_000_000_000),
    ("https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_audio_vae_fp32.safetensors", COMFY_DIR / "models" / "vae" / "minimax_h3_audio_vae_fp32.safetensors", 400_000_000),
    ("https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors", COMFY_DIR / "models" / "loras" / "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors", 1_000_000_000),
]

PE_MODELS = [
    ("https://huggingface.co/Comfy-Org/Qwen3.5/resolve/main/text_encoders/qwen3.5_9b_w4a8_gptq.safetensors", COMFY_DIR / "models" / "text_encoders" / "qwen3.5_9b_w4a8_gptq.safetensors", 5_000_000_000),
]

R2V_MODELS = [
    ("https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors", COMFY_DIR / "models" / "diffusion_models" / "minimax_h3_ref2va_pruned_int8_convrot.safetensors", 15_000_000_000),
    ("https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors", COMFY_DIR / "models" / "loras" / "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors", 1_000_000_000),
]

H3_EMBEDDINGS = [
    "minimaxh3_art_is_explosion.safetensors",
    "minimaxh3_blooming_flowers.safetensors",
    "minimaxh3_bullet_time.safetensors",
    "minimaxh3_dark_magic.safetensors",
    "minimaxh3_fire_breath.safetensors",
    "minimaxh3_four_seasons.safetensors",
    "minimaxh3_kiss_camera.safetensors",
    "minimaxh3_spiral_ascent.safetensors",
    "minimaxh3_storm_magic.safetensors",
    "minimaxh3_truman_show.safetensors",
]

WORKFLOW_FILES = [
    "image_minimax_h3_image_t2i.json",
    "image_minimax_h3_image_edit.json",
    "video_minimax_h3_t2v.json",
    "video_minimax_h3_i2v.json",
]

SAMPLE_INPUTS = [
    "vintage_yellow_taxi.png",
    "transparent_rgb_gaming_mouse.png",
    "red_superboy_on_city_roof.png",
    "mecha_dragon_lightning.png",
]


def check_environment() -> None:
    """检查 GPU 显存、系统内存、可用磁盘与 PyTorch CUDA 13.0 支持情况。"""
    print("=== [1/7] 检测 MiniMax-H3 运行环境 ===", flush=True)
    print("GPU:", subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], stdin=subprocess.DEVNULL, text=True).strip(), flush=True)
    print("内存:\n" + subprocess.check_output(["free", "-h"], stdin=subprocess.DEVNULL, text=True).strip(), flush=True)

    free_gib = shutil.disk_usage("/content").free / (1024 ** 3)
    print(f"可用磁盘空间 (/content): {free_gib:.1f} GiB", flush=True)
    needed_gib = 68.0 + (7.5 if WITH_PE else 0.0) + (22.0 if WITH_R2V else 0.0)
    if free_gib < needed_gib:
        raise RuntimeError(f"磁盘空间不足: 当前剩余 {free_gib:.1f} GiB，部署当前模式至少需要 ~{needed_gib:.1f} GiB")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("未检测到可用 CUDA GPU")
    vram_gib = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    cuda_ver = str(torch.version.cuda or "")
    print(f"PyTorch: {torch.__version__} | CUDA: {cuda_ver} | SM: {torch.cuda.get_device_capability(0)} | VRAM: {vram_gib:.1f} GiB", flush=True)
    major = int(cuda_ver.split(".")[0]) if cuda_ver and cuda_ver[0].isdigit() else 0
    if major < 13:
        print(f"[警告] 当前 PyTorch CUDA 版本为 {cuda_ver} (<13.0)，comfy-kitchen CUDA 加速核将自动禁用并回退。", flush=True)
    if vram_gib < 35.0:
        print(f"[警告] 当前显存仅 {vram_gib:.1f} GiB，运行 MiniMax-H3 INT8 33B 模型可能极慢或 OOM，推荐使用 Colab G4 (96GB) 或 A100-80GB。", flush=True)


def download_minimax_models() -> None:
    """下载 MiniMax-H3 生图 + 视频音频全套模型、Turbo LoRA 与 Embeddings。"""
    print(f"=== [4/7] 下载 MiniMax-H3 生态模型 (WITH_PE={WITH_PE}, WITH_R2V={WITH_R2V}) ===", flush=True)
    items = list(CORE_MODELS) + (PE_MODELS if WITH_PE else []) + (R2V_MODELS if WITH_R2V else [])
    for url, dst, min_bytes in items:
        download_and_verify(url, dst, min_bytes=min_bytes, chunk_size_m="8M", summary_interval=10)

    emb_dir = COMFY_DIR / "models" / "embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        futs = [
            pool.submit(download_with_urllib, f"https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/embeddings/{name}", emb_dir / name)
            for name in H3_EMBEDDINGS
            if not ((emb_dir / name).is_file() and (emb_dir / name).stat().st_size > 1024)
        ]
        for fut in as_completed(futs):
            fut.result()
    print(f"已确认 {len(H3_EMBEDDINGS)} 个 MiniMax-H3 Embeddings 就绪。", flush=True)


def install_workflows_and_inputs() -> None:
    """部署官方 MiniMax-H3 工作流 JSON（统一共享 NVFP4 编码器）与示例输入图片。"""
    print("=== [5/7] 部署开箱即用工作流模板与示例素材 ===", flush=True)
    wf_dir = COMFY_DIR / "user" / "default" / "workflows"
    wf_dir.mkdir(parents=True, exist_ok=True)
    in_dir = COMFY_DIR / "input"
    in_dir.mkdir(parents=True, exist_ok=True)

    for name in WORKFLOW_FILES + (["video_minimax_h3_r2v.json"] if WITH_R2V else []):
        dst = wf_dir / name
        download_with_urllib(f"https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/templates/{name}", dst)
        raw = dst.read_text(encoding="utf-8")
        if "qwen3vl_32b_minimax_h3_int8_convrot.safetensors" in raw:
            patched = raw.replace(
                "qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
                "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
            )
            dst.write_text(patched, encoding="utf-8")
        print(f"  工作流就绪: {dst.name}", flush=True)

    for img_name in SAMPLE_INPUTS:
        dst = in_dir / img_name
        if not (dst.is_file() and dst.stat().st_size > 1024):
            download_with_urllib(f"https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/input/{img_name}", dst)

    rclone_bin = shutil.which("rclone") or "/usr/local/bin/rclone"
    if RCLONE_CONF.is_file() and Path(rclone_bin).is_file():
        probe = subprocess.run([rclone_bin, "lsd", "--config", str(RCLONE_CONF), "--disable-http2", "gdrive_save:Colab_ComfyUI_Models"], stdin=subprocess.DEVNULL, capture_output=True, text=True, check=False)
        if probe.returncode == 0 and "minimax_loras" in (probe.stdout or ""):
            run_cmd([rclone_bin, "copy", "--config", str(RCLONE_CONF), "--disable-http2", "gdrive_save:Colab_ComfyUI_Models/minimax_loras", str(COMFY_DIR / "models" / "loras")], check=False)
        if probe.returncode == 0 and "workflows" in (probe.stdout or ""):
            run_cmd([rclone_bin, "copy", "--config", str(RCLONE_CONF), "--disable-http2", GDRIVE_WF_REMOTE, str(wf_dir), "--include", "*.json"], check=False)


def main() -> None:
    os.environ.setdefault("COMFY_LOG", "/root/comfy_bw1.log")
    check_environment()
    ensure_tools()
    print("=== [3/7] 准备最新官方 ComfyUI 与依赖 ===", flush=True)
    ensure_comfyui_repo("https://github.com/Comfy-Org/ComfyUI.git", pull_if_no_ref=True, require_comfy_kitchen=True)
    download_minimax_models()
    install_workflows_and_inputs()
    start_comfyui_and_wait(wait_timeout_sec=120)
    start_comfy_tunnel_standard()
    write_runtime_versions_record(
        "minimax",
        {
            "comfyui_commit": safe_cmd_line(["git", "-C", str(COMFY_DIR), "rev-parse", "--short", "HEAD"]),
            "bridge_version": 3,
            "with_r2v": WITH_R2V,
            "with_pe": WITH_PE,
        },
    )


if __name__ == "__main__":
    main()
