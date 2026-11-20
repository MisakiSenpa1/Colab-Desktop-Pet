"""colab_bootstrap_anima.py —— Google Colab GPU 实例内一键部署 Anima + ComfyUI 自举脚本。"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

from colab_remote_common import (
    COMFY_DIR,
    RCLONE_CONF,
    download_and_verify,
    ensure_comfyui_repo,
    ensure_tools,
    load_pet_bridge_init_py,
    run_cmd,
    safe_cmd_line,
    start_comfy_tunnel_standard,
    start_comfyui_and_wait,
    write_runtime_versions_record,
)

COMFY_DB = Path("/root/comfy_db")
GDRIVE_LORA_REMOTE = "gdrive_save:Colab_ComfyUI_Models/loras"
GDRIVE_WF_REMOTE = "gdrive_save:Colab_ComfyUI_Models/workflows"

_BRIDGE_BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else None
_PET_BRIDGE_INIT_PY = load_pet_bridge_init_py(_BRIDGE_BASE_DIR)

ANIMA_BASE_FILES = [
    (
        "diffusion_models/anima-base-v1.0.safetensors",
        "https://huggingface.co/circlestone-labs/Anima/resolve/main/split_files/diffusion_models/anima-base-v1.0.safetensors",
        4182218328,
    ),
    (
        "text_encoders/qwen_3_06b_base.safetensors",
        "https://huggingface.co/circlestone-labs/Anima/resolve/main/split_files/text_encoders/qwen_3_06b_base.safetensors",
        1192135096,
    ),
    (
        "vae/qwen_image_vae.safetensors",
        "https://huggingface.co/circlestone-labs/Anima/resolve/main/split_files/vae/qwen_image_vae.safetensors",
        253806246,
    ),
]

def _load_required_loras() -> dict[str, int]:
    """从可选环境变量 `REQUIRED_LORAS_JSON` 加载 `{文件名: 精确字节数}` 强校验字典。"""
    raw = (os.environ.get("REQUIRED_LORAS_JSON") or "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return {str(k).strip(): int(v) for k, v in parsed.items() if str(k).strip() and int(v) > 0}
    except (ValueError, TypeError) as exc:
        raise ValueError(f"环境变量 REQUIRED_LORAS_JSON 格式无效: {exc}") from exc
    return {}


REQUIRED_LORAS: dict[str, int] = _load_required_loras()


def check_gpu_and_sdpa() -> None:
    """检查 GPU 显卡型号、CUDA 可用性及 BF16 SDPA 算子自检。"""
    print("=== [1/7] GPU 与 PyTorch SDPA 自检 ===", flush=True)
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        raise RuntimeError("未检测到 nvidia-smi，当前实例可能不是 GPU 运行时")
    info = subprocess.check_output(
        [nvidia_smi, "--query-gpu=name,memory.total", "--format=csv,noheader"],
        text=True,
    ).strip()
    print(f"GPU 硬件信息: {info}", flush=True)

    import torch

    print(f"PyTorch 版本: {torch.__version__}, CUDA 可用: {torch.cuda.is_available()}", flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() 为 False，无法启动 GPU 推理")
    q = torch.randn(1, 16, 2048, 64, device="cuda", dtype=torch.bfloat16)
    out = torch.nn.functional.scaled_dot_product_attention(q, q, q)
    torch.cuda.synchronize()
    print(f"SDPA BF16 自检通过: {tuple(out.shape)}", flush=True)
    del q, out
    torch.cuda.empty_cache()


def download_anima_base() -> None:
    """从 Hugging Face 下载并校验 Anima 底座三大组件（UNet / CLIP / VAE）。"""
    print("=== [4/7] 下载并校验 Anima 底座权重 ===", flush=True)
    models_root = COMFY_DIR / "models"
    for rel_path, url, expected_size in ANIMA_BASE_FILES:
        download_and_verify(
            url,
            models_root / rel_path,
            exact_bytes=expected_size,
            chunk_size_m="1M",
            summary_interval=5,
        )


def _is_strict_lora_mode() -> bool:
    """判定是否开启个人增强版私训 LoRA 强制校验（`ANIMA_STRICT_LORAS=1` 或 `COLAB_PET_EDITION=personal`）。"""
    raw_strict = os.environ.get("ANIMA_STRICT_LORAS")
    if raw_strict is not None and raw_strict.strip() != "":
        return raw_strict.strip().lower() in ("1", "true", "yes", "on")
    edition = str(os.environ.get("COLAB_PET_EDITION") or "").strip().lower()
    return edition in ("personal", "enhanced", "pro", "private")


def pull_loras_from_gdrive() -> None:
    """按运行模式从 Google Drive 拉取自训 LoRA 与云盘工作流（公开基础版无 rclone.conf 时平滑跳过）。"""
    print("=== [5/7] 从 Google Drive 拉取自训 LoRA ===", flush=True)
    strict_mode = _is_strict_lora_mode()
    loras_dir = COMFY_DIR / "models" / "loras"
    loras_dir.mkdir(parents=True, exist_ok=True)
    wf_dir = COMFY_DIR / "user" / "default" / "workflows"
    wf_dir.mkdir(parents=True, exist_ok=True)

    if not RCLONE_CONF.is_file():
        if strict_mode:
            raise FileNotFoundError(
                f"未找到 {RCLONE_CONF}，个人增强模式要求先在本机通过 colab upload 上传 rclone.conf"
            )
        print(
            f"[公开基础版] 未检测到 {RCLONE_CONF}，跳过 Google Drive 私训 LoRA 与云盘工作流拉取，将直接使用 Anima 公开底座启动 ComfyUI。",
            flush=True,
        )
        return

    rclone_bin = shutil.which("rclone") or "/usr/local/bin/rclone"
    run_cmd(
        [
            rclone_bin,
            "copy",
            "--config",
            str(RCLONE_CONF),
            GDRIVE_LORA_REMOTE,
            str(loras_dir),
            "--include",
            "*.safetensors",
            "--transfers",
            "4",
        ],
        check=strict_mode,
    )

    if strict_mode:
        for name, expected_size in REQUIRED_LORAS.items():
            fpath = loras_dir / name
            if not fpath.is_file():
                raise FileNotFoundError(f"主力 LoRA 缺失: {fpath}")
            if fpath.stat().st_size != expected_size:
                raise ValueError(
                    f"主力 LoRA 字节数不符: {name} 实际={fpath.stat().st_size} B, 预期={expected_size} B"
                )
            print(f"LoRA 就绪: {name} ({expected_size} B)", flush=True)
    else:
        found_loras = sorted(p.name for p in loras_dir.glob("*.safetensors"))
        if found_loras:
            print(
                f"[公开基础版] 已拉取可选 LoRA 共 {len(found_loras)} 个（未开启 ANIMA_STRICT_LORAS=1，跳过私训 LoRA 强制校验）: {', '.join(found_loras)}",
                flush=True,
            )
        else:
            print(
                "[公开基础版] 未发现私训 LoRA，跳过私训校验并直接使用 Anima 公开底座启动 ComfyUI。",
                flush=True,
            )

    run_cmd(
        [
            rclone_bin,
            "copy",
            "--config",
            str(RCLONE_CONF),
            "--disable-http2",
            GDRIVE_WF_REMOTE,
            str(wf_dir),
            "--include",
            "*.json",
        ],
        check=False,
    )
    print(f"云盘随行工作流就绪，共 {sum(1 for _ in wf_dir.glob('*.json'))} 个", flush=True)


def main() -> None:
    os.environ.setdefault("COMFY_LOG", "/root/comfy_colab.log")
    check_gpu_and_sdpa()
    ensure_tools()
    print("=== [3/7] 准备 ComfyUI 与 Python 依赖 ===", flush=True)
    ensure_comfyui_repo("https://github.com/comfyanonymous/ComfyUI.git", pull_if_no_ref=False)
    download_anima_base()
    pull_loras_from_gdrive()
    COMFY_DB.mkdir(parents=True, exist_ok=True)
    start_comfyui_and_wait(
        db_url=f"sqlite:////{COMFY_DB.as_posix().lstrip('/')}/comfyui.db",
        extra_args=["--disable-auto-launch", "--use-pytorch-cross-attention", "--bf16-text-enc"],
        wait_timeout_sec=90,
    )
    loras = sorted(p.name for p in (COMFY_DIR / "models" / "loras").glob("*.safetensors"))
    if loras:
        print(f"ComfyUI 与 LoRA 已就绪，共 {len(loras)} 个: {', '.join(loras)}", flush=True)
    else:
        print("ComfyUI 已就绪（公开基础版纯 Anima 底座模式，当前未加载私训 LoRA）", flush=True)
    start_comfy_tunnel_standard()
    write_runtime_versions_record(
        "anima",
        {
            "comfyui_commit": safe_cmd_line(["git", "-C", str(COMFY_DIR), "rev-parse", "--short", "HEAD"]),
            "bridge_version": 3,
        },
    )


if __name__ == "__main__":
    main()
