"""verify_smoke.py —— Anima 新实例端到端冒烟验证（支持纯底座与可选环境变量 LoRA 双模式）。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from colab_remote_common import COMFY_DIR, COMFY_OUT, comfy_extract_outputs, comfy_submit_and_wait

UNET = "anima-base-v1.0.safetensors"
CLIP = "qwen_3_06b_base.safetensors"
VAE = "qwen_image_vae.safetensors"
LORA_CHAR = (os.environ.get("LORA_CHAR") or "").strip()
LORA_STYLE = (os.environ.get("LORA_STYLE") or "").strip()
PERSONAL_SMOKE_LORAS: tuple[str, ...] = ()
OUT_DIR = COMFY_OUT / "smoke_verify"


def _configured_smoke_loras() -> list[str]:
    """收集显式配置的可选冒烟 LoRA 文件名（优先环境变量或模块覆盖，不含硬编码私训权重）。"""
    if PERSONAL_SMOKE_LORAS:
        return [str(x).strip() for x in PERSONAL_SMOKE_LORAS if str(x or "").strip()]
    return [str(x).strip() for x in (LORA_CHAR, LORA_STYLE) if str(x or "").strip()]


def resolve_smoke_mode(
    loras_dir: Path | None = None,
    smoke_mode: str | None = None,
) -> str:
    """判定 Anima 冒烟模式（`base` 公开纯底座 vs `personal` 个人 LoRA 链路）。

    支持环境变量 `ANIMA_SMOKE_MODE=base|personal` 显式控制；未指定时仅当显式配置了
    `LORA_CHAR` / `LORA_STYLE` 且目标文件齐全才切入 `personal`，否则默认运行 `base` 纯底座模式。
    """
    target_dir = Path(loras_dir) if loras_dir is not None else (COMFY_DIR / "models" / "loras")
    raw_mode = str(
        smoke_mode if smoke_mode is not None else (os.environ.get("ANIMA_SMOKE_MODE") or "")
    ).strip().lower()
    if raw_mode == "base":
        return "base"

    configured = _configured_smoke_loras()
    if raw_mode == "personal":
        if not configured:
            raise FileNotFoundError(
                "个人增强版冒烟模式要求先通过 LORA_CHAR / LORA_STYLE 环境变量配置私训 LoRA 文件名"
            )
        missing = [
            name
            for name in configured
            if not ((target_dir / name).is_file() and (target_dir / name).stat().st_size > 0)
        ]
        if missing:
            raise FileNotFoundError(
                f"个人增强版冒烟模式要求私训 LoRA 存在，但 {target_dir} 中缺失: {', '.join(missing)}"
            )
        return "personal"

    if not configured:
        return "base"
    missing = [
        name
        for name in configured
        if not ((target_dir / name).is_file() and (target_dir / name).stat().st_size > 0)
    ]
    return "base" if missing else "personal"


def build_smoke_graph(mode: str = "base") -> tuple[dict[str, Any], str]:
    """按冒烟模式构建 ComfyUI API 工作流图及对应的验收提示文本。"""
    norm_mode = "personal" if str(mode or "").strip().lower() == "personal" else "base"
    custom_pos = (os.environ.get("SMOKE_PROMPT") or "").strip()
    if norm_mode == "personal":
        pos = custom_pos or (
            "masterpiece, best quality, anime illustration, 1girl, solo, "
            "serene sunlit garden, clear blue sky, soft lighting, detailed composition, looking at viewer"
        )
        neg = "worst quality, low quality, blurry, bad anatomy, extra fingers, watermark, text, multiple views"
        graph: dict[str, Any] = {
            "u": {"class_type": "UNETLoader", "inputs": {"unet_name": UNET, "weight_dtype": "default"}},
            "c": {"class_type": "CLIPLoader", "inputs": {"clip_name": CLIP, "type": "stable_diffusion"}},
            "v": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
            "pos": {"class_type": "CLIPTextEncode", "inputs": {"text": pos, "clip": ["c", 0]}},
            "neg": {"class_type": "CLIPTextEncode", "inputs": {"text": neg, "clip": ["c", 0]}},
            "lat": {"class_type": "EmptyLatentImage", "inputs": {"width": 832, "height": 1440, "batch_size": 1}},
            "d": {"class_type": "VAEDecode", "inputs": {"samples": ["ks", 0], "vae": ["v", 0]}},
            "save": {"class_type": "SaveImage", "inputs": {"images": ["d", 0], "filename_prefix": "smoke_verify/smoke"}},
        }
        model_ref: list[Any] = ["u", 0]
        if LORA_CHAR:
            graph["lc"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": model_ref, "lora_name": LORA_CHAR, "strength_model": 1.0}}
            model_ref = ["lc", 0]
        if LORA_STYLE:
            graph["ls"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": model_ref, "lora_name": LORA_STYLE, "strength_model": 1.0}}
            model_ref = ["ls", 0]
        graph["ks"] = {"class_type": "KSampler", "inputs": {"model": model_ref, "positive": ["pos", 0], "negative": ["neg", 0], "latent_image": ["lat", 0], "seed": 3401, "steps": 28, "cfg": 4.5, "sampler_name": "er_sde", "scheduler": "simple", "denoise": 1.0}}
        return graph, "已完成个人 LoRA 链路冒烟，请确认角色特征与画面质量"

    base_pos = custom_pos or (
        "masterpiece, best quality, anime illustration, 1girl, serene sunlit garden, "
        "clear blue sky, soft lighting, detailed composition, looking at viewer"
    )
    base_neg = "worst quality, low quality, blurry, bad anatomy, extra fingers, watermark, text"
    base_graph: dict[str, Any] = {
        "u": {"class_type": "UNETLoader", "inputs": {"unet_name": UNET, "weight_dtype": "default"}},
        "c": {"class_type": "CLIPLoader", "inputs": {"clip_name": CLIP, "type": "stable_diffusion"}},
        "v": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
        "pos": {"class_type": "CLIPTextEncode", "inputs": {"text": base_pos, "clip": ["c", 0]}},
        "neg": {"class_type": "CLIPTextEncode", "inputs": {"text": base_neg, "clip": ["c", 0]}},
        "lat": {"class_type": "EmptyLatentImage", "inputs": {"width": 832, "height": 1440, "batch_size": 1}},
        "ks": {"class_type": "KSampler", "inputs": {"model": ["u", 0], "positive": ["pos", 0], "negative": ["neg", 0], "latent_image": ["lat", 0], "seed": 3401, "steps": 28, "cfg": 4.5, "sampler_name": "er_sde", "scheduler": "simple", "denoise": 1.0}},
        "d": {"class_type": "VAEDecode", "inputs": {"samples": ["ks", 0], "vae": ["v", 0]}},
        "save": {"class_type": "SaveImage", "inputs": {"images": ["d", 0], "filename_prefix": "smoke_verify/smoke"}},
    }
    return base_graph, "公开基础版纯 Anima 底座冒烟通过，请确认图像清晰、构图正常且无明显噪点或色块畸变"


def main() -> None:
    mode = resolve_smoke_mode()
    graph, verify_hint = build_smoke_graph(mode)
    label = "Anima Smoke (Base)" if mode == "base" else "Anima Smoke"
    print(
        f"[Anima 冒烟] 当前模式={mode}（{'公开基础版纯底座 UNET->KSampler' if mode == 'base' else '个人增强版完整 LoRA 链路'}）",
        flush=True,
    )
    hist_item, _ = comfy_submit_and_wait(label, graph, timeout_s=360, poll_sec=2.0, log_interval=15.0)
    produced = comfy_extract_outputs(hist_item, COMFY_OUT, keys=("images",), fallback_dir=OUT_DIR)
    if not produced or any((not p.is_file() or p.stat().st_size == 0) for p in produced):
        raise RuntimeError(f"FAIL 未找到有效输出图片: {produced}")
    for f in produced:
        print("OK  生成成功:", f, f.stat().st_size // 1024, "KB")
    print(verify_hint)


if __name__ == "__main__":
    main()
