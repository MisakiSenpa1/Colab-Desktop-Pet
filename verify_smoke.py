"""verify_smoke.py —— Anima 新实例端到端冒烟验证（可选叠加环境变量 LoRA）。"""

from __future__ import annotations

import os

from colab_remote_common import COMFY_OUT, comfy_extract_outputs, comfy_submit_and_wait

UNET = "anima-base-v1.0.safetensors"
CLIP = "qwen_3_06b_base.safetensors"
VAE = "qwen_image_vae.safetensors"
LORA_CHAR = (os.environ.get("LORA_CHAR") or "").strip()
LORA_STYLE = (os.environ.get("LORA_STYLE") or "").strip()
OUT_DIR = COMFY_OUT / "smoke_verify"


def main() -> None:
    default_pos = (
        "masterpiece, best quality, score_7, 1girl, solo, "
        "silver hair, long hair, blue eyes, gentle smile, casual dress, "
        "simple background, white background, full body, standing, looking at viewer"
    )
    pos = (os.environ.get("SMOKE_PROMPT") or default_pos).strip()
    neg = "worst quality, low quality, blurry, bad anatomy, extra fingers, watermark, text, multiple views"
    graph = {
        "u": {"class_type": "UNETLoader", "inputs": {"unet_name": UNET, "weight_dtype": "default"}},
        "c": {"class_type": "CLIPLoader", "inputs": {"clip_name": CLIP, "type": "stable_diffusion"}},
        "v": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
        "pos": {"class_type": "CLIPTextEncode", "inputs": {"text": pos, "clip": ["c", 0]}},
        "neg": {"class_type": "CLIPTextEncode", "inputs": {"text": neg, "clip": ["c", 0]}},
        "lat": {"class_type": "EmptyLatentImage", "inputs": {"width": 832, "height": 1440, "batch_size": 1}},
        "d": {"class_type": "VAEDecode", "inputs": {"samples": ["ks", 0], "vae": ["v", 0]}},
        "save": {"class_type": "SaveImage", "inputs": {"images": ["d", 0], "filename_prefix": "smoke_verify/smoke"}},
    }
    model_ref = ["u", 0]
    if LORA_CHAR:
        graph["lc"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": model_ref, "lora_name": LORA_CHAR, "strength_model": 1.0}}
        model_ref = ["lc", 0]
    if LORA_STYLE:
        graph["ls"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": model_ref, "lora_name": LORA_STYLE, "strength_model": 1.0}}
        model_ref = ["ls", 0]
    graph["ks"] = {"class_type": "KSampler", "inputs": {"model": model_ref, "positive": ["pos", 0], "negative": ["neg", 0], "latent_image": ["lat", 0], "seed": 3401, "steps": 28, "cfg": 4.5, "sampler_name": "er_sde", "scheduler": "simple", "denoise": 1.0}}
    hist_item, _ = comfy_submit_and_wait("Anima Smoke", graph, timeout_s=360, poll_sec=2.0, log_interval=15.0)
    produced = comfy_extract_outputs(hist_item, COMFY_OUT, keys=("images",), fallback_dir=OUT_DIR)
    if not produced or any((not p.is_file() or p.stat().st_size == 0) for p in produced):
        raise RuntimeError(f"FAIL 未找到有效输出图片: {produced}")
    for f in produced:
        print("OK  生成成功:", f, f.stat().st_size // 1024, "KB")
    print("请核对生成图质量、分辨率与主体细节是否符合预期。")


if __name__ == "__main__":
    main()
