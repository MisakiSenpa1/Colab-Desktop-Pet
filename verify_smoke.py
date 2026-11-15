"""verify_smoke.py —— 新实例端到端验证：使用 Anima 底座生成 1 张测试图并落盘（可选叠加环境变量 LoRA）

用法（实例内，激活 comfy311 后，或通过 colab_ctl.py smoke 执行）：
    python verify_smoke.py
成功标志：打印 OK + 输出文件路径。
"""
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

API = os.environ.get("COMFY", "http://127.0.0.1:8188").rstrip("/")
UNET = "anima-base-v1.0.safetensors"
CLIP = "qwen_3_06b_base.safetensors"
VAE = "qwen_image_vae.safetensors"
LORA_CHAR = (os.environ.get("LORA_CHAR") or "").strip()
LORA_STYLE = (os.environ.get("LORA_STYLE") or "").strip()
COMFY_OUT_ROOT = Path(os.environ.get("COMFY_OUT", "/root/comfy_out"))
OUT_DIR = COMFY_OUT_ROOT / "smoke_verify"


def get_json(path):
    """请求 ComfyUI GET 接口并解析 JSON。"""
    with urllib.request.urlopen(f"{API}{path}", timeout=30) as r:
        return json.loads(r.read().decode("utf-8") or "{}")


def post(path, payload):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(f"{API}{path}", data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def main():
    default_pos = (
        "masterpiece, best quality, score_7, 1girl, solo, "
        "silver hair, long hair, blue eyes, gentle smile, casual dress, "
        "simple background, white background, full body, standing, looking at viewer"
    )
    pos = (os.environ.get("SMOKE_PROMPT") or default_pos).strip()
    neg = (
        "worst quality, low quality, blurry, bad anatomy, extra fingers, "
        "watermark, text, multiple views"
    )

    g = {
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
        g["lc"] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {"model": model_ref, "lora_name": LORA_CHAR, "strength_model": 1.0},
        }
        model_ref = ["lc", 0]
    if LORA_STYLE:
        g["ls"] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {"model": model_ref, "lora_name": LORA_STYLE, "strength_model": 1.0},
        }
        model_ref = ["ls", 0]
    g["ks"] = {
        "class_type": "KSampler",
        "inputs": {
            "model": model_ref,
            "positive": ["pos", 0],
            "negative": ["neg", 0],
            "latent_image": ["lat", 0],
            "seed": 3401,
            "steps": 28,
            "cfg": 4.5,
            "sampler_name": "er_sde",
            "scheduler": "simple",
            "denoise": 1.0,
        },
    }

    print("提交验证任务...")
    r = post("/prompt", {"prompt": g})
    if not isinstance(r, dict) or r.get("node_errors") or r.get("error") or not r.get("prompt_id"):
        print("FAIL 提交被拒绝:", json.dumps(r, ensure_ascii=False)[:500])
        sys.exit(1)

    pid = r["prompt_id"]
    print("prompt_id:", pid)

    # 仅依据当前 prompt_id 的 /history 执行结果与返回文件清单判定成败，避免误把旧图当成新任务成功
    hist_item = None
    for _ in range(180):
        hh = get_json(f"/history/{pid}")
        if isinstance(hh, dict) and pid in hh:
            hist_item = hh[pid]
            break
        time.sleep(2)

    if hist_item is None:
        print(f"FAIL 等待执行超时 (prompt_id={pid})，请检查 /root/comfy_bw1.log")
        sys.exit(1)

    status = hist_item.get("status") or {}
    status_str = status.get("status_str")
    completed = status.get("completed", True)
    if status_str != "success" or completed is False:
        print(f"FAIL 执行失败 status={status_str} completed={completed} messages={status.get('messages')}")
        sys.exit(1)

    # 从本次任务输出节点提取实际落盘文件
    produced = []
    for node_out in (hist_item.get("outputs") or {}).values():
        if not isinstance(node_out, dict):
            continue
        for img in node_out.get("images") or []:
            if img.get("type", "output") != "output" or not img.get("filename"):
                continue
            sub = img.get("subfolder") or ""
            p = (COMFY_OUT_ROOT / sub / img["filename"]) if sub else (OUT_DIR / img["filename"])
            produced.append(p)

    if not produced:
        print(f"FAIL 历史记录未返回输出文件 (prompt_id={pid})")
        sys.exit(1)

    missing = [p for p in produced if not p.is_file() or p.stat().st_size == 0]
    if missing:
        print("FAIL 本次任务返回的输出文件不存在或为空:", ", ".join(str(p) for p in missing))
        sys.exit(1)

    for f in produced:
        print("OK  生成成功:", f, f.stat().st_size // 1024, "KB")
    print("请核对生成图质量、分辨率与主体细节是否符合预期。")


if __name__ == "__main__":
    main()
