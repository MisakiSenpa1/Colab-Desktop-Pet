"""colab_bootstrap_drama3.py —— Google Colab 环境部署 ComfyUI + Fish Audio Drama 3 语音克隆生态。"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
from pathlib import Path

from colab_remote_common import (
    COMFY_API,
    COMFY_DIR,
    COMFY_OUT,
    RCLONE_CONF,
    download_and_verify,
    ensure_comfyui_repo,
    ensure_tools,
    install_requirements_no_torch,
    load_companion_text,
    load_pet_bridge_init_py,
    run_cmd,
    safe_cmd_line,
    start_comfy_tunnel_standard,
    start_comfyui_and_wait,
    sync_git_repo,
    write_runtime_versions_record,
)

GDRIVE_DRAMA3_REFS_REMOTE = "gdrive_save:Colab_ComfyUI_Models/drama3/refs"
GDRIVE_DRAMA3_VOICES_REMOTE = "gdrive_save:Colab_ComfyUI_Output/drama3_voices"
GDRIVE_WF_REMOTE = "gdrive_save:Colab_ComfyUI_Models/workflows"
_BRIDGE_BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else None
_PET_BRIDGE_INIT_PY = load_pet_bridge_init_py(_BRIDGE_BASE_DIR)

DRAMA3_MODELS = [
    ("https://huggingface.co/PedroMarinhoDev/Drama-3-ComfyUI/resolve/main/drama3_fp8experts_bf16.safetensors", COMFY_DIR / "models" / "diffusion_models" / "drama3_fp8experts_bf16.safetensors", 38578426208),
    ("https://huggingface.co/PedroMarinhoDev/Drama-3-ComfyUI/resolve/main/drama3_codec.safetensors", COMFY_DIR / "models" / "vae" / "drama3_codec.safetensors", 1565647484),
    ("https://huggingface.co/PedroMarinhoDev/Qwen3-ASR-ComfyUI/resolve/main/qwen3_asr_1.7b_bf16.safetensors", COMFY_DIR / "models" / "asr" / "qwen3_asr_1.7b_bf16.safetensors", 4076192608),
]


def check_environment() -> None:
    """检查 GPU 显存、系统内存与可用磁盘空间。"""
    print("=== [1/7] 检测 Drama 3 运行环境 ===", flush=True)
    print("GPU:", subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], stdin=subprocess.DEVNULL, text=True).strip(), flush=True)
    print("内存:\n" + subprocess.check_output(["free", "-h"], stdin=subprocess.DEVNULL, text=True).strip(), flush=True)

    free_gib = shutil.disk_usage("/content").free / (1024 ** 3)
    print(f"可用磁盘空间 (/content): {free_gib:.1f} GiB", flush=True)
    if free_gib < 46.0:
        raise RuntimeError(f"磁盘空间不足: 当前剩余 {free_gib:.1f} GiB，部署 Drama 3 FP8 生态至少需要 ~46.0 GiB")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("未检测到可用 CUDA GPU")
    vram_gib = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"PyTorch: {torch.__version__} | CUDA: {torch.version.cuda} | SM: {torch.cuda.get_device_capability(0)} | VRAM: {vram_gib:.1f} GiB", flush=True)
    if vram_gib < 35.0:
        print(f"[提示] 当前显存为 {vram_gib:.1f} GiB，FP8 专家将通过系统内存按需缓存运行；推荐 G4 (96GB) 或 A100-80GB 获得满血缓存速度。", flush=True)


def ensure_comfyui_and_drama3_node() -> None:
    """准备 ComfyUI 与 ComfyUI-Drama3 自定义节点及依赖。"""
    print("=== [3/7] 准备 ComfyUI 与 Drama 3 节点 ===", flush=True)
    ensure_comfyui_repo("https://github.com/Comfy-Org/ComfyUI.git", pull_if_no_ref=True)
    drama3_dir = COMFY_DIR / "custom_nodes" / "ComfyUI-Drama3"
    sync_git_repo(
        "https://github.com/PedroMarinhoDev/ComfyUI-Drama3.git",
        drama3_dir,
        ref=(os.environ.get("DRAMA3_NODE_GIT_REF") or "").strip(),
        pull_if_exists=True,
    )
    node_req = drama3_dir / "requirements.txt"
    if node_req.is_file():
        install_requirements_no_torch(node_req, tag="drama3_node")
    print("ComfyUI 与 ComfyUI-Drama3 节点依赖安装完成", flush=True)


def download_drama3_models() -> None:
    """下载 Drama 3 FP8 模型、Codec 编解码器与 Qwen3-ASR 1.7B。"""
    print("=== [4/7] 下载并校验 Drama 3 生态权重 ===", flush=True)
    for url, dst, expected_bytes in DRAMA3_MODELS:
        download_and_verify(url, dst, exact_bytes=expected_bytes, chunk_size_m="8M", summary_interval=5)


def install_workflows_and_refs() -> None:
    """部署 Drama 3 工作流并同步云盘参考音频与保存音色。"""
    print("=== [5/7] 部署工作流与同步精选参考音频 ===", flush=True)
    wf_dir = COMFY_DIR / "user" / "default" / "workflows"
    wf_dir.mkdir(parents=True, exist_ok=True)
    in_dir = COMFY_DIR / "input"
    in_dir.mkdir(parents=True, exist_ok=True)
    voices_dir = COMFY_OUT / "drama3_voices"
    voices_dir.mkdir(parents=True, exist_ok=True)

    rclone_bin = shutil.which("rclone") or "/usr/local/bin/rclone"
    if RCLONE_CONF.is_file() and Path(rclone_bin).is_file():
        probe = subprocess.run([rclone_bin, "lsd", "--config", str(RCLONE_CONF), "--disable-http2", "gdrive_save:Colab_ComfyUI_Models/drama3"], stdin=subprocess.DEVNULL, capture_output=True, text=True, check=False)
        if probe.returncode == 0 and "refs" in (probe.stdout or ""):
            run_cmd([rclone_bin, "copy", "--config", str(RCLONE_CONF), "--disable-http2", GDRIVE_DRAMA3_REFS_REMOTE, str(in_dir), "--include", "*.wav", "--transfers", "8"], check=False)
        probe_out = subprocess.run([rclone_bin, "lsd", "--config", str(RCLONE_CONF), "--disable-http2", "gdrive_save:Colab_ComfyUI_Output"], stdin=subprocess.DEVNULL, capture_output=True, text=True, check=False)
        if probe_out.returncode == 0 and "drama3_voices" in (probe_out.stdout or ""):
            run_cmd([rclone_bin, "copy", "--config", str(RCLONE_CONF), "--disable-http2", GDRIVE_DRAMA3_VOICES_REMOTE, str(voices_dir), "--include", "*.safetensors"], check=False)
        run_cmd([rclone_bin, "copy", "--config", str(RCLONE_CONF), "--disable-http2", GDRIVE_WF_REMOTE, str(wf_dir), "--include", "*.json"], check=False)

    wavs = sorted(p.name for p in in_dir.rglob("*.wav"))
    clone_raw = load_companion_text("drama3_voice_clone.json", _BRIDGE_BASE_DIR)
    tts_raw = load_companion_text("drama3_text_to_speech.json", _BRIDGE_BASE_DIR)
    if not clone_raw or not tts_raw:
        raise FileNotFoundError("未找到随行 Drama 3 工作流文件 (drama3_voice_clone.json / drama3_text_to_speech.json)")
    clone_wf = json.loads(clone_raw)
    tts_wf = json.loads(tts_raw)
    preferred_ref = str(os.environ.get("DRAMA3_DEFAULT_REF_WAV") or "").strip()
    selected_ref = ""
    if preferred_ref and (not wavs or preferred_ref in wavs):
        selected_ref = preferred_ref
    elif wavs:
        selected_ref = wavs[0]

    if selected_ref:
        for node in clone_wf.get("nodes") or []:
            if node.get("type") == "Drama3LoadReferenceAudio":
                if isinstance(node.get("widgets_values"), list) and node["widgets_values"]:
                    node["widgets_values"][0] = selected_ref
                if isinstance(node.get("widgets_values_named"), dict) and "audio" in node["widgets_values_named"]:
                    node["widgets_values_named"]["audio"] = selected_ref
                break

    (wf_dir / "drama3_voice_clone.json").write_text(json.dumps(clone_wf, ensure_ascii=False, indent=2), encoding="utf-8")
    (wf_dir / "drama3_text_to_speech.json").write_text(json.dumps(tts_wf, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  工作流就绪: drama3_voice_clone.json / drama3_text_to_speech.json (参考音频 {len(wavs)} 条)", flush=True)


def _check_drama3_node() -> str | None:
    import urllib.request

    # 存量 ComfyUI 未就绪或探测异常时回落至重启加载 Drama3 节点
    with contextlib.suppress(OSError, ValueError):
        with urllib.request.urlopen(f"{COMFY_API}/object_info/Drama3Loader", timeout=4) as resp:
            txt = resp.read().decode("utf-8", errors="ignore")
            if resp.status == 200 and "Drama3Loader" in txt and "drama3_fp8experts_bf16.safetensors" in txt:
                return None
    return "检测到存量 ComfyUI 未加载 Drama3 节点（跨生态热切换），正在重启 ComfyUI..."


def main() -> None:
    os.environ.setdefault("COMFY_LOG", "/root/comfy_bw1.log")
    check_environment()
    ensure_tools()
    ensure_comfyui_and_drama3_node()
    download_drama3_models()
    install_workflows_and_refs()
    start_comfyui_and_wait(wait_timeout_sec=120, extra_ready_probe=_check_drama3_node)
    start_comfy_tunnel_standard()
    write_runtime_versions_record(
        "drama3",
        {
            "comfyui_commit": safe_cmd_line(["git", "-C", str(COMFY_DIR), "rev-parse", "--short", "HEAD"]),
            "drama3_node_commit": safe_cmd_line(["git", "-C", str(COMFY_DIR / "custom_nodes" / "ComfyUI-Drama3"), "rev-parse", "--short", "HEAD"]),
            "bridge_version": 3,
        },
    )


if __name__ == "__main__":
    main()
