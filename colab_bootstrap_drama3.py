"""colab_bootstrap_drama3.py —— Google Colab 环境部署 ComfyUI + Fish Audio Drama 3 语音克隆生态。"""

from __future__ import annotations

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


def _default_voice_clone_workflow() -> dict:
    """内置开箱即用的 Drama 3 声音克隆工作流（默认选 main_01_我操宝宝原来你长这样.wav）。"""
    return {
        "id": "0e5a5f03-7e19-4014-b91c-4311d4434b1e",
        "revision": 0,
        "last_node_id": 11,
        "last_link_id": 11,
        "nodes": [
            {"id": 1, "type": "MarkdownNote", "pos": [20, 20], "size": [560, 640], "flags": {}, "order": 0, "mode": 0, "inputs": [], "outputs": [], "properties": {}, "widgets_values": ["## Fish Audio Drama 3 · 声音克隆工作流\n\n**已预置模型**\n- `drama3_fp8experts_bf16.safetensors` (38.6 GB, FP8 满血版)\n- `drama3_codec.safetensors` (1.57 GB)\n- `qwen3_asr_1.7b_bf16.safetensors` (4.08 GB)\n\n**使用步骤**\n1. 在 **Load Reference Audio** 下拉选一条预置 `main_01_...wav`（也可拖 `.wav` 到桌宠上传）。\n2. **Qwen3-ASR Transcribe** 自动识别参考音频内容并传入编码。\n3. 在 **Drama 3 Generate** 输入台词（支持 `[sighs]`、`[laughs]`、`[whispers]` 等标签），点 **Queue** 生成。\n4. 桌宠会自动把 `.wav` 拉回本地 `_downloads/all/`。"], "color": "#432", "bgcolor": "#653"},
            {"id": 2, "type": "Drama3Loader", "pos": [620, 20], "size": [360, 182], "flags": {}, "order": 1, "mode": 0, "inputs": [{"localized_name": "model_name", "name": "model_name", "type": "COMBO", "widget": {"name": "model_name"}, "link": None}, {"localized_name": "experts", "name": "experts", "type": "COMBO", "widget": {"name": "experts"}, "link": None}, {"localized_name": "expert_cache_gb", "name": "expert_cache_gb", "type": "FLOAT", "widget": {"name": "expert_cache_gb"}, "link": None}, {"localized_name": "second_gpu", "name": "second_gpu", "type": "COMBO", "widget": {"name": "second_gpu"}, "link": None}, {"localized_name": "split_layer", "name": "split_layer", "type": "INT", "widget": {"name": "split_layer"}, "link": None}, {"localized_name": "pin_experts", "name": "pin_experts", "type": "BOOLEAN", "widget": {"name": "pin_experts"}, "link": None}], "outputs": [{"localized_name": "DRAMA3_MODEL", "name": "DRAMA3_MODEL", "type": "DRAMA3_MODEL", "links": [1]}], "properties": {"Node name for S&R": "Drama3Loader"}, "widgets_values": ["drama3_fp8experts_bf16.safetensors", "in RAM", 0, "none", 0, True]},
            {"id": 3, "type": "Drama3CodecLoader", "pos": [620, 260], "size": [360, 58], "flags": {}, "order": 2, "mode": 0, "inputs": [{"localized_name": "codec_name", "name": "codec_name", "type": "COMBO", "widget": {"name": "codec_name"}, "link": None}], "outputs": [{"localized_name": "DRAMA3_CODEC", "name": "DRAMA3_CODEC", "type": "DRAMA3_CODEC", "links": [2, 5]}], "properties": {"Node name for S&R": "Drama3CodecLoader"}, "widgets_values": ["drama3_codec.safetensors"]},
            {"id": 4, "type": "Drama3LoadReferenceAudio", "pos": [620, 380], "size": [360, 230], "flags": {}, "order": 3, "mode": 0, "inputs": [{"localized_name": "audio", "name": "audio", "type": "COMBO", "widget": {"name": "audio"}, "link": None}, {"localized_name": "audioUI", "name": "audioUI", "type": "AUDIO_UI", "widget": {"name": "audioUI"}, "link": None}, {"localized_name": "choose file to upload", "name": "upload", "type": "AUDIOUPLOAD", "widget": {"name": "upload"}, "link": None}, {"localized_name": "trim", "name": "trim", "type": "BOOLEAN", "widget": {"name": "trim"}, "link": None}, {"localized_name": "length", "name": "length", "type": "COMBO", "widget": {"name": "length"}, "link": None}], "outputs": [{"localized_name": "AUDIO", "name": "AUDIO", "type": "AUDIO", "links": [3, 9]}], "properties": {"Node name for S&R": "Drama3LoadReferenceAudio"}, "widgets_values": ["main_01_我操宝宝原来你长这样.wav", "", True, "20 s"]},
            {"id": 5, "type": "Drama3EncodeVoice", "pos": [1020, 610], "size": [380, 108], "flags": {}, "order": 6, "mode": 0, "inputs": [{"localized_name": "codec", "name": "codec", "type": "DRAMA3_CODEC", "link": 2}, {"localized_name": "audio", "name": "audio", "type": "AUDIO", "link": 3}, {"localized_name": "transcript", "name": "transcript", "type": "STRING", "widget": {"name": "transcript"}, "link": 10}], "outputs": [{"localized_name": "DRAMA3_VOICE", "name": "DRAMA3_VOICE", "type": "DRAMA3_VOICE", "links": [4, 11]}], "properties": {"Node name for S&R": "Drama3EncodeVoice"}, "widgets_values": [""]},
            {"id": 6, "type": "Drama3Generate", "pos": [1020, 20], "size": [380, 372], "flags": {}, "order": 7, "mode": 0, "inputs": [{"localized_name": "model", "name": "model", "type": "DRAMA3_MODEL", "link": 1}, {"localized_name": "voice0", "name": "voices.voice0", "shape": 7, "type": "DRAMA3_VOICE", "link": 4}, {"localized_name": "voice1", "name": "voices.voice1", "shape": 7, "type": "DRAMA3_VOICE", "link": None}, {"localized_name": "text", "name": "text", "type": "STRING", "widget": {"name": "text"}, "link": None}, {"localized_name": "seed", "name": "seed", "type": "INT", "widget": {"name": "seed"}, "link": None}, {"localized_name": "temperature", "name": "temperature", "type": "FLOAT", "widget": {"name": "temperature"}, "link": None}, {"localized_name": "top_p", "name": "top_p", "type": "FLOAT", "widget": {"name": "top_p"}, "link": None}, {"localized_name": "top_k", "name": "top_k", "type": "INT", "widget": {"name": "top_k"}, "link": None}, {"localized_name": "max_frames", "name": "max_frames", "type": "INT", "widget": {"name": "max_frames"}, "link": None}, {"localized_name": "split_long_text", "name": "split_long_text", "type": "BOOLEAN", "widget": {"name": "split_long_text"}, "link": None}, {"localized_name": "speakers", "name": "speakers", "type": "COMBO", "widget": {"name": "speakers"}, "link": None}, {"localized_name": "voice_guidance", "name": "voice_guidance", "type": "FLOAT", "widget": {"name": "voice_guidance"}, "link": None}], "outputs": [{"localized_name": "DRAMA3_CODES", "name": "DRAMA3_CODES", "type": "DRAMA3_CODES", "links": [6]}], "properties": {"Node name for S&R": "Drama3Generate"}, "widgets_values": ["[sighs] 哎，这都第几次了？[laughs] 算了，看在今天心情不错的份上，就勉强原谅你这一次吧。", 1, "randomize", 1, 0.9, 0, 0, True, "one take per line", 0.0]},
            {"id": 7, "type": "Drama3Decode", "pos": [1440, 20], "size": [260, 74], "flags": {}, "order": 9, "mode": 0, "inputs": [{"localized_name": "codec", "name": "codec", "type": "DRAMA3_CODEC", "link": 5}, {"localized_name": "codes", "name": "codes", "type": "DRAMA3_CODES", "link": 6}, {"localized_name": "shorten_long_pauses", "name": "shorten_long_pauses", "type": "BOOLEAN", "widget": {"name": "shorten_long_pauses"}, "link": None}], "outputs": [{"localized_name": "AUDIO", "name": "AUDIO", "type": "AUDIO", "links": [7]}], "properties": {"Node name for S&R": "Drama3Decode"}, "widgets_values": [True]},
            {"id": 8, "type": "SaveAudio", "pos": [1440, 140], "size": [360, 112], "flags": {}, "order": 10, "mode": 0, "inputs": [{"localized_name": "audio", "name": "audio", "type": "AUDIO", "link": 7}, {"localized_name": "filename_prefix", "name": "filename_prefix", "type": "STRING", "widget": {"name": "filename_prefix"}, "link": None}, {"localized_name": "audioUI", "name": "audioUI", "type": "AUDIO_UI", "widget": {"name": "audioUI"}, "link": None}], "outputs": [{"localized_name": "audio", "name": "audio", "type": "AUDIO", "links": None}], "properties": {}, "widgets_values": ["drama3/drama3"]},
            {"id": 9, "type": "Qwen3ASRLoader", "pos": [620, 660], "size": [360, 58], "flags": {}, "order": 4, "mode": 0, "inputs": [{"localized_name": "model_name", "name": "model_name", "type": "COMBO", "widget": {"name": "model_name"}, "link": None}], "outputs": [{"localized_name": "QWEN3_ASR_MODEL", "name": "QWEN3_ASR_MODEL", "type": "QWEN3_ASR_MODEL", "links": [8]}], "properties": {"Node name for S&R": "Qwen3ASRLoader"}, "widgets_values": ["qwen3_asr_1.7b_bf16.safetensors"]},
            {"id": 10, "type": "Qwen3ASRTranscribe", "pos": [1020, 380], "size": [380, 180], "flags": {}, "order": 5, "mode": 0, "inputs": [{"localized_name": "model", "name": "model", "type": "QWEN3_ASR_MODEL", "link": 8}, {"localized_name": "audio", "name": "audio", "type": "AUDIO", "link": 9}, {"localized_name": "language", "name": "language", "type": "COMBO", "widget": {"name": "language"}, "link": None}, {"localized_name": "context", "name": "context", "type": "STRING", "widget": {"name": "context"}, "link": None}, {"localized_name": "max_new_tokens", "name": "max_new_tokens", "type": "INT", "widget": {"name": "max_new_tokens"}, "link": None}], "outputs": [{"localized_name": "text", "name": "text", "type": "STRING", "links": [10]}, {"localized_name": "language", "name": "language", "type": "STRING", "links": None}], "properties": {"Node name for S&R": "Qwen3ASRTranscribe"}, "widgets_values": ["Chinese", "", 4096]},
            {"id": 11, "type": "Drama3SaveVoice", "pos": [1440, 610], "size": [360, 58], "flags": {}, "order": 8, "mode": 2, "inputs": [{"localized_name": "voice", "name": "voice", "type": "DRAMA3_VOICE", "link": 11}, {"localized_name": "name", "name": "name", "type": "STRING", "widget": {"name": "name"}, "link": None}], "outputs": [], "properties": {"Node name for S&R": "Drama3SaveVoice"}, "widgets_values": ["drama3/voice"]},
        ],
        "links": [[2, 3, 0, 5, 0, "DRAMA3_CODEC"], [3, 4, 0, 5, 1, "AUDIO"], [5, 3, 0, 7, 0, "DRAMA3_CODEC"], [6, 6, 0, 7, 1, "DRAMA3_CODES"], [7, 7, 0, 8, 0, "AUDIO"], [1, 2, 0, 6, 0, "DRAMA3_MODEL"], [4, 5, 0, 6, 1, "DRAMA3_VOICE"], [8, 9, 0, 10, 0, "QWEN3_ASR_MODEL"], [9, 4, 0, 10, 1, "AUDIO"], [10, 10, 0, 5, 2, "STRING"], [11, 5, 0, 11, 0, "DRAMA3_VOICE"]],
        "groups": [],
        "config": {},
        "extra": {"ds": {"scale": 0.75, "offset": [0, 0]}},
        "version": 0.4,
    }


def _default_tts_workflow() -> dict:
    """内置开箱即用的无参考自由演戏工作流。"""
    return {
        "id": "82e2926c-00f6-40d6-b9e0-0754c41dea11",
        "revision": 0,
        "last_node_id": 6,
        "last_link_id": 4,
        "nodes": [
            {"id": 1, "type": "MarkdownNote", "pos": [20, 20], "size": [560, 470], "flags": {}, "order": 0, "mode": 0, "inputs": [], "outputs": [], "properties": {}, "widgets_values": ["## Fish Audio Drama 3 · 无参考纯文本演戏 (TTS)\n\n本工作流不传入参考音频，每次换 Seed 会随机生成一个新声音。\n要克隆固定音色，请打开 `drama3_voice_clone.json`。"], "color": "#432", "bgcolor": "#653"},
            {"id": 2, "type": "Drama3Loader", "pos": [620, 20], "size": [360, 182], "flags": {}, "order": 1, "mode": 0, "inputs": [{"localized_name": "model_name", "name": "model_name", "type": "COMBO", "widget": {"name": "model_name"}, "link": None}, {"localized_name": "experts", "name": "experts", "type": "COMBO", "widget": {"name": "experts"}, "link": None}, {"localized_name": "expert_cache_gb", "name": "expert_cache_gb", "type": "FLOAT", "widget": {"name": "expert_cache_gb"}, "link": None}, {"localized_name": "second_gpu", "name": "second_gpu", "type": "COMBO", "widget": {"name": "second_gpu"}, "link": None}, {"localized_name": "split_layer", "name": "split_layer", "type": "INT", "widget": {"name": "split_layer"}, "link": None}, {"localized_name": "pin_experts", "name": "pin_experts", "type": "BOOLEAN", "widget": {"name": "pin_experts"}, "link": None}], "outputs": [{"localized_name": "DRAMA3_MODEL", "name": "DRAMA3_MODEL", "type": "DRAMA3_MODEL", "links": [1]}], "properties": {"Node name for S&R": "Drama3Loader"}, "widgets_values": ["drama3_fp8experts_bf16.safetensors", "in RAM", 0, "none", 0, True]},
            {"id": 3, "type": "Drama3CodecLoader", "pos": [620, 260], "size": [360, 58], "flags": {}, "order": 2, "mode": 0, "inputs": [{"localized_name": "codec_name", "name": "codec_name", "type": "COMBO", "widget": {"name": "codec_name"}, "link": None}], "outputs": [{"localized_name": "DRAMA3_CODEC", "name": "DRAMA3_CODEC", "type": "DRAMA3_CODEC", "links": [2]}], "properties": {"Node name for S&R": "Drama3CodecLoader"}, "widgets_values": ["drama3_codec.safetensors"]},
            {"id": 4, "type": "Drama3Generate", "pos": [1020, 20], "size": [380, 332], "flags": {}, "order": 3, "mode": 0, "inputs": [{"localized_name": "model", "name": "model", "type": "DRAMA3_MODEL", "link": 1}, {"localized_name": "voice0", "name": "voices.voice0", "shape": 7, "type": "DRAMA3_VOICE", "link": None}, {"localized_name": "text", "name": "text", "type": "STRING", "widget": {"name": "text"}, "link": None}, {"localized_name": "seed", "name": "seed", "type": "INT", "widget": {"name": "seed"}, "link": None}, {"localized_name": "temperature", "name": "temperature", "type": "FLOAT", "widget": {"name": "temperature"}, "link": None}, {"localized_name": "top_p", "name": "top_p", "type": "FLOAT", "widget": {"name": "top_p"}, "link": None}, {"localized_name": "top_k", "name": "top_k", "type": "INT", "widget": {"name": "top_k"}, "link": None}, {"localized_name": "max_frames", "name": "max_frames", "type": "INT", "widget": {"name": "max_frames"}, "link": None}, {"localized_name": "split_long_text", "name": "split_long_text", "type": "BOOLEAN", "widget": {"name": "split_long_text"}, "link": None}, {"localized_name": "speakers", "name": "speakers", "type": "COMBO", "widget": {"name": "speakers"}, "link": None}, {"localized_name": "voice_guidance", "name": "voice_guidance", "type": "FLOAT", "widget": {"name": "voice_guidance"}, "link": None}], "outputs": [{"localized_name": "DRAMA3_CODES", "name": "DRAMA3_CODES", "type": "DRAMA3_CODES", "links": [3]}], "properties": {"Node name for S&R": "Drama3Generate"}, "widgets_values": ["你好呀！这是 Fish Audio Drama 3 无参考音色的自由演戏测试。[laughs] 每次更换随机种子，都会抽到一位新演员的嗓音。", 1, "randomize", 1, 0.9, 0, 0, True, "one take per line", 0.0]},
            {"id": 5, "type": "Drama3Decode", "pos": [1440, 20], "size": [260, 74], "flags": {}, "order": 4, "mode": 0, "inputs": [{"localized_name": "codec", "name": "codec", "type": "DRAMA3_CODEC", "link": 2}, {"localized_name": "codes", "name": "codes", "type": "DRAMA3_CODES", "link": 3}, {"localized_name": "shorten_long_pauses", "name": "shorten_long_pauses", "type": "BOOLEAN", "widget": {"name": "shorten_long_pauses"}, "link": None}], "outputs": [{"localized_name": "AUDIO", "name": "AUDIO", "type": "AUDIO", "links": [4]}], "properties": {"Node name for S&R": "Drama3Decode"}, "widgets_values": [True]},
            {"id": 6, "type": "SaveAudio", "pos": [1440, 140], "size": [360, 112], "flags": {}, "order": 5, "mode": 0, "inputs": [{"localized_name": "audio", "name": "audio", "type": "AUDIO", "link": 4}, {"localized_name": "filename_prefix", "name": "filename_prefix", "type": "STRING", "widget": {"name": "filename_prefix"}, "link": None}, {"localized_name": "audioUI", "name": "audioUI", "type": "AUDIO_UI", "widget": {"name": "audioUI"}, "link": None}], "outputs": [{"localized_name": "audio", "name": "audio", "type": "AUDIO", "links": None}], "properties": {}, "widgets_values": ["drama3/drama3_tts"]},
        ],
        "links": [[1, 2, 0, 4, 0, "DRAMA3_MODEL"], [2, 3, 0, 5, 0, "DRAMA3_CODEC"], [3, 4, 0, 5, 1, "DRAMA3_CODES"], [4, 5, 0, 6, 0, "AUDIO"]],
        "groups": [],
        "config": {},
        "extra": {"ds": {"scale": 0.9, "offset": [416, 110]}},
        "version": 0.4,
    }


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
    clone_wf = _default_voice_clone_workflow()
    if wavs and "main_01_我操宝宝原来你长这样.wav" not in wavs:
        for node in clone_wf["nodes"]:
            if node.get("type") == "Drama3LoadReferenceAudio":
                node["widgets_values"][0] = wavs[0]
                break

    (wf_dir / "drama3_voice_clone.json").write_text(json.dumps(clone_wf, ensure_ascii=False, indent=2), encoding="utf-8")
    (wf_dir / "drama3_text_to_speech.json").write_text(json.dumps(_default_tts_workflow(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  工作流就绪: drama3_voice_clone.json / drama3_text_to_speech.json (参考音频 {len(wavs)} 条)", flush=True)


def _check_drama3_node() -> str | None:
    import urllib.request

    try:
        with urllib.request.urlopen(f"{COMFY_API}/object_info/Drama3Loader", timeout=4) as resp:
            txt = resp.read().decode("utf-8", errors="ignore")
            if resp.status == 200 and "Drama3Loader" in txt and "drama3_fp8experts_bf16.safetensors" in txt:
                return None
    except Exception:
        pass
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
