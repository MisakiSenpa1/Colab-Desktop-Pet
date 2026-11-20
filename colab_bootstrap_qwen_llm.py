"""colab_bootstrap_qwen_llm.py —— Google Colab 单卡部署 Qwen3.8-Flash-Next-abliterated (llama-server) 自举脚本

设计目标：
    1. 零污染保留 Colab 预装环境：采用独立 C++/CUDA 推理引擎 `llama-server`（b12415），
       完全不修改全局 Python / PyTorch / CUDA 13.0 软件包。
    2. 三级引擎极速就绪：优先从 Google Drive (`rclone`) 秒拉按 GPU SM 架构归档的二进制包，
       未命中时尝试官方 Ubuntu CUDA 预编译包，兜底自动执行 Ninja 单架构源码编译并回传归档。
    3. 针对 180B 稀疏混合模型（125B MoE + 51.2B PLE 静态查表层）精细调优：
       强制通过 `-ot "^per_layer_token_embd\\.weight$=CPU"` 将静态查表层卸载至主机内存，
       并根据 GPU 显存容量动态计算 `--n-cpu-moe` 分流层数。
    4. 默认启用强随机 Bearer API Key 鉴权，零代理损耗直通 Cloudflared 公网隧道，
       并将完整服务契约写入权限 `0600` 的 `/root/colab_service_meta.json` 供本地桌宠读取。

环境变量可选项：
    LLM_QUANT         量化档位：UD-Q4_K_XL（默认）| UD-IQ4_XS | IQ3_XXS
    LLM_CTX_SIZE      上下文长度（默认 32768）
    LLM_WITH_MMPROJ   是否加载多模态视觉投影器（默认 1）
    LLM_WITH_WEBUI    是否启用 llama-server 内置现代聊天 WebUI（默认 1，设为 0 传 --no-ui）
    LLM_GDRIVE_CACHE  是否启用 Google Drive 模型分片读写缓存（默认 0）
    LLM_API_KEY       访问密钥（默认 auto，自动生成 sk-pet-... 并幂等复用已有配置）
"""
import contextlib
import json
import os
import secrets
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path, PurePosixPath

from colab_remote_common import (
    CLOUDFLARED_LOG,
    LLAMA_API_KEY_FILE,
    LLM_TUNNEL_URL_FILE,
    RCLONE_CONF,
    SERVICE_META_FILE,
    TUNNEL_URL_FILE,
    DEFAULT_TUNNEL_URL_RE as TUNNEL_URL_RE,
    download_and_verify,
    download_with_urllib as _download_with_urllib,
    ensure_tools,
    env_flag as _env_flag,
    load_companion_text,
    mask_secret as _mask_key,
    probe_comfy_ready,
    read_json_dict,
    run_cmd,
    safe_extract_tar_gz,
    start_cloudflared_tunnel,
    terminate_proc,
    verify_local_llm_auth_enforced,
    write_private_json as _write_private_meta,
    write_private_text,
    write_runtime_versions_record as _write_versions_common,
)

HF_REPO_180B = "huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-v2-GGUF"
HF_REPO_27B = "huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF"
LLAMA_CPP_TAG = (os.environ.get("LLAMA_CPP_TAG") or "b12415").strip() or "b12415"
LLAMA_DIR = Path("/opt/llama_cpp")

LLAMA_PORT = 18080
LLAMA_INTERNAL_PORT = 18081
LLAMA_API = f"http://127.0.0.1:{LLAMA_PORT}"
LLAMA_INTERNAL_API = f"http://127.0.0.1:{LLAMA_INTERNAL_PORT}"

LLAMA_LOG = Path("/root/llama_server.log")
SSE_GATEWAY_SRC_FILENAME = "colab_sse_gateway.py"
SSE_GATEWAY_VERSION = 2
SSE_GATEWAY_FILE = Path("/root/llm_sse_gateway.py")
SSE_GATEWAY_LOG = Path("/root/llm_sse_gateway.log")
CLOUDFLARED_LLM_LOG = Path("/root/cloudflared_llm.log")

GDRIVE_BIN_REMOTE = "gdrive_save:Colab_LLM_Binaries"

# 180B 精确量化分片配置表（v2 仓库实测路径与最小字节阈值，注意 Unsloth 首片仅为 ~10.9 MB 元数据头）
QUANT_SPECS_180B: dict[str, dict] = {
    "UD-Q4_K_XL": {
        "min_free_gib": 112.0,
        "shards": [
            ("unsloth-UD-Q4_K_XL-V2/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf", 5_000_000),
            ("unsloth-UD-Q4_K_XL-V2/Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf", 48_000_000_000),
            ("unsloth-UD-Q4_K_XL-V2/Qwen3.8-Flash-Next-UD-Q4_K_XL-00003-of-00004.gguf", 48_000_000_000),
            ("unsloth-UD-Q4_K_XL-V2/Qwen3.8-Flash-Next-UD-Q4_K_XL-00004-of-00004.gguf", 11_500_000_000),
        ],
        "mmproj": ("mmproj-Qwen3.8-Flash-Next-BF16.gguf", 850_000_000),
    },
    "UD-IQ4_XS": {
        "min_free_gib": 94.0,
        "shards": [
            ("unsloth-UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf", 5_000_000),
            ("unsloth-UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00002-of-00003.gguf", 48_000_000_000),
            ("unsloth-UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00003-of-00003.gguf", 42_000_000_000),
        ],
        "mmproj": ("mmproj-Qwen3.8-Flash-Next-BF16.gguf", 850_000_000),
    },
    "IQ3_XXS": {
        "min_free_gib": 78.0,
        "shards": [
            ("swift-IQ3_XXS/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf", 38_000_000_000),
            ("swift-IQ3_XXS/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf", 34_000_000_000),
        ],
        "mmproj": ("mmproj-Swift-Qwen3.8-Flash-Next-BF16.gguf", 850_000_000),
    },
}

# 27B 稠密破限版量化配置表（huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF 实测文件与最小字节阈值）
QUANT_SPECS_27B: dict[str, dict] = {
    "UD-Q4_K_XL": {
        "min_free_gib": 24.0,
        "shards": [
            ("Huihui-Qwen3.8-27B-abliterated-UD-Q4_K_XL.gguf", 17_000_000_000),
        ],
        "mmproj": ("mmproj-model-bf16.gguf", 900_000_000),
    },
    "UD-IQ4_XS": {
        "min_free_gib": 20.0,
        "shards": [
            ("Huihui-Qwen3.8-27B-abliterated-UD-IQ4_XS.gguf", 14_000_000_000),
        ],
        "mmproj": ("mmproj-model-bf16.gguf", 900_000_000),
    },
    "IQ3_XXS": {
        "min_free_gib": 16.0,
        "shards": [
            ("Huihui-Qwen3.8-27B-abliterated-UD-IQ3_XXS.gguf", 10_800_000_000),
        ],
        "mmproj": ("mmproj-model-bf16.gguf", 900_000_000),
    },
}


def _env_int(name: str, default: int) -> int:
    """读取整型环境变量，非法格式时回退默认值。"""
    val = (os.environ.get(name) or "").strip()
    if not val:
        return default
    try:
        parsed = int(val)
        return parsed if parsed >= 2048 else default
    except ValueError:
        return default


_raw_variant = (os.environ.get("LLM_MODEL_VARIANT") or "180b").strip().lower()
MODEL_VARIANT = "27b" if _raw_variant == "27b" else "180b"
COHOST_COMFY = _env_flag("LLM_COHOST_COMFY", False)

if MODEL_VARIANT == "27b":
    HF_REPO = HF_REPO_27B
    MODELS_ROOT = Path("/content/models/qwen38_27b")
    MODEL_ALIAS = "Qwen3.8-27B-abliterated"
    GDRIVE_MODEL_REMOTE = "gdrive_save:Colab_LLM_Models_27B"
    QUANT_SPECS = QUANT_SPECS_27B
    _DEFAULT_CTX = 65536
else:
    HF_REPO = HF_REPO_180B
    MODELS_ROOT = Path("/content/models/qwen38_flash_next")
    MODEL_ALIAS = "Qwen3.8-Flash-Next-abliterated"
    GDRIVE_MODEL_REMOTE = "gdrive_save:Colab_LLM_Models"
    QUANT_SPECS = QUANT_SPECS_180B
    _DEFAULT_CTX = 32768

REQ_QUANT = (os.environ.get("LLM_QUANT") or "UD-Q4_K_XL").strip().upper()
CTX_SIZE = _env_int("LLM_CTX_SIZE", _DEFAULT_CTX)
WITH_MMPROJ = _env_flag("LLM_WITH_MMPROJ", True)
WITH_WEBUI = _env_flag("LLM_WITH_WEBUI", True)
WITH_GDRIVE_CACHE = _env_flag("LLM_GDRIVE_CACHE", False)
RAW_API_KEY = (os.environ.get("LLM_API_KEY") or "auto").strip() or "auto"
CF_TUNNEL_TOKEN = (os.environ.get("CF_TUNNEL_TOKEN") or "").strip()
CF_TUNNEL_URL = (os.environ.get("CF_TUNNEL_URL") or "").strip().rstrip("/")
CF_COMFY_TUNNEL_URL = (os.environ.get("CF_COMFY_TUNNEL_URL") or "").strip().rstrip("/")
_VALID_EFFORTS = {"none", "low", "medium", "high", "xhigh"}
_raw_default_effort = (os.environ.get("LLM_DEFAULT_EFFORT") or "medium").strip().lower()
DEFAULT_EFFORT = _raw_default_effort if _raw_default_effort in _VALID_EFFORTS else "medium"


def _calc_n_parallel(vram_gib: float) -> int:
    """根据 GPU 显存、模型变体与是否双栖共驻动态计算并发槽位数。"""
    if COHOST_COMFY:
        return 2
    if MODEL_VARIANT == "27b":
        return 4 if vram_gib >= 38.0 else 2
    return 4 if vram_gib >= 88.0 else 2


def _existing_quant_gib(quant: str) -> float:
    """计算指定量化子目录下已落盘目标文件的总 GiB，保证重复运行时的幂等空间核算。"""
    spec = QUANT_SPECS.get(quant)
    if not spec:
        return 0.0
    qdir = MODELS_ROOT / quant
    if not qdir.is_dir():
        return 0.0
    entries = list(spec["shards"])
    if WITH_MMPROJ and spec.get("mmproj"):
        entries.append(spec["mmproj"])
    total_b = 0
    for rel_path, _ in entries:
        fp = qdir / PurePosixPath(rel_path).name
        if fp.is_file():
            total_b += fp.stat().st_size
    return total_b / (1024 ** 3)


def check_environment() -> tuple[str, float, str]:
    """检查 GPU 算力架构、总显存与 /content 磁盘空间，返回 `(active_quant, vram_gib, sm_num)`。"""
    global CTX_SIZE
    print(f"=== [1/7] 检测 {MODEL_ALIAS} 运行环境 (variant={MODEL_VARIANT}) ===", flush=True)
    if REQ_QUANT not in QUANT_SPECS:
        raise ValueError(
            f"不支持的量化档位 LLM_QUANT={REQ_QUANT!r}，可选值: {', '.join(QUANT_SPECS.keys())}"
        )

    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        raise RuntimeError("未检测到 nvidia-smi，当前实例不是 GPU 运行时！")

    gpu_info = subprocess.check_output(
        [nvidia_smi, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
        stdin=subprocess.DEVNULL,
        text=True,
    ).strip()
    print(f"GPU: {gpu_info}", flush=True)

    mem_info = subprocess.check_output(["free", "-h"], stdin=subprocess.DEVNULL, text=True).strip()
    print(f"内存:\n{mem_info}", flush=True)

    sm_num = ""
    vram_gib = 0.0
    try:
        import torch

        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability(0)
            sm_num = f"{cap[0]}{cap[1]}"
            vram_gib = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            print(
                f"PyTorch: {torch.__version__} | CUDA: {torch.version.cuda} | SM: sm_{sm_num} | VRAM: {vram_gib:.1f} GiB",
                flush=True,
            )
    except (ImportError, RuntimeError, OSError) as exc:
        print(f"[提示] PyTorch CUDA 探测跳过，改用 nvidia-smi ({exc})", flush=True)

    if not sm_num or vram_gib <= 0.0:
        raw_cap = subprocess.check_output(
            [nvidia_smi, "--query-gpu=compute_cap,memory.total", "--format=csv,noheader,nounits"],
            stdin=subprocess.DEVNULL,
            text=True,
        ).splitlines()[0]
        cap_str, mem_mib_str = [x.strip() for x in raw_cap.split(",", 1)]
        sm_num = cap_str.replace(".", "")
        vram_gib = float(mem_mib_str) / 1024.0
        print(f"nvidia-smi 硬件探测: SM=sm_{sm_num} | VRAM={vram_gib:.1f} GiB", flush=True)

    min_vram_gib = 14.5 if MODEL_VARIANT == "27b" else 35.0
    if vram_gib < min_vram_gib:
        if MODEL_VARIANT == "27b":
            raise RuntimeError(
                f"显存不足：当前 GPU 显存仅 {vram_gib:.1f} GiB (< 14.5 GiB)，无法运行 Qwen3.8-27B 模型，"
                f"请至少使用 Colab T4 (16GB) 或推荐的 L4 (24GB)！"
            )
        raise RuntimeError(
            f"显存不足：当前 GPU 显存仅 {vram_gib:.1f} GiB (< 35.0 GiB)，无法运行 180B 规模的 "
            f"Qwen3.8-Flash-Next 模型（T4/L4 不支持），请切换至 Colab G4 (96GB) 或 A100 (80GB 高内存)！"
        )

    Path("/content").mkdir(parents=True, exist_ok=True)
    free_gib = shutil.disk_usage("/content").free / (1024 ** 3)
    active_quant = REQ_QUANT

    if MODEL_VARIANT == "27b":
        # 低显存卡显存平滑自适应：16G T4 降为 IQ3_XXS；24G L4 运行 Q4 时钳制最大上下文至 96K
        if vram_gib < 20.0 and active_quant == "UD-Q4_K_XL":
            print(
                f"[警告] 当前显存仅 {vram_gib:.1f} GiB (< 20.0 GiB)，27B UD-Q4_K_XL 易爆显存，已自动平滑切换为 IQ3_XXS！",
                flush=True,
            )
            active_quant = "IQ3_XXS"
            if CTX_SIZE > 65536:
                CTX_SIZE = 65536
        elif 20.0 <= vram_gib < 26.0 and active_quant == "UD-Q4_K_XL" and CTX_SIZE > 98304:
            print(
                f"[提示] 当前显存 {vram_gib:.1f} GiB (L4 档) 运行 27B UD-Q4_K_XL，已将上下文平滑钳制为 98304 (96K) 保留显存余量。",
                flush=True,
            )
            CTX_SIZE = 98304

        eff_free_gib = free_gib + _existing_quant_gib(active_quant)
        min_needed = float(QUANT_SPECS[active_quant]["min_free_gib"])
        print(
            f"可用磁盘空间 (/content): 剩余 {free_gib:.1f} GiB (计入已有分片有效可用 {eff_free_gib:.1f} GiB, {active_quant} 需 >= {min_needed:.1f} GiB)",
            flush=True,
        )
        if active_quant == "UD-Q4_K_XL" and eff_free_gib < 24.0:
            iq4_eff_free = free_gib + _existing_quant_gib("UD-IQ4_XS")
            if iq4_eff_free >= 20.0:
                print(
                    f"[警告] /content 剩余可用空间 ({eff_free_gib:.1f} GiB) 不足 24.0 GiB，无法容纳 27B UD-Q4_K_XL，"
                    f"已自动平滑降级为 UD-IQ4_XS（需 >= 20.0 GiB）！",
                    flush=True,
                )
                active_quant = "UD-IQ4_XS"
                eff_free_gib = iq4_eff_free
                min_needed = float(QUANT_SPECS[active_quant]["min_free_gib"])
    else:
        eff_free_gib = free_gib + _existing_quant_gib(active_quant)
        min_needed = float(QUANT_SPECS[active_quant]["min_free_gib"])
        print(
            f"可用磁盘空间 (/content): 剩余 {free_gib:.1f} GiB (计入已有分片有效可用 {eff_free_gib:.1f} GiB, {active_quant} 需 >= {min_needed:.1f} GiB)",
            flush=True,
        )

        if active_quant == "UD-Q4_K_XL" and eff_free_gib < 112.0:
            iq4_eff_free = free_gib + _existing_quant_gib("UD-IQ4_XS")
            if iq4_eff_free >= 94.0:
                print(
                    f"[警告] /content 剩余可用空间 ({eff_free_gib:.1f} GiB) 不足 112.0 GiB，无法容纳 UD-Q4_K_XL，"
                    f"已自动平滑降级为 UD-IQ4_XS（需 >= 94.0 GiB）！",
                    flush=True,
                )
                active_quant = "UD-IQ4_XS"
                eff_free_gib = iq4_eff_free
                min_needed = float(QUANT_SPECS[active_quant]["min_free_gib"])

    if eff_free_gib < min_needed:
        raise RuntimeError(
            f"磁盘空间不足：/content 有效可用仅 {eff_free_gib:.1f} GiB，部署 {active_quant} 至少需要 "
            f"{min_needed:.1f} GiB（可尝试改选更小量化档位如 UD-IQ4_XS 或 IQ3_XXS）。"
        )

    return active_quant, vram_gib, sm_num


def _build_ld_library_path(search_root: Path) -> str:
    """搜集指定目录下全部动态库所在路径并与系统 CUDA 库路径拼接为 LD_LIBRARY_PATH。"""
    lib_dirs: list[str] = []
    if search_root.is_dir():
        for p in sorted(search_root.rglob("*.so*")):
            parent = str(p.parent.resolve())
            if parent not in lib_dirs:
                lib_dirs.append(parent)
    for sys_dir in ("/usr/local/cuda/lib64", "/usr/lib64-nvidia", "/usr/lib/x86_64-linux-gnu"):
        if Path(sys_dir).is_dir() and sys_dir not in lib_dirs:
            lib_dirs.append(sys_dir)
    old_ld = os.environ.get("LD_LIBRARY_PATH") or ""
    for item in old_ld.split(":"):
        if item and item not in lib_dirs:
            lib_dirs.append(item)
    return ":".join(lib_dirs)


def _verify_llama_cuda(search_root: Path) -> tuple[Path, str] | None:
    """在目录内定位 llama-server 并执行 --list-devices 校验是否成功识别 CUDA0。"""
    if not search_root.is_dir():
        return None
    candidates = [
        p for p in sorted(search_root.rglob("llama-server"))
        if p.is_file()
    ]
    if not candidates:
        return None
    ld_path = _build_ld_library_path(search_root)
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ld_path
    for bin_path in candidates:
        try:
            os.chmod(bin_path, 0o755)
            res = subprocess.run(
                [str(bin_path), "--list-devices"],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
                env=env,
            )
            combined = f"{res.stdout or ''}\n{res.stderr or ''}"
            if res.returncode == 0 and "CUDA0" in combined:
                print(f"llama-server CUDA 自检通过 ({bin_path}):\n{combined.strip()}", flush=True)
                return bin_path, ld_path
            print(
                f"[校验未通过] {bin_path} --list-devices (rc={res.returncode}): {combined.strip()[:300]}",
                flush=True,
            )
        except Exception as exc:
            print(f"[校验异常] {bin_path}: {exc}", flush=True)
    return None


def _fast_pull_tar(url: str, dst: Path) -> bool:
    """使用 aria2c 或 urllib 快速拉取压缩包，失败时返回 False 不中断主流程。"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.unlink(missing_ok=True)
    if shutil.which("aria2c"):
        res = run_cmd(
            [
                "aria2c",
                "-c",
                "-x", "16",
                "-s", "16",
                "-k", "1M",
                "--console-log-level=warn",
                "--allow-overwrite=true",
                "-d", str(dst.parent),
                "-o", dst.name,
                url,
            ],
            check=False,
        )
        if res.returncode == 0 and dst.is_file() and dst.stat().st_size > 10_000:
            return True
    try:
        _download_with_urllib(url, dst, retries=2)
        return dst.is_file() and dst.stat().st_size > 10_000
    except Exception as exc:
        print(f"  预编译包拉取失败 ({dst.name}): {exc}", flush=True)
        dst.unlink(missing_ok=True)
        return False


def _try_pull_llama_gdrive_cache(
    rclone_bin: str,
    remote_cache_uri: str,
    local_cache_tar: Path,
    cache_tar_name: str,
) -> tuple[Path, str] | None:
    """尝试从 Google Drive 拉取同架构预归档 llama-server 二进制缓存包并校验 CUDA。"""
    print(f"尝试从 Google Drive 拉取缓存引擎包: {remote_cache_uri}", flush=True)
    local_cache_tar.unlink(missing_ok=True)
    pull_res = run_cmd(
        [
            rclone_bin,
            "copyto",
            "--config", str(RCLONE_CONF),
            "--disable-http2",
            remote_cache_uri,
            str(local_cache_tar),
        ],
        check=False,
    )
    if not (pull_res.returncode == 0 and local_cache_tar.is_file() and local_cache_tar.stat().st_size > 100_000):
        return None

    shutil.rmtree(LLAMA_DIR, ignore_errors=True)
    LLAMA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        safe_extract_tar_gz(local_cache_tar, LLAMA_DIR)
        verified = _verify_llama_cuda(LLAMA_DIR)
        if verified is not None:
            print(f"已命中云盘引擎缓存并校验通过: {cache_tar_name}", flush=True)
            return verified
    except Exception as exc:
        print(f"[提示] 云盘缓存包解压或校验异常，继续拉取构建 ({exc})", flush=True)
    finally:
        local_cache_tar.unlink(missing_ok=True)
    return None


def _try_pull_llama_github_prebuilt() -> tuple[Path, str] | None:
    """尝试拉取并验证官方 GitHub Release Ubuntu CUDA 12.8 预编译包。"""
    shutil.rmtree(LLAMA_DIR, ignore_errors=True)
    LLAMA_DIR.mkdir(parents=True, exist_ok=True)
    base_rel = f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_CPP_TAG}"
    main_tar = Path(f"/tmp/llama-{LLAMA_CPP_TAG}-cuda.tar.gz")
    cudart_tar = Path(f"/tmp/cudart-llama-{LLAMA_CPP_TAG}-cuda.tar.gz")
    verified: tuple[Path, str] | None = None

    print(f"尝试拉取官方 CUDA 预编译包 ({LLAMA_CPP_TAG})...", flush=True)
    try:
        ok_main = _fast_pull_tar(
            f"{base_rel}/llama-{LLAMA_CPP_TAG}-bin-ubuntu-cuda-12.8-x64.tar.gz",
            main_tar,
        )
        if ok_main:
            ok_rt = _fast_pull_tar(
                f"{base_rel}/cudart-llama-{LLAMA_CPP_TAG}-bin-ubuntu-cuda-12.8-x64.tar.gz",
                cudart_tar,
            )
            safe_extract_tar_gz(main_tar, LLAMA_DIR)
            if ok_rt and cudart_tar.is_file():
                safe_extract_tar_gz(cudart_tar, LLAMA_DIR)
            verified = _verify_llama_cuda(LLAMA_DIR)
    except Exception as exc:
        print(f"[提示] 预编译包解压或验证未通过，准备回退源码编译: {exc}", flush=True)
        verified = None
    finally:
        main_tar.unlink(missing_ok=True)
        cudart_tar.unlink(missing_ok=True)
    return verified


def _select_cuda_build_env() -> dict[str, str]:
    """构造 CUDA 源码编译环境；当系统默认 gcc >= 15 且装有 gcc-14 时自动切换避免 char8_t 冲突。"""
    build_env = os.environ.copy()
    if shutil.which("gcc-14") and shutil.which("g++-14"):
        gcc_ver_res = subprocess.run(
            ["gcc", "-dumpversion"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            check=False,
        )
        raw_major = (gcc_ver_res.stdout or "").strip().split(".")[0]
        if raw_major.isdigit() and int(raw_major) >= 15:
            build_env["CC"] = "gcc-14"
            build_env["CXX"] = "g++-14"
    return build_env


def _install_compiled_llama_artifacts(src_dir: Path) -> None:
    """将源码编译产出的 `llama-server` 与 `.so` 动态库（保留符号链接）安装到 `LLAMA_DIR/bin`。"""
    shutil.rmtree(LLAMA_DIR, ignore_errors=True)
    bin_dst = LLAMA_DIR / "bin"
    bin_dst.mkdir(parents=True, exist_ok=True)

    built_bins = list((src_dir / "build").rglob("llama-server"))
    if not built_bins:
        raise RuntimeError("源码编译完成后未找到 llama-server 可执行文件")
    shutil.copy2(built_bins[0], bin_dst / "llama-server")
    os.chmod(bin_dst / "llama-server", 0o755)

    for so_file in (src_dir / "build").rglob("*.so*"):
        target_so = bin_dst / so_file.name
        if target_so.exists() or target_so.is_symlink():
            continue
        if so_file.is_symlink():
            link_name = PurePosixPath(os.readlink(so_file)).name
            if link_name and link_name != so_file.name:
                os.symlink(link_name, target_so)
        elif so_file.is_file():
            shutil.copy2(so_file, target_so)


def _compile_llama_server_from_source(sm_num: str) -> tuple[Path, str]:
    """使用 CMake + Ninja 按目标 SM 架构单架构源码编译 llama-server 并执行 CUDA 自检。"""
    print(f"预编译包未就绪或不适配当前架构，启动单架构 (sm_{sm_num}) 极速源码编译...", flush=True)
    run_cmd([
        "bash",
        "-lc",
        "apt-get update -qq && apt-get install -y -qq build-essential cmake ninja-build git",
    ])
    src_dir = Path("/tmp/llama_cpp_src")
    shutil.rmtree(src_dir, ignore_errors=True)
    clone_res = run_cmd(
        [
            "git", "clone", "--depth", "1", "--branch", LLAMA_CPP_TAG,
            "https://github.com/ggml-org/llama.cpp.git", str(src_dir),
        ],
        check=False,
    )
    if clone_res.returncode != 0:
        print(f"[提示] 分支标签 {LLAMA_CPP_TAG} 克隆未命中，回退最新主干克隆...", flush=True)
        shutil.rmtree(src_dir, ignore_errors=True)
        run_cmd([
            "git", "clone", "--depth", "1",
            "https://github.com/ggml-org/llama.cpp.git", str(src_dir),
        ])

    cmake_cmd = [
        "cmake", "-B", "build", "-G", "Ninja",
        "-DGGML_CUDA=ON", f"-DCMAKE_CUDA_ARCHITECTURES={sm_num}",
        "-DLLAMA_CURL=OFF", "-DCMAKE_BUILD_TYPE=Release",
    ]
    build_env = _select_cuda_build_env()
    run_cmd(cmake_cmd, cwd=src_dir, env=build_env)
    nproc = str(max(2, os.cpu_count() or 4))
    run_cmd(["cmake", "--build", "build", "--target", "llama-server", "-j", nproc], cwd=src_dir, env=build_env)

    _install_compiled_llama_artifacts(src_dir)
    shutil.rmtree(src_dir, ignore_errors=True)
    verified = _verify_llama_cuda(LLAMA_DIR)
    if verified is None:
        raise RuntimeError("llama-server 源码编译完成，但 --list-devices 仍未识别到 CUDA0！")
    return verified


def _upload_llama_gdrive_cache(
    rclone_bin: str,
    remote_cache_uri: str,
    local_cache_tar: Path,
) -> None:
    """将已验证通过的 llama-server 打包上传至 Google Drive 供后续实例秒拉复用。"""
    try:
        print(f"正在归档并上传 llama-server 二进制缓存至 {remote_cache_uri} ...", flush=True)
        local_cache_tar.unlink(missing_ok=True)
        with tarfile.open(local_cache_tar, "w:gz") as tf:
            for item in sorted(LLAMA_DIR.rglob("*")):
                if item.is_file() or item.is_symlink():
                    tf.add(item, arcname=item.relative_to(LLAMA_DIR).as_posix(), recursive=False)
        run_cmd(
            [
                rclone_bin,
                "copyto",
                "--config", str(RCLONE_CONF),
                "--disable-http2",
                str(local_cache_tar),
                remote_cache_uri,
            ],
            check=False,
        )
    except Exception as exc:
        print(f"[提示] 上传引擎二进制缓存失败（不影响本次部署）: {exc}", flush=True)
    finally:
        local_cache_tar.unlink(missing_ok=True)


def ensure_llama_server(sm_num: str) -> tuple[Path, str]:
    """按 云盘缓存 -> GitHub CUDA 预编译包 -> Ninja 单架构源码编译 三级准备 llama-server。"""
    print(f"=== [3/7] 准备 llama-server CUDA 推理引擎 (tag={LLAMA_CPP_TAG}, sm_{sm_num}) ===", flush=True)
    existing = _verify_llama_cuda(LLAMA_DIR)
    if existing is not None:
        print(f"检测到本地已就绪的 llama-server，直接复用: {existing[0]}", flush=True)
        return existing

    rclone_bin = shutil.which("rclone") or "/usr/local/bin/rclone"
    has_rclone_conf = RCLONE_CONF.is_file() and Path(rclone_bin).is_file()
    cache_tar_name = f"llama-server-{LLAMA_CPP_TAG}-sm{sm_num}.tar.gz"
    remote_cache_uri = f"{GDRIVE_BIN_REMOTE}/{cache_tar_name}"
    local_cache_tar = Path(f"/tmp/{cache_tar_name}")

    # 1) 优先尝试从 Google Drive 拉取同架构预归档二进制包
    if has_rclone_conf:
        cached = _try_pull_llama_gdrive_cache(rclone_bin, remote_cache_uri, local_cache_tar, cache_tar_name)
        if cached is not None:
            return cached

    # 2) 未命中缓存时尝试拉取官方 GitHub Release CUDA 预编译包，失败则回退源码编译
    verified = _try_pull_llama_github_prebuilt()
    if verified is None:
        verified = _compile_llama_server_from_source(sm_num)

    # 3) 若云盘尚无该架构缓存，打包上传至 Google Drive 供后续实例秒拉复用
    if has_rclone_conf:
        _upload_llama_gdrive_cache(rclone_bin, remote_cache_uri, local_cache_tar)

    print(f"推理引擎依赖安装完成: {verified[0]}", flush=True)
    return verified


def _download_model_file(rel_path: str, dst: Path, min_bytes: int, quant: str):
    """下载单个 GGUF 分片或 mmproj（支持可选云盘缓存与防 .aria2 空洞高速直拉）。"""
    from colab_remote_common import is_complete_file

    dst.parent.mkdir(parents=True, exist_ok=True)
    if is_complete_file(dst, min_bytes=min_bytes):
        print(f"  [已就绪] {dst.name} ({dst.stat().st_size / (1024**3):.2f} GiB)", flush=True)
        return

    url = f"https://huggingface.co/{HF_REPO}/resolve/main/{rel_path}"
    rclone_bin = shutil.which("rclone") or "/usr/local/bin/rclone"
    can_gdrive = WITH_GDRIVE_CACHE and RCLONE_CONF.is_file() and Path(rclone_bin).is_file()
    gdrive_uri = f"{GDRIVE_MODEL_REMOTE}/{quant}/{dst.name}"
    pulled_from_gdrive = False

    if can_gdrive:
        print(f"  [云盘拉取] {dst.name} <- {gdrive_uri}", flush=True)
        res_gd = run_cmd(
            [rclone_bin, "copyto", "--config", str(RCLONE_CONF), "--disable-http2", "--stats", "2s", "--stats-one-line", "-v", gdrive_uri, str(dst)],
            check=False,
        )
        pulled_from_gdrive = res_gd.returncode == 0 and is_complete_file(dst, min_bytes=min_bytes)

    if not pulled_from_gdrive:
        download_and_verify(url, dst, min_bytes=min_bytes, chunk_size_m="1M", summary_interval=2)

    if can_gdrive and not pulled_from_gdrive:
        print(f"  [云盘归档] 正在同步模型缓存至 {gdrive_uri} ...", flush=True)
        run_cmd([rclone_bin, "copyto", "--config", str(RCLONE_CONF), "--disable-http2", str(dst), gdrive_uri], check=False)


def download_qwen_models(active_quant: str) -> tuple[Path, Path | None]:
    """高速拉取目标量化分片与可选视觉投影器，返回 `(首分片路径, mmproj路径或None)`。"""
    print(
        f"=== [4/7] 高速拉取 {MODEL_ALIAS} 量化分片 (quant={active_quant}, mmproj={WITH_MMPROJ}) ===",
        flush=True,
    )
    spec = QUANT_SPECS[active_quant]
    qdir = MODELS_ROOT / active_quant
    qdir.mkdir(parents=True, exist_ok=True)

    items: list[tuple[str, int]] = list(spec["shards"])
    mmproj_entry = spec.get("mmproj") if WITH_MMPROJ else None
    if mmproj_entry:
        items.append(mmproj_entry)

    print(f"DEPLOY_DL_TOTAL: {len(items)}", flush=True)

    for rel_path, min_bytes in items:
        dst = qdir / PurePosixPath(rel_path).name
        _download_model_file(rel_path, dst, min_bytes, active_quant)

    first_shard_name = PurePosixPath(spec["shards"][0][0]).name
    first_shard_path = qdir / first_shard_name
    mmproj_path = (qdir / PurePosixPath(mmproj_entry[0]).name) if mmproj_entry else None
    return first_shard_path, mmproj_path


def _read_existing_meta() -> dict:
    """安全读取已有的 /root/colab_service_meta.json，不存在或损坏时返回空字典。"""
    return read_json_dict(SERVICE_META_FILE)


def _server_supports_api_key_file(server_bin: Path) -> bool:
    """探测 llama-server --help 是否支持 --api-key-file 参数，避免命令行明文暴露密钥。"""
    env = os.environ.copy()
    search_root = LLAMA_DIR if LLAMA_DIR.is_dir() else server_bin.parent
    ld_path = _build_ld_library_path(search_root)
    if ld_path:
        env["LD_LIBRARY_PATH"] = ld_path
    try:
        res = subprocess.run(
            [str(server_bin), "--help"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
            env=env,
            check=False,
        )
        help_txt = f"{res.stdout or ''}\n{res.stderr or ''}"
        return "--api-key-file" in help_txt
    except Exception:
        return False


def build_auth_and_args(
    server_bin: Path,
    first_shard: Path,
    mmproj_path: Path | None,
    active_quant: str,
    vram_gib: float,
) -> tuple[str, list[str]]:
    """生成或幂等复用 API Key，并按显存容量与配置组装精确 llama-server 启动命令。"""
    print("=== [5/7] 配置安全鉴权与运行参数 ===", flush=True)
    old_meta = _read_existing_meta()
    if RAW_API_KEY.lower() == "auto":
        prev_key = str(old_meta.get("api_key") or "").strip()
        if len(prev_key) >= 16:
            api_key = prev_key
            key_src = "复用已有实例密钥"
        else:
            api_key = f"sk-pet-{secrets.token_urlsafe(24)}"
            key_src = "自动生成高熵密钥"
    else:
        api_key = RAW_API_KEY
        key_src = "使用外部指定密钥"

    # 根据显存与模型形态动态配置分流参数：27B 为纯稠密架构，严禁附加 MoE/PLE 卸载参数
    n_cpu_moe: int | None = None
    if MODEL_VARIANT == "27b":
        batch_size = 2048
        ubatch_size = 512 if (COHOST_COMFY or vram_gib < 35.0) else 1024
    elif vram_gib >= 88.0:
        n_cpu_moe = 0
        batch_size = 2048 if COHOST_COMFY else 4096
        ubatch_size = 512 if COHOST_COMFY else 2048
    elif vram_gib >= 75.0:
        n_cpu_moe = (10 if CTX_SIZE >= 262144 else 6) if active_quant == "UD-Q4_K_XL" else 0
        batch_size = 2048
        ubatch_size = 512 if COHOST_COMFY else 1024
    else:
        n_cpu_moe = 28
        batch_size = 2048
        ubatch_size = 512 if COHOST_COMFY else 1024

    n_parallel = _calc_n_parallel(vram_gib)
    try:
        total_ram_mib = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") // (1024 * 1024)
    except Exception:
        total_ram_mib = 32768
    if COHOST_COMFY:
        # 双栖共驻模式下 G4(96G显存/176G内存) 可分配 24 GiB Prompt 缓存，常规卡保留 8 GiB 给 ComfyUI
        cache_ram_mib = 24576 if (total_ram_mib >= 131072 and vram_gib >= 60.0) else 8192
    elif total_ram_mib >= 131072:
        cache_ram_mib = 49152
    elif total_ram_mib >= 65536:
        cache_ram_mib = 16384
    else:
        cache_ram_mib = 8192
    threads_batch = max(4, min(48, os.cpu_count() or 16))
    write_private_text(LLAMA_API_KEY_FILE, api_key + "\n")
    auth_args = (
        ["--api-key-file", str(LLAMA_API_KEY_FILE)]
        if _server_supports_api_key_file(server_bin)
        else ["--api-key", api_key]
    )

    cmd = [
        str(server_bin),
        "-m", str(first_shard),
    ]
    if WITH_MMPROJ and mmproj_path is not None:
        cmd.extend(["--mmproj", str(mmproj_path)])

    cmd.extend([
        "-ngl", "99",
    ])
    if MODEL_VARIANT != "27b" and n_cpu_moe is not None:
        cmd.extend([
            "--n-cpu-moe", str(n_cpu_moe),
            "-ot", r"^per_layer_token_embd\.weight$=CPU",
        ])
    cmd.extend([
        "--flash-attn", "on",
        "-ctk", "q8_0",
        "-ctv", "q8_0",
        "-c", str(CTX_SIZE),
        "-np", str(n_parallel),
        "--kv-unified",
        "--cache-ram", str(cache_ram_mib),
        "-b", str(batch_size),
        "-ub", str(ubatch_size),
        "-tb", str(threads_batch),
        "--host", "127.0.0.1",
        "--port", str(LLAMA_INTERNAL_PORT),
        *auth_args,
        "--alias", MODEL_ALIAS,
    ])
    if DEFAULT_EFFORT == "none":
        cmd.extend(["--chat-template-kwargs", json.dumps({"enable_thinking": False}, separators=(",", ":"))])
    else:
        srv_effort = "xhigh" if DEFAULT_EFFORT == "high" else DEFAULT_EFFORT
        cmd.extend(["--reasoning-effort", srv_effort])
    if not WITH_WEBUI:
        cmd.append("--no-ui")

    moe_desc = "dense" if MODEL_VARIANT == "27b" else str(n_cpu_moe)
    print(
        f"鉴权配置就绪 ({key_src}): key={_mask_key(api_key)} | "
        f"variant={MODEL_VARIANT} | model={MODEL_ALIAS} | quant={active_quant} | ctx={CTX_SIZE} | n_cpu_moe={moe_desc} | "
        f"np={n_parallel} kvu=on cram={cache_ram_mib} cohost={COHOST_COMFY} effort={DEFAULT_EFFORT} b={batch_size} ub={ubatch_size} tb={threads_batch} | "
        f"webui={WITH_WEBUI} | mmproj={WITH_MMPROJ}",
        flush=True,
    )
    return api_key, cmd



def _probe_http_health(port: int, timeout: float = 3.0) -> bool:
    """探测本机指定端口 /health 是否返回 HTTP 200 且 status == 'ok'。"""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=timeout) as resp:
            if resp.status == 200:
                body = json.loads(resp.read().decode("utf-8", errors="ignore") or "{}")
                return body.get("status") == "ok"
    except Exception:
        return False
    return False


def _probe_llama_health(timeout: float = 3.0) -> bool:
    """探测本机内部 llama-server (:18081/health) 是否就绪。"""
    return _probe_http_health(LLAMA_INTERNAL_PORT, timeout=timeout)


def _query_steady_vram() -> tuple[str, float, float]:
    """通过 nvidia-smi 读取模型加载完成后的真实 GPU 名称与稳态显存占用 (GiB)。"""
    try:
        raw = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total", "--format=csv,noheader,nounits"],
            stdin=subprocess.DEVNULL,
            text=True,
        ).splitlines()[0]
        parts = [x.strip() for x in raw.split(",")]
        gpu_name = parts[0].replace("NVIDIA ", "").strip()
        used_gib = round(float(parts[1]) / 1024.0, 1)
        total_gib = round(float(parts[2]) / 1024.0, 1)
        return gpu_name, used_gib, total_gib
    except Exception:
        return "GPU", 0.0, 0.0


def _wait_slots_idle_briefly(api_key: str, max_wait_s: float = 10.0) -> None:
    """若旧 llama-server 尚有活跃生成槽位，最多等待 max_wait_s 秒让其平滑收尾后再重启。"""
    t0 = time.monotonic()
    announced = False
    while time.monotonic() - t0 < max_wait_s:
        active = False
        checked = False
        for port in (LLAMA_PORT, LLAMA_INTERNAL_PORT):
            try:
                req = urllib.request.Request(
                    f"http://127.0.0.1:{port}/slots",
                    headers={"Authorization": f"Bearer {api_key}"},
                )
                with urllib.request.urlopen(req, timeout=1.5) as resp:
                    if resp.status == 200:
                        slots = json.loads(resp.read().decode("utf-8", errors="ignore") or "[]")
                        checked = True
                        if isinstance(slots, list) and any(
                            isinstance(s, dict) and bool(s.get("is_processing")) for s in slots
                        ):
                            active = True
                        break
            except Exception:
                continue
        if not checked or not active:
            return
        if not announced:
            print("检测到旧版推理请求正在收尾，最多等待 10s 后平滑重启...", flush=True)
            announced = True
        time.sleep(1.0)


def _pkill_patterns(*patterns: str) -> None:
    """按模式列表静默清理远端残留进程。"""
    for pat in patterns:
        with contextlib.suppress(OSError, subprocess.SubprocessError):
            subprocess.run(["pkill", "-f", pat], check=False, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _cleanup_llm_processes(*procs: object) -> None:
    """鉴权校验失败时清理 SSE 网关与 llama-server 子进程，防止裸服务继续监听。"""
    for proc in procs:
        terminate_proc(proc)
    if shutil.which("pkill"):
        _pkill_patterns("llm_sse_gateway.py", "llama-server")


def _verify_llm_gateway_auth_enforced(port: int, api_key: str, timeout: float = 3.0) -> tuple[bool, str]:
    """核实本地 LLM 网关（/v1/models）已启用 API Key 鉴权且合法 Bearer 密钥可通行（复用公共实现）。"""
    return verify_local_llm_auth_enforced(port=port, api_key=api_key, timeout=timeout)


def start_llama_and_wait(
    cmd: list[str],
    ld_library_path: str,
    active_quant: str,
    api_key: str,
    n_parallel: int = 2,
) -> tuple[str, float, float]:
    """守护启动 llama-server (:18081) 与 SSE 保活网关 (:18080)，轮询就绪并严格验证鉴权与网关版本一致。"""
    print("=== [6/7] 启动 llama-server 与 SSE 保活网关并等待就绪 ===", flush=True)
    gw_src = (load_companion_text(SSE_GATEWAY_SRC_FILENAME) or "").strip()
    if not gw_src:
        raise FileNotFoundError(f"未找到独立 SSE 网关脚本: {SSE_GATEWAY_SRC_FILENAME}")
    gw_disk_ok = (
        SSE_GATEWAY_FILE.is_file()
        and SSE_GATEWAY_FILE.read_text(encoding="utf-8", errors="ignore").strip() == gw_src
    )
    old_meta = _read_existing_meta()
    gw_ver_ok = old_meta.get("sse_gateway_version") in (None, SSE_GATEWAY_VERSION)
    same_cfg = (
        old_meta.get("kind") in ("llm", "dual")
        and old_meta.get("model_variant", "180b") == MODEL_VARIANT
        and old_meta.get("model") == MODEL_ALIAS
        and old_meta.get("quant") == active_quant
        and int(old_meta.get("ctx_size") or 0) == CTX_SIZE
        and bool(old_meta.get("mmproj_enabled")) == WITH_MMPROJ
        and bool(old_meta.get("webui_enabled")) == WITH_WEBUI
        and bool(old_meta.get("sse_keepalive")) is True
        and bool(old_meta.get("kv_unified")) is True
        and int(old_meta.get("n_parallel") or 1) == n_parallel
        and str(old_meta.get("default_effort") or "") == DEFAULT_EFFORT
        and str(old_meta.get("api_key") or "") == api_key
        and gw_disk_ok
        and gw_ver_ok
    )
    if same_cfg and _probe_http_health(LLAMA_INTERNAL_PORT) and _probe_http_health(LLAMA_PORT):
        auth_ok, auth_reason = _verify_llm_gateway_auth_enforced(LLAMA_PORT, api_key)
        if auth_ok:
            gpu_name, used_g, tot_g = _query_steady_vram()
            print(
                f"llama-server 与 SSE 网关已在运行且配置一致，直接复用。GPU={gpu_name}, VRAM={used_g}/{tot_g} GiB",
                flush=True,
            )
            return gpu_name, used_g, tot_g
        print(f"[警告] 存量 LLM 网关鉴权校验未通过（{auth_reason}），将平滑重启服务...", flush=True)

    old_key = str(old_meta.get("api_key") or "").strip() or api_key
    if _probe_http_health(LLAMA_PORT, timeout=1.0) or _probe_http_health(LLAMA_INTERNAL_PORT, timeout=1.0):
        _wait_slots_idle_briefly(old_key, max_wait_s=10.0)

    # 停止旧版 SSE 网关与 llama-server；非双栖模式下同步清理残留 ComfyUI 释放显存
    kill_patterns = ("llm_sse_gateway.py", "llama-server") if COHOST_COMFY else ("ComfyUI/main.py", "main.py.*8188", "llm_sse_gateway.py", "llama-server")
    _pkill_patterns(*kill_patterns)
    time.sleep(1.5)

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ld_library_path

    shown_cmd = " ".join(_mask_key(x) if x == api_key else x for x in cmd)
    print(f"启动命令: {shown_cmd}", flush=True)

    log_f = open(LLAMA_LOG, "wb")
    try:
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    finally:
        log_f.close()

    print(f"已后台拉起 llama-server (PID={proc.pid}, 127.0.0.1:{LLAMA_INTERNAL_PORT})，日志: {LLAMA_LOG}", flush=True)

    t0 = time.monotonic()
    last_beat = 0.0
    llm_ready = False
    while time.monotonic() - t0 < 480.0:
        if _probe_llama_health(timeout=3.0):
            elapsed = time.monotonic() - t0
            gpu_name, used_g, tot_g = _query_steady_vram()
            print(
                f"llama-server 已就绪（耗时 {elapsed:.1f} 秒）| GPU: {gpu_name} | VRAM: {used_g}/{tot_g} GiB",
                flush=True,
            )
            llm_ready = True
            break

        if proc.poll() is not None:
            tail = LLAMA_LOG.read_text(encoding="utf-8", errors="ignore")[-2500:] if LLAMA_LOG.is_file() else ""
            if api_key in tail:
                tail = tail.replace(api_key, _mask_key(api_key))
            raise RuntimeError(f"llama-server 进程异常退出 (rc={proc.returncode})，末尾日志:\n{tail}")

        now = time.monotonic()
        if now - last_beat >= 10.0:
            print(f"  正在加载 {MODEL_ALIAS} 权重与分配 KV 缓存... (已等待 {now - t0:.0f}s)", flush=True)
            last_beat = now
        time.sleep(2.0)

    if not llm_ready:
        tail = LLAMA_LOG.read_text(encoding="utf-8", errors="ignore")[-2500:] if LLAMA_LOG.is_file() else ""
        if api_key in tail:
            tail = tail.replace(api_key, _mask_key(api_key))
        raise TimeoutError(f"等待 llama-server :{LLAMA_INTERNAL_PORT}/health 就绪超时（480s），末尾日志:\n{tail}")

    # 写入并守护启动本地 SSE 保活与思考强度归一化网关 (:18080 -> :18081)
    SSE_GATEWAY_FILE.write_text(gw_src + "\n", encoding="utf-8")
    os.chmod(SSE_GATEWAY_FILE, 0o700)
    gw_env = os.environ.copy()
    gw_env["SSE_BASE_MODEL"] = MODEL_ALIAS
    gw_env["SSE_DEFAULT_EFFORT"] = DEFAULT_EFFORT
    gw_env["SSE_API_KEY_FILE"] = str(LLAMA_API_KEY_FILE)
    gw_env["SSE_REQUIRE_AUTH"] = "1"
    gw_log_f = open(SSE_GATEWAY_LOG, "wb")
    try:
        gw_proc = subprocess.Popen(
            [sys.executable, str(SSE_GATEWAY_FILE)],
            env=gw_env,
            stdin=subprocess.DEVNULL,
            stdout=gw_log_f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    finally:
        gw_log_f.close()
    print(
        f"已后台拉起 SSE 保活网关 (PID={gw_proc.pid}, 127.0.0.1:{LLAMA_PORT} -> 127.0.0.1:{LLAMA_INTERNAL_PORT})，"
        f"日志: {SSE_GATEWAY_LOG}",
        flush=True,
    )

    gw_t0 = time.monotonic()
    while time.monotonic() - gw_t0 < 15.0:
        if _probe_http_health(LLAMA_PORT, timeout=2.0):
            print(f"SSE 保活网关 :{LLAMA_PORT}/health 探测通过", flush=True)
            auth_ok, auth_reason = _verify_llm_gateway_auth_enforced(LLAMA_PORT, api_key)
            if not auth_ok:
                print(f"[安全拦截] {auth_reason}", flush=True)
                _cleanup_llm_processes(gw_proc, proc)
                raise RuntimeError(f"LLM 网关 API Key 鉴权未生效，拒绝暴露公网隧道: {auth_reason}")
            return gpu_name, used_g, tot_g
        if gw_proc.poll() is not None:
            gw_tail = SSE_GATEWAY_LOG.read_text(encoding="utf-8", errors="ignore")[-1500:] if SSE_GATEWAY_LOG.is_file() else ""
            raise RuntimeError(f"SSE 保活网关异常退出 (rc={gw_proc.returncode})，末尾日志:\n{gw_tail}")
        time.sleep(0.5)

    gw_tail = SSE_GATEWAY_LOG.read_text(encoding="utf-8", errors="ignore")[-1500:] if SSE_GATEWAY_LOG.is_file() else ""
    raise TimeoutError(f"等待 SSE 保活网关 :{LLAMA_PORT}/health 就绪超时（15s），末尾日志:\n{gw_tail}")


def _probe_local_comfy_ready(timeout: float = 2.0) -> bool:
    """探测本机 ComfyUI (:8188/system_stats) 是否正在运行（复用公共实现）。"""
    return probe_comfy_ready(timeout=timeout) is not None


def start_tunnel_and_write_contract(
    active_quant: str,
    api_key: str,
    gpu_name: str,
    vram_used_gb: float,
    vram_total_gb: float,
    n_parallel: int = 2,
) -> str:
    """确保 API Key 鉴权已生效后再打通（或按双栖阶段配置延后）隧道并写入 0600 服务契约。"""
    print("=== [7/7] 打通 Cloudflared 隧道并写入服务契约 ===", flush=True)
    auth_ok, auth_reason = _verify_llm_gateway_auth_enforced(LLAMA_PORT, api_key)
    if not auth_ok:
        print(f"[安全拦截] {auth_reason}", flush=True)
        _cleanup_llm_processes()
        raise RuntimeError(f"LLM 网关 API Key 鉴权未生效，拒绝暴露公网隧道: {auth_reason}")

    defer_tunnel = _env_flag("LLM_DEFER_TUNNEL", False)
    tunnel_url = ""
    if not defer_tunnel:
        tunnel_url = start_cloudflared_tunnel(
            local_port=LLAMA_PORT,
            service_kind="llm",
            cf_token=CF_TUNNEL_TOKEN,
            cf_url=CF_TUNNEL_URL,
            cohost_other=COHOST_COMFY,
            tunnel_url_file=(LLM_TUNNEL_URL_FILE if COHOST_COMFY else TUNNEL_URL_FILE),
            log_file=(CLOUDFLARED_LLM_LOG if (COHOST_COMFY and not (CF_TUNNEL_TOKEN and CF_TUNNEL_URL)) else CLOUDFLARED_LOG),
            tunnel_re=TUNNEL_URL_RE,
            auth_key=api_key,
        )
        LLM_TUNNEL_URL_FILE.write_text(tunnel_url + "\n", encoding="utf-8")
        if not COHOST_COMFY:
            TUNNEL_URL_FILE.write_text(tunnel_url + "\n", encoding="utf-8")

    comfy_alive = False if defer_tunnel else _probe_local_comfy_ready(timeout=2.0)
    is_dual = bool(COHOST_COMFY or comfy_alive)
    old_meta = _read_existing_meta()
    comfy_url_val = CF_COMFY_TUNNEL_URL or ""
    if not comfy_url_val and comfy_alive and TUNNEL_URL_FILE.is_file():
        m_comfy = TUNNEL_URL_RE.search(TUNNEL_URL_FILE.read_text(encoding="utf-8", errors="ignore"))
        if m_comfy:
            comfy_url_val = m_comfy.group(0)
    if not comfy_url_val and is_dual and not defer_tunnel:
        old_comfy_url = str(old_meta.get("comfy_tunnel_url") or "").strip().rstrip("/")
        if not old_comfy_url and old_meta.get("kind") in ("comfy", "dual"):
            old_comfy_url = str(old_meta.get("tunnel_url") or "").strip().rstrip("/")
        comfy_url_val = old_comfy_url

    prof_id = "qwen_27b" if MODEL_VARIANT == "27b" else "qwen_llm"
    existing_comfy_profs = old_meta.get("comfy_profiles") if isinstance(old_meta.get("comfy_profiles"), list) else []
    existing_comfy_key = str(old_meta.get("comfy_api_key") or "").strip()

    meta = {
        "kind": "dual" if is_dual else "llm",
        "profile": prof_id,
        "llm_profile": prof_id,
        "model_variant": MODEL_VARIANT,
        "engine": "llama-server",
        "model": MODEL_ALIAS,
        "port": LLAMA_PORT,
        "llm_port": LLAMA_PORT,
        "internal_port": LLAMA_INTERNAL_PORT,
        "sse_gateway_version": SSE_GATEWAY_VERSION,
        "sse_keepalive": True,
        "kv_unified": True,
        "n_parallel": n_parallel,
        "default_effort": DEFAULT_EFFORT,
        "tunnel_url": (comfy_url_val or tunnel_url) if is_dual else tunnel_url,
        "llm_tunnel_url": tunnel_url,
        "api_key": api_key,
        "webui_enabled": WITH_WEBUI,
        "mmproj_enabled": WITH_MMPROJ,
        "ctx_size": CTX_SIZE,
        "quant": active_quant,
        "gpu_name": gpu_name,
        "vram_used_gb": vram_used_gb,
        "vram_total_gb": vram_total_gb,
    }
    if is_dual:
        meta["comfy_port"] = 8188
        if comfy_url_val and not defer_tunnel:
            meta["comfy_tunnel_url"] = comfy_url_val
        if existing_comfy_key:
            meta["comfy_api_key"] = existing_comfy_key
        if existing_comfy_profs:
            meta["comfy_profiles"] = existing_comfy_profs
            meta["profiles"] = [*existing_comfy_profs, prof_id]
    _write_private_meta(SERVICE_META_FILE, meta)

    if defer_tunnel:
        print("[双栖分阶段启动] LLM 本地鉴权已就绪，延后至 ComfyUI 就绪后统一发布公网隧道", flush=True)
        return ""

    print("========================================", flush=True)
    print(f"SERVICE_TUNNEL_URL: {tunnel_url}", flush=True)
    print(f"LLM_TUNNEL_URL: {tunnel_url}", flush=True)
    if not COHOST_COMFY:
        print(f"COMFYUI_TUNNEL_URL: {tunnel_url}", flush=True)
    print("========================================", flush=True)
    return tunnel_url


def write_runtime_versions_record(
    active_quant: str,
    gpu_name: str,
    vram_total_gb: float,
) -> None:
    """将当前 LLM 推理引擎与硬件/工具版本落盘至 0600 清单。"""
    _write_versions_common(
        "qwen_27b" if MODEL_VARIANT == "27b" else "qwen_llm",
        {
            "variant": MODEL_VARIANT,
            "engine": "llama-server",
            "llama_cpp_tag": LLAMA_CPP_TAG,
            "quant": active_quant,
            "ctx_size": CTX_SIZE,
            "gpu": gpu_name,
            "vram_total_gb": vram_total_gb,
        },
    )


def main():
    """串联执行 Qwen3.8 无审查 LLM 云端一键自举全流程（支持 180B-Flash 与 27B 稠密双版本）。"""
    active_quant, vram_gib, sm_num = check_environment()
    ensure_tools()
    server_bin, ld_library_path = ensure_llama_server(sm_num)
    first_shard, mmproj_path = download_qwen_models(active_quant)
    n_parallel = _calc_n_parallel(vram_gib)
    api_key, cmd = build_auth_and_args(server_bin, first_shard, mmproj_path, active_quant, vram_gib)
    gpu_name, vram_used_gb, vram_total_gb = start_llama_and_wait(
        cmd, ld_library_path, active_quant, api_key, n_parallel=n_parallel
    )
    start_tunnel_and_write_contract(
        active_quant, api_key, gpu_name, vram_used_gb, vram_total_gb, n_parallel=n_parallel
    )
    write_runtime_versions_record(active_quant, gpu_name, vram_total_gb)


if __name__ == "__main__":
    main()
