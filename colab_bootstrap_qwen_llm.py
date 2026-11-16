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
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

HF_REPO_180B = "huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-v2-GGUF"
HF_REPO_27B = "huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF"
LLAMA_CPP_TAG = "b12415"
LLAMA_DIR = Path("/opt/llama_cpp")

LLAMA_PORT = 18080
LLAMA_INTERNAL_PORT = 18081
LLAMA_API = f"http://127.0.0.1:{LLAMA_PORT}"
LLAMA_INTERNAL_API = f"http://127.0.0.1:{LLAMA_INTERNAL_PORT}"

LLAMA_LOG = Path("/root/llama_server.log")
SSE_GATEWAY_FILE = Path("/root/llm_sse_gateway.py")
SSE_GATEWAY_LOG = Path("/root/llm_sse_gateway.log")
CLOUDFLARED_BIN = Path("/usr/local/bin/cloudflared")
CLOUDFLARED_LOG = Path("/root/cloudflared.log")
CLOUDFLARED_LLM_LOG = Path("/root/cloudflared_llm.log")
TUNNEL_URL_FILE = Path("/root/comfy_tunnel_url.txt")
LLM_TUNNEL_URL_FILE = Path("/root/llm_tunnel_url.txt")
SERVICE_META_FILE = Path("/root/colab_service_meta.json")
RCLONE_CONF = Path("/tmp/rclone.conf")

GDRIVE_BIN_REMOTE = "gdrive_save:Colab_LLM_Binaries"

TUNNEL_URL_RE = re.compile(r"https://(?!api\.)[a-z0-9-]+\.trycloudflare\.com")

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


def _env_flag(name: str, default: bool) -> bool:
    """读取布尔环境变量。"""
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() not in ("0", "false", "no", "off", "")


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


def _mask_key(key: str) -> str:
    """生成安全脱敏密钥摘要，严禁在部署日志中打印完整明文密钥。"""
    if not key:
        return "(empty)"
    if key.startswith("sk-pet-"):
        return "sk-pet-****"
    if len(key) <= 8:
        return "****"
    return f"{key[:4]}****"


def run_cmd(cmd, check: bool = True, cwd=None, capture: bool = False, env=None, redact_secret: str = ""):
    """在非交互标准输入模式下流式执行子进程命令，防止 colab exec 挂起并支持敏感词脱敏。"""
    cmd_list = [str(x) for x in cmd]
    shown = [(_mask_key(x) if redact_secret and x == redact_secret else x) for x in cmd_list]
    print(f"[执行] {' '.join(shown)}", flush=True)
    if capture:
        return subprocess.run(
            cmd_list,
            cwd=str(cwd) if cwd else None,
            check=check,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
    proc = subprocess.Popen(
        cmd_list,
        cwd=str(cwd) if cwd else None,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=env,
    )
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            if redact_secret and redact_secret in line:
                line = line.replace(redact_secret, _mask_key(redact_secret))
            print(line, end="", flush=True)
    finally:
        proc.stdout.close()
    rc = proc.wait()
    if check and rc != 0:
        raise subprocess.CalledProcessError(rc, shown)
    return subprocess.CompletedProcess(shown, rc)


def _download_with_urllib(url: str, dst: Path, retries: int = 3):
    """使用标准库 urllib 流式分块下载文件并原子替换。"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_suffix(dst.suffix + ".part")
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                total = int(resp.headers.get("Content-Length") or 0)
                done = 0
                last_log = 0.0
                with open(part, "wb") as f:
                    while True:
                        chunk = resp.read(8 * 1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
                        done += len(chunk)
                        now = time.monotonic()
                        if total > 0 and now - last_log >= 5.0:
                            pct = done * 100.0 / total
                            print(
                                f"  [urllib] {dst.name}: {done / (1024**3):.2f}/{total / (1024**3):.2f} GiB ({pct:.1f}%)",
                                flush=True,
                            )
                            last_log = now
            part.replace(dst)
            return
        except Exception as exc:
            last_err = exc
            print(f"[下载重试] {dst.name} 第 {attempt}/{retries} 次失败: {exc}", flush=True)
            time.sleep(3 * attempt)
    raise RuntimeError(f"下载失败: {url} ({last_err})")


def _safe_extract_tar(tar_path: Path, dest_dir: Path):
    """安全解压 tar.gz 归档至指定目录，防御路径穿越并保留可执行权限与同级符号链接。"""
    dest_root = dest_dir.resolve()
    dest_root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tf:
        for member in tf.getmembers():
            norm_parts = PurePosixPath(member.name).parts
            if PurePosixPath(member.name).is_absolute() or ".." in norm_parts:
                raise RuntimeError(f"归档中存在越界路径，拒绝解压: {member.name}")
            if member.issym() or member.islnk():
                link_p = PurePosixPath(member.linkname)
                if link_p.is_absolute() or ".." in link_p.parts:
                    raise RuntimeError(f"归档符号链接存在越界目标，拒绝解压: {member.linkname}")
        if sys.version_info >= (3, 12):
            tf.extractall(dest_root, filter="tar")
        else:
            tf.extractall(dest_root)


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
    except Exception:
        pass

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


def ensure_tools():
    """确保 rclone、aria2c、cloudflared 基础传输与隧道工具已安装就绪。"""
    print("=== [2/7] 安装基础传输与隧道工具 ===", flush=True)
    rclone_bin = Path("/usr/local/bin/rclone")
    if not shutil.which("rclone"):
        subprocess.run(
            ["bash", "-lc", "apt-get update -qq && apt-get install -y -qq rclone aria2"],
            check=False,
            stdin=subprocess.DEVNULL,
        )
        if not shutil.which("rclone") and not rclone_bin.is_file():
            zip_path = Path("/tmp/rclone-linux-amd64.zip")
            _download_with_urllib("https://downloads.rclone.org/rclone-current-linux-amd64.zip", zip_path)
            with zipfile.ZipFile(zip_path, "r") as zf:
                members = [m for m in zf.namelist() if m.endswith("/rclone") and m.count("/") == 1]
                with zf.open(members[0]) as src, open(rclone_bin, "wb") as dst:
                    shutil.copyfileobj(src, dst)
            os.chmod(rclone_bin, 0o755)
            zip_path.unlink(missing_ok=True)
    if not shutil.which("aria2c"):
        subprocess.run(
            ["bash", "-lc", "apt-get update -qq && apt-get install -y -qq aria2"],
            check=False,
            stdin=subprocess.DEVNULL,
        )
    if not shutil.which("cloudflared") and not CLOUDFLARED_BIN.exists():
        cf_tmp = CLOUDFLARED_BIN.with_suffix(".part")
        _download_with_urllib(
            "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64",
            cf_tmp,
        )
        cf_tmp.replace(CLOUDFLARED_BIN)
        os.chmod(CLOUDFLARED_BIN, 0o755)
    print(
        f"工具就绪: rclone={shutil.which('rclone') or rclone_bin}, aria2c={shutil.which('aria2c')}, cloudflared={shutil.which('cloudflared') or CLOUDFLARED_BIN}",
        flush=True,
    )


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
    loaded_from_gdrive = False

    # 1) 优先尝试从 Google Drive 拉取同架构预归档二进制包
    if has_rclone_conf:
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
        if pull_res.returncode == 0 and local_cache_tar.is_file() and local_cache_tar.stat().st_size > 100_000:
            shutil.rmtree(LLAMA_DIR, ignore_errors=True)
            LLAMA_DIR.mkdir(parents=True, exist_ok=True)
            try:
                _safe_extract_tar(local_cache_tar, LLAMA_DIR)
                verified = _verify_llama_cuda(LLAMA_DIR)
                if verified is not None:
                    loaded_from_gdrive = True
                    local_cache_tar.unlink(missing_ok=True)
                    print(f"已命中云盘引擎缓存并校验通过: {cache_tar_name}", flush=True)
                    return verified
            except Exception as exc:
                print(f"[提示] 云盘缓存包解压或校验异常，继续拉取构建 ({exc})", flush=True)
            finally:
                local_cache_tar.unlink(missing_ok=True)

    # 2) 未命中缓存时尝试拉取官方 GitHub Release Ubuntu CUDA 12.8 预编译包
    verified: tuple[Path, str] | None = None
    shutil.rmtree(LLAMA_DIR, ignore_errors=True)
    LLAMA_DIR.mkdir(parents=True, exist_ok=True)
    base_rel = f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_CPP_TAG}"
    main_tar = Path(f"/tmp/llama-{LLAMA_CPP_TAG}-cuda.tar.gz")
    cudart_tar = Path(f"/tmp/cudart-llama-{LLAMA_CPP_TAG}-cuda.tar.gz")

    print(f"尝试拉取官方 CUDA 预编译包 ({LLAMA_CPP_TAG})...", flush=True)
    ok_main = _fast_pull_tar(
        f"{base_rel}/llama-{LLAMA_CPP_TAG}-bin-ubuntu-cuda-12.8-x64.tar.gz",
        main_tar,
    )
    if ok_main:
        ok_rt = _fast_pull_tar(
            f"{base_rel}/cudart-llama-{LLAMA_CPP_TAG}-bin-ubuntu-cuda-12.8-x64.tar.gz",
            cudart_tar,
        )
        try:
            _safe_extract_tar(main_tar, LLAMA_DIR)
            if ok_rt and cudart_tar.is_file():
                _safe_extract_tar(cudart_tar, LLAMA_DIR)
            verified = _verify_llama_cuda(LLAMA_DIR)
        except Exception as exc:
            print(f"[提示] 预编译包解压或验证未通过，准备回退源码编译: {exc}", flush=True)
            verified = None
    main_tar.unlink(missing_ok=True)
    cudart_tar.unlink(missing_ok=True)

    # 3) 若预编译包不可用或未识别到 CUDA0，自动回退 Ninja 单架构源码编译
    if verified is None:
        print(
            f"预编译包未就绪或不适配当前架构，启动单架构 (sm_{sm_num}) 极速源码编译...",
            flush=True,
        )
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
            "cmake",
            "-B", "build",
            "-G", "Ninja",
            "-DGGML_CUDA=ON",
            f"-DCMAKE_CUDA_ARCHITECTURES={sm_num}",
            "-DLLAMA_CURL=OFF",
            "-DCMAKE_BUILD_TYPE=Release",
        ]
        # CUDA 13 nvcc 若遇系统默认 gcc >= 15 可能报 char8_t 兼容问题，若存在 gcc-14 则优先选用
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

        run_cmd(cmake_cmd, cwd=src_dir, env=build_env)
        nproc = str(max(2, os.cpu_count() or 4))
        run_cmd(
            ["cmake", "--build", "build", "--target", "llama-server", "-j", nproc],
            cwd=src_dir,
            env=build_env,
        )

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

        shutil.rmtree(src_dir, ignore_errors=True)
        verified = _verify_llama_cuda(LLAMA_DIR)
        if verified is None:
            raise RuntimeError("llama-server 源码编译完成，但 --list-devices 仍未识别到 CUDA0！")

    # 4) 若云盘尚无该架构缓存，打包上传至 Google Drive 供后续实例秒拉复用
    if has_rclone_conf and not loaded_from_gdrive:
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

    print(f"推理引擎依赖安装完成: {verified[0]}", flush=True)
    return verified


def _download_model_file(rel_path: str, dst: Path, min_bytes: int, quant: str):
    """下载单个 GGUF 分片或 mmproj（支持可选云盘缓存与 aria2c 16 线程直拉）。"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_file() and dst.stat().st_size >= min_bytes:
        print(f"  [已就绪] {dst.name} ({dst.stat().st_size / (1024**3):.2f} GiB)", flush=True)
        return

    url = f"https://huggingface.co/{HF_REPO}/resolve/main/{rel_path}"
    print(f"  [开始下载] {dst.name} <- {url}", flush=True)
    t0 = time.monotonic()

    rclone_bin = shutil.which("rclone") or "/usr/local/bin/rclone"
    can_gdrive = WITH_GDRIVE_CACHE and RCLONE_CONF.is_file() and Path(rclone_bin).is_file()
    gdrive_uri = f"{GDRIVE_MODEL_REMOTE}/{quant}/{dst.name}"
    pulled_from_gdrive = False

    if can_gdrive:
        res_gd = run_cmd(
            [
                rclone_bin,
                "copyto",
                "--config", str(RCLONE_CONF),
                "--disable-http2",
                "--stats", "2s",
                "--stats-one-line",
                "-v",
                gdrive_uri,
                str(dst),
            ],
            check=False,
        )
        pulled_from_gdrive = res_gd.returncode == 0 and dst.is_file() and dst.stat().st_size >= min_bytes

    if not pulled_from_gdrive:
        used_aria = False
        if shutil.which("aria2c"):
            res_aria = run_cmd(
                [
                    "aria2c",
                    "-c",
                    "-x", "16",
                    "-s", "16",
                    "-k", "1M",
                    "--summary-interval=2",
                    "--console-log-level=warn",
                    "--show-console-readout=false",
                    "--allow-overwrite=true",
                    "-d", str(dst.parent),
                    "-o", dst.name,
                    url,
                ],
                check=False,
            )
            used_aria = res_aria.returncode == 0 and dst.is_file() and dst.stat().st_size >= min_bytes

        if not used_aria:
            _download_with_urllib(url, dst)

    if not dst.is_file() or dst.stat().st_size < min_bytes:
        raise RuntimeError(
            f"模型分片校验失败（大小不足或缺失）: {dst} "
            f"(实际={dst.stat().st_size if dst.is_file() else 0} B, 最低要求={min_bytes} B)"
        )

    dt = max(0.1, time.monotonic() - t0)
    gib = dst.stat().st_size / (1024 ** 3)
    mib_s = dst.stat().st_size / (1024 ** 2) / dt
    print(f"  [完成] {dst.name}: {gib:.2f} GiB ({dt:.1f}s, {mib_s:.1f} MiB/s)", flush=True)

    if can_gdrive and not pulled_from_gdrive:
        print(f"  [云盘归档] 正在同步模型缓存至 {gdrive_uri} ...", flush=True)
        run_cmd(
            [
                rclone_bin,
                "copyto",
                "--config", str(RCLONE_CONF),
                "--disable-http2",
                str(dst),
                gdrive_uri,
            ],
            check=False,
        )


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
    if not SERVICE_META_FILE.is_file():
        return {}
    try:
        data = json.loads(SERVICE_META_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


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
        # 双栖共驻模式下控制 llama-server Prompt RAM 缓存至 8 GiB，给 ComfyUI 模型动态调度留足内存
        cache_ram_mib = 8192
    elif total_ram_mib >= 131072:
        cache_ram_mib = 49152
    elif total_ram_mib >= 65536:
        cache_ram_mib = 16384
    else:
        cache_ram_mib = 8192
    threads_batch = max(4, min(48, os.cpu_count() or 16))

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
        "--api-key", api_key,
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


_SSE_GATEWAY_SCRIPT = r'''#!/usr/bin/env python3
"""本地零依赖 SSE 保活反向代理网关（127.0.0.1:18080 -> 127.0.0.1:18081）。

针对流式推理 POST 请求（stream=true），在 llama-server 长上下文 Prefill 期间，
按 SSE_HEARTBEAT_SEC 间隔向下游发送合法 SSE 注释帧 `: keepalive\n\n`（Chunked 编码），
防止 Cloudflare 代理因约 100 秒无首字节响应而触发超时断流；
其余请求（/health、/v1/models、非流式、401 鉴权错误等）保持零修改原样透传。
"""
import asyncio
import gzip
import json
import os
import sys

LISTEN_HOST = os.environ.get("SSE_GW_HOST", "127.0.0.1")
LISTEN_PORT = int(os.environ.get("SSE_GW_PORT", "18080"))
UPSTREAM_HOST = os.environ.get("SSE_UPSTREAM_HOST", "127.0.0.1")
UPSTREAM_PORT = int(os.environ.get("SSE_UPSTREAM_PORT", "18081"))
HEARTBEAT_SEC = float(os.environ.get("SSE_HEARTBEAT_SEC", "12.0"))
BASE_MODEL = (os.environ.get("SSE_BASE_MODEL") or "Qwen3.8-Flash-Next-abliterated").strip() or "Qwen3.8-Flash-Next-abliterated"
DEFAULT_EFFORT = (os.environ.get("SSE_DEFAULT_EFFORT") or "medium").strip().lower()
if DEFAULT_EFFORT not in {"none", "low", "medium", "high", "xhigh"}:
    DEFAULT_EFFORT = "medium"
MAX_STREAM_INSPECT_BYTES = 128 * 1024 * 1024

STREAM_PATHS = frozenset({
    "/v1/chat/completions",
    "/chat/completions",
    "/v1/completions",
    "/completions",
    "/v1/messages",
    "/messages",
})
MODELS_PATHS = frozenset({"/v1/models", "/models"})

_KEEPALIVE_PAYLOAD = b": keepalive\n\n"
_KEEPALIVE_CHUNK = f"{len(_KEEPALIVE_PAYLOAD):x}\r\n".encode("ascii") + _KEEPALIVE_PAYLOAD + b"\r\n"

_SUFFIX_TO_EFFORT = (
    ("-nothink", "none"),
    ("-minimal", "low"),
    ("-medium", "medium"),
    ("-xhigh", "xhigh"),
    ("-none", "none"),
    ("-fast", "low"),
    ("-off", "none"),
    ("-high", "xhigh"),
    ("-max", "xhigh"),
    ("-low", "low"),
)
_SUPPORTED_PARAMS = ["reasoning_effort", "thinking", "enable_thinking", "tools"]
_ALL_REASONING_LEVELS = ["none", "low", "medium", "high", "xhigh"]


def _map_effort_str(val: object) -> str | None:
    """将各类协议的思考强度字符串安全映射为 Jinja 模板合法集合（none/low/medium/xhigh）。"""
    if not isinstance(val, str):
        return None
    s = val.strip().lower()
    if s in {"none", "off", "disable", "disabled", "false", "0", "nothink"}:
        return "none"
    if s in {"minimal", "low", "fast"}:
        return "low"
    if s in {"medium", "med", "balanced", "default"}:
        return "medium"
    if s in {"high", "xhigh", "max", "heavy"}:
        return "xhigh"
    return None


def _augment_models_catalog(obj: dict) -> dict:
    """为 /v1/models 注入 reasoning_levels 与分档虚拟模型供 Magpie 一键识别。"""
    if not isinstance(obj, dict):
        return obj
    data_list = obj.get("data")
    if isinstance(data_list, list) and data_list:
        base_item = None
        for item in data_list:
            if isinstance(item, dict) and item.get("id") == BASE_MODEL:
                base_item = item
                break
        if base_item is None and isinstance(data_list[0], dict):
            base_item = data_list[0]
        if base_item is not None:
            base_id = str(base_item.get("id") or BASE_MODEL)
            base_item["reasoning_levels"] = list(_ALL_REASONING_LEVELS)
            base_item["supported_parameters"] = list(_SUPPORTED_PARAMS)
            existing_ids = {str(it.get("id")) for it in data_list if isinstance(it, dict) and it.get("id")}
            for suf, lvls in (
                ("-nothink", ["none"]),
                ("-low", ["low"]),
                ("-medium", ["medium"]),
                ("-xhigh", ["xhigh"]),
            ):
                vid = f"{base_id}{suf}"
                if vid in existing_ids:
                    continue
                clone = dict(base_item)
                clone["id"] = vid
                if "aliases" in clone:
                    clone["aliases"] = [vid]
                clone["reasoning_levels"] = lvls
                clone["supported_parameters"] = list(_SUPPORTED_PARAMS)
                data_list.append(clone)
                existing_ids.add(vid)

    models_list = obj.get("models")
    if isinstance(models_list, list) and models_list:
        base_m = None
        for item in models_list:
            if isinstance(item, dict) and (item.get("model") == BASE_MODEL or item.get("name") == BASE_MODEL):
                base_m = item
                break
        if base_m is None and isinstance(models_list[0], dict):
            base_m = models_list[0]
        if base_m is not None:
            base_id = str(base_m.get("model") or base_m.get("name") or BASE_MODEL)
            base_m["reasoning_levels"] = list(_ALL_REASONING_LEVELS)
            existing_names = {
                str(it.get("model") or it.get("name"))
                for it in models_list
                if isinstance(it, dict) and (it.get("model") or it.get("name"))
            }
            for suf, lvls in (
                ("-nothink", ["none"]),
                ("-low", ["low"]),
                ("-medium", ["medium"]),
                ("-xhigh", ["xhigh"]),
            ):
                vid = f"{base_id}{suf}"
                if vid in existing_names:
                    continue
                clone = dict(base_m)
                clone["name"] = vid
                clone["model"] = vid
                clone["reasoning_levels"] = lvls
                models_list.append(clone)
                existing_names.add(vid)
    return obj


def _normalize_inference_payload(payload: dict, path: str) -> dict:
    """解析虚拟模型后缀与多协议思考参数，统一规范化为 Jinja 安全参数并防报错。"""
    if not isinstance(payload, dict):
        return payload

    suffix_effort: str | None = None
    model_val = payload.get("model")
    if isinstance(model_val, str):
        stripped_val = model_val.strip()
        lower_val = stripped_val.lower()
        for suf, eff in _SUFFIX_TO_EFFORT:
            if lower_val.endswith(suf):
                suffix_effort = eff
                payload["model"] = BASE_MODEL
                break

    raw_ct_kwargs = payload.get("chat_template_kwargs")
    ct_kwargs = dict(raw_ct_kwargs) if isinstance(raw_ct_kwargs, dict) else {}

    explicit_disable = False
    explicit_enable = False
    for flag_val in (ct_kwargs.get("enable_thinking"), payload.get("enable_thinking")):
        if isinstance(flag_val, bool):
            if flag_val:
                explicit_enable = True
            else:
                explicit_disable = True

    thinking_cfg = payload.get("thinking")
    budget_effort: str | None = None
    if isinstance(thinking_cfg, dict):
        t_type = str(thinking_cfg.get("type") or "").strip().lower()
        if t_type == "disabled":
            explicit_disable = True
        elif t_type == "enabled":
            explicit_enable = True
            bt = thinking_cfg.get("budget_tokens")
            if isinstance(bt, (int, float)):
                if bt <= 0:
                    budget_effort = "none"
                elif bt <= 1024:
                    budget_effort = "low"
                elif bt <= 4096:
                    budget_effort = "medium"
                else:
                    budget_effort = "xhigh"

    reasoning_cfg = payload.get("reasoning")
    output_cfg = payload.get("output_config")
    explicit_level = (
        _map_effort_str(ct_kwargs.get("reasoning_effort"))
        or _map_effort_str(payload.get("reasoning_effort"))
        or (_map_effort_str(reasoning_cfg.get("effort")) if isinstance(reasoning_cfg, dict) else None)
        or (_map_effort_str(output_cfg.get("effort")) if isinstance(output_cfg, dict) else None)
        or budget_effort
    )

    default_mapped = _map_effort_str(DEFAULT_EFFORT) or "medium"
    if explicit_disable or explicit_level == "none":
        resolved = "none"
    elif explicit_level in {"low", "medium", "xhigh"}:
        resolved = explicit_level
    elif explicit_enable:
        if suffix_effort in {"low", "medium", "xhigh"}:
            resolved = suffix_effort
        elif default_mapped in {"low", "medium", "xhigh"}:
            resolved = default_mapped
        else:
            resolved = "medium"
    elif suffix_effort is not None:
        resolved = suffix_effort
    else:
        resolved = default_mapped

    # 清理可能透传进 llama-server Jinja 模板引发异常的顶层非标准字段
    payload.pop("reasoning_effort", None)
    payload.pop("enable_thinking", None)
    payload.pop("reasoning", None)
    payload.pop("output_config", None)

    if resolved == "none":
        ct_kwargs.pop("reasoning_effort", None)
        ct_kwargs["enable_thinking"] = False
        if path in {"/v1/messages", "/messages"}:
            payload["thinking"] = {"type": "disabled"}
        else:
            payload.pop("thinking", None)
    else:
        ct_kwargs["enable_thinking"] = True
        ct_kwargs["reasoning_effort"] = resolved

    payload["chat_template_kwargs"] = ct_kwargs
    return payload


def _parse_headers(raw_head: bytes) -> tuple[str, list[tuple[str, str]], dict[str, str]]:
    """解析 HTTP 起始行与大小写不敏感头字段映射。"""
    text = raw_head[:-4].decode("latin-1")
    lines = text.split("\r\n")
    start_line = lines[0]
    hdrs: list[tuple[str, str]] = []
    hdr_map: dict[str, str] = {}
    for line in lines[1:]:
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        k_s = k.strip()
        v_s = v.strip()
        hdrs.append((k_s, v_s))
        kl = k_s.lower()
        hdr_map[kl] = f"{hdr_map[kl]}, {v_s}" if kl in hdr_map else v_s
    return start_line, hdrs, hdr_map


async def _read_chunked_body(reader: asyncio.StreamReader, limit: int) -> bytes:
    """解码客户端 Chunked 请求体，限制最大缓冲大小防止内存溢出。"""
    buf = bytearray()
    while True:
        size_line = await reader.readline()
        if not size_line:
            raise EOFError("unexpected EOF reading chunk size")
        size_hex = size_line.split(b";", 1)[0].strip()
        chunk_size = int(size_hex, 16)
        if chunk_size == 0:
            while True:
                trailer = await reader.readline()
                if not trailer or trailer in (b"\r\n", b"\n"):
                    break
            break
        if len(buf) + chunk_size > limit:
            raise ValueError("payload_too_large")
        data = await reader.readexactly(chunk_size)
        buf.extend(data)
        crlf = await reader.readexactly(2)
        if crlf != b"\r\n":
            raise ValueError("invalid chunk terminator")
    return bytes(buf)


def _rebuild_headers(
    start_line: str,
    hdrs: list[tuple[str, str]],
    *,
    force_close: bool = True,
    content_length: int | None = None,
    strip_expect: bool = False,
    strip_accept_encoding: bool = False,
    strip_content_encoding: bool = False,
) -> bytes:
    """重组 HTTP 头：统一注入 Connection: close 以保证每次请求均经过网关调度。"""
    drop = {"connection", "keep-alive", "proxy-connection"} if force_close else set()
    if strip_expect:
        drop.add("expect")
    if strip_accept_encoding:
        drop.add("accept-encoding")
    if strip_content_encoding:
        drop.add("content-encoding")
    if content_length is not None:
        drop.update({"transfer-encoding", "content-length"})
    out = [start_line]
    for k, v in hdrs:
        if k.lower() in drop:
            continue
        out.append(f"{k}: {v}")
    if force_close:
        out.append("Connection: close")
    if content_length is not None:
        out.append(f"Content-Length: {content_length}")
    return ("\r\n".join(out) + "\r\n\r\n").encode("latin-1")


async def _pump_stream(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    *,
    as_chunked: bool = False,
    emit_last_chunk: bool = False,
) -> None:
    """将输入流单向转发至输出流，可选封装为 HTTP/1.1 Chunked 帧。"""
    while True:
        data = await reader.read(65536)
        if not data:
            break
        if as_chunked:
            writer.write(f"{len(data):x}\r\n".encode("ascii") + data + b"\r\n")
        else:
            writer.write(data)
        await writer.drain()
    if as_chunked and emit_last_chunk:
        writer.write(b"0\r\n\r\n")
        await writer.drain()


async def _wait_client_disconnect(reader: asyncio.StreamReader) -> None:
    """监听客户端关闭信号（read 返回 EOF 即视为断开）。"""
    while True:
        chunk = await reader.read(1024)
        if not chunk:
            return


async def _send_simple_error(writer: asyncio.StreamWriter, status: int, reason: str, message: str) -> None:
    """向下游返回标准 OpenAI 风格 JSON 错误响应。"""
    body = json.dumps({"error": {"message": message, "type": "gateway_error"}}, ensure_ascii=False).encode("utf-8")
    hdr = (
        f"HTTP/1.1 {status} {reason}\r\n"
        f"Content-Type: application/json; charset=utf-8\r\n"
        f"Content-Length: {len(body)}\r\n"
        f"Connection: close\r\n\r\n"
    ).encode("latin-1")
    writer.write(hdr + body)
    await writer.drain()


async def _close_writer(writer: asyncio.StreamWriter | None) -> None:
    """安全关闭 StreamWriter，忽略已断开连接异常。"""
    if writer is None:
        return
    try:
        if not writer.is_closing():
            writer.close()
        await writer.wait_closed()
    except Exception:
        pass


async def _handle_client(client_reader: asyncio.StreamReader, client_writer: asyncio.StreamWriter) -> None:
    """处理单条入站 TCP HTTP 请求，执行 SSE 保活调度或零修改透传。"""
    up_writer: asyncio.StreamWriter | None = None
    try:
        try:
            raw_req_head = await client_reader.readuntil(b"\r\n\r\n")
        except (asyncio.IncompleteReadError, asyncio.LimitOverrunError, ConnectionError):
            return

        start_line, hdrs, hdr_map = _parse_headers(raw_req_head)
        parts = start_line.split(" ", 2)
        if len(parts) < 2:
            await _send_simple_error(client_writer, 400, "Bad Request", "Malformed HTTP request line")
            return
        method, target = parts[0].upper(), parts[1]
        path = target.split("?", 1)[0]
        is_upgrade = "upgrade" in hdr_map
        is_chunked_req = "chunked" in hdr_map.get("transfer-encoding", "").lower()

        cl_raw = hdr_map.get("content-length")
        content_len: int | None = None
        if cl_raw is not None:
            try:
                content_len = int(cl_raw.strip())
            except ValueError:
                content_len = None

        should_inspect = (not is_upgrade) and (method == "POST") and (path in STREAM_PATHS)
        can_inspect = should_inspect and (
            is_chunked_req or (content_len is not None and 0 < content_len <= MAX_STREAM_INSPECT_BYTES)
        )

        body_bytes: bytes | None = None
        is_stream = False
        was_rewritten = False
        if can_inspect:
            if "100-continue" in hdr_map.get("expect", "").lower():
                client_writer.write(b"HTTP/1.1 100 Continue\r\n\r\n")
                await client_writer.drain()
            try:
                if is_chunked_req:
                    body_bytes = await _read_chunked_body(client_reader, MAX_STREAM_INSPECT_BYTES)
                else:
                    body_bytes = await client_reader.readexactly(content_len or 0)
            except ValueError as exc:
                if str(exc) == "payload_too_large":
                    await _send_simple_error(client_writer, 413, "Payload Too Large", "Request body exceeds 128 MiB limit")
                    return
                await _send_simple_error(client_writer, 400, "Bad Request", "Invalid chunked request body")
                return
            except (EOFError, asyncio.IncompleteReadError):
                return

            try:
                inspect_bytes = body_bytes
                if "gzip" in hdr_map.get("content-encoding", "").lower():
                    inspect_bytes = gzip.decompress(body_bytes)
                payload_obj = json.loads(inspect_bytes.decode("utf-8"))
                if isinstance(payload_obj, dict):
                    is_stream = bool(payload_obj.get("stream"))
                    payload_obj = _normalize_inference_payload(payload_obj, path)
                    body_bytes = json.dumps(payload_obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                    was_rewritten = True
            except Exception:
                is_stream = False

        try:
            up_reader, up_writer = await asyncio.open_connection(UPSTREAM_HOST, UPSTREAM_PORT, limit=1024 * 1024)
        except Exception as exc:
            await _send_simple_error(client_writer, 502, "Bad Gateway", f"Upstream unavailable: {exc}")
            return

        if (not is_upgrade) and (method == "GET") and (path in MODELS_PATHS):
            fwd_head = _rebuild_headers(start_line, hdrs, force_close=True, strip_accept_encoding=True)
            up_writer.write(fwd_head)
            await up_writer.drain()
            try:
                up_hdr_raw = await up_reader.readuntil(b"\r\n\r\n")
            except (asyncio.IncompleteReadError, asyncio.LimitOverrunError, ConnectionError) as exc:
                await _send_simple_error(client_writer, 502, "Bad Gateway", f"Upstream closed early: {exc}")
                return
            up_start, _, up_hdr_map = _parse_headers(up_hdr_raw)
            up_parts = up_start.split(" ", 2)
            try:
                up_status = int(up_parts[1]) if len(up_parts) >= 2 else 502
            except ValueError:
                up_status = 502
            if up_status != 200:
                client_writer.write(up_hdr_raw)
                await client_writer.drain()
                await _pump_stream(up_reader, client_writer)
                return
            if "chunked" in up_hdr_map.get("transfer-encoding", "").lower():
                raw_models_body = await _read_chunked_body(up_reader, 16 * 1024 * 1024)
            elif "content-length" in up_hdr_map:
                try:
                    raw_models_body = await up_reader.readexactly(int(up_hdr_map["content-length"].strip()))
                except Exception:
                    raw_models_body = await up_reader.read()
            else:
                raw_models_body = await up_reader.read()
            if "gzip" in up_hdr_map.get("content-encoding", "").lower():
                raw_models_body = gzip.decompress(raw_models_body)
            try:
                models_obj = json.loads(raw_models_body.decode("utf-8"))
                out_models_bytes = json.dumps(
                    _augment_models_catalog(models_obj),
                    ensure_ascii=False,
                ).encode("utf-8")
            except Exception:
                out_models_bytes = raw_models_body
            resp_hdr = (
                f"HTTP/1.1 200 OK\r\n"
                f"Content-Type: application/json; charset=utf-8\r\n"
                f"Content-Length: {len(out_models_bytes)}\r\n"
                f"Connection: close\r\n\r\n"
            ).encode("latin-1")
            client_writer.write(resp_hdr + out_models_bytes)
            await client_writer.drain()
            return

        if body_bytes is not None:
            fwd_head = _rebuild_headers(
                start_line,
                hdrs,
                force_close=True,
                content_length=len(body_bytes),
                strip_expect=True,
                strip_accept_encoding=is_stream,
                strip_content_encoding=was_rewritten,
            )
            up_writer.write(fwd_head + body_bytes)
            await up_writer.drain()
        elif not is_upgrade:
            fwd_head = _rebuild_headers(start_line, hdrs, force_close=True)
            up_writer.write(fwd_head)
            await up_writer.drain()
        else:
            up_writer.write(raw_req_head)
            await up_writer.drain()

        if not is_stream:
            # 常规请求 / 非流式推理 / Upgrade 连接：双向零改动转发
            if body_bytes is not None:
                u2c = asyncio.create_task(_pump_stream(up_reader, client_writer))
                disc = asyncio.create_task(_wait_client_disconnect(client_reader))
                done, pending = await asyncio.wait({u2c, disc}, return_when=asyncio.FIRST_COMPLETED)
                for t in pending:
                    t.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
                for t in done:
                    if not t.cancelled():
                        exc = t.exception()
                        if exc and not isinstance(exc, ConnectionError):
                            raise exc
                return

            c2u = asyncio.create_task(_pump_stream(client_reader, up_writer))
            u2c = asyncio.create_task(_pump_stream(up_reader, client_writer))
            try:
                done, pending = await asyncio.wait({c2u, u2c}, return_when=asyncio.FIRST_COMPLETED)
                if c2u in done and u2c in pending:
                    if not is_upgrade:
                        try:
                            if up_writer.can_write_eof():
                                up_writer.write_eof()
                        except Exception:
                            pass
                        await u2c
                elif u2c in done and c2u in pending:
                    c2u.cancel()
                    await asyncio.gather(c2u, return_exceptions=True)
            finally:
                for t in (c2u, u2c):
                    if not t.done():
                        t.cancel()
                await asyncio.gather(c2u, u2c, return_exceptions=True)
            return

        # ===== SSE 流式推理保活分支 (stream=true) =====
        hdr_task = asyncio.create_task(up_reader.readuntil(b"\r\n\r\n"))
        disc_task = asyncio.create_task(_wait_client_disconnect(client_reader))
        sent_sse_headers = False
        up_hdr_raw = b""

        try:
            while not hdr_task.done():
                done, _ = await asyncio.wait(
                    {hdr_task, disc_task},
                    timeout=HEARTBEAT_SEC,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if disc_task in done:
                    hdr_task.cancel()
                    await asyncio.gather(hdr_task, return_exceptions=True)
                    return
                if hdr_task in done:
                    break
                if not sent_sse_headers:
                    client_writer.write(
                        b"HTTP/1.1 200 OK\r\n"
                        b"Content-Type: text/event-stream; charset=utf-8\r\n"
                        b"Cache-Control: no-cache, no-transform\r\n"
                        b"Connection: close\r\n"
                        b"Transfer-Encoding: chunked\r\n"
                        b"X-Accel-Buffering: no\r\n"
                        b"\r\n"
                    )
                    sent_sse_headers = True
                client_writer.write(_KEEPALIVE_CHUNK)
                await client_writer.drain()

            try:
                up_hdr_raw = hdr_task.result()
            except (asyncio.IncompleteReadError, asyncio.LimitOverrunError, ConnectionError) as exc:
                if not sent_sse_headers:
                    await _send_simple_error(client_writer, 502, "Bad Gateway", f"Upstream closed early: {exc}")
                else:
                    err_evt = (
                        f"data: {json.dumps({'error': {'message': f'Upstream closed early: {exc}'}}, ensure_ascii=False)}\n\n"
                        f"data: [DONE]\n\n"
                    ).encode("utf-8")
                    client_writer.write(f"{len(err_evt):x}\r\n".encode("ascii") + err_evt + b"\r\n0\r\n\r\n")
                    await client_writer.drain()
                return

            if not sent_sse_headers:
                client_writer.write(up_hdr_raw)
                await client_writer.drain()
                u2c = asyncio.create_task(_pump_stream(up_reader, client_writer))
                done, pending = await asyncio.wait({u2c, disc_task}, return_when=asyncio.FIRST_COMPLETED)
                for t in pending:
                    t.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
                return

            up_start, _, up_hdr_map = _parse_headers(up_hdr_raw)
            up_parts = up_start.split(" ", 2)
            try:
                up_status = int(up_parts[1]) if len(up_parts) >= 2 else 502
            except ValueError:
                up_status = 502

            if up_status != 200:
                err_raw = await up_reader.read(65536)
                err_txt = err_raw.decode("utf-8", errors="ignore").strip() or f"Upstream HTTP {up_status}"
                err_evt = (
                    f"data: {json.dumps({'error': {'status': up_status, 'message': err_txt}}, ensure_ascii=False)}\n\n"
                    f"data: [DONE]\n\n"
                ).encode("utf-8")
                client_writer.write(f"{len(err_evt):x}\r\n".encode("ascii") + err_evt + b"\r\n0\r\n\r\n")
                await client_writer.drain()
                return

            up_is_chunked = "chunked" in up_hdr_map.get("transfer-encoding", "").lower()
            u2c = asyncio.create_task(
                _pump_stream(
                    up_reader,
                    client_writer,
                    as_chunked=(not up_is_chunked),
                    emit_last_chunk=(not up_is_chunked),
                )
            )
            done, pending = await asyncio.wait({u2c, disc_task}, return_when=asyncio.FIRST_COMPLETED)
            for t in pending:
                t.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
        finally:
            for t in (hdr_task, disc_task):
                if not t.done():
                    t.cancel()
            await asyncio.gather(hdr_task, disc_task, return_exceptions=True)
    except ConnectionError:
        pass
    except Exception as exc:
        print(f"[sse-gw] unexpected error: {exc}", file=sys.stderr, flush=True)
    finally:
        await _close_writer(up_writer)
        await _close_writer(client_writer)


async def _amain() -> None:
    server = await asyncio.start_server(
        _handle_client,
        host=LISTEN_HOST,
        port=LISTEN_PORT,
        limit=1024 * 1024,
    )
    print(
        f"[sse-gw] listening on {LISTEN_HOST}:{LISTEN_PORT} -> {UPSTREAM_HOST}:{UPSTREAM_PORT} "
        f"(heartbeat={HEARTBEAT_SEC}s)",
        flush=True,
    )
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    try:
        asyncio.run(_amain())
    except KeyboardInterrupt:
        pass
'''


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


def start_llama_and_wait(
    cmd: list[str],
    ld_library_path: str,
    active_quant: str,
    api_key: str,
    n_parallel: int = 2,
) -> tuple[str, float, float]:
    """守护启动 llama-server (:18081) 与 SSE 保活网关 (:18080)，轮询就绪并采集稳态显存。"""
    print("=== [6/7] 启动 llama-server 与 SSE 保活网关并等待就绪 ===", flush=True)
    old_meta = _read_existing_meta()
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
    )
    if same_cfg and _probe_http_health(LLAMA_INTERNAL_PORT) and _probe_http_health(LLAMA_PORT):
        gpu_name, used_g, tot_g = _query_steady_vram()
        print(
            f"llama-server 与 SSE 网关已在运行且配置一致，直接复用。GPU={gpu_name}, VRAM={used_g}/{tot_g} GiB",
            flush=True,
        )
        return gpu_name, used_g, tot_g

    old_key = str(old_meta.get("api_key") or "").strip() or api_key
    if _probe_http_health(LLAMA_PORT, timeout=1.0) or _probe_http_health(LLAMA_INTERNAL_PORT, timeout=1.0):
        _wait_slots_idle_briefly(old_key, max_wait_s=10.0)

    # 停止旧版 SSE 网关与 llama-server；非双栖模式下同步清理残留 ComfyUI 释放显存
    kill_patterns = ["llm_sse_gateway.py", "llama-server"]
    if not COHOST_COMFY:
        kill_patterns = ["ComfyUI/main.py", "main.py.*8188", *kill_patterns]
    for pattern in kill_patterns:
        subprocess.run(
            ["pkill", "-f", pattern],
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
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
    SSE_GATEWAY_FILE.write_text(_SSE_GATEWAY_SCRIPT.lstrip("\n"), encoding="utf-8")
    os.chmod(SSE_GATEWAY_FILE, 0o700)
    gw_env = os.environ.copy()
    gw_env["SSE_BASE_MODEL"] = MODEL_ALIAS
    gw_env["SSE_DEFAULT_EFFORT"] = DEFAULT_EFFORT
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
            return gpu_name, used_g, tot_g
        if gw_proc.poll() is not None:
            gw_tail = SSE_GATEWAY_LOG.read_text(encoding="utf-8", errors="ignore")[-1500:] if SSE_GATEWAY_LOG.is_file() else ""
            raise RuntimeError(f"SSE 保活网关异常退出 (rc={gw_proc.returncode})，末尾日志:\n{gw_tail}")
        time.sleep(0.5)

    gw_tail = SSE_GATEWAY_LOG.read_text(encoding="utf-8", errors="ignore")[-1500:] if SSE_GATEWAY_LOG.is_file() else ""
    raise TimeoutError(f"等待 SSE 保活网关 :{LLAMA_PORT}/health 就绪超时（15s），末尾日志:\n{gw_tail}")


def _write_private_meta(meta: dict):
    """以 0o600 严格私密权限原子写入 /root/colab_service_meta.json。"""
    tmp_meta = SERVICE_META_FILE.with_suffix(".json.tmp")
    fd = os.open(str(tmp_meta), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
            f.write("\n")
    except Exception:
        tmp_meta.unlink(missing_ok=True)
        raise
    os.chmod(tmp_meta, 0o600)
    tmp_meta.replace(SERVICE_META_FILE)
    os.chmod(SERVICE_META_FILE, 0o600)


def _probe_local_comfy_ready(timeout: float = 2.0) -> bool:
    """探测本机 ComfyUI (:8188/system_stats) 是否正在运行。"""
    try:
        with urllib.request.urlopen("http://127.0.0.1:8188/system_stats", timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def _probe_public_llm_tunnel(url: str, timeout: float = 5.0) -> bool:
    """校验现存公网隧道是否正转发至当前 llama-server (:18080/health)。"""
    try:
        req = urllib.request.Request(f"{url.rstrip('/')}/health", headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode("utf-8", errors="ignore") or "{}")
                return data.get("status") == "ok"
    except Exception:
        return False
    return False


def start_tunnel_and_write_contract(
    active_quant: str,
    api_key: str,
    gpu_name: str,
    vram_used_gb: float,
    vram_total_gb: float,
    n_parallel: int = 2,
) -> str:
    """确保 cloudflared 指向 127.0.0.1:18080（优先命名隧道），落盘 0600 服务契约并输出公网地址。"""
    print("=== [7/7] 打通 Cloudflared 隧道并写入服务契约 ===", flush=True)
    tunnel_url: str | None = None

    pgrep_res = subprocess.run(
        ["pgrep", "-af", "cloudflared tunnel"],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
    )
    proc_lines = pgrep_res.stdout or ""
    use_named_tunnel = bool(CF_TUNNEL_TOKEN and CF_TUNNEL_URL)

    if use_named_tunnel:
        if (
            pgrep_res.returncode == 0
            and " run" in proc_lines
            and _probe_public_llm_tunnel(CF_TUNNEL_URL)
        ):
            tunnel_url = CF_TUNNEL_URL
            print(f"复用运行中的命名隧道: {tunnel_url}", flush=True)

        if not tunnel_url:
            subprocess.run(
                ["pkill", "-f", "cloudflared tunnel"],
                check=False,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            time.sleep(0.5)

            cf_bin = shutil.which("cloudflared") or str(CLOUDFLARED_BIN)
            if not Path(cf_bin).is_file():
                raise FileNotFoundError(f"未找到 cloudflared 可执行文件: {cf_bin}")

            # 仅通过子进程环境变量传递连接器口令，严禁暴露于命令行参数
            cf_env = os.environ.copy()
            cf_env["TUNNEL_TOKEN"] = CF_TUNNEL_TOKEN

            log_f = open(CLOUDFLARED_LOG, "wb")
            try:
                proc = subprocess.Popen(
                    [cf_bin, "tunnel", "--no-autoupdate", "--protocol", "http2", "run"],
                    env=cf_env,
                    stdin=subprocess.DEVNULL,
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    close_fds=True,
                )
            finally:
                log_f.close()
            print(f"已拉起 cloudflared 命名隧道 (PID={proc.pid})", flush=True)

            registered = False
            for _ in range(35):
                if proc.poll() is not None:
                    tail = CLOUDFLARED_LOG.read_text(encoding="utf-8", errors="ignore")[-1500:] if CLOUDFLARED_LOG.is_file() else ""
                    if CF_TUNNEL_TOKEN and CF_TUNNEL_TOKEN in tail:
                        tail = tail.replace(CF_TUNNEL_TOKEN, "[REDACTED]")
                    raise RuntimeError(f"cloudflared 命名隧道异常退出 (rc={proc.returncode})，末尾日志:\n{tail}")
                log_text = CLOUDFLARED_LOG.read_text(encoding="utf-8", errors="ignore") if CLOUDFLARED_LOG.is_file() else ""
                if "Registered tunnel connection" in log_text:
                    registered = True
                    break
                if _probe_public_llm_tunnel(CF_TUNNEL_URL, timeout=2.0):
                    registered = True
                    tunnel_url = CF_TUNNEL_URL
                    break
                time.sleep(1.0)

            if not registered:
                tail = CLOUDFLARED_LOG.read_text(encoding="utf-8", errors="ignore")[-1500:] if CLOUDFLARED_LOG.is_file() else ""
                if CF_TUNNEL_TOKEN and CF_TUNNEL_TOKEN in tail:
                    tail = tail.replace(CF_TUNNEL_TOKEN, "[REDACTED]")
                raise TimeoutError(f"等待 cloudflared 命名隧道注册超时（35s），末尾日志:\n{tail}")

            if not tunnel_url:
                for _ in range(5):
                    if _probe_public_llm_tunnel(CF_TUNNEL_URL, timeout=3.0):
                        break
                    time.sleep(1.5)
                tunnel_url = CF_TUNNEL_URL
    else:
        quick_log = CLOUDFLARED_LLM_LOG if COHOST_COMFY else CLOUDFLARED_LOG
        quick_url_file = LLM_TUNNEL_URL_FILE if COHOST_COMFY else TUNNEL_URL_FILE
        if (
            pgrep_res.returncode == 0
            and f"127.0.0.1:{LLAMA_PORT}" in proc_lines
            and quick_url_file.is_file()
        ):
            cached = quick_url_file.read_text(encoding="utf-8", errors="ignore").strip()
            m_cached = TUNNEL_URL_RE.search(cached)
            if m_cached and _probe_public_llm_tunnel(m_cached.group(0)):
                tunnel_url = m_cached.group(0)
                print(f"复用运行中的 {LLAMA_PORT} 端口 cloudflared 隧道: {tunnel_url}", flush=True)

        if not tunnel_url:
            kill_pat = f"cloudflared tunnel.*{LLAMA_PORT}" if COHOST_COMFY else "cloudflared tunnel"
            subprocess.run(
                ["pkill", "-f", kill_pat],
                check=False,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            time.sleep(0.5)

            cf_bin = shutil.which("cloudflared") or str(CLOUDFLARED_BIN)
            if not Path(cf_bin).is_file():
                raise FileNotFoundError(f"未找到 cloudflared 可执行文件: {cf_bin}")

            log_f = open(quick_log, "wb")
            try:
                proc = subprocess.Popen(
                    [cf_bin, "tunnel", "--no-autoupdate", "--protocol", "http2", "--url", LLAMA_API],
                    stdin=subprocess.DEVNULL,
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    close_fds=True,
                )
            finally:
                log_f.close()
            print(f"已拉起 cloudflared (PID={proc.pid}, port={LLAMA_PORT})", flush=True)

            for _ in range(30):
                if quick_log.is_file():
                    m = TUNNEL_URL_RE.search(quick_log.read_text(encoding="utf-8", errors="ignore"))
                    if m:
                        tunnel_url = m.group(0)
                        break
                time.sleep(1.0)

            if not tunnel_url:
                tail = quick_log.read_text(encoding="utf-8", errors="ignore")[-1500:] if quick_log.is_file() else ""
                raise TimeoutError(f"等待 cloudflared 分配隧道地址超时（30s），末尾日志:\n{tail}")

    LLM_TUNNEL_URL_FILE.write_text(tunnel_url + "\n", encoding="utf-8")
    if not COHOST_COMFY:
        TUNNEL_URL_FILE.write_text(tunnel_url + "\n", encoding="utf-8")

    comfy_alive = _probe_local_comfy_ready(timeout=2.0)
    is_dual = bool(COHOST_COMFY or comfy_alive)
    comfy_url_val = CF_COMFY_TUNNEL_URL or ""
    if not comfy_url_val and comfy_alive and TUNNEL_URL_FILE.is_file():
        m_comfy = TUNNEL_URL_RE.search(TUNNEL_URL_FILE.read_text(encoding="utf-8", errors="ignore"))
        if m_comfy:
            comfy_url_val = m_comfy.group(0)

    prof_id = "qwen_27b" if MODEL_VARIANT == "27b" else "qwen_llm"
    old_meta = _read_existing_meta()
    existing_comfy_profs = old_meta.get("comfy_profiles") if isinstance(old_meta.get("comfy_profiles"), list) else []

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
        if comfy_url_val:
            meta["comfy_tunnel_url"] = comfy_url_val
        if existing_comfy_profs:
            meta["comfy_profiles"] = existing_comfy_profs
            meta["profiles"] = [*existing_comfy_profs, prof_id]
    _write_private_meta(meta)

    print("========================================", flush=True)
    print(f"SERVICE_TUNNEL_URL: {tunnel_url}", flush=True)
    print(f"LLM_TUNNEL_URL: {tunnel_url}", flush=True)
    if not COHOST_COMFY:
        print(f"COMFYUI_TUNNEL_URL: {tunnel_url}", flush=True)
    print("========================================", flush=True)
    return tunnel_url


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


if __name__ == "__main__":
    main()
