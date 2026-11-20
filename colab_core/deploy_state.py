"""colab_core.deploy_state —— 工作流生态规划、显卡防呆推荐、7 步进度流式解析与故障分类模块。"""

from __future__ import annotations

import re
import sys
from pathlib import Path, PurePosixPath
from typing import Any

_PKG_DIR = Path(__file__).resolve().parents[1]
if str(_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR))

from colab_security import redact_secrets  # noqa: E402

SUPPORTED_PROFILES = ("anima", "minimax", "drama3", "qwen_27b", "qwen_llm")
COMFY_PROFILES = ("anima", "minimax", "drama3")
LLM_PROFILES = ("qwen_27b", "qwen_llm")
SUPPORTED_BOOT_GPUS = ("T4", "L4", "A100", "G4", "H100")
SUPPORTED_LLM_QUANTS = ("UD-Q4_K_XL", "UD-IQ4_XS", "IQ3_XXS")

LLM_QUANT_BASE_SHARDS: dict[str, int] = {"UD-Q4_K_XL": 4, "UD-IQ4_XS": 3, "IQ3_XXS": 2}
LLM_27B_BASE_SHARDS = 1
PROFILE_LABELS: dict[str, str] = {
    "anima": "Anima",
    "minimax": "MiniMax",
    "drama3": "Drama 3",
    "qwen_27b": "Qwen27B",
    "qwen_llm": "Qwen180B",
}

STEP_BANNER_RE = re.compile(r"===\s*\[(\d+)/(\d+)\]\s*(.+?)\s*===")
PCT_TOKEN_RE = re.compile(r"\((\d{1,3}(?:\.\d+)?)%\)")
DL_SPEED_RE = re.compile(r"DL:\s*([0-9.]+\s*[KMGTP]?i?B(?:/s)?)", re.IGNORECASE)
DEPLOY_DL_TOTAL_RE = re.compile(r"DEPLOY_DL_TOTAL:\s*(\d+)")
DEPLOY_PHASE_RE = re.compile(r"DEPLOY_PHASE\s+(\d+)/(\d+):\s*(\S+)")
DEFAULT_DEPLOY_TUNNEL_URL_PATTERN = r"https://(?!api\.)[a-z0-9-]+\.trycloudflare\.com(?![a-z0-9.-])"
DEPLOY_TUNNEL_LINE_RE = re.compile(
    rf"(?:COMFYUI_TUNNEL_URL|SERVICE_TUNNEL_URL):\s*({DEFAULT_DEPLOY_TUNNEL_URL_PATTERN})"
)
DEPLOY_LLM_TUNNEL_LINE_RE = re.compile(
    rf"LLM_TUNNEL_URL:\s*({DEFAULT_DEPLOY_TUNNEL_URL_PATTERN})"
)
DEPLOY_STEP_BASE_PCT = {1: 16, 2: 23, 3: 31, 4: 42, 5: 78, 6: 86, 7: 95}

_BRACKET_ONLY_RE = re.compile(r"^[\s()\[\]{},;~^.:]+$")
_GENERIC_EXIT_WRAPPER_RE = re.compile(r"^\[错误\]\s*子进程退出码\s*-?\d+(?:[\s(（]|$)")
_DL_START_RE = re.compile(r"(?:开始下载底座文件:|\[开始下载\])\s*([^\s(]+)")

_COMFY_STEP_HINTS: dict[int, str] = {
    4: "正在连接模型源并校验本地缓存...",
    5: "正在拉取 LoRA 与工作流配置...",
    6: "正在拉起 ComfyUI 服务...",
    7: "正在申请 Cloudflared 公网隧道...",
}


def normalize_profiles(raw_profiles: Any, fallback_profile: str = "anima", *, strict: bool = False) -> list[str]:
    """规范化工作流生态组合并按官方优先级排序，同时处理 qwen_27b 与 qwen_llm 互斥。"""
    if isinstance(raw_profiles, str):
        items: list[Any] = raw_profiles.split(",")
    elif isinstance(raw_profiles, (list, tuple, set)):
        items = list(raw_profiles)
    else:
        items = []

    cand_list: list[str] = []
    for item in items:
        p = str(item or "").strip().lower()
        if not p:
            continue
        if p not in SUPPORTED_PROFILES:
            if strict:
                raise ValueError(f"不支持的工作流生态 {p!r}，可选值: {', '.join(SUPPORTED_PROFILES)}")
            continue
        if p not in cand_list:
            cand_list.append(p)

    if not cand_list:
        fb = str(fallback_profile or "anima").strip().lower()
        if strict and fb not in SUPPORTED_PROFILES:
            raise ValueError(f"不支持的工作流生态 {fb!r}，可选值: {', '.join(SUPPORTED_PROFILES)}")
        cand_list = [fb if fb in SUPPORTED_PROFILES else "anima"]
    if "qwen_27b" in cand_list and "qwen_llm" in cand_list:
        cand_list = [p for p in cand_list if p != "qwen_27b"]
    return [p for p in SUPPORTED_PROFILES if p in cand_list]


def parse_profiles_cli_arg(profile_arg: str | None, profiles_arg: str | None = None) -> list[str]:
    """解析 CLI `--profile` / `--profiles` 参数，非法值严格抛错。"""
    if profiles_arg and str(profiles_arg).strip():
        return normalize_profiles(str(profiles_arg), strict=True)
    if profile_arg and str(profile_arg).strip():
        return normalize_profiles(str(profile_arg), strict=True)
    return ["anima"]


def select_default_gpu(profiles: list[str]) -> str:
    """按生态组合显存需求选择默认推荐显卡。"""
    profs = normalize_profiles(profiles)
    if "qwen_llm" in profs or "minimax" in profs or "drama3" in profs or len(profs) >= 2:
        return "G4"
    return "L4"


def resolve_high_mem(gpu: str, requested_high_mem: bool) -> bool:
    """解析有效高内存标记（L4/G4 规格固定不传 --high-mem）。"""
    return False if str(gpu or "").strip().upper() in ("L4", "G4") else bool(requested_high_mem)


def profile_bootstrap_spec(
    profile: str,
    *,
    pkg_dir: Path = _PKG_DIR,
    anima_script: Path | None = None,
    minimax_script: Path | None = None,
    drama3_script: Path | None = None,
    qwen_llm_script: Path | None = None,
) -> tuple[Path, int]:
    """返回指定生态对应的云端自举脚本路径与执行超时时间（秒）。"""
    p = str(profile or "anima").strip().lower()
    if p in LLM_PROFILES:
        return Path(qwen_llm_script or (pkg_dir / "colab_bootstrap_qwen_llm.py")), 2400
    if p == "minimax":
        return Path(minimax_script or (pkg_dir / "colab_bootstrap_minimax_h3.py")), 2400
    if p == "drama3":
        return Path(drama3_script or (pkg_dir / "colab_bootstrap_drama3.py")), 1800
    return Path(anima_script or (pkg_dir / "colab_bootstrap_anima.py")), 600


def default_deploy_state() -> dict[str, Any]:
    """生成初始 7 步部署进度状态字典。"""
    return {
        "active": False,
        "profile": "anima",
        "gpu": "L4",
        "high_mem": False,
        "step_idx": 0,
        "step_total": 7,
        "stage_title": "",
        "sub_detail": "",
        "pct": 0,
        "started_ts": 0,
        "elapsed_sec": 0,
        "error_msg": "",
        "stockout_gpu": None,
        "suggest_gpu": None,
    }


def format_elapsed_cn(seconds: int) -> str:
    """将耗时秒数格式化为简洁中文时长。"""
    sec = max(0, int(seconds))
    return f"{sec}秒" if sec < 60 else f"{sec // 60}分{sec % 60:02d}秒"


def calc_phase_dl_total(
    phase_prof: str,
    llm_quant: str = "UD-Q4_K_XL",
    llm_mmproj: bool = True,
    llm27_mmproj: bool = True,
    minimax_r2v: bool = True,
    minimax_pe: bool = True,
) -> int:
    """估算指定阶段生态的预期权重下载文件总数。"""
    if phase_prof == "qwen_27b":
        return LLM_27B_BASE_SHARDS + (1 if llm27_mmproj else 0)
    if phase_prof == "qwen_llm":
        return LLM_QUANT_BASE_SHARDS.get(llm_quant, 4) + (1 if llm_mmproj else 0)
    if phase_prof == "minimax":
        return 7 + (1 if minimax_pe else 0) + (2 if minimax_r2v else 0)
    return 3


def scale_phase_pct(local_pct: int, phase_idx: int, phase_total: int) -> int:
    """将单阶段局部百分比映射至多阶段全局进度条区间。"""
    tot = max(1, int(phase_total))
    if tot <= 1:
        return max(0, min(100, int(local_pct)))
    idx = max(1, min(tot, int(phase_idx)))
    lo = 12.0 + (idx - 1) * (86.0 / tot)
    hi = 98.0 if idx >= tot else (12.0 + idx * (86.0 / tot))
    frac = max(0.0, min(1.0, (float(local_pct) - 12.0) / 88.0))
    return int(round(lo + frac * (hi - lo)))


def _compile_deploy_tunnel_regexes(
    tunnel_pattern: str | re.Pattern[str],
) -> tuple[re.Pattern[str], re.Pattern[str]]:
    """根据调用方传入的动态 `tunnel_pattern` 构造主隧道与 LLM 隧道提取正则。"""
    pat_str = (
        tunnel_pattern.pattern
        if isinstance(tunnel_pattern, re.Pattern)
        else str(tunnel_pattern or "").strip()
    )
    if not pat_str:
        return DEPLOY_TUNNEL_LINE_RE, DEPLOY_LLM_TUNNEL_LINE_RE
    return (
        re.compile(rf"(?:COMFYUI_TUNNEL_URL|SERVICE_TUNNEL_URL):\s*({pat_str})"),
        re.compile(rf"LLM_TUNNEL_URL:\s*({pat_str})"),
    )


def _scale_ctx_pct(ctx: dict[str, Any], local_pct: int) -> int:
    """基于阶段上下文将局部步骤进度换算为全局百分比。"""
    return scale_phase_pct(local_pct, int(ctx.get("phase_idx") or 1), int(ctx.get("phase_total") or 1))


def _apply_deploy_phase_banner(
    m_phase: re.Match[str],
    st: dict[str, Any],
    ctx: dict[str, Any],
    prefs: dict[str, Any],
    cur_pct: int,
) -> None:
    """处理 `DEPLOY_PHASE x/y: profile` 阶段切换横幅。"""
    p_idx, p_tot = max(1, int(m_phase.group(1))), max(1, int(m_phase.group(2)))
    p_prof = m_phase.group(3).strip().lower()
    ctx.update({
        "phase_idx": p_idx,
        "phase_total": p_tot,
        "phase_prof": p_prof,
        "phase_is_llm": p_prof in LLM_PROFILES,
        "dl_done": 0,
        "dl_name": "",
        "dl_total": calc_phase_dl_total(
            p_prof,
            str(prefs.get("boot_llm_quant") or "UD-Q4_K_XL"),
            bool(prefs.get("boot_llm_mmproj", True)),
            bool(prefs.get("boot_llm27_mmproj", True)),
            bool(prefs.get("boot_minimax_r2v", True)),
            bool(prefs.get("boot_minimax_pe", True)),
        ),
    })
    st["step_idx"], st["step_total"] = 0, 7
    if p_tot > 1:
        p_name = PROFILE_LABELS.get(p_prof, "Anima")
        st["stage_title"] = f"阶段 {p_idx}/{p_tot} · {p_name}"
        st["sub_detail"] = f"正在启动 {p_name} 云端部署..."
        st["pct"] = max(cur_pct, _scale_ctx_pct(ctx, 14))


def _apply_deploy_tunnel_hit(
    cand_url: str,
    st: dict[str, Any],
    ctx: dict[str, Any],
    cur_pct: int,
    total_steps: int,
    is_llm: bool,
) -> tuple[str | None, bool]:
    """处理主服务或中间阶段的隧道 URL 命中事件。"""
    p_idx, p_tot = int(ctx.get("phase_idx") or 1), int(ctx.get("phase_total") or 1)
    if p_idx < p_tot:
        if is_llm and not ctx.get("captured_llm_tunnel"):
            ctx["captured_llm_tunnel"] = cand_url
        st["step_idx"] = total_steps
        st["pct"] = max(cur_pct, _scale_ctx_pct(ctx, 100))
        st["stage_title"] = f"阶段 {p_idx}/{p_tot} 就绪"
        st["sub_detail"] = "前一服务就绪，继续部署下一工作流..."
        return None, False
    st["step_idx"], st["pct"] = total_steps, 100
    st["stage_title"] = f"[{total_steps}/{total_steps}] 部署全部完成"
    st["sub_detail"] = f"隧道已打通: {cand_url.replace('https://', '')}"
    should_warm = not bool(ctx.get("meta_warmed"))
    ctx["meta_warmed"] = True
    return cand_url, should_warm


def _apply_deploy_step_banner(
    m_step: re.Match[str],
    st: dict[str, Any],
    ctx: dict[str, Any],
    cur_pct: int,
    is_llm: bool,
) -> None:
    """处理 `=== [idx/tot] 标题 ===` 步骤横幅并刷新默认子阶段文案。"""
    p_tot = int(ctx.get("phase_total") or 1)
    idx, tot = int(m_step.group(1)), max(1, int(m_step.group(2)))
    raw_title = m_step.group(3).strip()
    max_len = 14 if p_tot > 1 else 18
    if len(raw_title) > max_len:
        raw_title = raw_title[: max_len - 1] + "…"
    base_pct = DEPLOY_STEP_BASE_PCT.get(idx, min(95, int(round(12 + idx * 84.0 / tot))))
    phase_tag = f"P{ctx.get('phase_idx', 1)}/{p_tot} " if p_tot > 1 else ""
    st.update({
        "step_idx": idx,
        "step_total": tot,
        "stage_title": f"{phase_tag}[{idx}/{tot}] {raw_title}",
        "pct": max(cur_pct, _scale_ctx_pct(ctx, base_pct)),
    })
    if is_llm:
        is_27b = ctx.get("phase_prof") == "qwen_27b"
        llm_hints = {
            3: "正在准备 llama-server CUDA 推理引擎...",
            4: "正在高速拉取 Qwen3.8 GGUF 量化分片...",
            5: "正在生成访问密钥与全卡推理参数..." if is_27b else "正在生成访问密钥与 PLE/MoE 卸载参数...",
            6: "正在加载 27B 模型并启动 llama-server..." if is_27b else "正在加载 180B 模型并启动 llama-server...",
            7: "正在打通 Cloudflared 隧道并写入服务契约...",
        }
        st["sub_detail"] = llm_hints.get(idx, raw_title)
    else:
        st["sub_detail"] = _COMFY_STEP_HINTS.get(idx, raw_title)


def _apply_prestep_session_hint(
    line: str,
    low: str,
    st: dict[str, Any],
    cur_pct: int,
    cur_step: int,
    gpu: str,
) -> bool:
    """匹配步骤开始前的会话申请、复用与 rclone 上传提示；命中返回 True。"""
    if "正在释放旧实例" in line:
        st.update({"pct": max(cur_pct, 3), "stage_title": "清理旧会话", "sub_detail": "正在停卡释放旧算力分配..."})
    elif cur_step == 0 and "[本机执行]" in line and "sessions" in low and cur_pct < 6:
        st.update({"pct": max(cur_pct, 4), "stage_title": f"核验云端会话 ({gpu})", "sub_detail": "正在查询当前账号现存 Colab 会话..."})
    elif "creating session" in low or "开启 colab 实例" in low:
        st.update({"pct": max(cur_pct, 6), "stage_title": f"申请 {gpu} 实例", "sub_detail": f"正在向 Colab 请求分配 {gpu} GPU..."})
    elif "session ready" in low:
        st.update({"pct": max(cur_pct, 11), "stage_title": "实例就绪", "sub_detail": f"{gpu} 已分配，准备上传云盘凭据..."})
    elif "跳过 colab new" in low:
        st.update({"pct": max(cur_pct, 11), "stage_title": "复用活跃实例", "sub_detail": "沿用现存运行时，上传配置并增量部署..."})
    elif "colab.exe upload" in low or ("upload" in low and "rclone.conf" in low):
        st.update({"pct": max(cur_pct, 13), "stage_title": "注入云盘配置", "sub_detail": "正在握手内核并启动自举..."})
    else:
        return False
    return True


def _apply_download_progress_line(
    line: str,
    st: dict[str, Any],
    ctx: dict[str, Any],
    cur_pct: int,
) -> None:
    """解析第 4 步模型权重下载开始、缓存命中/完成计数与实时速率百分比。"""
    m_start = _DL_START_RE.search(line)
    if m_start:
        fname = PurePosixPath(m_start.group(1)).name
        dl_name = fname[:20] + ("…" if len(fname) > 20 else "")
        ctx["dl_name"] = dl_name
        st["sub_detail"] = f"下载 {dl_name}..."
        return
    dl_tot = max(1, int(ctx.get("dl_total") or 3))
    if any(k in line for k in ("已存在且字节数匹配", "校验通过:", "[已就绪]", "[完成]")):
        dl_done = min(dl_tot, int(ctx.get("dl_done") or 0) + 1)
        ctx["dl_done"] = dl_done
        st["pct"] = max(cur_pct, _scale_ctx_pct(ctx, min(76, 42 + int(round((dl_done / dl_tot) * 34)))))
        st["sub_detail"] = line.replace("已存在且字节数匹配，跳过:", "已缓存")[:38]
        return
    m_pct = PCT_TOKEN_RE.search(line)
    if m_pct:
        file_pct = min(100.0, max(0.0, float(m_pct.group(1))))
        dl_done = int(ctx.get("dl_done") or 0)
        ratio = min(1.0, (dl_done + file_pct / 100.0) / dl_tot)
        st["pct"] = max(cur_pct, _scale_ctx_pct(ctx, min(76, 42 + int(round(ratio * 34)))))
        m_spd = DL_SPEED_RE.search(line)
        spd = f" · {m_spd.group(1)}" if m_spd else ""
        prefix = f"{ctx.get('dl_name')} " if ctx.get("dl_name") else "拉取模型 "
        st["sub_detail"] = f"{prefix}({int(round(file_pct))}%{spd})"


def _apply_step3_detail(line: str, low: str, st: dict[str, Any], ctx: dict[str, Any], cur_pct: int, is_llm: bool) -> None:
    """解析第 3 步环境与推理引擎准备阶段的细粒度进度。"""
    if is_llm:
        if "cmake" in low or "编译" in line:
            st.update({"sub_detail": "正在编译 llama-server CUDA 内核...", "pct": max(cur_pct, _scale_ctx_pct(ctx, 36))})
        elif "cuda0" in low or ("llama-server" in low and ("就绪" in line or "完成" in line)):
            st.update({"sub_detail": "llama-server CUDA 推理引擎已就绪", "pct": max(cur_pct, _scale_ctx_pct(ctx, 40))})
        elif any(k in line for k in ("预编译", "下载", "拉取")):
            st.update({"sub_detail": "正在拉取 llama-server CUDA 二进制包...", "pct": max(cur_pct, _scale_ctx_pct(ctx, 34))})
        return
    if "git" in low and "clone" in low:
        st.update({"sub_detail": "正在克隆官方 ComfyUI 仓库...", "pct": max(cur_pct, _scale_ctx_pct(ctx, 33))})
    elif "pip" in low and "install" in low:
        st.update({"sub_detail": "正在安装依赖（保留预装 CUDA）...", "pct": max(cur_pct, _scale_ctx_pct(ctx, 36))})
    elif "依赖安装完成" in line or "comfy_kitchen 已就绪" in line:
        st.update({"sub_detail": "ComfyUI 依赖与 CUDA 环境就绪", "pct": max(cur_pct, _scale_ctx_pct(ctx, 40))})


def _apply_step6_detail(line: str, low: str, st: dict[str, Any], ctx: dict[str, Any], cur_pct: int, is_llm: bool) -> None:
    """解析第 6 步服务拉起与端口就绪探测阶段的细粒度进度。"""
    if is_llm:
        if any(k in line for k in ("已就绪", "模型就绪", "复用")):
            st.update({"sub_detail": "Qwen3.8 推理引擎加载完成", "pct": max(cur_pct, _scale_ctx_pct(ctx, 93))})
        elif any(k in low for k in ("llama-server", "18080", "8080", "health", "加载")):
            st.update({"sub_detail": "等待 llama-server 18080 端口加载就绪...", "pct": max(cur_pct, _scale_ctx_pct(ctx, 89))})
        return
    if "已后台拉起 ComfyUI" in line or "已拉起 ComfyUI" in line:
        st.update({"sub_detail": "等待 ComfyUI 8188 端口响应...", "pct": max(cur_pct, _scale_ctx_pct(ctx, 89))})
    elif "ComfyUI 就绪" in line or "已就绪" in line:
        st.update({"sub_detail": "ComfyUI 引擎启动完成", "pct": max(cur_pct, _scale_ctx_pct(ctx, 93))})


def _apply_step_detail_line(
    line: str,
    low: str,
    st: dict[str, Any],
    ctx: dict[str, Any],
    cur_pct: int,
    cur_step: int,
    is_llm: bool,
) -> None:
    """根据当前步骤号分派单行细粒度部署日志解析。"""
    if cur_step == 1:
        if "自动降级" in line or "降级为" in line:
            st.update({"sub_detail": line[:38], "pct": max(cur_pct, _scale_ctx_pct(ctx, 20))})
        elif "gpu 硬件信息:" in low or line.startswith("GPU:"):
            st.update({"sub_detail": f"已就绪: {line.split(':', 1)[1].strip()[:36]}", "pct": max(cur_pct, _scale_ctx_pct(ctx, 19))})
    elif cur_step == 2:
        if any(k in line for k in ("已就绪", "安装完成", "工具就绪")):
            st.update({"pct": max(cur_pct, _scale_ctx_pct(ctx, 27)), "sub_detail": "传输与隧道基础工具已备齐"})
    elif cur_step == 3:
        _apply_step3_detail(line, low, st, ctx, cur_pct, is_llm)
    elif cur_step == 4:
        _apply_download_progress_line(line, st, ctx, cur_pct)
    elif cur_step == 5:
        if is_llm and any(k in line for k in ("sk-pet-", "鉴权", "参数", "卸载")):
            st.update({"sub_detail": "安全鉴权密钥与推理参数已就绪", "pct": max(cur_pct, _scale_ctx_pct(ctx, 83))})
        elif "LoRA 就绪:" in line or "工作流就绪:" in line:
            st.update({"sub_detail": line[:38], "pct": max(cur_pct, _scale_ctx_pct(ctx, 83))})
    elif cur_step == 6:
        _apply_step6_detail(line, low, st, ctx, cur_pct, is_llm)
    elif cur_step == 7 and "cloudflared" in low and any(k in line for k in ("拉起", "启动", "隧道")):
        st.update({"sub_detail": "等待 Cloudflare 分配公网域名...", "pct": max(cur_pct, _scale_ctx_pct(ctx, 97))})


def consume_deploy_line(
    raw_line: str,
    st: dict[str, Any],
    ctx: dict[str, Any],
    prefs: dict[str, Any],
    *,
    tunnel_pattern: str | re.Pattern[str] = "",
) -> tuple[str | None, bool]:
    """解析单行部署输出并原地推进 `st` 与阶段上下文 `ctx`，返回 `(tunnel_url, should_warm_meta)`。"""
    line = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", str(raw_line or "")).strip()
    if not line:
        return None, False

    tun_re, ltun_re = _compile_deploy_tunnel_regexes(tunnel_pattern)
    cur_pct = int(st.get("pct") or 0)
    cur_step = int(st.get("step_idx") or 0)
    total_steps = max(1, int(st.get("step_total") or 7))
    gpu = str(st.get("gpu") or "L4")
    is_llm = bool(ctx.get("phase_is_llm"))

    if m_phase := DEPLOY_PHASE_RE.search(line):
        _apply_deploy_phase_banner(m_phase, st, ctx, prefs, cur_pct)
        return None, False
    if m_ltun := ltun_re.search(line):
        ctx["captured_llm_tunnel"] = m_ltun.group(1).strip()
        return None, False
    if m_tot := DEPLOY_DL_TOTAL_RE.search(line):
        if int(m_tot.group(1)) > 0:
            ctx["dl_total"] = int(m_tot.group(1))
        return None, False
    if m_tun := tun_re.search(line):
        return _apply_deploy_tunnel_hit(m_tun.group(1).strip(), st, ctx, cur_pct, total_steps, is_llm)
    if m_step := STEP_BANNER_RE.search(line):
        _apply_deploy_step_banner(m_step, st, ctx, cur_pct, is_llm)
        return None, False

    low = line.lower()
    if not _apply_prestep_session_hint(line, low, st, cur_pct, cur_step, gpu):
        _apply_step_detail_line(line, low, st, ctx, cur_pct, cur_step, is_llm)
    return None, False


def recommend_fallback_gpu(
    failed_gpu: str,
    profile: str,
    eligible_gpus: list[str] | None,
    scarce_gpus: dict[str, Any],
    profiles: list[str] | None = None,
) -> str | None:
    """在目标显卡缺货或被拒时按生态显存需求推荐备选可用显卡。"""
    failed_up = str(failed_gpu or "").upper()
    profs = normalize_profiles(profiles, fallback_profile=profile)
    has_comfy = any(p in COMFY_PROFILES for p in profs)
    if "qwen_llm" in profs or ("qwen_27b" in profs and has_comfy):
        order: tuple[str, ...] = ("G4", "A100", "H100")
    elif profs == ["qwen_27b"]:
        order = ("L4", "A100", "G4", "H100", "T4")
    elif any(p in ("minimax", "drama3") for p in profs):
        order = ("G4", "A100", "H100", "L4", "T4")
    else:
        order = ("L4", "A100", "T4", "G4", "H100")

    allowed = {str(g).upper() for g in eligible_gpus if g} if isinstance(eligible_gpus, list) and eligible_gpus else None
    scarce_set = {str(k).upper() for k in (scarce_gpus or {}).keys()}
    for cand in order:
        if cand != failed_up and cand not in scarce_set and (allowed is None or cand in allowed):
            return cand
    return None


def _is_traceback_noise_line(line: str, in_traceback_frame: bool) -> bool:
    """判断单行日志是否属于 Python Traceback 语法噪声行。"""
    s = line.strip()
    if not s:
        return True
    low = s.lower()
    if s.startswith(("Traceback (most recent call last)", "During handling of the above exception", 'File "')) or _BRACKET_ONLY_RE.match(s):
        return True
    if low == "raise" or low.startswith("raise "):
        return True
    if in_traceback_frame and ("output=stdout" in low or "stderr=stderr" in low or "redact_secrets(" in low or s.endswith(",")):
        return True
    return False


def _pick_meaningful_tail_error(lines: list[str]) -> str:
    """从部署日志尾部挑出最具语义的真实错误行，避免被通用退出码或括号噪声掩盖。"""
    priority = ("[错误]", "RuntimeError:", "TimeoutError:", "ValueError:", "FileNotFoundError:", "ConnectionError:", "[colab]")
    fallback_wrapper = ""
    norm_lines = [redact_secrets(str(raw or "").strip()) for raw in lines]
    for s in reversed(norm_lines):
        if not s or _is_traceback_noise_line(s, False):
            continue
        if _GENERIC_EXIT_WRAPPER_RE.match(s):
            if not fallback_wrapper:
                fallback_wrapper = s
            continue
        if any(s.startswith(p) or p in s for p in priority):
            return s
    for s in reversed(norm_lines):
        if not s or _is_traceback_noise_line(s, True):
            continue
        if _GENERIC_EXIT_WRAPPER_RE.match(s):
            if not fallback_wrapper:
                fallback_wrapper = s
            continue
        low = s.lower()
        if ("calledprocesserror" in low and len(norm_lines) > 2) or (re.match(r"^[a-z_]+=", low) and any(k in low for k in ("cmd", "returncode", "stdout"))):
            continue
        return s
    return fallback_wrapper or "部署异常中止（详情见诊断日志）"


def _match_deploy_capacity_failure(
    joined_low: str,
    gpu_up: str,
    pre_assignments: int,
    recreate: bool,
) -> dict[str, Any] | None:
    """匹配显卡配额不足、并发上限或高峰期容量超时类故障。"""
    if "backend rejected accelerator" in joined_low or "not have quota or entitlement" in joined_low:
        return {
            "kind": "entitlement",
            "stockout_gpu": gpu_up,
            "reason": f"当前账号配额暂不支持 {gpu_up}",
            "message": f"⛔ 当前 Colab 账号无 {gpu_up} 配额或等级权限",
        }
    if "allocation refused" in joined_low or "toomanyassignments" in joined_low or ("412" in joined_low and "precondition" in joined_low):
        if pre_assignments > 0 and not recreate:
            return {
                "kind": "limit_or_stockout",
                "stockout_gpu": None,
                "reason": "已有运行中实例或目标卡高峰期名额紧张",
                "message": f"⚠️ 分配 {gpu_up} 被拒：并发上限或实例待纳管，可点增量部署或换卡",
            }
        return {
            "kind": "stockout",
            "stockout_gpu": gpu_up,
            "reason": f"高峰期 {gpu_up} 算力容量不足",
            "message": f"⚠️ 高峰期 {gpu_up} 算力紧缺无法分配",
        }
    is_read_timeout = any(k in joined_low for k in ("read timed out", "readtimeout", "connecttimeout"))
    is_capacity = (
        any(k in joined_low for k in ("resourceexhausted", "out of capacity", "no backend", "cannot connect to gpu backend", "failed to assign"))
        or ("503" in joined_low and "assign" in joined_low)
        or (is_read_timeout and any(k in joined_low for k in ("assign", "creating session", " new ", "colab.research.google.com")))
        or ("timeout" in joined_low and "creating session" in joined_low)
    )
    if not is_capacity:
        return None
    if is_read_timeout:
        return {
            "kind": "stockout",
            "stockout_gpu": gpu_up,
            "reason": f"分配 {gpu_up} 接口响应超时",
            "message": f"⚠️ 请求分配 {gpu_up} 超时，若后台已开机可直接点增量部署复用",
        }
    return {
        "kind": "stockout",
        "stockout_gpu": gpu_up,
        "reason": f"高峰期 {gpu_up} 分配拥堵/无空闲算力",
        "message": f"⚠️ 高峰期 {gpu_up} 无空闲卡或响应超时",
    }


def _match_deploy_tail_hint(cleaned: list[str]) -> dict[str, Any] | None:
    """倒序匹配磁盘/显存不足、llama-server/ComfyUI 启动超时与内核握手异常。"""
    for ln in reversed(cleaned):
        if "磁盘空间不足" in ln or "显存不足" in ln:
            return {"kind": "error", "stockout_gpu": None, "reason": ln, "message": f"⚠️ {ln[:60]}"}
        if "llama-server" in ln and any(k in ln for k in ("超时", "失败", "异常退出")):
            return {"kind": "error", "stockout_gpu": None, "reason": "llama-server 启动异常", "message": f"⚠️ {ln[:58]}"}
        if "等待 ComfyUI 就绪超时" in ln:
            return {"kind": "error", "stockout_gpu": None, "reason": "ComfyUI 启动超时", "message": "⚠️ 等待 ComfyUI 启动超时，可尝试重新增量部署"}
        if "WebSocket" in ln or "握手超时" in ln:
            return {"kind": "error", "stockout_gpu": None, "reason": "Colab 内核通信超时", "message": "⚠️ 远端内核握手超时，可重试或停卡重开"}
    return None


def classify_deploy_failure(lines: list[str], gpu: str, pre_assignments: int, profile: str, recreate: bool) -> dict[str, Any]:
    """对部署失败日志做结构化故障分类与中文建议文案归纳。"""
    cleaned = [redact_secrets(str(ln or "")).strip() for ln in lines if ln and str(ln).strip()]
    joined_text = "\n".join(cleaned)
    joined_low = joined_text.lower()
    gpu_up = str(gpu or "").upper()

    if "provided tunnel token is not valid" in joined_low or "cloudflared 命名隧道异常退出" in joined_text:
        return {
            "kind": "error",
            "stockout_gpu": None,
            "reason": "Cloudflare 固定隧道 Token 无效",
            "message": "⚠️ Cloudflare 固定隧道 Token 无效，请重填或切回免费隧道",
        }
    if cap_hit := _match_deploy_capacity_failure(joined_low, gpu_up, pre_assignments, recreate):
        return cap_hit
    if tail_hit := _match_deploy_tail_hint(cleaned):
        return tail_hit

    tail = _pick_meaningful_tail_error(cleaned)
    if len(tail) > 52:
        tail = tail[:49] + "..."
    return {"kind": "error", "stockout_gpu": None, "reason": tail, "message": f"⚠️ {profile} 部署异常: {tail}"}


def build_ctl_up_args(
    *,
    python_exe: str,
    ctl_script: Path,
    session: str,
    profiles: list[str],
    gpu: str,
    high_mem: bool = False,
    recreate: bool = False,
    endpoint: str = "",
    minimax_r2v: bool = True,
    minimax_pe: bool = True,
    llm_quant: str = "UD-Q4_K_XL",
    llm_ctx: int = 32768,
    llm_mmproj: bool = True,
    llm27_quant: str = "UD-Q4_K_XL",
    llm27_ctx: int = 65536,
    llm27_mmproj: bool = True,
    cohost_comfy: bool = False,
    account_id: str = "",
    no_named_tunnel: bool = False,
    ctl_prefix: list[str] | None = None,
) -> list[str]:
    """构造 `colab_ctl.py up` 完整命令行参数列表（兼容源码态与 `--internal-ctl` 冻结态）。"""
    norm_profs = normalize_profiles(profiles)
    profile = norm_profs[0]
    llm_prof = next((p for p in norm_profs if p in LLM_PROFILES), None)
    base = list(ctl_prefix) if ctl_prefix else [python_exe, "-u", str(ctl_script)]
    cmd = [*base, "up", "-s", session]
    cmd.extend(["--profiles", ",".join(norm_profs)] if len(norm_profs) >= 2 else ["--profile", profile])
    cmd.extend(["--gpu", str(gpu).strip().upper()])
    if endpoint:
        cmd.extend(["--endpoint", str(endpoint).strip()])
    if resolve_high_mem(gpu, high_mem):
        cmd.append("--high-mem")
    if recreate:
        cmd.append("--recreate")
    if cohost_comfy and len(norm_profs) < 2 and llm_prof is not None:
        cmd.append("--cohost-comfy")
    acct = str(account_id or "").strip()
    if acct:
        cmd.extend(["--account", acct])
    if no_named_tunnel:
        cmd.append("--no-named-tunnel")
    if "minimax" in norm_profs:
        if not minimax_r2v and not minimax_pe:
            cmd.append("--minimax-lite")
        else:
            if not minimax_r2v:
                cmd.append("--no-r2v")
            if not minimax_pe:
                cmd.append("--no-pe")
    if llm_prof == "qwen_27b":
        cmd.extend(["--llm-quant", llm27_quant, "--llm-ctx", str(int(llm27_ctx))])
        if not llm27_mmproj:
            cmd.append("--no-mmproj")
    elif llm_prof == "qwen_llm":
        cmd.extend(["--llm-quant", llm_quant, "--llm-ctx", str(int(llm_ctx))])
        if not llm_mmproj:
            cmd.append("--no-mmproj")
    return cmd


def parse_llm_boot_params(cfg: dict[str, Any], prefs: dict[str, Any]) -> tuple[str, int, bool, str, int, bool]:
    """统一解析并夹紧 180B 与 27B 模型的量化、上下文长度与多模态开机参数。"""
    def _one(kq: str, pkq: str, kc: str, pkc: str, km: str, pkm: str, d_ctx: int) -> tuple[str, int, bool]:
        rq = str(cfg.get(kq) or prefs.get(pkq) or "UD-Q4_K_XL").strip()
        q = rq if rq in SUPPORTED_LLM_QUANTS else "UD-Q4_K_XL"
        try:
            ctx = int(cfg.get(kc) or prefs.get(pkc) or d_ctx)
        except (TypeError, ValueError):
            ctx = d_ctx
        return q, max(4096, min(262144, ctx)), bool(cfg.get(km, prefs.get(pkm, True)))

    q180, c180, m180 = _one("llm_quant", "boot_llm_quant", "llm_ctx", "boot_llm_ctx", "llm_mmproj", "boot_llm_mmproj", 32768)
    q27, c27, m27 = _one("llm27_quant", "boot_llm27_quant", "llm27_ctx", "boot_llm27_ctx", "llm27_mmproj", "boot_llm27_mmproj", 65536)
    return q180, c180, m180, q27, c27, m27


__all__ = [
    "COMFY_PROFILES",
    "DEFAULT_DEPLOY_TUNNEL_URL_PATTERN",
    "DEPLOY_DL_TOTAL_RE",
    "DEPLOY_LLM_TUNNEL_LINE_RE",
    "DEPLOY_PHASE_RE",
    "DEPLOY_STEP_BASE_PCT",
    "DEPLOY_TUNNEL_LINE_RE",
    "DL_SPEED_RE",
    "LLM_27B_BASE_SHARDS",
    "LLM_PROFILES",
    "LLM_QUANT_BASE_SHARDS",
    "PCT_TOKEN_RE",
    "PROFILE_LABELS",
    "STEP_BANNER_RE",
    "SUPPORTED_BOOT_GPUS",
    "SUPPORTED_LLM_QUANTS",
    "SUPPORTED_PROFILES",
    "_BRACKET_ONLY_RE",
    "_GENERIC_EXIT_WRAPPER_RE",
    "_is_traceback_noise_line",
    "_pick_meaningful_tail_error",
    "build_ctl_up_args",
    "calc_phase_dl_total",
    "classify_deploy_failure",
    "consume_deploy_line",
    "default_deploy_state",
    "format_elapsed_cn",
    "normalize_profiles",
    "parse_llm_boot_params",
    "parse_profiles_cli_arg",
    "profile_bootstrap_spec",
    "recommend_fallback_gpu",
    "resolve_high_mem",
    "scale_phase_pct",
    "select_default_gpu",
]
