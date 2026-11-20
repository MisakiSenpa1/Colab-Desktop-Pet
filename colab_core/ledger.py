"""colab_core.ledger —— 多账号每日账本、偏好与状态安全清洗、以及 ComfyUI/LLM 遥测解析模块。"""

from __future__ import annotations

import json
import math
import re
import shutil
import sys
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

_PKG_DIR = Path(__file__).resolve().parents[1]
if str(_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR))

from colab_core.catalog import get_default_measured_gpu_rates  # noqa: E402
from colab_core.deploy_state import (  # noqa: E402
    SUPPORTED_BOOT_GPUS,
    SUPPORTED_LLM_QUANTS,
    SUPPORTED_PROFILES,
    normalize_profiles,
)
from colab_core.exec_runner import get_diag_logger  # noqa: E402
from colab_core.reasoning_specs import (  # noqa: E402
    SUPPORTED_AGENT_FORMATS,
    normalize_reasoning_effort_value,
    normalize_reasoning_level_list,
)
from colab_security import redact_secrets  # noqa: E402

MAX_PENDING_PULL_ITEMS = 4096
SCARCE_GPU_TTL_SEC = 600.0
SUPPORTED_IDLE_STOP_MINUTES = (0, 15, 30, 60)

DEFAULT_MEASURED_GPU_RATES: dict[str, float] = get_default_measured_gpu_rates()
DEFAULT_PREFS: dict[str, Any] = {
    "x": None,
    "y": None,
    "scale": 1.0,
    "on_top": True,
    "bubble_open": True,
    "left_flipped": False,
    "dock_edge": None,
    "edge_auto_hide": True,
    "poll_interval_sec": 60,
    "warn_hours": 10.0,
    "warn_balance": 50.0,
    "long_run_warn_minutes": 180,
    "auto_pull_outputs": True,
    "keep_alive": True,
    "idle_auto_stop_minutes": 60,
    "idle_archive_fail_policy": "stop_billing",
    "boot_profile": "anima",
    "boot_profiles": ["anima"],
    "boot_gpu": "L4",
    "boot_high_mem": False,
    "boot_minimax_r2v": True,
    "boot_minimax_pe": True,
    "boot_llm_quant": "UD-Q4_K_XL",
    "boot_llm_ctx": 32768,
    "boot_llm_mmproj": True,
    "boot_llm27_quant": "UD-Q4_K_XL",
    "boot_llm27_ctx": 65536,
    "boot_llm27_mmproj": True,
    "active_account_id": "",
    "auto_pool_schedule": True,
    "cluster_split_mode": True,
    "agent_driver": "native",
    "agent_brain_source": "auto",
    "agent_api_format": "openai",
    "agent_api_base_url": "",
    "agent_model": "",
    "agent_reasoning_effort": "auto",
    "agent_auto_approve_billing": False,
    "agent_providers": [],
    "agent_active_provider_id": "",
    "colab_exe_path": "",
    "colab_py_path": "",
    "rclone_exe_path": "",
    "rclone_conf_path": "",
}
ACCOUNT_TIER_LABELS: dict[str, str] = {
    "ultra": "Ultra",
    "pro_plus": "Pro+",
    "pro": "Pro",
    "free": "Free",
    "unknown": "Colab",
}
_AUTO_ACCT_LABEL_RE = re.compile(r"^(?:Ultra|Pro\+|Pro|Free|Colab)\s*(\d+)号$")

ACCOUNT_LEDGER_KEYS = (
    "date",
    "day_start_balance",
    "day_consumed_cu",
    "day_topup_cu",
    "last_balance",
    "peak_balance",
    "last_delta",
    "baseline_balance",
    "active_since_ts",
    "active_endpoint",
    "session_start_balance",
    "last_rate_hourly",
    "last_assignments_count",
    "last_sessions",
    "last_eligible_gpus",
    "last_ineligible_gpus",
    "last_eligible_tpus",
    "subscription_tier",
    "last_updated_ts",
    "tunnel_cache",
    "managed_binding",
)

_LOOPBACK_HTTP_ORIGIN_RE = re.compile(
    r"^http://(?:127\.0\.0\.1|localhost|\[::1\])(?::\d{1,5})?$",
    re.IGNORECASE,
)


def default_baseline(balance: float) -> float:
    """将余额向上取整到百位作为默认额度基线（最低 100 CU）。"""
    return 100.0 if balance <= 0 else float(max(100, int(math.ceil(balance / 100.0) * 100)))


def _infer_explicit_account_tier(raw_tier: str, peak: float) -> tuple[str, str, float] | None:
    """根据显式订阅等级字符串直接映射等级与基线额度。"""
    if not raw_tier:
        return None
    if "ultra" in raw_tier:
        return "ultra", ACCOUNT_TIER_LABELS["ultra"], max(1000.0, default_baseline(peak))
    if "pro_plus" in raw_tier or "pro+" in raw_tier or "proplus" in raw_tier:
        return "pro_plus", ACCOUNT_TIER_LABELS["pro_plus"], max(500.0, default_baseline(peak)) if peak > 500.0 else 500.0
    if "pro" in raw_tier:
        return "pro", ACCOUNT_TIER_LABELS["pro"], 200.0 if (peak > 100.0 or peak <= 0.0) else 100.0
    if "free" in raw_tier:
        return "free", ACCOUNT_TIER_LABELS["free"], 100.0
    return None


def infer_account_tier_and_baseline(
    balance: float | None,
    *,
    peak_balance: float | None = None,
    eligible_gpus: Any = None,
    ineligible_gpus: Any = None,
    explicit_tier: str = "",
) -> tuple[str, str, float]:
    """根据账号额度峰值与显卡解锁列表精确推断订阅等级 `(tier_key, tier_label, baseline)`。"""
    nums = [float(x) for x in (balance, peak_balance) if isinstance(x, (int, float))]
    peak = max(nums) if nums else 0.0
    eg = {str(g).upper() for g in (eligible_gpus or ()) if g}
    ig = {str(g).upper() for g in (ineligible_gpus or ()) if g}

    raw_tier = str(explicit_tier or "").strip().lower()
    if explicit_hit := _infer_explicit_account_tier(raw_tier, peak):
        return explicit_hit

    has_h100 = "H100" in eg and "H100" not in ig
    if peak > 550.0 or (has_h100 and (peak > 0.0 or balance is None)):
        return "ultra", ACCOUNT_TIER_LABELS["ultra"], max(1000.0, default_baseline(peak))
    if 250.0 < peak <= 550.0:
        return "pro_plus", ACCOUNT_TIER_LABELS["pro_plus"], 500.0
    if 0.0 < peak <= 250.0 or bool(eg & {"G4", "A100", "L4"}):
        return "pro", ACCOUNT_TIER_LABELS["pro"], 200.0 if (peak > 100.0 or peak <= 0.0) else 100.0
    if eg and eg <= {"T4", "CPU", "NONE"} and peak <= 0.0:
        return "free", ACCOUNT_TIER_LABELS["free"], 100.0
    return "unknown", ACCOUNT_TIER_LABELS["unknown"], 1000.0


def format_pool_tier_summary(tiers: list[str]) -> str:
    """将号池各账号等级汇总为简洁的中文摘要（如 `2 个 Ultra 账号` 或 `1 Ultra + 1 Pro`）。"""
    clean = [str(t or "unknown").strip().lower() for t in (tiers or [])]
    if not clean:
        return "暂无已绑定账号"
    counts: dict[str, int] = {}
    for t in clean:
        key = t if t in ACCOUNT_TIER_LABELS else "unknown"
        counts[key] = counts.get(key, 0) + 1
    if len(counts) == 1:
        only_t, cnt = next(iter(counts.items()))
        lbl = ACCOUNT_TIER_LABELS.get(only_t, "Colab")
        return f"{cnt} 个 {lbl} 账号"
    ordered = ("ultra", "pro_plus", "pro", "free", "unknown")
    parts = [f"{counts[k]} {ACCOUNT_TIER_LABELS[k]}" for k in ordered if counts.get(k, 0) > 0]
    sep = " + " if len(parts) <= 2 else " · "
    return sep.join(parts)


def default_state_dict() -> dict[str, Any]:
    """构造干净的默认桌宠持久化状态字典。"""
    return {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "day_start_balance": None,
        "day_consumed_cu": 0.0,
        "day_topup_cu": 0.0,
        "last_balance": None,
        "peak_balance": None,
        "last_delta": 0.0,
        "baseline_balance": 1000.0,
        "active_since_ts": None,
        "active_endpoint": None,
        "session_start_balance": None,
        "last_rate_hourly": 0.0,
        "last_assignments_count": 0,
        "last_sessions": [],
        "last_eligible_gpus": None,
        "last_ineligible_gpus": None,
        "last_eligible_tpus": None,
        "subscription_tier": "",
        "last_updated_ts": None,
        "measured_gpu_rates": dict(DEFAULT_MEASURED_GPU_RATES),
        "tunnel_cache": {},
        "managed_binding": None,
        "account_profiles": [],
        "account_ledgers": {},
        "onboarding_cache": {"gdrive_free_gb": None, "gdrive_total_gb": None, "gdrive_checked_ts": None},
        "pending_pull_items": [],
        "prefs": dict(DEFAULT_PREFS),
    }


def sanitize_model_capabilities(raw_caps: Any) -> dict[str, dict[str, Any]]:
    """规范化供应商下各模型的思考能力缓存字典。"""
    if not isinstance(raw_caps, dict):
        return {}
    cleaned: dict[str, dict[str, Any]] = {}
    for raw_mid, raw_spec in raw_caps.items():
        mid = str(raw_mid or "").strip()[:160]
        if not mid or not isinstance(raw_spec, dict):
            continue
        levels = normalize_reasoning_level_list(
            raw_spec.get("levels")
            or raw_spec.get("reasoning_levels")
            or raw_spec.get("supported_reasoning_levels")
        )
        reasoning_flag = bool(raw_spec.get("reasoning")) if "reasoning" in raw_spec else bool(levels)
        def_lvl = normalize_reasoning_effort_value(
            raw_spec.get("default_level")
            or raw_spec.get("default_reasoning_level")
            or raw_spec.get("default_reasoning_effort"),
            default="",
        )
        if def_lvl == "auto":
            def_lvl = ""
        if not reasoning_flag:
            levels, def_lvl = [], ""
        cleaned[mid] = {
            "reasoning": reasoning_flag,
            "levels": levels,
            "default_level": def_lvl,
        }
        if len(cleaned) >= 256:
            break
    return cleaned


def _sanitize_provider_models(raw_models: Any, default_model: str) -> tuple[list[str], str]:
    """清洗单个供应商的模型列表与默认模型名。"""
    items = [x.strip() for x in raw_models.split(",")] if isinstance(raw_models, str) else raw_models
    models: list[str] = []
    if isinstance(items, (list, tuple)):
        for m in items:
            ms = str(m or "").strip()[:160]
            if ms and ms not in models:
                models.append(ms)
            if len(models) >= 200:
                break
    eff_default = str(default_model or "").strip()[:160]
    if eff_default and eff_default not in models:
        models.insert(0, eff_default)
    elif not eff_default and models:
        eff_default = models[0]
    return models, eff_default


def sanitize_agent_providers(raw_providers: Any) -> list[dict[str, Any]]:
    """清洗多模型供应商目录偏好，强制剥离任何敏感密钥字段。"""
    if not isinstance(raw_providers, list):
        return []
    out: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for idx, item in enumerate(raw_providers):
        if len(out) >= 24 or not isinstance(item, dict):
            continue
        raw_id = str(item.get("id") or f"prov_{idx + 1}").strip()
        pid = re.sub(r"[^a-zA-Z0-9_-]", "_", raw_id)[:48] or f"prov_{idx + 1}"
        if pid in seen_ids:
            continue
        seen_ids.add(pid)
        fmt = str(item.get("api_format") or "openai").strip().lower()
        fmt = fmt if fmt in SUPPORTED_AGENT_FORMATS else "openai"
        models, default_model = _sanitize_provider_models(item.get("models"), str(item.get("default_model") or ""))
        out.append({
            "id": pid,
            "name": str(item.get("name") or "").strip()[:64] or "自定义供应商",
            "api_format": fmt,
            "base_url": str(item.get("base_url") or "").strip().rstrip("/")[:512],
            "models": models,
            "default_model": default_model,
            "model_capabilities": sanitize_model_capabilities(item.get("model_capabilities")),
        })
    return out


def _sanitize_llm_boot_prefs(cleaned: dict[str, Any]) -> None:
    """原地清洗 180B 与 27B 模型开机量化/上下文/多模态偏好。"""
    for q_key, ctx_key, mm_key, def_ctx in (
        ("boot_llm_quant", "boot_llm_ctx", "boot_llm_mmproj", 32768),
        ("boot_llm27_quant", "boot_llm27_ctx", "boot_llm27_mmproj", 65536),
    ):
        q_val = str(cleaned.get(q_key) or "UD-Q4_K_XL").strip()
        cleaned[q_key] = q_val if q_val in SUPPORTED_LLM_QUANTS else "UD-Q4_K_XL"
        try:
            cv = int(cleaned.get(ctx_key) or def_ctx)
        except (TypeError, ValueError):
            cv = def_ctx
        cleaned[ctx_key] = max(4096, min(262144, cv))
        cleaned[mm_key] = bool(cleaned.get(mm_key, True))


def _sanitize_agent_prefs(cleaned: dict[str, Any]) -> None:
    """原地清洗同进程 AI 副驾偏好配置（严禁落盘 agent_api_key）。"""
    drv = str(cleaned.get("agent_driver") or "native").strip().lower()
    cleaned["agent_driver"] = drv if drv in ("native", "claude_code") else "native"
    bs = str(cleaned.get("agent_brain_source") or "auto").strip().lower()
    cleaned["agent_brain_source"] = bs if bs in ("auto", "colab_llm", "external_api") else "auto"
    fmt = str(cleaned.get("agent_api_format") or "openai").strip().lower()
    cleaned["agent_api_format"] = fmt if fmt in SUPPORTED_AGENT_FORMATS else "openai"
    cleaned["agent_api_base_url"] = str(cleaned.get("agent_api_base_url") or "").strip()[:512]
    cleaned["agent_model"] = str(cleaned.get("agent_model") or "").strip()[:256]
    cleaned["agent_reasoning_effort"] = normalize_reasoning_effort_value(
        cleaned.get("agent_reasoning_effort"),
        default="auto",
    )
    cleaned["agent_active_provider_id"] = re.sub(
        r"[^a-zA-Z0-9_-]", "_", str(cleaned.get("agent_active_provider_id") or "").strip()
    )[:48]
    cleaned["agent_providers"] = sanitize_agent_providers(cleaned.get("agent_providers"))
    raw_aab = cleaned.get("agent_auto_approve_billing", False)
    if isinstance(raw_aab, str):
        cleaned["agent_auto_approve_billing"] = raw_aab.strip().lower() not in ("", "0", "false", "no", "off")
    else:
        cleaned["agent_auto_approve_billing"] = bool(raw_aab)
    cleaned.pop("agent_api_key", None)


def sanitize_prefs_dict(prefs: dict[str, Any]) -> dict[str, Any]:
    """逐项校验并规范化桌宠用户偏好字典。"""
    cleaned = dict(DEFAULT_PREFS)
    for k in DEFAULT_PREFS:
        if k in prefs:
            cleaned[k] = prefs[k]

    prof = str(cleaned.get("boot_profile") or "anima").strip().lower()
    fallback_p = prof if prof in SUPPORTED_PROFILES else "anima"
    raw_profiles = prefs.get("boot_profiles") if "boot_profiles" in prefs else [fallback_p]
    norm_profiles = normalize_profiles(raw_profiles, fallback_profile=fallback_p)
    cleaned["boot_profiles"] = norm_profiles
    cleaned["boot_profile"] = norm_profiles[0]

    _sanitize_llm_boot_prefs(cleaned)

    cleaned["keep_alive"] = bool(cleaned.get("keep_alive", True))
    cleaned["auto_pool_schedule"] = bool(cleaned.get("auto_pool_schedule", True))
    cleaned["cluster_split_mode"] = bool(cleaned.get("cluster_split_mode", True))
    try:
        idle_mins = int(cleaned.get("idle_auto_stop_minutes", 60))
    except (TypeError, ValueError):
        idle_mins = 60
    cleaned["idle_auto_stop_minutes"] = idle_mins if idle_mins in SUPPORTED_IDLE_STOP_MINUTES else 60
    pol = str(cleaned.get("idle_archive_fail_policy") or "stop_billing").strip().lower()
    cleaned["idle_archive_fail_policy"] = pol if pol in ("stop_billing", "keep_alive") else "stop_billing"
    for path_key in ("active_account_id", "colab_exe_path", "colab_py_path", "rclone_exe_path", "rclone_conf_path"):
        cleaned[path_key] = str(cleaned.get(path_key) or "").strip()[:512]

    _sanitize_agent_prefs(cleaned)
    return cleaned


def _backup_corrupt_state(state_path: Path, reason: str) -> None:
    """状态文件解析失败或结构非法时，尽力备份为 `.corrupt-<ts>` 并记录诊断告警。"""
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    bak_path = state_path.with_name(f"{state_path.stem}.corrupt-{ts}{state_path.suffix}")
    saved_bak = ""
    try:
        shutil.copy2(state_path, bak_path)
        saved_bak = str(bak_path)
    except OSError:
        saved_bak = ""
    try:
        get_diag_logger().warn(
            "state",
            f"检测到损坏或无效的状态文件，已恢复默认值: {reason}",
            path=str(state_path),
            backup=saved_bak,
        )
    except Exception as exc:
        # 诊断日志本身不可用时降级输出至 stderr，避免静默吞错
        if sys.stderr is not None:
            try:
                sys.stderr.write(f"[state] 诊断日志写入失败: {redact_secrets(str(exc))}\n")
            except OSError:
                return


def _safe_round_float(val: Any) -> float | None:
    """安全将值转为保留 2 位小数的 float，失败或为空返回 None。"""
    if val is None:
        return None
    try:
        return round(float(val), 2)
    except (TypeError, ValueError):
        return None


def _safe_int(val: Any) -> int | None:
    """安全将值转为 int，失败或为空返回 None。"""
    if val is None:
        return None
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def _sanitize_measured_gpu_rates(raw_rates: Any) -> dict[str, float]:
    """清洗已实测显卡 CU 费率字典（保留合法实测值，不覆写有效用户历史记录）。"""
    cleaned_rates = dict(DEFAULT_MEASURED_GPU_RATES)
    if not isinstance(raw_rates, dict):
        return cleaned_rates
    valid_keys = {f"{g}:{s}" for g in SUPPORTED_BOOT_GPUS for s in ("STD", "HM") if not (g == "L4" and s == "HM")}
    for rk, rv in raw_rates.items():
        if rk in valid_keys:
            try:
                fv = float(rv)
                if 0.02 < fv < 200.0:
                    cleaned_rates[rk] = round(fv, 2)
            except (TypeError, ValueError):
                continue
    return cleaned_rates


def _sanitize_tunnel_cache(raw_cache: Any, is_allowed_tunnel_fn: Callable[[str], bool]) -> dict[str, str]:
    """清洗并按隧道白名单校验端点隧道缓存字典。"""
    if not isinstance(raw_cache, dict):
        return {}
    return {
        ep: val.strip().rstrip("/")
        for ep, val in raw_cache.items()
        if isinstance(ep, str) and isinstance(val, str) and is_allowed_tunnel_fn(val.strip().rstrip("/"))
    }


def _sanitize_managed_binding(raw_mb: Any) -> dict[str, str] | None:
    """清洗当前受管实例绑定结构（`cli_scope` 为部署时实际使用的 colab-cli 账号作用域，仅在已记录时保留）。"""
    if isinstance(raw_mb, dict) and str(raw_mb.get("endpoint") or "").strip():
        out = {
            "account_id": re.sub(r"[^a-zA-Z0-9_-]", "", str(raw_mb.get("account_id") or ""))[:24],
            "session": str(raw_mb.get("session") or "comfy").strip()[:32] or "comfy",
            "endpoint": str(raw_mb.get("endpoint") or "").strip()[:64],
        }
        if "cli_scope" in raw_mb:
            out["cli_scope"] = re.sub(r"[^a-zA-Z0-9_-]", "", str(raw_mb.get("cli_scope") or ""))[:24]
        return out
    return None


def _sanitize_account_profiles(raw_profiles: Any, sanitize_label_fn: Callable[[str, str], str]) -> list[dict[str, Any]]:
    """去重并清洗账号档案列表。"""
    if not isinstance(raw_profiles, list):
        return []
    clean_accts: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for item in raw_profiles:
        if not isinstance(item, dict):
            continue
        aid = re.sub(r"[^a-zA-Z0-9_-]", "", str(item.get("id") or ""))[:24]
        if not aid or aid in seen_ids:
            continue
        seen_ids.add(aid)
        tier_val = str(item.get("tier") or "").strip().lower()
        tier_val = tier_val if tier_val in ACCOUNT_TIER_LABELS else ""
        clean_accts.append({
            "id": aid,
            "label": sanitize_label_fn(str(item.get("label") or ""), "Colab 账号"),
            "email_masked": str(item.get("email_masked") or "***")[:64],
            "last_balance": _safe_round_float(item.get("last_balance")),
            "last_is_running": bool(item.get("last_is_running", False)),
            "last_updated_ts": _safe_int(item.get("last_updated_ts")),
            **({"tier": tier_val} if tier_val else {}),
        })
    return clean_accts


def _sanitize_account_ledgers(raw_ledgers: Any) -> dict[str, dict[str, Any]]:
    """按白名单键集合过滤并清洗多账号历史账本映射。"""
    if not isinstance(raw_ledgers, dict):
        return {}
    return {
        aid: {lk: lobj[lk] for lk in ACCOUNT_LEDGER_KEYS if lk in lobj}
        for aid_raw, lobj in raw_ledgers.items()
        if (aid := re.sub(r"[^a-zA-Z0-9_-]", "", str(aid_raw or ""))[:24]) and isinstance(lobj, dict)
    }


def _sanitize_onboarding_cache(raw_oc: Any) -> dict[str, Any]:
    """清洗云盘容量探测缓存字段。"""
    if not isinstance(raw_oc, dict):
        return {"gdrive_free_gb": None, "gdrive_total_gb": None, "gdrive_checked_ts": None}
    return {
        "gdrive_free_gb": _safe_round_float(raw_oc.get("gdrive_free_gb")),
        "gdrive_total_gb": _safe_round_float(raw_oc.get("gdrive_total_gb")),
        "gdrive_checked_ts": _safe_int(raw_oc.get("gdrive_checked_ts")),
    }


def _sanitize_pending_pull_items(
    raw_items: Any,
    is_allowed_tunnel_fn: Callable[[str], bool],
) -> list[dict[str, str]]:
    """校验、去重并截断待补拉的 ComfyUI 输出文件队列，拦截路径穿越与非法源站。"""
    if not isinstance(raw_items, list):
        return []
    clean_pending: list[dict[str, str]] = []
    seen_pending: set[tuple[str, str, str, str]] = set()
    for raw_it in raw_items:
        if not isinstance(raw_it, dict):
            continue
        b_url = str(raw_it.get("base_url") or "").strip().rstrip("/")
        if not b_url or not (_LOOPBACK_HTTP_ORIGIN_RE.match(b_url) or is_allowed_tunnel_fn(b_url)):
            continue
        raw_fn = str(raw_it.get("filename") or "").strip()
        if not raw_fn or "/" in raw_fn or "\\" in raw_fn or raw_fn in (".", ".."):
            continue
        sub_s = str(raw_it.get("subfolder") or "").replace("\\", "/").strip("/")
        if ".." in PurePosixPath(sub_s).parts:
            continue
        tp_s = (str(raw_it.get("type") or "output").strip() or "output")[:32]
        dkey = (b_url, sub_s, raw_fn, tp_s)
        if dkey in seen_pending:
            continue
        seen_pending.add(dkey)
        clean_pending.append({"base_url": b_url, "filename": raw_fn, "subfolder": sub_s, "type": tp_s})
    if len(clean_pending) > MAX_PENDING_PULL_ITEMS:
        return clean_pending[-MAX_PENDING_PULL_ITEMS:]
    return clean_pending


def load_pet_state_dict(
    state_path: Path,
    *,
    is_allowed_tunnel_fn: Callable[[str], bool],
    sanitize_label_fn: Callable[[str, str], str],
) -> dict[str, Any]:
    """安全加载并逐字段校验清洗 `pet_state.json`，损坏时自动备份并恢复默认值。"""
    default = default_state_dict()
    if not state_path.exists():
        return default
    try:
        raw = json.loads(state_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            _backup_corrupt_state(state_path, f"根节点类型为 {type(raw).__name__}")
            return default
        merged = dict(default)
        field_cleaners: dict[str, Callable[[Any], Any]] = {
            "prefs": lambda v: sanitize_prefs_dict({**DEFAULT_PREFS, **v}) if isinstance(v, dict) else merged["prefs"],
            "measured_gpu_rates": lambda v: _sanitize_measured_gpu_rates(v) if isinstance(v, dict) else merged["measured_gpu_rates"],
            "tunnel_cache": lambda v: _sanitize_tunnel_cache(v, is_allowed_tunnel_fn) if isinstance(v, dict) else merged["tunnel_cache"],
            "managed_binding": _sanitize_managed_binding,
            "account_profiles": lambda v: _sanitize_account_profiles(v, sanitize_label_fn) if isinstance(v, list) else merged["account_profiles"],
            "account_ledgers": lambda v: _sanitize_account_ledgers(v) if isinstance(v, dict) else merged["account_ledgers"],
            "onboarding_cache": lambda v: _sanitize_onboarding_cache(v) if isinstance(v, dict) else merged["onboarding_cache"],
            "pending_pull_items": lambda v: _sanitize_pending_pull_items(v, is_allowed_tunnel_fn) if isinstance(v, list) else merged["pending_pull_items"],
        }
        for key in default:
            if key in field_cleaners:
                if key == "managed_binding" or key in raw:
                    merged[key] = field_cleaners[key](raw.get(key))
            elif key in raw:
                merged[key] = raw[key]
        return merged
    except Exception as exc:
        _backup_corrupt_state(state_path, redact_secrets(str(exc) or type(exc).__name__))
        return default


def estimate_session_start_balance(balance: float, rate_hourly: float, active_since_ts: int, ref_ts: int, day_start: float, day_topup: float) -> float:
    """按当前余额、小时速率与活跃起始时间倒推本次会话开机时余额估算值。"""
    est = rate_hourly * max(0, ref_ts - int(active_since_ts)) / 3600.0
    cap = day_start + day_topup
    return min(cap, balance + max(0.0, est)) if cap >= balance else (balance + max(0.0, est))


def gpu_rate_key(accelerator: Any, machine_shape: Any = None) -> str | None:
    """将显卡型号与内存规格归一化为费率表主键（如 `G4:STD`、`A100:HM`）。"""
    acc = str(accelerator or "").strip().upper()
    if acc not in SUPPORTED_BOOT_GPUS:
        return None
    if acc == "L4":
        return "L4:STD"
    shape = str(machine_shape or "").strip().upper()
    return f"{acc}:{'HM' if shape in ('HIGH_RAM', 'HM', '1') else 'STD'}"


def snapshot_current_ledger(state: dict[str, Any]) -> dict[str, Any]:
    """截取当前活跃账号在顶层状态字典中的全部账本字段快照。"""
    return {lk: state.get(lk) for lk in ACCOUNT_LEDGER_KEYS}


def restore_ledger_for_account(state: dict[str, Any], acct_id: str) -> None:
    """将指定账号已存档账本恢复到状态顶层（缺失字段用默认初值补齐）。"""
    fresh = default_state_dict()
    ledgers = state.setdefault("account_ledgers", {})
    saved = ledgers.get(acct_id) if (acct_id and isinstance(ledgers, dict)) else None
    for lk in ACCOUNT_LEDGER_KEYS:
        state[lk] = saved[lk] if (isinstance(saved, dict) and lk in saved) else fresh[lk]


def _relabel_auto_account_name(cur_lbl: str, new_tier: str) -> str:
    """若账号标签为系统自动生成的等级序号格式（如 `Colab 1号`），按最新等级更新前缀。"""
    if not new_tier or new_tier == "unknown":
        return cur_lbl
    m = _AUTO_ACCT_LABEL_RE.match(str(cur_lbl or "").strip())
    if not m:
        return cur_lbl
    t_lbl = ACCOUNT_TIER_LABELS.get(new_tier, "Colab")
    return f"{t_lbl} {m.group(1)}号"


def upsert_account_profile(
    state: dict[str, Any],
    acct_id: str,
    email_masked: str,
    preferred_label: str = "",
    balance: float | None = None,
    is_running: bool | None = None,
    tier: str = "",
) -> dict[str, Any]:
    """新增或更新账号档案条目，自动保留用户自定义标签并按订阅等级刷新自动标签。"""
    from colab_core.accounts import default_account_label, sanitize_account_label

    profiles = state.setdefault("account_profiles", [])
    now_ts = int(time.time())
    clean_tier = str(tier or "").strip().lower()
    if clean_tier not in ACCOUNT_TIER_LABELS:
        clean_tier = ""

    for item in profiles:
        if isinstance(item, dict) and item.get("id") == acct_id:
            if preferred_label:
                item["label"] = sanitize_account_label(preferred_label, str(item.get("label") or "Colab 账号"))
            if email_masked and email_masked != "***":
                item["email_masked"] = email_masked
            if balance is not None:
                item["last_balance"] = round(float(balance), 2)
                item["last_updated_ts"] = now_ts
            if is_running is not None:
                item["last_is_running"] = bool(is_running)
            if clean_tier:
                item["tier"] = clean_tier
                if not preferred_label:
                    item["label"] = _relabel_auto_account_name(str(item.get("label") or ""), clean_tier)
            return dict(item)

    new_item: dict[str, Any] = {
        "id": acct_id,
        "label": sanitize_account_label(preferred_label, default_account_label(profiles, tier=clean_tier or "unknown")),
        "email_masked": email_masked or f"acct-{acct_id[:6]}",
        "last_balance": round(float(balance), 2) if balance is not None else None,
        "last_is_running": bool(is_running) if is_running is not None else False,
        "last_updated_ts": now_ts if balance is not None else None,
        **({"tier": clean_tier} if clean_tier else {}),
    }
    profiles.append(new_item)
    return dict(new_item)


def sync_active_account_ledger(
    state: dict[str, Any],
    *,
    balance: float | None = None,
    is_running: bool | None = None,
    tier: str = "",
) -> None:
    """将顶层实时账本同步写回当前活跃账号槽位并更新其档案摘要。"""
    active_id = str((state.get("prefs") or {}).get("active_account_id") or "").strip()
    if not active_id:
        return
    clean_tier = str(tier or "").strip().lower()
    if clean_tier not in ACCOUNT_TIER_LABELS or clean_tier == "unknown":
        clean_tier = ""
    state.setdefault("account_ledgers", {})[active_id] = snapshot_current_ledger(state)
    now_ts = int(time.time())
    for item in state.get("account_profiles") or []:
        if isinstance(item, dict) and item.get("id") == active_id:
            if balance is not None:
                item["last_balance"] = round(float(balance), 2)
                item["last_updated_ts"] = now_ts
            if is_running is not None:
                item["last_is_running"] = bool(is_running)
            if clean_tier:
                item["tier"] = clean_tier
                item["label"] = _relabel_auto_account_name(str(item.get("label") or ""), clean_tier)
            break


def short_gpu_name(raw_name: str) -> str:
    """精简 GPU 设备名称前缀用于状态栏展示。"""
    return re.sub(r"^cuda:\d+\s*", "", str(raw_name or "")).split(":")[0].strip().replace("NVIDIA ", "").strip()


def format_comfy_error(node_type: str | None, exc_type: str | None, exc_msg: str | None) -> str:
    """将 ComfyUI 节点异常格式化为紧凑中文报错摘要。"""
    node = (node_type or "节点").strip() or "节点"
    etype, emsg = (exc_type or "").strip(), (exc_msg or "").strip()
    combined = f"{etype} {emsg}".lower()
    if any(k in combined for k in ("out of memory", "outofmemoryerror", "allocation on device")):
        out = f"{node}: 显存不足 (OOM)"
    elif any(k in combined for k in ("no such file", "filenotfound", "not found", "missing")):
        out = f"{node}: 缺少模型或文件"
    else:
        out = f"{node}: {etype or emsg or '执行异常'}"
    return (out[:35] + "…") if len(out) > 36 else out


def collect_formal_outputs(entry: dict[str, Any]) -> list[dict[str, str]]:
    """从 ComfyUI `/history` 单条记录中提取正式输出文件清单（忽略临时预览）。"""
    items: list[dict[str, str]] = []
    outputs = entry.get("outputs") or {}
    if not isinstance(outputs, dict):
        return items
    for node_out in outputs.values():
        if not isinstance(node_out, dict):
            continue
        for key in ("images", "gifs", "videos", "audio"):
            for item in node_out.get(key) or []:
                if isinstance(item, dict) and item.get("filename") and str(item.get("type") or "output") == "output":
                    items.append({"filename": str(item["filename"]), "subfolder": str(item.get("subfolder") or ""), "type": "output"})
    return items


def extract_history_error(entry: dict[str, Any]) -> str | None:
    """从 ComfyUI `/history` 失败条目提取格式化中文错误描述。"""
    status = entry.get("status") or {}
    if str(status.get("status_str") or "").lower() != "error":
        return None
    node_t, exc_t, exc_m = None, None, None
    for msg in status.get("messages") or []:
        if isinstance(msg, (list, tuple)) and len(msg) >= 2 and msg[0] == "execution_error":
            data = msg[1] if isinstance(msg[1], dict) else {}
            node_t, exc_t, exc_m = data.get("node_type") or data.get("node_id"), data.get("exception_type"), data.get("exception_message")
            break
    return format_comfy_error(str(node_t) if node_t else None, str(exc_t) if exc_t else None, str(exc_m) if exc_m else None)


def extract_slot_tps(slot: dict[str, Any]) -> float | None:
    """从 llama-server `/slots` 单槽位 JSON 中提取有效生成速率（tokens/s）。"""
    containers: list[Any] = [slot.get("timings"), slot.get("performance")]
    nt = slot.get("next_token")
    containers.append(nt[0] if isinstance(nt, list) and nt else nt)
    for sub in containers:
        if isinstance(sub, dict):
            for k in ("predicted_per_second", "tokens_per_second", "tps"):
                raw = sub.get(k)
                if isinstance(raw, (int, float)) and float(raw) > 0.05:
                    return round(float(raw), 1)
    return None


__all__ = [
    "ACCOUNT_LEDGER_KEYS",
    "ACCOUNT_TIER_LABELS",
    "DEFAULT_MEASURED_GPU_RATES",
    "DEFAULT_PREFS",
    "MAX_PENDING_PULL_ITEMS",
    "SCARCE_GPU_TTL_SEC",
    "SUPPORTED_IDLE_STOP_MINUTES",
    "_AUTO_ACCT_LABEL_RE",
    "_LOOPBACK_HTTP_ORIGIN_RE",
    "_backup_corrupt_state",
    "collect_formal_outputs",
    "default_baseline",
    "default_state_dict",
    "estimate_session_start_balance",
    "extract_history_error",
    "extract_slot_tps",
    "format_comfy_error",
    "format_pool_tier_summary",
    "gpu_rate_key",
    "infer_account_tier_and_baseline",
    "load_pet_state_dict",
    "restore_ledger_for_account",
    "sanitize_agent_providers",
    "sanitize_model_capabilities",
    "sanitize_prefs_dict",
    "short_gpu_name",
    "snapshot_current_ledger",
    "sync_active_account_ledger",
    "upsert_account_profile",
]
