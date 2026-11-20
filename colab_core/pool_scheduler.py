"""colab_core.pool_scheduler —— 多账号算力池调度与双号异构集群拆分纯函数核（零 I/O）。"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any

from colab_core.catalog import get_fallback_gpu_rate, get_fallback_gpu_rate_map
from colab_core.runtime import (
    COMFY_PROFILES,
    LLM_PROFILES,
    resolve_high_mem,
    select_default_gpu,
)

# 各显卡标准/高存小时费率兜底预估值（由 stack_manifest.json SSOT 统一派生）
_FALLBACK_GPU_RATES: dict[str, float] = get_fallback_gpu_rate_map()


@dataclass(slots=True, frozen=True)
class AccountNodeSnapshot:
    """单个 Google Colab 账号节点的只读算力快照。"""

    account_id: str
    label: str
    balance: float | None
    rate_hourly: float = 0.0
    eligible_gpus: tuple[str, ...] = ()
    scarce_gpus: tuple[str, ...] = ()
    sessions: tuple[dict[str, Any], ...] = ()
    auth_ok: bool = True


@dataclass(slots=True, frozen=True)
class ScheduleDecision:
    """单实例智能选号调度决策结果。"""

    ok: bool
    account_id: str = ""
    account_label: str = ""
    gpu: str = ""
    reuse_endpoint: str = ""
    switched: bool = False
    reason: str = ""
    ranked_account_ids: tuple[str, ...] = ()


@dataclass(slots=True, frozen=True)
class SplitSlotSpec:
    """双号异构集群单个服务槽位（LLM 或 Comfy）的部署规格。"""

    slot: str
    account_id: str
    account_label: str
    profiles: tuple[str, ...]
    gpu: str
    high_mem: bool
    reuse_endpoint: str = ""
    no_named_tunnel: bool = False


@dataclass(slots=True, frozen=True)
class SplitClusterPlan:
    """双号异构集群并行编排计划。"""

    ok: bool
    reason: str = ""
    llm_slot: SplitSlotSpec | None = None
    comfy_slot: SplitSlotSpec | None = None
    estimated_total_rate: float = 0.0


def _upper_gpu_set(values: Any) -> set[str]:
    """将显卡序列规范化为大写集合。"""
    if not values:
        return set()
    return {str(v).strip().upper() for v in values if str(v or "").strip()}


def _estimate_gpu_rate(gpu: str, high_mem: bool = False) -> float:
    """估算指定显卡规格的小时 CU 消耗速率（优先命中派生费率表，否则实时回查目录 SSOT）。"""
    g = str(gpu or "L4").strip().upper() or "L4"
    eff_hm = resolve_high_mem(g, high_mem)
    key = f"{g}:{'HM' if eff_hm else 'STD'}"
    if key in _FALLBACK_GPU_RATES:
        return float(_FALLBACK_GPU_RATES[key])
    return get_fallback_gpu_rate(g, high_mem=eff_hm)


def _find_reusable_endpoint(sessions: Any, target_gpu: str) -> str:
    """在账号活跃会话中查找与目标卡型一致且可复用的 Endpoint。"""
    if not sessions:
        return ""
    matches: list[dict[str, Any]] = []
    for item in sessions:
        if not isinstance(item, dict):
            continue
        acc = str(item.get("accelerator") or "").strip().upper()
        ep = str(item.get("endpoint") or "").strip()
        if acc == target_gpu and ep:
            matches.append(item)
    if not matches:
        return ""
    # 优先复用标准命名会话 comfy，其次非孤儿会话，最后回退首条匹配
    matches.sort(
        key=lambda s: (
            0 if str(s.get("name") or "").strip() == "comfy" else 1,
            1 if bool(s.get("orphan", False)) else 0,
        )
    )
    return str(matches[0].get("endpoint") or "").strip()


def _split_profiles_by_modality(profiles: list[str] | tuple[str, ...]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """在不触发默认回退的前提下，将工作流列表拆分为 Comfy 与 LLM 两组。"""
    seen: list[str] = []
    for raw in profiles or ():
        p = str(raw or "").strip().lower()
        if p and p not in seen:
            seen.append(p)
    if "qwen_27b" in seen and "qwen_llm" in seen:
        seen = [p for p in seen if p != "qwen_27b"]
    comfy_profs = tuple(p for p in COMFY_PROFILES if p in seen)
    llm_profs = tuple(p for p in LLM_PROFILES if p in seen)
    return comfy_profs, llm_profs


def select_best_account(
    nodes: list[AccountNodeSnapshot],
    *,
    profiles: list[str],
    preferred_gpu: str,
    active_account_id: str = "",
    min_balance_cu: float = 12.0,
    exclude_accounts: set[str] | None = None,
) -> ScheduleDecision:
    """根据账号认证、余额门槛、卡池权限、紧缺标记与实例复用情况，选出最优执行账号。"""
    target_gpu = str(preferred_gpu or "").strip().upper() or select_default_gpu(profiles)
    active_id = str(active_account_id or "").strip()
    excluded = {str(x).strip() for x in (exclude_accounts or set()) if str(x or "").strip()}
    min_cu = float(min_balance_cu)

    eligible_nodes: list[AccountNodeSnapshot] = []
    skip_reasons: list[str] = []

    for node in nodes:
        aid = str(node.account_id or "").strip()
        if not aid:
            continue
        if aid in excluded:
            skip_reasons.append(f"{aid}:已排除")
            continue
        if not node.auth_ok:
            skip_reasons.append(f"{aid}:授权失效")
            continue
        if node.balance is None or float(node.balance) < min_cu:
            skip_reasons.append(f"{aid}:余额不足")
            continue
        eligible_set = _upper_gpu_set(node.eligible_gpus)
        if eligible_set and target_gpu not in eligible_set:
            skip_reasons.append(f"{aid}:无{target_gpu}权限")
            continue
        eligible_nodes.append(node)

    if not eligible_nodes:
        detail = "；".join(skip_reasons[:4]) if skip_reasons else "号池为空"
        return ScheduleDecision(
            ok=False,
            gpu=target_gpu,
            reason=f"暂无满足 {target_gpu} 与最低 {min_cu:g} CU 要求的可用账号（{detail}）",
        )

    # 目标卡在紧缺列表时，若存在其他可用账号则优先避开紧缺节点
    non_scarce = [n for n in eligible_nodes if target_gpu not in _upper_gpu_set(n.scarce_gpus)]
    effective_nodes = non_scarce if non_scarce else eligible_nodes

    def _score_key(node: AccountNodeSnapshot) -> tuple[int, int, int, int, float, int, str]:
        aid = str(node.account_id or "").strip()
        reuse_ep = _find_reusable_endpoint(node.sessions, target_gpu)
        has_reusable = 1 if reuse_ep else 0
        sessions_list = [s for s in (node.sessions or ()) if isinstance(s, dict)]
        other_sessions = [
            s for s in sessions_list if str(s.get("endpoint") or "").strip() != reuse_ep
        ]
        no_other_sessions = 1 if len(other_sessions) == 0 else 0
        not_scarce = 0 if target_gpu in _upper_gpu_set(node.scarce_gpus) else 1
        is_active = 1 if (active_id and aid == active_id) else 0
        # 本号无阻塞优先：若可复用同卡型或当前无其余占坑且非紧缺，优先留在当前激活号免切号
        active_unblocked = 1 if (is_active and (has_reusable or (no_other_sessions and not_scarce))) else 0
        bal = float(node.balance or 0.0)
        # 注意 Python tuple 降序比较时 account_id 仅用于兜底确定性排序
        return (has_reusable, active_unblocked, no_other_sessions, not_scarce, bal, is_active, aid)

    ranked_nodes = sorted(effective_nodes, key=_score_key, reverse=True)
    best = ranked_nodes[0]
    best_id = str(best.account_id or "").strip()
    best_label = str(best.label or best_id).strip() or best_id
    reuse_ep = _find_reusable_endpoint(best.sessions, target_gpu)
    switched = bool(active_id and best_id != active_id)
    bal_val = float(best.balance or 0.0)

    if reuse_ep:
        reason = f"复用 {best_label} 已运行的 {target_gpu} 实例 ({reuse_ep})"
    elif switched:
        reason = f"智能调度至 {best_label}（{target_gpu}，余额 {bal_val:.1f} CU）"
    else:
        reason = f"当前账号 {best_label} 就绪（{target_gpu}，余额 {bal_val:.1f} CU）"

    return ScheduleDecision(
        ok=True,
        account_id=best_id,
        account_label=best_label,
        gpu=target_gpu,
        reuse_endpoint=reuse_ep,
        switched=switched,
        reason=reason,
        ranked_account_ids=tuple(str(n.account_id).strip() for n in ranked_nodes),
    )


def plan_split_cluster(
    nodes: list[AccountNodeSnapshot],
    *,
    profiles: list[str],
    comfy_gpu: str | None = None,
    llm_gpu: str | None = None,
    active_account_id: str = "",
    min_balance_cu: float = 12.0,
) -> SplitClusterPlan:
    """为双栖工作流规划跨账号异构双开方案，严格保证 LLM 与 Comfy 分属不同账号。"""
    comfy_profs, llm_profs = _split_profiles_by_modality(profiles)
    if not comfy_profs or not llm_profs:
        return SplitClusterPlan(
            ok=False,
            reason="双号异构部署要求同时包含 ComfyUI 与 LLM 工作流",
        )

    eff_llm_gpu = (
        str(llm_gpu).strip().upper()
        if str(llm_gpu or "").strip()
        else ("G4" if "qwen_llm" in llm_profs else "L4")
    )
    eff_comfy_gpu = (
        str(comfy_gpu).strip().upper()
        if str(comfy_gpu or "").strip()
        else ("G4" if any(p in ("minimax", "drama3") for p in comfy_profs) else "L4")
    )

    llm_all = select_best_account(
        nodes,
        profiles=list(llm_profs),
        preferred_gpu=eff_llm_gpu,
        active_account_id=active_account_id,
        min_balance_cu=min_balance_cu,
    )
    if not llm_all.ok:
        return SplitClusterPlan(
            ok=False,
            reason=f"LLM 槽位（{eff_llm_gpu}）无可用账号: {llm_all.reason}",
        )

    comfy_all = select_best_account(
        nodes,
        profiles=list(comfy_profs),
        preferred_gpu=eff_comfy_gpu,
        active_account_id=active_account_id,
        min_balance_cu=min_balance_cu,
    )
    if not comfy_all.ok:
        return SplitClusterPlan(
            ok=False,
            reason=f"Comfy 槽位（{eff_comfy_gpu}）无可用账号: {comfy_all.reason}",
        )

    node_by_id = {str(n.account_id).strip(): n for n in nodes if str(n.account_id or "").strip()}
    active_id = str(active_account_id or "").strip()

    # 同时评估「先锁定 Comfy 再配 LLM」与「先锁定 LLM 再配 Comfy」，避免高卡资格账号被低卡槽位抢占
    candidate_pairs: list[tuple[ScheduleDecision, ScheduleDecision, int]] = []

    # 策略 A：先选 Comfy，再排他选 LLM
    llm_after_comfy = select_best_account(
        nodes,
        profiles=list(llm_profs),
        preferred_gpu=eff_llm_gpu,
        active_account_id=active_account_id,
        min_balance_cu=min_balance_cu,
        exclude_accounts={comfy_all.account_id},
    )
    if llm_after_comfy.ok:
        candidate_pairs.append((llm_after_comfy, comfy_all, 1))

    # 策略 B：先选 LLM，再排他选 Comfy
    comfy_after_llm = select_best_account(
        nodes,
        profiles=list(comfy_profs),
        preferred_gpu=eff_comfy_gpu,
        active_account_id=active_account_id,
        min_balance_cu=min_balance_cu,
        exclude_accounts={llm_all.account_id},
    )
    if comfy_after_llm.ok:
        candidate_pairs.append((llm_all, comfy_after_llm, 0))

    # 策略 C：兜底枚举可用排名笛卡尔积，确保在非对称卡池权限下不漏解
    if not candidate_pairs:
        llm_rank_idx = {aid: i for i, aid in enumerate(llm_all.ranked_account_ids)}
        comfy_rank_idx = {aid: i for i, aid in enumerate(comfy_all.ranked_account_ids)}
        for l_id, c_id in product(llm_all.ranked_account_ids, comfy_all.ranked_account_ids):
            if l_id == c_id:
                continue
            l_node = node_by_id.get(l_id)
            c_node = node_by_id.get(c_id)
            if l_node is None or c_node is None:
                continue
            l_dec = ScheduleDecision(
                ok=True,
                account_id=l_id,
                account_label=str(l_node.label or l_id),
                gpu=eff_llm_gpu,
                reuse_endpoint=_find_reusable_endpoint(l_node.sessions, eff_llm_gpu),
                switched=bool(active_id and l_id != active_id),
                ranked_account_ids=llm_all.ranked_account_ids,
            )
            c_dec = ScheduleDecision(
                ok=True,
                account_id=c_id,
                account_label=str(c_node.label or c_id),
                gpu=eff_comfy_gpu,
                reuse_endpoint=_find_reusable_endpoint(c_node.sessions, eff_comfy_gpu),
                switched=bool(active_id and c_id != active_id),
                ranked_account_ids=comfy_all.ranked_account_ids,
            )
            rank_penalty = -(llm_rank_idx.get(l_id, 99) + comfy_rank_idx.get(c_id, 99))
            candidate_pairs.append((l_dec, c_dec, rank_penalty))

    if not candidate_pairs:
        return SplitClusterPlan(
            ok=False,
            reason="缺少两个可分别承载 LLM 与 Comfy 的独立可用付费账号",
        )

    def _pair_score(pair: tuple[ScheduleDecision, ScheduleDecision, int]) -> tuple[int, int, int, float]:
        l_dec, c_dec, tie_pref = pair
        reuse_cnt = (1 if l_dec.reuse_endpoint else 0) + (1 if c_dec.reuse_endpoint else 0)
        keeps_active = 1 if (active_id and active_id in (l_dec.account_id, c_dec.account_id)) else 0
        l_bal = float(getattr(node_by_id.get(l_dec.account_id), "balance", 0.0) or 0.0)
        c_bal = float(getattr(node_by_id.get(c_dec.account_id), "balance", 0.0) or 0.0)
        return (reuse_cnt, keeps_active, tie_pref, l_bal + c_bal)

    best_llm, best_comfy, _ = max(candidate_pairs, key=_pair_score)
    llm_hm = resolve_high_mem(eff_llm_gpu, False)
    comfy_hm = resolve_high_mem(eff_comfy_gpu, False)

    llm_slot = SplitSlotSpec(
        slot="llm",
        account_id=best_llm.account_id,
        account_label=best_llm.account_label,
        profiles=llm_profs,
        gpu=eff_llm_gpu,
        high_mem=llm_hm,
        reuse_endpoint=best_llm.reuse_endpoint,
        no_named_tunnel=False,
    )
    comfy_slot = SplitSlotSpec(
        slot="comfy",
        account_id=best_comfy.account_id,
        account_label=best_comfy.account_label,
        profiles=comfy_profs,
        gpu=eff_comfy_gpu,
        high_mem=comfy_hm,
        reuse_endpoint=best_comfy.reuse_endpoint,
        no_named_tunnel=True,
    )
    est_rate = round(
        _estimate_gpu_rate(eff_llm_gpu, llm_hm) + _estimate_gpu_rate(eff_comfy_gpu, comfy_hm),
        2,
    )
    return SplitClusterPlan(
        ok=True,
        reason=(
            f"双号异构拆分就绪：LLM={llm_slot.account_label}({eff_llm_gpu}) + "
            f"画布={comfy_slot.account_label}({eff_comfy_gpu})，预估总耗速 {est_rate:.2f} CU/h"
        ),
        llm_slot=llm_slot,
        comfy_slot=comfy_slot,
        estimated_total_rate=est_rate,
    )
