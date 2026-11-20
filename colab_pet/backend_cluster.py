"""Colab 桌宠多账号算力池探测、智能选号/容灾换号与异构双开集群编排子服务。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import json
import re
import threading
from typing import TYPE_CHECKING, Any

from colab_core.catalog import get_fallback_gpu_rate
from colab_core.runtime import (
    DEFAULT_MEASURED_GPU_RATES,
    DEFAULT_PREFS,
    PROFILE_LABELS,
    SCARCE_GPU_TTL_SEC,
    build_ctl_up_args,
    calc_phase_dl_total,
    classify_deploy_failure,
    consume_deploy_line,
    default_baseline,
    default_deploy_state,
    format_elapsed_cn,
    format_pool_tier_summary,
    gpu_rate_key,
    infer_account_tier_and_baseline,
    kill_process_tree,
    parse_llm_boot_params,
    restore_ledger_for_account,
    snapshot_current_ledger,
    upsert_account_profile,
)
from colab_security import redact_secrets

if TYPE_CHECKING:
    if __package__:
        from .colab_backend import ColabPetBackend
    else:
        from colab_backend import ColabPetBackend


def _pool_sched() -> Any:
    """延迟加载纯函数调度核，解耦初始化顺序并便于单元测试隔离导入。"""
    from colab_core import pool_scheduler

    return pool_scheduler


def _parse_pool_probe_stdout(stdout: str) -> dict[str, Any] | None:
    """解析 `pool-probe` 子进程输出的标准帧或裸 JSON 字典。"""
    txt = str(stdout or "").strip()
    if not txt:
        return None
    m = re.search(r"<<<COLAB_POOL_PROBE>>>(.*?)<<<END_COLAB_POOL_PROBE>>>", txt, re.S)
    raw = m.group(1).strip() if m else txt
    if not raw.startswith("{"):
        return None
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def _first_non_empty(*values: Any) -> str:
    """依次返回首个非空字符串值，均为空时返回空字符串。"""
    for val in values:
        text = str(val or "").strip()
        if text:
            return text
    return ""


def _parse_upper_str_list(raw: Any) -> list[str] | None:
    """将 GPU/TPU 字符串列表规范化为大写列表，非序列时返回 None。"""
    if not isinstance(raw, (list, tuple)):
        return None
    return [str(item).upper() for item in raw if item]


def _safe_round_float(raw: Any, default: float | None = None) -> float | None:
    """安全将任意输入转换为保留两位小数的浮点数。"""
    if raw is None:
        return default
    try:
        return round(float(raw), 2)
    except (TypeError, ValueError):
        return default


def _build_initial_cluster_slots_dict(llm_slot: Any, comfy_slot: Any) -> dict[str, dict[str, Any]]:
    """根据规划好的双槽规格构建持久化 `cluster_slots` 初始字典。"""
    return {
        skey: {
            "account_id": sp.account_id,
            "label": sp.account_label,
            "gpu": sp.gpu,
            "high_mem": bool(sp.high_mem),
            "profiles": list(sp.profiles),
            "session": "comfy",
            "endpoint": str(sp.reuse_endpoint or ""),
            "rate_hourly": 0.0,
            "tunnel_url": None,
            "online": False,
        }
        for skey, sp in (("llm", llm_slot), ("comfy", comfy_slot))
    }


@dataclass
class _SplitClusterCtx:
    """跨账号异构双开部署的运行态上下文容器。"""

    cfg: dict[str, Any]
    parsed: dict[str, Any]
    display_profiles: list[str]
    profile_label: str
    recreate: bool
    minimax_r2v: bool
    minimax_pe: bool
    llm_quant: str
    llm_ctx: int
    llm_mmproj: bool
    llm27_quant: str
    llm27_ctx: int
    llm27_mmproj: bool
    prefs_copy: dict[str, Any]
    tunnel_pat: Any
    start_epoch: int
    initial_deploy: dict[str, Any]
    defer_pool_refresh: bool
    slot_states: dict[str, dict[str, Any]]
    progress_lock: threading.Lock = field(default_factory=threading.Lock)
    last_emit_m: float = 0.0


class ClusterPoolService:
    """负责多号算力池并发探测、智能调度与故障换号、异构双开部署及分槽启停控制。"""

    def __init__(self, backend: ColabPetBackend) -> None:
        self._b = backend
        self._store = backend._store
        self._secrets = backend._secrets
        self._bus = backend._bus
        self._ctx = backend._ctx

        self._pool_probe_by_acct: dict[str, dict[str, Any]] = {}
        self._acct_scarce_gpus: dict[tuple[str, str], dict[str, Any]] = {}
        self._slot_last_activity_m: dict[str, float] = {}
        self._slot_idle_warned: dict[str, str | None] = {}
        self._last_pool_refresh_m: float = 0.0
        self._last_cluster_probe_m: float = 0.0
        self._slot_miss_counts: dict[str, int] = {}
        self._split_procs_lock = threading.Lock()
        self._split_deploy_procs: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # 账号紧缺标记与只读状态访问器（锁内只做纯内存读写）
    # ------------------------------------------------------------------

    def _prune_scarce_locked(self, now_ts: int | None = None) -> None:
        """清理已超出 TTL 的账号级显卡紧缺记录（需已持主状态锁）。"""
        cur = int(self._ctx.now()) if now_ts is None else int(now_ts)
        expired = [
            k
            for k, v in self._acct_scarce_gpus.items()
            if (cur - int(v.get("ts") or 0)) > SCARCE_GPU_TTL_SEC
        ]
        for k in expired:
            self._acct_scarce_gpus.pop(k, None)

    def mark_account_gpu_scarce(
        self, acct_id: str, gpu: str, reason: str = "", kind: str = "stockout"
    ) -> None:
        """记录指定账号与卡型的临时紧缺/配额受限状态，避免调度器重复命中。"""
        aid = str(acct_id or "").strip()
        gpu_up = str(gpu or "").strip().upper()
        if not aid or not gpu_up:
            return
        with self._store.lock:
            self._prune_scarce_locked()
            self._acct_scarce_gpus[(aid, gpu_up)] = {
                "ts": int(self._ctx.now()),
                "reason": str(reason or "算力紧缺"),
                "kind": str(kind or "stockout"),
            }

    def clear_account_gpu_scarce(self, acct_id: str, gpu: str) -> None:
        """在某账号成功开机后清除其对应卡型的紧缺标记。"""
        aid = str(acct_id or "").strip()
        gpu_up = str(gpu or "").strip().upper()
        if not aid or not gpu_up:
            return
        with self._store.lock:
            self._acct_scarce_gpus.pop((aid, gpu_up), None)

    def get_cluster_slots(self) -> dict[str, dict[str, Any]]:
        """返回当前持久化/活跃的异构分槽副本。"""
        raw = self._store.get_cluster_slots()
        return {k: dict(v) for k, v in raw.items() if k in ("llm", "comfy") and isinstance(v, dict)}

    def has_active_slots(self) -> bool:
        """判断是否处于跨账号异构集群模式（至少存在一个活跃分槽）。"""
        return bool(self.get_cluster_slots())

    def get_slot(self, slot: str) -> dict[str, Any] | None:
        """获取指定分槽（`llm` 或 `comfy`）配置副本。"""
        key = str(slot or "").strip().lower()
        if key not in ("llm", "comfy"):
            return None
        return self.get_cluster_slots().get(key)

    def resolve_account_for_endpoint(self, ep: str = "", base_url: str = "") -> str:
        """根据 Endpoint 或隧道地址解析归属的集群账号 ID（非集群/单号模式返回空串）。"""
        target_ep = str(ep or "").strip()
        clean_base = str(base_url or "").strip().rstrip("/")
        slots = self.get_cluster_slots()
        for slot_val in slots.values():
            s_ep = str(slot_val.get("endpoint") or "").strip()
            s_url = str(slot_val.get("tunnel_url") or "").strip().rstrip("/")
            if target_ep and s_ep and target_ep == s_ep:
                return str(slot_val.get("account_id") or "").strip()
            if clean_base and s_url and clean_base == s_url:
                return str(slot_val.get("account_id") or "").strip()
        return ""

    def touch_slot_activity(self, slot: str | None = None) -> None:
        """刷新指定分槽（或双槽）的最近活跃单调时间戳，隔离高耗画布与轻量 LLM 闲置计时。"""
        now_m = self._ctx.monotonic()
        with self._store.lock:
            keys = (slot,) if slot in ("comfy", "llm") else ("comfy", "llm")
            for skey in keys:
                self._slot_last_activity_m[skey] = now_m
                self._slot_idle_warned[skey] = None

    # ------------------------------------------------------------------
    # 号池并发探测（pool-probe）与隔离账本更新
    # ------------------------------------------------------------------

    def probe_accounts_raw(self, acct_ids: list[str]) -> dict[str, Any]:
        """在无锁状态下调用 `CLI_BRIDGE_SCRIPT pool-probe` 一次性并发探测多号状态。"""
        clean_ids = [str(aid).strip() for aid in acct_ids if str(aid or "").strip()]
        if not clean_ids:
            return {"ok": False, "accounts": {}, "error": "无待探测账号"}
        colab_py = self._ctx.get_colab_py()
        if not colab_py.exists():
            return {"ok": False, "accounts": {}, "error": f"未找到 colab-cli 运行环境: {colab_py}"}

        payload = json.dumps({"account_ids": clean_ids}, ensure_ascii=False)
        proc = self._ctx.run_hidden(
            [str(colab_py), str(self._ctx.cli_bridge_script), "pool-probe"],
            timeout=35,
            input_text=payload,
        )
        if proc.returncode != 0:
            err = redact_secrets((proc.stderr or proc.stdout or "").strip())
            tail = err.splitlines()[-1] if err else f"退出码 {proc.returncode}"
            return {"ok": False, "accounts": {}, "error": tail}

        parsed = _parse_pool_probe_stdout(proc.stdout or "")
        if not isinstance(parsed, dict) or not isinstance(parsed.get("accounts"), dict):
            return {"ok": False, "accounts": {}, "error": "pool-probe 返回格式异常"}
        return parsed

    @staticmethod
    def _parse_single_acct_probe_info(info: dict[str, Any]) -> dict[str, Any]:
        """将单账号原始探针字典规范化为强类型字段集。"""
        auth_ok = bool(info.get("auth_ok", info.get("ok", True)))
        if info.get("auth_missing") or info.get("auth_expired"):
            auth_ok = False
        bal = _safe_round_float(info.get("balance"), default=None)
        rate = float(_safe_round_float(info.get("rate_hourly"), default=0.0) or 0.0)
        sessions = [dict(s) for s in (info.get("sessions") or []) if isinstance(s, dict)]
        raw_cnt = info.get("assignments_count")
        try:
            assign_cnt = int(raw_cnt if raw_cnt is not None else len(sessions))
        except (TypeError, ValueError):
            assign_cnt = len(sessions)

        fallback_run = assign_cnt > 0 or rate > 0.01 or bool(sessions)
        is_running = bool(info.get("is_running", info.get("last_is_running", fallback_run)))
        return {
            "auth_ok": auth_ok,
            "balance": bal,
            "rate_hourly": rate,
            "sessions": sessions,
            "assignments_count": assign_cnt,
            "is_running": is_running,
            "eligible_gpus": _parse_upper_str_list(info.get("eligible_gpus")),
            "ineligible_gpus": _parse_upper_str_list(info.get("ineligible_gpus")),
            "eligible_tpus": _parse_upper_str_list(info.get("eligible_tpus")),
            "subscription_tier": str(info.get("subscription_tier") or "").strip(),
            "error": str(info.get("error") or ""),
        }

    def _update_single_acct_ledger_locked(
        self,
        aid: str,
        parsed: dict[str, Any],
        ledgers: dict[str, Any],
        prof_by_id: dict[str, dict[str, Any]],
        now_ts: int,
    ) -> None:
        """将单个账号探针写入内存缓存与隔离账本 `account_ledgers[aid]`，绝不覆写主账本顶级字段。"""
        acct_ledger = dict(ledgers.get(aid) or {})
        bal = parsed["balance"]
        peak_cands = (bal, acct_ledger.get("peak_balance"), acct_ledger.get("day_start_balance"))
        peak_nums = [float(x) for x in peak_cands if isinstance(x, (int, float))]
        peak_bal = round(max(peak_nums), 4) if peak_nums else None

        eg_list = parsed["eligible_gpus"]
        ig_list = parsed["ineligible_gpus"]
        et_list = parsed["eligible_tpus"]
        sub_tier = parsed["subscription_tier"]
        tier_key, _tlbl, inferred_base = infer_account_tier_and_baseline(
            bal,
            peak_balance=peak_bal,
            eligible_gpus=eg_list if eg_list is not None else acct_ledger.get("last_eligible_gpus"),
            ineligible_gpus=ig_list if ig_list is not None else acct_ledger.get("last_ineligible_gpus"),
            explicit_tier=sub_tier or str(acct_ledger.get("subscription_tier") or ""),
        )

        self._pool_probe_by_acct[aid] = {
            **parsed,
            "peak_balance": peak_bal,
            "tier": tier_key,
            "updated_ts": now_ts,
        }

        if bal is not None:
            acct_ledger["last_balance"] = bal
        if peak_bal is not None:
            acct_ledger["peak_balance"] = peak_bal
        if tier_key != "unknown":
            acct_ledger["baseline_balance"] = inferred_base
        acct_ledger["last_rate_hourly"] = parsed["rate_hourly"]
        acct_ledger["last_assignments_count"] = parsed["assignments_count"]
        acct_ledger["last_sessions"] = parsed["sessions"]
        acct_ledger["last_is_running"] = parsed["is_running"]
        if eg_list is not None:
            acct_ledger["last_eligible_gpus"] = eg_list
        if ig_list is not None:
            acct_ledger["last_ineligible_gpus"] = ig_list
        if et_list is not None:
            acct_ledger["last_eligible_tpus"] = et_list
        if sub_tier:
            acct_ledger["subscription_tier"] = sub_tier
        acct_ledger["last_updated_ts"] = now_ts
        ledgers[aid] = acct_ledger

        if aid in prof_by_id:
            upsert_account_profile(
                self._store.state,
                aid,
                "***",
                balance=bal,
                is_running=parsed["is_running"],
                tier=tier_key if tier_key != "unknown" else "",
            )

    @staticmethod
    def _sync_cluster_slots_for_acct_locked(
        aid: str, parsed: dict[str, Any], cluster_map: dict[str, Any]
    ) -> None:
        """仅按已绑定 Endpoint 精确刷新分槽状态；原实例消失时置 `binding_invalid`，严禁自动认领其他实例。"""
        sessions = parsed["sessions"]
        rate = float(parsed["rate_hourly"])
        auth_ok = bool(parsed["auth_ok"])
        for s_key in ("llm", "comfy"):
            slot_obj = cluster_map.get(s_key)
            if not isinstance(slot_obj, dict) or str(slot_obj.get("account_id") or "").strip() != aid:
                continue
            cur_ep = str(slot_obj.get("endpoint") or "").strip()
            matched_s = next(
                (s for s in sessions if cur_ep and str(s.get("endpoint") or "").strip() == cur_ep),
                None,
            )
            if matched_s:
                slot_obj["endpoint"] = cur_ep
                slot_obj["binding_invalid"] = False
                if matched_s.get("accelerator"):
                    slot_obj["gpu"] = str(matched_s["accelerator"]).upper()
                if rate > 0.01:
                    slot_obj["rate_hourly"] = rate
            elif auth_ok:
                slot_obj["binding_invalid"] = bool(cur_ep)
                slot_obj["online"] = False
                slot_obj["rate_hourly"] = 0.0
            elif rate > 0.01:
                slot_obj["rate_hourly"] = rate

    def apply_pool_probe_results(self, accounts_data: dict[str, Any]) -> int:
        """将探测结果隔离写入 `state['account_ledgers'][aid]` 与内存缓存，绝不污染当前主账本字段。"""
        now_ts = int(self._ctx.now())
        updated_cnt = 0

        with self._store.lock:
            ledgers = self._store.state.setdefault("account_ledgers", {})
            profiles = self._store.state.get("account_profiles") or []
            prof_by_id = {
                str(p.get("id") or ""): p for p in profiles if isinstance(p, dict) and p.get("id")
            }
            cluster_raw = self._store.state.get("cluster_slots")
            cluster_map = cluster_raw if isinstance(cluster_raw, dict) else {}

            for raw_aid, info in (accounts_data or {}).items():
                aid = str(raw_aid or "").strip()
                if not aid or not isinstance(info, dict):
                    continue
                updated_cnt += 1
                parsed = self._parse_single_acct_probe_info(info)
                self._update_single_acct_ledger_locked(aid, parsed, ledgers, prof_by_id, now_ts)
                self._sync_cluster_slots_for_acct_locked(aid, parsed, cluster_map)

            self._last_pool_refresh_m = self._ctx.monotonic()
        return updated_cnt

    def refresh_account_pool(self, *, emit: bool = True) -> dict[str, Any]:
        """并发探测全部已绑定账号的余额、解锁卡池与活跃会话，并刷新快照。"""
        b = self._b
        with self._store.lock:
            profiles = [dict(p) for p in (self._store.state.get("account_profiles") or []) if isinstance(p, dict)]
            acct_ids = [str(p.get("id") or "").strip() for p in profiles if p.get("id")]

        if not acct_ids:
            snap = b._rebuild_and_emit_snapshot() if emit else b.get_cached_snapshot()
            return {"ok": False, "snapshot": snap, "message": "暂无已绑定的 Google 账号"}

        try:
            probe_res = self.probe_accounts_raw(acct_ids)
        except Exception as exc:
            err_s = redact_secrets(str(exc))
            snap = b._rebuild_and_emit_snapshot() if emit else b.get_cached_snapshot()
            return {"ok": False, "snapshot": snap, "message": f"探测算力池异常: {err_s}"}

        if not probe_res.get("ok") and not probe_res.get("accounts"):
            err_s = redact_secrets(str(probe_res.get("error") or "探测失败"))
            snap = b._rebuild_and_emit_snapshot() if emit else b.get_cached_snapshot()
            return {"ok": False, "snapshot": snap, "message": f"算力池探测失败: {err_s}"}

        cnt = self.apply_pool_probe_results(probe_res.get("accounts") or {})
        self._store.save()
        if emit:
            snap = b._rebuild_and_emit_snapshot()
        else:
            with self._store.lock:
                snap = b._build_snapshot(
                    live_data=None,
                    stale=bool(b.last_snapshot.get("stale", False)),
                    error_msg=b.last_snapshot.get("error_msg"),
                )
                b.last_snapshot = snap
        healthy = int((snap.get("cluster") or {}).get("pool_healthy_count") or 0)
        return {
            "ok": True,
            "snapshot": snap,
            "probed_count": cnt,
            "message": f"🔄 已刷新 {cnt} 个账号算力池（健康付费号 {healthy} 个）",
        }

    def is_pool_fresh(self, max_age_sec: float = 45.0) -> bool:
        """判断当前多号算力池缓存是否均完备且未超出有效新鲜期。"""
        with self._store.lock:
            profiles = [dict(p) for p in (self._store.state.get("account_profiles") or []) if isinstance(p, dict)]
            acct_ids = [str(p.get("id") or "").strip() for p in profiles if p.get("id")]
            if len(acct_ids) < 2:
                return True
            now_m = self._ctx.monotonic()
            all_cached = all(
                aid in self._pool_probe_by_acct
                and self._pool_probe_by_acct[aid].get("eligible_gpus") is not None
                and self._pool_probe_by_acct[aid].get("balance") is not None
                for aid in acct_ids
            )
            return bool(all_cached and (now_m - self._last_pool_refresh_m) < max_age_sec)

    def ensure_pool_fresh_if_needed(self, max_age_sec: float = 45.0) -> bool:
        """开机编排前按需刷新号池；当缺乏真实 CLI 或处于离线单测桩时平滑降级到缓存账本。"""
        if self.is_pool_fresh(max_age_sec=max_age_sec):
            return False
        try:
            res = self.refresh_account_pool(emit=False)
            return bool(isinstance(res, dict) and res.get("ok"))
        except Exception:
            return False

    # ------------------------------------------------------------------
    # 快照组装（snapshot["cluster"] 与 onboarding["accounts"] 增强）
    # ------------------------------------------------------------------

    def _extract_main_acct_source_locked(self) -> dict[str, Any]:
        """提取当前激活账号主账本原始字段（需已持主状态锁）。"""
        st = self._store.state
        rate = float(st.get("last_rate_hourly") or 0.0)
        sessions = list(st.get("last_sessions") or [])
        return {
            "bal": st.get("last_balance"),
            "peak_bal": st.get("peak_balance"),
            "rate": rate,
            "sessions": sessions,
            "eg": st.get("last_eligible_gpus"),
            "ig": st.get("last_ineligible_gpus"),
            "sub_tier": str(st.get("subscription_tier") or ""),
            "auth_ok": not bool(self._b._auth_missing),
            "is_run": bool(int(st.get("last_assignments_count") or 0) > 0 or rate > 0.01 or sessions),
            "ledger_date": str(st.get("date") or ""),
            "day_cons": float(st.get("day_consumed_cu") or 0.0),
            "day_start": st.get("day_start_balance"),
        }

    @staticmethod
    def _extract_cached_acct_source(ledger: dict[str, Any], cached: dict[str, Any]) -> dict[str, Any]:
        """合并内存探针缓存与隔离账本日消耗字段。"""
        peak = cached.get("peak_balance")
        ig = cached.get("ineligible_gpus")
        return {
            "bal": cached.get("balance"),
            "peak_bal": peak if peak is not None else ledger.get("peak_balance"),
            "rate": float(cached.get("rate_hourly") or 0.0),
            "sessions": list(cached.get("sessions") or []),
            "eg": cached.get("eligible_gpus"),
            "ig": ig if ig is not None else ledger.get("last_ineligible_gpus"),
            "sub_tier": _first_non_empty(cached.get("subscription_tier"), ledger.get("subscription_tier")),
            "auth_ok": bool(cached.get("auth_ok", True)),
            "is_run": bool(cached.get("is_running", False)),
            "ledger_date": str(ledger.get("date") or ""),
            "day_cons": float(ledger.get("day_consumed_cu") or 0.0),
            "day_start": ledger.get("day_start_balance"),
        }

    def _pick_raw_acct_source_locked(
        self,
        aid: str,
        card: dict[str, Any],
        active_id: str,
        ledger: dict[str, Any],
        cached: dict[str, Any],
    ) -> dict[str, Any]:
        """按时间戳比较从主账本、内存探针或隔离账本中选择最新原始视图（需已持主状态锁）。"""
        cached_ts = int(cached.get("updated_ts") or 0)
        ledger_ts = int(ledger.get("last_updated_ts") or 0)
        main_ts = int(self._store.state.get("last_updated_ts") or 0) if aid == active_id else 0

        if aid == active_id and main_ts >= cached_ts and main_ts >= ledger_ts:
            return self._extract_main_acct_source_locked()
        if cached and cached_ts >= ledger_ts:
            return self._extract_cached_acct_source(ledger, cached)
        bal_val = ledger.get("last_balance") if ledger.get("last_balance") is not None else card.get("last_balance")
        return {
            "bal": bal_val,
            "peak_bal": ledger.get("peak_balance"),
            "rate": float(ledger.get("last_rate_hourly") or 0.0),
            "sessions": list(ledger.get("last_sessions") or []),
            "eg": ledger.get("last_eligible_gpus"),
            "ig": ledger.get("last_ineligible_gpus"),
            "sub_tier": str(ledger.get("subscription_tier") or ""),
            "auth_ok": bool(cached.get("auth_ok", True)) if cached else True,
            "is_run": bool(ledger.get("last_is_running", card.get("last_is_running", False))),
            "ledger_date": str(ledger.get("date") or ""),
            "day_cons": float(ledger.get("day_consumed_cu") or 0.0),
            "day_start": ledger.get("day_start_balance"),
        }

    def _resolve_acct_view_locked(self, aid: str, card: dict[str, Any], active_id: str) -> dict[str, Any]:
        """在已持主状态锁下合并单个账号的内存探针、隔离账本与主账本（仅当前激活号）视图。"""
        ledgers = self._store.state.get("account_ledgers") or {}
        ledger = ledgers.get(aid) if isinstance(ledgers.get(aid), dict) else {}
        cached = self._pool_probe_by_acct.get(aid) or {}
        raw = self._pick_raw_acct_source_locked(aid, card, active_id, ledger, cached)

        bal = raw["bal"]
        eg_list = _parse_upper_str_list(raw["eg"])
        ig_list = _parse_upper_str_list(raw["ig"])
        peak_vals = [float(x) for x in (bal, raw["peak_bal"], raw["day_start"]) if isinstance(x, (int, float))]
        eff_peak = round(max(peak_vals), 4) if peak_vals else None
        today_str = datetime.now().strftime("%Y-%m-%d")
        today_cons = round(max(0.0, float(raw["day_cons"])), 4) if raw["ledger_date"] == today_str else 0.0
        bal_rounded = round(float(bal), 2) if bal is not None else None

        tier_key, tier_label, inferred_base = infer_account_tier_and_baseline(
            bal_rounded,
            peak_balance=eff_peak,
            eligible_gpus=eg_list,
            ineligible_gpus=ig_list,
            explicit_tier=raw["sub_tier"] or str(card.get("tier") or ""),
        )
        return {
            "balance": bal_rounded,
            "peak_balance": eff_peak,
            "rate_hourly": round(float(raw["rate"]), 2),
            "sessions": raw["sessions"],
            "eligible_gpus": eg_list,
            "ineligible_gpus": ig_list,
            "tier": tier_key,
            "tier_label": tier_label,
            "baseline_balance": inferred_base,
            "day_consumed_cu": today_cons,
            "auth_ok": raw["auth_ok"],
            "is_running": raw["is_run"],
        }

    def build_pool_nodes(self) -> list[Any]:
        """构建供 `colab_core.pool_scheduler` 使用的不可变 `AccountNodeSnapshot` 列表。"""
        ps = _pool_sched()
        nodes: list[Any] = []
        with self._store.lock:
            self._prune_scarce_locked()
            profiles = [dict(p) for p in (self._store.state.get("account_profiles") or []) if isinstance(p, dict)]
            active_id = self._store.get_active_account_id()
            for card in profiles:
                aid = str(card.get("id") or "").strip()
                if not aid:
                    continue
                view = self._resolve_acct_view_locked(aid, card, active_id)
                scarce = tuple(sorted(gpu for (s_aid, gpu) in self._acct_scarce_gpus.keys() if s_aid == aid))
                nodes.append(
                    ps.AccountNodeSnapshot(
                        account_id=aid,
                        label=str(card.get("label") or aid),
                        balance=view["balance"],
                        rate_hourly=float(view["rate_hourly"]),
                        eligible_gpus=tuple(view["eligible_gpus"] or ()),
                        scarce_gpus=scarce,
                        sessions=tuple(dict(s) for s in view["sessions"] if isinstance(s, dict)),
                        auth_ok=bool(view["auth_ok"]),
                    )
                )
        return nodes

    def _summarize_pool_accounts_locked(
        self, profiles: list[dict[str, Any]], active_id: str
    ) -> dict[str, Any]:
        """在已持主状态锁下聚合全部已绑定账号的余额、基线、当日消耗与健康付费号数量。"""
        total_pool_bal = 0.0
        total_pool_base = 0.0
        total_day_consumed = 0.0
        healthy_cnt = 0
        acct_views: dict[str, dict[str, Any]] = {}
        tier_keys: list[str] = []

        for card in profiles:
            aid = str(card.get("id") or "").strip()
            if not aid:
                continue
            view = self._resolve_acct_view_locked(aid, card, active_id)
            acct_views[aid] = view
            tier_keys.append(str(view.get("tier") or "unknown"))
            bal = view["balance"]
            base_v = float(view.get("baseline_balance") or 0.0)
            if bal is not None:
                total_pool_bal += float(bal)
                total_pool_base += max(float(bal), base_v) if base_v > 0 else default_baseline(float(bal))
                if view["auth_ok"] and float(bal) >= 12.0:
                    healthy_cnt += 1
            elif base_v > 0:
                total_pool_base += base_v
            total_day_consumed += float(view.get("day_consumed_cu") or 0.0)

        return {
            "total_pool_bal": total_pool_bal,
            "total_pool_base": total_pool_base,
            "total_day_consumed": total_day_consumed,
            "healthy_cnt": healthy_cnt,
            "acct_views": acct_views,
            "tier_keys": tier_keys,
        }

    @staticmethod
    def _build_cluster_slot_views(
        slots_src: dict[str, dict[str, Any]],
        prof_by_id: dict[str, dict[str, Any]],
        acct_views: dict[str, dict[str, Any]],
        tunnel_cache: dict[str, str],
        rates_map: dict[str, Any],
        c_state: dict[str, Any],
    ) -> dict[str, dict[str, Any]]:
        """构建 `snapshot['cluster']['slots']` 中各分槽展示视图。"""
        slots_out: dict[str, dict[str, Any]] = {}
        for s_key in ("llm", "comfy"):
            s_raw = slots_src.get(s_key)
            if not s_raw:
                continue
            s_aid = str(s_raw.get("account_id") or "").strip()
            s_ep = str(s_raw.get("endpoint") or "").strip()
            s_gpu = str(s_raw.get("gpu") or "").strip().upper()
            s_label = _first_non_empty(s_raw.get("label"), (prof_by_id.get(s_aid) or {}).get("label"), s_aid)
            c_url_fallback = c_state.get("llm_tunnel_url") if s_key == "llm" else c_state.get("tunnel_url")
            s_url = _first_non_empty(s_raw.get("tunnel_url"), tunnel_cache.get(s_ep) if s_ep else "", c_url_fallback) or None

            c_online_key = "llm_online" if s_key == "llm" else "comfy_online"
            s_online = bool(s_raw.get("online", False) or c_state.get(c_online_key))

            view = acct_views.get(s_aid) or {}
            s_rate = float(s_raw.get("rate_hourly") or 0.0)
            if float(view.get("rate_hourly") or 0.0) > 0.01:
                s_rate = float(view["rate_hourly"])
            if s_rate <= 0.01 and (s_online or s_ep):
                is_hm = bool(s_raw.get("high_mem"))
                rkey = gpu_rate_key(s_gpu, "HIGH_RAM" if is_hm else "STANDARD")
                s_rate = float(rates_map.get(rkey or "") or get_fallback_gpu_rate(s_gpu, high_mem=is_hm))

            slots_out[s_key] = {
                "account_id": s_aid,
                "label": s_label,
                "gpu": s_gpu,
                "endpoint": s_ep,
                "rate_hourly": round(s_rate, 2),
                "tunnel_url": s_url,
                "online": s_online,
            }
        return slots_out

    @staticmethod
    def _enrich_onboarding_accounts(
        raw_accounts: Any,
        acct_views: dict[str, dict[str, Any]],
        slots_out: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """为 `onboarding['accounts']` 注入卡池、会员档位与当前异构集群角色字段。"""
        role_by_aid: dict[str, str] = {}
        for role_name in ("llm", "comfy"):
            slot_rec = slots_out.get(role_name)
            if slot_rec and slot_rec.get("account_id"):
                role_by_aid[str(slot_rec["account_id"])] = role_name

        enriched: list[dict[str, Any]] = []
        for raw_acct in raw_accounts or []:
            if not isinstance(raw_acct, dict):
                continue
            item = dict(raw_acct)
            aid = str(item.get("id") or "").strip()
            view = acct_views.get(aid)
            if view is not None:
                if view["balance"] is not None:
                    item["last_balance"] = view["balance"]
                item["last_is_running"] = bool(view["is_running"])
                item["eligible_gpus"] = view["eligible_gpus"]
                item["ineligible_gpus"] = view.get("ineligible_gpus")
                item["tier"] = view.get("tier") or "unknown"
                item["tier_label"] = view.get("tier_label") or "Colab"
                item["baseline_balance"] = round(float(view.get("baseline_balance") or 1000.0), 2)
            else:
                item.setdefault("eligible_gpus", None)
                item.setdefault("tier", "unknown")
                item.setdefault("tier_label", "Colab")
            item["cluster_role"] = role_by_aid.get(aid)
            enriched.append(item)
        return enriched

    def build_cluster_snapshot_and_enrich_onboarding(
        self,
        onboarding: dict[str, Any],
        *,
        current_rate_hourly: float,
        active_account_id: str,
        comfy_state: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """生成 `snapshot['cluster']` 字典，并为 `onboarding['accounts']` 注入卡池与集群角色字段。"""
        b = self._b
        ob_out = dict(onboarding)
        c_state = comfy_state if isinstance(comfy_state, dict) else {}

        with self._store.lock:
            profiles = [dict(p) for p in (self._store.state.get("account_profiles") or []) if isinstance(p, dict)]
            active_id = str(active_account_id or self._store.get_active_account_id()).strip()
            rates_map = dict(self._store.state.get("measured_gpu_rates") or DEFAULT_MEASURED_GPU_RATES)
            tunnel_cache = self._store.get_tunnel_cache()
            slots_src = self.get_cluster_slots()
            prof_by_id = {str(p.get("id") or ""): p for p in profiles if p.get("id")}
            summary = self._summarize_pool_accounts_locked(profiles, active_id)
            slots_out = self._build_cluster_slot_views(
                slots_src, prof_by_id, summary["acct_views"], tunnel_cache, rates_map, c_state
            )
            deploy_active = bool(b._deploy_state.get("active"))

        has_live_slot = any(
            bool(s.get("online") or s.get("endpoint") or s.get("tunnel_url") or float(s.get("rate_hourly") or 0.0) > 0.01)
            for s in slots_out.values()
        )
        cluster_active = bool(slots_out) and ((not deploy_active) or has_live_slot)
        healthy_cnt = summary["healthy_cnt"]
        can_split = len(profiles) >= 2 and healthy_cnt >= 2
        total_rate = (
            round(sum(float(s.get("rate_hourly") or 0.0) for s in slots_out.values()), 2)
            if cluster_active
            else round(float(current_rate_hourly or 0.0), 2)
        )

        enriched_accounts = self._enrich_onboarding_accounts(ob_out.get("accounts"), summary["acct_views"], slots_out)
        ob_out["accounts"] = enriched_accounts

        total_pool_bal = summary["total_pool_bal"]
        total_pool_base = summary["total_pool_base"]
        eff_pool_base = round(total_pool_base, 2) if total_pool_base > 0 else round(max(100.0, total_pool_bal), 2)
        pool_bal_pct = (
            max(0.0, min(100.0, round((total_pool_bal / eff_pool_base) * 100.0, 2)))
            if enriched_accounts and eff_pool_base > 0
            else 0.0
        )
        cluster_info = {
            "active": cluster_active,
            "can_split": can_split,
            "pool_account_count": len(enriched_accounts),
            "pool_healthy_count": healthy_cnt,
            "total_pool_balance": round(total_pool_bal, 2),
            "total_pool_baseline": eff_pool_base,
            "pool_balance_pct": pool_bal_pct,
            "pool_tier_summary": format_pool_tier_summary(summary["tier_keys"]),
            "total_day_consumed_cu": round(summary["total_day_consumed"], 2),
            "total_rate_hourly": total_rate,
            "slots": slots_out if cluster_active else {},
        }
        return cluster_info, ob_out

    # ------------------------------------------------------------------
    # 部署期自动容灾切号（不触发 is_deploying 互斥死锁）
    # ------------------------------------------------------------------

    def switch_account_during_deploy(self, target_acct_id: str) -> tuple[bool, int, str]:
        """在已持 `_deploy_lock` 的故障转移流程中切换活跃账号凭据与隔离账本，保留当前 `_deploy_state`。"""
        b = self._b
        target_id = re.sub(r"[^a-zA-Z0-9_-]", "", str(target_acct_id or ""))[:24]
        if not target_id:
            return False, 0, "无效的备用账号 ID"

        with self._store.lock:
            prev_id = self._store.get_active_account_id()
            cur_epoch = b._account_epoch

        if prev_id == target_id:
            return True, cur_epoch, ""

        colab_py = self._ctx.get_colab_py()
        if colab_py.is_file():
            payload = json.dumps(
                {"op": "switch", "target_acct_id": target_id, "prev_acct_id": prev_id},
                ensure_ascii=False,
            )
            try:
                proc = self._ctx.run_hidden(
                    [str(colab_py), str(self._ctx.cli_bridge_script), "account-op"],
                    timeout=25,
                    input_text=payload,
                )
                if proc.returncode != 0:
                    err = redact_secrets((proc.stderr or proc.stdout or "").strip()).splitlines()
                    return False, cur_epoch, (err[-1] if err else f"退出码 {proc.returncode}")
            except OSError as exc:
                # 兼容测试环境中 0 字节 fake_py 未 mock _run_hidden 的情况
                b._diag.warn("cluster", f"切换备用账号进程不可执行，回退至 scoped-cli 模式: {exc}")
            except Exception as exc:
                return False, cur_epoch, redact_secrets(str(exc))

        with self._store.lock:
            b._account_epoch += 1
            new_epoch = b._account_epoch
            b._auth_missing = False
            b._auth_expired = False
            if prev_id and prev_id != target_id:
                self._store.state.setdefault("account_ledgers", {})[prev_id] = snapshot_current_ledger(
                    self._store.state
                )
            restore_ledger_for_account(self._store.state, target_id)
            prefs = self._store.state.setdefault("prefs", dict(DEFAULT_PREFS))
            prefs["active_account_id"] = target_id
            b._clear_ephemeral_caches_locked()
            self._store.save()
        return True, new_epoch, ""

    # ------------------------------------------------------------------
    # 跨账号异构双开并行部署（start_split_cluster 及其 4 阶段拆解）
    # ------------------------------------------------------------------

    def _register_split_proc(self, slot_name: str, proc: Any) -> None:
        """登记某槽位正在执行的部署子进程句柄以便统一中止。"""
        with self._split_procs_lock:
            self._split_deploy_procs[slot_name] = proc

    def cancel_split_deploys(self) -> None:
        """中止跨账号并行部署中的所有子进程树。"""
        with self._split_procs_lock:
            procs = list(self._split_deploy_procs.values())
        for p in procs:
            kill_process_tree(p)

    def _validate_split_plan_and_preflight(self, plan: Any) -> tuple[Any, Any, dict[str, Any] | None]:
        """校验双号异构规划合法性并依次对双槽执行启动前预检。"""
        llm_slot = getattr(plan, "llm_slot", None)
        comfy_slot = getattr(plan, "comfy_slot", None)
        if not getattr(plan, "ok", False) or llm_slot is None or comfy_slot is None:
            return None, None, {"ok": False, "message": str(getattr(plan, "reason", "") or "双号异构规划失败")}
        if llm_slot.account_id == comfy_slot.account_id:
            return None, None, {"ok": False, "message": "双号异构必须分配两个不同的 Google 账号"}

        for sp, cn in ((llm_slot, "LLM"), (comfy_slot, "画布")):
            pre_chk = self._b.run_preflight(deep=False, profiles=list(sp.profiles), gpu=sp.gpu)
            if not pre_chk.get("ok"):
                return None, None, {
                    "ok": False,
                    "preflight_failed": True,
                    "preflight": pre_chk,
                    "message": str(pre_chk.get("blocking_message") or f"{cn}槽位环境自检未通过"),
                }
        return llm_slot, comfy_slot, None

    def _build_split_slot_cmd(
        self,
        slot_spec: Any,
        *,
        recreate: bool,
        minimax_r2v: bool,
        minimax_pe: bool,
        llm_quant: str,
        llm_ctx: int,
        llm_mmproj: bool,
        llm27_quant: str,
        llm27_ctx: int,
        llm27_mmproj: bool,
    ) -> list[str]:
        """构造单个异构分槽的 `colab_ctl up` 无窗口部署命令。"""
        return build_ctl_up_args(
            python_exe=self._ctx.hidden_python_exe(),
            ctl_script=self._ctx.ctl_script,
            session="comfy",
            profiles=list(slot_spec.profiles),
            gpu=slot_spec.gpu,
            high_mem=bool(slot_spec.high_mem),
            recreate=recreate,
            endpoint=str(slot_spec.reuse_endpoint or ""),
            minimax_r2v=minimax_r2v,
            minimax_pe=minimax_pe,
            llm_quant=llm_quant,
            llm_ctx=llm_ctx,
            llm_mmproj=llm_mmproj,
            llm27_quant=llm27_quant,
            llm27_ctx=llm27_ctx,
            llm27_mmproj=llm27_mmproj,
            cohost_comfy=False,
            account_id=slot_spec.account_id,
            no_named_tunnel=bool(slot_spec.no_named_tunnel),
            ctl_prefix=self._ctx.ctl_cmd_prefix(unbuffered=True),
        )

    @staticmethod
    def _init_split_slot_entry(
        sp: Any,
        cmd: list[str],
        is_llm: bool,
        *,
        llm_quant: str,
        llm_mmproj: bool,
        llm27_mmproj: bool,
        minimax_r2v: bool,
        minimax_pe: bool,
    ) -> dict[str, Any]:
        """构造单个分槽的独立进度状态与日志收集条目。"""
        st = default_deploy_state()
        st["gpu"] = sp.gpu
        return {
            "spec": sp,
            "cmd": cmd,
            "st": st,
            "ctx": {
                "phase_idx": 1,
                "phase_total": 1,
                "phase_prof": sp.profiles[0],
                "phase_is_llm": is_llm,
                "dl_done": 0,
                "dl_name": "",
                "dl_total": calc_phase_dl_total(
                    sp.profiles[0],
                    llm_quant=llm_quant,
                    llm_mmproj=llm_mmproj,
                    llm27_mmproj=llm27_mmproj,
                    minimax_r2v=minimax_r2v,
                    minimax_pe=minimax_pe,
                ),
                "captured_llm_tunnel": None,
                "meta_warmed": False,
            },
            "tail": [],
            "tunnel": None,
            "done": False,
            "rc": 1,
        }

    def _init_split_cluster_ctx(
        self,
        llm_slot: Any,
        comfy_slot: Any,
        cfg: dict[str, Any],
        parsed: dict[str, Any],
        defer_pool_refresh: bool,
    ) -> _SplitClusterCtx:
        """持久化偏好与初始分槽，初始化 `_deploy_state` 并构建 `_SplitClusterCtx`。"""
        b = self._b
        display_profiles = list(parsed["display_profiles"])
        recreate = bool(parsed["recreate"])
        minimax_r2v = bool(parsed["minimax_r2v"])
        minimax_pe = bool(parsed["minimax_pe"])

        with self._store.lock:
            prefs = self._store.state.setdefault("prefs", dict(DEFAULT_PREFS))
            llm_quant, llm_ctx, llm_mmproj, llm27_quant, llm27_ctx, llm27_mmproj = parse_llm_boot_params(cfg, prefs)
            prefs.update(
                {
                    "boot_profiles": list(display_profiles),
                    "boot_profile": display_profiles[0],
                    "boot_minimax_r2v": minimax_r2v,
                    "boot_minimax_pe": minimax_pe,
                    "boot_llm_quant": llm_quant,
                    "boot_llm_ctx": llm_ctx,
                    "boot_llm_mmproj": llm_mmproj,
                    "boot_llm27_quant": llm27_quant,
                    "boot_llm27_ctx": llm27_ctx,
                    "boot_llm27_mmproj": llm27_mmproj,
                    "cluster_split_mode": True,
                }
            )
            now_ts = int(self._ctx.now())
            b._deploy_cancelled = False
            b._deploy_meta_warmed = False
            b._deploy_dl_name = ""
            b._deploy_dl_done = 0
            b._deploy_captured_llm_tunnel = None
            start_epoch = b._account_epoch

            self._store.state["cluster_slots"] = _build_initial_cluster_slots_dict(llm_slot, comfy_slot)
            b._deploy_state = {
                "active": True,
                "profile": display_profiles[0],
                "profiles": list(display_profiles),
                "gpu": f"{llm_slot.gpu}+{comfy_slot.gpu}",
                "high_mem": bool(comfy_slot.high_mem),
                "step_idx": 0,
                "step_total": 7,
                "stage_title": (
                    "校验双号算力池"
                    if defer_pool_refresh
                    else f"双号异构部署 (LLM:{llm_slot.gpu} · 画布:{comfy_slot.gpu})"
                ),
                "sub_detail": (
                    "正在并发探测多号余额并确认双号异构分配..."
                    if defer_pool_refresh
                    else "正在并行启动跨账号双进程部署..."
                ),
                "pct": 2,
                "started_ts": now_ts,
                "elapsed_sec": 0,
                "error_msg": "",
                "stockout_gpu": None,
                "suggest_gpu": None,
                "cluster_split": True,
            }
            self._store.save()
            initial_deploy = b.get_deploy_state()
            prefs_copy = dict(prefs)

        with b._comfy_lock:
            b._comfy_state["service_kind"] = "dual"
            b._comfy_state["active_profiles"] = list(display_profiles)
            cstate_init = dict(b._comfy_state)
        with self._store.lock:
            b.last_snapshot["comfy"] = cstate_init

        cmd_kw = {
            "recreate": recreate,
            "minimax_r2v": minimax_r2v,
            "minimax_pe": minimax_pe,
            "llm_quant": llm_quant,
            "llm_ctx": llm_ctx,
            "llm_mmproj": llm_mmproj,
            "llm27_quant": llm27_quant,
            "llm27_ctx": llm27_ctx,
            "llm27_mmproj": llm27_mmproj,
        }
        entry_kw = {
            "llm_quant": llm_quant,
            "llm_mmproj": llm_mmproj,
            "llm27_mmproj": llm27_mmproj,
            "minimax_r2v": minimax_r2v,
            "minimax_pe": minimax_pe,
        }
        slot_states = {
            "llm": self._init_split_slot_entry(
                llm_slot, self._build_split_slot_cmd(llm_slot, **cmd_kw), True, **entry_kw
            ),
            "comfy": self._init_split_slot_entry(
                comfy_slot, self._build_split_slot_cmd(comfy_slot, **cmd_kw), False, **entry_kw
            ),
        }
        profile_label = " + ".join(PROFILE_LABELS.get(p, p) for p in display_profiles)
        self._bus.emit({"type": "deploy_progress", "deploy": initial_deploy})
        return _SplitClusterCtx(
            cfg=cfg,
            parsed=parsed,
            display_profiles=display_profiles,
            profile_label=profile_label,
            recreate=recreate,
            minimax_r2v=minimax_r2v,
            minimax_pe=minimax_pe,
            llm_quant=llm_quant,
            llm_ctx=llm_ctx,
            llm_mmproj=llm_mmproj,
            llm27_quant=llm27_quant,
            llm27_ctx=llm27_ctx,
            llm27_mmproj=llm27_mmproj,
            prefs_copy=prefs_copy,
            tunnel_pat=b._tunnel_url_regex_pattern(),
            start_epoch=start_epoch,
            initial_deploy=initial_deploy,
            defer_pool_refresh=defer_pool_refresh,
            slot_states=slot_states,
        )

    def _build_cmd_for_sctx(self, sctx: _SplitClusterCtx, slot_spec: Any) -> list[str]:
        """基于 `_SplitClusterCtx` 快速为指定槽位构造部署命令。"""
        return self._build_split_slot_cmd(
            slot_spec,
            recreate=sctx.recreate,
            minimax_r2v=sctx.minimax_r2v,
            minimax_pe=sctx.minimax_pe,
            llm_quant=sctx.llm_quant,
            llm_ctx=sctx.llm_ctx,
            llm_mmproj=sctx.llm_mmproj,
            llm27_quant=sctx.llm27_quant,
            llm27_ctx=sctx.llm27_ctx,
            llm27_mmproj=sctx.llm27_mmproj,
        )

    def _merge_and_emit_split_progress(
        self,
        sctx: _SplitClusterCtx,
        src_slot: str,
        detail_line: str = "",
        *,
        force_emit: bool = False,
    ) -> None:
        """聚合双槽部署进度、计算合并百分比并节流广播 `deploy_progress` 事件。"""
        b = self._b
        with sctx.progress_lock:
            l_entry = sctx.slot_states["llm"]
            c_entry = sctx.slot_states["comfy"]
            l_ok = bool(l_entry["done"] and l_entry["rc"] == 0)
            c_ok = bool(c_entry["done"] and c_entry["rc"] == 0)
            l_pct = 100 if l_ok else int(l_entry["st"].get("pct") or 0)
            c_pct = 100 if c_ok else int(c_entry["st"].get("pct") or 0)
            merged_pct = max(2, min(99, (l_pct + c_pct) // 2))
            l_step = int(l_entry["st"].get("step_idx") or 0)
            c_step = int(c_entry["st"].get("step_idx") or 0)
            merged_step = min(l_step, c_step) if (l_step > 0 and c_step > 0) else max(l_step, c_step)
            l_gpu = l_entry["spec"].gpu
            c_gpu = c_entry["spec"].gpu

            if l_ok and not c_entry["done"]:
                stage_title = f"LLM({l_gpu}) 就绪 · 等待画布({c_gpu} {c_pct}%)"
            elif c_ok and not l_entry["done"]:
                stage_title = f"画布({c_gpu}) 就绪 · 等待 LLM({l_gpu} {l_pct}%)"
            else:
                stage_title = f"双号并行 · LLM({l_gpu} {l_pct}%) + 画布({c_gpu} {c_pct}%)"

            tag = "LLM" if src_slot == "llm" else "画布"
            raw_sub = detail_line or str(sctx.slot_states[src_slot]["st"].get("sub_detail") or "")
            sub_detail = f"[{tag}] {raw_sub}"[:46] if raw_sub else "正在并行部署中..."

            now_m = self._ctx.monotonic()
            should_emit = force_emit or (now_m - sctx.last_emit_m >= 0.15)
            if should_emit:
                sctx.last_emit_m = now_m

        with self._store.lock:
            cur_pct = int(b._deploy_state.get("pct") or 0)
            b._deploy_state.update(
                {
                    "step_idx": max(int(b._deploy_state.get("step_idx") or 0), merged_step),
                    "pct": max(cur_pct, merged_pct),
                    "stage_title": stage_title,
                    "sub_detail": sub_detail,
                }
            )
            dep_snap = dict(b._deploy_state)
            if dep_snap.get("started_ts"):
                dep_snap["elapsed_sec"] = max(0, int(self._ctx.now()) - int(dep_snap["started_ts"]))
        if should_emit:
            self._bus.emit({"type": "deploy_progress", "deploy": dep_snap})

    def _try_split_slot_failover(
        self,
        sctx: _SplitClusterCtx,
        slot_key: str,
        entry: dict[str, Any],
        cur_spec: Any,
    ) -> bool:
        """双号并行部署遇单槽缺卡/配额受限时，尝试切换到第 3 个可用备用账号重试。"""
        diag = classify_deploy_failure(entry["tail"], cur_spec.gpu, 0, sctx.profile_label, sctx.recreate)
        if diag.get("kind") not in ("stockout", "entitlement", "limit_or_stockout"):
            return False

        failed_gpu = str(diag.get("stockout_gpu") or cur_spec.gpu).upper()
        self.mark_account_gpu_scarce(
            cur_spec.account_id,
            failed_gpu,
            str(diag.get("reason") or "算力紧缺"),
            kind=str(diag.get("kind") or "stockout"),
        )
        other_key = "comfy" if slot_key == "llm" else "llm"
        other_aid = str(sctx.slot_states[other_key]["spec"].account_id)
        ps = _pool_sched()
        fb = ps.select_best_account(
            self.build_pool_nodes(),
            profiles=list(cur_spec.profiles),
            preferred_gpu=cur_spec.gpu,
            exclude_accounts={cur_spec.account_id, other_aid},
        )
        if not fb.ok or not fb.account_id:
            return False

        new_spec = ps.SplitSlotSpec(
            slot=cur_spec.slot,
            account_id=fb.account_id,
            account_label=fb.account_label or fb.account_id,
            profiles=cur_spec.profiles,
            gpu=fb.gpu or cur_spec.gpu,
            high_mem=cur_spec.high_mem,
            reuse_endpoint=fb.reuse_endpoint,
            no_named_tunnel=cur_spec.no_named_tunnel,
        )
        with sctx.progress_lock:
            entry["spec"] = new_spec
            entry["cmd"] = self._build_cmd_for_sctx(sctx, new_spec)
            entry["tail"] = []
            entry["tunnel"] = None
            entry["st"] = default_deploy_state()
            entry["st"]["gpu"] = new_spec.gpu
        with self._store.lock:
            slots_dict = self._store.state.get("cluster_slots")
            if isinstance(slots_dict, dict) and slot_key in slots_dict:
                slots_dict[slot_key]["account_id"] = new_spec.account_id
                slots_dict[slot_key]["label"] = new_spec.account_label
                slots_dict[slot_key]["gpu"] = new_spec.gpu
                slots_dict[slot_key]["endpoint"] = new_spec.reuse_endpoint
                self._store.save()
        self._merge_and_emit_split_progress(
            sctx,
            slot_key,
            f"{failed_gpu} 紧缺，正切换至备号「{new_spec.account_label}」重试...",
            force_emit=True,
        )
        return True

    def _run_split_slot_worker(
        self, sctx: _SplitClusterCtx, slot_key: str, cancel_tok: Any
    ) -> None:
        """执行单个分槽的流式部署进程循环（含输出解析与第 3 账号自动容灾）。"""
        b = self._b
        ctl_env = self._ctx.build_ctl_env()
        ctl_env["PYTHONUNBUFFERED"] = "1"
        ctl_env["COLAB_CLI_REQUEST_TIMEOUT"] = "75"
        ctl_env["COLAB_CLI_ASSIGN_RETRIES"] = "2"
        ctl_env["COLAB_CLI_RETRY_BACKOFF"] = "5"

        entry = sctx.slot_states[slot_key]
        slot_retries = 0
        while True:
            cur_spec = entry["spec"]
            cur_cmd = entry["cmd"]

            def _on_line(sline: str) -> None:
                with sctx.progress_lock:
                    entry["tail"].append(sline)
                    if len(entry["tail"]) > 120:
                        del entry["tail"][:-120]
                    prev_s = int(entry["st"].get("step_idx") or 0)
                    tun, _warm = consume_deploy_line(
                        sline, entry["st"], entry["ctx"], sctx.prefs_copy, tunnel_pattern=sctx.tunnel_pat
                    )
                    cur_s = int(entry["st"].get("step_idx") or 0)
                    if tun:
                        entry["tunnel"] = tun
                    elif entry["ctx"].get("captured_llm_tunnel"):
                        entry["tunnel"] = entry["ctx"]["captured_llm_tunnel"]
                    det = str(entry["st"].get("sub_detail") or "")
                self._merge_and_emit_split_progress(sctx, slot_key, det, force_emit=bool(tun or cur_s != prev_s))

            stream_res = b._runner.run_streaming(
                cur_cmd,
                on_line=_on_line,
                extra_env=ctl_env,
                cancel_token=cancel_tok,
                popen_fn=self._ctx.subprocess_mod.Popen,
                on_proc_start=lambda p: self._register_split_proc(slot_key, p),
                task_name=f"deploy_split_{slot_key}",
            )
            rc = 0 if stream_res.ok else (stream_res.returncode or 1)
            if not stream_res.ok and stream_res.error and not entry["tail"]:
                entry["tail"].append(stream_res.error)

            cancelled_now = bool(b._deploy_cancelled or stream_res.cancelled or cancel_tok.is_cancelled)
            if rc == 0 or cancelled_now or slot_retries >= 1:
                entry["rc"] = rc
                entry["done"] = True
                if rc == 0:
                    self.clear_account_gpu_scarce(cur_spec.account_id, cur_spec.gpu)
                self._merge_and_emit_split_progress(sctx, slot_key, force_emit=True)
                return

            if self._try_split_slot_failover(sctx, slot_key, entry, cur_spec):
                slot_retries += 1
                continue

            entry["rc"] = rc
            entry["done"] = True
            self._merge_and_emit_split_progress(sctx, slot_key, force_emit=True)
            return

    def _abort_split_if_cancelled(self, sctx: _SplitClusterCtx, cancel_tok: Any) -> bool:
        """检查用户是否主动取消双号异构部署，若已取消则清空分槽并广播中止事件。"""
        b = self._b
        if not (b._deploy_cancelled or cancel_tok.is_cancelled):
            return False
        now_s = self._ctx.now()
        elapsed_now = max(0, int(now_s) - int(sctx.initial_deploy.get("started_ts") or now_s))
        with self._store.lock:
            self._store.state["cluster_slots"] = {}
            b._deploy_state.update(
                {
                    "active": False,
                    "pct": 0,
                    "stage_title": "已中止部署",
                    "sub_detail": "用户主动取消双号异构部署",
                    "error_msg": "已中止开机部署",
                    "elapsed_sec": elapsed_now,
                }
            )
            self._store.save()
        snap_c = b.fetch_live_status()
        self._bus.emit(
            {
                "type": "deploy_error",
                "cancelled": True,
                "deploy": b.get_deploy_state(),
                "snapshot": snap_c,
                "summary": "⏹️ 已中止双号异构部署；若实例已分配请点停卡释放",
            }
        )
        return True

    def _abort_split_with_error(self, sctx: _SplitClusterCtx, fail_msg: str) -> None:
        """双号异构重规划前置失败时回滚分槽并广播错误事件。"""
        b = self._b
        now_s = self._ctx.now()
        elapsed_now = max(0, int(now_s) - int(sctx.initial_deploy.get("started_ts") or now_s))
        with self._store.lock:
            self._store.state["cluster_slots"] = {}
            b._deploy_state.update(
                {
                    "active": False,
                    "stage_title": "部署失败",
                    "sub_detail": fail_msg[:42],
                    "error_msg": fail_msg,
                    "stockout_gpu": None,
                    "suggest_gpu": None,
                    "elapsed_sec": elapsed_now,
                }
            )
            self._store.save()
        snap_fail = b._rebuild_and_emit_snapshot()
        self._bus.emit(
            {
                "type": "deploy_error",
                "deploy": b.get_deploy_state(),
                "snapshot": snap_fail,
                "summary": fail_msg,
            }
        )

    def _refresh_split_plan_in_worker(self, sctx: _SplitClusterCtx, cancel_tok: Any) -> bool:
        """在后台线程内非阻塞刷新多号池并重新确认双槽规划，返回是否继续启动子进程。"""
        b = self._b
        if self._abort_split_if_cancelled(sctx, cancel_tok):
            return False
        if self.ensure_pool_fresh_if_needed():
            b._rebuild_and_emit_snapshot()
        if self._abort_split_if_cancelled(sctx, cancel_tok):
            return False

        with self._store.lock:
            if b._account_epoch != sctx.start_epoch:
                b._deploy_state = default_deploy_state()
                return False
            active_aid_now = self._store.get_active_account_id()

        ps_fresh = _pool_sched()
        nodes_fresh = self.build_pool_nodes()
        healthy_fresh = sum(1 for n in nodes_fresh if n.auth_ok and n.balance is not None and float(n.balance) >= 12.0)
        if healthy_fresh < 2:
            self._abort_split_with_error(sctx, "当前健康付费账号不足 2 个（需余额 >= 12 CU），无法双号异构启动")
            return False

        fresh_plan = ps_fresh.plan_split_cluster(
            nodes_fresh,
            profiles=list(sctx.parsed["profiles"]),
            comfy_gpu=sctx.cfg.get("comfy_gpu"),
            llm_gpu=sctx.cfg.get("llm_gpu"),
            active_account_id=active_aid_now,
        )
        invalid_plan = (
            not fresh_plan.ok
            or fresh_plan.llm_slot is None
            or fresh_plan.comfy_slot is None
            or fresh_plan.llm_slot.account_id == fresh_plan.comfy_slot.account_id
        )
        if invalid_plan:
            self._abort_split_with_error(sctx, str(fresh_plan.reason or "当前号池暂不满足双号异构双开条件"))
            return False

        f_llm = fresh_plan.llm_slot
        f_comfy = fresh_plan.comfy_slot
        with sctx.progress_lock:
            sctx.slot_states["llm"]["spec"] = f_llm
            sctx.slot_states["llm"]["cmd"] = self._build_cmd_for_sctx(sctx, f_llm)
            sctx.slot_states["llm"]["st"]["gpu"] = f_llm.gpu
            sctx.slot_states["comfy"]["spec"] = f_comfy
            sctx.slot_states["comfy"]["cmd"] = self._build_cmd_for_sctx(sctx, f_comfy)
            sctx.slot_states["comfy"]["st"]["gpu"] = f_comfy.gpu
        with self._store.lock:
            self._store.state["cluster_slots"] = _build_initial_cluster_slots_dict(f_llm, f_comfy)
            cur_pct = int(b._deploy_state.get("pct") or 0)
            b._deploy_state.update(
                {
                    "gpu": f"{f_llm.gpu}+{f_comfy.gpu}",
                    "high_mem": bool(f_comfy.high_mem),
                    "pct": max(cur_pct, 3),
                    "stage_title": f"双号异构部署 (LLM:{f_llm.gpu} · 画布:{f_comfy.gpu})",
                    "sub_detail": "号池核验完成，正在并行启动跨账号双进程部署...",
                }
            )
            self._store.save()
        self._bus.emit({"type": "deploy_progress", "deploy": b.get_deploy_state()})
        return True

    def _collect_split_slot_tunnels_and_probe(self, sctx: _SplitClusterCtx) -> None:
        """为已部署成功的分槽分别读取隧道与端点，并安全触发一次 `pool-probe` 同步。"""
        b = self._b
        for skey in ("llm", "comfy"):
            if sctx.slot_states[skey]["rc"] != 0:
                continue
            sp = sctx.slot_states[skey]["spec"]
            _, r_ep, fast_u = b._tunnel_svc.fast_read_slot_tunnel_url(
                str(sp.reuse_endpoint or ""), "comfy", account_id=sp.account_id
            )
            cap_u = sctx.slot_states[skey]["tunnel"] or fast_u
            ep_final = str(r_ep or sp.reuse_endpoint or "").strip()
            with self._store.lock:
                slot_rec = (self._store.state.get("cluster_slots") or {}).get(skey)
                if isinstance(slot_rec, dict):
                    if ep_final:
                        slot_rec["endpoint"] = ep_final
                        if cap_u:
                            self._store.set_tunnel_cache(ep_final, str(cap_u))
                    if cap_u:
                        slot_rec["tunnel_url"] = cap_u
                    slot_rec["online"] = True
            sctx.slot_states[skey]["endpoint"] = ep_final
            sctx.slot_states[skey]["tunnel"] = cap_u

        probe_targets = [
            sctx.slot_states[skey]["spec"].account_id
            for skey in ("llm", "comfy")
            if sctx.slot_states[skey]["rc"] == 0
        ]
        if probe_targets:
            try:
                p_res = self.probe_accounts_raw(probe_targets)
                if p_res.get("accounts"):
                    self.apply_pool_probe_results(p_res["accounts"])
            except Exception as exc:
                self._b._diag.debug(
                    "cluster",
                    f"双号部署后 pool-probe 同步失败（忽略不影响已就绪槽位）: {redact_secrets(str(exc))}",
                )

    def _finalize_split_both_success(
        self, sctx: _SplitClusterCtx, elapsed_total: int, elapsed_cn: str
    ) -> None:
        """双槽全部就绪后的状态落盘、`_comfy_state` 同步与 `deploy_done` 广播。"""
        b = self._b
        l_sp = sctx.slot_states["llm"]["spec"]
        c_sp = sctx.slot_states["comfy"]["spec"]
        cur_slots = self.get_cluster_slots()
        l_rec = cur_slots.get("llm") or {}
        c_rec = cur_slots.get("comfy") or {}
        l_ep = _first_non_empty(l_rec.get("endpoint"), sctx.slot_states["llm"].get("endpoint"))
        c_ep = _first_non_empty(c_rec.get("endpoint"), sctx.slot_states["comfy"].get("endpoint"))
        l_meta, _ = self._secrets.get_ep_meta_and_key(l_ep)
        c_meta, _ = self._secrets.get_ep_meta_and_key(c_ep)
        l_url = _first_non_empty(
            sctx.slot_states["llm"].get("tunnel"),
            l_rec.get("tunnel_url"),
            l_meta.get("llm_tunnel_url"),
            l_meta.get("tunnel_url"),
            b._read_cf_public_llm_url(),
        ) or None
        c_url = _first_non_empty(
            sctx.slot_states["comfy"].get("tunnel"),
            c_rec.get("tunnel_url"),
            c_meta.get("comfy_tunnel_url"),
            c_meta.get("tunnel_url"),
        ) or None
        used_q = sctx.llm27_quant if "qwen_27b" in l_sp.profiles else sctx.llm_quant
        used_ctx = sctx.llm27_ctx if "qwen_27b" in l_sp.profiles else sctx.llm_ctx

        self.touch_slot_activity(None)
        with self._store.lock:
            slots_d = self._store.state.get("cluster_slots")
            if isinstance(slots_d, dict):
                if isinstance(slots_d.get("llm"), dict) and l_url:
                    slots_d["llm"]["tunnel_url"] = l_url
                    if l_ep:
                        self._store.set_tunnel_cache(l_ep, l_url)
                if isinstance(slots_d.get("comfy"), dict) and c_url:
                    slots_d["comfy"]["tunnel_url"] = c_url
                    if c_ep:
                        self._store.set_tunnel_cache(c_ep, c_url)
            b._deploy_state.update(
                {
                    "active": False,
                    "step_idx": 7,
                    "pct": 100,
                    "stage_title": "双号异构部署完成",
                    "sub_detail": f"LLM({l_sp.gpu}) + 画布({c_sp.gpu}) 已就绪",
                    "error_msg": "",
                    "stockout_gpu": None,
                    "suggest_gpu": None,
                    "elapsed_sec": elapsed_total,
                }
            )
            self._store.save()

        with b._comfy_lock:
            b._comfy_state.update(
                {
                    "service_kind": "dual",
                    "online": True,
                    "comfy_online": True,
                    "llm_online": True,
                    "tunnel_url": c_url,
                    "llm_tunnel_url": l_url,
                    "active_profiles": list(sctx.display_profiles),
                    "model_name": b._infer_default_llm_model_name(l_meta),
                    "quant": str(l_meta.get("quant") or used_q),
                    "ctx_size": int(l_meta.get("ctx_size") or used_ctx),
                    "gpu_name": f"{l_sp.gpu}+{c_sp.gpu}",
                }
            )
            cstate = dict(b._comfy_state)
        with self._store.lock:
            b.last_snapshot["comfy"] = cstate

        snap = b.fetch_live_status()
        b._wake_monitor.set()
        self._bus.emit(
            {
                "type": "deploy_done",
                "deploy": b.get_deploy_state(),
                "snapshot": snap,
                "tunnel_url": c_url,
                "llm_tunnel_url": l_url,
                "summary": f"🎉 双号异构部署完成（LLM:{l_sp.gpu} + 画布:{c_sp.gpu} · {elapsed_cn}）！",
                "action": "open_comfy",
            }
        )

    def _finalize_split_partial_or_total_failure(
        self, sctx: _SplitClusterCtx, elapsed_total: int
    ) -> None:
        """处理双号并行部署中的单槽位或双槽位失败：清理失败槽位并保留已成功启动的槽位。"""
        b = self._b
        failed_keys = [k for k in ("comfy", "llm") if sctx.slot_states[k]["rc"] != 0]
        first_fail = failed_keys[0]
        f_sp = sctx.slot_states[first_fail]["spec"]
        diag = classify_deploy_failure(
            sctx.slot_states[first_fail]["tail"],
            f_sp.gpu,
            0,
            sctx.profile_label,
            sctx.recreate,
        )
        stockout_gpu = diag.get("stockout_gpu")
        if stockout_gpu:
            self.mark_account_gpu_scarce(
                f_sp.account_id,
                str(stockout_gpu),
                str(diag.get("reason") or "算力紧缺"),
                kind=str(diag.get("kind") or "stockout"),
            )

        with self._store.lock:
            slots_dict = self._store.state.get("cluster_slots")
            if isinstance(slots_dict, dict):
                for fk in failed_keys:
                    slots_dict.pop(fk, None)
            if not (isinstance(slots_dict, dict) and slots_dict):
                self._store.state["cluster_slots"] = {}
            self._store.save()

        snap = b.fetch_live_status()
        base_msg = str(diag.get("message") or "⚠️ 双号异构部署部分失败")
        if len(failed_keys) == 1:
            ok_cn = "LLM" if first_fail == "comfy" else "画布"
            fail_cn = "画布" if first_fail == "comfy" else "LLM"
            summary = f"⚠️ {fail_cn}槽位失败（已保留{ok_cn}槽位）：{base_msg}"
        else:
            summary = base_msg

        with self._store.lock:
            b._deploy_state.update(
                {
                    "active": False,
                    "stage_title": "部署失败",
                    "sub_detail": str(diag.get("reason") or summary)[:42],
                    "error_msg": summary,
                    "stockout_gpu": stockout_gpu,
                    "suggest_gpu": None,
                    "elapsed_sec": elapsed_total,
                }
            )
            snap["deploy"] = b.get_deploy_state()
            b.last_snapshot = snap

        self._bus.emit(
            {
                "type": "deploy_error",
                "deploy": b.get_deploy_state(),
                "snapshot": snap,
                "summary": summary,
                "stockout_gpu": stockout_gpu,
            }
        )

    def _run_split_deploy_worker(self, sctx: _SplitClusterCtx, cancel_tok: Any) -> None:
        """后台调度双号并行部署线程、心跳广播与收尾清理。"""
        b = self._b
        b._deploy_cancel_token = cancel_tok
        hb_stop = self._ctx.threading_mod.Event()

        def _heartbeat() -> None:
            while not hb_stop.wait(timeout=1.0):
                if not b.is_deploying():
                    break
                self._bus.emit({"type": "deploy_progress", "deploy": b.get_deploy_state()})

        self._ctx.threading_mod.Thread(target=_heartbeat, daemon=True).start()
        try:
            if sctx.defer_pool_refresh and not self._refresh_split_plan_in_worker(sctx, cancel_tok):
                return

            t_llm = self._ctx.threading_mod.Thread(
                target=self._run_split_slot_worker, args=(sctx, "llm", cancel_tok), daemon=True
            )
            t_comfy = self._ctx.threading_mod.Thread(
                target=self._run_split_slot_worker, args=(sctx, "comfy", cancel_tok), daemon=True
            )
            t_llm.start()
            t_comfy.start()
            t_llm.join()
            t_comfy.join()

            now_s = self._ctx.now()
            elapsed_total = max(0, int(now_s) - int(sctx.initial_deploy.get("started_ts") or now_s))
            elapsed_cn = format_elapsed_cn(elapsed_total)

            with self._store.lock:
                if b._account_epoch != sctx.start_epoch:
                    b._deploy_state = default_deploy_state()
                    return

            if self._abort_split_if_cancelled(sctx, cancel_tok):
                return

            self._collect_split_slot_tunnels_and_probe(sctx)
            if sctx.slot_states["llm"]["rc"] == 0 and sctx.slot_states["comfy"]["rc"] == 0:
                self._finalize_split_both_success(sctx, elapsed_total, elapsed_cn)
            else:
                self._finalize_split_partial_or_total_failure(sctx, elapsed_total)
        finally:
            hb_stop.set()
            with self._split_procs_lock:
                self._split_deploy_procs.clear()
            b._deploy_proc = None
            b._deploy_cancel_token = None
            b._deploy_lock.release()

    def start_split_cluster(
        self,
        plan: Any,
        cfg: dict[str, Any],
        parsed: dict[str, Any],
        *,
        defer_pool_refresh: bool = False,
    ) -> dict[str, Any]:
        """使用两个独立进程并行对 `llm_acct` 与 `comfy_acct` 执行 up 并合并直播进度。"""
        b = self._b
        llm_slot, comfy_slot, err_resp = self._validate_split_plan_and_preflight(plan)
        if err_resp is not None:
            return err_resp

        if not b._deploy_lock.acquire(blocking=False):
            return {"ok": False, "message": "已有开机部署任务在运行中"}

        try:
            sctx = self._init_split_cluster_ctx(llm_slot, comfy_slot, cfg, parsed, defer_pool_refresh)
        except Exception:
            b._deploy_lock.release()
            raise

        b._wake_monitor.set()
        b._tasks.submit(
            "deploy",
            lambda cancel_tok: self._run_split_deploy_worker(sctx, cancel_tok),
            exclusive=False,
        )
        return {
            "ok": True,
            "started": True,
            "cluster_split": True,
            "deploy": sctx.initial_deploy,
            "message": f"🔀 开始双号异构部署：LLM({llm_slot.gpu}·{llm_slot.account_label}) + 画布({comfy_slot.gpu}·{comfy_slot.account_label})...",
        }

    # ------------------------------------------------------------------
    # 异构双开监控轮询、双边独立保活与选择性闲置停卡
    # ------------------------------------------------------------------

    def _maybe_periodic_cluster_pool_probe(self, slots: dict[str, dict[str, Any]]) -> None:
        """每 45 秒周期性通过 `pool-probe` 同步活跃集群账号的费率与余额。"""
        now_m = self._ctx.monotonic()
        slot_aids = list(
            {str(s.get("account_id") or "").strip() for s in slots.values() if str(s.get("account_id") or "").strip()}
        )
        if not slot_aids or (now_m - self._last_cluster_probe_m) < 45.0:
            return
        self._last_cluster_probe_m = now_m
        try:
            p_res = self.probe_accounts_raw(slot_aids)
            if p_res.get("accounts"):
                self.apply_pool_probe_results(p_res["accounts"])
        except Exception as exc:
            self._b._diag.debug(
                "cluster",
                f"周期性集群 pool-probe 同步失败: {redact_secrets(str(exc))}",
            )

    def poll_cluster_monitor_tick(self) -> float:
        """异构集群活跃时的单轮健康探针：分别解析双号隧道并复用 `backend_comfy` 聚合状态。"""
        b = self._b
        slots = self.get_cluster_slots()
        if not slots:
            b._reset_comfy_offline()
            return 5.0

        comfy_slot = slots.get("comfy")
        llm_slot = slots.get("llm")
        comfy_url = self._resolve_slot_tunnel_url("comfy", comfy_slot) or "" if comfy_slot else ""
        llm_url = self._resolve_slot_tunnel_url("llm", llm_slot) or "" if llm_slot else ""
        llm_ep = str((llm_slot or {}).get("endpoint") or "")

        self._maybe_periodic_cluster_pool_probe(slots)

        if comfy_slot and llm_slot:
            if comfy_url:
                b._ensure_ws_target(comfy_url)
            else:
                b._stop_ws()
            if not (comfy_url or llm_url):
                b._reset_comfy_offline()
                return 5.0
            cstate = b.poll_dual_once(comfy_url, llm_url, ep=llm_ep)
            self._sync_slot_online_flags(
                comfy_online=bool(cstate.get("comfy_online")),
                llm_online=bool(cstate.get("llm_online")),
                comfy_url=comfy_url or None,
                llm_url=llm_url or None,
            )
            busy = (
                int(cstate.get("running_count") or 0)
                + int(cstate.get("pending_count") or 0)
                + int(cstate.get("llm_running_count") or 0)
            ) > 0
            return 2.5 if busy else 6.0

        if llm_slot:
            b._stop_ws()
            if not llm_url:
                b._reset_comfy_offline()
                return 5.0
            cstate = b.poll_llm_http_once(llm_url, ep=llm_ep)
            self._sync_slot_online_flags(
                comfy_online=False, llm_online=bool(cstate.get("llm_online")), llm_url=llm_url
            )
            return 2.5 if int(cstate.get("running_count") or 0) > 0 else 8.0

        if comfy_slot and comfy_url:
            b._ensure_ws_target(comfy_url)
            cstate = b.poll_comfy_http_once(comfy_url)
            self._sync_slot_online_flags(
                comfy_online=bool(cstate.get("comfy_online")), llm_online=False, comfy_url=comfy_url
            )
            busy = (int(cstate.get("running_count") or 0) + int(cstate.get("pending_count") or 0)) > 0
            return 2.5 if busy else 6.0

        b._reset_comfy_offline()
        return 5.0

    def _sync_slot_online_flags(
        self,
        *,
        comfy_online: bool | None = None,
        llm_online: bool | None = None,
        comfy_url: str | None = None,
        llm_url: str | None = None,
    ) -> None:
        """同步各分槽在线标记与隧道 URL，发生变更时立即重建并广播状态快照。"""
        b = self._b
        changed = False
        snap: dict[str, Any] | None = None
        with self._store.lock:
            raw = self._store.state.get("cluster_slots")
            if not isinstance(raw, dict):
                return
            for skey, online_val, url_val in (("comfy", comfy_online, comfy_url), ("llm", llm_online, llm_url)):
                slot_rec = raw.get(skey)
                if not isinstance(slot_rec, dict):
                    continue
                if online_val is not None and bool(slot_rec.get("online")) != bool(online_val):
                    slot_rec["online"] = bool(online_val)
                    changed = True
                if url_val and slot_rec.get("tunnel_url") != url_val:
                    slot_rec["tunnel_url"] = url_val
                    changed = True
            if changed:
                snap = b._build_snapshot(
                    live_data=None,
                    stale=bool(b.last_snapshot.get("stale", False)),
                    error_msg=b.last_snapshot.get("error_msg"),
                )
                b.last_snapshot = snap
        if changed and snap is not None:
            self._bus.emit({"type": "state_sync", "snapshot": snap})

    def _probe_cf_llm_fallback(self, cf_cfg_p: Any) -> str | None:
        """探测固定 Cloudflare 域名下的公网 LLM 隧道作为回退入口。"""
        cf_llm = self._b._read_cf_public_llm_url()
        if cf_llm and self._ctx.probe_tunnel_alive(cf_llm, timeout=3.0, kind="llm", cf_config_path=cf_cfg_p):
            return cf_llm
        return None

    def _save_resolved_slot_tunnel(self, slot_key: str, use_ep: str, fast_cand: str) -> None:
        """将探活通过的槽位隧道地址同步写入 `tunnel_cache` 与 `cluster_slots` 并持久化。"""
        with self._store.lock:
            if use_ep:
                self._store.set_tunnel_cache(use_ep, fast_cand)
            raw_slots = self._store.state.get("cluster_slots")
            if isinstance(raw_slots, dict) and isinstance(raw_slots.get(slot_key), dict):
                if use_ep:
                    raw_slots[slot_key]["endpoint"] = use_ep
                raw_slots[slot_key]["tunnel_url"] = fast_cand
            self._store.save()

    def _resolve_slot_tunnel_url(self, slot_key: str, slot_info: dict[str, Any]) -> str | None:
        """为指定集群分槽解析并验证存活的公网隧道地址（按槽位账号定向速读，绑定失效时拒绝串号）。"""
        b = self._b
        if bool(slot_info.get("binding_invalid")):
            return None
        aid = str(slot_info.get("account_id") or "").strip()
        ep = str(slot_info.get("endpoint") or "").strip()
        sname = str(slot_info.get("session") or "comfy").strip() or "comfy"
        cf_cfg_p = self._ctx.resolve_cf_config_path()

        cache = self._store.get_tunnel_cache()
        cand = _first_non_empty(slot_info.get("tunnel_url"), cache.get(ep) if ep else "")
        if cand:
            tok = b._get_comfy_key_for_url(cand, ep) if slot_key == "comfy" else ""
            if self._ctx.probe_tunnel_alive(cand, timeout=3.0, kind=slot_key, auth_token=tok, cf_config_path=cf_cfg_p):
                return cand

        now_m = self._ctx.monotonic()
        ts_attr = "_last_llm_tunnel_discover_ts" if slot_key == "llm" else "_last_tunnel_discover_ts"
        if (now_m - float(getattr(b, ts_attr, 0.0) or 0.0)) < 20.0:
            return self._probe_cf_llm_fallback(cf_cfg_p) if slot_key == "llm" else None

        setattr(b, ts_attr, now_m)
        tstatus, resolved_ep, fast_u = b._tunnel_svc.fast_read_slot_tunnel_url(ep, sname, account_id=aid)
        use_ep = str(resolved_ep or ep or "").strip()
        if ep and use_ep and use_ep != ep:
            return None
        meta, _ = self._secrets.get_ep_meta_and_key(use_ep)
        meta_u = meta.get("llm_tunnel_url") if slot_key == "llm" else meta.get("comfy_tunnel_url")
        fast_cand = _first_non_empty(meta_u, fast_u)

        if tstatus == "ok" and fast_cand:
            tok = b._get_comfy_key_for_url(fast_cand, use_ep) if slot_key == "comfy" else ""
            if self._ctx.probe_tunnel_alive(
                fast_cand, timeout=3.5, kind=slot_key, auth_token=tok, cf_config_path=cf_cfg_p
            ):
                self._save_resolved_slot_tunnel(slot_key, use_ep, fast_cand)
                return fast_cand

        return self._probe_cf_llm_fallback(cf_cfg_p) if slot_key == "llm" else None

    def maybe_trigger_cluster_keepalive(self, cur_m: float) -> None:
        """双号异构模式下对各在线分槽分别发送保活心跳（遵循单边 45m 纯闲置熔断）。"""
        b = self._b
        slots = self.get_cluster_slots()
        if not slots:
            return

        keepalive_interval = self._ctx.keepalive_interval_sec
        keepalive_max_idle = self._ctx.keepalive_max_idle_sec
        keepalive_cooldown = self._ctx.keepalive_retry_cooldown_sec

        with self._store.lock:
            if not bool((self._store.state.get("prefs") or {}).get("keep_alive", True)):
                return
            if b._last_keepalive_ts > 0.0 and (cur_m - b._last_keepalive_ts) < keepalive_interval:
                return
            targets: list[tuple[str, str, str]] = []
            for skey in ("llm", "comfy"):
                s_obj = slots.get(skey)
                if not s_obj:
                    continue
                if self._slot_last_activity_m.get(skey, 0.0) <= 0.0:
                    self._slot_last_activity_m[skey] = cur_m
                if max(0.0, cur_m - self._slot_last_activity_m[skey]) <= keepalive_max_idle:
                    ep = str(s_obj.get("endpoint") or "").strip()
                    aid = str(s_obj.get("account_id") or "").strip()
                    if ep or aid:
                        targets.append((skey, ep, aid))
            if not targets:
                return
            start_epoch = b._account_epoch

        if not b._keepalive_lock.acquire(blocking=False):
            return

        with self._store.lock:
            if b._account_epoch != start_epoch or (
                b._last_keepalive_ts > 0.0 and (cur_m - b._last_keepalive_ts) < keepalive_interval
            ):
                b._keepalive_lock.release()
                return
            b._last_keepalive_ts = cur_m

        def _worker() -> None:
            any_ok = False
            try:
                for _skey, ep, aid in targets:
                    if b._monitor._send_keepalive_for_account(target_ep=ep, account_id=aid):
                        any_ok = True
            except Exception:
                any_ok = False
            finally:
                if not any_ok and not b._monitor_stop.is_set():
                    with self._store.lock:
                        if b._account_epoch == start_epoch and b._last_keepalive_ts == cur_m:
                            b._last_keepalive_ts = max(1e-6, cur_m - (keepalive_interval - keepalive_cooldown))
                b._keepalive_lock.release()

        self._ctx.threading_mod.Thread(target=_worker, daemon=True).start()

    def _run_slot_archive_sync(
        self, comfy_slot: dict[str, Any], start_epoch: int, cancel_tok: Any
    ) -> tuple[bool, str]:
        """对高耗画布槽位执行带账号作用域的云盘成图归档校验（最多 2 轮）。"""
        b = self._b
        if not self._ctx.ctl_available():
            return True, ""
        aid = str(comfy_slot.get("account_id") or "").strip()
        sname = str(comfy_slot.get("session") or "comfy").strip() or "comfy"
        cmd = [*self._ctx.ctl_cmd_prefix(), "sync"]
        if aid:
            cmd.extend(["--account", aid])
        cmd.extend(["-s", sname])

        archive_err = ""
        for attempt in (1, 2):
            with self._store.lock:
                if b._account_epoch != start_epoch:
                    return False, "__SWITCHED__"
            sync_res = b._runner.run(
                cmd,
                timeout=120,
                extra_env=self._ctx.build_ctl_env(),
                cancel_token=cancel_tok,
                run_fn=self._ctx.run_hidden,
                task_name="cluster_idle_sync",
            )
            out_txt = sync_res.combined_output
            if sync_res.ok and ("<<<SYNC_VERIFIED>>>" in out_txt or "SKIP_LLM_SYNC" in out_txt):
                return True, ""
            archive_err = sync_res.error_summary or f"sync 退出码 {sync_res.returncode}"
            if attempt == 1:
                self._ctx.sleep(2.0)
        return False, archive_err

    def _collect_cluster_idle_metrics(
        self, slots: dict[str, dict[str, Any]], cur_m: float
    ) -> dict[str, Any]:
        """采集各分槽忙闲状态与闲置时长，并自动触活正在计算的槽位（严守锁偏序）。"""
        b = self._b
        with b._comfy_lock:
            c_online = bool(b._comfy_state.get("comfy_online"))
            l_online = bool(b._comfy_state.get("llm_online"))
            any_online = bool(b._comfy_state.get("online"))
            comfy_busy = (
                int(b._comfy_state.get("running_count") or 0)
                + int(b._comfy_state.get("pending_count") or 0)
            ) > 0
            llm_busy = int(b._comfy_state.get("llm_running_count") or 0) > 0

        if comfy_busy:
            self.touch_slot_activity("comfy")
        if llm_busy:
            self.touch_slot_activity("llm")

        with self._store.lock:
            for skey in slots:
                if self._slot_last_activity_m.get(skey, 0.0) <= 0.0:
                    self._slot_last_activity_m[skey] = cur_m
            comfy_idle = max(0.0, cur_m - self._slot_last_activity_m.get("comfy", cur_m)) if "comfy" in slots else 0.0
            llm_idle = max(0.0, cur_m - self._slot_last_activity_m.get("llm", cur_m)) if "llm" in slots else 0.0
            comfy_warned = self._slot_idle_warned.get("comfy")
            llm_warned = self._slot_idle_warned.get("llm")

        comfy_present = "comfy" in slots and (c_online or bool(slots["comfy"].get("online")) or any_online)
        llm_present = "llm" in slots and (l_online or bool(slots["llm"].get("online")) or any_online)
        return {
            "comfy_present": comfy_present,
            "llm_present": llm_present,
            "comfy_busy": comfy_busy,
            "llm_busy": llm_busy,
            "comfy_idle": comfy_idle,
            "llm_idle": llm_idle,
            "comfy_warned": comfy_warned,
            "llm_warned": llm_warned,
        }

    @staticmethod
    def _decide_idle_stop_target(metrics: dict[str, Any], limit_sec: float) -> str | None:
        """根据双槽在线与闲置时长决定释放目标（`all` / `comfy` / `llm` / `None`）。"""
        comfy_due = metrics["comfy_present"] and (not metrics["comfy_busy"]) and metrics["comfy_idle"] >= limit_sec
        llm_due = metrics["llm_present"] and (not metrics["llm_busy"]) and metrics["llm_idle"] >= limit_sec
        if comfy_due and llm_due:
            return "all"
        if comfy_due:
            return "comfy"
        if llm_due and not metrics["comfy_present"]:
            return "llm"
        return None

    def _run_cluster_idle_stop_worker(
        self,
        target_slot: str,
        slots: dict[str, dict[str, Any]],
        idle_stop_mins: int,
        policy: str,
        start_epoch: int,
        cancel_tok: Any,
    ) -> None:
        """后台执行集群闲置自动归档同步与定向释放，并广播结果事件。"""
        b = self._b
        try:
            if b.is_deploying() or b._upload_lock.locked():
                return
            with self._store.lock:
                if b._account_epoch != start_epoch:
                    return
            archive_needed = target_slot in ("comfy", "all") and ("comfy" in slots)
            archive_ok, archive_err = (True, "")
            if archive_needed:
                archive_ok, archive_err = self._run_slot_archive_sync(slots["comfy"], start_epoch, cancel_tok)
                if archive_err == "__SWITCHED__":
                    return
                if not archive_ok and policy == "keep_alive":
                    short_err = (redact_secrets(archive_err) or "云盘同步校验未通过")[:48]
                    with self._store.lock:
                        b._idle_archive_retry_after_m = self._ctx.monotonic() + 600.0
                        self._slot_idle_warned["comfy"] = None
                    self._bus.emit(
                        {
                            "type": "idle_archive_failed",
                            "is_error": True,
                            "summary": f"🚨 画布云盘归档失败（{short_err}）！已按「保图留机」保留画布槽位",
                            "action": "sync_outputs",
                            "error_detail": redact_secrets(archive_err),
                        }
                    )
                    return

            res = self.stop_cluster_slot(target_slot)
            archive_failed = bool(archive_needed and not archive_ok)
            if res.get("ok"):
                if target_slot == "comfy" and ("llm" in slots):
                    summary = (
                        f"🛑 画布槽位连续闲置满 {idle_stop_mins} 分钟，已归档并释放画布卡（保留 LLM 在线省积分）"
                        if not archive_failed
                        else "⚠️ 画布归档失败但已优先释放高耗画布卡（保留 LLM 在线）"
                    )
                else:
                    summary = (
                        f"🛑 连续闲置满 {idle_stop_mins} 分钟，已归档并释放集群实例省积分"
                        if not archive_failed
                        else "⚠️ 云盘归档失败，已按策略优先释放集群实例停止计费"
                    )
                self._bus.emit(
                    {
                        "type": "idle_stopped",
                        "archive_failed": archive_failed,
                        "is_error": archive_failed,
                        "summary": summary,
                        "snapshot": res.get("snapshot") or b.get_cached_snapshot(),
                    }
                )
            else:
                fail_msg = redact_secrets(str(res.get("message") or "请立即手动停卡"))
                with self._store.lock:
                    self._slot_idle_warned["comfy"] = None
                    b._idle_archive_retry_after_m = self._ctx.monotonic() + 60.0
                self._bus.emit(
                    {
                        "type": "idle_warning",
                        "is_error": True,
                        "summary": f"⚠️ 集群闲置自动停卡未成功: {fail_msg}",
                        "action": "keep_working",
                    }
                )
        finally:
            b._idle_stop_lock.release()

    def _maybe_emit_cluster_idle_warning(
        self, slots: dict[str, dict[str, Any]], metrics: dict[str, Any], warn_sec: float
    ) -> None:
        """在达到闲置停卡阈值前 2 分钟发出续期预警气泡。"""
        if metrics["comfy_present"] and (not metrics["comfy_busy"]) and metrics["comfy_idle"] >= warn_sec:
            ep_tag = str(slots["comfy"].get("endpoint") or "__comfy__")
            if metrics["comfy_warned"] != ep_tag:
                with self._store.lock:
                    self._slot_idle_warned["comfy"] = ep_tag
                self._bus.emit(
                    {
                        "type": "idle_warning",
                        "summary": f"⏳ 画布槽位已闲置 {int(metrics['comfy_idle'] // 60)} 分钟，约 2 分钟后将自动归档停卡（保留 LLM，点我续期）",
                        "action": "keep_working",
                    }
                )
        elif (
            metrics["llm_present"]
            and (not metrics["comfy_present"])
            and (not metrics["llm_busy"])
            and metrics["llm_idle"] >= warn_sec
        ):
            ep_tag = str(slots["llm"].get("endpoint") or "__llm__")
            if metrics["llm_warned"] != ep_tag:
                with self._store.lock:
                    self._slot_idle_warned["llm"] = ep_tag
                self._bus.emit(
                    {
                        "type": "idle_warning",
                        "summary": f"⏳ LLM 槽位已闲置 {int(metrics['llm_idle'] // 60)} 分钟，约 2 分钟后将自动停卡（点我续期）",
                        "action": "keep_working",
                    }
                )

    def maybe_trigger_cluster_idle_auto_stop(self, cur_m: float) -> None:
        """异构集群独立闲置检测：当高耗画布槽位连续闲置满阈值而 LLM 仍忙碌时，仅归档并释放画布槽位省积分。"""
        b = self._b
        if b.is_deploying() or b._upload_lock.locked() or b._action_lock.locked():
            return
        if cur_m < b._idle_archive_retry_after_m:
            return
        slots = self.get_cluster_slots()
        if not slots:
            return

        with self._store.lock:
            prefs = self._store.get_prefs()
            idle_stop_mins = int(prefs.get("idle_auto_stop_minutes", 60) or 0)
            if idle_stop_mins <= 0:
                return
            policy = str(prefs.get("idle_archive_fail_policy") or "stop_billing").strip().lower()
            if policy not in ("stop_billing", "keep_alive"):
                policy = "stop_billing"
            start_epoch = b._account_epoch

        metrics = self._collect_cluster_idle_metrics(slots, cur_m)
        limit_sec = float(idle_stop_mins * 60)
        warn_sec = max(60.0, limit_sec - 120.0)
        target_slot = self._decide_idle_stop_target(metrics, limit_sec)

        if target_slot is not None:
            if not b._idle_stop_lock.acquire(blocking=False):
                return
            hdl = b._tasks.submit(
                "cluster_idle_stop",
                lambda cancel_tok: self._run_cluster_idle_stop_worker(
                    target_slot, slots, idle_stop_mins, policy, start_epoch, cancel_tok
                ),
                exclusive=False,
            )
            if not hdl.accepted:
                b._idle_stop_lock.release()
            return

        self._maybe_emit_cluster_idle_warning(slots, metrics, warn_sec)

    # ------------------------------------------------------------------
    # 分槽定向停卡（stop_cluster_slot）与集群产物同步
    # ------------------------------------------------------------------

    def _precheck_slot_binding_live(self, skey: str, aid: str, ep: str) -> str | None:
        """前置核验槽位绑定实例是否仍存在，若已消失则标记 `binding_invalid` 并返回拒停文案。"""
        if not (aid and self._ctx.get_colab_py().exists()):
            return None
        try:
            pre_res = self.probe_accounts_raw([aid])
            acc_info = (pre_res.get("accounts") or {}).get(aid)
            if isinstance(acc_info, dict) and bool(acc_info.get("auth_ok", acc_info.get("ok", True))):
                pre_sess = [s for s in (acc_info.get("sessions") or []) if isinstance(s, dict)]
                pre_eps = {str(s.get("endpoint") or "").strip() for s in pre_sess if s.get("endpoint")}
                if ep not in pre_eps:
                    self.apply_pool_probe_results({aid: acc_info})
                    return f"{skey}({ep}) 原绑定实例已消失，同账号状态未确认，已拒绝盲停"
        except Exception as exc:
            self._b._diag.debug(
                "cluster",
                f"停卡前置 pool-probe 校验异常({skey}): {redact_secrets(str(exc))}",
            )
        return None

    @staticmethod
    def _is_ep_in_stop_verified_frame(stdout: str, ep: str) -> bool:
        """解析 `<<<STOP_VERIFIED>>>` 帧，确认目标端点是否仍留在剩余列表中。"""
        if not ep:
            return False
        m_ver = re.search(r"<<<STOP_VERIFIED>>>(.*?)<<<END_STOP_VERIFIED>>>", stdout or "", re.S)
        if not m_ver:
            return False
        try:
            vdata = json.loads(m_ver.group(1).strip())
            rem_eps = [str(x) for x in (vdata.get("remaining") or []) if x]
            return ep in rem_eps
        except Exception:
            return False

    def _probe_slot_still_running(self, aid: str, ep: str) -> bool:
        """通过 `pool-probe` 核实指定账号名下目标实例是否仍未释放。"""
        if not (aid and self._ctx.get_colab_py().exists()):
            return False
        try:
            p_res = self.probe_accounts_raw([aid])
            acc_info = (p_res.get("accounts") or {}).get(aid)
            if not isinstance(acc_info, dict):
                return False
            self.apply_pool_probe_results({aid: acc_info})
            acc_sess = acc_info.get("sessions") or []
            acc_eps = {str(s.get("endpoint") or "") for s in acc_sess if isinstance(s, dict)}
            if ep:
                return ep in acc_eps
            return int(acc_info.get("assignments_count") or len(acc_sess)) > 0
        except Exception:
            return False

    def _verify_slot_stopped(
        self, aid: str, ep: str, active_aid: str, proc: Any
    ) -> bool:
        """三重核实目标槽位是否仍处于计费态（标准帧 + `pool-probe` + 当前激活号实时快照）。"""
        still_running = self._is_ep_in_stop_verified_frame(proc.stdout or "", ep)
        if self._probe_slot_still_running(aid, ep):
            still_running = True
        if aid == active_aid:
            try:
                live_snap = self._b.fetch_live_status()
                live_eps = {
                    str(s.get("endpoint") or "")
                    for s in (live_snap.get("sessions") or [])
                    if isinstance(s, dict) and s.get("endpoint")
                }
                if ep and ep in live_eps:
                    still_running = True
            except Exception as exc:
                self._b._diag.debug(
                    "cluster",
                    f"停卡后拉取激活账号实时状态核实异常: {redact_secrets(str(exc))}",
                )
        return still_running

    def _stop_single_cluster_slot(
        self, skey: str, s_info: dict[str, Any], active_aid: str
    ) -> tuple[bool, bool, str]:
        """执行单个分槽的前置校验、定向停卡、停后核实与状态清理，返回 `(released, binding_invalid, err_msg)`。"""
        b = self._b
        aid = str(s_info.get("account_id") or "").strip()
        ep = str(s_info.get("endpoint") or "").strip()
        sname = str(s_info.get("session") or "comfy").strip() or "comfy"
        if (not ep) or bool(s_info.get("binding_invalid")):
            return False, True, f"{skey}({ep or aid}) 槽位绑定已失效或缺少明确实例，已拒绝停卡"

        pre_err = self._precheck_slot_binding_live(skey, aid, ep)
        if pre_err:
            return False, True, pre_err

        cmd = [*self._ctx.ctl_cmd_prefix(), "down"]
        if aid:
            cmd.extend(["--account", aid])
        cmd.extend(["-s", sname])
        if ep:
            cmd.extend(["--endpoint", ep])

        proc = b._runner.run(
            cmd,
            timeout=60,
            extra_env=self._ctx.build_ctl_env(),
            run_fn=self._ctx.run_hidden,
            task_name=f"stop_cluster_{skey}",
        )
        still_running = self._verify_slot_stopped(aid, ep, active_aid, proc)
        if (not proc.ok) or still_running:
            err_txt = (
                f"{skey}({ep or aid}) 仍处于计费状态"
                if still_running
                else (proc.error_summary or "停卡命令返回异常")
            )
            return False, False, err_txt

        with self._store.lock:
            cur_slots = self._store.state.get("cluster_slots")
            if isinstance(cur_slots, dict):
                cur_slots.pop(skey, None)
            b._drop_ep_ephemeral_locked(ep)
            self._slot_last_activity_m.pop(skey, None)
            self._slot_idle_warned.pop(skey, None)
            self._store.save()
        return True, False, ""

    def _reconcile_remaining_cluster_slots(self, rem_slots: dict[str, dict[str, Any]]) -> None:
        """根据停卡后剩余槽位情况清空离线态或平滑降级为单槽运行态。"""
        b = self._b
        if not rem_slots:
            with self._store.lock:
                self._store.state["cluster_slots"] = {}
                self._store.state["managed_binding"] = None
                self._store.save()
            b._reset_comfy_offline()
            return

        if len(rem_slots) != 1:
            return
        rem_k = "llm" if "llm" in rem_slots else "comfy"
        rem_info = rem_slots[rem_k]
        if rem_k == "llm":
            b._stop_ws()
        with b._comfy_lock:
            if rem_k == "llm":
                b._comfy_state.update(
                    {
                        "service_kind": "llm",
                        "comfy_online": False,
                        "ws_connected": False,
                        "running_count": int(b._comfy_state.get("llm_running_count") or 0),
                        "pending_count": 0,
                        "tunnel_url": rem_info.get("tunnel_url"),
                        "llm_tunnel_url": rem_info.get("tunnel_url"),
                        "active_profiles": list(rem_info.get("profiles") or ["qwen_27b"]),
                    }
                )
            else:
                b._comfy_state.update(
                    {
                        "service_kind": "comfy",
                        "llm_online": False,
                        "llm_running_count": 0,
                        "llm_tunnel_url": None,
                        "tps": None,
                        "tunnel_url": rem_info.get("tunnel_url"),
                        "active_profiles": list(rem_info.get("profiles") or ["minimax"]),
                    }
                )
            cstate = dict(b._comfy_state)
        with self._store.lock:
            b.last_snapshot["comfy"] = cstate

    def stop_cluster_slot(self, slot: str = "all") -> dict[str, Any]:
        """按槽位（`comfy` / `llm` / `all`）定向释放异构集群实例并核实释放结果。"""
        b = self._b
        slot_norm = str(slot or "all").strip().lower()
        if slot_norm not in ("comfy", "llm", "all"):
            return {"ok": False, "message": f"不支持的集群槽位: {slot}"}

        if b.is_deploying():
            b.cancel_deploy()
            self._ctx.sleep(0.4)

        if not self.has_active_slots():
            if slot_norm == "all":
                return b.stop_all_instances()
            return {"ok": False, "message": "当前未处于多号异构集群运行态"}

        if not b._action_lock.acquire(blocking=False):
            return {"ok": False, "message": "另一项操作正在执行，请稍等片刻"}

        try:
            if not self._ctx.ctl_available():
                return {"ok": False, "message": f"未找到控制脚本: {self._ctx.ctl_script.name}"}

            slots = self.get_cluster_slots()
            target_keys = (
                [k for k in ("comfy", "llm") if k in slots]
                if slot_norm == "all"
                else ([slot_norm] if slot_norm in slots else [])
            )
            if not target_keys:
                snap = b._rebuild_and_emit_snapshot()
                return {"ok": True, "snapshot": snap, "message": f"槽位 {slot_norm} 当前未运行"}

            active_aid = self._store.get_active_account_id()
            failures: list[str] = []
            released_keys: list[str] = []
            binding_invalid_hit = False

            for skey in target_keys:
                released, invalid_flag, err_txt = self._stop_single_cluster_slot(skey, slots[skey], active_aid)
                if released:
                    released_keys.append(skey)
                else:
                    if invalid_flag:
                        binding_invalid_hit = True
                    failures.append(err_txt)

            rem_slots = self.get_cluster_slots()
            self._reconcile_remaining_cluster_slots(rem_slots)
            snap = b._rebuild_and_emit_snapshot()

            if failures:
                err_resp: dict[str, Any] = {
                    "ok": False,
                    "still_running": True,
                    "released": released_keys,
                    "snapshot": snap,
                    "message": f"部分槽位停卡未确认释放: {'; '.join(failures)}",
                }
                if binding_invalid_hit:
                    err_resp["binding_invalid"] = True
                    err_resp["target_unresolved"] = True
                return err_resp

            if slot_norm == "comfy" and "llm" in rem_slots:
                msg = "✅ 已释放画布槽位，保留 L4 LLM 对话服务在线（省积分）"
            elif slot_norm == "llm" and "comfy" in rem_slots:
                msg = "✅ 已释放 LLM 槽位，保留 ComfyUI 画布在线"
            else:
                msg = "✅ 已释放全部异构集群实例"
            return {"ok": True, "released": released_keys, "snapshot": snap, "message": msg}
        except Exception as exc:
            return {"ok": False, "message": f"集群停卡异常: {redact_secrets(str(exc))}"}
        finally:
            b._action_lock.release()

    def sync_slot_outputs(self, session: str = "comfy") -> dict[str, Any]:
        """异构集群模式下对 ComfyUI 槽位执行带 `--account` 的成图拉回（无画布槽位时返回 LLM 接口配置）。"""
        b = self._b
        b.touch_activity("comfy")
        if b.is_deploying():
            return {"ok": False, "message": "正在执行开机部署，请等待就绪后再操作"}
        comfy_slot = self.get_slot("comfy")
        if comfy_slot is None:
            return b.get_llm_api_bundle(session="llm")
        if not b._action_lock.acquire(blocking=False):
            return {"ok": False, "message": "另一项操作正在执行，请稍等片刻"}
        try:
            if not self._ctx.ctl_available():
                return {"ok": False, "message": f"未找到控制脚本: {self._ctx.ctl_script.name}"}
            aid = str(comfy_slot.get("account_id") or "").strip()
            sname = str(comfy_slot.get("session") or session or "comfy").strip() or "comfy"
            cmd = [*self._ctx.ctl_cmd_prefix(), "sync", "--pull"]
            if aid:
                cmd.extend(["--account", aid])
            cmd.extend(["-s", sname])
            res = b._runner.run(
                cmd,
                timeout=180,
                extra_env=self._ctx.build_ctl_env(),
                run_fn=self._ctx.run_hidden,
                task_name="cluster_sync_outputs",
            )
            tail = (res.combined_output.splitlines() or ["同步结束"])[-1]
            if res.ok:
                if b.download_root.exists():
                    self._ctx.reveal_path(b.download_root)
                return {"ok": True, "message": f"拉回完成: {tail}"}
            return {"ok": False, "message": f"同步失败: {res.error_summary}"}
        except Exception as exc:
            return {"ok": False, "message": f"同步异常: {redact_secrets(str(exc))}"}
        finally:
            b._action_lock.release()
