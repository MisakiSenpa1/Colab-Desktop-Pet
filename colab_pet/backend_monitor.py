"""Colab 桌宠后台状态采集、账本核算、智能保活与闲置自动停卡子服务。"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from datetime import datetime
from typing import TYPE_CHECKING, Any

from colab_core.accounts import has_gdrive_save_section
from colab_core.catalog import export_frontend_catalog
from colab_core.runtime import (
    DEFAULT_MEASURED_GPU_RATES,
    DEFAULT_PREFS,
    default_baseline,
    estimate_session_start_balance,
    gpu_rate_key,
    infer_account_tier_and_baseline,
    sync_active_account_ledger,
    upsert_account_profile,
)
from colab_security import redact_secrets

if TYPE_CHECKING:
    from colab_backend import ColabPetBackend

# 账号未绑定或授权失效时需要重置的运行态字段与默认值
_AUTH_RESET_DEFAULTS: tuple[tuple[str, Any], ...] = (
    ("last_balance", None),
    ("last_rate_hourly", 0.0),
    ("last_assignments_count", 0),
    ("last_sessions", []),
    ("active_since_ts", None),
    ("active_endpoint", None),
    ("session_start_balance", None),
    ("managed_binding", None),
)


def _has_active_cluster_runtime(cluster_info: dict[str, Any]) -> bool:
    """判断异构集群快照中是否存在任意在线或计费中的槽位。"""
    if not cluster_info.get("active"):
        return False
    slots_map = cluster_info.get("slots") or {}
    return any(
        isinstance(sv, dict)
        and bool(sv.get("online") or sv.get("endpoint") or float(sv.get("rate_hourly") or 0.0) > 0.01)
        for sv in slots_map.values()
    )


class SessionMonitorService:
    """负责 Colab 实时探测、每日/单次账本核算、保活心跳与闲置存档停卡。"""

    def __init__(self, backend: ColabPetBackend) -> None:
        self._b = backend
        self._store = backend._store
        self._secrets = backend._secrets
        self._bus = backend._bus
        self._ctx = backend._ctx

    # ------------------------------------------------------------------
    # Colab 实时状态探测与受管会话对账
    # ------------------------------------------------------------------

    def query_colab_live(self) -> dict[str, Any]:
        """调用 colab_cli 获取实时 CU 余额、可用加速卡与活跃会话，并无感纳管当前登录账号。"""
        colab_py = self._ctx.get_colab_py()
        if not colab_py.exists():
            raise FileNotFoundError(f"未找到 colab-cli 运行环境: {colab_py}")

        with self._store.lock:
            state = self._store.state
            prefs = state.get("prefs") or {}
            active_id = str(prefs.get("active_account_id") or "").strip()
            profiles = state.get("account_profiles") or []
            known_ids = {str(p.get("id") or "") for p in profiles if isinstance(p, dict)}
            need_userinfo = (not profiles) or (not active_id) or (active_id not in known_ids)

        cmd = [str(colab_py), str(self._ctx.cli_bridge_script), "probe"]
        if need_userinfo:
            cmd.append("--need-userinfo")
        if active_id:
            cmd.append(f"--active-acct={active_id}")

        proc = self._ctx.run_hidden(cmd, timeout=25)
        if proc.returncode != 0:
            err = redact_secrets((proc.stderr or proc.stdout or "").strip())
            raise RuntimeError(
                f"Colab 查询失败: {err.splitlines()[-1] if err else f'退出码 {proc.returncode}'}"
            )

        stdout = proc.stdout or ""
        matched = re.search(r"<<<COLAB_PET_JSON>>>(.*?)<<<END_COLAB_PET_JSON>>>", stdout, re.S)
        json_text = matched.group(1).strip() if matched else stdout.strip()
        if not json_text:
            raise RuntimeError("Colab 返回空结果")
        return json.loads(json_text)

    @staticmethod
    def _pick_primary_session(sessions: list[dict[str, Any]]) -> dict[str, Any] | None:
        return next(
            (s for s in sessions if s.get("name") == "comfy" and s.get("endpoint")),
            None,
        ) or (sessions[0] if sessions and sessions[0].get("endpoint") else None)

    @staticmethod
    def _is_matching_account(record_acct: str, active_aid: str) -> bool:
        """判断记录归属账号与当前激活账号是否兼容（任一为空视为兼容）。"""
        return (not record_acct) or (not active_aid) or (record_acct == active_aid)

    def _collect_known_endpoint_names(self) -> dict[str, str]:
        """从托管绑定与集群分槽中收集归属当前账号的已知 Endpoint -> 会话名映射。"""
        known_ep_names: dict[str, str] = {}
        with self._store.lock:
            state = self._store.state
            active_aid = str((state.get("prefs") or {}).get("active_account_id") or "").strip()
            mb = state.get("managed_binding")
            if isinstance(mb, dict):
                mb_ep = str(mb.get("endpoint") or "").strip()
                mb_sess = str(mb.get("session") or "").strip()
                mb_acct = str(mb.get("account_id") or "").strip()
                if mb_ep and mb_sess and mb_sess != "?" and self._is_matching_account(mb_acct, active_aid):
                    known_ep_names[mb_ep] = mb_sess

            raw_slots = state.get("cluster_slots")
            if isinstance(raw_slots, dict):
                for slot_key, sv in raw_slots.items():
                    if not isinstance(sv, dict):
                        continue
                    s_ep = str(sv.get("endpoint") or "").strip()
                    s_name = str(sv.get("session") or slot_key or "").strip()
                    s_acct = str(sv.get("account_id") or "").strip()
                    if s_ep and s_name and s_name != "?" and self._is_matching_account(s_acct, active_aid):
                        known_ep_names.setdefault(s_ep, s_name)
        return known_ep_names

    def _reconcile_known_sessions(self, sessions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """将已知托管绑定或集群分槽 Endpoint 对账还原为受管会话名，消除孤儿误报。"""
        known_ep_names = self._collect_known_endpoint_names()
        out: list[dict[str, Any]] = []
        for item in sessions:
            if not isinstance(item, dict):
                continue
            s_copy = dict(item)
            ep = str(s_copy.get("endpoint") or "").strip()
            cur_name = str(s_copy.get("name") or "").strip()
            if ep and ep in known_ep_names and (bool(s_copy.get("orphan")) or cur_name in ("", "?")):
                s_copy["name"] = known_ep_names[ep]
                s_copy["orphan"] = False
            out.append(s_copy)
        return out

    # ------------------------------------------------------------------
    # 每日消耗与单次开机账本核算
    # ------------------------------------------------------------------

    def _update_daily_balance_locked(self, balance: float, today: str) -> float:
        """在持状态锁下更新跨日/当日余额、充值增量、峰值与基准余额，返回本次差值。"""
        state = self._store.state
        prev_bal = state.get("last_balance")
        diff = 0.0

        if state.get("date") != today:
            start_bal = float(prev_bal) if prev_bal is not None else balance
            state["date"] = today
            state["day_start_balance"] = start_bal
            state["day_topup_cu"] = 0.0
            raw_diff = round(balance - start_bal, 4)
            if raw_diff > 0.05:
                state["day_topup_cu"] = raw_diff
                state["last_delta"] = raw_diff
                diff = raw_diff
            elif raw_diff < -0.0005:
                state["last_delta"] = raw_diff
                diff = raw_diff
            else:
                state["last_delta"] = 0.0
            state["last_balance"] = balance
        else:
            if state.get("day_start_balance") is None:
                state["day_start_balance"] = balance
            if prev_bal is not None:
                raw_diff = round(balance - float(prev_bal), 4)
                if raw_diff < -0.0005:
                    state["last_delta"] = raw_diff
                    diff = raw_diff
                elif raw_diff > 0.05:
                    state["day_topup_cu"] = round(float(state.get("day_topup_cu") or 0.0) + raw_diff, 4)
                    state["last_delta"] = raw_diff
                    diff = raw_diff
            state["last_balance"] = balance

        day_start_f = float(state["day_start_balance"])
        day_topup_f = float(state.get("day_topup_cu") or 0.0)
        state["day_consumed_cu"] = round(max(0.0, day_start_f + day_topup_f - balance), 4)

        old_peak = state.get("peak_balance")
        cand_peaks = [balance, day_start_f]
        if isinstance(old_peak, (int, float)):
            cand_peaks.append(float(old_peak))
        state["peak_balance"] = round(max(cand_peaks), 4)

        baseline = float(state.get("baseline_balance") or 0.0)
        if baseline <= 0.0 or balance > baseline:
            state["baseline_balance"] = default_baseline(balance)
        return diff

    def _prune_stale_endpoint_caches_locked(self, active_eps: set[str]) -> None:
        """在持状态锁下清理已下线端点的隧道缓存与内存凭据（保留异构集群活跃槽位）。"""
        state = self._store.state
        cf_cfg_p = self._ctx.resolve_cf_config_path()
        keep_eps = set(active_eps) | self._store.get_active_cluster_endpoints()
        raw_cache = state.get("tunnel_cache")
        if isinstance(raw_cache, dict):
            state["tunnel_cache"] = {
                ep: url
                for ep, url in raw_cache.items()
                if ep in keep_eps
                and isinstance(url, str)
                and self._ctx.is_allowed_tunnel_url(url, config_path=cf_cfg_p)
            }
        else:
            state["tunnel_cache"] = {}
        self._secrets.retain_endpoints(keep_eps)

    def _sync_live_account_meta_locked(
        self, live: dict[str, Any], balance: float, is_running: bool
    ) -> None:
        """在持状态锁下同步当前探测结果携带的账号身份与余额档案。"""
        acct_meta = live.get("account_meta")
        if not (isinstance(acct_meta, dict) and acct_meta.get("id")):
            return
        aid = re.sub(r"[^a-zA-Z0-9_-]", "", str(acct_meta["id"]))[:24]
        if not aid:
            return
        state = self._store.state
        prefs = state.setdefault("prefs", dict(DEFAULT_PREFS))
        prefs["active_account_id"] = aid
        upsert_account_profile(
            state,
            aid,
            str(acct_meta.get("email_masked") or "***"),
            balance=balance,
            is_running=is_running,
        )

    def _update_active_session_window_locked(
        self,
        *,
        is_running: bool,
        cur_ep: str | None,
        prev_bal: Any,
        balance: float,
        rate_hourly: float,
        now_ts: int,
        day_start_f: float,
        day_topup_f: float,
        diff: float,
    ) -> None:
        """在持状态锁下维护连续开机起点、单次开机起始余额与活跃计时。"""
        b = self._b
        state = self._store.state
        if not is_running:
            state["active_since_ts"] = None
            state["active_endpoint"] = None
            state["session_start_balance"] = None
            state["managed_binding"] = None
            b._last_activity_m = None
            b._idle_warned_ep = None
            b._last_keepalive_ts = 0.0
            b._idle_archive_retry_after_m = 0.0
            return

        prev_since = state.get("active_since_ts")
        prev_ep = state.get("active_endpoint")
        if not prev_ep:
            prev_primary = self._pick_primary_session(list(state.get("last_sessions") or []))
            if prev_primary:
                prev_ep = str(prev_primary.get("endpoint"))
        switched_ep = bool(cur_ep and prev_ep and cur_ep != prev_ep)
        if not prev_since or switched_ep:
            state["active_since_ts"] = now_ts
            state["active_endpoint"] = cur_ep
            b._last_activity_m = self._ctx.monotonic()
            b._idle_warned_ep = None
            state["session_start_balance"] = (
                float(prev_bal)
                if (not switched_ep and prev_bal is not None and float(prev_bal) >= balance)
                else balance
            )
            return

        if cur_ep:
            state["active_endpoint"] = cur_ep
        if b._last_activity_m is None:
            b._last_activity_m = self._ctx.monotonic()
            b._idle_warned_ep = None
        if state.get("session_start_balance") is None:
            state["session_start_balance"] = estimate_session_start_balance(
                balance, rate_hourly, int(prev_since), now_ts, day_start_f, day_topup_f
            )
        elif diff > 0.05:
            state["session_start_balance"] = float(state["session_start_balance"]) + diff

    def _update_managed_binding_locked(
        self, is_running: bool, primary_sess: dict[str, Any] | None
    ) -> None:
        """在持状态锁下刷新主实例与激活账号的托管绑定记录。"""
        if not (
            is_running
            and primary_sess
            and primary_sess.get("endpoint")
            and primary_sess.get("name") not in (None, "?")
        ):
            return
        state = self._store.state
        active_aid = self._store.get_active_account_id()
        cur_bind = state.get("managed_binding")
        ep_str = str(primary_sess["endpoint"])
        if not isinstance(cur_bind, dict) or cur_bind.get("endpoint") != ep_str:
            state["managed_binding"] = {
                "account_id": active_aid,
                "session": str(primary_sess["name"]),
                "endpoint": ep_str,
            }

    def _update_runtime_capabilities_locked(
        self,
        live: dict[str, Any],
        sessions: list[dict[str, Any]],
        assignments_count: int,
        rate_hourly: float,
    ) -> None:
        """在持状态锁下更新实测卡种费率（保留其余历史记录）、可用卡清单与紧缺解除状态。"""
        b = self._b
        state = self._store.state
        if assignments_count == 1 and len(sessions) == 1 and 0.02 < rate_hourly < 200.0:
            s0 = sessions[0]
            rkey = gpu_rate_key(s0.get("accelerator"), s0.get("machine_shape"))
            if rkey:
                rates_map = state.setdefault("measured_gpu_rates", dict(DEFAULT_MEASURED_GPU_RATES))
                rates_map[rkey] = round(rate_hourly, 2)

        for live_key, state_key in (
            ("eligible_gpus", "last_eligible_gpus"),
            ("ineligible_gpus", "last_ineligible_gpus"),
            ("eligible_tpus", "last_eligible_tpus"),
        ):
            if live.get(live_key) is not None:
                state[state_key] = [str(item).upper() for item in live[live_key] if item]
        if live.get("subscription_tier"):
            state["subscription_tier"] = str(live["subscription_tier"]).strip()
        for sess in sessions:
            acc = str(sess.get("accelerator") or "").upper()
            if acc and acc in b._scarce_gpus:
                b._scarce_gpus.pop(acc, None)

    def update_ledger(self, live: dict[str, Any]) -> None:
        """更新每日消耗账本、基准额度、连续开机计时，并同步写入当前多账号槽位。"""
        state = self._store.state
        now_ts = int(self._ctx.now())
        today = datetime.now().strftime("%Y-%m-%d")
        balance = float(live["balance"])
        rate_hourly = float(live["rate_hourly"])
        assignments_count = int(live["assignments_count"])
        sessions = self._reconcile_known_sessions(list(live.get("sessions", [])))
        is_running = assignments_count > 0 or rate_hourly > 0.01

        self._sync_live_account_meta_locked(live, balance, is_running)

        prev_bal = state.get("last_balance")
        diff = self._update_daily_balance_locked(balance, today)
        day_start_f = float(state["day_start_balance"])
        day_topup_f = float(state.get("day_topup_cu") or 0.0)

        primary_sess = self._pick_primary_session(sessions)
        cur_ep = str(primary_sess.get("endpoint")) if primary_sess else None
        self._update_active_session_window_locked(
            is_running=is_running,
            cur_ep=cur_ep,
            prev_bal=prev_bal,
            balance=balance,
            rate_hourly=rate_hourly,
            now_ts=now_ts,
            day_start_f=day_start_f,
            day_topup_f=day_topup_f,
            diff=diff,
        )

        state["last_rate_hourly"] = rate_hourly
        state["last_assignments_count"] = assignments_count
        state["last_sessions"] = sessions
        state["last_updated_ts"] = now_ts
        self._update_managed_binding_locked(is_running, primary_sess)
        self._update_runtime_capabilities_locked(live, sessions, assignments_count, rate_hourly)

        tier_key, _tier_lbl, inferred_base = infer_account_tier_and_baseline(
            balance,
            peak_balance=state.get("peak_balance"),
            eligible_gpus=state.get("last_eligible_gpus"),
            ineligible_gpus=state.get("last_ineligible_gpus"),
            explicit_tier=str(state.get("subscription_tier") or ""),
        )
        if tier_key != "unknown":
            state["baseline_balance"] = max(inferred_base, default_baseline(balance))

        active_eps = {str(s.get("endpoint")) for s in sessions if s.get("endpoint")}
        self._prune_stale_endpoint_caches_locked(active_eps)
        sync_active_account_ledger(state, balance=balance, is_running=is_running, tier=tier_key)

    # ------------------------------------------------------------------
    # 第 5 页安全状态与完整前端快照拼装
    # ------------------------------------------------------------------

    def _resolve_onboarding_account_status(
        self, prefs: dict[str, Any], colab_cli_ok: bool
    ) -> dict[str, Any]:
        """解析当前账号档案、授权初始化态以及未完成 OAuth 超时状态。"""
        b = self._b
        state = self._store.state
        profiles_raw = state.get("account_profiles") or []
        profiles = [dict(p) for p in profiles_raw if isinstance(p, dict)]
        active_id = str(prefs.get("active_account_id") or "").strip()
        if not active_id and profiles and not b._auth_missing:
            active_id = str(profiles[0].get("id") or "")
        active_acct = next((p for p in profiles if p.get("id") == active_id), None)

        has_last_balance = state.get("last_balance") is not None
        auth_initializing = bool(
            colab_cli_ok and (not b._auth_missing) and active_acct is None and (not has_last_balance)
        )
        colab_auth_ok = bool(
            colab_cli_ok
            and (not b._auth_missing)
            and (active_acct is not None or has_last_balance or auth_initializing)
        )

        pending_oauth = bool(
            b._pending_oauth_verifier
            and b._pending_oauth_ts > 0
            and (self._ctx.monotonic() - b._pending_oauth_ts) < 1200.0
        )
        if b._pending_oauth_verifier and not pending_oauth:
            b._pending_oauth_verifier = ""
            b._pending_oauth_ts = 0.0

        return {
            "profiles": profiles,
            "active_id": active_id,
            "active_acct": active_acct,
            "auth_initializing": auth_initializing,
            "colab_auth_ok": colab_auth_ok,
            "pending_oauth": pending_oauth,
        }

    def get_onboarding_state(self) -> dict[str, Any]:
        """组装前端第 5 页（账号保险箱与运行环境）所需的安全状态快照，绝不返回明文凭据。"""
        b = self._b
        state = self._store.state
        prefs = state.get("prefs") or {}
        colab_exe = self._ctx.get_colab_exe()
        colab_py = self._ctx.get_colab_py()
        rclone_exe = self._ctx.get_rclone_exe()
        rclone_conf = self._ctx.get_rclone_conf()

        colab_cli_ok = bool(colab_py and colab_py.is_file())
        colab_exe_ok = bool(colab_exe and colab_exe.is_file())
        rclone_exe_ok = bool(rclone_exe and rclone_exe.is_file())
        gdrive_ok = has_gdrive_save_section(rclone_conf)

        acct_st = self._resolve_onboarding_account_status(prefs, colab_cli_ok)
        active_acct = acct_st["active_acct"]
        cf_data = self._ctx.load_cf_secret_raw(
            config_path=self._ctx.resolve_cf_config_path(), include_secrets=False
        )
        oc = state.get("onboarding_cache") or {}
        needs_attention = (
            (not colab_cli_ok) or (not rclone_exe_ok) or b._auth_missing or (not gdrive_ok)
        )

        return {
            "needs_attention": needs_attention,
            "colab_cli_ok": colab_cli_ok,
            "colab_exe_ok": colab_exe_ok,
            "colab_auth_ok": acct_st["colab_auth_ok"],
            "auth_initializing": acct_st["auth_initializing"],
            "auth_missing": bool(b._auth_missing),
            "auth_expired": bool(b._auth_expired),
            "rclone_exe_ok": rclone_exe_ok,
            "gdrive_ok": gdrive_ok,
            "gdrive_free_gb": oc.get("gdrive_free_gb"),
            "gdrive_total_gb": oc.get("gdrive_total_gb"),
            "active_account_id": acct_st["active_id"],
            "active_account_label": str(active_acct.get("label") or "") if isinstance(active_acct, dict) else "",
            "active_account": active_acct,
            "accounts": acct_st["profiles"],
            "pending_oauth": acct_st["pending_oauth"],
            "oauth_awaiting_code": acct_st["pending_oauth"],
            "gdrive_binding": b._gdrive_bind_proc is not None,
            "cf_tunnel": cf_data,
            "tool_paths": {
                "colab_exe": str(colab_exe) if colab_exe else "",
                "colab_py": str(colab_py) if colab_py else "",
                "rclone_exe": str(rclone_exe) if rclone_exe else "",
                "rclone_conf": str(rclone_conf) if rclone_conf else "",
            },
        }

    def _extract_snapshot_inputs(self, live_data: dict[str, Any] | None) -> dict[str, Any]:
        """从实时数据或本地持久化缓存提取快照计算所需的原始运行指标。"""
        state = self._store.state
        if live_data is not None:
            return {
                "balance": float(live_data["balance"]),
                "rate_hourly": float(live_data["rate_hourly"]),
                "assignments_count": int(live_data["assignments_count"]),
                "sessions": self._reconcile_known_sessions(list(live_data.get("sessions", []))),
                "eligible_gpus": live_data.get("eligible_gpus"),
                "eligible_tpus": live_data.get("eligible_tpus"),
            }
        return {
            "balance": state.get("last_balance"),
            "rate_hourly": float(state.get("last_rate_hourly") or 0.0),
            "assignments_count": int(state.get("last_assignments_count") or 0),
            "sessions": self._reconcile_known_sessions(list(state.get("last_sessions") or [])),
            "eligible_gpus": state.get("last_eligible_gpus"),
            "eligible_tpus": state.get("last_eligible_tpus"),
        }

    def _resolve_snapshot_baseline(self, balance: float | None, eligible_gpus: Any) -> float:
        """计算余额进度条基准上限（自动结合峰值与订阅档位推断）。"""
        state = self._store.state
        baseline = float(state.get("baseline_balance") or 1000.0)
        if balance is not None and balance > baseline:
            return default_baseline(float(balance))
        if baseline == 1000.0 and (eligible_gpus is not None or state.get("subscription_tier")):
            t_key, _tlbl, inferred_base = infer_account_tier_and_baseline(
                balance,
                peak_balance=state.get("peak_balance"),
                eligible_gpus=eligible_gpus,
                ineligible_gpus=state.get("last_ineligible_gpus"),
                explicit_tier=str(state.get("subscription_tier") or ""),
            )
            if t_key != "unknown":
                return inferred_base
        return baseline

    def _determine_pet_mood_and_status(
        self,
        *,
        balance: float | None,
        error_msg: str | None,
        stale: bool,
        low_balance: bool,
        is_running: bool,
        comfy: dict[str, Any],
        long_running_alert: bool,
        has_orphan: bool,
        cluster_info: dict[str, Any],
    ) -> tuple[str, str, str]:
        """根据运行状态、余额预警、错误与孤儿实例决定桌宠心情、告警级别与状态标签。"""
        b = self._b
        if error_msg and balance is None:
            if error_msg == "正在连接 Colab...":
                return "idle", "running", "正在连接..."
            if b._auth_missing:
                return (
                    ("sad" if b._auth_expired else "idle"),
                    "warn",
                    ("登录已失效" if b._auth_expired else "待绑定账号"),
                )
            return "sad", "error", "连接异常"
        if low_balance:
            return "sad", "danger", "余额预警"
        if stale and error_msg:
            return "sad", "warn", "网络波动(缓存)"
        if is_running and comfy.get("has_error"):
            return "sad", "danger", "推理报错中断"
        if not is_running:
            return "proud", "idle", "已停机省积分"

        mood = "sad" if long_running_alert else "idle"
        status_level = "warn" if (long_running_alert or has_orphan) else "running"
        if long_running_alert:
            status_label = "长时运行提醒"
        elif has_orphan:
            status_label = "发现孤儿实例"
        elif cluster_info.get("active") and len(cluster_info.get("slots") or {}) >= 2:
            status_label = "双号集群计费中"
        else:
            status_label = "GPU 运行计费中"
        return mood, status_level, status_label

    def _compute_display_consumption(
        self,
        *,
        balance: float | None,
        rate_hourly: float,
        is_running: bool,
        active_since: Any,
        updated_ts: Any,
        now_ts: int,
    ) -> dict[str, Any]:
        """核算供 UI 展示的当日消费、充值额与单次开机消耗。"""
        state = self._store.state
        disp_bal = round(float(balance), 2) if balance is not None else None
        raw_day_start = state.get("day_start_balance")
        disp_day_start = round(float(raw_day_start), 2) if raw_day_start is not None else None
        disp_day_topup = round(float(state.get("day_topup_cu") or 0.0), 2)
        if disp_bal is not None and disp_day_start is not None:
            disp_day_consumed = round(max(0.0, disp_day_start + disp_day_topup - disp_bal), 2)
        else:
            disp_day_consumed = round(float(state.get("day_consumed_cu") or 0.0), 2)

        sess_start = state.get("session_start_balance")
        if is_running and sess_start is None and active_since and balance is not None:
            sess_start = estimate_session_start_balance(
                float(balance),
                rate_hourly,
                int(active_since),
                int(updated_ts or now_ts),
                float(raw_day_start if raw_day_start is not None else balance),
                float(state.get("day_topup_cu") or 0.0),
            )
        disp_session_consumed = (
            round(max(0.0, round(float(sess_start), 2) - disp_bal), 2)
            if (is_running and sess_start is not None and disp_bal is not None)
            else 0.0
        )
        return {
            "disp_bal": disp_bal,
            "disp_day_start": disp_day_start,
            "disp_day_topup": disp_day_topup,
            "disp_day_consumed": disp_day_consumed,
            "disp_session_consumed": disp_session_consumed,
        }

    def build_snapshot(
        self,
        live_data: dict[str, Any] | None,
        stale: bool,
        error_msg: str | None = None,
    ) -> dict[str, Any]:
        """拼装前端所需的完整状态快照。"""
        b = self._b
        state = self._store.state
        now_ts = int(self._ctx.now())
        prefs = dict(DEFAULT_PREFS)
        prefs.update(state.get("prefs") or {})

        raw = self._extract_snapshot_inputs(live_data)
        balance = raw["balance"]
        rate_hourly = raw["rate_hourly"]
        assignments_count = raw["assignments_count"]
        sessions = raw["sessions"]
        eligible_gpus = raw["eligible_gpus"]
        eligible_tpus = raw["eligible_tpus"]

        scarce_gpus = b._get_active_scarce_gpus()
        deploy_state = b.get_deploy_state()
        baseline = self._resolve_snapshot_baseline(balance, eligible_gpus)

        remaining_hours = (
            round(float(balance) / rate_hourly, 2)
            if (balance is not None and rate_hourly > 0.01)
            else None
        )
        active_since = state.get("active_since_ts")
        active_minutes = max(0, int((now_ts - int(active_since)) // 60)) if active_since else 0
        updated_ts = state.get("last_updated_ts")
        updated_str = (
            datetime.fromtimestamp(int(updated_ts)).strftime("%H:%M:%S")
            if updated_ts
            else "--:--:--"
        )

        comfy = b.get_comfy_state()
        onboarding_raw = b._get_onboarding_state()
        cluster_info, onboarding = b._cluster_svc.build_cluster_snapshot_and_enrich_onboarding(
            onboarding_raw,
            current_rate_hourly=rate_hourly,
            active_account_id=str(prefs.get("active_account_id") or ""),
            comfy_state=comfy,
        )
        is_running = (
            assignments_count > 0
            or rate_hourly > 0.01
            or _has_active_cluster_runtime(cluster_info)
        )

        warn_bal = float(prefs.get("warn_balance", 50.0))
        warn_hrs = float(prefs.get("warn_hours", 10.0))
        long_warn = int(prefs.get("long_run_warn_minutes", 180))
        low_balance = balance is not None and (
            float(balance) <= warn_bal or (remaining_hours is not None and remaining_hours <= warn_hrs)
        )
        has_orphan = any(s.get("orphan") for s in sessions)
        long_running_alert = is_running and active_minutes >= long_warn

        mood, status_level, status_label = self._determine_pet_mood_and_status(
            balance=balance,
            error_msg=error_msg,
            stale=stale,
            low_balance=low_balance,
            is_running=is_running,
            comfy=comfy,
            long_running_alert=long_running_alert,
            has_orphan=has_orphan,
            cluster_info=cluster_info,
        )
        balance_pct = (
            max(0.0, min(100.0, round(float(balance) / baseline * 100.0, 2)))
            if (balance is not None and baseline > 0)
            else 0.0
        )
        rate_pct = max(0.0, min(100.0, round(rate_hourly / 15.0 * 100.0, 2)))
        cons = self._compute_display_consumption(
            balance=balance,
            rate_hourly=rate_hourly,
            is_running=is_running,
            active_since=active_since,
            updated_ts=updated_ts,
            now_ts=now_ts,
        )
        safe_err = redact_secrets(error_msg) if error_msg is not None else None

        return {
            "ok": safe_err is None,
            "stale": stale,
            "error_msg": safe_err,
            "balance": cons["disp_bal"],
            "baseline_balance": round(baseline, 2),
            "balance_pct": balance_pct,
            "rate_hourly": round(rate_hourly, 2),
            "rate_pct": rate_pct,
            "remaining_hours": remaining_hours,
            "assignments_count": assignments_count,
            "sessions": sessions,
            "is_running": is_running,
            "has_orphan": has_orphan,
            "active_minutes": active_minutes,
            "active_since_ts": int(active_since) if (is_running and active_since) else None,
            "long_running_alert": long_running_alert,
            "low_balance": low_balance,
            "mood": mood,
            "status_level": status_level,
            "status_label": status_label,
            "date": state.get("date"),
            "day_start_balance": cons["disp_day_start"],
            "day_consumed_cu": cons["disp_day_consumed"],
            "day_topup_cu": cons["disp_day_topup"],
            "session_consumed_cu": cons["disp_session_consumed"],
            "last_delta": round(float(state.get("last_delta") or 0.0), 3),
            "updated_str": updated_str,
            "updated_ts": updated_ts,
            "eligible_gpus": eligible_gpus,
            "eligible_tpus": eligible_tpus,
            "scarce_gpus": scarce_gpus,
            "measured_gpu_rates": dict(state.get("measured_gpu_rates") or DEFAULT_MEASURED_GPU_RATES),
            "deploy": deploy_state,
            "prefs": prefs,
            "comfy": comfy,
            "cluster": cluster_info,
            "onboarding": onboarding,
            "catalog": export_frontend_catalog(),
        }

    # ------------------------------------------------------------------
    # 线上状态串行刷新（含账号 epoch 围栏）
    # ------------------------------------------------------------------

    def _handle_auth_missing_response(
        self, live: dict[str, Any], start_epoch: int
    ) -> dict[str, Any]:
        """处理未登录/授权失效响应：重置运行态并保留集群槽位凭据，离线推送快照。"""
        b = self._b
        with self._store.lock:
            if b._account_epoch != start_epoch:
                return dict(b.last_snapshot)
            b._auth_missing = True
            b._auth_expired = bool(live.get("auth_expired"))
            b._consecutive_failures = 0
            state = self._store.state
            for field, default_val in _AUTH_RESET_DEFAULTS:
                state[field] = list(default_val) if isinstance(default_val, list) else default_val
            self._secrets.retain_endpoints(self._store.get_active_cluster_endpoints())
            self._store.save()
            err_text = (
                "Google 账号授权已失效，请重新绑定"
                if b._auth_expired
                else "尚未绑定 Google Colab 账号"
            )
            snap = b._build_snapshot(live_data=None, stale=True, error_msg=err_text)
            b.last_snapshot = snap
        b._reset_comfy_offline()
        self._bus.emit({"type": "state_sync", "snapshot": snap})
        return snap

    def _handle_fetch_exception(self, exc: Exception, start_epoch: int) -> dict[str, Any]:
        """处理在线探测异常：首次网络抖动宽限期内沿用缓存，其余标 stale 并脱敏报错。"""
        b = self._b
        with self._store.lock:
            if b._account_epoch != start_epoch:
                return dict(b.last_snapshot)
            b._consecutive_failures += 1
            state = self._store.state
            has_balance = state.get("last_balance") is not None
            last_updated = int(state.get("last_updated_ts") or 0)
            within_grace = has_balance and (int(self._ctx.now()) - last_updated < 180)
            if b._consecutive_failures == 1 and within_grace:
                snap = b._build_snapshot(live_data=None, stale=False, error_msg=None)
            else:
                snap = b._build_snapshot(
                    live_data=None, stale=True, error_msg=redact_secrets(str(exc))
                )
            b.last_snapshot = snap
        self._bus.emit({"type": "state_sync", "snapshot": snap})
        return snap

    def fetch_live_status(self) -> dict[str, Any]:
        """串行刷新一次线上状态并广播给前端；含账号 epoch 围栏防止切号期间旧结果污染新账本。"""
        b = self._b
        with self._store.lock:
            start_epoch = b._account_epoch
        epoch_lock = b._get_epoch_fetch_lock(start_epoch)
        with epoch_lock:
            try:
                live = b._query_colab_live()
                with self._store.lock:
                    if b._account_epoch != start_epoch:
                        return dict(b.last_snapshot)
                if live.get("auth_missing"):
                    return self._handle_auth_missing_response(live, start_epoch)

                with self._store.lock:
                    if b._account_epoch != start_epoch:
                        return dict(b.last_snapshot)
                    b._auth_missing = False
                    b._auth_expired = False
                    b._consecutive_failures = 0
                    b._update_ledger(live)
                    self._store.save()
                    snap = b._build_snapshot(live_data=live, stale=False, error_msg=None)
                    b.last_snapshot = snap
            except Exception as exc:
                return self._handle_fetch_exception(exc, start_epoch)
            self._bus.emit({"type": "state_sync", "snapshot": snap})
            return snap

    # ------------------------------------------------------------------
    # 智能保活心跳
    # ------------------------------------------------------------------

    def _send_keepalive_for_account(self, target_ep: str = "", account_id: str = "") -> bool:
        """为指定账号与 Endpoint 触发保活心跳，兼容底层 _send_colab_keepalive 桩测试。"""
        b = self._b
        prev_acct = getattr(b._keepalive_acct_ctx, "account_id", "")
        b._keepalive_acct_ctx.account_id = str(account_id or "").strip()
        try:
            return b._send_colab_keepalive(target_ep=target_ep)
        finally:
            b._keepalive_acct_ctx.account_id = prev_acct

    def send_colab_keepalive(self, target_ep: str = "") -> bool:
        """通过 Colab 官方通道执行一次无窗口三层保活心跳（支持定向 target_ep 与集群 --acct）。"""
        b = self._b
        colab_py = self._ctx.get_colab_py()
        if not colab_py.exists():
            return False
        ctx_acct = str(getattr(b._keepalive_acct_ctx, "account_id", "") or "").strip()
        try:
            cmd = [str(colab_py), str(self._ctx.cli_bridge_script), "keepalive"]
            if target_ep:
                cmd.append(str(target_ep))
            if ctx_acct:
                cmd.append(f"--acct={ctx_acct}")
            proc = self._ctx.run_hidden(cmd, timeout=25)
        except Exception:
            return False
        if proc.returncode != 0:
            return False
        matched = re.search(
            r"<<<COLAB_KEEPALIVE>>>(.*?)<<<END_COLAB_KEEPALIVE>>>",
            proc.stdout or "",
            re.S,
        )
        if not matched:
            return False
        try:
            data = json.loads(matched.group(1).strip())
            return bool(data.get("ok")) and int(data.get("pinged") or 0) > 0
        except Exception:
            return False

    def maybe_trigger_keepalive(self, now_m: float | None = None) -> None:
        """按 10 分钟节流周期向绑定的活跃 Colab 实例发送防断连保活心跳（连续纯空闲超 45m 自动停发）。"""
        b = self._b
        if b.is_deploying():
            return
        cur_m = self._ctx.monotonic() if now_m is None else float(now_m)
        if b._cluster_svc.has_active_slots():
            b._cluster_svc.maybe_trigger_cluster_keepalive(cur_m)
            return
        if not self._store.has_running_session():
            with self._store.lock:
                b._last_keepalive_ts = 0.0
                b._last_activity_m = None
            return

        interval_sec = self._ctx.keepalive_interval_sec
        max_idle_sec = self._ctx.keepalive_max_idle_sec
        retry_cooldown = self._ctx.keepalive_retry_cooldown_sec
        with self._store.lock:
            keep_alive = bool((self._store.state.get("prefs") or {}).get("keep_alive", True))
            if not keep_alive:
                return
            if b._last_activity_m is None:
                b._last_activity_m = cur_m
            if max(0.0, cur_m - b._last_activity_m) > max_idle_sec:
                return
            last_ts = b._last_keepalive_ts
            if last_ts > 0.0 and (cur_m - last_ts) < interval_sec:
                return
            start_epoch = b._account_epoch

        _, target_ep, _ = b._resolve_target_binding("comfy")
        if not b._keepalive_lock.acquire(blocking=False):
            return

        with self._store.lock:
            if b._account_epoch != start_epoch or (
                b._last_keepalive_ts > 0.0 and (cur_m - b._last_keepalive_ts) < interval_sec
            ):
                b._keepalive_lock.release()
                return
            b._last_keepalive_ts = cur_m

        def _worker() -> None:
            ok = False
            try:
                ok = b._send_colab_keepalive(target_ep=target_ep)
            except Exception:
                ok = False
            finally:
                if not ok and not b._monitor_stop.is_set():
                    with self._store.lock:
                        if b._account_epoch == start_epoch and b._last_keepalive_ts == cur_m:
                            b._last_keepalive_ts = max(
                                1e-6, cur_m - (interval_sec - retry_cooldown)
                            )
                b._keepalive_lock.release()

        self._ctx.threading_mod.Thread(target=_worker, daemon=True).start()

    # ------------------------------------------------------------------
    # 闲置归档校验与自动停卡拆解
    # ------------------------------------------------------------------

    def _run_idle_archive_sync(
        self, sname: str, start_epoch: int, cancel_tok: Any
    ) -> tuple[bool, bool, str]:
        """在闲置停卡前执行最多两轮强校验云盘同步，返回 `(need_archive, archive_ok, archive_err)`。"""
        b = self._b
        archive_needed = (not b.is_llm_mode_active()) and self._ctx.ctl_available()
        if not archive_needed:
            return False, True, ""
        archive_err = ""
        for attempt in (1, 2):
            with self._store.lock:
                if b._account_epoch != start_epoch:
                    return True, False, "__SWITCHED__"
                sess = self._store.pick_target_session(sname)
                ep_chk = str(
                    (sess.get("endpoint") if sess else None)
                    or self._store.state.get("active_endpoint")
                    or ""
                ).strip()
                sync_acct = self._store.resolve_single_node_account_id(ep_chk)
            sync_cmd = [*self._ctx.ctl_cmd_prefix(), "sync"]
            if sync_acct:
                sync_cmd.extend(["--account", sync_acct])
            sync_cmd.extend(["-s", sname])
            sync_res = b._runner.run(
                sync_cmd,
                timeout=120,
                extra_env=self._ctx.build_ctl_env(),
                cancel_token=cancel_tok,
                run_fn=self._ctx.run_hidden,
                task_name="idle_sync",
            )
            out_txt = sync_res.combined_output
            if sync_res.ok and ("<<<SYNC_VERIFIED>>>" in out_txt or "SKIP_LLM_SYNC" in out_txt):
                return True, True, ""
            archive_err = sync_res.error_summary or f"sync 退出码 {sync_res.returncode}"
            if attempt == 1:
                self._ctx.sleep(2.0)
        return True, False, archive_err

    def _get_comfy_online_and_busy(self) -> tuple[bool, int]:
        """读取远端服务在线态及当前合并忙碌任务数（ComfyUI + LLM）。"""
        b = self._b
        with b._comfy_lock:
            online = bool(b._comfy_state.get("online"))
            busy_cnt = (
                int(b._comfy_state.get("running_count") or 0)
                + int(b._comfy_state.get("pending_count") or 0)
                + int(b._comfy_state.get("llm_running_count") or 0)
            )
        return online, busy_cnt

    def _collect_idle_stop_context(self, cur_m: float) -> dict[str, Any] | None:
        """收集闲置停卡前置判定上下文；不满足触发前提时返回 None。"""
        b = self._b
        with self._store.lock:
            state = self._store.state
            prefs = state.get("prefs") or {}
            idle_stop_mins = int(prefs.get("idle_auto_stop_minutes", 60) or 0)
            if idle_stop_mins <= 0:
                return None
            policy = str(prefs.get("idle_archive_fail_policy") or "stop_billing").strip().lower()
            if policy not in ("stop_billing", "keep_alive"):
                policy = "stop_billing"
            if b._last_activity_m is None:
                b._last_activity_m = cur_m
                return None
            last_act = b._last_activity_m
            cur_ep = str(state.get("active_endpoint") or "__active__")
            warned_ep = b._idle_warned_ep
            start_epoch = b._account_epoch
            bound_acct = str(prefs.get("active_account_id") or "").strip()

        sname, target_ep, _ = b._resolve_target_binding("comfy")
        if not target_ep:
            sess_list = self._store.get_last_sessions()
            if len(sess_list) == 1 and sess_list[0].get("endpoint"):
                target_ep = str(sess_list[0]["endpoint"])

        comfy_online, busy_cnt = self._get_comfy_online_and_busy()
        if busy_cnt > 0:
            b.touch_activity()
            return None
        if not comfy_online:
            return None

        limit_sec = float(idle_stop_mins * 60)
        return {
            "idle_stop_mins": idle_stop_mins,
            "policy": policy,
            "cur_ep": cur_ep,
            "warned_ep": warned_ep,
            "start_epoch": start_epoch,
            "bound_acct": bound_acct,
            "sname": sname,
            "target_ep": target_ep,
            "idle_sec": max(0.0, cur_m - last_act),
            "limit_sec": limit_sec,
            "warn_sec": max(60.0, limit_sec - 120.0),
        }

    def _handle_idle_stop_result(
        self,
        res: dict[str, Any],
        *,
        archive_needed: bool,
        archive_ok: bool,
        archive_err: str,
        idle_stop_mins: int,
    ) -> None:
        """处理闲置停卡执行结果并向前端广播成功或失败告警。"""
        b = self._b
        archive_failed = bool(archive_needed and not archive_ok)
        if res.get("ok"):
            short_err = (redact_secrets(archive_err) or "云盘同步校验未通过")[:48]
            summary = (
                f"⚠️ 云盘归档重试仍失败（{short_err}），已按策略优先停卡停止计费！"
                if archive_failed
                else f"🛑 连续闲置满 {idle_stop_mins} 分钟，已归档并释放当前实例省积分"
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
            return

        fail_msg = redact_secrets(str(res.get("message") or "请立即手动停卡"))
        prefix = "归档与闲置自动停卡均失败" if archive_failed else "闲置自动停卡未成功"
        b._diag.error("idle_stop", f"{prefix}: {fail_msg}")
        with self._store.lock:
            b._idle_warned_ep = None
            b._idle_archive_retry_after_m = self._ctx.monotonic() + 60.0
        self._bus.emit(
            {
                "type": "idle_warning",
                "is_error": True,
                "summary": (
                    f"🚨 归档与自动停卡均失败：{fail_msg}"
                    if archive_failed
                    else f"⚠️ 闲置自动停卡未成功: {fail_msg}"
                ),
                "action": "keep_working",
            }
        )

    def _execute_idle_stop_worker(self, ctx: dict[str, Any], cancel_tok: Any) -> None:
        """在后台线程内执行二次闲置复核、强校验归档、保图留机分流与定向停卡。"""
        b = self._b
        start_epoch = ctx["start_epoch"]
        limit_sec = ctx["limit_sec"]
        try:
            if b.is_deploying() or b._upload_lock.locked() or not self._store.has_running_session():
                return
            with self._store.lock:
                if b._account_epoch != start_epoch:
                    return
            now_online, now_busy = self._get_comfy_online_and_busy()
            if not now_online or now_busy > 0:
                if now_busy > 0:
                    b.touch_activity()
                return
            if b._last_activity_m is None or (self._ctx.monotonic() - b._last_activity_m) < limit_sec:
                return

            archive_needed, archive_ok, archive_err = self._run_idle_archive_sync(
                ctx["sname"], start_epoch, cancel_tok
            )
            if archive_err == "__SWITCHED__":
                return
            with self._store.lock:
                if b._account_epoch != start_epoch:
                    return

            if archive_needed and (not archive_ok) and ctx["policy"] == "keep_alive":
                short_err = (redact_secrets(archive_err) or "云盘同步校验未通过")[:48]
                with self._store.lock:
                    b._idle_archive_retry_after_m = self._ctx.monotonic() + 600.0
                    b._idle_warned_ep = None
                self._bus.emit(
                    {
                        "type": "idle_archive_failed",
                        "is_error": True,
                        "summary": f"🚨 云盘归档重试失败（{short_err}）！已按「保图留机」保留实例，请尽快手动拉回并停卡",
                        "action": "sync_outputs",
                        "error_detail": redact_secrets(archive_err),
                    }
                )
                return

            res = b.stop_session_instance(
                session=ctx["sname"],
                endpoint=ctx["target_ep"] or None,
                account_id=ctx["bound_acct"],
            )
            if res.get("aborted_by_switch"):
                return
            self._handle_idle_stop_result(
                res,
                archive_needed=archive_needed,
                archive_ok=archive_ok,
                archive_err=archive_err,
                idle_stop_mins=ctx["idle_stop_mins"],
            )
        finally:
            b._idle_stop_lock.release()

    def maybe_trigger_idle_auto_stop(self, now_m: float | None = None) -> None:
        """若确认远端服务在线且连续全空闲达设定阈值（默认 60m），强校验归档并定向停卡。"""
        b = self._b
        if b.is_deploying() or b._upload_lock.locked() or b._action_lock.locked():
            return
        if not self._store.has_running_session():
            return

        cur_m = self._ctx.monotonic() if now_m is None else float(now_m)
        if b._cluster_svc.has_active_slots():
            b._cluster_svc.maybe_trigger_cluster_idle_auto_stop(cur_m)
            return
        if cur_m < b._idle_archive_retry_after_m:
            return

        ctx = self._collect_idle_stop_context(cur_m)
        if ctx is None:
            return

        if ctx["idle_sec"] >= ctx["limit_sec"]:
            if not b._idle_stop_lock.acquire(blocking=False):
                return
            hdl = b._tasks.submit(
                "idle_stop",
                lambda cancel_tok: self._execute_idle_stop_worker(ctx, cancel_tok),
                exclusive=False,
            )
            if not hdl.accepted:
                b._idle_stop_lock.release()
            return

        if ctx["idle_sec"] >= ctx["warn_sec"] and ctx["warned_ep"] != ctx["cur_ep"]:
            with self._store.lock:
                b._idle_warned_ep = ctx["cur_ep"]
            self._bus.emit(
                {
                    "type": "idle_warning",
                    "summary": f"⏳ 已闲置 {int(ctx['idle_sec'] // 60)} 分钟，约 2 分钟后将自动存档停卡（点我续期）",
                    "action": "keep_working",
                }
            )

    # ------------------------------------------------------------------
    # 后台监控守护循环
    # ------------------------------------------------------------------

    def _poll_running_service_tick(self) -> float:
        """在单机实例运行时按服务类型（comfy / llm / dual）执行一轮探测并返回下次等待秒数。"""
        b = self._b
        turl = b.resolve_active_tunnel_url(allow_fast_discover=True)
        target_sess = self._store.pick_target_session("comfy")
        cur_ep = str(target_sess.get("endpoint") or "") if target_sess else ""
        skind = b._get_ep_service_kind(cur_ep)

        if turl:
            if skind == "llm":
                b._stop_ws()
                cstate = b.poll_llm_http_once(turl, ep=cur_ep)
                return 2.5 if (cstate.get("running_count") or 0) > 0 else 8.0
            if skind == "dual":
                b._ensure_ws_target(turl)
                llm_u = b.resolve_active_llm_tunnel_url(allow_fast_discover=True) or ""
                cstate = b.poll_dual_once(turl, llm_u, ep=cur_ep)
                busy = (
                    (cstate.get("running_count") or 0)
                    + (cstate.get("pending_count") or 0)
                    + (cstate.get("llm_running_count") or 0)
                ) > 0
                return 2.5 if busy else 6.0
            b._ensure_ws_target(turl)
            cstate = b.poll_comfy_http_once(turl)
            busy = (cstate.get("running_count") or 0) + (cstate.get("pending_count") or 0) > 0
            return 2.5 if busy else 6.0

        if skind == "dual":
            b._stop_ws()
            llm_u = b.resolve_active_llm_tunnel_url(allow_fast_discover=True) or ""
            if llm_u:
                cstate = b.poll_dual_once("", llm_u, ep=cur_ep)
                return 2.5 if (cstate.get("llm_running_count") or 0) > 0 else 6.0

        b._reset_comfy_offline()
        return 5.0

    def _poll_single_monitor_tick(self) -> float:
        """执行监控守护线程的单轮调度逻辑，返回本轮休眠等待秒数。"""
        b = self._b
        if b.is_deploying():
            return 2.0
        if b._cluster_svc.has_active_slots():
            b._maybe_trigger_keepalive()
            wait_s = b._cluster_svc.poll_cluster_monitor_tick()
            b._maybe_trigger_idle_auto_stop()
            return wait_s
        if not self._store.has_running_session():
            b._maybe_trigger_keepalive()
            b._reset_comfy_offline()
            return 15.0
        b._maybe_trigger_keepalive()
        wait_s = self._poll_running_service_tick()
        b._maybe_trigger_idle_auto_stop()
        return wait_s

    def start_monitor(self, on_event: Callable[[dict[str, Any]], None]) -> None:
        """启动后台 ComfyUI / LLM 服务状态与显存监控守护线程。"""
        b = self._b
        self._bus.legacy_callback = on_event
        if b._monitor_thread and b._monitor_thread.is_alive():
            return
        b._monitor_stop.clear()

        def _loop() -> None:
            while not b._monitor_stop.is_set():
                b._wake_monitor.clear()
                try:
                    wait_s = self._poll_single_monitor_tick()
                except Exception as exc:
                    b._diag.error("monitor", f"监控线程单轮异常: {redact_secrets(str(exc))}")
                    wait_s = 5.0
                b._wake_monitor.wait(timeout=wait_s)

        b._monitor_thread = self._ctx.threading_mod.Thread(target=_loop, daemon=True)
        b._monitor_thread.start()

    def stop_monitor(self) -> None:
        """停止后台监控线程与 WebSocket 连接（退出时同步清理未完成部署进程）。"""
        b = self._b
        b._monitor_stop.set()
        b._wake_monitor.set()
        b._stop_ws()
        if b._deploy_proc is not None or b.is_deploying():
            b.cancel_deploy()
