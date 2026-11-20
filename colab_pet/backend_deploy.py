"""Colab 桌宠一键开机部署、热重载调参与安全停卡核实子服务。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from colab_core.colab_client import resolve_target_binding
from colab_core.runtime import (
    COMFY_PROFILES,
    DEFAULT_PREFS,
    LLM_PROFILES,
    PROFILE_LABELS,
    SUPPORTED_BOOT_GPUS,
    SUPPORTED_LLM_QUANTS,
    build_ctl_up_args,
    calc_phase_dl_total,
    classify_deploy_failure,
    consume_deploy_line,
    default_deploy_state,
    format_elapsed_cn,
    kill_process_tree,
    normalize_profiles,
    parse_llm_boot_params,
    recommend_fallback_gpu,
    short_gpu_name,
)
from colab_security import redact_secrets

if TYPE_CHECKING:
    if __package__:
        from .colab_backend import ColabPetBackend
    else:
        from colab_backend import ColabPetBackend


def _get_pool_sched() -> Any:
    """按双导入形态兼容获取号池调度器模块，避免循环依赖。"""
    if __package__:
        from .backend_cluster import _pool_sched
    else:
        from backend_cluster import _pool_sched
    return _pool_sched()


def _format_ctx_tag(ctx_size: int) -> str:
    """将上下文长度整数格式化为简洁 K 级展示标签。"""
    if ctx_size >= 262144:
        return "256K"
    if ctx_size >= 131072:
        return "128K"
    return f"{max(1, ctx_size // 1024)}K"


def _first_non_empty(*values: Any) -> str:
    """依次返回首个非空字符串值，均为空时返回空字符串。"""
    for val in values:
        text = str(val or "").strip()
        if text:
            return text
    return ""


@dataclass
class _BootDeployCtx:
    """单次开机部署或运行态热重载上下文对象。"""

    profiles: list[str]
    profile: str
    llm_prof: str | None
    comfy_profs: list[str]
    cohost_only_llm: bool
    display_profiles: list[str]
    is_dual: bool
    effective_dual: bool
    exec_order: list[str]
    gpu: str
    high_mem: bool
    minimax_r2v: bool
    minimax_pe: bool
    recreate: bool
    llm_quant: str
    llm_ctx: int
    llm_mmproj: bool
    llm27_quant: str
    llm27_ctx: int
    llm27_mmproj: bool
    target_kind: str
    profile_label: str
    cmd: list[str]
    pre_assignments: int
    start_epoch: int
    initial_deploy: dict[str, Any]
    account_id: str = ""
    cluster_slot_reconfigure: str = ""
    defer_pool_select: bool = False
    pool_acct_count: int = 0
    # 最终执行的 `colab_ctl up` 命令实际携带的 --account 作用域（空串表示全局 sessions.json）
    cli_scope_acct: str = ""


def _extract_cmd_account(cmd: list[str]) -> str:
    """从 `colab_ctl up` 命令行中提取 `--account` 参数值，未携带时返回空串。"""
    items = [str(x) for x in (cmd or [])]
    for idx, item in enumerate(items):
        if item == "--account" and idx + 1 < len(items):
            return items[idx + 1].strip()
        if item.startswith("--account="):
            return item.split("=", 1)[1].strip()
    return ""


class DeployOrchestrator:
    """负责 7 步一键部署、运行态 Qwen 参数热重载、主动取消部署及定向停卡核实。"""

    def __init__(self, backend: ColabPetBackend) -> None:
        self._b = backend
        self._store = backend._store
        self._secrets = backend._secrets
        self._bus = backend._bus
        self._ctx = backend._ctx

    # ------------------------------------------------------------------
    # 定向停卡与全部释放
    # ------------------------------------------------------------------

    def resolve_target_binding(
        self, session: str = "comfy", endpoint: str | None = None
    ) -> tuple[str, str, bool]:
        """解析目标操作实例的 `(session_name, endpoint, 是否纯无名孤儿)`。"""
        with self._store.lock:
            sessions = self._store.get_last_sessions()
            binding = self._store.state.get("managed_binding")
            active_aid = self._store.get_active_account_id()
        return resolve_target_binding(
            sessions=sessions,
            binding=binding,
            active_account_id=active_aid,
            session=session,
            endpoint=endpoint,
        )

    def _route_stop_to_cluster_slot(
        self,
        session: str,
        endpoint: str | None,
        account_id: str | None,
    ) -> dict[str, Any] | None:
        """存在活跃异构集群分槽时，将常规停卡请求路由至对应槽位释放流程。"""
        b = self._b
        if account_id is not None or endpoint is not None or not b._cluster_svc.has_active_slots():
            return None
        slots = b._cluster_svc.get_cluster_slots()
        req_sess = str(session or "").strip()
        if req_sess and req_sess != "comfy":
            if req_sess in slots:
                return b._cluster_svc.stop_cluster_slot(req_sess)
            return {
                "ok": False,
                "target_unresolved": True,
                "message": f"未找到指定集群槽位 '{req_sess}'，已取消停卡",
            }
        target_slot = "comfy" if "comfy" in slots else ("llm" if "llm" in slots else "")
        if target_slot:
            return b._cluster_svc.stop_cluster_slot(target_slot)
        return None

    def _resolve_stop_target(
        self, session: str, endpoint: str | None
    ) -> tuple[str, str, dict[str, Any] | None]:
        """锁定精准停卡目标实例；若无实例、仅孤儿或目标不明，返回对应的早退响应。"""
        b = self._b
        snap: dict[str, Any] | None = None
        sname, target_ep, orphan_only = b._resolve_target_binding(session=session, endpoint=endpoint)
        if not target_ep:
            snap = b.fetch_live_status()
            sname, target_ep, orphan_only = b._resolve_target_binding(session=session, endpoint=endpoint)
            if snap.get("ok", True) and snap.get("assignments_count", 0) == 0 and not (snap.get("sessions") or []):
                return sname, target_ep, {"ok": True, "message": "当前无活跃 Colab 实例", "snapshot": snap}
        if orphan_only and not target_ep:
            return sname, target_ep, {
                "ok": False,
                "orphan_only": True,
                "message": "当前仅为网页端孤儿实例，请右键「一键停卡」执行停止全部",
            }
        if not target_ep:
            return sname, target_ep, {
                "ok": False,
                "target_unresolved": True,
                "message": f"未能锁定会话 '{session}' 对应的目标实例，为防误停其他实例已拒绝执行，请核查或使用停止全部",
                **({"snapshot": snap} if snap is not None else {}),
            }
        return sname, target_ep, None

    def _execute_stop_session_cmd(
        self, sname: str, target_ep: str, expected_acct: str | None
    ) -> Any:
        """构造定向 `colab_ctl down` 命令并在隐藏子进程中执行。"""
        b = self._b
        cmd = [*self._ctx.ctl_cmd_prefix(), "down"]
        slot_acct = b._cluster_svc.resolve_account_for_endpoint(ep=target_ep) if target_ep else ""
        active_aid = self._store.get_active_account_id()
        down_acct = expected_acct if (expected_acct and expected_acct != active_aid) else slot_acct
        if down_acct:
            cmd.extend(["--account", down_acct])
        cmd.extend(["-s", sname])
        if target_ep:
            cmd.extend(["--endpoint", target_ep])
        return b._runner.run(
            cmd,
            timeout=60,
            extra_env=self._ctx.build_ctl_env(),
            run_fn=self._ctx.run_hidden,
            task_name="stop_session",
        )

    def _verify_and_cleanup_stopped_session(
        self, sname: str, target_ep: str, proc: Any, start_epoch: int
    ) -> dict[str, Any]:
        """停卡后校验账号纪元、实时复核是否已停计费，并清理临时凭据与托管绑定。"""
        b = self._b
        output = proc.combined_output
        with self._store.lock:
            if b._account_epoch != start_epoch:
                return {
                    "ok": False,
                    "aborted_by_switch": True,
                    "message": "停卡期间已切换账号，已放弃写回旧账号状态",
                }

        snap = b.fetch_live_status()
        rem_sessions = list(snap.get("sessions") or [])
        rem_eps = {str(s.get("endpoint") or "") for s in rem_sessions if s.get("endpoint")}

        if not proc.ok:
            tail = (output.splitlines() or [proc.error_summary or "停机指令异常"])[-1]
            return {"ok": False, "message": f"停卡失败: {tail}", "snapshot": snap}

        still_present = bool(target_ep and target_ep in rem_eps) or (
            not target_ep and int(snap.get("assignments_count") or 0) > 0
        )
        if still_present:
            ep_desc = target_ep or sname
            return {
                "ok": False,
                "still_running": True,
                "message": f"停卡未生效：目标实例 {ep_desc} 仍处于计费状态，请重试或点停止全部",
                "snapshot": snap,
            }

        with self._store.lock:
            if target_ep:
                b._drop_ep_ephemeral_locked(target_ep)
                mb = self._store.state.get("managed_binding")
                if isinstance(mb, dict) and str(mb.get("endpoint") or "") == target_ep:
                    self._store.state["managed_binding"] = None
            else:
                self._store.state["managed_binding"] = None
            self._store.save()

        if not rem_eps and int(snap.get("assignments_count") or 0) == 0:
            b._reset_comfy_offline()

        msg = (
            f"已释放目标实例 ({sname} · 保留其余 {len(rem_eps)} 台会话)"
            if rem_eps
            else f"已释放 Colab 实例 ({sname})"
        )
        return {
            "ok": True,
            "released_endpoint": target_ep,
            "remaining_count": len(rem_eps),
            "message": msg,
            "snapshot": snap,
        }

    def stop_session_instance(
        self,
        session: str = "comfy",
        endpoint: str | None = None,
        account_id: str | None = None,
    ) -> dict[str, Any]:
        """精准释放绑定会话对应的单个 Colab 实例，绝不误杀同账号并存的其他会话，并停后核实。"""
        b = self._b
        expected_acct = str(account_id).strip() if account_id is not None else None
        if expected_acct is not None and self._store.get_active_account_id() != expected_acct:
            return {"ok": False, "aborted_by_switch": True, "message": "账号已切换，取消原账号停卡"}

        if account_id is None and b.is_deploying():
            b.cancel_deploy()
            self._ctx.sleep(0.4)

        routed = self._route_stop_to_cluster_slot(session, endpoint, account_id)
        if routed is not None:
            return routed

        if not b._action_lock.acquire(blocking=False):
            return {"ok": False, "message": "另一项操作正在执行，请稍等片刻"}
        try:
            if expected_acct is not None and self._store.get_active_account_id() != expected_acct:
                return {"ok": False, "aborted_by_switch": True, "message": "账号已切换，取消原账号停卡"}
            if not self._ctx.ctl_available():
                return {"ok": False, "message": f"未找到控制脚本: {self._ctx.ctl_script.name}"}

            sname, target_ep, early_resp = self._resolve_stop_target(session, endpoint)
            if early_resp is not None:
                return early_resp

            with self._store.lock:
                start_epoch = b._account_epoch
            proc = self._execute_stop_session_cmd(sname, target_ep, expected_acct)
            return self._verify_and_cleanup_stopped_session(sname, target_ep, proc, start_epoch)
        except Exception as exc:
            return {"ok": False, "message": f"停卡异常: {redact_secrets(str(exc))}"}
        finally:
            b._action_lock.release()

    def stop_all_instances(self) -> dict[str, Any]:
        """独立操作：停止当前账号下全部 Colab 活跃实例与孤儿运行时并核实彻底释放。"""
        b = self._b
        if b._cluster_svc.has_active_slots():
            return b._cluster_svc.stop_cluster_slot("all")
        if b.is_deploying():
            b.cancel_deploy()
            self._ctx.sleep(0.4)
        if not b._action_lock.acquire(blocking=False):
            return {"ok": False, "message": "另一项操作正在执行，请稍等片刻"}
        try:
            if not self._ctx.ctl_available():
                return {"ok": False, "message": f"未找到控制脚本: {self._ctx.ctl_script.name}"}
            with self._store.lock:
                start_epoch = b._account_epoch
            proc = b._runner.run(
                [*self._ctx.ctl_cmd_prefix(), "down", "--all"],
                timeout=60,
                extra_env=self._ctx.build_ctl_env(),
                run_fn=self._ctx.run_hidden,
                task_name="stop_all",
            )
            output = proc.combined_output
            with self._store.lock:
                if b._account_epoch != start_epoch:
                    return {"ok": False, "aborted_by_switch": True, "message": "操作期间账号已切换"}
            snap = b.fetch_live_status()
            rem_cnt = max(
                int(snap.get("assignments_count") or 0),
                len(snap.get("sessions") or []),
            )
            if not proc.ok:
                tail = (output.splitlines() or [proc.error_summary or "停机指令异常"])[-1]
                return {"ok": False, "message": f"停机失败: {tail}", "snapshot": snap}
            if rem_cnt > 0:
                return {
                    "ok": False,
                    "still_running": True,
                    "message": f"停止全部未完全生效：仍有 {rem_cnt} 台实例未释放，请重试",
                    "snapshot": snap,
                }
            with self._store.lock:
                b._clear_ephemeral_caches_locked()
                self._store.state["tunnel_cache"] = {}
                self._store.state["managed_binding"] = None
                self._store.save()
            b._reset_comfy_offline()
            return {"ok": True, "message": "已关闭全部 Colab 实例", "snapshot": snap}
        except Exception as exc:
            return {"ok": False, "message": f"停机异常: {redact_secrets(str(exc))}"}
        finally:
            b._action_lock.release()

    # ------------------------------------------------------------------
    # 部署进度解析、参数校验与上下文准备
    # ------------------------------------------------------------------

    def consume_deploy_line(self, raw_line: str) -> str | None:
        """委托 `colab_core.runtime.consume_deploy_line` 推进 7 步进度并返回隧道 URL。"""
        b = self._b
        with self._store.lock:
            ctx = {
                "phase_idx": b._deploy_phase_idx,
                "phase_total": b._deploy_phase_total,
                "phase_prof": b._deploy_phase_prof,
                "phase_is_llm": b._deploy_phase_is_llm,
                "dl_done": b._deploy_dl_done,
                "dl_name": b._deploy_dl_name,
                "dl_total": b._deploy_dl_total,
                "captured_llm_tunnel": b._deploy_captured_llm_tunnel,
                "meta_warmed": b._deploy_meta_warmed,
            }
            tunnel_pat = b._tunnel_url_regex_pattern()
            tunnel_url, should_warm_meta = consume_deploy_line(
                raw_line,
                b._deploy_state,
                ctx,
                self._store.state.get("prefs") or {},
                tunnel_pattern=tunnel_pat,
            )
            b._deploy_phase_idx = int(ctx["phase_idx"])
            b._deploy_phase_total = int(ctx["phase_total"])
            b._deploy_phase_prof = str(ctx["phase_prof"])
            b._deploy_phase_is_llm = bool(ctx["phase_is_llm"])
            b._deploy_dl_done = int(ctx["dl_done"])
            b._deploy_dl_name = str(ctx["dl_name"])
            b._deploy_dl_total = int(ctx["dl_total"])
            b._deploy_captured_llm_tunnel = ctx["captured_llm_tunnel"]
            b._deploy_meta_warmed = bool(ctx["meta_warmed"])

        if tunnel_url is not None and should_warm_meta:
            self._ctx.threading_mod.Thread(
                target=lambda: b._fast_read_tunnel_url("", "comfy"),
                daemon=True,
            ).start()
        return tunnel_url

    @staticmethod
    def _parse_boot_profiles_and_gpu(cfg: dict[str, Any]) -> dict[str, Any]:
        """解析开机工作流组合、执行顺序及默认显卡档位。"""
        profiles = normalize_profiles(
            cfg.get("profiles"), fallback_profile=str(cfg.get("profile") or "anima")
        )
        profile = profiles[0]
        llm_prof = next((p for p in profiles if p in LLM_PROFILES), None)
        comfy_profs = [p for p in profiles if p in COMFY_PROFILES]
        cohost_only_llm = bool(cfg.get("_cohost_only_llm")) and (llm_prof is not None) and (not comfy_profs)
        cluster_slot_reconf = str(cfg.get("_cluster_slot_reconfigure") or "").strip().lower()
        preserved_raw = cfg.get("_preserved_profiles")
        display_profiles = (
            normalize_profiles(preserved_raw, fallback_profile=profile)
            if (cohost_only_llm or cluster_slot_reconf) and preserved_raw
            else list(profiles)
        )
        is_dual = bool(llm_prof and comfy_profs)
        effective_dual = bool(is_dual or cohost_only_llm or (cluster_slot_reconf and len(display_profiles) >= 2))
        exec_order = ([llm_prof] if llm_prof else []) + comfy_profs

        needs_big_gpu = (
            "qwen_llm" in profiles
            or "minimax" in profiles
            or "drama3" in profiles
            or len(profiles) >= 2
        )
        default_gpu = "G4" if needs_big_gpu else "L4"
        gpu = str(cfg.get("gpu") or default_gpu).strip().upper()
        if gpu not in SUPPORTED_BOOT_GPUS:
            gpu = default_gpu
        high_mem = False if gpu in ("L4", "G4") else bool(cfg.get("high_mem", False))

        return {
            "profiles": profiles,
            "profile": profile,
            "llm_prof": llm_prof,
            "comfy_profs": comfy_profs,
            "cohost_only_llm": cohost_only_llm,
            "display_profiles": display_profiles,
            "is_dual": is_dual,
            "effective_dual": effective_dual,
            "exec_order": exec_order,
            "gpu": gpu,
            "high_mem": high_mem,
            "minimax_r2v": bool(cfg.get("minimax_r2v", True)),
            "minimax_pe": bool(cfg.get("minimax_pe", True)),
            "recreate": bool(cfg.get("recreate", False)),
        }

    def _check_boot_onboarding_and_preflight(
        self, profiles: list[str], gpu: str, *, skip_preflight: bool
    ) -> dict[str, Any] | None:
        """核验控制脚本、引导态（CLI/OAuth/云盘）及启动前环境预检。"""
        b = self._b
        if not self._ctx.ctl_available():
            return {"ok": False, "message": f"未找到控制脚本: {self._ctx.ctl_script.name}"}

        ob = b._get_onboarding_state()
        if not ob["colab_cli_ok"] or not ob["rclone_exe_ok"]:
            return {
                "ok": False,
                "needs_onboarding": True,
                "target_tab": "env",
                "message": "缺少 colab-cli 或 rclone 组件，请先在第 5 页「云盘与环境」完成接入",
            }
        if ob["auth_missing"]:
            return {
                "ok": False,
                "needs_onboarding": True,
                "target_tab": "vault",
                "message": "尚未绑定 Google Colab 账号，请先在第 5 页「多账号」登录绑定",
            }
        if not ob["gdrive_ok"]:
            return {
                "ok": False,
                "needs_onboarding": True,
                "target_tab": "env",
                "message": "尚未绑定 Google Drive 云盘 (gdrive_save)，请先在第 5 页完成授权",
            }

        if not skip_preflight:
            pre = b.run_preflight(deep=False, profiles=profiles, gpu=gpu)
            if not pre.get("ok"):
                return {
                    "ok": False,
                    "preflight_failed": True,
                    "preflight": pre,
                    "message": str(pre.get("blocking_message") or "启动前环境检查未通过"),
                }
        return None

    def _validate_and_parse_boot_request(
        self, cfg: dict[str, Any], *, skip_preflight: bool = False
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        """解析开机参数并依次核验本地工具、OAuth、rclone 与 preflight。"""
        parsed = self._parse_boot_profiles_and_gpu(cfg)
        err_resp = self._check_boot_onboarding_and_preflight(
            parsed["profiles"], parsed["gpu"], skip_preflight=skip_preflight
        )
        if err_resp is not None:
            return None, err_resp
        return parsed, None

    def _build_deploy_cmd(
        self,
        *,
        profiles: list[str],
        gpu: str,
        high_mem: bool,
        recreate: bool,
        endpoint: str,
        minimax_r2v: bool,
        minimax_pe: bool,
        llm_quant: str,
        llm_ctx: int,
        llm_mmproj: bool,
        llm27_quant: str,
        llm27_ctx: int,
        llm27_mmproj: bool,
        cohost_only_llm: bool,
        account_id: str,
    ) -> list[str]:
        """构造无窗口 `colab_ctl up` 部署命令，单号默认模式不附带多余 `--account` 参数。"""
        extra_ctl_kwargs: dict[str, Any] = {"account_id": account_id} if account_id else {}
        return build_ctl_up_args(
            python_exe=self._ctx.hidden_python_exe(),
            ctl_script=self._ctx.ctl_script,
            session="comfy",
            profiles=profiles,
            gpu=gpu,
            high_mem=high_mem,
            recreate=recreate,
            endpoint=endpoint,
            minimax_r2v=minimax_r2v,
            minimax_pe=minimax_pe,
            llm_quant=llm_quant,
            llm_ctx=llm_ctx,
            llm_mmproj=llm_mmproj,
            llm27_quant=llm27_quant,
            llm27_ctx=llm27_ctx,
            llm27_mmproj=llm27_mmproj,
            cohost_comfy=cohost_only_llm,
            ctl_prefix=self._ctx.ctl_cmd_prefix(unbuffered=True),
            **extra_ctl_kwargs,
        )

    def _build_deploy_cmd_for_ctx(
        self, dctx: _BootDeployCtx, *, endpoint: str = "", account_id: str = ""
    ) -> list[str]:
        """基于已有 `_BootDeployCtx` 重建 `colab_ctl up` 命令行。"""
        return self._build_deploy_cmd(
            profiles=dctx.profiles,
            gpu=dctx.gpu,
            high_mem=dctx.high_mem,
            recreate=dctx.recreate,
            endpoint=endpoint,
            minimax_r2v=dctx.minimax_r2v,
            minimax_pe=dctx.minimax_pe,
            llm_quant=dctx.llm_quant,
            llm_ctx=dctx.llm_ctx,
            llm_mmproj=dctx.llm_mmproj,
            llm27_quant=dctx.llm27_quant,
            llm27_ctx=dctx.llm27_ctx,
            llm27_mmproj=dctx.llm27_mmproj,
            cohost_only_llm=dctx.cohost_only_llm,
            account_id=account_id,
        )

    def _persist_boot_prefs_and_init_state(
        self,
        cfg: dict[str, Any],
        parsed: dict[str, Any],
        cluster_slot_reconf: str,
        *,
        defer_pool_select: bool,
        pool_acct_count: int,
    ) -> dict[str, Any]:
        """持主状态锁持久化启动偏好并初始化 7 步部署状态机，返回组装上下文所需快照。"""
        b = self._b
        cohost_only_llm = parsed["cohost_only_llm"]
        is_hot_reload = bool(cohost_only_llm or cluster_slot_reconf)
        gpu = parsed["gpu"]
        exec_order = parsed["exec_order"]

        with self._store.lock:
            prefs = self._store.state.setdefault("prefs", dict(DEFAULT_PREFS))
            acct_profiles = [
                p for p in (self._store.state.get("account_profiles") or [])
                if isinstance(p, dict) and p.get("id")
            ]
            active_aid = str(prefs.get("active_account_id") or "").strip()
            auto_pool_pref = bool(prefs.get("auto_pool_schedule", True))
            llm_params = parse_llm_boot_params(cfg, prefs)
            llm_quant, llm_ctx, llm_mmproj, llm27_quant, llm27_ctx, llm27_mmproj = llm_params

            if is_hot_reload and parsed["llm_prof"] == "qwen_27b":
                llm_updates = {
                    "boot_llm27_quant": llm27_quant,
                    "boot_llm27_ctx": llm27_ctx,
                    "boot_llm27_mmproj": llm27_mmproj,
                }
            elif is_hot_reload:
                llm_updates = {
                    "boot_llm_quant": llm_quant,
                    "boot_llm_ctx": llm_ctx,
                    "boot_llm_mmproj": llm_mmproj,
                }
            else:
                llm_updates = {
                    "boot_profiles": list(parsed["profiles"]),
                    "boot_profile": parsed["profile"],
                    "boot_gpu": gpu,
                    "boot_high_mem": parsed["high_mem"],
                    "boot_minimax_r2v": parsed["minimax_r2v"],
                    "boot_minimax_pe": parsed["minimax_pe"],
                    "boot_llm_quant": llm_quant,
                    "boot_llm_ctx": llm_ctx,
                    "boot_llm_mmproj": llm_mmproj,
                    "boot_llm27_quant": llm27_quant,
                    "boot_llm27_ctx": llm27_ctx,
                    "boot_llm27_mmproj": llm27_mmproj,
                }
            prefs.update(llm_updates)
            if not is_hot_reload and "cluster_mode" in cfg:
                prefs["cluster_split_mode"] = str(cfg.get("cluster_mode") or "").strip().lower() != "cohost"
            self._store.save()
            b.last_snapshot["prefs"] = dict(prefs)

            pre_assignments = int(self._store.state.get("last_assignments_count") or 0)
            now_ts = int(self._ctx.now())
            b._deploy_cancelled = False
            b._deploy_meta_warmed = False
            b._deploy_dl_name = ""
            b._deploy_dl_done = 0
            first_phase = exec_order[0]
            b._deploy_phase_idx = 1
            b._deploy_phase_total = len(exec_order)
            b._deploy_phase_prof = first_phase
            b._deploy_phase_is_llm = first_phase in LLM_PROFILES
            b._deploy_captured_llm_tunnel = None
            b._deploy_dl_total = calc_phase_dl_total(
                first_phase,
                llm_quant=llm_quant,
                llm_mmproj=llm_mmproj,
                llm27_mmproj=llm27_mmproj,
                minimax_r2v=parsed["minimax_r2v"],
                minimax_pe=parsed["minimax_pe"],
            )

            if is_hot_reload:
                init_title = f"热重载 LLM ({gpu})"
                init_detail = "正在热重启推理引擎（保留现有实例与隧道）..."
            elif defer_pool_select:
                init_title = f"校验算力号池 ({gpu})"
                init_detail = f"正在并发探测 {max(2, int(pool_acct_count))} 个账号余额并优选最佳开机账号..."
            else:
                init_title = f"准备开机 ({gpu})"
                init_detail = "正在启动本地无窗口部署进程..."

            b._deploy_state = {
                "active": True,
                "profile": parsed["display_profiles"][0],
                "profiles": list(parsed["display_profiles"]),
                "gpu": gpu,
                "high_mem": parsed["high_mem"],
                "step_idx": 0,
                "step_total": 7,
                "stage_title": init_title,
                "sub_detail": init_detail,
                "pct": 2,
                "started_ts": now_ts,
                "elapsed_sec": 0,
                "error_msg": "",
                "stockout_gpu": None,
                "suggest_gpu": None,
            }
            initial_deploy = b.get_deploy_state()
            start_epoch = b._account_epoch

        return {
            "acct_profiles": acct_profiles,
            "active_aid": active_aid,
            "auto_pool_pref": auto_pool_pref,
            "llm_params": llm_params,
            "pre_assignments": pre_assignments,
            "initial_deploy": initial_deploy,
            "start_epoch": start_epoch,
        }

    def _resolve_deploy_target_endpoint(
        self, gpu: str, explicit_ep: str, reuse_ep_override: str
    ) -> str:
        """解析开机部署可复用的目标实例端点 ID（支持显式指定、绑定恢复与同卡型孤儿匹配）。"""
        if explicit_ep:
            return explicit_ep
        _, target_ep_up, _ = self._b._resolve_target_binding("comfy")
        if target_ep_up:
            return target_ep_up
        if reuse_ep_override:
            return reuse_ep_override
        cur_sessions = self._store.get_last_sessions()
        matched_sess = next(
            (s for s in cur_sessions if s.get("endpoint") and str(s.get("accelerator") or "").upper() == gpu),
            None,
        )
        if matched_sess is None and len(cur_sessions) == 1 and cur_sessions[0].get("endpoint"):
            matched_sess = cur_sessions[0]
        return str(matched_sess["endpoint"]) if matched_sess else ""

    def _prepare_deploy_ctx(
        self,
        cfg: dict[str, Any],
        parsed: dict[str, Any],
        *,
        defer_pool_select: bool = False,
        pool_acct_count: int = 0,
    ) -> _BootDeployCtx:
        """在持 `_deploy_lock` 下写入启动偏好、初始化部署状态机并构造启动命令。"""
        b = self._b
        cluster_slot_reconf = str(cfg.get("_cluster_slot_reconfigure") or "").strip().lower()
        explicit_acct = str(cfg.get("_target_account_id") or "").strip()
        explicit_ep = str(cfg.get("_target_endpoint") or "").strip()
        reuse_ep_override = str(cfg.get("_reuse_endpoint") or "").strip()

        init_info = self._persist_boot_prefs_and_init_state(
            cfg,
            parsed,
            cluster_slot_reconf,
            defer_pool_select=defer_pool_select,
            pool_acct_count=pool_acct_count,
        )
        llm_quant, llm_ctx, llm_mmproj, llm27_quant, llm27_ctx, llm27_mmproj = init_info["llm_params"]

        target_kind = "dual" if parsed["effective_dual"] else ("llm" if parsed["llm_prof"] else "comfy")
        if target_kind == "llm":
            b._stop_ws()
        with b._comfy_lock:
            b._comfy_state["service_kind"] = target_kind
            b._comfy_state["active_profiles"] = list(parsed["display_profiles"])
            cstate_init = dict(b._comfy_state)
        with self._store.lock:
            b.last_snapshot["comfy"] = cstate_init

        target_ep_up = self._resolve_deploy_target_endpoint(parsed["gpu"], explicit_ep, reuse_ep_override)

        # 关键兼容：单号模式下绝不传递多余的 --account 参数，保证原有测试命令形态不变
        cmd_account_id = ""
        if explicit_acct:
            cmd_account_id = explicit_acct
        elif (
            len(init_info["acct_profiles"]) >= 2
            and init_info["auto_pool_pref"]
            and not parsed["cohost_only_llm"]
            and init_info["active_aid"]
        ):
            cmd_account_id = init_info["active_aid"]

        cmd = self._build_deploy_cmd(
            profiles=parsed["profiles"],
            gpu=parsed["gpu"],
            high_mem=parsed["high_mem"],
            recreate=parsed["recreate"],
            endpoint=target_ep_up,
            minimax_r2v=parsed["minimax_r2v"],
            minimax_pe=parsed["minimax_pe"],
            llm_quant=llm_quant,
            llm_ctx=llm_ctx,
            llm_mmproj=llm_mmproj,
            llm27_quant=llm27_quant,
            llm27_ctx=llm27_ctx,
            llm27_mmproj=llm27_mmproj,
            cohost_only_llm=parsed["cohost_only_llm"],
            account_id=cmd_account_id,
        )
        profile_label = " + ".join(PROFILE_LABELS.get(p, p) for p in parsed["display_profiles"])
        self._bus.emit({"type": "deploy_progress", "deploy": init_info["initial_deploy"]})

        return _BootDeployCtx(
            profiles=parsed["profiles"],
            profile=parsed["profile"],
            llm_prof=parsed["llm_prof"],
            comfy_profs=parsed["comfy_profs"],
            cohost_only_llm=parsed["cohost_only_llm"],
            display_profiles=parsed["display_profiles"],
            is_dual=parsed["is_dual"],
            effective_dual=parsed["effective_dual"],
            exec_order=parsed["exec_order"],
            gpu=parsed["gpu"],
            high_mem=parsed["high_mem"],
            minimax_r2v=parsed["minimax_r2v"],
            minimax_pe=parsed["minimax_pe"],
            recreate=parsed["recreate"],
            llm_quant=llm_quant,
            llm_ctx=llm_ctx,
            llm_mmproj=llm_mmproj,
            llm27_quant=llm27_quant,
            llm27_ctx=llm27_ctx,
            llm27_mmproj=llm27_mmproj,
            target_kind=target_kind,
            profile_label=profile_label,
            cmd=cmd,
            pre_assignments=init_info["pre_assignments"],
            start_epoch=init_info["start_epoch"],
            initial_deploy=init_info["initial_deploy"],
            account_id=cmd_account_id,
            cluster_slot_reconfigure=cluster_slot_reconf,
            defer_pool_select=defer_pool_select,
            pool_acct_count=pool_acct_count,
        )

    # ------------------------------------------------------------------
    # 部署成功与失败收尾
    # ------------------------------------------------------------------

    def _mark_deploy_done_state(self, dctx: _BootDeployCtx, elapsed_total: int) -> None:
        """更新部署状态机为 100% 完成态并移除对应 GPU 稀缺标记。"""
        b = self._b
        if dctx.effective_dual:
            done_detail = "ComfyUI + Qwen3.8 双栖服务与隧道已就绪"
        elif dctx.llm_prof:
            done_detail = "Qwen3.8 引擎与隧道已就绪"
        else:
            done_detail = "ComfyUI 与隧道已就绪"

        if dctx.account_id:
            b._cluster_svc.clear_account_gpu_scarce(dctx.account_id, dctx.gpu)

        with self._store.lock:
            b._scarce_gpus.pop(dctx.gpu, None)
            b._deploy_state.update(
                {
                    "active": False,
                    "step_idx": int(b._deploy_state.get("step_total") or 7),
                    "pct": 100,
                    "stage_title": "部署完成",
                    "sub_detail": done_detail,
                    "error_msg": "",
                    "stockout_gpu": None,
                    "suggest_gpu": None,
                    "elapsed_sec": elapsed_total,
                }
            )

    def _finalize_cluster_llm_slot_reconfigure(
        self, dctx: _BootDeployCtx, captured_tunnel: str | None, elapsed_cn: str
    ) -> None:
        """完成异构集群 LLM 分槽热重载收尾：刷新槽位隧道、同步 Comfy 状态并广播事件。"""
        b = self._b
        llm_slot = b._cluster_svc.get_slot("llm") or {}
        comfy_slot = b._cluster_svc.get_slot("comfy")
        target_ep = str(llm_slot.get("endpoint") or "")
        _, r_ep, fast_u = b._tunnel_svc.fast_read_slot_tunnel_url(
            target_ep, "comfy", account_id=dctx.account_id
        )
        target_ep = r_ep or target_ep
        meta_ready, _ = self._secrets.get_ep_meta_and_key(target_ep)
        used_q = dctx.llm27_quant if dctx.llm_prof == "qwen_27b" else dctx.llm_quant
        used_ctx = dctx.llm27_ctx if dctx.llm_prof == "qwen_27b" else dctx.llm_ctx
        meta_dict = meta_ready if isinstance(meta_ready, dict) else {}
        llm_url_final = _first_non_empty(
            b._deploy_captured_llm_tunnel,
            meta_dict.get("llm_tunnel_url"),
            meta_dict.get("tunnel_url"),
            fast_u,
            captured_tunnel,
            b._read_cf_public_llm_url(),
            llm_slot.get("tunnel_url"),
        ) or None
        with self._store.lock:
            slots_d = self._store.state.get("cluster_slots")
            llm_rec = slots_d.get("llm") if isinstance(slots_d, dict) else None
            if isinstance(llm_rec, dict):
                if target_ep:
                    llm_rec["endpoint"] = target_ep
                if llm_url_final:
                    llm_rec["tunnel_url"] = llm_url_final
                    if target_ep:
                        self._store.set_tunnel_cache(target_ep, str(llm_url_final))
                llm_rec["online"] = True
            self._store.save()

        comfy_url_keep = str((comfy_slot or {}).get("tunnel_url") or "") or None
        with b._comfy_lock:
            b._comfy_state.update(
                {
                    "service_kind": "dual" if comfy_slot else "llm",
                    "active_profiles": list(dctx.display_profiles),
                    "comfy_online": bool(comfy_slot and comfy_slot.get("online")),
                    "llm_online": True,
                    "tunnel_url": comfy_url_keep if comfy_slot else llm_url_final,
                    "llm_tunnel_url": llm_url_final,
                    "model_name": b._infer_default_llm_model_name(meta_dict),
                    "quant": str(meta_dict.get("quant") or used_q),
                    "ctx_size": int(meta_dict.get("ctx_size") or used_ctx),
                }
            )
            cstate = dict(b._comfy_state)
        with self._store.lock:
            b.last_snapshot["comfy"] = cstate

        snap = b.fetch_live_status()
        b._wake_monitor.set()
        ctx_tag = _format_ctx_tag(int(cstate.get("ctx_size") or used_ctx))
        self._bus.emit(
            {
                "type": "deploy_done",
                "deploy": b.get_deploy_state(),
                "snapshot": snap,
                "tunnel_url": llm_url_final,
                "llm_tunnel_url": llm_url_final,
                "summary": f"🎉 {dctx.gpu} 推理已热重载至 {used_q}·{ctx_tag}({elapsed_cn})！",
                "action": "open_llm_webui",
            }
        )

    @staticmethod
    def _need_post_deploy_meta_refresh(
        dctx: _BootDeployCtx,
        meta_ready: dict[str, Any],
        key_ready: str,
        comfy_key_ready: bool,
    ) -> bool:
        """判断部署完成后是否需要额外速读远端元数据与隧道凭据。"""
        if dctx.cohost_only_llm or not meta_ready:
            return True
        if dctx.llm_prof and not key_ready:
            return True
        if dctx.effective_dual and meta_ready.get("kind") != "dual":
            return True
        return dctx.target_kind in ("comfy", "dual") and not comfy_key_ready

    def _sync_post_deploy_meta_and_tunnel(
        self, dctx: _BootDeployCtx, captured_tunnel: str | None
    ) -> tuple[str | None, dict[str, Any], str | None]:
        """部署完成后定位目标端点、写回隧道缓存并按需拉取远端自描述元数据与密钥。"""
        b = self._b
        b.fetch_live_status()
        with self._store.lock:
            sessions = self._store.get_last_sessions()
            primary = self._store.pick_target_session("comfy")
            target_ep = str((primary or {}).get("endpoint") or "") or None
            if not target_ep and sessions:
                target_ep = str(sessions[0].get("endpoint") or "") or None
            if captured_tunnel and target_ep and not dctx.cohost_only_llm:
                self._store.set_tunnel_cache(target_ep, captured_tunnel)
                self._store.save()

        if target_ep:
            b._record_managed_binding("comfy", target_ep, cli_scope=dctx.cli_scope_acct)

        meta_ready, key_ready = self._secrets.get_ep_meta_and_key(target_ep or "")
        comfy_key_ready = bool(target_ep and self._secrets.comfy_secrets_by_ep.get(target_ep))
        need_probe_scope = bool(dctx.llm_prof) or (dctx.target_kind in ("comfy", "dual") and not comfy_key_ready)
        if need_probe_scope:
            if self._need_post_deploy_meta_refresh(dctx, meta_ready, key_ready, comfy_key_ready):
                _, r_ep, fast_u = b._fast_read_tunnel_url(target_ep or "", "comfy")
                if r_ep and not target_ep:
                    target_ep = r_ep
                    b._record_managed_binding("comfy", target_ep, cli_scope=dctx.cli_scope_acct)
                if fast_u and (not captured_tunnel or dctx.cohost_only_llm):
                    captured_tunnel = fast_u
                meta_ready, _ = self._secrets.get_ep_meta_and_key(target_ep or "")
        elif target_ep:
            self._secrets.set_ep_meta(target_ep, {"kind": "comfy", "port": 8188}, runtime_secret="")

        return target_ep, meta_ready, captured_tunnel

    def _apply_llm_meta_to_comfy_state_locked(
        self, dctx: _BootDeployCtx, meta_ready: dict[str, Any]
    ) -> None:
        """在已持 `_comfy_lock` 下写入 LLM 模型名、量化、上下文与显存指标。"""
        b = self._b
        used_q = dctx.llm27_quant if dctx.llm_prof == "qwen_27b" else dctx.llm_quant
        used_ctx = dctx.llm27_ctx if dctx.llm_prof == "qwen_27b" else dctx.llm_ctx
        if dctx.target_kind == "llm":
            b._comfy_state["ws_connected"] = False
        b._comfy_state["model_name"] = b._infer_default_llm_model_name(meta_ready)
        b._comfy_state["quant"] = str(meta_ready.get("quant") or used_q)
        b._comfy_state["ctx_size"] = int(meta_ready.get("ctx_size") or used_ctx)
        if meta_ready.get("gpu_name"):
            b._comfy_state["gpu_name"] = short_gpu_name(str(meta_ready["gpu_name"]))
        v_u = round(float(meta_ready.get("vram_used_gb") or 0.0), 1)
        v_t = round(float(meta_ready.get("vram_total_gb") or 0.0), 1)
        if v_t > 0:
            b._comfy_state["vram_used_gb"] = v_u
            b._comfy_state["vram_total_gb"] = v_t
            b._comfy_state["vram_pct"] = round(v_u * 100.0 / v_t, 1)

    def _update_post_deploy_comfy_state(
        self,
        dctx: _BootDeployCtx,
        meta_ready: dict[str, Any],
        captured_tunnel: str | None,
    ) -> tuple[dict[str, Any], str | None]:
        """根据远端元数据更新内存态 `_comfy_state` 与快照，并返回最终 LLM 隧道 URL。"""
        b = self._b
        meta_dict = meta_ready if isinstance(meta_ready, dict) else {}
        fallback_meta_u = meta_dict.get("tunnel_url") if meta_dict.get("kind") in ("comfy", "dual") else ""
        comfy_url_meta = _first_non_empty(meta_dict.get("comfy_tunnel_url"), fallback_meta_u)
        llm_url_final = _first_non_empty(
            b._deploy_captured_llm_tunnel,
            meta_dict.get("llm_tunnel_url"),
            captured_tunnel if dctx.target_kind == "llm" else "",
            b._read_cf_public_llm_url() if dctx.llm_prof else "",
        ) or None
        with b._comfy_lock:
            b._comfy_state["service_kind"] = dctx.target_kind
            b._comfy_state["active_profiles"] = list(dctx.display_profiles)
            if dctx.cohost_only_llm and comfy_url_meta:
                b._comfy_state["tunnel_url"] = comfy_url_meta
            elif captured_tunnel and not dctx.cohost_only_llm:
                b._comfy_state["tunnel_url"] = captured_tunnel
            b._comfy_state["comfy_online"] = dctx.target_kind in ("comfy", "dual")
            b._comfy_state["llm_online"] = dctx.target_kind in ("llm", "dual")
            b._comfy_state["llm_tunnel_url"] = llm_url_final if dctx.target_kind in ("llm", "dual") else None
            if dctx.llm_prof:
                self._apply_llm_meta_to_comfy_state_locked(dctx, meta_dict)
            cstate = dict(b._comfy_state)
        with self._store.lock:
            b.last_snapshot["comfy"] = cstate
        return cstate, llm_url_final

    @staticmethod
    def _build_deploy_done_summary(
        dctx: _BootDeployCtx, cstate: dict[str, Any], elapsed_cn: str
    ) -> tuple[str, str]:
        """构造部署完成事件的操作意图 `action` 与中文提示摘要。"""
        used_q = dctx.llm27_quant if dctx.llm_prof == "qwen_27b" else dctx.llm_quant
        used_ctx = dctx.llm27_ctx if dctx.llm_prof == "qwen_27b" else dctx.llm_ctx
        if dctx.cohost_only_llm:
            ctx_tag = _format_ctx_tag(int(cstate.get("ctx_size") or used_ctx))
            return "open_llm_webui", f"🎉 {dctx.gpu} 推理已热重载至 {used_q}·{ctx_tag}({elapsed_cn})！"
        if dctx.effective_dual:
            return "open_comfy", f"🎉 {dctx.gpu}·{dctx.profile_label} 双栖就绪({elapsed_cn})！点我打开"
        if dctx.llm_prof:
            return "open_llm_webui", f"🎉 {dctx.gpu}·{dctx.profile_label} 部署完成({elapsed_cn})！点我打开对话页"
        return "open_comfy", f"🎉 {dctx.gpu}·{dctx.profile_label} 部署完成({elapsed_cn})！点我打开"

    def _finalize_deploy_success(
        self, dctx: _BootDeployCtx, captured_tunnel: str | None, elapsed_total: int, elapsed_cn: str
    ) -> None:
        """执行开机部署成功后的状态写回、元数据刷新与完成事件广播。"""
        b = self._b
        self._mark_deploy_done_state(dctx, elapsed_total)
        if dctx.cluster_slot_reconfigure == "llm":
            self._finalize_cluster_llm_slot_reconfigure(dctx, captured_tunnel, elapsed_cn)
            return

        _, meta_ready, captured_tunnel = self._sync_post_deploy_meta_and_tunnel(dctx, captured_tunnel)
        cstate, llm_url_final = self._update_post_deploy_comfy_state(dctx, meta_ready, captured_tunnel)
        b._wake_monitor.set()
        done_action, done_summary = self._build_deploy_done_summary(dctx, cstate, elapsed_cn)
        self._bus.emit(
            {
                "type": "deploy_done",
                "deploy": b.get_deploy_state(),
                "snapshot": b.get_cached_snapshot(),
                "tunnel_url": captured_tunnel,
                "llm_tunnel_url": llm_url_final,
                "summary": done_summary,
                "action": done_action,
            }
        )

    def _finalize_deploy_failure(
        self, dctx: _BootDeployCtx, tail_lines: list[str], elapsed_total: int
    ) -> None:
        """分类部署失败原因、登记稀缺显卡、推荐备选显卡并广播错误事件。"""
        b = self._b
        diag = classify_deploy_failure(
            tail_lines, dctx.gpu, dctx.pre_assignments, dctx.profile_label, dctx.recreate
        )
        stockout_gpu = diag.get("stockout_gpu")
        if stockout_gpu:
            with self._store.lock:
                b._scarce_gpus[str(stockout_gpu).upper()] = {
                    "ts": int(self._ctx.now()),
                    "reason": str(diag.get("reason") or "算力紧缺"),
                    "kind": str(diag.get("kind") or "stockout"),
                }
        snap = b.fetch_live_status()
        suggest_gpu = (
            recommend_fallback_gpu(
                str(stockout_gpu),
                dctx.profile,
                snap.get("eligible_gpus"),
                snap.get("scarce_gpus") or {},
                profiles=dctx.profiles,
            )
            if stockout_gpu
            else None
        )
        summary = str(diag.get("message") or "⚠️ 开机部署失败")
        if stockout_gpu and suggest_gpu and suggest_gpu not in summary:
            summary = f"{summary}，建议改选 {suggest_gpu}"

        with self._store.lock:
            b._deploy_state.update(
                {
                    "active": False,
                    "stage_title": "部署失败",
                    "sub_detail": str(diag.get("reason") or summary)[:42],
                    "error_msg": summary,
                    "stockout_gpu": stockout_gpu,
                    "suggest_gpu": suggest_gpu,
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
                "suggest_gpu": suggest_gpu,
            }
        )

    # ------------------------------------------------------------------
    # 一键开机部署与号池调度、并发流式执行
    # ------------------------------------------------------------------

    def _try_route_split_cluster(
        self,
        cfg: dict[str, Any],
        parsed: dict[str, Any],
        cluster_mode: str,
        acct_count: int,
        active_aid: str,
        cluster_split_pref: bool,
        pool_fresh: bool,
    ) -> dict[str, Any] | None:
        """当满足双栖且多号异构拆分条件时，规划并路由至 `start_split_cluster`。"""
        b = self._b
        is_slot_reconf = bool(cfg.get("_cluster_slot_reconfigure"))
        want_split = (
            parsed["is_dual"]
            and not parsed["cohost_only_llm"]
            and not is_slot_reconf
            and (cluster_mode == "split" or (cluster_mode == "auto" and cluster_split_pref))
            and acct_count >= 2
        )
        if not want_split:
            return None

        ps = _get_pool_sched()
        nodes = b._cluster_svc.build_pool_nodes()
        healthy_cnt = sum(1 for n in nodes if n.auth_ok and n.balance is not None and float(n.balance) >= 12.0)
        plan = (
            ps.plan_split_cluster(
                nodes,
                profiles=parsed["profiles"],
                comfy_gpu=cfg.get("comfy_gpu"),
                llm_gpu=cfg.get("llm_gpu"),
                active_account_id=active_aid,
            )
            if healthy_cnt >= 2
            else None
        )
        if healthy_cnt >= 2 and plan is not None and plan.ok:
            return b._cluster_svc.start_split_cluster(plan, cfg, parsed, defer_pool_refresh=(not pool_fresh))

        if cluster_mode != "split":
            return None

        if not pool_fresh:
            b._cluster_svc.ensure_pool_fresh_if_needed()
            nodes = b._cluster_svc.build_pool_nodes()
            healthy_cnt = sum(1 for n in nodes if n.auth_ok and n.balance is not None and float(n.balance) >= 12.0)
            if healthy_cnt >= 2:
                plan = ps.plan_split_cluster(
                    nodes,
                    profiles=parsed["profiles"],
                    comfy_gpu=cfg.get("comfy_gpu"),
                    llm_gpu=cfg.get("llm_gpu"),
                    active_account_id=active_aid,
                )
                if plan.ok:
                    return b._cluster_svc.start_split_cluster(plan, cfg, parsed)
                return {"ok": False, "message": plan.reason or "当前号池暂不满足双号异构双开条件"}
            return {"ok": False, "message": "当前健康付费账号不足 2 个（需余额 >= 12 CU），无法双号异构启动"}

        if healthy_cnt < 2:
            return {"ok": False, "message": "当前健康付费账号不足 2 个（需余额 >= 12 CU），无法双号异构启动"}
        return {"ok": False, "message": (plan.reason if plan else "") or "当前号池暂不满足双号异构双开条件"}

    def _evaluate_pool_select_strategy(
        self,
        cfg: dict[str, Any],
        parsed: dict[str, Any],
        acct_count: int,
        auto_pool_pref: bool,
        active_aid: str,
        pool_fresh: bool,
    ) -> bool:
        """评估多号池模式下是否需要把选号/切号推迟到后台工作线程执行，并按需填入复用端点。"""
        is_slot_reconf = bool(cfg.get("_cluster_slot_reconfigure"))
        if not (
            acct_count >= 2
            and auto_pool_pref
            and not parsed["cohost_only_llm"]
            and not is_slot_reconf
            and not str(cfg.get("_target_account_id") or "").strip()
        ):
            return False
        if not pool_fresh:
            return True
        ps = _get_pool_sched()
        nodes = self._b._cluster_svc.build_pool_nodes()
        decision = ps.select_best_account(
            nodes,
            profiles=parsed["profiles"],
            preferred_gpu=parsed["gpu"],
            active_account_id=active_aid,
        )
        if decision.ok and decision.account_id:
            if decision.account_id != active_aid:
                return True
            if decision.reuse_endpoint and not parsed["recreate"]:
                cfg["_reuse_endpoint"] = decision.reuse_endpoint
        return False

    def _abort_deploy_if_cancelled(self, dctx: _BootDeployCtx, cancel_tok: Any) -> bool:
        """检查取消标志；若已取消则落盘取消态、刷新快照并广播取消错误事件。"""
        b = self._b
        if not (b._deploy_cancelled or cancel_tok.is_cancelled):
            return False
        now_s = self._ctx.now()
        elapsed_now = max(0, int(now_s) - int(dctx.initial_deploy.get("started_ts") or now_s))
        with self._store.lock:
            b._deploy_state.update(
                {
                    "active": False,
                    "pct": 0,
                    "stage_title": "已中止部署",
                    "sub_detail": "用户主动取消开机流程",
                    "error_msg": "已中止开机部署",
                    "elapsed_sec": elapsed_now,
                }
            )
        snap_c = b.fetch_live_status()
        self._bus.emit(
            {
                "type": "deploy_error",
                "cancelled": True,
                "deploy": b.get_deploy_state(),
                "snapshot": snap_c,
                "summary": "⏹️ 已中止开机部署；若实例已分配可点一键停卡释放",
            }
        )
        return True

    def _run_deferred_pool_select_in_worker(
        self,
        dctx: _BootDeployCtx,
        cancel_tok: Any,
        cur_cmd: list[str],
        cur_acct_id: str,
    ) -> tuple[bool, list[str], str]:
        """在后台线程中非阻塞刷新号池、按需切换最优开机账号并重建启动命令。"""
        b = self._b
        if self._abort_deploy_if_cancelled(dctx, cancel_tok):
            return False, cur_cmd, cur_acct_id
        if b._cluster_svc.ensure_pool_fresh_if_needed():
            b._rebuild_and_emit_snapshot()
        if self._abort_deploy_if_cancelled(dctx, cancel_tok):
            return False, cur_cmd, cur_acct_id

        with self._store.lock:
            if b._account_epoch != dctx.start_epoch:
                b._deploy_state = default_deploy_state()
                return False, cur_cmd, cur_acct_id

        ps = _get_pool_sched()
        decision = ps.select_best_account(
            b._cluster_svc.build_pool_nodes(),
            profiles=dctx.profiles,
            preferred_gpu=dctx.gpu,
            active_account_id=cur_acct_id,
        )
        need_rebuild_cmd = False
        if decision.ok and decision.account_id:
            if decision.account_id != cur_acct_id:
                sw_label = decision.account_label or decision.account_id
                with self._store.lock:
                    cur_pct = int(b._deploy_state.get("pct") or 0)
                    b._deploy_state.update(
                        {
                            "pct": max(cur_pct, 3),
                            "stage_title": f"切换最优账号 ({sw_label})",
                            "sub_detail": "正在切换目标账号凭据并准备开机...",
                        }
                    )
                self._bus.emit({"type": "deploy_progress", "deploy": b.get_deploy_state()})
                sw_ok, new_epoch, sw_err = b._cluster_svc.switch_account_during_deploy(decision.account_id)
                if not sw_ok:
                    now_s = self._ctx.now()
                    elapsed_now = max(0, int(now_s) - int(dctx.initial_deploy.get("started_ts") or now_s))
                    self._finalize_deploy_failure(dctx, [f"切换最优账号失败: {sw_err}"], elapsed_now)
                    return False, cur_cmd, cur_acct_id
                dctx.start_epoch = new_epoch
                cur_acct_id = decision.account_id
                dctx.account_id = cur_acct_id
                with self._store.lock:
                    dctx.pre_assignments = int(self._store.state.get("last_assignments_count") or 0)
                b._rebuild_and_emit_snapshot()
                need_rebuild_cmd = True
            target_ep_next = "" if dctx.recreate else str(decision.reuse_endpoint or "")
            if need_rebuild_cmd or target_ep_next:
                cur_cmd = self._build_deploy_cmd_for_ctx(
                    dctx, endpoint=target_ep_next, account_id=cur_acct_id
                )

        if self._abort_deploy_if_cancelled(dctx, cancel_tok):
            return False, cur_cmd, cur_acct_id
        with self._store.lock:
            cur_pct = int(b._deploy_state.get("pct") or 0)
            b._deploy_state.update(
                {
                    "pct": max(cur_pct, 3),
                    "stage_title": f"准备开机 ({dctx.gpu})",
                    "sub_detail": "号池核验完成，正在启动本地无窗口部署进程...",
                }
            )
        self._bus.emit({"type": "deploy_progress", "deploy": b.get_deploy_state()})
        return True, cur_cmd, cur_acct_id

    def _try_failover_other_account(
        self,
        dctx: _BootDeployCtx,
        tail_lines: list[str],
        cur_acct_id: str,
        failover_retries: int,
    ) -> tuple[bool, list[str], str]:
        """多号智能调度下，遇缺卡或配额受限时自动切换备选账号重试一次。"""
        b = self._b
        if failover_retries >= 1 or dctx.cohost_only_llm or dctx.cluster_slot_reconfigure or not cur_acct_id:
            return False, [], cur_acct_id
        with self._store.lock:
            p_now = self._store.state.get("prefs") or {}
            n_accts = len(
                [p for p in (self._store.state.get("account_profiles") or []) if isinstance(p, dict) and p.get("id")]
            )
            can_failover = n_accts >= 2 and bool(p_now.get("auto_pool_schedule", True))
        if not can_failover:
            return False, [], cur_acct_id

        diag = classify_deploy_failure(
            tail_lines, dctx.gpu, dctx.pre_assignments, dctx.profile_label, dctx.recreate
        )
        if diag.get("kind") not in ("stockout", "entitlement", "limit_or_stockout"):
            return False, [], cur_acct_id

        failed_gpu = str(diag.get("stockout_gpu") or dctx.gpu).upper()
        b._cluster_svc.mark_account_gpu_scarce(
            cur_acct_id,
            failed_gpu,
            str(diag.get("reason") or "算力紧缺"),
            kind=str(diag.get("kind") or "stockout"),
        )
        ps = _get_pool_sched()
        fb = ps.select_best_account(
            b._cluster_svc.build_pool_nodes(),
            profiles=dctx.profiles,
            preferred_gpu=dctx.gpu,
            active_account_id=cur_acct_id,
            exclude_accounts={cur_acct_id},
        )
        if not (fb.ok and fb.account_id and fb.account_id != cur_acct_id):
            return False, [], cur_acct_id

        fb_name = fb.account_label or fb.account_id
        with self._store.lock:
            b._deploy_state.update(
                {
                    "step_idx": 0,
                    "pct": 5,
                    "stage_title": f"自动换号重试 ({fb_name})",
                    "sub_detail": f"{failed_gpu} 紧缺/受限，正自动切备号「{fb_name}」重试...",
                }
            )
        self._bus.emit({"type": "deploy_progress", "deploy": b.get_deploy_state()})
        sw_ok, new_epoch, _sw_err = b._cluster_svc.switch_account_during_deploy(fb.account_id)
        if not sw_ok:
            return False, [], cur_acct_id

        dctx.start_epoch = new_epoch
        new_acct_id = fb.account_id
        dctx.account_id = new_acct_id
        if fb.gpu:
            dctx.gpu = fb.gpu
        new_cmd = self._build_deploy_cmd_for_ctx(
            dctx, endpoint=fb.reuse_endpoint or "", account_id=new_acct_id
        )
        with self._store.lock:
            b._deploy_dl_done = 0
            b._deploy_dl_name = ""
            b._deploy_meta_warmed = False
            b._deploy_captured_llm_tunnel = None
            b._deploy_phase_idx = 1
        return True, new_cmd, new_acct_id

    def _run_deploy_worker(self, dctx: _BootDeployCtx, cancel_tok: Any) -> None:
        """后台流式执行 `colab_ctl up` 子进程、维持心跳广播并在结束时收尾或换号重试。"""
        b = self._b
        b._deploy_cancel_token = cancel_tok
        hb_stop = self._ctx.threading_mod.Event()
        tail_lines: list[str] = []
        captured_tunnel: str | None = None

        def _heartbeat() -> None:
            while not hb_stop.wait(timeout=1.0):
                if not b.is_deploying():
                    break
                self._bus.emit({"type": "deploy_progress", "deploy": b.get_deploy_state()})

        self._ctx.threading_mod.Thread(target=_heartbeat, daemon=True).start()
        try:
            ctl_env = self._ctx.build_ctl_env()
            ctl_env["PYTHONUNBUFFERED"] = "1"
            ctl_env["COLAB_CLI_REQUEST_TIMEOUT"] = "75"
            ctl_env["COLAB_CLI_ASSIGN_RETRIES"] = "2"
            ctl_env["COLAB_CLI_RETRY_BACKOFF"] = "5"
            last_emit_m = 0.0
            failover_retries = 0
            cur_cmd = list(dctx.cmd)
            cur_acct_id = dctx.account_id or self._store.get_active_account_id()

            if dctx.defer_pool_select:
                keep_going, cur_cmd, cur_acct_id = self._run_deferred_pool_select_in_worker(
                    dctx, cancel_tok, cur_cmd, cur_acct_id
                )
                if not keep_going:
                    return

            while True:
                def _on_deploy_line(sline: str) -> None:
                    nonlocal captured_tunnel, last_emit_m
                    tail_lines.append(sline)
                    if len(tail_lines) > 120:
                        del tail_lines[:-120]
                    prev_step = int(b._deploy_state.get("step_idx") or 0)
                    tun = b._consume_deploy_line(sline)
                    cur_step = int(b._deploy_state.get("step_idx") or 0)
                    if tun:
                        captured_tunnel = tun
                    now_m = self._ctx.monotonic()
                    if tun or cur_step != prev_step or (now_m - last_emit_m >= 0.12):
                        last_emit_m = now_m
                        self._bus.emit({"type": "deploy_progress", "deploy": b.get_deploy_state()})

                dctx.cli_scope_acct = _extract_cmd_account(cur_cmd)
                stream_res = b._runner.run_streaming(
                    cur_cmd,
                    on_line=_on_deploy_line,
                    extra_env=ctl_env,
                    cancel_token=cancel_tok,
                    popen_fn=self._ctx.subprocess_mod.Popen,
                    on_proc_start=lambda p: setattr(b, "_deploy_proc", p),
                    task_name="deploy_up",
                )
                rc = 0 if stream_res.ok else (stream_res.returncode or 1)
                if not stream_res.ok and stream_res.error and not tail_lines:
                    tail_lines.append(stream_res.error)

                cancelled = bool(b._deploy_cancelled or stream_res.cancelled or cancel_tok.is_cancelled)
                now_s = self._ctx.now()
                elapsed_total = max(0, int(now_s) - int(dctx.initial_deploy.get("started_ts") or now_s))
                elapsed_cn = format_elapsed_cn(elapsed_total)

                with self._store.lock:
                    if b._account_epoch != dctx.start_epoch:
                        b._deploy_state = default_deploy_state()
                        return

                if cancelled and self._abort_deploy_if_cancelled(dctx, cancel_tok):
                    return
                if rc == 0:
                    self._finalize_deploy_success(dctx, captured_tunnel, elapsed_total, elapsed_cn)
                    return

                switched, new_cmd, new_acct_id = self._try_failover_other_account(
                    dctx, tail_lines, cur_acct_id, failover_retries
                )
                if switched:
                    cur_cmd = new_cmd
                    cur_acct_id = new_acct_id
                    tail_lines.clear()
                    captured_tunnel = None
                    failover_retries += 1
                    continue

                self._finalize_deploy_failure(dctx, tail_lines, elapsed_total)
                return
        finally:
            hb_stop.set()
            b._deploy_proc = None
            b._deploy_cancel_token = None
            b._deploy_lock.release()

    def start_instance(self, config: dict[str, Any] | None = None) -> dict[str, Any]:
        """按用户所选工作流集合（支持多选双栖与双号异构拆分）与显卡一键无窗口部署并直播进度。"""
        b = self._b
        b.touch_activity()
        cfg = dict(config) if isinstance(config, dict) else {}
        cluster_mode = str(cfg.get("cluster_mode") or "auto").strip().lower()
        if cluster_mode not in ("split", "cohost", "auto"):
            cluster_mode = "auto"

        parsed, err_resp = self._validate_and_parse_boot_request(cfg, skip_preflight=True)
        if err_resp is not None or parsed is None:
            return err_resp or {"ok": False, "message": "开机参数解析失败"}

        with self._store.lock:
            prefs_now = self._store.get_prefs()
            acct_profiles = [
                p for p in (self._store.state.get("account_profiles") or [])
                if isinstance(p, dict) and p.get("id")
            ]
            acct_count = len(acct_profiles)
            active_aid = str(prefs_now.get("active_account_id") or "").strip()
            cluster_split_pref = bool(prefs_now.get("cluster_split_mode", True))
            auto_pool_pref = bool(prefs_now.get("auto_pool_schedule", True))

        pool_fresh = b._cluster_svc.is_pool_fresh() if acct_count >= 2 else True
        split_resp = self._try_route_split_cluster(
            cfg, parsed, cluster_mode, acct_count, active_aid, cluster_split_pref, pool_fresh
        )
        if split_resp is not None:
            return split_resp

        defer_pool_select = self._evaluate_pool_select_strategy(
            cfg, parsed, acct_count, auto_pool_pref, active_aid, pool_fresh
        )

        pre = b.run_preflight(deep=False, profiles=parsed["profiles"], gpu=parsed["gpu"])
        if not pre.get("ok"):
            return {
                "ok": False,
                "preflight_failed": True,
                "preflight": pre,
                "message": str(pre.get("blocking_message") or "启动前环境检查未通过"),
            }

        if not b._deploy_lock.acquire(blocking=False):
            return {"ok": False, "message": "已有开机部署任务在运行中"}

        try:
            dctx = self._prepare_deploy_ctx(
                cfg,
                parsed,
                defer_pool_select=defer_pool_select,
                pool_acct_count=acct_count,
            )
        except Exception:
            b._deploy_lock.release()
            raise

        b._wake_monitor.set()
        b._tasks.submit("deploy", lambda cancel_tok: self._run_deploy_worker(dctx, cancel_tok), exclusive=False)
        msg = (
            f"正在热重载 {dctx.gpu} · {dctx.profile_label}..."
            if dctx.cohost_only_llm
            else f"开始部署 {dctx.gpu} · {dctx.profile_label}..."
        )
        return {"ok": True, "started": True, "deploy": dctx.initial_deploy, "message": msg}

    # ------------------------------------------------------------------
    # 运行态 Qwen3.8 热重载调参与部署中止
    # ------------------------------------------------------------------

    def _resolve_running_llm_target(
        self, cluster_llm_slot: dict[str, Any] | None
    ) -> tuple[dict[str, Any], str, dict[str, Any], dict[str, Any]]:
        """采集热重载所需的偏好、运行卡型、远端元数据与 `_comfy_state` 快照（严守锁偏序）。"""
        b = self._b
        with self._store.lock:
            prefs = self._store.get_prefs()
            sessions = self._store.get_last_sessions()
            if cluster_llm_slot:
                target_ep = str(cluster_llm_slot.get("endpoint") or "")
                run_gpu = str(cluster_llm_slot.get("gpu") or "L4").strip().upper()
            else:
                target_ep = next(
                    (str(s.get("endpoint") or "") for s in sessions if s.get("name") == "comfy" and s.get("endpoint")),
                    str((sessions[0].get("endpoint") or "") if sessions else ""),
                )
                target_sess = next((s for s in sessions if s.get("name") == "comfy"), sessions[0] if sessions else None)
                run_gpu = str((target_sess or {}).get("accelerator") or prefs.get("boot_gpu") or "L4").strip().upper()
            meta_ep, _ = self._secrets.get_ep_meta_and_key(target_ep) if target_ep else ({}, "")
            if run_gpu not in SUPPORTED_BOOT_GPUS:
                run_gpu = "L4"

        with b._comfy_lock:
            cstate = dict(b._comfy_state)
        return prefs, run_gpu, meta_ep, cstate

    @staticmethod
    def _detect_live_llm_profile(
        meta_ep: dict[str, Any],
        cstate: dict[str, Any],
        act_profs: list[str],
        pref_profs: list[str],
    ) -> str | None:
        """从元数据、活跃配置与模型名称中识别当前在线的 LLM 档位。"""
        cand = _first_non_empty(
            meta_ep.get("llm_profile"),
            next((p for p in act_profs if p in LLM_PROFILES), ""),
            next((p for p in pref_profs if p in LLM_PROFILES), ""),
        )
        if cand in LLM_PROFILES:
            return cand
        if "27b" in str(cstate.get("model_name") or "").lower():
            return "qwen_27b"
        if cstate.get("service_kind") in ("llm", "dual") or cstate.get("llm_online"):
            return "qwen_llm"
        return None

    @classmethod
    def _infer_live_llm_and_dual_profiles(
        cls, meta_ep: dict[str, Any], cstate: dict[str, Any], prefs: dict[str, Any]
    ) -> tuple[str | None, bool, list[str]]:
        """推断当前在线实例的 LLM 配置档位、是否同机双栖以及需保留的 Comfy 工作流列表。"""
        act_profs = normalize_profiles(cstate.get("active_profiles"), prefs.get("boot_profile") or "anima")
        pref_profs = normalize_profiles(prefs.get("boot_profiles"), prefs.get("boot_profile") or "anima")
        llm_prof = cls._detect_live_llm_profile(meta_ep, cstate, act_profs, pref_profs)
        if not llm_prof:
            return None, False, []

        sk = str(cstate.get("service_kind") or meta_ep.get("kind") or "")
        meta_cp = meta_ep.get("comfy_profiles") if isinstance(meta_ep.get("comfy_profiles"), list) else []
        has_act_comfy = any(p in COMFY_PROFILES for p in act_profs)
        is_dual_live = bool(
            sk == "dual" or cstate.get("comfy_online") or meta_ep.get("kind") == "dual" or meta_cp or has_act_comfy
        )

        preserved_comfy: list[str] = []
        if is_dual_live:
            for cp in [*meta_cp, *act_profs, *pref_profs]:
                if cp in COMFY_PROFILES and cp not in preserved_comfy:
                    preserved_comfy.append(cp)
            if not preserved_comfy:
                preserved_comfy = ["anima"]
        return llm_prof, is_dual_live, [*preserved_comfy, llm_prof]

    @staticmethod
    def _parse_reconfigure_llm_params(
        cfg: dict[str, Any], prefs: dict[str, Any], cstate: dict[str, Any], is_27b: bool
    ) -> tuple[str, int, bool]:
        """解析并裁剪热重载所需的量化档位、上下文长度与视觉识图开关。"""
        quant_key = "boot_llm27_quant" if is_27b else "boot_llm_quant"
        ctx_key = "boot_llm27_ctx" if is_27b else "boot_llm_ctx"
        mm_key = "boot_llm27_mmproj" if is_27b else "boot_llm_mmproj"

        default_quant = str(prefs.get(quant_key) or cstate.get("quant") or "UD-Q4_K_XL").strip()
        raw_quant = str(cfg.get("quant") or cfg.get("llm27_quant" if is_27b else "llm_quant") or default_quant).strip()
        new_quant = raw_quant if raw_quant in SUPPORTED_LLM_QUANTS else "UD-Q4_K_XL"

        default_ctx = int(prefs.get(ctx_key) or cstate.get("ctx_size") or (65536 if is_27b else 32768))
        raw_ctx_val = cfg.get("ctx")
        if raw_ctx_val is None:
            raw_ctx_val = cfg.get("llm27_ctx" if is_27b else "llm_ctx")
        try:
            new_ctx = int(raw_ctx_val if raw_ctx_val is not None else default_ctx)
        except (TypeError, ValueError):
            new_ctx = default_ctx
        new_ctx = max(4096, min(262144, new_ctx))

        default_mm = bool(prefs.get(mm_key, True))
        raw_mm = cfg.get("mmproj")
        if raw_mm is None:
            raw_mm = cfg.get("llm27_mmproj" if is_27b else "llm_mmproj")
        new_mmproj = default_mm if raw_mm is None else bool(raw_mm)
        return new_quant, new_ctx, new_mmproj

    def reconfigure_running_llm(self, config: dict[str, Any] | None = None) -> dict[str, Any]:
        """在不释放实例且不断开同机 ComfyUI 的前提下，极速热重载运行中 Qwen3.8 的上下文/量化/识图参数。"""
        b = self._b
        b.touch_activity()
        if not self._store.has_running_session():
            return {"ok": False, "message": "当前无在线实例，请在「选卡开机」直接部署"}
        if b.is_deploying() or b._deploy_lock.locked():
            return {"ok": False, "message": "已有部署或热重载任务进行中，请稍候"}

        cfg = config if isinstance(config, dict) else {}
        has_slots = b._cluster_svc.has_active_slots()
        cluster_llm_slot = b._cluster_svc.get_slot("llm") if has_slots else None
        cluster_comfy_slot = b._cluster_svc.get_slot("comfy") if has_slots else None

        prefs, run_gpu, meta_ep, cstate = self._resolve_running_llm_target(cluster_llm_slot)
        llm_prof, is_dual_live, preserved_profiles = self._infer_live_llm_and_dual_profiles(meta_ep, cstate, prefs)
        if not llm_prof:
            return {"ok": False, "message": "当前实例未运行 Qwen3.8 服务，请先到「选卡开机」部署"}

        is_27b = llm_prof == "qwen_27b"
        new_quant, new_ctx, new_mmproj = self._parse_reconfigure_llm_params(cfg, prefs, cstate, is_27b)
        start_cfg: dict[str, Any] = {
            "profile": llm_prof,
            "profiles": [llm_prof],
            "gpu": run_gpu,
            "recreate": False,
            "_cohost_only_llm": False if cluster_llm_slot else is_dual_live,
            "_preserved_profiles": preserved_profiles,
            ("llm27_quant" if is_27b else "llm_quant"): new_quant,
            ("llm27_ctx" if is_27b else "llm_ctx"): new_ctx,
            ("llm27_mmproj" if is_27b else "llm_mmproj"): new_mmproj,
        }
        if cluster_llm_slot:
            start_cfg["_cluster_slot_reconfigure"] = "llm"
            start_cfg["_target_account_id"] = str(cluster_llm_slot.get("account_id") or "")
            start_cfg["_target_endpoint"] = str(cluster_llm_slot.get("endpoint") or "")
            if cluster_comfy_slot:
                c_profs = [p for p in (cluster_comfy_slot.get("profiles") or []) if p in COMFY_PROFILES]
                start_cfg["_preserved_profiles"] = [*(c_profs or ["minimax"]), llm_prof]
        return b.start_instance(start_cfg)

    def cancel_deploy(self) -> dict[str, Any]:
        """主动中止正在运行的一键开机部署子进程。"""
        b = self._b
        proc = b._deploy_proc
        tok = b._deploy_cancel_token
        if not b.is_deploying() and proc is None and not b._deploy_lock.locked():
            return {"ok": False, "message": "当前没有进行中的部署任务"}
        b._deploy_cancelled = True
        if tok is not None:
            tok.cancel()
        kill_process_tree(proc)
        b._cluster_svc.cancel_split_deploys()
        return {"ok": True, "message": "正在中止开机部署..."}
