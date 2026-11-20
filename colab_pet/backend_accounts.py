"""Colab 桌宠多账号保险箱、OAuth2 绑定、rclone 云盘与环境诊断子服务。"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from colab_core.accounts import (
    ToolPaths,
    extract_oauth_code,
    has_gdrive_save_section,
    query_gdrive_quota,
    sanitize_account_label,
    validate_and_import_rclone_conf,
)
from colab_core.health import export_diagnostic_bundle, run_preflight_checks
from colab_core.runtime import (
    CREATE_NO_WINDOW,
    DEFAULT_PREFS,
    clean_subprocess_env,
    kill_process_tree,
    restore_ledger_for_account,
    snapshot_current_ledger,
    suppress_logged,
    upsert_account_profile,
)
from colab_security import redact_secrets, register_ephemeral_secret

if TYPE_CHECKING:
    from colab_backend import ColabPetBackend


class AccountVaultService:
    """负责 Google Ultra 多账号 OAuth/切号、rclone 云盘绑定、工具路径与诊断包导出。"""

    def __init__(self, backend: ColabPetBackend) -> None:
        self._b = backend
        self._store = backend._store
        self._secrets = backend._secrets
        self._bus = backend._bus
        self._ctx = backend._ctx

    @staticmethod
    def _sanitize_account_id(acct_id: Any) -> str:
        """将外部传入的账号 ID 清洗为安全字符集并限制长度。"""
        return re.sub(r"[^a-zA-Z0-9_-]", "", str(acct_id or ""))[:24]

    @staticmethod
    def _extract_proc_error_tail(proc: Any) -> str:
        """提取子进程失败输出的最后一行脱敏错误摘要。"""
        err = redact_secrets((proc.stderr or proc.stdout or "").strip()).splitlines()
        return err[-1] if err else str(proc.returncode)

    def _collect_tool_paths(self) -> ToolPaths:
        """通过依赖上下文收集当前生效的本地工具路径集合。"""
        return ToolPaths(
            colab_exe=self._ctx.get_colab_exe(),
            colab_py=self._ctx.get_colab_py(),
            rclone_exe=self._ctx.get_rclone_exe(),
            rclone_conf=self._ctx.get_rclone_conf(),
        )

    def _has_busy_background_ops(self) -> bool:
        """判断当前是否存在部署、上传或自动停卡等不可切号的后台动作。"""
        b = self._b
        return bool(
            b.is_deploying()
            or b._deploy_lock.locked()
            or b._upload_lock.locked()
            or b._idle_stop_lock.locked()
        )

    def _commit_account_switch_locked(
        self,
        prev_id: str,
        target_id: str,
        *,
        clear_oauth: bool = False,
        email_masked: str = "",
        preferred_label: str = "",
        reset_idle_retry: bool = False,
    ) -> dict[str, Any] | None:
        """在持有主状态锁时切换活跃账号、快照/恢复独立账本、清理内存缓存并原子落盘。"""
        b = self._b
        state = self._store.state
        b._account_epoch += 1
        if clear_oauth:
            b._pending_oauth_verifier = ""
            b._pending_oauth_ts = 0.0
        b._auth_missing = False
        b._auth_expired = False
        if prev_id and prev_id != target_id:
            state.setdefault("account_ledgers", {})[prev_id] = snapshot_current_ledger(state)
        restore_ledger_for_account(state, target_id)
        prefs = state.setdefault("prefs", dict(DEFAULT_PREFS))
        prefs["active_account_id"] = target_id
        card = (
            upsert_account_profile(state, target_id, email_masked, preferred_label=preferred_label)
            if email_masked
            else None
        )
        b._clear_ephemeral_caches_locked()
        if reset_idle_retry:
            b._idle_archive_retry_after_m = 0.0
        self._store.save()
        return card

    def start_colab_oauth(self) -> dict[str, Any]:
        """生成带账号选择器的 Google OAuth2 授权链接并在浏览器打开，PKCE 校验码仅存内存。"""
        b = self._b
        colab_py = self._ctx.get_colab_py()
        if not colab_py.is_file():
            return {
                "ok": False,
                "message": "未检测到 colab-cli 运行环境，请先在「云盘与环境」安装或定位路径",
            }
        try:
            proc = self._ctx.run_hidden(
                [str(colab_py), str(self._ctx.cli_bridge_script), "oauth-start"],
                timeout=20,
            )
            if proc.returncode != 0:
                return {"ok": False, "message": f"生成授权链接失败: {self._extract_proc_error_tail(proc)}"}
            m = re.search(
                r"<<<COLAB_OAUTH_START>>>(.*?)<<<END_COLAB_OAUTH_START>>>",
                proc.stdout or "",
                re.S,
            )
            if not m:
                return {"ok": False, "message": "授权脚本返回格式异常"}
            data = json.loads(m.group(1))
            auth_url = str(data.get("auth_url") or "").strip()
            verifier = str(data.get("code_verifier") or "").strip()
            if not auth_url.startswith("https://"):
                return {"ok": False, "message": "未获取到合法 Google OAuth 链接"}
            register_ephemeral_secret(verifier)
            with self._store.lock:
                b._pending_oauth_verifier = verifier
                b._pending_oauth_ts = self._ctx.monotonic()
            self._ctx.open_browser(auth_url)
            snap = b._rebuild_and_emit_snapshot()
            return {
                "ok": True,
                "awaiting_code": True,
                "snapshot": snap,
                "message": "已打开 Google 账号选择与授权页，复制授权码后点击一键粘贴绑定",
            }
        except Exception as exc:
            return {"ok": False, "message": f"拉起 Google 授权失败: {redact_secrets(str(exc))}"}

    def cancel_colab_oauth(self) -> dict[str, Any]:
        """取消进行中的 Google OAuth2 绑定流程并清理内存临时校验码。"""
        b = self._b
        with self._store.lock:
            b._pending_oauth_verifier = ""
            b._pending_oauth_ts = 0.0
        snap = b._rebuild_and_emit_snapshot()
        return {"ok": True, "snapshot": snap, "message": "已取消本次账号绑定"}

    def _exchange_oauth_code(
        self,
        colab_py: Path,
        code: str,
        verifier: str,
        prev_acct_id: str,
    ) -> tuple[str, str, str | None]:
        """调用桥接脚本用授权码换取令牌并解析 `(acct_id, email_masked, error_msg)`。"""
        try:
            payload = json.dumps(
                {"code": code, "code_verifier": verifier, "prev_acct_id": prev_acct_id},
                ensure_ascii=False,
            )
            proc = self._ctx.run_hidden(
                [str(colab_py), str(self._ctx.cli_bridge_script), "oauth-complete"],
                timeout=35,
                input_text=payload,
            )
            if proc.returncode != 0:
                return "", "", f"授权换取令牌失败: {self._extract_proc_error_tail(proc)}"
            m = re.search(
                r"<<<COLAB_OAUTH_DONE>>>(.*?)<<<END_COLAB_OAUTH_DONE>>>",
                proc.stdout or "",
                re.S,
            )
            if not m:
                return "", "", "授权写入结果解析失败"
            info = json.loads(m.group(1))
            acct_id = self._sanitize_account_id(info.get("acct_id"))
            if not acct_id:
                return "", "", "未能识别账号唯一标识"
            return acct_id, str(info.get("email_masked") or "***"), None
        except Exception as exc:
            return "", "", f"绑定授权码异常: {redact_secrets(str(exc))}"

    def complete_colab_oauth(self, raw_code: str, label: str = "") -> dict[str, Any]:
        """用授权码换取 Token 并安全写入系统凭据库，自动建档并切换至新绑定账号。"""
        b = self._b
        if self._has_busy_background_ops():
            return {
                "ok": False,
                "busy": True,
                "message": "当前仍有后台任务（部署/上传/自动停卡）正在执行，请等待结束后再绑定新账号",
            }
        colab_py = self._ctx.get_colab_py()
        if not colab_py.is_file():
            return {"ok": False, "message": "未检测到 colab-cli 运行环境"}

        register_ephemeral_secret(raw_code)
        code = extract_oauth_code(raw_code)
        register_ephemeral_secret(code)
        if len(code) < 8:
            return {"ok": False, "message": "请粘贴有效的 Google 授权码"}

        with self._store.lock:
            verifier = b._pending_oauth_verifier
            pending_ts = b._pending_oauth_ts
            prev_acct_id = self._store.get_active_account_id()

        if not verifier or (pending_ts > 0 and (self._ctx.monotonic() - pending_ts) > 1200.0):
            return {"ok": False, "message": "授权会话已超时或未启动，请重新点「新增 Google 账号」"}
        register_ephemeral_secret(verifier)

        acct_id, email_masked, err_msg = self._exchange_oauth_code(colab_py, code, verifier, prev_acct_id)
        if err_msg:
            return {"ok": False, "message": err_msg}

        with self._store.lock:
            card = self._commit_account_switch_locked(
                prev_acct_id,
                acct_id,
                clear_oauth=True,
                email_masked=email_masked,
                preferred_label=label,
            )
        b._stop_ws()
        b._reset_comfy_offline()

        snap = b.fetch_live_status()
        with self._store.lock:
            profiles = self._store.state.get("account_profiles") or []
            refreshed_card = next(
                (dict(p) for p in profiles if isinstance(p, dict) and p.get("id") == acct_id),
                card or {"id": acct_id, "label": label or acct_id},
            )
        bal = snap.get("balance")
        bal_txt = f"余额 {bal:.2f} CU" if isinstance(bal, (int, float)) else "已同步"
        return {
            "ok": True,
            "account": refreshed_card,
            "snapshot": snap,
            "message": f"✅ 已绑定 {refreshed_card['label']} ({bal_txt})",
        }

    def _run_account_switch_bridge(self, colab_py: Path, target_id: str, prev_id: str) -> dict[str, Any] | None:
        """在 `_action_lock` 保护下调用桥接脚本切换凭据；成功返回 None，失败返回错误字典。"""
        b = self._b
        if not b._action_lock.acquire(blocking=False):
            return {
                "ok": False,
                "busy": True,
                "message": "当前账号仍有后台运维操作正在执行，请稍等片刻再切换",
            }
        try:
            if self._has_busy_background_ops():
                return {
                    "ok": False,
                    "busy": True,
                    "message": "当前账号仍有后台任务正在执行，请等待完成后再切换账号",
                }
            payload = json.dumps(
                {"op": "switch", "target_acct_id": target_id, "prev_acct_id": prev_id},
                ensure_ascii=False,
            )
            proc = self._ctx.run_hidden(
                [str(colab_py), str(self._ctx.cli_bridge_script), "account-op"],
                timeout=25,
                input_text=payload,
            )
            if proc.returncode != 0:
                return {"ok": False, "message": f"切换凭据失败: {self._extract_proc_error_tail(proc)}"}
            return None
        finally:
            b._action_lock.release()

    def switch_colab_account(self, acct_id: str) -> dict[str, Any]:
        """秒切到凭据保险箱中的另一组 Google Ultra 账号，完整隔离账本与 sessions.json，并在有后台任务时安全拦截。"""
        b = self._b
        if self._has_busy_background_ops():
            return {
                "ok": False,
                "busy": True,
                "message": "当前账号仍有后台任务（部署/上传/自动停卡）正在执行，请等待完成或先中止后再切换账号",
            }
        colab_py = self._ctx.get_colab_py()
        if not colab_py.is_file():
            return {"ok": False, "message": "未检测到 colab-cli 运行环境"}

        target_id = self._sanitize_account_id(acct_id)
        if not target_id:
            return {"ok": False, "message": "无效的账号 ID"}

        with self._store.lock:
            profiles = self._store.state.get("account_profiles") or []
            card = next((dict(p) for p in profiles if p.get("id") == target_id), None)
            if card is None:
                return {"ok": False, "message": "未在多账号列表中找到目标账号"}
            prev_id = self._store.get_active_account_id()

        if prev_id == target_id and not b._auth_missing:
            snap = b.fetch_live_status()
            bal = snap.get("balance")
            bal_txt = f"{bal:.2f} CU" if isinstance(bal, (int, float)) else "已同步"
            return {"ok": True, "snapshot": snap, "message": f"当前已是 {card['label']} ({bal_txt})"}

        err_res = self._run_account_switch_bridge(colab_py, target_id, prev_id)
        if err_res is not None:
            return err_res

        with self._store.lock:
            self._commit_account_switch_locked(prev_id, target_id, reset_idle_retry=True)
        b._stop_ws()
        b._reset_comfy_offline()

        snap = b.fetch_live_status()
        bal = snap.get("balance")
        bal_txt = f"{bal:.2f} CU" if isinstance(bal, (int, float)) else "同步中"
        return {
            "ok": True,
            "snapshot": snap,
            "message": f"⚡ 已秒切至 {card['label']} ({bal_txt})",
        }

    def rename_colab_account(self, acct_id: str, label: str) -> dict[str, Any]:
        """修改指定已绑定 Google 账号的显示备注（如 Ultra 主号 / 备用号）。"""
        b = self._b
        target_id = self._sanitize_account_id(acct_id)
        clean_label = sanitize_account_label(label, "")
        if not target_id or not clean_label:
            return {"ok": False, "message": "请输入非空账号备注名称"}
        with self._store.lock:
            profiles = self._store.state.get("account_profiles") or []
            found = next((p for p in profiles if isinstance(p, dict) and p.get("id") == target_id), None)
            if not found:
                return {"ok": False, "message": "未找到目标账号"}
            found["label"] = clean_label
            self._store.save()
        snap = b._rebuild_and_emit_snapshot()
        return {"ok": True, "snapshot": snap, "message": f"已重命名为「{clean_label}」"}

    def _remove_account_credentials(self, target_id: str, clear_active: bool) -> None:
        """尽力通过桥接脚本清理目标账号的系统凭据，忽略异常以确保本地可解绑。"""
        colab_py = self._ctx.get_colab_py()
        if not colab_py.is_file():
            return
        payload = json.dumps(
            {"op": "remove", "target_acct_id": target_id, "clear_active": clear_active},
            ensure_ascii=False,
        )
        with suppress_logged("accounts", throttled_key="remove_acct_creds", interval_sec=15.0):
            self._ctx.run_hidden(
                [str(colab_py), str(self._ctx.cli_bridge_script), "account-op"],
                timeout=20,
                input_text=payload,
            )

    def _apply_removed_account_state(
        self,
        target_id: str,
        remaining: list[dict[str, Any]],
        removing_active: bool,
    ) -> None:
        """在主状态锁保护下移除账号档案与独立账本，若已无余号则重置认证与内存缓存。"""
        b = self._b
        with self._store.lock:
            state = self._store.state
            state["account_profiles"] = remaining
            ledgers = state.get("account_ledgers")
            if isinstance(ledgers, dict):
                ledgers.pop(target_id, None)
            if removing_active and not remaining:
                b._account_epoch += 1
                state.setdefault("prefs", dict(DEFAULT_PREFS))["active_account_id"] = ""
                restore_ledger_for_account(state, "")
                b._auth_missing = True
                b._auth_expired = False
                b._clear_ephemeral_caches_locked()
            self._store.save()

    def remove_colab_account(self, acct_id: str) -> dict[str, Any]:
        """从保险箱与系统凭据库移除指定账号；若当前激活且仍有余号则自动切换过去。"""
        b = self._b
        if b.is_deploying() or b._deploy_lock.locked():
            return {"ok": False, "message": "正在执行开机部署，请先等待部署完成"}
        target_id = self._sanitize_account_id(acct_id)
        if not target_id:
            return {"ok": False, "message": "无效的账号 ID"}

        with self._store.lock:
            profiles = [dict(p) for p in (self._store.state.get("account_profiles") or []) if isinstance(p, dict)]
            removed_card = next((p for p in profiles if p.get("id") == target_id), None)
            if removed_card is None:
                return {"ok": False, "message": "目标账号不存在"}
            remaining = [p for p in profiles if p.get("id") != target_id]
            active_id = self._store.get_active_account_id()
            removing_active = active_id == target_id or not remaining

        self._remove_account_credentials(target_id, clear_active=removing_active and (len(remaining) == 0))
        self._apply_removed_account_state(target_id, remaining, removing_active)

        if removing_active and remaining:
            next_id = str(remaining[0]["id"])
            res = b.switch_colab_account(next_id)
            if res.get("ok"):
                res["message"] = f"已移除「{removed_card['label']}」，自动切至「{remaining[0]['label']}」"
            return res

        if removing_active and not remaining:
            b._stop_ws()
            b._reset_comfy_offline()
            snap = b._rebuild_and_emit_snapshot(stale=True, error_msg="尚未绑定 Google Colab 账号")
            return {"ok": True, "snapshot": snap, "message": f"已解绑「{removed_card['label']}」"}

        snap = b._rebuild_and_emit_snapshot()
        return {"ok": True, "snapshot": snap, "message": f"已移除「{removed_card['label']}」"}

    def query_gdrive_quota_locked(self) -> tuple[float | None, float | None]:
        """调用 rclone about 查询 gdrive_save 剩余与总容量（GiB），失败返回 (None, None)。"""
        free_gb, total_gb = query_gdrive_quota(
            self._ctx.get_rclone_exe(),
            self._ctx.get_rclone_conf(),
            run_fn=lambda cmd, timeout=15: self._ctx.run_hidden(cmd, timeout=timeout),
        )
        if free_gb is not None or total_gb is not None:
            with self._store.lock:
                self._store.state["onboarding_cache"] = {
                    "gdrive_free_gb": free_gb,
                    "gdrive_total_gb": total_gb,
                    "gdrive_checked_ts": int(self._ctx.now()),
                }
                self._store.save()
        return free_gb, total_gb

    def probe_environment_deep(self) -> dict[str, Any]:
        """执行一次完整的本地工具、Google Drive 剩余容量与 Colab 在线健康体检。"""
        b = self._b
        free_gb, _ = b._query_gdrive_quota_locked()
        snap = b.fetch_live_status()
        ob = snap.get("onboarding") or {}
        if ob.get("needs_attention"):
            if not ob.get("colab_cli_ok") or not ob.get("rclone_exe_ok"):
                msg = "🟠 缺少本地工具，请安装或定位路径"
            elif not ob.get("colab_auth_ok"):
                msg = "🟠 请绑定 Google Colab 账号"
            else:
                msg = "🟠 请授权绑定 Google Drive 云盘"
        elif free_gb is not None:
            msg = f"🟢 环境全部就绪 · 云盘剩余 {free_gb:.1f} GB"
        else:
            msg = "🟢 账号与环境全部就绪"
        return {"ok": True, "snapshot": snap, "message": msg}

    def _wait_gdrive_bind_worker(self, proc: Any, rclone_conf: Path, sub_mod: Any) -> None:
        """后台等待 rclone 云盘浏览器授权子进程结束并广播结果事件。"""
        b = self._b
        rc = 1
        out_txt = ""
        try:
            out_txt, _ = proc.communicate(timeout=180)
            rc = proc.returncode
        except sub_mod.TimeoutExpired:
            kill_process_tree(proc)
            out_txt = "浏览器授权等待超时（3 分钟）"
        except Exception as exc:
            out_txt = str(exc)
            b._diag.error("gdrive_bind", f"云盘授权异常: {redact_secrets(out_txt)}")
        finally:
            b._gdrive_bind_proc = None

        if b._gdrive_bind_cancelled:
            snap = b._rebuild_and_emit_snapshot()
            self._bus.emit(
                {
                    "type": "onboarding_event",
                    "ok": False,
                    "summary": "已取消 Google 云盘授权",
                    "snapshot": snap,
                }
            )
            return

        if rc == 0 and has_gdrive_save_section(rclone_conf):
            with self._store.lock:
                self._store.state.setdefault("prefs", dict(DEFAULT_PREFS))["rclone_conf_path"] = str(rclone_conf)
                self._store.save()
            free_gb, _ = b._query_gdrive_quota_locked()
            snap = b._rebuild_and_emit_snapshot()
            free_txt = f"（余 {free_gb:.1f}G）" if free_gb is not None else ""
            self._bus.emit(
                {
                    "type": "onboarding_event",
                    "ok": True,
                    "summary": f"✅ Google 云盘 gdrive_save 已绑定{free_txt}",
                    "snapshot": snap,
                }
            )
            return

        tail = redact_secrets((out_txt.strip().splitlines() or [f"退出码 {rc}"])[-1])
        snap = b._rebuild_and_emit_snapshot()
        self._bus.emit(
            {
                "type": "onboarding_event",
                "ok": False,
                "summary": f"⚠️ 云盘授权未完成: {tail[:42]}",
                "snapshot": snap,
            }
        )

    def start_gdrive_bind(self) -> dict[str, Any]:
        """在后台无黑框拉起 rclone 浏览器一键授权并生成 [gdrive_save] 云盘配置。"""
        b = self._b
        sub_mod = self._ctx.subprocess_mod
        rclone_exe = self._ctx.get_rclone_exe()
        if not rclone_exe.is_file():
            return {"ok": False, "message": "未找到 rclone.exe，请先安装或定位路径"}
        if b._gdrive_bind_proc is not None:
            return {"ok": False, "message": "正在浏览器等待 Google 云盘授权，请点击「允许」"}

        rclone_conf = self._ctx.get_rclone_conf()
        if not str(rclone_conf):
            appdata = os.environ.get("APPDATA", "")
            rclone_conf = (
                (Path(appdata) / "rclone" / "rclone.conf")
                if appdata
                else (Path.home() / ".config" / "rclone" / "rclone.conf")
            )
        try:
            rclone_conf.parent.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            return {"ok": False, "message": f"无法创建 rclone 配置目录: {exc}"}

        b._gdrive_bind_cancelled = False
        cmd = [
            str(rclone_exe),
            "config",
            "create",
            "gdrive_save",
            "drive",
            "scope=drive",
            "config_is_local=true",
            "--config",
            str(rclone_conf),
        ]
        try:
            proc = sub_mod.Popen(
                cmd,
                stdin=sub_mod.DEVNULL,
                stdout=sub_mod.PIPE,
                stderr=sub_mod.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=clean_subprocess_env(),
                creationflags=CREATE_NO_WINDOW,
            )
            b._gdrive_bind_proc = proc
        except Exception as exc:
            b._gdrive_bind_proc = None
            return {"ok": False, "message": f"拉起 rclone 授权失败: {exc}"}

        snap_start = b._rebuild_and_emit_snapshot()
        b._tasks.submit(
            "gdrive_bind",
            lambda _cancel_tok: self._wait_gdrive_bind_worker(proc, rclone_conf, sub_mod),
            exclusive=False,
        )
        return {
            "ok": True,
            "binding": True,
            "snapshot": snap_start,
            "message": "已打开浏览器，请登录存储盘账号并点「允许」",
        }

    def cancel_gdrive_bind(self) -> dict[str, Any]:
        """终止等待中的 rclone Google Drive 浏览器授权进程。"""
        b = self._b
        b._gdrive_bind_cancelled = True
        proc = b._gdrive_bind_proc
        b._gdrive_bind_proc = None
        if proc is not None:
            kill_process_tree(proc)
        snap = b._rebuild_and_emit_snapshot()
        return {"ok": True, "snapshot": snap, "message": "已中止等待云盘授权"}

    def import_rclone_conf(self, src_path: str) -> dict[str, Any]:
        """导入用户已有的 rclone.conf 并校验/适配 [gdrive_save] 云盘配置。"""
        b = self._b
        ok, target_p, err_msg = validate_and_import_rclone_conf(src_path, package_dir=self._ctx.package_dir)
        if not ok or target_p is None:
            return {"ok": False, "message": err_msg or "导入 rclone.conf 失败"}

        with self._store.lock:
            self._store.state.setdefault("prefs", dict(DEFAULT_PREFS))["rclone_conf_path"] = str(target_p)
            self._store.save()
        free_gb, _ = b._query_gdrive_quota_locked()
        snap = b._rebuild_and_emit_snapshot()
        free_txt = f" · 余 {free_gb:.1f}G" if free_gb is not None else ""
        return {"ok": True, "snapshot": snap, "message": f"✅ 已接入云盘配置{free_txt}"}

    def set_tool_path(self, kind: str, path_str: str) -> dict[str, Any]:
        """设置或重置 colab.exe / python.exe / rclone.exe / rclone.conf 路径偏好。"""
        b = self._b
        key_map = {
            "colab_exe": "colab_exe_path",
            "colab_py": "colab_py_path",
            "rclone_exe": "rclone_exe_path",
            "rclone_conf": "rclone_conf_path",
        }
        pref_key = key_map.get(str(kind or "").strip().lower())
        if not pref_key:
            return {"ok": False, "message": "未知工具类型"}

        val = str(path_str or "").strip()
        if val and not Path(val).is_file():
            return {"ok": False, "message": f"所选路径文件不存在: {val}"}

        if pref_key == "rclone_conf_path" and val:
            return b.import_rclone_conf(val)

        with self._store.lock:
            self._store.state.setdefault("prefs", dict(DEFAULT_PREFS))[pref_key] = val
            self._store.save()
        snap = b._rebuild_and_emit_snapshot()
        return {
            "ok": True,
            "snapshot": snap,
            "message": "✅ 已更新工具路径" if val else "已恢复自动探测路径",
        }

    def reset_tool_paths(self) -> dict[str, Any]:
        """清除所有手动指定的工具路径，恢复跨设备自动探测。"""
        b = self._b
        with self._store.lock:
            prefs = self._store.state.setdefault("prefs", dict(DEFAULT_PREFS))
            for k in ("colab_exe_path", "colab_py_path", "rclone_exe_path", "rclone_conf_path"):
                prefs[k] = ""
            self._store.save()
        snap = b._rebuild_and_emit_snapshot()
        return {"ok": True, "snapshot": snap, "message": "已恢复全部工具自动探测"}

    def open_local_outputs_dir(self) -> dict[str, Any]:
        """无需等待远端连接，直接创建并打开本机成图产出文件夹。"""
        b = self._b
        b.download_root.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(str(b.download_root))
        except Exception as exc:
            return {"ok": False, "message": f"打开目录失败: {exc}"}
        return {"ok": True, "path": str(b.download_root), "message": "📂 已打开本地成图夹"}

    def run_preflight(
        self,
        deep: bool = False,
        profiles: list[str] | str | None = None,
        gpu: str | None = None,
    ) -> dict[str, Any]:
        """执行启动前多维度环境、授权、磁盘、配置与显卡兼容性自检。"""
        b = self._b
        with self._store.lock:
            prefs = dict(self._store.state.get("prefs") or {})
            oc = dict(self._store.state.get("onboarding_cache") or {})
        eff_profiles = (
            profiles
            if profiles is not None
            else (prefs.get("boot_profiles") or prefs.get("boot_profile") or ["anima"])
        )
        eff_gpu = gpu or prefs.get("boot_gpu")
        tool_paths = self._collect_tool_paths()
        free_gb = oc.get("gdrive_free_gb")
        if deep and free_gb is None:
            free_gb, _ = b._query_gdrive_quota_locked()
        report = run_preflight_checks(
            mode="deep" if deep else "fast",
            profiles=eff_profiles,
            gpu=eff_gpu,
            tool_paths=tool_paths,
            state_path=self._store.state_path,
            download_dir=b.download_root,
            cf_config_path=self._ctx.resolve_cf_config_path(),
            pkg_dir=self._ctx.package_dir,
            gdrive_free_gb=free_gb,
        )
        res = report.to_dict()
        res["blocking_message"] = report.first_blocking_message()
        return res

    def export_diagnostics(self, out_dir: Path | None = None) -> dict[str, Any]:
        """生成全量脱敏的排障诊断 `.zip` 压缩包，方便定位异常。"""
        b = self._b
        try:
            with self._store.lock:
                state_copy = dict(self._store.state)
                snap_copy = dict(b.last_snapshot)
            tool_paths = self._collect_tool_paths()
            zip_path = export_diagnostic_bundle(
                out_dir=out_dir,
                pkg_dir=self._ctx.package_dir,
                state_path=self._store.state_path,
                state_dict=state_copy,
                snapshot=snap_copy,
                task_manager=b._tasks,
                diag_logger=b._diag,
                cf_config_path=self._ctx.resolve_cf_config_path(),
                tool_paths=tool_paths,
                deep_preflight=False,
            )
            is_sandbox_log = bool(str(os.environ.get("COLAB_PET_DIAG_LOG_PATH") or "").strip())
            if os.name == "nt" and out_dir is None and not is_sandbox_log:
                self._ctx.reveal_path(zip_path)
            return {
                "ok": True,
                "path": str(zip_path),
                "message": f"🩺 已导出脱敏诊断包: {zip_path.name}",
            }
        except Exception as exc:
            msg = redact_secrets(str(exc))
            b._diag.error("diagnostics", f"导出诊断包失败: {msg}")
            return {"ok": False, "message": f"导出诊断包失败: {msg}"}

    def open_logs_dir(self) -> dict[str, Any]:
        """创建并打开桌宠诊断日志文件夹。"""
        b = self._b
        log_dir = b._diag.log_path.parent
        log_dir.mkdir(parents=True, exist_ok=True)
        is_sandbox_log = bool(str(os.environ.get("COLAB_PET_DIAG_LOG_PATH") or "").strip())
        if not is_sandbox_log:
            try:
                os.startfile(str(log_dir))
            except Exception as exc:
                return {"ok": False, "message": f"打开日志目录失败: {redact_secrets(str(exc))}"}
        return {"ok": True, "path": str(log_dir), "message": "📄 已打开诊断日志文件夹"}
