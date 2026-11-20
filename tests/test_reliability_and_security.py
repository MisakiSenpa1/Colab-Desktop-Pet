"""第 1~3 步可靠性、安全性与关键边界离线验证测试套件。

覆盖核心场景（全量基于本地沙箱与 Mock，不触碰真实系统凭据、真实 .cf_tunnel_secret.json 或 GPU）：
1. 双会话并存下 `--recreate` 重建、精准停卡（`stop_session_instance`）与 `stop_all_instances` 隔离；
2. 闲置自动停卡的云盘归档强制校验、失败自动重试 1 次，以及 `stop_billing` / `keep_alive` 双策略分流；
3. 停卡接口失败或停后实例仍未释放的强核实拦截（`colab_ctl.stop_session` 与 `ColabPetBackend`）；
4. 一键开机部署中途取消（`cancel_deploy`）的进程清理、锁释放与事件准确上报；
5. 账号切换期间存在后台任务的忙碌拒绝与 `_account_epoch` 围栏防串台；
6. 系统凭据保险箱读写与旧版明文迁移、全域日志/报错/UI 零明文密钥脱敏、真实 `aiohttp` ComfyUI 公网认证网关（含免密票据防重放、防开放重定向、限流及后端 `401` 补水重试）以及 4 处桥接节点源码一致性。
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from PIL import Image
import pytest

PKG_DIR = Path(__file__).resolve().parents[1]
PET_DIR = PKG_DIR / "colab_pet"
for p_str in (str(PKG_DIR), str(PET_DIR)):
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

import colab_ctl  # noqa: E402
import colab_pet_bridge_node as bridge_node  # noqa: E402
import colab_security as csec  # noqa: E402
import colab_backend as cb  # noqa: E402


def _load_pet_app_api():
    import pet_app  # noqa: E402

    return pet_app


@pytest.fixture()
def isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """为每条用例创建完全隔离的沙箱凭据库、诊断日志与临时隧道配置文件。"""
    csec.clear_ephemeral_secrets()
    vault_file = tmp_path / "mock_vault.json"
    cf_file = tmp_path / ".cf_tunnel_secret.json"
    diag_log = tmp_path / "logs" / "colab_diag.log"
    monkeypatch.setenv("COLAB_PET_MOCK_VAULT_PATH", str(vault_file))
    monkeypatch.setenv("COLAB_PET_CF_CONFIG_PATH", str(cf_file))
    monkeypatch.setenv("COLAB_PET_DIAG_LOG_PATH", str(diag_log))
    yield SimpleNamespace(tmp=tmp_path, vault_file=vault_file, cf_file=cf_file, diag_log=diag_log)
    csec.clear_ephemeral_secrets()


def _make_backend(env_ns: SimpleNamespace, name: str = "state") -> cb.ColabPetBackend:
    return cb.ColabPetBackend(
        state_path=env_ns.tmp / f"{name}.json",
        download_root=env_ns.tmp / f"{name}_dl",
        cf_config_path=env_ns.cf_file,
    )


# =========================================================================
# 1. 双会话并存下定向重建、定向停卡与停止全部隔离测试
# =========================================================================


def test_dual_session_recreate_and_stop_isolation(isolated_env, monkeypatch: pytest.MonkeyPatch) -> None:
    tmp = isolated_env.tmp
    fake_exe = tmp / "colab.exe"
    fake_py = tmp / "colab_py.exe"
    fake_conf = tmp / "rclone.conf"
    fake_boot = tmp / "fake_anima_boot.py"
    for fp in (fake_exe, fake_py, fake_conf, fake_boot):
        fp.write_text("[gdrive_save]\ntype = drive\n", encoding="utf-8")

    sessions_db = {"comfy": "ep-comfy-old", "other": "ep-other-keep"}
    assignments = ["ep-comfy-old", "ep-other-keep"]
    unassigned_eps: list[str] = []

    def _fake_run(cmd, check=True, capture=False):
        cmd_s = [str(x) for x in cmd]
        if cmd_s[0] == str(fake_exe):
            sub = cmd_s[1] if len(cmd_s) > 1 else ""
            if sub == "sessions":
                lines = [f"{k} {v}" for k, v in sessions_db.items()]
                return subprocess.CompletedProcess(cmd_s, 0, stdout="\n".join(lines), stderr="")
            if sub == "stop":
                sname = cmd_s[cmd_s.index("-s") + 1] if "-s" in cmd_s else "comfy"
                ep = sessions_db.pop(sname, None)
                if ep and ep in assignments:
                    assignments.remove(ep)
                    unassigned_eps.append(ep)
                return subprocess.CompletedProcess(cmd_s, 0, stdout=f"stopped {sname}", stderr="")
            if sub == "new":
                sname = cmd_s[cmd_s.index("-s") + 1] if "-s" in cmd_s else "comfy"
                new_ep = f"ep-{sname}-new"
                sessions_db[sname] = new_ep
                assignments.append(new_ep)
                return subprocess.CompletedProcess(cmd_s, 0, stdout=f"created {new_ep}", stderr="")
            return subprocess.CompletedProcess(cmd_s, 0, stdout="ok", stderr="")

        if cmd_s[0] == str(fake_py) and len(cmd_s) >= 3 and cmd_s[1] == "-c":
            code = cmd_s[2]
            if "get_session" in code:
                return subprocess.CompletedProcess(
                    cmd_s, 0, stdout=sessions_db.get("comfy", ""), stderr=""
                )
            if "<<<STOP_VERIFIED>>>" in code:
                return subprocess.CompletedProcess(
                    cmd_s,
                    0,
                    stdout="<<<STOP_VERIFIED>>>"
                    + json.dumps({"remaining": list(assignments)})
                    + "<<<END_STOP_VERIFIED>>>",
                    stderr="",
                )
        return subprocess.CompletedProcess(cmd_s, 0, stdout="", stderr="")

    monkeypatch.setattr(colab_ctl, "COLAB_EXE", fake_exe)
    monkeypatch.setattr(colab_ctl, "COLAB_PY", fake_py)
    monkeypatch.setattr(colab_ctl, "RCLONE_CONF", fake_conf)
    monkeypatch.setattr(colab_ctl, "BOOTSTRAP_SCRIPT", fake_boot)
    monkeypatch.setattr(colab_ctl, "_run", _fake_run)
    monkeypatch.setattr(colab_ctl, "_colab_exec_file", lambda *a, **kw: subprocess.CompletedProcess([], 0, "", ""))
    monkeypatch.setattr(colab_ctl, "cmd_status", lambda *_a: None)

    args = argparse.Namespace(
        session="comfy",
        endpoint=None,
        profile="anima",
        profiles=None,
        gpu="L4",
        high_mem=False,
        recreate=True,
        minimax_lite=False,
        no_r2v=False,
        no_pe=False,
    )
    colab_ctl.cmd_up(args)

    assert "ep-comfy-old" in unassigned_eps
    assert "ep-other-keep" not in unassigned_eps
    assert sessions_db.get("comfy") == "ep-comfy-new"
    assert sessions_db.get("other") == "ep-other-keep"
    assert set(assignments) == {"ep-other-keep", "ep-comfy-new"}

    # 后端层验证：stop_session_instance("comfy") 仅传 down -s comfy --endpoint ...（绝不传 --all）
    backend = _make_backend(isolated_env, "dual_stop")
    with backend._lock:
        backend.state["last_assignments_count"] = 2
        backend.state["last_sessions"] = [
            {"name": "comfy", "endpoint": "ep-comfy-new", "accelerator": "L4", "orphan": False},
            {"name": "other", "endpoint": "ep-other-keep", "accelerator": "T4", "orphan": False},
        ]
        backend.state["managed_binding"] = {
            "account_id": "acct1",
            "session": "comfy",
            "endpoint": "ep-comfy-new",
        }
        backend.state["prefs"]["active_account_id"] = "acct1"

    ctl_cmds: list[list[str]] = []

    def _mock_run_hidden(cmd, timeout=25, extra_env=None, input_text=None):
        c = [str(x) for x in cmd]
        ctl_cmds.append(c)
        return subprocess.CompletedProcess(c, 0, stdout="<<<STOP_VERIFIED>>>", stderr="")

    with (
        patch.object(cb, "_run_hidden", side_effect=_mock_run_hidden),
        patch.object(
            backend,
            "fetch_live_status",
            return_value={
                "ok": True,
                "assignments_count": 1,
                "sessions": [{"name": "other", "endpoint": "ep-other-keep", "accelerator": "T4", "orphan": False}],
            },
        ),
    ):
        res = backend.stop_session_instance("comfy")

    assert res["ok"] is True, res
    assert res["remaining_count"] == 1
    assert any("down" in c and "--endpoint" in c and "ep-comfy-new" in c and "--all" not in c for c in ctl_cmds), ctl_cmds

    # 独立 stop_all_instances 应下发 down --all 并核实 0 残留
    ctl_cmds.clear()
    with (
        patch.object(cb, "_run_hidden", side_effect=_mock_run_hidden),
        patch.object(
            backend,
            "fetch_live_status",
            return_value={"ok": True, "assignments_count": 0, "sessions": []},
        ),
    ):
        res_all = backend.stop_all_instances()
    assert res_all["ok"] is True, res_all
    assert any("down" in c and "--all" in c for c in ctl_cmds), ctl_cmds


# =========================================================================
# 2. 闲置自动停卡归档校验、失败重试与 stop_billing / keep_alive 双策略
# =========================================================================


@pytest.mark.parametrize("policy_case", ["stop_billing_fail", "stop_billing_retry_ok", "keep_alive_fail"])
def test_idle_archive_retry_and_policies(
    isolated_env, monkeypatch: pytest.MonkeyPatch, policy_case: str
) -> None:
    monkeypatch.setattr(cb.time, "sleep", lambda *_a: None)
    backend = _make_backend(isolated_env, f"idle_{policy_case}")
    policy = "keep_alive" if policy_case == "keep_alive_fail" else "stop_billing"

    with backend._lock:
        backend.state["prefs"]["idle_auto_stop_minutes"] = 60
        backend.state["prefs"]["idle_archive_fail_policy"] = policy
        backend.state["prefs"]["active_account_id"] = "acct_idle"
        backend.state["last_assignments_count"] = 1
        backend.state["last_sessions"] = [
            {"name": "comfy", "endpoint": "ep-idle-1", "accelerator": "L4", "orphan": False}
        ]
        backend.state["active_endpoint"] = "ep-idle-1"
        backend.state["managed_binding"] = {
            "account_id": "acct_idle",
            "session": "comfy",
            "endpoint": "ep-idle-1",
        }
        backend._last_activity_m = time.monotonic() - 7200.0
    with backend._comfy_lock:
        backend._comfy_state["online"] = True
        backend._comfy_state["running_count"] = 0
        backend._comfy_state["pending_count"] = 0
        backend._comfy_state["llm_running_count"] = 0

    events: list[dict[str, Any]] = []
    done_evt = threading.Event()

    def _on_evt(evt: dict[str, Any]) -> None:
        events.append(dict(evt))
        if evt.get("type") in ("idle_stopped", "idle_archive_failed", "idle_warning"):
            done_evt.set()

    backend._event_cb = _on_evt

    sync_calls: list[list[str]] = []
    stop_calls: list[tuple[str, str | None, str | None]] = []

    def _fake_run_hidden(cmd, timeout=120, extra_env=None, input_text=None):
        c = [str(x) for x in cmd]
        if "sync" in c:
            sync_calls.append(c)
            if policy_case == "stop_billing_retry_ok" and len(sync_calls) == 2:
                return subprocess.CompletedProcess(
                    c, 0, stdout='<<<SYNC_VERIFIED>>>{"ok": true, "files": 2}<<<END_SYNC_VERIFIED>>>', stderr=""
                )
            return subprocess.CompletedProcess(c, 1, stdout="", stderr="rclone: storage quota exceeded")
        return subprocess.CompletedProcess(c, 0, stdout="", stderr="")

    def _fake_stop_session(session="comfy", endpoint=None, account_id=None):
        stop_calls.append((session, endpoint, account_id))
        return {
            "ok": True,
            "snapshot": {"ok": True, "is_running": False, "sessions": [], "assignments_count": 0},
        }

    with (
        patch.object(cb, "_run_hidden", side_effect=_fake_run_hidden),
        patch.object(backend, "stop_session_instance", side_effect=_fake_stop_session),
        patch.object(backend, "stop_all_instances", side_effect=AssertionError("不得误调用 down --all")),
    ):
        backend._maybe_trigger_idle_auto_stop()
        assert done_evt.wait(timeout=5.0), f"未收到预期闲置处理事件: {events}"

    assert len(sync_calls) == 2, f"归档失败应自动重试 1 次（总计 2 次 sync）: {sync_calls}"
    if policy_case == "stop_billing_fail":
        assert len(stop_calls) == 1 and stop_calls[0] == ("comfy", "ep-idle-1", "acct_idle")
        ev = events[-1]
        assert ev["type"] == "idle_stopped" and ev["archive_failed"] is True and ev["is_error"] is True
        assert "失败" in ev["summary"] and "停卡" in ev["summary"]
    elif policy_case == "stop_billing_retry_ok":
        assert len(stop_calls) == 1
        ev = events[-1]
        assert ev["type"] == "idle_stopped" and ev["archive_failed"] is False
    else:
        assert len(stop_calls) == 0, "keep_alive 策略下归档失败严禁停卡"
        ev = events[-1]
        assert ev["type"] == "idle_archive_failed" and ev["is_error"] is True
        assert ev.get("action") == "sync_outputs"
        assert backend._idle_archive_retry_after_m > time.monotonic()


# =========================================================================
# 3. 停卡接口失败及停后实例未释放强核实
# =========================================================================


def test_stop_failure_and_unreleased_verification(isolated_env, monkeypatch: pytest.MonkeyPatch) -> None:
    tmp = isolated_env.tmp
    fake_exe = tmp / "colab.exe"
    fake_py = tmp / "colab_py.exe"
    fake_exe.write_text("", encoding="utf-8")
    fake_py.write_text("", encoding="utf-8")

    monkeypatch.setattr(colab_ctl, "COLAB_EXE", fake_exe)
    monkeypatch.setattr(colab_ctl, "COLAB_PY", fake_py)
    monkeypatch.setattr(colab_ctl, "cmd_status", lambda *_a: None)

    def _stuck_run(cmd, check=True, capture=False):
        cmd_s = [str(x) for x in cmd]
        if cmd_s[0] == str(fake_exe):
            return subprocess.CompletedProcess(cmd_s, 0, stdout="stop ok", stderr="")
        if "<<<STOP_VERIFIED>>>" in cmd_s[2]:
            return subprocess.CompletedProcess(
                cmd_s,
                0,
                stdout='<<<STOP_VERIFIED>>>{"remaining": ["ep-stuck-999"]}<<<END_STOP_VERIFIED>>>',
                stderr="",
            )
        return subprocess.CompletedProcess(cmd_s, 0, stdout="", stderr="")

    monkeypatch.setattr(colab_ctl, "_run", _stuck_run)
    with pytest.raises(RuntimeError, match="ep-stuck-999"):
        colab_ctl.stop_session("comfy", clean_all=False, target_endpoint="ep-stuck-999")
    with pytest.raises(RuntimeError, match="ep-stuck-999"):
        colab_ctl.stop_session("comfy", clean_all=True)

    # 后端层：即使 down 返回 rc=0，只要 fetch_live_status 发现仍在计费就必须返回 ok=False
    backend = _make_backend(isolated_env, "stop_verify_fail")
    with backend._lock:
        backend.state["last_assignments_count"] = 1
        backend.state["last_sessions"] = [
            {"name": "comfy", "endpoint": "ep-stuck-999", "accelerator": "L4", "orphan": False}
        ]

    with (
        patch.object(
            cb,
            "_run_hidden",
            return_value=subprocess.CompletedProcess([], 0, stdout="stop issued", stderr=""),
        ),
        patch.object(
            backend,
            "fetch_live_status",
            return_value={
                "ok": True,
                "assignments_count": 1,
                "sessions": [{"name": "comfy", "endpoint": "ep-stuck-999", "accelerator": "L4", "orphan": False}],
            },
        ),
    ):
        r1 = backend.stop_session_instance("comfy")
        r2 = backend.stop_all_instances()

    assert r1["ok"] is False and r1.get("still_running") is True
    assert "ep-stuck-999" in r1["message"]
    assert r2["ok"] is False and r2.get("still_running") is True


# =========================================================================
# 4. 部署中途取消与资源/锁回收
# =========================================================================


def test_deploy_cancel_flow(isolated_env, monkeypatch: pytest.MonkeyPatch) -> None:
    tmp = isolated_env.tmp
    fake_exe = tmp / "colab.exe"
    fake_py = tmp / "colab_py.exe"
    fake_rclone = tmp / "rclone.exe"
    fake_conf = tmp / "rclone.conf"
    for p in (fake_exe, fake_py, fake_rclone):
        p.write_text("", encoding="utf-8")
    fake_conf.write_text("[gdrive_save]\ntype = drive\n", encoding="utf-8")

    monkeypatch.setattr(cb, "COLAB_EXE_DEFAULT", fake_exe)
    monkeypatch.setattr(cb, "COLAB_PY", fake_py)
    monkeypatch.setattr(cb, "RCLONE_EXE", fake_rclone)
    monkeypatch.setattr(cb, "RCLONE_CONF", fake_conf)

    backend = _make_backend(isolated_env, "cancel_deploy")
    with backend._lock:
        backend.state["account_profiles"] = [{"id": "acct1", "label": "主号", "email_masked": "a***1@gmail.com"}]
        backend.state["prefs"]["active_account_id"] = "acct1"

    events: list[dict[str, Any]] = []
    done_evt = threading.Event()
    started_evt = threading.Event()

    def _on_evt(evt: dict[str, Any]) -> None:
        events.append(dict(evt))
        if evt.get("type") == "deploy_error":
            done_evt.set()

    backend._event_cb = _on_evt

    class _FakeStdout:
        def __init__(self, stop_flag: threading.Event) -> None:
            self._stop_flag = stop_flag
            self._chunks = [
                "=== [1/7] GPU 检查 ===\n",
                "=== [3/7] 正在配置环境 ===\n",
            ]

        def read(self, _n: int) -> str:
            if self._chunks:
                started_evt.set()
                return self._chunks.pop(0)
            self._stop_flag.wait(timeout=5.0)
            return ""

        def close(self) -> None:
            pass

    class _FakeProc:
        def __init__(self) -> None:
            self.pid = None
            self._stopped = threading.Event()
            self.stdout = _FakeStdout(self._stopped)

        def terminate(self) -> None:
            self._stopped.set()

        def kill(self) -> None:
            self._stopped.set()

        def wait(self) -> int:
            self._stopped.wait(timeout=5.0)
            return 1

    with (
        patch.object(cb.subprocess, "Popen", return_value=_FakeProc()),
        patch.object(backend, "fetch_live_status", return_value={"ok": True, "is_running": False, "sessions": []}),
    ):
        res_start = backend.start_instance({"profile": "anima", "gpu": "L4"})
        assert res_start["ok"] is True
        assert started_evt.wait(timeout=3.0)
        assert backend.is_deploying() is True
        res_cancel = backend.cancel_deploy()
        assert res_cancel["ok"] is True
        assert done_evt.wait(timeout=5.0)

    deadline = time.monotonic() + 2.0
    while backend._deploy_lock.locked() and time.monotonic() < deadline:
        time.sleep(0.02)

    assert backend.is_deploying() is False
    assert backend._deploy_lock.locked() is False
    err_evts = [e for e in events if e.get("type") == "deploy_error"]
    assert err_evts and err_evts[-1].get("cancelled") is True


# =========================================================================
# 5. 账号切换期间后台任务忙碌拒绝 & _account_epoch 围栏测试
# =========================================================================


def test_account_switch_busy_and_epoch_fence(isolated_env, monkeypatch: pytest.MonkeyPatch) -> None:
    tmp = isolated_env.tmp
    fake_py = tmp / "colab_py.exe"
    fake_py.write_text("", encoding="utf-8")
    monkeypatch.setattr(cb, "COLAB_PY", fake_py)

    backend = _make_backend(isolated_env, "acct_fence")
    with backend._lock:
        backend.state["prefs"]["active_account_id"] = "acct_u1"
        backend.state["last_balance"] = 951.70
        backend.state["account_profiles"] = [
            {"id": "acct_u1", "label": "Ultra 1号", "email_masked": "u1***@gmail.com"},
            {"id": "acct_u2", "label": "Ultra 2号", "email_masked": "u2***@gmail.com"},
        ]
        backend.state["account_ledgers"] = {
            "acct_u1": {"last_balance": 951.70, "day_start_balance": 951.70, "date": "2026-11-16"},
            "acct_u2": {"last_balance": 1000.00, "day_start_balance": 1000.00, "date": "2026-11-16"},
        }

    # 5a. 部署/上传/自动停卡/操作锁任一忙碌时拒绝切号
    for lock_obj in (backend._upload_lock, backend._idle_stop_lock, backend._deploy_lock, backend._action_lock):
        assert lock_obj.acquire(blocking=False)
        try:
            busy_res = backend.switch_colab_account("acct_u2")
            assert busy_res["ok"] is False and busy_res.get("busy") is True, busy_res
            assert backend.state["prefs"]["active_account_id"] == "acct_u1"
        finally:
            lock_obj.release()

    with backend._lock:
        backend._deploy_state["active"] = True
    try:
        busy_deploy = backend.switch_colab_account("acct_u2")
        assert busy_deploy["ok"] is False and busy_deploy.get("busy") is True
    finally:
        with backend._lock:
            backend._deploy_state["active"] = False

    # 5b. 慢探针在途期间完成切换，_account_epoch 围栏丢弃 acct_u1 过时余额 951.70
    probe_u1_entered = threading.Event()
    probe_u1_release = threading.Event()
    call_seq = {"n": 0}

    def _mock_query_live() -> dict[str, Any]:
        call_seq["n"] += 1
        idx = call_seq["n"]
        if idx == 1:
            probe_u1_entered.set()
            assert probe_u1_release.wait(timeout=5.0)
            return {
                "auth_missing": False,
                "balance": 951.70,
                "rate_hourly": 0.0,
                "assignments_count": 0,
                "sessions": [],
            }
        return {
            "auth_missing": False,
            "balance": 1000.00,
            "rate_hourly": 0.0,
            "assignments_count": 0,
            "sessions": [],
        }

    t1_out: list[dict[str, Any]] = []
    with (
        patch.object(backend, "_query_colab_live", side_effect=_mock_query_live),
        patch.object(
            cb,
            "_run_hidden",
            return_value=subprocess.CompletedProcess([], 0, stdout="<<<COLAB_ACCT_OP>>>{}<<<END_COLAB_ACCT_OP>>>", stderr=""),
        ),
    ):
        t1 = threading.Thread(target=lambda: t1_out.append(backend.fetch_live_status()))
        t1.start()
        assert probe_u1_entered.wait(timeout=3.0)

        sw_res = backend.switch_colab_account("acct_u2")
        assert sw_res["ok"] is True
        assert abs(float(sw_res["snapshot"]["balance"]) - 1000.0) < 0.01

        probe_u1_release.set()
        t1.join(timeout=3.0)

    assert not t1.is_alive()
    assert abs(float(backend.state["last_balance"]) - 1000.0) < 0.01
    assert backend.state["prefs"]["active_account_id"] == "acct_u2"
    assert abs(float(backend.last_snapshot["balance"]) - 1000.0) < 0.01


# =========================================================================
# 6. 凭据保险箱迁移、全链路脱敏与 aiohttp 公网认证网关 + 源码一致性
# =========================================================================


def test_vault_migration_redaction_and_comfy_auth_gateway(
    isolated_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    tmp = isolated_env.tmp
    cf_file = isolated_env.cf_file

    # 6a. 旧版明文 .cf_tunnel_secret.json 自动平滑迁移至凭据库并抹除磁盘明文
    legacy_cf_tok = "eyJLegacyCfJwtHeader123456.PayloadDataABCDEF987654.SignPart111"
    legacy_llm_key = "sk-legacy-fixed-secret-key-99999"
    cf_file.write_text(
        json.dumps(
            {
                "tunnel_token": legacy_cf_tok,
                "llm_api_key": legacy_llm_key,
                "comfy_url": "https://comfy.my-studio.example",
                "llm_url": "https://llm.my-studio.example",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    loaded = csec.load_cf_tunnel_settings(cf_file, include_secrets=True)
    assert loaded["tunnel_token"] == legacy_cf_tok
    assert loaded["llm_api_key"] == legacy_llm_key
    disk_after_migrate = cf_file.read_text(encoding="utf-8")
    assert legacy_cf_tok not in disk_after_migrate
    assert legacy_llm_key not in disk_after_migrate
    assert "tunnel_token_ref" in disk_after_migrate and "llm_api_key_ref" in disk_after_migrate

    # 6b. 后端 save_cf_tunnel_config、日志/报错及 PetBridgeApi 出口零明文泄露
    backend = _make_backend(isolated_env, "sec_backend")
    pet_app = _load_pet_app_api()
    api = pet_app.PetBridgeApi(backend, threading.Event())
    new_cf_tok = "eyJNewCloudflareTunnelJWT12345678.BodySectionXYZ98765.SigPart222"
    new_llm_key = "sk-new-fixed-secret-key-abcdef"
    sv_res = backend.save_cf_tunnel_config(
        {
            "tunnel_token": new_cf_tok,
            "llm_api_key": new_llm_key,
            "comfy_url": "https://comfy2.my-studio.example",
            "llm_url": "https://llm2.my-studio.example",
        }
    )
    backend.update_prefs({"on_top": True})
    assert sv_res["ok"] is True
    assert new_cf_tok not in cf_file.read_text(encoding="utf-8")
    assert new_llm_key not in cf_file.read_text(encoding="utf-8")
    disk_state_txt = (tmp / "sec_backend.json").read_text(encoding="utf-8")
    assert new_cf_tok not in disk_state_txt and new_llm_key not in disk_state_txt

    # 模拟各种含敏感口令的脏日志与事件
    comfy_key = "cpk-TestComfyGatewayKey_0123456789abcdef"
    csec.register_ephemeral_secret(comfy_key)
    oauth_code = "4/0AbCdEfGhIjKlMnOpQrStUvWxYz1234567890"
    oauth_tok = "ya29.a0AbCdEfGhIjKlMnOpQrStUvWxYz_0987654321"
    ticket_sample = csec.create_comfy_handoff_ticket(comfy_key, next_path="/?pet_wf=a.json")
    dirty_text = (
        f"Authorization: Bearer {comfy_key} | "
        f"code={oauth_code} | tok={oauth_tok} | "
        f"llm={new_llm_key} | jwt={new_cf_tok} | "
        f"?authuser=0&colab-runtime-proxy-token=proxy_tok_secret_7788&ticket={ticket_sample}"
    )
    cleaned_text = csec.redact_secrets(dirty_text)
    for sensitive in (comfy_key, oauth_code, oauth_tok, new_llm_key, new_cf_tok, "proxy_tok_secret_7788", ticket_sample):
        assert sensitive not in cleaned_text, f"敏感串未过滤: {sensitive} -> {cleaned_text}"

    # 校验 _emit_event 与 _apply_clipboard_and_redact 过滤且正常写入剪贴板
    emitted: list[dict[str, Any]] = []
    backend._event_cb = lambda e: emitted.append(dict(e))
    backend._emit_event({"type": "upload_error", "summary": dirty_text})
    assert all(s not in emitted[-1]["summary"] for s in (comfy_key, oauth_code, oauth_tok, new_cf_tok))

    clip_box: list[str] = []
    api._set_clipboard = lambda t: clip_box.append(t) or True  # type: ignore[method-assign]
    redacted_api = api._apply_clipboard_and_redact(
        {"ok": False, "message": dirty_text, "clipboard_text": comfy_key}
    )
    assert clip_box == [comfy_key]
    assert "clipboard_text" not in redacted_api
    assert comfy_key not in redacted_api["message"] and oauth_code not in redacted_api["message"]

    # 6c. 真实 aiohttp ComfyUI 公网认证网关实测
    img_bytes_buf: list[bytes] = []
    tmp_img = tmp / "sample_out.png"
    Image.new("RGB", (16, 16), (120, 80, 220)).save(tmp_img)
    img_bytes_buf.append(tmp_img.read_bytes())
    uploaded_names: list[str] = []

    async def _run_gateway_checks() -> None:
        app = web.Application(middlewares=[bridge_node.create_comfy_auth_middleware(lambda: comfy_key)])

        async def _index(_req: web.Request) -> web.Response:
            return web.Response(text="<html>ComfyUI Canvas</html>", content_type="text/html")

        async def _stats(_req: web.Request) -> web.Response:
            return web.json_response(
                {"system": {"os": "posix"}, "devices": [{"name": "NVIDIA L4", "vram_total": 24 * (1024**3), "vram_free": 18 * (1024**3)}]}
            )

        async def _queue(_req: web.Request) -> web.Response:
            return web.json_response({"queue_running": [], "queue_pending": []})

        async def _history(_req: web.Request) -> web.Response:
            return web.json_response(
                {
                    "pid_auth_1": {
                        "outputs": {
                            "9": {"images": [{"filename": "auth_out.png", "subfolder": "", "type": "output"}]}
                        }
                    }
                }
            )

        async def _view(_req: web.Request) -> web.Response:
            return web.Response(body=img_bytes_buf[0], content_type="image/png")

        async def _upload(req: web.Request) -> web.Response:
            reader = await req.multipart()
            while True:
                part = await reader.next()
                if part is None:
                    break
                if part.name == "image":
                    uploaded_names.append(part.filename or "img.png")
                await part.read()
            return web.json_response({"name": uploaded_names[-1]})

        async def _ws(req: web.Request) -> web.WebSocketResponse:
            ws = web.WebSocketResponse()
            await ws.prepare(req)
            await ws.send_str(json.dumps({"type": "executing", "data": {"node": None}}))
            await ws.close()
            return ws

        app.router.add_get("/", _index)
        app.router.add_get("/system_stats", _stats)
        app.router.add_get("/queue", _queue)
        app.router.add_get("/history", _history)
        app.router.add_get("/view", _view)
        app.router.add_post("/upload/image", _upload)
        app.router.add_get("/ws", _ws)

        async with TestClient(TestServer(app)) as client:
            pub_hdrs = {"Host": "demo-pet.trycloudflare.com", "X-Forwarded-Proto": "https"}
            # 1) 公网未授权访问 API / 页面均返回 401
            r_unauth_api = await client.get("/system_stats", headers=pub_hdrs)
            assert r_unauth_api.status == 401
            r_unauth_html = await client.get("/", headers={**pub_hdrs, "Accept": "text/html"})
            assert r_unauth_html.status == 401 and "ComfyUI 工作台已受保护" in (await r_unauth_html.text())

            # 2) 伪造 Host: 127.0.0.1:8188 但携带 Cf-Connecting-Ip 转发头 -> 必须被 401 拦截
            r_spoof = await client.get(
                "/system_stats",
                headers={"Host": "127.0.0.1:8188", "Cf-Connecting-Ip": "203.0.113.88"},
            )
            assert r_spoof.status == 401

            # 3) 纯本机回环直连无转发头 -> 200 免密放行
            r_local = await client.get("/system_stats")
            assert r_local.status == 200

            # 4) 匿名 /__pet_health 返回最小健康检查且不泄露敏感信息
            r_health = await client.get("/__pet_health", headers=pub_hdrs)
            assert r_health.status == 200
            hb = await r_health.json()
            assert hb == {"status": "ok", "service": "comfy", "auth_required": True}

            # 5) WebSocket 未授权拦截 & Bearer 鉴权通过
            r_ws_fail = await client.get("/ws", headers={**pub_hdrs, "Upgrade": "websocket", "Connection": "Upgrade"})
            assert r_ws_fail.status == 401
            async with client.ws_connect("/ws", headers={**pub_hdrs, "Authorization": f"Bearer {comfy_key}"}) as ws:
                msg = await ws.receive_json()
                assert msg["type"] == "executing"

            # 6) 开放重定向攻击测试（next=https://evil.com）必须拒绝并返回 401
            bad_ticket = csec.create_comfy_handoff_ticket(comfy_key, next_path="/")
            r_redir_evil = await client.get(
                f"/__pet_auth?ticket={bad_ticket}&next=https://evil.com",
                headers=pub_hdrs,
                allow_redirects=False,
            )
            assert r_redir_evil.status == 401

            # 7) 桌宠合法一次性票据免密兑换 + 种下 HttpOnly Cookie + 重放二次消费返回 401
            good_ticket = csec.create_comfy_handoff_ticket(comfy_key, next_path="/?pet_wf=ok.json")
            r_handoff = await client.get(
                f"/__pet_auth?ticket={good_ticket}&next=%2F%3Fpet_wf%3Dok.json",
                headers=pub_hdrs,
                allow_redirects=False,
            )
            assert r_handoff.status == 302
            assert r_handoff.headers["Location"] == "/?pet_wf=ok.json"
            cookie_morsel = r_handoff.cookies.get("colab_pet_comfy_auth")
            assert cookie_morsel is not None and bool(cookie_morsel["httponly"])
            assert bool(cookie_morsel["secure"])

            # 本地 TestClient 为 http://，手动回传已颁发的 HttpOnly+Secure Cookie 模拟浏览器在 HTTPS Cloudflare 隧道下的请求
            r_cookie_ok = await client.get(
                "/",
                headers=pub_hdrs,
                cookies={"colab_pet_comfy_auth": cookie_morsel.value},
            )
            assert r_cookie_ok.status == 200
            client.session.cookie_jar.clear()

            r_replay = await client.get(
                f"/__pet_auth?ticket={good_ticket}&next=%2F%3Fpet_wf%3Dok.json",
                headers=pub_hdrs,
                allow_redirects=False,
            )
            assert r_replay.status == 401, "同一 ticket 二次重放必须被拒绝"

            # 8) 口令表单连续 5 次输错限流 429 + 新 IP 正确口令 303 登录
            brute_hdrs = {**pub_hdrs, "Cf-Connecting-Ip": "198.51.100.1"}
            for _ in range(5):
                r_bad_pw = await client.post("/__pet_auth", data={"token": "wrong", "next": "/"}, headers=brute_hdrs)
                assert r_bad_pw.status == 401
            r_429 = await client.post("/__pet_auth", data={"token": comfy_key, "next": "/"}, headers=brute_hdrs)
            assert r_429.status == 429

            good_hdrs = {**pub_hdrs, "Cf-Connecting-Ip": "198.51.100.2"}
            r_good_pw = await client.post(
                "/__pet_auth",
                data={"token": comfy_key, "next": "/"},
                headers=good_hdrs,
                allow_redirects=False,
            )
            assert r_good_pw.status == 303 and "colab_pet_comfy_auth" in r_good_pw.cookies

    asyncio.run(_run_gateway_checks())

    # 6d. ColabPetBackend 端到端对接带鉴权的独立线程化 aiohttp 服务（首轮 401 自动补水密钥并完成轮询/收图/传图）
    runner_box: dict[str, Any] = {}
    srv_ready = threading.Event()

    def _run_threaded_srv() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def _boot() -> None:
            app2 = web.Application(middlewares=[bridge_node.create_comfy_auth_middleware(lambda: comfy_key)])

            async def _st(_r):
                return web.json_response(
                    {"system": {"os": "posix"}, "devices": [{"name": "NVIDIA L4", "vram_total": 24 * (1024**3), "vram_free": 18 * (1024**3)}]}
                )

            async def _q(_r):
                return web.json_response({"queue_running": [], "queue_pending": []})

            hist_calls = {"n": 0}

            async def _h(_r):
                hist_calls["n"] += 1
                if hist_calls["n"] == 1:
                    return web.json_response({})
                return web.json_response(
                    {
                        "pid_sync_1": {
                            "outputs": {"9": {"images": [{"filename": "sync_ok.png", "subfolder": "", "type": "output"}]}}
                        }
                    }
                )

            async def _v(_r):
                return web.Response(body=img_bytes_buf[0], content_type="image/png")

            async def _up(req):
                reader = await req.multipart()
                while True:
                    part = await reader.next()
                    if part is None:
                        break
                    await part.read()
                return web.json_response({"name": "up_ref.png"})

            app2.router.add_get("/system_stats", _st)
            app2.router.add_get("/queue", _q)
            app2.router.add_get("/history", _h)
            app2.router.add_get("/view", _v)
            app2.router.add_post("/upload/image", _up)

            runner = web.AppRunner(app2)
            await runner.setup()
            site = web.TCPSite(runner, "127.0.0.1", 0)
            await site.start()
            port = site._server.sockets[0].getsockname()[1]  # type: ignore[index, union-attr]
            runner_box["runner"] = runner
            runner_box["loop"] = loop
            runner_box["port"] = port
            srv_ready.set()

        loop.run_until_complete(_boot())
        loop.run_forever()

    srv_thread = threading.Thread(target=_run_threaded_srv, daemon=True)
    srv_thread.start()
    assert srv_ready.wait(timeout=5.0)
    srv_url = f"http://127.0.0.1:{runner_box['port']}"
    try:
        with backend._lock:
            backend.state["last_assignments_count"] = 1
            backend.state["last_sessions"] = [
                {"name": "comfy", "endpoint": "ep-gw-test", "accelerator": "L4", "orphan": False}
            ]
            backend._comfy_secrets_by_ep.clear()
        # 注入代理头模拟真实公网请求，强制触发 401 -> 自动补水 -> 重试成功
        backend._http.headers["X-Forwarded-Host"] = "gw-mock.trycloudflare.com"

        def _fake_fast_read(_ep: str, _sname: str):
            backend._comfy_secrets_by_ep["ep-gw-test"] = comfy_key
            return "ok", "ep-gw-test", srv_url

        opened_launch_urls: list[str] = []
        with (
            patch.object(backend, "_fast_read_tunnel_url", side_effect=_fake_fast_read),
            patch.object(cb.webbrowser, "open", side_effect=lambda u: opened_launch_urls.append(u) or True),
        ):
            c1 = backend.poll_comfy_http_once(srv_url)
            assert c1["online"] is True
            c2 = backend.poll_comfy_http_once(srv_url)
            assert c2["last_output_count"] == 1
            backend._comfy_secrets_by_ep.clear()
            msg, up_name = backend._upload_media_file(tmp_img, srv_url)
            assert "up_ref.png" in up_name and "已投递" in msg
            ret_url = backend._open_browser_for_comfy(srv_url, ep="ep-gw-test", workflow_name="test_wf.json")
            assert "ticket=" not in ret_url
            assert opened_launch_urls and "/__pet_auth?ticket=" in opened_launch_urls[-1]
    finally:
        loop = runner_box["loop"]
        asyncio.run_coroutine_threadsafe(runner_box["runner"].cleanup(), loop).result(timeout=3.0)
        loop.call_soon_threadsafe(loop.stop)

    # 6e. 验证 4 处内嵌自定义节点源码与标准 colab_pet_bridge_node.py 100% 一致
    canonical_src = (PKG_DIR / "colab_pet_bridge_node.py").read_text(encoding="utf-8").strip()
    assert cb._COMFY_BRIDGE_NODE_INIT_PY == canonical_src
    for fname in ("colab_bootstrap_anima.py", "colab_bootstrap_drama3.py", "colab_bootstrap_minimax_h3.py"):
        fpath = PKG_DIR / fname
        spec = importlib.util.spec_from_file_location(fpath.stem, str(fpath))
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert mod._PET_BRIDGE_INIT_PY == canonical_src, f"{fname} 与 colab_pet_bridge_node.py 不同步"


# =========================================================================
# 7. CLI `--auto-stop` 防吞异常、Session not found 漏判拦截与桌宠停卡失败退避
# =========================================================================


def test_cmd_run_auto_stop_never_swallows_failure(
    isolated_env, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    tmp = isolated_env.tmp
    fake_exe = tmp / "colab.exe"
    fake_py = tmp / "colab_py.exe"
    fake_exe.write_text("", encoding="utf-8")
    fake_py.write_text("", encoding="utf-8")

    job_script = tmp / "batch_job.py"
    job_script.write_text("print('batch ok')\n", encoding="utf-8")

    monkeypatch.setattr(colab_ctl, "COLAB_EXE", fake_exe)
    monkeypatch.setattr(colab_ctl, "COLAB_PY", fake_py)
    monkeypatch.setattr(colab_ctl, "cmd_status", lambda *_a: None)
    monkeypatch.setattr(colab_ctl, "cmd_wait", lambda *_a: None)
    monkeypatch.setattr(colab_ctl, "sync_session_outputs", lambda *_a, **_kw: None)

    # Scenario 1: 主脚本与云盘同步成功，但 --auto-stop 停卡核实发现实例仍在计费 -> 绝不可吞掉异常并返回 0
    def _run_stuck_stop(cmd, check=True, capture=False):
        cmd_s = [str(x) for x in cmd]
        if cmd_s[0] == str(fake_exe):
            return subprocess.CompletedProcess(cmd_s, 0, stdout="stop ok", stderr="")
        if len(cmd_s) >= 3 and "get_session" in cmd_s[2]:
            return subprocess.CompletedProcess(cmd_s, 0, stdout="ep-bill-1", stderr="")
        if len(cmd_s) >= 3 and "<<<STOP_VERIFIED>>>" in cmd_s[2]:
            return subprocess.CompletedProcess(
                cmd_s,
                0,
                stdout='<<<STOP_VERIFIED>>>{"remaining": ["ep-bill-1"]}<<<END_STOP_VERIFIED>>>',
                stderr="",
            )
        return subprocess.CompletedProcess(cmd_s, 0, stdout="", stderr="")

    monkeypatch.setattr(colab_ctl, "_run", _run_stuck_stop)
    monkeypatch.setattr(colab_ctl, "_colab_exec_file", lambda *_a, **_kw: subprocess.CompletedProcess([], 0, "", ""))

    capsys.readouterr()
    rc1 = colab_ctl.main(["run", str(job_script), "--auto-stop"])
    cap1 = capsys.readouterr()
    assert rc1 == 1, "主任务成功但 --auto-stop 失败时退出码必须为非零"
    assert "ep-bill-1" in cap1.err and "自动释放实例失败" in cap1.err
    diag_txt = isolated_env.diag_log.read_text(encoding="utf-8")
    assert "ep-bill-1" in diag_txt and "auto_stop" in diag_txt

    # Scenario 2: 主脚本抛异常失败，抢救同步后 --auto-stop 也失败 -> 必须包含仍在计费告警且返回 1
    def _boom_exec(*_a, **_kw):
        raise RuntimeError("远端工作流提交失败")

    monkeypatch.setattr(colab_ctl, "_colab_exec_file", _boom_exec)
    rc2 = colab_ctl.main(["run", str(job_script), "--auto-stop"])
    cap2 = capsys.readouterr()
    assert rc2 == 1
    assert "远端工作流提交失败" in cap2.err
    assert "--auto-stop 自动停卡失败" in cap2.err and "实例仍在计费" in cap2.err

    # Scenario 3: 任务执行期间本地 sessions.json 丢失，"colab stop" 返回 rc=0 但打印 Session not found
    monkeypatch.setattr(colab_ctl, "_colab_exec_file", lambda *_a, **_kw: subprocess.CompletedProcess([], 0, "", ""))
    state_3 = {"session_lost": False, "active": ["ep-lost-99"], "fallback_acted": False}

    def _run_session_lost(cmd, check=True, capture=False):
        cmd_s = [str(x) for x in cmd]
        if cmd_s[0] == str(fake_exe) and len(cmd_s) > 1 and cmd_s[1] == "stop":
            if state_3["session_lost"]:
                return subprocess.CompletedProcess(
                    cmd_s, 0, stdout="[colab] Session 'comfy' not found.\n", stderr=""
                )
            return subprocess.CompletedProcess(cmd_s, 0, stdout="stopped", stderr="")
        if len(cmd_s) >= 3 and "get_session" in cmd_s[2]:
            # 第 1 次（cmd_run 前）捕获到 ep-lost-99，随后模拟运行中 sessions.json 丢失
            if not state_3["session_lost"]:
                state_3["session_lost"] = True
                return subprocess.CompletedProcess(cmd_s, 0, stdout="ep-lost-99", stderr="")
            return subprocess.CompletedProcess(cmd_s, 0, stdout="", stderr="")
        if len(cmd_s) >= 3 and "<<<STOP_FALLBACK>>>" in cmd_s[2]:
            state_3["fallback_acted"] = True
            if "ep-lost-99" in state_3["active"]:
                state_3["active"].remove("ep-lost-99")
            return subprocess.CompletedProcess(
                cmd_s,
                0,
                stdout='<<<STOP_FALLBACK>>>{"acted": true, "active_eps": []}<<<END_STOP_FALLBACK>>>',
                stderr="",
            )
        if len(cmd_s) >= 3 and "<<<STOP_VERIFIED>>>" in cmd_s[2]:
            return subprocess.CompletedProcess(
                cmd_s,
                0,
                stdout="<<<STOP_VERIFIED>>>" + json.dumps({"remaining": state_3["active"]}) + "<<<END_STOP_VERIFIED>>>",
                stderr="",
            )
        return subprocess.CompletedProcess(cmd_s, 0, stdout="", stderr="")

    monkeypatch.setattr(colab_ctl, "_run", _run_session_lost)
    rc3a = colab_ctl.main(["run", str(job_script), "--auto-stop"])
    capsys.readouterr()
    assert rc3a == 0
    assert state_3["fallback_acted"] is True and state_3["active"] == []

    # 3b: 直接 stop_session 无预绑定 endpoint，colab stop 返回 0 + Session not found，但云端仍有计费实例 -> 必抛错
    def _run_unclaimed_billing(cmd, check=True, capture=False):
        cmd_s = [str(x) for x in cmd]
        if cmd_s[0] == str(fake_exe) and len(cmd_s) > 1 and cmd_s[1] == "stop":
            return subprocess.CompletedProcess(
                cmd_s, 0, stdout="[colab] Session 'comfy' not found.\n", stderr=""
            )
        if len(cmd_s) >= 3 and "get_session" in cmd_s[2]:
            return subprocess.CompletedProcess(cmd_s, 0, stdout="", stderr="")
        if len(cmd_s) >= 3 and "<<<STOP_FALLBACK>>>" in cmd_s[2]:
            return subprocess.CompletedProcess(
                cmd_s,
                0,
                stdout='<<<STOP_FALLBACK>>>{"acted": false, "active_eps": ["ep-orphan-still-billing"]}<<<END_STOP_FALLBACK>>>',
                stderr="",
            )
        if len(cmd_s) >= 3 and "<<<STOP_VERIFIED>>>" in cmd_s[2]:
            return subprocess.CompletedProcess(
                cmd_s,
                0,
                stdout='<<<STOP_VERIFIED>>>{"remaining": ["ep-orphan-still-billing"]}<<<END_STOP_VERIFIED>>>',
                stderr="",
            )
        return subprocess.CompletedProcess(cmd_s, 0, stdout="", stderr="")

    monkeypatch.setattr(colab_ctl, "_run", _run_unclaimed_billing)
    with pytest.raises(RuntimeError, match="ep-orphan-still-billing"):
        colab_ctl.stop_session("comfy", clean_all=False)

    # Scenario 4: 桌宠闲置自动停卡返回 ok=False -> 记录诊断日志、推送错误提醒并设置 60s 重试退避
    backend = _make_backend(isolated_env, "idle_stop_fail_backoff")
    with backend._lock:
        backend.state["prefs"]["idle_auto_stop_minutes"] = 60
        backend.state["prefs"]["idle_archive_fail_policy"] = "stop_billing"
        backend.state["prefs"]["active_account_id"] = "acct_idle_fail"
        backend.state["last_assignments_count"] = 1
        backend.state["last_sessions"] = [
            {"name": "comfy", "endpoint": "ep-idle-fail", "accelerator": "L4", "orphan": False}
        ]
        backend.state["active_endpoint"] = "ep-idle-fail"
        backend.state["managed_binding"] = {
            "account_id": "acct_idle_fail",
            "session": "comfy",
            "endpoint": "ep-idle-fail",
        }
        backend._last_activity_m = time.monotonic() - 7200.0
    with backend._comfy_lock:
        backend._comfy_state["online"] = True
        backend._comfy_state["running_count"] = 0
        backend._comfy_state["pending_count"] = 0
        backend._comfy_state["llm_running_count"] = 0

    idle_events: list[dict[str, Any]] = []
    idle_done = threading.Event()

    def _on_idle_evt(evt: dict[str, Any]) -> None:
        idle_events.append(dict(evt))
        if evt.get("type") == "idle_warning" and evt.get("is_error"):
            idle_done.set()

    backend._event_cb = _on_idle_evt
    with (
        patch.object(
            cb,
            "_run_hidden",
            return_value=subprocess.CompletedProcess(
                [], 0, stdout='<<<SYNC_VERIFIED>>>{"ok": true, "files": 1}<<<END_SYNC_VERIFIED>>>', stderr=""
            ),
        ),
        patch.object(
            backend,
            "stop_session_instance",
            return_value={"ok": False, "still_running": True, "message": "实例 ep-idle-fail 仍处于活跃状态"},
        ),
    ):
        before_m = time.monotonic()
        backend._maybe_trigger_idle_auto_stop()
        assert idle_done.wait(timeout=5.0), f"未收到闲置停卡失败告警: {idle_events}"

    assert backend._idle_archive_retry_after_m >= before_m + 50.0
    assert any(e.get("is_error") and "ep-idle-fail" in str(e.get("summary")) for e in idle_events)
    assert "ep-idle-fail" in isolated_env.diag_log.read_text(encoding="utf-8")


# =========================================================================
# 8. 停卡链路子进程超时防挂起、进程树强杀与 Endpoint 锁定失败显式告警
# =========================================================================


def _framed(marker: str, payload: dict[str, Any]) -> str:
    return f"<<<{marker}>>>" + json.dumps(payload) + f"<<<END_{marker}>>>"


def test_ctl_run_timeout_kills_whole_process_tree(isolated_env) -> None:
    """_run 指定 timeout 时必须强杀整个进程树并抛 TimeoutExpired，不能被持有管道的孙进程拖死。"""
    # 子进程再派生一个继承 stdout/stderr 管道的孙进程，模拟 colab.exe(蹦床) -> python 的卡死场景
    child_code = (
        "import subprocess, sys, time;"
        "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']);"
        "time.sleep(60)"
    )
    t0 = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        colab_ctl._run([sys.executable, "-c", child_code], check=False, capture=True, timeout=1.5)
    assert time.monotonic() - t0 < 20, "超时后未能及时回收进程树，communicate 被孙进程管道拖住"
    assert "命令超时" in isolated_env.diag_log.read_text(encoding="utf-8")


def test_stop_chain_timeouts_never_hang_or_swallow(
    isolated_env, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    tmp = isolated_env.tmp
    fake_exe = tmp / "colab.exe"
    fake_py = tmp / "colab_py.exe"
    fake_exe.write_text("", encoding="utf-8")
    fake_py.write_text("", encoding="utf-8")
    job_script = tmp / "timeout_job.py"
    job_script.write_text("print('job ok')\n", encoding="utf-8")

    monkeypatch.setattr(colab_ctl, "COLAB_EXE", fake_exe)
    monkeypatch.setattr(colab_ctl, "COLAB_PY", fake_py)
    monkeypatch.setattr(colab_ctl, "cmd_status", lambda *_a: None)
    monkeypatch.setattr(colab_ctl, "cmd_wait", lambda *_a: None)
    monkeypatch.setattr(colab_ctl, "sync_session_outputs", lambda *_a, **_kw: None)
    monkeypatch.setattr(colab_ctl, "_colab_exec_file", lambda *_a, **_kw: subprocess.CompletedProcess([], 0, "", ""))

    def _cp(cmd_s: list[str], stdout: str = "", rc: int = 0, stderr: str = "") -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(cmd_s, rc, stdout=stdout, stderr=stderr)

    # Scenario A: colab stop 卡死超时 -> 不再傻等 3 轮，立即转云端兜底释放并核实成功；所有调用都必须带超时上限
    active_a = ["ep-hang-1"]
    seen_timeouts: list[Any] = []
    stop_calls: list[list[str]] = []

    def _run_stop_hang(cmd, check=True, capture=False, timeout=None):
        cmd_s = [str(x) for x in cmd]
        seen_timeouts.append(timeout)
        if cmd_s[0] == str(fake_exe) and len(cmd_s) > 1 and cmd_s[1] == "stop":
            stop_calls.append(cmd_s)
            raise subprocess.TimeoutExpired(cmd_s, timeout)
        code = cmd_s[2] if len(cmd_s) >= 3 else ""
        if "get_session" in code:
            return _cp(cmd_s, "ep-hang-1")
        if "<<<STOP_FALLBACK>>>" in code:
            if "ep-hang-1" in active_a:
                active_a.remove("ep-hang-1")
            return _cp(cmd_s, _framed("STOP_FALLBACK", {"acted": True, "active_eps": []}))
        if "<<<STOP_VERIFIED>>>" in code:
            return _cp(cmd_s, _framed("STOP_VERIFIED", {"remaining": list(active_a)}))
        return _cp(cmd_s)

    monkeypatch.setattr(colab_ctl, "_run", _run_stop_hang)
    capsys.readouterr()
    summary = colab_ctl.stop_session("comfy", clean_all=False)
    cap_a = capsys.readouterr()
    assert summary["released_endpoint"] == "ep-hang-1" and active_a == []
    assert len(stop_calls) == 1, "colab stop 超时后不应再重试 3 轮"
    assert seen_timeouts and all(t is not None and t > 0 for t in seen_timeouts), f"存在无超时调用: {seen_timeouts}"
    assert "超时" in cap_a.err

    # Scenario B: 停卡后核实步骤卡死 -> 必须抛错并提示可能仍在计费，cmd_run 退出码非零
    def _run_verify_hang(cmd, check=True, capture=False, timeout=None):
        cmd_s = [str(x) for x in cmd]
        if cmd_s[0] == str(fake_exe):
            return _cp(cmd_s, "stopped")
        code = cmd_s[2] if len(cmd_s) >= 3 else ""
        if "get_session" in code:
            return _cp(cmd_s, "ep-verify-hang")
        if "<<<STOP_VERIFIED>>>" in code:
            raise subprocess.TimeoutExpired(cmd_s, timeout)
        return _cp(cmd_s)

    monkeypatch.setattr(colab_ctl, "_run", _run_verify_hang)
    rc_b = colab_ctl.main(["run", str(job_script), "--auto-stop"])
    cap_b = capsys.readouterr()
    assert rc_b == 1
    assert "核实释放状态超时" in cap_b.err and "计费" in cap_b.err

    # Scenario C: Endpoint 预锁定查询失败不再静默吞掉 -> 明确告警；stop 成功但无法精确核实时列出剩余实例
    def _run_probe_fail(cmd, check=True, capture=False, timeout=None):
        cmd_s = [str(x) for x in cmd]
        if cmd_s[0] == str(fake_exe):
            return _cp(cmd_s, "stopped")
        code = cmd_s[2] if len(cmd_s) >= 3 else ""
        if "get_session" in code:
            return _cp(cmd_s, "", rc=1, stderr="Traceback ...\nModuleNotFoundError: No module named 'colab_cli'\n")
        if "<<<STOP_VERIFIED>>>" in code:
            return _cp(cmd_s, _framed("STOP_VERIFIED", {"remaining": ["ep-other-sess"]}))
        return _cp(cmd_s)

    monkeypatch.setattr(colab_ctl, "_run", _run_probe_fail)
    rc_c = colab_ctl.main(["run", str(job_script), "--auto-stop"])
    cap_c = capsys.readouterr()
    assert rc_c == 0, "colab stop 已成功且剩余实例可能属于其他会话时不应误判失败"
    assert "ModuleNotFoundError" in cap_c.err, "Endpoint 查询失败原因必须可见"
    assert "未能预先锁定" in cap_c.err and "无法精确核实" in cap_c.err
    assert "ep-other-sess" in cap_c.err


# =========================================================================
# 9. R1/R2 ComfyUI 成图自动/手动拉取全败禁误报与部分成功逐项重试收敛
# =========================================================================


class _DummyComfyResp:
    """用于沙箱模拟 ComfyUI HTTP 响应的轻量上下文对象。"""

    def __init__(
        self,
        *,
        status_code: int = 200,
        json_data: Any = None,
        body: bytes = b"",
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self._json_data = json_data
        self._body = body
        self.headers = headers or {}

    def __enter__(self):
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            import requests

            raise requests.HTTPError(f"HTTP {self.status_code}", response=self)  # type: ignore[arg-type]

    def json(self) -> Any:
        return self._json_data

    def iter_content(self, chunk_size: int = 65536) -> list[bytes]:
        if not self._body:
            return []
        return [
            self._body[i : i + chunk_size]
            for i in range(0, len(self._body), chunk_size)
        ]


def _make_png_bytes(color: tuple[int, int, int] = (90, 160, 240)) -> bytes:
    import io

    buf = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(buf, format="PNG")
    return buf.getvalue()


def test_comfy_auto_pull_all_failed_never_claims_saved_and_keeps_pending(
    isolated_env,
) -> None:
    """R1：auto_pull_outputs=True 下 /view 全部 502/超时时，绝不报「已存本机」且完整保留待重试队列。"""
    import requests

    backend = _make_backend(isolated_env, "r1_all_fail")
    with backend._lock:
        backend.state["prefs"]["auto_pull_outputs"] = True
    events: list[dict[str, Any]] = []
    backend._event_cb = lambda evt: events.append(dict(evt))

    hist_calls = {"n": 0}
    base_url = "https://mock-r1.trycloudflare.com"

    def _mock_http_get(url: str, *, base_url: str = "", ep: str = "", retry_on_401: bool = True, **kwargs: Any):
        del base_url, ep, retry_on_401
        if url.endswith("/system_stats"):
            return _DummyComfyResp(
                json_data={
                    "devices": [
                        {"name": "NVIDIA L4", "vram_total": 24 * (1024**3), "vram_free": 20 * (1024**3)}
                    ]
                }
            )
        if url.endswith("/queue"):
            return _DummyComfyResp(json_data={"queue_running": [], "queue_pending": []})
        if url.endswith("/history"):
            hist_calls["n"] += 1
            if hist_calls["n"] == 1:
                return _DummyComfyResp(json_data={})
            return _DummyComfyResp(
                json_data={
                    "pid_fail_1": {
                        "outputs": {
                            "9": {
                                "images": [
                                    {"filename": "bad_502.png", "subfolder": "", "type": "output"},
                                    {"filename": "bad_timeout.png", "subfolder": "", "type": "output"},
                                ]
                            }
                        }
                    }
                }
            )
        if url.endswith("/view"):
            fn = str((kwargs.get("params") or {}).get("filename") or "")
            if fn == "bad_502.png":
                return _DummyComfyResp(status_code=502)
            raise requests.exceptions.ReadTimeout("simulated view read timeout")
        raise AssertionError(f"意外请求路径: {url}")

    # 预先放入一份历史旧图，校验全败重试时严禁退回打开旧图假装成功
    old_ok = isolated_env.tmp / "old_prev.png"
    old_ok.write_bytes(_make_png_bytes())
    with backend._comfy_lock:
        backend._last_downloaded_files = [old_ok]

    revealed: list[Path] = []
    with (
        patch.object(backend, "_comfy_http_get", side_effect=_mock_http_get),
        patch.object(cb, "_reveal_path_in_explorer", side_effect=lambda p: revealed.append(Path(p))),
    ):
        c1 = backend.poll_comfy_http_once(base_url)
        assert c1["online"] is True

        events.clear()
        c2 = backend.poll_comfy_http_once(base_url)
        out_evts = [e for e in events if e.get("type") == "outputs_ready"]
        assert len(out_evts) == 1, events
        ev = out_evts[0]
        assert "已存本机" not in ev["summary"]
        assert "尚未存本机" in ev["summary"]
        assert ev["pulled"] is False
        assert ev["is_error"] is True
        assert ev["count"] == 2 and ev["failed_count"] == 2
        assert ev["files"] == []
        assert c2["pending_pull_count"] == 2
        assert c2["last_output_count"] == 0
        assert [m["filename"] for _, m in backend._pending_pull_items] == [
            "bad_502.png",
            "bad_timeout.png",
        ]
        assert not any(backend.download_root.rglob("*"))

        retry_res = backend.pull_or_open_latest_outputs()
        assert retry_res["ok"] is False
        assert retry_res["failed_count"] == 2
        assert "拉取失败" in retry_res["message"]
        assert revealed == []
        assert backend.get_comfy_state()["pending_pull_count"] == 2
        assert len(backend._pending_pull_items) == 2

    diag_txt = isolated_env.diag_log.read_text(encoding="utf-8")
    assert "bad_502.png" in diag_txt and "bad_timeout.png" in diag_txt


def test_comfy_partial_pull_and_incremental_retry_convergence(
    isolated_env,
) -> None:
    """R2：3 张成图首轮 1 成 2 败时仅落盘 1 张并精确保留后 2 张，后续逐轮重试精确收敛至 0。"""
    import requests

    backend = _make_backend(isolated_env, "r2_partial_retry")
    with backend._lock:
        backend.state["prefs"]["auto_pull_outputs"] = True
    events: list[dict[str, Any]] = []
    backend._event_cb = lambda evt: events.append(dict(evt))

    png_1 = _make_png_bytes((220, 90, 90))
    png_2 = _make_png_bytes((90, 220, 90))
    png_3 = _make_png_bytes((90, 90, 220))
    png_4 = _make_png_bytes((220, 220, 90))

    hist_seq = {"n": 0}
    view_mode: dict[str, Any] = {
        "img_1.png": png_1,
        "img_2.png": 502,
        "img_3.png": "timeout",
        "img_4.png": png_4,
    }
    base_url = "https://mock-r2.trycloudflare.com"

    def _mock_http_get(url: str, *, base_url: str = "", ep: str = "", retry_on_401: bool = True, **kwargs: Any):
        del base_url, ep, retry_on_401
        if url.endswith("/system_stats"):
            return _DummyComfyResp(
                json_data={
                    "devices": [
                        {"name": "NVIDIA L4", "vram_total": 24 * (1024**3), "vram_free": 18 * (1024**3)}
                    ]
                }
            )
        if url.endswith("/queue"):
            return _DummyComfyResp(json_data={"queue_running": [], "queue_pending": []})
        if url.endswith("/history"):
            hist_seq["n"] += 1
            if hist_seq["n"] == 1:
                return _DummyComfyResp(json_data={})
            if hist_seq["n"] == 2:
                return _DummyComfyResp(
                    json_data={
                        "pid_partial_3": {
                            "outputs": {
                                "9": {
                                    "images": [
                                        {"filename": "img_1.png", "subfolder": "", "type": "output"},
                                        {"filename": "img_2.png", "subfolder": "", "type": "output"},
                                        {"filename": "img_3.png", "subfolder": "", "type": "output"},
                                    ]
                                }
                            }
                        }
                    }
                )
            return _DummyComfyResp(
                json_data={
                    "pid_partial_3": {
                        "outputs": {
                            "9": {
                                "images": [
                                    {"filename": "img_1.png", "subfolder": "", "type": "output"},
                                    {"filename": "img_2.png", "subfolder": "", "type": "output"},
                                    {"filename": "img_3.png", "subfolder": "", "type": "output"},
                                ]
                            }
                        }
                    },
                    "pid_next_ok": {
                        "outputs": {
                            "9": {"images": [{"filename": "img_4.png", "subfolder": "", "type": "output"}]}
                        }
                    },
                }
            )
        if url.endswith("/view"):
            fn = str((kwargs.get("params") or {}).get("filename") or "")
            behavior = view_mode.get(fn)
            if isinstance(behavior, bytes):
                return _DummyComfyResp(
                    body=behavior,
                    headers={"Last-Modified": "Wed, 18 Nov 2026 04:00:00 GMT"},
                )
            if behavior == "timeout":
                raise requests.exceptions.Timeout("simulated view timeout")
            return _DummyComfyResp(status_code=int(behavior or 502))
        raise AssertionError(f"意外请求路径: {url}")

    revealed: list[Path] = []
    with (
        patch.object(backend, "_comfy_http_get", side_effect=_mock_http_get),
        patch.object(cb, "_reveal_path_in_explorer", side_effect=lambda p: revealed.append(Path(p))),
    ):
        backend.poll_comfy_http_once(base_url)

        # 3 张成图第 1 张成功、后 2 张失败 -> 仅落盘 1 张，pending_pull_count == 2 保留后 2 张
        events.clear()
        c2 = backend.poll_comfy_http_once(base_url)
        disk_files = [p for p in backend.download_root.rglob("*") if p.is_file()]
        assert len(disk_files) == 1 and "img_1" in disk_files[0].name
        assert c2["pending_pull_count"] == 2
        assert c2["last_output_count"] == 1
        assert [m["filename"] for _, m in backend._pending_pull_items] == ["img_2.png", "img_3.png"]

        ev2 = next(e for e in events if e.get("type") == "outputs_ready")
        assert ev2["pulled"] is False and ev2["partial"] is True and ev2["is_warning"] is True
        assert ev2["count"] == 1 and ev2["failed_count"] == 2
        assert len(ev2["files"]) == 1
        assert "已存本机 1 份" in ev2["summary"] and "2 份拉取失败" in ev2["summary"]

        # 后续另一批全成功新出图不得误清此前积压的 img_2/img_3 待重试项
        events.clear()
        c3 = backend.poll_comfy_http_once(base_url)
        ev3 = next(e for e in events if e.get("type") == "outputs_ready")
        assert ev3["pulled"] is True and ev3["count"] == 1
        assert c3["pending_pull_count"] == 2
        assert [m["filename"] for _, m in backend._pending_pull_items] == ["img_2.png", "img_3.png"]

        # 首次手动拉回：第 2 张成功、第 3 张仍失败 -> ok=False, partial=True, 精确只剩第 3 张
        view_mode["img_2.png"] = png_2
        res1 = backend.pull_or_open_latest_outputs()
        assert res1["ok"] is False
        assert res1["partial"] is True
        assert res1["saved_count"] == 1 and res1["failed_count"] == 1
        assert Path(res1["path"]).exists() and "img_2" in Path(res1["path"]).name
        assert backend.get_comfy_state()["pending_pull_count"] == 1
        assert [m["filename"] for _, m in backend._pending_pull_items] == ["img_3.png"]

        # 再次重试第 3 张成功 -> pending_pull_count == 0 且返回 ok=True
        view_mode["img_3.png"] = png_3
        res2 = backend.pull_or_open_latest_outputs()
        assert res2["ok"] is True
        assert Path(res2["path"]).exists() and "img_3" in Path(res2["path"]).name
        assert backend.get_comfy_state()["pending_pull_count"] == 0
        assert backend._pending_pull_items == []
        assert revealed[-1] == Path(res2["path"]).resolve()


def test_comfy_pending_dedup_and_monkeypatch_download_compat(
    isolated_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """验证待拉取队列去重保序辅助及 monkeypatch `_download_comfy_outputs` 兼容分支。"""
    from backend_comfy import ComfySyncService

    item_a1 = ("https://demo.example/", {"filename": "a.png", "subfolder": "", "type": "output"})
    item_a2 = ("https://demo.example", {"filename": "a.png"})
    item_b = ("https://demo.example", {"filename": "b.png", "subfolder": "sub", "type": "output"})
    merged = ComfySyncService._merge_pending_items([item_a1], [item_a2, item_b])
    assert len(merged) == 2
    assert merged[0][1]["filename"] == "a.png" and merged[1][1]["filename"] == "b.png"

    backend = _make_backend(isolated_env, "patch_dl_compat")
    with backend._lock:
        backend.state["prefs"]["auto_pull_outputs"] = True
    events: list[dict[str, Any]] = []
    backend._event_cb = lambda evt: events.append(dict(evt))

    hist_calls = {"n": 0}
    base_url = "https://mock-patch.trycloudflare.com"

    def _mock_http_get(url: str, **_kwargs: Any):
        if url.endswith("/system_stats"):
            return _DummyComfyResp(json_data={"devices": []})
        if url.endswith("/queue"):
            return _DummyComfyResp(json_data={"queue_running": [], "queue_pending": []})
        if url.endswith("/history"):
            hist_calls["n"] += 1
            if hist_calls["n"] == 1:
                return _DummyComfyResp(json_data={})
            return _DummyComfyResp(
                json_data={
                    "pid_patch_1": {
                        "outputs": {
                            "9": {
                                "images": [
                                    {"filename": "p_one.png", "subfolder": "", "type": "output"},
                                    {"filename": "p_two.png", "subfolder": "", "type": "output"},
                                ]
                            }
                        }
                    }
                }
            )
        raise AssertionError(f"patched 下载路径不应直接请求 {url}")

    saved_one = backend.download_root / "p_one_20261118-120000.png"
    saved_two = backend.download_root / "p_two_20261118-120001.png"
    backend.download_root.mkdir(parents=True, exist_ok=True)
    saved_one.write_bytes(_make_png_bytes())
    saved_two.write_bytes(_make_png_bytes())

    dl_mode = {"phase": "partial"}

    def _patched_download(items: list[tuple[str, dict[str, str]]]) -> list[Path]:
        if dl_mode["phase"] == "partial":
            assert len(items) == 2
            return [saved_one]
        assert len(items) == 1 and items[0][1]["filename"] == "p_two.png"
        return [saved_two]

    monkeypatch.setattr(backend, "_download_comfy_outputs", _patched_download)

    with (
        patch.object(backend, "_comfy_http_get", side_effect=_mock_http_get),
        patch.object(cb, "_reveal_path_in_explorer", return_value=True),
    ):
        backend.poll_comfy_http_once(base_url)
        c2 = backend.poll_comfy_http_once(base_url)
        assert c2["pending_pull_count"] == 1
        assert [m["filename"] for _, m in backend._pending_pull_items] == ["p_two.png"]
        ev = next(e for e in events if e.get("type") == "outputs_ready")
        assert ev["partial"] is True and ev["pulled"] is False

        dl_mode["phase"] = "retry_ok"
        res = backend.pull_or_open_latest_outputs()
        assert res["ok"] is True
        assert backend.get_comfy_state()["pending_pull_count"] == 0


def test_comfy_download_mkdir_failure_still_persists_pending_and_recovers(isolated_env) -> None:
    """验证创建下载目录失败时已先 Write-Ahead 持久化待重试队列，修复目录后可成功拉回。"""
    tmp = isolated_env.tmp
    blocked_root = tmp / "blocked_dl_file"
    blocked_root.write_text("file_blocks_mkdir", encoding="utf-8")

    state_p = tmp / "mkdir_fail_state.json"
    backend = cb.ColabPetBackend(
        state_path=state_p,
        download_root=blocked_root,
        cf_config_path=isolated_env.cf_file,
    )
    with backend._lock:
        backend.state["prefs"]["auto_pull_outputs"] = True
    events: list[dict[str, Any]] = []
    backend._event_cb = lambda evt: events.append(dict(evt))

    base_url = "https://mock-mkdir-fail.trycloudflare.com"
    hist_calls = {"n": 0}
    png_bytes = _make_png_bytes()

    def _mock_http_get(url: str, params: dict[str, Any] | None = None, **_kwargs: Any):
        if url.endswith("/system_stats"):
            return _DummyComfyResp(json_data={"devices": []})
        if url.endswith("/queue"):
            return _DummyComfyResp(json_data={"queue_running": [], "queue_pending": []})
        if url.endswith("/history"):
            hist_calls["n"] += 1
            if hist_calls["n"] == 1:
                return _DummyComfyResp(json_data={})
            return _DummyComfyResp(
                json_data={
                    "pid_mkdir_1": {
                        "outputs": {
                            "9": {"images": [{"filename": "mkdir_ok.png", "subfolder": "", "type": "output"}]}
                        }
                    }
                }
            )
        if url.endswith("/view") and (params or {}).get("filename") == "mkdir_ok.png":
            return _DummyComfyResp(body=png_bytes, headers={"Last-Modified": "Wed, 18 Nov 2026 12:00:00 GMT"})
        raise AssertionError(f"意外请求: {url}")

    with (
        patch.object(backend, "_comfy_http_get", side_effect=_mock_http_get),
        patch.object(cb, "_reveal_path_in_explorer", return_value=True),
    ):
        backend.poll_comfy_http_once(base_url)
        events.clear()
        cstate = backend.poll_comfy_http_once(base_url)
        assert cstate["pending_pull_count"] == 1
        assert [m["filename"] for _, m in backend._pending_pull_items] == ["mkdir_ok.png"]
        disk_pending = json.loads(state_p.read_text(encoding="utf-8")).get("pending_pull_items") or []
        assert len(disk_pending) == 1 and disk_pending[0]["filename"] == "mkdir_ok.png"

        ev = next(e for e in events if e.get("type") == "outputs_ready")
        assert ev["pulled"] is False and ev["is_error"] is True
        assert "已存本机" not in ev["summary"]

        # 修复 download_root 后手动重试成功并清空持久化队列
        blocked_root.unlink()
        res_ok = backend.pull_or_open_latest_outputs()
        assert res_ok["ok"] is True
        assert Path(res_ok["path"]).is_file()
        assert backend.get_comfy_state()["pending_pull_count"] == 0
        assert json.loads(state_p.read_text(encoding="utf-8")).get("pending_pull_items") == []


def test_comfy_pending_pull_survives_backend_restart(isolated_env) -> None:
    """验证下载失败进入队列后即使桌宠进程重启，待重试队列也能自动恢复并完成补拉。"""
    tmp = isolated_env.tmp
    state_p = tmp / "restart_pending_state.json"
    dl_root = tmp / "restart_dl"
    base_url = "https://mock-restart.trycloudflare.com"
    png_bytes = _make_png_bytes()

    b1 = cb.ColabPetBackend(state_path=state_p, download_root=dl_root, cf_config_path=isolated_env.cf_file)
    with b1._lock:
        b1.state["prefs"]["auto_pull_outputs"] = True

    hist_calls = {"n": 0}

    def _fail_first_get(url: str, **_kwargs: Any):
        if url.endswith("/system_stats"):
            return _DummyComfyResp(json_data={"devices": []})
        if url.endswith("/queue"):
            return _DummyComfyResp(json_data={"queue_running": [], "queue_pending": []})
        if url.endswith("/history"):
            hist_calls["n"] += 1
            if hist_calls["n"] == 1:
                return _DummyComfyResp(json_data={})
            return _DummyComfyResp(
                json_data={
                    "pid_rst_1": {
                        "outputs": {"9": {"images": [{"filename": "survive.png", "subfolder": "", "type": "output"}]}}
                    }
                }
            )
        if url.endswith("/view"):
            return _DummyComfyResp(status_code=502)
        raise AssertionError(url)

    with patch.object(b1, "_comfy_http_get", side_effect=_fail_first_get):
        b1.poll_comfy_http_once(base_url)
        b1.poll_comfy_http_once(base_url)

    # 模拟进程重启：构造全新 ColabPetBackend 加载同一 state_path
    b2 = cb.ColabPetBackend(state_path=state_p, download_root=dl_root, cf_config_path=isolated_env.cf_file)
    assert b2.get_comfy_state()["pending_pull_count"] == 1
    assert [m["filename"] for _, m in b2._pending_pull_items] == ["survive.png"]

    with (
        patch.object(
            b2,
            "_comfy_http_get",
            return_value=_DummyComfyResp(body=png_bytes, headers={"Last-Modified": "Wed, 18 Nov 2026 12:00:00 GMT"}),
        ),
        patch.object(cb, "_reveal_path_in_explorer", return_value=True),
    ):
        res = b2.pull_or_open_latest_outputs()
    assert res["ok"] is True and Path(res["path"]).is_file()
    assert b2.get_comfy_state()["pending_pull_count"] == 0
    assert json.loads(state_p.read_text(encoding="utf-8")).get("pending_pull_items") == []


def test_comfy_concurrent_pull_deduplicated_via_pull_lock(isolated_env) -> None:
    """验证双线程并发调用 pull_or_open_latest_outputs 时受 _pull_lock 互斥保护，每张图仅请求 1 次。"""
    tmp = isolated_env.tmp
    state_p = tmp / "concurrent_pull_state.json"
    dl_root = tmp / "concurrent_dl"
    base_url = "https://mock-concurrent.trycloudflare.com"
    png_bytes = _make_png_bytes()

    backend = cb.ColabPetBackend(state_path=state_p, download_root=dl_root, cf_config_path=isolated_env.cf_file)
    with backend._comfy_lock:
        backend._pending_pull_items = [
            (base_url, {"filename": "conc_1.png", "subfolder": "", "type": "output"}),
            (base_url, {"filename": "conc_2.png", "subfolder": "", "type": "output"}),
        ]
        backend._comfy_state["pending_pull_count"] = 2

    view_counts: dict[str, int] = {"conc_1.png": 0, "conc_2.png": 0}
    cnt_lock = threading.Lock()

    def _slow_view_get(_url: str, params: dict[str, Any] | None = None, **_kwargs: Any):
        fn = str((params or {}).get("filename") or "")
        with cnt_lock:
            view_counts[fn] = view_counts.get(fn, 0) + 1
        time.sleep(0.05)
        return _DummyComfyResp(body=png_bytes, headers={"Last-Modified": "Wed, 18 Nov 2026 12:00:00 GMT"})

    barrier = threading.Barrier(2)
    results: list[dict[str, Any]] = []
    res_lock = threading.Lock()

    def _worker() -> None:
        barrier.wait(timeout=3.0)
        r = backend.pull_or_open_latest_outputs()
        with res_lock:
            results.append(r)

    with (
        patch.object(backend, "_comfy_http_get", side_effect=_slow_view_get),
        patch.object(cb, "_reveal_path_in_explorer", return_value=True),
    ):
        t1 = threading.Thread(target=_worker)
        t2 = threading.Thread(target=_worker)
        t1.start()
        t2.start()
        t1.join(timeout=5.0)
        t2.join(timeout=5.0)

    assert len(results) == 2 and all(r.get("ok") is True for r in results)
    assert view_counts == {"conc_1.png": 1, "conc_2.png": 1}
    disk_files = [p for p in dl_root.rglob("*") if p.is_file()]
    assert len(disk_files) == 2
    assert backend.get_comfy_state()["pending_pull_count"] == 0
    assert json.loads(state_p.read_text(encoding="utf-8")).get("pending_pull_items") == []


def test_comfy_probe_cross_origin_redirect_never_leaks_key() -> None:
    """验证 ComfyUI 隧道探针与 HTTP 封装遇 302 跨源重定向时拒绝跳转且异源收到 0 请求。"""
    import http.server
    from colab_core import colab_client

    evil_reqs: list[dict[str, Any]] = []

    class _EvilHandler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *_a: Any) -> None:
            return

        def do_GET(self) -> None:  # noqa: N802
            evil_reqs.append({"path": self.path, "headers": dict(self.headers)})
            self.send_response(200)
            self.end_headers()

    evil_srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _EvilHandler)
    evil_port = int(evil_srv.server_address[1])
    evil_t = threading.Thread(target=evil_srv.serve_forever, daemon=True)
    evil_t.start()

    class _RedirHandler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *_a: Any) -> None:
            return

        def do_GET(self) -> None:  # noqa: N802
            self.send_response(302)
            self.send_header("Location", f"http://127.0.0.1:{evil_port}/system_stats")
            self.end_headers()

    orig_srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _RedirHandler)
    orig_port = int(orig_srv.server_address[1])
    orig_t = threading.Thread(target=orig_srv.serve_forever, daemon=True)
    orig_t.start()

    try:
        with pytest.raises(Exception, match="跨源重定向"):
            colab_client.probe_comfy_endpoint_ok(
                f"http://127.0.0.1:{orig_port}",
                comfy_key="cpk-never-leak-on-redirect",
                timeout=3.0,
            )
        assert evil_reqs == []
    finally:
        orig_srv.shutdown()
        orig_srv.server_close()
        orig_t.join(timeout=2.0)
        evil_srv.shutdown()
        evil_srv.server_close()
        evil_t.join(timeout=2.0)


def test_comfy_first_request_url_mismatch_never_attaches_wrong_key(isolated_env) -> None:
    """验证 ComfyUI 密钥按账号+实例+源站绑定，目标 URL 不匹配时首次请求与补水重试验收均不带错密钥。"""
    import http.server

    seen_auth_headers: list[str | None] = []
    stub_status = {"code": 401}

    class _SpyHandler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *_a: Any) -> None:
            return

        def do_GET(self) -> None:  # noqa: N802
            seen_auth_headers.append(self.headers.get("Authorization"))
            if stub_status["code"] == 200 or self.headers.get("Authorization") == "Bearer cpk-matched-target":
                raw = json.dumps({"system": {"os": "posix"}, "devices": []}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
                return
            self.send_response(401)
            self.end_headers()

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _SpyHandler)
    port = int(srv.server_address[1])
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    target_url = f"http://127.0.0.1:{port}"

    try:
        backend = _make_backend(isolated_env, "comfy_mismatch_key")
        with backend._lock:
            backend.state["prefs"]["active_account_id"] = "acct_b"
            backend.state["active_endpoint"] = "ep-a"
            backend.state["last_assignments_count"] = 1
            backend.state["last_sessions"] = [{"name": "comfy", "endpoint": "ep-a", "accelerator": "L4"}]
            backend.state["tunnel_cache"] = {"ep-a": "https://comfy-a.trycloudflare.com"}
            backend._comfy_secrets_by_ep = {"ep-a": "cpk-secret-a-never-leak"}
            backend._comfy_key_bindings_by_ep = {"ep-a": ("acct_a", "https://comfy-a.trycloudflare.com:443")}

        def _hydrate_other(_ep: str, _sname: str, account_id: str = ""):
            _ = account_id
            backend._comfy_secrets_by_ep["ep-a"] = "cpk-secret-a-never-leak"
            return "ok", "ep-a", "https://comfy-a.trycloudflare.com"

        with patch.object(backend._tunnel_svc, "fast_read_slot_tunnel_url", side_effect=_hydrate_other):
            # 1) 不传 ep 且唯一缓存密钥绑定在其它源站 -> 首次请求与 401 补水后均绝不带错密钥
            r1 = backend._comfy_http_get(f"{target_url}/system_stats", timeout=3.0)
            assert r1.status_code == 401
            # 2) 显式传入 ep="ep-a" 但 URL 源站不匹配 -> 同样绝不携带 ep-a 密钥
            backend._last_comfy_key_hydrate_m.clear()
            r2 = backend._comfy_http_get(f"{target_url}/system_stats", base_url=target_url, ep="ep-a", timeout=3.0)
            assert r2.status_code == 401
        assert seen_auth_headers and all(h is None for h in seen_auth_headers)

        # 3) 对照组：绑定完全匹配当前账号 + 实例 + 源站时，正常携带对应 Bearer 密钥
        seen_auth_headers.clear()
        with backend._lock:
            backend._comfy_secrets_by_ep["ep-b"] = "cpk-matched-target"
            backend._comfy_key_bindings_by_ep["ep-b"] = ("acct_b", f"http://127.0.0.1:{port}")
        r_ok = backend._comfy_http_get(f"{target_url}/system_stats", base_url=target_url, ep="ep-b", timeout=3.0)
        assert r_ok.status_code == 200
        assert seen_auth_headers == ["Bearer cpk-matched-target"]
    finally:
        srv.shutdown()
        srv.server_close()
        t.join(timeout=2.0)


def test_comfy_download_validates_complete_200_body_and_large_queue_survives_restart(
    isolated_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """验证非 200/空包/HTML/截断下载响应必判失败，以及 >256 积压队列重启零丢失与容量一致。"""
    from backend_comfy import ComfySyncService

    tmp = isolated_env.tmp
    state_p = tmp / "queue_and_resp_state.json"
    dl_root = tmp / "queue_dl"
    base_url = "https://mock-qcheck.trycloudflare.com"
    backend = cb.ColabPetBackend(state_path=state_p, download_root=dl_root, cf_config_path=isolated_env.cf_file)

    bad_cases = {
        "r_302.png": _DummyComfyResp(status_code=302, body=_make_png_bytes()),
        "r_204.png": _DummyComfyResp(status_code=204, body=b""),
        "r_empty.png": _DummyComfyResp(status_code=200, body=b""),
        "r_trunc.png": _DummyComfyResp(status_code=200, body=b"abc", headers={"Content-Length": "120"}),
        "r_html.png": _DummyComfyResp(status_code=200, body=b"<html>error</html>", headers={"Content-Type": "text/html; charset=utf-8"}),
    }
    bad_items = [(base_url, {"filename": fn, "subfolder": "", "type": "output"}) for fn in bad_cases]

    with patch.object(
        backend,
        "_comfy_http_get",
        side_effect=lambda _u, params=None, **_kw: bad_cases[str((params or {}).get("filename") or "")],
    ):
        saved, failed = backend._comfy_svc.download_comfy_outputs_detailed(bad_items)
    assert saved == [] and len(failed) == 5
    assert not any(dl_root.rglob("*"))

    # 300 项 (>256) 积压重启后必须全部恢复，尾部绝不截断丢失
    items_300 = [(base_url, {"filename": f"q_{i:03d}.png", "subfolder": "", "type": "output"}) for i in range(300)]
    with backend._comfy_lock:
        backend._pending_pull_items = items_300
    backend._comfy_svc._persist_pending_pull_items()

    restarted = cb.ColabPetBackend(state_path=state_p, download_root=dl_root, cf_config_path=isolated_env.cf_file)
    assert len(restarted._pending_pull_items) == 300
    assert restarted.get_comfy_state()["pending_pull_count"] == 300
    assert restarted._pending_pull_items[-1][1]["filename"] == "q_299.png"

    # 当队列超出统一上限时，内存与持久化一致保留最新尾部条目
    monkeypatch.setattr(ComfySyncService, "MAX_PENDING_PULL_ITEMS", 5)
    cap_items = [(base_url, {"filename": f"cap_{i}.png", "subfolder": "", "type": "output"}) for i in range(7)]
    merged = ComfySyncService._merge_pending_items([], cap_items)
    assert [m["filename"] for _, m in merged] == [f"cap_{i}.png" for i in range(2, 7)]
    with restarted._comfy_lock:
        restarted._pending_pull_items = cap_items
    restarted._comfy_svc._persist_pending_pull_items()
    disk_cap = json.loads(state_p.read_text(encoding="utf-8")).get("pending_pull_items") or []
    assert [it["filename"] for it in disk_cap] == [f"cap_{i}.png" for i in range(2, 7)]
