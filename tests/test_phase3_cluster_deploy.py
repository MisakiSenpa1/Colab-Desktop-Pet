"""Phase 3 · Wave 3B（路 2）`backend_cluster.py` 与 `backend_deploy.py` 容器化重构与热函数拆解回归测试。

严格离线隔离执行：所有用例通过 `COLAB_PET_MOCK_VAULT_PATH`、`COLAB_PET_CF_CONFIG_PATH`、
`COLAB_PET_DIAG_LOG_PATH` 与 `tmp_path` 状态路径隔离，绝不读写宿主机真实凭据或运行态文件。
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

PACKAGE_DIR = Path(__file__).resolve().parent.parent
PET_DIR = PACKAGE_DIR / "colab_pet"
for _p in (str(PACKAGE_DIR), str(PET_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import colab_backend as cb  # noqa: E402
from colab_core import pool_scheduler as ps  # noqa: E402


@pytest.fixture()
def isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """为每个测试用例构造完全隔离的 mock vault、CF 配置、诊断日志与假工具路径。"""
    vault_file = tmp_path / "mock_vault.json"
    cf_file = tmp_path / "cf_tunnel_secret.json"
    diag_file = tmp_path / "pet_diag.log"
    fake_py = tmp_path / "fake_colab_python.exe"
    fake_py.write_text("", encoding="utf-8")
    monkeypatch.setenv("COLAB_PET_MOCK_VAULT_PATH", str(vault_file))
    monkeypatch.setenv("COLAB_PET_CF_CONFIG_PATH", str(cf_file))
    monkeypatch.setenv("COLAB_PET_DIAG_LOG_PATH", str(diag_file))
    monkeypatch.setattr(cb, "COLAB_PY", fake_py)
    return SimpleNamespace(
        tmp=tmp_path,
        vault_file=vault_file,
        cf_file=cf_file,
        diag_file=diag_file,
        fake_py=fake_py,
    )


def _make_backend(env: SimpleNamespace, name: str) -> cb.ColabPetBackend:
    return cb.ColabPetBackend(
        state_path=env.tmp / f"{name}_state.json",
        download_root=env.tmp / f"{name}_dl",
        cf_config_path=env.cf_file,
    )


def _calc_ast_cc(node: ast.AST) -> int:
    """计算单个函数/方法 AST 节点的圈复杂度。"""
    cc = 1
    for child in ast.walk(node):
        if isinstance(
            child,
            (
                ast.If,
                ast.For,
                ast.AsyncFor,
                ast.While,
                ast.ExceptHandler,
                ast.With,
                ast.AsyncWith,
                ast.IfExp,
                ast.Assert,
            ),
        ):
            cc += 1
        elif isinstance(child, ast.BoolOp):
            cc += len(child.values) - 1
        elif isinstance(child, ast.comprehension):
            cc += 1 + len(child.ifs)
    return cc


# =====================================================================
# 1. 静态门禁：零 _cb_mod、禁导依赖、容器绑定、双导入与 CC 上限
# =====================================================================


def test_cluster_and_deploy_zero_cb_mod_and_ast_cc_ceiling(isolated_env: SimpleNamespace) -> None:
    """验证两文件彻底清除 `_cb_mod()`、无禁导模块、所有方法 CC <= 30 且容器绑定同对象。"""
    forbidden = {"PIL", "aiohttp", "pywin32", "win32api", "win32con", "win32gui"}
    for fname in ("backend_cluster.py", "backend_deploy.py"):
        fpath = PET_DIR / fname
        src = fpath.read_text(encoding="utf-8")
        assert len(re.findall(r"_cb_mod\s*\(", src)) == 0, f"{fname} 仍残留 _cb_mod() 调用"
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".")[0] not in forbidden, f"{fname} 违规导入 {alias.name}"
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".")[0] not in forbidden, f"{fname} 违规导入 {node.module}"
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                cc = _calc_ast_cc(node)
                assert cc <= 30, f"{fname}:{node.name} 圈复杂度 {cc} 超过上限 30"

    backend = _make_backend(isolated_env, "bind_check")
    for svc in (backend._cluster_svc, backend._deploy_svc):
        assert svc._b is backend
        assert svc._store is backend._store
        assert svc._secrets is backend._secrets
        assert svc._bus is backend._bus
        assert svc._ctx is backend._ctx


def test_dual_import_modes_cluster_and_deploy() -> None:
    """验证平铺 `sys.path` 与包导入双形态加载 `backend_cluster` / `backend_deploy` 均无循环依赖。"""
    flat_script = (
        "import sys\n"
        f"sys.path.insert(0, {str(PET_DIR)!r})\n"
        f"sys.path.insert(0, {str(PACKAGE_DIR)!r})\n"
        "import backend_cluster, backend_deploy\n"
        "assert hasattr(backend_cluster, 'ClusterPoolService')\n"
        "assert hasattr(backend_deploy, 'DeployOrchestrator')\n"
        "print('FLAT_OK')\n"
    )
    pkg_script = (
        "import sys\n"
        f"sys.path.insert(0, {str(PACKAGE_DIR)!r})\n"
        "import colab_pet.backend_cluster as bc\n"
        "import colab_pet.backend_deploy as bd\n"
        "assert hasattr(bc, 'ClusterPoolService')\n"
        "assert hasattr(bd, 'DeployOrchestrator')\n"
        "print('PKG_OK')\n"
    )
    for script, token in ((flat_script, "FLAT_OK"), (pkg_script, "PKG_OK")):
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=25,
            check=True,
        )
        assert token in proc.stdout


# =====================================================================
# 2. DeployOrchestrator 参数解析、定向停卡、热重载与容灾换号回归
# =====================================================================


def test_deploy_validate_parse_and_onboarding_guards(
    isolated_env: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证开机参数解析默认卡型规则及引导态/预检拦截图层顺序。"""
    backend = _make_backend(isolated_env, "deploy_parse")
    deploy = backend._deploy_svc

    p_single = deploy._parse_boot_profiles_and_gpu({"profile": "anima", "gpu": "UNKNOWN_GPU", "high_mem": True})
    assert p_single["gpu"] == "L4"
    assert p_single["high_mem"] is False
    assert p_single["is_dual"] is False

    p_dual = deploy._parse_boot_profiles_and_gpu({"profiles": ["minimax", "qwen_27b"]})
    assert p_dual["gpu"] == "G4"
    assert p_dual["is_dual"] is True
    assert p_dual["exec_order"] == ["qwen_27b", "minimax"]

    monkeypatch.setattr(cb, "_ctl_available", lambda: True)
    with patch.object(
        backend,
        "_get_onboarding_state",
        return_value={"colab_cli_ok": True, "rclone_exe_ok": True, "auth_missing": True, "gdrive_ok": True},
    ):
        parsed, err = deploy._validate_and_parse_boot_request({"profile": "anima"})
        assert parsed is None and err is not None
        assert err["needs_onboarding"] is True and err["target_tab"] == "vault"


def test_deploy_stop_session_isolation_and_still_running_guard(
    isolated_env: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证精准停卡仅释放目标端点（绝不传 --all）、停后仍存活必报 still_running、切号中止写回。"""
    backend = _make_backend(isolated_env, "stop_guard")
    with backend._store.lock:
        backend._store.state["prefs"]["active_account_id"] = "acct_a"
        backend._store.state["last_assignments_count"] = 2
        backend._store.state["last_sessions"] = [
            {"name": "comfy", "endpoint": "ep-comfy-1", "accelerator": "L4"},
            {"name": "other", "endpoint": "ep-other-2", "accelerator": "T4"},
        ]
        backend._store.state["managed_binding"] = {
            "account_id": "acct_a",
            "session": "comfy",
            "endpoint": "ep-comfy-1",
        }
    backend._secrets.set_comfy_secret("ep-comfy-1", "cpk-1", account_id="acct_a", origin="https://c1.trycloudflare.com")

    # 账号不匹配时必须直接中止
    abort_res = backend.stop_session_instance("comfy", account_id="acct_old")
    assert abort_res["ok"] is False and abort_res["aborted_by_switch"] is True

    seen_cmds: list[list[str]] = []

    def _fake_run(cmd, **_kw):
        seen_cmds.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, stdout="stopped", stderr="")

    monkeypatch.setattr(cb, "_ctl_available", lambda: True)
    monkeypatch.setattr(cb, "_run_hidden", _fake_run)

    # 场景 1：停后核实实例仍在计费 -> 必返 still_running=True
    with patch.object(
        backend,
        "fetch_live_status",
        return_value={"ok": True, "assignments_count": 2, "sessions": [{"name": "comfy", "endpoint": "ep-comfy-1"}]},
    ):
        r_still = backend.stop_session_instance("comfy")
    assert r_still["ok"] is False and r_still["still_running"] is True
    assert "--all" not in seen_cmds[-1]
    assert "--endpoint" in seen_cmds[-1] and "ep-comfy-1" in seen_cmds[-1]

    # 场景 2：停后仅剩另一会话 ep-other-2 -> 精准清理 ep-comfy-1 凭据且保留其余会话不强制全离线
    reset_calls: list[bool] = []
    with (
        patch.object(
            backend,
            "fetch_live_status",
            return_value={"ok": True, "assignments_count": 1, "sessions": [{"name": "other", "endpoint": "ep-other-2"}]},
        ),
        patch.object(backend, "_reset_comfy_offline", side_effect=lambda: reset_calls.append(True)),
    ):
        r_ok = backend.stop_session_instance("comfy")
    assert r_ok["ok"] is True
    assert r_ok["released_endpoint"] == "ep-comfy-1"
    assert r_ok["remaining_count"] == 1
    assert "ep-comfy-1" not in backend._secrets.comfy_secrets_by_ep
    assert backend._store.state["managed_binding"] is None
    assert reset_calls == []


def test_deploy_reconfigure_running_llm_cohost_and_cluster_slot(isolated_env: SimpleNamespace) -> None:
    """验证同机双栖与异构集群 LLM 槽位两种形态下的热重载参数装配与边界裁剪。"""
    backend = _make_backend(isolated_env, "reconf_llm")
    captured_cfgs: list[dict] = []

    with backend._store.lock:
        backend._store.state["prefs"]["boot_profile"] = "minimax"
        backend._store.state["prefs"]["boot_profiles"] = ["minimax", "qwen_27b"]
        backend._store.state["last_assignments_count"] = 1
        backend._store.state["last_sessions"] = [{"name": "comfy", "endpoint": "ep-dual-1", "accelerator": "G4"}]
    backend._secrets.set_ep_meta(
        "ep-dual-1",
        {"kind": "dual", "llm_profile": "qwen_27b", "comfy_profiles": ["minimax"]},
    )
    with backend._comfy_lock:
        backend._comfy_state["service_kind"] = "dual"
        backend._comfy_state["active_profiles"] = ["minimax", "qwen_27b"]
        backend._comfy_state["comfy_online"] = True
        backend._comfy_state["llm_online"] = True

    with patch.object(backend, "start_instance", side_effect=lambda c: captured_cfgs.append(dict(c)) or {"ok": True}):
        r_cohost = backend.reconfigure_running_llm({"ctx": 999999, "quant": "INVALID_QUANT", "mmproj": False})
    assert r_cohost["ok"] is True
    cfg1 = captured_cfgs[-1]
    assert cfg1["_cohost_only_llm"] is True
    assert cfg1["_preserved_profiles"] == ["minimax", "qwen_27b"]
    assert cfg1["llm27_ctx"] == 262144
    assert cfg1["llm27_quant"] == "UD-Q4_K_XL"
    assert cfg1["llm27_mmproj"] is False

    # 切换至异构集群分槽形态
    with backend._store.lock:
        backend._store.state["cluster_slots"] = {
            "llm": {"account_id": "acct_l4", "endpoint": "ep-l4-1", "gpu": "L4", "profiles": ["qwen_27b"]},
            "comfy": {"account_id": "acct_g4", "endpoint": "ep-g4-1", "gpu": "G4", "profiles": ["drama3"]},
        }
    with patch.object(backend, "start_instance", side_effect=lambda c: captured_cfgs.append(dict(c)) or {"ok": True}):
        r_slot = backend.reconfigure_running_llm({"ctx": 131072, "quant": "UD-Q4_K_XL", "mmproj": True})
    assert r_slot["ok"] is True
    cfg2 = captured_cfgs[-1]
    assert cfg2["_cohost_only_llm"] is False
    assert cfg2["_cluster_slot_reconfigure"] == "llm"
    assert cfg2["_target_account_id"] == "acct_l4"
    assert cfg2["_target_endpoint"] == "ep-l4-1"
    assert cfg2["_preserved_profiles"] == ["drama3", "qwen_27b"]


def test_deploy_failover_switches_account_on_stockout(
    isolated_env: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证单机开机遇缺卡时 `_try_failover_other_account` 自动标记紧缺卡并切换备号重建命令。"""
    backend = _make_backend(isolated_env, "deploy_failover")
    with backend._store.lock:
        backend._store.state["account_profiles"] = [
            {"id": "acct_1", "label": "主号"},
            {"id": "acct_2", "label": "备号"},
        ]
        backend._store.state["prefs"]["active_account_id"] = "acct_1"
        backend._store.state["prefs"]["auto_pool_schedule"] = True
        backend._store.state["account_ledgers"] = {
            "acct_1": {"last_balance": 100.0, "last_eligible_gpus": ["G4", "L4"], "last_updated_ts": 100},
            "acct_2": {"last_balance": 180.0, "last_eligible_gpus": ["G4", "L4"], "last_updated_ts": 100},
        }

    monkeypatch.setattr(cb, "_ctl_available", lambda: True)
    monkeypatch.setattr(cb, "_run_hidden", lambda cmd, **_kw: subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr=""))

    parsed = backend._deploy_svc._parse_boot_profiles_and_gpu({"profile": "minimax", "gpu": "G4"})
    dctx = backend._deploy_svc._prepare_deploy_ctx({"profile": "minimax", "gpu": "G4"}, parsed)
    dctx.account_id = "acct_1"

    switched, new_cmd, new_acct = backend._deploy_svc._try_failover_other_account(
        dctx,
        ["RuntimeError: Cannot connect to GPU backend due to usage limits / G4 unavailable"],
        "acct_1",
        0,
    )
    assert switched is True
    assert new_acct == "acct_2"
    assert "--account" in new_cmd and "acct_2" in new_cmd
    assert ("acct_1", "G4") in backend._cluster_svc._acct_scarce_gpus
    assert backend._store.get_active_account_id() == "acct_2"


# =====================================================================
# 3. ClusterPoolService 隔离账本、双号并行部分失败保留、分槽停卡与选择性闲置
# =====================================================================


def test_cluster_apply_pool_probe_isolates_main_ledger_and_marks_invalid_binding(
    isolated_env: SimpleNamespace,
) -> None:
    """验证 `apply_pool_probe_results` 绝不污染主账本顶级字段，并在绑定实例消失时置 `binding_invalid`。"""
    backend = _make_backend(isolated_env, "cluster_probe")
    with backend._store.lock:
        backend._store.state["prefs"]["active_account_id"] = "acct_main"
        backend._store.state["last_balance"] = 321.5
        backend._store.state["day_consumed_cu"] = 9.5
        backend._store.state["account_profiles"] = [
            {"id": "acct_main", "label": "主号"},
            {"id": "acct_aux", "label": "副号"},
        ]
        backend._store.state["cluster_slots"] = {
            "comfy": {"account_id": "acct_aux", "endpoint": "ep-comfy-old", "gpu": "G4", "online": True}
        }

    cnt = backend._cluster_svc.apply_pool_probe_results(
        {
            "acct_aux": {
                "auth_ok": True,
                "balance": 88.8,
                "rate_hourly": 1.94,
                "eligible_gpus": ["l4", "g4"],
                "sessions": [{"name": "other", "endpoint": "ep-unowned-999", "accelerator": "L4"}],
            }
        }
    )
    assert cnt == 1
    # 主账本顶级字段必须零污染
    assert backend._store.state["last_balance"] == 321.5
    assert backend._store.state["day_consumed_cu"] == 9.5
    # 隔离账本已精确更新
    aux_ledger = backend._store.state["account_ledgers"]["acct_aux"]
    assert aux_ledger["last_balance"] == 88.8
    assert aux_ledger["last_eligible_gpus"] == ["L4", "G4"]
    # 原绑定 ep-comfy-old 消失 -> 标记 binding_invalid 且绝不自动认领 ep-unowned-999
    c_slot = backend._cluster_svc.get_slot("comfy")
    assert c_slot is not None
    assert c_slot["endpoint"] == "ep-comfy-old"
    assert c_slot["binding_invalid"] is True
    assert c_slot["online"] is False


def test_cluster_snapshot_and_onboarding_enrichment(isolated_env: SimpleNamespace) -> None:
    """验证集群快照正确汇总多号池总余额、健康付费号数、双开可行性与账号集群角色。"""
    backend = _make_backend(isolated_env, "cluster_snap")
    with backend._store.lock:
        backend._store.state["prefs"]["active_account_id"] = "acct_1"
        backend._store.state["account_profiles"] = [
            {"id": "acct_1", "label": "一号"},
            {"id": "acct_2", "label": "二号"},
        ]
        backend._store.state["cluster_slots"] = {
            "llm": {"account_id": "acct_1", "endpoint": "ep-l1", "gpu": "L4", "online": True, "rate_hourly": 1.94},
            "comfy": {"account_id": "acct_2", "endpoint": "ep-c2", "gpu": "G4", "online": True, "rate_hourly": 10.05},
        }
    backend._cluster_svc.apply_pool_probe_results(
        {
            "acct_1": {"auth_ok": True, "balance": 120.0, "rate_hourly": 1.94, "eligible_gpus": ["L4"]},
            "acct_2": {"auth_ok": True, "balance": 240.0, "rate_hourly": 10.05, "eligible_gpus": ["G4", "L4"]},
        }
    )

    ob_in = {"accounts": [{"id": "acct_1", "label": "一号"}, {"id": "acct_2", "label": "二号"}]}
    c_info, ob_out = backend._cluster_svc.build_cluster_snapshot_and_enrich_onboarding(
        ob_in,
        current_rate_hourly=1.94,
        active_account_id="acct_1",
        comfy_state={"llm_online": True, "comfy_online": True},
    )
    assert c_info["active"] is True
    assert c_info["can_split"] is True
    assert c_info["pool_healthy_count"] == 2
    assert c_info["total_pool_balance"] == 360.0
    assert c_info["total_rate_hourly"] == 11.99
    roles = {a["id"]: a["cluster_role"] for a in ob_out["accounts"]}
    assert roles == {"acct_1": "llm", "acct_2": "comfy"}


def test_cluster_start_split_partial_failure_retains_healthy_slot(
    isolated_env: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证双号异构并行部署单槽失败时：清理失败槽位、登记稀缺卡，同时完整保留已成功槽位。"""
    backend = _make_backend(isolated_env, "split_partial")
    with backend._store.lock:
        backend._store.state["account_profiles"] = [
            {"id": "acct_llm", "label": "LLM号"},
            {"id": "acct_comfy", "label": "画布号"},
        ]
        backend._store.state["prefs"]["active_account_id"] = "acct_llm"

    plan = ps.SplitClusterPlan(
        ok=True,
        llm_slot=ps.SplitSlotSpec("llm", "acct_llm", "LLM号", ("qwen_27b",), "L4", False, "", False),
        comfy_slot=ps.SplitSlotSpec("comfy", "acct_comfy", "画布号", ("minimax",), "G4", False, "", True),
        reason="ok",
    )
    parsed = backend._deploy_svc._parse_boot_profiles_and_gpu({"profiles": ["minimax", "qwen_27b"]})
    done_evt = threading.Event()
    events: list[dict] = []

    def _on_evt(evt: dict) -> None:
        events.append(evt)
        if evt.get("type") in ("deploy_done", "deploy_error"):
            done_evt.set()

    backend.subscribe_events(_on_evt)
    monkeypatch.setattr(cb, "_ctl_available", lambda: True)

    def _fake_stream(cmd, on_line=None, **_kw):
        if "acct_llm" in cmd:
            if on_line:
                on_line("LLM_TUNNEL_URL=https://llm-ok.trycloudflare.com")
            return SimpleNamespace(ok=True, returncode=0, cancelled=False, error="")
        if on_line:
            on_line("Cannot connect to GPU backend due to usage limits")
        return SimpleNamespace(ok=False, returncode=1, cancelled=False, error="G4 unavailable")

    with (
        patch.object(backend, "run_preflight", return_value={"ok": True}),
        patch.object(backend._runner, "run_streaming", side_effect=_fake_stream),
        patch.object(
            backend._tunnel_svc,
            "fast_read_slot_tunnel_url",
            return_value=("ok", "ep-llm-ok", "https://llm-ok.trycloudflare.com"),
        ),
        patch.object(backend._cluster_svc, "probe_accounts_raw", return_value={"ok": False, "accounts": {}}),
        patch.object(backend, "fetch_live_status", return_value={"ok": True, "sessions": []}),
    ):
        res = backend._cluster_svc.start_split_cluster(plan, {"profiles": ["minimax", "qwen_27b"]}, parsed)
        assert res["ok"] is True and res["cluster_split"] is True
        assert done_evt.wait(timeout=8.0)

    rem_slots = backend._cluster_svc.get_cluster_slots()
    assert "llm" in rem_slots and rem_slots["llm"]["endpoint"] == "ep-llm-ok"
    assert "comfy" not in rem_slots
    assert ("acct_comfy", "G4") in backend._cluster_svc._acct_scarce_gpus
    err_evts = [e for e in events if e.get("type") == "deploy_error"]
    assert err_evts and "已保留LLM槽位" in err_evts[-1]["summary"]


def test_cluster_stop_slot_precheck_refusal_and_reconcile_downgrade(
    isolated_env: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 `stop_cluster_slot` 前置探针拒盲停，以及正常释放画布后平滑降级为单 LLM 槽位。"""
    backend = _make_backend(isolated_env, "stop_slot")
    with backend._store.lock:
        backend._store.state["prefs"]["active_account_id"] = "acct_llm"
        backend._store.state["cluster_slots"] = {
            "llm": {
                "account_id": "acct_llm",
                "endpoint": "ep-llm-1",
                "gpu": "L4",
                "tunnel_url": "https://llm1.trycloudflare.com",
                "profiles": ["qwen_27b"],
                "online": True,
            },
            "comfy": {
                "account_id": "acct_comfy",
                "endpoint": "ep-comfy-1",
                "gpu": "G4",
                "tunnel_url": "https://comfy1.trycloudflare.com",
                "profiles": ["minimax"],
                "online": True,
            },
        }
    monkeypatch.setattr(cb, "_ctl_available", lambda: True)

    down_calls: list[list[str]] = []
    monkeypatch.setattr(
        cb,
        "_run_hidden",
        lambda cmd, **_kw: (
            down_calls.append(list(cmd))
            or subprocess.CompletedProcess(
                cmd, 0, stdout='<<<STOP_VERIFIED>>>{"remaining": []}<<<END_STOP_VERIFIED>>>', stderr=""
            )
        ),
    )

    # 1) 前置探针发现 ep-comfy-1 已消失 -> 拒绝盲停且零 down 调用
    with patch.object(
        backend._cluster_svc,
        "probe_accounts_raw",
        return_value={"ok": True, "accounts": {"acct_comfy": {"auth_ok": True, "sessions": []}}},
    ):
        r_refuse = backend.stop_cluster_slot("comfy")
    assert r_refuse["ok"] is False
    assert r_refuse["binding_invalid"] is True and r_refuse["target_unresolved"] is True
    assert down_calls == []

    # 2) 恢复合法绑定：前置探针确认 ep-comfy-1 在线，停后探针确认已释放 -> 降级保留 LLM 槽位
    with backend._store.lock:
        backend._store.state["cluster_slots"]["comfy"]["binding_invalid"] = False

    probe_seq = iter(
        [
            {"ok": True, "accounts": {"acct_comfy": {"auth_ok": True, "sessions": [{"endpoint": "ep-comfy-1"}]}}},
            {"ok": True, "accounts": {"acct_comfy": {"auth_ok": True, "sessions": []}}},
        ]
    )
    with patch.object(backend._cluster_svc, "probe_accounts_raw", side_effect=lambda _aids: next(probe_seq)):
        r_ok = backend.stop_cluster_slot("comfy")
    assert r_ok["ok"] is True and r_ok["released"] == ["comfy"]
    assert len(down_calls) == 1
    assert set(backend._cluster_svc.get_cluster_slots().keys()) == {"llm"}
    assert backend.get_comfy_state()["service_kind"] == "llm"
    assert backend.get_comfy_state()["comfy_online"] is False


def test_cluster_idle_auto_stop_keep_alive_policy_and_warning(isolated_env: SimpleNamespace) -> None:
    """验证集群闲置预警与 `keep_alive` 保图留机策略在归档失败时不释放画布槽位。"""
    backend = _make_backend(isolated_env, "cluster_idle")
    with backend._store.lock:
        backend._store.state["prefs"]["idle_auto_stop_minutes"] = 60
        backend._store.state["prefs"]["idle_archive_fail_policy"] = "keep_alive"
        backend._store.state["cluster_slots"] = {
            "comfy": {"account_id": "acct_c", "endpoint": "ep-c", "gpu": "G4", "online": True},
            "llm": {"account_id": "acct_l", "endpoint": "ep-l", "gpu": "L4", "online": True},
        }
    with backend._comfy_lock:
        backend._comfy_state["online"] = True
        backend._comfy_state["comfy_online"] = True
        backend._comfy_state["llm_online"] = True
        backend._comfy_state["llm_running_count"] = 1

    events: list[dict] = []
    archive_failed_evt = threading.Event()

    def _on_evt(evt: dict) -> None:
        events.append(evt)
        if evt.get("type") == "idle_archive_failed":
            archive_failed_evt.set()

    backend.subscribe_events(_on_evt)

    # 1) 闲置 59 分钟（进入提前 2 分钟预警窗口） -> 触发 idle_warning
    backend._cluster_svc._slot_last_activity_m["comfy"] = 1000.0
    backend._cluster_svc._slot_last_activity_m["llm"] = 4500.0
    backend._cluster_svc.maybe_trigger_cluster_idle_auto_stop(1000.0 + 59 * 60.0)
    assert any(e.get("type") == "idle_warning" for e in events)

    # 2) 闲置满 61 分钟，且归档同步失败 + keep_alive -> 发出 idle_archive_failed，严禁调用 stop_cluster_slot
    stop_calls: list[str] = []
    with (
        patch.object(backend._cluster_svc, "_run_slot_archive_sync", return_value=(False, "gdrive timeout")),
        patch.object(backend._cluster_svc, "stop_cluster_slot", side_effect=lambda s: stop_calls.append(s) or {"ok": True}),
    ):
        backend._cluster_svc.maybe_trigger_cluster_idle_auto_stop(1000.0 + 61 * 60.0)
        assert archive_failed_evt.wait(timeout=5.0)
    assert stop_calls == []
    assert any(e.get("type") == "idle_archive_failed" for e in events)
