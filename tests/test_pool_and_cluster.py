"""智能算力池纯函数调度、账号作用域隔离、多号异构双开集群与 PyWebView 真窗自动化测试套件。

说明：全量基于隔离临时沙箱目录与环境变量（COLAB_PET_MOCK_VAULT_PATH、
COLAB_PET_CF_CONFIG_PATH、COLAB_PET_DIAG_LOG_PATH），绝不读取或修改宿主机真实凭据库、
`.cf_tunnel_secret.json` 或 `~/.config/colab-cli/sessions.json`。
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

PKG_DIR = Path(__file__).resolve().parents[1]
PET_DIR = PKG_DIR / "colab_pet"
for _p in (str(PKG_DIR), str(PET_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import build_windows_package as win_pkg  # noqa: E402
import colab_backend as cb  # noqa: E402
import colab_ctl  # noqa: E402
import colab_security as csec  # noqa: E402
from colab_core import pool_scheduler  # noqa: E402
from colab_core.runtime import (  # noqa: E402
    CommandResult,
    consume_deploy_line,
    default_deploy_state,
    format_pool_tier_summary,
    infer_account_tier_and_baseline,
)


@pytest.fixture()
def isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """为每条测试用例构建完全隔离的凭据保险箱、隧道配置与诊断日志沙箱。"""
    csec.clear_ephemeral_secrets()
    vault_file = tmp_path / "mock_vault.json"
    cf_file = tmp_path / ".cf_tunnel_secret.json"
    diag_log = tmp_path / "logs" / "colab_diag.log"
    monkeypatch.setenv("COLAB_PET_MOCK_VAULT_PATH", str(vault_file))
    monkeypatch.setenv("COLAB_PET_CF_CONFIG_PATH", str(cf_file))
    monkeypatch.setenv("COLAB_PET_DIAG_LOG_PATH", str(diag_log))
    monkeypatch.delenv("COLAB_PET_ACCOUNT_ID", raising=False)
    yield SimpleNamespace(
        tmp=tmp_path,
        vault_file=vault_file,
        cf_file=cf_file,
        diag_log=diag_log,
    )
    csec.clear_ephemeral_secrets()


def _prepare_fake_tools(tmp: Path) -> SimpleNamespace:
    """在沙箱目录生成模拟 CLI 可执行占位文件与含 gdrive_save 的 rclone 配置。"""
    fake_exe = tmp / "colab.exe"
    fake_py = tmp / "colab_py.exe"
    fake_rclone = tmp / "rclone.exe"
    fake_conf = tmp / "rclone.conf"
    for p in (fake_exe, fake_py, fake_rclone):
        p.write_bytes(b"MZ")
    fake_conf.write_text("[gdrive_save]\ntype = drive\n", encoding="utf-8")
    return SimpleNamespace(
        colab_exe=fake_exe,
        colab_py=fake_py,
        rclone_exe=fake_rclone,
        rclone_conf=fake_conf,
    )


def _make_backend(env_ns: SimpleNamespace, name: str = "cluster_state") -> cb.ColabPetBackend:
    """实例化绑定到隔离沙箱路径的 ColabPetBackend。"""
    return cb.ColabPetBackend(
        state_path=env_ns.tmp / f"{name}.json",
        download_root=env_ns.tmp / f"{name}_dl",
        cf_config_path=env_ns.cf_file,
    )


# =========================================================================
# 1. 纯函数调度核（pool_scheduler）：过滤、优先级、紧缺冷却与非对称异构双号规划
# =========================================================================


def test_pool_scheduler_pure_selection_and_split_plan() -> None:
    """验证纯函数调度核对免费/未授权/仅T4/紧缺节点的过滤排序，以及双号异构非对称拆分。"""
    nodes = [
        pool_scheduler.AccountNodeSnapshot(
            account_id="free_zero",
            label="零余额免费号",
            balance=0.0,
            eligible_gpus=("T4",),
            auth_ok=True,
        ),
        pool_scheduler.AccountNodeSnapshot(
            account_id="unauth_acct",
            label="过期未授权号",
            balance=120.0,
            eligible_gpus=("G4", "A100", "L4", "T4"),
            auth_ok=False,
        ),
        pool_scheduler.AccountNodeSnapshot(
            account_id="t4_only_acct",
            label="受限仅T4号",
            balance=85.0,
            eligible_gpus=("T4",),
            auth_ok=True,
        ),
        pool_scheduler.AccountNodeSnapshot(
            account_id="scarce_g4_acct",
            label="G4冷却号",
            balance=300.0,
            eligible_gpus=("G4", "L4", "T4"),
            scarce_gpus=("G4",),
            auth_ok=True,
        ),
        pool_scheduler.AccountNodeSnapshot(
            account_id="l4_member_acct",
            label="L4副号",
            balance=64.0,
            eligible_gpus=("L4", "T4"),
            auth_ok=True,
        ),
        pool_scheduler.AccountNodeSnapshot(
            account_id="g4_ultra_acct",
            label="Ultra主力号",
            balance=168.0,
            eligible_gpus=("G4", "A100", "L4", "T4"),
            auth_ok=True,
        ),
    ]

    # 1a. 请求 G4：自动过滤 0 CU、未授权、仅 T4 与无 G4 权限号，并避开 scarce_gpus 冷却号命中 g4_ultra_acct
    dec_g4 = pool_scheduler.select_best_account(
        nodes,
        profiles=["minimax"],
        preferred_gpu="G4",
        active_account_id="l4_member_acct",
    )
    assert dec_g4.ok is True
    assert dec_g4.account_id == "g4_ultra_acct"
    assert dec_g4.switched is True
    assert "free_zero" not in dec_g4.ranked_account_ids
    assert "unauth_acct" not in dec_g4.ranked_account_ids
    assert "t4_only_acct" not in dec_g4.ranked_account_ids

    # 1b. 同规格已开机实例复用优先级最高（高于更高余额空置号与当前激活号）
    nodes_with_running = [
        *nodes,
        pool_scheduler.AccountNodeSnapshot(
            account_id="reusable_g4_acct",
            label="已跑G4号",
            balance=25.0,
            eligible_gpus=("G4", "L4"),
            sessions=({"name": "comfy", "endpoint": "m-s-reuse-g4", "accelerator": "G4"},),
            auth_ok=True,
        ),
    ]
    dec_reuse = pool_scheduler.select_best_account(
        nodes_with_running,
        profiles=["minimax"],
        preferred_gpu="G4",
        active_account_id="g4_ultra_acct",
    )
    assert dec_reuse.ok is True
    assert dec_reuse.account_id == "reusable_g4_acct"
    assert dec_reuse.reuse_endpoint == "m-s-reuse-g4"

    # 1c. 本号无阻塞优先：当前激活号 l4_member_acct 已满足 L4 需求时免切号
    dec_stay = pool_scheduler.select_best_account(
        nodes,
        profiles=["qwen_27b"],
        preferred_gpu="L4",
        active_account_id="l4_member_acct",
    )
    assert dec_stay.ok is True
    assert dec_stay.account_id == "l4_member_acct"
    assert dec_stay.switched is False

    # 1d. 非对称卡池下的双号异构规划（MiniMax + Qwen27B）：
    #     scarce_g4_acct 的 G4 处于冷却，仅 g4_ultra_acct 可用 G4，l4_member_acct 可用 L4 ->
    #     必须输出账号互异的 LLM(L4, no_named_tunnel=False) + Comfy(G4, no_named_tunnel=True)
    split_nodes = [n for n in nodes if n.account_id != "scarce_g4_acct"]
    plan = pool_scheduler.plan_split_cluster(
        split_nodes,
        profiles=["minimax", "qwen_27b"],
        active_account_id="g4_ultra_acct",
    )
    assert plan.ok is True
    assert plan.llm_slot is not None and plan.comfy_slot is not None
    assert plan.llm_slot.account_id != plan.comfy_slot.account_id
    assert plan.llm_slot.account_id == "l4_member_acct"
    assert plan.llm_slot.gpu == "L4"
    assert plan.llm_slot.no_named_tunnel is False
    assert plan.comfy_slot.account_id == "g4_ultra_acct"
    assert plan.comfy_slot.gpu == "G4"
    assert plan.comfy_slot.no_named_tunnel is True
    assert abs(plan.estimated_total_rate - 11.99) < 0.01


# =========================================================================
# 2. 底层 scoped-cli / pool-probe 与 colab_ctl.py --account / --no-named-tunnel
# =========================================================================


def test_scoped_cli_pool_probe_and_ctl_no_named_tunnel(
    isolated_env: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """验证桥接脚本多账号隔离凭据/pool-probe/scoped-cli，以及 colab_ctl --account 与 --no-named-tunnel。"""
    tmp = isolated_env.tmp

    # 2a. 在真实 colab_py 环境（全量重定向会话目录至 tmp_path 沙箱）验证 _bind_account_state、pool-probe 与 scoped-cli
    if cb.COLAB_PY.is_file():
        sandbox_bridge_code = f"""
import io, json, os, sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, {str(PKG_DIR)!r})
import colab_cli_bridge as br

tmp = Path({str(tmp)!r})
br.CFG_DIR = tmp / "colab_cfg"
br.SESS_FILE = br.CFG_DIR / "sessions.json"
br.SESS_BACKUP_DIR = br.CFG_DIR / "pet_sessions_by_acct"
br.CFG_DIR.mkdir(parents=True, exist_ok=True)
br.SESS_FILE.write_text('{{"main_untouched": true}}\\n', encoding="utf-8")

# 写入两个独立测试账号凭据到沙箱 Mock Vault（含远期未过期 expiry）
tok_tpl = lambda t: json.dumps({{
    "token": t,
    "refresh_token": "r_" + t,
    "token_uri": "https://oauth2.googleapis.com/token",
    "client_id": "cid.apps.googleusercontent.com",
    "client_secret": "csec",
    "expiry": "2099-01-01T00:00:00Z",
}})
br._write_vault_token("acct_alpha", tok_tpl("ya29_alpha"))
br._write_vault_token("acct_beta", tok_tpl("ya29_beta"))

# 验证 _bind_account_state 指向独立会话文件且绝不触碰默认 sessions.json
p_alpha = br._bind_account_state("acct_alpha")
assert p_alpha == br.SESS_BACKUP_DIR / "acct_alpha.json"
assert br.state.config_path == str(p_alpha)
assert json.loads(br.SESS_FILE.read_text(encoding="utf-8")) == {{"main_untouched": True}}

# 验证 cmd_scoped_cli 绑定账号状态并传递 --config <per_acct_sess_path>
captured_cli = []
with patch.object(br, "colab_cli_app", side_effect=lambda args, prog_name: captured_cli.append((list(args), prog_name))):
    rc_sc = br.cmd_scoped_cli(["--acct=acct_beta", "--", "sessions"])
assert rc_sc == 0
assert captured_cli == [(["--config", str(br.SESS_BACKUP_DIR / "acct_beta.json"), "sessions"], "colab")]

# 验证 cmd_pool_probe 并发探测多号且各线程隔离不串号
def _fake_metrics(client, local_sessions):
    hdr = str(client.session.credentials.token)
    if "alpha" in hdr:
        return {{"balance": 52.5, "rate_hourly": 1.94, "assignments_count": 1, "eligible_gpus": ["L4", "T4"], "eligible_tpus": [], "sessions": [{{"name": "comfy", "endpoint": "ep-a", "accelerator": "L4"}}]}}
    return {{"balance": 166.0, "rate_hourly": 0.0, "assignments_count": 0, "eligible_gpus": ["G4", "A100", "L4", "T4"], "eligible_tpus": [], "sessions": []}}

buf = io.StringIO()
with (
    patch.object(br, "_probe_client_metrics", side_effect=_fake_metrics),
    patch.object(sys, "stdin", io.StringIO(json.dumps({{"account_ids": ["acct_alpha", "acct_beta", "acct_missing"]}}))),
    patch.object(sys, "stdout", buf),
):
    rc_pp = br.cmd_pool_probe([])
assert rc_pp == 0
out = buf.getvalue()
assert "<<<COLAB_POOL_PROBE>>>" in out
s = out.split("<<<COLAB_POOL_PROBE>>>", 1)[1].split("<<<END_COLAB_POOL_PROBE>>>", 1)[0]
parsed = json.loads(s)
assert parsed["ok"] is True
assert parsed["accounts"]["acct_alpha"]["balance"] == 52.5
assert parsed["accounts"]["acct_beta"]["eligible_gpus"] == ["G4", "A100", "L4", "T4"]
assert parsed["accounts"]["acct_missing"]["auth_missing"] is True
print("SANDBOX_BRIDGE_VERIFIED_OK")
"""
        proc_br = subprocess.run(
            [str(cb.COLAB_PY), "-c", sandbox_bridge_code],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=25,
            env=dict(os.environ),
        )
        assert proc_br.returncode == 0 and "SANDBOX_BRIDGE_VERIFIED_OK" in proc_br.stdout, (
            proc_br.stdout,
            proc_br.stderr,
        )

    # 2b. 验证 colab_ctl.cmd_up 的 --account 作用域隔离与 --no-named-tunnel 置空远端固定隧道变量
    tools = _prepare_fake_tools(tmp)
    fake_boot = tmp / "fake_minimax_boot.py"
    fake_boot.write_text("print('FAKE_MINIMAX_BOOT_OK')\n", encoding="utf-8")

    # 预置沙箱固定命名隧道配置，验证 --no-named-tunnel 能强制将其置空
    csec.save_cf_tunnel_settings(
        isolated_env.cf_file,
        comfy_url="https://comfy-fixed.example.com",
        llm_url="https://llm-fixed.example.com",
        tunnel_token="eyJhIjoiZHVtbXkiLCJ0IjoiZHVtbXkiLCJzIjoiZHVtbXkifQ==",
        llm_api_key="sk-pet-fixedtestkey12345678",
    )

    monkeypatch.setattr(colab_ctl, "COLAB_EXE", tools.colab_exe)
    monkeypatch.setattr(colab_ctl, "COLAB_PY", tools.colab_py)
    monkeypatch.setattr(colab_ctl, "RCLONE_EXE", tools.rclone_exe)
    monkeypatch.setattr(colab_ctl, "RCLONE_CONF", tools.rclone_conf)
    monkeypatch.setattr(colab_ctl, "MINIMAX_BOOTSTRAP_SCRIPT", fake_boot)

    seen_cmds: list[list[str]] = []
    wrapped_bodies: list[str] = []

    def _fake_ctl_run(cmd, check=True, capture=False, timeout=None):
        c = [str(x) for x in cmd]
        seen_cmds.append(c)
        if "sessions" in c:
            return subprocess.CompletedProcess(c, 0, stdout="comfy ep-g4-slot\n", stderr="")
        return subprocess.CompletedProcess(c, 0, stdout="", stderr="")

    def _fake_exec_file(session, script_path, **kwargs):
        wrapped_bodies.append(Path(script_path).read_text(encoding="utf-8"))
        assert kwargs.get("account_id") == "acct_comfy_slot"
        return subprocess.CompletedProcess(["ok"], 0, stdout="", stderr="")

    monkeypatch.setattr(colab_ctl, "_run", _fake_ctl_run)
    monkeypatch.setattr(colab_ctl, "_colab_exec_file", _fake_exec_file)

    up_args = colab_ctl.build_parser().parse_args(
        [
            "up",
            "--account",
            "acct_comfy_slot",
            "--no-named-tunnel",
            "--skip-preflight",
            "--profile",
            "minimax",
            "--gpu",
            "G4",
        ]
    )
    colab_ctl.cmd_up(up_args)

    # 校验退出 cmd_up 后未污染进程环境变量
    assert os.environ.get("COLAB_PET_ACCOUNT_ID") is None
    # 校验带 --account 时对 colab 的调用全部经由 scoped-cli --acct=acct_comfy_slot 隔离
    assert any("scoped-cli" in c and "--acct=acct_comfy_slot" in c for c in seen_cmds), seen_cmds
    # 校验 --no-named-tunnel 显式清空远端 CF_TUNNEL_TOKEN / CF_TUNNEL_URL / CF_COMFY_TUNNEL_URL
    assert len(wrapped_bodies) == 1
    body = wrapped_bodies[0]
    assert "os.environ['CF_TUNNEL_TOKEN'] = ''" in body
    assert "os.environ['CF_TUNNEL_URL'] = ''" in body
    assert "os.environ['CF_COMFY_TUNNEL_URL'] = ''" in body
    assert "comfy-fixed.example.com" not in body


# =========================================================================
# 3a. 后端 refresh_account_pool 并发隔离与 G4 缺卡自动换号 Failover 重试 1 次
# =========================================================================


def test_backend_pool_refresh_and_stockout_failover_once(isolated_env: SimpleNamespace) -> None:
    """验证号池探测不污染当前主账本，且单实例遇 G4 紧缺自动标记并切换备号重试 1 次成功。"""
    tools = _prepare_fake_tools(isolated_env.tmp)
    backend = _make_backend(isolated_env, "failover_state")
    backend.state["account_profiles"] = [
        {"id": "acct_primary", "label": "主力一号", "email_masked": "p1***@gmail.com", "last_balance": 120.0},
        {"id": "acct_backup", "label": "备用二号", "email_masked": "p2***@gmail.com", "last_balance": 190.0},
    ]
    backend.state["prefs"]["active_account_id"] = "acct_primary"
    backend.state["prefs"]["auto_pool_schedule"] = True
    backend.state["last_balance"] = 120.0
    backend.state["day_start_balance"] = 130.0
    backend.state["day_consumed_cu"] = 10.0

    pool_probe_resp = {
        "ok": True,
        "accounts": {
            "acct_primary": {
                "ok": True,
                "auth_ok": True,
                "balance": 118.5,
                "rate_hourly": 0.0,
                "assignments_count": 0,
                "eligible_gpus": ["G4", "L4", "T4"],
                "sessions": [],
            },
            "acct_backup": {
                "ok": True,
                "auth_ok": True,
                "balance": 188.0,
                "rate_hourly": 0.0,
                "assignments_count": 0,
                "eligible_gpus": ["G4", "A100", "L4", "T4"],
                "sessions": [],
            },
        },
    }

    switched_ops: list[dict[str, Any]] = []
    streaming_cmds: list[list[str]] = []

    def _mock_hidden(cmd: list[str], timeout: int = 25, extra_env: Any = None, input_text: Any = None):
        if "pool-probe" in cmd:
            return subprocess.CompletedProcess(
                cmd, 0, stdout=f"<<<COLAB_POOL_PROBE>>>{json.dumps(pool_probe_resp)}<<<END_COLAB_POOL_PROBE>>>", stderr=""
            )
        if "account-op" in cmd:
            req = json.loads(input_text or "{}")
            switched_ops.append(req)
            out = json.dumps({"ok": True, "op": "switch", "target_acct_id": req.get("target_acct_id")})
            return subprocess.CompletedProcess(cmd, 0, stdout=f"<<<COLAB_ACCT_OP>>>{out}<<<END_COLAB_ACCT_OP>>>", stderr="")
        if "probe" in cmd:
            cur_aid = backend.state["prefs"].get("active_account_id")
            live = {
                "balance": 188.0 if cur_aid == "acct_backup" else 118.5,
                "rate_hourly": 10.05 if streaming_cmds else 0.0,
                "assignments_count": 1 if len(streaming_cmds) >= 2 else 0,
                "eligible_gpus": ["G4", "L4", "T4"],
                "sessions": (
                    [{"name": "comfy", "endpoint": "ep-backup-g4", "accelerator": "G4", "machine_shape": "STANDARD"}]
                    if len(streaming_cmds) >= 2
                    else []
                ),
            }
            return subprocess.CompletedProcess(
                cmd, 0, stdout=f"<<<COLAB_PET_JSON>>>{json.dumps(live)}<<<END_COLAB_PET_JSON>>>", stderr=""
            )
        if "fast-tunnel" in cmd:
            pkt = {"status": "ok", "endpoint": "ep-backup-g4", "url": "https://backup-g4.trycloudflare.com", "meta": {"kind": "comfy", "port": 8188}}
            return subprocess.CompletedProcess(
                cmd, 0, stdout=f"<<<COLAB_TUNNEL>>>{json.dumps(pkt)}<<<END_COLAB_TUNNEL>>>", stderr=""
            )
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    def _mock_run_streaming(cmd: list[Any], *, on_line=None, **kwargs):
        c_list = [str(x) for x in cmd]
        streaming_cmds.append(c_list)
        if len(streaming_cmds) == 1:
            # 第 1 次在主力一号分配 G4 遭遇容量耗尽 stockout
            if on_line:
                on_line("=== 开启 Colab 实例 (session=comfy, gpu=G4, high_mem=False) ===")
                on_line("RuntimeError: ResourceExhausted: out of capacity for G4")
            return CommandResult(cmd=c_list, returncode=1, elapsed_sec=0.1)
        # 第 2 次已自动 Failover 切至备用二号，分配成功并产出隧道
        if on_line:
            on_line("=== [7/7] 启动公网隧道 ===")
            on_line("COMFYUI_TUNNEL_URL: https://backup-g4.trycloudflare.com")
        return CommandResult(cmd=c_list, returncode=0, elapsed_sec=0.2)

    with (
        patch.object(backend, "_get_colab_exe", return_value=tools.colab_exe),
        patch.object(backend, "_get_colab_py", return_value=tools.colab_py),
        patch.object(backend, "_get_rclone_exe", return_value=tools.rclone_exe),
        patch.object(backend, "_get_rclone_conf", return_value=tools.rclone_conf),
        patch.object(backend, "run_preflight", return_value={"ok": True}),
        patch.object(cb, "_run_hidden", side_effect=_mock_hidden),
        patch.object(backend._runner, "run_streaming", side_effect=_mock_run_streaming),
    ):
        pool_res = backend.refresh_account_pool()
        assert pool_res["ok"] is True
        # 刷新算力池严禁污染当前激活号主账本字段
        assert backend.state["last_balance"] == 120.0
        assert backend.state["day_consumed_cu"] == 10.0
        assert backend.state["account_ledgers"]["acct_backup"]["last_balance"] == 188.0

        # 发起单实例 G4 部署：本号 acct_primary 无阻塞先被选中 -> 遇 G4 stockout -> 自动切至 acct_backup 重试 1 次成功
        start_res = backend.start_instance({"profiles": ["minimax"], "gpu": "G4", "cluster_mode": "cohost"})
        assert start_res["ok"] is True

        deadline = time.monotonic() + 5.0
        while backend.is_deploying() and time.monotonic() < deadline:
            time.sleep(0.03)
        assert not backend.is_deploying()

        assert len(streaming_cmds) == 2
        assert "--account" in streaming_cmds[0] and streaming_cmds[0][streaming_cmds[0].index("--account") + 1] == "acct_primary"
        assert "--account" in streaming_cmds[1] and streaming_cmds[1][streaming_cmds[1].index("--account") + 1] == "acct_backup"
        assert ("acct_primary", "G4") in backend._cluster_svc._acct_scarce_gpus
        assert backend.state["prefs"]["active_account_id"] == "acct_backup"
        assert any(op.get("target_acct_id") == "acct_backup" for op in switched_ops)


# =========================================================================
# 3b. 双号异构并行部署、分槽停卡与跨号 Seam-A/B/C 全链路验证
# =========================================================================


def test_split_cluster_deploy_selective_stop_and_cross_account_seams(isolated_env: SimpleNamespace) -> None:
    """验证双号异构并行部署（LLM 保留命名隧道、Comfy --no-named-tunnel）、跨号 Seam-A/B/C 与分槽停卡。"""
    tools = _prepare_fake_tools(isolated_env.tmp)
    backend = _make_backend(isolated_env, "split_cluster")
    backend.state["account_profiles"] = [
        {"id": "acct_l4", "label": "Magpie对话号", "email_masked": "l4***@gmail.com", "last_balance": 66.0},
        {"id": "acct_g4", "label": "Ultra画布号", "email_masked": "g4***@gmail.com", "last_balance": 175.0},
    ]
    backend.state["prefs"]["active_account_id"] = "acct_l4"
    backend.state["prefs"]["cluster_split_mode"] = True
    backend.state["last_balance"] = 66.0
    backend.state["last_sessions"] = [{"name": "comfy", "endpoint": "ep-llm-l4", "accelerator": "L4"}]
    backend.state["last_assignments_count"] = 1

    active_by_acct: dict[str, list[dict[str, str]]] = {
        "acct_l4": [],
        "acct_g4": [],
    }
    split_up_cmds: list[list[str]] = []
    down_cmds: list[list[str]] = []
    fast_tunnel_cmds: list[list[str]] = []
    install_ext_cmds: list[list[str]] = []
    lora_pull_cmds: list[list[str]] = []

    def _mock_hidden(cmd: list[str], timeout: int = 25, extra_env: Any = None, input_text: Any = None):
        if "pool-probe" in cmd:
            req_ids = (json.loads(input_text or "{}").get("account_ids")) or []
            accts_out = {}
            for aid in req_ids:
                sess = list(active_by_acct.get(aid, []))
                accts_out[aid] = {
                    "ok": True,
                    "auth_ok": True,
                    "balance": 66.0 if aid == "acct_l4" else 175.0,
                    "rate_hourly": (1.94 if aid == "acct_l4" else 10.05) if sess else 0.0,
                    "assignments_count": len(sess),
                    "eligible_gpus": ["L4", "T4"] if aid == "acct_l4" else ["G4", "A100", "L4", "T4"],
                    "sessions": sess,
                }
            return subprocess.CompletedProcess(
                cmd, 0, stdout=f"<<<COLAB_POOL_PROBE>>>{json.dumps({'ok': True, 'accounts': accts_out})}<<<END_COLAB_POOL_PROBE>>>", stderr=""
            )
        if "fast-tunnel" in cmd:
            fast_tunnel_cmds.append(list(cmd))
            is_g4 = "--acct=acct_g4" in cmd
            pkt = (
                {
                    "status": "ok",
                    "endpoint": "ep-comfy-g4",
                    "url": "https://cfy-split.trycloudflare.com",
                    "comfy_api_key": "cpk-split-comfy-secret-9999",
                    "meta": {"kind": "comfy", "port": 8188, "comfy_tunnel_url": "https://cfy-split.trycloudflare.com"},
                }
                if is_g4
                else {
                    "status": "ok",
                    "endpoint": "ep-llm-l4",
                    "url": "https://llm-magpie.trycloudflare.com",
                    "api_key": "sk-pet-splitllmsecret8888",
                    "meta": {"kind": "llm", "port": 18080, "model": "Qwen3.8-27B-abliterated", "llm_tunnel_url": "https://llm-magpie.trycloudflare.com"},
                }
            )
            return subprocess.CompletedProcess(
                cmd, 0, stdout=f"<<<COLAB_TUNNEL>>>{json.dumps(pkt)}<<<END_COLAB_TUNNEL>>>", stderr=""
            )
        if "probe" in cmd:
            sess_l4 = list(active_by_acct["acct_l4"])
            live = {
                "balance": 66.0,
                "rate_hourly": 1.94 if sess_l4 else 0.0,
                "assignments_count": len(sess_l4),
                "eligible_gpus": ["L4", "T4"],
                "sessions": sess_l4,
            }
            return subprocess.CompletedProcess(
                cmd, 0, stdout=f"<<<COLAB_PET_JSON>>>{json.dumps(live)}<<<END_COLAB_PET_JSON>>>", stderr=""
            )
        if len(cmd) >= 3 and cmd[1] == "-c" and "REMOTE_LORA_READY" in cmd[2]:
            lora_pull_cmds.append(list(cmd))
            return subprocess.CompletedProcess(cmd, 0, stdout="REMOTE_LORA_READY\n", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    def _mock_split_streaming(cmd: list[Any], *, on_line=None, **kwargs):
        c_list = [str(x) for x in cmd]
        split_up_cmds.append(c_list)
        aid = c_list[c_list.index("--account") + 1]
        if aid == "acct_l4":
            active_by_acct["acct_l4"] = [{"name": "comfy", "endpoint": "ep-llm-l4", "accelerator": "L4"}]
            if on_line:
                on_line("=== [6/7] 启动 llama-server ===")
                on_line("SERVICE_TUNNEL_URL: https://llm-magpie.trycloudflare.com")
        else:
            active_by_acct["acct_g4"] = [{"name": "comfy", "endpoint": "ep-comfy-g4", "accelerator": "G4"}]
            if on_line:
                on_line("=== [6/7] 启动 ComfyUI ===")
                on_line("COMFYUI_TUNNEL_URL: https://cfy-split.trycloudflare.com")
        return CommandResult(cmd=c_list, returncode=0, elapsed_sec=0.15)

    def _mock_runner_run(cmd: list[Any], **kwargs):
        c_list = [str(x) for x in cmd]
        if "down" in c_list:
            down_cmds.append(c_list)
            aid = c_list[c_list.index("--account") + 1] if "--account" in c_list else ""
            active_by_acct[aid] = []
            v_out = "<<<STOP_VERIFIED>>>" + json.dumps({"ok": True, "remaining": []}) + "<<<END_STOP_VERIFIED>>>"
            return CommandResult(cmd=c_list, returncode=0, stdout=v_out)
        return CommandResult(cmd=c_list, returncode=0, stdout="")

    class _DummyHttpResp:
        status_code = 404
        text = ""

        def json(self) -> dict[str, Any]:
            return {}

        def close(self) -> None:
            pass

    def _mock_subproc_run(cmd: list[str], *args: Any, **kwargs: Any):
        if "install-ext" in cmd:
            install_ext_cmds.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="mock skip")

    with (
        patch.object(backend, "_get_colab_exe", return_value=tools.colab_exe),
        patch.object(backend, "_get_colab_py", return_value=tools.colab_py),
        patch.object(backend, "_get_rclone_exe", return_value=tools.rclone_exe),
        patch.object(backend, "_get_rclone_conf", return_value=tools.rclone_conf),
        patch.object(backend, "run_preflight", return_value={"ok": True}),
        patch.object(cb, "_run_hidden", side_effect=_mock_hidden),
        patch.object(backend._runner, "run_streaming", side_effect=_mock_split_streaming),
        patch.object(backend._runner, "run", side_effect=_mock_runner_run),
    ):
        backend.refresh_account_pool()
        res_start = backend.start_instance({"profiles": ["minimax", "qwen_27b"], "cluster_mode": "split"})
        assert res_start["ok"] is True and res_start.get("cluster_split") is True

        deadline = time.monotonic() + 5.0
        while backend.is_deploying() and time.monotonic() < deadline:
            time.sleep(0.03)
        assert not backend.is_deploying()

        # 验证两路并行 up 参数：LLM(acct_l4, L4, 保留固定隧道) vs Comfy(acct_g4, G4, --no-named-tunnel)
        assert len(split_up_cmds) == 2
        llm_cmd = next(c for c in split_up_cmds if "acct_l4" in c)
        comfy_cmd = next(c for c in split_up_cmds if "acct_g4" in c)
        assert "--gpu" in llm_cmd and llm_cmd[llm_cmd.index("--gpu") + 1] == "L4"
        assert "--no-named-tunnel" not in llm_cmd
        assert "--gpu" in comfy_cmd and comfy_cmd[comfy_cmd.index("--gpu") + 1] == "G4"
        assert "--no-named-tunnel" in comfy_cmd

        snap_active = backend.get_cached_snapshot()
        assert snap_active["cluster"]["active"] is True
        assert abs(snap_active["cluster"]["total_rate_hourly"] - 11.99) < 0.01

        # 验证跨号 Seam-A：清空内存密钥模拟桌宠重启后，仅凭 base_url (ep="") 触发 401 口令补水必带 --acct=acct_g4
        with backend._lock:
            backend._comfy_secrets_by_ep.clear()
            backend._last_comfy_key_hydrate_m.clear()
        fast_tunnel_cmds.clear()
        hydrated = backend._hydrate_comfy_key_if_needed("https://cfy-split.trycloudflare.com", ep="", force=True)
        assert hydrated == "cpk-split-comfy-secret-9999"
        assert fast_tunnel_cmds and "--acct=acct_g4" in fast_tunnel_cmds[-1]

        # 验证跨号 Seam-B：拖拽工作流直铺画布命中 ep-comfy-g4 且 install-ext 携带 --acct=acct_g4
        with (
            patch.object(cb.subprocess, "run", side_effect=_mock_subproc_run),
            patch.object(backend, "_comfy_http_get", return_value=_DummyHttpResp()),
            patch.object(backend, "_comfy_http_post", return_value=_DummyHttpResp()),
            patch.object(cb.webbrowser, "open", return_value=True),
        ):
            backend._push_or_open_workflow_on_canvas("https://cfy-split.trycloudflare.com", "test_wf.json")
        assert install_ext_cmds
        assert "ep-comfy-g4" in install_ext_cmds[-1] and "--acct=acct_g4" in install_ext_cmds[-1]

        # 验证跨号 Seam-C：拖拽 .safetensors LoRA 注入 _bind_account_state('acct_g4') 拉取至画布实例
        lora_file = isolated_env.tmp / "demo_lora.safetensors"
        lora_file.write_bytes(b"safetensors_stub")
        with (
            patch.object(backend, "_rclone_upload_file", return_value=None),
            patch.object(backend, "_comfy_http_get", return_value=_DummyHttpResp()),
        ):
            backend._upload_model_file(lora_file, "https://cfy-split.trycloudflare.com")
        assert lora_pull_cmds and "_bind_account_state('acct_g4')" in lora_pull_cmds[-1][2]

        # 验证仅停画布 stop_cluster_slot("comfy")：仅释放 acct_g4，保留 acct_l4 的 L4 LLM 在线供 Magpie 使用
        res_stop_comfy = backend.stop_cluster_slot("comfy")
        assert res_stop_comfy["ok"] is True
        assert down_cmds and "--account" in down_cmds[-1]
        assert down_cmds[-1][down_cmds[-1].index("--account") + 1] == "acct_g4"
        assert "--endpoint" in down_cmds[-1] and down_cmds[-1][down_cmds[-1].index("--endpoint") + 1] == "ep-comfy-g4"

        rem_slots = backend._cluster_svc.get_cluster_slots()
        assert "comfy" not in rem_slots and "llm" in rem_slots
        assert backend.get_comfy_state()["service_kind"] == "llm"

        # 验证全停 stop_cluster_slot("all")：释放剩余 LLM 槽位并彻底清空集群状态
        res_stop_all = backend.stop_cluster_slot("all")
        assert res_stop_all["ok"] is True
        assert backend._cluster_svc.get_cluster_slots() == {}
        assert backend.get_cached_snapshot()["cluster"]["active"] is False


# =========================================================================
# 4. 真实 PyWebView (392x548 frameless) 真窗 CDP 交互与 5 页 oy == 0 零溢出核验
# =========================================================================


def _run_pywebview_cluster_child(work_dir: Path) -> int:
    """在独立子进程 STA 主线程拉起真实 392x548 frameless 窗体执行 CDP 点击与几何溢出校验。"""
    import pet_app  # noqa: WPS433
    import webview  # noqa: WPS433

    backend = cb.ColabPetBackend(
        state_path=work_dir / "wv_state.json",
        download_root=work_dir / "wv_dl",
        cf_config_path=work_dir / "wv_cf.json",
    )
    backend.update_prefs(
        {
            "boot_profile": "minimax",
            "boot_profiles": ["minimax", "qwen_27b"],
            "boot_gpu": "G4",
            "cluster_split_mode": True,
            "auto_pool_schedule": True,
        }
    )
    stop_event = threading.Event()
    api = pet_app.PetBridgeApi(backend, stop_event)

    for m_name, exp_args in (
        ("refresh_account_pool", ["self"]),
        ("stop_cluster_slot", ["self", "acct_id"]),
    ):
        fn = getattr(api, m_name, None)
        assert callable(fn)
        spec = inspect.getfullargspec(fn.__func__)
        assert spec.varargs is None and spec.varkw is None and spec.args == exp_args

    calls: list[tuple[str, Any]] = []

    idle_snap: dict[str, Any] = {
        "ok": True,
        "stale": False,
        "error_msg": None,
        "balance": 168.40,
        "baseline_balance": 200.0,
        "balance_pct": 84.2,
        "rate_hourly": 0.0,
        "rate_pct": 0.0,
        "remaining_hours": None,
        "assignments_count": 0,
        "sessions": [],
        "is_running": False,
        "has_orphan": False,
        "active_minutes": 0,
        "active_since_ts": None,
        "long_running_alert": False,
        "low_balance": False,
        "mood": "proud",
        "status_level": "idle",
        "status_label": "已停机省积分",
        "date": "2026-11-18",
        "day_start_balance": 175.0,
        "day_consumed_cu": 6.60,
        "day_topup_cu": 0.0,
        "session_consumed_cu": 0.0,
        "last_delta": 0.0,
        "updated_str": "02:10:00",
        "updated_ts": int(time.time()),
        "eligible_gpus": ["G4", "A100", "L4", "T4"],
        "eligible_tpus": [],
        "scarce_gpus": {},
        "measured_gpu_rates": dict(cb.DEFAULT_MEASURED_GPU_RATES),
        "deploy": {"active": False},
        "prefs": dict(backend.state["prefs"]),
        "comfy": cb.ColabPetBackend._default_comfy_state(),
        "cluster": {
            "active": False,
            "can_split": True,
            "pool_healthy_count": 2,
            "total_pool_balance": 233.6,
            "total_rate_hourly": 0.0,
            "slots": {},
        },
        "onboarding": {
            "needs_attention": False,
            "colab_cli_ok": True,
            "colab_exe_ok": True,
            "colab_auth_ok": True,
            "auth_initializing": False,
            "auth_missing": False,
            "auth_expired": False,
            "rclone_exe_ok": True,
            "gdrive_ok": True,
            "gdrive_free_gb": 120.0,
            "gdrive_total_gb": 200.0,
            "active_account_id": "acct_g4",
            "accounts": [
                {
                    "id": "acct_l4",
                    "label": "Magpie对话副号",
                    "email_masked": "mag***pi@gmail.com",
                    "last_balance": 65.2,
                    "last_is_running": False,
                    "eligible_gpus": ["L4", "T4"],
                    "cluster_role": None,
                },
                {
                    "id": "acct_g4",
                    "label": "Ultra主力渲染号",
                    "email_masked": "ult***99@gmail.com",
                    "last_balance": 168.4,
                    "last_is_running": False,
                    "eligible_gpus": ["G4", "A100", "L4", "T4"],
                    "cluster_role": None,
                },
                {
                    "id": "acct_t4",
                    "label": "受限仅T4测试号",
                    "email_masked": "t4o***01@gmail.com",
                    "last_balance": 9.5,
                    "last_is_running": False,
                    "eligible_gpus": ["T4"],
                    "cluster_role": None,
                },
            ],
            "pending_oauth": False,
            "gdrive_binding": False,
            "cf_tunnel": {"mode": "named", "comfy_url": "https://cfy.example.com", "llm_url": "https://llm.example.com"},
            "tool_paths": {"colab_exe": "colab.exe", "colab_py": "python.exe", "rclone_exe": "rclone.exe", "rclone_conf": "rclone.conf"},
        },
    }

    active_snap: dict[str, Any] = json.loads(json.dumps(idle_snap))
    active_snap.update(
        {
            "is_running": True,
            "rate_hourly": 10.05,
            "rate_pct": 67.0,
            "assignments_count": 1,
            "active_minutes": 18,
            "active_since_ts": int(time.time()) - 18 * 60,
            "status_level": "running",
            "status_label": "双号集群计费中",
            "sessions": [{"name": "comfy", "endpoint": "ep-comfy-g4", "accelerator": "G4", "machine_shape": "STANDARD"}],
            "comfy": {
                **cb.ColabPetBackend._default_comfy_state(),
                "online": True,
                "service_kind": "dual",
                "comfy_online": True,
                "llm_online": True,
                "tunnel_url": "https://cfy-demo.trycloudflare.com",
                "llm_tunnel_url": "https://llm.example.com",
                "active_profiles": ["minimax", "qwen_27b"],
                "model_name": "Qwen3.8-27B-abliterated",
                "quant": "UD-Q4_K_XL",
                "ctx_size": 65536,
                "vram_used_gb": 62.4,
                "vram_total_gb": 96.0,
                "vram_pct": 65.0,
            },
            "cluster": {
                "active": True,
                "can_split": True,
                "pool_healthy_count": 2,
                "total_pool_balance": 233.6,
                "total_rate_hourly": 11.99,
                "slots": {
                    "llm": {
                        "account_id": "acct_l4",
                        "label": "Magpie对话副号",
                        "gpu": "L4",
                        "endpoint": "ep-llm-l4",
                        "rate_hourly": 1.94,
                        "tunnel_url": "https://llm.example.com",
                        "online": True,
                    },
                    "comfy": {
                        "account_id": "acct_g4",
                        "label": "Ultra主力渲染号",
                        "gpu": "G4",
                        "endpoint": "ep-comfy-g4",
                        "rate_hourly": 10.05,
                        "tunnel_url": "https://cfy-demo.trycloudflare.com",
                        "online": True,
                    },
                },
            },
        }
    )
    active_snap["onboarding"]["accounts"][0]["last_is_running"] = True
    active_snap["onboarding"]["accounts"][0]["cluster_role"] = "llm"
    active_snap["onboarding"]["accounts"][1]["last_is_running"] = True
    active_snap["onboarding"]["accounts"][1]["cluster_role"] = "comfy"

    backend.start_instance = lambda cfg=None: (calls.append(("start_instance", dict(cfg or {}))) or {"ok": True, "started": False})  # type: ignore[method-assign]
    backend.refresh_account_pool = lambda: (calls.append(("refresh_account_pool", None)) or {"ok": True, "snapshot": active_snap, "message": "已刷新全池"})  # type: ignore[method-assign]

    def _stop_slot_stub(slot: str = "all") -> dict[str, Any]:
        calls.append(("stop_cluster_slot", slot))
        next_s = json.loads(json.dumps(active_snap))
        if slot == "comfy":
            next_s["cluster"]["slots"].pop("comfy", None)
            next_s["cluster"]["total_rate_hourly"] = 1.94
            next_s["comfy"]["service_kind"] = "llm"
            next_s["comfy"]["comfy_online"] = False
        return {"ok": True, "snapshot": next_s, "message": "已停画布保留LLM"}

    backend.stop_cluster_slot = _stop_slot_stub  # type: ignore[method-assign]
    # 预置初始快照，消除 pywebviewready 异步调用 api.get_state() 覆盖后续测试快照的时序竞态
    backend.last_snapshot = json.loads(json.dumps(idle_snap))

    httpd = pet_app._ThreadingHTTPServer(("127.0.0.1", 0), pet_app._PetAssetHandler)
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()

    window = webview.create_window(
        title="Pytest 多号集群 UI 真窗校验",
        url=f"http://127.0.0.1:{port}/ui.html",
        js_api=api,
        width=pet_app.BASE_W,
        height=pet_app.BASE_H,
        x=180,
        y=110,
        resizable=False,
        frameless=True,
        easy_drag=False,
        shadow=False,
        transparent=False,
        background_color=pet_app.EDGE_BG_HEX,
        focus=True,
        on_top=True,
    )
    api._bind_window(window)
    errs: list[BaseException] = []

    def _worker() -> None:
        try:
            window.events.loaded.wait(timeout=10.0)
            for _ in range(60):
                if window.native is not None:
                    break
                time.sleep(0.05)
            assert window.native is not None
            form = window.native
            form.Invoke(pet_app.DotNetAction(lambda: (setattr(form, "TopMost", True), form.Show(), form.Activate())))
            window.resize(pet_app.BASE_W, pet_app.BASE_H)

            # 显式等待 pywebviewready 首轮 api.get_state() -> window.renderState(snap) 完全落定后再推进测试
            ready_deadline = time.monotonic() + 8.0
            ui_ready = False
            while time.monotonic() < ready_deadline:
                try:
                    ui_ready = bool(
                        window.evaluate_js(
                            "Boolean(window.pywebview && window.pywebview.api && typeof window.renderState === 'function' && latestSnap && latestSnap.cluster && bootState && bootState.prefsSynced)"
                        )
                    )
                except Exception:
                    ui_ready = False
                if ui_ready:
                    break
                time.sleep(0.04)
            assert ui_ready, "等待 pywebviewready 与首轮 window.renderState 超时"
            time.sleep(0.08)

            def _cdp(method: str, params: dict[str, Any]) -> None:
                t_box: list[Any] = []
                e_box: list[BaseException] = []
                p_str = json.dumps(params, ensure_ascii=False)

                def _start() -> None:
                    try:
                        wv = next(c for c in form.Controls if type(c).__name__ == "WebView2")
                        t_box.append(wv.CoreWebView2.CallDevToolsProtocolMethodAsync(method, p_str))
                    except BaseException as ex:
                        e_box.append(ex)

                form.Invoke(pet_app.DotNetAction(_start))
                if e_box:
                    raise e_box[0]
                t = t_box[0]
                deadline = time.monotonic() + 5.0
                while not t.IsCompleted:
                    if time.monotonic() > deadline:
                        raise TimeoutError(method)
                    time.sleep(0.005)
                form.Invoke(pet_app.DotNetAction(lambda: e_box.append(RuntimeError(str(t.Exception))) if t.IsFaulted else None))
                if e_box:
                    raise e_box[0]

            def _click_sel(sel: str) -> None:
                pt = window.evaluate_js(
                    f"(() => {{ const r = document.querySelector({sel!r}).getBoundingClientRect();"
                    " return {x: r.left + r.width / 2, y: r.top + r.height / 2}; })()"
                )
                fx, fy = float(pt["x"]), float(pt["y"])
                _cdp("Input.dispatchMouseEvent", {"type": "mouseMoved", "x": fx, "y": fy, "button": "none", "buttons": 0})
                _cdp("Input.dispatchMouseEvent", {"type": "mousePressed", "x": fx, "y": fy, "button": "left", "buttons": 1, "clickCount": 1})
                time.sleep(0.02)
                _cdp("Input.dispatchMouseEvent", {"type": "mouseReleased", "x": fx, "y": fy, "button": "left", "buttons": 0, "clickCount": 1})

            def _check_zero_overflow(label: str) -> None:
                m = window.evaluate_js(
                    "(() => ({"
                    "  stageSh: el('stage').scrollHeight,"
                    "  pages: [...document.querySelectorAll('#pages-track > .page-body')].map((p, i) => ({i, oy: p.scrollHeight - p.clientHeight}))"
                    "}))()"
                )
                assert m["stageSh"] <= 548, f"[{label}] stage 高度溢出: {m}"
                for pg in m["pages"]:
                    assert pg["oy"] == 0, f"[{label}] Page {pg['i']+1} 纵向溢出 oy={pg['oy']}"

            # 1) 第 3 页 #chip-cluster-split 切换与 cluster_mode 透传
            pet_app._push_to_ui(window, idle_snap)
            window.evaluate_js("setPage(2); setConsoleTab('boot', true);")
            time.sleep(0.45)
            _check_zero_overflow("P3-Boot")
            chip_diag = window.evaluate_js(
                "(() => {"
                "  const c = el('chip-cluster-split');"
                "  return {"
                "    txt: c ? c.textContent.trim() : '',"
                "    cls: c ? c.className : '',"
                "    profiles: bootState.profiles,"
                "    splitMode: bootState.clusterSplitMode,"
                "    canSplit: Boolean(latestSnap && latestSnap.cluster && latestSnap.cluster.can_split)"
                "  };"
                "})()"
            )
            assert chip_diag["txt"] == "🔀 双号异构" and "hidden" not in chip_diag["cls"], f"双号异构胶囊状态异常: {chip_diag}"
            _click_sel("#btn-boot-start")
            time.sleep(0.18)
            assert calls[-1][1].get("cluster_mode") == "split", calls[-1]
            _click_sel("#chip-cluster-split")
            time.sleep(0.18)
            chip_after = window.evaluate_js("({txt: el('chip-cluster-split').textContent.trim(), cls: el('chip-cluster-split').className})")
            assert chip_after["txt"] == "🔗 单号同驻", f"点击切换单号同驻未生效: {chip_after}"

            # 2) 第 1 页号池总积分、Σ符号、异构双在线跨号耗速、分槽标签与 #comfy-strip 零截断
            pet_app._push_to_ui(window, active_snap)
            window.evaluate_js("setPage(0); hideToast();")
            time.sleep(0.65)
            _check_zero_overflow("P1-DualOnline")
            p1 = window.evaluate_js(
                "(() => {"
                "  const l = el('comfy-left'), v = el('comfy-vram');"
                "  const sig = el('cu-sigma');"
                "  return {"
                "    cuText: el('cu-balance').textContent,"
                "    sigmaHidden: sig ? sig.classList.contains('hidden') : true,"
                "    miniCu: el('mini-cu').textContent,"
                "    balLbl: el('balance-label-k') ? el('balance-label-k').textContent : '',"
                "    badges: el('session-badges').textContent,"
                "    rate: el('rate-text').textContent,"
                "    leftTxt: l.textContent,"
                "    leftTrunc: l.scrollWidth > l.clientWidth + 1,"
                "    vramTrunc: v.scrollWidth > v.clientWidth + 1"
                "  };"
                "})()"
            )
            assert p1["cuText"] == "233.60", p1
            assert p1["sigmaHidden"] is False, p1
            assert p1["miniCu"] == "Σ 233.60 CU", p1
            assert "全池水位" in p1["balLbl"], p1
            assert "💬LLM(L4)" in p1["badges"] and "🎨画布(G4)" in p1["badges"]
            assert "跨号总耗速: 11.99 CU/hr" in p1["rate"]
            assert not p1["leftTrunc"] and not p1["vramTrunc"], p1

            # 3) 第 5 页卡池徽章/角色角标与刷新全池/智能寻优
            window.evaluate_js("setPage(4); setAcctTab('vault', true);")
            time.sleep(0.42)
            _check_zero_overflow("P5-Vault")
            p5_txt = window.evaluate_js("el('acct-vault-list').textContent")
            for badge in ("L4池", "💬LLM", "G4池", "🎨画布", "仅T4", "Pro"):
                assert badge in p5_txt
            _click_sel("#btn-pool-refresh")
            time.sleep(0.20)
            assert any(k == "refresh_account_pool" for k, _ in calls)
            _click_sel("#btn-pool-auto-schedule")
            time.sleep(0.18)
            assert backend.state["prefs"]["auto_pool_schedule"] is False

            # 4) 第 3 页 #act-stop 3 选单触发 stop_cluster_slot("comfy")
            window.evaluate_js("hideToast(); setPage(2); setConsoleTab('ops', true);")
            time.sleep(0.42)
            _click_sel("#act-stop")
            time.sleep(0.28)
            _check_zero_overflow("P3-StopChoices")
            labels = window.evaluate_js("[...document.querySelectorAll('#speech-toast .toast-choice-btn')].map(b => b.textContent.trim())")
            assert labels == ["仅停画布", "仅停LLM", "全停"]
            _click_sel("#speech-toast .toast-choice-btn.primary")
            time.sleep(0.25)
            assert ("stop_cluster_slot", "comfy") in calls
        except BaseException as exc:
            errs.append(exc)
        finally:
            stop_event.set()
            try:
                window.destroy()
            except Exception:
                pass
            try:
                httpd.shutdown()
                httpd.server_close()
            except Exception:
                pass

    webview.start(func=_worker)
    if errs:
        raise errs[0]
    print("PYWEBVIEW_CHILD_ALL_OK", flush=True)
    return 0


@pytest.mark.gui
def test_pywebview_frameless_cluster_ui_and_zero_overflow(isolated_env: SimpleNamespace) -> None:
    """拉起真实 392x548 frameless PyWebView 窗口核验全 5 页 oy==0、双在线零截断与 CDP 交互。"""
    work_dir = isolated_env.tmp / "webview_box"
    work_dir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--webview-child", str(work_dir)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=45,
        env=dict(os.environ),
    )
    assert proc.returncode == 0 and "PYWEBVIEW_CHILD_ALL_OK" in proc.stdout, (
        f"PyWebView 真窗子进程失败 (rc={proc.returncode}):\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )


# =========================================================================
# 5. Windows 便携包构建包含 backend_cluster.py 与 pool_scheduler.py 且过安全闸门
# =========================================================================


def test_account_tier_classification_and_mixed_pool_baseline(isolated_env: SimpleNamespace) -> None:
    """校验 Ultra (896 CU / H100) 与 Pro (200 CU / 无 H100) 混合号池精准识别、默认备注修正与全池基准聚合。"""
    # 1. 纯函数判定验证
    assert infer_account_tier_and_baseline(
        896.25, eligible_gpus=["H100", "G4", "A100", "L4", "T4"]
    ) == ("ultra", "Ultra", 1000.0)
    assert infer_account_tier_and_baseline(
        200.0, eligible_gpus=["G4", "A100", "L4", "T4"], ineligible_gpus=["H100"]
    ) == ("pro", "Pro", 200.0)
    assert infer_account_tier_and_baseline(
        420.0, eligible_gpus=["G4", "A100", "L4", "T4"], ineligible_gpus=["H100"]
    ) == ("pro_plus", "Pro+", 500.0)
    assert infer_account_tier_and_baseline(
        0.0, eligible_gpus=["T4"], ineligible_gpus=["H100", "G4"]
    ) == ("free", "Free", 100.0)
    # 即使 Ultra 余额用到 120 CU，只要有 H100 或历史峰值 > 550，仍识别为 Ultra
    assert infer_account_tier_and_baseline(
        120.0, eligible_gpus=["H100", "G4", "A100", "L4", "T4"]
    ) == ("ultra", "Ultra", 1000.0)
    assert format_pool_tier_summary(["ultra", "pro"]) == "1 Ultra + 1 Pro"
    assert format_pool_tier_summary(["pro", "pro"]) == "2 个 Pro 账号"

    # 2. 后端号池探针聚合：系统自动标签 "Ultra 2号" 应自修正为 "Pro 2号"，用户自定义标签保留不变
    backend = _make_backend(isolated_env, "mixed_pool")
    with backend._lock:
        backend.state["prefs"]["active_account_id"] = "acct_ultra"
        backend.state["account_profiles"] = [
            {"id": "acct_ultra", "label": "Ultra 1号", "email_masked": "u***1@gmail.com"},
            {"id": "acct_pro_auto", "label": "Ultra 2号", "email_masked": "p***2@gmail.com"},
            {"id": "acct_pro_custom", "label": "demo_acct_01", "email_masked": "de***3@gmail.com"},
        ]

    backend._cluster_svc.apply_pool_probe_results(
        {
            "acct_ultra": {
                "ok": True,
                "balance": 896.25,
                "eligible_gpus": ["H100", "G4", "A100", "L4", "T4"],
                "sessions": [],
            },
            "acct_pro_auto": {
                "ok": True,
                "balance": 200.0,
                "eligible_gpus": ["G4", "A100", "L4", "T4"],
                "ineligible_gpus": ["H100"],
                "sessions": [],
            },
            "acct_pro_custom": {
                "ok": True,
                "balance": 100.0,
                "eligible_gpus": ["G4", "A100", "L4", "T4"],
                "ineligible_gpus": ["H100"],
                "sessions": [],
            },
        }
    )
    snap = backend._rebuild_and_emit_snapshot(stale=False)
    cluster = snap["cluster"]
    assert cluster["total_pool_balance"] == 1196.25
    assert cluster["total_pool_baseline"] == 1300.0  # Ultra 1000 + Pro 200 + 100 CU(100)
    assert cluster["pool_tier_summary"] == "1 Ultra + 2 Pro"
    accts_by_id = {a["id"]: a for a in snap["onboarding"]["accounts"]}
    assert accts_by_id["acct_ultra"]["tier"] == "ultra"
    assert accts_by_id["acct_pro_auto"]["tier"] == "pro"
    assert accts_by_id["acct_pro_auto"]["label"] == "Pro 2号"
    assert accts_by_id["acct_pro_custom"]["tier"] == "pro"
    assert accts_by_id["acct_pro_custom"]["label"] == "demo_acct_01"
    assert snap["onboarding"]["active_account_label"] == "Ultra 1号"


def test_portable_package_contains_cluster_modules_and_passes_gate(isolated_env: SimpleNamespace) -> None:
    """调用 build_portable_package 验证 zip 内含 backend_cluster.py 与 pool_scheduler.py 且通过安全闸门。"""
    out_dir = isolated_env.tmp / "dist_out"
    bundle_dir, zip_path = win_pkg.build_portable_package(PKG_DIR, out_dir)
    assert bundle_dir.is_dir() and zip_path.is_file()

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = set(zf.namelist())
        assert "Colab桌宠/colab_pet/backend_cluster.py" in names
        assert "Colab桌宠/colab_core/pool_scheduler.py" in names
        assert "Colab桌宠/Colab桌宠.exe" in names
        # 确认无敏感状态文件进入发行包
        for forbidden in win_pkg.FORBIDDEN_FILENAMES:
            assert not any(n.endswith(f"/{forbidden}") for n in names)


# =========================================================================
# 6. 多账号隔离会话别名合并与受管实例对账（消除孤儿实例黄色误报）
# =========================================================================


def test_scoped_account_session_not_misclassified_as_orphan(
    isolated_env: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """校验分号隔离会话在主号探针中不误标为孤儿实例，且后端对账只还原受管绑定并保留真实网页孤儿。"""
    tmp = isolated_env.tmp

    # 6a. Bridge 层：会话仅存在于 pet_sessions_by_acct/<acct>.json，主 sessions.json 保持原样
    if cb.COLAB_PY.is_file():
        sandbox_probe_code = f"""
import io, json, sys
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path

sys.path.insert(0, {str(PKG_DIR)!r})
import colab_cli_bridge as br
from colab_cli.state import SessionState

tmp = Path({str(tmp)!r})
br.CFG_DIR = tmp / "orphan_fix_cfg"
br.SESS_FILE = br.CFG_DIR / "sessions.json"
br.SESS_BACKUP_DIR = br.CFG_DIR / "pet_sessions_by_acct"
br.CFG_DIR.mkdir(parents=True, exist_ok=True)
br.SESS_BACKUP_DIR.mkdir(parents=True, exist_ok=True)
br.SESS_FILE.write_text('{{"main_untouched": true}}\\n', encoding="utf-8")
br.state.config_path = str(br.SESS_FILE)
br.state._store = br.StateStore(str(br.SESS_FILE))

scoped_store = br.StateStore(str(br.SESS_BACKUP_DIR / "acct_7437.json"))
scoped_store.add(SessionState(
    name="comfy",
    token="rt_tok",
    url="https://rt.example.com",
    endpoint="ep-scoped-g4",
    variant="GPU",
    accelerator="G4",
    machine_shape="STANDARD",
))

class _FakeClient:
    colab_domain = "https://colab.research.google.com"
    session = SimpleNamespace(get=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("skip ccu")))
    def get_consumption_user_info(self):
        return SimpleNamespace(
            paid_compute_units_balance=895.23,
            consumption_rate_hourly=10.05,
            assignments_count=2,
        )
    def list_assignments(self):
        return [
            SimpleNamespace(
                endpoint="ep-scoped-g4",
                accelerator=SimpleNamespace(value="G4"),
                variant=SimpleNamespace(name="GPU"),
                machine_shape=SimpleNamespace(name="STANDARD"),
            ),
            SimpleNamespace(
                endpoint="ep-web-orphan",
                accelerator=SimpleNamespace(value="T4"),
                variant=SimpleNamespace(name="GPU"),
                machine_shape=SimpleNamespace(name="STANDARD"),
            ),
        ]

tok_json = json.dumps({{
    "token": "ya29_valid",
    "refresh_token": "r_valid",
    "token_uri": "https://oauth2.googleapis.com/token",
    "client_id": "cid",
    "client_secret": "csec",
    "expiry": "2099-01-01T00:00:00Z",
}})
br.state._client = _FakeClient()
buf = io.StringIO()
with (
    patch.object(br.colab_auth, "_get_token_from_keyring", return_value=tok_json),
    patch.object(sys, "stdout", buf),
):
    rc = br.cmd_probe(["--active-acct=acct_7437"])
assert rc == 0
out = buf.getvalue()
payload = json.loads(out.split("<<<COLAB_PET_JSON>>>", 1)[1].split("<<<END_COLAB_PET_JSON>>>", 1)[0])
by_ep = {{s["endpoint"]: s for s in payload["sessions"]}}
assert by_ep["ep-scoped-g4"]["name"] == "comfy" and by_ep["ep-scoped-g4"]["orphan"] is False
assert by_ep["ep-web-orphan"]["name"] == "?" and by_ep["ep-web-orphan"]["orphan"] is True
assert json.loads(br.SESS_FILE.read_text(encoding="utf-8")) == {{"main_untouched": True}}
print("BRIDGE_ORPHAN_FIX_OK")
"""
        proc = subprocess.run(
            [str(cb.COLAB_PY), "-c", sandbox_probe_code],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=25,
            env=dict(os.environ),
        )
        assert proc.returncode == 0 and "BRIDGE_ORPHAN_FIX_OK" in proc.stdout, (
            proc.stderr or proc.stdout
        )

    # 6b. Backend 对账兜底：已知 managed_binding 不再误报孤儿实例，未知网页孤儿仍精准预警
    backend = _make_backend(isolated_env, "reconcile_orphan")
    with backend._lock:
        backend.state["prefs"]["active_account_id"] = "acct_alpha"
        backend.state["managed_binding"] = {
            "account_id": "acct_alpha",
            "session": "comfy",
            "endpoint": "ep-bound-g4",
        }
        live_bound = {
            "balance": 895.0,
            "rate_hourly": 10.05,
            "assignments_count": 1,
            "sessions": [
                {
                    "name": "?",
                    "endpoint": "ep-bound-g4",
                    "accelerator": "G4",
                    "variant": "GPU",
                    "machine_shape": "STANDARD",
                    "orphan": True,
                }
            ],
        }
        backend._update_ledger(live_bound)
        snap = backend._build_snapshot(live_data=live_bound, stale=False, error_msg=None)

    assert snap["has_orphan"] is False
    assert snap["status_label"] == "GPU 运行计费中"
    assert snap["sessions"][0]["name"] == "comfy"
    assert snap["sessions"][0]["orphan"] is False

    # 真正的未托管网页孤儿实例依然触发黄标提醒
    with backend._lock:
        snap_orphan = backend._build_snapshot(
            live_data={
                "balance": 890.0,
                "rate_hourly": 10.05,
                "assignments_count": 1,
                "sessions": [
                    {
                        "name": "?",
                        "endpoint": "ep-real-web-orphan",
                        "accelerator": "G4",
                        "variant": "GPU",
                        "machine_shape": "STANDARD",
                        "orphan": True,
                    }
                ],
            },
            stale=False,
            error_msg=None,
        )
    assert snap_orphan["has_orphan"] is True
    assert snap_orphan["status_label"] == "发现孤儿实例"
    assert snap_orphan["sessions"][0]["orphan"] is True


def test_cluster_lost_slot_binding_never_autoclaims_and_stop_calls_zero(isolated_env: SimpleNamespace) -> None:
    """验证原槽位实例消失但同账号仍有另一台实例时，号池不自动认领且分槽停卡调用次数为 0。"""
    tools = _prepare_fake_tools(isolated_env.tmp)
    backend = _make_backend(isolated_env, "lost_slot_binding")
    with backend._lock:
        backend.state["account_profiles"] = [
            {"id": "acct_l4", "label": "对话号", "email_masked": "l4***@gmail.com", "last_balance": 50.0},
            {"id": "acct_g4", "label": "画布号", "email_masked": "g4***@gmail.com", "last_balance": 120.0},
        ]
        backend.state["prefs"]["active_account_id"] = "acct_l4"
        backend.state["cluster_slots"] = {
            "llm": {
                "account_id": "acct_l4", "label": "对话号", "gpu": "L4",
                "endpoint": "ep-llm-alive", "tunnel_url": "https://llm.trycloudflare.com", "online": True,
            },
            "comfy": {
                "account_id": "acct_g4", "label": "画布号", "gpu": "G4",
                "endpoint": "ep-comfy-old", "tunnel_url": "https://cfy-old.trycloudflare.com", "online": True,
            },
        }

    # 云端 ep-comfy-old 已消失，但同账号 acct_g4 下恰好存在另一台未归属实例 ep-other-active
    active_by_acct: dict[str, list[dict[str, str]]] = {
        "acct_l4": [{"name": "comfy", "endpoint": "ep-llm-alive", "accelerator": "L4"}],
        "acct_g4": [{"name": "other_task", "endpoint": "ep-other-active", "accelerator": "T4"}],
    }
    down_calls: list[list[str]] = []

    def _mock_hidden(cmd: list[str], timeout: int = 25, extra_env: Any = None, input_text: Any = None):
        _ = (timeout, extra_env)
        if "pool-probe" in cmd:
            req_ids = (json.loads(input_text or "{}").get("account_ids")) or []
            accts = {
                aid: {
                    "ok": True,
                    "auth_ok": True,
                    "balance": 50.0 if aid == "acct_l4" else 120.0,
                    "rate_hourly": 1.94,
                    "assignments_count": len(active_by_acct.get(aid, [])),
                    "eligible_gpus": ["L4", "T4"],
                    "sessions": list(active_by_acct.get(aid, [])),
                }
                for aid in req_ids
            }
            return subprocess.CompletedProcess(
                cmd, 0, stdout=f"<<<COLAB_POOL_PROBE>>>{json.dumps({'ok': True, 'accounts': accts})}<<<END_COLAB_POOL_PROBE>>>", stderr=""
            )
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    def _forbid_down(cmd: list[Any], **_kw: Any):
        c_list = [str(x) for x in cmd]
        if "down" in c_list:
            down_calls.append(c_list)
        return CommandResult(cmd=c_list, returncode=0, stdout="")

    with (
        patch.object(backend, "_get_colab_exe", return_value=tools.colab_exe),
        patch.object(backend, "_get_colab_py", return_value=tools.colab_py),
        patch.object(cb, "_run_hidden", side_effect=_mock_hidden),
        patch.object(backend._runner, "run", side_effect=_forbid_down),
    ):
        backend.refresh_account_pool(emit=False)
        slot_after_refresh = backend._cluster_svc.get_slot("comfy") or {}
        assert slot_after_refresh.get("endpoint") == "ep-comfy-old", "原实例消失时绝不可自动认领同号另一台实例"
        assert slot_after_refresh.get("binding_invalid") is True
        assert slot_after_refresh.get("online") is False

        res_stop_marked = backend.stop_cluster_slot("comfy")
        assert res_stop_marked["ok"] is False
        assert res_stop_marked.get("binding_invalid") is True
        assert res_stop_marked.get("target_unresolved") is True
        assert down_calls == []

        # 即使绕过刷新（binding_invalid 尚未置位），stop_cluster_slot 前置探针也必须拒绝盲停且调用次数为 0
        with backend._lock:
            backend.state["cluster_slots"]["comfy"]["binding_invalid"] = False
        res_stop_direct = backend.stop_cluster_slot("comfy")
        assert res_stop_direct["ok"] is False
        assert res_stop_direct.get("binding_invalid") is True
        assert down_calls == []
        assert (backend._cluster_svc.get_slot("comfy") or {}).get("endpoint") == "ep-comfy-old"


# =========================================================================
# 5. 部署点击零阻塞（后台异步号池刷新）与进度日志大小写无关解析回归测试
# =========================================================================


def test_start_instance_nonblocking_stale_pool_refresh_and_progress_parsing(
    isolated_env: SimpleNamespace,
) -> None:
    """验证号池过期时 start_instance 秒返（单模/双号异构均异步探测并支持探测期取消），及进度文案解析。"""
    # 1) 验证 consume_deploy_line 大小写无关匹配：sessions 核验 -> 开启 Colab 实例 -> GPU 硬件信息
    st = default_deploy_state()
    st["gpu"] = "G4"
    ctx = {"phase_idx": 1, "phase_total": 1, "phase_prof": "anima", "phase_is_llm": False}
    consume_deploy_line("[本机执行] colab.EXE sessions", st, ctx, {})
    assert st["pct"] >= 4 and "核验云端会话" in st["stage_title"], st
    consume_deploy_line("=== 开启 Colab 实例 (session=comfy, gpu=G4, high_mem=False) ===", st, ctx, {})
    assert st["pct"] == 6 and st["stage_title"] == "申请 G4 实例", st
    assert "G4 GPU" in st["sub_detail"], st
    consume_deploy_line("=== [1/7] 环境与显卡检查 ===", st, ctx, {})
    consume_deploy_line("GPU 硬件信息: NVIDIA RTX PRO 6000 Blackwell", st, ctx, {})
    assert "RTX PRO 6000" in st["sub_detail"], st

    tools = _prepare_fake_tools(isolated_env.tmp)
    backend = _make_backend(isolated_env, "async_boot_state")
    backend.state["account_profiles"] = [
        {"id": "acct_a", "label": "一号", "email_masked": "a***@gmail.com", "last_balance": 120.0, "tier": "pro"},
        {"id": "acct_b", "label": "二号", "email_masked": "b***@gmail.com", "last_balance": 200.0, "tier": "pro"},
        {"id": "acct_c", "label": "三号", "email_masked": "c***@gmail.com", "last_balance": 850.0, "tier": "ultra"},
    ]
    backend.state["account_ledgers"] = {
        "acct_a": {"last_balance": 120.0, "last_eligible_gpus": ["G4", "L4", "T4"]},
        "acct_b": {"last_balance": 200.0, "last_eligible_gpus": ["G4", "L4", "T4"]},
        "acct_c": {"last_balance": 850.0, "last_eligible_gpus": ["H100", "G4", "A100", "L4", "T4"]},
    }
    backend.state["prefs"]["active_account_id"] = "acct_a"
    backend.state["prefs"]["auto_pool_schedule"] = True
    backend.state["last_balance"] = 120.0

    streaming_cmds: list[list[str]] = []
    probe_delay_sec = 0.35
    probe_entered = threading.Event()
    probe_release: threading.Event | None = None
    probe_data: dict[str, dict[str, Any]] = {
        # 初始 acct_a 缓存看似有 120 CU，但后台刷新后发现仅剩 5.0 (< 12 CU 门槛)，触发自动优选切号至 acct_c
        "acct_a": {"ok": True, "auth_ok": True, "balance": 5.0, "rate_hourly": 0.0, "assignments_count": 0, "eligible_gpus": ["G4", "L4", "T4"], "sessions": []},
        "acct_b": {"ok": True, "auth_ok": True, "balance": 198.0, "rate_hourly": 0.0, "assignments_count": 0, "eligible_gpus": ["G4", "L4", "T4"], "sessions": []},
        "acct_c": {"ok": True, "auth_ok": True, "balance": 840.0, "rate_hourly": 0.0, "assignments_count": 0, "eligible_gpus": ["H100", "G4", "A100", "L4", "T4"], "sessions": []},
    }

    def _mock_hidden(cmd: list[str], timeout: int = 25, extra_env: Any = None, input_text: Any = None):
        _ = (timeout, extra_env)
        if "pool-probe" in cmd:
            probe_entered.set()
            if probe_release is not None:
                probe_release.wait(timeout=2.0)
            else:
                time.sleep(probe_delay_sec)
            req_ids = (json.loads(input_text or "{}").get("account_ids")) or list(probe_data.keys())
            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout=f"<<<COLAB_POOL_PROBE>>>{json.dumps({'ok': True, 'accounts': {aid: probe_data[aid] for aid in req_ids}})}<<<END_COLAB_POOL_PROBE>>>",
                stderr="",
            )
        if "account-op" in cmd:
            req = json.loads(input_text or "{}")
            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout=f"<<<COLAB_ACCT_OP>>>{json.dumps({'ok': True, 'op': 'switch', 'target_acct_id': req.get('target_acct_id')})}<<<END_COLAB_ACCT_OP>>>",
                stderr="",
            )
        if "probe" in cmd:
            aid = backend.state["prefs"].get("active_account_id") or "acct_a"
            bal = float(probe_data.get(aid, {}).get("balance") or 100.0)
            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout=f"<<<COLAB_PET_JSON>>>{json.dumps({'balance': bal, 'rate_hourly': 0.0, 'assignments_count': 0, 'eligible_gpus': ['G4', 'L4', 'T4'], 'sessions': []})}<<<END_COLAB_PET_JSON>>>",
                stderr="",
            )
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    def _mock_streaming(cmd: list[Any], *, on_line=None, **kwargs):
        c_list = [str(x) for x in cmd]
        streaming_cmds.append(c_list)
        if on_line:
            on_line("=== 开启 Colab 实例 (session=comfy, gpu=G4, high_mem=False) ===")
            on_line("=== [7/7] 启动公网隧道 ===")
            on_line("COMFYUI_TUNNEL_URL: https://async-boot-ok.trycloudflare.com")
        return CommandResult(cmd=c_list, returncode=0, elapsed_sec=0.05)

    with (
        patch.object(backend, "_get_colab_exe", return_value=tools.colab_exe),
        patch.object(backend, "_get_colab_py", return_value=tools.colab_py),
        patch.object(backend, "_get_rclone_exe", return_value=tools.rclone_exe),
        patch.object(backend, "_get_rclone_conf", return_value=tools.rclone_conf),
        patch.object(backend, "run_preflight", return_value={"ok": True}),
        patch.object(cb, "_run_hidden", side_effect=_mock_hidden),
        patch.object(backend._runner, "run_streaming", side_effect=_mock_streaming),
    ):
        # 2) 单节点 cohost：缓存过期时不阻塞主线程（< 0.25s 返回），后台刷新后切至余额最高的 acct_c 开机
        t0 = time.monotonic()
        res_cohost = backend.start_instance({"profiles": ["minimax"], "gpu": "G4", "cluster_mode": "cohost"})
        elapsed_call = time.monotonic() - t0
        assert elapsed_call < 0.25, f"start_instance 被号池探测阻塞了 {elapsed_call:.3f}s"
        assert res_cohost["ok"] is True and res_cohost["started"] is True
        assert "校验算力号池" in str(res_cohost["deploy"]["stage_title"])

        deadline = time.monotonic() + 5.0
        while (backend.is_deploying() or backend._deploy_lock.locked()) and time.monotonic() < deadline:
            time.sleep(0.02)
        assert not backend.is_deploying() and not backend._deploy_lock.locked()
        assert len(streaming_cmds) == 1
        assert "--account" in streaming_cmds[0] and streaming_cmds[0][streaming_cmds[0].index("--account") + 1] == "acct_c"

        # 3) 异步探测期间调用 cancel_deploy()：立即干净中止，绝不触发 run_streaming 且释放锁
        streaming_cmds.clear()
        backend._cluster_svc._last_pool_refresh_m = 0.0
        probe_entered.clear()
        probe_release = threading.Event()

        res_cancel_case = backend.start_instance({"profiles": ["minimax"], "gpu": "G4", "cluster_mode": "cohost"})
        assert res_cancel_case["ok"] is True
        assert probe_entered.wait(timeout=2.0)
        cancel_res = backend.cancel_deploy()
        assert cancel_res["ok"] is True
        probe_release.set()

        deadline = time.monotonic() + 5.0
        while (backend.is_deploying() or backend._deploy_lock.locked()) and time.monotonic() < deadline:
            time.sleep(0.02)
        assert not backend.is_deploying() and not backend._deploy_lock.locked()
        assert streaming_cmds == [], "探测期已取消部署却仍发起了 run_streaming"
        assert backend.get_deploy_state()["stage_title"] == "已中止部署"

        # 4) 双号异构 split：清空内存探针使之过期，但凭持久化账本秒级响应并后台重规划最新号池
        probe_release = None
        streaming_cmds.clear()
        backend._cluster_svc._pool_probe_by_acct.clear()
        backend._cluster_svc._last_pool_refresh_m = 0.0
        # 刷新后 acct_a 仍可用但余额更低，acct_b/acct_c 余额充沛且均能跑 L4/G4
        t1 = time.monotonic()
        res_split = backend.start_instance({"profiles": ["minimax", "qwen_27b"], "cluster_mode": "split"})
        elapsed_split = time.monotonic() - t1
        assert elapsed_split < 0.25, f"split start_instance 被号池探测阻塞了 {elapsed_split:.3f}s"
        assert res_split["ok"] is True and res_split.get("cluster_split") is True
        assert res_split["deploy"]["stage_title"] == "校验双号算力池"

        deadline = time.monotonic() + 5.0
        while (backend.is_deploying() or backend._deploy_lock.locked()) and time.monotonic() < deadline:
            time.sleep(0.02)
        assert not backend.is_deploying() and not backend._deploy_lock.locked()
        assert len(streaming_cmds) == 2


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--webview-child", default="")
    ns, _ = ap.parse_known_args()
    if ns.webview_child:
        rc = int(_run_pywebview_cluster_child(Path(ns.webview_child)))
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(rc)
    raise SystemExit(pytest.main([__file__, "-v"]))
