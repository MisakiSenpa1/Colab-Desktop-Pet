"""第 4~5 步模块重构、统一任务执行器、启动前自检、可复现基线、Windows 打包与双栖孤儿回归测试。

说明：全部基于隔离临时目录与本地沙箱环境变量，绝不读取或修改宿主机真实 `.cf_tunnel_secret.json`、
Windows 系统凭据库或 `~/.config/colab-cli/sessions.json`。
"""

from __future__ import annotations

import argparse
import io
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
for p_str in (str(PKG_DIR), str(PET_DIR)):
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

import colab_ctl  # noqa: E402
import colab_backend as cb  # noqa: E402
import colab_security as csec  # noqa: E402
import build_windows_package as win_pkg  # noqa: E402
from colab_core import agent_tools as core_agent_tools  # noqa: E402
from colab_core import health as core_health  # noqa: E402
from colab_core import runtime as core_runtime  # noqa: E402
core_deploy = core_runtime
core_preflight = core_health
core_repro = core_health
core_runner = core_runtime


@pytest.fixture()
def isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """创建隔离沙箱环境（模拟凭据库、隧道配置、结构化诊断日志目录）。"""
    csec.clear_ephemeral_secrets()
    vault_file = tmp_path / "mock_vault.json"
    cf_file = tmp_path / ".cf_tunnel_secret.json"
    diag_log = tmp_path / "logs" / "colab_diag.log"
    monkeypatch.setenv("COLAB_PET_MOCK_VAULT_PATH", str(vault_file))
    monkeypatch.setenv("COLAB_PET_CF_CONFIG_PATH", str(cf_file))
    monkeypatch.setenv("COLAB_PET_DIAG_LOG_PATH", str(diag_log))
    yield SimpleNamespace(
        tmp=tmp_path,
        vault_file=vault_file,
        cf_file=cf_file,
        diag_log=diag_log,
    )
    csec.clear_ephemeral_secrets()


def _prepare_fake_tools(tmp: Path) -> SimpleNamespace:
    fake_exe = tmp / "colab.exe"
    fake_py = tmp / "colab_py.exe"
    fake_rclone = tmp / "rclone.exe"
    fake_conf = tmp / "rclone.conf"
    for p in (fake_exe, fake_py, fake_rclone):
        p.write_text("", encoding="utf-8")
    fake_conf.write_text("[gdrive_save]\ntype = drive\n", encoding="utf-8")
    return SimpleNamespace(
        colab_exe=fake_exe,
        colab_py=fake_py,
        rclone_exe=fake_rclone,
        rclone_conf=fake_conf,
    )


# =========================================================================
# 1. MiniMax + Qwen27B 双栖孤儿 G4 纳管、分配超时放宽与 Traceback 括号过滤回归
# =========================================================================


def test_dual_minimax_qwen27b_orphan_adoption_and_traceback_classification(
    isolated_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    tools = _prepare_fake_tools(isolated_env.tmp)

    # 1a. CLI 层复现并验证：存在 `[?] gpu-g4-orphan-01 | Hardware: G4` 时切勿重复调用 colab new
    called_new = False
    adopt_called = False
    executed_scripts: list[str] = []

    def _fake_cli_run(cmd, check=True, capture=False):
        nonlocal called_new, adopt_called
        c = [str(x) for x in cmd]
        if c[0] == str(tools.colab_exe):
            sub = c[1] if len(c) > 1 else ""
            if sub == "sessions":
                return subprocess.CompletedProcess(
                    c,
                    0,
                    stdout="[?] gpu-g4-orphan-01 | Hardware: G4 | Shape: Standard | Variant: GPU\n",
                    stderr="",
                )
            if sub == "new":
                called_new = True
                raise AssertionError("存在匹配卡型的孤儿实例时不应重复调用 colab new")
            return subprocess.CompletedProcess(c, 0, stdout="ok", stderr="")
        if c[0] == str(tools.colab_py) and len(c) >= 3 and c[1] == "-c" and "COLAB_ADOPT_SESSION" in c[2]:
            adopt_called = True
            pkt = json.dumps(
                {
                    "adopted": True,
                    "already_named": False,
                    "endpoint": "gpu-g4-orphan-01",
                    "accelerator": "G4",
                }
            )
            return subprocess.CompletedProcess(
                c,
                0,
                stdout=f"<<<COLAB_ADOPT_SESSION>>>{pkt}<<<END_COLAB_ADOPT_SESSION>>>",
                stderr="",
            )
        return subprocess.CompletedProcess(c, 0, stdout="", stderr="")

    def _fake_exec_file(_session: str, script_path: Path, **_kw):
        executed_scripts.append(script_path.read_text(encoding="utf-8"))
        return subprocess.CompletedProcess([], 0, "", "")

    monkeypatch.setattr(colab_ctl, "COLAB_EXE", tools.colab_exe)
    monkeypatch.setattr(colab_ctl, "COLAB_PY", tools.colab_py)
    monkeypatch.setattr(colab_ctl, "RCLONE_EXE", tools.rclone_exe)
    monkeypatch.setattr(colab_ctl, "RCLONE_CONF", tools.rclone_conf)
    monkeypatch.setattr(colab_ctl, "_run", _fake_cli_run)
    monkeypatch.setattr(colab_ctl, "_colab_exec_file", _fake_exec_file)

    up_args = argparse.Namespace(
        session="comfy",
        endpoint=None,
        profile="minimax",
        profiles="minimax,qwen_27b",
        gpu="G4",
        high_mem=False,
        recreate=False,
        skip_preflight=False,
        minimax_lite=False,
        no_r2v=False,
        no_pe=False,
        llm_quant="UD-Q4_K_XL",
        llm_ctx=65536,
        no_mmproj=False,
        no_webui=False,
        gdrive_cache=False,
        llm_api_key="auto",
    )
    colab_ctl.cmd_up(up_args)

    assert adopt_called is True
    assert called_new is False
    assert len(executed_scripts) == 2
    assert "os.environ['LLM_MODEL_VARIANT'] = '27b'" in executed_scripts[0]
    assert "os.environ['LLM_COHOST_COMFY'] = '1'" in executed_scripts[0]
    assert "os.environ['COMFY_RESERVE_VRAM'] = '4.0'" in executed_scripts[1]
    assert "os.environ['COMFY_COHOST_LLM'] = '1'" in executed_scripts[1]

    # 验证双栖隧道进程隔离与版本留痕已落地到共享远端基建和自举脚本
    mm_src = (PKG_DIR / "colab_bootstrap_minimax_h3.py").read_text(encoding="utf-8")
    rc_src = (PKG_DIR / "colab_remote_common.py").read_text(encoding="utf-8")
    assert 'kill_pat = "cloudflared tunnel.* run" if use_named else f"cloudflared tunnel.*{local_port}"' in rc_src
    assert "write_runtime_versions_record(" in mm_src

    # 1b. 后端层 start_instance 孤儿 G4 自动传递 --endpoint 与 COLAB_CLI_REQUEST_TIMEOUT=75
    monkeypatch.setattr(cb, "COLAB_EXE_DEFAULT", tools.colab_exe)
    monkeypatch.setattr(cb, "COLAB_PY", tools.colab_py)
    monkeypatch.setattr(cb, "RCLONE_EXE", tools.rclone_exe)
    monkeypatch.setattr(cb, "RCLONE_CONF", tools.rclone_conf)

    backend = cb.ColabPetBackend(
        state_path=isolated_env.tmp / "pet_dual_state.json",
        download_root=isolated_env.tmp / "dl_dual",
        cf_config_path=isolated_env.cf_file,
    )
    with backend._lock:
        backend.state["account_profiles"] = [{"id": "a1", "label": "U1", "email_masked": "u***1@g.com"}]
        backend.state["prefs"]["active_account_id"] = "a1"
        backend.state["last_assignments_count"] = 1
        backend.state["last_sessions"] = [
            {
                "name": "?",
                "endpoint": "gpu-g4-orphan-01",
                "accelerator": "G4",
                "machine_shape": "STANDARD",
                "orphan": True,
            }
        ]

    seen_cmd: list[str] = []
    seen_env: dict[str, str] = {}
    deploy_done = threading.Event()
    backend._event_cb = lambda e: deploy_done.set() if e.get("type") in ("deploy_done", "deploy_error") else None

    class _MockPopenDual:
        def __init__(self, cmd: list[str], **kwargs: Any) -> None:
            seen_cmd.extend(cmd)
            seen_env.update(kwargs.get("env") or {})
            self.pid = None
            self.stdout = io.StringIO(
                "=== DEPLOY_PHASE 1/2: qwen_27b ===\n"
                "LLM_TUNNEL_URL: https://q27b-dual.trycloudflare.com\n"
                "=== DEPLOY_PHASE 2/2: minimax ===\n"
                "COMFYUI_TUNNEL_URL: https://mm-dual.trycloudflare.com\n"
            )

        def wait(self) -> int:
            return 0

    with (
        patch.object(cb.subprocess, "Popen", _MockPopenDual),
        patch.object(backend, "fetch_live_status", return_value=backend.get_cached_snapshot()),
        patch.object(backend, "_fast_read_tunnel_url", return_value=("ok", "gpu-g4-orphan-01", "https://mm-dual.trycloudflare.com")),
    ):
        sres = backend.start_instance(
            {
                "profiles": ["minimax", "qwen_27b"],
                "gpu": "G4",
                "recreate": False,
            }
        )
        assert sres["ok"] is True, sres
        assert deploy_done.wait(timeout=5.0)

    deadline = time.monotonic() + 2.0
    while backend._deploy_lock.locked() and time.monotonic() < deadline:
        time.sleep(0.02)

    assert "--endpoint" in seen_cmd and "gpu-g4-orphan-01" in seen_cmd, seen_cmd
    assert "--profiles" in seen_cmd and "minimax,qwen_27b" in seen_cmd, seen_cmd
    assert seen_env.get("COLAB_CLI_REQUEST_TIMEOUT") == "75"

    # 1c. Traceback 多行括弧截断回归断言（绝不再输出 ": )"）
    fake_py314_tb = [
        "[本机执行] colab.exe new -s comfy --gpu G4",
        "Traceback (most recent call last):",
        '  File "C:\\colab_pet\\colab_ctl.py", line 347, in _run',
        "    raise subprocess.CalledProcessError(",
        "        res.returncode,",
        "        cmd,",
        "        output=stdout, stderr=stderr,",
        "    )",
        "subprocess.CalledProcessError: Command '['colab.exe', 'new']' returned non-zero exit status 1.",
    ]
    cls_res = core_deploy.classify_deploy_failure(
        fake_py314_tb, "G4", pre_assignments=1, profile="MiniMax + Qwen27B", recreate=False
    )
    assert not cls_res["message"].rstrip().endswith(": )"), cls_res
    assert cls_res["reason"] != ")", cls_res

    timeout_tb = [
        "[本机执行] colab.exe new -s comfy --gpu G4",
        "requests.exceptions.ReadTimeout: HTTPSConnectionPool(host='colab.research.google.com', port=443): Read timed out. (read timeout=12)",
        "    )",
        "subprocess.CalledProcessError: returned non-zero exit status 1",
    ]
    cls_to = core_deploy.classify_deploy_failure(
        timeout_tb, "G4", pre_assignments=0, profile="MiniMax + Qwen27B", recreate=False
    )
    assert cls_to["kind"] == "stockout" and "超时" in cls_to["message"], cls_to


# =========================================================================
# 2. 统一子进程执行器、任务状态机与轮转脱敏日志验证
# =========================================================================


def test_task_runner_timeout_cancel_retry_and_diag_redaction(isolated_env) -> None:
    diag = core_runner.DiagLogger(log_path=isolated_env.diag_log, max_bytes=4096)
    runner = core_runner.ProcessRunner(diag=diag)
    tm = core_runner.TaskManager(diag=diag)

    # 2a. 瞬态失败自动重试并在成功时收敛
    call_cnt = {"n": 0}

    def _flaky_run(cmd, **_kw):
        call_cnt["n"] += 1
        if call_cnt["n"] == 1:
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="transient error")
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout='<<<MY_FRAME>>>{"val": 42}<<<END_MY_FRAME>>>',
            stderr="",
        )

    res_retry = runner.run(
        ["tool.exe", "arg"],
        retries=2,
        retry_delay_sec=0.01,
        run_fn=_flaky_run,
        task_name="flaky_test",
    )
    assert res_retry.ok is True and res_retry.attempts == 2
    assert core_runner.extract_framed_json(res_retry.stdout, "MY_FRAME") == {"val": 42}

    # 2b. 超时返回 timed_out=True 且 stderr 中的授权码/Bearer 强制脱敏
    def _timeout_run(_cmd, **_kw):
        raise subprocess.TimeoutExpired(
            cmd=["x"],
            timeout=1,
            stderr=b"Authorization: Bearer cpk-SecretLeakExample_1234567890",
        )

    res_to = runner.run(["x"], timeout=1, run_fn=_timeout_run, task_name="timeout_test")
    assert res_to.ok is False and res_to.timed_out is True and res_to.returncode == 124
    assert "cpk-SecretLeakExample_1234567890" not in res_to.stderr

    # 2c. TaskManager 捕获后台未捕获异常并写日志，绝不静默丢弃
    def _boom(_tok):
        raise RuntimeError("boom with sk-SecretTaskCrashKey9876543210")

    hdl = tm.submit("crash_task", _boom)
    assert hdl.join(timeout=3.0) is True
    rec = tm.get_record("crash_task")
    assert rec["status"] == "failed"
    assert "sk-SecretTaskCrashKey9876543210" not in rec["error"]

    # 2d. CancelToken 在流式执行前与执行中取消
    pre_tok = core_runner.CancelToken()
    pre_tok.cancel()
    res_pre = runner.run(["echo"], cancel_token=pre_tok, task_name="pre_cancel")
    assert res_pre.cancelled is True and res_pre.ok is False

    # 2e. DiagLogger 写入敏感数据触发脱敏 + >4KB 轮转
    for idx in range(60):
        diag.info(
            "sec",
            f"log line {idx} code=4/0AbCdEfGhIjKlMnOpQrStUvWxYz1234567890",
            jwt="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTYifQ.sig12345",
        )
    backup = isolated_env.diag_log.with_name(isolated_env.diag_log.name + ".1")
    assert backup.is_file(), "超过 4KB 应自动轮转生成 .1 备份日志"
    tail_txt = "\n".join(diag.read_tail_lines(200))
    assert "0AbCdEfGhIjKlMnOpQrStUvWxYz" not in tail_txt
    assert "eyJhbGciOi" not in tail_txt


# =========================================================================
# 3. CLI 与桌宠共享核心模块一致性 & --internal-ctl 分发检查
# =========================================================================


def test_cli_and_pet_shared_core_parity(isolated_env) -> None:
    tools = _prepare_fake_tools(isolated_env.tmp)
    # 3a. Profile 解析与显卡规则两套入口一致
    assert colab_ctl._parse_profiles_arg("minimax,qwen_27b") == ["minimax", "qwen_27b"]
    assert cb._normalize_boot_profiles(["qwen_27b", "qwen_llm"]) == ["qwen_llm"]
    assert core_deploy.select_default_gpu(["minimax", "qwen_27b"]) == "G4"
    assert core_deploy.resolve_high_mem("L4", True) is False
    assert core_deploy.resolve_high_mem("G4", True) is False
    assert core_deploy.resolve_high_mem("A100", True) is True

    # 3b. manifest 与 4 个自举脚本中的桥接节点源码 100% 一致
    m_check = core_repro.verify_manifest_and_assets(PKG_DIR)
    assert m_check["ok"] is True, m_check
    assert m_check["default_profile"] == "anima"

    # 3c. pet_app.py --internal-ctl doctor --json 无 GUI 极速转发测试
    env_sub = os.environ.copy()
    env_sub["PYTHONUTF8"] = "1"
    env_sub["COLAB_PET_COLAB_EXE"] = str(tools.colab_exe)
    env_sub["COLAB_PET_COLAB_PY"] = str(tools.colab_py)
    env_sub["COLAB_PET_RCLONE_EXE"] = str(tools.rclone_exe)
    env_sub["COLAB_PET_RCLONE_CONF"] = str(tools.rclone_conf)
    env_sub["COLAB_PET_MOCK_VAULT_PATH"] = str(isolated_env.vault_file)
    env_sub["COLAB_PET_CF_CONFIG_PATH"] = str(isolated_env.cf_file)
    env_sub["COLAB_PET_DIAG_LOG_PATH"] = str(isolated_env.diag_log)

    proc = subprocess.run(
        [sys.executable, str(PET_DIR / "pet_app.py"), "--internal-ctl", "doctor", "--json"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=25,
        env=env_sub,
    )
    doc_data = json.loads(proc.stdout)
    assert doc_data["mode"] == "deep"
    assert any(it["category"] == "profile" and it["status"] == "ok" for it in doc_data["items"])


# =========================================================================
# 4. 启动前全维度检查（Preflight）与 180B 显存红线拦截
# =========================================================================


def test_preflight_checks_and_cli_doctor(isolated_env, monkeypatch: pytest.MonkeyPatch) -> None:
    tools = _prepare_fake_tools(isolated_env.tmp)
    tp = core_preflight.ToolPaths(
        colab_exe=tools.colab_exe,
        colab_py=tools.colab_py,
        rclone_exe=tools.rclone_exe,
        rclone_conf=tools.rclone_conf,
    )

    # 4a. 全部通过（Anima + L4）
    ok_rep = core_preflight.run_preflight_checks(
        mode="fast",
        profiles=["anima"],
        gpu="L4",
        tool_paths=tp,
        download_dir=isolated_env.tmp,
        cf_config_path=isolated_env.cf_file,
        pkg_dir=PKG_DIR,
    )
    assert ok_rep.ok is True, ok_rep.format_text()

    # 4b. 180B 模型试图选用 L4 -> 必须被 profile 红线拦截
    vram_bad = core_preflight.run_preflight_checks(
        mode="fast",
        profiles=["qwen_llm"],
        gpu="L4",
        tool_paths=tp,
        download_dir=isolated_env.tmp,
        cf_config_path=isolated_env.cf_file,
        pkg_dir=PKG_DIR,
    )
    assert vram_bad.ok is False and "180B" in vram_bad.first_blocking_message()

    # 4c. 固定隧道配置文件引用了 vault key，但系统凭据库缺失 -> preflight 拦阻
    isolated_env.cf_file.write_text(
        json.dumps(
            {
                "comfy_url": "https://comfy.missing-tok.example",
                "tunnel_token_ref": "vault:secret:cf_tunnel_token",
            }
        ),
        encoding="utf-8",
    )
    bad_cf = core_preflight.run_preflight_checks(
        mode="fast",
        profiles=["anima"],
        gpu="L4",
        tool_paths=tp,
        download_dir=isolated_env.tmp,
        cf_config_path=isolated_env.cf_file,
        pkg_dir=PKG_DIR,
    )
    assert bad_cf.ok is False and "Tunnel Token" in bad_cf.first_blocking_message()
    isolated_env.cf_file.unlink(missing_ok=True)

    # 4d. 磁盘剩余空间 < 256 MiB 拦阻
    low_disk = core_preflight.run_preflight_checks(
        mode="fast",
        profiles=["anima"],
        gpu="L4",
        tool_paths=tp,
        download_dir=isolated_env.tmp,
        cf_config_path=isolated_env.cf_file,
        pkg_dir=PKG_DIR,
        disk_usage_fn=lambda _p: SimpleNamespace(total=10**11, used=10**11 - 64 * 1024 * 1024, free=64 * 1024 * 1024),
    )
    assert low_disk.ok is False and "磁盘" in str(low_disk.blocking_items()[0].name)

    # 4e. 缺失 [gdrive_save] 拦阻
    tools.rclone_conf.write_text("[other_remote]\ntype = s3\n", encoding="utf-8")
    no_gd = core_preflight.run_preflight_checks(
        mode="fast",
        profiles=["anima"],
        gpu="L4",
        tool_paths=tp,
        download_dir=isolated_env.tmp,
        cf_config_path=isolated_env.cf_file,
        pkg_dir=PKG_DIR,
    )
    assert no_gd.ok is False and "gdrive_save" in no_gd.first_blocking_message()

    # 4f. CLI doctor --json 验证
    tools.rclone_conf.write_text("[gdrive_save]\ntype = drive\n", encoding="utf-8")
    monkeypatch.setattr(colab_ctl, "COLAB_EXE", tools.colab_exe)
    monkeypatch.setattr(colab_ctl, "COLAB_PY", tools.colab_py)
    monkeypatch.setattr(colab_ctl, "RCLONE_EXE", tools.rclone_exe)
    monkeypatch.setattr(colab_ctl, "RCLONE_CONF", tools.rclone_conf)
    buf = io.StringIO()
    with patch("sys.stdout", buf):
        colab_ctl.main(["doctor", "--profile", "anima", "--gpu", "L4", "--json"])
    doc_parsed = json.loads(buf.getvalue())
    assert doc_parsed["ok"] is True


# =========================================================================
# 5. 默认基线（Anima/L4）完整闭环与全 Profile 扩展矩阵测试
# =========================================================================


def test_default_anima_route_complete_and_extension_profiles(
    isolated_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    tools = _prepare_fake_tools(isolated_env.tmp)
    dl_dir = isolated_env.tmp / "downloads"
    sessions_db: dict[str, str] = {}
    assignments: list[str] = []

    def _mock_run(cmd, check=True, capture=False):
        c = [str(x) for x in cmd]
        if c[0] == str(tools.colab_exe):
            sub = c[1] if len(c) > 1 else ""
            if sub == "sessions":
                txt = "\n".join(f"[{k}] {v} | Hardware: L4" for k, v in sessions_db.items()) or "No active sessions"
                return subprocess.CompletedProcess(c, 0, stdout=txt, stderr="")
            if sub == "new":
                sessions_db["comfy"] = "ep-anima-l4-1"
                assignments.append("ep-anima-l4-1")
                return subprocess.CompletedProcess(c, 0, stdout="Session READY", stderr="")
            if sub == "stop":
                ep = sessions_db.pop("comfy", None)
                if ep in assignments:
                    assignments.remove(ep)
                return subprocess.CompletedProcess(c, 0, stdout="stopped", stderr="")
            if sub == "download":
                import tarfile

                tar_dst = Path(c[-1])
                tar_dst.parent.mkdir(parents=True, exist_ok=True)
                sample = isolated_env.tmp / "xe_verify_sample.png"
                sample.write_bytes(b"PNGDATA_OK")
                with tarfile.open(tar_dst, "w:gz") as tf:
                    tf.add(sample, arcname="xe_verify/out_20261117-010000.png")
                return subprocess.CompletedProcess(c, 0, stdout="downloaded", stderr="")
            return subprocess.CompletedProcess(c, 0, stdout="ok", stderr="")

        if c[0] == str(tools.colab_py) and len(c) >= 3 and c[1] == "-c":
            code = c[2]
            if "get_session" in code:
                return subprocess.CompletedProcess(c, 0, stdout=sessions_db.get("comfy", ""), stderr="")
            if "<<<STOP_VERIFIED>>>" in code:
                return subprocess.CompletedProcess(
                    c,
                    0,
                    stdout="<<<STOP_VERIFIED>>>" + json.dumps({"remaining": list(assignments)}) + "<<<END_STOP_VERIFIED>>>",
                    stderr="",
                )
        return subprocess.CompletedProcess(c, 0, stdout="", stderr="")

    def _mock_exec_file(_session: str, script_path: Path, capture: bool = False, **_kw):
        if "REMOTE_PULL_TAR" in script_path.read_text(encoding="utf-8"):
            return subprocess.CompletedProcess([], 0, '<<<SYNC_VERIFIED>>>{"ok": true, "files": 1}<<<END_SYNC_VERIFIED>>>', "")
        return subprocess.CompletedProcess([], 0, "OK", "")

    monkeypatch.setattr(colab_ctl, "COLAB_EXE", tools.colab_exe)
    monkeypatch.setattr(colab_ctl, "COLAB_PY", tools.colab_py)
    monkeypatch.setattr(colab_ctl, "RCLONE_EXE", tools.rclone_exe)
    monkeypatch.setattr(colab_ctl, "RCLONE_CONF", tools.rclone_conf)
    monkeypatch.setattr(colab_ctl, "LOCAL_DOWNLOAD_DIR", dl_dir)
    monkeypatch.setattr(colab_ctl, "_run", _mock_run)
    monkeypatch.setattr(colab_ctl, "_colab_exec_file", _mock_exec_file)
    monkeypatch.setattr(colab_ctl, "cmd_status", lambda *_a: None)

    # 闭环：up (anima + L4) -> smoke -> sync -> down
    colab_ctl.cmd_up(argparse.Namespace(session="comfy", endpoint=None, profile="anima", profiles=None, gpu="L4", high_mem=False, recreate=False, skip_preflight=False))
    colab_ctl.cmd_smoke(argparse.Namespace(session="comfy", profile="anima"))
    assert (dl_dir / "xe_verify" / "xe_verify" / "out_20261117-010000.png").read_bytes() == b"PNGDATA_OK"
    stop_sum = colab_ctl.stop_session("comfy", clean_all=False)
    assert stop_sum["ok"] is True and stop_sum["remaining_count"] == 0 and not assignments

    # 扩展验证其余已登记 Profile（stack_manifest + 参数构建 + 语法编译）
    manifest = core_repro.load_stack_manifest()
    for prof_name, expected_gpu in (
        ("anima", "L4"),
        ("minimax", "G4"),
        ("drama3", "G4"),
        ("qwen_27b", "L4"),
        ("qwen_llm", "G4"),
    ):
        assert prof_name in manifest["profiles"]
        assert core_deploy.select_default_gpu([prof_name]) == expected_gpu
        cmd_built = core_deploy.build_ctl_up_args(
            python_exe="python.exe",
            ctl_script=PKG_DIR / "colab_ctl.py",
            session="comfy",
            profiles=[prof_name],
            gpu=expected_gpu,
        )
        assert "--gpu" in cmd_built and expected_gpu in cmd_built

    cmd_lite = core_deploy.build_ctl_up_args(
        python_exe="python.exe",
        ctl_script=PKG_DIR / "colab_ctl.py",
        session="comfy",
        profiles=["minimax"],
        gpu="G4",
        minimax_r2v=False,
        minimax_pe=False,
    )
    assert "--minimax-lite" in cmd_lite


# =========================================================================
# 6. 脱敏诊断包导出 & Windows 便携版原生打包实测（含安全闸门与 --self-test）
# =========================================================================


def test_diagnostic_bundle_and_windows_portable_package(
    isolated_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    tools = _prepare_fake_tools(isolated_env.tmp)
    monkeypatch.setattr(cb, "COLAB_EXE_DEFAULT", tools.colab_exe)
    monkeypatch.setattr(cb, "COLAB_PY", tools.colab_py)
    monkeypatch.setattr(cb, "RCLONE_EXE", tools.rclone_exe)
    monkeypatch.setattr(cb, "RCLONE_CONF", tools.rclone_conf)

    # 6a. 诊断包导出与敏感凭据零泄露检查
    dirty_oauth = "ya29.a0SuperSecretLeakOAuthToken_987654321"
    dirty_cf = "eyJSampleCfJwtHeader123456.PayloadSecretXYZ7890.SigSecret999"
    dirty_cpk = "cpk-DiagSecretComfyKey_0123456789abcdef"
    csec.register_ephemeral_secret(dirty_cpk)

    backend = cb.ColabPetBackend(
        state_path=isolated_env.tmp / "diag_state.json",
        download_root=isolated_env.tmp / "diag_dl",
        cf_config_path=isolated_env.cf_file,
    )
    backend._diag.info("test", f"Authorization: Bearer {dirty_cpk} | {dirty_oauth} | {dirty_cf}")

    diag_out_dir = isolated_env.tmp / "diag_export"
    res_diag = backend.export_diagnostics(out_dir=diag_out_dir)
    assert res_diag["ok"] is True, res_diag
    zip_p = Path(res_diag["path"])
    assert zip_p.is_file()

    with zipfile.ZipFile(zip_p, "r") as zf:
        names = set(zf.namelist())
        for expected in (
            "system_info.json",
            "preflight_report.json",
            "state_summary.json",
            "tasks_snapshot.json",
            "stack_manifest_summary.json",
            "logs/colab_diag_tail.log",
        ):
            assert expected in names, names
        for fname in names:
            content = zf.read(fname).decode("utf-8", errors="ignore")
            for secret_s in (dirty_oauth, dirty_cf, dirty_cpk):
                assert secret_s not in content, f"{fname} 泄露明文凭据: {secret_s}"

    # 验证打开日志目录接口
    logs_res = backend.open_logs_dir()
    assert logs_res["ok"] is True and Path(logs_res["path"]).is_dir()

    # 6b. 实测 Windows 便携版完整打包、安全扫描闸门及原生 EXE --self-test
    dist_out = isolated_env.tmp / "win_dist"
    bundle_dir, pkg_zip = win_pkg.build_portable_package(PKG_DIR, dist_out)
    launcher_exe = bundle_dir / "Colab桌宠.exe"
    assert launcher_exe.is_file() and launcher_exe.stat().st_size > 4096
    assert pkg_zip.is_file() and pkg_zip.stat().st_size > 10000

    # 运行编译出的 Colab桌宠.exe --self-test 验证 PE 可执行与同目录完整性自检
    st_proc = subprocess.run(
        [str(launcher_exe), "--self-test"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=15,
    )
    assert st_proc.returncode == 0, st_proc.stderr or st_proc.stdout
    assert "COLAB_PET_LAUNCHER_SELF_TEST_OK:1" in st_proc.stdout

    # 校验生成的 zip 绝不含真实凭据配置文件、个人状态或日志
    with zipfile.ZipFile(pkg_zip, "r") as zf:
        arc_names = zf.namelist()
        assert any(n.endswith("Colab桌宠.exe") for n in arc_names)
        assert any(n.endswith("stack_manifest.json") for n in arc_names)
        assert any(n.endswith("requirements-lock.txt") for n in arc_names)
        assert not any(".cf_tunnel_secret.json" in n for n in arc_names)
        assert not any("pet_state.json" in n for n in arc_names)
        assert not any(n.lower().endswith(".log") for n in arc_names)


# =========================================================================
# 7. /v1/messages 心跳后 Anthropic 错误帧 & 双栖运行态热重载 LLM 调参测试
# =========================================================================


def test_sse_anthropic_error_frame_and_live_llm_reconfigure(
    isolated_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    import socket
    import urllib.request
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    # 7a. 验证独立 SSE 网关模块在发出 : keepalive 后上游返回 400 上下文超限，对 /v1/messages 输出 Anthropic event: error
    gw_script = PKG_DIR / "colab_sse_gateway.py"
    assert gw_script.is_file()

    err_payload = json.dumps(
        {
            "error": {
                "code": 400,
                "message": "Context size has been exceeded.",
                "type": "exceed_context_size_error",
            }
        }
    ).encode("utf-8")

    class _MockErrHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, _fmt: str, *_args: Any) -> None:
            return

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                body = b'{"status":"ok"}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_response(404)
            self.end_headers()

        def do_POST(self) -> None:  # noqa: N802
            cl = int(self.headers.get("Content-Length") or "0")
            if cl > 0:
                self.rfile.read(cl)
            # 故意延迟超过测试心跳 0.15s，触发网关先下发 200 SSE + : keepalive
            time.sleep(0.32)
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(err_payload)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(err_payload)

    up_srv = ThreadingHTTPServer(("127.0.0.1", 0), _MockErrHandler)
    up_port = up_srv.server_port
    threading.Thread(target=up_srv.serve_forever, daemon=True).start()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s_tmp:
        s_tmp.bind(("127.0.0.1", 0))
        gw_port = s_tmp.getsockname()[1]

    gw_env = os.environ.copy()
    gw_env.update(
        {
            "SSE_GW_HOST": "127.0.0.1",
            "SSE_GW_PORT": str(gw_port),
            "SSE_UPSTREAM_HOST": "127.0.0.1",
            "SSE_UPSTREAM_PORT": str(up_port),
            "SSE_HEARTBEAT_SEC": "0.12",
            "SSE_BASE_MODEL": "Qwen3.8-27B-abliterated",
        }
    )
    gw_proc = subprocess.Popen(
        [sys.executable, str(gw_script)],
        env=gw_env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    gw_base = f"http://127.0.0.1:{gw_port}"
    try:
        for _ in range(40):
            try:
                with urllib.request.urlopen(f"{gw_base}/health", timeout=0.5) as r:
                    if r.status == 200:
                        break
            except Exception:
                time.sleep(0.05)

        anth_req = urllib.request.Request(
            f"{gw_base}/v1/messages",
            data=json.dumps({"model": "Qwen3.8-27B-abliterated", "messages": [{"role": "user", "content": "hi"}], "stream": True}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(anth_req, timeout=4.0) as r:
            anth_sse = r.read().decode("utf-8", errors="replace")
        assert ": keepalive" in anth_sse
        assert "event: error" in anth_sse
        assert '"type": "invalid_request_error"' in anth_sse
        assert "Context size has been exceeded." in anth_sse
        assert "data: [DONE]" not in anth_sse

        oai_req = urllib.request.Request(
            f"{gw_base}/v1/chat/completions",
            data=json.dumps({"model": "Qwen3.8-27B-abliterated", "messages": [{"role": "user", "content": "hi"}], "stream": True}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(oai_req, timeout=4.0) as r:
            oai_sse = r.read().decode("utf-8", errors="replace")
        assert ": keepalive" in oai_sse and "Context size has been exceeded." in oai_sse and "data: [DONE]" in oai_sse
    finally:
        gw_proc.terminate()
        try:
            gw_proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            gw_proc.kill()
        up_srv.shutdown()
        up_srv.server_close()

    # 7b. 验证桌宠双栖运行态在线热调参（保留 minimax+qwen_27b 与 Comfy 隧道，传 --cohost-comfy）
    tools = _prepare_fake_tools(isolated_env.tmp)
    monkeypatch.setattr(cb, "COLAB_EXE_DEFAULT", tools.colab_exe)
    monkeypatch.setattr(cb, "COLAB_PY", tools.colab_py)
    monkeypatch.setattr(cb, "RCLONE_EXE", tools.rclone_exe)
    monkeypatch.setattr(cb, "RCLONE_CONF", tools.rclone_conf)

    backend = cb.ColabPetBackend(
        state_path=isolated_env.tmp / "reconf_state.json",
        download_root=isolated_env.tmp / "reconf_dl",
        cf_config_path=isolated_env.cf_file,
    )
    comfy_url = "https://comfy.example.com"
    llm_url = "https://llm.example.com"
    csec.save_cf_tunnel_settings(
        isolated_env.cf_file,
        comfy_url=comfy_url,
        llm_url=llm_url,
        tunnel_token="eyJhIjoiZHVtbXkiLCJ0IjoiZHVtbXkiLCJzIjoiZHVtbXkifQ==",
    )
    ep_id = "ep-g4-dual-live"
    with backend._lock:
        backend.state["account_profiles"] = [{"id": "a1", "label": "U1", "email_masked": "u***1@g.com"}]
        backend.state["prefs"]["active_account_id"] = "a1"
        backend.state["prefs"]["boot_profiles"] = ["minimax", "qwen_27b"]
        backend.state["prefs"]["boot_profile"] = "minimax"
        backend.state["prefs"]["boot_gpu"] = "G4"
        backend.state["last_assignments_count"] = 1
        backend.state["last_sessions"] = [
            {"name": "comfy", "endpoint": ep_id, "accelerator": "G4", "machine_shape": "STANDARD", "orphan": False}
        ]
        backend.state["tunnel_cache"][ep_id] = comfy_url
        backend._service_meta_by_ep[ep_id] = {
            "kind": "dual",
            "llm_profile": "qwen_27b",
            "model_variant": "27b",
            "ctx_size": 65536,
            "quant": "UD-Q4_K_XL",
            "comfy_tunnel_url": comfy_url,
            "llm_tunnel_url": llm_url,
            "comfy_profiles": ["minimax"],
        }
    with backend._comfy_lock:
        backend._comfy_state.update(
            {
                "online": True,
                "comfy_online": True,
                "llm_online": True,
                "service_kind": "dual",
                "active_profiles": ["minimax", "qwen_27b"],
                "tunnel_url": comfy_url,
                "llm_tunnel_url": llm_url,
                "ctx_size": 65536,
                "quant": "UD-Q4_K_XL",
                "model_name": "Qwen3.8-27B-abliterated",
            }
        )

    seen_cmd: list[str] = []
    done_evt = threading.Event()
    backend._event_cb = lambda e: done_evt.set() if e.get("type") in ("deploy_done", "deploy_error") else None

    class _MockReconfPopen:
        def __init__(self, cmd: list[str], **_kwargs: Any) -> None:
            seen_cmd.extend(str(x) for x in cmd)
            self.pid = None
            self.stdout = io.StringIO(
                f"LLM_TUNNEL_URL: {llm_url}\nSERVICE_TUNNEL_URL: {llm_url}\n"
            )

        def wait(self) -> int:
            return 0

    def _mock_fast_read(_ep: str, _s: str):
        with backend._lock:
            backend._service_meta_by_ep[ep_id] = {
                "kind": "dual",
                "llm_profile": "qwen_27b",
                "model_variant": "27b",
                "ctx_size": 262144,
                "quant": "UD-Q4_K_XL",
                "tunnel_url": comfy_url,
                "comfy_tunnel_url": comfy_url,
                "llm_tunnel_url": llm_url,
                "comfy_profiles": ["minimax"],
            }
        return "ok", ep_id, comfy_url

    with (
        patch.object(cb.subprocess, "Popen", _MockReconfPopen),
        patch.object(backend, "fetch_live_status", return_value=backend.get_cached_snapshot()),
        patch.object(backend, "_fast_read_tunnel_url", side_effect=_mock_fast_read),
    ):
        r_reconf = backend.reconfigure_running_llm({"ctx": 262144, "quant": "UD-Q4_K_XL", "mmproj": True})
        assert r_reconf["ok"] is True, r_reconf
        assert done_evt.wait(timeout=5.0)

    deadline = time.monotonic() + 2.0
    while backend._deploy_lock.locked() and time.monotonic() < deadline:
        time.sleep(0.02)

    assert "--profile" in seen_cmd and "qwen_27b" in seen_cmd, seen_cmd
    assert "--cohost-comfy" in seen_cmd, seen_cmd
    assert "--llm-ctx" in seen_cmd and "262144" in seen_cmd, seen_cmd
    assert "--recreate" not in seen_cmd, seen_cmd
    assert backend.state["prefs"]["boot_profiles"] == ["minimax", "qwen_27b"]
    assert backend.state["prefs"]["boot_llm27_ctx"] == 262144
    assert backend._comfy_state["service_kind"] == "dual"
    assert backend._comfy_state["tunnel_url"] == comfy_url
    assert backend._comfy_state["ctx_size"] == 262144


# =========================================================================
# 8. 轻量化及安全加固防线综合回归测试 (.aria2 / Fail-Closed / CORS / L4双栖 / Frozen)
# =========================================================================


def test_hardened_safety_and_lightweight_boundaries(
    isolated_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    import asyncio
    import urllib.request
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer

    import colab_pet_bridge_node as bridge_node
    import colab_remote_common as rcom
    from colab_core import colab_client

    tmp = isolated_env.tmp

    # 8a. .aria2 控制文件拦截：即使目标文件字节数已满额，存在同名 .aria2 就必须判定为未完成
    full_file = tmp / "model.safetensors"
    full_file.write_bytes(b"0" * 1024)
    assert rcom.is_complete_file(full_file, exact_bytes=1024) is True
    (tmp / "model.safetensors.aria2").write_text("partial", encoding="utf-8")
    assert rcom.is_complete_file(full_file, exact_bytes=1024) is False

    # 8b. ComfyUI 公网桥接无密钥时对公网请求 Fail-Closed 返回 503，而纯本机回环仍允许放行
    monkeypatch.setenv("COMFY_AUTH_KEY_FILE", str(tmp / "missing_key"))
    monkeypatch.setattr(bridge_node, "_SERVICE_META_PATH", tmp / "missing_meta.json")

    async def _check_fail_closed() -> None:
        mw = bridge_node.create_comfy_auth_middleware()
        app = web.Application(middlewares=[mw])

        async def _stats(_r: web.Request) -> web.Response:
            return web.json_response({"system": {}})

        app.router.add_get("/system_stats", _stats)
        async with TestClient(TestServer(app)) as client:
            r_pub = await client.get("/system_stats", headers={"Host": "demo.trycloudflare.com", "Cf-Connecting-Ip": "198.51.100.9"})
            assert r_pub.status == 503
            r_local = await client.get("/system_stats")
            assert r_local.status == 200

    asyncio.run(_check_fail_closed())

    # 8c. 独立 SSE 网关模块 OPTIONS 预检与 /v1/models 合成响应必须保留 CORS 头
    import colab_sse_gateway as sse_gw

    gw_src = (PKG_DIR / "colab_sse_gateway.py").read_text(encoding="utf-8")
    assert "_cors_header_block" in gw_src and 'if method == "OPTIONS":' in gw_src

    # 直接导入 colab_sse_gateway 实测真实 TCP HTTP OPTIONS 与 /v1/models CORS 响应头
    async def _run_sse_cors_check() -> None:
        import socket

        def _free_port() -> int:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", 0))
                return int(s.getsockname()[1])

        up_port = _free_port()
        gw_port = _free_port()

        async def _up_models(_r: web.Request) -> web.Response:
            return web.json_response({"data": [{"id": "Qwen3.8-27B-abliterated"}]})

        async def _up_slow_sse(_r: web.Request) -> web.StreamResponse:
            await asyncio.sleep(0.25)
            resp = web.StreamResponse(status=200, headers={"Content-Type": "text/event-stream"})
            await resp.prepare(_r)
            await resp.write(b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n')
            await resp.write_eof()
            return resp

        up_app = web.Application()
        up_app.router.add_get("/v1/models", _up_models)
        up_app.router.add_post("/v1/chat/completions", _up_slow_sse)
        runner = web.AppRunner(up_app)
        await runner.setup()
        await web.TCPSite(runner, "127.0.0.1", up_port).start()

        monkeypatch.setattr(sse_gw, "LISTEN_PORT", gw_port)
        monkeypatch.setattr(sse_gw, "UPSTREAM_PORT", up_port)
        monkeypatch.setattr(sse_gw, "BASE_MODEL", "Qwen3.8-27B-abliterated")
        monkeypatch.setattr(sse_gw, "HEARTBEAT_SEC", 0.08)
        gw_srv = await asyncio.start_server(sse_gw._handle_client, "127.0.0.1", gw_port)
        try:
            def _do_opts() -> tuple[int, dict[str, str]]:
                req = urllib.request.Request(
                    f"http://127.0.0.1:{gw_port}/v1/chat/completions",
                    method="OPTIONS",
                    headers={"Origin": "https://chat.example.com"},
                )
                with urllib.request.urlopen(req, timeout=3.0) as resp:
                    return resp.status, {k.lower(): v for k, v in resp.headers.items()}

            def _do_models() -> tuple[int, dict[str, str], dict[str, Any]]:
                req = urllib.request.Request(
                    f"http://127.0.0.1:{gw_port}/v1/models",
                    headers={"Origin": "https://chat.example.com"},
                )
                with urllib.request.urlopen(req, timeout=3.0) as resp:
                    return resp.status, {k.lower(): v for k, v in resp.headers.items()}, json.loads(resp.read().decode("utf-8"))

            def _do_slow_stream() -> tuple[int, dict[str, str], str]:
                body = json.dumps({"model": "Qwen3.8-27B-abliterated-low", "stream": True, "messages": [{"role": "user", "content": "hi"}]}).encode("utf-8")
                req = urllib.request.Request(
                    f"http://127.0.0.1:{gw_port}/v1/chat/completions",
                    data=body,
                    method="POST",
                    headers={"Origin": "https://chat.example.com", "Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=5.0) as resp:
                    return resp.status, {k.lower(): v for k, v in resp.headers.items()}, resp.read().decode("utf-8", errors="replace")

            # 默认未配置 SSE_ALLOWED_ORIGINS 时返回通配 * 且绝不附带 Allow-Credentials
            monkeypatch.delenv("SSE_ALLOWED_ORIGINS", raising=False)
            monkeypatch.delenv("SSE_CORS_ALLOW_CREDENTIALS", raising=False)

            st_opt, h_opt = await asyncio.to_thread(_do_opts)
            assert st_opt == 204
            assert h_opt.get("access-control-allow-origin") == "*"
            assert "access-control-allow-credentials" not in h_opt

            st_mod, h_mod, body_mod = await asyncio.to_thread(_do_models)
            assert st_mod == 200
            assert h_mod.get("access-control-allow-origin") == "*"
            assert "access-control-allow-credentials" not in h_mod
            assert any(m.get("id", "").endswith("-nothink") for m in body_mod.get("data") or [])

            st_sse, h_sse, body_sse = await asyncio.to_thread(_do_slow_stream)
            assert st_sse == 200
            assert h_sse.get("access-control-allow-origin") == "*"
            assert "access-control-allow-credentials" not in h_sse
            assert ": keepalive" in body_sse and "data: [DONE]" in body_sse

            # 配置 SSE_ALLOWED_ORIGINS 白名单后精确回显命中 Origin 与 Vary: Origin（默认仍不附带 Credentials）
            monkeypatch.setenv("SSE_ALLOWED_ORIGINS", "https://chat.example.com")

            st_opt_wl, h_opt_wl = await asyncio.to_thread(_do_opts)
            assert st_opt_wl == 204
            assert h_opt_wl.get("access-control-allow-origin") == "https://chat.example.com"
            assert h_opt_wl.get("vary") == "Origin"
            assert "access-control-allow-credentials" not in h_opt_wl

            st_mod_wl, h_mod_wl, _ = await asyncio.to_thread(_do_models)
            assert st_mod_wl == 200
            assert h_mod_wl.get("access-control-allow-origin") == "https://chat.example.com"
            assert h_mod_wl.get("vary") == "Origin"
            assert "access-control-allow-credentials" not in h_mod_wl

            st_sse_wl, h_sse_wl, body_sse_wl = await asyncio.to_thread(_do_slow_stream)
            assert st_sse_wl == 200
            assert h_sse_wl.get("access-control-allow-origin") == "https://chat.example.com"
            assert h_sse_wl.get("vary") == "Origin"
            assert "access-control-allow-credentials" not in h_sse_wl
            assert ": keepalive" in body_sse_wl and "data: [DONE]" in body_sse_wl
            monkeypatch.delenv("SSE_ALLOWED_ORIGINS", raising=False)
        finally:
            gw_srv.close()
            await gw_srv.wait_closed()
            await runner.cleanup()

    asyncio.run(_run_sse_cors_check())

    # 8d. L4 (24GB) + 双栖在 Preflight 必须严格拦截 (ok=False)
    tools = _prepare_fake_tools(tmp)
    (tmp / "cf.json").write_text("{}", encoding="utf-8")
    l4_dual = core_preflight.run_preflight_checks(
        mode="fast",
        profiles=["anima", "qwen_27b"],
        gpu="L4",
        tool_paths=core_preflight.ToolPaths(tools.colab_exe, tools.colab_py, tools.rclone_exe, tools.rclone_conf),
        download_dir=tmp / "dl",
        cf_config_path=tmp / "cf.json",
        pkg_dir=PKG_DIR,
    )
    assert l4_dual.ok is False and any("L4" in it.summary for it in l4_dual.blocking_items())

    # 8e. PyInstaller frozen=True 态下 _ctl_available() 放行，且 _ctl_cmd_prefix 使用 --internal-ctl
    with patch.object(cb.sys, "frozen", True, create=True), patch.object(cb, "CTL_SCRIPT", tmp / "not_exist.py"):
        assert cb._ctl_available() is True
        assert cb._ctl_cmd_prefix() == [cb.sys.executable, "--internal-ctl"]

    # 8f. collect_local_py_deps 自动携带独立 JS、SSE 网关与桥接节点（含来自 Temp 临时包装文件）
    deps_anima = colab_client.collect_local_py_deps(PKG_DIR / "colab_bootstrap_anima.py")
    assert {
        "colab_remote_common.py",
        "colab_pet_bridge_node.py",
        "colab_pet_direct_canvas.js",
    }.issubset(deps_anima.keys())
    deps_qwen = colab_client.collect_local_py_deps(PKG_DIR / "colab_bootstrap_qwen_llm.py")
    assert "colab_sse_gateway.py" in deps_qwen
    temp_task = tmp / "remote_wrap.py"
    temp_task.write_text("from colab_remote_common import start_cloudflared_tunnel\n", encoding="utf-8")
    deps_tmp = colab_client.collect_local_py_deps(temp_task)
    assert {"colab_remote_common.py", "colab_pet_bridge_node.py", "colab_pet_direct_canvas.js"}.issubset(deps_tmp.keys())

    # 8g. 模拟 REMOTE_RUNNER_TEMPLATE（无 __file__、依赖落于临时 /content 目录）执行全部 4 套自举脚本顶层代码不报 NameError
    remote_dir = tmp / "fake_content"
    remote_dir.mkdir(parents=True, exist_ok=True)
    for dep_name, dep_txt in deps_qwen.items():
        (remote_dir / dep_name).write_text(dep_txt, encoding="utf-8")
    canonical_bridge = (PKG_DIR / "colab_pet_bridge_node.py").read_text(encoding="utf-8").strip()
    old_cwd = Path.cwd()
    sys.path.insert(0, str(remote_dir))
    saved_rcm = sys.modules.pop("colab_remote_common", None)
    try:
        os.chdir(remote_dir)
        for boot_name in (
            "colab_bootstrap_anima.py",
            "colab_bootstrap_minimax_h3.py",
            "colab_bootstrap_drama3.py",
            "colab_bootstrap_qwen_llm.py",
        ):
            boot_src = (PKG_DIR / boot_name).read_text(encoding="utf-8")
            ns: dict[str, object] = {"__name__": "__colab_remote_test__"}
            exec(compile(boot_src, boot_name, "exec"), ns)
            if "_PET_BRIDGE_INIT_PY" in ns:
                assert ns["_PET_BRIDGE_INIT_PY"] == canonical_bridge
    finally:
        os.chdir(old_cwd)
        if str(remote_dir) in sys.path:
            sys.path.remove(str(remote_dir))
        sys.modules.pop("colab_remote_common", None)
        if saved_rcm is not None:
            sys.modules["colab_remote_common"] = saved_rcm

    # 8h. 自定义固定域名经 _consume_deploy_line 可被动态正则捕获（不局限固定内置域名）
    custom_cf = tmp / "custom_cf.json"
    custom_cf.write_text(
        json.dumps({"comfy_url": "https://my-comfy.customdom.example", "llm_url": "https://my-llm.customdom.example"}),
        encoding="utf-8",
    )
    be_custom = cb.ColabPetBackend(
        state_path=tmp / "custom_state.json",
        download_root=tmp / "custom_dl",
        cf_config_path=custom_cf,
    )
    be_custom._consume_deploy_line("DEPLOY_PHASE 1/2: qwen_27b")
    be_custom._consume_deploy_line("LLM_TUNNEL_URL: https://my-llm.customdom.example")
    be_custom._consume_deploy_line("DEPLOY_PHASE 2/2: anima")
    cap_url = be_custom._consume_deploy_line("COMFYUI_TUNNEL_URL: https://my-comfy.customdom.example")
    assert cap_url == "https://my-comfy.customdom.example"
    assert be_custom._deploy_captured_llm_tunnel == "https://my-llm.customdom.example"

    # 8i. 验证 PetBridgeApi 全部前端暴露方法的 pywebview 参数签名反射与表驱动透传零污染
    import inspect
    import re as _re
    import pet_app as pa

    ui_src = (PKG_DIR / "colab_pet" / "ui.html").read_text(encoding="utf-8")
    ui_methods = sorted(set(_re.findall(r"window\.pywebview\.api\.([a-zA-Z0-9_]+)", ui_src)))
    bridge = pa.PetBridgeApi(be_custom, threading.Event())
    for m_name in ui_methods:
        attr = getattr(bridge, m_name, None)
        assert callable(attr) and inspect.ismethod(attr), f"UI 调用的方法缺失或非绑定方法: {m_name}"
        arg_names = inspect.getfullargspec(attr).args
        assert arg_names[0] == "self" and not any(a.startswith("_") for a in arg_names[1:]), (
            f"pywebview 反射参数受内部参数污染: {m_name} -> {arg_names}"
        )
    assert inspect.getfullargspec(bridge.get_state).args == ["self"]
    assert inspect.getfullargspec(bridge.start_instance).args == ["self", "config"]
    assert inspect.getfullargspec(bridge.switch_colab_account).args == ["self", "acct_id"]
    assert inspect.getfullargspec(bridge.stop_session_instance).args == ["self", "session"]

    # 8j. 验证含空格路径在资源管理器精确定位时 `/select,` 绝不被包进引号内部（避免误打开“文档”目录）
    spaced_dir = tmp / "Space Folder Test" / "outputs"
    spaced_dir.mkdir(parents=True, exist_ok=True)
    spaced_png = spaced_dir / "demo_00001_20261117-050000.png"
    spaced_png.write_bytes(b"PNG")
    cmdline = cb._build_explorer_select_cmdline(spaced_png)
    assert cmdline.startswith('explorer.exe /select,"') and cmdline.endswith('"')
    assert str(spaced_png.resolve()) in cmdline
    revealed: list[Path] = []
    with be_custom._comfy_lock:
        be_custom._last_downloaded_files = [spaced_png]
    with patch.object(cb, "_reveal_path_in_explorer", side_effect=lambda p: revealed.append(Path(p))):
        pull_res = be_custom.pull_or_open_latest_outputs()
    assert pull_res["ok"] is True and revealed == [spaced_png.resolve()]


# =========================================================================
# 9. Drama 3 伴随工作流收集加载、安全解压与 Qwen LLM 密钥文件防命令行泄露测试
# =========================================================================


def test_drama3_companion_deps_qwen_key_file_and_safe_extract(
    isolated_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """验证 Drama 3 工作流按伴随资产收集与部署、公共安全解压及 llama-server --api-key-file 适配。"""
    import io as _io
    import tarfile

    import colab_bootstrap_drama3 as boot_d3
    import colab_bootstrap_qwen_llm as boot_llm
    import colab_remote_common as rcom
    from colab_core import colab_client

    tmp = isolated_env.tmp

    # 9a. collect_local_py_deps：Drama 3 自动携带两份 JSON，其他生态与敏感 JSON 绝不误收集
    deps_d3 = colab_client.collect_local_py_deps(PKG_DIR / "colab_bootstrap_drama3.py")
    assert {
        "colab_remote_common.py",
        "colab_pet_bridge_node.py",
        "colab_pet_direct_canvas.js",
        "drama3_voice_clone.json",
        "drama3_text_to_speech.json",
    }.issubset(deps_d3.keys())
    deps_anima = colab_client.collect_local_py_deps(PKG_DIR / "colab_bootstrap_anima.py")
    assert "drama3_voice_clone.json" not in deps_anima and "drama3_text_to_speech.json" not in deps_anima

    evil_script = tmp / "evil_ref.py"
    evil_script.write_text('a = ".cf_tunnel_secret.json"\nb = "pet_state.json"\n', encoding="utf-8")
    deps_evil = colab_client.collect_local_py_deps(evil_script)
    assert ".cf_tunnel_secret.json" not in deps_evil and "pet_state.json" not in deps_evil

    # 9b. load_companion_text 兼容本地 workflows/drama3/ 与远端扁平 /content/ 目录，且每次返回独立解析对象
    local_clone_txt = rcom.load_companion_text("drama3_voice_clone.json", PKG_DIR)
    assert local_clone_txt and "Drama3Loader" in local_clone_txt

    fake_comfy = tmp / "ComfyUI"
    fake_out = tmp / "comfy_out"
    (fake_comfy / "input").mkdir(parents=True, exist_ok=True)
    (fake_comfy / "input" / "custom_ref_01.wav").write_bytes(b"RIFF")
    monkeypatch.setattr(boot_d3, "COMFY_DIR", fake_comfy)
    monkeypatch.setattr(boot_d3, "COMFY_OUT", fake_out)
    monkeypatch.setattr(boot_d3, "RCLONE_CONF", tmp / "missing_rclone.conf")
    boot_d3.install_workflows_and_refs()

    wf_dir = fake_comfy / "user" / "default" / "workflows"
    clone_deployed = json.loads((wf_dir / "drama3_voice_clone.json").read_text(encoding="utf-8"))
    tts_deployed = json.loads((wf_dir / "drama3_text_to_speech.json").read_text(encoding="utf-8"))
    ref_node = next(n for n in clone_deployed["nodes"] if n.get("type") == "Drama3LoadReferenceAudio")
    assert ref_node["widgets_values"][0] == "custom_ref_01.wav"
    assert "nodes" in tts_deployed

    # 验证远端无 workflows/drama3 子目录时从同级平铺目录读取成功，且二次部署回默认音频不受上次修改污染
    flat_dir = tmp / "flat_content"
    flat_dir.mkdir(parents=True, exist_ok=True)
    for jname in ("drama3_voice_clone.json", "drama3_text_to_speech.json"):
        (flat_dir / jname).write_text(deps_d3[jname], encoding="utf-8")
    (fake_comfy / "input" / "custom_ref_01.wav").unlink()
    monkeypatch.setattr(boot_d3, "_BRIDGE_BASE_DIR", flat_dir)
    boot_d3.install_workflows_and_refs()
    clone_default = json.loads((wf_dir / "drama3_voice_clone.json").read_text(encoding="utf-8"))
    ref_node_2 = next(n for n in clone_default["nodes"] if n.get("type") == "Drama3LoadReferenceAudio")
    assert ref_node_2["widgets_values"][0].startswith("main_01_")

    # 9c. safe_extract_tar_gz 支持正常文件与相对符号链接，拦截路径穿越与越界符号链接
    good_tar = tmp / "good.tar.gz"
    with tarfile.open(good_tar, "w:gz") as tf:
        payload = b"bin-data"
        ti = tarfile.TarInfo("bin/llama-server")
        ti.size = len(payload)
        ti.mode = 0o755
        tf.addfile(ti, _io.BytesIO(payload))
    good_dst = tmp / "extract_ok"
    rcom.safe_extract_tar_gz(good_tar, good_dst)
    assert (good_dst / "bin" / "llama-server").read_bytes() == b"bin-data"

    bad_tar = tmp / "traversal.tar.gz"
    with tarfile.open(bad_tar, "w:gz") as tf:
        ti_bad = tarfile.TarInfo("../escape.txt")
        ti_bad.size = 3
        tf.addfile(ti_bad, _io.BytesIO(b"bad"))
    with pytest.raises(RuntimeError, match="越界"):
        rcom.safe_extract_tar_gz(bad_tar, tmp / "extract_bad")

    bad_sym_tar = tmp / "bad_sym.tar.gz"
    with tarfile.open(bad_sym_tar, "w:gz") as tf:
        ti_sym = tarfile.TarInfo("bin/libx.so")
        ti_sym.type = tarfile.SYMTYPE
        ti_sym.linkname = "../../etc/passwd"
        tf.addfile(ti_sym)
    with pytest.raises(RuntimeError, match="越界"):
        rcom.safe_extract_tar_gz(bad_sym_tar, tmp / "extract_bad_sym")

    # 9d. build_auth_and_args：0600 落盘密钥文件 + --help 探测 --api-key-file 与回退
    key_file = tmp / "llama_api_key"
    meta_file = tmp / "service_meta.json"
    fake_bin = tmp / "llama-server"
    fake_bin.write_text("", encoding="utf-8")
    fake_shard = tmp / "model.gguf"
    fake_shard.write_bytes(b"GGUF")
    monkeypatch.setattr(boot_llm, "LLAMA_API_KEY_FILE", key_file)
    monkeypatch.setattr(boot_llm, "SERVICE_META_FILE", meta_file)
    monkeypatch.setattr(boot_llm, "LLAMA_DIR", tmp)
    monkeypatch.setattr(boot_llm, "RAW_API_KEY", "sk-pet-cmdline-guard-123456")

    with patch.object(
        boot_llm.subprocess,
        "run",
        return_value=subprocess.CompletedProcess([str(fake_bin), "--help"], 0, stdout="--api-key-file FNAME\n", stderr=""),
    ):
        api_key_1, cmd_1 = boot_llm.build_auth_and_args(fake_bin, fake_shard, None, "UD-Q4_K_XL", 96.0)
    assert api_key_1 == "sk-pet-cmdline-guard-123456"
    assert key_file.read_text(encoding="utf-8").strip() == api_key_1
    assert "--api-key-file" in cmd_1 and str(key_file) in cmd_1
    assert "--api-key" not in cmd_1 and api_key_1 not in cmd_1

    with patch.object(
        boot_llm.subprocess,
        "run",
        return_value=subprocess.CompletedProcess([str(fake_bin), "--help"], 0, stdout="--api-key KEY\n", stderr=""),
    ):
        api_key_2, cmd_2 = boot_llm.build_auth_and_args(fake_bin, fake_shard, None, "UD-Q4_K_XL", 96.0)
    assert "--api-key" in cmd_2 and api_key_2 in cmd_2
    assert "--api-key-file" not in cmd_2


# =========================================================================
# 10. 零 Pillow 运行态/打包、PNG 元数据解析、rclone 合并、损坏状态备份、路径注入拦截与脱敏回归
# =========================================================================


def test_lightweight_zero_pil_and_security_hardening_regressions(
    isolated_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """覆盖零 PIL 导入、PNG tEXt/zTXt/iTXt 解析、rclone 节保留、损坏状态备份、prefs 路径拦截与脱敏增强。"""
    import ast
    import struct
    import zlib

    from colab_core import accounts as core_accounts
    from colab_core import file_sync as core_file_sync

    tmp = isolated_env.tmp

    # 10a. 运行态与打包模块零 PIL 导入，且纯标准库 extract_png_comfy_workflow 兼容 tEXt / zTXt / iTXt
    for rel_path in ("colab_core/file_sync.py", "colab_pet/pet_app.py", "build_windows_package.py"):
        tree = ast.parse((PKG_DIR / rel_path).read_text(encoding="utf-8"), filename=rel_path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name != "PIL" and not alias.name.startswith("PIL."), f"{rel_path} 禁止导入 PIL"
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert node.module != "PIL" and not node.module.startswith("PIL."), f"{rel_path} 禁止从 PIL 导入"

    def _png_chunk(ctype: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(ctype + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + ctype + data + struct.pack(">I", crc)

    def _build_png(chunks: list[tuple[bytes, bytes]]) -> bytes:
        ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 6, 0, 0, 0)
        idat = zlib.compress(b"\x00\x00\x00\x00\x00")
        parts = [b"\x89PNG\r\n\x1a\n", _png_chunk(b"IHDR", ihdr)]
        parts.extend(_png_chunk(ct, dt) for ct, dt in chunks)
        parts.extend([_png_chunk(b"IDAT", idat), _png_chunk(b"IEND", b"")])
        return b"".join(parts)

    wf_json = json.dumps({"nodes": [{"id": 1, "type": "KSampler"}], "links": []}, ensure_ascii=False)
    wf_zh_json = json.dumps({"nodes": [{"id": 2, "title": "中文节点"}], "links": []}, ensure_ascii=False)
    prompt_json = json.dumps({"1": {"class_type": "KSampler", "inputs": {"seed": 7}}}, ensure_ascii=False)

    png_text = tmp / "meta_text.png"
    png_text.write_bytes(_build_png([(b"tEXt", b"workflow\x00" + wf_json.encode("latin-1"))]))
    assert core_file_sync.extract_png_comfy_workflow(png_text) == wf_json

    png_ztxt = tmp / "meta_ztxt.png"
    png_ztxt.write_bytes(_build_png([(b"zTXt", b"prompt\x00\x00" + zlib.compress(prompt_json.encode("utf-8")))]))
    assert core_file_sync.extract_png_comfy_workflow(png_ztxt) == prompt_json

    png_itxt_comp = tmp / "meta_itxt_comp.png"
    png_itxt_comp.write_bytes(
        _build_png([(b"iTXt", b"workflow\x00\x01\x00\x00\x00" + zlib.compress(wf_zh_json.encode("utf-8")))])
    )
    assert core_file_sync.extract_png_comfy_workflow(png_itxt_comp) == wf_zh_json

    png_itxt_raw = tmp / "meta_itxt_raw.png"
    png_itxt_raw.write_bytes(
        _build_png([(b"iTXt", b"prompt\x00\x00\x00\x00\x00" + prompt_json.encode("utf-8"))])
    )
    assert core_file_sync.extract_png_comfy_workflow(png_itxt_raw) == prompt_json

    not_png = tmp / "bad.png"
    not_png.write_bytes(b"NOT_A_PNG_HEADER")
    assert core_file_sync.extract_png_comfy_workflow(not_png) is None

    # 10b. validate_and_import_rclone_conf 合并导入时保留原有 [other_remote] 并生成 .conf.bak
    appdata_dir = tmp / "appdata"
    dst_conf = appdata_dir / "rclone" / "rclone.conf"
    dst_conf.parent.mkdir(parents=True, exist_ok=True)
    orig_conf_txt = (
        "[other_remote]\n"
        "type = s3\n"
        "provider = AWS\n\n"
        "[gdrive_save]\n"
        "type = drive\n"
        'token = {"access_token":"old_tok"}\n'
    )
    dst_conf.write_text(orig_conf_txt, encoding="utf-8")
    src_conf = tmp / "import_rclone.conf"
    src_conf.write_text(
        "[my_personal_drive]\n"
        "type = drive\n"
        "scope = drive\n"
        'token = {"access_token":"new_tok"}\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("APPDATA", str(appdata_dir))
    ok_imp, out_conf, err_imp = core_accounts.validate_and_import_rclone_conf(src_conf, package_dir=tmp)
    assert ok_imp is True, err_imp
    assert out_conf == dst_conf
    bak_conf = dst_conf.with_suffix(".conf.bak")
    assert bak_conf.is_file() and bak_conf.read_text(encoding="utf-8") == orig_conf_txt
    merged_txt = dst_conf.read_text(encoding="utf-8")
    assert "[other_remote]" in merged_txt and "provider = AWS" in merged_txt
    assert "[gdrive_save]" in merged_txt and "new_tok" in merged_txt and "old_tok" not in merged_txt

    # 10c. load_pet_state_dict 遇到损坏 JSON 时自动生成 .corrupt-* 备份文件并写告警日志
    bad_state_path = tmp / "pet_state_corrupt.json"
    bad_state_path.write_text("{corrupt json !!!", encoding="utf-8")
    loaded_def = core_runtime.load_pet_state_dict(
        bad_state_path,
        is_allowed_tunnel_fn=lambda _u: True,
        sanitize_label_fn=lambda lbl, fb: lbl or fb,
    )
    assert isinstance(loaded_def, dict) and "prefs" in loaded_def
    corrupt_backups = list(tmp.glob("pet_state_corrupt.corrupt-*.json"))
    assert corrupt_backups and corrupt_backups[0].read_text(encoding="utf-8") == "{corrupt json !!!"
    assert isolated_env.diag_log.is_file()

    # 10d. update_prefs 严禁前端通过 save_prefs/update_prefs 注入可执行工具路径
    tools = _prepare_fake_tools(tmp)
    monkeypatch.setattr(cb, "COLAB_EXE_DEFAULT", tools.colab_exe)
    monkeypatch.setattr(cb, "COLAB_PY", tools.colab_py)
    monkeypatch.setattr(cb, "RCLONE_EXE", tools.rclone_exe)
    monkeypatch.setattr(cb, "RCLONE_CONF", tools.rclone_conf)
    sec_backend = cb.ColabPetBackend(
        state_path=tmp / "prefs_guard_state.json",
        download_root=tmp / "prefs_guard_dl",
        cf_config_path=isolated_env.cf_file,
    )
    orig_tool_prefs = {
        k: sec_backend.state["prefs"][k]
        for k in ("colab_exe_path", "colab_py_path", "rclone_exe_path", "rclone_conf_path")
    }
    updated_prefs = sec_backend.update_prefs(
        {
            "colab_exe_path": "C:/evil/colab.exe",
            "colab_py_path": "C:/evil/python.exe",
            "rclone_exe_path": "C:/evil/rclone.exe",
            "rclone_conf_path": "C:/evil/rclone.conf",
            "on_top": False,
        }
    )
    assert updated_prefs["on_top"] is False
    for path_k, orig_v in orig_tool_prefs.items():
        assert updated_prefs[path_k] == orig_v
        assert sec_backend.state["prefs"][path_k] == orig_v

    # 10e. redact_secrets 覆盖 JSON/dict Authorization Bearer、GOCSPX-... 与 URL ?api_key=/token=/access_token=
    dirty_samples = (
        '{"Authorization": "Bearer json_bearer_secret_987654"}',
        "{'Authorization': 'Bearer py_bearer_secret_123456', 'access_token': 'py_acc_tok_secret_xyz', 'client_secret': 'py_cli_sec_secret_abc'}",
        "oauth_client_secret=GOCSPX-AbCdEfGhIjKlMnOpQrStUvWx1234",
        "https://api.example.com/v1/chat?foo=1&api_key=url_api_key_secret_999&bar=2",
        "https://api.example.com/ws?token=url_tok_secret_888#sec",
        "https://api.example.com/cb?access_token=url_acc_secret_777",
    )
    secret_fragments = (
        "json_bearer_secret_987654",
        "py_bearer_secret_123456",
        "py_acc_tok_secret_xyz",
        "py_cli_sec_secret_abc",
        "GOCSPX-AbCdEfGhIjKlMnOpQrStUvWx1234",
        "url_api_key_secret_999",
        "url_tok_secret_888",
        "url_acc_secret_777",
    )
    redacted_all = csec.redact_secrets("\n".join(dirty_samples))
    for frag in secret_fragments:
        assert frag not in redacted_all, f"redact_secrets 漏脱敏: {frag} -> {redacted_all}"
    assert "foo=1" in redacted_all and "bar=2" in redacted_all


# =========================================================================
# 11. Cloudflare Token 自动剥离与校验、错误摘要防掩盖、凭据防污染与 cmd_exec 护栏回归
# =========================================================================


def test_cf_token_extract_preflight_guard_and_deploy_error_priority(
    isolated_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    dummy_cf_tok = "eyJhIjoiZHVtbXkiLCJ0IjoiZHVtbXkiLCJzIjoiZHVtbXkifQ=="
    assert csec.extract_cf_tunnel_token(f"cloudflared.exe service install {dummy_cf_tok}") == dummy_cf_tok
    assert csec.extract_cf_tunnel_token(f'cloudflared tunnel run --token="{dummy_cf_tok}"') == dummy_cf_tok
    assert csec.is_valid_cf_tunnel_token(dummy_cf_tok) is True
    assert csec.is_valid_cf_tunnel_token("eyJFake.Payload.Sig") is False
    assert csec.is_valid_cf_tunnel_token("not-base64!") is False
    assert csec.is_valid_cf_tunnel_token("eyJmb28iOiJiYXIifQ==") is False

    # 保存时自动剥离前缀命令并入库
    csec.save_cf_tunnel_settings(
        isolated_env.cf_file,
        comfy_url="https://comfy.safe-test.example",
        tunnel_token=f"cloudflared.exe service install {dummy_cf_tok}",
    )
    loaded_cf = csec.load_cf_tunnel_settings(isolated_env.cf_file, include_secrets=True)
    assert loaded_cf["tunnel_token"] == dummy_cf_tok

    # Preflight 启动前阻断假 JWT Token，放行合法连接器 Token
    tools = _prepare_fake_tools(isolated_env.tmp)
    tp = core_preflight.ToolPaths(
        colab_exe=tools.colab_exe,
        colab_py=tools.colab_py,
        rclone_exe=tools.rclone_exe,
        rclone_conf=tools.rclone_conf,
    )
    ok_rep = core_preflight.run_preflight_checks(
        mode="fast",
        profiles=["anima"],
        gpu="G4",
        tool_paths=tp,
        download_dir=isolated_env.tmp,
        cf_config_path=isolated_env.cf_file,
        pkg_dir=PKG_DIR,
    )
    assert ok_rep.ok is True, ok_rep.format_text()

    csec.save_cf_tunnel_settings(
        isolated_env.cf_file,
        comfy_url="https://comfy.safe-test.example",
        tunnel_token="eyJFake.Payload.Sig",
    )
    bad_tok_rep = core_preflight.run_preflight_checks(
        mode="fast",
        profiles=["anima"],
        gpu="G4",
        tool_paths=tp,
        download_dir=isolated_env.tmp,
        cf_config_path=isolated_env.cf_file,
        pkg_dir=PKG_DIR,
    )
    assert bad_tok_rep.ok is False and "格式无效" in bad_tok_rep.first_blocking_message()

    # deploy 失败归因不被尾部 "[错误] 子进程退出码 1 (...)" 掩盖
    cf_fail = core_deploy.classify_deploy_failure(
        [
            "2026-11-18 ERR Provided Tunnel token is not valid.",
            "RuntimeError: cloudflared 命名隧道异常退出",
            "[错误] 子进程退出码 1 (C:\\Temp\\colab_runner.py)",
        ],
        "G4",
        pre_assignments=0,
        profile="Anima + MiniMax",
        recreate=True,
    )
    assert cf_fail["kind"] == "error" and "Token 无效" in cf_fail["message"], cf_fail

    gen_fail = core_deploy.classify_deploy_failure(
        [
            "RuntimeError: 远端下载权重 SHA256 校验失败",
            "[错误] 子进程退出码 1 (C:\\Temp\\colab_runner.py)",
        ],
        "G4",
        pre_assignments=0,
        profile="Anima",
        recreate=False,
    )
    assert "SHA256 校验失败" in gen_fail["reason"], gen_fail
    assert core_runtime._pick_meaningful_tail_error(["[错误] 子进程退出码 2 (colab.exe status)"]) == "[错误] 子进程退出码 2 (colab.exe status)"

    # pytest 未设沙箱时强拦截宿主机真实 Windows 凭据写入与删除
    if os.name == "nt":
        monkeypatch.delenv("COLAB_PET_MOCK_VAULT_PATH", raising=False)
        with pytest.raises(RuntimeError, match="COLAB_PET_MOCK_VAULT_PATH"):
            csec.VaultStore().set_secret("secret:test_guard", "should_block")
        with pytest.raises(RuntimeError, match="COLAB_PET_MOCK_VAULT_PATH"):
            csec.VaultStore().delete_secret("secret:test_guard")

    # colab_service_meta 窥探在 agent_tools 与 colab_ctl.cmd_exec 双层拦截
    assert core_agent_tools.check_remote_code_safety("cat /root/colab_service_meta.json", "bash")[0] is False
    monkeypatch.setattr(
        colab_ctl,
        "_exec_remote_py",
        lambda *_a, **_kw: (_ for _ in ()).throw(AssertionError("cmd_exec 拦截失败，意外触达远端执行")),
    )
    with pytest.raises(ValueError, match="安全护栏拦截"):
        colab_ctl.cmd_exec(
            argparse.Namespace(
                session="comfy",
                mode="bash",
                code="cat /root/colab_service_meta.json",
                timeout=30,
                account="",
            )
        )


# =========================================================================
# 12. 锁文件安装、PyInstaller 冻结包目录/控制台对齐与深检宿主依赖核验回归
# =========================================================================


def test_packaging_lockfile_pyinstaller_and_preflight_alignment(
    isolated_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """校验便携包锁文件安装、PyInstaller 批处理/spec 对齐、冻结路径解析及 deep preflight 宿主依赖核验。"""
    import ast
    import builtins
    import importlib.util
    import shutil

    import pet_app as pa

    tmp = isolated_env.tmp

    # 12a. 便携包安装批处理必须按 requirements-lock.txt 安装，且整包通过清单校验
    portable_dist = tmp / "portable_dist"
    bundle_dir, _ = win_pkg.build_portable_package(PKG_DIR, portable_dist)
    setup_bat = (bundle_dir / "一键环境自检与安装.bat").read_text(encoding="utf-8")
    assert "python -m pip install -r requirements-lock.txt" in setup_bat
    assert "requirements-lock.txt" in setup_bat
    assert "-r requirements.txt" not in setup_bat
    assert core_health.verify_manifest_and_assets(bundle_dir)["ok"] is True

    # 12b. PyInstaller 辅助批处理必须使用 start "" /b /wait 调用 --doctor 与 --export-diag
    pyi_dir = tmp / "pyi_bundle"
    pyi_dir.mkdir(parents=True, exist_ok=True)
    stale_setup = pyi_dir / "一键环境自检与安装.bat"
    stale_setup.write_text("legacy", encoding="utf-8")
    win_pkg.write_helper_scripts(pyi_dir, mode="pyinstaller")
    assert not stale_setup.exists()
    pyi_doctor = (pyi_dir / "一键环境自检.bat").read_text(encoding="utf-8")
    pyi_diag = (pyi_dir / "导出排障诊断包.bat").read_text(encoding="utf-8")
    assert 'start "" /b /wait "%~dp0Colab桌宠.exe" --doctor' in pyi_doctor
    assert 'start "" /b /wait "%~dp0Colab桌宠.exe" --export-diag' in pyi_diag

    # 12c. colab_pet.spec datas 必须覆盖 verify_manifest_and_assets 要求的全部顶层伴随文件
    spec_src = (PKG_DIR / "colab_pet.spec").read_text(encoding="utf-8")
    spec_tree = ast.parse(spec_src, filename="colab_pet.spec")
    spec_top_files: set[str] = set()
    for node in spec_tree.body:
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.List)
            and any(isinstance(t, ast.Name) and t.id == "datas" for t in node.targets)
        ):
            for elt in node.value.elts:
                if isinstance(elt, ast.Tuple) and len(elt.elts) == 2:
                    src_expr, dst_expr = elt.elts
                    if (
                        isinstance(dst_expr, ast.Constant)
                        and dst_expr.value == "."
                        and isinstance(src_expr, ast.Call)
                        and isinstance(src_expr.func, ast.Name)
                        and src_expr.func.id == "str"
                        and len(src_expr.args) == 1
                    ):
                        bin_op = src_expr.args[0]
                        if (
                            isinstance(bin_op, ast.BinOp)
                            and isinstance(bin_op.op, ast.Div)
                            and isinstance(bin_op.right, ast.Constant)
                            and isinstance(bin_op.right.value, str)
                        ):
                            spec_top_files.add(bin_op.right.value)

    manifest = core_health.load_stack_manifest(PKG_DIR / "stack_manifest.json")
    expected_top_files = {
        "stack_manifest.json",
        str(manifest["host_baseline"]["requirements_file"]),
        str(manifest["host_baseline"]["requirements_lock_file"]),
        "colab_ctl.py",
        "colab_security.py",
        "colab_pet_bridge_node.py",
        "colab_remote_common.py",
        "colab_cli_bridge.py",
        "colab_sse_gateway.py",
        "colab_pet_direct_canvas.js",
    }
    for prof_spec in manifest["profiles"].values():
        expected_top_files.add(str(prof_spec["bootstrap_script"]))
        expected_top_files.add(str(prof_spec["smoke_script"]))
    assert expected_top_files.issubset(spec_top_files), (
        f"colab_pet.spec datas 缺失顶层伴随文件: {sorted(expected_top_files - spec_top_files)}"
    )

    fake_meipass = tmp / "fake_internal"
    fake_meipass.mkdir(parents=True, exist_ok=True)
    for rel_name in spec_top_files:
        shutil.copy2(PKG_DIR / rel_name, fake_meipass / rel_name)
    (fake_meipass / "colab_pet").mkdir(parents=True, exist_ok=True)
    (fake_meipass / "colab_pet" / "ui.html").write_text("<html></html>", encoding="utf-8")
    assert core_health.verify_manifest_and_assets(fake_meipass)["ok"] is True

    # 核心伴随组件缺失时 verify_manifest_and_assets 必须报 fail
    for critical_comp in ("colab_sse_gateway.py", "colab_pet_direct_canvas.js", "colab_security.py"):
        comp_path = fake_meipass / critical_comp
        comp_path.unlink()
        bad_verify = core_health.verify_manifest_and_assets(fake_meipass)
        assert bad_verify["ok"] is False
        assert any(critical_comp in err for err in bad_verify["errors"])
        shutil.copy2(PKG_DIR / critical_comp, comp_path)

    # 12d. run_preflight_checks(mode="deep") 宿主桌面运行依赖与冻结态核验
    tools = _prepare_fake_tools(tmp)
    tp = core_preflight.ToolPaths(
        colab_exe=tools.colab_exe,
        colab_py=tools.colab_py,
        rclone_exe=tools.rclone_exe,
        rclone_conf=tools.rclone_conf,
    )

    def _fake_deep_run(cmd: list[str], **_kw: Any) -> subprocess.CompletedProcess[str]:
        joined = " ".join(str(x) for x in cmd)
        if "google-colab-cli" in joined:
            return subprocess.CompletedProcess(cmd, 0, stdout="0.8.0\n", stderr="")
        if "version" in joined:
            return subprocess.CompletedProcess(cmd, 0, stdout="rclone v1.76.0\n", stderr="")
        if "_get_token_from_keyring" in joined:
            return subprocess.CompletedProcess(cmd, 0, stdout='{"has_token": true}\n', stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    deep_ok = core_preflight.run_preflight_checks(
        mode="deep",
        profiles=["anima"],
        gpu="L4",
        tool_paths=tp,
        download_dir=tmp,
        cf_config_path=isolated_env.cf_file,
        pkg_dir=PKG_DIR,
        gdrive_free_gb=20.0,
        run_fn=_fake_deep_run,
    )
    dep_items_ok = [it for it in deep_ok.items if it.name == "桌面运行依赖"]
    assert len(dep_items_ok) == 1
    assert dep_items_ok[0].status == "ok" and dep_items_ok[0].summary == "Python 桌面运行依赖已就绪"

    orig_find_spec = importlib.util.find_spec

    def _missing_find_spec(name: str, package: str | None = None):
        if name in ("webview", "clr"):
            return None
        return orig_find_spec(name, package)

    with patch.object(importlib.util, "find_spec", side_effect=_missing_find_spec):
        deep_missing = core_preflight.run_preflight_checks(
            mode="deep",
            profiles=["anima"],
            gpu="L4",
            tool_paths=tp,
            download_dir=tmp,
            cf_config_path=isolated_env.cf_file,
            pkg_dir=PKG_DIR,
            gdrive_free_gb=20.0,
            run_fn=_fake_deep_run,
        )
    assert deep_missing.ok is False
    dep_items_fail = [it for it in deep_missing.items if it.name == "桌面运行依赖"]
    assert len(dep_items_fail) == 1
    assert dep_items_fail[0].status == "fail"
    assert "requirements-lock.txt" in dep_items_fail[0].fix_hint
    assert "webview" in dep_items_fail[0].summary and "clr" in dep_items_fail[0].summary

    with patch.object(core_health.sys, "frozen", True, create=True):
        deep_frozen = core_preflight.run_preflight_checks(
            mode="deep",
            profiles=["anima"],
            gpu="L4",
            tool_paths=tp,
            download_dir=tmp,
            cf_config_path=isolated_env.cf_file,
            pkg_dir=PKG_DIR,
            gdrive_free_gb=20.0,
            run_fn=_fake_deep_run,
        )
    dep_items_frozen = [it for it in deep_frozen.items if it.name == "桌面运行依赖"]
    assert len(dep_items_frozen) == 1
    assert dep_items_frozen[0].status == "ok" and dep_items_frozen[0].summary == "已内置独立运行时"

    # 12e. 冻结态路径解析（含 _MEIPASS 与仅 mock sys.frozen 兼容）及 _ensure_std_streams 父控制台挂载
    fake_exe_dir = tmp / "frozen_app"
    fake_exe_dir.mkdir(parents=True, exist_ok=True)
    fake_exe = fake_exe_dir / "Colab桌宠.exe"
    fake_exe.write_bytes(b"")

    monkeypatch.delenv("COLAB_PET_CF_CONFIG_PATH", raising=False)
    with (
        patch.object(cb.sys, "frozen", True, create=True),
        patch.object(cb.sys, "_MEIPASS", str(fake_meipass), create=True),
        patch.object(cb.sys, "executable", str(fake_exe)),
    ):
        pet_d, pkg_d, _ws_d, dl_d, ctl_d, state_d, cf_d = cb._resolve_backend_paths()
        assert pkg_d == fake_meipass.resolve()
        assert pet_d == (fake_meipass / "colab_pet").resolve()
        assert ctl_d == fake_meipass.resolve() / "colab_ctl.py"
        assert state_d == fake_exe_dir.resolve() / "pet_state.json"
        assert cf_d == fake_exe_dir.resolve() / ".cf_tunnel_secret.json"
        assert dl_d == fake_exe_dir.resolve() / "_downloads" / "all"

        be_frozen = cb.ColabPetBackend()
        assert be_frozen.state_path == fake_exe_dir.resolve() / "pet_state.json"
        assert be_frozen.download_root == fake_exe_dir.resolve() / "_downloads" / "all"
        assert be_frozen._resolve_cf_config_path() == fake_exe_dir.resolve() / ".cf_tunnel_secret.json"

    with patch.object(cb.sys, "frozen", True, create=True):
        assert cb._has_frozen_meipass() is False
        assert cb._resolve_backend_paths()[1] == cb.PACKAGE_DIR

    with (
        patch.object(core_health.sys, "frozen", True, create=True),
        patch.object(core_health.sys, "_MEIPASS", str(fake_meipass), create=True),
    ):
        assert core_health._resolve_pkg_dir() == fake_meipass.resolve()
    with patch.object(core_health.sys, "frozen", True, create=True):
        assert core_health._resolve_pkg_dir() == PKG_DIR

    with (
        patch.object(pa.sys, "frozen", True, create=True),
        patch.object(pa.sys, "_MEIPASS", str(fake_meipass), create=True),
    ):
        assert pa._resolve_pet_dir() == (fake_meipass / "colab_pet").resolve()

    opened_targets: list[tuple[str, str, str | None, str | None]] = []
    orig_open = builtins.open
    saved_handles_len = len(pa._DEVNULL_HANDLES)

    def _fake_stream_open(file: Any, mode: str = "r", *args: Any, **kwargs: Any):
        if str(file) in ("CONOUT$", os.devnull):
            opened_targets.append((str(file), mode, kwargs.get("encoding"), kwargs.get("errors")))
            return io.StringIO()
        return orig_open(file, mode, *args, **kwargs)

    fake_kernel32 = SimpleNamespace(AttachConsole=lambda _pid: 1)
    fake_windll = SimpleNamespace(kernel32=fake_kernel32)
    with (
        patch.object(pa.os, "name", "nt"),
        patch.object(pa.sys, "stdout", None),
        patch.object(pa.sys, "stderr", None),
        patch.object(pa.sys, "argv", ["Colab桌宠.exe", "--doctor"]),
        patch.object(pa.ctypes, "windll", fake_windll, create=True),
        patch.object(builtins, "open", side_effect=_fake_stream_open),
    ):
        pa._ensure_std_streams()
        assert isinstance(pa.sys.stdout, io.StringIO)
        assert isinstance(pa.sys.stderr, io.StringIO)
    assert opened_targets == [
        ("CONOUT$", "w", "utf-8", "replace"),
        ("CONOUT$", "w", "utf-8", "replace"),
    ]

    opened_targets.clear()
    fake_fail_windll = SimpleNamespace(kernel32=SimpleNamespace(AttachConsole=lambda _pid: 0))
    with (
        patch.object(pa.os, "name", "nt"),
        patch.object(pa.sys, "stdout", None),
        patch.object(pa.sys, "stderr", None),
        patch.object(pa.sys, "argv", ["Colab桌宠.exe", "--doctor"]),
        patch.object(pa.ctypes, "windll", fake_fail_windll, create=True),
        patch.object(builtins, "open", side_effect=_fake_stream_open),
    ):
        pa._ensure_std_streams()
    assert opened_targets == [
        (os.devnull, "w", "utf-8", "replace"),
        (os.devnull, "w", "utf-8", "replace"),
    ]

    opened_targets.clear()
    with (
        patch.object(pa.os, "name", "nt"),
        patch.object(pa.sys, "stdout", None),
        patch.object(pa.sys, "stderr", None),
        patch.object(pa.sys, "argv", ["Colab桌宠.exe"]),
        patch.object(builtins, "open", side_effect=_fake_stream_open),
    ):
        pa._ensure_std_streams()
    assert opened_targets == [
        (os.devnull, "w", "utf-8", "replace"),
        (os.devnull, "w", "utf-8", "replace"),
    ]
    del pa._DEVNULL_HANDLES[saved_handles_len:]


def test_sse_gateway_pre_auth_rejects_before_reading_and_bounds_decompress(
    isolated_env: SimpleNamespace,
) -> None:
    """验证 SSE 网关在鉴权前零读体直接 401 拦截，并对合法口令下的高压缩比解压载荷做 413 限额。"""
    import gzip
    import socket
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    upstream_reqs: list[str] = []

    class _MockUpstream(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, _fmt: str, *_args: Any) -> None:
            return

        def do_GET(self) -> None:  # noqa: N802
            upstream_reqs.append(f"GET {self.path}")
            body = b'{"status":"ok"}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:  # noqa: N802
            upstream_reqs.append(f"POST {self.path}")
            body = b'{"choices":[{"message":{"content":"ok"}}]}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    up_srv = ThreadingHTTPServer(("127.0.0.1", 0), _MockUpstream)
    up_port = int(up_srv.server_address[1])
    up_thread = threading.Thread(target=up_srv.serve_forever, daemon=True)
    up_thread.start()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        gw_port = int(s.getsockname()[1])

    secret_key = "sk-sse-preauth-secret-778899"
    key_file = isolated_env.tmp / "sse_api_key"
    key_file.write_text(secret_key + "\n", encoding="utf-8")

    env = os.environ.copy()
    env.update(
        {
            "SSE_GW_HOST": "127.0.0.1",
            "SSE_GW_PORT": str(gw_port),
            "SSE_UPSTREAM_HOST": "127.0.0.1",
            "SSE_UPSTREAM_PORT": str(up_port),
            "SSE_API_KEY_FILE": str(key_file),
            "SSE_REQUIRE_AUTH": "1",
            "SSE_MAX_DECOMPRESSED_BYTES": "4096",
        }
    )
    gw_proc = subprocess.Popen(
        [sys.executable, str(PKG_DIR / "colab_sse_gateway.py")],
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        # 等待网关端口就绪（/health 保持免密可测）
        deadline = time.monotonic() + 10.0
        ready = False
        while time.monotonic() < deadline:
            try:
                with socket.create_connection(("127.0.0.1", gw_port), timeout=0.5) as probe:
                    probe.sendall(b"GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n")
                    if b"200 OK" in probe.recv(1024):
                        ready = True
                        break
            except OSError:
                time.sleep(0.1)
        assert ready, "SSE 网关未及时就绪"
        upstream_reqs.clear()

        # 1) 未带口令且宣称 10MB Content-Length 但暂不发请求体：网关必须在零读体、零解压前秒回 401
        with socket.create_connection(("127.0.0.1", gw_port), timeout=2.0) as cli:
            cli.sendall(
                b"POST /v1/chat/completions HTTP/1.1\r\n"
                b"Host: 127.0.0.1\r\n"
                b"Content-Encoding: gzip\r\n"
                b"Content-Length: 10485760\r\n"
                b"Connection: close\r\n\r\n"
            )
            raw_resp = cli.recv(4096).decode("utf-8", errors="replace")
        assert "401 Unauthorized" in raw_resp
        assert upstream_reqs == []

        # 2) 携带合法密钥但发送超限高压缩比 gzip 载荷 (>4096 B) -> 有界解压拦截并返回 413
        bomb_json = json.dumps({"stream": True, "pad": "A" * 20000}).encode("utf-8")
        gz_payload = gzip.compress(bomb_json)
        assert len(gz_payload) < 512
        with socket.create_connection(("127.0.0.1", gw_port), timeout=3.0) as cli_ok:
            hdr = (
                f"POST /v1/chat/completions HTTP/1.1\r\n"
                f"Host: 127.0.0.1\r\n"
                f"Authorization: Bearer {secret_key}\r\n"
                f"Content-Encoding: gzip\r\n"
                f"Content-Length: {len(gz_payload)}\r\n"
                f"Connection: close\r\n\r\n"
            ).encode("ascii")
            cli_ok.sendall(hdr + gz_payload)
            resp_413 = cli_ok.recv(4096).decode("utf-8", errors="replace")
        assert "413 Payload Too Large" in resp_413
        assert upstream_reqs == []
    finally:
        gw_proc.terminate()
        try:
            gw_proc.wait(timeout=4)
        except subprocess.TimeoutExpired:
            gw_proc.kill()
        up_srv.shutdown()
        up_srv.server_close()
        up_thread.join(timeout=2.0)
