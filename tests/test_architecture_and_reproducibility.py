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
        '  File "C:\\Workspace\\Colab-Desktop-Pet\\colab_ctl.py", line 347, in _run',
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
    import re
    import socket
    import urllib.request
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    # 7a. 验证 SSE 网关在发出 : keepalive 后上游返回 400 上下文超限，对 /v1/messages 输出 Anthropic event: error
    boot_src = (PKG_DIR / "colab_bootstrap_qwen_llm.py").read_text(encoding="utf-8")
    m_gw = re.search(r"_SSE_GATEWAY_SCRIPT\s*=\s*r'''(.*?)'''", boot_src, flags=re.S)
    assert m_gw is not None
    gw_script = isolated_env.tmp / "llm_sse_gateway_test.py"
    gw_script.write_text(m_gw.group(1).lstrip("\n"), encoding="utf-8")

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

    # 8c. SSE 网关 OPTIONS 预检与 /v1/models 合成响应必须保留 CORS 头
    qwen_src = (PKG_DIR / "colab_bootstrap_qwen_llm.py").read_text(encoding="utf-8")
    m_gw = qwen_src.split("_SSE_GATEWAY_SCRIPT = r'''", 1)[1].split("'''", 1)[0]
    assert "_cors_header_block" in m_gw and 'if method == "OPTIONS":' in m_gw

    # 直接启动临时 SSE 网关实测真实 TCP HTTP OPTIONS 与 /v1/models CORS 响应头
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

        ns: dict[str, Any] = {}
        exec(compile(m_gw.split("if __name__ ==")[0], "<sse_gw>", "exec"), ns)
        ns["LISTEN_PORT"] = gw_port
        ns["UPSTREAM_PORT"] = up_port
        ns["BASE_MODEL"] = "Qwen3.8-27B-abliterated"
        ns["HEARTBEAT_SEC"] = 0.08
        gw_srv = await asyncio.start_server(ns["_handle_client"], "127.0.0.1", gw_port)
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

            st_opt, h_opt = await asyncio.to_thread(_do_opts)
            assert st_opt == 204
            assert h_opt.get("access-control-allow-origin") == "https://chat.example.com"

            st_mod, h_mod, body_mod = await asyncio.to_thread(_do_models)
            assert st_mod == 200
            assert h_mod.get("access-control-allow-origin") == "https://chat.example.com"
            assert any(m.get("id", "").endswith("-nothink") for m in body_mod.get("data") or [])

            st_sse, h_sse, body_sse = await asyncio.to_thread(_do_slow_stream)
            assert st_sse == 200
            assert h_sse.get("access-control-allow-origin") == "https://chat.example.com"
            assert ": keepalive" in body_sse and "data: [DONE]" in body_sse
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

    # 8f. collect_local_py_deps 自动携带 colab_remote_common.py 与 colab_pet_bridge_node.py（含来自 Temp 临时包装文件）
    deps_anima = colab_client.collect_local_py_deps(PKG_DIR / "colab_bootstrap_anima.py")
    assert "colab_remote_common.py" in deps_anima and "colab_pet_bridge_node.py" in deps_anima
    temp_task = tmp / "remote_wrap.py"
    temp_task.write_text("from colab_remote_common import start_cloudflared_tunnel\n", encoding="utf-8")
    deps_tmp = colab_client.collect_local_py_deps(temp_task)
    assert "colab_remote_common.py" in deps_tmp and "colab_pet_bridge_node.py" in deps_tmp

    # 8g. 模拟 REMOTE_RUNNER_TEMPLATE（无 __file__、依赖落于临时 /content 目录）执行全部 4 套自举脚本顶层代码不报 NameError
    remote_dir = tmp / "fake_content"
    remote_dir.mkdir(parents=True, exist_ok=True)
    for dep_name, dep_txt in deps_anima.items():
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


