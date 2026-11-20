"""Phase 3 Wave 3B（ComfySyncService 与 SessionMonitorService）离线回归测试。"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

PACKAGE_DIR = Path(__file__).resolve().parent.parent
PET_DIR = PACKAGE_DIR / "colab_pet"
for _p in (str(PACKAGE_DIR), str(PET_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import backend_comfy as bc_mod  # noqa: E402
import backend_monitor as bm_mod  # noqa: E402
import colab_backend as cb  # noqa: E402
from colab_core.runtime import DEFAULT_PREFS  # noqa: E402


class _FakeResp:
    """最小化 HTTP 响应桩，用于离线模拟 ComfyUI / LLM 探针接口。"""

    def __init__(
        self,
        status_code: int = 200,
        json_data: Any = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self._json_data = json_data
        self.headers = headers or {}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self) -> Any:
        return self._json_data


def _wait_until(predicate, timeout: float = 3.0, err_msg: str = "等待条件超时") -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError(err_msg)


def _calc_ast_cc(node: ast.AST) -> int:
    cc = 1
    for child in ast.walk(node):
        if isinstance(
            child,
            (
                ast.If,
                ast.For,
                ast.AsyncFor,
                ast.While,
                ast.With,
                ast.AsyncWith,
                ast.ExceptHandler,
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


@pytest.fixture()
def isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """隔离凭据库、隧道配置与诊断日志，确保不触碰宿主机任何真实配置。"""
    vault_file = tmp_path / "mock_vault.json"
    cf_file = tmp_path / "cf_tunnel_secret.json"
    diag_file = tmp_path / "diag.log"
    monkeypatch.setenv("COLAB_PET_MOCK_VAULT_PATH", str(vault_file))
    monkeypatch.setenv("COLAB_PET_CF_CONFIG_PATH", str(cf_file))
    monkeypatch.setenv("COLAB_PET_DIAG_LOG_PATH", str(diag_file))
    return SimpleNamespace(tmp=tmp_path, vault_file=vault_file, cf_file=cf_file, diag_file=diag_file)


def _make_backend(env: SimpleNamespace, name: str = "state") -> cb.ColabPetBackend:
    return cb.ColabPetBackend(
        state_path=env.tmp / f"{name}.json",
        download_root=env.tmp / f"{name}_dl",
        cf_config_path=env.cf_file,
    )


def test_comfy_and_monitor_zero_cb_mod_zero_heavy_deps_and_di_bound(
    isolated_env: SimpleNamespace,
) -> None:
    """验证两模块彻底清除 _cb_mod()、零重型依赖、圈复杂度达标且容器注入正确绑定。"""
    forbidden_imports = {"PIL", "aiohttp", "pywin32", "win32api", "win32gui", "win32con"}
    for fname in ("backend_comfy.py", "backend_monitor.py"):
        fpath = PET_DIR / fname
        src = fpath.read_text(encoding="utf-8")
        assert "_cb_mod(" not in src, f"{fname} 仍残留 _cb_mod() 调用"

        tree = ast.parse(src)
        imported_roots: set[str] = set()
        max_cc = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported_roots.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.split(".")[0])
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                max_cc = max(max_cc, _calc_ast_cc(node))

        assert imported_roots.isdisjoint(forbidden_imports), f"{fname} 违规引入重依赖: {imported_roots & forbidden_imports}"
        assert max_cc <= 30, f"{fname} 最高圈复杂度超标: {max_cc}"

    assert bm_mod.SessionMonitorService is not None
    backend = _make_backend(isolated_env, "di_check")
    for svc in (backend._comfy_svc, backend._monitor):
        assert svc._store is backend._store
        assert svc._secrets is backend._secrets
        assert svc._bus is backend._bus
        assert svc._ctx is backend._ctx


def test_probe_comfy_status_merges_host_metrics_history_error_and_wal_queue(
    isolated_env: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 ComfyUI 探针融合 /system_stats 与 /colab_pet/host_metrics、落盘待拉队列及 404 桥接兜底。"""
    backend = _make_backend(isolated_env, "probe_comfy")
    with backend._store.lock:
        backend._store.state["prefs"]["auto_pull_outputs"] = False

    base = "https://mock-comfy-w3b.trycloudflare.com"
    hist_round = {"n": 0}
    host_status = {"code": 200}
    bridge_calls: list[tuple[str, str]] = []

    monkeypatch.setattr(
        backend._comfy_svc,
        "_maybe_trigger_bridge_host_metrics_async",
        lambda b_url, hint="comfy": bridge_calls.append((b_url, hint)),
    )

    def _fake_comfy_get(url: str, **_kw: Any) -> _FakeResp:
        if url.endswith("/system_stats"):
            return _FakeResp(
                200,
                {
                    "devices": [{"name": "cuda:0 NVIDIA L4 : cudaMallocAsync", "vram_total": 24 * 1024**3, "vram_free": 18 * 1024**3}],
                    "system": {"ram_total": 50 * 1024**3, "ram_free": 30 * 1024**3},
                },
            )
        if url.endswith("/queue"):
            return _FakeResp(200, {"queue_running": [["p1"]], "queue_pending": []})
        if url.endswith("/history"):
            hist_round["n"] += 1
            if hist_round["n"] == 1:
                return _FakeResp(200, {"old_pid": {"status": {"status_str": "success"}, "outputs": {}}})
            return _FakeResp(
                200,
                {
                    "old_pid": {"status": {"status_str": "success"}, "outputs": {}},
                    "err_pid": {
                        "status": {
                            "status_str": "error",
                            "messages": [
                                [
                                    "execution_error",
                                    {
                                        "node_type": "KSampler",
                                        "exception_type": "RuntimeError",
                                        "exception_message": "CUDA out of memory",
                                    },
                                ]
                            ],
                        },
                        "outputs": {},
                    },
                    "ok_pid": {
                        "status": {"status_str": "success"},
                        "outputs": {
                            "9": {"images": [{"filename": "art_001.png", "subfolder": "", "type": "output"}]}
                        },
                    },
                },
            )
        if url.endswith("/colab_pet/host_metrics"):
            if host_status["code"] == 404:
                return _FakeResp(404, {})
            return _FakeResp(
                200,
                {
                    "ram_used_gb": 16.5,
                    "ram_total_gb": 53.0,
                    "ram_free_gb": 36.5,
                    "ram_pct": 31.1,
                    "disk_used_gb": 42.0,
                    "disk_total_gb": 200.0,
                    "disk_free_gb": 158.0,
                    "disk_pct": 21.0,
                    "shm_used_gb": 1.2,
                    "shm_total_gb": 16.0,
                    "shm_free_gb": 14.8,
                    "drive_persisted": True,
                    "drive_free_gb": 88.5,
                    "drive_total_gb": 100.0,
                },
            )
        return _FakeResp(404, {})

    monkeypatch.setattr(backend, "_comfy_http_get", _fake_comfy_get)

    # 第一轮：建立初始历史基线，不触发旧图入队
    ok1, st1, ev1 = backend._comfy_svc.probe_comfy_status(base)
    assert ok1 is True
    assert ev1 == []
    assert st1["gpu_full_name"] == "NVIDIA L4"
    assert st1["ram_total_gb"] == 53.0
    assert st1["disk_used_gb"] == 42.0
    assert st1["drive_persisted"] is True

    # 第二轮：新增报错 + 新成图，且 host_metrics 返回 404 触发异步桥接兜底
    host_status["code"] = 404
    ok2, st2, ev2 = backend._comfy_svc.probe_comfy_status(base)
    assert ok2 is True
    assert st2["ram_total_gb"] == 50.0  # 404 时平滑回退 /system_stats 内存
    assert bridge_calls == [(base, "comfy")]
    ev_types = [e["type"] for e in ev2]
    assert "comfy_error" in ev_types and "outputs_ready" in ev_types

    # 验证 WAL 待拉队列已落盘
    disk_state = json.loads((isolated_env.tmp / "probe_comfy.json").read_text(encoding="utf-8"))
    assert [it["filename"] for it in disk_state.get("pending_pull_items", [])] == ["art_001.png"]


def test_probe_llm_status_extracts_tps_gateway_metrics_and_404_fallback(
    isolated_env: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 LLM 探针携带同源 Bearer、提取 /slots TPS、融合 /__pet_metrics 及 404 桥接兜底。"""
    backend = _make_backend(isolated_env, "probe_llm")
    base = "https://mock-llm-w3b.trycloudflare.com"
    backend._secrets.set_ep_meta(
        "ep-llm-01",
        {"kind": "llm", "tunnel_url": base, "gpu_name": "L4", "vram_total_gb": 24.0},
        runtime_secret="sk-test-llm-secret",
    )

    seen_headers: list[dict[str, str]] = []
    metrics_code = {"val": 200}
    bridge_calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        backend._comfy_svc,
        "_maybe_trigger_bridge_host_metrics_async",
        lambda b_url, hint="comfy": bridge_calls.append((b_url, hint)),
    )

    def _fake_http_get(url: str, headers: dict[str, str] | None = None, **_kw: Any) -> _FakeResp:
        seen_headers.append(dict(headers or {}))
        if url.endswith("/health"):
            return _FakeResp(200, {"status": "ok"})
        if url.endswith("/slots"):
            return _FakeResp(
                200,
                [
                    {"id": 0, "is_processing": True, "timings": {"predicted_per_second": 42.56}},
                    {"id": 1, "is_processing": False},
                ],
            )
        if url.endswith("/__pet_metrics"):
            if metrics_code["val"] == 404:
                return _FakeResp(404, {})
            return _FakeResp(
                200,
                {
                    "gpu_full_name": "cuda:0 NVIDIA A100-SXM4-40GB",
                    "vram_used_gb": 22.4,
                    "vram_total_gb": 40.0,
                    "ram_used_gb": 12.0,
                    "ram_total_gb": 83.5,
                    "disk_used_gb": 55.0,
                    "disk_total_gb": 235.0,
                    "drive_persisted": True,
                },
            )
        return _FakeResp(404, {})

    monkeypatch.setattr(backend._http, "get", _fake_http_get)

    ok, info = backend._comfy_svc.probe_llm_status(base, ep="ep-llm-01")
    assert ok is True
    assert info["running_count"] == 1
    assert info["tps"] == pytest.approx(42.6, abs=0.2)
    assert info["gpu_full_name"] == "NVIDIA A100-SXM4-40GB"
    assert info["vram_total_gb"] == 40.0
    assert info["disk_used_gb"] == 55.0
    assert all(h.get("Authorization") == "Bearer sk-test-llm-secret" for h in seen_headers)

    metrics_code["val"] = 404
    ok2, _ = backend._comfy_svc.probe_llm_status(base, ep="ep-llm-01")
    assert ok2 is True
    assert bridge_calls == [(base, "llm")]


def test_on_ws_message_progress_throttle_state_and_execution_error(
    isolated_env: SimpleNamespace,
) -> None:
    """验证 WS 进度 250ms 节流、100% 强制发送、progress_state 解析与 execution_error 双推。"""
    backend = _make_backend(isolated_env, "ws_msg")
    clock = {"t": 100.0}
    backend._ctx._overrides["monotonic"] = lambda: clock["t"]
    backend._ws_last_emit_ts = 0.0

    events: list[dict[str, Any]] = []
    unsub = backend._bus.subscribe(events.append)
    try:
        svc = backend._comfy_svc
        svc.on_ws_message(None, json.dumps({"type": "execution_start", "data": {}}))
        # 首次进度（100.0 - 0.0 >= 0.25）应立即推送
        svc.on_ws_message(None, json.dumps({"type": "progress", "data": {"value": 2, "max": 10}}))
        # 间隔仅 0.1s，被节流拦截
        clock["t"] = 100.1
        svc.on_ws_message(None, json.dumps({"type": "progress", "data": {"value": 4, "max": 10}}))
        # 满 100% 即使间隔 0.05s 也必须立即推送
        clock["t"] = 100.15
        svc.on_ws_message(
            None,
            json.dumps(
                {
                    "type": "progress_state",
                    "data": {"nodes": {"n1": {"state": "running", "value": 10, "max": 10}}},
                }
            ),
        )
        # executing 节点切换与结束唤醒
        backend._wake_monitor.clear()
        svc.on_ws_message(None, json.dumps({"type": "executing", "data": {"node": "42"}}))
        assert backend.get_comfy_state()["current_node"] == "42"
        svc.on_ws_message(None, json.dumps({"type": "executing", "data": {"node": None, "prompt_id": "p-end"}}))
        assert backend._wake_monitor.is_set()

        # execution_error 同时更新红标状态并广播 comfy_tick + comfy_error
        svc.on_ws_message(
            None,
            json.dumps(
                {
                    "type": "execution_error",
                    "data": {
                        "prompt_id": "p-err",
                        "node_type": "VAEDecode",
                        "exception_type": "RuntimeError",
                        "exception_message": "OOM",
                    },
                }
            ),
        )
    finally:
        unsub()

    ticks = [e for e in events if e["type"] == "comfy_tick"]
    errs = [e for e in events if e["type"] == "comfy_error"]
    # execution_start(1) + progress 2/10(1) + progress_state 10/10(1) + execution_error(1) = 4
    assert len(ticks) == 4
    assert ticks[1]["comfy"]["step_pct"] == 20.0
    assert ticks[2]["comfy"]["step_pct"] == 100.0
    assert len(errs) == 1 and errs[0]["prompt_id"] == "p-err"
    assert backend.get_comfy_state()["has_error"] is True


def test_submit_dropped_files_routes_llm_dual_png_and_uploads_async(
    isolated_env: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证拖拽投递在 LLM/双栖/PNG 内嵌工作流分流及正常异步上传后释放 _upload_lock。"""
    backend = _make_backend(isolated_env, "drop_files")
    events: list[dict[str, Any]] = []
    unsub = backend._bus.subscribe(events.append)
    try:
        txt_file = isolated_env.tmp / "notes.txt"
        txt_file.write_text("hello colab pet", encoding="utf-8")
        jpg_file = isolated_env.tmp / "ref.jpg"
        jpg_file.write_bytes(b"\xff\xd8\xff\xe0dummy")
        wf_file = isolated_env.tmp / "demo_wf.json"
        wf_file.write_text(json.dumps({"nodes": [], "links": []}), encoding="utf-8")

        # 1) LLM 模式提取文本进剪贴板
        monkeypatch.setattr(backend, "is_llm_mode_active", lambda: True)
        res_llm = backend._comfy_svc.submit_dropped_files([str(txt_file)])
        assert res_llm["ok"] is True and res_llm["service_kind"] == "llm"
        assert "hello colab pet" in res_llm["clipboard_text"]

        # 2) 双栖模式拖图片触发 dual_image_choice
        monkeypatch.setattr(backend, "is_llm_mode_active", lambda: False)
        monkeypatch.setattr(backend, "is_dual_mode_active", lambda: True)
        res_dual = backend._comfy_svc.submit_dropped_files([str(jpg_file)])
        assert res_dual["ok"] is True and res_dual["choice_required"] is True
        assert any(e["type"] == "dual_image_choice" for e in events)

        # 3) 单张含工作流 PNG 触发 png_workflow_choice
        monkeypatch.setattr(backend, "is_dual_mode_active", lambda: False)
        png_file = isolated_env.tmp / "embed.png"
        png_file.write_bytes(b"\x89PNG\r\n\x1a\n")
        monkeypatch.setattr(bc_mod, "extract_png_comfy_workflow", lambda _p: '{"nodes": []}')
        res_png = backend._comfy_svc.submit_dropped_files([str(png_file)])
        assert res_png["ok"] is True and res_png["choice_required"] is True
        assert any(e["type"] == "png_workflow_choice" for e in events)

        # 4) 正常异步投递工作流并直铺画布
        events.clear()
        monkeypatch.setattr(
            backend,
            "_upload_workflow_file_ex",
            lambda _p, is_png_extract, tunnel_url: ("ok", True, "demo_wf.json"),
        )
        monkeypatch.setattr(
            backend,
            "_push_or_open_workflow_on_canvas",
            lambda _turl, _primary, extra_names=None: {"pushed_live": True},
        )
        res_up = backend._comfy_svc.submit_dropped_files(
            [str(wf_file)], base_url_override="https://mock-up.trycloudflare.com"
        )
        assert res_up["ok"] is True and res_up["started"] is True
        _wait_until(
            lambda: (not backend._upload_lock.locked())
            and any(e["type"] == "upload_done" for e in events),
            err_msg="异步上传未完成或未释放 _upload_lock",
        )
        done_ev = next(e for e in events if e["type"] == "upload_done")
        assert "直铺画布" in done_ev["summary"]
        assert done_ev["action"] == "open_comfy_wf:demo_wf.json"
    finally:
        unsub()


def test_bridge_host_metrics_unlocks_and_rolls_back_ts_on_start_error(
    isolated_env: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证桥接遥测线程启动失败时锁必释放且时间戳回滚，恢复后经 ctx.subprocess_mod 正常写入指标。"""
    backend = _make_backend(isolated_env, "bridge_hm_w3b")
    svc = backend._comfy_svc
    fake_py = isolated_env.tmp / "fake_py.exe"
    fake_py.write_bytes(b"MZ")
    backend._ctx._overrides["get_colab_py"] = lambda: fake_py
    with backend._store.lock:
        backend._store.state["prefs"]["active_account_id"] = "acct_b"
        backend._store.state["last_sessions"] = [{"name": "comfy", "endpoint": "ep-b1"}]

    real_thread = bc_mod.threading.Thread
    fail_once = {"v": True}

    class _FlakyThread:
        def __init__(self, target=None, daemon=None, **kw: Any) -> None:
            self._t = real_thread(target=target, daemon=daemon, **kw)

        def start(self) -> None:
            if fail_once["v"]:
                fail_once["v"] = False
                raise RuntimeError("boom start")
            self._t.start()

    monkeypatch.setattr(bc_mod.threading, "Thread", _FlakyThread)

    with pytest.raises(RuntimeError, match="boom start"):
        svc._maybe_trigger_bridge_host_metrics_async("https://b.trycloudflare.com", "comfy")
    assert svc._bridge_hm_lock.locked() is False
    assert "acct_b:ep-b1" not in svc._bridge_hm_last_ts

    done_ev = threading.Event()
    payload = json.dumps({"ok": True, "disk_used_gb": 33.3, "disk_total_gb": 100.0, "disk_free_gb": 66.7, "disk_pct": 33.3})

    def _fake_run(cmd, **_kw: Any):
        done_ev.set()
        return subprocess.CompletedProcess(
            cmd, 0, stdout=f"<<<COLAB_HOST_METRICS>>>{payload}<<<END_COLAB_HOST_METRICS>>>", stderr=""
        )

    with patch.object(cb.subprocess, "run", side_effect=_fake_run):
        svc._maybe_trigger_bridge_host_metrics_async("https://b.trycloudflare.com", "comfy")
        assert done_ev.wait(timeout=3.0)
        _wait_until(
            lambda: (not svc._bridge_hm_lock.locked()) and backend.get_comfy_state()["disk_used_gb"] == 33.3,
            err_msg="桥接遥测恢复后未写入磁盘指标或未释放锁",
        )


def test_update_ledger_preserves_measured_rates_handles_topup_and_reconciles_sessions(
    isolated_env: SimpleNamespace,
) -> None:
    """验证账本跨日充值核算、孤儿会话对账还原、历史 measured_gpu_rates 保护及旧端点清理。"""
    backend = _make_backend(isolated_env, "ledger_w3b")
    with backend._store.lock:
        st = backend._store.state
        st["date"] = "2020-01-01"
        st["last_balance"] = 100.0
        st["measured_gpu_rates"] = {"T4:STD": 1.11, "L4:STD": 2.22, "A100:HM": 9.99}
        st["managed_binding"] = {"account_id": "acct1", "session": "comfy", "endpoint": "ep-live-1"}
        st["tunnel_cache"] = {
            "ep-live-1": "https://live-one.trycloudflare.com",
            "ep-stale": "https://stale-one.trycloudflare.com",
        }
    backend._secrets.set_ep_meta("ep-live-1", {"kind": "comfy"})
    backend._secrets.set_ep_meta("ep-stale", {"kind": "comfy"})

    live = {
        "balance": 150.0,  # 较昨日 100 充值 +50
        "rate_hourly": 4.82,
        "assignments_count": 1,
        "sessions": [
            {"name": "?", "orphan": True, "endpoint": "ep-live-1", "accelerator": "L4", "machine_shape": "standard"}
        ],
        "account_meta": {"id": "acct1", "email_masked": "u***@gmail.com"},
        "eligible_gpus": ["T4", "L4"],
    }
    with backend._store.lock:
        backend._monitor.update_ledger(live)
        st = backend._store.state
        assert st["day_topup_cu"] == 50.0
        assert st["day_consumed_cu"] == 0.0
        # 单卡 L4:STD 实时采样仅更新 L4:STD，绝不覆写已持久化的 T4:STD / A100:HM
        assert st["measured_gpu_rates"]["L4:STD"] == 4.82
        assert st["measured_gpu_rates"]["T4:STD"] == 1.11
        assert st["measured_gpu_rates"]["A100:HM"] == 9.99
        # 孤儿会话被对账还原为受管 comfy
        assert st["last_sessions"][0]["name"] == "comfy"
        assert st["last_sessions"][0]["orphan"] is False
        # 陈旧端点缓存与内存凭据同步清理
        assert "ep-live-1" in st["tunnel_cache"] and "ep-stale" not in st["tunnel_cache"]
        assert "ep-live-1" in backend._secrets.service_meta_by_ep
        assert "ep-stale" not in backend._secrets.service_meta_by_ep


def test_build_snapshot_mood_levels_and_consumption_display(
    isolated_env: SimpleNamespace,
) -> None:
    """验证 build_snapshot 在停机/余额预警/运行报错/长时运行/连接异常下的心情与消费核算。"""
    backend = _make_backend(isolated_env, "snap_w3b")
    now_ts = 1_700_000_000
    backend._ctx._overrides["now"] = lambda: float(now_ts)

    with backend._store.lock:
        st = backend._store.state
        st["prefs"] = dict(DEFAULT_PREFS, warn_balance=50.0, warn_hours=10.0, long_run_warn_minutes=120)
        st["day_start_balance"] = 200.0
        st["day_topup_cu"] = 20.0
        st["session_start_balance"] = 195.0
        st["active_since_ts"] = now_ts - 150 * 60  # 已连续运行 150 分钟
        st["last_updated_ts"] = now_ts

    # 1) 已停机省积分
    snap_idle = backend._monitor.build_snapshot(
        {"balance": 180.0, "rate_hourly": 0.0, "assignments_count": 0, "sessions": []},
        stale=False,
    )
    assert (snap_idle["mood"], snap_idle["status_level"], snap_idle["status_label"]) == (
        "proud",
        "idle",
        "已停机省积分",
    )
    assert snap_idle["day_consumed_cu"] == 40.0  # 200 + 20 - 180
    assert snap_idle["session_consumed_cu"] == 0.0

    # 2) 长时运行提醒
    snap_long = backend._monitor.build_snapshot(
        {
            "balance": 180.0,
            "rate_hourly": 5.0,
            "assignments_count": 1,
            "sessions": [{"name": "comfy", "endpoint": "ep-1", "orphan": False}],
        },
        stale=False,
    )
    assert (snap_long["mood"], snap_long["status_level"], snap_long["status_label"]) == (
        "sad",
        "warn",
        "长时运行提醒",
    )
    assert snap_long["session_consumed_cu"] == 15.0  # 195 - 180

    # 3) 推理报错中断
    with backend._comfy_lock:
        backend._comfy_state["has_error"] = True
    snap_err = backend._monitor.build_snapshot(
        {
            "balance": 180.0,
            "rate_hourly": 5.0,
            "assignments_count": 1,
            "sessions": [{"name": "comfy", "endpoint": "ep-1", "orphan": False}],
        },
        stale=False,
    )
    assert (snap_err["mood"], snap_err["status_level"], snap_err["status_label"]) == (
        "sad",
        "danger",
        "推理报错中断",
    )

    # 4) 余额预警（优先级高于运行态）
    snap_low = backend._monitor.build_snapshot(
        {"balance": 30.0, "rate_hourly": 5.0, "assignments_count": 1, "sessions": []},
        stale=False,
    )
    assert (snap_low["mood"], snap_low["status_level"], snap_low["status_label"]) == (
        "sad",
        "danger",
        "余额预警",
    )


def test_idle_auto_stop_warning_keep_alive_and_stop_billing_policies(
    isolated_env: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证闲置 2 分钟前一次性预警，以及归档失败下「保图留机」与「优先停卡」双策略分支。"""
    backend = _make_backend(isolated_env, "idle_stop_w3b")
    clock = {"m": 1000.0}
    backend._ctx._overrides["monotonic"] = lambda: clock["m"]

    with backend._store.lock:
        st = backend._store.state
        st["prefs"]["idle_auto_stop_minutes"] = 60
        st["prefs"]["active_account_id"] = "acct_idle"
        st["last_assignments_count"] = 1
        st["last_rate_hourly"] = 5.0
        st["active_endpoint"] = "ep-idle-1"
        st["last_sessions"] = [{"name": "comfy", "endpoint": "ep-idle-1"}]
        backend._last_activity_m = 1000.0
    with backend._comfy_lock:
        backend._comfy_state["online"] = True
        backend._comfy_state["running_count"] = 0
        backend._comfy_state["pending_count"] = 0
        backend._comfy_state["llm_running_count"] = 0

    events: list[dict[str, Any]] = []
    unsub = backend._bus.subscribe(events.append)
    stop_calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        backend._monitor,
        "_run_idle_archive_sync",
        lambda _s, _e, _t: (True, False, "rclone timeout"),
    )
    monkeypatch.setattr(
        backend,
        "stop_session_instance",
        lambda **kw: (stop_calls.append(kw) or {"ok": True, "snapshot": {"ok": True}}),
    )

    try:
        # 1) 闲置 59 分钟触发一次性预警，重复轮询不刷屏
        clock["m"] = 1000.0 + 59 * 60
        backend._monitor.maybe_trigger_idle_auto_stop(now_m=clock["m"])
        backend._monitor.maybe_trigger_idle_auto_stop(now_m=clock["m"] + 10)
        warnings = [e for e in events if e["type"] == "idle_warning"]
        assert len(warnings) == 1

        # 2) 保图留机策略：归档失败不调用 stop_session_instance，推送 idle_archive_failed
        events.clear()
        with backend._store.lock:
            backend._store.state["prefs"]["idle_archive_fail_policy"] = "keep_alive"
        clock["m"] = 1000.0 + 61 * 60
        backend._monitor.maybe_trigger_idle_auto_stop(now_m=clock["m"])
        _wait_until(
            lambda: (not backend._idle_stop_lock.locked())
            and any(e["type"] == "idle_archive_failed" for e in events),
            err_msg="保图留机策略未推送 idle_archive_failed",
        )
        assert stop_calls == []
        assert backend._idle_archive_retry_after_m > clock["m"]

        # 3) 优先停卡策略：归档失败仍停卡止损并推送 idle_stopped(archive_failed=True)
        events.clear()
        with backend._store.lock:
            backend._store.state["prefs"]["idle_archive_fail_policy"] = "stop_billing"
            backend._idle_archive_retry_after_m = 0.0
        backend._monitor.maybe_trigger_idle_auto_stop(now_m=clock["m"])
        _wait_until(
            lambda: (not backend._idle_stop_lock.locked())
            and any(e["type"] == "idle_stopped" for e in events),
            err_msg="优先停卡策略未执行停卡或未推送 idle_stopped",
        )
        assert len(stop_calls) == 1 and stop_calls[0]["endpoint"] == "ep-idle-1"
        stopped_ev = next(e for e in events if e["type"] == "idle_stopped")
        assert stopped_ev["archive_failed"] is True
    finally:
        unsub()


def test_keepalive_throttle_idle_cutoff_and_failure_cooldown(
    isolated_env: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证保活心跳 10 分钟节流、失败冷却回退及纯空闲超 45 分钟自动停发。"""
    backend = _make_backend(isolated_env, "keepalive_w3b")
    with backend._store.lock:
        st = backend._store.state
        st["last_assignments_count"] = 1
        st["last_rate_hourly"] = 5.0
        st["last_sessions"] = [{"name": "comfy", "endpoint": "ep-ka-1"}]
        backend._last_activity_m = 1000.0
        backend._last_keepalive_ts = 0.0

    ka_calls: list[str] = []
    ka_result = {"ok": True}
    done_ev = threading.Event()

    def _fake_ka(target_ep: str = "") -> bool:
        ka_calls.append(target_ep)
        done_ev.set()
        return ka_result["ok"]

    monkeypatch.setattr(backend, "_send_colab_keepalive", _fake_ka)

    # 1) 首次触发成功
    backend._monitor.maybe_trigger_keepalive(now_m=1100.0)
    assert done_ev.wait(timeout=3.0)
    _wait_until(lambda: not backend._keepalive_lock.locked(), err_msg="保活锁未释放")
    assert ka_calls == ["ep-ka-1"]
    assert backend._last_keepalive_ts == 1100.0

    # 2) 节流期内（不足 600s）不重复发送
    backend._monitor.maybe_trigger_keepalive(now_m=1500.0)
    assert ka_calls == ["ep-ka-1"]

    # 3) 过节流期但本次保活失败 -> 时间戳回退以在重试冷却后重试
    done_ev.clear()
    ka_result["ok"] = False
    backend._monitor.maybe_trigger_keepalive(now_m=1800.0)
    assert done_ev.wait(timeout=3.0)
    _wait_until(lambda: not backend._keepalive_lock.locked(), err_msg="失败保活后锁未释放")
    assert len(ka_calls) == 2
    expected_rollback = 1800.0 - (
        backend._ctx.keepalive_interval_sec - backend._ctx.keepalive_retry_cooldown_sec
    )
    assert backend._last_keepalive_ts == pytest.approx(expected_rollback, abs=1e-3)

    # 4) 纯空闲超过 keepalive_max_idle_sec (45m) 自动停发保活
    backend._last_keepalive_ts = 0.0
    backend._monitor.maybe_trigger_keepalive(
        now_m=1000.0 + backend._ctx.keepalive_max_idle_sec + 30.0
    )
    assert len(ka_calls) == 2


def test_monitor_tick_dispatches_offline_comfy_llm_and_dual_modes(
    isolated_env: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证监控单轮调度在离线、ComfyUI、LLM 与 Dual 模式下返回正确轮询周期与分流动作。"""
    backend = _make_backend(isolated_env, "monitor_tick_w3b")
    mon = backend._monitor
    calls: list[str] = []

    monkeypatch.setattr(backend, "_maybe_trigger_keepalive", lambda: calls.append("ka"))
    monkeypatch.setattr(backend, "_maybe_trigger_idle_auto_stop", lambda: calls.append("idle"))
    monkeypatch.setattr(backend, "_reset_comfy_offline", lambda: calls.append("reset"))
    monkeypatch.setattr(backend, "_stop_ws", lambda: calls.append("stop_ws"))
    monkeypatch.setattr(backend, "_ensure_ws_target", lambda u: calls.append(f"ws:{u}"))

    # 1) 无运行实例 -> 复位离线并休眠 15s
    assert mon._poll_single_monitor_tick() == 15.0
    assert calls == ["ka", "reset"]

    # 2) 运行态 + LLM 忙碌 -> 停 WS、轮询 LLM、返回 2.5s
    calls.clear()
    with backend._store.lock:
        backend._store.state["last_assignments_count"] = 1
        backend._store.state["last_sessions"] = [{"name": "comfy", "endpoint": "ep-m1"}]
    backend._secrets.set_ep_meta("ep-m1", {"kind": "llm"})
    monkeypatch.setattr(
        backend, "resolve_active_tunnel_url", lambda allow_fast_discover=True: "https://llm.trycloudflare.com"
    )
    monkeypatch.setattr(backend, "poll_llm_http_once", lambda _u, ep="": {"running_count": 1})
    assert mon._poll_single_monitor_tick() == 2.5
    assert calls == ["ka", "stop_ws", "idle"]

    # 3) 运行态 + Dual 空闲 -> 启动 WS、轮询 Dual、返回 6.0s
    calls.clear()
    backend._secrets.set_ep_meta("ep-m1", {"kind": "dual"})
    monkeypatch.setattr(
        backend, "resolve_active_llm_tunnel_url", lambda allow_fast_discover=True: "https://llm2.trycloudflare.com"
    )
    monkeypatch.setattr(
        backend,
        "poll_dual_once",
        lambda _cu, _lu, ep="": {"running_count": 0, "pending_count": 0, "llm_running_count": 0},
    )
    assert mon._poll_single_monitor_tick() == 6.0
    assert calls == ["ka", "ws:https://llm.trycloudflare.com", "idle"]
