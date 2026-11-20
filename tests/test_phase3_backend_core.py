"""Phase 3 · Wave 3A（路 1）核心状态/凭据/上下文容器与 `colab_backend.py` 瘦身回归测试。

严格离线隔离执行：所有用例通过 `COLAB_PET_MOCK_VAULT_PATH`、`COLAB_PET_CF_CONFIG_PATH`
与临时 `state_path` 隔离，绝不读写宿主机真实 Windows 凭据库或污染工作区运行态 JSON。
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
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
from colab_core.runtime import DEFAULT_PREFS, sanitize_prefs_dict  # noqa: E402
from colab_pet.context import BackendContext, EventBus  # noqa: E402
from colab_pet.secret_store import EphemeralSecretStore  # noqa: E402
from colab_pet.state_store import AppStateStore  # noqa: E402


@pytest.fixture()
def isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """为每个测试用例构造完全隔离的 mock vault、CF 配置与诊断日志路径。"""
    vault_file = tmp_path / "mock_vault.json"
    cf_file = tmp_path / "cf_tunnel_secret.json"
    diag_file = tmp_path / "pet_diag.log"
    monkeypatch.setenv("COLAB_PET_MOCK_VAULT_PATH", str(vault_file))
    monkeypatch.setenv("COLAB_PET_CF_CONFIG_PATH", str(cf_file))
    monkeypatch.setenv("COLAB_PET_DIAG_LOG_PATH", str(diag_file))
    return SimpleNamespace(
        tmp=tmp_path,
        vault_file=vault_file,
        cf_file=cf_file,
        diag_file=diag_file,
    )


def _calc_ast_cc(node: ast.AST) -> int:
    """计算单个函数 AST 节点的圈复杂度（Cyclomatic Complexity）。"""
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
# 1. 包入口无头安全与双形态导入测试
# =====================================================================


def test_colab_pet_package_init_is_headless_safe() -> None:
    """验证 `import colab_pet` 保持极轻量，不会急切加载 GUI、Win32/CLR 或重型后端。"""
    probe_code = (
        "import json, sys\n"
        f"sys.path.insert(0, {str(PACKAGE_DIR)!r})\n"
        "import colab_pet\n"
        "loaded = sorted(sys.modules.keys())\n"
        "print(json.dumps({'version': colab_pet.__version__, 'loaded': loaded}))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", probe_code],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=20,
        check=True,
    )
    data = json.loads(proc.stdout.strip())
    assert data["version"]
    loaded = set(data["loaded"])
    forbidden_eager = {
        "pet_app",
        "colab_pet.pet_app",
        "clr",
        "webview",
        "colab_backend",
        "colab_pet.colab_backend",
    }
    assert loaded.isdisjoint(forbidden_eager), f"检测到不应急切导入的模块: {loaded & forbidden_eager}"


def test_dual_import_modes_flat_and_package() -> None:
    """验证平铺 `sys.path` 导入与 `colab_pet.*` 包导入双形态均可无循环依赖顺利加载。"""
    flat_code = (
        "import sys\n"
        f"sys.path.insert(0, {str(PET_DIR)!r})\n"
        f"sys.path.insert(0, {str(PACKAGE_DIR)!r})\n"
        "import state_store, secret_store, context, colab_backend\n"
        "assert hasattr(colab_backend, 'ColabPetBackend')\n"
        "assert state_store.AppStateStore is colab_backend.AppStateStore\n"
        "assert secret_store.EphemeralSecretStore is colab_backend.EphemeralSecretStore\n"
        "assert context.BackendContext is colab_backend.BackendContext\n"
        "print('FLAT_OK')\n"
    )
    pkg_code = (
        "import sys\n"
        f"sys.path.insert(0, {str(PACKAGE_DIR)!r})\n"
        "import colab_pet.state_store as ss\n"
        "import colab_pet.secret_store as sec\n"
        "import colab_pet.context as ctx\n"
        "import colab_pet.colab_backend as cb_pkg\n"
        "assert ss.AppStateStore is cb_pkg.AppStateStore\n"
        "assert sec.EphemeralSecretStore is cb_pkg.EphemeralSecretStore\n"
        "assert ctx.BackendContext is cb_pkg.BackendContext\n"
        "print('PKG_OK')\n"
    )
    for script, expected in ((flat_code, "FLAT_OK"), (pkg_code, "PKG_OK")):
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=25,
            check=True,
        )
        assert expected in proc.stdout


# =====================================================================
# 2. AppStateStore 状态存储容器测试
# =====================================================================


def test_state_store_module_documents_lock_order() -> None:
    """验证 `state_store.py` 模块文档明文记录全局 4 级锁获取偏序与并发 I/O 红线。"""
    import colab_pet.state_store as state_store_mod

    doc = state_store_mod.__doc__ or ""
    for token in (
        "_deploy_lock",
        "AppStateStore.lock",
        "_comfy_lock",
        "EventBus._lock",
        "阻塞网络",
    ):
        assert token in doc, f"state_store 模块 docstring 缺少关键锁偏序说明: {token}"


def test_state_store_load_save_and_cluster_slots_recovery(tmp_path: Path) -> None:
    """验证 `AppStateStore.load` 兼容恢复 `cluster_slots` 以及 `save` 原子落盘无 `.tmp` 残留。"""
    state_file = tmp_path / "nested" / "pet_state.json"
    state_file.parent.mkdir(parents=True, exist_ok=True)
    disk_payload = {
        "cluster_slots": {
            "llm": {"account_id": "acct_llm", "endpoint": "ep-llm-1", "tunnel_url": "https://llm.trycloudflare.com"},
            "comfy": {"account_id": "acct_comfy", "endpoint": "ep-comfy-1"},
            "unknown_slot": {"account_id": "acct_bad", "endpoint": "ep-bad"},
        }
    }
    state_file.write_text(json.dumps(disk_payload, ensure_ascii=False), encoding="utf-8")

    store = AppStateStore(state_file)
    loaded = store.load(lambda: {"prefs": dict(DEFAULT_PREFS), "last_sessions": []})
    assert set(loaded["cluster_slots"].keys()) == {"llm", "comfy"}
    assert store.get_active_cluster_endpoints() == {"ep-llm-1", "ep-comfy-1"}
    assert store.get_cluster_slots()["llm"]["account_id"] == "acct_llm"

    store.state["custom_flag"] = "atomic_ok"
    store.save()
    assert state_file.exists()
    assert not state_file.with_suffix(".tmp").exists()
    saved_disk = json.loads(state_file.read_text(encoding="utf-8"))
    assert saved_disk["custom_flag"] == "atomic_ok"


def test_state_store_update_prefs_blocks_tool_paths_and_syncs_profiles(tmp_path: Path) -> None:
    """验证 `update_prefs` 拦截受限工具路径键，并自动保持 `boot_profile` 与 `boot_profiles` 同步。"""
    state_file = tmp_path / "prefs_state.json"
    store = AppStateStore(
        state_file,
        initial_state={
            "prefs": {
                **DEFAULT_PREFS,
                "active_account_id": "acct_main",
                "colab_exe_path": "C:/safe/colab.exe",
            }
        },
    )

    updated = store.update_prefs(
        {
            "boot_profile": "qwen_27b",
            "colab_exe_path": "C:/evil/malware.exe",
            "rclone_exe_path": "C:/evil/rclone.exe",
            "unknown_key_ignored": 123,
        },
        sanitize_prefs_dict,
        blocked_keys=cb._TOOL_PATH_PREF_KEYS,
    )
    assert updated["boot_profile"] == "qwen_27b"
    assert updated["boot_profiles"] == ["qwen_27b"]
    assert updated["colab_exe_path"] == "C:/safe/colab.exe"
    assert "unknown_key_ignored" not in updated
    assert store.get_active_account_id() == "acct_main"
    assert store.get_prefs()["boot_profiles"] == ["qwen_27b"]

    # 同时传入 boot_profiles 时不应被单选 boot_profile 覆写
    updated2 = store.update_prefs(
        {"boot_profile": "anima", "boot_profiles": ["minimax", "qwen_27b"]},
        sanitize_prefs_dict,
        blocked_keys=cb._TOOL_PATH_PREF_KEYS,
    )
    assert updated2["boot_profiles"] == ["minimax", "qwen_27b"]


def test_state_store_tunnel_cache_sessions_and_managed_binding(tmp_path: Path) -> None:
    """验证隧道缓存、目标会话选取、运行态判定与托管绑定持久化。"""
    store = AppStateStore(
        tmp_path / "ops_state.json",
        initial_state={"prefs": {"active_account_id": "acct_01"}},
    )
    assert store.has_running_session() is False
    assert store.pick_target_session("comfy") is None

    store.set_tunnel_cache("ep-1", "https://alpha.trycloudflare.com/")
    assert store.get_tunnel_cache() == {"ep-1": "https://alpha.trycloudflare.com/"}
    assert store.find_endpoint_by_tunnel_url("https://alpha.trycloudflare.com") == "ep-1"
    assert store.find_endpoint_by_tunnel_url("https://miss.trycloudflare.com") == ""
    assert store.pop_tunnel_cache("ep-1") == "https://alpha.trycloudflare.com/"
    assert store.pop_tunnel_cache("ep-1") is None

    store.state["last_sessions"] = [
        {"name": "llm", "endpoint": "ep-llm"},
        {"name": "comfy", "endpoint": "ep-comfy"},
    ]
    assert len(store.get_last_sessions()) == 2
    assert store.pick_target_session("comfy")["endpoint"] == "ep-comfy"
    assert store.pick_target_session("missing")["endpoint"] == "ep-llm"
    assert store.has_running_session() is True

    binding = store.record_managed_binding("comfy", "ep-comfy")
    assert binding == {"account_id": "acct_01", "session": "comfy", "endpoint": "ep-comfy"}
    assert store.record_managed_binding("comfy", "") is None
    assert json.loads(store.state_path.read_text(encoding="utf-8"))["managed_binding"] == binding


# =====================================================================
# 3. EphemeralSecretStore 纯内存凭据与同源同号校验测试
# =====================================================================


def test_secret_store_lifecycle_meta_and_cleanup() -> None:
    """验证 `EphemeralSecretStore` 的元数据读写、端点清理与保留过滤行为。"""
    secrets = EphemeralSecretStore()
    assert secrets.get_ep_service_kind("ep-none", default_kind="fallback") == "fallback"

    secrets.set_ep_meta("ep-1", {"kind": "LLM", "model": "qwen"}, runtime_secret="sk-llm-1")
    secrets.set_ep_meta("ep-2", {"kind": "dual"}, runtime_secret="sk-dual-2")
    secrets.set_comfy_secret("ep-2", "cpk-2", account_id="acct_a", origin="https://c2.trycloudflare.com")
    secrets.comfy_key_checked_eps.update({"ep-1", "ep-2"})

    meta1, key1 = secrets.get_ep_meta_and_key("ep-1")
    assert meta1["model"] == "qwen" and key1 == "sk-llm-1"
    assert secrets.get_ep_service_kind("ep-1") == "llm"
    assert secrets.get_ep_service_kind("ep-2") == "dual"
    assert secrets.known_comfy_secret_values() == {"cpk-2"}

    # 仅保留 ep-2，清理 ep-1
    secrets.retain_endpoints({"ep-2"})
    assert "ep-1" not in secrets.service_meta_by_ep
    assert "ep-1" not in secrets.runtime_secrets
    assert "ep-2" in secrets.comfy_secrets_by_ep
    assert secrets.comfy_key_checked_eps == {"ep-2"}

    # 删除单个端点
    secrets.drop_endpoint("ep-2")
    assert secrets.service_meta_by_ep == {}
    assert secrets.comfy_secrets_by_ep == {}
    assert secrets.comfy_key_bindings_by_ep == {}
    assert secrets.comfy_key_checked_eps == set()


def test_secret_store_strict_origin_and_account_isolation() -> None:
    """验证 Comfy 密钥严格按 `(account_id, endpoint, origin)` 校验，绝不跨源/跨号/单密钥回退。"""
    secrets = EphemeralSecretStore()
    secrets.set_comfy_secret(
        "ep-a",
        "cpk-secret-a",
        account_id="acct_a",
        origin="https://comfy-a.trycloudflare.com",
    )
    # LLM 类型元数据的隧道 URL 不应混入 Comfy 候选源站
    secrets.set_ep_meta("ep-a", {"kind": "llm", "tunnel_url": "https://llm-only.trycloudflare.com"})

    invalid_slot = {
        "endpoint": "ep-a",
        "account_id": "acct_a",
        "tunnel_url": "https://invalid-slot.trycloudflare.com",
        "binding_invalid": True,
    }
    origins, bound_acct = secrets.collect_candidate_origins(
        "ep-a",
        tunnel_cache={"ep-a": "https://cache-a.trycloudflare.com"},
        comfy_slot=invalid_slot,
    )
    assert bound_acct == "acct_a"
    assert "https://comfy-a.trycloudflare.com:443" in origins
    assert "https://cache-a.trycloudflare.com:443" in origins
    assert "https://llm-only.trycloudflare.com:443" not in origins
    assert "https://invalid-slot.trycloudflare.com:443" not in origins

    # 同号同源命中
    assert (
        secrets.get_comfy_key_for_url(
            "https://comfy-a.trycloudflare.com",
            ep="ep-a",
            req_acct="acct_a",
        )
        == "cpk-secret-a"
    )
    # 不传 ep 但同号同源命中
    assert (
        secrets.get_comfy_key_for_url(
            "https://comfy-a.trycloudflare.com",
            ep="",
            req_acct="acct_a",
        )
        == "cpk-secret-a"
    )
    # 跨源拒绝（即便内存中仅此一把密钥也绝不回退）
    assert (
        secrets.get_comfy_key_for_url(
            "https://evil-other.trycloudflare.com",
            ep="ep-a",
            req_acct="acct_a",
        )
        == ""
    )
    assert (
        secrets.get_comfy_key_for_url(
            "https://evil-other.trycloudflare.com",
            ep="",
            req_acct="acct_a",
        )
        == ""
    )
    # 跨账号拒绝
    assert (
        secrets.get_comfy_key_for_url(
            "https://comfy-a.trycloudflare.com",
            ep="ep-a",
            req_acct="acct_b",
        )
        == ""
    )
    # 非法 URL 拒绝
    assert secrets.get_comfy_key_for_url("not-a-valid-url", ep="ep-a", req_acct="acct_a") == ""


def test_secret_store_hydrate_cooldown_and_binding_recording() -> None:
    """验证补水冷却节流与远端补水后绑定登记逻辑。"""
    secrets = EphemeralSecretStore()
    assert secrets.should_skip_hydrate("ep-h1", force=False, now_m=100.0) is False
    assert secrets.should_skip_hydrate("ep-h1", force=False, now_m=105.0) is True
    assert secrets.should_skip_hydrate("ep-h1", force=True, now_m=105.5) is False
    assert secrets.should_skip_hydrate("ep-h1", force=False, now_m=114.0) is False

    secrets.set_comfy_secret("ep-h1", "cpk-h1")
    secrets.set_ep_meta("ep-h1", {"kind": "dual", "comfy_tunnel_url": "https://dual-comfy.trycloudflare.com"})
    live_orig = secrets.record_hydrated_binding("ep-h1", "acct_h1", fast_url="https://fallback.trycloudflare.com")
    assert live_orig == "https://dual-comfy.trycloudflare.com:443"
    assert secrets.comfy_key_bindings_by_ep["ep-h1"] == ("acct_h1", "https://dual-comfy.trycloudflare.com:443")

    # 纯 LLM 端点绝不写入 Comfy 绑定
    secrets.set_comfy_secret("ep-llm", "cpk-should-not-bind")
    secrets.set_ep_meta("ep-llm", {"kind": "llm"})
    assert secrets.record_hydrated_binding("ep-llm", "acct_llm", fast_url="https://llm.trycloudflare.com") == ""
    assert "ep-llm" not in secrets.comfy_key_bindings_by_ep


# =====================================================================
# 4. EventBus 与 BackendContext 显式依赖注入测试
# =====================================================================


def test_event_bus_redacts_isolates_exceptions_and_unsubscribes() -> None:
    """验证 EventBus 广播前脱敏、单监听器异常隔离、去重 legacy_callback 与幂等退订。"""
    seen_a: list[dict] = []
    seen_legacy: list[dict] = []

    def _redact(obj):
        if isinstance(obj, dict):
            return {k: (v.replace("SECRET_TOKEN", "***") if isinstance(v, str) else v) for k, v in obj.items()}
        return obj

    bus = EventBus(redact_fn=_redact)

    def _broken(_evt: dict) -> None:
        raise RuntimeError("模拟监听器崩溃")

    unsub_broken = bus.subscribe(_broken)
    unsub_a = bus.subscribe(lambda evt: seen_a.append(evt))
    bus.legacy_callback = lambda evt: seen_legacy.append(evt)

    bus.emit({"type": "notice", "msg": "token=SECRET_TOKEN"})
    assert seen_a == [{"type": "notice", "msg": "token=***"}]
    assert seen_legacy == [{"type": "notice", "msg": "token=***"}]

    # 若 legacy_callback 同时在 listeners 中，只触发一次
    dup_hits: list[dict] = []

    def dup_cb(evt: dict) -> None:
        dup_hits.append(evt)

    unsub_dup = bus.subscribe(dup_cb)
    bus.legacy_callback = dup_cb
    bus.emit({"type": "ping"})
    assert len(dup_hits) == 1

    # 幂等取消订阅
    count_before_unsub = len(seen_a)
    assert count_before_unsub == 2
    unsub_a()
    unsub_a()
    unsub_broken()
    unsub_dup()
    bus.legacy_callback = None
    bus.emit({"type": "after_unsub"})
    assert len(seen_a) == count_before_unsub


def test_backend_context_from_module_compat_dynamic_monkeypatch_and_overrides(
    isolated_env: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 `BackendContext.from_module_compat` 动态穿透 `monkeypatch.setattr(cb, ...)` 与显式 overrides。"""
    backend = cb.ColabPetBackend(
        state_path=isolated_env.tmp / "ctx_state.json",
        download_root=isolated_env.tmp / "dl",
        cf_config_path=isolated_env.cf_file,
    )
    ctx = backend._ctx

    # 校验全部关键静态/常量依赖可读
    assert ctx.package_dir == cb.PACKAGE_DIR
    assert ctx.pet_dir == cb.PET_DIR
    assert ctx.cli_bridge_script == cb.CLI_BRIDGE_SCRIPT
    assert ctx.ctl_script == cb.CTL_SCRIPT
    assert ctx.bridge_node_init_py == cb._COMFY_BRIDGE_NODE_INIT_PY
    assert ctx.preserve_original_sub_re is cb._PRESERVE_ORIGINAL_SUB_RE
    assert ctx.subprocess_mod is cb.subprocess
    assert ctx.threading_mod is cb.threading
    assert ctx.webbrowser_mod is cb.webbrowser
    assert ctx.websocket_mod is cb.websocket
    assert ctx.resolve_cf_config_path() == isolated_env.cf_file

    # 构造后对 cb 模块打补丁 -> ctx 必须动态实时穿透
    calls: list[tuple] = []
    monkeypatch.setattr(cb, "KEEPALIVE_INTERVAL_SEC", 42.5)
    monkeypatch.setattr(cb, "KEEPALIVE_RETRY_COOLDOWN_SEC", 13.0)
    monkeypatch.setattr(cb, "KEEPALIVE_MAX_IDLE_SEC", 99.0)
    monkeypatch.setattr(cb, "_run_hidden", lambda *a, **kw: calls.append(("run", a, kw)) or "PATCHED_RUN")
    monkeypatch.setattr(cb, "_is_allowed_tunnel_url", lambda u, **_kw: u == "https://patched.trycloudflare.com")
    monkeypatch.setattr(cb, "_probe_tunnel_alive", lambda u, **_kw: u.endswith("/ok"))
    monkeypatch.setattr(cb, "_detect_tunnel_kind", lambda _u, **_kw: "llm")
    monkeypatch.setattr(cb, "_load_cf_secret_raw", lambda **_kw: {"configured": True})
    monkeypatch.setattr(cb, "_ctl_available", lambda: True)
    monkeypatch.setattr(cb, "_ctl_cmd_prefix", lambda unbuffered=False: ["py", "-u" if unbuffered else "norm"])
    monkeypatch.setattr(cb, "_hidden_python_exe", lambda: "C:/fake/python.exe")
    monkeypatch.setattr(cb, "_reveal_path_in_explorer", lambda p: calls.append(("reveal", p)))
    fake_time = SimpleNamespace(time=lambda: 1700000000.0, monotonic=lambda: 55.5, sleep=lambda s: calls.append(("sleep", s)))
    fake_wb = SimpleNamespace(open=lambda u, *a, **kw: calls.append(("wb", u)) or True)
    monkeypatch.setattr(cb, "time", fake_time)
    monkeypatch.setattr(cb, "webbrowser", fake_wb)

    assert ctx.keepalive_interval_sec == 42.5
    assert ctx.keepalive_retry_cooldown_sec == 13.0
    assert ctx.keepalive_max_idle_sec == 99.0
    assert ctx.run_hidden(["echo", "hi"], timeout=5) == "PATCHED_RUN"
    assert ctx.is_allowed_tunnel_url("https://patched.trycloudflare.com") is True
    assert ctx.is_allowed_tunnel_url("https://other.trycloudflare.com") is False
    assert ctx.probe_tunnel_alive("https://x/ok") is True
    assert ctx.detect_tunnel_kind("https://x") == "llm"
    assert ctx.load_cf_secret_raw() == {"configured": True}
    assert ctx.ctl_available() is True
    assert ctx.ctl_cmd_prefix(unbuffered=True) == ["py", "-u"]
    assert ctx.hidden_python_exe() == "C:/fake/python.exe"
    ctx.reveal_path(isolated_env.tmp)
    assert ctx.now() == 1700000000.0
    assert ctx.monotonic() == 55.5
    ctx.sleep(0.25)
    assert ctx.open_browser("https://example.com") is True
    assert ("sleep", 0.25) in calls
    assert ("wb", "https://example.com") in calls
    assert ("reveal", isolated_env.tmp) in calls

    # backend 实例工具路径补丁实时穿透
    fake_py = isolated_env.tmp / "fake_colab_python.exe"
    fake_py.write_text("", encoding="utf-8")
    with patch.object(backend, "_get_colab_py", return_value=fake_py):
        assert ctx.get_colab_py() == fake_py
        env_map = ctx.build_ctl_env()
        assert env_map["COLAB_PET_COLAB_PY"] == str(fake_py)
        assert env_map["COLAB_PET_CF_CONFIG_PATH"] == str(isolated_env.cf_file)

    # 显式 overrides 优先级最高，且未知键名必抛 TypeError
    overridden = BackendContext.from_module_compat(
        cb,
        backend=backend,
        now=lambda: 999.0,
        keepalive_interval_sec=7.0,
        get_colab_py=lambda: Path("D:/override/python.exe"),
    )
    assert overridden.now() == 999.0
    assert overridden.keepalive_interval_sec == 7.0
    assert overridden.get_colab_py() == Path("D:/override/python.exe")

    with pytest.raises(TypeError, match="未知的 BackendContext"):
        BackendContext.from_module_compat(cb, nonexistent_dep=123)


def test_backend_context_without_backend_uses_module_fallbacks(
    isolated_env: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证未传入 `backend` 时，`BackendContext.from_module_compat` 自动回退至模块级工具路径探测。"""
    fake_cexe = isolated_env.tmp / "colab.exe"
    fake_cpy = isolated_env.tmp / "python.exe"
    fake_rexe = isolated_env.tmp / "rclone.exe"
    fake_rconf = isolated_env.tmp / "rclone.conf"
    for p in (fake_cexe, fake_cpy, fake_rexe, fake_rconf):
        p.write_text("", encoding="utf-8")

    monkeypatch.setattr(cb, "COLAB_EXE_DEFAULT", fake_cexe)
    monkeypatch.setattr(cb, "COLAB_PY", fake_cpy)
    monkeypatch.setattr(cb, "RCLONE_EXE", fake_rexe)
    monkeypatch.setattr(cb, "RCLONE_CONF", fake_rconf)

    ctx = BackendContext.from_module_compat(cb)
    assert ctx.get_colab_exe() == fake_cexe
    assert ctx.get_colab_py() == fake_cpy
    assert ctx.get_rclone_exe() == fake_rexe
    assert ctx.get_rclone_conf() == fake_rconf
    assert ctx.resolve_cf_config_path() == isolated_env.cf_file

    env_map = ctx.build_ctl_env()
    assert env_map["COLAB_PET_COLAB_EXE"] == str(fake_cexe)
    assert env_map["COLAB_PET_COLAB_PY"] == str(fake_cpy)
    assert env_map["COLAB_PET_RCLONE_EXE"] == str(fake_rexe)
    assert env_map["COLAB_PET_RCLONE_CONF"] == str(fake_rconf)
    assert env_map["COLAB_PET_CF_CONFIG_PATH"] == str(isolated_env.cf_file)
    assert env_map["COLAB_PET_MOCK_VAULT_PATH"] == str(isolated_env.vault_file)


def test_backend_get_ep_service_kind_priority_and_lock_safety(isolated_env: SimpleNamespace) -> None:
    """验证 `_get_ep_service_kind` 按「集群分槽 -> 远端元数据 -> 在线状态 -> 启动配置」顺序判定。"""
    backend = cb.ColabPetBackend(
        state_path=isolated_env.tmp / "kind_state.json",
        download_root=isolated_env.tmp / "dl",
        cf_config_path=isolated_env.cf_file,
    )
    # 1. 默认回退 boot_profiles
    with backend._lock:
        backend.state["prefs"]["boot_profiles"] = ["qwen_27b"]
    assert backend._get_ep_service_kind() == "llm"

    # 2. 在线 _comfy_state 优先于 boot_profiles
    with backend._comfy_lock:
        backend._comfy_state["online"] = True
        backend._comfy_state["service_kind"] = "dual"
    assert backend._get_ep_service_kind() == "dual"

    # 3. 远端元数据优先于 _comfy_state
    backend._secrets.set_ep_meta("ep-k1", {"kind": "comfy"})
    assert backend._get_ep_service_kind("ep-k1") == "comfy"

    # 4. 未指定 ep 时异构集群分槽优先
    with backend._lock:
        backend.state["cluster_slots"] = {
            "llm": {"account_id": "a1", "endpoint": "ep-llm"},
            "comfy": {"account_id": "a2", "endpoint": "ep-comfy"},
        }
    assert backend._get_ep_service_kind() == "dual"


# =====================================================================
# 5. ColabPetBackend 容器同对象兼容别名与 AST 门禁测试
# =====================================================================


def test_backend_container_aliases_and_reassignment_sync(isolated_env: SimpleNamespace) -> None:
    """验证 `ColabPetBackend` 兼容别名与容器底层严格同对象，且直接整对象重赋仍保持同步。"""
    backend = cb.ColabPetBackend(
        state_path=isolated_env.tmp / "alias_state.json",
        download_root=isolated_env.tmp / "dl",
        cf_config_path=isolated_env.cf_file,
    )

    # 同对象引用校验
    assert backend._lock is backend._store.lock
    assert backend.state is backend._store.state
    assert backend._event_listeners_lock is backend._bus.lock
    assert backend._event_listeners is backend._bus.listeners
    assert backend._service_meta_by_ep is backend._secrets.service_meta_by_ep
    assert backend._runtime_secrets is backend._secrets.runtime_secrets
    assert backend._comfy_secrets_by_ep is backend._secrets.comfy_secrets_by_ep
    assert backend._comfy_key_bindings_by_ep is backend._secrets.comfy_key_bindings_by_ep
    assert backend._comfy_key_checked_eps is backend._secrets.comfy_key_checked_eps
    assert backend._last_comfy_key_hydrate_m is backend._secrets.last_comfy_key_hydrate_m
    assert not hasattr(cb.ColabPetBackend, "_cb_mod")

    # 模拟既有测试直接整对象重赋字典与回调，验证底层容器自动同步
    backend._comfy_secrets_by_ep = {"ep-sync": "cpk-sync-val"}
    backend._comfy_key_bindings_by_ep = {"ep-sync": ("acct_sync", "https://sync.trycloudflare.com:443")}
    assert backend._secrets.comfy_secrets_by_ep == {"ep-sync": "cpk-sync-val"}
    assert backend._secrets.comfy_key_bindings_by_ep == {"ep-sync": ("acct_sync", "https://sync.trycloudflare.com:443")}
    with backend._lock:
        backend.state["prefs"]["active_account_id"] = "acct_sync"
    assert backend._get_comfy_key_for_url("https://sync.trycloudflare.com", ep="ep-sync") == "cpk-sync-val"

    captured: list[dict] = []
    backend._event_cb = lambda evt: captured.append(evt)
    assert backend._bus.legacy_callback is backend._event_cb
    backend._emit_event({"type": "sync_check"})
    assert captured == [{"type": "sync_check"}]

    # _clear_ephemeral_caches_locked 保留集群活跃分槽端点，清除其余端点
    with backend._lock:
        backend.state["cluster_slots"] = {"comfy": {"account_id": "acct_sync", "endpoint": "ep-keep"}}
        backend._comfy_secrets_by_ep["ep-keep"] = "cpk-keep"
        backend._clear_ephemeral_caches_locked()
    assert "ep-keep" in backend._secrets.comfy_secrets_by_ep
    assert "ep-sync" not in backend._secrets.comfy_secrets_by_ep


def test_phase3_core_modules_ast_complexity_and_forbidden_imports() -> None:
    """AST 门禁：确保 5 个核心模块无禁导依赖，且圈复杂度满足 Phase 3 目标上限。"""
    forbidden = {"PIL", "aiohttp", "pywin32", "win32api", "win32con", "win32gui"}
    limits = {
        PET_DIR / "__init__.py": 5,
        PET_DIR / "state_store.py": 20,
        PET_DIR / "secret_store.py": 20,
        PET_DIR / "context.py": 20,
        PET_DIR / "colab_backend.py": 25,
    }
    for file_path, cc_limit in limits.items():
        tree = ast.parse(file_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".")[0] not in forbidden, f"{file_path.name} 违规导入 {alias.name}"
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".")[0] not in forbidden, f"{file_path.name} 违规导入 {node.module}"
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                cc = _calc_ast_cc(node)
                assert cc <= cc_limit, f"{file_path.name}:{node.name} 圈复杂度 {cc} 超过上限 {cc_limit}"
