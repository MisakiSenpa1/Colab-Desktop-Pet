"""Phase 2 路 2：`colab_core/runtime.py` 4 模块正交拆解与兼容门面离线回归测试。

全量使用 `tmp_path` 与 `COLAB_PET_MOCK_VAULT_PATH` 沙箱隔离，严禁触碰宿主机凭据库与状态文件。
"""

from __future__ import annotations

import ast
import io
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

PKG_DIR = Path(__file__).resolve().parents[1]
if str(PKG_DIR) not in sys.path:
    sys.path.insert(0, str(PKG_DIR))

import colab_core  # noqa: E402
import colab_security as csec  # noqa: E402
from colab_core import deploy_state, exec_runner, ledger, reasoning_specs, runtime  # noqa: E402


@pytest.fixture()
def isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """创建隔离的凭据沙箱、隧道配置与诊断日志路径。"""
    csec.clear_ephemeral_secrets()
    vault_file = tmp_path / "mock_vault.json"
    cf_file = tmp_path / ".cf_tunnel_secret.json"
    diag_log = tmp_path / "logs" / "colab_diag.log"
    monkeypatch.setenv("COLAB_PET_MOCK_VAULT_PATH", str(vault_file))
    monkeypatch.setenv("COLAB_PET_CF_CONFIG_PATH", str(cf_file))
    monkeypatch.setenv("COLAB_PET_DIAG_LOG_PATH", str(diag_log))
    yield SimpleNamespace(tmp=tmp_path, vault_file=vault_file, cf_file=cf_file, diag_log=diag_log)
    csec.clear_ephemeral_secrets()


# =========================================================================
# 1. 全模型家族 infer_model_reasoning_spec 回归矩阵与规范化辅助
# =========================================================================


@pytest.mark.parametrize(
    ("model_id", "api_format", "expected_levels", "expected_default", "expected_source"),
    [
        ("", "openai", ["none", "low", "medium", "high", "xhigh"], "", "fallback"),
        ("", "anthropic", ["none", "low", "medium", "high", "xhigh"], "", "fallback"),
        ("google/hydrogen-pro", "openai", ["low", "medium", "high", "xhigh"], "high", "inferred"),
        ("vendor/carbon-1", "openai", ["low", "medium", "high"], "high", "inferred"),
        ("argon-preview", "openai", ["low", "medium", "high"], "high", "inferred"),
        ("gemini-4-ultra", "gemini", ["low", "medium", "high"], "high", "inferred"),
        ("google/gemini-3-flash-preview", "gemini", ["minimal", "low", "medium", "high"], "medium", "inferred"),
        ("google/gemini-3-pro", "gemini", ["low", "medium", "high"], "high", "inferred"),
        ("gemini-2.5-pro", "gemini", ["low", "medium", "high"], "medium", "inferred"),
        ("gemini-2.5-flash", "gemini", ["none", "low", "medium", "high"], "medium", "inferred"),
        ("anthropic/claude-opus-4-6", "openai", ["none", "low", "medium", "high", "max"], "", "inferred"),
        ("claude-sonnet-4.6", "anthropic", ["none", "low", "medium", "high", "max"], "", "inferred"),
        ("custom-gateway-4.6", "anthropic", ["none", "low", "medium", "high", "max"], "", "inferred"),
        ("claude-3-7-sonnet", "openai", ["none", "low", "medium", "high"], "", "inferred"),
        ("custom-proxy-model", "anthropic", ["none", "low", "medium", "high"], "", "inferred"),
        ("openai/gpt-5.2-pro", "openai", ["none", "low", "medium", "high", "xhigh"], "medium", "inferred"),
        ("gpt-5.3", "openai_responses", ["none", "low", "medium", "high", "xhigh"], "medium", "inferred"),
        ("gpt-5.4-preview", "openai", ["none", "low", "medium", "high", "xhigh"], "medium", "inferred"),
        ("gpt-5-codex-mini", "openai", ["none", "low", "medium", "high", "xhigh"], "medium", "inferred"),
        ("gpt-5-mini", "openai", ["minimal", "low", "medium", "high"], "medium", "inferred"),
        ("o1-preview", "openai", ["low", "medium", "high"], "medium", "inferred"),
        ("o3-mini", "openai", ["low", "medium", "high"], "medium", "inferred"),
        ("o4-mini", "openai", ["low", "medium", "high"], "medium", "inferred"),
        ("deepseek/deepseek-r1-distill-qwen3", "openai", ["low", "medium", "high"], "medium", "inferred"),
        ("deepseek-reasoner", "openai", ["low", "medium", "high"], "medium", "inferred"),
        ("qwq-32b", "openai", ["low", "medium", "high"], "medium", "inferred"),
        ("deepseek-chat", "openai", ["none", "low", "medium", "high", "xhigh"], "", "inferred"),
        ("qwen3-235b", "openai", ["none", "low", "medium", "high", "xhigh"], "", "inferred"),
        ("glm-4.5-air", "openai", ["none", "low", "medium", "high", "xhigh"], "", "inferred"),
        ("glm-5", "openai", ["none", "low", "medium", "high", "xhigh"], "", "inferred"),
        ("kimi-k2-thinking", "openai", ["none", "low", "medium", "high", "xhigh"], "", "inferred"),
        ("llama-3.3-70b", "openai", ["none", "low", "medium", "high", "xhigh"], "", "fallback"),
    ],
)
def test_infer_model_reasoning_spec_full_matrix(
    isolated_env: SimpleNamespace,
    model_id: str,
    api_format: str,
    expected_levels: list[str],
    expected_default: str,
    expected_source: str,
) -> None:
    """验证数据驱动规则表覆盖全部模型家族与优先级，且每次返回独立可变列表。"""
    _ = isolated_env
    spec = reasoning_specs.infer_model_reasoning_spec(model_id, api_format=api_format)
    assert spec == {
        "reasoning": True,
        "levels": expected_levels,
        "default_level": expected_default,
        "source": expected_source,
    }
    # 修改返回列表不得污染后续调用
    spec["levels"].append("mutated")
    spec2 = runtime.infer_model_reasoning_spec(model_id, api_format=api_format)
    assert spec2["levels"] == expected_levels


def test_reasoning_effort_and_level_list_normalization(isolated_env: SimpleNamespace) -> None:
    """验证思考强度别名映射与档位列表排序去重。"""
    _ = isolated_env
    assert reasoning_specs.normalize_reasoning_effort_value("OFF") == "none"
    assert reasoning_specs.normalize_reasoning_effort_value("extra-high") == "xhigh"
    assert reasoning_specs.normalize_reasoning_effort_value("maximum") == "max"
    assert reasoning_specs.normalize_reasoning_effort_value("med") == "medium"
    assert reasoning_specs.normalize_reasoning_effort_value("invalid", default="low") == "low"
    assert reasoning_specs.normalize_reasoning_level_list(
        ["maximum", {"effort": "min"}, "auto", "off", {"id": "extra high"}, "medium", "invalid"]
    ) == ["none", "minimal", "medium", "xhigh", "max"]


# =========================================================================
# 2. consume_deploy_line 动态 tunnel_pattern 与 7 步阶段解析
# =========================================================================


def test_consume_deploy_line_phases_progress_and_dynamic_tunnel_pattern(isolated_env: SimpleNamespace) -> None:
    """验证双阶段部署、步骤横幅、下载进度速率以及动态 tunnel_pattern 捕获行为。"""
    _ = isolated_env
    st = deploy_state.default_deploy_state()
    st["gpu"] = "G4"
    ctx: dict[str, Any] = {"phase_idx": 1, "phase_total": 1}
    prefs = {"boot_llm_quant": "UD-Q4_K_XL", "boot_llm_mmproj": True, "boot_llm27_mmproj": True}
    custom_pat = r"https://(?:(?!api\.)[a-z0-9-]+\.trycloudflare\.com|(?:comfy|llm)\.demo-studio\.example)(?![a-z0-9.-])"

    # 前置步骤日志（含 sessions 核验、分配、复用与上传 rclone）
    deploy_state.consume_deploy_line("[本机执行] colab.exe sessions", st, ctx, prefs)
    assert st["pct"] == 4 and "核验云端会话" in st["stage_title"]
    deploy_state.consume_deploy_line("Creating session...", st, ctx, prefs)
    assert st["pct"] == 6 and "申请 G4 实例" in st["stage_title"]
    deploy_state.consume_deploy_line("colab.exe upload rclone.conf", st, ctx, prefs)
    assert st["pct"] == 13 and st["stage_title"] == "注入云盘配置"

    # 阶段 1/2：qwen_27b（验证动态 Pattern 捕获 LLM_TUNNEL_URL 且中间阶段不提前返回主隧道）
    deploy_state.consume_deploy_line("=== DEPLOY_PHASE 1/2: qwen_27b ===", st, ctx, prefs)
    assert ctx["phase_idx"] == 1 and ctx["phase_total"] == 2 and ctx["phase_is_llm"] is True
    deploy_state.consume_deploy_line("=== [3/7] 准备 llama-server 引擎 ===", st, ctx, prefs)
    assert st["step_idx"] == 3 and "llama-server" in st["sub_detail"]
    deploy_state.consume_deploy_line("cmake 正在编译 CUDA 核", st, ctx, prefs)
    assert "编译" in st["sub_detail"]
    deploy_state.consume_deploy_line("=== [4/7] 下载模型权重 ===", st, ctx, prefs)
    deploy_state.consume_deploy_line("DEPLOY_DL_TOTAL: 2", st, ctx, prefs)
    deploy_state.consume_deploy_line("[开始下载] qwen3.8-27b.gguf", st, ctx, prefs)
    deploy_state.consume_deploy_line("progress (50%) DL: 120.5 MiB/s", st, ctx, prefs)
    assert "50%" in st["sub_detail"] and "120.5 MiB/s" in st["sub_detail"]
    deploy_state.consume_deploy_line("[完成] qwen3.8-27b.gguf", st, ctx, prefs)
    assert ctx["dl_done"] == 1

    # 默认无 tunnel_pattern 不捕获自定义域名，传入动态正则后正确捕获
    assert deploy_state.consume_deploy_line("LLM_TUNNEL_URL: https://llm.demo-studio.example", st, ctx, prefs) == (None, False)
    assert not ctx.get("captured_llm_tunnel")
    assert deploy_state.consume_deploy_line(
        "LLM_TUNNEL_URL: https://llm.demo-studio.example",
        st,
        ctx,
        prefs,
        tunnel_pattern=custom_pat,
    ) == (None, False)
    assert ctx["captured_llm_tunnel"] == "https://llm.demo-studio.example"
    # 阶段 1/2 的 SERVICE_TUNNEL_URL 仅推进本阶段就绪，不提前结束全部部署
    assert deploy_state.consume_deploy_line(
        "SERVICE_TUNNEL_URL: https://llm.demo-studio.example",
        st,
        ctx,
        prefs,
        tunnel_pattern=custom_pat,
    ) == (None, False)
    assert "阶段 1/2 就绪" in st["stage_title"]

    # 阶段 2/2：anima 完成后返回最终隧道且仅首次触发 should_warm_meta=True
    deploy_state.consume_deploy_line("=== DEPLOY_PHASE 2/2: anima ===", st, ctx, prefs)
    deploy_state.consume_deploy_line("=== [6/7] 启动 ComfyUI 服务 ===", st, ctx, prefs)
    deploy_state.consume_deploy_line("已后台拉起 ComfyUI (PID=1234)", st, ctx, prefs)
    assert "8188" in st["sub_detail"]
    tun_url, should_warm = deploy_state.consume_deploy_line(
        "\x1b[32mCOMFYUI_TUNNEL_URL: https://comfy.demo-studio.example\x1b[0m",
        st,
        ctx,
        prefs,
        tunnel_pattern=re.compile(custom_pat),
    )
    assert tun_url == "https://comfy.demo-studio.example"
    assert should_warm is True and st["pct"] == 100
    tun_url_2, should_warm_2 = deploy_state.consume_deploy_line(
        "SERVICE_TUNNEL_URL: https://comfy.demo-studio.example",
        st,
        ctx,
        prefs,
        tunnel_pattern=custom_pat,
    )
    assert tun_url_2 == "https://comfy.demo-studio.example" and should_warm_2 is False


# =========================================================================
# 3. load_pet_state_dict 损坏备份恢复、账本迁移与偏好清洗
# =========================================================================


def test_load_pet_state_dict_corrupt_backup_and_field_sanitization(isolated_env: SimpleNamespace) -> None:
    """验证损坏 JSON 与非法根节点自动生成 .corrupt-* 备份恢复默认态，以及全字段清洗与账本切换。"""
    tmp = isolated_env.tmp

    # 3a. 损坏 JSON 备份恢复
    corrupt_json = tmp / "pet_broken.json"
    corrupt_json.write_text("{not valid json", encoding="utf-8")
    recovered = ledger.load_pet_state_dict(
        corrupt_json,
        is_allowed_tunnel_fn=lambda u: u.startswith("https://"),
        sanitize_label_fn=lambda s, fb: s.strip() or fb,
    )
    assert recovered["prefs"]["boot_profile"] == "anima"
    baks = list(tmp.glob("pet_broken.corrupt-*.json"))
    assert len(baks) == 1 and baks[0].read_text(encoding="utf-8") == "{not valid json"

    # 3b. 根节点非 dict（如数组）同样备份并恢复
    non_dict = tmp / "pet_list.json"
    non_dict.write_text("[1, 2, 3]", encoding="utf-8")
    recovered_list = ledger.load_pet_state_dict(
        non_dict,
        is_allowed_tunnel_fn=lambda _u: True,
        sanitize_label_fn=lambda s, fb: s or fb,
    )
    assert isinstance(recovered_list, dict) and recovered_list["account_profiles"] == []
    assert list(tmp.glob("pet_list.corrupt-*.json"))

    # 3c. 合法状态文件加载：保留实测费率、过滤非法隧道/路径穿越与敏感 agent_api_key、兼容 15/30/60 闲置停卡档位
    valid_state = tmp / "pet_valid.json"
    valid_state.write_text(
        json.dumps(
            {
                "prefs": {
                    "boot_profiles": ["minimax", "qwen_27b"],
                    "idle_auto_stop_minutes": 15,
                    "agent_api_key": "sk-should-be-stripped-123456",
                    "agent_reasoning_effort": "extra-high",
                },
                "measured_gpu_rates": {"G4:STD": 9.88, "L4:HM": 99.0, "T4:STD": -1.0},
                "tunnel_cache": {
                    "ep-ok": "https://demo-ok.trycloudflare.com/",
                    "ep-bad": "http://evil.example.com",
                },
                "managed_binding": {"account_id": "acct@1/..", "session": "comfy", "endpoint": "ep-ok"},
                "account_profiles": [
                    {"id": "acct1", "label": "Colab 1号", "email_masked": "a***@example.com", "last_balance": 888.5, "tier": "ultra"},
                    {"id": "acct1", "label": "重复应去重"},
                ],
                "account_ledgers": {"acct1": {"last_balance": 888.5, "unknown_field": "drop_me"}},
                "pending_pull_items": [
                    {"base_url": "https://demo-ok.trycloudflare.com", "filename": "ok.png", "subfolder": "sub/a", "type": "output"},
                    {"base_url": "https://demo-ok.trycloudflare.com", "filename": "../escape.png", "subfolder": "", "type": "output"},
                    {"base_url": "https://demo-ok.trycloudflare.com", "filename": "a.png", "subfolder": "../bad", "type": "output"},
                    {"base_url": "http://127.0.0.1:8188", "filename": "local.png", "subfolder": "", "type": "output"},
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    loaded = ledger.load_pet_state_dict(
        valid_state,
        is_allowed_tunnel_fn=lambda u: u.startswith("https://demo-ok.trycloudflare.com"),
        sanitize_label_fn=lambda s, fb: s.strip() or fb,
    )
    assert loaded["prefs"]["idle_auto_stop_minutes"] == 15
    assert loaded["prefs"]["agent_reasoning_effort"] == "xhigh"
    assert "agent_api_key" not in loaded["prefs"]
    assert loaded["measured_gpu_rates"]["G4:STD"] == 9.88
    assert "L4:HM" not in loaded["measured_gpu_rates"]
    assert loaded["tunnel_cache"] == {"ep-ok": "https://demo-ok.trycloudflare.com"}
    assert loaded["managed_binding"] == {"account_id": "acct1", "session": "comfy", "endpoint": "ep-ok"}
    assert len(loaded["account_profiles"]) == 1
    assert loaded["account_ledgers"] == {"acct1": {"last_balance": 888.5}}
    assert [it["filename"] for it in loaded["pending_pull_items"]] == ["ok.png", "local.png"]

    # 3d. upsert_account_profile 与 sync_active_account_ledger 自动重标等级前缀
    loaded["prefs"]["active_account_id"] = "acct1"
    loaded["last_balance"] = 920.0
    ledger.sync_active_account_ledger(loaded, balance=920.0, is_running=True, tier="ultra")
    assert loaded["account_profiles"][0]["label"] == "Ultra 1号"
    assert loaded["account_ledgers"]["acct1"]["last_balance"] == 920.0


# =========================================================================
# 4. exec_runner 流式执行与后台任务台账回归
# =========================================================================


def test_exec_runner_streaming_and_task_manager(isolated_env: SimpleNamespace) -> None:
    """验证拆分后的 ProcessRunner.run_streaming 脱敏分发与 TaskManager 互斥防重。"""
    diag = exec_runner.DiagLogger(log_path=isolated_env.diag_log, max_bytes=4096)
    runner = exec_runner.ProcessRunner(diag=diag)
    tm = exec_runner.TaskManager(diag=diag)

    class _FakeStreamProc:
        def __init__(self, *_a: Any, **_kw: Any) -> None:
            self.pid = None
            self.stdout = io.StringIO("line1\rAuthorization: Bearer cpk-SecretStreamKey_12345678\nline3")

        def wait(self) -> int:
            return 0

    seen: list[str] = []
    res = runner.run_streaming(
        ["demo"],
        on_line=seen.append,
        popen_fn=_FakeStreamProc,
        task_name="stream_test",
    )
    assert res.ok is True
    assert len(seen) == 3
    assert all("cpk-SecretStreamKey_12345678" not in ln for ln in seen)

    h = tm.submit("ok_job", lambda _tok: res)
    assert h.accepted is True and h.join(timeout=3.0) is True
    assert tm.get_record("ok_job")["status"] == "succeeded"


# =========================================================================
# 5. runtime.py 薄门面符号完整性、零禁用依赖与无循环导入校验
# =========================================================================


def test_runtime_facade_completeness_and_architecture_invariants(isolated_env: SimpleNamespace) -> None:
    """验证 runtime.py 门面完整透传全部符号、对象身份一致、零禁用三方依赖且 __init__.py __all__ 完备。"""
    _ = isolated_env
    expected_from_modules: dict[Any, list[str]] = {
        reasoning_specs: [
            "SUPPORTED_AGENT_FORMATS",
            "SUPPORTED_CONCRETE_REASONING_LEVELS",
            "SUPPORTED_REASONING_EFFORTS",
            "infer_model_reasoning_spec",
            "normalize_reasoning_effort_value",
            "normalize_reasoning_level_list",
        ],
        exec_runner: [
            "CREATE_NO_WINDOW",
            "MAX_DIAG_LOG_BYTES",
            "CancelToken",
            "CommandResult",
            "DiagLogger",
            "ProcessRunner",
            "TaskHandle",
            "TaskManager",
            "TaskRecord",
            "TaskStatus",
            "clean_subprocess_env",
            "extract_framed_json",
            "get_diag_logger",
            "kill_process_tree",
            "resolve_diag_log_path",
        ],
        deploy_state: [
            "COMFY_PROFILES",
            "DEFAULT_DEPLOY_TUNNEL_URL_PATTERN",
            "DEPLOY_DL_TOTAL_RE",
            "DEPLOY_LLM_TUNNEL_LINE_RE",
            "DEPLOY_PHASE_RE",
            "DEPLOY_STEP_BASE_PCT",
            "DEPLOY_TUNNEL_LINE_RE",
            "DL_SPEED_RE",
            "LLM_27B_BASE_SHARDS",
            "LLM_PROFILES",
            "LLM_QUANT_BASE_SHARDS",
            "PCT_TOKEN_RE",
            "PROFILE_LABELS",
            "STEP_BANNER_RE",
            "SUPPORTED_BOOT_GPUS",
            "SUPPORTED_LLM_QUANTS",
            "SUPPORTED_PROFILES",
            "_is_traceback_noise_line",
            "_pick_meaningful_tail_error",
            "build_ctl_up_args",
            "calc_phase_dl_total",
            "classify_deploy_failure",
            "consume_deploy_line",
            "default_deploy_state",
            "format_elapsed_cn",
            "normalize_profiles",
            "parse_llm_boot_params",
            "parse_profiles_cli_arg",
            "profile_bootstrap_spec",
            "recommend_fallback_gpu",
            "resolve_high_mem",
            "scale_phase_pct",
            "select_default_gpu",
        ],
        ledger: [
            "ACCOUNT_LEDGER_KEYS",
            "ACCOUNT_TIER_LABELS",
            "DEFAULT_MEASURED_GPU_RATES",
            "DEFAULT_PREFS",
            "MAX_PENDING_PULL_ITEMS",
            "SCARCE_GPU_TTL_SEC",
            "SUPPORTED_IDLE_STOP_MINUTES",
            "_backup_corrupt_state",
            "collect_formal_outputs",
            "default_baseline",
            "default_state_dict",
            "estimate_session_start_balance",
            "extract_history_error",
            "extract_slot_tps",
            "format_comfy_error",
            "format_pool_tier_summary",
            "gpu_rate_key",
            "infer_account_tier_and_baseline",
            "load_pet_state_dict",
            "restore_ledger_for_account",
            "sanitize_agent_providers",
            "sanitize_model_capabilities",
            "sanitize_prefs_dict",
            "short_gpu_name",
            "snapshot_current_ledger",
            "sync_active_account_ledger",
            "upsert_account_profile",
        ],
    }

    for src_mod, names in expected_from_modules.items():
        for sym in names:
            assert hasattr(runtime, sym), f"runtime.py 门面缺失符号: {sym}"
            assert getattr(runtime, sym) is getattr(src_mod, sym), f"runtime.{sym} 未指向 {src_mod.__name__}.{sym}"
            assert sym in runtime.__all__, f"runtime.__all__ 缺失符号: {sym}"

    # 校验 colab_core.__all__ 包含新增模块
    for mod_name in ("agent", "reasoning_specs", "exec_runner", "deploy_state", "ledger", "runtime"):
        assert mod_name in colab_core.__all__

    # 静态 AST 校验：4 个内核模块与 runtime 门面严禁引入 PIL / aiohttp / pywin32
    forbidden_roots = {"PIL", "aiohttp", "win32api", "win32con", "win32gui", "pywintypes", "pythoncom"}
    for fname in ("reasoning_specs.py", "exec_runner.py", "deploy_state.py", "ledger.py", "runtime.py"):
        fpath = PKG_DIR / "colab_core" / fname
        tree = ast.parse(fpath.read_text(encoding="utf-8"), filename=fname)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".", 1)[0] not in forbidden_roots, f"{fname} 禁止导入 {alias.name}"
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".", 1)[0] not in forbidden_roots, f"{fname} 禁止从 {node.module} 导入"
