"""Phase 1 · Wave B（路 3）：3 个超长核心函数拆解与 37 处 `S110` 清理回归测试。

严格离线沙箱执行：依赖 `tests/conftest.py` 与本地独立临时路径隔离，绝不触碰宿主机凭据或运行态文件。
"""

from __future__ import annotations

import ast
import asyncio
import hashlib
import hmac
import subprocess
import sys
import time
from collections import OrderedDict
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
import colab_bootstrap_qwen_llm as boot_llm  # noqa: E402
import colab_ctl  # noqa: E402
import colab_pet_bridge_node as bridge_node  # noqa: E402


def _calc_ast_cc(node: ast.AST) -> int:
    """计算单个函数 AST 节点的圈复杂度。"""
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
# 1. 静态门禁：3 个超长函数行数/CC 达标、云端禁导 colab_core、S110 = 0
# =====================================================================


def test_decomposed_three_long_functions_size_cc_and_zero_s110() -> None:
    """验证 3 个超长函数及拆出辅助均 <= 80L / CC <= 25，云端 5 脚本零 colab_core 导入，且本组 ruff 零错误。"""
    decomposed_targets = {
        "colab_ctl.py": (
            "cmd_tunnel",
            "_extract_fast_tunnel_candidate",
            "_try_reuse_fast_tunnel",
            "_probe_and_print_reusable_tunnel",
            "_build_remote_tunnel_script",
            "_verify_public_llm_tunnel_once",
            "_verify_tunnel_end_to_end",
        ),
        "colab_pet_bridge_node.py": (
            "create_comfy_auth_middleware",
            "_consume_auth_ticket",
            "_set_auth_cookie_response",
            "_prune_and_check_ratelimit",
            "_handle_pet_auth_request",
            "_build_unauthorized_response",
        ),
        "colab_bootstrap_qwen_llm.py": (
            "ensure_llama_server",
            "_try_pull_llama_gdrive_cache",
            "_try_pull_llama_github_prebuilt",
            "_select_cuda_build_env",
            "_install_compiled_llama_artifacts",
            "_compile_llama_server_from_source",
            "_upload_llama_gdrive_cache",
        ),
    }

    for rel, fn_names in decomposed_targets.items():
        tree = ast.parse((PACKAGE_DIR / rel).read_text(encoding="utf-8"))
        nodes = {
            n.name: n
            for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        for fn in fn_names:
            assert fn in nodes, f"{rel} 缺少预期函数 {fn}"
            node = nodes[fn]
            span = node.end_lineno - node.lineno + 1
            cc = _calc_ast_cc(node)
            assert span <= 80, f"{rel}:{fn} 行数 {span} 超过上限 80"
            assert cc <= 25, f"{rel}:{fn} 圈复杂度 {cc} 超过上限 25"

    # 5 个云端脚本严禁导入 colab_core
    cloud_scripts = (
        "colab_pet_bridge_node.py",
        "colab_bootstrap_qwen_llm.py",
        "colab_bootstrap_drama3.py",
        "colab_sse_gateway.py",
        "colab_remote_common.py",
    )
    for rel in cloud_scripts:
        tree = ast.parse((PACKAGE_DIR / rel).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".")[0] != "colab_core", f"云端脚本 {rel} 严禁导入 colab_core"
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".")[0] != "colab_core", f"云端脚本 {rel} 严禁导入 colab_core"

    # 本组 8 文件 ruff 检查必须 0 告警（S110 = 0 / UP = 0 / C901 = 0）
    owned_files = [
        str(PACKAGE_DIR / rel)
        for rel in (
            "colab_ctl.py",
            "colab_pet_bridge_node.py",
            "colab_bootstrap_qwen_llm.py",
            "colab_bootstrap_drama3.py",
            "colab_sse_gateway.py",
            "colab_remote_common.py",
            "colab_pet/backend_comfy.py",
            "colab_pet/backend_tunnel.py",
        )
    ]
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "ruff",
            "check",
            "--output-format=concise",
            "--target-version",
            "py311",
            "--select",
            "E,F,W,B,S110,UP,C901",
            "--ignore",
            "E501",
            "--config",
            "lint.mccabe.max-complexity=30",
            *owned_files,
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
    )
    assert proc.returncode == 0, f"ruff 发现未清理告警:\n{proc.stdout}\n{proc.stderr}"


# =====================================================================
# 2. colab_ctl.cmd_tunnel 拆解后的候选提取、双向鉴权与端到端校验回归
# =====================================================================


def test_cmd_tunnel_candidate_extraction_and_end_to_end_verify(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 `_extract_fast_tunnel_candidate` 固定域名校验、密钥优先级及 `_verify_tunnel_end_to_end` 鉴权拦截。"""
    # 1) 固定域名与返回 URL 不一致时强制清空 fast_url，防止串用错误入口
    cand = colab_ctl._extract_fast_tunnel_candidate(
        pkt={
            "url": "https://wrong.trycloudflare.com",
            "comfy_api_key": "cpk-remote-1",
            "api_key": "sk-remote-llm",
            "meta": {"kind": "dual", "llm_tunnel_url": "https://llm.custom.example"},
        },
        cf_cfg={
            "tunnel_token": "tok-abc",
            "comfy_url": "https://comfy.custom.example",
            "llm_url": "https://llm.custom.example",
        },
        fallback_llm_key="sk-local-fallback",
    )
    assert cand["fast_url"] == ""
    assert cand["fast_llm_url"] == "https://llm.custom.example"
    assert cand["fast_comfy_key"] == "cpk-remote-1"
    assert cand["fast_llm_key"] == "sk-remote-llm"

    # 2) _verify_tunnel_end_to_end 在 LLM 双向鉴权失败时重试 6 次并抛 RuntimeError
    sleeps: list[float] = []
    monkeypatch.setattr(colab_ctl.time, "sleep", lambda s: sleeps.append(s))
    with patch.object(
        colab_ctl,
        "_probe_llm_endpoint_auth_ok",
        return_value=(False, {"reason": "anonymous_allowed"}),
    ):
        with pytest.raises(RuntimeError, match="连续 6 次失败"):
            colab_ctl._verify_tunnel_end_to_end(
                service_kind="llm",
                tunnel_url="https://llm-bad.trycloudflare.com",
                llm_tunnel_url="",
                fast_comfy_key="",
                fast_llm_key="sk-test-key",
            )
    assert len(sleeps) == 6

    # 3) Comfy + Dual 双向鉴权均通过时正常返回
    with (
        patch.object(
            colab_ctl,
            "_probe_comfy_endpoint_ok",
            return_value=(True, {"devices": [{"name": "NVIDIA L4", "vram_total": 24 * (1024 ** 3)}]}),
        ),
        patch.object(colab_ctl, "_probe_llm_endpoint_auth_ok", return_value=(True, {"ok": True})),
    ):
        colab_ctl._verify_tunnel_end_to_end(
            service_kind="dual",
            tunnel_url="https://comfy-ok.trycloudflare.com",
            llm_tunnel_url="https://llm-ok.trycloudflare.com",
            fast_comfy_key="cpk-ok",
            fast_llm_key="sk-ok",
        )


# =====================================================================
# 3. colab_pet_bridge_node.create_comfy_auth_middleware 拆解回归
# =====================================================================


def test_comfy_auth_middleware_ticket_replay_open_redirect_and_ratelimit() -> None:
    """验证拆解后的票据一次性核销、开放重定向拦截、429 限流与中间件公网鉴权语义。"""
    secret = "cpk-unit-test-secret-01"
    used_nonces: OrderedDict[str, int] = OrderedDict()
    now_ts = int(time.time())
    nonce = "nonce123"
    sig = hmac.new(secret.encode(), f"ticket:{now_ts}:{nonce}:/queue".encode(), hashlib.sha256).hexdigest()
    ticket = f"{now_ts}.{nonce}.{sig}"

    # 1) 合法票据首兑成功，二次重放必拒
    ok1, next1 = bridge_node._consume_auth_ticket(used_nonces, secret, ticket, "/queue")
    assert ok1 is True and next1 == "/queue"
    ok2, _ = bridge_node._consume_auth_ticket(used_nonces, secret, ticket, "/queue")
    assert ok2 is False

    # 2) 开放重定向目标必拒
    bad_next = "//evil.example/phish"
    bad_sig = hmac.new(secret.encode(), f"ticket:{now_ts}:nonce2:{bad_next}".encode(), hashlib.sha256).hexdigest()
    ok_redir, clean_redir = bridge_node._consume_auth_ticket(
        used_nonces, secret, f"{now_ts}.nonce2.{bad_sig}", bad_next
    )
    assert ok_redir is False and clean_redir == "/"

    # 3) 速率限制窗口淘汰与 5 次超限触发
    failures: dict[str, list[float]] = {"1.2.3.4": [10.0, 20.0, 30.0, 40.0, 50.0]}
    limited, hist = bridge_node._prune_and_check_ratelimit(failures, "1.2.3.4", 55.0)
    assert limited is True and len(hist) == 5
    limited_after, hist_after = bridge_node._prune_and_check_ratelimit(failures, "1.2.3.4", 200.0)
    assert limited_after is False and hist_after == []

    # 4) 中间件公网无密钥 503 fail-closed 与 Bearer 放行
    mw_no_key = bridge_node.create_comfy_auth_middleware(lambda: "")

    class _FakeReq:
        def __init__(self, path: str, headers: dict[str, str], remote: str = "203.0.113.5") -> None:
            self.path = path
            self.method = "GET"
            self.headers = headers
            self.remote = remote
            self.host = "demo.trycloudflare.com"
            self.cookies: dict[str, str] = {}
            self.query: dict[str, str] = {}
            self.secure = True

    async def _ok_handler(_req):
        return SimpleNamespace(status=200, text="OK")

    async def _run_cases():
        r503 = await mw_no_key(_FakeReq("/system_stats", {"Cf-Connecting-Ip": "203.0.113.5"}), _ok_handler)
        assert r503.status == 503

        mw = bridge_node.create_comfy_auth_middleware(lambda: secret)
        r401 = await mw(_FakeReq("/system_stats", {"Cf-Connecting-Ip": "203.0.113.5"}), _ok_handler)
        assert r401.status == 401

        r200 = await mw(
            _FakeReq("/system_stats", {"Cf-Connecting-Ip": "203.0.113.5", "Authorization": f"Bearer {secret}"}),
            _ok_handler,
        )
        assert r200.status == 200

        r_health = await mw(_FakeReq("/__pet_health", {"Cf-Connecting-Ip": "203.0.113.5"}), _ok_handler)
        assert r_health.status == 200

    asyncio.run(_run_cases())


# =====================================================================
# 4. colab_bootstrap_qwen_llm.ensure_llama_server 三级回退管线测试
# =====================================================================


def test_ensure_llama_server_three_tier_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """验证 `ensure_llama_server` 的本地复用 -> 云盘缓存 -> GitHub 预编译 -> 源码编译分级回退与上传控制。"""
    fake_conf = tmp_path / "rclone.conf"
    fake_conf.write_text("[gdrive_save]\ntype = drive\n", encoding="utf-8")
    fake_rclone = tmp_path / "rclone"
    fake_rclone.write_text("", encoding="utf-8")
    monkeypatch.setattr(boot_llm, "RCLONE_CONF", fake_conf)
    monkeypatch.setattr(boot_llm.shutil, "which", lambda name: str(fake_rclone) if name == "rclone" else None)

    fake_bin = (tmp_path / "llama-server", "/usr/lib")
    uploads: list[str] = []

    # 1) 本地已存在 -> 立即复用，不拉云盘不下载不编译
    with (
        patch.object(boot_llm, "_verify_llama_cuda", return_value=fake_bin),
        patch.object(boot_llm, "_try_pull_llama_gdrive_cache") as m_gd,
        patch.object(boot_llm, "_try_pull_llama_github_prebuilt") as m_gh,
    ):
        assert boot_llm.ensure_llama_server("89") == fake_bin
        m_gd.assert_not_called()
        m_gh.assert_not_called()

    # 2) 云盘缓存命中 -> 直接返回且不重复上传云盘
    with (
        patch.object(boot_llm, "_verify_llama_cuda", return_value=None),
        patch.object(boot_llm, "_try_pull_llama_gdrive_cache", return_value=fake_bin),
        patch.object(boot_llm, "_try_pull_llama_github_prebuilt") as m_gh,
        patch.object(boot_llm, "_upload_llama_gdrive_cache", side_effect=lambda *a: uploads.append("up")),
    ):
        assert boot_llm.ensure_llama_server("89") == fake_bin
        m_gh.assert_not_called()
        assert uploads == []

    # 3) 云盘未命中、GitHub 预编译包未命中 -> 回退源码编译并回传云盘
    with (
        patch.object(boot_llm, "_verify_llama_cuda", return_value=None),
        patch.object(boot_llm, "_try_pull_llama_gdrive_cache", return_value=None),
        patch.object(boot_llm, "_try_pull_llama_github_prebuilt", return_value=None),
        patch.object(boot_llm, "_compile_llama_server_from_source", return_value=fake_bin) as m_build,
        patch.object(boot_llm, "_upload_llama_gdrive_cache", side_effect=lambda *a: uploads.append("up")),
    ):
        assert boot_llm.ensure_llama_server("120") == fake_bin
        m_build.assert_called_once_with("120")
        assert uploads == ["up"]


# =====================================================================
# 5. backend_comfy & backend_tunnel 清理 S110 后通过 suppress_logged 记录脱敏诊断
# =====================================================================


def test_backend_comfy_and_tunnel_log_suppressed_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """验证替换 `except Exception: pass` 后，非致命探针异常会经 `suppress_logged` 记录脱敏 DEBUG 日志。"""
    diag_log = tmp_path / "phase1_suppress_diag.log"
    monkeypatch.setenv("COLAB_PET_DIAG_LOG_PATH", str(diag_log))
    from colab_core import exec_runner

    # 重置单例 logger 指向本测试日志文件
    monkeypatch.setattr(exec_runner, "_GLOBAL_DIAG_LOGGER", None)

    backend = cb.ColabPetBackend(
        state_path=tmp_path / "sup_state.json",
        download_root=tmp_path / "dl",
        cf_config_path=tmp_path / "cf.json",
    )
    with patch.object(
        backend,
        "_comfy_http_get",
        side_effect=RuntimeError("mock queue failure Bearer cpk-secret-token-abcdef1234567890"),
    ):
        assert backend._comfy_svc._fetch_comfy_queue_counts("https://demo.trycloudflare.com") == (0, 0)
        backend._tunnel_svc.ensure_legacy_cmd_file("https://demo.trycloudflare.com", ep="ep-1")

    assert diag_log.is_file()
    log_txt = diag_log.read_text(encoding="utf-8")
    assert "comfy.queue" in log_txt
    assert "tunnel.legacy_cmd" in log_txt
    assert "cpk-secret-token-abcdef1234567890" not in log_txt
    assert "[REDACTED" in log_txt
