"""Phase 3 Wave 3B 路 1 专项离线回归测试：验证 `backend_agent.py` 与 `backend_accounts.py` 重构质量。

覆盖范围：
1. 静态架构契约：`_cb_mod` 彻底清零、零直接摸底 `b.state`/`b._lock`/`b._save_state`/`b._emit_event`/内存凭据字典、
   零禁用重依赖（`PIL`/`aiohttp`/`pywin32`）、容器同对象绑定及 AST 圈复杂度达标；
2. `_patch_comfy_api_graph` 拆解后的多拓扑（含 guider/noise/sigmas、标题兜底与参数钳制）行为等价性；
3. `AgentService` 多供应商配置、源站硬绑定清密、Colab LLM 探测、大脑路由、多会话无死锁管理与 4 级工作流模板兜底；
4. `AgentService` 流式轮次双引擎编排（`NativeLightDriver` / `ClaudeCodeDriver`）、事件总线脱敏及远端执行安全审批门禁；
5. `AccountVaultService` 基于 `_store`/`_secrets`/`_bus`/`_ctx` 的 OAuth2 绑定、秒切账号与账本隔离、解绑自动切换、
   rclone 云盘授权/配额查询、工具路径设置与诊断包导出全链路。
"""

from __future__ import annotations

import ast
import json
import re
import struct
import sys
import threading
import time
import zlib
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

import backend_accounts as bacc_mod  # noqa: E402
import backend_agent as ba_mod  # noqa: E402
import colab_backend as cb  # noqa: E402
import colab_security as csec  # noqa: E402
from colab_core.runtime import CancelToken  # noqa: E402


def _calc_ast_cc(node: ast.AST) -> int:
    """计算单个函数节点的 AST 圈复杂度。"""
    cc = 1
    for child in ast.walk(node):
        if isinstance(
            child,
            (ast.If, ast.While, ast.For, ast.AsyncFor, ast.ExceptHandler, ast.IfExp, ast.With, ast.AsyncWith, ast.Assert),
        ):
            cc += 1
        elif isinstance(child, ast.BoolOp):
            cc += len(child.values) - 1
        elif isinstance(child, ast.comprehension):
            cc += 1 + len(child.ifs)
    return cc


def _make_png_with_prompt(path: Path, prompt_obj: dict[str, Any]) -> None:
    """用纯标准库生成带 `tEXt:prompt` 元数据的最小合法 1x1 PNG 文件。"""
    def _chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    idat = zlib.compress(b"\x00\x00\x00\x00")
    txt_data = b"prompt\x00" + json.dumps(prompt_obj, ensure_ascii=False).encode("utf-8")
    png_bytes = (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"tEXt", txt_data)
        + _chunk(b"IDAT", idat)
        + _chunk(b"IEND", b"")
    )
    path.write_bytes(png_bytes)


def _wait_until(predicate, timeout: float = 5.0, err_msg: str = "等待超时"):
    """轮询等待条件满足，用于异步后台线程测试同步。"""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        val = predicate()
        if val:
            return val
        time.sleep(0.02)
    raise AssertionError(err_msg)


@pytest.fixture()
def isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """构造完全与宿主机隔离的沙箱目录、模拟凭据库与诊断日志路径。"""
    vault_file = tmp_path / "mock_vault.json"
    diag_log = tmp_path / "diag.log"
    cf_file = tmp_path / "cf_tunnel_secret.json"
    downloads_dir = tmp_path / "downloads"
    downloads_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("COLAB_PET_MOCK_VAULT_PATH", str(vault_file))
    monkeypatch.setenv("COLAB_PET_DIAG_LOG_PATH", str(diag_log))
    monkeypatch.setenv("COLAB_PET_CF_CONFIG_PATH", str(cf_file))
    monkeypatch.delenv("COLAB_AGENT_AUTO_APPROVE_EXEC", raising=False)
    monkeypatch.setattr(cb, "_default_cf_secret_path", lambda: cf_file)

    return SimpleNamespace(
        tmp_path=tmp_path,
        vault_file=vault_file,
        diag_log=diag_log,
        cf_file=cf_file,
        downloads_dir=downloads_dir,
    )


def _make_backend(env: SimpleNamespace, name: str = "state") -> cb.ColabPetBackend:
    """在沙箱目录中构造独立的 ColabPetBackend 实例。"""
    state_file = env.tmp_path / f"{name}.json"
    backend = cb.ColabPetBackend(state_path=state_file)
    backend.download_root = env.downloads_dir
    return backend


# =========================================================================
# 1. 静态架构契约、容器绑定与复杂度核验
# =========================================================================


def test_phase3_services_static_contract_and_container_binding(isolated_env: SimpleNamespace) -> None:
    """校验两文件 0 处 _cb_mod、0 处直接摸底内部容器、无禁用依赖、CC <= 30 且容器绑定一致。"""
    forbidden_imports = {"PIL", "Pillow", "aiohttp", "win32cred", "win32api", "pywintypes"}
    legacy_re = re.compile(
        r"(_cb_mod\s*\(|\bb\.state\b|\bb\._lock\b|\bb\._save_state\b|\bb\._emit_event\b"
        r"|\bb\._service_meta_by_ep\b|\bb\._runtime_secrets\b|\bb\._comfy_secrets_by_ep\b"
        r"|\bb\._comfy_key_bindings_by_ep\b|\bb\._comfy_key_checked_eps\b"
        r"|\bb\._get_colab_py\b|\bb\._get_colab_exe\b|\bb\._get_rclone_exe\b"
        r"|\bb\._get_rclone_conf\b|\bb\._resolve_cf_config_path\b|\bb\._build_ctl_env\b)"
    )

    for fname in ("backend_agent.py", "backend_accounts.py"):
        fpath = PET_DIR / fname
        src = fpath.read_text(encoding="utf-8")
        assert not legacy_re.search(src), f"{fname} 仍残留 _cb_mod 或直接摸底内部属性调用"
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".")[0] not in forbidden_imports
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".")[0] not in forbidden_imports
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                cc = _calc_ast_cc(node)
                assert cc <= 30, f"{fname}:{node.name} 圈复杂度过高: CC={cc}"

    assert bacc_mod.AccountVaultService is not None
    assert csec.VaultStore is not None
    backend = _make_backend(isolated_env, "bind_check")
    for svc in (backend._agent_svc, backend._accounts_svc):
        assert svc._b is backend
        assert svc._store is backend._store
        assert svc._secrets is backend._secrets
        assert svc._bus is backend._bus
        assert svc._ctx is backend._ctx


# =========================================================================
# 2. _patch_comfy_api_graph 多拓扑与参数边界校验
# =========================================================================


def test_patch_comfy_api_graph_guider_fallback_and_clamping() -> None:
    """验证 Flux guider/noise/sigmas 拓扑、标题兜底与 steps/batch_size 边界钳制，且不篡改原图。"""
    flux_graph = {
        "1": {"class_type": "CLIPTextEncode", "inputs": {"text": "old pos"}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "old neg"}},
        "3": {"class_type": "CFGGuider", "inputs": {"positive": ["1", 0], "negative": ["2", 0]}},
        "4": {"class_type": "RandomNoise", "inputs": {"noise_seed": 1}},
        "5": {"class_type": "BasicScheduler", "inputs": {"steps": 12}},
        "6": {"class_type": "EmptySD3LatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
        "7": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {"guider": ["3", 0], "noise": ["4", 0], "sigmas": ["5", 0], "latent_image": ["6", 0]},
        },
    }
    patched, applied = ba_mod._patch_comfy_api_graph(
        flux_graph,
        positive_prompt="cinematic sunset",
        negative_prompt="blurry artifacts",
        seed=987654,
        steps=999,
        batch_size=99,
    )
    assert flux_graph["1"]["inputs"]["text"] == "old pos"
    assert patched["1"]["inputs"]["text"] == "cinematic sunset"
    assert patched["2"]["inputs"]["text"] == "blurry artifacts"
    assert patched["4"]["inputs"]["noise_seed"] == 987654
    assert patched["5"]["inputs"]["steps"] == 150
    assert patched["6"]["inputs"]["batch_size"] == 8
    assert applied == {
        "positive_prompt": "cinematic sunset",
        "negative_prompt": "blurry artifacts",
        "seed": 987654,
        "steps": 150,
        "batch_size": 8,
    }

    # 无采样器连线时按标题兜底区分正向/反向编码器
    fallback_graph = {
        "10": {"class_type": "CLIPTextEncode", "_meta": {"title": "反向提示词"}, "inputs": {"text": "old_n"}},
        "11": {"class_type": "CLIPTextEncode", "_meta": {"title": "正面描述"}, "inputs": {"text": "old_p"}},
    }
    patched_fb, applied_fb = ba_mod._patch_comfy_api_graph(
        fallback_graph,
        positive_prompt="new_pos",
        negative_prompt="new_neg",
    )
    assert patched_fb["11"]["inputs"]["text"] == "new_pos"
    assert patched_fb["10"]["inputs"]["text"] == "new_neg"
    assert applied_fb == {"positive_prompt": "new_pos", "negative_prompt": "new_neg"}


# =========================================================================
# 3. AgentService 路由、配置、多会话管理与 4 级工作流模板加载
# =========================================================================


def test_agent_config_routing_sessions_and_discover_llm(isolated_env: SimpleNamespace) -> None:
    """覆盖 save_agent_config 四分支、Colab LLM 探测、多会话操作及并发无死锁。"""
    backend = _make_backend(isolated_env, "agent_core")
    svc = backend._agent_svc
    bus_events: list[dict[str, Any]] = []
    unsub = backend._bus.subscribe(lambda e: bus_events.append(dict(e)))

    try:
        # 1) 新增两个供应商并测试一键切换与删除回退
        s1 = svc.save_agent_config(
            {
                "upsert_provider": {
                    "name": "网关一",
                    "api_format": "openai",
                    "base_url": "https://p1.example/v1",
                    "models": ["gpt-4o-mini", "o3-mini"],
                    "default_model": "gpt-4o-mini",
                    "api_key": "sk-p1-secret-11112222",
                    "activate": True,
                }
            }
        )
        assert s1["ok"] is True
        pid1 = s1["config"]["agent_active_provider_id"]
        assert any(e.get("phase") == "config_updated" for e in bus_events)

        s2 = svc.save_agent_config(
            {
                "upsert_provider": {
                    "name": "本地回环网关",
                    "api_format": "anthropic",
                    "base_url": "http://127.0.0.1:3425",
                    "models": ["claude-sonnet-4-20250514"],
                    "activate": False,
                }
            }
        )
        assert s2["ok"] is True
        pid2 = next(p["id"] for p in s2["config"]["providers"] if p["id"] != pid1)

        sel = svc.save_agent_config({"select_provider_model": {"provider_id": pid2, "model": "claude-opus-4-20250514"}})
        assert sel["ok"] is True
        assert sel["config"]["agent_active_provider_id"] == pid2
        assert sel["config"]["agent_model"] == "claude-opus-4-20250514"
        assert sel["brain"]["external_is_loopback"] is True

        # 单独更新思考强度，返回专属提示语
        eff_res = svc.save_agent_config({"reasoning_effort": "high"})
        assert eff_res["ok"] is True
        assert "思考强度已设为" in eff_res["message"]

        # 删除当前活跃供应商，自动回退到剩余供应商 pid1
        del_res = svc.save_agent_config({"delete_provider_id": pid2})
        assert del_res["ok"] is True
        assert del_res["config"]["agent_active_provider_id"] == pid1

        # 2) 通过 _secrets 与 _store 探测在线 Colab LLM 端点与密钥
        backend._secrets.set_ep_meta(
            "m-llm-ep",
            {"kind": "llm", "llm_tunnel_url": "https://llm-demo.trycloudflare.com", "model_name": "Qwen3.8-32B"},
            runtime_secret="sk-colab-runtime-889900",
        )
        backend._store.set_tunnel_cache("m-llm-ep", "https://llm-demo.trycloudflare.com")
        with backend._comfy_lock:
            backend._comfy_state["llm_online"] = True
            backend._comfy_state["service_kind"] = "llm"
            backend._comfy_state["llm_tunnel_url"] = "https://llm-demo.trycloudflare.com"
            backend._comfy_state["model_name"] = "Qwen3.8-32B"

        discovered = svc._discover_colab_llm(probe_network=False)
        assert discovered["online"] is True
        assert discovered["api_base_url"] == "https://llm-demo.trycloudflare.com/v1"
        assert discovered["api_key"] == "sk-colab-runtime-889900"
        assert discovered["model"] == "Qwen3.8-32B"

        # 切到 colab_llm 路由应解析为 Colab Qwen3.8
        svc.save_agent_config({"brain_source": "colab_llm"})
        pub_b, rt_b = svc._resolve_brain_runtime(probe_network=False)
        assert pub_b["resolved_source"] == "colab_llm"
        assert pub_b["ready"] is True
        assert rt_b["api_key"] == "sk-colab-runtime-889900"

        # 3) 多会话管理动作全覆盖与并发无死锁检验
        st0 = svc.manage_agent_session({"action": "list"})
        sid0 = st0["active_session_id"]
        same_new = svc.manage_agent_session({"action": "new"})
        assert same_new["ok"] is True and same_new["created"] is False and same_new["active_session_id"] == sid0

        svc._append_message({"id": "m1", "role": "user", "content": "第一条问题", "ts": int(time.time())})
        created = svc.manage_agent_session({"action": "new", "title": "第二会话"})
        assert created["ok"] is True and created["created"] is True
        sid1 = created["active_session_id"]
        assert sid1 != sid0

        ren = svc.manage_agent_session({"action": "rename", "session_id": sid1, "title": "重命名会话"})
        assert ren["ok"] is True and "重命名会话" in ren["message"]

        # 切回 sid0 时，空的 sid1 应被自动回收
        sw = svc.manage_agent_session({"action": "switch", "session_id": sid0})
        assert sw["ok"] is True and sw["active_session_id"] == sid0
        assert all(s["id"] != sid1 for s in sw["state"]["sessions"])

        # 并发交错调用 get_agent_state 与 manage_agent_session 验证无锁反转死锁
        errs: list[Exception] = []

        def _reader() -> None:
            try:
                for _ in range(15):
                    svc.get_agent_state()
            except Exception as exc:
                errs.append(exc)

        def _writer() -> None:
            try:
                for i in range(15):
                    svc.manage_agent_session({"action": "rename", "title": f"并发标题_{i}"})
            except Exception as exc:
                errs.append(exc)

        t1 = threading.Thread(target=_reader, daemon=True)
        t2 = threading.Thread(target=_writer, daemon=True)
        t1.start()
        t2.start()
        t1.join(timeout=5.0)
        t2.join(timeout=5.0)
        assert not t1.is_alive() and not t2.is_alive() and not errs

        clr = svc.manage_agent_session({"action": "clear_all"})
        assert clr["ok"] is True and len(clr["state"]["sessions"]) == 1
        bad = svc.manage_agent_session({"action": "unknown_op"})
        assert bad["ok"] is False
    finally:
        unsub()


def test_agent_load_prompt_graph_template_four_tiers(
    isolated_env: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 _load_prompt_graph_template 的指定文件 -> 画廊 PNG -> 云端历史 -> 默认检查点 4 级兜底。"""
    backend = _make_backend(isolated_env, "prompt_tiers")
    svc = backend._agent_svc
    sample_prompt = {
        "1": {"class_type": "CLIPTextEncode", "inputs": {"text": "hello"}},
        "2": {"class_type": "KSampler", "inputs": {"positive": ["1", 0], "seed": 1, "steps": 20}},
    }

    # Tier 1: 指定 JSON 文件
    json_p = isolated_env.downloads_dir / "wf_api.json"
    json_p.write_text(json.dumps(sample_prompt), encoding="utf-8")
    g1, src1, err1 = svc._load_prompt_graph_template("wf_api.json", "https://c.trycloudflare.com", "ep1")
    assert err1 == "" and g1 is not None and src1 == "wf_api.json"

    # Tier 2: 本地画廊自动挑选含 API prompt 的 PNG
    png_p = isolated_env.downloads_dir / "gallery_sample.png"
    _make_png_with_prompt(png_p, sample_prompt)
    g2, src2, err2 = svc._load_prompt_graph_template("", "https://c.trycloudflare.com", "ep1")
    assert err2 == "" and g2 is not None and src2.endswith("gallery_sample.png")

    # 删除本地 PNG 后进入 Tier 3: 云端 /history
    png_p.unlink()

    class _Resp:
        def __init__(self, status: int, data: Any) -> None:
            self.status_code = status
            self._data = data

        def json(self) -> Any:
            return self._data

    monkeypatch.setattr(
        backend,
        "_comfy_http_get",
        lambda url, **_kw: _Resp(200, {"pid_9": {"prompt": [0, "pid_9", sample_prompt]}})
        if "/history" in url
        else _Resp(404, {}),
    )
    g3, src3, err3 = svc._load_prompt_graph_template("", "https://c.trycloudflare.com", "ep1")
    assert err3 == "" and g3 is not None and src3 == "remote_history:pid_9"

    # Tier 4: 云端历史为空时走 CheckpointLoaderSimple 兜底模板
    monkeypatch.setattr(
        backend,
        "_comfy_http_get",
        lambda url, **_kw: _Resp(200, {})
        if "/history" in url
        else _Resp(200, {"CheckpointLoaderSimple": {"input": {"required": {"ckpt_name": [["anima_v1.safetensors"]]}}}}),
    )
    g4, src4, err4 = svc._load_prompt_graph_template("", "https://c.trycloudflare.com", "ep1")
    assert err4 == "" and g4 is not None and src4 == "cloud_template:anima_v1.safetensors"


# =========================================================================
# 4. send_agent_message 流式编排、双引擎参数传递与远端执行审批护栏
# =========================================================================


def test_send_agent_message_stream_events_and_remote_exec_guard(
    isolated_env: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证拆解后的 send_agent_message 完整流式事件广播、密钥脱敏、ClaudeCodeDriver 参数与安全审批。"""
    backend = _make_backend(isolated_env, "agent_stream")
    svc = backend._agent_svc
    secret_key = "sk-stream-guard-9988776655"
    svc.save_agent_config(
        {
            "upsert_provider": {
                "name": "测试引擎源",
                "api_format": "openai",
                "base_url": "https://mock-llm.example/v1",
                "models": ["gpt-4o-mini"],
                "api_key": secret_key,
                "activate": True,
            }
        }
    )

    events: list[dict[str, Any]] = []
    unsub = backend._bus.subscribe(lambda e: events.append(dict(e)))

    class _FakeNativeDriver:
        def __init__(self, **kwargs: Any) -> None:
            self._on_event = kwargs["on_event"]
            self._history = kwargs["messages"]

        def run(self) -> dict[str, Any]:
            self._on_event({"type": "reasoning_delta", "text": f"思考中 key={secret_key}"})
            self._on_event({"type": "tool_start", "name": "list_gdrive_assets", "args": {}, "tier": "safe"})
            self._on_event({"type": "tool_end", "name": "list_gdrive_assets", "ok": True, "summary": "ok", "elapsed_ms": 12})
            self._on_event({"type": "delta", "text": f"完成答复 secret={secret_key}"})
            return {"ok": True, "text": f"完成答复 secret={secret_key}", "reasoning": f"思考中 key={secret_key}"}

    try:
        with patch.object(ba_mod, "NativeLightDriver", _FakeNativeDriver):
            res = svc.send_agent_message({"text": "帮我看下云盘状态", "context": {"view": "studio/panel"}})
            assert res["ok"] is True and res["started"] is True
            done_evt = _wait_until(
                lambda: next((e for e in events if e.get("phase") == "turn_done"), None),
                err_msg="未收到 turn_done 广播事件",
            )
            assert done_evt["ok"] is True
            assert secret_key not in json.dumps(events, ensure_ascii=False)
            assert "[REDACTED]" in done_evt["text"]
            assert any(e.get("phase") == "reasoning_delta" for e in events)
            assert any(e.get("phase") == "tool_start" for e in events)
            assert any(e.get("phase") == "tool_end" for e in events)

        # 确认 _busy_lock 与 _running 已被 finally 彻底释放
        _wait_until(lambda: not svc.get_agent_state()["busy"], err_msg="轮次结束后 busy 未释放")

        # 验证 ClaudeCodeDriver 通过 self._ctx 获取 ctl_script 与 ctl_cmd_prefix
        captured_cc: dict[str, Any] = {}

        class _FakeClaudeDriver:
            @staticmethod
            def is_available() -> bool:
                return True

            def __init__(self, **kwargs: Any) -> None:
                captured_cc.update(kwargs)

            def run(self, history: list[dict[str, Any]]) -> dict[str, Any]:
                captured_cc["history_len"] = len(history)
                return {"ok": True, "text": "claude ok"}

        monkeypatch.setattr(cb, "_ctl_cmd_prefix", lambda unbuffered=False: ["fake-py", "fake-ctl.py"])
        events.clear()
        svc.save_agent_config({"driver": "claude_code"})
        with patch.object(ba_mod, "ClaudeCodeDriver", _FakeClaudeDriver):
            cc_res = svc.send_agent_message({"text": "检查 Claude 驱动参数"})
            assert cc_res["ok"] is True
            _wait_until(
                lambda: next((e for e in events if e.get("phase") == "turn_done"), None),
                err_msg="ClaudeCodeDriver 未收到 turn_done",
            )
        assert captured_cc["ctl_prefix"] == ["fake-py", "fake-ctl.py"]
        assert captured_cc["ctl_script"] == backend._ctx.ctl_script

        # 验证 _tool_exec_remote_colab 保留模块级 subprocess.Popen 补丁兼容 + 安全三层门禁
        spawned_cmds: list[list[str]] = []

        class _FakePopen:
            def __init__(self, cmd: list[str], **_kw: Any) -> None:
                self._cmd = cmd
                self.returncode = 0
                self.pid = 4242

            def communicate(self, input: str = "", timeout: int = 180) -> tuple[str, str]:
                spawned_cmds.append(list(self._cmd))
                return f"OUT:{input}\n", ""

        monkeypatch.setattr(cb, "_ctl_available", lambda: True)
        with patch.object(ba_mod.subprocess, "Popen", side_effect=_FakePopen):
            blk = svc._execute_tool("exec_remote_colab", {"code": "cat /root/colab_service_meta.json", "mode": "bash"}, CancelToken(), ())
            assert blk["ok"] is False and blk.get("blocked") is True
            assert spawned_cmds == []

            ro = svc._execute_tool("exec_remote_colab", {"code": "nvidia-smi", "mode": "bash"}, CancelToken(), ())
            assert ro["ok"] is True and ro["output"] == "OUT:nvidia-smi"
            assert len(spawned_cmds) == 1
    finally:
        unsub()


# =========================================================================
# 5. AccountVaultService 容器化 OAuth / 切号 / 解绑 / 云盘 / 诊断全链路
# =========================================================================


def test_account_vault_oauth_switch_remove_quota_and_diagnostics(
    isolated_env: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 AccountVaultService 迁移至 _store/_secrets/_bus/_ctx 后的全量核心行为。"""
    backend = _make_backend(isolated_env, "acct_service")
    svc = backend._accounts_svc

    fake_colab_py = isolated_env.tmp_path / "fake_python.exe"
    fake_colab_py.write_text("stub", encoding="utf-8")
    fake_rclone_exe = isolated_env.tmp_path / "fake_rclone.exe"
    fake_rclone_exe.write_text("stub", encoding="utf-8")
    fake_rclone_conf = isolated_env.tmp_path / "rclone.conf"
    fake_rclone_conf.write_text("[gdrive_save]\ntype = drive\ntoken = {\"access_token\":\"x\"}\n", encoding="utf-8")

    monkeypatch.setattr(backend, "_get_colab_py", lambda: fake_colab_py)
    monkeypatch.setattr(backend, "_get_rclone_exe", lambda: fake_rclone_exe)
    monkeypatch.setattr(backend, "_get_rclone_conf", lambda: fake_rclone_conf)
    monkeypatch.setattr(backend, "fetch_live_status", lambda: {"balance": 88.5, "onboarding": {"needs_attention": False}})

    opened_urls: list[str] = []
    bridge_calls: list[tuple[str, str]] = []
    monkeypatch.setattr(cb.webbrowser, "open", lambda u, *_a, **_kw: opened_urls.append(str(u)))

    def _fake_run_hidden(cmd: list[str], timeout: int = 20, input_text: str | None = None, **_kw: Any) -> SimpleNamespace:
        _ = timeout
        cmd_strs = [str(x) for x in cmd]
        if "oauth-start" in cmd_strs:
            payload = {"auth_url": "https://accounts.google.com/o/oauth2/v2/auth?x=1", "code_verifier": "pkce-secret-verifier-999"}
            return SimpleNamespace(returncode=0, stdout=f"<<<COLAB_OAUTH_START>>>{json.dumps(payload)}<<<END_COLAB_OAUTH_START>>>", stderr="")
        if "oauth-complete" in cmd_strs:
            bridge_calls.append(("oauth-complete", str(input_text or "")))
            req = json.loads(input_text or "{}")
            acct = "acct_alpha" if "alpha" in req.get("code", "") else "acct_beta"
            done = {"acct_id": acct, "email_masked": f"{acct}@***.com"}
            return SimpleNamespace(returncode=0, stdout=f"<<<COLAB_OAUTH_DONE>>>{json.dumps(done)}<<<END_COLAB_DONE>>>".replace("END_COLAB_DONE", "END_COLAB_OAUTH_DONE"), stderr="")
        if "account-op" in cmd_strs:
            bridge_calls.append(("account-op", str(input_text or "")))
            return SimpleNamespace(returncode=0, stdout="ok", stderr="")
        if "about" in cmd_strs:
            return SimpleNamespace(returncode=0, stdout=json.dumps({"free": 50 * (1024**3), "total": 100 * (1024**3)}), stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(cb, "_run_hidden", _fake_run_hidden)

    # 1) OAuth 启动 + 完成绑定第一个账号
    st_res = svc.start_colab_oauth()
    assert st_res["ok"] is True and st_res["awaiting_code"] is True
    assert opened_urls == ["https://accounts.google.com/o/oauth2/v2/auth?x=1"]

    cp1 = svc.complete_colab_oauth("code_alpha_123456", label="主号")
    assert cp1["ok"] is True
    assert backend._store.get_active_account_id() == "acct_alpha"

    # 记录主号账本余额，再绑定第二个账号验证账本隔离
    with backend._store.lock:
        backend._store.state["last_balance"] = 123.4
    assert svc.start_colab_oauth()["ok"] is True
    cp2 = svc.complete_colab_oauth("code_beta_654321", label="副号")
    assert cp2["ok"] is True
    assert backend._store.get_active_account_id() == "acct_beta"
    assert "acct_alpha" in (backend._store.state.get("account_ledgers") or {})

    # 2) 重命名与忙碌拦截
    rn = svc.rename_colab_account("acct_beta", "备用Ultra")
    assert rn["ok"] is True and "备用Ultra" in rn["message"]

    with backend._deploy_lock:
        busy_sw = svc.switch_colab_account("acct_alpha")
        assert busy_sw["ok"] is False and busy_sw["busy"] is True

    # 3) 秒切回主号并验证余额恢复
    sw_ok = svc.switch_colab_account("acct_alpha")
    assert sw_ok["ok"] is True
    assert backend._store.get_active_account_id() == "acct_alpha"
    assert backend._store.state.get("last_balance") == 123.4

    # 4) 移除当前激活账号自动切到余号，再移除最后一个账号清空激活态
    rm1 = svc.remove_colab_account("acct_alpha")
    assert rm1["ok"] is True
    assert backend._store.get_active_account_id() == "acct_beta"
    assert "自动切至" in rm1["message"]

    rm2 = svc.remove_colab_account("acct_beta")
    assert rm2["ok"] is True
    assert backend._store.get_active_account_id() == ""
    assert backend._auth_missing is True

    # 5) rclone 配额查询写回 _store，启动/取消云盘绑定经 _bus 广播 onboarding_event
    free_gb, total_gb = svc.query_gdrive_quota_locked()
    assert free_gb == pytest.approx(50.0, rel=0.01)
    assert total_gb == pytest.approx(100.0, rel=0.01)
    assert backend._store.state["onboarding_cache"]["gdrive_free_gb"] == pytest.approx(50.0, rel=0.01)

    bus_events: list[dict[str, Any]] = []
    unsub = backend._bus.subscribe(lambda e: bus_events.append(dict(e)))

    class _StubBindProc:
        def __init__(self, *_a: Any, **_kw: Any) -> None:
            self.returncode = 0
            self.pid = 55555

        def communicate(self, timeout: int = 180) -> tuple[str, str]:
            _ = timeout
            return "ok", ""

    try:
        with patch.object(cb.subprocess, "Popen", side_effect=_StubBindProc):
            bind_res = svc.start_gdrive_bind()
            assert bind_res["ok"] is True and bind_res["binding"] is True
            ob_evt = _wait_until(
                lambda: next((e for e in bus_events if e.get("type") == "onboarding_event"), None),
                err_msg="start_gdrive_bind 未通过 EventBus 广播 onboarding_event",
            )
            assert ob_evt["ok"] is True
    finally:
        unsub()

    # 6) 工具路径设置/重置与脱敏诊断包导出
    set_p = svc.set_tool_path("colab_py", str(fake_colab_py))
    assert set_p["ok"] is True
    assert backend._store.get_prefs()["colab_py_path"] == str(fake_colab_py)

    rst_p = svc.reset_tool_paths()
    assert rst_p["ok"] is True
    assert backend._store.get_prefs()["colab_py_path"] == ""

    diag_dir = isolated_env.tmp_path / "diag_out"
    diag_res = svc.export_diagnostics(out_dir=diag_dir)
    assert diag_res["ok"] is True
    assert Path(diag_res["path"]).is_file()
