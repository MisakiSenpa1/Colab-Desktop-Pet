"""Phase 1 Wave A Step A1 专项离线回归测试：节流诊断日志基础设施 + 3 个超长核心函数拆解。

覆盖范围：
1. `DiagLogger.debug_throttled` 窗口抑制、到期放行、键上限淘汰防泄漏与多线程并发安全；
2. `suppress_logged` 上下文管理器的敏感凭据脱敏写盘、节流去重、异常类型过滤与 `runtime` 门面重导出；
3. `health.py::run_preflight_checks` 拆解管线与 `verify_manifest_and_assets` 可选 `gpu_specs` 正数费率校验；
4. `file_sync.py::api_prompt_to_ui_workflow` 有/无 `object_info` 转换、拓扑排版、连线推断与损坏 PNG 日志记录；
5. `colab_client.py::stop_session_cli` 显式 Endpoint 防盲停、未锁定残留实例 Fail-Closed 与释放核实门禁；
6. AST 静态度量门禁：3 个核心函数及其拆出辅助均满足 `<= 80 行`、`CC <= 25`，4 文件宽泛 `except: pass == 0`。
"""

from __future__ import annotations

import ast
import json
import struct
import sys
import threading
import zlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

PKG_DIR = Path(__file__).resolve().parents[1]
if str(PKG_DIR) not in sys.path:
    sys.path.insert(0, str(PKG_DIR))

from colab_core import colab_client, exec_runner, file_sync, health, runtime  # noqa: E402
from colab_core.accounts import ToolPaths  # noqa: E402
import colab_security as csec  # noqa: E402


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


def _make_png_with_text(path: Path, key: str, value: str) -> None:
    """用纯标准库生成包含指定 `tEXt` 元数据块的最小合法 PNG 文件。"""
    def _chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    idat = zlib.compress(b"\x00\x00\x00\x00")
    txt_data = key.encode("latin-1") + b"\x00" + value.encode("utf-8")
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"tEXt", txt_data)
        + _chunk(b"IDAT", idat)
        + _chunk(b"IEND", b"")
    )


# =========================================================================
# 1. DiagLogger.debug_throttled 窗口抑制、到期放行、淘汰防泄漏与线程安全
# =========================================================================


def test_diag_logger_debug_throttled_window_eviction_and_thread_safety(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 debug_throttled 在时间窗口内去重、到期放行、缓存上限淘汰与多线程并发精确去重。"""
    log_file = tmp_path / "throttled.log"
    fake_now = [1000.0]
    monkeypatch.setattr(exec_runner.time, "monotonic", lambda: fake_now[0])

    diag = exec_runner.DiagLogger(log_path=log_file, max_throttle_keys=16)

    # 窗口内同 (tag, key) 仅首条落盘
    assert diag.debug_throttled("poll", "net_err", "首次网络抖动", interval_sec=30.0) is True
    assert diag.debug_throttled("poll", "net_err", "重复网络抖动", interval_sec=30.0) is False
    # 不同 key 或不同 tag 各自独立放行
    assert diag.debug_throttled("poll", "other_key", "另一个键", interval_sec=30.0) is True
    assert diag.debug_throttled("sync", "net_err", "另一个标签", interval_sec=30.0) is True

    lines = diag.read_tail_lines(50)
    assert len(lines) == 3
    assert "首次网络抖动" in lines[0] and "[DEBUG]" in lines[0]

    # 推进时间越过窗口后再次放行
    fake_now[0] += 30.1
    assert diag.debug_throttled("poll", "net_err", "窗口过期后再次落盘", interval_sec=30.0) is True
    lines2 = diag.read_tail_lines(50)
    assert len(lines2) == 4 and "窗口过期后再次落盘" in lines2[-1]

    # 大量不同键写入必须触发缓存淘汰，防止长期运行内存泄漏
    for idx in range(64):
        fake_now[0] += 0.01
        assert diag.debug_throttled("flood", f"k_{idx}", f"msg_{idx}", interval_sec=60.0) is True
    assert len(diag._throttle_ts) <= 16

    # 高并发多线程竞争同一 (tag, key)：严格仅 1 个线程落盘
    conc_log = tmp_path / "concurrent.log"
    conc_diag = exec_runner.DiagLogger(log_path=conc_log)
    n_threads = 16
    barrier = threading.Barrier(n_threads)
    results: list[bool] = []
    res_lock = threading.Lock()

    def _worker(idx: int) -> None:
        barrier.wait()
        emitted = conc_diag.debug_throttled("race", "same_key", f"worker_{idx}", interval_sec=60.0)
        with res_lock:
            results.append(emitted)

    threads = [threading.Thread(target=_worker, args=(i,), daemon=True) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5.0)
    assert results.count(True) == 1
    assert results.count(False) == n_threads - 1
    assert len(conc_diag.read_tail_lines(50)) == 1


# =========================================================================
# 2. suppress_logged 脱敏写盘、节流抑制、异常类型过滤与门面导出
# =========================================================================


def test_suppress_logged_redaction_throttling_and_reexport(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 suppress_logged 正确脱敏落盘、按 throttled_key 节流、过滤异常类型并由 runtime 导出。"""
    assert runtime.suppress_logged is exec_runner.suppress_logged
    assert "suppress_logged" in runtime.__all__

    diag_file = tmp_path / "suppress_diag.log"
    monkeypatch.setenv("COLAB_PET_DIAG_LOG_PATH", str(diag_file))
    monkeypatch.setattr(exec_runner, "_GLOBAL_DIAG_LOGGER", exec_runner.DiagLogger(log_path=diag_file))

    secret_val = "sk-phase1-suppress-secret-99887766"
    csec.register_ephemeral_secret(secret_val)

    # 默认捕获 Exception 并脱敏落盘
    with runtime.suppress_logged("unit_scope"):
        raise RuntimeError(f"连接失败 key={secret_val}")

    txt = diag_file.read_text(encoding="utf-8")
    assert "[DEBUG]" in txt and "[unit_scope]" in txt
    assert secret_val not in txt
    assert "[REDACTED]" in txt

    # 指定 throttled_key 连续触发 5 次仅落盘 1 条
    before_cnt = len(diag_file.read_text(encoding="utf-8").splitlines())
    for i in range(5):
        with runtime.suppress_logged("unit_scope", ValueError, throttled_key="dup_val_err", interval_sec=60.0):
            raise ValueError(f"bad value #{i}")
    after_cnt = len(diag_file.read_text(encoding="utf-8").splitlines())
    assert after_cnt == before_cnt + 1

    # 未列入 *exc_types 的异常必须原样透传，绝不吞掉
    with pytest.raises(KeyError, match="must_escape"):
        with runtime.suppress_logged("unit_scope", ValueError):
            raise KeyError("must_escape")


# =========================================================================
# 3. health.py::run_preflight_checks 与 verify_manifest_and_assets 校验
# =========================================================================


def test_health_preflight_and_manifest_gpu_specs_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证拆解后的 run_preflight_checks 各分支与 verify_manifest_and_assets 可选 gpu_specs 费率校验。"""
    monkeypatch.setenv("COLAB_PET_MOCK_VAULT_PATH", str(tmp_path / "vault.json"))
    colab_exe = tmp_path / "colab.exe"
    colab_py = tmp_path / "python.exe"
    rclone_exe = tmp_path / "rclone.exe"
    rclone_conf = tmp_path / "rclone.conf"
    for p in (colab_exe, colab_py):
        p.write_text("stub", encoding="utf-8")

    # 3a. 公开基础版缺 rclone 与 gdrive_save 降级为 warn 不阻断
    tp_no_rc = ToolPaths(colab_exe=colab_exe, colab_py=colab_py, rclone_exe=rclone_exe, rclone_conf=rclone_conf)
    rep_pub = health.run_preflight_checks(
        mode="fast",
        profiles=["anima"],
        gpu="L4",
        tool_paths=tp_no_rc,
        download_dir=tmp_path,
        cf_config_path=tmp_path / "no_cf.json",
        pkg_dir=PKG_DIR,
        edition="public",
        disk_usage_fn=lambda _p: SimpleNamespace(free=20 * (1024**3)),
    )
    assert rep_pub.ok is True
    assert any(it.name == "rclone" and it.status == "warn" for it in rep_pub.items)
    assert any(it.name == "Google 云盘配置" and it.status == "warn" for it in rep_pub.items)

    # 3b. 个人增强版缺 rclone 与低磁盘均阻断 fail
    rep_pers = health.run_preflight_checks(
        mode="fast",
        profiles=["anima", "qwen_27b"],
        gpu="L4",
        tool_paths=tp_no_rc,
        download_dir=tmp_path,
        cf_config_path=tmp_path / "no_cf.json",
        pkg_dir=PKG_DIR,
        edition="personal",
        disk_usage_fn=lambda _p: SimpleNamespace(free=100 * (1024**2)),
    )
    assert rep_pers.ok is False
    fail_names = {it.name for it in rep_pers.blocking_items()}
    assert {"rclone", "Google 云盘配置", "下载目录磁盘", "工作流与显卡匹配"}.issubset(fail_names)

    # 3c. verify_manifest_and_assets 对可选 gpu_specs 正数费率与 gpu_order 一致性校验
    fake_pkg = tmp_path / "fake_pkg"
    fake_pkg.mkdir()
    base_manifest = health.load_stack_manifest(PKG_DIR / "stack_manifest.json")
    for fn in (
        "requirements.txt",
        "requirements-lock.txt",
        "colab_pet_bridge_node.py",
        "colab_remote_common.py",
        "colab_cli_bridge.py",
        "colab_sse_gateway.py",
        "colab_pet_direct_canvas.js",
        "colab_security.py",
    ):
        (fake_pkg / fn).write_text("ok", encoding="utf-8")
    for spec in (base_manifest.get("profiles") or {}).values():
        for k in ("bootstrap_script", "smoke_script"):
            (fake_pkg / spec[k]).write_text("ok", encoding="utf-8")

    # 合法正数 gpu_specs 通过
    m_ok = dict(base_manifest)
    m_ok["gpu_order"] = ["T4", "L4"]
    m_ok["gpu_specs"] = {
        "T4": {"default_rate_std": 1.15, "default_rate_hm": 1.31},
        "L4": {"default_rate_std": 1.94, "default_rate_hm": 1.94},
    }
    (fake_pkg / "stack_manifest.json").write_text(json.dumps(m_ok), encoding="utf-8")
    assert health.verify_manifest_and_assets(fake_pkg)["ok"] is True

    # 非法费率（0 / 负数 / 布尔）及未知 gpu_order 项必须报错
    m_bad = dict(base_manifest)
    m_bad["gpu_order"] = ["T4", "UNKNOWN_GPU"]
    m_bad["gpu_specs"] = {
        "T4": {"default_rate_std": 0, "default_rate_hm": -1.2},
        "L4": {"default_rate_std": True, "default_rate_hm": 1.94},
    }
    (fake_pkg / "stack_manifest.json").write_text(json.dumps(m_bad), encoding="utf-8")
    bad_res = health.verify_manifest_and_assets(fake_pkg)
    assert bad_res["ok"] is False
    joined_err = "\n".join(bad_res["errors"])
    assert "T4" in joined_err and "L4" in joined_err and "UNKNOWN_GPU" in joined_err


# =========================================================================
# 4. file_sync.py::api_prompt_to_ui_workflow 转换与 PNG 损坏日志记录
# =========================================================================


def test_api_prompt_to_ui_workflow_with_and_without_object_info(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 api_prompt_to_ui_workflow 在空输入、带 object_info 与无 object_info 下的完整等价转换。"""
    # 空输入返回标准空骨架
    empty_wf = file_sync.api_prompt_to_ui_workflow({"bad": "not_a_node"})
    assert empty_wf["nodes"] == [] and empty_wf["links"] == [] and empty_wf["last_node_id"] == 0

    prompt = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "anima.safetensors"}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "1girl, masterpiece", "clip": ["1", 1]}},
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0],
                "positive": ["2", 0],
                "seed": 42,
                "steps": 24,
                "extra_link": ["1", 2],
            },
        },
        "4": {"class_type": "SaveImage", "inputs": {"images": ["3", 0], "filename_prefix": "out"}},
    }
    object_info = {
        "CheckpointLoaderSimple": {
            "input": {"required": {"ckpt_name": [["anima.safetensors"]]}},
            "output": ["MODEL", "CLIP", "VAE"],
            "output_name": ["MODEL", "CLIP", "VAE"],
        },
        "CLIPTextEncode": {
            "input": {"required": {"text": ["STRING", {"default": ""}], "clip": ["CLIP"]}},
            "output": ["CONDITIONING"],
        },
        "KSampler": {
            "input": {
                "required": {
                    "model": ["MODEL"],
                    "positive": ["CONDITIONING"],
                    "seed": ["INT", {"default": 0}],
                    "steps": ["INT", {"default": 20}],
                }
            },
            "output": ["LATENT"],
        },
        "SaveImage": {
            "input": {"required": {"images": ["IMAGE"], "filename_prefix": ["STRING", {"default": "ComfyUI"}]}},
            "output": [],
        },
    }

    wf = file_sync.api_prompt_to_ui_workflow(prompt, object_info=object_info)
    assert wf["last_node_id"] == 4
    assert wf["last_link_id"] == 5
    by_type = {n["type"]: n for n in wf["nodes"]}
    # 拓扑列递增：CheckpointLoaderSimple (depth 0) < CLIPTextEncode (1) < KSampler (2) < SaveImage (3)
    assert by_type["CheckpointLoaderSimple"]["pos"][0] < by_type["CLIPTextEncode"]["pos"][0]
    assert by_type["CLIPTextEncode"]["pos"][0] < by_type["KSampler"]["pos"][0]
    assert by_type["KSampler"]["pos"][0] < by_type["SaveImage"]["pos"][0]
    # KSampler seed 后自动追加 "fixed" 控制项，且额外未知连线 extra_link 补入 inputs
    assert by_type["KSampler"]["widgets_values"] == [42, "fixed", 24]
    assert any(inp["name"] == "extra_link" and inp["link"] is not None for inp in by_type["KSampler"]["inputs"])

    # 无 object_info 兜底转换也能正确推断输出槽与 seed 控制项
    wf_fb = file_sync.api_prompt_to_ui_workflow(prompt, object_info=None)
    assert wf_fb["last_node_id"] == 4 and wf_fb["last_link_id"] == 5
    fb_ks = next(n for n in wf_fb["nodes"] if n["type"] == "KSampler")
    assert 42 in fb_ks["widgets_values"] and "fixed" in fb_ks["widgets_values"]

    # 损坏的 PNG workflow/prompt JSON 应返回 None 并写入节流 DEBUG 日志
    diag_file = tmp_path / "fs_diag.log"
    monkeypatch.setattr(exec_runner, "_GLOBAL_DIAG_LOGGER", exec_runner.DiagLogger(log_path=diag_file))
    bad_png = tmp_path / "broken_meta.png"
    _make_png_with_text(bad_png, "workflow", "{broken-json")
    assert file_sync.extract_png_comfy_workflow(bad_png) is None
    assert "png" in diag_file.read_text(encoding="utf-8").lower()


# =========================================================================
# 5. colab_client.py::stop_session_cli 精确停卡与 Fail-Closed 边界
# =========================================================================


def test_stop_session_cli_binding_and_fail_closed_boundaries(tmp_path: Path) -> None:
    """验证 stop_session_cli 拆解后的显式 Endpoint 防盲停、未释放残留拦截与环境缺失拒停。"""
    colab_exe = tmp_path / "colab.exe"
    colab_py = tmp_path / "python.exe"
    colab_exe.write_text("stub", encoding="utf-8")
    colab_py.write_text("stub", encoding="utf-8")

    # 5a. 显式 target_endpoint 绝不执行 `colab stop -s <session>` 盲停，直接走精确释放 + 核实
    seen_cmds: list[list[str]] = []

    def _run_exact(cmd: list[Any], **_kw: Any) -> SimpleNamespace:
        cmd_s = [str(x) for x in cmd]
        seen_cmds.append(cmd_s)
        script = cmd_s[-1] if len(cmd_s) >= 3 and cmd_s[1] == "-c" else ""
        if "STOP_FALLBACK" in script:
            return SimpleNamespace(
                returncode=0,
                stdout='<<<STOP_FALLBACK>>>{"acted": true, "active_eps": ["m-ep-1"]}<<<END_STOP_FALLBACK>>>\n',
                stderr="",
            )
        if "STOP_VERIFIED" in script:
            return SimpleNamespace(
                returncode=0,
                stdout='<<<STOP_VERIFIED>>>{"remaining": []}<<<END_STOP_VERIFIED>>>\n',
                stderr="",
            )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    res = colab_client.stop_session_cli(
        session="comfy",
        clean_all=False,
        target_endpoint="m-ep-1",
        verify_release=True,
        colab_exe=colab_exe,
        colab_py=colab_py,
        run_fn=_run_exact,
    )
    assert res["ok"] is True and res["released_endpoint"] == "m-ep-1" and res["remaining_count"] == 0
    assert not any("stop" in c and "-s" in c for c in seen_cmds)

    # 5b. verify_release 检测到目标 Endpoint 仍在线时必须 Fail-Closed 抛异常
    def _run_still_alive(cmd: list[Any], **_kw: Any) -> SimpleNamespace:
        script = str(cmd[-1])
        if "STOP_FALLBACK" in script:
            return SimpleNamespace(
                returncode=0,
                stdout='<<<STOP_FALLBACK>>>{"acted": true, "active_eps": ["m-ep-stuck"]}<<<END_STOP_FALLBACK>>>\n',
                stderr="",
            )
        return SimpleNamespace(
            returncode=0,
            stdout='<<<STOP_VERIFIED>>>{"remaining": ["m-ep-stuck"]}<<<END_STOP_VERIFIED>>>\n',
            stderr="",
        )

    with pytest.raises(RuntimeError, match="仍未释放"):
        colab_client.stop_session_cli(
            session="comfy",
            clean_all=False,
            target_endpoint="m-ep-stuck",
            verify_release=True,
            colab_exe=colab_exe,
            colab_py=colab_py,
            run_fn=_run_still_alive,
        )

    # 5c. 缺少 colab_py 且需要核实释放时直接 FileNotFoundError 拒停
    with pytest.raises(FileNotFoundError, match="已拒绝停卡"):
        colab_client.stop_session_cli(
            session="comfy",
            clean_all=False,
            target_endpoint="m-ep-1",
            verify_release=True,
            colab_exe=colab_exe,
            colab_py=tmp_path / "missing_python.exe",
            run_fn=_run_exact,
        )


# =========================================================================
# 6. AST 静态结构门禁：行数 <= 80、CC <= 25、宽泛 except-pass == 0、零重依赖
# =========================================================================


def test_phase1_core_decomp_metrics_and_no_broad_except_pass() -> None:
    """校验本组 3 个核心函数与拆出辅助均 <= 80L 且 CC <= 25，且 4 模块零宽泛 except-pass 与零重依赖。"""
    forbidden_imports = {"PIL", "Pillow", "aiohttp", "win32cred", "win32api", "pywintypes"}
    watched_funcs = {
        "colab_core/health.py": [
            "run_preflight_checks",
            "_probe_cli_version",
            "_check_preflight_tools",
            "_check_preflight_deep_desktop",
            "_check_preflight_auth",
            "_check_preflight_disk",
            "_check_preflight_cf_tunnel",
            "_check_preflight_config",
            "_check_preflight_profile_gpu",
            "_check_preflight_mock_vault",
            "_verify_manifest_gpu_specs",
        ],
        "colab_core/file_sync.py": [
            "api_prompt_to_ui_workflow",
            "_collect_valid_prompt_nodes",
            "_compute_prompt_node_depths",
            "_default_ui_node_size",
            "_build_ui_node_slots_with_info",
            "_build_ui_node_slots_fallback",
            "_build_single_ui_node",
            "_wire_ui_workflow_links",
        ],
        "colab_core/colab_client.py": [
            "stop_session_cli",
            "_validate_and_resolve_stop_target",
            "_run_cli_session_stop",
            "_build_stop_fallback_script",
            "_run_cloud_stop_fallback",
            "_verify_stop_release_state",
            "_format_stop_failure_hint",
        ],
    }

    for rel in (
        "colab_core/exec_runner.py",
        "colab_core/runtime.py",
        "colab_core/health.py",
        "colab_core/file_sync.py",
        "colab_core/colab_client.py",
    ):
        tree = ast.parse((PKG_DIR / rel).read_text(encoding="utf-8"))
        func_map: dict[str, tuple[int, int]] = {}
        broad_pass_lines: list[int] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    assert a.name.split(".")[0] not in forbidden_imports
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".")[0] not in forbidden_imports
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                span = (node.end_lineno or node.lineno) - node.lineno + 1
                func_map[node.name] = (span, _calc_ast_cc(node))
            elif isinstance(node, ast.ExceptHandler):
                is_broad = (node.type is None) or (
                    isinstance(node.type, ast.Name) and node.type.id in ("Exception", "BaseException")
                )
                is_pass_only = len(node.body) == 1 and isinstance(node.body[0], ast.Pass)
                if is_broad and is_pass_only:
                    broad_pass_lines.append(node.lineno)

        assert broad_pass_lines == [], f"{rel} 仍存在宽泛 except: pass @ {broad_pass_lines}"
        for fname in watched_funcs.get(rel, []):
            assert fname in func_map, f"{rel} 缺少拆解函数: {fname}"
            span, cc = func_map[fname]
            assert span <= 80, f"{rel}:{fname} 行数超限: {span}L > 80L"
            assert cc <= 25, f"{rel}:{fname} 圈复杂度超限: CC={cc} > 25"
