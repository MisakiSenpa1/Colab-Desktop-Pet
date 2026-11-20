"""colab_core.agent.mcp_server —— 标准库 JSON-RPC 2.0 MCP Stdio 工具服务与默认 Fail-Closed 执行器。"""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from colab_core.agent.safety import (
    _redact_obj,
    _safe_json_dumps,
    check_remote_code_safety,
    is_readonly_safe_remote_cmd,
)
from colab_core.agent.specs import AGENT_TOOL_MAP, AGENT_TOOL_SPECS
from colab_core.runtime import CREATE_NO_WINDOW, DiagLogger, clean_subprocess_env, get_diag_logger
from colab_security import redact_secrets

_ORIG_SUBPROCESS = subprocess


def _resolve_subprocess() -> Any:
    """解析 subprocess 模块引用，确保对 `agent_tools.subprocess` 或本模块的 patch 均 100% 生效。"""
    if subprocess is not _ORIG_SUBPROCESS:
        return subprocess
    facade = sys.modules.get("colab_core.agent_tools")
    facade_sub = getattr(facade, "subprocess", None)
    if facade_sub is not None and facade_sub is not _ORIG_SUBPROCESS:
        return facade_sub
    return subprocess


def _is_mcp_mutations_allowed(allow_unattended_mutations: bool | None) -> bool:
    """判定无头 MCP 执行器是否被显式授权执行非只读变更或计费操作（默认严格 Fail-Closed）。"""
    if allow_unattended_mutations is not None:
        return bool(allow_unattended_mutations)
    raw = str(os.environ.get("COLAB_MCP_ALLOW_MUTATIONS") or "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def _approval_blocked(tool_name: str, detail: str) -> dict[str, Any]:
    """构造无头 MCP 审批拦截的结构化失败返回。"""
    return {
        "ok": False,
        "blocked": True,
        "approval_required": True,
        "tool": tool_name,
        "error": detail,
    }


def _run_ctl_subprocess(
    ctl_script: Path, sub_args: list[str], *, input_text: str | None = None, timeout: int = 180
) -> dict[str, Any]:
    """通过子进程调用 `colab_ctl.py` 并统一脱敏返回输出。"""
    sub_mod = _resolve_subprocess()
    cmd = [sys.executable, str(ctl_script), *sub_args]
    try:
        proc = sub_mod.run(
            cmd,
            input=input_text,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            env=clean_subprocess_env(),
            creationflags=CREATE_NO_WINDOW,
        )
        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": redact_secrets(proc.stdout or "")[-12000:],
            "stderr": redact_secrets(proc.stderr or "")[-4000:],
        }
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"命令执行超时 ({timeout}s)"}
    except Exception as exc:
        return {"ok": False, "error": redact_secrets(str(exc) or type(exc).__name__)}


def _mcp_run_doctor(ctl_script: Path, args: dict[str, Any]) -> dict[str, Any]:
    sub = ["doctor", "--json"]
    if args.get("profile"):
        sub.extend(["--profile", str(args["profile"])])
    if args.get("gpu"):
        sub.extend(["--gpu", str(args["gpu"])])
    res = _run_ctl_subprocess(ctl_script, sub, timeout=60)
    try:
        parsed = json.loads(res.get("stdout") or "{}")
        if isinstance(parsed, dict):
            return {"ok": bool(res.get("ok")), "report": _redact_obj(parsed)}
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        get_diag_logger().debug_throttled("mcp", "doctor_json_parse", f"解析 doctor 输出 JSON 失败: {exc}", interval_sec=30.0)
    return res


def _mcp_tail_logs(ctl_script: Path, args: dict[str, Any]) -> dict[str, Any]:
    source = str(args.get("source") or "local_diag").strip().lower()
    lines = max(10, min(500, int(args.get("lines") or 100)))
    if source == "local_diag":
        return {"ok": True, "source": "local_diag", "lines": DiagLogger().read_tail_lines(lines)}
    log_file = "/root/llama_server.log" if source == "llama" else "/root/comfy_colab.log"
    sess = "llm" if source == "llama" else "comfy"
    code = f"tail -n {lines} {log_file} 2>/dev/null || echo '[暂无远端日志]'"
    return _run_ctl_subprocess(ctl_script, ["exec", "-s", sess, "--mode", "bash", "--timeout", "45"], input_text=code, timeout=60)


def _mcp_list_gallery(ctl_script: Path, args: dict[str, Any]) -> dict[str, Any]:
    dl_root = ctl_script.resolve().parent.parent / "_downloads"
    if not dl_root.is_dir():
        return {"ok": True, "items": [], "note": "本地下载目录尚无作品"}
    rel_path = str(args.get("rel_path") or "").strip()
    if rel_path:
        target = (dl_root / rel_path).resolve()
        try:
            target.relative_to(dl_root.resolve())
        except ValueError:
            return {"ok": False, "error": "越界路径禁止访问"}
        if not target.is_file():
            return {"ok": False, "error": f"文件不存在: {rel_path}"}
        from colab_core.file_sync import extract_png_comfy_workflow

        wf_raw = extract_png_comfy_workflow(target) if target.suffix.lower() == ".png" else None
        return {
            "ok": True,
            "rel_path": rel_path,
            "size_bytes": target.stat().st_size,
            "metadata_json": redact_secrets(wf_raw or "")[:12000],
        }
    limit = max(1, min(100, int(args.get("limit") or 20)))
    files = sorted((p for p in dl_root.rglob("*") if p.is_file()), key=lambda p: p.stat().st_mtime, reverse=True)
    items = [{"rel_path": p.relative_to(dl_root).as_posix(), "size_bytes": p.stat().st_size} for p in files[:limit]]
    return {"ok": True, "items": items}


def _mcp_list_gdrive_assets() -> dict[str, Any]:
    rclone_bin = shutil.which("rclone")
    if not rclone_bin:
        return {"ok": False, "error": "本机未检测到 rclone 可执行文件"}
    sub_mod = _resolve_subprocess()
    assets: dict[str, list[str]] = {}
    for tag, remote in (
        ("loras", "gdrive_save:Colab_Anima_LoRAs"),
        ("workflows", "gdrive_save:Colab_Comfy_Workflows"),
        ("refs", "gdrive_save:Colab_Drama3_Refs"),
    ):
        try:
            proc = sub_mod.run(
                [rclone_bin, "lsf", remote],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
                env=clean_subprocess_env(),
                creationflags=CREATE_NO_WINDOW,
            )
            assets[tag] = [ln.strip() for ln in (proc.stdout or "").splitlines() if ln.strip()] if proc.returncode == 0 else []
        except Exception:
            assets[tag] = []
    return {"ok": True, "assets": assets}


def _mcp_exec_remote(ctl_script: Path, args: dict[str, Any], *, mutations_allowed: bool) -> dict[str, Any]:
    code = str(args.get("code") or "")
    mode = str(args.get("mode") or "bash").strip().lower()
    safe, reason = check_remote_code_safety(code, mode)
    if not safe:
        return {"ok": False, "blocked": True, "error": reason}
    is_ro, ro_reason = is_readonly_safe_remote_cmd(code, mode)
    if not is_ro and not mutations_allowed:
        return _approval_blocked(
            "exec_remote_colab",
            f"无头 MCP 默认仅开放只读诊断命令，已拦截非只读远端代码执行（{ro_reason}）。"
            "如需在无人值守环境放行，请显式设置环境变量 COLAB_MCP_ALLOW_MUTATIONS=1"
            " 或改用 Colab Studio 交互审批。",
        )
    slot = str(args.get("slot") or "auto").strip().lower()
    sess = "llm" if slot == "llm" else "comfy"
    timeout_sec = max(5, min(600, int(args.get("timeout_sec") or 180)))
    return _run_ctl_subprocess(
        ctl_script,
        ["exec", "-s", sess, "--mode", mode, "--timeout", str(timeout_sec)],
        input_text=code,
        timeout=timeout_sec + 30,
    )


def _mcp_manage_instance(ctl_script: Path, args: dict[str, Any], *, mutations_allowed: bool) -> dict[str, Any]:
    if not mutations_allowed:
        return _approval_blocked(
            "manage_compute_instance",
            "无头 MCP 缺少人工交互审批通道，默认禁止自动开关计费实例 (manage_compute_instance)。"
            "如需在受控无人值守模式放行，请显式设置环境变量 COLAB_MCP_ALLOW_MUTATIONS=1"
            " 或在 Colab Studio 审批卡中手动确认。",
        )
    action = str(args.get("action") or "start").strip().lower()
    if action == "stop_all":
        return _run_ctl_subprocess(ctl_script, ["down", "--all"], timeout=240)
    if action == "stop_slot":
        slot = str(args.get("slot") or "comfy").strip() or "comfy"
        return _run_ctl_subprocess(ctl_script, ["down", "-s", slot], timeout=240)
    if action == "start":
        sub = ["up"]
        profs = args.get("profiles") or []
        if isinstance(profs, list) and profs:
            sub.extend(["--profiles", ",".join(str(p) for p in profs)])
        if args.get("gpu"):
            sub.extend(["--gpu", str(args["gpu"]).upper()])
        return _run_ctl_subprocess(ctl_script, sub, timeout=1800)
    return {"ok": False, "error": f"未知实例操作: {action}"}


def _default_mcp_executor(
    ctl_script: Path,
    *,
    allow_unattended_mutations: bool | None = None,
) -> Callable[[str, dict[str, Any]], dict[str, Any]]:
    """构造通过子进程调用 `colab_ctl.py` 的默认 MCP 工具执行器。

    安全边界（默认 Fail-Closed）：
    - 无头 MCP Stdio 默认无交互审批通道，仅开放 `tier == "safe"` 的结构化只读诊断工具，
      以及通过 `is_readonly_safe_remote_cmd` 严格白名单校验的只读 Bash 诊断命令；
    - 对非只读 `exec_remote_colab`、存在写副作用的 `sync_and_pull_outputs` 与计费/开关机工具
      `manage_compute_instance`，默认拒绝并返回 `approval_required=True` 的结构化审批拦截错误；
    - 仅当调用方显式传入 `allow_unattended_mutations=True` 或设置环境变量
      `COLAB_MCP_ALLOW_MUTATIONS=1` 时放行变更工具（但硬黑名单 `check_remote_code_safety` 永远不可绕过）。
    """
    script_path = Path(ctl_script)

    def _execute(name: str, args: dict[str, Any]) -> dict[str, Any]:
        mutations_allowed = _is_mcp_mutations_allowed(allow_unattended_mutations)
        if name == "run_environment_doctor":
            return _mcp_run_doctor(script_path, args)
        if name == "get_cluster_and_pool_status":
            return _run_ctl_subprocess(script_path, ["status"], timeout=45)
        if name == "tail_instance_logs":
            return _mcp_tail_logs(script_path, args)
        if name == "list_gallery_and_inspect":
            return _mcp_list_gallery(script_path, args)
        if name == "list_gdrive_assets":
            return _mcp_list_gdrive_assets()
        if name == "exec_remote_colab":
            return _mcp_exec_remote(script_path, args, mutations_allowed=mutations_allowed)
        if name == "sync_and_pull_outputs":
            if not mutations_allowed:
                return _approval_blocked(
                    name,
                    "无头 MCP 默认处于只读安全模式，已拦截具有写副作用的 sync_and_pull_outputs。"
                    "如需放行，请显式设置环境变量 COLAB_MCP_ALLOW_MUTATIONS=1 或在 Colab Studio 界面执行。",
                )
            return _run_ctl_subprocess(script_path, ["sync", "--pull"], timeout=300)
        if name == "manage_compute_instance":
            return _mcp_manage_instance(script_path, args, mutations_allowed=mutations_allowed)
        if name in ("queue_comfy_workflow", "reconfigure_llm_engine"):
            if not mutations_allowed:
                return _approval_blocked(
                    name,
                    f"无头 MCP 默认处于只读安全模式，且工具 {name} 需在同进程 Colab Studio 运行态并经受控上下文调用",
                )
            return {
                "ok": False,
                "error": f"工具 {name} 需在同进程 Colab Studio 运行态或指定在线 ComfyUI/LLM 端点上下文中调用",
            }
        return {"ok": False, "error": f"未知工具: {name}"}

    return _execute


def _read_json_line(stream: Any) -> str | None:
    line = stream.readline()
    if not line:
        return None
    if isinstance(line, bytes):
        return line.decode("utf-8", errors="replace")
    return str(line)


def _write_json_line(stream: Any, payload: dict[str, Any]) -> None:
    txt = json.dumps(payload, ensure_ascii=False) + "\n"
    raw = txt.encode("utf-8")
    if isinstance(stream, io.TextIOBase) or hasattr(stream, "encoding"):
        try:
            stream.write(txt)
            stream.flush()
            return
        except TypeError:
            pass
    stream.write(raw)
    stream.flush()


def _dispatch_jsonrpc_request(
    req: dict[str, Any],
    exec_fn: Callable[[str, dict[str, Any]], dict[str, Any]],
    mcp_tools: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """处理单条已解析的 MCP JSON-RPC 2.0 请求，通知类消息返回 None。"""
    req_id = req.get("id")
    has_id = "id" in req and req_id is not None
    method = str(req.get("method") or "").strip()
    params = req.get("params") if isinstance(req.get("params"), dict) else {}

    if method == "notifications/initialized" or (method.startswith("notifications/") and not has_id) or not has_id:
        return None

    if method == "initialize":
        proto = str(params.get("protocolVersion") or "2024-11-05")
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": proto,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "colab-studio-mcp", "version": "1.0.0"},
            },
        }
    if method == "ping":
        return {"jsonrpc": "2.0", "id": req_id, "result": {}}
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": mcp_tools}}
    if method == "tools/call":
        tname = str(params.get("name") or "").strip()
        targs = params.get("arguments") if isinstance(params.get("arguments"), dict) else {}
        if tname not in AGENT_TOOL_MAP:
            return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32602, "message": f"未知工具名称: {tname}"}}
        try:
            raw_res = exec_fn(tname, targs)
            res_dict = raw_res if isinstance(raw_res, dict) else {"ok": True, "result": raw_res}
        except Exception as exc:
            res_dict = {"ok": False, "error": redact_secrets(str(exc) or type(exc).__name__)}
        is_err = not bool(res_dict.get("ok", True)) or bool(res_dict.get("error") and "ok" not in res_dict)
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "content": [{"type": "text", "text": _safe_json_dumps(res_dict)}],
                "isError": is_err,
            },
        }
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": f"不支持的方法: {method}"}}


def run_mcp_stdio_server(
    ctl_script: Path,
    executor: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
    *,
    stdin: Any = None,
    stdout: Any = None,
    allow_unattended_mutations: bool | None = None,
) -> int:
    """运行基于标准库 JSON-RPC 2.0 的 MCP Stdio 工具服务器循环。"""
    in_stream = stdin if stdin is not None else sys.stdin.buffer
    out_stream = stdout if stdout is not None else sys.stdout.buffer
    exec_fn = (
        executor
        if executor is not None
        else _default_mcp_executor(Path(ctl_script), allow_unattended_mutations=allow_unattended_mutations)
    )
    mcp_tools = [
        {
            "name": spec["name"],
            "description": f"[{spec['tier']}] {spec['description']}",
            "inputSchema": spec["parameters"],
        }
        for spec in AGENT_TOOL_SPECS
    ]

    while True:
        raw_line = _read_json_line(in_stream)
        if raw_line is None:
            break
        line = raw_line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as exc:
            _write_json_line(out_stream, {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": f"JSON 解析失败: {exc}"}})
            continue
        if not isinstance(req, dict):
            continue
        resp = _dispatch_jsonrpc_request(req, exec_fn, mcp_tools)
        if resp is not None:
            _write_json_line(out_stream, resp)

    return 0


__all__ = [
    "_default_mcp_executor",
    "_is_mcp_mutations_allowed",
    "_read_json_line",
    "_write_json_line",
    "run_mcp_stdio_server",
]
