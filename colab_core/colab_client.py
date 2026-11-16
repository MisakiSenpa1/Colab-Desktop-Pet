"""colab_core.colab_client —— Colab 远端执行、会话纳管、停卡核实与公网隧道探测模块。"""

from __future__ import annotations

import ast
import inspect
import json
import re
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

_PKG_DIR = Path(__file__).resolve().parents[1]
if str(_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR))

from colab_core.accounts import load_cf_config  # noqa: E402
from colab_core.runtime import extract_framed_json  # noqa: E402
from colab_security import extract_valid_https_url, redact_secrets  # noqa: E402

DEFAULT_TUNNEL_URL_RE = re.compile(
    r"https://(?:(?!api\.)[a-z0-9-]+\.trycloudflare\.com)"
)

STOP_PROBE_TIMEOUT_SEC = 60.0
STOP_CMD_TIMEOUT_SEC = 120.0
STOP_FALLBACK_TIMEOUT_SEC = 180.0
STOP_VERIFY_TIMEOUT_SEC = 120.0

# 孤儿实例纳管短脚本（需保持 COLAB_ADOPT_SESSION 标记以兼容离线单测桩）
COLAB_ADOPT_ORPHAN_SCRIPT = """
import json
import sys
from colab_cli.common import state
from colab_cli.state import SessionState

req = json.loads(sys.stdin.read() or "{}")
sname = str(req.get("session") or "comfy").strip() or "comfy"
target_ep = str(req.get("target_endpoint") or "").strip()
pref_gpu = str(req.get("preferred_gpu") or "").strip().upper()

local_sessions = state.store.list() or {}
assignments = state.client.list_assignments() or []
if not assignments:
    print("<<<COLAB_ADOPT_SESSION>>>" + json.dumps({"adopted": False, "reason": "no_assignments"}) + "<<<END_COLAB_ADOPT_SESSION>>>", flush=True)
    raise SystemExit(0)

by_ep = {a.endpoint: a for a in assignments if getattr(a, "endpoint", None)}
existing = local_sessions.get(sname)
if existing and existing.endpoint in by_ep:
    a = by_ep[existing.endpoint]
    existing.token = a.runtime_proxy_info.token
    existing.url = a.runtime_proxy_info.url
    existing.token_expires_at = a.runtime_proxy_info.expires_at()
    state.store.add(existing)
    print("<<<COLAB_ADOPT_SESSION>>>" + json.dumps({"adopted": True, "already_named": True, "endpoint": a.endpoint, "accelerator": a.accelerator.value}) + "<<<END_COLAB_ADOPT_SESSION>>>", flush=True)
    raise SystemExit(0)

bound_other_eps = {s.endpoint for k, s in local_sessions.items() if k != sname and getattr(s, "endpoint", None)}
cand = by_ep.get(target_ep) if target_ep else None
if cand is None:
    unclaimed = [a for a in assignments if a.endpoint not in bound_other_eps]
    if pref_gpu:
        cand = next((a for a in unclaimed if str(a.accelerator.value).upper() == pref_gpu), None)
    if cand is None and len(unclaimed) == 1 and not pref_gpu:
        cand = unclaimed[0]

if cand is None:
    print("<<<COLAB_ADOPT_SESSION>>>" + json.dumps({"adopted": False, "reason": "no_matching_orphan"}) + "<<<END_COLAB_ADOPT_SESSION>>>", flush=True)
    raise SystemExit(0)

shape_obj = getattr(cand, "machine_shape", None)
var_obj = getattr(cand, "variant", None)
state.store.add(SessionState(
    name=sname,
    token=cand.runtime_proxy_info.token,
    url=cand.runtime_proxy_info.url,
    endpoint=cand.endpoint,
    token_expires_at=cand.runtime_proxy_info.expires_at(),
    variant=str(getattr(var_obj, "name", "GPU") or "GPU"),
    accelerator=str(cand.accelerator.value),
    machine_shape="HIGH_RAM" if getattr(shape_obj, "name", "") == "HIGH_RAM" else "STANDARD",
))
print("<<<COLAB_ADOPT_SESSION>>>" + json.dumps({"adopted": True, "already_named": False, "endpoint": cand.endpoint, "accelerator": str(cand.accelerator.value)}) + "<<<END_COLAB_ADOPT_SESSION>>>", flush=True)
""".strip()

# 在远端 Jupyter 内核拉取 LoRA 的短片段
COLAB_REMOTE_LORA_PULL_SCRIPT = """
import sys
from colab_cli.common import state
from colab_cli.runtime import ColabRuntime

session_name, src_remote, dst_local = sys.argv[1], sys.argv[2], sys.argv[3]
s = state.get_session(session_name)
if s is None:
    raise RuntimeError(f"未找到本地会话 '{session_name}'")
rt = ColabRuntime(s.url, s.token, session_name=session_name, kernel_id=s.kernel_id, session_id=s.session_id)
try:
    ws = rt.kernel_client._manager.client
    if not ws.connection_ready.is_set() and not ws.connection_ready.wait(timeout=25.0):
        raise RuntimeError("WebSocket 握手超时")
    code = f"import os, shutil, subprocess\\nos.makedirs(os.path.dirname({dst_local!r}), exist_ok=True)\\nrbin = shutil.which('rclone') or '/usr/local/bin/rclone'\\nsubprocess.run([rbin, 'copyto', '--config', '/tmp/rclone.conf', '--disable-http2', {src_remote!r}, {dst_local!r}], check=True)\\nprint('REMOTE_LORA_READY')"
    errs = [o for o in rt.execute_code(code, timeout=180.0) if o.get("output_type") == "error"]
    if errs:
        raise RuntimeError(str(errs[0]))
    print("REMOTE_LORA_READY")
finally:
    rt.stop()
""".strip()

REMOTE_RUNNER_TEMPLATE = r"""
import json, os, sys, time, traceback
from pathlib import Path
from colab_cli.common import state
from colab_cli.runtime import ColabRuntime
from colab_cli.utils import is_terminal_error

SESSION_NAME = __SESSION_NAME__
TARGET_SCRIPT = __TARGET_SCRIPT__
EXEC_TIMEOUT = __EXEC_TIMEOUT__
MAX_RETRIES = __MAX_RETRIES__
REMOTE_FILE = __REMOTE_FILE__
EXTRA_FILES = __EXTRA_FILES__
MANIFEST_OUT_DIR = __MANIFEST_OUT_DIR__

code = Path(TARGET_SCRIPT).read_text(encoding="utf-8")
exit_code = 1

for attempt in range(1, MAX_RETRIES + 1):
    s = state.get_session(SESSION_NAME)
    if s is None:
        print(f"错误: 未找到有效 Colab 会话 '{SESSION_NAME}'。", file=sys.stderr, flush=True)
        exit_code = 1
        break

    rt = ColabRuntime(
        s.url, s.token, session_name=SESSION_NAME, kernel_id=s.kernel_id, session_id=s.session_id,
        on_kernel_started=lambda kid: (setattr(s, "kernel_id", kid), state.store.add(s)),
        on_session_started=lambda sid: (setattr(s, "session_id", sid), state.store.add(s)),
    )
    printed_error = False
    try:
        ws_client = rt.kernel_client._manager.client
        if not ws_client.connection_ready.is_set() and not ws_client.connection_ready.wait(timeout=30.0):
            raise RuntimeError("WebSocket 连接握手超时 (30s)")

        prep = [
            "import os, sys, shutil",
            "from pathlib import Path",
            "os.makedirs('/content', exist_ok=True)",
            "os.chdir('/content')",
            "if '/content' not in sys.path: sys.path.insert(0, '/content')",
        ]
        if EXTRA_FILES:
            prep.append(f"for _fn, _ft in {json.dumps(EXTRA_FILES, ensure_ascii=False)}.items(): Path('/content', Path(_fn).name).write_text(_ft, encoding='utf-8')")
        if REMOTE_FILE:
            prep.extend([f"__file__ = {REMOTE_FILE!r}", f"sys.argv = [{REMOTE_FILE!r}]", f"Path({REMOTE_FILE!r}).write_text({code!r}, encoding='utf-8')"])
        else:
            prep.extend(["globals().pop('__file__', None)", "sys.argv = ['']"])
        if MANIFEST_OUT_DIR:
            prep.append("_colab_manifest_before = {p.name: p.stat().st_mtime_ns for p in Path('/content').glob('_manifest_*.json') if p.is_file()}")
        rt.execute_code("\n".join(prep), timeout=60.0)

        def _on_out(o):
            global printed_error
            ot = o.get("output_type")
            if ot == "stream":
                dst = sys.stderr if o.get("name") == "stderr" else sys.stdout
                dst.write(o.get("text", "")); dst.flush()
            elif ot in ("execute_result", "display_data"):
                txt = (o.get("data") or {}).get("text/plain")
                if txt is not None:
                    print(txt, flush=True)
            elif ot == "error":
                printed_error = True
                sys.stderr.write("\n".join(o.get("traceback") or [f"{o.get('ename', 'Error')}: {o.get('evalue', '')}"]) + "\n")
                sys.stderr.flush()

        outputs = rt.execute_code(code, output_hook=_on_out, timeout=float(EXEC_TIMEOUT))
        if MANIFEST_OUT_DIR:
            try:
                rt.execute_code(
                    f"_mout = Path({MANIFEST_OUT_DIR!r}); _mout.mkdir(parents=True, exist_ok=True); _mb = globals().get('_colab_manifest_before') or {{}}\n"
                    "for _mp in Path('/content').glob('_manifest_*.json'):\n"
                    "    if _mp.is_file() and _mp.stat().st_mtime_ns != _mb.get(_mp.name): shutil.copy2(_mp, _mout / _mp.name)\n",
                    timeout=30.0,
                )
            except Exception:
                pass
        errs = [o for o in outputs if o.get("output_type") == "error"]
        if errs:
            if not printed_error:
                for eo in errs:
                    sys.stderr.write("\n".join(eo.get("traceback") or [f"{eo.get('ename', 'Error')}: {eo.get('evalue', '')}"]) + "\n")
                sys.stderr.flush()
            exit_code = 1
            break
        exit_code = 0
        break
    except Exception as exc:
        msg = str(exc)
        is_conn = (not is_terminal_error(exc)) and (any(k in msg for k in ("Connection was lost", "WebSocket", "握手超时")) or "connection" in msg.lower() or isinstance(exc, (TimeoutError, ConnectionError, OSError)))
        if is_conn and attempt < MAX_RETRIES:
            wait_s = 3 * attempt
            print(f"[远端重连] 第 {attempt}/{MAX_RETRIES} 次连接异常 ({exc})，{wait_s} 秒后重试...", file=sys.stderr, flush=True)
            time.sleep(wait_s)
            continue
        traceback.print_exc()
        exit_code = 1
        break
    finally:
        try:
            rt.stop()
        except Exception:
            pass

sys.stdout.flush()
sys.stderr.flush()
os._exit(exit_code)
"""


def extract_valid_https_host(url: str | None) -> str | None:
    try:
        norm = extract_valid_https_url(url or "")
        return norm[8:] if norm else None
    except ValueError:
        return None


def build_tunnel_url_pattern(cf_cfg: dict[str, Any] | None = None, *, cf_config_path: Path | None = None, colab_py: Path | None = None) -> str:
    branches = [r"(?!api\.)[a-z0-9-]+\.trycloudflare\.com"]
    cfg = cf_cfg if isinstance(cf_cfg, dict) else (load_cf_config(cf_config_path, include_secrets=False, colab_py=colab_py) if cf_config_path else None)
    if isinstance(cfg, dict):
        dom_host = extract_valid_https_host("https://" + str(cfg.get("domain") or "").strip().lstrip("."))
        if dom_host:
            dom_branch = r"[a-z0-9-]+\." + re.escape(dom_host)
            if dom_branch not in branches:
                branches.append(dom_branch)
        for key in ("comfy_url", "llm_url"):
            host = extract_valid_https_host(str(cfg.get(key) or ""))
            if host and (esc := re.escape(host)) not in branches:
                branches.append(esc)
    return r"https://(?:" + "|".join(branches) + r")"


def is_allowed_tunnel_url(url: str, *, config_path: Path | None = None, base_regex: re.Pattern[str] = DEFAULT_TUNNEL_URL_RE) -> bool:
    cand = str(url or "").strip().rstrip("/")
    if not cand:
        return False
    if base_regex.match(cand):
        return True
    if config_path is not None:
        cfg = load_cf_config(config_path, include_secrets=False)
        for key in ("comfy_url", "llm_url"):
            saved = str(cfg.get(key) or "").strip().rstrip("/")
            if saved and cand.lower() == saved.lower():
                return True
    return False


def probe_comfy_endpoint_ok(url: str, comfy_key: str = "", timeout: float = 5.0) -> tuple[bool, dict[str, Any]]:
    base = str(url or "").strip().rstrip("/")
    headers = {"User-Agent": "Mozilla/5.0", **({"Authorization": f"Bearer {comfy_key}"} if comfy_key else {})}
    try:
        with urllib.request.urlopen(urllib.request.Request(f"{base}/system_stats", headers=headers), timeout=timeout) as resp:
            if resp.status == 200:
                body = json.loads(resp.read().decode("utf-8", errors="ignore") or "{}")
                return True, body if isinstance(body, dict) else {}
    except urllib.error.HTTPError as exc:
        if exc.code == 401:
            with urllib.request.urlopen(urllib.request.Request(f"{base}/__pet_health", headers={"User-Agent": "Mozilla/5.0"}), timeout=timeout) as hr:
                if hr.status == 200:
                    hb = json.loads(hr.read().decode("utf-8", errors="ignore") or "{}")
                    if isinstance(hb, dict) and str(hb.get("status") or "").lower() == "ok":
                        return True, {"auth_protected": True, **hb}
        raise
    return False, {}


def probe_tunnel_alive(
    url: str,
    timeout: float = 3.5,
    kind: str = "comfy",
    auth_token: str = "",
    cf_config_path: Path | None = None,
    allowed_checker: Callable[[str], bool] | None = None,
) -> bool:
    if not (allowed_checker(url) if allowed_checker else is_allowed_tunnel_url(url, config_path=cf_config_path)):
        return False
    base = url.strip().rstrip("/")
    mode = str(kind or "comfy").strip().lower()

    def _probe_single(k: str) -> bool:
        if k == "llm":
            try:
                with urllib.request.urlopen(urllib.request.Request(f"{base}/health", headers={"User-Agent": "Mozilla/5.0"}), timeout=timeout) as resp:
                    body = json.loads(resp.read().decode("utf-8", errors="ignore") or "{}")
                    return resp.status == 200 and isinstance(body, dict) and str(body.get("status") or "").lower() == "ok"
            except Exception:
                return False
        try:
            ok, body = probe_comfy_endpoint_ok(base, comfy_key=auth_token, timeout=timeout)
            return ok and isinstance(body, dict) and ("system" in body or "devices" in body or body.get("auth_protected") is True)
        except Exception:
            return False

    return _probe_single("llm") if mode == "llm" else (_probe_single("comfy") if mode == "comfy" else (_probe_single("comfy") or _probe_single("llm")))


def detect_tunnel_kind(
    url: str,
    timeout: float = 3.0,
    preferred: str = "comfy",
    auth_token: str = "",
    cf_config_path: Path | None = None,
    probe_fn: Callable[..., bool] | None = None,
) -> str | None:
    fn = probe_fn or probe_tunnel_alive
    for cand in (("llm", "comfy") if preferred == "llm" else ("comfy", "llm")):
        if fn(url, timeout=timeout, kind=cand, auth_token=auth_token, cf_config_path=cf_config_path):
            return cand
    return None


def collect_local_py_deps(entry_script: Path, max_files: int = 16, max_total_bytes: int = 512 * 1024) -> dict[str, str]:
    base_dir = entry_script.resolve().parent
    deps: dict[str, str] = {}
    visited: set[Path] = {entry_script.resolve()}
    queue: list[Path] = [entry_script.resolve()]
    total_bytes = 0

    def _enqueue(cand: Path) -> None:
        resolved = cand.resolve()
        if resolved not in visited and resolved.parent in (base_dir, _PKG_DIR) and resolved.is_file() and resolved.suffix == ".py":
            visited.add(resolved)
            queue.append(resolved)

    def _enqueue_name(fname: str) -> None:
        _enqueue(base_dir / fname)
        if not (base_dir / fname).is_file():
            _enqueue(_PKG_DIR / fname)

    while queue and len(deps) < max_files:
        cur = queue.pop(0)
        try:
            src = cur.read_text(encoding="utf-8")
        except Exception:
            continue
        if cur != entry_script.resolve():
            raw_len = len(src.encode("utf-8"))
            if total_bytes + raw_len > max_total_bytes:
                print(f"[警告] 跳过同级依赖 {cur.name}: 累计体积超出 {max_total_bytes} B 限制", flush=True)
                continue
            deps[cur.name] = src
            total_bytes += raw_len
        try:
            tree = ast.parse(src, filename=str(cur))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    _enqueue_name(f"{a.name.split('.', 1)[0]}.py")
            elif isinstance(node, ast.ImportFrom) and node.module:
                _enqueue_name(f"{node.module.split('.', 1)[0]}.py")
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                val = node.value.strip()
                if val.endswith(".py") and "/" not in val and "\\" not in val and len(val) <= 128:
                    _enqueue_name(val)
    return deps


def exec_remote_file(
    session: str,
    script_path: Path,
    *,
    colab_py: Path,
    run_fn: Callable[..., Any],
    timeout: int = 600,
    capture: bool = False,
    retries: int = 3,
    remote_file: str | None = None,
    extra_files: dict[str, str] | None = None,
    manifest_out_dir: str | None = None,
):
    if not colab_py.is_file():
        raise FileNotFoundError(f"未找到google-colab-cli python: {colab_py}")
    if not script_path.is_file():
        raise FileNotFoundError(f"未找到待执行脚本 {script_path.name}: {script_path}")

    merged_extra = collect_local_py_deps(script_path)
    if extra_files:
        merged_extra.update(extra_files)
    runner_code = (
        REMOTE_RUNNER_TEMPLATE
        .replace("__SESSION_NAME__", repr(session))
        .replace("__TARGET_SCRIPT__", repr(str(script_path)))
        .replace("__EXEC_TIMEOUT__", repr(int(timeout)))
        .replace("__MAX_RETRIES__", repr(int(retries)))
        .replace("__REMOTE_FILE__", repr(remote_file))
        .replace("__EXTRA_FILES__", repr(merged_extra))
        .replace("__MANIFEST_OUT_DIR__", repr(manifest_out_dir))
    )
    print(f"[远端执行] session={session} | script={script_path.name} (timeout={timeout}s)", flush=True)
    with tempfile.TemporaryDirectory(prefix="colab_runner_") as tmp_dir:
        runner_path = Path(tmp_dir) / "runner.py"
        runner_path.write_text(runner_code, encoding="utf-8")
        return run_fn([colab_py, runner_path], check=True, capture=capture)


def exec_remote_py(session: str, code: str, *, exec_file_fn: Callable[..., Any], timeout: int = 300, capture: bool = False):
    with tempfile.TemporaryDirectory(prefix="colab_remote_") as tmp_dir:
        script_path = Path(tmp_dir) / "remote_task.py"
        script_path.write_text(code, encoding="utf-8")
        return exec_file_fn(session, script_path, timeout=timeout, capture=capture)


def resolve_target_binding(
    *,
    sessions: list[dict[str, Any]],
    binding: dict[str, Any] | None,
    active_account_id: str,
    session: str = "comfy",
    endpoint: str | None = None,
) -> tuple[str, str, bool]:
    explicit_ep = str(endpoint or "").strip()
    wanted_sess = str(session or "comfy").strip() or "comfy"
    if explicit_ep:
        matched = next((s for s in sessions if str(s.get("endpoint") or "") == explicit_ep), None)
        return (str(matched["name"]) if matched and matched.get("name") not in (None, "?") else wanted_sess), explicit_ep, False
    named_hit = next((s for s in sessions if str(s.get("name") or "") == wanted_sess and s.get("endpoint")), None)
    if named_hit:
        return str(named_hit["name"]), str(named_hit["endpoint"]), False
    if isinstance(binding, dict) and str(binding.get("endpoint") or "").strip() and str(binding.get("account_id") or "").strip() == str(active_account_id or "").strip():
        b_ep = str(binding["endpoint"]).strip()
        if not sessions or any(str(s.get("endpoint") or "") == b_ep for s in sessions):
            return (str(binding.get("session") or wanted_sess).strip() or wanted_sess), b_ep, False
    any_named = next((s for s in sessions if s.get("name") not in (None, "?") and s.get("endpoint")), None)
    if any_named:
        return str(any_named["name"]), str(any_named["endpoint"]), False
    if sessions and all(s.get("orphan") or s.get("name") in (None, "?") for s in sessions):
        return wanted_sess, "", True
    return wanted_sess, "", False


def _session_name_in_cli_output(sessions_out: str, session: str) -> bool:
    if not sessions_out or "No active sessions" in sessions_out:
        return False
    return bool(re.search(rf"\[{re.escape(session)}\]", sessions_out) or re.search(rf"(?<![A-Za-z0-9_.-]){re.escape(session)}(?![A-Za-z0-9_.-])", sessions_out))


def _try_adopt_orphan_session(*, session: str, target_endpoint: str | None, preferred_gpu: str, colab_py: Path, run_fn: Callable[..., Any]) -> dict[str, Any] | None:
    if not colab_py.is_file():
        return None
    payload = json.dumps({"session": session, "target_endpoint": str(target_endpoint or "").strip(), "preferred_gpu": str(preferred_gpu or "").strip().upper()}, ensure_ascii=False)
    adopt_code = f"import io, sys\nsys.stdin = io.StringIO({payload!r})\n" + COLAB_ADOPT_ORPHAN_SCRIPT
    try:
        res = run_fn([colab_py, "-c", adopt_code], check=False, capture=True)
        pkt = extract_framed_json(getattr(res, "stdout", "") or "", "COLAB_ADOPT_SESSION")
        return pkt if isinstance(pkt, dict) and pkt.get("adopted") else None
    except Exception:
        return None


def ensure_or_adopt_session(
    *,
    session: str,
    gpu: str,
    high_mem: bool,
    recreate: bool,
    target_endpoint: str | None,
    colab_exe: Path,
    colab_py: Path,
    run_fn: Callable[..., Any],
    stop_fn: Callable[..., Any],
) -> bool:
    sessions_res = run_fn([colab_exe, "sessions"], check=False, capture=True)
    sessions_out = getattr(sessions_res, "stdout", "") or ""
    has_session = getattr(sessions_res, "returncode", 1) == 0 and _session_name_in_cli_output(sessions_out, session)
    target_ep_arg = str(target_endpoint or "").strip() or None

    if recreate and (has_session or target_ep_arg):
        print("=== 正在定向释放目标会话旧实例以重新分配显卡（同账号其余会话不受影响） ===", flush=True)
        stop_fn(session, clean_all=False, target_endpoint=target_ep_arg, verify_release=True)
        has_session = False

    if not has_session and not recreate and (target_ep_arg or "[?]" in sessions_out):
        adopted = _try_adopt_orphan_session(session=session, target_endpoint=target_ep_arg, preferred_gpu=gpu, colab_py=colab_py, run_fn=run_fn)
        if adopted:
            print(f"检测到已有同规格活跃实例 ({adopted.get('endpoint') or ''} · {adopted.get('accelerator') or gpu})，已自动纳管为会话 {session}，跳过 colab new，继续增量部署...", flush=True)
            return True

    if has_session:
        print(f"检测到会话 {session} 已存在，跳过 colab new，继续增量部署...", flush=True)
        return True

    print(f"=== 开启 Colab 实例 (session={session}, gpu={gpu}, high_mem={high_mem}) ===", flush=True)
    new_cmd: list[Any] = [colab_exe, "new", "-s", session, "--gpu", gpu] + (["--high-mem"] if high_mem else [])
    try:
        run_fn(new_cmd)
        return False
    except Exception:
        adopted_after = _try_adopt_orphan_session(session=session, target_endpoint=target_ep_arg, preferred_gpu=gpu, colab_py=colab_py, run_fn=run_fn)
        if adopted_after:
            print(f"提示: colab new 响应超时，但云端实例 ({adopted_after.get('endpoint') or ''}) 已完成分配并已自动纳管，继续部署...", flush=True)
            return False
        raise


def run_with_timeout(run_fn: Callable[..., Any], cmd: list[Any], timeout: float | None) -> Any:
    try:
        accepts = any(p.name == "timeout" or p.kind is inspect.Parameter.VAR_KEYWORD for p in inspect.signature(run_fn).parameters.values())
    except (TypeError, ValueError):
        accepts = False
    return run_fn(cmd, check=False, capture=True, timeout=timeout) if (timeout is not None and accepts) else run_fn(cmd, check=False, capture=True)


def probe_session_endpoint(session: str, *, colab_py: Path, run_fn: Callable[..., Any], timeout: float | None = STOP_PROBE_TIMEOUT_SEC) -> str | None:
    if not colab_py.is_file():
        print(f"[警告] 未找到 Colab CLI Python 解释器 ({colab_py})，无法查询会话 '{session}' 的 Endpoint", file=sys.stderr, flush=True)
        return None
    probe_ep_code = f"from colab_cli.common import state\ns = state.get_session({session!r}, ignore_missing_session=True)\nprint(s.endpoint if s and getattr(s, 'endpoint', None) else '')\n"
    try:
        ep_res = run_with_timeout(run_fn, [colab_py, "-c", probe_ep_code], timeout)
    except subprocess.TimeoutExpired:
        print(f"[警告] 查询会话 '{session}' 的 Endpoint 超时（{timeout:.0f}s），已放弃", file=sys.stderr, flush=True)
        return None
    except Exception as exc:
        print(f"[警告] 查询会话 '{session}' 的 Endpoint 失败: {redact_secrets(str(exc) or type(exc).__name__)}", file=sys.stderr, flush=True)
        return None
    if int(getattr(ep_res, "returncode", 1)) != 0:
        err_lines = [ln.strip() for ln in redact_secrets(str(getattr(ep_res, "stderr", "") or "")).splitlines() if ln.strip()]
        print(f"[警告] 查询会话 '{session}' 的 Endpoint 失败（退出码 {ep_res.returncode}）{': ' + err_lines[-1] if err_lines else ''}", file=sys.stderr, flush=True)
        return None
    cand = (getattr(ep_res, "stdout", "") or "").strip().splitlines()
    if cand and cand[-1].strip():
        return cand[-1].strip()
    print(f"[警告] 本地未找到会话 '{session}' 的 Endpoint 记录", file=sys.stderr, flush=True)
    return None


def stop_session_cli(
    *,
    session: str,
    clean_all: bool,
    target_endpoint: str | None,
    verify_release: bool,
    colab_exe: Path,
    colab_py: Path,
    run_fn: Callable[..., Any],
    status_fn: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    import time

    if not colab_exe.is_file():
        raise FileNotFoundError(f"未找到colab.exe: {colab_exe}")

    target_ep = str(target_endpoint or "").strip() or None
    if not target_ep and not clean_all:
        target_ep = probe_session_endpoint(session, colab_py=colab_py, run_fn=run_fn)
        if not target_ep:
            print(f"[警告] 未能预先锁定会话 '{session}' 的目标实例 Endpoint，停卡结果将无法精确核实", file=sys.stderr, flush=True)

    stop_ok = stop_missing = stop_timed_out = False
    last_rc = 1
    for attempt in range(1, 4):
        try:
            res = run_with_timeout(run_fn, [colab_exe, "stop", "-s", session], STOP_CMD_TIMEOUT_SEC)
        except subprocess.TimeoutExpired:
            last_rc, stop_timed_out = -1, True
            print(f"[警告] colab stop -s {session} 超时（{STOP_CMD_TIMEOUT_SEC:.0f}s），立即转入云端 Endpoint 兜底释放...", file=sys.stderr, flush=True)
            break
        last_rc = int(getattr(res, "returncode", 1))
        stop_out, stop_err = str(getattr(res, "stdout", "") or ""), str(getattr(res, "stderr", "") or "")
        if stop_out:
            print(redact_secrets(stop_out), end="" if stop_out.endswith("\n") else "\n", flush=True)
        if stop_err:
            print(redact_secrets(stop_err), file=sys.stderr, end="" if stop_err.endswith("\n") else "\n", flush=True)
        combined_low = f"{stop_out}\n{stop_err}".lower()
        if "not found" in combined_low and ("session" in combined_low or session.lower() in combined_low):
            stop_missing = True
            print(f"提示: colab stop -s {session} 返回会话未找到，立即转入云端 Endpoint 兜底核查...", flush=True)
            break
        if last_rc == 0:
            stop_ok = True
            break
        print(f"提示: 第 {attempt}/3 次 colab stop -s {session} 返回 {last_rc}，准备重试/兜底释放...", flush=True)
        if attempt < 3:
            time.sleep(2)

    fb_unhandled_eps: list[str] = []
    if colab_py.is_file() and (clean_all or not stop_ok):
        fallback_code = f"""
import json, time
from colab_cli.common import state
session_name, target_ep, clean_all = {session!r}, {target_ep!r}, {bool(clean_all)!r}
acted, active_eps = False, []
for attempt in range(1, 4):
    try:
        assignments = state.client.list_assignments()
        active_eps = [ep for item in (assignments or []) if (ep := getattr(item, "endpoint", None) or (item.get("endpoint") if isinstance(item, dict) else None))]
        if clean_all:
            for ep in active_eps:
                print(f"正在释放实例: {{ep}}", flush=True)
                state.client.unassign(ep)
            try:
                for sn in list((state.store.list() or {{}}).keys()): state.store.remove(sn)
            except Exception:
                pass
            acted = True
        elif target_ep:
            if target_ep in active_eps:
                print(f"兜底释放目标会话实例: {{target_ep}}", flush=True)
                state.client.unassign(target_ep)
            try:
                state.store.remove(session_name)
            except Exception:
                pass
            acted = True
        else:
            acted = (len(active_eps) == 0)
        break
    except Exception as exc:
        if attempt == 3: raise
        print(f"释放接口第 {{attempt}}/3 次异常 ({{exc}})，2 秒后重试...", flush=True)
        time.sleep(2)
print("<<<STOP_FALLBACK>>>" + json.dumps({{"acted": acted, "active_eps": active_eps}}, ensure_ascii=False) + "<<<END_STOP_FALLBACK>>>", flush=True)
"""
        fb_res = None
        try:
            fb_res = run_with_timeout(run_fn, [colab_py, "-c", fallback_code], STOP_FALLBACK_TIMEOUT_SEC)
        except subprocess.TimeoutExpired:
            print(f"[警告] 云端 Endpoint 兜底释放超时（{STOP_FALLBACK_TIMEOUT_SEC:.0f}s），按释放失败处理", file=sys.stderr, flush=True)
        fb_out = str(getattr(fb_res, "stdout", "") or "")
        fb_err = str(getattr(fb_res, "stderr", "") or "")
        clean_fb = re.sub(r"<<<STOP_FALLBACK>>>.*?<<<END_STOP_FALLBACK>>>", "", fb_out, flags=re.S).strip()
        if clean_fb:
            print(redact_secrets(clean_fb), flush=True)
        if fb_err.strip():
            print(redact_secrets(fb_err.strip()), file=sys.stderr, flush=True)
        fb_pkt = extract_framed_json(fb_out, "STOP_FALLBACK")
        if fb_res is not None and getattr(fb_res, "returncode", 1) == 0:
            if fb_pkt is not None:
                if fb_pkt.get("acted"):
                    stop_ok = True
                else:
                    fb_unhandled_eps = [str(x) for x in (fb_pkt.get("active_eps") or []) if x]
            elif clean_all or target_ep:
                stop_ok = True

    if fb_unhandled_eps and not clean_all:
        raise RuntimeError(f"停止 Colab 会话 '{session}' 失败：本地会话记录已丢失且无绑定 Endpoint，云端仍有活跃计费实例 ({', '.join(fb_unhandled_eps)})，请立即核查或手动释放")
    if not stop_ok and not clean_all and not (verify_release and colab_py.is_file() and not target_ep):
        hint = "本地会话未找到（Session not found）" if stop_missing else (f"colab stop 超时 {STOP_CMD_TIMEOUT_SEC:.0f}s 且兜底释放未成功" if stop_timed_out else f"colab stop 退出码 {last_rc}")
        raise RuntimeError(f"停止 Colab 会话 '{session}' 失败（{hint}）")

    summary: dict[str, Any] = {"ok": True, "clean_all": bool(clean_all), "released_endpoint": target_ep, "remaining_count": 0}
    if verify_release and colab_py.is_file():
        verify_code = f"""
import json, time
from colab_cli.common import state
clean_all, target_ep, remaining = {bool(clean_all)!r}, {target_ep!r}, []
for attempt in range(1, 4):
    assignments = state.client.list_assignments()
    remaining = [ep for item in (assignments or []) if (ep := getattr(item, "endpoint", None) or (item.get("endpoint") if isinstance(item, dict) else None))]
    if (clean_all and not remaining) or ((not clean_all) and target_ep and target_ep not in remaining) or ((not clean_all) and not target_ep):
        break
    if attempt < 3: time.sleep(1.5)
print("<<<STOP_VERIFIED>>>" + json.dumps({{"remaining": remaining}}, ensure_ascii=False) + "<<<END_STOP_VERIFIED>>>", flush=True)
"""
        try:
            v_res = run_with_timeout(run_fn, [colab_py, "-c", verify_code], STOP_VERIFY_TIMEOUT_SEC)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"停卡后核实释放状态超时（{STOP_VERIFY_TIMEOUT_SEC:.0f}s），实例可能仍在计费，请立即手动核查") from exc
        pkt = extract_framed_json(getattr(v_res, "stdout", "") or "", "STOP_VERIFIED")
        if getattr(v_res, "returncode", 1) != 0 or pkt is None:
            raise RuntimeError("停卡后核实释放状态失败（无法查询当前活跃实例）")
        rem_list = list(pkt.get("remaining") or [])
        if clean_all and rem_list:
            raise RuntimeError(f"停卡核实失败：仍有 {len(rem_list)} 台实例未释放 ({', '.join(rem_list)})")
        if (not clean_all) and target_ep and (target_ep in rem_list):
            raise RuntimeError(f"停卡核实失败：目标实例 {target_ep} 仍未释放")
        if (not clean_all) and (not stop_ok) and (not target_ep) and rem_list:
            raise RuntimeError(f"停卡核实失败：本地会话 '{session}' 记录丢失/停卡未确认，且云端仍有活跃计费实例 ({', '.join(rem_list)})")
        if (not clean_all) and stop_ok and (not target_ep):
            summary["unverified"] = True
            if rem_list:
                print(f"[警告] 未锁定目标 Endpoint，无法精确核实会话 '{session}' 是否已释放；云端当前仍有 {len(rem_list)} 台活跃实例 ({', '.join(rem_list)})，若其中没有其他会话在用，请立即手动停卡", file=sys.stderr, flush=True)
        summary["remaining_count"] = len(rem_list)
        print("<<<STOP_VERIFIED>>>" + json.dumps(summary, ensure_ascii=False) + "<<<END_STOP_VERIFIED>>>", flush=True)

    if status_fn is not None:
        try:
            status_fn(None)
        except Exception as exc:
            print(f"[警告] 停卡后查询状态异常（实例已触发释放）: {redact_secrets(str(exc))}", flush=True)
    return summary
