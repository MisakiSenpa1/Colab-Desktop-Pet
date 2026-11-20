"""colab_core.colab_client —— Colab 远端执行、会话纳管、停卡核实与公网隧道探测模块。"""

from __future__ import annotations

import ast
import inspect
import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

_PKG_DIR = Path(__file__).resolve().parents[1]
if str(_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR))

from colab_core.accounts import CLI_BRIDGE_SCRIPT, load_cf_config  # noqa: E402
from colab_core.runtime import extract_framed_json  # noqa: E402
from colab_security import extract_valid_https_url, redact_secrets, urlopen_safe, url_origin_key  # noqa: E402

DEFAULT_QUICK_TUNNEL_HOST_PATTERN = r"(?!api\.)[a-z0-9-]+\.trycloudflare\.com"
DEFAULT_TUNNEL_URL_RE = re.compile(
    rf"https://{DEFAULT_QUICK_TUNNEL_HOST_PATTERN}(?![a-z0-9.-])"
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
    """校验并提取标准化的小写 HTTPS 主机名；非法格式返回 None。"""
    try:
        norm = extract_valid_https_url(url or "")
        return norm[8:] if norm else None
    except ValueError:
        return None


def _strict_https_origin(url: str | None) -> tuple[str, str, int] | None:
    """严格解析 HTTPS 源站，返回 `(origin_key, host, port)`；拒绝非 HTTPS、带 userinfo 或非法主机。"""
    cand = str(url or "").strip()
    if not cand or any(ch in cand for ch in (" ", "\t", "\r", "\n", "\x00")):
        return None
    try:
        parsed = urllib.parse.urlsplit(cand)
        port_raw = parsed.port
    except ValueError:
        return None
    if parsed.scheme.lower() != "https":
        return None
    if parsed.username is not None or parsed.password is not None:
        return None
    host = (parsed.hostname or "").strip().lower()
    if not host or extract_valid_https_host(f"https://{host}") != host:
        return None
    port = int(port_raw) if port_raw is not None else 443
    if not (1 <= port <= 65535):
        return None
    return f"https://{host}:{port}", host, port


def build_tunnel_url_pattern(
    cf_cfg: dict[str, Any] | None = None,
    *,
    cf_config_path: Path | None = None,
    colab_py: Path | None = None,
    extra_urls: Iterable[str] | None = None,
) -> str:
    """构造用于日志与远端输出文本提取的隧道 URL 正则字符串（合并官方 trycloudflare 与用户配置的固定域名）。

    注意：本函数仅用于从文本流提取候选公网 URL，严禁直接作为凭据发送源站授权依据；
    向目标源站附带 Bearer 密钥/凭据前必须调用 `is_authorized_credential_origin`。
    """
    branches = [DEFAULT_QUICK_TUNNEL_HOST_PATTERN]
    cfg = (
        cf_cfg
        if isinstance(cf_cfg, dict)
        else (
            load_cf_config(cf_config_path, include_secrets=False, colab_py=colab_py)
            if cf_config_path
            else None
        )
    )
    candidate_urls: list[str] = []
    if isinstance(cfg, dict):
        for key in ("comfy_url", "llm_url"):
            val = str(cfg.get(key) or "").strip()
            if val:
                candidate_urls.append(val)
    if extra_urls:
        for val in extra_urls:
            s = str(val or "").strip()
            if s:
                candidate_urls.append(s)
    for raw_u in candidate_urls:
        host = extract_valid_https_host(raw_u)
        if host and (esc := re.escape(host)) not in branches:
            branches.append(esc)
    return r"https://(?:" + "|".join(branches) + r")(?![a-z0-9.-])"


def is_authorized_credential_origin(
    url: str,
    *,
    config_path: Path | None = None,
    allowed_origins: Iterable[str] | None = None,
    cf_cfg: dict[str, Any] | None = None,
) -> bool:
    """严格判定是否允许向目标源站发送 Bearer 密钥/凭据。

    安全约束：
    1. 严格要求 `https://` 协议、合法域名主机且无 `userinfo`（禁止 `http://` 与 `@` 欺骗）；
    2. 仅允许精确匹配显式受信任源站（`allowed_origins`，如活跃端点绑定源站）或当前有效配置（`config_path` / `cf_cfg`）中的源站；
    3. 绝不使用宽泛正则（例如任意 `*.trycloudflare.com`）授权凭据外发，杜绝向陌生隧道误发密钥。
    """
    parsed_target = _strict_https_origin(url)
    if parsed_target is None:
        return False
    target_origin = parsed_target[0]

    trusted_origins: set[str] = set()
    if allowed_origins is not None:
        for raw_allowed in allowed_origins:
            parsed_ok = _strict_https_origin(raw_allowed)
            if parsed_ok is not None:
                trusted_origins.add(parsed_ok[0])

    if isinstance(cf_cfg, dict):
        for key in ("comfy_url", "llm_url"):
            parsed_cfg = _strict_https_origin(str(cf_cfg.get(key) or ""))
            if parsed_cfg is not None:
                trusted_origins.add(parsed_cfg[0])

    if config_path is not None:
        cfg = load_cf_config(config_path, include_secrets=False)
        for key in ("comfy_url", "llm_url"):
            parsed_saved = _strict_https_origin(str(cfg.get(key) or ""))
            if parsed_saved is not None:
                trusted_origins.add(parsed_saved[0])

    return target_origin in trusted_origins


def is_allowed_tunnel_url(
    url: str,
    *,
    config_path: Path | None = None,
    base_regex: re.Pattern[str] = DEFAULT_TUNNEL_URL_RE,
    allowed_origins: Iterable[str] | None = None,
) -> bool:
    """校验 URL 是否属于允许的隧道地址（向后兼容调用：支持精确授信源站或官方 trycloudflare 完整域名匹配）。"""
    parsed_target = _strict_https_origin(url)
    if parsed_target is None:
        return False
    _, host, port = parsed_target
    if is_authorized_credential_origin(url, config_path=config_path, allowed_origins=allowed_origins):
        return True
    if port == 443 and base_regex.fullmatch(f"https://{host}"):
        return True
    return False


def probe_comfy_endpoint_ok(
    url: str,
    comfy_key: str = "",
    timeout: float = 5.0,
    *,
    require_auth: bool = True,
    require_key: bool = False,
) -> tuple[bool, dict[str, Any]]:
    """探测 ComfyUI 公网端点健康与鉴权状态（默认严禁放行匿名返回 HTTP 200 的无保护公网入口）。"""
    base = str(url or "").strip().rstrip("/")
    exp_origin = url_origin_key(base)
    if not base or not exp_origin:
        return False, {"reason": "invalid_url"}

    unauth_rejected = False
    try:
        with urlopen_safe(
            urllib.request.Request(f"{base}/system_stats", headers={"User-Agent": "Mozilla/5.0"}),
            timeout=timeout,
            expected_origin=exp_origin,
        ) as resp:
            if resp.status == 200:
                body = json.loads(resp.read().decode("utf-8", errors="ignore") or "{}")
                if require_auth:
                    return False, {"auth_protected": False, "unprotected": True}
                return True, body if isinstance(body, dict) else {}
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            unauth_rejected = True
        else:
            raise

    if not unauth_rejected:
        return False, {}

    key_s = str(comfy_key or "").strip()
    if key_s:
        try:
            with urlopen_safe(
                urllib.request.Request(
                    f"{base}/system_stats",
                    headers={"User-Agent": "Mozilla/5.0", "Authorization": f"Bearer {key_s}"},
                ),
                timeout=timeout,
                expected_origin=exp_origin,
            ) as auth_resp:
                if auth_resp.status == 200:
                    body = json.loads(auth_resp.read().decode("utf-8", errors="ignore") or "{}")
                    if isinstance(body, dict) and ("system" in body or "devices" in body):
                        return True, {"auth_protected": True, **body}
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403):
                return False, {"auth_protected": True, "auth_failed": True, "status_code": int(exc.code)}
            raise
        return False, {}

    if require_key:
        return False, {"auth_protected": True, "missing_key": True}

    with urlopen_safe(
        urllib.request.Request(f"{base}/__pet_health", headers={"User-Agent": "Mozilla/5.0"}),
        timeout=timeout,
        expected_origin=exp_origin,
    ) as hr:
        if hr.status == 200:
            hb = json.loads(hr.read().decode("utf-8", errors="ignore") or "{}")
            if isinstance(hb, dict) and str(hb.get("status") or "").lower() == "ok" and hb.get("auth_required") is not False:
                return True, {"auth_protected": True, **hb}
    return False, {}


def probe_llm_endpoint_auth_ok(
    url: str,
    api_key: str,
    timeout: float = 5.0,
) -> tuple[bool, dict[str, Any]]:
    """对公网 LLM 端点执行双向鉴权探针：/health 200 ok + 匿名 /v1/models 401/403 + 合法 Bearer /v1/models 200。"""
    base = str(url or "").strip().rstrip("/")
    key = str(api_key or "").strip()
    exp_origin = url_origin_key(base)
    if not base or not exp_origin:
        return False, {"reason": "invalid_url"}
    if not key:
        return False, {"reason": "missing_api_key"}

    try:
        with urlopen_safe(
            urllib.request.Request(f"{base}/health", headers={"User-Agent": "Mozilla/5.0"}),
            timeout=timeout,
            expected_origin=exp_origin,
        ) as h_resp:
            if h_resp.status != 200:
                return False, {"reason": f"health_status_{h_resp.status}"}
            health_body = json.loads(h_resp.read().decode("utf-8", errors="ignore") or "{}")
            if not isinstance(health_body, dict) or str(health_body.get("status") or "").lower() != "ok":
                return False, {"reason": "health_body_invalid"}
    except Exception as exc:
        return False, {"reason": redact_secrets(str(exc) or type(exc).__name__)}

    try:
        with urlopen_safe(
            urllib.request.Request(f"{base}/v1/models", headers={"User-Agent": "Mozilla/5.0"}),
            timeout=timeout,
            expected_origin=exp_origin,
        ) as unauth_resp:
            return False, {
                "auth_protected": False,
                "reason": f"unauthenticated_models_http_{unauth_resp.status}",
            }
    except urllib.error.HTTPError as exc:
        if exc.code not in (401, 403):
            return False, {"reason": f"unexpected_unauth_status_{exc.code}"}
    except Exception as exc:
        return False, {"reason": redact_secrets(str(exc) or type(exc).__name__)}

    try:
        with urlopen_safe(
            urllib.request.Request(
                f"{base}/v1/models",
                headers={"User-Agent": "Mozilla/5.0", "Authorization": f"Bearer {key}"},
            ),
            timeout=timeout,
            expected_origin=exp_origin,
        ) as auth_resp:
            if auth_resp.status != 200:
                return False, {"auth_protected": True, "reason": f"auth_models_status_{auth_resp.status}"}
            models_body = json.loads(auth_resp.read().decode("utf-8", errors="ignore") or "{}")
            if not isinstance(models_body, dict) or ("data" not in models_body and "object" not in models_body):
                return False, {"auth_protected": True, "reason": "models_body_invalid"}
            return True, {"auth_protected": True, "health": health_body, "models": models_body}
    except urllib.error.HTTPError as exc:
        return False, {"auth_protected": True, "reason": f"auth_rejected_{exc.code}"}
    except Exception as exc:
        return False, {"reason": redact_secrets(str(exc) or type(exc).__name__)}


def probe_tunnel_alive(
    url: str,
    timeout: float = 3.5,
    kind: str = "comfy",
    auth_token: str = "",
    cf_config_path: Path | None = None,
    allowed_checker: Callable[[str], bool] | None = None,
) -> bool:
    """探测隧道是否存活；对 ComfyUI 拒绝放行匿名 200 裸奔入口，对 LLM 在提供密钥时执行双向鉴权校验。"""
    if not (allowed_checker(url) if allowed_checker else is_allowed_tunnel_url(url, config_path=cf_config_path)):
        return False
    base = url.strip().rstrip("/")
    mode = str(kind or "comfy").strip().lower()

    def _probe_single(k: str) -> bool:
        if k == "llm":
            tok_s = str(auth_token or "").strip()
            if tok_s:
                ok_llm, _ = probe_llm_endpoint_auth_ok(base, tok_s, timeout=timeout)
                return ok_llm
            try:
                with urlopen_safe(
                    urllib.request.Request(f"{base}/health", headers={"User-Agent": "Mozilla/5.0"}),
                    timeout=timeout,
                    expected_origin=url_origin_key(base),
                ) as resp:
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
    """递归收集入口脚本依赖的同级 .py/.js 与已知安全伴随 .json 资产文件。"""
    base_dir = entry_script.resolve().parent
    pkg_resolved = _PKG_DIR.resolve()
    drama3_wf_dir = (pkg_resolved / "workflows" / "drama3").resolve()
    local_drama3_wf_dir = (base_dir / "workflows" / "drama3").resolve()
    # 仅允许随行收集已知安全的 Drama 3 伴随工作流，严防误传本地配置或状态文件
    allowed_drama3_json = {"drama3_voice_clone.json", "drama3_text_to_speech.json"}
    allowed_json_literals = allowed_drama3_json | {f"workflows/drama3/{name}" for name in allowed_drama3_json}

    deps: dict[str, str] = {}
    visited: set[Path] = {entry_script.resolve()}
    queue: list[Path] = [entry_script.resolve()]
    total_bytes = 0

    def _enqueue(cand: Path) -> None:
        resolved = cand.resolve()
        if resolved not in visited and resolved.parent in (base_dir, pkg_resolved) and resolved.is_file() and resolved.suffix in (".py", ".js"):
            visited.add(resolved)
            queue.append(resolved)

    def _enqueue_name(fname: str) -> None:
        _enqueue(base_dir / fname)
        if not (base_dir / fname).is_file():
            _enqueue(pkg_resolved / fname)

    def _enqueue_json_companion(lit: str) -> None:
        fname = Path(lit).name
        if fname not in allowed_drama3_json:
            return
        candidates = (
            base_dir / fname,
            local_drama3_wf_dir / fname,
            drama3_wf_dir / fname,
        )
        allowed_parents = {base_dir, local_drama3_wf_dir, drama3_wf_dir}
        for cand in candidates:
            resolved = cand.resolve()
            if (
                resolved not in visited
                and resolved.parent in allowed_parents
                and resolved.is_file()
                and resolved.suffix == ".json"
                and resolved.name == fname
            ):
                visited.add(resolved)
                queue.append(resolved)
                return

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
        if cur.suffix != ".py":
            continue
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
                if val.endswith((".py", ".js")) and "/" not in val and "\\" not in val and len(val) <= 128:
                    _enqueue_name(val)
                elif val in allowed_json_literals:
                    _enqueue_json_companion(val)
    return deps


def _resolve_effective_account_id(account_id: str | None = None) -> str:
    """解析有效账号作用域：显式参数优先，否则回退读取环境变量 COLAB_PET_ACCOUNT_ID。"""
    raw = account_id if account_id is not None else os.environ.get("COLAB_PET_ACCOUNT_ID")
    return str(raw or "").strip()


def build_scoped_colab_cmd(
    colab_exe: Path,
    colab_py: Path,
    args: list[Any],
    *,
    account_id: str | None = None,
) -> list[Any]:
    """当账号作用域非空时路由至 scoped-cli；若缺少隔离运行组件则严格失败关闭，严禁退回默认账号。"""
    eff_acct = _resolve_effective_account_id(account_id)
    if eff_acct:
        if not CLI_BRIDGE_SCRIPT.is_file() or not colab_py.is_file():
            raise FileNotFoundError(
                f"缺少多账号隔离运行组件 (bridge={CLI_BRIDGE_SCRIPT.is_file()}, py={colab_py.is_file()})，"
                "拒绝在未隔离状态下执行 Colab 指令"
            )
        return [colab_py, CLI_BRIDGE_SCRIPT, "scoped-cli", f"--acct={eff_acct}", "--", *args]
    return [colab_exe, *args]


def _scoped_state_bind_prefix(account_id: str) -> str:
    """生成在 colab_py 子脚本中绑定隔离账号状态的引导代码片段；账号非空但缺桥接脚本时拒绝回退。"""
    if not account_id:
        return ""
    if not CLI_BRIDGE_SCRIPT.is_file():
        raise FileNotFoundError(
            f"未找到多账号状态桥接脚本 ({CLI_BRIDGE_SCRIPT})，拒绝在未隔离状态下操作账号 '{account_id}'"
        )
    bridge_dir = str(CLI_BRIDGE_SCRIPT.parent)
    return (
        "import sys as _sys\n"
        f"if {bridge_dir!r} not in _sys.path:\n"
        f"    _sys.path.insert(0, {bridge_dir!r})\n"
        "from colab_cli_bridge import _bind_account_state as _colab_bind_acct\n"
        f"_colab_bind_acct({account_id!r})\n"
    )


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
    account_id: str | None = None,
):
    if not colab_py.is_file():
        raise FileNotFoundError(f"未找到google-colab-cli python: {colab_py}")
    if not script_path.is_file():
        raise FileNotFoundError(f"未找到待执行脚本 {script_path.name}: {script_path}")

    eff_acct = _resolve_effective_account_id(account_id)
    merged_extra = collect_local_py_deps(script_path)
    if extra_files:
        merged_extra.update(extra_files)
    runner_code = _scoped_state_bind_prefix(eff_acct) + (
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
    """严格解析指定会话对应的目标 Endpoint（禁止找不到会话时跨名回退选中其他活跃实例）。"""
    explicit_ep = str(endpoint or "").strip()
    wanted_sess = str(session or "comfy").strip() or "comfy"
    if explicit_ep:
        matched = next((s for s in sessions if str(s.get("endpoint") or "") == explicit_ep), None)
        return (str(matched["name"]) if matched and matched.get("name") not in (None, "?") else wanted_sess), explicit_ep, False
    named_hit = next((s for s in sessions if str(s.get("name") or "") == wanted_sess and s.get("endpoint")), None)
    if named_hit:
        return str(named_hit["name"]), str(named_hit["endpoint"]), False
    if (
        isinstance(binding, dict)
        and str(binding.get("endpoint") or "").strip()
        and str(binding.get("account_id") or "").strip() == str(active_account_id or "").strip()
        and str(binding.get("session") or "comfy").strip() == wanted_sess
    ):
        b_ep = str(binding["endpoint"]).strip()
        if not sessions or any(str(s.get("endpoint") or "") == b_ep for s in sessions):
            return wanted_sess, b_ep, False
    if sessions and all(s.get("orphan") or s.get("name") in (None, "?") for s in sessions):
        return wanted_sess, "", True
    return wanted_sess, "", False


def _session_name_in_cli_output(sessions_out: str, session: str) -> bool:
    if not sessions_out or "No active sessions" in sessions_out:
        return False
    return bool(re.search(rf"\[{re.escape(session)}\]", sessions_out) or re.search(rf"(?<![A-Za-z0-9_.-]){re.escape(session)}(?![A-Za-z0-9_.-])", sessions_out))


def _try_adopt_orphan_session(
    *,
    session: str,
    target_endpoint: str | None,
    preferred_gpu: str,
    colab_py: Path,
    run_fn: Callable[..., Any],
    account_id: str | None = None,
) -> dict[str, Any] | None:
    if not colab_py.is_file():
        return None
    eff_acct = _resolve_effective_account_id(account_id)
    payload = json.dumps({"session": session, "target_endpoint": str(target_endpoint or "").strip(), "preferred_gpu": str(preferred_gpu or "").strip().upper()}, ensure_ascii=False)
    adopt_code = _scoped_state_bind_prefix(eff_acct) + f"import io, sys\nsys.stdin = io.StringIO({payload!r})\n" + COLAB_ADOPT_ORPHAN_SCRIPT
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
    account_id: str | None = None,
) -> bool:
    eff_acct = _resolve_effective_account_id(account_id)
    sessions_cmd = build_scoped_colab_cmd(colab_exe, colab_py, ["sessions"], account_id=eff_acct)
    sessions_res = run_fn(sessions_cmd, check=False, capture=True)
    sessions_out = getattr(sessions_res, "stdout", "") or ""
    has_session = getattr(sessions_res, "returncode", 1) == 0 and _session_name_in_cli_output(sessions_out, session)
    target_ep_arg = str(target_endpoint or "").strip() or None

    if recreate and (has_session or target_ep_arg):
        print("=== 正在定向释放目标会话旧实例以重新分配显卡（同账号其余会话不受影响） ===", flush=True)
        try:
            stop_accepts_acct = any(
                p.name == "account_id" or p.kind is inspect.Parameter.VAR_KEYWORD
                for p in inspect.signature(stop_fn).parameters.values()
            )
        except (TypeError, ValueError):
            stop_accepts_acct = False
        if eff_acct and stop_accepts_acct:
            stop_fn(session, clean_all=False, target_endpoint=target_ep_arg, verify_release=True, account_id=eff_acct)
        else:
            stop_fn(session, clean_all=False, target_endpoint=target_ep_arg, verify_release=True)
        has_session = False

    if not has_session and not recreate and (target_ep_arg or "[?]" in sessions_out):
        adopted = _try_adopt_orphan_session(session=session, target_endpoint=target_ep_arg, preferred_gpu=gpu, colab_py=colab_py, run_fn=run_fn, account_id=eff_acct)
        if adopted:
            print(f"检测到已有同规格活跃实例 ({adopted.get('endpoint') or ''} · {adopted.get('accelerator') or gpu})，已自动纳管为会话 {session}，跳过 colab new，继续增量部署...", flush=True)
            return True

    if has_session:
        print(f"检测到会话 {session} 已存在，跳过 colab new，继续增量部署...", flush=True)
        return True

    print(f"=== 开启 Colab 实例 (session={session}, gpu={gpu}, high_mem={high_mem}) ===", flush=True)
    new_args: list[Any] = ["new", "-s", session, "--gpu", gpu] + (["--high-mem"] if high_mem else [])
    new_cmd = build_scoped_colab_cmd(colab_exe, colab_py, new_args, account_id=eff_acct)
    try:
        run_fn(new_cmd)
        return False
    except Exception:
        adopted_after = _try_adopt_orphan_session(session=session, target_endpoint=target_ep_arg, preferred_gpu=gpu, colab_py=colab_py, run_fn=run_fn, account_id=eff_acct)
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


def probe_session_endpoint(
    session: str,
    *,
    colab_py: Path,
    run_fn: Callable[..., Any],
    timeout: float | None = STOP_PROBE_TIMEOUT_SEC,
    account_id: str | None = None,
) -> str | None:
    if not colab_py.is_file():
        print(f"[警告] 未找到 Colab CLI Python 解释器 ({colab_py})，无法查询会话 '{session}' 的 Endpoint", file=sys.stderr, flush=True)
        return None
    eff_acct = _resolve_effective_account_id(account_id)
    probe_ep_code = _scoped_state_bind_prefix(eff_acct) + f"from colab_cli.common import state\ns = state.get_session({session!r}, ignore_missing_session=True)\nprint(s.endpoint if s and getattr(s, 'endpoint', None) else '')\n"
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


def _validate_and_resolve_stop_target(
    *,
    session: str,
    clean_all: bool,
    target_endpoint: str | None,
    verify_release: bool,
    colab_exe: Path,
    colab_py: Path,
    run_fn: Callable[..., Any],
    account_id: str | None,
) -> tuple[str, str | None, str | None]:
    """校验停卡前置运行环境并解析 `(eff_acct, explicit_target_ep, target_ep)`。"""
    if not colab_exe.is_file():
        raise FileNotFoundError(f"未找到colab.exe: {colab_exe}")

    eff_acct = _resolve_effective_account_id(account_id)
    explicit_target_ep = str(target_endpoint or "").strip() or None
    if (explicit_target_ep or eff_acct or clean_all or verify_release) and not colab_py.is_file():
        raise FileNotFoundError(
            f"未找到 Colab Python 运行环境 ({colab_py})，无法核实目标实例或释放状态，已拒绝停卡"
        )
    if eff_acct and not CLI_BRIDGE_SCRIPT.is_file():
        raise FileNotFoundError(
            f"未找到账号桥接脚本 ({CLI_BRIDGE_SCRIPT})，无法核实目标账号作用域，已拒绝停卡"
        )

    target_ep = explicit_target_ep
    if not target_ep and not clean_all:
        target_ep = probe_session_endpoint(session, colab_py=colab_py, run_fn=run_fn, account_id=eff_acct)
        if not target_ep:
            print(f"[警告] 未能预先锁定会话 '{session}' 的目标实例 Endpoint，停卡结果将无法精确核实", file=sys.stderr, flush=True)
    return eff_acct, explicit_target_ep, target_ep


def _run_cli_session_stop(
    *,
    session: str,
    colab_exe: Path,
    colab_py: Path,
    eff_acct: str,
    run_fn: Callable[..., Any],
) -> tuple[bool, bool, bool, int]:
    """按会话名执行最多 3 次 `colab stop -s <session>`，返回 `(stop_ok, stop_missing, stop_timed_out, last_rc)`。"""
    import time

    stop_cmd = build_scoped_colab_cmd(colab_exe, colab_py, ["stop", "-s", session], account_id=eff_acct)
    last_rc = 1
    for attempt in range(1, 4):
        try:
            res = run_with_timeout(run_fn, stop_cmd, STOP_CMD_TIMEOUT_SEC)
        except subprocess.TimeoutExpired:
            print(
                f"[警告] colab stop -s {session} 超时（{STOP_CMD_TIMEOUT_SEC:.0f}s），立即转入云端 Endpoint 兜底释放...",
                file=sys.stderr,
                flush=True,
            )
            return False, False, True, -1
        last_rc = int(getattr(res, "returncode", 1))
        stop_out, stop_err = str(getattr(res, "stdout", "") or ""), str(getattr(res, "stderr", "") or "")
        if stop_out:
            print(redact_secrets(stop_out), end="" if stop_out.endswith("\n") else "\n", flush=True)
        if stop_err:
            print(redact_secrets(stop_err), file=sys.stderr, end="" if stop_err.endswith("\n") else "\n", flush=True)
        combined_low = f"{stop_out}\n{stop_err}".lower()
        if "not found" in combined_low and ("session" in combined_low or session.lower() in combined_low):
            print(f"提示: colab stop -s {session} 返回会话未找到，立即转入云端 Endpoint 兜底核查...", flush=True)
            return False, True, False, last_rc
        if last_rc == 0:
            return True, False, False, 0
        print(f"提示: 第 {attempt}/3 次 colab stop -s {session} 返回 {last_rc}，准备重试/兜底释放...", flush=True)
        if attempt < 3:
            time.sleep(2)
    return False, False, False, last_rc


def _build_stop_fallback_script(session: str, target_ep: str | None, clean_all: bool, eff_acct: str) -> str:
    """构造云端 Endpoint 兜底释放短脚本（严格按目标 Endpoint 清理本地会话记录防误杀）。"""
    return _scoped_state_bind_prefix(eff_acct) + f"""
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
                # 仅清理绑定至 target_ep 的会话记录，绝不误删指向同账号其他实例的同名记录
                store_map = state.store.list() or {{}}
                for sn, s_val in list(store_map.items()):
                    if isinstance(s_val, dict):
                        sn_ep = s_val.get("endpoint")
                    elif isinstance(s_val, str):
                        sn_ep = s_val
                    else:
                        sn_ep = getattr(s_val, "endpoint", None)
                    sn_ep_s = str(sn_ep or "").strip()
                    if sn_ep_s == target_ep or (sn == session_name and not sn_ep_s):
                        state.store.remove(sn)
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


def _run_cloud_stop_fallback(
    *,
    session: str,
    target_ep: str | None,
    clean_all: bool,
    colab_py: Path,
    eff_acct: str,
    run_fn: Callable[..., Any],
) -> tuple[bool, list[str]]:
    """执行云端 Endpoint 兜底释放，返回 `(fb_acted, fb_unhandled_eps)`。"""
    fallback_code = _build_stop_fallback_script(session, target_ep, clean_all, eff_acct)
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
                return True, []
            return False, [str(x) for x in (fb_pkt.get("active_eps") or []) if x]
        if clean_all or target_ep:
            return True, []
    return False, []


def _verify_stop_release_state(
    *,
    session: str,
    target_ep: str | None,
    clean_all: bool,
    stop_ok: bool,
    colab_py: Path,
    eff_acct: str,
    run_fn: Callable[..., Any],
    summary: dict[str, Any],
) -> None:
    """停卡后复查云端活跃实例列表，确认目标实例已彻底释放（Fail-Closed）。"""
    verify_code = _scoped_state_bind_prefix(eff_acct) + f"""
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
            print(
                f"[警告] 未锁定目标 Endpoint，无法精确核实会话 '{session}' 是否已释放；云端当前仍有 {len(rem_list)} 台活跃实例 ({', '.join(rem_list)})，若其中没有其他会话在用，请立即手动停卡",
                file=sys.stderr,
                flush=True,
            )
    summary["remaining_count"] = len(rem_list)
    print("<<<STOP_VERIFIED>>>" + json.dumps(summary, ensure_ascii=False) + "<<<END_STOP_VERIFIED>>>", flush=True)


def _format_stop_failure_hint(
    *,
    skip_blind_session_stop: bool,
    explicit_target_ep: str | None,
    stop_missing: bool,
    stop_timed_out: bool,
    last_rc: int,
) -> str:
    """按停卡失败场景生成可读的故障原因摘要。"""
    if skip_blind_session_stop:
        return f"精确释放目标实例 {explicit_target_ep} 未成功"
    if stop_missing:
        return "本地会话未找到（Session not found）"
    if stop_timed_out:
        return f"colab stop 超时 {STOP_CMD_TIMEOUT_SEC:.0f}s 且兜底释放未成功"
    return f"colab stop 退出码 {last_rc}"


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
    account_id: str | None = None,
) -> dict[str, Any]:
    """停止指定 Colab 会话或全部活跃实例，并按需执行云端 Endpoint 兜底释放与释放核实。"""
    eff_acct, explicit_target_ep, target_ep = _validate_and_resolve_stop_target(
        session=session, clean_all=clean_all, target_endpoint=target_endpoint, verify_release=verify_release,
        colab_exe=colab_exe, colab_py=colab_py, run_fn=run_fn, account_id=account_id,
    )

    stop_ok = stop_missing = stop_timed_out = False
    last_rc = 1
    # 显式指定 target_endpoint 时严禁按会话名执行 colab stop -s <session> 盲停，防止误杀同账号另一台活跃实例
    skip_blind_session_stop = bool(explicit_target_ep and not clean_all)
    if skip_blind_session_stop:
        print(f"已显式指定目标实例 Endpoint ({explicit_target_ep})，跳过按会话名 '{session}' 盲停，直接执行精确 Endpoint 释放...", flush=True)
    else:
        stop_ok, stop_missing, stop_timed_out, last_rc = _run_cli_session_stop(
            session=session, colab_exe=colab_exe, colab_py=colab_py, eff_acct=eff_acct, run_fn=run_fn,
        )

    fb_unhandled_eps: list[str] = []
    if colab_py.is_file() and (clean_all or not stop_ok):
        fb_acted, fb_unhandled_eps = _run_cloud_stop_fallback(
            session=session, target_ep=target_ep, clean_all=clean_all, colab_py=colab_py, eff_acct=eff_acct, run_fn=run_fn,
        )
        if fb_acted:
            stop_ok = True

    if fb_unhandled_eps and not clean_all:
        raise RuntimeError(
            f"停止 Colab 会话 '{session}' 失败：本地会话记录已丢失且无绑定 Endpoint，云端仍有活跃计费实例 ({', '.join(fb_unhandled_eps)})，请立即核查或手动释放"
        )
    if not stop_ok and not clean_all and not (verify_release and colab_py.is_file() and not target_ep):
        hint = _format_stop_failure_hint(
            skip_blind_session_stop=skip_blind_session_stop,
            explicit_target_ep=explicit_target_ep,
            stop_missing=stop_missing,
            stop_timed_out=stop_timed_out,
            last_rc=last_rc,
        )
        raise RuntimeError(f"停止 Colab 会话 '{session}' 失败（{hint}）")

    summary: dict[str, Any] = {"ok": True, "clean_all": bool(clean_all), "released_endpoint": target_ep, "remaining_count": 0}
    if verify_release and colab_py.is_file():
        _verify_stop_release_state(
            session=session, target_ep=target_ep, clean_all=clean_all, stop_ok=stop_ok,
            colab_py=colab_py, eff_acct=eff_acct, run_fn=run_fn, summary=summary,
        )

    if status_fn is not None:
        try:
            status_fn(None)
        except Exception as exc:
            print(f"[警告] 停卡后查询状态异常（实例已触发释放）: {redact_secrets(str(exc))}", flush=True)
    return summary
