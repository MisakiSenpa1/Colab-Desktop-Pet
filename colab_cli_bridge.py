"""colab_cli_bridge.py —— 运行在 `google-colab-cli` Python 环境内的统一桥接助手。

收拢原散落于多处模块中的 `colab_py -c` 内嵌脚本，共用实例定位、Contents REST 读写、
邮箱脱敏、私钥文件原子写与多账号 OAuth/保险箱操作，消除字符串套字符串的重复维护。
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import sys
import time
from importlib import resources
from pathlib import Path
from urllib.parse import urljoin

import keyring
import requests
from google.auth.transport.requests import Request as GoogleAuthRequest
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from colab_cli import auth as colab_auth
from colab_cli.common import state
from colab_cli.runtime import ColabRuntime

VAULT_SERVICE_NAME = "google-colab-cli-pet-vault"
CFG_DIR = Path(os.path.expanduser("~/.config/colab-cli"))
SESS_FILE = CFG_DIR / "sessions.json"
SESS_BACKUP_DIR = CFG_DIR / "pet_sessions_by_acct"
DEFAULT_TUNNEL_PATTERN = r"https://(?:(?!api\.)[a-z0-9-]+\.trycloudflare\.com)"


def _mask_email(email: str) -> str:
    raw = str(email or "").strip()
    if "@" not in raw:
        return "***"
    local, dom = raw.split("@", 1)
    if len(local) <= 3:
        m_local = local[:1] + "***"
    elif len(local) <= 6:
        m_local = local[:2] + "***" + local[-1:]
    else:
        m_local = local[:3] + "***" + local[-2:]
    return f"{m_local}@{dom}"


def _extract_email(creds: Credentials) -> str:
    try:
        r = requests.get(
            "https://www.googleapis.com/oauth2/v2/userinfo",
            headers={"Authorization": f"Bearer {creds.token}"},
            timeout=8,
        )
        if r.status_code == 200:
            em = str((r.json() or {}).get("email") or "").strip().lower()
            if em and "@" in em:
                return em
    except Exception:
        pass
    try:
        id_tok = getattr(creds, "id_token", None)
        if isinstance(id_tok, str) and id_tok.count(".") >= 2:
            seg = id_tok.split(".")[1]
            seg += "=" * (-len(seg) % 4)
            claims = json.loads(base64.urlsafe_b64decode(seg.encode("ascii")).decode("utf-8", errors="ignore"))
            em = str((claims or {}).get("email") or "").strip().lower()
            if em and "@" in em:
                return em
    except Exception:
        pass
    return ""


def _write_private_text(path_obj: Path, content: str) -> None:
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    tmp_p = path_obj.with_suffix(".tmp")
    fd = os.open(str(tmp_p), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fp:
        fp.write(content)
    for p in (tmp_p, path_obj):
        try:
            os.chmod(str(p), 0o600)
        except OSError:
            pass
        if p is tmp_p:
            tmp_p.replace(path_obj)


def _backup_sessions(acct_id: str) -> None:
    if acct_id and SESS_FILE.is_file():
        SESS_BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        _write_private_text(SESS_BACKUP_DIR / f"{acct_id}.json", SESS_FILE.read_text(encoding="utf-8", errors="ignore"))


def _restore_sessions(acct_id: str) -> None:
    SESS_BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    src = SESS_BACKUP_DIR / f"{acct_id}.json" if acct_id else None
    if src is not None and src.is_file():
        _write_private_text(SESS_FILE, src.read_text(encoding="utf-8", errors="ignore"))
    else:
        _write_private_text(SESS_FILE, "{}\n")


def _load_oauth_client_config() -> dict:
    cfg_path = os.path.expanduser("~/.colab-cli-oauth-config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as fp:
            return json.load(fp)
    cfg_res = resources.files("colab_cli").joinpath("oauth_config.json")
    if cfg_res.is_file():
        return json.loads(cfg_res.read_text(encoding="utf-8"))
    raise FileNotFoundError("未找到 colab-cli OAuth 客户端配置")


def _pick_target_assignment(target_ep: str = "", session_name: str = "comfy"):
    local_sessions = state.store.list()
    assignments = state.client.list_assignments()
    if not assignments:
        return None, local_sessions
    target = next((a for a in assignments if target_ep and a.endpoint == target_ep), None)
    if target is None and session_name in local_sessions:
        wanted = local_sessions[session_name].endpoint
        target = next((a for a in assignments if a.endpoint == wanted), None)
    if target is None:
        target = assignments[0]
    return target, local_sessions


def _decode_contents(data: dict):
    fmt = data.get("format")
    raw = data.get("content", "")
    if isinstance(raw, dict):
        return raw
    if fmt == "base64" and isinstance(raw, str):
        return base64.b64decode(raw).decode("utf-8", errors="ignore")
    return str(raw or "")


def _contents_get(sess: requests.Session, base: str, tok: str, rel: str, content: str = "1", timeout: float = 8.0):
    return sess.get(
        f"{base}/api/contents/{rel.lstrip('/')}",
        params={"authuser": "0", "colab-runtime-proxy-token": tok, "content": content},
        timeout=timeout,
    )


def _retry(fn, attempts: int = 3, delay: float = 1.0):
    last_exc: Exception | None = None
    for idx in range(attempts):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if idx + 1 < attempts:
                time.sleep(delay)
    assert last_exc is not None
    raise last_exc


def cmd_probe(args: list[str]) -> int:
    need_userinfo = "--need-userinfo" in args
    active_acct_id = ""
    for arg in args:
        if arg.startswith("--active-acct="):
            active_acct_id = arg.split("=", 1)[1].strip()

    creds = None
    auth_expired = False
    try:
        tok_json = colab_auth._get_token_from_keyring()
        if tok_json:
            creds = Credentials.from_authorized_user_info(json.loads(tok_json), colab_auth.PUBLIC_SCOPES)
    except Exception:
        creds = None

    if creds is None and os.path.exists(colab_auth.LEGACY_TOKEN_PATH):
        try:
            creds = Credentials.from_authorized_user_file(colab_auth.LEGACY_TOKEN_PATH, colab_auth.PUBLIC_SCOPES)
            colab_auth._save_token_to_keyring(creds.to_json())
            colab_auth._remove_legacy_token()
        except Exception:
            creds = None

    if creds is not None and not creds.valid:
        if creds.expired and creds.refresh_token:
            try:
                creds.refresh(GoogleAuthRequest())
                new_json = creds.to_json()
                colab_auth._save_token_to_keyring(new_json)
                if active_acct_id:
                    try:
                        keyring.set_password(VAULT_SERVICE_NAME, f"acct:{active_acct_id}", new_json)
                    except Exception:
                        pass
            except Exception:
                creds = None
                auth_expired = True
        else:
            creds = None
            auth_expired = True

    if creds is None:
        print("<<<COLAB_PET_JSON>>>" + json.dumps({"auth_missing": True, "auth_expired": bool(auth_expired)}, ensure_ascii=False) + "<<<END_COLAB_PET_JSON>>>", flush=True)
        return 0

    account_meta = None
    if need_userinfo:
        email = _extract_email(creds)
        if email:
            acct_id = hashlib.sha256(email.encode("utf-8")).hexdigest()[:12]
            try:
                keyring.set_password(VAULT_SERVICE_NAME, f"acct:{acct_id}", creds.to_json())
            except Exception:
                pass
            account_meta = {"id": acct_id, "email_masked": _mask_email(email), "prefix": email.split("@", 1)[0][:10]}

    local_sessions = state.store.list()
    client = state.client
    eligible_gpus, eligible_tpus, balance, rate_hourly, assignments_count = None, None, None, 0.0, 0
    try:
        def _fetch_ccu():
            r = client.session.get(
                urljoin(client.colab_domain, "/tun/m/ccu-info"),
                params={"authuser": "0"},
                headers={"Accept": "application/json", "X-Colab-Client-Agent": "colab-cli"},
                timeout=15,
            )
            r.raise_for_status()
            body = client._strip_xssi_prefix(r.text or "")
            d = json.loads(body) if body else {}
            return d if isinstance(d, dict) else {}

        raw_ccu = _retry(_fetch_ccu)
        bal_val = raw_ccu.get("currentBalance", raw_ccu.get("paidComputeUnitsBalance"))
        if bal_val is not None:
            balance = float(bal_val)
            rate_hourly = float(raw_ccu.get("consumptionRateHourly") or 0.0)
            assignments_count = int(raw_ccu.get("assignmentsCount") or 0)
        if isinstance(raw_ccu.get("eligibleGpus"), list):
            eligible_gpus = [str(x).upper() for x in raw_ccu["eligibleGpus"] if x]
        if isinstance(raw_ccu.get("eligibleTpus"), list):
            eligible_tpus = [str(x).upper() for x in raw_ccu["eligibleTpus"] if x]
    except Exception:
        pass

    if balance is None:
        info = _retry(client.get_consumption_user_info)
        balance = float(info.paid_compute_units_balance)
        rate_hourly = float(info.consumption_rate_hourly)
        assignments_count = int(info.assignments_count)

    assignments = _retry(client.list_assignments)
    name_by_ep = {s.endpoint: s.name for s in local_sessions.values()}
    items = []
    for a in assignments:
        ep = a.endpoint
        sname = name_by_ep.get(ep, "?")
        accel_obj = getattr(a, "accelerator", None)
        accel = accel_obj.value if accel_obj else "CPU"
        if accel == "NONE":
            accel = "CPU"
        variant_obj = getattr(a, "variant", None)
        shape_obj = getattr(a, "machine_shape", None)
        items.append({
            "name": sname,
            "endpoint": ep,
            "accelerator": accel,
            "variant": variant_obj.name if variant_obj else "DEFAULT",
            "machine_shape": getattr(shape_obj, "name", "STANDARD") if shape_obj is not None else "STANDARD",
            "orphan": sname == "?",
        })

    payload = {
        "auth_missing": False,
        "account_meta": account_meta,
        "balance": float(balance),
        "rate_hourly": float(rate_hourly),
        "assignments_count": int(assignments_count),
        "eligible_gpus": eligible_gpus,
        "eligible_tpus": eligible_tpus,
        "sessions": items,
    }
    print("<<<COLAB_PET_JSON>>>" + json.dumps(payload, ensure_ascii=False) + "<<<END_COLAB_PET_JSON>>>", flush=True)
    return 0


def cmd_fast_tunnel(args: list[str]) -> int:
    target_ep = args[0] if len(args) > 0 else ""
    session_name = args[1] if len(args) > 1 else "comfy"
    pat_str = args[2] if len(args) > 2 and args[2] else DEFAULT_TUNNEL_PATTERN
    tun_re = re.compile(pat_str)

    target, _ = _pick_target_assignment(target_ep, session_name)
    if target is None:
        print("<<<COLAB_TUNNEL>>>" + json.dumps({"status": "no_session"}, ensure_ascii=False) + "<<<END_COLAB_TUNNEL>>>", flush=True)
        return 0

    info = target.runtime_proxy_info
    base, tok = info.url.rstrip("/"), info.token
    sess = requests.Session()

    def _read_text(rel: str) -> tuple[int, object]:
        r = _contents_get(sess, base, tok, rel, content="1", timeout=8)
        if r.status_code != 200:
            return r.status_code, ""
        return 200, _decode_contents(r.json())

    def _read_url_file(rel: str) -> str | None:
        st, txt = _read_text(rel)
        if st != 200:
            return None
        m = tun_re.search(str(txt or ""))
        return m.group(0) if m else None

    out = None
    st_m, dec_m = _read_text("root/colab_service_meta.json")
    if st_m == 200:
        try:
            meta_obj = dec_m if isinstance(dec_m, dict) else json.loads(str(dec_m))
        except Exception:
            meta_obj = None
        if isinstance(meta_obj, dict):
            api_key = str(meta_obj.get("api_key") or "").strip()
            comfy_api_key = str(meta_obj.get("comfy_api_key") or "").strip()
            clean_meta = {k: v for k, v in meta_obj.items() if k not in ("api_key", "comfy_api_key")}
            kind_raw = str(clean_meta.get("kind") or "llm").strip().lower()
            if kind_raw not in ("comfy", "llm", "dual"):
                kind_raw = "llm"
            clean_meta["kind"] = kind_raw
            clean_meta.setdefault("port", 18080 if kind_raw == "llm" else 8188)

            if kind_raw == "dual":
                m_cu = tun_re.search(str(meta_obj.get("comfy_tunnel_url") or meta_obj.get("tunnel_url") or ""))
                comfy_u = (m_cu.group(0) if m_cu else None) or _read_url_file("root/comfy_tunnel_url.txt")
                m_lu = tun_re.search(str(meta_obj.get("llm_tunnel_url") or ""))
                llm_u = (m_lu.group(0) if m_lu else None) or _read_url_file("root/llm_tunnel_url.txt")
                if comfy_u:
                    clean_meta["comfy_tunnel_url"] = comfy_u
                if llm_u:
                    clean_meta["llm_tunnel_url"] = llm_u
                tun_url = comfy_u or llm_u
            else:
                m_u = tun_re.search(str(meta_obj.get("tunnel_url") or ""))
                tun_url = (m_u.group(0) if m_u else None) or _read_url_file("root/comfy_tunnel_url.txt")

            if tun_url:
                clean_meta["tunnel_url"] = tun_url
            out = {
                "status": "ok" if tun_url else "missing",
                "endpoint": target.endpoint,
                **({"url": tun_url} if tun_url else {}),
                "meta": clean_meta,
                "api_key": api_key,
                "comfy_api_key": comfy_api_key,
            }
    elif st_m != 404:
        raise RuntimeError(f"读取服务端元数据 HTTP {st_m}")

    if out is None:
        comfy_u = _read_url_file("root/comfy_tunnel_url.txt")
        out = {
            "status": "ok" if comfy_u else "missing",
            "endpoint": target.endpoint,
            **({"url": comfy_u} if comfy_u else {}),
            "meta": {"kind": "comfy", "port": 8188},
            "comfy_api_key": "",
        }
    if not out.get("comfy_api_key"):
        st_k, raw_k = _read_text("root/comfy_auth_key")
        if st_k == 200 and str(raw_k).strip():
            out["comfy_api_key"] = str(raw_k).strip().splitlines()[0].strip()

    print("<<<COLAB_TUNNEL>>>" + json.dumps(out, ensure_ascii=False) + "<<<END_COLAB_TUNNEL>>>", flush=True)
    return 0


def cmd_contents_put(args: list[str]) -> int:
    target_ep, session_name, rel_path = args[0], args[1], args[2].lstrip("/")
    b64_data = sys.stdin.read().strip()
    target, _ = _pick_target_assignment(target_ep, session_name)
    if target is None:
        raise RuntimeError("无活跃 Colab 实例")
    info = target.runtime_proxy_info
    resp = requests.put(
        f"{info.url.rstrip('/')}/api/contents/{rel_path}",
        params={"authuser": "0", "colab-runtime-proxy-token": info.token},
        json={"type": "file", "format": "base64", "content": b64_data},
        timeout=15,
    )
    resp.raise_for_status()
    print("OK")
    return 0


def cmd_install_ext(args: list[str]) -> int:
    target_ep = args[0] if len(args) > 0 else ""
    session_name = args[1] if len(args) > 1 else "comfy"
    payload = json.loads(sys.stdin.read())
    js_b64, py_b64 = payload["js_b64"], payload["py_b64"]

    target, _ = _pick_target_assignment(target_ep, session_name)
    if target is None:
        raise RuntimeError("无活跃 Colab 实例")
    info = target.runtime_proxy_info
    base, tok = info.url.rstrip("/"), info.token
    sess = requests.Session()

    def _ensure_dir(rel_dir: str) -> None:
        cur = ""
        for part in [p for p in rel_dir.strip("/").split("/") if p]:
            cur = f"{cur}/{part}" if cur else part
            r = _contents_get(sess, base, tok, cur, content="0", timeout=10)
            if r.status_code == 200:
                continue
            if r.status_code == 404:
                pr = sess.put(f"{base}/api/contents/{cur}", params={"authuser": "0", "colab-runtime-proxy-token": tok}, json={"type": "directory"}, timeout=10)
                if pr.status_code not in (200, 201, 409):
                    pr.raise_for_status()
            else:
                r.raise_for_status()

    def _put_b64(rel_file: str, b64_str: str) -> None:
        parent = rel_file.strip("/").rsplit("/", 1)[0] if "/" in rel_file.strip("/") else ""
        if parent:
            _ensure_dir(parent)
        pr = sess.put(
            f"{base}/api/contents/{rel_file.lstrip('/')}",
            params={"authuser": "0", "colab-runtime-proxy-token": tok},
            json={"type": "file", "format": "base64", "content": b64_str},
            timeout=15,
        )
        pr.raise_for_status()

    web_root = ""
    for log_rel in ("root/comfy_colab.log", "root/comfy_bw1.log", "root/comfy_minimax.log", "tmp/comfy_h3.log", "tmp/comfyui.log"):
        r = _contents_get(sess, base, tok, log_rel, content="1", timeout=10)
        if r.status_code == 200:
            m = re.search(r"\[Prompt Server\] web root:\s*(\S+)", str(_decode_contents(r.json())))
            if m:
                web_root = m.group(1).strip().lstrip("/")
                break

    if not web_root:
        r = _contents_get(sess, base, tok, "usr/local/lib", content="1", timeout=10)
        if r.status_code == 200:
            for item in (r.json().get("content") or []):
                name = str(item.get("name") or "")
                if name.startswith("python3"):
                    cand = f"usr/local/lib/{name}/dist-packages/comfyui_frontend_package/static"
                    if _contents_get(sess, base, tok, cand, content="0", timeout=10).status_code == 200:
                        web_root = cand
                        break
    if not web_root:
        raise RuntimeError("未定位到云端 ComfyUI web_root")

    _put_b64(f"{web_root}/extensions/colab_pet/direct_canvas.js", js_b64)
    try:
        _put_b64("content/ComfyUI/custom_nodes/colab_pet_bridge/__init__.py", py_b64)
        _put_b64("content/ComfyUI/custom_nodes/colab_pet_bridge/js/direct_canvas.js", js_b64)
    except Exception:
        pass
    print("<<<COLAB_CANVAS_EXT>>>" + json.dumps({"ok": True, "web_root": web_root}, ensure_ascii=False) + "<<<END_COLAB_CANVAS_EXT>>>", flush=True)
    return 0


def cmd_keepalive(args: list[str]) -> int:
    target_ep = args[0].strip() if args else ""
    client = state.client
    assignments = client.list_assignments()
    if target_ep:
        assignments = [a for a in (assignments or []) if getattr(a, "endpoint", "") == target_ep]
    if not assignments:
        print("<<<COLAB_KEEPALIVE>>>" + json.dumps({"ok": True, "pinged": 0}, ensure_ascii=False) + "<<<END_COLAB_KEEPALIVE>>>", flush=True)
        return 0

    sess_by_ep = {s.endpoint: s for s in state.store.list().values()}
    pinged = 0
    for a in assignments:
        touched = False
        try:
            r1 = client.session.get(
                f"{client.colab_domain.rstrip('/')}/tun/m/{a.endpoint}/keep-alive/",
                params={"authuser": "0"},
                headers={"X-Colab-Tunnel": "Google", "X-Colab-Client-Agent": "colab-cli", "Accept": "application/json"},
                timeout=8,
            )
            touched = touched or r1.ok
        except Exception:
            pass

        proxy_url, tok = a.runtime_proxy_info.url.rstrip("/"), a.runtime_proxy_info.token
        idle_kid = None
        try:
            r2 = requests.get(f"{proxy_url}/api/kernels", params={"authuser": "0", "colab-runtime-proxy-token": tok}, timeout=8)
            if r2.ok:
                touched = True
                for k in (r2.json() if isinstance(r2.json(), list) else []):
                    if isinstance(k, dict) and k.get("execution_state") == "idle" and k.get("id"):
                        idle_kid = str(k["id"])
                        break
        except Exception:
            pass

        if idle_kid:
            sess = sess_by_ep.get(a.endpoint)
            rt = ColabRuntime(proxy_url, tok, kernel_id=idle_kid, session_id=getattr(sess, "session_id", None) if sess else None)
            try:
                ws = rt.kernel_client._manager.client
                if ws.connection_ready.is_set() or ws.connection_ready.wait(timeout=8.0):
                    rt.execute_code("1 + 1", timeout=6.0)
                    touched = True
            except Exception:
                pass
            finally:
                try:
                    rt.stop()
                except Exception:
                    pass
        if touched:
            pinged += 1

    print("<<<COLAB_KEEPALIVE>>>" + json.dumps({"ok": pinged > 0, "pinged": pinged}, ensure_ascii=False) + "<<<END_COLAB_KEEPALIVE>>>", flush=True)
    return 0


def cmd_oauth_start(_args: list[str]) -> int:
    flow = InstalledAppFlow.from_client_config(_load_oauth_client_config(), colab_auth.PUBLIC_SCOPES, autogenerate_code_verifier=True)
    flow.redirect_uri = colab_auth.REMOTE_REDIRECT_URI
    auth_url, _ = flow.authorization_url(prompt="select_account consent", token_usage="remote")
    print("<<<COLAB_OAUTH_START>>>" + json.dumps({"ok": True, "auth_url": auth_url, "code_verifier": str(getattr(flow, "code_verifier", "") or "")}, ensure_ascii=False) + "<<<END_COLAB_OAUTH_START>>>", flush=True)
    return 0


def cmd_oauth_complete(_args: list[str]) -> int:
    req_in = json.loads(sys.stdin.read() or "{}")
    code = str(req_in.get("code") or "").strip()
    verifier = str(req_in.get("code_verifier") or "").strip()
    prev_acct_id = str(req_in.get("prev_acct_id") or "").strip()
    client_config = _load_oauth_client_config()

    try:
        prev_tok_json = colab_auth._get_token_from_keyring()
        if prev_tok_json:
            backup_id = prev_acct_id
            if not backup_id:
                prev_creds = Credentials.from_authorized_user_info(json.loads(prev_tok_json), colab_auth.PUBLIC_SCOPES)
                if not prev_creds.valid and prev_creds.expired and prev_creds.refresh_token:
                    try:
                        prev_creds.refresh(GoogleAuthRequest())
                        prev_tok_json = prev_creds.to_json()
                    except Exception:
                        pass
                prev_email = _extract_email(prev_creds) if prev_creds.valid else ""
                if prev_email:
                    backup_id = hashlib.sha256(prev_email.encode("utf-8")).hexdigest()[:12]
            if backup_id:
                keyring.set_password(VAULT_SERVICE_NAME, f"acct:{backup_id}", prev_tok_json)
                _backup_sessions(backup_id)
    except Exception:
        pass

    flow = InstalledAppFlow.from_client_config(client_config, colab_auth.PUBLIC_SCOPES, autogenerate_code_verifier=False)
    flow.redirect_uri = colab_auth.REMOTE_REDIRECT_URI
    if verifier:
        flow.code_verifier = verifier
    flow.fetch_token(code=code)
    creds = flow.credentials
    email = _extract_email(creds)
    acct_id = hashlib.sha256((email or creds.refresh_token or creds.token or "").encode("utf-8")).hexdigest()[:12]
    tok_json = creds.to_json()
    keyring.set_password(VAULT_SERVICE_NAME, f"acct:{acct_id}", tok_json)
    colab_auth._save_token_to_keyring(tok_json)
    colab_auth._remove_legacy_token()
    _restore_sessions(acct_id)

    out = {
        "ok": True,
        "acct_id": acct_id,
        "email_masked": _mask_email(email) if email else f"acct-{acct_id[:6]}",
        "prefix": email.split("@", 1)[0][:10] if email else f"Ultra-{acct_id[:4]}",
    }
    print("<<<COLAB_OAUTH_DONE>>>" + json.dumps(out, ensure_ascii=False) + "<<<END_COLAB_OAUTH_DONE>>>", flush=True)
    return 0


def cmd_account_op(_args: list[str]) -> int:
    req_in = json.loads(sys.stdin.read() or "{}")
    op = str(req_in.get("op") or "").strip().lower()
    target_id = str(req_in.get("target_acct_id") or "").strip()
    prev_id = str(req_in.get("prev_acct_id") or "").strip()

    if op == "switch":
        if prev_id:
            try:
                cur_tok = colab_auth._get_token_from_keyring()
                if cur_tok:
                    keyring.set_password(VAULT_SERVICE_NAME, f"acct:{prev_id}", cur_tok)
            except Exception:
                pass
            _backup_sessions(prev_id)

        target_tok = keyring.get_password(VAULT_SERVICE_NAME, f"acct:{target_id}")
        if not target_tok:
            raise RuntimeError(f"系统凭据库中未找到账号 {target_id} 的授权令牌，请重新绑定")
        try:
            creds = Credentials.from_authorized_user_info(json.loads(target_tok), colab_auth.PUBLIC_SCOPES)
            if not creds.valid and creds.expired and creds.refresh_token:
                creds.refresh(GoogleAuthRequest())
                target_tok = creds.to_json()
                keyring.set_password(VAULT_SERVICE_NAME, f"acct:{target_id}", target_tok)
        except Exception:
            pass
        colab_auth._save_token_to_keyring(target_tok)
        colab_auth._remove_legacy_token()
        _restore_sessions(target_id)
        print("<<<COLAB_ACCT_OP>>>" + json.dumps({"ok": True, "op": "switch", "target_acct_id": target_id}) + "<<<END_COLAB_ACCT_OP>>>", flush=True)
        return 0

    if op == "remove":
        for fn in (
            lambda: keyring.delete_password(VAULT_SERVICE_NAME, f"acct:{target_id}"),
            lambda: (SESS_BACKUP_DIR / f"{target_id}.json").unlink(missing_ok=True),
        ):
            try:
                fn()
            except Exception:
                pass
        if bool(req_in.get("clear_active")):
            try:
                keyring.delete_password(colab_auth.KEYRING_SERVICE_NAME, colab_auth.KEYRING_USERNAME)
            except Exception:
                pass
            colab_auth._remove_legacy_token()
            _write_private_text(SESS_FILE, "{}\n")
        print("<<<COLAB_ACCT_OP>>>" + json.dumps({"ok": True, "op": "remove", "target_acct_id": target_id}) + "<<<END_COLAB_ACCT_OP>>>", flush=True)
        return 0

    raise ValueError(f"未知操作: {op}")


_DISPATCH = {
    "probe": cmd_probe,
    "fast-tunnel": cmd_fast_tunnel,
    "contents-put": cmd_contents_put,
    "install-ext": cmd_install_ext,
    "keepalive": cmd_keepalive,
    "oauth-start": cmd_oauth_start,
    "oauth-complete": cmd_oauth_complete,
    "account-op": cmd_account_op,
}


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] not in _DISPATCH:
        raise SystemExit(f"用法: colab_cli_bridge.py <{'|'.join(_DISPATCH)}> [args...]")
    return int(_DISPATCH[args[0]](args[1:]) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
