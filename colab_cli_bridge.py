"""colab_cli_bridge.py —— 运行在 `google-colab-cli` Python 环境内的统一桥接助手。

收拢原散落于多处模块中的 `colab_py -c` 内嵌脚本，共用实例定位、Contents REST 读写、
邮箱脱敏、私钥文件原子写与多账号 OAuth/保险箱操作，消除字符串套字符串的重复维护。
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from importlib import resources
from pathlib import Path
from urllib.parse import urljoin

import keyring
import requests
from google.auth.transport.requests import AuthorizedSession, Request as GoogleAuthRequest
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from colab_cli import auth as colab_auth
from colab_cli.cli import app as colab_cli_app
from colab_cli.client import Client, Prod
from colab_cli.common import state
from colab_cli.runtime import ColabRuntime
from colab_cli.state import StateStore

VAULT_SERVICE_NAME = "google-colab-cli-pet-vault"
CFG_DIR = Path(os.path.expanduser("~/.config/colab-cli"))
SESS_FILE = CFG_DIR / "sessions.json"
SESS_BACKUP_DIR = CFG_DIR / "pet_sessions_by_acct"
DEFAULT_TUNNEL_PATTERN = r"https://(?!api\.)[a-z0-9-]+\.trycloudflare\.com(?![a-z0-9.-])"
_VAULT_LOCK = threading.RLock()


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
    # 可选拉取 OAuth userinfo 邮箱，网络或接口不可用时自动降级解析本地 id_token
    with contextlib.suppress(requests.RequestException, ValueError, TypeError, KeyError):
        r = requests.get(
            "https://www.googleapis.com/oauth2/v2/userinfo",
            headers={"Authorization": f"Bearer {creds.token}"},
            timeout=8,
        )
        if r.status_code == 200:
            em = str((r.json() or {}).get("email") or "").strip().lower()
            if em and "@" in em:
                return em
    # 兜底解码 JWT id_token 载荷，格式异常或非标准令牌时返回空串由上层回退哈希 ID
    with contextlib.suppress(ValueError, TypeError, AttributeError, UnicodeDecodeError):
        id_tok = getattr(creds, "id_token", None)
        if isinstance(id_tok, str) and id_tok.count(".") >= 2:
            seg = id_tok.split(".")[1]
            seg += "=" * (-len(seg) % 4)
            claims = json.loads(base64.urlsafe_b64decode(seg.encode("ascii")).decode("utf-8", errors="ignore"))
            em = str((claims or {}).get("email") or "").strip().lower()
            if em and "@" in em:
                return em
    return ""


def _write_private_text(path_obj: Path, content: str) -> None:
    """以 0o600 私有权限与线程安全临时文件原子写入文本。"""
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    tmp_p = path_obj.with_name(f".{path_obj.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    fd = os.open(str(tmp_p), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fp:
            fp.write(content)
        try:
            os.chmod(str(tmp_p), 0o600)
        except OSError:
            pass
        tmp_p.replace(path_obj)
        try:
            os.chmod(str(path_obj), 0o600)
        except OSError:
            pass
    except BaseException:
        try:
            tmp_p.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _sanitize_acct_id(acct_id: str) -> str:
    """清洗并校验账号 ID，防止路径穿越。"""
    return re.sub(r"[^a-zA-Z0-9_-]", "", str(acct_id or "").strip())[:64]


def _mock_vault_path() -> Path | None:
    raw = str(os.environ.get("COLAB_PET_MOCK_VAULT_PATH") or "").strip()
    return Path(raw) if raw else None


def _read_vault_token(acct_id: str) -> str | None:
    """按 colab_security 模式从测试沙箱文件或系统凭据库读取指定账号的 OAuth JSON。"""
    clean_id = _sanitize_acct_id(acct_id)
    if not clean_id:
        return None
    vault_key = f"acct:{clean_id}"
    mock_p = _mock_vault_path()
    with _VAULT_LOCK:
        if mock_p is not None:
            if not mock_p.is_file():
                return None
            try:
                data = json.loads(mock_p.read_text(encoding="utf-8"))
                if isinstance(data, dict) and data.get(vault_key):
                    return str(data[vault_key])
            except Exception:
                return None
            return None
        return keyring.get_password(VAULT_SERVICE_NAME, vault_key)


def _write_vault_token(acct_id: str, token_json: str) -> None:
    """将指定账号的最新 OAuth JSON 写回测试沙箱文件或系统凭据库（不改动默认主槽位）。"""
    clean_id = _sanitize_acct_id(acct_id)
    if not clean_id:
        return
    vault_key = f"acct:{clean_id}"
    mock_p = _mock_vault_path()
    with _VAULT_LOCK:
        if mock_p is not None:
            data: dict[str, str] = {}
            if mock_p.is_file():
                try:
                    raw = json.loads(mock_p.read_text(encoding="utf-8"))
                    if isinstance(raw, dict):
                        data = {str(k): str(v) for k, v in raw.items()}
                except Exception:
                    data = {}
            data[vault_key] = str(token_json)
            _write_private_text(mock_p, json.dumps(data, ensure_ascii=False, indent=2) + "\n")
            return
        keyring.set_password(VAULT_SERVICE_NAME, vault_key, str(token_json))


def _delete_vault_token(acct_id: str) -> None:
    """删除指定账号在沙箱或系统凭据库中的 OAuth 槽位。"""
    clean_id = _sanitize_acct_id(acct_id)
    if not clean_id:
        return
    vault_key = f"acct:{clean_id}"
    mock_p = _mock_vault_path()
    with _VAULT_LOCK:
        if mock_p is not None:
            if not mock_p.is_file():
                return
            # 沙箱模拟凭据文件损坏或只读时安全跳过删除，避免阻断解绑主流程
            with contextlib.suppress(OSError, ValueError, TypeError):
                raw = json.loads(mock_p.read_text(encoding="utf-8"))
                if isinstance(raw, dict) and vault_key in raw:
                    raw.pop(vault_key, None)
                    _write_private_text(mock_p, json.dumps(raw, ensure_ascii=False, indent=2) + "\n")
            return
        keyring.delete_password(VAULT_SERVICE_NAME, vault_key)


def _load_account_creds(acct_id: str) -> tuple[Credentials | None, bool]:
    """加载指定账号凭据；若已过期且含 refresh_token 则自动刷新并仅回写该账号保险箱槽位。"""
    clean_id = _sanitize_acct_id(acct_id)
    if not clean_id:
        return None, False
    try:
        tok_json = _read_vault_token(clean_id)
        if not tok_json:
            return None, False
        creds = Credentials.from_authorized_user_info(json.loads(tok_json), colab_auth.PUBLIC_SCOPES)
    except Exception:
        return None, False

    if creds is not None and not creds.valid:
        if creds.expired and creds.refresh_token:
            try:
                creds.refresh(GoogleAuthRequest())
                _write_vault_token(clean_id, creds.to_json())
            except Exception:
                return None, True
        else:
            return None, True
    return creds, False


def _acct_session_file(acct_id: str) -> Path:
    clean_id = _sanitize_acct_id(acct_id)
    return SESS_BACKUP_DIR / f"{clean_id}.json"


def _bind_account_state(acct_id: str) -> Path:
    """在不改写默认 keyring 主槽与默认 sessions.json 的前提下，将全局 state 绑定到目标账号作用域。"""
    clean_id = _sanitize_acct_id(acct_id)
    if not clean_id:
        raise ValueError("无效的账号 ID")
    creds, expired = _load_account_creds(clean_id)
    if creds is None:
        hint = "授权已过期，请重新绑定" if expired else "未找到授权令牌，请重新绑定"
        raise RuntimeError(f"账号 {clean_id} {hint}")

    sess_path = _acct_session_file(clean_id)
    SESS_BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    state._client = Client(Prod(), AuthorizedSession(creds))
    state.config_path = str(sess_path)
    state._store = None
    state._sessions = None
    return sess_path


def _extract_acct_arg(args: list[str]) -> tuple[str, list[str]]:
    """从参数列表剥离可选的 --acct=<id> 或 --acct <id>，返回 (acct_id, 剩余参数)。"""
    acct_val = ""
    rest: list[str] = []
    skip_next = False
    for idx, item in enumerate(args):
        if skip_next:
            skip_next = False
            continue
        if item.startswith("--acct="):
            acct_val = item.split("=", 1)[1].strip()
        elif item == "--acct" and idx + 1 < len(args):
            acct_val = args[idx + 1].strip()
            skip_next = True
        else:
            rest.append(item)
    eff_acct = _sanitize_acct_id(acct_val or os.environ.get("COLAB_PET_ACCOUNT_ID") or "")
    return eff_acct, rest


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
        with open(cfg_path, encoding="utf-8") as fp:
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


def _collect_merged_sessions(
    *,
    acct_id: str = "",
    include_global: bool = True,
    include_all_backups: bool = False,
) -> dict:
    """只读聚合已知 Colab 会话映射（按 endpoint 去重，优先当前账号专属会话文件，绝不改写 sessions.json）。"""
    by_ep: dict[str, object] = {}

    def _absorb(store_obj: StateStore) -> None:
        try:
            sess_map = store_obj.list()
        except Exception:
            return
        for s in (sess_map or {}).values():
            ep = str(getattr(s, "endpoint", "") or "").strip()
            sname = str(getattr(s, "name", "") or "").strip()
            if ep and sname and sname != "?":
                by_ep[ep] = s

    if include_global:
        _absorb(state.store)

    if include_all_backups and SESS_BACKUP_DIR.is_dir():
        for bak_file in sorted(SESS_BACKUP_DIR.glob("*.json")):
            if bak_file.is_file():
                _absorb(StateStore(str(bak_file)))

    clean_id = _sanitize_acct_id(acct_id)
    if clean_id:
        acct_file = _acct_session_file(clean_id)
        if acct_file.is_file():
            _absorb(StateStore(str(acct_file)))

    return by_ep


def _probe_client_metrics(client: Client, local_sessions: dict) -> dict:
    """复用同一客户端查询账号 CU 配额、可用显卡池及在线会话列表。"""
    eligible_gpus, ineligible_gpus, eligible_tpus = None, None, None
    subscription_tier = ""
    balance, rate_hourly, assignments_count = None, 0.0, 0
    # 优先探测扩展 /tun/m/ccu-info（含可用显卡池），接口不可用时自动回退标准 consumption 接口
    with contextlib.suppress(requests.RequestException, ValueError, TypeError, KeyError, RuntimeError):
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
        if isinstance(raw_ccu.get("ineligibleGpus"), list):
            ineligible_gpus = [str(x).upper() for x in raw_ccu["ineligibleGpus"] if x]
        if isinstance(raw_ccu.get("eligibleTpus"), list):
            eligible_tpus = [str(x).upper() for x in raw_ccu["eligibleTpus"] if x]
        subscription_tier = str(
            raw_ccu.get("subscriptionTier") or raw_ccu.get("proTier") or ""
        ).strip()

    if balance is None:
        info = _retry(client.get_consumption_user_info)
        balance = float(info.paid_compute_units_balance)
        rate_hourly = float(info.consumption_rate_hourly)
        assignments_count = int(info.assignments_count)

    assignments = _retry(client.list_assignments)
    name_by_ep = {
        s.endpoint: s.name
        for s in local_sessions.values()
        if getattr(s, "endpoint", None) and getattr(s, "name", None) not in (None, "", "?")
    }
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

    return {
        "balance": float(balance),
        "rate_hourly": float(rate_hourly),
        "assignments_count": int(assignments_count),
        "eligible_gpus": eligible_gpus,
        "ineligible_gpus": ineligible_gpus,
        "eligible_tpus": eligible_tpus,
        "subscription_tier": subscription_tier,
        "sessions": items,
    }


def cmd_probe(args: list[str]) -> int:
    scoped_acct, clean_args = _extract_acct_arg(args)
    need_userinfo = "--need-userinfo" in clean_args
    active_acct_id = ""
    for arg in clean_args:
        if arg.startswith("--active-acct="):
            active_acct_id = arg.split("=", 1)[1].strip()

    creds = None
    auth_expired = False
    if scoped_acct:
        creds, auth_expired = _load_account_creds(scoped_acct)
    else:
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
                        # 同步镜像刷新后的令牌到多账号保险箱槽位，失败不影响当前探针主流程
                        with contextlib.suppress(Exception):
                            _write_vault_token(active_acct_id, new_json)
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
            # 尽力把当前活跃凭据镜像到多账号保险箱槽位，失败不影响状态探测返回
            with contextlib.suppress(Exception):
                _write_vault_token(acct_id, creds.to_json())
            account_meta = {"id": acct_id, "email_masked": _mask_email(email), "prefix": email.split("@", 1)[0][:10]}

    if scoped_acct:
        client = Client(Prod(), AuthorizedSession(creds))
    else:
        client = state.client

    eff_acct = scoped_acct or _sanitize_acct_id(active_acct_id) or _sanitize_acct_id(
        str((account_meta or {}).get("id") or "")
    )
    local_sessions = _collect_merged_sessions(
        acct_id=eff_acct,
        include_global=True,
        include_all_backups=not bool(eff_acct),
    )

    metrics = _probe_client_metrics(client, local_sessions)
    payload = {
        "auth_missing": False,
        "account_meta": account_meta,
        **metrics,
    }
    print("<<<COLAB_PET_JSON>>>" + json.dumps(payload, ensure_ascii=False) + "<<<END_COLAB_PET_JSON>>>", flush=True)
    return 0


def cmd_fast_tunnel(args: list[str]) -> int:
    scoped_acct, clean_args = _extract_acct_arg(args)
    if scoped_acct:
        _bind_account_state(scoped_acct)
    target_ep = clean_args[0] if len(clean_args) > 0 else ""
    session_name = clean_args[1] if len(clean_args) > 1 else "comfy"
    pat_str = clean_args[2] if len(clean_args) > 2 and clean_args[2] else DEFAULT_TUNNEL_PATTERN
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
    scoped_acct, clean_args = _extract_acct_arg(args)
    if scoped_acct:
        _bind_account_state(scoped_acct)
    target_ep, session_name, rel_path = clean_args[0], clean_args[1], clean_args[2].lstrip("/")
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
    scoped_acct, clean_args = _extract_acct_arg(args)
    if scoped_acct:
        _bind_account_state(scoped_acct)
    target_ep = clean_args[0] if len(clean_args) > 0 else ""
    session_name = clean_args[1] if len(clean_args) > 1 else "comfy"
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
    # 主前端扩展已落盘成功；同步 custom_nodes 副本为可选增强，失败不阻断直铺画布生效
    with contextlib.suppress(requests.RequestException, OSError, RuntimeError):
        _put_b64("content/ComfyUI/custom_nodes/colab_pet_bridge/__init__.py", py_b64)
        _put_b64("content/ComfyUI/custom_nodes/colab_pet_bridge/js/direct_canvas.js", js_b64)
    print("<<<COLAB_CANVAS_EXT>>>" + json.dumps({"ok": True, "web_root": web_root}, ensure_ascii=False) + "<<<END_COLAB_CANVAS_EXT>>>", flush=True)
    return 0


def cmd_keepalive(args: list[str]) -> int:
    scoped_acct, clean_args = _extract_acct_arg(args)
    if scoped_acct:
        _bind_account_state(scoped_acct)
    target_ep = clean_args[0].strip() if clean_args else ""
    client = state.client
    assignments = client.list_assignments()
    if target_ep:
        assignments = [a for a in (assignments or []) if getattr(a, "endpoint", "") == target_ep]
    if not assignments:
        print("<<<COLAB_KEEPALIVE>>>" + json.dumps({"ok": True, "pinged": 0}, ensure_ascii=False) + "<<<END_COLAB_KEEPALIVE>>>", flush=True)
        return 0

    sess_by_ep = _collect_merged_sessions(
        acct_id=scoped_acct,
        include_global=True,
        include_all_backups=not bool(scoped_acct),
    )
    pinged = 0
    for a in assignments:
        touched = False
        # 尽力探测 /keep-alive/ 接口保活，失败时继续尝试下方内核级轻触保活
        with contextlib.suppress(requests.RequestException, OSError, RuntimeError):
            r1 = client.session.get(
                f"{client.colab_domain.rstrip('/')}/tun/m/{a.endpoint}/keep-alive/",
                params={"authuser": "0"},
                headers={"X-Colab-Tunnel": "Google", "X-Colab-Client-Agent": "colab-cli", "Accept": "application/json"},
                timeout=8,
            )
            touched = touched or r1.ok

        proxy_url, tok = a.runtime_proxy_info.url.rstrip("/"), a.runtime_proxy_info.token
        idle_kid = None
        # 尽力读取空闲内核列表以执行轻量心跳，接口暂不可达时不阻断后续实例
        with contextlib.suppress(requests.RequestException, ValueError, OSError):
            r2 = requests.get(f"{proxy_url}/api/kernels", params={"authuser": "0", "colab-runtime-proxy-token": tok}, timeout=8)
            if r2.ok:
                touched = True
                for k in (r2.json() if isinstance(r2.json(), list) else []):
                    if isinstance(k, dict) and k.get("execution_state") == "idle" and k.get("id"):
                        idle_kid = str(k["id"])
                        break

        if idle_kid:
            sess = sess_by_ep.get(a.endpoint)
            rt = ColabRuntime(proxy_url, tok, kernel_id=idle_kid, session_id=getattr(sess, "session_id", None) if sess else None)
            try:
                # 轻触空闲内核保活失败属非致命抖动，外层按 touched 汇总统计
                with contextlib.suppress(Exception):
                    ws = rt.kernel_client._manager.client
                    if ws.connection_ready.is_set() or ws.connection_ready.wait(timeout=8.0):
                        rt.execute_code("1 + 1", timeout=6.0)
                        touched = True
            finally:
                # 释放 WebSocket 句柄时的清理异常可安全忽略
                with contextlib.suppress(Exception):
                    rt.stop()
        if touched:
            pinged += 1

    print("<<<COLAB_KEEPALIVE>>>" + json.dumps({"ok": pinged > 0, "pinged": pinged}, ensure_ascii=False) + "<<<END_COLAB_KEEPALIVE>>>", flush=True)
    return 0


_HM_KERNEL_PROBE_CODE = (
    "import json as _j, pathlib as _p, shutil as _s\n"
    "_rt=_rf=_ru=_rp=0.0\n"
    "_mi=_p.Path('/proc/meminfo')\n"
    "if _mi.is_file():\n"
    " _v={}\n"
    " for _l in _mi.read_text(encoding='utf-8',errors='ignore').splitlines():\n"
    "  if ':' in _l:\n"
    "   _k,_x=_l.split(':',1);_ps=_x.strip().split()\n"
    "   if _ps and _ps[0].isdigit(): _v[_k.strip()]=int(_ps[0])\n"
    " _tk=_v.get('MemTotal',0);_ak=_v.get('MemAvailable')\n"
    " if _ak is None and _tk>0: _ak=_v.get('MemFree',0)+_v.get('Buffers',0)+_v.get('Cached',0)\n"
    " if _tk>0 and _ak is not None:\n"
    "  _ak=max(0,min(_tk,_ak));_uk=max(0,_tk-_ak)\n"
    "  _rt=round(_tk/1048576,1);_rf=round(_ak/1048576,1);_ru=round(_uk/1048576,1);_rp=round(_uk*100/_tk,1)\n"
    "_dt=_df=_du=_dp=0.0\n"
    "try:\n"
    " _du_o=_s.disk_usage('/content' if _p.Path('/content').exists() else '/')\n"
    " if _du_o.total>0: _ub=max(0,_du_o.total-_du_o.free);_dt=round(_du_o.total/1073741824,1);_df=round(_du_o.free/1073741824,1);_du=round(_ub/1073741824,1);_dp=round(_ub*100/_du_o.total,1)\n"
    "except Exception: pass\n"
    "_st=_sf=_su=0.0\n"
    "try:\n"
    " if _p.Path('/dev/shm').exists():\n"
    "  _su_o=_s.disk_usage('/dev/shm')\n"
    "  if _su_o.total>0: _st=round(_su_o.total/1073741824,1);_sf=round(_su_o.free/1073741824,1);_su=round(max(0,_su_o.total-_su_o.free)/1073741824,1)\n"
    "except Exception: pass\n"
    "_dp_f=_p.Path('/content/drive/MyDrive').is_dir() or _p.Path('/tmp/rclone.conf').is_file()\n"
    "_dr_f=_dr_t=0.0\n"
    "try:\n"
    " if _p.Path('/content/drive/MyDrive').is_dir():\n"
    "  _do=_s.disk_usage('/content/drive/MyDrive')\n"
    "  if _do.total>0: _dr_t=round(_do.total/1073741824,1);_dr_f=round(_do.free/1073741824,1)\n"
    "except Exception: pass\n"
    "print('<<<COLAB_HM>>>'+_j.dumps({'ok':True,'ram_used_gb':_ru,'ram_total_gb':_rt,'ram_free_gb':_rf,'ram_pct':_rp,'disk_used_gb':_du,'disk_total_gb':_dt,'disk_free_gb':_df,'disk_pct':_dp,'shm_used_gb':_su,'shm_total_gb':_st,'shm_free_gb':_sf,'drive_persisted':bool(_dp_f),'drive_free_gb':_dr_f,'drive_total_gb':_dr_t},separators=(',',':'))+'<<<END_COLAB_HM>>>',flush=True)"
)


def cmd_host_metrics(args: list[str]) -> int:
    """只读采集老实例的磁盘 / 内存 / /dev/shm / 云盘遥测，仅在空闲内核上轻触一次。"""
    scoped_acct, clean_args = _extract_acct_arg(args)
    if scoped_acct:
        _bind_account_state(scoped_acct)
    target_ep = clean_args[0].strip() if len(clean_args) > 0 else ""
    session_name = clean_args[1].strip() if len(clean_args) > 1 else "comfy"

    target, _ = _pick_target_assignment(target_ep, session_name)
    if target is None:
        print("<<<COLAB_HOST_METRICS>>>" + json.dumps({"ok": False, "reason": "no_session"}, ensure_ascii=False) + "<<<END_COLAB_HOST_METRICS>>>", flush=True)
        return 0

    sess_by_ep = _collect_merged_sessions(
        acct_id=scoped_acct,
        include_global=True,
        include_all_backups=not bool(scoped_acct),
    )
    proxy_url, tok = target.runtime_proxy_info.url.rstrip("/"), target.runtime_proxy_info.token
    idle_kid = None
    # 可选查询空闲内核，查询失败时退化返回 no_idle_kernel 而不抛异常
    with contextlib.suppress(requests.RequestException, ValueError, OSError):
        r2 = requests.get(f"{proxy_url}/api/kernels", params={"authuser": "0", "colab-runtime-proxy-token": tok}, timeout=6)
        if r2.ok:
            for k in (r2.json() if isinstance(r2.json(), list) else []):
                if isinstance(k, dict) and k.get("execution_state") == "idle" and k.get("id"):
                    idle_kid = str(k["id"])
                    break

    if not idle_kid:
        print("<<<COLAB_HOST_METRICS>>>" + json.dumps({"ok": False, "reason": "no_idle_kernel"}, ensure_ascii=False) + "<<<END_COLAB_HOST_METRICS>>>", flush=True)
        return 0

    sess = sess_by_ep.get(target.endpoint)
    rt = ColabRuntime(proxy_url, tok, kernel_id=idle_kid, session_id=getattr(sess, "session_id", None) if sess else None)
    collected: list[str] = []
    try:
        # 遥测采集失败时按 probe_empty 优雅返回，避免打断主监控轮询
        with contextlib.suppress(Exception):
            ws = rt.kernel_client._manager.client
            if ws.connection_ready.is_set() or ws.connection_ready.wait(timeout=6.0):
                outputs = rt.execute_code(_HM_KERNEL_PROBE_CODE, timeout=8.0) or []
                for o in outputs:
                    if isinstance(o, dict) and o.get("output_type") == "stream":
                        collected.append(str(o.get("text") or ""))
    finally:
        # 清理 WebSocket 句柄异常可安全忽略
        with contextlib.suppress(Exception):
            rt.stop()

    raw_txt = "".join(collected)
    m = re.search(r"<<<COLAB_HM>>>(\{.*?\})<<<END_COLAB_HM>>>", raw_txt, flags=re.DOTALL)
    if m:
        print("<<<COLAB_HOST_METRICS>>>" + m.group(1) + "<<<END_COLAB_HOST_METRICS>>>", flush=True)
    else:
        print("<<<COLAB_HOST_METRICS>>>" + json.dumps({"ok": False, "reason": "probe_empty"}, ensure_ascii=False) + "<<<END_COLAB_HOST_METRICS>>>", flush=True)
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

    # 绑定新号前尽力备份旧账号令牌与会话映射，旧号已失效或无记录时不得阻塞新号绑定
    with contextlib.suppress(Exception):
        prev_tok_json = colab_auth._get_token_from_keyring()
        if prev_tok_json:
            backup_id = prev_acct_id
            if not backup_id:
                prev_creds = Credentials.from_authorized_user_info(json.loads(prev_tok_json), colab_auth.PUBLIC_SCOPES)
                if not prev_creds.valid and prev_creds.expired and prev_creds.refresh_token:
                    # 旧号刷新失败时沿用已有令牌尝试提取邮箱或跳过备份
                    with contextlib.suppress(Exception):
                        prev_creds.refresh(GoogleAuthRequest())
                        prev_tok_json = prev_creds.to_json()
                prev_email = _extract_email(prev_creds) if prev_creds.valid else ""
                if prev_email:
                    backup_id = hashlib.sha256(prev_email.encode("utf-8")).hexdigest()[:12]
            if backup_id:
                _write_vault_token(backup_id, prev_tok_json)
                _backup_sessions(backup_id)

    flow = InstalledAppFlow.from_client_config(client_config, colab_auth.PUBLIC_SCOPES, autogenerate_code_verifier=False)
    flow.redirect_uri = colab_auth.REMOTE_REDIRECT_URI
    if verifier:
        flow.code_verifier = verifier
    flow.fetch_token(code=code)
    creds = flow.credentials
    email = _extract_email(creds)
    acct_id = hashlib.sha256((email or creds.refresh_token or creds.token or "").encode("utf-8")).hexdigest()[:12]
    tok_json = creds.to_json()
    _write_vault_token(acct_id, tok_json)
    colab_auth._save_token_to_keyring(tok_json)
    colab_auth._remove_legacy_token()
    _restore_sessions(acct_id)

    out = {
        "ok": True,
        "acct_id": acct_id,
        "email_masked": _mask_email(email) if email else f"acct-{acct_id[:6]}",
        "prefix": email.split("@", 1)[0][:10] if email else f"Colab-{acct_id[:4]}",
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
            # 切号前尽力将旧账号当前 keyring 令牌备份回保险箱槽位，失败不阻塞切换
            with contextlib.suppress(Exception):
                cur_tok = colab_auth._get_token_from_keyring()
                if cur_tok:
                    _write_vault_token(prev_id, cur_tok)
            _backup_sessions(prev_id)

        target_tok = _read_vault_token(target_id)
        if not target_tok:
            raise RuntimeError(f"系统凭据库中未找到账号 {target_id} 的授权令牌，请重新绑定")
        # 切入前尽力预刷新过期令牌，网络短暂失败时仍允许写入旧令牌由后续探针处理
        with contextlib.suppress(Exception):
            creds = Credentials.from_authorized_user_info(json.loads(target_tok), colab_auth.PUBLIC_SCOPES)
            if not creds.valid and creds.expired and creds.refresh_token:
                creds.refresh(GoogleAuthRequest())
                target_tok = creds.to_json()
                _write_vault_token(target_id, target_tok)
        colab_auth._save_token_to_keyring(target_tok)
        colab_auth._remove_legacy_token()
        _restore_sessions(target_id)
        print("<<<COLAB_ACCT_OP>>>" + json.dumps({"ok": True, "op": "switch", "target_acct_id": target_id}) + "<<<END_COLAB_ACCT_OP>>>", flush=True)
        return 0

    if op == "remove":
        for fn in (
            lambda: _delete_vault_token(target_id),
            lambda: (SESS_BACKUP_DIR / f"{target_id}.json").unlink(missing_ok=True),
        ):
            # 清理目标槽位与备份会话文件时，条目已缺失属正常清理路径
            with contextlib.suppress(Exception):
                fn()
        if bool(req_in.get("clear_active")):
            # 最后一个账号解绑时尽力删除默认 keyring 槽位，条目不存在可安全忽略
            with contextlib.suppress(Exception):
                keyring.delete_password(colab_auth.KEYRING_SERVICE_NAME, colab_auth.KEYRING_USERNAME)
            colab_auth._remove_legacy_token()
            _write_private_text(SESS_FILE, "{}\n")
        print("<<<COLAB_ACCT_OP>>>" + json.dumps({"ok": True, "op": "remove", "target_acct_id": target_id}) + "<<<END_COLAB_ACCT_OP>>>", flush=True)
        return 0

    raise ValueError(f"未知操作: {op}")


def _probe_single_account(acct_id: str) -> dict:
    """在独立局部凭据与 StateStore 上下文中探测单个账号的余额、显卡权限与活跃实例。"""
    clean_id = _sanitize_acct_id(acct_id)
    if not clean_id:
        return {
            "ok": False,
            "auth_ok": False,
            "auth_missing": True,
            "auth_expired": False,
            "balance": None,
            "rate_hourly": 0.0,
            "assignments_count": 0,
            "eligible_gpus": None,
            "eligible_tpus": None,
            "sessions": [],
            "error": "无效账号 ID",
        }
    creds, auth_expired = _load_account_creds(clean_id)
    if creds is None:
        return {
            "ok": False,
            "auth_ok": False,
            "auth_missing": True,
            "auth_expired": bool(auth_expired),
            "balance": None,
            "rate_hourly": 0.0,
            "assignments_count": 0,
            "eligible_gpus": None,
            "eligible_tpus": None,
            "sessions": [],
            "error": "授权已过期" if auth_expired else "缺少授权令牌",
        }
    try:
        client = Client(Prod(), AuthorizedSession(creds))
        local_sessions = _collect_merged_sessions(acct_id=clean_id, include_global=True)
        metrics = _probe_client_metrics(client, local_sessions)
        return {
            "ok": True,
            "auth_ok": True,
            "auth_missing": False,
            "auth_expired": False,
            **metrics,
        }
    except Exception as exc:
        resp_status = getattr(getattr(exc, "response", None), "status_code", None)
        is_auth_err = resp_status in (401, 403)
        return {
            "ok": False,
            "auth_ok": not is_auth_err,
            "auth_missing": False,
            "auth_expired": bool(resp_status == 401),
            "balance": None,
            "rate_hourly": 0.0,
            "assignments_count": 0,
            "eligible_gpus": None,
            "eligible_tpus": None,
            "sessions": [],
            "error": f"{type(exc).__name__}: {str(exc)[:180]}",
        }


def cmd_pool_probe(_args: list[str]) -> int:
    """并发探测多个已绑定账号的 CCU 余额、可用显卡池与活跃会话状态。"""
    req_in = json.loads(sys.stdin.read() or "{}")
    raw_ids = req_in.get("account_ids") or []
    account_ids: list[str] = []
    for raw_id in raw_ids:
        cid = _sanitize_acct_id(str(raw_id or ""))
        if cid and cid not in account_ids:
            account_ids.append(cid)

    results: dict[str, dict] = {}
    if account_ids:
        with ThreadPoolExecutor(max_workers=4) as pool:
            fut_map = {pool.submit(_probe_single_account, aid): aid for aid in account_ids}
            raw_done: dict[str, dict] = {}
            for fut in as_completed(fut_map):
                aid = fut_map[fut]
                try:
                    raw_done[aid] = fut.result()
                except Exception as exc:
                    raw_done[aid] = {
                        "ok": False,
                        "auth_ok": False,
                        "auth_missing": False,
                        "auth_expired": False,
                        "balance": None,
                        "rate_hourly": 0.0,
                        "assignments_count": 0,
                        "eligible_gpus": None,
                        "eligible_tpus": None,
                        "sessions": [],
                        "error": f"{type(exc).__name__}: {str(exc)[:180]}",
                    }
        for aid in account_ids:
            results[aid] = raw_done[aid]

    print(
        "<<<COLAB_POOL_PROBE>>>"
        + json.dumps({"ok": True, "accounts": results}, ensure_ascii=False)
        + "<<<END_COLAB_POOL_PROBE>>>",
        flush=True,
    )
    return 0


def cmd_scoped_cli(args: list[str]) -> int:
    """在指定账号的隔离凭据与独立会话文件作用域内执行 google-colab-cli 子命令。"""
    if "--" in args:
        sep_idx = args.index("--")
        head_args, cli_args = args[:sep_idx], args[sep_idx + 1 :]
        acct_id, extra_head = _extract_acct_arg(head_args)
        cli_args = [*extra_head, *cli_args]
    else:
        acct_id, cli_args = _extract_acct_arg(args)

    if not acct_id:
        raise ValueError("scoped-cli 必须通过 --acct=<id> 或 COLAB_PET_ACCOUNT_ID 指定账号")
    if not cli_args:
        raise ValueError("scoped-cli 缺少待执行的 colab 子命令及参数")

    sess_path = _bind_account_state(acct_id)
    try:
        colab_cli_app(args=["--config", str(sess_path), *cli_args], prog_name="colab")
        return 0
    except SystemExit as exc:
        code = exc.code
        if code is None:
            return 0
        if isinstance(code, int):
            return code
        print(str(code), file=sys.stderr, flush=True)
        return 1


def _extract_tail_lines_from_contents(data: dict, line_limit: int) -> list[str]:
    """仅解码 Jupyter Contents 响应末尾字节流，跳过过大前缀并返回最后 N 行。"""
    max_tail_bytes = max(65536, int(line_limit) * 512)
    fmt = str(data.get("format") or "").strip().lower()
    raw = data.get("content", "")
    truncated = False

    if isinstance(raw, dict):
        text = json.dumps(raw, ensure_ascii=False, indent=2)
        if len(text) > max_tail_bytes:
            text = text[-max_tail_bytes:]
            truncated = True
    elif fmt == "base64" and isinstance(raw, str):
        clean_b64 = "".join(raw.split())
        tail_chars = ((max_tail_bytes + 2) // 3) * 4
        if len(clean_b64) > tail_chars:
            start = len(clean_b64) - tail_chars
            rem = start % 4
            if rem:
                start += 4 - rem
            clean_b64 = clean_b64[start:]
            truncated = True
        raw_bytes = base64.b64decode(clean_b64)
        text = raw_bytes.decode("utf-8", errors="replace")
    else:
        text = str(raw or "")
        if len(text) > max_tail_bytes:
            text = text[-max_tail_bytes:]
            truncated = True

    lines = text.splitlines()
    if truncated and len(lines) > 1:
        lines = lines[1:]
    return lines[-line_limit:]


def cmd_contents_tail(args: list[str]) -> int:
    """读取远端实例指定文件（或自动探测 Comfy 日志）的尾部 N 行内容。"""
    scoped_acct, clean_args = _extract_acct_arg(args)
    target_ep = ""
    session_name = "comfy"
    rel_path = ""
    line_limit = 120
    positional: list[str] = []

    idx = 0
    while idx < len(clean_args):
        item = clean_args[idx]
        if item.startswith("--ep="):
            target_ep = item.split("=", 1)[1].strip()
        elif item == "--ep" and idx + 1 < len(clean_args):
            idx += 1
            target_ep = clean_args[idx].strip()
        elif item.startswith("--session="):
            session_name = item.split("=", 1)[1].strip() or "comfy"
        elif item == "--session" and idx + 1 < len(clean_args):
            idx += 1
            session_name = clean_args[idx].strip() or "comfy"
        elif item.startswith("--path="):
            rel_path = item.split("=", 1)[1].strip()
        elif item == "--path" and idx + 1 < len(clean_args):
            idx += 1
            rel_path = clean_args[idx].strip()
        elif item.startswith("--lines="):
            try:
                line_limit = int(item.split("=", 1)[1].strip())
            except ValueError:
                line_limit = 120
        elif item == "--lines" and idx + 1 < len(clean_args):
            idx += 1
            try:
                line_limit = int(clean_args[idx].strip())
            except ValueError:
                line_limit = 120
        else:
            positional.append(item)
        idx += 1

    if positional and not target_ep:
        target_ep = positional[0].strip()
    if len(positional) > 1 and not rel_path:
        rel_path = positional[1].strip()
    if len(positional) > 2:
        try:
            line_limit = int(positional[2].strip())
        except ValueError:
            pass

    line_limit = max(10, min(400, int(line_limit)))

    if scoped_acct:
        _bind_account_state(scoped_acct)

    target, _ = _pick_target_assignment(target_ep, session_name)
    if target is None:
        print(
            "<<<COLAB_CONTENTS_TAIL>>>"
            + json.dumps({"ok": False, "path": "", "lines": [], "message": "无活跃 Colab 实例"}, ensure_ascii=False)
            + "<<<END_COLAB_CONTENTS_TAIL>>>",
            flush=True,
        )
        return 0

    info = target.runtime_proxy_info
    base, tok = info.url.rstrip("/"), info.token
    sess = requests.Session()

    clean_rel = rel_path.lstrip("/")
    if clean_rel:
        candidates = [clean_rel]
    else:
        candidates = [
            "root/comfy_colab.log",
            "root/comfy_minimax.log",
            "root/comfy_bw1.log",
            "tmp/comfy_h3.log",
        ]

    last_status = 404
    for cand in candidates:
        resp = _contents_get(sess, base, tok, cand, content="1", timeout=10.0)
        last_status = int(getattr(resp, "status_code", 0) or 0)
        if last_status != 200:
            continue
        payload = resp.json()
        if not isinstance(payload, dict) or payload.get("type") == "directory":
            continue
        tail_lines = _extract_tail_lines_from_contents(payload, line_limit)
        print(
            "<<<COLAB_CONTENTS_TAIL>>>"
            + json.dumps({"ok": True, "path": cand, "lines": tail_lines}, ensure_ascii=False)
            + "<<<END_COLAB_CONTENTS_TAIL>>>",
            flush=True,
        )
        return 0

    err_msg = f"未找到远端日志文件 (HTTP {last_status})"
    print(
        "<<<COLAB_CONTENTS_TAIL>>>"
        + json.dumps({"ok": False, "path": clean_rel, "lines": [], "message": err_msg}, ensure_ascii=False)
        + "<<<END_COLAB_CONTENTS_TAIL>>>",
        flush=True,
    )
    return 0


_DISPATCH = {
    "probe": cmd_probe,
    "fast-tunnel": cmd_fast_tunnel,
    "contents-put": cmd_contents_put,
    "contents-tail": cmd_contents_tail,
    "install-ext": cmd_install_ext,
    "keepalive": cmd_keepalive,
    "host-metrics": cmd_host_metrics,
    "pool-probe": cmd_pool_probe,
    "scoped-cli": cmd_scoped_cli,
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
