"""Colab 桌宠后端数据采集与控制模块。

负责通过本地 `google-colab-cli` 与 `colab_ctl.py` 无窗口获取：
1. Colab 计算单元 (CU) 余额、当前消耗速率 (`CU/hr`)、预计可用时长；
2. 活跃 GPU/TPU 会话与孤儿会话状态；
3. 本地每日积分消耗小账本与单次连续开机计时（不保存任何账号凭据）；
4. 快捷运维动作：打开 ComfyUI 隧道、跳转 Colab 会话控制台、拉回输出与一键停卡。
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
import urllib.request
import uuid
import webbrowser
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path, PurePosixPath
from typing import Any

import requests
from PIL import Image

try:
    import websocket
except Exception:
    websocket = None

PET_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = PET_DIR.parent
LOCAL_DOWNLOAD_ROOT = Path(
    os.environ.get("COLAB_DOWNLOAD_DIR") or (PACKAGE_DIR / "_downloads" / "all")
)
CTL_SCRIPT = PACKAGE_DIR / "colab_ctl.py"
STATE_FILE = PET_DIR / "pet_state.json"

_APPDATA_DIR = Path(os.environ.get("APPDATA") or (Path.home() / "AppData" / "Roaming"))
COLAB_PY = _APPDATA_DIR / "uv" / "tools" / "google-colab-cli" / "Scripts" / "python.exe"
COLAB_EXE_DEFAULT = Path.home() / ".local" / "bin" / "colab.exe"
RCLONE_EXE = Path.home() / "scoop" / "shims" / "rclone.exe"
RCLONE_CONF = _APPDATA_DIR / "rclone" / "rclone.conf"


def _pick_existing_file(candidates: list[Path], fallback: Path) -> Path:
    """按候选顺序返回首个存在的本地文件，全部未命中时返回兜底路径。"""
    for cand in candidates:
        try:
            if cand and cand.is_file():
                return cand
        except OSError:
            continue
    return fallback


def _discover_colab_exe(custom_path: str = "") -> Path | None:
    """跨设备自适应探测 colab.exe 可执行文件。"""
    cands: list[Path] = []
    for raw in (
        custom_path,
        str(os.environ.get("COLAB_PET_COLAB_EXE") or "").strip(),
        shutil.which("colab") or shutil.which("colab.exe") or "",
    ):
        if raw:
            cands.append(Path(raw))
    appdata = os.environ.get("APPDATA", "")
    localappdata = os.environ.get("LOCALAPPDATA", "")
    cands.append(Path.home() / ".local" / "bin" / "colab.exe")
    if appdata:
        cands.append(Path(appdata) / "uv" / "tools" / "google-colab-cli" / "Scripts" / "colab.exe")
    cands.append(
        Path.home() / "AppData" / "Roaming" / "uv" / "tools" / "google-colab-cli" / "Scripts" / "colab.exe"
    )
    if localappdata:
        cands.append(
            Path(localappdata) / "uv" / "tools" / "google-colab-cli" / "Scripts" / "colab.exe"
        )
    cands.append(Path.home() / ".local" / "pipx" / "venvs" / "google-colab-cli" / "Scripts" / "colab.exe")
    cands.append(COLAB_EXE_DEFAULT)
    for cand in cands:
        try:
            if cand and cand.is_file():
                return cand
        except OSError:
            continue
    return None


def _discover_colab_py(custom_path: str = "", colab_exe: Path | None = None) -> Path:
    """跨设备自适应探测安装有 google-colab-cli 的 python.exe。"""
    cands: list[Path] = []
    for raw in (custom_path, str(os.environ.get("COLAB_PET_COLAB_PY") or "").strip()):
        if raw:
            cands.append(Path(raw))
    appdata = os.environ.get("APPDATA", "")
    localappdata = os.environ.get("LOCALAPPDATA", "")
    if appdata:
        cands.append(Path(appdata) / "uv" / "tools" / "google-colab-cli" / "Scripts" / "python.exe")
    cands.append(
        Path.home() / "AppData" / "Roaming" / "uv" / "tools" / "google-colab-cli" / "Scripts" / "python.exe"
    )
    if localappdata:
        cands.append(
            Path(localappdata) / "uv" / "tools" / "google-colab-cli" / "Scripts" / "python.exe"
        )
    cands.append(Path.home() / ".local" / "pipx" / "venvs" / "google-colab-cli" / "Scripts" / "python.exe")
    if colab_exe is not None:
        try:
            cands.append(Path(colab_exe).resolve().with_name("python.exe"))
        except OSError:
            pass
    cands.append(COLAB_PY)
    return _pick_existing_file(cands, COLAB_PY)


def _discover_rclone_exe(custom_path: str = "") -> Path:
    """跨设备自适应探测 rclone.exe 可执行文件。"""
    cands: list[Path] = []
    for raw in (
        custom_path,
        str(os.environ.get("COLAB_PET_RCLONE_EXE") or "").strip(),
        shutil.which("rclone") or shutil.which("rclone.exe") or "",
    ):
        if raw:
            cands.append(Path(raw))
    localappdata = os.environ.get("LOCALAPPDATA", "")
    cands.append(Path.home() / "scoop" / "shims" / "rclone.exe")
    cands.append(Path.home() / "scoop" / "apps" / "rclone" / "current" / "rclone.exe")
    if localappdata:
        cands.append(Path(localappdata) / "Microsoft" / "WinGet" / "Links" / "rclone.exe")
    cands.append(Path(r"C:\ProgramData\chocolatey\bin\rclone.exe"))
    cands.append(RCLONE_EXE)
    return _pick_existing_file(cands, RCLONE_EXE)


def _discover_rclone_conf(custom_path: str = "", rclone_exe: Path | None = None) -> Path:
    """跨设备自适应探测 rclone.conf 配置文件路径。"""
    cands: list[Path] = []
    for raw in (custom_path, str(os.environ.get("COLAB_PET_RCLONE_CONF") or "").strip()):
        if raw:
            cands.append(Path(raw))
    appdata = os.environ.get("APPDATA", "")
    if appdata:
        cands.append(Path(appdata) / "rclone" / "rclone.conf")
    cands.append(Path.home() / ".config" / "rclone" / "rclone.conf")
    if rclone_exe is not None:
        try:
            exe_dir = Path(rclone_exe).resolve().parent
            cands.append(exe_dir / "rclone.conf")
            if exe_dir.name.lower() == "shims":
                cands.append(exe_dir.parent / "apps" / "rclone" / "current" / "rclone.conf")
        except OSError:
            pass
    cands.append(Path.home() / "scoop" / "apps" / "rclone" / "current" / "rclone.conf")
    cands.append(RCLONE_CONF)
    return _pick_existing_file(cands, RCLONE_CONF)


COLAB_EXE_DEFAULT = _discover_colab_exe() or COLAB_EXE_DEFAULT
COLAB_PY = _discover_colab_py(colab_exe=COLAB_EXE_DEFAULT)
RCLONE_EXE = _discover_rclone_exe()
RCLONE_CONF = _discover_rclone_conf(rclone_exe=RCLONE_EXE)

_INIT_COLAB_PY = COLAB_PY
_INIT_COLAB_EXE = COLAB_EXE_DEFAULT
_INIT_RCLONE_EXE = RCLONE_EXE
_INIT_RCLONE_CONF = RCLONE_CONF

GDRIVE_LORA_REMOTE = "gdrive_save:Colab_ComfyUI_Models/loras"
GDRIVE_WF_REMOTE = "gdrive_save:Colab_ComfyUI_Models/workflows"

MEDIA_EXTS = {
    ".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp",
    ".mp4", ".mov", ".webm", ".mkv",
    ".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus",
}
WORKFLOW_EXTS = {".json"}
MODEL_EXTS = {".safetensors", ".ckpt", ".pt", ".pth"}

# LLM 对话模式支持提取文本与识图引导的后缀集合
LLM_TEXT_EXTS = {
    ".txt", ".md", ".json", ".py", ".yaml", ".yml", ".csv",
    ".log", ".toml", ".ini", ".js", ".ts", ".html", ".css",
}
LLM_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
SUPPORTED_BOOT_PROFILES = ("anima", "minimax", "drama3", "qwen_27b", "qwen_llm")
COMFY_BOOT_PROFILES = ("anima", "minimax", "drama3")
LLM_BOOT_PROFILES = ("qwen_27b", "qwen_llm")
SUPPORTED_LLM_QUANTS = ("UD-Q4_K_XL", "UD-IQ4_XS", "IQ3_XXS")
LLM_QUANT_BASE_SHARDS: dict[str, int] = {
    "UD-Q4_K_XL": 4,
    "UD-IQ4_XS": 3,
    "IQ3_XXS": 2,
}
LLM_27B_BASE_SHARDS = 1


def _normalize_boot_profiles(
    raw_profiles: Any, fallback_profile: str = "anima"
) -> list[str]:
    """清洗并按规范顺序去重开机工作流列表；同时含两个 LLM 变体时保留 180B (qwen_llm)。"""
    cand_list: list[str] = []
    if isinstance(raw_profiles, str):
        items = raw_profiles.split(",")
    elif isinstance(raw_profiles, (list, tuple, set)):
        items = list(raw_profiles)
    else:
        items = []

    for item in items:
        p = str(item or "").strip().lower()
        if p in SUPPORTED_BOOT_PROFILES and p not in cand_list:
            cand_list.append(p)

    if not cand_list:
        fb = str(fallback_profile or "anima").strip().lower()
        cand_list = [fb if fb in SUPPORTED_BOOT_PROFILES else "anima"]

    if "qwen_27b" in cand_list and "qwen_llm" in cand_list:
        cand_list = [p for p in cand_list if p != "qwen_27b"]

    return [p for p in SUPPORTED_BOOT_PROFILES if p in cand_list]

# Windows 下禁止弹出控制台黑框
CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

# 匹配 trycloudflare 快速隧道或用户自配置的 HTTPS 命名隧道域名
_TUNNEL_URL_RE = re.compile(r"https://(?!api\.)(?:[a-z0-9-]+\.)+[a-z]{2,}(?::\d+)?")

# 单次调用 colab_cli 内部 API 的无窗口探针脚本（直读 /tun/m/ccu-info，无凭据时快速返回 auth_missing 避免挂起）
_COLAB_PROBE_SCRIPT = """
import hashlib
import json
import os
import sys
import time
from urllib.parse import urljoin
import keyring
import requests as req_lib
from google.auth.transport.requests import Request as GoogleAuthRequest
from google.oauth2.credentials import Credentials
from colab_cli import auth as colab_auth
from colab_cli.common import state

VAULT_SERVICE_NAME = "google-colab-cli-pet-vault"
need_userinfo = ("--need-userinfo" in sys.argv)
active_acct_id = ""
for arg in sys.argv[1:]:
    if arg.startswith("--active-acct="):
        active_acct_id = arg.split("=", 1)[1].strip()


def _retry(fn, attempts=3, delay=1.0):
    last_exc = None
    for idx in range(attempts):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if idx + 1 < attempts:
                time.sleep(delay)
    raise last_exc


def _check_existing_creds():
    creds = None
    try:
        tok_json = colab_auth._get_token_from_keyring()
        if tok_json:
            creds = Credentials.from_authorized_user_info(
                json.loads(tok_json), colab_auth.PUBLIC_SCOPES
            )
    except Exception:
        creds = None

    if creds is None and os.path.exists(colab_auth.LEGACY_TOKEN_PATH):
        try:
            creds = Credentials.from_authorized_user_file(
                colab_auth.LEGACY_TOKEN_PATH, colab_auth.PUBLIC_SCOPES
            )
            colab_auth._save_token_to_keyring(creds.to_json())
            colab_auth._remove_legacy_token()
        except Exception:
            creds = None

    if creds is None:
        return None, False

    if not creds.valid:
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
                return None, True
        else:
            return None, True
    return creds, False


def _mask_email(email):
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


def _extract_email(creds):
    try:
        r = req_lib.get(
            "https://www.googleapis.com/oauth2/v2/userinfo",
            headers={"Authorization": f"Bearer {creds.token}"},
            timeout=6,
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
            import base64 as _b64
            seg = id_tok.split(".")[1]
            seg += "=" * (-len(seg) % 4)
            claims = json.loads(_b64.urlsafe_b64decode(seg.encode("ascii")).decode("utf-8", errors="ignore"))
            em = str((claims or {}).get("email") or "").strip().lower()
            if em and "@" in em:
                return em
    except Exception:
        pass
    return ""


creds, auth_expired = _check_existing_creds()
if creds is None:
    print(
        "<<<COLAB_PET_JSON>>>"
        + json.dumps({"auth_missing": True, "auth_expired": bool(auth_expired)}, ensure_ascii=False)
        + "<<<END_COLAB_PET_JSON>>>",
        flush=True,
    )
    raise SystemExit(0)

account_meta = None
if need_userinfo:
    email = _extract_email(creds)
    if email:
        acct_id = hashlib.sha256(email.encode("utf-8")).hexdigest()[:12]
        try:
            keyring.set_password(VAULT_SERVICE_NAME, f"acct:{acct_id}", creds.to_json())
        except Exception:
            pass
        account_meta = {
            "id": acct_id,
            "email_masked": _mask_email(email),
            "prefix": email.split("@", 1)[0][:10],
        }


def _fetch_ccu_raw(client):
    url = urljoin(client.colab_domain, "/tun/m/ccu-info")
    resp = client.session.get(
        url,
        params={"authuser": "0"},
        headers={"Accept": "application/json", "X-Colab-Client-Agent": "colab-cli"},
        timeout=15,
    )
    resp.raise_for_status()
    body = client._strip_xssi_prefix(resp.text or "")
    data = json.loads(body) if body else {}
    return data if isinstance(data, dict) else {}


# 仅使用共享读锁读取本地会话列表，禁止调用 sync_sessions() 以免 stdout 打印剪枝日志或冲突写锁
local_sessions = state.store.list()
client = state.client

eligible_gpus = None
eligible_tpus = None
balance = None
rate_hourly = 0.0
assignments_count = 0

try:
    raw_ccu = _retry(lambda: _fetch_ccu_raw(client))
    bal_val = raw_ccu.get("currentBalance", raw_ccu.get("paidComputeUnitsBalance"))
    if bal_val is not None:
        balance = float(bal_val)
        rate_hourly = float(raw_ccu.get("consumptionRateHourly") or 0.0)
        assignments_count = int(raw_ccu.get("assignmentsCount") or 0)
    raw_gpus = raw_ccu.get("eligibleGpus")
    if isinstance(raw_gpus, list):
        eligible_gpus = [str(x).upper() for x in raw_gpus if x]
    raw_tpus = raw_ccu.get("eligibleTpus")
    if isinstance(raw_tpus, list):
        eligible_tpus = [str(x).upper() for x in raw_tpus if x]
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
    variant = variant_obj.name if variant_obj else "DEFAULT"
    shape_obj = getattr(a, "machine_shape", None)
    machine_shape = getattr(shape_obj, "name", "STANDARD") if shape_obj is not None else "STANDARD"
    items.append({
        "name": sname,
        "endpoint": ep,
        "accelerator": accel,
        "variant": variant,
        "machine_shape": machine_shape,
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
""".strip()

# 生成 Google OAuth2 授权链接（prompt=select_account 强制弹出账号选择器，PKCE 校验码仅回传后端内存）
_COLAB_OAUTH_START_SCRIPT = """
import json
import os
from importlib import resources
from colab_cli import auth as colab_auth
from google_auth_oauthlib.flow import InstalledAppFlow

cfg_path = os.path.expanduser("~/.colab-cli-oauth-config.json")
client_config = None
if os.path.exists(cfg_path):
    with open(cfg_path, "r", encoding="utf-8") as fp:
        client_config = json.load(fp)
if not client_config:
    cfg_res = resources.files("colab_cli").joinpath("oauth_config.json")
    if cfg_res.is_file():
        client_config = json.loads(cfg_res.read_text(encoding="utf-8"))
if not client_config:
    raise FileNotFoundError("未找到 colab-cli OAuth 客户端配置")

flow = InstalledAppFlow.from_client_config(
    client_config, colab_auth.PUBLIC_SCOPES, autogenerate_code_verifier=True
)
flow.redirect_uri = colab_auth.REMOTE_REDIRECT_URI
auth_url, _ = flow.authorization_url(prompt="select_account consent", token_usage="remote")
out = {
    "ok": True,
    "auth_url": auth_url,
    "code_verifier": str(getattr(flow, "code_verifier", "") or ""),
}
print("<<<COLAB_OAUTH_START>>>" + json.dumps(out, ensure_ascii=False) + "<<<END_COLAB_OAUTH_START>>>", flush=True)
""".strip()

# 通过 stdin 接收授权码完成 OAuth 交换，并将 Token 安全存入系统凭据库（不向标准输出打印任何 Token）
_COLAB_OAUTH_COMPLETE_SCRIPT = """
import base64
import hashlib
import json
import os
import shutil
import sys
from importlib import resources
from pathlib import Path
import keyring
import requests as req_lib
from google.auth.transport.requests import Request as GoogleAuthRequest
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from colab_cli import auth as colab_auth

VAULT_SERVICE_NAME = "google-colab-cli-pet-vault"
CFG_DIR = Path(os.path.expanduser("~/.config/colab-cli"))
SESS_FILE = CFG_DIR / "sessions.json"
SESS_BACKUP_DIR = CFG_DIR / "pet_sessions_by_acct"


def _mask_email(email):
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


def _extract_email(creds):
    try:
        r = req_lib.get(
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


def _write_private_text(path_obj, content):
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    tmp_p = path_obj.with_suffix(".tmp")
    fd = os.open(str(tmp_p), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
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


def _backup_current_sessions(acct_id):
    if not acct_id or not SESS_FILE.is_file():
        return
    SESS_BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    dst = SESS_BACKUP_DIR / f"{acct_id}.json"
    _write_private_text(dst, SESS_FILE.read_text(encoding="utf-8", errors="ignore"))


def _restore_sessions_for(acct_id):
    SESS_BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    src = SESS_BACKUP_DIR / f"{acct_id}.json" if acct_id else None
    if src is not None and src.is_file():
        _write_private_text(SESS_FILE, src.read_text(encoding="utf-8", errors="ignore"))
    else:
        _write_private_text(SESS_FILE, "{}\\n")


req_in = json.loads(sys.stdin.read() or "{}")
code = str(req_in.get("code") or "").strip()
verifier = str(req_in.get("code_verifier") or "").strip()
prev_acct_id = str(req_in.get("prev_acct_id") or "").strip()

cfg_path = os.path.expanduser("~/.colab-cli-oauth-config.json")
client_config = None
if os.path.exists(cfg_path):
    with open(cfg_path, "r", encoding="utf-8") as fp:
        client_config = json.load(fp)
if not client_config:
    cfg_res = resources.files("colab_cli").joinpath("oauth_config.json")
    if cfg_res.is_file():
        client_config = json.loads(cfg_res.read_text(encoding="utf-8"))
if not client_config:
    raise FileNotFoundError("未找到 colab-cli OAuth 客户端配置")

# 绑定新号前兜底备份当前活跃账号令牌与 sessions.json，防止切换覆盖
try:
    prev_tok_json = colab_auth._get_token_from_keyring()
    if prev_tok_json:
        backup_id = prev_acct_id
        if not backup_id:
            prev_creds = Credentials.from_authorized_user_info(
                json.loads(prev_tok_json), colab_auth.PUBLIC_SCOPES
            )
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
            _backup_current_sessions(backup_id)
except Exception:
    pass

flow = InstalledAppFlow.from_client_config(
    client_config, colab_auth.PUBLIC_SCOPES, autogenerate_code_verifier=False
)
flow.redirect_uri = colab_auth.REMOTE_REDIRECT_URI
if verifier:
    flow.code_verifier = verifier
flow.fetch_token(code=code)
creds = flow.credentials
email = _extract_email(creds)
acct_id = (
    hashlib.sha256(email.encode("utf-8")).hexdigest()[:12]
    if email
    else hashlib.sha256((creds.refresh_token or creds.token or "").encode("utf-8")).hexdigest()[:12]
)
tok_json = creds.to_json()
keyring.set_password(VAULT_SERVICE_NAME, f"acct:{acct_id}", tok_json)
colab_auth._save_token_to_keyring(tok_json)
colab_auth._remove_legacy_token()
_restore_sessions_for(acct_id)

out = {
    "ok": True,
    "acct_id": acct_id,
    "email_masked": _mask_email(email) if email else f"acct-{acct_id[:6]}",
    "prefix": email.split("@", 1)[0][:10] if email else f"Ultra-{acct_id[:4]}",
}
print("<<<COLAB_OAUTH_DONE>>>" + json.dumps(out, ensure_ascii=False) + "<<<END_COLAB_OAUTH_DONE>>>", flush=True)
""".strip()

# 在凭据保险箱与 colab-cli 活跃槽位间极速切换账号并隔离各自的 sessions.json
_COLAB_ACCOUNT_OP_SCRIPT = """
import json
import os
import sys
from pathlib import Path
import keyring
from google.auth.transport.requests import Request as GoogleAuthRequest
from google.oauth2.credentials import Credentials
from colab_cli import auth as colab_auth

VAULT_SERVICE_NAME = "google-colab-cli-pet-vault"
CFG_DIR = Path(os.path.expanduser("~/.config/colab-cli"))
SESS_FILE = CFG_DIR / "sessions.json"
SESS_BACKUP_DIR = CFG_DIR / "pet_sessions_by_acct"


def _write_private_text(path_obj, content):
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    tmp_p = path_obj.with_suffix(".tmp")
    fd = os.open(str(tmp_p), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
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
        if SESS_FILE.is_file():
            SESS_BACKUP_DIR.mkdir(parents=True, exist_ok=True)
            _write_private_text(SESS_BACKUP_DIR / f"{prev_id}.json", SESS_FILE.read_text(encoding="utf-8", errors="ignore"))

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

    src_sess = SESS_BACKUP_DIR / f"{target_id}.json"
    if src_sess.is_file():
        _write_private_text(SESS_FILE, src_sess.read_text(encoding="utf-8", errors="ignore"))
    else:
        _write_private_text(SESS_FILE, "{}\\n")

    print("<<<COLAB_ACCT_OP>>>" + json.dumps({"ok": True, "op": "switch", "target_acct_id": target_id}) + "<<<END_COLAB_ACCT_OP>>>", flush=True)

elif op == "remove":
    try:
        keyring.delete_password(VAULT_SERVICE_NAME, f"acct:{target_id}")
    except Exception:
        pass
    try:
        (SESS_BACKUP_DIR / f"{target_id}.json").unlink(missing_ok=True)
    except Exception:
        pass
    if bool(req_in.get("clear_active")):
        try:
            keyring.delete_password(colab_auth.KEYRING_SERVICE_NAME, colab_auth.KEYRING_USERNAME)
        except Exception:
            pass
        colab_auth._remove_legacy_token()
        _write_private_text(SESS_FILE, "{}\\n")
    print("<<<COLAB_ACCT_OP>>>" + json.dumps({"ok": True, "op": "remove", "target_acct_id": target_id}) + "<<<END_COLAB_ACCT_OP>>>", flush=True)
else:
    raise ValueError(f"未知操作: {op}")
""".strip()

# 通过 Colab Contents HTTP REST 接口免 WebSocket 直读服务元数据契约或隧道地址文件
_COLAB_FAST_TUNNEL_SCRIPT = """
import base64
import json
import re
import sys
import requests
from colab_cli.common import state

TUNNEL_RE = re.compile(r"https://(?!api\\.)(?:[a-z0-9-]+\\.)+[a-z]{2,}(?::\\d+)?")
target_ep = sys.argv[1] if len(sys.argv) > 1 else ""
session_name = sys.argv[2] if len(sys.argv) > 2 else "comfy"


def _decode_contents(data):
    fmt = data.get("format")
    raw = data.get("content", "")
    if isinstance(raw, dict):
        return raw
    if fmt == "base64" and isinstance(raw, str):
        return base64.b64decode(raw).decode("utf-8", errors="ignore")
    return str(raw or "")


local_sessions = state.store.list()
assignments = state.client.list_assignments()
if not assignments:
    print("<<<COLAB_TUNNEL>>>" + json.dumps({"status": "no_session"}, ensure_ascii=False) + "<<<END_COLAB_TUNNEL>>>", flush=True)
    raise SystemExit(0)

target = None
if target_ep:
    for a in assignments:
        if a.endpoint == target_ep:
            target = a
            break

if target is None and session_name in local_sessions:
    wanted_ep = local_sessions[session_name].endpoint
    for a in assignments:
        if a.endpoint == wanted_ep:
            target = a
            break

if target is None:
    target = assignments[0]

info = target.runtime_proxy_info
base = info.url.rstrip("/")
tok = info.token  # 仅在内存中用于鉴权请求，绝不打印或持久化
req_params = {"authuser": "0", "colab-runtime-proxy-token": tok, "content": "1"}

sess = requests.Session()
out = None

# 1) 优先读取 /root/colab_service_meta.json 获取自描述服务类型、模型参数与内存态密钥
resp_meta = sess.get(f"{base}/api/contents/root/colab_service_meta.json", params=req_params, timeout=8)
if resp_meta.status_code == 200:
    try:
        decoded_meta = _decode_contents(resp_meta.json())
        meta_obj = decoded_meta if isinstance(decoded_meta, dict) else json.loads(str(decoded_meta))
    except Exception:
        meta_obj = None
    if isinstance(meta_obj, dict):
        api_key = str(meta_obj.get("api_key") or "").strip()
        clean_meta = {k: v for k, v in meta_obj.items() if k != "api_key"}
        kind_raw = str(clean_meta.get("kind") or "llm").strip().lower()
        if kind_raw not in ("comfy", "llm", "dual"):
            kind_raw = "llm"
        clean_meta["kind"] = kind_raw
        clean_meta.setdefault("port", 18080 if kind_raw == "llm" else 8188)

        comfy_url_val = None
        llm_url_val = None
        if kind_raw == "dual":
            m_cu = TUNNEL_RE.search(str(meta_obj.get("comfy_tunnel_url") or meta_obj.get("tunnel_url") or ""))
            comfy_url_val = m_cu.group(0) if m_cu else None
            if not comfy_url_val:
                resp_txt_fb = sess.get(f"{base}/api/contents/root/comfy_tunnel_url.txt", params=req_params, timeout=6)
                if resp_txt_fb.status_code == 200:
                    m_fb = TUNNEL_RE.search(str(_decode_contents(resp_txt_fb.json()) or ""))
                    if m_fb:
                        comfy_url_val = m_fb.group(0)
            m_lu = TUNNEL_RE.search(str(meta_obj.get("llm_tunnel_url") or ""))
            llm_url_val = m_lu.group(0) if m_lu else None
            if not llm_url_val:
                resp_llm_fb = sess.get(f"{base}/api/contents/root/llm_tunnel_url.txt", params=req_params, timeout=6)
                if resp_llm_fb.status_code == 200:
                    m_lfb = TUNNEL_RE.search(str(_decode_contents(resp_llm_fb.json()) or ""))
                    if m_lfb:
                        llm_url_val = m_lfb.group(0)
            if comfy_url_val:
                clean_meta["comfy_tunnel_url"] = comfy_url_val
            if llm_url_val:
                clean_meta["llm_tunnel_url"] = llm_url_val
            tun_url = comfy_url_val or llm_url_val
        else:
            m_meta_url = TUNNEL_RE.search(str(meta_obj.get("tunnel_url") or ""))
            tun_url = m_meta_url.group(0) if m_meta_url else None
            if not tun_url:
                # 若元数据尚未填入隧道或隧道已由外部刷新，回退读取 comfy_tunnel_url.txt 补齐 URL
                resp_txt_fb = sess.get(f"{base}/api/contents/root/comfy_tunnel_url.txt", params=req_params, timeout=6)
                if resp_txt_fb.status_code == 200:
                    txt_fb = str(_decode_contents(resp_txt_fb.json()) or "")
                    m_fb = TUNNEL_RE.search(txt_fb)
                    if m_fb:
                        tun_url = m_fb.group(0)
        if tun_url:
            clean_meta["tunnel_url"] = tun_url
            out = {
                "status": "ok",
                "endpoint": target.endpoint,
                "url": tun_url,
                "meta": clean_meta,
                "api_key": api_key,
            }
        else:
            out = {
                "status": "missing",
                "endpoint": target.endpoint,
                "meta": clean_meta,
                "api_key": api_key,
            }
elif resp_meta.status_code not in (404,):
    resp_meta.raise_for_status()

# 2) 若不存在 colab_service_meta.json (404)，回退读取传统 ComfyUI 隧道地址文件
if out is None:
    comfy_meta = {"kind": "comfy", "port": 8188}
    resp = sess.get(
        f"{base}/api/contents/root/comfy_tunnel_url.txt",
        params=req_params,
        timeout=8,
    )
    if resp.status_code == 404:
        out = {"status": "missing", "endpoint": target.endpoint, "meta": comfy_meta}
    else:
        resp.raise_for_status()
        text = str(_decode_contents(resp.json()) or "")
        m = TUNNEL_RE.search(text)
        if m:
            out = {"status": "ok", "endpoint": target.endpoint, "url": m.group(0), "meta": comfy_meta}
        else:
            out = {"status": "missing", "endpoint": target.endpoint, "meta": comfy_meta}

print("<<<COLAB_TUNNEL>>>" + json.dumps(out, ensure_ascii=False) + "<<<END_COLAB_TUNNEL>>>", flush=True)
""".strip()

# 通过 Colab Contents HTTP PUT 将工作流 JSON 直写云端实例 workflows 目录
_COLAB_CONTENTS_PUT_SCRIPT = """
import base64
import json
import sys
import requests
from colab_cli.common import state

target_ep = sys.argv[1]
session_name = sys.argv[2]
rel_path = sys.argv[3].lstrip("/")
b64_data = sys.stdin.read().strip()

local_sessions = state.store.list()
assignments = state.client.list_assignments()
if not assignments:
    raise RuntimeError("无活跃 Colab 实例")

target = next((a for a in assignments if target_ep and a.endpoint == target_ep), None)
if target is None and session_name in local_sessions:
    wanted = local_sessions[session_name].endpoint
    target = next((a for a in assignments if a.endpoint == wanted), None)
if target is None:
    target = assignments[0]

info = target.runtime_proxy_info
base = info.url.rstrip("/")
tok = info.token
resp = requests.put(
    f"{base}/api/contents/{rel_path}",
    params={"authuser": "0", "colab-runtime-proxy-token": tok},
    json={"type": "file", "format": "base64", "content": b64_data},
    timeout=15,
)
resp.raise_for_status()
print("OK")
""".strip()

# 让运行中的 Colab 实例从 Google Drive 增量拉取刚上传的 LoRA
_COLAB_REMOTE_LORA_PULL_SCRIPT = """
import sys
from colab_cli.common import state
from colab_cli.runtime import ColabRuntime

session_name = sys.argv[1]
src_remote = sys.argv[2]
dst_local = sys.argv[3]

s = state.get_session(session_name)
if s is None:
    raise RuntimeError(f"未找到本地会话 '{session_name}'")

rt = ColabRuntime(s.url, s.token, session_name=session_name, kernel_id=s.kernel_id, session_id=s.session_id)
try:
    ws = rt.kernel_client._manager.client
    if not ws.connection_ready.is_set() and not ws.connection_ready.wait(timeout=25.0):
        raise RuntimeError("WebSocket 握手超时")
    code = f\"\"\"
import os, shutil, subprocess
os.makedirs(os.path.dirname({dst_local!r}), exist_ok=True)
rbin = shutil.which("rclone") or "/usr/local/bin/rclone"
subprocess.run([rbin, "copyto", "--config", "/tmp/rclone.conf", "--disable-http2", {src_remote!r}, {dst_local!r}], check=True)
print("REMOTE_LORA_READY")
\"\"\"
    outputs = rt.execute_code(code, timeout=180.0)
    errs = [o for o in outputs if o.get("output_type") == "error"]
    if errs:
        raise RuntimeError(str(errs[0]))
    print("REMOTE_LORA_READY")
finally:
    rt.stop()
""".strip()

DEFAULT_PREFS: dict[str, Any] = {
    "x": None,
    "y": None,
    "scale": 1.0,
    "on_top": True,
    "bubble_open": True,
    "left_flipped": False,
    "dock_edge": None,
    "edge_auto_hide": True,
    "poll_interval_sec": 60,
    "warn_hours": 10.0,
    "warn_balance": 50.0,
    "long_run_warn_minutes": 180,
    "auto_pull_outputs": True,
    "keep_alive": True,
    "idle_auto_stop_minutes": 60,
    "boot_profile": "anima",
    "boot_profiles": ["anima"],
    "boot_gpu": "L4",
    "boot_high_mem": False,
    "boot_minimax_r2v": True,
    "boot_minimax_pe": True,
    "boot_llm_quant": "UD-Q4_K_XL",
    "boot_llm_ctx": 32768,
    "boot_llm_mmproj": True,
    "boot_llm27_quant": "UD-Q4_K_XL",
    "boot_llm27_ctx": 65536,
    "boot_llm27_mmproj": True,
    "active_account_id": "",
    "colab_exe_path": "",
    "colab_py_path": "",
    "rclone_exe_path": "",
    "rclone_conf_path": "",
}

_ACCOUNT_LEDGER_KEYS = (
    "date",
    "day_start_balance",
    "day_consumed_cu",
    "day_topup_cu",
    "last_balance",
    "last_delta",
    "baseline_balance",
    "active_since_ts",
    "active_endpoint",
    "session_start_balance",
    "last_rate_hourly",
    "last_assignments_count",
    "last_sessions",
    "last_eligible_gpus",
    "last_eligible_tpus",
    "last_updated_ts",
    "tunnel_cache",
)
_CUSTOM_TUNNEL_HOST_RE = re.compile(r"^[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?$", re.IGNORECASE)
_RCLONE_GDRIVE_SECTION_RE = re.compile(r"^\s*\[gdrive_save\]\s*$", re.MULTILINE)
_RCLONE_INI_SECTION_RE = re.compile(r"^\s*\[([^\]]+)\]\s*$", re.MULTILINE)

# 支持桌面一键选卡开机的 GPU 型号与高峰期缺卡临时冷却时间（10 分钟）
SUPPORTED_BOOT_GPUS = ("T4", "L4", "A100", "G4", "H100")
# 账号已通过 /tun/m/ccu-info 实测校准过的单卡基准费率（后续单卡运行时自动动态刷新）
DEFAULT_MEASURED_GPU_RATES: dict[str, float] = {
    "T4:STD": 1.15,
    "L4:STD": 1.94,
    "G4:STD": 10.05,
}
SCARCE_GPU_TTL_SEC = 600.0
_STEP_BANNER_RE = re.compile(r"===\s*\[(\d+)/(\d+)\]\s*(.+?)\s*===")
_PCT_TOKEN_RE = re.compile(r"\((\d{1,3}(?:\.\d+)?)%\)")
_DL_SPEED_RE = re.compile(r"DL:\s*([0-9.]+\s*[KMGTP]?i?B(?:/s)?)", re.IGNORECASE)
_DEPLOY_DL_TOTAL_RE = re.compile(r"DEPLOY_DL_TOTAL:\s*(\d+)")
_DEPLOY_PHASE_RE = re.compile(r"DEPLOY_PHASE\s+(\d+)/(\d+):\s*(\S+)")
_DEPLOY_TUNNEL_LINE_RE = re.compile(
    r"(?:COMFYUI_TUNNEL_URL|SERVICE_TUNNEL_URL):\s*(https://(?!api\.)(?:[a-z0-9-]+\.)+[a-z]{2,}(?::\d+)?)"
)
_DEPLOY_LLM_TUNNEL_LINE_RE = re.compile(
    r"LLM_TUNNEL_URL:\s*(https://(?!api\.)(?:[a-z0-9-]+\.)+[a-z]{2,}(?::\d+)?)"
)
_CF_SECRET_FILE = PACKAGE_DIR / ".cf_tunnel_secret.json"
_DEPLOY_STEP_BASE_PCT = {1: 16, 2: 23, 3: 31, 4: 42, 5: 78, 6: 86, 7: 95}

# 防断连保活心跳周期（10 分钟）、失败重试冷却（2 分钟）与智能空闲熔断（45 分钟）
KEEPALIVE_INTERVAL_SEC = 600.0
KEEPALIVE_RETRY_COOLDOWN_SEC = 120.0
KEEPALIVE_MAX_IDLE_SEC = 45 * 60.0
SUPPORTED_IDLE_STOP_MINUTES = (0, 60)

_CST_TZ = timezone(timedelta(hours=8))
_TS_SUFFIX_RE = re.compile(r"_\d{8}-\d{6}(?:-\d+)?$")
_PRESERVE_ORIGINAL_SUB_RE = re.compile(r"(?:^|/)drama3_voices$", re.IGNORECASE)

# 向 Google Colab 官方通道发送轻量三层保活心跳（TFE keep-alive -> 代理内核列表 -> 空闲内核空操作）
_COLAB_KEEPALIVE_SCRIPT = """
import json
import requests
from colab_cli.common import state
from colab_cli.runtime import ColabRuntime

client = state.client
assignments = client.list_assignments()
if not assignments:
    print("<<<COLAB_KEEPALIVE>>>" + json.dumps({"ok": True, "pinged": 0}, ensure_ascii=False) + "<<<END_COLAB_KEEPALIVE>>>", flush=True)
    raise SystemExit(0)

local_sessions = state.store.list()
sess_by_ep = {s.endpoint: s for s in local_sessions.values()}
pinged = 0

for a in assignments:
    touched = False
    # 1) Colab 官方 TFE 保活端点，刷新服务端 LastActiveTime
    try:
        r1 = client.session.get(
            f"{client.colab_domain.rstrip('/')}/tun/m/{a.endpoint}/keep-alive/",
            params={"authuser": "0"},
            headers={
                "X-Colab-Tunnel": "Google",
                "X-Colab-Client-Agent": "colab-cli",
                "Accept": "application/json",
            },
            timeout=8,
        )
        if r1.ok:
            touched = True
    except Exception:
        pass

    # 2) Runtime Proxy /api/kernels 探活，并仅在存在空闲内核时附带轻量执行心跳
    proxy_url = a.runtime_proxy_info.url.rstrip("/")
    tok = a.runtime_proxy_info.token
    idle_kid = None
    try:
        r2 = requests.get(
            f"{proxy_url}/api/kernels",
            params={"authuser": "0", "colab-runtime-proxy-token": tok},
            timeout=8,
        )
        if r2.ok:
            touched = True
            kernels = r2.json()
            if isinstance(kernels, list):
                for k in kernels:
                    if isinstance(k, dict) and k.get("execution_state") == "idle" and k.get("id"):
                        idle_kid = str(k["id"])
                        break
    except Exception:
        pass

    # 3) 空闲内核执行一次毫秒级纯 CPU 空操作，刷新 Jupyter 内核 last_activity（绝不抢占正在执行命令的内核）
    if idle_kid:
        sess = sess_by_ep.get(a.endpoint)
        sid = getattr(sess, "session_id", None) if sess else None
        rt = ColabRuntime(proxy_url, tok, kernel_id=idle_kid, session_id=sid)
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
""".strip()

# 注入到云端 ComfyUI 前端的「一键直铺画布」扩展脚本（兼容原生 WebSocket 与存量热挂载双模式）
_COMFY_DIRECT_CANVAS_JS = r"""/* colab_pet_direct_canvas v2 */
(async function () {
  if (window.__COLAB_PET_DIRECT_CANVAS_V2__) return;
  window.__COLAB_PET_DIRECT_CANVAS_V2__ = true;
  window.__COLAB_PET_DIRECT_CANVAS_V1__ = true;

  let modApp = null;
  let modApi = null;
  try {
    modApp = await import("../../scripts/app.js");
  } catch (e) {}
  try {
    modApi = await import("../../scripts/api.js");
  } catch (e) {}

  const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
  // 后台/隐藏标签页下浏览器会暂停 requestAnimationFrame，改为限时降级避免队列永久卡死
  const nextFrameSafe = () => {
    if (typeof document !== "undefined" && document.hidden) {
      return sleep(32);
    }
    if (typeof requestAnimationFrame !== "function") {
      return sleep(32);
    }
    return Promise.race([
      new Promise((resolve) => requestAnimationFrame(() => resolve())),
      sleep(80)
    ]);
  };

  const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
  const generateUuidV4 = () => {
    try {
      if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
        return crypto.randomUUID();
      }
    } catch (e) {}
    return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
      const r = (Math.random() * 16) | 0;
      const v = c === "x" ? r : (r & 0x3) | 0x8;
      return v.toString(16);
    });
  };

  const ensureValidUuid = (graphData) => {
    if (!graphData || typeof graphData !== "object") return graphData;
    const rawId = typeof graphData.id === "string" ? graphData.id.trim() : "";
    if (!rawId || !UUID_RE.test(rawId)) {
      graphData.id = generateUuidV4();
    }
    return graphData;
  };

  const sanitizeWfName = (raw) => {
    const clean = String(raw || "").trim().replace(/^workflows\//i, "").split("/").pop();
    return clean ? (clean.toLowerCase().endsWith(".json") ? clean : `${clean}.json`) : "";
  };

  // 模块加载第一时间捕获并剥离 URL 中的一次性直铺参数，防止用户按 F5 覆盖画布修改
  const initUrl = new URL(window.location.href);
  let pendingWfName = sanitizeWfName(initUrl.searchParams.get("pet_wf") || "");
  let pendingExtraNames = (initUrl.searchParams.get("pet_extra") || "")
    .split(",")
    .map(sanitizeWfName)
    .filter((n) => n && n !== pendingWfName);
  if (initUrl.searchParams.has("pet_wf") || initUrl.searchParams.has("pet_extra") || initUrl.searchParams.has("pet_ts")) {
    initUrl.searchParams.delete("pet_wf");
    initUrl.searchParams.delete("pet_extra");
    initUrl.searchParams.delete("pet_ts");
    try {
      window.history.replaceState(null, "", initUrl.toString());
    } catch (e) {}
  }

  function getApp() {
    return (modApp && modApp.app) || window.comfyAPI?.app?.app || window.app || null;
  }

  function getApi() {
    return (modApi && modApi.api) || window.comfyAPI?.api?.api || getApp()?.api || null;
  }

  async function waitForCanvasReady(timeoutMs = 30000) {
    const start = Date.now();
    while (Date.now() - start < timeoutMs) {
      const app = getApp();
      const splashEl = typeof document !== "undefined" ? document.getElementById("splash-loader") : null;
      const spinnerBusy = Boolean(app?.extensionManager?.spinner);
      const cfgBusy = Boolean(app?.configuringGraphLevel);
      if (
        app &&
        app.rootGraph &&
        typeof app.loadGraphData === "function" &&
        !spinnerBusy &&
        !splashEl &&
        !cfgBusy
      ) {
        try {
          if (app.graphLoadQueue) {
            await Promise.resolve(app.graphLoadQueue);
          }
        } catch (e) {}
        await nextFrameSafe();
        await sleep(80);
        const appAfter = getApp();
        if (
          appAfter &&
          appAfter.rootGraph &&
          !Boolean(appAfter.extensionManager?.spinner) &&
          !Boolean(appAfter.configuringGraphLevel)
        ) {
          return true;
        }
      }
      await sleep(100);
    }
    const appEnd = getApp();
    return Boolean(appEnd && appEnd.rootGraph && typeof appEnd.loadGraphData === "function");
  }

  async function fetchWorkflowJson(wfPath) {
    const encoded = encodeURIComponent(wfPath);
    const urls = [
      `/api/userdata/${encoded}?t=${Date.now()}`,
      `/userdata/${encoded}?t=${Date.now()}`
    ];
    for (const u of urls) {
      try {
        const resp = await fetch(u, { cache: "no-store" });
        if (resp.ok) {
          return await resp.json();
        }
      } catch (e) {}
    }
    throw new Error(`无法读取工作流文件: ${wfPath}`);
  }

  const handledCmdIds = new Set();
  let openQueue = Promise.resolve();

  async function _doOpenWorkflowOntoCanvas(rawName, extraNames = [], showNotice = true) {
    const cleanName = sanitizeWfName(rawName);
    if (!cleanName) return false;
    const ready = await waitForCanvasReady();
    if (!ready) return false;

    const app = getApp();
    if (!app || typeof app.loadGraphData !== "function") return false;
    const wfStore = app.extensionManager?.workflow;
    const wfPath = `workflows/${cleanName}`;

    try {
      if (typeof wfStore?.syncWorkflows === "function") {
        await wfStore.syncWorkflows();
      }
    } catch (e) {}

    const graphData = await fetchWorkflowJson(wfPath);
    if (!graphData || typeof graphData !== "object") return false;
    ensureValidUuid(graphData);

    try {
      const bgPaths = (extraNames || [])
        .map(sanitizeWfName)
        .filter((n) => n && n !== cleanName)
        .map((n) => `workflows/${n}`)
        .filter((p) => Boolean(wfStore?.getWorkflowByPath?.(p)));
      if (bgPaths.length && typeof wfStore?.openWorkflowsInBackground === "function") {
        wfStore.openWorkflowsInBackground({ right: bgPaths });
      }
    } catch (e) {}

    const wfObj = wfStore?.getWorkflowByPath?.(wfPath) || null;
    if (wfObj) {
      // 严禁在 loadGraphData 之前手动调用 wfStore.openWorkflow(wfObj)，否则会干扰 beforeLoadNewGraph 状态暂存导致旧画布覆盖新工作流
      await app.loadGraphData(graphData, true, true, wfObj);
      try {
        if (
          wfStore &&
          typeof wfStore.isActive === "function" &&
          !wfStore.isActive(wfObj) &&
          typeof wfStore.openWorkflow === "function"
        ) {
          await wfStore.openWorkflow(wfObj);
        }
      } catch (e) {}
      try {
        wfObj.changeTracker?.reset?.();
        wfObj.isModified = false;
      } catch (e) {}
    } else {
      await app.loadGraphData(graphData, true, true, cleanName.replace(/\.json$/i, ""));
    }

    try {
      app.canvas?.setDirty?.(true, true);
    } catch (e) {}

    if (showNotice) {
      try {
        app.extensionManager?.toast?.add?.({
          severity: "success",
          summary: "✨ 已直铺工作流到画布",
          detail: cleanName,
          life: 2800
        });
      } catch (e) {}
    }
    return true;
  }

  function enqueueOpenWorkflow(rawName, extraNames = [], showNotice = true) {
    const task = openQueue
      .then(
        () => _doOpenWorkflowOntoCanvas(rawName, extraNames, showNotice),
        () => _doOpenWorkflowOntoCanvas(rawName, extraNames, showNotice)
      )
      .catch((err) => {
        console.warn("[colab_pet_direct_canvas] 直铺工作流失败:", err);
        return false;
      });
    openQueue = task.then(() => undefined, () => undefined);
    return task;
  }

  async function postNativeAck(id, name) {
    const payload = JSON.stringify({ id, name, ts: Date.now() / 1000 });
    for (const ep of ["/colab_pet/ack_workflow", "/api/colab_pet/ack_workflow"]) {
      try {
        const resp = await fetch(ep, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: payload
        });
        if (resp.ok) return;
      } catch (e) {}
    }
  }

  async function postLegacyAck(id, name) {
    try {
      await fetch("/userdata/colab_pet_bridge_ack.json?overwrite=true", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id, name, ts: Date.now() / 1000 })
      });
    } catch (e) {}
  }

  async function startLivePushBridge() {
    const api = getApi();
    let nativeOk = false;

    try {
      const pingResp = await fetch(`/colab_pet/ping?t=${Date.now()}`, { cache: "no-store" });
      if (pingResp.ok) {
        const pdata = await pingResp.json();
        nativeOk = Boolean(pdata && pdata.ok);
      }
    } catch (e) {}

    if (nativeOk && api && typeof api.addEventListener === "function") {
      api.addEventListener("colab_pet.open_workflow", async (ev) => {
        const d = ev?.detail || {};
        const cmdId = String(d.id || "");
        const wfName = sanitizeWfName(d.name || "");
        if (!wfName) return;
        if (cmdId) {
          if (handledCmdIds.has(cmdId)) return;
          handledCmdIds.add(cmdId);
        }
        // 后台标签页略微让步前台可见标签页优先接管，且自身不阻塞可正常完成加载
        if (typeof document !== "undefined" && document.hidden) {
          await sleep(180);
        }
        const extras = Array.isArray(d.extra_names) ? d.extra_names : [];
        const ok = await enqueueOpenWorkflow(wfName, extras, true);
        if (ok) {
          if (cmdId) await postNativeAck(cmdId, wfName);
        } else if (cmdId) {
          handledCmdIds.delete(cmdId);
        }
      });
      return;
    }

    // 存量实例未加载自定义节点路由时，回退轻量指令轮询（存储在 userdata 根目录，不污染 workflows 栏）
    const bootTs = Date.now() / 1000 - 2.0;
    let pollBusy = false;
    const pollCmd = async () => {
      if (pollBusy) return;
      pollBusy = true;
      try {
        const r = await fetch(`/userdata/colab_pet_bridge_cmd.json?t=${Date.now()}`, { cache: "no-store" });
        if (!r.ok) return;
        const cmd = await r.json();
        if (!cmd || !cmd.id || !cmd.name) return;
        const cmdId = String(cmd.id);
        if (handledCmdIds.has(cmdId)) return;
        if (Number(cmd.ts || 0) < bootTs) return;
        handledCmdIds.add(cmdId);
        const wfName = sanitizeWfName(cmd.name);
        if (!wfName) return;
        if (typeof document !== "undefined" && document.hidden) {
          await sleep(180);
        }
        const extras = Array.isArray(cmd.extra_names) ? cmd.extra_names : [];
        const ok = await enqueueOpenWorkflow(wfName, extras, true);
        if (ok) {
          await postLegacyAck(cmdId, wfName);
        } else {
          handledCmdIds.delete(cmdId);
        }
      } catch (e) {
      } finally {
        pollBusy = false;
      }
    };

    window.addEventListener("focus", () => pollCmd());
    document.addEventListener("visibilitychange", () => {
      if (!document.hidden) pollCmd();
    });
    setInterval(pollCmd, 1200);
  }

  let setupStarted = false;
  async function runStartupHooks() {
    if (setupStarted) return;
    setupStarted = true;
    startLivePushBridge();
    if (pendingWfName) {
      const targetWf = pendingWfName;
      const targetExtras = pendingExtraNames;
      pendingWfName = "";
      pendingExtraNames = [];
      await enqueueOpenWorkflow(targetWf, targetExtras, true);
    }
  }

  const app = getApp();
  if (app && typeof app.registerExtension === "function") {
    app.registerExtension({
      name: "ColabPet.DirectCanvas",
      async setup() {
        runStartupHooks();
      }
    });
  } else {
    waitForCanvasReady().then(() => runStartupHooks());
  }
})();
""".strip()

# 云端 ComfyUI 原生自定义节点源码（提供浏览器客户端感知、WS 实时直铺与 ACK 确认端点）
_COMFY_BRIDGE_NODE_INIT_PY = '''"""Colab 桌宠直铺画布服务端桥接扩展。"""
import time
from collections import OrderedDict
from aiohttp import web
import server

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
WEB_DIRECTORY = "./js"

_ACK_MAP = OrderedDict()


def _record_ack(req_id: str, name: str = "") -> None:
    if not req_id:
        return
    _ACK_MAP[req_id] = {"id": req_id, "name": name, "ts": time.time()}
    _ACK_MAP.move_to_end(req_id)
    while len(_ACK_MAP) > 64:
        _ACK_MAP.popitem(last=False)


def _count_browser_clients(ps) -> int:
    sockets = getattr(ps, "sockets", None) or {}
    count = 0
    for cid, ws in list(sockets.items()):
        if str(cid).startswith("colab_pet_"):
            continue
        if ws is None or getattr(ws, "closed", False):
            continue
        count += 1
    return count


def _register_routes() -> None:
    ps = getattr(server.PromptServer, "instance", None)
    if ps is None or getattr(ps, "_colab_pet_routes_ready", False):
        return
    ps._colab_pet_routes_ready = True
    routes = ps.routes

    @routes.get("/colab_pet/ping")
    async def colab_pet_ping(_request):
        return web.json_response({
            "ok": True,
            "version": 2,
            "browser_clients": _count_browser_clients(ps),
        })

    @routes.post("/colab_pet/open_workflow")
    async def colab_pet_open_workflow(request):
        try:
            data = await request.json()
        except Exception:
            data = {}
        req_id = str(data.get("id") or "")
        name = str(data.get("name") or "")
        extra_names = data.get("extra_names") if isinstance(data.get("extra_names"), list) else []
        clients = _count_browser_clients(ps)
        if name and clients > 0:
            ps.send_sync(
                "colab_pet.open_workflow",
                {"id": req_id, "name": name, "extra_names": extra_names},
            )
        return web.json_response({
            "ok": True,
            "id": req_id,
            "browser_clients": clients,
        })

    @routes.post("/colab_pet/ack_workflow")
    async def colab_pet_ack_post(request):
        try:
            data = await request.json()
        except Exception:
            data = {}
        req_id = str(data.get("id") or "")
        name = str(data.get("name") or "")
        _record_ack(req_id, name)
        return web.json_response({"ok": True, "id": req_id})

    @routes.get("/colab_pet/ack_workflow")
    async def colab_pet_ack_get(request):
        req_id = str(request.query.get("id") or "")
        info = _ACK_MAP.get(req_id)
        return web.json_response({
            "ok": True,
            "acked": info is not None,
            "info": info,
        })


try:
    _register_routes()
except Exception:
    pass
'''.strip()

# 免重启热安装直铺画布前端扩展与自定义节点到已运行 Colab 实例的 Contents REST 脚本
_COLAB_INSTALL_CANVAS_EXT_SCRIPT = """
import json
import re
import sys
import requests
from colab_cli.common import state

target_ep = sys.argv[1] if len(sys.argv) > 1 else ""
session_name = sys.argv[2] if len(sys.argv) > 2 else "comfy"
payload = json.loads(sys.stdin.read())
js_b64 = payload["js_b64"]
py_b64 = payload["py_b64"]

local_sessions = state.store.list()
assignments = state.client.list_assignments()
if not assignments:
    raise RuntimeError("无活跃 Colab 实例")

target = next((a for a in assignments if target_ep and a.endpoint == target_ep), None)
if target is None and session_name in local_sessions:
    wanted = local_sessions[session_name].endpoint
    target = next((a for a in assignments if a.endpoint == wanted), None)
if target is None:
    target = assignments[0]

info = target.runtime_proxy_info
base = info.url.rstrip("/")
tok = info.token


def _get_json(rel, content="1"):
    return requests.get(
        f"{base}/api/contents/{rel.lstrip('/')}",
        params={"authuser": "0", "colab-runtime-proxy-token": tok, "content": content},
        timeout=10,
    )


def _ensure_dir(rel_dir):
    parts = [p for p in rel_dir.strip("/").split("/") if p]
    cur = ""
    for part in parts:
        cur = f"{cur}/{part}" if cur else part
        r = _get_json(cur, content="0")
        if r.status_code == 200:
            continue
        if r.status_code == 404:
            pr = requests.put(
                f"{base}/api/contents/{cur}",
                params={"authuser": "0", "colab-runtime-proxy-token": tok},
                json={"type": "directory"},
                timeout=10,
            )
            if pr.status_code not in (200, 201, 409):
                pr.raise_for_status()
        else:
            r.raise_for_status()


def _put_b64(rel_file, b64_str):
    parent = rel_file.strip("/").rsplit("/", 1)[0] if "/" in rel_file.strip("/") else ""
    if parent:
        _ensure_dir(parent)
    pr = requests.put(
        f"{base}/api/contents/{rel_file.lstrip('/')}",
        params={"authuser": "0", "colab-runtime-proxy-token": tok},
        json={"type": "file", "format": "base64", "content": b64_str},
        timeout=15,
    )
    pr.raise_for_status()


web_root = ""
for log_rel in ("root/comfy_colab.log", "root/comfy_bw1.log", "root/comfy_minimax.log", "tmp/comfy_h3.log", "tmp/comfyui.log"):
    r = _get_json(log_rel, content="1")
    if r.status_code == 200:
        d = r.json()
        raw = d.get("content", "")
        if d.get("format") == "base64":
            import base64
            txt = base64.b64decode(raw).decode("utf-8", errors="ignore")
        else:
            txt = str(raw or "")
        m = re.search(r"\\[Prompt Server\\] web root:\\s*(\\S+)", txt)
        if m:
            web_root = m.group(1).strip().lstrip("/")
            break

if not web_root:
    r = _get_json("usr/local/lib", content="1")
    if r.status_code == 200:
        for item in (r.json().get("content") or []):
            name = str(item.get("name") or "")
            if name.startswith("python3"):
                cand = f"usr/local/lib/{name}/dist-packages/comfyui_frontend_package/static"
                cr = _get_json(cand, content="0")
                if cr.status_code == 200:
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
""".strip()


def extract_png_comfy_workflow(path: Path) -> str | None:
    """检查 PNG 元数据中是否内嵌合法 ComfyUI 工作流。

    同时支持网页端 `workflow`（画布格式）与脚本端 `prompt`（API 节点拓扑格式）；
    当两者并存时优先返回保留画布坐标与连线布局的 `workflow`。
    """
    try:
        with Image.open(path) as img:
            raw_wf = img.info.get("workflow")
            raw_prompt = img.info.get("prompt")
    except Exception:
        return None

    # 1) 优先提取网页端 workflow 画布格式
    if isinstance(raw_wf, str) and raw_wf.strip():
        try:
            parsed_wf = json.loads(raw_wf)
            if isinstance(parsed_wf, dict) and ("nodes" in parsed_wf or "links" in parsed_wf):
                return raw_wf
        except Exception:
            pass

    # 2) 回退兼容脚本 API 提交生成的 prompt 节点拓扑格式
    if isinstance(raw_prompt, str) and raw_prompt.strip():
        try:
            parsed_prompt = json.loads(raw_prompt)
            if isinstance(parsed_prompt, dict) and any(
                isinstance(node, dict)
                and isinstance(node.get("class_type"), str)
                and isinstance(node.get("inputs"), dict)
                for node in parsed_prompt.values()
            ):
                return raw_prompt
        except Exception:
            pass

    return None


def _is_api_prompt_dict(data: Any) -> bool:
    """判断 JSON 对象是否为 ComfyUI API 节点拓扑格式（prompt）。"""
    if not isinstance(data, dict) or "nodes" in data or "links" in data:
        return False
    return any(
        isinstance(node, dict)
        and isinstance(node.get("class_type"), str)
        and bool(node.get("class_type", "").strip())
        and isinstance(node.get("inputs"), dict)
        for node in data.values()
    )


def _is_prompt_link_ref(val: Any, valid_nodes: dict[str, dict[str, Any]]) -> bool:
    """判断 API prompt 输入值是否为合法的 `[源节点ID, 输出槽位]` 连线引用。"""
    return (
        isinstance(val, (list, tuple))
        and len(val) == 2
        and isinstance(val[0], (str, int))
        and not isinstance(val[0], bool)
        and isinstance(val[1], int)
        and not isinstance(val[1], bool)
        and val[1] >= 0
        and str(val[0]) in valid_nodes
    )


def _default_widget_value(raw_type: Any, cfg: dict[str, Any]) -> Any:
    """根据 /object_info 字段定义推算缺省控件值。"""
    if "default" in cfg:
        return cfg["default"]
    if isinstance(raw_type, list):
        return raw_type[0] if raw_type else ""
    if raw_type == "COMBO":
        opts = cfg.get("options")
        if isinstance(opts, list) and opts:
            return opts[0]
        return ""
    if raw_type == "INT":
        return 0
    if raw_type == "FLOAT":
        return 0.0
    if raw_type == "BOOLEAN":
        return False
    return ""


def _should_append_seed_control(inp_name: str, raw_type: Any, cfg: dict[str, Any]) -> bool:
    """判断是否需要为随机种子控件追加 ComfyUI 前端的运行控制项（如 fixed）。"""
    if cfg.get("control_after_generate") is True:
        return True
    return raw_type == "INT" and inp_name in ("seed", "noise_seed") and cfg.get("control_after_generate") is not False


def _ordered_node_input_specs(
    node_info: dict[str, Any],
) -> list[tuple[str, Any, dict[str, Any]]]:
    """按 ComfyUI 前端展示顺序提取节点的输入规格列表。"""
    input_sec = node_info.get("input") or {}
    if not isinstance(input_sec, dict):
        return []
    req_map = input_sec.get("required") if isinstance(input_sec.get("required"), dict) else {}
    opt_map = input_sec.get("optional") if isinstance(input_sec.get("optional"), dict) else {}
    order_sec = node_info.get("input_order") if isinstance(node_info.get("input_order"), dict) else {}

    specs: list[tuple[str, Any, dict[str, Any]]] = []
    for section_name, section_map in (("required", req_map), ("optional", opt_map)):
        raw_order = order_sec.get(section_name)
        keys = list(raw_order) if isinstance(raw_order, list) else list(section_map.keys())
        seen = set(keys)
        for k in section_map:
            if k not in seen:
                keys.append(k)
        for inp_name in keys:
            spec = section_map.get(inp_name)
            if not isinstance(spec, (list, tuple)) or not spec:
                continue
            raw_type = spec[0]
            cfg = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
            specs.append((inp_name, raw_type, cfg))
    return specs


_WORKFLOW_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


def _ensure_workflow_uuid(wf_dict: dict[str, Any]) -> bool:
    """确保 UI 工作流顶层具备合法 UUID，避免 ComfyUI v0.3.73 将缺失 id 的工作流误判为同一标签页。"""
    raw_id = wf_dict.get("id")
    if isinstance(raw_id, str) and _WORKFLOW_UUID_RE.match(raw_id.strip()):
        return False
    wf_dict["id"] = str(uuid.uuid4())
    return True


def _api_prompt_to_ui_workflow(
    prompt_dict: dict[str, Any],
    object_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """将 ComfyUI API `prompt` 字典转换为可在网页画布直接打开的标准 LiteGraph 工作流。

    当提供在线 `/object_info` 时，可精确还原各节点输入槽位顺序、`widgets_values`、
    `COMBO` 与连线类型，并在随机种子后保留 `"fixed"` 控制项以确保复现原图结果；
    当 `object_info` 缺失时，仍按兼容规则离线生成完整 `nodes` 与 `links` 结构。
    """
    valid_nodes: dict[str, dict[str, Any]] = {}
    orig_order: dict[str, int] = {}
    for idx, (raw_key, node_val) in enumerate(prompt_dict.items()):
        if (
            isinstance(node_val, dict)
            and isinstance(node_val.get("class_type"), str)
            and node_val.get("class_type", "").strip()
            and isinstance(node_val.get("inputs"), dict)
        ):
            key_str = str(raw_key)
            valid_nodes[key_str] = node_val
            orig_order[key_str] = idx

    ordered_keys = list(valid_nodes.keys())
    if not ordered_keys:
        return {
            "id": str(uuid.uuid4()),
            "last_node_id": 0,
            "last_link_id": 0,
            "nodes": [],
            "links": [],
            "groups": [],
            "config": {},
            "extra": {},
            "version": 0.4,
        }

    # 按有向无环拓扑深度从左到右排列节点，迭代上限防环
    depths: dict[str, int] = {k: 0 for k in ordered_keys}
    for _ in range(len(ordered_keys)):
        changed = False
        for k in ordered_keys:
            best = depths[k]
            for val in valid_nodes[k]["inputs"].values():
                if _is_prompt_link_ref(val, valid_nodes):
                    cand = depths[str(val[0])] + 1
                    if cand > best:
                        best = cand
            if best > depths[k]:
                depths[k] = best
                changed = True
        if not changed:
            break

    sorted_keys = sorted(ordered_keys, key=lambda k: (depths[k], orig_order[k]))
    str_to_int_id: dict[str, int] = {k: idx + 1 for idx, k in enumerate(sorted_keys)}

    # 统计每列行号并初始化 UI 节点骨架
    depth_rows: dict[int, int] = {}
    ui_nodes_by_key: dict[str, dict[str, Any]] = {}
    ui_nodes: list[dict[str, Any]] = []
    pending_links: list[tuple[str, int, str, int, str]] = []

    for order_idx, k in enumerate(sorted_keys):
        node_data = valid_nodes[k]
        class_type = node_data["class_type"]
        node_inputs_data: dict[str, Any] = node_data["inputs"]
        node_info = (
            object_info.get(class_type)
            if isinstance(object_info, dict) and isinstance(object_info.get(class_type), dict)
            else None
        )

        depth = depths[k]
        row_idx = depth_rows.get(depth, 0)
        depth_rows[depth] = row_idx + 1

        if "TextEncode" in class_type:
            size = [400, 260]
        elif "KSampler" in class_type:
            size = [330, 290]
        elif class_type in ("SaveImage", "PreviewImage"):
            size = [330, 270]
        else:
            size = [330, 150]

        ui_inputs: list[dict[str, Any]] = []
        widgets_values: list[Any] = []

        if node_info is not None:
            specs = _ordered_node_input_specs(node_info)
            known_names = {name for name, _, _ in specs}
            for inp_name, raw_type, cfg in specs:
                is_combo = isinstance(raw_type, list) or raw_type == "COMBO"
                slot_type = "COMBO" if is_combo else (str(raw_type) if isinstance(raw_type, str) else "*")
                force_input = bool(cfg.get("forceInput"))
                is_widget = (not force_input) and (
                    is_combo or raw_type in ("INT", "FLOAT", "STRING", "BOOLEAN")
                )
                has_val = inp_name in node_inputs_data
                raw_val = node_inputs_data.get(inp_name)
                is_linked = has_val and _is_prompt_link_ref(raw_val, valid_nodes)

                if is_widget:
                    if is_linked:
                        src_key = str(raw_val[0])
                        src_slot = int(raw_val[1])
                        target_slot = len(ui_inputs)
                        ui_inputs.append(
                            {
                                "name": inp_name,
                                "type": slot_type,
                                "widget": {"name": inp_name},
                                "link": None,
                            }
                        )
                        pending_links.append((src_key, src_slot, k, target_slot, slot_type))
                        widgets_values.append(_default_widget_value(raw_type, cfg))
                    else:
                        val = raw_val if has_val else _default_widget_value(raw_type, cfg)
                        widgets_values.append(val)
                    if _should_append_seed_control(inp_name, raw_type, cfg):
                        widgets_values.append("fixed")
                else:
                    target_slot = len(ui_inputs)
                    ui_inputs.append({"name": inp_name, "type": slot_type, "link": None})
                    if is_linked:
                        src_key = str(raw_val[0])
                        src_slot = int(raw_val[1])
                        pending_links.append((src_key, src_slot, k, target_slot, slot_type))

            # 兜底补充 prompt 中未列入 /object_info 的动态连线输入
            for extra_name, extra_val in node_inputs_data.items():
                if extra_name not in known_names and _is_prompt_link_ref(extra_val, valid_nodes):
                    target_slot = len(ui_inputs)
                    ui_inputs.append({"name": extra_name, "type": "*", "link": None})
                    pending_links.append(
                        (str(extra_val[0]), int(extra_val[1]), k, target_slot, "*")
                    )

            out_types = node_info.get("output") if isinstance(node_info.get("output"), list) else []
            out_names = (
                node_info.get("output_name")
                if isinstance(node_info.get("output_name"), list)
                else out_types
            )
            ui_outputs = [
                {
                    "name": str(out_names[idx] if idx < len(out_names) else out_t),
                    "type": str(out_t),
                    "links": [],
                    "slot_index": idx,
                }
                for idx, out_t in enumerate(out_types)
            ]
        else:
            # 离线兜底：区分连线槽位与标量控件值
            for inp_name, raw_val in node_inputs_data.items():
                if _is_prompt_link_ref(raw_val, valid_nodes):
                    target_slot = len(ui_inputs)
                    ui_inputs.append({"name": inp_name, "type": "*", "link": None})
                    pending_links.append(
                        (str(raw_val[0]), int(raw_val[1]), k, target_slot, "*")
                    )
                else:
                    widgets_values.append(raw_val)
                    if (
                        inp_name in ("seed", "noise_seed")
                        and isinstance(raw_val, int)
                        and not isinstance(raw_val, bool)
                    ):
                        widgets_values.append("fixed")
            ui_outputs = []

        ui_node: dict[str, Any] = {
            "id": str_to_int_id[k],
            "type": class_type,
            "pos": [80 + depth * 420, 80 + row_idx * 320],
            "size": size,
            "flags": {},
            "order": order_idx,
            "mode": 0,
            "inputs": ui_inputs,
            "outputs": ui_outputs,
            "properties": {"Node name for S&R": class_type},
            "widgets_values": widgets_values,
        }
        ui_nodes_by_key[k] = ui_node
        ui_nodes.append(ui_node)

    # 统一分配递增 link_id，确保 links 元组、目标输入 link 与源输出 links 三方一致
    ui_links: list[list[Any]] = []
    for link_idx, (src_key, src_slot, dst_key, dst_slot, target_slot_type) in enumerate(
        pending_links, start=1
    ):
        src_node = ui_nodes_by_key[src_key]
        dst_node = ui_nodes_by_key[dst_key]
        while len(src_node["outputs"]) <= src_slot:
            idx = len(src_node["outputs"])
            inferred_t = target_slot_type if idx == src_slot else "*"
            src_node["outputs"].append(
                {"name": inferred_t, "type": inferred_t, "links": [], "slot_index": idx}
            )
        src_out = src_node["outputs"][src_slot]
        src_out["links"].append(link_idx)
        link_type = src_out.get("type", "*")
        if link_type == "*" and target_slot_type != "*":
            link_type = target_slot_type
            src_out["type"] = link_type
            if src_out.get("name") == "*":
                src_out["name"] = link_type
        if dst_node["inputs"][dst_slot]["type"] == "*" and link_type != "*":
            dst_node["inputs"][dst_slot]["type"] = link_type
        dst_node["inputs"][dst_slot]["link"] = link_idx
        ui_links.append(
            [link_idx, src_node["id"], src_slot, dst_node["id"], dst_slot, link_type]
        )

    return {
        "id": str(uuid.uuid4()),
        "last_node_id": len(ui_nodes),
        "last_link_id": len(ui_links),
        "nodes": ui_nodes,
        "links": ui_links,
        "groups": [],
        "config": {},
        "extra": {},
        "version": 0.4,
    }


def _sanitize_filename(name: str, fallback: str = "file") -> str:
    """过滤非法文件名字符，保留单层文件名。"""
    base = Path(str(name).replace("\\", "/")).name
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", base).strip()
    return cleaned or fallback


def _safe_output_dest(root: Path, subfolder: str, filename: str) -> Path:
    """构造严格位于 root 目录下的安全输出落盘路径，防御路径穿越。"""
    root_resolved = root.resolve()
    parts: list[str] = []
    if subfolder:
        for seg in PurePosixPath(subfolder.replace("\\", "/")).parts:
            if seg and seg not in (".", "..", "/"):
                parts.append(_sanitize_filename(seg, "sub"))
    parts.append(_sanitize_filename(filename, "output.png"))
    dest = root_resolved.joinpath(*parts).resolve()
    if dest != root_resolved and root_resolved not in dest.parents:
        raise ValueError(f"非法输出路径: {subfolder}/{filename}")
    return dest


def _format_timestamped_filename(filename: str, mtime_sec: float | None = None) -> str:
    """为产出文件附加确定性北京时间时间戳（YYYYMMDD-HHMMSS），避免新实例序号重置覆写旧图。"""
    p = PurePosixPath(filename)
    stem, suffix = p.stem, p.suffix
    if _TS_SUFFIX_RE.search(stem):
        return filename
    ts_str = datetime.fromtimestamp(mtime_sec if mtime_sec is not None else time.time(), _CST_TZ).strftime(
        "%Y%m%d-%H%M%S"
    )
    sep = "" if stem.endswith("_") else "_"
    return f"{stem}{sep}{ts_str}{suffix}"


def _sha256_file(path: Path) -> str:
    """分块计算本地文件 SHA256 摘要。"""
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        while True:
            chunk = fp.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _resolve_non_clobber_dest(dest: Path, src_size: int, src_sha256: str) -> tuple[Path, bool]:
    """若本机已存在同名同内容文件则跳过覆写；若内容不同则递增后缀防止误覆盖历史作品。"""
    if not dest.exists():
        return dest, False
    candidate = dest
    idx = 2
    while candidate.exists():
        if candidate.is_file() and candidate.stat().st_size == src_size:
            if _sha256_file(candidate) == src_sha256:
                return candidate, True
        candidate = dest.with_name(f"{dest.stem}-{idx}{dest.suffix}")
        idx += 1
    return candidate, False


def _parse_http_mtime(last_modified: str | None) -> float | None:
    """解析 HTTP Last-Modified 响应头为 UNIX 秒，失败返回 None。"""
    if not last_modified:
        return None
    try:
        dt = parsedate_to_datetime(last_modified.strip())
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except Exception:
        return None


def _format_comfy_error(node_type: str | None, exc_type: str | None, exc_msg: str | None) -> str:
    """将 ComfyUI 节点异常格式化为紧凑的单行中文提示（不超过 36 字）。"""
    node = (node_type or "节点").strip() or "节点"
    etype = (exc_type or "").strip()
    emsg = (exc_msg or "").strip()
    combined = f"{etype} {emsg}".lower()
    if any(k in combined for k in ("out of memory", "outofmemoryerror", "allocation on device")):
        out = f"{node}: 显存不足 (OOM)"
    elif any(k in combined for k in ("no such file", "filenotfound", "not found", "missing")):
        out = f"{node}: 缺少模型或文件"
    else:
        detail = etype or emsg or "执行异常"
        out = f"{node}: {detail}"
    return (out[:35] + "…") if len(out) > 36 else out


class _StreamingMultipartFile:
    """零全量缓存的流式 multipart/form-data 读取器，支持字节级进度回调。"""

    def __init__(
        self,
        file_path: Path,
        field_name: str = "image",
        fields: dict[str, str] | None = None,
        on_progress: Callable[[int], None] | None = None,
    ) -> None:
        self.file_path = file_path
        self.boundary = uuid.uuid4().hex
        self.content_type = f"multipart/form-data; boundary={self.boundary}"
        self._on_progress = on_progress
        self._last_report_ts = 0.0
        self._last_pct = -1

        head_parts: list[bytes] = []
        for k, v in (fields or {}).items():
            head_parts.append(
                f"--{self.boundary}\r\n"
                f'Content-Disposition: form-data; name="{k}"\r\n\r\n'
                f"{v}\r\n".encode("utf-8")
            )
        ctype = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
        safe_name = _sanitize_filename(file_path.name).replace('"', "_")
        head_parts.append(
            f"--{self.boundary}\r\n"
            f'Content-Disposition: form-data; name="{field_name}"; filename="{safe_name}"\r\n'
            f"Content-Type: {ctype}\r\n\r\n".encode("utf-8")
        )
        self._head = b"".join(head_parts)
        self._tail = f"\r\n--{self.boundary}--\r\n".encode("utf-8")
        self._file_size = file_path.stat().st_size
        self._total_len = len(self._head) + self._file_size + len(self._tail)
        self._fp = open(file_path, "rb")
        self._stage = 0  # 0=head, 1=file, 2=tail, 3=eof
        self._offset = 0
        self._sent = 0

    def __len__(self) -> int:
        return self._total_len

    def read(self, size: int = -1) -> bytes:
        want = self._total_len if size is None or size < 0 else size
        if want == 0 or self._stage >= 3:
            return b""
        chunks: list[bytes] = []
        remaining = want
        while remaining > 0 and self._stage < 3:
            if self._stage == 0:
                piece = self._head[self._offset : self._offset + remaining]
                self._offset += len(piece)
                chunks.append(piece)
                remaining -= len(piece)
                if self._offset >= len(self._head):
                    self._stage = 1
                    self._offset = 0
            elif self._stage == 1:
                piece = self._fp.read(remaining)
                if piece:
                    chunks.append(piece)
                    remaining -= len(piece)
                else:
                    self._stage = 2
                    self._offset = 0
            elif self._stage == 2:
                piece = self._tail[self._offset : self._offset + remaining]
                self._offset += len(piece)
                chunks.append(piece)
                remaining -= len(piece)
                if self._offset >= len(self._tail):
                    self._stage = 3
        data = b"".join(chunks)
        self._sent += len(data)
        self._report()
        return data

    def _report(self) -> None:
        if not self._on_progress or self._total_len <= 0:
            return
        pct = min(100, int(round(self._sent * 100.0 / self._total_len)))
        now = time.monotonic()
        if pct != self._last_pct and (pct >= 100 or now - self._last_report_ts >= 0.12):
            self._last_pct = pct
            self._last_report_ts = now
            self._on_progress(pct)

    def close(self) -> None:
        try:
            self._fp.close()
        except Exception:
            pass

    def __enter__(self) -> "_StreamingMultipartFile":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()


def _clean_env(extra_env: dict[str, str] | None = None) -> dict[str, str]:
    """构造强制 UTF-8 与防挂起超时的子进程环境变量，并支持注入可移植工具路径。"""
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env.setdefault("COLAB_CLI_REQUEST_TIMEOUT", "12")
    if extra_env:
        for k, v in extra_env.items():
            if v is not None and str(v).strip():
                env[str(k)] = str(v)
    return env


def _sanitize_account_label(label: str, fallback: str = "Ultra 账号") -> str:
    """清洗账号备注名称，限制长度并剔除控制字符。"""
    cleaned = re.sub(r"[\r\n\t]+", " ", str(label or "")).strip()
    return cleaned[:16] if cleaned else fallback


def _extract_valid_https_url(raw_url: str) -> str:
    """校验并返回标准化的 https://<domain> 隧道地址；空串返回空串，非法值抛出异常。"""
    raw = str(raw_url or "").strip().rstrip("/")
    if not raw:
        return ""
    if not raw.lower().startswith("https://"):
        raise ValueError("固定隧道域名必须以 https:// 开头")
    host = raw[8:]
    if (
        not host
        or "." not in host
        or any(ch in host for ch in ("/", "?", "#", "@", ":", " "))
        or not _CUSTOM_TUNNEL_HOST_RE.match(host)
    ):
        raise ValueError(f"非法隧道域名格式: {raw}")
    return f"https://{host.lower()}"


def _load_cf_secret_raw() -> dict[str, Any]:
    """仅限后端写入校验与状态检查使用的内部加载器，禁止将返回字典直接序列化给前端。"""
    if not _CF_SECRET_FILE.is_file():
        return {}
    try:
        data = json.loads(_CF_SECRET_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _is_allowed_tunnel_url(url: str) -> bool:
    """判断 URL 是否命中默认白名单或本地已配置的 Cloudflare 固定域名。"""
    cand = str(url or "").strip().rstrip("/")
    if not cand:
        return False
    if _TUNNEL_URL_RE.match(cand):
        return True
    cfg = _load_cf_secret_raw()
    for key in ("comfy_url", "llm_url"):
        saved = str(cfg.get(key) or "").strip().rstrip("/")
        if saved and cand.lower() == saved.lower():
            return True
    return False


def _probe_tunnel_alive(url: str, timeout: float = 3.5, kind: str = "comfy") -> bool:
    """探测公网隧道是否在线可用（comfy 探测 /system_stats，llm 探测 /health，auto 自动双模探测）。"""
    if not _is_allowed_tunnel_url(url):
        return False
    base = url.strip().rstrip("/")
    mode = str(kind or "comfy").strip().lower()

    def _probe_single(k: str) -> bool:
        target = f"{base}/health" if k == "llm" else f"{base}/system_stats"
        try:
            req = urllib.request.Request(target, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if resp.status != 200:
                    return False
                body = json.loads(resp.read().decode("utf-8", errors="ignore"))
                if not isinstance(body, dict):
                    return False
                if k == "llm":
                    return str(body.get("status") or "").lower() == "ok"
                return "system" in body or "devices" in body
        except Exception:
            return False

    if mode == "llm":
        return _probe_single("llm")
    if mode == "comfy":
        return _probe_single("comfy")
    if mode in ("dual", "auto"):
        return _probe_single("comfy") or _probe_single("llm")
    return _probe_single("comfy") or _probe_single("llm")


def _detect_tunnel_kind(
    url: str, timeout: float = 3.0, preferred: str = "comfy"
) -> str | None:
    """自动识别存活隧道后端的服务类型，返回 'comfy'、'llm' 或 None。"""
    order = ("llm", "comfy") if preferred == "llm" else ("comfy", "llm")
    for cand in order:
        if _probe_tunnel_alive(url, timeout=timeout, kind=cand):
            return cand
    return None


def _sanitize_prefs_dict(prefs: dict[str, Any]) -> dict[str, Any]:
    """对本地持久化偏好设置执行类型与枚举范围清洗，确保不包含任何敏感字段。"""
    cleaned = dict(DEFAULT_PREFS)
    for k in DEFAULT_PREFS:
        if k in prefs:
            cleaned[k] = prefs[k]

    prof = str(cleaned.get("boot_profile") or "anima").strip().lower()
    fallback_p = prof if prof in SUPPORTED_BOOT_PROFILES else "anima"
    raw_profiles = prefs.get("boot_profiles") if "boot_profiles" in prefs else [fallback_p]
    norm_profiles = _normalize_boot_profiles(raw_profiles, fallback_profile=fallback_p)
    cleaned["boot_profiles"] = norm_profiles
    cleaned["boot_profile"] = norm_profiles[0]

    quant = str(cleaned.get("boot_llm_quant") or "UD-Q4_K_XL").strip()
    cleaned["boot_llm_quant"] = quant if quant in SUPPORTED_LLM_QUANTS else "UD-Q4_K_XL"

    try:
        ctx_val = int(cleaned.get("boot_llm_ctx") or 32768)
    except (TypeError, ValueError):
        ctx_val = 32768
    cleaned["boot_llm_ctx"] = max(4096, min(262144, ctx_val))
    cleaned["boot_llm_mmproj"] = bool(cleaned.get("boot_llm_mmproj", True))

    q27 = str(cleaned.get("boot_llm27_quant") or "UD-Q4_K_XL").strip()
    cleaned["boot_llm27_quant"] = q27 if q27 in SUPPORTED_LLM_QUANTS else "UD-Q4_K_XL"
    try:
        ctx27_val = int(cleaned.get("boot_llm27_ctx") or 65536)
    except (TypeError, ValueError):
        ctx27_val = 65536
    cleaned["boot_llm27_ctx"] = max(4096, min(262144, ctx27_val))
    cleaned["boot_llm27_mmproj"] = bool(cleaned.get("boot_llm27_mmproj", True))

    cleaned["keep_alive"] = bool(cleaned.get("keep_alive", True))
    try:
        idle_mins = int(cleaned.get("idle_auto_stop_minutes", 60))
    except (TypeError, ValueError):
        idle_mins = 60
    cleaned["idle_auto_stop_minutes"] = idle_mins if idle_mins in SUPPORTED_IDLE_STOP_MINUTES else 60
    for path_key in (
        "active_account_id",
        "colab_exe_path",
        "colab_py_path",
        "rclone_exe_path",
        "rclone_conf_path",
    ):
        cleaned[path_key] = str(cleaned.get(path_key) or "").strip()[:512]
    return cleaned


def _run_hidden(
    cmd: list[str],
    timeout: int = 25,
    extra_env: dict[str, str] | None = None,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    """以无窗口、UTF-8 方式执行本地子进程。"""
    return subprocess.run(
        cmd,
        input=input_text,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        env=_clean_env(extra_env),
        creationflags=CREATE_NO_WINDOW,
    )


def _hidden_python_exe() -> str:
    """当宿主为 pythonw.exe 时优先切回同目录 python.exe 配合无窗口标志执行 CLI。"""
    exe = Path(sys.executable)
    if exe.name.lower() == "pythonw.exe":
        py_console = exe.with_name("python.exe")
        if py_console.exists():
            return str(py_console)
    return str(exe)


def _find_colab_exe() -> str | None:
    """定位 colab.exe 可执行文件。"""
    if COLAB_EXE_DEFAULT.exists():
        return str(COLAB_EXE_DEFAULT)
    found = _discover_colab_exe()
    return str(found) if found else shutil.which("colab")


def _default_baseline(balance: float) -> float:
    """根据当前余额推算进度条满额基准值（按百位向上取整）。"""
    if balance <= 0:
        return 100.0
    return float(max(100, int(math.ceil(balance / 100.0) * 100)))


class ColabPetBackend:
    """Colab 桌宠状态管理与运维动作执行器。"""

    def __init__(
        self,
        state_path: Path = STATE_FILE,
        download_root: Path = LOCAL_DOWNLOAD_ROOT,
    ) -> None:
        self.state_path = state_path
        self.download_root = Path(download_root)
        self._lock = threading.RLock()
        self._fetch_lock = threading.Lock()
        self._comfy_lock = threading.RLock()
        self._action_lock = threading.Lock()
        self._upload_lock = threading.Lock()
        self._keepalive_lock = threading.Lock()
        self._deploy_lock = threading.Lock()
        self._deploy_proc: Any = None
        self._deploy_cancelled: bool = False
        self._deploy_dl_total: int = 3
        self._deploy_dl_done: int = 0
        self._deploy_dl_name: str = ""
        self._scarce_gpus: dict[str, dict[str, Any]] = {}
        self._deploy_state: dict[str, Any] = self._default_deploy_state()
        self._consecutive_failures: int = 0
        self._comfy_http_failures: int = 0
        self._last_tunnel_discover_ts: float = 0.0
        self._last_llm_tunnel_discover_ts: float = 0.0
        self._last_keepalive_ts: float = 0.0
        self._last_activity_m: float = 0.0
        self._idle_warned_ep: str | None = None
        self._idle_stop_lock = threading.Lock()
        self._event_cb: Callable[[dict[str, Any]], None] | None = None
        self._monitor_stop = threading.Event()
        self._wake_monitor = threading.Event()
        self._monitor_thread: threading.Thread | None = None
        self._ws_stop = threading.Event()
        self._ws_thread: threading.Thread | None = None
        self._ws_target_url: str | None = None
        self._ws_app: Any = None
        self._ws_last_emit_ts: float = 0.0
        self._http = requests.Session()
        self._http.headers.update({"User-Agent": "ColabPet/1.0"})
        self._seen_prompts_by_url: dict[str, set[str]] = {}
        self._reported_error_prompts: set[str] = set()
        self._pending_pull_items: list[tuple[str, dict[str, str]]] = []
        self._last_downloaded_files: list[Path] = []
        self._canvas_ext_ready_eps: dict[str, str] = {}
        self._canvas_ext_lock = threading.Lock()
        # 远端自描述元数据缓存（ep -> 非敏感元信息）与纯内存态密钥映射（严禁写入 self.state 或落盘）
        self._service_meta_by_ep: dict[str, dict[str, Any]] = {}
        self._runtime_secrets: dict[str, str] = {}
        self._deploy_meta_warmed: bool = False
        self._deploy_phase_idx: int = 1
        self._deploy_phase_total: int = 1
        self._deploy_phase_prof: str = "anima"
        self._deploy_phase_is_llm: bool = False
        self._deploy_captured_llm_tunnel: str | None = None
        self._comfy_state: dict[str, Any] = self._default_comfy_state()
        # 账号绑定与 OAuth2 PKCE 仅存于内存的临时态，绝不落盘
        self._pending_oauth_verifier: str = ""
        self._pending_oauth_ts: float = 0.0
        self._auth_missing: bool = False
        self._auth_expired: bool = False
        self._gdrive_bind_proc: Any = None
        self._gdrive_bind_cancelled: bool = False

        self.state = self._load_state()
        has_cached_bal = self.state.get("last_balance") is not None
        self.last_snapshot: dict[str, Any] = (
            self._build_snapshot(live_data=None, stale=False, error_msg=None)
            if has_cached_bal
            else self._build_snapshot(
                live_data=None,
                stale=True,
                error_msg="正在连接 Colab...",
            )
        )

    def _get_colab_exe(self) -> Path | None:
        """按测试覆写、用户自定义、环境变量及跨平台探测定位 colab.exe。"""
        if COLAB_EXE_DEFAULT != _INIT_COLAB_EXE:
            return COLAB_EXE_DEFAULT if COLAB_EXE_DEFAULT.is_file() else None
        prefs = (self.state.get("prefs") or {}) if hasattr(self, "state") else {}
        return _discover_colab_exe(str(prefs.get("colab_exe_path") or ""))

    def _get_colab_py(self) -> Path:
        """按测试覆写、用户自定义、环境变量及虚拟环境探测定位 colab-cli Python。"""
        if COLAB_PY != _INIT_COLAB_PY:
            return COLAB_PY
        prefs = (self.state.get("prefs") or {}) if hasattr(self, "state") else {}
        return _discover_colab_py(
            str(prefs.get("colab_py_path") or ""),
            colab_exe=self._get_colab_exe(),
        )

    def _get_rclone_exe(self) -> Path:
        """按测试覆写、用户自定义、环境变量及常见包管理器位置定位 rclone.exe。"""
        if RCLONE_EXE != _INIT_RCLONE_EXE:
            return RCLONE_EXE
        prefs = (self.state.get("prefs") or {}) if hasattr(self, "state") else {}
        return _discover_rclone_exe(str(prefs.get("rclone_exe_path") or ""))

    def _get_rclone_conf(self) -> Path:
        """按测试覆写、用户自定义、环境变量及标准配置位置定位 rclone.conf。"""
        if RCLONE_CONF != _INIT_RCLONE_CONF:
            return RCLONE_CONF
        prefs = (self.state.get("prefs") or {}) if hasattr(self, "state") else {}
        return _discover_rclone_conf(
            str(prefs.get("rclone_conf_path") or ""),
            rclone_exe=self._get_rclone_exe(),
        )

    def _build_ctl_env(self) -> dict[str, str]:
        """构造传递给 colab_ctl.py 子进程的可移植工具路径环境变量。"""
        env_map: dict[str, str] = {}
        cexe = self._get_colab_exe()
        if cexe is not None:
            env_map["COLAB_PET_COLAB_EXE"] = str(cexe)
        cpy = self._get_colab_py()
        if cpy.is_file():
            env_map["COLAB_PET_COLAB_PY"] = str(cpy)
        rexe = self._get_rclone_exe()
        if rexe.is_file():
            env_map["COLAB_PET_RCLONE_EXE"] = str(rexe)
        rconf = self._get_rclone_conf()
        if rconf.is_file():
            env_map["COLAB_PET_RCLONE_CONF"] = str(rconf)
        return env_map

    @staticmethod
    def _default_comfy_state() -> dict[str, Any]:
        return {
            "online": False,
            "service_kind": "comfy",
            "comfy_online": False,
            "llm_online": False,
            "ws_connected": False,
            "tunnel_url": None,
            "llm_tunnel_url": None,
            "running_count": 0,
            "pending_count": 0,
            "llm_running_count": 0,
            "active_profiles": [],
            "step_value": 0,
            "step_max": 0,
            "step_pct": 0.0,
            "current_node": "",
            "gpu_name": "",
            "vram_used_gb": 0.0,
            "vram_total_gb": 0.0,
            "vram_pct": 0.0,
            "has_error": False,
            "error_summary": "",
            "pending_pull_count": 0,
            "last_output_count": 0,
            "last_output_name": "",
            "model_name": "",
            "quant": "",
            "ctx_size": 0,
            "tps": None,
        }

    def get_comfy_state(self) -> dict[str, Any]:
        """返回线程安全的 ComfyUI 实时状态副本。"""
        with self._comfy_lock:
            return dict(self._comfy_state)

    def _emit_event(self, payload: dict[str, Any]) -> None:
        cb = self._event_cb
        if not cb:
            return
        try:
            cb(payload)
        except Exception:
            pass

    @staticmethod
    def _default_deploy_state() -> dict[str, Any]:
        """构建默认的一键开机与 7 步进度状态字典。"""
        return {
            "active": False,
            "profile": "anima",
            "gpu": "L4",
            "high_mem": False,
            "step_idx": 0,
            "step_total": 7,
            "stage_title": "",
            "sub_detail": "",
            "pct": 0,
            "started_ts": 0,
            "elapsed_sec": 0,
            "error_msg": "",
            "stockout_gpu": None,
            "suggest_gpu": None,
        }

    def is_deploying(self) -> bool:
        """当前是否正在执行一键开机部署。"""
        with self._lock:
            return bool(self._deploy_state.get("active"))

    def get_deploy_state(self) -> dict[str, Any]:
        """返回线程安全的部署进度副本并实时计算耗时读秒。"""
        with self._lock:
            st = dict(self._deploy_state)
            if st.get("active") and st.get("started_ts"):
                st["elapsed_sec"] = max(0, int(time.time()) - int(st["started_ts"]))
            return st

    def _get_active_scarce_gpus(self) -> dict[str, dict[str, Any]]:
        """清理并返回仍在冷却有效期内的紧缺/受限 GPU 映射。"""
        now_ts = int(time.time())
        with self._lock:
            expired = [
                k
                for k, v in self._scarce_gpus.items()
                if (now_ts - int(v.get("ts") or 0)) > SCARCE_GPU_TTL_SEC
            ]
            for k in expired:
                self._scarce_gpus.pop(k, None)
            return {k: dict(v) for k, v in self._scarce_gpus.items()}

    @staticmethod
    def _gpu_rate_key(accelerator: Any, machine_shape: Any = None) -> str | None:
        """将加速卡型号与内存形态映射为规范的单卡实测费率存储键。"""
        acc = str(accelerator or "").strip().upper()
        if acc not in SUPPORTED_BOOT_GPUS:
            return None
        if acc == "L4":
            return "L4:STD"
        shape = str(machine_shape or "").strip().upper()
        suffix = "HM" if shape in ("HIGH_RAM", "HM", "1") else "STD"
        return f"{acc}:{suffix}"

    def _default_state(self) -> dict[str, Any]:
        today = datetime.now().strftime("%Y-%m-%d")
        return {
            "date": today,
            "day_start_balance": None,
            "day_consumed_cu": 0.0,
            "day_topup_cu": 0.0,
            "last_balance": None,
            "last_delta": 0.0,
            "baseline_balance": 1000.0,
            "active_since_ts": None,
            "active_endpoint": None,
            "session_start_balance": None,
            "last_rate_hourly": 0.0,
            "last_assignments_count": 0,
            "last_sessions": [],
            "last_eligible_gpus": None,
            "last_eligible_tpus": None,
            "last_updated_ts": None,
            "measured_gpu_rates": dict(DEFAULT_MEASURED_GPU_RATES),
            "tunnel_cache": {},
            "account_profiles": [],
            "account_ledgers": {},
            "onboarding_cache": {
                "gdrive_free_gb": None,
                "gdrive_total_gb": None,
                "gdrive_checked_ts": None,
            },
            "prefs": dict(DEFAULT_PREFS),
        }

    def _load_state(self) -> dict[str, Any]:
        default = self._default_state()
        if not self.state_path.exists():
            return default
        try:
            raw = json.loads(self.state_path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                return default
            merged = dict(default)
            for key in default:
                if key == "prefs" and isinstance(raw.get("prefs"), dict):
                    merged["prefs"] = _sanitize_prefs_dict({**DEFAULT_PREFS, **raw["prefs"]})
                elif key == "measured_gpu_rates" and isinstance(raw.get("measured_gpu_rates"), dict):
                    cleaned_rates = dict(DEFAULT_MEASURED_GPU_RATES)
                    valid_keys = {
                        f"{g}:{s}"
                        for g in SUPPORTED_BOOT_GPUS
                        for s in ("STD", "HM")
                        if not (g == "L4" and s == "HM")
                    }
                    for rk, rv in raw["measured_gpu_rates"].items():
                        if rk in valid_keys:
                            try:
                                fv = float(rv)
                            except (TypeError, ValueError):
                                continue
                            if 0.02 < fv < 200.0:
                                cleaned_rates[rk] = round(fv, 2)
                    merged["measured_gpu_rates"] = cleaned_rates
                elif key == "tunnel_cache" and isinstance(raw.get("tunnel_cache"), dict):
                    cleaned_cache: dict[str, str] = {}
                    for ep, val in raw["tunnel_cache"].items():
                        if isinstance(ep, str) and isinstance(val, str):
                            clean_u = val.strip().rstrip("/")
                            if _is_allowed_tunnel_url(clean_u):
                                cleaned_cache[ep] = clean_u
                    merged["tunnel_cache"] = cleaned_cache
                elif key == "account_profiles" and isinstance(raw.get("account_profiles"), list):
                    clean_accts: list[dict[str, Any]] = []
                    seen_ids: set[str] = set()
                    for item in raw["account_profiles"]:
                        if not isinstance(item, dict):
                            continue
                        aid = re.sub(r"[^a-zA-Z0-9_-]", "", str(item.get("id") or ""))[:24]
                        if not aid or aid in seen_ids:
                            continue
                        seen_ids.add(aid)
                        bal_raw = item.get("last_balance")
                        try:
                            bal_val = round(float(bal_raw), 2) if bal_raw is not None else None
                        except (TypeError, ValueError):
                            bal_val = None
                        ts_raw = item.get("last_updated_ts")
                        try:
                            ts_val = int(ts_raw) if ts_raw is not None else None
                        except (TypeError, ValueError):
                            ts_val = None
                        clean_accts.append(
                            {
                                "id": aid,
                                "label": _sanitize_account_label(str(item.get("label") or ""), "Ultra 账号"),
                                "email_masked": str(item.get("email_masked") or "***")[:64],
                                "last_balance": bal_val,
                                "last_is_running": bool(item.get("last_is_running", False)),
                                "last_updated_ts": ts_val,
                            }
                        )
                    merged["account_profiles"] = clean_accts
                elif key == "account_ledgers" and isinstance(raw.get("account_ledgers"), dict):
                    clean_ledgers: dict[str, dict[str, Any]] = {}
                    for aid_raw, lobj in raw["account_ledgers"].items():
                        aid = re.sub(r"[^a-zA-Z0-9_-]", "", str(aid_raw or ""))[:24]
                        if aid and isinstance(lobj, dict):
                            clean_ledgers[aid] = {lk: lobj[lk] for lk in _ACCOUNT_LEDGER_KEYS if lk in lobj}
                    merged["account_ledgers"] = clean_ledgers
                elif key == "onboarding_cache" and isinstance(raw.get("onboarding_cache"), dict):
                    oc = raw["onboarding_cache"]
                    merged["onboarding_cache"] = {
                        "gdrive_free_gb": oc.get("gdrive_free_gb"),
                        "gdrive_total_gb": oc.get("gdrive_total_gb"),
                        "gdrive_checked_ts": oc.get("gdrive_checked_ts"),
                    }
                elif key in raw:
                    merged[key] = raw[key]
            return merged
        except Exception:
            return default

    def _save_state(self) -> None:
        """原子写入非敏感配置与账本状态，不保存任何登录凭据。"""
        tmp_path = self.state_path.with_suffix(".tmp")
        payload = json.dumps(self.state, ensure_ascii=False, indent=2)
        tmp_path.write_text(payload, encoding="utf-8")
        tmp_path.replace(self.state_path)

    @staticmethod
    def _default_account_label(prefix: str, existing_profiles: list[dict[str, Any]]) -> str:
        """为新纳入保险箱的 Google 账号生成简洁不重复的默认备注名称。"""
        used_labels = {str(p.get("label") or "").strip() for p in existing_profiles}
        idx = len(existing_profiles) + 1
        while True:
            cand = f"Ultra {idx}号"
            if cand not in used_labels:
                return cand
            idx += 1

    def _snapshot_current_ledger(self) -> dict[str, Any]:
        """提取当前激活账号对应的每日账本与会话快照，用于切号前独立归档。"""
        return {lk: self.state.get(lk) for lk in _ACCOUNT_LEDGER_KEYS}

    def _restore_ledger_for_account(self, acct_id: str) -> None:
        """加载指定账号的历史独立账本；首次纳入的账号重置为空账本以免串号误算消耗。"""
        fresh = self._default_state()
        ledgers = self.state.setdefault("account_ledgers", {})
        saved = ledgers.get(acct_id) if (acct_id and isinstance(ledgers, dict)) else None
        for lk in _ACCOUNT_LEDGER_KEYS:
            if isinstance(saved, dict) and lk in saved:
                self.state[lk] = saved[lk]
            else:
                self.state[lk] = fresh[lk]

    def _upsert_account_profile(
        self,
        acct_id: str,
        email_masked: str,
        preferred_label: str = "",
        balance: float | None = None,
        is_running: bool | None = None,
    ) -> dict[str, Any]:
        """登记或更新账号列表卡片（绝不存储完整邮箱与任何 Token），返回更新后的卡片。"""
        profiles = self.state.setdefault("account_profiles", [])
        now_ts = int(time.time())
        for item in profiles:
            if isinstance(item, dict) and item.get("id") == acct_id:
                if preferred_label:
                    item["label"] = _sanitize_account_label(preferred_label, str(item.get("label") or "Ultra 账号"))
                if email_masked and email_masked != "***":
                    item["email_masked"] = email_masked
                if balance is not None:
                    item["last_balance"] = round(float(balance), 2)
                    item["last_updated_ts"] = now_ts
                if is_running is not None:
                    item["last_is_running"] = bool(is_running)
                return dict(item)

        new_label = _sanitize_account_label(
            preferred_label,
            self._default_account_label(email_masked, profiles),
        )
        new_item: dict[str, Any] = {
            "id": acct_id,
            "label": new_label,
            "email_masked": email_masked or f"acct-{acct_id[:6]}",
            "last_balance": round(float(balance), 2) if balance is not None else None,
            "last_is_running": bool(is_running) if is_running is not None else False,
            "last_updated_ts": now_ts if balance is not None else None,
        }
        profiles.append(new_item)
        return dict(new_item)

    def _sync_active_account_ledger(
        self,
        balance: float | None = None,
        is_running: bool | None = None,
    ) -> None:
        """将当前账本与余额回写至 active_account_id 专属槽位，确保多号切换零串扰。"""
        prefs = self.state.get("prefs") or {}
        active_id = str(prefs.get("active_account_id") or "").strip()
        if not active_id:
            return
        ledgers = self.state.setdefault("account_ledgers", {})
        ledgers[active_id] = self._snapshot_current_ledger()
        now_ts = int(time.time())
        for item in self.state.get("account_profiles") or []:
            if isinstance(item, dict) and item.get("id") == active_id:
                if balance is not None:
                    item["last_balance"] = round(float(balance), 2)
                    item["last_updated_ts"] = now_ts
                if is_running is not None:
                    item["last_is_running"] = bool(is_running)
                break

    @staticmethod
    def _has_gdrive_save_section(conf_path: Path) -> bool:
        """快速校验 rclone.conf 是否已配置 gdrive_save 节（绝不读取其中敏感 token）。"""
        if not conf_path or not conf_path.is_file():
            return False
        try:
            content = conf_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return False
        return bool(_RCLONE_GDRIVE_SECTION_RE.search(content))

    def _query_colab_live(self) -> dict[str, Any]:
        """调用 colab_cli 获取实时 CU 余额、可用加速卡与活跃会话，并无感纳管当前登录账号。"""
        colab_py = self._get_colab_py()
        if not colab_py.exists():
            raise FileNotFoundError(f"未找到 colab-cli 运行环境: {colab_py}")

        with self._lock:
            prefs = self.state.get("prefs") or {}
            active_id = str(prefs.get("active_account_id") or "").strip()
            profiles = self.state.get("account_profiles") or []
            known_ids = {str(p.get("id") or "") for p in profiles if isinstance(p, dict)}
            need_userinfo = (not profiles) or (not active_id) or (active_id not in known_ids)

        cmd = [str(colab_py), "-c", _COLAB_PROBE_SCRIPT]
        if need_userinfo:
            cmd.append("--need-userinfo")
        if active_id:
            cmd.append(f"--active-acct={active_id}")

        proc = _run_hidden(cmd, timeout=25)
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "").strip()
            last_line = err.splitlines()[-1] if err else f"退出码 {proc.returncode}"
            raise RuntimeError(f"Colab 查询失败: {last_line}")

        stdout = proc.stdout or ""
        matched = re.search(
            r"<<<COLAB_PET_JSON>>>(.*?)<<<END_COLAB_PET_JSON>>>",
            stdout,
            re.S,
        )
        json_text = matched.group(1).strip() if matched else stdout.strip()
        if not json_text:
            raise RuntimeError("Colab 返回空结果")
        return json.loads(json_text)

    def _update_ledger(self, live: dict[str, Any]) -> None:
        """更新每日消耗账本、基准额度、连续开机计时，并同步写入当前多账号槽位。"""
        now_ts = int(time.time())
        today = datetime.now().strftime("%Y-%m-%d")
        balance = float(live["balance"])
        rate_hourly = float(live["rate_hourly"])
        assignments_count = int(live["assignments_count"])
        sessions = list(live.get("sessions", []))

        # 若探针识别并无感纳管了存量 Google 账号，先同步写入 account_profiles 与 active_account_id
        acct_meta = live.get("account_meta")
        if isinstance(acct_meta, dict) and acct_meta.get("id"):
            aid = re.sub(r"[^a-zA-Z0-9_-]", "", str(acct_meta["id"]))[:24]
            if aid:
                prefs = self.state.setdefault("prefs", dict(DEFAULT_PREFS))
                prefs["active_account_id"] = aid
                self._upsert_account_profile(
                    aid,
                    str(acct_meta.get("email_masked") or "***"),
                    balance=balance,
                    is_running=(assignments_count > 0 or rate_hourly > 0.01),
                )

        prev_bal = self.state.get("last_balance")
        diff = 0.0

        # 跨零点或全新日期：以前一刻余额无缝衔接为新一日的起始余额，不丢失跨午夜消耗
        if self.state.get("date") != today:
            start_bal = float(prev_bal) if prev_bal is not None else balance
            self.state["date"] = today
            self.state["day_start_balance"] = start_bal
            self.state["day_topup_cu"] = 0.0
            raw_diff = round(balance - start_bal, 4)
            if raw_diff > 0.05:
                self.state["day_topup_cu"] = raw_diff
                self.state["last_delta"] = raw_diff
                diff = raw_diff
            elif raw_diff < -0.0005:
                self.state["last_delta"] = raw_diff
                diff = raw_diff
            else:
                self.state["last_delta"] = 0.0
            self.state["last_balance"] = balance
        else:
            if self.state.get("day_start_balance") is None:
                self.state["day_start_balance"] = balance
            if prev_bal is not None:
                raw_diff = round(balance - float(prev_bal), 4)
                if raw_diff < -0.0005:
                    self.state["last_delta"] = raw_diff
                    diff = raw_diff
                elif raw_diff > 0.05:
                    # 充值/月度额度发放时不计入消耗
                    self.state["day_topup_cu"] = round(
                        float(self.state.get("day_topup_cu") or 0.0) + raw_diff,
                        4,
                    )
                    self.state["last_delta"] = raw_diff
                    diff = raw_diff
            self.state["last_balance"] = balance

        # 统一以 起始余额 + 今日充值 - 当前余额 精确对账，消除多次浮点累加尾差
        day_start_f = float(self.state["day_start_balance"])
        day_topup_f = float(self.state.get("day_topup_cu") or 0.0)
        self.state["day_consumed_cu"] = round(max(0.0, day_start_f + day_topup_f - balance), 4)

        # 进度条满额参考值：随最高余额自动升级到百位
        baseline = float(self.state.get("baseline_balance") or 0.0)
        if baseline <= 0.0 or balance > baseline:
            self.state["baseline_balance"] = _default_baseline(balance)

        # 连续开机计时与本次开机起始余额（支持换卡重置与存量运行会话平滑回填）
        primary_sess = next(
            (s for s in sessions if s.get("name") == "comfy" and s.get("endpoint")),
            None,
        ) or (sessions[0] if sessions and sessions[0].get("endpoint") else None)
        cur_ep = str(primary_sess.get("endpoint")) if primary_sess else None

        is_running = assignments_count > 0 or rate_hourly > 0.01
        if is_running:
            prev_since = self.state.get("active_since_ts")
            prev_ep = self.state.get("active_endpoint")
            if not prev_ep:
                prev_sessions = list(self.state.get("last_sessions") or [])
                prev_primary = next(
                    (s for s in prev_sessions if s.get("name") == "comfy" and s.get("endpoint")),
                    None,
                ) or (
                    prev_sessions[0]
                    if prev_sessions and prev_sessions[0].get("endpoint")
                    else None
                )
                if prev_primary:
                    prev_ep = str(prev_primary.get("endpoint"))
            switched_ep = bool(cur_ep and prev_ep and cur_ep != prev_ep)
            if not prev_since or switched_ep:
                self.state["active_since_ts"] = now_ts
                self.state["active_endpoint"] = cur_ep
                self._last_activity_m = time.monotonic()
                self._idle_warned_ep = None
                self.state["session_start_balance"] = (
                    float(prev_bal)
                    if (not switched_ep and prev_bal is not None and float(prev_bal) >= balance)
                    else balance
                )
            else:
                if cur_ep:
                    self.state["active_endpoint"] = cur_ep
                if self._last_activity_m <= 0.0:
                    self._last_activity_m = time.monotonic()
                    self._idle_warned_ep = None
                if self.state.get("session_start_balance") is None:
                    est = rate_hourly * max(0, now_ts - int(prev_since)) / 3600.0
                    cap = day_start_f + day_topup_f
                    self.state["session_start_balance"] = (
                        min(cap, balance + max(0.0, est))
                        if cap >= balance
                        else (balance + max(0.0, est))
                    )
                elif diff > 0.05:
                    # 运行中途发生充值时同步抬高会话起始基准，避免冲减本次实耗
                    self.state["session_start_balance"] = (
                        float(self.state["session_start_balance"]) + diff
                    )
        else:
            self.state["active_since_ts"] = None
            self.state["active_endpoint"] = None
            self.state["session_start_balance"] = None
            self._last_activity_m = 0.0
            self._idle_warned_ep = None
            self._last_keepalive_ts = 0.0

        self.state["last_rate_hourly"] = rate_hourly
        self.state["last_assignments_count"] = assignments_count
        self.state["last_sessions"] = sessions
        self.state["last_updated_ts"] = now_ts

        # 仅当单实例在线运行且返回有效耗速时自动校准对应卡型与内存档的实测单价（防止多卡汇总耗速污染单卡费率）
        if assignments_count == 1 and len(sessions) == 1 and 0.02 < rate_hourly < 200.0:
            s0 = sessions[0]
            rkey = self._gpu_rate_key(s0.get("accelerator"), s0.get("machine_shape"))
            if rkey:
                rates_map = self.state.setdefault(
                    "measured_gpu_rates", dict(DEFAULT_MEASURED_GPU_RATES)
                )
                rates_map[rkey] = round(rate_hourly, 2)

        if live.get("eligible_gpus") is not None:
            self.state["last_eligible_gpus"] = [
                str(g).upper() for g in live["eligible_gpus"] if g
            ]
        if live.get("eligible_tpus") is not None:
            self.state["last_eligible_tpus"] = [
                str(t).upper() for t in live["eligible_tpus"] if t
            ]
        # 若实测某型号 GPU 正在健康运行，则自动清除它的紧缺标记
        for s in sessions:
            acc = str(s.get("accelerator") or "").upper()
            if acc and acc in self._scarce_gpus:
                self._scarce_gpus.pop(acc, None)

        # 自动清理已停机实例对应的过期隧道地址缓存与内存态元数据/密钥
        active_eps = {str(s.get("endpoint")) for s in sessions if s.get("endpoint")}
        raw_cache = self.state.get("tunnel_cache")
        if isinstance(raw_cache, dict):
            self.state["tunnel_cache"] = {
                ep: url
                for ep, url in raw_cache.items()
                if ep in active_eps and isinstance(url, str) and _is_allowed_tunnel_url(url)
            }
        else:
            self.state["tunnel_cache"] = {}
        for stale_ep in list(self._service_meta_by_ep.keys()):
            if stale_ep not in active_eps:
                self._service_meta_by_ep.pop(stale_ep, None)
        for stale_ep in list(self._runtime_secrets.keys()):
            if stale_ep not in active_eps:
                self._runtime_secrets.pop(stale_ep, None)

        self._sync_active_account_ledger(balance=balance, is_running=is_running)

    def _get_onboarding_state(self) -> dict[str, Any]:
        """组装前端第 5 页（账号保险箱与运行环境）所需的安全状态快照，绝不返回明文凭据。"""
        prefs = self.state.get("prefs") or {}
        colab_exe = self._get_colab_exe()
        colab_py = self._get_colab_py()
        rclone_exe = self._get_rclone_exe()
        rclone_conf = self._get_rclone_conf()

        colab_cli_ok = bool(colab_py and colab_py.is_file())
        colab_exe_ok = bool(colab_exe and colab_exe.is_file())
        rclone_exe_ok = bool(rclone_exe and rclone_exe.is_file())
        gdrive_ok = self._has_gdrive_save_section(rclone_conf)

        profiles_raw = self.state.get("account_profiles") or []
        profiles = [dict(p) for p in profiles_raw if isinstance(p, dict)]
        active_id = str(prefs.get("active_account_id") or "").strip()
        if not active_id and profiles and not self._auth_missing:
            active_id = str(profiles[0].get("id") or "")
        active_acct = next((p for p in profiles if p.get("id") == active_id), None)

        auth_initializing = bool(
            colab_cli_ok
            and not self._auth_missing
            and active_acct is None
            and self.state.get("last_balance") is None
        )
        colab_auth_ok = bool(
            colab_cli_ok
            and not self._auth_missing
            and (active_acct is not None or self.state.get("last_balance") is not None or auth_initializing)
        )

        # 超时 20 分钟自动失效内存态等待中授权码流程
        if self._pending_oauth_verifier and (time.monotonic() - self._pending_oauth_ts) > 1200.0:
            self._pending_oauth_verifier = ""

        cf_cfg = _load_cf_secret_raw()
        has_token = bool(str(cf_cfg.get("tunnel_token") or "").strip())
        has_fixed_key = len(str(cf_cfg.get("llm_api_key") or "").strip()) >= 16
        comfy_u = str(cf_cfg.get("comfy_url") or "").strip().rstrip("/")
        llm_u = str(cf_cfg.get("llm_url") or "").strip().rstrip("/")

        oc = self.state.get("onboarding_cache") or {}
        return {
            "colab_cli_ok": colab_cli_ok,
            "colab_exe_ok": colab_exe_ok,
            "rclone_exe_ok": rclone_exe_ok,
            "colab_auth_ok": colab_auth_ok,
            "auth_initializing": auth_initializing,
            "auth_missing": bool(self._auth_missing),
            "auth_expired": bool(self._auth_expired),
            "oauth_awaiting_code": bool(self._pending_oauth_verifier),
            "active_account_id": active_id,
            "active_account_label": str(active_acct.get("label") or "") if active_acct else "",
            "accounts": profiles,
            "gdrive_ok": gdrive_ok,
            "gdrive_binding": bool(self._gdrive_bind_proc is not None),
            "gdrive_free_gb": oc.get("gdrive_free_gb"),
            "gdrive_total_gb": oc.get("gdrive_total_gb"),
            "cf_tunnel": {
                "mode": "named" if has_token else "quick",
                "has_token": has_token,
                "has_fixed_llm_key": has_fixed_key,
                "comfy_url": comfy_u,
                "llm_url": llm_u,
            },
            "tool_paths": {
                "colab_exe": str(colab_exe) if colab_exe else "",
                "colab_py": str(colab_py) if colab_py else "",
                "rclone_exe": str(rclone_exe) if rclone_exe else "",
                "rclone_conf": str(rclone_conf) if rclone_conf else "",
                "custom_colab_exe": bool(str(prefs.get("colab_exe_path") or "").strip()),
                "custom_colab_py": bool(str(prefs.get("colab_py_path") or "").strip()),
                "custom_rclone_exe": bool(str(prefs.get("rclone_exe_path") or "").strip()),
                "custom_rclone_conf": bool(str(prefs.get("rclone_conf_path") or "").strip()),
            },
            "needs_attention": not (colab_cli_ok and rclone_exe_ok and colab_auth_ok and gdrive_ok),
        }

    def _build_snapshot(
        self,
        live_data: dict[str, Any] | None,
        stale: bool,
        error_msg: str | None = None,
    ) -> dict[str, Any]:
        """拼装前端所需的完整状态快照。"""
        now_ts = int(time.time())
        prefs = dict(DEFAULT_PREFS)
        prefs.update(self.state.get("prefs") or {})

        if live_data is not None:
            balance = float(live_data["balance"])
            rate_hourly = float(live_data["rate_hourly"])
            assignments_count = int(live_data["assignments_count"])
            sessions = list(live_data.get("sessions", []))
            eligible_gpus = live_data.get("eligible_gpus")
            eligible_tpus = live_data.get("eligible_tpus")
        else:
            balance = self.state.get("last_balance")
            rate_hourly = float(self.state.get("last_rate_hourly") or 0.0)
            assignments_count = int(self.state.get("last_assignments_count") or 0)
            sessions = list(self.state.get("last_sessions") or [])
            eligible_gpus = self.state.get("last_eligible_gpus")
            eligible_tpus = self.state.get("last_eligible_tpus")

        scarce_gpus = self._get_active_scarce_gpus()
        deploy_state = self.get_deploy_state()

        baseline = float(self.state.get("baseline_balance") or 1000.0)
        if balance is not None and balance > baseline:
            baseline = _default_baseline(float(balance))

        remaining_hours: float | None = None
        if balance is not None and rate_hourly > 0.01:
            remaining_hours = round(float(balance) / rate_hourly, 2)

        active_since = self.state.get("active_since_ts")
        active_minutes = (
            max(0, int((now_ts - int(active_since)) // 60)) if active_since else 0
        )

        updated_ts = self.state.get("last_updated_ts")
        updated_str = (
            datetime.fromtimestamp(int(updated_ts)).strftime("%H:%M:%S")
            if updated_ts
            else "--:--:--"
        )

        # 确定整体状态与推荐表情（idle / proud / sad）
        is_running = assignments_count > 0 or rate_hourly > 0.01
        warn_bal = float(prefs.get("warn_balance", 50.0))
        warn_hrs = float(prefs.get("warn_hours", 10.0))
        long_warn = int(prefs.get("long_run_warn_minutes", 180))

        low_balance = balance is not None and (
            float(balance) <= warn_bal
            or (remaining_hours is not None and remaining_hours <= warn_hrs)
        )
        long_running_alert = is_running and active_minutes >= long_warn
        has_orphan = any(s.get("orphan") for s in sessions)
        comfy = self.get_comfy_state()

        if error_msg and balance is None:
            if error_msg == "正在连接 Colab...":
                mood = "idle"
                status_level = "running"
                status_label = "正在连接..."
            elif self._auth_missing:
                mood = "sad" if self._auth_expired else "idle"
                status_level = "warn"
                status_label = "登录已失效" if self._auth_expired else "待绑定账号"
            else:
                mood = "sad"
                status_level = "error"
                status_label = "连接异常"
        elif low_balance:
            mood = "sad"
            status_level = "danger"
            status_label = "余额预警"
        elif stale and error_msg:
            mood = "sad"
            status_level = "warn"
            status_label = "网络波动(缓存)"
        elif is_running and comfy.get("has_error"):
            mood = "sad"
            status_level = "danger"
            status_label = "推理报错中断"
        elif is_running:
            mood = "sad" if long_running_alert else "idle"
            status_level = "warn" if (long_running_alert or has_orphan) else "running"
            status_label = (
                "长时运行提醒"
                if long_running_alert
                else ("发现孤儿实例" if has_orphan else "GPU 运行计费中")
            )
        else:
            mood = "proud"
            status_level = "idle"
            status_label = "已停机省积分"

        balance_pct = (
            max(0.0, min(100.0, round(float(balance) / baseline * 100.0, 2)))
            if balance is not None and baseline > 0
            else 0.0
        )

        # 烧钱速度相对基准条（按常见 0~16 CU/hr 映射，最高 100%）
        rate_pct = max(0.0, min(100.0, round(rate_hourly / 15.0 * 100.0, 2)))

        # 保证看板与小账本两位小数严格对账（今日起始 + 今日充值 - 今日实耗 == 当前余额）
        disp_bal = round(float(balance), 2) if balance is not None else None
        raw_day_start = self.state.get("day_start_balance")
        disp_day_start = round(float(raw_day_start), 2) if raw_day_start is not None else None
        disp_day_topup = round(float(self.state.get("day_topup_cu") or 0.0), 2)
        if disp_bal is not None and disp_day_start is not None:
            disp_day_consumed = round(max(0.0, disp_day_start + disp_day_topup - disp_bal), 2)
        else:
            disp_day_consumed = round(float(self.state.get("day_consumed_cu") or 0.0), 2)

        sess_start = self.state.get("session_start_balance")
        if is_running and sess_start is None and active_since and balance is not None:
            ref_ts = int(updated_ts or now_ts)
            est = rate_hourly * max(0, ref_ts - int(active_since)) / 3600.0
            cap = (
                float(raw_day_start if raw_day_start is not None else balance)
                + float(self.state.get("day_topup_cu") or 0.0)
            )
            sess_start = (
                min(cap, float(balance) + max(0.0, est))
                if cap >= float(balance)
                else (float(balance) + max(0.0, est))
            )
        disp_session_consumed = (
            round(max(0.0, round(float(sess_start), 2) - disp_bal), 2)
            if (is_running and sess_start is not None and disp_bal is not None)
            else 0.0
        )

        return {
            "ok": error_msg is None,
            "stale": stale,
            "error_msg": error_msg,
            "balance": disp_bal,
            "baseline_balance": round(baseline, 2),
            "balance_pct": balance_pct,
            "rate_hourly": round(rate_hourly, 2),
            "rate_pct": rate_pct,
            "remaining_hours": remaining_hours,
            "assignments_count": assignments_count,
            "sessions": sessions,
            "is_running": is_running,
            "has_orphan": has_orphan,
            "active_minutes": active_minutes,
            "active_since_ts": int(active_since) if (is_running and active_since) else None,
            "long_running_alert": long_running_alert,
            "low_balance": low_balance,
            "mood": mood,
            "status_level": status_level,
            "status_label": status_label,
            "date": self.state.get("date"),
            "day_start_balance": disp_day_start,
            "day_consumed_cu": disp_day_consumed,
            "day_topup_cu": disp_day_topup,
            "session_consumed_cu": disp_session_consumed,
            "last_delta": round(float(self.state.get("last_delta") or 0.0), 3),
            "updated_str": updated_str,
            "updated_ts": updated_ts,
            "eligible_gpus": eligible_gpus,
            "eligible_tpus": eligible_tpus,
            "scarce_gpus": scarce_gpus,
            "measured_gpu_rates": dict(
                self.state.get("measured_gpu_rates") or DEFAULT_MEASURED_GPU_RATES
            ),
            "deploy": deploy_state,
            "prefs": prefs,
            "comfy": comfy,
            "onboarding": self._get_onboarding_state(),
        }

    def _rebuild_and_emit_snapshot(
        self, stale: bool | None = None, error_msg: str | None = None
    ) -> dict[str, Any]:
        """在锁外广播基于当前内存状态的最新快照，避免前端状态滞后。"""
        with self._lock:
            use_stale = bool(self.last_snapshot.get("stale", False)) if stale is None else bool(stale)
            use_err = self.last_snapshot.get("error_msg") if error_msg is None else error_msg
            snap = self._build_snapshot(live_data=None, stale=use_stale, error_msg=use_err)
            self.last_snapshot = snap
        self._emit_event({"type": "state_sync", "snapshot": snap})
        return snap

    def fetch_live_status(self) -> dict[str, Any]:
        """串行刷新一次线上状态并广播给前端；未登录快速收敛至引导态，单次网络瞬断沿用近期缓存。"""
        with self._fetch_lock:
            try:
                live = self._query_colab_live()
                if live.get("auth_missing"):
                    with self._lock:
                        self._auth_missing = True
                        self._auth_expired = bool(live.get("auth_expired"))
                        self._consecutive_failures = 0
                        # 未登录或授权吊销时清空待展示余额，确保界面精准提示去绑定/重登
                        for lk in (
                            "last_balance",
                            "last_rate_hourly",
                            "last_assignments_count",
                            "last_sessions",
                            "active_since_ts",
                            "active_endpoint",
                            "session_start_balance",
                        ):
                            self.state[lk] = [] if lk == "last_sessions" else (0 if lk == "last_assignments_count" else (0.0 if lk == "last_rate_hourly" else None))
                        self._service_meta_by_ep.clear()
                        self._runtime_secrets.clear()
                        self._save_state()
                        snap = self._build_snapshot(
                            live_data=None,
                            stale=True,
                            error_msg=("Google 账号授权已失效，请重新绑定" if self._auth_expired else "尚未绑定 Google Colab 账号"),
                        )
                        self.last_snapshot = snap
                    self._reset_comfy_offline()
                    self._emit_event({"type": "state_sync", "snapshot": snap})
                    return snap

                with self._lock:
                    self._auth_missing = False
                    self._auth_expired = False
                    self._consecutive_failures = 0
                    self._update_ledger(live)
                    self._save_state()
                    snap = self._build_snapshot(live_data=live, stale=False, error_msg=None)
                    self.last_snapshot = snap
            except Exception as exc:
                with self._lock:
                    self._consecutive_failures += 1
                    has_balance = self.state.get("last_balance") is not None
                    last_updated = int(self.state.get("last_updated_ts") or 0)
                    within_grace = has_balance and (int(time.time()) - last_updated < 180)
                    if self._consecutive_failures == 1 and within_grace:
                        snap = self._build_snapshot(live_data=None, stale=False, error_msg=None)
                    else:
                        snap = self._build_snapshot(
                            live_data=None,
                            stale=True,
                            error_msg=str(exc),
                        )
                    self.last_snapshot = snap
            self._emit_event({"type": "state_sync", "snapshot": snap})
            return snap

    def get_cached_snapshot(self) -> dict[str, Any]:
        """返回最近一次状态快照。"""
        with self._lock:
            return dict(self.last_snapshot)

    def touch_activity(self) -> None:
        """刷新近期交互/推理活跃时间戳，用于智能保活与闲置自动停卡防误杀。"""
        with self._lock:
            self._last_activity_m = time.monotonic()
            self._idle_warned_ep = None

    def update_prefs(self, updates: dict[str, Any]) -> dict[str, Any]:
        """保存窗口位置、置顶状态、缩放比例与工作流启动参数（自动校验合法范围，不保存敏感项）。"""
        with self._lock:
            prefs = self.state.setdefault("prefs", dict(DEFAULT_PREFS))
            upd = updates or {}
            for k, v in upd.items():
                if k in DEFAULT_PREFS:
                    prefs[k] = v
            if "boot_profile" in upd and "boot_profiles" not in upd:
                prefs["boot_profiles"] = [upd["boot_profile"]]
            prefs = _sanitize_prefs_dict(prefs)
            self.state["prefs"] = prefs
            self._save_state()
            self.last_snapshot["prefs"] = dict(prefs)
            if upd.get("keep_alive") is True or "idle_auto_stop_minutes" in upd:
                self._last_keepalive_ts = 0.0
                self._last_activity_m = time.monotonic()
                self._idle_warned_ep = None
                self._wake_monitor.set()
            return dict(prefs)

    save_prefs = update_prefs

    def _pick_target_session(self, preferred: str = "comfy") -> dict[str, Any] | None:
        """优先选择指定会话，若不存在则取首个活跃会话。"""
        with self._lock:
            sessions = list(self.state.get("last_sessions") or [])
        for s in sessions:
            if s.get("name") == preferred:
                return s
        return sessions[0] if sessions else None

    def _has_running_session(self) -> bool:
        """判断当前缓存状态中是否存在活跃 Colab GPU 会话或计费速率。"""
        with self._lock:
            ac = int(self.state.get("last_assignments_count") or 0)
            rate = float(self.state.get("last_rate_hourly") or 0.0)
            sessions = list(self.state.get("last_sessions") or [])
        return ac > 0 or rate > 0.01 or len(sessions) > 0

    def _get_ep_meta_and_key(self, ep: str = "") -> tuple[dict[str, Any], str]:
        """线程安全读取指定 endpoint 的内存态服务元信息与访问密钥。"""
        with self._lock:
            if ep and ep in self._service_meta_by_ep:
                return dict(self._service_meta_by_ep[ep]), str(self._runtime_secrets.get(ep) or "")
            if not ep and self._service_meta_by_ep:
                first_ep, first_meta = next(iter(self._service_meta_by_ep.items()))
                return dict(first_meta), str(self._runtime_secrets.get(first_ep) or "")
            return {}, str(self._runtime_secrets.get(ep) or "") if ep else ""

    @staticmethod
    def _infer_kind_from_profiles(profiles: Any) -> str:
        """根据开机/部署工作流列表推断服务形态 ('comfy' | 'llm' | 'dual')。"""
        norm = _normalize_boot_profiles(profiles, fallback_profile="anima")
        has_llm = any(p in LLM_BOOT_PROFILES for p in norm)
        has_comfy = any(p in COMFY_BOOT_PROFILES for p in norm)
        if has_llm and has_comfy:
            return "dual"
        if has_llm:
            return "llm"
        return "comfy"

    def _get_ep_service_kind(self, ep: str = "") -> str:
        """根据 endpoint 元信息、当前在线监控态或开机预设推断当前服务类型 ('comfy' | 'llm' | 'dual')。"""
        with self._lock:
            target_ep = ep
            if not target_ep:
                sessions = list(self.state.get("last_sessions") or [])
                primary = next(
                    (s for s in sessions if s.get("name") == "comfy" and s.get("endpoint")),
                    None,
                ) or (sessions[0] if sessions else None)
                if primary and primary.get("endpoint"):
                    target_ep = str(primary["endpoint"])
            if target_ep and target_ep in self._service_meta_by_ep:
                k = str(self._service_meta_by_ep[target_ep].get("kind") or "comfy").lower()
                return k if k in ("comfy", "llm", "dual") else "comfy"
            if self._service_meta_by_ep:
                first_meta = next(iter(self._service_meta_by_ep.values()))
                k = str(first_meta.get("kind") or "comfy").lower()
                return k if k in ("comfy", "llm", "dual") else "comfy"
            prefs = self.state.get("prefs") or {}
            boot_profs = prefs.get("boot_profiles") or [prefs.get("boot_profile") or "anima"]
        with self._comfy_lock:
            cur_kind = str(self._comfy_state.get("service_kind") or "")
            if self._comfy_state.get("online") and cur_kind in ("comfy", "llm", "dual"):
                return cur_kind
        return self._infer_kind_from_profiles(boot_profs)

    def is_llm_mode_active(self) -> bool:
        """判断当前是否处于纯 LLM 单模模式（双栖 'dual' 模式返回 False）。"""
        if self._has_running_session():
            return self._get_ep_service_kind() == "llm"
        with self._lock:
            prefs = self.state.get("prefs") or {}
            profs = prefs.get("boot_profiles") or [prefs.get("boot_profile") or "anima"]
        return self._infer_kind_from_profiles(profs) == "llm"

    def is_dual_mode_active(self) -> bool:
        """判断当前活跃实例（或停机预设）是否处于 ComfyUI + LLM 双栖同驻模式。"""
        if self._has_running_session():
            return self._get_ep_service_kind() == "dual"
        with self._lock:
            prefs = self.state.get("prefs") or {}
            profs = prefs.get("boot_profiles") or [prefs.get("boot_profile") or "anima"]
        return self._infer_kind_from_profiles(profs) == "dual"

    def has_llm_service_active(self) -> bool:
        """判断当前是否包含在线 LLM 对话服务（纯 'llm' 或双栖 'dual'）。"""
        return self._get_ep_service_kind() in ("llm", "dual")

    def _fast_read_tunnel_url(self, ep: str, sname: str) -> tuple[str, str, str | None]:
        """通过 Colab Contents HTTP 接口免 WebSocket 速读 /root/colab_service_meta.json 或隧道文件。"""
        colab_py = self._get_colab_py()
        if not colab_py.exists():
            return "error", ep, None
        fast_proc = _run_hidden(
            [str(colab_py), "-c", _COLAB_FAST_TUNNEL_SCRIPT, ep, sname],
            timeout=15,
        )
        if fast_proc.returncode != 0:
            return "error", ep, None
        m_json = re.search(
            r"<<<COLAB_TUNNEL>>>(.*?)<<<END_COLAB_TUNNEL>>>",
            fast_proc.stdout or "",
            re.S,
        )
        if not m_json:
            return "error", ep, None
        try:
            tdata = json.loads(m_json.group(1))
        except Exception:
            tdata = {}
        tstatus = str(tdata.get("status") or "missing")
        if tstatus == "no_session":
            with self._lock:
                self.state["last_sessions"] = []
                self.state["last_assignments_count"] = 0
                self.state["tunnel_cache"] = {}
                self._service_meta_by_ep.clear()
                self._runtime_secrets.clear()
                self._save_state()
            return "no_session", ep, None
        resolved_ep = str(tdata.get("endpoint") or ep)
        fast_url = str(tdata.get("url") or "").strip() or None

        # 将 api_key 严格剥离至纯内存态字典，非敏感元信息存入 _service_meta_by_ep
        raw_meta = tdata.get("meta")
        api_key = str(tdata.get("api_key") or "").strip()
        if resolved_ep and isinstance(raw_meta, dict):
            if not api_key and raw_meta.get("api_key"):
                api_key = str(raw_meta.get("api_key") or "").strip()
            clean_meta = {k: v for k, v in raw_meta.items() if k != "api_key"}
            kind_val = str(clean_meta.get("kind") or "comfy").lower()
            clean_meta["kind"] = kind_val if kind_val in ("comfy", "llm", "dual") else "comfy"
            clean_meta.setdefault("port", 18080 if clean_meta["kind"] == "llm" else 8188)
            if clean_meta["kind"] == "dual":
                clean_meta.setdefault("llm_port", 18080)
                if fast_url and not clean_meta.get("comfy_tunnel_url"):
                    clean_meta["comfy_tunnel_url"] = fast_url
            if fast_url:
                clean_meta["tunnel_url"] = fast_url
            with self._lock:
                self._service_meta_by_ep[resolved_ep] = clean_meta
                if clean_meta["kind"] in ("llm", "dual"):
                    if api_key:
                        self._runtime_secrets[resolved_ep] = api_key
                else:
                    self._runtime_secrets.pop(resolved_ep, None)

        return tstatus, resolved_ep, fast_url

    @staticmethod
    def _read_cf_public_llm_url() -> str | None:
        """从本地云端隧道配置中仅读取公开 llm_url（绝不读取或暴露敏感 token/key）。"""
        if not _CF_SECRET_FILE.is_file():
            return None
        try:
            data = json.loads(_CF_SECRET_FILE.read_text(encoding="utf-8"))
        except Exception:
            return None
        if isinstance(data, dict):
            u = str(data.get("llm_url") or "").strip().rstrip("/")
            if u and _is_allowed_tunnel_url(u):
                return u
        return None

    def _infer_default_llm_model_name(self, meta: dict[str, Any] | None = None) -> str:
        """按服务元数据或活跃/预设工作流推断默认 LLM 模型显示名（27B vs 180B）。"""
        if isinstance(meta, dict) and meta.get("model"):
            return str(meta["model"])
        with self._comfy_lock:
            act_profs = list(self._comfy_state.get("active_profiles") or [])
        with self._lock:
            prefs = self.state.get("prefs") or {}
            pref_profs = _normalize_boot_profiles(
                prefs.get("boot_profiles"), prefs.get("boot_profile") or "anima"
            )
        profs = act_profs or pref_profs
        if "qwen_27b" in profs and "qwen_llm" not in profs:
            return "Qwen3.8-27B-abliterated"
        return "Qwen3.8-Flash-Next-abliterated"

    def resolve_active_tunnel_url(
        self, session: str = "comfy", allow_fast_discover: bool = True
    ) -> str | None:
        """获取当前可连通的主公网隧道地址（ComfyUI/dual 优先 Comfy 主页，纯 LLM 返回对话页，支持补水）。"""
        if not self._has_running_session():
            return None
        target = self._pick_target_session(session)
        ep = str(target.get("endpoint") or "") if target else ""
        sname = target["name"] if target and target.get("name") not in (None, "?") else session

        cached_url: str | None = None
        known_meta: dict[str, Any] | None = None
        preferred_kind = "comfy"
        with self._lock:
            cache = self.state.get("tunnel_cache") or {}
            if ep and ep in cache:
                cached_url = cache.get(ep)
            elif cache:
                cached_url = next(iter(cache.values()), None)
            if ep and ep in self._service_meta_by_ep:
                known_meta = dict(self._service_meta_by_ep[ep])
            prefs = self.state.get("prefs") or {}
            pref_profs = prefs.get("boot_profiles") or [prefs.get("boot_profile") or "anima"]
            if self._infer_kind_from_profiles(pref_profs) == "llm":
                preferred_kind = "llm"

        if known_meta is not None:
            mkind = str(known_meta.get("kind") or "comfy")
            if mkind == "dual":
                cand_comfy = str(known_meta.get("comfy_tunnel_url") or cached_url or "").strip()
                if cand_comfy and _probe_tunnel_alive(cand_comfy, timeout=3.0, kind="comfy"):
                    return cand_comfy
            elif cached_url and _probe_tunnel_alive(cached_url, timeout=3.0, kind=mkind):
                return cached_url
        elif cached_url:
            detected = _detect_tunnel_kind(cached_url, timeout=3.0, preferred=preferred_kind)
            if detected is not None:
                # 缓存存活但进程刚启动缺少元信息与密钥：主动补水一次
                if allow_fast_discover:
                    now_m = time.monotonic()
                    if now_m - self._last_tunnel_discover_ts >= 15.0:
                        self._last_tunnel_discover_ts = now_m
                        tstatus, resolved_ep, fast_url = self._fast_read_tunnel_url(ep, sname)
                        meta_hydrated, _ = self._get_ep_meta_and_key(resolved_ep or ep)
                        hkind = str(meta_hydrated.get("kind") or detected)
                        h_probe_k = "comfy" if hkind == "dual" else hkind
                        if (
                            tstatus == "ok"
                            and fast_url
                            and fast_url != cached_url
                            and _probe_tunnel_alive(fast_url, timeout=3.0, kind=h_probe_k)
                        ):
                            if resolved_ep:
                                with self._lock:
                                    self.state.setdefault("tunnel_cache", {})[resolved_ep] = fast_url
                                    self._save_state()
                            return fast_url
                if ep:
                    with self._lock:
                        self._service_meta_by_ep.setdefault(
                            ep,
                            {"kind": detected, "port": 18080 if detected == "llm" else 8188},
                        )
                return cached_url

        if not allow_fast_discover:
            return None
        now_m = time.monotonic()
        if now_m - self._last_tunnel_discover_ts < 45.0:
            return None
        self._last_tunnel_discover_ts = now_m

        tstatus, resolved_ep, fast_url = self._fast_read_tunnel_url(ep, sname)
        if tstatus == "ok" and fast_url:
            meta_after, _ = self._get_ep_meta_and_key(resolved_ep or ep)
            mkind_after = str(meta_after.get("kind") or "auto")
            probe_kind = "comfy" if mkind_after == "dual" else mkind_after
            if _probe_tunnel_alive(fast_url, timeout=3.5, kind=probe_kind):
                if resolved_ep:
                    with self._lock:
                        self.state.setdefault("tunnel_cache", {})[resolved_ep] = fast_url
                        self._save_state()
                return fast_url
        return None

    def resolve_active_llm_tunnel_url(
        self, session: str = "comfy", allow_fast_discover: bool = True
    ) -> str | None:
        """获取当前在线 LLM 服务公网隧道地址（支持纯 LLM 模式与 Comfy+LLM 双栖模式）。"""
        if not self._has_running_session():
            return None
        target = self._pick_target_session(session)
        ep = str(target.get("endpoint") or "") if target else ""
        sname = target["name"] if target and target.get("name") not in (None, "?") else session

        meta, _ = self._get_ep_meta_and_key(ep)
        mkind = str(meta.get("kind") or "").lower()
        if mkind == "llm":
            return self.resolve_active_tunnel_url(
                session=session, allow_fast_discover=allow_fast_discover
            )
        if mkind == "comfy" and not self.is_dual_mode_active():
            return None

        cand = str(meta.get("llm_tunnel_url") or "").strip() if mkind == "dual" else ""
        if cand and _probe_tunnel_alive(cand, timeout=3.0, kind="llm"):
            return cand

        if allow_fast_discover:
            now_m = time.monotonic()
            if (not meta) or (now_m - self._last_llm_tunnel_discover_ts >= 15.0):
                self._last_llm_tunnel_discover_ts = now_m
                tstatus, resolved_ep, fast_url = self._fast_read_tunnel_url(ep, sname)
                if tstatus == "no_session":
                    return None
                meta, _ = self._get_ep_meta_and_key(resolved_ep or ep)
                mkind = str(meta.get("kind") or "").lower()
                if mkind == "comfy" and not self.is_dual_mode_active():
                    return None
                cand = str(
                    meta.get("llm_tunnel_url") or (fast_url if mkind == "llm" else "") or ""
                ).strip()
                if cand and _probe_tunnel_alive(cand, timeout=3.0, kind="llm"):
                    return cand

        if mkind in ("dual", "llm") or self.has_llm_service_active():
            cf_llm = self._read_cf_public_llm_url()
            if cf_llm and _probe_tunnel_alive(cf_llm, timeout=3.0, kind="llm"):
                if ep:
                    with self._lock:
                        cur = self._service_meta_by_ep.setdefault(
                            ep, {"kind": "dual" if self.is_dual_mode_active() else "llm", "port": 8188}
                        )
                        cur["llm_tunnel_url"] = cf_llm
                return cf_llm
        return None

    _get_or_fetch_tunnel_url = resolve_active_tunnel_url

    @staticmethod
    def _build_comfy_open_url(
        tunnel_url: str,
        workflow_name: str | None = None,
        extra_names: list[str] | None = None,
    ) -> str:
        """构造 ComfyUI 打开链接；指定工作流名时附加一次性直铺画布查询参数。"""
        base = tunnel_url.rstrip("/")
        if not workflow_name:
            return base
        q = f"?pet_wf={urllib.parse.quote(workflow_name)}&pet_ts={int(time.time() * 1000)}"
        extras = [str(n) for n in (extra_names or []) if n and n != workflow_name]
        if extras:
            q += f"&pet_extra={urllib.parse.quote(','.join(extras))}"
        return f"{base}/{q}"

    def _ensure_legacy_cmd_file(self, base_url: str) -> None:
        """确保 userdata 根目录存在桥接指令文件，避免浏览器首次轮询出现 404 日志。"""
        base = base_url.rstrip("/")
        try:
            chk = self._http.get(f"{base}/userdata/colab_pet_bridge_cmd.json", timeout=4)
            if chk.status_code == 404:
                self._http.post(
                    f"{base}/userdata/colab_pet_bridge_cmd.json?overwrite=true",
                    json={"id": "", "name": "", "ts": 0},
                    timeout=5,
                )
        except Exception:
            pass

    def _ensure_direct_canvas_support(
        self, tunnel_url: str, ep: str = "", sname: str = "comfy"
    ) -> tuple[str, bool]:
        """探测或免重启热注入/热升级云端 ComfyUI 直铺画布前端扩展，返回 `(模式, 本次是否刚安装)`。"""
        base = tunnel_url.rstrip("/")
        cache_key = ep or base
        with self._canvas_ext_lock:
            cached = self._canvas_ext_ready_eps.get(cache_key)
        if cached in ("native", "legacy"):
            return cached, False

        native_route_alive = False
        # 1) 优先探测新版实例内置自定义节点接口（若服务端仍是 v1 则进一步校验 JS 是否已热升级到 v2）
        for attempt in range(2):
            try:
                r = self._http.get(f"{base}/colab_pet/ping", timeout=4)
                if r.status_code == 200:
                    pdata = r.json()
                    if isinstance(pdata, dict) and pdata.get("ok"):
                        native_route_alive = True
                        ver = int(pdata.get("version") or 1)
                        if ver >= 2:
                            with self._canvas_ext_lock:
                                self._canvas_ext_ready_eps[cache_key] = "native"
                            return "native", False
                        try:
                            jr = self._http.get(
                                f"{base}/extensions/colab_pet_bridge/direct_canvas.js",
                                timeout=4,
                            )
                            if jr.status_code == 200 and "colab_pet_direct_canvas v2" in (jr.text or ""):
                                with self._canvas_ext_lock:
                                    self._canvas_ext_ready_eps[cache_key] = "native"
                                return "native", False
                        except Exception:
                            pass
                break
            except Exception:
                if attempt == 0:
                    time.sleep(0.25)

        # 2) 当服务端路由不存在时，探测已挂载的前端扩展（仅认可 v2，旧 v1 将自动覆盖升级）
        if not native_route_alive:
            for attempt in range(2):
                try:
                    r2 = self._http.get(f"{base}/extensions/colab_pet/direct_canvas.js", timeout=4)
                    if r2.status_code == 200 and "colab_pet_direct_canvas v2" in (r2.text or ""):
                        self._ensure_legacy_cmd_file(base)
                        with self._canvas_ext_lock:
                            self._canvas_ext_ready_eps[cache_key] = "legacy"
                        return "legacy", False
                    break
                except Exception:
                    if attempt == 0:
                        time.sleep(0.25)

        # 3) 存量实例尚未挂载或仍为 v1 时，经 Colab Contents REST 免重启热写入/覆盖 v2 扩展
        colab_py = self._get_colab_py()
        if not colab_py.exists() or not self._has_running_session():
            return ("native" if native_route_alive else "none"), False

        try:
            payload = {
                "js_b64": base64.b64encode(_COMFY_DIRECT_CANVAS_JS.encode("utf-8")).decode("ascii"),
                "py_b64": base64.b64encode(_COMFY_BRIDGE_NODE_INIT_PY.encode("utf-8")).decode("ascii"),
            }
            proc = subprocess.run(
                [str(colab_py), "-c", _COLAB_INSTALL_CANVAS_EXT_SCRIPT, ep, sname],
                input=json.dumps(payload, ensure_ascii=False),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
                env=_clean_env(),
                creationflags=CREATE_NO_WINDOW,
            )
            if proc.returncode != 0:
                return ("native" if native_route_alive else "none"), False

            # 验证静态资源已立刻通过 ComfyUI 隧道返回 v2 内容
            check_urls = (
                [
                    f"{base}/extensions/colab_pet_bridge/direct_canvas.js",
                    f"{base}/extensions/colab_pet/direct_canvas.js",
                ]
                if native_route_alive
                else [f"{base}/extensions/colab_pet/direct_canvas.js"]
            )
            for _ in range(3):
                for cu in check_urls:
                    try:
                        vr = self._http.get(cu, timeout=4)
                        if vr.status_code == 200 and "colab_pet_direct_canvas v2" in (vr.text or ""):
                            target_mode = "native" if native_route_alive else "legacy"
                            if target_mode == "legacy":
                                self._ensure_legacy_cmd_file(base)
                            with self._canvas_ext_lock:
                                self._canvas_ext_ready_eps[cache_key] = target_mode
                            return target_mode, True
                    except Exception:
                        pass
                time.sleep(0.25)
        except Exception:
            pass
        return ("native" if native_route_alive else "none"), False

    def _push_or_open_workflow_on_canvas(
        self,
        tunnel_url: str,
        safe_name: str,
        extra_names: list[str] | None = None,
        session: str = "comfy",
    ) -> dict[str, Any]:
        """优先向已打开的 ComfyUI 浏览器标签页实时推送工作流；无标签页回执时自动拉起浏览器直铺。"""
        base = tunnel_url.rstrip("/")
        target = self._pick_target_session(session)
        ep = str(target.get("endpoint") or "") if target else ""
        sname = target["name"] if target and target.get("name") not in (None, "?") else session

        mode, just_installed = self._ensure_direct_canvas_support(base, ep=ep, sname=sname)
        req_id = uuid.uuid4().hex[:12]
        open_url = self._build_comfy_open_url(base, safe_name, extra_names=extra_names)

        if mode == "native" and not just_installed:
            try:
                pr = self._http.post(
                    f"{base}/colab_pet/open_workflow",
                    json={"id": req_id, "name": safe_name, "extra_names": extra_names or []},
                    timeout=5,
                )
                if pr.status_code == 200:
                    pdata = pr.json()
                    if int(pdata.get("browser_clients") or 0) > 0:
                        deadline = time.monotonic() + 3.2
                        while time.monotonic() < deadline:
                            time.sleep(0.18)
                            try:
                                ar = self._http.get(
                                    f"{base}/colab_pet/ack_workflow",
                                    params={"id": req_id},
                                    timeout=3,
                                )
                                if ar.status_code == 200 and ar.json().get("acked"):
                                    return {
                                        "pushed_live": True,
                                        "opened_browser": False,
                                        "url": open_url,
                                        "mode": mode,
                                    }
                            except Exception:
                                pass
            except Exception:
                pass
        elif mode == "legacy" and not just_installed:
            try:
                pr = self._http.post(
                    f"{base}/userdata/colab_pet_bridge_cmd.json?overwrite=true",
                    json={
                        "id": req_id,
                        "name": safe_name,
                        "extra_names": extra_names or [],
                        "ts": time.time(),
                    },
                    timeout=5,
                )
                if pr.status_code in (200, 201):
                    deadline = time.monotonic() + 3.5
                    while time.monotonic() < deadline:
                        time.sleep(0.25)
                        try:
                            ar = self._http.get(
                                f"{base}/userdata/colab_pet_bridge_ack.json",
                                timeout=3,
                            )
                            if ar.status_code == 200 and str((ar.json() or {}).get("id") or "") == req_id:
                                return {
                                    "pushed_live": True,
                                    "opened_browser": False,
                                    "url": open_url,
                                    "mode": mode,
                                }
                        except Exception:
                            pass
            except Exception:
                pass

        webbrowser.open(open_url)
        return {
            "pushed_live": False,
            "opened_browser": True,
            "url": open_url,
            "mode": mode,
        }

    def _backup_workflow_to_gdrive_async(self, safe_name: str, wf_bytes: bytes) -> None:
        """在后台静默同步一份工作流 JSON 至 Google Drive，实现跨实例开机自动继承。"""
        rclone_exe = self._get_rclone_exe()
        rclone_conf = self._get_rclone_conf()
        if not rclone_exe.is_file() or not rclone_conf.is_file():
            return

        def _bg() -> None:
            try:
                with tempfile.TemporaryDirectory(prefix="pet_wf_bak_") as tmp_dir:
                    tmp_wf = Path(tmp_dir) / safe_name
                    tmp_wf.write_bytes(wf_bytes)
                    _run_hidden(
                        [
                            str(rclone_exe),
                            "copyto",
                            "--config",
                            str(rclone_conf),
                            "--disable-http2",
                            str(tmp_wf),
                            f"{GDRIVE_WF_REMOTE}/{safe_name}",
                        ],
                        timeout=45,
                    )
            except Exception:
                pass

        threading.Thread(target=_bg, daemon=True).start()

    def _open_browser_for_comfy(
        self,
        tunnel_url: str,
        ep: str = "",
        sname: str = "comfy",
        workflow_name: str | None = None,
    ) -> str:
        """装配 ComfyUI 打开链接并拉起默认浏览器（若携带工作流则顺带确保扩展就绪）。"""
        if workflow_name:
            self._ensure_direct_canvas_support(tunnel_url, ep=ep, sname=sname)
        open_url = self._build_comfy_open_url(tunnel_url, workflow_name)
        webbrowser.open(open_url)
        return open_url

    def _open_resolved_service_url(
        self,
        tunnel_url: str,
        ep: str = "",
        sname: str = "comfy",
        workflow_name: str | None = None,
        stage: str = "normal",
        detected_kind: str | None = None,
        already_fetched_meta: bool = False,
    ) -> dict[str, Any]:
        """根据解析出的存活隧道地址分流执行打开动作（LLM 直开对话页并回传密钥，ComfyUI 打开画布）。"""
        meta, api_key = self._get_ep_meta_and_key(ep)
        resolved_ep = ep
        if not meta and not already_fetched_meta:
            _, r_ep, _ = self._fast_read_tunnel_url(ep, sname)
            if r_ep:
                resolved_ep = r_ep
            meta, api_key = self._get_ep_meta_and_key(resolved_ep)
            already_fetched_meta = True

        raw_kind = str(meta.get("kind") or detected_kind or "comfy").lower()
        kind = str(detected_kind or "comfy").lower() if raw_kind == "dual" else raw_kind
        if kind == "llm":
            if not api_key and not already_fetched_meta:
                _, r_ep, _ = self._fast_read_tunnel_url(resolved_ep, sname)
                if r_ep:
                    resolved_ep = r_ep
                _, api_key = self._get_ep_meta_and_key(resolved_ep)
            clean_url = tunnel_url.strip().rstrip("/")
            webbrowser.open(clean_url)
            msg = (
                "💬 已打开 Qwen3.8 对话页 · 访问密钥已复制，首次直接 Ctrl+V 登录"
                if api_key
                else "💬 已打开 Qwen3.8 对话页"
            )
            return {
                "ok": True,
                "url": clean_url,
                "service_kind": "llm",
                "clipboard_text": api_key,
                "key_copied": bool(api_key),
                "message": msg,
            }

        open_url = self._open_browser_for_comfy(
            tunnel_url, ep=resolved_ep, sname=sname, workflow_name=workflow_name
        )
        if workflow_name:
            if stage == "restarted":
                msg = f"已重启隧道并打开工作流: {workflow_name}"
            else:
                msg = f"已在画布打开工作流: {workflow_name}"
        else:
            if stage == "cache":
                msg = "已秒开 ComfyUI 隧道 (缓存命中)"
            elif stage == "restarted":
                msg = "已重启并打开 ComfyUI 隧道网页"
            else:
                msg = "已打开 ComfyUI 隧道网页"
        return {"ok": True, "url": open_url, "service_kind": "comfy", "message": msg}

    def open_comfy_tunnel(
        self, session: str = "comfy", workflow_name: str | None = None
    ) -> dict[str, Any]:
        """三级极速打开云端 ComfyUI / LLM 隧道：本地缓存秒开 -> HTTP Contents 直读 -> 自动重启兜底。"""
        self.touch_activity()
        if self.is_deploying():
            return {"ok": False, "message": "正在执行 7 步开机部署，请等待就绪后再打开"}
        if not self._action_lock.acquire(blocking=False):
            return {"ok": False, "message": "另一项操作正在执行，请稍等片刻"}
        try:
            # 0. 若本地暂无活跃会话缓存则即时探活；确无实例时秒级短路提示
            target = self._pick_target_session(session)
            if target is None:
                snap = self.fetch_live_status()
                target = self._pick_target_session(session)
                if target is None and snap.get("assignments_count", 0) == 0 and not snap.get("sessions"):
                    return {"ok": False, "message": "当前无运行中的 Colab 实例，请先部署开机"}

            ep = str(target.get("endpoint") or "") if target else ""
            sname = target["name"] if target and target.get("name") not in (None, "?") else session
            is_orphan = bool(target and target.get("orphan") and target.get("name") in (None, "?"))

            # 1. 第一级：本地持久化公网域名缓存命中且双模探活通过 -> <1s 秒开
            cached_url = None
            known_meta, _ = self._get_ep_meta_and_key(ep)
            sk_guess = self._get_ep_service_kind(ep)
            preferred_kind = "llm" if sk_guess == "llm" else "comfy"
            if ep:
                with self._lock:
                    cached_url = (self.state.get("tunnel_cache") or {}).get(ep)
            if known_meta:
                mkind = str(known_meta.get("kind") or "comfy")
                if mkind == "dual":
                    cand_comfy = str(known_meta.get("comfy_tunnel_url") or cached_url or "").strip()
                    if cand_comfy and _probe_tunnel_alive(cand_comfy, timeout=3.0, kind="comfy"):
                        return self._open_resolved_service_url(
                            cand_comfy,
                            ep=ep,
                            sname=sname,
                            workflow_name=workflow_name,
                            stage="cache",
                            detected_kind="comfy",
                        )
                elif cached_url and _probe_tunnel_alive(cached_url, timeout=3.0, kind=mkind):
                    return self._open_resolved_service_url(
                        cached_url,
                        ep=ep,
                        sname=sname,
                        workflow_name=workflow_name,
                        stage="cache",
                        detected_kind=mkind,
                    )
            elif cached_url:
                det_kind = _detect_tunnel_kind(cached_url, timeout=3.0, preferred=preferred_kind)
                if det_kind is not None:
                    return self._open_resolved_service_url(
                        cached_url,
                        ep=ep,
                        sname=sname,
                        workflow_name=workflow_name,
                        stage="cache",
                        detected_kind=det_kind,
                    )

            # 2. 第二级：通过 Colab Contents HTTP REST 免 WebSocket 直读服务元数据契约或隧道文件
            tstatus, resolved_ep, fast_url = self._fast_read_tunnel_url(ep, sname)
            if tstatus == "no_session":
                return {"ok": False, "message": "当前无运行中的 Colab 实例，请先部署开机"}
            meta_fast, _ = self._get_ep_meta_and_key(resolved_ep or ep)
            fast_kind = str(meta_fast.get("kind") or "auto")
            cand_fast = (
                str(meta_fast.get("comfy_tunnel_url") or fast_url or "").strip()
                if fast_kind == "dual"
                else (fast_url or "")
            )
            probe_fast_k = "comfy" if fast_kind == "dual" else fast_kind
            if tstatus == "ok" and cand_fast and _probe_tunnel_alive(cand_fast, timeout=4.0, kind=probe_fast_k):
                if resolved_ep:
                    with self._lock:
                        self.state.setdefault("tunnel_cache", {})[resolved_ep] = cand_fast
                        self._save_state()
                resolved_kind = (
                    "comfy"
                    if fast_kind == "dual"
                    else (fast_kind if fast_kind in ("comfy", "llm") else _detect_tunnel_kind(cand_fast, timeout=2.5))
                )
                return self._open_resolved_service_url(
                    cand_fast,
                    ep=resolved_ep or ep,
                    sname=sname,
                    workflow_name=workflow_name,
                    stage="normal",
                    detected_kind=resolved_kind,
                    already_fetched_meta=True,
                )

            # 网页端开启的孤儿实例无法通过本地 session 名称重建隧道，直接给出清晰引导
            if is_orphan:
                return {
                    "ok": False,
                    "message": "该实例为网页孤儿实例且未检测到可用隧道，请用面板一键开机或在网页端重启隧道",
                }

            # 3. 第三级：前两级均不可用时回退 colab_ctl.py tunnel --restart 重建隧道
            if not CTL_SCRIPT.exists():
                return {"ok": False, "message": f"未找到控制脚本: {CTL_SCRIPT.name}"}
            proc = _run_hidden(
                [_hidden_python_exe(), str(CTL_SCRIPT), "tunnel", "--restart", "-s", sname],
                timeout=60,
                extra_env=self._build_ctl_env(),
            )
            output = f"{proc.stdout}\n{proc.stderr}"
            match = re.search(
                r"(?:(?:COMFYUI|SERVICE)_TUNNEL_URL:\s*|\[(?:ComfyUI|LLM|Qwen[^]]*) WebUI 地址\]\s*)(https://(?!api\.)(?:[a-z0-9-]+\.)+[a-z]{2,}(?::\d+)?)",
                output,
            )
            if proc.returncode == 0 and match:
                url = match.group(1).strip()
                use_ep = resolved_ep or ep
                if use_ep:
                    with self._lock:
                        self.state.setdefault("tunnel_cache", {})[use_ep] = url
                        if use_ep in self._service_meta_by_ep:
                            self._service_meta_by_ep[use_ep]["tunnel_url"] = url
                            if self._service_meta_by_ep[use_ep].get("kind") == "dual":
                                self._service_meta_by_ep[use_ep]["comfy_tunnel_url"] = url
                        self._save_state()
                det_after = (
                    "comfy"
                    if fast_kind == "dual"
                    else _detect_tunnel_kind(url, timeout=3.0, preferred=preferred_kind)
                )
                return self._open_resolved_service_url(
                    url,
                    ep=use_ep,
                    sname=sname,
                    workflow_name=workflow_name,
                    stage="restarted",
                    detected_kind=det_after,
                    already_fetched_meta=bool(meta_fast),
                )

            if ep:
                with self._lock:
                    cache = self.state.get("tunnel_cache")
                    if isinstance(cache, dict) and ep in cache:
                        cache.pop(ep, None)
                        self._save_state()
            tail = (output.strip().splitlines() or ["未获取到隧道地址"])[-1]
            return {"ok": False, "message": f"隧道开启失败: {tail}"}
        except Exception as exc:
            return {"ok": False, "message": f"隧道操作异常: {exc}"}
        finally:
            self._action_lock.release()

    def open_llm_tunnel(self, session: str = "comfy") -> dict[str, Any]:
        """在双栖或纯 LLM 实例下专一打开 Qwen3.8 对话网页并复制密钥。"""
        self.touch_activity()
        if self.is_deploying():
            return {"ok": False, "message": "正在执行 7 步开机部署，请等待就绪后再打开"}
        if not self._action_lock.acquire(blocking=False):
            return {"ok": False, "message": "另一项操作正在执行，请稍等片刻"}
        try:
            target = self._pick_target_session(session)
            if target is None:
                snap = self.fetch_live_status()
                target = self._pick_target_session(session)
                if target is None and snap.get("assignments_count", 0) == 0 and not snap.get("sessions"):
                    return {"ok": False, "message": "当前无运行中的 Colab 实例，请先部署开机"}

            ep = str(target.get("endpoint") or "") if target else ""
            sname = target["name"] if target and target.get("name") not in (None, "?") else session
            is_orphan = bool(target and target.get("orphan") and target.get("name") in (None, "?"))

            llm_url = self.resolve_active_llm_tunnel_url(session=sname, allow_fast_discover=True)
            if llm_url:
                return self._open_resolved_service_url(
                    llm_url,
                    ep=ep,
                    sname=sname,
                    stage="cache",
                    detected_kind="llm",
                    already_fetched_meta=bool(self._service_meta_by_ep.get(ep)),
                )

            # 显式按钮点击绕过 discover 冷却直读一次远端元数据契约
            tstatus, resolved_ep, fast_url = self._fast_read_tunnel_url(ep, sname)
            if tstatus == "no_session":
                return {"ok": False, "message": "当前无运行中的 Colab 实例，请先部署开机"}
            use_ep = resolved_ep or ep
            meta_fast, _ = self._get_ep_meta_and_key(use_ep)
            fast_kind = str(meta_fast.get("kind") or "").lower()
            if fast_kind == "comfy" and not self.is_dual_mode_active():
                return {"ok": False, "message": "当前运行的是纯 ComfyUI 实例，未挂载 LLM 对话服务"}

            cand_llm = str(
                meta_fast.get("llm_tunnel_url") or (fast_url if fast_kind == "llm" else "") or ""
            ).strip()
            if cand_llm and _probe_tunnel_alive(cand_llm, timeout=4.0, kind="llm"):
                return self._open_resolved_service_url(
                    cand_llm,
                    ep=use_ep,
                    sname=sname,
                    stage="normal",
                    detected_kind="llm",
                    already_fetched_meta=True,
                )

            if is_orphan:
                return {"ok": False, "message": "该孤儿实例未检测到可用 LLM 对话隧道"}
            if not CTL_SCRIPT.exists():
                return {"ok": False, "message": f"未找到控制脚本: {CTL_SCRIPT.name}"}

            proc = _run_hidden(
                [_hidden_python_exe(), str(CTL_SCRIPT), "tunnel", "--restart", "-s", sname],
                timeout=60,
                extra_env=self._build_ctl_env(),
            )
            output = f"{proc.stdout}\n{proc.stderr}"
            m_llm = _DEPLOY_LLM_TUNNEL_LINE_RE.search(output)
            if not m_llm and fast_kind == "llm":
                m_llm = re.search(
                    r"(?:SERVICE_TUNNEL_URL:\s*|\[LLM WebUI / API 地址\]\s*)(https://(?!api\.)(?:[a-z0-9-]+\.)+[a-z]{2,}(?::\d+)?)",
                    output,
                )
            if proc.returncode == 0 and m_llm:
                url = m_llm.group(1).strip()
                if use_ep:
                    with self._lock:
                        cur_m = self._service_meta_by_ep.setdefault(use_ep, {"kind": fast_kind or "dual", "port": 8188})
                        cur_m["llm_tunnel_url"] = url
                return self._open_resolved_service_url(
                    url,
                    ep=use_ep,
                    sname=sname,
                    stage="restarted",
                    detected_kind="llm",
                    already_fetched_meta=bool(meta_fast),
                )

            tail = (output.strip().splitlines() or ["未获取到 LLM 对话隧道"])[-1]
            return {"ok": False, "message": f"暂未获取到在线 Qwen3.8 对话隧道: {tail}"}
        except Exception as exc:
            return {"ok": False, "message": f"打开 LLM 对话页异常: {exc}"}
        finally:
            self._action_lock.release()

    def get_llm_api_bundle(self, session: str = "comfy") -> dict[str, Any]:
        """组装当前在线 Qwen3.8 实例（单模或双栖）的 OpenAI 兼容接口配置包（不持久化密钥）。"""
        self.touch_activity()
        if self.is_deploying():
            return {"ok": False, "message": "正在执行 7 步开机部署，请等待就绪后再复制配置"}

        target = self._pick_target_session(session)
        if target is None and not self._has_running_session():
            snap = self.fetch_live_status()
            target = self._pick_target_session(session)
            if target is None and snap.get("assignments_count", 0) == 0 and not snap.get("sessions"):
                return {"ok": False, "message": "当前无运行中的 Qwen3.8 实例，请先部署开机"}

        ep = str(target.get("endpoint") or "") if target else ""
        sname = target["name"] if target and target.get("name") not in (None, "?") else session

        turl = self.resolve_active_llm_tunnel_url(session=sname, allow_fast_discover=True)
        fetched_now = False
        if not turl:
            tstatus, resolved_ep, fast_url = self._fast_read_tunnel_url(ep, sname)
            fetched_now = True
            if tstatus == "no_session":
                return {"ok": False, "message": "当前无运行中的 Qwen3.8 实例，请先部署开机"}
            if resolved_ep:
                ep = resolved_ep
            meta_chk, _ = self._get_ep_meta_and_key(ep)
            kind_chk = str(meta_chk.get("kind") or "").lower()
            llm_cand = str(
                meta_chk.get("llm_tunnel_url") or (fast_url if kind_chk == "llm" else "") or ""
            ).strip()
            if llm_cand and _probe_tunnel_alive(llm_cand, timeout=3.5, kind="llm"):
                turl = llm_cand
                if ep and kind_chk == "llm":
                    with self._lock:
                        self.state.setdefault("tunnel_cache", {})[ep] = llm_cand
                        self._save_state()

        meta, key = self._get_ep_meta_and_key(ep)
        if (not meta or not key) and not fetched_now:
            _, resolved_ep, fast_url = self._fast_read_tunnel_url(ep, sname)
            if resolved_ep:
                ep = resolved_ep
            meta, key = self._get_ep_meta_and_key(ep)
            kind_chk = str(meta.get("kind") or "").lower()
            llm_cand = str(
                meta.get("llm_tunnel_url") or (fast_url if kind_chk == "llm" else "") or ""
            ).strip()
            if not turl and llm_cand and _probe_tunnel_alive(llm_cand, timeout=3.5, kind="llm"):
                turl = llm_cand

        if meta and str(meta.get("kind") or "").lower() == "comfy":
            with self._comfy_lock:
                cur_sk = str(self._comfy_state.get("service_kind") or "comfy")
            if cur_sk not in ("llm", "dual") and not self.is_dual_mode_active():
                return {"ok": False, "message": "当前运行的是 ComfyUI 实例，非 LLM 对话服务"}

        if not turl:
            with self._comfy_lock:
                cur_sk2 = str(self._comfy_state.get("service_kind") or "")
                if cur_sk2 in ("llm", "dual"):
                    turl = str(
                        self._comfy_state.get("llm_tunnel_url")
                        or (self._comfy_state.get("tunnel_url") if cur_sk2 == "llm" else "")
                        or ""
                    )

        if not turl:
            return {"ok": False, "message": "暂未获取到在线 Qwen3.8 隧道，请稍候或先点打开对话页"}

        clean_turl = turl.strip().rstrip("/")
        base_url = f"{clean_turl}/v1"
        model = self._infer_default_llm_model_name(meta)
        clipboard_text = f"Base URL: {base_url}\nAPI Key: {key}\nModel: {model}"
        return {
            "ok": True,
            "base_url": base_url,
            "model": model,
            "has_key": bool(key),
            "clipboard_text": clipboard_text,
            "message": "📋 已复制 OpenAI 接口配置 (Base URL + Key)",
        }

    copy_llm_api_config = get_llm_api_bundle

    def open_colab_url(self, session: str = "comfy") -> dict[str, Any]:
        """在浏览器中打开 Colab 运行时关联网页或 Colab 首页。"""
        self.touch_activity()
        target = self._pick_target_session(session)
        if target and target.get("endpoint"):
            ep = target["endpoint"]
            url = (
                "https://colab.research.google.com/notebooks/empty.ipynb"
                f"?dbu=%2Ftun%2Fm%2F{ep}"
                f"#datalabBackendUrl=https://colab.research.google.com/tun/m/{ep}"
            )
            webbrowser.open(url)
            label = target.get("name") or ep
            return {"ok": True, "url": url, "message": f"已打开 Colab 会话 ({label})"}

        colab_exe = self._get_colab_exe()
        if colab_exe and colab_exe.is_file():
            try:
                proc = _run_hidden([str(colab_exe), "url", "-s", session], timeout=12)
                m = re.search(r"https://colab\.research\.google\.com\S+", proc.stdout)
                if proc.returncode == 0 and m:
                    webbrowser.open(m.group(0))
                    return {"ok": True, "url": m.group(0), "message": "已打开 Colab 会话页"}
            except Exception:
                pass

        home_url = "https://colab.research.google.com/"
        webbrowser.open(home_url)
        return {"ok": True, "url": home_url, "message": "当前无活跃会话，已打开 Colab 主页"}

    def sync_outputs(self, session: str = "comfy") -> dict[str, Any]:
        """将云端 ComfyUI 成图拉回本地目录；若当前活跃为 LLM 对话服务则平滑分流为返回 OpenAI 接口配置。"""
        self.touch_activity()
        if self.is_deploying():
            return {"ok": False, "message": "正在执行 7 步开机部署，请等待就绪后再操作"}
        if self.is_llm_mode_active():
            return self.get_llm_api_bundle(session=session)
        if not self._action_lock.acquire(blocking=False):
            return {"ok": False, "message": "另一项操作正在执行，请稍等片刻"}
        try:
            target = self._pick_target_session(session)
            if target is None:
                snap = self.fetch_live_status()
                target = self._pick_target_session(session)
                if (
                    target is None
                    and snap.get("assignments_count", 0) == 0
                    and not snap.get("sessions")
                ):
                    self.download_root.mkdir(parents=True, exist_ok=True)
                    try:
                        os.startfile(str(self.download_root))
                    except Exception:
                        pass
                    return {"ok": True, "message": "当前未开机，已打开本地成图目录"}

            # 二次核查远端是否实际运行着 LLM 服务（兼容刚启动桌宠且尚未跑过首轮探针的场景）
            ep_chk = str(target.get("endpoint") or "") if target else ""
            sname_chk = target["name"] if target and target.get("name") not in (None, "?") else session
            meta_chk, _ = self._get_ep_meta_and_key(ep_chk)
            if not meta_chk and ep_chk:
                self._fast_read_tunnel_url(ep_chk, sname_chk)
                meta_chk, _ = self._get_ep_meta_and_key(ep_chk)
            if meta_chk and str(meta_chk.get("kind") or "").lower() == "llm":
                return self.get_llm_api_bundle(session=sname_chk)

            if not CTL_SCRIPT.exists():
                return {"ok": False, "message": f"未找到控制脚本: {CTL_SCRIPT.name}"}
            sname = sname_chk
            proc = _run_hidden(
                [_hidden_python_exe(), str(CTL_SCRIPT), "sync", "--pull", "-s", sname],
                timeout=180,
                extra_env=self._build_ctl_env(),
            )
            output = f"{proc.stdout}\n{proc.stderr}".strip()
            tail = (output.splitlines() or ["同步结束"])[-1]
            if proc.returncode == 0:
                if self.download_root.exists():
                    try:
                        os.startfile(str(self.download_root))  # Windows 直接打开本地收图文件夹
                    except Exception:
                        pass
                return {"ok": True, "message": f"拉回完成: {tail}"}
            return {"ok": False, "message": f"同步失败: {tail}"}
        except Exception as exc:
            return {"ok": False, "message": f"同步异常: {exc}"}
        finally:
            self._action_lock.release()

    def stop_all_instances(self) -> dict[str, Any]:
        """一键停止全部 Colab 活跃实例与孤儿运行时并立即刷新余额。"""
        if self.is_deploying():
            self.cancel_deploy()
            time.sleep(0.4)
        if not self._action_lock.acquire(blocking=False):
            return {"ok": False, "message": "另一项操作正在执行，请稍等片刻"}
        try:
            if not CTL_SCRIPT.exists():
                return {"ok": False, "message": f"未找到控制脚本: {CTL_SCRIPT.name}"}
            proc = _run_hidden(
                [_hidden_python_exe(), str(CTL_SCRIPT), "down", "--all"],
                timeout=60,
                extra_env=self._build_ctl_env(),
            )
            output = f"{proc.stdout}\n{proc.stderr}".strip()
            if proc.returncode == 0:
                with self._lock:
                    self._service_meta_by_ep.clear()
                    self._runtime_secrets.clear()
                    self.state["tunnel_cache"] = {}
                    self._save_state()
                self._reset_comfy_offline()
            snap = self.fetch_live_status()
            if proc.returncode == 0:
                return {"ok": True, "message": "已关闭全部 Colab 实例", "snapshot": snap}
            tail = (output.splitlines() or ["停机指令异常"])[-1]
            return {"ok": False, "message": f"停机失败: {tail}", "snapshot": snap}
        except Exception as exc:
            return {"ok": False, "message": f"停机异常: {exc}"}
        finally:
            self._action_lock.release()

    # =========================================================================
    # 功能 2.5：面板直接选卡一键开机、实时算力紧缺检测与 7 步进度直播
    # =========================================================================

    @staticmethod
    def _format_elapsed_cn(seconds: int) -> str:
        """将秒数格式化为简短可读的中文耗时字符串。"""
        sec = max(0, int(seconds))
        if sec < 60:
            return f"{sec}秒"
        return f"{sec // 60}分{sec % 60:02d}秒"

    @staticmethod
    def _kill_process_tree(proc: Any) -> None:
        """尽力终止部署子进程及其 Windows 派生树（兼容测试替身对象）。"""
        if proc is None:
            return
        pid = getattr(proc, "pid", None)
        if os.name == "nt" and isinstance(pid, int) and pid > 0:
            try:
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(pid)],
                    capture_output=True,
                    creationflags=CREATE_NO_WINDOW,
                    timeout=8,
                )
            except Exception:
                pass
        for method_name in ("terminate", "kill"):
            fn = getattr(proc, method_name, None)
            if callable(fn):
                try:
                    fn()
                except Exception:
                    pass

    @staticmethod
    def _recommend_fallback_gpu(
        failed_gpu: str,
        profile: str,
        eligible_gpus: list[str] | None,
        scarce_gpus: dict[str, Any],
        profiles: list[str] | None = None,
    ) -> str | None:
        """在指定显卡缺卡或受限时，按生态偏好筛选首个当前可用替代 GPU。"""
        failed_up = str(failed_gpu or "").upper()
        profs = _normalize_boot_profiles(profiles, fallback_profile=profile)
        has_comfy = any(p in COMFY_BOOT_PROFILES for p in profs)
        if "qwen_llm" in profs or ("qwen_27b" in profs and has_comfy):
            # 180B 稀疏模型与 Comfy+27B 双栖堆叠均需高显存卡，备选池限制在 G4/A100/H100
            order: tuple[str, ...] = ("G4", "A100", "H100")
        elif profs == ["qwen_27b"]:
            order = ("L4", "A100", "G4", "H100", "T4")
        elif any(p in ("minimax", "drama3") for p in profs):
            order = ("G4", "A100", "H100", "L4", "T4")
        else:
            order = ("L4", "A100", "T4", "G4", "H100")

        allowed: set[str] | None = None
        if isinstance(eligible_gpus, list) and len(eligible_gpus) > 0:
            allowed = {str(g).upper() for g in eligible_gpus if g}

        scarce_set = {str(k).upper() for k in (scarce_gpus or {}).keys()}
        for cand in order:
            if cand == failed_up or cand in scarce_set:
                continue
            if allowed is not None and cand not in allowed:
                continue
            return cand
        return None

    @staticmethod
    def _classify_deploy_failure(
        lines: list[str],
        gpu: str,
        pre_assignments: int,
        profile: str,
        recreate: bool,
    ) -> dict[str, Any]:
        """分析开机/自举失败末尾日志，精准区分高峰期缺卡、配额受限与环境异常。"""
        cleaned = [ln.strip() for ln in lines if ln and ln.strip()]
        joined = "\n".join(cleaned)
        joined_low = joined.lower()
        gpu_up = str(gpu or "").upper()

        if "backend rejected accelerator" in joined_low or "not have quota or entitlement" in joined_low:
            return {
                "kind": "entitlement",
                "stockout_gpu": gpu_up,
                "reason": f"当前账号配额暂不支持 {gpu_up}",
                "message": f"⛔ 当前 Colab 账号无 {gpu_up} 配额或等级权限",
            }

        is_precondition = (
            "allocation refused" in joined_low
            or "toomanyassignments" in joined_low
            or ("412" in joined_low and "precondition" in joined_low)
        )
        is_capacity = (
            "resourceexhausted" in joined_low
            or "out of capacity" in joined_low
            or "no backend" in joined_low
            or "cannot connect to gpu backend" in joined_low
            or "failed to assign" in joined_low
            or ("503" in joined_low and "assign" in joined_low)
            or ("readtimeout" in joined_low and ("assign" in joined_low or "creating session" in joined_low))
            or ("timeout" in joined_low and "creating session" in joined_low)
        )
        if is_precondition:
            if pre_assignments > 0 and not recreate:
                return {
                    "kind": "limit_or_stockout",
                    "stockout_gpu": None,
                    "reason": "已有运行中实例或目标卡高峰期名额紧张",
                    "message": f"⚠️ 分配 {gpu_up} 被拒：并发实例上限或算力紧张，建议先停卡或换卡",
                }
            return {
                "kind": "stockout",
                "stockout_gpu": gpu_up,
                "reason": f"高峰期 {gpu_up} 算力容量不足",
                "message": f"⚠️ 高峰期 {gpu_up} 算力紧缺无法分配",
            }

        if is_capacity:
            return {
                "kind": "stockout",
                "stockout_gpu": gpu_up,
                "reason": f"高峰期 {gpu_up} 分配拥堵/无空闲算力",
                "message": f"⚠️ 高峰期 {gpu_up} 无空闲卡或响应超时",
            }

        for ln in reversed(cleaned):
            if "磁盘空间不足" in ln or "显存不足" in ln:
                return {
                    "kind": "error",
                    "stockout_gpu": None,
                    "reason": ln,
                    "message": f"⚠️ {ln[:60]}",
                }
            if "llama-server" in ln and ("超时" in ln or "失败" in ln or "异常退出" in ln):
                return {
                    "kind": "error",
                    "stockout_gpu": None,
                    "reason": "llama-server 启动异常",
                    "message": f"⚠️ {ln[:58]}",
                }
            if "等待 ComfyUI 就绪超时" in ln:
                return {
                    "kind": "error",
                    "stockout_gpu": None,
                    "reason": "ComfyUI 启动超时",
                    "message": "⚠️ 等待 ComfyUI 启动超时，可尝试重新增量部署",
                }
            if "WebSocket" in ln or "握手超时" in ln:
                return {
                    "kind": "error",
                    "stockout_gpu": None,
                    "reason": "Colab 内核通信超时",
                    "message": "⚠️ 远端内核握手超时，可重试或停卡重开",
                }

        tail = "部署中断"
        for ln in reversed(cleaned):
            low = ln.lower()
            if ln.startswith('File "') or ln.startswith("Traceback ") or low == "raise" or low.startswith("raise "):
                continue
            if set(ln) <= {"~", "^", " "} or "output=stdout, stderr=stderr)" in low:
                continue
            if "calledprocesserror" in low and len(cleaned) > 1:
                continue
            tail = ln
            break
        if len(tail) > 52:
            tail = tail[:49] + "..."
        return {
            "kind": "error",
            "stockout_gpu": None,
            "reason": tail,
            "message": f"⚠️ {profile} 部署异常: {tail}",
        }

    @staticmethod
    def _calc_phase_dl_total(
        phase_prof: str,
        llm_quant: str = "UD-Q4_K_XL",
        llm_mmproj: bool = True,
        llm27_mmproj: bool = True,
        minimax_r2v: bool = True,
        minimax_pe: bool = True,
    ) -> int:
        """估算单个子阶段在步骤 4 需要校验/拉取的主要模型分片数量。"""
        if phase_prof == "qwen_27b":
            return LLM_27B_BASE_SHARDS + (1 if llm27_mmproj else 0)
        if phase_prof == "qwen_llm":
            return LLM_QUANT_BASE_SHARDS.get(llm_quant, 4) + (1 if llm_mmproj else 0)
        if phase_prof == "minimax":
            return 7 + (1 if minimax_pe else 0) + (2 if minimax_r2v else 0)
        return 3

    def _scale_phase_pct(self, local_pct: int) -> int:
        """将当前子阶段的局部百分比 [12..100] 平滑映射到多阶段总进度条。"""
        tot = max(1, int(self._deploy_phase_total))
        if tot <= 1:
            return max(0, min(100, int(local_pct)))
        idx = max(1, min(tot, int(self._deploy_phase_idx)))
        lo = 12.0 + (idx - 1) * (86.0 / tot)
        hi = 98.0 if idx >= tot else (12.0 + idx * (86.0 / tot))
        frac = max(0.0, min(1.0, (float(local_pct) - 12.0) / 88.0))
        return int(round(lo + frac * (hi - lo)))

    def _consume_deploy_line(self, raw_line: str) -> str | None:
        """解析一行开机/部署输出并增量推进 _deploy_state（兼容单/多阶段），返回最终隧道 URL。"""
        line = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", str(raw_line or "")).strip()
        if not line:
            return None

        tunnel_url: str | None = None
        should_warm_meta = False
        with self._lock:
            st = self._deploy_state
            cur_pct = int(st.get("pct") or 0)
            cur_step = int(st.get("step_idx") or 0)
            total_steps = max(1, int(st.get("step_total") or 7))
            gpu = str(st.get("gpu") or "L4")
            is_llm = bool(self._deploy_phase_is_llm)

            m_phase = _DEPLOY_PHASE_RE.search(line)
            if m_phase:
                p_idx = max(1, int(m_phase.group(1)))
                p_tot = max(1, int(m_phase.group(2)))
                p_prof = m_phase.group(3).strip().lower()
                self._deploy_phase_idx = p_idx
                self._deploy_phase_total = p_tot
                self._deploy_phase_prof = p_prof
                self._deploy_phase_is_llm = p_prof in LLM_BOOT_PROFILES
                self._deploy_dl_done = 0
                self._deploy_dl_name = ""
                prefs = self.state.get("prefs") or {}
                self._deploy_dl_total = self._calc_phase_dl_total(
                    p_prof,
                    str(prefs.get("boot_llm_quant") or "UD-Q4_K_XL"),
                    bool(prefs.get("boot_llm_mmproj", True)),
                    bool(prefs.get("boot_llm27_mmproj", True)),
                    bool(prefs.get("boot_minimax_r2v", True)),
                    bool(prefs.get("boot_minimax_pe", True)),
                )
                st["step_idx"] = 0
                st["step_total"] = 7
                p_name = {
                    "qwen_27b": "Qwen27B",
                    "qwen_llm": "Qwen180B",
                    "minimax": "MiniMax",
                    "drama3": "Drama3",
                }.get(p_prof, "Anima")
                if p_tot > 1:
                    st["stage_title"] = f"阶段 {p_idx}/{p_tot} · {p_name}"
                    st["sub_detail"] = f"正在启动 {p_name} 云端部署..."
                    st["pct"] = max(cur_pct, self._scale_phase_pct(14))
                return None

            m_ltun = _DEPLOY_LLM_TUNNEL_LINE_RE.search(line)
            if m_ltun:
                self._deploy_captured_llm_tunnel = m_ltun.group(1).strip()
                return None

            m_tot = _DEPLOY_DL_TOTAL_RE.search(line)
            if m_tot:
                tot_files = int(m_tot.group(1))
                if tot_files > 0:
                    self._deploy_dl_total = tot_files
                return None

            m_tun = _DEPLOY_TUNNEL_LINE_RE.search(line)
            if m_tun:
                cand_url = m_tun.group(1).strip()
                if self._deploy_phase_idx < self._deploy_phase_total:
                    if is_llm and not self._deploy_captured_llm_tunnel:
                        self._deploy_captured_llm_tunnel = cand_url
                    st["step_idx"] = total_steps
                    st["pct"] = max(cur_pct, self._scale_phase_pct(100))
                    st["stage_title"] = f"阶段 {self._deploy_phase_idx}/{self._deploy_phase_total} 就绪"
                    st["sub_detail"] = "前一服务就绪，继续部署下一工作流..."
                    return None
                tunnel_url = cand_url
                st["step_idx"] = total_steps
                st["pct"] = 100
                st["stage_title"] = f"[{total_steps}/{total_steps}] 部署全部完成"
                st["sub_detail"] = f"隧道已打通: {tunnel_url.replace('https://', '')}"
                if not self._deploy_meta_warmed:
                    self._deploy_meta_warmed = True
                    should_warm_meta = True

        if tunnel_url is not None:
            if should_warm_meta:
                threading.Thread(
                    target=lambda: self._fast_read_tunnel_url("", "comfy"),
                    daemon=True,
                ).start()
            return tunnel_url

        with self._lock:
            st = self._deploy_state
            cur_pct = int(st.get("pct") or 0)
            cur_step = int(st.get("step_idx") or 0)
            gpu = str(st.get("gpu") or "L4")
            is_llm = bool(self._deploy_phase_is_llm)
            is_27b = self._deploy_phase_prof == "qwen_27b"

            m_step = _STEP_BANNER_RE.search(line)
            if m_step:
                idx = int(m_step.group(1))
                tot = max(1, int(m_step.group(2)))
                raw_title = m_step.group(3).strip()
                max_title_len = 14 if self._deploy_phase_total > 1 else 18
                if len(raw_title) > max_title_len:
                    raw_title = raw_title[: max_title_len - 1] + "…"
                base_pct = _DEPLOY_STEP_BASE_PCT.get(
                    idx, min(95, int(round(12 + idx * 84.0 / tot)))
                )
                phase_tag = (
                    f"P{self._deploy_phase_idx}/{self._deploy_phase_total} "
                    if self._deploy_phase_total > 1
                    else ""
                )
                st["step_idx"] = idx
                st["step_total"] = tot
                st["stage_title"] = f"{phase_tag}[{idx}/{tot}] {raw_title}"
                st["pct"] = max(cur_pct, self._scale_phase_pct(base_pct))
                if is_llm:
                    if idx == 3:
                        st["sub_detail"] = "正在准备 llama-server CUDA 推理引擎..."
                    elif idx == 4:
                        st["sub_detail"] = "正在高速拉取 Qwen3.8 GGUF 量化分片..."
                    elif idx == 5:
                        st["sub_detail"] = (
                            "正在生成访问密钥与全卡推理参数..."
                            if is_27b
                            else "正在生成访问密钥与 PLE/MoE 卸载参数..."
                        )
                    elif idx == 6:
                        st["sub_detail"] = (
                            "正在加载 27B 模型并启动 llama-server..."
                            if is_27b
                            else "正在加载 180B 模型并启动 llama-server..."
                        )
                    elif idx == 7:
                        st["sub_detail"] = "正在打通 Cloudflared 隧道并写入服务契约..."
                    else:
                        st["sub_detail"] = raw_title
                else:
                    if idx == 4:
                        st["sub_detail"] = "正在连接模型源并校验本地缓存..."
                    elif idx == 5:
                        st["sub_detail"] = "正在拉取 LoRA 与工作流配置..."
                    elif idx == 6:
                        st["sub_detail"] = "正在拉起 ComfyUI 服务..."
                    elif idx == 7:
                        st["sub_detail"] = "正在申请 Cloudflared 公网隧道..."
                    else:
                        st["sub_detail"] = raw_title
                return None

            low = line.lower()
            if "正在释放旧实例" in line:
                st["pct"] = max(cur_pct, 3)
                st["stage_title"] = "清理旧会话"
                st["sub_detail"] = "正在停卡释放旧算力分配..."
            elif "creating session" in low or "开启 colab 实例" in line:
                st["pct"] = max(cur_pct, 6)
                st["stage_title"] = f"申请 {gpu} 实例"
                st["sub_detail"] = f"正在向 Colab 请求分配 {gpu} GPU..."
            elif "session ready" in low:
                st["pct"] = max(cur_pct, 11)
                st["stage_title"] = "实例就绪"
                st["sub_detail"] = f"{gpu} 已分配，准备上传云盘凭据..."
            elif "跳过 colab new" in line:
                st["pct"] = max(cur_pct, 11)
                st["stage_title"] = "复用活跃实例"
                st["sub_detail"] = "沿用现存运行时，上传配置并增量部署..."
            elif "colab.exe upload" in low or ("upload" in low and "rclone.conf" in low):
                st["pct"] = max(cur_pct, 13)
                st["stage_title"] = "注入云盘配置"
                st["sub_detail"] = "正在握手内核并启动自举..."
            elif cur_step == 1:
                if "自动降级" in line or "降级为" in line:
                    st["sub_detail"] = line[:38]
                    st["pct"] = max(cur_pct, self._scale_phase_pct(20))
                elif "gpu 硬件信息:" in line or line.startswith("GPU:"):
                    info = line.split(":", 1)[1].strip()
                    st["sub_detail"] = f"已就绪: {info[:36]}"
                    st["pct"] = max(cur_pct, self._scale_phase_pct(19))
            elif cur_step == 2 and ("已就绪" in line or "安装完成" in line or "工具就绪" in line):
                st["pct"] = max(cur_pct, self._scale_phase_pct(27))
                st["sub_detail"] = "传输与隧道基础工具已备齐"
            elif cur_step == 3:
                if is_llm:
                    if "cmake" in low or "编译" in line:
                        st["sub_detail"] = "正在编译 llama-server CUDA 内核..."
                        st["pct"] = max(cur_pct, self._scale_phase_pct(36))
                    elif "cuda0" in low or ("llama-server" in low and ("就绪" in line or "完成" in line)):
                        st["sub_detail"] = "llama-server CUDA 推理引擎已就绪"
                        st["pct"] = max(cur_pct, self._scale_phase_pct(40))
                    elif "预编译" in line or "下载" in line or "拉取" in line:
                        st["sub_detail"] = "正在拉取 llama-server CUDA 二进制包..."
                        st["pct"] = max(cur_pct, self._scale_phase_pct(34))
                else:
                    if "git" in low and "clone" in low:
                        st["sub_detail"] = "正在克隆官方 ComfyUI 仓库..."
                        st["pct"] = max(cur_pct, self._scale_phase_pct(33))
                    elif "pip" in low and "install" in low:
                        st["sub_detail"] = "正在安装依赖（保留预装 CUDA）..."
                        st["pct"] = max(cur_pct, self._scale_phase_pct(36))
                    elif "依赖安装完成" in line or "comfy_kitchen 已就绪" in line:
                        st["sub_detail"] = "ComfyUI 依赖与 CUDA 环境就绪"
                        st["pct"] = max(cur_pct, self._scale_phase_pct(40))
            elif cur_step == 4:
                m_start = re.search(r"(?:开始下载底座文件:|\[开始下载\])\s*([^\s(]+)", line)
                if m_start:
                    fname = PurePosixPath(m_start.group(1)).name
                    self._deploy_dl_name = fname[:20] + ("…" if len(fname) > 20 else "")
                    st["sub_detail"] = f"下载 {self._deploy_dl_name}..."
                    return None

                if (
                    "已存在且字节数匹配" in line
                    or "校验通过:" in line
                    or "[已就绪]" in line
                    or "[完成]" in line
                ):
                    self._deploy_dl_done = min(
                        max(1, self._deploy_dl_total),
                        self._deploy_dl_done + 1,
                    )
                    ratio = self._deploy_dl_done / max(1, self._deploy_dl_total)
                    local_p = min(76, 42 + int(round(ratio * 34)))
                    st["pct"] = max(cur_pct, self._scale_phase_pct(local_p))
                    short_msg = line.replace("已存在且字节数匹配，跳过:", "已缓存")
                    st["sub_detail"] = short_msg[:38]
                    return None

                m_pct = _PCT_TOKEN_RE.search(line)
                if m_pct:
                    file_pct = min(100.0, max(0.0, float(m_pct.group(1))))
                    ratio = min(
                        1.0,
                        (self._deploy_dl_done + file_pct / 100.0) / max(1, self._deploy_dl_total),
                    )
                    local_p = min(76, 42 + int(round(ratio * 34)))
                    st["pct"] = max(cur_pct, self._scale_phase_pct(local_p))
                    m_spd = _DL_SPEED_RE.search(line)
                    spd = f" · {m_spd.group(1)}" if m_spd else ""
                    prefix = f"{self._deploy_dl_name} " if self._deploy_dl_name else "拉取模型 "
                    st["sub_detail"] = f"{prefix}({int(round(file_pct))}%{spd})"
            elif cur_step == 5:
                if is_llm and ("sk-pet-" in line or "鉴权" in line or "参数" in line or "卸载" in line):
                    st["sub_detail"] = "安全鉴权密钥与推理参数已就绪"
                    st["pct"] = max(cur_pct, self._scale_phase_pct(83))
                elif "LoRA 就绪:" in line or "工作流就绪:" in line:
                    st["sub_detail"] = line[:38]
                    st["pct"] = max(cur_pct, self._scale_phase_pct(83))
            elif cur_step == 6:
                if is_llm:
                    if "已就绪" in line or "模型就绪" in line or "复用" in line:
                        st["sub_detail"] = "Qwen3.8 推理引擎加载完成"
                        st["pct"] = max(cur_pct, self._scale_phase_pct(93))
                    elif "llama-server" in low or "18080" in line or "8080" in line or "health" in low or "加载" in line:
                        st["sub_detail"] = "等待 llama-server 18080 端口加载就绪..."
                        st["pct"] = max(cur_pct, self._scale_phase_pct(89))
                else:
                    if "已后台拉起 ComfyUI" in line or "已拉起 ComfyUI" in line:
                        st["sub_detail"] = "等待 ComfyUI 8188 端口响应..."
                        st["pct"] = max(cur_pct, self._scale_phase_pct(89))
                    elif "ComfyUI 就绪" in line or "已就绪" in line:
                        st["sub_detail"] = "ComfyUI 引擎启动完成"
                        st["pct"] = max(cur_pct, self._scale_phase_pct(93))
            elif cur_step == 7 and "cloudflared" in low and ("拉起" in line or "启动" in line or "隧道" in line):
                st["sub_detail"] = "等待 Cloudflare 分配公网域名..."
                st["pct"] = max(cur_pct, self._scale_phase_pct(97))
        return tunnel_url

    def start_instance(self, config: dict[str, Any] | None = None) -> dict[str, Any]:
        """按用户所选工作流集合（支持多选双栖）与显卡一键无窗口部署并直播进度。"""
        self.touch_activity()
        cfg = config if isinstance(config, dict) else {}
        profiles = _normalize_boot_profiles(
            cfg.get("profiles"), fallback_profile=str(cfg.get("profile") or "anima")
        )
        profile = profiles[0]
        llm_prof = next((p for p in profiles if p in LLM_BOOT_PROFILES), None)
        comfy_profs = [p for p in profiles if p in COMFY_BOOT_PROFILES]
        is_dual = bool(llm_prof and comfy_profs)
        exec_order = ([llm_prof] if llm_prof else []) + comfy_profs

        needs_big_gpu = (
            "qwen_llm" in profiles
            or "minimax" in profiles
            or "drama3" in profiles
            or len(profiles) >= 2
        )
        default_gpu = "G4" if needs_big_gpu else "L4"
        gpu = str(cfg.get("gpu") or default_gpu).strip().upper()
        if gpu not in SUPPORTED_BOOT_GPUS:
            gpu = default_gpu

        # L4 与 G4 标准形态已自带充足内存且 Colab 无独立 HIGH_RAM 选项，强制关闭 high_mem 防分配被拒
        high_mem = False if gpu in ("L4", "G4") else bool(cfg.get("high_mem", False))
        minimax_r2v = bool(cfg.get("minimax_r2v", True))
        minimax_pe = bool(cfg.get("minimax_pe", True))
        recreate = bool(cfg.get("recreate", False))

        if not CTL_SCRIPT.exists():
            return {"ok": False, "message": f"未找到控制脚本: {CTL_SCRIPT.name}"}

        # 开机前前置环境自检：缺工具、确认未登录或未绑云盘时精准引导到第 5 页对应标签页
        ob = self._get_onboarding_state()
        if not ob["colab_cli_ok"] or not ob["rclone_exe_ok"]:
            return {
                "ok": False,
                "needs_onboarding": True,
                "target_tab": "env",
                "message": "缺少 colab-cli 或 rclone 组件，请先在第 5 页「云盘与环境」完成接入",
            }
        if ob["auth_missing"]:
            return {
                "ok": False,
                "needs_onboarding": True,
                "target_tab": "vault",
                "message": "尚未绑定 Google Colab 账号，请先在第 5 页「多账号」登录绑定",
            }
        if not ob["gdrive_ok"]:
            return {
                "ok": False,
                "needs_onboarding": True,
                "target_tab": "env",
                "message": "尚未绑定 Google Drive 云盘 (gdrive_save)，请先在第 5 页完成授权",
            }

        if not self._deploy_lock.acquire(blocking=False):
            return {"ok": False, "message": "已有开机部署任务在运行中"}

        # 保存本次选卡与工作流配置，并记录初始活跃会话数辅助精准区分缺卡与并发上限
        with self._lock:
            prefs = self.state.setdefault("prefs", dict(DEFAULT_PREFS))
            raw_quant = str(cfg.get("llm_quant") or prefs.get("boot_llm_quant") or "UD-Q4_K_XL").strip()
            llm_quant = raw_quant if raw_quant in SUPPORTED_LLM_QUANTS else "UD-Q4_K_XL"
            try:
                llm_ctx = int(cfg.get("llm_ctx") or prefs.get("boot_llm_ctx") or 32768)
            except (TypeError, ValueError):
                llm_ctx = 32768
            llm_ctx = max(4096, min(262144, llm_ctx))
            llm_mmproj = bool(cfg.get("llm_mmproj", prefs.get("boot_llm_mmproj", True)))

            raw_q27 = str(cfg.get("llm27_quant") or prefs.get("boot_llm27_quant") or "UD-Q4_K_XL").strip()
            llm27_quant = raw_q27 if raw_q27 in SUPPORTED_LLM_QUANTS else "UD-Q4_K_XL"
            try:
                llm27_ctx = int(cfg.get("llm27_ctx") or prefs.get("boot_llm27_ctx") or 65536)
            except (TypeError, ValueError):
                llm27_ctx = 65536
            llm27_ctx = max(4096, min(262144, llm27_ctx))
            llm27_mmproj = bool(cfg.get("llm27_mmproj", prefs.get("boot_llm27_mmproj", True)))

            prefs["boot_profiles"] = list(profiles)
            prefs["boot_profile"] = profile
            prefs["boot_gpu"] = gpu
            prefs["boot_high_mem"] = high_mem
            prefs["boot_minimax_r2v"] = minimax_r2v
            prefs["boot_minimax_pe"] = minimax_pe
            prefs["boot_llm_quant"] = llm_quant
            prefs["boot_llm_ctx"] = llm_ctx
            prefs["boot_llm_mmproj"] = llm_mmproj
            prefs["boot_llm27_quant"] = llm27_quant
            prefs["boot_llm27_ctx"] = llm27_ctx
            prefs["boot_llm27_mmproj"] = llm27_mmproj
            self._save_state()
            self.last_snapshot["prefs"] = dict(prefs)

            pre_assignments = int(self.state.get("last_assignments_count") or 0)
            now_ts = int(time.time())
            self._deploy_cancelled = False
            self._deploy_meta_warmed = False
            self._deploy_dl_name = ""
            self._deploy_dl_done = 0
            first_phase = exec_order[0]
            self._deploy_phase_idx = 1
            self._deploy_phase_total = len(exec_order)
            self._deploy_phase_prof = first_phase
            self._deploy_phase_is_llm = first_phase in LLM_BOOT_PROFILES
            self._deploy_captured_llm_tunnel = None
            self._deploy_dl_total = self._calc_phase_dl_total(
                first_phase,
                llm_quant=llm_quant,
                llm_mmproj=llm_mmproj,
                llm27_mmproj=llm27_mmproj,
                minimax_r2v=minimax_r2v,
                minimax_pe=minimax_pe,
            )

            self._deploy_state = {
                "active": True,
                "profile": profile,
                "profiles": list(profiles),
                "gpu": gpu,
                "high_mem": high_mem,
                "step_idx": 0,
                "step_total": 7,
                "stage_title": f"准备开机 ({gpu})",
                "sub_detail": "正在启动本地无窗口部署进程...",
                "pct": 2,
                "started_ts": now_ts,
                "elapsed_sec": 0,
                "error_msg": "",
                "stockout_gpu": None,
                "suggest_gpu": None,
            }
            initial_deploy = self.get_deploy_state()

        target_kind = "dual" if is_dual else ("llm" if llm_prof else "comfy")
        if target_kind == "llm":
            self._stop_ws()
        with self._comfy_lock:
            self._comfy_state["service_kind"] = target_kind
            self._comfy_state["active_profiles"] = list(profiles)
            cstate_init = dict(self._comfy_state)
        with self._lock:
            self.last_snapshot["comfy"] = cstate_init

        cmd = [
            _hidden_python_exe(),
            "-u",
            str(CTL_SCRIPT),
            "up",
            "-s",
            "comfy",
        ]
        if len(profiles) >= 2:
            cmd.extend(["--profiles", ",".join(profiles)])
        else:
            cmd.extend(["--profile", profile])
        cmd.extend(["--gpu", gpu])

        if high_mem:
            cmd.append("--high-mem")
        if recreate:
            cmd.append("--recreate")
        if "minimax" in profiles:
            if not minimax_r2v and not minimax_pe:
                cmd.append("--minimax-lite")
            else:
                if not minimax_r2v:
                    cmd.append("--no-r2v")
                if not minimax_pe:
                    cmd.append("--no-pe")
        if llm_prof == "qwen_27b":
            cmd.extend(["--llm-quant", llm27_quant, "--llm-ctx", str(llm27_ctx)])
            if not llm27_mmproj:
                cmd.append("--no-mmproj")
        elif llm_prof == "qwen_llm":
            cmd.extend(["--llm-quant", llm_quant, "--llm-ctx", str(llm_ctx)])
            if not llm_mmproj:
                cmd.append("--no-mmproj")

        name_map = {
            "anima": "Anima",
            "minimax": "MiniMax",
            "drama3": "Drama 3",
            "qwen_27b": "Qwen27B",
            "qwen_llm": "Qwen180B",
        }
        profile_label = " + ".join(name_map.get(p, p) for p in profiles)
        self._emit_event({"type": "deploy_progress", "deploy": initial_deploy})

        def _worker() -> None:
            hb_stop = threading.Event()
            tail_lines: list[str] = []
            captured_tunnel: str | None = None
            rc = 1

            def _heartbeat() -> None:
                while not hb_stop.wait(timeout=1.0):
                    if not self.is_deploying():
                        break
                    self._emit_event({"type": "deploy_progress", "deploy": self.get_deploy_state()})

            hb_thread = threading.Thread(target=_heartbeat, daemon=True)
            hb_thread.start()

            try:
                env = _clean_env(self._build_ctl_env())
                env["PYTHONUNBUFFERED"] = "1"
                # 加快高峰期 503/分配拒绝响应，避免用户干等数十秒
                env["COLAB_CLI_ASSIGN_RETRIES"] = "1"
                env["COLAB_CLI_RETRY_BACKOFF"] = "4"

                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    env=env,
                    creationflags=CREATE_NO_WINDOW,
                )
                self._deploy_proc = proc

                last_emit_m = 0.0
                if proc.stdout is not None:
                    buf = ""
                    try:
                        while True:
                            ch = proc.stdout.read(256)
                            if not ch:
                                break
                            buf += ch.replace("\r", "\n")
                            while "\n" in buf:
                                raw_line, buf = buf.split("\n", 1)
                                sline = raw_line.strip()
                                if not sline:
                                    continue
                                tail_lines = (tail_lines + [sline])[-40:]
                                prev_step = int(self._deploy_state.get("step_idx") or 0)
                                tun = self._consume_deploy_line(sline)
                                cur_step = int(self._deploy_state.get("step_idx") or 0)
                                if tun:
                                    captured_tunnel = tun
                                now_m = time.monotonic()
                                if tun or cur_step != prev_step or (now_m - last_emit_m >= 0.12):
                                    last_emit_m = now_m
                                    self._emit_event(
                                        {"type": "deploy_progress", "deploy": self.get_deploy_state()}
                                    )
                        if buf.strip():
                            sline = buf.strip()
                            tail_lines = (tail_lines + [sline])[-40:]
                            tun = self._consume_deploy_line(sline)
                            if tun:
                                captured_tunnel = tun
                    finally:
                        proc.stdout.close()

                rc = proc.wait()
            except Exception as exc:
                tail_lines.append(str(exc))
                rc = 1
            finally:
                hb_stop.set()
                self._deploy_proc = None

            cancelled = self._deploy_cancelled
            elapsed_total = max(0, int(time.time()) - int(initial_deploy.get("started_ts") or time.time()))
            elapsed_cn = self._format_elapsed_cn(elapsed_total)

            if cancelled:
                with self._lock:
                    self._deploy_state.update(
                        {
                            "active": False,
                            "pct": 0,
                            "stage_title": "已中止部署",
                            "sub_detail": "用户主动取消开机流程",
                            "error_msg": "已中止开机部署",
                            "elapsed_sec": elapsed_total,
                        }
                    )
                snap = self.fetch_live_status()
                self._emit_event(
                    {
                        "type": "deploy_error",
                        "cancelled": True,
                        "deploy": self.get_deploy_state(),
                        "snapshot": snap,
                        "summary": "⏹️ 已中止开机部署；若实例已分配可点一键停卡释放",
                    }
                )
            elif rc == 0:
                if is_dual:
                    done_detail = "ComfyUI + Qwen3.8 双栖服务与隧道已就绪"
                elif llm_prof:
                    done_detail = "Qwen3.8 引擎与隧道已就绪"
                else:
                    done_detail = "ComfyUI 与隧道已就绪"

                with self._lock:
                    self._scarce_gpus.pop(gpu, None)
                    self._deploy_state.update(
                        {
                            "active": False,
                            "step_idx": int(self._deploy_state.get("step_total") or 7),
                            "pct": 100,
                            "stage_title": "部署完成",
                            "sub_detail": done_detail,
                            "error_msg": "",
                            "stockout_gpu": None,
                            "suggest_gpu": None,
                            "elapsed_sec": elapsed_total,
                        }
                    )
                snap = self.fetch_live_status()
                target_ep: str | None = None
                with self._lock:
                    sessions = list(self.state.get("last_sessions") or [])
                    target_ep = next(
                        (str(s.get("endpoint")) for s in sessions if s.get("name") == "comfy" and s.get("endpoint")),
                        None,
                    )
                    if not target_ep and sessions:
                        target_ep = str(sessions[0].get("endpoint") or "")
                    if captured_tunnel and target_ep:
                        self.state.setdefault("tunnel_cache", {})[target_ep] = captured_tunnel
                        self._save_state()

                # 确保最新服务契约与内存态密钥已就位，使首次打开对话页或复制配置零延迟
                meta_ready, key_ready = self._get_ep_meta_and_key(target_ep or "")
                if llm_prof:
                    need_refresh = (not meta_ready) or (not key_ready) or (is_dual and meta_ready.get("kind") != "dual")
                    if need_refresh:
                        _, r_ep, fast_u = self._fast_read_tunnel_url(target_ep or "", "comfy")
                        if r_ep and not target_ep:
                            target_ep = r_ep
                        if fast_u and not captured_tunnel:
                            captured_tunnel = fast_u
                        meta_ready, _ = self._get_ep_meta_and_key(target_ep or "")
                elif target_ep:
                    with self._lock:
                        self._service_meta_by_ep[target_ep] = {"kind": "comfy", "port": 8188}
                        self._runtime_secrets.pop(target_ep, None)

                used_q = llm27_quant if llm_prof == "qwen_27b" else llm_quant
                used_ctx = llm27_ctx if llm_prof == "qwen_27b" else llm_ctx
                llm_url_final = (
                    self._deploy_captured_llm_tunnel
                    or (str(meta_ready.get("llm_tunnel_url") or "") if isinstance(meta_ready, dict) else "")
                    or (captured_tunnel if target_kind == "llm" else "")
                    or (self._read_cf_public_llm_url() if llm_prof else "")
                    or None
                )
                with self._comfy_lock:
                    self._comfy_state["service_kind"] = target_kind
                    self._comfy_state["active_profiles"] = list(profiles)
                    if captured_tunnel:
                        self._comfy_state["tunnel_url"] = captured_tunnel
                    self._comfy_state["comfy_online"] = target_kind in ("comfy", "dual")
                    self._comfy_state["llm_online"] = target_kind in ("llm", "dual")
                    self._comfy_state["llm_tunnel_url"] = llm_url_final if target_kind in ("llm", "dual") else None
                    if llm_prof:
                        if target_kind == "llm":
                            self._comfy_state["ws_connected"] = False
                        self._comfy_state["model_name"] = self._infer_default_llm_model_name(meta_ready)
                        self._comfy_state["quant"] = str(meta_ready.get("quant") or used_q)
                        self._comfy_state["ctx_size"] = int(meta_ready.get("ctx_size") or used_ctx)
                        if meta_ready.get("gpu_name"):
                            self._comfy_state["gpu_name"] = self._short_gpu_name(
                                str(meta_ready["gpu_name"])
                            )
                        v_u = round(float(meta_ready.get("vram_used_gb") or 0.0), 1)
                        v_t = round(float(meta_ready.get("vram_total_gb") or 0.0), 1)
                        if v_t > 0:
                            self._comfy_state["vram_used_gb"] = v_u
                            self._comfy_state["vram_total_gb"] = v_t
                            self._comfy_state["vram_pct"] = round(v_u * 100.0 / v_t, 1)
                    cstate = dict(self._comfy_state)
                with self._lock:
                    self.last_snapshot["comfy"] = cstate

                self._wake_monitor.set()
                if is_dual:
                    done_action = "open_comfy"
                    done_summary = f"🎉 {gpu}·{profile_label} 双栖就绪({elapsed_cn})！点我打开"
                elif llm_prof:
                    done_action = "open_llm_webui"
                    done_summary = f"🎉 {gpu}·{profile_label} 部署完成({elapsed_cn})！点我打开对话页"
                else:
                    done_action = "open_comfy"
                    done_summary = f"🎉 {gpu}·{profile_label} 部署完成({elapsed_cn})！点我打开"
                self._emit_event(
                    {
                        "type": "deploy_done",
                        "deploy": self.get_deploy_state(),
                        "snapshot": self.get_cached_snapshot(),
                        "tunnel_url": captured_tunnel,
                        "llm_tunnel_url": llm_url_final,
                        "summary": done_summary,
                        "action": done_action,
                    }
                )
            else:
                diag = self._classify_deploy_failure(tail_lines, gpu, pre_assignments, profile_label, recreate)
                stockout_gpu = diag.get("stockout_gpu")
                if stockout_gpu:
                    with self._lock:
                        self._scarce_gpus[str(stockout_gpu).upper()] = {
                            "ts": int(time.time()),
                            "reason": str(diag.get("reason") or "算力紧缺"),
                            "kind": str(diag.get("kind") or "stockout"),
                        }
                snap = self.fetch_live_status()
                suggest_gpu = None
                if stockout_gpu:
                    suggest_gpu = self._recommend_fallback_gpu(
                        str(stockout_gpu),
                        profile,
                        snap.get("eligible_gpus"),
                        snap.get("scarce_gpus") or {},
                        profiles=profiles,
                    )
                summary = str(diag.get("message") or "⚠️ 开机部署失败")
                if stockout_gpu and suggest_gpu and suggest_gpu not in summary:
                    summary = f"{summary}，建议改选 {suggest_gpu}"

                with self._lock:
                    self._deploy_state.update(
                        {
                            "active": False,
                            "stage_title": "部署失败",
                            "sub_detail": str(diag.get("reason") or summary)[:42],
                            "error_msg": summary,
                            "stockout_gpu": stockout_gpu,
                            "suggest_gpu": suggest_gpu,
                            "elapsed_sec": elapsed_total,
                        }
                    )
                    snap["deploy"] = self.get_deploy_state()
                    self.last_snapshot = snap

                self._emit_event(
                    {
                        "type": "deploy_error",
                        "deploy": self.get_deploy_state(),
                        "snapshot": snap,
                        "summary": summary,
                        "stockout_gpu": stockout_gpu,
                        "suggest_gpu": suggest_gpu,
                    }
                )

            self._deploy_lock.release()

        threading.Thread(target=_worker, daemon=True).start()
        return {
            "ok": True,
            "started": True,
            "deploy": initial_deploy,
            "message": f"开始部署 {gpu} · {profile_label}...",
        }

    def cancel_deploy(self) -> dict[str, Any]:
        """主动中止正在运行的一键开机部署子进程。"""
        proc = self._deploy_proc
        if not self.is_deploying() and proc is None:
            return {"ok": False, "message": "当前没有进行中的部署任务"}
        self._deploy_cancelled = True
        self._kill_process_tree(proc)
        return {"ok": True, "message": "正在中止开机部署..."}

    # =========================================================================
    # 功能 3：ComfyUI 实时队列 / 显存监控 / 自动或手动拉图 / 异常中断提醒
    # =========================================================================

    def acknowledge_comfy_error(self) -> None:
        """用户点击报错气泡前往处理后，清除当前红标报警。"""
        with self._comfy_lock:
            self._comfy_state["has_error"] = False
            self._comfy_state["error_summary"] = ""
            cstate = dict(self._comfy_state)
        self._emit_event({"type": "comfy_tick", "comfy": cstate})

    def _reset_comfy_offline(self) -> None:
        self._stop_ws()
        inferred_kind = "comfy"
        if self._has_running_session():
            inferred_kind = self._get_ep_service_kind()
        with self._comfy_lock:
            was_online = bool(self._comfy_state.get("online"))
            prev_kind = str(self._comfy_state.get("service_kind") or "comfy")
            self._comfy_state = self._default_comfy_state()
            self._comfy_state["service_kind"] = inferred_kind
            self._comfy_state["pending_pull_count"] = len(self._pending_pull_items)
            cstate = dict(self._comfy_state)
        with self._lock:
            self.last_snapshot["comfy"] = cstate
        if was_online or prev_kind != inferred_kind:
            self._emit_event({"type": "comfy_tick", "comfy": cstate})

    def _download_comfy_outputs(
        self, items: list[tuple[str, dict[str, str]]]
    ) -> list[Path]:
        """经由 ComfyUI /view 接口下载正式产出文件，自动附加服务端修改时间戳并防止同名异图覆写。"""
        saved: list[Path] = []
        self.download_root.mkdir(parents=True, exist_ok=True)
        for base_url, meta in items:
            url = f"{base_url.rstrip('/')}/view"
            subfolder = str(meta.get("subfolder") or "")
            raw_name = str(meta["filename"])
            params = {
                "filename": raw_name,
                "subfolder": subfolder,
                "type": meta.get("type", "output"),
            }
            tmp_path = self.download_root / f".dl_{uuid.uuid4().hex[:12]}.part"
            try:
                with self._http.get(url, params=params, stream=True, timeout=(8, 60)) as resp:
                    resp.raise_for_status()
                    mtime_sec = _parse_http_mtime(resp.headers.get("Last-Modified"))
                    if mtime_sec is None:
                        mtime_sec = time.time()
                    h = hashlib.sha256()
                    total_bytes = 0
                    with open(tmp_path, "wb") as fp:
                        for chunk in resp.iter_content(chunk_size=65536):
                            if chunk:
                                fp.write(chunk)
                                h.update(chunk)
                                total_bytes += len(chunk)

                norm_sub = subfolder.replace("\\", "/").strip("/")
                if _PRESERVE_ORIGINAL_SUB_RE.search(norm_sub) and raw_name.lower().endswith(".safetensors"):
                    final_name = raw_name
                else:
                    final_name = _format_timestamped_filename(raw_name, mtime_sec)

                dest = _safe_output_dest(self.download_root, subfolder, final_name)
                dest.parent.mkdir(parents=True, exist_ok=True)
                final_dest, skip_write = _resolve_non_clobber_dest(dest, total_bytes, h.hexdigest())
                if skip_write:
                    tmp_path.unlink(missing_ok=True)
                else:
                    tmp_path.replace(final_dest)
                    try:
                        os.utime(final_dest, (mtime_sec, mtime_sec))
                    except OSError:
                        pass
                saved.append(final_dest)
            except Exception:
                try:
                    tmp_path.unlink(missing_ok=True)
                except Exception:
                    pass
        return saved

    @staticmethod
    def _short_gpu_name(raw_name: str) -> str:
        name = re.sub(r"^cuda:\d+\s*", "", str(raw_name or "")).split(":")[0].strip()
        return name.replace("NVIDIA ", "").strip()

    @staticmethod
    def _collect_formal_outputs(entry: dict[str, Any]) -> list[dict[str, str]]:
        items: list[dict[str, str]] = []
        outputs = entry.get("outputs") or {}
        if not isinstance(outputs, dict):
            return items
        for node_out in outputs.values():
            if not isinstance(node_out, dict):
                continue
            for key in ("images", "gifs", "videos", "audio"):
                arr = node_out.get(key)
                if not isinstance(arr, list):
                    continue
                for item in arr:
                    if not isinstance(item, dict) or not item.get("filename"):
                        continue
                    if str(item.get("type") or "output") != "output":
                        continue
                    items.append(
                        {
                            "filename": str(item["filename"]),
                            "subfolder": str(item.get("subfolder") or ""),
                            "type": "output",
                        }
                    )
        return items

    @staticmethod
    def _extract_history_error(entry: dict[str, Any]) -> str | None:
        status = entry.get("status") or {}
        if str(status.get("status_str") or "").lower() != "error":
            return None
        node_t, exc_t, exc_m = None, None, None
        for msg in status.get("messages") or []:
            if isinstance(msg, (list, tuple)) and len(msg) >= 2 and msg[0] == "execution_error":
                data = msg[1] if isinstance(msg[1], dict) else {}
                node_t = data.get("node_type") or data.get("node_id")
                exc_t = data.get("exception_type")
                exc_m = data.get("exception_message")
                break
        return _format_comfy_error(
            str(node_t) if node_t else None,
            str(exc_t) if exc_t else None,
            str(exc_m) if exc_m else None,
        )

    def _clear_stale_deploy_error_locked(self) -> dict[str, Any] | None:
        """在持 self._lock 下清理服务恢复在线后残留的历史部署报错状态。"""
        if not self._deploy_state.get("active") and self._deploy_state.get("error_msg"):
            self._deploy_state["error_msg"] = ""
            if self._deploy_state.get("stage_title") == "部署失败":
                self._deploy_state["stage_title"] = ""
                self._deploy_state["sub_detail"] = ""
            snap_deploy = dict(self._deploy_state)
            self.last_snapshot["deploy"] = snap_deploy
            return snap_deploy
        return None

    def _probe_comfy_status(self, base: str) -> tuple[bool, dict[str, Any], list[dict[str, Any]]]:
        """探测 ComfyUI /system_stats、/queue 与 /history，返回 `(是否在线, 状态字典, 待发送事件列表)`。"""
        events: list[dict[str, Any]] = []
        try:
            r_stats = self._http.get(f"{base}/system_stats", timeout=4.0)
            r_stats.raise_for_status()
            stats = r_stats.json()
        except Exception:
            return False, {}, []

        devices = stats.get("devices") or []
        gpu_name = ""
        v_used = 0.0
        v_total = 0.0
        v_pct = 0.0
        if devices and isinstance(devices[0], dict):
            d0 = devices[0]
            gpu_name = self._short_gpu_name(d0.get("name", ""))
            tot_b = float(d0.get("vram_total") or 0.0)
            free_b = float(d0.get("vram_free") or 0.0)
            used_b = max(0.0, tot_b - free_b)
            if tot_b > 0:
                v_total = round(tot_b / (1024 ** 3), 1)
                v_used = round(used_b / (1024 ** 3), 1)
                v_pct = round(used_b * 100.0 / tot_b, 1)

        running_cnt = 0
        pending_cnt = 0
        try:
            r_q = self._http.get(f"{base}/queue", timeout=4.0)
            if r_q.status_code == 200:
                qdata = r_q.json()
                running_cnt = len(qdata.get("queue_running") or [])
                pending_cnt = len(qdata.get("queue_pending") or [])
        except Exception:
            pass

        with self._comfy_lock:
            prev_running = int(self._comfy_state.get("running_count") or 0)
            self._comfy_state["ws_connected"] = bool(
                self._ws_app is not None and self._ws_thread and self._ws_thread.is_alive()
            )
            self._comfy_state["tunnel_url"] = base
            self._comfy_state["gpu_name"] = gpu_name
            self._comfy_state["vram_used_gb"] = v_used
            self._comfy_state["vram_total_gb"] = v_total
            self._comfy_state["vram_pct"] = v_pct
            self._comfy_state["running_count"] = running_cnt
            self._comfy_state["pending_count"] = pending_cnt
            if running_cnt > 0 and prev_running == 0:
                self._comfy_state["has_error"] = False
                self._comfy_state["error_summary"] = ""
            if running_cnt == 0:
                self._comfy_state["step_value"] = 0
                self._comfy_state["step_max"] = 0
                self._comfy_state["step_pct"] = 0.0
                self._comfy_state["current_node"] = ""

        if running_cnt > 0 or pending_cnt > 0:
            self.touch_activity()

        try:
            r_hist = self._http.get(f"{base}/history", params={"max_items": 6}, timeout=5.0)
            if r_hist.status_code == 200:
                hdata = r_hist.json()
                if isinstance(hdata, dict):
                    first_conn = base not in self._seen_prompts_by_url
                    seen = self._seen_prompts_by_url.setdefault(base, set())
                    if first_conn:
                        seen.update(str(pid) for pid in hdata.keys())
                    else:
                        new_pids = [str(pid) for pid in hdata.keys() if str(pid) not in seen]
                        if new_pids:
                            self.touch_activity()
                        new_outputs: list[tuple[str, dict[str, str]]] = []
                        latest_err: tuple[str, str] | None = None
                        for pid in new_pids:
                            seen.add(pid)
                            entry = hdata.get(pid)
                            if not isinstance(entry, dict):
                                continue
                            err_str = self._extract_history_error(entry)
                            if err_str and pid not in self._reported_error_prompts:
                                self._reported_error_prompts.add(pid)
                                latest_err = (pid, err_str)
                            for meta in self._collect_formal_outputs(entry):
                                new_outputs.append((base, meta))

                        if latest_err:
                            with self._comfy_lock:
                                self._comfy_state["has_error"] = True
                                self._comfy_state["error_summary"] = latest_err[1]
                            events.append(
                                {
                                    "type": "comfy_error",
                                    "summary": latest_err[1],
                                    "prompt_id": latest_err[0],
                                }
                            )

                        if new_outputs:
                            with self._lock:
                                auto_pull = bool(
                                    (self.state.get("prefs") or {}).get("auto_pull_outputs", True)
                                )
                            if auto_pull:
                                saved = self._download_comfy_outputs(new_outputs)
                                count = len(saved) or len(new_outputs)
                                with self._comfy_lock:
                                    if saved:
                                        self._last_downloaded_files = saved
                                        self._comfy_state["last_output_name"] = saved[-1].name
                                    self._pending_pull_items.clear()
                                    self._comfy_state["pending_pull_count"] = 0
                                    self._comfy_state["last_output_count"] = count
                                events.append(
                                    {
                                        "type": "outputs_ready",
                                        "count": count,
                                        "pulled": True,
                                        "summary": f"✨ 新出图 {count} 份，已存本机！点我查看",
                                        "files": [str(p) for p in saved],
                                    }
                                )
                            else:
                                with self._comfy_lock:
                                    self._pending_pull_items.extend(new_outputs)
                                    self._comfy_state["pending_pull_count"] = len(
                                        self._pending_pull_items
                                    )
                                    self._comfy_state["last_output_count"] = len(new_outputs)
                                    self._comfy_state["last_output_name"] = new_outputs[-1][1][
                                        "filename"
                                    ]
                                events.append(
                                    {
                                        "type": "outputs_ready",
                                        "count": len(new_outputs),
                                        "pulled": False,
                                        "summary": f"✨ 新出图 {len(new_outputs)} 份！点我拉回查看",
                                    }
                                )
        except Exception:
            pass

        return (
            True,
            {
                "gpu_name": gpu_name,
                "vram_used_gb": v_used,
                "vram_total_gb": v_total,
                "vram_pct": v_pct,
                "running_count": running_cnt,
                "pending_count": pending_cnt,
            },
            events,
        )

    def _probe_llm_status(self, base: str, ep: str = "") -> tuple[bool, dict[str, Any]]:
        """探测 llama-server /health 与 /slots 状态，返回 `(是否在线, 推理指标字典)`。"""
        if not ep:
            target = self._pick_target_session("comfy")
            if target and target.get("endpoint"):
                ep = str(target["endpoint"])

        meta, api_key = self._get_ep_meta_and_key(ep)
        if not meta:
            with self._lock:
                for m_ep, m_val in self._service_meta_by_ep.items():
                    cand_urls = (
                        str(m_val.get("tunnel_url") or "").rstrip("/"),
                        str(m_val.get("llm_tunnel_url") or "").rstrip("/"),
                    )
                    if base in cand_urls:
                        ep = m_ep
                        meta = dict(m_val)
                        api_key = str(self._runtime_secrets.get(m_ep) or "")
                        break

        headers: dict[str, str] = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        try:
            r_health = self._http.get(f"{base}/health", headers=headers, timeout=4.0)
            if r_health.status_code != 200:
                raise RuntimeError(f"HTTP {r_health.status_code}")
            h_body = r_health.json()
            if not isinstance(h_body, dict) or str(h_body.get("status") or "").lower() != "ok":
                raise RuntimeError("unhealthy")
        except Exception:
            return False, {}

        running_cnt = 0
        tps: float | None = None
        try:
            r_slots = self._http.get(f"{base}/slots", headers=headers, timeout=4.0)
            if r_slots.status_code == 200:
                sdata = r_slots.json()
                slots_list = sdata if isinstance(sdata, list) else (
                    sdata.get("slots") if isinstance(sdata, dict) else []
                )
                if isinstance(slots_list, list):
                    active_tps: float | None = None
                    fallback_tps: float | None = None
                    for s_item in slots_list:
                        if not isinstance(s_item, dict):
                            continue
                        is_proc = bool(s_item.get("is_processing")) or (
                            int(s_item.get("state") or 0) == 1
                        )
                        cand_tps = self._extract_slot_tps(s_item)
                        if is_proc:
                            running_cnt += 1
                            if cand_tps is not None:
                                active_tps = cand_tps
                        elif cand_tps is not None and fallback_tps is None:
                            fallback_tps = cand_tps
                    tps = active_tps if active_tps is not None else (
                        fallback_tps if running_cnt > 0 else None
                    )
        except Exception:
            pass

        if running_cnt > 0:
            self.touch_activity()

        with self._lock:
            prefs = self.state.get("prefs") or {}
            pref_profs = _normalize_boot_profiles(
                prefs.get("boot_profiles"), prefs.get("boot_profile") or "anima"
            )
            use_27b = "qwen_27b" in pref_profs and "qwen_llm" not in pref_profs
            pref_quant = str(
                (prefs.get("boot_llm27_quant") if use_27b else prefs.get("boot_llm_quant"))
                or "UD-Q4_K_XL"
            )
            pref_ctx = int(
                (prefs.get("boot_llm27_ctx") if use_27b else prefs.get("boot_llm_ctx"))
                or (65536 if use_27b else 32768)
            )

        gpu_name = self._short_gpu_name(str(meta.get("gpu_name") or ""))
        v_used = round(float(meta.get("vram_used_gb") or 0.0), 1)
        v_total = round(float(meta.get("vram_total_gb") or 0.0), 1)
        v_pct = round(v_used * 100.0 / v_total, 1) if v_total > 0 else 0.0
        model_name = self._infer_default_llm_model_name(meta)
        quant = str(meta.get("quant") or pref_quant)
        ctx_size = int(meta.get("ctx_size") or pref_ctx)
        return (
            True,
            {
                "running_count": running_cnt,
                "tps": tps,
                "gpu_name": gpu_name,
                "vram_used_gb": v_used,
                "vram_total_gb": v_total,
                "vram_pct": v_pct,
                "model_name": model_name,
                "quant": quant,
                "ctx_size": ctx_size,
            },
        )

    def poll_comfy_http_once(self, tunnel_url: str) -> dict[str, Any]:
        """执行单轮 ComfyUI /system_stats、/queue 与 /history 轮询并触发提醒/拉回。"""
        base = tunnel_url.strip().rstrip("/")
        c_ok, _, events = self._probe_comfy_status(base)
        if not c_ok:
            with self._comfy_lock:
                self._comfy_http_failures += 1
                self._comfy_state["comfy_online"] = False
                if self._comfy_http_failures >= 2:
                    self._comfy_state["online"] = False
                cstate = dict(self._comfy_state)
            self._emit_event({"type": "comfy_tick", "comfy": cstate})
            return cstate

        with self._comfy_lock:
            self._comfy_http_failures = 0
            self._comfy_state["online"] = True
            self._comfy_state["service_kind"] = "comfy"
            self._comfy_state["comfy_online"] = True
            self._comfy_state["llm_online"] = False
            self._comfy_state["llm_tunnel_url"] = None
            self._comfy_state["llm_running_count"] = 0
            self._comfy_state["model_name"] = ""
            self._comfy_state["quant"] = ""
            self._comfy_state["ctx_size"] = 0
            self._comfy_state["tps"] = None
            self._comfy_state["pending_pull_count"] = len(self._pending_pull_items)
            cstate = dict(self._comfy_state)
        with self._lock:
            self.last_snapshot["comfy"] = cstate
            cleared_deploy = self._clear_stale_deploy_error_locked()
        if cleared_deploy is not None:
            self._emit_event({"type": "deploy_progress", "deploy": cleared_deploy})
        self._emit_event({"type": "comfy_tick", "comfy": cstate})
        for ev in events:
            self._emit_event(ev)
        return cstate

    @staticmethod
    def _extract_slot_tps(slot: dict[str, Any]) -> float | None:
        """从 llama-server /slots 单槽位字典中尝试提取生成速率 (tokens/s)。"""
        for container_key in ("timings", "performance"):
            sub = slot.get(container_key)
            if isinstance(sub, dict):
                for val_key in ("predicted_per_second", "tokens_per_second", "tps"):
                    raw = sub.get(val_key)
                    if isinstance(raw, (int, float)) and float(raw) > 0.05:
                        return round(float(raw), 1)
        nt = slot.get("next_token")
        if isinstance(nt, list) and nt and isinstance(nt[0], dict):
            nt = nt[0]
        if isinstance(nt, dict):
            for val_key in ("predicted_per_second", "tokens_per_second", "tps"):
                raw = nt.get(val_key)
                if isinstance(raw, (int, float)) and float(raw) > 0.05:
                    return round(float(raw), 1)
        return None

    def poll_llm_http_once(self, tunnel_url: str, ep: str = "") -> dict[str, Any]:
        """执行单轮 llama-server /health 与 /slots 健康度及生成状态轮询，更新 _comfy_state。"""
        base = tunnel_url.strip().rstrip("/")
        l_ok, info = self._probe_llm_status(base, ep=ep)
        if not l_ok:
            with self._comfy_lock:
                self._comfy_http_failures += 1
                self._comfy_state["service_kind"] = "llm"
                self._comfy_state["comfy_online"] = False
                self._comfy_state["llm_online"] = False
                self._comfy_state["ws_connected"] = False
                if self._comfy_http_failures >= 2:
                    self._comfy_state["online"] = False
                cstate = dict(self._comfy_state)
            with self._lock:
                self.last_snapshot["comfy"] = cstate
            self._emit_event({"type": "comfy_tick", "comfy": cstate})
            return cstate

        running_cnt = int(info.get("running_count") or 0)
        with self._comfy_lock:
            self._comfy_http_failures = 0
            self._comfy_state.update(
                {
                    "online": True,
                    "service_kind": "llm",
                    "comfy_online": False,
                    "llm_online": True,
                    "ws_connected": False,
                    "tunnel_url": base,
                    "llm_tunnel_url": base,
                    "running_count": running_cnt,
                    "pending_count": 0,
                    "llm_running_count": running_cnt,
                    "step_value": 0,
                    "step_max": 0,
                    "step_pct": 0.0,
                    "current_node": "",
                    "gpu_name": info.get("gpu_name", ""),
                    "vram_used_gb": info.get("vram_used_gb", 0.0),
                    "vram_total_gb": info.get("vram_total_gb", 0.0),
                    "vram_pct": info.get("vram_pct", 0.0),
                    "has_error": False,
                    "error_summary": "",
                    "pending_pull_count": len(self._pending_pull_items),
                    "model_name": info.get("model_name", "Qwen3.8-Flash-Next-abliterated"),
                    "quant": info.get("quant", ""),
                    "ctx_size": info.get("ctx_size", 0),
                    "tps": info.get("tps"),
                }
            )
            cstate = dict(self._comfy_state)
        with self._lock:
            self.last_snapshot["comfy"] = cstate
            cleared_deploy = self._clear_stale_deploy_error_locked()
        if cleared_deploy is not None:
            self._emit_event({"type": "deploy_progress", "deploy": cleared_deploy})
        self._emit_event({"type": "comfy_tick", "comfy": cstate})
        return cstate

    def poll_dual_once(
        self, comfy_url: str, llm_url: str, ep: str = ""
    ) -> dict[str, Any]:
        """双栖同驻单轮同步轮询：同时汇总 ComfyUI 与 Qwen3.8 LLM 健康度、显存与并发指标。"""
        comfy_base = comfy_url.strip().rstrip("/") if comfy_url else ""
        llm_base = llm_url.strip().rstrip("/") if llm_url else ""

        c_ok = False
        events: list[dict[str, Any]] = []
        if comfy_base:
            c_ok, _, events = self._probe_comfy_status(comfy_base)

        l_ok = False
        l_info: dict[str, Any] = {}
        if llm_base:
            l_ok, l_info = self._probe_llm_status(llm_base, ep=ep)

        any_ok = c_ok or l_ok
        with self._comfy_lock:
            self._comfy_state["service_kind"] = "dual"
            self._comfy_state["comfy_online"] = c_ok
            self._comfy_state["llm_online"] = l_ok
            if comfy_base:
                self._comfy_state["tunnel_url"] = comfy_base
            if llm_base:
                self._comfy_state["llm_tunnel_url"] = llm_base
            if not c_ok:
                self._comfy_state["ws_connected"] = False
                self._comfy_state["running_count"] = 0
                self._comfy_state["pending_count"] = 0

            if any_ok:
                self._comfy_http_failures = 0
                self._comfy_state["online"] = True
                if not c_ok and l_ok and float(l_info.get("vram_total_gb") or 0.0) > 0:
                    self._comfy_state["gpu_name"] = l_info.get("gpu_name", "")
                    self._comfy_state["vram_used_gb"] = l_info.get("vram_used_gb", 0.0)
                    self._comfy_state["vram_total_gb"] = l_info.get("vram_total_gb", 0.0)
                    self._comfy_state["vram_pct"] = l_info.get("vram_pct", 0.0)
                if l_ok:
                    self._comfy_state["llm_running_count"] = int(l_info.get("running_count") or 0)
                    self._comfy_state["model_name"] = l_info.get("model_name", "")
                    self._comfy_state["quant"] = l_info.get("quant", "")
                    self._comfy_state["ctx_size"] = l_info.get("ctx_size", 0)
                    self._comfy_state["tps"] = l_info.get("tps")
                else:
                    self._comfy_state["llm_running_count"] = 0
                    self._comfy_state["tps"] = None
            else:
                self._comfy_http_failures += 1
                self._comfy_state["llm_running_count"] = 0
                self._comfy_state["tps"] = None
                if self._comfy_http_failures >= 2:
                    self._comfy_state["online"] = False

            self._comfy_state["pending_pull_count"] = len(self._pending_pull_items)
            cstate = dict(self._comfy_state)

        with self._lock:
            self.last_snapshot["comfy"] = cstate
            cleared_deploy = self._clear_stale_deploy_error_locked() if any_ok else None
        if cleared_deploy is not None:
            self._emit_event({"type": "deploy_progress", "deploy": cleared_deploy})
        self._emit_event({"type": "comfy_tick", "comfy": cstate})
        for ev in events:
            self._emit_event(ev)
        return cstate

    def pull_or_open_latest_outputs(self) -> dict[str, Any]:
        """点击出图提醒时按需拉回待下载产出，或直接在资源管理器高亮选中最新成图。"""
        self.touch_activity()
        with self._comfy_lock:
            pending = list(self._pending_pull_items)
            recent = list(self._last_downloaded_files)

        pulled_now = False
        if pending:
            saved = self._download_comfy_outputs(pending)
            if saved:
                pulled_now = True
                with self._comfy_lock:
                    self._pending_pull_items.clear()
                    self._last_downloaded_files = saved
                    self._comfy_state["pending_pull_count"] = 0
                    self._comfy_state["last_output_name"] = saved[-1].name
                    cstate = dict(self._comfy_state)
                recent = saved
                self._emit_event({"type": "comfy_tick", "comfy": cstate})

        existing = [p for p in recent if p.exists()]
        if existing:
            latest = existing[-1].resolve()
            try:
                subprocess.Popen(
                    ["explorer.exe", f"/select,{latest}"],
                    creationflags=CREATE_NO_WINDOW,
                )
            except Exception:
                try:
                    os.startfile(str(latest.parent))
                except Exception:
                    pass
            msg = f"已拉回并定位: {latest.name}" if pulled_now else f"已打开位置: {latest.name}"
            return {"ok": True, "path": str(latest), "message": msg}

        if self.download_root.exists():
            try:
                os.startfile(str(self.download_root))
                return {"ok": True, "message": "已打开本地成图目录"}
            except Exception:
                pass
        return self.sync_outputs()

    def _on_ws_message(self, _ws: Any, message: Any) -> None:
        if isinstance(message, (bytes, bytearray)) or not isinstance(message, str):
            return
        try:
            payload = json.loads(message)
        except Exception:
            return
        mtype = payload.get("type")
        mdata = payload.get("data") if isinstance(payload.get("data"), dict) else {}

        if mtype in ("execution_start", "progress", "progress_state", "executing"):
            self.touch_activity()

        if mtype == "execution_start":
            with self._comfy_lock:
                self._comfy_state["has_error"] = False
                self._comfy_state["error_summary"] = ""
                self._comfy_state["step_value"] = 0
                self._comfy_state["step_max"] = 0
                self._comfy_state["step_pct"] = 0.0
                cstate = dict(self._comfy_state)
            self._emit_event({"type": "comfy_tick", "comfy": cstate})
        elif mtype == "progress":
            val = int(mdata.get("value") or 0)
            mx = int(mdata.get("max") or 0)
            pct = round(val * 100.0 / mx, 1) if mx > 0 else 0.0
            with self._comfy_lock:
                self._comfy_state["step_value"] = val
                self._comfy_state["step_max"] = mx
                self._comfy_state["step_pct"] = pct
                self._comfy_state["running_count"] = max(1, int(self._comfy_state.get("running_count") or 0))
                cstate = dict(self._comfy_state)
            now = time.monotonic()
            if now - self._ws_last_emit_ts >= 0.25 or pct >= 100.0:
                self._ws_last_emit_ts = now
                self._emit_event({"type": "comfy_tick", "comfy": cstate})
        elif mtype == "progress_state":
            nodes = mdata.get("nodes") if isinstance(mdata.get("nodes"), dict) else {}
            picked = None
            for ninfo in nodes.values():
                if isinstance(ninfo, dict) and ninfo.get("state") == "running" and (ninfo.get("max") or 0) > 0:
                    picked = ninfo
                    break
            if picked:
                val = int(picked.get("value") or 0)
                mx = int(picked.get("max") or 0)
                pct = round(val * 100.0 / mx, 1) if mx > 0 else 0.0
                with self._comfy_lock:
                    self._comfy_state["step_value"] = val
                    self._comfy_state["step_max"] = mx
                    self._comfy_state["step_pct"] = pct
                    self._comfy_state["running_count"] = max(1, int(self._comfy_state.get("running_count") or 0))
                    cstate = dict(self._comfy_state)
                now = time.monotonic()
                if now - self._ws_last_emit_ts >= 0.25 or pct >= 100.0:
                    self._ws_last_emit_ts = now
                    self._emit_event({"type": "comfy_tick", "comfy": cstate})
        elif mtype == "executing":
            node_id = mdata.get("node")
            if node_id is None and mdata.get("prompt_id"):
                self._wake_monitor.set()
            elif node_id is not None:
                with self._comfy_lock:
                    self._comfy_state["current_node"] = str(node_id)
        elif mtype == "execution_error":
            summary = _format_comfy_error(
                str(mdata.get("node_type") or mdata.get("node_id") or "节点"),
                str(mdata.get("exception_type") or ""),
                str(mdata.get("exception_message") or ""),
            )
            pid = str(mdata.get("prompt_id") or "")
            if pid:
                self._reported_error_prompts.add(pid)
            with self._comfy_lock:
                self._comfy_state["has_error"] = True
                self._comfy_state["error_summary"] = summary
                self._comfy_state["step_value"] = 0
                self._comfy_state["step_max"] = 0
                self._comfy_state["step_pct"] = 0.0
                cstate = dict(self._comfy_state)
            self._emit_event({"type": "comfy_tick", "comfy": cstate})
            self._emit_event({"type": "comfy_error", "summary": summary, "prompt_id": pid})

    def _stop_ws(self) -> None:
        self._ws_stop.set()
        app = self._ws_app
        self._ws_app = None
        self._ws_target_url = None
        if app is not None:
            try:
                app.close()
            except Exception:
                pass

    def _ensure_ws_target(self, tunnel_url: str) -> None:
        if websocket is None:
            return
        base = tunnel_url.strip().rstrip("/")
        if self._ws_target_url == base and self._ws_thread and self._ws_thread.is_alive():
            return
        self._stop_ws()
        self._ws_stop.clear()
        self._ws_target_url = base
        ws_base = re.sub(r"^http", "ws", base, count=1)
        ws_url = f"{ws_base}/ws?clientId=colab_pet_{uuid.uuid4().hex[:8]}"

        def _ws_runner() -> None:
            while not self._ws_stop.is_set() and not self._monitor_stop.is_set():
                try:
                    app = websocket.WebSocketApp(ws_url, on_message=self._on_ws_message)
                    self._ws_app = app
                    app.run_forever(ping_interval=20, ping_timeout=10)
                except Exception:
                    pass
                finally:
                    self._ws_app = None
                if self._ws_stop.wait(timeout=3.0):
                    break

        self._ws_thread = threading.Thread(target=_ws_runner, daemon=True)
        self._ws_thread.start()

    def _send_colab_keepalive(self) -> bool:
        """通过 Colab 官方通道执行一次无窗口三层保活心跳（TFE keep-alive / Proxy / 空闲内核）。"""
        colab_py = self._get_colab_py()
        if not colab_py.exists():
            return False
        try:
            proc = _run_hidden([str(colab_py), "-c", _COLAB_KEEPALIVE_SCRIPT], timeout=25)
        except Exception:
            return False
        if proc.returncode != 0:
            return False
        matched = re.search(
            r"<<<COLAB_KEEPALIVE>>>(.*?)<<<END_COLAB_KEEPALIVE>>>",
            proc.stdout or "",
            re.S,
        )
        if not matched:
            return False
        try:
            data = json.loads(matched.group(1).strip())
            return bool(data.get("ok")) and int(data.get("pinged") or 0) > 0
        except Exception:
            return False

    def _maybe_trigger_keepalive(self, now_m: float | None = None) -> None:
        """按 10 分钟节流周期向活跃 Colab 实例发送防断连保活心跳（连续纯空闲超 45m 自动停发，防通宵续命烧卡）。"""
        if self.is_deploying():
            return
        cur_m = time.monotonic() if now_m is None else float(now_m)
        if not self._has_running_session():
            with self._lock:
                self._last_keepalive_ts = 0.0
                self._last_activity_m = 0.0
            return

        with self._lock:
            keep_alive = bool((self.state.get("prefs") or {}).get("keep_alive", True))
            if not keep_alive:
                return
            if self._last_activity_m <= 0.0:
                self._last_activity_m = cur_m
            idle_sec = max(0.0, cur_m - self._last_activity_m)
            if idle_sec > KEEPALIVE_MAX_IDLE_SEC:
                return
            last_ts = self._last_keepalive_ts
            if last_ts > 0.0 and (cur_m - last_ts) < KEEPALIVE_INTERVAL_SEC:
                return

        if not self._keepalive_lock.acquire(blocking=False):
            return

        # 先记录本次出发时间戳避免监控线程高频率重复拉起子进程
        with self._lock:
            if self._last_keepalive_ts > 0.0 and (cur_m - self._last_keepalive_ts) < KEEPALIVE_INTERVAL_SEC:
                self._keepalive_lock.release()
                return
            self._last_keepalive_ts = cur_m

        def _worker() -> None:
            ok = False
            try:
                ok = self._send_colab_keepalive()
            except Exception:
                ok = False
            finally:
                if not ok and not self._monitor_stop.is_set():
                    with self._lock:
                        # 失败时回拨时间戳，使约 2 分钟后可自动重试一次
                        if self._last_keepalive_ts == cur_m:
                            self._last_keepalive_ts = max(
                                1e-6,
                                cur_m - (KEEPALIVE_INTERVAL_SEC - KEEPALIVE_RETRY_COOLDOWN_SEC),
                            )
                self._keepalive_lock.release()

        threading.Thread(target=_worker, daemon=True).start()

    def _maybe_trigger_idle_auto_stop(self, now_m: float | None = None) -> None:
        """若确认远端服务在线且连续全空闲达设定阈值（默认 60m），先归档云盘再自动停卡止损。"""
        if self.is_deploying() or self._upload_lock.locked() or self._action_lock.locked():
            return
        if not self._has_running_session():
            return

        cur_m = time.monotonic() if now_m is None else float(now_m)
        with self._lock:
            prefs = self.state.get("prefs") or {}
            idle_stop_mins = int(prefs.get("idle_auto_stop_minutes", 60) or 0)
            if idle_stop_mins <= 0:
                return
            if self._last_activity_m <= 0.0:
                self._last_activity_m = cur_m
                return
            last_act = self._last_activity_m
            cur_ep = str(self.state.get("active_endpoint") or "__active__")
            warned_ep = self._idle_warned_ep

        with self._comfy_lock:
            comfy_online = bool(self._comfy_state.get("online"))
            busy_cnt = (
                int(self._comfy_state.get("running_count") or 0)
                + int(self._comfy_state.get("pending_count") or 0)
                + int(self._comfy_state.get("llm_running_count") or 0)
            )
        if busy_cnt > 0:
            self.touch_activity()
            return
        if not comfy_online:
            return

        idle_sec = max(0.0, cur_m - last_act)
        limit_sec = float(idle_stop_mins * 60)
        warn_sec = max(60.0, limit_sec - 120.0)

        if idle_sec >= limit_sec:
            if not self._idle_stop_lock.acquire(blocking=False):
                return

            def _idle_stop_worker() -> None:
                try:
                    if (
                        self.is_deploying()
                        or self._upload_lock.locked()
                        or not self._has_running_session()
                    ):
                        return
                    with self._comfy_lock:
                        now_busy = (
                            int(self._comfy_state.get("running_count") or 0)
                            + int(self._comfy_state.get("pending_count") or 0)
                            + int(self._comfy_state.get("llm_running_count") or 0)
                        )
                        now_online = bool(self._comfy_state.get("online"))
                    if not now_online or now_busy > 0:
                        if now_busy > 0:
                            self.touch_activity()
                        return
                    if (time.monotonic() - self._last_activity_m) < limit_sec:
                        return

                    if not self.is_llm_mode_active() and CTL_SCRIPT.exists():
                        target = self._pick_target_session("comfy")
                        sname = (
                            target["name"]
                            if target and target.get("name") not in (None, "?")
                            else "comfy"
                        )
                        try:
                            _run_hidden(
                                [_hidden_python_exe(), str(CTL_SCRIPT), "sync", "-s", sname],
                                timeout=120,
                            )
                        except Exception:
                            pass

                    res = self.stop_all_instances()
                    if res.get("ok"):
                        self._emit_event(
                            {
                                "type": "idle_stopped",
                                "summary": f"🛑 连续闲置满 {idle_stop_mins} 分钟，已归档并自动停卡省积分",
                                "snapshot": res.get("snapshot") or self.get_cached_snapshot(),
                            }
                        )
                    else:
                        with self._lock:
                            self._idle_warned_ep = None
                        self._emit_event(
                            {
                                "type": "idle_warning",
                                "summary": f"⚠️ 闲置自动停卡未成功: {res.get('message', '请手动停卡')}",
                                "action": "keep_working",
                            }
                        )
                finally:
                    self._idle_stop_lock.release()

            threading.Thread(target=_idle_stop_worker, daemon=True).start()
        elif idle_sec >= warn_sec and warned_ep != cur_ep:
            with self._lock:
                self._idle_warned_ep = cur_ep
            self._emit_event(
                {
                    "type": "idle_warning",
                    "summary": f"⏳ 已闲置 {int(idle_sec // 60)} 分钟，约 2 分钟后将自动存档停卡（点我续期）",
                    "action": "keep_working",
                }
            )

    def start_monitor(self, on_event: Callable[[dict[str, Any]], None]) -> None:
        """启动后台 ComfyUI / LLM 服务状态与显存监控守护线程。"""
        self._event_cb = on_event
        if self._monitor_thread and self._monitor_thread.is_alive():
            return
        self._monitor_stop.clear()

        def _loop() -> None:
            while not self._monitor_stop.is_set():
                self._wake_monitor.clear()
                wait_s = 2.0
                if self.is_deploying():
                    wait_s = 2.0
                elif not self._has_running_session():
                    self._maybe_trigger_keepalive()
                    self._reset_comfy_offline()
                else:
                    self._maybe_trigger_keepalive()
                    turl = self.resolve_active_tunnel_url(allow_fast_discover=True)
                    target_sess = self._pick_target_session("comfy")
                    cur_ep = str(target_sess.get("endpoint") or "") if target_sess else ""
                    skind = self._get_ep_service_kind(cur_ep)
                    if turl:
                        if skind == "llm":
                            # LLM 模式下主动关闭 ComfyUI /ws 连接，防止向 llama-server 刷 404 重连风暴
                            self._stop_ws()
                            cstate = self.poll_llm_http_once(turl, ep=cur_ep)
                            busy = (cstate.get("running_count") or 0) > 0
                            wait_s = 2.5 if busy else 8.0
                        elif skind == "dual":
                            self._ensure_ws_target(turl)
                            llm_u = self.resolve_active_llm_tunnel_url(allow_fast_discover=True) or ""
                            cstate = self.poll_dual_once(turl, llm_u, ep=cur_ep)
                            busy = (
                                (cstate.get("running_count") or 0)
                                + (cstate.get("pending_count") or 0)
                                + (cstate.get("llm_running_count") or 0)
                            ) > 0
                            wait_s = 2.5 if busy else 6.0
                        else:
                            self._ensure_ws_target(turl)
                            cstate = self.poll_comfy_http_once(turl)
                            busy = (cstate.get("running_count") or 0) + (cstate.get("pending_count") or 0) > 0
                            wait_s = 2.5 if busy else 6.0
                    elif skind == "dual":
                        self._stop_ws()
                        llm_u = self.resolve_active_llm_tunnel_url(allow_fast_discover=True) or ""
                        if llm_u:
                            cstate = self.poll_dual_once("", llm_u, ep=cur_ep)
                            busy = (cstate.get("llm_running_count") or 0) > 0
                            wait_s = 2.5 if busy else 6.0
                        else:
                            self._reset_comfy_offline()
                            wait_s = 5.0
                    else:
                        self._reset_comfy_offline()
                        wait_s = 5.0
                    self._maybe_trigger_idle_auto_stop()
                self._wake_monitor.wait(timeout=wait_s)

        self._monitor_thread = threading.Thread(target=_loop, daemon=True)
        self._monitor_thread.start()

    def stop_monitor(self) -> None:
        """停止后台监控线程与 WebSocket 连接（退出时同步清理未完成部署进程）。"""
        self._monitor_stop.set()
        self._wake_monitor.set()
        self._stop_ws()
        if self._deploy_proc is not None:
            self._deploy_cancelled = True
            self._kill_process_tree(self._deploy_proc)

    # =========================================================================
    # 功能 4：文件拖拽与智能分流投递（参考图 / 工作流 JSON / PNG 彩蛋 / LoRA）
    # =========================================================================

    def _emit_upload_progress(self, name: str, pct: int, stage: str) -> None:
        self._emit_event(
            {
                "type": "upload_progress",
                "name": name,
                "pct": max(0, min(100, int(pct))),
                "summary": f"📤 {stage} {name} ({max(0, min(100, int(pct)))}%)",
            }
        )

    def _rclone_upload_file(
        self,
        src: Path,
        remote_dst: str,
        display_name: str,
        pct_start: int = 0,
        pct_span: int = 100,
    ) -> None:
        rclone_exe = self._get_rclone_exe()
        rclone_conf = self._get_rclone_conf()
        if not rclone_exe.is_file():
            raise FileNotFoundError(f"未找到 rclone.exe: {rclone_exe}")
        if not rclone_conf.is_file():
            raise FileNotFoundError(f"未找到 rclone.conf: {rclone_conf}")

        cmd = [
            str(rclone_exe),
            "copyto",
            "--config",
            str(rclone_conf),
            "--disable-http2",
            "--stats",
            "1s",
            "--stats-one-line",
            str(src),
            remote_dst,
        ]
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=_clean_env(),
            creationflags=CREATE_NO_WINDOW,
        )
        assert proc.stdout is not None
        tail_lines: list[str] = []
        buf = ""
        last_pct = -1
        try:
            while True:
                ch = proc.stdout.read(256)
                if not ch:
                    break
                buf += ch.replace("\r", "\n")
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    sline = line.strip()
                    if sline:
                        tail_lines = (tail_lines + [sline])[-8:]
                        m = re.search(r"(\d{1,3})%", sline)
                        if m:
                            raw_pct = min(100, int(m.group(1)))
                            mapped = pct_start + int(round(raw_pct * pct_span / 100.0))
                            if mapped != last_pct:
                                last_pct = mapped
                                self._emit_upload_progress(display_name, mapped, "传云盘")
        finally:
            proc.stdout.close()

        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(tail_lines[-1] if tail_lines else f"rclone 退出码 {rc}")

    def _write_remote_file_via_contents(
        self, rel_path: str, raw_bytes: bytes, session: str = "comfy"
    ) -> None:
        colab_py = self._get_colab_py()
        if not colab_py.exists():
            raise FileNotFoundError(f"未找到 colab-cli Python: {colab_py}")
        target = self._pick_target_session(session)
        ep = str(target.get("endpoint") or "") if target else ""
        sname = target["name"] if target and target.get("name") not in (None, "?") else session
        b64 = base64.b64encode(raw_bytes).decode("ascii")
        proc = subprocess.run(
            [str(colab_py), "-c", _COLAB_CONTENTS_PUT_SCRIPT, ep, sname, rel_path],
            input=b64,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=25,
            env=_clean_env(),
            creationflags=CREATE_NO_WINDOW,
        )
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "").strip().splitlines()
            raise RuntimeError(err[-1] if err else "Contents 写入失败")

    def _fetch_comfy_object_info(self, tunnel_url: str) -> dict[str, Any] | None:
        """从在线 ComfyUI 拉取节点定义（/object_info），用于将 API prompt 还原为标准画布工作流。"""
        base = tunnel_url.rstrip("/")
        try:
            resp = self._http.get(f"{base}/object_info", timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, dict):
                    return data
        except Exception:
            pass
        return None

    def _upload_workflow_file_ex(
        self, file_path: Path, is_png_extract: bool, tunnel_url: str | None
    ) -> tuple[str, bool, str]:
        """解析、按需转换并上传工作流 JSON，返回 `(状态文案, 是否已写入在线实例, 安全文件名)`。"""
        if is_png_extract:
            wf_text = extract_png_comfy_workflow(file_path)
            if not wf_text:
                raise ValueError(f"{file_path.name} 中未发现合法 ComfyUI 工作流")
            parsed = json.loads(wf_text)
            if not isinstance(parsed, dict):
                raise ValueError(f"{file_path.name} 中提取的工作流结构无效")
            safe_name = _sanitize_filename(file_path.with_suffix(".json").name, "workflow.json")
        else:
            wf_text = file_path.read_text(encoding="utf-8")
            parsed = json.loads(wf_text)
            if not isinstance(parsed, dict):
                raise ValueError("JSON 顶层结构不是工作流对象")
            safe_name = _sanitize_filename(file_path.name, "workflow.json")

        self._emit_upload_progress(safe_name, 20, "导工作流")
        if _is_api_prompt_dict(parsed):
            obj_info = self._fetch_comfy_object_info(tunnel_url) if tunnel_url else None
            converted = _api_prompt_to_ui_workflow(parsed, obj_info)
            wf_text = json.dumps(converted, ensure_ascii=False, indent=2)
        elif _ensure_workflow_uuid(parsed):
            wf_text = json.dumps(parsed, ensure_ascii=False, indent=2)

        self._emit_upload_progress(safe_name, 30, "导工作流")
        wf_bytes = wf_text.encode("utf-8")

        if tunnel_url:
            base = tunnel_url.rstrip("/")
            target_ep = f"{base}/userdata/{urllib.parse.quote('workflows/' + safe_name, safe='')}?overwrite=true"
            resp = self._http.post(
                target_ep,
                data=wf_bytes,
                headers={"Content-Type": "application/json"},
                timeout=15,
            )
            if resp.status_code in (200, 201):
                self._backup_workflow_to_gdrive_async(safe_name, wf_bytes)
                self._emit_upload_progress(safe_name, 90, "导工作流")
                return f"✅ 已导入工作流: {safe_name} (点我打开画布)", True, safe_name
            if resp.status_code != 404:
                resp.raise_for_status()

        if self._has_running_session():
            self._write_remote_file_via_contents(
                f"content/ComfyUI/user/default/workflows/{safe_name}",
                wf_bytes,
            )
            self._backup_workflow_to_gdrive_async(safe_name, wf_bytes)
            self._emit_upload_progress(safe_name, 100, "导工作流")
            return f"✅ 已写入实例工作流: {safe_name} (点我打开画布)", True, safe_name

        with tempfile.TemporaryDirectory(prefix="pet_wf_") as tmp_dir:
            tmp_wf = Path(tmp_dir) / safe_name
            tmp_wf.write_bytes(wf_bytes)
            self._rclone_upload_file(tmp_wf, f"{GDRIVE_WF_REMOTE}/{safe_name}", safe_name)
        return f"✅ 工作流已存云盘(下次开机自动载入): {safe_name}", False, safe_name

    def _upload_workflow_file(
        self, file_path: Path, is_png_extract: bool, tunnel_url: str | None
    ) -> tuple[str, bool]:
        """保留向后兼容的二元组返回签名。"""
        msg, online_ok, _ = self._upload_workflow_file_ex(file_path, is_png_extract, tunnel_url)
        return msg, online_ok

    def _upload_media_file(self, file_path: Path, tunnel_url: str | None) -> tuple[str, str]:
        if not tunnel_url:
            raise RuntimeError("当前无在线 ComfyUI 隧道，请先开机后再传参考素材")
        base = tunnel_url.rstrip("/")
        url = f"{base}/upload/image"
        with _StreamingMultipartFile(
            file_path,
            field_name="image",
            fields={"type": "input", "overwrite": "true"},
            on_progress=lambda pct: self._emit_upload_progress(file_path.name, pct, "传参考图"),
        ) as stream:
            resp = self._http.post(
                url,
                data=stream,
                headers={
                    "Content-Type": stream.content_type,
                    "Content-Length": str(len(stream)),
                },
                timeout=(10, 300),
            )
            resp.raise_for_status()
        uploaded_name = file_path.name
        try:
            data = resp.json()
            if isinstance(data, dict) and data.get("name"):
                uploaded_name = str(data["name"])
        except Exception:
            pass
        return f"✅ 已投递 {uploaded_name}，文件名已复制", uploaded_name

    def _upload_model_file(
        self, file_path: Path, tunnel_url: str | None, session: str = "comfy"
    ) -> tuple[str, bool]:
        safe_name = _sanitize_filename(file_path.name, "model.safetensors")
        remote_path = f"{GDRIVE_LORA_REMOTE}/{safe_name}"
        running = self._has_running_session()
        self._rclone_upload_file(
            file_path,
            remote_path,
            safe_name,
            pct_start=0,
            pct_span=70 if running else 100,
        )
        if not running:
            return f"✅ LoRA 已零耗电预传云盘: {safe_name}", False

        self._emit_upload_progress(safe_name, 78, "同步到实例")
        colab_py = self._get_colab_py()
        if colab_py.exists():
            target = self._pick_target_session(session)
            sname = target["name"] if target and target.get("name") not in (None, "?") else session
            proc = _run_hidden(
                [
                    str(colab_py),
                    "-c",
                    _COLAB_REMOTE_LORA_PULL_SCRIPT,
                    sname,
                    remote_path,
                    f"/content/ComfyUI/models/loras/{safe_name}",
                ],
                timeout=180,
            )
            if proc.returncode != 0:
                err = (proc.stderr or proc.stdout or "").strip().splitlines()
                raise RuntimeError(err[-1] if err else "实例拉取 LoRA 失败")

        if tunnel_url:
            try:
                self._http.get(f"{tunnel_url.rstrip('/')}/object_info/LoraLoader", timeout=5)
            except Exception:
                pass
        self._emit_upload_progress(safe_name, 100, "同步到实例")
        return f"✅ LoRA 已同步至云端并刷新: {safe_name}", True

    @staticmethod
    def _read_text_file_safe(fpath: Path, max_chars: int = 200_000) -> tuple[str, bool]:
        """安全读取本地纯文本文件（自动兼容 UTF-8 / GB18030，超长时平滑截断）。"""
        raw = fpath.read_bytes()
        text = ""
        for enc in ("utf-8-sig", "utf-8", "gb18030"):
            try:
                text = raw.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        else:
            text = raw.decode("utf-8", errors="replace")
        truncated = len(text) > max_chars
        if truncated:
            text = text[:max_chars] + "\n...[内容过长已截断]..."
        return text, truncated

    def _handle_llm_dropped_files(self, valid_files: list[Path]) -> dict[str, Any]:
        """在 LLM 对话模式下分流处理拖入或选取的文件：文本提取入剪贴板，图片引导网页直传。"""
        text_files = [p for p in valid_files if p.suffix.lower() in LLM_TEXT_EXTS]
        image_files = [p for p in valid_files if p.suffix.lower() in LLM_IMAGE_EXTS]
        gguf_or_model = [
            p for p in valid_files if p.suffix.lower() == ".gguf" or p.suffix.lower() in MODEL_EXTS
        ]
        running = self._has_running_session()
        done_action = "open_llm_webui" if running else None

        if text_files:
            chunks: list[str] = []
            any_trunc = False
            for fp in text_files:
                body, trunc = self._read_text_file_safe(fp)
                any_trunc = any_trunc or trunc
                if len(text_files) == 1:
                    chunks.append(body)
                else:
                    chunks.append(f"=== {fp.name} ===\n{body}")
            clip_text = "\n\n".join(chunks)
            trunc_tag = "（超长已截断）" if any_trunc else ""
            if len(text_files) == 1:
                summary = (
                    f"📄 已提取 {text_files[0].name} ({len(clip_text)} 字{trunc_tag}) 到剪贴板，可直接 Ctrl+V 提问"
                )
            else:
                summary = (
                    f"📄 已合并提取 {len(text_files)} 份文本 ({len(clip_text)} 字{trunc_tag}) 到剪贴板"
                )
            self._emit_event(
                {
                    "type": "upload_done",
                    "summary": summary,
                    "clipboard_text": clip_text,
                    "action": done_action,
                }
            )
            return {
                "ok": True,
                "started": False,
                "service_kind": "llm",
                "count": len(text_files),
                "clipboard_text": clip_text,
                "message": summary,
            }

        if image_files:
            summary = "🖼️ 识图请直接将图片拖入或粘贴到 Qwen3.8 对话网页输入框（点我打开对话页）"
            self._emit_event(
                {
                    "type": "upload_done",
                    "summary": summary,
                    "action": done_action,
                }
            )
            return {
                "ok": True,
                "started": False,
                "service_kind": "llm",
                "count": len(image_files),
                "message": summary,
            }

        if gguf_or_model:
            msg = "💡 Qwen3.8 量化档位请在启动抽屉中切换 (Q4 / IQ4 / IQ3)，无需手动上传模型权重"
            self._emit_event({"type": "upload_error", "summary": msg})
            return {"ok": False, "service_kind": "llm", "message": msg}

        msg = "⚠️ 对话模式支持提取文本 (.txt/.md/.json/.py/.yaml/.csv) 或将图片拖入对话网页"
        self._emit_event({"type": "upload_error", "summary": msg})
        return {"ok": False, "service_kind": "llm", "message": msg}

    def submit_dropped_files(
        self,
        paths: list[str],
        png_mode: str | None = None,
        base_url_override: str | None = None,
    ) -> dict[str, Any]:
        """智能识别拖入或选取的本地文件，分流上传为参考素材、工作流、LoRA 或 LLM 文本提取。"""
        self.touch_activity()
        valid_files: list[Path] = []
        for raw_p in paths or []:
            if not raw_p:
                continue
            p = Path(str(raw_p))
            if p.is_file():
                valid_files.append(p)

        if not valid_files:
            return {"ok": False, "message": "未识别到有效本地文件"}

        if base_url_override is None and png_mode is None and self.is_llm_mode_active():
            return self._handle_llm_dropped_files(valid_files)

        single_wf_png = (
            len(valid_files) == 1
            and valid_files[0].suffix.lower() == ".png"
            and png_mode is None
            and extract_png_comfy_workflow(valid_files[0])
        )

        if base_url_override is None and png_mode is None and self.is_dual_mode_active():
            pure_llm_text_exts = LLM_TEXT_EXTS - WORKFLOW_EXTS
            if all(p.suffix.lower() in pure_llm_text_exts for p in valid_files):
                return self._handle_llm_dropped_files(valid_files)
            if not single_wf_png and all(p.suffix.lower() in LLM_IMAGE_EXTS for p in valid_files):
                self._emit_event(
                    {
                        "type": "dual_image_choice",
                        "paths": [str(p) for p in valid_files],
                        "name": valid_files[0].name,
                        "count": len(valid_files),
                    }
                )
                return {
                    "ok": True,
                    "choice_required": True,
                    "service_kind": "dual",
                    "message": "检测到图片，请选择传 ComfyUI 参考图或打开 27B 识图页",
                }

        if single_wf_png:
            target_png = valid_files[0]
            self._emit_event(
                {
                    "type": "png_workflow_choice",
                    "path": str(target_png),
                    "name": target_png.name,
                }
            )
            return {
                "ok": True,
                "choice_required": True,
                "message": "检测到 PNG 内嵌工作流，请选择投递方式",
            }

        if not self._upload_lock.acquire(blocking=False):
            return {"ok": False, "message": "正在上传其他文件，请稍候"}

        def _worker() -> None:
            try:
                turl = (
                    base_url_override.strip().rstrip("/")
                    if base_url_override
                    else self.resolve_active_tunnel_url(allow_fast_discover=True)
                )
                last_summary = ""
                last_clip: str | None = None
                can_open_comfy = False
                uploaded_wf_names: list[str] = []
                for idx, fpath in enumerate(valid_files, start=1):
                    ext = fpath.suffix.lower()
                    self._emit_upload_progress(fpath.name, 5, f"准备({idx}/{len(valid_files)})")
                    if ext in WORKFLOW_EXTS or (ext == ".png" and png_mode in ("workflow", "canvas")):
                        msg, online_ok, safe_wf_name = self._upload_workflow_file_ex(
                            fpath, is_png_extract=(ext == ".png"), tunnel_url=turl
                        )
                        last_summary = msg
                        can_open_comfy = can_open_comfy or online_ok
                        if online_ok and safe_wf_name:
                            uploaded_wf_names.append(safe_wf_name)
                    elif ext in MEDIA_EXTS:
                        msg, clip = self._upload_media_file(fpath, tunnel_url=turl)
                        last_summary = msg
                        last_clip = clip
                    elif ext in MODEL_EXTS:
                        msg, online_ok = self._upload_model_file(fpath, tunnel_url=turl)
                        last_summary = msg
                        can_open_comfy = can_open_comfy or online_ok
                    else:
                        raise ValueError(f"暂不支持该文件格式: {fpath.name}")

                done_action: str | None = "open_comfy" if can_open_comfy else None
                if uploaded_wf_names:
                    primary_wf = uploaded_wf_names[0]
                    extra_wfs = uploaded_wf_names[1:]
                    done_action = f"open_comfy_wf:{primary_wf}"
                    if turl:
                        self._emit_upload_progress(primary_wf, 95, "直铺画布")
                        canvas_res = self._push_or_open_workflow_on_canvas(
                            turl, primary_wf, extra_names=extra_wfs
                        )
                        if canvas_res.get("pushed_live"):
                            if len(uploaded_wf_names) == 1:
                                last_summary = f"✨ 已直铺画布: {primary_wf} (点我新开)"
                            else:
                                last_summary = f"✨ 已直铺 {len(uploaded_wf_names)} 个工作流到当前画布"
                        else:
                            if len(uploaded_wf_names) == 1:
                                last_summary = f"✨ 已打开并直铺: {primary_wf}"
                            else:
                                last_summary = f"✨ 已打开并直铺 {len(uploaded_wf_names)} 个工作流"
                    elif len(valid_files) > 1:
                        last_summary = f"✅ 已完成 {len(valid_files)} 个文件投递"
                elif len(valid_files) > 1:
                    last_summary = f"✅ 已完成 {len(valid_files)} 个文件投递"

                self._emit_event(
                    {
                        "type": "upload_done",
                        "summary": last_summary,
                        "clipboard_text": last_clip,
                        "action": done_action,
                    }
                )
            except Exception as exc:
                self._emit_event(
                    {
                        "type": "upload_error",
                        "summary": f"⚠️ 投递失败: {exc}",
                    }
                )
            finally:
                self._upload_lock.release()

        threading.Thread(target=_worker, daemon=True).start()
        return {
            "ok": True,
            "started": True,
            "count": len(valid_files),
            "message": f"开始投递 {len(valid_files)} 个文件...",
        }

    # =========================================================================
    # 功能 5：Google Ultra 多账号保险箱秒切、云盘 rclone 一键绑定与环境配置
    # =========================================================================

    def start_colab_oauth(self) -> dict[str, Any]:
        """生成带账号选择器的 Google OAuth2 授权链接并在浏览器打开，PKCE 校验码仅存内存。"""
        colab_py = self._get_colab_py()
        if not colab_py.is_file():
            return {
                "ok": False,
                "message": "未检测到 colab-cli 运行环境，请先在「云盘与环境」安装或定位路径",
            }
        try:
            proc = _run_hidden([str(colab_py), "-c", _COLAB_OAUTH_START_SCRIPT], timeout=20)
            if proc.returncode != 0:
                err = (proc.stderr or proc.stdout or "").strip().splitlines()
                return {"ok": False, "message": f"生成授权链接失败: {err[-1] if err else proc.returncode}"}
            m = re.search(
                r"<<<COLAB_OAUTH_START>>>(.*?)<<<END_COLAB_OAUTH_START>>>",
                proc.stdout or "",
                re.S,
            )
            if not m:
                return {"ok": False, "message": "授权脚本返回格式异常"}
            data = json.loads(m.group(1))
            auth_url = str(data.get("auth_url") or "").strip()
            verifier = str(data.get("code_verifier") or "").strip()
            if not auth_url.startswith("https://"):
                return {"ok": False, "message": "未获取到合法 Google OAuth 链接"}
            with self._lock:
                self._pending_oauth_verifier = verifier
                self._pending_oauth_ts = time.monotonic()
            webbrowser.open(auth_url)
            snap = self._rebuild_and_emit_snapshot()
            return {
                "ok": True,
                "awaiting_code": True,
                "snapshot": snap,
                "message": "已打开 Google 账号选择与授权页，复制授权码后点击一键粘贴绑定",
            }
        except Exception as exc:
            return {"ok": False, "message": f"拉起 Google 授权失败: {exc}"}

    def cancel_colab_oauth(self) -> dict[str, Any]:
        """取消进行中的 Google OAuth2 绑定流程并清理内存临时校验码。"""
        with self._lock:
            self._pending_oauth_verifier = ""
            self._pending_oauth_ts = 0.0
        snap = self._rebuild_and_emit_snapshot()
        return {"ok": True, "snapshot": snap, "message": "已取消本次账号绑定"}

    @staticmethod
    def _extract_oauth_code(raw_input: str) -> str:
        """从纯授权码或重定向 URL 中提取纯净的 Google OAuth code。"""
        txt = str(raw_input or "").strip()
        if not txt:
            return ""
        if "code=" in txt and ("://" in txt or txt.startswith("?")):
            parsed = urllib.parse.urlparse(txt if "://" in txt else f"https://localhost/{txt.lstrip('?')}")
            qs = urllib.parse.parse_qs(parsed.query)
            if qs.get("code"):
                return str(qs["code"][0]).strip()
        return txt.splitlines()[0].strip()

    def complete_colab_oauth(self, raw_code: str, label: str = "") -> dict[str, Any]:
        """用授权码换取 Token 并安全写入系统凭据库，自动建档并切换至新绑定账号。"""
        if self.is_deploying():
            return {"ok": False, "message": "正在执行开机部署，请等待结束后再绑定新账号"}
        colab_py = self._get_colab_py()
        if not colab_py.is_file():
            return {"ok": False, "message": "未检测到 colab-cli 运行环境"}

        code = self._extract_oauth_code(raw_code)
        if len(code) < 8:
            return {"ok": False, "message": "请粘贴有效的 Google 授权码"}

        with self._lock:
            verifier = self._pending_oauth_verifier
            pending_ts = self._pending_oauth_ts
            prev_acct_id = str((self.state.get("prefs") or {}).get("active_account_id") or "").strip()

        if not verifier or (pending_ts > 0 and (time.monotonic() - pending_ts) > 1200.0):
            return {"ok": False, "message": "授权会话已超时或未启动，请重新点「新增 Google 账号」"}

        try:
            payload = json.dumps(
                {
                    "code": code,
                    "code_verifier": verifier,
                    "prev_acct_id": prev_acct_id,
                },
                ensure_ascii=False,
            )
            proc = _run_hidden(
                [str(colab_py), "-c", _COLAB_OAUTH_COMPLETE_SCRIPT],
                timeout=35,
                input_text=payload,
            )
            if proc.returncode != 0:
                err = (proc.stderr or proc.stdout or "").strip().splitlines()
                return {"ok": False, "message": f"授权换取令牌失败: {err[-1] if err else proc.returncode}"}
            m = re.search(
                r"<<<COLAB_OAUTH_DONE>>>(.*?)<<<END_COLAB_OAUTH_DONE>>>",
                proc.stdout or "",
                re.S,
            )
            if not m:
                return {"ok": False, "message": "授权写入结果解析失败"}
            info = json.loads(m.group(1))
            acct_id = re.sub(r"[^a-zA-Z0-9_-]", "", str(info.get("acct_id") or ""))[:24]
            if not acct_id:
                return {"ok": False, "message": "未能识别账号唯一标识"}
            email_masked = str(info.get("email_masked") or "***")
        except Exception as exc:
            return {"ok": False, "message": f"绑定授权码异常: {exc}"}

        # 归档旧账号账本 -> 激活新账号槽位与独立账本 -> 清空旧账号内存态会话与隧道缓存
        with self._lock:
            self._pending_oauth_verifier = ""
            self._pending_oauth_ts = 0.0
            self._auth_missing = False
            self._auth_expired = False
            if prev_acct_id and prev_acct_id != acct_id:
                self.state.setdefault("account_ledgers", {})[prev_acct_id] = self._snapshot_current_ledger()
            self._restore_ledger_for_account(acct_id)
            prefs = self.state.setdefault("prefs", dict(DEFAULT_PREFS))
            prefs["active_account_id"] = acct_id
            card = self._upsert_account_profile(
                acct_id,
                email_masked,
                preferred_label=label,
            )
            self._service_meta_by_ep.clear()
            self._runtime_secrets.clear()
            self._save_state()
        self._stop_ws()
        self._reset_comfy_offline()

        snap = self.fetch_live_status()
        bal = snap.get("balance")
        bal_txt = f"{bal:.2f} CU" if isinstance(bal, (int, float)) else "余额同步中"
        return {
            "ok": True,
            "account": card,
            "snapshot": snap,
            "message": f"✅ 已绑定 {card['label']} ({bal_txt})",
        }

    def switch_colab_account(self, acct_id: str) -> dict[str, Any]:
        """秒切到凭据保险箱中的另一组 Google Ultra 账号，完整隔离账本与 sessions.json。"""
        if self.is_deploying():
            return {"ok": False, "message": "正在执行开机部署，请等待结束后再切换账号"}
        colab_py = self._get_colab_py()
        if not colab_py.is_file():
            return {"ok": False, "message": "未检测到 colab-cli 运行环境"}

        target_id = re.sub(r"[^a-zA-Z0-9_-]", "", str(acct_id or ""))[:24]
        if not target_id:
            return {"ok": False, "message": "无效的账号 ID"}

        with self._lock:
            profiles = self.state.get("account_profiles") or []
            card = next((dict(p) for p in profiles if p.get("id") == target_id), None)
            if card is None:
                return {"ok": False, "message": "未在多账号列表中找到目标账号"}
            prev_id = str((self.state.get("prefs") or {}).get("active_account_id") or "").strip()

        if prev_id == target_id and not self._auth_missing:
            snap = self.fetch_live_status()
            bal = snap.get("balance")
            bal_txt = f"{bal:.2f} CU" if isinstance(bal, (int, float)) else "已同步"
            return {"ok": True, "snapshot": snap, "message": f"当前已是 {card['label']} ({bal_txt})"}

        if not self._action_lock.acquire(blocking=False):
            return {"ok": False, "message": "另一项操作正在执行，请稍等片刻"}
        try:
            payload = json.dumps(
                {
                    "op": "switch",
                    "target_acct_id": target_id,
                    "prev_acct_id": prev_id,
                },
                ensure_ascii=False,
            )
            proc = _run_hidden(
                [str(colab_py), "-c", _COLAB_ACCOUNT_OP_SCRIPT],
                timeout=25,
                input_text=payload,
            )
            if proc.returncode != 0:
                err = (proc.stderr or proc.stdout or "").strip().splitlines()
                return {"ok": False, "message": f"切换凭据失败: {err[-1] if err else proc.returncode}"}
        finally:
            self._action_lock.release()

        with self._lock:
            self._auth_missing = False
            self._auth_expired = False
            if prev_id and prev_id != target_id:
                self.state.setdefault("account_ledgers", {})[prev_id] = self._snapshot_current_ledger()
            self._restore_ledger_for_account(target_id)
            prefs = self.state.setdefault("prefs", dict(DEFAULT_PREFS))
            prefs["active_account_id"] = target_id
            self._service_meta_by_ep.clear()
            self._runtime_secrets.clear()
            self._save_state()
        self._stop_ws()
        self._reset_comfy_offline()

        snap = self.fetch_live_status()
        bal = snap.get("balance")
        bal_txt = f"{bal:.2f} CU" if isinstance(bal, (int, float)) else "同步中"
        return {
            "ok": True,
            "snapshot": snap,
            "message": f"⚡ 已秒切至 {card['label']} ({bal_txt})",
        }

    def rename_colab_account(self, acct_id: str, label: str) -> dict[str, Any]:
        """修改指定已绑定 Google 账号的显示备注（如 Ultra 主号 / 备用号）。"""
        target_id = re.sub(r"[^a-zA-Z0-9_-]", "", str(acct_id or ""))[:24]
        clean_label = _sanitize_account_label(label, "")
        if not target_id or not clean_label:
            return {"ok": False, "message": "请输入非空账号备注名称"}
        with self._lock:
            profiles = self.state.get("account_profiles") or []
            found = next((p for p in profiles if isinstance(p, dict) and p.get("id") == target_id), None)
            if not found:
                return {"ok": False, "message": "未找到目标账号"}
            found["label"] = clean_label
            self._save_state()
        snap = self._rebuild_and_emit_snapshot()
        return {"ok": True, "snapshot": snap, "message": f"已重命名为「{clean_label}」"}

    def remove_colab_account(self, acct_id: str) -> dict[str, Any]:
        """从保险箱与系统凭据库移除指定账号；若当前激活且仍有余号则自动切换过去。"""
        if self.is_deploying():
            return {"ok": False, "message": "正在执行开机部署，请先等待部署完成"}
        target_id = re.sub(r"[^a-zA-Z0-9_-]", "", str(acct_id or ""))[:24]
        if not target_id:
            return {"ok": False, "message": "无效的账号 ID"}

        with self._lock:
            profiles = [dict(p) for p in (self.state.get("account_profiles") or []) if isinstance(p, dict)]
            removed_card = next((p for p in profiles if p.get("id") == target_id), None)
            if removed_card is None:
                return {"ok": False, "message": "目标账号不存在"}
            remaining = [p for p in profiles if p.get("id") != target_id]
            active_id = str((self.state.get("prefs") or {}).get("active_account_id") or "").strip()
            removing_active = active_id == target_id or not remaining

        colab_py = self._get_colab_py()
        if colab_py.is_file():
            payload = json.dumps(
                {
                    "op": "remove",
                    "target_acct_id": target_id,
                    "clear_active": removing_active and (len(remaining) == 0),
                },
                ensure_ascii=False,
            )
            try:
                _run_hidden([str(colab_py), "-c", _COLAB_ACCOUNT_OP_SCRIPT], timeout=20, input_text=payload)
            except Exception:
                pass

        with self._lock:
            self.state["account_profiles"] = remaining
            ledgers = self.state.get("account_ledgers")
            if isinstance(ledgers, dict):
                ledgers.pop(target_id, None)
            if removing_active and not remaining:
                self.state.setdefault("prefs", dict(DEFAULT_PREFS))["active_account_id"] = ""
                self._restore_ledger_for_account("")
                self._auth_missing = True
                self._auth_expired = False
                self._service_meta_by_ep.clear()
                self._runtime_secrets.clear()
            self._save_state()

        if removing_active and remaining:
            next_id = str(remaining[0]["id"])
            res = self.switch_colab_account(next_id)
            if res.get("ok"):
                res["message"] = f"已移除「{removed_card['label']}」，自动切至「{remaining[0]['label']}」"
            return res

        if removing_active and not remaining:
            self._stop_ws()
            self._reset_comfy_offline()
            snap = self._rebuild_and_emit_snapshot(stale=True, error_msg="尚未绑定 Google Colab 账号")
            return {"ok": True, "snapshot": snap, "message": f"已解绑「{removed_card['label']}」"}

        snap = self._rebuild_and_emit_snapshot()
        return {"ok": True, "snapshot": snap, "message": f"已移除「{removed_card['label']}」"}

    def _query_gdrive_quota_locked(self) -> tuple[float | None, float | None]:
        """调用 rclone about 查询 gdrive_save 剩余与总容量（GiB），失败返回 (None, None)。"""
        rclone_exe = self._get_rclone_exe()
        rclone_conf = self._get_rclone_conf()
        if not rclone_exe.is_file() or not self._has_gdrive_save_section(rclone_conf):
            return None, None
        try:
            proc = _run_hidden(
                [str(rclone_exe), "about", "gdrive_save:", "--json", "--config", str(rclone_conf)],
                timeout=15,
            )
            if proc.returncode != 0:
                return None, None
            info = json.loads(proc.stdout or "{}")
            free_b = info.get("free")
            total_b = info.get("total")
            free_gb = round(float(free_b) / (1024 ** 3), 1) if isinstance(free_b, (int, float)) else None
            total_gb = round(float(total_b) / (1024 ** 3), 1) if isinstance(total_b, (int, float)) else None
            with self._lock:
                self.state["onboarding_cache"] = {
                    "gdrive_free_gb": free_gb,
                    "gdrive_total_gb": total_gb,
                    "gdrive_checked_ts": int(time.time()),
                }
                self._save_state()
            return free_gb, total_gb
        except Exception:
            return None, None

    def probe_environment_deep(self) -> dict[str, Any]:
        """执行一次完整的本地工具、Google Drive 剩余容量与 Colab 在线健康体检。"""
        free_gb, _ = self._query_gdrive_quota_locked()
        snap = self.fetch_live_status()
        ob = snap.get("onboarding") or {}
        if ob.get("needs_attention"):
            if not ob.get("colab_cli_ok") or not ob.get("rclone_exe_ok"):
                msg = "🟠 缺少本地工具，请安装或定位路径"
            elif not ob.get("colab_auth_ok"):
                msg = "🟠 请绑定 Google Colab 账号"
            else:
                msg = "🟠 请授权绑定 Google Drive 云盘"
        elif free_gb is not None:
            msg = f"🟢 环境全部就绪 · 云盘剩余 {free_gb:.1f} GB"
        else:
            msg = "🟢 账号与环境全部就绪"
        return {"ok": True, "snapshot": snap, "message": msg}

    def start_gdrive_bind(self) -> dict[str, Any]:
        """在后台无黑框拉起 rclone 浏览器一键授权并生成 [gdrive_save] 云盘配置。"""
        rclone_exe = self._get_rclone_exe()
        if not rclone_exe.is_file():
            return {"ok": False, "message": "未找到 rclone.exe，请先安装或定位路径"}
        if self._gdrive_bind_proc is not None:
            return {"ok": False, "message": "正在浏览器等待 Google 云盘授权，请点击「允许」"}

        rclone_conf = self._get_rclone_conf()
        if not str(rclone_conf):
            appdata = os.environ.get("APPDATA", "")
            rclone_conf = (Path(appdata) / "rclone" / "rclone.conf") if appdata else (Path.home() / ".config" / "rclone" / "rclone.conf")
        try:
            rclone_conf.parent.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            return {"ok": False, "message": f"无法创建 rclone 配置目录: {exc}"}

        self._gdrive_bind_cancelled = False
        cmd = [
            str(rclone_exe),
            "config",
            "create",
            "gdrive_save",
            "drive",
            "scope=drive",
            "config_is_local=true",
            "--config",
            str(rclone_conf),
            "--non-interactive",
        ]
        # 兼容性说明：部分 rclone 版本在非交互模式下触发自动本地回调需去掉 --non-interactive，走默认确认即可自动完成
        cmd = [c for c in cmd if c != "--non-interactive"]

        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=_clean_env(),
                creationflags=CREATE_NO_WINDOW,
            )
            self._gdrive_bind_proc = proc
        except Exception as exc:
            self._gdrive_bind_proc = None
            return {"ok": False, "message": f"拉起 rclone 授权失败: {exc}"}

        snap_start = self._rebuild_and_emit_snapshot()

        def _wait_worker() -> None:
            rc = 1
            out_txt = ""
            try:
                out_txt, _ = proc.communicate(timeout=180)
                rc = proc.returncode
            except subprocess.TimeoutExpired:
                self._kill_process_tree(proc)
                out_txt = "浏览器授权等待超时（3 分钟）"
            except Exception as exc:
                out_txt = str(exc)
            finally:
                self._gdrive_bind_proc = None

            if self._gdrive_bind_cancelled:
                snap = self._rebuild_and_emit_snapshot()
                self._emit_event(
                    {
                        "type": "onboarding_event",
                        "ok": False,
                        "summary": "已取消 Google 云盘授权",
                        "snapshot": snap,
                    }
                )
                return

            if rc == 0 and self._has_gdrive_save_section(rclone_conf):
                with self._lock:
                    self.state.setdefault("prefs", dict(DEFAULT_PREFS))["rclone_conf_path"] = str(rclone_conf)
                    self._save_state()
                free_gb, _ = self._query_gdrive_quota_locked()
                snap = self._rebuild_and_emit_snapshot()
                free_txt = f"（余 {free_gb:.1f}G）" if free_gb is not None else ""
                self._emit_event(
                    {
                        "type": "onboarding_event",
                        "ok": True,
                        "summary": f"✅ Google 云盘 gdrive_save 已绑定{free_txt}",
                        "snapshot": snap,
                    }
                )
            else:
                tail = (out_txt.strip().splitlines() or [f"退出码 {rc}"])[-1]
                snap = self._rebuild_and_emit_snapshot()
                self._emit_event(
                    {
                        "type": "onboarding_event",
                        "ok": False,
                        "summary": f"⚠️ 云盘授权未完成: {tail[:42]}",
                        "snapshot": snap,
                    }
                )

        threading.Thread(target=_wait_worker, daemon=True).start()
        return {
            "ok": True,
            "binding": True,
            "snapshot": snap_start,
            "message": "已打开浏览器，请登录存储盘账号并点「允许」",
        }

    def cancel_gdrive_bind(self) -> dict[str, Any]:
        """终止等待中的 rclone Google Drive 浏览器授权进程。"""
        self._gdrive_bind_cancelled = True
        proc = self._gdrive_bind_proc
        self._gdrive_bind_proc = None
        if proc is not None:
            self._kill_process_tree(proc)
        snap = self._rebuild_and_emit_snapshot()
        return {"ok": True, "snapshot": snap, "message": "已中止等待云盘授权"}

    def import_rclone_conf(self, src_path: str) -> dict[str, Any]:
        """导入用户已有的 rclone.conf 并校验/适配 [gdrive_save] 云盘配置。"""
        p = Path(str(src_path or "").strip())
        if not p.is_file():
            return {"ok": False, "message": "所选 rclone.conf 文件不存在"}
        try:
            content = p.read_text(encoding="utf-8", errors="ignore")
        except Exception as exc:
            return {"ok": False, "message": f"读取配置文件失败: {exc}"}

        if not _RCLONE_GDRIVE_SECTION_RE.search(content):
            sections = _RCLONE_INI_SECTION_RE.findall(content)
            if len(sections) == 1 and re.search(r"^\s*type\s*=\s*drive\b", content, re.MULTILINE):
                # 自动将单远端 Google Drive 配置别名复制到本地 rclone.conf 并命名为 gdrive_save
                appdata = os.environ.get("APPDATA", "")
                dst_conf = (Path(appdata) / "rclone" / "rclone.conf") if appdata else (PACKAGE_DIR / "rclone.conf")
                dst_conf.parent.mkdir(parents=True, exist_ok=True)
                rewritten = _RCLONE_INI_SECTION_RE.sub("[gdrive_save]", content, count=1)
                fd = os.open(str(dst_conf), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
                with os.fdopen(fd, "w", encoding="utf-8") as fp:
                    fp.write(rewritten)
                p = dst_conf
            else:
                return {
                    "ok": False,
                    "message": "该 rclone.conf 缺少 [gdrive_save] 远端，建议直接点「一键绑定云盘」",
                }

        with self._lock:
            self.state.setdefault("prefs", dict(DEFAULT_PREFS))["rclone_conf_path"] = str(p)
            self._save_state()
        free_gb, _ = self._query_gdrive_quota_locked()
        snap = self._rebuild_and_emit_snapshot()
        free_txt = f" · 余 {free_gb:.1f}G" if free_gb is not None else ""
        return {"ok": True, "snapshot": snap, "message": f"✅ 已接入云盘配置{free_txt}"}

    def set_tool_path(self, kind: str, path_str: str) -> dict[str, Any]:
        """设置或重置 colab.exe / python.exe / rclone.exe / rclone.conf 路径偏好。"""
        key_map = {
            "colab_exe": "colab_exe_path",
            "colab_py": "colab_py_path",
            "rclone_exe": "rclone_exe_path",
            "rclone_conf": "rclone_conf_path",
        }
        pref_key = key_map.get(str(kind or "").strip().lower())
        if not pref_key:
            return {"ok": False, "message": "未知工具类型"}

        val = str(path_str or "").strip()
        if val and not Path(val).is_file():
            return {"ok": False, "message": f"所选路径文件不存在: {val}"}

        if pref_key == "rclone_conf_path" and val:
            return self.import_rclone_conf(val)

        with self._lock:
            self.state.setdefault("prefs", dict(DEFAULT_PREFS))[pref_key] = val
            self._save_state()
        snap = self._rebuild_and_emit_snapshot()
        return {
            "ok": True,
            "snapshot": snap,
            "message": "✅ 已更新工具路径" if val else "已恢复自动探测路径",
        }

    def reset_tool_paths(self) -> dict[str, Any]:
        """清除所有手动指定的工具路径，恢复跨设备自动探测。"""
        with self._lock:
            prefs = self.state.setdefault("prefs", dict(DEFAULT_PREFS))
            for k in ("colab_exe_path", "colab_py_path", "rclone_exe_path", "rclone_conf_path"):
                prefs[k] = ""
            self._save_state()
        snap = self._rebuild_and_emit_snapshot()
        return {"ok": True, "snapshot": snap, "message": "已恢复全部工具自动探测"}

    def save_cf_tunnel_config(self, config: dict[str, Any] | None = None) -> dict[str, Any]:
        """单向安全写入 Cloudflare 固定命名隧道与域名配置（0o600 落盘，绝不向前端回显密钥）。"""
        cfg_in = config if isinstance(config, dict) else {}
        try:
            comfy_url = _extract_valid_https_url(str(cfg_in.get("comfy_url") or ""))
            llm_url = _extract_valid_https_url(str(cfg_in.get("llm_url") or ""))
        except ValueError as exc:
            return {"ok": False, "message": str(exc)}

        existing = _load_cf_secret_raw()
        new_tok = str(cfg_in.get("tunnel_token") or "").strip()
        tok = new_tok if new_tok else str(existing.get("tunnel_token") or "").strip()

        new_key = str(cfg_in.get("llm_api_key") or "").strip()
        llm_key = new_key if new_key else str(existing.get("llm_api_key") or "").strip()

        if not tok:
            return {"ok": False, "message": "首次启用固定域名请填写 Cloudflare Tunnel Token"}
        if not (comfy_url or llm_url):
            return {"ok": False, "message": "请至少填写一个 https:// 公网域名"}

        out_obj: dict[str, str] = {"tunnel_token": tok}
        if comfy_url:
            out_obj["comfy_url"] = comfy_url
        if llm_url:
            out_obj["llm_url"] = llm_url
        if llm_key:
            out_obj["llm_api_key"] = llm_key

        tmp_p = _CF_SECRET_FILE.with_suffix(".json.tmp")
        fd = os.open(str(tmp_p), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fp:
            json.dump(out_obj, fp, ensure_ascii=False, indent=2)
            fp.write("\n")
        try:
            os.chmod(str(tmp_p), 0o600)
        except OSError:
            pass
        tmp_p.replace(_CF_SECRET_FILE)
        try:
            os.chmod(str(_CF_SECRET_FILE), 0o600)
        except OSError:
            pass

        snap = self._rebuild_and_emit_snapshot()
        return {"ok": True, "snapshot": snap, "message": "✅ 已启用 Cloudflare 固定域名隧道"}

    def clear_cf_tunnel_config(self) -> dict[str, Any]:
        """清除固定命名隧道配置，切回免费即用的 trycloudflare 临时隧道。"""
        try:
            _CF_SECRET_FILE.unlink(missing_ok=True)
        except OSError as exc:
            return {"ok": False, "message": f"切回失败: {exc}"}
        snap = self._rebuild_and_emit_snapshot()
        return {"ok": True, "snapshot": snap, "message": "已切回免费 trycloudflare 临时隧道"}

    def open_local_outputs_dir(self) -> dict[str, Any]:
        """无需等待远端连接，直接创建并打开本机成图产出文件夹。"""
        self.download_root.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(str(self.download_root))
        except Exception as exc:
            return {"ok": False, "message": f"打开目录失败: {exc}"}
        return {"ok": True, "path": str(self.download_root), "message": "📂 已打开本地成图夹"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Colab 桌宠后端状态探针")
    parser.add_argument("--once", action="store_true", help="采集一次并打印 JSON 快照")
    args = parser.parse_args()

    backend = ColabPetBackend()
    if args.once:
        snap = backend.fetch_live_status()
        print(json.dumps(snap, ensure_ascii=False, indent=2))
        return 0 if snap.get("ok") else 1

    print(json.dumps(backend.get_cached_snapshot(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
