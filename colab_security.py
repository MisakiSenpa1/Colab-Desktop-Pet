"""colab_security.py —— 共享安全底座：系统凭据保险箱、公网票据鉴权与日志脱敏过滤。

设计要点：
1. 系统凭据保险箱（VaultStore）：
   - 若环境变量 `COLAB_PET_MOCK_VAULT_PATH` 存在，则在沙箱 JSON 文件（0o600）中读写，
     用于离线测试隔离，绝不读写或篡改宿主机真实凭据库；
   - Windows 下直接通过 `advapi32.dll` 原生 CredReadW / CredWriteW / CredDeleteW 接入系统凭据管理器；
   - 其他平台回退 `keyring` 库或指定 `colab_py` 解释器子进程（通过 stdin 传密）。
2. Cloudflare 隧道配置：
   - 配置文件 `.cf_tunnel_secret.json` 仅保存公开域名与 `vault:` 引用字段，不存明文密钥；
   - 自动检测并平滑迁移旧版明文配置：写入系统凭据库并回读校验无误后，才擦除磁盘明文。
3. ComfyUI 公网认证票据与敏感日志脱敏：
   - 提供一次性的站内跳转签名票据生成与 `sanitize_next_path` 开放重定向防护；
   - 提供统一 `redact_secrets` 脱敏器，屏蔽 Bearer Key、OAuth 授权码、CF Tunnel Token、代理令牌与票据。
"""
from __future__ import annotations

import ctypes
import hashlib
import hmac
import json
import os
import re
import secrets
import subprocess
import threading
import time
import urllib.parse
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable

VAULT_SERVICE_NAME = "google-colab-cli-pet-vault"
CF_VAULT_TUNNEL_TOKEN_KEY = "secret:cf_tunnel_token"
CF_VAULT_LLM_API_KEY = "secret:cf_llm_api_key"

_CUSTOM_TUNNEL_HOST_RE = re.compile(r"^[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?$", re.IGNORECASE)
_MOCK_VAULT_LOCK = threading.RLock()
_EPHEMERAL_LOCK = threading.Lock()
_EPHEMERAL_SECRETS: OrderedDict[str, None] = OrderedDict()
_MAX_EPHEMERAL_SECRETS = 128

_CRED_TYPE_GENERIC = 1
_CRED_PERSIST_LOCAL_MACHINE = 2
_ERROR_NOT_FOUND = 1168


def _write_private_text(path_obj: Path, content: str) -> None:
    """以 0o600 私有权限原子写入文本文件，避免中途损坏或权限过宽。"""
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    tmp_p = path_obj.with_name(f".{path_obj.name}.{os.getpid()}.tmp")
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


def register_ephemeral_secret(secret: str | None) -> None:
    """将运行时密钥或一次性口令加入有界脱敏注册表。"""
    val = str(secret or "").strip()
    if len(val) < 8:
        return
    with _EPHEMERAL_LOCK:
        _EPHEMERAL_SECRETS[val] = None
        _EPHEMERAL_SECRETS.move_to_end(val)
        while len(_EPHEMERAL_SECRETS) > _MAX_EPHEMERAL_SECRETS:
            _EPHEMERAL_SECRETS.popitem(last=False)


def clear_ephemeral_secrets() -> None:
    """清空内存态脱敏注册表（主要用于单元测试隔离）。"""
    with _EPHEMERAL_LOCK:
        _EPHEMERAL_SECRETS.clear()


_RE_BEARER = re.compile(r"(Authorization\s*:\s*Bearer\s+)[A-Za-z0-9._~+/=-]{8,}", re.IGNORECASE)
_RE_PREFIXED_KEY = re.compile(r"\b(?:sk|cpk)-[A-Za-z0-9_-]{12,}\b")
_RE_OAUTH_TOK = re.compile(
    r"(?<![A-Za-z0-9_-])(?:4/[0-9A-Za-z_-]{16,}|ya29\.[0-9A-Za-z._-]{16,}|1//[0-9A-Za-z_-]{16,})(?![A-Za-z0-9_-])"
)
_RE_JWT_TOK = re.compile(r"\beyJ[A-Za-z0-9_-]{12,}\.[A-Za-z0-9_-]{8,}(?:\.[A-Za-z0-9_-]*)?")
_RE_CF_CLI_TOK = re.compile(
    r"((?:TUNNEL_TOKEN|CF_TUNNEL_TOKEN)\s*=\s*|CF_TUNNEL_TOKEN['\"]?\s*\]?\s*=\s*|CF_TUNNEL_TOKEN\s*,\s*|--token(?:=|\s+))(['\"]?)[^'\"\s,;)]+(\2)",
    re.IGNORECASE,
)
_RE_PROXY_OR_TICKET = re.compile(
    r"((?:colab-runtime-proxy-token|ticket)['\"]?\s*[:=]\s*|(?<=[?&])(?:colab-runtime-proxy-token|ticket|code)=)([^&\s'\"#,)}]+)",
    re.IGNORECASE,
)
_RE_JSON_SECRET = re.compile(
    r'("(?:access_token|refresh_token|client_secret|tunnel_token|llm_api_key|comfy_api_key|api_key|code_verifier)"\s*:\s*")[^"]+(")',
    re.IGNORECASE,
)


def redact_secrets(text: str | None, extra_secrets: Iterable[str] = ()) -> str:
    """统一过滤日志、异常信息与提示弹窗中的密钥、OAuth 授权码、隧道令牌和票据。"""
    if text is None:
        return ""
    out = str(text)
    if not out:
        return ""

    candidates: set[str] = set()
    with _EPHEMERAL_LOCK:
        for k in _EPHEMERAL_SECRETS:
            candidates.add(k)
    for item in extra_secrets:
        s = str(item or "").strip()
        if len(s) >= 8:
            candidates.add(s)

    # 优先按长度倒序精确替换已知密钥，避免短子串截断替换后残留尾缀
    for secret_val in sorted(candidates, key=len, reverse=True):
        if secret_val in out:
            out = out.replace(secret_val, "[REDACTED]")

    out = _RE_BEARER.sub(r"\1[REDACTED]", out)
    out = _RE_PREFIXED_KEY.sub("[REDACTED_KEY]", out)
    out = _RE_OAUTH_TOK.sub("[REDACTED_OAUTH]", out)
    out = _RE_JWT_TOK.sub("[REDACTED_CF_TOKEN]", out)
    out = _RE_CF_CLI_TOK.sub(r"\1\2[REDACTED_CF_TOKEN]\3", out)
    out = _RE_PROXY_OR_TICKET.sub(r"\1[REDACTED]", out)
    out = _RE_JSON_SECRET.sub(r"\1[REDACTED]\2", out)
    return out


# =========================================================================
# Windows 原生凭据管理器 (advapi32.dll) 与多后端凭据保险箱 VaultStore
# =========================================================================

if os.name == "nt":
    from ctypes import wintypes

    class _FILETIME(ctypes.Structure):
        _fields_ = [
            ("dwLowDateTime", wintypes.DWORD),
            ("dwHighDateTime", wintypes.DWORD),
        ]

    class _CREDENTIALW(ctypes.Structure):
        _fields_ = [
            ("Flags", wintypes.DWORD),
            ("Type", wintypes.DWORD),
            ("TargetName", wintypes.LPWSTR),
            ("Comment", wintypes.LPWSTR),
            ("LastWritten", _FILETIME),
            ("CredentialBlobSize", wintypes.DWORD),
            ("CredentialBlob", ctypes.POINTER(ctypes.c_ubyte)),
            ("Persist", wintypes.DWORD),
            ("AttributeCount", wintypes.DWORD),
            ("Attributes", ctypes.c_void_p),
            ("TargetAlias", wintypes.LPWSTR),
            ("UserName", wintypes.LPWSTR),
        ]

    _PCREDENTIALW = ctypes.POINTER(_CREDENTIALW)
    _ADVAPI32 = ctypes.WinDLL("advapi32", use_last_error=True)
    _ADVAPI32.CredReadW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.POINTER(_PCREDENTIALW),
    ]
    _ADVAPI32.CredReadW.restype = wintypes.BOOL
    _ADVAPI32.CredWriteW.argtypes = [_PCREDENTIALW, wintypes.DWORD]
    _ADVAPI32.CredWriteW.restype = wintypes.BOOL
    _ADVAPI32.CredDeleteW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD]
    _ADVAPI32.CredDeleteW.restype = wintypes.BOOL
    _ADVAPI32.CredFree.argtypes = [ctypes.c_void_p]
    _ADVAPI32.CredFree.restype = None


def _win_cred_read(target_name: str) -> str | None:
    pcred = _PCREDENTIALW()
    ok = _ADVAPI32.CredReadW(target_name, _CRED_TYPE_GENERIC, 0, ctypes.byref(pcred))
    if not ok:
        err = ctypes.get_last_error()
        if err == _ERROR_NOT_FOUND:
            return None
        raise OSError(err, f"CredReadW 失败 (code={err}): {ctypes.FormatError(err)}")
    try:
        size = int(pcred.contents.CredentialBlobSize)
        if size <= 0 or not pcred.contents.CredentialBlob:
            return ""
        raw = ctypes.string_at(pcred.contents.CredentialBlob, size)
        return raw.decode("utf-8", errors="strict")
    finally:
        _ADVAPI32.CredFree(pcred)


def _win_cred_write(target_name: str, user_name: str, secret_val: str) -> None:
    raw = str(secret_val).encode("utf-8")
    blob_arr = (ctypes.c_ubyte * len(raw)).from_buffer_copy(raw) if raw else None
    cred = _CREDENTIALW()
    cred.Flags = 0
    cred.Type = _CRED_TYPE_GENERIC
    cred.TargetName = target_name
    cred.Comment = "ColabPet 凭据保险箱"
    cred.CredentialBlobSize = len(raw)
    cred.CredentialBlob = ctypes.cast(blob_arr, ctypes.POINTER(ctypes.c_ubyte)) if blob_arr else None
    cred.Persist = _CRED_PERSIST_LOCAL_MACHINE
    cred.AttributeCount = 0
    cred.Attributes = None
    cred.TargetAlias = None
    cred.UserName = user_name
    ok = _ADVAPI32.CredWriteW(ctypes.byref(cred), 0)
    if not ok:
        err = ctypes.get_last_error()
        raise OSError(err, f"CredWriteW 失败 (code={err}): {ctypes.FormatError(err)}")


def _win_cred_delete(target_name: str) -> bool:
    ok = _ADVAPI32.CredDeleteW(target_name, _CRED_TYPE_GENERIC, 0)
    if not ok:
        err = ctypes.get_last_error()
        if err == _ERROR_NOT_FOUND:
            return False
        raise OSError(err, f"CredDeleteW 失败 (code={err}): {ctypes.FormatError(err)}")
    return True


class VaultStore:
    """跨环境凭据保险箱：优先使用测试沙箱文件，其次 Windows 原生凭据库或 keyring。"""

    def __init__(self, service_name: str = VAULT_SERVICE_NAME) -> None:
        self.service_name = str(service_name or VAULT_SERVICE_NAME)

    @staticmethod
    def _mock_file() -> Path | None:
        raw = str(os.environ.get("COLAB_PET_MOCK_VAULT_PATH") or "").strip()
        return Path(raw) if raw else None

    def _target_name(self, name: str) -> str:
        return f"{self.service_name}:{name}"

    def _read_mock(self, path_obj: Path) -> dict[str, str]:
        if not path_obj.is_file():
            return {}
        try:
            data = json.loads(path_obj.read_text(encoding="utf-8"))
            return {str(k): str(v) for k, v in data.items()} if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _write_mock(self, path_obj: Path, data: dict[str, str]) -> None:
        _write_private_text(path_obj, json.dumps(data, ensure_ascii=False, indent=2) + "\n")

    def get_secret(self, name: str, colab_py: Path | None = None) -> str | None:
        """从保险箱读取指定键名密钥；不存在时返回 None。"""
        clean_name = str(name or "").strip()
        if not clean_name:
            return None
        mock_p = self._mock_file()
        if mock_p is not None:
            with _MOCK_VAULT_LOCK:
                return self._read_mock(mock_p).get(clean_name)

        if os.name == "nt":
            return _win_cred_read(self._target_name(clean_name))

        try:
            import keyring  # type: ignore[import-not-found]

            return keyring.get_password(self.service_name, clean_name)
        except Exception:
            if colab_py is not None and Path(colab_py).is_file():
                return self._subprocess_keyring("get", clean_name, None, Path(colab_py))
            raise

    def set_secret(self, name: str, value: str, colab_py: Path | None = None) -> None:
        """将密钥安全存入保险箱。"""
        clean_name = str(name or "").strip()
        if not clean_name:
            raise ValueError("凭据键名不能为空")
        secret_val = str(value or "")
        mock_p = self._mock_file()
        if mock_p is not None:
            with _MOCK_VAULT_LOCK:
                data = self._read_mock(mock_p)
                data[clean_name] = secret_val
                self._write_mock(mock_p, data)
            return

        if os.name == "nt":
            _win_cred_write(self._target_name(clean_name), clean_name, secret_val)
            return

        try:
            import keyring  # type: ignore[import-not-found]

            keyring.set_password(self.service_name, clean_name, secret_val)
        except Exception:
            if colab_py is not None and Path(colab_py).is_file():
                self._subprocess_keyring("set", clean_name, secret_val, Path(colab_py))
                return
            raise

    def delete_secret(self, name: str, colab_py: Path | None = None) -> bool:
        """从保险箱删除指定密钥，不存在返回 False。"""
        clean_name = str(name or "").strip()
        if not clean_name:
            return False
        mock_p = self._mock_file()
        if mock_p is not None:
            with _MOCK_VAULT_LOCK:
                data = self._read_mock(mock_p)
                existed = clean_name in data
                if existed:
                    data.pop(clean_name, None)
                    self._write_mock(mock_p, data)
                return existed

        if os.name == "nt":
            return _win_cred_delete(self._target_name(clean_name))

        try:
            import keyring  # type: ignore[import-not-found]

            try:
                keyring.delete_password(self.service_name, clean_name)
                return True
            except Exception:
                return False
        except Exception:
            if colab_py is not None and Path(colab_py).is_file():
                res = self._subprocess_keyring("delete", clean_name, None, Path(colab_py))
                return res == "1"
            return False

    def _subprocess_keyring(
        self, op: str, name: str, value: str | None, colab_py: Path
    ) -> str | None:
        code = (
            "import json, sys, keyring\n"
            "req = json.loads(sys.stdin.read() or '{}')\n"
            "op, svc, key = req['op'], req['svc'], req['key']\n"
            "if op == 'get':\n"
            "    v = keyring.get_password(svc, key)\n"
            "    print(json.dumps({'val': v}))\n"
            "elif op == 'set':\n"
            "    keyring.set_password(svc, key, req['val'])\n"
            "    print(json.dumps({'ok': True}))\n"
            "elif op == 'delete':\n"
            "    try:\n"
            "        keyring.delete_password(svc, key)\n"
            "        print(json.dumps({'val': '1'}))\n"
            "    except Exception:\n"
            "        print(json.dumps({'val': '0'}))\n"
        )
        payload = json.dumps(
            {"op": op, "svc": self.service_name, "key": name, "val": value},
            ensure_ascii=False,
        )
        proc = subprocess.run(
            [str(colab_py), "-c", code],
            input=payload,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if proc.returncode != 0:
            raise RuntimeError(redact_secrets(proc.stderr or proc.stdout or "keyring 子进程异常"))
        out = json.loads((proc.stdout or "{}").strip().splitlines()[-1])
        return out.get("val")


# =========================================================================
# Cloudflare 固定隧道与固定密钥的仅引用配置管理
# =========================================================================


def extract_valid_https_url(raw_url: str | None) -> str:
    """校验并返回标准化的 https://<domain>；空串返回空串，非法格式抛 ValueError。"""
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


def _vault_key_from_ref(ref_val: Any, default_key: str) -> str:
    raw = str(ref_val or "").strip()
    if raw.startswith("vault:"):
        key = raw[6:].strip()
        if key:
            return key
    return default_key


def load_cf_tunnel_settings(
    config_path: Path | None = None,
    *,
    include_secrets: bool = True,
    colab_py: Path | None = None,
    vault: VaultStore | None = None,
) -> dict[str, Any]:
    """加载 Cloudflare 固定隧道配置；发现旧版明文字段时自动安全迁移至系统凭据库并抹除明文。"""
    cfg_path = Path(config_path) if config_path is not None else Path(
        os.environ.get("COLAB_PET_CF_CONFIG_PATH") or ""
    )
    if not str(cfg_path) or not cfg_path.is_file():
        out: dict[str, Any] = {
            "mode": "quick",
            "has_token": False,
            "has_fixed_llm_key": False,
            "comfy_url": "",
            "llm_url": "",
        }
        if include_secrets:
            out["tunnel_token"] = ""
            out["llm_api_key"] = ""
        return out

    try:
        raw_obj = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        raw_obj = {}
    if not isinstance(raw_obj, dict):
        raw_obj = {}

    store = vault or VaultStore()
    migrated = False
    clean_disk: dict[str, Any] = {}

    for url_key in ("comfy_url", "llm_url"):
        val = str(raw_obj.get(url_key) or "").strip().rstrip("/")
        if val:
            clean_disk[url_key] = val

    tok_ref_key = _vault_key_from_ref(raw_obj.get("tunnel_token_ref"), CF_VAULT_TUNNEL_TOKEN_KEY)
    key_ref_key = _vault_key_from_ref(raw_obj.get("llm_api_key_ref"), CF_VAULT_LLM_API_KEY)

    legacy_tok = str(raw_obj.get("tunnel_token") or "").strip()
    if "tunnel_token" in raw_obj:
        migrated = True
    if legacy_tok:
        store.set_secret(tok_ref_key, legacy_tok, colab_py=colab_py)
        verified_tok = store.get_secret(tok_ref_key, colab_py=colab_py)
        if verified_tok != legacy_tok:
            raise RuntimeError("系统凭据库回读校验失败，已中止擦除原隧道 Token")
        clean_disk["tunnel_token_ref"] = f"vault:{tok_ref_key}"
    elif raw_obj.get("tunnel_token_ref"):
        clean_disk["tunnel_token_ref"] = f"vault:{tok_ref_key}"

    legacy_llm_key = str(raw_obj.get("llm_api_key") or "").strip()
    if "llm_api_key" in raw_obj:
        migrated = True
    if legacy_llm_key:
        store.set_secret(key_ref_key, legacy_llm_key, colab_py=colab_py)
        verified_key = store.get_secret(key_ref_key, colab_py=colab_py)
        if verified_key != legacy_llm_key:
            raise RuntimeError("系统凭据库回读校验失败，已中止擦除原固定 LLM Key")
        clean_disk["llm_api_key_ref"] = f"vault:{key_ref_key}"
    elif raw_obj.get("llm_api_key_ref"):
        clean_disk["llm_api_key_ref"] = f"vault:{key_ref_key}"

    if migrated:
        _write_private_text(cfg_path, json.dumps(clean_disk, ensure_ascii=False, indent=2) + "\n")

    tok_val = ""
    if clean_disk.get("tunnel_token_ref"):
        try:
            tok_val = str(store.get_secret(tok_ref_key, colab_py=colab_py) or "").strip()
        except Exception:
            tok_val = ""

    llm_key_val = ""
    if clean_disk.get("llm_api_key_ref"):
        try:
            llm_key_val = str(store.get_secret(key_ref_key, colab_py=colab_py) or "").strip()
        except Exception:
            llm_key_val = ""

    if tok_val:
        register_ephemeral_secret(tok_val)
    if llm_key_val:
        register_ephemeral_secret(llm_key_val)

    comfy_u = str(clean_disk.get("comfy_url") or "")
    llm_u = str(clean_disk.get("llm_url") or "")
    has_tok = bool(tok_val)
    has_fixed_key = len(llm_key_val) >= 16

    result: dict[str, Any] = {
        "mode": "named" if has_tok else "quick",
        "has_token": has_tok,
        "has_fixed_llm_key": has_fixed_key,
        "comfy_url": comfy_u,
        "llm_url": llm_u,
    }
    if include_secrets:
        result["tunnel_token"] = tok_val
        result["llm_api_key"] = llm_key_val
    return result


def save_cf_tunnel_settings(
    config_path: Path,
    *,
    comfy_url: str = "",
    llm_url: str = "",
    tunnel_token: str = "",
    llm_api_key: str = "",
    colab_py: Path | None = None,
    vault: VaultStore | None = None,
) -> dict[str, Any]:
    """将固定隧道密钥存入系统凭据库，配置文件仅写入域名与 `vault:` 引用。"""
    clean_comfy = extract_valid_https_url(comfy_url)
    clean_llm = extract_valid_https_url(llm_url)
    if not (clean_comfy or clean_llm):
        raise ValueError("请至少填写一个 https:// 公网域名")

    store = vault or VaultStore()
    existing = load_cf_tunnel_settings(
        config_path, include_secrets=True, colab_py=colab_py, vault=store
    )
    new_tok = str(tunnel_token or "").strip()
    tok = new_tok if new_tok else str(existing.get("tunnel_token") or "").strip()
    if not tok:
        raise ValueError("首次启用固定域名请填写 Cloudflare Tunnel Token")

    new_key = str(llm_api_key or "").strip()
    if new_key and len(new_key) < 16:
        raise ValueError("固定 LLM API Key 长度至少需要 16 位")
    final_llm_key = new_key if new_key else str(existing.get("llm_api_key") or "").strip()

    store.set_secret(CF_VAULT_TUNNEL_TOKEN_KEY, tok, colab_py=colab_py)
    if store.get_secret(CF_VAULT_TUNNEL_TOKEN_KEY, colab_py=colab_py) != tok:
        raise RuntimeError("写入 Cloudflare Tunnel Token 后回读校验失败")
    register_ephemeral_secret(tok)

    disk_obj: dict[str, Any] = {
        "tunnel_token_ref": f"vault:{CF_VAULT_TUNNEL_TOKEN_KEY}",
    }
    if clean_comfy:
        disk_obj["comfy_url"] = clean_comfy
    if clean_llm:
        disk_obj["llm_url"] = clean_llm

    if final_llm_key:
        store.set_secret(CF_VAULT_LLM_API_KEY, final_llm_key, colab_py=colab_py)
        if store.get_secret(CF_VAULT_LLM_API_KEY, colab_py=colab_py) != final_llm_key:
            raise RuntimeError("写入固定 LLM API Key 后回读校验失败")
        register_ephemeral_secret(final_llm_key)
        disk_obj["llm_api_key_ref"] = f"vault:{CF_VAULT_LLM_API_KEY}"

    _write_private_text(Path(config_path), json.dumps(disk_obj, ensure_ascii=False, indent=2) + "\n")
    return {
        "mode": "named",
        "has_token": True,
        "has_fixed_llm_key": len(final_llm_key) >= 16,
        "comfy_url": clean_comfy,
        "llm_url": clean_llm,
    }


def clear_cf_tunnel_settings(
    config_path: Path,
    *,
    colab_py: Path | None = None,
    vault: VaultStore | None = None,
) -> None:
    """删除系统凭据库中的隧道敏感凭据及配置文件，恢复默认免费临时隧道。"""
    store = vault or VaultStore()
    for key_name in (CF_VAULT_TUNNEL_TOKEN_KEY, CF_VAULT_LLM_API_KEY):
        try:
            store.delete_secret(key_name, colab_py=colab_py)
        except Exception:
            pass
    Path(config_path).unlink(missing_ok=True)


# =========================================================================
# ComfyUI 公网认证密钥、一次性免输票据与跳转路径安全校验
# =========================================================================


def generate_comfy_api_key() -> str:
    """生成带固定 `cpk-` 前缀的高熵 ComfyUI 公网访问密钥。"""
    return "cpk-" + secrets.token_urlsafe(24)


def sanitize_next_path(raw_next: str | None) -> str:
    """严格限制登录后跳转路径为站内相对路径，彻底阻断开放重定向。"""
    s = str(raw_next or "").strip()
    if not s:
        return "/"
    if any(ch in s for ch in ("\r", "\n", "\x00", "\\")):
        return "/"
    if not s.startswith("/") or s.startswith("//"):
        return "/"
    parsed = urllib.parse.urlsplit(s)
    if parsed.scheme or parsed.netloc or not parsed.path.startswith("/"):
        return "/"
    return s


def create_comfy_handoff_ticket(
    secret_key: str,
    next_path: str = "/",
    now_ts: int | None = None,
) -> str:
    """签发有效期 120 秒的一次性桌宠免输直进 HMAC 票据。"""
    key_str = str(secret_key or "").strip()
    if not key_str:
        raise ValueError("签发 ComfyUI 票据需要非空访问密钥")
    ts = int(now_ts if now_ts is not None else time.time())
    nonce = secrets.token_urlsafe(12)
    clean_next = sanitize_next_path(next_path)
    msg = f"ticket:{ts}:{nonce}:{clean_next}".encode("utf-8")
    sig = hmac.new(key_str.encode("utf-8"), msg, hashlib.sha256).hexdigest()
    ticket = f"{ts}.{nonce}.{sig}"
    register_ephemeral_secret(ticket)
    return ticket


def build_comfy_handoff_url(
    base_url: str,
    secret_key: str,
    next_path: str = "/",
) -> str:
    """构造带一次性签名票据的浏览器免输登录跳转链接。"""
    clean_base = str(base_url or "").strip().rstrip("/")
    clean_next = sanitize_next_path(next_path)
    ticket = create_comfy_handoff_ticket(secret_key, next_path=clean_next)
    return f"{clean_base}/__pet_auth?ticket={ticket}&next={urllib.parse.quote(clean_next, safe='')}"
