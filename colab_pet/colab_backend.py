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
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import webbrowser
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

try:
    import websocket
except Exception:
    websocket = None

PET_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = PET_DIR.parent
WORKSPACE_ROOT = PACKAGE_DIR
LOCAL_DOWNLOAD_ROOT = Path(os.environ.get("COLAB_DOWNLOAD_DIR") or (PACKAGE_DIR / "_downloads")) / "all"
CTL_SCRIPT = PACKAGE_DIR / "colab_ctl.py"
STATE_FILE = PET_DIR / "pet_state.json"

if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from colab_core.accounts import (  # noqa: E402
    CLI_BRIDGE_SCRIPT,
    ToolPaths,
    extract_oauth_code as _core_extract_oauth_code,
    has_gdrive_save_section as _core_has_gdrive_save_section,
    query_gdrive_quota as _core_query_gdrive_quota,
    resolve_colab_exe as _core_discover_colab_exe,
    resolve_colab_py as _core_discover_colab_py,
    resolve_rclone_conf as _core_discover_rclone_conf,
    resolve_rclone_exe as _core_discover_rclone_exe,
    sanitize_account_label as _sanitize_account_label,
    validate_and_import_rclone_conf as _core_validate_import_rclone,
)
from colab_core.colab_client import (  # noqa: E402
    COLAB_REMOTE_LORA_PULL_SCRIPT as _COLAB_REMOTE_LORA_PULL_SCRIPT,
    build_tunnel_url_pattern as _core_build_tunnel_url_pattern,
    detect_tunnel_kind as _core_detect_tunnel_kind,
    is_allowed_tunnel_url as _core_is_allowed_tunnel_url,
    probe_tunnel_alive as _core_probe_tunnel_alive,
    resolve_target_binding as _core_resolve_target_binding,
)
from colab_core.file_sync import (  # noqa: E402
    GDRIVE_LORA_REMOTE,
    GDRIVE_WF_REMOTE,
    LLM_IMAGE_EXTS,
    LLM_TEXT_EXTS,
    MEDIA_EXTS,
    MODEL_EXTS,
    StreamingMultipartFile as _StreamingMultipartFile,
    WORKFLOW_EXTS,
    api_prompt_to_ui_workflow as _api_prompt_to_ui_workflow,
    ensure_workflow_uuid as _ensure_workflow_uuid,
    extract_png_comfy_workflow,
    format_timestamped_filename as _format_timestamped_filename,
    is_api_prompt_dict as _is_api_prompt_dict,
    parse_http_mtime as _parse_http_mtime,
    resolve_non_clobber_dest as _resolve_non_clobber_dest,
    safe_output_dest as _safe_output_dest,
    sanitize_filename as _sanitize_filename,
)
from colab_core.health import export_diagnostic_bundle, run_preflight_checks  # noqa: E402
from colab_core.runtime import (  # noqa: E402
    COMFY_PROFILES as COMFY_BOOT_PROFILES,
    CancelToken,
    DEFAULT_MEASURED_GPU_RATES,
    DEFAULT_PREFS,
    LLM_PROFILES as LLM_BOOT_PROFILES,
    PROFILE_LABELS as _DEPLOY_PROFILE_LABELS,
    ProcessRunner,
    SCARCE_GPU_TTL_SEC,
    SUPPORTED_BOOT_GPUS,
    SUPPORTED_LLM_QUANTS,
    TaskManager,
    build_ctl_up_args as _core_build_ctl_up_args,
    calc_phase_dl_total as _core_calc_phase_dl_total,
    classify_deploy_failure as _core_classify_deploy_failure,
    clean_subprocess_env as _clean_env,
    collect_formal_outputs as _core_collect_formal_outputs,
    consume_deploy_line as _core_consume_deploy_line,
    default_baseline as _default_baseline,
    default_deploy_state as _core_default_deploy_state,
    estimate_session_start_balance as _core_est_sess_start,
    extract_history_error as _core_extract_history_error,
    extract_slot_tps as _core_extract_slot_tps,
    format_comfy_error as _format_comfy_error,
    format_elapsed_cn as _core_format_elapsed_cn,
    get_diag_logger,
    gpu_rate_key as _core_gpu_rate_key,
    kill_process_tree as _core_kill_process_tree,
    load_pet_state_dict as _core_load_pet_state_dict,
    normalize_profiles as _normalize_boot_profiles,
    recommend_fallback_gpu as _core_recommend_fallback_gpu,
    restore_ledger_for_account as _core_restore_ledger_for_account,
    sanitize_prefs_dict as _sanitize_prefs_dict,
    short_gpu_name as _core_short_gpu_name,
    snapshot_current_ledger as _core_snapshot_current_ledger,
    sync_active_account_ledger as _core_sync_active_account_ledger,
    upsert_account_profile as _core_upsert_account_profile,
)
from colab_remote_common import (  # noqa: E402
    DIRECT_CANVAS_JS as _COMFY_DIRECT_CANVAS_JS,
    load_pet_bridge_init_py as _load_bridge_node_py,
)
from colab_security import (  # noqa: E402
    build_comfy_handoff_url,
    clear_cf_tunnel_settings,
    load_cf_tunnel_settings,
    redact_secrets,
    register_ephemeral_secret,
    save_cf_tunnel_settings,
)

_COMFY_BRIDGE_NODE_INIT_PY = _load_bridge_node_py(PACKAGE_DIR)

_APPDATA_DIR = Path(os.environ.get("APPDATA") or (Path.home() / "AppData" / "Roaming"))
COLAB_PY = _APPDATA_DIR / "uv" / "tools" / "google-colab-cli" / "Scripts" / "python.exe"
COLAB_EXE_DEFAULT = Path.home() / ".local" / "bin" / "colab.exe"
RCLONE_EXE = Path.home() / "scoop" / "shims" / "rclone.exe"
RCLONE_CONF = _APPDATA_DIR / "rclone" / "rclone.conf"

_discover_colab_exe = lambda custom_path="": _core_discover_colab_exe(custom_path, allow_fallback=False)  # noqa: E731
_discover_colab_py = lambda custom_path="", colab_exe=None: _core_discover_colab_py(custom_path, colab_exe=colab_exe)  # noqa: E731
_discover_rclone_exe = lambda custom_path="": _core_discover_rclone_exe(custom_path)  # noqa: E731
_discover_rclone_conf = lambda custom_path="", rclone_exe=None: _core_discover_rclone_conf(custom_path, rclone_exe=rclone_exe)  # noqa: E731

COLAB_EXE_DEFAULT = _discover_colab_exe() or COLAB_EXE_DEFAULT
COLAB_PY = _discover_colab_py(colab_exe=COLAB_EXE_DEFAULT)
RCLONE_EXE = _discover_rclone_exe()
RCLONE_CONF = _discover_rclone_conf(rclone_exe=RCLONE_EXE)

_INIT_COLAB_PY = COLAB_PY
_INIT_COLAB_EXE = COLAB_EXE_DEFAULT
_INIT_RCLONE_EXE = RCLONE_EXE
_INIT_RCLONE_CONF = RCLONE_CONF

CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
_CF_SECRET_FILE = PACKAGE_DIR / ".cf_tunnel_secret.json"
_INIT_CF_SECRET_FILE = _CF_SECRET_FILE

KEEPALIVE_INTERVAL_SEC = 600.0
KEEPALIVE_RETRY_COOLDOWN_SEC = 120.0
KEEPALIVE_MAX_IDLE_SEC = 45 * 60.0
_PRESERVE_ORIGINAL_SUB_RE = re.compile(r"(?:^|/)drama3_voices$", re.IGNORECASE)



def _load_cf_secret_raw(config_path: Path | None = None, *, include_secrets: bool = False) -> dict[str, Any]:
    target_p = Path(config_path) if config_path is not None else Path(os.environ.get("COLAB_PET_CF_CONFIG_PATH") or str(_CF_SECRET_FILE))
    if _CF_SECRET_FILE != _INIT_CF_SECRET_FILE and config_path is None:
        target_p = _CF_SECRET_FILE
    return load_cf_tunnel_settings(target_p, include_secrets=include_secrets)


def _is_allowed_tunnel_url(url: str, config_path: Path | None = None) -> bool:
    cfg_p = config_path or Path(os.environ.get("COLAB_PET_CF_CONFIG_PATH") or str(_CF_SECRET_FILE))
    return _core_is_allowed_tunnel_url(url, config_path=cfg_p)


def _probe_tunnel_alive(url: str, timeout: float = 3.5, kind: str = "comfy", auth_token: str = "", cf_config_path: Path | None = None) -> bool:
    return _core_probe_tunnel_alive(
        url, timeout=timeout, kind=kind, auth_token=auth_token, cf_config_path=cf_config_path,
        allowed_checker=lambda u: _is_allowed_tunnel_url(u, config_path=cf_config_path),
    )


def _detect_tunnel_kind(url: str, timeout: float = 3.0, preferred: str = "comfy", auth_token: str = "", cf_config_path: Path | None = None) -> str | None:
    return _core_detect_tunnel_kind(
        url, timeout=timeout, preferred=preferred, auth_token=auth_token, cf_config_path=cf_config_path,
        probe_fn=_probe_tunnel_alive,
    )


def _redact_payload(obj: Any) -> Any:
    if isinstance(obj, str):
        return redact_secrets(obj)
    if isinstance(obj, list):
        return [_redact_payload(x) for x in obj]
    if isinstance(obj, dict):
        return {k: (v if k == "clipboard_text" else _redact_payload(v)) for k, v in obj.items()}
    return obj


def _run_hidden(cmd: list[str], timeout: int = 25, extra_env: dict[str, str] | None = None, input_text: str | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, input=input_text, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout, env=_clean_env(extra_env), creationflags=CREATE_NO_WINDOW)


def _hidden_python_exe() -> str:
    exe = Path(sys.executable)
    if exe.name.lower() == "pythonw.exe" and exe.with_name("python.exe").exists():
        return str(exe.with_name("python.exe"))
    return str(exe)


def _ctl_available() -> bool:
    return bool(getattr(sys, "frozen", False) or CTL_SCRIPT.exists())


def _ctl_cmd_prefix(unbuffered: bool = False) -> list[str]:
    if getattr(sys, "frozen", False):
        return [sys.executable, "--internal-ctl"]
    return [_hidden_python_exe(), "-u", str(CTL_SCRIPT)] if unbuffered else [_hidden_python_exe(), str(CTL_SCRIPT)]


class ColabPetBackend:
    """Colab 桌宠状态管理与运维动作执行器。"""

    def __init__(
        self,
        state_path: Path = STATE_FILE,
        download_root: Path = LOCAL_DOWNLOAD_ROOT,
        cf_config_path: Path | None = None,
    ) -> None:
        self.state_path = state_path
        self.download_root = Path(download_root)
        self._cf_config_path: Path | None = Path(cf_config_path) if cf_config_path is not None else None
        self._lock = threading.RLock()
        self._fetch_lock = threading.Lock()
        self._comfy_lock = threading.RLock()
        self._action_lock = threading.Lock()
        self._upload_lock = threading.Lock()
        self._keepalive_lock = threading.Lock()
        self._deploy_lock = threading.Lock()
        self._deploy_proc: Any = None
        self._deploy_cancelled: bool = False
        self._deploy_cancel_token: CancelToken | None = None
        self._diag = get_diag_logger()
        self._runner = ProcessRunner(diag=self._diag)
        self._tasks = TaskManager(diag=self._diag)
        self._deploy_dl_total: int = 3
        self._deploy_dl_done: int = 0
        self._deploy_dl_name: str = ""
        self._scarce_gpus: dict[str, dict[str, Any]] = {}
        self._deploy_state: dict[str, Any] = _core_default_deploy_state()
        self._consecutive_failures: int = 0
        self._comfy_http_failures: int = 0
        self._last_tunnel_discover_ts: float = 0.0
        self._last_llm_tunnel_discover_ts: float = 0.0
        self._last_keepalive_ts: float = 0.0
        self._last_activity_m: float = 0.0
        self._idle_warned_ep: str | None = None
        self._idle_stop_lock = threading.Lock()
        self._idle_archive_retry_after_m: float = 0.0
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
        self._comfy_secrets_by_ep: dict[str, str] = {}
        self._comfy_key_checked_eps: set[str] = set()
        self._last_comfy_key_hydrate_m: dict[str, float] = {}
        self._account_epoch: int = 0
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

    def _resolve_cf_config_path(self) -> Path:
        """解析当前实例绑定的 Cloudflare 固定隧道配置文件路径（兼容测试注入与模块覆盖）。"""
        if self._cf_config_path is not None:
            return self._cf_config_path
        if _CF_SECRET_FILE != _INIT_CF_SECRET_FILE:
            return _CF_SECRET_FILE
        env_p = str(os.environ.get("COLAB_PET_CF_CONFIG_PATH") or "").strip()
        if env_p:
            return Path(env_p)
        return _CF_SECRET_FILE

    def _current_account_key(self) -> str:
        """返回当前激活账号标识（用于会话绑定与切号围栏）。"""
        with self._lock:
            return str((self.state.get("prefs") or {}).get("active_account_id") or "").strip()

    def _record_managed_binding(self, session: str, endpoint: str) -> None:
        """记录当前账号正在纳管的 (account_id, session, endpoint) 绑定关系。"""
        ep = str(endpoint or "").strip()
        sname = str(session or "comfy").strip() or "comfy"
        if not ep:
            return
        with self._lock:
            acct_id = str((self.state.get("prefs") or {}).get("active_account_id") or "").strip()
            self.state["managed_binding"] = {
                "account_id": acct_id,
                "session": sname,
                "endpoint": ep,
            }
            self._save_state()

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
        """构造传递给 colab_ctl.py 子进程的可移植工具路径与隔离配置环境变量。"""
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
        env_map["COLAB_PET_CF_CONFIG_PATH"] = str(self._resolve_cf_config_path())
        mock_vault = str(os.environ.get("COLAB_PET_MOCK_VAULT_PATH") or "").strip()
        if mock_vault:
            env_map["COLAB_PET_MOCK_VAULT_PATH"] = mock_vault
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
            cb(_redact_payload(payload))
        except Exception:
            pass

    def is_deploying(self) -> bool:
        with self._lock:
            return bool(self._deploy_state.get("active"))

    def get_deploy_state(self) -> dict[str, Any]:
        with self._lock:
            st = dict(self._deploy_state)
            if st.get("active") and st.get("started_ts"):
                st["elapsed_sec"] = max(0, int(time.time()) - int(st["started_ts"]))
            return st

    def _get_active_scarce_gpus(self) -> dict[str, dict[str, Any]]:
        now_ts = int(time.time())
        with self._lock:
            for k in [k for k, v in self._scarce_gpus.items() if (now_ts - int(v.get("ts") or 0)) > SCARCE_GPU_TTL_SEC]:
                self._scarce_gpus.pop(k, None)
            return {k: dict(v) for k, v in self._scarce_gpus.items()}

    def _load_state(self) -> dict[str, Any]:
        cf_cfg_p = self._resolve_cf_config_path()
        return _core_load_pet_state_dict(
            self.state_path,
            is_allowed_tunnel_fn=lambda u: _is_allowed_tunnel_url(u, config_path=cf_cfg_p),
            sanitize_label_fn=_sanitize_account_label,
        )

    def _save_state(self) -> None:
        """原子写入非敏感配置与账本状态，不保存任何登录凭据。"""
        tmp_path = self.state_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(self.state, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_path.replace(self.state_path)

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

        cmd = [str(colab_py), str(CLI_BRIDGE_SCRIPT), "probe"]
        if need_userinfo:
            cmd.append("--need-userinfo")
        if active_id:
            cmd.append(f"--active-acct={active_id}")

        proc = _run_hidden(cmd, timeout=25)
        if proc.returncode != 0:
            err = redact_secrets((proc.stderr or proc.stdout or "").strip())
            raise RuntimeError(f"Colab 查询失败: {err.splitlines()[-1] if err else f'退出码 {proc.returncode}'}")

        stdout = proc.stdout or ""
        matched = re.search(r"<<<COLAB_PET_JSON>>>(.*?)<<<END_COLAB_PET_JSON>>>", stdout, re.S)
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
                _core_upsert_account_profile(
                    self.state,
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
                    self.state["session_start_balance"] = _core_est_sess_start(
                        balance, rate_hourly, int(prev_since), now_ts, day_start_f, day_topup_f
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
            self.state["managed_binding"] = None
            self._last_activity_m = 0.0
            self._idle_warned_ep = None
            self._last_keepalive_ts = 0.0
            self._idle_archive_retry_after_m = 0.0

        self.state["last_rate_hourly"] = rate_hourly
        self.state["last_assignments_count"] = assignments_count
        self.state["last_sessions"] = sessions
        self.state["last_updated_ts"] = now_ts
        if is_running and primary_sess and primary_sess.get("endpoint") and primary_sess.get("name") not in (None, "?"):
            active_aid = str((self.state.get("prefs") or {}).get("active_account_id") or "").strip()
            cur_bind = self.state.get("managed_binding")
            if not isinstance(cur_bind, dict) or cur_bind.get("endpoint") != str(primary_sess["endpoint"]):
                self.state["managed_binding"] = {
                    "account_id": active_aid,
                    "session": str(primary_sess["name"]),
                    "endpoint": str(primary_sess["endpoint"]),
                }

        # 仅当单实例在线运行且返回有效耗速时自动校准对应卡型与内存档的实测单价（防止多卡汇总耗速污染单卡费率）
        if assignments_count == 1 and len(sessions) == 1 and 0.02 < rate_hourly < 200.0:
            s0 = sessions[0]
            rkey = _core_gpu_rate_key(s0.get("accelerator"), s0.get("machine_shape"))
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
        cf_cfg_p = self._resolve_cf_config_path()
        active_eps = {str(s.get("endpoint")) for s in sessions if s.get("endpoint")}
        raw_cache = self.state.get("tunnel_cache")
        if isinstance(raw_cache, dict):
            self.state["tunnel_cache"] = {
                ep: url
                for ep, url in raw_cache.items()
                if ep in active_eps and isinstance(url, str) and _is_allowed_tunnel_url(url, config_path=cf_cfg_p)
            }
        else:
            self.state["tunnel_cache"] = {}
        for stale_ep in list(self._service_meta_by_ep.keys()):
            if stale_ep not in active_eps:
                self._service_meta_by_ep.pop(stale_ep, None)
        for stale_ep in list(self._runtime_secrets.keys()):
            if stale_ep not in active_eps:
                self._runtime_secrets.pop(stale_ep, None)
        for stale_ep in list(self._comfy_secrets_by_ep.keys()):
            if stale_ep not in active_eps:
                self._comfy_secrets_by_ep.pop(stale_ep, None)
        self._comfy_key_checked_eps.intersection_update(active_eps)

        _core_sync_active_account_ledger(self.state, balance=balance, is_running=is_running)

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
        gdrive_ok = _core_has_gdrive_save_section(rclone_conf)

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

        cf_cfg = _load_cf_secret_raw(config_path=self._resolve_cf_config_path(), include_secrets=False)
        has_token = bool(cf_cfg.get("has_token"))
        has_fixed_key = bool(cf_cfg.get("has_fixed_llm_key"))
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
            sess_start = _core_est_sess_start(
                float(balance),
                rate_hourly,
                int(active_since),
                int(updated_ts or now_ts),
                float(raw_day_start if raw_day_start is not None else balance),
                float(self.state.get("day_topup_cu") or 0.0),
            )
        disp_session_consumed = (
            round(max(0.0, round(float(sess_start), 2) - disp_bal), 2)
            if (is_running and sess_start is not None and disp_bal is not None)
            else 0.0
        )

        safe_err = redact_secrets(error_msg) if error_msg is not None else None
        return {
            "ok": safe_err is None,
            "stale": stale,
            "error_msg": safe_err,
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

    def _get_epoch_fetch_lock(self, epoch: int) -> threading.Lock:
        """按账号 epoch 隔离状态探针锁，切号时无需等待旧账号挂起中的慢网络探针。"""
        with self._lock:
            locks = getattr(self, "_epoch_fetch_locks", None)
            if locks is None:
                locks = {}
                self._epoch_fetch_locks = locks
            if epoch not in locks:
                locks[epoch] = threading.Lock()
                if len(locks) > 8:
                    for old_k in sorted(locks.keys())[:-8]:
                        locks.pop(old_k, None)
            return locks[epoch]

    def fetch_live_status(self) -> dict[str, Any]:
        """串行刷新一次线上状态并广播给前端；含账号 epoch 围栏防止切号期间旧结果污染新账本。"""
        with self._lock:
            start_epoch = self._account_epoch
        epoch_lock = self._get_epoch_fetch_lock(start_epoch)
        with epoch_lock:
            try:
                live = self._query_colab_live()
                with self._lock:
                    if self._account_epoch != start_epoch:
                        return dict(self.last_snapshot)
                if live.get("auth_missing"):
                    with self._lock:
                        if self._account_epoch != start_epoch:
                            return dict(self.last_snapshot)
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
                            "managed_binding",
                        ):
                            self.state[lk] = [] if lk == "last_sessions" else (0 if lk == "last_assignments_count" else (0.0 if lk == "last_rate_hourly" else None))
                        self._service_meta_by_ep.clear()
                        self._runtime_secrets.clear()
                        self._comfy_secrets_by_ep.clear()
                        self._comfy_key_checked_eps.clear()
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
                    if self._account_epoch != start_epoch:
                        return dict(self.last_snapshot)
                    self._auth_missing = False
                    self._auth_expired = False
                    self._consecutive_failures = 0
                    self._update_ledger(live)
                    self._save_state()
                    snap = self._build_snapshot(live_data=live, stale=False, error_msg=None)
                    self.last_snapshot = snap
            except Exception as exc:
                with self._lock:
                    if self._account_epoch != start_epoch:
                        return dict(self.last_snapshot)
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
                            error_msg=redact_secrets(str(exc)),
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

    def _get_comfy_key_for_url(self, base_url: str = "", ep: str = "") -> str:
        """从纯内存密钥表按 endpoint 或当前活跃 ComfyUI 隧道匹配公网访问密钥。"""
        clean_base = str(base_url or "").strip().rstrip("/")
        with self._lock:
            if ep and ep in self._comfy_secrets_by_ep:
                return str(self._comfy_secrets_by_ep.get(ep) or "")
            if clean_base:
                cache = self.state.get("tunnel_cache") or {}
                for c_ep, c_url in cache.items():
                    if str(c_url or "").strip().rstrip("/") == clean_base and c_ep in self._comfy_secrets_by_ep:
                        return str(self._comfy_secrets_by_ep.get(c_ep) or "")
                for m_ep, m_val in self._service_meta_by_ep.items():
                    if isinstance(m_val, dict):
                        for uk in ("comfy_tunnel_url", "tunnel_url"):
                            if str(m_val.get(uk) or "").strip().rstrip("/") == clean_base and m_ep in self._comfy_secrets_by_ep:
                                return str(self._comfy_secrets_by_ep.get(m_ep) or "")
            act_ep = str(self.state.get("active_endpoint") or "").strip()
            if act_ep and act_ep in self._comfy_secrets_by_ep:
                return str(self._comfy_secrets_by_ep.get(act_ep) or "")
            if len(self._comfy_secrets_by_ep) == 1:
                return str(next(iter(self._comfy_secrets_by_ep.values())) or "")
        return ""

    def _hydrate_comfy_key_if_needed(
        self, base_url: str = "", ep: str = "", *, force: bool = False
    ) -> str:
        """遇到 401 且尚缺 ComfyUI 公网密钥时触发一次免 WS 密钥补水；已有密钥仍 401 时遵守 8 秒冷却。"""
        if not self._has_running_session():
            return ""
        key_tag = ep or str(base_url or "").strip().rstrip("/") or "__default__"
        now_m = time.monotonic()
        with self._lock:
            last_m = float(self._last_comfy_key_hydrate_m.get(key_tag) or 0.0)
            if (not force) and (now_m - last_m) < 8.0:
                return self._get_comfy_key_for_url(base_url, ep)
            self._last_comfy_key_hydrate_m[key_tag] = now_m
        _, resolved_ep, _ = self._fast_read_tunnel_url(ep, "comfy")
        return self._get_comfy_key_for_url(base_url, resolved_ep or ep)

    def _comfy_http_request(
        self,
        method: str,
        url: str,
        *,
        base_url: str = "",
        ep: str = "",
        retry_on_401: bool = True,
        **kwargs: Any,
    ) -> requests.Response:
        """统一执行带公网认证头（Bearer）的 ComfyUI HTTP 请求，遇 401 自动补水密钥重试一次。"""
        inferred_base = base_url.strip().rstrip("/") if base_url else ""
        if not inferred_base and "://" in url:
            parsed = urllib.parse.urlsplit(url)
            inferred_base = f"{parsed.scheme}://{parsed.netloc}"

        headers = dict(kwargs.pop("headers", None) or {})
        comfy_key = self._get_comfy_key_for_url(inferred_base, ep)
        if comfy_key and "Authorization" not in headers:
            headers["Authorization"] = f"Bearer {comfy_key}"

        resp = self._http.request(method, url, headers=headers, **kwargs)
        if resp.status_code == 401 and retry_on_401 and self._has_running_session():
            try:
                resp.close()
            except Exception:
                pass
            new_key = self._hydrate_comfy_key_if_needed(
                inferred_base, ep, force=(not bool(comfy_key))
            )
            if new_key and new_key != comfy_key:
                headers["Authorization"] = f"Bearer {new_key}"
                resp = self._http.request(method, url, headers=headers, **kwargs)
        return resp

    def _comfy_http_get(
        self,
        url: str,
        *,
        base_url: str = "",
        ep: str = "",
        retry_on_401: bool = True,
        **kwargs: Any,
    ) -> requests.Response:
        return self._comfy_http_request(
            "GET",
            url,
            base_url=base_url,
            ep=ep,
            retry_on_401=retry_on_401,
            **kwargs,
        )

    def _comfy_http_post(
        self,
        url: str,
        *,
        base_url: str = "",
        ep: str = "",
        retry_on_401: bool = True,
        **kwargs: Any,
    ) -> requests.Response:
        return self._comfy_http_request(
            "POST",
            url,
            base_url=base_url,
            ep=ep,
            retry_on_401=retry_on_401,
            **kwargs,
        )

    def _tunnel_url_regex_pattern(self) -> str:
        return _core_build_tunnel_url_pattern(_load_cf_secret_raw(config_path=self._resolve_cf_config_path(), include_secrets=False))

    def _fast_read_tunnel_url(self, ep: str, sname: str) -> tuple[str, str, str | None]:
        """通过 Colab Contents HTTP 接口免 WebSocket 速读 /root/colab_service_meta.json 或隧道文件。"""
        colab_py = self._get_colab_py()
        if not colab_py.exists():
            return "error", ep, None
        fast_proc = _run_hidden(
            [str(colab_py), str(CLI_BRIDGE_SCRIPT), "fast-tunnel", ep, sname, self._tunnel_url_regex_pattern()],
            timeout=15,
        )
        if fast_proc.returncode != 0:
            return "error", ep, None
        m_json = re.search(r"<<<COLAB_TUNNEL>>>(.*?)<<<END_COLAB_TUNNEL>>>", fast_proc.stdout or "", re.S)
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
                self.state["managed_binding"] = None
                self._service_meta_by_ep.clear()
                self._runtime_secrets.clear()
                self._comfy_secrets_by_ep.clear()
                self._comfy_key_checked_eps.clear()
                self._save_state()
            return "no_session", ep, None
        resolved_ep = str(tdata.get("endpoint") or ep)
        fast_url = str(tdata.get("url") or "").strip() or None

        # 将 api_key 与 comfy_api_key 严格剥离至纯内存态字典，非敏感元信息存入 _service_meta_by_ep
        raw_meta = tdata.get("meta")
        api_key = str(tdata.get("api_key") or "").strip()
        comfy_key = str(tdata.get("comfy_api_key") or "").strip()
        if isinstance(raw_meta, dict):
            if not api_key and raw_meta.get("api_key"):
                api_key = str(raw_meta.get("api_key") or "").strip()
            if not comfy_key and raw_meta.get("comfy_api_key"):
                comfy_key = str(raw_meta.get("comfy_api_key") or "").strip()
        if api_key:
            register_ephemeral_secret(api_key)
        if comfy_key:
            register_ephemeral_secret(comfy_key)

        if resolved_ep:
            with self._lock:
                self._comfy_key_checked_eps.add(resolved_ep)
                if comfy_key:
                    self._comfy_secrets_by_ep[resolved_ep] = comfy_key

        if resolved_ep and isinstance(raw_meta, dict):
            clean_meta = {k: v for k, v in raw_meta.items() if k not in ("api_key", "comfy_api_key")}
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

    def _read_cf_public_llm_url(self) -> str | None:
        """从本地云端隧道配置中仅读取公开 llm_url（绝不读取或暴露敏感 token/key）。"""
        cf_cfg_p = self._resolve_cf_config_path()
        data = _load_cf_secret_raw(config_path=cf_cfg_p, include_secrets=False)
        u = str(data.get("llm_url") or "").strip().rstrip("/")
        if u and _is_allowed_tunnel_url(u, config_path=cf_cfg_p):
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

        cf_cfg_p = self._resolve_cf_config_path()
        comfy_tok = self._get_comfy_key_for_url(cached_url or "", ep)
        if known_meta is not None:
            mkind = str(known_meta.get("kind") or "comfy")
            if mkind == "dual":
                cand_comfy = str(known_meta.get("comfy_tunnel_url") or cached_url or "").strip()
                cand_tok = self._get_comfy_key_for_url(cand_comfy, ep)
                if cand_comfy and _probe_tunnel_alive(
                    cand_comfy, timeout=3.0, kind="comfy", auth_token=cand_tok, cf_config_path=cf_cfg_p
                ):
                    return cand_comfy
            elif cached_url and _probe_tunnel_alive(
                cached_url, timeout=3.0, kind=mkind, auth_token=comfy_tok, cf_config_path=cf_cfg_p
            ):
                return cached_url
        elif cached_url:
            detected = _detect_tunnel_kind(
                cached_url, timeout=3.0, preferred=preferred_kind, auth_token=comfy_tok, cf_config_path=cf_cfg_p
            )
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
                        htok = self._get_comfy_key_for_url(fast_url or "", resolved_ep or ep)
                        if (
                            tstatus == "ok"
                            and fast_url
                            and fast_url != cached_url
                            and _probe_tunnel_alive(
                                fast_url, timeout=3.0, kind=h_probe_k, auth_token=htok, cf_config_path=cf_cfg_p
                            )
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
            ftok = self._get_comfy_key_for_url(fast_url, resolved_ep or ep)
            if _probe_tunnel_alive(
                fast_url, timeout=3.5, kind=probe_kind, auth_token=ftok, cf_config_path=cf_cfg_p
            ):
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

        cf_cfg_p = self._resolve_cf_config_path()
        cand = str(meta.get("llm_tunnel_url") or "").strip() if mkind == "dual" else ""
        if cand and _probe_tunnel_alive(cand, timeout=3.0, kind="llm", cf_config_path=cf_cfg_p):
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
                if cand and _probe_tunnel_alive(cand, timeout=3.0, kind="llm", cf_config_path=cf_cfg_p):
                    return cand

        if mkind in ("dual", "llm") or self.has_llm_service_active():
            cf_llm = self._read_cf_public_llm_url()
            if cf_llm and _probe_tunnel_alive(cf_llm, timeout=3.0, kind="llm", cf_config_path=cf_cfg_p):
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

    def _ensure_legacy_cmd_file(self, base_url: str, ep: str = "") -> None:
        """确保 userdata 根目录存在桥接指令文件，避免浏览器首次轮询出现 404 日志。"""
        base = base_url.rstrip("/")
        try:
            chk = self._comfy_http_get(f"{base}/userdata/colab_pet_bridge_cmd.json", base_url=base, ep=ep, timeout=4)
            if chk.status_code == 404:
                self._comfy_http_post(
                    f"{base}/userdata/colab_pet_bridge_cmd.json?overwrite=true",
                    base_url=base,
                    ep=ep,
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
                r = self._comfy_http_get(f"{base}/colab_pet/ping", base_url=base, ep=ep, timeout=4)
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
                            jr = self._comfy_http_get(
                                f"{base}/extensions/colab_pet_bridge/direct_canvas.js",
                                base_url=base,
                                ep=ep,
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
                    r2 = self._comfy_http_get(
                        f"{base}/extensions/colab_pet/direct_canvas.js",
                        base_url=base,
                        ep=ep,
                        timeout=4,
                    )
                    if r2.status_code == 200 and "colab_pet_direct_canvas v2" in (r2.text or ""):
                        self._ensure_legacy_cmd_file(base, ep=ep)
                        with self._canvas_ext_lock:
                            self._canvas_ext_ready_eps[cache_key] = "legacy"
                        return "legacy", False
                    break
                except Exception:
                    if attempt == 0:
                        time.sleep(0.25)

        # 3) 存量实例尚未挂载或仍为 v1 时，经 Colab Contents REST 免重启热写入/覆盖最新扩展
        colab_py = self._get_colab_py()
        if not colab_py.exists() or not self._has_running_session():
            return ("native" if native_route_alive else "none"), False

        try:
            payload = {
                "js_b64": base64.b64encode(_COMFY_DIRECT_CANVAS_JS.encode("utf-8")).decode("ascii"),
                "py_b64": base64.b64encode(_COMFY_BRIDGE_NODE_INIT_PY.encode("utf-8")).decode("ascii"),
            }
            proc = subprocess.run(
                [str(colab_py), str(CLI_BRIDGE_SCRIPT), "install-ext", ep, sname],
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
                        vr = self._comfy_http_get(cu, base_url=base, ep=ep, timeout=4)
                        if vr.status_code == 200 and "colab_pet_direct_canvas v2" in (vr.text or ""):
                            target_mode = "native" if native_route_alive else "legacy"
                            if target_mode == "legacy":
                                self._ensure_legacy_cmd_file(base, ep=ep)
                            with self._canvas_ext_lock:
                                self._canvas_ext_ready_eps[cache_key] = target_mode
                            return target_mode, True
                    except Exception:
                        pass
                time.sleep(0.25)
        except Exception:
            pass
        return ("native" if native_route_alive else "none"), False

    def _launch_comfy_browser(
        self,
        tunnel_url: str,
        open_url: str,
        *,
        ep: str = "",
        sname: str = "comfy",
        already_fetched_meta: bool = False,
    ) -> None:
        """打开 ComfyUI 浏览器窗口：存在公网保护密钥时自动签发一次性票据实现免输口令直进。"""
        base = tunnel_url.strip().rstrip("/")
        comfy_key = self._get_comfy_key_for_url(base, ep)
        with self._lock:
            checked = bool(ep and ep in self._comfy_key_checked_eps)
        if not comfy_key and not already_fetched_meta and not checked and self._has_running_session():
            _, resolved_ep, _ = self._fast_read_tunnel_url(ep, sname)
            comfy_key = self._get_comfy_key_for_url(base, resolved_ep or ep)
        launch_url = open_url
        if comfy_key:
            next_rel = open_url[len(base):] if open_url.startswith(base) else "/"
            if not next_rel.startswith("/"):
                next_rel = f"/{next_rel}" if next_rel else "/"
            launch_url = build_comfy_handoff_url(base, comfy_key, next_path=next_rel)
        webbrowser.open(launch_url)

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
                pr = self._comfy_http_post(
                    f"{base}/colab_pet/open_workflow",
                    base_url=base,
                    ep=ep,
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
                                ar = self._comfy_http_get(
                                    f"{base}/colab_pet/ack_workflow",
                                    base_url=base,
                                    ep=ep,
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
                pr = self._comfy_http_post(
                    f"{base}/userdata/colab_pet_bridge_cmd.json?overwrite=true",
                    base_url=base,
                    ep=ep,
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
                            ar = self._comfy_http_get(
                                f"{base}/userdata/colab_pet_bridge_ack.json",
                                base_url=base,
                                ep=ep,
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

        self._launch_comfy_browser(base, open_url, ep=ep, sname=sname)
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

        def _bg(_cancel: CancelToken) -> None:
            with tempfile.TemporaryDirectory(prefix="pet_wf_bak_") as tmp_dir:
                tmp_wf = Path(tmp_dir) / safe_name
                tmp_wf.write_bytes(wf_bytes)
                self._runner.run(
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
                    run_fn=_run_hidden,
                    task_name="backup_workflow",
                )

        self._tasks.submit("backup_workflow", _bg, exclusive=False)

    def _open_browser_for_comfy(
        self,
        tunnel_url: str,
        ep: str = "",
        sname: str = "comfy",
        workflow_name: str | None = None,
        already_fetched_meta: bool = False,
    ) -> str:
        """装配 ComfyUI 打开链接并拉起默认浏览器（含公网访问保护时自动附带一次性免密直进票据）。"""
        if workflow_name:
            self._ensure_direct_canvas_support(tunnel_url, ep=ep, sname=sname)
        open_url = self._build_comfy_open_url(tunnel_url, workflow_name)
        self._launch_comfy_browser(
            tunnel_url,
            open_url,
            ep=ep,
            sname=sname,
            already_fetched_meta=already_fetched_meta,
        )
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
            tunnel_url,
            ep=resolved_ep,
            sname=sname,
            workflow_name=workflow_name,
            already_fetched_meta=already_fetched_meta,
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

    def _ensure_target_session(
        self, session: str, empty_msg: str, *, always_probe_when_none: bool = False
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        """定位目标活跃会话（无缓存时按需刷新状态），无活跃实例时返回标准失败响应。"""
        target = self._pick_target_session(session)
        if target is None and (always_probe_when_none or not self._has_running_session()):
            snap = self.fetch_live_status()
            target = self._pick_target_session(session)
            if target is None and snap.get("assignments_count", 0) == 0 and not snap.get("sessions"):
                return None, {"ok": False, "message": empty_msg}
        return target, None

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
            target, err_resp = self._ensure_target_session(
                session, "当前无运行中的 Colab 实例，请先部署开机", always_probe_when_none=True
            )
            if err_resp is not None:
                return err_resp

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
            cf_cfg_p = self._resolve_cf_config_path()
            cached_tok = self._get_comfy_key_for_url(cached_url or "", ep)
            if known_meta:
                mkind = str(known_meta.get("kind") or "comfy")
                if mkind == "dual":
                    cand_comfy = str(known_meta.get("comfy_tunnel_url") or cached_url or "").strip()
                    cand_tok = self._get_comfy_key_for_url(cand_comfy, ep)
                    if cand_comfy and _probe_tunnel_alive(
                        cand_comfy, timeout=3.0, kind="comfy", auth_token=cand_tok, cf_config_path=cf_cfg_p
                    ):
                        return self._open_resolved_service_url(
                            cand_comfy,
                            ep=ep,
                            sname=sname,
                            workflow_name=workflow_name,
                            stage="cache",
                            detected_kind="comfy",
                        )
                elif cached_url and _probe_tunnel_alive(
                    cached_url, timeout=3.0, kind=mkind, auth_token=cached_tok, cf_config_path=cf_cfg_p
                ):
                    return self._open_resolved_service_url(
                        cached_url,
                        ep=ep,
                        sname=sname,
                        workflow_name=workflow_name,
                        stage="cache",
                        detected_kind=mkind,
                    )
            elif cached_url:
                det_kind = _detect_tunnel_kind(
                    cached_url, timeout=3.0, preferred=preferred_kind, auth_token=cached_tok, cf_config_path=cf_cfg_p
                )
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
            ftok = self._get_comfy_key_for_url(cand_fast, resolved_ep or ep)
            if tstatus == "ok" and cand_fast and _probe_tunnel_alive(
                cand_fast, timeout=4.0, kind=probe_fast_k, auth_token=ftok, cf_config_path=cf_cfg_p
            ):
                if resolved_ep:
                    with self._lock:
                        self.state.setdefault("tunnel_cache", {})[resolved_ep] = cand_fast
                        self._save_state()
                resolved_kind = (
                    "comfy"
                    if fast_kind == "dual"
                    else (
                        fast_kind
                        if fast_kind in ("comfy", "llm")
                        else _detect_tunnel_kind(cand_fast, timeout=2.5, auth_token=ftok, cf_config_path=cf_cfg_p)
                    )
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

            # 3. 第三级：前两级均不可用时回退 colab_ctl tunnel --restart 重建隧道（兼容冻结态）
            if not _ctl_available():
                return {"ok": False, "message": f"未找到控制脚本: {CTL_SCRIPT.name}"}
            proc = _run_hidden(
                [*_ctl_cmd_prefix(), "tunnel", "--restart", "-s", sname],
                timeout=60,
                extra_env=self._build_ctl_env(),
            )
            output = redact_secrets(f"{proc.stdout}\n{proc.stderr}")
            tunnel_pat = self._tunnel_url_regex_pattern()
            match = re.search(
                rf"(?:(?:COMFYUI|SERVICE)_TUNNEL_URL:\s*|\[(?:ComfyUI|LLM|Qwen[^]]*) WebUI 地址\]\s*)({tunnel_pat})",
                output,
            )
            if proc.returncode == 0 and match:
                url = match.group(1).strip()
                use_ep = resolved_ep or ep
                self._fast_read_tunnel_url(use_ep, sname)
                if use_ep:
                    with self._lock:
                        self.state.setdefault("tunnel_cache", {})[use_ep] = url
                        if use_ep in self._service_meta_by_ep:
                            self._service_meta_by_ep[use_ep]["tunnel_url"] = url
                            if self._service_meta_by_ep[use_ep].get("kind") == "dual":
                                self._service_meta_by_ep[use_ep]["comfy_tunnel_url"] = url
                        self._save_state()
                rtok = self._get_comfy_key_for_url(url, use_ep)
                det_after = (
                    "comfy"
                    if fast_kind == "dual"
                    else _detect_tunnel_kind(
                        url, timeout=3.0, preferred=preferred_kind, auth_token=rtok, cf_config_path=cf_cfg_p
                    )
                )
                return self._open_resolved_service_url(
                    url,
                    ep=use_ep,
                    sname=sname,
                    workflow_name=workflow_name,
                    stage="restarted",
                    detected_kind=det_after,
                    already_fetched_meta=True,
                )

            if ep:
                with self._lock:
                    cache = self.state.get("tunnel_cache")
                    if isinstance(cache, dict) and ep in cache:
                        cache.pop(ep, None)
                        self._save_state()
            tail = (output.strip().splitlines() or ["未获取到隧道地址"])[-1]
            return {"ok": False, "message": f"隧道开启失败: {redact_secrets(tail)}"}
        except Exception as exc:
            return {"ok": False, "message": f"隧道操作异常: {redact_secrets(str(exc))}"}
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
            target, err_resp = self._ensure_target_session(
                session, "当前无运行中的 Colab 实例，请先部署开机", always_probe_when_none=True
            )
            if err_resp is not None:
                return err_resp

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

            cf_cfg_p = self._resolve_cf_config_path()
            cand_llm = str(
                meta_fast.get("llm_tunnel_url") or (fast_url if fast_kind == "llm" else "") or ""
            ).strip()
            if cand_llm and _probe_tunnel_alive(cand_llm, timeout=4.0, kind="llm", cf_config_path=cf_cfg_p):
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
            if not _ctl_available():
                return {"ok": False, "message": f"未找到控制脚本: {CTL_SCRIPT.name}"}

            proc = _run_hidden(
                [*_ctl_cmd_prefix(), "tunnel", "--restart", "-s", sname],
                timeout=60,
                extra_env=self._build_ctl_env(),
            )
            output = redact_secrets(f"{proc.stdout}\n{proc.stderr}")
            tunnel_pat = self._tunnel_url_regex_pattern()
            m_llm = re.search(rf"LLM_TUNNEL_URL:\s*({tunnel_pat})", output)
            if not m_llm and fast_kind == "llm":
                m_llm = re.search(
                    rf"(?:SERVICE_TUNNEL_URL:\s*|\[LLM WebUI / API 地址\]\s*)({tunnel_pat})",
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
            return {"ok": False, "message": f"暂未获取到在线 Qwen3.8 对话隧道: {redact_secrets(tail)}"}
        except Exception as exc:
            return {"ok": False, "message": f"打开 LLM 对话页异常: {redact_secrets(str(exc))}"}
        finally:
            self._action_lock.release()

    def get_comfy_access_bundle(self, session: str = "comfy") -> dict[str, Any]:
        """组装当前 ComfyUI 公网入口链接与访问口令（仅供剪贴板写入，不回传明文口令）。"""
        self.touch_activity()
        if self.is_deploying():
            return {"ok": False, "message": "正在执行 7 步开机部署，请等待就绪后再操作"}
        target, err_resp = self._ensure_target_session(session, "当前无运行中的 ComfyUI 实例，请先部署开机")
        if err_resp is not None:
            return err_resp

        ep = str(target.get("endpoint") or "") if target else ""
        sname = target["name"] if target and target.get("name") not in (None, "?") else session
        turl = self.resolve_active_tunnel_url(session=sname, allow_fast_discover=True)
        if not turl:
            tstatus, resolved_ep, fast_url = self._fast_read_tunnel_url(ep, sname)
            if tstatus == "no_session":
                return {"ok": False, "message": "当前无运行中的 ComfyUI 实例"}
            if resolved_ep:
                ep = resolved_ep
            if fast_url:
                turl = fast_url
        if not turl:
            return {"ok": False, "message": "暂未获取到在线 ComfyUI 公网链接，请先左键打开一次"}

        clean_url = turl.strip().rstrip("/")
        comfy_key = self._get_comfy_key_for_url(clean_url, ep)
        if not comfy_key and self._has_running_session():
            _, resolved_ep, _ = self._fast_read_tunnel_url(ep, sname)
            comfy_key = self._get_comfy_key_for_url(clean_url, resolved_ep or ep)

        clip_text = (
            f"ComfyUI URL: {clean_url}\n访问口令: {comfy_key}"
            if comfy_key
            else clean_url
        )
        return {
            "ok": True,
            "url": clean_url,
            "has_key": bool(comfy_key),
            "clipboard_text": clip_text,
            "message": (
                "📋 已复制 ComfyUI 公网链接与访问口令"
                if comfy_key
                else "📋 已复制 ComfyUI 公网链接"
            ),
        }

    def get_llm_api_bundle(self, session: str = "comfy") -> dict[str, Any]:
        """组装当前在线 Qwen3.8 实例（单模或双栖）的 OpenAI 兼容接口配置包（不持久化密钥）。"""
        self.touch_activity()
        if self.is_deploying():
            return {"ok": False, "message": "正在执行 7 步开机部署，请等待就绪后再复制配置"}

        target, err_resp = self._ensure_target_session(session, "当前无运行中的 Qwen3.8 实例，请先部署开机")
        if err_resp is not None:
            return err_resp

        ep = str(target.get("endpoint") or "") if target else ""
        sname = target["name"] if target and target.get("name") not in (None, "?") else session

        turl = self.resolve_active_llm_tunnel_url(session=sname, allow_fast_discover=True)
        fetched_now = False
        cf_cfg_p = self._resolve_cf_config_path()
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
            if llm_cand and _probe_tunnel_alive(llm_cand, timeout=3.5, kind="llm", cf_config_path=cf_cfg_p):
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
            if not turl and llm_cand and _probe_tunnel_alive(llm_cand, timeout=3.5, kind="llm", cf_config_path=cf_cfg_p):
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

            if not _ctl_available():
                return {"ok": False, "message": f"未找到控制脚本: {CTL_SCRIPT.name}"}
            sname = sname_chk
            if ep_chk:
                self._record_managed_binding(sname, ep_chk)
            res = self._runner.run(
                [*_ctl_cmd_prefix(), "sync", "--pull", "-s", sname],
                timeout=180,
                extra_env=self._build_ctl_env(),
                run_fn=_run_hidden,
                task_name="sync_outputs",
            )
            output = res.combined_output
            tail = (output.splitlines() or ["同步结束"])[-1]
            if res.ok:
                if self.download_root.exists():
                    try:
                        os.startfile(str(self.download_root))  # Windows 直接打开本地收图文件夹
                    except Exception:
                        pass
                return {"ok": True, "message": f"拉回完成: {tail}"}
            return {"ok": False, "message": f"同步失败: {res.error_summary}"}
        except Exception as exc:
            self._diag.error("backend", f"同步异常: {redact_secrets(str(exc))}")
            return {"ok": False, "message": f"同步异常: {redact_secrets(str(exc))}"}
        finally:
            self._action_lock.release()

    def _resolve_target_binding(
        self, session: str = "comfy", endpoint: str | None = None
    ) -> tuple[str, str, bool]:
        """解析目标操作实例的 (session_name, endpoint, 是否纯无名孤儿)。"""
        with self._lock:
            sessions = list(self.state.get("last_sessions") or [])
            binding = self.state.get("managed_binding")
            active_aid = str((self.state.get("prefs") or {}).get("active_account_id") or "").strip()
        return _core_resolve_target_binding(
            sessions=sessions,
            binding=binding,
            active_account_id=active_aid,
            session=session,
            endpoint=endpoint,
        )

    def stop_session_instance(
        self,
        session: str = "comfy",
        endpoint: str | None = None,
        account_id: str | None = None,
    ) -> dict[str, Any]:
        """精准释放绑定会话对应的单个 Colab 实例，绝不误杀同账号并存的其他会话，并停后核实。"""
        expected_acct = str(account_id).strip() if account_id is not None else None
        if expected_acct is not None and self._current_account_key() != expected_acct:
            return {"ok": False, "aborted_by_switch": True, "message": "账号已切换，取消原账号停卡"}

        if account_id is None and self.is_deploying():
            self.cancel_deploy()
            time.sleep(0.4)

        if not self._action_lock.acquire(blocking=False):
            return {"ok": False, "message": "另一项操作正在执行，请稍等片刻"}
        try:
            if expected_acct is not None and self._current_account_key() != expected_acct:
                return {"ok": False, "aborted_by_switch": True, "message": "账号已切换，取消原账号停卡"}
            if not _ctl_available():
                return {"ok": False, "message": f"未找到控制脚本: {CTL_SCRIPT.name}"}

            sname, target_ep, orphan_only = self._resolve_target_binding(session=session, endpoint=endpoint)
            if not target_ep and not self._has_running_session():
                snap = self.fetch_live_status()
                sname, target_ep, orphan_only = self._resolve_target_binding(session=session, endpoint=endpoint)
                if snap.get("assignments_count", 0) == 0 and not (snap.get("sessions") or []):
                    return {"ok": True, "message": "当前无活跃 Colab 实例", "snapshot": snap}

            if orphan_only and not target_ep:
                return {
                    "ok": False,
                    "orphan_only": True,
                    "message": "当前仅为网页端孤儿实例，请右键「一键停卡」执行停止全部",
                }

            with self._lock:
                start_epoch = self._account_epoch

            cmd = [*_ctl_cmd_prefix(), "down", "-s", sname]
            if target_ep:
                cmd.extend(["--endpoint", target_ep])
            proc = self._runner.run(
                cmd,
                timeout=60,
                extra_env=self._build_ctl_env(),
                run_fn=_run_hidden,
                task_name="stop_session",
            )
            output = proc.combined_output

            with self._lock:
                if self._account_epoch != start_epoch:
                    return {
                        "ok": False,
                        "aborted_by_switch": True,
                        "message": "停卡期间已切换账号，已放弃写回旧账号状态",
                    }

            snap = self.fetch_live_status()
            rem_sessions = list(snap.get("sessions") or [])
            rem_eps = {str(s.get("endpoint") or "") for s in rem_sessions if s.get("endpoint")}

            if not proc.ok:
                tail = (output.splitlines() or [proc.error_summary or "停机指令异常"])[-1]
                return {"ok": False, "message": f"停卡失败: {tail}", "snapshot": snap}

            still_present = bool(target_ep and target_ep in rem_eps) or (
                not target_ep and int(snap.get("assignments_count") or 0) > 0
            )
            if still_present:
                ep_desc = target_ep or sname
                return {
                    "ok": False,
                    "still_running": True,
                    "message": f"停卡未生效：目标实例 {ep_desc} 仍处于计费状态，请重试或点停止全部",
                    "snapshot": snap,
                }

            with self._lock:
                if target_ep:
                    self._service_meta_by_ep.pop(target_ep, None)
                    self._runtime_secrets.pop(target_ep, None)
                    self._comfy_secrets_by_ep.pop(target_ep, None)
                    self._comfy_key_checked_eps.discard(target_ep)
                    cache = self.state.get("tunnel_cache")
                    if isinstance(cache, dict):
                        cache.pop(target_ep, None)
                    mb = self.state.get("managed_binding")
                    if isinstance(mb, dict) and str(mb.get("endpoint") or "") == target_ep:
                        self.state["managed_binding"] = None
                else:
                    self.state["managed_binding"] = None
                self._save_state()

            if not rem_eps and int(snap.get("assignments_count") or 0) == 0:
                self._reset_comfy_offline()

            msg = (
                f"已释放目标实例 ({sname} · 保留其余 {len(rem_eps)} 台会话)"
                if rem_eps
                else f"已释放 Colab 实例 ({sname})"
            )
            return {
                "ok": True,
                "released_endpoint": target_ep,
                "remaining_count": len(rem_eps),
                "message": msg,
                "snapshot": snap,
            }
        except Exception as exc:
            return {"ok": False, "message": f"停卡异常: {redact_secrets(str(exc))}"}
        finally:
            self._action_lock.release()

    def stop_all_instances(self) -> dict[str, Any]:
        """独立操作：停止当前账号下全部 Colab 活跃实例与孤儿运行时并核实彻底释放。"""
        if self.is_deploying():
            self.cancel_deploy()
            time.sleep(0.4)
        if not self._action_lock.acquire(blocking=False):
            return {"ok": False, "message": "另一项操作正在执行，请稍等片刻"}
        try:
            if not _ctl_available():
                return {"ok": False, "message": f"未找到控制脚本: {CTL_SCRIPT.name}"}
            with self._lock:
                start_epoch = self._account_epoch
            proc = self._runner.run(
                [*_ctl_cmd_prefix(), "down", "--all"],
                timeout=60,
                extra_env=self._build_ctl_env(),
                run_fn=_run_hidden,
                task_name="stop_all",
            )
            output = proc.combined_output
            with self._lock:
                if self._account_epoch != start_epoch:
                    return {"ok": False, "aborted_by_switch": True, "message": "操作期间账号已切换"}
            snap = self.fetch_live_status()
            rem_cnt = max(
                int(snap.get("assignments_count") or 0),
                len(snap.get("sessions") or []),
            )
            if not proc.ok:
                tail = (output.splitlines() or [proc.error_summary or "停机指令异常"])[-1]
                return {"ok": False, "message": f"停机失败: {tail}", "snapshot": snap}
            if rem_cnt > 0:
                return {
                    "ok": False,
                    "still_running": True,
                    "message": f"停止全部未完全生效：仍有 {rem_cnt} 台实例未释放，请重试",
                    "snapshot": snap,
                }
            with self._lock:
                self._service_meta_by_ep.clear()
                self._runtime_secrets.clear()
                self._comfy_secrets_by_ep.clear()
                self._comfy_key_checked_eps.clear()
                self.state["tunnel_cache"] = {}
                self.state["managed_binding"] = None
                self._save_state()
            self._reset_comfy_offline()
            return {"ok": True, "message": "已关闭全部 Colab 实例", "snapshot": snap}
        except Exception as exc:
            return {"ok": False, "message": f"停机异常: {redact_secrets(str(exc))}"}
        finally:
            self._action_lock.release()

    # =========================================================================
    # 功能 2.5：面板直接选卡一键开机、实时算力紧缺检测与 7 步进度直播
    # =========================================================================

    def _consume_deploy_line(self, raw_line: str) -> str | None:
        """委托 `colab_core.runtime.consume_deploy_line` 推进 7 步进度并返回隧道 URL。"""
        with self._lock:
            ctx = {
                "phase_idx": self._deploy_phase_idx,
                "phase_total": self._deploy_phase_total,
                "phase_prof": self._deploy_phase_prof,
                "phase_is_llm": self._deploy_phase_is_llm,
                "dl_done": self._deploy_dl_done,
                "dl_name": self._deploy_dl_name,
                "dl_total": self._deploy_dl_total,
                "captured_llm_tunnel": self._deploy_captured_llm_tunnel,
                "meta_warmed": self._deploy_meta_warmed,
            }
            tunnel_pat = self._tunnel_url_regex_pattern()
            tunnel_url, should_warm_meta = _core_consume_deploy_line(
                raw_line,
                self._deploy_state,
                ctx,
                self.state.get("prefs") or {},
                tunnel_pattern=tunnel_pat,
            )
            self._deploy_phase_idx = int(ctx["phase_idx"])
            self._deploy_phase_total = int(ctx["phase_total"])
            self._deploy_phase_prof = str(ctx["phase_prof"])
            self._deploy_phase_is_llm = bool(ctx["phase_is_llm"])
            self._deploy_dl_done = int(ctx["dl_done"])
            self._deploy_dl_name = str(ctx["dl_name"])
            self._deploy_dl_total = int(ctx["dl_total"])
            self._deploy_captured_llm_tunnel = ctx["captured_llm_tunnel"]
            self._deploy_meta_warmed = bool(ctx["meta_warmed"])

        if tunnel_url is not None and should_warm_meta:
            threading.Thread(target=lambda: self._fast_read_tunnel_url("", "comfy"), daemon=True).start()
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
        cohost_only_llm = bool(cfg.get("_cohost_only_llm")) and (llm_prof is not None) and (not comfy_profs)
        preserved_raw = cfg.get("_preserved_profiles")
        display_profiles = (
            _normalize_boot_profiles(preserved_raw, fallback_profile=profile)
            if cohost_only_llm and preserved_raw
            else list(profiles)
        )
        is_dual = bool(llm_prof and comfy_profs)
        effective_dual = bool(is_dual or cohost_only_llm)
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

        if not _ctl_available():
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

        pre = self.run_preflight(deep=False, profiles=profiles, gpu=gpu)
        if not pre.get("ok"):
            return {
                "ok": False,
                "preflight_failed": True,
                "preflight": pre,
                "message": str(pre.get("blocking_message") or "启动前环境检查未通过"),
            }

        if not self._deploy_lock.acquire(blocking=False):
            return {"ok": False, "message": "已有开机部署任务在运行中"}

        try:
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

                if cohost_only_llm:
                    if llm_prof == "qwen_27b":
                        prefs["boot_llm27_quant"] = llm27_quant
                        prefs["boot_llm27_ctx"] = llm27_ctx
                        prefs["boot_llm27_mmproj"] = llm27_mmproj
                    else:
                        prefs["boot_llm_quant"] = llm_quant
                        prefs["boot_llm_ctx"] = llm_ctx
                        prefs["boot_llm_mmproj"] = llm_mmproj
                else:
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
                self._deploy_dl_total = _core_calc_phase_dl_total(
                    first_phase,
                    llm_quant=llm_quant,
                    llm_mmproj=llm_mmproj,
                    llm27_mmproj=llm27_mmproj,
                    minimax_r2v=minimax_r2v,
                    minimax_pe=minimax_pe,
                )

                self._deploy_state = {
                    "active": True,
                    "profile": display_profiles[0],
                    "profiles": list(display_profiles),
                    "gpu": gpu,
                    "high_mem": high_mem,
                    "step_idx": 0,
                    "step_total": 7,
                    "stage_title": (f"热重载 LLM ({gpu})" if cohost_only_llm else f"准备开机 ({gpu})"),
                    "sub_detail": ("正在热重启推理引擎（保留现有实例与隧道）..." if cohost_only_llm else "正在启动本地无窗口部署进程..."),
                    "pct": 2,
                    "started_ts": now_ts,
                    "elapsed_sec": 0,
                    "error_msg": "",
                    "stockout_gpu": None,
                    "suggest_gpu": None,
                }
                initial_deploy = self.get_deploy_state()

            target_kind = "dual" if effective_dual else ("llm" if llm_prof else "comfy")
            if target_kind == "llm":
                self._stop_ws()
            with self._comfy_lock:
                self._comfy_state["service_kind"] = target_kind
                self._comfy_state["active_profiles"] = list(display_profiles)
                cstate_init = dict(self._comfy_state)
            with self._lock:
                self.last_snapshot["comfy"] = cstate_init

            with self._lock:
                start_epoch = self._account_epoch
            _, target_ep_up, _ = self._resolve_target_binding("comfy")
            if not target_ep_up:
                with self._lock:
                    cur_sessions = list(self.state.get("last_sessions") or [])
                matched_sess = next(
                    (
                        s
                        for s in cur_sessions
                        if s.get("endpoint") and str(s.get("accelerator") or "").upper() == gpu
                    ),
                    None,
                )
                if matched_sess is None and len(cur_sessions) == 1 and cur_sessions[0].get("endpoint"):
                    matched_sess = cur_sessions[0]
                if matched_sess:
                    target_ep_up = str(matched_sess["endpoint"])

            cmd = _core_build_ctl_up_args(
                python_exe=_hidden_python_exe(),
                ctl_script=CTL_SCRIPT,
                session="comfy",
                profiles=profiles,
                gpu=gpu,
                high_mem=high_mem,
                recreate=recreate,
                endpoint=target_ep_up,
                minimax_r2v=minimax_r2v,
                minimax_pe=minimax_pe,
                llm_quant=llm_quant,
                llm_ctx=llm_ctx,
                llm_mmproj=llm_mmproj,
                llm27_quant=llm27_quant,
                llm27_ctx=llm27_ctx,
                llm27_mmproj=llm27_mmproj,
                cohost_comfy=cohost_only_llm,
                ctl_prefix=_ctl_cmd_prefix(unbuffered=True),
            )
            profile_label = " + ".join(_DEPLOY_PROFILE_LABELS.get(p, p) for p in display_profiles)
            self._emit_event({"type": "deploy_progress", "deploy": initial_deploy})
        except Exception:
            self._deploy_lock.release()
            raise

        def _worker(cancel_tok: CancelToken) -> None:
            self._deploy_cancel_token = cancel_tok
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
                ctl_env = self._build_ctl_env()
                ctl_env["PYTHONUNBUFFERED"] = "1"
                # G4 等高配卡分配请求放宽至 75s，避免客户端早退留下未绑定孤儿实例
                ctl_env["COLAB_CLI_REQUEST_TIMEOUT"] = "75"
                ctl_env["COLAB_CLI_ASSIGN_RETRIES"] = "2"
                ctl_env["COLAB_CLI_RETRY_BACKOFF"] = "5"

                last_emit_m = 0.0

                def _on_deploy_line(sline: str) -> None:
                    nonlocal captured_tunnel, last_emit_m
                    tail_lines.append(sline)
                    if len(tail_lines) > 120:
                        del tail_lines[:-120]
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

                stream_res = self._runner.run_streaming(
                    cmd,
                    on_line=_on_deploy_line,
                    extra_env=ctl_env,
                    cancel_token=cancel_tok,
                    popen_fn=subprocess.Popen,
                    on_proc_start=lambda p: setattr(self, "_deploy_proc", p),
                    task_name="deploy_up",
                )
                rc = 0 if stream_res.ok else (stream_res.returncode or 1)
                if not stream_res.ok and stream_res.error and not tail_lines:
                    tail_lines.append(stream_res.error)

                cancelled = bool(self._deploy_cancelled or stream_res.cancelled or cancel_tok.is_cancelled)
                elapsed_total = max(
                    0, int(time.time()) - int(initial_deploy.get("started_ts") or time.time())
                )
                elapsed_cn = _core_format_elapsed_cn(elapsed_total)

                with self._lock:
                    if self._account_epoch != start_epoch:
                        self._deploy_state = _core_default_deploy_state()
                        return

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
                    if effective_dual:
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
                        if captured_tunnel and target_ep and not cohost_only_llm:
                            self.state.setdefault("tunnel_cache", {})[target_ep] = captured_tunnel
                            self._save_state()

                    if target_ep:
                        self._record_managed_binding("comfy", target_ep)

                    meta_ready, key_ready = self._get_ep_meta_and_key(target_ep or "")
                    comfy_key_ready = bool(target_ep and self._comfy_secrets_by_ep.get(target_ep))
                    if llm_prof or (target_kind in ("comfy", "dual") and not comfy_key_ready):
                        need_refresh = (
                            cohost_only_llm
                            or (not meta_ready)
                            or (llm_prof and not key_ready)
                            or (effective_dual and meta_ready.get("kind") != "dual")
                            or (target_kind in ("comfy", "dual") and not comfy_key_ready)
                        )
                        if need_refresh:
                            _, r_ep, fast_u = self._fast_read_tunnel_url(target_ep or "", "comfy")
                            if r_ep and not target_ep:
                                target_ep = r_ep
                                self._record_managed_binding("comfy", target_ep)
                            if fast_u and (not captured_tunnel or cohost_only_llm):
                                captured_tunnel = fast_u
                            meta_ready, _ = self._get_ep_meta_and_key(target_ep or "")
                    elif target_ep:
                        with self._lock:
                            self._service_meta_by_ep[target_ep] = {"kind": "comfy", "port": 8188}
                            self._runtime_secrets.pop(target_ep, None)

                    used_q = llm27_quant if llm_prof == "qwen_27b" else llm_quant
                    used_ctx = llm27_ctx if llm_prof == "qwen_27b" else llm_ctx
                    comfy_url_meta = (
                        str(meta_ready.get("comfy_tunnel_url") or (meta_ready.get("tunnel_url") if meta_ready.get("kind") in ("comfy", "dual") else "") or "").strip()
                        if isinstance(meta_ready, dict)
                        else ""
                    )
                    llm_url_final = (
                        self._deploy_captured_llm_tunnel
                        or (str(meta_ready.get("llm_tunnel_url") or "") if isinstance(meta_ready, dict) else "")
                        or (captured_tunnel if target_kind == "llm" else "")
                        or (self._read_cf_public_llm_url() if llm_prof else "")
                        or None
                    )
                    with self._comfy_lock:
                        self._comfy_state["service_kind"] = target_kind
                        self._comfy_state["active_profiles"] = list(display_profiles)
                        if cohost_only_llm and comfy_url_meta:
                            self._comfy_state["tunnel_url"] = comfy_url_meta
                        elif captured_tunnel and not cohost_only_llm:
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
                                self._comfy_state["gpu_name"] = _core_short_gpu_name(
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
                    if cohost_only_llm:
                        done_action = "open_llm_webui"
                        ctx_tag = "256K" if int(cstate.get("ctx_size") or used_ctx) >= 262144 else (
                            "128K" if int(cstate.get("ctx_size") or used_ctx) >= 131072 else f"{max(1, int((cstate.get('ctx_size') or used_ctx) // 1024))}K"
                        )
                        done_summary = f"🎉 {gpu} 推理已热重载至 {used_q}·{ctx_tag}({elapsed_cn})！"
                    elif effective_dual:
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
                    diag = _core_classify_deploy_failure(tail_lines, gpu, pre_assignments, profile_label, recreate)
                    stockout_gpu = diag.get("stockout_gpu")
                    if stockout_gpu:
                        with self._lock:
                            self._scarce_gpus[str(stockout_gpu).upper()] = {
                                "ts": int(time.time()),
                                "reason": str(diag.get("reason") or "算力紧缺"),
                                "kind": str(diag.get("kind") or "stockout"),
                            }
                    snap = self.fetch_live_status()
                    suggest_gpu = (
                        _core_recommend_fallback_gpu(
                            str(stockout_gpu), profile, snap.get("eligible_gpus"), snap.get("scarce_gpus") or {}, profiles=profiles
                        )
                        if stockout_gpu
                        else None
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
            finally:
                hb_stop.set()
                self._deploy_proc = None
                self._deploy_cancel_token = None
                self._deploy_lock.release()

        self._tasks.submit("deploy", _worker, exclusive=False)
        return {
            "ok": True,
            "started": True,
            "deploy": initial_deploy,
            "message": (f"正在热重载 {gpu} · {profile_label}..." if cohost_only_llm else f"开始部署 {gpu} · {profile_label}..."),
        }

    def reconfigure_running_llm(self, config: dict[str, Any] | None = None) -> dict[str, Any]:
        """在不释放实例且不断开同机 ComfyUI 的前提下，极速热重载运行中 Qwen3.8 的上下文/量化/识图参数。"""
        self.touch_activity()
        if not self._has_running_session():
            return {"ok": False, "message": "当前无在线实例，请在「选卡开机」直接部署"}
        if self.is_deploying() or self._deploy_lock.locked():
            return {"ok": False, "message": "已有部署或热重载任务进行中，请稍候"}

        cfg = config if isinstance(config, dict) else {}
        with self._lock:
            prefs = dict(self.state.get("prefs") or {})
            sessions = list(self.state.get("last_sessions") or [])
            target_ep = next(
                (str(s.get("endpoint") or "") for s in sessions if s.get("name") == "comfy" and s.get("endpoint")),
                str((sessions[0].get("endpoint") or "") if sessions else ""),
            )
            meta_ep = dict(self._service_meta_by_ep.get(target_ep) or {}) if target_ep else {}
            target_sess = next(
                (s for s in sessions if s.get("name") == "comfy"),
                sessions[0] if sessions else None,
            )
            run_gpu = str((target_sess or {}).get("accelerator") or prefs.get("boot_gpu") or "L4").strip().upper()
            if run_gpu not in SUPPORTED_BOOT_GPUS:
                run_gpu = "L4"

        with self._comfy_lock:
            cstate = dict(self._comfy_state)

        act_profs = _normalize_boot_profiles(cstate.get("active_profiles"), prefs.get("boot_profile") or "anima")
        pref_profs = _normalize_boot_profiles(prefs.get("boot_profiles"), prefs.get("boot_profile") or "anima")
        llm_prof = (
            (str(meta_ep.get("llm_profile") or "").strip() or None)
            or next((p for p in act_profs if p in LLM_BOOT_PROFILES), None)
            or next((p for p in pref_profs if p in LLM_BOOT_PROFILES), None)
        )
        if llm_prof not in LLM_BOOT_PROFILES:
            if "27b" in str(cstate.get("model_name") or "").lower():
                llm_prof = "qwen_27b"
            elif cstate.get("service_kind") in ("llm", "dual") or cstate.get("llm_online"):
                llm_prof = "qwen_llm"
            else:
                return {"ok": False, "message": "当前实例未运行 Qwen3.8 服务，请先到「选卡开机」部署"}

        is_27b = llm_prof == "qwen_27b"
        sk = str(cstate.get("service_kind") or meta_ep.get("kind") or "")
        comfy_profs_cur = [p for p in (act_profs + pref_profs) if p in COMFY_BOOT_PROFILES]
        is_dual_live = bool(
            sk == "dual"
            or cstate.get("comfy_online")
            or meta_ep.get("kind") == "dual"
            or (meta_ep.get("comfy_profiles") if isinstance(meta_ep.get("comfy_profiles"), list) else [])
            or bool(comfy_profs_cur and any(p in act_profs for p in COMFY_BOOT_PROFILES))
        )

        # 合并去重恢复完整的运行态工作流列表（例如 ["minimax", "qwen_27b"]）
        preserved_comfy: list[str] = []
        if is_dual_live:
            meta_cp = meta_ep.get("comfy_profiles") if isinstance(meta_ep.get("comfy_profiles"), list) else []
            for cp in [*meta_cp, *act_profs, *pref_profs]:
                if cp in COMFY_BOOT_PROFILES and cp not in preserved_comfy:
                    preserved_comfy.append(cp)
            if not preserved_comfy:
                preserved_comfy = ["anima"]
        preserved_profiles = [*preserved_comfy, llm_prof]

        default_quant = str(
            (prefs.get("boot_llm27_quant") if is_27b else prefs.get("boot_llm_quant"))
            or cstate.get("quant")
            or "UD-Q4_K_XL"
        ).strip()
        raw_quant = str(
            cfg.get("quant")
            or cfg.get("llm27_quant" if is_27b else "llm_quant")
            or default_quant
        ).strip()
        new_quant = raw_quant if raw_quant in SUPPORTED_LLM_QUANTS else "UD-Q4_K_XL"

        default_ctx = int(
            (prefs.get("boot_llm27_ctx") if is_27b else prefs.get("boot_llm_ctx"))
            or cstate.get("ctx_size")
            or (65536 if is_27b else 32768)
        )
        raw_ctx_val = cfg.get("ctx")
        if raw_ctx_val is None:
            raw_ctx_val = cfg.get("llm27_ctx" if is_27b else "llm_ctx")
        try:
            new_ctx = int(raw_ctx_val if raw_ctx_val is not None else default_ctx)
        except (TypeError, ValueError):
            new_ctx = default_ctx
        new_ctx = max(4096, min(262144, new_ctx))

        default_mm = bool(
            prefs.get("boot_llm27_mmproj" if is_27b else "boot_llm_mmproj", True)
        )
        raw_mm = cfg.get("mmproj")
        if raw_mm is None:
            raw_mm = cfg.get("llm27_mmproj" if is_27b else "llm_mmproj")
        new_mmproj = default_mm if raw_mm is None else bool(raw_mm)

        start_cfg: dict[str, Any] = {
            "profile": llm_prof,
            "profiles": [llm_prof],
            "gpu": run_gpu,
            "recreate": False,
            "_cohost_only_llm": is_dual_live,
            "_preserved_profiles": preserved_profiles,
            ("llm27_quant" if is_27b else "llm_quant"): new_quant,
            ("llm27_ctx" if is_27b else "llm_ctx"): new_ctx,
            ("llm27_mmproj" if is_27b else "llm_mmproj"): new_mmproj,
        }
        return self.start_instance(start_cfg)

    def cancel_deploy(self) -> dict[str, Any]:
        """主动中止正在运行的一键开机部署子进程。"""
        proc = self._deploy_proc
        tok = self._deploy_cancel_token
        if not self.is_deploying() and proc is None and not self._deploy_lock.locked():
            return {"ok": False, "message": "当前没有进行中的部署任务"}
        self._deploy_cancelled = True
        if tok is not None:
            tok.cancel()
        _core_kill_process_tree(proc)
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
                with self._comfy_http_get(
                    url, base_url=base_url, params=params, stream=True, timeout=(8, 60)
                ) as resp:
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
            r_stats = self._comfy_http_get(f"{base}/system_stats", base_url=base, timeout=4.0)
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
            gpu_name = _core_short_gpu_name(d0.get("name", ""))
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
            r_q = self._comfy_http_get(f"{base}/queue", base_url=base, timeout=4.0)
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
            r_hist = self._comfy_http_get(
                f"{base}/history", base_url=base, params={"max_items": 6}, timeout=5.0
            )
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
                            err_str = _core_extract_history_error(entry)
                            if err_str and pid not in self._reported_error_prompts:
                                self._reported_error_prompts.add(pid)
                                latest_err = (pid, err_str)
                            for meta in _core_collect_formal_outputs(entry):
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
                        cand_tps = _core_extract_slot_tps(s_item)
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

        gpu_name = _core_short_gpu_name(str(meta.get("gpu_name") or ""))
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

    def _poll_service_once(self, mode: str, comfy_url: str = "", llm_url: str = "", ep: str = "") -> dict[str, Any]:
        """统一单轮状态轮询（兼容 comfy / llm / dual），更新 _comfy_state 与快照并推送事件。"""
        with self._lock:
            start_epoch = self._account_epoch
        c_base = comfy_url.strip().rstrip("/") if comfy_url else ""
        l_base = llm_url.strip().rstrip("/") if llm_url else ""

        c_ok, events = (False, [])
        if mode in ("comfy", "dual") and c_base:
            c_ok, _, events = self._probe_comfy_status(c_base)
        l_ok, l_info = (False, {})
        if mode in ("llm", "dual") and l_base:
            l_ok, l_info = self._probe_llm_status(l_base, ep=ep)

        with self._lock:
            if self._account_epoch != start_epoch:
                return self.get_comfy_state()

        any_ok = c_ok if mode == "comfy" else (l_ok if mode == "llm" else (c_ok or l_ok))
        with self._comfy_lock:
            self._comfy_state["service_kind"] = mode
            self._comfy_state["comfy_online"] = c_ok
            self._comfy_state["llm_online"] = l_ok
            if mode == "comfy" and c_ok:
                self._comfy_state.update({
                    "llm_tunnel_url": None, "llm_running_count": 0,
                    "model_name": "", "quant": "", "ctx_size": 0, "tps": None,
                })
            elif mode == "llm":
                self._comfy_state["ws_connected"] = False
                if l_ok:
                    rc_l = int(l_info.get("running_count") or 0)
                    self._comfy_state.update({
                        "tunnel_url": l_base, "llm_tunnel_url": l_base,
                        "running_count": rc_l, "pending_count": 0, "llm_running_count": rc_l,
                        "step_value": 0, "step_max": 0, "step_pct": 0.0, "current_node": "",
                        "gpu_name": l_info.get("gpu_name", ""),
                        "vram_used_gb": l_info.get("vram_used_gb", 0.0),
                        "vram_total_gb": l_info.get("vram_total_gb", 0.0),
                        "vram_pct": l_info.get("vram_pct", 0.0),
                        "has_error": False, "error_summary": "",
                        "model_name": l_info.get("model_name", "Qwen3.8-Flash-Next-abliterated"),
                        "quant": l_info.get("quant", ""), "ctx_size": l_info.get("ctx_size", 0),
                        "tps": l_info.get("tps"),
                    })
            elif mode == "dual":
                if c_base:
                    self._comfy_state["tunnel_url"] = c_base
                if l_base:
                    self._comfy_state["llm_tunnel_url"] = l_base
                if not c_ok:
                    self._comfy_state.update({"ws_connected": False, "running_count": 0, "pending_count": 0})
                if not c_ok and l_ok and float(l_info.get("vram_total_gb") or 0.0) > 0:
                    self._comfy_state.update({
                        "gpu_name": l_info.get("gpu_name", ""),
                        "vram_used_gb": l_info.get("vram_used_gb", 0.0),
                        "vram_total_gb": l_info.get("vram_total_gb", 0.0),
                        "vram_pct": l_info.get("vram_pct", 0.0),
                    })
                if any_ok and l_ok:
                    self._comfy_state.update({
                        "llm_running_count": int(l_info.get("running_count") or 0),
                        "model_name": l_info.get("model_name", ""),
                        "quant": l_info.get("quant", ""),
                        "ctx_size": l_info.get("ctx_size", 0),
                        "tps": l_info.get("tps"),
                    })
                else:
                    self._comfy_state.update({"llm_running_count": 0, "tps": None})

            if any_ok:
                self._comfy_http_failures = 0
                self._comfy_state["online"] = True
            else:
                self._comfy_http_failures += 1
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

    def poll_comfy_http_once(self, tunnel_url: str) -> dict[str, Any]:
        return self._poll_service_once("comfy", comfy_url=tunnel_url)

    def poll_llm_http_once(self, tunnel_url: str, ep: str = "") -> dict[str, Any]:
        return self._poll_service_once("llm", llm_url=tunnel_url, ep=ep)

    def poll_dual_once(self, comfy_url: str, llm_url: str, ep: str = "") -> dict[str, Any]:
        return self._poll_service_once("dual", comfy_url=comfy_url, llm_url=llm_url, ep=ep)

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
                    comfy_key = self._get_comfy_key_for_url(base)
                    if not comfy_key and self._has_running_session():
                        try:
                            chk = self._http.get(f"{base}/system_stats", timeout=3.0)
                            if chk.status_code == 401:
                                comfy_key = self._hydrate_comfy_key_if_needed(base)
                        except Exception:
                            pass
                    ws_headers = [f"Authorization: Bearer {comfy_key}"] if comfy_key else None
                    app = websocket.WebSocketApp(ws_url, header=ws_headers, on_message=self._on_ws_message)
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

    def _send_colab_keepalive(self, target_ep: str = "") -> bool:
        """通过 Colab 官方通道执行一次无窗口三层保活心跳（支持定向 target_ep）。"""
        colab_py = self._get_colab_py()
        if not colab_py.exists():
            return False
        try:
            cmd = [str(colab_py), str(CLI_BRIDGE_SCRIPT), "keepalive"]
            if target_ep:
                cmd.append(str(target_ep))
            proc = _run_hidden(cmd, timeout=25)
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
        """按 10 分钟节流周期向绑定的活跃 Colab 实例发送防断连保活心跳（连续纯空闲超 45m 自动停发）。"""
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
            start_epoch = self._account_epoch

        _, target_ep, _ = self._resolve_target_binding("comfy")
        if not self._keepalive_lock.acquire(blocking=False):
            return

        # 先记录本次出发时间戳避免监控线程高频率重复拉起子进程
        with self._lock:
            if (
                self._account_epoch != start_epoch
                or (self._last_keepalive_ts > 0.0 and (cur_m - self._last_keepalive_ts) < KEEPALIVE_INTERVAL_SEC)
            ):
                self._keepalive_lock.release()
                return
            self._last_keepalive_ts = cur_m

        def _worker() -> None:
            ok = False
            try:
                ok = self._send_colab_keepalive(target_ep=target_ep)
            except Exception:
                ok = False
            finally:
                if not ok and not self._monitor_stop.is_set():
                    with self._lock:
                        # 若期间已切换账号则绝不回改新账号的心跳节流状态
                        if self._account_epoch == start_epoch and self._last_keepalive_ts == cur_m:
                            self._last_keepalive_ts = max(
                                1e-6,
                                cur_m - (KEEPALIVE_INTERVAL_SEC - KEEPALIVE_RETRY_COOLDOWN_SEC),
                            )
                self._keepalive_lock.release()

        threading.Thread(target=_worker, daemon=True).start()

    def _maybe_trigger_idle_auto_stop(self, now_m: float | None = None) -> None:
        """若确认远端服务在线且连续全空闲达设定阈值（默认 60m），强校验归档并定向停卡。"""
        if self.is_deploying() or self._upload_lock.locked() or self._action_lock.locked():
            return
        if not self._has_running_session():
            return

        cur_m = time.monotonic() if now_m is None else float(now_m)
        if cur_m < self._idle_archive_retry_after_m:
            return
        with self._lock:
            prefs = self.state.get("prefs") or {}
            idle_stop_mins = int(prefs.get("idle_auto_stop_minutes", 60) or 0)
            if idle_stop_mins <= 0:
                return
            policy = str(prefs.get("idle_archive_fail_policy") or "stop_billing").strip().lower()
            if policy not in ("stop_billing", "keep_alive"):
                policy = "stop_billing"
            if self._last_activity_m <= 0.0:
                self._last_activity_m = cur_m
                return
            last_act = self._last_activity_m
            cur_ep = str(self.state.get("active_endpoint") or "__active__")
            warned_ep = self._idle_warned_ep
            start_epoch = self._account_epoch
            bound_acct = str(prefs.get("active_account_id") or "").strip()

        sname, target_ep, _ = self._resolve_target_binding("comfy")
        if not target_ep:
            with self._lock:
                sess_list = list(self.state.get("last_sessions") or [])
            if len(sess_list) == 1 and sess_list[0].get("endpoint"):
                target_ep = str(sess_list[0]["endpoint"])

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

            def _idle_stop_worker(cancel_tok: CancelToken) -> None:
                try:
                    if (
                        self.is_deploying()
                        or self._upload_lock.locked()
                        or not self._has_running_session()
                    ):
                        return
                    with self._lock:
                        if self._account_epoch != start_epoch:
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

                    archive_needed = (not self.is_llm_mode_active()) and _ctl_available()
                    archive_ok = True
                    archive_err = ""
                    if archive_needed:
                        archive_ok = False
                        for attempt in (1, 2):
                            with self._lock:
                                if self._account_epoch != start_epoch:
                                    return
                            sync_res = self._runner.run(
                                [*_ctl_cmd_prefix(), "sync", "-s", sname],
                                timeout=120,
                                extra_env=self._build_ctl_env(),
                                cancel_token=cancel_tok,
                                run_fn=_run_hidden,
                                task_name="idle_sync",
                            )
                            out_txt = sync_res.combined_output
                            if sync_res.ok and (
                                "<<<SYNC_VERIFIED>>>" in out_txt or "SKIP_LLM_SYNC" in out_txt
                            ):
                                archive_ok = True
                                break
                            archive_err = sync_res.error_summary or f"sync 退出码 {sync_res.returncode}"
                            if attempt == 1:
                                time.sleep(2.0)

                    with self._lock:
                        if self._account_epoch != start_epoch:
                            return

                    if archive_needed and not archive_ok:
                        short_err = (redact_secrets(archive_err) or "云盘同步校验未通过")[:48]
                        if policy == "keep_alive":
                            with self._lock:
                                self._idle_archive_retry_after_m = time.monotonic() + 600.0
                                self._idle_warned_ep = None
                            self._emit_event(
                                {
                                    "type": "idle_archive_failed",
                                    "is_error": True,
                                    "summary": f"🚨 云盘归档重试失败（{short_err}）！已按「保图留机」保留实例，请尽快手动拉回并停卡",
                                    "action": "sync_outputs",
                                    "error_detail": redact_secrets(archive_err),
                                }
                            )
                            return

                        res = self.stop_session_instance(
                            session=sname,
                            endpoint=target_ep or None,
                            account_id=bound_acct,
                        )
                        if res.get("aborted_by_switch"):
                            return
                        if res.get("ok"):
                            self._emit_event(
                                {
                                    "type": "idle_stopped",
                                    "archive_failed": True,
                                    "is_error": True,
                                    "summary": f"⚠️ 云盘归档重试仍失败（{short_err}），已按策略优先停卡停止计费！",
                                    "snapshot": res.get("snapshot") or self.get_cached_snapshot(),
                                }
                            )
                        else:
                            fail_msg = redact_secrets(str(res.get("message") or "请立即手动停卡"))
                            self._diag.error("idle_stop", f"归档与闲置自动停卡均失败: {fail_msg}")
                            with self._lock:
                                self._idle_warned_ep = None
                                self._idle_archive_retry_after_m = time.monotonic() + 60.0
                            self._emit_event(
                                {
                                    "type": "idle_warning",
                                    "is_error": True,
                                    "summary": f"🚨 归档与自动停卡均失败：{fail_msg}",
                                    "action": "keep_working",
                                }
                            )
                        return

                    res = self.stop_session_instance(
                        session=sname,
                        endpoint=target_ep or None,
                        account_id=bound_acct,
                    )
                    if res.get("aborted_by_switch"):
                        return
                    if res.get("ok"):
                        self._emit_event(
                            {
                                "type": "idle_stopped",
                                "archive_failed": False,
                                "summary": f"🛑 连续闲置满 {idle_stop_mins} 分钟，已归档并释放当前实例省积分",
                                "snapshot": res.get("snapshot") or self.get_cached_snapshot(),
                            }
                        )
                    else:
                        fail_msg = redact_secrets(str(res.get("message") or "请手动停卡"))
                        self._diag.error("idle_stop", f"闲置自动停卡未成功: {fail_msg}")
                        with self._lock:
                            self._idle_warned_ep = None
                            self._idle_archive_retry_after_m = time.monotonic() + 60.0
                        self._emit_event(
                            {
                                "type": "idle_warning",
                                "is_error": True,
                                "summary": f"⚠️ 闲置自动停卡未成功: {fail_msg}",
                                "action": "keep_working",
                            }
                        )
                finally:
                    self._idle_stop_lock.release()

            hdl = self._tasks.submit("idle_stop", _idle_stop_worker, exclusive=False)
            if not hdl.accepted:
                self._idle_stop_lock.release()
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
        if self._deploy_proc is not None or self.is_deploying():
            self.cancel_deploy()

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
            raise RuntimeError(redact_secrets(tail_lines[-1] if tail_lines else f"rclone 退出码 {rc}"))

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
            [str(colab_py), str(CLI_BRIDGE_SCRIPT), "contents-put", ep, sname, rel_path],
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
            err = redact_secrets((proc.stderr or proc.stdout or "").strip()).splitlines()
            raise RuntimeError(err[-1] if err else "Contents 写入失败")

    def _fetch_comfy_object_info(self, tunnel_url: str) -> dict[str, Any] | None:
        """从在线 ComfyUI 拉取节点定义（/object_info），用于将 API prompt 还原为标准画布工作流。"""
        base = tunnel_url.rstrip("/")
        try:
            resp = self._comfy_http_get(f"{base}/object_info", base_url=base, timeout=10)
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
            resp = self._comfy_http_post(
                target_ep,
                base_url=base,
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
        resp: requests.Response | None = None
        for attempt in (1, 2):
            with _StreamingMultipartFile(
                file_path,
                field_name="image",
                fields={"type": "input", "overwrite": "true"},
                on_progress=lambda pct: self._emit_upload_progress(file_path.name, pct, "传参考图"),
            ) as stream:
                resp = self._comfy_http_post(
                    url,
                    base_url=base,
                    retry_on_401=False,
                    data=stream,
                    headers={
                        "Content-Type": stream.content_type,
                        "Content-Length": str(len(stream)),
                    },
                    timeout=(10, 300),
                )
            if resp.status_code == 401 and attempt == 1 and self._has_running_session():
                had_key = bool(self._get_comfy_key_for_url(base))
                try:
                    resp.close()
                except Exception:
                    pass
                new_key = self._hydrate_comfy_key_if_needed(base, force=(not had_key))
                if new_key:
                    continue
            resp.raise_for_status()
            break
        assert resp is not None
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
                err = redact_secrets((proc.stderr or proc.stdout or "").strip()).splitlines()
                raise RuntimeError(err[-1] if err else "实例拉取 LoRA 失败")

        if tunnel_url:
            try:
                base_u = tunnel_url.rstrip("/")
                self._comfy_http_get(f"{base_u}/object_info/LoraLoader", base_url=base_u, timeout=5)
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

        def _worker(_cancel_tok: CancelToken) -> None:
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
                self._diag.error("upload", f"投递失败: {redact_secrets(str(exc))}")
                self._emit_event(
                    {
                        "type": "upload_error",
                        "summary": f"⚠️ 投递失败: {redact_secrets(str(exc))}",
                    }
                )
            finally:
                self._upload_lock.release()

        hdl = self._tasks.submit("upload_files", _worker, exclusive=False)
        if not hdl.accepted:
            self._upload_lock.release()
            return {"ok": False, "message": hdl.reason or "上传任务忙"}
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
            proc = _run_hidden([str(colab_py), str(CLI_BRIDGE_SCRIPT), "oauth-start"], timeout=20)
            if proc.returncode != 0:
                err = redact_secrets((proc.stderr or proc.stdout or "").strip()).splitlines()
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
            register_ephemeral_secret(verifier)
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
            return {"ok": False, "message": f"拉起 Google 授权失败: {redact_secrets(str(exc))}"}

    def cancel_colab_oauth(self) -> dict[str, Any]:
        """取消进行中的 Google OAuth2 绑定流程并清理内存临时校验码。"""
        with self._lock:
            self._pending_oauth_verifier = ""
            self._pending_oauth_ts = 0.0
        snap = self._rebuild_and_emit_snapshot()
        return {"ok": True, "snapshot": snap, "message": "已取消本次账号绑定"}

    def complete_colab_oauth(self, raw_code: str, label: str = "") -> dict[str, Any]:
        """用授权码换取 Token 并安全写入系统凭据库，自动建档并切换至新绑定账号。"""
        if (
            self.is_deploying()
            or self._deploy_lock.locked()
            or self._upload_lock.locked()
            or self._idle_stop_lock.locked()
        ):
            return {
                "ok": False,
                "busy": True,
                "message": "当前仍有后台任务（部署/上传/自动停卡）正在执行，请等待结束后再绑定新账号",
            }
        colab_py = self._get_colab_py()
        if not colab_py.is_file():
            return {"ok": False, "message": "未检测到 colab-cli 运行环境"}

        register_ephemeral_secret(raw_code)
        code = _core_extract_oauth_code(raw_code)
        register_ephemeral_secret(code)
        if len(code) < 8:
            return {"ok": False, "message": "请粘贴有效的 Google 授权码"}

        with self._lock:
            verifier = self._pending_oauth_verifier
            pending_ts = self._pending_oauth_ts
            prev_acct_id = str((self.state.get("prefs") or {}).get("active_account_id") or "").strip()

        if not verifier or (pending_ts > 0 and (time.monotonic() - pending_ts) > 1200.0):
            return {"ok": False, "message": "授权会话已超时或未启动，请重新点「新增 Google 账号」"}
        register_ephemeral_secret(verifier)

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
                [str(colab_py), str(CLI_BRIDGE_SCRIPT), "oauth-complete"],
                timeout=35,
                input_text=payload,
            )
            if proc.returncode != 0:
                err = redact_secrets((proc.stderr or proc.stdout or "").strip()).splitlines()
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
            return {"ok": False, "message": f"绑定授权码异常: {redact_secrets(str(exc))}"}

        # 归档旧账号账本 -> 激活新账号槽位与独立账本 -> 递增账号 epoch 并清空旧账号内存态密钥
        with self._lock:
            self._account_epoch += 1
            self._pending_oauth_verifier = ""
            self._pending_oauth_ts = 0.0
            self._auth_missing = False
            self._auth_expired = False
            if prev_acct_id and prev_acct_id != acct_id:
                self.state.setdefault("account_ledgers", {})[prev_acct_id] = _core_snapshot_current_ledger(self.state)
            _core_restore_ledger_for_account(self.state, acct_id)
            prefs = self.state.setdefault("prefs", dict(DEFAULT_PREFS))
            prefs["active_account_id"] = acct_id
            card = _core_upsert_account_profile(
                self.state,
                acct_id,
                email_masked,
                preferred_label=label,
            )
            self._service_meta_by_ep.clear()
            self._runtime_secrets.clear()
            self._comfy_secrets_by_ep.clear()
            self._comfy_key_checked_eps.clear()
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
        """秒切到凭据保险箱中的另一组 Google Ultra 账号，完整隔离账本与 sessions.json，并在有后台任务时安全拦截。"""
        if (
            self.is_deploying()
            or self._deploy_lock.locked()
            or self._upload_lock.locked()
            or self._idle_stop_lock.locked()
        ):
            return {
                "ok": False,
                "busy": True,
                "message": "当前账号仍有后台任务（部署/上传/自动停卡）正在执行，请等待完成或先中止后再切换账号",
            }
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
            return {
                "ok": False,
                "busy": True,
                "message": "当前账号仍有后台运维操作正在执行，请稍等片刻再切换",
            }
        try:
            if (
                self.is_deploying()
                or self._deploy_lock.locked()
                or self._upload_lock.locked()
                or self._idle_stop_lock.locked()
            ):
                return {
                    "ok": False,
                    "busy": True,
                    "message": "当前账号仍有后台任务正在执行，请等待完成后再切换账号",
                }
            payload = json.dumps(
                {
                    "op": "switch",
                    "target_acct_id": target_id,
                    "prev_acct_id": prev_id,
                },
                ensure_ascii=False,
            )
            proc = _run_hidden(
                [str(colab_py), str(CLI_BRIDGE_SCRIPT), "account-op"],
                timeout=25,
                input_text=payload,
            )
            if proc.returncode != 0:
                err = redact_secrets((proc.stderr or proc.stdout or "").strip()).splitlines()
                return {"ok": False, "message": f"切换凭据失败: {err[-1] if err else proc.returncode}"}
        finally:
            self._action_lock.release()

        with self._lock:
            self._account_epoch += 1
            self._auth_missing = False
            self._auth_expired = False
            if prev_id and prev_id != target_id:
                self.state.setdefault("account_ledgers", {})[prev_id] = _core_snapshot_current_ledger(self.state)
            _core_restore_ledger_for_account(self.state, target_id)
            prefs = self.state.setdefault("prefs", dict(DEFAULT_PREFS))
            prefs["active_account_id"] = target_id
            self._service_meta_by_ep.clear()
            self._runtime_secrets.clear()
            self._comfy_secrets_by_ep.clear()
            self._comfy_key_checked_eps.clear()
            self._idle_archive_retry_after_m = 0.0
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
        if self.is_deploying() or self._deploy_lock.locked():
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
                _run_hidden([str(colab_py), str(CLI_BRIDGE_SCRIPT), "account-op"], timeout=20, input_text=payload)
            except Exception:
                pass

        with self._lock:
            self.state["account_profiles"] = remaining
            ledgers = self.state.get("account_ledgers")
            if isinstance(ledgers, dict):
                ledgers.pop(target_id, None)
            if removing_active and not remaining:
                self._account_epoch += 1
                self.state.setdefault("prefs", dict(DEFAULT_PREFS))["active_account_id"] = ""
                _core_restore_ledger_for_account(self.state, "")
                self._auth_missing = True
                self._auth_expired = False
                self._service_meta_by_ep.clear()
                self._runtime_secrets.clear()
                self._comfy_secrets_by_ep.clear()
                self._comfy_key_checked_eps.clear()
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
        free_gb, total_gb = _core_query_gdrive_quota(
            self._get_rclone_exe(),
            self._get_rclone_conf(),
            run_fn=lambda cmd, timeout=15: _run_hidden(cmd, timeout=timeout),
        )
        if free_gb is not None or total_gb is not None:
            with self._lock:
                self.state["onboarding_cache"] = {
                    "gdrive_free_gb": free_gb,
                    "gdrive_total_gb": total_gb,
                    "gdrive_checked_ts": int(time.time()),
                }
                self._save_state()
        return free_gb, total_gb

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

        def _wait_worker(_cancel_tok: CancelToken) -> None:
            rc = 1
            out_txt = ""
            try:
                out_txt, _ = proc.communicate(timeout=180)
                rc = proc.returncode
            except subprocess.TimeoutExpired:
                _core_kill_process_tree(proc)
                out_txt = "浏览器授权等待超时（3 分钟）"
            except Exception as exc:
                out_txt = str(exc)
                self._diag.error("gdrive_bind", f"云盘授权异常: {redact_secrets(out_txt)}")
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

            if rc == 0 and _core_has_gdrive_save_section(rclone_conf):
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
                tail = redact_secrets((out_txt.strip().splitlines() or [f"退出码 {rc}"])[-1])
                snap = self._rebuild_and_emit_snapshot()
                self._emit_event(
                    {
                        "type": "onboarding_event",
                        "ok": False,
                        "summary": f"⚠️ 云盘授权未完成: {tail[:42]}",
                        "snapshot": snap,
                    }
                )

        self._tasks.submit("gdrive_bind", _wait_worker, exclusive=False)
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
            _core_kill_process_tree(proc)
        snap = self._rebuild_and_emit_snapshot()
        return {"ok": True, "snapshot": snap, "message": "已中止等待云盘授权"}

    def import_rclone_conf(self, src_path: str) -> dict[str, Any]:
        """导入用户已有的 rclone.conf 并校验/适配 [gdrive_save] 云盘配置。"""
        ok, target_p, err_msg = _core_validate_import_rclone(src_path, package_dir=PACKAGE_DIR)
        if not ok or target_p is None:
            return {"ok": False, "message": err_msg or "导入 rclone.conf 失败"}

        with self._lock:
            self.state.setdefault("prefs", dict(DEFAULT_PREFS))["rclone_conf_path"] = str(target_p)
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
        """将 Cloudflare 隧道 Token 与固定密钥存入系统凭据库，配置文件仅存域名与引用。"""
        cfg_in = config if isinstance(config, dict) else {}
        tok_raw = str(cfg_in.get("tunnel_token") or "").strip()
        key_raw = str(cfg_in.get("llm_api_key") or "").strip()
        if tok_raw:
            register_ephemeral_secret(tok_raw)
        if key_raw:
            register_ephemeral_secret(key_raw)

        try:
            save_cf_tunnel_settings(
                self._resolve_cf_config_path(),
                comfy_url=str(cfg_in.get("comfy_url") or ""),
                llm_url=str(cfg_in.get("llm_url") or ""),
                tunnel_token=tok_raw,
                llm_api_key=key_raw,
                colab_py=self._get_colab_py(),
            )
        except Exception as exc:
            return {"ok": False, "message": redact_secrets(str(exc))}

        snap = self._rebuild_and_emit_snapshot()
        return {"ok": True, "snapshot": snap, "message": "✅ 已启用 Cloudflare 固定域名隧道"}

    def clear_cf_tunnel_config(self) -> dict[str, Any]:
        """清除固定命名隧道凭据与配置，切回免费即用的 trycloudflare 临时隧道。"""
        try:
            clear_cf_tunnel_settings(
                self._resolve_cf_config_path(),
                colab_py=self._get_colab_py(),
            )
        except OSError as exc:
            return {"ok": False, "message": f"切回失败: {redact_secrets(str(exc))}"}
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

    def get_task_snapshot(self) -> dict[str, dict[str, Any]]:
        """返回所有受控后台任务的最新状态快照。"""
        return self._tasks.snapshot()

    def run_preflight(
        self,
        deep: bool = False,
        profiles: list[str] | str | None = None,
        gpu: str | None = None,
    ) -> dict[str, Any]:
        """执行启动前多维度环境、授权、磁盘、配置与显卡兼容性自检。"""
        with self._lock:
            prefs = dict(self.state.get("prefs") or {})
            oc = dict(self.state.get("onboarding_cache") or {})
        eff_profiles = profiles if profiles is not None else (
            prefs.get("boot_profiles") or prefs.get("boot_profile") or ["anima"]
        )
        eff_gpu = gpu or prefs.get("boot_gpu")
        tool_paths = ToolPaths(
            colab_exe=self._get_colab_exe(),
            colab_py=self._get_colab_py(),
            rclone_exe=self._get_rclone_exe(),
            rclone_conf=self._get_rclone_conf(),
        )
        free_gb = oc.get("gdrive_free_gb")
        if deep and free_gb is None:
            free_gb, _ = self._query_gdrive_quota_locked()
        report = run_preflight_checks(
            mode="deep" if deep else "fast",
            profiles=eff_profiles,
            gpu=eff_gpu,
            tool_paths=tool_paths,
            state_path=self.state_path,
            download_dir=self.download_root,
            cf_config_path=self._resolve_cf_config_path(),
            pkg_dir=PACKAGE_DIR,
            gdrive_free_gb=free_gb,
        )
        res = report.to_dict()
        res["blocking_message"] = report.first_blocking_message()
        return res

    def export_diagnostics(self, out_dir: Path | None = None) -> dict[str, Any]:
        """生成全量脱敏的排障诊断 `.zip` 压缩包，方便定位异常。"""
        try:
            with self._lock:
                state_copy = dict(self.state)
                snap_copy = dict(self.last_snapshot)
            tool_paths = ToolPaths(
                colab_exe=self._get_colab_exe(),
                colab_py=self._get_colab_py(),
                rclone_exe=self._get_rclone_exe(),
                rclone_conf=self._get_rclone_conf(),
            )
            zip_path = export_diagnostic_bundle(
                out_dir=out_dir,
                pkg_dir=PACKAGE_DIR,
                state_path=self.state_path,
                state_dict=state_copy,
                snapshot=snap_copy,
                task_manager=self._tasks,
                diag_logger=self._diag,
                cf_config_path=self._resolve_cf_config_path(),
                tool_paths=tool_paths,
                deep_preflight=False,
            )
            is_sandbox_log = bool(str(os.environ.get("COLAB_PET_DIAG_LOG_PATH") or "").strip())
            if os.name == "nt" and out_dir is None and not is_sandbox_log:
                try:
                    subprocess.Popen(
                        ["explorer.exe", f"/select,{zip_path.resolve()}"],
                        creationflags=CREATE_NO_WINDOW,
                    )
                except Exception:
                    pass
            return {
                "ok": True,
                "path": str(zip_path),
                "message": f"🩺 已导出脱敏诊断包: {zip_path.name}",
            }
        except Exception as exc:
            msg = redact_secrets(str(exc))
            self._diag.error("diagnostics", f"导出诊断包失败: {msg}")
            return {"ok": False, "message": f"导出诊断包失败: {msg}"}

    def open_logs_dir(self) -> dict[str, Any]:
        """创建并打开桌宠诊断日志文件夹。"""
        log_dir = self._diag.log_path.parent
        log_dir.mkdir(parents=True, exist_ok=True)
        is_sandbox_log = bool(str(os.environ.get("COLAB_PET_DIAG_LOG_PATH") or "").strip())
        if not is_sandbox_log:
            try:
                os.startfile(str(log_dir))
            except Exception as exc:
                return {"ok": False, "message": f"打开日志目录失败: {redact_secrets(str(exc))}"}
        return {"ok": True, "path": str(log_dir), "message": "📄 已打开诊断日志文件夹"}


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
