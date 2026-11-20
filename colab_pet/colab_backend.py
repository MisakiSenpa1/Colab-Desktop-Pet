"""Colab 桌宠后端数据采集与控制模块。

负责汇总本地 `google-colab-cli` 与 `colab_ctl.py` 状态、管理核心持久化与内存态凭据映射，
并将监控账本、一键部署、隧道直铺、Comfy/LLM 同步与多账号保险箱编排委托给各自正交子服务。
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import ctypes
import functools
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.parse
import webbrowser
from collections.abc import Callable
from pathlib import Path
from typing import Any

import requests

try:
    import websocket
except Exception:
    websocket = None

def _has_frozen_meipass() -> bool:
    """判断当前是否运行于带 `_MEIPASS` 资源目录的 PyInstaller 冻结发行包中。"""
    return bool(getattr(sys, "frozen", False) and getattr(sys, "_MEIPASS", None))


def _resolve_backend_paths() -> tuple[Path, Path, Path, Path, Path, Path, Path]:
    """解析静态资源目录与用户可写运行态路径（冻结包将可写状态锚定在 exe 外层目录，避免写入 `_internal`）。"""
    if _has_frozen_meipass():
        pkg_dir = Path(sys._MEIPASS).resolve()
        pet_candidate = (pkg_dir / "colab_pet").resolve()
        pet_dir = pet_candidate if pet_candidate.is_dir() else Path(__file__).resolve().parent
        runtime_dir = Path(sys.executable).resolve().parent
        return (
            pet_dir,
            pkg_dir,
            runtime_dir,
            runtime_dir / "_downloads" / "all",
            pkg_dir / "colab_ctl.py",
            runtime_dir / "pet_state.json",
            runtime_dir / ".cf_tunnel_secret.json",
        )

    pet_dir = Path(__file__).resolve().parent
    pkg_dir = pet_dir.parent
    workspace_root = pkg_dir.parent
    return (
        pet_dir,
        pkg_dir,
        workspace_root,
        workspace_root / "_downloads" / "all",
        pkg_dir / "colab_ctl.py",
        pet_dir / "pet_state.json",
        pkg_dir / ".cf_tunnel_secret.json",
    )


(
    PET_DIR,
    PACKAGE_DIR,
    WORKSPACE_ROOT,
    LOCAL_DOWNLOAD_ROOT,
    CTL_SCRIPT,
    STATE_FILE,
    _CF_SECRET_FILE,
) = _resolve_backend_paths()
_INIT_STATE_FILE = STATE_FILE
_INIT_LOCAL_DOWNLOAD_ROOT = LOCAL_DOWNLOAD_ROOT
_INIT_CF_SECRET_FILE = _CF_SECRET_FILE

if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))
if str(PET_DIR) not in sys.path:
    sys.path.insert(0, str(PET_DIR))

from colab_core.accounts import (  # noqa: E402
    CLI_BRIDGE_SCRIPT,
    DEFAULT_FALLBACK_COLAB_EXE,
    resolve_colab_exe as _core_discover_colab_exe,
    resolve_colab_py as _core_discover_colab_py,
    resolve_rclone_conf as _core_discover_rclone_conf,
    resolve_rclone_exe as _core_discover_rclone_exe,
    sanitize_account_label as _sanitize_account_label,
)
from colab_core.colab_client import (  # noqa: E402
    COLAB_REMOTE_LORA_PULL_SCRIPT as _COLAB_REMOTE_LORA_PULL_SCRIPT,
    detect_tunnel_kind as _core_detect_tunnel_kind,
    is_allowed_tunnel_url as _core_is_allowed_tunnel_url,
    probe_tunnel_alive as _core_probe_tunnel_alive,
)
from colab_core.file_sync import (  # noqa: E402
    GDRIVE_LORA_REMOTE,
    sanitize_filename as _sanitize_filename,
)
from colab_core.runtime import (  # noqa: E402
    COMFY_PROFILES as COMFY_BOOT_PROFILES,
    CancelToken,
    DEFAULT_MEASURED_GPU_RATES,
    DEFAULT_PREFS,
    LLM_PROFILES as LLM_BOOT_PROFILES,
    ProcessRunner,
    SCARCE_GPU_TTL_SEC,
    TaskManager,
    clean_subprocess_env as _clean_env,
    default_deploy_state as _core_default_deploy_state,
    get_diag_logger,
    load_pet_state_dict as _core_load_pet_state_dict,
    normalize_profiles as _normalize_boot_profiles,
    sanitize_prefs_dict as _sanitize_prefs_dict,
    suppress_logged,
)
from colab_remote_common import load_pet_bridge_init_py as _load_bridge_node_py  # noqa: E402
from colab_security import load_cf_tunnel_settings, redact_secrets, url_origin_key  # noqa: E402

if __package__:
    from .backend_accounts import AccountVaultService  # noqa: E402
    from .backend_agent import AgentService  # noqa: E402
    from .backend_cluster import ClusterPoolService  # noqa: E402
    from .backend_comfy import ComfySyncService  # noqa: E402
    from .backend_deploy import DeployOrchestrator  # noqa: E402
    from .backend_monitor import SessionMonitorService  # noqa: E402
    from .backend_studio import StudioService  # noqa: E402
    from .backend_tunnel import TunnelService  # noqa: E402
    from .context import BackendContext, EventBus  # noqa: E402
    from .secret_store import EphemeralSecretStore  # noqa: E402
    from .state_store import AppStateStore  # noqa: E402
else:
    from backend_accounts import AccountVaultService  # noqa: E402
    from backend_agent import AgentService  # noqa: E402
    from backend_cluster import ClusterPoolService  # noqa: E402
    from backend_comfy import ComfySyncService  # noqa: E402
    from backend_deploy import DeployOrchestrator  # noqa: E402
    from backend_monitor import SessionMonitorService  # noqa: E402
    from backend_studio import StudioService  # noqa: E402
    from backend_tunnel import TunnelService  # noqa: E402
    from context import BackendContext, EventBus  # noqa: E402
    from secret_store import EphemeralSecretStore  # noqa: E402
    from state_store import AppStateStore  # noqa: E402

_COMFY_BRIDGE_NODE_INIT_PY = _load_bridge_node_py(PACKAGE_DIR)

def _discover_colab_exe(custom_path: str = "") -> Path | None:
    """探测本机可用的 colab.exe 路径。"""
    return _core_discover_colab_exe(custom_path, allow_fallback=False)


def _discover_colab_py(custom_path: str = "", colab_exe: Path | None = None) -> Path:
    """探测 google-colab-cli 托管的 Python 解释器路径。"""
    return _core_discover_colab_py(custom_path, colab_exe=colab_exe)


def _discover_rclone_exe(custom_path: str = "") -> Path:
    """探测本机可用的 rclone.exe 路径。"""
    return _core_discover_rclone_exe(custom_path)


def _discover_rclone_conf(custom_path: str = "", rclone_exe: Path | None = None) -> Path:
    """探测本机 rclone.conf 配置文件路径。"""
    return _core_discover_rclone_conf(custom_path, rclone_exe=rclone_exe)


# 使用通用 Path.home() 默认占位后再执行跨设备探测，避免任何硬编码私人路径
COLAB_EXE_DEFAULT = _discover_colab_exe() or DEFAULT_FALLBACK_COLAB_EXE
COLAB_PY = _discover_colab_py(colab_exe=COLAB_EXE_DEFAULT)
RCLONE_EXE = _discover_rclone_exe()
RCLONE_CONF = _discover_rclone_conf(rclone_exe=RCLONE_EXE)

_INIT_COLAB_PY = COLAB_PY
_INIT_COLAB_EXE = COLAB_EXE_DEFAULT
_INIT_RCLONE_EXE = RCLONE_EXE
_INIT_RCLONE_CONF = RCLONE_CONF

CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

KEEPALIVE_INTERVAL_SEC = 600.0
KEEPALIVE_RETRY_COOLDOWN_SEC = 120.0
KEEPALIVE_MAX_IDLE_SEC = 45 * 60.0
_PRESERVE_ORIGINAL_SUB_RE = re.compile(r"(?:^|/)drama3_voices$", re.IGNORECASE)
_TOOL_PATH_PREF_KEYS = frozenset(
    {"colab_exe_path", "colab_py_path", "rclone_exe_path", "rclone_conf_path"}
)
_BACKEND_MOD = sys.modules[__name__]

__all__ = [
    "CLI_BRIDGE_SCRIPT",
    "COLAB_EXE_DEFAULT",
    "COLAB_PY",
    "CTL_SCRIPT",
    "ColabPetBackend",
    "DEFAULT_MEASURED_GPU_RATES",
    "DEFAULT_PREFS",
    "KEEPALIVE_INTERVAL_SEC",
    "KEEPALIVE_MAX_IDLE_SEC",
    "KEEPALIVE_RETRY_COOLDOWN_SEC",
    "LOCAL_DOWNLOAD_ROOT",
    "PACKAGE_DIR",
    "PET_DIR",
    "RCLONE_CONF",
    "RCLONE_EXE",
    "STATE_FILE",
    "WORKSPACE_ROOT",
    "_COMFY_BRIDGE_NODE_INIT_PY",
    "_PRESERVE_ORIGINAL_SUB_RE",
    "_build_explorer_select_cmdline",
    "_ctl_available",
    "_ctl_cmd_prefix",
    "_detect_tunnel_kind",
    "_hidden_python_exe",
    "_is_allowed_tunnel_url",
    "_load_cf_secret_raw",
    "_normalize_boot_profiles",
    "_probe_tunnel_alive",
    "_redact_payload",
    "_reveal_path_in_explorer",
    "_run_hidden",
    "subprocess",
    "sys",
    "threading",
    "time",
    "webbrowser",
    "websocket",
]


def _default_cf_secret_path() -> Path:
    """解析默认 Cloudflare 固定隧道配置文件路径（兼容模块覆盖、沙箱环境变量与 PyInstaller 冻结布局）。"""
    if _CF_SECRET_FILE != _INIT_CF_SECRET_FILE:
        return _CF_SECRET_FILE
    env_p = str(os.environ.get("COLAB_PET_CF_CONFIG_PATH") or "").strip()
    if env_p:
        return Path(env_p)
    if _has_frozen_meipass():
        return Path(sys.executable).resolve().parent / ".cf_tunnel_secret.json"
    return _CF_SECRET_FILE


def _load_cf_secret_raw(config_path: Path | None = None, *, include_secrets: bool = False) -> dict[str, Any]:
    target_p = Path(config_path) if config_path is not None else _default_cf_secret_path()
    return load_cf_tunnel_settings(target_p, include_secrets=include_secrets)


def _is_allowed_tunnel_url(url: str, config_path: Path | None = None) -> bool:
    cfg_p = config_path or _default_cf_secret_path()
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


def _build_explorer_select_cmdline(target: Path) -> str:
    """构造 Windows Explorer 精确选中文件的安全命令行（`/select,` 严禁被放进双引号内，否则含空格路径会退回打开“文档”）。"""
    safe_path = str(target.resolve()).replace('"', "")
    return f'explorer.exe /select,"{safe_path}"'


def _reveal_path_in_explorer(target: Path) -> None:
    """在系统文件管理器中打开目录或精确高亮定位文件（优先 Win32 SHOpenFolderAndSelectItems，绕开命令行转义陷阱）。"""
    resolved = target.resolve()
    if resolved.is_dir():
        with suppress_logged("explorer", OSError, RuntimeError, throttled_key="startfile_dir"):
            os.startfile(str(resolved))
        return

    if os.name == "nt" and resolved.exists():
        with suppress_logged("explorer", throttled_key="shell_select"):
            shell32 = ctypes.windll.shell32
            ole32 = ctypes.windll.ole32
            shell32.ILCreateFromPathW.argtypes = [ctypes.c_wchar_p]
            shell32.ILCreateFromPathW.restype = ctypes.c_void_p
            shell32.SHOpenFolderAndSelectItems.argtypes = [
                ctypes.c_void_p,
                ctypes.c_uint,
                ctypes.c_void_p,
                ctypes.c_ulong,
            ]
            shell32.SHOpenFolderAndSelectItems.restype = ctypes.c_long
            shell32.ILFree.argtypes = [ctypes.c_void_p]
            shell32.ILFree.restype = None
            ole32.CoInitialize(None)
            try:
                pidl = shell32.ILCreateFromPathW(str(resolved))
                if pidl:
                    try:
                        if shell32.SHOpenFolderAndSelectItems(pidl, 0, None, 0) == 0:
                            return
                    finally:
                        shell32.ILFree(pidl)
            finally:
                ole32.CoUninitialize()

        with suppress_logged("explorer", OSError, RuntimeError, throttled_key="explorer_select_popen"):
            subprocess.Popen(_build_explorer_select_cmdline(resolved))
            return

    with suppress_logged("explorer", OSError, RuntimeError, throttled_key="startfile_parent"):
        os.startfile(str(resolved.parent if resolved.parent.exists() else resolved))


class ColabPetBackend:
    """Colab 桌宠状态管理与运维动作门面（内部拆分为 5 个正交子服务）。"""

    def __init__(
        self,
        state_path: Path | None = None,
        download_root: Path | None = None,
        cf_config_path: Path | None = None,
    ) -> None:
        if state_path is None:
            if STATE_FILE != _INIT_STATE_FILE:
                eff_state = Path(STATE_FILE)
            elif _has_frozen_meipass():
                eff_state = Path(sys.executable).resolve().parent / "pet_state.json"
            else:
                eff_state = Path(STATE_FILE)
        else:
            eff_state = Path(state_path)

        if download_root is None:
            if LOCAL_DOWNLOAD_ROOT != _INIT_LOCAL_DOWNLOAD_ROOT:
                eff_dl = Path(LOCAL_DOWNLOAD_ROOT)
            elif _has_frozen_meipass():
                eff_dl = Path(sys.executable).resolve().parent / "_downloads" / "all"
            else:
                eff_dl = Path(LOCAL_DOWNLOAD_ROOT)
        else:
            eff_dl = Path(download_root)

        self._store = AppStateStore(eff_state)
        self.download_root = eff_dl
        self._cf_config_path: Path | None = Path(cf_config_path) if cf_config_path is not None else None
        self._lock = self._store.lock
        self._secrets = EphemeralSecretStore(lock=self._store.lock)
        self._bus = EventBus(redact_fn=_redact_payload)
        self._event_listeners_lock = self._bus.lock
        self._ctx = BackendContext.from_module_compat(_BACKEND_MOD, backend=self)

        self._fetch_lock = threading.Lock()
        self._comfy_lock = threading.RLock()
        self._pull_lock = threading.Lock()
        self._action_lock = threading.Lock()
        self._upload_lock = threading.Lock()
        self._keepalive_lock = threading.Lock()
        self._deploy_lock = threading.Lock()
        self._idle_stop_lock = threading.Lock()
        self._canvas_ext_lock = threading.Lock()

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
        self._last_activity_m: float | None = None
        self._idle_warned_ep: str | None = None
        self._idle_archive_retry_after_m: float = 0.0
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

        self._account_epoch: int = 0
        self._deploy_meta_warmed: bool = False
        self._deploy_phase_idx: int = 1
        self._deploy_phase_total: int = 1
        self._deploy_phase_prof: str = "anima"
        self._deploy_phase_is_llm: bool = False
        self._deploy_captured_llm_tunnel: str | None = None
        self._comfy_state: dict[str, Any] = self._default_comfy_state()
        self._pending_oauth_verifier: str = ""
        self._pending_oauth_ts: float = 0.0
        self._auth_missing: bool = False
        self._auth_expired: bool = False
        self._gdrive_bind_proc: Any = None
        self._gdrive_bind_cancelled: bool = False

        self._touch_ctx = threading.local()
        self._tunnel_acct_ctx = threading.local()
        self._keepalive_acct_ctx = threading.local()

        self._monitor = SessionMonitorService(self)
        self._deploy_svc = DeployOrchestrator(self)
        self._tunnel_svc = TunnelService(self)
        self._comfy_svc = ComfySyncService(self)
        self._accounts_svc = AccountVaultService(self)
        self._cluster_svc = ClusterPoolService(self)
        self._studio_svc = StudioService(self)
        self._agent_svc = AgentService(self)

        self.state = self._load_state()
        restored_pending: list[tuple[str, dict[str, str]]] = []
        for raw_it in self.state.get("pending_pull_items") or []:
            if isinstance(raw_it, dict) and raw_it.get("base_url") and raw_it.get("filename"):
                restored_pending.append(
                    (
                        str(raw_it["base_url"]).rstrip("/"),
                        {
                            "filename": str(raw_it["filename"]),
                            "subfolder": str(raw_it.get("subfolder") or ""),
                            "type": str(raw_it.get("type") or "output"),
                        },
                    )
                )
        self._pending_pull_items = restored_pending
        self._comfy_state["pending_pull_count"] = len(restored_pending)
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

    # ------------------------------------------------------------------
    # 容器底层同对象兼容属性（支持原地修改与整对象重赋同步）
    # ------------------------------------------------------------------

    @property
    def state_path(self) -> Path:
        return self._store.state_path

    @state_path.setter
    def state_path(self, value: Path) -> None:
        self._store.state_path = Path(value)

    @property
    def state(self) -> dict[str, Any]:
        return self._store.state

    @state.setter
    def state(self, value: dict[str, Any]) -> None:
        self._store.state = value

    @property
    def _event_cb(self) -> Callable[[dict[str, Any]], None] | None:
        return self._bus.legacy_callback

    @_event_cb.setter
    def _event_cb(self, value: Callable[[dict[str, Any]], None] | None) -> None:
        self._bus.legacy_callback = value

    @property
    def _event_listeners(self) -> list[Callable[[dict[str, Any]], None]]:
        return self._bus.listeners

    @_event_listeners.setter
    def _event_listeners(self, value: list[Callable[[dict[str, Any]], None]]) -> None:
        self._bus.listeners = value

    @property
    def _service_meta_by_ep(self) -> dict[str, dict[str, Any]]:
        return self._secrets.service_meta_by_ep

    @_service_meta_by_ep.setter
    def _service_meta_by_ep(self, value: dict[str, dict[str, Any]]) -> None:
        self._secrets.service_meta_by_ep = value

    @property
    def _runtime_secrets(self) -> dict[str, str]:
        return self._secrets.runtime_secrets

    @_runtime_secrets.setter
    def _runtime_secrets(self, value: dict[str, str]) -> None:
        self._secrets.runtime_secrets = value

    @property
    def _comfy_secrets_by_ep(self) -> dict[str, str]:
        return self._secrets.comfy_secrets_by_ep

    @_comfy_secrets_by_ep.setter
    def _comfy_secrets_by_ep(self, value: dict[str, str]) -> None:
        self._secrets.comfy_secrets_by_ep = value

    @property
    def _comfy_key_bindings_by_ep(self) -> dict[str, tuple[str, str]]:
        return self._secrets.comfy_key_bindings_by_ep

    @_comfy_key_bindings_by_ep.setter
    def _comfy_key_bindings_by_ep(self, value: dict[str, tuple[str, str]]) -> None:
        self._secrets.comfy_key_bindings_by_ep = value

    @property
    def _comfy_key_checked_eps(self) -> set[str]:
        return self._secrets.comfy_key_checked_eps

    @_comfy_key_checked_eps.setter
    def _comfy_key_checked_eps(self, value: set[str]) -> None:
        self._secrets.comfy_key_checked_eps = value

    @property
    def _last_comfy_key_hydrate_m(self) -> dict[str, float]:
        return self._secrets.last_comfy_key_hydrate_m

    @_last_comfy_key_hydrate_m.setter
    def _last_comfy_key_hydrate_m(self, value: dict[str, float]) -> None:
        self._secrets.last_comfy_key_hydrate_m = value

    # ------------------------------------------------------------------
    # 核心共享状态与基础设施访问器
    # ------------------------------------------------------------------

    def _ephemeral_stores(self) -> tuple[dict[str, Any], ...]:
        return (self._service_meta_by_ep, self._runtime_secrets, self._comfy_secrets_by_ep, self._comfy_key_bindings_by_ep)

    def _drop_ep_ephemeral_locked(self, ep: str) -> None:
        self._secrets.drop_endpoint(ep)
        self._store.pop_tunnel_cache(ep)

    def _retain_ep_ephemeral_locked(self, keep_eps: set[str]) -> None:
        self._secrets.retain_endpoints(keep_eps)

    def _clear_ephemeral_caches_locked(self) -> None:
        # 若异构集群仍有活跃分槽，仅清理非集群分槽端点，避免切号时误伤另一账号的隧道密钥
        self._secrets.retain_endpoints(self._store.get_active_cluster_endpoints())

    def _resolve_cf_config_path(self) -> Path:
        """解析当前实例绑定的 Cloudflare 固定隧道配置文件路径（兼容测试注入、模块覆盖与冻结包布局）。"""
        if self._cf_config_path is not None:
            return self._cf_config_path
        return _default_cf_secret_path()

    def _current_account_key(self) -> str:
        return self._store.get_active_account_id()

    def _record_managed_binding(self, session: str, endpoint: str, cli_scope: str | None = None) -> None:
        with self._lock:
            binding = self._store.record_managed_binding(session, endpoint, persist=False, cli_scope=cli_scope)
            if binding is not None:
                self._save_state()

    def _get_colab_exe(self) -> Path | None:
        if COLAB_EXE_DEFAULT != _INIT_COLAB_EXE:
            return COLAB_EXE_DEFAULT if COLAB_EXE_DEFAULT.is_file() else None
        prefs = (self.state.get("prefs") or {}) if hasattr(self, "state") else {}
        return _discover_colab_exe(str(prefs.get("colab_exe_path") or ""))

    def _get_colab_py(self) -> Path:
        if COLAB_PY != _INIT_COLAB_PY:
            return COLAB_PY
        prefs = (self.state.get("prefs") or {}) if hasattr(self, "state") else {}
        return _discover_colab_py(str(prefs.get("colab_py_path") or ""), colab_exe=self._get_colab_exe())

    def _get_rclone_exe(self) -> Path:
        if RCLONE_EXE != _INIT_RCLONE_EXE:
            return RCLONE_EXE
        prefs = (self.state.get("prefs") or {}) if hasattr(self, "state") else {}
        return _discover_rclone_exe(str(prefs.get("rclone_exe_path") or ""))

    def _get_rclone_conf(self) -> Path:
        if RCLONE_CONF != _INIT_RCLONE_CONF:
            return RCLONE_CONF
        prefs = (self.state.get("prefs") or {}) if hasattr(self, "state") else {}
        return _discover_rclone_conf(str(prefs.get("rclone_conf_path") or ""), rclone_exe=self._get_rclone_exe())

    def _build_ctl_env(self) -> dict[str, str]:
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
            "online": False, "service_kind": "comfy", "comfy_online": False, "llm_online": False,
            "ws_connected": False, "tunnel_url": None, "llm_tunnel_url": None,
            "running_count": 0, "pending_count": 0, "llm_running_count": 0, "active_profiles": [],
            "step_value": 0, "step_max": 0, "step_pct": 0.0, "current_node": "",
            "gpu_name": "", "gpu_full_name": "", "vram_used_gb": 0.0, "vram_total_gb": 0.0, "vram_pct": 0.0,
            "ram_used_gb": 0.0, "ram_total_gb": 0.0, "ram_free_gb": 0.0, "ram_pct": 0.0,
            "disk_used_gb": 0.0, "disk_total_gb": 0.0, "disk_free_gb": 0.0, "disk_pct": 0.0,
            "shm_used_gb": 0.0, "shm_total_gb": 0.0, "shm_free_gb": 0.0,
            "drive_persisted": False, "drive_free_gb": 0.0, "drive_total_gb": 0.0,
            "has_error": False, "error_summary": "", "pending_pull_count": 0,
            "last_output_count": 0, "last_output_name": "",
            "model_name": "", "quant": "", "ctx_size": 0, "tps": None,
        }

    def get_comfy_state(self) -> dict[str, Any]:
        with self._comfy_lock:
            return dict(self._comfy_state)

    def subscribe_events(self, callback: Callable[[dict[str, Any]], None]) -> Callable[[], None]:
        """订阅后端实时事件（同进程多窗口广播），返回线程安全的取消订阅函数。"""
        return self._bus.subscribe(callback)

    def _emit_event(self, payload: dict[str, Any]) -> None:
        self._bus.emit(payload)

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
        return self._store.load(
            lambda: _core_load_pet_state_dict(
                self.state_path,
                is_allowed_tunnel_fn=lambda u: _is_allowed_tunnel_url(u, config_path=cf_cfg_p),
                sanitize_label_fn=_sanitize_account_label,
            )
        )

    def _save_state(self) -> None:
        self._store.save()

    def _rebuild_and_emit_snapshot(
        self, stale: bool | None = None, error_msg: str | None = None
    ) -> dict[str, Any]:
        with self._lock:
            use_stale = bool(self.last_snapshot.get("stale", False)) if stale is None else bool(stale)
            use_err = self.last_snapshot.get("error_msg") if error_msg is None else error_msg
            snap = self._build_snapshot(live_data=None, stale=use_stale, error_msg=use_err)
            self.last_snapshot = snap
        self._emit_event({"type": "state_sync", "snapshot": snap})
        return snap

    def _get_epoch_fetch_lock(self, epoch: int) -> threading.Lock:
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

    def get_cached_snapshot(self) -> dict[str, Any]:
        with self._lock:
            return dict(self.last_snapshot)

    def touch_activity(self, slot: str | None = None) -> None:
        with self._lock:
            self._last_activity_m = time.monotonic()
            self._idle_warned_ep = None
        src = slot if slot in ("comfy", "llm") else getattr(self._touch_ctx, "source", None)
        if src in ("llm", "llm_probe"):
            self._cluster_svc.touch_slot_activity("llm")
        elif src in ("comfy", "comfy_probe"):
            self._cluster_svc.touch_slot_activity("comfy")
        else:
            self._cluster_svc.touch_slot_activity(None)

    def update_prefs(self, updates: dict[str, Any]) -> dict[str, Any]:
        """更新前端偏好设置，严格拦截通过通用偏好接口注入本地工具路径。"""
        upd = updates or {}
        with self._lock:
            prefs = self._store.update_prefs(
                upd,
                _sanitize_prefs_dict,
                _TOOL_PATH_PREF_KEYS,
                default_prefs=DEFAULT_PREFS,
                persist=False,
            )
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
        return self._store.pick_target_session(preferred)

    def _has_running_session(self) -> bool:
        return self._store.has_running_session()

    def _get_ep_meta_and_key(self, ep: str = "") -> tuple[dict[str, Any], str]:
        return self._secrets.get_ep_meta_and_key(ep)

    @staticmethod
    def _infer_kind_from_profiles(profiles: Any) -> str:
        norm = _normalize_boot_profiles(profiles, fallback_profile="anima")
        has_llm = any(p in LLM_BOOT_PROFILES for p in norm)
        has_comfy = any(p in COMFY_BOOT_PROFILES for p in norm)
        return "dual" if (has_llm and has_comfy) else ("llm" if has_llm else "comfy")

    @staticmethod
    def _infer_cluster_slots_kind(raw_slots: Any) -> str:
        """从异构集群分槽结构中推断服务形态（无有效分槽时返回空字符串）。"""
        if not isinstance(raw_slots, dict) or not raw_slots:
            return ""
        has_llm_slot = isinstance(raw_slots.get("llm"), dict)
        has_comfy_slot = isinstance(raw_slots.get("comfy"), dict)
        if has_llm_slot and has_comfy_slot:
            return "dual"
        if has_llm_slot:
            return "llm"
        if has_comfy_slot:
            return "comfy"
        return ""

    def _get_ep_service_kind(self, ep: str = "") -> str:
        with self._lock:
            target_ep = ep
            if not target_ep:
                slot_kind = self._infer_cluster_slots_kind(self.state.get("cluster_slots"))
                if slot_kind:
                    return slot_kind
                primary = self._store.pick_target_session("comfy")
                if primary and primary.get("endpoint"):
                    target_ep = str(primary["endpoint"])
            meta_kind = self._secrets.get_ep_service_kind(target_ep, default_kind="")
            if meta_kind:
                return meta_kind
            prefs = self.state.get("prefs") or {}
            boot_profs = prefs.get("boot_profiles") or [prefs.get("boot_profile") or "anima"]
        with self._comfy_lock:
            cur_kind = str(self._comfy_state.get("service_kind") or "")
            if self._comfy_state.get("online") and cur_kind in ("comfy", "llm", "dual"):
                return cur_kind
        return self._infer_kind_from_profiles(boot_profs)

    def _effective_mode_kind(self) -> str:
        if self._has_running_session():
            return self._get_ep_service_kind()
        with self._lock:
            prefs = self.state.get("prefs") or {}
            profs = prefs.get("boot_profiles") or [prefs.get("boot_profile") or "anima"]
        return self._infer_kind_from_profiles(profs)

    def is_llm_mode_active(self) -> bool:
        return self._effective_mode_kind() == "llm"

    def is_dual_mode_active(self) -> bool:
        return self._effective_mode_kind() == "dual"

    def has_llm_service_active(self) -> bool:
        return self._get_ep_service_kind() in ("llm", "dual")

    def _get_comfy_key_for_url(self, base_url: str = "", ep: str = "", account_id: str = "") -> str:
        """按 `(account_id, endpoint, origin)` 严格匹配 ComfyUI 密钥，禁止跨源/跨号/单密钥盲猜回退。"""
        if not url_origin_key(base_url):
            return ""
        req_acct = str(
            account_id
            or self._cluster_svc.resolve_account_for_endpoint(ep=ep, base_url=base_url)
            or self._current_account_key()
            or ""
        ).strip()
        with self._lock:
            cache = self.state.get("tunnel_cache") or {}
            raw_slots = self.state.get("cluster_slots")
            comfy_slot = (
                raw_slots.get("comfy")
                if isinstance(raw_slots, dict) and isinstance(raw_slots.get("comfy"), dict)
                else None
            )
            return self._secrets.get_comfy_key_for_url(
                base_url,
                ep,
                req_acct,
                tunnel_cache=cache,
                comfy_slot=comfy_slot,
            )

    def _resolve_hydrate_target(self, clean_base: str, raw_ep: str) -> tuple[str, str]:
        """解析 ComfyUI 密钥补水目标端点与归属账号 ID。"""
        use_ep = str(raw_ep or "").strip()
        if (not use_ep) and clean_base:
            use_ep = self._store.find_endpoint_by_tunnel_url(clean_base)
        slot_acct = self._cluster_svc.resolve_account_for_endpoint(ep=use_ep, base_url=clean_base)
        if self._cluster_svc.has_active_slots():
            cs = self._cluster_svc.get_slot("comfy")
            if cs and not bool(cs.get("binding_invalid")):
                c_ep = str(cs.get("endpoint") or "").strip()
                c_aid = str(cs.get("account_id") or "").strip()
                if (not slot_acct) or slot_acct == c_aid:
                    use_ep = use_ep or c_ep
                    slot_acct = slot_acct or c_aid
        return use_ep, str(slot_acct or "").strip()

    def _hydrate_comfy_key_if_needed(self, base_url: str = "", ep: str = "", *, force: bool = False) -> str:
        """在会话在线且未命中冷却时向远端补水 ComfyUI 密钥并记录同源同号绑定。"""
        if not self._has_running_session():
            return ""
        clean_base = str(base_url or "").strip().rstrip("/")
        use_ep, slot_acct = self._resolve_hydrate_target(clean_base, ep)
        key_tag = use_ep or clean_base or "__default__"
        if self._secrets.should_skip_hydrate(key_tag, force=force, now_m=time.monotonic()):
            return self._get_comfy_key_for_url(base_url, use_ep, account_id=slot_acct)

        tstatus, resolved_ep, fast_url = self._tunnel_svc.fast_read_slot_tunnel_url(
            use_ep, "comfy", account_id=slot_acct
        )
        eff_ep = str(resolved_ep or use_ep or "").strip()
        eff_acct = str(
            slot_acct
            or self._cluster_svc.resolve_account_for_endpoint(ep=eff_ep, base_url=clean_base)
            or self._current_account_key()
            or ""
        ).strip()
        if tstatus == "ok" and eff_ep:
            self._secrets.record_hydrated_binding(eff_ep, eff_acct, fast_url)
        return self._get_comfy_key_for_url(clean_base, eff_ep, account_id=eff_acct)

    def _prepare_comfy_auth_headers(self, headers: dict[str, Any], comfy_key: str) -> dict[str, Any]:
        """清理错源 Bearer 凭据头并按需附加已校验的同源 ComfyUI 密钥。"""
        auth_val = str(headers.get("Authorization") or "")
        if auth_val.lower().startswith("bearer "):
            raw_tok = auth_val[7:].strip()
            known_secrets = self._secrets.known_comfy_secret_values()
            if raw_tok in known_secrets and raw_tok != comfy_key:
                headers.pop("Authorization", None)
        if comfy_key and "Authorization" not in headers:
            headers["Authorization"] = f"Bearer {comfy_key}"
        return headers

    def _comfy_http_request(
        self, method: str, url: str, *, base_url: str = "", ep: str = "", retry_on_401: bool = True, **kwargs: Any
    ) -> requests.Response:
        req_origin = url_origin_key(url)
        if not req_origin:
            raise ValueError("安全拦截：拒绝向非法 URL 发送 ComfyUI 凭据请求")
        inferred_base = str(base_url or "").strip().rstrip("/")
        if inferred_base:
            exp_origin = url_origin_key(inferred_base)
            if not exp_origin or req_origin != exp_origin:
                raise ValueError("安全拦截：拒绝向非目标源站发送 ComfyUI 凭据请求")
        else:
            parsed = urllib.parse.urlsplit(url)
            inferred_base = f"{parsed.scheme}://{parsed.netloc}"

        kwargs.setdefault("allow_redirects", False)
        headers = dict(kwargs.pop("headers", None) or {})
        acct_scope = self._cluster_svc.resolve_account_for_endpoint(ep=ep, base_url=inferred_base) or self._current_account_key()
        comfy_key = self._get_comfy_key_for_url(inferred_base, ep, account_id=acct_scope)
        headers = self._prepare_comfy_auth_headers(headers, comfy_key)

        resp = self._http.request(method, url, headers=headers, **kwargs)
        if resp.status_code == 401 and retry_on_401 and self._has_running_session():
            # 关闭待重试的 401 响应流，忽略底层连接已断开引发的清理异常
            with contextlib.suppress(OSError, RuntimeError, AttributeError, ValueError, requests.RequestException):
                resp.close()
            new_key = self._hydrate_comfy_key_if_needed(inferred_base, ep, force=(not bool(comfy_key)))
            verified_key = self._get_comfy_key_for_url(inferred_base, ep, account_id=acct_scope) if new_key else ""
            if new_key and verified_key == new_key and new_key != comfy_key:
                headers["Authorization"] = f"Bearer {new_key}"
                resp = self._http.request(method, url, headers=headers, **kwargs)
        return resp

    def _comfy_http_get(self, url: str, *, base_url: str = "", ep: str = "", retry_on_401: bool = True, **kwargs: Any) -> requests.Response:
        return self._comfy_http_request("GET", url, base_url=base_url, ep=ep, retry_on_401=retry_on_401, **kwargs)

    def _comfy_http_post(self, url: str, *, base_url: str = "", ep: str = "", retry_on_401: bool = True, **kwargs: Any) -> requests.Response:
        return self._comfy_http_request("POST", url, base_url=base_url, ep=ep, retry_on_401=retry_on_401, **kwargs)

    def _ensure_target_session(
        self, session: str, empty_msg: str, *, always_probe_when_none: bool = False
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        target = self._pick_target_session(session)
        if target is None and (always_probe_when_none or not self._has_running_session()):
            snap = self.fetch_live_status()
            target = self._pick_target_session(session)
            if target is None and snap.get("assignments_count", 0) == 0 and not snap.get("sessions"):
                return None, {"ok": False, "message": empty_msg}
        return target, None

    def _clear_stale_deploy_error_locked(self) -> dict[str, Any] | None:
        if not self._deploy_state.get("active") and self._deploy_state.get("error_msg"):
            self._deploy_state["error_msg"] = ""
            if self._deploy_state.get("stage_title") == "部署失败":
                self._deploy_state["stage_title"] = ""
                self._deploy_state["sub_detail"] = ""
            snap_deploy = dict(self._deploy_state)
            self.last_snapshot["deploy"] = snap_deploy
            return snap_deploy
        return None

    # ------------------------------------------------------------------
    # 保留含业务分支或上下文包装的门面方法（其余纯转发由下方表驱动绑定）
    # ------------------------------------------------------------------

    _build_comfy_open_url = staticmethod(TunnelService.build_comfy_open_url)
    _read_text_file_safe = staticmethod(ComfySyncService.read_text_file_safe)

    def _probe_comfy_status(self, base: str) -> tuple[bool, dict[str, Any], list[dict[str, Any]]]:
        prev_src = getattr(self._touch_ctx, "source", None)
        self._touch_ctx.source = "comfy_probe"
        try:
            return self._comfy_svc.probe_comfy_status(base)
        finally:
            self._touch_ctx.source = prev_src

    def _probe_llm_status(self, base: str, ep: str = "") -> tuple[bool, dict[str, Any]]:
        prev_src = getattr(self._touch_ctx, "source", None)
        self._touch_ctx.source = "llm_probe"
        try:
            return self._comfy_svc.probe_llm_status(base, ep=ep)
        finally:
            self._touch_ctx.source = prev_src

    def poll_comfy_http_once(self, tunnel_url: str) -> dict[str, Any]:
        return self._poll_service_once("comfy", comfy_url=tunnel_url)

    def poll_llm_http_once(self, tunnel_url: str, ep: str = "") -> dict[str, Any]:
        return self._poll_service_once("llm", llm_url=tunnel_url, ep=ep)

    def poll_dual_once(self, comfy_url: str, llm_url: str, ep: str = "") -> dict[str, Any]:
        return self._poll_service_once("dual", comfy_url=comfy_url, llm_url=llm_url, ep=ep)

    def sync_outputs(self, session: str = "comfy") -> dict[str, Any]:
        """将云端 ComfyUI 成图拉回本地目录（异构集群模式下自动分流至画布槽位）。"""
        if self._cluster_svc.has_active_slots():
            return self._cluster_svc.sync_slot_outputs(session=session)
        return self._comfy_svc.sync_outputs(session=session)

    def _write_remote_file_via_contents(self, rel_path: str, raw_bytes: bytes, session: str = "comfy") -> None:
        comfy_slot = self._cluster_svc.get_slot("comfy") if self._cluster_svc.has_active_slots() else None
        if comfy_slot and comfy_slot.get("account_id"):
            colab_py = self._get_colab_py()
            if not colab_py.exists():
                raise FileNotFoundError(f"未找到 colab-cli Python: {colab_py}")
            ep = str(comfy_slot.get("endpoint") or "")
            sname = str(comfy_slot.get("session") or session or "comfy")
            aid = str(comfy_slot.get("account_id") or "")
            b64 = base64.b64encode(raw_bytes).decode("ascii")
            cmd = [str(colab_py), str(CLI_BRIDGE_SCRIPT), "contents-put", ep, sname, rel_path]
            if aid:
                cmd.append(f"--acct={aid}")
            proc = _run_hidden(cmd, timeout=25, input_text=b64)
            if proc.returncode != 0:
                err = redact_secrets((proc.stderr or proc.stdout or "").strip()).splitlines()
                raise RuntimeError(err[-1] if err else "Contents 写入失败")
            return
        self._comfy_svc.write_remote_file_via_contents(rel_path, raw_bytes, session=session)

    def _upload_workflow_file(self, file_path: Path, is_png_extract: bool, tunnel_url: str | None) -> tuple[str, bool]:
        msg, online_ok, _ = self._upload_workflow_file_ex(file_path, is_png_extract, tunnel_url)
        return msg, online_ok

    def _upload_model_file(self, file_path: Path, tunnel_url: str | None, session: str = "comfy") -> tuple[str, bool]:
        comfy_slot = self._cluster_svc.get_slot("comfy") if self._cluster_svc.has_active_slots() else None
        if comfy_slot and comfy_slot.get("account_id"):
            safe_name = _sanitize_filename(file_path.name, "model.safetensors")
            remote_path = f"{GDRIVE_LORA_REMOTE}/{safe_name}"
            self._rclone_upload_file(
                file_path,
                remote_path,
                safe_name,
                pct_start=0,
                pct_span=70,
            )
            self._comfy_svc._emit_upload_progress(safe_name, 78, "同步到实例")
            aid = str(comfy_slot.get("account_id") or "").strip()
            sname = str(comfy_slot.get("session") or session or "comfy").strip() or "comfy"
            comfy_ep = str(comfy_slot.get("endpoint") or "").strip()
            colab_py = self._get_colab_py()
            if colab_py.exists():
                scoped_lora_script = (
                    "import sys\n"
                    f"if {str(PACKAGE_DIR)!r} not in sys.path:\n"
                    f"    sys.path.insert(0, {str(PACKAGE_DIR)!r})\n"
                    "from colab_cli_bridge import _bind_account_state\n"
                    f"_bind_account_state({aid!r})\n"
                    + _COLAB_REMOTE_LORA_PULL_SCRIPT
                )
                proc = _run_hidden(
                    [
                        str(colab_py),
                        "-c",
                        scoped_lora_script,
                        sname,
                        remote_path,
                        f"/content/ComfyUI/models/loras/{safe_name}",
                    ],
                    timeout=180,
                )
                if proc.returncode != 0:
                    err = redact_secrets((proc.stderr or proc.stdout or "").strip()).splitlines()
                    raise RuntimeError(err[-1] if err else "画布槽位实例拉取 LoRA 失败")

            if tunnel_url:
                with suppress_logged("comfy", throttled_key="lora_loader_refresh"):
                    base_u = tunnel_url.rstrip("/")
                    self._comfy_http_get(
                        f"{base_u}/object_info/LoraLoader", base_url=base_u, ep=comfy_ep, timeout=5
                    )
            self._comfy_svc._emit_upload_progress(safe_name, 100, "同步到实例")
            return f"✅ LoRA 已同步至云端并刷新: {safe_name}", True
        return self._comfy_svc.upload_model_file(file_path, tunnel_url, session=session)


_BACKEND_DELEGATE_GROUPS: tuple[
    tuple[str, type[Any], tuple[str | tuple[str, str], ...]], ...
] = (
    (
        "_monitor",
        SessionMonitorService,
        (
            ("_query_colab_live", "query_colab_live"),
            ("_update_ledger", "update_ledger"),
            ("_get_onboarding_state", "get_onboarding_state"),
            ("_build_snapshot", "build_snapshot"),
            "fetch_live_status",
            ("_send_colab_keepalive", "send_colab_keepalive"),
            ("_maybe_trigger_keepalive", "maybe_trigger_keepalive"),
            ("_maybe_trigger_idle_auto_stop", "maybe_trigger_idle_auto_stop"),
            "start_monitor",
            "stop_monitor",
        ),
    ),
    (
        "_tunnel_svc",
        TunnelService,
        (
            ("_tunnel_url_regex_pattern", "tunnel_url_regex_pattern"),
            ("_fast_read_tunnel_url", "fast_read_tunnel_url"),
            ("_read_cf_public_llm_url", "read_cf_public_llm_url"),
            ("_infer_default_llm_model_name", "infer_default_llm_model_name"),
            "resolve_active_tunnel_url",
            "resolve_active_llm_tunnel_url",
            ("_ensure_legacy_cmd_file", "ensure_legacy_cmd_file"),
            ("_ensure_direct_canvas_support", "ensure_direct_canvas_support"),
            ("_launch_comfy_browser", "launch_comfy_browser"),
            ("_push_or_open_workflow_on_canvas", "push_or_open_workflow_on_canvas"),
            ("_backup_workflow_to_gdrive_async", "backup_workflow_to_gdrive_async"),
            ("_open_browser_for_comfy", "open_browser_for_comfy"),
            ("_open_resolved_service_url", "open_resolved_service_url"),
            "open_comfy_tunnel",
            "open_llm_tunnel",
            "get_comfy_access_bundle",
            "get_llm_api_bundle",
            "open_colab_url",
            "save_cf_tunnel_config",
            "clear_cf_tunnel_config",
        ),
    ),
    (
        "_deploy_svc",
        DeployOrchestrator,
        (
            ("_resolve_target_binding", "resolve_target_binding"),
            "stop_session_instance",
            "stop_all_instances",
            ("_consume_deploy_line", "consume_deploy_line"),
            "start_instance",
            "reconfigure_running_llm",
            "cancel_deploy",
        ),
    ),
    (
        "_comfy_svc",
        ComfySyncService,
        (
            "acknowledge_comfy_error",
            ("_reset_comfy_offline", "reset_comfy_offline"),
            ("_download_comfy_outputs", "download_comfy_outputs"),
            ("_poll_service_once", "poll_service_once"),
            "pull_or_open_latest_outputs",
            ("_on_ws_message", "on_ws_message"),
            ("_stop_ws", "stop_ws"),
            ("_ensure_ws_target", "ensure_ws_target"),
            ("_rclone_upload_file", "rclone_upload_file"),
            ("_fetch_comfy_object_info", "fetch_comfy_object_info"),
            ("_upload_workflow_file_ex", "upload_workflow_file_ex"),
            ("_upload_media_file", "upload_media_file"),
            ("_handle_llm_dropped_files", "handle_llm_dropped_files"),
            "submit_dropped_files",
        ),
    ),
    (
        "_accounts_svc",
        AccountVaultService,
        (
            "start_colab_oauth",
            "cancel_colab_oauth",
            "complete_colab_oauth",
            "switch_colab_account",
            "rename_colab_account",
            "remove_colab_account",
            ("_query_gdrive_quota_locked", "query_gdrive_quota_locked"),
            "probe_environment_deep",
            "start_gdrive_bind",
            "cancel_gdrive_bind",
            "import_rclone_conf",
            "set_tool_path",
            "reset_tool_paths",
            "open_local_outputs_dir",
            "run_preflight",
            "export_diagnostics",
            "open_logs_dir",
        ),
    ),
    (
        "_cluster_svc",
        ClusterPoolService,
        (
            "refresh_account_pool",
            "stop_cluster_slot",
        ),
    ),
    (
        "_studio_svc",
        StudioService,
        (
            "list_gallery_items",
            "inspect_gallery_item",
            "push_gallery_workflow_to_canvas",
            "reveal_gallery_item",
            "list_cloud_assets",
            "tail_live_logs",
        ),
    ),
    (
        "_agent_svc",
        AgentService,
        (
            "get_agent_state",
            "save_agent_config",
            "fetch_agent_models",
            "probe_agent_provider",
            "send_agent_message",
            "resolve_agent_confirm",
            "cancel_agent_turn",
            "clear_agent_history",
            "manage_agent_session",
        ),
    ),
    (
        "_tasks",
        TaskManager,
        (("get_task_snapshot", "snapshot"),),
    ),
)


def _make_service_delegate(
    public_name: str, svc_attr: str, target_fn: Callable[..., Any]
) -> Callable[..., Any]:
    """构造保留目标方法真实签名元数据与文档字符串的表驱动门面转发方法。"""
    target_name = target_fn.__name__

    @functools.wraps(target_fn)
    def _delegate(self: ColabPetBackend, *args: Any, **kwargs: Any) -> Any:
        return getattr(getattr(self, svc_attr), target_name)(*args, **kwargs)

    _delegate.__name__ = public_name
    _delegate.__qualname__ = f"ColabPetBackend.{public_name}"
    return _delegate


def _bind_backend_delegates(cls: type[ColabPetBackend]) -> None:
    """将纯一行子服务委托方法批量绑定至 ColabPetBackend 类体。"""
    for svc_attr, svc_cls, specs in _BACKEND_DELEGATE_GROUPS:
        for spec in specs:
            public_name, target_name = spec if isinstance(spec, tuple) else (spec, spec)
            target_fn = getattr(svc_cls, target_name)
            setattr(cls, public_name, _make_service_delegate(public_name, svc_attr, target_fn))
    cls._get_or_fetch_tunnel_url = cls.resolve_active_tunnel_url  # type: ignore[attr-defined]
    cls.copy_llm_api_config = cls.get_llm_api_bundle  # type: ignore[attr-defined]


_bind_backend_delegates(ColabPetBackend)


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
