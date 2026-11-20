"""后端事件总线（EventBus）与显式依赖注入上下文（BackendContext）。

本模块提供两项核心基础设施：
1. `EventBus`：独占事件监听器列表与历史单回调 `legacy_callback`，在广播前统一脱敏并对每个监听器做异常隔离。
2. `BackendContext`：显式封装原 `colab_backend` 模块的全部外部依赖与运行时常量，并通过
   `BackendContext.from_module_compat(...)` 保持动态属性解析，确保既有测试中的
   `monkeypatch.setattr(cb, ...)` 与实例方法打桩继续 100% 透明生效。
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
import webbrowser
from collections.abc import Callable
from pathlib import Path
from typing import Any

_PET_DIR = Path(__file__).resolve().parent
_PACKAGE_DIR = _PET_DIR.parent
if str(_PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_DIR))

from colab_core.exec_runner import suppress_logged  # noqa: E402

__all__ = ["EventBus", "BackendContext"]

_MODULE_ATTR_MAP: dict[str, str] = {
    "time_mod": "time",
    "subprocess_mod": "subprocess",
    "threading_mod": "threading",
    "webbrowser_mod": "webbrowser",
    "websocket_mod": "websocket",
    "package_dir": "PACKAGE_DIR",
    "pet_dir": "PET_DIR",
    "cli_bridge_script": "CLI_BRIDGE_SCRIPT",
    "ctl_script": "CTL_SCRIPT",
    "bridge_node_init_py": "_COMFY_BRIDGE_NODE_INIT_PY",
    "preserve_original_sub_re": "_PRESERVE_ORIGINAL_SUB_RE",
    "keepalive_interval_sec": "KEEPALIVE_INTERVAL_SEC",
    "keepalive_retry_cooldown_sec": "KEEPALIVE_RETRY_COOLDOWN_SEC",
    "keepalive_max_idle_sec": "KEEPALIVE_MAX_IDLE_SEC",
}

_MODULE_FUNC_MAP: dict[str, str] = {
    "run_hidden": "_run_hidden",
    "reveal_path": "_reveal_path_in_explorer",
    "ctl_available": "_ctl_available",
    "ctl_cmd_prefix": "_ctl_cmd_prefix",
    "hidden_python_exe": "_hidden_python_exe",
    "load_cf_secret_raw": "_load_cf_secret_raw",
    "is_allowed_tunnel_url": "_is_allowed_tunnel_url",
    "probe_tunnel_alive": "_probe_tunnel_alive",
    "detect_tunnel_kind": "_detect_tunnel_kind",
}

_STDLIB_FALLBACKS: dict[str, Any] = {
    "time_mod": time,
    "subprocess_mod": subprocess,
    "threading_mod": threading,
    "webbrowser_mod": webbrowser,
    "websocket_mod": None,
}

_ALLOWED_OVERRIDE_KEYS: frozenset[str] = frozenset(
    set(_MODULE_ATTR_MAP.keys())
    | set(_MODULE_FUNC_MAP.keys())
    | {
        "now",
        "monotonic",
        "sleep",
        "open_browser",
        "get_colab_exe",
        "get_colab_py",
        "get_rclone_exe",
        "get_rclone_conf",
        "build_ctl_env",
        "resolve_cf_config_path",
    }
)


class EventBus:
    """线程安全的同进程多窗口事件广播总线（含自动脱敏与异常隔离）。"""

    def __init__(self, redact_fn: Callable[[Any], Any] | None = None) -> None:
        self._lock: threading.RLock = threading.RLock()
        self._listeners: list[Callable[[dict[str, Any]], None]] = []
        self.legacy_callback: Callable[[dict[str, Any]], None] | None = None
        self._redact_fn = redact_fn

    @property
    def lock(self) -> threading.RLock:
        """暴露底层监听器锁，供门面保持向后兼容别名。"""
        return self._lock

    @property
    def listeners(self) -> list[Callable[[dict[str, Any]], None]]:
        """暴露底层监听器列表，供门面保持向后兼容别名。"""
        return self._listeners

    @listeners.setter
    def listeners(self, value: list[Callable[[dict[str, Any]], None]]) -> None:
        """允许兼容层替换底层监听器列表。"""
        with self._lock:
            self._listeners = value

    def subscribe(self, cb: Callable[[dict[str, Any]], None]) -> Callable[[], None]:
        """订阅后端实时事件，返回幂等且线程安全的取消订阅函数。"""
        with self._lock:
            if cb not in self._listeners:
                self._listeners.append(cb)

        def _unsubscribe() -> None:
            with self._lock:
                while cb in self._listeners:
                    self._listeners.remove(cb)

        return _unsubscribe

    def emit(self, payload: dict[str, Any]) -> None:
        """先对事件载荷脱敏，再逐一广播给历史回调与所有订阅者（单监听器异常隔离）。"""
        safe_payload = self._redact_fn(payload) if self._redact_fn is not None else payload
        with self._lock:
            listeners = list(self._listeners)
        legacy_cb = self.legacy_callback
        if legacy_cb and legacy_cb not in listeners:
            self._safe_invoke(legacy_cb, safe_payload)
        for listener in listeners:
            self._safe_invoke(listener, safe_payload)

    @staticmethod
    def _safe_invoke(callback: Callable[[Any], None], payload: Any) -> None:
        """安全执行单个事件回调，单监听器异常隔离并写入节流诊断日志。"""
        cb_name = getattr(callback, "__name__", type(callback).__name__)
        with suppress_logged("event_bus", throttled_key=f"cb:{cb_name}"):
            callback(dict(payload) if isinstance(payload, dict) else payload)


class BackendContext:
    """面向后端子服务的显式依赖注入上下文。

    通过 `from_module_compat(cb_mod, backend=backend, **overrides)` 构造时，所有未显式覆写的
    模块常量与工具函数均在访问时动态解析 `getattr(cb_mod, ...)`，保证测试热补丁实时生效。
    """

    def __init__(
        self,
        *,
        cb_mod: Any = None,
        backend: Any = None,
        **overrides: Any,
    ) -> None:
        unknown = set(overrides.keys()) - _ALLOWED_OVERRIDE_KEYS
        if unknown:
            bad = ", ".join(sorted(unknown))
            raise TypeError(f"未知的 BackendContext 依赖覆写项: {bad}")
        self._compat_mod = cb_mod
        self._backend = backend
        self._overrides: dict[str, Any] = dict(overrides)

    @classmethod
    def from_module_compat(
        cls,
        cb_mod: Any,
        *,
        backend: Any = None,
        **overrides: Any,
    ) -> BackendContext:
        """构造与 `colab_backend` 模块动态兼容的依赖上下文，支持按需覆写单项依赖。"""
        return cls(cb_mod=cb_mod, backend=backend, **overrides)

    def _resolve_attr(self, key: str) -> Any:
        """优先读取显式覆写值，否则动态读取兼容模块对应属性。"""
        if key in self._overrides:
            return self._overrides[key]
        mod_attr = _MODULE_ATTR_MAP[key]
        if self._compat_mod is not None:
            return getattr(self._compat_mod, mod_attr)
        if key in _STDLIB_FALLBACKS:
            return _STDLIB_FALLBACKS[key]
        raise AttributeError(f"BackendContext 未配置依赖项: {key}")

    def _call_mod_func(self, key: str, *args: Any, **kwargs: Any) -> Any:
        """优先调用显式覆写函数，否则动态读取兼容模块上的同名工具函数执行。"""
        if key in self._overrides:
            return self._overrides[key](*args, **kwargs)
        mod_fn_name = _MODULE_FUNC_MAP[key]
        if self._compat_mod is not None:
            return getattr(self._compat_mod, mod_fn_name)(*args, **kwargs)
        raise AttributeError(f"BackendContext 未配置调用依赖项: {key}")

    def _call_backend_or_override(
        self,
        key: str,
        backend_method: str,
        fallback_fn: Callable[[], Any],
    ) -> Any:
        """按「显式覆写 -> 绑定 backend 方法 -> 模块回退」顺序解析工具路径与环境依赖。"""
        if key in self._overrides:
            val = self._overrides[key]
            return val() if callable(val) else val
        if self._backend is not None and hasattr(self._backend, backend_method):
            return getattr(self._backend, backend_method)()
        return fallback_fn()

    # ------------------------------------------------------------------
    # 标准库与第三方模块依赖属性
    # ------------------------------------------------------------------

    @property
    def time_mod(self) -> Any:
        return self._resolve_attr("time_mod")

    @property
    def subprocess_mod(self) -> Any:
        return self._resolve_attr("subprocess_mod")

    @property
    def threading_mod(self) -> Any:
        return self._resolve_attr("threading_mod")

    @property
    def webbrowser_mod(self) -> Any:
        return self._resolve_attr("webbrowser_mod")

    @property
    def websocket_mod(self) -> Any:
        return self._resolve_attr("websocket_mod")

    # ------------------------------------------------------------------
    # 路径、正则与调度常量属性
    # ------------------------------------------------------------------

    @property
    def package_dir(self) -> Path:
        return self._resolve_attr("package_dir")

    @property
    def pet_dir(self) -> Path:
        return self._resolve_attr("pet_dir")

    @property
    def cli_bridge_script(self) -> Path:
        return self._resolve_attr("cli_bridge_script")

    @property
    def ctl_script(self) -> Path:
        return self._resolve_attr("ctl_script")

    @property
    def bridge_node_init_py(self) -> str:
        return self._resolve_attr("bridge_node_init_py")

    @property
    def preserve_original_sub_re(self) -> Any:
        return self._resolve_attr("preserve_original_sub_re")

    @property
    def keepalive_interval_sec(self) -> float:
        return float(self._resolve_attr("keepalive_interval_sec"))

    @property
    def keepalive_retry_cooldown_sec(self) -> float:
        return float(self._resolve_attr("keepalive_retry_cooldown_sec"))

    @property
    def keepalive_max_idle_sec(self) -> float:
        return float(self._resolve_attr("keepalive_max_idle_sec"))

    # ------------------------------------------------------------------
    # 时钟与浏览器操作封装
    # ------------------------------------------------------------------

    def now(self) -> float:
        """返回当前 Unix 时间戳（秒）。"""
        if "now" in self._overrides:
            return float(self._overrides["now"]())
        return float(self.time_mod.time())

    def monotonic(self) -> float:
        """返回单调时钟时间戳（秒）。"""
        if "monotonic" in self._overrides:
            return float(self._overrides["monotonic"]())
        return float(self.time_mod.monotonic())

    def sleep(self, seconds: float) -> None:
        """休眠指定秒数。"""
        if "sleep" in self._overrides:
            self._overrides["sleep"](seconds)
            return
        self.time_mod.sleep(seconds)

    def open_browser(self, url: str, *args: Any, **kwargs: Any) -> Any:
        """调用系统默认浏览器打开目标 URL。"""
        if "open_browser" in self._overrides:
            return self._overrides["open_browser"](url, *args, **kwargs)
        return self.webbrowser_mod.open(url, *args, **kwargs)

    # ------------------------------------------------------------------
    # 子进程、控制脚本与隧道探测函数封装
    # ------------------------------------------------------------------

    def run_hidden(self, *args: Any, **kwargs: Any) -> Any:
        """以隐藏控制台窗口方式执行子进程命令。"""
        return self._call_mod_func("run_hidden", *args, **kwargs)

    def reveal_path(self, target: Path) -> None:
        """在系统文件管理器中打开目录或高亮选中指定文件。"""
        self._call_mod_func("reveal_path", target)

    def ctl_available(self) -> bool:
        """检查 `colab_ctl.py` 或冻结内部控制入口是否可用。"""
        return bool(self._call_mod_func("ctl_available"))

    def ctl_cmd_prefix(self, unbuffered: bool = False) -> list[str]:
        """构造调用 `colab_ctl` 的命令前缀。"""
        return list(self._call_mod_func("ctl_cmd_prefix", unbuffered=unbuffered))

    def hidden_python_exe(self) -> str:
        """返回适合后台无窗执行脚本的 Python 解释器路径。"""
        return str(self._call_mod_func("hidden_python_exe"))

    def load_cf_secret_raw(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """读取 Cloudflare 固定隧道配置（默认脱敏，按需含密钥）。"""
        return self._call_mod_func("load_cf_secret_raw", *args, **kwargs)

    def is_allowed_tunnel_url(self, *args: Any, **kwargs: Any) -> bool:
        """校验目标 URL 是否属于受信任的公网隧道源站。"""
        return bool(self._call_mod_func("is_allowed_tunnel_url", *args, **kwargs))

    def probe_tunnel_alive(self, *args: Any, **kwargs: Any) -> bool:
        """探测公网隧道端点是否在线可达。"""
        return bool(self._call_mod_func("probe_tunnel_alive", *args, **kwargs))

    def detect_tunnel_kind(self, *args: Any, **kwargs: Any) -> str | None:
        """识别指定隧道 URL 对应的远端服务类型（comfy / llm）。"""
        return self._call_mod_func("detect_tunnel_kind", *args, **kwargs)

    # ------------------------------------------------------------------
    # 本地工具链路径与子进程环境构造
    # ------------------------------------------------------------------

    def _fallback_colab_exe(self) -> Path | None:
        if self._compat_mod is None:
            return None
        cur = getattr(self._compat_mod, "COLAB_EXE_DEFAULT", None)
        init = getattr(self._compat_mod, "_INIT_COLAB_EXE", None)
        if cur is not None and cur != init:
            return cur if Path(cur).is_file() else None
        return self._compat_mod._discover_colab_exe("")

    def _fallback_colab_py(self) -> Path:
        if self._compat_mod is None:
            raise AttributeError("BackendContext 未配置 get_colab_py 依赖")
        cur = self._compat_mod.COLAB_PY
        init = self._compat_mod._INIT_COLAB_PY
        if cur != init:
            return cur
        return self._compat_mod._discover_colab_py("", colab_exe=self.get_colab_exe())

    def _fallback_rclone_exe(self) -> Path:
        if self._compat_mod is None:
            raise AttributeError("BackendContext 未配置 get_rclone_exe 依赖")
        cur = self._compat_mod.RCLONE_EXE
        init = self._compat_mod._INIT_RCLONE_EXE
        if cur != init:
            return cur
        return self._compat_mod._discover_rclone_exe("")

    def _fallback_rclone_conf(self) -> Path:
        if self._compat_mod is None:
            raise AttributeError("BackendContext 未配置 get_rclone_conf 依赖")
        cur = self._compat_mod.RCLONE_CONF
        init = self._compat_mod._INIT_RCLONE_CONF
        if cur != init:
            return cur
        return self._compat_mod._discover_rclone_conf("", rclone_exe=self.get_rclone_exe())

    def _fallback_cf_config_path(self) -> Path:
        if self._compat_mod is None:
            raise AttributeError("BackendContext 未配置 resolve_cf_config_path 依赖")
        return self._compat_mod._default_cf_secret_path()

    def _fallback_build_ctl_env(self) -> dict[str, str]:
        env_map: dict[str, str] = {}
        cexe = self.get_colab_exe()
        if cexe is not None:
            env_map["COLAB_PET_COLAB_EXE"] = str(cexe)
        cpy = self.get_colab_py()
        if cpy.is_file():
            env_map["COLAB_PET_COLAB_PY"] = str(cpy)
        rexe = self.get_rclone_exe()
        if rexe.is_file():
            env_map["COLAB_PET_RCLONE_EXE"] = str(rexe)
        rconf = self.get_rclone_conf()
        if rconf.is_file():
            env_map["COLAB_PET_RCLONE_CONF"] = str(rconf)
        env_map["COLAB_PET_CF_CONFIG_PATH"] = str(self.resolve_cf_config_path())
        mock_vault = str(os.environ.get("COLAB_PET_MOCK_VAULT_PATH") or "").strip()
        if mock_vault:
            env_map["COLAB_PET_MOCK_VAULT_PATH"] = mock_vault
        return env_map

    def get_colab_exe(self) -> Path | None:
        """解析当前生效的 `colab.exe` 路径。"""
        return self._call_backend_or_override(
            "get_colab_exe", "_get_colab_exe", self._fallback_colab_exe
        )

    def get_colab_py(self) -> Path:
        """解析当前生效的 `colab-cli` Python 解释器路径。"""
        return self._call_backend_or_override(
            "get_colab_py", "_get_colab_py", self._fallback_colab_py
        )

    def get_rclone_exe(self) -> Path:
        """解析当前生效的 `rclone.exe` 路径。"""
        return self._call_backend_or_override(
            "get_rclone_exe", "_get_rclone_exe", self._fallback_rclone_exe
        )

    def get_rclone_conf(self) -> Path:
        """解析当前生效的 `rclone.conf` 配置文件路径。"""
        return self._call_backend_or_override(
            "get_rclone_conf", "_get_rclone_conf", self._fallback_rclone_conf
        )

    def resolve_cf_config_path(self) -> Path:
        """解析当前生效的 Cloudflare 固定隧道配置文件路径。"""
        return self._call_backend_or_override(
            "resolve_cf_config_path",
            "_resolve_cf_config_path",
            self._fallback_cf_config_path,
        )

    def build_ctl_env(self) -> dict[str, str]:
        """构造传递给 `colab_ctl` 子进程的环境变量字典。"""
        return self._call_backend_or_override(
            "build_ctl_env", "_build_ctl_env", self._fallback_build_ctl_env
        )
