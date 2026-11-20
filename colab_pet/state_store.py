"""桌宠后端持久化状态容器（AppStateStore）。

负责独占管理本地状态字典 `self._state`、可重入主状态锁 `self.lock` 以及 `pet_state.json`
的加载与原子落盘，同时向上层与子服务提供线程安全的偏好设置、异构集群分槽、隧道缓存与
活跃会话查询接口。

全局锁获取偏序规范（全后端与所有子服务必须严格遵守，防止死锁）：
1. 高层动作锁：`_deploy_lock` / `_action_lock` / `_upload_lock` / `_idle_stop_lock` /
   `_fetch_lock` / `_keepalive_lock`
2. 主状态锁：`AppStateStore.lock`（即 `ColabPetBackend._lock`）
3. 远端运行态锁：`_comfy_lock`（**严禁**持 `_lock` 时嵌套获取 `_comfy_lock`，必须先释放 `_lock` 再取 `_comfy_lock`）
4. 叶子锁：`_pull_lock` / `_canvas_ext_lock` / `EventBus._lock`

并发安全硬规则：
- 严禁在持有任何锁的期间执行阻塞网络请求、远端探针或子进程 I/O。
"""

from __future__ import annotations

import json
import sys
import threading
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

_PET_DIR = Path(__file__).resolve().parent
_PACKAGE_DIR = _PET_DIR.parent
if str(_PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_DIR))

from colab_core.runtime import DEFAULT_PREFS  # noqa: E402

__all__ = ["AppStateStore"]


class AppStateStore:
    """线程安全的本地状态持久化与查询容器。"""

    def __init__(
        self,
        state_path: Path,
        initial_state: dict[str, Any] | None = None,
    ) -> None:
        self.state_path: Path = Path(state_path)
        self.lock: threading.RLock = threading.RLock()
        self._state: dict[str, Any] = dict(initial_state) if isinstance(initial_state, dict) else {}

    @property
    def state(self) -> dict[str, Any]:
        """返回底层状态字典引用（兼容既有模块与单测在持锁下直接读写）。"""
        return self._state

    @state.setter
    def state(self, value: dict[str, Any]) -> None:
        """替换底层状态字典（用于初始化装载或测试重置）。"""
        with self.lock:
            self._state = value

    def load(self, loader_fn: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        """调用外部装载器加载状态，并兼容恢复磁盘中的多号异构分槽 `cluster_slots` 字段。"""
        loaded = loader_fn()
        if not isinstance(loaded, dict):
            loaded = {}
        if "cluster_slots" not in loaded and self.state_path.exists():
            restored_slots = self._read_cluster_slots_from_disk()
            if restored_slots is not None:
                loaded["cluster_slots"] = restored_slots
        with self.lock:
            self._state = loaded
            return self._state

    def _read_cluster_slots_from_disk(self) -> dict[str, dict[str, Any]] | None:
        """从磁盘状态文件中容错恢复 `cluster_slots` 分槽记录。"""
        try:
            raw_disk = json.loads(self.state_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        raw_cs = raw_disk.get("cluster_slots") if isinstance(raw_disk, dict) else None
        if not isinstance(raw_cs, dict):
            return None
        clean_cs: dict[str, dict[str, Any]] = {}
        for slot_key in ("llm", "comfy"):
            slot_val = raw_cs.get(slot_key)
            if isinstance(slot_val, dict) and slot_val.get("account_id"):
                clean_cs[slot_key] = dict(slot_val)
        return clean_cs

    def save(self) -> None:
        """将当前状态以原子 `.tmp` 替换方式安全持久化到 `self.state_path`。"""
        with self.lock:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self.state_path.with_suffix(".tmp")
            payload = json.dumps(self._state, ensure_ascii=False, indent=2)
            tmp_path.write_text(payload, encoding="utf-8")
            tmp_path.replace(self.state_path)

    def get_prefs(self) -> dict[str, Any]:
        """返回当前用户偏好设置的浅拷贝。"""
        with self.lock:
            prefs = self._state.get("prefs")
            return dict(prefs) if isinstance(prefs, dict) else {}

    def get_active_account_id(self) -> str:
        """返回当前激活账号 ID（未绑定时返回空字符串）。"""
        with self.lock:
            prefs = self._state.get("prefs")
            if not isinstance(prefs, dict):
                return ""
            return str(prefs.get("active_account_id") or "").strip()

    def update_prefs(
        self,
        updates: dict[str, Any] | None,
        sanitize_fn: Callable[[dict[str, Any]], dict[str, Any]],
        blocked_keys: Iterable[str] = (),
        *,
        default_prefs: dict[str, Any] | None = None,
        persist: bool = True,
    ) -> dict[str, Any]:
        """更新偏好设置，自动拦截受限键（如本地工具路径）并按约定同步单选/多选配置。"""
        base_defaults = default_prefs if default_prefs is not None else DEFAULT_PREFS
        blocked = frozenset(blocked_keys)
        upd = updates or {}
        with self.lock:
            prefs = self._state.setdefault("prefs", dict(base_defaults))
            for key, val in upd.items():
                if key in base_defaults and key not in blocked:
                    prefs[key] = val
            if "boot_profile" in upd and "boot_profiles" not in upd:
                prefs["boot_profiles"] = [upd["boot_profile"]]
            sanitized = sanitize_fn(prefs)
            self._state["prefs"] = sanitized
            if persist:
                self.save()
            return dict(sanitized)

    def get_cluster_slots(self) -> dict[str, dict[str, Any]]:
        """返回当前异构集群分槽映射的浅拷贝。"""
        with self.lock:
            raw_slots = self._state.get("cluster_slots")
            if not isinstance(raw_slots, dict):
                return {}
            return {
                str(slot_k): dict(slot_v)
                for slot_k, slot_v in raw_slots.items()
                if isinstance(slot_v, dict)
            }

    def get_active_cluster_endpoints(self) -> set[str]:
        """返回当前异构集群中所有已记录端点的集合。"""
        with self.lock:
            raw_slots = self._state.get("cluster_slots")
            if not isinstance(raw_slots, dict):
                return set()
            endpoints: set[str] = set()
            for slot_v in raw_slots.values():
                if isinstance(slot_v, dict) and slot_v.get("endpoint"):
                    ep = str(slot_v["endpoint"]).strip()
                    if ep:
                        endpoints.add(ep)
            return endpoints

    def get_tunnel_cache(self) -> dict[str, str]:
        """返回端点到公网隧道 URL 的缓存浅拷贝。"""
        with self.lock:
            raw_cache = self._state.get("tunnel_cache")
            if not isinstance(raw_cache, dict):
                return {}
            return {str(k): str(v) for k, v in raw_cache.items()}

    def set_tunnel_cache(self, ep: str, url: str) -> None:
        """写入或更新指定端点的公网隧道 URL 缓存。"""
        clean_ep = str(ep or "").strip()
        clean_url = str(url or "").strip()
        if not clean_ep or not clean_url:
            return
        with self.lock:
            cache = self._state.setdefault("tunnel_cache", {})
            if isinstance(cache, dict):
                cache[clean_ep] = clean_url

    def pop_tunnel_cache(self, ep: str) -> str | None:
        """移除并返回指定端点的隧道 URL 缓存（不存在时返回 None）。"""
        clean_ep = str(ep or "").strip()
        if not clean_ep:
            return None
        with self.lock:
            cache = self._state.get("tunnel_cache")
            if isinstance(cache, dict):
                val = cache.pop(clean_ep, None)
                return str(val) if val is not None else None
            return None

    def find_endpoint_by_tunnel_url(self, url: str) -> str:
        """根据公网隧道 URL 反向查找对应端点 ID（未命中返回空字符串）。"""
        clean_url = str(url or "").strip().rstrip("/")
        if not clean_url:
            return ""
        with self.lock:
            cache = self._state.get("tunnel_cache")
            if not isinstance(cache, dict):
                return ""
            for cand_ep, cand_url in cache.items():
                if str(cand_url or "").strip().rstrip("/") == clean_url:
                    return str(cand_ep)
            return ""

    def get_last_sessions(self) -> list[dict[str, Any]]:
        """返回最近一次采集到的运行中会话列表副本。"""
        with self.lock:
            raw = self._state.get("last_sessions") or []
            return [dict(item) if isinstance(item, dict) else item for item in raw]

    def pick_target_session(self, preferred: str = "comfy") -> dict[str, Any] | None:
        """优先选择指定名称的活跃会话，未命中时回退首个活跃会话。"""
        with self.lock:
            sessions = list(self._state.get("last_sessions") or [])
        for sess in sessions:
            if isinstance(sess, dict) and sess.get("name") == preferred:
                return sess
        return sessions[0] if sessions else None

    def has_running_session(self) -> bool:
        """判断当前状态是否表明存在运行中的 Colab 实例或异构集群分槽。"""
        with self.lock:
            assignments_count = int(self._state.get("last_assignments_count") or 0)
            hourly_rate = float(self._state.get("last_rate_hourly") or 0.0)
            sessions = list(self._state.get("last_sessions") or [])
            raw_slots = self._state.get("cluster_slots")
            has_cluster = bool(
                isinstance(raw_slots, dict)
                and any(
                    slot_k in ("llm", "comfy") and isinstance(slot_v, dict)
                    for slot_k, slot_v in raw_slots.items()
                )
            )
        return assignments_count > 0 or hourly_rate > 0.01 or len(sessions) > 0 or has_cluster

    def record_managed_binding(
        self,
        session: str,
        endpoint: str,
        *,
        persist: bool = True,
        cli_scope: str | None = None,
    ) -> dict[str, str] | None:
        """记录当前托管实例与账号的绑定关系，并按需原子落盘。

        `cli_scope` 为部署该实例时 `colab_ctl up --account` 实际使用的账号作用域（空串表示走全局
        sessions.json）；传 None 时若端点未变则沿用已记录的作用域，端点变化则视为未知。
        """
        clean_ep = str(endpoint or "").strip()
        clean_session = str(session or "comfy").strip() or "comfy"
        if not clean_ep:
            return None
        with self.lock:
            acct_id = self.get_active_account_id()
            binding = {
                "account_id": acct_id,
                "session": clean_session,
                "endpoint": clean_ep,
            }
            prev = self._state.get("managed_binding")
            if cli_scope is not None:
                binding["cli_scope"] = str(cli_scope or "").strip()
            elif isinstance(prev, dict) and str(prev.get("endpoint") or "") == clean_ep and "cli_scope" in prev:
                binding["cli_scope"] = str(prev.get("cli_scope") or "").strip()
            self._state["managed_binding"] = binding
            if persist:
                self.save()
            return dict(binding)

    def resolve_single_node_account_id(self, ep: str = "") -> str:
        """返回单机实例部署时实际使用的 colab-cli 账号作用域（未记录或端点不符时返回空串，保持旧行为）。"""
        clean_ep = str(ep or "").strip()
        with self.lock:
            mb = self._state.get("managed_binding")
            if not isinstance(mb, dict) or "cli_scope" not in mb:
                return ""
            mb_ep = str(mb.get("endpoint") or "").strip()
            if clean_ep and mb_ep and mb_ep != clean_ep:
                return ""
            return str(mb.get("cli_scope") or "").strip()
