"""纯内存态远端元数据与密钥凭据容器（EphemeralSecretStore）。

负责独占保存运行时远端自描述元数据与公网隧道凭据映射，严格遵守以下安全红线：
1. 所有密钥仅存于进程内存，严禁写入 `pet_state.json` 或任何持久化磁盘文件。
2. ComfyUI 公网凭据严格按 `(account_id, endpoint, origin)` 三元组同源同号校验，
   严禁跨源、跨账号或单密钥盲猜回退。
3. 锁协议与 `AppStateStore.lock` 保持一致（由 `ColabPetBackend` 注入同一把 `RLock`）。
"""

from __future__ import annotations

import sys
import threading
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

_PET_DIR = Path(__file__).resolve().parent
_PACKAGE_DIR = _PET_DIR.parent
if str(_PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_DIR))

from colab_security import url_origin_key  # noqa: E402

__all__ = ["EphemeralSecretStore"]

_VALID_SERVICE_KINDS = frozenset({"comfy", "llm", "dual"})


def _normalize_service_kind(raw_kind: Any) -> str:
    """将远端服务类型规范化为 comfy / llm / dual 之一。"""
    kind = str(raw_kind or "comfy").strip().lower()
    return kind if kind in _VALID_SERVICE_KINDS else "comfy"


def _add_origin_if_valid(origins: set[str], raw_url: Any) -> None:
    """提取合法 URL 的源站键并加入候选集合。"""
    origin = url_origin_key(str(raw_url or ""))
    if origin:
        origins.add(origin)


class EphemeralSecretStore:
    """纯内存态远端元数据与 Comfy/LLM 密钥凭据存储。"""

    def __init__(self, lock: threading.RLock | None = None) -> None:
        self._lock: threading.RLock = lock if lock is not None else threading.RLock()
        self.service_meta_by_ep: dict[str, dict[str, Any]] = {}
        self.runtime_secrets: dict[str, str] = {}
        self.comfy_secrets_by_ep: dict[str, str] = {}
        self.comfy_key_bindings_by_ep: dict[str, tuple[str, str]] = {}
        self.comfy_key_checked_eps: set[str] = set()
        self.last_comfy_key_hydrate_m: dict[str, float] = {}

    def _ep_dict_stores(self) -> tuple[dict[str, Any], ...]:
        """返回所有以端点 ID 为键的内存字典视图。"""
        return (
            self.service_meta_by_ep,
            self.runtime_secrets,
            self.comfy_secrets_by_ep,
            self.comfy_key_bindings_by_ep,
        )

    def drop_endpoint(self, ep: str) -> None:
        """清除单个端点对应的全部内存元数据、密钥与绑定缓存。"""
        clean_ep = str(ep or "").strip()
        if not clean_ep:
            return
        with self._lock:
            for store in self._ep_dict_stores():
                store.pop(clean_ep, None)
            self.comfy_key_checked_eps.discard(clean_ep)

    def retain_endpoints(self, keep_eps: Iterable[str]) -> None:
        """仅保留指定端点集合对应的内存凭据与元数据，其余全部清空。"""
        keep_set = {str(ep).strip() for ep in (keep_eps or ()) if str(ep).strip()}
        with self._lock:
            for store in self._ep_dict_stores():
                if not keep_set:
                    store.clear()
                else:
                    for ep_key in [k for k in store.keys() if k not in keep_set]:
                        store.pop(ep_key, None)
            if not keep_set:
                self.comfy_key_checked_eps.clear()
            else:
                self.comfy_key_checked_eps.intersection_update(keep_set)

    def get_ep_meta_and_key(self, ep: str = "") -> tuple[dict[str, Any], str]:
        """查询指定端点的远端元数据与运行时密钥；未传端点时回退首个已知端点。"""
        with self._lock:
            if ep and ep in self.service_meta_by_ep:
                return dict(self.service_meta_by_ep[ep]), str(self.runtime_secrets.get(ep) or "")
            if not ep and self.service_meta_by_ep:
                first_ep, first_meta = next(iter(self.service_meta_by_ep.items()))
                return dict(first_meta), str(self.runtime_secrets.get(first_ep) or "")
            return {}, str(self.runtime_secrets.get(ep) or "") if ep else ""

    def get_ep_service_kind(self, ep: str = "", default_kind: str = "") -> str:
        """从内存自描述元数据中读取端点的服务形态，无记录时返回 `default_kind`。"""
        with self._lock:
            if ep and ep in self.service_meta_by_ep:
                return _normalize_service_kind(self.service_meta_by_ep[ep].get("kind"))
            if self.service_meta_by_ep:
                first_meta = next(iter(self.service_meta_by_ep.values()))
                return _normalize_service_kind(first_meta.get("kind"))
            return default_kind

    def set_ep_meta(
        self,
        ep: str,
        meta: dict[str, Any],
        *,
        runtime_secret: str | None = None,
    ) -> None:
        """写入指定端点的远端自描述元数据，并可选同步更新其运行时密钥。"""
        clean_ep = str(ep or "").strip()
        if not clean_ep:
            return
        with self._lock:
            self.service_meta_by_ep[clean_ep] = dict(meta)
            if runtime_secret is not None:
                self.set_runtime_secret(clean_ep, runtime_secret)

    def set_runtime_secret(self, ep: str, secret: str) -> None:
        """设置或清除指定端点的运行时密钥（如 LLM API Key）。"""
        clean_ep = str(ep or "").strip()
        clean_secret = str(secret or "").strip()
        if not clean_ep:
            return
        with self._lock:
            if clean_secret:
                self.runtime_secrets[clean_ep] = clean_secret
            else:
                self.runtime_secrets.pop(clean_ep, None)

    def set_comfy_secret(
        self,
        ep: str,
        secret: str,
        *,
        account_id: str = "",
        origin: str = "",
    ) -> None:
        """设置或清除指定端点的 ComfyUI 密钥，并在提供源站时写入 `(account_id, origin)` 绑定。"""
        clean_ep = str(ep or "").strip()
        clean_secret = str(secret or "").strip()
        if not clean_ep:
            return
        with self._lock:
            if not clean_secret:
                self.comfy_secrets_by_ep.pop(clean_ep, None)
                self.comfy_key_bindings_by_ep.pop(clean_ep, None)
                return
            self.comfy_secrets_by_ep[clean_ep] = clean_secret
            norm_origin = url_origin_key(str(origin or "").strip())
            if norm_origin:
                self.comfy_key_bindings_by_ep[clean_ep] = (
                    str(account_id or "").strip(),
                    norm_origin,
                )

    def known_comfy_secret_values(self) -> set[str]:
        """返回当前内存中全部已知的 ComfyUI 密钥值集合（用于拦截错源 Bearer 头）。"""
        with self._lock:
            return {str(val) for val in self.comfy_secrets_by_ep.values() if val}

    def collect_candidate_origins(
        self,
        cand_ep: str,
        tunnel_cache: dict[str, Any] | None = None,
        comfy_slot: dict[str, Any] | None = None,
    ) -> tuple[set[str], str]:
        """汇总某端点所有可信来源的源站集合与已绑定账号 ID。"""
        cand_origins: set[str] = set()
        bound_acct = ""
        with self._lock:
            binding = self.comfy_key_bindings_by_ep.get(cand_ep)
            if isinstance(binding, tuple) and len(binding) == 2:
                bound_acct = str(binding[0] or "").strip()
                raw_bound_orig = str(binding[1] or "").strip()
                if raw_bound_orig:
                    cand_origins.add(raw_bound_orig)
                    _add_origin_if_valid(cand_origins, raw_bound_orig)

            if isinstance(tunnel_cache, dict):
                _add_origin_if_valid(cand_origins, tunnel_cache.get(cand_ep))

            meta = self.service_meta_by_ep.get(cand_ep)
            if isinstance(meta, dict) and str(meta.get("kind") or "comfy").lower() != "llm":
                _add_origin_if_valid(cand_origins, meta.get("comfy_tunnel_url"))
                _add_origin_if_valid(cand_origins, meta.get("tunnel_url"))

        if (
            isinstance(comfy_slot, dict)
            and not bool(comfy_slot.get("binding_invalid"))
            and str(comfy_slot.get("endpoint") or "").strip() == cand_ep
        ):
            _add_origin_if_valid(cand_origins, comfy_slot.get("tunnel_url"))
            bound_acct = bound_acct or str(comfy_slot.get("account_id") or "").strip()

        return cand_origins, bound_acct

    def _match_endpoint_key(
        self,
        cand_ep: str,
        target_origin: str,
        req_acct: str,
        tunnel_cache: dict[str, Any] | None,
        comfy_slot: dict[str, Any] | None,
    ) -> str:
        """校验单个候选端点是否同时满足同源与同账号约束，命中则返回密钥。"""
        if not cand_ep or cand_ep not in self.comfy_secrets_by_ep:
            return ""
        cand_origins, bound_acct = self.collect_candidate_origins(
            cand_ep,
            tunnel_cache=tunnel_cache,
            comfy_slot=comfy_slot,
        )
        if target_origin not in cand_origins:
            return ""
        if bound_acct and req_acct and bound_acct != req_acct:
            return ""
        return str(self.comfy_secrets_by_ep.get(cand_ep) or "")

    def get_comfy_key_for_url(
        self,
        base_url: str = "",
        ep: str = "",
        req_acct: str = "",
        tunnel_cache: dict[str, Any] | None = None,
        comfy_slot: dict[str, Any] | None = None,
    ) -> str:
        """按 `(account_id, endpoint, origin)` 严格匹配 ComfyUI 密钥，禁止跨源/跨号/单密钥回退。"""
        target_origin = url_origin_key(base_url)
        if not target_origin:
            return ""
        clean_ep = str(ep or "").strip()
        clean_acct = str(req_acct or "").strip()
        with self._lock:
            if clean_ep:
                return self._match_endpoint_key(
                    clean_ep, target_origin, clean_acct, tunnel_cache, comfy_slot
                )
            for cand_ep in list(self.comfy_secrets_by_ep.keys()):
                hit = self._match_endpoint_key(
                    cand_ep, target_origin, clean_acct, tunnel_cache, comfy_slot
                )
                if hit:
                    return hit
            return ""

    def should_skip_hydrate(
        self,
        key_tag: str,
        *,
        force: bool = False,
        now_m: float | None = None,
        cooldown_sec: float = 8.0,
    ) -> bool:
        """判断指定标识是否处于补水冷却期内；未跳过时原子登记本次补水时间戳。"""
        clean_tag = str(key_tag or "__default__").strip() or "__default__"
        current_m = time.monotonic() if now_m is None else float(now_m)
        with self._lock:
            last_m = float(self.last_comfy_key_hydrate_m.get(clean_tag) or 0.0)
            if (not force) and (current_m - last_m) < cooldown_sec:
                return True
            self.last_comfy_key_hydrate_m[clean_tag] = current_m
            return False

    def record_hydrated_binding(
        self,
        ep: str,
        account_id: str,
        fast_url: str = "",
    ) -> str:
        """在远端补水成功后登记 `(account_id, live_origin)` 绑定，返回生效源站键。"""
        clean_ep = str(ep or "").strip()
        if not clean_ep:
            return ""
        clean_acct = str(account_id or "").strip()
        with self._lock:
            meta = self.service_meta_by_ep.get(clean_ep) or {}
            if (
                clean_ep in self.comfy_secrets_by_ep
                and str(meta.get("kind") or "comfy").lower() != "llm"
            ):
                live_orig = url_origin_key(str(meta.get("comfy_tunnel_url") or fast_url or "").strip())
                if live_orig:
                    self.comfy_key_bindings_by_ep[clean_ep] = (clean_acct, live_orig)
                    return live_orig
            return ""
