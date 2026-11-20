"""Colab 桌宠 ComfyUI / LLM 监控轮询、WebSocket 进度、自动/手动收图与文件投递子服务。"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import json
import os
import re
import tempfile
import threading
import urllib.parse
import uuid
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any

from colab_core.colab_client import COLAB_REMOTE_LORA_PULL_SCRIPT
from colab_core.file_sync import (
    GDRIVE_LORA_REMOTE,
    GDRIVE_WF_REMOTE,
    LLM_IMAGE_EXTS,
    LLM_TEXT_EXTS,
    MEDIA_EXTS,
    MODEL_EXTS,
    WORKFLOW_EXTS,
    StreamingMultipartFile,
    api_prompt_to_ui_workflow,
    ensure_workflow_uuid,
    extract_png_comfy_workflow,
    format_timestamped_filename,
    is_api_prompt_dict,
    parse_http_mtime,
    resolve_non_clobber_dest,
    safe_output_dest,
    sanitize_filename,
)
from colab_core.runtime import (
    CREATE_NO_WINDOW,
    MAX_PENDING_PULL_ITEMS,
    clean_subprocess_env,
    collect_formal_outputs,
    extract_history_error,
    extract_slot_tps,
    format_comfy_error,
    normalize_profiles,
    short_gpu_name,
    suppress_logged,
)
from colab_security import redact_secrets, url_origin_key

if TYPE_CHECKING:
    from colab_backend import ColabPetBackend

# 离线干净态必须全为假/空值的状态字段清单
_OFFLINE_FALSY_FIELDS: tuple[str, ...] = (
    "online",
    "comfy_online",
    "llm_online",
    "ws_connected",
    "has_error",
    "error_summary",
    "running_count",
    "pending_count",
    "llm_running_count",
    "active_profiles",
    "step_value",
    "vram_total_gb",
    "gpu_name",
    "model_name",
    "last_output_count",
    "last_output_name",
)

# 宿主机遥测中需在 LLM / Dual 轮询时同步写入 _comfy_state 的基础硬件字段
_TELEMETRY_SYNC_DEFAULTS: tuple[tuple[str, Any], ...] = (
    ("gpu_name", ""),
    ("gpu_full_name", ""),
    ("vram_used_gb", 0.0),
    ("vram_total_gb", 0.0),
    ("vram_pct", 0.0),
    ("ram_used_gb", 0.0),
    ("ram_total_gb", 0.0),
    ("ram_free_gb", 0.0),
    ("ram_pct", 0.0),
    ("disk_used_gb", 0.0),
    ("disk_total_gb", 0.0),
    ("disk_free_gb", 0.0),
    ("disk_pct", 0.0),
    ("shm_used_gb", 0.0),
    ("shm_total_gb", 0.0),
    ("shm_free_gb", 0.0),
    ("drive_free_gb", 0.0),
    ("drive_total_gb", 0.0),
)


def _clean_gpu_full_name(raw: str) -> str:
    """清洗设备完整名称（去除 cuda:0 前缀与分配器后缀，保留可读型号全称）。"""
    txt = str(raw or "").strip()
    if not txt:
        return ""
    txt = re.sub(r"^cuda:\d+\s+", "", txt, flags=re.IGNORECASE)
    txt = re.sub(r"\s+:\s+\S+$", "", txt)
    return txt.strip()


def _round_gb(raw: Any) -> float:
    """将远端上报的 GB / 百分比数值安全规整到一位小数。"""
    return round(float(raw or 0.0), 1)


def _parse_host_metrics(raw: dict[str, Any] | None) -> dict[str, Any]:
    """统一解析远端宿主机 RAM / 磁盘 / /dev/shm / 云盘遥测指标。"""
    src = raw if isinstance(raw, dict) else {}
    return {
        "ram_used_gb": _round_gb(src.get("ram_used_gb")),
        "ram_total_gb": _round_gb(src.get("ram_total_gb")),
        "ram_free_gb": _round_gb(src.get("ram_free_gb")),
        "ram_pct": _round_gb(src.get("ram_pct")),
        "disk_used_gb": _round_gb(src.get("disk_used_gb")),
        "disk_total_gb": _round_gb(src.get("disk_total_gb")),
        "disk_free_gb": _round_gb(src.get("disk_free_gb")),
        "disk_pct": _round_gb(src.get("disk_pct")),
        "shm_used_gb": _round_gb(src.get("shm_used_gb")),
        "shm_total_gb": _round_gb(src.get("shm_total_gb")),
        "shm_free_gb": _round_gb(src.get("shm_free_gb")),
        "drive_persisted": bool(src.get("drive_persisted", False)),
        "drive_free_gb": _round_gb(src.get("drive_free_gb")),
        "drive_total_gb": _round_gb(src.get("drive_total_gb")),
    }


def _apply_host_metrics_locked(
    comfy_state: dict[str, Any],
    metrics: dict[str, Any],
    *,
    write_drive: bool,
    drive_requires_total: bool = False,
) -> None:
    """在持有 `_comfy_lock` 期间将已解析的宿主机遥测字段按有效容量写入状态字典。"""
    if metrics["ram_total_gb"] > 0:
        comfy_state["ram_used_gb"] = metrics["ram_used_gb"]
        comfy_state["ram_total_gb"] = metrics["ram_total_gb"]
        comfy_state["ram_free_gb"] = metrics["ram_free_gb"]
        comfy_state["ram_pct"] = metrics["ram_pct"]
    if metrics["disk_total_gb"] > 0:
        comfy_state["disk_used_gb"] = metrics["disk_used_gb"]
        comfy_state["disk_total_gb"] = metrics["disk_total_gb"]
        comfy_state["disk_free_gb"] = metrics["disk_free_gb"]
        comfy_state["disk_pct"] = metrics["disk_pct"]
    if metrics["shm_total_gb"] > 0:
        comfy_state["shm_used_gb"] = metrics["shm_used_gb"]
        comfy_state["shm_total_gb"] = metrics["shm_total_gb"]
        comfy_state["shm_free_gb"] = metrics["shm_free_gb"]
    if write_drive:
        comfy_state["drive_persisted"] = bool(metrics["drive_persisted"])
        if (not drive_requires_total) or metrics["drive_total_gb"] > 0:
            comfy_state["drive_free_gb"] = metrics["drive_free_gb"]
            comfy_state["drive_total_gb"] = metrics["drive_total_gb"]


def _extract_telemetry_patch(info: dict[str, Any]) -> dict[str, Any]:
    """从 LLM 探测结果中抽取硬件与宿主机遥测补丁字典。"""
    patch = {key: info.get(key, default) for key, default in _TELEMETRY_SYNC_DEFAULTS}
    patch["drive_persisted"] = bool(info.get("drive_persisted", False))
    return patch


def _parse_comfy_system_stats(stats: dict[str, Any]) -> dict[str, Any]:
    """从 ComfyUI `/system_stats` 响应中提取 GPU 型号、显存及兜底宿主机内存。"""
    gpu_name, gpu_full_name, v_used, v_total, v_pct = "", "", 0.0, 0.0, 0.0
    devices = stats.get("devices") or []
    if devices and isinstance(devices[0], dict):
        d0 = devices[0]
        raw_dev_name = str(d0.get("name", "") or "")
        gpu_name = short_gpu_name(raw_dev_name)
        gpu_full_name = _clean_gpu_full_name(raw_dev_name)
        tot_b = float(d0.get("vram_total") or 0.0)
        free_b = float(d0.get("vram_free") or 0.0)
        used_b = max(0.0, tot_b - free_b)
        if tot_b > 0:
            v_total = round(tot_b / (1024**3), 1)
            v_used = round(used_b / (1024**3), 1)
            v_pct = round(used_b * 100.0 / tot_b, 1)

    ram_used, ram_total, ram_free, ram_pct = 0.0, 0.0, 0.0, 0.0
    sys_info = stats.get("system") or {}
    if isinstance(sys_info, dict):
        rt_b = float(sys_info.get("ram_total") or 0.0)
        rf_b = float(sys_info.get("ram_free") or 0.0)
        if rt_b > 0:
            ru_b = max(0.0, rt_b - rf_b)
            ram_total = round(rt_b / (1024**3), 1)
            ram_free = round(rf_b / (1024**3), 1)
            ram_used = round(ru_b / (1024**3), 1)
            ram_pct = round(ru_b * 100.0 / rt_b, 1)

    return {
        "gpu_name": gpu_name,
        "gpu_full_name": gpu_full_name,
        "vram_used_gb": v_used,
        "vram_total_gb": v_total,
        "vram_pct": v_pct,
        "ram_used_gb": ram_used,
        "ram_total_gb": ram_total,
        "ram_free_gb": ram_free,
        "ram_pct": ram_pct,
    }


def _match_saved_item_index(
    saved_path: Path,
    meta_parts: list[tuple[str, str, str] | None],
    matched: set[int],
) -> int | None:
    """按精确文件名 -> 时间戳变体 -> 前缀回退顺序匹配待拉取条目下标。"""
    pn, ps, pf = saved_path.name.lower(), saved_path.stem.lower(), saved_path.suffix.lower()
    for mode in (1, 2, 3):
        for idx, mp in enumerate(meta_parts):
            if idx in matched or mp is None:
                continue
            mn, ms, mf = mp
            if mode == 1 and pn == mn:
                return idx
            if mode == 2 and pf == mf and (
                ps == ms or re.fullmatch(rf"{re.escape(ms)}_?\d{{8}}-\d{{6}}(?:-\d+)?", ps) is not None
            ):
                return idx
            if mode == 3 and pf == mf and ps.startswith(ms):
                return idx
    return next((idx for idx in range(len(meta_parts)) if idx not in matched), None)


class ComfySyncService:
    """负责 ComfyUI/LLM 在线探针、WS 监听、非覆写收图与拖拽文件分流上传。"""

    MAX_PENDING_PULL_ITEMS = MAX_PENDING_PULL_ITEMS

    def __init__(self, backend: ColabPetBackend) -> None:
        self._b = backend
        self._store = backend._store
        self._secrets = backend._secrets
        self._bus = backend._bus
        self._ctx = backend._ctx
        self._bridge_hm_lock = threading.Lock()
        self._bridge_hm_last_ts: dict[str, float] = {}

    # ------------------------------------------------------------------
    # 老实例桥接遥测异步兜底
    # ------------------------------------------------------------------

    def _resolve_bridge_metrics_target(
        self, base: str, session_hint: str
    ) -> tuple[str, str, str]:
        """解析桥接宿主机遥测目标 `(endpoint, session_name, slot_account_id)`。"""
        b = self._b
        target = self._store.pick_target_session(session_hint)
        use_ep = str((target or {}).get("endpoint") or "").strip()
        if (not use_ep) and session_hint != "comfy":
            target = self._store.pick_target_session("comfy")
            use_ep = str((target or {}).get("endpoint") or "").strip()
        sname = str((target or {}).get("name") or session_hint or "comfy").strip() or "comfy"
        if sname == "?":
            sname = session_hint or "comfy"
        slot_acct = b._cluster_svc.resolve_account_for_endpoint(ep=use_ep, base_url=base)
        if (not slot_acct) and b._cluster_svc.has_active_slots():
            slot_key = "llm" if session_hint == "llm" else "comfy"
            hint_slot = b._cluster_svc.get_slot(slot_key) or b._cluster_svc.get_slot("comfy")
            if hint_slot:
                slot_acct = str(hint_slot.get("account_id") or "").strip()
                if (not use_ep) and hint_slot.get("endpoint"):
                    use_ep = str(hint_slot["endpoint"])
        return use_ep, sname, slot_acct

    def _run_bridge_host_metrics_probe(
        self,
        colab_py: Path,
        use_ep: str,
        sname: str,
        slot_acct: str,
        start_epoch: int,
    ) -> None:
        """执行一次无窗桥接宿主机指标探测并刷新 `_comfy_state` 与快照。"""
        b = self._b
        cmd = [str(colab_py), str(self._ctx.cli_bridge_script), "host-metrics", use_ep, sname]
        if slot_acct:
            cmd.append(f"--acct={slot_acct}")
        proc = self._ctx.subprocess_mod.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
            env=clean_subprocess_env(),
            creationflags=CREATE_NO_WINDOW,
        )
        matched = re.search(
            r"<<<COLAB_HOST_METRICS>>>(\{.*?\})<<<END_COLAB_HOST_METRICS>>>",
            proc.stdout or "",
            flags=re.DOTALL,
        )
        if not matched:
            return
        data = json.loads(matched.group(1))
        if not isinstance(data, dict) or not data.get("ok"):
            return
        with self._store.lock:
            if b._account_epoch != start_epoch:
                return
        metrics = _parse_host_metrics(data)
        with b._comfy_lock:
            _apply_host_metrics_locked(
                b._comfy_state, metrics, write_drive=True, drive_requires_total=True
            )
            cstate = dict(b._comfy_state)
        with self._store.lock:
            if b._account_epoch != start_epoch:
                return
            b.last_snapshot["comfy"] = cstate
        self._bus.emit({"type": "comfy_tick", "comfy": cstate})

    def _maybe_trigger_bridge_host_metrics_async(
        self, base: str, session_hint: str = "comfy"
    ) -> None:
        """兼容老实例：当远端尚无新路由(404)时，按 30s 节流低频通过空闲内核兜底采集磁盘与内存盘指标。"""
        b = self._b
        if b._monitor_stop.is_set() or b.is_deploying():
            return
        colab_py = self._ctx.get_colab_py()
        if (not colab_py.is_file()) or (not self._ctx.cli_bridge_script.is_file()):
            return
        use_ep, sname, slot_acct = self._resolve_bridge_metrics_target(base, session_hint)
        if not use_ep:
            return

        cache_key = f"{slot_acct}:{use_ep}"
        if not self._bridge_hm_lock.acquire(blocking=False):
            return

        started = False
        ts_updated = False
        had_prev_ts = False
        prev_ts = 0.0
        try:
            now_m = self._ctx.monotonic()
            if now_m - float(self._bridge_hm_last_ts.get(cache_key) or 0.0) < 30.0:
                return
            had_prev_ts = cache_key in self._bridge_hm_last_ts
            prev_ts = float(self._bridge_hm_last_ts.get(cache_key) or 0.0)
            self._bridge_hm_last_ts[cache_key] = now_m
            ts_updated = True
            with self._store.lock:
                start_epoch = b._account_epoch

            def _worker() -> None:
                try:
                    with suppress_logged("comfy.bridge_hm", throttled_key=f"bridge_hm:{use_ep or sname}"):
                        self._run_bridge_host_metrics_probe(
                            colab_py, use_ep, sname, slot_acct, start_epoch
                        )
                finally:
                    self._bridge_hm_lock.release()

            # 保留模块级 threading.Thread 以便单测验证线程启动异常时的锁释放与时间戳回滚
            threading.Thread(target=_worker, daemon=True).start()
            started = True
        finally:
            if not started:
                if ts_updated:
                    if had_prev_ts:
                        self._bridge_hm_last_ts[cache_key] = prev_ts
                    else:
                        self._bridge_hm_last_ts.pop(cache_key, None)
                self._bridge_hm_lock.release()

    # ------------------------------------------------------------------
    # 错误确认与离线重置
    # ------------------------------------------------------------------

    def acknowledge_comfy_error(self) -> None:
        """用户点击报错气泡前往处理后，清除当前红标报警。"""
        b = self._b
        with b._comfy_lock:
            b._comfy_state["has_error"] = False
            b._comfy_state["error_summary"] = ""
            cstate = dict(b._comfy_state)
        self._bus.emit({"type": "comfy_tick", "comfy": cstate})

    @staticmethod
    def _is_clean_offline_state(
        cur: dict[str, Any], inferred_kind: str, pending_cnt: int
    ) -> bool:
        """检查当前运行态字典是否已处于目标服务类型的离线干净态。"""
        return (
            cur.get("service_kind") == inferred_kind
            and cur.get("pending_pull_count") == pending_cnt
            and cur.get("tunnel_url") is None
            and cur.get("llm_tunnel_url") is None
            and all(not cur.get(field) for field in _OFFLINE_FALSY_FIELDS)
        )

    def reset_comfy_offline(self) -> None:
        """将 ComfyUI/LLM 运行态重置为离线干净态；若已处于离线干净态则短路返回。"""
        b = self._b
        if b._ws_app is not None or b._ws_target_url is not None:
            b._stop_ws()
        inferred_kind = b._get_ep_service_kind() if self._store.has_running_session() else "comfy"
        with b._comfy_lock:
            cur = b._comfy_state
            pending_cnt = len(b._pending_pull_items)
            if self._is_clean_offline_state(cur, inferred_kind, pending_cnt):
                return
            was_online = bool(cur.get("online"))
            prev_kind = str(cur.get("service_kind") or "comfy")
            b._comfy_state = b._default_comfy_state()
            b._comfy_state["service_kind"] = inferred_kind
            b._comfy_state["pending_pull_count"] = pending_cnt
            cstate = dict(b._comfy_state)
        with self._store.lock:
            b.last_snapshot["comfy"] = cstate
        if was_online or prev_kind != inferred_kind:
            self._bus.emit({"type": "comfy_tick", "comfy": cstate})

    # ------------------------------------------------------------------
    # 待拉取队列管理与持久化
    # ------------------------------------------------------------------

    @staticmethod
    def _pending_item_key(item: tuple[str, dict[str, str]]) -> tuple[str, str, str, str]:
        """生成待拉取成图条目的规范化去重键。"""
        base_url, meta = item
        meta_dict = meta if isinstance(meta, dict) else {}
        return (
            str(base_url or "").rstrip("/"),
            str(meta_dict.get("subfolder") or ""),
            str(meta_dict.get("filename") or ""),
            str(meta_dict.get("type") or "output"),
        )

    @classmethod
    def _merge_pending_items(
        cls,
        existing: list[tuple[str, dict[str, str]]],
        incoming: list[tuple[str, dict[str, str]]],
        *,
        max_items: int | None = None,
    ) -> list[tuple[str, dict[str, str]]]:
        """按规范化键去重并保序合并待拉取成图列表，超出容量时保留最新尾部条目。"""
        merged: list[tuple[str, dict[str, str]]] = []
        seen: set[tuple[str, str, str, str]] = set()
        for item in (*existing, *incoming):
            key = cls._pending_item_key(item)
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)
        cap = max(1, int(max_items or cls.MAX_PENDING_PULL_ITEMS))
        return merged[-cap:] if len(merged) > cap else merged

    @staticmethod
    def _remove_succeeded_pending_items(
        existing: list[tuple[str, dict[str, str]]],
        succeeded: list[tuple[str, dict[str, str]]],
    ) -> list[tuple[str, dict[str, str]]]:
        """从现有待拉取队列中精准剔除本次已成功的同键项，保留其余历史待重试项。"""
        if not succeeded:
            return list(existing)
        done_keys = {ComfySyncService._pending_item_key(item) for item in succeeded}
        return [
            item
            for item in existing
            if ComfySyncService._pending_item_key(item) not in done_keys
        ]

    @staticmethod
    def _split_succeeded_items(
        attempted: list[tuple[str, dict[str, str]]],
        failed: list[tuple[str, dict[str, str]]],
    ) -> list[tuple[str, dict[str, str]]]:
        """根据本次尝试列表与失败列表的差集计算成功条目。"""
        rem_fail = Counter(ComfySyncService._pending_item_key(item) for item in failed)
        succeeded: list[tuple[str, dict[str, str]]] = []
        for item in attempted:
            key = ComfySyncService._pending_item_key(item)
            if rem_fail.get(key, 0) > 0:
                rem_fail[key] -= 1
            else:
                succeeded.append(item)
        return succeeded

    @classmethod
    def _serialize_pending_items(
        cls,
        items: list[tuple[str, dict[str, str]]],
    ) -> list[dict[str, str]]:
        """将待拉取产出列表序列化为可持久化 JSON 对象数组（与内存共用容量上限）。"""
        out: list[dict[str, str]] = []
        for base_url, meta in items:
            meta_d = meta if isinstance(meta, dict) else {}
            b_url = str(base_url or "").strip().rstrip("/")
            fn = str(meta_d.get("filename") or "").strip()
            if not b_url or not fn:
                continue
            out.append(
                {
                    "base_url": b_url,
                    "filename": fn,
                    "subfolder": str(meta_d.get("subfolder") or ""),
                    "type": str(meta_d.get("type") or "output"),
                }
            )
        cap = max(1, int(cls.MAX_PENDING_PULL_ITEMS))
        return out[-cap:] if len(out) > cap else out

    def _persist_pending_pull_items(self) -> None:
        """严守非嵌套锁规范：先快照 `_pending_pull_items`，释放 `_comfy_lock` 后再落盘。"""
        b = self._b
        with b._comfy_lock:
            serialized = self._serialize_pending_items(b._pending_pull_items)
        with self._store.lock:
            try:
                self._store.state["pending_pull_items"] = serialized
                self._store.save()
            except OSError as exc:
                b._diag.warn("comfy", f"持久化待拉取队列失败: {redact_secrets(str(exc))}")

    def _execute_pull_batch(
        self,
        items: list[tuple[str, dict[str, str]]],
        *,
        zero_count_on_empty: bool = False,
    ) -> tuple[list[Path], list[tuple[str, dict[str, str]]], dict[str, Any]]:
        """执行一批成图拉取，按严格状态差集更新内存与持久化待拉队列。"""
        b = self._b
        saved, failed = self._download_outputs_detailed_compat(items)
        if not saved and not failed:
            failed = list(items)
        succeeded = self._split_succeeded_items(items, failed)
        with b._comfy_lock:
            rem_pending = self._remove_succeeded_pending_items(b._pending_pull_items, succeeded)
            b._pending_pull_items = self._merge_pending_items(rem_pending, failed)
            b._comfy_state["pending_pull_count"] = len(b._pending_pull_items)
            if saved:
                b._last_downloaded_files = saved
                b._comfy_state["last_output_name"] = saved[-1].name
                b._comfy_state["last_output_count"] = len(saved)
            elif zero_count_on_empty:
                b._comfy_state["last_output_count"] = 0
            cstate = dict(b._comfy_state)
        self._persist_pending_pull_items()
        return saved, failed, cstate

    @staticmethod
    def _infer_failed_from_patched_saved(
        items: list[tuple[str, dict[str, str]]],
        saved: list[Path],
    ) -> list[tuple[str, dict[str, str]]]:
        """当单测 monkeypatch 了 `_download_comfy_outputs` 时，按落盘文件名尽力反推未成功项。"""
        if not saved:
            return list(items)
        if len(saved) >= len(items):
            return []
        meta_parts: list[tuple[str, str, str] | None] = []
        for _u, meta in items:
            raw_fn = sanitize_filename(str(meta.get("filename") or ""), "") if isinstance(meta, dict) else ""
            if not raw_fn:
                meta_parts.append(None)
                continue
            mp = Path(raw_fn)
            meta_parts.append((raw_fn.lower(), mp.stem.lower(), mp.suffix.lower()))

        matched: set[int] = set()
        for saved_path in map(Path, saved):
            picked = _match_saved_item_index(saved_path, meta_parts, matched)
            if picked is not None:
                matched.add(picked)
        return [item for idx, item in enumerate(items) if idx not in matched]

    def _is_download_patched(self) -> bool:
        """判断 `b._download_comfy_outputs` 或本服务下载方法是否被外部测试 monkeypatch。"""
        b = self._b
        backend_cls = type(b)
        facade_fn = getattr(b, "_download_comfy_outputs", None)
        cls_facade_fn = getattr(backend_cls, "_download_comfy_outputs", None)
        if facade_fn is None or cls_facade_fn is None:
            return True
        if getattr(facade_fn, "__func__", None) is not cls_facade_fn:
            return True
        if getattr(cls_facade_fn, "__wrapped__", None) is not ComfySyncService.download_comfy_outputs:
            return True
        return getattr(self.download_comfy_outputs, "__func__", None) is not ComfySyncService.download_comfy_outputs

    def _download_outputs_detailed_compat(
        self, items: list[tuple[str, dict[str, str]]]
    ) -> tuple[list[Path], list[tuple[str, dict[str, str]]]]:
        """未 patch 时直接执行逐项详细下载；若被单测 patch 则兼容调用原门面钩子并推导失败项。"""
        if not self._is_download_patched():
            return self.download_comfy_outputs_detailed(items)
        raw_saved = self._b._download_comfy_outputs(items)
        saved = [Path(p) for p in (raw_saved or [])]
        failed = self._infer_failed_from_patched_saved(items, saved)
        return saved, failed

    # ------------------------------------------------------------------
    # 成图流式校验下载
    # ------------------------------------------------------------------

    def _validate_download_response(self, resp: Any, raw_name: str) -> tuple[int | None, float]:
        """校验下载 HTTP 响应的 200 状态码、非 HTML 载荷与 Content-Length，返回 `(预期字节数, 修改时间戳)`。"""
        resp.raise_for_status()
        status_code = int(getattr(resp, "status_code", 0) or 0)
        if status_code != 200:
            raise RuntimeError(f"下载响应状态码异常: HTTP {status_code}（预期 200）")
        resp_headers = getattr(resp, "headers", None) or {}
        ctype = str(resp_headers.get("Content-Type") or resp_headers.get("content-type") or "").strip().lower()
        if ctype.startswith("text/html") and not raw_name.lower().endswith((".html", ".htm")):
            raise RuntimeError("下载响应为 HTML 页面，非预期作品文件")
        raw_cl = resp_headers.get("Content-Length") or resp_headers.get("content-length")
        expected_len: int | None = None
        if raw_cl is not None and str(raw_cl).strip():
            expected_len = int(str(raw_cl).strip())
            if expected_len <= 0:
                raise RuntimeError(f"下载响应 Content-Length 异常 ({expected_len})")
        mtime_sec = parse_http_mtime(resp_headers.get("Last-Modified") or resp_headers.get("last-modified"))
        if mtime_sec is None:
            mtime_sec = self._ctx.now()
        return expected_len, float(mtime_sec)

    def _download_single_output(
        self, base_url: str, meta: dict[str, str], tmp_path: Path
    ) -> Path:
        """执行单个 ComfyUI 正式产出文件的流式下载、完整性校验与非覆写原子落盘。"""
        b = self._b
        raw_name = str(meta.get("filename") or "") if isinstance(meta, dict) else ""
        if not isinstance(meta, dict) or not raw_name:
            raise ValueError("成图元数据无效或缺少 filename")
        url = f"{base_url.rstrip('/')}/view"
        subfolder = str(meta.get("subfolder") or "")
        params = {"filename": raw_name, "subfolder": subfolder, "type": meta.get("type", "output")}

        with b._comfy_http_get(
            url, base_url=base_url, params=params, stream=True, timeout=(8, 60)
        ) as resp:
            expected_len, mtime_sec = self._validate_download_response(resp, raw_name)
            digest = hashlib.sha256()
            total_bytes = 0
            with open(tmp_path, "wb") as fp:
                for chunk in resp.iter_content(chunk_size=65536):
                    if chunk:
                        fp.write(chunk)
                        digest.update(chunk)
                        total_bytes += len(chunk)
            if total_bytes <= 0:
                raise RuntimeError("下载响应正文为空（0 字节）")
            if expected_len is not None and total_bytes != expected_len:
                raise RuntimeError(f"下载文件字节数不完整 ({total_bytes}/{expected_len})")

        norm_sub = subfolder.replace("\\", "/").strip("/")
        preserve_orig = bool(
            self._ctx.preserve_original_sub_re.search(norm_sub)
            and raw_name.lower().endswith(".safetensors")
        )
        final_name = raw_name if preserve_orig else format_timestamped_filename(raw_name, mtime_sec)
        dest = safe_output_dest(b.download_root, subfolder, final_name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        final_dest, skip_write = resolve_non_clobber_dest(dest, total_bytes, digest.hexdigest())
        if skip_write:
            tmp_path.unlink(missing_ok=True)
        else:
            tmp_path.replace(final_dest)
            try:
                os.utime(final_dest, (mtime_sec, mtime_sec))
            except OSError:
                pass
        return final_dest

    def download_comfy_outputs_detailed(
        self, items: list[tuple[str, dict[str, str]]]
    ) -> tuple[list[Path], list[tuple[str, dict[str, str]]]]:
        """经由 ComfyUI /view 接口逐项下载正式产出文件，返回 `(已存本机路径列表, 失败条目列表)`。"""
        b = self._b
        saved: list[Path] = []
        failed: list[tuple[str, dict[str, str]]] = []
        try:
            b.download_root.mkdir(parents=True, exist_ok=True)
        except Exception as exc:  # noqa: BLE001
            b._diag.warn(
                "comfy",
                f"创建本地下载目录失败 ({redact_secrets(str(b.download_root))}): {redact_secrets(str(exc))}",
            )
            return [], list(items)
        for base_url, meta in items:
            raw_name = str(meta.get("filename") or "") if isinstance(meta, dict) else ""
            tmp_path = b.download_root / f".dl_{uuid.uuid4().hex[:12]}.part"
            try:
                saved.append(self._download_single_output(base_url, meta, tmp_path))
            except Exception as exc:  # noqa: BLE001
                try:
                    tmp_path.unlink(missing_ok=True)
                except OSError as rm_exc:
                    b._diag.warn("comfy", f"清理临时下载文件失败: {redact_secrets(str(rm_exc))}")
                b._diag.warn(
                    "comfy",
                    f"下载产出文件失败 ({redact_secrets(raw_name or 'unknown')}): {redact_secrets(str(exc))}",
                )
                failed.append((base_url, meta))
        return saved, failed

    def download_comfy_outputs(self, items: list[tuple[str, dict[str, str]]]) -> list[Path]:
        """经由 ComfyUI /view 接口下载正式产出文件，自动附加服务端修改时间戳并防止同名异图覆写。"""
        return self.download_comfy_outputs_detailed(items)[0]

    # ------------------------------------------------------------------
    # ComfyUI 在线探针拆解
    # ------------------------------------------------------------------

    def _fetch_comfy_queue_counts(self, base: str) -> tuple[int, int]:
        """探测 ComfyUI `/queue` 接口，返回 `(running_count, pending_count)`。"""
        with suppress_logged("comfy.queue", throttled_key=f"queue:{base}"):
            r_q = self._b._comfy_http_get(f"{base}/queue", base_url=base, timeout=4.0)
            if r_q.status_code == 200:
                qdata = r_q.json()
                return (
                    len(qdata.get("queue_running") or []),
                    len(qdata.get("queue_pending") or []),
                )
        return 0, 0

    def _enqueue_and_auto_pull_outputs(
        self, new_outputs: list[tuple[str, dict[str, str]]]
    ) -> list[dict[str, Any]]:
        """先 Write-Ahead 持久化入队新成图，再按偏好自动拉取并返回待推送事件列表。"""
        b = self._b
        with b._comfy_lock:
            merged_items = self._merge_pending_items(b._pending_pull_items, new_outputs)
            trimmed = (
                len(b._pending_pull_items) + len(new_outputs) > len(merged_items)
                and len(merged_items) >= self.MAX_PENDING_PULL_ITEMS
            )
            b._pending_pull_items = merged_items
            b._comfy_state["pending_pull_count"] = len(b._pending_pull_items)
            b._comfy_state["last_output_name"] = str((new_outputs[-1][1] or {}).get("filename") or "")
        if trimmed:
            b._diag.warn(
                "comfy",
                f"待拉取队列超出容量上限 ({self.MAX_PENDING_PULL_ITEMS})，已保留最新条目",
            )
        self._persist_pending_pull_items()

        auto_pull = bool(self._store.get_prefs().get("auto_pull_outputs", True))
        if not auto_pull:
            with b._comfy_lock:
                b._comfy_state["last_output_count"] = len(new_outputs)
            return [
                {
                    "type": "outputs_ready",
                    "count": len(new_outputs),
                    "pulled": False,
                    "summary": f"✨ 新出图 {len(new_outputs)} 份！点我拉回查看",
                }
            ]

        with b._pull_lock:
            wanted_keys = {self._pending_item_key(it) for it in new_outputs}
            with b._comfy_lock:
                to_pull = [
                    it for it in b._pending_pull_items if self._pending_item_key(it) in wanted_keys
                ]
            if not to_pull:
                return []
            saved, failed, _ = self._execute_pull_batch(to_pull, zero_count_on_empty=True)

        if saved and not failed:
            return [
                {
                    "type": "outputs_ready",
                    "count": len(saved),
                    "pulled": True,
                    "summary": f"✨ 新出图 {len(saved)} 份，已存本机！点我查看",
                    "files": [str(p) for p in saved],
                }
            ]
        if saved and failed:
            return [
                {
                    "type": "outputs_ready",
                    "count": len(saved),
                    "failed_count": len(failed),
                    "pulled": False,
                    "partial": True,
                    "is_warning": True,
                    "summary": f"⚠️ 已存本机 {len(saved)} 份，另有 {len(failed)} 份拉取失败！点我重试",
                    "files": [str(p) for p in saved],
                }
            ]
        return [
            {
                "type": "outputs_ready",
                "count": len(failed),
                "failed_count": len(failed),
                "pulled": False,
                "is_error": True,
                "summary": f"⚠️ 新出图 {len(failed)} 份自动拉取失败，尚未存本机！点我重试",
                "files": [],
            }
        ]

    def _process_comfy_history(self, base: str) -> list[dict[str, Any]]:
        """探测 ComfyUI `/history`，识别新增任务报错与正式产出文件。"""
        b = self._b
        events: list[dict[str, Any]] = []
        with suppress_logged("comfy.history", throttled_key=f"history:{base}"):
            r_hist = b._comfy_http_get(
                f"{base}/history", base_url=base, params={"max_items": 6}, timeout=5.0
            )
            if r_hist.status_code != 200:
                return events
            hdata = r_hist.json()
            if not isinstance(hdata, dict):
                return events
            first_conn = base not in b._seen_prompts_by_url
            seen = b._seen_prompts_by_url.setdefault(base, set())
            if first_conn:
                seen.update(str(pid) for pid in hdata.keys())
                return events

            new_pids = [str(pid) for pid in hdata.keys() if str(pid) not in seen]
            if new_pids:
                b.touch_activity()
            new_outputs: list[tuple[str, dict[str, str]]] = []
            latest_err: tuple[str, str] | None = None
            for pid in new_pids:
                seen.add(pid)
                entry = hdata.get(pid)
                if not isinstance(entry, dict):
                    continue
                err_str = extract_history_error(entry)
                if err_str and pid not in b._reported_error_prompts:
                    b._reported_error_prompts.add(pid)
                    latest_err = (pid, err_str)
                for meta in collect_formal_outputs(entry):
                    new_outputs.append((base, meta))

            if latest_err:
                with b._comfy_lock:
                    b._comfy_state["has_error"] = True
                    b._comfy_state["error_summary"] = latest_err[1]
                events.append(
                    {"type": "comfy_error", "summary": latest_err[1], "prompt_id": latest_err[0]}
                )
            if new_outputs:
                events.extend(self._enqueue_and_auto_pull_outputs(new_outputs))
        return events

    def _fetch_comfy_host_metrics(self, base: str) -> dict[str, Any]:
        """探测直铺节点 `/colab_pet/host_metrics`，遇到 404 时按 30s 节流触发桥接兜底。"""
        with suppress_logged("comfy.host_metrics", throttled_key=f"comfy_hm:{base}"):
            r_host = self._b._comfy_http_get(f"{base}/colab_pet/host_metrics", base_url=base, timeout=2.0)
            st_code = int(getattr(r_host, "status_code", 0) or 0)
            if st_code == 200:
                hm_json = r_host.json()
                if isinstance(hm_json, dict):
                    return hm_json
            elif st_code == 404:
                self._maybe_trigger_bridge_host_metrics_async(base, "comfy")
        return {}

    def probe_comfy_status(self, base: str) -> tuple[bool, dict[str, Any], list[dict[str, Any]]]:
        """探测 ComfyUI /system_stats、/queue 与 /history，返回 `(是否在线, 状态字典, 待发送事件列表)`。"""
        b = self._b
        try:
            r_stats = b._comfy_http_get(f"{base}/system_stats", base_url=base, timeout=4.0)
            r_stats.raise_for_status()
            stats = r_stats.json()
        except Exception:
            return False, {}, []

        sys_metrics = _parse_comfy_system_stats(stats if isinstance(stats, dict) else {})
        running_cnt, pending_cnt = self._fetch_comfy_queue_counts(base)

        with b._comfy_lock:
            prev_running = int(b._comfy_state.get("running_count") or 0)
            b._comfy_state["ws_connected"] = bool(
                b._ws_app is not None and b._ws_thread and b._ws_thread.is_alive()
            )
            b._comfy_state["tunnel_url"] = base
            for key in ("gpu_name", "gpu_full_name", "vram_used_gb", "vram_total_gb", "vram_pct"):
                b._comfy_state[key] = sys_metrics[key]
            if sys_metrics["ram_total_gb"] > 0:
                for key in ("ram_used_gb", "ram_total_gb", "ram_free_gb", "ram_pct"):
                    b._comfy_state[key] = sys_metrics[key]
            b._comfy_state["running_count"] = running_cnt
            b._comfy_state["pending_count"] = pending_cnt
            if running_cnt > 0 and prev_running == 0:
                b._comfy_state["has_error"] = False
                b._comfy_state["error_summary"] = ""
            if running_cnt == 0:
                b._comfy_state.update(
                    {"step_value": 0, "step_max": 0, "step_pct": 0.0, "current_node": ""}
                )

        if running_cnt > 0 or pending_cnt > 0:
            b.touch_activity()

        events = self._process_comfy_history(base)
        host_m = self._fetch_comfy_host_metrics(base)
        parsed_hm = _parse_host_metrics(host_m)
        if parsed_hm["ram_total_gb"] <= 0:
            for key in ("ram_used_gb", "ram_total_gb", "ram_free_gb", "ram_pct"):
                parsed_hm[key] = sys_metrics[key]

        with b._comfy_lock:
            _apply_host_metrics_locked(
                b._comfy_state, parsed_hm, write_drive=bool(host_m), drive_requires_total=False
            )

        status_dict = {
            "gpu_name": sys_metrics["gpu_name"],
            "gpu_full_name": sys_metrics["gpu_full_name"],
            "vram_used_gb": sys_metrics["vram_used_gb"],
            "vram_total_gb": sys_metrics["vram_total_gb"],
            "vram_pct": sys_metrics["vram_pct"],
            **parsed_hm,
            "running_count": running_cnt,
            "pending_count": pending_cnt,
        }
        return True, status_dict, events

    # ------------------------------------------------------------------
    # LLM 在线探针拆解
    # ------------------------------------------------------------------

    def _resolve_llm_meta_and_headers(
        self, base: str, ep: str
    ) -> tuple[dict[str, Any], dict[str, str]]:
        """解析 LLM 端点元数据与同源鉴权请求头。"""
        use_ep = ep
        if not use_ep:
            target = self._store.pick_target_session("comfy")
            if target and target.get("endpoint"):
                use_ep = str(target["endpoint"])

        meta, api_key = self._secrets.get_ep_meta_and_key(use_ep)
        if not meta:
            with self._store.lock:
                for m_ep, m_val in self._secrets.service_meta_by_ep.items():
                    cand_urls = (
                        str(m_val.get("tunnel_url") or "").rstrip("/"),
                        str(m_val.get("llm_tunnel_url") or "").rstrip("/"),
                    )
                    if base in cand_urls:
                        meta = dict(m_val)
                        api_key = str(self._secrets.runtime_secrets.get(m_ep) or "")
                        break

        headers: dict[str, str] = {}
        if api_key and url_origin_key(base):
            headers["Authorization"] = f"Bearer {api_key}"
        return meta, headers

    def _fetch_llm_slots_metrics(
        self, base: str, headers: dict[str, str]
    ) -> tuple[int, float | None]:
        """探测 llama-server `/slots` 接口，返回 `(活跃槽位数, 当前 TPS)`。"""
        running_cnt = 0
        active_tps: float | None = None
        fallback_tps: float | None = None
        try:
            r_slots = self._b._http.get(
                f"{base}/slots", headers=headers, timeout=4.0, allow_redirects=False
            )
            if r_slots.status_code != 200:
                return 0, None
            sdata = r_slots.json()
            slots_list = (
                sdata if isinstance(sdata, list) else (sdata.get("slots") if isinstance(sdata, dict) else [])
            )
            if not isinstance(slots_list, list):
                return 0, None
            for s_item in slots_list:
                if not isinstance(s_item, dict):
                    continue
                is_proc = bool(s_item.get("is_processing")) or (int(s_item.get("state") or 0) == 1)
                cand_tps = extract_slot_tps(s_item)
                if is_proc:
                    running_cnt += 1
                    if cand_tps is not None:
                        active_tps = cand_tps
                elif cand_tps is not None and fallback_tps is None:
                    fallback_tps = cand_tps
        except Exception:
            return 0, None
        tps = active_tps if active_tps is not None else (fallback_tps if running_cnt > 0 else None)
        return running_cnt, tps

    def _resolve_llm_pref_defaults(self) -> tuple[str, int]:
        """读取用户偏好中的 LLM 默认量化档位与上下文长度。"""
        prefs = self._store.get_prefs()
        pref_profs = normalize_profiles(
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
        return pref_quant, pref_ctx

    def _fetch_llm_gateway_metrics(
        self, base: str, headers: dict[str, str]
    ) -> dict[str, Any]:
        """探测 SSE 网关受保护端点 `/__pet_metrics`，404 时触发桥接兜底遥测。"""
        with suppress_logged("comfy.llm_metrics", throttled_key=f"llm_metrics:{base}"):
            r_hm = self._b._http.get(
                f"{base}/__pet_metrics", headers=headers, timeout=2.0, allow_redirects=False
            )
            if r_hm.status_code == 200:
                hm_obj = r_hm.json()
                if isinstance(hm_obj, dict):
                    return hm_obj
            elif r_hm.status_code == 404:
                self._maybe_trigger_bridge_host_metrics_async(base, "llm")
        return {}

    def probe_llm_status(self, base: str, ep: str = "") -> tuple[bool, dict[str, Any]]:
        """探测 llama-server /health 与 /slots 状态，返回 `(是否在线, 推理指标字典)`。"""
        b = self._b
        meta, headers = self._resolve_llm_meta_and_headers(base, ep)
        try:
            r_health = b._http.get(
                f"{base}/health", headers=headers, timeout=4.0, allow_redirects=False
            )
            if r_health.status_code != 200:
                raise RuntimeError(f"HTTP {r_health.status_code}")
            h_body = r_health.json()
            if not isinstance(h_body, dict) or str(h_body.get("status") or "").lower() != "ok":
                raise RuntimeError("unhealthy")
        except Exception:
            return False, {}

        running_cnt, tps = self._fetch_llm_slots_metrics(base, headers)
        if running_cnt > 0:
            b.touch_activity()

        pref_quant, pref_ctx = self._resolve_llm_pref_defaults()
        raw_llm_gpu = str(meta.get("gpu_full_name") or meta.get("gpu_name") or "")
        gpu_name = short_gpu_name(raw_llm_gpu)
        gpu_full_name = _clean_gpu_full_name(raw_llm_gpu)
        v_used = _round_gb(meta.get("vram_used_gb"))
        v_total = _round_gb(meta.get("vram_total_gb"))
        v_pct = round(v_used * 100.0 / v_total, 1) if v_total > 0 else 0.0

        llm_hm = self._fetch_llm_gateway_metrics(base, headers)
        if str(llm_hm.get("gpu_full_name") or "").strip():
            gpu_full_name = _clean_gpu_full_name(str(llm_hm["gpu_full_name"]))
            if not gpu_name:
                gpu_name = short_gpu_name(gpu_full_name)
        if float(llm_hm.get("vram_total_gb") or 0.0) > 0:
            v_used = _round_gb(llm_hm.get("vram_used_gb"))
            v_total = _round_gb(llm_hm.get("vram_total_gb"))
            v_pct = round(float(llm_hm.get("vram_pct") or (v_used * 100.0 / v_total)), 1)

        parsed_hm = _parse_host_metrics(llm_hm)
        return (
            True,
            {
                "running_count": running_cnt,
                "tps": tps,
                "gpu_name": gpu_name,
                "gpu_full_name": gpu_full_name,
                "vram_used_gb": v_used,
                "vram_total_gb": v_total,
                "vram_pct": v_pct,
                **parsed_hm,
                "model_name": b._infer_default_llm_model_name(meta),
                "quant": str(meta.get("quant") or pref_quant),
                "ctx_size": int(meta.get("ctx_size") or pref_ctx),
            },
        )

    # ------------------------------------------------------------------
    # 单轮统一状态轮询
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_llm_mode_state_locked(
        comfy_state: dict[str, Any], l_base: str, l_ok: bool, l_info: dict[str, Any]
    ) -> None:
        """在持 `_comfy_lock` 下刷新纯 LLM 模式状态字段。"""
        comfy_state["ws_connected"] = False
        if not l_ok:
            return
        rc_l = int(l_info.get("running_count") or 0)
        comfy_state.update(
            {
                "tunnel_url": l_base,
                "llm_tunnel_url": l_base,
                "running_count": rc_l,
                "pending_count": 0,
                "llm_running_count": rc_l,
                "step_value": 0,
                "step_max": 0,
                "step_pct": 0.0,
                "current_node": "",
                **_extract_telemetry_patch(l_info),
                "has_error": False,
                "error_summary": "",
                "model_name": l_info.get("model_name", "Qwen3.8-Flash-Next-abliterated"),
                "quant": l_info.get("quant", ""),
                "ctx_size": l_info.get("ctx_size", 0),
                "tps": l_info.get("tps"),
            }
        )

    @staticmethod
    def _apply_dual_mode_state_locked(
        comfy_state: dict[str, Any],
        c_base: str,
        l_base: str,
        c_ok: bool,
        l_ok: bool,
        l_info: dict[str, Any],
        any_ok: bool,
    ) -> None:
        """在持 `_comfy_lock` 下刷新双栖（ComfyUI + LLM）模式状态字段。"""
        if c_base:
            comfy_state["tunnel_url"] = c_base
        if l_base:
            comfy_state["llm_tunnel_url"] = l_base
        if not c_ok:
            comfy_state.update({"ws_connected": False, "running_count": 0, "pending_count": 0})
        if (not c_ok) and l_ok and float(l_info.get("vram_total_gb") or 0.0) > 0:
            comfy_state.update(_extract_telemetry_patch(l_info))
        if any_ok and l_ok:
            comfy_state.update(
                {
                    "llm_running_count": int(l_info.get("running_count") or 0),
                    "model_name": l_info.get("model_name", ""),
                    "quant": l_info.get("quant", ""),
                    "ctx_size": l_info.get("ctx_size", 0),
                    "tps": l_info.get("tps"),
                }
            )
        else:
            comfy_state.update({"llm_running_count": 0, "tps": None})

    def poll_service_once(
        self, mode: str, comfy_url: str = "", llm_url: str = "", ep: str = ""
    ) -> dict[str, Any]:
        """统一单轮状态轮询（兼容 comfy / llm / dual），更新 _comfy_state 与快照并推送事件。"""
        b = self._b
        with self._store.lock:
            start_epoch = b._account_epoch
        c_base = comfy_url.strip().rstrip("/") if comfy_url else ""
        l_base = llm_url.strip().rstrip("/") if llm_url else ""

        c_ok, events = b._probe_comfy_status(c_base)[::2] if (mode in ("comfy", "dual") and c_base) else (False, [])
        l_ok, l_info = b._probe_llm_status(l_base, ep=ep) if (mode in ("llm", "dual") and l_base) else (False, {})

        with self._store.lock:
            if b._account_epoch != start_epoch:
                return b.get_comfy_state()

        any_ok = c_ok if mode == "comfy" else (l_ok if mode == "llm" else (c_ok or l_ok))
        with b._comfy_lock:
            b._comfy_state["service_kind"] = mode
            b._comfy_state["comfy_online"] = c_ok
            b._comfy_state["llm_online"] = l_ok
            if mode == "comfy" and c_ok:
                b._comfy_state.update(
                    {
                        "llm_tunnel_url": None,
                        "llm_running_count": 0,
                        "model_name": "",
                        "quant": "",
                        "ctx_size": 0,
                        "tps": None,
                    }
                )
            elif mode == "llm":
                self._apply_llm_mode_state_locked(b._comfy_state, l_base, l_ok, l_info)
            elif mode == "dual":
                self._apply_dual_mode_state_locked(
                    b._comfy_state, c_base, l_base, c_ok, l_ok, l_info, any_ok
                )

            if any_ok:
                b._comfy_http_failures = 0
                b._comfy_state["online"] = True
            else:
                b._comfy_http_failures += 1
                if b._comfy_http_failures >= 2:
                    b._comfy_state["online"] = False
            b._comfy_state["pending_pull_count"] = len(b._pending_pull_items)
            cstate = dict(b._comfy_state)

        with self._store.lock:
            b.last_snapshot["comfy"] = cstate
            cleared_deploy = b._clear_stale_deploy_error_locked() if any_ok else None
        if cleared_deploy is not None:
            self._bus.emit({"type": "deploy_progress", "deploy": cleared_deploy})
        self._bus.emit({"type": "comfy_tick", "comfy": cstate})
        for ev in events:
            self._bus.emit(ev)
        return cstate

    # ------------------------------------------------------------------
    # 手动同步与拉取最新成图
    # ------------------------------------------------------------------

    def sync_outputs(self, session: str = "comfy") -> dict[str, Any]:
        """将云端 ComfyUI 成图拉回本地目录；若当前活跃为 LLM 对话服务则平滑分流为返回 OpenAI 接口配置。"""
        b = self._b
        b.touch_activity()
        if b.is_deploying():
            return {"ok": False, "message": "正在执行 7 步开机部署，请等待就绪后再操作"}
        if b.is_llm_mode_active():
            return b.get_llm_api_bundle(session=session)
        if not b._action_lock.acquire(blocking=False):
            return {"ok": False, "message": "另一项操作正在执行，请稍等片刻"}
        try:
            return self._sync_outputs_locked(session)
        except Exception as exc:
            b._diag.error("backend", f"同步异常: {redact_secrets(str(exc))}")
            return {"ok": False, "message": f"同步异常: {redact_secrets(str(exc))}"}
        finally:
            b._action_lock.release()

    def _sync_outputs_locked(self, session: str) -> dict[str, Any]:
        """在持有 `_action_lock` 下执行云盘拉回或分流到 LLM 接口配置。"""
        b = self._b
        target = self._store.pick_target_session(session)
        if target is None:
            snap = b.fetch_live_status()
            target = self._store.pick_target_session(session)
            if target is None and snap.get("assignments_count", 0) == 0 and not snap.get("sessions"):
                b.download_root.mkdir(parents=True, exist_ok=True)
                self._ctx.reveal_path(b.download_root)
                return {"ok": True, "message": "当前未开机，已打开本地成图目录"}

        ep_chk = str(target.get("endpoint") or "") if target else ""
        sname = target["name"] if target and target.get("name") not in (None, "?") else session
        meta_chk, _ = self._secrets.get_ep_meta_and_key(ep_chk)
        if not meta_chk and ep_chk:
            b._fast_read_tunnel_url(ep_chk, sname)
            meta_chk, _ = self._secrets.get_ep_meta_and_key(ep_chk)
        if meta_chk and str(meta_chk.get("kind") or "").lower() == "llm":
            return b.get_llm_api_bundle(session=sname)

        if not self._ctx.ctl_available():
            return {"ok": False, "message": f"未找到控制脚本: {self._ctx.ctl_script.name}"}
        if ep_chk:
            self._store.record_managed_binding(sname, ep_chk, persist=True)
        sync_cmd = [*self._ctx.ctl_cmd_prefix(), "sync", "--pull"]
        sync_acct = self._store.resolve_single_node_account_id(ep_chk)
        if sync_acct:
            sync_cmd.extend(["--account", sync_acct])
        sync_cmd.extend(["-s", sname])
        res = b._runner.run(
            sync_cmd,
            timeout=180,
            extra_env=self._ctx.build_ctl_env(),
            run_fn=self._ctx.run_hidden,
            task_name="sync_outputs",
        )
        tail = (res.combined_output.splitlines() or ["同步结束"])[-1]
        if res.ok:
            if b.download_root.exists():
                self._ctx.reveal_path(b.download_root)
            return {"ok": True, "message": f"拉回完成: {tail}"}
        return {"ok": False, "message": f"同步失败: {res.error_summary}"}

    def pull_or_open_latest_outputs(self) -> dict[str, Any]:
        """点击出图提醒时互斥按需拉回待下载产出并同步持久化队列，或高亮最新成图。"""
        b = self._b
        b.touch_activity()
        with b._pull_lock:
            with b._comfy_lock:
                pending = list(b._pending_pull_items)
                recent = list(b._last_downloaded_files)
            if pending:
                return self._pull_pending_outputs_locked(pending)

        existing = [p for p in recent if p.exists()]
        if existing:
            latest = existing[-1].resolve()
            self._ctx.reveal_path(latest)
            return {"ok": True, "path": str(latest), "message": f"已打开位置: {latest.name}"}
        if b.download_root.exists():
            self._ctx.reveal_path(b.download_root)
            return {"ok": True, "message": "已打开本地成图目录"}
        return b.sync_outputs()

    def _pull_pending_outputs_locked(
        self, pending: list[tuple[str, dict[str, str]]]
    ) -> dict[str, Any]:
        """在持有 `_pull_lock` 下拉取待下载列表并高亮定位已落盘文件。"""
        b = self._b
        saved, failed, cstate = self._execute_pull_batch(pending)
        self._bus.emit({"type": "comfy_tick", "comfy": cstate})
        if not saved:
            return {
                "ok": False,
                "failed_count": len(failed),
                "message": f"拉取失败：仍有 {len(failed)} 份作品未存入本机，请检查网络隧道后重试",
            }
        existing_saved = [p for p in saved if p.exists()]
        latest = (existing_saved[-1] if existing_saved else saved[-1]).resolve()
        if existing_saved:
            self._ctx.reveal_path(latest)
        elif b.download_root.exists():
            self._ctx.reveal_path(b.download_root)
        if not failed:
            return {"ok": True, "path": str(latest), "message": f"已拉回并定位: {latest.name}"}
        return {
            "ok": False,
            "partial": True,
            "saved_count": len(saved),
            "failed_count": len(failed),
            "path": str(latest),
            "message": f"已拉回 {len(saved)} 份，仍有 {len(failed)} 份拉取失败，请重试",
        }

    # ------------------------------------------------------------------
    # WebSocket 实时进度监听
    # ------------------------------------------------------------------

    def _emit_ws_progress_throttled(self, val: int, mx: int) -> None:
        """统一更新采样步数进度，并按 250ms 节流（满 100% 立即发送）广播事件。"""
        b = self._b
        pct = round(val * 100.0 / mx, 1) if mx > 0 else 0.0
        with b._comfy_lock:
            b._comfy_state["step_value"] = val
            b._comfy_state["step_max"] = mx
            b._comfy_state["step_pct"] = pct
            b._comfy_state["running_count"] = max(1, int(b._comfy_state.get("running_count") or 0))
            cstate = dict(b._comfy_state)
        now = self._ctx.monotonic()
        if now - b._ws_last_emit_ts >= 0.25 or pct >= 100.0:
            b._ws_last_emit_ts = now
            self._bus.emit({"type": "comfy_tick", "comfy": cstate})

    def _handle_ws_progress_state(self, mdata: dict[str, Any]) -> None:
        """解析 ComfyUI 新版 `progress_state` 多节点进度载荷并委托节流广播。"""
        nodes = mdata.get("nodes") if isinstance(mdata.get("nodes"), dict) else {}
        picked = next(
            (
                node
                for node in nodes.values()
                if isinstance(node, dict) and node.get("state") == "running" and (node.get("max") or 0) > 0
            ),
            None,
        )
        if picked:
            self._emit_ws_progress_throttled(int(picked.get("value") or 0), int(picked.get("max") or 0))

    def _handle_ws_execution_error(self, mdata: dict[str, Any]) -> None:
        """处理 WS `execution_error` 报错帧，更新状态并双推 `comfy_tick` 与 `comfy_error`。"""
        b = self._b
        summary = format_comfy_error(
            str(mdata.get("node_type") or mdata.get("node_id") or "节点"),
            str(mdata.get("exception_type") or ""),
            str(mdata.get("exception_message") or ""),
        )
        pid = str(mdata.get("prompt_id") or "")
        if pid:
            b._reported_error_prompts.add(pid)
        with b._comfy_lock:
            b._comfy_state.update(
                {
                    "has_error": True,
                    "error_summary": summary,
                    "step_value": 0,
                    "step_max": 0,
                    "step_pct": 0.0,
                }
            )
            cstate = dict(b._comfy_state)
        self._bus.emit({"type": "comfy_tick", "comfy": cstate})
        self._bus.emit({"type": "comfy_error", "summary": summary, "prompt_id": pid})

    def on_ws_message(self, _ws: Any, message: Any) -> None:
        """分发 ComfyUI WebSocket 文本事件帧（执行开始、进度、节点切换与异常报警）。"""
        b = self._b
        if isinstance(message, (bytes, bytearray)) or not isinstance(message, str):
            return
        try:
            payload = json.loads(message)
        except Exception:
            return
        mtype = payload.get("type")
        mdata = payload.get("data") if isinstance(payload.get("data"), dict) else {}

        if mtype in ("execution_start", "progress", "progress_state", "executing"):
            b.touch_activity()

        if mtype == "execution_start":
            with b._comfy_lock:
                b._comfy_state.update(
                    {
                        "has_error": False,
                        "error_summary": "",
                        "step_value": 0,
                        "step_max": 0,
                        "step_pct": 0.0,
                    }
                )
                cstate = dict(b._comfy_state)
            self._bus.emit({"type": "comfy_tick", "comfy": cstate})
        elif mtype == "progress":
            self._emit_ws_progress_throttled(int(mdata.get("value") or 0), int(mdata.get("max") or 0))
        elif mtype == "progress_state":
            self._handle_ws_progress_state(mdata)
        elif mtype == "executing":
            node_id = mdata.get("node")
            if node_id is None and mdata.get("prompt_id"):
                b._wake_monitor.set()
            elif node_id is not None:
                with b._comfy_lock:
                    b._comfy_state["current_node"] = str(node_id)
        elif mtype == "execution_error":
            self._handle_ws_execution_error(mdata)

    def stop_ws(self) -> None:
        """停止并关闭当前 ComfyUI WebSocket 长连接。"""
        b = self._b
        b._ws_stop.set()
        app = b._ws_app
        b._ws_app = None
        b._ws_target_url = None
        if app is not None:
            # 停止 WS 客户端时连接可能已断开，忽略关闭异常
            with contextlib.suppress(OSError, RuntimeError, AttributeError):
                app.close()

    def _resolve_ws_auth_headers(self, base: str) -> list[str] | None:
        """解析 WS 同源鉴权头；必要时在 401 时触发一次远端 ComfyUI 密钥补水。"""
        b = self._b
        comfy_key = b._get_comfy_key_for_url(base)
        if not comfy_key and self._store.has_running_session():
            with suppress_logged("comfy.ws_auth", throttled_key=f"ws_auth:{base}"):
                chk = b._http.get(f"{base}/system_stats", timeout=3.0, allow_redirects=False)
                if chk.status_code == 401:
                    comfy_key = b._hydrate_comfy_key_if_needed(base)
        return [f"Authorization: Bearer {comfy_key}"] if comfy_key else None

    def ensure_ws_target(self, tunnel_url: str) -> None:
        """确保针对指定隧道地址维持单条后台 WebSocket 进度监听线程。"""
        b = self._b
        ws_mod = self._ctx.websocket_mod
        if ws_mod is None:
            return
        base = tunnel_url.strip().rstrip("/")
        if b._ws_target_url == base and b._ws_thread and b._ws_thread.is_alive():
            return
        b._stop_ws()
        b._ws_stop.clear()
        b._ws_target_url = base
        ws_base = re.sub(r"^http", "ws", base, count=1)
        ws_url = f"{ws_base}/ws?clientId=colab_pet_{uuid.uuid4().hex[:8]}"

        def _ws_runner() -> None:
            while not b._ws_stop.is_set() and not b._monitor_stop.is_set():
                try:
                    with suppress_logged("comfy.ws_loop", throttled_key=f"ws_loop:{base}"):
                        ws_headers = self._resolve_ws_auth_headers(base)
                        app = ws_mod.WebSocketApp(ws_url, header=ws_headers, on_message=b._on_ws_message)
                        b._ws_app = app
                        app.run_forever(ping_interval=20, ping_timeout=10)
                finally:
                    b._ws_app = None
                if b._ws_stop.wait(timeout=3.0):
                    break

        b._ws_thread = self._ctx.threading_mod.Thread(target=_ws_runner, daemon=True)
        b._ws_thread.start()

    # ------------------------------------------------------------------
    # 云盘传输、工作流/素材/模型上传与拖拽分流
    # ------------------------------------------------------------------

    def _emit_upload_progress(self, name: str, pct: int, stage: str) -> None:
        val = max(0, min(100, int(pct)))
        self._bus.emit(
            {
                "type": "upload_progress",
                "name": name,
                "pct": val,
                "summary": f"📤 {stage} {name} ({val}%)",
            }
        )

    def rclone_upload_file(
        self,
        src: Path,
        remote_dst: str,
        display_name: str,
        pct_start: int = 0,
        pct_span: int = 100,
    ) -> None:
        """调用本地 rclone 将文件上传到指定云盘目标，并流式汇报进度百分比。"""
        rclone_exe = self._ctx.get_rclone_exe()
        rclone_conf = self._ctx.get_rclone_conf()
        if not rclone_exe.is_file():
            raise FileNotFoundError(f"未找到 rclone.exe: {rclone_exe}")
        if not rclone_conf.is_file():
            raise FileNotFoundError(f"未找到 rclone.conf: {rclone_conf}")

        subproc = self._ctx.subprocess_mod
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
        proc = subproc.Popen(
            cmd,
            stdout=subproc.PIPE,
            stderr=subproc.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=clean_subprocess_env(),
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
                    if not sline:
                        continue
                    tail_lines = (tail_lines + [sline])[-8:]
                    matched = re.search(r"(\d{1,3})%", sline)
                    if matched:
                        raw_pct = min(100, int(matched.group(1)))
                        mapped = pct_start + int(round(raw_pct * pct_span / 100.0))
                        if mapped != last_pct:
                            last_pct = mapped
                            self._emit_upload_progress(display_name, mapped, "传云盘")
        finally:
            proc.stdout.close()

        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(redact_secrets(tail_lines[-1] if tail_lines else f"rclone 退出码 {rc}"))

    def write_remote_file_via_contents(
        self, rel_path: str, raw_bytes: bytes, session: str = "comfy"
    ) -> None:
        """通过 Colab Contents API 将字节载荷写入活跃实例的相对路径。"""
        colab_py = self._ctx.get_colab_py()
        if not colab_py.exists():
            raise FileNotFoundError(f"未找到 colab-cli Python: {colab_py}")
        target = self._store.pick_target_session(session)
        ep = str(target.get("endpoint") or "") if target else ""
        sname = target["name"] if target and target.get("name") not in (None, "?") else session
        b64 = base64.b64encode(raw_bytes).decode("ascii")
        proc = self._ctx.subprocess_mod.run(
            [str(colab_py), str(self._ctx.cli_bridge_script), "contents-put", ep, sname, rel_path],
            input=b64,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=25,
            env=clean_subprocess_env(),
            creationflags=CREATE_NO_WINDOW,
        )
        if proc.returncode != 0:
            err = redact_secrets((proc.stderr or proc.stdout or "").strip()).splitlines()
            raise RuntimeError(err[-1] if err else "Contents 写入失败")

    def fetch_comfy_object_info(self, tunnel_url: str) -> dict[str, Any] | None:
        """从在线 ComfyUI 拉取节点定义（/object_info），用于将 API prompt 还原为标准画布工作流。"""
        base = tunnel_url.rstrip("/")
        with suppress_logged("comfy.object_info", throttled_key=f"object_info:{base}"):
            resp = self._b._comfy_http_get(f"{base}/object_info", base_url=base, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, dict):
                    return data
        return None

    def _prepare_workflow_payload(
        self, file_path: Path, is_png_extract: bool, tunnel_url: str | None
    ) -> tuple[str, bytes]:
        """解析本地工作流 JSON 或 PNG 内嵌工作流，按需转换为画布格式并返回 `(安全文件名, UTF-8 字节)`。"""
        b = self._b
        if is_png_extract:
            wf_text = extract_png_comfy_workflow(file_path)
            if not wf_text:
                raise ValueError(f"{file_path.name} 中未发现合法 ComfyUI 工作流")
            parsed = json.loads(wf_text)
            if not isinstance(parsed, dict):
                raise ValueError(f"{file_path.name} 中提取的工作流结构无效")
            safe_name = sanitize_filename(file_path.with_suffix(".json").name, "workflow.json")
        else:
            wf_text = file_path.read_text(encoding="utf-8")
            parsed = json.loads(wf_text)
            if not isinstance(parsed, dict):
                raise ValueError("JSON 顶层结构不是工作流对象")
            safe_name = sanitize_filename(file_path.name, "workflow.json")

        self._emit_upload_progress(safe_name, 20, "导工作流")
        if is_api_prompt_dict(parsed):
            obj_info = b._fetch_comfy_object_info(tunnel_url) if tunnel_url else None
            converted = api_prompt_to_ui_workflow(parsed, obj_info)
            wf_text = json.dumps(converted, ensure_ascii=False, indent=2)
        elif ensure_workflow_uuid(parsed):
            wf_text = json.dumps(parsed, ensure_ascii=False, indent=2)

        self._emit_upload_progress(safe_name, 30, "导工作流")
        return safe_name, wf_text.encode("utf-8")

    def upload_workflow_file_ex(
        self, file_path: Path, is_png_extract: bool, tunnel_url: str | None
    ) -> tuple[str, bool, str]:
        """解析、按需转换并上传工作流 JSON，返回 `(状态文案, 是否已写入在线实例, 安全文件名)`。"""
        b = self._b
        safe_name, wf_bytes = self._prepare_workflow_payload(file_path, is_png_extract, tunnel_url)

        if tunnel_url:
            base = tunnel_url.rstrip("/")
            target_ep = f"{base}/userdata/{urllib.parse.quote('workflows/' + safe_name, safe='')}?overwrite=true"
            resp = b._comfy_http_post(
                target_ep,
                base_url=base,
                data=wf_bytes,
                headers={"Content-Type": "application/json"},
                timeout=15,
            )
            if resp.status_code in (200, 201):
                b._backup_workflow_to_gdrive_async(safe_name, wf_bytes)
                self._emit_upload_progress(safe_name, 90, "导工作流")
                return f"✅ 已导入工作流: {safe_name} (点我打开画布)", True, safe_name
            if resp.status_code != 404:
                resp.raise_for_status()

        if self._store.has_running_session():
            b._write_remote_file_via_contents(
                f"content/ComfyUI/user/default/workflows/{safe_name}", wf_bytes
            )
            b._backup_workflow_to_gdrive_async(safe_name, wf_bytes)
            self._emit_upload_progress(safe_name, 100, "导工作流")
            return f"✅ 已写入实例工作流: {safe_name} (点我打开画布)", True, safe_name

        with tempfile.TemporaryDirectory(prefix="pet_wf_") as tmp_dir:
            tmp_wf = Path(tmp_dir) / safe_name
            tmp_wf.write_bytes(wf_bytes)
            b._rclone_upload_file(tmp_wf, f"{GDRIVE_WF_REMOTE}/{safe_name}", safe_name)
        return f"✅ 工作流已存云盘(下次开机自动载入): {safe_name}", False, safe_name

    def upload_media_file(self, file_path: Path, tunnel_url: str | None) -> tuple[str, str]:
        """流式上传参考图片或视频到 ComfyUI `/upload/image`（401 时自动补水重试一次）。"""
        b = self._b
        if not tunnel_url:
            raise RuntimeError("当前无在线 ComfyUI 隧道，请先开机后再传参考素材")
        base = tunnel_url.rstrip("/")
        url = f"{base}/upload/image"
        resp = None
        for attempt in (1, 2):
            with StreamingMultipartFile(
                file_path,
                field_name="image",
                fields={"type": "input", "overwrite": "true"},
                on_progress=lambda pct: self._emit_upload_progress(file_path.name, pct, "传参考图"),
            ) as stream:
                resp = b._comfy_http_post(
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
            if resp.status_code == 401 and attempt == 1 and self._store.has_running_session():
                had_key = bool(b._get_comfy_key_for_url(base))
                # 401 重试前尝试释放旧 HTTP 响应连接，忽略关闭异常
                with contextlib.suppress(OSError, RuntimeError, AttributeError):
                    resp.close()
                if b._hydrate_comfy_key_if_needed(base, force=(not had_key)):
                    continue
            resp.raise_for_status()
            break
        assert resp is not None
        uploaded_name = file_path.name
        with suppress_logged("comfy.upload_media", throttled_key="upload_media_json"):
            data = resp.json()
            if isinstance(data, dict) and data.get("name"):
                uploaded_name = str(data["name"])
        return f"✅ 已投递 {uploaded_name}，文件名已复制", uploaded_name

    def upload_model_file(
        self, file_path: Path, tunnel_url: str | None, session: str = "comfy"
    ) -> tuple[str, bool]:
        """上传 LoRA 模型权重至云盘，若实例在线则同步热拉到 ComfyUI 并刷新节点列表。"""
        b = self._b
        safe_name = sanitize_filename(file_path.name, "model.safetensors")
        remote_path = f"{GDRIVE_LORA_REMOTE}/{safe_name}"
        running = self._store.has_running_session()
        b._rclone_upload_file(
            file_path,
            remote_path,
            safe_name,
            pct_start=0,
            pct_span=70 if running else 100,
        )
        if not running:
            return f"✅ LoRA 已零耗电预传云盘: {safe_name}", False

        self._emit_upload_progress(safe_name, 78, "同步到实例")
        colab_py = self._ctx.get_colab_py()
        if colab_py.exists():
            target = self._store.pick_target_session(session)
            sname = target["name"] if target and target.get("name") not in (None, "?") else session
            proc = self._ctx.run_hidden(
                [
                    str(colab_py),
                    "-c",
                    COLAB_REMOTE_LORA_PULL_SCRIPT,
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
            with suppress_logged("comfy.refresh_lora", throttled_key=f"refresh_lora:{tunnel_url}"):
                base_u = tunnel_url.rstrip("/")
                b._comfy_http_get(f"{base_u}/object_info/LoraLoader", base_url=base_u, timeout=5)
        self._emit_upload_progress(safe_name, 100, "同步到实例")
        return f"✅ LoRA 已同步至云端并刷新: {safe_name}", True

    @staticmethod
    def read_text_file_safe(fpath: Path, max_chars: int = 200_000) -> tuple[str, bool]:
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

    def handle_llm_dropped_files(self, valid_files: list[Path]) -> dict[str, Any]:
        """在 LLM 对话模式下分流处理拖入或选取的文件：文本提取入剪贴板，图片引导网页直传。"""
        b = self._b
        text_files = [p for p in valid_files if p.suffix.lower() in LLM_TEXT_EXTS]
        image_files = [p for p in valid_files if p.suffix.lower() in LLM_IMAGE_EXTS]
        gguf_or_model = [
            p for p in valid_files if p.suffix.lower() == ".gguf" or p.suffix.lower() in MODEL_EXTS
        ]
        done_action = "open_llm_webui" if self._store.has_running_session() else None

        if text_files:
            chunks: list[str] = []
            any_trunc = False
            for fp in text_files:
                body, trunc = b._read_text_file_safe(fp)
                any_trunc = any_trunc or trunc
                chunks.append(body if len(text_files) == 1 else f"=== {fp.name} ===\n{body}")
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
            self._bus.emit(
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
            self._bus.emit({"type": "upload_done", "summary": summary, "action": done_action})
            return {
                "ok": True,
                "started": False,
                "service_kind": "llm",
                "count": len(image_files),
                "message": summary,
            }

        if gguf_or_model:
            msg = "💡 Qwen3.8 量化档位请在启动抽屉中切换 (Q4 / IQ4 / IQ3)，无需手动上传模型权重"
            self._bus.emit({"type": "upload_error", "summary": msg})
            return {"ok": False, "service_kind": "llm", "message": msg}

        msg = "⚠️ 对话模式支持提取文本 (.txt/.md/.json/.py/.yaml/.csv) 或将图片拖入对话网页"
        self._bus.emit({"type": "upload_error", "summary": msg})
        return {"ok": False, "service_kind": "llm", "message": msg}

    # ------------------------------------------------------------------
    # 拖拽文件分流上传拆解
    # ------------------------------------------------------------------

    @staticmethod
    def _collect_valid_dropped_files(paths: list[str]) -> list[Path]:
        """过滤并校验本地拖拽路径，仅保留存在的常规文件。"""
        valid: list[Path] = []
        for raw_p in paths or []:
            if not raw_p:
                continue
            path = Path(str(raw_p))
            if path.is_file():
                valid.append(path)
        return valid

    def _check_drop_preconditions(
        self,
        valid_files: list[Path],
        png_mode: str | None,
        base_url_override: str | None,
    ) -> dict[str, Any] | None:
        """检查是否需要直接分流给 LLM、弹出双栖识图选择或弹出 PNG 工作流选择。"""
        b = self._b
        if base_url_override is None and png_mode is None and b.is_llm_mode_active():
            return b._handle_llm_dropped_files(valid_files)

        single_wf_png = bool(
            len(valid_files) == 1
            and valid_files[0].suffix.lower() == ".png"
            and png_mode is None
            and extract_png_comfy_workflow(valid_files[0])
        )

        if base_url_override is None and png_mode is None and b.is_dual_mode_active():
            pure_llm_text_exts = LLM_TEXT_EXTS - WORKFLOW_EXTS
            if all(p.suffix.lower() in pure_llm_text_exts for p in valid_files):
                return b._handle_llm_dropped_files(valid_files)
            if (not single_wf_png) and all(p.suffix.lower() in LLM_IMAGE_EXTS for p in valid_files):
                self._bus.emit(
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
            self._bus.emit(
                {"type": "png_workflow_choice", "path": str(target_png), "name": target_png.name}
            )
            return {
                "ok": True,
                "choice_required": True,
                "message": "检测到 PNG 内嵌工作流，请选择投递方式",
            }
        return None

    def _upload_single_dropped_file(
        self,
        fpath: Path,
        idx: int,
        total: int,
        png_mode: str | None,
        turl: str | None,
    ) -> tuple[str, str | None, bool, str | None]:
        """按扩展名执行单个文件的投递，返回 `(状态文案, 剪贴板文本, 是否在线可开画布, 在线工作流名)`。"""
        b = self._b
        ext = fpath.suffix.lower()
        self._emit_upload_progress(fpath.name, 5, f"准备({idx}/{total})")
        if ext in WORKFLOW_EXTS or (ext == ".png" and png_mode in ("workflow", "canvas")):
            msg, online_ok, safe_wf_name = b._upload_workflow_file_ex(
                fpath, is_png_extract=(ext == ".png"), tunnel_url=turl
            )
            return msg, None, online_ok, (safe_wf_name if (online_ok and safe_wf_name) else None)
        if ext in MEDIA_EXTS:
            msg, clip = b._upload_media_file(fpath, tunnel_url=turl)
            return msg, clip, False, None
        if ext in MODEL_EXTS:
            msg, online_ok = b._upload_model_file(fpath, tunnel_url=turl)
            return msg, None, online_ok, None
        raise ValueError(f"暂不支持该文件格式: {fpath.name}")

    def _finalize_uploaded_workflows(
        self,
        uploaded_wf_names: list[str],
        turl: str | None,
        total_files: int,
        last_summary: str,
        can_open_comfy: bool,
    ) -> tuple[str, str | None]:
        """汇总本批上传的最终展示文案与点击动作，并在导入在线工作流时尝试直铺画布。"""
        b = self._b
        done_action: str | None = "open_comfy" if can_open_comfy else None
        if not uploaded_wf_names:
            summary = f"✅ 已完成 {total_files} 个文件投递" if total_files > 1 else last_summary
            return summary, done_action

        primary_wf = uploaded_wf_names[0]
        done_action = f"open_comfy_wf:{primary_wf}"
        if not turl:
            summary = f"✅ 已完成 {total_files} 个文件投递" if total_files > 1 else last_summary
            return summary, done_action

        self._emit_upload_progress(primary_wf, 95, "直铺画布")
        canvas_res = b._push_or_open_workflow_on_canvas(
            turl, primary_wf, extra_names=uploaded_wf_names[1:]
        )
        if canvas_res.get("pushed_live"):
            summary = (
                f"✨ 已直铺画布: {primary_wf} (点我新开)"
                if len(uploaded_wf_names) == 1
                else f"✨ 已直铺 {len(uploaded_wf_names)} 个工作流到当前画布"
            )
        else:
            summary = (
                f"✨ 已打开并直铺: {primary_wf}"
                if len(uploaded_wf_names) == 1
                else f"✨ 已打开并直铺 {len(uploaded_wf_names)} 个工作流"
            )
        return summary, done_action

    def _run_drop_upload_worker(
        self,
        valid_files: list[Path],
        png_mode: str | None,
        base_url_override: str | None,
    ) -> None:
        """在后台任务线程中串行上传文件批次，并统一发送成功或错误事件。"""
        b = self._b
        try:
            turl = (
                base_url_override.strip().rstrip("/")
                if base_url_override
                else b.resolve_active_tunnel_url(allow_fast_discover=True)
            )
            last_summary = ""
            last_clip: str | None = None
            can_open_comfy = False
            uploaded_wf_names: list[str] = []
            for idx, fpath in enumerate(valid_files, start=1):
                msg, clip, online_ok, wf_name = self._upload_single_dropped_file(
                    fpath, idx, len(valid_files), png_mode, turl
                )
                last_summary = msg
                if clip is not None:
                    last_clip = clip
                can_open_comfy = can_open_comfy or online_ok
                if wf_name:
                    uploaded_wf_names.append(wf_name)

            final_summary, done_action = self._finalize_uploaded_workflows(
                uploaded_wf_names, turl, len(valid_files), last_summary, can_open_comfy
            )
            self._bus.emit(
                {
                    "type": "upload_done",
                    "summary": final_summary,
                    "clipboard_text": last_clip,
                    "action": done_action,
                }
            )
        except Exception as exc:
            b._diag.error("upload", f"投递失败: {redact_secrets(str(exc))}")
            self._bus.emit(
                {
                    "type": "upload_error",
                    "summary": f"⚠️ 投递失败: {redact_secrets(str(exc))}",
                }
            )
        finally:
            b._upload_lock.release()

    def submit_dropped_files(
        self,
        paths: list[str],
        png_mode: str | None = None,
        base_url_override: str | None = None,
    ) -> dict[str, Any]:
        """智能识别拖入或选取的本地文件，分流上传为参考素材、工作流、LoRA 或 LLM 文本提取。"""
        b = self._b
        b.touch_activity()
        valid_files = self._collect_valid_dropped_files(paths)
        if not valid_files:
            return {"ok": False, "message": "未识别到有效本地文件"}

        early_res = self._check_drop_preconditions(valid_files, png_mode, base_url_override)
        if early_res is not None:
            return early_res

        if not b._upload_lock.acquire(blocking=False):
            return {"ok": False, "message": "正在上传其他文件，请稍候"}

        hdl = b._tasks.submit(
            "upload_files",
            lambda _tok: self._run_drop_upload_worker(valid_files, png_mode, base_url_override),
            exclusive=False,
        )
        if not hdl.accepted:
            b._upload_lock.release()
            return {"ok": False, "message": hdl.reason or "上传任务忙"}
        return {
            "ok": True,
            "started": True,
            "count": len(valid_files),
            "message": f"开始投递 {len(valid_files)} 个文件...",
        }
