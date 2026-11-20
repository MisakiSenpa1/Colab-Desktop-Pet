"""colab_core.catalog —— 显卡默认费率、硬件规格与工作流目录单一事实来源（SSOT）。

本模块严格保持纯 Python 标准库实现（零重依赖、零循环导入），通过只读 `lru_cache`
加载 `stack_manifest.json`（完整兼容源码布局与 PyInstaller `_MEIPASS` 冻结包布局），
向后端账本、算力池调度器、集群快照及前端桌宠/Studio 界面统一输出权威默认规格与费率。
"""

from __future__ import annotations

import functools
import json
import sys
from pathlib import Path
from typing import Any

__all__ = [
    "clear_catalog_cache",
    "export_frontend_catalog",
    "get_default_measured_gpu_rates",
    "get_fallback_gpu_rate",
    "get_fallback_gpu_rate_map",
    "get_gpu_order",
    "get_gpu_specs",
    "get_profile_catalog",
]

# 仅在极端缺文件兜底场景使用的内置默认目录（正常情况 100% 读取 stack_manifest.json）
_FALLBACK_GPU_ORDER: tuple[str, ...] = ("T4", "L4", "A100", "G4", "H100")
_FALLBACK_GPU_SPECS: dict[str, dict[str, Any]] = {
    "T4": {
        "vram": "16G",
        "vram_hm": "16G高",
        "ram": "13G内存",
        "ram_hm": "51G高存",
        "default_rate_std": 1.15,
        "default_rate_hm": 1.31,
        "hm_locked": False,
        "note": "省钱入门",
    },
    "L4": {
        "vram": "24G高",
        "vram_hm": "24G高",
        "ram": "53G高存",
        "ram_hm": "53G高存",
        "default_rate_std": 1.94,
        "default_rate_hm": 1.94,
        "hm_locked": True,
        "note": "Anima甜点",
    },
    "A100": {
        "vram": "40G",
        "vram_hm": "80G高",
        "ram": "83G内存",
        "ram_hm": "167G高存",
        "default_rate_std": 5.5,
        "default_rate_hm": 11.5,
        "hm_locked": False,
        "note": "极速生图",
    },
    "G4": {
        "vram": "96G",
        "vram_hm": "96G",
        "ram": "176G高存",
        "ram_hm": "176G高存",
        "default_rate_std": 10.05,
        "default_rate_hm": 10.05,
        "hm_locked": True,
        "note": "大显存甜点",
    },
    "H100": {
        "vram": "80G",
        "vram_hm": "80G高",
        "ram": "167G内存",
        "ram_hm": "230G高存",
        "default_rate_std": 12.6,
        "default_rate_hm": 14.2,
        "hm_locked": False,
        "note": "旗舰算力",
    },
}
_DEFAULT_MEASURED_GPU_KEYS: tuple[str, ...] = ("T4", "L4", "G4")


def _resolve_manifest_path(manifest_path: Path | str | None = None) -> Path:
    """解析 `stack_manifest.json` 绝对路径（兼容开发目录、便携包与 PyInstaller `_MEIPASS`）。"""
    if manifest_path is not None:
        return Path(manifest_path).resolve()
    if getattr(sys, "frozen", False) and getattr(sys, "_MEIPASS", None):
        meipass_cand = Path(sys._MEIPASS).resolve() / "stack_manifest.json"
        if meipass_cand.is_file():
            return meipass_cand
    return Path(__file__).resolve().parents[1] / "stack_manifest.json"


def _normalize_gpu_spec(gpu_name: str, raw: Any) -> dict[str, Any]:
    """将单张显卡规格字典规范化为强类型字段集。"""
    fallback = _FALLBACK_GPU_SPECS.get(gpu_name, {})
    src = raw if isinstance(raw, dict) else {}
    hm_locked = bool(src.get("hm_locked", fallback.get("hm_locked", False)))

    try:
        rate_std = float(src.get("default_rate_std", fallback.get("default_rate_std", 0.0)))
    except (TypeError, ValueError):
        rate_std = float(fallback.get("default_rate_std", 0.0))

    try:
        raw_hm = src.get("default_rate_hm", rate_std if hm_locked else fallback.get("default_rate_hm", rate_std))
        rate_hm = rate_std if hm_locked else float(raw_hm)
    except (TypeError, ValueError):
        rate_hm = rate_std if hm_locked else float(fallback.get("default_rate_hm", rate_std))

    return {
        "vram": str(src.get("vram", fallback.get("vram", ""))),
        "vram_hm": str(src.get("vram_hm", fallback.get("vram_hm", src.get("vram", "")))),
        "ram": str(src.get("ram", fallback.get("ram", ""))),
        "ram_hm": str(src.get("ram_hm", fallback.get("ram_hm", src.get("ram", "")))),
        "default_rate_std": round(rate_std, 4),
        "default_rate_hm": round(rate_hm, 4),
        "hm_locked": hm_locked,
        "note": str(src.get("note", fallback.get("note", ""))),
    }


@functools.lru_cache(maxsize=8)
def _load_manifest_cached(resolved_path_str: str) -> dict[str, Any]:
    """按路径缓存只读清单解析结果，文件缺失或损坏时安全回退内置只读目录。"""
    target = Path(resolved_path_str)
    if target.is_file():
        try:
            data = json.loads(target.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except (OSError, ValueError, TypeError):
            pass
    return {
        "schema_version": 2,
        "default_profile": "anima",
        "gpu_order": list(_FALLBACK_GPU_ORDER),
        "gpu_specs": {k: dict(v) for k, v in _FALLBACK_GPU_SPECS.items()},
        "profiles": {},
    }


def clear_catalog_cache() -> None:
    """清空目录只读缓存（供测试更换临时清单路径后刷新使用）。"""
    _load_manifest_cached.cache_clear()


def _get_manifest_dict(manifest_path: Path | str | None = None) -> dict[str, Any]:
    """返回指定或默认路径对应的清单只读字典。"""
    return _load_manifest_cached(str(_resolve_manifest_path(manifest_path)))


def get_gpu_order(*, manifest_path: Path | str | None = None) -> tuple[str, ...]:
    """返回显卡展示与调度顺序元组（如 `('T4', 'L4', 'A100', 'G4', 'H100')`）。"""
    manifest = _get_manifest_dict(manifest_path)
    raw_order = manifest.get("gpu_order")
    ordered: list[str] = []
    if isinstance(raw_order, list):
        for item in raw_order:
            g = str(item or "").strip().upper()
            if g and g not in ordered:
                ordered.append(g)
    raw_specs = manifest.get("gpu_specs")
    if isinstance(raw_specs, dict):
        for k in raw_specs.keys():
            g = str(k or "").strip().upper()
            if g and g not in ordered:
                ordered.append(g)
    return tuple(ordered) if ordered else _FALLBACK_GPU_ORDER


def get_gpu_specs(*, manifest_path: Path | str | None = None) -> dict[str, dict[str, Any]]:
    """返回所有已登记显卡的规范化硬件规格与默认 CU 费率字典副本。"""
    manifest = _get_manifest_dict(manifest_path)
    raw_specs = manifest.get("gpu_specs")
    specs_map = raw_specs if isinstance(raw_specs, dict) else {}
    order = get_gpu_order(manifest_path=manifest_path)
    out: dict[str, dict[str, Any]] = {}
    for g in order:
        raw_entry = specs_map.get(g)
        if raw_entry is None:
            raw_entry = specs_map.get(g.lower())
        out[g] = _normalize_gpu_spec(g, raw_entry)
    return out


def get_fallback_gpu_rate(
    gpu: str,
    high_mem: bool = False,
    *,
    manifest_path: Path | str | None = None,
) -> float:
    """查询指定显卡在标准/高存档位下的默认兜底小时费率（`hm_locked` 卡自动归并至标准档）。"""
    g = str(gpu or "").strip().upper()
    if not g:
        return 0.0
    specs = get_gpu_specs(manifest_path=manifest_path)
    spec = specs.get(g)
    if not spec:
        return 0.0
    if spec["hm_locked"] or not high_mem:
        return float(spec["default_rate_std"])
    return float(spec["default_rate_hm"])


def get_fallback_gpu_rate_map(*, manifest_path: Path | str | None = None) -> dict[str, float]:
    """派生形如 `{'T4:STD': 1.15, 'T4:HM': 1.31, ...}` 的全量显卡兜底费率映射表。"""
    specs = get_gpu_specs(manifest_path=manifest_path)
    rate_map: dict[str, float] = {}
    for g, spec in specs.items():
        std_v = float(spec["default_rate_std"])
        hm_v = std_v if spec["hm_locked"] else float(spec["default_rate_hm"])
        rate_map[f"{g}:STD"] = std_v
        rate_map[f"{g}:HM"] = hm_v
    return rate_map


def get_default_measured_gpu_rates(*, manifest_path: Path | str | None = None) -> dict[str, float]:
    """从清单派生历史兼容默认实测基准费率字典（仅含 T4:STD / L4:STD / G4:STD）。"""
    specs = get_gpu_specs(manifest_path=manifest_path)
    defaults: dict[str, float] = {}
    for g in _DEFAULT_MEASURED_GPU_KEYS:
        if g in specs:
            defaults[f"{g}:STD"] = float(specs[g]["default_rate_std"])
    return defaults


def get_profile_catalog(*, manifest_path: Path | str | None = None) -> dict[str, dict[str, Any]]:
    """返回清单中各工作流 Profile 的只读展示与资源门槛摘要字典。"""
    manifest = _get_manifest_dict(manifest_path)
    raw_profiles = manifest.get("profiles")
    if not isinstance(raw_profiles, dict):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for key, val in raw_profiles.items():
        if not isinstance(val, dict):
            continue
        engine = val.get("engine") if isinstance(val.get("engine"), dict) else {}
        out[str(key)] = {
            "label": str(val.get("label") or key),
            "category": str(val.get("category") or ""),
            "verified_default": bool(val.get("verified_default", False)),
            "recommended_gpu": str(val.get("recommended_gpu") or "L4").upper(),
            "min_vram_gib": float(val.get("min_vram_gib") or 0.0),
            "min_cloud_free_gib": float(val.get("min_cloud_free_gib") or 0.0),
            "min_local_free_gib": float(val.get("min_local_free_gib") or 0.0),
            "engine_kind": str(engine.get("kind") or ""),
        }
    return out


def export_frontend_catalog(*, manifest_path: Path | str | None = None) -> dict[str, Any]:
    """导出面向前端快照（桌宠气泡与 Studio 窗口）的只读规格与工作流目录载荷。"""
    manifest = _get_manifest_dict(manifest_path)
    return {
        "schema_version": int(manifest.get("schema_version") or 2),
        "default_profile": str(manifest.get("default_profile") or "anima"),
        "gpu_order": list(get_gpu_order(manifest_path=manifest_path)),
        "gpu_specs": get_gpu_specs(manifest_path=manifest_path),
        "profiles": get_profile_catalog(manifest_path=manifest_path),
    }
