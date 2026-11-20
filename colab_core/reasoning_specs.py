"""colab_core.reasoning_specs —— AI 模型思考强度枚举规范化与模型家族能力推断模块。"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

SUPPORTED_AGENT_FORMATS = ("openai", "openai_responses", "anthropic", "gemini")
SUPPORTED_CONCRETE_REASONING_LEVELS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
SUPPORTED_REASONING_EFFORTS = ("auto",) + SUPPORTED_CONCRETE_REASONING_LEVELS

_REASONING_LEVEL_ALIASES: dict[str, str] = {
    "off": "none",
    "disabled": "none",
    "disable": "none",
    "0": "none",
    "min": "minimal",
    "med": "medium",
    "mid": "medium",
    "extra": "xhigh",
    "extra_high": "xhigh",
    "extra-high": "xhigh",
    "extra high": "xhigh",
    "x-high": "xhigh",
    "maximum": "max",
}

_FALLBACK_LEVELS: tuple[str, ...] = ("none", "low", "medium", "high", "xhigh")

# 数据驱动的模型家族思考能力规则表（按严格优先级顺序匹配，命中即返回）
_MODEL_REASONING_RULES: tuple[
    tuple[Callable[[str, str], bool], tuple[str, ...], str],
    ...,
] = (
    (
        lambda tail, _fmt: "hydrogen" in tail,
        ("low", "medium", "high", "xhigh"),
        "high",
    ),
    (
        lambda tail, _fmt: any(k in tail for k in ("carbon", "argon", "gemini-4")),
        ("low", "medium", "high"),
        "high",
    ),
    (
        lambda tail, _fmt: "gemini-3" in tail and "flash" in tail,
        ("minimal", "low", "medium", "high"),
        "medium",
    ),
    (
        lambda tail, _fmt: "gemini-3" in tail,
        ("low", "medium", "high"),
        "high",
    ),
    (
        lambda tail, _fmt: "gemini-2.5" in tail and "pro" in tail,
        ("low", "medium", "high"),
        "medium",
    ),
    (
        lambda tail, _fmt: "gemini-2.5" in tail,
        ("none", "low", "medium", "high"),
        "medium",
    ),
    (
        lambda tail, fmt: ("claude" in tail or fmt == "anthropic") and ("4-6" in tail or "4.6" in tail),
        ("none", "low", "medium", "high", "max"),
        "",
    ),
    (
        lambda tail, fmt: "claude" in tail or fmt == "anthropic",
        ("none", "low", "medium", "high"),
        "",
    ),
    (
        lambda tail, _fmt: any(k in tail for k in ("gpt-5.2", "gpt-5.3", "gpt-5.4", "codex")),
        ("none", "low", "medium", "high", "xhigh"),
        "medium",
    ),
    (
        lambda tail, _fmt: "gpt-5" in tail,
        ("minimal", "low", "medium", "high"),
        "medium",
    ),
    (
        lambda tail, _fmt: any(k in tail for k in ("o1", "o3", "o4")),
        ("low", "medium", "high"),
        "medium",
    ),
    (
        lambda tail, _fmt: any(k in tail for k in ("deepseek-r1", "reasoner", "qwq")),
        ("low", "medium", "high"),
        "medium",
    ),
    (
        lambda tail, _fmt: any(k in tail for k in ("deepseek", "qwen3", "glm-4.5", "glm-5", "kimi-k2")),
        ("none", "low", "medium", "high", "xhigh"),
        "",
    ),
)


def normalize_reasoning_effort_value(raw: Any, default: str = "auto") -> str:
    """规范化单条思考强度枚举，兼容 extra/off 等常见别名。"""
    val = str(raw or "").strip().lower()
    if not val:
        return default
    val = _REASONING_LEVEL_ALIASES.get(val, val)
    return val if val in SUPPORTED_REASONING_EFFORTS else default


def normalize_reasoning_level_list(raw_levels: Any) -> list[str]:
    """规范化模型支持的具体思考档位清单（不含 auto），按从低到高排序并去重。"""
    if not isinstance(raw_levels, (list, tuple, set)):
        return []
    seen: set[str] = set()
    for item in raw_levels:
        if isinstance(item, dict):
            raw_val = item.get("effort") or item.get("level") or item.get("id") or item.get("name") or ""
        else:
            raw_val = item
        norm = normalize_reasoning_effort_value(raw_val, default="")
        if norm and norm != "auto":
            seen.add(norm)
    return [lvl for lvl in SUPPORTED_CONCRETE_REASONING_LEVELS if lvl in seen]


def _make_reasoning_spec(levels: tuple[str, ...], default_level: str, source: str) -> dict[str, Any]:
    """构造标准思考能力规格字典（确保 levels 为独立列表）。"""
    return {
        "reasoning": True,
        "levels": list(levels),
        "default_level": default_level,
        "source": source,
    }


def infer_model_reasoning_spec(model_id: str, api_format: str = "openai") -> dict[str, Any]:
    """按模型家族推断默认支持的思考档位与默认强度（网关未下发元数据时的精确兜底）。"""
    mid = str(model_id or "").strip().lower()
    tail = mid.rsplit("/", 1)[-1]
    fmt = str(api_format or "openai").strip().lower()
    if not tail:
        return _make_reasoning_spec(_FALLBACK_LEVELS, "", "fallback")
    for matcher, levels, default_level in _MODEL_REASONING_RULES:
        if matcher(tail, fmt):
            return _make_reasoning_spec(levels, default_level, "inferred")
    return _make_reasoning_spec(_FALLBACK_LEVELS, "", "fallback")


__all__ = [
    "SUPPORTED_AGENT_FORMATS",
    "SUPPORTED_CONCRETE_REASONING_LEVELS",
    "SUPPORTED_REASONING_EFFORTS",
    "_MODEL_REASONING_RULES",
    "_REASONING_LEVEL_ALIASES",
    "infer_model_reasoning_spec",
    "normalize_reasoning_effort_value",
    "normalize_reasoning_level_list",
]
