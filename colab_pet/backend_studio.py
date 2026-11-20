"""Colab Studio 工作台后端聚合子服务模块。

负责提供媒体画廊扫描、ComfyUI PNG 元数据解析、工作流推送直开画布、
文件管理器定位、Google Drive 云盘资产清单查询与多端实时日志拉取能力。
"""

from __future__ import annotations

import json
import os
import re
import struct
import threading
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from colab_core.accounts import has_gdrive_save_section
from colab_core.file_sync import (
    GDRIVE_DRAMA3_REFS_REMOTE,
    GDRIVE_LORA_REMOTE,
    GDRIVE_WF_REMOTE,
    _read_png_text_chunks,
    extract_png_comfy_workflow,
)
from colab_core.runtime import extract_framed_json
from colab_security import redact_secrets

if TYPE_CHECKING:
    from colab_backend import ColabPetBackend

_IMAGE_EXTS = frozenset({".png", ".webp", ".jpg", ".jpeg"})
_VIDEO_EXTS = frozenset({".mp4", ".webm"})
_AUDIO_EXTS = frozenset({".wav", ".mp3", ".flac"})

_SAMPLER_CLASS_HINTS = (
    "ksampler",
    "samplercustom",
    "sampler",
)
_PROMPT_TEXT_CLASS_HINTS = (
    "cliptextencode",
    "textencode",
    "prompt",
    "string",
)
_MODEL_CLASS_KEYS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("checkpoint", ("ckpt_name",)),
    ("unet", ("unet_name", "model_name", "diffusion_model")),
    ("diffusionmodel", ("model_name", "unet_name")),
    ("modelloader", ("model_name", "ckpt_name")),
)
_CLOUD_ASSET_REMOTES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("loras", (GDRIVE_LORA_REMOTE, "gdrive_save:Colab_Comfy_LoRAs")),
    ("workflows", (GDRIVE_WF_REMOTE, "gdrive_save:Colab_Comfy_Workflows")),
    ("refs", (GDRIVE_DRAMA3_REFS_REMOTE, "gdrive_save:Colab_Drama3_Refs")),
)
_WIN_DRIVE_RE = re.compile(r"^[A-Za-z]:")
_NAT_SPLIT_RE = re.compile(r"(\d+)")
_LORA_INDEXED_KEY_RE = re.compile(r"^lora_name_\d+$")
_DIM_HEADER_CACHE: dict[str, tuple[int, int, int | None, int | None]] = {}
_DIM_CACHE_LOCK = threading.Lock()


def _empty_artwork_meta(
    *,
    width: int | None = None,
    height: int | None = None,
    include_has_workflow: bool = False,
) -> dict[str, Any]:
    """构造标准化的媒体元数据空模板。"""
    meta: dict[str, Any] = {
        "positive_prompt": "",
        "negative_prompt": "",
        "seed": None,
        "steps": None,
        "cfg": None,
        "sampler": "",
        "scheduler": "",
        "model": "",
        "loras": [],
        "width": width,
        "height": height,
    }
    if include_has_workflow:
        meta["has_workflow"] = False
    return meta


def _natural_sort_key(text: str) -> list[tuple[int, Any]]:
    """构造数字自然排序键，确保如 `00002` 排在 `00010` 之前。"""
    return [
        (0, int(tok)) if tok.isdigit() else (1, tok.lower())
        for tok in _NAT_SPLIT_RE.split(str(text or ""))
        if tok
    ]


def _get_cached_image_dims(resolved: Path, st: os.stat_result) -> tuple[int | None, int | None]:
    """按 `(mtime_ns, size_bytes)` 缓存图片头宽高，避免每次画廊刷新重复读取文件头。"""
    key = str(resolved)
    mtime_ns = int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1_000_000_000)))
    size_bytes = int(st.st_size)
    with _DIM_CACHE_LOCK:
        cached = _DIM_HEADER_CACHE.get(key)
        if cached is not None and cached[0] == mtime_ns and cached[1] == size_bytes:
            return cached[2], cached[3]
    w, h = _read_image_dimensions(resolved)
    with _DIM_CACHE_LOCK:
        if len(_DIM_HEADER_CACHE) >= 2048 and key not in _DIM_HEADER_CACHE:
            try:
                _DIM_HEADER_CACHE.pop(next(iter(_DIM_HEADER_CACHE)))
            except Exception:
                _DIM_HEADER_CACHE.clear()
        _DIM_HEADER_CACHE[key] = (mtime_ns, size_bytes, w, h)
    return w, h


def _classify_media_kind(ext: str) -> str | None:
    """根据文件扩展名判定媒体类别（image / video / audio）。"""
    low = ext.lower()
    if low in _IMAGE_EXTS:
        return "image"
    if low in _VIDEO_EXTS:
        return "video"
    if low in _AUDIO_EXTS:
        return "audio"
    return None


def _build_gallery_item_dict(root: Path, resolved: Path, kind: str) -> dict[str, Any]:
    """构造前端与 Inspector 通用的媒体条目结构。"""
    st = resolved.stat()
    rel = resolved.relative_to(root)
    rel_posix = rel.as_posix()
    sub_posix = rel.parent.as_posix()
    w, h = _get_cached_image_dims(resolved, st) if kind == "image" else (None, None)
    dt_obj = datetime.fromtimestamp(st.st_mtime)
    return {
        "rel_path": rel_posix,
        "media_url": "/__media/" + urllib.parse.quote(rel_posix),
        "name": resolved.name,
        "kind": kind,
        "ext": resolved.suffix.lower(),
        "size_bytes": int(st.st_size),
        "mtime_ts": float(st.st_mtime),
        "mtime_str": dt_obj.strftime("%Y-%m-%d %H:%M"),
        "date_str": dt_obj.strftime("%Y-%m-%d"),
        "width": w,
        "height": h,
        "subfolder": "" if sub_posix == "." else sub_posix,
    }


# ------------------------------------------------------------------
# ComfyUI Prompt 拓扑解析纯函数组
# ------------------------------------------------------------------


def _resolve_node_scalar(
    prompt_graph: dict[str, dict[str, Any]],
    val: Any,
    candidate_keys: tuple[str, ...],
    depth: int = 0,
) -> Any:
    """解析 ComfyUI 节点输入值（兼容直填标量与上游引用节点）。"""
    if depth > 6 or val is None:
        return None
    if isinstance(val, (str, int, float, bool)):
        return val
    if not isinstance(val, (list, tuple)) or not val:
        return None
    node = prompt_graph.get(str(val[0]))
    inputs = node.get("inputs") if isinstance(node, dict) else None
    if not isinstance(inputs, dict):
        return None
    for key in candidate_keys:
        if key in inputs and inputs[key] is not None:
            found = _resolve_node_scalar(prompt_graph, inputs[key], candidate_keys, depth + 1)
            if found is not None:
                return found
    for fallback_key in ("value", "string", "text", "seed", "noise_seed"):
        if fallback_key in inputs and isinstance(inputs[fallback_key], (str, int, float)):
            return inputs[fallback_key]
    return None


def _collect_conditioning_texts(
    prompt_graph: dict[str, dict[str, Any]],
    link_val: Any,
    visited: set[str] | None = None,
    depth: int = 0,
) -> list[str]:
    """沿 Conditioning 链接链递归回溯并提取编码文本提示词。"""
    if depth > 12 or not isinstance(link_val, (list, tuple)) or not link_val:
        return []
    node_id = str(link_val[0])
    seen = visited if visited is not None else set()
    if node_id in seen:
        return []
    seen.add(node_id)

    node = prompt_graph.get(node_id)
    inputs = node.get("inputs") if isinstance(node, dict) else None
    if not isinstance(inputs, dict):
        return []

    texts: list[str] = []
    for txt_key in ("text", "prompt", "t5xxl", "clip_l", "positive", "string"):
        if txt_key not in inputs:
            continue
        raw_v = inputs[txt_key]
        if isinstance(raw_v, str) and raw_v.strip():
            texts.append(raw_v.strip())
        elif isinstance(raw_v, (list, tuple)):
            resolved = _resolve_node_scalar(
                prompt_graph, raw_v, ("text", "string", "value", "prompt"), depth + 1
            )
            if isinstance(resolved, str) and resolved.strip():
                texts.append(resolved.strip())

    # 递归穿透 ConditioningCombine / FluxGuidance / ControlNetApply 等中间调节节点
    for next_key in (
        "conditioning",
        "conditioning_1",
        "conditioning_2",
        "conditioning_to",
        "conditioning_from",
        "positive",
        "negative",
    ):
        next_val = inputs.get(next_key)
        if isinstance(next_val, (list, tuple)):
            texts.extend(_collect_conditioning_texts(prompt_graph, next_val, seen, depth + 1))
    return texts


def _add_lora_entry(
    loras: list[dict[str, Any]],
    seen_names: set[str],
    raw_name: Any,
    raw_strength: Any = 1.0,
) -> None:
    """规范化并去重写入 LoRA 名称与权重条目。"""
    name_str = str(raw_name or "").strip()
    if not name_str or name_str.lower() == "none":
        return
    clean_name = Path(name_str.replace("\\", "/")).name
    if not clean_name or clean_name in seen_names:
        return
    try:
        strength_val = round(float(raw_strength if raw_strength is not None else 1.0), 4)
    except (TypeError, ValueError):
        strength_val = 1.0
    seen_names.add(clean_name)
    loras.append({"name": clean_name, "strength": strength_val})


def _coerce_int_scalar(val: Any) -> int | None:
    """安全将非布尔标量转换为整数。"""
    if val is None or isinstance(val, bool):
        return None
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def _coerce_float_scalar(val: Any, digits: int = 4) -> float | None:
    """安全将非布尔标量转换为保留指定位数的浮点数。"""
    if val is None or isinstance(val, bool):
        return None
    try:
        return round(float(val), digits)
    except (TypeError, ValueError):
        return None


def _find_primary_sampler_node(
    prompt_graph: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    """按节点类型与关键采样字段评分，定位工作流主采样器节点。"""
    sampler_node: dict[str, Any] | None = None
    best_score = -1
    for node in prompt_graph.values():
        if not isinstance(node, dict):
            continue
        cls_low = str(node.get("class_type") or "").lower()
        inputs = node.get("inputs") if isinstance(node.get("inputs"), dict) else {}
        score = 4 if any(h in cls_low for h in _SAMPLER_CLASS_HINTS) else 0
        for key in (
            "steps",
            "cfg",
            "sampler_name",
            "scheduler",
            "seed",
            "noise_seed",
            "guider",
            "sigmas",
            "noise",
        ):
            if key in inputs:
                score += 1
        if "positive" in inputs:
            score += 2
        if score > best_score and score >= 3:
            best_score = score
            sampler_node = node
    return sampler_node


def _first_present(inputs: dict[str, Any], keys: tuple[str, ...]) -> Any:
    """按优先级返回输入字典中首个非 `None` 键的值。"""
    for key in keys:
        val = inputs.get(key)
        if val is not None:
            return val
    return None


def _extract_sampler_conditioning(
    prompt_graph: dict[str, dict[str, Any]], s_in: dict[str, Any]
) -> tuple[list[str], list[str]]:
    """从采样器节点（含上游 `guider` 节点）回溯提取正/负向提示词列表。"""
    pos_texts = _collect_conditioning_texts(prompt_graph, s_in.get("positive")) if "positive" in s_in else []
    neg_texts = _collect_conditioning_texts(prompt_graph, s_in.get("negative")) if "negative" in s_in else []
    guider_ref = s_in.get("guider")
    if (pos_texts and neg_texts) or not isinstance(guider_ref, (list, tuple)) or not guider_ref:
        return pos_texts, neg_texts

    guider_node = prompt_graph.get(str(guider_ref[0]))
    g_in = guider_node.get("inputs") if isinstance(guider_node, dict) else None
    if not isinstance(g_in, dict):
        return pos_texts, neg_texts
    if not pos_texts and "positive" in g_in:
        pos_texts = _collect_conditioning_texts(prompt_graph, g_in.get("positive"))
    elif not pos_texts and "conditioning" in g_in:
        pos_texts = _collect_conditioning_texts(prompt_graph, g_in.get("conditioning"))
    if not neg_texts and "negative" in g_in:
        neg_texts = _collect_conditioning_texts(prompt_graph, g_in.get("negative"))
    return pos_texts, neg_texts


def _extract_sampler_params(
    prompt_graph: dict[str, dict[str, Any]],
    sampler_node: dict[str, Any],
    meta: dict[str, Any],
) -> tuple[list[str], list[str]]:
    """从主采样器节点填充 seed/steps/cfg/sampler/scheduler 并提取提示词。"""
    s_in = sampler_node.get("inputs") if isinstance(sampler_node.get("inputs"), dict) else {}

    seed_raw = _resolve_node_scalar(
        prompt_graph,
        _first_present(s_in, ("seed", "noise_seed", "noise")),
        ("seed", "noise_seed", "value"),
    )
    meta["seed"] = _coerce_int_scalar(seed_raw)

    steps_raw = _resolve_node_scalar(
        prompt_graph, _first_present(s_in, ("steps", "sigmas")), ("steps", "value")
    )
    meta["steps"] = _coerce_int_scalar(steps_raw)

    cfg_raw = _resolve_node_scalar(
        prompt_graph, _first_present(s_in, ("cfg", "guider")), ("cfg", "value")
    )
    meta["cfg"] = _coerce_float_scalar(cfg_raw)

    sampler_raw = _resolve_node_scalar(
        prompt_graph,
        _first_present(s_in, ("sampler_name", "sampler")),
        ("sampler_name", "value"),
    )
    if isinstance(sampler_raw, str) and sampler_raw.strip():
        meta["sampler"] = sampler_raw.strip()

    sched_raw = _resolve_node_scalar(
        prompt_graph, _first_present(s_in, ("scheduler", "sigmas")), ("scheduler", "value")
    )
    if isinstance(sched_raw, str) and sched_raw.strip():
        meta["scheduler"] = sched_raw.strip()

    return _extract_sampler_conditioning(prompt_graph, s_in)


def _extract_node_prompt_text(
    prompt_graph: dict[str, dict[str, Any]], inputs: dict[str, Any]
) -> str:
    """从单个文本编码节点中按优先级提取首段非空提示词文本。"""
    for key in ("text", "prompt", "t5xxl", "clip_l", "string"):
        cand = _resolve_node_scalar(prompt_graph, inputs.get(key), ("text", "string", "value"))
        if isinstance(cand, str) and cand.strip():
            return cand.strip()
    return ""


def _collect_fallback_prompt_texts(
    prompt_graph: dict[str, dict[str, Any]],
) -> tuple[list[str], list[str]]:
    """当采样器未连出提示词时，按节点标题与出现顺序扫描全图编码节点兜底。"""
    pos_texts: list[str] = []
    neg_texts: list[str] = []
    fallback_all: list[str] = []
    for node in prompt_graph.values():
        if not isinstance(node, dict):
            continue
        cls_low = str(node.get("class_type") or "").lower()
        if not any(h in cls_low for h in _PROMPT_TEXT_CLASS_HINTS):
            continue
        inputs = node.get("inputs") if isinstance(node.get("inputs"), dict) else {}
        txt_val = _extract_node_prompt_text(prompt_graph, inputs)
        if not txt_val:
            continue
        title = str((node.get("_meta") or {}).get("title") or "").lower()
        if any(w in title for w in ("neg", "负面", "反向")):
            neg_texts.append(txt_val)
        elif any(w in title for w in ("pos", "正面", "正向")):
            pos_texts.append(txt_val)
        else:
            fallback_all.append(txt_val)

    if not pos_texts and fallback_all:
        pos_texts.append(fallback_all[0])
        if not neg_texts and len(fallback_all) > 1:
            neg_texts.append(fallback_all[1])
    return pos_texts, neg_texts


def _extract_node_model_name(cls_low: str, inputs: dict[str, Any]) -> str:
    """若当前节点为模型加载器，提取规范化后的主模型文件名。"""
    for cls_hint, key_tuple in _MODEL_CLASS_KEYS:
        if cls_hint not in cls_low:
            continue
        for mk in key_tuple:
            mv = inputs.get(mk)
            if isinstance(mv, str) and mv.strip():
                return Path(mv.strip().replace("\\", "/")).name
    return ""


def _extract_node_loras(
    inputs: dict[str, Any], loras: list[dict[str, Any]], seen_loras: set[str]
) -> None:
    """从单个 LoRA 相关节点提取标准单 LoRA、Power Lora Loader 字典及多槽位 LoRA。"""
    if "lora_name" in inputs:
        str_val = inputs.get("strength_model")
        if str_val is None:
            str_val = inputs.get("strength", 1.0)
        _add_lora_entry(loras, seen_loras, inputs.get("lora_name"), str_val)
    for key, val in inputs.items():
        if isinstance(val, dict) and val.get("lora") and val.get("on", True):
            _add_lora_entry(
                loras,
                seen_loras,
                val.get("lora"),
                val.get("strength", val.get("strengthTwo", 1.0)),
            )
        elif _LORA_INDEXED_KEY_RE.match(str(key)):
            idx_suf = str(key).rsplit("_", 1)[-1]
            w_val = inputs.get(f"lora_wt_{idx_suf}", inputs.get(f"strength_{idx_suf}", 1.0))
            _add_lora_entry(loras, seen_loras, val, w_val)


def _extract_graph_model_loras_and_dims(
    prompt_graph: dict[str, dict[str, Any]], meta: dict[str, Any]
) -> None:
    """遍历工作流节点提取主底模、LoRA 列表与 Latent 尺寸。"""
    loras: list[dict[str, Any]] = []
    seen_loras: set[str] = set()
    for node in prompt_graph.values():
        if not isinstance(node, dict):
            continue
        cls_low = str(node.get("class_type") or "").lower()
        inputs = node.get("inputs") if isinstance(node.get("inputs"), dict) else {}

        if not meta["model"]:
            model_name = _extract_node_model_name(cls_low, inputs)
            if model_name:
                meta["model"] = model_name

        if "lora" in cls_low:
            _extract_node_loras(inputs, loras, seen_loras)

        if (meta["width"] is None or meta["height"] is None) and "latent" in cls_low:
            w_val = inputs.get("width")
            h_val = inputs.get("height")
            if isinstance(w_val, int) and isinstance(h_val, int) and w_val > 0 and h_val > 0:
                meta["width"] = w_val
                meta["height"] = h_val
    meta["loras"] = loras


def _parse_comfy_prompt_graph(prompt_graph: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """从 ComfyUI API prompt 拓扑字典结构化提取提示词、采样参数、底模与 LoRA 列表。"""
    meta = _empty_artwork_meta()
    sampler_node = _find_primary_sampler_node(prompt_graph)
    pos_texts, neg_texts = (
        _extract_sampler_params(prompt_graph, sampler_node, meta)
        if sampler_node is not None
        else ([], [])
    )
    if not pos_texts and not neg_texts:
        pos_texts, neg_texts = _collect_fallback_prompt_texts(prompt_graph)

    meta["positive_prompt"] = "\n".join(dict.fromkeys(pos_texts))
    meta["negative_prompt"] = "\n".join(dict.fromkeys(neg_texts))
    _extract_graph_model_loras_and_dims(prompt_graph, meta)
    return meta


# ------------------------------------------------------------------
# 纯标准库图片文件头宽高解析（严格零 Pillow 依赖）
# ------------------------------------------------------------------


def _valid_dims(w: int, h: int) -> tuple[int | None, int | None]:
    """校验宽高均为正整数，否则返回 `(None, None)`。"""
    return (int(w), int(h)) if (w > 0 and h > 0) else (None, None)


def _parse_png_dimensions(head: bytes) -> tuple[int | None, int | None]:
    """从 PNG IHDR 文件头提取宽高。"""
    if len(head) >= 24 and head[12:16] == b"IHDR":
        w, h = struct.unpack(">II", head[16:24])
        return _valid_dims(w, h)
    return (None, None)


def _parse_webp_dimensions(head: bytes) -> tuple[int | None, int | None]:
    """从 WebP（VP8X / VP8 / VP8L）文件头提取宽高。"""
    if len(head) < 20:
        return (None, None)
    chunk_tag = head[12:16]
    if chunk_tag == b"VP8X" and len(head) >= 30:
        w = 1 + int.from_bytes(head[24:27], "little")
        h = 1 + int.from_bytes(head[27:30], "little")
        return _valid_dims(w, h)
    if chunk_tag == b"VP8 " and len(head) >= 30 and head[23:26] == b"\x9d\x01\x2a":
        raw_w, raw_h = struct.unpack("<HH", head[26:30])
        return _valid_dims(raw_w & 0x3FFF, raw_h & 0x3FFF)
    if chunk_tag == b"VP8L" and len(head) >= 25 and head[20] == 0x2F:
        bits = struct.unpack("<I", head[21:25])[0]
        return _valid_dims((bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1)
    return (None, None)


def _read_next_jpeg_marker(fp: Any) -> int | None:
    """跳过非标记字节与连续 `0xFF` 填充，返回下一个 JPEG 段标记字节。"""
    while True:
        byte = fp.read(1)
        if not byte:
            return None
        if byte == b"\xff":
            break
    while byte == b"\xff":
        byte = fp.read(1)
        if not byte:
            return None
    return byte[0]


def _parse_jpeg_dimensions(fp: Any) -> tuple[int | None, int | None]:
    """扫描 JPEG SOF0..SOF15 段头提取宽高（不解码像素数据）。"""
    fp.seek(2)
    while True:
        marker = _read_next_jpeg_marker(fp)
        if marker is None or marker == 0xDA:
            return (None, None)
        if marker in (0x00, 0x01, 0xD8, 0xD9) or (0xD0 <= marker <= 0xD7):
            continue
        len_raw = fp.read(2)
        if len(len_raw) < 2:
            return (None, None)
        seg_len = struct.unpack(">H", len_raw)[0]
        if seg_len < 2:
            return (None, None)
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            payload = fp.read(5)
            if len(payload) < 5:
                return (None, None)
            h, w = struct.unpack(">HH", payload[1:5])
            return _valid_dims(w, h)
        fp.seek(seg_len - 2, 1)


def _read_image_dimensions(path: Path) -> tuple[int | None, int | None]:
    """仅读取 PNG / WebP / JPEG 文件头字节解析图片宽高，不解码整图且零第三方依赖。"""
    try:
        with open(path, "rb") as fp:
            head = fp.read(32)
            if head[:8] == b"\x89PNG\r\n\x1a\n":
                return _parse_png_dimensions(head)
            if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
                return _parse_webp_dimensions(head)
            if head[:2] == b"\xff\xd8":
                return _parse_jpeg_dimensions(fp)
    except Exception:
        return (None, None)
    return (None, None)


# ------------------------------------------------------------------
# PNG 文本块与画廊过滤排序纯函数组
# ------------------------------------------------------------------


def _apply_webui_parameters_fallback(params_raw: Any, meta: dict[str, Any]) -> None:
    """在缺少 ComfyUI prompt 时兼容提取传统 WebUI PNG `parameters` 文本块。"""
    if meta["positive_prompt"] or not isinstance(params_raw, str):
        return
    params_txt = params_raw.strip()
    if "Negative prompt:" in params_txt:
        pos_part, rest_part = params_txt.split("Negative prompt:", 1)
        meta["positive_prompt"] = pos_part.strip()
        meta["negative_prompt"] = rest_part.split("\nSteps:", 1)[0].strip()
    elif params_txt:
        meta["positive_prompt"] = params_txt.split("\nSteps:", 1)[0].strip()


def _detect_png_has_workflow(
    target: Path, raw_workflow: Any, parsed_prompt: dict[str, Any] | None
) -> bool:
    """判断 PNG 文件是否内嵌可恢复的 ComfyUI workflow 或 API prompt。"""
    if parsed_prompt or extract_png_comfy_workflow(target):
        return True
    if isinstance(raw_workflow, str) and raw_workflow.strip():
        try:
            wf_obj = json.loads(raw_workflow)
            return isinstance(wf_obj, dict) and ("nodes" in wf_obj or "links" in wf_obj)
        except Exception:
            return False
    return False


def _parse_png_metadata(target: Path) -> dict[str, Any]:
    """使用纯标准库解析 PNG 尺寸与 ComfyUI 元数据（零 Pillow 依赖）。"""
    width, height = _read_image_dimensions(target)
    try:
        png_info = _read_png_text_chunks(target)
    except Exception:
        png_info = {}

    meta = _empty_artwork_meta(width=width, height=height, include_has_workflow=True)
    raw_prompt = png_info.get("prompt")
    parsed_prompt: dict[str, Any] | None = None
    if isinstance(raw_prompt, str) and raw_prompt.strip():
        try:
            obj = json.loads(raw_prompt)
            if isinstance(obj, dict):
                parsed_prompt = obj
        except Exception:
            parsed_prompt = None

    if parsed_prompt:
        valid_graph = {
            str(k): v
            for k, v in parsed_prompt.items()
            if isinstance(v, dict) and isinstance(v.get("inputs"), dict)
        }
        if valid_graph:
            extracted = _parse_comfy_prompt_graph(valid_graph)
            for k, v in extracted.items():
                if k in ("width", "height") and meta[k] is not None:
                    continue
                meta[k] = v

    _apply_webui_parameters_fallback(png_info.get("parameters"), meta)
    meta["has_workflow"] = _detect_png_has_workflow(
        target, png_info.get("workflow"), parsed_prompt
    )

    # 脱敏所有字符串字段，避免工作流自定义节点内意外残留凭据
    for txt_key in ("positive_prompt", "negative_prompt", "sampler", "scheduler", "model"):
        if isinstance(meta.get(txt_key), str):
            meta[txt_key] = redact_secrets(meta[txt_key])
    return meta


def _normalize_gallery_query(config: dict[str, Any] | None) -> dict[str, Any]:
    """规范化画廊筛选、排序与分页参数。"""
    cfg = config if isinstance(config, dict) else {}
    kind_filter = str(cfg.get("kind") or "all").strip().lower()
    if kind_filter not in ("all", "image", "video", "audio"):
        kind_filter = "all"
    sub_filter = str(cfg.get("subfolder") or "all").strip() or "all"
    date_range = str(cfg.get("date_range") or "all").strip().lower()
    if date_range not in ("all", "today", "7d", "30d"):
        date_range = "all"
    sort_by = str(cfg.get("sort") or "newest").strip().lower()
    if sort_by not in ("newest", "oldest", "name_asc", "name_desc", "size_desc"):
        sort_by = "newest"
    try:
        limit = max(1, min(2000, int(cfg.get("limit") or 500)))
    except (TypeError, ValueError):
        limit = 500
    return {
        "kind": kind_filter,
        "subfolder": sub_filter,
        "date_range": date_range,
        "sort": sort_by,
        "query": str(cfg.get("query") or cfg.get("search") or "").strip().lower(),
        "limit": limit,
    }


def _scan_gallery_directory(root: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """安全遍历沙箱根目录内的媒体文件，返回 `(全部条目, 分类计数)`。"""
    counts = {"all": 0, "image": 0, "video": 0, "audio": 0}
    all_items: list[dict[str, Any]] = []
    for entry in root.rglob("*"):
        try:
            resolved = entry.resolve()
            resolved.relative_to(root)
            if not resolved.is_file():
                continue
            kind = _classify_media_kind(resolved.suffix)
            if kind is None:
                continue
            item = _build_gallery_item_dict(root, resolved, kind)
        except Exception:
            continue
        counts["all"] += 1
        counts[kind] += 1
        all_items.append(item)
    return all_items, counts


def _aggregate_gallery_folders(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按子目录聚合媒体数量与最新修改时间，并按最新时间倒序排列。"""
    folder_agg: dict[str, dict[str, Any]] = {}
    for it in items:
        sub = str(it.get("subfolder") or "")
        fid = sub or "__root__"
        rec = folder_agg.get(fid)
        if rec is None:
            folder_agg[fid] = {
                "id": fid,
                "name": sub or "根目录",
                "subfolder": sub,
                "count": 1,
                "latest_ts": float(it["mtime_ts"]),
            }
        else:
            rec["count"] += 1
            if float(it["mtime_ts"]) > rec["latest_ts"]:
                rec["latest_ts"] = float(it["mtime_ts"])
    return sorted(folder_agg.values(), key=lambda f: (-f["latest_ts"], f["name"].lower()))


def _resolve_date_cutoff(date_range: str, now_ts: float) -> float | None:
    """将日期筛选枚举转换为 Unix 时间戳下限。"""
    if date_range == "today":
        return datetime.fromtimestamp(now_ts).replace(
            hour=0, minute=0, second=0, microsecond=0
        ).timestamp()
    if date_range == "7d":
        return now_ts - 7 * 86400
    if date_range == "30d":
        return now_ts - 30 * 86400
    return None


def _matches_gallery_filters(
    item: dict[str, Any],
    sub_filter: str,
    cutoff_ts: float | None,
    query_txt: str,
) -> bool:
    """判断单条媒体记录是否命中目录、时间与关键词筛选条件。"""
    if sub_filter != "all":
        expected_sub = "" if sub_filter == "__root__" else sub_filter
        if item["subfolder"] != expected_sub:
            return False
    if cutoff_ts is not None and float(item["mtime_ts"]) < cutoff_ts:
        return False
    if query_txt and query_txt not in item["name"].lower() and query_txt not in item["rel_path"].lower():
        return False
    return True


def _sort_gallery_items(items: list[dict[str, Any]], sort_by: str) -> list[dict[str, Any]]:
    """按指定排序策略原地排序并返回媒体条目列表。"""
    if sort_by == "oldest":
        items.sort(key=lambda it: (it["mtime_ts"], it["rel_path"]))
    elif sort_by == "name_asc":
        items.sort(key=lambda it: (_natural_sort_key(it["name"]), it["rel_path"].lower()))
    elif sort_by == "name_desc":
        items.sort(key=lambda it: (_natural_sort_key(it["name"]), it["rel_path"].lower()), reverse=True)
    elif sort_by == "size_desc":
        items.sort(key=lambda it: (it["size_bytes"], it["mtime_ts"], it["rel_path"]), reverse=True)
    else:
        items.sort(key=lambda it: (it["mtime_ts"], it["rel_path"]), reverse=True)
    return items


def _parse_rclone_lsjson_output(raw_stdout: str) -> tuple[bool, list[dict[str, Any]], str]:
    """解析 `rclone lsjson` 输出并过滤目录条目。"""
    try:
        parsed = json.loads((raw_stdout or "[]").strip() or "[]")
    except Exception as exc:
        return False, [], f"解析云盘列表失败: {redact_secrets(str(exc))}"
    if not isinstance(parsed, list):
        return False, [], ""
    items: list[dict[str, Any]] = []
    for entry in parsed:
        if not isinstance(entry, dict) or entry.get("IsDir"):
            continue
        items.append(
            {
                "name": str(entry.get("Name") or Path(str(entry.get("Path") or "")).name),
                "path": str(entry.get("Path") or entry.get("Name") or ""),
                "size_bytes": int(entry.get("Size") or 0),
                "mod_time": str(entry.get("ModTime") or ""),
                "mime_type": str(entry.get("MimeType") or ""),
            }
        )
    return True, items, ""


def _normalize_tail_log_config(config: dict[str, Any] | None) -> tuple[str, int, str]:
    """规范化日志拉取参数，返回 `(source, limit, path_override)`。"""
    cfg = config if isinstance(config, dict) else {}
    source = str(cfg.get("source") or "local_diag").strip().lower()
    if source in ("local", "diag"):
        source = "local_diag"
    if source not in ("local_diag", "llama", "comfy"):
        source = "local_diag"
    try:
        limit = max(10, min(400, int(cfg.get("lines") or cfg.get("limit") or 120)))
    except (TypeError, ValueError):
        limit = 120
    return source, limit, str(cfg.get("path") or "").strip()


def _extract_remote_tail_error(framed: dict[str, Any] | None, proc: Any) -> str:
    """从远端日志拉取失败结果中提炼末行可读错误消息。"""
    if framed and framed.get("message"):
        return str(framed["message"])
    if getattr(proc, "returncode", 0) != 0:
        raw_err = (getattr(proc, "stderr", "") or getattr(proc, "stdout", "") or "").strip()
        err_lines = raw_err.splitlines()
        return err_lines[-1] if err_lines else f"contents-tail 退出码 {proc.returncode}"
    return "未解析到远端日志内容"


class StudioService:
    """Colab Studio 桌面端媒体库、元数据分析、云盘资产与实时日志服务。"""

    def __init__(self, backend: ColabPetBackend) -> None:
        self._b = backend
        self._store = backend._store
        self._secrets = backend._secrets
        self._bus = backend._bus
        self._ctx = backend._ctx

    def _sandbox_root(self) -> Path:
        """返回严格隔离的唯一媒体沙箱根目录（download_root）。"""
        return self._b.download_root.resolve()

    def _resolve_media_item_path(self, rel_path: str) -> tuple[Path, Path, str]:
        """将相对路径解析为沙箱内绝对路径并拦截任何 `..` 或跨盘越界访问。"""
        raw = str(rel_path or "").strip()
        if not raw or "\x00" in raw:
            raise PermissionError("空路径或包含非法字符")
        clean = raw.replace("\\", "/").lstrip("/")
        if not clean or _WIN_DRIVE_RE.match(clean) or Path(clean).is_absolute():
            raise PermissionError("禁止使用绝对路径或驱动器前缀")
        root = self._sandbox_root()
        target = (root / clean).resolve()
        try:
            rel_posix = target.relative_to(root).as_posix()
        except ValueError as exc:
            raise PermissionError("检测到目录越界访问，已拦截") from exc
        if target == root or not rel_posix:
            raise PermissionError("目标不能为沙箱根目录")
        return root, target, rel_posix

    def list_gallery_items(self, config: dict[str, Any] | None = None) -> dict[str, Any]:
        """扫描本地 `download_root` 沙箱内的图像/视频/语音作品，支持目录/时间筛选与多维排序。"""
        root = self._sandbox_root()
        q = _normalize_gallery_query(config)
        if not root.is_dir():
            return {
                "ok": True,
                "root": str(root),
                "items": [],
                "counts": {"all": 0, "image": 0, "video": 0, "audio": 0},
                "folders": [],
                "total": 0,
            }

        all_items, counts = _scan_gallery_directory(root)
        kind_matched = [
            it for it in all_items if (q["kind"] == "all" or it["kind"] == q["kind"])
        ]
        folders_out = _aggregate_gallery_folders(kind_matched)
        cutoff_ts = _resolve_date_cutoff(q["date_range"], self._ctx.now())
        filtered = [
            it
            for it in kind_matched
            if _matches_gallery_filters(it, q["subfolder"], cutoff_ts, q["query"])
        ]
        _sort_gallery_items(filtered, q["sort"])
        return {
            "ok": True,
            "root": str(root),
            "items": filtered[: q["limit"]],
            "counts": counts,
            "folders": folders_out,
            "total": len(filtered),
        }

    def inspect_gallery_item(self, rel_path: str) -> dict[str, Any]:
        """安全解析单个作品的元数据（含 PNG 内嵌 ComfyUI prompt/workflow 与 LoRA 权重）。"""
        try:
            root, target, _ = self._resolve_media_item_path(rel_path)
        except PermissionError as exc:
            return {"ok": False, "error": "forbidden", "message": str(exc)}

        if not target.is_file():
            return {"ok": False, "error": "not_found", "message": "作品文件不存在"}

        kind = _classify_media_kind(target.suffix) or "other"
        item = _build_gallery_item_dict(root, target, kind)
        if target.suffix.lower() == ".png":
            meta = _parse_png_metadata(target)
        else:
            w, h = _read_image_dimensions(target) if kind == "image" else (None, None)
            meta = _empty_artwork_meta(width=w, height=h, include_has_workflow=True)
        return {"ok": True, "item": item, "meta": meta}

    def _resolve_comfy_canvas_tunnel_url(self) -> str | None:
        """优先从集群 ComfyUI 槽位解析在线隧道，未命中时回退主隧道解析。"""
        b = self._b
        if b._cluster_svc.has_active_slots():
            comfy_slot = b._cluster_svc.get_slot("comfy")
            if comfy_slot:
                turl = str(comfy_slot.get("tunnel_url") or "").strip() or None
                if turl:
                    return turl
                _, _, fast_u = b._tunnel_svc.fast_read_slot_tunnel_url(
                    str(comfy_slot.get("endpoint") or ""),
                    str(comfy_slot.get("session") or "comfy"),
                    account_id=str(comfy_slot.get("account_id") or ""),
                )
                if fast_u:
                    return fast_u
        return b.resolve_active_tunnel_url(allow_fast_discover=True)

    def push_gallery_workflow_to_canvas(self, rel_path: str) -> dict[str, Any]:
        """将画廊作品内嵌的工作流安全提取并推送到当前在线 ComfyUI 画布（兼容集群槽位）。"""
        b = self._b
        try:
            _, target, _ = self._resolve_media_item_path(rel_path)
        except PermissionError as exc:
            return {"ok": False, "error": "forbidden", "message": str(exc)}

        if not target.is_file():
            return {"ok": False, "error": "not_found", "message": "作品文件不存在"}

        if not extract_png_comfy_workflow(target):
            return {
                "ok": False,
                "message": f"{target.name} 中未发现合法的 ComfyUI 工作流元数据",
            }

        try:
            b.touch_activity("comfy")
            turl = self._resolve_comfy_canvas_tunnel_url()
            msg, online_ok, safe_wf_name = b._upload_workflow_file_ex(
                target, is_png_extract=True, tunnel_url=turl
            )
            pushed_live = False
            opened_browser = False
            open_url = ""

            if online_ok and turl:
                push_res = b._push_or_open_workflow_on_canvas(turl, safe_wf_name)
                pushed_live = bool(push_res.get("pushed_live"))
                opened_browser = bool(push_res.get("opened_browser"))
                open_url = str(push_res.get("url") or "")
                msg = (
                    f"已推送工作流至在线画布: {safe_wf_name}"
                    if pushed_live
                    else f"已在画布打开工作流: {safe_wf_name}"
                )
            elif online_ok or self._store.has_running_session():
                open_res = b.open_comfy_tunnel(workflow_name=safe_wf_name)
                opened_browser = bool(open_res.get("ok"))
                open_url = str(open_res.get("url") or "")
                if open_res.get("message"):
                    msg = str(open_res["message"])

            return {
                "ok": True,
                "workflow_name": safe_wf_name,
                "online": bool(online_ok),
                "pushed_live": pushed_live,
                "opened_browser": opened_browser,
                "url": redact_secrets(open_url),
                "message": redact_secrets(msg),
            }
        except Exception as exc:
            err_msg = redact_secrets(str(exc))
            b._diag.error("studio", f"推送画廊工作流失败: {err_msg}")
            return {"ok": False, "message": f"推送工作流失败: {err_msg}"}

    def reveal_gallery_item(self, rel_path: str) -> dict[str, Any]:
        """在系统文件管理器中高亮定位指定的沙箱内媒体文件。"""
        try:
            _, target, _ = self._resolve_media_item_path(rel_path)
        except PermissionError as exc:
            return {"ok": False, "error": "forbidden", "message": str(exc)}

        if not target.is_file():
            return {"ok": False, "error": "not_found", "message": "作品文件不存在"}

        try:
            self._ctx.reveal_path(target)
            return {
                "ok": True,
                "path": str(target),
                "message": f"已在文件管理器中定位: {target.name}",
            }
        except Exception as exc:
            return {"ok": False, "message": f"定位文件失败: {redact_secrets(str(exc))}"}

    def _query_cloud_category(
        self, rclone_exe: Path, rclone_conf: Path, candidates: tuple[str, ...]
    ) -> tuple[bool, list[dict[str, Any]], str]:
        """依次查询单个云盘资产类别的候选远端路径，返回 `(可达, 文件列表, 末次错误)`。"""
        last_err = ""
        any_ok = False
        for remote_uri in candidates:
            cmd = [
                str(rclone_exe),
                "lsjson",
                "--config",
                str(rclone_conf),
                "--disable-http2",
                remote_uri,
            ]
            try:
                proc = self._ctx.run_hidden(cmd, timeout=10)
            except Exception as exc:
                last_err = redact_secrets(str(exc))
                continue
            if getattr(proc, "returncode", 1) != 0:
                err_lines = redact_secrets(
                    (getattr(proc, "stderr", "") or getattr(proc, "stdout", "") or "").strip()
                ).splitlines()
                last_err = err_lines[-1] if err_lines else f"rclone 退出码 {proc.returncode}"
                continue
            ok_parse, items, parse_err = _parse_rclone_lsjson_output(getattr(proc, "stdout", ""))
            if not ok_parse:
                if parse_err:
                    last_err = parse_err
                continue
            any_ok = True
            if items:
                return True, items, ""
        return any_ok, [], last_err

    def list_cloud_assets(self) -> dict[str, Any]:
        """安全并发查询 Google Drive 云盘上的 LoRA、工作流与配音参考资产清单。"""
        rclone_exe = self._ctx.get_rclone_exe()
        rclone_conf = self._ctx.get_rclone_conf()
        empty_base = {"ok": False, "configured": False, "loras": [], "workflows": [], "refs": []}

        if not rclone_exe.is_file() or not rclone_conf.is_file():
            return {**empty_base, "message": "未检测到 rclone 或 rclone.conf 配置文件"}
        if not has_gdrive_save_section(rclone_conf):
            return {**empty_base, "message": "rclone.conf 中尚未配置 [gdrive_save] 云盘"}

        out_map: dict[str, list[dict[str, Any]]] = {"loras": [], "workflows": [], "refs": []}
        ok_count = 0
        errors: list[str] = []
        try:
            with ThreadPoolExecutor(max_workers=3) as pool:
                fut_map = {
                    pool.submit(self._query_cloud_category, rclone_exe, rclone_conf, remotes): cat
                    for cat, remotes in _CLOUD_ASSET_REMOTES
                }
                for fut, cat in fut_map.items():
                    try:
                        cat_ok, items, err = fut.result(timeout=15)
                        out_map[cat] = items
                        if cat_ok:
                            ok_count += 1
                        elif err:
                            errors.append(f"{cat}: {err}")
                    except Exception as exc:
                        errors.append(f"{cat}: {redact_secrets(str(exc))}")
        except Exception as exc:
            return {
                "ok": False,
                "configured": True,
                "message": f"云盘查询异常: {redact_secrets(str(exc))}",
                "loras": [],
                "workflows": [],
                "refs": [],
            }

        ok_flag = ok_count > 0
        msg = (
            f"已加载云盘资产：LoRA {len(out_map['loras'])} / 工作流 {len(out_map['workflows'])} / 音色 {len(out_map['refs'])}"
            if ok_flag
            else (errors[0] if errors else "云盘离线或目录为空")
        )
        return {
            "ok": ok_flag,
            "configured": True,
            "message": redact_secrets(msg),
            "loras": out_map["loras"],
            "workflows": out_map["workflows"],
            "refs": out_map["refs"],
        }

    def _tail_local_diag_logs(self, limit: int, updated_str: str) -> dict[str, Any]:
        """读取本地诊断轮转日志末尾指定行数并全量脱敏。"""
        try:
            raw_lines = self._b._diag.read_tail_lines(limit)
            safe_lines = [redact_secrets(str(ln)) for ln in raw_lines][-limit:]
            return {
                "ok": True,
                "source": "local_diag",
                "path": str(self._b._diag.log_path),
                "lines": safe_lines,
                "updated_str": updated_str,
            }
        except Exception as exc:
            return {
                "ok": False,
                "source": "local_diag",
                "path": "",
                "lines": [],
                "message": f"读取本地诊断日志失败: {redact_secrets(str(exc))}",
                "updated_str": updated_str,
            }

    def _resolve_remote_log_target(self, source: str) -> tuple[str, str]:
        """结合异构集群分槽与主会话解析远端日志归属 `(account_id, endpoint)`。"""
        slots = self._b._cluster_svc.get_cluster_slots()
        slot_obj = slots.get(source)
        if not isinstance(slot_obj, dict) and source == "llama":
            slot_obj = slots.get("llm")
        if isinstance(slot_obj, dict):
            return (
                str(slot_obj.get("account_id") or "").strip(),
                str(slot_obj.get("endpoint") or "").strip(),
            )
        acct_id = self._store.get_active_account_id()
        sess = self._store.pick_target_session("comfy")
        ep = str(sess.get("endpoint") or "").strip() if isinstance(sess, dict) else ""
        return acct_id, ep

    def tail_live_logs(self, config: dict[str, Any] | None = None) -> dict[str, Any]:
        """读取本地诊断日志或远端 LLM / ComfyUI 实例末尾 N 行日志（全量脱敏）。"""
        source, limit, path_override = _normalize_tail_log_config(config)
        updated_str = datetime.fromtimestamp(self._ctx.now()).strftime("%H:%M:%S")
        if source == "local_diag":
            return self._tail_local_diag_logs(limit, updated_str)

        acct_id, ep = self._resolve_remote_log_target(source)
        if not ep and not self._store.has_running_session():
            return {
                "ok": False,
                "source": source,
                "path": "",
                "lines": [],
                "message": f"当前无运行中的 {source.upper()} 云端实例",
                "updated_str": updated_str,
            }

        colab_py = self._ctx.get_colab_py()
        if not colab_py.is_file():
            return {
                "ok": False,
                "source": source,
                "path": "",
                "lines": [],
                "message": "未找到 google-colab-cli Python 解释器",
                "updated_str": updated_str,
            }

        cmd = [
            str(colab_py),
            str(self._ctx.cli_bridge_script),
            "contents-tail",
            f"--lines={limit}",
        ]
        if ep:
            cmd.append(f"--ep={ep}")
        if acct_id:
            cmd.append(f"--acct={acct_id}")
        if path_override:
            cmd.append(f"--path={path_override}")
        elif source == "llama":
            cmd.append("--path=root/llama_server.log")

        try:
            proc = self._ctx.run_hidden(cmd, timeout=18)
        except Exception as exc:
            return {
                "ok": False,
                "source": source,
                "path": "",
                "lines": [],
                "message": f"拉取远端日志失败: {redact_secrets(str(exc))}",
                "updated_str": updated_str,
            }

        framed = extract_framed_json(getattr(proc, "stdout", ""), "COLAB_CONTENTS_TAIL")
        if framed and framed.get("ok"):
            safe_lines = [redact_secrets(str(ln)) for ln in (framed.get("lines") or [])][-limit:]
            return {
                "ok": True,
                "source": source,
                "path": redact_secrets(str(framed.get("path") or "")),
                "lines": safe_lines,
                "updated_str": updated_str,
            }

        err_msg = _extract_remote_tail_error(framed, proc)
        return {
            "ok": False,
            "source": source,
            "path": redact_secrets(str((framed or {}).get("path") or path_override or "")),
            "lines": [],
            "message": redact_secrets(err_msg),
            "updated_str": updated_str,
        }
