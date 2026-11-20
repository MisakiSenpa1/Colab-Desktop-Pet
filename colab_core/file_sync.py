"""文件同步、云盘强校验归档、产出时间防覆盖下载与 ComfyUI 工作流转换模块。"""

from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import re
import struct
import sys
import tarfile
import tempfile
import time
import uuid
import zlib
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path, PurePosixPath
from typing import Any

_PKG_DIR = Path(__file__).resolve().parents[1]
if str(_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR))

from colab_core.exec_runner import get_diag_logger, suppress_logged  # noqa: E402

_CST_TZ = timezone(timedelta(hours=8))
_TS_SUFFIX_RE = re.compile(r"_\d{8}-\d{6}(?:-\d+)?$")
_WORKFLOW_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)

GDRIVE_LORA_REMOTE = "gdrive_save:Colab_ComfyUI_Models/loras"
GDRIVE_DRAMA3_REFS_REMOTE = "gdrive_save:Colab_ComfyUI_Models/drama3/refs"
GDRIVE_OUT_REMOTE = "gdrive_save:Colab_ComfyUI_Output"
GDRIVE_WF_REMOTE = "gdrive_save:Colab_ComfyUI_Models/workflows"
REMOTE_OUT_ROOT = "/root/comfy_out"
REMOTE_RCLONE_CONF = "/tmp/rclone.conf"
REMOTE_PULL_TAR = "/tmp/comfy_pull.tar.gz"

MEDIA_EXTS = {
    ".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp",
    ".mp4", ".mov", ".webm", ".mkv",
    ".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus",
}
WORKFLOW_EXTS = {".json"}
MODEL_EXTS = {".safetensors", ".ckpt", ".pt", ".pth"}
LLM_TEXT_EXTS = {
    ".txt", ".md", ".json", ".py", ".yaml", ".yml", ".csv",
    ".log", ".toml", ".ini", ".js", ".ts", ".html", ".css",
}
LLM_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}

# 可选生产级 Anima LoRA 清单：(LoRA备份下相对路径, 精确字节数, SHA256)；公开仓库默认留空按目录自动发现
ANIMA_LORAS: list[tuple[str, int | None, str | None]] = []


REMOTE_SYNC_SCRIPT = r"""
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath

REMOTE_SRC = __REMOTE_SRC__
GDRIVE_DST = __GDRIVE_DST__
REMOTE_RCLONE_CONF = __REMOTE_RCLONE_CONF__
REMOTE_PULL_TAR = __REMOTE_PULL_TAR__
DO_PULL = __DO_PULL__

CST_TZ = timezone(timedelta(hours=8))
TS_SUFFIX_RE = re.compile(r"_\d{8}-\d{6}(?:-\d+)?$")
PRESERVE_RE = re.compile(r"(?:^|/)drama3_voices/[^/]+\.safetensors$", re.IGNORECASE)

meta_file = Path("/root/colab_service_meta.json")
if meta_file.is_file():
    try:
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        if isinstance(meta, dict) and meta.get("kind") == "llm":
            print("[提示] 当前实例运行的是 Qwen3.8 LLM 对话服务，无 ComfyUI 成图目录需要同步，已安全跳过。", flush=True)
            print("SKIP_LLM_SYNC", flush=True)
            sys.exit(0)
    except Exception:
        pass

src = Path(REMOTE_SRC)
if not src.exists():
    raise FileNotFoundError(f"远端输出目录不存在: {src}")

rclone_bin = shutil.which("rclone") or "/usr/local/bin/rclone"
if not Path(REMOTE_RCLONE_CONF).is_file():
    raise FileNotFoundError("远端缺少 /tmp/rclone.conf")


def _timestamped_name(name: str, mtime_sec: float) -> str:
    p = PurePosixPath(name)
    stem, suffix = p.stem, p.suffix
    if TS_SUFFIX_RE.search(stem):
        return name
    ts_str = datetime.fromtimestamp(mtime_sec, CST_TZ).strftime("%Y%m%d-%H%M%S")
    sep = "" if stem.endswith("_") else "_"
    return f"{stem}{sep}{ts_str}{suffix}"


def _unique_rel_path(rel: PurePosixPath, mtime_sec: float, used: set[str]) -> PurePosixPath:
    rel_posix = rel.as_posix()
    if PRESERVE_RE.search(rel_posix):
        candidate_name = rel.name
    else:
        candidate_name = _timestamped_name(rel.name, mtime_sec)
    cand = rel.with_name(candidate_name)
    cand_posix = cand.as_posix()
    if cand_posix not in used:
        used.add(cand_posix)
        return cand
    base_p = PurePosixPath(candidate_name)
    idx = 2
    while True:
        alt_name = f"{base_p.stem}-{idx}{base_p.suffix}"
        alt_rel = rel.with_name(alt_name)
        alt_posix = alt_rel.as_posix()
        if alt_posix not in used:
            used.add(alt_posix)
            return alt_rel
        idx += 1


stage_root = Path("/root/.comfy_sync_stage")
stage_dir = stage_root / f"job_{os.getpid()}_{time.time_ns()}"
stage_dir.mkdir(parents=True, exist_ok=True)

try:
    src_files = [
        p for p in sorted(src.rglob("*"))
        if p.is_file() and ".comfy_sync_stage" not in p.parts
    ]
    used_rels: set[str] = set()
    staged_pairs: list[tuple[Path, str]] = []
    for f in src_files:
        rel = PurePosixPath(f.relative_to(src).as_posix())
        st = f.stat()
        ts_rel = _unique_rel_path(rel, st.st_mtime, used_rels)
        dst_file = stage_dir / ts_rel.as_posix()
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(f, dst_file)
        except OSError:
            shutil.copy2(f, dst_file)
        staged_pairs.append((dst_file, ts_rel.as_posix()))

    if staged_pairs:
        cmd = [
            rclone_bin, "copy",
            "--config", REMOTE_RCLONE_CONF,
            "--disable-http2",
            str(stage_dir), GDRIVE_DST,
            "--transfers", "8", "-v",
        ]
        print("[云端同步]", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
        print(f"[云端同步] 已完成上传至 {GDRIVE_DST}（共 {len(staged_pairs)} 个带时间戳文件）", flush=True)
    else:
        print(f"[云端同步] 远端输出目录暂无文件，跳过上传: {src}", flush=True)

    tar_path = Path(REMOTE_PULL_TAR)
    if tar_path.exists():
        tar_path.unlink()

    if DO_PULL:
        print(f"[云端打包] 准备打包 {len(staged_pairs)} 个文件 -> {tar_path}", flush=True)
        with tarfile.open(tar_path, "w:gz") as tf:
            for sf, arcname in staged_pairs:
                tf.add(sf, arcname=arcname)
        print(f"[云端打包] 完成，大小={tar_path.stat().st_size} B", flush=True)
    print("<<<SYNC_VERIFIED>>>" + json.dumps({"ok": True, "files": len(staged_pairs)}) + "<<<END_SYNC_VERIFIED>>>", flush=True)
finally:
    shutil.rmtree(stage_dir, ignore_errors=True)
"""


def sha256_file(path: Path) -> str:
    """分块计算文件 SHA256 摘要，避免大文件一次性加载入内存。"""
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        while True:
            chunk = fp.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def sanitize_filename(name: str, fallback: str = "file") -> str:
    """清洗文件名非法字符并仅保留单层文件名。"""
    base = Path(str(name).replace("\\", "/")).name
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", base).strip()
    return cleaned or fallback


def safe_output_dest(root: Path, subfolder: str, filename: str) -> Path:
    """构造严格位于 root 目录下的安全输出落盘路径，防御路径穿越。"""
    root_resolved = root.resolve()
    parts: list[str] = []
    if subfolder:
        for seg in PurePosixPath(subfolder.replace("\\", "/")).parts:
            if seg and seg not in (".", "..", "/"):
                parts.append(sanitize_filename(seg, "sub"))
    parts.append(sanitize_filename(filename, "output.png"))
    dest = root_resolved.joinpath(*parts).resolve()
    if dest != root_resolved and root_resolved not in dest.parents:
        raise ValueError(f"非法输出路径: {subfolder}/{filename}")
    return dest


def format_timestamped_filename(filename: str, mtime_sec: float | None = None) -> str:
    """为文件名附加确定性北京时间时间戳后缀，避免重开实例后序号覆写历史作品。"""
    p = PurePosixPath(filename)
    stem, suffix = p.stem, p.suffix
    if _TS_SUFFIX_RE.search(stem):
        return filename
    ts_str = datetime.fromtimestamp(
        mtime_sec if mtime_sec is not None else time.time(), _CST_TZ
    ).strftime("%Y%m%d-%H%M%S")
    sep = "" if stem.endswith("_") else "_"
    return f"{stem}{sep}{ts_str}{suffix}"


def resolve_non_clobber_dest(dest: Path, src_size: int, src_sha256: str) -> tuple[Path, bool]:
    """若目标文件同内容已存在则跳过覆写；若内容不同则递增序号防止误覆盖。"""
    if not dest.exists():
        return dest, False
    candidate = dest
    idx = 2
    while candidate.exists():
        if candidate.is_file() and candidate.stat().st_size == src_size:
            if sha256_file(candidate) == src_sha256:
                return candidate, True
        candidate = dest.with_name(f"{dest.stem}-{idx}{dest.suffix}")
        idx += 1
    return candidate, False


def normalize_prefix(prefix: str | None) -> str:
    """校验并规范化子目录前缀，阻止路径穿越。"""
    if not prefix:
        return ""
    cleaned = prefix.replace("\\", "/").strip("/")
    if not cleaned:
        return ""
    posix_p = PurePosixPath(cleaned)
    if posix_p.is_absolute() or ".." in posix_p.parts:
        raise ValueError(f"非法的子目录前缀: {prefix}")
    return posix_p.as_posix()


def safe_extract_tar(tar_path: Path, dest_dir: Path) -> None:
    """安全解压 tar.gz 归档至本地目标目录，拦截越界路径与链接项并防同名覆写。"""
    dest_root = dest_dir.resolve()
    dest_root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tf:
        members = tf.getmembers()
        for member in members:
            member_path = (dest_root / member.name).resolve()
            if dest_root != member_path and dest_root not in member_path.parents:
                raise RuntimeError(f"归档中存在越界路径，拒绝解压: {member.name}")
            if member.issym() or member.islnk():
                raise RuntimeError(f"归档中存在危险链接项，拒绝解压: {member.name}")

        for member in members:
            target = (dest_root / member.name).resolve()
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            src_f = tf.extractfile(member)
            if src_f is None:
                continue
            tmp_fd, tmp_name = tempfile.mkstemp(prefix=".pull_tmp_", dir=str(target.parent))
            tmp_path = Path(tmp_name)
            try:
                h = hashlib.sha256()
                size = 0
                with os.fdopen(tmp_fd, "wb") as out_fp:
                    while True:
                        chunk = src_f.read(1024 * 1024)
                        if not chunk:
                            break
                        out_fp.write(chunk)
                        h.update(chunk)
                        size += len(chunk)
                final_dest, skip_write = resolve_non_clobber_dest(target, size, h.hexdigest())
                if skip_write:
                    tmp_path.unlink(missing_ok=True)
                else:
                    tmp_path.replace(final_dest)
                    if member.mtime:
                        try:
                            os.utime(final_dest, (member.mtime, member.mtime))
                        except OSError:
                            pass
            except Exception:
                tmp_path.unlink(missing_ok=True)
                raise
            finally:
                src_f.close()


def parse_http_mtime(last_modified: str | None) -> float | None:
    """解析 HTTP Last-Modified 响应头为 UNIX 秒，失败返回 None。"""
    if not last_modified:
        return None
    try:
        dt = parsedate_to_datetime(last_modified.strip())
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt.timestamp()
    except Exception:
        return None


class StreamingMultipartFile:
    """零全量内存缓存在途传输的流式 multipart/form-data 构造器。"""

    def __init__(
        self,
        file_path: Path,
        field_name: str = "image",
        fields: dict[str, str] | None = None,
        on_progress: Callable[[int], None] | None = None,
    ) -> None:
        self.file_path = file_path
        self.boundary = uuid.uuid4().hex
        self.content_type = f"multipart/form-data; boundary={self.boundary}"
        self._on_progress = on_progress
        self._last_report_ts = 0.0
        self._last_pct = -1

        head_parts: list[bytes] = []
        for k, v in (fields or {}).items():
            head_parts.append(
                f"--{self.boundary}\r\n"
                f'Content-Disposition: form-data; name="{k}"\r\n\r\n'
                f"{v}\r\n".encode()
            )
        ctype = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
        safe_name = sanitize_filename(file_path.name).replace('"', "_")
        head_parts.append(
            f"--{self.boundary}\r\n"
            f'Content-Disposition: form-data; name="{field_name}"; filename="{safe_name}"\r\n'
            f"Content-Type: {ctype}\r\n\r\n".encode()
        )
        self._head = b"".join(head_parts)
        self._tail = f"\r\n--{self.boundary}--\r\n".encode()
        self._file_size = file_path.stat().st_size
        self._total_len = len(self._head) + self._file_size + len(self._tail)
        self._fp = open(file_path, "rb")
        self._stage = 0
        self._offset = 0
        self._sent = 0

    def __len__(self) -> int:
        return self._total_len

    def read(self, size: int = -1) -> bytes:
        want = self._total_len if size is None or size < 0 else size
        if want == 0 or self._stage >= 3:
            return b""
        chunks: list[bytes] = []
        remaining = want
        while remaining > 0 and self._stage < 3:
            if self._stage == 0:
                piece = self._head[self._offset : self._offset + remaining]
                self._offset += len(piece)
                chunks.append(piece)
                remaining -= len(piece)
                if self._offset >= len(self._head):
                    self._stage = 1
                    self._offset = 0
            elif self._stage == 1:
                piece = self._fp.read(remaining)
                if piece:
                    chunks.append(piece)
                    remaining -= len(piece)
                else:
                    self._stage = 2
                    self._offset = 0
            elif self._stage == 2:
                piece = self._tail[self._offset : self._offset + remaining]
                self._offset += len(piece)
                chunks.append(piece)
                remaining -= len(piece)
                if self._offset >= len(self._tail):
                    self._stage = 3
        data = b"".join(chunks)
        self._sent += len(data)
        self._report()
        return data

    def _report(self) -> None:
        if not self._on_progress or self._total_len <= 0:
            return
        pct = min(100, int(round(self._sent * 100.0 / self._total_len)))
        now = time.monotonic()
        if pct != self._last_pct and (pct >= 100 or now - self._last_report_ts >= 0.12):
            self._last_pct = pct
            self._last_report_ts = now
            self._on_progress(pct)

    def close(self) -> None:
        with suppress_logged("file_sync", throttled_key="multipart_close_err", interval_sec=30.0):
            self._fp.close()

    def __enter__(self) -> StreamingMultipartFile:
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()


_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_MAX_PNG_CHUNK_BYTES = 16 * 1024 * 1024
_PNG_TEXT_CHUNK_TYPES = (b"tEXt", b"zTXt", b"iTXt")


def _safe_zlib_decompress(raw: bytes, max_bytes: int = _MAX_PNG_CHUNK_BYTES) -> bytes | None:
    """安全解压 PNG 压缩文本载荷，并限制最大输出体积以防异常膨胀。"""
    try:
        dec = zlib.decompressobj()
        out = dec.decompress(raw, max_bytes + 1)
        if len(out) > max_bytes or not dec.eof:
            return None
        return out
    except zlib.error:
        return None


def _decode_png_text(raw: bytes) -> str:
    """优先按 UTF-8 解码 PNG 文本块内容，遇到非 UTF-8 字节回退 latin-1。"""
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1", errors="replace")


def _read_png_text_chunks(path: Path) -> dict[str, str]:
    """使用标准库 struct 与 zlib 读取 PNG 文件的 tEXt / zTXt / iTXt 文本元数据块。"""
    chunks: dict[str, str] = {}
    try:
        with open(path, "rb") as fp:
            if fp.read(8) != _PNG_SIGNATURE:
                return chunks
            while True:
                header = fp.read(8)
                if len(header) != 8:
                    break
                length, chunk_type = struct.unpack(">I4s", header)
                if chunk_type in (b"IDAT", b"IEND"):
                    break
                if length > _MAX_PNG_CHUNK_BYTES or chunk_type not in _PNG_TEXT_CHUNK_TYPES:
                    fp.seek(length + 4, os.SEEK_CUR)
                    continue
                data = fp.read(length)
                crc = fp.read(4)
                if len(data) != length or len(crc) != 4:
                    break
                if b"\x00" not in data:
                    continue
                k_raw, rest = data.split(b"\x00", 1)
                key = k_raw.decode("latin-1", errors="replace")
                if not key:
                    continue
                if chunk_type == b"tEXt":
                    chunks[key] = _decode_png_text(rest)
                elif chunk_type == b"zTXt":
                    if not rest or rest[0] != 0:
                        continue
                    raw_val = _safe_zlib_decompress(rest[1:])
                    if raw_val is not None:
                        chunks[key] = _decode_png_text(raw_val)
                elif chunk_type == b"iTXt":
                    if len(rest) < 2:
                        continue
                    comp_flag, comp_method, tail = rest[0], rest[1], rest[2:]
                    parts = tail.split(b"\x00", 2)
                    if len(parts) != 3:
                        continue
                    text_raw = parts[2]
                    if comp_flag == 1:
                        if comp_method != 0:
                            continue
                        decompressed = _safe_zlib_decompress(text_raw)
                        if decompressed is None:
                            continue
                        text_raw = decompressed
                    elif comp_flag != 0:
                        continue
                    chunks[key] = _decode_png_text(text_raw)
    except Exception:
        return chunks
    return chunks


def extract_png_comfy_workflow(path: Path) -> str | None:
    """检查 PNG 元数据中是否内嵌合法 ComfyUI `workflow` 或 API `prompt`。"""
    try:
        chunks = _read_png_text_chunks(path)
        raw_wf = chunks.get("workflow")
        raw_prompt = chunks.get("prompt")
    except Exception:
        return None

    if isinstance(raw_wf, str) and raw_wf.strip():
        try:
            parsed_wf = json.loads(raw_wf)
            if isinstance(parsed_wf, dict) and ("nodes" in parsed_wf or "links" in parsed_wf):
                return raw_wf
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            get_diag_logger().debug_throttled(
                "file_sync",
                "png_workflow_json_err",
                f"解析 PNG workflow 元数据失败: {exc}",
                interval_sec=30.0,
            )

    if isinstance(raw_prompt, str) and raw_prompt.strip():
        try:
            parsed_prompt = json.loads(raw_prompt)
            if isinstance(parsed_prompt, dict) and any(
                isinstance(node, dict)
                and isinstance(node.get("class_type"), str)
                and isinstance(node.get("inputs"), dict)
                for node in parsed_prompt.values()
            ):
                return raw_prompt
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            get_diag_logger().debug_throttled(
                "file_sync",
                "png_prompt_json_err",
                f"解析 PNG prompt 元数据失败: {exc}",
                interval_sec=30.0,
            )

    return None


def is_api_prompt_dict(data: Any) -> bool:
    """判断 JSON 对象是否为 ComfyUI API 节点拓扑格式（prompt）。"""
    if not isinstance(data, dict) or "nodes" in data or "links" in data:
        return False
    return any(
        isinstance(node, dict)
        and isinstance(node.get("class_type"), str)
        and bool(node.get("class_type", "").strip())
        and isinstance(node.get("inputs"), dict)
        for node in data.values()
    )


def _is_prompt_link_ref(val: Any, valid_nodes: dict[str, dict[str, Any]]) -> bool:
    return (
        isinstance(val, (list, tuple))
        and len(val) == 2
        and isinstance(val[0], (str, int))
        and not isinstance(val[0], bool)
        and isinstance(val[1], int)
        and not isinstance(val[1], bool)
        and val[1] >= 0
        and str(val[0]) in valid_nodes
    )


def _default_widget_value(raw_type: Any, cfg: dict[str, Any]) -> Any:
    if "default" in cfg:
        return cfg["default"]
    if isinstance(raw_type, list):
        return raw_type[0] if raw_type else ""
    if raw_type == "COMBO":
        opts = cfg.get("options")
        if isinstance(opts, list) and opts:
            return opts[0]
        return ""
    if raw_type == "INT":
        return 0
    if raw_type == "FLOAT":
        return 0.0
    if raw_type == "BOOLEAN":
        return False
    return ""


def _should_append_seed_control(inp_name: str, raw_type: Any, cfg: dict[str, Any]) -> bool:
    if cfg.get("control_after_generate") is True:
        return True
    return raw_type == "INT" and inp_name in ("seed", "noise_seed") and cfg.get("control_after_generate") is not False


def _ordered_node_input_specs(node_info: dict[str, Any]) -> list[tuple[str, Any, dict[str, Any]]]:
    input_sec = node_info.get("input") or {}
    if not isinstance(input_sec, dict):
        return []
    req_map = input_sec.get("required") if isinstance(input_sec.get("required"), dict) else {}
    opt_map = input_sec.get("optional") if isinstance(input_sec.get("optional"), dict) else {}
    order_sec = node_info.get("input_order") if isinstance(node_info.get("input_order"), dict) else {}

    specs: list[tuple[str, Any, dict[str, Any]]] = []
    for section_name, section_map in (("required", req_map), ("optional", opt_map)):
        raw_order = order_sec.get(section_name)
        keys = list(raw_order) if isinstance(raw_order, list) else list(section_map.keys())
        seen = set(keys)
        for k in section_map:
            if k not in seen:
                keys.append(k)
        for inp_name in keys:
            spec = section_map.get(inp_name)
            if not isinstance(spec, (list, tuple)) or not spec:
                continue
            raw_type = spec[0]
            cfg = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
            specs.append((inp_name, raw_type, cfg))
    return specs


def ensure_workflow_uuid(wf_dict: dict[str, Any]) -> bool:
    """确保 UI 工作流顶层具备合法 UUID。"""
    raw_id = wf_dict.get("id")
    if isinstance(raw_id, str) and _WORKFLOW_UUID_RE.match(raw_id.strip()):
        return False
    wf_dict["id"] = str(uuid.uuid4())
    return True


def _collect_valid_prompt_nodes(
    prompt_dict: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    """筛选合法 ComfyUI API 节点并记录其原始声明顺序。"""
    valid_nodes: dict[str, dict[str, Any]] = {}
    orig_order: dict[str, int] = {}
    for idx, (raw_key, node_val) in enumerate(prompt_dict.items()):
        if (
            isinstance(node_val, dict)
            and isinstance(node_val.get("class_type"), str)
            and node_val.get("class_type", "").strip()
            and isinstance(node_val.get("inputs"), dict)
        ):
            key_str = str(raw_key)
            valid_nodes[key_str] = node_val
            orig_order[key_str] = idx
    return valid_nodes, orig_order


def _compute_prompt_node_depths(
    valid_nodes: dict[str, dict[str, Any]],
    ordered_keys: list[str],
) -> dict[str, int]:
    """根据节点间连线依赖迭代计算每个节点的拓扑深度（列层级）。"""
    depths: dict[str, int] = {k: 0 for k in ordered_keys}
    for _ in range(len(ordered_keys)):
        changed = False
        for k in ordered_keys:
            best = depths[k]
            for val in valid_nodes[k]["inputs"].values():
                if _is_prompt_link_ref(val, valid_nodes):
                    cand = depths[str(val[0])] + 1
                    if cand > best:
                        best = cand
            if best > depths[k]:
                depths[k] = best
                changed = True
        if not changed:
            break
    return depths


def _default_ui_node_size(class_type: str) -> list[int]:
    """根据 ComfyUI 节点类名返回画布默认显示宽高。"""
    if "TextEncode" in class_type:
        return [400, 260]
    if "KSampler" in class_type:
        return [330, 290]
    if class_type in ("SaveImage", "PreviewImage"):
        return [330, 270]
    return [330, 150]


def _build_ui_node_slots_with_info(
    node_key: str,
    node_inputs_data: dict[str, Any],
    node_info: dict[str, Any],
    valid_nodes: dict[str, dict[str, Any]],
    pending_links: list[tuple[str, int, str, int, str]],
) -> tuple[list[dict[str, Any]], list[Any], list[dict[str, Any]]]:
    """在具备 `object_info` 节点元数据时，构造输入槽、控件值与输出槽定义。"""
    ui_inputs: list[dict[str, Any]] = []
    widgets_values: list[Any] = []
    specs = _ordered_node_input_specs(node_info)
    known_names = {name for name, _, _ in specs}

    for inp_name, raw_type, cfg in specs:
        is_combo = isinstance(raw_type, list) or raw_type == "COMBO"
        slot_type = "COMBO" if is_combo else (str(raw_type) if isinstance(raw_type, str) else "*")
        is_widget = (not bool(cfg.get("forceInput"))) and (
            is_combo or raw_type in ("INT", "FLOAT", "STRING", "BOOLEAN")
        )
        has_val = inp_name in node_inputs_data
        raw_val = node_inputs_data.get(inp_name)
        is_linked = has_val and _is_prompt_link_ref(raw_val, valid_nodes)

        if is_widget:
            if is_linked:
                target_slot = len(ui_inputs)
                ui_inputs.append({"name": inp_name, "type": slot_type, "widget": {"name": inp_name}, "link": None})
                pending_links.append((str(raw_val[0]), int(raw_val[1]), node_key, target_slot, slot_type))
                widgets_values.append(_default_widget_value(raw_type, cfg))
            else:
                widgets_values.append(raw_val if has_val else _default_widget_value(raw_type, cfg))
            if _should_append_seed_control(inp_name, raw_type, cfg):
                widgets_values.append("fixed")
        else:
            target_slot = len(ui_inputs)
            ui_inputs.append({"name": inp_name, "type": slot_type, "link": None})
            if is_linked:
                pending_links.append((str(raw_val[0]), int(raw_val[1]), node_key, target_slot, slot_type))

    for extra_name, extra_val in node_inputs_data.items():
        if extra_name not in known_names and _is_prompt_link_ref(extra_val, valid_nodes):
            target_slot = len(ui_inputs)
            ui_inputs.append({"name": extra_name, "type": "*", "link": None})
            pending_links.append((str(extra_val[0]), int(extra_val[1]), node_key, target_slot, "*"))

    out_types = node_info.get("output") if isinstance(node_info.get("output"), list) else []
    out_names = node_info.get("output_name") if isinstance(node_info.get("output_name"), list) else out_types
    ui_outputs = [
        {
            "name": str(out_names[idx] if idx < len(out_names) else out_t),
            "type": str(out_t),
            "links": [],
            "slot_index": idx,
        }
        for idx, out_t in enumerate(out_types)
    ]
    return ui_inputs, widgets_values, ui_outputs


def _build_ui_node_slots_fallback(
    node_key: str,
    node_inputs_data: dict[str, Any],
    valid_nodes: dict[str, dict[str, Any]],
    pending_links: list[tuple[str, int, str, int, str]],
) -> tuple[list[dict[str, Any]], list[Any], list[dict[str, Any]]]:
    """在缺少 `object_info` 元数据时按输入值形态兜底构造输入槽与控件列表。"""
    ui_inputs: list[dict[str, Any]] = []
    widgets_values: list[Any] = []
    for inp_name, raw_val in node_inputs_data.items():
        if _is_prompt_link_ref(raw_val, valid_nodes):
            target_slot = len(ui_inputs)
            ui_inputs.append({"name": inp_name, "type": "*", "link": None})
            pending_links.append((str(raw_val[0]), int(raw_val[1]), node_key, target_slot, "*"))
        else:
            widgets_values.append(raw_val)
            if inp_name in ("seed", "noise_seed") and isinstance(raw_val, int) and not isinstance(raw_val, bool):
                widgets_values.append("fixed")
    return ui_inputs, widgets_values, []


def _wire_ui_workflow_links(
    pending_links: list[tuple[str, int, str, int, str]],
    ui_nodes_by_key: dict[str, dict[str, Any]],
) -> list[list[Any]]:
    """根据待连线清单补齐缺失输出槽、回填槽位类型并生成 LiteGraph `links` 数组。"""
    ui_links: list[list[Any]] = []
    for link_idx, (src_key, src_slot, dst_key, dst_slot, target_slot_type) in enumerate(pending_links, start=1):
        src_node = ui_nodes_by_key[src_key]
        dst_node = ui_nodes_by_key[dst_key]
        while len(src_node["outputs"]) <= src_slot:
            idx = len(src_node["outputs"])
            inferred_t = target_slot_type if idx == src_slot else "*"
            src_node["outputs"].append({"name": inferred_t, "type": inferred_t, "links": [], "slot_index": idx})
        src_out = src_node["outputs"][src_slot]
        src_out["links"].append(link_idx)
        link_type = src_out.get("type", "*")
        if link_type == "*" and target_slot_type != "*":
            link_type = target_slot_type
            src_out["type"] = link_type
            if src_out.get("name") == "*":
                src_out["name"] = link_type
        if dst_node["inputs"][dst_slot]["type"] == "*" and link_type != "*":
            dst_node["inputs"][dst_slot]["type"] = link_type
        dst_node["inputs"][dst_slot]["link"] = link_idx
        ui_links.append([link_idx, src_node["id"], src_slot, dst_node["id"], dst_slot, link_type])
    return ui_links


def _build_single_ui_node(
    *,
    node_key: str,
    node_int_id: int,
    order_idx: int,
    depth: int,
    row_idx: int,
    node_data: dict[str, Any],
    object_info: dict[str, Any] | None,
    valid_nodes: dict[str, dict[str, Any]],
    pending_links: list[tuple[str, int, str, int, str]],
) -> dict[str, Any]:
    """构造单个 LiteGraph UI 节点字典（自动按 `object_info` 可用性选择槽位构建器）。"""
    class_type = node_data["class_type"]
    node_info = (
        object_info.get(class_type)
        if isinstance(object_info, dict) and isinstance(object_info.get(class_type), dict)
        else None
    )
    if node_info is not None:
        ui_inputs, widgets_values, ui_outputs = _build_ui_node_slots_with_info(
            node_key, node_data["inputs"], node_info, valid_nodes, pending_links
        )
    else:
        ui_inputs, widgets_values, ui_outputs = _build_ui_node_slots_fallback(
            node_key, node_data["inputs"], valid_nodes, pending_links
        )
    return {
        "id": node_int_id,
        "type": class_type,
        "pos": [80 + depth * 420, 80 + row_idx * 320],
        "size": _default_ui_node_size(class_type),
        "flags": {},
        "order": order_idx,
        "mode": 0,
        "inputs": ui_inputs,
        "outputs": ui_outputs,
        "properties": {"Node name for S&R": class_type},
        "widgets_values": widgets_values,
    }


def api_prompt_to_ui_workflow(
    prompt_dict: dict[str, Any],
    object_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """将 ComfyUI API `prompt` 字典转换为标准网页画布 LiteGraph 工作流。"""
    valid_nodes, orig_order = _collect_valid_prompt_nodes(prompt_dict)
    ordered_keys = list(valid_nodes.keys())
    if not ordered_keys:
        return {
            "id": str(uuid.uuid4()),
            "last_node_id": 0,
            "last_link_id": 0,
            "nodes": [],
            "links": [],
            "groups": [],
            "config": {},
            "extra": {},
            "version": 0.4,
        }

    depths = _compute_prompt_node_depths(valid_nodes, ordered_keys)
    sorted_keys = sorted(ordered_keys, key=lambda k: (depths[k], orig_order[k]))
    str_to_int_id = {k: idx + 1 for idx, k in enumerate(sorted_keys)}

    depth_rows: dict[int, int] = {}
    ui_nodes_by_key: dict[str, dict[str, Any]] = {}
    ui_nodes: list[dict[str, Any]] = []
    pending_links: list[tuple[str, int, str, int, str]] = []

    for order_idx, k in enumerate(sorted_keys):
        depth = depths[k]
        row_idx = depth_rows.get(depth, 0)
        depth_rows[depth] = row_idx + 1
        ui_node = _build_single_ui_node(
            node_key=k,
            node_int_id=str_to_int_id[k],
            order_idx=order_idx,
            depth=depth,
            row_idx=row_idx,
            node_data=valid_nodes[k],
            object_info=object_info,
            valid_nodes=valid_nodes,
            pending_links=pending_links,
        )
        ui_nodes_by_key[k] = ui_node
        ui_nodes.append(ui_node)

    ui_links = _wire_ui_workflow_links(pending_links, ui_nodes_by_key)
    return {
        "id": str(uuid.uuid4()),
        "last_node_id": len(ui_nodes),
        "last_link_id": len(ui_links),
        "nodes": ui_nodes,
        "links": ui_links,
        "groups": [],
        "config": {},
        "extra": {},
        "version": 0.4,
    }


def sync_session_outputs_cli(
    *,
    session: str,
    prefix: str | None,
    pull: bool,
    colab_exe: Path,
    local_download_dir: Path,
    run_fn: Callable[..., Any],
    exec_remote_py_fn: Callable[..., Any],
    redact_fn: Callable[[str], str],
) -> None:
    """将远端输出目录以确定性时间戳归档同步至 Google Drive 并强校验确认标记。"""
    norm_prefix = normalize_prefix(prefix)
    remote_src = f"{REMOTE_OUT_ROOT}/{norm_prefix}" if norm_prefix else REMOTE_OUT_ROOT
    gdrive_dst = f"{GDRIVE_OUT_REMOTE}/{norm_prefix}" if norm_prefix else GDRIVE_OUT_REMOTE

    code = (
        REMOTE_SYNC_SCRIPT
        .replace("__REMOTE_SRC__", repr(remote_src))
        .replace("__GDRIVE_DST__", repr(gdrive_dst))
        .replace("__REMOTE_RCLONE_CONF__", repr(REMOTE_RCLONE_CONF))
        .replace("__REMOTE_PULL_TAR__", repr(REMOTE_PULL_TAR))
        .replace("__DO_PULL__", repr(bool(pull)))
    )
    res = exec_remote_py_fn(session, code, timeout=600, capture=True)
    out_text = redact_fn(getattr(res, "stdout", "") or "")
    if out_text:
        print(out_text, end="" if out_text.endswith("\n") else "\n", flush=True)
    if "SKIP_LLM_SYNC" in out_text:
        return
    if "<<<SYNC_VERIFIED>>>" not in out_text:
        raise RuntimeError("云盘归档未通过完整性校验（缺少确认标记）")

    if pull:
        local_target = local_download_dir / (norm_prefix if norm_prefix else "all")
        with tempfile.TemporaryDirectory(prefix="colab_pull_") as tmp_dir:
            local_tar = Path(tmp_dir) / "comfy_pull.tar.gz"
            run_fn([colab_exe, "download", "-s", session, REMOTE_PULL_TAR, local_tar])
            safe_extract_tar(local_tar, local_target)
        print(f"已拉回并解压至本机目录: {local_target}", flush=True)


def _rclone_mkdir_and_lsl(rclone_exe: Path, rclone_conf: Path, remote_dir: str, run_fn: Callable[..., Any]) -> dict[str, int]:
    """确保远端云盘目录存在并返回 `{相对路径或文件名: 字节数}` 字典。"""
    base_flags = ["--config", rclone_conf, "--disable-http2", "--retries", "6", "--retries-sleep", "5s"]
    run_fn([rclone_exe, "mkdir", *base_flags, remote_dir])
    res = run_fn([rclone_exe, "lsl", *base_flags, remote_dir], capture=True)
    sizes: dict[str, int] = {}
    for line in (getattr(res, "stdout", "") or "").splitlines():
        parts = line.strip().split(None, 3)
        if len(parts) == 4 and parts[0].isdigit():
            sizes[parts[3]] = int(parts[0])
    return sizes


def sync_anima_loras_cli(
    *,
    rclone_exe: Path,
    rclone_conf: Path,
    lora_backup_dir: Path,
    run_fn: Callable[..., Any],
) -> None:
    """零 GPU 消耗：核对并上传 7 个生产级 Anima LoRA 至存储号 Google Drive。"""
    if not rclone_exe.is_file():
        raise FileNotFoundError(f"未找到rclone.exe: {rclone_exe}")
    if not rclone_conf.is_file():
        raise FileNotFoundError(f"未找到rclone.conf: {rclone_conf}")

    print("=== [1/4] 校验本地 Anima LoRA 完整性 ===", flush=True)
    verified = []
    if ANIMA_LORAS:
        for rel_path, expected_size, expected_sha in ANIMA_LORAS:
            src = lora_backup_dir / rel_path
            if not src.is_file():
                raise FileNotFoundError(f"本地 LoRA 缺失: {src}")
            actual_size = src.stat().st_size
            if expected_size is not None and actual_size != expected_size:
                raise ValueError(f"LoRA 字节数不符: {src.name} 实际={actual_size} B, 预期={expected_size} B")
            actual_sha = sha256_file(src)
            if expected_sha and actual_sha != expected_sha:
                raise ValueError(f"LoRA SHA256 不符: {src.name} 实际={actual_sha}, 预期={expected_sha}")
            verified.append((src, actual_size, actual_sha))
            print(f"  [OK] {src.name} ({actual_size} B)", flush=True)
    else:
        candidates = sorted(p for p in lora_backup_dir.rglob("*.safetensors") if p.is_file())
        if not candidates:
            raise FileNotFoundError(f"未在目录中找到任何 .safetensors LoRA 文件: {lora_backup_dir}（可通过 --src 或 COLAB_LORA_DIR 指定）")
        for src in candidates:
            actual_size = src.stat().st_size
            actual_sha = sha256_file(src)
            verified.append((src, actual_size, actual_sha))
            print(f"  [OK] {src.name} ({actual_size} B)", flush=True)

    print("=== [2/4] 预检远端目录并增量并发上传 LoRA (HTTP/1.1 + 8M 分块 + 抗限流退避) ===", flush=True)
    existing_sizes = _rclone_mkdir_and_lsl(rclone_exe, rclone_conf, GDRIVE_LORA_REMOTE, run_fn)
    todo = [(src, sz, sha) for src, sz, sha in verified if existing_sizes.get(src.name) != sz]
    for src, sz, _ in verified:
        if existing_sizes.get(src.name) == sz:
            print(f"  [远端已就绪，跳过] {src.name} ({sz} B)", flush=True)

    def _upload_one(src: Path) -> None:
        t0 = time.time()
        cmd = [
            str(rclone_exe), "copyto", "--config", str(rclone_conf), "--disable-http2",
            "--drive-chunk-size", "8M", "--timeout", "5m", "--contimeout", "30s",
            "--low-level-retries", "10", "--retries", "10", "--retries-sleep", "6s",
            "-v", "--stats-one-line", "--stats", "5s",
            str(src), f"{GDRIVE_LORA_REMOTE}/{src.name}",
        ]
        print(f"  [开始上传] {src.name}", flush=True)
        run_fn(cmd, check=True)
        print(f"  [上传完成] {src.name} (耗时 {time.time() - t0:.1f}s)", flush=True)

    if todo:
        with ThreadPoolExecutor(max_workers=2) as pool:
            for fut in as_completed([pool.submit(_upload_one, src) for src, _, _ in todo]):
                fut.result()
    else:
        print("  所有 7 个 LoRA 均已存在于远端，无需重复上传。", flush=True)

    print("=== [3/4] 生成并上传 sha256sums.txt ===", flush=True)
    with tempfile.TemporaryDirectory(prefix="lora_sums_") as tmp_dir:
        sums_file = Path(tmp_dir) / "sha256sums.txt"
        sums_file.write_text("\n".join(f"{sha}  {src.name}" for src, _, sha in verified) + "\n", encoding="utf-8")
        run_fn([rclone_exe, "copyto", "--config", rclone_conf, "--disable-http2", "--retries", "6", "--retries-sleep", "5s", sums_file, f"{GDRIVE_LORA_REMOTE}/sha256sums.txt"])

    print("=== [4/4] 核对 Google Drive 远端文件清单与字节数 ===", flush=True)
    remote_sizes = _rclone_mkdir_and_lsl(rclone_exe, rclone_conf, GDRIVE_LORA_REMOTE, run_fn)
    for src, expected_size, _ in verified:
        if src.name not in remote_sizes:
            raise RuntimeError(f"远端缺少已上传 LoRA: {src.name}")
        if remote_sizes[src.name] != expected_size:
            raise RuntimeError(f"远端字节数不一致: {src.name} 远端={remote_sizes[src.name]}, 预期={expected_size}")
    if remote_sizes.get("sha256sums.txt", 0) <= 0:
        raise RuntimeError("远端缺少有效的 sha256sums.txt")
    print(f"LoRA 上传并核验完成！共 {len(verified)} 个权重（总计 {sum(sz for _, sz, _ in verified) / (1024 ** 2):.2f} MiB）。", flush=True)


def sync_drama3_refs_cli(
    *,
    src_dir: Path,
    rclone_exe: Path,
    rclone_conf: Path,
    run_fn: Callable[..., Any],
) -> None:
    """零 GPU 消耗：将本地精选 Drama 3 参考 .wav 上传至存储号 Google Drive（防子目录重名覆写）。"""
    if not rclone_exe.is_file():
        raise FileNotFoundError(f"未找到rclone.exe: {rclone_exe}")
    if not rclone_conf.is_file():
        raise FileNotFoundError(f"未找到rclone.conf: {rclone_conf}")
    if not src_dir.is_dir():
        raise FileNotFoundError(f"未找到 Drama 3 参考音频目录: {src_dir}")

    wavs = sorted(src_dir.rglob("*.wav"))
    if not wavs:
        raise RuntimeError(f"目录下未找到任何 .wav 文件: {src_dir}")

    # 保持唯一相对路径避免跨子目录同名音频相互覆写
    has_dup_name = len({w.name for w in wavs}) != len(wavs)
    paired = [(w, w.relative_to(src_dir).as_posix() if has_dup_name else w.name) for w in wavs]

    print(f"=== [1/3] 发现 {len(wavs)} 条精选参考音频，准备预检云盘目录 ===", flush=True)
    existing_sizes = _rclone_mkdir_and_lsl(rclone_exe, rclone_conf, GDRIVE_DRAMA3_REFS_REMOTE, run_fn)
    todo = []
    for wf, rel_key in paired:
        sz = wf.stat().st_size
        if existing_sizes.get(rel_key) == sz:
            print(f"  [远端已就绪，跳过] {rel_key} ({sz} B)", flush=True)
        else:
            todo.append((wf, rel_key, sz))

    print(f"=== [2/3] 增量上传 Drama 3 参考音频（待传 {len(todo)}/{len(wavs)} 条）===", flush=True)
    for wf, rel_key, _ in todo:
        run_fn([rclone_exe, "copyto", "--config", rclone_conf, "--disable-http2", "--retries", "8", "--retries-sleep", "5s", wf, f"{GDRIVE_DRAMA3_REFS_REMOTE}/{rel_key}"])

    print("=== [3/3] 核对远端字节完整性 ===", flush=True)
    remote_sizes = _rclone_mkdir_and_lsl(rclone_exe, rclone_conf, GDRIVE_DRAMA3_REFS_REMOTE, run_fn)
    for wf, rel_key, _ in paired:
        if remote_sizes.get(rel_key) != wf.stat().st_size:
            raise RuntimeError(f"云盘参考音频校验失败: {rel_key}")
    print(f"Drama 3 参考音频上传并校验完成！共 {len(wavs)} 条（总计 {sum(w.stat().st_size for w in wavs) / (1024 ** 2):.2f} MiB）。", flush=True)
