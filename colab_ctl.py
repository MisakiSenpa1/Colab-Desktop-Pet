"""colab_ctl.py —— Google Colab ComfyUI 工作流本机总控脚手架

用途：
    在本机 Windows 环境下通过单一 Python CLI 统一调度 `google-colab-cli` 与
    `rclone`，完成 LoRA 预上传、Colab GPU 实例部署、WebUI 隧道检测与重启、
    冒烟验证、出图脚本投递、队列完成守护等待、Google Drive 同步、结果拉回本机
    以及自动停卡省积分。

常用示例：
    python colab_ctl.py sync-loras
    python colab_ctl.py status
    python colab_ctl.py up --gpu L4
    python colab_ctl.py tunnel
    python colab_ctl.py smoke
    python colab_ctl.py run my_workflow_runner.py --prefix batch1 --pull --auto-stop
    python colab_ctl.py down --all
"""
import argparse
import ast
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath

def _first_existing_path(candidates: list[Path], fallback: Path) -> Path:
    """按优先级返回第一个存在的本地文件，均缺失时回退至默认路径。"""
    for cand in candidates:
        if cand.is_file():
            return cand
    return fallback


def _resolve_colab_exe() -> Path:
    """定位 colab 可执行文件（支持环境变量 COLAB_EXE 覆盖）。"""
    env_p = (os.environ.get("COLAB_EXE") or "").strip()
    if env_p:
        return Path(env_p)
    which_p = shutil.which("colab")
    default_p = Path.home() / ".local" / "bin" / "colab.exe"
    return _first_existing_path(([Path(which_p)] if which_p else []) + [default_p], default_p)


def _resolve_colab_py() -> Path:
    """定位 google-colab-cli 托管 Python 解释器（支持环境变量 COLAB_PY 覆盖）。"""
    env_p = (os.environ.get("COLAB_PY") or "").strip()
    if env_p:
        return Path(env_p)
    appdata = Path(os.environ.get("APPDATA") or (Path.home() / "AppData" / "Roaming"))
    uv_py = appdata / "uv" / "tools" / "google-colab-cli" / "Scripts" / "python.exe"
    return _first_existing_path(
        [uv_py, Path.home() / ".local" / "share" / "uv" / "tools" / "google-colab-cli" / "Scripts" / "python.exe"],
        uv_py,
    )


def _resolve_rclone_exe() -> Path:
    """定位 rclone 可执行文件（可选，支持环境变量 RCLONE_EXE 覆盖）。"""
    env_p = (os.environ.get("RCLONE_EXE") or "").strip()
    if env_p:
        return Path(env_p)
    which_p = shutil.which("rclone")
    scoop_p = Path.home() / "scoop" / "shims" / "rclone.exe"
    return _first_existing_path(([Path(which_p)] if which_p else []) + [scoop_p], Path(which_p) if which_p else scoop_p)


def _resolve_rclone_conf() -> Path:
    """定位 rclone 配置文件（可选，支持环境变量 RCLONE_CONF 覆盖）。"""
    env_p = (os.environ.get("RCLONE_CONF") or "").strip()
    if env_p:
        return Path(env_p)
    appdata = Path(os.environ.get("APPDATA") or (Path.home() / "AppData" / "Roaming"))
    default_p = appdata / "rclone" / "rclone.conf"
    return _first_existing_path(
        [
            default_p,
            Path.home() / ".config" / "rclone" / "rclone.conf",
            Path.home() / "scoop" / "apps" / "rclone" / "current" / "rclone.conf",
        ],
        default_p,
    )


# 本机工具与目录配置（均支持环境变量覆盖，避免硬编码用户路径）
COLAB_EXE = _resolve_colab_exe()
COLAB_PY = _resolve_colab_py()
RCLONE_EXE = _resolve_rclone_exe()
RCLONE_CONF = _resolve_rclone_conf()

PACKAGE_DIR = Path(__file__).resolve().parent
LORA_BACKUP_DIR = Path(os.environ.get("COLAB_LORA_DIR") or (PACKAGE_DIR / "loras"))
LORA_MANIFEST_FILE = PACKAGE_DIR / "loras_manifest.json"
LOCAL_DOWNLOAD_DIR = Path(os.environ.get("COLAB_DOWNLOAD_DIR") or (PACKAGE_DIR / "_downloads"))
BOOTSTRAP_SCRIPT = PACKAGE_DIR / "colab_bootstrap_anima.py"
SMOKE_SCRIPT = PACKAGE_DIR / "verify_smoke.py"
MINIMAX_BOOTSTRAP_SCRIPT = PACKAGE_DIR / "colab_bootstrap_minimax_h3.py"
MINIMAX_SMOKE_SCRIPT = PACKAGE_DIR / "verify_minimax_h3.py"
QWEN_LLM_BOOTSTRAP_SCRIPT = PACKAGE_DIR / "colab_bootstrap_qwen_llm.py"
QWEN_LLM_SMOKE_SCRIPT = PACKAGE_DIR / "verify_qwen_llm.py"
DRAMA3_BOOTSTRAP_SCRIPT = PACKAGE_DIR / "colab_bootstrap_drama3.py"
DRAMA3_SMOKE_SCRIPT = PACKAGE_DIR / "verify_drama3.py"
DRAMA3_REFS_LOCAL_DIR = Path(os.environ.get("COLAB_DRAMA3_REFS_DIR") or (PACKAGE_DIR / "drama3_refs"))
CF_TUNNEL_SECRET_FILE = PACKAGE_DIR / ".cf_tunnel_secret.json"


def _load_cf_tunnel_secret() -> dict:
    """安全读取本地 Cloudflare 永久命名隧道与固定密钥配置，缺失或损坏时返回空字典。"""
    if not CF_TUNNEL_SECRET_FILE.is_file():
        return {}
    try:
        data = json.loads(CF_TUNNEL_SECRET_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


_RCLONE_REMOTE = (os.environ.get("COLAB_RCLONE_REMOTE") or "gdrive_save").strip().rstrip(":") or "gdrive_save"
GDRIVE_LORA_REMOTE = f"{_RCLONE_REMOTE}:Colab_ComfyUI_Models/loras"
GDRIVE_DRAMA3_REFS_REMOTE = f"{_RCLONE_REMOTE}:Colab_ComfyUI_Models/drama3/refs"
GDRIVE_OUT_REMOTE = f"{_RCLONE_REMOTE}:Colab_ComfyUI_Output"
REMOTE_OUT_ROOT = "/root/comfy_out"
REMOTE_RCLONE_CONF = "/tmp/rclone.conf"
REMOTE_PULL_TAR = "/tmp/comfy_pull.tar.gz"


def _discover_local_loras(lora_dir: Path) -> list[tuple[str, int | None, str | None]]:
    """加载可选 loras_manifest.json 清单，否则自动扫描目录下的 *.safetensors 文件。"""
    if LORA_MANIFEST_FILE.is_file():
        try:
            raw = json.loads(LORA_MANIFEST_FILE.read_text(encoding="utf-8"))
            items = []
            for entry in (raw if isinstance(raw, list) else []):
                if isinstance(entry, dict) and entry.get("path"):
                    sz = int(entry["size"]) if entry.get("size") is not None else None
                    sha = str(entry["sha256"]).strip() if entry.get("sha256") else None
                    items.append((str(entry["path"]), sz, sha))
            if items:
                return items
        except Exception:
            pass
    if not lora_dir.is_dir():
        return []
    return [
        (p.relative_to(lora_dir).as_posix(), None, None)
        for p in sorted(lora_dir.rglob("*.safetensors"))
        if p.is_file()
    ]


def _ensure_bin(path: Path, desc: str):
    """检查关键本机可执行文件或配置文件是否存在。"""
    if not path.is_file():
        raise FileNotFoundError(f"未找到{desc}: {path}")


def _run(cmd, check=True, capture=False):
    """统一执行本机外部命令并打印命令行摘要（强制 UTF-8 编码防止 Windows 解码异常）。"""
    cmd_str = " ".join(str(x) for x in cmd)
    print(f"[本机执行] {cmd_str}", flush=True)
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run(
        [str(x) for x in cmd],
        check=check,
        text=True,
        capture_output=capture,
        encoding="utf-8",
        errors="replace",
        env=env,
    )


def _sha256_file(path: Path) -> str:
    """分块计算文件 SHA256 摘要，避免大文件一次性读入内存。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(4 * 1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


_REMOTE_RUNNER_TEMPLATE = r"""
import json
import os
import sys
import time
import traceback
from pathlib import Path

from colab_cli.common import state
from colab_cli.runtime import ColabRuntime
from colab_cli.utils import is_terminal_error

SESSION_NAME = __SESSION_NAME__
TARGET_SCRIPT = __TARGET_SCRIPT__
EXEC_TIMEOUT = __EXEC_TIMEOUT__
MAX_RETRIES = __MAX_RETRIES__
REMOTE_FILE = __REMOTE_FILE__
EXTRA_FILES = __EXTRA_FILES__
MANIFEST_OUT_DIR = __MANIFEST_OUT_DIR__

code = Path(TARGET_SCRIPT).read_text(encoding="utf-8")
exit_code = 1

for attempt in range(1, MAX_RETRIES + 1):
    s = state.get_session(SESSION_NAME)
    if s is None:
        print(f"错误: 未找到有效 Colab 会话 '{SESSION_NAME}'。", file=sys.stderr, flush=True)
        exit_code = 1
        break

    def on_started(kid):
        s.kernel_id = kid
        state.store.add(s)

    def on_sess_started(sid):
        s.session_id = sid
        state.store.add(s)

    rt = ColabRuntime(
        s.url,
        s.token,
        session_name=SESSION_NAME,
        kernel_id=s.kernel_id,
        session_id=s.session_id,
        on_kernel_started=on_started,
        on_session_started=on_sess_started,
    )
    printed_error = False
    try:
        ws_client = rt.kernel_client._manager.client
        if not ws_client.connection_ready.is_set():
            if not ws_client.connection_ready.wait(timeout=30.0):
                raise RuntimeError("WebSocket 连接握手超时 (30s)")

        prep_parts = [
            "import os, sys, shutil",
            "from pathlib import Path",
            "os.makedirs('/content', exist_ok=True)",
            "os.chdir('/content')",
            "if '/content' not in sys.path: sys.path.insert(0, '/content')",
        ]
        if EXTRA_FILES:
            prep_parts.append(f"_colab_extra_files = {json.dumps(EXTRA_FILES, ensure_ascii=False)}")
            prep_parts.append(
                "for _fn, _ft in _colab_extra_files.items():\n"
                "    Path('/content', Path(_fn).name).write_text(_ft, encoding='utf-8')"
            )
        if REMOTE_FILE:
            prep_parts.append(f"__file__ = {REMOTE_FILE!r}")
            prep_parts.append(f"sys.argv = [{REMOTE_FILE!r}]")
            prep_parts.append(f"Path({REMOTE_FILE!r}).write_text({code!r}, encoding='utf-8')")
        else:
            prep_parts.append("globals().pop('__file__', None)")
            prep_parts.append("sys.argv = ['']")
        if MANIFEST_OUT_DIR:
            prep_parts.append(
                "_colab_manifest_before = {p.name: p.stat().st_mtime_ns "
                "for p in Path('/content').glob('_manifest_*.json') if p.is_file()}"
            )
        rt.execute_code("\n".join(prep_parts), timeout=60.0)

        def _on_out(o):
            global printed_error
            otype = o.get("output_type")
            if otype == "stream":
                dst = sys.stderr if o.get("name") == "stderr" else sys.stdout
                dst.write(o.get("text", ""))
                dst.flush()
            elif otype in ("execute_result", "display_data"):
                txt = (o.get("data") or {}).get("text/plain")
                if txt is not None:
                    print(txt, flush=True)
            elif otype == "error":
                printed_error = True
                tb = o.get("traceback") or [f"{o.get('ename', 'Error')}: {o.get('evalue', '')}"]
                sys.stderr.write("\n".join(tb) + "\n")
                sys.stderr.flush()

        outputs = rt.execute_code(code, output_hook=_on_out, timeout=float(EXEC_TIMEOUT))
        if MANIFEST_OUT_DIR:
            try:
                collect_code = (
                    f"_mout = Path({MANIFEST_OUT_DIR!r})\n"
                    "_mout.mkdir(parents=True, exist_ok=True)\n"
                    "_mb = globals().get('_colab_manifest_before') or {}\n"
                    "for _mp in Path('/content').glob('_manifest_*.json'):\n"
                    "    if _mp.is_file() and _mp.stat().st_mtime_ns != _mb.get(_mp.name):\n"
                    "        shutil.copy2(_mp, _mout / _mp.name)\n"
                )
                rt.execute_code(collect_code, timeout=30.0)
            except Exception:
                pass
        err_outputs = [o for o in outputs if o.get("output_type") == "error"]
        if err_outputs:
            if not printed_error:
                for eo in err_outputs:
                    tb = eo.get("traceback") or [f"{eo.get('ename', 'Error')}: {eo.get('evalue', '')}"]
                    sys.stderr.write("\n".join(tb) + "\n")
                sys.stderr.flush()
            exit_code = 1
            break

        exit_code = 0
        break
    except Exception as exc:
        err_msg = str(exc)
        is_conn_err = (
            not is_terminal_error(exc)
            and (
                "Connection was lost" in err_msg
                or "WebSocket" in err_msg
                or "握手超时" in err_msg
                or "connection" in err_msg.lower()
                or isinstance(exc, (TimeoutError, ConnectionError, OSError))
            )
        )
        if is_conn_err and attempt < MAX_RETRIES:
            wait_s = 3 * attempt
            print(
                f"[远端重连] 第 {attempt}/{MAX_RETRIES} 次连接异常 ({exc})，{wait_s} 秒后重试...",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(wait_s)
            continue
        traceback.print_exc()
        exit_code = 1
        break
    finally:
        try:
            rt.stop()
        except Exception:
            pass

sys.stdout.flush()
sys.stderr.flush()
os._exit(exit_code)
"""


def _collect_local_py_deps(
    entry_script: Path,
    max_files: int = 16,
    max_total_bytes: int = 512 * 1024,
) -> dict[str, str]:
    """扫描本地入口脚本引用的同目录 .py 依赖文件，便于在远端 /content 一并预置。"""
    base_dir = entry_script.resolve().parent
    deps: dict[str, str] = {}
    visited: set[Path] = {entry_script.resolve()}
    queue: list[Path] = [entry_script.resolve()]
    total_bytes = 0

    def _enqueue_candidate(cand: Path) -> None:
        resolved = cand.resolve()
        if (
            resolved not in visited
            and resolved.parent == base_dir
            and resolved.is_file()
            and resolved.suffix == ".py"
        ):
            visited.add(resolved)
            queue.append(resolved)

    while queue and len(deps) < max_files:
        cur = queue.pop(0)
        try:
            src = cur.read_text(encoding="utf-8")
        except Exception:
            continue
        if cur != entry_script.resolve():
            raw_len = len(src.encode("utf-8"))
            if total_bytes + raw_len > max_total_bytes:
                print(f"[警告] 跳过同级依赖 {cur.name}: 累计体积超出 {max_total_bytes} B 限制", flush=True)
                continue
            deps[cur.name] = src
            total_bytes += raw_len

        try:
            tree = ast.parse(src, filename=str(cur))
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".", 1)[0]
                    _enqueue_candidate(base_dir / f"{top}.py")
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    top = node.module.split(".", 1)[0]
                    _enqueue_candidate(base_dir / f"{top}.py")
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                val = node.value.strip()
                if val.endswith(".py") and "/" not in val and "\\" not in val and len(val) <= 128:
                    _enqueue_candidate(base_dir / val)

    return deps


def _colab_exec_file(
    session: str,
    script_path: Path,
    timeout: int = 600,
    capture: bool = False,
    retries: int = 3,
    remote_file: str | None = None,
    extra_files: dict[str, str] | None = None,
    manifest_out_dir: str | None = None,
):
    """通过 colab_cli 虚拟环境稳健执行远端 Python 脚本，延长 WSS 握手等待并支持自动重连。"""
    _ensure_bin(COLAB_PY, "google-colab-cli python")
    _ensure_bin(script_path, f"待执行脚本 {script_path.name}")
    runner_code = (
        _REMOTE_RUNNER_TEMPLATE
        .replace("__SESSION_NAME__", repr(session))
        .replace("__TARGET_SCRIPT__", repr(str(script_path)))
        .replace("__EXEC_TIMEOUT__", repr(int(timeout)))
        .replace("__MAX_RETRIES__", repr(int(retries)))
        .replace("__REMOTE_FILE__", repr(remote_file))
        .replace("__EXTRA_FILES__", repr(dict(extra_files or {})))
        .replace("__MANIFEST_OUT_DIR__", repr(manifest_out_dir))
    )
    print(f"[远端执行] session={session} | script={script_path.name} (timeout={timeout}s)", flush=True)
    with tempfile.TemporaryDirectory(prefix="colab_runner_") as tmp_dir:
        runner_path = Path(tmp_dir) / "runner.py"
        runner_path.write_text(runner_code, encoding="utf-8")
        return _run([COLAB_PY, runner_path], check=True, capture=capture)


def _exec_remote_py(session: str, code: str, timeout: int = 300, capture: bool = False):
    """将 Python 片段写入 UTF-8 临时文件后通过 _colab_exec_file 在远端稳健运行。"""
    with tempfile.TemporaryDirectory(prefix="colab_remote_") as tmp_dir:
        script_path = Path(tmp_dir) / "remote_task.py"
        script_path.write_text(code, encoding="utf-8")
        return _colab_exec_file(session, script_path, timeout=timeout, capture=capture)


_CST_TZ = timezone(timedelta(hours=8))
_TS_SUFFIX_RE = re.compile(r"_\d{8}-\d{6}(?:-\d+)?$")
_PRESERVE_ORIGINAL_REL_RE = re.compile(r"(?:^|/)drama3_voices/[^/]+\.safetensors$", re.IGNORECASE)


def _format_timestamped_filename(filename: str, mtime_sec: float | None = None) -> str:
    """为文件名追加确定性北京时间时间戳（YYYYMMDD-HHMMSS），已有时间戳后缀则幂等跳过。"""
    p = PurePosixPath(filename)
    stem, suffix = p.stem, p.suffix
    if _TS_SUFFIX_RE.search(stem):
        return filename
    ts_str = datetime.fromtimestamp(mtime_sec if mtime_sec is not None else time.time(), _CST_TZ).strftime(
        "%Y%m%d-%H%M%S"
    )
    sep = "" if stem.endswith("_") else "_"
    return f"{stem}{sep}{ts_str}{suffix}"


def _resolve_non_clobber_dest(dest: Path, src_size: int, src_sha256: str) -> tuple[Path, bool]:
    """若目标文件已存在且内容相同则跳过覆写；若内容不同则递增后缀防止误覆盖历史作品。"""
    if not dest.exists():
        return dest, False
    candidate = dest
    idx = 2
    while candidate.exists():
        if candidate.is_file() and candidate.stat().st_size == src_size:
            if _sha256_file(candidate) == src_sha256:
                return candidate, True
        candidate = dest.with_name(f"{dest.stem}-{idx}{dest.suffix}")
        idx += 1
    return candidate, False


def _normalize_prefix(prefix: str | None) -> str:
    """校验并规范化子目录前缀，防止路径穿越。"""
    if not prefix:
        return ""
    cleaned = prefix.replace("\\", "/").strip("/")
    if not cleaned:
        return ""
    posix_p = PurePosixPath(cleaned)
    if posix_p.is_absolute() or ".." in posix_p.parts:
        raise ValueError(f"非法的子目录前缀: {prefix}")
    return posix_p.as_posix()


def _safe_extract_tar(tar_path: Path, dest_dir: Path):
    """安全解压 tar.gz 归档到本地目标目录，阻止路径穿越并防止同名异内容覆写。"""
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
                final_dest, skip_write = _resolve_non_clobber_dest(target, size, h.hexdigest())
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


def cmd_sync_loras(args):
    """零 GPU 消耗：核对并增量上传本地 LoRA 目录至 Google Drive。"""
    _ensure_bin(RCLONE_EXE, "rclone.exe")
    _ensure_bin(RCLONE_CONF, "rclone.conf")

    lora_dir = Path(getattr(args, "src", None) or LORA_BACKUP_DIR)
    entries = _discover_local_loras(lora_dir)
    if not entries:
        raise FileNotFoundError(
            f"未在目录中找到任何 .safetensors LoRA 文件: {lora_dir}（可通过 --src 或 COLAB_LORA_DIR 指定）"
        )

    print(f"=== [1/4] 校验本地 {len(entries)} 个 LoRA 完整性 ===", flush=True)
    verified = []
    for rel_path, expected_size, expected_sha in entries:
        src = lora_dir / rel_path
        if not src.is_file():
            raise FileNotFoundError(f"本地 LoRA 缺失: {src}")
        actual_size = src.stat().st_size
        if expected_size is not None and actual_size != expected_size:
            raise ValueError(
                f"LoRA 字节数不符: {src.name} 实际={actual_size} B, 预期={expected_size} B"
            )
        actual_sha = _sha256_file(src)
        if expected_sha is not None and actual_sha != expected_sha:
            raise ValueError(
                f"LoRA SHA256 不符: {src.name} 实际={actual_sha}, 预期={expected_sha}"
            )
        verified.append((src, actual_size, actual_sha))
        print(f"  [OK] {src.name} ({actual_size} B)", flush=True)

    def _parse_lsl_sizes(stdout: str) -> dict[str, int]:
        """解析 rclone lsl 输出为 {文件名: 字节数} 字典。"""
        sizes: dict[str, int] = {}
        for line in stdout.splitlines():
            parts = line.strip().split(None, 3)
            if len(parts) == 4 and parts[0].isdigit():
                sizes[parts[3]] = int(parts[0])
        return sizes

    print("=== [2/4] 预检远端目录并增量并发上传 LoRA (HTTP/1.1 + 8M 分块 + 抗限流退避) ===", flush=True)
    _run([
        RCLONE_EXE, "mkdir",
        "--config", RCLONE_CONF,
        "--disable-http2",
        "--retries", "6",
        "--retries-sleep", "5s",
        GDRIVE_LORA_REMOTE,
    ])
    pre_res = _run(
        [
            RCLONE_EXE, "lsl",
            "--config", RCLONE_CONF,
            "--disable-http2",
            "--retries", "6",
            "--retries-sleep", "5s",
            GDRIVE_LORA_REMOTE,
        ],
        capture=True,
    )
    existing_sizes = _parse_lsl_sizes(pre_res.stdout)

    todo = []
    for src, sz, sha in verified:
        if existing_sizes.get(src.name) == sz:
            print(f"  [远端已就绪，跳过] {src.name} ({sz} B)", flush=True)
        else:
            todo.append((src, sz, sha))

    def _upload_one(src: Path):
        """通过 HTTP/1.1 强制穿透本地代理并带 403 限流退避上传单个 LoRA 权重。"""
        t0 = time.time()
        cmd = [
            str(RCLONE_EXE),
            "copyto",
            "--config", str(RCLONE_CONF),
            "--disable-http2",
            "--drive-chunk-size", "8M",
            "--timeout", "5m",
            "--contimeout", "30s",
            "--low-level-retries", "10",
            "--retries", "10",
            "--retries-sleep", "6s",
            "-v",
            "--stats-one-line",
            "--stats", "5s",
            str(src),
            f"{GDRIVE_LORA_REMOTE}/{src.name}",
        ]
        print(f"  [开始上传] {src.name}", flush=True)
        subprocess.run(cmd, check=True)
        print(f"  [上传完成] {src.name} (耗时 {time.time() - t0:.1f}s)", flush=True)

    if todo:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(_upload_one, src) for src, _, _ in todo]
            for fut in as_completed(futures):
                fut.result()
    else:
        print(f"  所有 {len(verified)} 个 LoRA 均已存在于远端，无需重复上传。", flush=True)

    print("=== [3/4] 生成并上传 sha256sums.txt ===", flush=True)
    with tempfile.TemporaryDirectory(prefix="lora_sums_") as tmp_dir:
        sums_file = Path(tmp_dir) / "sha256sums.txt"
        lines = [f"{sha}  {src.name}" for src, _, sha in verified]
        sums_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
        _run([
            RCLONE_EXE,
            "copyto",
            "--config", RCLONE_CONF,
            "--disable-http2",
            "--retries", "6",
            "--retries-sleep", "5s",
            sums_file,
            f"{GDRIVE_LORA_REMOTE}/sha256sums.txt",
        ])

    print("=== [4/4] 核对 Google Drive 远端文件清单与字节数 ===", flush=True)
    res = _run(
        [
            RCLONE_EXE, "lsl",
            "--config", RCLONE_CONF,
            "--disable-http2",
            "--retries", "6",
            "--retries-sleep", "5s",
            GDRIVE_LORA_REMOTE,
        ],
        capture=True,
    )
    print(res.stdout, end="", flush=True)

    remote_sizes = _parse_lsl_sizes(res.stdout)

    for src, expected_size, _ in verified:
        if src.name not in remote_sizes:
            raise RuntimeError(f"远端缺少已上传 LoRA: {src.name}")
        if remote_sizes[src.name] != expected_size:
            raise RuntimeError(
                f"远端字节数不一致: {src.name} 远端={remote_sizes[src.name]}, 预期={expected_size}"
            )
    if remote_sizes.get("sha256sums.txt", 0) <= 0:
        raise RuntimeError("远端缺少有效的 sha256sums.txt")

    total_mib = sum(sz for _, sz, _ in verified) / (1024 ** 2)
    print(f"LoRA 上传并核验完成！共 {len(verified)} 个权重（总计 {total_mib:.2f} MiB）。", flush=True)


def cmd_sync_drama3_refs(args):
    """零 GPU 消耗：将本地精选 Drama 3 参考 .wav 上传至存储号 Google Drive。"""
    _ensure_bin(RCLONE_EXE, "rclone.exe")
    _ensure_bin(RCLONE_CONF, "rclone.conf")
    src_dir = Path(getattr(args, "src", None) or DRAMA3_REFS_LOCAL_DIR)
    if not src_dir.is_dir():
        raise FileNotFoundError(f"未找到 Drama 3 参考音频目录: {src_dir}")

    wavs = sorted(src_dir.rglob("*.wav"))
    if not wavs:
        raise RuntimeError(f"目录下未找到任何 .wav 文件: {src_dir}")

    print(f"=== [1/3] 发现 {len(wavs)} 条精选参考音频，准备预检云盘目录 ===", flush=True)
    _run([
        RCLONE_EXE, "mkdir",
        "--config", RCLONE_CONF,
        "--disable-http2",
        "--retries", "6",
        "--retries-sleep", "5s",
        GDRIVE_DRAMA3_REFS_REMOTE,
    ])
    pre_res = _run(
        [
            RCLONE_EXE, "lsl",
            "--config", RCLONE_CONF,
            "--disable-http2",
            "--retries", "6",
            "--retries-sleep", "5s",
            GDRIVE_DRAMA3_REFS_REMOTE,
        ],
        capture=True,
    )
    existing_sizes: dict[str, int] = {}
    for line in (pre_res.stdout or "").splitlines():
        parts = line.strip().split(None, 3)
        if len(parts) == 4 and parts[0].isdigit():
            existing_sizes[parts[3]] = int(parts[0])

    todo = []
    for wf in wavs:
        sz = wf.stat().st_size
        if existing_sizes.get(wf.name) == sz:
            print(f"  [远端已就绪，跳过] {wf.name} ({sz} B)", flush=True)
        else:
            todo.append((wf, sz))

    print(f"=== [2/3] 增量上传 Drama 3 参考音频（待传 {len(todo)}/{len(wavs)} 条）===", flush=True)
    for wf, _sz in todo:
        _run([
            RCLONE_EXE, "copyto",
            "--config", RCLONE_CONF,
            "--disable-http2",
            "--retries", "8",
            "--retries-sleep", "5s",
            wf,
            f"{GDRIVE_DRAMA3_REFS_REMOTE}/{wf.name}",
        ])

    print("=== [3/3] 核对远端字节完整性 ===", flush=True)
    chk_res = _run(
        [
            RCLONE_EXE, "lsl",
            "--config", RCLONE_CONF,
            "--disable-http2",
            "--retries", "6",
            "--retries-sleep", "5s",
            GDRIVE_DRAMA3_REFS_REMOTE,
        ],
        capture=True,
    )
    remote_sizes: dict[str, int] = {}
    for line in (chk_res.stdout or "").splitlines():
        parts = line.strip().split(None, 3)
        if len(parts) == 4 and parts[0].isdigit():
            remote_sizes[parts[3]] = int(parts[0])
    for wf in wavs:
        if remote_sizes.get(wf.name) != wf.stat().st_size:
            raise RuntimeError(f"云盘参考音频校验失败: {wf.name}")
    total_mb = sum(w.stat().st_size for w in wavs) / (1024 ** 2)
    print(f"Drama 3 参考音频上传并校验完成！共 {len(wavs)} 条（总计 {total_mb:.2f} MiB）。", flush=True)


def cmd_status(_args):
    """查看 Colab 当前活跃会话及 CU 积分余量/消耗速率。"""
    _ensure_bin(COLAB_EXE, "colab.exe")
    print("=== Colab 活跃会话 ===", flush=True)
    _run([COLAB_EXE, "sessions"])
    print("=== Colab 配额与速率 ===", flush=True)
    _run([COLAB_EXE, "usage"])


def cmd_up(args):
    """创建或复用 Colab GPU 会话，上传可选 rclone 配置并完成指定生态（ComfyUI 或 Qwen3.8 LLM）自举部署。"""
    _ensure_bin(COLAB_EXE, "colab.exe")

    profile = getattr(args, "profile", "anima") or "anima"
    if profile == "qwen_llm":
        target_bootstrap = QWEN_LLM_BOOTSTRAP_SCRIPT
        default_gpu = "G4"
        exec_timeout = 2400
    elif profile == "minimax":
        target_bootstrap = MINIMAX_BOOTSTRAP_SCRIPT
        default_gpu = "G4"
        exec_timeout = 2400
    elif profile == "drama3":
        target_bootstrap = DRAMA3_BOOTSTRAP_SCRIPT
        default_gpu = "G4"
        exec_timeout = 1800
    else:
        target_bootstrap = BOOTSTRAP_SCRIPT
        default_gpu = "L4"
        exec_timeout = 600
    _ensure_bin(target_bootstrap, target_bootstrap.name)

    gpu = getattr(args, "gpu", None) or default_gpu
    # L4 与 G4 均无需也不能叠加 --high-mem（Colab 后端不支持 G4 HIGH_RAM 会拒分配）
    high_mem = False if str(gpu).upper() in ("L4", "G4") else bool(getattr(args, "high_mem", False))
    recreate = bool(getattr(args, "recreate", False))
    minimax_lite = bool(getattr(args, "minimax_lite", False))
    no_r2v = bool(getattr(args, "no_r2v", False))
    no_pe = bool(getattr(args, "no_pe", False))

    sessions_res = _run([COLAB_EXE, "sessions"], check=False, capture=True)
    sessions_out = sessions_res.stdout or ""
    has_session = (
        sessions_res.returncode == 0
        and "No active sessions" not in sessions_out
        and re.search(rf"(?<![A-Za-z0-9_-]){re.escape(args.session)}(?![A-Za-z0-9_-])", sessions_out) is not None
    )

    if recreate and has_session:
        print("=== 正在释放旧实例以重新分配显卡 ===", flush=True)
        _run([COLAB_EXE, "stop", "-s", args.session], check=False)
        if COLAB_PY.is_file():
            cleanup_code = """
from colab_cli.common import state

assignments = state.client.list_assignments()
for item in assignments:
    ep = getattr(item, "endpoint", None) or (item.get("endpoint") if isinstance(item, dict) else None)
    if ep:
        print(f"正在释放实例: {ep}", flush=True)
        state.client.unassign(ep)
"""
            _run([COLAB_PY, "-c", cleanup_code], check=False)
        has_session = False

    if has_session:
        print(
            f"检测到会话 {args.session} 已存在，跳过 colab new，继续增量部署 ({profile})...",
            flush=True,
        )
    else:
        print(f"=== 开启 Colab 实例 (session={args.session}, profile={profile}, gpu={gpu}, high_mem={high_mem}) ===", flush=True)
        new_cmd = [COLAB_EXE, "new", "-s", args.session, "--gpu", gpu]
        if high_mem:
            new_cmd.append("--high-mem")
        _run(new_cmd)

    if RCLONE_CONF.is_file():
        _run([COLAB_EXE, "upload", "-s", args.session, RCLONE_CONF, REMOTE_RCLONE_CONF])
    else:
        print(f"[提示] 未检测到 rclone.conf ({RCLONE_CONF})，跳过云盘凭据上传（仍可正常使用公共模型与公网隧道）。", flush=True)
    cf_cfg = _load_cf_tunnel_secret()
    cf_tok = str(cf_cfg.get("tunnel_token") or "").strip()
    cf_llm_url = str(cf_cfg.get("llm_url") or "").strip().rstrip("/")
    cf_comfy_url = str(cf_cfg.get("comfy_url") or "").strip().rstrip("/")

    if profile == "qwen_llm":
        llm_quant = str(getattr(args, "llm_quant", None) or "UD-Q4_K_XL").strip()
        llm_ctx = str(int(getattr(args, "llm_ctx", None) or 32768))
        llm_mmproj = "0" if bool(getattr(args, "no_mmproj", False)) else "1"
        llm_webui = "0" if bool(getattr(args, "no_webui", False)) else "1"
        llm_gdrive = "1" if bool(getattr(args, "gdrive_cache", False)) else "0"
        llm_api_key = str(getattr(args, "llm_api_key", None) or "auto").strip() or "auto"
        if llm_api_key.lower() == "auto":
            fixed_key = str(cf_cfg.get("llm_api_key") or "").strip()
            if len(fixed_key) >= 16:
                llm_api_key = fixed_key

        env_lines = [
            "import os",
            f"os.environ['LLM_QUANT'] = {llm_quant!r}",
            f"os.environ['LLM_CTX_SIZE'] = {llm_ctx!r}",
            f"os.environ['LLM_WITH_MMPROJ'] = {llm_mmproj!r}",
            f"os.environ['LLM_WITH_WEBUI'] = {llm_webui!r}",
            f"os.environ['LLM_GDRIVE_CACHE'] = {llm_gdrive!r}",
            f"os.environ['LLM_API_KEY'] = {llm_api_key!r}",
        ]
        if cf_tok and cf_llm_url:
            env_lines.append(f"os.environ['CF_TUNNEL_TOKEN'] = {cf_tok!r}")
            env_lines.append(f"os.environ['CF_TUNNEL_URL'] = {cf_llm_url!r}")
        env_header = "\n".join(env_lines) + "\n"
        with tempfile.TemporaryDirectory(prefix="qwen_llm_opts_") as tmp_dir:
            wrapped = Path(tmp_dir) / target_bootstrap.name
            wrapped.write_text(
                env_header + target_bootstrap.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            _colab_exec_file(args.session, wrapped, timeout=exec_timeout)
    else:
        # 部署 ComfyUI 生态（anima / minimax / drama3）时，前置清理可能遗留的 LLM 服务契约与进程
        prep_lines = [
            "import os, subprocess",
            "from pathlib import Path",
            "_meta_p = Path('/root/colab_service_meta.json')",
            "if _meta_p.exists():",
            "    _meta_p.unlink(missing_ok=True)",
            "    subprocess.run(['pkill', '-f', 'cloudflared tunnel'], check=False, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)",
            "subprocess.run(['pkill', '-f', 'llm_sse_gateway.py'], check=False, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)",
            "subprocess.run(['pkill', '-f', 'llama-server'], check=False, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)",
        ]
        if cf_tok and cf_comfy_url:
            prep_lines.append(f"os.environ['CF_TUNNEL_TOKEN'] = {cf_tok!r}")
            prep_lines.append(f"os.environ['CF_TUNNEL_URL'] = {cf_comfy_url!r}")
        if profile == "minimax":
            if minimax_lite:
                prep_lines.append("os.environ['MINIMAX_LITE'] = '1'")
            if no_r2v:
                prep_lines.append("os.environ['MINIMAX_WITH_R2V'] = '0'")
            if no_pe:
                prep_lines.append("os.environ['MINIMAX_WITH_PE'] = '0'")
        prep_header = "\n".join(prep_lines) + "\n"
        with tempfile.TemporaryDirectory(prefix=f"{profile}_opts_") as tmp_dir:
            wrapped = Path(tmp_dir) / target_bootstrap.name
            wrapped.write_text(
                prep_header + target_bootstrap.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            _colab_exec_file(args.session, wrapped, timeout=exec_timeout)


def cmd_tunnel(args):
    """获取或重启远端 ComfyUI / LLM cloudflared 隧道（支持命名隧道无缝切换），并在本机按服务类型校验公网连通性。"""
    force_restart = bool(getattr(args, "restart", False))
    cf_cfg = _load_cf_tunnel_secret()
    cf_tok = str(cf_cfg.get("tunnel_token") or "").strip()
    cf_llm_url = str(cf_cfg.get("llm_url") or "").strip().rstrip("/")
    cf_comfy_url = str(cf_cfg.get("comfy_url") or "").strip().rstrip("/")

    # 非强制重启时，优先走 Colab Contents HTTP REST 通道免 WebSocket 直读服务元数据与隧道地址（~1.5s）
    if not force_restart and COLAB_PY.is_file():
        fast_read_code = """
import base64, json, re, sys, requests
from colab_cli.common import state

TUNNEL_RE = re.compile(r"https://(?!api\\.)(?:[a-z0-9-]+\\.)+[a-z]{2,}(?::\\d+)?")
sname = sys.argv[1] if len(sys.argv) > 1 else "comfy"
local_sessions = state.store.list()
assignments = state.client.list_assignments()
if not assignments:
    raise SystemExit(2)

target = None
if sname in local_sessions:
    wanted_ep = local_sessions[sname].endpoint
    for a in assignments:
        if a.endpoint == wanted_ep:
            target = a
            break
if target is None:
    target = assignments[0]

info = target.runtime_proxy_info
base = info.url.rstrip("/")

def _read_remote_text(rel_path: str):
    r = requests.get(
        f"{base}/api/contents/{rel_path.lstrip('/')}",
        params={"authuser": "0", "colab-runtime-proxy-token": info.token, "content": "1"},
        timeout=8,
    )
    if r.status_code != 200:
        return None
    d = r.json()
    raw = d.get("content", "")
    if d.get("format") == "base64" and isinstance(raw, str):
        return base64.b64decode(raw).decode("utf-8", errors="ignore")
    return str(raw or "")

meta_txt = _read_remote_text("root/colab_service_meta.json")
kind = "comfy"
port = 8188
url = ""
model = ""
gpu_desc = ""

if meta_txt:
    try:
        meta = json.loads(meta_txt)
        if isinstance(meta, dict):
            kind = str(meta.get("kind") or "comfy")
            port = int(meta.get("port") or (18080 if kind == "llm" else 8188))
            url = str(meta.get("tunnel_url") or "").strip()
            model = str(meta.get("model") or "")
            gname = str(meta.get("gpu_name") or "")
            vtot = meta.get("vram_total_gb")
            if gname:
                gpu_desc = f"{gname} ({vtot} GiB)" if vtot else gname
    except Exception:
        pass

if not url:
    txt = _read_remote_text("root/comfy_tunnel_url.txt") or ""
    m = TUNNEL_RE.search(txt)
    if m:
        url = m.group(0)

m_valid = TUNNEL_RE.search(url)
if not m_valid:
    raise SystemExit(4)

out = {"kind": kind, "port": port, "url": m_valid.group(0), "model": model, "gpu_desc": gpu_desc}
print("<<<FAST_TUNNEL>>>" + json.dumps(out, ensure_ascii=False) + "<<<END_FAST_TUNNEL>>>", flush=True)
""".strip()
        try:
            env = os.environ.copy()
            env["PYTHONUTF8"] = "1"
            env["PYTHONIOENCODING"] = "utf-8"
            env.setdefault("COLAB_CLI_REQUEST_TIMEOUT", "12")
            fast_res = subprocess.run(
                [str(COLAB_PY), "-c", fast_read_code, args.session],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=15,
                env=env,
            )
            if fast_res.returncode == 0:
                m_pkt = re.search(
                    r"<<<FAST_TUNNEL>>>(.*?)<<<END_FAST_TUNNEL>>>",
                    fast_res.stdout or "",
                    re.S,
                )
                if m_pkt:
                    pkt = json.loads(m_pkt.group(1))
                    fast_url = str(pkt.get("url") or "").strip()
                    fast_kind = str(pkt.get("kind") or "comfy")
                    expected_cf_url = (cf_llm_url if fast_kind == "llm" else cf_comfy_url) if cf_tok else ""
                    if expected_cf_url and fast_url.rstrip("/") != expected_cf_url.rstrip("/"):
                        fast_url = ""
                    if fast_url and fast_kind == "llm":
                        req = urllib.request.Request(
                            f"{fast_url}/health",
                            headers={"User-Agent": "Mozilla/5.0"},
                        )
                        with urllib.request.urlopen(req, timeout=5) as resp:
                            if resp.status == 200:
                                body = json.loads(resp.read().decode("utf-8") or "{}")
                                if body.get("status") == "ok":
                                    model_lbl = pkt.get("model") or "Qwen3.8-Flash-Next-abliterated"
                                    gpu_lbl = pkt.get("gpu_desc") or "GPU 在线"
                                    print("复用运行中的 cloudflared 隧道（HTTP 快速通道）。", flush=True)
                                    print(f"SERVICE_TUNNEL_URL: {fast_url}", flush=True)
                                    print(f"COMFYUI_TUNNEL_URL: {fast_url}", flush=True)
                                    print(f"[隧道正常] HTTP {resp.status} | LLM: {model_lbl} | {gpu_lbl}", flush=True)
                                    print(f"[LLM WebUI / API 地址] {fast_url}", flush=True)
                                    return
                    elif fast_url:
                        req = urllib.request.Request(
                            f"{fast_url}/system_stats",
                            headers={"User-Agent": "Mozilla/5.0"},
                        )
                        with urllib.request.urlopen(req, timeout=5) as resp:
                            if resp.status == 200:
                                body = json.loads(resp.read().decode("utf-8"))
                                devices = body.get("devices") or []
                                dev_desc = ", ".join(
                                    f"{d.get('name', 'unknown')} ({d.get('vram_total', 0) / (1024 ** 3):.1f} GiB)"
                                    for d in devices
                                ) or "未返回设备列表"
                                print("复用运行中的 cloudflared 隧道（HTTP 快速通道）。", flush=True)
                                print(f"COMFYUI_TUNNEL_URL: {fast_url}", flush=True)
                                print(f"[隧道正常] HTTP {resp.status} | GPU: {dev_desc}", flush=True)
                                print(f"[ComfyUI WebUI 地址] {fast_url}", flush=True)
                                return
        except Exception:
            pass

    remote_code = f"""
import json, os, re, shutil, subprocess, time, urllib.request
from pathlib import Path

CLOUDFLARED_BIN = Path("/usr/local/bin/cloudflared")
CLOUDFLARED_LOG = Path("/root/cloudflared.log")
TUNNEL_URL_FILE = Path("/root/comfy_tunnel_url.txt")
SERVICE_META_FILE = Path("/root/colab_service_meta.json")
TUNNEL_RE = re.compile(r"https://(?!api\\.)(?:[a-z0-9-]+\\.)+[a-z]{2,}(?::\\d+)?")
FORCE_RESTART = {force_restart!r}
CF_TOKEN = {cf_tok!r}
CF_LLM_URL = {cf_llm_url!r}
CF_COMFY_URL = {cf_comfy_url!r}

kind = "comfy"
port = 8188
meta = None
if SERVICE_META_FILE.is_file():
    try:
        loaded = json.loads(SERVICE_META_FILE.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            meta = loaded
            kind = str(meta.get("kind") or "comfy")
            port = int(meta.get("port") or (18080 if kind == "llm" else 8188))
    except Exception:
        meta = None

cf_target_url = (CF_LLM_URL if kind == "llm" else CF_COMFY_URL).strip().rstrip("/")
use_named = bool(CF_TOKEN and cf_target_url)

pgrep_res = subprocess.run(
    ["pgrep", "-af", "cloudflared tunnel"],
    stdin=subprocess.DEVNULL,
    capture_output=True,
    text=True,
)
proc_out = pgrep_res.stdout or ""
tunnel_url = None
if use_named:
    if (
        not FORCE_RESTART
        and pgrep_res.returncode == 0
        and " run" in proc_out
        and TUNNEL_URL_FILE.is_file()
        and TUNNEL_URL_FILE.read_text(encoding="utf-8", errors="ignore").strip().rstrip("/") == cf_target_url
    ):
        tunnel_url = cf_target_url
        print(f"复用运行中的 cloudflared 命名隧道 (127.0.0.1:{{port}})。", flush=True)
elif (
    not FORCE_RESTART
    and pgrep_res.returncode == 0
    and f"127.0.0.1:{{port}}" in proc_out
    and TUNNEL_URL_FILE.is_file()
):
    cached = TUNNEL_URL_FILE.read_text(encoding="utf-8", errors="ignore").strip()
    m = TUNNEL_RE.search(cached)
    if m:
        tunnel_url = m.group(0)
        print(f"复用运行中的 cloudflared 隧道 (127.0.0.1:{{port}})。", flush=True)

if not tunnel_url:
    subprocess.run(
        ["pkill", "-f", "cloudflared tunnel"],
        check=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(0.5)

    cf_bin = shutil.which("cloudflared") or str(CLOUDFLARED_BIN)
    if not Path(cf_bin).is_file():
        cf_url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
        print(f"正在远端下载 cloudflared: {{cf_url}}", flush=True)
        tmp_cf = CLOUDFLARED_BIN.with_suffix(".part")
        req = urllib.request.Request(cf_url, headers={{"User-Agent": "Mozilla/5.0"}})
        with urllib.request.urlopen(req, timeout=120) as resp, open(tmp_cf, "wb") as f:
            while True:
                chunk = resp.read(4 * 1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
        tmp_cf.replace(CLOUDFLARED_BIN)
        os.chmod(CLOUDFLARED_BIN, 0o755)
        cf_bin = str(CLOUDFLARED_BIN)

    if use_named:
        cf_env = os.environ.copy()
        cf_env["TUNNEL_TOKEN"] = CF_TOKEN
        cmd = [cf_bin, "tunnel", "--no-autoupdate", "--protocol", "http2", "run"]
    else:
        cf_env = os.environ.copy()
        cmd = [
            cf_bin,
            "tunnel",
            "--no-autoupdate",
            "--protocol", "http2",
            "--url", f"http://127.0.0.1:{{port}}",
        ]

    log_f = open(CLOUDFLARED_LOG, "wb")
    try:
        proc = subprocess.Popen(
            cmd,
            env=cf_env,
            stdin=subprocess.DEVNULL,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    finally:
        log_f.close()
    print(f"已重启 cloudflared 隧道进程 -> 127.0.0.1:{{port}} (PID={{proc.pid}})", flush=True)

    if use_named:
        for _ in range(35):
            if proc.poll() is not None:
                tail = CLOUDFLARED_LOG.read_text(encoding="utf-8", errors="ignore")[-2000:] if CLOUDFLARED_LOG.is_file() else ""
                if CF_TOKEN and CF_TOKEN in tail:
                    tail = tail.replace(CF_TOKEN, "[REDACTED]")
                raise RuntimeError(f"cloudflared 命名隧道异常退出 (rc={{proc.returncode}})，末尾日志:\\n{{tail}}")
            if CLOUDFLARED_LOG.is_file() and "Registered tunnel connection" in CLOUDFLARED_LOG.read_text(encoding="utf-8", errors="ignore"):
                tunnel_url = cf_target_url
                break
            time.sleep(1)
        if not tunnel_url:
            tail = CLOUDFLARED_LOG.read_text(encoding="utf-8", errors="ignore")[-2000:] if CLOUDFLARED_LOG.is_file() else ""
            if CF_TOKEN and CF_TOKEN in tail:
                tail = tail.replace(CF_TOKEN, "[REDACTED]")
            raise TimeoutError(f"等待 cloudflared 命名隧道注册超时（35s），末尾日志:\\n{{tail}}")
    else:
        for _ in range(30):
            if CLOUDFLARED_LOG.is_file():
                txt = CLOUDFLARED_LOG.read_text(encoding="utf-8", errors="ignore")
                m = TUNNEL_RE.search(txt)
                if m:
                    tunnel_url = m.group(0)
                    break
            time.sleep(1)

        if not tunnel_url:
            tail = CLOUDFLARED_LOG.read_text(encoding="utf-8", errors="ignore")[-2000:] if CLOUDFLARED_LOG.is_file() else ""
            raise TimeoutError(f"等待 cloudflared 隧道地址超时（30s），末尾日志:\\n{{tail}}")

    TUNNEL_URL_FILE.write_text(tunnel_url + "\\n", encoding="utf-8")
    if isinstance(meta, dict):
        meta["tunnel_url"] = tunnel_url
        tmp_meta = SERVICE_META_FILE.with_suffix(".json.tmp")
        fd = os.open(str(tmp_meta), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as mf:
            json.dump(meta, mf, ensure_ascii=False, indent=2)
            mf.write("\\n")
        os.chmod(tmp_meta, 0o600)
        tmp_meta.replace(SERVICE_META_FILE)
        os.chmod(SERVICE_META_FILE, 0o600)

print(f"SERVICE_KIND: {{kind}}", flush=True)
if kind == "llm":
    print(f"SERVICE_TUNNEL_URL: {{tunnel_url}}", flush=True)
print(f"COMFYUI_TUNNEL_URL: {{tunnel_url}}", flush=True)
"""
    res = _exec_remote_py(args.session, remote_code, timeout=120, capture=True)
    remote_out = res.stdout or ""
    print(remote_out, end="" if remote_out.endswith("\n") else "\n", flush=True)

    kind_match = re.search(r"SERVICE_KIND:\s*(\w+)", remote_out)
    service_kind = kind_match.group(1).strip().lower() if kind_match else "comfy"

    url_match = re.search(
        r"(?:SERVICE_TUNNEL_URL|COMFYUI_TUNNEL_URL):\s*(https://(?!api\.)(?:[a-z0-9-]+\.)+[a-z]{2,}(?::\d+)?)",
        remote_out,
    )
    if not url_match:
        url_match = re.search(r"https://(?!api\.)[a-z0-9-]+\.trycloudflare\.com", remote_out)
    if not url_match:
        raise RuntimeError("未能从远端输出中解析出隧道公网 URL")

    tunnel_url = url_match.group(1) if url_match.lastindex else url_match.group(0)
    probe_path = "/health" if service_kind == "llm" else "/system_stats"
    print(f"=== 本机端到端校验隧道连通性: {tunnel_url}{probe_path} ===", flush=True)

    last_err = None
    for attempt in range(1, 7):
        try:
            req = urllib.request.Request(
                f"{tunnel_url}{probe_path}",
                headers={"User-Agent": "Mozilla/5.0"},
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                status = resp.status
                body = json.loads(resp.read().decode("utf-8") or "{}")
            if service_kind == "llm":
                if status != 200 or body.get("status") != "ok":
                    raise RuntimeError(f"LLM /health 未就绪: HTTP {status}, body={body}")
                print(f"[隧道正常] HTTP {status} | LLM 服务在线 (status=ok)", flush=True)
                print(f"[LLM WebUI / API 地址] {tunnel_url}", flush=True)
                return

            devices = body.get("devices") or []
            dev_desc = ", ".join(
                f"{d.get('name', 'unknown')} ({d.get('vram_total', 0) / (1024 ** 3):.1f} GiB)"
                for d in devices
            ) or "未返回设备列表"
            print(f"[隧道正常] HTTP {status} | GPU: {dev_desc}", flush=True)
            print(f"[ComfyUI WebUI 地址] {tunnel_url}", flush=True)
            return
        except Exception as exc:
            last_err = exc
            print(f"  第 {attempt}/6 次探测暂未就绪 ({exc})，3 秒后重试...", flush=True)
            time.sleep(3)

    raise RuntimeError(
        f"本机访问隧道 {tunnel_url}{probe_path} 连续 6 次失败，最后错误: {last_err}"
    )


_REMOTE_WAIT_SCRIPT = r"""
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

TIMEOUT_SEC = __TIMEOUT_SEC__
meta_file = Path("/root/colab_service_meta.json")
kind = "comfy"
port = 8188
api_key = ""
if meta_file.is_file():
    try:
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        if isinstance(meta, dict):
            kind = str(meta.get("kind") or "comfy")
            port = int(meta.get("port") or (18080 if kind == "llm" else 8188))
            api_key = str(meta.get("api_key") or "").strip()
    except Exception:
        pass


def _is_proc_alive(pattern: str) -> bool:
    try:
        res = subprocess.run(
            ["pgrep", "-f", pattern],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
        )
        return res.returncode == 0 and bool((res.stdout or "").strip())
    except Exception:
        return True


def _read_tail_log(paths: list[str], limit: int = 2000) -> str:
    for p_str in paths:
        p = Path(p_str)
        if p.is_file():
            try:
                return f"--- {p_str} ---\n" + p.read_text(encoding="utf-8", errors="ignore")[-limit:]
            except Exception:
                pass
    return ""


t0 = time.time()
last_print = 0.0
dead_checks = 0
conn_errors = 0

if kind == "llm":
    slots_url = f"http://127.0.0.1:{port}/slots"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    while True:
        try:
            req = urllib.request.Request(slots_url, headers=headers)
            with urllib.request.urlopen(req, timeout=15) as r:
                slots = json.loads(r.read().decode("utf-8") or "[]")
            running = sum(1 for s in slots if isinstance(s, dict) and s.get("is_processing")) if isinstance(slots, list) else 0
            dead_checks = 0
            conn_errors = 0
        except Exception as exc:
            conn_errors += 1
            print(f"[等待队列] 查询 LLM /slots 异常 ({conn_errors}): {exc}", flush=True)
            running = -1
            if not _is_proc_alive("llama-server"):
                dead_checks += 1
                if dead_checks >= 2:
                    tail = _read_tail_log(["/root/llama_server.log"])
                    print(f"[熔断退出] llama-server 进程已消失，停止无意义等待！\n{tail}", file=sys.stderr, flush=True)
                    sys.exit(2)
            elif conn_errors >= 12:
                tail = _read_tail_log(["/root/llama_server.log"])
                print(f"[熔断退出] llama-server 连续 {conn_errors} 次无响应，中止等待！\n{tail}", file=sys.stderr, flush=True)
                sys.exit(2)

        now = time.time()
        if running == 0:
            print(f"[等待队列] LLM 并发槽位已全部空闲（耗时 {now - t0:.1f} 秒）。", flush=True)
            break
        if now - t0 > TIMEOUT_SEC:
            print(f"[等待队列] 等待 LLM 空闲超时（{TIMEOUT_SEC} 秒），剩余活跃槽位={running}", flush=True)
            sys.exit(1)
        if now - last_print >= 5.0:
            print(f"[等待队列] LLM 活跃推理槽位={running}, 已等待={now - t0:.0f}s", flush=True)
            last_print = now
        time.sleep(3)
else:
    api = "http://127.0.0.1:8188/queue"
    while True:
        try:
            with urllib.request.urlopen(api, timeout=15) as r:
                q = json.loads(r.read().decode("utf-8"))
            running = len(q.get("queue_running") or [])
            pending = len(q.get("queue_pending") or [])
            dead_checks = 0
            conn_errors = 0
        except Exception as exc:
            conn_errors += 1
            print(f"[等待队列] 查询 ComfyUI 队列异常 ({conn_errors}): {exc}", flush=True)
            running, pending = -1, -1
            if not _is_proc_alive(r"ComfyUI/main\.py|main\.py.*8188"):
                dead_checks += 1
                if dead_checks >= 2:
                    tail = _read_tail_log(["/root/comfy_colab.log", "/root/comfy_bw1.log"])
                    print(f"[熔断退出] ComfyUI 进程已消失（可能 OOM 或崩溃），停止等待！\n{tail}", file=sys.stderr, flush=True)
                    sys.exit(2)
            elif conn_errors >= 12:
                tail = _read_tail_log(["/root/comfy_colab.log", "/root/comfy_bw1.log"])
                print(f"[熔断退出] ComfyUI 连续 {conn_errors} 次无响应，中止等待！\n{tail}", file=sys.stderr, flush=True)
                sys.exit(2)

        now = time.time()
        if running == 0 and pending == 0:
            print(f"[等待队列] 队列已全部清空（耗时 {now - t0:.1f} 秒）。", flush=True)
            break

        if now - t0 > TIMEOUT_SEC:
            print(f"[等待队列] 等待超时（{TIMEOUT_SEC} 秒），剩余 running={running}, pending={pending}", flush=True)
            sys.exit(1)

        if now - last_print >= 5.0:
            print(f"[等待队列] 运行中={running}, 排队中={pending}, 已等待={now - t0:.0f}s", flush=True)
            last_print = now
        time.sleep(5)
"""


def cmd_wait(args):
    """阻塞轮询远端任务状态（ComfyUI /queue 或 LLM /slots），后端崩溃时快速熔断。"""
    code = _REMOTE_WAIT_SCRIPT.replace("__TIMEOUT_SEC__", repr(int(args.timeout)))
    _exec_remote_py(args.session, code, timeout=max(120, int(args.timeout) + 60))


_REMOTE_SYNC_SCRIPT = r"""
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
finally:
    shutil.rmtree(stage_dir, ignore_errors=True)
"""


def sync_session_outputs(session: str, prefix: str | None, pull: bool):
    """将远端输出目录以确定性时间戳归档同步至 Google Drive，按需打包拉回本机 _downloads（LLM 模式友好跳过）。"""
    norm_prefix = _normalize_prefix(prefix)
    remote_src = f"{REMOTE_OUT_ROOT}/{norm_prefix}" if norm_prefix else REMOTE_OUT_ROOT
    gdrive_dst = f"{GDRIVE_OUT_REMOTE}/{norm_prefix}" if norm_prefix else GDRIVE_OUT_REMOTE

    code = (
        _REMOTE_SYNC_SCRIPT
        .replace("__REMOTE_SRC__", repr(remote_src))
        .replace("__GDRIVE_DST__", repr(gdrive_dst))
        .replace("__REMOTE_RCLONE_CONF__", repr(REMOTE_RCLONE_CONF))
        .replace("__REMOTE_PULL_TAR__", repr(REMOTE_PULL_TAR))
        .replace("__DO_PULL__", repr(bool(pull)))
    )
    res = _exec_remote_py(session, code, timeout=600, capture=True)
    out_text = res.stdout or ""
    if out_text:
        print(out_text, end="" if out_text.endswith("\n") else "\n", flush=True)
    if "SKIP_LLM_SYNC" in out_text:
        return

    if pull:
        local_target = LOCAL_DOWNLOAD_DIR / (norm_prefix if norm_prefix else "all")
        with tempfile.TemporaryDirectory(prefix="colab_pull_") as tmp_dir:
            local_tar = Path(tmp_dir) / "comfy_pull.tar.gz"
            _run([COLAB_EXE, "download", "-s", session, REMOTE_PULL_TAR, local_tar])
            _safe_extract_tar(local_tar, local_target)
        print(f"已拉回并解压至本机目录: {local_target}", flush=True)


def cmd_sync(args):
    """处理 sync 子命令。"""
    sync_session_outputs(args.session, args.prefix, args.pull)


def cmd_smoke(args):
    """在远端运行冒烟验证（ComfyUI 模式同步并拉回验证结果，LLM 模式执行对话与鉴权冒烟）。"""
    profile = getattr(args, "profile", "anima") or "anima"
    if profile == "qwen_llm":
        _ensure_bin(QWEN_LLM_SMOKE_SCRIPT, "verify_qwen_llm.py")
        _colab_exec_file(args.session, QWEN_LLM_SMOKE_SCRIPT, timeout=300)
        return
    if profile == "minimax":
        _ensure_bin(MINIMAX_SMOKE_SCRIPT, "verify_minimax_h3.py")
        _colab_exec_file(args.session, MINIMAX_SMOKE_SCRIPT, timeout=1800)
        sync_session_outputs(args.session, prefix="minimax_verify", pull=True)
        return
    if profile == "drama3":
        _ensure_bin(DRAMA3_SMOKE_SCRIPT, "verify_drama3.py")
        _colab_exec_file(args.session, DRAMA3_SMOKE_SCRIPT, timeout=1200)
        sync_session_outputs(args.session, prefix="drama3_verify", pull=True)
        return

    _ensure_bin(SMOKE_SCRIPT, "verify_smoke.py")
    _colab_exec_file(args.session, SMOKE_SCRIPT, timeout=300)
    sync_session_outputs(args.session, prefix="smoke_verify", pull=True)


def stop_session(session: str, clean_all: bool):
    """停止指定 Colab 会话，带失败重试与 Endpoint 兜底释放，可选清理全量孤儿实例。"""
    _ensure_bin(COLAB_EXE, "colab.exe")

    target_ep = None
    if COLAB_PY.is_file():
        probe_ep_code = f"""
from colab_cli.common import state
s = state.get_session({session!r})
print(s.endpoint if s and getattr(s, "endpoint", None) else "")
"""
        ep_res = _run([COLAB_PY, "-c", probe_ep_code], check=False, capture=True)
        cand = (ep_res.stdout or "").strip().splitlines()
        if cand and cand[-1].strip():
            target_ep = cand[-1].strip()

    stop_ok = False
    last_rc = 1
    for attempt in range(1, 4):
        res = _run([COLAB_EXE, "stop", "-s", session], check=False)
        last_rc = res.returncode
        if res.returncode == 0:
            stop_ok = True
            break
        print(
            f"提示: 第 {attempt}/3 次 colab stop -s {session} 返回 {res.returncode}，准备重试/兜底释放...",
            flush=True,
        )
        if attempt < 3:
            time.sleep(2)

    if COLAB_PY.is_file() and (clean_all or not stop_ok):
        fallback_code = f"""
import time
from colab_cli.common import state

session_name = {session!r}
target_ep = {target_ep!r}
clean_all = {bool(clean_all)!r}

for attempt in range(1, 4):
    try:
        assignments = state.client.list_assignments()
        if clean_all:
            if not assignments:
                print("无孤儿实例需要清理。", flush=True)
            else:
                for item in assignments:
                    ep = getattr(item, "endpoint", None) or (item.get("endpoint") if isinstance(item, dict) else None)
                    if ep:
                        print(f"正在释放实例: {{ep}}", flush=True)
                        state.client.unassign(ep)
        elif target_ep:
            active_eps = {{
                getattr(item, "endpoint", None) or (item.get("endpoint") if isinstance(item, dict) else None)
                for item in assignments
            }}
            if target_ep in active_eps:
                print(f"兜底释放目标会话实例: {{target_ep}}", flush=True)
                state.client.unassign(target_ep)
            try:
                state.store.remove(session_name)
            except Exception:
                pass
        break
    except Exception as exc:
        if attempt == 3:
            raise
        print(f"释放接口第 {{attempt}}/3 次异常 ({{exc}})，2 秒后重试...", flush=True)
        time.sleep(2)
"""
        fb_res = _run([COLAB_PY, "-c", fallback_code], check=False)
        if fb_res.returncode == 0:
            stop_ok = True

    if not stop_ok and not clean_all:
        raise RuntimeError(f"停止 Colab 会话 '{session}' 失败（colab stop 退出码 {last_rc}）")

    try:
        cmd_status(None)
    except Exception as exc:
        print(f"[警告] 停卡后查询状态异常（实例已触发释放）: {exc}", flush=True)


def cmd_down(args):
    """处理 down 子命令。"""
    stop_session(args.session, args.all)


def cmd_run(args):
    """提交任意本地 ComfyUI Python 出图脚本并强制保障异常抢救同步与 --auto-stop 停卡。"""
    script_path = Path(args.script)
    if not script_path.is_absolute():
        cwd_cand = (Path.cwd() / script_path).resolve()
        script_path = cwd_cand if cwd_cand.is_file() else (PACKAGE_DIR / script_path).resolve()
    if not script_path.is_file():
        raise FileNotFoundError(f"未找到待执行脚本: {script_path}")

    norm_prefix = _normalize_prefix(args.prefix)
    manifest_out_dir = f"{REMOTE_OUT_ROOT}/{norm_prefix}" if norm_prefix else REMOTE_OUT_ROOT
    deps = _collect_local_py_deps(script_path)
    if deps:
        print(f"[依赖扫描] 自动随同上传 {len(deps)} 个同级依赖模块: {', '.join(sorted(deps))}", flush=True)

    ran_remote = False
    synced = False
    try:
        ran_remote = True
        _colab_exec_file(
            args.session,
            script_path,
            timeout=300,
            remote_file=f"/content/{script_path.name}",
            extra_files=deps,
            manifest_out_dir=manifest_out_dir,
        )

        if not args.no_wait:
            wait_ns = argparse.Namespace(session=args.session, timeout=args.wait_timeout)
            cmd_wait(wait_ns)

        if not args.no_sync:
            sync_session_outputs(args.session, prefix=args.prefix, pull=args.pull)
            synced = True
    except BaseException as exc:
        print(f"[任务中断] cmd_run 执行异常 ({type(exc).__name__}: {exc})", file=sys.stderr, flush=True)
        if ran_remote and not args.no_sync and not synced:
            print("[抢救同步] 尝试抢救同步云端已生成的输出与清单...", flush=True)
            try:
                sync_session_outputs(args.session, prefix=args.prefix, pull=args.pull)
            except Exception as sync_exc:
                print(f"[抢救跳过] 抢救同步未成功 ({sync_exc})", file=sys.stderr, flush=True)
        raise
    finally:
        if args.auto_stop:
            print("=== 触发 --auto-stop：强制释放 Colab 实例停止计费 ===", flush=True)
            try:
                stop_session(args.session, clean_all=False)
            except BaseException as stop_exc:
                print(
                    f"[严重告警] --auto-stop 自动释放实例失败，请立即手动停卡: {stop_exc}",
                    file=sys.stderr,
                    flush=True,
                )
                if sys.exc_info()[0] is None:
                    raise


def build_parser() -> argparse.ArgumentParser:
    """构建中文命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description="Google Colab + rclone 本机总控工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_loras = sub.add_parser("sync-loras", help="核对并上传本地 LoRA 到 Google Drive（零 CU 消耗）")
    p_loras.add_argument("--src", default=None, help="本地 LoRA 目录（默认 ./loras 或环境变量 COLAB_LORA_DIR）")
    p_loras.set_defaults(func=cmd_sync_loras)

    p_d3_refs = sub.add_parser("sync-drama3-refs", help="上传精选 Drama 3 参考音频 (.wav) 到 Google Drive（零 CU 消耗）")
    p_d3_refs.add_argument("--src", default=None, help="本地参考音频目录（默认 ./drama3_refs 或环境变量 COLAB_DRAMA3_REFS_DIR）")
    p_d3_refs.set_defaults(func=cmd_sync_drama3_refs)

    p_status = sub.add_parser("status", help="查看当前活跃 Colab 会话及 CU 积分余量/消耗速率")
    p_status.set_defaults(func=cmd_status)

    p_up = sub.add_parser("up", help="创建 GPU 实例、上传 rclone 配置并部署指定生态（ComfyUI 或 Qwen3.8 LLM）")
    p_up.add_argument("--session", "-s", default="comfy", help="Colab 会话名称（默认 comfy）")
    p_up.add_argument("--profile", default="anima", choices=["anima", "minimax", "drama3", "qwen_llm"], help="部署工作流生态配置（默认 anima，可选 minimax / drama3 / qwen_llm）")
    p_up.add_argument("--gpu", default=None, choices=["T4", "L4", "G4", "A100", "H100"], help="GPU 型号（anima 默认 L4，minimax / drama3 / qwen_llm 默认 G4）")
    p_up.add_argument("--high-mem", action="store_true", help="申请高内存实例形态（如 A100 80GB High-RAM）")
    p_up.add_argument("--recreate", action="store_true", help="若目标会话已存在，先释放旧实例与遗留分配再按新显卡规格重新创建")
    p_up.add_argument("--minimax-lite", action="store_true", help="仅安装 MiniMax 核心生图 + FL2VA 视频（跳过 R2V 与扩写器）")
    p_up.add_argument("--no-r2v", action="store_true", help="MiniMax 模式下跳过 Ref2VA 多图参考视频模型（省 ~21.4 GiB）")
    p_up.add_argument("--no-pe", action="store_true", help="MiniMax 模式下跳过 Qwen3.5-9B 提示词扩写模型（省 ~6.8 GiB）")
    p_up.add_argument("--llm-quant", default="UD-Q4_K_XL", choices=["UD-Q4_K_XL", "UD-IQ4_XS", "IQ3_XXS"], help="Qwen3.8 LLM 量化档位（默认 UD-Q4_K_XL）")
    p_up.add_argument("--llm-ctx", type=int, default=32768, help="Qwen3.8 LLM 上下文窗口长度（默认 32768）")
    p_up.add_argument("--no-mmproj", action="store_true", help="Qwen3.8 LLM 模式下跳过加载多模态视觉投影器（省 ~0.9 GiB）")
    p_up.add_argument("--no-webui", action="store_true", help="Qwen3.8 LLM 模式下关闭内置聊天 WebUI（传 --no-ui）")
    p_up.add_argument("--gdrive-cache", action="store_true", help="Qwen3.8 LLM 模式下启用 Google Drive 量化分片缓存")
    p_up.add_argument("--llm-api-key", default="auto", help="Qwen3.8 LLM 接口 Bearer 密钥（默认 auto 自动生成高熵密钥）")
    p_up.set_defaults(func=cmd_up)

    p_tunnel = sub.add_parser("tunnel", help="查看或重启远端 ComfyUI / LLM Cloudflared 隧道并验证连通性")
    p_tunnel.add_argument("--session", "-s", default="comfy", help="Colab 会话名称（默认 comfy）")
    p_tunnel.add_argument("--restart", action="store_true", help="强制重启 cloudflared 并刷新公网隧道地址")
    p_tunnel.set_defaults(func=cmd_tunnel)

    p_smoke = sub.add_parser("smoke", help="运行冒烟验证并同步/拉回结果")
    p_smoke.add_argument("--session", "-s", default="comfy", help="Colab 会话名称（默认 comfy）")
    p_smoke.add_argument("--profile", default="anima", choices=["anima", "minimax", "drama3", "qwen_llm"], help="验证配置（默认 anima，可选 minimax / drama3 / qwen_llm）")
    p_smoke.set_defaults(func=cmd_smoke)

    p_wait = sub.add_parser("wait", help="等待远端 ComfyUI 队列清空后再返回")
    p_wait.add_argument("--session", "-s", default="comfy", help="Colab 会话名称（默认 comfy）")
    p_wait.add_argument("--timeout", type=int, default=7200, help="最长等待秒数（默认 7200）")
    p_wait.set_defaults(func=cmd_wait)

    p_sync = sub.add_parser("sync", help="同步远端输出图到 Google Drive，可选打包拉回本机")
    p_sync.add_argument("--session", "-s", default="comfy", help="Colab 会话名称（默认 comfy）")
    p_sync.add_argument("--prefix", default=None, help="仅同步 /root/comfy_out 下的指定子目录，如 batch1")
    p_sync.add_argument("--pull", action="store_true", help="同时将对应输出打包下载解压至本机 _downloads/")
    p_sync.set_defaults(func=cmd_sync)

    p_run = sub.add_parser("run", help="提交本地出图脚本到远端运行，支持等队列+同步+拉回+自动停卡")
    p_run.add_argument("script", help="本地脚本路径（相对工作区或绝对路径）")
    p_run.add_argument("--session", "-s", default="comfy", help="Colab 会话名称（默认 comfy）")
    p_run.add_argument("--prefix", default=None, help="输出子目录前缀（用于精简同步与下载）")
    p_run.add_argument("--wait-timeout", type=int, default=7200, help="队列等待超时秒数（默认 7200）")
    p_run.add_argument("--no-wait", action="store_true", help="提交入队后立即返回，不等待队列跑完")
    p_run.add_argument("--no-sync", action="store_true", help="跑完后不自动同步至 Google Drive")
    p_run.add_argument("--pull", action="store_true", help="同步后同时打包下载解压到本机 _downloads/")
    p_run.add_argument("--auto-stop", action="store_true", help="同步完成后自动关闭实例停止计费")
    p_run.set_defaults(func=cmd_run)

    p_down = sub.add_parser("down", help="释放 Colab 实例并查看积分速率是否归零")
    p_down.add_argument("--session", "-s", default="comfy", help="Colab 会话名称（默认 comfy）")
    p_down.add_argument("--all", action="store_true", help="同时释放包括网页端遗留实例在内的全部会话")
    p_down.set_defaults(func=cmd_down)

    return parser


def main():
    """CLI 入口。"""
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
