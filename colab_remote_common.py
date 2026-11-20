"""Colab 远端实例部署与冒烟验证共享基础库。

供 `colab_bootstrap_*.py`、`verify_*.py` 与 `colab_ctl.py` 远端任务复用，由 `collect_local_py_deps`
自动打包随主脚本上传至 `/content/`，消除多份自举脚本间的桥接扩展重复拷贝与工具链漂移。
"""

from __future__ import annotations

import contextlib
import http.client
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Any

COMFY_DIR = Path(os.environ.get("COMFY_DIR", "/content/ComfyUI"))
COMFY_OUT = Path(os.environ.get("COMFY_OUT", "/root/comfy_out"))
COMFY_DB = Path(os.environ.get("COMFY_DB", "/root/comfy_db"))
COMFY_LOG = Path(os.environ.get("COMFY_LOG", "/root/comfy_colab.log"))
COMFY_API = os.environ.get("COMFY", "http://127.0.0.1:8188").rstrip("/")
RCLONE_CONF = Path("/tmp/rclone.conf")

CLOUDFLARED_BIN = Path("/usr/local/bin/cloudflared")
CLOUDFLARED_LOG = Path("/root/cloudflared.log")
TUNNEL_URL_FILE = Path("/root/comfy_tunnel_url.txt")
LLM_TUNNEL_URL_FILE = Path("/root/llm_tunnel_url.txt")
SERVICE_META_FILE = Path("/root/colab_service_meta.json")
AUTH_KEY_FILE_DEFAULT = Path("/root/comfy_auth_key")
LLAMA_API_KEY_FILE = Path("/root/llama_api_key")

BRIDGE_NODE_FILENAME = "colab_pet_bridge_node.py"
DIRECT_CANVAS_JS_FILENAME = "colab_pet_direct_canvas.js"
DEFAULT_TUNNEL_URL_RE = re.compile(
    r"https://(?!api\.)[a-z0-9-]+\.trycloudflare\.com(?![a-z0-9.-])"
)
TORCH_SKIP_RE = re.compile(r"^(torch|torchvision|torchaudio)([=<>!~\[]|$)", re.IGNORECASE)

def load_companion_text(filename: str, base_dir: Path | None = None) -> str:
    """从同目录、Drama 3 工作流子目录或当前工作目录读取伴随单源脚本/资产文本。"""
    anchors = [base_dir] if base_dir is not None else []
    mod_file = globals().get("__file__")
    if mod_file:
        anchors.append(Path(mod_file).resolve().parent)
    anchors.append(Path.cwd())
    for root in anchors:
        root_p = Path(root)
        for cand in (root_p / filename, root_p / "workflows" / "drama3" / filename):
            if cand.is_file():
                return cand.read_text(encoding="utf-8").strip()
    return ""


def safe_extract_tar_gz(tar_path: Path, dest_dir: Path) -> None:
    """安全解压 tar.gz 归档至指定目录，防御路径穿越并保留可执行权限与同级安全符号链接。"""
    dest_root = dest_dir.resolve()
    dest_root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tf:
        for member in tf.getmembers():
            posix_p = PurePosixPath(member.name)
            if posix_p.is_absolute() or ".." in posix_p.parts:
                raise RuntimeError(f"归档中存在越界路径，拒绝解压: {member.name}")
            member_resolved = (dest_root / member.name).resolve()
            if member_resolved != dest_root and dest_root not in member_resolved.parents:
                raise RuntimeError(f"归档中存在越界路径，拒绝解压: {member.name}")
            if member.issym() or member.islnk():
                link_p = PurePosixPath(member.linkname)
                if link_p.is_absolute() or ".." in link_p.parts:
                    raise RuntimeError(f"归档符号链接存在越界目标，拒绝解压: {member.linkname}")
                link_base = dest_root if member.islnk() else (dest_root / member.name).parent
                link_target = (link_base / member.linkname).resolve()
                if link_target != dest_root and dest_root not in link_target.parents:
                    raise RuntimeError(f"归档符号链接指向越界目标，拒绝解压: {member.linkname}")
        if sys.version_info >= (3, 12):
            tf.extractall(dest_root, filter="tar")
        else:
            tf.extractall(dest_root)


def load_direct_canvas_js(base_dir: Path | None = None) -> str:
    """从同目录 `colab_pet_direct_canvas.js` 读取直铺画布前端扩展源码。"""
    return load_companion_text(DIRECT_CANVAS_JS_FILENAME, base_dir)


DIRECT_CANVAS_JS = load_direct_canvas_js()


def env_flag(name: str, default: bool) -> bool:
    """读取布尔环境变量。"""
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() not in ("0", "false", "no", "off", "")


def mask_secret(val: str) -> str:
    """生成安全脱敏密钥摘要，禁止在日志中暴露完整密钥。"""
    s = str(val or "").strip()
    if len(s) <= 10:
        return "***"
    return f"{s[:6]}...{s[-4:]}"


def safe_cmd_line(cmd: list[str]) -> str:
    """安全获取外部工具或 Git 仓库的单行版本摘要。"""
    # 可选版本探测失败（未安装或超时）时安全回退空串
    with contextlib.suppress(OSError, subprocess.SubprocessError):
        res = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=8, check=False)
        if res.returncode == 0:
            lines = [ln.strip() for ln in (res.stdout or "").splitlines() if ln.strip()]
            return lines[0] if lines else ""
    return ""


def write_private_text(path: Path, text: str) -> None:
    """以 `0o600` 私密权限原子写入文本文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_p = path.with_suffix(path.suffix + ".tmp")
    fd = os.open(str(tmp_p), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fp:
            fp.write(text)
    except Exception:
        tmp_p.unlink(missing_ok=True)
        raise
    try:
        os.chmod(tmp_p, 0o600)
    except OSError:
        pass
    tmp_p.replace(path)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def write_private_json(path: Path, data: Any) -> None:
    """以 `0o600` 私密权限原子写入格式化 JSON。"""
    write_private_text(path, json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def read_json_dict(path: Path) -> dict[str, Any]:
    """安全读取 JSON 字典，文件缺失或损坏时返回空字典。"""
    if not path.is_file():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def load_pet_bridge_init_py(base_dir: Path | None = None) -> str:
    """从同目录 `colab_pet_bridge_node.py` 读取单源服务端桥接中间件完整源码。"""
    return load_companion_text(BRIDGE_NODE_FILENAME, base_dir)


def install_pet_canvas_bridge(comfy_dir: Path = COMFY_DIR, base_dir: Path | None = None) -> None:
    """在 ComfyUI `custom_nodes` 安装桌宠直铺画布与公网鉴权桥接扩展。"""
    bridge_src = load_pet_bridge_init_py(base_dir)
    if not bridge_src:
        raise FileNotFoundError(f"未找到桥接扩展标准文件 {BRIDGE_NODE_FILENAME}")
    bridge_dir = comfy_dir / "custom_nodes" / "colab_pet_bridge"
    js_dir = bridge_dir / "js"
    js_dir.mkdir(parents=True, exist_ok=True)
    (bridge_dir / "__init__.py").write_text(bridge_src + "\n", encoding="utf-8")
    canvas_js = load_direct_canvas_js(base_dir) or DIRECT_CANVAS_JS
    if not canvas_js:
        raise FileNotFoundError(f"未找到直铺画布前端脚本 {DIRECT_CANVAS_JS_FILENAME}")
    (js_dir / "direct_canvas.js").write_text(canvas_js + "\n", encoding="utf-8")
    print(f"已安装桌宠直铺画布桥接扩展: {bridge_dir}", flush=True)


def ensure_comfy_auth_key(key_file: Path | None = None) -> str:
    """确保 ComfyUI 公网鉴权密钥存在（缺失时自动生成高熵 `cpk-...` 并 0600 落盘，杜绝无密钥裸奔）。"""
    target = Path(key_file or os.environ.get("COMFY_AUTH_KEY_FILE") or str(AUTH_KEY_FILE_DEFAULT))
    if target.is_file():
        raw = target.read_text(encoding="utf-8", errors="ignore").strip()
        if raw:
            key = raw.splitlines()[0].strip()
            os.environ["COMFY_AUTH_KEY_FILE"] = str(target)
            return key
    meta = read_json_dict(SERVICE_META_FILE)
    meta_key = str(meta.get("comfy_api_key") or "").strip()
    key = meta_key if meta_key else f"cpk-{secrets.token_urlsafe(24)}"
    write_private_text(target, key + "\n")
    os.environ["COMFY_AUTH_KEY_FILE"] = str(target)
    return key


def run_cmd(
    cmd: list[Any],
    *,
    check: bool = True,
    cwd: Any = None,
    capture: bool = False,
    env: dict[str, str] | None = None,
    redact_secret: str = "",
) -> subprocess.CompletedProcess[str]:
    """以非交互标准输入模式流式执行外部命令并支持敏感串脱敏。"""
    cmd_list = [str(x) for x in cmd]
    shown = [(mask_secret(x) if redact_secret and x == redact_secret else x) for x in cmd_list]
    print(f"[执行] {' '.join(shown)}", flush=True)
    if capture:
        return subprocess.run(
            cmd_list,
            cwd=str(cwd) if cwd else None,
            check=check,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
    proc = subprocess.Popen(
        cmd_list,
        cwd=str(cwd) if cwd else None,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=env,
    )
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            if redact_secret and redact_secret in line:
                line = line.replace(redact_secret, mask_secret(redact_secret))
            print(line, end="", flush=True)
    finally:
        proc.stdout.close()
    rc = proc.wait()
    if check and rc != 0:
        raise subprocess.CalledProcessError(rc, shown)
    return subprocess.CompletedProcess(shown, rc)


def download_with_urllib(url: str, dst: Path, retries: int = 3) -> None:
    """使用标准库 urllib 分块流式下载并原子替换（8 MiB 缓冲）。"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_suffix(dst.suffix + ".part")
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                total = int(resp.headers.get("Content-Length") or 0)
                done = 0
                last_log = 0.0
                with open(part, "wb") as f:
                    while True:
                        chunk = resp.read(8 * 1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
                        done += len(chunk)
                        now = time.monotonic()
                        if total > 0 and now - last_log >= 5.0:
                            pct = done * 100.0 / total
                            print(
                                f"  [urllib] {dst.name}: {done / (1024**3):.2f}/{total / (1024**3):.2f} GiB ({pct:.1f}%)",
                                flush=True,
                            )
                            last_log = now
            part.replace(dst)
            return
        except Exception as exc:
            last_err = exc
            part.unlink(missing_ok=True)
            print(f"[下载重试] {dst.name} 第 {attempt}/{retries} 次失败: {exc}", flush=True)
            if attempt < retries:
                time.sleep(3 * attempt)
    raise RuntimeError(f"下载失败: {url} ({last_err})")


def is_complete_file(
    dst: Path,
    *,
    exact_bytes: int | None = None,
    min_bytes: int = 1,
) -> bool:
    """校验目标文件完整落盘：同名 `.aria2` 控制文件存在即视为未完成坏片。"""
    aria2_ctrl = dst.parent / f"{dst.name}.aria2"
    if aria2_ctrl.exists() or not dst.is_file():
        return False
    sz = dst.stat().st_size
    if exact_bytes is not None:
        return sz == exact_bytes
    return sz >= min_bytes


def download_and_verify(
    url: str,
    dst: Path,
    *,
    exact_bytes: int | None = None,
    min_bytes: int = 1,
    summary_interval: int = 5,
    chunk_size_m: str = "8M",
) -> None:
    """高速下载并校验文件（检测并拦截 `.aria2` 预分配空洞残片，优先 16 线程 aria2c 失败回退 urllib）。"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    aria2_ctrl = dst.parent / f"{dst.name}.aria2"
    if is_complete_file(dst, exact_bytes=exact_bytes, min_bytes=min_bytes):
        sz_gib = dst.stat().st_size / (1024 ** 3)
        lbl = f"{sz_gib:.2f} GiB" if sz_gib >= 0.05 else f"{dst.stat().st_size} B"
        print(f"  [已就绪] {dst.name} ({lbl})", flush=True)
        return

    if aria2_ctrl.exists():
        print(f"  [断点续传] 检测到未完成的 .aria2 控制文件，继续校验下载: {dst.name}", flush=True)
    elif dst.exists():
        # 无 .aria2 控制文件但字节不足/越界，直接清理避免污染续传
        dst.unlink(missing_ok=True)

    print(f"  [开始下载] {dst.name} <- {url}", flush=True)
    t0 = time.monotonic()
    used_aria = False
    aria_bin = shutil.which("aria2c")
    if aria_bin:
        res = run_cmd(
            [
                aria_bin,
                "--console-log-level=notice",
                "--show-console-readout=false",
                f"--summary-interval={summary_interval}",
                "-x", "16",
                "-s", "16",
                "-k", chunk_size_m,
                "--allow-overwrite=true",
                "-c",
                "-d", str(dst.parent),
                "-o", dst.name,
                url,
            ],
            check=False,
        )
        used_aria = res.returncode == 0 and is_complete_file(
            dst, exact_bytes=exact_bytes, min_bytes=min_bytes
        )

    if not used_aria:
        # 回退 urllib 前彻底删除半截文件与 .aria2 残片，避免与单线程流式下载混杂
        aria2_ctrl.unlink(missing_ok=True)
        dst.unlink(missing_ok=True)
        download_with_urllib(url, dst)

    if not is_complete_file(dst, exact_bytes=exact_bytes, min_bytes=min_bytes):
        actual = dst.stat().st_size if dst.is_file() else -1
        expect_desc = f"=={exact_bytes}" if exact_bytes is not None else f">={min_bytes}"
        raise RuntimeError(
            f"模型文件校验失败: {dst.name} 实际={actual} B, 预期{expect_desc} B, aria2_ctrl={aria2_ctrl.exists()}"
        )

    dt = max(0.1, time.monotonic() - t0)
    sz = dst.stat().st_size
    print(f"  [完成] {dst.name}: {sz / (1024**3):.2f} GiB ({dt:.1f}s, {sz / (1024**2) / dt:.1f} MiB/s)", flush=True)


def ensure_tools() -> None:
    """非交互确保远端实例具备 rclone、aria2c 与 cloudflared 基础工具。"""
    print("=== [2/7] 检查 rclone、aria2c 与 cloudflared ===", flush=True)
    rclone_bin = Path("/usr/local/bin/rclone")
    if not shutil.which("rclone") and not rclone_bin.exists():
        zip_path = Path("/tmp/rclone.zip")
        url = "https://downloads.rclone.org/rclone-current-linux-amd64.zip"
        try:
            download_with_urllib(url, zip_path, retries=2)
        except Exception:
            run_cmd(["bash", "-lc", "apt-get update -qq && apt-get install -y -qq rclone aria2"], check=False)
        if zip_path.is_file():
            with zipfile.ZipFile(zip_path, "r") as zf:
                members = [m for m in zf.namelist() if m.endswith("/rclone") and m.count("/") == 1]
                if not members:
                    raise RuntimeError("rclone 压缩包内未找到可执行二进制文件")
                with zf.open(members[0]) as src, open(rclone_bin, "wb") as dst:
                    shutil.copyfileobj(src, dst)
            os.chmod(rclone_bin, 0o755)
            zip_path.unlink(missing_ok=True)
        if not shutil.which("rclone") and not rclone_bin.is_file():
            raise RuntimeError("rclone 安装失败")

    if not shutil.which("aria2c"):
        try:
            run_cmd(["bash", "-lc", "apt-get update -qq && apt-get install -y -qq aria2"], check=False)
        except Exception as exc:
            print(f"警告: aria2c 安装失败，后续将回退 Python 流式下载 ({exc})", flush=True)

    if not shutil.which("cloudflared") and not CLOUDFLARED_BIN.exists():
        cf_url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
        cf_tmp = CLOUDFLARED_BIN.with_suffix(".part")
        download_with_urllib(cf_url, cf_tmp)
        cf_tmp.replace(CLOUDFLARED_BIN)
        os.chmod(CLOUDFLARED_BIN, 0o755)

    print(
        f"工具就绪: rclone={shutil.which('rclone') or rclone_bin}, aria2c={shutil.which('aria2c')}, cloudflared={shutil.which('cloudflared') or CLOUDFLARED_BIN}",
        flush=True,
    )


def sync_git_repo(repo_url: str, dst_dir: Path, ref: str = "", *, pull_if_exists: bool = False) -> None:
    """幂等克隆或检出指定版本的 Git 仓库。"""
    if not (dst_dir / ".git").exists() and not (dst_dir / "main.py").is_file() and not (dst_dir / "__init__.py").is_file():
        run_cmd(["git", "clone", "--depth", "1", repo_url, str(dst_dir)])
    elif pull_if_exists and not ref:
        run_cmd(["git", "-C", str(dst_dir), "pull", "--ff-only"], check=False)
    if ref:
        print(f"按指定版本锁定仓库 ({dst_dir.name}): {ref}", flush=True)
        res = run_cmd(["git", "-C", str(dst_dir), "fetch", "--depth", "1", "origin", ref], check=False)
        if res.returncode == 0:
            run_cmd(["git", "-C", str(dst_dir), "checkout", "--force", "FETCH_HEAD"])
        else:
            run_cmd(["git", "-C", str(dst_dir), "checkout", "--force", ref])


def install_requirements_no_torch(req_src: Path, tag: str = "comfy") -> list[str]:
    """过滤 torch/torchvision/torchaudio 后安装依赖，保护 Colab 预装 CUDA。"""
    if not req_src.is_file():
        raise FileNotFoundError(f"未找到依赖清单: {req_src}")
    kept, skipped = [], []
    for raw in req_src.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if TORCH_SKIP_RE.match(line):
            skipped.append(line)
            continue
        kept.append(raw)
    tmp_req = Path(f"/tmp/reqs_notorch_{tag}.txt")
    tmp_req.write_text("\n".join(kept) + "\n", encoding="utf-8")
    uv_bin = shutil.which("uv")
    if kept:
        if uv_bin:
            run_cmd([uv_bin, "pip", "install", "--system", "-q", "-r", str(tmp_req)])
        else:
            run_cmd([sys.executable, "-m", "pip", "install", "-q", "-r", str(tmp_req)])
    return skipped


def ensure_comfyui_repo(
    repo_url: str = "https://github.com/Comfy-Org/ComfyUI.git",
    *,
    pull_if_no_ref: bool = True,
    require_comfy_kitchen: bool = False,
    comfy_dir: Path = COMFY_DIR,
) -> None:
    """准备 ComfyUI 仓库、保护 CUDA 安装依赖并预装桥接扩展。"""
    comfy_ref = (os.environ.get("COMFY_GIT_REF") or "").strip()
    sync_git_repo(repo_url, comfy_dir, ref=comfy_ref, pull_if_exists=pull_if_no_ref)
    skipped = install_requirements_no_torch(comfy_dir / "requirements.txt", tag="comfy")
    if skipped:
        print(f"已跳过覆盖预装 CUDA 包: {skipped}", flush=True)
    if require_comfy_kitchen:
        import comfy_kitchen  # noqa: F401
        print(f"comfy_kitchen 已就绪: {getattr(comfy_kitchen, '__version__', 'ok')}", flush=True)
    install_pet_canvas_bridge(comfy_dir)


def _origin_key_common(url: str) -> str:
    """提取标准化的协议+主机+端口源站标识，用于校验跨源重定向。"""
    try:
        parsed = urllib.parse.urlparse(str(url or "").strip())
        scheme = parsed.scheme.lower()
        host = (parsed.hostname or "").lower()
        if not scheme or not host:
            return ""
        port = parsed.port or (443 if scheme == "https" else 80)
        return f"{scheme}://{host}:{port}"
    except Exception:
        return ""


class _NoCrossOriginRedirectHandler(urllib.request.HTTPRedirectHandler):
    """禁止 HTTP 301/302/303/307/308 跳转至异源地址，防止认证头随重定向外泄。"""

    def redirect_request(self, req: urllib.request.Request, fp: Any, code: int, msg: str, headers: Any, newurl: str):
        req_origin = _origin_key_common(req.full_url)
        resolved = urllib.parse.urljoin(req.full_url, str(newurl or ""))
        next_origin = _origin_key_common(resolved)
        if not req_origin or not next_origin or next_origin != req_origin:
            raise urllib.error.HTTPError(
                req.full_url,
                code,
                f"安全拦截：拒绝跨源重定向至 {next_origin or resolved}",
                headers,
                fp,
            )
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _safe_urlopen(req_or_url: urllib.request.Request | str, *, timeout: float = 4.0) -> Any:
    """使用同源重定向保护 opener 发起 HTTP 请求。"""
    opener = urllib.request.build_opener(_NoCrossOriginRedirectHandler())
    return opener.open(req_or_url, timeout=timeout)


def probe_comfy_ready(timeout: float = 3.0) -> str | None:
    """探测本机 ComfyUI /system_stats 是否可响应。"""
    try:
        with _safe_urlopen(f"{COMFY_API}/system_stats", timeout=timeout) as resp:
            if resp.status == 200:
                return resp.read().decode("utf-8", errors="ignore")
    except Exception:
        return None
    return None


def probe_bridge_version(timeout: float = 3.0) -> int:
    """探测运行中 ComfyUI 已加载的桌宠桥接节点版本号。"""
    # 服务未启动或旧版无此路由时安全返回 0 以触发安装/重载
    with contextlib.suppress(OSError, ValueError, TypeError):
        with _safe_urlopen(f"{COMFY_API}/colab_pet/ping", timeout=timeout) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode("utf-8", errors="ignore") or "{}")
                return int(data.get("version") or 0)
    return 0


def _pkill_quiet(pattern: str) -> None:
    """静默按匹配模式终止后台进程（不中断后续清理流程）。"""
    with contextlib.suppress(OSError, subprocess.SubprocessError):
        subprocess.run(["pkill", "-f", pattern], check=False, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def terminate_proc(proc: Any) -> None:
    """安全终止尚未通过安全校验的后台子进程（兼容真实 Popen 与测试替身对象）。"""
    if proc is None:
        return
    with contextlib.suppress(OSError, subprocess.SubprocessError, ValueError, RuntimeError):
        poll_fn = getattr(proc, "poll", None)
        if callable(poll_fn) and poll_fn() is not None:
            return
        term_fn = getattr(proc, "terminate", None)
        if callable(term_fn):
            term_fn()
        wait_fn = getattr(proc, "wait", None)
        if callable(wait_fn):
            try:
                wait_fn(timeout=5)
            except TypeError:
                wait_fn()
            except (OSError, subprocess.SubprocessError, ValueError, RuntimeError):
                kill_fn = getattr(proc, "kill", None)
                if callable(kill_fn):
                    kill_fn()


_terminate_proc = terminate_proc


def _probe_http_status_and_body(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout: float = 3.0,
) -> tuple[int | None, bytes, str]:
    """使用同源重定向保护 opener 发起单次探测，返回 `(status, body_bytes, err_msg)`。"""
    req = urllib.request.Request(url, headers=headers or {})
    try:
        with _safe_urlopen(req, timeout=timeout) as resp:
            return int(getattr(resp, "status", 200)), resp.read() or b"", ""
    except urllib.error.HTTPError as exc:
        st = int(exc.code)
        with contextlib.suppress(OSError):
            exc.close()
        return st, b"", ""
    except (urllib.error.URLError, http.client.HTTPException, OSError, TimeoutError, ValueError) as exc:
        return None, b"", str(exc)


def _check_two_way_bearer(
    url: str,
    key: str,
    *,
    extra_headers: dict[str, str] | None = None,
    timeout: float = 3.0,
    title: str = "服务",
) -> tuple[bool, bytes, str]:
    """双向 Bearer 鉴权核验：匿名请求必拒 (401/403)，合法 Bearer 密钥必通 (200) 并返回正文。"""
    hdrs = dict(extra_headers or {})
    unauth_st, _, err_u = _probe_http_status_and_body(url, headers=hdrs or None, timeout=timeout)
    if unauth_st is None:
        return False, b"", f"探测 {title} 未授权响应异常: {err_u}"
    if unauth_st == 200:
        return False, b"", f"{title} {url} 未携带 Authorization 仍返回 HTTP 200，鉴权未生效"
    if unauth_st not in (401, 403):
        return False, b"", f"{title} {url} 对未授权请求返回异常状态码 HTTP {unauth_st}（预期 401/403）"
    if not key:
        return True, b"", ""
    hdrs["Authorization"] = f"Bearer {key}"
    auth_st, raw_auth, err_a = _probe_http_status_and_body(url, headers=hdrs, timeout=timeout)
    if auth_st is None:
        return False, b"", f"探测 {title} 合法密钥响应异常: {err_a}"
    if auth_st != 200:
        return False, b"", f"{title} {url} 携带 Bearer 密钥校验未通过（实际 HTTP {auth_st}，预期 200）"
    return True, raw_auth, ""


def verify_local_comfy_auth_enforced(
    auth_key: str | None = None,
    timeout: float = 3.0,
    api_url: str | None = None,
) -> tuple[bool, str]:
    """核实本机 ComfyUI v3 公网鉴权网关已真实生效，未授权公网代理请求必被拦截且合法密钥可通行。"""
    base_api = str(api_url or COMFY_API).strip().rstrip("/")
    if not api_url or base_api == COMFY_API.rstrip("/"):
        bridge_ver = probe_bridge_version(timeout=timeout)
    else:
        st_p, raw_p, _ = _probe_http_status_and_body(f"{base_api}/colab_pet/ping", timeout=timeout)
        try:
            bridge_ver = int(json.loads(raw_p.decode("utf-8", errors="ignore") or "{}").get("version") or 0) if st_p == 200 else 0
        except Exception:
            bridge_ver = 0
    if bridge_ver < 3:
        return False, f"ComfyUI 桥接扩展版本不足或未加载（version={bridge_ver} < 3）"

    eff_key = ensure_comfy_auth_key() if auth_key is None else str(auth_key).strip()
    ok, _, msg = _check_two_way_bearer(
        f"{base_api}/system_stats",
        eff_key,
        extra_headers={"X-Forwarded-For": "203.0.113.1"},
        timeout=timeout,
        title="ComfyUI",
    )
    return (True, "ComfyUI 公网鉴权网关校验通过") if ok else (False, msg)


def load_local_llm_auth_key(explicit_key: str | None = None) -> str:
    """按优先级从显式参数、本地私有密钥文件及服务契约加载 LLM API Key。"""
    cand = str(explicit_key or "").strip()
    if cand:
        return cand
    if LLAMA_API_KEY_FILE.is_file():
        with contextlib.suppress(OSError):
            raw = LLAMA_API_KEY_FILE.read_text(encoding="utf-8", errors="ignore").strip()
            if raw:
                return raw.splitlines()[0].strip()
    meta = read_json_dict(SERVICE_META_FILE)
    return str(meta.get("api_key") or "").strip()


def verify_local_llm_auth_enforced(
    port: int = 18080,
    api_key: str | None = None,
    timeout: float = 3.0,
) -> tuple[bool, str]:
    """核实本地 LLM 网关（/v1/models）已启用 API Key 鉴权且合法 Bearer 密钥可通行。"""
    key = load_local_llm_auth_key(None) if api_key is None else str(api_key).strip()
    if not key:
        return False, "未找到有效 LLM API Key，拒绝无鉴权暴露公网隧道"
    ok, _, msg = _check_two_way_bearer(
        f"http://127.0.0.1:{int(port)}/v1/models", key, timeout=timeout, title="LLM 网关"
    )
    return (True, "LLM 网关 API Key 鉴权校验通过") if ok else (False, msg)


def start_comfyui_and_wait(
    *,
    db_url: str | None = None,
    extra_args: list[str] | None = None,
    wait_timeout_sec: int = 120,
    extra_ready_probe: Callable[[], str | None] | None = None,
) -> None:
    """非阻塞守护启动 ComfyUI（自动确保随机鉴权密钥存在且严格校验公网认证中间件生效）。"""
    print("=== [6/7] 启动 ComfyUI 服务并等待就绪 ===", flush=True)
    COMFY_OUT.mkdir(parents=True, exist_ok=True)
    auth_key = ensure_comfy_auth_key()

    reserve_vram_raw = (os.environ.get("COMFY_RESERVE_VRAM") or "").strip()
    ensure_reserve = (os.environ.get("COMFY_ENSURE_RESERVE_VRAM") or "").strip() == "1"
    reserve_val: str | None = None
    if reserve_vram_raw:
        try:
            if float(reserve_vram_raw) > 0:
                reserve_val = reserve_vram_raw
        except ValueError:
            reserve_val = None

    ready_payload = probe_comfy_ready()
    if ready_payload:
        restart_reason = ""
        if probe_bridge_version() < 3:
            restart_reason = "检测到存量 ComfyUI 未启用 v3 公网认证网关，正在平滑重启加载..."
        else:
            auth_ok, auth_reason = verify_local_comfy_auth_enforced(auth_key)
            if not auth_ok:
                restart_reason = f"检测到存量 ComfyUI 公网鉴权网关未生效（{auth_reason}），正在平滑重启加载..."
            elif extra_ready_probe is not None:
                restart_reason = extra_ready_probe() or ""
        if not restart_reason and ensure_reserve and reserve_val:
            pg = subprocess.run(["pgrep", "-af", "main.py.*8188"], stdin=subprocess.DEVNULL, capture_output=True, text=True)
            if "--reserve-vram" not in (pg.stdout or ""):
                restart_reason = f"检测到存量 ComfyUI 未启用 --reserve-vram {reserve_val}（双栖显存保护），正在平滑重启..."
        if not restart_reason:
            print(f"ComfyUI 已在运行中，直接复用。system_stats: {ready_payload[:240]}", flush=True)
            return

        print(restart_reason, flush=True)
        _pkill_quiet("main.py.*8188")
        for _ in range(15):
            if probe_comfy_ready() is None:
                break
            time.sleep(1)

    if db_url is None:
        db_file = COMFY_DIR / "user" / "comfyui.db"
        db_file.parent.mkdir(parents=True, exist_ok=True)
        db_url = f"sqlite:///{db_file}"

    cmd = [
        sys.executable,
        "main.py",
        "--listen", "127.0.0.1",
        "--port", "8188",
        *(extra_args or []),
        "--output-directory", str(COMFY_OUT),
        "--database-url", db_url,
    ]
    if reserve_val:
        cmd.extend(["--reserve-vram", reserve_val])
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    log_f = open(COMFY_LOG, "ab")
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(COMFY_DIR),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    finally:
        log_f.close()
    print(f"已后台拉起 ComfyUI (PID={proc.pid})，日志: {COMFY_LOG}", flush=True)

    steps = max(1, wait_timeout_sec // 2)
    auth_fail_streak = 0
    last_auth_reason = ""
    for idx in range(1, steps + 1):
        ready_payload = probe_comfy_ready()
        if ready_payload:
            auth_ok, auth_reason = verify_local_comfy_auth_enforced(auth_key)
            if auth_ok:
                print(f"ComfyUI 就绪且公网鉴权网关校验通过（耗时约 {idx * 2} 秒）", flush=True)
                print(f"system_stats: {ready_payload[:300]}", flush=True)
                return
            auth_fail_streak += 1
            last_auth_reason = auth_reason
            print(
                f"[警告] ComfyUI /system_stats 已响应但公网鉴权网关尚未通过校验 ({auth_fail_streak}/3): {auth_reason}",
                flush=True,
            )
            if auth_fail_streak >= 3:
                _terminate_proc(proc)
                raise RuntimeError(f"ComfyUI 公网鉴权网关未生效，拒绝暴露公网隧道: {auth_reason}")
        else:
            auth_fail_streak = 0
        time.sleep(2)

    if auth_fail_streak > 0:
        _terminate_proc(proc)
        raise RuntimeError(f"ComfyUI 公网鉴权网关未生效，拒绝暴露公网隧道: {last_auth_reason}")

    tail = COMFY_LOG.read_text(encoding="utf-8", errors="ignore")[-2000:] if COMFY_LOG.is_file() else ""
    raise TimeoutError(f"等待 ComfyUI 就绪超时（{wait_timeout_sec}s），末尾日志:\n{tail}")


def verify_public_tunnel_auth_enforced(
    url: str,
    *,
    kind: str = "comfy",
    auth_key: str = "",
    timeout: float = 5.0,
) -> tuple[bool, str]:
    """对公网隧道地址执行双向鉴权验收（匿名请求必拒 401/403，合法 Bearer 请求必通 200 且结构合法）。"""
    base = str(url or "").strip().rstrip("/")
    if not base.startswith(("https://", "http://127.0.0.1:", "http://localhost:")):
        return False, "非法的公网隧道地址协议"
    norm_kind = "llm" if str(kind or "").strip().lower() == "llm" else "comfy"
    key = (
        (str(auth_key).strip() or load_local_llm_auth_key(None))
        if norm_kind == "llm"
        else (str(auth_key).strip() or ensure_comfy_auth_key())
    )
    if not key:
        return False, f"{norm_kind} 缺少鉴权密钥，拒绝验收公网隧道"

    target = f"{base}/v1/models" if norm_kind == "llm" else f"{base}/system_stats"
    ok, raw_auth, msg = _check_two_way_bearer(
        target, key, extra_headers={"User-Agent": "Mozilla/5.0"}, timeout=timeout, title="公网隧道"
    )
    if not ok:
        return False, msg
    try:
        body = json.loads(raw_auth.decode("utf-8", errors="ignore") or "{}")
    except Exception:
        body = None
    valid_body = isinstance(body, dict) and (
        ("data" in body or "object" in body) if norm_kind == "llm" else ("system" in body or "devices" in body)
    )
    if not valid_body:
        return False, f"公网隧道 {target} 返回内容格式异常"
    return True, f"{norm_kind} 公网隧道双向鉴权验收通过"


def _await_public_tunnel_auth(
    url: str,
    *,
    kind: str,
    auth_key: str,
    abort_fn: Callable[[str], None] | None = None,
    max_attempts: int = 12,
    interval: float = 1.5,
    timeout: float = 4.0,
) -> None:
    """轮询等待公网隧道边缘路由就绪并完成双向鉴权验收；若匿名直通 HTTP 200 则立即中止。"""
    def _fail(msg: str) -> None:
        if abort_fn is not None:
            abort_fn(msg)
        raise RuntimeError(msg)

    last_reason = "未知原因"
    attempts = max(1, int(max_attempts))
    for idx in range(attempts):
        ok, reason = verify_public_tunnel_auth_enforced(url, kind=kind, auth_key=auth_key, timeout=timeout)
        if ok:
            return
        last_reason = reason
        if "HTTP 200" in reason:
            _fail(reason)
        if idx + 1 < attempts:
            time.sleep(interval)
    _fail(f"公网隧道双向鉴权验收未通过: {last_reason}")


def probe_public_tunnel_url(url: str, *, kind: str = "comfy", auth_key: str = "", timeout: float = 4.0) -> bool:
    """通过公网 HTTPS 验证隧道双向鉴权真实可用（未授权 401/403 且合法 Bearer 200，禁止跨源重定向）。"""
    return verify_public_tunnel_auth_enforced(url, kind=kind, auth_key=auth_key, timeout=timeout)[0]


def _verify_local_auth_before_tunnel(
    norm_kind: str,
    local_port: int,
    auth_key: str,
    use_named: bool,
    cohost_other: bool,
    abort_fn: Callable[[str], None],
) -> str:
    """暴露公网隧道前先强校验本机 Comfy/LLM 鉴权网关均已实际生效，返回生效密钥。"""
    if norm_kind == "llm":
        eff_llm_key = load_local_llm_auth_key(auth_key or None)
        llm_ok, llm_reason = verify_local_llm_auth_enforced(port=int(local_port), api_key=eff_llm_key)
        if not llm_ok:
            abort_fn(f"LLM 网关 API Key 鉴权未生效，拒绝暴露公网隧道: {llm_reason}")
        if use_named and cohost_other:
            comfy_ok, comfy_reason = verify_local_comfy_auth_enforced()
            if not comfy_ok:
                abort_fn(f"ComfyUI 公网鉴权网关未生效，拒绝暴露双栖命名隧道: {comfy_reason}")
        return eff_llm_key

    eff_comfy_key = str(auth_key or "").strip() or ensure_comfy_auth_key()
    comfy_api_target = COMFY_API if int(local_port) == 8188 else f"http://127.0.0.1:{int(local_port)}"
    comfy_ok, comfy_reason = verify_local_comfy_auth_enforced(auth_key=eff_comfy_key, api_url=comfy_api_target)
    if not comfy_ok:
        abort_fn(f"ComfyUI 公网鉴权网关未生效，拒绝暴露公网隧道: {comfy_reason}")
    if use_named and cohost_other:
        meta = read_json_dict(SERVICE_META_FILE)
        llm_port = int(meta.get("llm_port") or 18080)
        st_h, _, _ = _probe_http_status_and_body(f"http://127.0.0.1:{llm_port}/health", timeout=2.0)
        if meta.get("kind") == "dual" or st_h == 200:
            llm_ok, llm_reason = verify_local_llm_auth_enforced(
                port=llm_port,
                api_key=str(meta.get("api_key") or "").strip() or None,
            )
            if not llm_ok:
                abort_fn(f"LLM 网关 API Key 鉴权未生效，拒绝暴露双栖命名隧道: {llm_reason}")
    return eff_comfy_key


def _try_reuse_running_cloudflared(
    *,
    use_named: bool,
    local_port: int,
    eff_url: str,
    service_kind: str,
    auth_k: str,
    url_file: Path,
    tunnel_re: re.Pattern[str],
) -> str | None:
    """尝试复用已在运行且通过公网双向鉴权的 cloudflared 隧道，未命中返回 None。"""
    pgrep_pat = "cloudflared tunnel" if use_named else f"cloudflared tunnel.*{local_port}"
    pgrep_res = subprocess.run(
        ["pgrep", "-af" if use_named else "-f", pgrep_pat],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
    )
    if pgrep_res.returncode != 0:
        return None
    proc_out = pgrep_res.stdout or ""
    if use_named and " run" in proc_out and probe_public_tunnel_url(eff_url, kind=service_kind, auth_key=auth_k):
        print(f"复用运行中的命名隧道 ({service_kind}): {eff_url}", flush=True)
        url_file.write_text(eff_url + "\n", encoding="utf-8")
        return eff_url
    if (not use_named) and f"127.0.0.1:{local_port}" in proc_out and url_file.is_file():
        m_cached = tunnel_re.search(url_file.read_text(encoding="utf-8", errors="ignore"))
        if m_cached and probe_public_tunnel_url(m_cached.group(0), kind=service_kind, auth_key=auth_k):
            print(f"复用运行中的 cloudflared 隧道 (:{local_port}): {m_cached.group(0)}", flush=True)
            return m_cached.group(0)
    return None


def _wait_for_cloudflared_tunnel_url(
    proc: subprocess.Popen[Any],
    *,
    use_named: bool,
    eff_url: str,
    eff_tok: str,
    service_kind: str,
    auth_k: str,
    log_path: Path,
    tunnel_re: re.Pattern[str],
) -> str:
    """轮询 cloudflared 日志或命名隧道健康状态，直至获取发布地址或超时。"""

    def _log_tail(redact: str = "") -> str:
        tail = log_path.read_text(encoding="utf-8", errors="ignore")[-1500:] if log_path.is_file() else ""
        return tail.replace(redact, "[REDACTED]") if redact else tail

    if use_named:
        for _ in range(35):
            if proc.poll() is not None:
                raise RuntimeError(f"cloudflared 命名隧道异常退出 (rc={proc.returncode})，日志:\n{_log_tail(eff_tok)}")
            txt = log_path.read_text(encoding="utf-8", errors="ignore") if log_path.is_file() else ""
            if "Registered tunnel connection" in txt or probe_public_tunnel_url(
                eff_url, kind=service_kind, auth_key=auth_k, timeout=2.0
            ):
                return eff_url
            time.sleep(1)
        raise TimeoutError(f"等待 cloudflared 命名隧道注册超时（35s），日志:\n{_log_tail(eff_tok)}")

    for _ in range(30):
        if log_path.is_file():
            m = tunnel_re.search(log_path.read_text(encoding="utf-8", errors="ignore"))
            if m:
                return m.group(0)
        time.sleep(1)
    raise TimeoutError(f"等待 cloudflared 分配隧道地址超时（30s），日志:\n{_log_tail()}")


def start_cloudflared_tunnel(
    *,
    local_port: int = 8188,
    service_kind: str = "comfy",
    cf_token: str = "",
    cf_url: str = "",
    cohost_other: bool = False,
    force_restart: bool = False,
    tunnel_url_file: Path | None = None,
    log_file: Path | None = None,
    tunnel_re: re.Pattern[str] = DEFAULT_TUNNEL_URL_RE,
    auth_key: str = "",
) -> str:
    """统一拉起或复用 `cloudflared` 隧道（本机鉴权与公网双向鉴权均通过后才发布地址）。"""
    eff_tok = (cf_token or "").strip()
    eff_url = (cf_url or "").strip().rstrip("/")
    use_named = bool(eff_tok and eff_url)
    url_file = Path(tunnel_url_file or (LLM_TUNNEL_URL_FILE if (service_kind == "llm" and cohost_other) else TUNNEL_URL_FILE))
    log_path = Path(log_file or (Path("/root/cloudflared_llm.log") if (service_kind == "llm" and cohost_other and not use_named) else CLOUDFLARED_LOG))

    if cohost_other:
        kill_pat = "cloudflared tunnel.* run" if use_named else f"cloudflared tunnel.*{local_port}"
    else:
        kill_pat = "cloudflared tunnel"

    def _abort_unprotected(err_msg: str) -> None:
        _pkill_quiet(kill_pat)
        with contextlib.suppress(OSError):
            url_file.unlink(missing_ok=True)
        raise RuntimeError(err_msg)

    norm_kind = str(service_kind or "comfy").strip().lower()
    auth_k = _verify_local_auth_before_tunnel(norm_kind, int(local_port), auth_key, use_named, cohost_other, _abort_unprotected)

    if not force_restart:
        reused = _try_reuse_running_cloudflared(
            use_named=use_named,
            local_port=int(local_port),
            eff_url=eff_url,
            service_kind=service_kind,
            auth_k=auth_k,
            url_file=url_file,
            tunnel_re=tunnel_re,
        )
        if reused:
            return reused

    _pkill_quiet(kill_pat)
    time.sleep(0.5)

    cf_bin = shutil.which("cloudflared") or str(CLOUDFLARED_BIN)
    if not Path(cf_bin).is_file():
        raise FileNotFoundError(f"未找到 cloudflared 可执行文件: {cf_bin}")

    cf_env = os.environ.copy()
    if use_named:
        cf_env["TUNNEL_TOKEN"] = eff_tok
        cmd = [cf_bin, "tunnel", "--no-autoupdate", "--protocol", "http2", "run"]
    else:
        cmd = [cf_bin, "tunnel", "--no-autoupdate", "--protocol", "http2", "--url", f"http://127.0.0.1:{local_port}"]

    with open(log_path, "wb") as log_f:
        proc = subprocess.Popen(cmd, env=cf_env, stdin=subprocess.DEVNULL, stdout=log_f, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
    print(f"已后台拉起 cloudflared (PID={proc.pid}, port={local_port})，日志: {log_path}", flush=True)

    tunnel_url = _wait_for_cloudflared_tunnel_url(
        proc,
        use_named=use_named,
        eff_url=eff_url,
        eff_tok=eff_tok,
        service_kind=service_kind,
        auth_k=auth_k,
        log_path=log_path,
        tunnel_re=tunnel_re,
    )
    _await_public_tunnel_auth(
        tunnel_url,
        kind=norm_kind,
        auth_key=auth_k,
        abort_fn=_abort_unprotected,
        max_attempts=(12 if use_named else 15),
        interval=1.5,
    )
    url_file.write_text(tunnel_url + "\n", encoding="utf-8")
    return tunnel_url


def start_comfy_tunnel_standard() -> str:
    """ComfyUI 自举标准步骤 [7/7]：先强制校验公网鉴权网关已生效，再发布或按分阶段配置延后隧道。"""
    print("=== [7/7] 启动 ComfyUI Cloudflared 公网隧道 ===", flush=True)
    auth_ok, auth_reason = verify_local_comfy_auth_enforced()
    if not auth_ok:
        raise RuntimeError(f"ComfyUI 公网鉴权网关未生效，拒绝暴露公网隧道: {auth_reason}")
    if env_flag("COMFY_DEFER_TUNNEL", False):
        print("[多工作流分阶段启动] ComfyUI 本地鉴权已就绪，延后至最终阶段统一发布隧道", flush=True)
        return ""
    url = start_cloudflared_tunnel(
        local_port=8188,
        service_kind="comfy",
        cf_token=os.environ.get("CF_TUNNEL_TOKEN") or "",
        cf_url=os.environ.get("CF_TUNNEL_URL") or "",
        cohost_other=env_flag("COMFY_COHOST_LLM", False),
        tunnel_url_file=TUNNEL_URL_FILE,
    )
    print("========================================", flush=True)
    print(f"COMFYUI_TUNNEL_URL: {url}", flush=True)
    print("========================================", flush=True)
    return url


def finalize_service_contract(
    *,
    is_dual: bool,
    profiles: list[str],
    comfy_profiles: list[str],
    llm_profile: str | None = None,
    cf_comfy_url: str = "",
    cf_llm_url: str = "",
    tunnel_pattern: str = "",
) -> dict[str, Any]:
    """汇总并在远端以 0600 写入 `/root/colab_service_meta.json`，确保双服务均通过公网双向鉴权后才发布。"""
    tun_re = re.compile(tunnel_pattern) if tunnel_pattern else DEFAULT_TUNNEL_URL_RE
    meta = read_json_dict(SERVICE_META_FILE) if is_dual else {}

    def _read_url(p: Path, fb: str) -> str:
        if p.is_file():
            m = tun_re.search(p.read_text(encoding="utf-8", errors="ignore"))
            if m:
                return m.group(0).rstrip("/")
        return str(fb or "").strip().rstrip("/")

    comfy_file_u = _read_url(TUNNEL_URL_FILE, "")
    llm_u = _read_url(LLM_TUNNEL_URL_FILE, str(meta.get("llm_tunnel_url") or cf_llm_url)) if is_dual else ""
    comfy_auth_k = ensure_comfy_auth_key()

    # 隧道地址文件可能是同一实例上次部署遗留的（进程已被清理），必须公网实测通过才可信；否则在此主动拉起并验收
    if comfy_file_u and probe_public_tunnel_url(comfy_file_u, kind="comfy", auth_key=comfy_auth_k):
        comfy_u = comfy_file_u
    else:
        if comfy_file_u:
            print(f"[隧道复核] 已记录的 ComfyUI 隧道 {comfy_file_u} 公网不可达，正在重新拉起...", flush=True)
            comfy_file_u = ""
        cf_tok_env = (os.environ.get("CF_TUNNEL_TOKEN") or "").strip()
        comfy_u = start_cloudflared_tunnel(
            local_port=8188,
            service_kind="comfy",
            cf_token=cf_tok_env,
            cf_url=cf_comfy_url,
            cohost_other=is_dual,
            tunnel_url_file=TUNNEL_URL_FILE,
            tunnel_re=tun_re,
            auth_key=comfy_auth_k,
        ).rstrip("/")

    if is_dual:
        llm_port = int(meta.get("llm_port") or 18080)
        llm_key = load_local_llm_auth_key(str(meta.get("api_key") or "").strip() or None)
        cf_tok = (os.environ.get("CF_TUNNEL_TOKEN") or "").strip()

        def _abort_dual(msg: str) -> None:
            _pkill_quiet("cloudflared tunnel")
            with contextlib.suppress(OSError):
                TUNNEL_URL_FILE.unlink(missing_ok=True)
                LLM_TUNNEL_URL_FILE.unlink(missing_ok=True)
            raise RuntimeError(msg)

        if cf_tok and cf_llm_url:
            llm_ok, llm_reason = verify_local_llm_auth_enforced(port=llm_port, api_key=llm_key or None)
            if not llm_ok:
                _abort_dual(f"LLM 网关 API Key 鉴权未生效，拒绝发布双栖命名隧道: {llm_reason}")
            clean_cf_llm = cf_llm_url.strip().rstrip("/")
            _await_public_tunnel_auth(
                clean_cf_llm,
                kind="llm",
                auth_key=llm_key,
                abort_fn=_abort_dual,
                max_attempts=12,
                interval=1.5,
            )
            LLM_TUNNEL_URL_FILE.write_text(clean_cf_llm + "\n", encoding="utf-8")
            llm_u = clean_cf_llm
        elif not llm_u:
            llm_u = start_cloudflared_tunnel(
                local_port=llm_port,
                service_kind="llm",
                cohost_other=True,
                tunnel_url_file=LLM_TUNNEL_URL_FILE,
                tunnel_re=tun_re,
                auth_key=llm_key,
            ).rstrip("/")

        meta.update({
            "kind": "dual",
            "profiles": list(profiles),
            "comfy_profiles": list(comfy_profiles),
            "llm_profile": llm_profile,
            "comfy_port": 8188,
            "llm_port": llm_port,
            "port": 8188,
            "tunnel_url": comfy_u or llm_u,
            "comfy_tunnel_url": comfy_u,
            "llm_tunnel_url": llm_u,
            "comfy_api_key": comfy_auth_k,
        })
    else:
        meta = {
            "kind": "comfy",
            "profiles": list(profiles),
            "comfy_profiles": list(comfy_profiles),
            "comfy_port": 8188,
            "port": 8188,
            "tunnel_url": comfy_u,
            "comfy_tunnel_url": comfy_u,
            "comfy_api_key": comfy_auth_k,
        }
    write_private_json(SERVICE_META_FILE, meta)
    if is_dual:
        print("========================================", flush=True)
        if llm_u:
            print(f"LLM_TUNNEL_URL: {llm_u}", flush=True)
        if comfy_u:
            print(f"COMFYUI_TUNNEL_URL: {comfy_u}", flush=True)
            print(f"SERVICE_TUNNEL_URL: {comfy_u}", flush=True)
        print("========================================", flush=True)
    elif not comfy_file_u and comfy_u:
        print("========================================", flush=True)
        print(f"COMFYUI_TUNNEL_URL: {comfy_u}", flush=True)
        print("========================================", flush=True)
    return meta


def write_runtime_versions_record(profile: str, extra: dict[str, Any] | None = None) -> None:
    """以 0600 落盘实例运行时组件版本清单。"""
    info: dict[str, Any] = {
        "profile": profile,
        "python": sys.version.split()[0],
        "rclone_version": safe_cmd_line([shutil.which("rclone") or "/usr/local/bin/rclone", "version"]),
        "cloudflared_version": safe_cmd_line([shutil.which("cloudflared") or str(CLOUDFLARED_BIN), "--version"]),
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if extra:
        info.update(extra)
    # PyTorch 未安装或 CUDA 不可用时跳过可选显卡版本字段
    with contextlib.suppress(ImportError, RuntimeError, OSError):
        import torch

        info["torch"] = str(torch.__version__)
        info["cuda"] = str(torch.version.cuda or "")
        if torch.cuda.is_available():
            info["gpu"] = str(torch.cuda.get_device_name(0))
    # 版本摘要文件属可选诊断产物，无写权限时不阻断主流程
    with contextlib.suppress(OSError):
        write_private_json(Path("/root/colab_runtime_versions.json"), info)
    print(f"[VERSIONS] {json.dumps(info, ensure_ascii=False)}", flush=True)


# --- 供 verify_*.py 共用的 ComfyUI HTTP 冒烟助手 ---
def comfy_get_json(path: str, api: str = COMFY_API) -> dict:
    with urllib.request.urlopen(f"{api.rstrip('/')}{path}", timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8") or "{}")


def comfy_post_json(path: str, payload: dict, api: str = COMFY_API) -> dict:
    req = urllib.request.Request(
        f"{api.rstrip('/')}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8") or "{}")


def comfy_submit_and_wait(
    label: str,
    graph: dict,
    *,
    timeout_s: int = 900,
    poll_sec: float = 3.0,
    log_interval: float = 15.0,
    api: str = COMFY_API,
) -> tuple[dict, float]:
    print(f"\n=== [{label}] 提交任务 ===", flush=True)
    t0 = time.monotonic()
    res = comfy_post_json("/prompt", {"prompt": graph}, api=api)
    if not isinstance(res, dict) or res.get("node_errors") or res.get("error") or not res.get("prompt_id"):
        raise RuntimeError(f"[{label}] 提交被 ComfyUI 拒绝: {json.dumps(res, ensure_ascii=False)[:800]}")
    pid = res["prompt_id"]
    print(f"[{label}] prompt_id={pid}，等待执行...", flush=True)

    hist_item = None
    last_log = 0.0
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        hh = comfy_get_json(f"/history/{pid}", api=api)
        if isinstance(hh, dict) and pid in hh:
            hist_item = hh[pid]
            break
        now = time.monotonic()
        if now - last_log >= log_interval:
            print(f"[{label}] 运行中... 已等待 {now - t0:.0f}s", flush=True)
            last_log = now
        time.sleep(poll_sec)
    if hist_item is None:
        raise TimeoutError(f"[{label}] 等待执行超时 ({timeout_s}s)")
    status = hist_item.get("status") or {}
    if status.get("status_str") != "success" or status.get("completed") is False:
        raise RuntimeError(f"[{label}] 执行失败: status={status.get('status_str')}, messages={status.get('messages')}")
    elapsed = time.monotonic() - t0
    print(f"[{label}] 执行成功（耗时 {elapsed:.1f}s）", flush=True)
    return hist_item, elapsed


def comfy_extract_outputs(
    hist_item: dict,
    out_root: Path = COMFY_OUT,
    *,
    keys: tuple[str, ...] = ("images", "videos", "gifs", "audio"),
    fallback_dir: Path | None = None,
) -> list[Path]:
    files: list[Path] = []
    for node_out in (hist_item.get("outputs") or {}).values():
        if not isinstance(node_out, dict):
            continue
        for key in keys:
            for item in node_out.get(key) or []:
                if item.get("type", "output") != "output" or not item.get("filename"):
                    continue
                fname = item["filename"]
                sub = item.get("subfolder") or ""
                p = (out_root / sub / fname) if sub else ((fallback_dir or out_root) / fname)
                if p.is_file() and p not in files:
                    files.append(p)
    return files
