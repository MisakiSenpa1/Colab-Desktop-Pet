"""Colab 桌宠后端数据采集与控制模块。

负责通过本地 `google-colab-cli` 与 `colab_ctl.py` 无窗口获取：
1. Colab 计算单元 (CU) 余额、当前消耗速率 (`CU/hr`)、预计可用时长；
2. 活跃 GPU/TPU 会话与孤儿会话状态；
3. 本地每日积分消耗小账本与单次连续开机计时（不保存任何账号凭据）；
4. 快捷运维动作：打开 ComfyUI 隧道、跳转 Colab 会话控制台、拉回输出与一键停卡。
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
import urllib.request
import uuid
import webbrowser
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path, PurePosixPath
from typing import Any

import requests
from PIL import Image

try:
    import websocket
except Exception:
    websocket = None

PET_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = PET_DIR.parent
LOCAL_DOWNLOAD_ROOT = Path(
    os.environ.get("COLAB_DOWNLOAD_DIR") or (PACKAGE_DIR / "_downloads" / "all")
)
CTL_SCRIPT = PACKAGE_DIR / "colab_ctl.py"
STATE_FILE = PET_DIR / "pet_state.json"


def _first_existing_path(candidates: list[Path], fallback: Path) -> Path:
    """按优先级查找第一个存在的本地文件，均缺失时返回标准默认路径。"""
    for cand in candidates:
        if cand.is_file():
            return cand
    return fallback


def _resolve_colab_py() -> Path:
    """解析 google-colab-cli 虚拟环境 Python 解释器路径。"""
    env_p = (os.environ.get("COLAB_PY") or "").strip()
    if env_p:
        return Path(env_p)
    appdata = Path(os.environ.get("APPDATA") or (Path.home() / "AppData" / "Roaming"))
    uv_py = appdata / "uv" / "tools" / "google-colab-cli" / "Scripts" / "python.exe"
    return _first_existing_path(
        [uv_py, Path.home() / ".local" / "share" / "uv" / "tools" / "google-colab-cli" / "Scripts" / "python.exe"],
        uv_py,
    )


def _resolve_colab_exe() -> Path:
    """解析 colab 可执行文件路径。"""
    env_p = (os.environ.get("COLAB_EXE") or "").strip()
    if env_p:
        return Path(env_p)
    which_p = shutil.which("colab")
    default_p = Path.home() / ".local" / "bin" / "colab.exe"
    cands = ([Path(which_p)] if which_p else []) + [default_p]
    return _first_existing_path(cands, default_p)


def _resolve_rclone_exe() -> Path:
    """解析 rclone 可执行文件路径（可选依赖）。"""
    env_p = (os.environ.get("RCLONE_EXE") or "").strip()
    if env_p:
        return Path(env_p)
    which_p = shutil.which("rclone")
    scoop_p = Path.home() / "scoop" / "shims" / "rclone.exe"
    cands = ([Path(which_p)] if which_p else []) + [scoop_p]
    return _first_existing_path(cands, Path(which_p) if which_p else scoop_p)


def _resolve_rclone_conf() -> Path:
    """解析 rclone 配置文件路径（可选依赖）。"""
    env_p = (os.environ.get("RCLONE_CONF") or "").strip()
    if env_p:
        return Path(env_p)
    appdata = Path(os.environ.get("APPDATA") or (Path.home() / "AppData" / "Roaming"))
    default_p = appdata / "rclone" / "rclone.conf"
    cands = [
        default_p,
        Path.home() / ".config" / "rclone" / "rclone.conf",
        Path.home() / "scoop" / "apps" / "rclone" / "current" / "rclone.conf",
    ]
    return _first_existing_path(cands, default_p)


COLAB_PY = _resolve_colab_py()
COLAB_EXE_DEFAULT = _resolve_colab_exe()
RCLONE_EXE = _resolve_rclone_exe()
RCLONE_CONF = _resolve_rclone_conf()

_RCLONE_REMOTE = (os.environ.get("COLAB_RCLONE_REMOTE") or "gdrive_save").strip().rstrip(":") or "gdrive_save"
GDRIVE_LORA_REMOTE = f"{_RCLONE_REMOTE}:Colab_ComfyUI_Models/loras"
GDRIVE_WF_REMOTE = f"{_RCLONE_REMOTE}:Colab_ComfyUI_Models/workflows"

MEDIA_EXTS = {
    ".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp",
    ".mp4", ".mov", ".webm", ".mkv",
    ".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus",
}
WORKFLOW_EXTS = {".json"}
MODEL_EXTS = {".safetensors", ".ckpt", ".pt", ".pth"}

# LLM 对话模式支持提取文本与识图引导的后缀集合
LLM_TEXT_EXTS = {
    ".txt", ".md", ".json", ".py", ".yaml", ".yml", ".csv",
    ".log", ".toml", ".ini", ".js", ".ts", ".html", ".css",
}
LLM_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
SUPPORTED_BOOT_PROFILES = ("anima", "minimax", "drama3", "qwen_llm")
SUPPORTED_LLM_QUANTS = ("UD-Q4_K_XL", "UD-IQ4_XS", "IQ3_XXS")
LLM_QUANT_BASE_SHARDS: dict[str, int] = {
    "UD-Q4_K_XL": 4,
    "UD-IQ4_XS": 3,
    "IQ3_XXS": 2,
}

# Windows 下禁止弹出控制台黑框
CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

# 匹配 trycloudflare 快速隧道或用户自配置的 HTTPS 命名隧道域名
_TUNNEL_URL_RE = re.compile(r"https://(?!api\.)(?:[a-z0-9-]+\.)+[a-z]{2,}(?::\d+)?")

# 单次调用 colab_cli 内部 API 的无窗口探针脚本（直读 /tun/m/ccu-info 获取余额与实时可用显卡池 + 活跃会话）
_COLAB_PROBE_SCRIPT = """
import json
import time
from urllib.parse import urljoin
from colab_cli.common import state


def _retry(fn, attempts=3, delay=1.0):
    last_exc = None
    for idx in range(attempts):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if idx + 1 < attempts:
                time.sleep(delay)
    raise last_exc


def _fetch_ccu_raw(client):
    url = urljoin(client.colab_domain, "/tun/m/ccu-info")
    resp = client.session.get(
        url,
        params={"authuser": "0"},
        headers={"Accept": "application/json", "X-Colab-Client-Agent": "colab-cli"},
        timeout=15,
    )
    resp.raise_for_status()
    body = client._strip_xssi_prefix(resp.text or "")
    data = json.loads(body) if body else {}
    return data if isinstance(data, dict) else {}


# 仅使用共享读锁读取本地会话列表，禁止调用 sync_sessions() 以免 stdout 打印剪枝日志或冲突写锁
local_sessions = state.store.list()
client = state.client

eligible_gpus = None
eligible_tpus = None
balance = None
rate_hourly = 0.0
assignments_count = 0

try:
    raw_ccu = _retry(lambda: _fetch_ccu_raw(client))
    bal_val = raw_ccu.get("currentBalance", raw_ccu.get("paidComputeUnitsBalance"))
    if bal_val is not None:
        balance = float(bal_val)
        rate_hourly = float(raw_ccu.get("consumptionRateHourly") or 0.0)
        assignments_count = int(raw_ccu.get("assignmentsCount") or 0)
    raw_gpus = raw_ccu.get("eligibleGpus")
    if isinstance(raw_gpus, list):
        eligible_gpus = [str(x).upper() for x in raw_gpus if x]
    raw_tpus = raw_ccu.get("eligibleTpus")
    if isinstance(raw_tpus, list):
        eligible_tpus = [str(x).upper() for x in raw_tpus if x]
except Exception:
    pass

if balance is None:
    info = _retry(client.get_consumption_user_info)
    balance = float(info.paid_compute_units_balance)
    rate_hourly = float(info.consumption_rate_hourly)
    assignments_count = int(info.assignments_count)

assignments = _retry(client.list_assignments)
name_by_ep = {s.endpoint: s.name for s in local_sessions.values()}

items = []
for a in assignments:
    ep = a.endpoint
    sname = name_by_ep.get(ep, "?")
    accel_obj = getattr(a, "accelerator", None)
    accel = accel_obj.value if accel_obj else "CPU"
    if accel == "NONE":
        accel = "CPU"
    variant_obj = getattr(a, "variant", None)
    variant = variant_obj.name if variant_obj else "DEFAULT"
    shape_obj = getattr(a, "machine_shape", None)
    machine_shape = getattr(shape_obj, "name", "STANDARD") if shape_obj is not None else "STANDARD"
    items.append({
        "name": sname,
        "endpoint": ep,
        "accelerator": accel,
        "variant": variant,
        "machine_shape": machine_shape,
        "orphan": sname == "?",
    })

payload = {
    "balance": float(balance),
    "rate_hourly": float(rate_hourly),
    "assignments_count": int(assignments_count),
    "eligible_gpus": eligible_gpus,
    "eligible_tpus": eligible_tpus,
    "sessions": items,
}
print("<<<COLAB_PET_JSON>>>" + json.dumps(payload, ensure_ascii=False) + "<<<END_COLAB_PET_JSON>>>", flush=True)
""".strip()

# 通过 Colab Contents HTTP REST 接口免 WebSocket 直读服务元数据契约或隧道地址文件
_COLAB_FAST_TUNNEL_SCRIPT = """
import base64
import json
import re
import sys
import requests
from colab_cli.common import state

TUNNEL_RE = re.compile(r"https://(?!api\\.)(?:[a-z0-9-]+\\.)+[a-z]{2,}(?::\\d+)?")
target_ep = sys.argv[1] if len(sys.argv) > 1 else ""
session_name = sys.argv[2] if len(sys.argv) > 2 else "comfy"


def _decode_contents(data):
    fmt = data.get("format")
    raw = data.get("content", "")
    if isinstance(raw, dict):
        return raw
    if fmt == "base64" and isinstance(raw, str):
        return base64.b64decode(raw).decode("utf-8", errors="ignore")
    return str(raw or "")


local_sessions = state.store.list()
assignments = state.client.list_assignments()
if not assignments:
    print("<<<COLAB_TUNNEL>>>" + json.dumps({"status": "no_session"}, ensure_ascii=False) + "<<<END_COLAB_TUNNEL>>>", flush=True)
    raise SystemExit(0)

target = None
if target_ep:
    for a in assignments:
        if a.endpoint == target_ep:
            target = a
            break

if target is None and session_name in local_sessions:
    wanted_ep = local_sessions[session_name].endpoint
    for a in assignments:
        if a.endpoint == wanted_ep:
            target = a
            break

if target is None:
    target = assignments[0]

info = target.runtime_proxy_info
base = info.url.rstrip("/")
tok = info.token  # 仅在内存中用于鉴权请求，绝不打印或持久化
req_params = {"authuser": "0", "colab-runtime-proxy-token": tok, "content": "1"}

sess = requests.Session()
out = None

# 1) 优先读取 /root/colab_service_meta.json 获取自描述服务类型、模型参数与内存态密钥
resp_meta = sess.get(f"{base}/api/contents/root/colab_service_meta.json", params=req_params, timeout=8)
if resp_meta.status_code == 200:
    try:
        decoded_meta = _decode_contents(resp_meta.json())
        meta_obj = decoded_meta if isinstance(decoded_meta, dict) else json.loads(str(decoded_meta))
    except Exception:
        meta_obj = None
    if isinstance(meta_obj, dict):
        api_key = str(meta_obj.get("api_key") or "").strip()
        clean_meta = {k: v for k, v in meta_obj.items() if k != "api_key"}
        clean_meta.setdefault("kind", "llm")
        clean_meta.setdefault("port", 18080 if clean_meta.get("kind") == "llm" else 8188)
        m_meta_url = TUNNEL_RE.search(str(meta_obj.get("tunnel_url") or ""))
        tun_url = m_meta_url.group(0) if m_meta_url else None
        if not tun_url:
            # 若元数据尚未填入隧道或隧道已由外部刷新，回退读取 comfy_tunnel_url.txt 补齐 URL
            resp_txt_fb = sess.get(f"{base}/api/contents/root/comfy_tunnel_url.txt", params=req_params, timeout=6)
            if resp_txt_fb.status_code == 200:
                txt_fb = str(_decode_contents(resp_txt_fb.json()) or "")
                m_fb = TUNNEL_RE.search(txt_fb)
                if m_fb:
                    tun_url = m_fb.group(0)
        if tun_url:
            clean_meta["tunnel_url"] = tun_url
            out = {
                "status": "ok",
                "endpoint": target.endpoint,
                "url": tun_url,
                "meta": clean_meta,
                "api_key": api_key,
            }
        else:
            out = {
                "status": "missing",
                "endpoint": target.endpoint,
                "meta": clean_meta,
                "api_key": api_key,
            }
elif resp_meta.status_code not in (404,):
    resp_meta.raise_for_status()

# 2) 若不存在 colab_service_meta.json (404)，回退读取传统 ComfyUI 隧道地址文件
if out is None:
    comfy_meta = {"kind": "comfy", "port": 8188}
    resp = sess.get(
        f"{base}/api/contents/root/comfy_tunnel_url.txt",
        params=req_params,
        timeout=8,
    )
    if resp.status_code == 404:
        out = {"status": "missing", "endpoint": target.endpoint, "meta": comfy_meta}
    else:
        resp.raise_for_status()
        text = str(_decode_contents(resp.json()) or "")
        m = TUNNEL_RE.search(text)
        if m:
            out = {"status": "ok", "endpoint": target.endpoint, "url": m.group(0), "meta": comfy_meta}
        else:
            out = {"status": "missing", "endpoint": target.endpoint, "meta": comfy_meta}

print("<<<COLAB_TUNNEL>>>" + json.dumps(out, ensure_ascii=False) + "<<<END_COLAB_TUNNEL>>>", flush=True)
""".strip()

# 通过 Colab Contents HTTP PUT 将工作流 JSON 直写云端实例 workflows 目录
_COLAB_CONTENTS_PUT_SCRIPT = """
import base64
import json
import sys
import requests
from colab_cli.common import state

target_ep = sys.argv[1]
session_name = sys.argv[2]
rel_path = sys.argv[3].lstrip("/")
b64_data = sys.stdin.read().strip()

local_sessions = state.store.list()
assignments = state.client.list_assignments()
if not assignments:
    raise RuntimeError("无活跃 Colab 实例")

target = next((a for a in assignments if target_ep and a.endpoint == target_ep), None)
if target is None and session_name in local_sessions:
    wanted = local_sessions[session_name].endpoint
    target = next((a for a in assignments if a.endpoint == wanted), None)
if target is None:
    target = assignments[0]

info = target.runtime_proxy_info
base = info.url.rstrip("/")
tok = info.token
resp = requests.put(
    f"{base}/api/contents/{rel_path}",
    params={"authuser": "0", "colab-runtime-proxy-token": tok},
    json={"type": "file", "format": "base64", "content": b64_data},
    timeout=15,
)
resp.raise_for_status()
print("OK")
""".strip()

# 让运行中的 Colab 实例从 Google Drive 增量拉取刚上传的 LoRA
_COLAB_REMOTE_LORA_PULL_SCRIPT = """
import sys
from colab_cli.common import state
from colab_cli.runtime import ColabRuntime

session_name = sys.argv[1]
src_remote = sys.argv[2]
dst_local = sys.argv[3]

s = state.get_session(session_name)
if s is None:
    raise RuntimeError(f"未找到本地会话 '{session_name}'")

rt = ColabRuntime(s.url, s.token, session_name=session_name, kernel_id=s.kernel_id, session_id=s.session_id)
try:
    ws = rt.kernel_client._manager.client
    if not ws.connection_ready.is_set() and not ws.connection_ready.wait(timeout=25.0):
        raise RuntimeError("WebSocket 握手超时")
    code = f\"\"\"
import os, shutil, subprocess
os.makedirs(os.path.dirname({dst_local!r}), exist_ok=True)
rbin = shutil.which("rclone") or "/usr/local/bin/rclone"
subprocess.run([rbin, "copyto", "--config", "/tmp/rclone.conf", "--disable-http2", {src_remote!r}, {dst_local!r}], check=True)
print("REMOTE_LORA_READY")
\"\"\"
    outputs = rt.execute_code(code, timeout=180.0)
    errs = [o for o in outputs if o.get("output_type") == "error"]
    if errs:
        raise RuntimeError(str(errs[0]))
    print("REMOTE_LORA_READY")
finally:
    rt.stop()
""".strip()

DEFAULT_PREFS: dict[str, Any] = {
    "x": None,
    "y": None,
    "scale": 1.0,
    "on_top": True,
    "bubble_open": True,
    "left_flipped": False,
    "dock_edge": None,
    "edge_auto_hide": True,
    "poll_interval_sec": 60,
    "warn_hours": 10.0,
    "warn_balance": 50.0,
    "long_run_warn_minutes": 180,
    "auto_pull_outputs": True,
    "keep_alive": True,
    "idle_auto_stop_minutes": 60,
    "boot_profile": "anima",
    "boot_gpu": "L4",
    "boot_high_mem": False,
    "boot_minimax_r2v": True,
    "boot_minimax_pe": True,
    "boot_llm_quant": "UD-Q4_K_XL",
    "boot_llm_ctx": 32768,
    "boot_llm_mmproj": True,
}

# 支持桌面一键选卡开机的 GPU 型号与高峰期缺卡临时冷却时间（10 分钟）
SUPPORTED_BOOT_GPUS = ("T4", "L4", "A100", "G4", "H100")
# 账号已通过 /tun/m/ccu-info 实测校准过的单卡基准费率（后续单卡运行时自动动态刷新）
DEFAULT_MEASURED_GPU_RATES: dict[str, float] = {
    "T4:STD": 1.15,
    "L4:STD": 1.94,
    "G4:STD": 10.05,
}
SCARCE_GPU_TTL_SEC = 600.0
_STEP_BANNER_RE = re.compile(r"===\s*\[(\d+)/(\d+)\]\s*(.+?)\s*===")
_PCT_TOKEN_RE = re.compile(r"\((\d{1,3}(?:\.\d+)?)%\)")
_DL_SPEED_RE = re.compile(r"DL:\s*([0-9.]+\s*[KMGTP]?i?B(?:/s)?)", re.IGNORECASE)
_DEPLOY_DL_TOTAL_RE = re.compile(r"DEPLOY_DL_TOTAL:\s*(\d+)")
_DEPLOY_TUNNEL_LINE_RE = re.compile(
    r"(?:COMFYUI_TUNNEL_URL|SERVICE_TUNNEL_URL):\s*(https://(?!api\.)(?:[a-z0-9-]+\.)+[a-z]{2,}(?::\d+)?)"
)
_DEPLOY_STEP_BASE_PCT = {1: 16, 2: 23, 3: 31, 4: 42, 5: 78, 6: 86, 7: 95}

# 防断连保活心跳周期（10 分钟）、失败重试冷却（2 分钟）与智能空闲熔断（45 分钟）
KEEPALIVE_INTERVAL_SEC = 600.0
KEEPALIVE_RETRY_COOLDOWN_SEC = 120.0
KEEPALIVE_MAX_IDLE_SEC = 45 * 60.0
SUPPORTED_IDLE_STOP_MINUTES = (0, 60)

_CST_TZ = timezone(timedelta(hours=8))
_TS_SUFFIX_RE = re.compile(r"_\d{8}-\d{6}(?:-\d+)?$")
_PRESERVE_ORIGINAL_SUB_RE = re.compile(r"(?:^|/)drama3_voices$", re.IGNORECASE)

# 向 Google Colab 官方通道发送轻量三层保活心跳（TFE keep-alive -> 代理内核列表 -> 空闲内核空操作）
_COLAB_KEEPALIVE_SCRIPT = """
import json
import requests
from colab_cli.common import state
from colab_cli.runtime import ColabRuntime

client = state.client
assignments = client.list_assignments()
if not assignments:
    print("<<<COLAB_KEEPALIVE>>>" + json.dumps({"ok": True, "pinged": 0}, ensure_ascii=False) + "<<<END_COLAB_KEEPALIVE>>>", flush=True)
    raise SystemExit(0)

local_sessions = state.store.list()
sess_by_ep = {s.endpoint: s for s in local_sessions.values()}
pinged = 0

for a in assignments:
    touched = False
    # 1) Colab 官方 TFE 保活端点，刷新服务端 LastActiveTime
    try:
        r1 = client.session.get(
            f"{client.colab_domain.rstrip('/')}/tun/m/{a.endpoint}/keep-alive/",
            params={"authuser": "0"},
            headers={
                "X-Colab-Tunnel": "Google",
                "X-Colab-Client-Agent": "colab-cli",
                "Accept": "application/json",
            },
            timeout=8,
        )
        if r1.ok:
            touched = True
    except Exception:
        pass

    # 2) Runtime Proxy /api/kernels 探活，并仅在存在空闲内核时附带轻量执行心跳
    proxy_url = a.runtime_proxy_info.url.rstrip("/")
    tok = a.runtime_proxy_info.token
    idle_kid = None
    try:
        r2 = requests.get(
            f"{proxy_url}/api/kernels",
            params={"authuser": "0", "colab-runtime-proxy-token": tok},
            timeout=8,
        )
        if r2.ok:
            touched = True
            kernels = r2.json()
            if isinstance(kernels, list):
                for k in kernels:
                    if isinstance(k, dict) and k.get("execution_state") == "idle" and k.get("id"):
                        idle_kid = str(k["id"])
                        break
    except Exception:
        pass

    # 3) 空闲内核执行一次毫秒级纯 CPU 空操作，刷新 Jupyter 内核 last_activity（绝不抢占正在执行命令的内核）
    if idle_kid:
        sess = sess_by_ep.get(a.endpoint)
        sid = getattr(sess, "session_id", None) if sess else None
        rt = ColabRuntime(proxy_url, tok, kernel_id=idle_kid, session_id=sid)
        try:
            ws = rt.kernel_client._manager.client
            if ws.connection_ready.is_set() or ws.connection_ready.wait(timeout=8.0):
                rt.execute_code("1 + 1", timeout=6.0)
                touched = True
        except Exception:
            pass
        finally:
            try:
                rt.stop()
            except Exception:
                pass

    if touched:
        pinged += 1

print("<<<COLAB_KEEPALIVE>>>" + json.dumps({"ok": pinged > 0, "pinged": pinged}, ensure_ascii=False) + "<<<END_COLAB_KEEPALIVE>>>", flush=True)
""".strip()

# 注入到云端 ComfyUI 前端的「一键直铺画布」扩展脚本（兼容原生 WebSocket 与存量热挂载双模式）
_COMFY_DIRECT_CANVAS_JS = r"""/* colab_pet_direct_canvas v2 */
(async function () {
  if (window.__COLAB_PET_DIRECT_CANVAS_V2__) return;
  window.__COLAB_PET_DIRECT_CANVAS_V2__ = true;
  window.__COLAB_PET_DIRECT_CANVAS_V1__ = true;

  let modApp = null;
  let modApi = null;
  try {
    modApp = await import("../../scripts/app.js");
  } catch (e) {}
  try {
    modApi = await import("../../scripts/api.js");
  } catch (e) {}

  const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
  // 后台/隐藏标签页下浏览器会暂停 requestAnimationFrame，改为限时降级避免队列永久卡死
  const nextFrameSafe = () => {
    if (typeof document !== "undefined" && document.hidden) {
      return sleep(32);
    }
    if (typeof requestAnimationFrame !== "function") {
      return sleep(32);
    }
    return Promise.race([
      new Promise((resolve) => requestAnimationFrame(() => resolve())),
      sleep(80)
    ]);
  };

  const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
  const generateUuidV4 = () => {
    try {
      if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
        return crypto.randomUUID();
      }
    } catch (e) {}
    return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
      const r = (Math.random() * 16) | 0;
      const v = c === "x" ? r : (r & 0x3) | 0x8;
      return v.toString(16);
    });
  };

  const ensureValidUuid = (graphData) => {
    if (!graphData || typeof graphData !== "object") return graphData;
    const rawId = typeof graphData.id === "string" ? graphData.id.trim() : "";
    if (!rawId || !UUID_RE.test(rawId)) {
      graphData.id = generateUuidV4();
    }
    return graphData;
  };

  const sanitizeWfName = (raw) => {
    const clean = String(raw || "").trim().replace(/^workflows\//i, "").split("/").pop();
    return clean ? (clean.toLowerCase().endsWith(".json") ? clean : `${clean}.json`) : "";
  };

  // 模块加载第一时间捕获并剥离 URL 中的一次性直铺参数，防止用户按 F5 覆盖画布修改
  const initUrl = new URL(window.location.href);
  let pendingWfName = sanitizeWfName(initUrl.searchParams.get("pet_wf") || "");
  let pendingExtraNames = (initUrl.searchParams.get("pet_extra") || "")
    .split(",")
    .map(sanitizeWfName)
    .filter((n) => n && n !== pendingWfName);
  if (initUrl.searchParams.has("pet_wf") || initUrl.searchParams.has("pet_extra") || initUrl.searchParams.has("pet_ts")) {
    initUrl.searchParams.delete("pet_wf");
    initUrl.searchParams.delete("pet_extra");
    initUrl.searchParams.delete("pet_ts");
    try {
      window.history.replaceState(null, "", initUrl.toString());
    } catch (e) {}
  }

  function getApp() {
    return (modApp && modApp.app) || window.comfyAPI?.app?.app || window.app || null;
  }

  function getApi() {
    return (modApi && modApi.api) || window.comfyAPI?.api?.api || getApp()?.api || null;
  }

  async function waitForCanvasReady(timeoutMs = 30000) {
    const start = Date.now();
    while (Date.now() - start < timeoutMs) {
      const app = getApp();
      const splashEl = typeof document !== "undefined" ? document.getElementById("splash-loader") : null;
      const spinnerBusy = Boolean(app?.extensionManager?.spinner);
      const cfgBusy = Boolean(app?.configuringGraphLevel);
      if (
        app &&
        app.rootGraph &&
        typeof app.loadGraphData === "function" &&
        !spinnerBusy &&
        !splashEl &&
        !cfgBusy
      ) {
        try {
          if (app.graphLoadQueue) {
            await Promise.resolve(app.graphLoadQueue);
          }
        } catch (e) {}
        await nextFrameSafe();
        await sleep(80);
        const appAfter = getApp();
        if (
          appAfter &&
          appAfter.rootGraph &&
          !Boolean(appAfter.extensionManager?.spinner) &&
          !Boolean(appAfter.configuringGraphLevel)
        ) {
          return true;
        }
      }
      await sleep(100);
    }
    const appEnd = getApp();
    return Boolean(appEnd && appEnd.rootGraph && typeof appEnd.loadGraphData === "function");
  }

  async function fetchWorkflowJson(wfPath) {
    const encoded = encodeURIComponent(wfPath);
    const urls = [
      `/api/userdata/${encoded}?t=${Date.now()}`,
      `/userdata/${encoded}?t=${Date.now()}`
    ];
    for (const u of urls) {
      try {
        const resp = await fetch(u, { cache: "no-store" });
        if (resp.ok) {
          return await resp.json();
        }
      } catch (e) {}
    }
    throw new Error(`无法读取工作流文件: ${wfPath}`);
  }

  const handledCmdIds = new Set();
  let openQueue = Promise.resolve();

  async function _doOpenWorkflowOntoCanvas(rawName, extraNames = [], showNotice = true) {
    const cleanName = sanitizeWfName(rawName);
    if (!cleanName) return false;
    const ready = await waitForCanvasReady();
    if (!ready) return false;

    const app = getApp();
    if (!app || typeof app.loadGraphData !== "function") return false;
    const wfStore = app.extensionManager?.workflow;
    const wfPath = `workflows/${cleanName}`;

    try {
      if (typeof wfStore?.syncWorkflows === "function") {
        await wfStore.syncWorkflows();
      }
    } catch (e) {}

    const graphData = await fetchWorkflowJson(wfPath);
    if (!graphData || typeof graphData !== "object") return false;
    ensureValidUuid(graphData);

    try {
      const bgPaths = (extraNames || [])
        .map(sanitizeWfName)
        .filter((n) => n && n !== cleanName)
        .map((n) => `workflows/${n}`)
        .filter((p) => Boolean(wfStore?.getWorkflowByPath?.(p)));
      if (bgPaths.length && typeof wfStore?.openWorkflowsInBackground === "function") {
        wfStore.openWorkflowsInBackground({ right: bgPaths });
      }
    } catch (e) {}

    const wfObj = wfStore?.getWorkflowByPath?.(wfPath) || null;
    if (wfObj) {
      // 严禁在 loadGraphData 之前手动调用 wfStore.openWorkflow(wfObj)，否则会干扰 beforeLoadNewGraph 状态暂存导致旧画布覆盖新工作流
      await app.loadGraphData(graphData, true, true, wfObj);
      try {
        if (
          wfStore &&
          typeof wfStore.isActive === "function" &&
          !wfStore.isActive(wfObj) &&
          typeof wfStore.openWorkflow === "function"
        ) {
          await wfStore.openWorkflow(wfObj);
        }
      } catch (e) {}
      try {
        wfObj.changeTracker?.reset?.();
        wfObj.isModified = false;
      } catch (e) {}
    } else {
      await app.loadGraphData(graphData, true, true, cleanName.replace(/\.json$/i, ""));
    }

    try {
      app.canvas?.setDirty?.(true, true);
    } catch (e) {}

    if (showNotice) {
      try {
        app.extensionManager?.toast?.add?.({
          severity: "success",
          summary: "✨ 已直铺工作流到画布",
          detail: cleanName,
          life: 2800
        });
      } catch (e) {}
    }
    return true;
  }

  function enqueueOpenWorkflow(rawName, extraNames = [], showNotice = true) {
    const task = openQueue
      .then(
        () => _doOpenWorkflowOntoCanvas(rawName, extraNames, showNotice),
        () => _doOpenWorkflowOntoCanvas(rawName, extraNames, showNotice)
      )
      .catch((err) => {
        console.warn("[colab_pet_direct_canvas] 直铺工作流失败:", err);
        return false;
      });
    openQueue = task.then(() => undefined, () => undefined);
    return task;
  }

  async function postNativeAck(id, name) {
    const payload = JSON.stringify({ id, name, ts: Date.now() / 1000 });
    for (const ep of ["/colab_pet/ack_workflow", "/api/colab_pet/ack_workflow"]) {
      try {
        const resp = await fetch(ep, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: payload
        });
        if (resp.ok) return;
      } catch (e) {}
    }
  }

  async function postLegacyAck(id, name) {
    try {
      await fetch("/userdata/colab_pet_bridge_ack.json?overwrite=true", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id, name, ts: Date.now() / 1000 })
      });
    } catch (e) {}
  }

  async function startLivePushBridge() {
    const api = getApi();
    let nativeOk = false;

    try {
      const pingResp = await fetch(`/colab_pet/ping?t=${Date.now()}`, { cache: "no-store" });
      if (pingResp.ok) {
        const pdata = await pingResp.json();
        nativeOk = Boolean(pdata && pdata.ok);
      }
    } catch (e) {}

    if (nativeOk && api && typeof api.addEventListener === "function") {
      api.addEventListener("colab_pet.open_workflow", async (ev) => {
        const d = ev?.detail || {};
        const cmdId = String(d.id || "");
        const wfName = sanitizeWfName(d.name || "");
        if (!wfName) return;
        if (cmdId) {
          if (handledCmdIds.has(cmdId)) return;
          handledCmdIds.add(cmdId);
        }
        // 后台标签页略微让步前台可见标签页优先接管，且自身不阻塞可正常完成加载
        if (typeof document !== "undefined" && document.hidden) {
          await sleep(180);
        }
        const extras = Array.isArray(d.extra_names) ? d.extra_names : [];
        const ok = await enqueueOpenWorkflow(wfName, extras, true);
        if (ok) {
          if (cmdId) await postNativeAck(cmdId, wfName);
        } else if (cmdId) {
          handledCmdIds.delete(cmdId);
        }
      });
      return;
    }

    // 存量实例未加载自定义节点路由时，回退轻量指令轮询（存储在 userdata 根目录，不污染 workflows 栏）
    const bootTs = Date.now() / 1000 - 2.0;
    let pollBusy = false;
    const pollCmd = async () => {
      if (pollBusy) return;
      pollBusy = true;
      try {
        const r = await fetch(`/userdata/colab_pet_bridge_cmd.json?t=${Date.now()}`, { cache: "no-store" });
        if (!r.ok) return;
        const cmd = await r.json();
        if (!cmd || !cmd.id || !cmd.name) return;
        const cmdId = String(cmd.id);
        if (handledCmdIds.has(cmdId)) return;
        if (Number(cmd.ts || 0) < bootTs) return;
        handledCmdIds.add(cmdId);
        const wfName = sanitizeWfName(cmd.name);
        if (!wfName) return;
        if (typeof document !== "undefined" && document.hidden) {
          await sleep(180);
        }
        const extras = Array.isArray(cmd.extra_names) ? cmd.extra_names : [];
        const ok = await enqueueOpenWorkflow(wfName, extras, true);
        if (ok) {
          await postLegacyAck(cmdId, wfName);
        } else {
          handledCmdIds.delete(cmdId);
        }
      } catch (e) {
      } finally {
        pollBusy = false;
      }
    };

    window.addEventListener("focus", () => pollCmd());
    document.addEventListener("visibilitychange", () => {
      if (!document.hidden) pollCmd();
    });
    setInterval(pollCmd, 1200);
  }

  let setupStarted = false;
  async function runStartupHooks() {
    if (setupStarted) return;
    setupStarted = true;
    startLivePushBridge();
    if (pendingWfName) {
      const targetWf = pendingWfName;
      const targetExtras = pendingExtraNames;
      pendingWfName = "";
      pendingExtraNames = [];
      await enqueueOpenWorkflow(targetWf, targetExtras, true);
    }
  }

  const app = getApp();
  if (app && typeof app.registerExtension === "function") {
    app.registerExtension({
      name: "ColabPet.DirectCanvas",
      async setup() {
        runStartupHooks();
      }
    });
  } else {
    waitForCanvasReady().then(() => runStartupHooks());
  }
})();
""".strip()

# 云端 ComfyUI 原生自定义节点源码（提供浏览器客户端感知、WS 实时直铺与 ACK 确认端点）
_COMFY_BRIDGE_NODE_INIT_PY = '''"""Colab 桌宠直铺画布服务端桥接扩展。"""
import time
from collections import OrderedDict
from aiohttp import web
import server

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
WEB_DIRECTORY = "./js"

_ACK_MAP = OrderedDict()


def _record_ack(req_id: str, name: str = "") -> None:
    if not req_id:
        return
    _ACK_MAP[req_id] = {"id": req_id, "name": name, "ts": time.time()}
    _ACK_MAP.move_to_end(req_id)
    while len(_ACK_MAP) > 64:
        _ACK_MAP.popitem(last=False)


def _count_browser_clients(ps) -> int:
    sockets = getattr(ps, "sockets", None) or {}
    count = 0
    for cid, ws in list(sockets.items()):
        if str(cid).startswith("colab_pet_"):
            continue
        if ws is None or getattr(ws, "closed", False):
            continue
        count += 1
    return count


def _register_routes() -> None:
    ps = getattr(server.PromptServer, "instance", None)
    if ps is None or getattr(ps, "_colab_pet_routes_ready", False):
        return
    ps._colab_pet_routes_ready = True
    routes = ps.routes

    @routes.get("/colab_pet/ping")
    async def colab_pet_ping(_request):
        return web.json_response({
            "ok": True,
            "version": 2,
            "browser_clients": _count_browser_clients(ps),
        })

    @routes.post("/colab_pet/open_workflow")
    async def colab_pet_open_workflow(request):
        try:
            data = await request.json()
        except Exception:
            data = {}
        req_id = str(data.get("id") or "")
        name = str(data.get("name") or "")
        extra_names = data.get("extra_names") if isinstance(data.get("extra_names"), list) else []
        clients = _count_browser_clients(ps)
        if name and clients > 0:
            ps.send_sync(
                "colab_pet.open_workflow",
                {"id": req_id, "name": name, "extra_names": extra_names},
            )
        return web.json_response({
            "ok": True,
            "id": req_id,
            "browser_clients": clients,
        })

    @routes.post("/colab_pet/ack_workflow")
    async def colab_pet_ack_post(request):
        try:
            data = await request.json()
        except Exception:
            data = {}
        req_id = str(data.get("id") or "")
        name = str(data.get("name") or "")
        _record_ack(req_id, name)
        return web.json_response({"ok": True, "id": req_id})

    @routes.get("/colab_pet/ack_workflow")
    async def colab_pet_ack_get(request):
        req_id = str(request.query.get("id") or "")
        info = _ACK_MAP.get(req_id)
        return web.json_response({
            "ok": True,
            "acked": info is not None,
            "info": info,
        })


try:
    _register_routes()
except Exception:
    pass
'''.strip()

# 免重启热安装直铺画布前端扩展与自定义节点到已运行 Colab 实例的 Contents REST 脚本
_COLAB_INSTALL_CANVAS_EXT_SCRIPT = """
import json
import re
import sys
import requests
from colab_cli.common import state

target_ep = sys.argv[1] if len(sys.argv) > 1 else ""
session_name = sys.argv[2] if len(sys.argv) > 2 else "comfy"
payload = json.loads(sys.stdin.read())
js_b64 = payload["js_b64"]
py_b64 = payload["py_b64"]

local_sessions = state.store.list()
assignments = state.client.list_assignments()
if not assignments:
    raise RuntimeError("无活跃 Colab 实例")

target = next((a for a in assignments if target_ep and a.endpoint == target_ep), None)
if target is None and session_name in local_sessions:
    wanted = local_sessions[session_name].endpoint
    target = next((a for a in assignments if a.endpoint == wanted), None)
if target is None:
    target = assignments[0]

info = target.runtime_proxy_info
base = info.url.rstrip("/")
tok = info.token


def _get_json(rel, content="1"):
    return requests.get(
        f"{base}/api/contents/{rel.lstrip('/')}",
        params={"authuser": "0", "colab-runtime-proxy-token": tok, "content": content},
        timeout=10,
    )


def _ensure_dir(rel_dir):
    parts = [p for p in rel_dir.strip("/").split("/") if p]
    cur = ""
    for part in parts:
        cur = f"{cur}/{part}" if cur else part
        r = _get_json(cur, content="0")
        if r.status_code == 200:
            continue
        if r.status_code == 404:
            pr = requests.put(
                f"{base}/api/contents/{cur}",
                params={"authuser": "0", "colab-runtime-proxy-token": tok},
                json={"type": "directory"},
                timeout=10,
            )
            if pr.status_code not in (200, 201, 409):
                pr.raise_for_status()
        else:
            r.raise_for_status()


def _put_b64(rel_file, b64_str):
    parent = rel_file.strip("/").rsplit("/", 1)[0] if "/" in rel_file.strip("/") else ""
    if parent:
        _ensure_dir(parent)
    pr = requests.put(
        f"{base}/api/contents/{rel_file.lstrip('/')}",
        params={"authuser": "0", "colab-runtime-proxy-token": tok},
        json={"type": "file", "format": "base64", "content": b64_str},
        timeout=15,
    )
    pr.raise_for_status()


web_root = ""
for log_rel in ("root/comfy_colab.log", "root/comfy_bw1.log", "root/comfy_minimax.log", "tmp/comfy_h3.log", "tmp/comfyui.log"):
    r = _get_json(log_rel, content="1")
    if r.status_code == 200:
        d = r.json()
        raw = d.get("content", "")
        if d.get("format") == "base64":
            import base64
            txt = base64.b64decode(raw).decode("utf-8", errors="ignore")
        else:
            txt = str(raw or "")
        m = re.search(r"\\[Prompt Server\\] web root:\\s*(\\S+)", txt)
        if m:
            web_root = m.group(1).strip().lstrip("/")
            break

if not web_root:
    r = _get_json("usr/local/lib", content="1")
    if r.status_code == 200:
        for item in (r.json().get("content") or []):
            name = str(item.get("name") or "")
            if name.startswith("python3"):
                cand = f"usr/local/lib/{name}/dist-packages/comfyui_frontend_package/static"
                cr = _get_json(cand, content="0")
                if cr.status_code == 200:
                    web_root = cand
                    break

if not web_root:
    raise RuntimeError("未定位到云端 ComfyUI web_root")

_put_b64(f"{web_root}/extensions/colab_pet/direct_canvas.js", js_b64)
try:
    _put_b64("content/ComfyUI/custom_nodes/colab_pet_bridge/__init__.py", py_b64)
    _put_b64("content/ComfyUI/custom_nodes/colab_pet_bridge/js/direct_canvas.js", js_b64)
except Exception:
    pass

print("<<<COLAB_CANVAS_EXT>>>" + json.dumps({"ok": True, "web_root": web_root}, ensure_ascii=False) + "<<<END_COLAB_CANVAS_EXT>>>", flush=True)
""".strip()


def extract_png_comfy_workflow(path: Path) -> str | None:
    """检查 PNG 元数据中是否内嵌合法 ComfyUI 工作流。

    同时支持网页端 `workflow`（画布格式）与脚本端 `prompt`（API 节点拓扑格式）；
    当两者并存时优先返回保留画布坐标与连线布局的 `workflow`。
    """
    try:
        with Image.open(path) as img:
            raw_wf = img.info.get("workflow")
            raw_prompt = img.info.get("prompt")
    except Exception:
        return None

    # 1) 优先提取网页端 workflow 画布格式
    if isinstance(raw_wf, str) and raw_wf.strip():
        try:
            parsed_wf = json.loads(raw_wf)
            if isinstance(parsed_wf, dict) and ("nodes" in parsed_wf or "links" in parsed_wf):
                return raw_wf
        except Exception:
            pass

    # 2) 回退兼容脚本 API 提交生成的 prompt 节点拓扑格式
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
        except Exception:
            pass

    return None


def _is_api_prompt_dict(data: Any) -> bool:
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
    """判断 API prompt 输入值是否为合法的 `[源节点ID, 输出槽位]` 连线引用。"""
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
    """根据 /object_info 字段定义推算缺省控件值。"""
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
    """判断是否需要为随机种子控件追加 ComfyUI 前端的运行控制项（如 fixed）。"""
    if cfg.get("control_after_generate") is True:
        return True
    return raw_type == "INT" and inp_name in ("seed", "noise_seed") and cfg.get("control_after_generate") is not False


def _ordered_node_input_specs(
    node_info: dict[str, Any],
) -> list[tuple[str, Any, dict[str, Any]]]:
    """按 ComfyUI 前端展示顺序提取节点的输入规格列表。"""
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


_WORKFLOW_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


def _ensure_workflow_uuid(wf_dict: dict[str, Any]) -> bool:
    """确保 UI 工作流顶层具备合法 UUID，避免 ComfyUI v0.3.73 将缺失 id 的工作流误判为同一标签页。"""
    raw_id = wf_dict.get("id")
    if isinstance(raw_id, str) and _WORKFLOW_UUID_RE.match(raw_id.strip()):
        return False
    wf_dict["id"] = str(uuid.uuid4())
    return True


def _api_prompt_to_ui_workflow(
    prompt_dict: dict[str, Any],
    object_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """将 ComfyUI API `prompt` 字典转换为可在网页画布直接打开的标准 LiteGraph 工作流。

    当提供在线 `/object_info` 时，可精确还原各节点输入槽位顺序、`widgets_values`、
    `COMBO` 与连线类型，并在随机种子后保留 `"fixed"` 控制项以确保复现原图结果；
    当 `object_info` 缺失时，仍按兼容规则离线生成完整 `nodes` 与 `links` 结构。
    """
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

    # 按有向无环拓扑深度从左到右排列节点，迭代上限防环
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

    sorted_keys = sorted(ordered_keys, key=lambda k: (depths[k], orig_order[k]))
    str_to_int_id: dict[str, int] = {k: idx + 1 for idx, k in enumerate(sorted_keys)}

    # 统计每列行号并初始化 UI 节点骨架
    depth_rows: dict[int, int] = {}
    ui_nodes_by_key: dict[str, dict[str, Any]] = {}
    ui_nodes: list[dict[str, Any]] = []
    pending_links: list[tuple[str, int, str, int, str]] = []

    for order_idx, k in enumerate(sorted_keys):
        node_data = valid_nodes[k]
        class_type = node_data["class_type"]
        node_inputs_data: dict[str, Any] = node_data["inputs"]
        node_info = (
            object_info.get(class_type)
            if isinstance(object_info, dict) and isinstance(object_info.get(class_type), dict)
            else None
        )

        depth = depths[k]
        row_idx = depth_rows.get(depth, 0)
        depth_rows[depth] = row_idx + 1

        if "TextEncode" in class_type:
            size = [400, 260]
        elif "KSampler" in class_type:
            size = [330, 290]
        elif class_type in ("SaveImage", "PreviewImage"):
            size = [330, 270]
        else:
            size = [330, 150]

        ui_inputs: list[dict[str, Any]] = []
        widgets_values: list[Any] = []

        if node_info is not None:
            specs = _ordered_node_input_specs(node_info)
            known_names = {name for name, _, _ in specs}
            for inp_name, raw_type, cfg in specs:
                is_combo = isinstance(raw_type, list) or raw_type == "COMBO"
                slot_type = "COMBO" if is_combo else (str(raw_type) if isinstance(raw_type, str) else "*")
                force_input = bool(cfg.get("forceInput"))
                is_widget = (not force_input) and (
                    is_combo or raw_type in ("INT", "FLOAT", "STRING", "BOOLEAN")
                )
                has_val = inp_name in node_inputs_data
                raw_val = node_inputs_data.get(inp_name)
                is_linked = has_val and _is_prompt_link_ref(raw_val, valid_nodes)

                if is_widget:
                    if is_linked:
                        src_key = str(raw_val[0])
                        src_slot = int(raw_val[1])
                        target_slot = len(ui_inputs)
                        ui_inputs.append(
                            {
                                "name": inp_name,
                                "type": slot_type,
                                "widget": {"name": inp_name},
                                "link": None,
                            }
                        )
                        pending_links.append((src_key, src_slot, k, target_slot, slot_type))
                        widgets_values.append(_default_widget_value(raw_type, cfg))
                    else:
                        val = raw_val if has_val else _default_widget_value(raw_type, cfg)
                        widgets_values.append(val)
                    if _should_append_seed_control(inp_name, raw_type, cfg):
                        widgets_values.append("fixed")
                else:
                    target_slot = len(ui_inputs)
                    ui_inputs.append({"name": inp_name, "type": slot_type, "link": None})
                    if is_linked:
                        src_key = str(raw_val[0])
                        src_slot = int(raw_val[1])
                        pending_links.append((src_key, src_slot, k, target_slot, slot_type))

            # 兜底补充 prompt 中未列入 /object_info 的动态连线输入
            for extra_name, extra_val in node_inputs_data.items():
                if extra_name not in known_names and _is_prompt_link_ref(extra_val, valid_nodes):
                    target_slot = len(ui_inputs)
                    ui_inputs.append({"name": extra_name, "type": "*", "link": None})
                    pending_links.append(
                        (str(extra_val[0]), int(extra_val[1]), k, target_slot, "*")
                    )

            out_types = node_info.get("output") if isinstance(node_info.get("output"), list) else []
            out_names = (
                node_info.get("output_name")
                if isinstance(node_info.get("output_name"), list)
                else out_types
            )
            ui_outputs = [
                {
                    "name": str(out_names[idx] if idx < len(out_names) else out_t),
                    "type": str(out_t),
                    "links": [],
                    "slot_index": idx,
                }
                for idx, out_t in enumerate(out_types)
            ]
        else:
            # 离线兜底：区分连线槽位与标量控件值
            for inp_name, raw_val in node_inputs_data.items():
                if _is_prompt_link_ref(raw_val, valid_nodes):
                    target_slot = len(ui_inputs)
                    ui_inputs.append({"name": inp_name, "type": "*", "link": None})
                    pending_links.append(
                        (str(raw_val[0]), int(raw_val[1]), k, target_slot, "*")
                    )
                else:
                    widgets_values.append(raw_val)
                    if (
                        inp_name in ("seed", "noise_seed")
                        and isinstance(raw_val, int)
                        and not isinstance(raw_val, bool)
                    ):
                        widgets_values.append("fixed")
            ui_outputs = []

        ui_node: dict[str, Any] = {
            "id": str_to_int_id[k],
            "type": class_type,
            "pos": [80 + depth * 420, 80 + row_idx * 320],
            "size": size,
            "flags": {},
            "order": order_idx,
            "mode": 0,
            "inputs": ui_inputs,
            "outputs": ui_outputs,
            "properties": {"Node name for S&R": class_type},
            "widgets_values": widgets_values,
        }
        ui_nodes_by_key[k] = ui_node
        ui_nodes.append(ui_node)

    # 统一分配递增 link_id，确保 links 元组、目标输入 link 与源输出 links 三方一致
    ui_links: list[list[Any]] = []
    for link_idx, (src_key, src_slot, dst_key, dst_slot, target_slot_type) in enumerate(
        pending_links, start=1
    ):
        src_node = ui_nodes_by_key[src_key]
        dst_node = ui_nodes_by_key[dst_key]
        while len(src_node["outputs"]) <= src_slot:
            idx = len(src_node["outputs"])
            inferred_t = target_slot_type if idx == src_slot else "*"
            src_node["outputs"].append(
                {"name": inferred_t, "type": inferred_t, "links": [], "slot_index": idx}
            )
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
        ui_links.append(
            [link_idx, src_node["id"], src_slot, dst_node["id"], dst_slot, link_type]
        )

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


def _sanitize_filename(name: str, fallback: str = "file") -> str:
    """过滤非法文件名字符，保留单层文件名。"""
    base = Path(str(name).replace("\\", "/")).name
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", base).strip()
    return cleaned or fallback


def _safe_output_dest(root: Path, subfolder: str, filename: str) -> Path:
    """构造严格位于 root 目录下的安全输出落盘路径，防御路径穿越。"""
    root_resolved = root.resolve()
    parts: list[str] = []
    if subfolder:
        for seg in PurePosixPath(subfolder.replace("\\", "/")).parts:
            if seg and seg not in (".", "..", "/"):
                parts.append(_sanitize_filename(seg, "sub"))
    parts.append(_sanitize_filename(filename, "output.png"))
    dest = root_resolved.joinpath(*parts).resolve()
    if dest != root_resolved and root_resolved not in dest.parents:
        raise ValueError(f"非法输出路径: {subfolder}/{filename}")
    return dest


def _format_timestamped_filename(filename: str, mtime_sec: float | None = None) -> str:
    """为产出文件附加确定性北京时间时间戳（YYYYMMDD-HHMMSS），避免新实例序号重置覆写旧图。"""
    p = PurePosixPath(filename)
    stem, suffix = p.stem, p.suffix
    if _TS_SUFFIX_RE.search(stem):
        return filename
    ts_str = datetime.fromtimestamp(mtime_sec if mtime_sec is not None else time.time(), _CST_TZ).strftime(
        "%Y%m%d-%H%M%S"
    )
    sep = "" if stem.endswith("_") else "_"
    return f"{stem}{sep}{ts_str}{suffix}"


def _sha256_file(path: Path) -> str:
    """分块计算本地文件 SHA256 摘要。"""
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        while True:
            chunk = fp.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _resolve_non_clobber_dest(dest: Path, src_size: int, src_sha256: str) -> tuple[Path, bool]:
    """若本机已存在同名同内容文件则跳过覆写；若内容不同则递增后缀防止误覆盖历史作品。"""
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


def _parse_http_mtime(last_modified: str | None) -> float | None:
    """解析 HTTP Last-Modified 响应头为 UNIX 秒，失败返回 None。"""
    if not last_modified:
        return None
    try:
        dt = parsedate_to_datetime(last_modified.strip())
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except Exception:
        return None


def _format_comfy_error(node_type: str | None, exc_type: str | None, exc_msg: str | None) -> str:
    """将 ComfyUI 节点异常格式化为紧凑的单行中文提示（不超过 36 字）。"""
    node = (node_type or "节点").strip() or "节点"
    etype = (exc_type or "").strip()
    emsg = (exc_msg or "").strip()
    combined = f"{etype} {emsg}".lower()
    if any(k in combined for k in ("out of memory", "outofmemoryerror", "allocation on device")):
        out = f"{node}: 显存不足 (OOM)"
    elif any(k in combined for k in ("no such file", "filenotfound", "not found", "missing")):
        out = f"{node}: 缺少模型或文件"
    else:
        detail = etype or emsg or "执行异常"
        out = f"{node}: {detail}"
    return (out[:35] + "…") if len(out) > 36 else out


class _StreamingMultipartFile:
    """零全量缓存的流式 multipart/form-data 读取器，支持字节级进度回调。"""

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
                f"{v}\r\n".encode("utf-8")
            )
        ctype = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
        safe_name = _sanitize_filename(file_path.name).replace('"', "_")
        head_parts.append(
            f"--{self.boundary}\r\n"
            f'Content-Disposition: form-data; name="{field_name}"; filename="{safe_name}"\r\n'
            f"Content-Type: {ctype}\r\n\r\n".encode("utf-8")
        )
        self._head = b"".join(head_parts)
        self._tail = f"\r\n--{self.boundary}--\r\n".encode("utf-8")
        self._file_size = file_path.stat().st_size
        self._total_len = len(self._head) + self._file_size + len(self._tail)
        self._fp = open(file_path, "rb")
        self._stage = 0  # 0=head, 1=file, 2=tail, 3=eof
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
        try:
            self._fp.close()
        except Exception:
            pass

    def __enter__(self) -> "_StreamingMultipartFile":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()


def _clean_env() -> dict[str, str]:
    """构造强制 UTF-8 与防挂起超时的子进程环境变量。"""
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env.setdefault("COLAB_CLI_REQUEST_TIMEOUT", "12")
    return env


def _probe_tunnel_alive(url: str, timeout: float = 3.5, kind: str = "comfy") -> bool:
    """探测公网隧道是否在线可用（comfy 探测 /system_stats，llm 探测 /health，auto 自动双模探测）。"""
    if not url or not _TUNNEL_URL_RE.match(url.strip()):
        return False
    base = url.strip().rstrip("/")
    mode = str(kind or "comfy").strip().lower()

    def _probe_single(k: str) -> bool:
        target = f"{base}/health" if k == "llm" else f"{base}/system_stats"
        try:
            req = urllib.request.Request(target, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if resp.status != 200:
                    return False
                body = json.loads(resp.read().decode("utf-8", errors="ignore"))
                if not isinstance(body, dict):
                    return False
                if k == "llm":
                    return str(body.get("status") or "").lower() == "ok"
                return "system" in body or "devices" in body
        except Exception:
            return False

    if mode == "llm":
        return _probe_single("llm")
    if mode == "comfy":
        return _probe_single("comfy")
    return _probe_single("comfy") or _probe_single("llm")


def _detect_tunnel_kind(
    url: str, timeout: float = 3.0, preferred: str = "comfy"
) -> str | None:
    """自动识别存活隧道后端的服务类型，返回 'comfy'、'llm' 或 None。"""
    order = ("llm", "comfy") if preferred == "llm" else ("comfy", "llm")
    for cand in order:
        if _probe_tunnel_alive(url, timeout=timeout, kind=cand):
            return cand
    return None


def _sanitize_prefs_dict(prefs: dict[str, Any]) -> dict[str, Any]:
    """对本地持久化偏好设置执行类型与枚举范围清洗，确保不包含任何敏感字段。"""
    cleaned = dict(DEFAULT_PREFS)
    for k in DEFAULT_PREFS:
        if k in prefs:
            cleaned[k] = prefs[k]

    prof = str(cleaned.get("boot_profile") or "anima").strip().lower()
    cleaned["boot_profile"] = prof if prof in SUPPORTED_BOOT_PROFILES else "anima"

    quant = str(cleaned.get("boot_llm_quant") or "UD-Q4_K_XL").strip()
    cleaned["boot_llm_quant"] = quant if quant in SUPPORTED_LLM_QUANTS else "UD-Q4_K_XL"

    try:
        ctx_val = int(cleaned.get("boot_llm_ctx") or 32768)
    except (TypeError, ValueError):
        ctx_val = 32768
    cleaned["boot_llm_ctx"] = max(4096, min(262144, ctx_val))

    cleaned["boot_llm_mmproj"] = bool(cleaned.get("boot_llm_mmproj", True))
    cleaned["keep_alive"] = bool(cleaned.get("keep_alive", True))
    try:
        idle_mins = int(cleaned.get("idle_auto_stop_minutes", 60))
    except (TypeError, ValueError):
        idle_mins = 60
    cleaned["idle_auto_stop_minutes"] = idle_mins if idle_mins in SUPPORTED_IDLE_STOP_MINUTES else 60
    return cleaned


def _run_hidden(cmd: list[str], timeout: int = 25) -> subprocess.CompletedProcess[str]:
    """以无窗口、UTF-8 方式执行本地子进程。"""
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        env=_clean_env(),
        creationflags=CREATE_NO_WINDOW,
    )


def _hidden_python_exe() -> str:
    """当宿主为 pythonw.exe 时优先切回同目录 python.exe 配合无窗口标志执行 CLI。"""
    exe = Path(sys.executable)
    if exe.name.lower() == "pythonw.exe":
        py_console = exe.with_name("python.exe")
        if py_console.exists():
            return str(py_console)
    return str(exe)


def _find_colab_exe() -> str | None:
    """定位 colab.exe 可执行文件。"""
    if COLAB_EXE_DEFAULT.exists():
        return str(COLAB_EXE_DEFAULT)
    return shutil.which("colab")


def _default_baseline(balance: float) -> float:
    """根据当前余额推算进度条满额基准值（按百位向上取整）。"""
    if balance <= 0:
        return 100.0
    return float(max(100, int(math.ceil(balance / 100.0) * 100)))


class ColabPetBackend:
    """Colab 桌宠状态管理与运维动作执行器。"""

    def __init__(
        self,
        state_path: Path = STATE_FILE,
        download_root: Path = LOCAL_DOWNLOAD_ROOT,
    ) -> None:
        self.state_path = state_path
        self.download_root = Path(download_root)
        self._lock = threading.RLock()
        self._fetch_lock = threading.Lock()
        self._comfy_lock = threading.RLock()
        self._action_lock = threading.Lock()
        self._upload_lock = threading.Lock()
        self._keepalive_lock = threading.Lock()
        self._deploy_lock = threading.Lock()
        self._deploy_proc: Any = None
        self._deploy_cancelled: bool = False
        self._deploy_dl_total: int = 3
        self._deploy_dl_done: int = 0
        self._deploy_dl_name: str = ""
        self._scarce_gpus: dict[str, dict[str, Any]] = {}
        self._deploy_state: dict[str, Any] = self._default_deploy_state()
        self._consecutive_failures: int = 0
        self._comfy_http_failures: int = 0
        self._last_tunnel_discover_ts: float = 0.0
        self._last_keepalive_ts: float = 0.0
        self._last_activity_m: float = 0.0
        self._idle_warned_ep: str | None = None
        self._idle_stop_lock = threading.Lock()
        self._event_cb: Callable[[dict[str, Any]], None] | None = None
        self._monitor_stop = threading.Event()
        self._wake_monitor = threading.Event()
        self._monitor_thread: threading.Thread | None = None
        self._ws_stop = threading.Event()
        self._ws_thread: threading.Thread | None = None
        self._ws_target_url: str | None = None
        self._ws_app: Any = None
        self._ws_last_emit_ts: float = 0.0
        self._http = requests.Session()
        self._http.headers.update({"User-Agent": "ColabPet/1.0"})
        self._seen_prompts_by_url: dict[str, set[str]] = {}
        self._reported_error_prompts: set[str] = set()
        self._pending_pull_items: list[tuple[str, dict[str, str]]] = []
        self._last_downloaded_files: list[Path] = []
        self._canvas_ext_ready_eps: dict[str, str] = {}
        self._canvas_ext_lock = threading.Lock()
        # 远端自描述元数据缓存（ep -> 非敏感元信息）与纯内存态密钥映射（严禁写入 self.state 或落盘）
        self._service_meta_by_ep: dict[str, dict[str, Any]] = {}
        self._runtime_secrets: dict[str, str] = {}
        self._deploy_meta_warmed: bool = False
        self._comfy_state: dict[str, Any] = self._default_comfy_state()

        self.state = self._load_state()
        has_cached_bal = self.state.get("last_balance") is not None
        self.last_snapshot: dict[str, Any] = (
            self._build_snapshot(live_data=None, stale=False, error_msg=None)
            if has_cached_bal
            else self._build_snapshot(
                live_data=None,
                stale=True,
                error_msg="正在连接 Colab...",
            )
        )

    @staticmethod
    def _default_comfy_state() -> dict[str, Any]:
        return {
            "online": False,
            "service_kind": "comfy",
            "ws_connected": False,
            "tunnel_url": None,
            "running_count": 0,
            "pending_count": 0,
            "step_value": 0,
            "step_max": 0,
            "step_pct": 0.0,
            "current_node": "",
            "gpu_name": "",
            "vram_used_gb": 0.0,
            "vram_total_gb": 0.0,
            "vram_pct": 0.0,
            "has_error": False,
            "error_summary": "",
            "pending_pull_count": 0,
            "last_output_count": 0,
            "last_output_name": "",
            "model_name": "",
            "quant": "",
            "ctx_size": 0,
            "tps": None,
        }

    def get_comfy_state(self) -> dict[str, Any]:
        """返回线程安全的 ComfyUI 实时状态副本。"""
        with self._comfy_lock:
            return dict(self._comfy_state)

    def _emit_event(self, payload: dict[str, Any]) -> None:
        cb = self._event_cb
        if not cb:
            return
        try:
            cb(payload)
        except Exception:
            pass

    @staticmethod
    def _default_deploy_state() -> dict[str, Any]:
        """构建默认的一键开机与 7 步进度状态字典。"""
        return {
            "active": False,
            "profile": "anima",
            "gpu": "L4",
            "high_mem": False,
            "step_idx": 0,
            "step_total": 7,
            "stage_title": "",
            "sub_detail": "",
            "pct": 0,
            "started_ts": 0,
            "elapsed_sec": 0,
            "error_msg": "",
            "stockout_gpu": None,
            "suggest_gpu": None,
        }

    def is_deploying(self) -> bool:
        """当前是否正在执行一键开机部署。"""
        with self._lock:
            return bool(self._deploy_state.get("active"))

    def get_deploy_state(self) -> dict[str, Any]:
        """返回线程安全的部署进度副本并实时计算耗时读秒。"""
        with self._lock:
            st = dict(self._deploy_state)
            if st.get("active") and st.get("started_ts"):
                st["elapsed_sec"] = max(0, int(time.time()) - int(st["started_ts"]))
            return st

    def _get_active_scarce_gpus(self) -> dict[str, dict[str, Any]]:
        """清理并返回仍在冷却有效期内的紧缺/受限 GPU 映射。"""
        now_ts = int(time.time())
        with self._lock:
            expired = [
                k
                for k, v in self._scarce_gpus.items()
                if (now_ts - int(v.get("ts") or 0)) > SCARCE_GPU_TTL_SEC
            ]
            for k in expired:
                self._scarce_gpus.pop(k, None)
            return {k: dict(v) for k, v in self._scarce_gpus.items()}

    @staticmethod
    def _gpu_rate_key(accelerator: Any, machine_shape: Any = None) -> str | None:
        """将加速卡型号与内存形态映射为规范的单卡实测费率存储键。"""
        acc = str(accelerator or "").strip().upper()
        if acc not in SUPPORTED_BOOT_GPUS:
            return None
        if acc == "L4":
            return "L4:STD"
        shape = str(machine_shape or "").strip().upper()
        suffix = "HM" if shape in ("HIGH_RAM", "HM", "1") else "STD"
        return f"{acc}:{suffix}"

    def _default_state(self) -> dict[str, Any]:
        today = datetime.now().strftime("%Y-%m-%d")
        return {
            "date": today,
            "day_start_balance": None,
            "day_consumed_cu": 0.0,
            "day_topup_cu": 0.0,
            "last_balance": None,
            "last_delta": 0.0,
            "baseline_balance": 1000.0,
            "active_since_ts": None,
            "active_endpoint": None,
            "session_start_balance": None,
            "last_rate_hourly": 0.0,
            "last_assignments_count": 0,
            "last_sessions": [],
            "last_eligible_gpus": None,
            "last_eligible_tpus": None,
            "last_updated_ts": None,
            "measured_gpu_rates": dict(DEFAULT_MEASURED_GPU_RATES),
            "tunnel_cache": {},
            "prefs": dict(DEFAULT_PREFS),
        }

    def _load_state(self) -> dict[str, Any]:
        default = self._default_state()
        if not self.state_path.exists():
            return default
        try:
            raw = json.loads(self.state_path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                return default
            merged = dict(default)
            for key in default:
                if key == "prefs" and isinstance(raw.get("prefs"), dict):
                    merged["prefs"] = _sanitize_prefs_dict({**DEFAULT_PREFS, **raw["prefs"]})
                elif key == "measured_gpu_rates" and isinstance(raw.get("measured_gpu_rates"), dict):
                    cleaned_rates = dict(DEFAULT_MEASURED_GPU_RATES)
                    valid_keys = {
                        f"{g}:{s}"
                        for g in SUPPORTED_BOOT_GPUS
                        for s in ("STD", "HM")
                        if not (g == "L4" and s == "HM")
                    }
                    for rk, rv in raw["measured_gpu_rates"].items():
                        if rk in valid_keys:
                            try:
                                fv = float(rv)
                            except (TypeError, ValueError):
                                continue
                            if 0.02 < fv < 200.0:
                                cleaned_rates[rk] = round(fv, 2)
                    merged["measured_gpu_rates"] = cleaned_rates
                elif key == "tunnel_cache" and isinstance(raw.get("tunnel_cache"), dict):
                    cleaned_cache: dict[str, str] = {}
                    for ep, val in raw["tunnel_cache"].items():
                        if isinstance(ep, str) and isinstance(val, str):
                            m = _TUNNEL_URL_RE.search(val.strip())
                            if m:
                                cleaned_cache[ep] = m.group(0)
                    merged["tunnel_cache"] = cleaned_cache
                elif key in raw:
                    merged[key] = raw[key]
            return merged
        except Exception:
            return default

    def _save_state(self) -> None:
        """原子写入非敏感配置与账本状态，不保存任何登录凭据。"""
        tmp_path = self.state_path.with_suffix(".tmp")
        payload = json.dumps(self.state, ensure_ascii=False, indent=2)
        tmp_path.write_text(payload, encoding="utf-8")
        tmp_path.replace(self.state_path)

    def _query_colab_live(self) -> dict[str, Any]:
        """调用 colab_cli 获取实时 CU 余额与活跃运行时会话。"""
        if not COLAB_PY.exists():
            raise FileNotFoundError(f"未找到 colab-cli 运行环境: {COLAB_PY}")

        proc = _run_hidden([str(COLAB_PY), "-c", _COLAB_PROBE_SCRIPT], timeout=25)
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "").strip()
            last_line = err.splitlines()[-1] if err else f"退出码 {proc.returncode}"
            raise RuntimeError(f"Colab 查询失败: {last_line}")

        stdout = proc.stdout or ""
        matched = re.search(
            r"<<<COLAB_PET_JSON>>>(.*?)<<<END_COLAB_PET_JSON>>>",
            stdout,
            re.S,
        )
        json_text = matched.group(1).strip() if matched else stdout.strip()
        if not json_text:
            raise RuntimeError("Colab 返回空结果")
        return json.loads(json_text)

    def _update_ledger(self, live: dict[str, Any]) -> None:
        """更新每日消耗账本、基准额度与连续开机计时。"""
        now_ts = int(time.time())
        today = datetime.now().strftime("%Y-%m-%d")
        balance = float(live["balance"])
        rate_hourly = float(live["rate_hourly"])
        assignments_count = int(live["assignments_count"])
        sessions = list(live.get("sessions", []))

        prev_bal = self.state.get("last_balance")
        diff = 0.0

        # 跨零点或全新日期：以前一刻余额无缝衔接为新一日的起始余额，不丢失跨午夜消耗
        if self.state.get("date") != today:
            start_bal = float(prev_bal) if prev_bal is not None else balance
            self.state["date"] = today
            self.state["day_start_balance"] = start_bal
            self.state["day_topup_cu"] = 0.0
            raw_diff = round(balance - start_bal, 4)
            if raw_diff > 0.05:
                self.state["day_topup_cu"] = raw_diff
                self.state["last_delta"] = raw_diff
                diff = raw_diff
            elif raw_diff < -0.0005:
                self.state["last_delta"] = raw_diff
                diff = raw_diff
            else:
                self.state["last_delta"] = 0.0
            self.state["last_balance"] = balance
        else:
            if self.state.get("day_start_balance") is None:
                self.state["day_start_balance"] = balance
            if prev_bal is not None:
                raw_diff = round(balance - float(prev_bal), 4)
                if raw_diff < -0.0005:
                    self.state["last_delta"] = raw_diff
                    diff = raw_diff
                elif raw_diff > 0.05:
                    # 充值/月度额度发放时不计入消耗
                    self.state["day_topup_cu"] = round(
                        float(self.state.get("day_topup_cu") or 0.0) + raw_diff,
                        4,
                    )
                    self.state["last_delta"] = raw_diff
                    diff = raw_diff
            self.state["last_balance"] = balance

        # 统一以 起始余额 + 今日充值 - 当前余额 精确对账，消除多次浮点累加尾差
        day_start_f = float(self.state["day_start_balance"])
        day_topup_f = float(self.state.get("day_topup_cu") or 0.0)
        self.state["day_consumed_cu"] = round(max(0.0, day_start_f + day_topup_f - balance), 4)

        # 进度条满额参考值：随最高余额自动升级到百位
        baseline = float(self.state.get("baseline_balance") or 0.0)
        if baseline <= 0.0 or balance > baseline:
            self.state["baseline_balance"] = _default_baseline(balance)

        # 连续开机计时与本次开机起始余额（支持换卡重置与存量运行会话平滑回填）
        primary_sess = next(
            (s for s in sessions if s.get("name") == "comfy" and s.get("endpoint")),
            None,
        ) or (sessions[0] if sessions and sessions[0].get("endpoint") else None)
        cur_ep = str(primary_sess.get("endpoint")) if primary_sess else None

        is_running = assignments_count > 0 or rate_hourly > 0.01
        if is_running:
            prev_since = self.state.get("active_since_ts")
            prev_ep = self.state.get("active_endpoint")
            if not prev_ep:
                prev_sessions = list(self.state.get("last_sessions") or [])
                prev_primary = next(
                    (s for s in prev_sessions if s.get("name") == "comfy" and s.get("endpoint")),
                    None,
                ) or (
                    prev_sessions[0]
                    if prev_sessions and prev_sessions[0].get("endpoint")
                    else None
                )
                if prev_primary:
                    prev_ep = str(prev_primary.get("endpoint"))
            switched_ep = bool(cur_ep and prev_ep and cur_ep != prev_ep)
            if not prev_since or switched_ep:
                self.state["active_since_ts"] = now_ts
                self.state["active_endpoint"] = cur_ep
                self._last_activity_m = time.monotonic()
                self._idle_warned_ep = None
                self.state["session_start_balance"] = (
                    float(prev_bal)
                    if (not switched_ep and prev_bal is not None and float(prev_bal) >= balance)
                    else balance
                )
            else:
                if cur_ep:
                    self.state["active_endpoint"] = cur_ep
                if self._last_activity_m <= 0.0:
                    self._last_activity_m = time.monotonic()
                    self._idle_warned_ep = None
                if self.state.get("session_start_balance") is None:
                    est = rate_hourly * max(0, now_ts - int(prev_since)) / 3600.0
                    cap = day_start_f + day_topup_f
                    self.state["session_start_balance"] = (
                        min(cap, balance + max(0.0, est))
                        if cap >= balance
                        else (balance + max(0.0, est))
                    )
                elif diff > 0.05:
                    # 运行中途发生充值时同步抬高会话起始基准，避免冲减本次实耗
                    self.state["session_start_balance"] = (
                        float(self.state["session_start_balance"]) + diff
                    )
        else:
            self.state["active_since_ts"] = None
            self.state["active_endpoint"] = None
            self.state["session_start_balance"] = None
            self._last_activity_m = 0.0
            self._idle_warned_ep = None
            self._last_keepalive_ts = 0.0

        self.state["last_rate_hourly"] = rate_hourly
        self.state["last_assignments_count"] = assignments_count
        self.state["last_sessions"] = sessions
        self.state["last_updated_ts"] = now_ts

        # 仅当单实例在线运行且返回有效耗速时自动校准对应卡型与内存档的实测单价（防止多卡汇总耗速污染单卡费率）
        if assignments_count == 1 and len(sessions) == 1 and 0.02 < rate_hourly < 200.0:
            s0 = sessions[0]
            rkey = self._gpu_rate_key(s0.get("accelerator"), s0.get("machine_shape"))
            if rkey:
                rates_map = self.state.setdefault(
                    "measured_gpu_rates", dict(DEFAULT_MEASURED_GPU_RATES)
                )
                rates_map[rkey] = round(rate_hourly, 2)

        if live.get("eligible_gpus") is not None:
            self.state["last_eligible_gpus"] = [
                str(g).upper() for g in live["eligible_gpus"] if g
            ]
        if live.get("eligible_tpus") is not None:
            self.state["last_eligible_tpus"] = [
                str(t).upper() for t in live["eligible_tpus"] if t
            ]
        # 若实测某型号 GPU 正在健康运行，则自动清除它的紧缺标记
        for s in sessions:
            acc = str(s.get("accelerator") or "").upper()
            if acc and acc in self._scarce_gpus:
                self._scarce_gpus.pop(acc, None)

        # 自动清理已停机实例对应的过期隧道地址缓存与内存态元数据/密钥
        active_eps = {str(s.get("endpoint")) for s in sessions if s.get("endpoint")}
        raw_cache = self.state.get("tunnel_cache")
        if isinstance(raw_cache, dict):
            self.state["tunnel_cache"] = {
                ep: url
                for ep, url in raw_cache.items()
                if ep in active_eps and isinstance(url, str) and _TUNNEL_URL_RE.match(url)
            }
        else:
            self.state["tunnel_cache"] = {}
        for stale_ep in list(self._service_meta_by_ep.keys()):
            if stale_ep not in active_eps:
                self._service_meta_by_ep.pop(stale_ep, None)
        for stale_ep in list(self._runtime_secrets.keys()):
            if stale_ep not in active_eps:
                self._runtime_secrets.pop(stale_ep, None)

    def _build_snapshot(
        self,
        live_data: dict[str, Any] | None,
        stale: bool,
        error_msg: str | None = None,
    ) -> dict[str, Any]:
        """拼装前端所需的完整状态快照。"""
        now_ts = int(time.time())
        prefs = dict(DEFAULT_PREFS)
        prefs.update(self.state.get("prefs") or {})

        if live_data is not None:
            balance = float(live_data["balance"])
            rate_hourly = float(live_data["rate_hourly"])
            assignments_count = int(live_data["assignments_count"])
            sessions = list(live_data.get("sessions", []))
            eligible_gpus = live_data.get("eligible_gpus")
            eligible_tpus = live_data.get("eligible_tpus")
        else:
            balance = self.state.get("last_balance")
            rate_hourly = float(self.state.get("last_rate_hourly") or 0.0)
            assignments_count = int(self.state.get("last_assignments_count") or 0)
            sessions = list(self.state.get("last_sessions") or [])
            eligible_gpus = self.state.get("last_eligible_gpus")
            eligible_tpus = self.state.get("last_eligible_tpus")

        scarce_gpus = self._get_active_scarce_gpus()
        deploy_state = self.get_deploy_state()

        baseline = float(self.state.get("baseline_balance") or 1000.0)
        if balance is not None and balance > baseline:
            baseline = _default_baseline(float(balance))

        remaining_hours: float | None = None
        if balance is not None and rate_hourly > 0.01:
            remaining_hours = round(float(balance) / rate_hourly, 2)

        active_since = self.state.get("active_since_ts")
        active_minutes = (
            max(0, int((now_ts - int(active_since)) // 60)) if active_since else 0
        )

        updated_ts = self.state.get("last_updated_ts")
        updated_str = (
            datetime.fromtimestamp(int(updated_ts)).strftime("%H:%M:%S")
            if updated_ts
            else "--:--:--"
        )

        # 确定整体状态与推荐表情（idle / proud / sad）
        is_running = assignments_count > 0 or rate_hourly > 0.01
        warn_bal = float(prefs.get("warn_balance", 50.0))
        warn_hrs = float(prefs.get("warn_hours", 10.0))
        long_warn = int(prefs.get("long_run_warn_minutes", 180))

        low_balance = balance is not None and (
            float(balance) <= warn_bal
            or (remaining_hours is not None and remaining_hours <= warn_hrs)
        )
        long_running_alert = is_running and active_minutes >= long_warn
        has_orphan = any(s.get("orphan") for s in sessions)
        comfy = self.get_comfy_state()

        if error_msg and balance is None:
            if error_msg == "正在连接 Colab...":
                mood = "idle"
                status_level = "running"
                status_label = "正在连接..."
            else:
                mood = "sad"
                status_level = "error"
                status_label = "连接异常"
        elif low_balance:
            mood = "sad"
            status_level = "danger"
            status_label = "余额预警"
        elif stale and error_msg:
            mood = "sad"
            status_level = "warn"
            status_label = "网络波动(缓存)"
        elif is_running and comfy.get("has_error"):
            mood = "sad"
            status_level = "danger"
            status_label = "推理报错中断"
        elif is_running:
            mood = "sad" if long_running_alert else "idle"
            status_level = "warn" if (long_running_alert or has_orphan) else "running"
            status_label = (
                "长时运行提醒"
                if long_running_alert
                else ("发现孤儿实例" if has_orphan else "GPU 运行计费中")
            )
        else:
            mood = "proud"
            status_level = "idle"
            status_label = "已停机省积分"

        balance_pct = (
            max(0.0, min(100.0, round(float(balance) / baseline * 100.0, 2)))
            if balance is not None and baseline > 0
            else 0.0
        )

        # 烧钱速度相对基准条（按常见 0~16 CU/hr 映射，最高 100%）
        rate_pct = max(0.0, min(100.0, round(rate_hourly / 15.0 * 100.0, 2)))

        # 保证看板与小账本两位小数严格对账（今日起始 + 今日充值 - 今日实耗 == 当前余额）
        disp_bal = round(float(balance), 2) if balance is not None else None
        raw_day_start = self.state.get("day_start_balance")
        disp_day_start = round(float(raw_day_start), 2) if raw_day_start is not None else None
        disp_day_topup = round(float(self.state.get("day_topup_cu") or 0.0), 2)
        if disp_bal is not None and disp_day_start is not None:
            disp_day_consumed = round(max(0.0, disp_day_start + disp_day_topup - disp_bal), 2)
        else:
            disp_day_consumed = round(float(self.state.get("day_consumed_cu") or 0.0), 2)

        sess_start = self.state.get("session_start_balance")
        if is_running and sess_start is None and active_since and balance is not None:
            ref_ts = int(updated_ts or now_ts)
            est = rate_hourly * max(0, ref_ts - int(active_since)) / 3600.0
            cap = (
                float(raw_day_start if raw_day_start is not None else balance)
                + float(self.state.get("day_topup_cu") or 0.0)
            )
            sess_start = (
                min(cap, float(balance) + max(0.0, est))
                if cap >= float(balance)
                else (float(balance) + max(0.0, est))
            )
        disp_session_consumed = (
            round(max(0.0, round(float(sess_start), 2) - disp_bal), 2)
            if (is_running and sess_start is not None and disp_bal is not None)
            else 0.0
        )

        return {
            "ok": error_msg is None,
            "stale": stale,
            "error_msg": error_msg,
            "balance": disp_bal,
            "baseline_balance": round(baseline, 2),
            "balance_pct": balance_pct,
            "rate_hourly": round(rate_hourly, 2),
            "rate_pct": rate_pct,
            "remaining_hours": remaining_hours,
            "assignments_count": assignments_count,
            "sessions": sessions,
            "is_running": is_running,
            "has_orphan": has_orphan,
            "active_minutes": active_minutes,
            "active_since_ts": int(active_since) if (is_running and active_since) else None,
            "long_running_alert": long_running_alert,
            "low_balance": low_balance,
            "mood": mood,
            "status_level": status_level,
            "status_label": status_label,
            "date": self.state.get("date"),
            "day_start_balance": disp_day_start,
            "day_consumed_cu": disp_day_consumed,
            "day_topup_cu": disp_day_topup,
            "session_consumed_cu": disp_session_consumed,
            "last_delta": round(float(self.state.get("last_delta") or 0.0), 3),
            "updated_str": updated_str,
            "updated_ts": updated_ts,
            "eligible_gpus": eligible_gpus,
            "eligible_tpus": eligible_tpus,
            "scarce_gpus": scarce_gpus,
            "measured_gpu_rates": dict(
                self.state.get("measured_gpu_rates") or DEFAULT_MEASURED_GPU_RATES
            ),
            "deploy": deploy_state,
            "prefs": prefs,
            "comfy": comfy,
        }

    def fetch_live_status(self) -> dict[str, Any]:
        """串行刷新一次线上状态并广播给前端；单次瞬断在宽限期内静默沿用近期可靠缓存。"""
        with self._fetch_lock:
            try:
                live = self._query_colab_live()
                with self._lock:
                    self._consecutive_failures = 0
                    self._update_ledger(live)
                    self._save_state()
                    snap = self._build_snapshot(live_data=live, stale=False, error_msg=None)
                    self.last_snapshot = snap
            except Exception as exc:
                with self._lock:
                    self._consecutive_failures += 1
                    has_balance = self.state.get("last_balance") is not None
                    last_updated = int(self.state.get("last_updated_ts") or 0)
                    within_grace = has_balance and (int(time.time()) - last_updated < 180)
                    if self._consecutive_failures == 1 and within_grace:
                        snap = self._build_snapshot(live_data=None, stale=False, error_msg=None)
                    else:
                        snap = self._build_snapshot(
                            live_data=None,
                            stale=True,
                            error_msg=str(exc),
                        )
                    self.last_snapshot = snap
            self._emit_event({"type": "state_sync", "snapshot": snap})
            return snap

    def get_cached_snapshot(self) -> dict[str, Any]:
        """返回最近一次状态快照。"""
        with self._lock:
            return dict(self.last_snapshot)

    def touch_activity(self) -> None:
        """刷新近期交互/推理活跃时间戳，用于智能保活与闲置自动停卡防误杀。"""
        with self._lock:
            self._last_activity_m = time.monotonic()
            self._idle_warned_ep = None

    def update_prefs(self, updates: dict[str, Any]) -> dict[str, Any]:
        """保存窗口位置、置顶状态、缩放比例与工作流启动参数（自动校验合法范围，不保存敏感项）。"""
        with self._lock:
            prefs = self.state.setdefault("prefs", dict(DEFAULT_PREFS))
            for k, v in (updates or {}).items():
                if k in DEFAULT_PREFS:
                    prefs[k] = v
            prefs = _sanitize_prefs_dict(prefs)
            self.state["prefs"] = prefs
            self._save_state()
            self.last_snapshot["prefs"] = dict(prefs)
            if (updates or {}).get("keep_alive") is True or "idle_auto_stop_minutes" in (updates or {}):
                self._last_keepalive_ts = 0.0
                self._last_activity_m = time.monotonic()
                self._idle_warned_ep = None
                self._wake_monitor.set()
            return dict(prefs)

    save_prefs = update_prefs

    def _pick_target_session(self, preferred: str = "comfy") -> dict[str, Any] | None:
        """优先选择指定会话，若不存在则取首个活跃会话。"""
        with self._lock:
            sessions = list(self.state.get("last_sessions") or [])
        for s in sessions:
            if s.get("name") == preferred:
                return s
        return sessions[0] if sessions else None

    def _has_running_session(self) -> bool:
        """判断当前缓存状态中是否存在活跃 Colab GPU 会话或计费速率。"""
        with self._lock:
            ac = int(self.state.get("last_assignments_count") or 0)
            rate = float(self.state.get("last_rate_hourly") or 0.0)
            sessions = list(self.state.get("last_sessions") or [])
        return ac > 0 or rate > 0.01 or len(sessions) > 0

    def _get_ep_meta_and_key(self, ep: str = "") -> tuple[dict[str, Any], str]:
        """线程安全读取指定 endpoint 的内存态服务元信息与访问密钥。"""
        with self._lock:
            if ep and ep in self._service_meta_by_ep:
                return dict(self._service_meta_by_ep[ep]), str(self._runtime_secrets.get(ep) or "")
            if not ep and self._service_meta_by_ep:
                first_ep, first_meta = next(iter(self._service_meta_by_ep.items()))
                return dict(first_meta), str(self._runtime_secrets.get(first_ep) or "")
            return {}, str(self._runtime_secrets.get(ep) or "") if ep else ""

    def _get_ep_service_kind(self, ep: str = "") -> str:
        """根据 endpoint 元信息、当前在线监控态或开机预设推断当前服务类型 ('comfy' | 'llm')。"""
        with self._lock:
            target_ep = ep
            if not target_ep:
                sessions = list(self.state.get("last_sessions") or [])
                primary = next(
                    (s for s in sessions if s.get("name") == "comfy" and s.get("endpoint")),
                    None,
                ) or (sessions[0] if sessions else None)
                if primary and primary.get("endpoint"):
                    target_ep = str(primary["endpoint"])
            if target_ep and target_ep in self._service_meta_by_ep:
                return str(self._service_meta_by_ep[target_ep].get("kind") or "comfy")
            if self._service_meta_by_ep:
                first_meta = next(iter(self._service_meta_by_ep.values()))
                return str(first_meta.get("kind") or "comfy")
            boot_prof = str((self.state.get("prefs") or {}).get("boot_profile") or "anima")
        with self._comfy_lock:
            cur_kind = str(self._comfy_state.get("service_kind") or "")
            if self._comfy_state.get("online") and cur_kind in ("comfy", "llm"):
                return cur_kind
        return "llm" if boot_prof == "qwen_llm" else "comfy"

    def is_llm_mode_active(self) -> bool:
        """判断当前活跃实例（或停机状态下的预设工作流）是否处于 LLM 对话模式。"""
        if self._has_running_session():
            return self._get_ep_service_kind() == "llm"
        with self._lock:
            prof = str((self.state.get("prefs") or {}).get("boot_profile") or "anima")
        return prof == "qwen_llm"

    def _fast_read_tunnel_url(self, ep: str, sname: str) -> tuple[str, str, str | None]:
        """通过 Colab Contents HTTP 接口免 WebSocket 速读 /root/colab_service_meta.json 或隧道文件。"""
        if not COLAB_PY.exists():
            return "error", ep, None
        fast_proc = _run_hidden(
            [str(COLAB_PY), "-c", _COLAB_FAST_TUNNEL_SCRIPT, ep, sname],
            timeout=15,
        )
        if fast_proc.returncode != 0:
            return "error", ep, None
        m_json = re.search(
            r"<<<COLAB_TUNNEL>>>(.*?)<<<END_COLAB_TUNNEL>>>",
            fast_proc.stdout or "",
            re.S,
        )
        if not m_json:
            return "error", ep, None
        try:
            tdata = json.loads(m_json.group(1))
        except Exception:
            tdata = {}
        tstatus = str(tdata.get("status") or "missing")
        if tstatus == "no_session":
            with self._lock:
                self.state["last_sessions"] = []
                self.state["last_assignments_count"] = 0
                self.state["tunnel_cache"] = {}
                self._service_meta_by_ep.clear()
                self._runtime_secrets.clear()
                self._save_state()
            return "no_session", ep, None
        resolved_ep = str(tdata.get("endpoint") or ep)
        fast_url = str(tdata.get("url") or "").strip() or None

        # 将 api_key 严格剥离至纯内存态字典，非敏感元信息存入 _service_meta_by_ep
        raw_meta = tdata.get("meta")
        api_key = str(tdata.get("api_key") or "").strip()
        if resolved_ep and isinstance(raw_meta, dict):
            if not api_key and raw_meta.get("api_key"):
                api_key = str(raw_meta.get("api_key") or "").strip()
            clean_meta = {k: v for k, v in raw_meta.items() if k != "api_key"}
            kind_val = str(clean_meta.get("kind") or "comfy").lower()
            clean_meta["kind"] = "llm" if kind_val == "llm" else "comfy"
            clean_meta.setdefault("port", 18080 if clean_meta["kind"] == "llm" else 8188)
            if fast_url:
                clean_meta["tunnel_url"] = fast_url
            with self._lock:
                self._service_meta_by_ep[resolved_ep] = clean_meta
                if clean_meta["kind"] == "llm":
                    if api_key:
                        self._runtime_secrets[resolved_ep] = api_key
                else:
                    self._runtime_secrets.pop(resolved_ep, None)

        return tstatus, resolved_ep, fast_url

    def resolve_active_tunnel_url(
        self, session: str = "comfy", allow_fast_discover: bool = True
    ) -> str | None:
        """获取当前可连通的 ComfyUI / LLM 公网隧道地址（支持重启后缓存补水元数据与密钥）。"""
        if not self._has_running_session():
            return None
        target = self._pick_target_session(session)
        ep = str(target.get("endpoint") or "") if target else ""
        sname = target["name"] if target and target.get("name") not in (None, "?") else session

        cached_url: str | None = None
        known_meta: dict[str, Any] | None = None
        preferred_kind = "comfy"
        with self._lock:
            cache = self.state.get("tunnel_cache") or {}
            if ep and ep in cache:
                cached_url = cache.get(ep)
            elif cache:
                cached_url = next(iter(cache.values()), None)
            if ep and ep in self._service_meta_by_ep:
                known_meta = dict(self._service_meta_by_ep[ep])
            if str((self.state.get("prefs") or {}).get("boot_profile") or "") == "qwen_llm":
                preferred_kind = "llm"

        if cached_url:
            if known_meta is not None:
                mkind = str(known_meta.get("kind") or "comfy")
                if _probe_tunnel_alive(cached_url, timeout=3.0, kind=mkind):
                    return cached_url
            else:
                detected = _detect_tunnel_kind(cached_url, timeout=3.0, preferred=preferred_kind)
                if detected is not None:
                    # 缓存存活但进程刚启动缺少元信息与密钥：主动补水一次
                    if allow_fast_discover:
                        now_m = time.monotonic()
                        if now_m - self._last_tunnel_discover_ts >= 15.0:
                            self._last_tunnel_discover_ts = now_m
                            tstatus, resolved_ep, fast_url = self._fast_read_tunnel_url(ep, sname)
                            if (
                                tstatus == "ok"
                                and fast_url
                                and fast_url != cached_url
                                and _probe_tunnel_alive(fast_url, timeout=3.0, kind=detected)
                            ):
                                if resolved_ep:
                                    with self._lock:
                                        self.state.setdefault("tunnel_cache", {})[resolved_ep] = fast_url
                                        self._save_state()
                                return fast_url
                    if ep:
                        with self._lock:
                            self._service_meta_by_ep.setdefault(
                                ep,
                                {"kind": detected, "port": 18080 if detected == "llm" else 8188},
                            )
                    return cached_url

        if not allow_fast_discover:
            return None
        now_m = time.monotonic()
        if now_m - self._last_tunnel_discover_ts < 45.0:
            return None
        self._last_tunnel_discover_ts = now_m

        tstatus, resolved_ep, fast_url = self._fast_read_tunnel_url(ep, sname)
        if tstatus == "ok" and fast_url:
            meta_after, _ = self._get_ep_meta_and_key(resolved_ep or ep)
            probe_kind = str(meta_after.get("kind") or "auto")
            if _probe_tunnel_alive(fast_url, timeout=3.5, kind=probe_kind):
                if resolved_ep:
                    with self._lock:
                        self.state.setdefault("tunnel_cache", {})[resolved_ep] = fast_url
                        self._save_state()
                return fast_url
        return None

    _get_or_fetch_tunnel_url = resolve_active_tunnel_url

    @staticmethod
    def _build_comfy_open_url(
        tunnel_url: str,
        workflow_name: str | None = None,
        extra_names: list[str] | None = None,
    ) -> str:
        """构造 ComfyUI 打开链接；指定工作流名时附加一次性直铺画布查询参数。"""
        base = tunnel_url.rstrip("/")
        if not workflow_name:
            return base
        q = f"?pet_wf={urllib.parse.quote(workflow_name)}&pet_ts={int(time.time() * 1000)}"
        extras = [str(n) for n in (extra_names or []) if n and n != workflow_name]
        if extras:
            q += f"&pet_extra={urllib.parse.quote(','.join(extras))}"
        return f"{base}/{q}"

    def _ensure_legacy_cmd_file(self, base_url: str) -> None:
        """确保 userdata 根目录存在桥接指令文件，避免浏览器首次轮询出现 404 日志。"""
        base = base_url.rstrip("/")
        try:
            chk = self._http.get(f"{base}/userdata/colab_pet_bridge_cmd.json", timeout=4)
            if chk.status_code == 404:
                self._http.post(
                    f"{base}/userdata/colab_pet_bridge_cmd.json?overwrite=true",
                    json={"id": "", "name": "", "ts": 0},
                    timeout=5,
                )
        except Exception:
            pass

    def _ensure_direct_canvas_support(
        self, tunnel_url: str, ep: str = "", sname: str = "comfy"
    ) -> tuple[str, bool]:
        """探测或免重启热注入/热升级云端 ComfyUI 直铺画布前端扩展，返回 `(模式, 本次是否刚安装)`。"""
        base = tunnel_url.rstrip("/")
        cache_key = ep or base
        with self._canvas_ext_lock:
            cached = self._canvas_ext_ready_eps.get(cache_key)
        if cached in ("native", "legacy"):
            return cached, False

        native_route_alive = False
        # 1) 优先探测新版实例内置自定义节点接口（若服务端仍是 v1 则进一步校验 JS 是否已热升级到 v2）
        for attempt in range(2):
            try:
                r = self._http.get(f"{base}/colab_pet/ping", timeout=4)
                if r.status_code == 200:
                    pdata = r.json()
                    if isinstance(pdata, dict) and pdata.get("ok"):
                        native_route_alive = True
                        ver = int(pdata.get("version") or 1)
                        if ver >= 2:
                            with self._canvas_ext_lock:
                                self._canvas_ext_ready_eps[cache_key] = "native"
                            return "native", False
                        try:
                            jr = self._http.get(
                                f"{base}/extensions/colab_pet_bridge/direct_canvas.js",
                                timeout=4,
                            )
                            if jr.status_code == 200 and "colab_pet_direct_canvas v2" in (jr.text or ""):
                                with self._canvas_ext_lock:
                                    self._canvas_ext_ready_eps[cache_key] = "native"
                                return "native", False
                        except Exception:
                            pass
                break
            except Exception:
                if attempt == 0:
                    time.sleep(0.25)

        # 2) 当服务端路由不存在时，探测已挂载的前端扩展（仅认可 v2，旧 v1 将自动覆盖升级）
        if not native_route_alive:
            for attempt in range(2):
                try:
                    r2 = self._http.get(f"{base}/extensions/colab_pet/direct_canvas.js", timeout=4)
                    if r2.status_code == 200 and "colab_pet_direct_canvas v2" in (r2.text or ""):
                        self._ensure_legacy_cmd_file(base)
                        with self._canvas_ext_lock:
                            self._canvas_ext_ready_eps[cache_key] = "legacy"
                        return "legacy", False
                    break
                except Exception:
                    if attempt == 0:
                        time.sleep(0.25)

        # 3) 存量实例尚未挂载或仍为 v1 时，经 Colab Contents REST 免重启热写入/覆盖 v2 扩展
        if not COLAB_PY.exists() or not self._has_running_session():
            return ("native" if native_route_alive else "none"), False

        try:
            payload = {
                "js_b64": base64.b64encode(_COMFY_DIRECT_CANVAS_JS.encode("utf-8")).decode("ascii"),
                "py_b64": base64.b64encode(_COMFY_BRIDGE_NODE_INIT_PY.encode("utf-8")).decode("ascii"),
            }
            proc = subprocess.run(
                [str(COLAB_PY), "-c", _COLAB_INSTALL_CANVAS_EXT_SCRIPT, ep, sname],
                input=json.dumps(payload, ensure_ascii=False),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
                env=_clean_env(),
                creationflags=CREATE_NO_WINDOW,
            )
            if proc.returncode != 0:
                return ("native" if native_route_alive else "none"), False

            # 验证静态资源已立刻通过 ComfyUI 隧道返回 v2 内容
            check_urls = (
                [
                    f"{base}/extensions/colab_pet_bridge/direct_canvas.js",
                    f"{base}/extensions/colab_pet/direct_canvas.js",
                ]
                if native_route_alive
                else [f"{base}/extensions/colab_pet/direct_canvas.js"]
            )
            for _ in range(3):
                for cu in check_urls:
                    try:
                        vr = self._http.get(cu, timeout=4)
                        if vr.status_code == 200 and "colab_pet_direct_canvas v2" in (vr.text or ""):
                            target_mode = "native" if native_route_alive else "legacy"
                            if target_mode == "legacy":
                                self._ensure_legacy_cmd_file(base)
                            with self._canvas_ext_lock:
                                self._canvas_ext_ready_eps[cache_key] = target_mode
                            return target_mode, True
                    except Exception:
                        pass
                time.sleep(0.25)
        except Exception:
            pass
        return ("native" if native_route_alive else "none"), False

    def _push_or_open_workflow_on_canvas(
        self,
        tunnel_url: str,
        safe_name: str,
        extra_names: list[str] | None = None,
        session: str = "comfy",
    ) -> dict[str, Any]:
        """优先向已打开的 ComfyUI 浏览器标签页实时推送工作流；无标签页回执时自动拉起浏览器直铺。"""
        base = tunnel_url.rstrip("/")
        target = self._pick_target_session(session)
        ep = str(target.get("endpoint") or "") if target else ""
        sname = target["name"] if target and target.get("name") not in (None, "?") else session

        mode, just_installed = self._ensure_direct_canvas_support(base, ep=ep, sname=sname)
        req_id = uuid.uuid4().hex[:12]
        open_url = self._build_comfy_open_url(base, safe_name, extra_names=extra_names)

        if mode == "native" and not just_installed:
            try:
                pr = self._http.post(
                    f"{base}/colab_pet/open_workflow",
                    json={"id": req_id, "name": safe_name, "extra_names": extra_names or []},
                    timeout=5,
                )
                if pr.status_code == 200:
                    pdata = pr.json()
                    if int(pdata.get("browser_clients") or 0) > 0:
                        deadline = time.monotonic() + 3.2
                        while time.monotonic() < deadline:
                            time.sleep(0.18)
                            try:
                                ar = self._http.get(
                                    f"{base}/colab_pet/ack_workflow",
                                    params={"id": req_id},
                                    timeout=3,
                                )
                                if ar.status_code == 200 and ar.json().get("acked"):
                                    return {
                                        "pushed_live": True,
                                        "opened_browser": False,
                                        "url": open_url,
                                        "mode": mode,
                                    }
                            except Exception:
                                pass
            except Exception:
                pass
        elif mode == "legacy" and not just_installed:
            try:
                pr = self._http.post(
                    f"{base}/userdata/colab_pet_bridge_cmd.json?overwrite=true",
                    json={
                        "id": req_id,
                        "name": safe_name,
                        "extra_names": extra_names or [],
                        "ts": time.time(),
                    },
                    timeout=5,
                )
                if pr.status_code in (200, 201):
                    deadline = time.monotonic() + 3.5
                    while time.monotonic() < deadline:
                        time.sleep(0.25)
                        try:
                            ar = self._http.get(
                                f"{base}/userdata/colab_pet_bridge_ack.json",
                                timeout=3,
                            )
                            if ar.status_code == 200 and str((ar.json() or {}).get("id") or "") == req_id:
                                return {
                                    "pushed_live": True,
                                    "opened_browser": False,
                                    "url": open_url,
                                    "mode": mode,
                                }
                        except Exception:
                            pass
            except Exception:
                pass

        webbrowser.open(open_url)
        return {
            "pushed_live": False,
            "opened_browser": True,
            "url": open_url,
            "mode": mode,
        }

    def _backup_workflow_to_gdrive_async(self, safe_name: str, wf_bytes: bytes) -> None:
        """在后台静默同步一份工作流 JSON 至 Google Drive，实现跨实例开机自动继承。"""
        if not RCLONE_EXE.is_file() or not RCLONE_CONF.is_file():
            return

        def _bg() -> None:
            try:
                with tempfile.TemporaryDirectory(prefix="pet_wf_bak_") as tmp_dir:
                    tmp_wf = Path(tmp_dir) / safe_name
                    tmp_wf.write_bytes(wf_bytes)
                    _run_hidden(
                        [
                            str(RCLONE_EXE),
                            "copyto",
                            "--config",
                            str(RCLONE_CONF),
                            "--disable-http2",
                            str(tmp_wf),
                            f"{GDRIVE_WF_REMOTE}/{safe_name}",
                        ],
                        timeout=45,
                    )
            except Exception:
                pass

        threading.Thread(target=_bg, daemon=True).start()

    def _open_browser_for_comfy(
        self,
        tunnel_url: str,
        ep: str = "",
        sname: str = "comfy",
        workflow_name: str | None = None,
    ) -> str:
        """装配 ComfyUI 打开链接并拉起默认浏览器（若携带工作流则顺带确保扩展就绪）。"""
        if workflow_name:
            self._ensure_direct_canvas_support(tunnel_url, ep=ep, sname=sname)
        open_url = self._build_comfy_open_url(tunnel_url, workflow_name)
        webbrowser.open(open_url)
        return open_url

    def _open_resolved_service_url(
        self,
        tunnel_url: str,
        ep: str = "",
        sname: str = "comfy",
        workflow_name: str | None = None,
        stage: str = "normal",
        detected_kind: str | None = None,
        already_fetched_meta: bool = False,
    ) -> dict[str, Any]:
        """根据解析出的存活隧道地址分流执行打开动作（LLM 直开对话页并回传密钥，ComfyUI 打开画布）。"""
        meta, api_key = self._get_ep_meta_and_key(ep)
        resolved_ep = ep
        if not meta and not already_fetched_meta:
            _, r_ep, _ = self._fast_read_tunnel_url(ep, sname)
            if r_ep:
                resolved_ep = r_ep
            meta, api_key = self._get_ep_meta_and_key(resolved_ep)
            already_fetched_meta = True

        kind = str(meta.get("kind") or detected_kind or "comfy").lower()
        if kind == "llm":
            if not api_key and not already_fetched_meta:
                _, r_ep, _ = self._fast_read_tunnel_url(resolved_ep, sname)
                if r_ep:
                    resolved_ep = r_ep
                _, api_key = self._get_ep_meta_and_key(resolved_ep)
            clean_url = tunnel_url.strip().rstrip("/")
            webbrowser.open(clean_url)
            msg = (
                "💬 已打开 Qwen3.8 对话页 · 访问密钥已复制，首次直接 Ctrl+V 登录"
                if api_key
                else "💬 已打开 Qwen3.8 对话页"
            )
            return {
                "ok": True,
                "url": clean_url,
                "service_kind": "llm",
                "clipboard_text": api_key,
                "key_copied": bool(api_key),
                "message": msg,
            }

        open_url = self._open_browser_for_comfy(
            tunnel_url, ep=resolved_ep, sname=sname, workflow_name=workflow_name
        )
        if workflow_name:
            if stage == "restarted":
                msg = f"已重启隧道并打开工作流: {workflow_name}"
            else:
                msg = f"已在画布打开工作流: {workflow_name}"
        else:
            if stage == "cache":
                msg = "已秒开 ComfyUI 隧道 (缓存命中)"
            elif stage == "restarted":
                msg = "已重启并打开 ComfyUI 隧道网页"
            else:
                msg = "已打开 ComfyUI 隧道网页"
        return {"ok": True, "url": open_url, "service_kind": "comfy", "message": msg}

    def open_comfy_tunnel(
        self, session: str = "comfy", workflow_name: str | None = None
    ) -> dict[str, Any]:
        """三级极速打开云端 ComfyUI / LLM 隧道：本地缓存秒开 -> HTTP Contents 直读 -> 自动重启兜底。"""
        self.touch_activity()
        if self.is_deploying():
            return {"ok": False, "message": "正在执行 7 步开机部署，请等待就绪后再打开"}
        if not self._action_lock.acquire(blocking=False):
            return {"ok": False, "message": "另一项操作正在执行，请稍等片刻"}
        try:
            # 0. 若本地暂无活跃会话缓存则即时探活；确无实例时秒级短路提示
            target = self._pick_target_session(session)
            if target is None:
                snap = self.fetch_live_status()
                target = self._pick_target_session(session)
                if target is None and snap.get("assignments_count", 0) == 0 and not snap.get("sessions"):
                    return {"ok": False, "message": "当前无运行中的 Colab 实例，请先部署开机"}

            ep = str(target.get("endpoint") or "") if target else ""
            sname = target["name"] if target and target.get("name") not in (None, "?") else session
            is_orphan = bool(target and target.get("orphan") and target.get("name") in (None, "?"))

            # 1. 第一级：本地持久化公网域名缓存命中且双模探活通过 -> <1s 秒开
            cached_url = None
            known_meta, _ = self._get_ep_meta_and_key(ep)
            preferred_kind = "llm" if self._get_ep_service_kind(ep) == "llm" else "comfy"
            if ep:
                with self._lock:
                    cached_url = (self.state.get("tunnel_cache") or {}).get(ep)
            if cached_url:
                if known_meta:
                    mkind = str(known_meta.get("kind") or "comfy")
                    if _probe_tunnel_alive(cached_url, timeout=3.0, kind=mkind):
                        return self._open_resolved_service_url(
                            cached_url,
                            ep=ep,
                            sname=sname,
                            workflow_name=workflow_name,
                            stage="cache",
                            detected_kind=mkind,
                        )
                else:
                    det_kind = _detect_tunnel_kind(cached_url, timeout=3.0, preferred=preferred_kind)
                    if det_kind is not None:
                        return self._open_resolved_service_url(
                            cached_url,
                            ep=ep,
                            sname=sname,
                            workflow_name=workflow_name,
                            stage="cache",
                            detected_kind=det_kind,
                        )

            # 2. 第二级：通过 Colab Contents HTTP REST 免 WebSocket 直读服务元数据契约或隧道文件
            tstatus, resolved_ep, fast_url = self._fast_read_tunnel_url(ep, sname)
            if tstatus == "no_session":
                return {"ok": False, "message": "当前无运行中的 Colab 实例，请先部署开机"}
            meta_fast, _ = self._get_ep_meta_and_key(resolved_ep or ep)
            fast_kind = str(meta_fast.get("kind") or "auto")
            if tstatus == "ok" and fast_url and _probe_tunnel_alive(fast_url, timeout=4.0, kind=fast_kind):
                if resolved_ep:
                    with self._lock:
                        self.state.setdefault("tunnel_cache", {})[resolved_ep] = fast_url
                        self._save_state()
                resolved_kind = fast_kind if fast_kind in ("comfy", "llm") else _detect_tunnel_kind(fast_url, timeout=2.5)
                return self._open_resolved_service_url(
                    fast_url,
                    ep=resolved_ep or ep,
                    sname=sname,
                    workflow_name=workflow_name,
                    stage="normal",
                    detected_kind=resolved_kind,
                    already_fetched_meta=True,
                )

            # 网页端开启的孤儿实例无法通过本地 session 名称重建隧道，直接给出清晰引导
            if is_orphan:
                return {
                    "ok": False,
                    "message": "该实例为网页孤儿实例且未检测到可用隧道，请用面板一键开机或在网页端重启隧道",
                }

            # 3. 第三级：前两级均不可用时回退 colab_ctl.py tunnel --restart 重建隧道
            if not CTL_SCRIPT.exists():
                return {"ok": False, "message": f"未找到控制脚本: {CTL_SCRIPT.name}"}
            proc = _run_hidden(
                [_hidden_python_exe(), str(CTL_SCRIPT), "tunnel", "--restart", "-s", sname],
                timeout=60,
            )
            output = f"{proc.stdout}\n{proc.stderr}"
            match = re.search(
                r"(?:(?:COMFYUI|SERVICE)_TUNNEL_URL:\s*|\[(?:ComfyUI|LLM|Qwen[^]]*) WebUI 地址\]\s*)(https://(?!api\.)(?:[a-z0-9-]+\.)+[a-z]{2,}(?::\d+)?)",
                output,
            )
            if proc.returncode == 0 and match:
                url = match.group(1).strip()
                use_ep = resolved_ep or ep
                if use_ep:
                    with self._lock:
                        self.state.setdefault("tunnel_cache", {})[use_ep] = url
                        if use_ep in self._service_meta_by_ep:
                            self._service_meta_by_ep[use_ep]["tunnel_url"] = url
                        self._save_state()
                det_after = _detect_tunnel_kind(url, timeout=3.0, preferred=preferred_kind)
                return self._open_resolved_service_url(
                    url,
                    ep=use_ep,
                    sname=sname,
                    workflow_name=workflow_name,
                    stage="restarted",
                    detected_kind=det_after,
                    already_fetched_meta=bool(meta_fast),
                )

            if ep:
                with self._lock:
                    cache = self.state.get("tunnel_cache")
                    if isinstance(cache, dict) and ep in cache:
                        cache.pop(ep, None)
                        self._save_state()
            tail = (output.strip().splitlines() or ["未获取到隧道地址"])[-1]
            return {"ok": False, "message": f"隧道开启失败: {tail}"}
        except Exception as exc:
            return {"ok": False, "message": f"隧道操作异常: {exc}"}
        finally:
            self._action_lock.release()

    def get_llm_api_bundle(self, session: str = "comfy") -> dict[str, Any]:
        """组装当前在线 Qwen3.8 实例的 OpenAI 兼容接口配置包（含剪贴板多行文本，不持久化密钥）。"""
        self.touch_activity()
        if self.is_deploying():
            return {"ok": False, "message": "正在执行 7 步开机部署，请等待就绪后再复制配置"}

        target = self._pick_target_session(session)
        if target is None and not self._has_running_session():
            snap = self.fetch_live_status()
            target = self._pick_target_session(session)
            if target is None and snap.get("assignments_count", 0) == 0 and not snap.get("sessions"):
                return {"ok": False, "message": "当前无运行中的 Qwen3.8 实例，请先部署开机"}

        ep = str(target.get("endpoint") or "") if target else ""
        sname = target["name"] if target and target.get("name") not in (None, "?") else session

        turl = self.resolve_active_tunnel_url(session=sname, allow_fast_discover=True)
        fetched_now = False
        if not turl:
            tstatus, resolved_ep, fast_url = self._fast_read_tunnel_url(ep, sname)
            fetched_now = True
            if tstatus == "no_session":
                return {"ok": False, "message": "当前无运行中的 Qwen3.8 实例，请先部署开机"}
            if resolved_ep:
                ep = resolved_ep
            if fast_url and _probe_tunnel_alive(fast_url, timeout=3.5, kind="llm"):
                turl = fast_url
                if ep:
                    with self._lock:
                        self.state.setdefault("tunnel_cache", {})[ep] = fast_url
                        self._save_state()

        meta, key = self._get_ep_meta_and_key(ep)
        if (not meta or not key) and not fetched_now:
            _, resolved_ep, fast_url = self._fast_read_tunnel_url(ep, sname)
            if resolved_ep:
                ep = resolved_ep
            meta, key = self._get_ep_meta_and_key(ep)
            if not turl and fast_url and _probe_tunnel_alive(fast_url, timeout=3.5, kind="llm"):
                turl = fast_url

        if meta and str(meta.get("kind") or "").lower() == "comfy":
            with self._comfy_lock:
                cur_sk = str(self._comfy_state.get("service_kind") or "comfy")
            if cur_sk != "llm":
                return {"ok": False, "message": "当前运行的是 ComfyUI 实例，非 LLM 对话服务"}

        if not turl:
            with self._comfy_lock:
                if self._comfy_state.get("service_kind") == "llm" and self._comfy_state.get("tunnel_url"):
                    turl = str(self._comfy_state["tunnel_url"])

        if not turl:
            return {"ok": False, "message": "暂未获取到在线 Qwen3.8 隧道，请稍候或先点打开对话页"}

        clean_turl = turl.strip().rstrip("/")
        base_url = f"{clean_turl}/v1"
        model = str(meta.get("model") or "Qwen3.8-Flash-Next-abliterated")
        clipboard_text = f"Base URL: {base_url}\nAPI Key: {key}\nModel: {model}"
        return {
            "ok": True,
            "base_url": base_url,
            "model": model,
            "has_key": bool(key),
            "clipboard_text": clipboard_text,
            "message": "📋 已复制 OpenAI 接口配置 (Base URL + Key)",
        }

    copy_llm_api_config = get_llm_api_bundle

    def open_colab_url(self, session: str = "comfy") -> dict[str, Any]:
        """在浏览器中打开 Colab 运行时关联网页或 Colab 首页。"""
        self.touch_activity()
        target = self._pick_target_session(session)
        if target and target.get("endpoint"):
            ep = target["endpoint"]
            url = (
                "https://colab.research.google.com/notebooks/empty.ipynb"
                f"?dbu=%2Ftun%2Fm%2F{ep}"
                f"#datalabBackendUrl=https://colab.research.google.com/tun/m/{ep}"
            )
            webbrowser.open(url)
            label = target.get("name") or ep
            return {"ok": True, "url": url, "message": f"已打开 Colab 会话 ({label})"}

        colab_exe = _find_colab_exe()
        if colab_exe:
            try:
                proc = _run_hidden([colab_exe, "url", "-s", session], timeout=12)
                m = re.search(r"https://colab\.research\.google\.com\S+", proc.stdout)
                if proc.returncode == 0 and m:
                    webbrowser.open(m.group(0))
                    return {"ok": True, "url": m.group(0), "message": "已打开 Colab 会话页"}
            except Exception:
                pass

        home_url = "https://colab.research.google.com/"
        webbrowser.open(home_url)
        return {"ok": True, "url": home_url, "message": "当前无活跃会话，已打开 Colab 主页"}

    def sync_outputs(self, session: str = "comfy") -> dict[str, Any]:
        """将云端 ComfyUI 成图拉回本地目录；若当前活跃为 LLM 对话服务则平滑分流为返回 OpenAI 接口配置。"""
        self.touch_activity()
        if self.is_deploying():
            return {"ok": False, "message": "正在执行 7 步开机部署，请等待就绪后再操作"}
        if self.is_llm_mode_active():
            return self.get_llm_api_bundle(session=session)
        if not self._action_lock.acquire(blocking=False):
            return {"ok": False, "message": "另一项操作正在执行，请稍等片刻"}
        try:
            target = self._pick_target_session(session)
            if target is None:
                snap = self.fetch_live_status()
                target = self._pick_target_session(session)
                if (
                    target is None
                    and snap.get("assignments_count", 0) == 0
                    and not snap.get("sessions")
                ):
                    self.download_root.mkdir(parents=True, exist_ok=True)
                    try:
                        os.startfile(str(self.download_root))
                    except Exception:
                        pass
                    return {"ok": True, "message": "当前未开机，已打开本地成图目录"}

            # 二次核查远端是否实际运行着 LLM 服务（兼容刚启动桌宠且尚未跑过首轮探针的场景）
            ep_chk = str(target.get("endpoint") or "") if target else ""
            sname_chk = target["name"] if target and target.get("name") not in (None, "?") else session
            meta_chk, _ = self._get_ep_meta_and_key(ep_chk)
            if not meta_chk and ep_chk:
                self._fast_read_tunnel_url(ep_chk, sname_chk)
                meta_chk, _ = self._get_ep_meta_and_key(ep_chk)
            if meta_chk and str(meta_chk.get("kind") or "").lower() == "llm":
                return self.get_llm_api_bundle(session=sname_chk)

            if not CTL_SCRIPT.exists():
                return {"ok": False, "message": f"未找到控制脚本: {CTL_SCRIPT.name}"}
            sname = sname_chk
            proc = _run_hidden(
                [_hidden_python_exe(), str(CTL_SCRIPT), "sync", "--pull", "-s", sname],
                timeout=180,
            )
            output = f"{proc.stdout}\n{proc.stderr}".strip()
            tail = (output.splitlines() or ["同步结束"])[-1]
            if proc.returncode == 0:
                if self.download_root.exists():
                    try:
                        os.startfile(str(self.download_root))  # Windows 直接打开本地收图文件夹
                    except Exception:
                        pass
                return {"ok": True, "message": f"拉回完成: {tail}"}
            return {"ok": False, "message": f"同步失败: {tail}"}
        except Exception as exc:
            return {"ok": False, "message": f"同步异常: {exc}"}
        finally:
            self._action_lock.release()

    def stop_all_instances(self) -> dict[str, Any]:
        """一键停止全部 Colab 活跃实例与孤儿运行时并立即刷新余额。"""
        if self.is_deploying():
            self.cancel_deploy()
            time.sleep(0.4)
        if not self._action_lock.acquire(blocking=False):
            return {"ok": False, "message": "另一项操作正在执行，请稍等片刻"}
        try:
            if not CTL_SCRIPT.exists():
                return {"ok": False, "message": f"未找到控制脚本: {CTL_SCRIPT.name}"}
            proc = _run_hidden(
                [_hidden_python_exe(), str(CTL_SCRIPT), "down", "--all"],
                timeout=60,
            )
            output = f"{proc.stdout}\n{proc.stderr}".strip()
            if proc.returncode == 0:
                with self._lock:
                    self._service_meta_by_ep.clear()
                    self._runtime_secrets.clear()
                    self.state["tunnel_cache"] = {}
                    self._save_state()
                self._reset_comfy_offline()
            snap = self.fetch_live_status()
            if proc.returncode == 0:
                return {"ok": True, "message": "已关闭全部 Colab 实例", "snapshot": snap}
            tail = (output.splitlines() or ["停机指令异常"])[-1]
            return {"ok": False, "message": f"停机失败: {tail}", "snapshot": snap}
        except Exception as exc:
            return {"ok": False, "message": f"停机异常: {exc}"}
        finally:
            self._action_lock.release()

    # =========================================================================
    # 功能 2.5：面板直接选卡一键开机、实时算力紧缺检测与 7 步进度直播
    # =========================================================================

    @staticmethod
    def _format_elapsed_cn(seconds: int) -> str:
        """将秒数格式化为简短可读的中文耗时字符串。"""
        sec = max(0, int(seconds))
        if sec < 60:
            return f"{sec}秒"
        return f"{sec // 60}分{sec % 60:02d}秒"

    @staticmethod
    def _kill_process_tree(proc: Any) -> None:
        """尽力终止部署子进程及其 Windows 派生树（兼容测试替身对象）。"""
        if proc is None:
            return
        pid = getattr(proc, "pid", None)
        if os.name == "nt" and isinstance(pid, int) and pid > 0:
            try:
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(pid)],
                    capture_output=True,
                    creationflags=CREATE_NO_WINDOW,
                    timeout=8,
                )
            except Exception:
                pass
        for method_name in ("terminate", "kill"):
            fn = getattr(proc, method_name, None)
            if callable(fn):
                try:
                    fn()
                except Exception:
                    pass

    @staticmethod
    def _recommend_fallback_gpu(
        failed_gpu: str,
        profile: str,
        eligible_gpus: list[str] | None,
        scarce_gpus: dict[str, Any],
    ) -> str | None:
        """在指定显卡缺卡或受限时，按生态偏好筛选首个当前可用替代 GPU。"""
        failed_up = str(failed_gpu or "").upper()
        if profile == "qwen_llm":
            # 180B 稀疏模型严禁在 L4/T4 运行，备选池仅限高显存卡
            order: tuple[str, ...] = ("G4", "A100", "H100")
        elif profile in ("minimax", "drama3"):
            order = ("G4", "A100", "H100", "L4", "T4")
        else:
            order = ("L4", "A100", "T4", "G4", "H100")

        allowed: set[str] | None = None
        if isinstance(eligible_gpus, list) and len(eligible_gpus) > 0:
            allowed = {str(g).upper() for g in eligible_gpus if g}

        scarce_set = {str(k).upper() for k in (scarce_gpus or {}).keys()}
        for cand in order:
            if cand == failed_up or cand in scarce_set:
                continue
            if allowed is not None and cand not in allowed:
                continue
            return cand
        return None

    @staticmethod
    def _classify_deploy_failure(
        lines: list[str],
        gpu: str,
        pre_assignments: int,
        profile: str,
        recreate: bool,
    ) -> dict[str, Any]:
        """分析开机/自举失败末尾日志，精准区分高峰期缺卡、配额受限与环境异常。"""
        cleaned = [ln.strip() for ln in lines if ln and ln.strip()]
        joined = "\n".join(cleaned)
        joined_low = joined.lower()
        gpu_up = str(gpu or "").upper()

        if "backend rejected accelerator" in joined_low or "not have quota or entitlement" in joined_low:
            return {
                "kind": "entitlement",
                "stockout_gpu": gpu_up,
                "reason": f"当前账号配额暂不支持 {gpu_up}",
                "message": f"⛔ 当前 Colab 账号无 {gpu_up} 配额或等级权限",
            }

        is_precondition = (
            "allocation refused" in joined_low
            or "toomanyassignments" in joined_low
            or ("412" in joined_low and "precondition" in joined_low)
        )
        is_capacity = (
            "resourceexhausted" in joined_low
            or "out of capacity" in joined_low
            or "no backend" in joined_low
            or "cannot connect to gpu backend" in joined_low
            or "failed to assign" in joined_low
            or ("503" in joined_low and "assign" in joined_low)
            or ("readtimeout" in joined_low and ("assign" in joined_low or "creating session" in joined_low))
            or ("timeout" in joined_low and "creating session" in joined_low)
        )
        if is_precondition:
            if pre_assignments > 0 and not recreate:
                return {
                    "kind": "limit_or_stockout",
                    "stockout_gpu": None,
                    "reason": "已有运行中实例或目标卡高峰期名额紧张",
                    "message": f"⚠️ 分配 {gpu_up} 被拒：并发实例上限或算力紧张，建议先停卡或换卡",
                }
            return {
                "kind": "stockout",
                "stockout_gpu": gpu_up,
                "reason": f"高峰期 {gpu_up} 算力容量不足",
                "message": f"⚠️ 高峰期 {gpu_up} 算力紧缺无法分配",
            }

        if is_capacity:
            return {
                "kind": "stockout",
                "stockout_gpu": gpu_up,
                "reason": f"高峰期 {gpu_up} 分配拥堵/无空闲算力",
                "message": f"⚠️ 高峰期 {gpu_up} 无空闲卡或响应超时",
            }

        for ln in reversed(cleaned):
            if "磁盘空间不足" in ln or "显存不足" in ln:
                return {
                    "kind": "error",
                    "stockout_gpu": None,
                    "reason": ln,
                    "message": f"⚠️ {ln[:60]}",
                }
            if "llama-server" in ln and ("超时" in ln or "失败" in ln or "异常退出" in ln):
                return {
                    "kind": "error",
                    "stockout_gpu": None,
                    "reason": "llama-server 启动异常",
                    "message": f"⚠️ {ln[:58]}",
                }
            if "等待 ComfyUI 就绪超时" in ln:
                return {
                    "kind": "error",
                    "stockout_gpu": None,
                    "reason": "ComfyUI 启动超时",
                    "message": "⚠️ 等待 ComfyUI 启动超时，可尝试重新增量部署",
                }
            if "WebSocket" in ln or "握手超时" in ln:
                return {
                    "kind": "error",
                    "stockout_gpu": None,
                    "reason": "Colab 内核通信超时",
                    "message": "⚠️ 远端内核握手超时，可重试或停卡重开",
                }

        tail = "部署中断"
        for ln in reversed(cleaned):
            low = ln.lower()
            if ln.startswith('File "') or ln.startswith("Traceback ") or low == "raise" or low.startswith("raise "):
                continue
            if set(ln) <= {"~", "^", " "} or "output=stdout, stderr=stderr)" in low:
                continue
            if "calledprocesserror" in low and len(cleaned) > 1:
                continue
            tail = ln
            break
        if len(tail) > 52:
            tail = tail[:49] + "..."
        return {
            "kind": "error",
            "stockout_gpu": None,
            "reason": tail,
            "message": f"⚠️ {profile} 部署异常: {tail}",
        }

    def _consume_deploy_line(self, raw_line: str) -> str | None:
        """解析一行开机/部署输出并增量推进 _deploy_state，返回捕获到的隧道 URL（如有）。"""
        line = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", str(raw_line or "")).strip()
        if not line:
            return None

        tunnel_url: str | None = None
        should_warm_meta = False
        with self._lock:
            st = self._deploy_state
            cur_pct = int(st.get("pct") or 0)
            cur_step = int(st.get("step_idx") or 0)
            total_steps = max(1, int(st.get("step_total") or 7))
            gpu = str(st.get("gpu") or "L4")
            is_llm = str(st.get("profile") or "") == "qwen_llm"

            m_tot = _DEPLOY_DL_TOTAL_RE.search(line)
            if m_tot:
                tot_files = int(m_tot.group(1))
                if tot_files > 0:
                    self._deploy_dl_total = tot_files
                return None

            m_tun = _DEPLOY_TUNNEL_LINE_RE.search(line)
            if m_tun:
                tunnel_url = m_tun.group(1).strip()
                st["step_idx"] = total_steps
                st["pct"] = 100
                st["stage_title"] = f"[{total_steps}/{total_steps}] 部署全部完成"
                st["sub_detail"] = f"隧道已打通: {tunnel_url.replace('https://', '')}"
                if not self._deploy_meta_warmed:
                    self._deploy_meta_warmed = True
                    should_warm_meta = True

        if tunnel_url is not None:
            if should_warm_meta:
                threading.Thread(
                    target=lambda: self._fast_read_tunnel_url("", "comfy"),
                    daemon=True,
                ).start()
            return tunnel_url

        with self._lock:
            st = self._deploy_state
            cur_pct = int(st.get("pct") or 0)
            cur_step = int(st.get("step_idx") or 0)
            gpu = str(st.get("gpu") or "L4")
            is_llm = str(st.get("profile") or "") == "qwen_llm"

            m_step = _STEP_BANNER_RE.search(line)
            if m_step:
                idx = int(m_step.group(1))
                tot = max(1, int(m_step.group(2)))
                raw_title = m_step.group(3).strip()
                if len(raw_title) > 18:
                    raw_title = raw_title[:17] + "…"
                base_pct = _DEPLOY_STEP_BASE_PCT.get(
                    idx, min(95, int(round(12 + idx * 84.0 / tot)))
                )
                st["step_idx"] = idx
                st["step_total"] = tot
                st["stage_title"] = f"[{idx}/{tot}] {raw_title}"
                st["pct"] = max(cur_pct, base_pct)
                if is_llm:
                    if idx == 3:
                        st["sub_detail"] = "正在准备 llama-server CUDA 推理引擎..."
                    elif idx == 4:
                        st["sub_detail"] = "正在高速拉取 Qwen3.8 GGUF 量化分片..."
                    elif idx == 5:
                        st["sub_detail"] = "正在生成访问密钥与 PLE/MoE 卸载参数..."
                    elif idx == 6:
                        st["sub_detail"] = "正在加载 180B 模型并启动 llama-server..."
                    elif idx == 7:
                        st["sub_detail"] = "正在打通 Cloudflared 隧道并写入服务契约..."
                    else:
                        st["sub_detail"] = raw_title
                else:
                    if idx == 4:
                        st["sub_detail"] = "正在连接模型源并校验本地缓存..."
                    elif idx == 5:
                        st["sub_detail"] = "正在拉取 LoRA 与工作流配置..."
                    elif idx == 6:
                        st["sub_detail"] = "正在拉起 ComfyUI 服务..."
                    elif idx == 7:
                        st["sub_detail"] = "正在申请 Cloudflared 公网隧道..."
                    else:
                        st["sub_detail"] = raw_title
                return None

            low = line.lower()
            if "正在释放旧实例" in line:
                st["pct"] = max(cur_pct, 3)
                st["stage_title"] = "清理旧会话"
                st["sub_detail"] = "正在停卡释放旧算力分配..."
            elif "creating session" in low or "开启 colab 实例" in line:
                st["pct"] = max(cur_pct, 6)
                st["stage_title"] = f"申请 {gpu} 实例"
                st["sub_detail"] = f"正在向 Colab 请求分配 {gpu} GPU..."
            elif "session ready" in low:
                st["pct"] = max(cur_pct, 11)
                st["stage_title"] = "实例就绪"
                st["sub_detail"] = f"{gpu} 已分配，准备上传云盘凭据..."
            elif "跳过 colab new" in line:
                st["pct"] = max(cur_pct, 11)
                st["stage_title"] = "复用活跃实例"
                st["sub_detail"] = "沿用现存运行时，上传配置并增量部署..."
            elif "colab.exe upload" in low or ("upload" in low and "rclone.conf" in low):
                st["pct"] = max(cur_pct, 13)
                st["stage_title"] = "注入云盘配置"
                st["sub_detail"] = "正在握手内核并启动 7 步自举..."
            elif cur_step == 1:
                if "自动降级" in line or "降级为" in line:
                    st["sub_detail"] = line[:38]
                    st["pct"] = max(cur_pct, 20)
                elif "gpu 硬件信息:" in line or line.startswith("GPU:"):
                    info = line.split(":", 1)[1].strip()
                    st["sub_detail"] = f"已就绪: {info[:36]}"
                    st["pct"] = max(cur_pct, 19)
            elif cur_step == 2 and ("已就绪" in line or "安装完成" in line or "工具就绪" in line):
                st["pct"] = max(cur_pct, 27)
                st["sub_detail"] = "传输与隧道基础工具已备齐"
            elif cur_step == 3:
                if is_llm:
                    if "cmake" in low or "编译" in line:
                        st["sub_detail"] = "正在编译 llama-server CUDA 内核..."
                        st["pct"] = max(cur_pct, 36)
                    elif "cuda0" in low or ("llama-server" in low and ("就绪" in line or "完成" in line)):
                        st["sub_detail"] = "llama-server CUDA 推理引擎已就绪"
                        st["pct"] = max(cur_pct, 40)
                    elif "预编译" in line or "下载" in line or "拉取" in line:
                        st["sub_detail"] = "正在拉取 llama-server CUDA 二进制包..."
                        st["pct"] = max(cur_pct, 34)
                else:
                    if "git" in low and "clone" in low:
                        st["sub_detail"] = "正在克隆官方 ComfyUI 仓库..."
                        st["pct"] = max(cur_pct, 33)
                    elif "pip" in low and "install" in low:
                        st["sub_detail"] = "正在安装依赖（保留预装 CUDA）..."
                        st["pct"] = max(cur_pct, 36)
                    elif "依赖安装完成" in line or "comfy_kitchen 已就绪" in line:
                        st["sub_detail"] = "ComfyUI 依赖与 CUDA 环境就绪"
                        st["pct"] = max(cur_pct, 40)
            elif cur_step == 4:
                m_start = re.search(r"(?:开始下载底座文件:|\[开始下载\])\s*([^\s(]+)", line)
                if m_start:
                    fname = PurePosixPath(m_start.group(1)).name
                    self._deploy_dl_name = fname[:20] + ("…" if len(fname) > 20 else "")
                    st["sub_detail"] = f"下载 {self._deploy_dl_name}..."
                    return None

                if (
                    "已存在且字节数匹配" in line
                    or "校验通过:" in line
                    or "[已就绪]" in line
                    or "[完成]" in line
                ):
                    self._deploy_dl_done = min(
                        max(1, self._deploy_dl_total),
                        self._deploy_dl_done + 1,
                    )
                    ratio = self._deploy_dl_done / max(1, self._deploy_dl_total)
                    st["pct"] = max(cur_pct, min(76, 42 + int(round(ratio * 34))))
                    short_msg = line.replace("已存在且字节数匹配，跳过:", "已缓存")
                    st["sub_detail"] = short_msg[:38]
                    return None

                m_pct = _PCT_TOKEN_RE.search(line)
                if m_pct:
                    file_pct = min(100.0, max(0.0, float(m_pct.group(1))))
                    ratio = min(
                        1.0,
                        (self._deploy_dl_done + file_pct / 100.0) / max(1, self._deploy_dl_total),
                    )
                    st["pct"] = max(cur_pct, min(76, 42 + int(round(ratio * 34))))
                    m_spd = _DL_SPEED_RE.search(line)
                    spd = f" · {m_spd.group(1)}" if m_spd else ""
                    prefix = f"{self._deploy_dl_name} " if self._deploy_dl_name else "拉取模型 "
                    st["sub_detail"] = f"{prefix}({int(round(file_pct))}%{spd})"
            elif cur_step == 5:
                if is_llm and ("sk-pet-" in line or "鉴权" in line or "参数" in line or "卸载" in line):
                    st["sub_detail"] = "安全鉴权密钥与推理参数已就绪"
                    st["pct"] = max(cur_pct, 83)
                elif "LoRA 就绪:" in line or "工作流就绪:" in line:
                    st["sub_detail"] = line[:38]
                    st["pct"] = max(cur_pct, 83)
            elif cur_step == 6:
                if is_llm:
                    if "已就绪" in line or "模型就绪" in line or "复用" in line:
                        st["sub_detail"] = "Qwen3.8 推理引擎加载完成"
                        st["pct"] = max(cur_pct, 93)
                    elif "llama-server" in low or "18080" in line or "8080" in line or "health" in low or "加载" in line:
                        st["sub_detail"] = "等待 llama-server 18080 端口加载就绪..."
                        st["pct"] = max(cur_pct, 89)
                else:
                    if "已后台拉起 ComfyUI" in line or "已拉起 ComfyUI" in line:
                        st["sub_detail"] = "等待 ComfyUI 8188 端口响应..."
                        st["pct"] = max(cur_pct, 89)
                    elif "ComfyUI 就绪" in line or "已就绪" in line:
                        st["sub_detail"] = "ComfyUI 引擎启动完成"
                        st["pct"] = max(cur_pct, 93)
            elif cur_step == 7 and "cloudflared" in low and ("拉起" in line or "启动" in line or "隧道" in line):
                st["sub_detail"] = "等待 Cloudflare 分配公网域名..."
                st["pct"] = max(cur_pct, 97)
        return tunnel_url

    def start_instance(self, config: dict[str, Any] | None = None) -> dict[str, Any]:
        """按用户所选生态与显卡一键无窗口部署，并通过事件总线直播 7 步进度。"""
        self.touch_activity()
        cfg = config if isinstance(config, dict) else {}
        profile = str(cfg.get("profile") or "anima").strip().lower()
        if profile not in SUPPORTED_BOOT_PROFILES:
            profile = "anima"

        default_gpu = "G4" if profile in ("minimax", "drama3", "qwen_llm") else "L4"
        gpu = str(cfg.get("gpu") or default_gpu).strip().upper()
        if gpu not in SUPPORTED_BOOT_GPUS:
            gpu = default_gpu

        # L4 与 G4 标准形态已自带充足内存且 Colab 无独立 HIGH_RAM 选项，强制关闭 high_mem 防分配被拒
        high_mem = False if gpu in ("L4", "G4") else bool(cfg.get("high_mem", False))
        minimax_r2v = bool(cfg.get("minimax_r2v", True))
        minimax_pe = bool(cfg.get("minimax_pe", True))
        recreate = bool(cfg.get("recreate", False))

        if not CTL_SCRIPT.exists():
            return {"ok": False, "message": f"未找到控制脚本: {CTL_SCRIPT.name}"}

        if not self._deploy_lock.acquire(blocking=False):
            return {"ok": False, "message": "已有开机部署任务在运行中"}

        # 保存本次选卡与生态配置，并记录初始活跃会话数辅助精准区分缺卡与并发上限
        with self._lock:
            prefs = self.state.setdefault("prefs", dict(DEFAULT_PREFS))
            raw_quant = str(cfg.get("llm_quant") or prefs.get("boot_llm_quant") or "UD-Q4_K_XL").strip()
            llm_quant = raw_quant if raw_quant in SUPPORTED_LLM_QUANTS else "UD-Q4_K_XL"
            try:
                llm_ctx = int(cfg.get("llm_ctx") or prefs.get("boot_llm_ctx") or 32768)
            except (TypeError, ValueError):
                llm_ctx = 32768
            llm_ctx = max(4096, min(262144, llm_ctx))
            llm_mmproj = bool(cfg.get("llm_mmproj", prefs.get("boot_llm_mmproj", True)))

            prefs["boot_profile"] = profile
            prefs["boot_gpu"] = gpu
            prefs["boot_high_mem"] = high_mem
            prefs["boot_minimax_r2v"] = minimax_r2v
            prefs["boot_minimax_pe"] = minimax_pe
            prefs["boot_llm_quant"] = llm_quant
            prefs["boot_llm_ctx"] = llm_ctx
            prefs["boot_llm_mmproj"] = llm_mmproj
            self._save_state()
            self.last_snapshot["prefs"] = dict(prefs)

            pre_assignments = int(self.state.get("last_assignments_count") or 0)
            now_ts = int(time.time())
            self._deploy_cancelled = False
            self._deploy_meta_warmed = False
            self._deploy_dl_name = ""
            self._deploy_dl_done = 0
            if profile == "qwen_llm":
                self._deploy_dl_total = LLM_QUANT_BASE_SHARDS[llm_quant] + (
                    1 if llm_mmproj else 0
                )
            elif profile == "minimax":
                self._deploy_dl_total = (
                    7 + (1 if minimax_pe else 0) + (2 if minimax_r2v else 0)
                )
            else:
                self._deploy_dl_total = 3

            self._deploy_state = {
                "active": True,
                "profile": profile,
                "gpu": gpu,
                "high_mem": high_mem,
                "step_idx": 0,
                "step_total": 7,
                "stage_title": f"准备开机 ({gpu})",
                "sub_detail": "正在启动本地无窗口部署进程...",
                "pct": 2,
                "started_ts": now_ts,
                "elapsed_sec": 0,
                "error_msg": "",
                "stockout_gpu": None,
                "suggest_gpu": None,
            }
            initial_deploy = self.get_deploy_state()

        if profile == "qwen_llm":
            self._stop_ws()
        with self._comfy_lock:
            self._comfy_state["service_kind"] = "llm" if profile == "qwen_llm" else "comfy"
            cstate_init = dict(self._comfy_state)
        with self._lock:
            self.last_snapshot["comfy"] = cstate_init

        cmd = [
            _hidden_python_exe(),
            "-u",
            str(CTL_SCRIPT),
            "up",
            "-s",
            "comfy",
            "--profile",
            profile,
            "--gpu",
            gpu,
        ]
        if high_mem:
            cmd.append("--high-mem")
        if recreate:
            cmd.append("--recreate")
        if profile == "minimax":
            if not minimax_r2v and not minimax_pe:
                cmd.append("--minimax-lite")
            else:
                if not minimax_r2v:
                    cmd.append("--no-r2v")
                if not minimax_pe:
                    cmd.append("--no-pe")
        elif profile == "qwen_llm":
            cmd.extend(["--llm-quant", llm_quant, "--llm-ctx", str(llm_ctx)])
            if not llm_mmproj:
                cmd.append("--no-mmproj")

        profile_label = {
            "minimax": "MiniMax",
            "drama3": "Drama 3",
            "qwen_llm": "Qwen3.8",
        }.get(profile, "Anima")
        self._emit_event({"type": "deploy_progress", "deploy": initial_deploy})

        def _worker() -> None:
            hb_stop = threading.Event()
            tail_lines: list[str] = []
            captured_tunnel: str | None = None
            rc = 1

            def _heartbeat() -> None:
                while not hb_stop.wait(timeout=1.0):
                    if not self.is_deploying():
                        break
                    self._emit_event({"type": "deploy_progress", "deploy": self.get_deploy_state()})

            hb_thread = threading.Thread(target=_heartbeat, daemon=True)
            hb_thread.start()

            try:
                env = _clean_env()
                env["PYTHONUNBUFFERED"] = "1"
                # 加快高峰期 503/分配拒绝响应，避免用户干等数十秒
                env["COLAB_CLI_ASSIGN_RETRIES"] = "1"
                env["COLAB_CLI_RETRY_BACKOFF"] = "4"

                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    env=env,
                    creationflags=CREATE_NO_WINDOW,
                )
                self._deploy_proc = proc

                last_emit_m = 0.0
                if proc.stdout is not None:
                    buf = ""
                    try:
                        while True:
                            ch = proc.stdout.read(256)
                            if not ch:
                                break
                            buf += ch.replace("\r", "\n")
                            while "\n" in buf:
                                raw_line, buf = buf.split("\n", 1)
                                sline = raw_line.strip()
                                if not sline:
                                    continue
                                tail_lines = (tail_lines + [sline])[-40:]
                                prev_step = int(self._deploy_state.get("step_idx") or 0)
                                tun = self._consume_deploy_line(sline)
                                cur_step = int(self._deploy_state.get("step_idx") or 0)
                                if tun:
                                    captured_tunnel = tun
                                now_m = time.monotonic()
                                if tun or cur_step != prev_step or (now_m - last_emit_m >= 0.12):
                                    last_emit_m = now_m
                                    self._emit_event(
                                        {"type": "deploy_progress", "deploy": self.get_deploy_state()}
                                    )
                        if buf.strip():
                            sline = buf.strip()
                            tail_lines = (tail_lines + [sline])[-40:]
                            tun = self._consume_deploy_line(sline)
                            if tun:
                                captured_tunnel = tun
                    finally:
                        proc.stdout.close()

                rc = proc.wait()
            except Exception as exc:
                tail_lines.append(str(exc))
                rc = 1
            finally:
                hb_stop.set()
                self._deploy_proc = None

            cancelled = self._deploy_cancelled
            elapsed_total = max(0, int(time.time()) - int(initial_deploy.get("started_ts") or time.time()))
            elapsed_cn = self._format_elapsed_cn(elapsed_total)

            if cancelled:
                with self._lock:
                    self._deploy_state.update(
                        {
                            "active": False,
                            "pct": 0,
                            "stage_title": "已中止部署",
                            "sub_detail": "用户主动取消开机流程",
                            "error_msg": "已中止开机部署",
                            "elapsed_sec": elapsed_total,
                        }
                    )
                snap = self.fetch_live_status()
                self._emit_event(
                    {
                        "type": "deploy_error",
                        "cancelled": True,
                        "deploy": self.get_deploy_state(),
                        "snapshot": snap,
                        "summary": "⏹️ 已中止开机部署；若实例已分配可点一键停卡释放",
                    }
                )
            elif rc == 0:
                done_detail = (
                    "Qwen3.8 引擎与隧道已就绪"
                    if profile == "qwen_llm"
                    else "ComfyUI 与隧道已就绪"
                )
                with self._lock:
                    self._scarce_gpus.pop(gpu, None)
                    self._deploy_state.update(
                        {
                            "active": False,
                            "step_idx": int(self._deploy_state.get("step_total") or 7),
                            "pct": 100,
                            "stage_title": "部署完成",
                            "sub_detail": done_detail,
                            "error_msg": "",
                            "stockout_gpu": None,
                            "suggest_gpu": None,
                            "elapsed_sec": elapsed_total,
                        }
                    )
                snap = self.fetch_live_status()
                target_ep: str | None = None
                with self._lock:
                    sessions = list(self.state.get("last_sessions") or [])
                    target_ep = next(
                        (str(s.get("endpoint")) for s in sessions if s.get("name") == "comfy" and s.get("endpoint")),
                        None,
                    )
                    if not target_ep and sessions:
                        target_ep = str(sessions[0].get("endpoint") or "")
                    if captured_tunnel and target_ep:
                        self.state.setdefault("tunnel_cache", {})[target_ep] = captured_tunnel
                        self._save_state()

                # 确保最新服务契约与内存态密钥已就位，使首次打开对话页或复制配置零延迟
                meta_ready, key_ready = self._get_ep_meta_and_key(target_ep or "")
                if profile == "qwen_llm":
                    if not meta_ready or not key_ready:
                        _, r_ep, fast_u = self._fast_read_tunnel_url(target_ep or "", "comfy")
                        if r_ep and not target_ep:
                            target_ep = r_ep
                        if fast_u and not captured_tunnel:
                            captured_tunnel = fast_u
                        meta_ready, _ = self._get_ep_meta_and_key(target_ep or "")
                elif target_ep:
                    with self._lock:
                        self._service_meta_by_ep[target_ep] = {"kind": "comfy", "port": 8188}
                        self._runtime_secrets.pop(target_ep, None)

                with self._comfy_lock:
                    self._comfy_state["service_kind"] = "llm" if profile == "qwen_llm" else "comfy"
                    if captured_tunnel:
                        self._comfy_state["tunnel_url"] = captured_tunnel
                    if profile == "qwen_llm":
                        self._comfy_state["ws_connected"] = False
                        self._comfy_state["model_name"] = str(
                            meta_ready.get("model") or "Qwen3.8-Flash-Next-abliterated"
                        )
                        self._comfy_state["quant"] = str(meta_ready.get("quant") or llm_quant)
                        self._comfy_state["ctx_size"] = int(meta_ready.get("ctx_size") or llm_ctx)
                        if meta_ready.get("gpu_name"):
                            self._comfy_state["gpu_name"] = self._short_gpu_name(
                                str(meta_ready["gpu_name"])
                            )
                        v_u = round(float(meta_ready.get("vram_used_gb") or 0.0), 1)
                        v_t = round(float(meta_ready.get("vram_total_gb") or 0.0), 1)
                        if v_t > 0:
                            self._comfy_state["vram_used_gb"] = v_u
                            self._comfy_state["vram_total_gb"] = v_t
                            self._comfy_state["vram_pct"] = round(v_u * 100.0 / v_t, 1)
                    cstate = dict(self._comfy_state)
                with self._lock:
                    self.last_snapshot["comfy"] = cstate

                self._wake_monitor.set()
                done_action = "open_llm_webui" if profile == "qwen_llm" else "open_comfy"
                done_summary = (
                    f"🎉 {gpu}·{profile_label} 部署完成({elapsed_cn})！点我打开对话页"
                    if profile == "qwen_llm"
                    else f"🎉 {gpu}·{profile_label} 部署完成({elapsed_cn})！点我打开"
                )
                self._emit_event(
                    {
                        "type": "deploy_done",
                        "deploy": self.get_deploy_state(),
                        "snapshot": self.get_cached_snapshot(),
                        "tunnel_url": captured_tunnel,
                        "summary": done_summary,
                        "action": done_action,
                    }
                )
            else:
                diag = self._classify_deploy_failure(tail_lines, gpu, pre_assignments, profile, recreate)
                stockout_gpu = diag.get("stockout_gpu")
                if stockout_gpu:
                    with self._lock:
                        self._scarce_gpus[str(stockout_gpu).upper()] = {
                            "ts": int(time.time()),
                            "reason": str(diag.get("reason") or "算力紧缺"),
                            "kind": str(diag.get("kind") or "stockout"),
                        }
                snap = self.fetch_live_status()
                suggest_gpu = None
                if stockout_gpu:
                    suggest_gpu = self._recommend_fallback_gpu(
                        str(stockout_gpu),
                        profile,
                        snap.get("eligible_gpus"),
                        snap.get("scarce_gpus") or {},
                    )
                summary = str(diag.get("message") or "⚠️ 开机部署失败")
                if stockout_gpu and suggest_gpu and suggest_gpu not in summary:
                    summary = f"{summary}，建议改选 {suggest_gpu}"

                with self._lock:
                    self._deploy_state.update(
                        {
                            "active": False,
                            "stage_title": "部署失败",
                            "sub_detail": str(diag.get("reason") or summary)[:42],
                            "error_msg": summary,
                            "stockout_gpu": stockout_gpu,
                            "suggest_gpu": suggest_gpu,
                            "elapsed_sec": elapsed_total,
                        }
                    )
                    snap["deploy"] = self.get_deploy_state()
                    self.last_snapshot = snap

                self._emit_event(
                    {
                        "type": "deploy_error",
                        "deploy": self.get_deploy_state(),
                        "snapshot": snap,
                        "summary": summary,
                        "stockout_gpu": stockout_gpu,
                        "suggest_gpu": suggest_gpu,
                    }
                )

            self._deploy_lock.release()

        threading.Thread(target=_worker, daemon=True).start()
        return {
            "ok": True,
            "started": True,
            "deploy": initial_deploy,
            "message": f"开始部署 {gpu} · {profile_label}...",
        }

    def cancel_deploy(self) -> dict[str, Any]:
        """主动中止正在运行的一键开机部署子进程。"""
        proc = self._deploy_proc
        if not self.is_deploying() and proc is None:
            return {"ok": False, "message": "当前没有进行中的部署任务"}
        self._deploy_cancelled = True
        self._kill_process_tree(proc)
        return {"ok": True, "message": "正在中止开机部署..."}

    # =========================================================================
    # 功能 3：ComfyUI 实时队列 / 显存监控 / 自动或手动拉图 / 异常中断提醒
    # =========================================================================

    def acknowledge_comfy_error(self) -> None:
        """用户点击报错气泡前往处理后，清除当前红标报警。"""
        with self._comfy_lock:
            self._comfy_state["has_error"] = False
            self._comfy_state["error_summary"] = ""
            cstate = dict(self._comfy_state)
        self._emit_event({"type": "comfy_tick", "comfy": cstate})

    def _reset_comfy_offline(self) -> None:
        self._stop_ws()
        inferred_kind = "comfy"
        if self._has_running_session():
            inferred_kind = self._get_ep_service_kind()
        with self._comfy_lock:
            was_online = bool(self._comfy_state.get("online"))
            prev_kind = str(self._comfy_state.get("service_kind") or "comfy")
            self._comfy_state = self._default_comfy_state()
            self._comfy_state["service_kind"] = inferred_kind
            self._comfy_state["pending_pull_count"] = len(self._pending_pull_items)
            cstate = dict(self._comfy_state)
        with self._lock:
            self.last_snapshot["comfy"] = cstate
        if was_online or prev_kind != inferred_kind:
            self._emit_event({"type": "comfy_tick", "comfy": cstate})

    def _download_comfy_outputs(
        self, items: list[tuple[str, dict[str, str]]]
    ) -> list[Path]:
        """经由 ComfyUI /view 接口下载正式产出文件，自动附加服务端修改时间戳并防止同名异图覆写。"""
        saved: list[Path] = []
        self.download_root.mkdir(parents=True, exist_ok=True)
        for base_url, meta in items:
            url = f"{base_url.rstrip('/')}/view"
            subfolder = str(meta.get("subfolder") or "")
            raw_name = str(meta["filename"])
            params = {
                "filename": raw_name,
                "subfolder": subfolder,
                "type": meta.get("type", "output"),
            }
            tmp_path = self.download_root / f".dl_{uuid.uuid4().hex[:12]}.part"
            try:
                with self._http.get(url, params=params, stream=True, timeout=(8, 60)) as resp:
                    resp.raise_for_status()
                    mtime_sec = _parse_http_mtime(resp.headers.get("Last-Modified"))
                    if mtime_sec is None:
                        mtime_sec = time.time()
                    h = hashlib.sha256()
                    total_bytes = 0
                    with open(tmp_path, "wb") as fp:
                        for chunk in resp.iter_content(chunk_size=65536):
                            if chunk:
                                fp.write(chunk)
                                h.update(chunk)
                                total_bytes += len(chunk)

                norm_sub = subfolder.replace("\\", "/").strip("/")
                if _PRESERVE_ORIGINAL_SUB_RE.search(norm_sub) and raw_name.lower().endswith(".safetensors"):
                    final_name = raw_name
                else:
                    final_name = _format_timestamped_filename(raw_name, mtime_sec)

                dest = _safe_output_dest(self.download_root, subfolder, final_name)
                dest.parent.mkdir(parents=True, exist_ok=True)
                final_dest, skip_write = _resolve_non_clobber_dest(dest, total_bytes, h.hexdigest())
                if skip_write:
                    tmp_path.unlink(missing_ok=True)
                else:
                    tmp_path.replace(final_dest)
                    try:
                        os.utime(final_dest, (mtime_sec, mtime_sec))
                    except OSError:
                        pass
                saved.append(final_dest)
            except Exception:
                try:
                    tmp_path.unlink(missing_ok=True)
                except Exception:
                    pass
        return saved

    @staticmethod
    def _short_gpu_name(raw_name: str) -> str:
        name = re.sub(r"^cuda:\d+\s*", "", str(raw_name or "")).split(":")[0].strip()
        return name.replace("NVIDIA ", "").strip()

    @staticmethod
    def _collect_formal_outputs(entry: dict[str, Any]) -> list[dict[str, str]]:
        items: list[dict[str, str]] = []
        outputs = entry.get("outputs") or {}
        if not isinstance(outputs, dict):
            return items
        for node_out in outputs.values():
            if not isinstance(node_out, dict):
                continue
            for key in ("images", "gifs", "videos", "audio"):
                arr = node_out.get(key)
                if not isinstance(arr, list):
                    continue
                for item in arr:
                    if not isinstance(item, dict) or not item.get("filename"):
                        continue
                    if str(item.get("type") or "output") != "output":
                        continue
                    items.append(
                        {
                            "filename": str(item["filename"]),
                            "subfolder": str(item.get("subfolder") or ""),
                            "type": "output",
                        }
                    )
        return items

    @staticmethod
    def _extract_history_error(entry: dict[str, Any]) -> str | None:
        status = entry.get("status") or {}
        if str(status.get("status_str") or "").lower() != "error":
            return None
        node_t, exc_t, exc_m = None, None, None
        for msg in status.get("messages") or []:
            if isinstance(msg, (list, tuple)) and len(msg) >= 2 and msg[0] == "execution_error":
                data = msg[1] if isinstance(msg[1], dict) else {}
                node_t = data.get("node_type") or data.get("node_id")
                exc_t = data.get("exception_type")
                exc_m = data.get("exception_message")
                break
        return _format_comfy_error(
            str(node_t) if node_t else None,
            str(exc_t) if exc_t else None,
            str(exc_m) if exc_m else None,
        )

    def _clear_stale_deploy_error_locked(self) -> dict[str, Any] | None:
        """在持 self._lock 下清理服务恢复在线后残留的历史部署报错状态。"""
        if not self._deploy_state.get("active") and self._deploy_state.get("error_msg"):
            self._deploy_state["error_msg"] = ""
            if self._deploy_state.get("stage_title") == "部署失败":
                self._deploy_state["stage_title"] = ""
                self._deploy_state["sub_detail"] = ""
            snap_deploy = dict(self._deploy_state)
            self.last_snapshot["deploy"] = snap_deploy
            return snap_deploy
        return None

    def poll_comfy_http_once(self, tunnel_url: str) -> dict[str, Any]:
        """执行单轮 ComfyUI /system_stats、/queue 与 /history 轮询并触发提醒/拉回。"""
        base = tunnel_url.strip().rstrip("/")
        events: list[dict[str, Any]] = []

        # 1) /system_stats —— GPU 与显存水位
        try:
            r_stats = self._http.get(f"{base}/system_stats", timeout=4.0)
            r_stats.raise_for_status()
            stats = r_stats.json()
        except Exception:
            with self._comfy_lock:
                self._comfy_http_failures += 1
                if self._comfy_http_failures >= 2:
                    self._comfy_state["online"] = False
                cstate = dict(self._comfy_state)
            self._emit_event({"type": "comfy_tick", "comfy": cstate})
            return cstate

        devices = stats.get("devices") or []
        gpu_name = ""
        v_used = 0.0
        v_total = 0.0
        v_pct = 0.0
        if devices and isinstance(devices[0], dict):
            d0 = devices[0]
            gpu_name = self._short_gpu_name(d0.get("name", ""))
            tot_b = float(d0.get("vram_total") or 0.0)
            free_b = float(d0.get("vram_free") or 0.0)
            used_b = max(0.0, tot_b - free_b)
            if tot_b > 0:
                v_total = round(tot_b / (1024 ** 3), 1)
                v_used = round(used_b / (1024 ** 3), 1)
                v_pct = round(used_b * 100.0 / tot_b, 1)

        # 2) /queue —— 运行与排队数量
        running_cnt = 0
        pending_cnt = 0
        try:
            r_q = self._http.get(f"{base}/queue", timeout=4.0)
            if r_q.status_code == 200:
                qdata = r_q.json()
                running_cnt = len(qdata.get("queue_running") or [])
                pending_cnt = len(qdata.get("queue_pending") or [])
        except Exception:
            pass

        with self._comfy_lock:
            self._comfy_http_failures = 0
            prev_running = int(self._comfy_state.get("running_count") or 0)
            self._comfy_state["online"] = True
            self._comfy_state["service_kind"] = "comfy"
            self._comfy_state["ws_connected"] = bool(
                self._ws_app is not None and self._ws_thread and self._ws_thread.is_alive()
            )
            self._comfy_state["tunnel_url"] = base
            self._comfy_state["gpu_name"] = gpu_name
            self._comfy_state["vram_used_gb"] = v_used
            self._comfy_state["vram_total_gb"] = v_total
            self._comfy_state["vram_pct"] = v_pct
            self._comfy_state["running_count"] = running_cnt
            self._comfy_state["pending_count"] = pending_cnt
            self._comfy_state["model_name"] = ""
            self._comfy_state["quant"] = ""
            self._comfy_state["ctx_size"] = 0
            self._comfy_state["tps"] = None
            if running_cnt > 0 and prev_running == 0:
                self._comfy_state["has_error"] = False
                self._comfy_state["error_summary"] = ""
            if running_cnt == 0:
                self._comfy_state["step_value"] = 0
                self._comfy_state["step_max"] = 0
                self._comfy_state["step_pct"] = 0.0
                self._comfy_state["current_node"] = ""

        if running_cnt > 0 or pending_cnt > 0:
            self.touch_activity()

        # 3) /history?max_items=6 —— 捕获新出图或报错，首次仅建基线
        try:
            r_hist = self._http.get(f"{base}/history", params={"max_items": 6}, timeout=5.0)
            if r_hist.status_code == 200:
                hdata = r_hist.json()
                if isinstance(hdata, dict):
                    first_conn = base not in self._seen_prompts_by_url
                    seen = self._seen_prompts_by_url.setdefault(base, set())
                    if first_conn:
                        seen.update(str(pid) for pid in hdata.keys())
                    else:
                        new_pids = [str(pid) for pid in hdata.keys() if str(pid) not in seen]
                        if new_pids:
                            self.touch_activity()
                        new_outputs: list[tuple[str, dict[str, str]]] = []
                        latest_err: tuple[str, str] | None = None
                        for pid in new_pids:
                            seen.add(pid)
                            entry = hdata.get(pid)
                            if not isinstance(entry, dict):
                                continue
                            err_str = self._extract_history_error(entry)
                            if err_str and pid not in self._reported_error_prompts:
                                self._reported_error_prompts.add(pid)
                                latest_err = (pid, err_str)
                            for meta in self._collect_formal_outputs(entry):
                                new_outputs.append((base, meta))

                        if latest_err:
                            with self._comfy_lock:
                                self._comfy_state["has_error"] = True
                                self._comfy_state["error_summary"] = latest_err[1]
                            events.append(
                                {
                                    "type": "comfy_error",
                                    "summary": latest_err[1],
                                    "prompt_id": latest_err[0],
                                }
                            )

                        if new_outputs:
                            with self._lock:
                                auto_pull = bool(
                                    (self.state.get("prefs") or {}).get("auto_pull_outputs", True)
                                )
                            if auto_pull:
                                saved = self._download_comfy_outputs(new_outputs)
                                count = len(saved) or len(new_outputs)
                                with self._comfy_lock:
                                    if saved:
                                        self._last_downloaded_files = saved
                                        self._comfy_state["last_output_name"] = saved[-1].name
                                    self._pending_pull_items.clear()
                                    self._comfy_state["pending_pull_count"] = 0
                                    self._comfy_state["last_output_count"] = count
                                events.append(
                                    {
                                        "type": "outputs_ready",
                                        "count": count,
                                        "pulled": True,
                                        "summary": f"✨ 新出图 {count} 份，已存本机！点我查看",
                                        "files": [str(p) for p in saved],
                                    }
                                )
                            else:
                                with self._comfy_lock:
                                    self._pending_pull_items.extend(new_outputs)
                                    self._comfy_state["pending_pull_count"] = len(
                                        self._pending_pull_items
                                    )
                                    self._comfy_state["last_output_count"] = len(new_outputs)
                                    self._comfy_state["last_output_name"] = new_outputs[-1][1][
                                        "filename"
                                    ]
                                events.append(
                                    {
                                        "type": "outputs_ready",
                                        "count": len(new_outputs),
                                        "pulled": False,
                                        "summary": f"✨ 新出图 {len(new_outputs)} 份！点我拉回查看",
                                    }
                                )
        except Exception:
            pass

        with self._comfy_lock:
            self._comfy_state["pending_pull_count"] = len(self._pending_pull_items)
            cstate = dict(self._comfy_state)
        with self._lock:
            self.last_snapshot["comfy"] = cstate
            cleared_deploy = self._clear_stale_deploy_error_locked()
        if cleared_deploy is not None:
            self._emit_event({"type": "deploy_progress", "deploy": cleared_deploy})
        self._emit_event({"type": "comfy_tick", "comfy": cstate})
        for ev in events:
            self._emit_event(ev)
        return cstate

    @staticmethod
    def _extract_slot_tps(slot: dict[str, Any]) -> float | None:
        """从 llama-server /slots 单槽位字典中尝试提取生成速率 (tokens/s)。"""
        for container_key in ("timings", "performance"):
            sub = slot.get(container_key)
            if isinstance(sub, dict):
                for val_key in ("predicted_per_second", "tokens_per_second", "tps"):
                    raw = sub.get(val_key)
                    if isinstance(raw, (int, float)) and float(raw) > 0.05:
                        return round(float(raw), 1)
        nt = slot.get("next_token")
        if isinstance(nt, list) and nt and isinstance(nt[0], dict):
            nt = nt[0]
        if isinstance(nt, dict):
            for val_key in ("predicted_per_second", "tokens_per_second", "tps"):
                raw = nt.get(val_key)
                if isinstance(raw, (int, float)) and float(raw) > 0.05:
                    return round(float(raw), 1)
        return None

    def poll_llm_http_once(self, tunnel_url: str, ep: str = "") -> dict[str, Any]:
        """执行单轮 llama-server /health 与 /slots 健康度及生成状态轮询，更新 _comfy_state。"""
        base = tunnel_url.strip().rstrip("/")
        if not ep:
            target = self._pick_target_session("comfy")
            if target and target.get("endpoint"):
                ep = str(target["endpoint"])

        meta, api_key = self._get_ep_meta_and_key(ep)
        if not meta:
            with self._lock:
                for m_ep, m_val in self._service_meta_by_ep.items():
                    if str(m_val.get("tunnel_url") or "").rstrip("/") == base:
                        ep = m_ep
                        meta = dict(m_val)
                        api_key = str(self._runtime_secrets.get(m_ep) or "")
                        break

        headers: dict[str, str] = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        # 1) 探测 /health 确认模型已加载完毕且可正常服务
        try:
            r_health = self._http.get(f"{base}/health", headers=headers, timeout=4.0)
            if r_health.status_code != 200:
                raise RuntimeError(f"HTTP {r_health.status_code}")
            h_body = r_health.json()
            if not isinstance(h_body, dict) or str(h_body.get("status") or "").lower() != "ok":
                raise RuntimeError("unhealthy")
        except Exception:
            with self._comfy_lock:
                self._comfy_http_failures += 1
                self._comfy_state["service_kind"] = "llm"
                self._comfy_state["ws_connected"] = False
                if self._comfy_http_failures >= 2:
                    self._comfy_state["online"] = False
                cstate = dict(self._comfy_state)
            with self._lock:
                self.last_snapshot["comfy"] = cstate
            self._emit_event({"type": "comfy_tick", "comfy": cstate})
            return cstate

        # 2) 请求 /slots 获取活跃推理并发数与速率（若鉴权受限则优雅降级为 0/None，不影响在线态）
        running_cnt = 0
        tps: float | None = None
        try:
            r_slots = self._http.get(f"{base}/slots", headers=headers, timeout=4.0)
            if r_slots.status_code == 200:
                sdata = r_slots.json()
                slots_list = sdata if isinstance(sdata, list) else (
                    sdata.get("slots") if isinstance(sdata, dict) else []
                )
                if isinstance(slots_list, list):
                    active_tps: float | None = None
                    fallback_tps: float | None = None
                    for s_item in slots_list:
                        if not isinstance(s_item, dict):
                            continue
                        is_proc = bool(s_item.get("is_processing")) or (
                            int(s_item.get("state") or 0) == 1
                        )
                        cand_tps = self._extract_slot_tps(s_item)
                        if is_proc:
                            running_cnt += 1
                            if cand_tps is not None:
                                active_tps = cand_tps
                        elif cand_tps is not None and fallback_tps is None:
                            fallback_tps = cand_tps
                    tps = active_tps if active_tps is not None else (
                        fallback_tps if running_cnt > 0 else None
                    )
        except Exception:
            pass

        if running_cnt > 0:
            self.touch_activity()

        with self._lock:
            prefs = self.state.get("prefs") or {}
            pref_quant = str(prefs.get("boot_llm_quant") or "UD-Q4_K_XL")
            pref_ctx = int(prefs.get("boot_llm_ctx") or 32768)

        gpu_name = self._short_gpu_name(str(meta.get("gpu_name") or ""))
        v_used = round(float(meta.get("vram_used_gb") or 0.0), 1)
        v_total = round(float(meta.get("vram_total_gb") or 0.0), 1)
        v_pct = round(v_used * 100.0 / v_total, 1) if v_total > 0 else 0.0
        model_name = str(meta.get("model") or "Qwen3.8-Flash-Next-abliterated")
        quant = str(meta.get("quant") or pref_quant)
        ctx_size = int(meta.get("ctx_size") or pref_ctx)

        with self._comfy_lock:
            self._comfy_http_failures = 0
            self._comfy_state.update(
                {
                    "online": True,
                    "service_kind": "llm",
                    "ws_connected": False,
                    "tunnel_url": base,
                    "running_count": running_cnt,
                    "pending_count": 0,
                    "step_value": 0,
                    "step_max": 0,
                    "step_pct": 0.0,
                    "current_node": "",
                    "gpu_name": gpu_name,
                    "vram_used_gb": v_used,
                    "vram_total_gb": v_total,
                    "vram_pct": v_pct,
                    "has_error": False,
                    "error_summary": "",
                    "pending_pull_count": len(self._pending_pull_items),
                    "model_name": model_name,
                    "quant": quant,
                    "ctx_size": ctx_size,
                    "tps": tps,
                }
            )
            cstate = dict(self._comfy_state)
        with self._lock:
            self.last_snapshot["comfy"] = cstate
            cleared_deploy = self._clear_stale_deploy_error_locked()
        if cleared_deploy is not None:
            self._emit_event({"type": "deploy_progress", "deploy": cleared_deploy})
        self._emit_event({"type": "comfy_tick", "comfy": cstate})
        return cstate

    def pull_or_open_latest_outputs(self) -> dict[str, Any]:
        """点击出图提醒时按需拉回待下载产出，或直接在资源管理器高亮选中最新成图。"""
        self.touch_activity()
        with self._comfy_lock:
            pending = list(self._pending_pull_items)
            recent = list(self._last_downloaded_files)

        pulled_now = False
        if pending:
            saved = self._download_comfy_outputs(pending)
            if saved:
                pulled_now = True
                with self._comfy_lock:
                    self._pending_pull_items.clear()
                    self._last_downloaded_files = saved
                    self._comfy_state["pending_pull_count"] = 0
                    self._comfy_state["last_output_name"] = saved[-1].name
                    cstate = dict(self._comfy_state)
                recent = saved
                self._emit_event({"type": "comfy_tick", "comfy": cstate})

        existing = [p for p in recent if p.exists()]
        if existing:
            latest = existing[-1].resolve()
            try:
                subprocess.Popen(
                    ["explorer.exe", f"/select,{latest}"],
                    creationflags=CREATE_NO_WINDOW,
                )
            except Exception:
                try:
                    os.startfile(str(latest.parent))
                except Exception:
                    pass
            msg = f"已拉回并定位: {latest.name}" if pulled_now else f"已打开位置: {latest.name}"
            return {"ok": True, "path": str(latest), "message": msg}

        if self.download_root.exists():
            try:
                os.startfile(str(self.download_root))
                return {"ok": True, "message": "已打开本地成图目录"}
            except Exception:
                pass
        return self.sync_outputs()

    def _on_ws_message(self, _ws: Any, message: Any) -> None:
        if isinstance(message, (bytes, bytearray)) or not isinstance(message, str):
            return
        try:
            payload = json.loads(message)
        except Exception:
            return
        mtype = payload.get("type")
        mdata = payload.get("data") if isinstance(payload.get("data"), dict) else {}

        if mtype in ("execution_start", "progress", "progress_state", "executing"):
            self.touch_activity()

        if mtype == "execution_start":
            with self._comfy_lock:
                self._comfy_state["has_error"] = False
                self._comfy_state["error_summary"] = ""
                self._comfy_state["step_value"] = 0
                self._comfy_state["step_max"] = 0
                self._comfy_state["step_pct"] = 0.0
                cstate = dict(self._comfy_state)
            self._emit_event({"type": "comfy_tick", "comfy": cstate})
        elif mtype == "progress":
            val = int(mdata.get("value") or 0)
            mx = int(mdata.get("max") or 0)
            pct = round(val * 100.0 / mx, 1) if mx > 0 else 0.0
            with self._comfy_lock:
                self._comfy_state["step_value"] = val
                self._comfy_state["step_max"] = mx
                self._comfy_state["step_pct"] = pct
                self._comfy_state["running_count"] = max(1, int(self._comfy_state.get("running_count") or 0))
                cstate = dict(self._comfy_state)
            now = time.monotonic()
            if now - self._ws_last_emit_ts >= 0.25 or pct >= 100.0:
                self._ws_last_emit_ts = now
                self._emit_event({"type": "comfy_tick", "comfy": cstate})
        elif mtype == "progress_state":
            nodes = mdata.get("nodes") if isinstance(mdata.get("nodes"), dict) else {}
            picked = None
            for ninfo in nodes.values():
                if isinstance(ninfo, dict) and ninfo.get("state") == "running" and (ninfo.get("max") or 0) > 0:
                    picked = ninfo
                    break
            if picked:
                val = int(picked.get("value") or 0)
                mx = int(picked.get("max") or 0)
                pct = round(val * 100.0 / mx, 1) if mx > 0 else 0.0
                with self._comfy_lock:
                    self._comfy_state["step_value"] = val
                    self._comfy_state["step_max"] = mx
                    self._comfy_state["step_pct"] = pct
                    self._comfy_state["running_count"] = max(1, int(self._comfy_state.get("running_count") or 0))
                    cstate = dict(self._comfy_state)
                now = time.monotonic()
                if now - self._ws_last_emit_ts >= 0.25 or pct >= 100.0:
                    self._ws_last_emit_ts = now
                    self._emit_event({"type": "comfy_tick", "comfy": cstate})
        elif mtype == "executing":
            node_id = mdata.get("node")
            if node_id is None and mdata.get("prompt_id"):
                self._wake_monitor.set()
            elif node_id is not None:
                with self._comfy_lock:
                    self._comfy_state["current_node"] = str(node_id)
        elif mtype == "execution_error":
            summary = _format_comfy_error(
                str(mdata.get("node_type") or mdata.get("node_id") or "节点"),
                str(mdata.get("exception_type") or ""),
                str(mdata.get("exception_message") or ""),
            )
            pid = str(mdata.get("prompt_id") or "")
            if pid:
                self._reported_error_prompts.add(pid)
            with self._comfy_lock:
                self._comfy_state["has_error"] = True
                self._comfy_state["error_summary"] = summary
                self._comfy_state["step_value"] = 0
                self._comfy_state["step_max"] = 0
                self._comfy_state["step_pct"] = 0.0
                cstate = dict(self._comfy_state)
            self._emit_event({"type": "comfy_tick", "comfy": cstate})
            self._emit_event({"type": "comfy_error", "summary": summary, "prompt_id": pid})

    def _stop_ws(self) -> None:
        self._ws_stop.set()
        app = self._ws_app
        self._ws_app = None
        self._ws_target_url = None
        if app is not None:
            try:
                app.close()
            except Exception:
                pass

    def _ensure_ws_target(self, tunnel_url: str) -> None:
        if websocket is None:
            return
        base = tunnel_url.strip().rstrip("/")
        if self._ws_target_url == base and self._ws_thread and self._ws_thread.is_alive():
            return
        self._stop_ws()
        self._ws_stop.clear()
        self._ws_target_url = base
        ws_base = re.sub(r"^http", "ws", base, count=1)
        ws_url = f"{ws_base}/ws?clientId=colab_pet_{uuid.uuid4().hex[:8]}"

        def _ws_runner() -> None:
            while not self._ws_stop.is_set() and not self._monitor_stop.is_set():
                try:
                    app = websocket.WebSocketApp(ws_url, on_message=self._on_ws_message)
                    self._ws_app = app
                    app.run_forever(ping_interval=20, ping_timeout=10)
                except Exception:
                    pass
                finally:
                    self._ws_app = None
                if self._ws_stop.wait(timeout=3.0):
                    break

        self._ws_thread = threading.Thread(target=_ws_runner, daemon=True)
        self._ws_thread.start()

    def _send_colab_keepalive(self) -> bool:
        """通过 Colab 官方通道执行一次无窗口三层保活心跳（TFE keep-alive / Proxy / 空闲内核）。"""
        if not COLAB_PY.exists():
            return False
        try:
            proc = _run_hidden([str(COLAB_PY), "-c", _COLAB_KEEPALIVE_SCRIPT], timeout=25)
        except Exception:
            return False
        if proc.returncode != 0:
            return False
        matched = re.search(
            r"<<<COLAB_KEEPALIVE>>>(.*?)<<<END_COLAB_KEEPALIVE>>>",
            proc.stdout or "",
            re.S,
        )
        if not matched:
            return False
        try:
            data = json.loads(matched.group(1).strip())
            return bool(data.get("ok")) and int(data.get("pinged") or 0) > 0
        except Exception:
            return False

    def _maybe_trigger_keepalive(self, now_m: float | None = None) -> None:
        """按 10 分钟节流周期向活跃 Colab 实例发送防断连保活心跳（连续纯空闲超 45m 自动停发，防通宵续命烧卡）。"""
        if self.is_deploying():
            return
        cur_m = time.monotonic() if now_m is None else float(now_m)
        if not self._has_running_session():
            with self._lock:
                self._last_keepalive_ts = 0.0
                self._last_activity_m = 0.0
            return

        with self._lock:
            keep_alive = bool((self.state.get("prefs") or {}).get("keep_alive", True))
            if not keep_alive:
                return
            if self._last_activity_m <= 0.0:
                self._last_activity_m = cur_m
            idle_sec = max(0.0, cur_m - self._last_activity_m)
            if idle_sec > KEEPALIVE_MAX_IDLE_SEC:
                return
            last_ts = self._last_keepalive_ts
            if last_ts > 0.0 and (cur_m - last_ts) < KEEPALIVE_INTERVAL_SEC:
                return

        if not self._keepalive_lock.acquire(blocking=False):
            return

        # 先记录本次出发时间戳避免监控线程高频率重复拉起子进程
        with self._lock:
            if self._last_keepalive_ts > 0.0 and (cur_m - self._last_keepalive_ts) < KEEPALIVE_INTERVAL_SEC:
                self._keepalive_lock.release()
                return
            self._last_keepalive_ts = cur_m

        def _worker() -> None:
            ok = False
            try:
                ok = self._send_colab_keepalive()
            except Exception:
                ok = False
            finally:
                if not ok and not self._monitor_stop.is_set():
                    with self._lock:
                        # 失败时回拨时间戳，使约 2 分钟后可自动重试一次
                        if self._last_keepalive_ts == cur_m:
                            self._last_keepalive_ts = max(
                                1e-6,
                                cur_m - (KEEPALIVE_INTERVAL_SEC - KEEPALIVE_RETRY_COOLDOWN_SEC),
                            )
                self._keepalive_lock.release()

        threading.Thread(target=_worker, daemon=True).start()

    def _maybe_trigger_idle_auto_stop(self, now_m: float | None = None) -> None:
        """若确认远端服务在线且连续全空闲达设定阈值（默认 60m），先归档云盘再自动停卡止损。"""
        if self.is_deploying() or self._upload_lock.locked() or self._action_lock.locked():
            return
        if not self._has_running_session():
            return

        cur_m = time.monotonic() if now_m is None else float(now_m)
        with self._lock:
            prefs = self.state.get("prefs") or {}
            idle_stop_mins = int(prefs.get("idle_auto_stop_minutes", 60) or 0)
            if idle_stop_mins <= 0:
                return
            if self._last_activity_m <= 0.0:
                self._last_activity_m = cur_m
                return
            last_act = self._last_activity_m
            cur_ep = str(self.state.get("active_endpoint") or "__active__")
            warned_ep = self._idle_warned_ep

        with self._comfy_lock:
            comfy_online = bool(self._comfy_state.get("online"))
            busy_cnt = int(self._comfy_state.get("running_count") or 0) + int(
                self._comfy_state.get("pending_count") or 0
            )
        if busy_cnt > 0:
            self.touch_activity()
            return
        if not comfy_online:
            return

        idle_sec = max(0.0, cur_m - last_act)
        limit_sec = float(idle_stop_mins * 60)
        warn_sec = max(60.0, limit_sec - 120.0)

        if idle_sec >= limit_sec:
            if not self._idle_stop_lock.acquire(blocking=False):
                return

            def _idle_stop_worker() -> None:
                try:
                    if (
                        self.is_deploying()
                        or self._upload_lock.locked()
                        or not self._has_running_session()
                    ):
                        return
                    with self._comfy_lock:
                        now_busy = int(self._comfy_state.get("running_count") or 0) + int(
                            self._comfy_state.get("pending_count") or 0
                        )
                        now_online = bool(self._comfy_state.get("online"))
                    if not now_online or now_busy > 0:
                        if now_busy > 0:
                            self.touch_activity()
                        return
                    if (time.monotonic() - self._last_activity_m) < limit_sec:
                        return

                    if not self.is_llm_mode_active() and CTL_SCRIPT.exists():
                        target = self._pick_target_session("comfy")
                        sname = (
                            target["name"]
                            if target and target.get("name") not in (None, "?")
                            else "comfy"
                        )
                        try:
                            _run_hidden(
                                [_hidden_python_exe(), str(CTL_SCRIPT), "sync", "-s", sname],
                                timeout=120,
                            )
                        except Exception:
                            pass

                    res = self.stop_all_instances()
                    if res.get("ok"):
                        self._emit_event(
                            {
                                "type": "idle_stopped",
                                "summary": f"🛑 连续闲置满 {idle_stop_mins} 分钟，已归档并自动停卡省积分",
                                "snapshot": res.get("snapshot") or self.get_cached_snapshot(),
                            }
                        )
                    else:
                        with self._lock:
                            self._idle_warned_ep = None
                        self._emit_event(
                            {
                                "type": "idle_warning",
                                "summary": f"⚠️ 闲置自动停卡未成功: {res.get('message', '请手动停卡')}",
                                "action": "keep_working",
                            }
                        )
                finally:
                    self._idle_stop_lock.release()

            threading.Thread(target=_idle_stop_worker, daemon=True).start()
        elif idle_sec >= warn_sec and warned_ep != cur_ep:
            with self._lock:
                self._idle_warned_ep = cur_ep
            self._emit_event(
                {
                    "type": "idle_warning",
                    "summary": f"⏳ 已闲置 {int(idle_sec // 60)} 分钟，约 2 分钟后将自动存档停卡（点我续期）",
                    "action": "keep_working",
                }
            )

    def start_monitor(self, on_event: Callable[[dict[str, Any]], None]) -> None:
        """启动后台 ComfyUI / LLM 服务状态与显存监控守护线程。"""
        self._event_cb = on_event
        if self._monitor_thread and self._monitor_thread.is_alive():
            return
        self._monitor_stop.clear()

        def _loop() -> None:
            while not self._monitor_stop.is_set():
                self._wake_monitor.clear()
                wait_s = 2.0
                if self.is_deploying():
                    wait_s = 2.0
                elif not self._has_running_session():
                    self._maybe_trigger_keepalive()
                    self._reset_comfy_offline()
                else:
                    self._maybe_trigger_keepalive()
                    turl = self.resolve_active_tunnel_url(allow_fast_discover=True)
                    if turl:
                        target_sess = self._pick_target_session("comfy")
                        cur_ep = str(target_sess.get("endpoint") or "") if target_sess else ""
                        skind = self._get_ep_service_kind(cur_ep)
                        if skind == "llm":
                            # LLM 模式下主动关闭 ComfyUI /ws 连接，防止向 llama-server 刷 404 重连风暴
                            self._stop_ws()
                            cstate = self.poll_llm_http_once(turl, ep=cur_ep)
                            busy = (cstate.get("running_count") or 0) > 0
                            wait_s = 2.5 if busy else 8.0
                        else:
                            self._ensure_ws_target(turl)
                            cstate = self.poll_comfy_http_once(turl)
                            busy = (cstate.get("running_count") or 0) + (cstate.get("pending_count") or 0) > 0
                            wait_s = 2.5 if busy else 6.0
                    else:
                        self._reset_comfy_offline()
                        wait_s = 5.0
                    self._maybe_trigger_idle_auto_stop()
                self._wake_monitor.wait(timeout=wait_s)

        self._monitor_thread = threading.Thread(target=_loop, daemon=True)
        self._monitor_thread.start()

    def stop_monitor(self) -> None:
        """停止后台监控线程与 WebSocket 连接（退出时同步清理未完成部署进程）。"""
        self._monitor_stop.set()
        self._wake_monitor.set()
        self._stop_ws()
        if self._deploy_proc is not None:
            self._deploy_cancelled = True
            self._kill_process_tree(self._deploy_proc)

    # =========================================================================
    # 功能 4：文件拖拽与智能分流投递（参考图 / 工作流 JSON / PNG 彩蛋 / LoRA）
    # =========================================================================

    def _emit_upload_progress(self, name: str, pct: int, stage: str) -> None:
        self._emit_event(
            {
                "type": "upload_progress",
                "name": name,
                "pct": max(0, min(100, int(pct))),
                "summary": f"📤 {stage} {name} ({max(0, min(100, int(pct)))}%)",
            }
        )

    def _rclone_upload_file(
        self,
        src: Path,
        remote_dst: str,
        display_name: str,
        pct_start: int = 0,
        pct_span: int = 100,
    ) -> None:
        if not RCLONE_EXE.is_file():
            raise FileNotFoundError(f"未找到 rclone.exe: {RCLONE_EXE}")
        if not RCLONE_CONF.is_file():
            raise FileNotFoundError(f"未找到 rclone.conf: {RCLONE_CONF}")

        cmd = [
            str(RCLONE_EXE),
            "copyto",
            "--config",
            str(RCLONE_CONF),
            "--disable-http2",
            "--stats",
            "1s",
            "--stats-one-line",
            str(src),
            remote_dst,
        ]
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=_clean_env(),
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
                    if sline:
                        tail_lines = (tail_lines + [sline])[-8:]
                        m = re.search(r"(\d{1,3})%", sline)
                        if m:
                            raw_pct = min(100, int(m.group(1)))
                            mapped = pct_start + int(round(raw_pct * pct_span / 100.0))
                            if mapped != last_pct:
                                last_pct = mapped
                                self._emit_upload_progress(display_name, mapped, "传云盘")
        finally:
            proc.stdout.close()

        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(tail_lines[-1] if tail_lines else f"rclone 退出码 {rc}")

    def _write_remote_file_via_contents(
        self, rel_path: str, raw_bytes: bytes, session: str = "comfy"
    ) -> None:
        if not COLAB_PY.exists():
            raise FileNotFoundError(f"未找到 colab-cli Python: {COLAB_PY}")
        target = self._pick_target_session(session)
        ep = str(target.get("endpoint") or "") if target else ""
        sname = target["name"] if target and target.get("name") not in (None, "?") else session
        b64 = base64.b64encode(raw_bytes).decode("ascii")
        proc = subprocess.run(
            [str(COLAB_PY), "-c", _COLAB_CONTENTS_PUT_SCRIPT, ep, sname, rel_path],
            input=b64,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=25,
            env=_clean_env(),
            creationflags=CREATE_NO_WINDOW,
        )
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "").strip().splitlines()
            raise RuntimeError(err[-1] if err else "Contents 写入失败")

    def _fetch_comfy_object_info(self, tunnel_url: str) -> dict[str, Any] | None:
        """从在线 ComfyUI 拉取节点定义（/object_info），用于将 API prompt 还原为标准画布工作流。"""
        base = tunnel_url.rstrip("/")
        try:
            resp = self._http.get(f"{base}/object_info", timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, dict):
                    return data
        except Exception:
            pass
        return None

    def _upload_workflow_file_ex(
        self, file_path: Path, is_png_extract: bool, tunnel_url: str | None
    ) -> tuple[str, bool, str]:
        """解析、按需转换并上传工作流 JSON，返回 `(状态文案, 是否已写入在线实例, 安全文件名)`。"""
        if is_png_extract:
            wf_text = extract_png_comfy_workflow(file_path)
            if not wf_text:
                raise ValueError(f"{file_path.name} 中未发现合法 ComfyUI 工作流")
            parsed = json.loads(wf_text)
            if not isinstance(parsed, dict):
                raise ValueError(f"{file_path.name} 中提取的工作流结构无效")
            safe_name = _sanitize_filename(file_path.with_suffix(".json").name, "workflow.json")
        else:
            wf_text = file_path.read_text(encoding="utf-8")
            parsed = json.loads(wf_text)
            if not isinstance(parsed, dict):
                raise ValueError("JSON 顶层结构不是工作流对象")
            safe_name = _sanitize_filename(file_path.name, "workflow.json")

        self._emit_upload_progress(safe_name, 20, "导工作流")
        if _is_api_prompt_dict(parsed):
            obj_info = self._fetch_comfy_object_info(tunnel_url) if tunnel_url else None
            converted = _api_prompt_to_ui_workflow(parsed, obj_info)
            wf_text = json.dumps(converted, ensure_ascii=False, indent=2)
        elif _ensure_workflow_uuid(parsed):
            wf_text = json.dumps(parsed, ensure_ascii=False, indent=2)

        self._emit_upload_progress(safe_name, 30, "导工作流")
        wf_bytes = wf_text.encode("utf-8")

        if tunnel_url:
            base = tunnel_url.rstrip("/")
            target_ep = f"{base}/userdata/{urllib.parse.quote('workflows/' + safe_name, safe='')}?overwrite=true"
            resp = self._http.post(
                target_ep,
                data=wf_bytes,
                headers={"Content-Type": "application/json"},
                timeout=15,
            )
            if resp.status_code in (200, 201):
                self._backup_workflow_to_gdrive_async(safe_name, wf_bytes)
                self._emit_upload_progress(safe_name, 90, "导工作流")
                return f"✅ 已导入工作流: {safe_name} (点我打开画布)", True, safe_name
            if resp.status_code != 404:
                resp.raise_for_status()

        if self._has_running_session():
            self._write_remote_file_via_contents(
                f"content/ComfyUI/user/default/workflows/{safe_name}",
                wf_bytes,
            )
            self._backup_workflow_to_gdrive_async(safe_name, wf_bytes)
            self._emit_upload_progress(safe_name, 100, "导工作流")
            return f"✅ 已写入实例工作流: {safe_name} (点我打开画布)", True, safe_name

        with tempfile.TemporaryDirectory(prefix="pet_wf_") as tmp_dir:
            tmp_wf = Path(tmp_dir) / safe_name
            tmp_wf.write_bytes(wf_bytes)
            self._rclone_upload_file(tmp_wf, f"{GDRIVE_WF_REMOTE}/{safe_name}", safe_name)
        return f"✅ 工作流已存云盘(下次开机自动载入): {safe_name}", False, safe_name

    def _upload_workflow_file(
        self, file_path: Path, is_png_extract: bool, tunnel_url: str | None
    ) -> tuple[str, bool]:
        """保留向后兼容的二元组返回签名。"""
        msg, online_ok, _ = self._upload_workflow_file_ex(file_path, is_png_extract, tunnel_url)
        return msg, online_ok

    def _upload_media_file(self, file_path: Path, tunnel_url: str | None) -> tuple[str, str]:
        if not tunnel_url:
            raise RuntimeError("当前无在线 ComfyUI 隧道，请先开机后再传参考素材")
        base = tunnel_url.rstrip("/")
        url = f"{base}/upload/image"
        with _StreamingMultipartFile(
            file_path,
            field_name="image",
            fields={"type": "input", "overwrite": "true"},
            on_progress=lambda pct: self._emit_upload_progress(file_path.name, pct, "传参考图"),
        ) as stream:
            resp = self._http.post(
                url,
                data=stream,
                headers={
                    "Content-Type": stream.content_type,
                    "Content-Length": str(len(stream)),
                },
                timeout=(10, 300),
            )
            resp.raise_for_status()
        uploaded_name = file_path.name
        try:
            data = resp.json()
            if isinstance(data, dict) and data.get("name"):
                uploaded_name = str(data["name"])
        except Exception:
            pass
        return f"✅ 已投递 {uploaded_name}，文件名已复制", uploaded_name

    def _upload_model_file(
        self, file_path: Path, tunnel_url: str | None, session: str = "comfy"
    ) -> tuple[str, bool]:
        safe_name = _sanitize_filename(file_path.name, "model.safetensors")
        remote_path = f"{GDRIVE_LORA_REMOTE}/{safe_name}"
        running = self._has_running_session()
        self._rclone_upload_file(
            file_path,
            remote_path,
            safe_name,
            pct_start=0,
            pct_span=70 if running else 100,
        )
        if not running:
            return f"✅ LoRA 已零耗电预传云盘: {safe_name}", False

        self._emit_upload_progress(safe_name, 78, "同步到实例")
        if COLAB_PY.exists():
            target = self._pick_target_session(session)
            sname = target["name"] if target and target.get("name") not in (None, "?") else session
            proc = _run_hidden(
                [
                    str(COLAB_PY),
                    "-c",
                    _COLAB_REMOTE_LORA_PULL_SCRIPT,
                    sname,
                    remote_path,
                    f"/content/ComfyUI/models/loras/{safe_name}",
                ],
                timeout=180,
            )
            if proc.returncode != 0:
                err = (proc.stderr or proc.stdout or "").strip().splitlines()
                raise RuntimeError(err[-1] if err else "实例拉取 LoRA 失败")

        if tunnel_url:
            try:
                self._http.get(f"{tunnel_url.rstrip('/')}/object_info/LoraLoader", timeout=5)
            except Exception:
                pass
        self._emit_upload_progress(safe_name, 100, "同步到实例")
        return f"✅ LoRA 已同步至云端并刷新: {safe_name}", True

    @staticmethod
    def _read_text_file_safe(fpath: Path, max_chars: int = 200_000) -> tuple[str, bool]:
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

    def _handle_llm_dropped_files(self, valid_files: list[Path]) -> dict[str, Any]:
        """在 LLM 对话模式下分流处理拖入或选取的文件：文本提取入剪贴板，图片引导网页直传。"""
        text_files = [p for p in valid_files if p.suffix.lower() in LLM_TEXT_EXTS]
        image_files = [p for p in valid_files if p.suffix.lower() in LLM_IMAGE_EXTS]
        gguf_or_model = [
            p for p in valid_files if p.suffix.lower() == ".gguf" or p.suffix.lower() in MODEL_EXTS
        ]
        running = self._has_running_session()
        done_action = "open_llm_webui" if running else None

        if text_files:
            chunks: list[str] = []
            any_trunc = False
            for fp in text_files:
                body, trunc = self._read_text_file_safe(fp)
                any_trunc = any_trunc or trunc
                if len(text_files) == 1:
                    chunks.append(body)
                else:
                    chunks.append(f"=== {fp.name} ===\n{body}")
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
            self._emit_event(
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
            self._emit_event(
                {
                    "type": "upload_done",
                    "summary": summary,
                    "action": done_action,
                }
            )
            return {
                "ok": True,
                "started": False,
                "service_kind": "llm",
                "count": len(image_files),
                "message": summary,
            }

        if gguf_or_model:
            msg = "💡 Qwen3.8 量化档位请在启动抽屉中切换 (Q4 / IQ4 / IQ3)，无需手动上传模型权重"
            self._emit_event({"type": "upload_error", "summary": msg})
            return {"ok": False, "service_kind": "llm", "message": msg}

        msg = "⚠️ 对话模式支持提取文本 (.txt/.md/.json/.py/.yaml/.csv) 或将图片拖入对话网页"
        self._emit_event({"type": "upload_error", "summary": msg})
        return {"ok": False, "service_kind": "llm", "message": msg}

    def submit_dropped_files(
        self,
        paths: list[str],
        png_mode: str | None = None,
        base_url_override: str | None = None,
    ) -> dict[str, Any]:
        """智能识别拖入或选取的本地文件，分流上传为参考素材、工作流、LoRA 或 LLM 文本提取。"""
        self.touch_activity()
        valid_files: list[Path] = []
        for raw_p in paths or []:
            if not raw_p:
                continue
            p = Path(str(raw_p))
            if p.is_file():
                valid_files.append(p)

        if not valid_files:
            return {"ok": False, "message": "未识别到有效本地文件"}

        if base_url_override is None and png_mode is None and self.is_llm_mode_active():
            return self._handle_llm_dropped_files(valid_files)

        if (
            len(valid_files) == 1
            and valid_files[0].suffix.lower() == ".png"
            and png_mode is None
            and extract_png_comfy_workflow(valid_files[0])
        ):
            target_png = valid_files[0]
            self._emit_event(
                {
                    "type": "png_workflow_choice",
                    "path": str(target_png),
                    "name": target_png.name,
                }
            )
            return {
                "ok": True,
                "choice_required": True,
                "message": "检测到 PNG 内嵌工作流，请选择投递方式",
            }

        if not self._upload_lock.acquire(blocking=False):
            return {"ok": False, "message": "正在上传其他文件，请稍候"}

        def _worker() -> None:
            try:
                turl = (
                    base_url_override.strip().rstrip("/")
                    if base_url_override
                    else self.resolve_active_tunnel_url(allow_fast_discover=True)
                )
                last_summary = ""
                last_clip: str | None = None
                can_open_comfy = False
                uploaded_wf_names: list[str] = []
                for idx, fpath in enumerate(valid_files, start=1):
                    ext = fpath.suffix.lower()
                    self._emit_upload_progress(fpath.name, 5, f"准备({idx}/{len(valid_files)})")
                    if ext in WORKFLOW_EXTS or (ext == ".png" and png_mode in ("workflow", "canvas")):
                        msg, online_ok, safe_wf_name = self._upload_workflow_file_ex(
                            fpath, is_png_extract=(ext == ".png"), tunnel_url=turl
                        )
                        last_summary = msg
                        can_open_comfy = can_open_comfy or online_ok
                        if online_ok and safe_wf_name:
                            uploaded_wf_names.append(safe_wf_name)
                    elif ext in MEDIA_EXTS:
                        msg, clip = self._upload_media_file(fpath, tunnel_url=turl)
                        last_summary = msg
                        last_clip = clip
                    elif ext in MODEL_EXTS:
                        msg, online_ok = self._upload_model_file(fpath, tunnel_url=turl)
                        last_summary = msg
                        can_open_comfy = can_open_comfy or online_ok
                    else:
                        raise ValueError(f"暂不支持该文件格式: {fpath.name}")

                done_action: str | None = "open_comfy" if can_open_comfy else None
                if uploaded_wf_names:
                    primary_wf = uploaded_wf_names[0]
                    extra_wfs = uploaded_wf_names[1:]
                    done_action = f"open_comfy_wf:{primary_wf}"
                    if turl:
                        self._emit_upload_progress(primary_wf, 95, "直铺画布")
                        canvas_res = self._push_or_open_workflow_on_canvas(
                            turl, primary_wf, extra_names=extra_wfs
                        )
                        if canvas_res.get("pushed_live"):
                            if len(uploaded_wf_names) == 1:
                                last_summary = f"✨ 已直铺画布: {primary_wf} (点我新开)"
                            else:
                                last_summary = f"✨ 已直铺 {len(uploaded_wf_names)} 个工作流到当前画布"
                        else:
                            if len(uploaded_wf_names) == 1:
                                last_summary = f"✨ 已打开并直铺: {primary_wf}"
                            else:
                                last_summary = f"✨ 已打开并直铺 {len(uploaded_wf_names)} 个工作流"
                    elif len(valid_files) > 1:
                        last_summary = f"✅ 已完成 {len(valid_files)} 个文件投递"
                elif len(valid_files) > 1:
                    last_summary = f"✅ 已完成 {len(valid_files)} 个文件投递"

                self._emit_event(
                    {
                        "type": "upload_done",
                        "summary": last_summary,
                        "clipboard_text": last_clip,
                        "action": done_action,
                    }
                )
            except Exception as exc:
                self._emit_event(
                    {
                        "type": "upload_error",
                        "summary": f"⚠️ 投递失败: {exc}",
                    }
                )
            finally:
                self._upload_lock.release()

        threading.Thread(target=_worker, daemon=True).start()
        return {
            "ok": True,
            "started": True,
            "count": len(valid_files),
            "message": f"开始投递 {len(valid_files)} 个文件...",
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="Colab 桌宠后端状态探针")
    parser.add_argument("--once", action="store_true", help="采集一次并打印 JSON 快照")
    args = parser.parse_args()

    backend = ColabPetBackend()
    if args.once:
        snap = backend.fetch_live_status()
        print(json.dumps(snap, ensure_ascii=False, indent=2))
        return 0 if snap.get("ok") else 1

    print(json.dumps(backend.get_cached_snapshot(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
