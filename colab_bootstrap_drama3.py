"""colab_bootstrap_drama3.py —— Google Colab 环境部署 ComfyUI + Fish Audio Drama 3 语音克隆生态

设计目标：
    1. 保留 Colab 预装的 PyTorch + CUDA 环境，拉取最新官方 ComfyUI（>=0.40.0）与
       PedroMarinhoDev/ComfyUI-Drama3 原生节点。
    2. 下载并精确校验官方 FP8 专家模型（38.58 GB）、Drama 3 Codec（1.57 GB）与
       Qwen3-ASR 1.7B（4.08 GB），合计约 44.2 GB；推荐挂载 Colab G4 (96GB) 或 A100-80GB。
    3. 自动部署开箱即用的中文化语音克隆与自由演戏工作流至 user/default/workflows，
       并从存储号 Google Drive 拉取本地预清洗好的精选参考音频（/content/ComfyUI/input）与已保存音色。
    4. 后台非阻塞拉起 ComfyUI（127.0.0.1:8188）与 Cloudflared 公网隧道。
"""
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

COMFY_DIR = Path("/content/ComfyUI")
COMFY_OUT = Path("/root/comfy_out")
COMFY_LOG = Path("/root/comfy_bw1.log")
CLOUDFLARED_LOG = Path("/root/cloudflared.log")
TUNNEL_URL_FILE = Path("/root/comfy_tunnel_url.txt")
RCLONE_CONF = Path("/tmp/rclone.conf")
CLOUDFLARED_BIN = Path("/usr/local/bin/cloudflared")
COMFY_API = "http://127.0.0.1:8188"

GDRIVE_DRAMA3_REFS_REMOTE = "gdrive_save:Colab_ComfyUI_Models/drama3/refs"
GDRIVE_DRAMA3_VOICES_REMOTE = "gdrive_save:Colab_ComfyUI_Output/drama3_voices"
GDRIVE_WF_REMOTE = "gdrive_save:Colab_ComfyUI_Models/workflows"

TORCH_SKIP_RE = re.compile(r"^(torch|torchvision|torchaudio)([=<>!~\[]|$)", re.IGNORECASE)
TUNNEL_URL_RE = re.compile(r"https://(?!api\.)[a-z0-9-]+\.trycloudflare\.com")

# 精确校验的模型清单：(下载URL, 落盘相对路径, 精确字节数)
DRAMA3_MODELS = [
    (
        "https://huggingface.co/PedroMarinhoDev/Drama-3-ComfyUI/resolve/main/drama3_fp8experts_bf16.safetensors",
        COMFY_DIR / "models" / "diffusion_models" / "drama3_fp8experts_bf16.safetensors",
        38578426208,
    ),
    (
        "https://huggingface.co/PedroMarinhoDev/Drama-3-ComfyUI/resolve/main/drama3_codec.safetensors",
        COMFY_DIR / "models" / "vae" / "drama3_codec.safetensors",
        1565647484,
    ),
    (
        "https://huggingface.co/PedroMarinhoDev/Qwen3-ASR-ComfyUI/resolve/main/qwen3_asr_1.7b_bf16.safetensors",
        COMFY_DIR / "models" / "asr" / "qwen3_asr_1.7b_bf16.safetensors",
        4076192608,
    ),
]

_PET_BRIDGE_INIT_PY = '''"""Colab 桌宠直铺画布服务端桥接扩展。"""
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

_PET_DIRECT_CANVAS_JS = r"""/* colab_pet_direct_canvas v2 */
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


def install_pet_canvas_bridge():
    """在 ComfyUI custom_nodes 安装轻量直铺画布桥接扩展，实现桌宠一键直铺到画布。"""
    bridge_dir = COMFY_DIR / "custom_nodes" / "colab_pet_bridge"
    js_dir = bridge_dir / "js"
    js_dir.mkdir(parents=True, exist_ok=True)
    (bridge_dir / "__init__.py").write_text(_PET_BRIDGE_INIT_PY + "\n", encoding="utf-8")
    (js_dir / "direct_canvas.js").write_text(_PET_DIRECT_CANVAS_JS + "\n", encoding="utf-8")
    print(f"已安装桌宠直铺画布桥接扩展: {bridge_dir}", flush=True)


def run_cmd(cmd, check=True, cwd=None, capture=False):
    """在非交互标准输入模式下执行外部命令并实时转发输出。"""
    cmd_list = [str(x) for x in cmd]
    print(f"[执行] {' '.join(cmd_list)}", flush=True)
    if capture:
        return subprocess.run(
            cmd_list,
            cwd=str(cwd) if cwd else None,
            check=check,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
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
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
    rc = proc.wait()
    if check and rc != 0:
        raise subprocess.CalledProcessError(rc, cmd_list)
    return subprocess.CompletedProcess(cmd_list, rc)


def check_environment():
    """检查 GPU 显存、系统内存与可用磁盘空间。"""
    print("=== [1/7] 检测 Drama 3 运行环境 ===", flush=True)
    gpu_info = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
        stdin=subprocess.DEVNULL,
        text=True,
    ).strip()
    print(f"GPU: {gpu_info}", flush=True)

    mem_info = subprocess.check_output(["free", "-h"], stdin=subprocess.DEVNULL, text=True).strip()
    print(f"内存:\n{mem_info}", flush=True)

    disk_stat = shutil.disk_usage("/content")
    free_gib = disk_stat.free / (1024 ** 3)
    print(f"可用磁盘空间 (/content): {free_gib:.1f} GiB", flush=True)
    if free_gib < 46.0:
        raise RuntimeError(
            f"磁盘空间不足: 当前剩余 {free_gib:.1f} GiB，部署 Drama 3 FP8 生态至少需要 ~46.0 GiB"
        )

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("未检测到可用 CUDA GPU")
    vram_gib = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    cap = torch.cuda.get_device_capability(0)
    cuda_ver = str(torch.version.cuda or "")
    print(f"PyTorch: {torch.__version__} | CUDA: {cuda_ver} | SM: {cap} | VRAM: {vram_gib:.1f} GiB", flush=True)
    if vram_gib < 35.0:
        print(
            f"[提示] 当前显存为 {vram_gib:.1f} GiB，FP8 专家将通过系统内存按需缓存运行；推荐 G4 (96GB) 或 A100-80GB 获得满血缓存速度。",
            flush=True,
        )


def _download_with_urllib(url: str, dst: Path, retries: int = 3):
    """使用标准库 urllib 流式下载文件并原子替换。"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_suffix(dst.suffix + ".part")
    last_err = None
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
            print(f"[下载重试] {dst.name} 第 {attempt}/{retries} 次失败: {exc}", flush=True)
            time.sleep(3 * attempt)
    raise RuntimeError(f"下载失败: {url} ({last_err})")


def ensure_tools():
    """确保 rclone、aria2c 与 cloudflared 已安装就绪。"""
    print("=== [2/7] 安装传输与隧道工具 ===", flush=True)
    rclone_bin = Path("/usr/local/bin/rclone")
    if not shutil.which("rclone"):
        subprocess.run(
            ["bash", "-lc", "apt-get update -qq && apt-get install -y -qq rclone aria2"],
            check=False,
            stdin=subprocess.DEVNULL,
        )
        if not shutil.which("rclone"):
            zip_path = Path("/tmp/rclone-linux-amd64.zip")
            _download_with_urllib("https://downloads.rclone.org/rclone-current-linux-amd64.zip", zip_path)
            with zipfile.ZipFile(zip_path, "r") as zf:
                members = [m for m in zf.namelist() if m.endswith("/rclone") and m.count("/") == 1]
                with zf.open(members[0]) as src, open(rclone_bin, "wb") as dst:
                    shutil.copyfileobj(src, dst)
            os.chmod(rclone_bin, 0o755)
            zip_path.unlink(missing_ok=True)
    if not shutil.which("aria2c"):
        subprocess.run(
            ["bash", "-lc", "apt-get update -qq && apt-get install -y -qq aria2"],
            check=False,
            stdin=subprocess.DEVNULL,
        )
    if not shutil.which("cloudflared") and not CLOUDFLARED_BIN.exists():
        cf_tmp = CLOUDFLARED_BIN.with_suffix(".part")
        _download_with_urllib(
            "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64",
            cf_tmp,
        )
        cf_tmp.replace(CLOUDFLARED_BIN)
        os.chmod(CLOUDFLARED_BIN, 0o755)
    print(
        f"工具就绪: rclone={shutil.which('rclone') or rclone_bin}, aria2c={shutil.which('aria2c')}, cloudflared={shutil.which('cloudflared') or CLOUDFLARED_BIN}",
        flush=True,
    )


def ensure_comfyui_and_drama3_node():
    """准备最新官方 ComfyUI 与 ComfyUI-Drama3 自定义节点。"""
    print("=== [3/7] 准备 ComfyUI 与 Drama 3 节点 ===", flush=True)
    if not (COMFY_DIR / "main.py").is_file():
        run_cmd(["git", "clone", "--depth", "1", "https://github.com/Comfy-Org/ComfyUI.git", str(COMFY_DIR)])
    else:
        run_cmd(["git", "-C", str(COMFY_DIR), "pull", "--ff-only"], check=False)

    req_src = COMFY_DIR / "requirements.txt"
    req_lines = []
    for raw in req_src.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or TORCH_SKIP_RE.match(line):
            continue
        req_lines.append(line)
    req_filtered = Path("/tmp/comfy_drama3_reqs.txt")
    req_filtered.write_text("\n".join(req_lines) + "\n", encoding="utf-8")

    uv_bin = shutil.which("uv")
    if uv_bin:
        run_cmd([uv_bin, "pip", "install", "--system", "-q", "-r", str(req_filtered)])
    else:
        run_cmd([sys.executable, "-m", "pip", "install", "-q", "-r", str(req_filtered)])

    drama3_dir = COMFY_DIR / "custom_nodes" / "ComfyUI-Drama3"
    if not (drama3_dir / "__init__.py").is_file():
        run_cmd(["git", "clone", "--depth", "1", "https://github.com/PedroMarinhoDev/ComfyUI-Drama3.git", str(drama3_dir)])
    else:
        run_cmd(["git", "-C", str(drama3_dir), "pull", "--ff-only"], check=False)

    node_req = drama3_dir / "requirements.txt"
    if node_req.is_file():
        node_lines = [
            ln.strip()
            for ln in node_req.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.strip().startswith("#") and not TORCH_SKIP_RE.match(ln.strip())
        ]
        if node_lines:
            node_req_tmp = Path("/tmp/drama3_node_reqs.txt")
            node_req_tmp.write_text("\n".join(node_lines) + "\n", encoding="utf-8")
            run_cmd([sys.executable, "-m", "pip", "install", "-q", "-r", str(node_req_tmp)])

    install_pet_canvas_bridge()
    print("ComfyUI 与 ComfyUI-Drama3 节点依赖安装完成", flush=True)


def _download_exact(url: str, dst: Path, expected_bytes: int):
    """下载并校验单个模型文件精确字节数。"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_file() and dst.stat().st_size == expected_bytes:
        print(f"  [已就绪] {dst.name} ({dst.stat().st_size / (1024**3):.2f} GiB)", flush=True)
        return

    print(f"  [开始下载] {dst.name} <- {url}", flush=True)
    t0 = time.monotonic()
    used_aria = False
    if shutil.which("aria2c"):
        res = run_cmd(
            [
                "aria2c",
                "--console-log-level=notice",
                "--show-console-readout=false",
                "--summary-interval=5",
                "-x", "16",
                "-s", "16",
                "-k", "8M",
                "--allow-overwrite=true",
                "-c",
                "-d", str(dst.parent),
                "-o", dst.name,
                url,
            ],
            check=False,
        )
        used_aria = res.returncode == 0 and dst.is_file() and dst.stat().st_size == expected_bytes

    if not used_aria:
        if dst.exists() and dst.stat().st_size != expected_bytes:
            dst.unlink(missing_ok=True)
        _download_with_urllib(url, dst)

    if not dst.is_file() or dst.stat().st_size != expected_bytes:
        actual = dst.stat().st_size if dst.is_file() else -1
        raise RuntimeError(f"模型文件校验失败: {dst.name} 实际={actual} B, 预期={expected_bytes} B")

    dt = max(0.1, time.monotonic() - t0)
    gib = expected_bytes / (1024 ** 3)
    mib_s = expected_bytes / (1024 ** 2) / dt
    print(f"  [完成] {dst.name}: {gib:.2f} GiB ({dt:.1f}s, {mib_s:.1f} MiB/s)", flush=True)


def download_drama3_models():
    """下载 Drama 3 FP8 模型、Codec 编解码器与 Qwen3-ASR 1.7B。"""
    print("=== [4/7] 下载并校验 Drama 3 生态权重 ===", flush=True)
    for url, dst, expected_bytes in DRAMA3_MODELS:
        _download_exact(url, dst, expected_bytes)


def _default_voice_clone_workflow() -> dict:
    """内置开箱即用的 Drama 3 声音克隆工作流（默认选 main_01_我操宝宝原来你长这样.wav）。"""
    return {
        "id": "0e5a5f03-7e19-4014-b91c-4311d4434b1e",
        "revision": 0,
        "last_node_id": 11,
        "last_link_id": 11,
        "nodes": [
            {
                "id": 1,
                "type": "MarkdownNote",
                "pos": [20, 20],
                "size": [560, 640],
                "flags": {},
                "order": 0,
                "mode": 0,
                "inputs": [],
                "outputs": [],
                "properties": {},
                "widgets_values": [
                    "## Fish Audio Drama 3 · 声音克隆工作流\n\n**已预置模型**\n- `drama3_fp8experts_bf16.safetensors` (38.6 GB, FP8 满血版)\n- `drama3_codec.safetensors` (1.57 GB)\n- `qwen3_asr_1.7b_bf16.safetensors` (4.08 GB)\n\n**使用步骤**\n1. 在 **Load Reference Audio** 下拉选一条预置 `main_01_...wav`（也可拖 `.wav` 到桌宠上传）。\n2. **Qwen3-ASR Transcribe** 自动识别参考音频内容并传入编码。\n3. 在 **Drama 3 Generate** 输入台词（支持 `[sighs]`、`[laughs]`、`[whispers]` 等标签），点 **Queue** 生成。\n4. 桌宠会自动把 `.wav` 拉回本地 `_downloads/all/`。"
                ],
                "color": "#432",
                "bgcolor": "#653",
            },
            {
                "id": 2,
                "type": "Drama3Loader",
                "pos": [620, 20],
                "size": [360, 182],
                "flags": {},
                "order": 1,
                "mode": 0,
                "inputs": [
                    {"localized_name": "model_name", "name": "model_name", "type": "COMBO", "widget": {"name": "model_name"}, "link": None},
                    {"localized_name": "experts", "name": "experts", "type": "COMBO", "widget": {"name": "experts"}, "link": None},
                    {"localized_name": "expert_cache_gb", "name": "expert_cache_gb", "type": "FLOAT", "widget": {"name": "expert_cache_gb"}, "link": None},
                    {"localized_name": "second_gpu", "name": "second_gpu", "type": "COMBO", "widget": {"name": "second_gpu"}, "link": None},
                    {"localized_name": "split_layer", "name": "split_layer", "type": "INT", "widget": {"name": "split_layer"}, "link": None},
                    {"localized_name": "pin_experts", "name": "pin_experts", "type": "BOOLEAN", "widget": {"name": "pin_experts"}, "link": None},
                ],
                "outputs": [{"localized_name": "DRAMA3_MODEL", "name": "DRAMA3_MODEL", "type": "DRAMA3_MODEL", "links": [1]}],
                "properties": {"Node name for S&R": "Drama3Loader"},
                "widgets_values": ["drama3_fp8experts_bf16.safetensors", "in RAM", 0, "none", 0, True],
            },
            {
                "id": 3,
                "type": "Drama3CodecLoader",
                "pos": [620, 260],
                "size": [360, 58],
                "flags": {},
                "order": 2,
                "mode": 0,
                "inputs": [{"localized_name": "codec_name", "name": "codec_name", "type": "COMBO", "widget": {"name": "codec_name"}, "link": None}],
                "outputs": [{"localized_name": "DRAMA3_CODEC", "name": "DRAMA3_CODEC", "type": "DRAMA3_CODEC", "links": [2, 5]}],
                "properties": {"Node name for S&R": "Drama3CodecLoader"},
                "widgets_values": ["drama3_codec.safetensors"],
            },
            {
                "id": 4,
                "type": "Drama3LoadReferenceAudio",
                "pos": [620, 380],
                "size": [360, 230],
                "flags": {},
                "order": 3,
                "mode": 0,
                "inputs": [
                    {"localized_name": "audio", "name": "audio", "type": "COMBO", "widget": {"name": "audio"}, "link": None},
                    {"localized_name": "audioUI", "name": "audioUI", "type": "AUDIO_UI", "widget": {"name": "audioUI"}, "link": None},
                    {"localized_name": "choose file to upload", "name": "upload", "type": "AUDIOUPLOAD", "widget": {"name": "upload"}, "link": None},
                    {"localized_name": "trim", "name": "trim", "type": "BOOLEAN", "widget": {"name": "trim"}, "link": None},
                    {"localized_name": "length", "name": "length", "type": "COMBO", "widget": {"name": "length"}, "link": None},
                ],
                "outputs": [{"localized_name": "AUDIO", "name": "AUDIO", "type": "AUDIO", "links": [3, 9]}],
                "properties": {"Node name for S&R": "Drama3LoadReferenceAudio"},
                "widgets_values": ["main_01_我操宝宝原来你长这样.wav", "", True, "20 s"],
            },
            {
                "id": 5,
                "type": "Drama3EncodeVoice",
                "pos": [1020, 610],
                "size": [380, 108],
                "flags": {},
                "order": 6,
                "mode": 0,
                "inputs": [
                    {"localized_name": "codec", "name": "codec", "type": "DRAMA3_CODEC", "link": 2},
                    {"localized_name": "audio", "name": "audio", "type": "AUDIO", "link": 3},
                    {"localized_name": "transcript", "name": "transcript", "type": "STRING", "widget": {"name": "transcript"}, "link": 10},
                ],
                "outputs": [{"localized_name": "DRAMA3_VOICE", "name": "DRAMA3_VOICE", "type": "DRAMA3_VOICE", "links": [4, 11]}],
                "properties": {"Node name for S&R": "Drama3EncodeVoice"},
                "widgets_values": [""],
            },
            {
                "id": 6,
                "type": "Drama3Generate",
                "pos": [1020, 20],
                "size": [380, 372],
                "flags": {},
                "order": 7,
                "mode": 0,
                "inputs": [
                    {"localized_name": "model", "name": "model", "type": "DRAMA3_MODEL", "link": 1},
                    {"localized_name": "voice0", "name": "voices.voice0", "shape": 7, "type": "DRAMA3_VOICE", "link": 4},
                    {"localized_name": "voice1", "name": "voices.voice1", "shape": 7, "type": "DRAMA3_VOICE", "link": None},
                    {"localized_name": "text", "name": "text", "type": "STRING", "widget": {"name": "text"}, "link": None},
                    {"localized_name": "seed", "name": "seed", "type": "INT", "widget": {"name": "seed"}, "link": None},
                    {"localized_name": "temperature", "name": "temperature", "type": "FLOAT", "widget": {"name": "temperature"}, "link": None},
                    {"localized_name": "top_p", "name": "top_p", "type": "FLOAT", "widget": {"name": "top_p"}, "link": None},
                    {"localized_name": "top_k", "name": "top_k", "type": "INT", "widget": {"name": "top_k"}, "link": None},
                    {"localized_name": "max_frames", "name": "max_frames", "type": "INT", "widget": {"name": "max_frames"}, "link": None},
                    {"localized_name": "split_long_text", "name": "split_long_text", "type": "BOOLEAN", "widget": {"name": "split_long_text"}, "link": None},
                    {"localized_name": "speakers", "name": "speakers", "type": "COMBO", "widget": {"name": "speakers"}, "link": None},
                    {"localized_name": "voice_guidance", "name": "voice_guidance", "type": "FLOAT", "widget": {"name": "voice_guidance"}, "link": None},
                ],
                "outputs": [{"localized_name": "DRAMA3_CODES", "name": "DRAMA3_CODES", "type": "DRAMA3_CODES", "links": [6]}],
                "properties": {"Node name for S&R": "Drama3Generate"},
                "widgets_values": [
                    "[sighs] 哎，这都第几次了？[laughs] 算了，看在今天心情不错的份上，就勉强原谅你这一次吧。",
                    1,
                    "randomize",
                    1,
                    0.9,
                    0,
                    0,
                    True,
                    "one take per line",
                    0.0,
                ],
            },
            {
                "id": 7,
                "type": "Drama3Decode",
                "pos": [1440, 20],
                "size": [260, 74],
                "flags": {},
                "order": 9,
                "mode": 0,
                "inputs": [
                    {"localized_name": "codec", "name": "codec", "type": "DRAMA3_CODEC", "link": 5},
                    {"localized_name": "codes", "name": "codes", "type": "DRAMA3_CODES", "link": 6},
                    {"localized_name": "shorten_long_pauses", "name": "shorten_long_pauses", "type": "BOOLEAN", "widget": {"name": "shorten_long_pauses"}, "link": None},
                ],
                "outputs": [{"localized_name": "AUDIO", "name": "AUDIO", "type": "AUDIO", "links": [7]}],
                "properties": {"Node name for S&R": "Drama3Decode"},
                "widgets_values": [True],
            },
            {
                "id": 8,
                "type": "SaveAudio",
                "pos": [1440, 140],
                "size": [360, 112],
                "flags": {},
                "order": 10,
                "mode": 0,
                "inputs": [
                    {"localized_name": "audio", "name": "audio", "type": "AUDIO", "link": 7},
                    {"localized_name": "filename_prefix", "name": "filename_prefix", "type": "STRING", "widget": {"name": "filename_prefix"}, "link": None},
                    {"localized_name": "audioUI", "name": "audioUI", "type": "AUDIO_UI", "widget": {"name": "audioUI"}, "link": None},
                ],
                "outputs": [{"localized_name": "audio", "name": "audio", "type": "AUDIO", "links": None}],
                "properties": {},
                "widgets_values": ["drama3/drama3"],
            },
            {
                "id": 9,
                "type": "Qwen3ASRLoader",
                "pos": [620, 660],
                "size": [360, 58],
                "flags": {},
                "order": 4,
                "mode": 0,
                "inputs": [{"localized_name": "model_name", "name": "model_name", "type": "COMBO", "widget": {"name": "model_name"}, "link": None}],
                "outputs": [{"localized_name": "QWEN3_ASR_MODEL", "name": "QWEN3_ASR_MODEL", "type": "QWEN3_ASR_MODEL", "links": [8]}],
                "properties": {"Node name for S&R": "Qwen3ASRLoader"},
                "widgets_values": ["qwen3_asr_1.7b_bf16.safetensors"],
            },
            {
                "id": 10,
                "type": "Qwen3ASRTranscribe",
                "pos": [1020, 380],
                "size": [380, 180],
                "flags": {},
                "order": 5,
                "mode": 0,
                "inputs": [
                    {"localized_name": "model", "name": "model", "type": "QWEN3_ASR_MODEL", "link": 8},
                    {"localized_name": "audio", "name": "audio", "type": "AUDIO", "link": 9},
                    {"localized_name": "language", "name": "language", "type": "COMBO", "widget": {"name": "language"}, "link": None},
                    {"localized_name": "context", "name": "context", "type": "STRING", "widget": {"name": "context"}, "link": None},
                    {"localized_name": "max_new_tokens", "name": "max_new_tokens", "type": "INT", "widget": {"name": "max_new_tokens"}, "link": None},
                ],
                "outputs": [
                    {"localized_name": "text", "name": "text", "type": "STRING", "links": [10]},
                    {"localized_name": "language", "name": "language", "type": "STRING", "links": None},
                ],
                "properties": {"Node name for S&R": "Qwen3ASRTranscribe"},
                "widgets_values": ["Chinese", "", 4096],
            },
            {
                "id": 11,
                "type": "Drama3SaveVoice",
                "pos": [1440, 610],
                "size": [360, 58],
                "flags": {},
                "order": 8,
                "mode": 2,
                "inputs": [
                    {"localized_name": "voice", "name": "voice", "type": "DRAMA3_VOICE", "link": 11},
                    {"localized_name": "name", "name": "name", "type": "STRING", "widget": {"name": "name"}, "link": None},
                ],
                "outputs": [],
                "properties": {"Node name for S&R": "Drama3SaveVoice"},
                "widgets_values": ["drama3/voice"],
            },
        ],
        "links": [
            [2, 3, 0, 5, 0, "DRAMA3_CODEC"],
            [3, 4, 0, 5, 1, "AUDIO"],
            [5, 3, 0, 7, 0, "DRAMA3_CODEC"],
            [6, 6, 0, 7, 1, "DRAMA3_CODES"],
            [7, 7, 0, 8, 0, "AUDIO"],
            [1, 2, 0, 6, 0, "DRAMA3_MODEL"],
            [4, 5, 0, 6, 1, "DRAMA3_VOICE"],
            [8, 9, 0, 10, 0, "QWEN3_ASR_MODEL"],
            [9, 4, 0, 10, 1, "AUDIO"],
            [10, 10, 0, 5, 2, "STRING"],
            [11, 5, 0, 11, 0, "DRAMA3_VOICE"],
        ],
        "groups": [],
        "config": {},
        "extra": {"ds": {"scale": 0.75, "offset": [0, 0]}},
        "version": 0.4,
    }


def _default_tts_workflow() -> dict:
    """内置开箱即用的无参考自由演戏工作流。"""
    return {
        "id": "82e2926c-00f6-40d6-b9e0-0754c41dea11",
        "revision": 0,
        "last_node_id": 6,
        "last_link_id": 4,
        "nodes": [
            {
                "id": 1,
                "type": "MarkdownNote",
                "pos": [20, 20],
                "size": [560, 470],
                "flags": {},
                "order": 0,
                "mode": 0,
                "inputs": [],
                "outputs": [],
                "properties": {},
                "widgets_values": [
                    "## Fish Audio Drama 3 · 无参考纯文本演戏 (TTS)\n\n本工作流不传入参考音频，每次换 Seed 会随机生成一个新声音。\n要克隆固定音色，请打开 `drama3_voice_clone.json`。"
                ],
                "color": "#432",
                "bgcolor": "#653",
            },
            {
                "id": 2,
                "type": "Drama3Loader",
                "pos": [620, 20],
                "size": [360, 182],
                "flags": {},
                "order": 1,
                "mode": 0,
                "inputs": [
                    {"localized_name": "model_name", "name": "model_name", "type": "COMBO", "widget": {"name": "model_name"}, "link": None},
                    {"localized_name": "experts", "name": "experts", "type": "COMBO", "widget": {"name": "experts"}, "link": None},
                    {"localized_name": "expert_cache_gb", "name": "expert_cache_gb", "type": "FLOAT", "widget": {"name": "expert_cache_gb"}, "link": None},
                    {"localized_name": "second_gpu", "name": "second_gpu", "type": "COMBO", "widget": {"name": "second_gpu"}, "link": None},
                    {"localized_name": "split_layer", "name": "split_layer", "type": "INT", "widget": {"name": "split_layer"}, "link": None},
                    {"localized_name": "pin_experts", "name": "pin_experts", "type": "BOOLEAN", "widget": {"name": "pin_experts"}, "link": None},
                ],
                "outputs": [{"localized_name": "DRAMA3_MODEL", "name": "DRAMA3_MODEL", "type": "DRAMA3_MODEL", "links": [1]}],
                "properties": {"Node name for S&R": "Drama3Loader"},
                "widgets_values": ["drama3_fp8experts_bf16.safetensors", "in RAM", 0, "none", 0, True],
            },
            {
                "id": 3,
                "type": "Drama3CodecLoader",
                "pos": [620, 260],
                "size": [360, 58],
                "flags": {},
                "order": 2,
                "mode": 0,
                "inputs": [{"localized_name": "codec_name", "name": "codec_name", "type": "COMBO", "widget": {"name": "codec_name"}, "link": None}],
                "outputs": [{"localized_name": "DRAMA3_CODEC", "name": "DRAMA3_CODEC", "type": "DRAMA3_CODEC", "links": [2]}],
                "properties": {"Node name for S&R": "Drama3CodecLoader"},
                "widgets_values": ["drama3_codec.safetensors"],
            },
            {
                "id": 4,
                "type": "Drama3Generate",
                "pos": [1020, 20],
                "size": [380, 332],
                "flags": {},
                "order": 3,
                "mode": 0,
                "inputs": [
                    {"localized_name": "model", "name": "model", "type": "DRAMA3_MODEL", "link": 1},
                    {"localized_name": "voice0", "name": "voices.voice0", "shape": 7, "type": "DRAMA3_VOICE", "link": None},
                    {"localized_name": "text", "name": "text", "type": "STRING", "widget": {"name": "text"}, "link": None},
                    {"localized_name": "seed", "name": "seed", "type": "INT", "widget": {"name": "seed"}, "link": None},
                    {"localized_name": "temperature", "name": "temperature", "type": "FLOAT", "widget": {"name": "temperature"}, "link": None},
                    {"localized_name": "top_p", "name": "top_p", "type": "FLOAT", "widget": {"name": "top_p"}, "link": None},
                    {"localized_name": "top_k", "name": "top_k", "type": "INT", "widget": {"name": "top_k"}, "link": None},
                    {"localized_name": "max_frames", "name": "max_frames", "type": "INT", "widget": {"name": "max_frames"}, "link": None},
                    {"localized_name": "split_long_text", "name": "split_long_text", "type": "BOOLEAN", "widget": {"name": "split_long_text"}, "link": None},
                    {"localized_name": "speakers", "name": "speakers", "type": "COMBO", "widget": {"name": "speakers"}, "link": None},
                    {"localized_name": "voice_guidance", "name": "voice_guidance", "type": "FLOAT", "widget": {"name": "voice_guidance"}, "link": None},
                ],
                "outputs": [{"localized_name": "DRAMA3_CODES", "name": "DRAMA3_CODES", "type": "DRAMA3_CODES", "links": [3]}],
                "properties": {"Node name for S&R": "Drama3Generate"},
                "widgets_values": [
                    "你好呀！这是 Fish Audio Drama 3 无参考音色的自由演戏测试。[laughs] 每次更换随机种子，都会抽到一位新演员的嗓音。",
                    1,
                    "randomize",
                    1,
                    0.9,
                    0,
                    0,
                    True,
                    "one take per line",
                    0.0,
                ],
            },
            {
                "id": 5,
                "type": "Drama3Decode",
                "pos": [1440, 20],
                "size": [260, 74],
                "flags": {},
                "order": 4,
                "mode": 0,
                "inputs": [
                    {"localized_name": "codec", "name": "codec", "type": "DRAMA3_CODEC", "link": 2},
                    {"localized_name": "codes", "name": "codes", "type": "DRAMA3_CODES", "link": 3},
                    {"localized_name": "shorten_long_pauses", "name": "shorten_long_pauses", "type": "BOOLEAN", "widget": {"name": "shorten_long_pauses"}, "link": None},
                ],
                "outputs": [{"localized_name": "AUDIO", "name": "AUDIO", "type": "AUDIO", "links": [4]}],
                "properties": {"Node name for S&R": "Drama3Decode"},
                "widgets_values": [True],
            },
            {
                "id": 6,
                "type": "SaveAudio",
                "pos": [1440, 140],
                "size": [360, 112],
                "flags": {},
                "order": 5,
                "mode": 0,
                "inputs": [
                    {"localized_name": "audio", "name": "audio", "type": "AUDIO", "link": 4},
                    {"localized_name": "filename_prefix", "name": "filename_prefix", "type": "STRING", "widget": {"name": "filename_prefix"}, "link": None},
                    {"localized_name": "audioUI", "name": "audioUI", "type": "AUDIO_UI", "widget": {"name": "audioUI"}, "link": None},
                ],
                "outputs": [{"localized_name": "audio", "name": "audio", "type": "AUDIO", "links": None}],
                "properties": {},
                "widgets_values": ["drama3/drama3_tts"],
            },
        ],
        "links": [
            [1, 2, 0, 4, 0, "DRAMA3_MODEL"],
            [2, 3, 0, 5, 0, "DRAMA3_CODEC"],
            [3, 4, 0, 5, 1, "DRAMA3_CODES"],
            [4, 5, 0, 6, 0, "AUDIO"],
        ],
        "groups": [],
        "config": {},
        "extra": {"ds": {"scale": 0.9, "offset": [416, 110]}},
        "version": 0.4,
    }


def install_workflows_and_refs():
    """部署开箱即用 Drama 3 工作流并从云盘拉取精选参考 .wav 到 ComfyUI input 目录。"""
    print("=== [5/7] 部署工作流与同步精选参考音频 ===", flush=True)
    wf_dir = COMFY_DIR / "user" / "default" / "workflows"
    wf_dir.mkdir(parents=True, exist_ok=True)
    in_dir = COMFY_DIR / "input"
    in_dir.mkdir(parents=True, exist_ok=True)
    voices_dir = COMFY_OUT / "drama3_voices"
    voices_dir.mkdir(parents=True, exist_ok=True)

    rclone_bin = shutil.which("rclone") or "/usr/local/bin/rclone"
    if RCLONE_CONF.is_file() and Path(rclone_bin).is_file():
        probe = subprocess.run(
            [rclone_bin, "lsd", "--config", str(RCLONE_CONF), "--disable-http2", "gdrive_save:Colab_ComfyUI_Models/drama3"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            check=False,
        )
        if probe.returncode == 0 and "refs" in (probe.stdout or ""):
            print("从 Google Drive 拉取精选参考音频 (.wav)...", flush=True)
            run_cmd(
                [
                    rclone_bin, "copy", "--config", str(RCLONE_CONF), "--disable-http2",
                    GDRIVE_DRAMA3_REFS_REMOTE, str(in_dir),
                    "--include", "*.wav", "--transfers", "8",
                ],
                check=False,
            )

        # 同步云端保存的音色特征文件 (.safetensors)，先探测目录避免首次无目录时重试等待
        probe_out = subprocess.run(
            [rclone_bin, "lsd", "--config", str(RCLONE_CONF), "--disable-http2", "gdrive_save:Colab_ComfyUI_Output"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            check=False,
        )
        if probe_out.returncode == 0 and "drama3_voices" in (probe_out.stdout or ""):
            run_cmd(
                [
                    rclone_bin, "copy", "--config", str(RCLONE_CONF), "--disable-http2",
                    GDRIVE_DRAMA3_VOICES_REMOTE, str(voices_dir),
                    "--include", "*.safetensors",
                ],
                check=False,
            )

        # 同步云盘随行工作流
        run_cmd(
            [
                rclone_bin, "copy", "--config", str(RCLONE_CONF), "--disable-http2",
                GDRIVE_WF_REMOTE, str(wf_dir),
                "--include", "*.json",
            ],
            check=False,
        )

    # 校验参考音频下拉默认文件；若未拉到则挑首个可用 wav 填入工作流默认值
    wavs = sorted(p.name for p in in_dir.glob("*.wav"))
    clone_wf = _default_voice_clone_workflow()
    if wavs and "main_01_我操宝宝原来你长这样.wav" not in wavs:
        for node in clone_wf["nodes"]:
            if node.get("type") == "Drama3LoadReferenceAudio":
                node["widgets_values"][0] = wavs[0]
                break

    (wf_dir / "drama3_voice_clone.json").write_text(
        json.dumps(clone_wf, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (wf_dir / "drama3_text_to_speech.json").write_text(
        json.dumps(_default_tts_workflow(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"  工作流就绪: drama3_voice_clone.json / drama3_text_to_speech.json (参考音频 {len(wavs)} 条)", flush=True)


def _probe_comfy_ready() -> str | None:
    try:
        with urllib.request.urlopen(f"{COMFY_API}/system_stats", timeout=3) as resp:
            if resp.status == 200:
                return resp.read().decode("utf-8", errors="ignore")
    except Exception:
        return None
    return None


def _probe_drama3_node_loaded() -> bool:
    """检查运行中的 ComfyUI 是否已加载 Drama3Loader 节点及模型列表。"""
    try:
        with urllib.request.urlopen(f"{COMFY_API}/object_info/Drama3Loader", timeout=4) as resp:
            if resp.status != 200:
                return False
            txt = resp.read().decode("utf-8", errors="ignore")
            return "Drama3Loader" in txt and "drama3_fp8experts_bf16.safetensors" in txt
    except Exception:
        return False


def start_comfyui_and_wait():
    """后台非阻塞启动 ComfyUI 并等待 8188 端口就绪（跨生态复用实例时自动热重启加载新节点）。"""
    print("=== [6/7] 启动 ComfyUI 服务 ===", flush=True)
    COMFY_OUT.mkdir(parents=True, exist_ok=True)
    db_path = COMFY_DIR / "user" / "comfyui.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    reserve_vram_raw = (os.environ.get("COMFY_RESERVE_VRAM") or "").strip()
    ensure_reserve = (os.environ.get("COMFY_ENSURE_RESERVE_VRAM") or "").strip() == "1"
    reserve_val: str | None = None
    if reserve_vram_raw:
        try:
            if float(reserve_vram_raw) > 0:
                reserve_val = reserve_vram_raw
        except ValueError:
            reserve_val = None

    ready_payload = _probe_comfy_ready()
    if ready_payload:
        need_restart = False
        if not _probe_drama3_node_loaded():
            need_restart = True
            print("检测到存量 ComfyUI 未加载 Drama3 节点（跨生态热切换），正在重启 ComfyUI...", flush=True)
        elif ensure_reserve and reserve_val:
            pg = subprocess.run(
                ["pgrep", "-af", "main.py.*8188"],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
            )
            if "--reserve-vram" not in (pg.stdout or ""):
                need_restart = True
                print(
                    f"检测到存量 ComfyUI 未启用 --reserve-vram {reserve_val}（双栖显存保护），正在平滑重启...",
                    flush=True,
                )
        if not need_restart:
            print(f"ComfyUI 与 Drama3 节点已在运行中，直接复用。system_stats: {ready_payload[:240]}", flush=True)
            return
        subprocess.run(
            ["pkill", "-f", "main.py.*8188"],
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        for _ in range(15):
            if _probe_comfy_ready() is None:
                break
            time.sleep(1)

    cmd = [
        sys.executable,
        "main.py",
        "--listen", "127.0.0.1",
        "--port", "8188",
        "--enable-cors-header", "*",
        "--output-directory", str(COMFY_OUT),
        "--database-url", f"sqlite:///{db_path}",
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
    print(f"已拉起 ComfyUI (PID={proc.pid})，日志文件: {COMFY_LOG}", flush=True)

    for idx in range(1, 61):
        ready_payload = _probe_comfy_ready()
        if ready_payload:
            print(f"ComfyUI 就绪（耗时约 {idx * 2} 秒）", flush=True)
            print(f"system_stats: {ready_payload[:300]}", flush=True)
            return
        time.sleep(2)

    tail = COMFY_LOG.read_text(encoding="utf-8", errors="ignore")[-2000:] if COMFY_LOG.is_file() else ""
    raise TimeoutError(f"等待 ComfyUI 就绪超时（120s），末尾日志:\n{tail}")


def start_cloudflared_tunnel(force_restart: bool = False) -> str:
    """以守护模式启动或复用 cloudflared 暴露 ComfyUI 公网隧道（优先命名隧道）。"""
    print("=== [7/7] 启动 ComfyUI Cloudflared 公网隧道 ===", flush=True)
    cf_token = (os.environ.get("CF_TUNNEL_TOKEN") or "").strip()
    cf_url = (os.environ.get("CF_TUNNEL_URL") or "").strip().rstrip("/")
    use_named = bool(cf_token and cf_url)
    cohost_llm = (os.environ.get("COMFY_COHOST_LLM") or "").strip() == "1"

    pgrep_res = subprocess.run(
        ["pgrep", "-af" if use_named else "-f", "cloudflared tunnel"],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
    )
    if use_named:
        log_ok = CLOUDFLARED_LOG.is_file() and "Registered tunnel connection" in CLOUDFLARED_LOG.read_text(encoding="utf-8", errors="ignore")
        if (
            not force_restart
            and pgrep_res.returncode == 0
            and " run" in (pgrep_res.stdout or "")
            and (
                (TUNNEL_URL_FILE.is_file() and TUNNEL_URL_FILE.read_text(encoding="utf-8", errors="ignore").strip().rstrip("/") == cf_url)
                or (cohost_llm and log_ok)
            )
        ):
            print("复用运行中的 cloudflared 命名隧道。", flush=True)
            TUNNEL_URL_FILE.write_text(cf_url + "\n", encoding="utf-8")
            print("========================================", flush=True)
            print(f"COMFYUI_TUNNEL_URL: {cf_url}", flush=True)
            print("========================================", flush=True)
            return cf_url
    elif not force_restart and pgrep_res.returncode == 0 and TUNNEL_URL_FILE.is_file():
        cached_text = TUNNEL_URL_FILE.read_text(encoding="utf-8", errors="ignore").strip()
        m_cached = TUNNEL_URL_RE.search(cached_text)
        if m_cached:
            url = m_cached.group(0)
            print("复用运行中的 cloudflared 隧道。", flush=True)
            print("========================================", flush=True)
            print(f"COMFYUI_TUNNEL_URL: {url}", flush=True)
            print("========================================", flush=True)
            return url

    kill_pat = f"cloudflared tunnel.*8188" if (cohost_llm and not use_named) else "cloudflared tunnel"
    subprocess.run(
        ["pkill", "-f", kill_pat],
        check=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(0.5)
    cf_bin = shutil.which("cloudflared") or str(CLOUDFLARED_BIN)
    if use_named:
        # 仅通过子进程环境变量传递连接器口令，严禁暴露于命令行参数
        cf_env = os.environ.copy()
        cf_env["TUNNEL_TOKEN"] = cf_token
        cmd = [cf_bin, "tunnel", "--no-autoupdate", "--protocol", "http2", "run"]
    else:
        cf_env = os.environ.copy()
        cmd = [cf_bin, "tunnel", "--no-autoupdate", "--protocol", "http2", "--url", COMFY_API]

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
    print(f"已拉起 cloudflared (PID={proc.pid})", flush=True)

    tunnel_url = None
    if use_named:
        for _ in range(35):
            if proc.poll() is not None:
                tail = CLOUDFLARED_LOG.read_text(encoding="utf-8", errors="ignore")[-1500:] if CLOUDFLARED_LOG.is_file() else ""
                if cf_token in tail:
                    tail = tail.replace(cf_token, "[REDACTED]")
                raise RuntimeError(f"cloudflared 命名隧道异常退出 (rc={proc.returncode})，日志:\n{tail}")
            if CLOUDFLARED_LOG.is_file() and "Registered tunnel connection" in CLOUDFLARED_LOG.read_text(encoding="utf-8", errors="ignore"):
                tunnel_url = cf_url
                break
            time.sleep(1)
        if not tunnel_url:
            tail = CLOUDFLARED_LOG.read_text(encoding="utf-8", errors="ignore")[-1500:] if CLOUDFLARED_LOG.is_file() else ""
            if cf_token in tail:
                tail = tail.replace(cf_token, "[REDACTED]")
            raise TimeoutError(f"等待 cloudflared 命名隧道注册超时（35s），日志:\n{tail}")
    else:
        for _ in range(30):
            if CLOUDFLARED_LOG.is_file():
                m = TUNNEL_URL_RE.search(CLOUDFLARED_LOG.read_text(encoding="utf-8", errors="ignore"))
                if m:
                    tunnel_url = m.group(0)
                    break
            time.sleep(1)

        if not tunnel_url:
            tail = CLOUDFLARED_LOG.read_text(encoding="utf-8", errors="ignore")[-1500:] if CLOUDFLARED_LOG.is_file() else ""
            raise TimeoutError(f"等待 cloudflared 隧道地址超时，日志:\n{tail}")

    TUNNEL_URL_FILE.write_text(tunnel_url + "\n", encoding="utf-8")
    print("========================================", flush=True)
    print(f"COMFYUI_TUNNEL_URL: {tunnel_url}", flush=True)
    print("========================================", flush=True)
    return tunnel_url


def main():
    """串联执行 Drama 3 全生态部署流程。"""
    check_environment()
    ensure_tools()
    ensure_comfyui_and_drama3_node()
    download_drama3_models()
    install_workflows_and_refs()
    start_comfyui_and_wait()
    start_cloudflared_tunnel()


if __name__ == "__main__":
    main()
