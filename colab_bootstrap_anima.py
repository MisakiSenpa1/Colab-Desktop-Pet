"""colab_bootstrap_anima.py —— Google Colab GPU 实例内一键部署 Anima + ComfyUI 自举脚本

用途：
    通过本机 `colab exec -s <会话名> -f colab_bootstrap_anima.py` 在远端 Colab
    GPU 实例内运行，自动完成环境校验、工具安装、ComfyUI 拉取（保留预装 CUDA
    PyTorch）、Hugging Face 底座权重高速下载、Google Drive 自训 LoRA 拉取、
    ComfyUI 后台守护启动，以及 Cloudflared 公网 WebUI 隧道拉起。

特点：
    - 全流程幂等设计，重复运行会自动跳过已完成步骤。
    - 不依赖交互式终端，后台启动 ComfyUI 与 cloudflared 时主动重定向并释放
      标准输入输出句柄，防止 `colab exec` 阻塞挂起。
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
COMFY_DB = Path("/root/comfy_db")
COMFY_LOG = Path("/root/comfy_colab.log")
RCLONE_CONF = Path("/tmp/rclone.conf")
_RCLONE_REMOTE = (os.environ.get("COLAB_RCLONE_REMOTE") or "gdrive_save").strip().rstrip(":") or "gdrive_save"
GDRIVE_LORA_REMOTE = f"{_RCLONE_REMOTE}:Colab_ComfyUI_Models/loras"
GDRIVE_WF_REMOTE = f"{_RCLONE_REMOTE}:Colab_ComfyUI_Models/workflows"
COMFY_API = "http://127.0.0.1:8188"

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


def install_pet_canvas_bridge():
    """在 ComfyUI custom_nodes 安装轻量直铺画布桥接扩展，实现桌宠一键直铺到画布。"""
    bridge_dir = COMFY_DIR / "custom_nodes" / "colab_pet_bridge"
    js_dir = bridge_dir / "js"
    js_dir.mkdir(parents=True, exist_ok=True)
    (bridge_dir / "__init__.py").write_text(_PET_BRIDGE_INIT_PY + "\n", encoding="utf-8")
    (js_dir / "direct_canvas.js").write_text(_PET_DIRECT_CANVAS_JS + "\n", encoding="utf-8")
    print(f"已安装桌宠直铺画布桥接扩展: {bridge_dir}", flush=True)

CLOUDFLARED_BIN = Path("/usr/local/bin/cloudflared")
CLOUDFLARED_LOG = Path("/root/cloudflared.log")
TUNNEL_URL_FILE = Path("/root/comfy_tunnel_url.txt")
TUNNEL_URL_RE = re.compile(r"https://(?!api\.)[a-z0-9-]+\.trycloudflare\.com")

# Anima 公开无门禁底座权重清单（模型相对路径, 下载地址, 精确字节数）
ANIMA_BASE_FILES = [
    (
        "diffusion_models/anima-base-v1.0.safetensors",
        "https://huggingface.co/circlestone-labs/Anima/resolve/main/split_files/diffusion_models/anima-base-v1.0.safetensors",
        4182218328,
    ),
    (
        "text_encoders/qwen_3_06b_base.safetensors",
        "https://huggingface.co/circlestone-labs/Anima/resolve/main/split_files/text_encoders/qwen_3_06b_base.safetensors",
        1192135096,
    ),
    (
        "vae/qwen_image_vae.safetensors",
        "https://huggingface.co/circlestone-labs/Anima/resolve/main/split_files/vae/qwen_image_vae.safetensors",
        253806246,
    ),
]

# 可选必需 LoRA 字节数校验清单（通过环境变量 REQUIRED_LORAS_JSON 传入 {文件名: 精确字节数}）
def _load_required_loras() -> dict[str, int]:
    """从环境变量加载可选 LoRA 字节校验字典。"""
    raw = (os.environ.get("REQUIRED_LORAS_JSON") or "").strip()
    if not raw:
        return {}
    data = json.loads(raw)
    if not isinstance(data, dict):
        return {}
    return {str(k): int(v) for k, v in data.items() if str(k).strip()}


REQUIRED_LORAS = _load_required_loras()


def run_cmd(cmd, cwd=None, check=True):
    """流式执行子进程命令并实时回显输出，失败时按需抛出异常。"""
    cmd_list = [str(x) for x in cmd]
    print(f"[CMD] {' '.join(cmd_list)}", flush=True)
    proc = subprocess.Popen(
        cmd_list,
        cwd=cwd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            print(line, end="", flush=True)
    finally:
        proc.stdout.close()
    returncode = proc.wait()
    if check and returncode != 0:
        raise subprocess.CalledProcessError(returncode, cmd_list)
    return subprocess.CompletedProcess(cmd_list, returncode)


def check_gpu_and_sdpa():
    """检查 GPU 显卡型号、CUDA 可用性及 BF16 SDPA 算子自检。"""
    print("=== [1/7] GPU 与 PyTorch SDPA 自检 ===", flush=True)
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        raise RuntimeError("未检测到 nvidia-smi，当前实例可能不是 GPU 运行时")

    info = subprocess.check_output(
        [nvidia_smi, "--query-gpu=name,memory.total", "--format=csv,noheader"],
        text=True,
    ).strip()
    print(f"GPU 硬件信息: {info}", flush=True)

    import torch

    print(f"PyTorch 版本: {torch.__version__}, CUDA 可用: {torch.cuda.is_available()}", flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() 为 False，无法启动 GPU 推理")

    # 在 CUDA 上以 BF16 验证 scaled_dot_product_attention 是否正常工作
    q = torch.randn(1, 16, 2048, 64, device="cuda", dtype=torch.bfloat16)
    out = torch.nn.functional.scaled_dot_product_attention(q, q, q)
    torch.cuda.synchronize()
    print(f"SDPA BF16 自检通过: {tuple(out.shape)}", flush=True)
    del q, out
    torch.cuda.empty_cache()


def ensure_tools():
    """确保实例内具备 rclone、aria2c 与 cloudflared 工具（无交互式安装）。"""
    print("=== [2/7] 检查 rclone、aria2c 与 cloudflared ===", flush=True)
    rclone_bin = Path("/usr/local/bin/rclone")
    if not shutil.which("rclone") and not rclone_bin.exists():
        zip_path = Path("/tmp/rclone.zip")
        url = "https://downloads.rclone.org/rclone-current-linux-amd64.zip"
        print(f"正在下载 rclone: {url}", flush=True)
        try:
            _download_with_urllib(url, zip_path)
        except Exception as exc:
            print(f"提示: urllib 下载 rclone 失败 ({exc})，尝试 curl / apt 回退安装...", flush=True)
            curl_res = run_cmd(
                ["curl", "-fsSL", "-A", "Mozilla/5.0", "-o", str(zip_path), url],
                check=False,
            )
            if curl_res.returncode != 0:
                zip_path.unlink(missing_ok=True)
                run_cmd(["bash", "-lc", "apt-get update -qq && apt-get install -y -qq rclone"])

        if zip_path.is_file():
            with zipfile.ZipFile(zip_path, "r") as zf:
                members = [m for m in zf.namelist() if m.endswith("/rclone") and m.count("/") == 1]
                if not members:
                    raise RuntimeError("rclone 压缩包内未找到可执行二进制文件")
                with zf.open(members[0]) as src, open(rclone_bin, "wb") as dst:
                    shutil.copyfileobj(src, dst)
            os.chmod(rclone_bin, 0o755)
            zip_path.unlink(missing_ok=True)
            print(f"已安装 rclone 至 {rclone_bin}", flush=True)
        elif shutil.which("rclone"):
            print(f"已通过 apt 安装 rclone: {shutil.which('rclone')}", flush=True)
        else:
            raise RuntimeError("rclone 安装失败")
    else:
        print(f"rclone 已就绪: {shutil.which('rclone') or rclone_bin}", flush=True)

    if not shutil.which("aria2c"):
        try:
            run_cmd(["bash", "-lc", "apt-get update -qq && apt-get install -y -qq aria2"])
            print(f"aria2c 安装完成: {shutil.which('aria2c')}", flush=True)
        except Exception as exc:
            # 下载可回退至 Python urllib，不在此中断流程
            print(f"警告: aria2c 安装失败，后续将回退 Python 流式下载 ({exc})", flush=True)
    else:
        print(f"aria2c 已就绪: {shutil.which('aria2c')}", flush=True)

    if not shutil.which("cloudflared") and not CLOUDFLARED_BIN.exists():
        cf_url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
        cf_tmp = CLOUDFLARED_BIN.with_suffix(".part")
        print(f"正在下载 cloudflared: {cf_url}", flush=True)
        _download_with_urllib(cf_url, cf_tmp)
        cf_tmp.replace(CLOUDFLARED_BIN)
        os.chmod(CLOUDFLARED_BIN, 0o755)
        print(f"已安装 cloudflared 至 {CLOUDFLARED_BIN}", flush=True)
    else:
        print(f"cloudflared 已就绪: {shutil.which('cloudflared') or CLOUDFLARED_BIN}", flush=True)


def ensure_comfyui():
    """克隆官方 ComfyUI 仓库并安装依赖（过滤 torch 相关包以保护预装 CUDA 环境）。"""
    print("=== [3/7] 准备 ComfyUI 与 Python 依赖 ===", flush=True)
    if not (COMFY_DIR / "main.py").is_file():
        run_cmd([
            "git",
            "clone",
            "--depth",
            "1",
            "https://github.com/comfyanonymous/ComfyUI.git",
            str(COMFY_DIR),
        ])
    else:
        print(f"ComfyUI 目录已存在，跳过克隆: {COMFY_DIR}", flush=True)

    req_src = COMFY_DIR / "requirements.txt"
    if not req_src.is_file():
        raise FileNotFoundError(f"未找到 ComfyUI 依赖清单: {req_src}")

    # 剔除 torch/torchvision/torchaudio 行，防止覆盖 Colab 预装版本
    filtered_lines = []
    for raw_line in req_src.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        pkg_head = stripped.split(";")[0].strip().lower()
        pkg_name = pkg_head.split("=")[0].split("<")[0].split(">")[0].split("!")[0].split("[")[0].strip()
        if pkg_name in {"torch", "torchvision", "torchaudio"}:
            continue
        filtered_lines.append(raw_line)

    req_filtered = Path("/tmp/comfy_reqs_notorch.txt")
    req_filtered.write_text("\n".join(filtered_lines) + "\n", encoding="utf-8")

    uv_bin = shutil.which("uv")
    if uv_bin:
        run_cmd([uv_bin, "pip", "install", "--system", "-q", "-r", str(req_filtered)])
    else:
        run_cmd([sys.executable, "-m", "pip", "install", "-q", "-r", str(req_filtered)])
    install_pet_canvas_bridge()
    print("ComfyUI 依赖安装完成（已保留预装 torch/CUDA）", flush=True)


def _download_with_urllib(url: str, dst: Path):
    """使用 Python 标准库按 8 MiB 分块下载文件，作为 aria2c 的回退手段。"""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    tmp = dst.with_suffix(dst.suffix + ".part")
    with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as f:
        while True:
            chunk = resp.read(8 * 1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    tmp.replace(dst)


def download_anima_base():
    """从 Hugging Face 下载并校验 Anima 底座三大组件（UNet / CLIP / VAE）。"""
    print("=== [4/7] 下载并校验 Anima 底座权重 ===", flush=True)
    models_root = COMFY_DIR / "models"
    aria2_bin = shutil.which("aria2c")

    for rel_path, url, expected_size in ANIMA_BASE_FILES:
        dst = models_root / rel_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.is_file() and dst.stat().st_size == expected_size:
            print(f"已存在且字节数匹配，跳过: {rel_path} ({expected_size} B)", flush=True)
            continue

        print(f"开始下载底座文件: {rel_path} (预期 {expected_size / (1024 ** 2):.2f} MiB)", flush=True)
        used_aria2 = False
        if aria2_bin:
            res = run_cmd(
                [
                    aria2_bin,
                    "-x", "16",
                    "-s", "16",
                    "-k", "1M",
                    "--summary-interval=5",
                    "--console-log-level=warn",
                    "--allow-overwrite=true",
                    "-d", str(dst.parent),
                    "-o", dst.name,
                    url,
                ],
                check=False,
            )
            used_aria2 = (res.returncode == 0)

        if not used_aria2:
            if dst.exists() and dst.stat().st_size != expected_size:
                dst.unlink(missing_ok=True)
            _download_with_urllib(url, dst)

        if not dst.is_file():
            raise FileNotFoundError(f"下载完成后仍未找到文件: {dst}")
        actual_size = dst.stat().st_size
        if actual_size != expected_size:
            raise ValueError(f"文件大小校验失败: {rel_path} 实际={actual_size} B, 预期={expected_size} B")
        print(f"校验通过: {rel_path} ({actual_size} B)", flush=True)


def pull_loras_from_gdrive():
    """若已上传 rclone.conf，则从 Google Drive 拉取自训 LoRA 与备份工作流。"""
    print("=== [5/7] 同步 Google Drive LoRA 与工作流（可选） ===", flush=True)
    if not RCLONE_CONF.is_file():
        print(
            f"未检测到 {RCLONE_CONF}，跳过 Google Drive LoRA/工作流拉取（将仅使用公共底座模型启动 ComfyUI）。",
            flush=True,
        )
        return

    loras_dir = COMFY_DIR / "models" / "loras"
    loras_dir.mkdir(parents=True, exist_ok=True)
    rclone_bin = shutil.which("rclone") or "/usr/local/bin/rclone"

    run_cmd(
        [
            rclone_bin,
            "copy",
            "--config", str(RCLONE_CONF),
            "--disable-http2",
            GDRIVE_LORA_REMOTE,
            str(loras_dir),
            "--include", "*.safetensors",
            "--transfers", "4",
        ],
        check=False,
    )

    for name, expected_size in REQUIRED_LORAS.items():
        fpath = loras_dir / name
        if not fpath.is_file():
            raise FileNotFoundError(f"主力 LoRA 缺失: {fpath}")
        actual_size = fpath.stat().st_size
        if actual_size != expected_size:
            raise ValueError(f"主力 LoRA 字节数不符: {name} 实际={actual_size} B, 预期={expected_size} B")
        print(f"LoRA 就绪: {name} ({actual_size} B)", flush=True)

    lora_count = sum(1 for _ in loras_dir.glob("*.safetensors"))
    print(f"LoRA 目录就绪，共 {lora_count} 个 .safetensors 文件", flush=True)

    # 增量拉取 Google Drive 云盘中随行备份的历史工作流（目录不存在时静默忽略）
    wf_dir = COMFY_DIR / "user" / "default" / "workflows"
    wf_dir.mkdir(parents=True, exist_ok=True)
    run_cmd(
        [
            rclone_bin,
            "copy",
            "--config",
            str(RCLONE_CONF),
            "--disable-http2",
            GDRIVE_WF_REMOTE,
            str(wf_dir),
            "--include",
            "*.json",
        ],
        check=False,
    )
    wf_count = sum(1 for _ in wf_dir.glob("*.json"))
    print(f"云盘随行工作流就绪，共 {wf_count} 个", flush=True)


def _probe_comfy_ready(timeout=2):
    """探测 ComfyUI 本地 API 是否已响应，就绪则返回 system_stats JSON 文本。"""
    try:
        with urllib.request.urlopen(f"{COMFY_API}/system_stats", timeout=timeout) as resp:
            if resp.status == 200:
                return resp.read().decode("utf-8", errors="ignore")
    except Exception:
        return None
    return None


def start_comfyui_and_wait():
    """以不阻塞 colab exec 的守护模式启动 ComfyUI，并轮询直至接口就绪。"""
    print("=== [6/7] 启动 ComfyUI 服务并等待就绪 ===", flush=True)
    COMFY_OUT.mkdir(parents=True, exist_ok=True)
    COMFY_DB.mkdir(parents=True, exist_ok=True)

    reserve_vram_raw = (os.environ.get("COMFY_RESERVE_VRAM") or "").strip()
    ensure_reserve = (os.environ.get("COMFY_ENSURE_RESERVE_VRAM") or "").strip() == "1"
    reserve_val: str | None = None
    if reserve_vram_raw:
        try:
            if float(reserve_vram_raw) > 0:
                reserve_val = reserve_vram_raw
        except ValueError:
            reserve_val = None

    existing = _probe_comfy_ready()
    need_restart = False
    if existing and ensure_reserve and reserve_val:
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

    if existing and not need_restart:
        print("ComfyUI 已在运行中，无需重复启动。", flush=True)
        print(f"system_stats: {existing[:300]}", flush=True)
    else:
        db_url = f"sqlite:////{COMFY_DB.as_posix().lstrip('/')}/comfyui.db"
        cmd = [
            sys.executable,
            "main.py",
            "--listen", "127.0.0.1",
            "--port", "8188",
            "--disable-auto-launch",
            "--use-pytorch-cross-attention",
            "--bf16-text-enc",
            "--enable-cors-header", "*",
            "--output-directory", str(COMFY_OUT),
            "--database-url", db_url,
        ]
        if reserve_val:
            cmd.extend(["--reserve-vram", reserve_val])
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        # 必须以独立文件描述符重定向 stdout/stderr 并及时在父进程关闭句柄，
        # 避免 colab exec 因等待管道 EOF 而永久卡死。
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

        print(f"已后台拉起 ComfyUI (PID={proc.pid})，日志文件: {COMFY_LOG}", flush=True)

        ready_payload = None
        for idx in range(1, 46):
            ready_payload = _probe_comfy_ready()
            if ready_payload:
                print(f"ComfyUI 就绪（耗时约 {idx * 2} 秒）", flush=True)
                print(f"system_stats: {ready_payload[:300]}", flush=True)
                break
            time.sleep(2)

        if not ready_payload:
            tail = ""
            if COMFY_LOG.is_file():
                tail = COMFY_LOG.read_text(encoding="utf-8", errors="ignore")[-2000:]
            raise TimeoutError(f"等待 ComfyUI 就绪超时（90s），末尾日志:\n{tail}")

    loras = sorted(p.name for p in (COMFY_DIR / "models" / "loras").glob("*.safetensors"))
    print(f"ComfyUI 与 LoRA 已就绪，共 {len(loras)} 个: {', '.join(loras)}", flush=True)


def start_cloudflared_tunnel(force_restart: bool = False) -> str:
    """以守护模式启动 cloudflared 隧道暴露 ComfyUI WebUI（优先命名隧道），返回公网 HTTPS 地址。"""
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
            TUNNEL_URL_FILE.write_text(cf_url + "\n", encoding="utf-8")
            print("========================================", flush=True)
            print(f"COMFYUI_TUNNEL_URL: {cf_url}", flush=True)
            print("========================================", flush=True)
            return cf_url
    elif not force_restart and pgrep_res.returncode == 0 and TUNNEL_URL_FILE.is_file():
        cached_text = TUNNEL_URL_FILE.read_text(encoding="utf-8", errors="ignore").strip()
        match = TUNNEL_URL_RE.search(cached_text)
        if match:
            url = match.group(0)
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
    if not Path(cf_bin).is_file():
        raise FileNotFoundError(f"未找到 cloudflared 可执行文件: {cf_bin}")

    if use_named:
        # 仅通过子进程环境变量传递连接器口令，严禁暴露于命令行参数
        cf_env = os.environ.copy()
        cf_env["TUNNEL_TOKEN"] = cf_token
        cmd = [cf_bin, "tunnel", "--no-autoupdate", "--protocol", "http2", "run"]
    else:
        cf_env = os.environ.copy()
        cmd = [cf_bin, "tunnel", "--no-autoupdate", "--protocol", "http2", "--url", COMFY_API]

    # 以独立日志文件守护拉起并立刻释放父进程句柄，防止 colab exec 等待 EOF 挂起
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

    print(f"已后台拉起 cloudflared (PID={proc.pid})，日志文件: {CLOUDFLARED_LOG}", flush=True)

    tunnel_url = None
    if use_named:
        for _ in range(35):
            if proc.poll() is not None:
                tail = CLOUDFLARED_LOG.read_text(encoding="utf-8", errors="ignore")[-2000:] if CLOUDFLARED_LOG.is_file() else ""
                if cf_token in tail:
                    tail = tail.replace(cf_token, "[REDACTED]")
                raise RuntimeError(f"cloudflared 命名隧道异常退出 (rc={proc.returncode})，末尾日志:\n{tail}")
            if CLOUDFLARED_LOG.is_file() and "Registered tunnel connection" in CLOUDFLARED_LOG.read_text(encoding="utf-8", errors="ignore"):
                tunnel_url = cf_url
                break
            time.sleep(1)
        if not tunnel_url:
            tail = CLOUDFLARED_LOG.read_text(encoding="utf-8", errors="ignore")[-2000:] if CLOUDFLARED_LOG.is_file() else ""
            if cf_token in tail:
                tail = tail.replace(cf_token, "[REDACTED]")
            raise TimeoutError(f"等待 cloudflared 命名隧道注册超时（35s），末尾日志:\n{tail}")
    else:
        for _ in range(30):
            if CLOUDFLARED_LOG.is_file():
                log_text = CLOUDFLARED_LOG.read_text(encoding="utf-8", errors="ignore")
                match = TUNNEL_URL_RE.search(log_text)
                if match:
                    tunnel_url = match.group(0)
                    break
            time.sleep(1)

        if not tunnel_url:
            tail = ""
            if CLOUDFLARED_LOG.is_file():
                tail = CLOUDFLARED_LOG.read_text(encoding="utf-8", errors="ignore")[-2000:]
            raise TimeoutError(f"等待 cloudflared 分配公网隧道超时（30s），末尾日志:\n{tail}")

    TUNNEL_URL_FILE.write_text(tunnel_url + "\n", encoding="utf-8")
    print("========================================", flush=True)
    print(f"COMFYUI_TUNNEL_URL: {tunnel_url}", flush=True)
    print("========================================", flush=True)
    return tunnel_url


def main():
    """串联执行全部部署步骤。"""
    check_gpu_and_sdpa()
    ensure_tools()
    ensure_comfyui()
    download_anima_base()
    pull_loras_from_gdrive()
    start_comfyui_and_wait()
    start_cloudflared_tunnel()


if __name__ == "__main__":
    main()
