"""colab_bootstrap_minimax_h3.py —— Google Colab 环境部署 ComfyUI + MiniMax-H3 全生态（生图 + 视频音频）

设计目标：
    1. 严格保留 Colab 预装的 PyTorch + CUDA 13.0 环境（对 RTX PRO 6000 Blackwell sm_120 与
       comfy-kitchen CUDA 加速核至关重要），仅安装缺失依赖。
    2. 部署共享 32B 多模态文本编码器（NVFP4 AWQ）、MiniMax-H3-Image（文生图/图生图编辑，
       可选 Qwen3.5-9B 扩写器）、MiniMax-H3 视频音频大模型（FL2VA 文生视频/首尾帧图生视频，
       可选 Ref2VA 多图音视频参考生视频）、对应 VAE、Turbo 加速 LoRA 与官方 Embeddings。
    3. 自动拉取并修正官方 ComfyUI MiniMax-H3 工作流 JSON（写入 user/default/workflows）及示例输入图，
       开箱即可在 WebUI 左侧 Workflows 面板直接加载运行，无需弹窗补下模型。
    4. 后台非阻塞拉起 ComfyUI（127.0.0.1:8188）与 Cloudflared 公网隧道。

环境变量可选项：
    MINIMAX_WITH_R2V=1   是否同时下载 Reference-to-Video (ref2va + 4step LoRA)，约 +21.35 GiB（默认 1）
    MINIMAX_WITH_PE=1    是否同时下载 Qwen3.5-9B 提示词扩写器，约 +6.84 GiB（默认 1）
    MINIMAX_LITE=0       设为 1 时等价于关闭 R2V 与 PE，仅装核心生图 + FL2VA 视频（约 59.8 GiB）
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

COMFY_DIR = Path("/content/ComfyUI")
COMFY_OUT = Path("/root/comfy_out")
COMFY_LOG = Path("/root/comfy_bw1.log")
CLOUDFLARED_LOG = Path("/root/cloudflared.log")
TUNNEL_URL_FILE = Path("/root/comfy_tunnel_url.txt")
RCLONE_CONF = Path("/tmp/rclone.conf")
CLOUDFLARED_BIN = Path("/usr/local/bin/cloudflared")
COMFY_API = "http://127.0.0.1:8188"

GDRIVE_MINIMAX_REMOTE = "gdrive_save:Colab_ComfyUI_Models/minimax_h3"
GDRIVE_WF_REMOTE = "gdrive_save:Colab_ComfyUI_Models/workflows"

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

# 与 ComfyUI 官方源码 requirements.txt 对齐但跳过会覆盖预装 cu130 的核心 torch 包
TORCH_SKIP_RE = re.compile(r"^(torch|torchvision|torchaudio)([=<>!~\[]|$)", re.IGNORECASE)
TUNNEL_URL_RE = re.compile(r"https://(?!api\.)[a-z0-9-]+\.trycloudflare\.com")


def _env_flag(name: str, default: bool) -> bool:
    """读取布尔环境变量。"""
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() not in ("0", "false", "no", "off", "")


LITE_MODE = _env_flag("MINIMAX_LITE", False)
WITH_R2V = False if LITE_MODE else _env_flag("MINIMAX_WITH_R2V", True)
WITH_PE = False if LITE_MODE else _env_flag("MINIMAX_WITH_PE", True)

# 核心共享权重（生图 + 视频共用同一个 Qwen3-VL-32B NVFP4 编码器，省磁盘与显存）
CORE_MODELS = [
    (
        "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
        COMFY_DIR / "models" / "text_encoders" / "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
        10_000_000_000,
    ),
    (
        "https://huggingface.co/Comfy-Org/MiniMax-H3-Image/resolve/main/diffusion_models/minimax_h3_image_pruned_int8_convrot.safetensors",
        COMFY_DIR / "models" / "diffusion_models" / "minimax_h3_image_pruned_int8_convrot.safetensors",
        15_000_000_000,
    ),
    (
        "https://huggingface.co/Comfy-Org/MiniMax-H3-Image/resolve/main/vae/minimax_h3_image_vae_fp16.safetensors",
        COMFY_DIR / "models" / "vae" / "minimax_h3_image_vae_fp16.safetensors",
        900_000_000,
    ),
    (
        "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors",
        COMFY_DIR / "models" / "diffusion_models" / "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
        15_000_000_000,
    ),
    (
        "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_video_vae_int8_convrot.safetensors",
        COMFY_DIR / "models" / "vae" / "minimax_h3_video_vae_int8_convrot.safetensors",
        2_000_000_000,
    ),
    (
        "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_audio_vae_fp32.safetensors",
        COMFY_DIR / "models" / "vae" / "minimax_h3_audio_vae_fp32.safetensors",
        400_000_000,
    ),
    (
        "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
        COMFY_DIR / "models" / "loras" / "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
        1_000_000_000,
    ),
]

PE_MODELS = [
    (
        "https://huggingface.co/Comfy-Org/Qwen3.5/resolve/main/text_encoders/qwen3.5_9b_w4a8_gptq.safetensors",
        COMFY_DIR / "models" / "text_encoders" / "qwen3.5_9b_w4a8_gptq.safetensors",
        5_000_000_000,
    ),
]

R2V_MODELS = [
    (
        "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors",
        COMFY_DIR / "models" / "diffusion_models" / "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
        15_000_000_000,
    ),
    (
        "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
        COMFY_DIR / "models" / "loras" / "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
        1_000_000_000,
    ),
]

H3_EMBEDDINGS = [
    "minimaxh3_art_is_explosion.safetensors",
    "minimaxh3_blooming_flowers.safetensors",
    "minimaxh3_bullet_time.safetensors",
    "minimaxh3_dark_magic.safetensors",
    "minimaxh3_fire_breath.safetensors",
    "minimaxh3_four_seasons.safetensors",
    "minimaxh3_kiss_camera.safetensors",
    "minimaxh3_spiral_ascent.safetensors",
    "minimaxh3_storm_magic.safetensors",
    "minimaxh3_truman_show.safetensors",
]

WORKFLOW_FILES = [
    "image_minimax_h3_image_t2i.json",
    "image_minimax_h3_image_edit.json",
    "video_minimax_h3_t2v.json",
    "video_minimax_h3_i2v.json",
]

SAMPLE_INPUTS = [
    "vintage_yellow_taxi.png",
    "transparent_rgb_gaming_mouse.png",
    "red_superboy_on_city_roof.png",
    "mecha_dragon_lightning.png",
]


def run_cmd(cmd, check=True, cwd=None, capture=False):
    """在非交互标准输入模式下执行外部命令，流式转发输出防止挂起。"""
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
    """检查 GPU 显存、系统内存、可用磁盘与 PyTorch CUDA 13.0 支持情况。"""
    print("=== [1/7] 检测 MiniMax-H3 运行环境 ===", flush=True)
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

    needed_gib = 68.0 + (7.5 if WITH_PE else 0.0) + (22.0 if WITH_R2V else 0.0)
    if free_gib < needed_gib:
        raise RuntimeError(
            f"磁盘空间不足: 当前剩余 {free_gib:.1f} GiB，部署当前模式至少需要 ~{needed_gib:.1f} GiB"
        )

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("未检测到可用 CUDA GPU")
    vram_gib = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    cap = torch.cuda.get_device_capability(0)
    cuda_ver = str(torch.version.cuda or "")
    print(f"PyTorch: {torch.__version__} | CUDA: {cuda_ver} | SM: {cap} | VRAM: {vram_gib:.1f} GiB", flush=True)

    major = int(cuda_ver.split(".")[0]) if cuda_ver and cuda_ver[0].isdigit() else 0
    if major < 13:
        print(
            f"[警告] 当前 PyTorch CUDA 版本为 {cuda_ver} (<13.0)，comfy-kitchen CUDA 加速核将自动禁用并回退。",
            flush=True,
        )
    if vram_gib < 35.0:
        print(
            f"[警告] 当前显存仅 {vram_gib:.1f} GiB，运行 MiniMax-H3 INT8 33B 模型可能极慢或 OOM，推荐使用 Colab G4 (96GB) 或 A100-80GB。",
            flush=True,
        )


def _download_with_urllib(url: str, dst: Path, retries: int = 3):
    """使用标准库 urllib 流式下载文件并校验临时文件替换。"""
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
    """确保 rclone、aria2c、cloudflared 已安装就绪。"""
    print("=== [2/7] 安装基础传输与隧道工具 ===", flush=True)
    rclone_bin = Path("/usr/local/bin/rclone")
    if not shutil.which("rclone"):
        res = subprocess.run(
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


def ensure_comfyui():
    """克隆或更新最新官方 ComfyUI 并安装依赖（保护预装 cu130 torch/torchvision）。"""
    print("=== [3/7] 准备最新官方 ComfyUI 与依赖 ===", flush=True)
    if not (COMFY_DIR / "main.py").is_file():
        run_cmd(["git", "clone", "--depth", "1", "https://github.com/Comfy-Org/ComfyUI.git", str(COMFY_DIR)])
    else:
        run_cmd(["git", "-C", str(COMFY_DIR), "pull", "--ff-only"])

    req_src = COMFY_DIR / "requirements.txt"
    req_lines = []
    skipped = []
    for raw in req_src.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if TORCH_SKIP_RE.match(line):
            skipped.append(line)
            continue
        req_lines.append(line)

    req_filtered = Path("/tmp/comfy_minimax_reqs.txt")
    req_filtered.write_text("\n".join(req_lines) + "\n", encoding="utf-8")
    print(f"已跳过覆盖预装包: {skipped}", flush=True)

    run_cmd([sys.executable, "-m", "pip", "install", "-q", "-r", str(req_filtered)])

    import comfy_kitchen  # noqa: F401
    print(f"comfy_kitchen 已就绪: {getattr(comfy_kitchen, '__version__', 'ok')}", flush=True)
    install_pet_canvas_bridge()


def _download_one(url: str, dst: Path, min_bytes: int):
    """下载单个模型文件（优先 aria2c 16线程，失败回退 urllib）。"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_file() and dst.stat().st_size >= min_bytes:
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
                "--summary-interval=10",
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
        used_aria = res.returncode == 0 and dst.is_file() and dst.stat().st_size >= min_bytes

    if not used_aria:
        _download_with_urllib(url, dst)

    if not dst.is_file() or dst.stat().st_size < min_bytes:
        raise RuntimeError(f"模型文件校验失败（尺寸过小或不存在）: {dst}")

    dt = max(0.1, time.monotonic() - t0)
    gib = dst.stat().st_size / (1024 ** 3)
    mib_s = dst.stat().st_size / (1024 ** 2) / dt
    print(f"  [完成] {dst.name}: {gib:.2f} GiB ({dt:.1f}s, {mib_s:.1f} MiB/s)", flush=True)


def download_minimax_models():
    """下载 MiniMax-H3 生图 + 视频音频全套模型、Turbo LoRA 与 Embeddings。"""
    print(
        f"=== [4/7] 下载 MiniMax-H3 生态模型 (WITH_PE={WITH_PE}, WITH_R2V={WITH_R2V}) ===",
        flush=True,
    )
    items = list(CORE_MODELS)
    if WITH_PE:
        items.extend(PE_MODELS)
    if WITH_R2V:
        items.extend(R2V_MODELS)

    for url, dst, min_bytes in items:
        _download_one(url, dst, min_bytes)

    # 并发下载 10 个官方轻量化 Embeddings
    emb_dir = COMFY_DIR / "models" / "embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        futs = []
        for name in H3_EMBEDDINGS:
            u = f"https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/embeddings/{name}"
            p = emb_dir / name
            if not (p.is_file() and p.stat().st_size > 1024):
                futs.append(pool.submit(_download_with_urllib, u, p))
        for fut in as_completed(futs):
            fut.result()
    print(f"已确认 {len(H3_EMBEDDINGS)} 个 MiniMax-H3 Embeddings 就绪。", flush=True)


def install_workflows_and_inputs():
    """拉取官方 MiniMax-H3 工作流 JSON（统一使用共享 NVFP4 文本编码器）并下载测试参考图。"""
    print("=== [5/7] 部署开箱即用工作流模板与示例素材 ===", flush=True)
    wf_dir = COMFY_DIR / "user" / "default" / "workflows"
    wf_dir.mkdir(parents=True, exist_ok=True)
    in_dir = COMFY_DIR / "input"
    in_dir.mkdir(parents=True, exist_ok=True)

    wf_list = list(WORKFLOW_FILES)
    if WITH_R2V:
        wf_list.append("video_minimax_h3_r2v.json")

    for name in wf_list:
        url = f"https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/templates/{name}"
        dst = wf_dir / name
        _download_with_urllib(url, dst)
        # 将生图工作流里默认指代的 int8_convrot 编码器统一映射为已下载的共享 nvfp4_awq，
        # 这样生图与视频共用一个 14.6 GiB 文本编码器，打开即跑不报缺少模型。
        raw = dst.read_text(encoding="utf-8")
        if "qwen3vl_32b_minimax_h3_int8_convrot.safetensors" in raw:
            patched = raw.replace(
                "qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
                "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
            )
            patched = patched.replace(
                "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
                "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
            )
            dst.write_text(patched, encoding="utf-8")
        print(f"  工作流就绪: {dst.name}", flush=True)

    for img_name in SAMPLE_INPUTS:
        url = f"https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/input/{img_name}"
        dst = in_dir / img_name
        if not (dst.is_file() and dst.stat().st_size > 1024):
            _download_with_urllib(url, dst)
    print("官方示例输入图片就绪。", flush=True)

    # 如云盘中有用户自定 MiniMax LoRA 或工作流补充目录则顺带拉取
    rclone_bin = shutil.which("rclone") or "/usr/local/bin/rclone"
    if RCLONE_CONF.is_file() and Path(rclone_bin).is_file():
        probe = subprocess.run(
            [rclone_bin, "lsd", "--config", str(RCLONE_CONF), "--disable-http2", "gdrive_save:Colab_ComfyUI_Models"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            check=False,
        )
        if probe.returncode == 0 and "minimax_loras" in (probe.stdout or ""):
            print("发现云盘 minimax_loras 目录，同步增量 LoRA...", flush=True)
            run_cmd(
                [
                    rclone_bin, "copy", "--config", str(RCLONE_CONF), "--disable-http2",
                    "gdrive_save:Colab_ComfyUI_Models/minimax_loras",
                    str(COMFY_DIR / "models" / "loras"),
                ],
                check=False,
            )
        if probe.returncode == 0 and "workflows" in (probe.stdout or ""):
            print("发现云盘 workflows 目录，同步随行工作流...", flush=True)
            run_cmd(
                [
                    rclone_bin, "copy", "--config", str(RCLONE_CONF), "--disable-http2",
                    GDRIVE_WF_REMOTE,
                    str(wf_dir),
                    "--include", "*.json",
                ],
                check=False,
            )


def _probe_comfy_ready() -> str | None:
    """探测本机 ComfyUI /system_stats 是否可响应。"""
    try:
        with urllib.request.urlopen(f"{COMFY_API}/system_stats", timeout=3) as resp:
            if resp.status == 200:
                return resp.read().decode("utf-8", errors="ignore")
    except Exception:
        return None
    return None


def start_comfyui_and_wait():
    """后台非阻塞启动 ComfyUI 并等待 8188 端口就绪。"""
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
        if ensure_reserve and reserve_val:
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
            print(f"ComfyUI 已在运行中，直接复用。system_stats: {ready_payload[:240]}", flush=True)
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
    print(f"已拉起 ComfyUI (PID={proc.pid})，日志: {COMFY_LOG}", flush=True)

    for idx in range(1, 61):
        ready_payload = _probe_comfy_ready()
        if ready_payload:
            print(f"ComfyUI 就绪（耗时约 {idx * 2} 秒）", flush=True)
            print(f"system_stats: {ready_payload[:300]}", flush=True)
            return
        time.sleep(2)

    tail = COMFY_LOG.read_text(encoding="utf-8", errors="ignore")[-2000:] if COMFY_LOG.is_file() else ""
    raise TimeoutError(f"等待 ComfyUI 就绪超时（120s），末尾日志:\n{tail}")


def start_cloudflared_tunnel() -> str:
    """以守护模式启动 cloudflared 暴露 ComfyUI 公网隧道（优先命名隧道）。"""
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
        if pgrep_res.returncode == 0 and " run" in (pgrep_res.stdout or "") and log_ok:
            print(f"复用运行中的 cloudflared 命名隧道: {cf_url}", flush=True)
            TUNNEL_URL_FILE.write_text(cf_url + "\n", encoding="utf-8")
            print("========================================", flush=True)
            print(f"COMFYUI_TUNNEL_URL: {cf_url}", flush=True)
            print("========================================", flush=True)
            return cf_url

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
    """串联执行 MiniMax-H3 全生态部署流程。"""
    check_environment()
    ensure_tools()
    ensure_comfyui()
    download_minimax_models()
    install_workflows_and_inputs()
    start_comfyui_and_wait()
    start_cloudflared_tunnel()


if __name__ == "__main__":
    main()
