"""Colab 远端实例部署与冒烟验证共享基础库。

供 `colab_bootstrap_*.py`、`verify_*.py` 与 `colab_ctl.py` 远端任务复用，由 `collect_local_py_deps`
自动打包随主脚本上传至 `/content/`，消除多份自举脚本间的桥接扩展重复拷贝与工具链漂移。
"""

from __future__ import annotations

import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Callable

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

BRIDGE_NODE_FILENAME = "colab_pet_bridge_node.py"
DEFAULT_TUNNEL_URL_RE = re.compile(
    r"https://(?:(?!api\.)[a-z0-9-]+\.trycloudflare\.com)"
)
TORCH_SKIP_RE = re.compile(r"^(torch|torchvision|torchaudio)([=<>!~\[]|$)", re.IGNORECASE)

# 唯一单源直铺画布前端扩展脚本（由桥接自定义节点挂载至 ComfyUI WEB_DIRECTORY=./js）
DIRECT_CANVAS_JS = r"""/* colab_pet_direct_canvas v2 */
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
    try:
        res = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=8, check=False)
        if res.returncode == 0:
            lines = [ln.strip() for ln in (res.stdout or "").splitlines() if ln.strip()]
            return lines[0] if lines else ""
    except Exception:
        pass
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
    anchors = [base_dir] if base_dir is not None else []
    anchors.extend([Path(__file__).resolve().parent, Path.cwd()])
    for root in anchors:
        cand = Path(root) / BRIDGE_NODE_FILENAME
        if cand.is_file():
            return cand.read_text(encoding="utf-8").strip()
    return ""


def install_pet_canvas_bridge(comfy_dir: Path = COMFY_DIR, base_dir: Path | None = None) -> None:
    """在 ComfyUI `custom_nodes` 安装桌宠直铺画布与公网鉴权桥接扩展。"""
    bridge_src = load_pet_bridge_init_py(base_dir)
    if not bridge_src:
        raise FileNotFoundError(f"未找到桥接扩展标准文件 {BRIDGE_NODE_FILENAME}")
    bridge_dir = comfy_dir / "custom_nodes" / "colab_pet_bridge"
    js_dir = bridge_dir / "js"
    js_dir.mkdir(parents=True, exist_ok=True)
    (bridge_dir / "__init__.py").write_text(bridge_src + "\n", encoding="utf-8")
    (js_dir / "direct_canvas.js").write_text(DIRECT_CANVAS_JS + "\n", encoding="utf-8")
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


def probe_comfy_ready(timeout: float = 3.0) -> str | None:
    """探测本机 ComfyUI /system_stats 是否可响应。"""
    try:
        with urllib.request.urlopen(f"{COMFY_API}/system_stats", timeout=timeout) as resp:
            if resp.status == 200:
                return resp.read().decode("utf-8", errors="ignore")
    except Exception:
        return None
    return None


def probe_bridge_version(timeout: float = 3.0) -> int:
    """探测运行中 ComfyUI 已加载的桌宠桥接节点版本号。"""
    try:
        with urllib.request.urlopen(f"{COMFY_API}/colab_pet/ping", timeout=timeout) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode("utf-8", errors="ignore") or "{}")
                return int(data.get("version") or 0)
    except Exception:
        pass
    return 0


def start_comfyui_and_wait(
    *,
    db_url: str | None = None,
    extra_args: list[str] | None = None,
    wait_timeout_sec: int = 120,
    extra_ready_probe: Callable[[], str | None] | None = None,
) -> None:
    """非阻塞守护启动 ComfyUI（自动确保随机鉴权密钥存在且关闭裸通配符 CORS）。"""
    print("=== [6/7] 启动 ComfyUI 服务并等待就绪 ===", flush=True)
    COMFY_OUT.mkdir(parents=True, exist_ok=True)
    ensure_comfy_auth_key()

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
        subprocess.run(["pkill", "-f", "main.py.*8188"], check=False, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
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
    for idx in range(1, steps + 1):
        ready_payload = probe_comfy_ready()
        if ready_payload:
            print(f"ComfyUI 就绪（耗时约 {idx * 2} 秒）", flush=True)
            print(f"system_stats: {ready_payload[:300]}", flush=True)
            return
        time.sleep(2)

    tail = COMFY_LOG.read_text(encoding="utf-8", errors="ignore")[-2000:] if COMFY_LOG.is_file() else ""
    raise TimeoutError(f"等待 ComfyUI 就绪超时（{wait_timeout_sec}s），末尾日志:\n{tail}")


def probe_public_tunnel_url(url: str, *, kind: str = "comfy", auth_key: str = "", timeout: float = 4.0) -> bool:
    """通过公网 HTTPS 验证隧道是否真正可用（防止仅凭旧日志假死复用）。"""
    base = str(url or "").strip().rstrip("/")
    if not base.startswith("https://"):
        return False
    target = f"{base}/health" if kind == "llm" else f"{base}/system_stats"
    headers = {"User-Agent": "Mozilla/5.0"}
    if kind != "llm" and auth_key:
        headers["Authorization"] = f"Bearer {auth_key}"
    try:
        req = urllib.request.Request(target, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status != 200:
                return False
            body = json.loads(resp.read().decode("utf-8", errors="ignore") or "{}")
            return isinstance(body, dict) and (body.get("status") == "ok" if kind == "llm" else ("system" in body or "devices" in body))
    except urllib.error.HTTPError as exc:
        if kind != "llm" and exc.code == 401:
            try:
                with urllib.request.urlopen(urllib.request.Request(f"{base}/__pet_health", headers={"User-Agent": "Mozilla/5.0"}), timeout=timeout) as hr:
                    hb = json.loads(hr.read().decode("utf-8", errors="ignore") or "{}")
                    return hr.status == 200 and isinstance(hb, dict) and hb.get("status") == "ok"
            except Exception:
                return False
        return False
    except Exception:
        return False


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
) -> str:
    """统一拉起或复用 `cloudflared` 隧道（支持 quick/named 与双栖隔离，复用前真实 HTTPS 验活）。"""
    eff_tok = (cf_token or "").strip()
    eff_url = (cf_url or "").strip().rstrip("/")
    use_named = bool(eff_tok and eff_url)
    url_file = Path(tunnel_url_file or (LLM_TUNNEL_URL_FILE if (service_kind == "llm" and cohost_other) else TUNNEL_URL_FILE))
    log_path = Path(log_file or (Path("/root/cloudflared_llm.log") if (service_kind == "llm" and cohost_other and not use_named) else CLOUDFLARED_LOG))

    pgrep_pat = "cloudflared tunnel" if use_named else f"cloudflared tunnel.*{local_port}"
    pgrep_res = subprocess.run(["pgrep", "-af" if use_named else "-f", pgrep_pat], stdin=subprocess.DEVNULL, capture_output=True, text=True)
    proc_out = pgrep_res.stdout or ""
    auth_k = ensure_comfy_auth_key() if service_kind != "llm" else ""

    if not force_restart and pgrep_res.returncode == 0:
        if use_named and " run" in proc_out and probe_public_tunnel_url(eff_url, kind=service_kind, auth_key=auth_k):
            print(f"复用运行中的命名隧道 ({service_kind}): {eff_url}", flush=True)
            url_file.write_text(eff_url + "\n", encoding="utf-8")
            return eff_url
        if (not use_named) and f"127.0.0.1:{local_port}" in proc_out and url_file.is_file():
            m_cached = tunnel_re.search(url_file.read_text(encoding="utf-8", errors="ignore"))
            if m_cached and probe_public_tunnel_url(m_cached.group(0), kind=service_kind, auth_key=auth_k):
                print(f"复用运行中的 cloudflared 隧道 (:{local_port}): {m_cached.group(0)}", flush=True)
                return m_cached.group(0)

    if cohost_other:
        kill_pat = "cloudflared tunnel.* run" if use_named else f"cloudflared tunnel.*{local_port}"
    else:
        kill_pat = "cloudflared tunnel"
    subprocess.run(["pkill", "-f", kill_pat], check=False, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
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

    log_f = open(log_path, "wb")
    try:
        proc = subprocess.Popen(cmd, env=cf_env, stdin=subprocess.DEVNULL, stdout=log_f, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
    finally:
        log_f.close()
    print(f"已后台拉起 cloudflared (PID={proc.pid}, port={local_port})，日志: {log_path}", flush=True)

    tunnel_url: str | None = None
    if use_named:
        for _ in range(35):
            if proc.poll() is not None:
                tail = log_path.read_text(encoding="utf-8", errors="ignore")[-1500:] if log_path.is_file() else ""
                if eff_tok:
                    tail = tail.replace(eff_tok, "[REDACTED]")
                raise RuntimeError(f"cloudflared 命名隧道异常退出 (rc={proc.returncode})，日志:\n{tail}")
            txt = log_path.read_text(encoding="utf-8", errors="ignore") if log_path.is_file() else ""
            if "Registered tunnel connection" in txt or probe_public_tunnel_url(eff_url, kind=service_kind, auth_key=auth_k, timeout=2.0):
                tunnel_url = eff_url
                break
            time.sleep(1)
        if not tunnel_url:
            tail = log_path.read_text(encoding="utf-8", errors="ignore")[-1500:] if log_path.is_file() else ""
            if eff_tok:
                tail = tail.replace(eff_tok, "[REDACTED]")
            raise TimeoutError(f"等待 cloudflared 命名隧道注册超时（35s），日志:\n{tail}")
    else:
        for _ in range(30):
            if log_path.is_file():
                m = tunnel_re.search(log_path.read_text(encoding="utf-8", errors="ignore"))
                if m:
                    tunnel_url = m.group(0)
                    break
            time.sleep(1)
        if not tunnel_url:
            tail = log_path.read_text(encoding="utf-8", errors="ignore")[-1500:] if log_path.is_file() else ""
            raise TimeoutError(f"等待 cloudflared 分配隧道地址超时（30s），日志:\n{tail}")

    url_file.write_text(tunnel_url + "\n", encoding="utf-8")
    return tunnel_url


def start_comfy_tunnel_standard() -> str:
    """ComfyUI 自举标准步骤 [7/7]：读取环境隧道配置、启停隧道并打印标准标志。"""
    print("=== [7/7] 启动 ComfyUI Cloudflared 公网隧道 ===", flush=True)
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
    """汇总并在远端以 0600 写入 `/root/colab_service_meta.json`，严禁把 Comfy 域名错写进 LLM 槽位。"""
    tun_re = re.compile(tunnel_pattern) if tunnel_pattern else DEFAULT_TUNNEL_URL_RE
    meta = read_json_dict(SERVICE_META_FILE) if is_dual else {}

    def _read_url(p: Path, fb: str) -> str:
        if p.is_file():
            m = tun_re.search(p.read_text(encoding="utf-8", errors="ignore"))
            if m:
                return m.group(0).rstrip("/")
        return str(fb or "").strip().rstrip("/")

    comfy_u = _read_url(TUNNEL_URL_FILE, str(meta.get("comfy_tunnel_url") or cf_comfy_url))
    llm_u = _read_url(LLM_TUNNEL_URL_FILE, str(meta.get("llm_tunnel_url") or cf_llm_url)) if is_dual else ""
    comfy_auth_k = ensure_comfy_auth_key()

    if is_dual:
        meta.update({
            "kind": "dual",
            "profiles": list(profiles),
            "comfy_profiles": list(comfy_profiles),
            "llm_profile": llm_profile,
            "comfy_port": 8188,
            "llm_port": 18080,
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
    try:
        import torch

        info["torch"] = str(torch.__version__)
        info["cuda"] = str(torch.version.cuda or "")
        if torch.cuda.is_available():
            info["gpu"] = str(torch.cuda.get_device_name(0))
    except Exception:
        pass
    try:
        write_private_json(Path("/root/colab_runtime_versions.json"), info)
    except Exception:
        pass
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
