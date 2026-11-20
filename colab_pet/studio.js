/* Colab Studio 前端控制脚本 —— 双窗实时同步、算力拓扑、画廊检视、热调参及诊断终端 */
(function () {
  "use strict";

  /* ----------------------------------------------------------------
     基础工具与安全渲染辅助
     ---------------------------------------------------------------- */
  const el = (id) => document.getElementById(id);
  const api = () => (window.pywebview && window.pywebview.api) || null;
  const hasApi = (name) => Boolean(api() && typeof api()[name] === "function");

  async function apiCall(name, ...args) {
    if (!hasApi(name)) return null;
    try {
      return await api()[name](...args);
    } catch (err) {
      return { ok: false, message: cleanText(err && err.message ? err.message : String(err)) };
    }
  }

  /* 过滤后端可能附带的 Emoji 与符号字符，保障 Studio 界面保持纯矢量精致感 */
  function cleanText(raw) {
    return String(raw == null ? "" : raw)
      .replace(/[\u{1F300}-\u{1FAFF}\u{2600}-\u{27BF}\u{FE00}-\u{FE0F}\u{200D}]/gu, "")
      .replace(/\s{2,}/g, " ")
      .trim();
  }

  /* 仅剔除 Emoji 与杂项符号，保留代码块缩进与换行排版 */
  function stripEmoji(raw) {
    return String(raw == null ? "" : raw)
      .replace(/[\u{1F300}-\u{1FAFF}\u{2600}-\u{27BF}\u{FE00}-\u{FE0F}\u{200D}]/gu, "");
  }

  function h(tag, props, ...children) {
    const node = document.createElement(tag);
    if (props) {
      for (const [k, v] of Object.entries(props)) {
        if (v === undefined || v === null) continue;
        if (k === "cls") node.className = v;
        else if (k === "txt") node.textContent = String(v);
        else if (k === "data") Object.assign(node.dataset, v);
        else if (k === "attrs") {
          for (const [ak, av] of Object.entries(v)) {
            node.setAttribute(ak, String(av));
          }
        } else if (k === "on") {
          for (const [ek, fn] of Object.entries(v)) {
            node.addEventListener(ek, fn);
          }
        } else {
          node[k] = v;
        }
      }
    }
    for (const child of children.flat()) {
      if (child !== undefined && child !== null && child !== false) {
        node.appendChild(typeof child === "string" ? document.createTextNode(child) : child);
      }
    }
    return node;
  }

  /* 创建统一 1.75px 圆角几何矢量 SVG 图标 */
  function createSvgIcon(kind, size = 14, cls = "icon-svg") {
    const ns = "http://www.w3.org/2000/svg";
    const svg = document.createElementNS(ns, "svg");
    svg.setAttribute("width", String(size));
    svg.setAttribute("height", String(size));
    svg.setAttribute("viewBox", "0 0 24 24");
    svg.setAttribute("fill", "none");
    svg.setAttribute("stroke", "currentColor");
    svg.setAttribute("stroke-width", "1.75");
    svg.setAttribute("stroke-linecap", "round");
    svg.setAttribute("stroke-linejoin", "round");
    svg.setAttribute("aria-hidden", "true");
    if (cls) svg.setAttribute("class", cls);

    const pathsByKind = {
      image: [
        ["rect", { x: "3", y: "4", width: "18", height: "16", rx: "2.5" }],
        ["circle", { cx: "8.5", cy: "9.5", r: "1.5" }],
        ["path", { d: "M21 15l-4.5-4.5L7 20" }],
      ],
      video: [
        ["rect", { x: "2.5", y: "5", width: "14", height: "14", rx: "2.5" }],
        ["path", { d: "M16.5 10l5-3v10l-5-3z" }],
      ],
      audio: [
        ["path", { d: "M12 3a3 3 0 0 0-3 3v5a3 3 0 0 0 6 0V6a3 3 0 0 0-3-3z" }],
        ["path", { d: "M19 10v1a7 7 0 0 1-14 0v-1" }],
        ["line", { x1: "12", y1: "18", x2: "12", y2: "21" }],
      ],
      llm: [
        ["path", { d: "M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" }],
      ],
      star: [
        ["polygon", { points: "12 2 15.1 8.3 22 9.3 17 14.1 18.2 21 12 17.7 5.8 21 7 14.1 2 9.3 8.9 8.3 12 2" }],
      ],
      pencil: [
        ["path", { d: "M12 20h9" }],
        ["path", { d: "M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4z" }],
      ],
      close: [
        ["path", { d: "M18 6L6 18" }],
        ["path", { d: "M6 6l12 12" }],
      ],
      checkCircle: [
        ["circle", { cx: "12", cy: "12", r: "9" }],
        ["path", { d: "M9 12l2 2 4-4" }],
      ],
      alertTriangle: [
        ["path", { d: "M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" }],
        ["line", { x1: "12", y1: "9", x2: "12", y2: "13" }],
        ["line", { x1: "12", y1: "17", x2: "12.01", y2: "17" }],
      ],
      xCircle: [
        ["circle", { cx: "12", cy: "12", r: "9" }],
        ["path", { d: "M15 9l-6 6" }],
        ["path", { d: "M9 9l6 6" }],
      ],
      spark: [
        ["path", { d: "M12 3l1.9 5.1L19 10l-5.1 1.9L12 17l-1.9-5.1L5 10l5.1-1.9z" }],
      ],
      chevronRight: [
        ["polyline", { points: "9 18 15 12 9 6" }],
      ],
      check: [
        ["polyline", { points: "20 6 9 17 4 12" }],
      ],
      zap: [
        ["polygon", { points: "13 2 3 14 12 14 11 22 21 10 12 10 13 2" }],
      ],
      plus: [
        ["path", { d: "M12 5v14" }],
        ["path", { d: "M5 12h14" }],
      ],
      sliders: [
        ["line", { x1: "4", y1: "21", x2: "4", y2: "14" }],
        ["line", { x1: "4", y1: "10", x2: "4", y2: "3" }],
        ["line", { x1: "12", y1: "21", x2: "12", y2: "12" }],
        ["line", { x1: "12", y1: "8", x2: "12", y2: "3" }],
        ["line", { x1: "20", y1: "21", x2: "20", y2: "16" }],
        ["line", { x1: "20", y1: "12", x2: "20", y2: "3" }],
        ["line", { x1: "1", y1: "14", x2: "7", y2: "14" }],
        ["line", { x1: "9", y1: "8", x2: "15", y2: "8" }],
        ["line", { x1: "17", y1: "16", x2: "23", y2: "16" }],
      ],
      shield: [
        ["path", { d: "M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" }],
      ],
      terminal: [
        ["polyline", { points: "4 17 10 11 4 5" }],
        ["line", { x1: "12", y1: "19", x2: "20", y2: "19" }],
      ],
      brain: [
        ["circle", { cx: "12", cy: "12", r: "3" }],
        ["path", { d: "M12 2v3M12 19v3M4.9 4.9l2.1 2.1M17 17l2.1 2.1M2 12h3M19 12h3M4.9 19.1L7 17M17 7l2.1-2.1" }],
      ],
      history: [
        ["path", { d: "M3 12a9 9 0 1 0 3-6.7L3 8" }],
        ["path", { d: "M3 3v5h5" }],
        ["polyline", { points: "12 7 12 12 15.5 14" }],
      ],
      arrowDown: [
        ["line", { x1: "12", y1: "5", x2: "12", y2: "19" }],
        ["polyline", { points: "19 12 12 19 5 12" }],
      ],
      trash: [
        ["polyline", { points: "3 6 5 6 21 6" }],
        ["path", { d: "M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6" }],
        ["path", { d: "M10 11v6M14 11v6" }],
        ["path", { d: "M9 6V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2" }],
      ],
    };

    const specs = pathsByKind[kind] || pathsByKind.image;
    for (const [tagName, attrs] of specs) {
      const child = document.createElementNS(ns, tagName);
      for (const [ak, av] of Object.entries(attrs)) {
        child.setAttribute(ak, av);
      }
      svg.appendChild(child);
    }
    return svg;
  }

  function formatBytes(bytes) {
    const num = Number(bytes || 0);
    if (!Number.isFinite(num) || num <= 0) return "0 B";
    if (num < 1024) return `${num} B`;
    if (num < 1024 * 1024) return `${(num / 1024).toFixed(1)} KB`;
    if (num < 1024 * 1024 * 1024) return `${(num / (1024 * 1024)).toFixed(2)} MB`;
    return `${(num / (1024 * 1024 * 1024)).toFixed(2)} GB`;
  }

  function formatElapsedMmSs(sec) {
    const s = Math.max(0, Math.floor(Number(sec || 0)));
    const mm = String(Math.floor(s / 60)).padStart(2, "0");
    const ss = String(s % 60).padStart(2, "0");
    return `${mm}:${ss}`;
  }

  function formatShortQuantTag(q) {
    const val = String(q || "").toUpperCase();
    if (val.includes("IQ3")) return "IQ3";
    if (val.includes("IQ4")) return "IQ4";
    return "Q4";
  }

  function formatShortCtxTag(ctxVal) {
    const n = Number(ctxVal || 0);
    if (!Number.isFinite(n) || n <= 0) return "";
    if (n >= 262144) return "256K";
    if (n >= 131072) return "128K";
    if (n >= 65536) return "64K";
    if (n >= 32768) return "32K";
    return `${Math.max(4, Math.round(n / 1024))}K`;
  }

  /* ----------------------------------------------------------------
     全局 Toast 提示与二次确认守卫
     ---------------------------------------------------------------- */
  let toastTimer = null;
  function showToast(msg, opts = {}) {
    const box = el("studio-toast");
    if (!box) return;
    const cfg = typeof opts === "number" ? { duration: opts } : (opts || {});
    const text = cleanText(msg);
    if (!text) return;
    clearTimeout(toastTimer);
    box.textContent = text;
    box.classList.toggle("error", Boolean(cfg.isError));
    box.classList.add("show");
    box.setAttribute("aria-hidden", "false");
    toastTimer = setTimeout(() => {
      box.classList.remove("show");
      box.setAttribute("aria-hidden", "true");
    }, Number(cfg.duration || 2600));
  }

  function createArmGuard(defaultMs = 4000) {
    let armedKey = "";
    let armedUntil = 0;
    let timer = null;
    let expireCb = null;

    function disarm() {
      clearTimeout(timer);
      timer = null;
      armedKey = "";
      armedUntil = 0;
      expireCb = null;
    }

    function isArmed(key) {
      if (Date.now() > armedUntil) {
        if (armedUntil > 0) disarm();
        return false;
      }
      return key === undefined ? true : armedKey === String(key);
    }

    function arm(key, onExpire, ms = defaultMs) {
      clearTimeout(timer);
      armedKey = String(key || "default");
      armedUntil = Date.now() + ms;
      expireCb = typeof onExpire === "function" ? onExpire : null;
      timer = setTimeout(() => {
        const cb = expireCb;
        disarm();
        if (cb) cb();
      }, ms);
    }

    return { arm, disarm, isArmed };
  }

  const deployCancelArm = createArmGuard(4000);
  const stopAllArm = createArmGuard(4000);
  const acctDelArm = createArmGuard(4000);
  const cfClearArm = createArmGuard(4000);

  /* ----------------------------------------------------------------
     静态规格目录（与桌宠保持同源数据结构，不含任何 Emoji）
     ---------------------------------------------------------------- */
  const WORKFLOW_CATALOG = [
    {
      id: "anima",
      name: "Anima",
      title: "Anima 二次元生图",
      category: "image",
      iconKind: "image",
      recGpu: "L4",
      gpuPriority: ["L4", "A100", "T4", "G4", "H100"],
      autoHmOnA100: false,
      desc: "自训 7 组角色与画风 LoRA · 约 2.5m 就绪",
      specTail: () => "约 2.5m",
      modules: [],
    },
    {
      id: "minimax",
      name: "MiniMax",
      title: "MiniMax 影视视频",
      category: "video",
      iconKind: "video",
      recGpu: "G4",
      gpuPriority: ["G4", "A100", "H100", "L4", "T4"],
      autoHmOnA100: true,
      desc: "H3 视频工作流 · 支持多图参考与提示词扩写",
      specTail: (st) => {
        const gb = 60 + (st.minimax_r2v ? 21 : 0) + (st.minimax_pe ? 7 : 0);
        return `${!st.minimax_r2v && !st.minimax_pe ? "精简" : "约"}${gb}G`;
      },
      modules: [
        { key: "minimax_r2v", prefKey: "boot_minimax_r2v", label: "R2V参考", sizeHint: "+21G" },
        { key: "minimax_pe", prefKey: "boot_minimax_pe", label: "Qwen扩写", sizeHint: "+7G" },
      ],
    },
    {
      id: "drama3",
      name: "Drama 3",
      title: "Drama 3 语音克隆",
      category: "audio",
      iconKind: "audio",
      recGpu: "A100",
      gpuPriority: ["A100", "G4", "L4", "H100", "T4"],
      autoHmOnA100: false,
      desc: "35B MoE 声音克隆与多角色演戏配音",
      specTail: () => "约 44G",
      modules: [],
    },
    {
      id: "qwen_27b",
      name: "Qwen27B",
      title: "Qwen3.8-27B 多模态对话",
      category: "llm",
      iconKind: "llm",
      recGpu: "L4",
      gpuPriority: ["L4", "A100", "G4", "H100", "T4"],
      autoHmOnA100: false,
      desc: "27B 稠密多模态 · L4 甜点约 16G · 支持与画布异构/同驻",
      specTail: (st) => {
        const qMap = { "UD-Q4_K_XL": 16, "UD-IQ4_XS": 13, "IQ3_XXS": 10 };
        const baseGb = qMap[String(st.llm27_quant)] || 16;
        const totalGb = baseGb + (st.llm27_mmproj ? 1 : 0);
        const ctxTag = formatShortCtxTag(st.llm27_ctx) || "64K";
        return `约 ${totalGb}G · ${ctxTag}`;
      },
      modules: [
        {
          key: "llm27_quant",
          prefKey: "boot_llm27_quant",
          type: "choice",
          options: [
            { val: "UD-Q4_K_XL", label: "Q4 旗舰" },
            { val: "UD-IQ4_XS", label: "IQ4 省显" },
            { val: "IQ3_XXS", label: "IQ3 轻量" },
          ],
        },
        {
          key: "llm27_ctx",
          prefKey: "boot_llm27_ctx",
          type: "choice",
          options: [
            { val: 65536, label: "64K" },
            { val: 131072, label: "128K" },
            { val: 262144, label: "256K" },
          ],
        },
        {
          key: "llm27_mmproj",
          prefKey: "boot_llm27_mmproj",
          type: "toggle",
          label: "视觉识图",
          sizeHint: "+0.9G",
        },
      ],
    },
    {
      id: "qwen_llm",
      name: "Qwen180B",
      title: "Qwen3.8-180B 旗舰无审查",
      category: "llm",
      iconKind: "llm",
      recGpu: "G4",
      gpuPriority: ["G4", "A100", "H100", "L4", "T4"],
      autoHmOnA100: true,
      desc: "180B MoE 多模态（激活 6B）· WebUI + OpenAI 接口",
      specTail: (st) => {
        const qMap = { "UD-Q4_K_XL": 104, "UD-IQ4_XS": 87, "IQ3_XXS": 71 };
        const baseGb = qMap[String(st.llm_quant)] || 104;
        const totalGb = baseGb + (st.llm_mmproj ? 1 : 0);
        const ctxTag = formatShortCtxTag(st.llm_ctx) || "32K";
        return `约 ${totalGb}G · ${ctxTag}`;
      },
      modules: [
        {
          key: "llm_quant",
          prefKey: "boot_llm_quant",
          type: "choice",
          options: [
            { val: "UD-Q4_K_XL", label: "Q4 旗舰" },
            { val: "UD-IQ4_XS", label: "IQ4 省盘" },
            { val: "IQ3_XXS", label: "IQ3 应急" },
          ],
        },
        {
          key: "llm_ctx",
          prefKey: "boot_llm_ctx",
          type: "choice",
          options: [
            { val: 32768, label: "32K" },
            { val: 131072, label: "128K" },
            { val: 262144, label: "256K" },
          ],
        },
        {
          key: "llm_mmproj",
          prefKey: "boot_llm_mmproj",
          type: "toggle",
          label: "视觉识图",
          sizeHint: "+0.9G",
        },
      ],
    },
  ];

  const GPU_ORDER = ["T4", "L4", "A100", "G4", "H100"];
  const GPU_SPECS = {
    T4: { vram: "16G", vramHm: "16G高", ram: "13G内存", ramHm: "51G高存", cu: 1.15, cuHm: 1.31, hmLock: false },
    L4: { vram: "24G高", vramHm: "24G高", ram: "53G高存", ramHm: "53G高存", cu: 1.94, cuHm: 1.94, hmLock: true },
    A100: { vram: "40G", vramHm: "80G高", ram: "83G内存", ramHm: "167G高存", cu: 5.5, cuHm: 11.5, hmLock: false },
    G4: { vram: "96G", vramHm: "96G", ram: "176G高存", ramHm: "176G高存", cu: 10.05, cuHm: 10.05, hmLock: true },
    H100: { vram: "80G", vramHm: "80G高", ram: "167G内存", ramHm: "230G高存", cu: 12.6, cuHm: 14.2, hmLock: false },
  };

  /* 收到后端快照后按 stack_manifest.json 权威目录原地同步显卡顺序与默认费率（保留上方常量作首屏兜底） */
  function applyCatalogFromSnap(snap) {
    const cat = snap && snap.catalog;
    if (!cat || typeof cat !== "object") return;
    if (Array.isArray(cat.gpu_order) && cat.gpu_order.length > 0) {
      const nextOrder = cat.gpu_order.map((g) => String(g || "").toUpperCase()).filter(Boolean);
      if (nextOrder.length > 0) {
        GPU_ORDER.splice(0, GPU_ORDER.length, ...nextOrder);
      }
    }
    const rawSpecs = cat.gpu_specs;
    if (rawSpecs && typeof rawSpecs === "object") {
      for (const [rawKey, item] of Object.entries(rawSpecs)) {
        const g = String(rawKey || "").toUpperCase();
        if (!g || !item || typeof item !== "object") continue;
        const prev = GPU_SPECS[g] || {};
        const stdRate = Number(item.default_rate_std);
        const hmRate = Number(item.default_rate_hm);
        const cuVal = Number.isFinite(stdRate) && stdRate > 0 ? stdRate : (prev.cu || 0);
        const cuHmVal = Number.isFinite(hmRate) && hmRate > 0 ? hmRate : (cuVal || prev.cuHm || 0);
        GPU_SPECS[g] = {
          vram: String(item.vram ?? prev.vram ?? ""),
          vramHm: String(item.vram_hm ?? prev.vramHm ?? item.vram ?? ""),
          ram: String(item.ram ?? prev.ram ?? ""),
          ramHm: String(item.ram_hm ?? prev.ramHm ?? item.ram ?? ""),
          cu: cuVal,
          cuHm: item.hm_locked ? cuVal : cuHmVal,
          note: String(item.note ?? prev.note ?? ""),
          hmLock: Boolean(item.hm_locked),
        };
      }
    }
  }

  /* ----------------------------------------------------------------
     应用全局状态
     ---------------------------------------------------------------- */
  const VIEW_META = {
    overview: {
      title: "算力与集群",
      subtitle: "全池余额总览与异构节点编排",
      primaryLabel: "刷新算力",
    },
    gallery: {
      title: "作品画廊",
      subtitle: "瀑布流归档与 ComfyUI 元数据检视",
      primaryLabel: "拉回成图",
    },
    tuner: {
      title: "引擎与云盘",
      subtitle: "Qwen3.8 在线热调参与云端资产库",
      primaryLabel: "刷新云盘",
    },
    console: {
      title: "控制台与体检",
      subtitle: "三端实时脱敏日志与环境深度诊断",
      primaryLabel: "深度体检",
    },
  };

  let currentView = "overview";
  let latestSnap = null;
  let forceShowBootStandby = false;
  let userTouchedBoot = false;
  let wfCategory = "all";

  const bootState = {
    profile: "anima",
    profiles: ["anima"],
    gpu: "L4",
    high_mem: false,
    minimax_r2v: true,
    minimax_pe: true,
    llm_quant: "UD-Q4_K_XL",
    llm_ctx: 32768,
    llm_mmproj: true,
    llm27_quant: "UD-Q4_K_XL",
    llm27_ctx: 65536,
    llm27_mmproj: true,
    clusterSplitMode: true,
    prefsSynced: false,
  };

  const galleryState = {
    kind: "all",
    allItems: [],
    items: [],
    counts: { all: 0, image: 0, video: 0, audio: 0 },
    folders: [],
    subfolder: "all",
    folderSearch: "",
    dateRange: "all",
    groupBy: "date",
    sortBy: "newest",
    layout: "masonry",
    query: "",
    pageSize: 120,
    visibleLimit: 120,
    openPopover: "",
    lastColCount: 0,
    loaded: false,
    selectedPath: "",
    inspectData: null,
  };

  const tunerState = {
    ctx: 65536,
    quant: "UD-Q4_K_XL",
    mmproj: true,
    touched: false,
    cloudBucket: "loras",
    cloudLoaded: false,
    cloudData: { ok: false, configured: false, message: "点击刷新读取云盘资产", loras: [], workflows: [], refs: [] },
  };

  const consoleState = {
    source: "local_diag",
    filter: "",
    lines: [],
    updatedStr: "--:--:--",
    logPath: "",
    errorMsg: "",
    logsLoaded: false,
    doctorReport: null,
  };

  /* 左侧遥测弧线历史与右侧实时运行日志面板状态 */
  const telemetryHistory = [];
  let telemetrySplitMode = false;
  let tunnelMasked = true;
  let sbAcctMenuOpen = false;
  let _sbUptimeTimer = null;

  const rightLogState = {
    open: true,
    source: "all", // all -> local_diag, comfy -> comfy, llm -> llama
    lines: [],
    liveEvents: [],
    loading: false,
  };

  let privacyMasked = false;
  try {
    privacyMasked = localStorage.getItem("colab_studio_privacy_mask") === "1";
    if (privacyMasked) tunnelMasked = true;
  } catch (_) {
    privacyMasked = false;
  }

  const sbReleaseArm = createArmGuard(4000);

  /* 工具原名到简体中文标题映射表 */
  const AGENT_TOOL_LABELS = {
    get_cluster_and_pool_status: "查看号池与集群状态",
    tail_instance_logs: "读取实例实时日志",
    list_gallery_and_inspect: "检索画廊作品与参数",
    list_gdrive_assets: "查看云盘资产库",
    run_environment_doctor: "执行环境深度体检",
    exec_remote_colab: "远端 Colab 执行指令",
    queue_comfy_workflow: "提交 ComfyUI 生成任务",
    sync_and_pull_outputs: "同步拉回云端生成成果",
    reconfigure_llm_engine: "热重载 Qwen3.8 引擎",
    manage_compute_instance: "算力实例开关卡调度",
  };

  const REASONING_EFFORT_LABELS = {
    auto: "自动",
    none: "关闭",
    minimal: "极简",
    low: "低",
    medium: "中",
    high: "高",
    xhigh: "超高",
    max: "极高",
  };

  const REASONING_EFFORT_FULL_LABELS = {
    auto: "自动自适应",
    none: "已关闭思考",
    minimal: "极简思考",
    low: "低思考",
    medium: "中思考",
    high: "高思考",
    xhigh: "超高思考",
    max: "极高思考",
  };

  const EFFORT_OPTION_META = {
    auto: { title: "自动 (Auto)", desc: "遵循网关与模型默认设置" },
    none: { title: "关闭 (Off)", desc: "快速直答，不展开思考链" },
    minimal: { title: "极简 (Minimal)", desc: "超轻量推理，优先极速响应" },
    low: { title: "低 (Low)", desc: "轻量推演，兼顾响应速度" },
    medium: { title: "中 (Medium)", desc: "日常调度与排错均衡推荐" },
    high: { title: "高 (High)", desc: "深度多步诊断与复杂任务" },
    xhigh: { title: "超高 (Extra)", desc: "Extra-High 深度推演预算" },
    max: { title: "极高 (Max)", desc: "最大推理预算上限" },
  };

  const copilotState = {
    open: false,
    popoverOpen: false,
    effortPopoverOpen: false,
    plusPopoverOpen: false,
    historyPopoverOpen: false,
    loaded: false,
    busy: false,
    driver: "native",
    claudeAvailable: false,
    brainSource: "auto",
    apiFormat: "openai",
    apiBaseUrl: "",
    model: "",
    reasoningEffort: "auto",
    supportedReasoningEfforts: ["none", "low", "medium", "high", "xhigh"],
    reasoningSupported: true,
    defaultReasoningEffort: "",
    reasoningSpecSource: "fallback",
    _syncedCapProvIds: new Set(),
    activeProviderId: "",
    activeProviderName: "",
    providers: [],
    quickFilter: "",
    hasApiKey: false,
    autoApproveBilling: false,
    colabLlmOnline: false,
    resolvedSource: "auto",
    resolvedBrainLabel: "Native · 自动路由",
    activeSessionId: "",
    sessions: [],
    contextStats: { messageCount: 0, maxMessages: 40, approxChars: 0, approxTokens: 0 },
    historyFilter: "",
    stickToBottom: true,
    turnEmittedText: false,
    attachedContext: null,
    items: [],
    pendingConfirm: null,
    activeStreamIdx: -1,
    cfgDraft: {
      driver: "native",
      brainSource: "auto",
      apiFormat: "openai",
      reasoningEffort: "auto",
      autoApproveBilling: false,
    },
  };

  const histClearArm = createArmGuard(3500);

  const PROVIDER_PROTO_LABELS = {
    openai: "OpenAI",
    openai_responses: "Responses",
    anthropic: "Anthropic",
    gemini: "Gemini",
  };

  const provDelArm = createArmGuard(4000);

  const providerModalState = {
    open: false,
    editingId: "",
    isNew: false,
    name: "",
    apiFormat: "openai",
    baseUrl: "",
    probeModel: "",
    modelsText: "",
    selectedModels: [],
    defaultModel: "",
    fetchedModels: [],
    fetchedCapabilities: {},
    modelFilter: "",
    testResults: {},
    hasStoredKey: false,
    showKey: false,
    busyFetch: false,
    busyDetect: false,
    busyTest: false,
    statusMsg: "",
    statusKind: "",
  };

  /* ----------------------------------------------------------------
     工作流与显卡推理辅助
     ---------------------------------------------------------------- */
  function getWorkflowMeta(id) {
    const t = String(id || "anima").toLowerCase();
    return WORKFLOW_CATALOG.find((w) => w.id === t) || WORKFLOW_CATALOG[0];
  }

  function normalizeProfiles(rawList, fallbackId = "anima") {
    const valid = WORKFLOW_CATALOG.map((w) => w.id);
    const list = Array.isArray(rawList)
      ? rawList.map((x) => String(x || "").trim().toLowerCase()).filter((x) => valid.includes(x))
      : [];
    const uniq = [];
    for (const id of list) {
      if (!uniq.includes(id)) uniq.push(id);
    }
    if (uniq.length === 0) {
      const fb = String(fallbackId || "anima").trim().toLowerCase();
      uniq.push(valid.includes(fb) ? fb : "anima");
    }
    if (uniq.includes("qwen_27b") && uniq.includes("qwen_llm")) {
      const i27 = uniq.indexOf("qwen_27b");
      const i180 = uniq.indexOf("qwen_llm");
      uniq.splice(i27 < i180 ? i27 : i180, 1);
    }
    return valid.filter((id) => uniq.includes(id));
  }

  function getActiveLlmProfile(profiles) {
    const list = Array.isArray(profiles) ? profiles : [];
    if (list.includes("qwen_llm")) return "qwen_llm";
    if (list.includes("qwen_27b")) return "qwen_27b";
    return null;
  }

  function isDualProfileSet(profiles) {
    const list = Array.isArray(profiles) ? profiles : [];
    const hasLlm = Boolean(getActiveLlmProfile(list));
    const hasComfy = list.some((p) => ["anima", "minimax", "drama3"].includes(p));
    return hasLlm && hasComfy;
  }

  function resolveGpuRate(gpu, highMem, snap) {
    const g = String(gpu || "L4").toUpperCase();
    const spec = GPU_SPECS[g] || GPU_SPECS.L4;
    const hm = !spec.hmLock && Boolean(highMem);
    const rkey = spec.hmLock ? `${g}:STD` : `${g}:${hm ? "HM" : "STD"}`;
    const measuredMap = (snap && snap.measured_gpu_rates) || {};
    const measuredVal = Number(measuredMap[rkey]);
    if (Number.isFinite(measuredVal) && measuredVal > 0.02) {
      return {
        rate: measuredVal,
        measured: true,
        shortLabel: `${measuredVal.toFixed(2)}/h`,
        badgeLabel: `实测 ${measuredVal.toFixed(2)} CU/h`,
      };
    }
    const est = hm ? spec.cuHm : spec.cu;
    const estShort = est >= 10 ? est.toFixed(1) : est.toFixed(2).replace(/0$/, "");
    return {
      rate: est,
      measured: false,
      shortLabel: `~${estShort}/h`,
      badgeLabel: `预估 ~${estShort} CU/h`,
    };
  }

  function pickBestGpuForProfiles(profiles, snap) {
    const list = normalizeProfiles(profiles, bootState.profile);
    const eligible = snap && Array.isArray(snap.eligible_gpus) ? snap.eligible_gpus : null;
    const scarce = (snap && snap.scarce_gpus) || {};
    const wf = getWorkflowMeta(list[0]);
    let order;
    let recFallback = wf.recGpu || "L4";
    if (list.length <= 1) {
      const raw = Array.isArray(wf.gpuPriority) ? wf.gpuPriority : GPU_ORDER;
      order = wf.id === "qwen_llm" ? raw.filter((g) => g !== "T4" && g !== "L4") : raw;
    } else if (isDualProfileSet(list) || list.includes("qwen_llm")) {
      order = ["G4", "A100", "H100"];
      recFallback = "G4";
    } else if (list.includes("minimax") || list.includes("drama3")) {
      order = ["G4", "A100", "H100", "L4", "T4"];
      recFallback = "G4";
    } else {
      order = ["L4", "A100", "T4", "G4", "H100"];
    }
    for (const g of order) {
      if (scarce[g]) continue;
      if (eligible && eligible.length > 0 && !eligible.includes(g)) continue;
      return g;
    }
    return recFallback;
  }

  function resolveSplitPlanPreview(profs, snap) {
    const s = snap || latestSnap || {};
    const dualStack = isDualProfileSet(profs);
    const cluster = s.cluster || {};
    const canSplit = Boolean(dualStack && cluster.can_split);
    const useSplit = Boolean(canSplit && bootState.clusterSplitMode);
    if (!useSplit) {
      return { canSplit, useSplit: false, llmGpu: "", comfyGpu: "", llmRate: null, comfyRate: null, totalRate: 0 };
    }
    const llmProf = getActiveLlmProfile(profs);
    const comfyProfs = profs.filter((p) => ["anima", "minimax", "drama3"].includes(p));
    const llmGpu = String(cluster.preview_llm_gpu || (llmProf === "qwen_llm" ? "G4" : "L4")).toUpperCase();
    const defComfy = (comfyProfs.includes("minimax") || comfyProfs.includes("drama3")) ? "G4" : "L4";
    const comfyGpu = String(cluster.preview_comfy_gpu || defComfy).toUpperCase();
    const llmRate = resolveGpuRate(llmGpu, false, s);
    const comfyRate = resolveGpuRate(comfyGpu, false, s);
    return {
      canSplit,
      useSplit: true,
      llmGpu,
      comfyGpu,
      llmRate,
      comfyRate,
      totalRate: Number((llmRate.rate + comfyRate.rate).toFixed(2)),
    };
  }

  /* ----------------------------------------------------------------
     主题切换与持久化（light / dark / light dark）
     ---------------------------------------------------------------- */
  function getEffectiveScheme() {
    return document.documentElement.dataset.theme || "light";
  }

  function setStudioTheme(scheme) {
    const next = (scheme === "dark" || scheme === "light" || scheme === "light dark") ? scheme : "light";
    document.documentElement.style.colorScheme = next;
    document.documentElement.dataset.theme = next;
    try {
      localStorage.setItem("studio-color-scheme", next);
    } catch (_) {
      /* 忽略受限环境存储异常 */
    }
    const btn = el("btn-theme-toggle");
    if (btn) {
      const isDark = next === "dark" || (next === "light dark" && window.matchMedia("(prefers-color-scheme: dark)").matches);
      btn.title = isDark ? "切换为浅色星晶白瓷主题" : "切换为深空暗房主题";
    }
  }

  function toggleTheme() {
    const cur = getEffectiveScheme();
    const curIsDark = cur === "dark" || (cur === "light dark" && window.matchMedia("(prefers-color-scheme: dark)").matches);
    setStudioTheme(curIsDark ? "light" : "dark");
  }

  window.setStudioTheme = setStudioTheme;

  /* ----------------------------------------------------------------
     无边框窗体缩放、拖拽与交通灯控制
     ---------------------------------------------------------------- */
  function setupWindowChrome() {
    document.querySelectorAll(".traffic-lights .tl-btn").forEach((btn) => {
      btn.addEventListener("mousedown", (e) => {
        e.stopPropagation();
      });
    });

    if (el("btn-win-close")) {
      el("btn-win-close").addEventListener("click", async (e) => {
        e.stopPropagation();
        await apiCall("studio_window_ctl", "close");
      });
    }

    if (el("btn-win-min")) {
      el("btn-win-min").addEventListener("click", async (e) => {
        e.stopPropagation();
        await apiCall("studio_window_ctl", "minimize");
      });
    }

    const syncMaxState = async () => {
      const res = await apiCall("studio_window_ctl", "toggle_maximize");
      const maxFlag = Boolean(res && res.maximized);
      document.body.classList.toggle("is-maximized", maxFlag);
      const shell = el("studio-shell");
      if (shell) shell.classList.toggle("is-maximized", maxFlag);
    };

    if (el("btn-win-max")) {
      el("btn-win-max").addEventListener("click", async (e) => {
        e.stopPropagation();
        await syncMaxState();
      });
    }

    const bindDragRegion = (node) => {
      if (!node) return;
      let armed = false;
      let startX = 0;
      let startY = 0;

      node.addEventListener("mousedown", (e) => {
        if (e.button !== 0 || e.target.closest("button, input, select, textarea, a, .traffic-lights")) return;
        armed = true;
        startX = e.screenX;
        startY = e.screenY;
      });

      window.addEventListener("mousemove", (e) => {
        if (!armed) return;
        if (Math.hypot(e.screenX - startX, e.screenY - startY) > 3) {
          armed = false;
          apiCall("begin_studio_drag");
        }
      });

      window.addEventListener("mouseup", () => {
        armed = false;
      });
    };

    bindDragRegion(el("sidebar-drag"));
    bindDragRegion(el("titlebar-drag"));
    bindDragRegion(el("log-panel-drag"));

    if (el("titlebar-drag")) {
      el("titlebar-drag").addEventListener("dblclick", async (e) => {
        if (e.target.closest("button, input")) return;
        await syncMaxState();
      });
    }

    document.querySelectorAll(".resize-edge[data-resize]").forEach((edgeEl) => {
      edgeEl.addEventListener("mousedown", (e) => {
        if (e.button !== 0) return;
        e.preventDefault();
        e.stopPropagation();
        apiCall("begin_studio_resize", edgeEl.dataset.resize);
      });
    });
  }

  /* ----------------------------------------------------------------
     View Transitions 丝滑视图切换
     ---------------------------------------------------------------- */
  function applyViewSwitch(viewName) {
    const target = VIEW_META[viewName] ? viewName : "overview";
    currentView = target;

    document.querySelectorAll(".sidebar-nav .nav-item").forEach((btn) => {
      const isMatch = btn.dataset.nav === target;
      btn.classList.toggle("active", isMatch);
      btn.setAttribute("aria-selected", String(isMatch));
    });

    document.querySelectorAll(".view-pane").forEach((pane) => {
      pane.classList.toggle("active", pane.dataset.view === target);
    });

    const meta = VIEW_META[target];
    if (el("view-title")) el("view-title").textContent = meta.title;
    if (el("view-subtitle")) el("view-subtitle").textContent = meta.subtitle;
    if (el("btn-title-primary-label")) el("btn-title-primary-label").textContent = meta.primaryLabel;

    if (target === "gallery") {
      if (!galleryState.loaded) {
        loadGalleryItems();
      } else {
        requestAnimationFrame(() => syncGalleryResponsiveCols());
      }
    } else if (target === "tuner" && !tunerState.cloudLoaded) {
      loadCloudAssets(false);
    } else if (target === "console") {
      if (!consoleState.logsLoaded) loadLiveLogs();
      if (!consoleState.doctorReport) runDoctorPreflight(false);
    }
    renderCopilotChips();
  }

  function switchStudioTab(viewName) {
    const target = VIEW_META[viewName] ? viewName : "overview";
    if (target === currentView) return;
    const reduceMotion = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (!reduceMotion && typeof document.startViewTransition === "function") {
      document.startViewTransition(() => {
        applyViewSwitch(target);
      });
    } else {
      applyViewSwitch(target);
    }
  }

  window.switchStudioTab = switchStudioTab;

  /* ----------------------------------------------------------------
     视图 1：算力与集群渲染与交互
     ---------------------------------------------------------------- */
  function syncBootPrefsFromSnap(snap) {
    if (!snap || !snap.prefs) return;
    const prefs = snap.prefs;
    const isBusy = Boolean(snap.is_running || (snap.deploy && snap.deploy.active));
    if (bootState.prefsSynced && (userTouchedBoot || isBusy)) return;

    bootState.prefsSynced = true;
    const fbProf = prefs.boot_profile && WORKFLOW_CATALOG.some((w) => w.id === String(prefs.boot_profile).toLowerCase())
      ? String(prefs.boot_profile).toLowerCase()
      : "anima";
    const initProfs = Array.isArray(prefs.boot_profiles) && prefs.boot_profiles.length > 0
      ? normalizeProfiles(prefs.boot_profiles, fbProf)
      : [fbProf];
    bootState.profiles = initProfs;
    bootState.profile = initProfs[0];
    if (GPU_ORDER.includes(String(prefs.boot_gpu || "").toUpperCase())) {
      bootState.gpu = String(prefs.boot_gpu).toUpperCase();
    }
    const spec = GPU_SPECS[bootState.gpu] || {};
    bootState.high_mem = spec.hmLock ? false : Boolean(prefs.boot_high_mem);
    bootState.minimax_r2v = prefs.boot_minimax_r2v !== false;
    bootState.minimax_pe = prefs.boot_minimax_pe !== false;
    if (["UD-Q4_K_XL", "UD-IQ4_XS", "IQ3_XXS"].includes(String(prefs.boot_llm_quant || ""))) {
      bootState.llm_quant = String(prefs.boot_llm_quant);
    }
    if ([32768, 131072, 262144].includes(Number(prefs.boot_llm_ctx))) {
      bootState.llm_ctx = Number(prefs.boot_llm_ctx);
    }
    bootState.llm_mmproj = prefs.boot_llm_mmproj !== false;
    if (["UD-Q4_K_XL", "UD-IQ4_XS", "IQ3_XXS"].includes(String(prefs.boot_llm27_quant || ""))) {
      bootState.llm27_quant = String(prefs.boot_llm27_quant);
    }
    if ([65536, 131072, 262144].includes(Number(prefs.boot_llm27_ctx))) {
      bootState.llm27_ctx = Number(prefs.boot_llm27_ctx);
    }
    bootState.llm27_mmproj = prefs.boot_llm27_mmproj !== false;
    bootState.clusterSplitMode = prefs.cluster_split_mode !== false;
  }

  function resolveAccountTierInfo(acct, isCur, snap) {
    const s = snap || {};
    const rawTier = String((acct && acct.tier) || "").trim().toLowerCase();
    const labelMap = { ultra: "Ultra", pro_plus: "Pro+", pro: "Pro", free: "Free" };
    const clsMap = { ultra: "accent", pro_plus: "tier-pro-plus", pro: "tier-pro", free: "warn" };
    if (labelMap[rawTier]) {
      return { tier: rawTier, label: String((acct && acct.tier_label) || labelMap[rawTier]), pillCls: clsMap[rawTier] };
    }
    const bal = isCur && typeof s.balance === "number" ? s.balance : (acct && acct.last_balance);
    const numBal = (typeof bal === "number" && !Number.isNaN(bal)) ? bal : null;
    const rawElig = Array.isArray(acct && acct.eligible_gpus)
      ? acct.eligible_gpus
      : (isCur && Array.isArray(s.eligible_gpus) ? s.eligible_gpus : []);
    const eg = rawElig.map((g) => String(g || "").toUpperCase());
    const ig = Array.isArray(acct && acct.ineligible_gpus)
      ? acct.ineligible_gpus.map((g) => String(g || "").toUpperCase())
      : [];
    const hasH100 = eg.includes("H100") && !ig.includes("H100");
    if ((numBal !== null && numBal > 550) || (hasH100 && (numBal === null || numBal > 0))) {
      return { tier: "ultra", label: "Ultra", pillCls: "accent" };
    }
    if (numBal !== null && numBal > 250) {
      return { tier: "pro_plus", label: "Pro+", pillCls: "tier-pro-plus" };
    }
    if ((numBal !== null && numBal > 0) || eg.some((g) => ["G4", "A100", "L4"].includes(g))) {
      return { tier: "pro", label: "Pro", pillCls: "tier-pro" };
    }
    if (eg.length > 0 && numBal !== null && numBal <= 0) {
      return { tier: "free", label: "Free", pillCls: "warn" };
    }
    return { tier: "unknown", label: "Colab", pillCls: "" };
  }

  function formatOverviewPoolTierSummary(accounts, activeId, snap) {
    const labelMap = { ultra: "Ultra", pro_plus: "Pro+", pro: "Pro", free: "Free", unknown: "Colab" };
    const counts = {};
    for (const acct of accounts) {
      const aid = String((acct && acct.id) || "");
      const isCur = Boolean(aid && aid === activeId && !(snap && snap.onboarding && snap.onboarding.auth_missing));
      const info = resolveAccountTierInfo(acct, isCur, snap);
      counts[info.tier] = (counts[info.tier] || 0) + 1;
    }
    const keys = Object.keys(counts);
    if (keys.length === 0) return "暂无已绑定账号";
    if (keys.length === 1) {
      const k = keys[0];
      return `${counts[k]} 个 ${labelMap[k] || "Colab"} 账号`;
    }
    const order = ["ultra", "pro_plus", "pro", "free", "unknown"];
    const parts = order.filter((k) => counts[k] > 0).map((k) => `${counts[k]} ${labelMap[k]}`);
    return parts.join(parts.length <= 2 ? " + " : " · ");
  }

  function renderSidebarStatus(snap) {
    const s = snap || {};
    const ob = s.onboarding || {};
    const deploy = s.deploy || {};
    const comfy = s.comfy || {};
    const dot = el("sb-status-dot");
    const acctEl = el("sb-acct-name");
    const txtEl = el("sb-status-text");
    const timeEl = el("sb-updated-at");
    const ovPulse = el("nav-dot-overview");
    const conPulse = el("nav-dot-console");

    const acctLabel = ob.active_account_label
      || (ob.active_account && ob.active_account.label)
      || (ob.auth_missing ? "待绑定账号" : "Colab 主号");
    if (acctEl) acctEl.textContent = cleanText(acctLabel);

    let level = "idle";
    let statusTxt = cleanText(s.status_label || "待机省积分");
    if (deploy.active) {
      level = "running";
      statusTxt = `部署中 ${Math.round(Number(deploy.pct || 0))}%`;
    } else if (comfy.has_error || !s.ok) {
      level = "danger";
    } else if (s.is_running || (s.cluster && s.cluster.active)) {
      level = "running";
    } else if (ob.auth_missing) {
      level = "warn";
    }

    if (dot) dot.className = `status-dot ${level}`;
    if (txtEl) txtEl.textContent = statusTxt;
    if (timeEl) timeEl.textContent = s.updated_str || "--:--:--";

    if (ovPulse) {
      ovPulse.className = "nav-pulse";
      if (deploy.active || s.is_running || (s.cluster && s.cluster.active)) ovPulse.classList.add("live");
      else if (comfy.has_error) ovPulse.classList.add("err");
    }
    if (conPulse) {
      conPulse.className = "nav-pulse";
      if (comfy.has_error) conPulse.classList.add("err");
      else if (ob.needs_attention) conPulse.classList.add("warn");
    }
  }

  /* ----------------------------------------------------------------
     EasyColab 三栏左卡控制区、显存弧线与右侧实时日志面板
     ---------------------------------------------------------------- */
  function syncRightPanelsLayout() {
    const shell = el("studio-shell");
    const panel = el("studio-log-panel");
    const btn = el("btn-log-panel-toggle");
    const cardsScroll = el("sidebar-cards-scroll");
    const narrowLogHidden = window.innerWidth <= 1180;
    const narrowCardsHidden = window.innerWidth <= 1080;
    const logVisible = Boolean(rightLogState.open && !copilotState.open && !narrowLogHidden);

    if (shell) {
      shell.classList.toggle("log-panel-collapsed", !rightLogState.open);
    }
    if (btn) {
      btn.classList.toggle("is-active", logVisible);
      btn.setAttribute("aria-expanded", String(logVisible));
    }
    if (panel) {
      panel.setAttribute("aria-hidden", String(!logVisible));
      if (logVisible) {
        panel.removeAttribute("inert");
        panel.inert = false;
      } else {
        if (panel.contains(document.activeElement) && document.activeElement.blur) {
          document.activeElement.blur();
        }
        panel.setAttribute("inert", "");
        panel.inert = true;
      }
    }
    if (cardsScroll) {
      cardsScroll.setAttribute("aria-hidden", String(narrowCardsHidden));
      if (narrowCardsHidden) {
        cardsScroll.setAttribute("inert", "");
        cardsScroll.inert = true;
      } else {
        cardsScroll.removeAttribute("inert");
        cardsScroll.inert = false;
      }
    }
  }

  function applyPrivacyMask() {
    document.body.classList.toggle("privacy-masked", Boolean(privacyMasked));
    const btn = el("btn-privacy-mask");
    if (btn) {
      btn.classList.toggle("is-active", Boolean(privacyMasked));
      btn.setAttribute("aria-pressed", String(Boolean(privacyMasked)));
      const sp = btn.querySelector("span");
      if (sp) sp.textContent = privacyMasked ? "已打码" : "打码";
    }
    try {
      localStorage.setItem("colab_studio_privacy_mask", privacyMasked ? "1" : "0");
    } catch (_) {
      /* 忽略存储受限环境异常 */
    }
  }

  function buildSmoothSparklinePaths(values, width = 284, height = 88, padY = 5) {
    const raw = Array.isArray(values) && values.length > 0 ? values : [0, 0];
    const ptsRaw = raw.length === 1 ? [raw[0], raw[0]] : raw;
    const n = ptsRaw.length;
    const usableH = Math.max(8, height - padY * 2);
    const coords = ptsRaw.map((val, idx) => {
      const v = Math.max(0, Math.min(100, Number(val) || 0));
      const x = n <= 1 ? 0 : (idx / (n - 1)) * width;
      const y = height - padY - (v / 100) * usableH;
      return { x, y };
    });

    // 采用 Catmull-Rom 转三次贝塞尔曲线平滑采样轨迹；平坦段直线连接防止过冲
    let d = `M ${coords[0].x.toFixed(1)} ${coords[0].y.toFixed(1)}`;
    for (let i = 0; i < coords.length - 1; i += 1) {
      const p0 = coords[Math.max(0, i - 1)];
      const p1 = coords[i];
      const p2 = coords[i + 1];
      const p3 = coords[Math.min(coords.length - 1, i + 2)];
      if (Math.abs(p2.y - p1.y) < 0.25) {
        d += ` L ${p2.x.toFixed(1)} ${p2.y.toFixed(1)}`;
      } else {
        const cp1x = p1.x + (p2.x - p0.x) / 6;
        const cp1y = Math.max(padY, Math.min(height - padY, p1.y + (p2.y - p0.y) / 6));
        const cp2x = p2.x - (p3.x - p1.x) / 6;
        const cp2y = Math.max(padY, Math.min(height - padY, p2.y - (p3.y - p1.y) / 6));
        d += ` C ${cp1x.toFixed(1)} ${cp1y.toFixed(1)}, ${cp2x.toFixed(1)} ${cp2y.toFixed(1)}, ${p2.x.toFixed(1)} ${p2.y.toFixed(1)}`;
      }
    }
    const last = coords[coords.length - 1];
    const areaPath = `${d} L ${last.x.toFixed(1)} ${height} L ${coords[0].x.toFixed(1)} ${height} Z`;
    return {
      linePath: d,
      areaPath,
      lastX: last.x.toFixed(1),
      lastY: last.y.toFixed(1),
    };
  }

  function _calcMetricPct(usedVal, totalVal, rawPct) {
    const used = Math.max(0, Number(usedVal || 0));
    const total = Math.max(0, Number(totalVal || 0));
    let pct = Number(rawPct || 0);
    if (total > 0 && pct <= 0) {
      pct = (used / total) * 100;
    }
    if (total <= 0) return 0;
    return Math.max(0, Math.min(100, Number.isFinite(pct) ? pct : 0));
  }

  function _createSplitMetricSvg(samples, colorHex, gradId) {
    const ns = "http://www.w3.org/2000/svg";
    const svg = document.createElementNS(ns, "svg");
    svg.setAttribute("class", "sb-split-svg");
    svg.setAttribute("viewBox", "0 0 260 36");
    svg.setAttribute("preserveAspectRatio", "none");
    svg.setAttribute("aria-hidden", "true");

    const defs = document.createElementNS(ns, "defs");
    const lg = document.createElementNS(ns, "linearGradient");
    lg.setAttribute("id", gradId);
    lg.setAttribute("x1", "0");
    lg.setAttribute("y1", "0");
    lg.setAttribute("x2", "0");
    lg.setAttribute("y2", "1");
    const st1 = document.createElementNS(ns, "stop");
    st1.setAttribute("offset", "0%");
    st1.setAttribute("stop-color", colorHex);
    st1.setAttribute("stop-opacity", "0.28");
    const st2 = document.createElementNS(ns, "stop");
    st2.setAttribute("offset", "100%");
    st2.setAttribute("stop-color", colorHex);
    st2.setAttribute("stop-opacity", "0.02");
    lg.appendChild(st1);
    lg.appendChild(st2);
    defs.appendChild(lg);
    svg.appendChild(defs);

    const grid = document.createElementNS(ns, "g");
    grid.setAttribute("stroke", "currentColor");
    grid.setAttribute("stroke-width", "0.65");
    grid.setAttribute("stroke-dasharray", "2.5 2.5");
    grid.setAttribute("opacity", "0.2");
    for (const gy of [9, 18, 27]) {
      const ln = document.createElementNS(ns, "line");
      ln.setAttribute("x1", "0");
      ln.setAttribute("y1", String(gy));
      ln.setAttribute("x2", "260");
      ln.setAttribute("y2", String(gy));
      grid.appendChild(ln);
    }
    svg.appendChild(grid);

    const { linePath, areaPath, lastX, lastY } = buildSmoothSparklinePaths(samples, 260, 36, 4);
    const pArea = document.createElementNS(ns, "path");
    pArea.setAttribute("d", areaPath);
    pArea.setAttribute("fill", `url(#${gradId})`);
    svg.appendChild(pArea);

    const pLine = document.createElementNS(ns, "path");
    pLine.setAttribute("d", linePath);
    pLine.setAttribute("fill", "none");
    pLine.setAttribute("stroke", colorHex);
    pLine.setAttribute("stroke-width", "1.65");
    pLine.setAttribute("stroke-linecap", "round");
    pLine.setAttribute("stroke-linejoin", "round");
    svg.appendChild(pLine);

    const dot = document.createElementNS(ns, "circle");
    dot.setAttribute("cx", lastX);
    dot.setAttribute("cy", lastY);
    dot.setAttribute("r", "2.4");
    dot.setAttribute("fill", colorHex);
    svg.appendChild(dot);

    return svg;
  }

  function renderTelemetrySparkline(snap) {
    const s = snap || latestSnap || {};
    const comfy = s.comfy || {};

    const vramUsed = Math.max(0, Number(comfy.vram_used_gb || 0));
    const vramTotal = Math.max(0, Number(comfy.vram_total_gb || 0));
    const vramFree = vramTotal > 0 ? Math.max(0, vramTotal - vramUsed) : 0;
    const vramPct = _calcMetricPct(vramUsed, vramTotal, comfy.vram_pct);

    const ramUsed = Math.max(0, Number(comfy.ram_used_gb || 0));
    const ramTotal = Math.max(0, Number(comfy.ram_total_gb || 0));
    const ramFree = Number.isFinite(Number(comfy.ram_free_gb)) && Number(comfy.ram_free_gb) > 0
      ? Number(comfy.ram_free_gb)
      : (ramTotal > 0 ? Math.max(0, ramTotal - ramUsed) : 0);
    const ramPct = _calcMetricPct(ramUsed, ramTotal, comfy.ram_pct);

    const diskUsed = Math.max(0, Number(comfy.disk_used_gb || 0));
    const diskTotal = Math.max(0, Number(comfy.disk_total_gb || 0));
    const diskFree = Number.isFinite(Number(comfy.disk_free_gb)) && Number(comfy.disk_free_gb) > 0
      ? Number(comfy.disk_free_gb)
      : (diskTotal > 0 ? Math.max(0, diskTotal - diskUsed) : 0);
    const diskPct = _calcMetricPct(diskUsed, diskTotal, comfy.disk_pct);

    const nowMs = Date.now();
    const lastSample = telemetryHistory[telemetryHistory.length - 1];
    const diffLarge = !lastSample
      || Math.abs(lastSample.vramPct - vramPct) >= 0.3
      || Math.abs(lastSample.ramPct - ramPct) >= 0.3
      || Math.abs(lastSample.diskPct - diskPct) >= 0.3;
    if (diffLarge || (nowMs - lastSample.ts) >= 2500) {
      telemetryHistory.push({ ts: nowMs, vramPct, ramPct, diskPct });
      if (telemetryHistory.length > 28) telemetryHistory.shift();
    }

    const vSeries = telemetryHistory.length >= 2 ? telemetryHistory.map((i) => i.vramPct) : [vramPct, vramPct];
    const rSeries = telemetryHistory.length >= 2 ? telemetryHistory.map((i) => i.ramPct) : [ramPct, ramPct];
    const dSeries = telemetryHistory.length >= 2 ? telemetryHistory.map((i) => i.diskPct) : [diskPct, diskPct];

    // 同屏 vs 分项视图切换
    const combBox = el("sb-tele-combined");
    const splitBox = el("sb-tele-split");
    const modeLbl = el("sb-tele-mode-lbl");
    if (modeLbl) modeLbl.textContent = telemetrySplitMode ? "同屏" : "分项";
    if (combBox) combBox.classList.toggle("hidden", telemetrySplitMode);
    if (splitBox) splitBox.classList.toggle("hidden", !telemetrySplitMode);

    // 填充同屏三线与 3 行图例
    const vBuilt = buildSmoothSparklinePaths(vSeries, 284, 88, 5);
    const rBuilt = buildSmoothSparklinePaths(rSeries, 284, 88, 5);
    const dBuilt = buildSmoothSparklinePaths(dSeries, 284, 88, 5);

    const setLineAndDot = (lineId, dotId, areaId, built, visible) => {
      const lEl = el(lineId);
      const dEl = el(dotId);
      const aEl = areaId ? el(areaId) : null;
      if (lEl) {
        lEl.setAttribute("d", built.linePath);
        lEl.style.display = visible ? "" : "none";
      }
      if (dEl) {
        dEl.setAttribute("cx", built.lastX);
        dEl.setAttribute("cy", built.lastY);
        dEl.style.display = visible ? "" : "none";
      }
      if (aEl) {
        aEl.setAttribute("d", built.areaPath);
        aEl.style.display = visible ? "" : "none";
      }
    };
    setLineAndDot("sb-vram-line", "sb-vram-dot", "sb-vram-area", vBuilt, vramTotal > 0);
    setLineAndDot("sb-ram-line", "sb-ram-dot", null, rBuilt, ramTotal > 0);
    setLineAndDot("sb-disk-line", "sb-disk-dot", null, dBuilt, diskTotal > 0);

    if (el("sb-leg-vram-gb")) el("sb-leg-vram-gb").textContent = vramTotal > 0 ? `${vramUsed.toFixed(1)} / ${vramTotal.toFixed(1)} GB` : "-- / -- GB";
    if (el("sb-leg-vram-pct")) el("sb-leg-vram-pct").textContent = vramTotal > 0 ? `${Math.round(vramPct)}%` : "--%";
    if (el("sb-leg-ram-gb")) el("sb-leg-ram-gb").textContent = ramTotal > 0 ? `${ramUsed.toFixed(1)} / ${ramTotal.toFixed(1)} GB` : "-- / -- GB";
    if (el("sb-leg-ram-pct")) el("sb-leg-ram-pct").textContent = ramTotal > 0 ? `${Math.round(ramPct)}%` : "--%";
    if (el("sb-leg-disk-gb")) el("sb-leg-disk-gb").textContent = diskTotal > 0 ? `${diskUsed.toFixed(1)} / ${diskTotal.toFixed(1)} GB` : "-- / -- GB";
    if (el("sb-leg-disk-pct")) el("sb-leg-disk-pct").textContent = diskTotal > 0 ? `${Math.round(diskPct)}%` : "--%";

    // 分项视图：渲染显存 / 内存 / 硬盘独立折线卡
    if (splitBox && telemetrySplitMode) {
      splitBox.innerHTML = "";
      const cards = [
        { key: "vram", label: "显存 VRAM", color: "#5487FA", used: vramUsed, free: vramFree, total: vramTotal, pct: vramPct, series: vSeries },
        { key: "ram", label: "内存 RAM", color: "#1AC287", used: ramUsed, free: ramFree, total: ramTotal, pct: ramPct, series: rSeries },
        { key: "disk", label: "硬盘 Disk", color: "#0DBAE0", used: diskUsed, free: diskFree, total: diskTotal, pct: diskPct, series: dSeries },
      ];
      for (const item of cards) {
        const pctTxt = item.total > 0 ? `${Math.round(item.pct)}%` : "--%";
        const detailTxt = item.total > 0
          ? `${item.used.toFixed(1)}/${item.total.toFixed(1)}G · 余${item.free.toFixed(1)}G`
          : "等待采样";
        const svgWrap = h("div", { cls: "sb-split-svg-box" });
        svgWrap.appendChild(_createSplitMetricSvg(item.series, item.color, `sbSplitGrad_${item.key}`));
        splitBox.appendChild(h("div", { cls: "sb-split-item" },
          h("div", { cls: "sb-split-head" },
            h("div", { cls: "sb-split-title-wrap" },
              h("span", { cls: "sb-split-dot", style: `background:${item.color}` }),
              h("span", { cls: "sb-split-title", txt: item.label }),
              h("span", {
                cls: "sb-split-pct-pill",
                style: `color:${item.color};background:${item.color}1F`,
                txt: pctTxt,
              })
            ),
            h("span", { cls: "sb-split-detail", txt: detailTxt })
          ),
          svgWrap
        ));
      }
    }

    // 底部内存盘与云盘可用微文案（兼容 onboarding 统计的 rclone 云盘空间）
    const ob = s.onboarding || {};
    const shmFree = Number(comfy.shm_free_gb || 0);
    const shmTotal = Number(comfy.shm_total_gb || 0);
    const driveFree = Number(comfy.drive_free_gb || ob.gdrive_free_gb || 0);
    const driveTotal = Number(comfy.drive_total_gb || ob.gdrive_total_gb || 0);
    if (el("sb-shm-text")) {
      el("sb-shm-text").textContent = shmTotal >= 1.0
        ? `内存盘可用 ${shmFree.toFixed(1)}/${shmTotal.toFixed(0)}G`
        : "内存盘可用 --";
    }
    if (el("sb-drive-free-text")) {
      el("sb-drive-free-text").textContent = (driveTotal > 0 || driveFree > 0)
        ? `云盘可用 ${driveFree.toFixed(1)}G`
        : "";
    }
  }

  function _formatDurationHhMmSs(totalSec) {
    const sec = Math.max(0, Math.floor(Number(totalSec) || 0));
    const hh = String(Math.floor(sec / 3600)).padStart(2, "0");
    const mm = String(Math.floor((sec % 3600) / 60)).padStart(2, "0");
    const ss = String(sec % 60).padStart(2, "0");
    return `${hh}:${mm}:${ss}`;
  }

  function updateSidebarUptimeTicker() {
    const upEl = el("sb-kpi-uptime");
    if (!upEl) return;
    const s = latestSnap || {};
    const cluster = s.cluster || {};
    const isLive = Boolean(s.is_running || cluster.active);
    if (!isLive) {
      upEl.textContent = "00:00:00";
      return;
    }
    const sinceTs = Number(s.active_since_ts || 0);
    if (sinceTs > 0) {
      const elapsed = Math.max(0, Math.floor(Date.now() / 1000) - sinceTs);
      upEl.textContent = _formatDurationHhMmSs(elapsed);
      return;
    }
    const activeMins = Number(s.active_minutes || 0);
    upEl.textContent = activeMins > 0 ? _formatDurationHhMmSs(activeMins * 60) : "00:00:00";
  }

  function _setAcctPopoverOpen(openFlag) {
    sbAcctMenuOpen = Boolean(openFlag);
    const pop = el("sb-acct-popover");
    const btn = el("btn-sb-acct-menu");
    if (btn) btn.setAttribute("aria-expanded", String(sbAcctMenuOpen));
    if (pop) {
      pop.classList.toggle("hidden", !sbAcctMenuOpen);
      pop.setAttribute("aria-hidden", String(!sbAcctMenuOpen));
    }
  }

  function renderAccountSwitcherPopover(snap) {
    const pop = el("sb-acct-popover");
    if (!pop) return;
    const s = snap || latestSnap || {};
    const ob = s.onboarding || {};
    const accounts = Array.isArray(ob.accounts) ? ob.accounts : [];
    const activeId = String(ob.active_account_id || "");

    pop.innerHTML = "";
    if (accounts.length === 0) {
      pop.appendChild(h("div", {
        cls: "sb-acct-opt",
        txt: "暂无可切换的已绑定账号",
      }));
    } else {
      for (const acct of accounts) {
        const aid = String((acct && acct.id) || "");
        if (!aid) continue;
        const isAct = aid === activeId;
        const lbl = cleanText(acct.email_masked || acct.label || aid);
        const cuVal = typeof acct.last_balance === "number" ? `${acct.last_balance.toFixed(1)} CU` : "--";
        pop.appendChild(h("button", {
          type: "button",
          cls: `sb-acct-opt${isAct ? " is-active" : ""}`,
          role: "menuitem",
          on: {
            click: async () => {
              _setAcctPopoverOpen(false);
              if (isAct) return;
              const res = await apiCall("switch_colab_account", aid);
              if (res && res.snapshot) window.renderStudioState(res.snapshot);
              if (res && res.message) showToast(res.message, { isError: res.ok === false });
            },
          },
        },
          h("span", { cls: "privacy-sensitive", txt: lbl }),
          h("span", { cls: "sb-acct-opt-meta", txt: isAct ? `当前 · ${cuVal}` : cuVal })
        ));
      }
    }

    pop.appendChild(h("div", { cls: "sb-acct-sep" }));
    pop.appendChild(h("button", {
      type: "button",
      cls: "sb-acct-opt",
      role: "menuitem",
      on: {
        click: async () => {
          _setAcctPopoverOpen(false);
          const obNow = (latestSnap && latestSnap.onboarding) || {};
          if (obNow.pending_oauth || obNow.oauth_awaiting_code) {
            openModal("oauth");
            return;
          }
          const res = await apiCall("start_colab_oauth");
          if (res && res.snapshot) window.renderStudioState(res.snapshot);
          if (res && res.ok) openModal("oauth");
          else if (res && res.message) showToast(res.message, { isError: true });
        },
      },
    }, h("span", { txt: "+ 添加 Google 账号..." })));

    pop.appendChild(h("button", {
      type: "button",
      cls: "sb-acct-opt",
      role: "menuitem",
      on: {
        click: () => {
          _setAcctPopoverOpen(false);
          switchStudioTab("overview");
          setTimeout(() => {
            const pool = el("ov-pool-list");
            if (pool && pool.scrollIntoView) pool.scrollIntoView({ behavior: "smooth", block: "nearest" });
          }, 60);
        },
      },
    }, h("span", { txt: "打开算力号池总览" })));
  }

  function _syncTunnelMaskUI(activeTunnel) {
    const inp = el("sb-tunnel-input");
    const eyeBtn = el("btn-sb-tunnel-eye");
    if (eyeBtn) {
      eyeBtn.classList.toggle("is-masked", Boolean(tunnelMasked));
      eyeBtn.setAttribute("aria-pressed", String(Boolean(tunnelMasked)));
    }
    if (!inp) return;
    inp.dataset.rawUrl = activeTunnel || "";
    if (!activeTunnel) {
      inp.value = "";
      inp.placeholder = "等待隧道地址就绪";
      inp.title = "当前暂无活动隧道";
    } else if (tunnelMasked) {
      inp.value = "••••••••••••••••••••••••••••";
      inp.placeholder = "";
      inp.title = "点击复制已脱敏隐藏的隧道地址";
    } else {
      inp.value = activeTunnel;
      inp.placeholder = "";
      inp.title = `点击复制：${activeTunnel}`;
    }
  }

  function renderSidebarControlCards(snap) {
    const s = snap || latestSnap || {};
    const ob = s.onboarding || {};
    const deploy = s.deploy || {};
    const cluster = s.cluster || {};
    const slots = cluster.slots || {};
    const comfy = s.comfy || {};
    const prefs = s.prefs || {};
    const accounts = Array.isArray(ob.accounts) ? ob.accounts : [];
    const activeId = String(ob.active_account_id || "");

    // 顶栏实时状态灯同步
    const tbDot = el("tb-status-dot");
    if (tbDot) {
      let cls = "tb-status-dot";
      if (deploy.active) cls += " is-busy";
      else if (comfy.has_error || s.ok === false) cls += " is-err";
      else if (s.is_running || cluster.active) cls += " is-online";
      tbDot.className = cls;
    }

    // 卡片 1：Colab 算力实例
    const onlineBadge = el("sb-card-online-badge");
    if (onlineBadge) {
      if (deploy.active) {
        onlineBadge.className = "sb-ec-status-pill is-busy";
        onlineBadge.textContent = `部署 ${Math.round(Number(deploy.pct || 0))}%`;
      } else if (s.is_running || cluster.active) {
        onlineBadge.className = "sb-ec-status-pill is-live";
        onlineBadge.textContent = "在线";
      } else {
        onlineBadge.className = "sb-ec-status-pill";
        onlineBadge.textContent = "待机";
      }
    }

    const activeObj = accounts.find((a) => String((a && a.id) || "") === activeId) || ob.active_account || null;
    const emailTxt = cleanText(
      (activeObj && activeObj.email_masked)
      || ob.active_account_email
      || ob.active_account_label
      || (ob.active_account && ob.active_account.label)
      || ""
    );
    const emailEl = el("sb-card-acct-email");
    if (emailEl) {
      emailEl.textContent = emailTxt || (ob.auth_missing ? "未绑定账号" : "Colab 主号");
      emailEl.classList.toggle("is-empty", !emailTxt && Boolean(ob.auth_missing));
      emailEl.title = emailTxt || "";
    }
    renderAccountSwitcherPopover(s);

    const gpuSel = el("sb-gpu-select");
    if (gpuSel) {
      const targetGpu = String(bootState.gpu || "G4").toUpperCase();
      if (Array.from(gpuSel.options).some((opt) => opt.value === targetGpu)) {
        gpuSel.value = targetGpu;
      }
      gpuSel.disabled = Boolean(deploy.active);
    }

    const deployBtn = el("btn-sb-quick-deploy");
    const deployLbl = el("sb-quick-deploy-lbl");
    if (deployBtn) deployBtn.disabled = Boolean(deploy.active);
    if (deployLbl) deployLbl.textContent = deploy.active ? "唤醒部署中..." : "唤醒实例并部署";

    const slotComfyUrl = slots.comfy && slots.comfy.tunnel_url ? slots.comfy.tunnel_url : "";
    const slotLlmUrl = slots.llm && slots.llm.tunnel_url ? slots.llm.tunnel_url : "";
    const activeTunnel = cleanText(comfy.tunnel_url || comfy.llm_tunnel_url || slotComfyUrl || slotLlmUrl || "");
    _syncTunnelMaskUI(activeTunnel);

    // 硬件状态行：GPU 完整型号、云盘持久化胶囊
    const gpuDot = el("sb-gpu-dot");
    if (gpuDot) {
      let gCls = "sb-gpu-dot";
      if (s.is_running || cluster.active) gCls += " is-online";
      else if (deploy.active) gCls += " is-idle";
      gpuDot.className = gCls;
    }
    const hwName = cleanText(comfy.gpu_full_name || comfy.gpu_name || "");
    if (el("sb-gpu-fullname")) {
      el("sb-gpu-fullname").textContent = hwName || (deploy.active ? "正在分配 Colab GPU 实例..." : "等待连接 GPU 实例");
      el("sb-gpu-fullname").title = hwName || "";
    }
    const drivePill = el("sb-drive-pill");
    if (drivePill) {
      const persisted = Boolean(comfy.drive_persisted || ob.gdrive_ok);
      drivePill.textContent = persisted ? "云盘持久化" : "未挂载云盘";
      drivePill.classList.toggle("is-off", !persisted);
    }

    renderTelemetrySparkline(s);

    // 卡片 2：算力额度与自动休眠
    let poolTotal = Number(cluster.total_pool_balance);
    if (!Number.isFinite(poolTotal) || (accounts.length === 0 && typeof s.balance === "number")) {
      if (accounts.length > 0) {
        poolTotal = accounts.reduce((acc, a) => acc + (typeof a.last_balance === "number" ? a.last_balance : 0), 0);
      } else {
        poolTotal = typeof s.balance === "number" ? s.balance : NaN;
      }
    }
    const rate = Boolean(cluster.active) && Number.isFinite(Number(cluster.total_rate_hourly))
      ? Number(cluster.total_rate_hourly)
      : Number(s.rate_hourly || 0);

    if (el("sb-kpi-pool-cu")) {
      el("sb-kpi-pool-cu").textContent = Number.isFinite(poolTotal) ? `${poolTotal.toFixed(2)} CU` : "-- CU";
    }
    const rateEl = el("sb-kpi-rate");
    if (rateEl) {
      rateEl.textContent = `${rate.toFixed(2)} CU/h`;
      rateEl.style.color = rate > 0.01 ? "" : "var(--ink-muted)";
    }
    updateSidebarUptimeTicker();

    const idleSel = el("sb-idle-select");
    if (idleSel && document.activeElement !== idleSel) {
      const rawIdle = Number(prefs.idle_auto_stop_minutes ?? 60);
      const validVals = ["0", "15", "30", "60"];
      idleSel.value = validVals.includes(String(rawIdle)) ? String(rawIdle) : "60";
    }

    let activeVmCount = 0;
    if (cluster.active && slots) {
      if (slots.llm && slots.llm.active !== false && slots.llm.endpoint) activeVmCount += 1;
      if (slots.comfy && slots.comfy.active !== false && slots.comfy.endpoint) activeVmCount += 1;
      if (activeVmCount === 0) activeVmCount = 1;
    } else if (Number(s.assignments_count || 0) > 0) {
      activeVmCount = Number(s.assignments_count);
    } else if (s.is_running || deploy.active) {
      activeVmCount = 1;
    }

    const relBtn = el("btn-sb-release-all");
    const relLbl = el("sb-release-label");
    if (relBtn && relLbl) {
      if (sbReleaseArm.isArmed("all")) {
        relBtn.className = "sb-release-btn is-armed";
        relLbl.textContent = "再点确认释放全部实例";
      } else {
        relBtn.className = "sb-release-btn";
        relLbl.textContent = activeVmCount > 0
          ? `释放全部实例 (${activeVmCount} 台运行中)`
          : "释放全部实例";
      }
    }

    // 卡片 3：快捷启动工作流列表
    const selected = normalizeProfiles(bootState.profiles, bootState.profile);
    const wfBadge = el("sb-wf-badge");
    if (wfBadge) {
      if (deploy.active) {
        wfBadge.className = "sb-card-badge is-busy";
        wfBadge.textContent = "部署中";
      } else if (s.is_running || cluster.active) {
        wfBadge.className = "sb-card-badge is-live";
        wfBadge.textContent = "运行中";
      } else {
        wfBadge.className = "sb-card-badge";
        wfBadge.textContent = `已选 ${selected.length} 项`;
      }
    }

    const wfList = el("sb-wf-list");
    if (wfList) {
      wfList.innerHTML = "";
      for (const wf of WORKFLOW_CATALOG) {
        const isSel = selected.includes(wf.id);
        wfList.appendChild(h("button", {
          type: "button",
          cls: `sb-wf-item${isSel ? " is-selected" : ""}`,
          data: { sbWf: wf.id },
          attrs: { "aria-pressed": String(isSel) },
          on: {
            click: () => toggleBootProfile(wf.id),
          },
        },
          h("span", { cls: "sb-wf-icon" }, createSvgIcon(wf.iconKind, 13)),
          h("span", { cls: "sb-wf-body" },
            h("span", { cls: "sb-wf-name", txt: wf.name }),
            h("span", { cls: "sb-wf-desc", txt: wf.title })
          ),
          h("span", { cls: "sb-wf-tag tabular", txt: cleanText(wf.specTail(bootState)) })
        ));
      }
    }
  }

  function renderOverviewHero(snap) {
    const s = snap || {};
    const ob = s.onboarding || {};
    const cluster = s.cluster || {};
    const accounts = Array.isArray(ob.accounts) ? ob.accounts : [];
    const activeId = String(ob.active_account_id || "");

    // 全池总家底计算：优先使用 cluster.total_pool_balance，否则汇总号池或回落单号余额
    let poolTotal = Number(cluster.total_pool_balance);
    if (!Number.isFinite(poolTotal) || (accounts.length === 0 && typeof s.balance === "number")) {
      if (accounts.length > 0) {
        poolTotal = accounts.reduce((acc, a) => acc + (typeof a.last_balance === "number" ? a.last_balance : 0), 0);
      } else {
        poolTotal = typeof s.balance === "number" ? s.balance : NaN;
      }
    }
    if (el("ov-pool-cu")) {
      el("ov-pool-cu").textContent = Number.isFinite(poolTotal) ? poolTotal.toFixed(2) : "--.--";
    }
    const healthyCnt = Number(cluster.pool_healthy_count || 0);
    if (el("ov-pool-sub")) {
      const tierSummary = cluster.pool_tier_summary || formatOverviewPoolTierSummary(accounts, activeId, s);
      el("ov-pool-sub").textContent = accounts.length > 0
        ? `${tierSummary} · ${healthyCnt} 个余额充沛`
        : (ob.auth_missing ? "尚未绑定 Google Colab 账号" : "单账号运行模式");
    }

    // 当前激活号 CU 与水位
    if (el("ov-active-cu")) {
      el("ov-active-cu").textContent = typeof s.balance === "number" ? s.balance.toFixed(2) : "--.--";
    }
    if (el("ov-active-acct")) {
      const balPct = Number(s.balance_pct || 0);
      const activeObj = accounts.find((a) => String(a && a.id || "") === activeId) || ob.active_account || null;
      const acctLbl = ob.active_account_label || (activeObj && activeObj.label) || "未绑定";
      const tInfo = activeObj ? resolveAccountTierInfo(activeObj, true, s) : null;
      const tierBit = (tInfo && tInfo.tier !== "unknown") ? ` · ${tInfo.label}` : "";
      el("ov-active-acct").textContent = typeof s.balance === "number"
        ? `${cleanText(acctLbl)}${tierBit} · 水位 ${balPct.toFixed(1)}%`
        : cleanText(acctLbl);
    }
    const balBar = el("ov-bal-bar");
    if (balBar) {
      const pct = Math.max(0, Math.min(100, Number(s.balance_pct || 0)));
      balBar.style.width = `${pct}%`;
      balBar.className = "hairline-progress-fill";
      if (pct <= 10 || s.low_balance) balBar.classList.add("danger");
      else if (pct <= 30) balBar.classList.add("warn");
    }

    // 跨号总耗速与续航
    const clusterActive = Boolean(cluster.active);
    const rate = clusterActive && Number.isFinite(Number(cluster.total_rate_hourly))
      ? Number(cluster.total_rate_hourly)
      : Number(s.rate_hourly || 0);
    if (el("ov-rate")) el("ov-rate").textContent = rate.toFixed(2);
    let etaHrs = s.remaining_hours;
    if ((etaHrs === null || etaHrs === undefined) && clusterActive && Number(cluster.total_pool_balance || 0) > 0 && rate > 0.01) {
      etaHrs = Number(cluster.total_pool_balance) / rate;
    }
    if (el("ov-eta")) {
      if (rate <= 0.01) el("ov-eta").textContent = "零计费待机中";
      else if (etaHrs !== null && etaHrs !== undefined) el("ov-eta").textContent = `预计续航约 ${Number(etaHrs).toFixed(1)}h`;
      else el("ov-eta").textContent = "实时计费中";
    }

    const pill = el("ov-status-pill");
    const dot = el("ov-status-dot");
    const stTxt = el("ov-status-text");
    const lvl = s.status_level || "idle";
    if (pill) pill.className = `status-chip ${lvl}`;
    if (dot) dot.className = `status-dot ${lvl}`;
    if (stTxt) stTxt.textContent = cleanText(s.status_label || "已停机省积分");

    if (el("ov-day-consumed")) {
      const dayConsumed = (accounts.length >= 2 && Number.isFinite(Number(cluster.total_day_consumed_cu)) && Number(cluster.total_day_consumed_cu) > Number(s.day_consumed_cu || 0))
        ? Number(cluster.total_day_consumed_cu)
        : Number(s.day_consumed_cu || 0);
      el("ov-day-consumed").textContent = `今日实耗 ${dayConsumed.toFixed(2)} CU`;
    }
  }

  function renderBootWorkflowList() {
    const listEl = el("ov-wf-list");
    const modsBar = el("ov-wf-mods-bar");
    if (!listEl) return;

    const selected = normalizeProfiles(bootState.profiles, bootState.profile);
    bootState.profiles = selected;
    bootState.profile = selected[0];

    document.querySelectorAll("#ov-wf-cat-seg .seg-item").forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.cat === wfCategory);
    });

    listEl.innerHTML = "";
    const filtered = WORKFLOW_CATALOG.filter((wf) => wfCategory === "all" || wf.category === wfCategory);
    for (const wf of filtered) {
      const isSelected = selected.includes(wf.id);
      const card = h("button", {
        type: "button",
        cls: `wf-option-card${isSelected ? " active" : ""}`,
        data: { wf: wf.id },
        attrs: { "aria-pressed": String(isSelected) },
        on: {
          click: () => toggleBootProfile(wf.id),
        },
      },
        h("div", { cls: "wf-option-left" },
          h("span", { cls: "wf-option-icon" }, createSvgIcon(wf.iconKind, 15)),
          h("div", { cls: "wf-option-text" },
            h("span", { cls: "wf-option-name", txt: wf.title }),
            h("span", { cls: "wf-option-desc", txt: wf.desc })
          )
        ),
        h("div", { cls: "wf-option-right" },
          h("span", { cls: `pill-tag${isSelected ? " accent" : ""}`, txt: `${wf.recGpu} 推荐` }),
          h("span", { cls: "pill-tag tabular", txt: wf.specTail(bootState) })
        )
      );
      listEl.appendChild(card);
    }

    // 汇总渲染已选工作流的可调子模块开关
    if (!modsBar) return;
    modsBar.innerHTML = "";
    const modItems = [];
    for (const pid of selected) {
      const wf = getWorkflowMeta(pid);
      for (const mod of wf.modules || []) {
        modItems.push({ wf, mod });
      }
    }
    if (modItems.length === 0) {
      modsBar.classList.add("hidden");
      return;
    }
    modsBar.classList.remove("hidden");
    modsBar.appendChild(h("span", { cls: "metric-sub", txt: "子模块参数:" }));

    for (const { wf, mod } of modItems) {
      if (mod.type === "choice" && Array.isArray(mod.options)) {
        const curOpt = mod.options.find((o) => String(o.val) === String(bootState[mod.key])) || mod.options[0];
        modsBar.appendChild(h("button", {
          type: "button",
          cls: "btn btn-xs btn-subtle active",
          txt: `${wf.name} · ${curOpt.label}`,
          title: "点击循环切换参数档位",
          on: {
            click: () => {
              const curIdx = Math.max(0, mod.options.findIndex((o) => String(o.val) === String(bootState[mod.key])));
              const nextOpt = mod.options[(curIdx + 1) % mod.options.length];
              userTouchedBoot = true;
              bootState[mod.key] = nextOpt.val;
              if (mod.prefKey) apiCall("save_prefs", { [mod.prefKey]: nextOpt.val });
              renderBootCockpit(latestSnap);
            },
          },
        }));
      } else {
        const on = Boolean(bootState[mod.key]);
        modsBar.appendChild(h("button", {
          type: "button",
          cls: `btn btn-xs btn-subtle${on ? " active" : ""}`,
          txt: `${mod.label} ${mod.sizeHint || ""}`.trim(),
          attrs: { "aria-pressed": String(on) },
          on: {
            click: () => {
              const nextVal = !Boolean(bootState[mod.key]);
              userTouchedBoot = true;
              bootState[mod.key] = nextVal;
              if (mod.prefKey) apiCall("save_prefs", { [mod.prefKey]: nextVal });
              renderBootCockpit(latestSnap);
            },
          },
        }));
      }
    }
  }

  function toggleBootProfile(profId) {
    const nextList = [...normalizeProfiles(bootState.profiles, bootState.profile)];
    const idx = nextList.indexOf(profId);
    if (idx >= 0) {
      if (nextList.length <= 1) {
        showToast("至少保留 1 项云端工作流");
        return;
      }
      nextList.splice(idx, 1);
    } else {
      if (profId === "qwen_27b") {
        const i180 = nextList.indexOf("qwen_llm");
        if (i180 >= 0) nextList.splice(i180, 1);
      } else if (profId === "qwen_llm") {
        const i27 = nextList.indexOf("qwen_27b");
        if (i27 >= 0) nextList.splice(i27, 1);
      }
      nextList.push(profId);
    }
    userTouchedBoot = true;
    bootState.profiles = normalizeProfiles(nextList, bootState.profile);
    bootState.profile = bootState.profiles[0];
    bootState.gpu = pickBestGpuForProfiles(bootState.profiles, latestSnap);
    const anyAutoHm = bootState.profiles.some((p) => getWorkflowMeta(p).autoHmOnA100);
    if (GPU_SPECS[bootState.gpu] && GPU_SPECS[bootState.gpu].hmLock) {
      bootState.high_mem = false;
    } else if (anyAutoHm && bootState.gpu === "A100") {
      bootState.high_mem = true;
    } else {
      bootState.high_mem = false;
    }
    apiCall("save_prefs", {
      boot_profiles: bootState.profiles,
      boot_profile: bootState.profile,
      boot_gpu: bootState.gpu,
      boot_high_mem: bootState.high_mem,
    });
    renderBootCockpit(latestSnap);
  }

  function renderBootCockpit(snap) {
    const s = snap || latestSnap || {};
    renderBootWorkflowList();

    const profs = bootState.profiles;
    const wf = getWorkflowMeta(profs[0]);
    const dualStack = isDualProfileSet(profs);
    const llmProf = getActiveLlmProfile(profs);
    const splitPrev = resolveSplitPlanPreview(profs, s);
    const recGpu = (dualStack || profs.includes("qwen_llm") || profs.includes("minimax") || profs.includes("drama3"))
      ? (profs.length === 1 && wf.recGpu ? wf.recGpu : "G4")
      : (wf.recGpu || "L4");

    // 集群分段控制（双号异构 / 单号同驻）
    const clusterSeg = el("ov-cluster-seg");
    if (clusterSeg) {
      clusterSeg.classList.toggle("hidden", !splitPrev.canSplit);
      clusterSeg.querySelectorAll(".seg-item").forEach((btn) => {
        const wantSplit = btn.dataset.mode === "split";
        btn.classList.toggle("active", wantSplit === splitPrev.useSplit);
      });
    }

    const hmBtn = el("btn-boot-hm");
    const curSpec = GPU_SPECS[bootState.gpu] || GPU_SPECS.L4;
    if (hmBtn) {
      hmBtn.classList.toggle("hidden", splitPrev.useSplit);
      hmBtn.classList.toggle("active", !curSpec.hmLock && Boolean(bootState.high_mem));
      hmBtn.textContent = curSpec.hmLock ? "自带高存" : "高内存";
      hmBtn.disabled = Boolean(curSpec.hmLock);
    }

    // 5 档显卡矩阵
    const gpuGrid = el("ov-gpu-grid");
    const eligible = Array.isArray(s.eligible_gpus) ? s.eligible_gpus.map((x) => String(x).toUpperCase()) : null;
    const scarce = s.scarce_gpus || {};
    const hmOn = Boolean(bootState.high_mem);
    if (gpuGrid) {
      gpuGrid.innerHTML = "";
      for (const g of GPU_ORDER) {
        const gs = GPU_SPECS[g];
        const isScarce = Boolean(scarce[g]);
        const isRestr = eligible !== null && !eligible.includes(g);
        const cardHm = !gs.hmLock && hmOn;
        const rInfo = resolveGpuRate(g, cardHm, s);
        const btn = h("button", {
          type: "button",
          cls: `gpu-option${bootState.gpu === g ? " active" : ""}${isScarce ? " scarce" : ""}${isRestr ? " restricted" : ""}`,
          data: { gpu: g },
          on: {
            click: () => {
              userTouchedBoot = true;
              bootState.gpu = g;
              if (gs.hmLock) bootState.high_mem = false;
              else if (profs.some((p) => getWorkflowMeta(p).autoHmOnA100) && g === "A100") bootState.high_mem = true;
              apiCall("save_prefs", { boot_gpu: bootState.gpu, boot_high_mem: bootState.high_mem });
              renderBootCockpit(latestSnap);
            },
          },
        },
          h("div", { cls: "gpu-opt-head" },
            h("span", { txt: g }),
            g === recGpu ? createSvgIcon("star", 11, "icon-svg icon-accent") : null
          ),
          h("span", { cls: "gpu-opt-vram tabular", txt: isScarce ? "紧缺" : (isRestr ? "受限" : (cardHm ? gs.vramHm : gs.vram)) }),
          h("span", { cls: "gpu-opt-rate tabular", txt: isRestr ? "未开放" : rInfo.shortLabel })
        );
        gpuGrid.appendChild(btn);
      }
    }

    // 摘要提示与启动按钮文案
    const sumEl = el("ov-boot-summary");
    const selHm = !curSpec.hmLock && hmOn;
    const selRate = resolveGpuRate(bootState.gpu, selHm, s);
    const curScarce = scarce[bootState.gpu];
    const curRestr = eligible !== null && !eligible.includes(bootState.gpu);
    const fallbackGpu = pickBestGpuForProfiles(profs, s);

    if (sumEl) {
      sumEl.className = "boot-summary-bar tabular";
      if (splitPrev.useSplit) {
        sumEl.textContent = `LLM (${splitPrev.llmGpu} ${splitPrev.llmRate.shortLabel}) + 画布 (${splitPrev.comfyGpu} ${splitPrev.comfyRate.shortLabel}) · 双号分担 ${splitPrev.totalRate.toFixed(2)} CU/h`;
      } else if (llmProf === "qwen_llm" && (bootState.gpu === "T4" || bootState.gpu === "L4")) {
        sumEl.classList.add("danger");
        sumEl.textContent = "显存不足：180B 模型无法在 T4/L4 运行，请改选 G4 或 A100 高存";
      } else if (dualStack && (bootState.gpu === "T4" || bootState.gpu === "L4")) {
        sumEl.classList.add("danger");
        sumEl.textContent = "显存不足：双栖组合无法在 T4/L4 同驻，请改选 G4 或启用双号异构";
      } else if (curRestr) {
        sumEl.classList.add("danger");
        sumEl.textContent = `当前账号未开放 ${bootState.gpu}${fallbackGpu !== bootState.gpu ? `，建议改选 ${fallbackGpu}` : ""}`;
      } else if (curScarce) {
        sumEl.classList.add("warn");
        sumEl.textContent = `${bootState.gpu} 高峰紧缺${fallbackGpu !== bootState.gpu ? `，建议改选 ${fallbackGpu}` : ""}`;
      } else {
        const vramTxt = (selHm ? curSpec.vramHm : curSpec.vram).replace(/高$/, "");
        const ramTxt = selHm ? curSpec.ramHm : curSpec.ram;
        const tailTxt = profs.length >= 2 ? `${dualStack ? "双栖" : "合并"}共 ${profs.length} 套` : wf.specTail(bootState);
        sumEl.textContent = `${bootState.gpu} · ${vramTxt} 显存 · ${ramTxt} · ${tailTxt} · ${selRate.badgeLabel}`;
      }
    }

    const profLabel = profs.map((p) => getWorkflowMeta(p).name).join(" + ");
    if (el("btn-ov-boot-label")) {
      el("btn-ov-boot-label").textContent = splitPrev.useSplit
        ? `双号异构部署 ${profLabel}`
        : `一键开机部署 ${profLabel} · ${bootState.gpu}${selHm ? " (高存)" : ""}`;
    }
    renderSidebarControlCards(s);
  }

  function renderOverviewTopology(snap) {
    const s = snap || {};
    const deploy = s.deploy || {};
    const cluster = s.cluster || {};
    const slots = cluster.slots || {};
    const comfy = s.comfy || {};
    const isDeploying = Boolean(deploy.active);
    const clusterActive = Boolean(cluster.active);
    const isRunning = Boolean(s.is_running || clusterActive);

    const deployCard = el("ov-deploy-card");
    const liveCard = el("ov-live-topo");
    const bootCard = el("ov-boot-cockpit");
    const modeSub = el("ov-topo-mode");
    const toggleBootBtn = el("btn-ov-toggle-boot");

    if (isDeploying) {
      if (deployCard) deployCard.classList.remove("hidden");
      if (liveCard) liveCard.classList.add("hidden");
      if (bootCard) bootCard.classList.add("hidden");
      if (toggleBootBtn) toggleBootBtn.classList.add("hidden");
      if (modeSub) modeSub.textContent = "正在执行 7 步云端部署";

      const pct = Math.max(2, Math.min(100, Number(deploy.pct || 0)));
      const stepIdx = Number(deploy.step_idx || 0);
      if (el("ov-deploy-stage")) el("ov-deploy-stage").textContent = cleanText(deploy.stage_title || `[${stepIdx}/7] 正在部署`);
      if (el("ov-deploy-pct")) el("ov-deploy-pct").textContent = `${Math.round(pct)}%`;
      if (el("ov-deploy-elapsed")) el("ov-deploy-elapsed").textContent = formatElapsedMmSs(deploy.elapsed_sec || 0);
      if (el("ov-deploy-detail")) el("ov-deploy-detail").textContent = cleanText(deploy.sub_detail || "正在连接 Colab...");
      if (el("ov-deploy-bar")) el("ov-deploy-bar").style.width = `${pct}%`;

      document.querySelectorAll("#ov-deploy-steps .step-pill").forEach((pill) => {
        const idx = Number(pill.dataset.step);
        pill.className = "step-pill";
        if (stepIdx > idx || pct >= 100) pill.classList.add("done");
        else if (stepIdx === idx) pill.classList.add("current");
      });
      const dProfs = Array.isArray(deploy.profiles) && deploy.profiles.length > 0
        ? normalizeProfiles(deploy.profiles, deploy.profile || "anima")
        : [String(deploy.profile || "anima")];
      if (el("ov-deploy-foot")) {
        el("ov-deploy-foot").textContent = `${dProfs.map((p) => getWorkflowMeta(p).name).join(" + ")} · ${deploy.gpu || "L4"}`;
      }
      return;
    }

    if (deployCard) deployCard.classList.add("hidden");

    if (isRunning) {
      if (toggleBootBtn) {
        toggleBootBtn.classList.remove("hidden");
        toggleBootBtn.textContent = forceShowBootStandby ? "返回实时拓扑" : "切换开机配置";
      }
      if (forceShowBootStandby) {
        if (liveCard) liveCard.classList.add("hidden");
        if (bootCard) bootCard.classList.remove("hidden");
        if (modeSub) modeSub.textContent = "查看或切换开机参数";
        renderBootCockpit(s);
        return;
      }

      if (liveCard) liveCard.classList.remove("hidden");
      if (bootCard) bootCard.classList.add("hidden");

      const sk = String(comfy.service_kind || "comfy");
      const hasLlmSlot = Boolean(slots.llm);
      const hasComfySlot = Boolean(slots.comfy);
      const isDual = (hasLlmSlot && hasComfySlot) || sk === "dual";
      const llmOn = Boolean((slots.llm && slots.llm.online) || comfy.llm_online || (comfy.online && sk === "llm"));
      const comfyOn = Boolean((slots.comfy && slots.comfy.online) || comfy.comfy_online || (comfy.online && sk !== "llm"));
      const llmRunning = Number(comfy.llm_running_count ?? comfy.running_cnt ?? 0);
      const comfyRunning = Number(comfy.running_count || 0);
      const comfyPending = Number(comfy.pending_count || 0);

      if (modeSub) {
        modeSub.textContent = (hasLlmSlot && hasComfySlot)
          ? "双号异构分担在线"
          : (sk === "dual" ? "同机双栖运行中" : (llmOn ? "LLM 单模在线" : "ComfyUI 单模在线"));
      }

      const sessions = Array.isArray(s.sessions) ? s.sessions : [];
      const firstSess = sessions[0] || {};
      const activeAcctLabel = (s.onboarding && s.onboarding.active_account_label) || "主账号";

      // 渲染 LLM 节点
      const llmCard = el("topo-llm-card");
      const llmActive = Boolean(hasLlmSlot || llmOn || sk === "llm" || sk === "dual");
      if (llmCard) {
        llmCard.className = "topo-node";
        if (llmOn) llmCard.classList.add("online");
        if (llmRunning > 0) llmCard.classList.add("sampling");
      }
      if (el("topo-llm-badge")) {
        el("topo-llm-badge").className = `pill-tag${llmOn ? (llmRunning > 0 ? " accent" : " ok") : ""}`;
        el("topo-llm-badge").textContent = llmRunning > 0 ? `生成中 (${llmRunning})` : (llmOn ? "在线" : (llmActive ? "连通中" : "未挂载"));
      }
      if (el("topo-llm-model")) {
        el("topo-llm-model").textContent = llmActive
          ? (comfy.model_name || "Qwen3.8")
          : "LLM 未挂载";
      }
      if (el("topo-llm-specs")) {
        const lGpu = (slots.llm && slots.llm.gpu) || firstSess.accelerator || "--";
        const lAcct = (slots.llm && slots.llm.label) || activeAcctLabel;
        const qTag = comfy.quant ? formatShortQuantTag(comfy.quant) : "";
        const cTag = comfy.ctx_size ? formatShortCtxTag(comfy.ctx_size) : "";
        const specParts = [cleanText(lAcct), lGpu, [qTag, cTag].filter(Boolean).join("·")].filter(Boolean);
        el("topo-llm-specs").textContent = llmActive ? specParts.join(" · ") : "--";
      }
      if (el("topo-llm-vram")) {
        const tps = Number(comfy.tps || 0);
        const tpsTxt = tps > 0 ? ` · ${tps.toFixed(1)} tok/s` : "";
        const hasVram = Number(comfy.vram_total_gb || 0) > 0 && (sk === "llm" || sk === "dual");
        el("topo-llm-vram").textContent = hasVram
          ? `VRAM ${Number(comfy.vram_used_gb).toFixed(1)}/${Number(comfy.vram_total_gb).toFixed(1)}G${tpsTxt}`
          : (llmOn ? `公网接口已连通${tpsTxt}` : "等待隧道");
      }
      if (el("topo-llm-rate")) {
        const lRate = slots.llm ? Number(slots.llm.rate_hourly || 0) : (sk === "llm" ? Number(s.rate_hourly || 0) : 0);
        el("topo-llm-rate").textContent = `${lRate.toFixed(2)} CU/hr`;
      }
      if (el("topo-llm-bar")) {
        const vp = (sk === "llm" || sk === "dual") ? Number(comfy.vram_pct || 0) : (llmOn ? 100 : 0);
        el("topo-llm-bar").style.width = `${Math.max(llmOn ? 12 : 0, Math.min(100, vp || (llmOn ? 40 : 0)))}%`;
      }
      if (el("btn-topo-llm-open")) el("btn-topo-llm-open").disabled = !llmActive;
      if (el("btn-topo-llm-copy")) el("btn-topo-llm-copy").disabled = !llmActive;

      // 渲染桥接标签
      const bridge = el("topo-bridge");
      if (bridge) bridge.classList.toggle("active", isDual);
      if (el("topo-bridge-label")) {
        el("topo-bridge-label").textContent = (hasLlmSlot && hasComfySlot)
          ? "双号异构"
          : (isDual ? "同机双栖" : "单节点");
      }

      // 渲染 ComfyUI 画布节点
      const comfyCard = el("topo-comfy-card");
      const comfyActive = Boolean(hasComfySlot || comfyOn || sk === "comfy" || sk === "dual");
      if (comfyCard) {
        comfyCard.className = "topo-node";
        if (comfy.has_error) comfyCard.classList.add("error");
        else if (comfyOn) comfyCard.classList.add("online");
        if (comfyRunning > 0 || comfyPending > 0) comfyCard.classList.add("sampling");
      }
      if (el("topo-comfy-badge")) {
        let cTagCls = "";
        let cTagTxt = "未挂载";
        if (comfy.has_error) {
          cTagCls = " danger";
          cTagTxt = "报错中断";
        } else if (comfyRunning > 0 || comfyPending > 0) {
          cTagCls = " accent";
          cTagTxt = Number(comfy.step_max || 0) > 0
            ? `采样 ${Math.round(Number(comfy.step_pct || 0))}%`
            : `队列 ${comfyRunning}/${comfyRunning + comfyPending}`;
        } else if (comfyOn) {
          cTagCls = " ok";
          cTagTxt = "队列空闲";
        } else if (comfyActive) {
          cTagTxt = "连通中";
        }
        el("topo-comfy-badge").className = `pill-tag${cTagCls}`;
        el("topo-comfy-badge").textContent = cTagTxt;
      }
      if (el("topo-comfy-profile")) {
        const actProfs = Array.isArray(comfy.active_profiles) ? comfy.active_profiles : [];
        const comfyProfs = actProfs.filter((p) => ["anima", "minimax", "drama3"].includes(p));
        el("topo-comfy-profile").textContent = comfyActive
          ? (comfyProfs.length > 0 ? comfyProfs.map((p) => getWorkflowMeta(p).name).join(" + ") : "ComfyUI 画布")
          : "画布未挂载";
      }
      if (el("topo-comfy-specs")) {
        const cGpu = (slots.comfy && slots.comfy.gpu) || firstSess.accelerator || "--";
        const cAcct = (slots.comfy && slots.comfy.label) || activeAcctLabel;
        el("topo-comfy-specs").textContent = comfyActive ? `${cleanText(cAcct)} · ${cGpu}` : "--";
      }
      if (el("topo-comfy-vram")) {
        const hasVram = Number(comfy.vram_total_gb || 0) > 0;
        const pendPull = Number(comfy.pending_pull_count || 0);
        el("topo-comfy-vram").textContent = comfy.has_error
          ? cleanText(comfy.error_summary || "推理报错")
          : (hasVram
            ? `VRAM ${Number(comfy.vram_used_gb).toFixed(1)}/${Number(comfy.vram_total_gb).toFixed(1)}G${pendPull > 0 ? ` · 待拉回 ${pendPull}` : ""}`
            : (comfyOn ? "公网画布已就绪" : "等待隧道"));
      }
      if (el("topo-comfy-rate")) {
        const cRate = slots.comfy ? Number(slots.comfy.rate_hourly || 0) : (sk !== "llm" ? Number(s.rate_hourly || 0) : 0);
        el("topo-comfy-rate").textContent = `${cRate.toFixed(2)} CU/hr`;
      }
      if (el("topo-comfy-bar")) {
        const vp = Number(comfy.vram_pct || 0);
        const barFill = el("topo-comfy-bar");
        barFill.style.width = `${Math.max(comfyOn ? 8 : 0, Math.min(100, vp || (comfyOn ? 30 : 0)))}%`;
        barFill.className = "hairline-progress-fill";
        if (comfy.has_error || vp >= 93) barFill.classList.add("danger");
        else if (vp >= 85) barFill.classList.add("warn");
      }
      if (el("btn-topo-comfy-open")) el("btn-topo-comfy-open").disabled = !comfyActive;
      if (el("btn-topo-comfy-copy")) el("btn-topo-comfy-copy").disabled = !comfyActive;
      if (el("btn-topo-comfy-sync")) el("btn-topo-comfy-sync").disabled = !comfyActive;

      // 分槽独立停卡按钮显隐
      const stopComfyBtn = el("btn-stop-comfy-slot");
      const stopLlmBtn = el("btn-stop-llm-slot");
      const stopAllBtn = el("btn-stop-all-slot");
      if (stopComfyBtn) stopComfyBtn.classList.toggle("hidden", !(clusterActive && hasComfySlot));
      if (stopLlmBtn) stopLlmBtn.classList.toggle("hidden", !(clusterActive && hasLlmSlot));
      if (stopAllBtn) {
        stopAllBtn.textContent = stopAllArm.isArmed()
          ? "再点确认全停"
          : (clusterActive ? "全停" : "一键停卡");
      }
      return;
    }

    // 待机状态
    forceShowBootStandby = false;
    if (liveCard) liveCard.classList.add("hidden");
    if (toggleBootBtn) toggleBootBtn.classList.add("hidden");
    if (bootCard) bootCard.classList.remove("hidden");
    if (modeSub) modeSub.textContent = "选择工作流与算力配置开机";
    renderBootCockpit(s);
  }

  function renderAccountPool(snap) {
    const s = snap || {};
    const ob = s.onboarding || {};
    const cluster = s.cluster || {};
    const slots = cluster.slots || {};
    const accounts = Array.isArray(ob.accounts) ? ob.accounts : [];
    const activeId = String(ob.active_account_id || "");

    if (el("ov-pool-count")) el("ov-pool-count").textContent = `${accounts.length} 个账号`;

    const autoBtn = el("btn-pool-auto");
    if (autoBtn) {
      const autoOn = !(s.prefs && s.prefs.auto_pool_schedule === false);
      autoBtn.classList.toggle("active", autoOn);
      autoBtn.setAttribute("aria-pressed", String(autoOn));
    }

    const addBtn = el("btn-pool-add");
    if (addBtn) {
      addBtn.textContent = (ob.pending_oauth || ob.oauth_awaiting_code) ? "提交授权码" : "新增账号";
    }

    const listEl = el("ov-account-list");
    if (!listEl) return;
    listEl.innerHTML = "";

    if (accounts.length === 0) {
      listEl.appendChild(h("div", { cls: "cloud-empty-row" },
        ob.auth_expired
          ? "Google 登录授权已过期，请点击右上角「新增账号」重新绑定"
          : "尚未绑定 Colab Pro / Ultra 账号，绑定后支持多号池余额聚合与秒切调度"
      ));
      return;
    }

    for (const acct of accounts) {
      const aid = String(acct.id || "");
      const isCur = Boolean(aid && aid === activeId && !ob.auth_missing);
      const isRunning = isCur ? Boolean(s.is_running) : Boolean(acct.last_is_running);
      const bal = isCur && typeof s.balance === "number" ? s.balance : acct.last_balance;
      const balText = (typeof bal === "number" && !Number.isNaN(bal)) ? `${bal.toFixed(1)} CU` : "-- CU";

      const tInfo = resolveAccountTierInfo(acct, isCur, s);
      const tierBadge = tInfo.tier !== "unknown"
        ? h("span", { cls: `pill-tag ${tInfo.pillCls}`, txt: tInfo.label, title: `账号权益等级: Colab ${tInfo.label}` })
        : null;

      const rawElig = Array.isArray(acct.eligible_gpus)
        ? acct.eligible_gpus
        : (isCur && Array.isArray(s.eligible_gpus) ? s.eligible_gpus : null);
      let poolBadge = null;
      if (Array.isArray(rawElig) && rawElig.length > 0) {
        const up = rawElig.map((g) => String(g || "").toUpperCase());
        const hasG4 = up.includes("G4");
        const hasA100 = up.includes("A100");
        const hasH100 = up.includes("H100");
        if (hasH100 || hasG4 || hasA100) {
          const tags = [hasH100 ? "H100" : "", hasG4 ? "G4" : "", hasA100 ? "A100" : ""].filter(Boolean).slice(0, 2).join(" / ");
          poolBadge = h("span", { cls: "pill-tag accent", txt: tags, title: `支持显卡池: ${up.join(" / ")}` });
        } else if (up.includes("L4")) {
          poolBadge = h("span", { cls: "pill-tag", txt: "L4 / T4", title: `支持显卡池: ${up.join(" / ")}` });
        } else {
          poolBadge = h("span", { cls: "pill-tag warn", txt: "仅 T4", title: "当前配额受限，仅支持 T4" });
        }
      }

      let role = String(acct.cluster_role || "").toLowerCase();
      if (!role && cluster.active) {
        if (slots.llm && String(slots.llm.account_id || "") === aid) role = "llm";
        else if (slots.comfy && String(slots.comfy.account_id || "") === aid) role = "comfy";
      }
      const roleBadge = role === "llm"
        ? h("span", { cls: "pill-tag ok", txt: "LLM 槽位" })
        : (role === "comfy" ? h("span", { cls: "pill-tag accent", txt: "画布槽位" }) : null);

      const dotCls = isRunning ? "running" : (isCur ? "idle" : "");

      const row = h("div", { cls: "inset-row", data: { acctId: aid } },
        h("div", { cls: "acct-row-left" },
          h("span", { cls: `status-dot ${dotCls}` }),
          h("div", { cls: "acct-row-info" },
            h("div", { cls: "acct-row-title" },
              h("span", { cls: "acct-row-name", txt: cleanText(acct.label || "Colab 账号") }),
              tierBadge,
              h("button", {
                type: "button",
                cls: "btn btn-xs btn-ghost acct-row-edit-btn",
                title: "修改备注名",
                attrs: { "aria-label": "修改备注名" },
                on: { click: () => openModal("rename", { account: acct }) },
              }, createSvgIcon("pencil", 12))
            ),
            h("span", { cls: "acct-row-email", txt: acct.email_masked || "***" })
          )
        ),
        h("div", { cls: "acct-row-right" },
          poolBadge,
          roleBadge,
          h("span", { cls: "acct-row-cu tabular", txt: balText }),
          isCur
            ? h("span", { cls: "pill-tag ok", txt: "当前使用" })
            : h("button", {
                type: "button",
                cls: "btn btn-xs btn-subtle",
                txt: "秒切",
                on: {
                  click: async () => {
                    showToast(`正在切换至 ${cleanText(acct.label || "目标账号")}...`);
                    const res = await apiCall("switch_colab_account", aid);
                    if (res && res.snapshot) window.renderStudioState(res.snapshot);
                    if (res && res.message) showToast(res.message, { isError: res.ok === false });
                  },
                },
              }),
          h("button", {
            type: "button",
            cls: `btn btn-xs ${acctDelArm.isArmed(aid) ? "btn-danger" : "btn-ghost"}`,
            txt: acctDelArm.isArmed(aid) ? "确认移除" : null,
            title: isCur ? "解绑当前账号" : "移除账号",
            attrs: { "aria-label": "移除账号" },
            on: {
              click: async () => {
                if (!acctDelArm.isArmed(aid)) {
                  acctDelArm.arm(aid, () => renderAccountPool(latestSnap));
                  renderAccountPool(latestSnap);
                  return;
                }
                acctDelArm.disarm();
                const res = await apiCall("remove_colab_account", aid);
                if (res && res.snapshot) window.renderStudioState(res.snapshot);
                if (res && res.message) showToast(res.message, { isError: res.ok === false });
              },
            },
          }, acctDelArm.isArmed(aid) ? null : createSvgIcon("close", 12))
        )
      );
      listEl.appendChild(row);
    }
  }

  /* ----------------------------------------------------------------
     视图 2：作品画廊、多维筛选与 Finder Inspector
     ---------------------------------------------------------------- */
  const GAL_NAT_COLLATOR = new Intl.Collator("zh-CN", { numeric: true, sensitivity: "base" });
  const GAL_DATE_LABELS = { all: "全部时间", today: "今天", "7d": "近 7 天", "30d": "近 30 天" };
  const GAL_GROUP_LABELS = { date: "按日期分组", folder: "按目录分组", none: "平铺不分组" };
  const GAL_SORT_LABELS = {
    newest: "最新优先",
    oldest: "最早优先",
    name_asc: "名称批次 A-Z",
    name_desc: "名称批次 Z-A",
    size_desc: "体积从大到小",
  };
  const GAL_POPOVER_MAP = {
    folder: ["gal-folder-popover", "btn-gal-folder"],
    date: ["gal-date-popover", "btn-gal-date"],
    group: ["gal-group-popover", "btn-gal-group"],
    sort: ["gal-sort-popover", "btn-gal-sort"],
  };

  function _localDateKey(dtObj) {
    const y = dtObj.getFullYear();
    const m = String(dtObj.getMonth() + 1).padStart(2, "0");
    const d = String(dtObj.getDate()).padStart(2, "0");
    return `${y}-${m}-${d}`;
  }

  function _formatCardShortTime(mtimeStr) {
    const s = String(mtimeStr || "");
    return s.length >= 16 ? s.slice(5, 16) : s;
  }

  function _calcGalleryColCount() {
    const stage = el("gallery-stage");
    if (!stage) return 3;
    const sw = stage.clientWidth;
    if (!sw || sw <= 0) return galleryState.lastColCount || 3;
    const padX = sw <= 560 ? 28 : 40;
    const gap = sw <= 560 ? 10 : 14;
    const availW = Math.max(160, sw - padX);
    return Math.max(1, Math.min(6, Math.floor((availW + gap) / (205 + gap))));
  }

  function syncGalleryResponsiveCols() {
    if (currentView !== "gallery" || galleryState.layout !== "masonry") return;
    const nextCols = _calcGalleryColCount();
    if (nextCols > 0 && nextCols !== galleryState.lastColCount && Array.isArray(galleryState.items) && galleryState.items.length > 0) {
      renderGalleryGrid();
    }
  }

  function closeAllGalleryPopovers(except = "") {
    for (const [key, [popId, btnId]] of Object.entries(GAL_POPOVER_MAP)) {
      if (key === except) continue;
      _setFloatingPopoverVisible(el(popId), el(btnId), false);
    }
    if (!except) {
      galleryState.openPopover = "";
    }
  }

  function toggleGalleryPopover(kindKey, forceOpen) {
    const pair = GAL_POPOVER_MAP[kindKey];
    if (!pair) return;
    const [popId, btnId] = pair;
    const pop = el(popId);
    const btn = el(btnId);
    if (!pop) return;
    const isCurrentlyOpen = galleryState.openPopover === kindKey;
    const nextOpen = typeof forceOpen === "boolean" ? forceOpen : !isCurrentlyOpen;
    closeAllGalleryPopovers(nextOpen ? kindKey : "");
    galleryState.openPopover = nextOpen ? kindKey : "";
    if (nextOpen) {
      if (kindKey === "folder") {
        galleryState.folderSearch = "";
        if (el("inp-gal-folder-search")) el("inp-gal-folder-search").value = "";
        renderGalleryFolderOptions();
      }
      if (btn) {
        const br = btn.getBoundingClientRect();
        const pane = el("view-gallery");
        const pr = pane ? pane.getBoundingClientRect() : { right: window.innerWidth };
        const maxRight = Math.min(window.innerWidth, pr.right || window.innerWidth);
        pop.classList.toggle("align-right", (br.left + 245) > (maxRight - 8));
      }
    }
    _setFloatingPopoverVisible(pop, btn, nextOpen);
  }

  function renderGalleryFolderOptions() {
    const listEl = el("gal-folder-list");
    const searchWrap = el("gal-folder-search-wrap");
    if (!listEl) return;
    listEl.innerHTML = "";

    const folders = Array.isArray(galleryState.folders) ? galleryState.folders : [];
    const kw = String(galleryState.folderSearch || "").trim().toLowerCase();
    if (searchWrap) {
      searchWrap.classList.toggle("hidden", folders.length < 6 && !kw);
    }

    const totalForKind = (galleryState.allItems || []).length;
    const allOpt = h("button", {
      type: "button",
      cls: `gal-pop-opt${galleryState.subfolder === "all" ? " active" : ""}`,
      data: { folder: "all" },
    },
      h("span", { cls: "gal-pop-opt-label", txt: "全部目录" }),
      h("span", { cls: "gal-pop-opt-count", txt: String(totalForKind) })
    );
    listEl.appendChild(allOpt);

    const filtered = kw
      ? folders.filter((f) => String(f.name || "").toLowerCase().includes(kw))
      : folders;

    for (const f of filtered) {
      const fid = String(f.id || f.subfolder || "__root__");
      const fname = String(f.name || "根目录");
      const isAct = galleryState.subfolder === fid;
      listEl.appendChild(
        h("button", {
          type: "button",
          cls: `gal-pop-opt${isAct ? " active" : ""}`,
          data: { folder: fid },
          title: fname,
        },
          h("span", { cls: "gal-pop-opt-label", txt: fname }),
          h("span", { cls: "gal-pop-opt-count", txt: String(Number(f.count || 0)) })
        )
      );
    }

    if (kw && filtered.length === 0) {
      listEl.appendChild(h("div", { cls: "hist-empty", txt: "无匹配子目录" }));
    }
  }

  function updateGalleryFilterControlsUI() {
    document.querySelectorAll("#gallery-filter-seg .seg-item").forEach((btn) => {
      const active = btn.dataset.kind === galleryState.kind;
      btn.classList.toggle("active", active);
      btn.setAttribute("aria-selected", String(active));
    });

    const mBtn = el("btn-gal-layout-masonry");
    const gBtn = el("btn-gal-layout-grid");
    if (mBtn) {
      const isM = galleryState.layout !== "grid";
      mBtn.classList.toggle("active", isM);
      mBtn.setAttribute("aria-pressed", String(isM));
    }
    if (gBtn) {
      const isG = galleryState.layout === "grid";
      gBtn.classList.toggle("active", isG);
      gBtn.setAttribute("aria-pressed", String(isG));
    }

    const fLbl = el("gal-folder-label");
    const fBtn = el("btn-gal-folder");
    if (fLbl) {
      fLbl.textContent = galleryState.subfolder === "all"
        ? "全部目录"
        : (galleryState.subfolder === "__root__" ? "根目录" : galleryState.subfolder);
    }
    if (fBtn) fBtn.classList.toggle("is-active", galleryState.subfolder !== "all");

    const dLbl = el("gal-date-label");
    const dBtn = el("btn-gal-date");
    if (dLbl) dLbl.textContent = GAL_DATE_LABELS[galleryState.dateRange] || "全部时间";
    if (dBtn) dBtn.classList.toggle("is-active", galleryState.dateRange !== "all");
    document.querySelectorAll("#gal-date-list .gal-pop-opt").forEach((b) => {
      b.classList.toggle("active", b.dataset.date === galleryState.dateRange);
    });

    const grpLbl = el("gal-group-label");
    const grpBtn = el("btn-gal-group");
    if (grpLbl) grpLbl.textContent = GAL_GROUP_LABELS[galleryState.groupBy] || "按日期分组";
    if (grpBtn) grpBtn.classList.toggle("is-active", galleryState.groupBy !== "none");
    document.querySelectorAll("#gal-group-list .gal-pop-opt").forEach((b) => {
      b.classList.toggle("active", b.dataset.group === galleryState.groupBy);
    });

    const sLbl = el("gal-sort-label");
    const sBtn = el("btn-gal-sort");
    if (sLbl) sLbl.textContent = GAL_SORT_LABELS[galleryState.sortBy] || "最新优先";
    if (sBtn) sBtn.classList.toggle("is-active", galleryState.sortBy !== "newest");
    document.querySelectorAll("#gal-sort-list .gal-pop-opt").forEach((b) => {
      b.classList.toggle("active", b.dataset.sort === galleryState.sortBy);
    });

    const hasCustomFilter = Boolean(
      galleryState.kind !== "all"
      || galleryState.subfolder !== "all"
      || galleryState.dateRange !== "all"
      || Boolean(String(galleryState.query || "").trim())
      || galleryState.sortBy !== "newest"
    );
    const resetBtn = el("btn-gal-reset-filter");
    if (resetBtn) resetBtn.classList.toggle("hidden", !hasCustomFilter);
  }

  function recomputeAndRenderGallery({ resetPage = false } = {}) {
    if (resetPage) galleryState.visibleLimit = galleryState.pageSize;

    const src = Array.isArray(galleryState.allItems) ? galleryState.allItems : [];
    const subF = String(galleryState.subfolder || "all");
    const dr = String(galleryState.dateRange || "all");
    const q = String(galleryState.query || "").trim().toLowerCase();

    const nowSec = Date.now() / 1000;
    let cutoffSec = null;
    if (dr === "today") {
      const midnight = new Date();
      midnight.setHours(0, 0, 0, 0);
      cutoffSec = midnight.getTime() / 1000;
    } else if (dr === "7d") {
      cutoffSec = nowSec - 7 * 86400;
    } else if (dr === "30d") {
      cutoffSec = nowSec - 30 * 86400;
    }

    const filtered = src.filter((it) => {
      if (galleryState.kind !== "all" && it.kind !== galleryState.kind) return false;
      if (subF !== "all") {
        const sub = String(it.subfolder || "");
        if (subF === "__root__") {
          if (sub !== "") return false;
        } else if (sub !== subF) {
          return false;
        }
      }
      if (cutoffSec !== null && Number(it.mtime_ts || 0) < cutoffSec) return false;
      if (q) {
        const nm = String(it.name || "").toLowerCase();
        const rp = String(it.rel_path || "").toLowerCase();
        if (!nm.includes(q) && !rp.includes(q)) return false;
      }
      return true;
    });

    const sortBy = galleryState.sortBy || "newest";
    filtered.sort((a, b) => {
      if (sortBy === "oldest") {
        return (Number(a.mtime_ts || 0) - Number(b.mtime_ts || 0))
          || String(a.rel_path || "").localeCompare(String(b.rel_path || ""));
      }
      if (sortBy === "name_asc") {
        return GAL_NAT_COLLATOR.compare(String(a.name || ""), String(b.name || ""))
          || String(a.rel_path || "").localeCompare(String(b.rel_path || ""));
      }
      if (sortBy === "name_desc") {
        return GAL_NAT_COLLATOR.compare(String(b.name || ""), String(a.name || ""))
          || String(b.rel_path || "").localeCompare(String(a.rel_path || ""));
      }
      if (sortBy === "size_desc") {
        return (Number(b.size_bytes || 0) - Number(a.size_bytes || 0))
          || (Number(b.mtime_ts || 0) - Number(a.mtime_ts || 0));
      }
      return (Number(b.mtime_ts || 0) - Number(a.mtime_ts || 0))
        || String(b.rel_path || "").localeCompare(String(a.rel_path || ""));
    });

    galleryState.items = filtered;
    updateGalleryFilterControlsUI();
    renderGalleryGrid();
  }

  function resetGalleryFilters() {
    const prevKind = galleryState.kind;
    galleryState.kind = "all";
    galleryState.subfolder = "all";
    galleryState.dateRange = "all";
    galleryState.sortBy = "newest";
    galleryState.query = "";
    if (el("inp-gal-search")) el("inp-gal-search").value = "";
    closeAllGalleryPopovers();
    if (prevKind !== "all") {
      loadGalleryItems("all");
    } else {
      recomputeAndRenderGallery({ resetPage: true });
    }
  }

  function applyGalleryListResponse(res) {
    galleryState.loaded = true;
    if (res && Array.isArray(res.items)) {
      galleryState.allItems = res.items;
      galleryState.counts = res.counts || galleryState.counts;
      galleryState.folders = Array.isArray(res.folders) ? res.folders : [];
    } else if (Array.isArray(res)) {
      galleryState.allItems = res;
      galleryState.folders = [];
    } else {
      galleryState.allItems = [];
      galleryState.folders = [];
    }
    if (galleryState.subfolder !== "all") {
      const validIds = new Set(galleryState.folders.map((f) => String(f.id || f.subfolder || "__root__")));
      if (!validIds.has(galleryState.subfolder)) {
        galleryState.subfolder = "all";
      }
    }
    if (galleryState.openPopover === "folder") {
      renderGalleryFolderOptions();
    }
    recomputeAndRenderGallery({ resetPage: true });
  }

  async function loadGalleryItems(kindOverride = null) {
    if (kindOverride !== null) galleryState.kind = kindOverride;
    updateGalleryFilterControlsUI();
    const res = await apiCall("list_gallery_items", { kind: galleryState.kind, limit: 1000 });
    applyGalleryListResponse(res);
  }

  function _buildGalleryCardNode(item) {
    const isSelected = Boolean(galleryState.selectedPath && galleryState.selectedPath === item.rel_path);
    const kindLabelMap = { image: "图像", video: "视频", audio: "语音" };
    let mediaNode;
    if (item.kind === "video") {
      mediaNode = h("video", {
        cls: "gallery-media-video",
        src: item.media_url,
        controls: true,
        preload: "metadata",
        playsInline: true,
      });
    } else if (item.kind === "audio") {
      const wave = document.createElementNS("http://www.w3.org/2000/svg", "svg");
      wave.setAttribute("viewBox", "0 0 180 38");
      wave.setAttribute("class", "audio-wave-svg");
      wave.setAttribute("aria-hidden", "true");
      const heights = [10, 18, 28, 14, 32, 22, 12, 26, 34, 18, 28, 16, 24, 12, 30, 20, 14, 22, 16, 10];
      heights.forEach((bh, idx) => {
        const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
        rect.setAttribute("x", String(idx * 9 + 2));
        rect.setAttribute("y", String(Math.round((38 - bh) / 2)));
        rect.setAttribute("width", "4");
        rect.setAttribute("height", String(bh));
        rect.setAttribute("rx", "2");
        rect.setAttribute("fill", "currentColor");
        rect.setAttribute("opacity", String(0.38 + (idx % 3) * 0.22));
        wave.appendChild(rect);
      });
      mediaNode = h("div", { cls: "gallery-audio-box" },
        wave,
        h("audio", {
          cls: "gallery-audio-player",
          src: item.media_url,
          controls: true,
          preload: "metadata",
        })
      );
    } else {
      mediaNode = h("img", {
        cls: "gallery-media-img",
        src: item.media_url,
        alt: item.name,
        loading: "lazy",
      });
      if (item.width && item.height && galleryState.layout !== "grid") {
        mediaNode.style.aspectRatio = `${item.width} / ${item.height}`;
      }
    }

    const folderLabel = item.subfolder ? String(item.subfolder) : "根目录";
    const timeLabel = _formatCardShortTime(item.mtime_str);
    return h("article", {
      cls: `gallery-card${isSelected ? " selected" : ""}`,
      data: { relPath: item.rel_path, kind: item.kind },
      on: {
        click: () => openGalleryInspector(item),
      },
    },
      mediaNode,
      h("div", { cls: "gallery-card-foot" },
        h("div", { cls: "gallery-card-row1" },
          h("span", { cls: "gallery-card-name", title: item.rel_path, txt: item.name }),
          h("span", { cls: "gallery-card-kind", txt: kindLabelMap[item.kind] || item.ext || "媒体" })
        ),
        h("div", { cls: "gallery-card-row2" },
          h("span", { cls: "gallery-card-folder", title: folderLabel, txt: folderLabel }),
          h("span", { cls: "gallery-card-time", txt: timeLabel })
        )
      )
    );
  }

  function renderGalleryGrid() {
    const masonry = el("gallery-masonry");
    const emptyEl = el("gallery-empty");
    const countEl = el("gallery-count");
    const navBadge = el("nav-count-gallery");
    const loadMoreWrap = el("gal-load-more-wrap");
    const loadMoreBtn = el("btn-gal-load-more");

    const items = galleryState.items || [];
    const allTotal = Number((galleryState.counts && galleryState.counts.all) || galleryState.allItems.length || items.length);
    const kindTotal = (galleryState.allItems || []).length;
    if (countEl) {
      countEl.textContent = (items.length !== kindTotal && kindTotal > 0)
        ? `${items.length} / ${kindTotal} 项`
        : `${items.length} 项作品`;
    }
    if (navBadge) {
      navBadge.textContent = String(allTotal);
      navBadge.classList.toggle("hidden", allTotal <= 0);
    }

    if (!masonry || !emptyEl) return;
    masonry.innerHTML = "";

    if (items.length === 0) {
      masonry.classList.add("hidden");
      if (loadMoreWrap) loadMoreWrap.classList.add("hidden");
      emptyEl.classList.remove("hidden");
      const eTitle = el("gallery-empty-title");
      const eDesc = el("gallery-empty-desc");
      if (allTotal > 0) {
        if (eTitle) eTitle.textContent = "当前筛选条件下暂无匹配作品";
        if (eDesc) eDesc.textContent = "可尝试更换子目录、时间范围或清空搜索关键词，也可以点击上方「重置筛选」";
      } else {
        if (eTitle) eTitle.textContent = "本地作品目录暂无媒体文件";
        if (eDesc) eDesc.textContent = "从云端拉回成图、视频或合成语音后，将在此按时间与目录自动归档并解析生成参数";
      }
      return;
    }

    masonry.classList.remove("hidden");
    emptyEl.classList.add("hidden");

    const visibleItems = items.slice(0, galleryState.visibleLimit);
    const hasMore = items.length > visibleItems.length;
    if (loadMoreWrap && loadMoreBtn) {
      loadMoreWrap.classList.toggle("hidden", !hasMore);
      if (hasMore) {
        loadMoreBtn.textContent = `展开更多作品 (剩余 ${items.length - visibleItems.length} 项)`;
      }
    }

    const colCount = _calcGalleryColCount();
    galleryState.lastColCount = colCount;

    const groupMode = galleryState.groupBy || "date";
    const todayStr = _localDateKey(new Date());
    const yestStr = _localDateKey(new Date(Date.now() - 86400000));
    const groups = [];
    const groupsByKey = new Map();

    if (groupMode === "none") {
      groups.push({ key: "__all__", title: "", items: visibleItems });
    } else {
      for (const item of visibleItems) {
        let gKey = "";
        let gTitle = "";
        if (groupMode === "folder") {
          gKey = item.subfolder || "__root__";
          gTitle = item.subfolder || "根目录";
        } else {
          const dStr = item.date_str || String(item.mtime_str || "").slice(0, 10) || "未知日期";
          gKey = dStr;
          if (dStr === todayStr) gTitle = `今天 · ${dStr.slice(5)}`;
          else if (dStr === yestStr) gTitle = `昨天 · ${dStr.slice(5)}`;
          else gTitle = dStr;
        }
        let grp = groupsByKey.get(gKey);
        if (!grp) {
          grp = { key: gKey, title: gTitle, items: [] };
          groupsByKey.set(gKey, grp);
          groups.push(grp);
        }
        grp.items.push(item);
      }
    }

    for (const grp of groups) {
      const sectionChildren = [];
      if (grp.title) {
        sectionChildren.push(
          h("div", { cls: "gallery-group-head" },
            h("span", { cls: "gallery-group-title", txt: grp.title }),
            h("span", { cls: "gallery-group-count", txt: `${grp.items.length}` }),
            h("span", { cls: "gallery-group-line" })
          )
        );
      }

      if (galleryState.layout === "grid") {
        const gridWrap = h("div", { cls: "gallery-uniform-grid" });
        for (const item of grp.items) {
          gridWrap.appendChild(_buildGalleryCardNode(item));
        }
        sectionChildren.push(gridWrap);
      } else {
        const useCols = Math.max(1, colCount);
        const colsWrap = h("div", { cls: "gallery-masonry-cols" });
        colsWrap.style.setProperty("--gal-cols", String(useCols));
        const colNodes = [];
        const colHeights = new Array(useCols).fill(0);
        for (let c = 0; c < useCols; c++) {
          const colDiv = h("div", { cls: "gallery-masonry-col" });
          colNodes.push(colDiv);
          colsWrap.appendChild(colDiv);
        }
        for (const item of grp.items) {
          let bestCol = 0;
          for (let c = 1; c < useCols; c++) {
            if (colHeights[c] < colHeights[bestCol]) bestCol = c;
          }
          colNodes[bestCol].appendChild(_buildGalleryCardNode(item));
          const ratio = (item.width && item.height)
            ? Math.min(2.2, Math.max(0.45, Number(item.height) / Number(item.width)))
            : (item.kind === "audio" ? 0.45 : 1.0);
          colHeights[bestCol] += ratio * 220 + 58;
        }
        sectionChildren.push(colsWrap);
      }

      masonry.appendChild(h("section", { cls: "gallery-group-section" }, sectionChildren));
    }
  }

  async function openGalleryInspector(item) {
    if (!item || !item.rel_path) return;
    galleryState.selectedPath = item.rel_path;
    document.querySelectorAll(".gallery-card").forEach((c) => {
      c.classList.toggle("selected", c.dataset.relPath === item.rel_path);
    });

    const panel = el("gallery-inspector");
    if (panel) {
      panel.classList.add("open");
      panel.removeAttribute("inert");
      panel.inert = false;
      panel.setAttribute("aria-hidden", "false");
    }

    const res = await apiCall("inspect_gallery_item", item.rel_path);
    const fullItem = (res && res.item) || item;
    const meta = (res && res.meta) || {};
    galleryState.inspectData = { item: fullItem, meta };
    renderInspectorPanel(fullItem, meta);
    renderCopilotChips();
  }

  function closeGalleryInspector() {
    galleryState.selectedPath = "";
    galleryState.inspectData = null;
    document.querySelectorAll(".gallery-card.selected").forEach((c) => c.classList.remove("selected"));
    const panel = el("gallery-inspector");
    if (!panel) return;
    if (panel.contains(document.activeElement) && document.activeElement.blur) {
      document.activeElement.blur();
    }
    panel.classList.remove("open");
    panel.setAttribute("aria-hidden", "true");
    panel.setAttribute("inert", "");
    panel.inert = true;
    renderCopilotChips();
  }

  function renderInspectorPanel(item, meta) {
    const previewEl = el("insp-preview");
    if (previewEl) {
      previewEl.innerHTML = "";
      if (item.kind === "video") {
        previewEl.appendChild(h("video", { src: item.media_url, controls: true, preload: "metadata", playsInline: true }));
      } else if (item.kind === "audio") {
        previewEl.appendChild(h("audio", { src: item.media_url, controls: true, preload: "metadata" }));
      } else {
        previewEl.appendChild(h("img", { src: item.media_url, alt: item.name }));
      }
    }

    if (el("insp-filename")) el("insp-filename").textContent = item.name || "--";

    const kvEl = el("insp-meta-list");
    if (kvEl) {
      kvEl.innerHTML = "";
      const dimTxt = (meta.width && meta.height) ? `${meta.width} × ${meta.height}` : "--";
      const rows = [
        ["尺寸", dimTxt],
        ["大小", formatBytes(item.size_bytes)],
        ["修改时间", item.mtime_str || "--"],
        ["子目录", item.subfolder || "根目录"],
      ];
      for (const [k, v] of rows) {
        kvEl.appendChild(h("span", { cls: "insp-k", txt: k }));
        kvEl.appendChild(h("span", { cls: "insp-v tabular", txt: String(v) }));
      }
    }

    const genSec = el("insp-gen-section");
    const hasAnyGen = Boolean(
      meta.positive_prompt || meta.negative_prompt || meta.seed != null || meta.steps != null
      || meta.cfg != null || meta.sampler || meta.model || (Array.isArray(meta.loras) && meta.loras.length > 0)
    );
    if (genSec) genSec.classList.toggle("hidden", !hasAnyGen);

    if (el("insp-pos-prompt")) el("insp-pos-prompt").textContent = meta.positive_prompt || "未记录正向提示词";
    if (el("insp-neg-wrap")) el("insp-neg-wrap").classList.toggle("hidden", !meta.negative_prompt);
    if (el("insp-neg-prompt")) el("insp-neg-prompt").textContent = meta.negative_prompt || "";

    const paramsGrid = el("insp-params-grid");
    if (paramsGrid) {
      paramsGrid.innerHTML = "";
      const pairs = [
        ["Seed", meta.seed != null ? String(meta.seed) : "--"],
        ["Steps", meta.steps != null ? String(meta.steps) : "--"],
        ["CFG", meta.cfg != null ? String(meta.cfg) : "--"],
        ["Sampler", meta.sampler || "--"],
        ["Scheduler", meta.scheduler || "--"],
        ["Model", meta.model || "--"],
      ];
      for (const [pk, pv] of pairs) {
        paramsGrid.appendChild(h("div", { cls: "insp-param-cell" },
          h("span", { cls: "insp-param-k", txt: pk }),
          h("span", { cls: "insp-param-v tabular", title: pv, txt: pv })
        ));
      }
    }

    const loraWrap = el("insp-loras-wrap");
    const loraList = el("insp-loras");
    const loras = Array.isArray(meta.loras) ? meta.loras : [];
    if (loraWrap && loraList) {
      loraWrap.classList.toggle("hidden", loras.length === 0);
      loraList.innerHTML = "";
      for (const lr of loras) {
        const w = lr.strength != null ? ` (${lr.strength})` : "";
        loraList.appendChild(h("span", { cls: "pill-tag accent tabular", txt: `${lr.name}${w}` }));
      }
    }

    const copyBtn = el("btn-insp-copy-prompt");
    if (copyBtn) copyBtn.disabled = !meta.positive_prompt && !meta.negative_prompt;
    const pushBtn = el("btn-insp-push-canvas");
    if (pushBtn) pushBtn.disabled = !meta.has_workflow;
  }

  /* ----------------------------------------------------------------
     视图 3：引擎与云盘（热调参 + gdrive_save + 折叠配置）
     ---------------------------------------------------------------- */
  function resolveLiveLlmCtx(snap) {
    const s = snap || latestSnap || {};
    const c = s.comfy || {};
    const prefs = s.prefs || {};
    const actProfs = Array.isArray(c.active_profiles) ? c.active_profiles : [];
    const prefProfs = Array.isArray(prefs.boot_profiles) ? prefs.boot_profiles : bootState.profiles;
    const is27b = (
      actProfs.includes("qwen_27b")
      || String(c.model_name || "").includes("27B")
      || (prefProfs.includes("qwen_27b") && !prefProfs.includes("qwen_llm"))
    );
    const quant = String(
      c.quant || (is27b ? prefs.boot_llm27_quant : prefs.boot_llm_quant) || "UD-Q4_K_XL"
    );
    const ctx = Number(
      c.ctx_size || (is27b ? prefs.boot_llm27_ctx : prefs.boot_llm_ctx) || (is27b ? 65536 : 32768)
    );
    const mmproj = is27b
      ? (prefs.boot_llm27_mmproj !== undefined ? Boolean(prefs.boot_llm27_mmproj) : true)
      : (prefs.boot_llm_mmproj !== undefined ? Boolean(prefs.boot_llm_mmproj) : true);
    return { is27b, quant, ctx, mmproj };
  }

  function renderTunerView(snap) {
    const s = snap || latestSnap || {};
    const c = s.comfy || {};
    const ob = s.onboarding || {};
    const prefs = s.prefs || {};
    const cluster = s.cluster || {};
    const slots = cluster.slots || {};

    const live = resolveLiveLlmCtx(s);
    if (!tunerState.touched) {
      tunerState.ctx = live.ctx;
      tunerState.quant = live.quant;
      tunerState.mmproj = live.mmproj;
    }

    const llmOn = Boolean(c.llm_online || (c.online && c.service_kind === "llm") || (slots.llm && slots.llm.online));
    const stPill = el("tuner-llm-status");
    if (stPill) {
      stPill.className = `pill-tag${llmOn ? " ok" : ""}`;
      stPill.textContent = llmOn ? (live.is27b ? "27B 在线" : "180B 在线") : "待机预览";
    }
    if (el("tuner-llm-mode")) {
      el("tuner-llm-mode").textContent = live.is27b
        ? "当前目标: Qwen3.8-27B · 不断卡热重载"
        : "当前目标: Qwen3.8-180B · 不断卡热重载";
    }

    document.querySelectorAll("#tuner-ctx-seg .seg-item").forEach((btn) => {
      btn.classList.toggle("active", Number(btn.dataset.ctx) === Number(tunerState.ctx));
    });
    document.querySelectorAll("#tuner-quant-seg .seg-item").forEach((btn) => {
      btn.classList.toggle("active", String(btn.dataset.quant) === String(tunerState.quant));
    });
    const mmSwitch = el("btn-tuner-mmproj");
    if (mmSwitch) {
      mmSwitch.classList.toggle("on", Boolean(tunerState.mmproj));
      mmSwitch.setAttribute("aria-checked", String(Boolean(tunerState.mmproj)));
    }

    const llmUrl = (slots.llm && slots.llm.tunnel_url) || c.llm_tunnel_url || (c.service_kind === "llm" ? c.tunnel_url : "");
    if (el("tuner-api-url")) {
      el("tuner-api-url").textContent = llmUrl ? `${String(llmUrl).replace(/\/+$/, "")}/v1` : "开机后自动分配公网隧道";
    }
    if (el("tuner-api-model")) {
      el("tuner-api-model").textContent = c.model_name || (live.is27b ? "Qwen3.8-27B" : "Qwen3.8-180B");
    }

    // 云盘绑定状态
    const gBadge = el("tuner-gdrive-badge");
    const gBindBtn = el("btn-tuner-gdrive-bind");
    if (gBadge && gBindBtn) {
      if (ob.gdrive_binding) {
        gBadge.className = "pill-tag accent";
        gBadge.textContent = "等待授权";
        gBindBtn.textContent = "取消等待";
      } else if (ob.gdrive_ok) {
        const freeGb = ob.gdrive_free_gb;
        gBadge.className = "pill-tag ok";
        gBadge.textContent = typeof freeGb === "number" ? `余 ${freeGb.toFixed(1)} GB` : "已连通";
        gBindBtn.textContent = "重绑云盘";
      } else {
        gBadge.className = "pill-tag warn";
        gBadge.textContent = "未绑定";
        gBindBtn.textContent = "一键绑定";
      }
    }

    renderCloudAssetsList();

    // Cloudflare 折叠区
    const cf = ob.cf_tunnel || {};
    const cfBadge = el("tuner-cf-badge");
    if (cfBadge) {
      const isNamed = cf.mode === "named";
      cfBadge.className = `pill-tag${isNamed ? " ok" : ""}`;
      cfBadge.textContent = isNamed ? "固定域名已启用" : "免费临时隧道";
    }
    if (el("inp-cf-comfy") && document.activeElement !== el("inp-cf-comfy")) {
      el("inp-cf-comfy").value = cf.comfy_url || "";
    }
    if (el("inp-cf-llm") && document.activeElement !== el("inp-cf-llm")) {
      el("inp-cf-llm").value = cf.llm_url || "";
    }

    // 闲置停卡策略同步
    const kaMode = prefs.keep_alive === false
      ? "off"
      : (Number(prefs.idle_auto_stop_minutes ?? 60) > 0 ? "stop60" : "smart");
    document.querySelectorAll("#pref-keepalive-seg .seg-item").forEach((b) => {
      b.classList.toggle("active", b.dataset.val === kaMode);
    });
    const archPol = String(prefs.idle_archive_fail_policy || "stop_billing") === "keep_alive" ? "keep_alive" : "stop_billing";
    document.querySelectorAll("#pref-archive-seg .seg-item").forEach((b) => {
      b.classList.toggle("active", b.dataset.val === archPol);
    });
    const autoPull = prefs.auto_pull_outputs === false ? "0" : "1";
    document.querySelectorAll("#pref-autopull-seg .seg-item").forEach((b) => {
      b.classList.toggle("active", b.dataset.val === autoPull);
    });
    document.querySelectorAll("#pref-longrun-seg .seg-item").forEach((b) => {
      b.classList.toggle("active", Number(b.dataset.val) === Number(prefs.long_run_warn_minutes || 180));
    });
    document.querySelectorAll("#pref-poll-seg .seg-item").forEach((b) => {
      b.classList.toggle("active", Number(b.dataset.val) === Number(prefs.poll_interval_sec || 60));
    });
    if (el("tuner-idle-summary")) {
      const kaTxt = kaMode === "stop60" ? "智能保活 + 闲置 1h 停卡" : (kaMode === "smart" ? "智能保活" : "关闭保活");
      const apTxt = autoPull === "1" ? "自动拉图" : "手动拉图";
      el("tuner-idle-summary").textContent = `${kaTxt} · ${apTxt}`;
    }

    // 本地工具路径
    const toolsBadge = el("tuner-tools-badge");
    const toolsList = el("tuner-tools-list");
    const tp = ob.tool_paths || {};
    const cliOk = Boolean(ob.colab_cli_ok);
    const rcOk = Boolean(ob.rclone_exe_ok);
    if (toolsBadge) {
      toolsBadge.className = `pill-tag${cliOk && rcOk ? " ok" : " warn"}`;
      toolsBadge.textContent = cliOk && rcOk ? "2/2 就绪" : "待定位";
    }
    if (toolsList) {
      toolsList.innerHTML = "";
      const rows = [
        { kind: "colab_py", label: "colab-cli Python", val: tp.colab_py },
        { kind: "colab_exe", label: "colab.exe", val: tp.colab_exe },
        { kind: "rclone_exe", label: "rclone.exe", val: tp.rclone_exe },
        { kind: "rclone_conf", label: "rclone.conf", val: tp.rclone_conf },
      ];
      for (const row of rows) {
        toolsList.appendChild(h("div", { cls: "inset-row" },
          h("div", { cls: "acct-row-info" },
            h("span", { cls: "form-label", txt: row.label }),
            h("span", { cls: "metric-sub mono", title: row.val || "", txt: row.val || "自动探测中 / 未设置" })
          ),
          h("button", {
            type: "button",
            cls: "btn btn-xs btn-subtle",
            txt: "定位",
            on: {
              click: async () => {
                const res = await apiCall("pick_tool_binary", row.kind);
                if (!res || res.cancelled) return;
                if (res.snapshot) window.renderStudioState(res.snapshot);
                if (res.message) showToast(res.message, { isError: res.ok === false });
              },
            },
          })
        ));
      }
    }
  }

  async function loadCloudAssets(notify = true) {
    const btn = el("btn-tuner-cloud-refresh");
    if (btn) btn.disabled = true;
    try {
      const res = await apiCall("list_cloud_assets");
      tunerState.cloudLoaded = true;
      if (res && typeof res === "object") {
        tunerState.cloudData = res;
        if (notify && res.message) showToast(res.message, { isError: res.ok === false });
      }
      renderCloudAssetsList();
    } finally {
      if (btn) btn.disabled = false;
    }
  }

  function renderCloudAssetsList() {
    document.querySelectorAll("#tuner-cloud-seg .seg-item").forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.bucket === tunerState.cloudBucket);
    });
    const listEl = el("tuner-cloud-list");
    if (!listEl) return;
    listEl.innerHTML = "";

    const data = tunerState.cloudData || {};
    const items = Array.isArray(data[tunerState.cloudBucket]) ? data[tunerState.cloudBucket] : [];

    if (items.length === 0) {
      listEl.appendChild(h("div", {
        cls: "cloud-empty-row",
        txt: cleanText(data.message || "该云盘分类下暂无文件"),
      }));
      return;
    }

    for (const item of items) {
      const modStr = item.mod_time ? String(item.mod_time).slice(0, 16).replace("T", " ") : "";
      listEl.appendChild(h("div", { cls: "inset-row" },
        h("div", { cls: "acct-row-info" },
          h("span", { cls: "form-label", title: item.path || item.name, txt: item.name }),
          h("span", { cls: "metric-sub tabular", txt: modStr || item.path || "--" })
        ),
        h("span", { cls: "pill-tag tabular", txt: formatBytes(item.size_bytes) })
      ));
    }
  }

  /* ----------------------------------------------------------------
     视图 4：控制台与体检（三端实时日志 + Doctor 清单）
     ---------------------------------------------------------------- */
  async function loadLiveLogs() {
    const res = await apiCall("tail_live_logs", { source: consoleState.source, lines: 160 });
    consoleState.logsLoaded = true;
    if (res && res.ok) {
      consoleState.lines = Array.isArray(res.lines) ? res.lines : [];
      consoleState.updatedStr = res.updated_str || "--:--:--";
      consoleState.logPath = res.path || "";
      consoleState.errorMsg = "";
      if (rightLogState.source === "all" && consoleState.source === "local_diag") {
        rightLogState.lines = consoleState.lines.slice(-120);
        renderRightLogPanel();
      }
    } else {
      consoleState.lines = [];
      consoleState.updatedStr = (res && res.updated_str) || "--:--:--";
      consoleState.errorMsg = (res && res.message) || "暂无可读取的日志内容";
    }
    renderConsoleLogs();
  }

  function _formatLogNowHhMmSs() {
    const d = new Date();
    return [d.getHours(), d.getMinutes(), d.getSeconds()]
      .map((n) => String(n).padStart(2, "0"))
      .join(":");
  }

  function _classifyRightLogLevel(text, explicitLevel = "") {
    if (explicitLevel) return explicitLevel;
    const low = String(text || "").toLowerCase();
    if (low.includes("error") || low.includes("fail") || low.includes("traceback") || low.includes("exception") || low.includes("异常") || low.includes("失败")) {
      return "rlog-err";
    }
    if (low.includes("warn") || low.includes("紧缺") || low.includes("超时")) {
      return "rlog-warn";
    }
    if (low.includes("就绪") || low.includes("完成") || low.includes("成功") || low.includes("已连接")) {
      return "rlog-ok";
    }
    return "";
  }

  function appendRightLogEvent(rawText, level = "", src = "all") {
    const msg = stripEmoji(rawText).trim();
    if (!msg) return;
    const prev = rightLogState.liveEvents[rightLogState.liveEvents.length - 1];
    const nowTs = Date.now();
    if (prev && prev.msg === msg && prev.src === src && (nowTs - prev.ts) < 2000) {
      return;
    }
    rightLogState.liveEvents.push({
      ts: nowTs,
      time: _formatLogNowHhMmSs(),
      msg,
      level: level ? `rlog-${level.replace(/^rlog-/, "")}` : _classifyRightLogLevel(msg),
      src: src || "all",
    });
    if (rightLogState.liveEvents.length > 100) {
      rightLogState.liveEvents.splice(0, rightLogState.liveEvents.length - 100);
    }
  }

  async function loadRightPanelLogs() {
    if (rightLogState.loading) return;
    rightLogState.loading = true;
    const srcMap = { all: "local_diag", comfy: "comfy", llm: "llama" };
    const backendSrc = srcMap[rightLogState.source] || "local_diag";
    try {
      const res = await apiCall("tail_live_logs", { source: backendSrc, lines: 120 });
      if (res && res.ok && Array.isArray(res.lines)) {
        rightLogState.lines = res.lines.slice(-120);
      }
    } finally {
      rightLogState.loading = false;
      renderRightLogPanel();
    }
  }

  function renderRightLogPanel() {
    document.querySelectorAll("#rlog-source-seg .rlog-seg-btn").forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.rlogSrc === rightLogState.source);
    });

    const s = latestSnap || {};
    const deploy = s.deploy || {};
    const comfy = s.comfy || {};
    const cluster = s.cluster || {};
    const cSlots = cluster.slots || {};
    const dot = el("rlog-live-dot");
    const liveTxt = el("rlog-live-text");
    const taskTxt = el("rlog-task-text");
    const progFill = el("rlog-progress-fill");

    let dotCls = "rlog-live-dot";
    let statusLbl = "实时就绪";
    let taskLbl = "系统待命 · 等待日志流";
    let pct = 0;

    if (deploy.active) {
      dotCls = "rlog-live-dot is-busy";
      statusLbl = "部署直播";
      pct = Math.max(4, Math.min(100, Number(deploy.pct || 0)));
      taskLbl = cleanText(deploy.sub_detail || deploy.stage_title || `云端部署中 ${Math.round(pct)}%`);
    } else if (Number(comfy.running_count || 0) > 0 || Number(comfy.pending_count || 0) > 0) {
      dotCls = "rlog-live-dot is-live";
      statusLbl = "推理生成中";
      pct = Math.max(0, Math.min(100, Number(comfy.step_pct || 0)));
      taskLbl = cleanText(
        comfy.current_node
          ? `${comfy.current_node} · ${Math.round(pct)}%`
          : `队列 ${Number(comfy.running_count || 0)} 运行 / ${Number(comfy.pending_count || 0)} 等待`
      );
    } else if (s.is_running || cluster.active) {
      const tunnelReady = Boolean(
        comfy.online
        || comfy.comfy_online
        || comfy.llm_online
        || (cluster.active && ((cSlots.comfy && cSlots.comfy.online) || (cSlots.llm && cSlots.llm.online)))
      );
      dotCls = tunnelReady ? "rlog-live-dot is-live" : "rlog-live-dot is-busy";
      statusLbl = "实例在线";
      taskLbl = tunnelReady ? "隧道已连通 · 等待任务触发" : "实例在线 · 等待隧道连通";
    }

    if (dot) dot.className = dotCls;
    if (liveTxt) liveTxt.textContent = statusLbl;
    if (taskTxt) taskTxt.textContent = taskLbl;
    if (progFill) progFill.style.width = `${pct}%`;

    const term = el("rlog-terminal");
    if (!term) return;

    const entries = [];
    for (const raw of (rightLogState.lines || [])) {
      const cleaned = stripEmoji(raw).trim();
      if (!cleaned) continue;
      let tStr = "--:--:--";
      let body = cleaned;
      const m = cleaned.match(/^\[?(?:\d{4}-\d{2}-\d{2}\s+)?(\d{2}:\d{2}:\d{2})(?:\.\d+)?\]?\s*(.*)$/);
      if (m && m[2]) {
        tStr = m[1];
        body = m[2];
      }
      entries.push({ time: tStr, msg: body, level: _classifyRightLogLevel(body) });
    }
    for (const ev of (rightLogState.liveEvents || [])) {
      if (rightLogState.source !== "all" && ev.src !== "all" && ev.src !== rightLogState.source) {
        continue;
      }
      entries.push({ time: ev.time, msg: ev.msg, level: ev.level });
    }

    term.innerHTML = "";
    const sliced = entries.slice(-140);
    if (sliced.length === 0) {
      term.appendChild(h("div", { cls: "rlog-line" },
        h("span", { cls: "rlog-time", txt: "--:--:--" }),
        h("span", { cls: "rlog-msg", txt: "等待连接集群与实时日志流..." })
      ));
      return;
    }
    for (const row of sliced) {
      term.appendChild(h("div", { cls: `rlog-line${row.level ? " " + row.level : ""}` },
        h("span", { cls: "rlog-time", txt: row.time }),
        h("span", { cls: "rlog-msg", txt: row.msg })
      ));
    }
    term.scrollTop = term.scrollHeight;
  }

  function renderConsoleLogs() {
    const titleMap = {
      local_diag: "本地诊断日志",
      llama: "远端 LLM 推理日志",
      comfy: "远端 ComfyUI 画布日志",
    };
    document.querySelectorAll("#console-source-seg .seg-item").forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.source === consoleState.source);
    });
    if (el("console-log-title")) el("console-log-title").textContent = titleMap[consoleState.source] || "诊断日志";
    if (el("console-updated-at")) el("console-updated-at").textContent = consoleState.updatedStr;

    const viewEl = el("console-log-view");
    if (!viewEl) return;
    viewEl.innerHTML = "";

    const kw = String(consoleState.filter || "").trim().toLowerCase();
    const rawLines = consoleState.lines || [];
    const filtered = kw ? rawLines.filter((ln) => String(ln).toLowerCase().includes(kw)) : rawLines;

    if (el("console-line-count")) el("console-line-count").textContent = `${filtered.length} 行`;

    if (filtered.length === 0) {
      viewEl.appendChild(h("div", {
        cls: "cloud-empty-row",
        txt: cleanText(consoleState.errorMsg || (kw ? "无匹配关键词的日志记录" : "当前日志流为空")),
      }));
      return;
    }

    filtered.forEach((line, idx) => {
      const txt = String(line || "");
      const low = txt.toLowerCase();
      let lvlCls = "lvl-info";
      if (low.includes("error") || low.includes("fail") || low.includes("traceback") || low.includes("exception")) {
        lvlCls = "lvl-err";
      } else if (low.includes("warn")) {
        lvlCls = "lvl-warn";
      }
      viewEl.appendChild(h("div", { cls: `log-line ${lvlCls}` },
        h("span", { cls: "log-ln tabular", txt: String(idx + 1).padStart(3, "0") }),
        h("span", { cls: "log-txt", txt })
      ));
    });
    viewEl.scrollTop = viewEl.scrollHeight;
  }

  async function runDoctorPreflight(deep = true) {
    const btn = el("btn-console-doctor");
    if (btn) btn.disabled = true;
    try {
      const res = hasApi("run_preflight")
        ? await apiCall("run_preflight", Boolean(deep))
        : await apiCall("probe_environment_deep");
      if (res && Array.isArray(res.items)) {
        consoleState.doctorReport = res;
        renderDoctorChecklist(res);
        if (deep) {
          showToast(res.ok ? "深度体检完成，全部核心项正常" : (res.blocking_message || "体检发现需要处理的阻塞项"), {
            isError: !res.ok,
          });
        }
      } else if (res && res.snapshot) {
        window.renderStudioState(res.snapshot);
        if (deep && res.message) showToast(res.message);
      }
    } finally {
      if (btn) btn.disabled = false;
    }
  }

  function renderDoctorChecklist(report) {
    const pill = el("doctor-summary-pill");
    const verEl = el("doctor-versions");
    const listEl = el("doctor-items-list");
    if (!listEl) return;
    listEl.innerHTML = "";

    if (!report || !Array.isArray(report.items)) {
      if (pill) {
        pill.className = "pill-tag";
        pill.textContent = "待执行";
      }
      listEl.appendChild(h("div", { cls: "cloud-empty-row", txt: "点击「环境深度体检 (Doctor)」开始检查本机组件与账号配置" }));
      return;
    }

    const okCnt = Number(report.ok_count || 0);
    const warnCnt = Number(report.warn_count || 0);
    const failCnt = Number(report.fail_count || 0);
    if (pill) {
      pill.className = `pill-tag ${failCnt > 0 ? "danger" : (warnCnt > 0 ? "warn" : "ok")}`;
      pill.textContent = `通过 ${okCnt} · 警告 ${warnCnt} · 阻塞 ${failCnt}`;
    }

    if (verEl) {
      verEl.innerHTML = "";
      const versions = report.versions || {};
      for (const [vk, vv] of Object.entries(versions)) {
        if (!vv) continue;
        verEl.appendChild(h("span", { cls: "pill-tag", txt: `${vk} ${vv}` }));
      }
    }

    for (const item of report.items) {
      const st = String(item.status || "ok").toLowerCase();
      const iconKind = st === "fail" ? "xCircle" : (st === "warn" ? "alertTriangle" : "checkCircle");
      listEl.appendChild(h("div", { cls: "doctor-item" },
        createSvgIcon(iconKind, 15, `icon-svg doctor-icon ${st}`),
        h("div", { cls: "doctor-content" },
          h("div", { cls: "doctor-top" },
            h("span", { cls: "doctor-name", txt: cleanText(item.name) }),
            h("span", { cls: "doctor-cat", txt: cleanText(item.category) })
          ),
          h("div", { cls: "doctor-summary", txt: cleanText(item.summary) }),
          item.detail ? h("div", { cls: "metric-sub mono", title: item.detail, txt: cleanText(item.detail) }) : null,
          item.fix_hint && st !== "ok" ? h("div", { cls: "doctor-hint", txt: `建议: ${cleanText(item.fix_hint)}` }) : null
        )
      ));
    }
  }

  /* ----------------------------------------------------------------
     右侧 AI 副驾抽屉、安全富文本渲染与流式事件闭环
     ---------------------------------------------------------------- */
  function resizeCopilotInput() {
    const inp = el("copilot-input");
    if (!inp) return;
    inp.style.height = "auto";
    const nextH = Math.max(44, Math.min(136, inp.scrollHeight || 44));
    inp.style.height = `${nextH}px`;
  }

  function isCopilotNearBottom() {
    const box = el("copilot-messages");
    if (!box) return true;
    return (box.scrollHeight - box.scrollTop - box.clientHeight) <= 56;
  }

  function updateCopilotScrollBtn() {
    const btn = el("btn-copilot-scroll-bottom");
    const box = el("copilot-messages");
    if (!btn || !box) return;
    const anyPop = Boolean(
      copilotState.popoverOpen
      || copilotState.effortPopoverOpen
      || copilotState.plusPopoverOpen
      || copilotState.historyPopoverOpen
    );
    const dist = box.scrollHeight - box.scrollTop - box.clientHeight;
    const hasScroll = box.scrollHeight > box.clientHeight + 24;
    const show = !anyPop && hasScroll && dist > 72 && Array.isArray(copilotState.items) && copilotState.items.length > 0;
    btn.classList.toggle("hidden", !show);
  }

  function scrollCopilotBottom(force = false) {
    const box = el("copilot-messages");
    if (!box) return;
    if (force) {
      copilotState.stickToBottom = true;
    } else if (!copilotState.stickToBottom) {
      updateCopilotScrollBtn();
      return;
    }
    box.scrollTop = box.scrollHeight;
    if (force) {
      requestAnimationFrame(() => {
        const b2 = el("copilot-messages");
        if (b2 && copilotState.stickToBottom) {
          b2.scrollTop = b2.scrollHeight;
        }
        updateCopilotScrollBtn();
      });
    } else {
      updateCopilotScrollBtn();
    }
  }

  /* 安全解析内联语法（`code`、**bold**、换行），绝不使用 innerHTML */
  function appendInlineRichNodes(parent, rawLine) {
    const line = String(rawLine || "");
    const tokenRe = /(`[^`\n]+`|\*\*[^*\n]+\*\*)/g;
    let lastIdx = 0;
    let match;
    while ((match = tokenRe.exec(line)) !== null) {
      if (match.index > lastIdx) {
        parent.appendChild(document.createTextNode(line.slice(lastIdx, match.index)));
      }
      const tok = match[0];
      if (tok.startsWith("`") && tok.endsWith("`")) {
        parent.appendChild(h("code", { cls: "copilot-inline-code", txt: tok.slice(1, -1) }));
      } else if (tok.startsWith("**") && tok.endsWith("**")) {
        parent.appendChild(h("strong", { txt: tok.slice(2, -2) }));
      } else {
        parent.appendChild(document.createTextNode(tok));
      }
      lastIdx = tokenRe.lastIndex;
    }
    if (lastIdx < line.length) {
      parent.appendChild(document.createTextNode(line.slice(lastIdx)));
    }
  }

  /* 安全渲染轻量 Markdown 富文本（段落、列表、围栏代码块、行内代码、流式光标） */
  function renderCopilotRichText(container, rawText, isStreaming = false) {
    container.innerHTML = "";
    const text = stripEmoji(rawText).replace(/\r\n?/g, "\n");
    if (!text && !isStreaming) return;

    const segments = [];
    const fenceRe = /```[a-zA-Z0-9_-]*\n?([\s\S]*?)```/g;
    let last = 0;
    let m;
    while ((m = fenceRe.exec(text)) !== null) {
      if (m.index > last) {
        segments.push({ type: "text", content: text.slice(last, m.index) });
      }
      segments.push({ type: "code", content: m[1].replace(/\n$/, "") });
      last = fenceRe.lastIndex;
    }
    if (last < text.length) {
      const tail = text.slice(last);
      const openIdx = tail.indexOf("```");
      if (openIdx >= 0) {
        if (openIdx > 0) segments.push({ type: "text", content: tail.slice(0, openIdx) });
        const afterFence = tail.slice(openIdx + 3).replace(/^[a-zA-Z0-9_-]*\n?/, "");
        segments.push({ type: "code", content: afterFence });
      } else {
        segments.push({ type: "text", content: tail });
      }
    }

    let lastBlockNode = null;
    for (const seg of segments) {
      if (seg.type === "code") {
        const pre = h("pre", { cls: "copilot-code-block" }, h("code", { txt: seg.content }));
        container.appendChild(pre);
        lastBlockNode = null;
        continue;
      }
      const blocks = seg.content.split(/\n{2,}/);
      for (const blk of blocks) {
        const trimmed = blk.replace(/^\n+|\n+$/g, "");
        if (!trimmed) continue;
        const lines = trimmed.split("\n");
        const isList = lines.length > 0 && lines.every((ln) => /^\s*[-*]\s+/.test(ln));
        if (isList) {
          const ul = h("ul", { cls: "copilot-list" });
          for (const ln of lines) {
            const li = h("li");
            appendInlineRichNodes(li, ln.replace(/^\s*[-*]\s+/, ""));
            ul.appendChild(li);
          }
          container.appendChild(ul);
          lastBlockNode = ul.lastElementChild;
        } else {
          const p = h("p", { cls: "copilot-p" });
          lines.forEach((ln, idx) => {
            if (idx > 0) p.appendChild(document.createElement("br"));
            appendInlineRichNodes(p, ln);
          });
          container.appendChild(p);
          lastBlockNode = p;
        }
      }
    }

    if (isStreaming) {
      const caret = h("span", { cls: "copilot-stream-caret", attrs: { "aria-hidden": "true" } });
      if (lastBlockNode) lastBlockNode.appendChild(caret);
      else container.appendChild(caret);
    }
  }

  function formatToolArgsSummary(args) {
    if (!args || typeof args !== "object") return cleanText(String(args || ""));
    const entries = Object.entries(args).filter(([, v]) => v !== undefined && v !== null && v !== "");
    if (entries.length === 0) return "";
    return entries
      .slice(0, 3)
      .map(([k, v]) => {
        const valStr = typeof v === "string" ? v : JSON.stringify(v);
        const clipped = valStr.length > 44 ? `${valStr.slice(0, 44)}...` : valStr;
        return `${k}: ${cleanText(clipped)}`;
      })
      .join(" · ");
  }

  function formatToolDetailJson(obj) {
    if (obj === undefined || obj === null || obj === "") return "";
    if (typeof obj === "string") return stripEmoji(obj);
    try {
      return stripEmoji(JSON.stringify(obj, null, 2));
    } catch (_) {
      return stripEmoji(String(obj));
    }
  }

  function renderCopilotToolCard(item) {
    const label = AGENT_TOOL_LABELS[item.name] || cleanText(item.name || "工具调用");
    const statusText = item.status === "running"
      ? "执行中"
      : (item.status === "error"
        ? "已拦截/失败"
        : (item.elapsedMs != null ? `${Math.round(Number(item.elapsedMs))}ms` : "已完成"));
    const tagCls = item.status === "running" ? "accent" : (item.status === "error" ? "danger" : "ok");
    const dotCls = item.status === "running" ? "running" : (item.status === "error" ? "error" : "ok");
    const summaryLine = cleanText(item.summary || "") || formatToolArgsSummary(item.args);

    const detailBlocks = [];
    if (item.name) {
      detailBlocks.push(h("div", { cls: "cfg-label", txt: `工具标识: ${cleanText(item.name)}` }));
    }
    const argsTxt = formatToolDetailJson(item.args);
    if (argsTxt) {
      detailBlocks.push(h("div", { cls: "cfg-label", txt: "调用参数" }));
      detailBlocks.push(h("pre", { cls: "copilot-tool-pre", txt: argsTxt }));
    }
    const resTxt = formatToolDetailJson(item.detail || item.summary);
    if (resTxt) {
      detailBlocks.push(h("div", { cls: "cfg-label", txt: "执行结果" }));
      detailBlocks.push(h("pre", { cls: "copilot-tool-pre", txt: resTxt }));
    }
    if (detailBlocks.length === 0) {
      detailBlocks.push(h("span", { cls: "metric-sub", txt: "等待执行返回..." }));
    }

    const details = h("details", { cls: "copilot-tool-card" },
      h("summary", { cls: "copilot-tool-summary" },
        h("div", { cls: "copilot-tool-left" },
          h("span", { cls: `tool-status-dot ${dotCls}` }),
          h("span", { cls: "tool-title", txt: label }),
          item.name ? h("code", { cls: "tool-raw-name", txt: cleanText(item.name) }) : null
        ),
        h("div", { cls: "copilot-tool-right" },
          h("span", { cls: `pill-tag ${tagCls} tabular`, txt: statusText }),
          createSvgIcon("chevronRight", 12, "icon-svg details-chevron")
        )
      ),
      summaryLine ? h("div", { cls: "copilot-tool-arg-line", title: summaryLine, txt: summaryLine }) : null,
      h("div", { cls: "copilot-tool-detail" }, detailBlocks)
    );
    item._node = details;
    return details;
  }

  function updateCopilotToolCardNode(item) {
    const oldNode = item && item._node;
    const parent = oldNode && oldNode.parentNode;
    if (!parent) return;
    const wasOpen = Boolean(oldNode.open);
    const replacement = renderCopilotToolCard(item);
    replacement.open = wasOpen;
    parent.replaceChild(replacement, oldNode);
  }

  function replaceCopilotConfirmCardNode(item) {
    const oldNode = item && item._node;
    const parent = oldNode && oldNode.parentNode;
    if (!parent) return;
    const nextCard = renderCopilotConfirmCard(item);
    parent.replaceChild(nextCard, oldNode);
  }

  function renderCopilotConfirmCard(item) {
    const resolvedState = item.resolvedState || "";
    const cardCls = `copilot-confirm-card${resolvedState === "approved" ? " resolved-approved" : (resolvedState === "rejected" ? " resolved-rejected" : "")}`;
    const titleStr = cleanText(item.title || "计费与高危操作审批");
    const detailStr = stripEmoji(item.detail || "该操作将启停云端 GPU 计算节点并产生计费变动，请确认是否放行执行。");
    const confirmId = String(item.confirmId || "");

    const actionsNode = resolvedState
      ? h("div", { cls: "confirm-card-actions" },
          h("span", {
            cls: `pill-tag ${resolvedState === "approved" ? "ok" : ""}`,
            txt: resolvedState === "approved" ? "已批准执行" : "已取消操作",
          })
        )
      : h("div", { cls: "confirm-card-actions" },
          h("button", {
            type: "button",
            cls: "btn btn-xs btn-primary btn-confirm-approve",
            txt: "批准执行",
            on: {
              click: async () => {
                await submitCopilotConfirm(item, true);
              },
            },
          }),
          h("button", {
            type: "button",
            cls: "btn btn-xs btn-subtle btn-confirm-reject",
            txt: "取消",
            on: {
              click: async () => {
                await submitCopilotConfirm(item, false);
              },
            },
          })
        );

    const card = h("div", { cls: cardCls, data: { confirmId } },
      h("div", { cls: "confirm-card-head" },
        h("div", { cls: "confirm-card-title-wrap" },
          createSvgIcon(resolvedState === "approved" ? "checkCircle" : "alertTriangle", 14),
          h("span", { txt: titleStr })
        ),
        item.tool ? h("span", { cls: "pill-tag", txt: cleanText(item.tool) }) : null
      ),
      h("div", { cls: "confirm-card-detail", txt: detailStr }),
      actionsNode
    );
    item._node = card;
    return card;
  }

  async function submitCopilotConfirm(item, approved) {
    if (!item || item.resolvedState || item._submitting) return;
    item._submitting = true;
    const cid = String(item.confirmId || "");
    if (item._node) {
      item._node.querySelectorAll("button").forEach((b) => { b.disabled = true; });
    }
    const res = await apiCall("resolve_agent_confirm", { confirm_id: cid, approved: Boolean(approved) });
    item._submitting = false;
    if (res && res.ok === false) {
      if (item._node) {
        item._node.querySelectorAll("button").forEach((b) => { b.disabled = false; });
      }
      showToast(res.message || "提交审批状态失败", { isError: true });
      return;
    }
    item.resolvedState = approved ? "approved" : "rejected";
    if (copilotState.pendingConfirm && String(copilotState.pendingConfirm.id || copilotState.pendingConfirm.confirm_id || "") === cid) {
      copilotState.pendingConfirm = null;
    }
    replaceCopilotConfirmCardNode(item);
    showToast(approved ? "已批准执行该计费操作" : "已取消该操作");
  }

  function buildCopilotThinkCard(item) {
    const rText = stripEmoji(item.reasoning || "");
    const streaming = Boolean(item.streamingReasoning);
    const metaStr = streaming ? "正在思考" : `${rText.length} 字`;
    const bodyEl = h("div", { cls: "copilot-think-body", txt: rText });
    const titleEl = h("span", { cls: "copilot-think-title", txt: streaming ? "深度思考中..." : "思考过程" });
    const metaEl = h("span", { cls: "copilot-think-meta", txt: metaStr });
    const summaryEl = h("summary", { cls: "copilot-think-summary" },
      h("div", { cls: "copilot-think-left" },
        createSvgIcon("brain", 12),
        titleEl
      ),
      h("div", { cls: "copilot-tool-right" },
        metaEl,
        createSvgIcon("chevronRight", 11, "icon-svg details-chevron")
      )
    );
    const details = h("details", {
      cls: `copilot-think-card${streaming ? " streaming" : ""}`,
    }, summaryEl, bodyEl);
    details.open = Boolean(item.thinkOpen);
    summaryEl.addEventListener("click", () => {
      item.userToggledThink = true;
      setTimeout(() => {
        item.thinkOpen = Boolean(details.open);
      }, 0);
    });
    item._thinkDetails = details;
    item._thinkTitleEl = titleEl;
    item._thinkMetaEl = metaEl;
    item._thinkBodyEl = bodyEl;
    return details;
  }

  function syncCopilotThinkCard(item) {
    if (!item || !item._node) return;
    const rText = stripEmoji(item.reasoning || "");
    if (!rText) return;
    if (!item._thinkDetails) {
      const card = buildCopilotThinkCard(item);
      if (item._richEl && item._richEl.parentNode === item._node) {
        item._node.insertBefore(card, item._richEl);
      } else {
        item._node.appendChild(card);
      }
      return;
    }
    const streaming = Boolean(item.streamingReasoning);
    item._thinkDetails.classList.toggle("streaming", streaming);
    item._thinkDetails.open = Boolean(item.thinkOpen);
    if (item._thinkTitleEl) item._thinkTitleEl.textContent = streaming ? "深度思考中..." : "思考过程";
    if (item._thinkMetaEl) item._thinkMetaEl.textContent = streaming ? "正在思考" : `${rText.length} 字`;
    if (item._thinkBodyEl) {
      item._thinkBodyEl.textContent = rText;
      if (streaming) item._thinkBodyEl.scrollTop = item._thinkBodyEl.scrollHeight;
    }
  }

  function renderCopilotAssistantNode(item) {
    const children = [];
    if (item.reasoning && String(item.reasoning).trim()) {
      children.push(buildCopilotThinkCard(item));
    }
    const rawTxt = String(item.text || "");
    const hasBody = Boolean(rawTxt.trim());
    const richWrap = h("div", { cls: `copilot-rich${(!hasBody && !item.streaming) ? " hidden" : ""}` });
    if (hasBody || item.streaming) {
      renderCopilotRichText(richWrap, rawTxt, Boolean(item.streaming));
    }
    children.push(richWrap);
    const node = h("div", { cls: `copilot-msg assistant${item.isError ? " error" : ""}` }, children);
    item._node = node;
    item._richEl = richWrap;
    return node;
  }

  function renderCopilotUserNode(item) {
    const children = [];
    if (item.ctxTag) {
      children.push(
        h("div", { cls: "copilot-msg-ctx-tag", title: item.ctxTag },
          createSvgIcon(item.ctxIcon || "image", 11),
          h("span", { txt: cleanText(item.ctxTag) })
        )
      );
    }
    children.push(document.createTextNode(stripEmoji(item.text || "")));
    const node = h("div", { cls: "copilot-msg user" }, children);
    item._node = node;
    return node;
  }

  function getContextualSuggestions() {
    if (currentView === "gallery") {
      if (galleryState.selectedPath) {
        return [
          {
            label: "基于选中图改词重绘",
            icon: "spark",
            fillOnly: true,
            prompt: `请基于选中作品 "${galleryState.selectedPath}" 的当前提示词与工作流，帮我改词重绘：`,
          },
          {
            label: "解析选中图参数",
            icon: "image",
            prompt: `请查看并解析作品 "${galleryState.selectedPath}" 的生成参数、LoRA 权重与优化建议。`,
          },
          {
            label: "拉回云端成图",
            icon: "zap",
            prompt: "请同步拉回云端最新生成的成品到画廊。",
          },
        ];
      }
      return [
        { label: "查看最新成图参数", icon: "image", prompt: "请列出画廊最近的生成作品并解析它们的提示词和核心参数。" },
        { label: "拉回云端成图", icon: "zap", prompt: "请同步拉回云端最新生成的成品到画廊。" },
        { label: "验号池选卡", icon: "shield", prompt: "请检查当前号池余额与各账号 GPU 配额，推荐当前最优算力方案。" },
      ];
    }
    if (currentView === "tuner") {
      return [
        { label: "查远端显存磁盘", icon: "terminal", prompt: "请检查当前在线实例与远端 GPU 显存及磁盘剩余空间。" },
        { label: "查云盘资产", icon: "spark", prompt: "请查看 gdrive_save 云盘中的 LoRA、工作流与音色资产列表。" },
        { label: "优化 Qwen 调参", icon: "brain", prompt: "请结合当前 GPU 显存余量，分析 Qwen3.8 的最优量化精度与上下文窗口配置。" },
      ];
    }
    if (currentView === "console") {
      return [
        {
          label: "诊断当前日志报错",
          icon: "terminal",
          prompt: `请读取并诊断「${consoleState.source}」日志中的最新异常或报错，定位根因并给出修复建议。`,
        },
        { label: "环境深度体检", icon: "shield", prompt: "请执行一次环境深度体检（Doctor），汇总阻塞或警告项修复方案。" },
        { label: "查远端显存磁盘", icon: "zap", prompt: "请检查远端服务日志、GPU 显存占用与磁盘剩余空间。" },
      ];
    }
    return [
      { label: "验号池选卡", icon: "shield", prompt: "请检查当前号池余额与各账号 GPU 配额，并推荐当前最优开机方案。" },
      { label: "查远端显存磁盘", icon: "terminal", prompt: "请检查当前集群节点状态、远端 GPU 显存与磁盘剩余空间。" },
      { label: "环境深度体检", icon: "spark", prompt: "请执行一次本机与云端环境深度体检并汇总核心建议。" },
    ];
  }

  function renderCopilotCtxStrip() {
    const strip = el("copilot-ctx-strip");
    if (!strip) return;
    strip.innerHTML = "";
    const ctx = copilotState.attachedContext;
    if (!ctx) {
      strip.classList.add("hidden");
      return;
    }
    strip.classList.remove("hidden");
    const previewEl = ctx.thumbUrl
      ? h("img", { cls: "copilot-ctx-thumb", src: ctx.thumbUrl, alt: ctx.name || "附件预览" })
      : h("span", { cls: "copilot-ctx-icon" }, createSvgIcon(ctx.kind === "logs" ? "terminal" : "image", 13));

    strip.appendChild(
      h("div", { cls: "copilot-ctx-chip" },
        previewEl,
        h("div", { cls: "copilot-ctx-meta" },
          h("span", { cls: "copilot-ctx-name", title: ctx.relPath || ctx.name || "", txt: cleanText(ctx.name || "上下文附件") }),
          ctx.sub ? h("span", { cls: "copilot-ctx-sub", txt: cleanText(ctx.sub) }) : null
        ),
        h("button", {
          type: "button",
          cls: "copilot-ctx-remove",
          title: "移除附加上下文",
          attrs: { "aria-label": "移除附件" },
          on: { click: () => clearCopilotContext() },
        }, createSvgIcon("close", 10))
      )
    );
  }

  function attachCopilotContext(ctxObj) {
    copilotState.attachedContext = ctxObj || null;
    renderCopilotCtxStrip();
  }

  function clearCopilotContext() {
    copilotState.attachedContext = null;
    renderCopilotCtxStrip();
  }

  function renderCopilotMessages() {
    const listEl = el("copilot-messages");
    if (!listEl) return;
    listEl.innerHTML = "";

    if (!Array.isArray(copilotState.items) || copilotState.items.length === 0) {
      const suggestions = getContextualSuggestions();
      const promptPills = h("div", { cls: "copilot-empty-prompts" },
        suggestions.map((ch) =>
          h("button", {
            type: "button",
            cls: "copilot-chip",
            txt: ch.label,
            on: {
              click: () => {
                if (ch.fillOnly || copilotState.busy) {
                  const inp = el("copilot-input");
                  if (inp) {
                    inp.value = ch.prompt;
                    resizeCopilotInput();
                    inp.focus();
                    inp.setSelectionRange(inp.value.length, inp.value.length);
                  }
                } else {
                  sendCopilotMessage(ch.prompt);
                }
              },
            },
          })
        )
      );
      listEl.appendChild(
        h("div", { cls: "copilot-empty" },
          h("div", { cls: "copilot-empty-badge" }, createSvgIcon("spark", 19)),
          h("div", { cls: "copilot-empty-title", txt: "Colab Studio AI 副驾" }),
          h("div", {
            cls: "copilot-empty-desc",
            txt: "核验号池选卡、诊断报错日志、热调 Qwen3.8，或基于画廊作品改词重绘。",
          }),
          promptPills
        )
      );
      copilotState.stickToBottom = true;
      updateCopilotScrollBtn();
      return;
    }

    for (const item of copilotState.items) {
      if (item.kind === "user") {
        listEl.appendChild(renderCopilotUserNode(item));
      } else if (item.kind === "assistant") {
        listEl.appendChild(renderCopilotAssistantNode(item));
      } else if (item.kind === "tool") {
        listEl.appendChild(renderCopilotToolCard(item));
      } else if (item.kind === "confirm") {
        listEl.appendChild(renderCopilotConfirmCard(item));
      }
    }
    scrollCopilotBottom(true);
  }

  function appendCopilotItem(item) {
    const listEl = el("copilot-messages");
    if (!listEl) return;
    if (copilotState.items.length === 0) {
      listEl.innerHTML = "";
    }
    copilotState.items.push(item);
    let node = null;
    if (item.kind === "user") {
      node = renderCopilotUserNode(item);
    } else if (item.kind === "assistant") {
      node = renderCopilotAssistantNode(item);
    } else if (item.kind === "tool") {
      node = renderCopilotToolCard(item);
    } else if (item.kind === "confirm") {
      node = renderCopilotConfirmCard(item);
    }
    if (node) listEl.appendChild(node);
    scrollCopilotBottom(item.kind === "user");
  }

  function finalizeCopilotStreamItem() {
    if (copilotState.activeStreamIdx >= 0 && copilotState.items[copilotState.activeStreamIdx]) {
      const prev = copilotState.items[copilotState.activeStreamIdx];
      prev.streaming = false;
      prev.streamingReasoning = false;
      if (!prev.userToggledThink && prev.reasoning) {
        prev.thinkOpen = false;
      }
      syncCopilotThinkCard(prev);
      if (prev._richEl) {
        const hasTxt = Boolean(String(prev.text || "").trim());
        prev._richEl.classList.toggle("hidden", !hasTxt);
        if (hasTxt) {
          renderCopilotRichText(prev._richEl, prev.text || "", false);
        } else {
          prev._richEl.innerHTML = "";
        }
      }
    }
    copilotState.activeStreamIdx = -1;
  }

  function formatCompactModelLabel() {
    const _shortSlug = (rawName) => {
      const m = cleanText(rawName || "");
      if (!m) return "";
      return m.includes("/") ? (m.split("/").filter(Boolean).pop() || m) : m;
    };
    if (copilotState.driver === "claude_code") {
      return _shortSlug(copilotState.model) || "Claude Code";
    }
    if (copilotState.brainSource === "colab_llm" || copilotState.resolvedSource === "colab_llm") {
      return _shortSlug(copilotState.model) || cleanText(copilotState.resolvedBrainLabel || "Colab Qwen");
    }
    if (copilotState.model) {
      return _shortSlug(copilotState.model);
    }
    const raw = cleanText(copilotState.resolvedBrainLabel || "");
    if (!raw) return "自动路由";
    const parts = raw.split(" · ");
    return _shortSlug(parts.length > 1 ? parts[parts.length - 1] : raw);
  }

  function formatRouteBadgeLabel() {
    if (copilotState.driver === "claude_code") return "Claude Code";
    if (copilotState.resolvedSource === "colab_llm") return "Colab Qwen";
    if (copilotState.activeProviderName) return cleanText(copilotState.activeProviderName);
    if (copilotState.resolvedSource === "external_api") return "外部网关";
    if (copilotState.resolvedSource === "none") return "待配置";
    return "自动路由";
  }

  function renderCopilotHeaderAndControls() {
    const shell = el("studio-shell");
    const drawer = el("copilot-drawer");
    const toggleBtn = el("btn-copilot-toggle");
    if (shell) shell.classList.toggle("copilot-open", Boolean(copilotState.open));
    if (toggleBtn) toggleBtn.setAttribute("aria-expanded", String(Boolean(copilotState.open)));
    if (drawer) {
      drawer.setAttribute("aria-hidden", String(!copilotState.open));
      if (copilotState.open) {
        drawer.removeAttribute("inert");
        drawer.inert = false;
      } else {
        if (drawer.contains(document.activeElement) && document.activeElement.blur) {
          document.activeElement.blur();
        }
        drawer.setAttribute("inert", "");
        drawer.inert = true;
      }
    }

    const dot = el("copilot-toggle-dot");
    const headDot = el("copilot-head-dot");
    const brainDot = el("copilot-brain-dot");
    if (dot) {
      dot.className = `copilot-dot${copilotState.busy ? " busy" : (copilotState.pendingConfirm ? " warn" : "")}`;
    }
    const dotCls = `status-dot ${copilotState.busy ? "running" : (copilotState.pendingConfirm ? "warn" : "idle")}`;
    if (headDot) headDot.className = dotCls;
    if (brainDot) brainDot.className = dotCls;

    const routeBadge = el("copilot-route-badge");
    if (routeBadge) {
      routeBadge.textContent = formatRouteBadgeLabel();
      routeBadge.title = cleanText(copilotState.resolvedBrainLabel || routeBadge.textContent);
    }

    const brainLbl = el("copilot-brain-label");
    const brainPill = el("copilot-brain-pill");
    const compactModel = formatCompactModelLabel();
    if (brainLbl) {
      brainLbl.textContent = compactModel;
    }
    if (brainPill) {
      brainPill.title = cleanText(copilotState.resolvedBrainLabel || compactModel);
    }

    const effortBtn = el("btn-copilot-effort");
    const effortLbl = el("copilot-effort-label");
    const supportedRsn = copilotState.reasoningSupported !== false;
    const eff = supportedRsn ? String(copilotState.reasoningEffort || "auto") : "auto";
    const effShort = supportedRsn ? (REASONING_EFFORT_LABELS[eff] || "自动") : "直答";
    const turbo = Boolean(copilotState.autoApproveBilling);
    if (effortLbl) {
      effortLbl.textContent = turbo ? (eff === "auto" ? (supportedRsn ? "Turbo" : "直答 · Turbo") : `${effShort} · Turbo`) : effShort;
    }
    if (effortBtn) {
      effortBtn.classList.toggle("is-turbo", turbo);
      effortBtn.classList.toggle("is-tuned", !turbo && supportedRsn && eff !== "auto");
      const effFull = supportedRsn ? (REASONING_EFFORT_FULL_LABELS[eff] || effShort) : "标准直答（模型不支持思考档位）";
      effortBtn.title = `思考强度：${effFull} | ${turbo ? "Turbo 自动放行已开启" : "计费操作安全审批"}`;
    }

    const sendBtn = el("btn-copilot-send");
    if (sendBtn) {
      const iconSend = sendBtn.querySelector(".icon-send");
      const iconStop = sendBtn.querySelector(".icon-stop");
      sendBtn.classList.toggle("is-stop", Boolean(copilotState.busy));
      sendBtn.title = copilotState.busy ? "停止当前生成" : "发送消息 (Enter 确认 · Shift+Enter 换行)";
      sendBtn.setAttribute("aria-label", copilotState.busy ? "停止生成" : "发送消息");
      if (iconSend) iconSend.classList.toggle("hidden", Boolean(copilotState.busy));
      if (iconStop) iconStop.classList.toggle("hidden", !copilotState.busy);
    }
    renderCopilotCtxStrip();
  }

  function renderCopilotChips() {
    if (Array.isArray(copilotState.items) && copilotState.items.length === 0) {
      renderCopilotMessages();
    }
    if (copilotState.plusPopoverOpen) {
      renderCopilotPlusPopover();
    }
  }

  function renderQuickProviderModelsList() {
    const listEl = el("cfg-quick-model-list");
    const searchWrap = el("cfg-quick-search-wrap");
    if (!listEl) return;
    listEl.innerHTML = "";

    const providers = Array.isArray(copilotState.providers) ? copilotState.providers : [];
    if (providers.length === 0) {
      if (searchWrap) searchWrap.classList.add("hidden");
      listEl.appendChild(
        h("div", { cls: "quick-empty-guide" },
          h("div", { cls: "quick-empty-desc", txt: "尚未导入外部供应商或网关模型。可一键接入本机 Magpie 网关，或添加自定义供应商拉取可用模型。" }),
          h("div", { cls: "quick-empty-actions" },
            h("button", {
              type: "button",
              id: "btn-quick-magpie-preset",
              cls: "btn btn-xs btn-primary",
              on: {
                click: () => {
                  toggleCopilotConfigPopover(false);
                  openProviderModal({ presetMagpie: true });
                },
              },
            }, createSvgIcon("zap", 11), h("span", { txt: "接入 Magpie 本机网关" })),
            h("button", {
              type: "button",
              id: "btn-quick-add-provider",
              cls: "btn btn-xs btn-subtle",
              txt: "自定义供应商",
              on: {
                click: () => {
                  toggleCopilotConfigPopover(false);
                  openProviderModal({ isNew: true });
                },
              },
            })
          )
        )
      );
      return;
    }

    const totalModels = providers.reduce((acc, p) => {
      const ms = Array.isArray(p.models) ? p.models : [];
      return acc + (ms.length || (p.default_model ? 1 : 0));
    }, 0);
    const kw = String(copilotState.quickFilter || "").trim().toLowerCase();
    if (searchWrap) {
      searchWrap.classList.toggle("hidden", totalModels < 4 && !kw);
    }

    let renderedCount = 0;
    for (const p of providers) {
      const pid = String(p.id || "");
      const pname = cleanText(p.name || "自定义供应商");
      const protoLbl = PROVIDER_PROTO_LABELS[p.api_format] || "OpenAI";
      let models = Array.isArray(p.models) ? p.models.filter(Boolean) : [];
      if (models.length === 0 && p.default_model) {
        models = [String(p.default_model)];
      }
      const filteredModels = kw
        ? models.filter((m) => String(m).toLowerCase().includes(kw) || pname.toLowerCase().includes(kw))
        : models;
      if (kw && filteredModels.length === 0) continue;

      const isActiveProv = Boolean(
        (copilotState.activeProviderId && copilotState.activeProviderId === pid)
        || (!copilotState.activeProviderId && pid === "__legacy__")
      );
      const group = h("div", { cls: "quick-provider-group" },
        h("div", { cls: "quick-provider-head" },
          h("div", { cls: "quick-provider-title-wrap" },
            h("span", { cls: `status-dot ${p.ready ? "running" : "idle"}` }),
            h("span", { cls: "quick-provider-name", txt: pname }),
            h("span", { cls: "quick-provider-proto", txt: protoLbl })
          ),
          h("button", {
            type: "button",
            cls: "btn btn-xs btn-ghost",
            txt: "管理",
            on: {
              click: () => {
                toggleCopilotConfigPopover(false);
                openProviderModal({ providerId: pid });
              },
            },
          })
        )
      );

      const modelsBox = h("div", { cls: "quick-provider-models" });
      if (filteredModels.length === 0) {
        modelsBox.appendChild(
          h("button", {
            type: "button",
            cls: "quick-model-btn",
            on: {
              click: () => {
                toggleCopilotConfigPopover(false);
                openProviderModal({ providerId: pid });
              },
            },
          },
            h("span", { cls: "quick-model-id", txt: "点击获取或填写可用模型..." }),
            createSvgIcon("chevronRight", 12)
          )
        );
      } else {
        for (const mdl of filteredModels) {
          const isCur = Boolean(isActiveProv && mdl === copilotState.model);
          modelsBox.appendChild(
            h("button", {
              type: "button",
              cls: `quick-model-btn${isCur ? " active" : ""}`,
              data: { providerId: pid, model: mdl },
              title: `${pname} · ${mdl}`,
              on: {
                click: async () => {
                  const res = await apiCall("save_agent_config", {
                    select_provider_model: { provider_id: pid, model: mdl },
                  });
                  if (res && res.ok === false) {
                    showToast(res.message || "切换模型失败", { isError: true });
                    return;
                  }
                  if (res && typeof res === "object") {
                    syncAgentStatePayload(res.state || res, { replaceItems: false });
                  }
                  toggleCopilotConfigPopover(false);
                  showToast((res && res.message) || `已切换至 ${mdl}`);
                },
              },
            },
              h("span", { cls: "quick-model-id", txt: mdl }),
              isCur ? createSvgIcon("check", 12) : null
            )
          );
          renderedCount += 1;
        }
      }
      group.appendChild(modelsBox);
      listEl.appendChild(group);
    }

    if (kw && renderedCount === 0) {
      listEl.appendChild(
        h("div", { cls: "prov-models-empty", txt: "没有匹配搜索关键词的已导入模型" })
      );
    }
  }

  function renderCopilotPopoverControls() {
    document.querySelectorAll("#cfg-driver-seg .seg-item").forEach((btn) => {
      const val = btn.dataset.val;
      btn.classList.toggle("active", val === copilotState.driver);
      if (val === "claude_code") {
        btn.disabled = !copilotState.claudeAvailable;
        btn.textContent = copilotState.claudeAvailable ? "Claude Code" : "Claude (未装)";
      }
    });
    if (el("cfg-claude-hint")) {
      el("cfg-claude-hint").classList.toggle("hidden", Boolean(copilotState.claudeAvailable));
    }

    document.querySelectorAll("#cfg-brain-seg .seg-item").forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.val === copilotState.brainSource);
    });

    const extGroup = el("cfg-external-group");
    if (extGroup) {
      extGroup.classList.toggle("dimmed", copilotState.brainSource === "colab_llm");
    }

    const colabBadge = el("cfg-colab-llm-badge");
    if (colabBadge) {
      colabBadge.className = `pill-tag${copilotState.colabLlmOnline ? " ok" : ""}`;
      colabBadge.textContent = copilotState.colabLlmOnline ? "Colab Qwen 在线" : "Colab Qwen 离线";
    }

    const statusBadge = el("cfg-status-badge");
    if (statusBadge) {
      statusBadge.className = `pill-tag${copilotState.busy ? " accent" : " ok"}`;
      statusBadge.textContent = copilotState.busy ? "执行中" : (copilotState.driver === "claude_code" ? "Claude 就绪" : "Native 就绪");
    }

    const clearKeyBtn = el("btn-agent-clear-key");
    if (clearKeyBtn) {
      clearKeyBtn.classList.toggle("hidden", !copilotState.hasApiKey);
    }
    renderQuickProviderModelsList();
  }

  function renderCopilotEffortPopover() {
    const supportedRsn = copilotState.reasoningSupported !== false;
    const eff = supportedRsn ? String(copilotState.reasoningEffort || "auto") : "auto";
    const list = el("effort-options-list");
    if (list) {
      list.innerHTML = "";
      if (!supportedRsn) {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "effort-option active";
        btn.dataset.effort = "auto";
        const textWrap = document.createElement("span");
        textWrap.className = "effort-opt-text";
        const titleEl = document.createElement("span");
        titleEl.className = "effort-opt-title";
        titleEl.textContent = "直答模式 (无思考档位)";
        const descEl = document.createElement("span");
        descEl.className = "effort-opt-desc";
        descEl.textContent = "当前模型不支持独立调节思考强度";
        textWrap.append(titleEl, descEl);
        btn.append(textWrap, createSvgIcon("check", 13, "icon-svg effort-check"));
        list.appendChild(btn);
      } else {
        const concrete = Array.isArray(copilotState.supportedReasoningEfforts) && copilotState.supportedReasoningEfforts.length > 0
          ? copilotState.supportedReasoningEfforts
          : ["none", "low", "medium", "high", "xhigh"];
        const levels = ["auto", ...concrete.filter((lvl) => lvl && lvl !== "auto" && EFFORT_OPTION_META[lvl])];
        for (const lvl of levels) {
          const meta = EFFORT_OPTION_META[lvl];
          if (!meta) continue;
          const btn = document.createElement("button");
          btn.type = "button";
          btn.className = `effort-option${lvl === eff ? " active" : ""}`;
          btn.dataset.effort = lvl;
          const textWrap = document.createElement("span");
          textWrap.className = "effort-opt-text";
          const titleEl = document.createElement("span");
          titleEl.className = "effort-opt-title";
          titleEl.textContent = meta.title;
          const descEl = document.createElement("span");
          descEl.className = "effort-opt-desc";
          if (lvl === "auto" && copilotState.defaultReasoningEffort) {
            const defShort = REASONING_EFFORT_LABELS[copilotState.defaultReasoningEffort] || copilotState.defaultReasoningEffort;
            descEl.textContent = `遵循网关默认（推荐: ${defShort}）`;
          } else {
            descEl.textContent = meta.desc;
          }
          textWrap.append(titleEl, descEl);
          btn.append(textWrap, createSvgIcon("check", 13, "icon-svg effort-check"));
          list.appendChild(btn);
        }
      }
    }
    const badge = el("effort-status-badge");
    if (badge) {
      if (!supportedRsn) {
        badge.className = "pill-tag";
        badge.textContent = "标准直答";
      } else {
        badge.className = `pill-tag${eff !== "auto" ? " accent" : ""}`;
        badge.textContent = REASONING_EFFORT_FULL_LABELS[eff] || "自动自适应";
      }
    }
    const sw = el("btn-agent-auto-approve");
    if (sw) {
      sw.classList.toggle("on", Boolean(copilotState.autoApproveBilling));
      sw.setAttribute("aria-checked", String(Boolean(copilotState.autoApproveBilling)));
    }
  }

  function renderCopilotPlusPopover() {
    const pop = el("copilot-plus-popover");
    if (!pop) return;
    pop.innerHTML = "";

    const attachActions = [];
    if (galleryState.selectedPath) {
      const selItem = (galleryState.items || []).find((it) => it.rel_path === galleryState.selectedPath);
      const shortName = String(galleryState.selectedPath).split("/").pop();
      attachActions.push({
        label: `附加画廊图: ${shortName}`,
        icon: "image",
        onClick: () => {
          attachCopilotContext({
            kind: "artwork",
            relPath: galleryState.selectedPath,
            thumbUrl: (selItem && selItem.media_url) || "",
            name: shortName,
            sub: "画廊作品上下文",
          });
          toggleCopilotPlusPopover(false);
          if (el("copilot-input")) el("copilot-input").focus();
        },
      });
    }
    if (Array.isArray(consoleState.lines) && consoleState.lines.length > 0) {
      attachActions.push({
        label: `附加当前报错日志 (${consoleState.source})`,
        icon: "terminal",
        onClick: () => {
          const snippet = consoleState.lines
            .filter((ln) => /error|fail|traceback|exception|warn/i.test(String(ln || "")))
            .slice(-6)
            .join("\n") || consoleState.lines.slice(-6).join("\n");
          attachCopilotContext({
            kind: "logs",
            source: consoleState.source || "local_diag",
            name: `日志: ${consoleState.source || "local_diag"}`,
            sub: `${consoleState.lines.length} 行上下文`,
            snippet: stripEmoji(snippet),
          });
          toggleCopilotPlusPopover(false);
          if (el("copilot-input")) el("copilot-input").focus();
        },
      });
    }

    if (attachActions.length > 0) {
      pop.appendChild(h("div", { cls: "plus-menu-section", txt: "附加上下文" }));
      for (const act of attachActions) {
        pop.appendChild(
          h("button", {
            type: "button",
            cls: "plus-menu-item",
            on: { click: act.onClick },
          },
            createSvgIcon(act.icon, 13),
            h("span", { cls: "plus-menu-label", txt: act.label })
          )
        );
      }
      pop.appendChild(h("div", { cls: "plus-menu-divider" }));
    }

    pop.appendChild(h("div", { cls: "plus-menu-section", txt: "快捷指令" }));
    for (const ch of getContextualSuggestions()) {
      pop.appendChild(
        h("button", {
          type: "button",
          cls: "plus-menu-item",
          on: {
            click: () => {
              toggleCopilotPlusPopover(false);
              if (ch.fillOnly || copilotState.busy) {
                const inp = el("copilot-input");
                if (inp) {
                  inp.value = ch.prompt;
                  resizeCopilotInput();
                  inp.focus();
                }
              } else {
                sendCopilotMessage(ch.prompt);
              }
            },
          },
        },
          createSvgIcon(ch.icon || "spark", 13),
          h("span", { cls: "plus-menu-label", txt: ch.label })
        )
      );
    }

    pop.appendChild(h("div", { cls: "plus-menu-divider" }));
    pop.appendChild(
      h("button", {
        type: "button",
        cls: "plus-menu-item",
        on: {
          click: () => {
            toggleCopilotPlusPopover(false);
            openProviderModal({ providerId: copilotState.activeProviderId || null });
          },
        },
      },
        createSvgIcon("sliders", 13),
        h("span", { cls: "plus-menu-label", txt: "管理供应商与导入模型..." })
      )
    );
  }

  function _setFloatingPopoverVisible(pop, trigger, visible) {
    if (!pop) return;
    if (visible) {
      pop.classList.remove("hidden");
      pop.removeAttribute("inert");
      pop.inert = false;
      pop.setAttribute("aria-hidden", "false");
    } else {
      if (pop.contains(document.activeElement) && document.activeElement.blur) {
        document.activeElement.blur();
      }
      pop.classList.add("hidden");
      pop.setAttribute("aria-hidden", "true");
      pop.setAttribute("inert", "");
      pop.inert = true;
    }
    if (trigger) trigger.setAttribute("aria-expanded", String(Boolean(visible)));
  }

  function closeAllCopilotPopovers(except = "") {
    if (except !== "config" && copilotState.popoverOpen) {
      copilotState.popoverOpen = false;
      _setFloatingPopoverVisible(el("copilot-config-popover"), el("copilot-brain-pill"), false);
    }
    if (except !== "effort" && copilotState.effortPopoverOpen) {
      copilotState.effortPopoverOpen = false;
      _setFloatingPopoverVisible(el("copilot-effort-popover"), el("btn-copilot-effort"), false);
    }
    if (except !== "plus" && copilotState.plusPopoverOpen) {
      copilotState.plusPopoverOpen = false;
      _setFloatingPopoverVisible(el("copilot-plus-popover"), el("btn-copilot-plus"), false);
    }
    if (except !== "history" && copilotState.historyPopoverOpen) {
      copilotState.historyPopoverOpen = false;
      histClearArm.disarm();
      _setFloatingPopoverVisible(el("copilot-history-popover"), el("btn-copilot-history"), false);
    }
    updateCopilotScrollBtn();
  }

  function _formatSessionRelTime(tsSec) {
    const t = Number(tsSec || 0);
    if (!Number.isFinite(t) || t <= 0) return "";
    const d = new Date(t * 1000);
    const now = new Date();
    const hh = String(d.getHours()).padStart(2, "0");
    const mm = String(d.getMinutes()).padStart(2, "0");
    if (d.toDateString() === now.toDateString()) {
      return `${hh}:${mm}`;
    }
    const mo = String(d.getMonth() + 1).padStart(2, "0");
    const da = String(d.getDate()).padStart(2, "0");
    return `${mo}/${da} ${hh}:${mm}`;
  }

  function renderCopilotHistoryPopover() {
    const badge = el("history-count-badge");
    const sessions = Array.isArray(copilotState.sessions) ? copilotState.sessions : [];
    if (badge) {
      badge.textContent = `${sessions.length} 个会话`;
    }

    const stats = copilotState.contextStats || {};
    const msgCnt = Number(stats.messageCount || 0);
    const maxMsgs = Math.max(1, Number(stats.maxMessages || 40));
    const approxTok = Number(stats.approxTokens || 0);
    const ctxTxt = el("history-ctx-text");
    if (ctxTxt) {
      ctxTxt.textContent = `当前上下文：${msgCnt} / ${maxMsgs} 条 · 约 ${approxTok} tok`;
    }
    const ctxFill = el("history-ctx-fill");
    if (ctxFill) {
      const pct = Math.min(100, Math.round((msgCnt / maxMsgs) * 100));
      ctxFill.style.width = `${pct}%`;
      ctxFill.classList.toggle("warn", pct >= 80);
    }

    const clearAllBtn = el("btn-hist-clear-all");
    if (clearAllBtn) {
      const armed = histClearArm.isArmed();
      clearAllBtn.textContent = armed ? "确认清空" : "清空全部";
      clearAllBtn.classList.toggle("btn-danger", armed);
      clearAllBtn.classList.toggle("btn-ghost", !armed);
    }

    const searchWrap = el("history-search-wrap");
    const kw = String(copilotState.historyFilter || "").trim().toLowerCase();
    if (searchWrap) {
      searchWrap.classList.toggle("hidden", sessions.length < 4 && !kw);
    }

    const listEl = el("copilot-history-list");
    if (!listEl) return;
    listEl.innerHTML = "";

    const filtered = kw
      ? sessions.filter((s) => {
          const ttl = String(s.title || "").toLowerCase();
          const prv = String(s.preview || "").toLowerCase();
          return ttl.includes(kw) || prv.includes(kw);
        })
      : sessions;

    if (filtered.length === 0) {
      listEl.appendChild(
        h("div", { cls: "hist-empty", txt: kw ? "没有匹配的对话记录" : "暂无历史对话" })
      );
      return;
    }

    for (const s of filtered) {
      const sid = String(s.id || "");
      const isCur = sid === String(copilotState.activeSessionId || "");
      const count = Number(s.message_count || 0);
      const tStr = _formatSessionRelTime(s.updated_ts || s.created_ts);
      const prv = cleanText(s.preview || "");
      const metaParts = [`${count} 条`];
      if (tStr) metaParts.push(tStr);
      if (prv) metaParts.push(prv);

      const mainBtn = h("button", {
        type: "button",
        cls: "hist-session-main",
        data: { sessionId: sid },
        title: cleanText(s.title || "新对话"),
        on: {
          click: async () => {
            await switchCopilotSession(sid);
          },
        },
      },
        h("span", { cls: "hist-session-title", txt: cleanText(s.title || "新对话") }),
        h("span", { cls: "hist-session-meta", txt: metaParts.join(" · ") })
      );

      const delBtn = h("button", {
        type: "button",
        cls: "hist-session-del",
        data: { delSessionId: sid },
        title: "删除此会话",
        attrs: { "aria-label": "删除此会话" },
        on: {
          click: async (e) => {
            e.stopPropagation();
            await deleteCopilotSession(sid);
          },
        },
      }, createSvgIcon("close", 11));

      listEl.appendChild(
        h("div", { cls: `hist-session-item${isCur ? " active" : ""}` }, mainBtn, delBtn)
      );
    }
  }

  function toggleCopilotHistoryPopover(forceOpen) {
    const pop = el("copilot-history-popover");
    const btn = el("btn-copilot-history");
    if (!pop) return;
    const nextOpen = typeof forceOpen === "boolean" ? forceOpen : !copilotState.historyPopoverOpen;
    if (nextOpen) {
      closeAllCopilotPopovers("history");
      copilotState.historyFilter = "";
      if (el("inp-history-filter")) el("inp-history-filter").value = "";
      histClearArm.disarm();
    }
    copilotState.historyPopoverOpen = nextOpen;
    if (nextOpen) {
      renderCopilotHistoryPopover();
    }
    _setFloatingPopoverVisible(pop, btn, nextOpen);
    updateCopilotScrollBtn();
  }

  async function createNewCopilotSession() {
    if (copilotState.busy) {
      showToast("副驾正在生成中，请先停止后再新建对话");
      return false;
    }
    const res = await apiCall("manage_agent_session", { action: "new" });
    if (res && res.ok === false) {
      showToast(res.message || "新建对话失败", { isError: true });
      return false;
    }
    if (!copilotState.open) setCopilotOpen(true, { focusInput: false });
    if (res && typeof res === "object") {
      syncAgentStatePayload(res.state || res, { forceReplace: true });
    } else {
      await loadAgentState(true, true);
    }
    closeAllCopilotPopovers();
    const inp = el("copilot-input");
    if (inp && copilotState.open) inp.focus();
    if (res && res.message) showToast(res.message);
    return true;
  }

  async function switchCopilotSession(sessionId) {
    if (copilotState.busy) {
      showToast("副驾正在生成中，请先停止后再切换会话");
      return false;
    }
    const sid = String(sessionId || "").trim();
    if (!sid) return false;
    if (sid === String(copilotState.activeSessionId || "")) {
      toggleCopilotHistoryPopover(false);
      return true;
    }
    const res = await apiCall("manage_agent_session", { action: "switch", session_id: sid });
    if (res && res.ok === false) {
      showToast(res.message || "切换会话失败", { isError: true });
      return false;
    }
    if (res && typeof res === "object") {
      syncAgentStatePayload(res.state || res, { forceReplace: true });
    } else {
      await loadAgentState(true, true);
    }
    toggleCopilotHistoryPopover(false);
    if (res && res.message) showToast(res.message);
    return true;
  }

  async function deleteCopilotSession(sessionId) {
    const sid = String(sessionId || "").trim();
    if (!sid) return false;
    if (copilotState.busy && sid === String(copilotState.activeSessionId || "")) {
      showToast("当前会话正在生成中，请先停止后再删除");
      return false;
    }
    const res = await apiCall("manage_agent_session", { action: "delete", session_id: sid });
    if (res && res.ok === false) {
      showToast(res.message || "删除会话失败", { isError: true });
      return false;
    }
    if (res && typeof res === "object") {
      syncAgentStatePayload(res.state || res, { forceReplace: true });
    } else {
      await loadAgentState(true, true);
    }
    if (copilotState.historyPopoverOpen) renderCopilotHistoryPopover();
    if (res && res.message) showToast(res.message);
    return true;
  }

  async function clearAllCopilotSessions() {
    if (copilotState.busy) {
      showToast("副驾正在生成中，请先停止后再清空全部会话");
      return false;
    }
    if (!histClearArm.isArmed()) {
      histClearArm.arm(() => {
        if (copilotState.historyPopoverOpen) renderCopilotHistoryPopover();
      });
      renderCopilotHistoryPopover();
      return false;
    }
    histClearArm.disarm();
    const res = await apiCall("manage_agent_session", { action: "clear_all" });
    if (res && res.ok === false) {
      showToast(res.message || "清空历史会话失败", { isError: true });
      return false;
    }
    if (res && typeof res === "object") {
      syncAgentStatePayload(res.state || res, { forceReplace: true });
    } else {
      await loadAgentState(true, true);
    }
    toggleCopilotHistoryPopover(false);
    if (res && res.message) showToast(res.message);
    return true;
  }

  function toggleCopilotConfigPopover(forceOpen) {
    const pop = el("copilot-config-popover");
    const pill = el("copilot-brain-pill");
    if (!pop) return;
    const nextOpen = typeof forceOpen === "boolean" ? forceOpen : !copilotState.popoverOpen;
    if (nextOpen) closeAllCopilotPopovers("config");
    copilotState.popoverOpen = nextOpen;

    if (nextOpen) {
      copilotState.quickFilter = "";
      if (el("inp-quick-model-filter")) el("inp-quick-model-filter").value = "";
      const keyInp = el("inp-agent-api-key");
      if (keyInp && !providerModalState.open) {
        keyInp.value = "";
      }
      renderCopilotPopoverControls();
    }
    _setFloatingPopoverVisible(pop, pill, nextOpen);
    updateCopilotScrollBtn();
  }

  function toggleCopilotEffortPopover(forceOpen) {
    const pop = el("copilot-effort-popover");
    const btn = el("btn-copilot-effort");
    if (!pop) return;
    const nextOpen = typeof forceOpen === "boolean" ? forceOpen : !copilotState.effortPopoverOpen;
    if (nextOpen) closeAllCopilotPopovers("effort");
    copilotState.effortPopoverOpen = nextOpen;
    if (nextOpen) {
      renderCopilotEffortPopover();
    }
    _setFloatingPopoverVisible(pop, btn, nextOpen);
    updateCopilotScrollBtn();
  }

  function toggleCopilotPlusPopover(forceOpen) {
    const pop = el("copilot-plus-popover");
    const btn = el("btn-copilot-plus");
    if (!pop) return;
    const nextOpen = typeof forceOpen === "boolean" ? forceOpen : !copilotState.plusPopoverOpen;
    if (nextOpen) closeAllCopilotPopovers("plus");
    copilotState.plusPopoverOpen = nextOpen;
    if (nextOpen) {
      renderCopilotPlusPopover();
    }
    _setFloatingPopoverVisible(pop, btn, nextOpen);
    updateCopilotScrollBtn();
  }

  async function applyAgentConfigPatch(patch, toastFallback = "") {
    const res = await apiCall("save_agent_config", patch);
    if (res && res.ok === false) {
      showToast(res.message || "更新副驾配置失败", { isError: true });
      return null;
    }
    if (res && typeof res === "object") {
      syncAgentStatePayload(res.state || res, { replaceItems: false });
    } else {
      await loadAgentState(false);
    }
    if ((res && res.message) || toastFallback) {
      showToast((res && res.message) || toastFallback);
    }
    return res;
  }

  /* ----------------------------------------------------------------
     自定义供应商与 Magpie 模型导入模态弹窗
     ---------------------------------------------------------------- */
  function _parseModelListText(raw) {
    const seen = [];
    for (const part of String(raw || "").split(/[,\n]/)) {
      const m = part.trim();
      if (m && !seen.includes(m)) seen.push(m);
    }
    return seen;
  }

  function _isLoopbackUrlStr(u) {
    return /^https?:\/\/(127\.0\.0\.1|localhost|\[::1\])(?::\d+)?(?:\/|$)/i.test(String(u || "").trim());
  }

  function _setProviderStatus(msg, kind = "") {
    providerModalState.statusMsg = cleanText(msg || "");
    providerModalState.statusKind = kind;
    const node = el("prov-status-text");
    if (node) {
      node.textContent = providerModalState.statusMsg || "支持导入 Magpie 本机网关及兼容 OpenAI / Claude / Gemini 接口";
      node.className = `prov-status-text${kind ? ` ${kind}` : ""}`;
    }
  }

  function _loadProviderIntoDraft(providerId, { isNew = false, presetMagpie = false } = {}) {
    provDelArm.disarm();
    const providers = Array.isArray(copilotState.providers) ? copilotState.providers : [];
    providerModalState.fetchedModels = [];
    providerModalState.fetchedCapabilities = {};
    providerModalState.modelFilter = "";
    providerModalState.testResults = {};
    providerModalState.probeModel = "";
    providerModalState.showKey = false;

    const keyInp = el("inp-agent-api-key");
    if (keyInp) {
      keyInp.value = "";
      keyInp.type = "password";
    }

    if (presetMagpie) {
      const existingMagpie = providers.find((p) => String(p.base_url || "").includes("127.0.0.1:3425"));
      if (existingMagpie) {
        providerId = existingMagpie.id;
        isNew = false;
      } else {
        providerModalState.isNew = true;
        providerModalState.editingId = "";
        providerModalState.name = "Magpie 本机网关";
        providerModalState.apiFormat = "openai";
        providerModalState.baseUrl = "http://127.0.0.1:3425/v1";
        providerModalState.selectedModels = [];
        providerModalState.defaultModel = "";
        providerModalState.modelsText = "";
        providerModalState.hasStoredKey = false;
        _setProviderStatus("已填入 Magpie 本机默认网关，可点「获取模型」自动读取可用模型", "");
        return;
      }
    }

    if (isNew) {
      providerModalState.isNew = true;
      providerModalState.editingId = "";
      providerModalState.name = "";
      providerModalState.apiFormat = "openai";
      providerModalState.baseUrl = "";
      providerModalState.selectedModels = [];
      providerModalState.defaultModel = "";
      providerModalState.modelsText = "";
      providerModalState.hasStoredKey = false;
      _setProviderStatus("填写网关名称与 Base URL，然后点击「获取模型」或手填模型 ID", "");
      return;
    }

    const picked = providers.find((p) => p.id === providerId)
      || providers.find((p) => p.id === copilotState.activeProviderId)
      || providers[0];
    if (!picked) {
      _loadProviderIntoDraft("", { isNew: true });
      return;
    }

    const models = Array.isArray(picked.models) ? picked.models.filter(Boolean) : [];
    providerModalState.isNew = false;
    providerModalState.editingId = String(picked.id || "");
    providerModalState.name = cleanText(picked.name || "");
    providerModalState.apiFormat = String(picked.api_format || "openai");
    providerModalState.baseUrl = String(picked.base_url || "");
    providerModalState.selectedModels = models.slice();
    providerModalState.defaultModel = String(picked.default_model || models[0] || "");
    providerModalState.modelsText = models.join(", ");
    providerModalState.fetchedCapabilities = (picked.model_capabilities && typeof picked.model_capabilities === "object")
      ? Object.assign({}, picked.model_capabilities)
      : {};
    providerModalState.hasStoredKey = Boolean(picked.has_api_key);
    _setProviderStatus(
      picked.ready ? `当前「${providerModalState.name}」配置就绪` : "请补充 Base URL 或 API Key",
      picked.ready ? "ok" : ""
    );
  }

  function renderProviderModelsGrid() {
    const grid = el("prov-models-grid");
    const countEl = el("prov-models-count");
    if (!grid) return;
    grid.innerHTML = "";

    const selected = providerModalState.selectedModels || [];
    const fetched = providerModalState.fetchedModels || [];
    const union = [];
    for (const m of selected) {
      if (m && !union.includes(m)) union.push(m);
    }
    for (const m of fetched) {
      if (m && !union.includes(m)) union.push(m);
    }

    const kw = String(providerModalState.modelFilter || "").trim().toLowerCase();
    const visible = kw ? union.filter((m) => m.toLowerCase().includes(kw)) : union;

    if (countEl) {
      countEl.textContent = fetched.length > 0
        ? `已选 ${selected.length} / ${fetched.length}`
        : `已选 ${selected.length}`;
    }

    if (visible.length === 0) {
      grid.appendChild(
        h("div", {
          cls: "prov-models-empty",
          txt: union.length === 0
            ? "点击上方「获取模型」自动读取网关模型，或在输入框填入模型 ID（逗号分隔）"
            : "当前搜索词未匹配到模型",
        })
      );
      return;
    }

    for (const mdl of visible) {
      const isSel = selected.includes(mdl);
      const isDef = isSel && providerModalState.defaultModel === mdl;
      const probe = providerModalState.testResults[mdl];

      let badgeNode = null;
      if (probe && probe.testing) {
        badgeNode = h("span", { cls: "latency-pill testing tabular", txt: "测速中" });
      } else if (probe && probe.ok) {
        badgeNode = h("span", {
          cls: "latency-pill ok tabular",
          txt: `${probe.latency_ms || 1}ms`,
          title: `连通成功 · ${probe.latency_ms} ms`,
        });
      } else if (probe && probe.ok === false) {
        badgeNode = h("span", {
          cls: "latency-pill err",
          txt: "失败",
          title: cleanText(probe.error || "连接失败"),
        });
      }

      const chip = h("div", {
        cls: `prov-model-chip${isSel ? " selected" : ""}`,
        data: { model: mdl },
      },
        h("button", {
          type: "button",
          cls: "prov-model-toggle",
          title: mdl,
          on: {
            click: () => {
              const idx = providerModalState.selectedModels.indexOf(mdl);
              if (idx >= 0) {
                providerModalState.selectedModels.splice(idx, 1);
                if (providerModalState.defaultModel === mdl) {
                  providerModalState.defaultModel = providerModalState.selectedModels[0] || "";
                }
              } else {
                providerModalState.selectedModels.push(mdl);
                if (!providerModalState.defaultModel) {
                  providerModalState.defaultModel = mdl;
                }
              }
              providerModalState.modelsText = providerModalState.selectedModels.join(", ");
              if (el("inp-prov-models")) el("inp-prov-models").value = providerModalState.modelsText;
              renderProviderModelsGrid();
            },
          },
        },
          h("span", { cls: "prov-model-check" }, isSel ? createSvgIcon("check", 10) : null),
          h("span", { cls: "prov-model-name", txt: mdl })
        ),
        h("div", { cls: "prov-model-chip-right" },
          badgeNode,
          isSel
            ? h("button", {
                type: "button",
                cls: `prov-star-btn${isDef ? " is-default" : ""}`,
                title: isDef ? "当前默认模型" : "设为默认模型",
                attrs: { "aria-label": isDef ? "当前默认模型" : "设为默认模型" },
                on: {
                  click: (e) => {
                    e.stopPropagation();
                    providerModalState.defaultModel = mdl;
                    renderProviderModelsGrid();
                  },
                },
              }, createSvgIcon("star", 11))
            : null
        )
      );
      grid.appendChild(chip);
    }
  }

  function renderProviderModal() {
    const tabsBar = el("prov-tabs-bar");
    const providers = Array.isArray(copilotState.providers) ? copilotState.providers : [];
    if (tabsBar) {
      tabsBar.innerHTML = "";
      for (const p of providers) {
        const pid = String(p.id || "");
        const active = !providerModalState.isNew && providerModalState.editingId === pid;
        tabsBar.appendChild(
          h("button", {
            type: "button",
            cls: `prov-tab-chip${active ? " active" : ""}`,
            data: { provId: pid },
            on: {
              click: () => {
                _loadProviderIntoDraft(pid);
                renderProviderModal();
              },
            },
          },
            h("span", { cls: `status-dot ${p.ready ? "running" : "idle"}` }),
            h("span", { txt: cleanText(p.name || "供应商") })
          )
        );
      }
      tabsBar.appendChild(
        h("button", {
          type: "button",
          cls: `prov-tab-chip add-btn${providerModalState.isNew ? " active" : ""}`,
          id: "btn-prov-tab-new",
          txt: "+ 新增供应商",
          on: {
            click: () => {
              _loadProviderIntoDraft("", { isNew: true });
              renderProviderModal();
            },
          },
        })
      );
    }

    if (el("inp-prov-name")) el("inp-prov-name").value = providerModalState.name || "";
    if (el("inp-prov-base-url")) el("inp-prov-base-url").value = providerModalState.baseUrl || "";
    if (el("inp-prov-probe-model")) el("inp-prov-probe-model").value = providerModalState.probeModel || "";
    if (el("inp-prov-models")) el("inp-prov-models").value = providerModalState.modelsText || "";
    if (el("inp-prov-model-filter")) el("inp-prov-model-filter").value = providerModalState.modelFilter || "";

    document.querySelectorAll("#prov-format-seg .seg-item").forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.val === providerModalState.apiFormat);
    });

    const keyInp = el("inp-agent-api-key");
    const toggleBtn = el("btn-prov-toggle-key");
    if (keyInp) {
      keyInp.type = providerModalState.showKey ? "text" : "password";
      if (providerModalState.hasStoredKey) {
        keyInp.placeholder = "已安全存入系统凭据库（留空保持不变）";
      } else if (_isLoopbackUrlStr(providerModalState.baseUrl)) {
        keyInp.placeholder = "本机网关（如 127.0.0.1:3425）免鉴权可留空";
      } else {
        keyInp.placeholder = "粘贴 API 密钥，加密存入系统凭据库";
      }
    }
    if (toggleBtn) {
      toggleBtn.textContent = providerModalState.showKey ? "隐藏" : "显示";
    }

    const delBtn = el("btn-prov-delete");
    if (delBtn) {
      const canDel = !providerModalState.isNew && Boolean(providerModalState.editingId);
      delBtn.classList.toggle("hidden", !canDel);
      delBtn.classList.toggle("btn-danger", provDelArm.isArmed(providerModalState.editingId));
      delBtn.classList.toggle("btn-ghost", !provDelArm.isArmed(providerModalState.editingId));
      delBtn.textContent = provDelArm.isArmed(providerModalState.editingId) ? "再点确认删除" : "删除供应商";
    }

    const fetchBtn = el("btn-prov-fetch-models");
    if (fetchBtn) {
      fetchBtn.disabled = providerModalState.busyFetch;
      fetchBtn.textContent = providerModalState.busyFetch ? "获取中..." : "获取模型";
    }
    const detectBtn = el("btn-prov-detect-proto");
    if (detectBtn) {
      detectBtn.disabled = providerModalState.busyDetect;
      detectBtn.textContent = providerModalState.busyDetect ? "检测中..." : "检测协议";
    }
    const testBtn = el("btn-prov-test-models");
    if (testBtn) {
      testBtn.disabled = providerModalState.busyTest;
      testBtn.textContent = providerModalState.busyTest ? "测速中..." : "逐个检测已选模型";
    }

    _setProviderStatus(providerModalState.statusMsg, providerModalState.statusKind);
    renderProviderModelsGrid();
  }

  async function fetchProviderModelsAction({ silentOnErr = false } = {}) {
    const baseUrl = String((el("inp-prov-base-url") && el("inp-prov-base-url").value) || providerModalState.baseUrl || "").trim();
    providerModalState.baseUrl = baseUrl;
    if (!baseUrl) {
      if (!silentOnErr) _setProviderStatus("请先填写网关 Base URL", "err");
      return;
    }
    const keyInp = el("inp-agent-api-key");
    const typedKey = keyInp ? String(keyInp.value || "").trim() : "";

    providerModalState.busyFetch = true;
    renderProviderModal();
    try {
      const res = await apiCall("fetch_agent_models", {
        provider_id: providerModalState.isNew ? "" : providerModalState.editingId,
        base_url: baseUrl,
        api_format: providerModalState.apiFormat,
        api_key: typedKey,
      });
      if (res && res.ok) {
        const ids = Array.isArray(res.model_ids) ? res.model_ids : [];
        providerModalState.fetchedModels = ids;
        if (res.model_capabilities && typeof res.model_capabilities === "object") {
          providerModalState.fetchedCapabilities = Object.assign(
            {},
            providerModalState.fetchedCapabilities || {},
            res.model_capabilities
          );
        }
        if (ids.length > 0 && providerModalState.selectedModels.length === 0) {
          providerModalState.selectedModels = ids.slice(0, 6);
          providerModalState.defaultModel = providerModalState.selectedModels[0];
          providerModalState.modelsText = providerModalState.selectedModels.join(", ");
        }
        _setProviderStatus(res.message || `已从网关读取 ${ids.length} 个可用模型`, "ok");
      } else if (!silentOnErr) {
        _setProviderStatus((res && res.message) || "获取模型失败，请确认网关地址与端口", "err");
      }
    } finally {
      providerModalState.busyFetch = false;
      renderProviderModal();
    }
  }

  function openProviderModal({ providerId = null, isNew = false, presetMagpie = false } = {}) {
    const backdrop = el("provider-modal");
    if (!backdrop) return;
    if (copilotState.popoverOpen) {
      toggleCopilotConfigPopover(false);
    }
    const providers = Array.isArray(copilotState.providers) ? copilotState.providers : [];
    const wantNew = Boolean(isNew || (providers.length === 0 && !providerId));
    _loadProviderIntoDraft(providerId, { isNew: wantNew, presetMagpie });
    providerModalState.open = true;
    renderProviderModal();
    backdrop.classList.remove("hidden");
    backdrop.removeAttribute("inert");
    backdrop.inert = false;
    backdrop.setAttribute("aria-hidden", "false");
    if (presetMagpie) {
      fetchProviderModelsAction({ silentOnErr: false });
    }
  }

  function closeProviderModal() {
    const backdrop = el("provider-modal");
    if (!backdrop) return;
    providerModalState.open = false;
    provDelArm.disarm();
    const keyInp = el("inp-agent-api-key");
    if (keyInp) {
      keyInp.value = "";
      keyInp.type = "password";
    }
    if (backdrop.contains(document.activeElement) && document.activeElement.blur) {
      document.activeElement.blur();
    }
    backdrop.classList.add("hidden");
    backdrop.setAttribute("aria-hidden", "true");
    backdrop.setAttribute("inert", "");
    backdrop.inert = true;
  }

  function syncAgentStatePayload(data, { replaceItems = false, forceReplace = false } = {}) {
    if (!data || typeof data !== "object") return;
    const cfg = (data.config && typeof data.config === "object")
      ? data.config
      : ((data.prefs && typeof data.prefs === "object") ? data.prefs : data);
    const brain = (data.brain && typeof data.brain === "object") ? data.brain : data;

    const prevSessionId = String(copilotState.activeSessionId || "");
    copilotState.loaded = true;
    if (data.active_session_id !== undefined) {
      copilotState.activeSessionId = String(data.active_session_id || "");
    }
    if (Array.isArray(data.sessions)) {
      copilotState.sessions = data.sessions;
    }
    if (data.context_stats && typeof data.context_stats === "object") {
      const cs = data.context_stats;
      copilotState.contextStats = {
        messageCount: Number(cs.message_count != null ? cs.message_count : (cs.messageCount || 0)),
        maxMessages: Number(cs.max_messages != null ? cs.max_messages : (cs.maxMessages || 40)),
        approxChars: Number(cs.approx_chars != null ? cs.approx_chars : (cs.approxChars || 0)),
        approxTokens: Number(cs.approx_tokens != null ? cs.approx_tokens : (cs.approxTokens || 0)),
      };
    }
    const drvVal = cfg.agent_driver || brain.driver || data.driver;
    if (drvVal === "claude_code" || drvVal === "native") {
      copilotState.driver = drvVal;
    }
    if (brain.claude_code_available !== undefined) {
      copilotState.claudeAvailable = Boolean(brain.claude_code_available);
    } else if (data.claude_available !== undefined) {
      copilotState.claudeAvailable = Boolean(data.claude_available);
    }
    const srcVal = cfg.agent_brain_source || brain.brain_source || data.brain_source;
    if (["auto", "colab_llm", "external_api"].includes(srcVal)) {
      copilotState.brainSource = srcVal;
    }
    const fmtVal = cfg.agent_api_format || data.api_format;
    if (["openai", "openai_responses", "anthropic", "gemini"].includes(fmtVal)) {
      copilotState.apiFormat = fmtVal;
    }
    if (cfg.agent_api_base_url !== undefined) copilotState.apiBaseUrl = String(cfg.agent_api_base_url || "");
    else if (data.api_base_url !== undefined) copilotState.apiBaseUrl = String(data.api_base_url || "");
    if (cfg.agent_model !== undefined) copilotState.model = String(cfg.agent_model || "");
    else if (data.model !== undefined) copilotState.model = String(data.model || "");
    if (cfg.agent_active_provider_id !== undefined) {
      copilotState.activeProviderId = String(cfg.agent_active_provider_id || "");
    } else if (brain.active_provider_id !== undefined) {
      copilotState.activeProviderId = String(brain.active_provider_id || "");
    }
    if (brain.active_provider_name !== undefined) {
      copilotState.activeProviderName = String(brain.active_provider_name || "");
    }
    if (brain.resolved_source !== undefined) {
      copilotState.resolvedSource = String(brain.resolved_source || "auto");
    }
    const effVal = String(brain.reasoning_effort || cfg.agent_reasoning_effort || "").trim().toLowerCase();
    if (["auto", "none", "minimal", "low", "medium", "high", "xhigh", "max"].includes(effVal)) {
      copilotState.reasoningEffort = effVal;
    }
    if (Array.isArray(brain.supported_reasoning_efforts)) {
      copilotState.supportedReasoningEfforts = brain.supported_reasoning_efforts.slice();
    }
    if (brain.reasoning_supported !== undefined) {
      copilotState.reasoningSupported = Boolean(brain.reasoning_supported);
    }
    if (brain.default_reasoning_effort !== undefined) {
      copilotState.defaultReasoningEffort = String(brain.default_reasoning_effort || "");
    }
    if (brain.reasoning_spec_source !== undefined) {
      copilotState.reasoningSpecSource = String(brain.reasoning_spec_source || "fallback");
    }
    if (Array.isArray(cfg.providers)) {
      copilotState.providers = cfg.providers;
    }
    const syncPid = String(copilotState.activeProviderId || "");
    if (
      syncPid
      && syncPid !== "__legacy__"
      && copilotState.reasoningSpecSource !== "gateway"
      && !copilotState._syncedCapProvIds.has(syncPid)
    ) {
      copilotState._syncedCapProvIds.add(syncPid);
      setTimeout(async () => {
        try {
          const capRes = await apiCall("fetch_agent_models", { provider_id: syncPid });
          if (capRes && capRes.ok && capRes.model_capabilities && Object.keys(capRes.model_capabilities).length > 0) {
            await loadAgentState(false);
          }
        } catch (_e) {
          /* 网关离线时保持静默，沿用本地推断档位 */
        }
      }, 60);
    }
    if (cfg.has_api_key !== undefined) copilotState.hasApiKey = Boolean(cfg.has_api_key);
    else if (brain.has_external_key !== undefined) copilotState.hasApiKey = Boolean(brain.has_external_key);
    else if (data.has_api_key !== undefined) copilotState.hasApiKey = Boolean(data.has_api_key);
    else if (data.api_key_configured !== undefined) copilotState.hasApiKey = Boolean(data.api_key_configured);

    if (cfg.agent_auto_approve_billing !== undefined) {
      copilotState.autoApproveBilling = Boolean(cfg.agent_auto_approve_billing);
    } else if (brain.auto_approve_billing !== undefined) {
      copilotState.autoApproveBilling = Boolean(brain.auto_approve_billing);
    } else if (data.auto_approve_billing !== undefined) {
      copilotState.autoApproveBilling = Boolean(data.auto_approve_billing);
    }
    if (brain.colab_llm_online !== undefined) copilotState.colabLlmOnline = Boolean(brain.colab_llm_online);
    else if (data.colab_llm_online !== undefined) copilotState.colabLlmOnline = Boolean(data.colab_llm_online);
    if (brain.resolved_brain_label) copilotState.resolvedBrainLabel = cleanText(brain.resolved_brain_label);
    else if (data.resolved_brain_label) copilotState.resolvedBrainLabel = cleanText(data.resolved_brain_label);
    if (data.busy !== undefined) copilotState.busy = Boolean(data.busy);

    const sessionChanged = Boolean(
      prevSessionId
      && copilotState.activeSessionId
      && copilotState.activeSessionId !== prevSessionId
    );
    const shouldRebuild = Boolean(forceReplace || sessionChanged || (replaceItems && copilotState.items.length === 0));
    if (shouldRebuild && Array.isArray(data.messages) && !copilotState.busy) {
      const rebuilt = [];
      for (const msg of data.messages) {
        if (!msg || typeof msg !== "object") continue;
        const role = String(msg.role || "");
        if (role === "user") {
          rebuilt.push({
            kind: "user",
            text: String(msg.content || msg.text || ""),
            ctxTag: String(msg.ctx_tag || ""),
            ctxIcon: String(msg.ctx_icon || "image"),
          });
        } else if (role === "assistant") {
          if (Array.isArray(msg.tool_calls)) {
            for (const tc of msg.tool_calls) {
              if (!tc || typeof tc !== "object") continue;
              rebuilt.push({
                kind: "tool",
                callId: String(tc.call_id || tc.id || ""),
                name: String(tc.name || "tool"),
                args: tc.args || {},
                tier: String(tc.tier || "exec"),
                status: tc.status === "error" ? "error" : "ok",
                summary: cleanText(tc.summary || ""),
                detail: tc.detail !== undefined ? tc.detail : "",
                elapsedMs: tc.elapsed_ms != null ? tc.elapsed_ms : null,
              });
            }
          }
          const content = String(msg.content || msg.text || "");
          const reasoning = String(msg.reasoning || "");
          if (content.trim() || reasoning.trim()) {
            rebuilt.push({
              kind: "assistant",
              text: content,
              reasoning,
              thinkOpen: false,
              streamingReasoning: false,
              streaming: false,
              isError: Boolean(msg.error),
            });
          }
        }
      }
      copilotState.pendingConfirm = null;
      if (data.pending_confirm && typeof data.pending_confirm === "object") {
        const pc = data.pending_confirm;
        const cid = String(pc.id || pc.confirm_id || "");
        if (cid) {
          copilotState.pendingConfirm = pc;
          rebuilt.push({
            kind: "confirm",
            confirmId: cid,
            tool: pc.tool || "manage_compute_instance",
            title: pc.title || "计费与高危操作审批",
            detail: pc.detail || "",
            args: pc.args || {},
            resolvedState: "",
          });
        }
      }
      copilotState.items = rebuilt;
      copilotState.activeStreamIdx = -1;
      copilotState.stickToBottom = true;
      renderCopilotMessages();
    }

    renderCopilotHeaderAndControls();
    if (copilotState.popoverOpen) renderCopilotPopoverControls();
    if (copilotState.effortPopoverOpen) renderCopilotEffortPopover();
    if (copilotState.historyPopoverOpen) renderCopilotHistoryPopover();
  }

  async function loadAgentState(replaceItems = false, forceReplace = false) {
    const res = await apiCall("get_agent_state");
    if (res && typeof res === "object") {
      syncAgentStatePayload(res, { replaceItems, forceReplace });
    }
  }

  function setCopilotOpen(openFlag, { focusInput = true } = {}) {
    const next = Boolean(openFlag);
    copilotState.open = next;
    if (!next) {
      closeAllCopilotPopovers();
    }
    syncRightPanelsLayout();
    renderCopilotHeaderAndControls();
    renderCopilotChips();
    if (next) {
      if (!copilotState.loaded) loadAgentState(true);
      if (copilotState.items.length === 0) {
        renderCopilotMessages();
      } else if (copilotState.stickToBottom) {
        scrollCopilotBottom(true);
        setTimeout(() => {
          if (copilotState.open && copilotState.stickToBottom) scrollCopilotBottom(true);
        }, 260);
      } else {
        updateCopilotScrollBtn();
      }
      if (focusInput) {
        setTimeout(() => {
          const inp = el("copilot-input");
          if (inp && copilotState.open) inp.focus();
        }, 40);
      }
    }
  }

  window.openStudioCopilot = function (presetText) {
    setCopilotOpen(true, { focusInput: false });
    const inp = el("copilot-input");
    if (typeof presetText === "string" && presetText.trim() && inp) {
      inp.value = stripEmoji(presetText);
      resizeCopilotInput();
    }
    if (inp) {
      setTimeout(() => {
        inp.focus();
        const len = inp.value.length;
        try {
          inp.setSelectionRange(len, len);
        } catch (_) {
          /* 忽略受限框选区异常 */
        }
      }, 45);
    }
  };

  async function sendCopilotMessage(overrideText = null) {
    if (copilotState.busy) {
      showToast("副驾正在执行当前任务，可点击停止后再发新指令");
      return;
    }
    const inp = el("copilot-input");
    const raw = overrideText !== null ? String(overrideText) : String((inp && inp.value) || "");
    const text = stripEmoji(raw).trim();
    if (!text) return;

    const snappedCtx = copilotState.attachedContext ? { ...copilotState.attachedContext } : null;
    if (overrideText === null && inp) {
      inp.value = "";
      resizeCopilotInput();
    }
    if (snappedCtx) clearCopilotContext();

    if (!copilotState.open) {
      setCopilotOpen(true, { focusInput: false });
    }
    closeAllCopilotPopovers();

    finalizeCopilotStreamItem();
    appendCopilotItem({
      kind: "user",
      text,
      ctxTag: snappedCtx ? (snappedCtx.name || snappedCtx.relPath || "") : "",
      ctxIcon: snappedCtx && snappedCtx.kind === "logs" ? "terminal" : "image",
    });
    copilotState.busy = true;
    renderCopilotHeaderAndControls();

    const ctxPayload = {
      view: currentView,
      selected_rel_path: galleryState.selectedPath || "",
      console_source: consoleState.source || "local_diag",
    };
    if (snappedCtx) {
      if (snappedCtx.relPath) ctxPayload.attached_rel_path = snappedCtx.relPath;
      if (snappedCtx.snippet) ctxPayload.attached_logs = snappedCtx.snippet;
    }
    const payload = {
      message: text,
      text,
      reasoning_effort: copilotState.reasoningEffort || "auto",
      ctx_tag: snappedCtx ? (snappedCtx.name || snappedCtx.relPath || "") : "",
      context: ctxPayload,
    };
    copilotState.turnEmittedText = false;
    const res = await apiCall("send_agent_message", payload);
    if (res && res.ok === false) {
      copilotState.busy = false;
      finalizeCopilotStreamItem();
      appendCopilotItem({
        kind: "assistant",
        text: cleanText(res.message || "发送指令失败，请检查副驾引擎或 API 配置。"),
        streaming: false,
        isError: true,
      });
      renderCopilotHeaderAndControls();
      if (res.needs_config) {
        toggleCopilotConfigPopover(true);
      }
      if (res.message) showToast(res.message, { isError: true });
    } else if (res && res.done) {
      copilotState.busy = false;
      finalizeCopilotStreamItem();
      renderCopilotHeaderAndControls();
    }
  }

  function handleAgentEvent(evt) {
    if (!evt || typeof evt !== "object") return;
    const phase = String(evt.phase || evt.event || "").toLowerCase();

    if (phase === "turn_start") {
      copilotState.busy = true;
      copilotState.turnEmittedText = false;
      renderCopilotHeaderAndControls();
      return;
    }

    if (phase === "reasoning_delta") {
      const rChunk = stripEmoji(evt.text !== undefined ? evt.text : (evt.delta || ""));
      if (!rChunk) return;
      if (copilotState.activeStreamIdx >= 0 && copilotState.items[copilotState.activeStreamIdx]) {
        const cur = copilotState.items[copilotState.activeStreamIdx];
        cur.reasoning = `${cur.reasoning || ""}${rChunk}`;
        cur.streamingReasoning = true;
        if (!cur.userToggledThink) cur.thinkOpen = true;
        syncCopilotThinkCard(cur);
      } else {
        const item = {
          kind: "assistant",
          text: "",
          reasoning: rChunk,
          thinkOpen: true,
          userToggledThink: false,
          streamingReasoning: true,
          streaming: true,
          isError: false,
        };
        appendCopilotItem(item);
        copilotState.activeStreamIdx = copilotState.items.length - 1;
      }
      scrollCopilotBottom();
      copilotState.busy = true;
      renderCopilotHeaderAndControls();
      return;
    }

    if (phase === "delta") {
      const chunk = stripEmoji(evt.text !== undefined ? evt.text : (evt.delta || ""));
      if (!chunk) return;
      copilotState.turnEmittedText = true;
      if (copilotState.activeStreamIdx >= 0 && copilotState.items[copilotState.activeStreamIdx]) {
        const cur = copilotState.items[copilotState.activeStreamIdx];
        cur.text = `${cur.text || ""}${chunk}`;
        cur.streaming = true;
        if (cur.streamingReasoning) {
          cur.streamingReasoning = false;
          if (!cur.userToggledThink) cur.thinkOpen = false;
          syncCopilotThinkCard(cur);
        }
        if (cur._richEl) {
          cur._richEl.classList.remove("hidden");
          renderCopilotRichText(cur._richEl, cur.text, true);
        }
        scrollCopilotBottom();
      } else {
        const item = {
          kind: "assistant",
          text: chunk,
          reasoning: "",
          thinkOpen: false,
          streamingReasoning: false,
          streaming: true,
          isError: false,
        };
        appendCopilotItem(item);
        copilotState.activeStreamIdx = copilotState.items.length - 1;
      }
      copilotState.busy = true;
      renderCopilotHeaderAndControls();
      return;
    }

    if (phase === "tool_start") {
      finalizeCopilotStreamItem();
      const toolItem = {
        kind: "tool",
        callId: String(evt.call_id || evt.id || ""),
        name: String(evt.name || evt.tool || "tool"),
        args: evt.args !== undefined ? evt.args : (evt.input || {}),
        tier: String(evt.tier || "safe"),
        status: "running",
        summary: "",
        detail: evt.detail || "",
        elapsedMs: null,
      };
      appendCopilotItem(toolItem);
      copilotState.busy = true;
      renderCopilotHeaderAndControls();
      return;
    }

    if (phase === "tool_end") {
      const cid = String(evt.call_id || evt.id || "");
      const tName = String(evt.name || evt.tool || "");
      let target = null;
      for (let i = copilotState.items.length - 1; i >= 0; i--) {
        const it = copilotState.items[i];
        if (it.kind !== "tool" || it.status !== "running") continue;
        if (cid && it.callId === cid) {
          target = it;
          break;
        }
        if (!target && (!tName || it.name === tName)) {
          target = it;
        }
      }
      if (target) {
        target.status = evt.ok === false ? "error" : "ok";
        target.summary = cleanText(evt.summary || "");
        target.elapsedMs = evt.elapsed_ms != null ? evt.elapsed_ms : null;
        if (evt.detail !== undefined || evt.result !== undefined) {
          target.detail = evt.detail !== undefined ? evt.detail : evt.result;
        }
        updateCopilotToolCardNode(target);
        scrollCopilotBottom();
      } else {
        appendCopilotItem({
          kind: "tool",
          callId: cid,
          name: tName || "tool",
          args: evt.args || {},
          status: evt.ok === false ? "error" : "ok",
          summary: cleanText(evt.summary || ""),
          detail: evt.detail !== undefined ? evt.detail : evt.result,
          elapsedMs: evt.elapsed_ms != null ? evt.elapsed_ms : null,
        });
      }
      return;
    }

    if (phase === "confirm_required") {
      finalizeCopilotStreamItem();
      const c = (evt.confirm && typeof evt.confirm === "object") ? evt.confirm : evt;
      const cid = String(c.id || c.confirm_id || `cfm_${Date.now()}`);
      const existing = copilotState.items.find((it) => it.kind === "confirm" && it.confirmId === cid);
      if (!existing) {
        const confirmItem = {
          kind: "confirm",
          confirmId: cid,
          tool: String(c.tool || "manage_compute_instance"),
          title: cleanText(c.title || "计费与高危操作审批"),
          detail: stripEmoji(c.detail || ""),
          args: c.args || {},
          resolvedState: "",
        };
        copilotState.pendingConfirm = c;
        appendCopilotItem(confirmItem);
        if (!copilotState.open) setCopilotOpen(true, { focusInput: false });
      }
      renderCopilotHeaderAndControls();
      return;
    }

    if (phase === "confirm_resolved") {
      const cid = String(evt.confirm_id || (evt.confirm && evt.confirm.id) || "");
      const approved = Boolean(evt.approved);
      const target = copilotState.items.find((it) => it.kind === "confirm" && (!cid || it.confirmId === cid));
      if (target && !target.resolvedState) {
        target.resolvedState = approved ? "approved" : "rejected";
        replaceCopilotConfirmCardNode(target);
      }
      copilotState.pendingConfirm = null;
      renderCopilotHeaderAndControls();
      return;
    }

    if (phase === "config_updated") {
      syncAgentStatePayload(evt, { replaceItems: false });
      return;
    }

    if (phase === "history_cleared") {
      if (!copilotState.busy) {
        copilotState.items = [];
        copilotState.activeStreamIdx = -1;
        copilotState.pendingConfirm = null;
        copilotState.stickToBottom = true;
        renderCopilotMessages();
        renderCopilotHeaderAndControls();
        loadAgentState(false);
      }
      return;
    }

    if (phase === "session_changed") {
      if (!copilotState.busy) {
        copilotState.activeStreamIdx = -1;
        copilotState.pendingConfirm = null;
        copilotState.stickToBottom = true;
        loadAgentState(true, true);
      }
      return;
    }

    if (phase === "turn_done" || phase === "turn_cancelled") {
      const wasCancelled = phase === "turn_cancelled" || Boolean(evt.cancelled);
      copilotState.busy = false;
      if (copilotState.activeStreamIdx >= 0 && copilotState.items[copilotState.activeStreamIdx]) {
        const curDone = copilotState.items[copilotState.activeStreamIdx];
        if (!copilotState.turnEmittedText && evt.text && String(evt.text).trim()) {
          curDone.text = String(evt.text);
        }
        finalizeCopilotStreamItem();
        scrollCopilotBottom();
      } else if ((evt.text && String(evt.text).trim()) || (evt.reasoning && String(evt.reasoning).trim())) {
        appendCopilotItem({
          kind: "assistant",
          text: String(evt.text || ""),
          reasoning: String(evt.reasoning || ""),
          thinkOpen: false,
          streamingReasoning: false,
          streaming: false,
          isError: false,
        });
      }
      copilotState.turnEmittedText = false;
      renderCopilotHeaderAndControls();
      if (wasCancelled) {
        showToast(cleanText(evt.reason || "已停止副驾当前生成"));
      }
      loadAgentState(false);
      return;
    }

    if (phase === "turn_error") {
      copilotState.busy = false;
      finalizeCopilotStreamItem();
      const errMsg = cleanText(evt.message || evt.error || "执行发生异常");
      if (errMsg) {
        appendCopilotItem({ kind: "assistant", text: errMsg, streaming: false, isError: true });
      }
      renderCopilotHeaderAndControls();
      if (errMsg) showToast(errMsg, { isError: true });
    }
  }

  /* ----------------------------------------------------------------
     轻量账号绑定 / 重命名模态弹窗
     ---------------------------------------------------------------- */
  function openModal(mode, opts = {}) {
    const backdrop = el("studio-modal");
    const titleEl = el("modal-title");
    const bodyEl = el("modal-body");
    if (!backdrop || !titleEl || !bodyEl) return;

    bodyEl.innerHTML = "";
      if (mode === "rename" && opts.account) {
      const acct = opts.account;
      titleEl.textContent = "修改账号备注名";
      const inp = h("input", {
        cls: "mac-input",
        id: "inp-modal-rename",
        type: "text",
        value: cleanText(acct.label || ""),
        maxLength: 16,
        placeholder: "最多 16 个字符",
      });
      bodyEl.append(
        h("span", { cls: "metric-sub", txt: `账号: ${acct.email_masked || acct.id}` }),
        inp,
        h("div", { cls: "details-actions" },
          h("button", {
            type: "button",
            cls: "btn btn-sm btn-primary",
            txt: "保存备注",
            on: {
              click: async () => {
                const res = await apiCall("rename_colab_account", acct.id, inp.value);
                if (res && res.snapshot) window.renderStudioState(res.snapshot);
                if (res && res.ok) {
                  closeModal();
                  showToast(res.message || "已更新备注");
                } else {
                  showToast((res && res.message) || "保存失败", { isError: true });
                }
              },
            },
          }),
          h("button", { type: "button", cls: "btn btn-sm btn-ghost", txt: "取消", on: { click: closeModal } })
        )
      );
    } else if (mode === "png_choice" && opts.path) {
      titleEl.textContent = "检测到内嵌工作流 PNG";
      bodyEl.append(
        h("span", { cls: "metric-sub", txt: `文件: ${opts.name || opts.path}` }),
        h("span", { cls: "metric-sub", txt: "请选择投递至云端画布的方式：" }),
        h("div", { cls: "details-actions" },
          h("button", {
            type: "button",
            cls: "btn btn-sm btn-primary",
            txt: "提取并直铺工作流",
            on: {
              click: async () => {
                closeModal();
                showToast("正在提取并直铺工作流到画布...");
                await apiCall("confirm_png_drop", opts.path, "workflow");
              },
            },
          }),
          h("button", {
            type: "button",
            cls: "btn btn-sm btn-subtle",
            txt: "作为参考图上传",
            on: {
              click: async () => {
                closeModal();
                showToast("正在按参考图上传...");
                await apiCall("confirm_png_drop", opts.path, "image");
              },
            },
          }),
          h("button", { type: "button", cls: "btn btn-sm btn-ghost", txt: "取消", on: { click: closeModal } })
        )
      );
    } else if (mode === "dual_image_choice" && Array.isArray(opts.paths)) {
      titleEl.textContent = "双栖模式图片投递分流";
      bodyEl.append(
        h("span", { cls: "metric-sub", txt: `已拖入 ${opts.count || opts.paths.length} 张图片，请选择投递目标：` }),
        h("div", { cls: "details-actions" },
          h("button", {
            type: "button",
            cls: "btn btn-sm btn-primary",
            txt: "传至 ComfyUI 参考图",
            on: {
              click: async () => {
                closeModal();
                showToast("正在投递参考图至 ComfyUI...");
                await apiCall("confirm_dual_image_drop", opts.paths, "comfy");
              },
            },
          }),
          h("button", {
            type: "button",
            cls: "btn btn-sm btn-subtle",
            txt: "打开 Qwen3.8 视觉识图",
            on: {
              click: async () => {
                closeModal();
                showToast("正在打开 Qwen3.8 对话页...");
                await apiCall("confirm_dual_image_drop", opts.paths, "llm");
              },
            },
          }),
          h("button", { type: "button", cls: "btn btn-sm btn-ghost", txt: "取消", on: { click: closeModal } })
        )
      );
    } else {
      titleEl.textContent = "绑定 Google Colab 账号";
      const inpLabel = h("input", { cls: "mac-input", type: "text", maxLength: 16, placeholder: "备注名（选填，如 Ultra 主号 / Pro 副号）" });
      const inpCode = h("input", { cls: "mac-input", type: "text", placeholder: "粘贴授权码（留空可自动读取剪贴板）" });
      bodyEl.append(
        h("span", { cls: "metric-sub", txt: "在浏览器完成 Google 登录并复制授权码后，点击下方提交完成绑定。" }),
        h("label", { cls: "field-group" }, h("span", { cls: "field-label", txt: "账号备注名" }), inpLabel),
        h("label", { cls: "field-group" }, h("span", { cls: "field-label", txt: "Google 授权码" }), inpCode),
        h("div", { cls: "details-actions" },
          h("button", {
            type: "button",
            cls: "btn btn-sm btn-primary",
            txt: "验证并绑定",
            on: {
              click: async () => {
                const res = await apiCall("complete_colab_oauth", String(inpCode.value || "").trim(), String(inpLabel.value || "").trim());
                if (res && res.snapshot) window.renderStudioState(res.snapshot);
                if (res && res.ok) {
                  closeModal();
                  showToast(res.message || "账号绑定成功");
                } else {
                  showToast((res && res.message) || "绑定失败", { isError: true });
                }
              },
            },
          }),
          h("button", {
            type: "button",
            cls: "btn btn-sm btn-subtle",
            txt: "重开授权页",
            on: {
              click: async () => {
                const res = await apiCall("start_colab_oauth");
                if (res && res.snapshot) window.renderStudioState(res.snapshot);
                if (res && res.message) showToast(res.message, { isError: res.ok === false });
              },
            },
          }),
          h("button", {
            type: "button",
            cls: "btn btn-sm btn-ghost",
            txt: "取消",
            on: {
              click: async () => {
                const res = await apiCall("cancel_colab_oauth");
                if (res && res.snapshot) window.renderStudioState(res.snapshot);
                closeModal();
              },
            },
          })
        )
      );
    }

    backdrop.classList.remove("hidden");
    backdrop.removeAttribute("inert");
    backdrop.inert = false;
    backdrop.setAttribute("aria-hidden", "false");
    const firstInput = bodyEl.querySelector("input");
    if (firstInput) setTimeout(() => firstInput.focus(), 30);
  }

  function closeModal() {
    const backdrop = el("studio-modal");
    if (!backdrop) return;
    if (backdrop.contains(document.activeElement) && document.activeElement.blur) {
      document.activeElement.blur();
    }
    backdrop.classList.add("hidden");
    backdrop.setAttribute("aria-hidden", "true");
    backdrop.setAttribute("inert", "");
    backdrop.inert = true;
  }

  /* ----------------------------------------------------------------
     全局双向状态渲染与事件总线
     ---------------------------------------------------------------- */
  window.renderStudioState = function (snap) {
    if (!snap || typeof snap !== "object") return;
    applyCatalogFromSnap(snap);
    latestSnap = snap;
    syncBootPrefsFromSnap(snap);
    renderSidebarStatus(snap);
    renderSidebarControlCards(snap);
    renderOverviewHero(snap);
    renderOverviewTopology(snap);
    renderAccountPool(snap);
    renderTunerView(snap);
    renderRightLogPanel();
  };

  let _lastRlogTickSig = "";

  window.onStudioBackendEvent = function (evt) {
    if (!evt || !evt.type) return;
    if (evt.type === "agent_event") {
      handleAgentEvent(evt);
      return;
    }
    if (evt.type === "state_sync" && evt.snapshot) {
      window.renderStudioState(evt.snapshot);
      return;
    }
    if (evt.type === "comfy_tick") {
      if (latestSnap) {
        latestSnap.comfy = evt.comfy;
        if (evt.comfy && evt.comfy.has_error) latestSnap.mood = "sad";
        renderSidebarStatus(latestSnap);
        renderSidebarControlCards(latestSnap);
        renderOverviewTopology(latestSnap);
        renderTunerView(latestSnap);
        const c = evt.comfy || {};
        const tickSig = c.has_error
          ? `err:${c.error_summary || ""}`
          : (c.current_node ? `run:${c.current_node}` : (c.tunnel_url ? `up:${c.tunnel_url}` : ""));
        if (tickSig && tickSig !== _lastRlogTickSig) {
          _lastRlogTickSig = tickSig;
          if (c.has_error) {
            appendRightLogEvent(c.error_summary || "ComfyUI 节点异常", "err", "comfy");
          } else if (c.current_node) {
            appendRightLogEvent(`正在执行节点：${c.current_node}`, "", "comfy");
          }
        }
        renderRightLogPanel();
      }
      return;
    }
    if (evt.type === "deploy_progress") {
      if (latestSnap) {
        latestSnap.deploy = evt.deploy;
        const dep = evt.deploy || {};
        appendRightLogEvent(dep.sub_detail || dep.stage_title || "部署推进中...", "accent", "all");
        renderSidebarStatus(latestSnap);
        renderSidebarControlCards(latestSnap);
        renderOverviewTopology(latestSnap);
        renderRightLogPanel();
      }
      return;
    }
    if (evt.type === "deploy_done" || evt.type === "deploy_error") {
      if (evt.summary) {
        appendRightLogEvent(evt.summary, evt.type === "deploy_error" && !evt.cancelled ? "err" : "ok", "all");
      }
      if (evt.snapshot) {
        window.renderStudioState(evt.snapshot);
      } else if (latestSnap) {
        latestSnap.deploy = evt.deploy;
        renderSidebarStatus(latestSnap);
        renderSidebarControlCards(latestSnap);
        renderOverviewTopology(latestSnap);
        renderRightLogPanel();
      }
      if (evt.summary) showToast(evt.summary, { isError: evt.type === "deploy_error" && !evt.cancelled });
      return;
    }
    if (evt.type === "outputs_ready") {
      appendRightLogEvent(evt.summary || "新生成作品已同步到本地", "ok", "comfy");
      renderRightLogPanel();
      showToast(evt.summary || "新生成作品已同步到本地");
      if (currentView === "gallery" || galleryState.loaded) {
        loadGalleryItems();
      }
      return;
    }
    if (evt.type === "png_workflow_choice" && evt.path) {
      openModal("png_choice", { path: evt.path, name: evt.name });
      return;
    }
    if (evt.type === "dual_image_choice" && Array.isArray(evt.paths)) {
      openModal("dual_image_choice", { paths: evt.paths, count: evt.count });
      return;
    }
    if (evt.type === "upload_done" || evt.type === "upload_error" || evt.type === "comfy_error" || evt.type === "idle_stopped" || evt.type === "idle_archive_failed" || evt.type === "onboarding_event") {
      if (evt.summary) {
        const isErr = evt.type.endsWith("error") || evt.type === "idle_archive_failed" || evt.ok === false;
        appendRightLogEvent(evt.summary, isErr ? "err" : "warn", evt.type.startsWith("comfy") ? "comfy" : "all");
      }
      if (evt.snapshot) {
        window.renderStudioState(evt.snapshot);
      } else {
        renderRightLogPanel();
      }
      if (evt.summary) {
        showToast(evt.summary, {
          isError: evt.type.endsWith("error") || evt.type === "idle_archive_failed" || evt.ok === false,
        });
      }
    }
  };

  /* ----------------------------------------------------------------
     事件绑定初始化
     ---------------------------------------------------------------- */
  function bindInteractions() {
    // 左侧导航切页
    document.querySelectorAll(".sidebar-nav .nav-item").forEach((btn) => {
      btn.addEventListener("click", () => {
        if (btn.dataset.nav) switchStudioTab(btn.dataset.nav);
      });
    });

    if (el("btn-theme-toggle")) {
      el("btn-theme-toggle").addEventListener("click", toggleTheme);
    }

    if (el("btn-sb-sync")) {
      el("btn-sb-sync").addEventListener("click", async () => {
        const icon = el("btn-sb-sync").querySelector("svg");
        if (icon) icon.classList.add("spin");
        try {
          const snap = await apiCall("refresh_now");
          if (snap) window.renderStudioState(snap);
          showToast("已与云端实时同步");
        } finally {
          if (icon) icon.classList.remove("spin");
        }
      });
    }

    if (el("btn-title-primary")) {
      el("btn-title-primary").addEventListener("click", async () => {
        if (currentView === "overview") {
          const snap = await apiCall("refresh_now");
          if (snap) window.renderStudioState(snap);
          showToast("已刷新算力池状态");
        } else if (currentView === "gallery") {
          const res = await apiCall("sync_outputs");
          if (res && res.message) showToast(res.message, { isError: res.ok === false });
          await loadGalleryItems();
        } else if (currentView === "tuner") {
          await loadCloudAssets(true);
        } else if (currentView === "console") {
          await runDoctorPreflight(true);
        }
      });
    }

    // 视图 1：工作流分类、集群模式、开机与分槽停卡
    if (el("ov-wf-cat-seg")) {
      el("ov-wf-cat-seg").addEventListener("click", (e) => {
        const btn = e.target.closest(".seg-item[data-cat]");
        if (!btn) return;
        wfCategory = btn.dataset.cat || "all";
        renderBootWorkflowList();
      });
    }

    if (el("ov-cluster-seg")) {
      el("ov-cluster-seg").addEventListener("click", (e) => {
        const btn = e.target.closest(".seg-item[data-mode]");
        if (!btn) return;
        userTouchedBoot = true;
        bootState.clusterSplitMode = btn.dataset.mode === "split";
        apiCall("save_prefs", { cluster_split_mode: bootState.clusterSplitMode });
        renderBootCockpit(latestSnap);
      });
    }

    if (el("btn-boot-hm")) {
      el("btn-boot-hm").addEventListener("click", () => {
        const spec = GPU_SPECS[bootState.gpu] || {};
        if (spec.hmLock) return;
        userTouchedBoot = true;
        bootState.high_mem = !bootState.high_mem;
        apiCall("save_prefs", { boot_high_mem: bootState.high_mem });
        renderBootCockpit(latestSnap);
      });
    }

    if (el("btn-ov-toggle-boot")) {
      el("btn-ov-toggle-boot").addEventListener("click", () => {
        forceShowBootStandby = !forceShowBootStandby;
        renderOverviewTopology(latestSnap);
      });
    }

    if (el("btn-ov-boot-start")) {
      el("btn-ov-boot-start").addEventListener("click", async () => {
        const btn = el("btn-ov-boot-start");
        const lbl = el("btn-ov-boot-label");
        if (btn.disabled) return;
        const profs = normalizeProfiles(bootState.profiles, bootState.profile);
        const curSpec = GPU_SPECS[bootState.gpu] || {};
        const splitPrev = resolveSplitPlanPreview(profs, latestSnap);
        btn.disabled = true;
        btn.setAttribute("aria-busy", "true");
        if (lbl) lbl.textContent = "正在提交部署...";
        let startedActive = false;
        try {
          const res = await apiCall("start_instance", {
            profile: profs[0],
            profiles: profs,
            gpu: bootState.gpu,
            high_mem: curSpec.hmLock ? false : Boolean(bootState.high_mem),
            minimax_r2v: Boolean(bootState.minimax_r2v),
            minimax_pe: Boolean(bootState.minimax_pe),
            llm_quant: String(bootState.llm_quant || "UD-Q4_K_XL"),
            llm_ctx: Number(bootState.llm_ctx || 32768),
            llm_mmproj: Boolean(bootState.llm_mmproj),
            llm27_quant: String(bootState.llm27_quant || "UD-Q4_K_XL"),
            llm27_ctx: Number(bootState.llm27_ctx || 65536),
            llm27_mmproj: Boolean(bootState.llm27_mmproj),
            cluster_mode: splitPrev.useSplit ? "split" : "cohost",
            recreate: false,
          });
          if (res && res.deploy && latestSnap) {
            startedActive = Boolean(res.deploy.active);
            forceShowBootStandby = false;
            latestSnap.deploy = res.deploy;
            renderSidebarStatus(latestSnap);
            renderOverviewTopology(latestSnap);
          }
          if (res && res.message) showToast(res.message, { isError: res.ok === false });
        } finally {
          btn.disabled = false;
          btn.removeAttribute("aria-busy");
          if (!startedActive) renderBootCockpit(latestSnap);
        }
      });
    }

    if (el("btn-ov-deploy-cancel")) {
      el("btn-ov-deploy-cancel").addEventListener("click", async () => {
        const btn = el("btn-ov-deploy-cancel");
        if (!deployCancelArm.isArmed()) {
          btn.classList.add("armed");
          btn.textContent = "再点确认中止";
          deployCancelArm.arm("cancel", () => {
            btn.classList.remove("armed");
            btn.textContent = "中止部署";
          });
          return;
        }
        deployCancelArm.disarm();
        btn.classList.remove("armed");
        btn.textContent = "中止部署";
        const res = await apiCall("cancel_deploy");
        if (res && res.message) showToast(res.message);
      });
    }

    // EasyColab 三栏左卡快捷操作、隐私打码与右侧运行日志流控制
    if (el("btn-privacy-mask")) {
      el("btn-privacy-mask").addEventListener("click", () => {
        privacyMasked = !privacyMasked;
        if (privacyMasked) tunnelMasked = true;
        applyPrivacyMask();
        if (latestSnap) renderSidebarControlCards(latestSnap);
        else _syncTunnelMaskUI("");
        showToast(privacyMasked ? "已对账号邮箱与隧道地址启用隐私打码" : "已恢复显示账号与隧道原文");
      });
    }

    if (el("btn-log-panel-toggle")) {
      el("btn-log-panel-toggle").addEventListener("click", () => {
        if (copilotState.open) {
          rightLogState.open = true;
          setCopilotOpen(false, { focusInput: false });
        } else {
          rightLogState.open = !rightLogState.open;
          syncRightPanelsLayout();
        }
        if (rightLogState.open && rightLogState.lines.length === 0) {
          loadRightPanelLogs();
        }
      });
    }

    if (el("btn-sb-card-refresh")) {
      el("btn-sb-card-refresh").addEventListener("click", async () => {
        const snap = await apiCall("refresh_now");
        if (snap) window.renderStudioState(snap);
        showToast("已刷新账号额度与实例状态");
      });
    }

    if (el("btn-sb-acct-menu")) {
      el("btn-sb-acct-menu").addEventListener("click", (e) => {
        e.stopPropagation();
        _setAcctPopoverOpen(!sbAcctMenuOpen);
      });
    }

    document.addEventListener("click", (e) => {
      if (!sbAcctMenuOpen) return;
      const pop = el("sb-acct-popover");
      const btn = el("btn-sb-acct-menu");
      if (pop && pop.contains(e.target)) return;
      if (btn && btn.contains(e.target)) return;
      _setAcctPopoverOpen(false);
    });

    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && sbAcctMenuOpen) {
        _setAcctPopoverOpen(false);
      }
    });

    if (el("sb-gpu-select")) {
      el("sb-gpu-select").addEventListener("change", (e) => {
        const gpuVal = String(e.target.value || "G4").toUpperCase();
        const gs = GPU_SPECS[gpuVal] || GPU_SPECS.L4;
        userTouchedBoot = true;
        bootState.gpu = gpuVal;
        if (gs.hmLock) {
          bootState.high_mem = false;
        } else if (bootState.profiles.some((p) => getWorkflowMeta(p).autoHmOnA100) && gpuVal === "A100") {
          bootState.high_mem = true;
        }
        apiCall("save_prefs", { boot_gpu: bootState.gpu, boot_high_mem: bootState.high_mem });
        renderBootCockpit(latestSnap);
      });
    }

    if (el("btn-sb-quick-deploy")) {
      el("btn-sb-quick-deploy").addEventListener("click", () => {
        const bootBtn = el("btn-ov-boot-start");
        if (bootBtn && !bootBtn.disabled) {
          bootBtn.click();
        }
      });
    }

    const _copyActiveTunnel = async () => {
      const s = latestSnap || {};
      const comfy = s.comfy || {};
      const slots = (s.cluster && s.cluster.slots) || {};
      const hasComfy = Boolean(comfy.tunnel_url || (slots.comfy && slots.comfy.tunnel_url));
      const hasLlm = Boolean(comfy.llm_tunnel_url || (slots.llm && slots.llm.tunnel_url));
      if (!hasComfy && !hasLlm) {
        showToast("当前暂无可用隧道，请先开机部署");
        return;
      }
      const res = hasComfy
        ? await apiCall("copy_comfy_access_config")
        : await apiCall("copy_llm_api_config");
      if (res && res.message) showToast(res.message, { isError: res.ok === false });
    };

    if (el("sb-tunnel-input")) {
      el("sb-tunnel-input").addEventListener("click", _copyActiveTunnel);
    }

    if (el("btn-sb-tunnel-eye")) {
      el("btn-sb-tunnel-eye").addEventListener("click", () => {
        tunnelMasked = !tunnelMasked;
        const inp = el("sb-tunnel-input");
        _syncTunnelMaskUI((inp && inp.dataset.rawUrl) || "");
      });
    }

    if (el("btn-sb-tunnel-refresh")) {
      el("btn-sb-tunnel-refresh").addEventListener("click", async () => {
        const snap = await apiCall("refresh_now");
        if (snap) window.renderStudioState(snap);
        await loadRightPanelLogs();
        showToast("已刷新实例连接与硬件遥测");
      });
    }

    if (el("btn-sb-tunnel-help")) {
      el("btn-sb-tunnel-help").addEventListener("click", () => {
        switchStudioTab("console");
        showToast("已打开控制台与环境深度体检");
      });
    }

    if (el("btn-sb-tele-mode")) {
      el("btn-sb-tele-mode").addEventListener("click", () => {
        telemetrySplitMode = !telemetrySplitMode;
        renderTelemetrySparkline(latestSnap);
      });
    }

    if (el("sb-idle-select")) {
      el("sb-idle-select").addEventListener("change", (e) => {
        const mins = Number(e.target.value || 0);
        const prefs = (latestSnap && latestSnap.prefs) || {};
        const patch = { idle_auto_stop_minutes: mins };
        if (mins > 0) patch.keep_alive = true;
        Object.assign(prefs, patch);
        apiCall("save_prefs", patch);
        renderTunerView(latestSnap);
        renderSidebarControlCards(latestSnap);
        showToast(mins > 0 ? `已设为闲置 ${mins} 分钟自动释放` : "已关闭闲置自动释放");
      });
    }

    if (el("btn-sb-release-all")) {
      el("btn-sb-release-all").addEventListener("click", async () => {
        const s = latestSnap || {};
        const clusterActive = Boolean(s.cluster && s.cluster.active);
        const hasLive = Boolean(s.is_running || clusterActive || (s.deploy && s.deploy.active));
        if (!hasLive) {
          showToast("当前无运行中节点，无需释放");
          return;
        }
        if (!sbReleaseArm.isArmed("all")) {
          sbReleaseArm.arm("all", () => renderSidebarControlCards(latestSnap));
          renderSidebarControlCards(latestSnap);
          return;
        }
        sbReleaseArm.disarm();
        const res = clusterActive
          ? await apiCall("stop_cluster_slot", "all")
          : await apiCall("stop_all_instances");
        if (res && res.snapshot) window.renderStudioState(res.snapshot);
        else renderSidebarControlCards(latestSnap);
        if (res && res.message) showToast(res.message, { isError: res.ok === false });
      });
    }

    if (!_sbUptimeTimer) {
      _sbUptimeTimer = setInterval(updateSidebarUptimeTicker, 1000);
    }

    if (el("rlog-source-seg")) {
      el("rlog-source-seg").addEventListener("click", (e) => {
        const btn = e.target.closest(".rlog-seg-btn[data-rlog-src]");
        if (!btn) return;
        const nextSrc = btn.dataset.rlogSrc || "all";
        if (nextSrc === rightLogState.source && rightLogState.lines.length > 0) return;
        rightLogState.source = nextSrc;
        loadRightPanelLogs();
      });
    }

    if (el("btn-rlog-refresh")) {
      el("btn-rlog-refresh").addEventListener("click", () => loadRightPanelLogs());
    }

    if (el("btn-rlog-clear")) {
      el("btn-rlog-clear").addEventListener("click", () => {
        rightLogState.lines = [];
        rightLogState.liveEvents = [];
        renderRightLogPanel();
      });
    }

    if (el("btn-rlog-copy")) {
      el("btn-rlog-copy").addEventListener("click", async () => {
        const term = el("rlog-terminal");
        const text = term
          ? Array.from(term.querySelectorAll(".rlog-line"))
              .map((row) => row.textContent.trim())
              .filter(Boolean)
              .join("\n")
          : "";
        if (!text) {
          showToast("当前无可复制的日志内容");
          return;
        }
        try {
          if (navigator.clipboard && navigator.clipboard.writeText) {
            await navigator.clipboard.writeText(text);
          } else {
            const ta = document.createElement("textarea");
            ta.value = text;
            ta.style.position = "fixed";
            ta.style.opacity = "0";
            document.body.appendChild(ta);
            ta.select();
            document.execCommand("copy");
            document.body.removeChild(ta);
          }
          showToast("已复制当前运行日志到剪贴板");
        } catch (_) {
          showToast("复制日志失败，请手动选中复制", { isError: true });
        }
      });
    }

    window.addEventListener("resize", syncRightPanelsLayout);

    // 拓扑舱动作
    if (el("btn-topo-llm-open")) {
      el("btn-topo-llm-open").addEventListener("click", async () => {
        const res = await apiCall("open_llm_tunnel");
        if (res && res.message) showToast(res.message, { isError: res.ok === false });
      });
    }
    if (el("btn-topo-llm-copy")) {
      el("btn-topo-llm-copy").addEventListener("click", async () => {
        const btn = el("btn-topo-llm-copy");
        const res = await apiCall("copy_llm_api_config");
        if (res && res.ok) {
          btn.setAttribute("data-copied", "true");
          btn.textContent = "已复制 API";
          setTimeout(() => {
            btn.removeAttribute("data-copied");
            btn.textContent = "复制 API";
          }, 2000);
        }
        if (res && res.message) showToast(res.message, { isError: res.ok === false });
      });
    }
    if (el("btn-topo-comfy-open")) {
      el("btn-topo-comfy-open").addEventListener("click", async () => {
        const res = await apiCall("open_comfy_tunnel");
        if (res && res.message) showToast(res.message, { isError: res.ok === false });
      });
    }
    if (el("btn-topo-comfy-copy")) {
      el("btn-topo-comfy-copy").addEventListener("click", async () => {
        const btn = el("btn-topo-comfy-copy");
        const res = await apiCall("copy_comfy_access_config");
        if (res && res.ok) {
          btn.setAttribute("data-copied", "true");
          btn.textContent = "已复制口令";
          setTimeout(() => {
            btn.removeAttribute("data-copied");
            btn.textContent = "复制口令";
          }, 2000);
        }
        if (res && res.message) showToast(res.message, { isError: res.ok === false });
      });
    }
    if (el("btn-topo-comfy-sync")) {
      el("btn-topo-comfy-sync").addEventListener("click", async () => {
        showToast("正在从云端拉回成图...");
        const res = await apiCall("sync_outputs");
        if (res && res.message) showToast(res.message, { isError: res.ok === false });
        loadGalleryItems();
      });
    }
    if (el("btn-ov-colab")) {
      el("btn-ov-colab").addEventListener("click", async () => {
        const res = await apiCall("open_colab_url");
        if (res && res.message) showToast(res.message);
      });
    }
    if (el("btn-ov-upload")) {
      el("btn-ov-upload").addEventListener("click", async () => {
        const res = await apiCall("pick_and_upload_files");
        if (res && !res.cancelled && res.message) showToast(res.message, { isError: res.ok === false });
      });
    }
    if (el("btn-stop-comfy-slot")) {
      el("btn-stop-comfy-slot").addEventListener("click", async () => {
        showToast("正在关停画布槽位（保留 LLM）...");
        const res = await apiCall("stop_cluster_slot", "comfy");
        if (res && res.snapshot) window.renderStudioState(res.snapshot);
        if (res && res.message) showToast(res.message, { isError: res.ok === false });
      });
    }
    if (el("btn-stop-llm-slot")) {
      el("btn-stop-llm-slot").addEventListener("click", async () => {
        showToast("正在关停 LLM 槽位（保留画布）...");
        const res = await apiCall("stop_cluster_slot", "llm");
        if (res && res.snapshot) window.renderStudioState(res.snapshot);
        if (res && res.message) showToast(res.message, { isError: res.ok === false });
      });
    }
    if (el("btn-stop-all-slot")) {
      el("btn-stop-all-slot").addEventListener("click", async () => {
        const btn = el("btn-stop-all-slot");
        if (!stopAllArm.isArmed()) {
          stopAllArm.arm("all", () => renderOverviewTopology(latestSnap));
          renderOverviewTopology(latestSnap);
          return;
        }
        stopAllArm.disarm();
        const clusterActive = Boolean(latestSnap && latestSnap.cluster && latestSnap.cluster.active);
        const res = clusterActive
          ? await apiCall("stop_cluster_slot", "all")
          : await apiCall("stop_all_instances");
        if (res && res.snapshot) window.renderStudioState(res.snapshot);
        if (res && res.message) showToast(res.message, { isError: res.ok === false });
        if (btn) btn.textContent = clusterActive ? "全停" : "一键停卡";
      });
    }

    // 号池工具
    if (el("btn-pool-refresh")) {
      el("btn-pool-refresh").addEventListener("click", async () => {
        const btn = el("btn-pool-refresh");
        btn.disabled = true;
        btn.textContent = "探测中...";
        try {
          const res = await apiCall("refresh_account_pool");
          if (res && res.snapshot) window.renderStudioState(res.snapshot);
          if (res && res.message) showToast(res.message, { isError: res.ok === false });
        } finally {
          btn.disabled = false;
          btn.textContent = "并发刷新全池";
        }
      });
    }
    if (el("btn-pool-auto")) {
      el("btn-pool-auto").addEventListener("click", () => {
        const curOn = !(latestSnap && latestSnap.prefs && latestSnap.prefs.auto_pool_schedule === false);
        const nextOn = !curOn;
        if (latestSnap && latestSnap.prefs) latestSnap.prefs.auto_pool_schedule = nextOn;
        apiCall("save_prefs", { auto_pool_schedule: nextOn });
        renderAccountPool(latestSnap);
        showToast(nextOn ? "已开启智能号池寻优调度" : "已关闭智能寻优");
      });
    }
    if (el("btn-pool-add")) {
      el("btn-pool-add").addEventListener("click", async () => {
        const ob = (latestSnap && latestSnap.onboarding) || {};
        if (ob.pending_oauth || ob.oauth_awaiting_code) {
          openModal("oauth");
          return;
        }
        const res = await apiCall("start_colab_oauth");
        if (res && res.snapshot) window.renderStudioState(res.snapshot);
        if (res && res.ok) openModal("oauth");
        else if (res && res.message) showToast(res.message, { isError: true });
      });
    }

    // 视图 2：画廊与检视栏
    if (el("gallery-filter-seg")) {
      el("gallery-filter-seg").addEventListener("click", (e) => {
        const btn = e.target.closest(".seg-item[data-kind]");
        if (!btn) return;
        loadGalleryItems(btn.dataset.kind || "all");
      });
    }
    if (el("gal-layout-seg")) {
      el("gal-layout-seg").addEventListener("click", (e) => {
        const btn = e.target.closest(".seg-item[data-layout]");
        if (!btn) return;
        const nextLayout = btn.dataset.layout === "grid" ? "grid" : "masonry";
        if (nextLayout === galleryState.layout) return;
        galleryState.layout = nextLayout;
        updateGalleryFilterControlsUI();
        renderGalleryGrid();
      });
    }
    if (el("inp-gal-search")) {
      el("inp-gal-search").addEventListener("input", (e) => {
        galleryState.query = e.target.value || "";
        recomputeAndRenderGallery({ resetPage: true });
      });
    }
    for (const [key, [, btnId]] of Object.entries(GAL_POPOVER_MAP)) {
      if (el(btnId)) {
        el(btnId).addEventListener("click", (e) => {
          e.stopPropagation();
          toggleGalleryPopover(key);
        });
      }
    }
    if (el("inp-gal-folder-search")) {
      el("inp-gal-folder-search").addEventListener("input", (e) => {
        galleryState.folderSearch = e.target.value || "";
        renderGalleryFolderOptions();
      });
    }
    if (el("gal-folder-list")) {
      el("gal-folder-list").addEventListener("click", (e) => {
        const btn = e.target.closest(".gal-pop-opt[data-folder]");
        if (!btn) return;
        galleryState.subfolder = btn.dataset.folder || "all";
        closeAllGalleryPopovers();
        recomputeAndRenderGallery({ resetPage: true });
      });
    }
    if (el("gal-date-list")) {
      el("gal-date-list").addEventListener("click", (e) => {
        const btn = e.target.closest(".gal-pop-opt[data-date]");
        if (!btn) return;
        galleryState.dateRange = btn.dataset.date || "all";
        closeAllGalleryPopovers();
        recomputeAndRenderGallery({ resetPage: true });
      });
    }
    if (el("gal-group-list")) {
      el("gal-group-list").addEventListener("click", (e) => {
        const btn = e.target.closest(".gal-pop-opt[data-group]");
        if (!btn) return;
        galleryState.groupBy = btn.dataset.group || "date";
        closeAllGalleryPopovers();
        recomputeAndRenderGallery({ resetPage: false });
      });
    }
    if (el("gal-sort-list")) {
      el("gal-sort-list").addEventListener("click", (e) => {
        const btn = e.target.closest(".gal-pop-opt[data-sort]");
        if (!btn) return;
        galleryState.sortBy = btn.dataset.sort || "newest";
        closeAllGalleryPopovers();
        recomputeAndRenderGallery({ resetPage: true });
      });
    }
    if (el("btn-gal-reset-filter")) {
      el("btn-gal-reset-filter").addEventListener("click", () => resetGalleryFilters());
    }
    if (el("btn-gal-load-more")) {
      el("btn-gal-load-more").addEventListener("click", () => {
        galleryState.visibleLimit += galleryState.pageSize;
        renderGalleryGrid();
      });
    }
    if (typeof ResizeObserver === "function" && el("gallery-stage")) {
      const galResizeObs = new ResizeObserver(() => {
        syncGalleryResponsiveCols();
      });
      galResizeObs.observe(el("gallery-stage"));
    }
    document.addEventListener("mousedown", (e) => {
      if (!galleryState.openPopover) return;
      if (e.target.closest("#gallery-filter-bar")) return;
      closeAllGalleryPopovers();
    });
    if (el("btn-gal-refresh")) el("btn-gal-refresh").addEventListener("click", () => loadGalleryItems());
    if (el("btn-gal-open-dir")) el("btn-gal-open-dir").addEventListener("click", () => apiCall("open_local_outputs_dir"));
    if (el("btn-gal-empty-dir")) el("btn-gal-empty-dir").addEventListener("click", () => apiCall("open_local_outputs_dir"));
    const syncAndReload = async () => {
      const res = await apiCall("sync_outputs");
      if (res && res.message) showToast(res.message, { isError: res.ok === false });
      await loadGalleryItems();
    };
    if (el("btn-gal-sync")) el("btn-gal-sync").addEventListener("click", syncAndReload);
    if (el("btn-gal-empty-sync")) el("btn-gal-empty-sync").addEventListener("click", syncAndReload);

    if (el("btn-inspector-close")) el("btn-inspector-close").addEventListener("click", closeGalleryInspector);
    if (el("btn-insp-copy-prompt")) {
      el("btn-insp-copy-prompt").addEventListener("click", async () => {
        const meta = galleryState.inspectData && galleryState.inspectData.meta;
        if (!meta) return;
        const parts = [];
        if (meta.positive_prompt) parts.push(meta.positive_prompt);
        if (meta.negative_prompt) parts.push(`Negative prompt: ${meta.negative_prompt}`);
        const text = parts.join("\n\n").trim();
        if (!text) return;
        let copied = false;
        if (text.length <= 500 && hasApi("copy_text")) {
          const r = await apiCall("copy_text", text);
          copied = Boolean(r && r.ok);
        }
        if (!copied && navigator.clipboard && navigator.clipboard.writeText) {
          try {
            await navigator.clipboard.writeText(text);
            copied = true;
          } catch (_) {
            copied = false;
          }
        }
        showToast(copied ? "已复制提示词到剪贴板" : "复制失败，请手动选中文本复制", { isError: !copied });
      });
    }
    if (el("btn-insp-push-canvas")) {
      el("btn-insp-push-canvas").addEventListener("click", async () => {
        const item = galleryState.inspectData && galleryState.inspectData.item;
        if (!item) return;
        showToast("正在提取并推送工作流到在线画布...");
        const res = await apiCall("push_gallery_workflow_to_canvas", item.rel_path);
        if (res && res.message) showToast(res.message, { isError: res.ok === false });
      });
    }
    if (el("btn-insp-reveal")) {
      el("btn-insp-reveal").addEventListener("click", async () => {
        const item = galleryState.inspectData && galleryState.inspectData.item;
        if (!item) return;
        const res = await apiCall("reveal_gallery_item", item.rel_path);
        if (res && res.ok === false && res.message) showToast(res.message, { isError: true });
      });
    }
    if (el("btn-gal-copilot")) {
      el("btn-gal-copilot").addEventListener("click", () => {
        const item = galleryState.inspectData && galleryState.inspectData.item;
        const relPath = (item && item.rel_path) || galleryState.selectedPath || "";
        if (!relPath) {
          showToast("请先选择需要改词重绘的作品");
          return;
        }
        const meta = (galleryState.inspectData && galleryState.inspectData.meta) || {};
        const rawPos = String(meta.positive_prompt || "").trim();
        const shortName = String(relPath).split("/").pop();
        attachCopilotContext({
          kind: "artwork",
          relPath,
          thumbUrl: (item && item.media_url) || "",
          name: shortName,
          sub: rawPos ? rawPos.slice(0, 40) : "画廊参考图",
        });
        const promptHint = rawPos
          ? `（当前正向提示词：${rawPos.slice(0, 72)}${rawPos.length > 72 ? "..." : ""}）`
          : "";
        const preset = `请基于画廊作品 "${relPath}" ${promptHint}帮我改词并重抽，我想调整为：`;
        window.openStudioCopilot(preset);
      });
    }

    // 视图 3：Qwen3.8 在线热调参 & 云盘 & 折叠配置
    if (el("tuner-ctx-seg")) {
      el("tuner-ctx-seg").addEventListener("click", (e) => {
        const btn = e.target.closest(".seg-item[data-ctx]");
        if (!btn) return;
        tunerState.touched = true;
        tunerState.ctx = Number(btn.dataset.ctx || 65536);
        renderTunerView(latestSnap);
      });
    }
    if (el("tuner-quant-seg")) {
      el("tuner-quant-seg").addEventListener("click", (e) => {
        const btn = e.target.closest(".seg-item[data-quant]");
        if (!btn) return;
        tunerState.touched = true;
        tunerState.quant = String(btn.dataset.quant || "UD-Q4_K_XL");
        renderTunerView(latestSnap);
      });
    }
    if (el("btn-tuner-mmproj")) {
      el("btn-tuner-mmproj").addEventListener("click", () => {
        tunerState.touched = true;
        tunerState.mmproj = !tunerState.mmproj;
        renderTunerView(latestSnap);
      });
    }
    if (el("btn-tuner-reload")) {
      el("btn-tuner-reload").addEventListener("click", async () => {
        const live = resolveLiveLlmCtx(latestSnap);
        const prefPatch = live.is27b
          ? { boot_llm27_ctx: tunerState.ctx, boot_llm27_quant: tunerState.quant, boot_llm27_mmproj: tunerState.mmproj }
          : { boot_llm_ctx: tunerState.ctx, boot_llm_quant: tunerState.quant, boot_llm_mmproj: tunerState.mmproj };
        apiCall("save_prefs", prefPatch);
        const res = await apiCall("reconfigure_running_llm", {
          ctx: tunerState.ctx,
          quant: tunerState.quant,
          mmproj: tunerState.mmproj,
        });
        if (res && res.deploy && latestSnap) {
          latestSnap.deploy = res.deploy;
          renderSidebarStatus(latestSnap);
          renderOverviewTopology(latestSnap);
        }
        if (res && res.message) showToast(res.message, { isError: res.ok === false });
      });
    }
    if (el("btn-tuner-copy-api")) {
      el("btn-tuner-copy-api").addEventListener("click", async () => {
        const res = await apiCall("copy_llm_api_config");
        if (res && res.message) showToast(res.message, { isError: res.ok === false });
      });
    }
    if (el("btn-tuner-open-llm")) {
      el("btn-tuner-open-llm").addEventListener("click", async () => {
        const res = await apiCall("open_llm_tunnel");
        if (res && res.message) showToast(res.message, { isError: res.ok === false });
      });
    }

    if (el("tuner-cloud-seg")) {
      el("tuner-cloud-seg").addEventListener("click", (e) => {
        const btn = e.target.closest(".seg-item[data-bucket]");
        if (!btn) return;
        tunerState.cloudBucket = btn.dataset.bucket || "loras";
        renderCloudAssetsList();
      });
    }
    if (el("btn-tuner-cloud-refresh")) {
      el("btn-tuner-cloud-refresh").addEventListener("click", () => loadCloudAssets(true));
    }
    if (el("btn-tuner-gdrive-bind")) {
      el("btn-tuner-gdrive-bind").addEventListener("click", async () => {
        const ob = (latestSnap && latestSnap.onboarding) || {};
        const res = ob.gdrive_binding ? await apiCall("cancel_gdrive_bind") : await apiCall("start_gdrive_bind");
        if (res && res.snapshot) window.renderStudioState(res.snapshot);
        if (res && res.message) showToast(res.message, { isError: res.ok === false });
      });
    }
    if (el("btn-tuner-gdrive-import")) {
      el("btn-tuner-gdrive-import").addEventListener("click", async () => {
        const res = await apiCall("pick_rclone_conf");
        if (!res || res.cancelled) return;
        if (res.snapshot) window.renderStudioState(res.snapshot);
        if (res.message) showToast(res.message, { isError: res.ok === false });
      });
    }
    if (el("btn-tuner-upload")) {
      el("btn-tuner-upload").addEventListener("click", async () => {
        const res = await apiCall("pick_and_upload_files");
        if (res && !res.cancelled && res.message) showToast(res.message, { isError: res.ok === false });
      });
    }

    // 折叠区：Cloudflare 固定域名
    if (el("btn-cf-save")) {
      el("btn-cf-save").addEventListener("click", async () => {
        const res = await apiCall("save_cf_tunnel_config", {
          tunnel_token: (el("inp-cf-token") && el("inp-cf-token").value) || "",
          comfy_url: (el("inp-cf-comfy") && el("inp-cf-comfy").value) || "",
          llm_url: (el("inp-cf-llm") && el("inp-cf-llm").value) || "",
          llm_api_key: (el("inp-cf-llmkey") && el("inp-cf-llmkey").value) || "",
        });
        if (res && res.snapshot) window.renderStudioState(res.snapshot);
        if (res && res.message) showToast(res.message, { isError: res.ok === false });
      });
    }
    if (el("btn-cf-clear")) {
      el("btn-cf-clear").addEventListener("click", async () => {
        const btn = el("btn-cf-clear");
        if (!cfClearArm.isArmed()) {
          btn.classList.add("armed");
          btn.textContent = "再点确认切回";
          cfClearArm.arm("cf", () => {
            btn.classList.remove("armed");
            btn.textContent = "切回免费隧道";
          });
          return;
        }
        cfClearArm.disarm();
        btn.classList.remove("armed");
        btn.textContent = "切回免费隧道";
        const res = await apiCall("clear_cf_tunnel_config");
        if (res && res.snapshot) window.renderStudioState(res.snapshot);
        if (res && res.message) showToast(res.message, { isError: res.ok === false });
      });
    }

    // 折叠区：闲置与同步偏好
    const bindPrefSeg = (segId, mapper) => {
      const node = el(segId);
      if (!node) return;
      node.addEventListener("click", (e) => {
        const btn = e.target.closest(".seg-item[data-val]");
        if (!btn) return;
        const patch = mapper(btn.dataset.val);
        if (latestSnap && latestSnap.prefs) Object.assign(latestSnap.prefs, patch);
        apiCall("save_prefs", patch);
        renderTunerView(latestSnap);
        showToast("已保存策略配置");
      });
    };
    bindPrefSeg("pref-keepalive-seg", (val) => ({
      keep_alive: val !== "off",
      idle_auto_stop_minutes: val === "stop60" ? 60 : 0,
    }));
    bindPrefSeg("pref-archive-seg", (val) => ({
      idle_archive_fail_policy: val === "keep_alive" ? "keep_alive" : "stop_billing",
    }));
    bindPrefSeg("pref-autopull-seg", (val) => ({
      auto_pull_outputs: val === "1",
    }));
    bindPrefSeg("pref-longrun-seg", (val) => ({
      long_run_warn_minutes: Number(val || 180),
    }));
    bindPrefSeg("pref-poll-seg", (val) => ({
      poll_interval_sec: Number(val || 60),
    }));

    if (el("btn-tools-reset")) {
      el("btn-tools-reset").addEventListener("click", async () => {
        const res = await apiCall("reset_tool_paths");
        if (res && res.snapshot) window.renderStudioState(res.snapshot);
        if (res && res.message) showToast(res.message);
      });
    }
    if (el("btn-tools-copy-colab")) {
      el("btn-tools-copy-colab").addEventListener("click", async () => {
        await apiCall("copy_text", "uv tool install google-colab-cli");
        showToast("已复制: uv tool install google-colab-cli");
      });
    }
    if (el("btn-tools-copy-rclone")) {
      el("btn-tools-copy-rclone").addEventListener("click", async () => {
        await apiCall("copy_text", "winget install Rclone.Rclone");
        showToast("已复制: winget install Rclone.Rclone");
      });
    }

    // 视图 4：控制台与体检
    if (el("console-source-seg")) {
      el("console-source-seg").addEventListener("click", (e) => {
        const btn = e.target.closest(".seg-item[data-source]");
        if (!btn) return;
        consoleState.source = btn.dataset.source || "local_diag";
        loadLiveLogs();
      });
    }
    if (el("console-filter")) {
      el("console-filter").addEventListener("input", (e) => {
        consoleState.filter = e.target.value || "";
        renderConsoleLogs();
      });
    }
    if (el("btn-console-refresh")) el("btn-console-refresh").addEventListener("click", () => loadLiveLogs());
    if (el("btn-console-doctor")) el("btn-console-doctor").addEventListener("click", () => runDoctorPreflight(true));
    if (el("btn-console-copilot")) {
      el("btn-console-copilot").addEventListener("click", () => {
        const srcMap = { local_diag: "本地诊断日志", llama: "远端 LLM 日志", comfy: "远端 ComfyUI 日志" };
        const srcName = srcMap[consoleState.source] || consoleState.source;
        const rawLines = Array.isArray(consoleState.lines) ? consoleState.lines : [];
        const errSnippet = rawLines
          .filter((ln) => /error|fail|traceback|exception|warn/i.test(String(ln || "")))
          .slice(-6)
          .join("\n");
        const prompt = errSnippet
          ? `请读取并诊断「${srcName}」(source=${consoleState.source}) 中的最新报错，定位根因并给出修复建议。近期异常摘录：\n\`\`\`\n${stripEmoji(errSnippet)}\n\`\`\``
          : `请读取并诊断「${srcName}」(source=${consoleState.source}) 的最新运行日志，检查是否存在异常报错或瓶颈并给出修复建议。`;
        window.openStudioCopilot("");
        sendCopilotMessage(prompt);
      });
    }
    if (el("btn-console-export")) {
      el("btn-console-export").addEventListener("click", async () => {
        const res = await apiCall("export_diagnostics");
        if (res && res.message) showToast(res.message, { isError: res.ok === false });
      });
    }
    if (el("btn-console-open-dir")) {
      el("btn-console-open-dir").addEventListener("click", async () => {
        const res = await apiCall("open_logs_dir");
        if (res && res.message) showToast(res.message, { isError: res.ok === false });
      });
    }
    if (el("btn-console-copy-logs")) {
      el("btn-console-copy-logs").addEventListener("click", async () => {
        const kw = String(consoleState.filter || "").trim().toLowerCase();
        const lines = (consoleState.lines || []).filter((ln) => !kw || String(ln).toLowerCase().includes(kw));
        if (lines.length === 0) return;
        const text = lines.join("\n");
        if (navigator.clipboard && navigator.clipboard.writeText) {
          try {
            await navigator.clipboard.writeText(text);
            showToast(`已复制 ${lines.length} 行日志`);
            return;
          } catch (_) {
            /* 回退剪贴板 */
          }
        }
        showToast("当前日志区已可直接框选复制");
      });
    }

    // 右侧 AI 副驾抽屉交互与三组向上弹出浮层
    if (el("btn-copilot-toggle")) {
      el("btn-copilot-toggle").addEventListener("click", () => setCopilotOpen(!copilotState.open));
    }
    if (el("btn-copilot-close")) {
      el("btn-copilot-close").addEventListener("click", () => setCopilotOpen(false));
    }
    if (el("btn-copilot-new-chat")) {
      el("btn-copilot-new-chat").addEventListener("click", async () => {
        await createNewCopilotSession();
      });
    }
    if (el("btn-copilot-history")) {
      el("btn-copilot-history").addEventListener("click", (e) => {
        e.stopPropagation();
        toggleCopilotHistoryPopover();
      });
    }
    if (el("btn-hist-new")) {
      el("btn-hist-new").addEventListener("click", async () => {
        await createNewCopilotSession();
      });
    }
    if (el("btn-hist-clear-all")) {
      el("btn-hist-clear-all").addEventListener("click", async () => {
        await clearAllCopilotSessions();
      });
    }
    if (el("inp-history-filter")) {
      el("inp-history-filter").addEventListener("input", (e) => {
        copilotState.historyFilter = e.target.value || "";
        renderCopilotHistoryPopover();
      });
    }
    if (el("copilot-messages")) {
      el("copilot-messages").addEventListener("scroll", () => {
        copilotState.stickToBottom = isCopilotNearBottom();
        updateCopilotScrollBtn();
      }, { passive: true });
    }
    if (el("btn-copilot-scroll-bottom")) {
      el("btn-copilot-scroll-bottom").addEventListener("click", () => {
        scrollCopilotBottom(true);
      });
    }
    if (el("btn-copilot-settings")) {
      el("btn-copilot-settings").addEventListener("click", () => {
        closeAllCopilotPopovers();
        openProviderModal({ providerId: copilotState.activeProviderId || null });
      });
    }
    if (el("btn-copilot-clear")) {
      el("btn-copilot-clear").addEventListener("click", async () => {
        if (copilotState.busy) {
          showToast("副驾正在执行中，请先停止生成后再清空会话");
          return;
        }
        const res = await apiCall("clear_agent_history");
        if (res && res.ok === false) {
          showToast(res.message || "清空会话失败", { isError: true });
          return;
        }
        if (res && res.state) {
          syncAgentStatePayload(res.state, { forceReplace: true });
        } else {
          copilotState.items = [];
          copilotState.activeStreamIdx = -1;
          copilotState.pendingConfirm = null;
          copilotState.stickToBottom = true;
          renderCopilotMessages();
          renderCopilotHeaderAndControls();
        }
        showToast("已清空 AI 副驾会话记录");
      });
    }
    if (el("copilot-brain-pill")) {
      el("copilot-brain-pill").addEventListener("click", (e) => {
        e.stopPropagation();
        toggleCopilotConfigPopover();
      });
    }
    if (el("btn-copilot-effort")) {
      el("btn-copilot-effort").addEventListener("click", (e) => {
        e.stopPropagation();
        toggleCopilotEffortPopover();
      });
    }
    if (el("btn-copilot-plus")) {
      el("btn-copilot-plus").addEventListener("click", (e) => {
        e.stopPropagation();
        toggleCopilotPlusPopover();
      });
    }
    if (el("btn-agent-cfg-close")) {
      el("btn-agent-cfg-close").addEventListener("click", () => toggleCopilotConfigPopover(false));
    }
    if (el("cfg-driver-seg")) {
      el("cfg-driver-seg").addEventListener("click", async (e) => {
        const btn = e.target.closest(".seg-item[data-val]");
        if (!btn || btn.disabled) return;
        const nextDrv = btn.dataset.val || "native";
        await applyAgentConfigPatch({ agent_driver: nextDrv });
      });
    }
    if (el("cfg-brain-seg")) {
      el("cfg-brain-seg").addEventListener("click", async (e) => {
        const btn = e.target.closest(".seg-item[data-val]");
        if (!btn) return;
        const nextBrain = btn.dataset.val || "auto";
        await applyAgentConfigPatch({ agent_brain_source: nextBrain });
      });
    }
    if (el("effort-options-list")) {
      el("effort-options-list").addEventListener("click", async (e) => {
        const btn = e.target.closest(".effort-option[data-effort]");
        if (!btn || btn.disabled) return;
        const eff = btn.dataset.effort || "auto";
        const ok = await applyAgentConfigPatch({ agent_reasoning_effort: eff });
        if (ok) {
          toggleCopilotEffortPopover(false);
        }
      });
    }
    if (el("inp-quick-model-filter")) {
      el("inp-quick-model-filter").addEventListener("input", (e) => {
        copilotState.quickFilter = e.target.value || "";
        renderQuickProviderModelsList();
      });
    }
    if (el("btn-open-provider-modal")) {
      el("btn-open-provider-modal").addEventListener("click", () => {
        openProviderModal({ providerId: copilotState.activeProviderId || null });
      });
    }
    if (el("btn-agent-auto-approve")) {
      el("btn-agent-auto-approve").addEventListener("click", async () => {
        const nextVal = !Boolean(copilotState.autoApproveBilling);
        await applyAgentConfigPatch(
          { agent_auto_approve_billing: nextVal },
          nextVal ? "已开启 Turbo 自动放行计费审批" : "已恢复安全审批确认"
        );
      });
    }
    if (el("btn-agent-clear-key")) {
      el("btn-agent-clear-key").addEventListener("click", async () => {
        const res = await apiCall("save_agent_config", { clear_api_key: true });
        if (res && res.ok === false) {
          showToast(res.message || "清除 API Key 失败", { isError: true });
          return;
        }
        const keyInp = el("inp-agent-api-key");
        if (keyInp) keyInp.value = "";
        if (res && typeof res === "object") {
          syncAgentStatePayload(res.state || res, { replaceItems: false });
        } else {
          await loadAgentState(false);
        }
        showToast("已从系统凭据库移除外部 API Key");
      });
    }

    // 供应商管理与 Magpie 模型导入模态弹窗交互绑定
    if (el("btn-provider-modal-close")) el("btn-provider-modal-close").addEventListener("click", closeProviderModal);
    if (el("btn-prov-cancel")) el("btn-prov-cancel").addEventListener("click", closeProviderModal);
    if (el("provider-modal")) {
      el("provider-modal").addEventListener("click", (e) => {
        if (e.target === el("provider-modal")) closeProviderModal();
      });
    }
    if (el("btn-prov-preset-magpie")) {
      el("btn-prov-preset-magpie").addEventListener("click", () => {
        _loadProviderIntoDraft(null, { presetMagpie: true });
        renderProviderModal();
        fetchProviderModelsAction({ silentOnErr: false });
      });
    }
    if (el("prov-format-seg")) {
      el("prov-format-seg").addEventListener("click", (e) => {
        const btn = e.target.closest(".seg-item[data-val]");
        if (!btn) return;
        providerModalState.apiFormat = btn.dataset.val || "openai";
        document.querySelectorAll("#prov-format-seg .seg-item").forEach((it) => {
          it.classList.toggle("active", it === btn);
        });
      });
    }
    if (el("inp-prov-name")) {
      el("inp-prov-name").addEventListener("input", (e) => {
        providerModalState.name = e.target.value || "";
      });
    }
    if (el("inp-prov-base-url")) {
      el("inp-prov-base-url").addEventListener("input", (e) => {
        providerModalState.baseUrl = e.target.value || "";
        const keyInp = el("inp-agent-api-key");
        if (keyInp && !providerModalState.hasStoredKey) {
          keyInp.placeholder = _isLoopbackUrlStr(providerModalState.baseUrl)
            ? "本机网关（如 127.0.0.1:3425）免鉴权可留空"
            : "粘贴 API 密钥，加密存入系统凭据库";
        }
      });
    }
    if (el("inp-prov-probe-model")) {
      el("inp-prov-probe-model").addEventListener("input", (e) => {
        providerModalState.probeModel = e.target.value || "";
      });
    }
    if (el("btn-prov-toggle-key")) {
      el("btn-prov-toggle-key").addEventListener("click", () => {
        providerModalState.showKey = !providerModalState.showKey;
        const keyInp = el("inp-agent-api-key");
        if (keyInp) keyInp.type = providerModalState.showKey ? "text" : "password";
        el("btn-prov-toggle-key").textContent = providerModalState.showKey ? "隐藏" : "显示";
      });
    }
    if (el("inp-prov-models")) {
      el("inp-prov-models").addEventListener("input", (e) => {
        providerModalState.modelsText = e.target.value || "";
        providerModalState.selectedModels = _parseModelListText(providerModalState.modelsText);
        if (!providerModalState.selectedModels.includes(providerModalState.defaultModel)) {
          providerModalState.defaultModel = providerModalState.selectedModels[0] || "";
        }
        renderProviderModelsGrid();
      });
    }
    if (el("inp-prov-model-filter")) {
      el("inp-prov-model-filter").addEventListener("input", (e) => {
        providerModalState.modelFilter = e.target.value || "";
        renderProviderModelsGrid();
      });
    }
    if (el("btn-prov-select-all")) {
      el("btn-prov-select-all").addEventListener("click", () => {
        const kw = String(providerModalState.modelFilter || "").trim().toLowerCase();
        const pool = Array.from(new Set([...(providerModalState.selectedModels || []), ...(providerModalState.fetchedModels || [])]));
        const toAdd = kw ? pool.filter((m) => m.toLowerCase().includes(kw)) : pool;
        for (const m of toAdd) {
          if (!providerModalState.selectedModels.includes(m)) {
            providerModalState.selectedModels.push(m);
          }
        }
        if (!providerModalState.defaultModel && providerModalState.selectedModels.length > 0) {
          providerModalState.defaultModel = providerModalState.selectedModels[0];
        }
        providerModalState.modelsText = providerModalState.selectedModels.join(", ");
        renderProviderModal();
      });
    }
    if (el("btn-prov-select-none")) {
      el("btn-prov-select-none").addEventListener("click", () => {
        providerModalState.selectedModels = [];
        providerModalState.defaultModel = "";
        providerModalState.modelsText = "";
        renderProviderModal();
      });
    }
    if (el("btn-prov-fetch-models")) {
      el("btn-prov-fetch-models").addEventListener("click", () => fetchProviderModelsAction({ silentOnErr: false }));
    }
    if (el("btn-prov-detect-proto")) {
      el("btn-prov-detect-proto").addEventListener("click", async () => {
        const baseUrl = String((el("inp-prov-base-url") && el("inp-prov-base-url").value) || "").trim();
        providerModalState.baseUrl = baseUrl;
        if (!baseUrl) {
          _setProviderStatus("请先填写网关 Base URL", "err");
          return;
        }
        const keyInp = el("inp-agent-api-key");
        const typedKey = keyInp ? String(keyInp.value || "").trim() : "";
        const hintModel = String((el("inp-prov-probe-model") && el("inp-prov-probe-model").value) || "").trim()
          || providerModalState.defaultModel
          || providerModalState.selectedModels[0]
          || "";
        providerModalState.busyDetect = true;
        renderProviderModal();
        try {
          const res = await apiCall("probe_agent_provider", {
            mode: "detect_protocol",
            provider_id: providerModalState.isNew ? "" : providerModalState.editingId,
            base_url: baseUrl,
            api_key: typedKey,
            probe_model: hintModel,
          });
          if (res && res.ok && res.api_format) {
            providerModalState.apiFormat = res.api_format;
            _setProviderStatus(res.message || `已识别协议：${res.api_format}`, "ok");
          } else {
            _setProviderStatus((res && res.message) || "协议检测未命中", "err");
          }
        } finally {
          providerModalState.busyDetect = false;
          renderProviderModal();
        }
      });
    }
    if (el("btn-prov-test-models")) {
      el("btn-prov-test-models").addEventListener("click", async () => {
        const baseUrl = String((el("inp-prov-base-url") && el("inp-prov-base-url").value) || "").trim();
        providerModalState.baseUrl = baseUrl;
        if (!baseUrl) {
          _setProviderStatus("请先填写网关 Base URL", "err");
          return;
        }
        const probeSingle = String((el("inp-prov-probe-model") && el("inp-prov-probe-model").value) || "").trim();
        let targets = probeSingle
          ? [probeSingle]
          : providerModalState.selectedModels.slice(0, 16);
        if (targets.length === 0 && providerModalState.fetchedModels.length > 0) {
          targets = providerModalState.fetchedModels.slice(0, 8);
        }
        if (targets.length === 0) {
          _setProviderStatus("请先勾选模型或在试用框填入模型 ID", "err");
          return;
        }
        for (const m of targets) {
          providerModalState.testResults[m] = { testing: true };
        }
        providerModalState.busyTest = true;
        renderProviderModal();
        const keyInp = el("inp-agent-api-key");
        const typedKey = keyInp ? String(keyInp.value || "").trim() : "";
        try {
          const res = await apiCall("probe_agent_provider", {
            mode: "test_models",
            provider_id: providerModalState.isNew ? "" : providerModalState.editingId,
            base_url: baseUrl,
            api_format: providerModalState.apiFormat,
            models: targets,
            api_key: typedKey,
          });
          for (const item of (res && Array.isArray(res.results) ? res.results : [])) {
            providerModalState.testResults[item.model] = {
              ok: Boolean(item.ok),
              latency_ms: Number(item.latency_ms || 0),
              error: String(item.error || ""),
            };
          }
          _setProviderStatus((res && res.message) || "模型检测完成", (res && res.ok) ? "ok" : "err");
        } finally {
          providerModalState.busyTest = false;
          renderProviderModal();
        }
      });
    }
    if (el("btn-prov-delete")) {
      el("btn-prov-delete").addEventListener("click", async () => {
        const pid = providerModalState.editingId;
        if (providerModalState.isNew || !pid) return;
        if (!provDelArm.isArmed(pid)) {
          provDelArm.arm(pid, () => renderProviderModal());
          renderProviderModal();
          return;
        }
        provDelArm.disarm();
        const res = await apiCall("save_agent_config", { delete_provider_id: pid });
        if (res && res.ok === false) {
          _setProviderStatus(res.message || "删除供应商失败", "err");
          return;
        }
        if (res && typeof res === "object") {
          syncAgentStatePayload(res.state || res, { replaceItems: false });
        }
        _loadProviderIntoDraft(copilotState.activeProviderId || null, {
          isNew: (copilotState.providers || []).length === 0,
        });
        renderProviderModal();
        showToast((res && res.message) || "已删除供应商");
      });
    }
    if (el("btn-prov-save")) {
      el("btn-prov-save").addEventListener("click", async () => {
        const baseUrl = String((el("inp-prov-base-url") && el("inp-prov-base-url").value) || "").trim();
        if (!baseUrl) {
          _setProviderStatus("请填写网关 Base URL", "err");
          return;
        }
        const name = String((el("inp-prov-name") && el("inp-prov-name").value) || "").trim();
        const modelsFromInput = _parseModelListText((el("inp-prov-models") && el("inp-prov-models").value) || "");
        const finalModels = Array.from(new Set([...(providerModalState.selectedModels || []), ...modelsFromInput]));
        const defModel = (providerModalState.defaultModel && finalModels.includes(providerModalState.defaultModel))
          ? providerModalState.defaultModel
          : (finalModels[0] || "");
        const keyInp = el("inp-agent-api-key");
        const typedKey = keyInp ? String(keyInp.value || "").trim() : "";

        const upsert = {
          id: providerModalState.isNew ? "" : providerModalState.editingId,
          name,
          api_format: providerModalState.apiFormat,
          base_url: baseUrl,
          models: finalModels,
          default_model: defModel,
          model_capabilities: providerModalState.fetchedCapabilities || {},
          activate: true,
        };
        if (typedKey) {
          upsert.api_key = typedKey;
        }

        const res = await apiCall("save_agent_config", { upsert_provider: upsert });
        if (res && res.ok === false) {
          _setProviderStatus(res.message || "保存供应商失败", "err");
          showToast(res.message || "保存供应商失败", { isError: true });
          return;
        }
        if (keyInp) keyInp.value = "";
        if (res && typeof res === "object") {
          syncAgentStatePayload(res.state || res, { replaceItems: false });
        } else {
          await loadAgentState(false);
        }
        closeProviderModal();
        showToast((res && res.message) || "已保存并启用模型供应商");
      });
    }
    if (el("copilot-drawer")) {
      el("copilot-drawer").addEventListener("mousedown", (e) => {
        if (
          !copilotState.popoverOpen
          && !copilotState.effortPopoverOpen
          && !copilotState.plusPopoverOpen
          && !copilotState.historyPopoverOpen
        ) {
          return;
        }
        if (
          e.target.closest(
            "#copilot-config-popover, #copilot-brain-pill, #copilot-effort-popover, #btn-copilot-effort, #copilot-plus-popover, #btn-copilot-plus, #copilot-history-popover, #btn-copilot-history"
          )
        ) {
          return;
        }
        closeAllCopilotPopovers();
      });
    }
    if (el("copilot-input")) {
      const inp = el("copilot-input");
      inp.addEventListener("input", resizeCopilotInput);
      inp.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
          e.preventDefault();
          if (e.isComposing || e.keyCode === 229) return;
          sendCopilotMessage();
        }
      });
    }
    if (el("btn-copilot-send")) {
      el("btn-copilot-send").addEventListener("click", async () => {
        if (copilotState.busy) {
          const res = await apiCall("cancel_agent_turn");
          copilotState.busy = false;
          finalizeCopilotStreamItem();
          renderCopilotHeaderAndControls();
          showToast((res && res.message) || "已停止副驾当前生成");
          return;
        }
        sendCopilotMessage();
      });
    }

    // 模态框与全局 Ctrl+K / Esc 快捷键
    if (el("btn-modal-close")) el("btn-modal-close").addEventListener("click", closeModal);
    if (el("studio-modal")) {
      el("studio-modal").addEventListener("click", (e) => {
        if (e.target === el("studio-modal")) closeModal();
      });
    }

    window.addEventListener("keydown", (e) => {
      if ((e.ctrlKey || e.metaKey) && !e.shiftKey && !e.altKey && String(e.key || "").toLowerCase() === "k") {
        e.preventDefault();
        setCopilotOpen(!copilotState.open);
        return;
      }
      if (e.key === "Escape") {
        if (providerModalState.open) {
          e.preventDefault();
          closeProviderModal();
          return;
        }
        if (copilotState.popoverOpen || copilotState.effortPopoverOpen || copilotState.plusPopoverOpen || copilotState.historyPopoverOpen) {
          e.preventDefault();
          closeAllCopilotPopovers();
          return;
        }
        const modal = el("studio-modal");
        if (modal && !modal.classList.contains("hidden")) {
          e.preventDefault();
          closeModal();
          return;
        }
        if (galleryState.openPopover) {
          e.preventDefault();
          closeAllGalleryPopovers();
          return;
        }
        const insp = el("gallery-inspector");
        if (insp && insp.classList.contains("open")) {
          e.preventDefault();
          closeGalleryInspector();
        }
      }
    });

    // 全局文件拖放视觉反馈（不阻断冒泡，保留 pywebview 原生全路径读取）
    let dragTimer = null;
    const clearDrag = () => {
      clearTimeout(dragTimer);
      document.body.classList.remove("drag-over");
    };
    const onDrag = (e) => {
      e.preventDefault();
      if (e.dataTransfer) e.dataTransfer.dropEffect = "copy";
      document.body.classList.add("drag-over");
      clearTimeout(dragTimer);
      dragTimer = setTimeout(clearDrag, 180);
    };
    window.addEventListener("dragenter", onDrag);
    window.addEventListener("dragover", onDrag);
    window.addEventListener("dragleave", () => {
      clearTimeout(dragTimer);
      dragTimer = setTimeout(clearDrag, 120);
    });
    window.addEventListener("drop", (e) => {
      e.preventDefault();
      clearDrag();
    });
  }

  /* ----------------------------------------------------------------
     启动入口与初始对齐
     ---------------------------------------------------------------- */
  setupWindowChrome();
  bindInteractions();
  setStudioTheme(getEffectiveScheme());
  applyPrivacyMask();
  syncRightPanelsLayout();
  renderDoctorChecklist(null);

  // 支持通过 URL ?tab=... 直达指定工作区或展开 AI 副驾抽屉
  try {
    const params = new URLSearchParams(window.location.search || "");
    const initTab = params.get("tab");
    if (initTab === "copilot") {
      window.openStudioCopilot("");
    } else if (initTab && VIEW_META[initTab]) {
      applyViewSwitch(initTab);
    }
  } catch (_) {
    /* 忽略 URL 参数解析异常 */
  }
  renderCopilotHeaderAndControls();
  renderCopilotChips();
  renderCopilotMessages();

  window.addEventListener("pywebviewready", async () => {
    const [snap, galRes, winRes, agentRes] = await Promise.all([
      apiCall("get_state"),
      apiCall("list_gallery_items", { kind: galleryState.kind, limit: 1000 }),
      apiCall("studio_window_ctl", "get_window_state"),
      apiCall("get_agent_state"),
      loadRightPanelLogs(),
    ]);
    if (snap) window.renderStudioState(snap);
    if (agentRes && typeof agentRes === "object") {
      syncAgentStatePayload(agentRes, { replaceItems: true });
    }
    if (galRes && (Array.isArray(galRes.items) || Array.isArray(galRes))) {
      applyGalleryListResponse(galRes);
    }
    if (winRes && typeof winRes.maximized === "boolean") {
      document.body.classList.toggle("is-maximized", winRes.maximized);
      const shell = el("studio-shell");
      if (shell) shell.classList.toggle("is-maximized", winRes.maximized);
    }
  });
})();
