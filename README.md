# Colab-Desktop-Pet（Colab 桌面宠物与 Studio 双窗算力控制台）

面向 Windows 桌面的 Google Colab 计算单元（CU）监控桌宠、Studio 双窗工作台与一键 GPU 工作流控制台，基于 `pywebview` + Edge WebView2 + Win32 `SetWindowRgn` 像素级不规则裁剪悬浮窗构建。完整运维教程、踩坑排障与模块详解见 [Colab迁移与使用指南.md](Colab迁移与使用指南.md)。

> 本仓库源代码遵循 [LICENSE](LICENSE)（MIT 开源协议）。按 [colab_pet/ASSETS_NOTICE.txt](colab_pet/ASSETS_NOTICE.txt) 声明，仓库随附重绘的四表情 Q 版「夏儿」桌宠 `.webp` 立绘（仅限非商业个人展示与测试），且**不包含**任何个人账号凭据、邮箱、自定义域名、私训 LoRA 清单或本地私有路径。

---

## 核心特性

### 1. 桌面悬浮挂件 & Colab Studio 双窗工作台（[启动Colab桌宠.pyw](启动Colab桌宠.pyw) & [colab_pet/](colab_pet/__init__.py)）
- **像素级无边框透明浮窗与零 Pillow 裁剪**：使用预提取的 `alpha_masks.json` + Win32 `SetWindowRgn` 精确贴合气泡与 `.webp` 立绘轮廓，透明区穿透鼠标，支持单实例互斥热替换、自动贴边吸附与跨屏高帧率拖拽；运行时零 `Pillow` / `aiohttp` / `pywin32` 依赖。
- **五页桌宠气泡面板（[colab_pet/ui.html](colab_pet/ui.html)）**：集成 **积分看板 / 今日小账本 / 快捷控制台 / 偏好设置 / 账号与环境**，支持多生态/双栖开机、三级极速隧道、文件拖拽投递与防烧卡闲置自动停卡。
- **独立无边框 Studio 工作台（[colab_pet/studio.html](colab_pet/studio.html)）**：一键展开大窗，内置 **算力与集群总览**、**作品瀑布流画廊**（Range 206 流式音视频播放 + ComfyUI PNG 元数据工作流直推画布）、**引擎热调参与云盘浏览**、**三端脱敏控制台与体检**，支持浅/深色主题无闪烁切换。
- **多账号算力池与异构双开（Split Cluster）**：系统凭据库（`keyring`）与账号账本按账号物理隔离，支持并发余额/层级探针、稀缺 GPU 容灾换号，以及在两个账号间并发运行 `LLM 对话 + ComfyUI 生图` 双槽位与分槽定向停卡。
- **内置 AI 副驾（Copilot）与分层安全护栏**：`Ctrl+K` 呼出副驾抽屉，支持内置轻量 `NativeLightDriver`（OpenAI Chat / Responses / Anthropic / Gemini 四协议 + 思考档位映射）与外部 `ClaudeCodeDriver`（`colab_ctl.py mcp-stdio` 10 工具），遵循「白名单只读直通 / 高危硬拦截 / 启停算力与非只读远端代码强制弹窗人工审批」（桌面 GUI 彻底忽略 `COLAB_AGENT_AUTO_APPROVE_EXEC`）。

### 2. CLI 总控、5 套生态与共享内核（[colab_ctl.py](colab_ctl.py) & [colab_core/](colab_core/__init__.py)）
CLI 与桌面端共用同一套正交共享内核（13 顶层模块 + `agent/` 7 文件子包，由 [stack_manifest.json](stack_manifest.json) 与 [colab_core/catalog.py](colab_core/catalog.py) 提供卡型/费率单一事实来源，[colab_core/runtime.py](colab_core/runtime.py) 与 [colab_core/agent_tools.py](colab_core/agent_tools.py) 保留向后兼容重导出）：
1. **`anima`（默认 `L4`）**：[colab_bootstrap_anima.py](colab_bootstrap_anima.py) + [verify_smoke.py](verify_smoke.py) — ComfyUI + Anima 底座（公开模式无需私训 LoRA 或 `[gdrive_save]` 云盘即可开箱冒烟）。
2. **`minimax`（默认 `G4`）**：[colab_bootstrap_minimax_h3.py](colab_bootstrap_minimax_h3.py) + [verify_minimax_h3.py](verify_minimax_h3.py) + [workflows/minimax_h3/](workflows/minimax_h3/) — MiniMax-H3 生图 / 图像编辑 / 视频生声全套工作流。
3. **`drama3`（默认 `G4`）**：[colab_bootstrap_drama3.py](colab_bootstrap_drama3.py) + [verify_drama3.py](verify_drama3.py) + [workflows/drama3/](workflows/drama3/) — Fish Audio Drama 3 语音演戏与克隆。
4. **`qwen_27b`（默认 `L4`，兼容 `T4`）**：[colab_bootstrap_qwen_llm.py](colab_bootstrap_qwen_llm.py) + [verify_qwen_llm.py](verify_qwen_llm.py) — `llama.cpp` + SSE 网关（Qwen3.8-27B，适合 `16–24 GiB` 显存或双栖 co-host）。
5. **`qwen_llm`（默认 `G4`）**：[colab_bootstrap_qwen_llm.py](colab_bootstrap_qwen_llm.py) + [verify_qwen_llm.py](verify_qwen_llm.py) — `llama.cpp` + Qwen3.8-180B MoE Bearer 鉴权服务。

---

## 目录结构

```text
Colab-Desktop-Pet/
├── 启动Colab桌宠.pyw                # Windows 无控制台黑框桌宠启动入口
├── colab_ctl.py                     # 本机总控 CLI（up/status/tunnel/smoke/sync/run/exec/mcp-stdio/down/doctor/diag）
├── colab_core/                      # 共享内核：catalog/accounts/colab_client/file_sync/health/pool_scheduler/
│   │                                #   exec_runner/deploy_state/ledger/reasoning_specs + agent/(7 模块) + 兼容门面
├── colab_pet/                       # 桌面宿主与正交后端：
│   ├── pet_app.py / bridge_api.py / asset_server.py / win32_window.py
│   ├── state_store.py / secret_store.py / context.py / colab_backend.py
│   ├── backend_{accounts,agent,cluster,comfy,deploy,monitor,studio,tunnel}.py
│   ├── ui.html / studio.html / studio.css / studio.js
│   ├── ASSETS_NOTICE.txt            # 立绘素材声明
│   └── assets/                      # 四表情 .webp + pet.ico + alpha_masks.json
├── colab_security.py                # 系统凭据保险箱、密钥脱敏与来源站/票据安全校验
├── colab_cli_bridge.py              # google-colab-cli 虚拟环境统一桥接助手
├── colab_remote_common.py           # 云端部署、双向鉴权核验与共享基础库
├── colab_pet_bridge_node.py / colab_pet_direct_canvas.js  # ComfyUI 直铺画布 + 公网鉴权网关
├── colab_sse_gateway.py             # 零依赖异步 SSE 流式鉴权网关
├── colab_bootstrap_*.py / verify_*.py / workflows/
├── stack_manifest.json              # schema v2 单一事实源清单（GPU 规格/默认费率 + 5 套生态）
├── build_windows_package.py / colab_pet.spec              # 便携版与 PyInstaller 打包脚本
├── ruff.toml / pytest.ini / .github/workflows/ci.yml      # Ruff (C901<=25) 与 Windows CI 门禁
├── tests/                           # 含 conftest.py 全局沙箱隔离与 200+ 离线/GUI 回归测试
├── .cf_tunnel_secret.example.json   # Cloudflare 命名隧道可选配置模板
└── requirements.txt / requirements-lock.txt / requirements-dev.txt
```

---

## 快速开始

### 1. 环境准备（Windows 10 / 11，Python 3.11+）
1. 确保系统已安装 Microsoft Edge WebView2 Runtime（Win11 默认自带）。
2. 安装并登录 Google 官方 Colab CLI（亦可启动桌宠后在界面内一键 OAuth2 登录）：
   ```powershell
   uv tool install google-colab-cli
   colab whoami
   ```
3. 安装运行依赖：
   ```powershell
   pip install -r requirements-lock.txt
   ```

### 2. 启动桌宠与常见 CLI 用法
```powershell
# 启动悬浮桌宠（支持点击快捷按钮展开 Colab Studio 工作台）
pythonw 启动Colab桌宠.pyw

# 公开基础版环境自检（无需绑定云盘或私训 LoRA）
python colab_ctl.py doctor --profile anima --gpu L4

# 一键部署 / 查看或复用公网隧道 / 端到端冒烟 / 停卡归零
python colab_ctl.py up --profile anima --gpu L4
python colab_ctl.py tunnel
python colab_ctl.py smoke --profile anima
python colab_ctl.py down --all
```

### 3. 质量门禁、测试与 Windows 打包
```powershell
# 安装开发与测试依赖（pytest / aiohttp / pillow / ruff）
pip install -r requirements-dev.txt

# 静态质量门禁（C901 <= 25, S110=0）
python -m ruff check --config ruff.toml .

# 离线全套非 GUI 回归测试
python -m pytest -q -c pytest.ini -m "not gui" tests

# Windows 桌面真机 PyWebView GUI 视觉回归
python -m pytest -v -c pytest.ini -m gui tests

# 构建零额外依赖原生 EXE 便携包 / PyInstaller 免 Python 独立目录包
python build_windows_package.py --mode portable
python build_windows_package.py --mode pyinstaller
```

---

## 可选配置与环境变量

| 变量名 / 配置文件 | 作用 | 默认值 / 说明 |
|---|---|---|
| `.cf_tunnel_secret.json` | 绑定个人 Cloudflare Named Tunnel 固定域名与 `vault:` 密钥引用 | 复制 [.cf_tunnel_secret.example.json](.cf_tunnel_secret.example.json) 模板配置，严禁明文入库 |
| `COLAB_PET_EDITION` | 切换预检口径（`public` / `personal`） | `colab_ctl.py doctor` 默认 `public` |
| `REQUIRED_LORAS_JSON` | 传入 `{文件名: 精确字节数}` JSON 启用云端 LoRA 严格校验 | 默认 `{}`（仅同步可用 `*.safetensors`） |
| `LORA_CHAR` / `LORA_STYLE` / `SMOKE_PROMPT` | 为 `verify_smoke.py` 指定可选测试 LoRA 与提示词 | 默认纯 `anima-base-v1.0.safetensors` 底座冒烟 |
| `COLAB_LORA_DIR` / `COLAB_DRAMA3_REFS_DIR` / `COLAB_DOWNLOAD_DIR` | 指定本地 LoRA、参考音频与结果输出目录 | 默认项目下 `loras/`、`drama3_refs/`、`_downloads/` |
| `COLAB_RCLONE_REMOTE` | 自定义 `rclone` 远端名称 | 默认 `gdrive_save` |
