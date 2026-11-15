# Colab-Desktop-Pet（Colab 桌面宠物与一键算力控制台）

面向 Windows 桌面的 Google Colab 计算单元（CU）监控桌宠与一键 GPU 工作流控制台，基于 `pywebview` + Edge WebView2 + Win32 `SetWindowRgn` 像素级不规则裁剪悬浮窗构建。

> 本仓库代码遵循 [LICENSE](LICENSE)（MIT 开源协议）。按 [colab_pet/ASSETS_NOTICE.txt](colab_pet/ASSETS_NOTICE.txt) 声明，公开仓库**不包含**第三方人物立绘 PNG 素材，也**不包含**任何个人账号凭据、邮箱、自定义域名或本地私有路径。

---

## 核心功能

### 1. 桌面悬浮宠物挂件（[启动Colab桌宠.pyw](启动Colab桌宠.pyw) & [colab_pet/](colab_pet/)）
- **像素级无边框透明浮窗**：使用 Win32 `SetWindowRgn` 贴合立绘与气泡边缘裁剪，鼠标穿透透明区域，支持单实例互斥与跨屏高帧率拖拽。
- **实时算力监控与防烧卡保护**：
  - 轮询 `google-colab-cli` 获取当前 CU 余额、实时消耗速率（`CU/hr`）、可用 GPU 池状态与活跃/孤儿会话；
  - 本地记录今日小账本（今日实耗 CU、连续开机计时），仅保存在本地忽略文件 `colab_pet/pet_state.json`，不持久化账号令牌；
  - 定期发送 Colab 三层轻保活心跳，并支持空闲超时自动停卡与算力告警。
- **四合一桌宠面板**：集成 **积分看板 / 今日小账本 / 快捷控制台 / 偏好设置**，支持一键开机、打开 WebUI 隧道、工作流/素材/LoRA 拖拽投递、输出自动拉回与内置 LLM 对话气泡。
- **内置矢量兜底立绘**：未在 [colab_pet/assets/](colab_pet/assets/README.md) 放入自定义 PNG 时，自动生成 `500 × 418` RGBA 原创矢量角色保证开箱可运行；放入 `idle.png` / `blink.png` / `proud.png` / `sad.png` 后自动切换为自定义立绘。

### 2. CLI 总控与四套云端一键自举（[colab_ctl.py](colab_ctl.py)）
支持命令行与桌宠界面双重触发，内置 4 套幂等部署 Profile：
1. **`anima`（默认 `L4`）**：[colab_bootstrap_anima.py](colab_bootstrap_anima.py) + [verify_smoke.py](verify_smoke.py) — ComfyUI + Anima 底座 + Cloudflared 公网隧道。
2. **`minimax`（默认 `G4`）**：[colab_bootstrap_minimax_h3.py](colab_bootstrap_minimax_h3.py) + [verify_minimax_h3.py](verify_minimax_h3.py) + [workflows/minimax_h3/](workflows/minimax_h3/) — MiniMax-H3 生图 / 图像编辑 / 视频生声全套模型与工作流。
3. **`drama3`（默认 `G4`）**：[colab_bootstrap_drama3.py](colab_bootstrap_drama3.py) + [verify_drama3.py](verify_drama3.py) + [workflows/drama3/](workflows/drama3/) — Fish Audio Drama 3 语音演戏与克隆（含 Qwen3-ASR 辅助）。
4. **`qwen_llm`（默认 `G4`）**：[colab_bootstrap_qwen_llm.py](colab_bootstrap_qwen_llm.py) + [verify_qwen_llm.py](verify_qwen_llm.py) — `llama.cpp`（Qwen3.8-Flash-Next）+ SSE 网关 + Bearer Key 鉴权推理服务。

---

## 目录结构

```text
Colab-Desktop-Pet/
├── 启动Colab桌宠.pyw                # Windows 无控制台黑框桌宠启动入口
├── colab_ctl.py                     # 本机总控 CLI（up/status/tunnel/smoke/sync/run/down）
├── colab_bootstrap_*.py             # 四套云端 Colab 实例幂等自举脚本
├── verify_*.py                      # 四套端到端冒烟验证脚本
├── workflows/                       # 预置 ComfyUI 工作流 JSON（drama3 / minimax_h3）
├── .cf_tunnel_secret.example.json   # Cloudflare 命名隧道可选配置模板
├── requirements.txt                 # 桌宠 Python 依赖
└── colab_pet/
    ├── pet_app.py                   # pywebview + Edge WebView2 + Win32 区域裁剪窗体
    ├── colab_backend.py             # 状态采集、账本统计、保活、隧道与任务调度后端
    ├── ui.html                      # 桌宠气泡与控制台前端界面
    ├── ASSETS_NOTICE.txt            # 立绘素材版权声明
    └── assets/README.md             # 自定义立绘替换说明
```

---

## 快速开始

### 1. 环境准备（Windows 10 / 11）
1. 确保系统已安装 Microsoft Edge WebView2 Runtime（Win11 默认自带）。
2. 安装 Google 官方 Colab CLI 并完成本机登录授权（推荐用 `uv`）：
   ```powershell
   uv tool install google-colab-cli
   colab auth login
   ```
3. 安装桌宠 Python 依赖：
   ```powershell
   pip install -r requirements.txt
   ```
4. （可选）如需使用 Google Drive 持久化同步 LoRA、参考音频与出图结果，请安装 [rclone](https://rclone.org/) 并配置好远端（默认远端别名 `gdrive_save`，可通过环境变量 `COLAB_RCLONE_REMOTE` 修改）。

### 2. 启动桌宠
双击 [启动Colab桌宠.pyw](启动Colab桌宠.pyw)，或通过命令行启动：
```powershell
pythonw 启动Colab桌宠.pyw
```
（使用 `--restart` 参数可替换已运行的桌宠实例。）

### 3. 命令行常见用法
```powershell
# 查看当前会话与 CU 余额
python colab_ctl.py status

# 一键拉起 Anima (L4) / MiniMax-H3 (G4) / Drama3 (G4) / Qwen LLM (G4)
python colab_ctl.py up --profile anima --gpu L4
python colab_ctl.py up --profile qwen_llm --gpu G4

# 查看或重建 Cloudflared 公网隧道
python colab_ctl.py tunnel

# 运行端到端冒烟验证并拉回结果
python colab_ctl.py smoke --profile anima

# 同步输出至 Google Drive 并拉回本地 _downloads/
python colab_ctl.py sync --pull

# 关闭会话并停止计费（--all 可一并释放网页端遗留实例）
python colab_ctl.py down --all
```

---

## 可选配置与环境变量

| 变量名 / 配置文件 | 作用 | 默认值 / 说明 |
|---|---|---|
| `.cf_tunnel_secret.json` | 绑定个人 Cloudflare Named Tunnel 固定域名与固定 LLM Key | 复制 `.cf_tunnel_secret.example.json` 修改后放到项目根目录（已忽略提交） |
| `COLAB_EXE` / `COLAB_PY` | 手动指定 `colab.exe` 或 `google-colab-cli` Python 路径 | 默认自动从 `PATH`、`~/.local/bin` 与 `%APPDATA%\uv\tools\google-colab-cli` 查找 |
| `RCLONE_EXE` / `RCLONE_CONF` | 手动指定 `rclone.exe` 与 `rclone.conf` 路径 | 默认自动从 `PATH`、`%APPDATA%\rclone`、`~/.config/rclone` 或 Scoop 查找 |
| `COLAB_RCLONE_REMOTE` | 自定义 `rclone` 远端名称 | 默认 `gdrive_save` |
| `COLAB_LORA_DIR` / `COLAB_DRAMA3_REFS_DIR` / `COLAB_DOWNLOAD_DIR` | 指定本地 LoRA、参考音频或结果拉回目录 | 默认项目目录下 `loras/`、`drama3_refs/`、`_downloads/` |
| `REQUIRED_LORAS_JSON` | 传入 `{文件名: 精确字节数}` JSON 启用云端 LoRA 严格校验 | 默认不校验特定文件名，仅按 `*.safetensors` 同步 |
| `LORA_CHAR` / `LORA_STYLE` / `SMOKE_PROMPT` | 为 `verify_smoke.py` 指定测试 LoRA 与提示词 | 默认使用公共 `anima-base-v1.0.safetensors` 直接完成底座生图校验 |

---

## 隐私与安全约定

- 所有与个人账号相关的配置（`.cf_tunnel_secret.json`、`colab_pet/pet_state.json`、`colab_pet/pet_error.log`、`rclone.conf`、`*.log`）均已加入 `.gitignore`。
- `colab-cli` 与 `rclone` 的登录授权均留在本地系统凭据库（`keyring`）或仅开机时临时传输至远端内存/临时目录，桌宠状态文件不保存任何 OAuth Refresh Token 或 LLM Bearer Key 明文。
- 如使用默认的 `*.trycloudflare.com` 临时公网域名且无外部鉴权，请注意用完即点「一键停卡」释放实例与隧道。
