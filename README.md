# Colab-Desktop-Pet（Colab 桌面宠物与一键算力控制台）

面向 Windows 桌面的 Google Colab 计算单元（CU）监控桌宠与一键 GPU 工作流控制台，基于 `pywebview` + Edge WebView2 + Win32 `SetWindowRgn` 像素级不规则裁剪悬浮窗构建。

> 本仓库源代码遵循 [LICENSE](LICENSE)（MIT 开源协议）。按 [colab_pet/ASSETS_NOTICE.txt](colab_pet/ASSETS_NOTICE.txt) 声明，仓库已随附一套重绘的四表情 Q 版桌宠立绘（仅限非商业个人展示与学习参考），且**不包含**任何个人账号凭据、邮箱、自定义域名或本地私有路径。

---

## 核心功能

### 1. 桌面悬浮宠物挂件（[启动Colab桌宠.pyw](启动Colab桌宠.pyw) & [colab_pet/](colab_pet/)）
- **像素级无边框透明浮窗**：使用 Win32 `SetWindowRgn` 贴合立绘与气泡边缘裁剪，鼠标穿透透明区域，支持单实例互斥与跨屏高帧率拖拽。
- **实时算力监控与防烧卡保护**：
  - 轮询 `google-colab-cli` 获取当前 CU 余额、实时消耗速率（`CU/hr`）、可用 GPU 池状态与活跃/孤儿会话；
  - 本地记录今日小账本（今日实耗 CU、连续开机计时），仅保存在本地忽略文件 `colab_pet/pet_state.json`，不持久化账号令牌；
  - 定期发送 Colab 三层轻保活心跳，并支持空闲超时自动停卡与算力告警。
- **五合一桌宠面板**：集成 **积分看板 / 今日小账本 / 快捷控制台 / 偏好设置 / 账号与环境**，支持多生态/双栖并发开机、双隧道独立直达、工作流/素材/LoRA 拖拽投递与智能路由弹窗、输出自动拉回与内置 LLM 对话气泡。
- **开箱即用账号与环境接入（第 5 页）**：
  - **多账号切换保险箱**：支持前端一键发起 Google OAuth2 授权绑定/切换多组 Colab 账号，令牌与会话列表隔离保存于系统凭据库（`keyring`）及用户私密配置目录，状态文件仅展示脱敏邮箱；
  - **Google Drive 云盘引导**：支持弹窗一键触发 `rclone` 浏览器授权 `[gdrive_save]` 或导入已有 `rclone.conf`；
  - **工具链自适应探测 & 固定隧道配置**：自动探测本机 `colab.exe` / `rclone.exe` / 虚拟环境并支持文件选择器手动定位，可在前端直接保存 Cloudflare 固定域名与 Bearer Key（单向写入 `.cf_tunnel_secret.json`，不回显明文密钥）。
- **内置 Q 版四表情立绘 + 矢量兜底**：[colab_pet/assets/](colab_pet/assets/README.md) 现已开箱附带统一轮廓对齐的 `1200 × 1004` 四表情透明立绘（`idle` / `blink` / `proud` / `sad`）；支持随时替换为自定义 PNG，若删除图片则自动切回内存生成的 `500 × 418` 原创矢量兜底角色。

### 2. CLI 总控、5 套生态与双栖共驻（[colab_ctl.py](colab_ctl.py)）
支持命令行与桌宠界面双重触发，内置 5 套幂等部署 Profile，并支持通过 `--profiles` 将「ComfyUI 生态 + Qwen3.8 LLM」并发部署至同一 GPU 实例（双栖模式自带显存预留保护与独立双隧道）：
1. **`anima`（默认 `L4`）**：[colab_bootstrap_anima.py](colab_bootstrap_anima.py) + [verify_smoke.py](verify_smoke.py) — ComfyUI + Anima 底座 + Cloudflared 公网隧道。
2. **`minimax`（默认 `G4`）**：[colab_bootstrap_minimax_h3.py](colab_bootstrap_minimax_h3.py) + [verify_minimax_h3.py](verify_minimax_h3.py) + [workflows/minimax_h3/](workflows/minimax_h3/) — MiniMax-H3 生图 / 图像编辑 / 视频生声全套模型与工作流。
3. **`drama3`（默认 `G4`）**：[colab_bootstrap_drama3.py](colab_bootstrap_drama3.py) + [verify_drama3.py](verify_drama3.py) + [workflows/drama3/](workflows/drama3/) — Fish Audio Drama 3 语音演戏与克隆（含 Qwen3-ASR 辅助）。
4. **`qwen_27b`（默认 `L4`，兼容 `T4`）**：[colab_bootstrap_qwen_llm.py](colab_bootstrap_qwen_llm.py) + [verify_qwen_llm.py](verify_qwen_llm.py) — `llama.cpp`（Qwen3.8-27B 轻量化稠密版，适合 16G–24G 显存或双栖 co-host）。
5. **`qwen_llm`（默认 `G4`）**：[colab_bootstrap_qwen_llm.py](colab_bootstrap_qwen_llm.py) + [verify_qwen_llm.py](verify_qwen_llm.py) — `llama.cpp`（Qwen3.8-Flash-Next 180B MoE）+ SSE 网关 + Bearer Key 鉴权推理服务。

---

## 目录结构

```text
Colab-Desktop-Pet/
├── 启动Colab桌宠.pyw                # Windows 无控制台黑框桌宠启动入口
├── colab_ctl.py                     # 本机总控 CLI（up/status/tunnel/smoke/sync/run/down/doctor/diag）
├── colab_core/                      # 共享内核（accounts / colab_client / file_sync / health / runtime）
├── colab_security.py                # 系统凭据保险箱 (keyring)、密钥脱敏与域名安全校验
├── colab_cli_bridge.py              # google-colab-cli 虚拟环境统一桥接助手
├── colab_remote_common.py           # 远端部署与冒烟验证共享基础库
├── colab_pet_bridge_node.py         # ComfyUI 服务端直铺画布 + 公网口令鉴权扩展
├── colab_bootstrap_*.py             # 四套云端 Colab 实例幂等自举脚本
├── verify_*.py                      # 四套端到端冒烟验证脚本
├── build_windows_package.py         # Windows 便携版 / PyInstaller 发行包构建与安全扫描脚本
├── colab_pet.spec                   # PyInstaller 单目录 (onedir) 打包配置
├── stack_manifest.json              # 五套生态依赖、模型与显存门槛版本清单
├── workflows/                       # 预置 ComfyUI 工作流 JSON（drama3 / minimax_h3）
├── tests/                           # 架构复现与安全可靠性 pytest 测试集
├── .cf_tunnel_secret.example.json   # Cloudflare 命名隧道可选配置模板
├── requirements.txt                 # 桌宠 Python 依赖基线
├── requirements-lock.txt            # 已验证精确锁定依赖清单
└── colab_pet/
    ├── pet_app.py                   # pywebview + Edge WebView2 + Win32 区域裁剪窗体
    ├── colab_backend.py             # 状态采集、账本统计、保活、隧道与任务调度后端
    ├── ui.html                      # 桌宠气泡与控制台前端界面
    ├── ASSETS_NOTICE.txt            # 立绘素材使用声明与非商业边界
    └── assets/                      # 开箱内置 Q 版四表情透明立绘 + 替换说明
```

---

## 快速开始

### 1. 环境准备（Windows 10 / 11）
1. 确保系统已安装 Microsoft Edge WebView2 Runtime（Win11 默认自带）。
2. 安装 Google 官方 Colab CLI 并完成登录授权（推荐用 `uv`；也可先启动桌宠后在第 5 页「账号与环境」通过界面引导登录）：
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

# 一键拉起 Anima (L4) / Qwen 27B (L4) / Qwen 180B (G4) / MiniMax+Qwen27B 双栖共驻 (G4)
python colab_ctl.py up --profile anima --gpu L4
python colab_ctl.py up --profile qwen_27b --gpu L4
python colab_ctl.py up --profile qwen_llm --gpu G4
python colab_ctl.py up --profiles anima,qwen_27b --gpu G4

# 查看或重建 Cloudflared 公网隧道
python colab_ctl.py tunnel

# 运行端到端冒烟验证并拉回结果
python colab_ctl.py smoke --profile anima

# 同步输出至 Google Drive 并拉回本地 _downloads/
python colab_ctl.py sync --pull

# 启动前全维度环境自检 / 导出全量脱敏排障诊断包 (.zip)
python colab_ctl.py doctor --profile anima --gpu L4
python colab_ctl.py diag

# 关闭会话并停止计费（--all 可一并释放网页端遗留实例）
python colab_ctl.py down --all
```

### 4. Windows 打包与测试校验
```powershell
# 运行完整自动化测试套件
python -m pytest tests -v

# 构建零额外依赖原生 EXE 便携发布包（输出至 dist/）
python build_windows_package.py --mode portable

# 或构建免安装 Python 的 PyInstaller 独立目录包
python build_windows_package.py --mode pyinstaller
```

---

## 可选配置与环境变量

| 变量名 / 配置文件 | 作用 | 默认值 / 说明 |
|---|---|---|
| `.cf_tunnel_secret.json` | 绑定个人 Cloudflare Named Tunnel 固定域名与固定 LLM Key | 复制 `.cf_tunnel_secret.example.json` 修改后放到项目根目录（已忽略提交） |
| `COLAB_EXE` / `COLAB_PY`（或 `COLAB_PET_*`） | 手动指定 `colab.exe` 或 `google-colab-cli` Python 路径 | 默认自动从 `PATH`、`~/.local/bin`、`%APPDATA%\uv\tools\google-colab-cli` 与桌宠第 5 页偏好查找 |
| `RCLONE_EXE` / `RCLONE_CONF`（或 `COLAB_PET_*`） | 手动指定 `rclone.exe` 与 `rclone.conf` 路径 | 默认自动从 `PATH`、`%APPDATA%\rclone`、`~/.config/rclone`、Scoop 或桌宠偏好查找 |
| `COLAB_RCLONE_REMOTE` | 自定义 `rclone` 远端名称 | 默认 `gdrive_save` |
| `COLAB_LORA_DIR` / `COLAB_DRAMA3_REFS_DIR` / `COLAB_DOWNLOAD_DIR` | 指定本地 LoRA、参考音频或结果拉回目录 | 默认项目目录下 `loras/`、`drama3_refs/`、`_downloads/` |
| `REQUIRED_LORAS_JSON` | 传入 `{文件名: 精确字节数}` JSON 启用云端 LoRA 严格校验 | 默认不校验特定文件名，仅按 `*.safetensors` 同步 |
| `LORA_CHAR` / `LORA_STYLE` / `SMOKE_PROMPT` | 为 `verify_smoke.py` 指定测试 LoRA 与提示词 | 默认使用公共 `anima-base-v1.0.safetensors` 直接完成底座生图校验 |

---

## 隐私与安全约定

- 所有与个人账号相关的配置（`.cf_tunnel_secret.json`、`colab_pet/pet_state.json`、`colab_pet/pet_error.log`、`rclone.conf`、`*.log`）均已加入 `.gitignore`。
- `colab-cli` 与 `rclone` 的登录授权均留在本地系统凭据库（`keyring`）或仅开机时临时传输至远端内存/临时目录，桌宠状态文件不保存任何 OAuth Refresh Token 或 LLM Bearer Key 明文。
- 如使用默认的 `*.trycloudflare.com` 临时公网域名且无外部鉴权，请注意用完即点「一键停卡」释放实例与隧道。
