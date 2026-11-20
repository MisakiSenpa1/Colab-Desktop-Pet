# Google Colab 算力部署、桌宠 Studio 双窗工作台与排障总指南

> 适用范围：基于 Google Colab GPU 运行时，面向 ComfyUI（Anima / MiniMax-H3 / Drama 3）与 `llama.cpp`（Qwen3.8-27B / Qwen3.8-180B）实现「本机零重依赖、云端高速拉取、可选跨盘持久化、双向鉴权隧道、跑完即停卡、多账号异构双开与 AI 副驾运维」。

---

## 一、双模式开箱与架构总览

本仓库采用 **单一代码库 + 配置驱动** 的双模式设计，新机器无需私训权重或预先绑定云盘即可直接体验公开基础版，也支持按需开启个人增强能力。

### 1.1 「公开基础版」vs「个人增强版」对比

| 能力维度 | 公开基础版（默认开箱友好） | 个人增强版（配置驱动开启） |
|---|---|---|
| **前置门槛** | 仅需已登录 `google-colab-cli`，**无需** `[gdrive_save]` 云盘、**无需**自定义域名、**无需**私训 LoRA | 绑定 `rclone` `[gdrive_save]`，可选配置固定 Cloudflare Named Tunnel 与私训素材 |
| **启动前预检（`doctor` / Preflight）** | `colab_ctl.py doctor` 未指定 `--edition` 且未设 `COLAB_PET_EDITION` 时默认按公开基础版体检；缺失 `[gdrive_save]` 降级为 `warn`（提示可随时选配云盘持久化），**不阻塞开机** | 显式传入 `--edition personal` 或设置 `COLAB_PET_EDITION=personal` 触发个人增强版强校验（强制校验 `[gdrive_save]` 云盘与个人资产配置） |
| **Anima 云端自举** | 自动跳过私训 LoRA 拉取，直接高速拉取官方 Anima 三大底座权重启动 ComfyUI | 设置 `ANIMA_STRICT_LORAS=1` 或上传含 `[gdrive_save]` 的 `rclone.conf` 时拉取并强校验自训 LoRA |
| **端到端冒烟（`smoke`）** | 检测到无私训 LoRA（或设 `ANIMA_SMOKE_MODE=base`）时，自动切换纯 Anima 底座工作流（`UNET -> KSampler`）出图验证 | 私训 LoRA 齐备且未强制 `base` 时，执行完整 LoRA 冒烟工作流 |
| **Drama 3 参考音频** | 默认回退工作区 `drama3_refs` 目录，云端自动选用首个可用 `.wav` | 可通过 `COLAB_DRAMA3_REFS_DIR` 与 `DRAMA3_DEFAULT_REF_WAV` 指定本机目录与默认音色 |

### 1.2 账号分工（单账号 / 双账号 / 多账号算力池）

| 角色 | Google 账号示例 | 承载工具 | 职责说明 |
|---|---|---|---|
| **算力计算号（可配置多号）** | `<compute-account@example.com>`（可在桌宠多账号池绑定多个 Google 账号） | `google-colab-cli` (v0.8.0) | 按需申请、控制和释放 Colab GPU 运行时（主力 `L4`，大模型可选 `A100` / `G4` / `H100`），仅使用云端临时盘；多账号可自动容灾换号或异构双开。 |
| **云端存储号（可选）** | `<storage-account@example.com>`（可选绑定独立 Google Drive 账号） | `rclone` (v1.76.0，远端名 `gdrive_save`) | 可选配置，用于持久保存自训 LoRA、工作流模板、参考音频与 ComfyUI 产出归档。 |

### 1.3 本机关键路径与核心入口脚本

| 组件 / 文件 | 通用路径示例 / 仓库相对路径 | 作用说明 |
|---|---|---|
| Colab CLI 主程序 | `%USERPROFILE%\.local\bin\colab.exe` | 官方命令行入口（基线版本 `0.8.0`，可自动探测或通过环境变量覆盖） |
| Colab CLI 虚拟环境 | `%USERPROFILE%\AppData\Roaming\uv\tools\google-colab-cli\Scripts\python.exe` | 托管 Python 环境，供本机桥接脚本安全调用 `ColabRuntime` |
| Colab 会话状态缓存 | `%USERPROFILE%\.config\colab-cli\sessions.json` | 缓存活跃会话 `endpoint`、`url`、`kernel_id`、`session_id` |
| rclone 程序与配置（可选） | `<RCLONE_DIR>\rclone.exe`<br>`%APPDATA%\rclone\rclone.conf`（或任意自定义路径） | 含 `[gdrive_save]` OAuth 刷新令牌（仅开机按需上传至云端 `/tmp/rclone.conf`，严禁提交公开库） |
| **桌面双窗启动入口** | [启动Colab桌宠.pyw](启动Colab桌宠.pyw) | 无终端黑框启动 Q 版悬浮桌宠挂件，并可一键打开 Colab Studio 双窗工作台 |
| **本机总控 CLI** | [colab_ctl.py](colab_ctl.py) | 统一调度 `doctor`、`diag`、`status`、`up`、`tunnel`、`smoke`、`run`、`wait`、`sync`、`exec`、`mcp-stdio`、`down` 等命令 |
| **云端自举脚本集** | [colab_bootstrap_anima.py](colab_bootstrap_anima.py) / [colab_bootstrap_minimax_h3.py](colab_bootstrap_minimax_h3.py)<br>[colab_bootstrap_drama3.py](colab_bootstrap_drama3.py) / [colab_bootstrap_qwen_llm.py](colab_bootstrap_qwen_llm.py) | 在 Colab VM 内执行 GPU 自检、工具部署、模型高速拉取、鉴权网关与隧道启动 |
| **端到端冒烟验证集** | [verify_smoke.py](verify_smoke.py) / [verify_minimax_h3.py](verify_minimax_h3.py)<br>[verify_drama3.py](verify_drama3.py) / [verify_qwen_llm.py](verify_qwen_llm.py) | 分别验证 Anima（支持纯底座与 LoRA 双模式）、MiniMax-H3、Drama 3 与 Qwen3.8 LLM 全链路可用 |

### 1.4 云端 Google Drive（可选）与 Colab VM 目录结构

- **Google Drive 云盘（`gdrive_save:`，个人增强版启用）**：
  - `gdrive_save:Colab_ComfyUI_Models/loras/`：存放自训 Anima LoRA 权重及 `sha256sums.txt` 校验清单；
  - `gdrive_save:Colab_ComfyUI_Models/minimax_h3/workflows/`：存放固化后的 MiniMax-H3 工作流模板；
  - `gdrive_save:Colab_ComfyUI_Output/`：接收云端 `/root/comfy_out/` 强校验同步回来的成品与测试图。
- **Colab VM 内部临时目录**（实例释放即清空）：
  - `/content/ComfyUI`：官方 ComfyUI 仓库（保留 Colab 预装 CUDA PyTorch）
  - `/root/comfy_out`：ComfyUI 输出根目录
  - `/root/comfy_db/comfyui.db`：SQLite 状态库
  - `/root/comfy_colab.log` & `/root/llama_server.log`：ComfyUI / LLM 守护进程日志
  - `/root/cloudflared.log`、`/root/comfy_tunnel_url.txt`、`/root/llm_tunnel_url.txt`：隧道日志与当前公网 URL
  - `/root/colab_runtime_versions.json`：当前云端推理引擎与模型版本快照
  - `/tmp/rclone.conf`：临时云盘凭据（仅启用云盘同步时按需传入）

---

## 二、连接、授权与凭据安全备忘

### 2.1 `google-colab-cli` 登录原理与换号步骤

- **无需 `gcloud` / 无需绑卡**：CLI 默认使用 `--auth=oauth2` 及内置桌面端 OAuth 流程，直接在浏览器登录个人 Google 账号授权即可。
- **凭据存储位置**：OAuth Refresh Token 加密保存在 Windows 系统凭据管理器（`keyring` 服务名 `google-colab-cli`；桌宠多账号保险箱与隧道/副驾密钥统一通过 [colab_security.py](colab_security.py) 存入 `google-colab-cli-pet-vault`）。
- **查看当前登录账号**：
  ```powershell
  colab whoami
  ```
- **授权失效或需要手动切换默认算力账号时**（推荐直接在桌宠第 5 页「账号与环境」或 Studio 中点击管理多账号，无需手工删库）：
  ```powershell
  # 1. 清理 Windows 系统凭据库中的旧 colab OAuth 令牌
  & "$env:USERPROFILE\AppData\Roaming\uv\tools\google-colab-cli\Scripts\python.exe" -c "import keyring; keyring.delete_password('google-colab-cli', 'oauth_token')"

  # 2. 重新触发 OAuth2 浏览器授权流程
  colab whoami
  ```

### 2.2 `rclone` 云盘重连与自建 `client_id` 建议（可选）

- **查看云盘状态与余量**：
  ```powershell
  rclone --config "<你的 rclone.conf 路径>" about gdrive_save:
  ```
- **令牌过期重新授权**：
  ```powershell
  rclone --config "<你的 rclone.conf 路径>" config reconnect gdrive_save:
  ```
- **长期防限流优化提示**：rclone 默认共享 Google Drive `client_id` 高频上传易触发 `403 RATE_LIMIT_EXCEEDED`。如需频繁批量上传，可按官方文档（`https://rclone.org/drive/#making-your-own-client-id`）在 Google Cloud Console 免费创建一个仅启用 Drive API 的「桌面应用」`client_id` 填入配置，无需绑卡。

### 2.3 Cloudflare 隧道与凭据存储安全约定

- 本地 `.cf_tunnel_secret.json`（已被 [.gitignore](.gitignore) 忽略）**仅保存公开固定域名与 `vault:` 键名引用**，严禁落盘明文 `tunnel_token` 或 `llm_api_key`；历史明文配置会在加载时自动迁移写入 Windows 系统凭据库并擦除磁盘明文。
- **跨机迁移提示**：拷贝 `.cf_tunnel_secret.json` 中的 `vault:` 引用**不等于**迁移 Windows 系统凭据库里的真实密钥；换机或重装系统后请在桌宠/Studio 中重新录入或绑定凭据。

---

## 三、硬件选型、速度基准与 CU 计费规则

### 3.1 硬件控制与计费机制

1. **硬件规格绑定于 GPU 档位**：CPU 核数与临时盘容量由 Colab 档位决定；仅部分卡型支持 Standard / High-RAM（`--high-mem`）切换，而 **`L4` 与 `G4` 为固定高配内存规格**（如 `L4` 约 `53 GiB` RAM，`G4` 约 `176 GiB` RAM）。
2. **开机即持续扣费（Compute Units, CU）**：**只要实例未终止（即使空闲），都会按固定速率持续扣减 CU**。建议配合桌宠闲置自动停卡或 `--auto-stop` 使用。
3. **存储与计算解耦**：模型校验与本地文件整理均在零 CU 下完成，开机后利用 Google 机房内网高速并发拉取权重。
4. **费率估算与实测优先级**：系统内置各卡型默认估算费率仅作开机前参考；当账号实际运行采集到实测耗速（持久化于本地 `measured_gpu_rates`）后，始终以账号实测费率为准。

### 3.2 各加速卡参考对比

| 加速卡 | 显存 / 特性 | 参考每小时 CU 消耗 | 适用场景与结论 |
|---|---|---|---|
| **CPU** | 无独显（约 `12.7 GB` RAM） | 约 `0.08 CU/hr` | 仅适合环境调试与轻量中转，无法高速推理 |
| **T4** | `15360 MiB`（Turing，无原生 BF16） | 约 `1.15 CU/hr` | 显存偏紧且无原生 BF16，跑 Anima 大分辨率易爆显存或偏色，不推荐 |
| **`L4`（Anima / 27B 主力 ★）** | `23034 MiB` (`22 GiB`)，Ada 架构，原生 BF16 + SDPA | 约 `1.94 CU/hr` | 性价比极高，轻松承载 Anima 生图或 `qwen_27b` 对话服务 |
| **A100** | `40GB` / `80GB`（`--high-mem`） | 约 `5.4 – 7.5 CU/hr` *（估算参考）* | `G4` 紧缺时的重型视频/大模型高显存备选 |
| **`G4`（MiniMax-H3 / 180B 首选 ★）** | RTX PRO 6000 Blackwell（`96 GB` 显存 / `176 GiB` RAM / CUDA 13） | 约 `10.05 CU/hr` | 完美承载 MiniMax-H3 全生态（含 `4096×4096` 16MP 直出）、Drama 3 与 `qwen_llm` 180B 旗舰对话 |
| **H100** | 旗舰计算卡 | 约 `14.5 – 18+ CU/hr` *（估算参考）* | 受账号层级限制且消耗较快，按需启用 |

### 3.3 参考带宽与耗时基准

- **Colab 机房 → Hugging Face**（`aria2c -x 16 -s 16`）：实测 `200 ~ 900 MiB/s`；Anima 三大底座权重（合计约 `5.6 GB`）拉取仅需 ~20 秒；MiniMax-H3 全生态（~`88 GiB`）约 5 分钟拉齐。
- **ComfyUI 冷启动与出图**：Anima 冷启约 `28s`、`832×1440` 28 步出图与回收全程约 `60s`；MiniMax-H3 在 `G4` 上 `2048×2048` 文生图冷启约 `90s`、热态图像编辑约 `15s`、`768×768` 73 帧原生有声视频约 `45s`。

---

## 四、日常标准操作与多生态工作流（CLI）

### 4.1 公开基础版新机开箱标准流程（零私训 LoRA / 可选云盘）

```powershell
# 1. 启动前环境体检（未指定 --edition 且未设 COLAB_PET_EDITION 时默认按公开基础版检查，缺 [gdrive_save] 仅报 warn 不阻塞）
python colab_ctl.py doctor --profile anima --gpu L4

# 2. 开 L4 卡一键部署 Anima 底座 + 双向鉴权公网隧道
python colab_ctl.py up --gpu L4

# 3. 端到端冒烟验证（未部署私训 LoRA 时自动走纯 Anima 底座工作流）
python colab_ctl.py smoke

# 4. 用完立即停卡并核实 Usage rate 归零
python colab_ctl.py down --all
```

> **体检模式说明**：`python colab_ctl.py doctor --profile anima --gpu L4` 在未指定 `--edition` 且未设置 `COLAB_PET_EDITION` 时，默认按**公开基础版**执行体检（缺失 `[gdrive_save]` 仅提示 `warn`，返回码为 `0`）；如需触发**个人增强版**强校验，请显式运行 `python colab_ctl.py doctor --profile anima --gpu L4 --edition personal` 或设置环境变量 `COLAB_PET_EDITION=personal`。

### 4.2 个人增强版工作流（含云盘同步与批量跑批）

```powershell
# 1. 零 CU 校验并增量上传自训 LoRA 到 Google Drive（已存在自动跳过）
python colab_ctl.py sync-loras

# 2. 查看当前活跃会话、CU 余额与实时速率
python colab_ctl.py status

# 3. 提交任意本地出图脚本，自动等待队列清空 -> 同步云盘 -> 拉回本机 -> 自动停卡省 CU
python colab_ctl.py run <your_batch_script.py> --prefix batch01 --pull --auto-stop
```

常用参数说明：
- `--prefix <目录>`：仅同步 `/root/comfy_out/<目录>`，避免重复打包历史旧图。
- `--pull`：同步后打包下载并安全解压至本机 `_downloads/<目录>/`（带路径穿越防呆与时间戳防覆盖）。
- `--auto-stop`：队列完成且同步校验通过后自动销毁实例，杜绝忘关机烧 CU。
- `--no-wait` / `--no-sync`：连续提交多脚本时跳过中间等待，最后统一 `wait` + `sync`。

### 4.3 公网隧道管理与安全鉴权机制

```powershell
# 查看当前隧道地址，并在本机端到端验证公网双向鉴权（无密钥拒 401/403 + 有效密钥通 200）
python colab_ctl.py tunnel

# 隧道长时间闲置断连或需轮换地址时，一键重启重建隧道
python colab_ctl.py tunnel --restart

# 网页端手工出图后随时归档并拉回本机 _downloads/all/
python colab_ctl.py sync --pull
```

- **公网双向鉴权保护**：
  - ComfyUI 入口通过云端 v3 桥接节点 [colab_pet_bridge_node.py](colab_pet_bridge_node.py) 启用一次性签名票据与鉴权网关保护；
  - LLM 对话网关通过 [colab_sse_gateway.py](colab_sse_gateway.py) 强制要求 `Authorization: Bearer` / `x-api-key` 恒时校验，默认不反射未知浏览器 `Origin` 与 `Allow-Credentials`（需要特定 Web 前端跨域时可通过 `SSE_ALLOWED_ORIGINS` 配置精确白名单）；
  - 本机将「日志 URL 提取正则」与「凭据发送源站同源授权」严格解耦，避免向未授权域名泄露密钥；`tunnel` 快速复用通道同样强制核验公网双向鉴权通过后才放行。
- **备用本地 SSH 端口转发**（需本机存在 `~/.ssh/id_ed25519.pub` 等公钥）：
  ```powershell
  ssh -o "ProxyCommand=%USERPROFILE%\.local\bin\colab.exe ssh --proxy-mode -s comfy" -N -L 8188:127.0.0.1:8188 root@colab
  ```

### 4.4 多生态 Profile 一键切换（MiniMax-H3 / Drama 3 / Qwen3.8 LLM）

所有支持路线均在 [stack_manifest.json](stack_manifest.json) 统一登记：

```powershell
# 1. MiniMax-H3 全生态（生图 / 编辑 / 原生有声视频，推荐 G4，约 88 GiB）
python colab_ctl.py up --profile minimax
# 省盘模式（仅核心生图 + FL2VA 视频，约 59.8 GiB）
python colab_ctl.py up --profile minimax --minimax-lite
python colab_ctl.py smoke --profile minimax

# 2. Drama 3 语音克隆与自由演戏（推荐 G4；支持 COLAB_DRAMA3_REFS_DIR 指定本地参考音频目录）
python colab_ctl.py sync-drama3-refs
python colab_ctl.py up --profile drama3
python colab_ctl.py smoke --profile drama3

# 3. Qwen3.8 大模型对话（llama.cpp + SSE 鉴权网关：qwen_27b 推荐 L4，qwen_llm 180B 推荐 G4）
python colab_ctl.py up --profile qwen_27b --gpu L4
python colab_ctl.py smoke --profile qwen_27b
```

- 固化工作流模板存放在 [workflows/minimax_h3/](workflows/minimax_h3/) 与 [workflows/drama3/](workflows/drama3/)，云端部署后可直接于 ComfyUI 左侧 `Workflows` 加载。

### 4.5 远端诊断执行与 MCP Stdio 工具服务

```powershell
# 在当前活跃 Colab 实例执行只读诊断命令
python colab_ctl.py exec --mode bash --code "nvidia-smi"

# 启动 JSON-RPC 2.0 MCP Stdio 工具服务（供外部 Agent / Claude Code CLI 挂载）
python colab_ctl.py mcp-stdio
```

> **安全边界提示**：独立无头 `mcp-stdio` 模式在没有交互式审批弹窗时默认采用 **Fail-Closed 只读策略**——仅开放结构化只读诊断与白名单只读命令，默认拒绝无审批执行任意远端写代码或开关计费实例（仅在显式配置受控无头授权标志时放行）。

---

## 五、已实战踩坑与关键排障方案

| 问题现象 | 根本原因 | 已固化的工程解法 |
|---|---|---|
| **`colab exec` 偶发 `RuntimeError: Connection was lost.` 且进程挂死** | 经本机代理首次 WSS 握手常超默认 `10s`，官方 CLI 不重试且异常时未停非守护 WS 线程；Windows 未设 `PYTHONUTF8=1` 易中文编码报错。 | [colab_core/colab_client.py](colab_core/colab_client.py) 与 [colab_cli_bridge.py](colab_cli_bridge.py) 封装 `30s` 握手等待 + 自动重试 + `PYTHONUTF8=1` + `finally` 强制清理。 |
| **云端 `urllib` 下载 `rclone` 报 `HTTP 403`** | WAF 拦截 `Python-urllib` 默认 User-Agent。 | 自举脚本统一使用浏览器 UA，并内置 `curl` 与 `apt-get` 两级回退。 |
| **通过公网隧道打开 ComfyUI 提交队列或 WS 报 `403`** | ComfyUI 默认同源中间件拦截 `Host (127.0.0.1)` 与公网 `Origin` 不一致请求。 | 后端仅监听 `127.0.0.1`，配合 `--enable-cors-header "*"` 及 [colab_pet_bridge_node.py](colab_pet_bridge_node.py) 公网认证网关实现安全访问。 |
| **远端后台起守护进程导致 `exec` 永不返回** | 子进程继承 Jupyter/IPython 标准流管道导致等待 EOF。 | 后台守护统一设置 `stdin=DEVNULL`、`start_new_session=True`、`close_fds=True` 并立即关闭父进程日志句柄。 |
| **本机经代理用 `rclone` 上传大文件在 100% 假死或限流** | 本地代理对 Go HTTP/2 长传偶发假死，大分块积压超时，或触发共享 ID 限流。 | `sync-loras` 已固化 `--disable-http2` + `--drive-chunk-size 8M` + `--timeout 5m` + `--retries 10` + 预检 `lsl` 幂等跳过。 |
| **云端重装 ComfyUI 依赖覆写坏 Colab 预装 CUDA PyTorch** | 官方 `requirements.txt` 包含 `torch`/`torchvision`/`torchaudio`。 | 云端部署自动剔除 PyTorch 三件套后再通过 `uv pip install --system` 补齐轻依赖。 |
| **PowerShell 内联 Python 引号转义频发语法错** | PowerShell 对双引号嵌套极易拆分异常。 | 远端脚本统一写入 UTF-8 临时文件执行；本机多行片段使用单引号 heredoc（`@' ... '@ \| python -`）。 |

---

## 六、Q 版桌宠挂件、Studio 双窗工作台、多账号算力池与 AI 副驾

### 6.1 Q 版悬浮桌宠挂件（[启动Colab桌宠.pyw](启动Colab桌宠.pyw)）

基于本机 `pythonw.exe + pywebview + Edge WebView2` 运行（**零终端黑框、运行时零 `Pillow` / `aiohttp` / `pywin32` 依赖**，自带单实例互斥锁与左/右贴边自动吸附转向）：

- **五页思维气泡看板（[colab_pet/ui.html](colab_pet/ui.html)，顶部 `< 1/5 >` 翻页）**：
  1. **积分看板 (`1/5`)**：后台自动无感轮询实时 CU 余额、消耗速率（`CU/hr`）、预计剩余时长、活跃 GPU 槽位与网页端孤儿实例提醒。
  2. **今日小账本 (`2/5`)**：按账号独立记录每日起始余额、今日实耗 CU（充值/续费不会冲减已耗统计）与连续运行时长，本地持久化于 `pet_state.json`（不含明文 Token）。
  3. **快捷控制台 (`3/5`)**：选择生态 Profile 与 GPU 档位一键部署开机、三级极速打开 ComfyUI / LLM 隧道、一键同步拉回成图、跳转 Colab 网页及**二次确认停卡**。
  4. **偏好设置 (`4/5`)**：设置总是置顶、轮询间隔（`30s / 60s / 2m`）、桌宠体型缩放、长时忘关提醒及闲置自动停卡策略。
  5. **账号与环境 (`5/5`)**：管理多账号算力池、云盘绑定、固定隧道域名设置，并可在底部一键 **导出脱敏诊断包** 或 **打开日志**。
- **立绘素材与零 Pillow 窗口裁剪**：
  - 静态资源位于 [colab_pet/assets/](colab_pet/assets/README.md)，已采用轻量化 `.webp` 资产（由独立重绘的四表情 `1200 × 1004`「夏儿」透明立绘离线导出）：`idle.webp`（日常待机）、`blink.webp`（眨眼）、`proud.webp`（开心/闲置省电）、`sad.webp`（低余额/异常/忘关提醒）、多尺寸 `pet.ico` 与预提取二值轮廓遮罩 `alpha_masks.json`；
  - 本地静态服务严格校验 `Host` 环回与 `/assets/` 白名单扩展名，非商业使用边界详见 [colab_pet/ASSETS_NOTICE.txt](colab_pet/ASSETS_NOTICE.txt)。
- **三级极速隧道与抗波动机制**：
  - 无卡时 `<2.5s` 秒退提示；已开机优先命中本地缓存域名（不落盘 Token）`<1s` 直开；次选 Colab Contents REST 直读 `~1.5–2.5s`；仅失效时触发远端隧道重建。
  - 内置 **180 秒单次瞬断宽限**与只读锁快照读取，避免代理偶发抖动引起界面频繁闪烁告警。

### 6.2 Colab Studio 双窗工作台（[colab_pet/studio.html](colab_pet/studio.html)）

在桌宠快捷操作中可一键展开独立无边框 **Colab Studio** 工作台窗口（支持浅色 / 深色主题无闪烁切换，关闭工作台时桌宠继续后台运行监控）：

1. **算力与集群**：展示全池所有绑定账号的总 CU 余额、健康度与活跃会话卡片，支持单号部署与双号异构集群编排、分槽单独停卡。
2. **作品画廊**：扫描本机 `_downloads/` 输出目录（支持图片/视频/音频，音频与视频支持 HTTP Range 206 流式拖动播放），内置 **ComfyUI PNG 元数据/工作流解析 Inspector**，支持一键将选中图片的工作流推送直开云端 ComfyUI 画布（配合 [colab_pet_direct_canvas.js](colab_pet_direct_canvas.js)）。
3. **引擎与云盘**：支持 Qwen3.8 在线热调参（思考档位/上下文参数动态重载）及 Google Drive 云盘资产（LoRA / 工作流 / 参考音频）浏览与同步。
4. **控制台与体检**：切换查看本地诊断日志、部署实时流与远端服务日志，内置一键 `doctor` 体检与脱敏 `.zip` 诊断包导出。

### 6.3 多账号算力池与异构双开集群

由纯函数调度核 [colab_core/pool_scheduler.py](colab_core/pool_scheduler.py) 与后端编排服务 [colab_pet/backend_cluster.py](colab_pet/backend_cluster.py) 协作实现：

- **多账号安全隔离托管**：多个 Google 算力账号的 OAuth 令牌分别隔离保存在 Windows 系统凭据库，按账号独立维护会话备份与每日 CU 消费账本，互不串号。
- **并发算力池探针与容灾换号**：并发探测各账号余额与层级，优先挑选高性价比空闲账号；遇到某账号目标 GPU 缺货/配额受限时自动标记临时紧缺并平滑故障转移换号。
- **异构双开（Split Cluster）**：支持将 **LLM 对话网关（如 `qwen_27b` / `qwen_llm`）** 与 **ComfyUI 画布（如 `anima` / `minimax` / `drama3`）** 自动拆分到两个不同 Google 账号的独立 GPU 槽位并发运行，各自拥有独立的公网鉴权隧道、消费监控、分槽定向停卡（Fail-Closed 防误停其他槽位）与闲置自动停卡保护。

### 6.4 内置 AI 副驾（Copilot）与分层安全护栏

在 Studio 右上角点击 **「AI 副驾」** 或按快捷键 `Ctrl+K` 呼出侧边抽屉，底层由 [colab_core/agent/](colab_core/agent/__init__.py)（经 [colab_core/agent_tools.py](colab_core/agent_tools.py) 兼容门面导出）与 [colab_pet/backend_agent.py](colab_pet/backend_agent.py) 驱动：

- **双引擎与四协议路由**：
  - **内置轻量引擎（`NativeLightDriver`）**：零额外重依赖，原生支持 OpenAI Chat Completions、OpenAI Responses、Anthropic Messages 与 Google Gemini 四大协议（含思考强度档位映射与模型能力自动探测），并可自动路由至当前在线的云端 Colab Qwen3.8 网关或外部 API 供应商；
  - **外部引擎（`ClaudeCodeDriver`）**：可选调用本机 `claude` CLI，通过临时 `--mcp-config` 挂载本机 `colab_ctl.py mcp-stdio` 10 工具集。
- **凭据源站硬绑定与全链路脱敏**：AI 供应商 API Key 仅存系统凭据库并与目标 `Origin` 硬绑定（修改 Base URL 自动失效旧 Key），跨源 HTTP 302 重定向严禁携带凭据；多会话历史持久于 `agent_sessions.json` 前强制全链路脱敏。
- **三级工具护栏与人工审批**：
  - **只读诊断（`safe` + 白名单只读命令）**：查询集群状态、余额、日志、工作流与只读 `nvidia-smi` / `df` / `ps` 等自动放行；
  - **高危硬拦截**：破坏性命令与读取 `/tmp/rclone.conf`、凭据/密钥文件的尝试直接拦截，不提供审批入口；
  - **人工审批确认（`confirm`）**：开关/启停计费 GPU 实例（`manage_compute_instance`）及所有非只读远端 Bash/Python 代码执行（`exec_remote_colab`）均会挂起并弹出交互式审批卡，明文展示完整命令与目标槽位，经人工批准后方可执行。

---

## 七、模块化共享内核架构、可复现基线与 Windows 打包

### 7.1 模块职责拆分（CLI 与桌宠复用同一套核心逻辑）

- **共享业务内核 [colab_core/](colab_core/__init__.py)（13 个顶层模块 + 7 文件 `agent/` 子包）**：
  - **基础配置与事实来源**：
    - [colab_core/accounts.py](colab_core/accounts.py)：跨设备本机工具路径发现、多账号凭据保险箱辅助、rclone `[gdrive_save]` 绑定校验与固定隧道配置引用读写；
    - [colab_core/catalog.py](colab_core/catalog.py)：显卡默认估算费率、硬件规格与工作流目录单一事实来源（SSOT，只读缓存加载 [stack_manifest.json](stack_manifest.json)，绝不覆写账户实测费率 `measured_gpu_rates`）；
    - [colab_core/reasoning_specs.py](colab_core/reasoning_specs.py)：数据驱动 AI 模型思考强度枚举规范化与模型家族推理档位推断规则表。
  - **云端交互、同步、调度与体检**：
    - [colab_core/colab_client.py](colab_core/colab_client.py)：Colab 会话探针、远端脚本执行（自动伴随携带 [colab_remote_common.py](colab_remote_common.py)）、孤儿实例纳管、定向停卡与释放核实、日志隧道 URL 提取与凭据发送源站同源校验；
    - [colab_core/file_sync.py](colab_core/file_sync.py)：云盘强校验同步（`<<<SYNC_VERIFIED>>>`）、时间戳防覆盖安全解压与断点恢复、ComfyUI PNG 元数据与 API 工作流转换；
    - [colab_core/health.py](colab_core/health.py)：[stack_manifest.json](stack_manifest.json) 与桥接源码一致性校验（`verify_manifest_and_assets`）、公开基础版/个人增强版多维度启动前体检（`run_preflight_checks`）与一键全量脱敏 `.zip` 诊断包导出；
    - [colab_core/pool_scheduler.py](colab_core/pool_scheduler.py)：零 I/O 多账号算力池 CU 估算、节点评分、库存紧缺容灾换号与单/双号异构集群拆分规划纯函数核。
  - **运行时解聚内核（4 个正交模块 + 兼容门面）**：
    - [colab_core/exec_runner.py](colab_core/exec_runner.py)：统一同步/流式子进程执行器（`ProcessRunner`）、取消令牌（`CancelToken`）、后台任务台账（`TaskManager`）、512 KiB 轮转脱敏日志器（`DiagLogger`，含 `debug_throttled` 节流）及异常降级日志辅助（`suppress_logged`）；
    - [colab_core/deploy_state.py](colab_core/deploy_state.py)：工作流生态规划、显存防呆显卡推荐、7 步部署流式日志解析状态机与多行堆栈故障分类；
    - [colab_core/ledger.py](colab_core/ledger.py)：多账号独立每日 CU 账本、账号层级推断、状态文件安全清洗/损坏备份与 ComfyUI/LLM 遥测解析；
    - [colab_core/runtime.py](colab_core/runtime.py)：向后兼容重导出门面，汇总导出 `exec_runner`、`deploy_state`、`ledger` 与 `reasoning_specs` 全部公开符号。
  - **AI 副驾模块化子包 [colab_core/agent/](colab_core/agent/__init__.py)（7 文件 + 兼容门面）**：
    - [colab_core/agent/specs.py](colab_core/agent/specs.py)（10 工具 JSON Schema 与系统提示词）、[colab_core/agent/safety.py](colab_core/agent/safety.py)（共享执行边界安全护栏与严格只读诊断白名单）、[colab_core/agent/protocols.py](colab_core/agent/protocols.py)（OpenAI Chat / OpenAI Responses / Anthropic / Gemini 四协议适配器）、[colab_core/agent/provider_probe.py](colab_core/agent/provider_probe.py)（模型列表拉取与四协议自动探测）、[colab_core/agent/drivers.py](colab_core/agent/drivers.py)（`NativeLightDriver` 与 `ClaudeCodeDriver` 双引擎）、[colab_core/agent/mcp_server.py](colab_core/agent/mcp_server.py)（JSON-RPC 2.0 MCP Stdio 服务，无头默认 Fail-Closed 只读）及 [colab_core/agent/__init__.py](colab_core/agent/__init__.py)；
    - [colab_core/agent_tools.py](colab_core/agent_tools.py)：向后兼容重导出门面，保持存量导入路径 100% 兼容。
- **共享安全底座与本机桥接**：
  - [colab_security.py](colab_security.py)：Windows 原生 `advapi32` 凭据管理器接入（及测试隔离沙箱 `COLAB_PET_MOCK_VAULT_PATH`）、公网一次性跳转票据、开放重定向防护与全局密钥脱敏器 `redact_secrets`；
  - [colab_cli_bridge.py](colab_cli_bridge.py)：运行在本机 `google-colab-cli` Python 环境中的统一桥接助手（会话读写、Contents REST 直读、多账号凭据库管理与算力池并发探测）。
- **桌面双窗应用 [colab_pet/](colab_pet/__init__.py)（4 桌面宿主模块 + 3 核心容器 + 协调门面 + 8 正交子服务）**：
  - **4 个桌面宿主模块**：[colab_pet/pet_app.py](colab_pet/pet_app.py)（CLI 入口与双窗生命周期装配）、[colab_pet/bridge_api.py](colab_pet/bridge_api.py)（双窗 JS Bridge 显式 API 门面与跨窗事件推送）、[colab_pet/asset_server.py](colab_pet/asset_server.py)（纯标准库本地只读 HTTP 资产服务，含 `/assets/` 白名单与 `/__media/` Range 206 流式播放）、[colab_pet/win32_window.py](colab_pet/win32_window.py)（Win32 原生窗口控制、单实例唤醒、Alpha/圆角 Region 裁剪与零 Pillow GDI+ 截图）；
  - **3 个核心状态与依赖容器**：[colab_pet/state_store.py](colab_pet/state_store.py)（`AppStateStore` 独占持久化状态、4 级全局锁偏序与原子落盘）、[colab_pet/secret_store.py](colab_pet/secret_store.py)（`EphemeralSecretStore` 纯内存元数据与同源同号临时密钥隔离）、[colab_pet/context.py](colab_pet/context.py)（`EventBus` 脱敏异常隔离多播总线与 `BackendContext` 显式依赖注入上下文）；
  - **后端协调门面与 8 个正交业务子服务**：[colab_pet/colab_backend.py](colab_pet/colab_backend.py)（薄协调门面），下辖 [backend_accounts.py](colab_pet/backend_accounts.py)、[backend_agent.py](colab_pet/backend_agent.py)、[backend_cluster.py](colab_pet/backend_cluster.py)、[backend_comfy.py](colab_pet/backend_comfy.py)、[backend_deploy.py](colab_pet/backend_deploy.py)、[backend_monitor.py](colab_pet/backend_monitor.py)、[backend_studio.py](colab_pet/backend_studio.py)、[backend_tunnel.py](colab_pet/backend_tunnel.py)；
  - **前端双窗视图**：[colab_pet/ui.html](colab_pet/ui.html)（桌宠挂件）与 [colab_pet/studio.html](colab_pet/studio.html) / [colab_pet/studio.js](colab_pet/studio.js) / [colab_pet/studio.css](colab_pet/studio.css)（Studio 工作台）。

### 7.2 依赖锁定、技术栈清单、前置自检与静态质量门禁

- **依赖文件分工**：
  - [requirements.txt](requirements.txt)：精简核心运行依赖（零 `Pillow` / `aiohttp` / `pywin32`）；
  - [requirements-lock.txt](requirements-lock.txt)：已验证精确锁版本运行基线；
  - [requirements-dev.txt](requirements-dev.txt)：离线测试（`pytest`）、Mock 网关测试（`aiohttp`）、静态资产离线预处理（`pillow`）与静态代码质量门禁（`ruff`）依赖。
- **技术栈清单**：[stack_manifest.json](stack_manifest.json) 默认主线为 `anima`（`L4`），并统一记录 GPU 规格/默认估算费率及 `minimax`、`drama3`、`qwen_27b`、`qwen_llm` 的模型体积门槛与引擎检出变量（`COMFY_GIT_REF`、`DRAMA3_NODE_GIT_REF`、`LLAMA_CPP_TAG`）。
- **自动化测试隔离与静态质量门禁**：
  - [tests/conftest.py](tests/conftest.py) 为全部 `pytest` 用例提供全局默认沙箱隔离（自动绑定临时 `COLAB_PET_MOCK_VAULT_PATH` 与诊断日志路径，绝不触碰宿主机真实 Windows 凭据库），配合 [pytest.ini](pytest.ini) 精确区分非 GUI 离线回归（`-m "not gui"`）与真机桌面视觉回归（`-m gui`）；
  - [ruff.toml](ruff.toml) 启用语法、潜在缺陷（`B`）、裸 `except-pass`（`S110`）、现代语法（`UP`）与 McCabe 圈复杂度上限（`C901 ≤ 25`）静态质量门禁，并已接入 GitHub Actions CI。
- **前置体检、静态门禁与诊断导出**：
  ```powershell
  # 深度环境体检（默认公开基础版，缺 [gdrive_save] 仅报 warn；支持 --edition personal 或 COLAB_PET_EDITION=personal 触发个人增强版强校验）
  python colab_ctl.py doctor --profile anima --gpu L4
  python colab_ctl.py doctor --profile anima --gpu L4 --edition personal

  # 静态代码质量与圈复杂度门禁检查（CI 同款）
  python -m ruff check --config "ruff.toml" .

  # 导出全量脱敏的排障诊断包 (.zip)，绝不包含任何明文密钥或个人会话凭据
  python colab_ctl.py diag
  ```

### 7.3 Windows 发行打包与安全合规闸门

运行打包构建脚本 [build_windows_package.py](build_windows_package.py)（配合 [colab_pet.spec](colab_pet.spec)）：

```powershell
# 方式 A：构建便携桌面发行包（编译无黑框原生 Colab桌宠.exe + 一键环境自检 + 安全闸门校验）
python build_windows_package.py --mode portable

# 方式 B：配合 PyInstaller 构建免 Python 独立目录包
python build_windows_package.py --mode pyinstaller
```

- **构建安全闸门**：自动扫描并阻断 `.cf_tunnel_secret.json`、`pet_state.json`、`agent_sessions.json`、日志及测试临时文件混入发行包；
- **立绘使用边界**：随附「夏儿」Q 版 `.webp` 与 `.ico` 资产仅限非商业个人展示与测试，商业场景或独立二次分发请替换为自有版权素材或删除 `.webp` 启用矢量兜底，详见 [colab_pet/ASSETS_NOTICE.txt](colab_pet/ASSETS_NOTICE.txt)。
