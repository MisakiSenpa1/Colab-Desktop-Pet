"""colab_core.agent.specs —— AI 副驾标准化 10 工具协议清单、分级策略与默认系统提示词。"""

from __future__ import annotations

from typing import Any

# =========================================================================
# 1. 标准化 10 工具协议清单与分级策略
# =========================================================================

AGENT_TOOL_SPECS: tuple[dict[str, Any], ...] = (
    {
        "name": "get_cluster_and_pool_status",
        "description": "获取当前多账号算力池余额、活跃 Colab 实例/分槽状态、GPU 显存、隧道在线状态与实时消费速率。",
        "tier": "safe",
        "parameters": {
            "type": "object",
            "properties": {
                "refresh_pool": {
                    "type": "boolean",
                    "description": "是否实时并发探测刷新全部账号余额与会话（约耗时 2~5 秒），默认 false 读取缓存快照",
                    "default": False,
                },
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "tail_instance_logs",
        "description": "读取本机诊断日志或远端 llama-server / ComfyUI 实时日志尾部内容（自动脱敏密钥）。",
        "tier": "safe",
        "parameters": {
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "enum": ["local_diag", "llama", "comfy"],
                    "description": "日志来源：local_diag（本机诊断日志）、llama（远端 LLM 推理日志）、comfy（远端 ComfyUI 日志）",
                    "default": "local_diag",
                },
                "lines": {
                    "type": "integer",
                    "minimum": 10,
                    "maximum": 500,
                    "description": "返回日志尾部行数，默认 100",
                    "default": 100,
                },
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "list_gallery_and_inspect",
        "description": "列出本地已拉取的作品画廊，或在传入 rel_path 时解析指定 PNG 作品的 ComfyUI 工作流、提示词、LoRA 与采样参数。",
        "tier": "safe",
        "parameters": {
            "type": "object",
            "properties": {
                "rel_path": {
                    "type": "string",
                    "description": "可选：待解析作品的相对路径；留空则返回画廊最新条目列表",
                    "default": "",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "description": "列表最大返回条目数，默认 20",
                    "default": 20,
                },
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "list_gdrive_assets",
        "description": "查询 Google Drive (gdrive_save) 云端已备份的 LoRA、工作流模板与 Drama3 音频资产清单。",
        "tier": "safe",
        "parameters": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
    {
        "name": "run_environment_doctor",
        "description": "运行本地工具链与配置环境深度体检（colab CLI、rclone、隧道配置与磁盘空间检查）。",
        "tier": "safe",
        "parameters": {
            "type": "object",
            "properties": {
                "profile": {
                    "type": "string",
                    "enum": ["anima", "minimax", "drama3", "qwen_27b", "qwen_llm"],
                    "description": "可选：目标工作流生态，默认 anima",
                    "default": "anima",
                },
                "gpu": {
                    "type": "string",
                    "enum": ["T4", "L4", "G4", "A100", "H100"],
                    "description": "可选：预期目标 GPU 卡型",
                },
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "exec_remote_colab",
        "description": "在在线 Colab 远端实例中执行受安全护栏保护的 Bash 命令或 Python 代码片段（适合显存/磁盘/进程排查）。",
        "tier": "exec",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "要在远端 Colab 实例执行的 Bash 命令或 Python 代码",
                },
                "mode": {
                    "type": "string",
                    "enum": ["bash", "python"],
                    "description": "执行模式：bash 或 python，默认 bash",
                    "default": "bash",
                },
                "slot": {
                    "type": "string",
                    "enum": ["auto", "comfy", "llm"],
                    "description": "集群多槽目标：auto（自动选择）、comfy 或 llm",
                    "default": "auto",
                },
                "timeout_sec": {
                    "type": "integer",
                    "minimum": 5,
                    "maximum": 600,
                    "description": "执行超时秒数，默认 180",
                    "default": 180,
                },
            },
            "required": ["code"],
            "additionalProperties": False,
        },
    },
    {
        "name": "queue_comfy_workflow",
        "description": "基于指定画廊作品（或最新作品/模板）覆写正反向提示词、种子、步数或出图数量并投递至在线 ComfyUI /prompt 队列。",
        "tier": "exec",
        "parameters": {
            "type": "object",
            "properties": {
                "base_rel_path": {
                    "type": "string",
                    "description": "基础工作流作品的 rel_path（从 list_gallery_and_inspect 获取）；留空则自动挑选最新作品",
                    "default": "",
                },
                "positive_prompt": {
                    "type": "string",
                    "description": "可选：覆写的正向提示词，留空保留原词",
                    "default": "",
                },
                "negative_prompt": {
                    "type": "string",
                    "description": "可选：覆写的负向提示词，留空保留原词",
                    "default": "",
                },
                "seed": {
                    "type": "integer",
                    "description": "可选：采样随机种子（不传则保留原值或使用随机种子）",
                },
                "steps": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 150,
                    "description": "可选：采样步数",
                },
                "batch_size": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 8,
                    "description": "出图数量，默认 1",
                    "default": 1,
                },
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "sync_and_pull_outputs",
        "description": "将远端 ComfyUI 生成作品归档同步至 Google Drive 并拉回本地下载目录。",
        "tier": "exec",
        "parameters": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
    {
        "name": "reconfigure_llm_engine",
        "description": "在线热调参正在运行的 Colab Qwen3.8 LLM 推理引擎（上下文窗口 ctx、量化精度 quant 或视觉投影 mmproj）。",
        "tier": "exec",
        "parameters": {
            "type": "object",
            "properties": {
                "ctx": {
                    "type": "integer",
                    "minimum": 4096,
                    "maximum": 262144,
                    "description": "可选：上下文长度（如 32768、65536、131072）",
                },
                "quant": {
                    "type": "string",
                    "enum": ["UD-Q4_K_XL", "UD-IQ4_XS", "IQ3_XXS"],
                    "description": "可选：GGUF 量化等级",
                },
                "mmproj": {
                    "type": "boolean",
                    "description": "可选：是否启用多模态视觉投影 (mmproj)",
                },
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "manage_compute_instance",
        "description": "开机部署或停卡释放 Colab GPU 实例（涉及 CU 计费与算力中断，需用户审批或显式自动放行策略）。",
        "tier": "confirm",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["start", "stop_slot", "stop_all"],
                    "description": "操作类型：start（开机部署）、stop_slot（停止指定分槽）、stop_all（停止全部计费实例）",
                    "default": "start",
                },
                "profiles": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["anima", "minimax", "drama3", "qwen_27b", "qwen_llm"],
                    },
                    "description": "action=start 时要部署的生态列表，例如 [\"anima\"] 或 [\"minimax\", \"qwen_27b\"]",
                    "default": [],
                },
                "gpu": {
                    "type": "string",
                    "enum": ["T4", "L4", "G4", "A100", "H100"],
                    "description": "可选：action=start 时指定的 GPU 卡型",
                },
                "cluster_mode": {
                    "type": "string",
                    "enum": ["auto", "single", "split"],
                    "description": "集群模式：auto（自动）、single（单号）、split（异构双号拆分），默认 auto",
                    "default": "auto",
                },
                "slot": {
                    "type": "string",
                    "enum": ["comfy", "llm", "all"],
                    "description": "action=stop_slot 时要关闭的目标分槽/会话名，默认 comfy",
                    "default": "comfy",
                },
            },
            "required": ["action"],
            "additionalProperties": False,
        },
    },
)

AGENT_TOOL_MAP: dict[str, dict[str, Any]] = {spec["name"]: spec for spec in AGENT_TOOL_SPECS}

DEFAULT_SYSTEM_PROMPT = (
    "你是 Colab Studio 同进程 AI 副驾，协助管理 Google Colab GPU 算力池、多账号调度、"
    "ComfyUI 出图/视频工作流、Google Drive 资产及 Qwen3.8 LLM 推理服务。\n"
    "工作准则：\n"
    "1. 先核实再操作：不了解集群、日志或作品参数时，优先调用只读 (safe) 工具查明实情，禁止编造状态。\n"
    "2. 改词重绘规范：先调 list_gallery_and_inspect 获取作品 rel_path 与工作流参数，再调 queue_comfy_workflow 精准覆写提示词或步数。\n"
    "3. 成本与安全边界：调用 manage_compute_instance 开关卡会产生 CU 计费或中断任务，调用前须简明说明目的与影响；严禁输出或探测任何凭据密钥。\n"
    "4. 表达简洁准确：使用简体中文，直接给出关键结论、异常根因与可执行建议。"
)


def get_tool_spec(name: str) -> dict[str, Any] | None:
    """按工具名称查找标准协议定义。"""
    return AGENT_TOOL_MAP.get(str(name or "").strip())


__all__ = [
    "AGENT_TOOL_MAP",
    "AGENT_TOOL_SPECS",
    "DEFAULT_SYSTEM_PROMPT",
    "get_tool_spec",
]
