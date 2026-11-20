"""colab_core —— Colab 总控脚手架与桌面挂件共享业务内核。

模块职责边界：
- accounts        : 工具路径发现、多账号保险箱辅助、rclone 绑定与固定隧道配置；
- agent           : 同进程 AI 副驾模块化内核子包（规格、安全、协议、探测、驱动与 MCP）；
- agent_tools     : AI 副驾向后兼容门面；
- catalog         : 显卡默认费率、硬件规格与工作流目录单一事实来源（SSOT）；
- colab_client    : 会话探针、远端脚本执行（含 colab_remote_common 自动携带）、停卡核实与隧道探测；
- deploy_state    : 工作流生态规范、显卡防呆推荐、7 步部署流式解析与故障分类；
- exec_runner     : 统一同步/流式子进程执行器、取消令牌、后台任务台账与脱敏轮转日志；
- file_sync       : 云盘强校验同步、时间戳防覆盖归档、模型/工作流转换；
- health          : 技术栈清单一致性校验、启动前环境自检 (Preflight) 与脱敏诊断包导出；
- ledger          : 多账号每日账本、偏好与状态安全清洗及 ComfyUI/LLM 遥测解析；
- pool_scheduler  : 多账号算力池 CU 估算、节点评分与单/双号异构聚类部署规划；
- reasoning_specs : AI 模型思考强度规范化与数据驱动模型家族能力推断；
- runtime         : 运行时向后兼容门面（透传 reasoning_specs / exec_runner / deploy_state / ledger）。
"""

from __future__ import annotations

__all__ = [
    "accounts",
    "agent",
    "agent_tools",
    "catalog",
    "colab_client",
    "deploy_state",
    "exec_runner",
    "file_sync",
    "health",
    "ledger",
    "pool_scheduler",
    "reasoning_specs",
    "runtime",
]
