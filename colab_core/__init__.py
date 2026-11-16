"""colab_core —— Colab 总控脚手架与桌面挂件共享业务内核（5 模块精简架构）。

模块职责边界：
- accounts     : 工具路径发现、多账号保险箱辅助、rclone 绑定与固定隧道配置；
- colab_client : 会话探针、远端脚本执行（含 colab_remote_common 自动携带）、停卡核实与隧道探测；
- runtime      : 部署调度/7步状态机、统一子进程执行与轮转脱敏日志、多账号每日账本；
- file_sync    : 云盘强校验同步、时间戳防覆盖归档、模型/工作流转换；
- health       : 技术栈清单一致性校验、启动前环境自检 (Preflight) 与脱敏诊断包导出。
"""

from __future__ import annotations

__all__ = [
    "accounts",
    "colab_client",
    "file_sync",
    "health",
    "runtime",
]
