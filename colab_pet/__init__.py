"""Colab 桌宠前端与本地后端服务包。

本包入口严格保持轻量设计，禁止在包初始化阶段急切导入 `pet_app`、`clr`、`webview`
或重型后端模块，确保无头环境、单元测试与 CLI 子进程执行 `import colab_pet` 时安全无副作用。
"""

from __future__ import annotations

__version__ = "3.0.0"

__all__ = ["__version__"]
