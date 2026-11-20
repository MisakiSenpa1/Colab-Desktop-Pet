"""全局 pytest 测试隔离夹具。

说明：
- 为全仓所有测试用例提供默认沙箱隔离，确保未显式声明局部隔离夹具的测试也绝不读写
  宿主机真实 Windows 凭据库或诊断日志；
- 不强制设置 `COLAB_PET_CF_CONFIG_PATH`，避免影响校验默认 Cloudflare 配置文件路径的用例；
- 单个测试仍可随时通过自身的 `monkeypatch.setenv` / `monkeypatch.delenv` 覆盖或清除变量。
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate_host_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """默认启用 UTF-8 与临时凭据库/诊断日志沙箱，防止测试污染宿主机真实运行态。"""
    monkeypatch.setenv("PYTHONUTF8", "1")
    monkeypatch.setenv("COLAB_PET_MOCK_VAULT_PATH", str(tmp_path / "_conftest_mock_vault.json"))
    monkeypatch.setenv("COLAB_PET_DIAG_LOG_PATH", str(tmp_path / "_conftest_diag.log"))
