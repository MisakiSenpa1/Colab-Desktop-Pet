"""Colab 桌宠无黑框双点击启动器 (.pyw)。"""

from __future__ import annotations

import ctypes
from datetime import datetime
import os
from pathlib import Path
import sys
import traceback
from typing import TextIO

os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

# 持有兜底空流句柄，避免 pythonw.exe 无控制台模式下句柄被 GC 回收关闭
_DEVNULL_HANDLES: list[TextIO] = []


def _ensure_std_streams() -> None:
    """在 pythonw.exe 无控制台导致 sys.stdout/sys.stderr 为 None 时提供安全兜底。"""
    for attr in ("stdout", "stderr"):
        if getattr(sys, attr, None) is None:
            handle = open(os.devnull, "w", encoding="utf-8", errors="replace")
            _DEVNULL_HANDLES.append(handle)
            setattr(sys, attr, handle)


_ensure_std_streams()

PKG_DIR = Path(__file__).resolve().parent
PET_DIR = PKG_DIR / "colab_pet"
ERROR_LOG_PATH = PET_DIR / "pet_error.log"
MAX_LOG_BYTES = 256 * 1024
KEEP_LOG_BYTES = 128 * 1024

for _p in (str(PKG_DIR), str(PET_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _redact_safe(text: str) -> str:
    """尽最大努力对异常文本执行凭据脱敏。"""
    try:
        from colab_security import redact_secrets

        return redact_secrets(text)
    except Exception:
        return str(text or "")


def _write_error_log(trace_text: str) -> None:
    """将脱敏后的启动异常堆栈写入日志并在超过 256 KB 时截断保留最近 128 KB。"""
    clean_trace = _redact_safe(trace_text)
    try:
        from colab_core.runtime import get_diag_logger

        get_diag_logger().error("launcher", f"桌宠启动异常: {clean_trace.splitlines()[-1] if clean_trace else ''}")
    except Exception:
        pass
    try:
        ERROR_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        existing = b""
        if ERROR_LOG_PATH.exists():
            raw = ERROR_LOG_PATH.read_bytes()
            if len(raw) > MAX_LOG_BYTES:
                # 若原日志超出上限，则截断到末尾 128 KB 后从完整行首开始保留
                tail = raw[-KEEP_LOG_BYTES:]
                nl = tail.find(b"\n")
                if nl != -1:
                    tail = tail[nl + 1 :]
                existing = "[...日志超限已自动截断...]\n".encode("utf-8") + tail
            else:
                existing = raw

        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = (
            f"\n=== {stamp} 启动异常 ===\n{clean_trace.rstrip()}\n"
        ).encode("utf-8", errors="replace")
        ERROR_LOG_PATH.write_bytes(existing + entry)
    except Exception:
        pass


def _show_error_box(exc: Exception) -> None:
    """通过原生 Win32 MessageBoxW 提示脱敏后的启动失败详情与日志路径。"""
    try:
        msg = (
            f"Colab 桌宠启动时发生异常：\n{_redact_safe(str(exc))}\n\n"
            f"详细堆栈已写入日志：\n{ERROR_LOG_PATH}"
        )
        ctypes.windll.user32.MessageBoxW(0, msg, "Colab 桌宠启动失败", 0x10)
    except Exception:
        pass


if __name__ == "__main__":
    try:
        from pet_app import main  # noqa: E402

        exit_code = main()
    except Exception as exc:
        _write_error_log(traceback.format_exc())
        _show_error_box(exc)
        raise SystemExit(1)
    raise SystemExit(exit_code)
