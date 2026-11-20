# -*- mode: python ; coding: utf-8 -*-
"""Colab 桌宠 PyInstaller Windows 单目录打包规格文件 (onedir, 无控制台窗口)。

用法示例：
    python build_windows_package.py --mode pyinstaller
或手动执行：
    pyinstaller --noconfirm --clean colab_pet.spec
"""

import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

SPEC_DIR = Path(SPECPATH).resolve()  # noqa: F821
if str(SPEC_DIR) not in sys.path:
    sys.path.insert(0, str(SPEC_DIR))

datas = [
    (str(SPEC_DIR / "colab_pet" / "ui.html"), "colab_pet"),
    (str(SPEC_DIR / "colab_pet" / "studio.html"), "colab_pet"),
    (str(SPEC_DIR / "colab_pet" / "studio.css"), "colab_pet"),
    (str(SPEC_DIR / "colab_pet" / "studio.js"), "colab_pet"),
    (str(SPEC_DIR / "colab_pet" / "ASSETS_NOTICE.txt"), "colab_pet"),
    (str(SPEC_DIR / "colab_pet" / "assets"), "colab_pet/assets"),
    (str(SPEC_DIR / "stack_manifest.json"), "."),
    (str(SPEC_DIR / "requirements.txt"), "."),
    (str(SPEC_DIR / "requirements-lock.txt"), "."),
    (str(SPEC_DIR / "colab_ctl.py"), "."),
    (str(SPEC_DIR / "colab_cli_bridge.py"), "."),
    (str(SPEC_DIR / "colab_remote_common.py"), "."),
    (str(SPEC_DIR / "colab_security.py"), "."),
    (str(SPEC_DIR / "colab_pet_bridge_node.py"), "."),
    (str(SPEC_DIR / "colab_pet_direct_canvas.js"), "."),
    (str(SPEC_DIR / "colab_sse_gateway.py"), "."),
    (str(SPEC_DIR / "colab_bootstrap_anima.py"), "."),
    (str(SPEC_DIR / "colab_bootstrap_minimax_h3.py"), "."),
    (str(SPEC_DIR / "colab_bootstrap_drama3.py"), "."),
    (str(SPEC_DIR / "colab_bootstrap_qwen_llm.py"), "."),
    (str(SPEC_DIR / "verify_smoke.py"), "."),
    (str(SPEC_DIR / "verify_minimax_h3.py"), "."),
    (str(SPEC_DIR / "verify_drama3.py"), "."),
    (str(SPEC_DIR / "verify_qwen_llm.py"), "."),
]
wf_dir = SPEC_DIR / "workflows"
if wf_dir.is_dir():
    datas.append((str(wf_dir), "workflows"))

for pkg in ("webview", "pythonnet", "clr_loader"):
    try:
        datas.extend(collect_data_files(pkg))
    except Exception:
        pass

hiddenimports = [
    "colab_ctl",
    "colab_security",
    "colab_backend",
    "state_store",
    "secret_store",
    "context",
    "win32_window",
    "asset_server",
    "bridge_api",
    "backend_monitor",
    "backend_deploy",
    "backend_tunnel",
    "backend_comfy",
    "backend_accounts",
    "backend_cluster",
    "backend_studio",
    "backend_agent",
    "requests",
    "websocket",
]
for pkg in ("colab_core", "webview", "clr_loader", "pythonnet"):
    try:
        hiddenimports.extend(collect_submodules(pkg))
    except Exception:
        pass

icon_file = SPEC_DIR / "build" / "pet.ico"

a = Analysis(  # noqa: F821
    [str(SPEC_DIR / "colab_pet" / "pet_app.py")],
    pathex=[str(SPEC_DIR), str(SPEC_DIR / "colab_pet")],
    binaries=[],
    datas=datas,
    hiddenimports=sorted(set(hiddenimports)),
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "PIL",
        "pytest",
        "_pytest",
        "tkinter",
        "aiohttp",
        "pywin32",
        "win32api",
        "win32con",
        "win32com",
        "pythoncom",
        "pywintypes",
        "cryptography",
        "cffi",
        "charset_normalizer",
        "unittest",
        "pydoc",
        "doctest",
        "sqlite3",
        "multiprocessing",
        "webview.platforms.android",
        "webview.platforms.cocoa",
        "webview.platforms.gtk",
        "webview.platforms.qt",
        "webview.platforms.cef",
        "webview.platforms.mshtml",
    ],
    noarchive=False,
)

# 剔除 Android JAR、ARM64/x86 非目标架构二进制与 .pdb 调试符号（保留 colab_pet/assets 与 workflows）
_PRUNE_MARKERS = (
    "pywebview-android.jar",
    "win-arm64",
    "win-x86",
    ".x86.dll",
    "/x86/",
    "\\x86\\",
)


def _should_prune_toc_entry(entry: tuple[str, str, str]) -> bool:
    """判定 TOC 条目是否属于需要裁剪的跨平台、非 x64 架构或调试符号文件。"""
    for raw_path in entry[:2]:
        low = str(raw_path or "").lower()
        norm = low.replace("\\", "/")
        if norm.endswith(".pdb"):
            return True
        if any(marker in low or marker in norm for marker in _PRUNE_MARKERS):
            return True
    return False


a.binaries = [item for item in a.binaries if not _should_prune_toc_entry(item)]
a.datas = [item for item in a.datas if not _should_prune_toc_entry(item)]
pyz = PYZ(a.pure)  # noqa: F821

exe = EXE(  # noqa: F821
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Colab桌宠",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    icon=str(icon_file) if icon_file.is_file() else None,
)

coll = COLLECT(  # noqa: F821
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="Colab桌宠",
)
