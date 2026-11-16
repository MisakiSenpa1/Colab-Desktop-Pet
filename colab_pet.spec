# -*- mode: python ; coding: utf-8 -*-
"""Colab 桌宠 PyInstaller Windows 单目录打包规格文件 (onedir, 无控制台窗口)。

用法示例：
    python build_windows_package.py --mode pyinstaller
或手动执行：
    pyinstaller --noconfirm --clean colab_pet.spec
"""

from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

SPEC_DIR = Path(SPECPATH).resolve()

datas = [
    (str(SPEC_DIR / "colab_pet" / "ui.html"), "colab_pet"),
    (str(SPEC_DIR / "colab_pet" / "ASSETS_NOTICE.txt"), "colab_pet"),
    (str(SPEC_DIR / "colab_pet" / "assets"), "colab_pet/assets"),
    (str(SPEC_DIR / "stack_manifest.json"), "."),
    (str(SPEC_DIR / "requirements.txt"), "."),
    (str(SPEC_DIR / "requirements-lock.txt"), "."),
    (str(SPEC_DIR / "colab_cli_bridge.py"), "."),
    (str(SPEC_DIR / "colab_remote_common.py"), "."),
    (str(SPEC_DIR / "colab_pet_bridge_node.py"), "."),
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
    "requests",
    "PIL",
    "websocket",
]
for pkg in ("colab_core", "webview", "clr_loader", "pythonnet"):
    try:
        hiddenimports.extend(collect_submodules(pkg))
    except Exception:
        pass

icon_file = SPEC_DIR / "build" / "pet.ico"

a = Analysis(
    [str(SPEC_DIR / "colab_pet" / "pet_app.py")],
    pathex=[str(SPEC_DIR), str(SPEC_DIR / "colab_pet")],
    binaries=[],
    datas=datas,
    hiddenimports=sorted(set(hiddenimports)),
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["pytest", "_pytest", "tkinter", "aiohttp"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
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

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="Colab桌宠",
)
