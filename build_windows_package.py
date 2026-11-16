"""build_windows_package.py —— Colab 桌宠面向普通用户的 Windows 打包与安全发行构建脚本。

支持两种构建模式：
1. `--mode portable`（默认，零额外打包依赖）：
   - 从 `colab_pet/assets/idle.png` 生成多尺寸图标 `pet.ico`；
   - 使用 Windows 系统自带 .NET C# 编译器 (`csc.exe`) 编译原生无黑框 GUI 启动器 `Colab桌宠.exe`
     （支持 `--self-test`、`--doctor`、`--export-diag` 与自动探测本机 Python / 内嵌运行时）；
   - 生成 `一键环境自检与安装.bat`、`导出排障诊断包.bat`、`Windows打包版使用说明.txt`；
   - 经由严格白名单复制源码与资产 + 敏感密钥安全闸门全量扫描后，产出独立 `.zip` 发布包。
2. `--mode pyinstaller`：
   - 调用 `colab_pet.spec` 打出免安装 Python 的独立可执行目录与 `.zip` 发行包。
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Iterable

from PIL import Image

PKG_DIR = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = PKG_DIR / "dist"
CSC_CANDIDATES = (
    Path(r"C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe"),
    Path(r"C:\Windows\Microsoft.NET\Framework\v4.0.30319\csc.exe"),
)

# 严禁进入发行包的敏感/运行期个人状态文件与后缀
FORBIDDEN_FILENAMES = {
    ".cf_tunnel_secret.json",
    "pet_state.json",
    "pet_error.log",
    "rclone.conf",
    "sessions.json",
}
FORBIDDEN_DIRS = {"__pycache__", ".pytest_cache", "logs", "tests", "dist", "build"}
SCAN_TEXT_EXTS = {".py", ".pyw", ".json", ".txt", ".bat", ".md", ".html", ".yml", ".spec", ".cs"}

# 发行安全闸门：扫描发布文件文本内容，拦截意外残留的个人敏感凭据
SECRET_LEAK_PATTERNS = (
    re.compile(r"\bya29\.[0-9A-Za-z._-]{20,}"),
    re.compile(r"(?<![A-Za-z0-9_-])1//[0-9A-Za-z_-]{20,}"),
    re.compile(r"\beyJ[A-Za-z0-9_-]{18,}\.[A-Za-z0-9_-]{12,}\.[A-Za-z0-9_-]{10,}"),
    re.compile(r"\b(?:sk-pet|cpk)-[A-Za-z0-9_-]{16,}"),
    re.compile(r"refresh_token\s*=\s*1//"),
)

PORTABLE_TOP_FILES = (
    "启动Colab桌宠.pyw",
    "colab_ctl.py",
    "colab_cli_bridge.py",
    "colab_remote_common.py",
    "colab_security.py",
    "colab_pet_bridge_node.py",
    "colab_bootstrap_anima.py",
    "colab_bootstrap_minimax_h3.py",
    "colab_bootstrap_drama3.py",
    "colab_bootstrap_qwen_llm.py",
    "verify_smoke.py",
    "verify_minimax_h3.py",
    "verify_drama3.py",
    "verify_qwen_llm.py",
    "stack_manifest.json",
    "requirements.txt",
    "requirements-lock.txt",
)

PORTABLE_PET_FILES = (
    "pet_app.py",
    "colab_backend.py",
    "ui.html",
    "ASSETS_NOTICE.txt",
)

LAUNCHER_CS_SOURCE = r"""
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Reflection;
using System.Windows.Forms;

[assembly: AssemblyTitle("Colab 桌宠")]
[assembly: AssemblyProduct("Google Colab AI Desktop Pet")]
[assembly: AssemblyVersion("1.0.0.0")]

namespace ColabPetLauncher
{
    internal static class Program
    {
        [STAThread]
        private static int Main(string[] args)
        {
            string baseDir = AppDomain.CurrentDomain.BaseDirectory.TrimEnd('\\');
            string entryScript = Path.Combine(baseDir, "启动Colab桌宠.pyw");
            string petScript = Path.Combine(baseDir, "colab_pet", "pet_app.py");
            string setupBat = Path.Combine(baseDir, "一键环境自检与安装.bat");

            foreach (string a in args)
            {
                if (string.Equals(a, "--self-test", StringComparison.OrdinalIgnoreCase))
                {
                    bool hasEntry = File.Exists(entryScript) && File.Exists(petScript);
                    Console.WriteLine("COLAB_PET_LAUNCHER_SELF_TEST_OK:" + (hasEntry ? "1" : "0"));
                    return hasEntry ? 0 : 2;
                }
            }

            if (!File.Exists(entryScript) || !File.Exists(petScript))
            {
                MessageBox.Show(
                    "未找到桌宠核心启动文件，请确认压缩包已完整解压后再双击运行。\n当前目录：" + baseDir,
                    "Colab 桌宠",
                    MessageBoxButtons.OK,
                    MessageBoxIcon.Error
                );
                return 1;
            }

            bool consoleMode = false;
            foreach (string a in args)
            {
                if (a == "--doctor" || a == "--export-diag" || a == "--internal-ctl")
                {
                    consoleMode = true;
                    break;
                }
            }

            string pyExe = FindPythonInterpreter(baseDir, preferConsole: consoleMode);
            if (string.IsNullOrEmpty(pyExe))
            {
                DialogResult dr = MessageBox.Show(
                    "本机尚未检测到 Python 3.11+ 运行环境。\n是否立即打开「一键环境自检与安装.bat」进行环境检查与引导？",
                    "Colab 桌宠 · 首次运行引导",
                    MessageBoxButtons.YesNo,
                    MessageBoxIcon.Information
                );
                if (dr == DialogResult.Yes && File.Exists(setupBat))
                {
                    Process.Start(new ProcessStartInfo("cmd.exe", "/c \"" + setupBat + "\"")
                    {
                        WorkingDirectory = baseDir,
                        UseShellExecute = true
                    });
                }
                return 1;
            }

            string targetScript = consoleMode ? petScript : entryScript;
            List<string> quotedArgs = new List<string>();
            quotedArgs.Add(QuoteArg(targetScript));
            foreach (string a in args)
            {
                quotedArgs.Add(QuoteArg(a));
            }

            var psi = new ProcessStartInfo
            {
                FileName = pyExe,
                Arguments = string.Join(" ", quotedArgs.ToArray()),
                WorkingDirectory = baseDir,
                UseShellExecute = false,
                CreateNoWindow = !consoleMode
            };
            psi.EnvironmentVariables["PYTHONUTF8"] = "1";
            psi.EnvironmentVariables["PYTHONIOENCODING"] = "utf-8";

            try
            {
                Process proc = Process.Start(psi);
                if (consoleMode && proc != null)
                {
                    proc.WaitForExit();
                    return proc.ExitCode;
                }
                return 0;
            }
            catch (Exception ex)
            {
                MessageBox.Show(
                    "启动 Colab 桌宠失败：" + ex.Message + "\n可先运行「一键环境自检与安装.bat」修复依赖。",
                    "Colab 桌宠",
                    MessageBoxButtons.OK,
                    MessageBoxIcon.Error
                );
                return 1;
            }
        }

        private static string QuoteArg(string arg)
        {
            if (string.IsNullOrEmpty(arg)) return "\"\"";
            if (arg.IndexOfAny(new[] { ' ', '\t', '"' }) < 0) return arg;
            return "\"" + arg.Replace("\"", "\\\"") + "\"";
        }

        private static string FindPythonInterpreter(string baseDir, bool preferConsole)
        {
            string exeName = preferConsole ? "python.exe" : "pythonw.exe";
            string altName = preferConsole ? "pythonw.exe" : "python.exe";
            var candidates = new List<string>
            {
                Path.Combine(baseDir, "runtime", exeName),
                Path.Combine(baseDir, "runtime", altName),
                Path.Combine(baseDir, ".venv", "Scripts", exeName),
                Path.Combine(baseDir, ".venv", "Scripts", altName)
            };

            string pathEnv = Environment.GetEnvironmentVariable("PATH") ?? "";
            foreach (string dir in pathEnv.Split(';'))
            {
                string trimmed = (dir ?? "").Trim();
                if (trimmed.Length == 0) continue;
                candidates.Add(Path.Combine(trimmed, exeName));
            }

            string localApp = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
            foreach (string root in new[]
            {
                Path.Combine(localApp, "Programs", "Python"),
                @"C:\Program Files"
            })
            {
                if (!Directory.Exists(root)) continue;
                try
                {
                    string[] dirs = Directory.GetDirectories(root, "Python3*");
                    Array.Sort(dirs);
                    Array.Reverse(dirs);
                    foreach (string d in dirs)
                    {
                        candidates.Add(Path.Combine(d, exeName));
                        candidates.Add(Path.Combine(d, altName));
                    }
                }
                catch { }
            }

            foreach (string cand in candidates)
            {
                if (File.Exists(cand) && cand.IndexOf("WindowsApps", StringComparison.OrdinalIgnoreCase) < 0)
                {
                    return cand;
                }
            }
            return "";
        }
    }
}
"""


def find_csc_compiler() -> Path:
    """定位 Windows 系统内置 .NET Framework 64 位 C# 编译器。"""
    for cand in CSC_CANDIDATES:
        if cand.is_file():
            return cand
    found = shutil.which("csc") or shutil.which("csc.exe")
    if found and Path(found).is_file():
        return Path(found)
    raise FileNotFoundError("未找到 Windows .NET csc.exe 编译器，无法生成原生 EXE 启动器")


def generate_pet_ico(src_png: Path, dst_ico: Path) -> Path:
    """将桌宠立绘 PNG 转换为包含多尺寸的 Windows `.ico` 应用图标。"""
    dst_ico.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src_png) as raw_img:
        img = raw_img.convert("RGBA")
        # 居中填充到正方形画布保持人物头像完整居中
        side = max(img.width, img.height)
        square = Image.new("RGBA", (side, side), (0, 0, 0, 0))
        square.paste(img, ((side - img.width) // 2, (side - img.height) // 2), img)
        square.save(
            dst_ico,
            format="ICO",
            sizes=[(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)],
        )
    return dst_ico


def compile_native_launcher(out_exe: Path, ico_path: Path | None = None) -> Path:
    """利用 Windows 原生 csc.exe 编译零依赖、无黑框启动器 `Colab桌宠.exe`。"""
    csc = find_csc_compiler()
    out_exe.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pet_csc_") as tmp_dir:
        cs_file = Path(tmp_dir) / "Launcher.cs"
        cs_file.write_text(LAUNCHER_CS_SOURCE.lstrip(), encoding="utf-8-sig")
        cmd = [
            str(csc),
            "/nologo",
            "/target:winexe",
            "/optimize+",
            "/utf8output",
            "/codepage:65001",
            "/r:System.Windows.Forms.dll",
            f"/out:{out_exe}",
        ]
        if ico_path is not None and ico_path.is_file():
            cmd.append(f"/win32icon:{ico_path}")
        cmd.append(str(cs_file))
        res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if res.returncode != 0 or not out_exe.is_file():
            raise RuntimeError(f"编译 Colab桌宠.exe 失败:\n{res.stdout}\n{res.stderr}")
    return out_exe


def _write_crlf_utf8(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = text.replace("\r\n", "\n").replace("\n", "\r\n")
    path.write_bytes(content.encode("utf-8"))


def write_helper_scripts(bundle_dir: Path, *, mode: str) -> None:
    """生成普通用户友好的自检安装批处理、诊断包导出脚本及中文使用说明。"""
    setup_bat = """@echo off
chcp 65001 >nul
setlocal EnableExtensions
cd /d "%~dp0"
echo ========================================================
echo   Colab 桌宠 · 一键环境自检与依赖安装
echo ========================================================
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未检测到 Python 3.11+，请先安装 Python 3.11~3.14 并勾选 Add to PATH。
    pause
    exit /b 1
)
echo [1/3] 正在按锁定基线检查/安装桌面依赖 (requirements.txt)...
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo [警告] 依赖安装返回异常，请检查网络或代理设置后重试。
    pause
    exit /b 1
)
echo [2/3] 检查 google-colab-cli 与 rclone 工具...
where colab >nul 2>&1 || echo [提示] 未检测到 colab，建议执行: uv tool install google-colab-cli
where rclone >nul 2>&1 || echo [提示] 未检测到 rclone，建议执行: winget install Rclone.Rclone
echo [3/3] 执行启动前深度体检...
python colab_ctl.py doctor
echo.
pause
"""
    diag_bat = """@echo off
chcp 65001 >nul
setlocal EnableExtensions
cd /d "%~dp0"
echo 正在导出全量脱敏排障诊断包 (.zip)...
if exist "Colab桌宠.exe" (
    python --version >nul 2>&1 && (
        python colab_ctl.py diag
    ) || (
        "Colab桌宠.exe" --export-diag
    )
) else (
    python colab_ctl.py diag
)
echo 诊断包已保存至 colab_pet\\logs\\ 目录，可安全用于反馈排障（已剔除全部令牌与密钥）。
pause
"""
    readme_txt = f"""Google Colab 桌面宠物挂件（Windows 发行版 · 模式: {mode}）
==========================================================

1. 启动方式
   - 直接双击「Colab桌宠.exe」（若使用源码便携版，也可双击「启动Colab桌宠.pyw」）。
   - 首次运行前请先双击「一键环境自检与安装.bat」确保 google-colab-cli、rclone 及依赖就绪。

2. 多账号与云盘配置（第 5 页）
   - 在桌宠气泡顶部分页切换到第 5 页「账号与环境」即可图形化绑定多个 Google Ultra 账号及 Google Drive (gdrive_save)。
   - 固定 Cloudflare 隧道口令、Google OAuth 刷新令牌与 LLM API Key 均通过系统凭据保险箱保存，本地配置文件不存明文。

3. ComfyUI 公网保护说明
   - 公网访问 ComfyUI 时默认启用密码与一次性票据保护；在桌宠左键「打开 ComfyUI」可免输直进画布，右键可复制公网地址与口令。

4. 故障定位与诊断包导出
   - 遇到开机或同步异常时，可在第 5 页「路径管理 ›」底部点击「🩺 导出诊断包」，或直接双击本目录「导出排障诊断包.bat」。
   - 诊断包 (.zip) 会自动脱敏所有 Token、授权码、隧道口令与访问密钥，可放心提交排障。
"""
    _write_crlf_utf8(bundle_dir / "一键环境自检与安装.bat", setup_bat)
    _write_crlf_utf8(bundle_dir / "导出排障诊断包.bat", diag_bat)
    _write_crlf_utf8(bundle_dir / "Windows打包版使用说明.txt", readme_txt)


def audit_release_directory(bundle_dir: Path) -> None:
    """发布安全闸门：校验打包目录绝对不含任何私密状态文件与明文泄露密钥。"""
    for p in bundle_dir.rglob("*"):
        if any(part in FORBIDDEN_DIRS for part in p.parts):
            raise RuntimeError(f"打包安全闸门拦截到禁止目录: {p}")
        if p.name in FORBIDDEN_FILENAMES or p.suffix.lower() == ".log":
            raise RuntimeError(f"打包安全闸门拦截到个人状态/日志文件: {p}")
        if p.is_file() and p.suffix.lower() in SCAN_TEXT_EXTS:
            txt = p.read_text(encoding="utf-8", errors="ignore")
            for rx in SECRET_LEAK_PATTERNS:
                if rx.search(txt):
                    raise RuntimeError(f"打包安全闸门检测到疑似明文凭据残留 ({p.name})")


def zip_directory(src_dir: Path, zip_path: Path) -> Path:
    """将发行目录压为相对目录规范的 `.zip` 发布包。"""
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_zip = zip_path.with_suffix(".zip.tmp")
    base_parent = src_dir.parent
    with zipfile.ZipFile(tmp_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_p in sorted(src_dir.rglob("*")):
            if file_p.is_file():
                zf.write(file_p, arcname=file_p.relative_to(base_parent).as_posix())
    tmp_zip.replace(zip_path)
    return zip_path


def _copy_clean_tree(src: Path, dst: Path) -> None:
    if not src.is_dir():
        return
    for item in src.rglob("*"):
        if any(part in FORBIDDEN_DIRS for part in item.relative_to(src).parts):
            continue
        if item.name in FORBIDDEN_FILENAMES or item.suffix.lower() == ".log":
            continue
        rel = item.relative_to(src)
        target = dst / rel
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif item.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)


def build_portable_package(pkg_dir: Path, out_dir: Path) -> tuple[Path, Path]:
    """构建 Windows 便携发布包（内置原生 Colab桌宠.exe + 完整运行脚本 + 安全校验）。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle_dir = out_dir / "Colab桌宠"
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    for rel in PORTABLE_TOP_FILES:
        src = pkg_dir / rel
        if not src.is_file():
            raise FileNotFoundError(f"缺少打包必需文件: {src}")
        shutil.copy2(src, bundle_dir / rel)

    _copy_clean_tree(pkg_dir / "colab_core", bundle_dir / "colab_core")
    _copy_clean_tree(pkg_dir / "workflows", bundle_dir / "workflows")

    pet_dst = bundle_dir / "colab_pet"
    pet_dst.mkdir(parents=True, exist_ok=True)
    for rel in PORTABLE_PET_FILES:
        src = pkg_dir / "colab_pet" / rel
        if not src.is_file():
            raise FileNotFoundError(f"缺少桌宠必需文件: {src}")
        shutil.copy2(src, pet_dst / rel)
    _copy_clean_tree(pkg_dir / "colab_pet" / "assets", pet_dst / "assets")

    idle_src = pkg_dir / "colab_pet" / "assets" / "idle.png"
    ico_path = generate_pet_ico(idle_src, bundle_dir / "colab_pet" / "assets" / "pet.ico") if idle_src.is_file() else None
    compile_native_launcher(bundle_dir / "Colab桌宠.exe", ico_path=ico_path)
    write_helper_scripts(bundle_dir, mode="portable")

    audit_release_directory(bundle_dir)
    zip_path = zip_directory(bundle_dir, out_dir / "Colab桌宠_Windows便携版.zip")
    return bundle_dir, zip_path


def build_pyinstaller_package(pkg_dir: Path, out_dir: Path) -> tuple[Path, Path]:
    """使用 PyInstaller 与 `colab_pet.spec` 构建免 Python 独立目录包。"""
    try:
        import PyInstaller  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("当前 Python 环境未安装 PyInstaller，请先 pip install pyinstaller 或改用 --mode portable") from exc

    out_dir.mkdir(parents=True, exist_ok=True)
    build_work = out_dir / "_pyinstaller_work"
    build_work.mkdir(parents=True, exist_ok=True)
    idle_src = pkg_dir / "colab_pet" / "assets" / "idle.png"
    if idle_src.is_file():
        generate_pet_ico(idle_src, pkg_dir / "build" / "pet.ico")

    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--distpath",
        str(out_dir),
        "--workpath",
        str(build_work),
        str(pkg_dir / "colab_pet.spec"),
    ]
    res = subprocess.run(cmd, cwd=str(pkg_dir))
    if res.returncode != 0:
        raise RuntimeError(f"PyInstaller 打包失败（退出码 {res.returncode}）")

    bundle_dir = out_dir / "Colab桌宠"
    if not (bundle_dir / "Colab桌宠.exe").is_file():
        raise FileNotFoundError(f"未生成预期产物: {bundle_dir / 'Colab桌宠.exe'}")

    write_helper_scripts(bundle_dir, mode="pyinstaller")
    audit_release_directory(bundle_dir)
    zip_path = zip_directory(bundle_dir, out_dir / "Colab桌宠_Windows独立版.zip")
    return bundle_dir, zip_path


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Colab 桌宠 Windows 发行包构建脚本")
    parser.add_argument(
        "--mode",
        default="portable",
        choices=("portable", "pyinstaller"),
        help="打包模式：portable（默认零依赖原生启动器）或 pyinstaller（全独立单目录）",
    )
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUT_DIR),
        help="构建输出目录（默认工作区 dist/）",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    out_dir = Path(args.out_dir).resolve()

    if args.mode == "pyinstaller":
        bundle_dir, zip_path = build_pyinstaller_package(PKG_DIR, out_dir)
    else:
        bundle_dir, zip_path = build_portable_package(PKG_DIR, out_dir)

    print(f"BUILD_OK | dir={bundle_dir} | zip={zip_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
