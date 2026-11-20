"""colab_core.health —— 技术栈清单校验、启动前多维度自检 (Preflight) 与脱敏诊断包导出模块。"""

from __future__ import annotations

import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import types
import zipfile
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


def _resolve_pkg_dir() -> Path:
    """解析项目根目录（兼容 PyInstaller 冻结态 `_MEIPASS` 布局与单测仅 mock `sys.frozen` 场景）。"""
    meipass_raw = getattr(sys, "_MEIPASS", None)
    if getattr(sys, "frozen", False) and meipass_raw:
        return Path(meipass_raw).resolve()
    return Path(__file__).resolve().parents[1]


_PKG_DIR = _resolve_pkg_dir()
if str(_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR))

from colab_core.accounts import (  # noqa: E402
    ToolPaths,
    discover_tool_paths,
    has_gdrive_save_section,
    load_cf_config,
    query_gdrive_quota,
)
from colab_core.runtime import (  # noqa: E402
    COMFY_PROFILES,
    DiagLogger,
    TaskManager,
    clean_subprocess_env,
    get_diag_logger,
    normalize_profiles,
    profile_bootstrap_spec,
    select_default_gpu,
    suppress_logged,
)
from colab_security import VaultStore, extract_valid_https_url, is_valid_cf_tunnel_token, redact_secrets  # noqa: E402

CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
MANIFEST_PATH = _PKG_DIR / "stack_manifest.json"
BRIDGE_NODE_PATH = _PKG_DIR / "colab_pet_bridge_node.py"
MIN_LOCAL_DISK_FAIL_BYTES = 256 * 1024 * 1024
MIN_LOCAL_DISK_WARN_BYTES = 5 * 1024 * 1024 * 1024
MIN_GDRIVE_WARN_GIB = 2.0


def load_stack_manifest(manifest_path: Path | None = None) -> dict[str, Any]:
    target = Path(manifest_path or MANIFEST_PATH)
    if not target.is_file():
        raise FileNotFoundError(f"未找到技术栈版本清单: {target}")
    data = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("profiles"), dict):
        raise ValueError(f"技术栈版本清单格式无效: {target}")
    return data


def _is_positive_number(val: Any) -> bool:
    """判断配置值是否为严格大于 0 的非布尔数值。"""
    return isinstance(val, (int, float)) and not isinstance(val, bool) and float(val) > 0.0


def _verify_manifest_gpu_specs(manifest: dict[str, Any], errors: list[str]) -> None:
    """校验清单中可选 `gpu_specs` 与 `gpu_order` 的结构及正数费率合法性。"""
    if "gpu_specs" not in manifest:
        return
    gpu_specs = manifest.get("gpu_specs")
    if not isinstance(gpu_specs, dict) or not gpu_specs:
        errors.append("stack_manifest.json 的 gpu_specs 必须是非空字典")
        return
    for gpu_name, spec in gpu_specs.items():
        if not isinstance(spec, dict):
            errors.append(f"GPU 规格 {gpu_name} 定义不是有效字典")
            continue
        for rate_key in ("default_rate_std", "default_rate_hm"):
            if not _is_positive_number(spec.get(rate_key)):
                errors.append(f"GPU 规格 {gpu_name} 的 {rate_key} 必须为正数")
    if "gpu_order" in manifest:
        gpu_order = manifest.get("gpu_order")
        if not isinstance(gpu_order, list) or not gpu_order:
            errors.append("stack_manifest.json 的 gpu_order 必须是非空列表")
        else:
            for g in gpu_order:
                if str(g) not in gpu_specs:
                    errors.append(f"gpu_order 引用了未定义的 GPU 规格: {g}")


def verify_manifest_and_assets(pkg_dir: Path | None = None) -> dict[str, Any]:
    """校验清单中声明的依赖锁文件、自举/冒烟脚本、共享远端与桥接组件及可选 GPU 费率完整性。"""
    root = Path(pkg_dir or _PKG_DIR)
    manifest = load_stack_manifest(root / "stack_manifest.json")
    host = manifest.get("host_baseline") or {}
    errors: list[str] = []

    for req_key in ("requirements_file", "requirements_lock_file"):
        rel = str(host.get(req_key) or "").strip()
        if not rel or not (root / rel).is_file():
            errors.append(f"缺少依赖描述文件: {rel or req_key}")

    for core_helper in (
        "colab_pet_bridge_node.py",
        "colab_remote_common.py",
        "colab_cli_bridge.py",
        "colab_sse_gateway.py",
        "colab_pet_direct_canvas.js",
        "colab_security.py",
    ):
        if not (root / core_helper).is_file():
            errors.append(f"缺少核心组件文件: {core_helper}")

    profiles = manifest.get("profiles") or {}
    for prof_name, spec in profiles.items():
        if not isinstance(spec, dict):
            errors.append(f"Profile {prof_name} 定义不是有效字典")
            continue
        for script_key in ("bootstrap_script", "smoke_script"):
            sname = str(spec.get(script_key) or "").strip()
            if not sname or not (root / sname).is_file():
                errors.append(f"Profile {prof_name} 缺少脚本 {sname or script_key}")

    _verify_manifest_gpu_specs(manifest, errors)

    return {
        "ok": len(errors) == 0,
        "default_profile": manifest.get("default_profile", "anima"),
        "profile_count": len(profiles),
        "errors": errors,
    }


@dataclass
class CheckItem:
    category: str
    name: str
    status: str  # "ok" | "warn" | "fail"
    summary: str
    detail: str = ""
    fix_hint: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PreflightReport:
    mode: str
    items: list[CheckItem] = field(default_factory=list)
    versions: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not any(it.status == "fail" for it in self.items)

    @property
    def has_warnings(self) -> bool:
        return any(it.status == "warn" for it in self.items)

    def blocking_items(self) -> list[CheckItem]:
        return [it for it in self.items if it.status == "fail"]

    def first_blocking_message(self) -> str:
        fails = self.blocking_items()
        if not fails:
            return ""
        first = fails[0]
        return f"{first.summary}（{first.fix_hint}）" if first.fix_hint else first.summary

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "mode": self.mode,
            "has_warnings": self.has_warnings,
            "fail_count": sum(1 for x in self.items if x.status == "fail"),
            "warn_count": sum(1 for x in self.items if x.status == "warn"),
            "ok_count": sum(1 for x in self.items if x.status == "ok"),
            "versions": dict(self.versions),
            "items": [x.to_dict() for x in self.items],
        }

    def format_text(self) -> str:
        badge = {"ok": "[PASS]", "warn": "[WARN]", "fail": "[FAIL]"}
        lines = [f"=== Colab 启动前环境自检 (mode={self.mode}, passed={self.ok}) ==="]
        for it in self.items:
            line = f"  {badge.get(it.status, '[INFO]')} [{it.category}] {it.name}: {it.summary}"
            if it.detail:
                line += f" | {it.detail}"
            if it.fix_hint and it.status != "ok":
                line += f" -> 建议: {it.fix_hint}"
            lines.append(line)
        return "\n".join(lines)


def detect_webview2_version() -> str | None:
    if os.name != "nt":
        return None
    try:
        import winreg
    except ImportError:
        return None
    guid = "{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}"
    for root, subkey in (
        (winreg.HKEY_LOCAL_MACHINE, rf"SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{guid}"),
        (winreg.HKEY_CURRENT_USER, rf"Software\Microsoft\EdgeUpdate\Clients\{guid}"),
        (winreg.HKEY_LOCAL_MACHINE, rf"SOFTWARE\Microsoft\EdgeUpdate\Clients\{guid}"),
    ):
        try:
            with winreg.OpenKey(root, subkey) as key:
                ver = str(winreg.QueryValueEx(key, "pv")[0] or "").strip()
                if ver and ver != "0.0.0.0":
                    return ver
        except OSError:
            continue
    return None


def _existing_anchor_dir(path_obj: Path) -> Path:
    cur = Path(path_obj)
    for _ in range(8):
        if cur.exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return Path.home()


def _default_sync_run(cmd: list[str], timeout: int = 12) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(x) for x in cmd],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        env=clean_subprocess_env(),
        creationflags=CREATE_NO_WINDOW,
    )


def _has_desktop_module(mod_name: str) -> bool:
    """探测宿主桌面运行依赖模块是否可用（兼容 pythonnet 导入后 `clr.__spec__ is None` 的特性）。"""
    try:
        return importlib.util.find_spec(mod_name) is not None
    except ValueError:
        # pythonnet 导入 `clr` 后会以无 `__spec__` 的动态模块替换 `sys.modules["clr"]`
        return isinstance(sys.modules.get(mod_name), types.ModuleType)
    except (ImportError, AttributeError):
        return False


SUPPORTED_EDITIONS = ("public", "personal")


def resolve_runtime_edition(
    edition: str | None = None,
    *,
    public_mode: bool | None = None,
    default_edition: str = "personal",
) -> str:
    """解析当前运行版本模式（`public` 公开基础版 vs `personal` 个人增强版）。

    优先级：显式 `public_mode` > 显式 `edition` > 环境变量 `COLAB_PET_EDITION` > `default_edition`。
    """
    if public_mode is not None:
        return "public" if bool(public_mode) else "personal"
    raw = str(edition if edition is not None else (os.environ.get("COLAB_PET_EDITION") or "")).strip().lower()
    if raw in ("public", "base", "openbox", "community"):
        return "public"
    if raw in ("personal", "enhanced", "pro", "private"):
        return "personal"
    return "public" if str(default_edition or "").strip().lower() == "public" else "personal"


def _probe_cli_version(
    runner: Callable[..., Any],
    cmd: list[str],
    *,
    take_last_line: bool,
    throttled_key: str,
) -> str:
    """在深度自检中安全探测本地 CLI 工具版本号，失败时节流记日志并返回空串。"""
    with suppress_logged("preflight", throttled_key=throttled_key, interval_sec=60.0):
        proc = runner(cmd, timeout=10)
        if getattr(proc, "returncode", 1) == 0:
            lines = (getattr(proc, "stdout", "") or "").strip().splitlines()
            if lines:
                return (lines[-1] if take_last_line else lines[0]).strip()
    return ""


def _check_preflight_tools(
    paths: ToolPaths,
    eff_mode: str,
    strict_rclone: bool,
    norm_profs: list[str],
    pkg_dir: Path,
    runner: Callable[..., Any],
    versions: dict[str, Any],
) -> list[CheckItem]:
    """检查本地 CLI 工具链（colab.exe / colab_py / rclone）与云端自举脚本完整性。"""
    items: list[CheckItem] = []
    if paths.colab_exe and paths.colab_exe.is_file():
        items.append(CheckItem("tools", "colab.exe", "ok", "Colab CLI 已就绪", str(paths.colab_exe)))
    else:
        items.append(CheckItem("tools", "colab.exe", "fail", "未找到 colab.exe", str(paths.colab_exe or ""), "运行 uv tool install google-colab-cli 或在第 5 页指定路径"))

    if paths.colab_py and paths.colab_py.is_file():
        colab_ver = (
            _probe_cli_version(
                runner,
                [str(paths.colab_py), "-c", "import importlib.metadata as m; print(m.version('google-colab-cli'))"],
                take_last_line=True,
                throttled_key="probe_colab_ver",
            )
            if eff_mode == "deep"
            else ""
        )
        if colab_ver:
            versions["google-colab-cli"] = colab_ver
        items.append(CheckItem("tools", "colab-cli Python", "ok", f"google-colab-cli 环境已就绪{f' (v{colab_ver})' if colab_ver else ''}", str(paths.colab_py)))
    else:
        items.append(CheckItem("tools", "colab-cli Python", "fail", "未找到 google-colab-cli Python 运行环境", str(paths.colab_py or ""), "运行 uv tool install google-colab-cli 安装托管工具"))

    if paths.rclone_exe and paths.rclone_exe.is_file():
        rc_ver = (
            _probe_cli_version(
                runner,
                [str(paths.rclone_exe), "version"],
                take_last_line=False,
                throttled_key="probe_rclone_ver",
            )
            if eff_mode == "deep"
            else ""
        )
        if rc_ver:
            versions["rclone"] = rc_ver
        items.append(CheckItem("tools", "rclone", "ok", f"rclone 已就绪{f' ({rc_ver})' if rc_ver else ''}", str(paths.rclone_exe)))
    else:
        items.append(CheckItem("tools", "rclone", "fail" if strict_rclone else "warn", "未找到 rclone.exe", str(paths.rclone_exe or ""), "运行 winget install Rclone.Rclone 或在第 5 页指定路径"))

    missing_boots = [
        profile_bootstrap_spec(p, pkg_dir=pkg_dir)[0].name
        for p in norm_profs
        if not profile_bootstrap_spec(p, pkg_dir=pkg_dir)[0].is_file()
    ]
    if missing_boots:
        items.append(CheckItem("tools", "云端自举脚本", "fail", f"缺少部署脚本: {', '.join(missing_boots)}", str(pkg_dir), "请保持项目根目录下所有 colab_bootstrap_*.py 完整"))
    else:
        items.append(CheckItem("tools", "云端自举脚本", "ok", f"已就绪 ({', '.join(norm_profs)})"))
    return items


def _check_preflight_deep_desktop(pkg_dir: Path, versions: dict[str, Any]) -> list[CheckItem]:
    """在深度自检模式下核验宿主桌面依赖、Edge WebView2 与技术栈清单一致性。"""
    items: list[CheckItem] = []
    if getattr(sys, "frozen", False):
        items.append(CheckItem("tools", "桌面运行依赖", "ok", "已内置独立运行时"))
    else:
        missing_mods = [m for m in ("requests", "websocket", "webview", "clr") if not _has_desktop_module(m)]
        if missing_mods:
            joined = ", ".join(missing_mods)
            items.append(CheckItem("tools", "桌面运行依赖", "fail", f"缺少宿主桌面运行依赖: {joined}", f"缺失模块: {joined}", "运行 python -m pip install -r requirements-lock.txt"))
        else:
            items.append(CheckItem("tools", "桌面运行依赖", "ok", "Python 桌面运行依赖已就绪"))

    if os.name == "nt":
        wv2 = detect_webview2_version()
        if wv2:
            versions["webview2"] = wv2
            items.append(CheckItem("tools", "Edge WebView2", "ok", f"WebView2 v{wv2} 已安装"))
        else:
            items.append(CheckItem("tools", "Edge WebView2", "warn", "未在注册表检测到 Edge WebView2 Runtime", "", "若桌宠窗口无法打开请安装 Microsoft Edge WebView2 Evergreen Runtime"))

    try:
        mres = verify_manifest_and_assets(pkg_dir)
        if mres["ok"]:
            items.append(CheckItem("tools", "stack_manifest", "ok", f"版本清单与桥接源码一致性校验通过 (默认路线={mres['default_profile']})"))
        else:
            items.append(CheckItem("tools", "stack_manifest", "warn", "组件一致性存在告警", "; ".join(mres["errors"][:3])))
    except Exception as exc:
        items.append(CheckItem("tools", "stack_manifest", "warn", f"版本清单校验异常: {exc}"))
    return items


def _check_preflight_auth(
    paths: ToolPaths,
    eff_mode: str,
    strict_rclone: bool,
    gdrive_free_gb: float | None,
    runner: Callable[..., Any],
) -> tuple[list[CheckItem], float | None]:
    """检查 rclone 云盘授权、深度 Colab OAuth 令牌状态并按需探测云盘余量。"""
    items: list[CheckItem] = []
    has_gd = has_gdrive_save_section(paths.rclone_conf)
    if has_gd:
        items.append(CheckItem("auth", "Google 云盘配置", "ok", "已配置 [gdrive_save]", str(paths.rclone_conf)))
    elif strict_rclone:
        items.append(CheckItem("auth", "Google 云盘配置", "fail", "未配置 rclone [gdrive_save] 云盘", str(paths.rclone_conf or ""), "在第 5 页「云盘与环境」点击一键绑定或导入 rclone.conf"))
    else:
        items.append(
            CheckItem(
                "auth",
                "Google 云盘配置",
                "warn",
                "未配置 rclone [gdrive_save] 云盘（公开基础版可跳过，仅影响可选云盘持久化与私训资产同步）",
                str(paths.rclone_conf or ""),
                "如需云盘持久化归档成图或拉取私训 LoRA，可在第 5 页「云盘与环境」可选绑定 rclone.conf",
            )
        )

    detected_gdrive_free = gdrive_free_gb
    if eff_mode == "deep":
        if paths.colab_py and paths.colab_py.is_file():
            try:
                ap = runner(
                    [str(paths.colab_py), "-c", "import json; from colab_cli import auth as a; print(json.dumps({'has_token': bool(a._get_token_from_keyring())}))"],
                    timeout=12,
                )
                if getattr(ap, "returncode", 1) == 0:
                    has_tok = bool(json.loads((getattr(ap, "stdout", "") or "").strip().splitlines()[-1]).get("has_token"))
                    items.append(
                        CheckItem(
                            "auth",
                            "Colab OAuth",
                            "ok" if has_tok else "warn",
                            "系统凭据库已存在有效 Colab 登录令牌" if has_tok else "尚未检测到本地 Colab 登录令牌",
                            "",
                            "" if has_tok else "可在桌宠第 5 页「多账号」新增绑定或运行 colab whoami",
                        )
                    )
            except Exception as exc:
                items.append(CheckItem("auth", "Colab OAuth", "warn", f"探测 Colab 凭据异常: {redact_secrets(str(exc))}"))
        if detected_gdrive_free is None and paths.rclone_exe and paths.rclone_exe.is_file() and has_gd:
            detected_gdrive_free, _ = query_gdrive_quota(paths.rclone_exe, paths.rclone_conf, run_fn=runner)
    return items, detected_gdrive_free


def _check_preflight_disk(
    download_dir: Path | None,
    pkg_dir: Path,
    detected_gdrive_free: float | None,
    du_fn: Callable[[Any], Any],
) -> list[CheckItem]:
    """检查下载目录磁盘、系统临时盘以及可选 Google 云盘剩余空间。"""
    items: list[CheckItem] = []
    dl_anchor = _existing_anchor_dir(download_dir or (pkg_dir.parent / "_downloads"))
    tmp_anchor = _existing_anchor_dir(Path(tempfile.gettempdir()))
    disk_targets = [("下载目录磁盘", dl_anchor)]
    if tmp_anchor.drive.upper() != dl_anchor.drive.upper():
        disk_targets.append(("系统临时盘", tmp_anchor))

    for disk_label, anchor in disk_targets:
        try:
            free_b = int(du_fn(str(anchor)).free)
            free_gib = free_b / (1024 ** 3)
            if free_b < MIN_LOCAL_DISK_FAIL_BYTES:
                items.append(CheckItem("disk", disk_label, "fail", f"本地剩余空间不足 ({free_b / (1024 ** 2):.0f} MiB < 256 MiB)", str(anchor), "请先清理本地磁盘空间再启动部署与拉图"))
            elif free_b < MIN_LOCAL_DISK_WARN_BYTES:
                items.append(CheckItem("disk", disk_label, "warn", f"本地磁盘剩余偏低 ({free_gib:.1f} GiB)", str(anchor), "建议预留至少 5 GiB 存放拉回的成图和视频"))
            else:
                items.append(CheckItem("disk", disk_label, "ok", f"可用空间充足 ({free_gib:.1f} GiB)", str(anchor)))
        except Exception as exc:
            items.append(CheckItem("disk", disk_label, "warn", f"读取磁盘空间失败: {redact_secrets(str(exc))}"))

    if detected_gdrive_free is not None:
        if float(detected_gdrive_free) < MIN_GDRIVE_WARN_GIB:
            items.append(CheckItem("disk", "Google 云盘余量", "warn", f"Google 云盘剩余仅 {float(detected_gdrive_free):.1f} GiB (< 2 GiB)", fix_hint="建议清理旧成图防止自动归档失败"))
        else:
            items.append(CheckItem("disk", "Google 云盘余量", "ok", f"剩余 {float(detected_gdrive_free):.1f} GiB"))
    return items


def _check_preflight_cf_tunnel(cf_cfg_p: Path, colab_py: Path | None) -> CheckItem:
    """校验 Cloudflare 固定隧道配置文件及系统凭据库中的令牌/密钥引用。"""
    if not cf_cfg_p.is_file():
        return CheckItem("config", "公网隧道配置", "ok", "使用免费 trycloudflare 临时隧道")
    try:
        raw_cf = json.loads(cf_cfg_p.read_text(encoding="utf-8"))
        if not isinstance(raw_cf, dict):
            raise ValueError("固定隧道配置文件内容必须是 JSON 对象")
        cf_resolved = load_cf_config(cf_cfg_p, include_secrets=True, colab_py=colab_py)
        if not any(raw_cf.get(k) for k in ("tunnel_token_ref", "tunnel_token", "comfy_url", "llm_url")):
            return CheckItem("config", "公网隧道配置", "ok", "使用默认临时隧道")
        for url_k in ("comfy_url", "llm_url"):
            u_val = str(cf_resolved.get(url_k) or "").strip()
            if u_val:
                extract_valid_https_url(u_val)
        if not cf_resolved.get("has_token"):
            return CheckItem("config", "公网隧道配置", "fail", "固定隧道已启用但系统凭据库中缺少 Cloudflare Tunnel Token", str(cf_cfg_p), "请在第 5 页「配置域名 ›」重新保存令牌，或切回免费临时隧道")
        if not is_valid_cf_tunnel_token(cf_resolved.get("tunnel_token")):
            return CheckItem("config", "公网隧道配置", "fail", "系统凭据库中的 Cloudflare Tunnel Token 格式无效（非合法连接器令牌）", str(cf_cfg_p), "请前往 Cloudflare Zero Trust 复制最新 Tunnel Token 重新保存，或切回免费临时隧道")
        if raw_cf.get("llm_api_key_ref") and not cf_resolved.get("has_fixed_llm_key"):
            return CheckItem("config", "公网隧道配置", "fail", "固定 LLM API Key 引用存在但系统凭据库读取为空", str(cf_cfg_p), "请重新输入固定 LLM Key 或留空改回每次随机生成")
        return CheckItem("config", "公网隧道配置", "ok", "已启用 Cloudflare 固定域名（系统凭据库引用校验通过）")
    except Exception as exc:
        return CheckItem("config", "公网隧道配置", "fail", f"隧道配置或凭据校验失败: {redact_secrets(str(exc))}", str(cf_cfg_p), "请检查域名格式与系统凭据库权限，或先切回免费临时隧道")


def _check_preflight_config(
    state_path: Path | None,
    cf_config_path: Path | None,
    pkg_dir: Path,
    paths: ToolPaths,
) -> list[CheckItem]:
    """检查桌宠状态 JSON 可读性与公网隧道配置完整性。"""
    items: list[CheckItem] = []
    if state_path is not None and Path(state_path).is_file():
        try:
            st_obj = json.loads(Path(state_path).read_text(encoding="utf-8"))
            is_dict = isinstance(st_obj, dict)
            items.append(CheckItem("config", "桌宠状态文件", "ok" if is_dict else "warn", "状态 JSON 校验通过" if is_dict else "状态文件根节点非字典，启动时将自动重置"))
        except Exception as exc:
            items.append(CheckItem("config", "桌宠状态文件", "warn", f"状态文件解析失败（将自动重建）: {exc}"))

    cf_cfg_p = Path(
        cf_config_path
        if cf_config_path is not None
        else (os.environ.get("COLAB_PET_CF_CONFIG_PATH") or str(pkg_dir / ".cf_tunnel_secret.json"))
    )
    items.append(_check_preflight_cf_tunnel(cf_cfg_p, paths.colab_py))
    return items


def _check_preflight_profile_gpu(norm_profs: list[str], eff_gpu: str) -> CheckItem:
    """检查所选工作流组合与目标 GPU 显存/磁盘承载上限是否匹配（含 L4 24GB 双栖阻断）。"""
    has_comfy = any(p in COMFY_PROFILES for p in norm_profs)
    prof_label = "+".join(norm_profs)
    if "qwen_llm" in norm_profs and eff_gpu in ("T4", "L4"):
        return CheckItem("profile", "工作流与显卡匹配", "fail", f"Qwen3.8-180B 无法在 {eff_gpu} 显卡上运行（最低需 A100/G4/H100）", f"profiles={','.join(norm_profs)}, gpu={eff_gpu}", "请改选 G4 / A100 显卡，或将对话模型切换为 Qwen3.8-27B")
    if ("qwen_27b" in norm_profs and has_comfy) and eff_gpu in ("T4", "L4"):
        return CheckItem("profile", "工作流与显卡匹配", "fail", f"ComfyUI + Qwen27B 双栖同驻超出 {eff_gpu} 显存承受上限", f"profiles={','.join(norm_profs)}, gpu={eff_gpu}", "双栖请选用 G4 (96GB) / A100 / H100")
    if ("minimax" in norm_profs or "drama3" in norm_profs) and eff_gpu == "T4":
        return CheckItem("profile", "工作流与显卡匹配", "fail", f"{prof_label} 对显存与磁盘需求超出 T4 (16GB) 承载范围", f"gpu={eff_gpu}", "请改选 G4 (96GB) 或 A100 显卡")
    if "minimax" in norm_profs and eff_gpu == "L4":
        return CheckItem("profile", "工作流与显卡匹配", "warn", f"当前在 L4 (24GB) 上运行 {prof_label} 显存较为吃紧", fix_hint="推荐在 G4 (96GB) 上运行 MiniMax-H3")
    return CheckItem("profile", "工作流与显卡匹配", "ok", f"{prof_label} 与 {eff_gpu} 硬件规格兼容")


def _check_preflight_mock_vault(eff_mode: str) -> list[CheckItem]:
    """在深度自检且启用模拟凭据库时执行一次读写探针校验。"""
    if eff_mode != "deep" or not os.environ.get("COLAB_PET_MOCK_VAULT_PATH"):
        return []
    try:
        vs = VaultStore()
        vs.set_secret("__preflight_probe__", "ok")
        vs.delete_secret("__preflight_probe__")
        return []
    except Exception as exc:
        return [CheckItem("config", "系统凭据保险箱", "warn", f"保险箱探针异常: {exc}")]


def run_preflight_checks(
    *,
    mode: str = "fast",
    profiles: list[str] | str | None = None,
    gpu: str | None = None,
    tool_paths: ToolPaths | None = None,
    state_path: Path | None = None,
    download_dir: Path | None = None,
    cf_config_path: Path | None = None,
    pkg_dir: Path = _PKG_DIR,
    gdrive_free_gb: float | None = None,
    require_local_rclone: bool | None = None,
    edition: str | None = None,
    public_mode: bool | None = None,
    run_fn: Callable[..., Any] | None = None,
    disk_usage_fn: Callable[[Any], Any] | None = None,
) -> PreflightReport:
    """执行启动前多维度环境、授权、磁盘、配置与显卡兼容性自检。

    公开基础版（`edition="public"` / `public_mode=True` / `COLAB_PET_EDITION=public` 或 `require_local_rclone=False`）下，
    缺失本地 rclone 与未绑定 `[gdrive_save]` 均降级为 `warn`（提示可选绑定云盘持久化）而不阻塞开机；
    个人增强版（默认或显式 `require_local_rclone=True`）保留原有强校验语义。
    """
    eff_mode = "deep" if str(mode or "fast").strip().lower() == "deep" else "fast"
    eff_edition = resolve_runtime_edition(edition, public_mode=public_mode, default_edition="personal")
    strict_rclone = bool(require_local_rclone) if require_local_rclone is not None else (eff_edition != "public")
    paths = tool_paths or discover_tool_paths(state_path=state_path)
    norm_profs = normalize_profiles(profiles) if profiles is not None else ["anima"]
    eff_gpu = str(gpu or select_default_gpu(norm_profs)).strip().upper()
    runner = run_fn or _default_sync_run
    du_fn = disk_usage_fn or shutil.disk_usage

    versions: dict[str, Any] = {"python": sys.version.split()[0], "edition": eff_edition}
    items = _check_preflight_tools(paths, eff_mode, strict_rclone, norm_profs, pkg_dir, runner, versions)
    if eff_mode == "deep":
        items.extend(_check_preflight_deep_desktop(pkg_dir, versions))
    auth_items, detected_gdrive_free = _check_preflight_auth(paths, eff_mode, strict_rclone, gdrive_free_gb, runner)
    items.extend(auth_items)
    items.extend(_check_preflight_disk(download_dir, pkg_dir, detected_gdrive_free, du_fn))
    items.extend(_check_preflight_config(state_path, cf_config_path, pkg_dir, paths))
    items.append(_check_preflight_profile_gpu(norm_profs, eff_gpu))
    items.extend(_check_preflight_mock_vault(eff_mode))
    return PreflightReport(mode=eff_mode, items=items, versions=versions)


# =========================================================================
# 脱敏排障诊断信息收集与 .zip 打包导出
# =========================================================================

_SENSITIVE_KEYS = frozenset({"api_key", "comfy_api_key", "llm_api_key", "tunnel_token", "access_token", "refresh_token", "token", "code_verifier", "clipboard_text", "secret"})


def _redact_json_obj(obj: Any) -> Any:
    if isinstance(obj, str):
        return redact_secrets(obj)
    if isinstance(obj, list):
        return [_redact_json_obj(x) for x in obj]
    if isinstance(obj, dict):
        return {
            str(k): ("[REDACTED]" if str(k).strip().lower() in _SENSITIVE_KEYS else _redact_json_obj(v))
            for k, v in obj.items()
        }
    return obj


def _safe_tail_text(path: Path, max_lines: int = 200) -> str:
    if not path.is_file():
        return ""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()[-max(1, int(max_lines)) :]
        return "\n".join(redact_secrets(ln) for ln in lines) + ("\n" if lines else "")
    except OSError:
        return ""


def export_diagnostic_bundle(
    out_dir: Path | None = None,
    *,
    pkg_dir: Path = _PKG_DIR,
    state_path: Path | None = None,
    state_dict: dict[str, Any] | None = None,
    snapshot: dict[str, Any] | None = None,
    task_manager: TaskManager | None = None,
    diag_logger: DiagLogger | None = None,
    cf_config_path: Path | None = None,
    tool_paths: ToolPaths | None = None,
    deep_preflight: bool = False,
) -> Path:
    eff_state_path = Path(state_path) if state_path is not None else (pkg_dir / "colab_pet" / "pet_state.json")
    logger = diag_logger or get_diag_logger()
    paths = tool_paths or discover_tool_paths(state_path=eff_state_path)

    raw_state = state_dict
    if raw_state is None and eff_state_path.is_file():
        try:
            loaded = json.loads(eff_state_path.read_text(encoding="utf-8"))
            raw_state = loaded if isinstance(loaded, dict) else {}
        except Exception:
            raw_state = {}
    if not isinstance(raw_state, dict):
        raw_state = {}

    prefs = raw_state.get("prefs") if isinstance(raw_state.get("prefs"), dict) else {}
    preflight = run_preflight_checks(
        mode="deep" if deep_preflight else "fast",
        profiles=prefs.get("boot_profiles") or prefs.get("boot_profile") or ["anima"],
        gpu=prefs.get("boot_gpu"),
        tool_paths=paths,
        state_path=eff_state_path,
        cf_config_path=cf_config_path,
        pkg_dir=pkg_dir,
    )
    try:
        manifest = load_stack_manifest(pkg_dir / "stack_manifest.json")
        manifest_verify = verify_manifest_and_assets(pkg_dir)
    except Exception as exc:
        manifest, manifest_verify = {"error": redact_secrets(str(exc))}, {"ok": False, "errors": [redact_secrets(str(exc))]}

    state_summary = _redact_json_obj({
        "date": raw_state.get("date"),
        "last_balance": raw_state.get("last_balance"),
        "last_rate_hourly": raw_state.get("last_rate_hourly"),
        "last_assignments_count": raw_state.get("last_assignments_count"),
        "last_sessions": raw_state.get("last_sessions") or [],
        "managed_binding": raw_state.get("managed_binding"),
        "account_profiles": raw_state.get("account_profiles") or [],
        "prefs": prefs,
        "snapshot_comfy": (snapshot or {}).get("comfy") if isinstance(snapshot, dict) else None,
        "snapshot_deploy": (snapshot or {}).get("deploy") if isinstance(snapshot, dict) else None,
    })
    sys_info = _redact_json_obj({
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "os": platform.platform(),
        "python": sys.version,
        "executable": sys.executable,
        "frozen": bool(getattr(sys, "frozen", False)),
        "webview2": detect_webview2_version(),
        "tool_paths": {
            "colab_exe": str(paths.colab_exe) if paths.colab_exe else "",
            "colab_py": str(paths.colab_py) if paths.colab_py else "",
            "rclone_exe": str(paths.rclone_exe) if paths.rclone_exe else "",
            "rclone_conf_exists": bool(paths.rclone_conf and paths.rclone_conf.is_file()),
        },
    })

    target_dir = Path(out_dir) if out_dir is not None else logger.log_path.parent
    target_dir.mkdir(parents=True, exist_ok=True)
    zip_path = target_dir / f"colab_pet_diag_{datetime.now().strftime('%Y%m%d-%H%M%S')}.zip"
    tmp_zip = zip_path.with_suffix(".zip.tmp")

    def _safe_json_bytes(obj: Any) -> bytes:
        return redact_secrets(json.dumps(_redact_json_obj(obj), ensure_ascii=False, indent=2) + "\n").encode("utf-8", errors="replace")

    with zipfile.ZipFile(tmp_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("system_info.json", _safe_json_bytes(sys_info))
        zf.writestr("preflight_report.json", _safe_json_bytes(preflight.to_dict()))
        zf.writestr("state_summary.json", _safe_json_bytes(state_summary))
        zf.writestr("tasks_snapshot.json", _safe_json_bytes(task_manager.snapshot() if task_manager is not None else {}))
        zf.writestr("stack_manifest_summary.json", _safe_json_bytes({
            "verify": manifest_verify,
            "default_profile": manifest.get("default_profile"),
            "profiles": list((manifest.get("profiles") or {}).keys()) if isinstance(manifest.get("profiles"), dict) else [],
        }))
        zf.writestr("logs/colab_diag_tail.log", redact_secrets("\n".join(logger.read_tail_lines(max_lines=300)) or "(暂无诊断日志)\n").encode("utf-8", errors="replace"))
        zf.writestr("logs/pet_error_tail.log", (_safe_tail_text(pkg_dir / "colab_pet" / "pet_error.log", max_lines=200) or "(暂无启动异常日志)\n").encode("utf-8", errors="replace"))

    tmp_zip.replace(zip_path)
    logger.info("diagnostics", "已导出脱敏排障诊断包", path=str(zip_path))
    return zip_path
