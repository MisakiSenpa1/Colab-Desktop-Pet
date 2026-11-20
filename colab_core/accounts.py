"""colab_core.accounts —— 跨设备工具探测、rclone 云盘绑定与多账号凭据辅助模块。"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
import urllib.parse
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_PKG_DIR = Path(__file__).resolve().parents[1]
if str(_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR))

from colab_security import load_cf_tunnel_settings, redact_secrets  # noqa: E402

CLI_BRIDGE_SCRIPT = _PKG_DIR / "colab_cli_bridge.py"
RCLONE_GDRIVE_SECTION_RE = re.compile(r"^\s*\[gdrive_save\]\s*$", re.MULTILINE)
RCLONE_INI_SECTION_RE = re.compile(r"^\s*\[([^\]]+)\]\s*$", re.MULTILINE)

DEFAULT_FALLBACK_COLAB_EXE = Path.home() / ".local" / "bin" / "colab.exe"
DEFAULT_FALLBACK_COLAB_PY = (
    Path.home()
    / "AppData"
    / "Roaming"
    / "uv"
    / "tools"
    / "google-colab-cli"
    / "Scripts"
    / "python.exe"
)
DEFAULT_FALLBACK_RCLONE_EXE = Path.home() / "scoop" / "shims" / "rclone.exe"
DEFAULT_FALLBACK_RCLONE_CONF = (
    (Path.home() / "AppData" / "Roaming" / "rclone" / "rclone.conf")
    if os.name == "nt"
    else (Path.home() / ".config" / "rclone" / "rclone.conf")
)


@dataclass
class ToolPaths:
    colab_exe: Path | None
    colab_py: Path
    rclone_exe: Path
    rclone_conf: Path


def _pick_first_existing(candidates: list[Path], fallback: Path) -> Path:
    for cand in candidates:
        try:
            if cand and cand.is_file():
                return cand
        except OSError:
            continue
    return fallback


def load_pet_prefs_paths(state_path: Path | None = None) -> dict[str, str]:
    target = Path(state_path) if state_path is not None else (_PKG_DIR / "colab_pet" / "pet_state.json")
    if not target.is_file():
        return {}
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
        prefs = raw.get("prefs") if isinstance(raw, dict) else None
        if not isinstance(prefs, dict):
            return {}
        return {
            k: str(prefs[k]).strip()
            for k in ("colab_exe_path", "colab_py_path", "rclone_exe_path", "rclone_conf_path")
            if str(prefs.get(k) or "").strip()
        }
    except Exception:
        return {}


def resolve_colab_exe(custom_path: str = "", *, state_path: Path | None = None, allow_fallback: bool = True) -> Path | None:
    pref_p = load_pet_prefs_paths(state_path).get("colab_exe_path", "") if not custom_path else ""
    cands = [Path(raw) for raw in (custom_path, str(os.environ.get("COLAB_PET_COLAB_EXE") or "").strip(), pref_p, shutil.which("colab") or shutil.which("colab.exe") or "") if raw]
    appdata, localappdata = os.environ.get("APPDATA", ""), os.environ.get("LOCALAPPDATA", "")
    cands.append(Path.home() / ".local" / "bin" / "colab.exe")
    if appdata:
        cands.append(Path(appdata) / "uv" / "tools" / "google-colab-cli" / "Scripts" / "colab.exe")
    cands.append(Path.home() / "AppData" / "Roaming" / "uv" / "tools" / "google-colab-cli" / "Scripts" / "colab.exe")
    if localappdata:
        cands.append(Path(localappdata) / "uv" / "tools" / "google-colab-cli" / "Scripts" / "colab.exe")
    cands.extend([Path.home() / ".local" / "pipx" / "venvs" / "google-colab-cli" / "Scripts" / "colab.exe", DEFAULT_FALLBACK_COLAB_EXE])
    for cand in cands:
        try:
            if cand and cand.is_file():
                return cand
        except OSError:
            continue
    return DEFAULT_FALLBACK_COLAB_EXE if allow_fallback else None


def resolve_colab_py(custom_path: str = "", *, colab_exe: Path | None = None, state_path: Path | None = None) -> Path:
    pref_p = load_pet_prefs_paths(state_path).get("colab_py_path", "") if not custom_path else ""
    cands = [Path(raw) for raw in (custom_path, str(os.environ.get("COLAB_PET_COLAB_PY") or "").strip(), pref_p) if raw]
    appdata, localappdata = os.environ.get("APPDATA", ""), os.environ.get("LOCALAPPDATA", "")
    if appdata:
        cands.append(Path(appdata) / "uv" / "tools" / "google-colab-cli" / "Scripts" / "python.exe")
    cands.append(Path.home() / "AppData" / "Roaming" / "uv" / "tools" / "google-colab-cli" / "Scripts" / "python.exe")
    if localappdata:
        cands.append(Path(localappdata) / "uv" / "tools" / "google-colab-cli" / "Scripts" / "python.exe")
    cands.append(Path.home() / ".local" / "pipx" / "venvs" / "google-colab-cli" / "Scripts" / "python.exe")
    if colab_exe is not None:
        try:
            cands.append(Path(colab_exe).resolve().with_name("python.exe"))
        except OSError:
            pass
    cands.append(DEFAULT_FALLBACK_COLAB_PY)
    return _pick_first_existing(cands, DEFAULT_FALLBACK_COLAB_PY)


def resolve_rclone_exe(custom_path: str = "", *, state_path: Path | None = None) -> Path:
    pref_p = load_pet_prefs_paths(state_path).get("rclone_exe_path", "") if not custom_path else ""
    cands = [Path(raw) for raw in (custom_path, str(os.environ.get("COLAB_PET_RCLONE_EXE") or "").strip(), pref_p, shutil.which("rclone") or shutil.which("rclone.exe") or "") if raw]
    localappdata = os.environ.get("LOCALAPPDATA", "")
    cands.extend([Path.home() / "scoop" / "shims" / "rclone.exe", Path.home() / "scoop" / "apps" / "rclone" / "current" / "rclone.exe"])
    if localappdata:
        cands.append(Path(localappdata) / "Microsoft" / "WinGet" / "Links" / "rclone.exe")
    cands.extend([Path(r"C:\ProgramData\chocolatey\bin\rclone.exe"), DEFAULT_FALLBACK_RCLONE_EXE])
    return _pick_first_existing(cands, DEFAULT_FALLBACK_RCLONE_EXE)


def resolve_rclone_conf(custom_path: str = "", *, rclone_exe: Path | None = None, state_path: Path | None = None) -> Path:
    pref_p = load_pet_prefs_paths(state_path).get("rclone_conf_path", "") if not custom_path else ""
    cands = [Path(raw) for raw in (custom_path, str(os.environ.get("COLAB_PET_RCLONE_CONF") or "").strip(), pref_p) if raw]
    appdata = os.environ.get("APPDATA", "")
    if appdata:
        cands.append(Path(appdata) / "rclone" / "rclone.conf")
    cands.append(Path.home() / ".config" / "rclone" / "rclone.conf")
    if rclone_exe is not None:
        try:
            exe_dir = Path(rclone_exe).resolve().parent
            cands.append(exe_dir / "rclone.conf")
            if exe_dir.name.lower() == "shims":
                cands.append(exe_dir.parent / "apps" / "rclone" / "current" / "rclone.conf")
        except OSError:
            pass
    cands.extend([Path.home() / "scoop" / "apps" / "rclone" / "current" / "rclone.conf", DEFAULT_FALLBACK_RCLONE_CONF])
    return _pick_first_existing(cands, DEFAULT_FALLBACK_RCLONE_CONF)


def discover_tool_paths(prefs: dict[str, Any] | None = None, *, state_path: Path | None = None) -> ToolPaths:
    pref_map = prefs if isinstance(prefs, dict) else load_pet_prefs_paths(state_path)
    cexe = resolve_colab_exe(str(pref_map.get("colab_exe_path") or ""), state_path=state_path)
    cpy = resolve_colab_py(str(pref_map.get("colab_py_path") or ""), colab_exe=cexe, state_path=state_path)
    rexe = resolve_rclone_exe(str(pref_map.get("rclone_exe_path") or ""), state_path=state_path)
    rconf = resolve_rclone_conf(str(pref_map.get("rclone_conf_path") or ""), rclone_exe=rexe, state_path=state_path)
    return ToolPaths(colab_exe=cexe, colab_py=cpy, rclone_exe=rexe, rclone_conf=rconf)


def has_gdrive_save_section(conf_path: Path | None) -> bool:
    if not conf_path or not Path(conf_path).is_file():
        return False
    try:
        return bool(RCLONE_GDRIVE_SECTION_RE.search(Path(conf_path).read_text(encoding="utf-8", errors="ignore")))
    except OSError:
        return False


def sanitize_account_label(label: str, fallback: str = "Colab 账号") -> str:
    cleaned = re.sub(r"[\r\n\t]+", " ", str(label or "")).strip()
    return cleaned[:16] if cleaned else fallback


def default_account_label(existing_profiles: list[dict[str, Any]], tier: str = "unknown") -> str:
    prefix_map = {"ultra": "Ultra", "pro_plus": "Pro+", "pro": "Pro", "free": "Free"}
    prefix = prefix_map.get(str(tier or "").strip().lower(), "Colab")
    used = {str(p.get("label") or "").strip() for p in existing_profiles}
    idx = len(existing_profiles) + 1
    while f"{prefix} {idx}号" in used:
        idx += 1
    return f"{prefix} {idx}号"


def extract_oauth_code(raw_input: str) -> str:
    txt = str(raw_input or "").strip()
    if not txt:
        return ""
    if "code=" in txt and ("://" in txt or txt.startswith("?")):
        parsed = urllib.parse.urlparse(txt if "://" in txt else f"https://localhost/{txt.lstrip('?')}")
        qs = urllib.parse.parse_qs(parsed.query)
        if qs.get("code"):
            return str(qs["code"][0]).strip()
    return txt.splitlines()[0].strip()


def _split_ini_sections(text: str) -> tuple[list[str], list[tuple[str, list[str]]]]:
    """按 INI 节头拆分 rclone.conf，完整保留节前注释与各个 [remote] 配置节。"""
    preamble: list[str] = []
    sections: list[tuple[str, list[str]]] = []
    cur_name: str | None = None
    cur_lines: list[str] = []
    for line in text.splitlines():
        m = re.match(r"^\s*\[([^\]]+)\]\s*$", line)
        if m:
            if cur_name is not None:
                sections.append((cur_name, cur_lines))
            cur_name = m.group(1).strip()
            cur_lines = [line]
        elif cur_name is None:
            preamble.append(line)
        else:
            cur_lines.append(line)
    if cur_name is not None:
        sections.append((cur_name, cur_lines))
    return preamble, sections


def _merge_gdrive_save_section(existing_text: str, new_gdrive_text: str) -> str:
    """将新的 [gdrive_save] 节合入既有 rclone.conf 内容，保留其他 [remote] 配置节不变。"""
    preamble, existing_sections = _split_ini_sections(existing_text)
    _, new_sections = _split_ini_sections(new_gdrive_text)
    gdrive_lines = next(
        (lines for name, lines in new_sections if name == "gdrive_save"),
        new_gdrive_text.strip().splitlines(),
    )
    cleaned_gdrive_lines = [ln for ln in gdrive_lines]
    while cleaned_gdrive_lines and not cleaned_gdrive_lines[-1].strip():
        cleaned_gdrive_lines.pop()

    merged_sections: list[list[str]] = []
    replaced = False
    for sec_name, sec_lines in existing_sections:
        lines_copy = list(sec_lines)
        while lines_copy and not lines_copy[-1].strip():
            lines_copy.pop()
        if sec_name == "gdrive_save":
            if not replaced:
                merged_sections.append(cleaned_gdrive_lines)
                replaced = True
        else:
            merged_sections.append(lines_copy)
    if not replaced:
        merged_sections.append(cleaned_gdrive_lines)

    while preamble and not preamble[-1].strip():
        preamble.pop()
    blocks: list[str] = []
    if preamble:
        blocks.append("\n".join(preamble))
    for sec_lines in merged_sections:
        if sec_lines:
            blocks.append("\n".join(sec_lines))
    return "\n\n".join(blocks) + "\n"


def _atomic_write_private_conf(dst_conf: Path, content: str) -> None:
    """通过同目录 .tmp 临时文件与 replace 原子写入 rclone 配置，并尝试收紧权限为 0o600。"""
    dst_conf.parent.mkdir(parents=True, exist_ok=True)
    tmp_conf = dst_conf.with_name(f".{dst_conf.name}.{os.getpid()}.tmp")
    fd = os.open(str(tmp_conf), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fp:
            fp.write(content)
        try:
            os.chmod(str(tmp_conf), 0o600)
        except OSError:
            pass
        tmp_conf.replace(dst_conf)
        try:
            os.chmod(str(dst_conf), 0o600)
        except OSError:
            pass
    except BaseException:
        tmp_conf.unlink(missing_ok=True)
        raise


def validate_and_import_rclone_conf(src_path: str | Path, *, package_dir: Path = _PKG_DIR) -> tuple[bool, Path | None, str]:
    """校验并安全导入 rclone.conf；合并写入目标配置时自动备份并保留原有其他 remote。"""
    p = Path(str(src_path or "").strip())
    if not p.is_file():
        return False, None, "所选 rclone.conf 文件不存在"
    try:
        content = p.read_text(encoding="utf-8", errors="ignore")
    except Exception as exc:
        return False, None, f"读取配置文件失败: {redact_secrets(str(exc))}"

    if not RCLONE_GDRIVE_SECTION_RE.search(content):
        sections = RCLONE_INI_SECTION_RE.findall(content)
        if len(sections) == 1 and re.search(r"^\s*type\s*=\s*drive\b", content, re.MULTILINE):
            appdata = os.environ.get("APPDATA", "")
            dst_conf = (Path(appdata) / "rclone" / "rclone.conf") if appdata else (package_dir / "rclone.conf")
            dst_conf.parent.mkdir(parents=True, exist_ok=True)
            rewritten = RCLONE_INI_SECTION_RE.sub("[gdrive_save]", content, count=1)
            if dst_conf.is_file():
                bak_conf = dst_conf.with_suffix(".conf.bak")
                shutil.copy2(dst_conf, bak_conf)
                try:
                    os.chmod(str(bak_conf), 0o600)
                except OSError:
                    pass
                try:
                    same_target = p.resolve() == dst_conf.resolve()
                except OSError:
                    same_target = False
                if same_target:
                    merged_text = rewritten.rstrip() + "\n"
                else:
                    existing_text = dst_conf.read_text(encoding="utf-8", errors="ignore")
                    merged_text = _merge_gdrive_save_section(existing_text, rewritten)
            else:
                merged_text = rewritten.rstrip() + "\n"
            _atomic_write_private_conf(dst_conf, merged_text)
            return True, dst_conf, ""
        return False, None, "该 rclone.conf 缺少 [gdrive_save] 远端，建议直接点「一键绑定云盘」"
    return True, p, ""


def query_gdrive_quota(rclone_exe: Path, rclone_conf: Path, *, run_fn: Callable[..., Any]) -> tuple[float | None, float | None]:
    if not rclone_exe or not rclone_exe.is_file() or not has_gdrive_save_section(rclone_conf):
        return None, None
    try:
        proc = run_fn([str(rclone_exe), "about", "gdrive_save:", "--json", "--config", str(rclone_conf)], timeout=15)
        if getattr(proc, "returncode", 1) != 0:
            return None, None
        info = json.loads(getattr(proc, "stdout", "") or "{}")
        free_b, total_b = info.get("free"), info.get("total")
        return (
            round(float(free_b) / (1024 ** 3), 1) if isinstance(free_b, (int, float)) else None,
            round(float(total_b) / (1024 ** 3), 1) if isinstance(total_b, (int, float)) else None,
        )
    except Exception:
        return None, None


def load_cf_config(config_path: Path, *, include_secrets: bool = False, colab_py: Path | None = None) -> dict[str, Any]:
    return load_cf_tunnel_settings(config_path, include_secrets=include_secrets, colab_py=colab_py)
