"""colab_core.agent.safety —— 远端代码硬黑名单护栏、表驱动只读诊断白名单与全链路脱敏辅助。"""

from __future__ import annotations

import json
import re
import shlex
from collections.abc import Callable
from typing import Any

from colab_security import redact_secrets

# =========================================================================
# 1. 远端代码硬黑名单安全护栏
# =========================================================================

_DANGEROUS_CODE_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"(?i)\brm\b[^\n;|&]*(?:-[a-z]*r[a-z]*|--recursive)\b[^\n;|&]*\s+"
            r"(?:--no-preserve-root\s+)?['\"]?(?:/\*?|/root/?\*?|/etc/?\*?|/usr/?\*?|"
            r"/var/?\*?|/bin/?\*?|/lib/?\*?|/boot/?\*?|/sys/?\*?|/proc/?\*?|~/?\*?)['\"]?(?=$|[\s;|&])"
        ),
        "禁止对根目录或系统关键目录执行递归删除 (rm -rf)",
    ),
    (
        re.compile(
            r"""(?i)['"]rm['"]\s*,\s*['"][^'"]*r[^'"]*['"]\s*,\s*['"](?:/\*?|/root/?\*?|/etc/?\*?|/usr/?\*?|~/?\*?)['"]"""
        ),
        "禁止通过参数列表调用 rm 递归删除根目录或系统目录",
    ),
    (
        re.compile(
            r"""(?i)\bshutil\.rmtree\s*\(\s*['"](?:/|/\*|/root|/etc|/usr|/var|[A-Za-z]:[/\\]?)['"]"""
        ),
        "禁止通过 shutil.rmtree 删除根路径或系统目录",
    ),
    (
        re.compile(r"(?i)\b(?:mkfs(?:\.[a-z0-9_+-]+)?|mke2fs|wipefs)\b"),
        "禁止执行磁盘格式化命令 (mkfs/wipefs)",
    ),
    (
        re.compile(r"(?i)\bdd\b[^\n;|&]*\bif="),
        "禁止执行裸磁盘块写入命令 (dd if=)",
    ),
    (
        re.compile(r"(?i)>\s*/dev/(?:sd[a-z]\d*|nvme\d+n\d+(?:p\d+)?|vd[a-z]\d*|loop\d+)"),
        "禁止直接覆写块设备节点",
    ),
    (
        re.compile(r"(?i)(?:^|[;\s|&(])(?:shutdown|reboot|poweroff|halt|init\s+[06]|systemctl\s+(?:poweroff|reboot|halt))\b"),
        "禁止执行关机/重启系统指令",
    ),
    (
        re.compile(r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;?\s*:|\b(\w+)\s*\(\s*\)\s*\{\s*\1\s*\|\s*\1\s*&"),
        "拦截疑似 Fork Bomb 进程炸弹",
    ),
    (
        re.compile(r"(?is)while\s+True\s*:[^\n]*\bos\.fork\s*\("),
        "拦截无限递归派生进程 (os.fork)",
    ),
    (
        re.compile(r"(?i)\bchmod\b[^\n;|&]*\s+-R\s+[^\n;|&]*\s+/\s*(?:$|[\s;|&])"),
        "禁止对根目录执行递归权限放开 (chmod -R /)",
    ),
    (
        re.compile(
            r"(?i)(?:\.cf_tunnel_secret(?:\.json)?|\.config[/\\]colab-cli|sessions\.json|"
            r"pet_state\.json|mock_vault|google-colab-cli-pet-vault|rclone\.conf|"
            r"comfy_auth_key|llama_api_key|colab_service_meta(?:\.json)?|/etc/shadow|%APPDATA%|%USERPROFILE%|"
            r"[A-Za-z]:[/\\]Users[/\\]|CredReadW?|keyring\.get_password)"
        ),
        "禁止访问宿主机或远端敏感凭据配置文件路径",
    ),
)


def check_remote_code_safety(code: str, mode: str) -> tuple[bool, str]:
    """校验待执行的远端 Bash/Python 代码，阻断破坏性命令与凭据窥探。"""
    clean_mode = str(mode or "").strip().lower()
    if clean_mode not in ("bash", "python"):
        return False, f"不支持的远端执行模式: {mode}"
    raw_code = str(code or "")
    if not raw_code.strip():
        return False, "待执行代码内容不能为空"
    if len(raw_code) > 32768:
        return False, "待执行代码长度超出 32 KiB 安全上限"
    for pattern, reason in _DANGEROUS_CODE_RULES:
        if pattern.search(raw_code):
            return False, f"安全护栏拦截: {reason}"
    return True, ""


# =========================================================================
# 2. 表驱动严格只读诊断白名单（单命令短验证器，每个验证器低圈复杂度）
# =========================================================================

_READONLY_FORBIDDEN_SHELL_CHARS: tuple[str, ...] = (
    "\r",
    "\n",
    "\x00",
    ";",
    "|",
    ">",
    "<",
    "`",
    "$",
    "(",
    ")",
    "{",
    "}",
    "*",
    "?",
    "[",
    "]",
    "~",
    "\\",
    "!",
    "#",
)
_READONLY_SENSITIVE_PATH_RE = re.compile(
    r"(?i)(?:auth|secret|token|passwd|shadow|rclone|vault|credential|session|"
    r"cookie|netrc|id_rsa|id_ed25519|gnupg|ssh|comfy_auth_key|llama_api_key|"
    r"colab_service_meta|cf_tunnel|\.env|environ|cmdline|maps|mem)"
)
_READONLY_ALLOWED_LOG_PATH_RES: tuple[re.Pattern[str], ...] = (
    re.compile(r"^/root/comfy_[A-Za-z0-9_.-]+\.log$"),
    re.compile(r"^/root/llama_server\.log$"),
    re.compile(r"^/root/llm_sse_gateway\.log$"),
    re.compile(r"^/root/cloudflared(?:_[A-Za-z0-9_.-]+)?\.log$"),
    re.compile(r"^/tmp/comfy_[A-Za-z0-9_.-]+\.log$"),
    re.compile(r"^/tmp/llama_server\.log$"),
    re.compile(r"^/content/ComfyUI/comfyui\.log$"),
)
_READONLY_SAFE_VALUE_RE = re.compile(r"^[A-Za-z0-9_.,:/=+-]+$")
_READONLY_MAX_CHAIN_PARTS = 4
_READONLY_MAX_LOG_LINES = 500

_READONLY_NVIDIA_SMI_NO_VAL_FLAGS = frozenset(
    {"-L", "--list-gpus", "-q", "--query", "-a", "--all", "-h", "--help", "--version", "-x", "--xml-format"}
)
_READONLY_NVIDIA_SMI_VAL_FLAGS = frozenset(
    {"-i", "--id", "-d", "--display", "--query-gpu", "--query-compute-apps", "--query-apps", "--format"}
)
_READONLY_FREE_FLAGS = frozenset(
    {
        "-b",
        "-k",
        "-m",
        "-g",
        "-h",
        "-t",
        "-w",
        "-l",
        "--bytes",
        "--kilo",
        "--mega",
        "--giga",
        "--human",
        "--total",
        "--wide",
        "--lohi",
        "--si",
        "--tebi",
        "--gibi",
        "--mebi",
        "--kibi",
    }
)
_READONLY_DF_LONG_FLAGS = frozenset(
    {"--human-readable", "--si", "--print-type", "--inodes", "--portability", "--local", "--total"}
)
_READONLY_UPTIME_FLAGS = frozenset({"-p", "--pretty", "-s", "--since", "-V", "--version"})
_READONLY_UNAME_LONG_FLAGS = frozenset(
    {
        "--all",
        "--kernel-name",
        "--nodename",
        "--kernel-release",
        "--kernel-version",
        "--machine",
        "--processor",
        "--hardware-platform",
        "--operating-system",
    }
)
_READONLY_PS_SAFE_FIELDS = frozenset(
    {
        "pid",
        "ppid",
        "user",
        "uid",
        "pcpu",
        "%cpu",
        "pmem",
        "%mem",
        "rss",
        "vsz",
        "stat",
        "state",
        "etime",
        "time",
        "comm",
        "cmd",
        "args",
        "command",
    }
)
_READONLY_LS_LONG_FLAGS = frozenset({"--human-readable", "--color=never", "--time-style=iso"})


def _has_forbidden_path_segment(path_str: str) -> bool:
    """校验路径是否包含越界遍历、隐藏点文件/目录或敏感凭据关键词。"""
    if not path_str or len(path_str) > 256 or not path_str.startswith("/"):
        return True
    if ".." in path_str or "\\" in path_str or _READONLY_SENSITIVE_PATH_RE.search(path_str):
        return True
    return any(seg.startswith(".") for seg in path_str.split("/") if seg)


def _is_allowed_readonly_log_path(path_str: str) -> bool:
    """校验路径是否严格命中允许只读查看的远端服务日志文件白名单。"""
    if _has_forbidden_path_segment(path_str):
        return False
    return any(pat.match(path_str) for pat in _READONLY_ALLOWED_LOG_PATH_RES)


def _is_safe_readonly_ls_path(path_str: str) -> bool:
    """校验 `ls` 目标路径是否位于安全工作目录或白名单日志路径内。"""
    if _is_allowed_readonly_log_path(path_str):
        return True
    if _has_forbidden_path_segment(path_str):
        return False
    norm = path_str.rstrip("/") or "/"
    return norm in ("/content", "/var/log") or norm.startswith(("/content/", "/var/log/"))


def _is_safe_readonly_df_path(path_str: str) -> bool:
    """校验 `df` 目标路径是否位于只读磁盘挂载查询允许范围内。"""
    if path_str == "/":
        return True
    if _has_forbidden_path_segment(path_str):
        return False
    norm = path_str.rstrip("/") or "/"
    return norm in ("/content", "/tmp", "/root", "/var", "/var/log") or norm.startswith("/content/")


def _validate_nvidia_smi_args(args: list[str]) -> tuple[bool, str]:
    """校验 `nvidia-smi` 仅携带只读 GPU 状态查询参数。"""
    idx = 0
    while idx < len(args):
        tok = args[idx]
        if tok in _READONLY_NVIDIA_SMI_NO_VAL_FLAGS:
            idx += 1
            continue
        if tok in _READONLY_NVIDIA_SMI_VAL_FLAGS:
            if idx + 1 >= len(args) or not _READONLY_SAFE_VALUE_RE.match(args[idx + 1]):
                return False, f"nvidia-smi 参数值不合法: {tok}"
            idx += 2
            continue
        if "=" in tok:
            opt, val = tok.split("=", 1)
            if opt in _READONLY_NVIDIA_SMI_VAL_FLAGS and val and _READONLY_SAFE_VALUE_RE.match(val):
                idx += 1
                continue
        return False, f"nvidia-smi 包含非只读或未授权参数: {tok}"
    return True, ""


def _validate_free_args(args: list[str]) -> tuple[bool, str]:
    """校验 `free` 仅携带只读内存展示标志。"""
    for tok in args:
        if tok not in _READONLY_FREE_FLAGS:
            return False, f"free 包含未授权参数: {tok}"
    return True, ""


def _validate_df_args(args: list[str]) -> tuple[bool, str]:
    """校验 `df` 选项与目标路径位于磁盘只读查询白名单内。"""
    paths: list[str] = []
    for tok in args:
        if tok.startswith("--"):
            if tok in _READONLY_DF_LONG_FLAGS:
                continue
            if tok.startswith("--output=") and _READONLY_SAFE_VALUE_RE.match(tok.split("=", 1)[1]):
                continue
            return False, f"df 包含未授权长选项: {tok}"
        if tok.startswith("-") and len(tok) > 1:
            if not re.match(r"^-[hHkmTiPal]+$", tok):
                return False, f"df 包含未授权短选项: {tok}"
            continue
        if not _is_safe_readonly_df_path(tok):
            return False, f"df 目标路径不在安全只读白名单内: {tok}"
        paths.append(tok)
    if len(paths) > 4:
        return False, "df 目标路径数量超出限制"
    return True, ""


def _validate_uptime_args(args: list[str]) -> tuple[bool, str]:
    """校验 `uptime` 只读选项。"""
    for tok in args:
        if tok not in _READONLY_UPTIME_FLAGS:
            return False, f"uptime 包含未授权参数: {tok}"
    return True, ""


def _validate_uname_args(args: list[str]) -> tuple[bool, str]:
    """校验 `uname` 只读选项。"""
    for tok in args:
        if tok not in _READONLY_UNAME_LONG_FLAGS and not re.match(r"^-[asnrvmpio]+$", tok):
            return False, f"uname 包含未授权参数: {tok}"
    return True, ""


def _validate_ps_fields(raw_fields: str) -> bool:
    """校验 `ps -o / -eo / --format` 的列集合严格属于安全非敏感字段。"""
    fields = [f.strip().split("=", 1)[0].lower() for f in raw_fields.split(",") if f.strip()]
    return bool(fields) and all(f in _READONLY_PS_SAFE_FIELDS for f in fields)


def _consume_ps_valued_opt(args: list[str], idx: int) -> tuple[int, bool, str]:
    """解析 `ps` 带值选项（返回 `(步进长度, 是否触发UNIX -e/-A, 错误原因)`；步进 0 表示未匹配）。"""
    tok = args[idx]
    if tok in ("-o", "-eo", "--format"):
        if idx + 1 >= len(args):
            return 0, False, f"ps 缺少字段参数: {tok}"
        if not _validate_ps_fields(args[idx + 1]):
            return 0, False, f"ps 输出字段包含未授权或敏感列: {args[idx + 1]}"
        return 2, (tok == "-eo"), ""
    if tok.startswith("--format="):
        if not _validate_ps_fields(tok.split("=", 1)[1]):
            return 0, False, f"ps 输出字段包含未授权或敏感列: {tok}"
        return 1, False, ""
    if tok in ("-p", "--pid"):
        if idx + 1 >= len(args) or not re.match(r"^[0-9,]+$", args[idx + 1]):
            return 0, False, f"ps PID 参数不合法: {tok}"
        return 2, False, ""
    if tok.startswith("--pid="):
        if not re.match(r"^[0-9,]+$", tok.split("=", 1)[1]):
            return 0, False, f"ps PID 参数不合法: {tok}"
        return 1, False, ""
    if tok == "-C":
        if idx + 1 >= len(args) or not re.match(r"^[A-Za-z0-9_.-]+$", args[idx + 1]):
            return 0, False, "ps -C 进程名不合法"
        return 2, False, ""
    return 0, False, ""


def _validate_ps_args(args: list[str]) -> tuple[bool, str]:
    """校验 `ps` 参数，严禁触发 BSD 环境变量转储（如 `-auxe`/`-xe`）及跨 token 风格混用。"""
    idx = 0
    has_bsd_opt = False
    has_unix_e_or_a_opt = False
    while idx < len(args):
        tok = args[idx]
        if tok == "--no-headers":
            idx += 1
            continue
        if tok.startswith("--sort="):
            if tok.split("=", 1)[1].lstrip("+-") in _READONLY_PS_SAFE_FIELDS:
                idx += 1
                continue
            return False, f"ps 排序字段未授权: {tok}"
        step, marks_unix_e, err = _consume_ps_valued_opt(args, idx)
        if err:
            return False, err
        if step > 0:
            has_unix_e_or_a_opt = has_unix_e_or_a_opt or marks_unix_e
            idx += step
            continue
        # 带短横短选项严格收紧为纯 UNIX 安全标志（严禁出现 x/u/a，防止 procps 回退 BSD 解析转储 environ）
        if tok.startswith("-") and re.match(r"^-[efAwH]+$", tok):
            if "e" in tok or "A" in tok:
                has_unix_e_or_a_opt = True
            idx += 1
            continue
        # 无短横 BSD 选项严格禁止含 e（防止打印进程环境变量）
        if not tok.startswith("-") and re.match(r"^[auxw]+$", tok):
            has_bsd_opt = True
            idx += 1
            continue
        return False, f"ps 包含未授权参数或可能泄露环境变量的选项: {tok}"
    if has_bsd_opt and has_unix_e_or_a_opt:
        return False, "ps 禁止混用 BSD 无短横选项与 UNIX -e/-A 选项（防止解析回退泄露进程环境变量）"
    return True, ""


def _validate_ls_args(args: list[str]) -> tuple[bool, str]:
    """校验 `ls` 选项与路径严格限定在安全工作目录或白名单日志路径，禁止隐式枚举隐藏文件。"""
    paths: list[str] = []
    has_all_flag = False
    for tok in args:
        if tok in _READONLY_LS_LONG_FLAGS:
            continue
        if tok.startswith("-") and len(tok) > 1:
            if not re.match(r"^-[lhtrS1dFaA]+$", tok):
                return False, f"ls 包含未授权选项: {tok}"
            if "a" in tok or "A" in tok:
                has_all_flag = True
            continue
        if not _is_safe_readonly_ls_path(tok):
            return False, f"ls 目标路径不在只读白名单目录内: {tok}"
        paths.append(tok)
    if len(paths) > 4:
        return False, "ls 目标路径数量超出限制"
    if not paths and has_all_flag:
        return False, "ls 未指定安全目录时禁止列出隐藏文件 (-a/-A)"
    if has_all_flag and any(p.startswith(("/root/", "/tmp/")) for p in paths):
        return False, "禁止对 /root 或 /tmp 路径使用隐藏文件选项 (-a/-A)"
    return True, ""


def _validate_head_tail_args(cmd: str, args: list[str]) -> tuple[bool, str]:
    """校验 `head` / `tail` 仅按受限行数读取受控服务日志白名单文件。"""
    files: list[str] = []
    idx = 0
    while idx < len(args):
        tok = args[idx]
        if tok in ("-q", "--quiet"):
            idx += 1
            continue
        if tok == "-n":
            if idx + 1 >= len(args) or not re.match(r"^[1-9][0-9]*$", args[idx + 1]):
                return False, f"{cmd} -n 行数参数必须为正整数"
            if int(args[idx + 1]) > _READONLY_MAX_LOG_LINES:
                return False, f"{cmd} 单次最多允许查看 {_READONLY_MAX_LOG_LINES} 行"
            idx += 2
            continue
        m_n = re.match(r"^(?:-n|--lines=|-)([1-9][0-9]*)$", tok)
        if m_n:
            if int(m_n.group(1)) > _READONLY_MAX_LOG_LINES:
                return False, f"{cmd} 单次最多允许查看 {_READONLY_MAX_LOG_LINES} 行"
            idx += 1
            continue
        if tok.startswith("-"):
            return False, f"{cmd} 包含未授权选项: {tok}"
        if not _is_allowed_readonly_log_path(tok):
            return False, f"{cmd} 仅允许读取受控服务日志白名单路径，拒绝访问: {tok}"
        files.append(tok)
        idx += 1
    if not files or len(files) > 2:
        return False, f"{cmd} 必须指定 1~2 个受控服务日志路径"
    return True, ""


def _validate_du_args(args: list[str]) -> tuple[bool, str]:
    """校验 `du` 仅以有限深度统计 `/content` 下的非敏感目录。"""
    paths: list[str] = []
    idx = 0
    while idx < len(args):
        tok = args[idx]
        if tok == "-d":
            if idx + 1 >= len(args) or not re.match(r"^[0-3]$", args[idx + 1]):
                return False, "du -d 深度仅允许 0~3"
            idx += 2
            continue
        if re.match(r"^--max-depth=[0-3]$", tok) or re.match(r"^-[shHkmc]+$", tok):
            idx += 1
            continue
        if tok.startswith("-"):
            return False, f"du 包含未授权选项: {tok}"
        if _has_forbidden_path_segment(tok) or not (tok == "/content" or tok.startswith("/content/")):
            return False, f"du 仅允许统计 /content 下的安全目录: {tok}"
        paths.append(tok)
        idx += 1
    if not paths or len(paths) > 2:
        return False, "du 必须指定 1~2 个 /content 下的目录"
    return True, ""


def _validate_no_args_cmd(cmd: str, args: list[str]) -> tuple[bool, str]:
    """校验无参只读诊断命令不接受任意额外参数。"""
    if args:
        return False, f"{cmd} 只读白名单不允许附带额外参数"
    return True, ""


def _validate_date_args(args: list[str]) -> tuple[bool, str]:
    """校验 `date` 仅接受固定只读格式选项。"""
    if not args or all(tok in ("-u", "--utc", "-I", "-Is", "-R") for tok in args):
        return True, ""
    return False, "date 包含未授权参数"


# 表驱动命令分派字典：每个只读命令绑定独立短验证器
_READONLY_CMD_VALIDATORS: dict[str, Callable[[list[str]], tuple[bool, str]]] = {
    "nvidia-smi": _validate_nvidia_smi_args,
    "free": _validate_free_args,
    "df": _validate_df_args,
    "uptime": _validate_uptime_args,
    "uname": _validate_uname_args,
    "ps": _validate_ps_args,
    "ls": _validate_ls_args,
    "head": lambda a: _validate_head_tail_args("head", a),
    "tail": lambda a: _validate_head_tail_args("tail", a),
    "du": _validate_du_args,
    "pwd": lambda a: _validate_no_args_cmd("pwd", a),
    "whoami": lambda a: _validate_no_args_cmd("whoami", a),
    "nproc": lambda a: _validate_no_args_cmd("nproc", a),
    "lscpu": lambda a: _validate_no_args_cmd("lscpu", a),
    "date": _validate_date_args,
}


def _validate_single_readonly_bash_cmd(segment: str) -> tuple[bool, str]:
    """通过 `_READONLY_CMD_VALIDATORS` 分派表按严格白名单校验单条只读 Bash 命令。"""
    try:
        tokens = shlex.split(segment, posix=True)
    except ValueError as exc:
        return False, f"命令语法无法解析或存在未闭合引号: {exc}"
    if not tokens:
        return False, "命令内容为空"
    cmd = tokens[0]
    if "/" in cmd or "\\" in cmd:
        return False, f"只读白名单禁止通过路径形式调用可执行文件: {cmd}"
    validator = _READONLY_CMD_VALIDATORS.get(cmd)
    if validator is None:
        return False, f"命令 `{cmd}` 不在免审批只读诊断白名单内"
    return validator(tokens[1:])


def is_readonly_safe_remote_cmd(code: str, mode: str) -> tuple[bool, str]:
    """严格判定远端命令是否属于明确无副作用的只读诊断白名单命令。

    规则：
    1. 前置执行 `check_remote_code_safety` 硬黑名单，任何高危或凭据窥探直接返回 False；
    2. 仅允许 `mode == "bash"`，所有 Python 代码一律视为非只读（返回 False）；
    3. 严禁重定向（`>`/`>>`/`<`）、命令替换（`` ` ``/`$(`）、变量展开（`$`）、
       分号/管道/逻辑或（`;`/`|`/`||`）、后台运行（单 `&`）、多行命令、通配符探测（`*`/`?`/`[`）
       及任意解释器或网络下载工具；
    4. 仅当全部单命令（或 `&&` 连接的纯只读子命令，最多 4 段）均严格命中白名单时返回 `(True, "")`。
    """
    safe_ok, safe_reason = check_remote_code_safety(code, mode)
    if not safe_ok:
        return False, safe_reason

    if str(mode or "").strip().lower() != "bash":
        return False, "所有 Python 代码均视为非只读操作，需人工审批或显式授权"

    raw = str(code or "").strip()
    if not raw:
        return False, "待执行命令不能为空"
    if len(raw) > 512:
        return False, "只读诊断命令长度超出 512 字符白名单上限"

    for ch in _READONLY_FORBIDDEN_SHELL_CHARS:
        if ch in raw:
            label = repr(ch) if ch in ("\r", "\n", "\x00") else ch
            return False, f"只读诊断白名单禁止包含 Shell 控制/通配/重定向元字符: {label}"

    # 仅允许 `&&` 连接纯只读诊断命令，严禁单个 `&` 后台执行或异常连续 `&&&`
    if "&" in raw.replace("&&", ""):
        return False, "只读诊断白名单禁止后台符 `&` 或非法逻辑连接"

    segments = [seg.strip() for seg in raw.split("&&")]
    if not segments or any(not seg for seg in segments):
        return False, "存在空子命令段，拒绝按只读放行"
    if len(segments) > _READONLY_MAX_CHAIN_PARTS:
        return False, f"只读诊断链式子命令段数超过上限 ({_READONLY_MAX_CHAIN_PARTS})"

    for seg in segments:
        seg_ok, seg_reason = _validate_single_readonly_bash_cmd(seg)
        if not seg_ok:
            return False, seg_reason
    return True, ""


# =========================================================================
# 3. 脱敏与结构化辅助函数
# =========================================================================


def _redact_obj(value: Any, extra_secrets: tuple[str, ...] = ()) -> Any:
    """递归对字典/列表/字符串执行密钥脱敏，保留原始数据结构。"""
    if isinstance(value, str):
        return redact_secrets(value, extra_secrets)
    if isinstance(value, dict):
        return {str(k): _redact_obj(v, extra_secrets) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact_obj(item, extra_secrets) for item in value]
    return value


def _safe_json_dumps(value: Any, *, extra_secrets: tuple[str, ...] = (), max_len: int = 16000) -> str:
    """将对象安全脱敏并序列化为紧凑 JSON 文本。"""
    cleaned = _redact_obj(value, extra_secrets)
    try:
        raw = json.dumps(cleaned, ensure_ascii=False, default=str)
    except Exception:
        raw = str(cleaned)
    redacted = redact_secrets(raw, extra_secrets)
    if len(redacted) > max_len:
        return redacted[: max_len - 18] + "...(内容已截断)"
    return redacted


def _summarize_tool_result(res_obj: Any, extra_secrets: tuple[str, ...] = ()) -> tuple[bool, str]:
    """提取工具执行结果的成功标记与脱敏摘要。"""
    if isinstance(res_obj, dict):
        if res_obj.get("cancelled"):
            ok = False
        elif "ok" in res_obj:
            ok = bool(res_obj.get("ok"))
        elif res_obj.get("error"):
            ok = False
        else:
            ok = True
        for key in ("summary", "message", "reason", "error"):
            val = str(res_obj.get(key) or "").strip()
            if val:
                return ok, redact_secrets(val, extra_secrets)[:220]
        return ok, _safe_json_dumps(res_obj, extra_secrets=extra_secrets, max_len=220)
    txt = redact_secrets(str(res_obj or ""), extra_secrets).strip()
    return True, (txt[:220] if txt else "执行完成")


__all__ = [
    "_READONLY_CMD_VALIDATORS",
    "_redact_obj",
    "_safe_json_dumps",
    "_summarize_tool_result",
    "check_remote_code_safety",
    "is_readonly_safe_remote_cmd",
]
