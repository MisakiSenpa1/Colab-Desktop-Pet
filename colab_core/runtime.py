"""colab_core.runtime —— 运行时执行器、部署规划/进度状态机、多账号账本与结构化日志模块。"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, Callable

_PKG_DIR = Path(__file__).resolve().parents[1]
if str(_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR))

from colab_security import redact_secrets  # noqa: E402

CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
MAX_DIAG_LOG_BYTES = 512 * 1024

SUPPORTED_PROFILES = ("anima", "minimax", "drama3", "qwen_27b", "qwen_llm")
COMFY_PROFILES = ("anima", "minimax", "drama3")
LLM_PROFILES = ("qwen_27b", "qwen_llm")
SUPPORTED_BOOT_GPUS = ("T4", "L4", "A100", "G4", "H100")
SUPPORTED_LLM_QUANTS = ("UD-Q4_K_XL", "UD-IQ4_XS", "IQ3_XXS")
SUPPORTED_IDLE_STOP_MINUTES = (0, 60)
SCARCE_GPU_TTL_SEC = 600.0

LLM_QUANT_BASE_SHARDS: dict[str, int] = {"UD-Q4_K_XL": 4, "UD-IQ4_XS": 3, "IQ3_XXS": 2}
LLM_27B_BASE_SHARDS = 1
PROFILE_LABELS: dict[str, str] = {
    "anima": "Anima",
    "minimax": "MiniMax",
    "drama3": "Drama 3",
    "qwen_27b": "Qwen27B",
    "qwen_llm": "Qwen180B",
}

DEFAULT_MEASURED_GPU_RATES: dict[str, float] = {
    "T4:STD": 1.15,
    "L4:STD": 1.94,
    "G4:STD": 10.05,
}
DEFAULT_PREFS: dict[str, Any] = {
    "x": None,
    "y": None,
    "scale": 1.0,
    "on_top": True,
    "bubble_open": True,
    "left_flipped": False,
    "dock_edge": None,
    "edge_auto_hide": True,
    "poll_interval_sec": 60,
    "warn_hours": 10.0,
    "warn_balance": 50.0,
    "long_run_warn_minutes": 180,
    "auto_pull_outputs": True,
    "keep_alive": True,
    "idle_auto_stop_minutes": 60,
    "idle_archive_fail_policy": "stop_billing",
    "boot_profile": "anima",
    "boot_profiles": ["anima"],
    "boot_gpu": "L4",
    "boot_high_mem": False,
    "boot_minimax_r2v": True,
    "boot_minimax_pe": True,
    "boot_llm_quant": "UD-Q4_K_XL",
    "boot_llm_ctx": 32768,
    "boot_llm_mmproj": True,
    "boot_llm27_quant": "UD-Q4_K_XL",
    "boot_llm27_ctx": 65536,
    "boot_llm27_mmproj": True,
    "active_account_id": "",
    "colab_exe_path": "",
    "colab_py_path": "",
    "rclone_exe_path": "",
    "rclone_conf_path": "",
}
ACCOUNT_LEDGER_KEYS = (
    "date",
    "day_start_balance",
    "day_consumed_cu",
    "day_topup_cu",
    "last_balance",
    "last_delta",
    "baseline_balance",
    "active_since_ts",
    "active_endpoint",
    "session_start_balance",
    "last_rate_hourly",
    "last_assignments_count",
    "last_sessions",
    "last_eligible_gpus",
    "last_eligible_tpus",
    "last_updated_ts",
    "tunnel_cache",
    "managed_binding",
)

STEP_BANNER_RE = re.compile(r"===\s*\[(\d+)/(\d+)\]\s*(.+?)\s*===")
PCT_TOKEN_RE = re.compile(r"\((\d{1,3}(?:\.\d+)?)%\)")
DL_SPEED_RE = re.compile(r"DL:\s*([0-9.]+\s*[KMGTP]?i?B(?:/s)?)", re.IGNORECASE)
DEPLOY_DL_TOTAL_RE = re.compile(r"DEPLOY_DL_TOTAL:\s*(\d+)")
DEPLOY_PHASE_RE = re.compile(r"DEPLOY_PHASE\s+(\d+)/(\d+):\s*(\S+)")
DEPLOY_TUNNEL_LINE_RE = re.compile(
    r"(?:COMFYUI_TUNNEL_URL|SERVICE_TUNNEL_URL):\s*(https://(?:(?!api\.)[a-z0-9-]+\.trycloudflare\.com))"
)
DEPLOY_LLM_TUNNEL_LINE_RE = re.compile(
    r"LLM_TUNNEL_URL:\s*(https://(?:(?!api\.)[a-z0-9-]+\.trycloudflare\.com))"
)
DEPLOY_STEP_BASE_PCT = {1: 16, 2: 23, 3: 31, 4: 42, 5: 78, 6: 86, 7: 95}
_BRACKET_ONLY_RE = re.compile(r"^[\s()\[\]{},;~^.:]+$")


# =========================================================================
# 1. 子进程执行、取消令牌、任务调度与脱敏轮转日志
# =========================================================================

def resolve_diag_log_path(log_path: Path | None = None) -> Path:
    if log_path is not None:
        return Path(log_path)
    env_p = str(os.environ.get("COLAB_PET_DIAG_LOG_PATH") or "").strip()
    return Path(env_p) if env_p else (_PKG_DIR / "colab_pet" / "logs" / "colab_diag.log")


def clean_subprocess_env(extra_env: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env.setdefault("COLAB_CLI_REQUEST_TIMEOUT", "30")
    if extra_env:
        for k, v in extra_env.items():
            if v is not None and str(v).strip():
                env[str(k)] = str(v)
    return env


def kill_process_tree(proc: Any) -> None:
    if proc is None:
        return
    pid = getattr(proc, "pid", None)
    if os.name == "nt" and isinstance(pid, int) and pid > 0:
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                capture_output=True,
                creationflags=CREATE_NO_WINDOW,
                timeout=8,
            )
        except Exception:
            pass
    for m_name in ("terminate", "kill"):
        fn = getattr(proc, m_name, None)
        if callable(fn):
            try:
                fn()
            except Exception:
                pass


def extract_framed_json(text: str | None, marker: str) -> dict[str, Any] | None:
    if not text or not marker:
        return None
    m = re.search(rf"<<<{re.escape(marker)}>>>(.*?)<<<END_{re.escape(marker)}>>>", str(text), re.S)
    if not m or not m.group(1).strip():
        return None
    try:
        parsed = json.loads(m.group(1).strip())
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


class DiagLogger:
    """线程安全、带 512 KiB 轮转与自动凭据脱敏的结构化诊断日志器。"""

    def __init__(self, log_path: Path | None = None, *, max_bytes: int = MAX_DIAG_LOG_BYTES) -> None:
        self._explicit_path: Path | None = Path(log_path) if log_path is not None else None
        self.max_bytes = max(4096, int(max_bytes or MAX_DIAG_LOG_BYTES))
        self._lock = threading.Lock()

    @property
    def log_path(self) -> Path:
        return resolve_diag_log_path(self._explicit_path)

    def _rotate_locked(self, target: Path, incoming: int) -> None:
        try:
            if target.is_file() and (target.stat().st_size + incoming) > self.max_bytes:
                bak = target.with_name(target.name + ".1")
                if bak.exists():
                    bak.unlink(missing_ok=True)
                target.replace(bak)
        except OSError:
            pass

    def log(self, level: str, component: str, message: str, **fields: Any) -> None:
        lvl = str(level or "INFO").strip().upper()
        comp = str(component or "core").strip() or "core"
        msg_clean = redact_secrets(str(message or "").replace("\r", " ").replace("\n", " "))
        ctx = ""
        if fields:
            try:
                ctx = " | " + redact_secrets(json.dumps(fields, ensure_ascii=False, default=str))
            except Exception:
                pass
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        encoded = f"[{stamp}] [{lvl}] [{comp}] {msg_clean}{ctx}\n".encode("utf-8", errors="replace")
        target = self.log_path
        with self._lock:
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                self._rotate_locked(target, len(encoded))
                with open(target, "ab") as fp:
                    fp.write(encoded)
            except OSError:
                pass

    def info(self, component: str, message: str, **fields: Any) -> None:
        self.log("INFO", component, message, **fields)

    def warn(self, component: str, message: str, **fields: Any) -> None:
        self.log("WARN", component, message, **fields)

    def error(self, component: str, message: str, **fields: Any) -> None:
        self.log("ERROR", component, message, **fields)

    def read_tail_lines(self, max_lines: int = 200) -> list[str]:
        target = self.log_path
        bak = target.with_name(target.name + ".1")
        chunks: list[str] = []
        with self._lock:
            for p in (bak, target):
                if p.is_file():
                    try:
                        chunks.append(p.read_text(encoding="utf-8", errors="replace"))
                    except OSError:
                        pass
        if not chunks:
            return []
        lines = "\n".join(chunks).splitlines()[-max(1, int(max_lines)) :]
        return [redact_secrets(ln) for ln in lines]


_GLOBAL_DIAG_LOCK = threading.Lock()
_GLOBAL_DIAG_LOGGER: DiagLogger | None = None


def get_diag_logger(log_path: Path | None = None) -> DiagLogger:
    global _GLOBAL_DIAG_LOGGER
    if log_path is not None:
        return DiagLogger(log_path=log_path)
    with _GLOBAL_DIAG_LOCK:
        if _GLOBAL_DIAG_LOGGER is None:
            _GLOBAL_DIAG_LOGGER = DiagLogger()
        return _GLOBAL_DIAG_LOGGER


class CancelToken:
    """线程安全取消令牌，支持在取消发生时立即触发回调。"""

    def __init__(self) -> None:
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._callbacks: list[Callable[[], None]] = []

    @property
    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def register_callback(self, fn: Callable[[], None]) -> None:
        call_now = False
        with self._lock:
            if self._event.is_set():
                call_now = True
            else:
                self._callbacks.append(fn)
        if call_now:
            try:
                fn()
            except Exception:
                pass

    def cancel(self) -> bool:
        with self._lock:
            already = self._event.is_set()
            self._event.set()
            cbs = list(self._callbacks)
            self._callbacks.clear()
        if already:
            return False
        for cb in cbs:
            try:
                cb()
            except Exception:
                pass
        return True


@dataclass
class CommandResult:
    cmd: list[str]
    returncode: int
    stdout: str = ""
    stderr: str = ""
    elapsed_sec: float = 0.0
    attempts: int = 1
    timed_out: bool = False
    cancelled: bool = False
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out and not self.cancelled and not self.error

    @property
    def combined_output(self) -> str:
        return "\n".join(p for p in (self.stdout, self.stderr, self.error) if p).strip()

    @property
    def error_summary(self) -> str:
        if self.cancelled:
            return "操作已手动取消"
        if self.timed_out:
            return self.error or "外部命令执行超时"
        lines = [ln.strip() for ln in self.combined_output.splitlines() if ln.strip()]
        if lines:
            return redact_secrets(lines[-1])[:240]
        return redact_secrets(self.error or f"命令退出码 {self.returncode}")[:240]


class ProcessRunner:
    """统一同步与流式子进程执行器（集中超时、重试、取消杀树与脱敏日志）。"""

    def __init__(self, diag: DiagLogger | None = None) -> None:
        self.diag = diag or get_diag_logger()

    def run(
        self,
        cmd: list[Any],
        *,
        timeout: int = 25,
        retries: int = 1,
        retry_delay_sec: float = 1.5,
        retry_if: Callable[[CommandResult], bool] | None = None,
        input_text: str | None = None,
        extra_env: dict[str, str] | None = None,
        cancel_token: CancelToken | None = None,
        run_fn: Callable[..., subprocess.CompletedProcess[Any]] | None = None,
        task_name: str = "",
    ) -> CommandResult:
        cmd_list = [str(x) for x in cmd]
        cmd_safe = redact_secrets(" ".join(cmd_list))
        max_attempts = max(1, int(retries))
        tag = task_name or (Path(cmd_list[0]).name if cmd_list else "cmd")

        last_res = CommandResult(cmd=cmd_list, returncode=1)
        for attempt in range(1, max_attempts + 1):
            if cancel_token is not None and cancel_token.is_cancelled:
                return CommandResult(cmd=[redact_secrets(c) for c in cmd_list], returncode=1, cancelled=True, attempts=attempt, error="任务在执行前已取消")

            t0 = time.monotonic()
            try:
                if run_fn is not None:
                    proc = run_fn(cmd_list, timeout=timeout, extra_env=extra_env, input_text=input_text)
                else:
                    proc = subprocess.run(
                        cmd_list,
                        input=input_text,
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        timeout=timeout,
                        env=clean_subprocess_env(extra_env),
                        creationflags=CREATE_NO_WINDOW,
                    )
                res = CommandResult(
                    cmd=[redact_secrets(c) for c in cmd_list],
                    returncode=int(getattr(proc, "returncode", 1)),
                    stdout=redact_secrets(getattr(proc, "stdout", "") or ""),
                    stderr=redact_secrets(getattr(proc, "stderr", "") or ""),
                    elapsed_sec=round(time.monotonic() - t0, 3),
                    attempts=attempt,
                )
            except subprocess.TimeoutExpired as exc:
                out_raw = getattr(exc, "stdout", "") or ""
                err_raw = getattr(exc, "stderr", "") or ""
                if isinstance(out_raw, bytes):
                    out_raw = out_raw.decode("utf-8", errors="replace")
                if isinstance(err_raw, bytes):
                    err_raw = err_raw.decode("utf-8", errors="replace")
                res = CommandResult(
                    cmd=[redact_secrets(c) for c in cmd_list],
                    returncode=124,
                    stdout=redact_secrets(out_raw),
                    stderr=redact_secrets(err_raw),
                    elapsed_sec=round(time.monotonic() - t0, 3),
                    attempts=attempt,
                    timed_out=True,
                    error=f"命令执行超时 ({timeout}s)",
                )
            except Exception as exc:
                res = CommandResult(
                    cmd=[redact_secrets(c) for c in cmd_list],
                    returncode=1,
                    elapsed_sec=round(time.monotonic() - t0, 3),
                    attempts=attempt,
                    error=redact_secrets(str(exc)),
                )

            last_res = res
            if res.ok:
                self.diag.info("runner", f"执行成功: {tag}", attempt=attempt, elapsed_sec=res.elapsed_sec, cmd=cmd_safe[:180])
                return res
            should_retry = attempt < max_attempts and (retry_if(res) if retry_if is not None else True)
            self.diag.warn("runner", f"执行失败: {tag} | rc={res.returncode} | {res.error_summary}", attempt=attempt, will_retry=should_retry)
            if should_retry and retry_delay_sec > 0:
                time.sleep(retry_delay_sec)
            elif not should_retry:
                break
        return last_res

    def run_streaming(
        self,
        cmd: list[Any],
        *,
        on_line: Callable[[str], None] | None = None,
        timeout: float | None = None,
        extra_env: dict[str, str] | None = None,
        cancel_token: CancelToken | None = None,
        popen_fn: Callable[..., Any] | None = None,
        on_proc_start: Callable[[Any], None] | None = None,
        task_name: str = "",
    ) -> CommandResult:
        cmd_list = [str(x) for x in cmd]
        cmd_safe = redact_secrets(" ".join(cmd_list))
        tag = task_name or (Path(cmd_list[0]).name if cmd_list else "stream")
        t0 = time.monotonic()

        if cancel_token is not None and cancel_token.is_cancelled:
            return CommandResult(cmd=[redact_secrets(c) for c in cmd_list], returncode=1, cancelled=True, error="任务在启动前已取消")

        timed_out_flag = threading.Event()
        watchdog: threading.Timer | None = None
        try:
            proc = (popen_fn or subprocess.Popen)(
                cmd_list,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=clean_subprocess_env(extra_env),
                creationflags=CREATE_NO_WINDOW,
            )
            if on_proc_start is not None:
                on_proc_start(proc)
            if cancel_token is not None:
                cancel_token.register_callback(lambda: kill_process_tree(proc))
            if timeout is not None and float(timeout) > 0:
                def _on_timeout() -> None:
                    timed_out_flag.set()
                    kill_process_tree(proc)

                watchdog = threading.Timer(float(timeout), _on_timeout)
                watchdog.daemon = True
                watchdog.start()

            self.diag.info("runner", f"流式进程已启动: {tag}", cmd=cmd_safe[:180])
            buf = ""
            lines_collected: list[str] = []
            stdout_stream = getattr(proc, "stdout", None)
            if stdout_stream is not None:
                while True:
                    ch = stdout_stream.read(256)
                    if not ch:
                        if buf.strip():
                            for raw_ln in buf.splitlines():
                                sl = redact_secrets(raw_ln.strip())
                                if sl:
                                    lines_collected.append(sl)
                                    if on_line is not None:
                                        on_line(sl)
                        break
                    buf += ch.replace("\r", "\n")
                    while "\n" in buf:
                        raw_ln, buf = buf.split("\n", 1)
                        sl = redact_secrets(raw_ln.strip())
                        if sl:
                            lines_collected.append(sl)
                            if on_line is not None:
                                on_line(sl)
                try:
                    stdout_stream.close()
                except Exception:
                    pass

            rc = int(proc.wait())
            dt = round(time.monotonic() - t0, 3)
            cancelled = bool(cancel_token is not None and cancel_token.is_cancelled)
            timed_out = timed_out_flag.is_set()
            return CommandResult(
                cmd=[redact_secrets(c) for c in cmd_list],
                returncode=rc,
                stdout="\n".join(lines_collected),
                elapsed_sec=dt,
                timed_out=timed_out,
                cancelled=cancelled,
                error=f"流式执行超时 ({timeout}s)" if timed_out else ("任务已取消" if cancelled else ""),
            )
        except Exception as exc:
            err_txt = redact_secrets(str(exc))
            self.diag.error("runner", f"流式启动失败: {tag} | {err_txt}", cmd=cmd_safe[:180])
            return CommandResult(
                cmd=[redact_secrets(c) for c in cmd_list],
                returncode=1,
                elapsed_sec=round(time.monotonic() - t0, 3),
                cancelled=bool(cancel_token is not None and cancel_token.is_cancelled),
                error=err_txt,
            )
        finally:
            if watchdog is not None:
                watchdog.cancel()


class TaskStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMED_OUT = "timed_out"


@dataclass
class TaskRecord:
    name: str
    status: TaskStatus = TaskStatus.IDLE
    started_at: float = 0.0
    finished_at: float = 0.0
    error: str = ""
    result: Any = None
    run_count: int = 0
    cancel_token: CancelToken | None = field(default=None, repr=False)

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "started_at": round(self.started_at, 3) if self.started_at else 0.0,
            "finished_at": round(self.finished_at, 3) if self.finished_at else 0.0,
            "error": redact_secrets(self.error),
            "run_count": self.run_count,
        }


@dataclass
class TaskHandle:
    name: str
    accepted: bool
    token: CancelToken
    thread: threading.Thread | None = None
    reason: str = ""

    def cancel(self) -> bool:
        return self.token.cancel()

    def join(self, timeout: float | None = None) -> bool:
        if self.thread is not None:
            self.thread.join(timeout=timeout)
            return not self.thread.is_alive()
        return True


class TaskManager:
    """受控后台任务调度与异常捕捉台账。"""

    def __init__(self, diag: DiagLogger | None = None) -> None:
        self.diag = diag or get_diag_logger()
        self._lock = threading.RLock()
        self._records: dict[str, TaskRecord] = {}
        self._threads: dict[str, threading.Thread] = {}

    def get_record(self, name: str) -> dict[str, Any]:
        with self._lock:
            rec = self._records.get(name)
            return rec.to_public_dict() if rec else TaskRecord(name=name).to_public_dict()

    def snapshot(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return {k: v.to_public_dict() for k, v in self._records.items()}

    def submit(
        self,
        name: str,
        fn: Callable[[CancelToken], Any],
        *,
        exclusive: bool = True,
        on_error: Callable[[Exception], None] | None = None,
        daemon: bool = True,
    ) -> TaskHandle:
        token = CancelToken()
        with self._lock:
            cur = self._records.get(name)
            if exclusive and cur and cur.status in (TaskStatus.RUNNING, TaskStatus.CANCELLING):
                return TaskHandle(name=name, accepted=False, token=cur.cancel_token or token, thread=self._threads.get(name), reason=f"任务 {name} 正在执行中")
            rec = TaskRecord(
                name=name,
                status=TaskStatus.RUNNING,
                started_at=time.time(),
                run_count=(cur.run_count + 1) if cur else 1,
                cancel_token=token,
            )
            self._records[name] = rec

        def _worker() -> None:
            final_status, err_msg, res_val = TaskStatus.SUCCEEDED, "", None
            try:
                res_val = fn(token)
                if isinstance(res_val, CommandResult):
                    if res_val.cancelled:
                        final_status = TaskStatus.CANCELLED
                    elif res_val.timed_out:
                        final_status, err_msg = TaskStatus.TIMED_OUT, res_val.error_summary
                    elif not res_val.ok:
                        final_status, err_msg = TaskStatus.FAILED, res_val.error_summary
                elif token.is_cancelled:
                    final_status = TaskStatus.CANCELLED
            except TimeoutError as exc:
                final_status = TaskStatus.TIMED_OUT
                err_msg = redact_secrets(str(exc) or "任务执行超时")
                self.diag.error("task", f"任务超时: {name} | {err_msg}")
                if on_error:
                    on_error(exc)
            except Exception as exc:
                final_status = TaskStatus.CANCELLED if token.is_cancelled else TaskStatus.FAILED
                err_msg = redact_secrets(str(exc) or type(exc).__name__)
                self.diag.error("task", f"任务未捕获异常: {name} | {err_msg}")
                if on_error:
                    on_error(exc)

            with self._lock:
                cur_rec = self._records.get(name)
                if cur_rec is not None and cur_rec.cancel_token is token:
                    cur_rec.status = final_status
                    cur_rec.finished_at = time.time()
                    cur_rec.error = err_msg
                    cur_rec.result = res_val
                if self._threads.get(name) is threading.current_thread():
                    self._threads.pop(name, None)

        th = threading.Thread(target=_worker, name=f"colab-task-{name}", daemon=daemon)
        with self._lock:
            self._threads[name] = th
        th.start()
        return TaskHandle(name=name, accepted=True, token=token, thread=th)


# =========================================================================
# 2. 部署规范、防呆推荐、7 步进度流式解析与故障分类
# =========================================================================

def normalize_profiles(raw_profiles: Any, fallback_profile: str = "anima", *, strict: bool = False) -> list[str]:
    items = raw_profiles.split(",") if isinstance(raw_profiles, str) else (list(raw_profiles) if isinstance(raw_profiles, (list, tuple, set)) else [])
    cand_list: list[str] = []
    for item in items:
        p = str(item or "").strip().lower()
        if not p:
            continue
        if p not in SUPPORTED_PROFILES:
            if strict:
                raise ValueError(f"不支持的工作流生态 {p!r}，可选值: {', '.join(SUPPORTED_PROFILES)}")
            continue
        if p not in cand_list:
            cand_list.append(p)

    if not cand_list:
        fb = str(fallback_profile or "anima").strip().lower()
        if strict and fb not in SUPPORTED_PROFILES:
            raise ValueError(f"不支持的工作流生态 {fb!r}，可选值: {', '.join(SUPPORTED_PROFILES)}")
        cand_list = [fb if fb in SUPPORTED_PROFILES else "anima"]
    if "qwen_27b" in cand_list and "qwen_llm" in cand_list:
        cand_list = [p for p in cand_list if p != "qwen_27b"]
    return [p for p in SUPPORTED_PROFILES if p in cand_list]


def parse_profiles_cli_arg(profile_arg: str | None, profiles_arg: str | None = None) -> list[str]:
    if profiles_arg and str(profiles_arg).strip():
        return normalize_profiles(str(profiles_arg), strict=True)
    if profile_arg and str(profile_arg).strip():
        return normalize_profiles(str(profile_arg), strict=True)
    return ["anima"]


def select_default_gpu(profiles: list[str]) -> str:
    profs = normalize_profiles(profiles)
    if "qwen_llm" in profs or "minimax" in profs or "drama3" in profs or len(profs) >= 2:
        return "G4"
    return "L4"


def resolve_high_mem(gpu: str, requested_high_mem: bool) -> bool:
    return False if str(gpu or "").strip().upper() in ("L4", "G4") else bool(requested_high_mem)


def profile_bootstrap_spec(
    profile: str,
    *,
    pkg_dir: Path = _PKG_DIR,
    anima_script: Path | None = None,
    minimax_script: Path | None = None,
    drama3_script: Path | None = None,
    qwen_llm_script: Path | None = None,
) -> tuple[Path, int]:
    p = str(profile or "anima").strip().lower()
    if p in LLM_PROFILES:
        return Path(qwen_llm_script or (pkg_dir / "colab_bootstrap_qwen_llm.py")), 2400
    if p == "minimax":
        return Path(minimax_script or (pkg_dir / "colab_bootstrap_minimax_h3.py")), 2400
    if p == "drama3":
        return Path(drama3_script or (pkg_dir / "colab_bootstrap_drama3.py")), 1800
    return Path(anima_script or (pkg_dir / "colab_bootstrap_anima.py")), 600


def default_deploy_state() -> dict[str, Any]:
    return {
        "active": False,
        "profile": "anima",
        "gpu": "L4",
        "high_mem": False,
        "step_idx": 0,
        "step_total": 7,
        "stage_title": "",
        "sub_detail": "",
        "pct": 0,
        "started_ts": 0,
        "elapsed_sec": 0,
        "error_msg": "",
        "stockout_gpu": None,
        "suggest_gpu": None,
    }


def format_elapsed_cn(seconds: int) -> str:
    sec = max(0, int(seconds))
    return f"{sec}秒" if sec < 60 else f"{sec // 60}分{sec % 60:02d}秒"


def calc_phase_dl_total(
    phase_prof: str,
    llm_quant: str = "UD-Q4_K_XL",
    llm_mmproj: bool = True,
    llm27_mmproj: bool = True,
    minimax_r2v: bool = True,
    minimax_pe: bool = True,
) -> int:
    if phase_prof == "qwen_27b":
        return LLM_27B_BASE_SHARDS + (1 if llm27_mmproj else 0)
    if phase_prof == "qwen_llm":
        return LLM_QUANT_BASE_SHARDS.get(llm_quant, 4) + (1 if llm_mmproj else 0)
    if phase_prof == "minimax":
        return 7 + (1 if minimax_pe else 0) + (2 if minimax_r2v else 0)
    return 3


def scale_phase_pct(local_pct: int, phase_idx: int, phase_total: int) -> int:
    tot = max(1, int(phase_total))
    if tot <= 1:
        return max(0, min(100, int(local_pct)))
    idx = max(1, min(tot, int(phase_idx)))
    lo = 12.0 + (idx - 1) * (86.0 / tot)
    hi = 98.0 if idx >= tot else (12.0 + idx * (86.0 / tot))
    frac = max(0.0, min(1.0, (float(local_pct) - 12.0) / 88.0))
    return int(round(lo + frac * (hi - lo)))


def consume_deploy_line(
    raw_line: str,
    st: dict[str, Any],
    ctx: dict[str, Any],
    prefs: dict[str, Any],
    *,
    tunnel_pattern: str = "",
) -> tuple[str | None, bool]:
    """解析单行部署输出并原地推进 `st` 与阶段上下文 `ctx`，返回 `(tunnel_url, should_warm_meta)`。"""
    line = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", str(raw_line or "")).strip()
    if not line:
        return None, False

    if tunnel_pattern:
        tun_re = re.compile(rf"(?:COMFYUI_TUNNEL_URL|SERVICE_TUNNEL_URL):\s*({tunnel_pattern})")
        ltun_re = re.compile(rf"LLM_TUNNEL_URL:\s*({tunnel_pattern})")
    else:
        tun_re = DEPLOY_TUNNEL_LINE_RE
        ltun_re = DEPLOY_LLM_TUNNEL_LINE_RE

    def _sp(local_p: int) -> int:
        return scale_phase_pct(local_p, int(ctx.get("phase_idx") or 1), int(ctx.get("phase_total") or 1))

    cur_pct = int(st.get("pct") or 0)
    cur_step = int(st.get("step_idx") or 0)
    total_steps = max(1, int(st.get("step_total") or 7))
    gpu = str(st.get("gpu") or "L4")
    is_llm = bool(ctx.get("phase_is_llm"))

    m_phase = DEPLOY_PHASE_RE.search(line)
    if m_phase:
        p_idx, p_tot = max(1, int(m_phase.group(1))), max(1, int(m_phase.group(2)))
        p_prof = m_phase.group(3).strip().lower()
        ctx.update({
            "phase_idx": p_idx,
            "phase_total": p_tot,
            "phase_prof": p_prof,
            "phase_is_llm": p_prof in LLM_PROFILES,
            "dl_done": 0,
            "dl_name": "",
            "dl_total": calc_phase_dl_total(
                p_prof,
                str(prefs.get("boot_llm_quant") or "UD-Q4_K_XL"),
                bool(prefs.get("boot_llm_mmproj", True)),
                bool(prefs.get("boot_llm27_mmproj", True)),
                bool(prefs.get("boot_minimax_r2v", True)),
                bool(prefs.get("boot_minimax_pe", True)),
            ),
        })
        st["step_idx"], st["step_total"] = 0, 7
        if p_tot > 1:
            p_name = PROFILE_LABELS.get(p_prof, "Anima")
            st["stage_title"] = f"阶段 {p_idx}/{p_tot} · {p_name}"
            st["sub_detail"] = f"正在启动 {p_name} 云端部署..."
            st["pct"] = max(cur_pct, _sp(14))
        return None, False

    m_ltun = ltun_re.search(line)
    if m_ltun:
        ctx["captured_llm_tunnel"] = m_ltun.group(1).strip()
        return None, False

    m_tot = DEPLOY_DL_TOTAL_RE.search(line)
    if m_tot:
        if int(m_tot.group(1)) > 0:
            ctx["dl_total"] = int(m_tot.group(1))
        return None, False

    m_tun = tun_re.search(line)
    if m_tun:
        cand_url = m_tun.group(1).strip()
        p_idx, p_tot = int(ctx.get("phase_idx") or 1), int(ctx.get("phase_total") or 1)
        if p_idx < p_tot:
            if is_llm and not ctx.get("captured_llm_tunnel"):
                ctx["captured_llm_tunnel"] = cand_url
            st["step_idx"] = total_steps
            st["pct"] = max(cur_pct, _sp(100))
            st["stage_title"] = f"阶段 {p_idx}/{p_tot} 就绪"
            st["sub_detail"] = "前一服务就绪，继续部署下一工作流..."
            return None, False
        st["step_idx"], st["pct"] = total_steps, 100
        st["stage_title"] = f"[{total_steps}/{total_steps}] 部署全部完成"
        st["sub_detail"] = f"隧道已打通: {cand_url.replace('https://', '')}"
        should_warm = not bool(ctx.get("meta_warmed"))
        ctx["meta_warmed"] = True
        return cand_url, should_warm

    is_27b = ctx.get("phase_prof") == "qwen_27b"
    p_tot = int(ctx.get("phase_total") or 1)
    m_step = STEP_BANNER_RE.search(line)
    if m_step:
        idx, tot = int(m_step.group(1)), max(1, int(m_step.group(2)))
        raw_title = m_step.group(3).strip()
        max_len = 14 if p_tot > 1 else 18
        if len(raw_title) > max_len:
            raw_title = raw_title[: max_len - 1] + "…"
        base_pct = DEPLOY_STEP_BASE_PCT.get(idx, min(95, int(round(12 + idx * 84.0 / tot))))
        phase_tag = f"P{ctx.get('phase_idx', 1)}/{p_tot} " if p_tot > 1 else ""
        st.update({
            "step_idx": idx,
            "step_total": tot,
            "stage_title": f"{phase_tag}[{idx}/{tot}] {raw_title}",
            "pct": max(cur_pct, _sp(base_pct)),
        })
        if is_llm:
            llm_hints = {
                3: "正在准备 llama-server CUDA 推理引擎...",
                4: "正在高速拉取 Qwen3.8 GGUF 量化分片...",
                5: "正在生成访问密钥与全卡推理参数..." if is_27b else "正在生成访问密钥与 PLE/MoE 卸载参数...",
                6: "正在加载 27B 模型并启动 llama-server..." if is_27b else "正在加载 180B 模型并启动 llama-server...",
                7: "正在打通 Cloudflared 隧道并写入服务契约...",
            }
            st["sub_detail"] = llm_hints.get(idx, raw_title)
        else:
            comfy_hints = {
                4: "正在连接模型源并校验本地缓存...",
                5: "正在拉取 LoRA 与工作流配置...",
                6: "正在拉起 ComfyUI 服务...",
                7: "正在申请 Cloudflared 公网隧道...",
            }
            st["sub_detail"] = comfy_hints.get(idx, raw_title)
        return None, False

    low = line.lower()
    if "正在释放旧实例" in line:
        st.update({"pct": max(cur_pct, 3), "stage_title": "清理旧会话", "sub_detail": "正在停卡释放旧算力分配..."})
    elif "creating session" in low or "开启 colab 实例" in line:
        st.update({"pct": max(cur_pct, 6), "stage_title": f"申请 {gpu} 实例", "sub_detail": f"正在向 Colab 请求分配 {gpu} GPU..."})
    elif "session ready" in low:
        st.update({"pct": max(cur_pct, 11), "stage_title": "实例就绪", "sub_detail": f"{gpu} 已分配，准备上传云盘凭据..."})
    elif "跳过 colab new" in line:
        st.update({"pct": max(cur_pct, 11), "stage_title": "复用活跃实例", "sub_detail": "沿用现存运行时，上传配置并增量部署..."})
    elif "colab.exe upload" in low or ("upload" in low and "rclone.conf" in low):
        st.update({"pct": max(cur_pct, 13), "stage_title": "注入云盘配置", "sub_detail": "正在握手内核并启动自举..."})
    elif cur_step == 1:
        if "自动降级" in line or "降级为" in line:
            st.update({"sub_detail": line[:38], "pct": max(cur_pct, _sp(20))})
        elif "gpu 硬件信息:" in line or line.startswith("GPU:"):
            st.update({"sub_detail": f"已就绪: {line.split(':', 1)[1].strip()[:36]}", "pct": max(cur_pct, _sp(19))})
    elif cur_step == 2 and any(k in line for k in ("已就绪", "安装完成", "工具就绪")):
        st.update({"pct": max(cur_pct, _sp(27)), "sub_detail": "传输与隧道基础工具已备齐"})
    elif cur_step == 3:
        if is_llm:
            if "cmake" in low or "编译" in line:
                st.update({"sub_detail": "正在编译 llama-server CUDA 内核...", "pct": max(cur_pct, _sp(36))})
            elif "cuda0" in low or ("llama-server" in low and ("就绪" in line or "完成" in line)):
                st.update({"sub_detail": "llama-server CUDA 推理引擎已就绪", "pct": max(cur_pct, _sp(40))})
            elif any(k in line for k in ("预编译", "下载", "拉取")):
                st.update({"sub_detail": "正在拉取 llama-server CUDA 二进制包...", "pct": max(cur_pct, _sp(34))})
        else:
            if "git" in low and "clone" in low:
                st.update({"sub_detail": "正在克隆官方 ComfyUI 仓库...", "pct": max(cur_pct, _sp(33))})
            elif "pip" in low and "install" in low:
                st.update({"sub_detail": "正在安装依赖（保留预装 CUDA）...", "pct": max(cur_pct, _sp(36))})
            elif "依赖安装完成" in line or "comfy_kitchen 已就绪" in line:
                st.update({"sub_detail": "ComfyUI 依赖与 CUDA 环境就绪", "pct": max(cur_pct, _sp(40))})
    elif cur_step == 4:
        m_start = re.search(r"(?:开始下载底座文件:|\[开始下载\])\s*([^\s(]+)", line)
        if m_start:
            fname = PurePosixPath(m_start.group(1)).name
            dl_name = fname[:20] + ("…" if len(fname) > 20 else "")
            ctx["dl_name"] = dl_name
            st["sub_detail"] = f"下载 {dl_name}..."
            return None, False
        dl_tot = max(1, int(ctx.get("dl_total") or 3))
        if any(k in line for k in ("已存在且字节数匹配", "校验通过:", "[已就绪]", "[完成]")):
            dl_done = min(dl_tot, int(ctx.get("dl_done") or 0) + 1)
            ctx["dl_done"] = dl_done
            st["pct"] = max(cur_pct, _sp(min(76, 42 + int(round((dl_done / dl_tot) * 34)))))
            st["sub_detail"] = line.replace("已存在且字节数匹配，跳过:", "已缓存")[:38]
            return None, False
        m_pct = PCT_TOKEN_RE.search(line)
        if m_pct:
            file_pct = min(100.0, max(0.0, float(m_pct.group(1))))
            dl_done = int(ctx.get("dl_done") or 0)
            ratio = min(1.0, (dl_done + file_pct / 100.0) / dl_tot)
            st["pct"] = max(cur_pct, _sp(min(76, 42 + int(round(ratio * 34)))))
            m_spd = DL_SPEED_RE.search(line)
            spd = f" · {m_spd.group(1)}" if m_spd else ""
            prefix = f"{ctx.get('dl_name')} " if ctx.get("dl_name") else "拉取模型 "
            st["sub_detail"] = f"{prefix}({int(round(file_pct))}%{spd})"
    elif cur_step == 5:
        if is_llm and any(k in line for k in ("sk-pet-", "鉴权", "参数", "卸载")):
            st.update({"sub_detail": "安全鉴权密钥与推理参数已就绪", "pct": max(cur_pct, _sp(83))})
        elif "LoRA 就绪:" in line or "工作流就绪:" in line:
            st.update({"sub_detail": line[:38], "pct": max(cur_pct, _sp(83))})
    elif cur_step == 6:
        if is_llm:
            if any(k in line for k in ("已就绪", "模型就绪", "复用")):
                st.update({"sub_detail": "Qwen3.8 推理引擎加载完成", "pct": max(cur_pct, _sp(93))})
            elif any(k in low for k in ("llama-server", "18080", "8080", "health", "加载")):
                st.update({"sub_detail": "等待 llama-server 18080 端口加载就绪...", "pct": max(cur_pct, _sp(89))})
        else:
            if "已后台拉起 ComfyUI" in line or "已拉起 ComfyUI" in line:
                st.update({"sub_detail": "等待 ComfyUI 8188 端口响应...", "pct": max(cur_pct, _sp(89))})
            elif "ComfyUI 就绪" in line or "已就绪" in line:
                st.update({"sub_detail": "ComfyUI 引擎启动完成", "pct": max(cur_pct, _sp(93))})
    elif cur_step == 7 and "cloudflared" in low and any(k in line for k in ("拉起", "启动", "隧道")):
        st.update({"sub_detail": "等待 Cloudflare 分配公网域名...", "pct": max(cur_pct, _sp(97))})
    return None, False


def recommend_fallback_gpu(
    failed_gpu: str,
    profile: str,
    eligible_gpus: list[str] | None,
    scarce_gpus: dict[str, Any],
    profiles: list[str] | None = None,
) -> str | None:
    failed_up = str(failed_gpu or "").upper()
    profs = normalize_profiles(profiles, fallback_profile=profile)
    has_comfy = any(p in COMFY_PROFILES for p in profs)
    if "qwen_llm" in profs or ("qwen_27b" in profs and has_comfy):
        order: tuple[str, ...] = ("G4", "A100", "H100")
    elif profs == ["qwen_27b"]:
        order = ("L4", "A100", "G4", "H100", "T4")
    elif any(p in ("minimax", "drama3") for p in profs):
        order = ("G4", "A100", "H100", "L4", "T4")
    else:
        order = ("L4", "A100", "T4", "G4", "H100")

    allowed = {str(g).upper() for g in eligible_gpus if g} if isinstance(eligible_gpus, list) and eligible_gpus else None
    scarce_set = {str(k).upper() for k in (scarce_gpus or {}).keys()}
    for cand in order:
        if cand != failed_up and cand not in scarce_set and (allowed is None or cand in allowed):
            return cand
    return None


def _is_traceback_noise_line(line: str, in_traceback_frame: bool) -> bool:
    s = line.strip()
    if not s:
        return True
    low = s.lower()
    if s.startswith(("Traceback (most recent call last)", "During handling of the above exception", 'File "')) or _BRACKET_ONLY_RE.match(s):
        return True
    if low == "raise" or low.startswith("raise "):
        return True
    if in_traceback_frame and ("output=stdout" in low or "stderr=stderr" in low or "redact_secrets(" in low or s.endswith(",")):
        return True
    return False


def _pick_meaningful_tail_error(lines: list[str]) -> str:
    priority = ("[错误]", "RuntimeError:", "TimeoutError:", "ValueError:", "FileNotFoundError:", "ConnectionError:", "[colab]")
    for raw in reversed(lines):
        s = redact_secrets(str(raw or "").strip())
        if s and not _is_traceback_noise_line(s, False) and any(s.startswith(p) or p in s for p in priority):
            return s
    for raw in reversed(lines):
        s = redact_secrets(str(raw or "").strip())
        if not s or _is_traceback_noise_line(s, True):
            continue
        low = s.lower()
        if ("calledprocesserror" in low and len(lines) > 2) or (re.match(r"^[a-z_]+=", low) and any(k in low for k in ("cmd", "returncode", "stdout"))):
            continue
        return s
    return "部署异常中止（详情见诊断日志）"


def classify_deploy_failure(lines: list[str], gpu: str, pre_assignments: int, profile: str, recreate: bool) -> dict[str, Any]:
    cleaned = [redact_secrets(str(ln or "")).strip() for ln in lines if ln and str(ln).strip()]
    joined_low = "\n".join(cleaned).lower()
    gpu_up = str(gpu or "").upper()

    if "backend rejected accelerator" in joined_low or "not have quota or entitlement" in joined_low:
        return {"kind": "entitlement", "stockout_gpu": gpu_up, "reason": f"当前账号配额暂不支持 {gpu_up}", "message": f"⛔ 当前 Colab 账号无 {gpu_up} 配额或等级权限"}

    if "allocation refused" in joined_low or "toomanyassignments" in joined_low or ("412" in joined_low and "precondition" in joined_low):
        if pre_assignments > 0 and not recreate:
            return {"kind": "limit_or_stockout", "stockout_gpu": None, "reason": "已有运行中实例或目标卡高峰期名额紧张", "message": f"⚠️ 分配 {gpu_up} 被拒：并发上限或实例待纳管，可点增量部署或换卡"}
        return {"kind": "stockout", "stockout_gpu": gpu_up, "reason": f"高峰期 {gpu_up} 算力容量不足", "message": f"⚠️ 高峰期 {gpu_up} 算力紧缺无法分配"}

    is_read_timeout = any(k in joined_low for k in ("read timed out", "readtimeout", "connecttimeout"))
    is_capacity = any(k in joined_low for k in ("resourceexhausted", "out of capacity", "no backend", "cannot connect to gpu backend", "failed to assign")) or ("503" in joined_low and "assign" in joined_low) or (is_read_timeout and any(k in joined_low for k in ("assign", "creating session", " new ", "colab.research.google.com"))) or ("timeout" in joined_low and "creating session" in joined_low)
    if is_capacity:
        if is_read_timeout:
            return {"kind": "stockout", "stockout_gpu": gpu_up, "reason": f"分配 {gpu_up} 接口响应超时", "message": f"⚠️ 请求分配 {gpu_up} 超时，若后台已开机可直接点增量部署复用"}
        return {"kind": "stockout", "stockout_gpu": gpu_up, "reason": f"高峰期 {gpu_up} 分配拥堵/无空闲算力", "message": f"⚠️ 高峰期 {gpu_up} 无空闲卡或响应超时"}

    for ln in reversed(cleaned):
        if "磁盘空间不足" in ln or "显存不足" in ln:
            return {"kind": "error", "stockout_gpu": None, "reason": ln, "message": f"⚠️ {ln[:60]}"}
        if "llama-server" in ln and any(k in ln for k in ("超时", "失败", "异常退出")):
            return {"kind": "error", "stockout_gpu": None, "reason": "llama-server 启动异常", "message": f"⚠️ {ln[:58]}"}
        if "等待 ComfyUI 就绪超时" in ln:
            return {"kind": "error", "stockout_gpu": None, "reason": "ComfyUI 启动超时", "message": "⚠️ 等待 ComfyUI 启动超时，可尝试重新增量部署"}
        if "WebSocket" in ln or "握手超时" in ln:
            return {"kind": "error", "stockout_gpu": None, "reason": "Colab 内核通信超时", "message": "⚠️ 远端内核握手超时，可重试或停卡重开"}

    tail = _pick_meaningful_tail_error(cleaned)
    if len(tail) > 52:
        tail = tail[:49] + "..."
    return {"kind": "error", "stockout_gpu": None, "reason": tail, "message": f"⚠️ {profile} 部署异常: {tail}"}


def build_ctl_up_args(
    *,
    python_exe: str,
    ctl_script: Path,
    session: str,
    profiles: list[str],
    gpu: str,
    high_mem: bool = False,
    recreate: bool = False,
    endpoint: str = "",
    minimax_r2v: bool = True,
    minimax_pe: bool = True,
    llm_quant: str = "UD-Q4_K_XL",
    llm_ctx: int = 32768,
    llm_mmproj: bool = True,
    llm27_quant: str = "UD-Q4_K_XL",
    llm27_ctx: int = 65536,
    llm27_mmproj: bool = True,
    cohost_comfy: bool = False,
    ctl_prefix: list[str] | None = None,
) -> list[str]:
    norm_profs = normalize_profiles(profiles)
    profile = norm_profs[0]
    llm_prof = next((p for p in norm_profs if p in LLM_PROFILES), None)
    base = list(ctl_prefix) if ctl_prefix else [python_exe, "-u", str(ctl_script)]
    cmd = [*base, "up", "-s", session]
    cmd.extend(["--profiles", ",".join(norm_profs)] if len(norm_profs) >= 2 else ["--profile", profile])
    cmd.extend(["--gpu", str(gpu).strip().upper()])
    if endpoint:
        cmd.extend(["--endpoint", str(endpoint).strip()])
    if resolve_high_mem(gpu, high_mem):
        cmd.append("--high-mem")
    if recreate:
        cmd.append("--recreate")
    if cohost_comfy and len(norm_profs) < 2 and llm_prof is not None:
        cmd.append("--cohost-comfy")
    if "minimax" in norm_profs:
        if not minimax_r2v and not minimax_pe:
            cmd.append("--minimax-lite")
        else:
            if not minimax_r2v:
                cmd.append("--no-r2v")
            if not minimax_pe:
                cmd.append("--no-pe")
    if llm_prof == "qwen_27b":
        cmd.extend(["--llm-quant", llm27_quant, "--llm-ctx", str(int(llm27_ctx))])
        if not llm27_mmproj:
            cmd.append("--no-mmproj")
    elif llm_prof == "qwen_llm":
        cmd.extend(["--llm-quant", llm_quant, "--llm-ctx", str(int(llm_ctx))])
        if not llm_mmproj:
            cmd.append("--no-mmproj")
    return cmd


# =========================================================================
# 3. 多账号账本、偏好清洗与遥测解析
# =========================================================================

def default_baseline(balance: float) -> float:
    return 100.0 if balance <= 0 else float(max(100, int(math.ceil(balance / 100.0) * 100)))


def default_state_dict() -> dict[str, Any]:
    return {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "day_start_balance": None,
        "day_consumed_cu": 0.0,
        "day_topup_cu": 0.0,
        "last_balance": None,
        "last_delta": 0.0,
        "baseline_balance": 1000.0,
        "active_since_ts": None,
        "active_endpoint": None,
        "session_start_balance": None,
        "last_rate_hourly": 0.0,
        "last_assignments_count": 0,
        "last_sessions": [],
        "last_eligible_gpus": None,
        "last_eligible_tpus": None,
        "last_updated_ts": None,
        "measured_gpu_rates": dict(DEFAULT_MEASURED_GPU_RATES),
        "tunnel_cache": {},
        "managed_binding": None,
        "account_profiles": [],
        "account_ledgers": {},
        "onboarding_cache": {"gdrive_free_gb": None, "gdrive_total_gb": None, "gdrive_checked_ts": None},
        "prefs": dict(DEFAULT_PREFS),
    }


def sanitize_prefs_dict(prefs: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(DEFAULT_PREFS)
    for k in DEFAULT_PREFS:
        if k in prefs:
            cleaned[k] = prefs[k]

    prof = str(cleaned.get("boot_profile") or "anima").strip().lower()
    fallback_p = prof if prof in SUPPORTED_PROFILES else "anima"
    raw_profiles = prefs.get("boot_profiles") if "boot_profiles" in prefs else [fallback_p]
    norm_profiles = normalize_profiles(raw_profiles, fallback_profile=fallback_p)
    cleaned["boot_profiles"] = norm_profiles
    cleaned["boot_profile"] = norm_profiles[0]

    for q_key, ctx_key, mm_key, def_ctx in (
        ("boot_llm_quant", "boot_llm_ctx", "boot_llm_mmproj", 32768),
        ("boot_llm27_quant", "boot_llm27_ctx", "boot_llm27_mmproj", 65536),
    ):
        q_val = str(cleaned.get(q_key) or "UD-Q4_K_XL").strip()
        cleaned[q_key] = q_val if q_val in SUPPORTED_LLM_QUANTS else "UD-Q4_K_XL"
        try:
            cv = int(cleaned.get(ctx_key) or def_ctx)
        except (TypeError, ValueError):
            cv = def_ctx
        cleaned[ctx_key] = max(4096, min(262144, cv))
        cleaned[mm_key] = bool(cleaned.get(mm_key, True))

    cleaned["keep_alive"] = bool(cleaned.get("keep_alive", True))
    try:
        idle_mins = int(cleaned.get("idle_auto_stop_minutes", 60))
    except (TypeError, ValueError):
        idle_mins = 60
    cleaned["idle_auto_stop_minutes"] = idle_mins if idle_mins in SUPPORTED_IDLE_STOP_MINUTES else 60
    pol = str(cleaned.get("idle_archive_fail_policy") or "stop_billing").strip().lower()
    cleaned["idle_archive_fail_policy"] = pol if pol in ("stop_billing", "keep_alive") else "stop_billing"
    for path_key in ("active_account_id", "colab_exe_path", "colab_py_path", "rclone_exe_path", "rclone_conf_path"):
        cleaned[path_key] = str(cleaned.get(path_key) or "").strip()[:512]
    return cleaned


def load_pet_state_dict(
    state_path: Path,
    *,
    is_allowed_tunnel_fn: Callable[[str], bool],
    sanitize_label_fn: Callable[[str, str], str],
) -> dict[str, Any]:
    """安全加载并逐字段校验清洗 `pet_state.json`，拦截过期或非法结构。"""
    default = default_state_dict()
    if not state_path.exists():
        return default
    try:
        raw = json.loads(state_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return default
        merged = dict(default)
        for key in default:
            if key == "prefs" and isinstance(raw.get("prefs"), dict):
                merged["prefs"] = sanitize_prefs_dict({**DEFAULT_PREFS, **raw["prefs"]})
            elif key == "measured_gpu_rates" and isinstance(raw.get("measured_gpu_rates"), dict):
                cleaned_rates = dict(DEFAULT_MEASURED_GPU_RATES)
                valid_keys = {f"{g}:{s}" for g in SUPPORTED_BOOT_GPUS for s in ("STD", "HM") if not (g == "L4" and s == "HM")}
                for rk, rv in raw["measured_gpu_rates"].items():
                    if rk in valid_keys:
                        try:
                            fv = float(rv)
                            if 0.02 < fv < 200.0:
                                cleaned_rates[rk] = round(fv, 2)
                        except (TypeError, ValueError):
                            pass
                merged["measured_gpu_rates"] = cleaned_rates
            elif key == "tunnel_cache" and isinstance(raw.get("tunnel_cache"), dict):
                merged["tunnel_cache"] = {
                    ep: val.strip().rstrip("/")
                    for ep, val in raw["tunnel_cache"].items()
                    if isinstance(ep, str) and isinstance(val, str) and is_allowed_tunnel_fn(val.strip().rstrip("/"))
                }
            elif key == "managed_binding":
                mb = raw.get("managed_binding")
                if isinstance(mb, dict) and str(mb.get("endpoint") or "").strip():
                    merged["managed_binding"] = {
                        "account_id": re.sub(r"[^a-zA-Z0-9_-]", "", str(mb.get("account_id") or ""))[:24],
                        "session": str(mb.get("session") or "comfy").strip()[:32] or "comfy",
                        "endpoint": str(mb.get("endpoint") or "").strip()[:64],
                    }
                else:
                    merged["managed_binding"] = None
            elif key == "account_profiles" and isinstance(raw.get("account_profiles"), list):
                clean_accts, seen_ids = [], set()
                for item in raw["account_profiles"]:
                    if not isinstance(item, dict):
                        continue
                    aid = re.sub(r"[^a-zA-Z0-9_-]", "", str(item.get("id") or ""))[:24]
                    if not aid or aid in seen_ids:
                        continue
                    seen_ids.add(aid)
                    try:
                        bal_val = round(float(item["last_balance"]), 2) if item.get("last_balance") is not None else None
                    except (TypeError, ValueError):
                        bal_val = None
                    try:
                        ts_val = int(item["last_updated_ts"]) if item.get("last_updated_ts") is not None else None
                    except (TypeError, ValueError):
                        ts_val = None
                    clean_accts.append({
                        "id": aid,
                        "label": sanitize_label_fn(str(item.get("label") or ""), "Ultra 账号"),
                        "email_masked": str(item.get("email_masked") or "***")[:64],
                        "last_balance": bal_val,
                        "last_is_running": bool(item.get("last_is_running", False)),
                        "last_updated_ts": ts_val,
                    })
                merged["account_profiles"] = clean_accts
            elif key == "account_ledgers" and isinstance(raw.get("account_ledgers"), dict):
                merged["account_ledgers"] = {
                    aid: {lk: lobj[lk] for lk in ACCOUNT_LEDGER_KEYS if lk in lobj}
                    for aid_raw, lobj in raw["account_ledgers"].items()
                    if (aid := re.sub(r"[^a-zA-Z0-9_-]", "", str(aid_raw or ""))[:24]) and isinstance(lobj, dict)
                }
            elif key == "onboarding_cache" and isinstance(raw.get("onboarding_cache"), dict):
                oc = raw["onboarding_cache"]
                try:
                    fg = round(float(oc["gdrive_free_gb"]), 2) if oc.get("gdrive_free_gb") is not None else None
                except (TypeError, ValueError):
                    fg = None
                try:
                    tg = round(float(oc["gdrive_total_gb"]), 2) if oc.get("gdrive_total_gb") is not None else None
                except (TypeError, ValueError):
                    tg = None
                try:
                    ct = int(oc["gdrive_checked_ts"]) if oc.get("gdrive_checked_ts") is not None else None
                except (TypeError, ValueError):
                    ct = None
                merged["onboarding_cache"] = {"gdrive_free_gb": fg, "gdrive_total_gb": tg, "gdrive_checked_ts": ct}
            elif key in raw:
                merged[key] = raw[key]
        return merged
    except Exception:
        return default


def estimate_session_start_balance(balance: float, rate_hourly: float, active_since_ts: int, ref_ts: int, day_start: float, day_topup: float) -> float:
    est = rate_hourly * max(0, ref_ts - int(active_since_ts)) / 3600.0
    cap = day_start + day_topup
    return min(cap, balance + max(0.0, est)) if cap >= balance else (balance + max(0.0, est))


def gpu_rate_key(accelerator: Any, machine_shape: Any = None) -> str | None:
    acc = str(accelerator or "").strip().upper()
    if acc not in SUPPORTED_BOOT_GPUS:
        return None
    if acc == "L4":
        return "L4:STD"
    shape = str(machine_shape or "").strip().upper()
    return f"{acc}:{'HM' if shape in ('HIGH_RAM', 'HM', '1') else 'STD'}"


def snapshot_current_ledger(state: dict[str, Any]) -> dict[str, Any]:
    return {lk: state.get(lk) for lk in ACCOUNT_LEDGER_KEYS}


def restore_ledger_for_account(state: dict[str, Any], acct_id: str) -> None:
    fresh = default_state_dict()
    ledgers = state.setdefault("account_ledgers", {})
    saved = ledgers.get(acct_id) if (acct_id and isinstance(ledgers, dict)) else None
    for lk in ACCOUNT_LEDGER_KEYS:
        state[lk] = saved[lk] if (isinstance(saved, dict) and lk in saved) else fresh[lk]


def upsert_account_profile(
    state: dict[str, Any],
    acct_id: str,
    email_masked: str,
    preferred_label: str = "",
    balance: float | None = None,
    is_running: bool | None = None,
) -> dict[str, Any]:
    from colab_core.accounts import default_account_label, sanitize_account_label

    profiles = state.setdefault("account_profiles", [])
    now_ts = int(time.time())
    for item in profiles:
        if isinstance(item, dict) and item.get("id") == acct_id:
            if preferred_label:
                item["label"] = sanitize_account_label(preferred_label, str(item.get("label") or "Ultra 账号"))
            if email_masked and email_masked != "***":
                item["email_masked"] = email_masked
            if balance is not None:
                item["last_balance"] = round(float(balance), 2)
                item["last_updated_ts"] = now_ts
            if is_running is not None:
                item["last_is_running"] = bool(is_running)
            return dict(item)

    new_item: dict[str, Any] = {
        "id": acct_id,
        "label": sanitize_account_label(preferred_label, default_account_label(profiles)),
        "email_masked": email_masked or f"acct-{acct_id[:6]}",
        "last_balance": round(float(balance), 2) if balance is not None else None,
        "last_is_running": bool(is_running) if is_running is not None else False,
        "last_updated_ts": now_ts if balance is not None else None,
    }
    profiles.append(new_item)
    return dict(new_item)


def sync_active_account_ledger(state: dict[str, Any], *, balance: float | None = None, is_running: bool | None = None) -> None:
    active_id = str((state.get("prefs") or {}).get("active_account_id") or "").strip()
    if not active_id:
        return
    state.setdefault("account_ledgers", {})[active_id] = snapshot_current_ledger(state)
    now_ts = int(time.time())
    for item in state.get("account_profiles") or []:
        if isinstance(item, dict) and item.get("id") == active_id:
            if balance is not None:
                item["last_balance"] = round(float(balance), 2)
                item["last_updated_ts"] = now_ts
            if is_running is not None:
                item["last_is_running"] = bool(is_running)
            break


def short_gpu_name(raw_name: str) -> str:
    return re.sub(r"^cuda:\d+\s*", "", str(raw_name or "")).split(":")[0].strip().replace("NVIDIA ", "").strip()


def format_comfy_error(node_type: str | None, exc_type: str | None, exc_msg: str | None) -> str:
    node = (node_type or "节点").strip() or "节点"
    etype, emsg = (exc_type or "").strip(), (exc_msg or "").strip()
    combined = f"{etype} {emsg}".lower()
    if any(k in combined for k in ("out of memory", "outofmemoryerror", "allocation on device")):
        out = f"{node}: 显存不足 (OOM)"
    elif any(k in combined for k in ("no such file", "filenotfound", "not found", "missing")):
        out = f"{node}: 缺少模型或文件"
    else:
        out = f"{node}: {etype or emsg or '执行异常'}"
    return (out[:35] + "…") if len(out) > 36 else out


def collect_formal_outputs(entry: dict[str, Any]) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    outputs = entry.get("outputs") or {}
    if not isinstance(outputs, dict):
        return items
    for node_out in outputs.values():
        if not isinstance(node_out, dict):
            continue
        for key in ("images", "gifs", "videos", "audio"):
            for item in node_out.get(key) or []:
                if isinstance(item, dict) and item.get("filename") and str(item.get("type") or "output") == "output":
                    items.append({"filename": str(item["filename"]), "subfolder": str(item.get("subfolder") or ""), "type": "output"})
    return items


def extract_history_error(entry: dict[str, Any]) -> str | None:
    status = entry.get("status") or {}
    if str(status.get("status_str") or "").lower() != "error":
        return None
    node_t, exc_t, exc_m = None, None, None
    for msg in status.get("messages") or []:
        if isinstance(msg, (list, tuple)) and len(msg) >= 2 and msg[0] == "execution_error":
            data = msg[1] if isinstance(msg[1], dict) else {}
            node_t, exc_t, exc_m = data.get("node_type") or data.get("node_id"), data.get("exception_type"), data.get("exception_message")
            break
    return format_comfy_error(str(node_t) if node_t else None, str(exc_t) if exc_t else None, str(exc_m) if exc_m else None)


def extract_slot_tps(slot: dict[str, Any]) -> float | None:
    containers: list[Any] = [slot.get("timings"), slot.get("performance")]
    nt = slot.get("next_token")
    containers.append(nt[0] if isinstance(nt, list) and nt else nt)
    for sub in containers:
        if isinstance(sub, dict):
            for k in ("predicted_per_second", "tokens_per_second", "tps"):
                raw = sub.get(k)
                if isinstance(raw, (int, float)) and float(raw) > 0.05:
                    return round(float(raw), 1)
    return None
