"""colab_core.exec_runner —— 统一子进程执行器、取消令牌、后台任务台账与脱敏轮转日志模块。"""

from __future__ import annotations

import contextlib
import json
import os
import re
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

_PKG_DIR = Path(__file__).resolve().parents[1]
if str(_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR))

from colab_security import redact_secrets  # noqa: E402

CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
MAX_DIAG_LOG_BYTES = 512 * 1024
MAX_THROTTLE_KEYS = 512


def resolve_diag_log_path(log_path: Path | None = None) -> Path:
    """解析诊断日志落盘路径（显式路径 > 环境变量 COLAB_PET_DIAG_LOG_PATH > 默认目录）。"""
    if log_path is not None:
        return Path(log_path)
    env_p = str(os.environ.get("COLAB_PET_DIAG_LOG_PATH") or "").strip()
    return Path(env_p) if env_p else (_PKG_DIR / "colab_pet" / "logs" / "colab_diag.log")


def clean_subprocess_env(extra_env: dict[str, str] | None = None) -> dict[str, str]:
    """构造带 UTF-8 编码与默认请求超时的干净子进程环境变量。"""
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
    """跨平台强制终止子进程及其子孙进程树。"""
    if proc is None:
        return
    pid = getattr(proc, "pid", None)
    if os.name == "nt" and isinstance(pid, int) and pid > 0:
        with suppress_logged("proc", throttled_key="taskkill_err", interval_sec=30.0):
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                capture_output=True,
                creationflags=CREATE_NO_WINDOW,
                timeout=8,
            )
    for m_name in ("terminate", "kill"):
        fn = getattr(proc, m_name, None)
        if callable(fn):
            with suppress_logged("proc", throttled_key=f"proc_{m_name}_err", interval_sec=30.0):
                fn()


def extract_framed_json(text: str | None, marker: str) -> dict[str, Any] | None:
    """从标准输出中提取 `<<<MARKER>>>...<<<END_MARKER>>>` 包裹的结构化 JSON 字典。"""
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
    """线程安全、带 512 KiB 轮转、带键节流与自动凭据脱敏的结构化诊断日志器。"""

    def __init__(
        self,
        log_path: Path | None = None,
        *,
        max_bytes: int = MAX_DIAG_LOG_BYTES,
        max_throttle_keys: int = MAX_THROTTLE_KEYS,
    ) -> None:
        self._explicit_path: Path | None = Path(log_path) if log_path is not None else None
        self.max_bytes = max(4096, int(max_bytes or MAX_DIAG_LOG_BYTES))
        self._lock = threading.Lock()
        self._throttle_lock = threading.Lock()
        self._throttle_ts: dict[tuple[str, str], float] = {}
        self._max_throttle_keys = max(16, int(max_throttle_keys or MAX_THROTTLE_KEYS))

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

    def _evict_throttle_keys_locked(self, now_m: float, interval_sec: float) -> None:
        """在节流缓存达到上限时先清过期键、再淘汰最旧键，防止长期运行内存泄漏。"""
        expire_span = max(interval_sec, 60.0)
        expired = [k for k, ts in self._throttle_ts.items() if (now_m - ts) >= expire_span]
        for k in expired:
            self._throttle_ts.pop(k, None)
        if len(self._throttle_ts) >= self._max_throttle_keys:
            drop_n = max(1, len(self._throttle_ts) - (self._max_throttle_keys // 2) + 1)
            oldest = sorted(self._throttle_ts.items(), key=lambda kv: kv[1])[:drop_n]
            for k, _ in oldest:
                self._throttle_ts.pop(k, None)

    def log(self, level: str, component: str, message: str, **fields: Any) -> None:
        lvl = str(level or "INFO").strip().upper()
        comp = str(component or "core").strip() or "core"
        msg_clean = redact_secrets(str(message or "").replace("\r", " ").replace("\n", " "))
        ctx = ""
        if fields:
            try:
                ctx = " | " + redact_secrets(json.dumps(fields, ensure_ascii=False, default=str))
            except (TypeError, ValueError, OverflowError):
                ctx = " | <fields_unserializable>"
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

    def debug(self, component: str, message: str, **fields: Any) -> None:
        """写入 DEBUG 级别脱敏诊断日志。"""
        self.log("DEBUG", component, message, **fields)

    def debug_throttled(
        self,
        tag: str,
        key: str,
        msg: str,
        interval_sec: float = 60.0,
        **fields: Any,
    ) -> bool:
        """线程安全带键节流 DEBUG 日志：同一 `(tag, key)` 在 `interval_sec` 窗口内仅落盘一次。"""
        norm_tag = str(tag or "core").strip() or "core"
        norm_key = str(key or "").strip()
        eff_interval = max(0.0, float(interval_sec))
        now_m = time.monotonic()
        bucket = (norm_tag, norm_key)
        with self._throttle_lock:
            last_ts = self._throttle_ts.get(bucket)
            if eff_interval > 0.0 and last_ts is not None and (now_m - last_ts) < eff_interval:
                return False
            if bucket not in self._throttle_ts and len(self._throttle_ts) >= self._max_throttle_keys:
                self._evict_throttle_keys_locked(now_m, eff_interval)
            self._throttle_ts[bucket] = now_m
        self.debug(norm_tag, msg, **fields)
        return True

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
    """获取进程级共享诊断日志器（传入显式路径时返回独立实例）。"""
    global _GLOBAL_DIAG_LOGGER
    if log_path is not None:
        return DiagLogger(log_path=log_path)
    with _GLOBAL_DIAG_LOCK:
        if _GLOBAL_DIAG_LOGGER is None:
            _GLOBAL_DIAG_LOGGER = DiagLogger()
        return _GLOBAL_DIAG_LOGGER


@contextlib.contextmanager
def suppress_logged(
    scope: str,
    *exc_types: type[BaseException],
    throttled_key: str | None = None,
    interval_sec: float = 60.0,
) -> Iterator[None]:
    """安全抑制非致命异常，并向全局诊断日志器写入脱敏后的 DEBUG 诊断信息。"""
    caught_types = exc_types or (Exception,)
    try:
        yield
    except caught_types as exc:
        err_detail = redact_secrets(str(exc) or type(exc).__name__)
        msg = f"已抑制异常 ({type(exc).__name__}): {err_detail}"
        diag = get_diag_logger()
        if throttled_key:
            diag.debug_throttled(scope, throttled_key, msg, interval_sec=interval_sec)
        else:
            diag.debug(scope, msg)


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
            with suppress_logged("cancel_token", throttled_key="register_cb_err", interval_sec=10.0):
                fn()

    def cancel(self) -> bool:
        with self._lock:
            already = self._event.is_set()
            self._event.set()
            cbs = list(self._callbacks)
            self._callbacks.clear()
        if already:
            return False
        for cb in cbs:
            with suppress_logged("cancel_token", throttled_key="cancel_cb_err", interval_sec=10.0):
                cb()
        return True


@dataclass
class CommandResult:
    """标准化外部命令执行结果。"""

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


def _decode_text_maybe(raw: Any) -> str:
    """安全将 bytes/str/None 转为 UTF-8 字符串。"""
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace")
    return str(raw or "")


def _emit_stream_line(raw_ln: str, collected: list[str], on_line: Callable[[str], None] | None) -> None:
    """脱敏并分发单行流式标准输出。"""
    sl = redact_secrets(raw_ln.strip())
    if not sl:
        return
    collected.append(sl)
    if on_line is not None:
        on_line(sl)


def _drain_stream_lines(stdout_stream: Any, on_line: Callable[[str], None] | None) -> list[str]:
    """按块增量读取子进程输出流，兼容 `\\r` 进度刷新并在结束时安全关闭句柄。"""
    if stdout_stream is None:
        return []
    buf = ""
    lines_collected: list[str] = []
    try:
        while True:
            ch = stdout_stream.read(256)
            if not ch:
                if buf.strip():
                    for raw_ln in buf.splitlines():
                        _emit_stream_line(raw_ln, lines_collected, on_line)
                break
            buf += ch.replace("\r", "\n")
            while "\n" in buf:
                raw_ln, buf = buf.split("\n", 1)
                _emit_stream_line(raw_ln, lines_collected, on_line)
    finally:
        with suppress_logged("runner", throttled_key="stream_close_err", interval_sec=30.0):
            stdout_stream.close()
    return lines_collected


class ProcessRunner:
    """统一同步与流式子进程执行器（集中超时、重试、取消杀树与脱敏日志）。"""

    def __init__(self, diag: DiagLogger | None = None) -> None:
        self.diag = diag or get_diag_logger()

    def _run_once(
        self,
        cmd_list: list[str],
        *,
        attempt: int,
        timeout: int,
        input_text: str | None,
        extra_env: dict[str, str] | None,
        run_fn: Callable[..., subprocess.CompletedProcess[Any]] | None,
    ) -> CommandResult:
        """执行单次同步子进程并将正常/超时/异常结果统一封装为 CommandResult。"""
        safe_cmd = [redact_secrets(c) for c in cmd_list]
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
            return CommandResult(
                cmd=safe_cmd,
                returncode=int(getattr(proc, "returncode", 1)),
                stdout=redact_secrets(getattr(proc, "stdout", "") or ""),
                stderr=redact_secrets(getattr(proc, "stderr", "") or ""),
                elapsed_sec=round(time.monotonic() - t0, 3),
                attempts=attempt,
            )
        except subprocess.TimeoutExpired as exc:
            return CommandResult(
                cmd=safe_cmd,
                returncode=124,
                stdout=redact_secrets(_decode_text_maybe(getattr(exc, "stdout", ""))),
                stderr=redact_secrets(_decode_text_maybe(getattr(exc, "stderr", ""))),
                elapsed_sec=round(time.monotonic() - t0, 3),
                attempts=attempt,
                timed_out=True,
                error=f"命令执行超时 ({timeout}s)",
            )
        except Exception as exc:
            return CommandResult(
                cmd=safe_cmd,
                returncode=1,
                elapsed_sec=round(time.monotonic() - t0, 3),
                attempts=attempt,
                error=redact_secrets(str(exc)),
            )

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
        """执行同步外部命令，支持重试、取消令牌与自动脱敏日志。"""
        cmd_list = [str(x) for x in cmd]
        cmd_safe = redact_secrets(" ".join(cmd_list))
        max_attempts = max(1, int(retries))
        tag = task_name or (Path(cmd_list[0]).name if cmd_list else "cmd")

        last_res = CommandResult(cmd=cmd_list, returncode=1)
        for attempt in range(1, max_attempts + 1):
            if cancel_token is not None and cancel_token.is_cancelled:
                return CommandResult(
                    cmd=[redact_secrets(c) for c in cmd_list],
                    returncode=1,
                    cancelled=True,
                    attempts=attempt,
                    error="任务在执行前已取消",
                )
            res = self._run_once(
                cmd_list,
                attempt=attempt,
                timeout=timeout,
                input_text=input_text,
                extra_env=extra_env,
                run_fn=run_fn,
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
        """流式执行外部命令并逐行脱敏回调，支持看门狗超时与取消杀树。"""
        cmd_list = [str(x) for x in cmd]
        safe_cmd = [redact_secrets(c) for c in cmd_list]
        cmd_safe = redact_secrets(" ".join(cmd_list))
        tag = task_name or (Path(cmd_list[0]).name if cmd_list else "stream")
        t0 = time.monotonic()

        if cancel_token is not None and cancel_token.is_cancelled:
            return CommandResult(cmd=safe_cmd, returncode=1, cancelled=True, error="任务在启动前已取消")

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
            lines_collected = _drain_stream_lines(getattr(proc, "stdout", None), on_line)
            rc = int(proc.wait())
            dt = round(time.monotonic() - t0, 3)
            cancelled = bool(cancel_token is not None and cancel_token.is_cancelled)
            timed_out = timed_out_flag.is_set()
            return CommandResult(
                cmd=safe_cmd,
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
                cmd=safe_cmd,
                returncode=1,
                elapsed_sec=round(time.monotonic() - t0, 3),
                cancelled=bool(cancel_token is not None and cancel_token.is_cancelled),
                error=err_txt,
            )
        finally:
            if watchdog is not None:
                watchdog.cancel()


class TaskStatus(StrEnum):
    """后台受控任务状态枚举。"""

    IDLE = "idle"
    RUNNING = "running"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMED_OUT = "timed_out"


@dataclass
class TaskRecord:
    """单条后台任务运行记录。"""

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
    """后台任务句柄，提供取消与等待能力。"""

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


def _resolve_task_outcome(res_val: Any, token: CancelToken) -> tuple[TaskStatus, str]:
    """根据任务返回值与取消令牌推导最终任务状态与脱敏错误摘要。"""
    if isinstance(res_val, CommandResult):
        if res_val.cancelled:
            return TaskStatus.CANCELLED, ""
        if res_val.timed_out:
            return TaskStatus.TIMED_OUT, res_val.error_summary
        if not res_val.ok:
            return TaskStatus.FAILED, res_val.error_summary
    elif token.is_cancelled:
        return TaskStatus.CANCELLED, ""
    return TaskStatus.SUCCEEDED, ""


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
        """提交受控后台任务，自动记录运行态、互斥防重与异常落盘。"""
        token = CancelToken()
        with self._lock:
            cur = self._records.get(name)
            if exclusive and cur and cur.status in (TaskStatus.RUNNING, TaskStatus.CANCELLING):
                return TaskHandle(
                    name=name,
                    accepted=False,
                    token=cur.cancel_token or token,
                    thread=self._threads.get(name),
                    reason=f"任务 {name} 正在执行中",
                )
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
                final_status, err_msg = _resolve_task_outcome(res_val, token)
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


__all__ = [
    "CREATE_NO_WINDOW",
    "CancelToken",
    "CommandResult",
    "DiagLogger",
    "MAX_DIAG_LOG_BYTES",
    "MAX_THROTTLE_KEYS",
    "ProcessRunner",
    "TaskHandle",
    "TaskManager",
    "TaskRecord",
    "TaskStatus",
    "_GLOBAL_DIAG_LOCK",
    "_GLOBAL_DIAG_LOGGER",
    "_PKG_DIR",
    "clean_subprocess_env",
    "extract_framed_json",
    "get_diag_logger",
    "kill_process_tree",
    "resolve_diag_log_path",
    "suppress_logged",
]
