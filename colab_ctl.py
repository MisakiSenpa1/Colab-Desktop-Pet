"""colab_ctl.py —— Google Colab ComfyUI / LLM 工作流本机总控脚手架。"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

_CTL_PKG_DIR = Path(__file__).resolve().parent
if str(_CTL_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_CTL_PKG_DIR))

from colab_core.accounts import (  # noqa: E402
    CLI_BRIDGE_SCRIPT,
    ToolPaths,
    resolve_colab_exe as _core_resolve_colab_exe,
    resolve_colab_py as _core_resolve_colab_py,
    resolve_rclone_conf as _core_resolve_rclone_conf,
    resolve_rclone_exe as _core_resolve_rclone_exe,
)
from colab_core.colab_client import (  # noqa: E402
    build_tunnel_url_pattern as _core_build_tunnel_url_pattern,
    collect_local_py_deps as _collect_local_py_deps,
    ensure_or_adopt_session as _core_ensure_or_adopt_session,
    exec_remote_file as _core_exec_remote_file,
    exec_remote_py as _core_exec_remote_py,
    probe_comfy_endpoint_ok as _probe_comfy_endpoint_ok,
    probe_session_endpoint as _core_probe_session_endpoint,
    stop_session_cli as _core_stop_session_cli,
)
from colab_core.file_sync import (  # noqa: E402
    REMOTE_OUT_ROOT,
    REMOTE_RCLONE_CONF,
    normalize_prefix as _normalize_prefix,
    sync_anima_loras_cli,
    sync_drama3_refs_cli,
    sync_session_outputs_cli,
)
from colab_core.health import export_diagnostic_bundle, run_preflight_checks  # noqa: E402
from colab_core.runtime import (  # noqa: E402
    COMFY_PROFILES,
    LLM_PROFILES,
    clean_subprocess_env,
    extract_framed_json,
    get_diag_logger,
    kill_process_tree,
    parse_profiles_cli_arg as _parse_profiles_arg,
    profile_bootstrap_spec as _core_profile_bootstrap_spec,
    resolve_high_mem as _core_resolve_high_mem,
    select_default_gpu as _core_select_default_gpu,
)
from colab_security import (  # noqa: E402
    generate_comfy_api_key,
    load_cf_tunnel_settings,
    redact_secrets,
    register_ephemeral_secret,
)

resolve_colab_exe = lambda: _core_resolve_colab_exe()  # noqa: E731
resolve_colab_py = lambda colab_exe=None: _core_resolve_colab_py(colab_exe=colab_exe)  # noqa: E731
resolve_rclone_exe = lambda: _core_resolve_rclone_exe()  # noqa: E731
resolve_rclone_conf = lambda rclone_exe=None: _core_resolve_rclone_conf(rclone_exe=rclone_exe)  # noqa: E731

COLAB_EXE = resolve_colab_exe()
COLAB_PY = resolve_colab_py(COLAB_EXE)
RCLONE_EXE = resolve_rclone_exe()
RCLONE_CONF = resolve_rclone_conf(RCLONE_EXE)

WORKSPACE_ROOT = _CTL_PKG_DIR
LORA_BACKUP_DIR = Path(os.environ.get("COLAB_LORA_DIR") or (_CTL_PKG_DIR / "loras"))
LOCAL_DOWNLOAD_DIR = Path(os.environ.get("COLAB_DOWNLOAD_DIR") or (_CTL_PKG_DIR / "_downloads"))
BOOTSTRAP_SCRIPT = _CTL_PKG_DIR / "colab_bootstrap_anima.py"
SMOKE_SCRIPT = _CTL_PKG_DIR / "verify_smoke.py"
MINIMAX_BOOTSTRAP_SCRIPT = _CTL_PKG_DIR / "colab_bootstrap_minimax_h3.py"
MINIMAX_SMOKE_SCRIPT = _CTL_PKG_DIR / "verify_minimax_h3.py"
QWEN_LLM_BOOTSTRAP_SCRIPT = _CTL_PKG_DIR / "colab_bootstrap_qwen_llm.py"
QWEN_LLM_SMOKE_SCRIPT = _CTL_PKG_DIR / "verify_qwen_llm.py"
DRAMA3_BOOTSTRAP_SCRIPT = _CTL_PKG_DIR / "colab_bootstrap_drama3.py"
DRAMA3_SMOKE_SCRIPT = _CTL_PKG_DIR / "verify_drama3.py"
DRAMA3_REFS_LOCAL_DIR = Path(os.environ.get("COLAB_DRAMA3_REFS_DIR") or (_CTL_PKG_DIR / "drama3_refs"))
CF_TUNNEL_SECRET_FILE = _CTL_PKG_DIR / ".cf_tunnel_secret.json"


def _resolve_cf_cfg_path() -> Path:
    return Path(os.environ.get("COLAB_PET_CF_CONFIG_PATH") or str(CF_TUNNEL_SECRET_FILE))


def _load_cf_tunnel_secret() -> dict:
    return load_cf_tunnel_settings(_resolve_cf_cfg_path(), include_secrets=True, colab_py=COLAB_PY)


def _build_tunnel_url_pattern(cf_cfg: dict | None = None) -> str:
    return _core_build_tunnel_url_pattern(cf_cfg if isinstance(cf_cfg, dict) else _load_cf_tunnel_secret())


def _ensure_bin(path: Path | None, desc: str) -> None:
    if not path or not Path(path).is_file():
        raise FileNotFoundError(f"未找到{desc}: {path}")


def _run(cmd, check: bool = True, capture: bool = False, timeout: float | None = None):
    diag = get_diag_logger()
    cmd_str = redact_secrets(" ".join(str(x) for x in cmd))
    print(f"[本机执行] {cmd_str}", flush=True)
    env, str_cmd = clean_subprocess_env(), [str(x) for x in cmd]
    if timeout is None:
        res = subprocess.run(str_cmd, check=False, text=True, capture_output=capture, encoding="utf-8", errors="replace", env=env)
    else:
        pipe = subprocess.PIPE if capture else None
        proc = subprocess.Popen(str_cmd, stdout=pipe, stderr=pipe, text=True, encoding="utf-8", errors="replace", env=env)
        try:
            out, err = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            kill_process_tree(proc)
            try:
                proc.communicate(timeout=5)
            except Exception:
                pass
            diag.error("ctl", f"命令超时（{timeout:.0f}s）已强制终止进程树: {cmd_str[:160]}")
            raise subprocess.TimeoutExpired([redact_secrets(x) for x in str_cmd], timeout) from None
        res = subprocess.CompletedProcess(str_cmd, proc.returncode, stdout=out, stderr=err)
    (diag.info if res.returncode == 0 else diag.warn)("ctl", f"命令{'完成' if res.returncode == 0 else f'非零退出 rc={res.returncode}'}: {cmd_str[:160]}")
    if check and res.returncode != 0:
        if capture:
            if res.stdout:
                print(redact_secrets(res.stdout), end="", flush=True)
            if res.stderr:
                print(redact_secrets(res.stderr), file=sys.stderr, end="", flush=True)
        raise subprocess.CalledProcessError(
            res.returncode,
            [redact_secrets(str(x)) for x in cmd],
            output=redact_secrets(res.stdout) if capture and res.stdout else None,
            stderr=redact_secrets(res.stderr) if capture and res.stderr else None,
        )
    return res


def _colab_exec_file(session: str, script_path: Path, timeout: int = 600, capture: bool = False, retries: int = 3, remote_file: str | None = None, extra_files: dict[str, str] | None = None, manifest_out_dir: str | None = None):
    return _core_exec_remote_file(session, script_path, colab_py=COLAB_PY, run_fn=_run, timeout=timeout, capture=capture, retries=retries, remote_file=remote_file, extra_files=extra_files, manifest_out_dir=manifest_out_dir)


def _exec_remote_py(session: str, code: str, timeout: int = 300, capture: bool = False):
    return _core_exec_remote_py(session, code, exec_file_fn=_colab_exec_file, timeout=timeout, capture=capture)


def _profile_bootstrap_spec(profile: str) -> tuple[Path, int]:
    return _core_profile_bootstrap_spec(
        profile,
        pkg_dir=_CTL_PKG_DIR,
        anima_script=BOOTSTRAP_SCRIPT,
        minimax_script=MINIMAX_BOOTSTRAP_SCRIPT,
        drama3_script=DRAMA3_BOOTSTRAP_SCRIPT,
        qwen_llm_script=QWEN_LLM_BOOTSTRAP_SCRIPT,
    )


def cmd_sync_loras(args=None) -> None:
    src_dir = Path(getattr(args, "src", None) or LORA_BACKUP_DIR) if args is not None else LORA_BACKUP_DIR
    sync_anima_loras_cli(rclone_exe=RCLONE_EXE, rclone_conf=RCLONE_CONF, lora_backup_dir=src_dir, run_fn=_run)


def cmd_sync_drama3_refs(args) -> None:
    sync_drama3_refs_cli(src_dir=Path(getattr(args, "src", None) or DRAMA3_REFS_LOCAL_DIR), rclone_exe=RCLONE_EXE, rclone_conf=RCLONE_CONF, run_fn=_run)


def cmd_status(_args) -> None:
    _ensure_bin(COLAB_EXE, "colab.exe")
    print("=== Colab 活跃会话 ===", flush=True)
    _run([COLAB_EXE, "sessions"])
    print("=== Colab 配额与速率 ===", flush=True)
    _run([COLAB_EXE, "usage"])


def cmd_up(args) -> None:
    """创建、纳管或复用 Colab GPU 会话，上传可选 rclone 配置并完成单/双栖自举部署。"""
    _ensure_bin(COLAB_EXE, "colab.exe")

    profiles_list = _parse_profiles_arg(getattr(args, "profile", None), getattr(args, "profiles", None))
    llm_prof = next((p for p in profiles_list if p in LLM_PROFILES), None)
    comfy_profs = [p for p in profiles_list if p in COMFY_PROFILES]

    default_gpu = _core_select_default_gpu(profiles_list)
    exec_order: list[str] = ([llm_prof] if llm_prof else []) + comfy_profs
    for p in exec_order:
        script_p, _ = _profile_bootstrap_spec(p)
        _ensure_bin(script_p, script_p.name)

    gpu = getattr(args, "gpu", None) or default_gpu
    high_mem = _core_resolve_high_mem(gpu, bool(getattr(args, "high_mem", False)))
    recreate = bool(getattr(args, "recreate", False))
    minimax_lite = bool(getattr(args, "minimax_lite", False))
    no_r2v = bool(getattr(args, "no_r2v", False))
    no_pe = bool(getattr(args, "no_pe", False))

    if not bool(getattr(args, "skip_preflight", False)):
        report = run_preflight_checks(
            mode="fast",
            profiles=profiles_list,
            gpu=gpu,
            tool_paths=ToolPaths(COLAB_EXE, COLAB_PY, RCLONE_EXE, RCLONE_CONF),
            download_dir=LOCAL_DOWNLOAD_DIR,
            cf_config_path=_resolve_cf_cfg_path(),
            pkg_dir=_CTL_PKG_DIR,
            require_local_rclone=False,
        )
        if not report.ok:
            raise RuntimeError(f"启动前自检未通过: {report.first_blocking_message()}")

    _core_ensure_or_adopt_session(
        session=args.session,
        gpu=gpu,
        high_mem=high_mem,
        recreate=recreate,
        target_endpoint=str(getattr(args, "endpoint", None) or "").strip() or None,
        colab_exe=COLAB_EXE,
        colab_py=COLAB_PY,
        run_fn=_run,
        stop_fn=stop_session,
    )

    if RCLONE_CONF.is_file():
        _run([COLAB_EXE, "upload", "-s", args.session, RCLONE_CONF, REMOTE_RCLONE_CONF])
    else:
        print(f"[提示] 未检测到 rclone.conf ({RCLONE_CONF})，跳过云盘凭据上传（仍可使用公共 Hugging Face 模型与公网隧道）。", flush=True)
    cf_cfg = _load_cf_tunnel_secret()
    cf_tok = str(cf_cfg.get("tunnel_token") or "").strip()
    cf_llm_url = str(cf_cfg.get("llm_url") or "").strip().rstrip("/")
    cf_comfy_url = str(cf_cfg.get("comfy_url") or "").strip().rstrip("/")

    is_dual = bool(llm_prof and comfy_profs)
    cohost_comfy = bool(is_dual or getattr(args, "cohost_comfy", False))
    total_phases = len(exec_order)
    comfy_key_seed = generate_comfy_api_key() if comfy_profs else ""
    if comfy_key_seed:
        register_ephemeral_secret(comfy_key_seed)

    for phase_idx, cur_prof in enumerate(exec_order, start=1):
        target_bootstrap, exec_timeout = _profile_bootstrap_spec(cur_prof)
        if total_phases >= 2:
            print(f"=== DEPLOY_PHASE {phase_idx}/{total_phases}: {cur_prof} ===", flush=True)

        if cur_prof in LLM_PROFILES:
            variant = "27b" if cur_prof == "qwen_27b" else "180b"
            llm_quant = str(getattr(args, "llm_quant", None) or "UD-Q4_K_XL").strip()
            raw_ctx = getattr(args, "llm_ctx", None)
            llm_ctx = str(int(raw_ctx if raw_ctx is not None else (65536 if variant == "27b" else 32768)))
            llm_api_key = str(getattr(args, "llm_api_key", None) or "auto").strip() or "auto"
            if llm_api_key.lower() == "auto" and len(str(cf_cfg.get("llm_api_key") or "").strip()) >= 16:
                llm_api_key = str(cf_cfg["llm_api_key"]).strip()

            env_lines = [
                "import os",
                f"os.environ['LLM_MODEL_VARIANT'] = {variant!r}",
                f"os.environ['LLM_COHOST_COMFY'] = {'1' if cohost_comfy else '0'!r}",
                f"os.environ['LLM_QUANT'] = {llm_quant!r}",
                f"os.environ['LLM_CTX_SIZE'] = {llm_ctx!r}",
                f"os.environ['LLM_WITH_MMPROJ'] = {'0' if getattr(args, 'no_mmproj', False) else '1'!r}",
                f"os.environ['LLM_WITH_WEBUI'] = {'0' if getattr(args, 'no_webui', False) else '1'!r}",
                f"os.environ['LLM_GDRIVE_CACHE'] = {'1' if getattr(args, 'gdrive_cache', False) else '0'!r}",
                f"os.environ['LLM_API_KEY'] = {llm_api_key!r}",
            ]
            if cf_tok and cf_llm_url:
                env_lines.extend([f"os.environ['CF_TUNNEL_TOKEN'] = {cf_tok!r}", f"os.environ['CF_TUNNEL_URL'] = {cf_llm_url!r}"])
            if cohost_comfy and cf_comfy_url:
                env_lines.append(f"os.environ['CF_COMFY_TUNNEL_URL'] = {cf_comfy_url!r}")
            body_head = "\n".join(env_lines) + "\n"
            body_tail = ""
        else:
            prep_lines = [
                "import os, subprocess",
                "from pathlib import Path",
                "from colab_remote_common import ensure_comfy_auth_key, write_private_text",
                "_auth_kf = Path('/root/comfy_auth_key')",
                f"if not (_auth_kf.is_file() and _auth_kf.read_text(encoding='utf-8', errors='ignore').strip()): write_private_text(_auth_kf, {comfy_key_seed!r} + '\\n')",
                "os.environ['COMFY_AUTH_KEY_FILE'] = '/root/comfy_auth_key'",
            ]
            if not is_dual and phase_idx == 1:
                prep_lines.extend([
                    "_meta_p = Path('/root/colab_service_meta.json')",
                    "if _meta_p.exists(): _meta_p.unlink(missing_ok=True); subprocess.run(['pkill', '-f', 'cloudflared tunnel'], check=False, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)",
                    "subprocess.run(['pkill', '-f', 'llm_sse_gateway.py'], check=False, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)",
                    "subprocess.run(['pkill', '-f', 'llama-server'], check=False, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)",
                ])
            if is_dual:
                prep_lines.extend(["os.environ['COMFY_RESERVE_VRAM'] = '4.0'", "os.environ['COMFY_ENSURE_RESERVE_VRAM'] = '1'", "os.environ['COMFY_COHOST_LLM'] = '1'"])
            if cf_tok and cf_comfy_url:
                prep_lines.extend([f"os.environ['CF_TUNNEL_TOKEN'] = {cf_tok!r}", f"os.environ['CF_TUNNEL_URL'] = {cf_comfy_url!r}"])
            if cur_prof == "minimax":
                if minimax_lite:
                    prep_lines.append("os.environ['MINIMAX_LITE'] = '1'")
                if no_r2v:
                    prep_lines.append("os.environ['MINIMAX_WITH_R2V'] = '0'")
                if no_pe:
                    prep_lines.append("os.environ['MINIMAX_WITH_PE'] = '0'")
            body_head = "\n".join(prep_lines) + "\n"
            body_tail = ""
            if phase_idx == total_phases:
                body_tail = (
                    "\nfrom colab_remote_common import finalize_service_contract as _fsc\n"
                    f"_fsc(is_dual={is_dual!r}, profiles={profiles_list!r}, comfy_profiles={comfy_profs!r}, "
                    f"llm_profile={llm_prof!r}, cf_comfy_url={cf_comfy_url!r}, cf_llm_url={cf_llm_url!r}, "
                    f"tunnel_pattern={_build_tunnel_url_pattern(cf_cfg)!r})\n"
                )

        with tempfile.TemporaryDirectory(prefix=f"{cur_prof}_opts_") as tmp_dir:
            wrapped = Path(tmp_dir) / target_bootstrap.name
            wrapped.write_text(body_head + target_bootstrap.read_text(encoding="utf-8") + body_tail, encoding="utf-8")
            _colab_exec_file(args.session, wrapped, timeout=exec_timeout, extra_files=_collect_local_py_deps(target_bootstrap))


def _verify_llm_health(url: str, timeout: float = 15.0) -> dict:
    with urllib.request.urlopen(urllib.request.Request(f"{url}/health", headers={"User-Agent": "Mozilla/5.0"}), timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8") or "{}")
        if resp.status != 200 or body.get("status") != "ok":
            raise RuntimeError(f"LLM /health 未就绪: HTTP {resp.status}, body={body}")
        return body


def cmd_tunnel(args) -> None:
    """获取或重启远端 ComfyUI / LLM cloudflared 隧道，并在本机校验公网连通性。"""
    force_restart = bool(getattr(args, "restart", False))
    cf_cfg = _load_cf_tunnel_secret()
    cf_tok = str(cf_cfg.get("tunnel_token") or "").strip()
    cf_llm_url = str(cf_cfg.get("llm_url") or "").strip().rstrip("/")
    cf_comfy_url = str(cf_cfg.get("comfy_url") or "").strip().rstrip("/")
    tunnel_pat = _build_tunnel_url_pattern(cf_cfg)
    fast_comfy_key = ""

    if not force_restart and COLAB_PY.is_file() and CLI_BRIDGE_SCRIPT.is_file():
        try:
            fast_res = subprocess.run(
                [str(COLAB_PY), str(CLI_BRIDGE_SCRIPT), "fast-tunnel", "", args.session, tunnel_pat],
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=15, env=clean_subprocess_env(),
            )
            pkt = extract_framed_json(fast_res.stdout or "", "COLAB_TUNNEL") if fast_res.returncode == 0 else None
            if pkt and pkt.get("status") == "ok":
                fast_comfy_key = str(pkt.get("comfy_api_key") or "").strip()
                if fast_comfy_key:
                    register_ephemeral_secret(fast_comfy_key)
                meta = pkt.get("meta") if isinstance(pkt.get("meta"), dict) else {}
                fast_kind = str(meta.get("kind") or "comfy")
                fast_url = str(pkt.get("url") or "").strip()
                fast_llm_url = str(meta.get("llm_tunnel_url") or "").strip()
                expected_cf = (cf_llm_url if fast_kind == "llm" else cf_comfy_url) if cf_tok else ""
                if expected_cf and fast_url.rstrip("/") != expected_cf.rstrip("/"):
                    fast_url = ""
                if fast_kind == "dual" and cf_tok and cf_llm_url and fast_llm_url.rstrip("/") != cf_llm_url.rstrip("/"):
                    fast_llm_url = ""
                if fast_url and fast_kind == "dual" and fast_llm_url:
                    c_ok, cbody = _probe_comfy_endpoint_ok(fast_url, comfy_key=fast_comfy_key, timeout=5.0)
                    _verify_llm_health(fast_llm_url, timeout=5.0)
                    if c_ok:
                        devs = ", ".join(f"{d.get('name', 'unknown')} ({d.get('vram_total', 0) / (1024 ** 3):.1f} GiB)" for d in (cbody.get("devices") or [])) or ("公网鉴权保护中" if cbody.get("auth_protected") else "GPU 在线")
                        print(f"复用运行中的双栖 cloudflared 隧道（HTTP 快速通道）。\nCOMFYUI_TUNNEL_URL: {fast_url}\nLLM_TUNNEL_URL: {fast_llm_url}\n[隧道正常] 双栖在线 | GPU: {devs} | LLM: {meta.get('model') or 'Qwen3.8'}\n[ComfyUI WebUI 地址] {fast_url}\n[LLM WebUI / API 地址] {fast_llm_url}", flush=True)
                        return
                elif fast_url and fast_kind == "llm":
                    _verify_llm_health(fast_url, timeout=5.0)
                    print(f"复用运行中的 cloudflared 隧道（HTTP 快速通道）。\nSERVICE_TUNNEL_URL: {fast_url}\nCOMFYUI_TUNNEL_URL: {fast_url}\n[隧道正常] HTTP 200 | LLM: {meta.get('model') or 'Qwen3.8-Flash-Next-abliterated'}\n[LLM WebUI / API 地址] {fast_url}", flush=True)
                    return
                elif fast_url:
                    c_ok, body = _probe_comfy_endpoint_ok(fast_url, comfy_key=fast_comfy_key, timeout=5.0)
                    if c_ok:
                        devs = ", ".join(f"{d.get('name', 'unknown')} ({d.get('vram_total', 0) / (1024 ** 3):.1f} GiB)" for d in (body.get("devices") or [])) or ("公网鉴权保护中" if body.get("auth_protected") else "未返回设备列表")
                        print(f"复用运行中的 cloudflared 隧道（HTTP 快速通道）。\nCOMFYUI_TUNNEL_URL: {fast_url}\n[隧道正常] HTTP 200 | GPU: {devs}\n[ComfyUI WebUI 地址] {fast_url}", flush=True)
                        return
        except Exception:
            pass

    remote_code = f"""
import re
from colab_remote_common import (
    LLM_TUNNEL_URL_FILE, SERVICE_META_FILE, TUNNEL_URL_FILE,
    read_json_dict, start_cloudflared_tunnel, write_private_json,
)

tun_re = re.compile({tunnel_pat!r})
meta = read_json_dict(SERVICE_META_FILE)
kind = str(meta.get("kind") or "comfy")
port = int(meta.get("port") or (18080 if kind == "llm" else 8188))
cf_tok, cf_llm, cf_comfy = {cf_tok!r}, {cf_llm_url!r}, {cf_comfy_url!r}
cf_target = (cf_llm if kind == "llm" else cf_comfy).strip().rstrip("/")
is_dual = (kind == "dual")

tunnel_url = start_cloudflared_tunnel(
    local_port=port,
    service_kind=("llm" if kind == "llm" else "comfy"),
    cf_token=cf_tok,
    cf_url=cf_target,
    cohost_other=is_dual,
    force_restart={force_restart!r},
    tunnel_url_file=TUNNEL_URL_FILE,
    tunnel_re=tun_re,
)
dual_llm_url = None
if is_dual:
    if cf_tok and cf_llm:
        dual_llm_url = cf_llm
        LLM_TUNNEL_URL_FILE.write_text(dual_llm_url + "\\n", encoding="utf-8")
    elif not (cf_tok and cf_comfy):
        llm_port = int(meta.get("llm_port") or 18080)
        dual_llm_url = start_cloudflared_tunnel(
            local_port=llm_port,
            service_kind="llm",
            cohost_other=True,
            force_restart={force_restart!r},
            tunnel_url_file=LLM_TUNNEL_URL_FILE,
            tunnel_re=tun_re,
        )
if meta:
    meta["tunnel_url"] = tunnel_url
    if is_dual:
        meta["comfy_tunnel_url"] = tunnel_url
        if dual_llm_url:
            meta["llm_tunnel_url"] = dual_llm_url
    write_private_json(SERVICE_META_FILE, meta)

print(f"SERVICE_KIND: {{kind}}", flush=True)
if is_dual:
    shown_llm = str(meta.get("llm_tunnel_url") or dual_llm_url or "")
    if shown_llm:
        print(f"LLM_TUNNEL_URL: {{shown_llm}}", flush=True)
    print(f"SERVICE_TUNNEL_URL: {{tunnel_url}}", flush=True)
elif kind == "llm":
    print(f"SERVICE_TUNNEL_URL: {{tunnel_url}}", flush=True)
print(f"COMFYUI_TUNNEL_URL: {{tunnel_url}}", flush=True)
"""
    res = _exec_remote_py(args.session, remote_code, timeout=120, capture=True)
    remote_out = redact_secrets(res.stdout or "")
    print(remote_out, end="" if remote_out.endswith("\n") else "\n", flush=True)

    kind_m = re.search(r"SERVICE_KIND:\s*(\w+)", remote_out)
    service_kind = kind_m.group(1).strip().lower() if kind_m else "comfy"
    url_m = re.search(rf"(?:SERVICE_TUNNEL_URL|COMFYUI_TUNNEL_URL):\s*({tunnel_pat})", remote_out) or re.search(tunnel_pat, remote_out)
    if not url_m:
        raise RuntimeError("未能从远端输出中解析出隧道公网 URL")
    tunnel_url = url_m.group(1) if url_m.lastindex else url_m.group(0)
    llm_m = re.search(rf"LLM_TUNNEL_URL:\s*({tunnel_pat})", remote_out)
    llm_tunnel_url = llm_m.group(1) if llm_m else ""
    probe_path = "/health" if service_kind == "llm" else "/system_stats"
    print(f"=== 本机端到端校验隧道连通性: {tunnel_url}{probe_path} ===", flush=True)

    last_err: Exception | None = None
    for attempt in range(1, 7):
        try:
            if service_kind == "llm":
                _verify_llm_health(tunnel_url, timeout=15.0)
                print(f"[隧道正常] HTTP 200 | LLM 服务在线 (status=ok)\n[LLM WebUI / API 地址] {tunnel_url}", flush=True)
                return
            c_ok, body = _probe_comfy_endpoint_ok(tunnel_url, comfy_key=fast_comfy_key, timeout=15.0)
            if not c_ok:
                raise RuntimeError("ComfyUI 隧道状态异常")
            if service_kind == "dual" and llm_tunnel_url:
                _verify_llm_health(llm_tunnel_url, timeout=15.0)
            devs = ", ".join(f"{d.get('name', 'unknown')} ({d.get('vram_total', 0) / (1024 ** 3):.1f} GiB)" for d in (body.get("devices") or [])) or ("公网鉴权保护中" if body.get("auth_protected") else "未返回设备列表")
            print(f"[隧道正常] HTTP 200 | GPU: {devs}\n[ComfyUI WebUI 地址] {tunnel_url}", flush=True)
            if service_kind == "dual" and llm_tunnel_url:
                print(f"[LLM WebUI / API 地址] {llm_tunnel_url}", flush=True)
            return
        except Exception as exc:
            last_err = exc
            print(f"  第 {attempt}/6 次探测暂未就绪 ({redact_secrets(str(exc))})，3 秒后重试...", flush=True)
            time.sleep(3)
    raise RuntimeError(f"本机访问隧道 {tunnel_url}{probe_path} 连续 6 次失败，最后错误: {redact_secrets(str(last_err))}")


_REMOTE_WAIT_SCRIPT = r"""
import json, subprocess, sys, time, urllib.request
from pathlib import Path

TIMEOUT_SEC = __TIMEOUT_SEC__
meta = {}
mf = Path("/root/colab_service_meta.json")
if mf.is_file():
    try:
        meta = json.loads(mf.read_text(encoding="utf-8"))
        if not isinstance(meta, dict): meta = {}
    except Exception:
        meta = {}
kind = str(meta.get("kind") or "comfy")
port = int(meta.get("port") or (18080 if kind == "llm" else 8188))
api_key = str(meta.get("api_key") or "").strip()

def _is_alive(pat: str) -> bool:
    try:
        r = subprocess.run(["pgrep", "-f", pat], stdin=subprocess.DEVNULL, capture_output=True, text=True)
        return r.returncode == 0 and bool((r.stdout or "").strip())
    except Exception:
        return True

def _tail(paths: list[str]) -> str:
    for ps in paths:
        p = Path(ps)
        if p.is_file():
            try: return f"--- {ps} ---\n" + p.read_text(encoding="utf-8", errors="ignore")[-2000:]
            except Exception: pass
    return ""

t0, last_print, dead_checks, conn_errors = time.time(), 0.0, 0, 0
is_llm = (kind == "llm")
url = f"http://127.0.0.1:{port}/slots" if is_llm else "http://127.0.0.1:8188/queue"
headers = {"Authorization": f"Bearer {api_key}"} if (is_llm and api_key) else {}
proc_pat = "llama-server" if is_llm else r"ComfyUI/main\.py|main\.py.*8188"
log_paths = ["/root/llama_server.log"] if is_llm else ["/root/comfy_colab.log", "/root/comfy_bw1.log"]

while True:
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=15) as r:
            data = json.loads(r.read().decode("utf-8") or ("[]" if is_llm else "{}"))
        if is_llm:
            busy = sum(1 for s in data if isinstance(s, dict) and s.get("is_processing")) if isinstance(data, list) else 0
            pending = 0
        else:
            busy, pending = len(data.get("queue_running") or []), len(data.get("queue_pending") or [])
        dead_checks = conn_errors = 0
    except Exception as exc:
        conn_errors += 1
        print(f"[等待队列] 查询异常 ({conn_errors}): {exc}", flush=True)
        busy = pending = -1
        if not _is_alive(proc_pat):
            dead_checks += 1
            if dead_checks >= 2:
                print(f"[熔断退出] 远端服务进程已消失，停止等待！\n{_tail(log_paths)}", file=sys.stderr, flush=True)
                sys.exit(2)
        elif conn_errors >= 12:
            print(f"[熔断退出] 远端服务连续 {conn_errors} 次无响应，中止等待！\n{_tail(log_paths)}", file=sys.stderr, flush=True)
            sys.exit(2)

    now = time.time()
    if busy == 0 and pending == 0:
        print(f"[等待队列] 队列已全部空闲（耗时 {now - t0:.1f} 秒）。", flush=True)
        break
    if now - t0 > TIMEOUT_SEC:
        print(f"[等待队列] 等待超时（{TIMEOUT_SEC} 秒），剩余 busy={busy}, pending={pending}", flush=True)
        sys.exit(1)
    if now - last_print >= 5.0:
        print(f"[等待队列] 运行中={busy}, 排队中={pending}, 已等待={now - t0:.0f}s", flush=True)
        last_print = now
    time.sleep(3 if is_llm else 5)
"""


def cmd_wait(args) -> None:
    _exec_remote_py(args.session, _REMOTE_WAIT_SCRIPT.replace("__TIMEOUT_SEC__", repr(int(args.timeout))), timeout=max(120, int(args.timeout) + 60))


def sync_session_outputs(session: str, prefix: str | None, pull: bool) -> None:
    sync_session_outputs_cli(session=session, prefix=prefix, pull=pull, colab_exe=COLAB_EXE, local_download_dir=LOCAL_DOWNLOAD_DIR, run_fn=_run, exec_remote_py_fn=_exec_remote_py, redact_fn=redact_secrets)


def cmd_sync(args) -> None:
    sync_session_outputs(args.session, args.prefix, args.pull)


def cmd_smoke(args) -> None:
    profile = getattr(args, "profile", "anima") or "anima"
    smoke_map = {
        "minimax": (MINIMAX_SMOKE_SCRIPT, 1800, "minimax_verify"),
        "drama3": (DRAMA3_SMOKE_SCRIPT, 1200, "drama3_verify"),
        "anima": (SMOKE_SCRIPT, 300, "xe_verify"),
    }
    if profile in LLM_PROFILES:
        _ensure_bin(QWEN_LLM_SMOKE_SCRIPT, "verify_qwen_llm.py")
        _colab_exec_file(args.session, QWEN_LLM_SMOKE_SCRIPT, timeout=300)
        return
    script, tout, sync_prefix = smoke_map.get(profile, smoke_map["anima"])
    _ensure_bin(script, script.name)
    _colab_exec_file(args.session, script, timeout=tout)
    sync_session_outputs(args.session, prefix=sync_prefix, pull=True)


def stop_session(session: str, clean_all: bool, target_endpoint: str | None = None, verify_release: bool = True) -> dict:
    return _core_stop_session_cli(session=session, clean_all=clean_all, target_endpoint=target_endpoint, verify_release=verify_release, colab_exe=COLAB_EXE, colab_py=COLAB_PY, run_fn=_run, status_fn=cmd_status)


def cmd_down(args) -> None:
    stop_session(args.session, args.all, target_endpoint=getattr(args, "endpoint", None))


def cmd_run(args) -> None:
    script_path = Path(args.script)
    if not script_path.is_absolute():
        cwd_cand = (Path.cwd() / script_path).resolve()
        script_path = cwd_cand if cwd_cand.is_file() else (_CTL_PKG_DIR / script_path).resolve()
    if not script_path.is_file():
        raise FileNotFoundError(f"未找到待执行脚本: {script_path}")

    norm_prefix = _normalize_prefix(args.prefix)
    manifest_out_dir = f"{REMOTE_OUT_ROOT}/{norm_prefix}" if norm_prefix else REMOTE_OUT_ROOT
    deps = _collect_local_py_deps(script_path)
    if deps:
        print(f"[依赖扫描] 自动随同上传 {len(deps)} 个同级依赖模块: {', '.join(sorted(deps))}", flush=True)

    bound_ep = str(getattr(args, "endpoint", None) or "").strip() or None
    if args.auto_stop and not bound_ep:
        bound_ep = _core_probe_session_endpoint(args.session, colab_py=COLAB_PY, run_fn=_run)
        if not bound_ep:
            print(f"[警告] 未能预先锁定会话 '{args.session}' 的目标实例 Endpoint，--auto-stop 停卡结果将无法精确核实（可用 --endpoint 显式指定）", file=sys.stderr, flush=True)

    ran_remote = synced = False
    primary_exc: BaseException | None = None
    try:
        ran_remote = True
        _colab_exec_file(args.session, script_path, timeout=300, remote_file=f"/content/{script_path.name}", extra_files=deps, manifest_out_dir=manifest_out_dir)
        if not args.no_wait:
            cmd_wait(argparse.Namespace(session=args.session, timeout=args.wait_timeout))
        if not args.no_sync:
            sync_session_outputs(args.session, prefix=args.prefix, pull=args.pull)
            synced = True
    except BaseException as exc:
        primary_exc = exc
        print(f"[任务中断] cmd_run 执行异常 ({type(exc).__name__}: {redact_secrets(str(exc))})", file=sys.stderr, flush=True)
        if ran_remote and not args.no_sync and not synced:
            print("[抢救同步] 尝试抢救同步云端已生成的输出与清单...", flush=True)
            try:
                sync_session_outputs(args.session, prefix=args.prefix, pull=args.pull)
            except Exception as sync_exc:
                print(f"[抢救跳过] 抢救同步未成功 ({redact_secrets(str(sync_exc))})", file=sys.stderr, flush=True)
        raise
    finally:
        if args.auto_stop:
            print("=== 触发 --auto-stop：强制释放 Colab 实例停止计费 ===", flush=True)
            try:
                stop_session(args.session, clean_all=False, target_endpoint=bound_ep)
            except BaseException as stop_exc:
                safe_stop_err = redact_secrets(str(stop_exc) or type(stop_exc).__name__)
                get_diag_logger().error("auto_stop", f"--auto-stop 停卡失败，实例仍在计费 (session={args.session}, endpoint={bound_ep or 'unknown'}): {safe_stop_err}")
                print(f"[严重告警] --auto-stop 自动释放实例失败，实例可能仍在计费，请立即手动停卡: {safe_stop_err}", file=sys.stderr, flush=True)
                if primary_exc is None:
                    raise
                raise RuntimeError(
                    f"任务执行中断 ({type(primary_exc).__name__}: {redact_secrets(str(primary_exc) or type(primary_exc).__name__)})，"
                    f"且 --auto-stop 自动停卡失败（实例仍在计费，请立即手动停卡）: {safe_stop_err}"
                ) from primary_exc


def cmd_doctor(args) -> None:
    profiles = _parse_profiles_arg(getattr(args, "profile", "anima"), None)
    gpu = getattr(args, "gpu", None) or _core_select_default_gpu(profiles)
    report = run_preflight_checks(
        mode="deep", profiles=profiles, gpu=gpu,
        tool_paths=ToolPaths(COLAB_EXE, COLAB_PY, RCLONE_EXE, RCLONE_CONF),
        state_path=_CTL_PKG_DIR / "colab_pet" / "pet_state.json",
        download_dir=LOCAL_DOWNLOAD_DIR, cf_config_path=_resolve_cf_cfg_path(), pkg_dir=_CTL_PKG_DIR,
    )
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2) if getattr(args, "json", False) else report.format_text(), flush=True)
    if not report.ok:
        raise SystemExit(1)


def cmd_diag(args) -> None:
    zip_path = export_diagnostic_bundle(
        out_dir=Path(args.out).resolve() if getattr(args, "out", None) else None,
        pkg_dir=_CTL_PKG_DIR, cf_config_path=_resolve_cf_cfg_path(),
        tool_paths=ToolPaths(COLAB_EXE, COLAB_PY, RCLONE_EXE, RCLONE_CONF),
    )
    print(f"已导出脱敏排障诊断包: {zip_path}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Google Colab + rclone 本机总控工具", formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p_loras = sub.add_parser("sync-loras", help="核对并上传本地 LoRA 到 Google Drive（零 CU 消耗）")
    p_loras.add_argument("--src", default=None, help="本地 LoRA 目录（默认 ./loras 或环境变量 COLAB_LORA_DIR）")
    p_loras.set_defaults(func=cmd_sync_loras)

    p_d3 = sub.add_parser("sync-drama3-refs", help="上传精选 Drama 3 参考音频 (.wav) 到 Google Drive（零 CU 消耗）")
    p_d3.add_argument("--src", default=None, help="本地参考音频目录")
    p_d3.set_defaults(func=cmd_sync_drama3_refs)

    sub.add_parser("status", help="查看当前活跃 Colab 会话及 CU 积分余量/消耗速率").set_defaults(func=cmd_status)

    p_up = sub.add_parser("up", help="创建或纳管 GPU 实例、上传 rclone 配置并部署指定单/多工作流生态")
    p_up.add_argument("--session", "-s", default="comfy")
    p_up.add_argument("--endpoint", "-e", default=None)
    p_up.add_argument("--profile", default="anima")
    p_up.add_argument("--profiles", default=None)
    p_up.add_argument("--gpu", default=None, choices=["T4", "L4", "G4", "A100", "H100"])
    p_up.add_argument("--high-mem", action="store_true")
    p_up.add_argument("--recreate", action="store_true")
    p_up.add_argument("--skip-preflight", action="store_true")
    p_up.add_argument("--minimax-lite", action="store_true")
    p_up.add_argument("--no-r2v", action="store_true")
    p_up.add_argument("--no-pe", action="store_true")
    p_up.add_argument("--llm-quant", default="UD-Q4_K_XL", choices=["UD-Q4_K_XL", "UD-IQ4_XS", "IQ3_XXS"])
    p_up.add_argument("--llm-ctx", type=int, default=None)
    p_up.add_argument("--no-mmproj", action="store_true")
    p_up.add_argument("--no-webui", action="store_true")
    p_up.add_argument("--gdrive-cache", action="store_true")
    p_up.add_argument("--cohost-comfy", action="store_true")
    p_up.add_argument("--llm-api-key", default="auto")
    p_up.set_defaults(func=cmd_up)

    p_tun = sub.add_parser("tunnel", help="查看或重启远端 ComfyUI / LLM Cloudflared 隧道并验证连通性")
    p_tun.add_argument("--session", "-s", default="comfy")
    p_tun.add_argument("--restart", action="store_true")
    p_tun.set_defaults(func=cmd_tunnel)

    p_smk = sub.add_parser("smoke", help="运行冒烟验证并同步/拉回结果")
    p_smk.add_argument("--session", "-s", default="comfy")
    p_smk.add_argument("--profile", default="anima", choices=["anima", "minimax", "drama3", "qwen_27b", "qwen_llm"])
    p_smk.set_defaults(func=cmd_smoke)

    p_wait = sub.add_parser("wait", help="等待远端队列清空后再返回")
    p_wait.add_argument("--session", "-s", default="comfy")
    p_wait.add_argument("--timeout", type=int, default=7200)
    p_wait.set_defaults(func=cmd_wait)

    p_sync = sub.add_parser("sync", help="同步远端输出图到 Google Drive，可选打包拉回本机")
    p_sync.add_argument("--session", "-s", default="comfy")
    p_sync.add_argument("--prefix", default=None)
    p_sync.add_argument("--pull", action="store_true")
    p_sync.set_defaults(func=cmd_sync)

    p_run = sub.add_parser("run", help="提交本地出图脚本到远端运行")
    p_run.add_argument("script")
    p_run.add_argument("--session", "-s", default="comfy")
    p_run.add_argument("--endpoint", "-e", default=None)
    p_run.add_argument("--prefix", default=None)
    p_run.add_argument("--wait-timeout", type=int, default=7200)
    p_run.add_argument("--no-wait", action="store_true")
    p_run.add_argument("--no-sync", action="store_true")
    p_run.add_argument("--pull", action="store_true")
    p_run.add_argument("--auto-stop", action="store_true")
    p_run.set_defaults(func=cmd_run)

    p_down = sub.add_parser("down", help="释放 Colab 实例并核实积分速率是否归零")
    p_down.add_argument("--session", "-s", default="comfy")
    p_down.add_argument("--endpoint", "-e", default=None)
    p_down.add_argument("--all", action="store_true")
    p_down.set_defaults(func=cmd_down)

    p_doc = sub.add_parser("doctor", help="执行启动前深度环境体检")
    p_doc.add_argument("--profile", "-p", default="anima")
    p_doc.add_argument("--gpu", "-g", default=None, choices=["T4", "L4", "G4", "A100", "H100"])
    p_doc.add_argument("--json", action="store_true")
    p_doc.set_defaults(func=cmd_doctor)

    p_diag = sub.add_parser("diag", help="生成全量脱敏的排障诊断报告压缩包 (.zip)")
    p_diag.add_argument("--out", "-o", default=None)
    p_diag.set_defaults(func=cmd_diag)
    return parser


def main(argv: list[str] | None = None) -> int:
    diag = get_diag_logger()
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
        return 0
    except SystemExit:
        raise
    except subprocess.CalledProcessError as exc:
        cmd_s = redact_secrets(" ".join(str(x) for x in (exc.cmd or [])))
        err_tail = ""
        for raw_src in (exc.stderr, exc.output):
            if raw_src:
                lines = [ln.strip() for ln in redact_secrets(str(raw_src)).splitlines() if ln.strip()]
                if lines:
                    err_tail = lines[-1]
                    break
        detail = err_tail or f"子进程退出码 {exc.returncode} ({cmd_s[:120]})"
        diag.error("ctl", f"CLI 执行失败: {detail}")
        print(f"[错误] {detail}", file=sys.stderr, flush=True)
        return 1
    except Exception as exc:
        msg = redact_secrets(str(exc) or type(exc).__name__)
        diag.error("ctl", f"CLI 异常 ({type(exc).__name__}): {msg}")
        print(f"[错误] {type(exc).__name__}: {msg}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
