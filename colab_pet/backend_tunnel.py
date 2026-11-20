"""Colab 桌宠公网隧道发现、直铺画布热推送与访问凭据分发子服务。"""

from __future__ import annotations

import base64
import json
import re
import tempfile
import time
import urllib.parse
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from colab_core.colab_client import build_tunnel_url_pattern
from colab_core.file_sync import GDRIVE_WF_REMOTE
from colab_core.runtime import CREATE_NO_WINDOW, clean_subprocess_env, normalize_profiles, suppress_logged
from colab_remote_common import DIRECT_CANVAS_JS
from colab_security import (
    build_comfy_handoff_url,
    clear_cf_tunnel_settings,
    redact_secrets,
    register_ephemeral_secret,
    save_cf_tunnel_settings,
    url_origin_key,
)

if TYPE_CHECKING:
    from colab_backend import ColabPetBackend


def _parse_fast_tunnel_payload(stdout: str) -> dict[str, Any] | None:
    """从 fast-tunnel 子进程标准输出中提取并反序列化标准帧 JSON。"""
    match = re.search(r"<<<COLAB_TUNNEL>>>(.*?)<<<END_COLAB_TUNNEL>>>", stdout or "", re.S)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(1))
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _extract_fast_tunnel_secrets(
    tdata: dict[str, Any],
) -> tuple[dict[str, Any] | None, str, str]:
    """提取并向脱敏器登记 fast-tunnel 响应中的远端元数据与内存密钥。"""
    raw_meta = tdata.get("meta")
    api_key = str(tdata.get("api_key") or "").strip()
    comfy_key = str(tdata.get("comfy_api_key") or "").strip()
    if isinstance(raw_meta, dict):
        if not api_key and raw_meta.get("api_key"):
            api_key = str(raw_meta.get("api_key") or "").strip()
        if not comfy_key and raw_meta.get("comfy_api_key"):
            comfy_key = str(raw_meta.get("comfy_api_key") or "").strip()
    if api_key:
        register_ephemeral_secret(api_key)
    if comfy_key:
        register_ephemeral_secret(comfy_key)
    return (raw_meta if isinstance(raw_meta, dict) else None), api_key, comfy_key


def _sanitize_remote_service_meta(
    raw_meta: dict[str, Any], fast_url: str | None
) -> dict[str, Any]:
    """清洗远端自描述元数据，剥离明文字段并补齐规范服务形态与默认端口。"""
    clean_meta = {k: v for k, v in raw_meta.items() if k not in ("api_key", "comfy_api_key")}
    kind_val = str(clean_meta.get("kind") or "comfy").lower()
    clean_meta["kind"] = kind_val if kind_val in ("comfy", "llm", "dual") else "comfy"
    clean_meta.setdefault("port", 18080 if clean_meta["kind"] == "llm" else 8188)
    if clean_meta["kind"] == "dual":
        clean_meta.setdefault("llm_port", 18080)
        if fast_url and not clean_meta.get("comfy_tunnel_url"):
            clean_meta["comfy_tunnel_url"] = fast_url
    if fast_url:
        clean_meta["tunnel_url"] = fast_url
    return clean_meta


def _format_comfy_open_message(workflow_name: str | None, stage: str) -> str:
    """根据是否携带工作流及命中阶段生成 ComfyUI 隧道打开提示文案。"""
    if workflow_name:
        return (
            f"已重启隧道并打开工作流: {workflow_name}"
            if stage == "restarted"
            else f"已在画布打开工作流: {workflow_name}"
        )
    if stage == "cache":
        return "已秒开 ComfyUI 隧道 (缓存命中)"
    if stage == "restarted":
        return "已重启并打开 ComfyUI 隧道网页"
    return "已打开 ComfyUI 隧道网页"


def _extract_target_ep_and_name(
    target: dict[str, Any] | None, fallback_session: str
) -> tuple[str, str, bool]:
    """从目标会话字典提取 `(endpoint, session_name, is_orphan)`。"""
    ep = str(target.get("endpoint") or "") if target else ""
    raw_name = target.get("name") if target else None
    sname = str(raw_name) if raw_name not in (None, "?") else fallback_session
    is_orphan = bool(target and target.get("orphan") and raw_name in (None, "?"))
    return ep, sname, is_orphan


class TunnelService:
    """负责 CF 固定/临时隧道解析、浏览器直连唤醒与服务凭据打包。"""

    def __init__(self, backend: ColabPetBackend) -> None:
        self._b = backend
        self._store = backend._store
        self._secrets = backend._secrets
        self._bus = backend._bus
        self._ctx = backend._ctx

    # ------------------------------------------------------------------
    # 隧道正则、账号槽位上下文与速读元数据
    # ------------------------------------------------------------------

    def tunnel_url_regex_pattern(self) -> str:
        """构造包含官方临时隧道及用户自定义固定域名的日志 URL 提取正则。"""
        cf_cfg = self._ctx.load_cf_secret_raw(
            config_path=self._ctx.resolve_cf_config_path(),
            include_secrets=False,
        )
        return build_tunnel_url_pattern(cf_cfg)

    def fast_read_slot_tunnel_url(
        self, ep: str, sname: str, account_id: str = ""
    ) -> tuple[str, str, str | None]:
        """按指定账号上下文速读隧道与元数据，仍经由 `_fast_read_tunnel_url` 以兼容单测桩。"""
        b = self._b
        prev_acct = getattr(b._tunnel_acct_ctx, "account_id", "")
        b._tunnel_acct_ctx.account_id = str(account_id or "").strip()
        try:
            return b._fast_read_tunnel_url(ep, sname)
        finally:
            b._tunnel_acct_ctx.account_id = prev_acct

    def _resolve_fast_read_context(self, ep: str) -> tuple[str, str]:
        """解析 fast-tunnel 调用应使用的目标端点与所属账号 ID。"""
        b = self._b
        use_ep = str(ep or "").strip()
        ctx_acct = str(getattr(b._tunnel_acct_ctx, "account_id", "") or "").strip()
        if not ctx_acct and use_ep:
            ctx_acct = b._cluster_svc.resolve_account_for_endpoint(ep=use_ep)
        if not ctx_acct and not use_ep and b._cluster_svc.has_active_slots():
            fallback_slot = b._cluster_svc.get_slot("comfy") or b._cluster_svc.get_slot("llm")
            if fallback_slot:
                ctx_acct = str(fallback_slot.get("account_id") or "").strip()
                use_ep = str(fallback_slot.get("endpoint") or "").strip()
        if not ctx_acct and not b._cluster_svc.has_active_slots():
            # 单机实例沿用部署时记录的 colab-cli 账号作用域，避免读错会话文件
            ctx_acct = self._store.resolve_single_node_account_id(use_ep)
        return use_ep, ctx_acct

    def _handle_fast_read_no_session(
        self, ep: str, ctx_acct: str
    ) -> tuple[str, str, str | None]:
        """在远端返回 `no_session` 时按单号/集群模式安全清理过期会话与缓存。"""
        if self._b._cluster_svc.has_active_slots() and ctx_acct:
            return "no_session", ep, None
        with self._store.lock:
            state = self._store.state
            state["last_sessions"] = []
            state["last_assignments_count"] = 0
            state["tunnel_cache"] = {}
            state["managed_binding"] = None
            self._secrets.retain_endpoints(self._store.get_active_cluster_endpoints())
            self._store.save()
        return "no_session", ep, None

    def _store_fast_tunnel_metadata(
        self,
        resolved_ep: str,
        fast_url: str | None,
        raw_meta: dict[str, Any] | None,
        api_key: str,
        comfy_key: str,
        ctx_acct: str,
    ) -> None:
        """将速读到的远端元数据与公网凭据安全写入纯内存 `EphemeralSecretStore`。"""
        if not resolved_ep:
            return
        raw_kind = str(raw_meta.get("kind") or "comfy").lower() if raw_meta else "comfy"
        comfy_bind_u = (
            str((raw_meta.get("comfy_tunnel_url") if raw_meta else "") or fast_url or "").strip()
            if raw_kind != "llm"
            else ""
        )
        comfy_orig = url_origin_key(comfy_bind_u) if comfy_bind_u else ""
        eff_acct = str(
            ctx_acct
            or self._b._cluster_svc.resolve_account_for_endpoint(
                ep=resolved_ep, base_url=comfy_bind_u
            )
            or self._store.get_active_account_id()
            or ""
        ).strip()

        with self._store.lock:
            self._secrets.comfy_key_checked_eps.add(resolved_ep)
            if comfy_key:
                self._secrets.set_comfy_secret(
                    resolved_ep, comfy_key, account_id=eff_acct, origin=comfy_orig
                )

        if raw_meta is not None:
            clean_meta = _sanitize_remote_service_meta(raw_meta, fast_url)
            kind = clean_meta["kind"]
            runtime_secret: str | None = None
            if kind in ("llm", "dual"):
                runtime_secret = api_key if api_key else None
            else:
                runtime_secret = ""
            self._secrets.set_ep_meta(
                resolved_ep, clean_meta, runtime_secret=runtime_secret
            )

    def fast_read_tunnel_url(self, ep: str, sname: str) -> tuple[str, str, str | None]:
        """通过 Colab Contents HTTP 接口免 WebSocket 速读 `/root/colab_service_meta.json` 或隧道文件。"""
        colab_py = self._ctx.get_colab_py()
        if not colab_py.exists():
            return "error", ep, None
        use_ep, ctx_acct = self._resolve_fast_read_context(ep)
        cmd = [
            str(colab_py),
            str(self._ctx.cli_bridge_script),
            "fast-tunnel",
            use_ep,
            sname,
            self._b._tunnel_url_regex_pattern(),
        ]
        if ctx_acct:
            cmd.append(f"--acct={ctx_acct}")
        fast_proc = self._ctx.run_hidden(cmd, timeout=15)
        if fast_proc.returncode != 0:
            return "error", ep, None
        tdata = _parse_fast_tunnel_payload(fast_proc.stdout or "")
        if tdata is None:
            return "error", ep, None

        tstatus = str(tdata.get("status") or "missing")
        if tstatus == "no_session":
            return self._handle_fast_read_no_session(ep, ctx_acct)

        resolved_ep = str(tdata.get("endpoint") or use_ep or ep)
        fast_url = str(tdata.get("url") or "").strip() or None
        raw_meta, api_key, comfy_key = _extract_fast_tunnel_secrets(tdata)
        self._store_fast_tunnel_metadata(
            resolved_ep, fast_url, raw_meta, api_key, comfy_key, ctx_acct
        )
        return tstatus, resolved_ep, fast_url

    def read_cf_public_llm_url(self) -> str | None:
        """从本地云端隧道配置中仅读取公开 `llm_url`（绝不读取或暴露敏感 token/key）。"""
        cf_cfg_p = self._ctx.resolve_cf_config_path()
        data = self._ctx.load_cf_secret_raw(config_path=cf_cfg_p, include_secrets=False)
        u = str(data.get("llm_url") or "").strip().rstrip("/")
        if u and self._ctx.is_allowed_tunnel_url(u, config_path=cf_cfg_p):
            return u
        return None

    def infer_default_llm_model_name(self, meta: dict[str, Any] | None = None) -> str:
        """按服务元数据或活跃/预设工作流推断默认 LLM 模型显示名（27B vs 180B）。"""
        if isinstance(meta, dict) and meta.get("model"):
            return str(meta["model"])
        with self._b._comfy_lock:
            act_profs = list(self._b._comfy_state.get("active_profiles") or [])
        prefs = self._store.get_prefs()
        pref_profs = normalize_profiles(
            prefs.get("boot_profiles"), prefs.get("boot_profile") or "anima"
        )
        profs = act_profs or pref_profs
        if "qwen_27b" in profs and "qwen_llm" not in profs:
            return "Qwen3.8-27B-abliterated"
        return "Qwen3.8-Flash-Next-abliterated"

    # ------------------------------------------------------------------
    # 活跃主隧道与 LLM 隧道解析
    # ------------------------------------------------------------------

    def _resolve_cluster_active_url(self, session: str) -> tuple[bool, str | None]:
        """在多号异构集群模式下直接按槽位解析活跃隧道 URL。"""
        cluster_svc = self._b._cluster_svc
        if not cluster_svc.has_active_slots():
            return False, None
        slots = cluster_svc.get_cluster_slots()
        if session == "llm" and "llm" in slots:
            return True, cluster_svc._resolve_slot_tunnel_url("llm", slots["llm"])
        if "comfy" in slots:
            return True, cluster_svc._resolve_slot_tunnel_url("comfy", slots["comfy"])
        if "llm" in slots:
            return True, cluster_svc._resolve_slot_tunnel_url("llm", slots["llm"])
        return False, None

    def _read_cached_tunnel_context(
        self, ep: str
    ) -> tuple[str | None, dict[str, Any] | None, str]:
        """持主锁原子读取缓存隧道 URL、已知端点元数据与预期服务偏好类型。"""
        cached_url: str | None = None
        known_meta: dict[str, Any] | None = None
        preferred_kind = "comfy"
        with self._store.lock:
            cache = self._store.get_tunnel_cache()
            if ep and ep in cache:
                cached_url = cache.get(ep)
            elif (not ep) and len(cache) == 1:
                cached_url = next(iter(cache.values()), None)
            if ep and ep in self._secrets.service_meta_by_ep:
                known_meta = dict(self._secrets.service_meta_by_ep[ep])
            prefs = self._store.get_prefs()
            pref_profs = prefs.get("boot_profiles") or [prefs.get("boot_profile") or "anima"]
            if self._b._infer_kind_from_profiles(pref_profs) == "llm":
                preferred_kind = "llm"
        return cached_url, known_meta, preferred_kind

    def _persist_tunnel_cache(self, ep: str, url: str) -> None:
        """将已验证存活的公网隧道地址写入持久化缓存。"""
        if not ep or not url:
            return
        with self._store.lock:
            self._store.set_tunnel_cache(ep, url)
            self._store.save()

    def _probe_known_meta_tunnel(
        self, ep: str, cached_url: str | None, known_meta: dict[str, Any], cf_cfg_p: Path
    ) -> str | None:
        """基于已知远端元数据探测缓存隧道是否在线可用。"""
        mkind = str(known_meta.get("kind") or "comfy")
        if mkind == "dual":
            cand_comfy = str(known_meta.get("comfy_tunnel_url") or cached_url or "").strip()
            cand_tok = self._b._get_comfy_key_for_url(cand_comfy, ep)
            if cand_comfy and self._ctx.probe_tunnel_alive(
                cand_comfy,
                timeout=3.0,
                kind="comfy",
                auth_token=cand_tok,
                cf_config_path=cf_cfg_p,
            ):
                return cand_comfy
            return None
        comfy_tok = self._b._get_comfy_key_for_url(cached_url or "", ep)
        if cached_url and self._ctx.probe_tunnel_alive(
            cached_url,
            timeout=3.0,
            kind=mkind,
            auth_token=comfy_tok,
            cf_config_path=cf_cfg_p,
        ):
            return cached_url
        return None

    def _verify_or_hydrate_cached_url(
        self,
        ep: str,
        sname: str,
        cached_url: str,
        preferred_kind: str,
        cf_cfg_p: Path,
        allow_fast_discover: bool,
    ) -> str | None:
        """在缺少元数据时探测缓存隧道形态，并按冷却间隔顺带补水远端元数据。"""
        comfy_tok = self._b._get_comfy_key_for_url(cached_url, ep)
        detected = self._ctx.detect_tunnel_kind(
            cached_url,
            timeout=3.0,
            preferred=preferred_kind,
            auth_token=comfy_tok,
            cf_config_path=cf_cfg_p,
        )
        if detected is None:
            return None
        if allow_fast_discover:
            now_m = self._ctx.monotonic()
            if now_m - self._b._last_tunnel_discover_ts >= 15.0:
                self._b._last_tunnel_discover_ts = now_m
                hydrated = self._try_upgrade_cached_url_via_fast_read(
                    ep, sname, cached_url, detected, cf_cfg_p
                )
                if hydrated:
                    return hydrated
        if ep:
            with self._store.lock:
                self._secrets.service_meta_by_ep.setdefault(
                    ep,
                    {"kind": detected, "port": 18080 if detected == "llm" else 8188},
                )
        return cached_url

    def _try_upgrade_cached_url_via_fast_read(
        self,
        ep: str,
        sname: str,
        cached_url: str,
        detected: str,
        cf_cfg_p: Path,
    ) -> str | None:
        """通过 fast-tunnel 补水元数据并在远端隧道已轮换时切换至新地址。"""
        tstatus, resolved_ep, fast_url = self._b._fast_read_tunnel_url(ep, sname)
        meta_hydrated, _ = self._secrets.get_ep_meta_and_key(resolved_ep or ep)
        hkind = str(meta_hydrated.get("kind") or detected)
        h_probe_k = "comfy" if hkind == "dual" else hkind
        htok = self._b._get_comfy_key_for_url(fast_url or "", resolved_ep or ep)
        if (
            tstatus == "ok"
            and fast_url
            and fast_url != cached_url
            and self._ctx.probe_tunnel_alive(
                fast_url,
                timeout=3.0,
                kind=h_probe_k,
                auth_token=htok,
                cf_config_path=cf_cfg_p,
            )
        ):
            self._persist_tunnel_cache(resolved_ep, fast_url)
            return fast_url
        return None

    def _discover_tunnel_via_fast_read(
        self, ep: str, sname: str, cf_cfg_p: Path
    ) -> str | None:
        """按 45s 冷却策略调用 fast-tunnel 重新发现并验证存活的主隧道 URL。"""
        now_m = self._ctx.monotonic()
        if now_m - self._b._last_tunnel_discover_ts < 45.0:
            return None
        self._b._last_tunnel_discover_ts = now_m

        tstatus, resolved_ep, fast_url = self._b._fast_read_tunnel_url(ep, sname)
        if tstatus != "ok" or not fast_url:
            return None
        meta_after, _ = self._secrets.get_ep_meta_and_key(resolved_ep or ep)
        mkind_after = str(meta_after.get("kind") or "auto")
        probe_kind = "comfy" if mkind_after == "dual" else mkind_after
        ftok = self._b._get_comfy_key_for_url(fast_url, resolved_ep or ep)
        if self._ctx.probe_tunnel_alive(
            fast_url,
            timeout=3.5,
            kind=probe_kind,
            auth_token=ftok,
            cf_config_path=cf_cfg_p,
        ):
            self._persist_tunnel_cache(resolved_ep, fast_url)
            return fast_url
        return None

    def resolve_active_tunnel_url(
        self, session: str = "comfy", allow_fast_discover: bool = True
    ) -> str | None:
        """获取当前可连通的主公网隧道地址（ComfyUI/dual 优先 Comfy 主页，纯 LLM 返回对话页，支持补水）。"""
        handled, cluster_url = self._resolve_cluster_active_url(session)
        if handled:
            return cluster_url
        if not self._store.has_running_session():
            return None
        target = self._store.pick_target_session(session)
        ep, sname, _ = _extract_target_ep_and_name(target, session)
        cached_url, known_meta, preferred_kind = self._read_cached_tunnel_context(ep)
        cf_cfg_p = self._ctx.resolve_cf_config_path()

        if known_meta is not None:
            hit = self._probe_known_meta_tunnel(ep, cached_url, known_meta, cf_cfg_p)
            if hit:
                return hit
        elif cached_url:
            hit = self._verify_or_hydrate_cached_url(
                ep, sname, cached_url, preferred_kind, cf_cfg_p, allow_fast_discover
            )
            if hit:
                return hit

        if not allow_fast_discover:
            return None
        return self._discover_tunnel_via_fast_read(ep, sname, cf_cfg_p)

    def _try_fast_discover_llm_url(
        self, ep: str, sname: str, meta: dict[str, Any], cf_cfg_p: Path
    ) -> tuple[str, str | None, str]:
        """在冷却允许时执行 fast-tunnel 发现 LLM 隧道，返回 `(状态, URL, 最新形态)`。"""
        now_m = self._ctx.monotonic()
        if meta and (now_m - self._b._last_llm_tunnel_discover_ts < 15.0):
            return "miss", None, str(meta.get("kind") or "").lower()
        self._b._last_llm_tunnel_discover_ts = now_m
        tstatus, resolved_ep, fast_url = self._b._fast_read_tunnel_url(ep, sname)
        if tstatus == "no_session":
            return "stop", None, ""
        meta_new, _ = self._secrets.get_ep_meta_and_key(resolved_ep or ep)
        mkind = str(meta_new.get("kind") or "").lower()
        if mkind == "comfy" and not self._b.is_dual_mode_active():
            return "stop", None, mkind
        cand = str(
            meta_new.get("llm_tunnel_url") or (fast_url if mkind == "llm" else "") or ""
        ).strip()
        if cand and self._ctx.probe_tunnel_alive(
            cand, timeout=3.0, kind="llm", cf_config_path=cf_cfg_p
        ):
            return "found", cand, mkind
        return "miss", None, mkind

    def _try_cf_fallback_llm_url(self, ep: str, cf_cfg_p: Path) -> str | None:
        """探测本地 CF 固定隧道 `llm_url` 并在存活时缓存至端点元数据。"""
        cf_llm = self._b._read_cf_public_llm_url()
        if not cf_llm:
            return None
        if not self._ctx.probe_tunnel_alive(
            cf_llm, timeout=3.0, kind="llm", cf_config_path=cf_cfg_p
        ):
            return None
        if ep:
            with self._store.lock:
                cur = self._secrets.service_meta_by_ep.setdefault(
                    ep,
                    {"kind": "dual" if self._b.is_dual_mode_active() else "llm", "port": 8188},
                )
                cur["llm_tunnel_url"] = cf_llm
        return cf_llm

    def resolve_active_llm_tunnel_url(
        self, session: str = "comfy", allow_fast_discover: bool = True
    ) -> str | None:
        """获取当前在线 LLM 服务公网隧道地址（支持纯 LLM 模式与 Comfy+LLM 双栖模式）。"""
        cluster_svc = self._b._cluster_svc
        if cluster_svc.has_active_slots():
            llm_slot = cluster_svc.get_slot("llm")
            if llm_slot is None:
                return None
            return cluster_svc._resolve_slot_tunnel_url("llm", llm_slot)
        if not self._store.has_running_session():
            return None
        target = self._store.pick_target_session(session)
        ep, sname, _ = _extract_target_ep_and_name(target, session)

        meta, _ = self._secrets.get_ep_meta_and_key(ep)
        mkind = str(meta.get("kind") or "").lower()
        if mkind == "llm":
            return self._b.resolve_active_tunnel_url(
                session=session, allow_fast_discover=allow_fast_discover
            )
        if mkind == "comfy" and not self._b.is_dual_mode_active():
            return None

        cf_cfg_p = self._ctx.resolve_cf_config_path()
        cand = str(meta.get("llm_tunnel_url") or "").strip() if mkind == "dual" else ""
        if cand and self._ctx.probe_tunnel_alive(
            cand, timeout=3.0, kind="llm", cf_config_path=cf_cfg_p
        ):
            return cand

        if allow_fast_discover:
            disc_status, disc_url, mkind = self._try_fast_discover_llm_url(
                ep, sname, meta, cf_cfg_p
            )
            if disc_status == "found":
                return disc_url
            if disc_status == "stop":
                return None

        if mkind in ("dual", "llm") or self._b.has_llm_service_active():
            return self._try_cf_fallback_llm_url(ep, cf_cfg_p)
        return None

    # ------------------------------------------------------------------
    # 直铺画布扩展热检测、热安装与浏览器直推
    # ------------------------------------------------------------------

    @staticmethod
    def build_comfy_open_url(
        tunnel_url: str,
        workflow_name: str | None = None,
        extra_names: list[str] | None = None,
        now_ts_ms: int | None = None,
    ) -> str:
        """构造 ComfyUI 打开链接；指定工作流名时附加一次性直铺画布查询参数。"""
        base = tunnel_url.rstrip("/")
        if not workflow_name:
            return base
        ts_val = int(time.time() * 1000) if now_ts_ms is None else int(now_ts_ms)
        q = f"?pet_wf={urllib.parse.quote(workflow_name)}&pet_ts={ts_val}"
        extras = [str(n) for n in (extra_names or []) if n and n != workflow_name]
        if extras:
            q += f"&pet_extra={urllib.parse.quote(','.join(extras))}"
        return f"{base}/{q}"

    def ensure_legacy_cmd_file(self, base_url: str, ep: str = "") -> None:
        """确保 userdata 根目录存在桥接指令文件，避免浏览器首次轮询出现 404 日志。"""
        base = base_url.rstrip("/")
        with suppress_logged("tunnel.legacy_cmd", throttled_key=f"legacy_cmd:{ep or base}"):
            chk = self._b._comfy_http_get(
                f"{base}/userdata/colab_pet_bridge_cmd.json", base_url=base, ep=ep, timeout=4
            )
            if chk.status_code == 404:
                self._b._comfy_http_post(
                    f"{base}/userdata/colab_pet_bridge_cmd.json?overwrite=true",
                    base_url=base,
                    ep=ep,
                    json={"id": "", "name": "", "ts": 0},
                    timeout=5,
                )

    def _resolve_comfy_slot_target_ctx(self, ep: str, base: str) -> tuple[str, str]:
        """解析 ComfyUI 端点及所属集群账号 ID（供扩展安装与浏览器票据补水共用）。"""
        cluster_svc = self._b._cluster_svc
        use_ep = ep
        slot_acct = cluster_svc.resolve_account_for_endpoint(ep=use_ep, base_url=base)
        if (not slot_acct) and cluster_svc.has_active_slots():
            c_slot = cluster_svc.get_slot("comfy")
            if c_slot:
                slot_acct = str(c_slot.get("account_id") or "").strip()
                if (not use_ep) and c_slot.get("endpoint"):
                    use_ep = str(c_slot["endpoint"])
        return use_ep, slot_acct

    def _is_direct_canvas_js_v2(self, url: str, base: str, ep: str) -> bool:
        """检查指定扩展脚本 URL 是否返回 `colab_pet_direct_canvas v2` 内容。"""
        try:
            resp = self._b._comfy_http_get(url, base_url=base, ep=ep, timeout=4)
            return resp.status_code == 200 and "colab_pet_direct_canvas v2" in (resp.text or "")
        except Exception:
            return False

    def _probe_native_canvas_ext(self, base: str, ep: str) -> tuple[bool, bool]:
        """探测云端原生 `colab_pet_bridge` 扩展，返回 `(原生路由存活, v2已就绪)`。"""
        for attempt in range(2):
            try:
                r = self._b._comfy_http_get(
                    f"{base}/colab_pet/ping", base_url=base, ep=ep, timeout=4
                )
                if r.status_code == 200:
                    pdata = r.json()
                    if isinstance(pdata, dict) and pdata.get("ok"):
                        if int(pdata.get("version") or 1) >= 2:
                            return True, True
                        ready = self._is_direct_canvas_js_v2(
                            f"{base}/extensions/colab_pet_bridge/direct_canvas.js", base, ep
                        )
                        return True, ready
                return False, False
            except Exception:
                if attempt == 0:
                    self._ctx.sleep(0.25)
        return False, False

    def _probe_legacy_canvas_ext(self, base: str, ep: str) -> bool:
        """探测旧版 `colab_pet` 前端直铺扩展是否已安装且为 v2。"""
        url = f"{base}/extensions/colab_pet/direct_canvas.js"
        for attempt in range(2):
            try:
                r2 = self._b._comfy_http_get(url, base_url=base, ep=ep, timeout=4)
                return r2.status_code == 200 and "colab_pet_direct_canvas v2" in (r2.text or "")
            except Exception:
                if attempt == 0:
                    self._ctx.sleep(0.25)
        return False

    def _install_and_verify_canvas_ext(
        self, base: str, ep: str, sname: str, native_route_alive: bool
    ) -> str | None:
        """经本机 CLI 桥热注入直铺扩展并轮询确认浏览器可访问。"""
        colab_py = self._ctx.get_colab_py()
        if not colab_py.exists() or not self._store.has_running_session():
            return None

        payload = {
            "js_b64": base64.b64encode(DIRECT_CANVAS_JS.encode("utf-8")).decode("ascii"),
            "py_b64": base64.b64encode(
                self._ctx.bridge_node_init_py.encode("utf-8")
            ).decode("ascii"),
        }
        use_ep, slot_acct = self._resolve_comfy_slot_target_ctx(ep, base)
        ext_cmd = [str(colab_py), str(self._ctx.cli_bridge_script), "install-ext", use_ep, sname]
        if slot_acct:
            ext_cmd.append(f"--acct={slot_acct}")
        proc = self._ctx.subprocess_mod.run(
            ext_cmd,
            input=json.dumps(payload, ensure_ascii=False),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            env=clean_subprocess_env(),
            creationflags=CREATE_NO_WINDOW,
        )
        if proc.returncode != 0:
            return None

        check_urls = (
            [
                f"{base}/extensions/colab_pet_bridge/direct_canvas.js",
                f"{base}/extensions/colab_pet/direct_canvas.js",
            ]
            if native_route_alive
            else [f"{base}/extensions/colab_pet/direct_canvas.js"]
        )
        target_mode = "native" if native_route_alive else "legacy"
        for _ in range(3):
            for cu in check_urls:
                if self._is_direct_canvas_js_v2(cu, base, ep):
                    if target_mode == "legacy":
                        self._b._ensure_legacy_cmd_file(base, ep=ep)
                    return target_mode
            self._ctx.sleep(0.25)
        return None

    def ensure_direct_canvas_support(
        self, tunnel_url: str, ep: str = "", sname: str = "comfy"
    ) -> tuple[str, bool]:
        """探测或免重启热注入/热升级云端 ComfyUI 直铺画布前端扩展，返回 `(模式, 本次是否刚安装)`。"""
        base = tunnel_url.rstrip("/")
        cache_key = ep or base
        with self._b._canvas_ext_lock:
            cached = self._b._canvas_ext_ready_eps.get(cache_key)
        if cached in ("native", "legacy"):
            return cached, False

        native_route_alive, native_ready = self._probe_native_canvas_ext(base, ep)
        if native_ready:
            with self._b._canvas_ext_lock:
                self._b._canvas_ext_ready_eps[cache_key] = "native"
            return "native", False

        if not native_route_alive and self._probe_legacy_canvas_ext(base, ep):
            self._b._ensure_legacy_cmd_file(base, ep=ep)
            with self._b._canvas_ext_lock:
                self._b._canvas_ext_ready_eps[cache_key] = "legacy"
            return "legacy", False

        with suppress_logged("tunnel.canvas_ext", throttled_key=f"canvas_ext:{cache_key}"):
            installed_mode = self._install_and_verify_canvas_ext(
                base, ep, sname, native_route_alive
            )
            if installed_mode in ("native", "legacy"):
                with self._b._canvas_ext_lock:
                    self._b._canvas_ext_ready_eps[cache_key] = installed_mode
                return installed_mode, True
        return ("native" if native_route_alive else "none"), False

    def launch_comfy_browser(
        self,
        tunnel_url: str,
        open_url: str,
        *,
        ep: str = "",
        sname: str = "comfy",
        already_fetched_meta: bool = False,
    ) -> None:
        """打开 ComfyUI 浏览器窗口：存在公网保护密钥时自动签发一次性票据实现免输口令直进。"""
        base = tunnel_url.strip().rstrip("/")
        use_ep, slot_acct = self._resolve_comfy_slot_target_ctx(ep, base)
        comfy_key = self._b._get_comfy_key_for_url(base, use_ep)
        with self._store.lock:
            checked = bool(use_ep and use_ep in self._secrets.comfy_key_checked_eps)
        if (
            not comfy_key
            and not already_fetched_meta
            and not checked
            and self._store.has_running_session()
        ):
            _, resolved_ep, _ = self.fast_read_slot_tunnel_url(use_ep, sname, account_id=slot_acct)
            comfy_key = self._b._get_comfy_key_for_url(base, resolved_ep or use_ep)

        launch_url = open_url
        if comfy_key:
            next_rel = open_url[len(base):] if open_url.startswith(base) else "/"
            if not next_rel.startswith("/"):
                next_rel = f"/{next_rel}" if next_rel else "/"
            launch_url = build_comfy_handoff_url(base, comfy_key, next_path=next_rel)
        self._ctx.open_browser(launch_url)

    def _push_workflow_live_native(
        self, base: str, ep: str, req_id: str, safe_name: str, extra_names: list[str]
    ) -> bool:
        """通过原生 `colab_pet_bridge` 接口实时推送工作流并等待浏览器回执。"""
        with suppress_logged("tunnel.push_native", throttled_key=f"push_native:{ep or base}"):
            pr = self._b._comfy_http_post(
                f"{base}/colab_pet/open_workflow",
                base_url=base,
                ep=ep,
                json={"id": req_id, "name": safe_name, "extra_names": extra_names},
                timeout=5,
            )
            if pr.status_code != 200 or int((pr.json() or {}).get("browser_clients") or 0) <= 0:
                return False
            deadline = self._ctx.monotonic() + 3.2
            while self._ctx.monotonic() < deadline:
                self._ctx.sleep(0.18)
                with suppress_logged("tunnel.push_native_ack", throttled_key=f"push_native_ack:{ep or base}"):
                    ar = self._b._comfy_http_get(
                        f"{base}/colab_pet/ack_workflow",
                        base_url=base,
                        ep=ep,
                        params={"id": req_id},
                        timeout=3,
                    )
                    if ar.status_code == 200 and (ar.json() or {}).get("acked"):
                        return True
        return False

    def _push_workflow_live_legacy(
        self, base: str, ep: str, req_id: str, safe_name: str, extra_names: list[str]
    ) -> bool:
        """通过旧版 userdata 命令文件推送工作流并等待浏览器回执。"""
        with suppress_logged("tunnel.push_legacy", throttled_key=f"push_legacy:{ep or base}"):
            pr = self._b._comfy_http_post(
                f"{base}/userdata/colab_pet_bridge_cmd.json?overwrite=true",
                base_url=base,
                ep=ep,
                json={
                    "id": req_id,
                    "name": safe_name,
                    "extra_names": extra_names,
                    "ts": self._ctx.now(),
                },
                timeout=5,
            )
            if pr.status_code not in (200, 201):
                return False
            deadline = self._ctx.monotonic() + 3.5
            while self._ctx.monotonic() < deadline:
                self._ctx.sleep(0.25)
                with suppress_logged("tunnel.push_legacy_ack", throttled_key=f"push_legacy_ack:{ep or base}"):
                    ar = self._b._comfy_http_get(
                        f"{base}/userdata/colab_pet_bridge_ack.json",
                        base_url=base,
                        ep=ep,
                        timeout=3,
                    )
                    if ar.status_code == 200 and str((ar.json() or {}).get("id") or "") == req_id:
                        return True
        return False

    def push_or_open_workflow_on_canvas(
        self,
        tunnel_url: str,
        safe_name: str,
        extra_names: list[str] | None = None,
        session: str = "comfy",
    ) -> dict[str, Any]:
        """优先向已打开的 ComfyUI 浏览器标签页实时推送工作流；无标签页回执时自动拉起浏览器直铺。"""
        base = tunnel_url.rstrip("/")
        cluster_target, _ = self._resolve_cluster_slot_target("comfy", session)
        target = (
            cluster_target
            if cluster_target is not None
            else self._store.pick_target_session(session)
        )
        ep, sname, _ = _extract_target_ep_and_name(target, session)

        mode, just_installed = self._b._ensure_direct_canvas_support(base, ep=ep, sname=sname)
        req_id = uuid.uuid4().hex[:12]
        open_url = self._b._build_comfy_open_url(
            base, safe_name, extra_names=extra_names, now_ts_ms=int(self._ctx.now() * 1000)
        )
        extras = extra_names or []

        pushed_live = False
        if not just_installed:
            if mode == "native":
                pushed_live = self._push_workflow_live_native(base, ep, req_id, safe_name, extras)
            elif mode == "legacy":
                pushed_live = self._push_workflow_live_legacy(base, ep, req_id, safe_name, extras)

        if pushed_live:
            return {
                "pushed_live": True,
                "opened_browser": False,
                "url": open_url,
                "mode": mode,
            }
        self._b._launch_comfy_browser(base, open_url, ep=ep, sname=sname)
        return {
            "pushed_live": False,
            "opened_browser": True,
            "url": open_url,
            "mode": mode,
        }

    def backup_workflow_to_gdrive_async(self, safe_name: str, wf_bytes: bytes) -> None:
        """在后台静默同步一份工作流 JSON 至 Google Drive，实现跨实例开机自动继承。"""
        rclone_exe = self._ctx.get_rclone_exe()
        rclone_conf = self._ctx.get_rclone_conf()
        if not rclone_exe.is_file() or not rclone_conf.is_file():
            return

        def _bg(_cancel: Any) -> None:
            with tempfile.TemporaryDirectory(prefix="pet_wf_bak_") as tmp_dir:
                tmp_wf = Path(tmp_dir) / safe_name
                tmp_wf.write_bytes(wf_bytes)
                self._b._runner.run(
                    [
                        str(rclone_exe),
                        "copyto",
                        "--config",
                        str(rclone_conf),
                        "--disable-http2",
                        str(tmp_wf),
                        f"{GDRIVE_WF_REMOTE}/{safe_name}",
                    ],
                    timeout=45,
                    run_fn=self._ctx.run_hidden,
                    task_name="backup_workflow",
                )

        self._b._tasks.submit("backup_workflow", _bg, exclusive=False)

    def open_browser_for_comfy(
        self,
        tunnel_url: str,
        ep: str = "",
        sname: str = "comfy",
        workflow_name: str | None = None,
        already_fetched_meta: bool = False,
    ) -> str:
        """装配 ComfyUI 打开链接并拉起默认浏览器（含公网访问保护时自动附带一次性免密直进票据）。"""
        if workflow_name:
            self._b._ensure_direct_canvas_support(tunnel_url, ep=ep, sname=sname)
        open_url = self._b._build_comfy_open_url(
            tunnel_url, workflow_name, now_ts_ms=int(self._ctx.now() * 1000)
        )
        self._b._launch_comfy_browser(
            tunnel_url,
            open_url,
            ep=ep,
            sname=sname,
            already_fetched_meta=already_fetched_meta,
        )
        return open_url

    def _resolve_service_meta_for_open(
        self,
        tunnel_url: str,
        ep: str,
        sname: str,
        detected_kind: str | None,
        already_fetched_meta: bool,
    ) -> tuple[str, str, dict[str, Any], str, bool]:
        """在打开隧道前补齐端点归属账号与远端元数据。"""
        cluster_svc = self._b._cluster_svc
        resolved_ep = ep
        slot_acct = cluster_svc.resolve_account_for_endpoint(ep=resolved_ep, base_url=tunnel_url)
        if (not slot_acct) and cluster_svc.has_active_slots():
            pref_slot = "llm" if str(detected_kind or "").lower() == "llm" else "comfy"
            picked = (
                cluster_svc.get_slot(pref_slot)
                or cluster_svc.get_slot("llm")
                or cluster_svc.get_slot("comfy")
            )
            if picked:
                slot_acct = str(picked.get("account_id") or "").strip()
                if (not resolved_ep) and picked.get("endpoint"):
                    resolved_ep = str(picked["endpoint"])

        meta, api_key = self._secrets.get_ep_meta_and_key(resolved_ep)
        fetched = already_fetched_meta
        if not meta and not fetched:
            _, r_ep, _ = self.fast_read_slot_tunnel_url(resolved_ep, sname, account_id=slot_acct)
            if r_ep:
                resolved_ep = r_ep
            meta, api_key = self._secrets.get_ep_meta_and_key(resolved_ep)
            fetched = True
        return resolved_ep, slot_acct, meta, api_key, fetched

    def open_resolved_service_url(
        self,
        tunnel_url: str,
        ep: str = "",
        sname: str = "comfy",
        workflow_name: str | None = None,
        stage: str = "normal",
        detected_kind: str | None = None,
        already_fetched_meta: bool = False,
    ) -> dict[str, Any]:
        """根据解析出的存活隧道地址分流执行打开动作（LLM 直开对话页并回传密钥，ComfyUI 打开画布）。"""
        resolved_ep, slot_acct, meta, api_key, fetched = self._resolve_service_meta_for_open(
            tunnel_url, ep, sname, detected_kind, already_fetched_meta
        )
        raw_kind = str(meta.get("kind") or detected_kind or "comfy").lower()
        kind = str(detected_kind or "comfy").lower() if raw_kind == "dual" else raw_kind
        if kind == "llm":
            if not api_key and not fetched:
                _, r_ep, _ = self.fast_read_slot_tunnel_url(resolved_ep, sname, account_id=slot_acct)
                if r_ep:
                    resolved_ep = r_ep
                _, api_key = self._secrets.get_ep_meta_and_key(resolved_ep)
            clean_url = tunnel_url.strip().rstrip("/")
            self._ctx.open_browser(clean_url)
            msg = (
                "💬 已打开 Qwen3.8 对话页 · 访问密钥已复制，首次直接 Ctrl+V 登录"
                if api_key
                else "💬 已打开 Qwen3.8 对话页"
            )
            return {
                "ok": True,
                "url": clean_url,
                "service_kind": "llm",
                "clipboard_text": api_key,
                "key_copied": bool(api_key),
                "message": msg,
            }

        open_url = self._b._open_browser_for_comfy(
            tunnel_url,
            ep=resolved_ep,
            sname=sname,
            workflow_name=workflow_name,
            already_fetched_meta=fetched,
        )
        return {
            "ok": True,
            "url": open_url,
            "service_kind": "comfy",
            "message": _format_comfy_open_message(workflow_name, stage),
        }

    # ------------------------------------------------------------------
    # 三级极速打开 ComfyUI / LLM 隧道与访问凭据打包
    # ------------------------------------------------------------------

    def _resolve_cluster_slot_target(
        self, slot_name: str, session: str
    ) -> tuple[dict[str, Any] | None, str]:
        """在多号异构集群模式下解析指定槽位的伪会话对象与所属账号 ID。"""
        cluster_svc = self._b._cluster_svc
        if not cluster_svc.has_active_slots():
            return None, ""
        slot_obj = cluster_svc.get_slot(slot_name)
        if not slot_obj:
            return None, ""
        return (
            {
                "name": str(slot_obj.get("session") or session or "comfy"),
                "endpoint": str(slot_obj.get("endpoint") or ""),
                "accelerator": str(slot_obj.get("gpu") or ""),
                "orphan": False,
            },
            str(slot_obj.get("account_id") or "").strip(),
        )

    def _resolve_service_target(
        self,
        slot_name: str,
        session: str,
        empty_msg: str,
        *,
        missing_cluster_slot_msg: str = "",
        always_probe_when_none: bool = False,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None, str, dict[str, Any] | None]:
        """统一解析单号会话或异构集群目标槽位，返回 `(target, cluster_target, slot_acct, err_resp)`。"""
        cluster_target, slot_acct = self._resolve_cluster_slot_target(slot_name, session)
        if missing_cluster_slot_msg and self._b._cluster_svc.has_active_slots() and cluster_target is None:
            return None, None, "", {"ok": False, "message": missing_cluster_slot_msg}
        if cluster_target is not None:
            return cluster_target, cluster_target, slot_acct, None
        target, err_resp = self._b._ensure_target_session(
            session, empty_msg, always_probe_when_none=always_probe_when_none
        )
        if err_resp is not None:
            return None, None, "", err_resp
        ep_t = str(target.get("endpoint") or "").strip() if isinstance(target, dict) else ""
        single_acct = self._store.resolve_single_node_account_id(ep_t)
        return target, None, single_acct, None

    def _run_ctl_tunnel_restart(self, sname: str, slot_acct: str) -> tuple[int, str]:
        """调用 `colab_ctl tunnel --restart` 重建远端隧道并返回 `(退出码, 脱敏输出)`。"""
        tun_cmd = [*self._ctx.ctl_cmd_prefix(), "tunnel", "--restart"]
        if slot_acct:
            tun_cmd.extend(["--account", slot_acct])
        tun_cmd.extend(["-s", sname])
        proc = self._ctx.run_hidden(
            tun_cmd,
            timeout=60,
            extra_env=self._ctx.build_ctl_env(),
        )
        output = redact_secrets(f"{proc.stdout}\n{proc.stderr}")
        return int(proc.returncode), output

    def _try_open_comfy_from_cache(
        self,
        ep: str,
        sname: str,
        workflow_name: str | None,
        cluster_target: dict[str, Any] | None,
        cf_cfg_p: Path,
    ) -> tuple[dict[str, Any] | None, str]:
        """第 1 级（缓存秒开）：校验本地隧道缓存，命中则返回打开结果，同时返回推断服务偏好。"""
        known_meta, _ = self._secrets.get_ep_meta_and_key(ep)
        sk_guess = "comfy" if cluster_target is not None else self._b._get_ep_service_kind(ep)
        preferred_kind = "llm" if sk_guess == "llm" else "comfy"

        cached_url = self._store.get_tunnel_cache().get(ep) if ep else None
        if not cached_url and cluster_target is not None:
            c_slot = self._b._cluster_svc.get_slot("comfy") or {}
            cached_url = str(c_slot.get("tunnel_url") or "").strip() or None

        if known_meta:
            mkind = str(known_meta.get("kind") or "comfy")
            hit_url = self._probe_known_meta_tunnel(ep, cached_url, known_meta, cf_cfg_p)
            if hit_url:
                det_kind = "comfy" if mkind == "dual" else mkind
                return (
                    self._b._open_resolved_service_url(
                        hit_url,
                        ep=ep,
                        sname=sname,
                        workflow_name=workflow_name,
                        stage="cache",
                        detected_kind=det_kind,
                    ),
                    preferred_kind,
                )
            return None, preferred_kind

        if cached_url:
            cached_tok = self._b._get_comfy_key_for_url(cached_url, ep)
            det_kind = self._ctx.detect_tunnel_kind(
                cached_url,
                timeout=3.0,
                preferred=preferred_kind,
                auth_token=cached_tok,
                cf_config_path=cf_cfg_p,
            )
            if det_kind is not None:
                return (
                    self._b._open_resolved_service_url(
                        cached_url,
                        ep=ep,
                        sname=sname,
                        workflow_name=workflow_name,
                        stage="cache",
                        detected_kind=det_kind,
                    ),
                    preferred_kind,
                )
        return None, preferred_kind

    def _try_open_comfy_from_fast_read(
        self,
        ep: str,
        sname: str,
        slot_acct: str,
        workflow_name: str | None,
        cf_cfg_p: Path,
    ) -> tuple[dict[str, Any] | None, str, str]:
        """第 2 级（HTTP Contents 速读）：补水远端元数据并探测直开，返回 `(结果, resolved_ep, fast_kind)`。"""
        tstatus, resolved_ep, fast_url = self.fast_read_slot_tunnel_url(
            ep, sname, account_id=slot_acct
        )
        if tstatus == "no_session":
            return {"ok": False, "message": "当前无运行中的 Colab 实例，请先部署开机"}, resolved_ep, ""

        eff_ep = resolved_ep or ep
        meta_fast, _ = self._secrets.get_ep_meta_and_key(eff_ep)
        fast_kind = str(meta_fast.get("kind") or "auto")
        cand_fast = (
            str(meta_fast.get("comfy_tunnel_url") or fast_url or "").strip()
            if fast_kind == "dual"
            else (fast_url or "")
        )
        probe_fast_k = "comfy" if fast_kind == "dual" else fast_kind
        ftok = self._b._get_comfy_key_for_url(cand_fast, eff_ep)
        if tstatus == "ok" and cand_fast and self._ctx.probe_tunnel_alive(
            cand_fast, timeout=4.0, kind=probe_fast_k, auth_token=ftok, cf_config_path=cf_cfg_p
        ):
            self._persist_tunnel_cache(resolved_ep, cand_fast)
            if fast_kind == "dual":
                resolved_kind: str | None = "comfy"
            elif fast_kind in ("comfy", "llm"):
                resolved_kind = fast_kind
            else:
                resolved_kind = self._ctx.detect_tunnel_kind(
                    cand_fast, timeout=2.5, auth_token=ftok, cf_config_path=cf_cfg_p
                )
            return (
                self._b._open_resolved_service_url(
                    cand_fast,
                    ep=eff_ep,
                    sname=sname,
                    workflow_name=workflow_name,
                    stage="normal",
                    detected_kind=resolved_kind,
                    already_fetched_meta=True,
                ),
                resolved_ep,
                fast_kind,
            )
        return None, resolved_ep, fast_kind

    def _restart_comfy_tunnel_and_open(
        self,
        ep: str,
        resolved_ep: str,
        sname: str,
        slot_acct: str,
        fast_kind: str,
        preferred_kind: str,
        workflow_name: str | None,
        cluster_target: dict[str, Any] | None,
        cf_cfg_p: Path,
    ) -> dict[str, Any]:
        """第 3 级（断线重启兜底）：重建公网隧道、刷新缓存元数据并打开对应服务。"""
        if not self._ctx.ctl_available():
            return {"ok": False, "message": f"未找到控制脚本: {self._ctx.ctl_script.name}"}

        rc, output = self._run_ctl_tunnel_restart(sname, slot_acct)
        tunnel_pat = self._b._tunnel_url_regex_pattern()
        match = re.search(
            rf"(?:(?:COMFYUI|SERVICE)_TUNNEL_URL:\s*|\[(?:ComfyUI|LLM|Qwen[^]]*) WebUI 地址\]\s*)({tunnel_pat})",
            output,
        )
        if rc == 0 and match:
            url = match.group(1).strip()
            use_ep = resolved_ep or ep
            self.fast_read_slot_tunnel_url(use_ep, sname, account_id=slot_acct)
            self._record_restarted_comfy_url(use_ep, url, update_cluster_slot=(cluster_target is not None))
            rtok = self._b._get_comfy_key_for_url(url, use_ep)
            det_after = (
                "comfy"
                if fast_kind == "dual"
                else self._ctx.detect_tunnel_kind(
                    url, timeout=3.0, preferred=preferred_kind, auth_token=rtok, cf_config_path=cf_cfg_p
                )
            )
            return self._b._open_resolved_service_url(
                url,
                ep=use_ep,
                sname=sname,
                workflow_name=workflow_name,
                stage="restarted",
                detected_kind=det_after,
                already_fetched_meta=True,
            )

        if ep:
            with self._store.lock:
                if self._store.pop_tunnel_cache(ep) is not None:
                    self._store.save()
        tail = (output.strip().splitlines() or ["未获取到隧道地址"])[-1]
        return {"ok": False, "message": f"隧道开启失败: {redact_secrets(tail)}"}

    def _record_restarted_comfy_url(
        self, use_ep: str, url: str, *, update_cluster_slot: bool
    ) -> None:
        """持主锁更新重启后的 ComfyUI 隧道缓存、内存元数据及集群分槽地址。"""
        if not use_ep:
            return
        with self._store.lock:
            self._store.set_tunnel_cache(use_ep, url)
            if use_ep in self._secrets.service_meta_by_ep:
                meta = self._secrets.service_meta_by_ep[use_ep]
                meta["tunnel_url"] = url
                if meta.get("kind") == "dual":
                    meta["comfy_tunnel_url"] = url
            slots_d = self._store.state.get("cluster_slots")
            if (
                update_cluster_slot
                and isinstance(slots_d, dict)
                and isinstance(slots_d.get("comfy"), dict)
            ):
                slots_d["comfy"]["tunnel_url"] = url
            self._store.save()

    def open_comfy_tunnel(
        self, session: str = "comfy", workflow_name: str | None = None
    ) -> dict[str, Any]:
        """三级极速打开云端 ComfyUI / LLM 隧道：本地缓存秒开 -> HTTP Contents 直读 -> 自动重启兜底。"""
        b = self._b
        if (
            b._cluster_svc.has_active_slots()
            and b._cluster_svc.get_slot("comfy") is None
            and b._cluster_svc.get_slot("llm") is not None
        ):
            return self.open_llm_tunnel(session=session)
        b.touch_activity("comfy")
        if b.is_deploying():
            return {"ok": False, "message": "正在执行 7 步开机部署，请等待就绪后再打开"}
        if not b._action_lock.acquire(blocking=False):
            return {"ok": False, "message": "另一项操作正在执行，请稍等片刻"}
        try:
            target, cluster_target, slot_acct, err_resp = self._resolve_service_target(
                "comfy",
                session,
                "当前无运行中的 Colab 实例，请先部署开机",
                always_probe_when_none=True,
            )
            if err_resp is not None:
                return err_resp
            ep, sname, is_orphan = _extract_target_ep_and_name(target, session)
            cf_cfg_p = self._ctx.resolve_cf_config_path()

            cache_resp, preferred_kind = self._try_open_comfy_from_cache(
                ep, sname, workflow_name, cluster_target, cf_cfg_p
            )
            if cache_resp is not None:
                return cache_resp

            fast_resp, resolved_ep, fast_kind = self._try_open_comfy_from_fast_read(
                ep, sname, slot_acct, workflow_name, cf_cfg_p
            )
            if fast_resp is not None:
                return fast_resp

            if is_orphan:
                return {
                    "ok": False,
                    "message": "该实例为网页孤儿实例且未检测到可用隧道，请用面板一键开机或在网页端重启隧道",
                }
            return self._restart_comfy_tunnel_and_open(
                ep,
                resolved_ep,
                sname,
                slot_acct,
                fast_kind,
                preferred_kind,
                workflow_name,
                cluster_target,
                cf_cfg_p,
            )
        except Exception as exc:
            return {"ok": False, "message": f"隧道操作异常: {redact_secrets(str(exc))}"}
        finally:
            b._action_lock.release()

    def _try_open_llm_from_fast_read(
        self, ep: str, sname: str, slot_acct: str, cf_cfg_p: Path
    ) -> tuple[dict[str, Any] | None, str, dict[str, Any], str]:
        """速读远端 LLM 隧道并尝试直接打开，返回 `(结果, use_ep, meta_fast, fast_kind)`。"""
        tstatus, resolved_ep, fast_url = self.fast_read_slot_tunnel_url(
            ep, sname, account_id=slot_acct
        )
        if tstatus == "no_session":
            return {"ok": False, "message": "当前无运行中的 Colab 实例，请先部署开机"}, ep, {}, ""

        use_ep = resolved_ep or ep
        meta_fast, _ = self._secrets.get_ep_meta_and_key(use_ep)
        fast_kind = str(meta_fast.get("kind") or "").lower()
        if fast_kind == "comfy" and not self._b.is_dual_mode_active():
            return {"ok": False, "message": "当前运行的是纯 ComfyUI 实例，未挂载 LLM 对话服务"}, use_ep, meta_fast, fast_kind

        cand_llm = str(
            meta_fast.get("llm_tunnel_url") or (fast_url if fast_kind == "llm" else "") or ""
        ).strip()
        if cand_llm and self._ctx.probe_tunnel_alive(
            cand_llm, timeout=4.0, kind="llm", cf_config_path=cf_cfg_p
        ):
            return (
                self._b._open_resolved_service_url(
                    cand_llm,
                    ep=use_ep,
                    sname=sname,
                    stage="normal",
                    detected_kind="llm",
                    already_fetched_meta=True,
                ),
                use_ep,
                meta_fast,
                fast_kind,
            )
        return None, use_ep, meta_fast, fast_kind

    def _restart_llm_tunnel_and_open(
        self,
        use_ep: str,
        sname: str,
        slot_acct: str,
        fast_kind: str,
        meta_fast: dict[str, Any],
        cluster_target: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """重启并提取 LLM 公网隧道地址，持久化槽位映射并打开对话网页。"""
        if not self._ctx.ctl_available():
            return {"ok": False, "message": f"未找到控制脚本: {self._ctx.ctl_script.name}"}

        rc, output = self._run_ctl_tunnel_restart(sname, slot_acct)
        tunnel_pat = self._b._tunnel_url_regex_pattern()
        m_llm = re.search(rf"LLM_TUNNEL_URL:\s*({tunnel_pat})", output)
        if not m_llm and (fast_kind == "llm" or cluster_target is not None):
            m_llm = re.search(
                rf"(?:SERVICE_TUNNEL_URL:\s*|\[LLM WebUI / API 地址\]\s*)({tunnel_pat})",
                output,
            )
        if rc == 0 and m_llm:
            url = m_llm.group(1).strip()
            if use_ep:
                with self._store.lock:
                    cur_m = self._secrets.service_meta_by_ep.setdefault(
                        use_ep, {"kind": fast_kind or "dual", "port": 8188}
                    )
                    cur_m["llm_tunnel_url"] = url
                    slots_d = self._store.state.get("cluster_slots")
                    if (
                        cluster_target is not None
                        and isinstance(slots_d, dict)
                        and isinstance(slots_d.get("llm"), dict)
                    ):
                        slots_d["llm"]["tunnel_url"] = url
                    self._store.save()
            return self._b._open_resolved_service_url(
                url,
                ep=use_ep,
                sname=sname,
                stage="restarted",
                detected_kind="llm",
                already_fetched_meta=bool(meta_fast),
            )

        tail = (output.strip().splitlines() or ["未获取到 LLM 对话隧道"])[-1]
        return {"ok": False, "message": f"暂未获取到在线 Qwen3.8 对话隧道: {redact_secrets(tail)}"}

    def open_llm_tunnel(self, session: str = "comfy") -> dict[str, Any]:
        """在双栖或纯 LLM 实例下专一打开 Qwen3.8 对话网页并复制密钥。"""
        b = self._b
        b.touch_activity("llm")
        if b.is_deploying():
            return {"ok": False, "message": "正在执行 7 步开机部署，请等待就绪后再打开"}
        if not b._action_lock.acquire(blocking=False):
            return {"ok": False, "message": "另一项操作正在执行，请稍等片刻"}
        try:
            target, cluster_target, slot_acct, err_resp = self._resolve_service_target(
                "llm",
                session,
                "当前无运行中的 Colab 实例，请先部署开机",
                missing_cluster_slot_msg="当前仅运行 ComfyUI 画布槽位，未挂载 LLM 对话服务",
                always_probe_when_none=True,
            )
            if err_resp is not None:
                return err_resp
            ep, sname, is_orphan = _extract_target_ep_and_name(target, session)

            llm_url = b.resolve_active_llm_tunnel_url(session=sname, allow_fast_discover=True)
            if llm_url:
                with self._store.lock:
                    has_meta = bool(self._secrets.service_meta_by_ep.get(ep))
                return b._open_resolved_service_url(
                    llm_url,
                    ep=ep,
                    sname=sname,
                    stage="cache",
                    detected_kind="llm",
                    already_fetched_meta=has_meta,
                )

            cf_cfg_p = self._ctx.resolve_cf_config_path()
            fast_resp, use_ep, meta_fast, fast_kind = self._try_open_llm_from_fast_read(
                ep, sname, slot_acct, cf_cfg_p
            )
            if fast_resp is not None:
                return fast_resp
            if is_orphan:
                return {"ok": False, "message": "该孤儿实例未检测到可用 LLM 对话隧道"}
            return self._restart_llm_tunnel_and_open(
                use_ep, sname, slot_acct, fast_kind, meta_fast, cluster_target
            )
        except Exception as exc:
            return {"ok": False, "message": f"打开 LLM 对话页异常: {redact_secrets(str(exc))}"}
        finally:
            b._action_lock.release()

    def get_comfy_access_bundle(self, session: str = "comfy") -> dict[str, Any]:
        """组装当前 ComfyUI 公网入口链接与访问口令（仅供剪贴板写入，不回传明文口令）。"""
        b = self._b
        b.touch_activity("comfy")
        if b.is_deploying():
            return {"ok": False, "message": "正在执行 7 步开机部署，请等待就绪后再操作"}
        target, _, slot_acct, err_resp = self._resolve_service_target(
            "comfy",
            session,
            "当前无运行中的 ComfyUI 实例，请先部署开机",
            missing_cluster_slot_msg="当前无运行中的 ComfyUI 槽位",
        )
        if err_resp is not None:
            return err_resp

        ep, sname, _ = _extract_target_ep_and_name(target, session)
        turl = b.resolve_active_tunnel_url(session=sname, allow_fast_discover=True)
        if not turl:
            tstatus, resolved_ep, fast_url = self.fast_read_slot_tunnel_url(
                ep, sname, account_id=slot_acct
            )
            if tstatus == "no_session":
                return {"ok": False, "message": "当前无运行中的 ComfyUI 实例"}
            ep = resolved_ep or ep
            turl = fast_url or turl
        if not turl:
            return {"ok": False, "message": "暂未获取到在线 ComfyUI 公网链接，请先左键打开一次"}

        clean_url = turl.strip().rstrip("/")
        comfy_key = b._get_comfy_key_for_url(clean_url, ep)
        if not comfy_key and self._store.has_running_session():
            _, resolved_ep, _ = self.fast_read_slot_tunnel_url(ep, sname, account_id=slot_acct)
            comfy_key = b._get_comfy_key_for_url(clean_url, resolved_ep or ep)

        clip_text = f"ComfyUI URL: {clean_url}\n访问口令: {comfy_key}" if comfy_key else clean_url
        return {
            "ok": True,
            "url": clean_url,
            "has_key": bool(comfy_key),
            "clipboard_text": clip_text,
            "message": "📋 已复制 ComfyUI 公网链接与访问口令" if comfy_key else "📋 已复制 ComfyUI 公网链接",
        }

    def _probe_llm_candidate_from_fast_read(
        self,
        ep: str,
        sname: str,
        slot_acct: str,
        cf_cfg_p: Path,
        *,
        persist_cache: bool,
    ) -> tuple[str, str, str | None]:
        """执行一次 fast-tunnel 并探测其中携带的 LLM 隧道地址是否在线可用。"""
        tstatus, resolved_ep, fast_url = self.fast_read_slot_tunnel_url(
            ep, sname, account_id=slot_acct
        )
        eff_ep = resolved_ep or ep
        if tstatus == "no_session":
            return tstatus, eff_ep, None
        meta_chk, _ = self._secrets.get_ep_meta_and_key(eff_ep)
        kind_chk = str(meta_chk.get("kind") or "").lower()
        llm_cand = str(
            meta_chk.get("llm_tunnel_url") or (fast_url if kind_chk == "llm" else "") or ""
        ).strip()
        if llm_cand and self._ctx.probe_tunnel_alive(
            llm_cand, timeout=3.5, kind="llm", cf_config_path=cf_cfg_p
        ):
            if persist_cache and eff_ep and kind_chk == "llm":
                self._persist_tunnel_cache(eff_ep, llm_cand)
            return tstatus, eff_ep, llm_cand
        return tstatus, eff_ep, None

    def _is_pure_comfy_non_llm(self, meta: dict[str, Any]) -> bool:
        """判断当前实例是否为未挂载 LLM 服务的纯 ComfyUI 运行时。"""
        if not meta or str(meta.get("kind") or "").lower() != "comfy":
            return False
        with self._b._comfy_lock:
            cur_sk = str(self._b._comfy_state.get("service_kind") or "comfy")
        return cur_sk not in ("llm", "dual") and not self._b.is_dual_mode_active()

    def _fallback_llm_url_from_comfy_state(self) -> str:
        """从最近一次遥测运行态 `_comfy_state` 中兜底提取 LLM 隧道 URL。"""
        with self._b._comfy_lock:
            cur_sk = str(self._b._comfy_state.get("service_kind") or "")
            if cur_sk not in ("llm", "dual"):
                return ""
            return str(
                self._b._comfy_state.get("llm_tunnel_url")
                or (self._b._comfy_state.get("tunnel_url") if cur_sk == "llm" else "")
                or ""
            )

    def get_llm_api_bundle(self, session: str = "comfy") -> dict[str, Any]:
        """组装当前在线 Qwen3.8 实例（单模或双栖）的 OpenAI 兼容接口配置包（不持久化密钥）。"""
        b = self._b
        b.touch_activity("llm")
        if b.is_deploying():
            return {"ok": False, "message": "正在执行 7 步开机部署，请等待就绪后再复制配置"}

        target, _, slot_acct, err_resp = self._resolve_service_target(
            "llm",
            session,
            "当前无运行中的 Qwen3.8 实例，请先部署开机",
            missing_cluster_slot_msg="当前仅运行 ComfyUI 画布槽位，未挂载 LLM 对话服务",
        )
        if err_resp is not None:
            return err_resp

        ep, sname, _ = _extract_target_ep_and_name(target, session)
        turl = b.resolve_active_llm_tunnel_url(session=sname, allow_fast_discover=True)
        fetched_now = False
        cf_cfg_p = self._ctx.resolve_cf_config_path()

        if not turl:
            tstatus, ep, probed_url = self._probe_llm_candidate_from_fast_read(
                ep, sname, slot_acct, cf_cfg_p, persist_cache=True
            )
            fetched_now = True
            if tstatus == "no_session":
                return {"ok": False, "message": "当前无运行中的 Qwen3.8 实例，请先部署开机"}
            turl = probed_url

        meta, key = self._secrets.get_ep_meta_and_key(ep)
        if (not meta or not key) and not fetched_now:
            _, ep, probed_url = self._probe_llm_candidate_from_fast_read(
                ep, sname, slot_acct, cf_cfg_p, persist_cache=False
            )
            meta, key = self._secrets.get_ep_meta_and_key(ep)
            turl = turl or probed_url

        if self._is_pure_comfy_non_llm(meta):
            return {"ok": False, "message": "当前运行的是 ComfyUI 实例，非 LLM 对话服务"}

        turl = turl or self._fallback_llm_url_from_comfy_state()
        if not turl:
            return {"ok": False, "message": "暂未获取到在线 Qwen3.8 隧道，请稍候或先点打开对话页"}

        clean_turl = turl.strip().rstrip("/")
        base_url = f"{clean_turl}/v1"
        model = b._infer_default_llm_model_name(meta)
        clipboard_text = f"Base URL: {base_url}\nAPI Key: {key}\nModel: {model}"
        return {
            "ok": True,
            "base_url": base_url,
            "model": model,
            "has_key": bool(key),
            "clipboard_text": clipboard_text,
            "message": "📋 已复制 OpenAI 接口配置 (Base URL + Key)",
        }

    # ------------------------------------------------------------------
    # Colab 原生网页唤醒与 CF 固定隧道配置管理
    # ------------------------------------------------------------------

    def open_colab_url(self, session: str = "comfy") -> dict[str, Any]:
        """在浏览器中打开 Colab 运行时关联网页或 Colab 首页。"""
        b = self._b
        b.touch_activity()
        target = None
        if b._cluster_svc.has_active_slots():
            slots = b._cluster_svc.get_cluster_slots()
            picked = (
                slots.get("llm") if session == "llm" else (slots.get("comfy") or slots.get("llm"))
            )
            if picked and picked.get("endpoint"):
                target = {
                    "endpoint": picked["endpoint"],
                    "name": picked.get("label") or picked["endpoint"],
                }
        if target is None:
            target = self._store.pick_target_session(session)
        if target and target.get("endpoint"):
            ep = target["endpoint"]
            url = (
                "https://colab.research.google.com/notebooks/empty.ipynb"
                f"?dbu=%2Ftun%2Fm%2F{ep}"
                f"#datalabBackendUrl=https://colab.research.google.com/tun/m/{ep}"
            )
            self._ctx.open_browser(url)
            label = target.get("name") or ep
            return {"ok": True, "url": url, "message": f"已打开 Colab 会话 ({label})"}

        colab_exe = self._ctx.get_colab_exe()
        if colab_exe and colab_exe.is_file():
            with suppress_logged("tunnel.open_colab_url", throttled_key=f"colab_url:{session}"):
                proc = self._ctx.run_hidden([str(colab_exe), "url", "-s", session], timeout=12)
                m = re.search(r"https://colab\.research\.google\.com\S+", proc.stdout)
                if proc.returncode == 0 and m:
                    self._ctx.open_browser(m.group(0))
                    return {"ok": True, "url": m.group(0), "message": "已打开 Colab 会话页"}

        home_url = "https://colab.research.google.com/"
        self._ctx.open_browser(home_url)
        return {"ok": True, "url": home_url, "message": "当前无活跃会话，已打开 Colab 主页"}

    def save_cf_tunnel_config(self, config: dict[str, Any] | None = None) -> dict[str, Any]:
        """将 Cloudflare 隧道 Token 与固定密钥存入系统凭据库，配置文件仅存域名与引用。"""
        cfg_in = config if isinstance(config, dict) else {}
        tok_raw = str(cfg_in.get("tunnel_token") or "").strip()
        key_raw = str(cfg_in.get("llm_api_key") or "").strip()
        if tok_raw:
            register_ephemeral_secret(tok_raw)
        if key_raw:
            register_ephemeral_secret(key_raw)

        try:
            save_cf_tunnel_settings(
                self._ctx.resolve_cf_config_path(),
                comfy_url=str(cfg_in.get("comfy_url") or ""),
                llm_url=str(cfg_in.get("llm_url") or ""),
                tunnel_token=tok_raw,
                llm_api_key=key_raw,
                colab_py=self._ctx.get_colab_py(),
            )
        except Exception as exc:
            return {"ok": False, "message": redact_secrets(str(exc))}

        snap = self._b._rebuild_and_emit_snapshot()
        return {"ok": True, "snapshot": snap, "message": "✅ 已启用 Cloudflare 固定域名隧道"}

    def clear_cf_tunnel_config(self) -> dict[str, Any]:
        """清除固定命名隧道凭据与配置，切回免费即用的 trycloudflare 临时隧道。"""
        try:
            clear_cf_tunnel_settings(
                self._ctx.resolve_cf_config_path(),
                colab_py=self._ctx.get_colab_py(),
            )
        except OSError as exc:
            return {"ok": False, "message": f"切回失败: {redact_secrets(str(exc))}"}
        snap = self._b._rebuild_and_emit_snapshot()
        return {"ok": True, "snapshot": snap, "message": "已切回免费 trycloudflare 临时隧道"}
