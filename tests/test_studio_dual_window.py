"""Colab Studio「同进程 · 单 Backend · 双窗口」架构、媒体安全路由与 PyWebView 双窗真渲染验收套件。

说明：全量基于隔离临时沙箱目录与环境变量（COLAB_PET_MOCK_VAULT_PATH、
COLAB_PET_CF_CONFIG_PATH、COLAB_PET_DIAG_LOG_PATH），绝不读取或修改宿主机真实凭据库、
`.cf_tunnel_secret.json` 或 `~/.config/colab-cli/sessions.json`。
"""

from __future__ import annotations

import argparse
import base64
import http.client
import inspect
import json
import os
import re
import struct
import subprocess
import sys
import threading
import time
import wave
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
from PIL import Image, ImageDraw
from PIL.PngImagePlugin import PngInfo

PKG_DIR = Path(__file__).resolve().parents[1]
PET_DIR = PKG_DIR / "colab_pet"
for _p in (str(PKG_DIR), str(PET_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import build_windows_package as win_pkg  # noqa: E402
import colab_backend as cb  # noqa: E402
import colab_security as csec  # noqa: E402
import pet_app  # noqa: E402
from colab_core.runtime import CommandResult  # noqa: E402

_EMOJI_RE = re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]")


@pytest.fixture()
def isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """为每条测试用例构建完全隔离的凭据保险箱、隧道配置与诊断日志沙箱。"""
    csec.clear_ephemeral_secrets()
    vault_file = tmp_path / "mock_vault.json"
    cf_file = tmp_path / ".cf_tunnel_secret.json"
    diag_log = tmp_path / "logs" / "colab_diag.log"
    fake_rclone_exe = tmp_path / "sandbox_rclone.exe"
    fake_rclone_exe.write_bytes(b"MZ")
    unconf_rclone = tmp_path / "sandbox_rclone.conf"
    unconf_rclone.write_text("[other_remote]\ntype = local\n", encoding="utf-8")
    fake_colab_exe = tmp_path / "sandbox_colab.exe"
    fake_colab_exe.write_bytes(b"MZ")
    fake_colab_py = tmp_path / "sandbox_colab_py.exe"
    fake_colab_py.write_bytes(b"MZ")
    monkeypatch.setenv("COLAB_PET_MOCK_VAULT_PATH", str(vault_file))
    monkeypatch.setenv("COLAB_PET_CF_CONFIG_PATH", str(cf_file))
    monkeypatch.setenv("COLAB_PET_DIAG_LOG_PATH", str(diag_log))
    monkeypatch.setenv("COLAB_PET_RCLONE_EXE", str(fake_rclone_exe))
    monkeypatch.setenv("COLAB_PET_RCLONE_CONF", str(unconf_rclone))
    monkeypatch.setenv("COLAB_PET_COLAB_EXE", str(fake_colab_exe))
    monkeypatch.setenv("COLAB_PET_COLAB_PY", str(fake_colab_py))
    monkeypatch.delenv("COLAB_PET_ACCOUNT_ID", raising=False)
    yield SimpleNamespace(
        tmp=tmp_path,
        vault_file=vault_file,
        cf_file=cf_file,
        diag_log=diag_log,
    )
    pet_app._set_active_media_root(None)
    csec.clear_ephemeral_secrets()


def _make_backend(env_ns: SimpleNamespace, name: str = "studio_state") -> cb.ColabPetBackend:
    """实例化绑定到隔离沙箱路径的 ColabPetBackend。"""
    return cb.ColabPetBackend(
        state_path=env_ns.tmp / f"{name}.json",
        download_root=env_ns.tmp / f"{name}_dl",
        cf_config_path=env_ns.cf_file,
    )


def _create_comfy_artwork_png(
    out_path: Path,
    *,
    width: int = 768,
    height: int = 512,
    palette: tuple[tuple[int, int, int], tuple[int, int, int]] = ((124, 58, 237), (236, 229, 255)),
    positive: str = "1girl, anime cel shading, lavender hair, sunlit atelier, masterpiece",
    negative: str = "lowres, blurry, extra fingers, watermark",
    seed: int = 4208819273,
    steps: int = 28,
    cfg: float = 4.5,
    sampler: str = "euler",
    scheduler: str = "normal",
    model: str = "anima_v2_fp16.safetensors",
    loras: tuple[tuple[str, float], ...] = (
        ("chara_mizuki_v3.safetensors", 0.85),
        ("pastel_anime_line.safetensors", 0.65),
    ),
    mtime_ts: float | None = None,
) -> Path:
    """生成带完整 ComfyUI prompt/workflow 元数据与渐变几何画面纹理的测试 PNG。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (width, height), palette[1])
    draw = ImageDraw.Draw(img)
    c1, c2 = palette
    for y in range(height):
        t = y / max(1, height - 1)
        r = int(c1[0] * (1.0 - t) + c2[0] * t)
        g = int(c1[1] * (1.0 - t) + c2[1] * t)
        b = int(c1[2] * (1.0 - t) + c2[2] * t)
        draw.line([(0, y), (width, y)], fill=(r, g, b))
    draw.ellipse(
        [int(width * 0.14), int(height * 0.15), int(width * 0.58), int(height * 0.72)],
        fill=(255, 255, 255),
        outline=c1,
        width=4,
    )
    draw.rounded_rectangle(
        [int(width * 0.44), int(height * 0.34), int(width * 0.86), int(height * 0.84)],
        radius=28,
        fill=(min(255, c1[0] + 45), min(255, c1[1] + 45), min(255, c1[2] + 15)),
        outline=(255, 255, 255),
        width=3,
    )

    prompt_graph: dict[str, Any] = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": model}},
        "5": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": width, "height": height, "batch_size": 1},
        },
    }
    model_ref: list[Any] = ["1", 0]
    clip_ref: list[Any] = ["1", 1]
    next_id = 10
    for lora_name, strength in loras:
        nid = str(next_id)
        next_id += 1
        prompt_graph[nid] = {
            "class_type": "LoraLoader",
            "inputs": {
                "model": model_ref,
                "clip": clip_ref,
                "lora_name": lora_name,
                "strength_model": strength,
                "strength_clip": strength,
            },
        }
        model_ref = [nid, 0]
        clip_ref = [nid, 1]

    prompt_graph["20"] = {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": positive, "clip": clip_ref},
        "_meta": {"title": "Positive Prompt"},
    }
    prompt_graph["21"] = {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": negative, "clip": clip_ref},
        "_meta": {"title": "Negative Prompt"},
    }
    prompt_graph["30"] = {
        "class_type": "KSampler",
        "inputs": {
            "model": model_ref,
            "positive": ["20", 0],
            "negative": ["21", 0],
            "latent_image": ["5", 0],
            "seed": seed,
            "steps": steps,
            "cfg": cfg,
            "sampler_name": sampler,
            "scheduler": scheduler,
            "denoise": 1.0,
        },
    }
    prompt_graph["31"] = {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["30", 0], "vae": ["1", 2]},
    }
    prompt_graph["32"] = {
        "class_type": "SaveImage",
        "inputs": {"images": ["31", 0], "filename_prefix": "StudioTest"},
    }

    workflow_obj = {
        "last_node_id": 32,
        "last_link_id": 12,
        "nodes": [
            {"id": 1, "type": "CheckpointLoaderSimple"},
            {"id": 30, "type": "KSampler"},
        ],
        "links": [[1, 1, 0, 30, 0, "MODEL"]],
    }
    png_info = PngInfo()
    png_info.add_text("prompt", json.dumps(prompt_graph, ensure_ascii=False))
    png_info.add_text("workflow", json.dumps(workflow_obj, ensure_ascii=False))
    img.save(out_path, format="PNG", pnginfo=png_info)
    if mtime_ts is not None:
        os.utime(out_path, (mtime_ts, mtime_ts))
    return out_path


def _create_mock_wav(out_path: Path, duration_sec: float = 0.25, mtime_ts: float | None = None) -> Path:
    """写入标准 PCM WAV 音频文件供画廊扫描、HTTP Range 206 与浏览器播放测试使用。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sample_rate = 16000
    frames = int(sample_rate * duration_sec)
    with wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        buf = bytearray()
        for i in range(frames):
            val = int(4000 if (i // 40) % 2 == 0 else -4000)
            buf.extend(struct.pack("<h", val))
        wf.writeframes(bytes(buf))
    if mtime_ts is not None:
        os.utime(out_path, (mtime_ts, mtime_ts))
    return out_path


# =========================================================================
# 1. StudioService 功能、媒体安全路由 (/studio.*, /__media/*, 206, 403) 与脱敏校验
# =========================================================================


def test_studio_service_gallery_metadata_security_and_logs(
    isolated_env: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """校验 StudioService 画廊沙箱、PNG 元数据提取、HTTP 200/206/403 路由与日志脱敏。"""
    backend = _make_backend(isolated_env, "svc_state")
    dl_root = backend.download_root.resolve()
    dl_root.mkdir(parents=True, exist_ok=True)

    # 在 download_root 同级父目录放置越界诱饵文件，验证扫描与路由均绝不泄露
    sibling_png = dl_root.parent / "sibling_secret.png"
    _create_comfy_artwork_png(sibling_png, positive="SHOULD_NEVER_LEAK")
    outside_secret = dl_root.parent / "secret.txt"
    outside_secret.write_text("TOP_SECRET_VALUE", encoding="utf-8")

    now_ts = time.time()
    wav_path = _create_mock_wav(dl_root / "sub" / "mock.wav", mtime_ts=now_ts - 30)
    mp4_path = dl_root / "mock.mp4"
    mp4_bytes = b"FTYP0123456789MP4_TEST_STREAM_PAYLOAD"
    mp4_path.write_bytes(mp4_bytes)
    os.utime(mp4_path, (now_ts - 20, now_ts - 20))

    png_path = _create_comfy_artwork_png(
        dl_root / "sub" / "anima_render.png",
        width=640,
        height=480,
        seed=987654321,
        steps=30,
        cfg=5.2,
        sampler="dpmpp_2m",
        scheduler="karras",
        model="anima_v2.safetensors",
        loras=(("style_pastel.safetensors", 0.75), ("chara_luna.safetensors", 0.9)),
        mtime_ts=now_ts - 5,
    )

    # 1) 画廊列表严格隔离到 download_root 并按 mtime 倒序返回
    all_res = backend.list_gallery_items({})
    assert all_res["ok"] is True
    assert Path(all_res["root"]).resolve() == dl_root
    rel_paths = [it["rel_path"] for it in all_res["items"]]
    assert rel_paths == ["sub/anima_render.png", "mock.mp4", "sub/mock.wav"]
    assert all("sibling_secret" not in rp for rp in rel_paths)
    assert all_res["counts"] == {"all": 3, "image": 1, "video": 1, "audio": 1}

    first_item = all_res["items"][0]
    assert first_item["media_url"] == "/__media/sub/anima_render.png"
    assert first_item["subfolder"] == "sub"
    assert first_item["kind"] == "image"
    assert first_item["width"] == 640 and first_item["height"] == 480
    assert first_item["date_str"]
    assert {f["id"]: f["count"] for f in all_res["folders"]} == {"sub": 2, "__root__": 1}

    assert [x["name"] for x in backend.list_gallery_items({"kind": "video"})["items"]] == ["mock.mp4"]
    assert [x["name"] for x in backend.list_gallery_items({"kind": "audio"})["items"]] == ["mock.wav"]
    assert [x["name"] for x in backend.list_gallery_items({"subfolder": "sub"})["items"]] == ["anima_render.png", "mock.wav"]
    assert [x["name"] for x in backend.list_gallery_items({"subfolder": "__root__"})["items"]] == ["mock.mp4"]
    assert [x["name"] for x in backend.list_gallery_items({"sort": "oldest"})["items"]] == ["mock.wav", "mock.mp4", "anima_render.png"]
    assert [x["name"] for x in backend.list_gallery_items({"sort": "name_asc"})["items"]] == ["anima_render.png", "mock.mp4", "mock.wav"]
    assert [x["name"] for x in backend.list_gallery_items({"query": "anima"})["items"]] == ["anima_render.png"]

    # 2) inspect_gallery_item 精准解析 prompt / negative / seed / steps / loras
    insp = backend.inspect_gallery_item("sub/anima_render.png")
    assert insp["ok"] is True
    meta = insp["meta"]
    assert "anime cel shading" in meta["positive_prompt"]
    assert "extra fingers" in meta["negative_prompt"]
    assert meta["seed"] == 987654321
    assert meta["steps"] == 30
    assert meta["cfg"] == pytest.approx(5.2, abs=1e-3)
    assert meta["sampler"] == "dpmpp_2m"
    assert meta["scheduler"] == "karras"
    assert meta["model"] == "anima_v2.safetensors"
    assert meta["width"] == 640 and meta["height"] == 480
    assert meta["has_workflow"] is True
    assert meta["loras"] == [
        {"name": "style_pastel.safetensors", "strength": 0.75},
        {"name": "chara_luna.safetensors", "strength": 0.9},
    ]

    # 3) 路径越界拦截校验
    for bad_rel in ("../secret.txt", "..\\secret.txt", "/../../secret.txt", "sub/../../sibling_secret.png"):
        bad_insp = backend.inspect_gallery_item(bad_rel)
        assert bad_insp["ok"] is False and bad_insp.get("error") == "forbidden", bad_rel
        bad_rev = backend.reveal_gallery_item(bad_rel)
        assert bad_rev["ok"] is False and bad_rev.get("error") == "forbidden", bad_rel
        bad_push = backend.push_gallery_workflow_to_canvas(bad_rel)
        assert bad_push["ok"] is False and bad_push.get("error") == "forbidden", bad_rel

    # 4) HTTP _PetAssetHandler 校验静态文件 200、/__media 200、Range 206 与越界 403
    httpd = pet_app._ThreadingHTTPServer(("127.0.0.1", 0), pet_app._PetAssetHandler)
    httpd.media_root = dl_root  # type: ignore[attr-defined]
    port = httpd.server_address[1]
    srv_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    srv_thread.start()
    try:
        for static_path, ctype_hint in (
            ("/studio.html", "text/html"),
            ("/studio.css", "text/css"),
            ("/studio.js", "application/javascript"),
        ):
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            conn.request("GET", static_path)
            resp = conn.getresponse()
            body = resp.read()
            conn.close()
            assert resp.status == 200, static_path
            assert ctype_hint in (resp.getheader("Content-Type") or ""), static_path
            assert "no-store" in (resp.getheader("Cache-Control") or ""), static_path
            assert len(body) > 100

        # 全量 GET 媒体文件 -> 200
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("GET", "/__media/mock.mp4")
        resp = conn.getresponse()
        full_data = resp.read()
        conn.close()
        assert resp.status == 200
        assert full_data == mp4_bytes
        assert resp.getheader("Accept-Ranges") == "bytes"

        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("GET", "/__media/sub/mock.wav")
        resp = conn.getresponse()
        wav_data = resp.read()
        conn.close()
        assert resp.status == 200
        assert wav_data == wav_path.read_bytes()

        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("GET", "/__media/sub/anima_render.png")
        resp = conn.getresponse()
        png_data = resp.read()
        conn.close()
        assert resp.status == 200
        assert png_data == png_path.read_bytes()

        # Range 请求 -> 206 Partial Content
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("GET", "/__media/mock.mp4", headers={"Range": "bytes=0-3"})
        resp = conn.getresponse()
        part_data = resp.read()
        conn.close()
        assert resp.status == 206
        assert part_data == b"FTYP"
        assert resp.getheader("Content-Range") == f"bytes 0-3/{len(mp4_bytes)}"

        # 路径越界 -> 403
        for trav_url in ("/__media/../../secret.txt", "/__media/%2e%2e/secret.txt"):
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            conn.request("GET", trav_url)
            resp = conn.getresponse()
            resp.read()
            conn.close()
            assert resp.status == 403, f"越界路径未被 403 拦截: {trav_url} -> {resp.status}"
    finally:
        httpd.shutdown()
        httpd.server_close()

    # 5) tail_live_logs 本地诊断日志与远端日志密钥强制脱敏
    raw_oauth = "ya29.a0AfH6SMFakeAccessToken1234567890abcdef"
    raw_refresh = "1//09FakeRefreshToken1234567890abcdef"
    raw_api_key = "sk-pet-FakeStudioKey1234567890abcd"
    raw_jwt = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
    backend._diag.info(
        "studio-test",
        f"auth={raw_oauth} ref={raw_refresh} key={raw_api_key} jwt={raw_jwt}",
    )
    local_logs = backend.tail_live_logs({"source": "local_diag", "lines": 50})
    assert local_logs["ok"] is True
    joined_local = "\n".join(local_logs["lines"])
    for secret in (raw_oauth, raw_refresh, raw_api_key, raw_jwt):
        assert secret not in joined_local
    assert "studio-test" in joined_local

    # 6) 远端日志（comfy / llama）拉取与脱敏、工作流推回画布、云盘资产结构化兜底
    fake_py = isolated_env.tmp / "fake_python.exe"
    fake_py.write_bytes(b"MZ")
    backend._get_colab_py = lambda: fake_py  # type: ignore[method-assign]
    backend.state["cluster_slots"] = {
        "comfy": {
            "account_id": "acct_g4",
            "endpoint": "ep-comfy-1",
            "tunnel_url": "https://cfy-test.trycloudflare.com",
            "online": True,
        },
        "llm": {
            "account_id": "acct_l4",
            "endpoint": "ep-llm-1",
            "tunnel_url": "https://llm-test.trycloudflare.com",
            "online": True,
        },
    }
    fake_tail_payload = {
        "ok": True,
        "path": "root/comfy_colab.log",
        "lines": [
            f"INFO booted token={raw_api_key}",
            f"WARN oauth={raw_oauth}",
        ],
    }
    with patch.object(
        cb,
        "_run_hidden",
        return_value=CommandResult(
            ["fake"],
            0,
            stdout=f"<<<COLAB_CONTENTS_TAIL>>>{json.dumps(fake_tail_payload)}<<<END_COLAB_CONTENTS_TAIL>>>",
            stderr="",
        ),
    ):
        remote_logs = backend.tail_live_logs({"source": "comfy", "lines": 80})
    assert remote_logs["ok"] is True
    joined_remote = "\n".join(remote_logs["lines"])
    assert raw_api_key not in joined_remote and raw_oauth not in joined_remote
    assert "INFO booted" in joined_remote

    # push_gallery_workflow_to_canvas（复用 _upload_workflow_file_ex 与直推画布）
    with patch.object(
        backend,
        "_upload_workflow_file_ex",
        return_value=("已上传工作流", True, "anima_render.json"),
    ), patch.object(
        backend,
        "_push_or_open_workflow_on_canvas",
        return_value={"pushed_live": True, "opened_browser": False, "url": "https://cfy-test.trycloudflare.com"},
    ):
        push_res = backend.push_gallery_workflow_to_canvas("sub/anima_render.png")
    assert push_res["ok"] is True
    assert push_res["pushed_live"] is True
    assert push_res["workflow_name"] == "anima_render.json"

    # list_cloud_assets 未配置时安全返回 ok=False 且不抛异常
    cloud_unconf = backend.list_cloud_assets()
    assert cloud_unconf["ok"] is False
    assert cloud_unconf["loras"] == [] and cloud_unconf["workflows"] == [] and cloud_unconf["refs"] == []

    # 配置 [gdrive_save] 后验证 list_cloud_assets 结构化聚合
    gdrive_conf = isolated_env.tmp / "gdrive_rclone.conf"
    gdrive_conf.write_text("[gdrive_save]\ntype = drive\n", encoding="utf-8")
    monkeypatch.setenv("COLAB_PET_RCLONE_CONF", str(gdrive_conf))
    with patch.object(
        cb,
        "_run_hidden",
        return_value=CommandResult(
            ["rclone"],
            0,
            stdout=json.dumps([{"Name": "demo_asset.bin", "Path": "demo_asset.bin", "Size": 2048, "IsDir": False}]),
            stderr="",
        ),
    ):
        cloud_ok = backend.list_cloud_assets()
    assert cloud_ok["ok"] is True
    assert len(cloud_ok["loras"]) == 1 and cloud_ok["loras"][0]["name"] == "demo_asset.bin"

    # 7) backend.subscribe_events 多播订阅、脱敏与取消订阅隔离
    sub_a: list[dict[str, Any]] = []
    sub_b: list[dict[str, Any]] = []
    unsub_a = backend.subscribe_events(sub_a.append)
    unsub_b = backend.subscribe_events(sub_b.append)
    backend._emit_event({"type": "state_sync", "msg": f"leaked={raw_api_key}"})
    assert len(sub_a) == 1 and len(sub_b) == 1
    assert raw_api_key not in sub_a[0]["msg"]
    unsub_a()
    backend._emit_event({"type": "state_sync", "msg": "second"})
    assert len(sub_a) == 1 and len(sub_b) == 2
    unsub_b()

    # 8) colab_cli_bridge.cmd_contents_tail 在真实 colab_py 环境验证自动探测 comfy 日志并输出标准帧
    if cb.COLAB_PY.is_file():
        bridge_code = f"""
import io, json, sys
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, {str(PKG_DIR)!r})
from pathlib import Path
import colab_cli_bridge as br

tmp = Path({str(isolated_env.tmp)!r})
br.CFG_DIR = tmp / "colab_cfg_tail"
br.SESS_FILE = br.CFG_DIR / "sessions.json"
br.SESS_BACKUP_DIR = br.CFG_DIR / "pet_sessions_by_acct"
br.CFG_DIR.mkdir(parents=True, exist_ok=True)

fake_target = SimpleNamespace(
    runtime_proxy_info=SimpleNamespace(url="https://proxy.example.com/", token="tok-x")
)
fake_resp = SimpleNamespace(
    status_code=200,
    json=lambda: {{"type": "file", "format": "text", "content": "line-1\\nline-2\\nline-3\\n"}},
)
with patch.object(br, "_pick_target_assignment", return_value=(fake_target, None)), patch.object(
    br, "_contents_get", return_value=fake_resp
):
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = br.cmd_contents_tail(["--ep=ep-comfy-1", "--lines=2"])
    assert rc == 0
    out_txt = buf.getvalue()
    assert "<<<COLAB_CONTENTS_TAIL>>>" in out_txt and "root/comfy_colab.log" in out_txt
    assert "line-2" in out_txt and "line-3" in out_txt
"""
        sub = subprocess.run(
            [str(cb.COLAB_PY), "-c", bridge_code],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=25,
            env=dict(os.environ),
        )
        assert sub.returncode == 0, f"colab_py contents-tail 校验失败:\n{sub.stdout}\n{sub.stderr}"


# =========================================================================
# 2. 同进程单 Backend 双窗口架构 + 真实 PyWebView 渲染 + 5 张高清截图核验
# =========================================================================


def _build_demo_snapshots(backend: cb.ColabPetBackend) -> tuple[dict[str, Any], dict[str, Any]]:
    """构造用于双窗真渲染测试的待机态与双号集群在线态快照。"""
    idle_snap: dict[str, Any] = {
        "ok": True,
        "stale": False,
        "error_msg": None,
        "balance": 168.40,
        "baseline_balance": 200.0,
        "balance_pct": 84.2,
        "rate_hourly": 0.0,
        "rate_pct": 0.0,
        "remaining_hours": None,
        "assignments_count": 0,
        "sessions": [],
        "is_running": False,
        "has_orphan": False,
        "active_minutes": 0,
        "active_since_ts": None,
        "long_running_alert": False,
        "low_balance": False,
        "mood": "proud",
        "status_level": "idle",
        "status_label": "已停机省积分",
        "date": "2026-11-18",
        "day_start_balance": 175.0,
        "day_consumed_cu": 6.60,
        "day_topup_cu": 0.0,
        "session_consumed_cu": 0.0,
        "last_delta": 0.0,
        "updated_str": "03:30:00",
        "updated_ts": int(time.time()),
        "eligible_gpus": ["G4", "A100", "L4", "T4"],
        "eligible_tpus": [],
        "scarce_gpus": {},
        "measured_gpu_rates": dict(cb.DEFAULT_MEASURED_GPU_RATES),
        "deploy": {"active": False},
        "prefs": dict(backend.state["prefs"]),
        "comfy": cb.ColabPetBackend._default_comfy_state(),
        "cluster": {
            "active": False,
            "can_split": True,
            "pool_healthy_count": 2,
            "total_pool_balance": 233.6,
            "total_rate_hourly": 0.0,
            "slots": {},
        },
        "onboarding": {
            "needs_attention": False,
            "colab_cli_ok": True,
            "colab_exe_ok": True,
            "colab_auth_ok": True,
            "auth_initializing": False,
            "auth_missing": False,
            "auth_expired": False,
            "rclone_exe_ok": True,
            "gdrive_ok": True,
            "gdrive_free_gb": 142.8,
            "gdrive_total_gb": 200.0,
            "active_account_id": "acct_g4",
            "active_account_label": "Ultra主力渲染号",
            "accounts": [
                {
                    "id": "acct_l4",
                    "label": "Magpie对话副号",
                    "email_masked": "mag***pi@gmail.com",
                    "last_balance": 65.2,
                    "last_is_running": False,
                    "eligible_gpus": ["L4", "T4"],
                    "cluster_role": None,
                },
                {
                    "id": "acct_g4",
                    "label": "Ultra主力渲染号",
                    "email_masked": "ult***99@gmail.com",
                    "last_balance": 168.4,
                    "last_is_running": False,
                    "eligible_gpus": ["G4", "A100", "L4", "T4"],
                    "cluster_role": None,
                },
                {
                    "id": "acct_t4",
                    "label": "受限仅T4备用号",
                    "email_masked": "t4o***01@gmail.com",
                    "last_balance": 9.5,
                    "last_is_running": False,
                    "eligible_gpus": ["T4"],
                    "cluster_role": None,
                },
            ],
            "pending_oauth": False,
            "gdrive_binding": False,
            "cf_tunnel": {
                "mode": "named",
                "comfy_url": "https://cfy.example.com",
                "llm_url": "https://llm.example.com",
            },
            "tool_paths": {
                "colab_exe": "colab.exe",
                "colab_py": "python.exe",
                "rclone_exe": "rclone.exe",
                "rclone_conf": "rclone.conf",
            },
        },
    }

    active_snap = json.loads(json.dumps(idle_snap))
    active_snap.update(
        {
            "is_running": True,
            "rate_hourly": 10.05,
            "rate_pct": 67.0,
            "remaining_hours": 19.4,
            "assignments_count": 1,
            "active_minutes": 24,
            "active_since_ts": int(time.time()) - 24 * 60,
            "status_level": "running",
            "status_label": "双号集群计费中",
            "sessions": [
                {
                    "name": "comfy",
                    "endpoint": "ep-comfy-g4",
                    "accelerator": "G4",
                    "machine_shape": "STANDARD",
                }
            ],
            "comfy": {
                **cb.ColabPetBackend._default_comfy_state(),
                "online": True,
                "service_kind": "dual",
                "comfy_online": True,
                "llm_online": True,
                "tunnel_url": "https://cfy-demo.trycloudflare.com",
                "llm_tunnel_url": "https://llm.example.com",
                "active_profiles": ["anima", "qwen_27b"],
                "model_name": "Qwen3.8-27B-abliterated",
                "quant": "UD-Q4_K_XL",
                "ctx_size": 65536,
                "gpu_name": "RTX PRO 6000",
                "gpu_full_name": "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
                "vram_used_gb": 54.8,
                "vram_total_gb": 96.0,
                "vram_pct": 57.1,
                "ram_used_gb": 103.4,
                "ram_total_gb": 176.9,
                "ram_free_gb": 73.5,
                "ram_pct": 58.5,
                "disk_used_gb": 135.1,
                "disk_total_gb": 235.7,
                "disk_free_gb": 100.6,
                "disk_pct": 57.3,
                "shm_used_gb": 0.0,
                "shm_total_gb": 88.0,
                "shm_free_gb": 88.0,
                "drive_persisted": True,
                "drive_free_gb": 142.8,
                "drive_total_gb": 200.0,
                "tps": 42.6,
            },
            "cluster": {
                "active": True,
                "can_split": True,
                "pool_healthy_count": 2,
                "total_pool_balance": 233.6,
                "total_rate_hourly": 11.99,
                "slots": {
                    "llm": {
                        "account_id": "acct_l4",
                        "label": "Magpie对话副号",
                        "gpu": "L4",
                        "endpoint": "ep-llm-l4",
                        "rate_hourly": 1.94,
                        "tunnel_url": "https://llm.example.com",
                        "online": True,
                    },
                    "comfy": {
                        "account_id": "acct_g4",
                        "label": "Ultra主力渲染号",
                        "gpu": "G4",
                        "endpoint": "ep-comfy-g4",
                        "rate_hourly": 10.05,
                        "tunnel_url": "https://cfy-demo.trycloudflare.com",
                        "online": True,
                    },
                },
            },
        }
    )
    active_snap["onboarding"]["accounts"][0]["last_is_running"] = True
    active_snap["onboarding"]["accounts"][0]["cluster_role"] = "llm"
    active_snap["onboarding"]["accounts"][1]["last_is_running"] = True
    active_snap["onboarding"]["accounts"][1]["cluster_role"] = "comfy"
    return idle_snap, active_snap


def _run_pywebview_dual_window_child(work_dir: Path, shot_dir: Path) -> int:
    """在独立 STA 主线程拉起桌宠 + Studio 同进程双窗口，执行真实 CDP 点击、事件同步、像素核验与截图。"""
    import webview  # noqa: WPS433

    shot_dir.mkdir(parents=True, exist_ok=True)
    backend = cb.ColabPetBackend(
        state_path=work_dir / "wv_state.json",
        download_root=work_dir / "wv_dl",
        cf_config_path=work_dir / "wv_cf.json",
    )
    backend.update_prefs(
        {
            "boot_profile": "anima",
            "boot_profiles": ["anima", "qwen_27b"],
            "boot_gpu": "L4",
            "cluster_split_mode": True,
            "auto_pool_schedule": True,
            "on_top": True,
        }
    )

    # 预置丰富画廊作品（含真实 ComfyUI 元数据 PNG、MP4、WAV），保证瀑布流与 Inspector 美观真实
    dl_root = backend.download_root.resolve()
    now_ts = time.time()
    _create_mock_wav(dl_root / "drama3_voice_line.wav", mtime_ts=now_ts - 40)
    (dl_root / "minimax_motion_clip.mp4").write_bytes(b"FTYP_MOCK_VIDEO_STREAM")
    os.utime(dl_root / "minimax_motion_clip.mp4", (now_ts - 30, now_ts - 30))
    _create_comfy_artwork_png(
        dl_root / "pastel_garden_03.png",
        width=720,
        height=540,
        palette=((167, 139, 250), (245, 243, 255)),
        seed=11223344,
        steps=24,
        cfg=4.0,
        mtime_ts=now_ts - 20,
    )
    _create_comfy_artwork_png(
        dl_root / "cyber_set" / "cyber_rain_02.png",
        width=640,
        height=800,
        palette=((79, 70, 229), (224, 231, 255)),
        seed=55667788,
        steps=26,
        cfg=4.8,
        mtime_ts=now_ts - 10,
    )
    _create_comfy_artwork_png(
        dl_root / "anime_atelier_01.png",
        width=768,
        height=576,
        palette=((124, 58, 237), (250, 245, 255)),
        positive="1girl, lavender eyes, soft sunlit studio desk, delicate ink lineart, masterpiece",
        negative="lowres, bad anatomy, text, watermark, blurry",
        seed=4208819273,
        steps=28,
        cfg=4.5,
        sampler="euler",
        scheduler="normal",
        model="anima_v2_fp16.safetensors",
        loras=(
            ("chara_mizuki_v3.safetensors", 0.85),
            ("pastel_anime_line.safetensors", 0.65),
        ),
        mtime_ts=now_ts - 2,
    )

    # 预置本地诊断日志与模拟云盘资产 / 体检响应，使 4 个视图内容饱满无假空状态
    backend._diag.info("pet", "Colab Studio 同进程双窗服务初始化完成")
    backend._diag.info("cluster", "异构集群探测就绪: LLM(L4) + ComfyUI(G4)")
    backend._diag.warn("tunnel", "Cloudflare 边缘延迟略高 (142ms)，已自动保活")
    backend._diag.info("gallery", "已索引本地 5 项创作资产（图像/视频/语音）")

    backend._studio_svc.list_cloud_assets = lambda: {  # type: ignore[method-assign]
        "ok": True,
        "configured": True,
        "message": "已加载云盘资产：LoRA 3 / 工作流 2 / 音色 2",
        "loras": [
            {"name": "chara_mizuki_v3.safetensors", "path": "chara_mizuki_v3.safetensors", "size_bytes": 152043520, "mod_time": "2026-11-17T22:10:00Z"},
            {"name": "pastel_anime_line.safetensors", "path": "pastel_anime_line.safetensors", "size_bytes": 75497472, "mod_time": "2026-11-16T18:30:00Z"},
            {"name": "velvet_lighting_xl.safetensors", "path": "velvet_lighting_xl.safetensors", "size_bytes": 118489088, "mod_time": "2026-11-15T14:05:00Z"},
        ],
        "workflows": [
            {"name": "anima_cel_shading_v2.json", "path": "anima_cel_shading_v2.json", "size_bytes": 48120, "mod_time": "2026-11-17T20:00:00Z"},
            {"name": "minimax_h3_r2v.json", "path": "minimax_h3_r2v.json", "size_bytes": 62410, "mod_time": "2026-11-16T11:20:00Z"},
        ],
        "refs": [
            {"name": "mizuki_calm_ref.wav", "path": "mizuki_calm_ref.wav", "size_bytes": 924180, "mod_time": "2026-11-17T09:15:00Z"},
            {"name": "narrator_warm_zh.wav", "path": "narrator_warm_zh.wav", "size_bytes": 1245184, "mod_time": "2026-11-15T16:40:00Z"},
        ],
    }
    backend.run_preflight = lambda deep=True: {  # type: ignore[method-assign]
        "ok": True,
        "ok_count": 5,
        "warn_count": 1,
        "fail_count": 0,
        "versions": {"Python": "3.14", "PyWebView": "5.x", "rclone": "1.68.2"},
        "items": [
            {"name": "google-colab-cli", "category": "CLI", "status": "ok", "summary": "已就绪并完成授权校验", "detail": "colab-cli 0.4.2"},
            {"name": "Google Ultra 多号池", "category": "账号", "status": "ok", "summary": "3 个账号已纳管，余额总计 233.60 CU", "detail": "acct_g4 / acct_l4 / acct_t4"},
            {"name": "Cloudflare Named Tunnel", "category": "网络", "status": "ok", "summary": "固定域名路由正常", "detail": "cfy.example.com · llm.example.com"},
            {"name": "rclone gdrive_save", "category": "云盘", "status": "ok", "summary": "云端资产仓库读写正常 (142.8 GB 可用)", "detail": "Colab_Comfy_LoRAs / Workflows / Refs"},
            {"name": "Windows 凭据保险箱", "category": "安全", "status": "ok", "summary": "DPAPI / Vault 零明文落盘保护已启用", "detail": "COLAB_PET_MOCK_VAULT_PATH"},
            {"name": "FFmpeg 媒体转码", "category": "可选", "status": "warn", "summary": "未检测到系统 ffmpeg（不影响浏览器原生播片）", "detail": "可选安装"},
        ],
    }

    idle_snap, active_snap = _build_demo_snapshots(backend)
    backend.last_snapshot = json.loads(json.dumps(active_snap))

    stop_event = threading.Event()
    api = pet_app.PetBridgeApi(backend, stop_event)

    # 校验所有暴露到 JS 的新 Studio API 签名均无 *args / **kwargs
    for m_name in (
        "open_studio_window",
        "studio_window_ctl",
        "begin_studio_drag",
        "begin_studio_resize",
        "list_gallery_items",
        "inspect_gallery_item",
        "push_gallery_workflow_to_canvas",
        "reveal_gallery_item",
        "list_cloud_assets",
        "tail_live_logs",
        "run_preflight",
    ):
        fn = getattr(api, m_name, None)
        assert callable(fn), f"PetBridgeApi 缺少方法: {m_name}"
        spec = inspect.getfullargspec(fn.__func__)
        assert spec.varargs is None and spec.varkw is None, f"方法参数含可变参数: {m_name} -> {spec}"

    httpd = pet_app._ThreadingHTTPServer(("127.0.0.1", 0), pet_app._PetAssetHandler)
    httpd.media_root = dl_root  # type: ignore[attr-defined]
    port = httpd.server_address[1]
    api._asset_base_url = f"http://127.0.0.1:{port}"
    threading.Thread(target=httpd.serve_forever, daemon=True).start()

    pet_win = webview.create_window(
        title="Colab 桌宠 · 双窗验收",
        url=f"http://127.0.0.1:{port}/ui.html",
        js_api=api,
        width=pet_app.BASE_W,
        height=pet_app.BASE_H,
        x=80,
        y=80,
        resizable=False,
        frameless=True,
        easy_drag=False,
        shadow=False,
        transparent=False,
        background_color=pet_app.EDGE_BG_HEX,
        focus=True,
        on_top=True,
    )
    api._bind_window(pet_win)
    # 模拟 run_app 中桌宠事件推送回调，验证单 Backend 多播同时送达桌宠与 Studio 双窗
    backend._event_cb = lambda evt: None if stop_event.is_set() else pet_app._push_event_to_ui(pet_win, api, evt)
    errs: list[BaseException] = []

    def _cdp_call(win: webview.Window, method: str, params: dict[str, Any], *, allow_dispose: bool = False) -> Any:
        form = win.native
        assert form is not None
        t_box: list[Any] = []
        e_box: list[BaseException] = []
        p_str = json.dumps(params, ensure_ascii=False)

        def _start() -> None:
            try:
                wv = next(c for c in form.Controls if type(c).__name__ == "WebView2")
                t_box.append(wv.CoreWebView2.CallDevToolsProtocolMethodAsync(method, p_str))
            except BaseException as ex:
                e_box.append(ex)

        try:
            form.Invoke(pet_app.DotNetAction(_start))
        except BaseException as exc:
            if allow_dispose and ("ObjectDisposed" in repr(exc) or getattr(form, "IsDisposed", False)):
                return ""
            raise
        if e_box:
            raise e_box[0]
        task = t_box[0]
        deadline = time.monotonic() + 8.0
        while not task.IsCompleted:
            if allow_dispose and getattr(form, "IsDisposed", False):
                return ""
            if time.monotonic() > deadline:
                raise TimeoutError(f"CDP 调用超时: {method}")
            time.sleep(0.005)
        res_box: list[Any] = []

        def _read_res() -> None:
            if task.IsFaulted:
                e_box.append(RuntimeError(str(task.Exception)))
            else:
                res_box.append(task.Result)

        if allow_dispose and getattr(form, "IsDisposed", False):
            return ""
        try:
            form.Invoke(pet_app.DotNetAction(_read_res))
        except BaseException as exc:
            if allow_dispose and ("ObjectDisposed" in repr(exc) or getattr(form, "IsDisposed", False)):
                return ""
            raise
        if e_box:
            raise e_box[0]
        return res_box[0] if res_box else ""

    def _click_sel(win: webview.Window, sel: str, *, allow_dispose: bool = False) -> None:
        pt = win.evaluate_js(
            f"(() => {{ const el = document.querySelector({sel!r});"
            " if (!el) return null;"
            " const r = el.getBoundingClientRect();"
            " return {x: r.left + r.width / 2, y: r.top + r.height / 2, w: r.width, h: r.height}; })()"
        )
        assert pt and pt["w"] > 0 and pt["h"] > 0, f"目标元素不可见或未找到: {sel}"
        fx, fy = float(pt["x"]), float(pt["y"])
        _cdp_call(win, "Input.dispatchMouseEvent", {"type": "mouseMoved", "x": fx, "y": fy, "button": "none", "buttons": 0})
        _cdp_call(win, "Input.dispatchMouseEvent", {"type": "mousePressed", "x": fx, "y": fy, "button": "left", "buttons": 1, "clickCount": 1})
        time.sleep(0.02)
        _cdp_call(
            win,
            "Input.dispatchMouseEvent",
            {"type": "mouseReleased", "x": fx, "y": fy, "button": "left", "buttons": 0, "clickCount": 1},
            allow_dispose=allow_dispose,
        )

    def _save_cdp_screenshot(win: webview.Window, filename: str) -> Path:
        raw_json = _cdp_call(win, "Page.captureScreenshot", {"format": "png"})
        parsed = json.loads(raw_json)
        png_bytes = base64.b64decode(parsed["data"])
        target = shot_dir / filename
        target.write_bytes(png_bytes)
        return target

    def _worker() -> None:
        try:
            pet_win.events.loaded.wait(timeout=10.0)
            for _ in range(60):
                if pet_win.native is not None:
                    break
                time.sleep(0.05)
            assert pet_win.native is not None
            pet_form = pet_win.native
            pet_form.Invoke(pet_app.DotNetAction(lambda: (setattr(pet_form, "TopMost", True), pet_form.Show(), pet_form.Activate())))
            pet_win.resize(pet_app.BASE_W, pet_app.BASE_H)

            # 等待桌宠 pywebviewready 完成首轮快照同步
            ready_deadline = time.monotonic() + 8.0
            pet_ready = False
            while time.monotonic() < ready_deadline:
                try:
                    pet_ready = bool(
                        pet_win.evaluate_js(
                            "Boolean(window.pywebview && window.pywebview.api && typeof window.renderState === 'function' && latestSnap)"
                        )
                    )
                except Exception:
                    pet_ready = False
                if pet_ready:
                    break
                time.sleep(0.05)
            assert pet_ready, "等待桌宠 pywebviewready 超时"
            pet_app._push_to_ui(pet_win, active_snap)
            pet_win.evaluate_js("setPage(0); toggleBubble(true);")
            time.sleep(0.45)

            # 1) 校验桌宠 5 页 oy == 0 且 .side-tools 与 #tool-studio 无裁切
            for pg_idx in range(5):
                pet_win.evaluate_js(f"setPage({pg_idx});")
                time.sleep(0.22)
                pet_diag = pet_win.evaluate_js(
                    "(() => {"
                    "  const st = el('stage');"
                    "  const tools = el('side-tools');"
                    "  const stuBtn = el('tool-studio');"
                    "  const bbl = el('bubble-box');"
                    "  const tr = tools.getBoundingClientRect();"
                    "  const sr = stuBtn.getBoundingClientRect();"
                    "  const br = bbl.getBoundingClientRect();"
                    "  return {"
                    "    stageSh: st.scrollHeight,"
                    "    pages: [...document.querySelectorAll('#pages-track > .page-body')].map((p, i) => ({i, oy: p.scrollHeight - p.clientHeight})),"
                    "    toolsRect: {l: tr.left, t: tr.top, r: tr.right, b: tr.bottom, sw: tools.scrollWidth, cw: tools.clientWidth, sh: tools.scrollHeight, ch: tools.clientHeight},"
                    "    studioRect: {l: sr.left, t: sr.top, r: sr.right, b: sr.bottom, w: sr.width, h: sr.height},"
                    "    bubbleBottom: br.bottom,"
                    "    studioText: stuBtn.textContent.trim()"
                    "  };"
                    "})()"
                )
                assert pet_diag["stageSh"] <= 548, f"桌宠第 {pg_idx+1} 页 stage 高度溢出: {pet_diag}"
                for pg in pet_diag["pages"]:
                    assert pg["oy"] == 0, f"桌宠 Page {pg['i']+1} 纵向溢出 oy={pg['oy']}"
                assert pet_diag["studioRect"]["w"] >= 24 and pet_diag["studioRect"]["h"] >= 24, pet_diag
                assert pet_diag["toolsRect"]["t"] >= pet_diag["bubbleBottom"] - 1, f".side-tools 触碰主气泡底边: {pet_diag}"
                assert pet_diag["toolsRect"]["r"] <= 392 and pet_diag["toolsRect"]["b"] <= 548, f".side-tools 被裁切: {pet_diag}"
                assert not _EMOJI_RE.search(pet_diag["studioText"]), f"#tool-studio 不得含 Emoji: {pet_diag}"

            pet_win.evaluate_js("setPage(0);")
            time.sleep(0.25)
            # 截图 1：桌宠+工作台入口图（用带窗口阴影/区域合成的原生截图）
            pet_app._capture_hwnd_png(api._native_hwnd(), shot_dir / "01_pet_studio_entry.png")

            # 安装桌宠端事件监听间谍，准备核验同进程单 Backend 事件多播
            pet_win.evaluate_js(
                "window.__petEvents = [];"
                "const __origPetEvt = window.onBackendEvent;"
                "window.onBackendEvent = function(e){ window.__petEvents.push(e); if (__origPetEvt) return __origPetEvt(e); };"
            )

            # 2) 从桌宠窗体 #tool-studio 点击打开 1360×840 无边框 Studio 窗口
            _click_sel(pet_win, "#tool-studio")
            studio_deadline = time.monotonic() + 10.0
            while time.monotonic() < studio_deadline:
                if api._studio_window is not None and api._studio_hwnd():
                    break
                time.sleep(0.05)
            studio_win = api._studio_window
            assert studio_win is not None and api._studio_hwnd(), "点击 #tool-studio 未成功打开 Studio 窗口"
            assert studio_win is not pet_win

            for _ in range(80):
                if studio_win.native is not None:
                    break
                time.sleep(0.05)
            assert studio_win.native is not None

            studio_ready = False
            ready_dl2 = time.monotonic() + 10.0
            while time.monotonic() < ready_dl2:
                try:
                    studio_ready = bool(
                        studio_win.evaluate_js(
                            "Boolean(window.pywebview && window.pywebview.api && typeof window.renderStudioState === 'function' && typeof window.onStudioBackendEvent === 'function')"
                        )
                    )
                except Exception:
                    studio_ready = False
                if studio_ready:
                    break
                time.sleep(0.05)
            assert studio_ready, "Studio 窗口 JS 上下文就绪超时"
            time.sleep(0.35)

            # 校验同进程 os.getpid() 与 id(backend) 同一对象，及 1360×840 默认尺寸（兼容小屏工作区钳制）
            reuse_info = api.open_studio_window()
            assert reuse_info["ok"] is True
            assert reuse_info["reused"] is True
            assert reuse_info["pid"] == os.getpid()
            assert reuse_info["backend_id"] == id(backend)

            win_state = api.studio_window_ctl("get_window_state")
            assert win_state["open"] is True
            assert 1075 <= win_state["width"] <= 1370 and 695 <= win_state["height"] <= 850, win_state

            # 校验 maximize / restore 窗口控制
            max_res = api.studio_window_ctl("toggle_maximize")
            assert max_res["ok"] is True and max_res["maximized"] is True
            time.sleep(0.15)
            rest_res = api.studio_window_ctl("toggle_maximize")
            assert rest_res["ok"] is True and rest_res["maximized"] is False
            time.sleep(0.20)

            # 安装 Studio 事件间谍并校验双窗口共同接收同一个 backend._emit_event 多播
            studio_win.evaluate_js(
                "window.__studioEvents = [];"
                "const __origStuEvt = window.onStudioBackendEvent;"
                "window.onStudioBackendEvent = function(e){ window.__studioEvents.push(e); if (__origStuEvt) return __origStuEvt(e); };"
            )
            pushed_snap = json.loads(json.dumps(active_snap))
            pushed_snap["updated_str"] = "03:33:33"
            backend._emit_event({"type": "state_sync", "snapshot": pushed_snap})
            time.sleep(0.35)

            pet_evts = pet_win.evaluate_js("window.__petEvents || []")
            stu_evts = studio_win.evaluate_js("window.__studioEvents || []")
            assert any(e.get("type") == "state_sync" and e.get("snapshot", {}).get("updated_str") == "03:33:33" for e in pet_evts), pet_evts
            assert any(e.get("type") == "state_sync" and e.get("snapshot", {}).get("updated_str") == "03:33:33" for e in stu_evts), stu_evts

            def _audit_studio_view(label: str) -> None:
                diag = studio_win.evaluate_js(
                    "(() => {"
                    "  const docOverflow = document.documentElement.scrollWidth - window.innerWidth;"
                    "  const ws = document.querySelector('.studio-workspace');"
                    "  const wsOverflow = ws ? (ws.scrollWidth - ws.clientWidth) : 0;"
                    "  const pane = document.querySelector('.view-pane.active');"
                    "  const paneOverflow = pane ? (pane.scrollWidth - pane.clientWidth) : 0;"
                    "  const scroller = pane ? pane.querySelector('.view-scroll') : null;"
                    "  const scOverflow = scroller ? (scroller.scrollWidth - scroller.clientWidth) : 0;"
                    "  const texts = [];"
                    "  document.querySelectorAll('.sidebar-nav .nav-item, .titlebar-actions button, .view-pane.active button, .pill-tag, .status-chip, .section-title').forEach(n => {"
                    "    const t = (n.textContent || '').trim();"
                    "    if (t) texts.push(t);"
                    "  });"
                    "  const clippedButtons = [];"
                    "  document.querySelectorAll('.sidebar-nav .nav-item, .titlebar-actions button, .view-pane.active button:not(.hidden)').forEach(b => {"
                    "    if (b.closest('.hidden') || b.closest('[aria-hidden=\"true\"]')) return;"
                    "    const r = b.getBoundingClientRect();"
                    "    if (r.width === 0 || r.height === 0) return;"
                    "    if (b.scrollWidth > b.clientWidth + 2 || r.right > window.innerWidth + 1) {"
                    "      clippedButtons.push({id: b.id || '', cls: b.className, txt: (b.textContent || '').trim(), sw: b.scrollWidth, cw: b.clientWidth, r: r.right, iw: window.innerWidth});"
                    "    }"
                    "  });"
                    "  const navTrunc = [];"
                    "  document.querySelectorAll('.nav-label').forEach(l => {"
                    "    if (l.scrollWidth > l.clientWidth + 1) navTrunc.push({txt: l.textContent, sw: l.scrollWidth, cw: l.clientWidth});"
                    "  });"
                    "  return {docOverflow, wsOverflow, paneOverflow, scOverflow, texts, clippedButtons, navTrunc, theme: document.documentElement.dataset.theme};"
                    "})()"
                )
                assert diag["docOverflow"] <= 0, f"[{label}] 窗口横向溢出: {diag}"
                assert diag["wsOverflow"] <= 1, f"[{label}] 工作区横向溢出: {diag}"
                assert diag["paneOverflow"] <= 1, f"[{label}] 视图横向溢出: {diag}"
                assert diag["scOverflow"] <= 1, f"[{label}] 滚动区横向溢出: {diag}"
                assert diag["clippedButtons"] == [], f"[{label}] 按钮文本被截断或越界: {diag['clippedButtons']}"
                assert diag["navTrunc"] == [], f"[{label}] 侧栏导航截断: {diag['navTrunc']}"
                for txt in diag["texts"]:
                    assert not _EMOJI_RE.search(txt), f"[{label}] 检测到禁止出现的 Emoji 字符: {txt!r}"

            # 3) 视图 1：Studio 浅色总览（overview）+ 1:1 左栏三线遥测与分项/同屏切换验收
            studio_win.evaluate_js("window.setStudioTheme('light'); window.switchStudioTab('overview');")
            time.sleep(0.35)
            _audit_studio_view("overview-light")
            tele_diag = studio_win.evaluate_js(
                "({vram: document.getElementById('sb-leg-vram-gb')?.textContent,"
                " ram: document.getElementById('sb-leg-ram-gb')?.textContent,"
                " disk: document.getElementById('sb-leg-disk-gb')?.textContent,"
                " shm: document.getElementById('sb-shm-text')?.textContent,"
                " uptime: document.getElementById('sb-kpi-uptime')?.textContent})"
            )
            assert "96.0 GB" in (tele_diag["vram"] or ""), tele_diag
            assert "176.9 GB" in (tele_diag["ram"] or ""), tele_diag
            assert "235.7 GB" in (tele_diag["disk"] or ""), tele_diag
            assert "88.0/88G" in (tele_diag["shm"] or ""), tele_diag
            assert (tele_diag["uptime"] or "") not in ("", "00:00:00"), tele_diag
            _save_cdp_screenshot(studio_win, "02_studio_overview_light.png")

            # 验证「分项 / 同屏」按钮可无溢出切换
            _click_sel(studio_win, "#btn-sb-tele-mode")
            time.sleep(0.18)
            assert int(studio_win.evaluate_js("document.querySelectorAll('#sb-tele-split .sb-split-item').length") or 0) == 3
            _audit_studio_view("overview-split-telemetry")
            _click_sel(studio_win, "#btn-sb-tele-mode")
            time.sleep(0.15)
            assert studio_win.evaluate_js("document.getElementById('sb-tele-split').classList.contains('hidden')") is True

            # 4) 视图 2：作品画廊 + 点击画廊作品打开右侧 Finder Inspector
            _click_sel(studio_win, "#nav-gallery")
            gal_deadline = time.monotonic() + 5.0
            while time.monotonic() < gal_deadline:
                ok_gal = studio_win.evaluate_js(
                    "document.getElementById('view-gallery').classList.contains('active')"
                    " && document.querySelectorAll('#gallery-masonry .gallery-card').length >= 5"
                )
                if ok_gal:
                    break
                time.sleep(0.05)
            time.sleep(0.45)
            assert int(studio_win.evaluate_js("document.querySelectorAll('#gallery-masonry .gallery-card').length") or 0) == 5

            # 校验瀑布流真从左到右按行首优先排布（第 1/2/3 新作品严格分居第 1/2/3 列首行）
            row_order = studio_win.evaluate_js(
                "[...document.querySelectorAll('#gallery-masonry .gallery-masonry-col')].slice(0, 3).map(col => {"
                "  const c = col.querySelector('.gallery-card');"
                "  if (!c) return null;"
                "  const r = c.getBoundingClientRect();"
                "  return {rel: c.dataset.relPath, left: r.left, top: r.top};"
                "}).filter(Boolean)"
            )
            assert len(row_order) == 3, row_order
            assert [x["rel"] for x in row_order] == [
                "anime_atelier_01.png",
                "cyber_set/cyber_rain_02.png",
                "pastel_garden_03.png",
            ], f"瀑布流首行从左到右顺序未对齐最新时间序: {row_order}"
            assert row_order[0]["left"] < row_order[1]["left"] < row_order[2]["left"], row_order
            assert abs(row_order[0]["top"] - row_order[1]["top"]) <= 2 and abs(row_order[1]["top"] - row_order[2]["top"]) <= 2, row_order

            # 校验筛选栏：目录筛选 -> 重置筛选 -> 实时搜索 -> 规整网格视图切换 -> 切回瀑布流
            _click_sel(studio_win, "#btn-gal-folder")
            _audit_studio_view("gallery-folder-popover")
            _click_sel(studio_win, '#gal-folder-list .gal-pop-opt[data-folder="cyber_set"]')
            time.sleep(0.15)
            assert int(studio_win.evaluate_js("document.querySelectorAll('#gallery-masonry .gallery-card').length") or 0) == 1
            _click_sel(studio_win, "#btn-gal-reset-filter")
            time.sleep(0.15)
            assert int(studio_win.evaluate_js("document.querySelectorAll('#gallery-masonry .gallery-card').length") or 0) == 5

            studio_win.evaluate_js(
                "(() => { const i = document.getElementById('inp-gal-search'); i.value = 'pastel'; i.dispatchEvent(new Event('input')); })()"
            )
            time.sleep(0.12)
            assert int(studio_win.evaluate_js("document.querySelectorAll('#gallery-masonry .gallery-card').length") or 0) == 1
            studio_win.evaluate_js(
                "(() => { const i = document.getElementById('inp-gal-search'); i.value = ''; i.dispatchEvent(new Event('input')); })()"
            )
            time.sleep(0.12)
            assert int(studio_win.evaluate_js("document.querySelectorAll('#gallery-masonry .gallery-card').length") or 0) == 5

            _click_sel(studio_win, "#btn-gal-layout-grid")
            time.sleep(0.2)
            assert studio_win.evaluate_js("Boolean(document.querySelector('.gallery-uniform-grid'))") is True
            _audit_studio_view("gallery-grid-mode")
            _click_sel(studio_win, "#btn-gal-layout-masonry")
            time.sleep(0.2)

            _click_sel(studio_win, '.gallery-card[data-rel-path="anime_atelier_01.png"]')
            insp_deadline = time.monotonic() + 5.0
            pos_txt = ""
            while time.monotonic() < insp_deadline:
                pos_txt = studio_win.evaluate_js(
                    "document.getElementById('gallery-inspector').classList.contains('open') ? document.getElementById('insp-pos-prompt').textContent : ''"
                ) or ""
                if "lavender eyes" in pos_txt:
                    break
                time.sleep(0.05)
            insp_diag = studio_win.evaluate_js(
                "({open: document.getElementById('gallery-inspector').classList.contains('open'),"
                " pos: document.getElementById('insp-pos-prompt').textContent,"
                " file: document.getElementById('insp-filename').textContent,"
                " sel: document.querySelector('.gallery-card.selected')?.dataset?.relPath || ''})"
            )
            assert "lavender eyes" in pos_txt, f"Inspector 未渲染出期望提示词: {insp_diag}"
            time.sleep(0.35)
            _audit_studio_view("gallery-inspector")
            _save_cdp_screenshot(studio_win, "03_studio_gallery_inspector.png")

            # 5) 视图 3：引擎与云盘（tuner）
            _click_sel(studio_win, "#nav-tuner")
            tuner_deadline = time.monotonic() + 5.0
            while time.monotonic() < tuner_deadline:
                rows = studio_win.evaluate_js("document.querySelectorAll('#tuner-cloud-list .inset-row').length")
                if int(rows or 0) >= 3:
                    break
                time.sleep(0.05)
            assert int(studio_win.evaluate_js("document.querySelectorAll('#tuner-cloud-list .inset-row').length") or 0) == 3
            time.sleep(0.35)
            _audit_studio_view("tuner-cloud")
            _save_cdp_screenshot(studio_win, "04_studio_tuner_cloud.png")

            # 6) 视图 4：控制台与体检 + 一键切换深空暗房主题（console-dark）
            _click_sel(studio_win, "#nav-console")
            con_deadline = time.monotonic() + 5.0
            while time.monotonic() < con_deadline:
                ok_con = studio_win.evaluate_js(
                    "document.getElementById('view-console').classList.contains('active')"
                    " && document.querySelectorAll('#console-log-view .log-line').length >= 4"
                    " && document.querySelectorAll('#doctor-items-list .doctor-item').length >= 5"
                )
                if ok_con:
                    break
                time.sleep(0.05)
            assert studio_win.evaluate_js("document.querySelectorAll('#console-log-view .log-line').length >= 4")
            _click_sel(studio_win, "#btn-theme-toggle")
            time.sleep(0.35)
            assert studio_win.evaluate_js("document.documentElement.dataset.theme") == "dark"
            _audit_studio_view("console-dark")
            _save_cdp_screenshot(studio_win, "05_studio_console_dark.png")

            # 切回浅色模式确认双态均正常
            _click_sel(studio_win, "#btn-theme-toggle")
            time.sleep(0.15)
            assert studio_win.evaluate_js("document.documentElement.dataset.theme") == "light"

            # 7) 验证交通灯关闭钮 -> 同进程重开 -> studio_window_ctl("close") 关闭后桌宠保活与后端依然在线
            _click_sel(studio_win, "#btn-win-close", allow_dispose=True)
            close_dl = time.monotonic() + 5.0
            while time.monotonic() < close_dl:
                if api._studio_window is None:
                    break
                time.sleep(0.05)
            assert api._studio_window is None and stop_event.is_set() is False

            _click_sel(pet_win, "#tool-studio")
            reopen_dl = time.monotonic() + 8.0
            while time.monotonic() < reopen_dl:
                if api._studio_window is not None and api._studio_hwnd():
                    break
                time.sleep(0.05)
            assert api._studio_window is not None
            time.sleep(0.25)

            close_res = api.studio_window_ctl("close")
            assert close_res["ok"] is True and close_res["closed"] is True
            time.sleep(0.25)
            assert api._studio_window is None
            assert api.studio_window_ctl("get_window_state")["open"] is False
            assert stop_event.is_set() is False
            assert backend.get_cached_snapshot()["ok"] is True
            assert pet_win.evaluate_js("typeof window.renderState === 'function' && Boolean(el('tool-studio'))") is True
        except BaseException as exc:
            errs.append(exc)
        finally:
            stop_event.set()
            try:
                api._close_studio_window_internal()
            except Exception:
                pass
            try:
                pet_win.destroy()
            except Exception:
                pass
            try:
                httpd.shutdown()
                httpd.server_close()
            except Exception:
                pass

    webview.start(func=_worker)
    if errs:
        raise errs[0]
    print("PYWEBVIEW_DUAL_WINDOW_ALL_OK", flush=True)
    return 0


@pytest.mark.gui
def test_pywebview_dual_window_architecture_and_visual_audit(isolated_env: SimpleNamespace) -> None:
    """真拉起桌宠 + Studio 同进程双窗，核验同进程/同Backend/多播/保活、零Emoji零溢出与 5 张高清截图。"""
    work_dir = isolated_env.tmp / "dual_webview_box"
    work_dir.mkdir(parents=True, exist_ok=True)
    env_shot = str(os.environ.get("COLAB_PET_STUDIO_SCREENSHOT_DIR") or "").strip()
    shot_dir = Path(env_shot) if env_shot else (isolated_env.tmp / "screenshots")
    shot_dir.mkdir(parents=True, exist_ok=True)

    proc = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--dual-webview-child",
            str(work_dir),
            "--shot-dir",
            str(shot_dir),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
        env=dict(os.environ),
    )
    assert proc.returncode == 0 and "PYWEBVIEW_DUAL_WINDOW_ALL_OK" in proc.stdout, (
        f"PyWebView 双窗子进程失败 (rc={proc.returncode}):\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )

    expected_shots = (
        "01_pet_studio_entry.png",
        "02_studio_overview_light.png",
        "03_studio_gallery_inspector.png",
        "04_studio_tuner_cloud.png",
        "05_studio_console_dark.png",
    )
    for name in expected_shots:
        p = shot_dir / name
        assert p.is_file() and p.stat().st_size > 10_000, f"截图缺失或异常过小: {p}"
        with Image.open(p) as im:
            w, h = im.size
            if name == "01_pet_studio_entry.png":
                assert w >= 380 and h >= 520, (name, im.size)
            else:
                assert w >= 1075 and h >= 695, (name, im.size)


# =========================================================================
# 3. Windows 便携打包包含 Studio 全套资产并通过敏感文件安全审计
# =========================================================================


def test_portable_package_contains_studio_assets_and_passes_gate(isolated_env: SimpleNamespace) -> None:
    """调用 build_portable_package 验证 zip 内含 Studio 前后端文件且通过安全闸门。"""
    out_dir = isolated_env.tmp / "dist_out"
    bundle_dir, zip_path = win_pkg.build_portable_package(PKG_DIR, out_dir)
    assert bundle_dir.is_dir() and zip_path.is_file()

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = set(zf.namelist())
        for rel in (
            "Colab桌宠/colab_pet/backend_studio.py",
            "Colab桌宠/colab_pet/studio.html",
            "Colab桌宠/colab_pet/studio.css",
            "Colab桌宠/colab_pet/studio.js",
            "Colab桌宠/Colab桌宠.exe",
        ):
            assert rel in names, f"发行 zip 缺少 Studio 资产: {rel}"
        for forbidden in win_pkg.FORBIDDEN_FILENAMES:
            assert not any(n.endswith(f"/{forbidden}") for n in names), f"发行包夹带禁止文件: {forbidden}"

    # 校验 colab_pet.spec 已同步包含 Studio 前后端静态资产与 backend_studio 模块
    spec_txt = (PKG_DIR / "colab_pet.spec").read_text(encoding="utf-8")
    for required_tok in ("studio.html", "studio.css", "studio.js", "backend_studio"):
        assert required_tok in spec_txt, f"colab_pet.spec 缺少 Studio 条目: {required_tok}"

    # 校验 backend_studio.py 彻底零 Pillow 导入，且纯标准库可准确读取 PNG / WebP / JPEG 宽高
    import ast
    import backend_studio

    studio_ast = ast.parse((PET_DIR / "backend_studio.py").read_text(encoding="utf-8"))
    for node in ast.walk(studio_ast):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name != "PIL" and not alias.name.startswith("PIL.")
        elif isinstance(node, ast.ImportFrom) and node.module:
            assert node.module != "PIL" and not node.module.startswith("PIL.")

    webp_dims = backend_studio._read_image_dimensions(PET_DIR / "assets" / "idle.webp")
    assert webp_dims == (1200, 1004), webp_dims
    jpg_demo = isolated_env.tmp / "dim_probe.jpg"
    Image.new("RGB", (321, 213), (90, 120, 200)).save(jpg_demo, format="JPEG")
    assert backend_studio._read_image_dimensions(jpg_demo) == (321, 213)

    # 校验待机停机态 (proud) 也允许自然眨眼，且文件选择框优先绑定当前可见的 Studio 窗口
    ui_txt = (PET_DIR / "ui.html").read_text(encoding="utf-8")
    assert 'mood === "idle" || mood === "proud"' in ui_txt

    dlg_backend = _make_backend(isolated_env, "dlg_route")
    dlg_api = pet_app.PetBridgeApi(dlg_backend, threading.Event())
    fake_pet_win = SimpleNamespace(native=SimpleNamespace(Handle=SimpleNamespace(ToInt64=lambda: 101)))
    fake_studio_win = SimpleNamespace(native=SimpleNamespace(Handle=SimpleNamespace(ToInt64=lambda: 202)))
    dlg_api._bind_window(fake_pet_win)
    assert dlg_api._resolve_dialog_window() is fake_pet_win
    dlg_api._studio_window = fake_studio_win
    with (
        patch.object(pet_app.user32, "IsWindow", side_effect=lambda h: int(h) == 202),
        patch.object(pet_app.user32, "IsIconic", return_value=False),
        patch.object(pet_app.user32, "GetForegroundWindow", return_value=202),
    ):
        assert dlg_api._resolve_dialog_window() is fake_studio_win
    with (
        patch.object(pet_app.user32, "IsWindow", side_effect=lambda h: int(h) == 202),
        patch.object(pet_app.user32, "IsIconic", return_value=False),
        patch.object(pet_app.user32, "GetForegroundWindow", return_value=101),
    ):
        assert dlg_api._resolve_dialog_window() is fake_pet_win


def test_host_telemetry_routes_and_idle_prefs_gate(
    isolated_env: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """校验 4 档闲置停机偏好、Comfy/SSE 网关硬件遥测端点公网 401 门禁及后端合并解析。"""
    import asyncio
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer

    import colab_pet_bridge_node as bridge_node
    import colab_sse_gateway as sse_gw
    from colab_core.runtime import sanitize_prefs_dict

    # 1) idle_auto_stop_minutes 支持 0 / 15 / 30 / 60 四档并拒绝异常值
    for valid_min in (0, 15, 30, 60):
        assert sanitize_prefs_dict({"idle_auto_stop_minutes": valid_min})["idle_auto_stop_minutes"] == valid_min
    assert sanitize_prefs_dict({"idle_auto_stop_minutes": 999})["idle_auto_stop_minutes"] == 60

    # 2) colab_pet_bridge_node /colab_pet/host_metrics 公网匿名 401 + Bearer 鉴权 200
    auth_key_file = isolated_env.tmp / "comfy_key.txt"
    auth_key_file.write_text("bridge-telemetry-key-123\n", encoding="utf-8")
    monkeypatch.setenv("COMFY_AUTH_KEY_FILE", str(auth_key_file))

    async def _check_bridge() -> None:
        app = web.Application(middlewares=[bridge_node.create_comfy_auth_middleware()])

        async def _host_metrics(_req):
            return web.json_response(bridge_node._collect_host_metrics())

        app.router.add_get("/colab_pet/host_metrics", _host_metrics)
        async with TestClient(TestServer(app)) as client:
            pub_hdrs = {"Host": "demo.trycloudflare.com", "Cf-Connecting-Ip": "203.0.113.5"}
            r_unauth = await client.get("/colab_pet/host_metrics", headers=pub_hdrs)
            assert r_unauth.status == 401
            r_ok = await client.get(
                "/colab_pet/host_metrics",
                headers={**pub_hdrs, "Authorization": "Bearer bridge-telemetry-key-123"},
            )
            assert r_ok.status == 200
            body = await r_ok.json()
            assert body["ok"] is True and body["disk_total_gb"] > 0
            for req_k in ("ram_used_gb", "ram_total_gb", "ram_free_gb", "disk_used_gb", "shm_total_gb", "drive_persisted"):
                assert req_k in body

    asyncio.run(_check_bridge())

    # 3) colab_sse_gateway /__pet_metrics 公网匿名 401 + Bearer 鉴权 200
    assert sse_gw._is_protected_api_path("/__pet_metrics") is True
    llm_key_file = isolated_env.tmp / "llm_key.txt"
    llm_key_file.write_text("sse-metrics-secret-999\n", encoding="utf-8")
    monkeypatch.setenv("SSE_API_KEY_FILE", str(llm_key_file))
    monkeypatch.setenv("SSE_REQUIRE_AUTH", "1")

    async def _check_sse_metrics() -> None:
        srv = await asyncio.start_server(sse_gw._handle_client, "127.0.0.1", 0)
        port = srv.sockets[0].getsockname()[1]
        try:
            # 匿名请求必须被 401 拒绝
            r1, w1 = await asyncio.open_connection("127.0.0.1", port)
            w1.write(b"GET /__pet_metrics HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n")
            await w1.drain()
            raw1 = await r1.read()
            w1.close()
            assert raw1.startswith(b"HTTP/1.1 401 "), raw1[:120]

            # 携带 Bearer Key 必须返回 200 JSON
            r2, w2 = await asyncio.open_connection("127.0.0.1", port)
            w2.write(
                b"GET /__pet_metrics HTTP/1.1\r\nHost: 127.0.0.1\r\n"
                b"Authorization: Bearer sse-metrics-secret-999\r\nConnection: close\r\n\r\n"
            )
            await w2.drain()
            raw2 = await r2.read()
            w2.close()
            assert raw2.startswith(b"HTTP/1.1 200 "), raw2[:120]
            payload = json.loads(raw2.split(b"\r\n\r\n", 1)[1].decode("utf-8"))
            assert payload["ok"] is True and payload["disk_total_gb"] > 0
            for req_k in ("gpu_full_name", "vram_total_gb", "ram_total_gb", "disk_total_gb", "shm_free_gb", "drive_persisted"):
                assert req_k in payload
        finally:
            srv.close()
            await srv.wait_closed()

    asyncio.run(_check_sse_metrics())

    # 4) backend_comfy::probe_comfy_status 融合 /system_stats 与 /colab_pet/host_metrics
    b = _make_backend(isolated_env, "tele_probe")

    class _MockResp:
        def __init__(self, status_code: int, data: Any):
            self.status_code = status_code
            self._data = data

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise RuntimeError(f"HTTP {self.status_code}")

        def json(self) -> Any:
            return self._data

    def _mock_comfy_get(url: str, **_kwargs: Any):
        if url.endswith("/system_stats"):
            return _MockResp(
                200,
                {
                    "system": {"ram_total": 180 * (1024 ** 3), "ram_free": 80 * (1024 ** 3)},
                    "devices": [
                        {
                            "name": "cuda:0 NVIDIA RTX PRO 6000 Blackwell Workstation Edition : cudaMallocAsync",
                            "vram_total": 95 * (1024 ** 3),
                            "vram_free": 94 * (1024 ** 3),
                        }
                    ],
                },
            )
        if url.endswith("/queue"):
            return _MockResp(200, {"queue_running": [], "queue_pending": []})
        if url.endswith("/history"):
            return _MockResp(200, {})
        if url.endswith("/colab_pet/host_metrics"):
            return _MockResp(
                200,
                {
                    "ok": True,
                    "ram_used_gb": 103.4,
                    "ram_total_gb": 176.9,
                    "ram_free_gb": 73.5,
                    "ram_pct": 58.5,
                    "disk_used_gb": 135.1,
                    "disk_total_gb": 235.7,
                    "disk_free_gb": 100.6,
                    "disk_pct": 57.3,
                    "shm_used_gb": 0.0,
                    "shm_total_gb": 88.0,
                    "shm_free_gb": 88.0,
                    "drive_persisted": True,
                    "drive_free_gb": 142.8,
                    "drive_total_gb": 200.0,
                },
            )
        return _MockResp(404, {})

    with patch.object(b, "_comfy_http_get", side_effect=_mock_comfy_get):
        ok, info, _evts = b._probe_comfy_status("https://cfy-tele-demo.trycloudflare.com")
    assert ok is True
    assert info["gpu_full_name"] == "NVIDIA RTX PRO 6000 Blackwell Workstation Edition"
    st = b.get_comfy_state()
    assert st["ram_used_gb"] == 103.4 and st["ram_total_gb"] == 176.9
    assert st["disk_used_gb"] == 135.1 and st["disk_total_gb"] == 235.7
    assert st["shm_free_gb"] == 88.0 and st["shm_total_gb"] == 88.0
    assert st["drive_persisted"] is True and st["drive_free_gb"] == 142.8

    # 5) 校验老实例 /colab_pet/host_metrics 返回 404 时触发 host-metrics 兜底采集与 30s 冷却节流
    import ast

    bridge_src = (PKG_DIR / "colab_cli_bridge.py").read_text(encoding="utf-8")
    bridge_ast = ast.parse(bridge_src)
    fn_names = {n.name for n in bridge_ast.body if isinstance(n, ast.FunctionDef)}
    assert "cmd_host_metrics" in fn_names
    dispatch_node = next(
        (n.value for n in bridge_ast.body if isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "_DISPATCH" for t in n.targets)),
        None,
    )
    assert isinstance(dispatch_node, ast.Dict)
    dispatch_keys = {ast.literal_eval(k) for k in dispatch_node.keys if k is not None}
    assert "host-metrics" in dispatch_keys

    # 在当前 Python 解释器直接实测 _HM_KERNEL_PROBE_CODE 纯标准库输出格式
    hm_code_node = next(
        (n.value for n in bridge_ast.body if isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "_HM_KERNEL_PROBE_CODE" for t in n.targets)),
        None,
    )
    hm_snippet = ast.literal_eval(hm_code_node)
    hm_proc = subprocess.run([sys.executable, "-c", hm_snippet], capture_output=True, text=True, timeout=10)
    assert hm_proc.returncode == 0, hm_proc.stderr
    hm_match = re.search(r"<<<COLAB_HM>>>(\{.*?\})<<<END_COLAB_HM>>>", hm_proc.stdout)
    assert hm_match is not None, hm_proc.stdout
    hm_parsed = json.loads(hm_match.group(1))
    assert hm_parsed["ok"] is True and hm_parsed["disk_total_gb"] > 0

    b_old = _make_backend(isolated_env, "tele_old_404")
    b_old.state["last_sessions"] = [{"name": "comfy", "endpoint": "ep-legacy-404"}]
    bridge_calls: list[list[str]] = []

    def _mock_old_get(url: str, **_kwargs: Any):
        if url.endswith("/system_stats"):
            return _MockResp(
                200,
                {
                    "system": {"ram_total": 176.9 * (1024 ** 3), "ram_free": 134.1 * (1024 ** 3)},
                    "devices": [{"name": "cuda:0 NVIDIA RTX PRO 6000 : cudaMallocAsync", "vram_total": 95 * (1024 ** 3), "vram_free": 68.8 * (1024 ** 3)}],
                },
            )
        if url.endswith("/queue"):
            return _MockResp(200, {"queue_running": [], "queue_pending": []})
        if url.endswith("/history"):
            return _MockResp(200, {})
        if url.endswith("/colab_pet/host_metrics"):
            return _MockResp(404, {})
        return _MockResp(404, {})

    def _mock_bridge_run(cmd: list[str], **_kwargs: Any):
        bridge_calls.append(list(cmd))
        payload_hm = json.dumps(
            {
                "ok": True,
                "disk_used_gb": 111.2,
                "disk_total_gb": 235.7,
                "disk_free_gb": 124.5,
                "disk_pct": 47.2,
                "shm_used_gb": 0.1,
                "shm_total_gb": 88.0,
                "shm_free_gb": 87.9,
                "drive_persisted": True,
            }
        )
        return SimpleNamespace(
            returncode=0,
            stdout=f"<<<COLAB_HOST_METRICS>>>{payload_hm}<<<END_COLAB_HOST_METRICS>>>\n",
            stderr="",
        )

    with (
        patch.object(b_old, "_comfy_http_get", side_effect=_mock_old_get),
        patch.object(cb.subprocess, "run", side_effect=_mock_bridge_run),
    ):
        ok_old, _, _ = b_old._probe_comfy_status("https://cfy-legacy-demo.trycloudflare.com")
        ok_old2, _, _ = b_old._probe_comfy_status("https://cfy-legacy-demo.trycloudflare.com")
        assert ok_old is True and ok_old2 is True
        deadline_hm = time.monotonic() + 3.0
        while time.monotonic() < deadline_hm:
            if b_old.get_comfy_state().get("disk_total_gb", 0.0) > 0:
                break
            time.sleep(0.03)
    st_old = b_old.get_comfy_state()
    assert len(bridge_calls) == 1 and "host-metrics" in bridge_calls[0], bridge_calls
    assert st_old["disk_used_gb"] == 111.2 and st_old["disk_total_gb"] == 235.7
    assert st_old["shm_free_gb"] == 87.9 and st_old["drive_persisted"] is True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dual-webview-child", default="")
    ap.add_argument("--shot-dir", default="")
    ns, _ = ap.parse_known_args()
    if ns.dual_webview_child:
        w_dir = Path(ns.dual_webview_child)
        s_dir = Path(ns.shot_dir) if ns.shot_dir else (w_dir / "screenshots")
        rc = int(_run_pywebview_dual_window_child(w_dir, s_dir))
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(rc)
    raise SystemExit(pytest.main([__file__, "-v"]))
