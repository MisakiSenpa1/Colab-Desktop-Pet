"""Phase 1 基础设施与本地安全边界测试（`conftest.py` 全局隔离 + `asset_server.py` Host 防重绑定）。"""

from __future__ import annotations

import ast
import os
import socket
import sys
from pathlib import Path

import pytest

PKG_DIR = Path(__file__).resolve().parents[1]
PET_DIR = PKG_DIR / "colab_pet"
for _p in (str(PKG_DIR), str(PET_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import asset_server  # noqa: E402
import colab_security as csec  # noqa: E402
from colab_core.exec_runner import resolve_diag_log_path  # noqa: E402


def _send_raw_http_request(port: int, raw_request: bytes) -> tuple[int, bytes]:
    """通过原始 TCP Socket 发送 HTTP 请求，以精确覆盖缺失 Host / 多 Host / 伪造 Host 等边界。"""
    with socket.create_connection(("127.0.0.1", port), timeout=5.0) as sock:
        sock.sendall(raw_request)
        chunks: list[bytes] = []
        while True:
            buf = sock.recv(4096)
            if not buf:
                break
            chunks.append(buf)
    raw_resp = b"".join(chunks)
    header_part, _, body_part = raw_resp.partition(b"\r\n\r\n")
    status_line = header_part.split(b"\r\n", 1)[0].decode("latin-1", errors="replace")
    parts = status_line.split(" ", 2)
    assert len(parts) >= 2 and parts[1].isdigit(), f"异常 HTTP 响应状态行: {status_line!r}"
    return int(parts[1]), body_part


def test_is_allowed_loopback_host_unit_matrix() -> None:
    """单元测试 `_is_allowed_loopback_host` 放行与拦截矩阵。"""
    port = 18765

    # 合法本地环回 Host（含大小写与 IPv6 字面量）
    for good in (
        f"127.0.0.1:{port}",
        f"localhost:{port}",
        f"LOCALHOST:{port}",
        f"[::1]:{port}",
        "127.0.0.1",
        "localhost",
        "[::1]",
    ):
        assert asset_server._is_allowed_loopback_host(good, port) is True, f"应放行合法 Host: {good!r}"

    # 空白、外部域名、后缀混淆、错误端口、畸形端口、控制字符与多值 Host 一律拦截
    bad_cases = (
        "",
        "   ",
        f"evil.com:{port}",
        f"127.0.0.1.evil.com:{port}",
        f"localhost.evil.com:{port}",
        "demo.trycloudflare.com",
        f"demo.trycloudflare.com:{port}",
        "127.0.0.1:9999",
        "localhost:9999",
        "[::1]:9999",
        "127.0.0.1:0",
        f"127.0.0.1:0{port}",
        "127.0.0.1:65536",
        "127.0.0.1:abc",
        "127.0.0.1:",
        "::1",
        f"[::ffff:127.0.0.1]:{port}",
        f"0.0.0.0:{port}",
        f"127.0.0.1:{port}, evil.com:{port}",
        f"127.0.0.1:{port}@evil.com",
        f"127.0.0.1:{port}/ui.html",
        f"127.0.0.1:{port}\r\nX-Injected: 1",
    )
    for bad in bad_cases:
        assert asset_server._is_allowed_loopback_host(bad, port) is False, f"必须拦截非法 Host: {bad!r}"


def test_asset_server_host_header_rebinding_guard_e2e(tmp_path: Path) -> None:
    """端到端校验合法环回 Host 返回 200，缺失/伪造/错端口/多 Host 在所有路由均返回 403。"""
    media_root = tmp_path / "media_box"
    media_root.mkdir(parents=True, exist_ok=True)
    (media_root / "sample.png").write_bytes(b"\x89PNG\r\n\x1a\nDEMO")

    assets_root = tmp_path / "assets_box"
    assets_root.mkdir(parents=True, exist_ok=True)
    (assets_root / "icon.webp").write_bytes(b"RIFF_TEST_WEBP")
    (assets_root / "secret.py").write_text("TOKEN = 'nope'\n", encoding="utf-8")

    httpd, port = asset_server.start_asset_http_server(media_root)
    httpd.assets_root = assets_root  # type: ignore[attr-defined]
    try:
        # 1) 合法 Host：127.0.0.1:{port} / localhost:{port} / [::1]:{port} 全部返回 200
        for valid_host in (f"127.0.0.1:{port}", f"localhost:{port}", f"[::1]:{port}"):
            st_ui, body_ui = _send_raw_http_request(
                port,
                f"GET /ui.html HTTP/1.1\r\nHost: {valid_host}\r\nConnection: close\r\n\r\n".encode("ascii"),
            )
            assert st_ui == 200 and len(body_ui) > 200, f"Host={valid_host} 访问 /ui.html 失败: {st_ui}"

            st_asset, body_asset = _send_raw_http_request(
                port,
                f"GET /assets/icon.webp HTTP/1.1\r\nHost: {valid_host}\r\nConnection: close\r\n\r\n".encode("ascii"),
            )
            assert st_asset == 200 and body_asset == b"RIFF_TEST_WEBP"

            st_media, body_media = _send_raw_http_request(
                port,
                f"GET /__media/sample.png HTTP/1.1\r\nHost: {valid_host}\r\nConnection: close\r\n\r\n".encode("ascii"),
            )
            assert st_media == 200 and body_media == b"\x89PNG\r\n\x1a\nDEMO"

        # 原有 /assets/ 白名单护栏严保不变
        st_py, _ = _send_raw_http_request(
            port,
            f"GET /assets/secret.py HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n\r\n".encode("ascii"),
        )
        assert st_py == 403
        assert asset_server._ALLOWED_ASSET_EXTS == frozenset({".webp", ".png", ".ico", ".svg", ".json"})

        # 2) 缺失 Host 头（覆盖 GET 与 HEAD 以及 /ui.html、/assets/、/__media/）一律返回 403
        for method, path in (
            ("GET", "/ui.html"),
            ("HEAD", "/ui.html"),
            ("GET", "/assets/icon.webp"),
            ("GET", "/__media/sample.png"),
        ):
            st_missing, _ = _send_raw_http_request(
                port,
                f"{method} {path} HTTP/1.1\r\nConnection: close\r\n\r\n".encode("ascii"),
            )
            assert st_missing == 403, f"缺失 Host 时 {method} {path} 未返回 403，实际={st_missing}"

        # 3) 伪造域名、前缀仿冒、错端口、外部隧道域与空 Host 一律返回 403
        wrong_port = 9999 if port != 9999 else 9998
        evil_hosts = (
            "",
            f"evil.com:{port}",
            f"127.0.0.1.evil.com:{port}",
            f"127.0.0.1:{wrong_port}",
            "demo.trycloudflare.com",
            f"demo.trycloudflare.com:{port}",
        )
        for bad_host in evil_hosts:
            for path in ("/ui.html", "/assets/icon.webp", "/__media/sample.png"):
                st_bad, _ = _send_raw_http_request(
                    port,
                    f"GET {path} HTTP/1.1\r\nHost: {bad_host}\r\nConnection: close\r\n\r\n".encode("ascii"),
                )
                assert st_bad == 403, f"非法 Host={bad_host!r} 访问 {path} 未被拦截，实际={st_bad}"

        # 4) 重复多个 Host 头（HTTP Request Smuggling / Host 混淆攻击）一律返回 403
        st_dup, _ = _send_raw_http_request(
            port,
            (
                f"GET /ui.html HTTP/1.1\r\n"
                f"Host: 127.0.0.1:{port}\r\n"
                f"Host: evil.com:{port}\r\n"
                f"Connection: close\r\n\r\n"
            ).encode("ascii"),
        )
        assert st_dup == 403
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_asset_server_remains_pure_stdlib_zero_heavy_deps() -> None:
    """确认修改后的 `asset_server.py` 依然严格仅依赖 Python 标准库。"""
    src = (PET_DIR / "asset_server.py").read_text(encoding="utf-8")
    tree = ast.parse(src, filename="asset_server.py")
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    non_stdlib = sorted(m for m in imported if m not in sys.stdlib_module_names)
    assert not non_stdlib, f"asset_server.py 引入了非标准库模块: {non_stdlib}"


def test_conftest_autouse_isolation_active_and_overridable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 `conftest.py` 自动隔离生效且落盘至 `tmp_path`，同时支持单测显式覆写与删除。"""
    assert os.environ.get("PYTHONUTF8") == "1"

    expected_vault = tmp_path / "_conftest_mock_vault.json"
    expected_diag = tmp_path / "_conftest_diag.log"
    assert Path(os.environ["COLAB_PET_MOCK_VAULT_PATH"]) == expected_vault
    assert Path(os.environ["COLAB_PET_DIAG_LOG_PATH"]) == expected_diag
    assert resolve_diag_log_path() == expected_diag

    # 验证未显式声明局部 fixture 时，写入凭据严格落在临时沙箱而非宿主机真实凭据库
    vault = csec.VaultStore()
    vault.set_secret("secret:conftest_smoke", "sandbox_only_token")
    assert expected_vault.is_file()
    assert vault.get_secret("secret:conftest_smoke") == "sandbox_only_token"

    # 验证单测仍可通过 monkeypatch 覆盖为自定义沙箱路径
    custom_vault = tmp_path / "custom_vault.json"
    monkeypatch.setenv("COLAB_PET_MOCK_VAULT_PATH", str(custom_vault))
    custom_store = csec.VaultStore()
    custom_store.set_secret("secret:override", "ok")
    assert custom_vault.is_file()
    assert custom_store.get_secret("secret:override") == "ok"


def test_sys_path_insert_upper_bound() -> None:
    """锁定非测试源码中的 `sys.path.insert` 数量上限（只减不增），防止导入耦合继续扩散。"""
    py_non_test = [
        p for p in PKG_DIR.rglob("*.py")
        if "tests" not in p.parts and "__pycache__" not in p.parts
    ]
    total = sum(p.read_text(encoding="utf-8").count("sys.path.insert") for p in py_non_test)
    assert total <= 23, f"非测试源码中 sys.path.insert 数量超上限 ({total} > 23)"
