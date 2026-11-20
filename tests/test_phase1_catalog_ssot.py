"""Phase 1 · Wave A Step A1：默认费率与硬件规格 SSOT（`stack_manifest.json` v2 + `colab_core/catalog.py`）回归测试。

严格离线隔离执行：全部用例均在临时目录与 `COLAB_PET_MOCK_VAULT_PATH` 沙箱内运行，
绝不触碰宿主机真实凭据库或工作区运行态 JSON 文件。
"""

from __future__ import annotations

import ast
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PACKAGE_DIR = Path(__file__).resolve().parent.parent
PET_DIR = PACKAGE_DIR / "colab_pet"
for _p in (str(PACKAGE_DIR), str(PET_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import colab_backend as cb  # noqa: E402
import colab_core  # noqa: E402
from colab_core import catalog, ledger, pool_scheduler  # noqa: E402
import backend_cluster  # noqa: E402


@pytest.fixture()
def isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """为每个测试构造完全隔离的临时凭据、CF 配置与诊断日志路径。"""
    vault_file = tmp_path / "mock_vault.json"
    cf_file = tmp_path / "cf_tunnel_secret.json"
    diag_file = tmp_path / "pet_diag.log"
    monkeypatch.setenv("COLAB_PET_MOCK_VAULT_PATH", str(vault_file))
    monkeypatch.setenv("COLAB_PET_CF_CONFIG_PATH", str(cf_file))
    monkeypatch.setenv("COLAB_PET_DIAG_LOG_PATH", str(diag_file))
    return SimpleNamespace(
        tmp=tmp_path,
        vault_file=vault_file,
        cf_file=cf_file,
        diag_file=diag_file,
    )


def test_stack_manifest_v2_schema_and_gpu_specs() -> None:
    """验证 `stack_manifest.json` 升至 v2 且保留原有结构并精确定义 5 卡规格与默认费率。"""
    manifest_path = PACKAGE_DIR / "stack_manifest.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert data["schema_version"] == 2
    assert data["default_profile"] == "anima"
    assert isinstance(data.get("host_baseline"), dict)
    assert set(data.get("profiles", {}).keys()) >= {"anima", "minimax", "drama3", "qwen_27b", "qwen_llm"}
    assert data["gpu_order"] == ["T4", "L4", "A100", "G4", "H100"]

    expected_rates = {
        "T4": (1.15, 1.31, False),
        "L4": (1.94, 1.94, True),
        "A100": (5.5, 11.5, False),
        "G4": (10.05, 10.05, True),
        "H100": (12.6, 14.2, False),
    }
    specs = data["gpu_specs"]
    assert set(specs.keys()) == set(expected_rates.keys())
    required_fields = {"vram", "vram_hm", "ram", "ram_hm", "default_rate_std", "default_rate_hm", "hm_locked", "note"}
    for gpu, (exp_std, exp_hm, exp_lock) in expected_rates.items():
        entry = specs[gpu]
        assert required_fields.issubset(entry.keys())
        assert entry["default_rate_std"] == exp_std
        assert entry["default_rate_hm"] == exp_hm
        assert entry["hm_locked"] is exp_lock
        assert str(entry["note"]).strip()

    # 锁定 catalog.py 内置兜底表与 stack_manifest.json 完全一致，防止改了清单忘改兜底导致静默漂移
    assert catalog._FALLBACK_GPU_ORDER == tuple(data["gpu_order"])
    assert catalog._FALLBACK_GPU_SPECS == data["gpu_specs"]


def test_catalog_api_read_only_caching_and_frozen_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """验证 `colab_core.catalog` 全部接口、纯标准库边界、防污染深拷贝与 `_MEIPASS` 冻结路径解析。"""
    assert "catalog" in colab_core.__all__

    # 纯标准库门禁（严禁导入重依赖或循环导入）
    cat_src = (PACKAGE_DIR / "colab_core" / "catalog.py").read_text(encoding="utf-8")
    tree = ast.parse(cat_src)
    forbidden = {"PIL", "aiohttp", "pywin32", "requests", "colab_pet"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] not in forbidden
        elif isinstance(node, ast.ImportFrom) and node.module:
            assert node.module.split(".")[0] not in forbidden

    # 默认目录接口行为
    assert catalog.get_gpu_order() == ("T4", "L4", "A100", "G4", "H100")
    specs = catalog.get_gpu_specs()
    assert specs["A100"]["default_rate_std"] == 5.5
    assert specs["A100"]["default_rate_hm"] == 11.5
    assert specs["H100"]["default_rate_std"] == 12.6
    assert specs["H100"]["default_rate_hm"] == 14.2

    # 修改返回副本不得污染 lru_cache
    specs["A100"]["default_rate_std"] = 999.0
    assert catalog.get_gpu_specs()["A100"]["default_rate_std"] == 5.5

    # hm_locked 显卡强制归并至标准档
    assert catalog.get_fallback_gpu_rate("L4", high_mem=True) == 1.94
    assert catalog.get_fallback_gpu_rate("G4", high_mem=True) == 10.05
    assert catalog.get_fallback_gpu_rate("A100", high_mem=False) == 5.5
    assert catalog.get_fallback_gpu_rate("A100", high_mem=True) == 11.5
    assert catalog.get_fallback_gpu_rate("H100", high_mem=True) == 14.2
    assert catalog.get_fallback_gpu_rate("UNKNOWN_GPU") == 0.0

    # Profile 目录与前端载荷导出
    profs = catalog.get_profile_catalog()
    assert "anima" in profs and profs["anima"]["recommended_gpu"] == "L4"
    front = catalog.export_frontend_catalog()
    assert front["schema_version"] == 2
    assert front["gpu_order"] == ["T4", "L4", "A100", "G4", "H100"]
    assert front["gpu_specs"]["H100"]["default_rate_hm"] == 14.2

    # 模拟 PyInstaller _MEIPASS 冻结态路径解析
    fake_meipass = tmp_path / "frozen_meipass"
    fake_meipass.mkdir()
    custom_manifest = {
        "schema_version": 2,
        "default_profile": "anima",
        "gpu_order": ["T4", "A100"],
        "gpu_specs": {
            "T4": {"vram": "16G", "vram_hm": "16G高", "ram": "13G", "ram_hm": "51G", "default_rate_std": 1.25, "default_rate_hm": 1.45, "hm_locked": False, "note": "冻测"},
            "A100": {"vram": "40G", "vram_hm": "80G", "ram": "83G", "ram_hm": "167G", "default_rate_std": 6.1, "default_rate_hm": 11.9, "hm_locked": False, "note": "冻测"},
        },
        "profiles": {},
    }
    (fake_meipass / "stack_manifest.json").write_text(json.dumps(custom_manifest, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(fake_meipass), raising=False)
    catalog.clear_catalog_cache()
    try:
        assert catalog.get_gpu_order() == ("T4", "A100")
        assert catalog.get_fallback_gpu_rate("A100", high_mem=False) == 6.1
        assert catalog.get_fallback_gpu_rate("A100", high_mem=True) == 11.9
    finally:
        catalog.clear_catalog_cache()


def test_backend_rate_ssot_eliminates_a100_h100_drift(isolated_env: SimpleNamespace) -> None:
    """验证 ledger / pool_scheduler / backend_cluster 全链路费率统一来自清单，彻底消除 A100/H100 漂移。"""
    # 1. ledger.DEFAULT_MEASURED_GPU_RATES 与 catalog 派生完全一致且保留历史 3 键
    assert ledger.DEFAULT_MEASURED_GPU_RATES == catalog.get_default_measured_gpu_rates()
    assert ledger.DEFAULT_MEASURED_GPU_RATES == {"T4:STD": 1.15, "L4:STD": 1.94, "G4:STD": 10.05}

    # 2. pool_scheduler._FALLBACK_GPU_RATES 与 catalog 完全一致
    assert pool_scheduler._FALLBACK_GPU_RATES == catalog.get_fallback_gpu_rate_map()
    assert pool_scheduler._estimate_gpu_rate("A100", high_mem=False) == 5.5
    assert pool_scheduler._estimate_gpu_rate("A100", high_mem=True) == 11.5
    assert pool_scheduler._estimate_gpu_rate("H100", high_mem=False) == 12.6
    assert pool_scheduler._estimate_gpu_rate("H100", high_mem=True) == 14.2
    assert pool_scheduler._estimate_gpu_rate("T4", high_mem=True) == 1.31
    assert pool_scheduler._estimate_gpu_rate("L4", high_mem=True) == 1.94

    # 3. backend_cluster 已删除旧漂移表 _SLOT_FALLBACK_RATES，并按 SSOT 推算 A100/H100 槽位费率
    assert not hasattr(backend_cluster, "_SLOT_FALLBACK_RATES")

    backend = cb.ColabPetBackend(
        state_path=isolated_env.tmp / "cluster_ssot_state.json",
        download_root=isolated_env.tmp / "dl",
        cf_config_path=isolated_env.cf_file,
    )
    with backend._store.lock:
        backend._store.state["account_profiles"] = [
            {"id": "acct_a100", "label": "A100号"},
            {"id": "acct_h100", "label": "H100号"},
        ]
        backend._store.state["cluster_slots"] = {
            "llm": {"account_id": "acct_a100", "endpoint": "ep-a100", "gpu": "A100", "high_mem": False, "online": True, "rate_hourly": 0.0},
            "comfy": {"account_id": "acct_h100", "endpoint": "ep-h100", "gpu": "H100", "high_mem": True, "online": True, "rate_hourly": 0.0},
        }
    c_info, _ = backend._cluster_svc.build_cluster_snapshot_and_enrich_onboarding(
        {"accounts": [{"id": "acct_a100"}, {"id": "acct_h100"}]},
        current_rate_hourly=0.0,
        active_account_id="acct_a100",
        comfy_state={"llm_online": True, "comfy_online": True},
    )
    assert c_info["slots"]["llm"]["rate_hourly"] == 5.5
    assert c_info["slots"]["comfy"]["rate_hourly"] == 14.2
    assert c_info["total_rate_hourly"] == round(5.5 + 14.2, 2)


def test_frontend_gpu_specs_and_catalog_sync_parity() -> None:
    """验证 `ui.html` 与 `studio.js` 首屏兜底费率与清单逐卡一致，且 `applyCatalogFromSnap` 可原地刷新。"""
    specs = catalog.get_gpu_specs()
    for fname in ("ui.html", "studio.js"):
        src = (PET_DIR / fname).read_text(encoding="utf-8")
        assert "function applyCatalogFromSnap(snap)" in src, f"{fname} 缺少 applyCatalogFromSnap"
        assert "applyCatalogFromSnap(snap);" in src, f"{fname} 未在快照入口调用 applyCatalogFromSnap"
        # 验证内联首屏常量中各卡默认费率与 SSOT 一致
        for gpu, spec in specs.items():
            pat = re.compile(
                rf"\b{gpu}\s*:\s*\{{[^}}]*?cu\s*:\s*([0-9.]+)[^}}]*?cuHm\s*:\s*([0-9.]+)[^}}]*?hmLock\s*:\s*(true|false)",
                re.S,
            )
            m = pat.search(src)
            assert m is not None, f"{fname} 未找到首屏兜底 {gpu} 规格定义"
            assert float(m.group(1)) == spec["default_rate_std"]
            assert float(m.group(2)) == spec["default_rate_hm"]
            assert (m.group(3) == "true") is spec["hm_locked"]

    # 若本机存在 node，真跑一段 JS 校验 studio.js / ui.html 的 catalog 原地刷新映射
    if shutil.which("node"):
        front_payload = json.dumps(catalog.export_frontend_catalog(), ensure_ascii=False)
        js_probe = (
            "const GPU_ORDER = ['T4'];\n"
            "const GPU_SPECS = { T4: { vram: 'old', vramHm: 'old', ram: 'old', ramHm: 'old', cu: 0.5, cuHm: 0.5, note: 'old', hmLock: false } };\n"
            "function applyCatalogFromSnap(snap) {\n"
            "  const cat = snap && snap.catalog; if (!cat || typeof cat !== 'object') return;\n"
            "  if (Array.isArray(cat.gpu_order) && cat.gpu_order.length > 0) {\n"
            "    const nextOrder = cat.gpu_order.map((g) => String(g || '').toUpperCase()).filter(Boolean);\n"
            "    if (nextOrder.length > 0) GPU_ORDER.splice(0, GPU_ORDER.length, ...nextOrder);\n"
            "  }\n"
            "  const rawSpecs = cat.gpu_specs;\n"
            "  if (rawSpecs && typeof rawSpecs === 'object') {\n"
            "    for (const [rawKey, item] of Object.entries(rawSpecs)) {\n"
            "      const g = String(rawKey || '').toUpperCase();\n"
            "      if (!g || !item || typeof item !== 'object') continue;\n"
            "      const prev = GPU_SPECS[g] || {};\n"
            "      const stdRate = Number(item.default_rate_std); const hmRate = Number(item.default_rate_hm);\n"
            "      const cuVal = Number.isFinite(stdRate) && stdRate > 0 ? stdRate : (prev.cu || 0);\n"
            "      const cuHmVal = Number.isFinite(hmRate) && hmRate > 0 ? hmRate : (cuVal || prev.cuHm || 0);\n"
            "      GPU_SPECS[g] = { vram: String(item.vram ?? prev.vram ?? ''), vramHm: String(item.vram_hm ?? prev.vramHm ?? item.vram ?? ''), ram: String(item.ram ?? prev.ram ?? ''), ramHm: String(item.ram_hm ?? prev.ramHm ?? item.ram ?? ''), cu: cuVal, cuHm: item.hm_locked ? cuVal : cuHmVal, note: String(item.note ?? prev.note ?? ''), hmLock: Boolean(item.hm_locked) };\n"
            "    }\n"
            "  }\n"
            "}\n"
            f"applyCatalogFromSnap({{ catalog: {front_payload} }});\n"
            "console.log(JSON.stringify({ order: GPU_ORDER, a100: GPU_SPECS.A100, h100: GPU_SPECS.H100 }));\n"
        )
        proc = subprocess.run(["node", "-e", js_probe], capture_output=True, text=True, encoding="utf-8", check=True, timeout=15)
        res = json.loads(proc.stdout.strip())
        assert res["order"] == ["T4", "L4", "A100", "G4", "H100"]
        assert res["a100"]["cu"] == 5.5 and res["a100"]["cuHm"] == 11.5 and res["a100"]["hmLock"] is False
        assert res["h100"]["cu"] == 12.6 and res["h100"]["cuHm"] == 14.2 and res["h100"]["note"] == "旗舰算力"


def test_custom_measured_gpu_rates_and_historical_ledgers_preserved_verbatim(
    isolated_env: SimpleNamespace,
) -> None:
    """显式写入自定义 `measured_gpu_rates` 与历史 `account_ledgers`，验证加载与 `build_snapshot` 后逐字段无损不被覆写。"""
    state_file = isolated_env.tmp / "preserve_ledger_state.json"
    custom_measured = {
        "T4:STD": 1.15,
        "L4:STD": 2.88,
        "A100:HM": 12.34,
        "G4:STD": 9.87,
    }
    historical_ledgers = {
        "acct_alpha": {
            "date": "2026-11-19",
            "day_start_balance": 198.5,
            "day_consumed_cu": 23.75,
            "day_topup_cu": 100.0,
            "last_balance": 174.75,
            "peak_balance": 250.0,
            "last_delta": -1.25,
            "baseline_balance": 200.0,
            "active_since_ts": None,
            "active_endpoint": None,
            "session_start_balance": None,
            "last_rate_hourly": 0.0,
            "last_assignments_count": 0,
            "last_sessions": [],
            "last_eligible_gpus": ["T4", "L4", "G4"],
            "last_ineligible_gpus": ["H100"],
            "last_eligible_tpus": [],
            "subscription_tier": "PRO",
            "last_updated_ts": 1763550000,
            "tunnel_cache": {},
            "managed_binding": None,
        },
        "acct_beta": {
            "date": "2026-11-20",
            "day_start_balance": 990.0,
            "day_consumed_cu": 45.2,
            "day_topup_cu": 0.0,
            "last_balance": 944.8,
            "peak_balance": 1000.0,
            "last_delta": -2.1,
            "baseline_balance": 1000.0,
            "active_since_ts": None,
            "active_endpoint": None,
            "session_start_balance": None,
            "last_rate_hourly": 0.0,
            "last_assignments_count": 0,
            "last_sessions": [],
            "last_eligible_gpus": ["T4", "L4", "A100", "G4", "H100"],
            "last_ineligible_gpus": [],
            "last_eligible_tpus": [],
            "subscription_tier": "ULTRA",
            "last_updated_ts": 1763600000,
            "tunnel_cache": {},
            "managed_binding": None,
        },
    }
    seed_state = {
        "date": "2026-11-20",
        "day_start_balance": 174.75,
        "day_consumed_cu": 11.11,
        "last_balance": 163.64,
        "peak_balance": 250.0,
        "baseline_balance": 200.0,
        "measured_gpu_rates": dict(custom_measured),
        "account_profiles": [
            {"id": "acct_alpha", "label": "甲号", "email_masked": "a***@gmail.com"},
            {"id": "acct_beta", "label": "乙号", "email_masked": "b***@gmail.com"},
        ],
        "account_ledgers": historical_ledgers,
        "prefs": {"active_account_id": "acct_alpha", "boot_gpu": "L4"},
    }
    state_file.write_text(json.dumps(seed_state, ensure_ascii=False, indent=2), encoding="utf-8")

    backend = cb.ColabPetBackend(
        state_path=state_file,
        download_root=isolated_env.tmp / "dl",
        cf_config_path=isolated_env.cf_file,
    )
    snap = backend._build_snapshot(live_data=None, stale=False, error_msg=None)

    # 1. 快照携带只读 SSOT catalog
    assert "catalog" in snap
    assert snap["catalog"]["gpu_order"] == ["T4", "L4", "A100", "G4", "H100"]
    assert snap["catalog"]["gpu_specs"]["L4"]["default_rate_std"] == 1.94
    assert snap["catalog"]["gpu_specs"]["A100"]["default_rate_hm"] == 11.5

    # 2. 自定义实测费率逐字段无损，绝不被默认值（如 L4=1.94 / G4=10.05）覆写
    for k, expected_v in custom_measured.items():
        assert backend.state["measured_gpu_rates"][k] == expected_v
        assert snap["measured_gpu_rates"][k] == expected_v

    # 3. 历史多账号账本逐字段完全无损
    for aid, orig_ledger in historical_ledgers.items():
        loaded_ledger = backend.state["account_ledgers"][aid]
        for field_k, expected_val in orig_ledger.items():
            assert loaded_ledger[field_k] == expected_val, f"历史账本 {aid}.{field_k} 被意外改动"


def test_no_broad_except_pass_in_ledger_and_cluster() -> None:
    """AST 门禁：验证 `colab_core/ledger.py` 与 `colab_pet/backend_cluster.py` 无宽泛 `except ...: pass`。"""
    for rel in ("colab_core/ledger.py", "colab_pet/backend_cluster.py"):
        fpath = PACKAGE_DIR / rel
        tree = ast.parse(fpath.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler):
                is_pass_only = len(node.body) == 1 and isinstance(node.body[0], ast.Pass)
                if is_pass_only:
                    is_broad = node.type is None or (isinstance(node.type, ast.Name) and node.type.id == "Exception")
                    assert not is_broad, f"{rel}:L{node.lineno} 仍存在宽泛 except: pass"
