"""HYPER-FLY's planner must honour the "HF max cores" cap.

The concurrency was sized from ALL of the machine's cores and only then divided
by the cap, so on a 128-core box a 16-core cap ran 32 files × 1 core — twice
the cap, and a quarter of the per-file floor (MIN_CORES_PER_FILE).
"""
import pytest

import firefly.analysis.fa_hyperfly as hf
import firefly.analysis.fa_memory as mem


@pytest.fixture
def falcon(monkeypatch):
    """A 128-core / 752 GB box with plenty of RAM free and 8 GB/file."""
    monkeypatch.setattr(hf, "N_CPUS", 128)
    monkeypatch.setattr(hf, "_total_ram_gb", lambda: 752.0)
    monkeypatch.setattr(hf, "_free_ram_gb", lambda: 700.0)
    monkeypatch.setattr(hf, "_per_file_peak_gb", lambda p: 8.0)
    monkeypatch.setattr(mem, "_user_ram_reserve_gb", lambda: 8.0)
    monkeypatch.setenv("FIREFLY_HYPERFLY", "auto")
    for k in ("MAX_FILES", "MAX_RAM_GB", "GPU_SLOTS"):
        monkeypatch.setenv(f"FIREFLY_HYPERFLY_{k}", "0")
    return [{"file": f"f{i}.tif"} for i in range(32)]


@pytest.mark.parametrize("cap", [0, 64, 16, 12])
def test_core_cap_is_honoured(falcon, monkeypatch, cap):
    monkeypatch.setenv("FIREFLY_HYPERFLY_MAX_CORES", str(cap))
    plan = hf.plan_concurrency(falcon)
    assert plan["active"]
    used = plan["n_concurrent"] * plan["per_file_workers"]
    assert used <= (cap or 128)
    assert plan["per_file_workers"] >= hf.MIN_CORES_PER_FILE


def test_a_cap_too_small_to_split_runs_serially_and_says_why(falcon, monkeypatch):
    monkeypatch.setenv("FIREFLY_HYPERFLY_MAX_CORES", "6")
    plan = hf.plan_concurrency(falcon)
    assert not plan["active"]
    assert "core cap" in plan["reason"]
