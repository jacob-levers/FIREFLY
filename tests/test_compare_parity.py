"""Golden parity guardrail for the compute/draw refactor of `compare_groups`.

The Analysis tab's live figure and the exported report both come from
``compare_groups``; the whole point of the refactor is that live == report.  This
test freezes the *numeric* output — per-replicate ``summary_df`` and the per-metric
``stats`` dict — for two fixed designs (one-way and two-factor) and asserts it never
changes.  The refactor (extracting ``compute_report`` / ``render_report``) must be
behaviour-identical, so this must stay green through it.

The effect-size CIs bootstrap with a FIXED ``seed=0``, so the values are
deterministic.  NOTE: the Phase-C CI vectorisation will legitimately change the
bootstrapped ``*_ci`` bounds (different resample pattern) — regenerate the golden
then, after confirming point estimates + p-values are unchanged.

Regenerate the fixture with:
    ./sptpalm-env/bin/python -m tests.capture_compare_golden   # (see __main__ below)
or simply delete tests/fixtures/compare_golden.json and run this test once.
"""
import json
import math
import os

import numpy as np
import pytest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from firefly.analysis.fa_compare import compare_groups
from firefly.ui.controllers.workspace import workspace_data as wd
from test_workspace_data import make_run_folder

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "compare_golden.json")


def _canon(o):
    """Recursively canonicalise to a JSON-safe, order-stable, float-rounded form."""
    if isinstance(o, dict):
        return {str(k): _canon(o[k]) for k in sorted(o, key=str)}
    if isinstance(o, (list, tuple)):
        return [_canon(x) for x in o]
    if isinstance(o, np.generic):
        o = o.item()
    if isinstance(o, bool):
        return o
    if isinstance(o, float):
        if not math.isfinite(o):
            return "nan" if math.isnan(o) else ("inf" if o > 0 else "-inf")
        return round(o, 6)
    return o


def _digest(groups):
    panels = set(wd.PANEL_KEYS)
    fig, sdf, stats = compare_groups(groups, output_dir=None, panels=panels,
                                     pdf_report=False, theme="Dark")
    plt.close(fig)
    # Drop the absolute-path column (tmp dir differs per run); stem/cell are stable.
    cols = [c for c in sdf.columns if c != "folder"]
    sdf = (sdf[cols].sort_values([c for c in ("group", "timepoint", "stem") if c in cols])
           .reset_index(drop=True))
    return {"summary": _canon(sdf.to_dict("records")), "stats": _canon(stats)}


def _build_oneway(tmp):
    g0 = [make_run_folder(tmp, f"c{k}", seed=k, d_centre=0.06) for k in range(3)]
    g1 = [make_run_folder(tmp, f"d{k}", seed=9 + k, d_centre=0.30) for k in range(3)]
    return [{"folders": g0, "label": "Ctrl", "color": "#58a6ff"},
            {"folders": g1, "label": "Drug", "color": "#f78166"}]


def _build_twofactor(tmp):
    out = []
    for gi, (name, col) in enumerate([("DMSO", "#58a6ff"), ("KCl", "#f78166")]):
        for tp, seedbase, d in [("pre", gi * 10, 0.10), ("post", gi * 10 + 5, 0.22)]:
            folders = [make_run_folder(tmp, f"{name}_{tp}{k}", seed=seedbase + k, d_centre=d)
                       for k in range(3)]
            out.append({"folders": folders, "label": name, "color": col, "timepoint": tp})
    return out


_BUILDERS = {"oneway": _build_oneway, "twofactor": _build_twofactor}


def _load_golden():
    if not os.path.exists(FIXTURE):
        return None
    with open(FIXTURE, encoding="utf-8") as fh:
        return json.load(fh)


def _save_golden(data):
    os.makedirs(os.path.dirname(FIXTURE), exist_ok=True)
    with open(FIXTURE, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=1, sort_keys=True)


@pytest.mark.parametrize("case", list(_BUILDERS))
def test_compare_output_matches_golden(tmp_path, case):
    dig = _digest(_BUILDERS[case](str(tmp_path)))
    golden = _load_golden()
    if golden is None or case not in golden:
        # First run (or a newly added case) captures the fixture, then passes.
        merged = dict(golden or {}); merged[case] = dig
        _save_golden(merged)
        pytest.skip(f"golden fixture captured for '{case}' — commit "
                    f"tests/fixtures/compare_golden.json")
    assert dig == golden[case], (
        f"compare_groups numeric output changed for '{case}'. If this is the "
        f"Phase-C CI vectorisation, confirm only *_ci bounds moved (within "
        f"Monte-Carlo tolerance) and regenerate the fixture.")


if __name__ == "__main__":   # capture both goldens fresh
    import tempfile
    data = {}
    for name, build in _BUILDERS.items():
        data[name] = _digest(build(tempfile.mkdtemp(prefix=f"golden_{name}_")))
    _save_golden(data)
    print(f"wrote {FIXTURE}")
