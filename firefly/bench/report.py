"""Score tool runs against ground truth and render a comparison table + figure.

Strictly pyplot-free / Qt-free: uses matplotlib's OO Agg API (`Figure` +
`FigureCanvasAgg`).
"""
from __future__ import annotations

import io as _io

import numpy as np
import pandas as pd

from firefly.bench.metrics import (detection_metrics, tracking_isbi,
                                    diffusion_recovery, crlb_sigma_nm)

_TABLE_COLS = ["tool", "f1", "jaccard", "precision", "recall", "rmse_nm",
               "jsc", "jsc_theta", "isbi_alpha", "isbi_beta", "isbi_rmse_nm",
               "n_tracks", "n_matched"]


def evaluate(tool_result, sim_result, *, tol_px=None, gate_px=None) -> dict:
    """Run every metric for one tool against the simulated ground truth."""
    meta = sim_result.meta
    px = meta["pixel_size_um"]
    sigma = meta["psf_sigma_px"]
    if tol_px is None:
        tol_px = max(2.0 * sigma, 250.0 / (px * 1000.0))   # ≥2σ or 250 nm
    if gate_px is None:
        gate_px = tol_px
    det = detection_metrics(tool_result.locs, sim_result.gt_locs, tol_px, px)
    trk = tracking_isbi(tool_result.tracks, sim_result.gt_tracks, gate_px, px)
    dif = diffusion_recovery(tool_result.diff, sim_result.gt_tracks,
                             trk.get("pairs", []), meta)
    return dict(
        tool=tool_result.name,
        f1=det["f1"], jaccard=det["jaccard"], precision=det["precision"],
        recall=det["recall"], rmse_nm=det["rmse_nm"], n_matched=det["n_matched"],
        jsc=trk["jsc"], jsc_theta=trk["jsc_theta"], isbi_alpha=trk["alpha"],
        isbi_beta=trk["beta"], isbi_rmse_nm=trk["rmse_nm"],
        n_tracks=trk["n_est_tracks"],
        _detail=dict(detection=det, tracking=trk, diffusion=dif),
    )


def build_report_table(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([{k: r.get(k) for k in _TABLE_COLS} for r in rows])


def _bar(ax, labels, series: dict, title, ylim=None):
    n = len(series)
    x = np.arange(len(labels))
    w = 0.8 / max(n, 1)
    for i, (name, vals) in enumerate(series.items()):
        ax.bar(x + i * w - 0.4 + w / 2, vals, w, label=name)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_title(title, fontsize=10)
    if ylim:
        ax.set_ylim(*ylim)
    if n > 1:
        ax.legend(fontsize=7, frameon=False)


def render_report_figure(rows: list[dict], sim_result, out_path: str | None = None) -> bytes:
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    tools = [r["tool"] for r in rows]
    fig = Figure(figsize=(14, 9)); FigureCanvasAgg(fig)
    fig.suptitle("FIREFLY benchmark vs ground truth", fontsize=13)
    axes = fig.subplots(2, 3)

    _bar(axes[0, 0], tools,
         {"F1": [r["f1"] for r in rows], "Jaccard": [r["jaccard"] for r in rows]},
         "Detection F1 / Jaccard", ylim=(0, 1.05))
    _bar(axes[0, 1], tools,
         {"JSC": [r["jsc"] for r in rows], "JSCθ": [r["jsc_theta"] for r in rows]},
         "Tracking JSC / JSCθ", ylim=(0, 1.05))
    _bar(axes[0, 2], tools,
         {"α": [r["isbi_alpha"] for r in rows], "β": [r["isbi_beta"] for r in rows]},
         "ISBI tracking α / β", ylim=(0, 1.05))

    # loc RMSE vs CRLB reference
    ax = axes[1, 0]
    meta = sim_result.meta
    crlb = crlb_sigma_nm(meta["photons_per_emitter"], meta["bg_photons"],
                         meta["psf_sigma_px"], meta["pixel_size_um"])
    ax.bar(tools, [r["rmse_nm"] for r in rows], color="#4477aa")
    if np.isfinite(crlb):
        ax.axhline(crlb, color="#cc3311", ls="--", lw=1.2, label=f"CRLB ≈ {crlb:.1f} nm")
        ax.legend(fontsize=8, frameon=False)
    ax.set_title("Localisation RMSE (nm)", fontsize=10)
    ax.tick_params(axis="x", labelsize=8)

    # recovered vs true D (first tool's detail)
    ax = axes[1, 1]
    det = rows[0]["_detail"]["diffusion"]["per_population"]
    if det:
        names = list(det.keys())
        true = [det[n]["D_true"] for n in names]
        est = [det[n]["D_est_median"] for n in names]
        ax.scatter(true, est, s=40)
        for n, t, e in zip(names, true, est):
            ax.annotate(n, (t, e), fontsize=7)
        lim = max([v for v in true + est if np.isfinite(v)] + [0.01]) * 1.2
        ax.plot([0, lim], [0, lim], "k--", lw=0.8)
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("true D (µm²/s)", fontsize=8)
    ax.set_ylabel("recovered D", fontsize=8)
    ax.set_title(f"Diffusion recovery — {rows[0]['tool']}", fontsize=10)

    # detection precision–recall points
    ax = axes[1, 2]
    for r in rows:
        ax.scatter(r["recall"], r["precision"], s=50, label=r["tool"])
    ax.set_xlim(0, 1.05); ax.set_ylim(0, 1.05)
    ax.set_xlabel("recall", fontsize=8); ax.set_ylabel("precision", fontsize=8)
    ax.set_title("Detection precision–recall", fontsize=10)
    ax.legend(fontsize=7, frameon=False)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    buf = _io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    data = buf.getvalue()
    if out_path:
        with open(out_path, "wb") as fh:
            fh.write(data)
    return data
