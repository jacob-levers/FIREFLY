"""Tests for the benchmark report (headless Agg figure + summary table)."""
from firefly.bench.report import build_report_table, render_report_figure


def _row():
    return dict(
        tool="FIREFLY", f1=0.95, jaccard=0.90, precision=1.0, recall=0.90,
        rmse_nm=20.0, jsc=0.85, jsc_theta=0.84, isbi_alpha=0.90, isbi_beta=0.88,
        isbi_rmse_nm=22.0, n_tracks=16, n_matched=500,
        _detail=dict(
            detection={}, tracking={},
            diffusion=dict(per_population={
                "immobile": dict(D_true=0.0, D_est_median=0.0),
                "brownian": dict(D_true=0.05, D_est_median=0.06)})),
    )


class _Sim:
    meta = dict(photons_per_emitter=2000, bg_photons=8.0,
                psf_sigma_px=0.9, pixel_size_um=0.106)


def test_build_report_table_columns():
    table = build_report_table([_row()])
    assert len(table) == 1
    for col in ("tool", "f1", "jaccard", "rmse_nm", "jsc_theta", "isbi_alpha"):
        assert col in table.columns


def test_render_report_figure_png_headless():
    png = render_report_figure([_row()], _Sim())
    assert isinstance(png, bytes) and len(png) > 1000
    assert png[:8] == b"\x89PNG\r\n\x1a\n"   # valid PNG signature
