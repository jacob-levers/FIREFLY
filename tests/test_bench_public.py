"""External ground-truth loader gate (firefly.bench.public.load_gt_dataset).

Round-trips a simulated dataset OUT to the external on-disk convention (a TIFF
stack + a per-localisation GT CSV with nm coordinates, à la EPFL SMLM /
SPT-Simulator) and back in, asserting the loader recovers the same stack shape
and pixel-space ground truth.  CI-safe — no network / external download.
"""
import numpy as np
import pandas as pd
import pytest

from firefly.bench.config import SimConfig, DiffusionPopulation
from firefly.bench.simulator import simulate


def test_load_gt_dataset_roundtrips_nm_csv(tmp_path):
    pytest.importorskip("tifffile")
    from firefly.bench import io as bio
    from firefly.bench.public import load_gt_dataset

    cfg = SimConfig(seed=4, n_frames=10, height=64, width=64, n_emitters=10,
                    photons_per_emitter=2000, photon_cv=0.0, bleach_prob=0.0,
                    k_on=0.4, k_off=0.05,
                    populations=(DiffusionPopulation("immobile", 1.0),))
    sim = simulate(cfg)
    px_nm = cfg.pixel_size_um * 1000.0

    tif = str(tmp_path / "stack.tif")
    bio.write_tiff(sim.stack, tif)

    # Write GT in the external nm convention (1-indexed frames, x/y in nm).
    gt = sim.gt_locs
    pd.DataFrame({
        "frame": gt["frame"].to_numpy() + 1,
        "x [nm]": gt["x"].to_numpy() * px_nm,
        "y [nm]": gt["y"].to_numpy() * px_nm,
        "intensity [photon]": gt["photons"].to_numpy(),
    }).to_csv(str(tmp_path / "gt.csv"), index=False)

    loaded = load_gt_dataset(tif, str(tmp_path / "gt.csv"),
                             pixel_size_um=cfg.pixel_size_um)

    assert loaded.stack.shape == sim.stack.shape
    assert len(loaded.gt_locs) == len(sim.gt_locs)
    assert loaded.gt_locs["frame"].min() == 0          # 1-indexed CSV → 0-indexed
    a = loaded.gt_locs.sort_values(["frame", "x"]).reset_index(drop=True)
    b = sim.gt_locs.sort_values(["frame", "x"]).reset_index(drop=True)
    assert np.allclose(a["x"], b["x"], atol=1e-3)      # nm → px round-trips
    assert np.allclose(a["y"], b["y"], atol=1e-3)


def test_load_gt_dataset_infers_pixel_size(tmp_path):
    """With no pixel_size given and nm coordinates, the loader infers nm/px from
    the field extent (emitters span ~the full frame)."""
    pytest.importorskip("tifffile")
    from firefly.bench import io as bio
    from firefly.bench.public import load_gt_dataset

    cfg = SimConfig(seed=6, n_frames=6, height=64, width=64, n_emitters=40,
                    photons_per_emitter=2000, photon_cv=0.0, bleach_prob=0.0,
                    k_on=0.6, k_off=0.04,
                    populations=(DiffusionPopulation("immobile", 1.0),))
    sim = simulate(cfg)
    px_nm = cfg.pixel_size_um * 1000.0
    tif = str(tmp_path / "stack.tif")
    bio.write_tiff(sim.stack, tif)
    pd.DataFrame({
        "frame": sim.gt_locs["frame"].to_numpy() + 1,
        "x [nm]": sim.gt_locs["x"].to_numpy() * px_nm,
        "y [nm]": sim.gt_locs["y"].to_numpy() * px_nm,
    }).to_csv(str(tmp_path / "gt.csv"), index=False)

    loaded = load_gt_dataset(tif, str(tmp_path / "gt.csv"))   # no pixel_size_um
    # inferred px should be within ~15% of the true value (depends on coverage)
    assert abs(loaded.meta["pixel_size_um"] - cfg.pixel_size_um) < 0.15 * cfg.pixel_size_um
