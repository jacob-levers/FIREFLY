"""Known-truth translation, reference isolation, and unsupported-data checks."""
import numpy as np
import pandas as pd
import pytest
from firefly.analysis.fa_drift import (correct_drift, _segment_bounds, _solve_pairs,
                                      select_drift_reference, save_drift_diagnostic)


def stationary_movie(n=400, n_spots=35, seed=71, nonlinear=False):
    rng = np.random.default_rng(seed)
    positions = rng.uniform(8, 70, (n_spots, 2))
    frame = np.repeat(np.arange(n), n_spots)
    dx = .009*np.arange(n)
    dy = -.004*np.arange(n)
    if nonlinear: dx += .7*np.sin(np.arange(n)*2*np.pi/n)
    xy = np.tile(positions, (n, 1)) + np.column_stack((dx[frame], dy[frame]))
    xy += rng.normal(0, .03, xy.shape)
    return pd.DataFrame(dict(frame=frame, x=xy[:,0], y=xy[:,1], particle=np.tile(np.arange(n_spots), n))), np.column_stack((dx,dy))


def test_rcc_recovers_subpixel_nonlinear_translation_and_diagnostics(tmp_path):
    loc, truth = stationary_movie(nonlinear=True)
    corrected, drift = correct_drift(loc, n_seg_frames=80, min_locs=100, smooth_sigma=.3)
    assert drift.attrs['diagnostics']['status'] == 'applied'
    truth -= truth.mean(axis=0)
    assert np.sqrt(np.mean((drift[['dx','dy']].to_numpy()-truth)**2)) < .12
    assert corrected.groupby('particle').x.std().median() < .13
    assert drift.support_pairs.min() >= 2
    save_drift_diagnostic(drift, tmp_path/'diagnostic.png', .02)
    assert (tmp_path/'diagnostic.png').stat().st_size > 1000


def test_adaptive_windows_expand_for_sparse_and_contract_for_dense():
    f = np.r_[np.repeat(np.arange(400), 10), np.arange(400,1000)]
    bounds = _segment_bounds(f, 1000, 100, True, 200)
    widths = np.diff(bounds)
    assert np.median(widths[bounds[:-1] < 300]) < np.median(widths[bounds[:-1] >= 400])
    assert bounds[0] == 0 and bounds[-1] == 1000
    assert widths.max() <= 400 and len(widths) <= 64


def test_unrelated_noise_does_not_receive_a_drift_curve():
    rng = np.random.default_rng(34)
    loc = pd.DataFrame(dict(frame=np.repeat(np.arange(300),2), x=rng.uniform(0,100,600),y=rng.uniform(0,100,600)))
    corrected, drift = correct_drift(loc, n_seg_frames=50, adaptive=False, min_locs=30)
    assert drift.attrs['diagnostics']['status'] == 'skipped'
    np.testing.assert_array_equal(corrected[['x','y']],loc[['x','y']])
    assert (drift[['dx','dy']].to_numpy() == 0).all()


def test_missing_time_segment_is_not_silently_interpolated():
    loc, _ = stationary_movie()
    loc = loc[~loc.frame.between(100,199)]
    _, drift = correct_drift(loc, n_seg_frames=100, adaptive=False, min_locs=100)
    assert drift.attrs['diagnostics']['status'] == 'skipped'
    assert drift.support_pairs.eq(0).all()


def test_graph_rejects_gross_inconsistent_pair_and_disconnected_islands():
    pairs = [dict(i=i,j=j,dx=float(j-i),dy=0.,weight=1.,accepted=True,reason='accepted')
             for i in range(7) for j in range(i+1,7)]
    pairs[3]['dx'] += 30
    estimate, supported, _ = _solve_pairs(7,pairs,.5,6)
    assert supported and not pairs[3]['accepted']
    np.testing.assert_allclose(estimate[:,0],np.arange(7),atol=1e-8)
    islands = [p for p in pairs if (p['i']<3)==(p['j']<3)]
    assert _solve_pairs(7,islands,.5,3)[1] is False


def test_stationary_reference_preserves_biological_translation():
    reference, truth = stationary_movie(n_spots=5)
    sample = reference.copy()
    sample['x'] += sample.frame*.025
    corrected, drift = correct_drift(sample, reference_locs=reference, method='fiducials',
                                     min_locs=30, n_seg_frames=60, smooth_sigma=0)
    assert drift.attrs['diagnostics']['status']=='applied'
    slope=np.polyfit(corrected.frame,corrected.x,1)[0]
    assert slope==pytest.approx(.025,abs=.0002)
    assert np.sqrt(np.mean((drift.dx-(truth[:,0]-truth[:,0].mean()))**2)) < .04


def test_fiducial_disagreement_is_not_applied():
    loc, _ = stationary_movie(n_spots=3)
    loc['x'] += loc.frame * loc.particle * .2
    _, drift = correct_drift(loc, reference_locs=loc, method='fiducials', adaptive=False,
                             min_locs=20,n_seg_frames=80)
    assert drift.attrs['diagnostics']['status']=='skipped'


def test_rectangle_is_independent_of_analysis_roi_and_csv_provenance(tmp_path):
    full, _ = stationary_movie()
    selected, source = select_drift_reference(full,full.iloc[:0],
        dict(drift_reference='Separate rectangle',drift_ref_rect=[0,0,100,100]))
    assert len(selected)==len(full) and source['rectangle_px']==[0,0,100,100]
    path=tmp_path/'movie_reference.csv';full.to_csv(path,index=False)
    selected, source = select_drift_reference(full,full.iloc[:0],
        dict(file=str(tmp_path/'movie.tif'),drift_reference='Reference CSV',drift_reference_file='{stem}_reference.csv'))
    assert len(selected)==len(full) and len(source['sha256'])==64
    with pytest.raises(ValueError,match='separate stationary'):
        select_drift_reference(full,full,dict(drift_method='fiducials'))


def test_empty_reference_and_invalid_frames():
    loc, _ = stationary_movie()
    _, drift=correct_drift(loc,reference_locs=loc.iloc[:0])
    assert drift.attrs['diagnostics']['status']=='skipped'
    loc.loc[0,'frame']=-1
    with pytest.raises(ValueError,match='nonnegative integer'):
        correct_drift(loc)


def test_single_bridge_between_supported_groups_is_not_enough():
    from firefly.analysis.fa_drift import _redundantly_connected
    pairs=[dict(i=i,j=j) for i,j in [(0,1),(1,2),(0,2),(2,3),(3,4),(4,5),(3,5)]]
    assert not _redundantly_connected(6,pairs)
    assert _redundantly_connected(6,pairs+[dict(i=1,j=4)])


def test_fiducial_detections_without_ids_are_linked():
    reference, _ = stationary_movie(n=120,n_spots=5)
    _, drift = correct_drift(reference.drop(columns='particle'),method='fiducials',
        n_seg_frames=30,adaptive=False,min_locs=20,min_fiducials=3)
    assert drift.attrs['diagnostics']['status']=='applied'
