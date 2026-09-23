"""Known-truth regressions from the MB543B scientific audit."""
import contextlib
import io
import numpy as np
import pandas as pd
import pytest
from firefly.analysis.fa_diffusion import _msd_and_fit_one, compute_jdd
from firefly.analysis.fa_linking import link_trajectories
from firefly.analysis.fa_preprocess import filter_raw_contrast, _preprocess_fast


def fit(xy):
    return _msd_and_fit_one(xy, np.arange(len(xy)), 0, np.arange(1, 11)*.02, 10, 4)[2]


def test_ballistic_is_not_immobile_and_generalized_coefficient_is_separate():
    xy = np.column_stack((np.arange(100)*.02, np.zeros(100)))
    row = fit(xy)
    assert row['motion'] == 'Directed'
    assert row['alpha'] == pytest.approx(2, abs=.001)
    assert row['K_alpha'] == pytest.approx(.25, rel=.01)
    # Linear fit to t² over .02,.04,.06,.08 gives slope=.10, D=.025.
    assert row['D'] == pytest.approx(.025)
    assert row['MSD0'] < 0
    assert np.isnan(row['loc_sigma_nm'])


def test_no_silent_track_length_gate_on_the_motion_classes():
    """A blanket "fewer than 20 observations -> Unknown" rule emptied the panel:
    the median real track is 12 points, so ~80% of tracks lost their class. A
    short track's alpha IS less reliable, but that is a documented caveat, not
    grounds for discarding the classification silently."""
    row = fit(np.column_stack((np.arange(8)*.02, np.zeros(8))))
    assert row['motion'] != 'Unknown'
    assert row['alpha_fit_status'] != 'short_track'


def test_a_track_that_never_leaves_its_noise_floor_is_immobile_not_unknown():
    """alpha is unidentifiable when the dynamic rise vanishes into the static
    offset -- but that is the definition of not moving, and the DISPLACEMENT
    says so without needing an exponent."""
    # 200 points so the flat MSD is estimated precisely enough that the fit
    # cannot mistake sampling scatter for a rise.
    rng = np.random.default_rng(4)
    row = fit(rng.normal(0, .002, (200, 2)))
    assert not np.isfinite(row['alpha'])
    assert row['alpha_fit_status'] == 'offset_dominated'
    assert row['motion'] == 'Immobile'


def test_blurred_brownian_linear_D_recovers_known_ensemble_mean():
    rng = np.random.default_rng(81)
    estimates = []
    for _ in range(80):
        increments = rng.normal(0, np.sqrt(2*.05*.02/32), (400*32,2))
        xy = increments.cumsum(axis=0).reshape(400,32,2).mean(axis=1)
        row = fit(xy)
        estimates.append(row['D_linear_raw'])
        assert np.isnan(row['loc_sigma_nm'])
    assert np.mean(estimates) == pytest.approx(.05, rel=.06)


def test_empty_frames_respect_memory_and_nondefault_indices():
    loc = pd.DataFrame({'frame':[0,1,2,100,101,102], 'x':20.,'y':20.}, index=np.arange(6)*3)
    out = link_trajectories(loc, search_range=3, memory=5, min_len=2)
    assert out.particle.nunique() == 2
    assert out.groupby('particle').frame.apply(lambda f: f.max()-f.min()).max() == 2


def test_one_emitter_plateau_produces_one_track_and_preview_per_frame():
    pytest.importorskip('torch')
    from firefly.analysis.fa_localize_backends import TorchBackend
    y,x = np.mgrid[:64,:64]
    pp = _preprocess_fast(np.exp(-((x-31.5)**2+(y-31.)**2)/(2*1.3**2)),10)
    previews=[]
    impl=TorchBackend();impl._forced_device='cpu'
    loc=impl.localise(np.repeat(pp[None],12,axis=0), diameter=7,minmass=.45,
                      workers=1,characterize=True,preview_cb=lambda f,img,x,y,n:previews.append(len(x)))
    assert len(loc)==12
    assert loc.groupby('frame').size().eq(1).all()
    assert all(n==1 for n in previews)
    assert link_trajectories(loc,search_range=3,memory=1,min_len=8).particle.nunique()==1


def test_raw_contrast_rejects_noise_and_is_invariant_to_intensity_rescaling():
    rng=np.random.default_rng(71)
    raw=rng.normal(2000,10,(1,64,64))
    y,x=np.mgrid[:64,:64]
    raw[0]+=150*np.exp(-((x-20)**2+(y-20)**2)/(2*1.3**2))
    loc=pd.DataFrame({'frame':[0,0], 'x':[20.,45.], 'y':[20.,45.]})
    out=filter_raw_contrast(loc,raw,7,3.)
    scaled=filter_raw_contrast(loc,raw*4+1000,7,3.)
    assert out.x.tolist()==[20.]
    np.testing.assert_allclose(out.raw_cnr,scaled.raw_cnr)
    assert len(filter_raw_contrast(loc,raw,7,0))==2


def test_jdd_does_not_claim_precision_without_calibration():
    rng=np.random.default_rng(53)
    xy=rng.normal(0,.1,(1000,2)).cumsum(axis=0)
    tr=pd.DataFrame({'frame':np.arange(1000),'particle':0,'x':xy[:,0],'y':xy[:,1]})
    jdd=compute_jdd(tr,1.,.02,n_components=1)
    assert jdd['diffusion_interpretation']=='apparent_uncorrected'
    assert np.isnan(jdd['sigma_loc_um'])


def test_schema_three_is_not_pooled_with_previous_diffusion_estimator():
    from firefly.analysis.fa_compare import _comparison_metric_contracts
    warnings,_,_=_comparison_metric_contracts([[{'metrics_schema_version':2}], [{'metrics_schema_version':3}]])
    assert 'diffusion' in warnings


def test_builtins_reset_all_scientific_controls():
    import json
    from pathlib import Path
    from firefly.ui.controllers.params.sidebar_schema import FIELDS
    required={k for f in FIELDS if f['key'].startswith('analysis/') for k in (f['key'],f.get('key2')) if k}
    for p in (Path(__file__).parents[1]/'firefly/ui/presets').glob('*.json'):
        state=json.loads(p.read_text())
        assert required <= state.keys()
        # The raw-contrast gate ships OFF: at 3 it removes ~half of control
        # detections and ~99% of dense ones, and discards everything within
        # `diameter` px of the frame edge. It is a per-experiment choice.
        assert state['analysis/min_cnr']==0
        # Torch-first backend selection is the shipped default.
        assert state['analysis/backend']=='Auto'
        if p.stem=='Drosophila Neurons':
            assert state['analysis/minmass_mode']!='Density-matched'
            assert state['analysis/memory']==5


def test_constant_drift_is_not_flattened_by_segment_smoothing():
    from firefly.analysis.fa_drift import correct_drift
    rng = np.random.default_rng(54)
    emitters = rng.uniform(5, 50, (50, 2))
    frames = np.repeat(np.arange(1600), len(emitters))
    xy = np.tile(emitters, (1600, 1))
    xy[:, 0] += frames*.002
    loc = pd.DataFrame({'frame':frames,'x':xy[:,0],'y':xy[:,1]})
    corrected, drift = correct_drift(loc, n_seg_frames=400)
    centroids = corrected.groupby('frame').x.mean()
    assert abs(np.polyfit(np.arange(1600), centroids, 1)[0]) < .0002
