"""Single-frame manual-threshold preview using the production localization path."""
import numpy as np
from firefly.analysis.fa_localize import preprocess_and_localise_adaptive
from firefly.analysis.fa_preprocess import measure_raw_contrast
from firefly.analysis.fa_roi import apply_roi_mask


def preview_detections(frame, *, diameter=7, minmass=.45, bg_radius=10,
                       bg_method='uniform_filter', backend='auto', min_cnr=0.,
                       roi_mask=None, roi_known=True, wavelet_threshold=None):
    """Candidates have passed the detector's mass test AND duplicate suppression.

    This never emulates a detector by filtering another detector's mass values.
    ROI unknown is explicit; no single-frame result promises final track retention.
    """
    frame = np.asarray(frame, dtype=np.float32)
    if frame.ndim != 2 or not np.isfinite(frame).all():
        raise ValueError('Preview requires one finite raw camera frame.')
    if not np.isfinite(minmass) or minmass < 0 or not np.isfinite(min_cnr) or min_cnr < 0:
        raise ValueError('Thresholds must be finite and nonnegative.')
    locs, _, _, _, _ = preprocess_and_localise_adaptive(
        frame[None], diameter=diameter, minmass=float(minmass), percentile=64,
        bg_radius=bg_radius, bg_method=bg_method, backend=backend,
        workers=1, chunk_size=1,
        **({'wavelet_threshold': float(wavelet_threshold)}
           if wavelet_threshold is not None else {}))
    rows = measure_raw_contrast(locs, frame[None], diameter).reset_index(drop=True)
    rows['candidate_id'] = np.arange(len(rows))
    return classify_candidates(rows, min_cnr=min_cnr, roi_mask=roi_mask,
                               roi_known=roi_known, shape=frame.shape,
                               backend=backend)


def classify_candidates(rows, *, min_cnr=0., roi_mask=None, roi_known=True,
                        shape=None, backend=''):
    """Label already-detected candidates against a contrast cutoff and an ROI.

    Split out from :func:`preview_detections` because editing an ROI does not
    change what was DETECTED — only whether each candidate is inside it.  The
    viewer re-labels the cached candidates instead of re-running the detector,
    so the overlay survives a polygon edit or a brush stroke rather than
    emptying and taking a second to come back.
    """
    rows = rows.copy()
    rows['passes_contrast'] = ((np.isfinite(rows.raw_cnr) & (rows.raw_cnr >= min_cnr))
                                if min_cnr > 0 else True)
    rows['inside_roi'] = None
    if roi_known:
        if roi_mask is None:
            rows['inside_roi'] = True
        else:
            if shape is not None and roi_mask.shape != tuple(shape):
                raise ValueError('ROI and raw frame have different shapes.')
            included = apply_roi_mask(rows, roi_mask).candidate_id
            rows['inside_roi'] = rows.candidate_id.isin(included)
    rows['decision'] = 'passes_detection' if roi_known else 'roi_unchecked'
    if roi_known:
        rows.loc[~rows.inside_roi.astype(bool), 'decision'] = 'outside_roi'
    rows.loc[~rows.passes_contrast, 'decision'] = 'low_contrast'
    rows.loc[~rows.passes_contrast & ~np.isfinite(rows.raw_cnr), 'decision'] = 'contrast_unavailable'
    counts = rows.decision.value_counts().to_dict()
    return rows, dict(candidates=len(rows), contrast_rejected=int((~rows.passes_contrast).sum()),
                      outside_roi=int(counts.get('outside_roi', 0)),
                      passed=int(rows.passes_contrast.sum()) if not roi_known else int(counts.get('passes_detection',0)),
                      roi_known=roi_known, backend=backend)


def below_threshold(kept, low, *, radius):
    """Candidates of a lower-threshold detection that the threshold drops.

    ``kept`` is the detection at the threshold, ``low`` the same detector at a
    lower one.  A low candidate within ``radius`` px of a kept spot is that
    spot (a lower threshold grows a spot and can nudge its centre); the rest
    are what the threshold excludes.  Kept spots stay exactly the detector's —
    nothing is emulated by filtering masses.
    """
    if not len(low) or not len(kept):
        return low.copy()
    k = kept[["x", "y"]].to_numpy(float)
    lo = low[["x", "y"]].to_numpy(float)
    d2 = ((lo[:, None, :] - k[None, :, :]) ** 2).sum(-1)
    return low[(d2.min(axis=1) > float(radius) ** 2)].copy()
