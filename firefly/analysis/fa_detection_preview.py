"""Single-frame manual-threshold preview using the production localization path."""
import numpy as np
from firefly.analysis.fa_localize import preprocess_and_localise_adaptive
from firefly.analysis.fa_preprocess import measure_raw_contrast
from firefly.analysis.fa_roi import apply_roi_mask


def preview_detections(frame, *, diameter=7, minmass=.45, bg_radius=10,
                       bg_method='uniform_filter', backend='auto', min_cnr=0.,
                       roi_mask=None, roi_known=True):
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
        workers=1, chunk_size=1)
    rows = measure_raw_contrast(locs, frame[None], diameter).reset_index(drop=True)
    rows['candidate_id'] = np.arange(len(rows))
    rows['passes_contrast'] = ((np.isfinite(rows.raw_cnr) & (rows.raw_cnr >= min_cnr))
                                if min_cnr > 0 else True)
    rows['inside_roi'] = None
    if roi_known:
        if roi_mask is None:
            rows['inside_roi'] = True
        else:
            if roi_mask.shape != frame.shape:
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
