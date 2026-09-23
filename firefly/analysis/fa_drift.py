"""Redundant cross-correlation (RCC) drift correction.

Extracted from sptpalm_analysis.py (#7); re-exported there for compatibility.
"""
from __future__ import annotations

import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from firefly.analysis.fa_constants import N_CPUS

import numpy as np
from scipy.interpolate import interp1d


def align_frames_to_drift(frames, frame_indices, drift_df):
    """Shift each frame by −drift[frame] so a projection of them is sharp and
    aligned with drift-corrected localisations.

    The figure's background is a small subset of raw movie frames; the tracks
    over it are drift-corrected, so without this the background stays smeared by
    the very drift that was removed from the tracks.  Shifting each sampled
    frame by the same per-frame drift fixes that.

    frames        : (K, H, W) array (modified in place and returned).
    frame_indices : length-K movie frame index for each frame in ``frames``.
    drift_df      : DataFrame with per-frame 'dx','dy' (px) from correct_drift.
    """
    frames = np.asarray(frames)
    if drift_df is None or frames.ndim != 3 or len(frames) == 0:
        return frames
    from scipy.ndimage import shift as _shift
    dx = np.asarray(drift_df["dx"], dtype=float)
    dy = np.asarray(drift_df["dy"], dtype=float)
    nf = len(dx)
    if nf == 0:
        return frames
    for k, fidx in enumerate(np.asarray(frame_indices)):
        fi = int(min(max(int(fidx), 0), nf - 1))
        sx, sy = float(dx[fi]), float(dy[fi])
        if abs(sx) > 1e-3 or abs(sy) > 1e-3:
            frames[k] = _shift(frames[k], (-sy, -sx), order=1, mode="nearest")
    return frames


def _validate_reference(table):
    required = {"frame", "x", "y"}
    if not required <= set(table):
        raise ValueError("Drift reference requires frame, x, y columns in camera pixels.")
    values = table[["frame", "x", "y"]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values[:, 0] < 0).any() or (values[:, 0] % 1 != 0).any():
        raise ValueError("Drift reference must contain finite coordinates and nonnegative integer frames.")


def _segment_bounds(frames, n_frames, target, adaptive, min_locs, max_segments=64):
    """Count-driven windows, bounded in time and number; never bridge empty time silently."""
    target = max(1, int(target))
    min_width = max(1, int(np.ceil(n_frames / max_segments)), target // 4 if adaptive else target)
    if not adaptive:
        return np.unique(np.r_[np.arange(0, n_frames, min_width), n_frames]).astype(int)
    max_width = max(min_width, target * 4)
    cumulative = np.r_[0, np.cumsum(np.bincount(frames, minlength=n_frames))]
    bounds = [0]
    while bounds[-1] < n_frames:
        start = bounds[-1]
        enough = int(np.searchsorted(cumulative, cumulative[start] + min_locs))
        end = min(n_frames, start + max_width, max(start + min_width, enough))
        bounds.append(end)
    # Merge a short or underpopulated terminal segment, but not across an
    # arbitrarily long missing interval.
    if len(bounds) > 2 and bounds[-1] - bounds[-3] <= max_width:
        if (bounds[-1] - bounds[-2] < min_width or
                cumulative[bounds[-1]] - cumulative[bounds[-2]] < min_locs):
            bounds.pop(-2)
    return np.asarray(bounds, dtype=int)


def _redundantly_connected(n, pairs):
    """Every segment must be connected without relying on a single bridge edge."""
    neighbours = [set() for _ in range(n)]
    for p in pairs:
        neighbours[p['i']].add(p['j']); neighbours[p['j']].add(p['i'])
    discovered = [-1]*n
    low = [0]*n
    clock = 0
    bridge = False

    def visit(node, parent):
        nonlocal clock, bridge
        discovered[node] = low[node] = clock
        clock += 1
        for other in neighbours[node]:
            if other == parent: continue
            if discovered[other] < 0:
                visit(other, node)
                low[node] = min(low[node], low[other])
                if low[other] > discovered[node]: bridge = True
            else:
                low[node] = min(low[node], discovered[other])
    visit(0, -1)
    return all(t >= 0 for t in discovered) and not bridge


def _solve_pairs(n, pairs, tolerance, outlier_k):
    """Weighted RCC graph solve; disconnected estimates must never be applied."""
    active = [p for p in pairs if p['accepted']]
    estimate = np.zeros((n, 2))
    if n < 3 or not _redundantly_connected(n, active):
        return estimate, False, 'Disconnected or nonredundant reference segments'
    # A spanning tree alone has no redundant consistency evidence.
    if len(active) < n:
        return estimate, False, 'Insufficient redundant segment pairs'
    for iteration in range(5):
        A = np.zeros((len(active), n - 1))
        b = np.array([[p['dx'], p['dy']] for p in active])
        weights = np.sqrt([p['weight'] for p in active])
        for row, p in enumerate(active):
            if p['i']: A[row, p['i']-1] = -1
            if p['j']: A[row, p['j']-1] = 1
        estimate[1:] = np.linalg.lstsq(A*weights[:, None], b*weights[:, None], rcond=None)[0]
        residual = np.array([np.linalg.norm(estimate[p['j']]-estimate[p['i']]-b[k])
                             for k, p in enumerate(active)])
        med = np.median(residual)
        threshold = max(tolerance, outlier_k*1.4826*np.median(np.abs(residual-med)))
        bad = residual > threshold
        # One gross edge can spread its error across many least-squares
        # residuals and inflate MAD. Remove the worst edge if the absolute
        # consistency limit still fails, then re-solve/check connectivity.
        if not bad.any() and residual.max() > tolerance*3:
            bad[np.argmax(residual)] = True
        for p, r in zip(active, residual): p['residual_px'] = float(r)
        if not bad.any():
            break
        # At the iteration limit, don't return a curve before re-solving the
        # rejected graph. Treat unresolved inconsistency as unsupported.
        if iteration == 4:
            return estimate, False, 'Pair rejection did not converge'
        for p, reject in zip(active, bad):
            if reject: p.update(accepted=False, reason='inconsistent_shift')
        active = [p for p in active if p['accepted']]
        if len(active) < n or not _redundantly_connected(n, active):
            return estimate, False, 'Rejected shifts leave insufficient connected support'
    if max(p['residual_px'] for p in active) > tolerance * 3:
        return estimate, False, 'Large residual disagreement between references'
    return estimate, True, 'Supported by connected redundant measurements'


def _rcc_pairs(reference, bounds, scale, max_shift_frac, min_locs, min_correlation, min_psr, stop_event):
    from scipy.fft import rfft2, irfft2, next_fast_len
    from firefly.analysis.fa_constants import _Cancelled
    xy = reference[['x', 'y']].to_numpy(float)
    f = reference.frame.to_numpy(int)
    origin = xy.min(axis=0) - 2
    extent = np.ptp(xy, axis=0) + 5
    # Bound memory independently of camera dimensions or long acquisitions.
    scale = min(float(scale), 511.0 / max(extent))
    W, H = np.maximum(16, np.ceil(extent*scale).astype(int) + 1)
    shape = (next_fast_len(2*H-1), next_fast_len(2*W-1))
    maps, norms, counts = [], [], []
    for start, end in zip(bounds[:-1], bounds[1:]):
        if stop_event is not None and stop_event.is_set(): raise _Cancelled()
        sel = (f >= start) & (f < end)
        counts.append(int(sel.sum()))
        dm = np.zeros((H, W), np.float32)
        coords = np.floor((xy[sel]-origin)*scale).astype(int)
        np.add.at(dm, (coords[:, 1], coords[:, 0]), 1)
        dm = gaussian_filter(dm, max(.5, scale*.7))
        # Remove broad density gradients so a bright diffuse region cannot
        # dominate the correlation solely by its total brightness.
        dm -= gaussian_filter(dm, max(2., scale*4))
        norms.append(float(np.linalg.norm(dm)))
        maps.append(rfft2(dm, s=shape))
    lag_y = np.fft.fftfreq(shape[0])*shape[0]
    lag_x = np.fft.fftfreq(shape[1])*shape[1]
    ry, rx = max(1, int(max_shift_frac*H)), max(1, int(max_shift_frac*W))
    iy = np.flatnonzero(np.abs(lag_y) <= ry)
    ix = np.flatnonzero(np.abs(lag_x) <= rx)
    grid_y, grid_x = np.meshgrid(lag_y[iy], lag_x[ix], indexing='ij')

    def measure(ij):
        if stop_event is not None and stop_event.is_set(): raise _Cancelled()
        i, j = ij
        p = dict(i=i, j=j, dx=0., dy=0., weight=0., correlation=0., psr=0.,
                 accepted=False, reason='sparse_segment', residual_px=None)
        if min(counts[i], counts[j]) < min_locs or min(norms[i], norms[j]) <= 0: return p
        cross = irfft2(maps[j]*maps[i].conj(), s=shape)
        window = cross[np.ix_(iy, ix)]
        py, px = np.unravel_index(np.argmax(window), window.shape)
        y, x = int(iy[py]), int(ix[px])
        ly, lx = float(grid_y[py, px]), float(grid_x[py, px])
        peak = float(cross[y, x])
        sidelobe = window[((grid_y-ly)**2+(grid_x-lx)**2) > max(2., 2*scale)**2]
        if sidelobe.size < 10: p['reason'] = 'insufficient_peak_background'; return p
        psr = (peak-float(sidelobe.mean())) / max(float(sidelobe.std()), 1e-20)
        corr = peak / (norms[i]*norms[j])
        uniqueness = peak / max(float(sidelobe.max()), 1e-20)
        # Three-point parabolic refinement, in the full wrapped correlation.
        def offset(a, b, c):
            denom = a-2*b+c
            return float(np.clip(.5*(a-c)/denom, -.5, .5)) if denom < 0 else 0.
        subx = offset(cross[y, (x-1)%shape[1]], peak, cross[y, (x+1)%shape[1]])
        suby = offset(cross[(y-1)%shape[0], x], peak, cross[(y+1)%shape[0], x])
        reason = ('search_boundary' if abs(lx) >= rx or abs(ly) >= ry else
                  'weak_peak' if corr < min_correlation or psr < min_psr else
                  'ambiguous_peak' if uniqueness < 1.05 else 'accepted')
        p.update(dx=(lx+subx)/scale, dy=(ly+suby)/scale,
                 correlation=float(corr), psr=float(psr), peak_ratio=float(uniqueness),
                 weight=max(1e-6, min(corr, 1)**2 * min(psr, 30)**2),
                 accepted=reason == 'accepted', reason=reason)
        return p

    indices = [(i, j) for i in range(len(counts)) for j in range(i+1, len(counts))]
    with ThreadPoolExecutor(max_workers=min(4, N_CPUS)) as pool:
        pairs = list(pool.map(measure, indices))
    return pairs, counts, scale


def _fiducial_pairs(reference, bounds, min_fiducials, search_range, tolerance, stop_event=None):
    """Match user-designated stationary markers; use robust common motion.

    A particle column supplies identities. Otherwise Trackpy links the selected
    reference detections. This cannot identify stationary beads biologically.
    """
    if 'particle' not in reference:
        from firefly.analysis.fa_linking import _link_via_trackpy
        reference = _link_via_trackpy(reference.sort_values('frame').copy(),
                                     search_range=search_range, memory=2, stop_event=stop_event)
    if reference.particle.isna().any() or reference.duplicated(['particle', 'frame']).any():
        raise ValueError('Fiducials require unique particle/frame observations and nonmissing identities.')
    segments, counts = [], []
    from firefly.analysis.fa_constants import _Cancelled
    for start, end in zip(bounds[:-1], bounds[1:]):
        if stop_event is not None and stop_event.is_set(): raise _Cancelled()
        sub = reference[(reference.frame >= start) & (reference.frame < end)]
        counts.append(len(sub))
        # All beads in a segment must sample the same drift epoch: require
        # observations on both sides of the window center.
        center = (start+end-1)/2
        grouped = sub.groupby('particle')
        spans = grouped.frame.agg(['min', 'max', 'count'])
        ids = spans.index[(spans['min'] <= center) & (spans['max'] >= center) & (spans['count'] >= 3)]
        # Interpolate each bead at the same segment center; medians at different
        # observation times would bias drifting, asynchronously blinking beads.
        positions = {}
        for pid, g in sub[sub.particle.isin(ids)].groupby('particle'):
            g = g.sort_values('frame')
            if np.diff(g.frame).max(initial=0) > max(2, (end-start)/4): continue
            positions[pid] = [np.interp(center, g.frame, g.x), np.interp(center, g.frame, g.y)]
        segments.append(pd.DataFrame.from_dict(positions, orient='index', columns=['x','y']))
    pairs = []
    for i in range(len(segments)):
        for j in range(i+1, len(segments)):
            ids = segments[i].index.intersection(segments[j].index)
            p = dict(i=i, j=j, dx=0., dy=0., weight=0., accepted=False,
                     reason='too_few_shared_fiducials', n_fiducials=len(ids), residual_px=None)
            if len(ids) >= min_fiducials:
                shifts = (segments[j].loc[ids]-segments[i].loc[ids]).to_numpy()
                median = np.median(shifts, axis=0)
                good = np.linalg.norm(shifts-median, axis=1) <= tolerance
                if good.sum() >= min_fiducials:
                    median = np.median(shifts[good], axis=0)
                    p.update(dx=float(median[0]), dy=float(median[1]), weight=float(good.sum()),
                             accepted=True, reason='accepted', n_fiducials=int(good.sum()))
                else: p['reason'] = 'fiducials_disagree'
            pairs.append(p)
    return pairs, counts


def correct_drift(locs, n_seg_frames=200, upsampling=4, smooth_sigma=1.5,
                  max_shift_frac=0.30, outlier_k=6.0, outlier_tol_px=6.0, *,
                  reference_locs=None, adaptive=True, min_locs=200,
                  min_correlation=.2, min_psr=6., method='rcc', min_fiducials=3,
                  fiducial_search_range=2., stop_event=None):
    """Estimate and subtract supported 2D translation in camera pixels.

    RCC uses quality-weighted redundant density correlations (Wang et al. 2014,
    doi:10.1364/OE.22.015982), bounded count-adaptive windows and subpixel peaks.
    Optional reference_locs are independent of the analysis ROI. Fiducials must
    be user-designated stationary markers; particle identities can be supplied
    or linked within this reference. No method distinguishes common biological
    motion from stage drift without such a reference.

    The returned dx/dy are the APPLIED shift. Unsupported estimates apply zero,
    carry status='skipped', and preserve the reason in drift_df.attrs['diagnostics'].
    Pair/segment quality scores are diagnostics, not calibrated uncertainties.
    outlier_tol_px retains its historic upsampled-pixel units; internally all
    solved shifts and reported residuals use camera pixels.
    """
    from firefly.analysis.fa_constants import _Cancelled
    if stop_event is not None and stop_event.is_set(): raise _Cancelled()
    _validate_reference(locs)
    reference = locs.copy() if reference_locs is None else reference_locs.copy()
    _validate_reference(reference)
    if method not in ('rcc', 'fiducials'): raise ValueError('Unknown drift method')
    if (n_seg_frames < 1 or upsampling <= 0 or min_locs < 5 or smooth_sigma < 0
            or not 0 < max_shift_frac < .5 or not 0 <= min_correlation <= 1
            or min_psr <= 0 or min_fiducials < 1 or fiducial_search_range <= 0):
        raise ValueError('Invalid drift estimation settings')
    n_frames = int(locs.frame.max())+1 if len(locs) else 1
    reference = reference[reference.frame < n_frames]
    bounds = _segment_bounds(reference.frame.to_numpy(int), n_frames, n_seg_frames, adaptive, min_locs)
    centers = (bounds[:-1]+bounds[1:]-1)/2
    n = len(centers)
    counts, pairs, scale = [0]*n, [], float(upsampling)
    estimate, supported, reason = np.zeros((n, 2)), False, 'Too few reference localisations or time segments'
    if len(reference) >= min_locs and n >= 3:
        if method == 'rcc':
            pairs, counts, scale = _rcc_pairs(reference, bounds, upsampling, max_shift_frac,
                                            min_locs, min_correlation, min_psr, stop_event)
        else:
            pairs, counts = _fiducial_pairs(reference, bounds, min_fiducials,
                                            fiducial_search_range, outlier_tol_px/upsampling, stop_event)
        estimate, supported, reason = _solve_pairs(n, pairs, outlier_tol_px/upsampling, outlier_k)
    frame_arr = np.arange(n_frames)
    applied = np.zeros((n_frames, 2))
    if supported:
        # Smoothing on a UNIFORM time grid: adaptive segments are not equally
        # spaced. Preserve a linear trend and avoid a time-dependent kernel.
        grid = np.linspace(centers[0], centers[-1], max(n, 3))
        for axis in range(2):
            values = np.interp(grid, centers, estimate[:, axis])
            trend = np.polyval(np.polyfit(grid, values, 1), grid)
            if smooth_sigma > 0: values = trend + gaussian_filter1d(values-trend, smooth_sigma)
            applied[:, axis] = interp1d(grid, values, bounds_error=False, fill_value='extrapolate')(frame_arr)
        applied -= applied.mean(axis=0)
    degree = np.zeros(n, int)
    residuals = [[] for _ in range(n)]
    for pair in pairs:
        if pair['accepted']:
            for k in (pair['i'], pair['j']):
                degree[k] += 1
                if pair.get('residual_px') is not None: residuals[k].append(pair['residual_px'])
    diagnostics = dict(version=2, method=method, status='applied' if supported else 'skipped', reason=reason,
                       settings=dict(target_segment_frames=n_seg_frames, min_locs=min_locs,
                                     requested_upsampling=upsampling, smooth_sigma=smooth_sigma,
                                     max_shift_frac=max_shift_frac, outlier_k=outlier_k,
                                     residual_floor_camera_px=outlier_tol_px/upsampling,
                                     min_fiducials=min_fiducials, fiducial_search_range=fiducial_search_range,
                                     max_segments=64, min_peak_ratio=1.05),
                       adaptive=bool(adaptive), effective_upsampling=scale, n_reference=len(reference),
                       n_segments=n, n_pairs=len(pairs), n_accepted_pairs=sum(p['accepted'] for p in pairs),
                       min_correlation=min_correlation, min_psr=min_psr,
                       confidence_interpretation='Diagnostic support, not a calibrated probability',
                       segments=[dict(start=int(bounds[i]), end=int(bounds[i+1]), center=float(centers[i]),
                                      n_locs=int(counts[i]), accepted_pairs=int(degree[i]),
                                      residual_px=float(np.median(residuals[i])) if residuals[i] else None,
                                      estimate_dx=float(estimate[i,0]), estimate_dy=float(estimate[i,1])) for i in range(n)],
                       pairs=pairs)
    segment = np.clip(np.searchsorted(bounds[1:], frame_arr, side='right'), 0, n-1)
    support = np.where(supported, degree[segment], 0)
    status = diagnostics['status']
    drift_df = pd.DataFrame(dict(frame=frame_arr, dx=applied[:,0], dy=applied[:,1],
                                 support_pairs=support, status=status,
                                 extrapolated=(frame_arr < centers[0]) | (frame_arr > centers[-1])))
    drift_df.attrs['diagnostics'] = diagnostics
    result = locs.copy()
    fi = result.frame.to_numpy(int)
    result['x'] = result.x.to_numpy()-applied[fi,0]
    result['y'] = result.y.to_numpy()-applied[fi,1]
    print(f"  Drift {status}: {reason} ({diagnostics['n_accepted_pairs']}/{len(pairs)} accepted pairs)")
    return result, drift_df


def save_drift_diagnostic(drift_df, path, frame_interval=1.):
    """Standalone curve and segment-support plot, including skipped corrections."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    d = drift_df.attrs['diagnostics']
    fig = Figure(figsize=(9, 6), layout='constrained'); FigureCanvasAgg(fig)
    ax, support = fig.subplots(2, 1)
    t = drift_df.frame*frame_interval
    ax.plot(t, drift_df.dx, label='Applied x'); ax.plot(t, drift_df.dy, label='Applied y')
    ax.set(ylabel='Translation (camera px)', title=f"Drift {d['status']}: {d['reason']}")
    ax.legend()
    seg = d['segments']
    times = [s['center']*frame_interval for s in seg]
    support.plot(times, [s['accepted_pairs'] for s in seg], 'o-', label='Accepted pairs')
    support.set(xlabel='Time (s)', ylabel='Supporting pairs per segment',
                title='Support is a diagnostic, not calibrated uncertainty')
    support.legend(); fig.savefig(path, dpi=150)


def drift_reference_file_metadata(params):
    """Content identity for replay and cache invalidation of an external reference."""
    from pathlib import Path
    import hashlib
    movie = Path(params.get('file', ''))
    requested = str(params.get('drift_reference_file', '')).strip()
    if not requested: raise ValueError('Select a drift reference CSV first.')
    path = Path(requested.replace('{stem}', movie.stem)).expanduser()
    if not path.is_absolute(): path = movie.parent/path
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): digest.update(block)
    return dict(path=str(path.resolve()), sha256=digest.hexdigest(),
                coordinates='camera pixels', frame_clock='same acquisition, zero-based')


def select_drift_reference(all_locs, analysis_locs, params, image_shape=None):
    """Resolve a deliberately selected reference without inheriting the analysis ROI."""
    mode = params.get('drift_reference', 'Analysis region')
    source = {'mode': mode}
    if params.get('drift_method', 'rcc') == 'fiducials' and mode == 'Analysis region':
        raise ValueError('Fiducial drift requires a separate stationary-marker rectangle or reference CSV.')
    if mode == 'Analysis region':
        reference = analysis_locs
    elif mode == 'Separate rectangle':
        rect = np.asarray(params.get('drift_ref_rect', []), dtype=float)
        if rect.shape != (4,) or not np.isfinite(rect).all() or (rect[:2] < 0).any() or (rect[2:] <= 0).any():
            raise ValueError('Drift reference rectangle requires left, top, positive width and height in pixels.')
        x, y, w, h = rect
        if image_shape and (x+w > image_shape[1] or y+h > image_shape[0]):
            raise ValueError('Drift reference rectangle extends beyond the movie frame.')
        reference = all_locs[(all_locs.x >= x) & (all_locs.x < x+w) &
                             (all_locs.y >= y) & (all_locs.y < y+h)]
        source['rectangle_px'] = rect.tolist()
    elif mode == 'Reference CSV':
        source.update(drift_reference_file_metadata(params))
        reference = pd.read_csv(source['path'])
    else:
        raise ValueError(f'Unknown drift reference: {mode}')
    _validate_reference(reference)
    return reference, source
