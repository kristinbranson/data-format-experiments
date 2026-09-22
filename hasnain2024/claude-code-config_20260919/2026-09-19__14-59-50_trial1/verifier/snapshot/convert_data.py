#!/usr/bin/env python3
"""
Convert the two-context (DR / WC) ALM electrophysiology sessions of

    Hasnain, Birnbaum et al., "Separating cognitive and motor processes in the
    behaving mouse", Nature Neuroscience 2024   (Zenodo DOI 10.5281/zenodo.13941415)

into the decoder-ready pickle format described in the task specification.

Usage
-----
    python -u convert_data.py <outpicklefile> [--full | --sample] [--show-processing]

    --full             process all 12 two-context sessions (default)
    --sample           process only the first 2 sessions (quick test)
    --show-processing  save processing_<session_id>.png diagnostics for <=2 sessions

The processing mirrors the reference MATLAB pipeline
(`DataLoadingScripts/processData.m`, `funcs/kinematics/*`, `loadMotionEnergy.m`):

  * align everything to the **go cue** (`obj.bp.ev.goCue`),
  * bin spikes at dt = 10 ms over [-2.5, 2.5] s and smooth with the reference
    causal Gaussian kernel (`mySmooth(x, 15, 'reflect')`),
  * keep clusters on the ALM probe whose quality is not garbage/noisy/real?
    and whose condition-averaged mean rate exceeds 1 Hz,
  * drop early-lick and photoinactivation trials,
  * resample DeepLabCut kinematics and motion energy onto the same 10 ms grid
    using the session's video offset (`findVideoOffset`).
"""

import argparse
import os
import pickle
import sys
import time

import h5py
import numpy as np
import scipy.io as sio

# --------------------------------------------------------------------------------------
# Reference parameters (from the figure scripts in /app/code/Scripts, e.g. Figure8a_thru_c.m,
# EDFigure2a_Left.m and the WorkingWithDataObjs.m tutorial)
# --------------------------------------------------------------------------------------
TMIN = -2.5          # s relative to the go cue (params.tmin)
TMAX = 2.5           # s relative to the go cue (params.tmax)
DT = 0.01            # s, 10 ms bins (params.dt = 1/100)
SMOOTH = 15          # samples, causal Gaussian window (params.smooth)
BCTYPE = 'reflect'   # boundary condition for smoothing (params.bctype)
LOW_FR = 1.0         # Hz, params.lowFR
BAD_QUALITY = ('garbage', 'gabrga', 'noisy', 'real?')   # findClusters.m, params.quality={'all'}
SINGLE_UNIT_QUALITY = ('excellent', 'great', 'good', 'fair')  # used only for reporting

DATA_DIR = '/app/data/Ephys_Behavior'

# The 12 two-context (DR + WC) sessions, with the ALM probe number taken from
# /app/code/DataLoadingScripts/Recording and video/load<ANM>_ALMVideo.m
# (these are exactly the sessions loaded by Scripts/Figure 8 and Scripts/EDFigure 2a-left).
SESSIONS = [
    ('JEB6',  '2021-04-18', 2),
    ('JEB7',  '2021-04-29', 1),
    ('JEB7',  '2021-04-30', 1),
    ('EKH1',  '2021-08-07', 2),
    ('EKH3',  '2021-08-11', 2),
    ('JGR2',  '2021-11-16', 1),
    ('JGR2',  '2021-11-17', 1),
    ('JGR3',  '2021-11-18', 1),
    ('JEB19', '2023-04-18', 1),
    ('JEB19', '2023-04-19', 1),
    ('JEB19', '2023-04-20', 1),
    ('JEB19', '2023-04-21', 1),
]

OUTPUT_NAMES = ['lick_direction', 'context', 'outcome',
                'tongue_velocity', 'paw_velocity', 'motion_energy']
OUTPUT_VALUES = [
    ['left', 'right', 'none'],
    ['WC', 'DR'],
    ['incorrect', 'correct', 'ignore'],
    ['below_median', 'above_median', 'not_visible'],
    ['below_median', 'above_median', 'not_visible'],
    ['below_median', 'above_median', 'no_video'],
]
INPUT_NAMES = ['time_from_go_cue']


# --------------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------------
def matstr(f, ref):
    """Decode a MATLAB char array stored in an HDF5 .mat file."""
    d = np.array(f[ref])
    if d.dtype.kind in 'ui':
        return ''.join(chr(c) for c in d.flatten() if c != 0)
    return ''


def mode_value(x):
    """MATLAB `mode` for a 1-D float vector (smallest most-frequent value), ignoring NaN."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    vals, counts = np.unique(x, return_counts=True)
    return vals[np.argmax(counts)]


def gausswin(n, alpha=2.5):
    """MATLAB gausswin(n) with default alpha."""
    k = np.arange(n) - (n - 1) / 2.0
    return np.exp(-0.5 * (alpha * k / ((n - 1) / 2.0)) ** 2)


def make_causal_kernel(n=SMOOTH):
    """Reference kernel from utils/mySmooth.m: gausswin(n) with the first floor(n/2) taps
    zeroed (making it causal) and renormalised to sum to 1."""
    k = gausswin(n)
    k[:n // 2] = 0.0
    return k / k.sum()


_KERNEL = make_causal_kernel()


def my_smooth(x, kernel=_KERNEL, bctype=BCTYPE):
    """Port of utils/mySmooth.m. Smooths along axis 0 of a 1-D or 2-D array.

    conv(x, kernel, 'same') with a causal kernel; 'reflect' prepends the first
    N samples of the signal and trims them afterwards.
    """
    n = len(kernel)
    if n <= 1:
        return x
    x = np.atleast_2d(x.T).T if x.ndim == 1 else x
    if bctype == 'reflect':
        xf = np.concatenate([x[:n], x], axis=0)
        trim = n
    else:
        xf = x
        trim = 0
    # np.convolve(v, k, 'same') matches MATLAB conv(v, k, 'same') for odd len(k)
    out = np.empty_like(xf, dtype=float)
    for j in range(xf.shape[1]):
        out[:, j] = np.convolve(xf[:, j], kernel, mode='same')
    return out[trim:]


def nearest_fill(x):
    """MATLAB fillmissing(x, 'nearest') for a 1-D array."""
    x = np.asarray(x, float)
    good = ~np.isnan(x)
    if good.all() or not good.any():
        return x.copy()
    idx = np.arange(len(x))
    gi = idx[good]
    # index of the nearest valid sample for every position
    pos = np.searchsorted(gi, idx)
    pos_lo = np.clip(pos - 1, 0, len(gi) - 1)
    pos_hi = np.clip(pos, 0, len(gi) - 1)
    lo, hi = gi[pos_lo], gi[pos_hi]
    take = np.where(np.abs(idx - lo) <= np.abs(hi - idx), lo, hi)
    out = x.copy()
    out[~good] = x[take[~good]]
    return out


# --------------------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------------------
def session_paths(anm, date):
    return (os.path.join(DATA_DIR, f'data_structure_{anm}_{date}.mat'),
            os.path.join(DATA_DIR, f'motionEnergy_{anm}_{date}.mat'))


def load_bpod(f, o):
    """Per-trial task variables (obj.bp), as boolean/float numpy arrays."""
    bp = o['bp']
    n = int(np.array(bp['Ntrials'])[0, 0])

    def g(name):
        return np.array(bp[name]).flatten()[:n]

    out = dict(
        Ntrials=n,
        hit=g('hit').astype(bool), miss=g('miss').astype(bool), no=g('no').astype(bool),
        R=g('R').astype(bool), L=g('L').astype(bool),
        early=g('early').astype(bool), autowater=g('autowater').astype(bool),
    )
    try:
        out['stim'] = np.array(bp['stim']['enable']).flatten()[:n].astype(bool)
    except (KeyError, TypeError):
        out['stim'] = np.zeros(n, bool)
    ev = bp['ev']
    for name in ('bitStart', 'sample', 'delay', 'goCue'):
        out[name] = np.array(ev[name]).flatten()[:n]
    out['lickL'] = [np.array(f[r]).flatten() for r in np.array(ev['lickL']).flatten()[:n]]
    out['lickR'] = [np.array(f[r]).flatten() for r in np.array(ev['lickR']).flatten()[:n]]
    return out


def video_offset(f, o, bp):
    """funcs/findVideoOffset.m: offset between the video clock and the Bpod clock (s)."""
    fs = float(np.array(o['sglx']['fs'])[0, 0])
    bitstart = np.array(o['sglx']['bitcode']['bitstart']).flatten()
    return mode_value(bitstart) / fs - mode_value(bp['bitStart'])


def bin_spikes(f, clu, keep_idx, gocue, ntrials, edges):
    """Spike counts / dt, aligned to the go cue: returns (T, nunits, ntrials) firing rate."""
    t = len(edges) - 1
    rate = np.zeros((t, len(keep_idx), ntrials), dtype=np.float32)
    trial_bins = np.arange(0.5, ntrials + 1.5)
    for k, ci in enumerate(keep_idx):
        tm = np.array(f[clu['trialtm'][ci, 0]]).flatten()
        tr = np.array(f[clu['trial'][ci, 0]]).flatten().astype(int)
        if tm.size == 0:
            continue
        ok = (tr >= 1) & (tr <= ntrials) & ~np.isnan(tm)
        tm, tr = tm[ok], tr[ok]
        aligned = tm - gocue[tr - 1]
        # half-open bins, exactly like MATLAB histc(...)(1:end-1)
        inwin = (aligned >= edges[0]) & (aligned < edges[-1])
        h, _, _ = np.histogram2d(aligned[inwin], tr[inwin], bins=[edges, trial_bins])
        rate[:, k, :] = h / DT
    return rate


def fig8_conditions(bp):
    """params.condition from Scripts/Figure 8/Figure8a_thru_c.m -- the script that analyses
    exactly this two-context data set. Used only for the low-firing-rate unit filter, to
    reproduce removeLowFRClusters.m."""
    hit, miss, no = bp['hit'], bp['miss'], bp['no']
    early, aw, stim = bp['early'], bp['autowater'], bp['stim']
    return [
        hit | miss | no,                    # all trials
        hit & ~stim & ~aw,                  # DR hits
        hit & ~stim & aw,                   # WC hits
        miss & ~stim & ~aw,                 # DR errors
        miss & ~stim & aw,                  # WC errors
        hit & ~stim & ~aw & ~early,         # DR hits, no early lick
        hit & ~stim & aw & ~early,          # WC hits, no early lick
    ]


def low_fr_mask(rate_smooth, conditions):
    """Port of removeLowFRClusters.m: mean over conditions (omitnan) then over time."""
    psth = np.full((rate_smooth.shape[0], rate_smooth.shape[1], len(conditions)), np.nan)
    for j, c in enumerate(conditions):
        if c.sum() == 0:
            continue
        psth[:, :, j] = rate_smooth[:, :, c].mean(axis=2)
    with np.errstate(invalid='ignore'):
        mean_fr = np.nanmean(np.nanmean(psth, axis=2), axis=0)
    return mean_fr > LOW_FR, mean_fr


# --------------------------------------------------------------------------------------
# kinematics / motion energy
# --------------------------------------------------------------------------------------
def feature_names(f, traj_view):
    return [matstr(f, r) for r in np.array(f[traj_view['featNames'][0, 0]]).flatten()]


def trial_frame_times(f, traj_view, trial, vidshift, gocue_t):
    """Video frame times of one trial, in seconds relative to the go cue.
    Returns None when the trial has no usable video."""
    ft = np.array(f[traj_view['frameTimes'][trial, 0]]).flatten()
    if ft.size == 0 or np.all(np.isnan(ft)):
        return None
    if np.isnan(ft).any():
        return None
    return ft - vidshift - gocue_t


def extract_kinematics(f, o, bp, vidshift, taxis, me_cells):
    """Resample tongue position (side cam), paw positions (bottom cam) and motion energy
    onto the go-cue-aligned time axis, and return speed + visibility for each.

    Follows funcs/kinematics/findPosition.m + findVelocity.m and
    DataLoadingScripts/loadMotionEnergy.m:
      * interp1(frameTimes - vidshift - alignTime, value, taxis), NaN outside the range,
      * velocity = gradient of the *interpolated* position,
      * non-tongue features get their baseline drift median(diff(pos)) removed,
      * motion energy is nearest-filled (fillmissing) as in loadMotionEnergy.m.
    """
    n = bp['Ntrials']
    t = len(taxis)
    v1 = f[o['traj'][0, 0]]      # side camera
    v2 = f[o['traj'][1, 0]]      # bottom camera
    n1 = feature_names(f, v1)
    n2 = feature_names(f, v2)
    i_tongue = n1.index('tongue')
    i_paws = [n2.index(p) for p in ('top_paw', 'bottom_paw') if p in n2]

    tongue_speed = np.full((t, n), np.nan)
    tongue_vis = np.zeros((t, n), bool)
    paw_speed = np.full((t, n), np.nan)
    paw_vis = np.zeros((t, n), bool)
    me = np.full((t, n), np.nan)
    has_video = np.zeros(n, bool)

    for tr in range(n):
        tt = trial_frame_times(f, v1, tr, vidshift, bp['goCue'][tr])
        tt2 = trial_frame_times(f, v2, tr, vidshift, bp['goCue'][tr])
        if tt is None:
            continue
        has_video[tr] = True
        ts1 = np.array(f[v1['ts'][tr, 0]])   # (nfeat, [x,y,conf], nframes)
        ts2 = np.array(f[v2['ts'][tr, 0]]) if tt2 is not None else None
        nf = min(len(tt), ts1.shape[2])
        tt = tt[:nf]

        def resample(y, times=None):
            times = tt if times is None else times
            k = min(len(times), len(y))
            return np.interp(taxis, times[:k], y[:k], left=np.nan, right=np.nan)

        # ---- tongue (side camera). NaN = tongue not detected by DeepLabCut ----
        x = resample(ts1[i_tongue, 0])
        y = resample(ts1[i_tongue, 1])
        vis = ~np.isnan(x)
        if vis.any():
            vx = np.gradient(nearest_fill(x))
            vy = np.gradient(nearest_fill(y))
            tongue_speed[:, tr] = np.hypot(vx, vy)
        tongue_vis[:, tr] = vis

        # ---- paws (bottom camera only, per Methods) ----
        speeds, viss = [], []
        for i in (i_paws if ts2 is not None else []):
            px = resample(ts2[i, 0], tt2)
            py = resample(ts2[i, 1], tt2)
            vi = ~np.isnan(px)
            if not vi.any():
                continue
            pxf, pyf = nearest_fill(px), nearest_fill(py)
            vx = np.gradient(pxf) - np.median(np.diff(pxf))
            vy = np.gradient(pyf) - np.median(np.diff(pyf))
            s = np.hypot(vx, vy)
            s[~vi] = np.nan
            speeds.append(s)
            viss.append(vi)
        if speeds:
            sm = np.vstack(speeds)
            cnt = np.sum(~np.isnan(sm), axis=0)
            with np.errstate(invalid='ignore', divide='ignore'):
                paw_speed[:, tr] = np.where(cnt > 0, np.nansum(sm, axis=0) / np.maximum(cnt, 1),
                                            np.nan)
            paw_vis[:, tr] = np.any(np.vstack(viss), axis=0)

        # ---- motion energy ----
        if me_cells is not None and tr < len(me_cells):
            mv = np.asarray(me_cells[tr], float).flatten()
            k = min(len(mv), nf)
            if k > 1:
                # exactly as loadMotionEnergy.m: interp1 onto the aligned axis, then
                # fillmissing(...,'nearest') for the few samples at the trial edges that
                # the video does not cover. me.data itself contains no NaNs, so after
                # this a trial either has motion energy everywhere or nowhere ("no video").
                me[:, tr] = nearest_fill(
                    np.interp(taxis, tt[:k], mv[:k], left=np.nan, right=np.nan))

    return dict(tongue_speed=tongue_speed, tongue_vis=tongue_vis,
                paw_speed=paw_speed, paw_vis=paw_vis,
                me=me, has_video=has_video)


def load_motion_energy(me_path):
    """Load motionEnergy_<ANM>_<DATE>.mat -> (list of per-trial 400 Hz traces, moveThresh)."""
    if not os.path.exists(me_path):
        return None, np.nan
    m = sio.loadmat(me_path, struct_as_record=False, squeeze_me=False)
    me = m['me'][0, 0]
    data = me.data
    if not isinstance(data, np.ndarray) or data.dtype != object:
        # some objs store me.data as a struct with a .data field
        data = data[0, 0].data
    cells = [np.asarray(data[i, 0]).flatten() for i in range(data.shape[0])]
    thresh = float(np.asarray(me.moveThresh).flatten()[0]) if hasattr(me, 'moveThresh') else np.nan
    return cells, thresh


def discretize(values, visible, keep_trials):
    """0 = below the session's 50th percentile, 1 = at/above it, 2 = not visible / no video.

    The percentile is computed over the *visible* time points of the kept trials only,
    because 'not visible' is its own category.
    """
    vals = values[:, keep_trials]
    vis = visible[:, keep_trials] & ~np.isnan(vals)
    if vis.sum() == 0:
        return np.full(vals.shape, 2, dtype=np.int64), np.nan
    thresh = np.percentile(vals[vis], 50)
    out = np.full(vals.shape, 2, dtype=np.int64)
    out[vis & (vals < thresh)] = 0
    out[vis & (vals >= thresh)] = 1
    return out, float(thresh)


# --------------------------------------------------------------------------------------
# per-session conversion
# --------------------------------------------------------------------------------------
def convert_session(anm, date, probe, verbose=True):
    t_start = time.time()
    data_path, me_path = session_paths(anm, date)
    f = h5py.File(data_path, 'r')
    o = f['obj']

    timing = {}
    bp = load_bpod(f, o)
    n = bp['Ntrials']

    edges = np.arange(TMIN, TMAX + DT / 2, DT)
    taxis = (edges[:-1] + edges[1:]) / 2.0           # == edges + dt/2 (drop last), getSeq.m
    t = len(taxis)

    # ---------------- trial curation ----------------
    # every params.condition in the reference excludes early-lick and photostim trials
    keep = (bp['hit'] | bp['miss'] | bp['no']) & ~bp['early'] & ~bp['stim'] & ~np.isnan(bp['goCue'])

    # ---------------- neurons ----------------
    t0 = time.time()
    clu = f[o['clu'][probe - 1, 0]]
    quality = [matstr(f, r).strip() for r in np.array(clu['quality']).flatten()]
    qual_keep = [i for i, q in enumerate(quality) if q not in BAD_QUALITY]
    rate = bin_spikes(f, clu, qual_keep, bp['goCue'], n, edges)
    timing['bin_spikes'] = time.time() - t0

    t0 = time.time()
    rate_s = my_smooth(rate.reshape(t, -1)).reshape(rate.shape).astype(np.float32)
    timing['smooth'] = time.time() - t0

    fr_mask, mean_fr = low_fr_mask(rate_s, fig8_conditions(bp))
    unit_idx = np.nonzero(fr_mask)[0]
    neural = rate_s[:, unit_idx, :]
    qualities_kept = [quality[qual_keep[i]] for i in unit_idx]
    n_single = sum(q.lower() in SINGLE_UNIT_QUALITY for q in qualities_kept)

    # ---------------- video-derived behaviour ----------------
    t0 = time.time()
    vidshift = video_offset(f, o, bp)
    me_cells, move_thresh = load_motion_energy(me_path)
    kin = extract_kinematics(f, o, bp, vidshift, taxis, me_cells)
    timing['kinematics'] = time.time() - t0

    # Trials with no usable video are dropped: three of the six decoder outputs
    # (tongue velocity, paw velocity, motion energy) are undefined for them, and the
    # reference funcs/kinematics/findPosition.m likewise skips trials whose video is
    # missing (NdroppedFrames = NaN).
    n_dropped_no_video = int((keep & ~kin['has_video']).sum())
    keep = keep & kin['has_video']
    keep_idx = np.nonzero(keep)[0]

    # ---------------- outputs ----------------
    lick_dir = np.full(n, -1, dtype=np.int64)
    lick_dir[(bp['L'] & bp['hit']) | (bp['R'] & bp['miss'])] = 0      # licked left
    lick_dir[(bp['R'] & bp['hit']) | (bp['L'] & bp['miss'])] = 1      # licked right
    lick_dir[bp['no']] = 2                                            # no response
    context = np.where(bp['autowater'], 0, 1).astype(np.int64)        # 0 = WC, 1 = DR
    outcome = np.full(n, -1, dtype=np.int64)
    outcome[bp['miss']] = 0
    outcome[bp['hit']] = 1
    outcome[bp['no']] = 2

    assert (lick_dir[keep_idx] >= 0).all(), f'{anm} {date}: undefined lick direction'
    assert (outcome[keep_idx] >= 0).all(), f'{anm} {date}: undefined outcome'

    tongue_cat, tongue_thresh = discretize(kin['tongue_speed'], kin['tongue_vis'], keep_idx)
    paw_cat, paw_thresh = discretize(kin['paw_speed'], kin['paw_vis'], keep_idx)
    me_vis = ~np.isnan(kin['me']) & kin['has_video'][None, :]
    me_cat, me_thresh = discretize(kin['me'], me_vis, keep_idx)

    # ---------------- assemble per-trial arrays ----------------
    input_trial = taxis.astype(np.float32)[None, :]      # (1, T), identical for every trial
    neural_trials, input_trials, output_trials = [], [], []
    for j, tr in enumerate(keep_idx):
        neural_trials.append(np.ascontiguousarray(neural[:, :, tr].T))          # (nunits, T)
        input_trials.append(input_trial.copy())
        out = np.empty((6, t), dtype=np.int64)
        out[0, :] = lick_dir[tr]
        out[1, :] = context[tr]
        out[2, :] = outcome[tr]
        out[3, :] = tongue_cat[:, j]
        out[4, :] = paw_cat[:, j]
        out[5, :] = me_cat[:, j]
        output_trials.append(out)

    f.close()

    info = dict(
        session_id=f'{anm}_{date}', animal=anm, date=date, probe=probe,
        n_trials_total=int(n), n_trials_kept=int(len(keep_idx)),
        n_early=int(bp['early'].sum()), n_stim=int(bp['stim'].sum()),
        n_clusters_raw=len(quality), n_clusters_quality=len(qual_keep),
        n_units=int(len(unit_idx)), n_single_units=int(n_single),
        vidshift=float(vidshift), move_thresh=move_thresh,
        tongue_thresh=tongue_thresh, paw_thresh=paw_thresh, me_thresh=me_thresh,
        n_trials_dropped_no_video=n_dropped_no_video,
        # provenance: 1-based indices into obj.bp / obj.clu of the exported trials,
        # and 0-based indices into obj.clu{probe} of the exported units
        trial_ids=(keep_idx + 1).astype(np.int64),
        cluster_ids=np.array([qual_keep[i] for i in unit_idx], dtype=np.int64),
        unit_qualities=qualities_kept,
        timing=timing, elapsed=time.time() - t_start,
    )
    if verbose:
        print(f"  {anm} {date} probe{probe}: {info['n_clusters_raw']} clusters -> "
              f"{info['n_clusters_quality']} after quality -> {info['n_units']} after >1Hz "
              f"({info['n_single_units']} single) | trials {n} -> {len(keep_idx)} "
              f"(early {info['n_early']}, stim {info['n_stim']}, no-video {n_dropped_no_video}) | "
              f"vidshift {vidshift:.3f}s | thresholds tongue {tongue_thresh:.3f} "
              f"paw {paw_thresh:.3f} me {me_thresh:.2f} (moveThresh {move_thresh}) | "
              f"{info['elapsed']:.1f}s {timing}")

    extras = dict(bp=bp, kin=kin, taxis=taxis, keep_idx=keep_idx, mean_fr=mean_fr,
                  neural=neural, tongue_cat=tongue_cat, paw_cat=paw_cat, me_cat=me_cat,
                  lick_dir=lick_dir, context=context, outcome=outcome,
                  tongue_thresh=tongue_thresh, paw_thresh=paw_thresh, me_thresh=me_thresh,
                  rate=rate, unit_idx=unit_idx, qual_keep=qual_keep, vidshift=vidshift,
                  raw_path=data_path)
    return neural_trials, input_trials, output_trials, info, extras


# --------------------------------------------------------------------------------------
# diagnostics plot
# --------------------------------------------------------------------------------------
def plot_processing(anm, date, probe, extras, info):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    taxis = extras['taxis']
    keep_idx = extras['keep_idx']
    bp = extras['bp']
    kin = extras['kin']
    sid = f'{anm}_{date}'

    fig, ax = plt.subplots(4, 3, figsize=(24, 18))

    # (0,0) raw spike raster vs binned+smoothed rate for one unit/trial ------------------
    f = h5py.File(extras['raw_path'], 'r')
    clu = f[f['obj']['clu'][probe - 1, 0]]
    u = int(np.argmax(extras['neural'][:, :, keep_idx].mean(axis=(0, 2))))  # busiest unit
    ci = extras['qual_keep'][extras['unit_idx'][u]]
    tm = np.array(f[clu['trialtm'][ci, 0]]).flatten()
    tr = np.array(f[clu['trial'][ci, 0]]).flatten().astype(int)
    f.close()
    a = ax[0, 0]
    ntr_plot = min(60, len(keep_idx))
    for row, t_i in enumerate(keep_idx[:ntr_plot]):
        s = tm[tr == (t_i + 1)] - bp['goCue'][t_i]
        s = s[(s >= TMIN) & (s < TMAX)]
        a.plot(s, np.full(len(s), row), '|k', markersize=3)
    a.axvline(0, color='r')
    a.set_title(f'{sid}: raw spikes of unit {ci} (aligned to go cue)')
    a.set_xlabel('time from go cue (s)'); a.set_ylabel('trial'); a.set_xlim(TMIN, TMAX)

    a = ax[0, 1]
    ex_tr = keep_idx[min(5, len(keep_idx) - 1)]
    a.step(taxis, extras['rate'][:, extras['unit_idx'][u], ex_tr], where='mid',
           label='binned (10 ms) / dt')
    a.plot(taxis, extras['neural'][:, u, ex_tr], 'r', lw=2, label='causal-Gaussian smoothed')
    s = tm[tr == (ex_tr + 1)] - bp['goCue'][ex_tr]
    a.plot(s, np.full(len(s), -2.0), '|k', markersize=8, label='spikes')
    a.axvline(0, color='r', ls='--')
    a.legend(); a.set_xlim(TMIN, TMAX)
    a.set_title(f'binning + smoothing, unit {ci}, trial {ex_tr}')
    a.set_xlabel('time from go cue (s)'); a.set_ylabel('spks/s')

    # (0,2) mean FR distribution + 1 Hz cut ---------------------------------------------
    a = ax[0, 2]
    mf = extras['mean_fr']
    a.hist(np.log10(np.maximum(mf, 1e-3)), bins=40)
    a.axvline(0, color='r', label='1 Hz cut')
    a.set_xlabel('log10 mean FR (Hz)'); a.set_ylabel('# clusters'); a.legend()
    a.set_title(f'low-FR filter: {len(mf)} -> {info["n_units"]} units')

    # (1,0) population PSTH sorted, DR right vs left hits --------------------------------
    a = ax[1, 0]
    m = extras['neural'][:, :, keep_idx].mean(axis=2)
    order = np.argsort(np.argmax(m, axis=0))
    im = a.imshow(m[:, order].T, aspect='auto', origin='lower',
                  extent=[taxis[0], taxis[-1], 0, m.shape[1]], cmap='magma')
    a.axvline(0, color='w'); plt.colorbar(im, ax=a)
    a.set_title('trial-averaged firing rate (spks/s), units sorted by peak time')
    a.set_xlabel('time from go cue (s)'); a.set_ylabel('unit')

    # (1,1) lick raster, sorted by lick direction (alignment check) ----------------------
    a = ax[1, 1]
    ld = extras['lick_dir'][keep_idx]
    order = np.argsort(ld, kind='stable')
    for row, j in enumerate(order):
        t_i = keep_idx[j]
        for licks, col in ((bp['lickL'][t_i], 'r'), (bp['lickR'][t_i], 'b')):
            s = np.asarray(licks).flatten() - bp['goCue'][t_i]
            s = s[(s >= TMIN) & (s < TMAX)]
            a.plot(s, np.full(len(s), row), '.', color=col, markersize=1.5)
    a.axvline(0, color='k')
    nleft = int((ld == 0).sum()); nright = int((ld == 1).sum())
    a.axhline(nleft, color='g'); a.axhline(nleft + nright, color='g')
    a.set_title('lick raster sorted by lick_direction (left | right | none)\n'
                'red = left port, blue = right port')
    a.set_xlabel('time from go cue (s)'); a.set_ylabel('trial (sorted)')
    a.set_xlim(TMIN, TMAX)

    # (1,2) per-trial outputs ------------------------------------------------------------
    a = ax[1, 2]
    a.plot(extras['context'][keep_idx] + 0.0, '.', label='context (0=WC,1=DR)')
    a.plot(extras['lick_dir'][keep_idx] + 2.5, '.', label='lick dir +2.5')
    a.plot(extras['outcome'][keep_idx] + 6.0, '.', label='outcome +6')
    a.plot(bp['autowater'][keep_idx] * 0.8 + 10, '-', lw=0.8, label='raw autowater +10')
    a.legend(fontsize=8); a.set_xlabel('kept trial #')
    a.set_title('per-trial outputs vs. raw bpod autowater (block structure)')

    # (2,0) tongue: raw DLC y vs resampled, one trial -------------------------------------
    a = ax[2, 0]
    f = h5py.File(extras['raw_path'], 'r')
    v1 = f[f['obj']['traj'][0, 0]]
    ex_tr = keep_idx[min(5, len(keep_idx) - 1)]
    ft = np.array(f[v1['frameTimes'][ex_tr, 0]]).flatten() - extras['vidshift'] - bp['goCue'][ex_tr]
    names = [matstr(f, r) for r in np.array(f[v1['featNames'][0, 0]]).flatten()]
    ts = np.array(f[v1['ts'][ex_tr, 0]])
    f.close()
    raw_y = ts[names.index('tongue'), 1, :len(ft)]
    a.plot(ft, raw_y, '.', markersize=4, label='raw DLC tongue y (400 Hz)')
    res_y = np.interp(taxis, ft[:len(raw_y)], raw_y, left=np.nan, right=np.nan)
    res_y[~kin['tongue_vis'][:, ex_tr]] = np.nan
    a.plot(taxis, res_y, 'r.-', markersize=3, lw=0.7,
           label='resampled onto 10 ms grid (NaN = not visible)')
    a.axvline(0, color='r'); a.set_xlim(TMIN, TMAX); a.legend(fontsize=8)
    a.set_title(f'tongue tracking, trial {ex_tr} (video->go-cue alignment)')
    a.set_xlabel('time from go cue (s)')

    # (2,1) tongue speed + threshold, (2,2) categories ------------------------------------
    a = ax[2, 1]
    sp = kin['tongue_speed'][:, ex_tr].copy()
    sp[~kin['tongue_vis'][:, ex_tr]] = np.nan
    a.plot(taxis, sp, 'k'); a.axhline(extras['tongue_thresh'], color='r', label='session median')
    a.axvline(0, color='r', ls='--'); a.legend(fontsize=8)
    a.set_title(f'tongue speed, trial {ex_tr}'); a.set_xlabel('time from go cue (s)')
    a2 = a.twinx(); a2.plot(taxis, extras['tongue_cat'][:, list(keep_idx).index(ex_tr)],
                            'b.', markersize=3)
    a2.set_ylabel('category (0/1/2)', color='b')

    a = ax[2, 2]
    im = a.imshow(extras['tongue_cat'].T, aspect='auto', origin='lower', interpolation='nearest',
                  extent=[taxis[0], taxis[-1], 0, len(keep_idx)], cmap='viridis', vmin=0, vmax=2)
    a.axvline(0, color='w'); plt.colorbar(im, ax=a)
    a.set_title('tongue_velocity category (0 low, 1 high, 2 not visible)')
    a.set_xlabel('time from go cue (s)'); a.set_ylabel('kept trial')

    # (3,0) paw speed, (3,1) motion energy, (3,2) ME categories ---------------------------
    a = ax[3, 0]
    a.plot(taxis, kin['paw_speed'][:, ex_tr], 'k')
    a.axhline(extras['paw_thresh'], color='r', label='session median')
    a.axvline(0, color='r', ls='--'); a.legend(fontsize=8)
    a.set_title(f'paw speed, trial {ex_tr}'); a.set_xlabel('time from go cue (s)')
    a2 = a.twinx(); a2.plot(taxis, extras['paw_cat'][:, list(keep_idx).index(ex_tr)],
                            'b.', markersize=3)
    a2.set_ylabel('category', color='b')

    a = ax[3, 1]
    a.plot(taxis, kin['me'][:, ex_tr], 'k', label='motion energy')
    a.axhline(extras['me_thresh'], color='r', label='session median (used)')
    if np.isfinite(info['move_thresh']):
        a.axhline(info['move_thresh'], color='g', ls=':', label="me.moveThresh (paper's)")
    a.axvline(0, color='r', ls='--'); a.legend(fontsize=8)
    a.set_title(f'motion energy, trial {ex_tr}'); a.set_xlabel('time from go cue (s)')

    a = ax[3, 2]
    im = a.imshow(extras['me_cat'].T, aspect='auto', origin='lower', interpolation='nearest',
                  extent=[taxis[0], taxis[-1], 0, len(keep_idx)], cmap='viridis', vmin=0, vmax=2)
    a.axvline(0, color='w'); plt.colorbar(im, ax=a)
    a.set_title('motion_energy category (0 low, 1 high, 2 no video)')
    a.set_xlabel('time from go cue (s)'); a.set_ylabel('kept trial')

    fig.suptitle(f'Processing steps: {sid} (probe {probe})', fontsize=16)
    fig.tight_layout()
    out = f'/app/processing_{sid}.png'
    fig.savefig(out, dpi=90)
    plt.close(fig)
    print(f'  wrote {out}')


# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('outfile', help='output pickle path')
    ap.add_argument('--full', action='store_true', default=True, help='process all sessions')
    ap.add_argument('--sample', action='store_true', help='process only 2 sessions')
    ap.add_argument('--show-processing', action='store_true',
                    help='save processing_<session_id>.png for up to 2 sessions')
    args = ap.parse_args()

    sessions = SESSIONS[:2] if args.sample else SESSIONS
    print(f'Converting {len(sessions)} session(s) -> {args.outfile}')

    data = dict(neural=[], input=[], output=[], subjects=[], subject_idx=[],
                brain_regions=['ALM'], brain_region_idx=[],
                input_names=INPUT_NAMES, output_names=OUTPUT_NAMES,
                output_values=OUTPUT_VALUES, metadata={})
    infos = []
    t_all = time.time()
    for k, (anm, date, probe) in enumerate(sessions):
        neural, inp, out, info, extras = convert_session(anm, date, probe)
        if len(neural) < 2:
            print(f'  SKIPPING {anm} {date}: fewer than 2 usable trials')
            continue
        if info['n_units'] < 10:
            print(f'  SKIPPING {anm} {date}: fewer than 10 units')
            continue
        data['neural'].append(neural)
        data['input'].append(inp)
        data['output'].append(out)
        if anm not in data['subjects']:
            data['subjects'].append(anm)
        data['subject_idx'].append(data['subjects'].index(anm))
        data['brain_region_idx'].append(np.zeros(info['n_units'], dtype=np.int64))
        infos.append(info)
        if args.show_processing and k < 2:
            plot_processing(anm, date, probe, extras, info)
        del extras

    data['subject_idx'] = np.array(data['subject_idx'], dtype=np.int64)
    data['metadata'] = dict(
        task_description=(
            'Head-fixed mice alternate block-wise between two directional-licking tasks '
            'that require the same instructed motor output. In the delayed-response (DR) '
            'context an auditory sample tone (1.3 s) indicates the rewarded lickport, a '
            '0.9 s delay follows, and an auditory go cue instructs the animal to lick. In '
            'the water-cued (WC, obj.bp.autowater) context all auditory cues are omitted '
            'and ~3 ul of water is simply presented at a randomly chosen port. Mice receive '
            'no explicit cue about the current block. Decoded from ALM spiking are the '
            'lick direction (left/right/none), the behavioural context (WC/DR), the trial '
            'outcome (incorrect/correct/ignore) and three binarised uninstructed-movement '
            'variables (tongue speed, paw speed, whole-frame motion energy), each split at '
            'its own per-session median with a third class for time points at which the '
            'feature was not visible / no video was available.'),
        time_bin_size=DT * 1000.0,
        temporal_alignment_event=('go cue onset (obj.bp.ev.goCue): the auditory go cue on DR '
                                  'trials, the water presentation on WC trials'),
        off_start=TMIN,
        off_end=TMAX,
        n_timepoints=int(round((TMAX - TMIN) / DT)),
        smoothing='causal Gaussian, gausswin(15) with the first 7 taps zeroed '
                  '(utils/mySmooth.m, reflect boundary)',
        neural_units='spikes/s (single-trial smoothed firing rate, obj.trialdat equivalent)',
        neuron_filtering=('ALM probe only (probe listed in load<ANM>_ALMVideo.m); clusters of '
                          'quality garbage/noisy/real? discarded (findClusters.m with '
                          "params.quality={'all'}); clusters with condition-averaged mean "
                          'firing rate <= 1 Hz discarded (removeLowFRClusters.m, '
                          'params.lowFR = 1)'),
        trial_filtering=('early-lick trials (obj.bp.early) and photoinactivation trials '
                         '(obj.bp.stim.enable) removed, as in every params.condition of the '
                         'reference code. Ignore trials (obj.bp.no) are RETAINED because the '
                         'decoder task requires an "ignore" outcome class; the paper itself '
                         'excludes them.'),
        session_selection=('the 12 two-context (DR + WC) ALM recording sessions loaded by '
                           'Scripts/Figure 8 and Scripts/EDFigure 2a-left; these are the only '
                           'ephys sessions that contain both behavioural contexts'),
        video='DeepLabCut at 400 Hz, two cameras; video clock aligned with '
              'findVideoOffset.m (mode(sglx.bitcode.bitstart)/fs - mode(bp.ev.bitStart))',
        source='Hasnain, Birnbaum et al., Nature Neuroscience 2024; '
               'Zenodo DOI 10.5281/zenodo.13941415',
        session_info=infos,
    )

    with open(args.outfile, 'wb') as fh:
        pickle.dump(data, fh)

    # ---- summary ----
    ntr = [len(s) for s in data['neural']]
    nun = [s[0].shape[0] for s in data['neural']]
    print('\n=== conversion summary ===')
    print(f'sessions: {len(data["neural"])}   subjects: {len(data["subjects"])} '
          f'{data["subjects"]}')
    print(f'trials: total {sum(ntr)}, per session {ntr}')
    print(f'units: total {sum(nun)}, per session {nun}, '
          f'single units total {sum(i["n_single_units"] for i in infos)}')
    allout = np.concatenate([o for s in data['output'] for o in s], axis=1)
    for i, name in enumerate(OUTPUT_NAMES):
        cnt = np.bincount(allout[i], minlength=len(OUTPUT_VALUES[i]))
        frac = cnt / cnt.sum()
        print(f'  {name}: ' + ', '.join(f'{v} {frac[j]:.3f}'
                                        for j, v in enumerate(OUTPUT_VALUES[i])))
    print(f'wrote {args.outfile} ({os.path.getsize(args.outfile)/1e6:.1f} MB) '
          f'in {time.time()-t_all:.1f}s')


if __name__ == '__main__':
    sys.exit(main())
