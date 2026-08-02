#!/usr/bin/env python3
"""
Convert Hasnain, Birnbaum et al (2024) electrophysiology data to decoder-compatible format.

Usage:
    python -u convert_data.py <output.pkl> [--sample] [--full] [--show-processing]
"""

import argparse
import glob
import os
import pickle
import sys
import time
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as scipy_stats
from scipy.signal import windows as scipy_windows
from scipy.interpolate import interp1d

warnings.filterwarnings('ignore', category=FutureWarning)

# ============================================================
# SESSION REGISTRY: which sessions to load with which probes
# Based on the loading scripts in code/DataLoadingScripts/Recording and video/
# ============================================================

EPHYS_SESSIONS = [
    # (animal, date, probes_for_ALM, data_dir)
    ('EKH1', '2021-08-07', [2], 'Ephys_Behavior'),
    ('EKH3', '2021-08-11', [2], 'Ephys_Behavior'),
    ('JEB6', '2021-04-18', [2], 'Ephys_Behavior'),
    ('JEB7', '2021-04-29', [1], 'Ephys_Behavior'),
    ('JEB7', '2021-04-30', [1], 'Ephys_Behavior'),
    ('JEB13', '2022-09-13', [2], 'Ephys_Behavior'),
    ('JEB13', '2022-09-14', [2], 'Ephys_Behavior'),
    ('JEB13', '2022-09-21', [1], 'Ephys_Behavior'),
    ('JEB13', '2022-09-24', [1], 'Ephys_Behavior'),
    ('JEB13', '2022-09-25', [1], 'Ephys_Behavior'),
    ('JEB14', '2022-08-22', [1], 'Ephys_Behavior'),
    ('JEB14', '2022-08-23', [1], 'Ephys_Behavior'),
    ('JEB14', '2022-08-24', [1], 'Ephys_Behavior'),
    ('JEB14', '2022-08-25', [1], 'Ephys_Behavior'),
    ('JEB15', '2022-07-26', [1, 2], 'Ephys_Behavior'),
    ('JEB15', '2022-07-27', [1, 2], 'Ephys_Behavior'),
    ('JEB15', '2022-07-28', [1, 2], 'Ephys_Behavior'),
    ('JEB15', '2022-07-29', [2], 'Ephys_Behavior'),
    ('JEB19', '2023-04-18', [1], 'Ephys_Behavior'),
    ('JEB19', '2023-04-19', [1], 'Ephys_Behavior'),
    ('JEB19', '2023-04-20', [1], 'Ephys_Behavior'),
    ('JEB19', '2023-04-21', [1], 'Ephys_Behavior'),
    ('JGR2', '2021-11-16', [1], 'Ephys_Behavior'),
    ('JGR2', '2021-11-17', [1], 'Ephys_Behavior'),
    ('JGR3', '2021-11-18', [1], 'Ephys_Behavior'),
]

RANDOMIZED_DELAY_SESSIONS = [
    ('JEB11', '2022-05-10', [1], 'RandomizedDelay_Ephys_Behavior'),
    ('JEB11', '2022-05-11', [1], 'RandomizedDelay_Ephys_Behavior'),
    ('JEB12', '2022-05-12', [1], 'RandomizedDelay_Ephys_Behavior'),
    ('JEB12', '2022-05-13', [1], 'RandomizedDelay_Ephys_Behavior'),
    ('JEB23', '2023-10-10', [1], 'RandomizedDelay_Ephys_Behavior'),
    ('JEB23', '2023-10-11', [1], 'RandomizedDelay_Ephys_Behavior'),
    ('JEB23', '2023-10-12', [1], 'RandomizedDelay_Ephys_Behavior'),
    ('JEB23', '2023-10-13', [1], 'RandomizedDelay_Ephys_Behavior'),
    ('JEB23', '2023-10-18', [1], 'RandomizedDelay_Ephys_Behavior'),
    ('JEB23', '2023-10-19', [1], 'RandomizedDelay_Ephys_Behavior'),
    # JEB23 2023-10-20 commented out in loading script
    ('JEB23', '2023-10-21', [1], 'RandomizedDelay_Ephys_Behavior'),
    # JEB24 2023-10-03 and 2023-10-04 not in loading scripts
    ('JEB24', '2023-10-23', [1], 'RandomizedDelay_Ephys_Behavior'),
    ('JEB24', '2023-10-24', [1], 'RandomizedDelay_Ephys_Behavior'),
    ('JEB24', '2023-10-25', [1], 'RandomizedDelay_Ephys_Behavior'),
    ('JEB24', '2023-10-26', [1], 'RandomizedDelay_Ephys_Behavior'),
    ('JEB24', '2023-10-27', [1], 'RandomizedDelay_Ephys_Behavior'),
    ('JEB24', '2023-10-31', [1], 'RandomizedDelay_Ephys_Behavior'),
    ('JEB24', '2023-11-02', [1], 'RandomizedDelay_Ephys_Behavior'),
    ('JEB24', '2023-11-03', [1], 'RandomizedDelay_Ephys_Behavior'),
]

ALL_SESSIONS = EPHYS_SESSIONS + RANDOMIZED_DELAY_SESSIONS

# ============================================================
# PARAMETERS (matching reference code)
# ============================================================

ALIGN_EVENT = 'goCue'
TMIN = -2.5  # seconds
TMAX = 2.5   # seconds
DT = 1.0 / 100  # 10 ms bins
SMOOTH_WINDOW = 15  # bins, causal Gaussian
BC_TYPE = 'reflect'
LOW_FR = 1.0  # Hz, minimum firing rate
EXCLUDED_QUALITIES = {'garbage', 'gabrga', 'noisy', 'real?'}
VIDEO_FR = 400  # Hz
DATA_ROOT = '/app/data'

# ============================================================
# SMOOTHING (matching mySmooth.m)
# ============================================================

def causal_gaussian_smooth(x, N, bctype='reflect'):
    """Causal Gaussian smoothing matching mySmooth.m.
    x: (T,) or (T, C) array
    N: window size in bins
    """
    if N <= 1:
        return x

    was_1d = x.ndim == 1
    if was_1d:
        x = x[:, None]

    # Build causal Gaussian kernel
    kern = np.array(scipy_windows.gaussian(N, std=N/6.0))  # gausswin equivalent
    kern[:N//2] = 0  # causal: zero out first half
    kern = kern / kern.sum()

    # Handle boundary conditions
    if bctype == 'reflect':
        x_filt = np.concatenate([x[:N], x], axis=0)
        trim = N
    elif bctype == 'zeropad':
        x_filt = np.concatenate([np.zeros((N, x.shape[1])), x], axis=0)
        trim = N
    else:  # 'none'
        x_filt = x
        trim = 0

    out = np.zeros_like(x_filt)
    for j in range(x_filt.shape[1]):
        out[:, j] = np.convolve(x_filt[:, j], kern, mode='same')
    out = out[trim:]

    if was_1d:
        out = out[:, 0]
    return out


# ============================================================
# DATA LOADING
# ============================================================

def _load_v5_session(data_path):
    """Load a MATLAB v5 format session file and convert to mat73-like dict structure."""
    import scipy.io
    raw = scipy.io.loadmat(data_path, squeeze_me=False)
    o = raw['obj'][0, 0]

    obj = {}

    # bp (behavioral data)
    bp_raw = o['bp'][0, 0]
    bp = {}
    for field in bp_raw.dtype.names:
        val = bp_raw[field]
        if field == 'ev':
            ev_raw = val[0, 0]
            ev = {}
            for ef in ev_raw.dtype.names:
                ev_val = ev_raw[ef]
                if ev_val.dtype == object:
                    # Cell array (e.g., lickL, lickR)
                    ev[ef] = [ev_val.flat[i] for i in range(ev_val.size)]
                else:
                    ev[ef] = ev_val.flatten()
            bp['ev'] = ev
        elif field == 'stim':
            stim_raw = val[0, 0] if val.size > 0 and val.dtype.names else None
            if stim_raw is not None and stim_raw.dtype.names:
                stim = {}
                for sf in stim_raw.dtype.names:
                    sv = stim_raw[sf]
                    if sv.dtype == object:
                        stim[sf] = sv
                    else:
                        stim[sf] = sv.flatten()
                bp['stim'] = stim
            else:
                bp['stim'] = {'enable': np.zeros(int(bp_raw['Ntrials'].flat[0]))}
        elif val.dtype == object:
            bp[field] = val
        else:
            try:
                if val.size == 1:
                    bp[field] = float(val.flat[0])
                else:
                    bp[field] = val.flatten()
            except (TypeError, ValueError):
                bp[field] = val
    obj['bp'] = bp

    # clu (cluster data)
    clu_raw = o['clu']
    n_probes = clu_raw.shape[1] if clu_raw.ndim >= 2 else 1
    clu_list = []
    for p in range(n_probes):
        probe_raw = clu_raw[0, p] if clu_raw.ndim >= 2 else clu_raw[0]
        if probe_raw is None or probe_raw.size == 0:
            clu_list.append(None)
            continue
        n_clusters = probe_raw.shape[1] if probe_raw.ndim >= 2 else probe_raw.shape[0]
        probe = {}
        for field in probe_raw.dtype.names:
            vals = []
            for c in range(n_clusters):
                v = probe_raw[0, c][field] if probe_raw.ndim >= 2 else probe_raw[c][field]
                if field == 'quality':
                    vals.append(str(v.flat[0]).strip() if v.size > 0 else '')
                elif field in ('tm', 'trialtm', 'trial'):
                    vals.append(v.flatten())
                elif field == 'site' or field == 'channel':
                    vals.append(v.flatten() if v.size > 0 else np.array([]))
                else:
                    vals.append(v)
            probe[field] = vals
        clu_list.append(probe)
    obj['clu'] = clu_list

    # sglx
    sglx_raw = o['sglx'][0, 0]
    sglx = {}
    for field in sglx_raw.dtype.names:
        val = sglx_raw[field]
        if field == 'bitcode':
            bc_raw = val[0, 0]
            bc = {}
            for bf in bc_raw.dtype.names:
                bv = bc_raw[bf]
                if bv.dtype == object:
                    bc[bf] = bv
                else:
                    bc[bf] = bv.flatten()
            sglx['bitcode'] = bc
        elif field == 'fs':
            sglx['fs'] = float(val.flat[0])
        else:
            sglx[field] = val
    obj['sglx'] = sglx

    # traj
    traj_raw = o['traj']
    n_views = traj_raw.shape[1] if traj_raw.ndim >= 2 else traj_raw.shape[0]
    traj_list = []
    for v in range(n_views):
        view_raw = traj_raw[0, v] if traj_raw.ndim >= 2 else traj_raw[v]
        if view_raw is None or view_raw.size == 0:
            traj_list.append(None)
            continue
        n_trials_traj = view_raw.shape[1] if view_raw.ndim >= 2 else view_raw.shape[0]
        view = {}
        for field in view_raw.dtype.names:
            vals = []
            for t in range(n_trials_traj):
                trial_data = view_raw[0, t] if view_raw.ndim >= 2 else view_raw[t]
                v_data = trial_data[field]
                if field == 'featNames':
                    if v_data.dtype == object:
                        names = []
                        for fi in range(v_data.size):
                            name = v_data.flat[fi]
                            if isinstance(name, np.ndarray):
                                names.append([str(name.flat[0])] if name.size > 0 else [''])
                            else:
                                names.append([str(name)])
                        vals.append(names)
                    else:
                        vals.append(v_data)
                elif field == 'frameTimes':
                    vals.append(v_data.flatten())
                elif field == 'ts':
                    vals.append(v_data)
                elif field == 'fn':
                    vals.append(str(v_data.flat[0]) if v_data.size > 0 else '')
                else:
                    vals.append(v_data)
            view[field] = vals
        traj_list.append(view)
    obj['traj'] = traj_list

    # ex
    ex_raw = o['ex'][0, 0]
    ex = {}
    for field in ex_raw.dtype.names:
        val = ex_raw[field]
        if val.dtype.kind in ('U', 'S', 'O'):
            ex[field] = str(val.flat[0]) if val.size > 0 else ''
        else:
            ex[field] = val
    obj['ex'] = ex

    return obj


def load_session_data(anm, date, data_dir):
    """Load a session's data structure and motion energy."""
    import mat73
    import scipy.io

    data_path = os.path.join(DATA_ROOT, data_dir, f'data_structure_{anm}_{date}.mat')
    t0 = time.time()
    try:
        obj = mat73.loadmat(data_path)['obj']
    except TypeError:
        # MATLAB v5 format - use scipy and convert to similar structure
        obj = _load_v5_session(data_path)
    t_load = time.time() - t0

    # Load motion energy
    me_path = os.path.join(DATA_ROOT, data_dir, f'motionEnergy_{anm}_{date}.mat')
    me_raw = None
    me_thresh = None
    if os.path.exists(me_path):
        me_file = scipy.io.loadmat(me_path)
        me_var = me_file['me']

        if me_var.dtype.names:
            # Structured array (me is a struct with .data and .moveThresh)
            me_struct = me_var[0, 0]
            me_data = me_struct['data']
            # Handle nested struct: if me.data is itself a struct with .data field
            if me_data.dtype.names and 'data' in me_data.dtype.names:
                inner = me_data[0, 0]
                me_raw = inner['data']
                me_thresh = float(inner['moveThresh'].flat[0])
            else:
                me_raw = me_data
                me_thresh = float(me_struct['moveThresh'].flat[0])
        elif me_var.dtype == object:
            # Direct cell array of motion energy per trial (no struct wrapper)
            me_raw = me_var
            me_thresh = None  # No threshold available

    return obj, me_raw, me_thresh, t_load


def compute_video_offset(obj):
    """Compute video-neural offset matching findVideoOffset.m."""
    bitStart = scipy_stats.mode(np.array(obj['bp']['ev']['bitStart']).flatten(), keepdims=False).mode
    bc_bitstart = np.array(obj['sglx']['bitcode']['bitstart']).flatten()
    bc_mode = scipy_stats.mode(bc_bitstart, keepdims=False).mode
    fs = float(obj['sglx']['fs'])
    vidshift = bc_mode / fs - bitStart
    return vidshift


def get_valid_cluster_indices(clu_probe, excluded_qualities=EXCLUDED_QUALITIES):
    """Find valid cluster indices matching findClusters.m with quality='all'."""
    qualities = clu_probe['quality']
    valid = []
    for i, q in enumerate(qualities):
        if isinstance(q, str):
            q_stripped = q.strip()
        else:
            q_stripped = ''
        if q_stripped.lower() not in {e.lower() for e in excluded_qualities}:
            valid.append(i)
    return valid


def align_and_bin_spikes(clu_probe, cluster_indices, align_times, ntrials,
                          tmin=TMIN, tmax=TMAX, dt=DT, smooth_win=SMOOTH_WINDOW,
                          bctype=BC_TYPE):
    """Align spikes to event and bin into firing rates.

    Returns:
        trialdat: (n_time, n_neurons, n_trials) firing rate array
        time_axis: (n_time,) time axis
    """
    edges = np.arange(tmin, tmax + dt, dt)
    # Time axis: center of each bin (matching getSeq.m: edges + dt/2, exclude last)
    time_axis = edges[:-1] + dt / 2
    n_time = len(time_axis)
    n_neurons = len(cluster_indices)

    trialdat = np.zeros((n_time, n_neurons, ntrials), dtype=np.float32)

    for i, clu_idx in enumerate(cluster_indices):
        trial_arr = np.array(clu_probe['trial'][clu_idx]).flatten()
        trialtm_arr = np.array(clu_probe['trialtm'][clu_idx]).flatten()

        for j in range(ntrials):
            trial_num = j + 1  # MATLAB 1-indexed
            spk_mask = trial_arr == trial_num
            if not np.any(spk_mask):
                continue

            # Align: trialtm_aligned = trialtm - alignTime
            aligned = trialtm_arr[spk_mask] - align_times[j]

            # Bin
            counts, _ = np.histogram(aligned, bins=edges)

            # Convert to firing rate and smooth
            fr = counts.astype(np.float32) / dt
            fr_smooth = causal_gaussian_smooth(fr, smooth_win, bctype)
            trialdat[:, i, j] = fr_smooth

    return trialdat, time_axis


def remove_low_fr_clusters(trialdat, cluster_indices, low_fr=LOW_FR):
    """Remove clusters with mean FR < low_fr across all trials.
    Matches removeLowFRClusters.m: mean of mean across conditions."""
    # Mean FR across all time and trials
    mean_frs = np.nanmean(np.nanmean(trialdat, axis=2), axis=0)  # (n_neurons,)
    keep = mean_frs > low_fr
    return trialdat[:, keep, :], [cluster_indices[i] for i in range(len(cluster_indices)) if keep[i]], keep


def get_motion_energy_aligned(me_raw, obj, align_times, time_axis, vidshift, ntrials):
    """Align motion energy to neural time axis matching loadMotionEnergy.m."""
    me_aligned = np.zeros((len(time_axis), ntrials), dtype=np.float32)

    traj_view0 = obj['traj'][0]  # side cam

    for trix in range(ntrials):
        me_trial = np.array(me_raw[trix, 0]).flatten()

        try:
            ft = np.array(traj_view0['frameTimes'][trix]).flatten()
            old_time = ft - vidshift - align_times[trix]
        except:
            # Fallback: generate frame times at 400 Hz
            nframes = len(me_trial)
            ft = np.arange(1, nframes + 1) / VIDEO_FR
            old_time = ft - 0.5 - align_times[trix]

        # Interpolate to neural time axis
        if len(old_time) > 1 and len(me_trial) > 1:
            f_interp = interp1d(old_time, me_trial, kind='linear',
                               bounds_error=False, fill_value=np.nan)
            me_aligned[:, trix] = f_interp(time_axis)

    # Fill NaN with nearest (matching fillmissing(,'nearest'))
    for trix in range(ntrials):
        col = me_aligned[:, trix]
        nans = np.isnan(col)
        if nans.any() and not nans.all():
            valid_idx = np.where(~nans)[0]
            nan_idx = np.where(nans)[0]
            nearest = np.searchsorted(valid_idx, nan_idx).clip(0, len(valid_idx)-1)
            col[nans] = col[valid_idx[nearest]]
            me_aligned[:, trix] = col

    return me_aligned


def compute_tongue_velocity(obj, align_times, time_axis, vidshift, ntrials):
    """Compute tongue velocity from DLC bottom camera data.
    Per paper methods: tongue missing values are NOT filled with nearest.
    Velocity is 0 where tongue is not visible."""
    traj_bottom = obj['traj'][1]  # bottom cam
    feat_names_raw = traj_bottom['featNames'][0]
    feat_names = []
    for fn in feat_names_raw:
        if isinstance(fn, list):
            feat_names.append(fn[0] if fn else '')
        else:
            feat_names.append(str(fn))

    # Find tongue feature index (top_tongue)
    tongue_idx = None
    for i, name in enumerate(feat_names):
        if name == 'top_tongue':
            tongue_idx = i
            break
    if tongue_idx is None:
        for i, name in enumerate(feat_names):
            if 'tongue' in name.lower():
                tongue_idx = i
                break

    if tongue_idx is None:
        return np.full((len(time_axis), ntrials), np.nan, dtype=np.float32)

    tongue_vel = np.full((len(time_axis), ntrials), np.nan, dtype=np.float32)

    for trix in range(ntrials):
        ts = np.array(traj_bottom['ts'][trix])
        if ts.ndim < 3:
            continue

        x = ts[:, 0, tongue_idx].copy()
        y = ts[:, 1, tongue_idx].copy()

        # DO NOT fill NaN for tongue (per paper methods)
        # Compute velocity only where tongue is visible (not NaN)
        vx = np.full_like(x, np.nan)
        vy = np.full_like(y, np.nan)

        valid = ~np.isnan(x) & ~np.isnan(y)
        if np.sum(valid) >= 2:
            # Compute gradient only on valid segments
            vx_all = np.gradient(x) * VIDEO_FR
            vy_all = np.gradient(y) * VIDEO_FR
            speed = np.sqrt(vx_all**2 + vy_all**2)
            # Only keep speed where tongue is visible
            speed[~valid] = np.nan
        else:
            speed = np.full_like(x, np.nan)

        # Align to neural time axis
        try:
            ft = np.array(traj_bottom['frameTimes'][trix]).flatten()
            old_time = ft - vidshift - align_times[trix]
        except:
            nframes = ts.shape[0]
            ft = np.arange(1, nframes + 1) / VIDEO_FR
            old_time = ft - 0.5 - align_times[trix]

        if len(old_time) > 1:
            f_interp = interp1d(old_time, speed, kind='linear',
                               bounds_error=False, fill_value=np.nan)
            tongue_vel[:, trix] = f_interp(time_axis)

    # Set NaN to 0 (tongue not visible = no tongue movement)
    tongue_vel = np.nan_to_num(tongue_vel, nan=0.0)

    return tongue_vel


def compute_paw_velocity(obj, align_times, time_axis, vidshift, ntrials):
    """Compute paw velocity from DLC bottom camera data."""
    traj_bottom = obj['traj'][1]  # bottom cam
    feat_names_raw = traj_bottom['featNames'][0]
    feat_names = []
    for fn in feat_names_raw:
        if isinstance(fn, list):
            feat_names.append(fn[0] if fn else '')
        else:
            feat_names.append(str(fn))

    # Find paw indices (top_paw, bottom_paw)
    paw_indices = []
    for i, name in enumerate(feat_names):
        if 'paw' in name.lower():
            paw_indices.append(i)

    if not paw_indices:
        return np.zeros((len(time_axis), ntrials), dtype=np.float32)

    paw_vel = np.zeros((len(time_axis), ntrials), dtype=np.float32)

    for trix in range(ntrials):
        ts = np.array(traj_bottom['ts'][trix])
        if ts.ndim < 3:
            continue

        speeds = []
        for pidx in paw_indices:
            x = ts[:, 0, pidx].copy()
            y = ts[:, 1, pidx].copy()

            for arr in [x, y]:
                nans = np.isnan(arr)
                if nans.any() and not nans.all():
                    valid = np.where(~nans)[0]
                    nan_pos = np.where(nans)[0]
                    nearest = np.searchsorted(valid, nan_pos).clip(0, len(valid)-1)
                    arr[nans] = arr[valid[nearest]]

            vx = np.gradient(x) * VIDEO_FR
            vy = np.gradient(y) * VIDEO_FR
            speeds.append(np.sqrt(vx**2 + vy**2))

        # Average across paw features
        avg_speed = np.mean(speeds, axis=0)

        try:
            ft = np.array(traj_bottom['frameTimes'][trix]).flatten()
            old_time = ft - vidshift - align_times[trix]
        except:
            nframes = ts.shape[0]
            ft = np.arange(1, nframes + 1) / VIDEO_FR
            old_time = ft - 0.5 - align_times[trix]

        if len(old_time) > 1:
            f_interp = interp1d(old_time, avg_speed, kind='linear',
                               bounds_error=False, fill_value=np.nan)
            paw_vel[:, trix] = f_interp(time_axis)

    # Fill NaN
    for trix in range(ntrials):
        col = paw_vel[:, trix]
        nans = np.isnan(col)
        if nans.any() and not nans.all():
            valid_idx = np.where(~nans)[0]
            nan_idx = np.where(nans)[0]
            nearest = np.searchsorted(valid_idx, nan_idx).clip(0, len(valid_idx)-1)
            col[nans] = col[valid_idx[nearest]]

    return paw_vel


def discretize_per_session(data_2d, percentile=50):
    """Discretize time-varying data using per-session threshold.
    data_2d: (n_time, n_trials)
    Returns: (n_time, n_trials) binary array (0: < threshold, 1: >= threshold)
    """
    all_vals = data_2d[~np.isnan(data_2d)]
    if len(all_vals) == 0:
        return np.zeros_like(data_2d, dtype=np.int32)
    threshold = np.percentile(all_vals, percentile)
    # If threshold is 0 (e.g. speed data with many zeros), use a tiny positive value
    # so that exact zeros become "low" and any positive value is "high"
    if threshold == 0:
        threshold = np.finfo(np.float32).eps
    result = (data_2d >= threshold).astype(np.int32)
    # Replace any remaining NaN positions with 0
    result[np.isnan(data_2d)] = 0
    return result


# ============================================================
# MAIN PROCESSING
# ============================================================

def process_session(anm, date, probes, data_dir, show_processing=False, session_idx=0):
    """Process a single session and return formatted data."""

    t_start = time.time()

    # Load data
    obj, me_raw, me_thresh, t_load = load_session_data(anm, date, data_dir)
    print(f"  Loaded in {t_load:.1f}s")

    bp = obj['bp']
    ntrials_total = int(bp['Ntrials'])

    # Get event times
    ev = bp['ev']
    goCue = np.array(ev['goCue']).flatten()

    # Get trial masks
    R = np.array(bp['R']).flatten().astype(bool)
    L = np.array(bp['L']).flatten().astype(bool)
    hit = np.array(bp['hit']).flatten().astype(bool)
    miss = np.array(bp['miss']).flatten().astype(bool)
    no = np.array(bp['no']).flatten().astype(bool)
    autowater = np.array(bp['autowater']).flatten().astype(bool)
    early = np.array(bp['early']).flatten().astype(bool)

    stim_enable = np.zeros(ntrials_total, dtype=bool)
    stim = bp.get('stim')
    if stim is not None and isinstance(stim, dict):
        se = stim.get('enable')
        if se is not None:
            stim_enable = np.array(se).flatten().astype(bool)

    # Trial selection: exclude early, stim, and no-response (ignore) trials
    valid_mask = ~early & ~stim_enable & ~no
    # Also need a lick response (hit or miss)
    valid_mask = valid_mask & (hit | miss)

    valid_trials = np.where(valid_mask)[0]  # 0-indexed
    n_valid = len(valid_trials)

    if n_valid < 2:
        print(f"  WARNING: Only {n_valid} valid trials, skipping session")
        return None

    print(f"  Trials: {ntrials_total} total, {n_valid} valid ({n_valid/ntrials_total*100:.0f}%)")

    # Compute video offset
    vidshift = compute_video_offset(obj)

    # Get alignment times
    align_times_all = goCue  # for all trials

    # Process neural data from specified probes
    all_cluster_indices = []
    clu_data = obj['clu']

    for probe_num in probes:
        probe_idx = probe_num - 1  # 0-indexed
        clu_probe = clu_data[probe_idx]

        if clu_probe is None:
            continue

        if isinstance(clu_probe, dict):
            valid_clu = get_valid_cluster_indices(clu_probe)
            all_cluster_indices.append((probe_idx, valid_clu))
            print(f"  Probe {probe_num}: {len(clu_probe['quality'])} total, {len(valid_clu)} valid clusters")

    # Bin and smooth spikes for all valid clusters across probes
    t_bin_start = time.time()
    edges = np.arange(TMIN, TMAX + DT, DT)
    time_axis = edges[:-1] + DT / 2
    n_time = len(time_axis)

    # Count total neurons
    total_neurons = sum(len(v) for _, v in all_cluster_indices)

    # Process spikes for all neurons at once
    trialdat = np.zeros((n_time, total_neurons, ntrials_total), dtype=np.float32)

    neuron_offset = 0
    for probe_idx, valid_clu in all_cluster_indices:
        clu_probe = clu_data[probe_idx]

        for i, clu_idx in enumerate(valid_clu):
            trial_arr = np.array(clu_probe['trial'][clu_idx]).flatten()
            trialtm_arr = np.array(clu_probe['trialtm'][clu_idx]).flatten()

            for j in range(ntrials_total):
                trial_num = j + 1
                spk_mask = trial_arr == trial_num
                if not np.any(spk_mask):
                    continue

                aligned = trialtm_arr[spk_mask] - align_times_all[j]
                counts, _ = np.histogram(aligned, bins=edges)
                fr = counts.astype(np.float32) / DT
                trialdat[:, neuron_offset + i, j] = causal_gaussian_smooth(fr, SMOOTH_WINDOW, BC_TYPE)

        neuron_offset += len(valid_clu)

    t_bin = time.time() - t_bin_start
    print(f"  Spike binning: {t_bin:.1f}s for {total_neurons} neurons")

    # Remove low FR clusters
    trialdat, kept_mask = remove_low_fr_neurons(trialdat, LOW_FR)
    n_neurons_final = trialdat.shape[1]
    print(f"  After FR filter (>{LOW_FR} Hz): {n_neurons_final} neurons (removed {total_neurons - n_neurons_final})")

    if n_neurons_final < 10:
        print(f"  WARNING: Only {n_neurons_final} neurons after filtering, skipping session")
        return None

    # Extract valid trials only
    trialdat_valid = trialdat[:, :, valid_trials]  # (n_time, n_neurons, n_valid)

    # Process behavioral outputs
    # 1. Lick direction: R&hit or L&miss -> right(1), L&hit or R&miss -> left(0)
    lick_right = (R & hit) | (L & miss)
    lick_direction = lick_right[valid_trials].astype(np.int32)

    # 2. Context: autowater -> WC(0), ~autowater -> DR(1)
    context = (~autowater[valid_trials]).astype(np.int32)

    # 3. Outcome: hit -> correct(1), miss -> incorrect(0)
    outcome = hit[valid_trials].astype(np.int32)

    # Process behavioral time series
    t_behav_start = time.time()

    # 4. Tongue velocity
    tongue_vel = compute_tongue_velocity(obj, align_times_all, time_axis, vidshift, ntrials_total)
    tongue_vel_valid = tongue_vel[:, valid_trials]
    tongue_vel_disc = discretize_per_session(tongue_vel_valid)

    # 5. Paw velocity
    paw_vel = compute_paw_velocity(obj, align_times_all, time_axis, vidshift, ntrials_total)
    paw_vel_valid = paw_vel[:, valid_trials]
    paw_vel_disc = discretize_per_session(paw_vel_valid)

    # 6. Motion energy
    if me_raw is not None:
        me_aligned = get_motion_energy_aligned(me_raw, obj, align_times_all, time_axis, vidshift, ntrials_total)
        me_valid = me_aligned[:, valid_trials]
        me_disc = discretize_per_session(me_valid)
    else:
        me_disc = np.zeros((n_time, n_valid), dtype=np.int32)

    t_behav = time.time() - t_behav_start
    print(f"  Behavioral processing: {t_behav:.1f}s")

    # Build trial-level data structures
    neural_trials = []
    input_trials = []
    output_trials = []

    for t_idx in range(n_valid):
        # Neural: (n_neurons, n_time)
        neural_trials.append(trialdat_valid[:, :, t_idx].T.astype(np.float32))

        # Input: time from goCue in seconds (1, n_time)
        input_trials.append(time_axis.reshape(1, -1).astype(np.float32))

        # Output: (n_output, n_time) for time-varying, (n_output,) for per-trial
        # Outputs 0-2 are per-trial, 3-5 are time-varying
        out = np.zeros((6, n_time), dtype=np.int32)
        out[0, :] = lick_direction[t_idx]  # broadcast per-trial to time
        out[1, :] = context[t_idx]
        out[2, :] = outcome[t_idx]
        out[3, :] = tongue_vel_disc[:, t_idx]
        out[4, :] = paw_vel_disc[:, t_idx]
        out[5, :] = me_disc[:, t_idx]
        output_trials.append(out)

    # Show processing plots
    if show_processing and n_valid > 0:
        plot_processing(anm, date, session_idx, time_axis, trialdat_valid,
                       tongue_vel_valid, paw_vel_valid,
                       me_valid if me_raw is not None else None,
                       tongue_vel_disc, paw_vel_disc, me_disc,
                       lick_direction, context, outcome, valid_trials, align_times_all)

    t_total = time.time() - t_start
    print(f"  Total: {t_total:.1f}s, {n_valid} trials, {n_neurons_final} neurons")

    return {
        'neural': neural_trials,
        'input': input_trials,
        'output': output_trials,
        'animal': anm,
        'n_neurons': n_neurons_final,
        'n_trials': n_valid,
    }


def remove_low_fr_neurons(trialdat, low_fr):
    """Remove neurons with mean FR below threshold.
    trialdat: (n_time, n_neurons, n_trials)
    """
    mean_fr = np.nanmean(np.nanmean(trialdat, axis=2), axis=0)
    keep = mean_fr > low_fr
    return trialdat[:, keep, :], keep


def plot_processing(anm, date, session_idx, time_axis, trialdat,
                   tongue_vel, paw_vel, me_aligned,
                   tongue_disc, paw_disc, me_disc,
                   lick_dir, context, outcome, valid_trials, align_times):
    """Plot processing steps for visual inspection."""
    fig, axes = plt.subplots(4, 3, figsize=(18, 16))
    fig.suptitle(f'Processing: {anm}_{date}', fontsize=14)

    n_trials_plot = min(5, trialdat.shape[2])

    # Row 0: Neural activity (few neurons, few trials)
    ax = axes[0, 0]
    n_neurons_plot = min(5, trialdat.shape[1])
    for n in range(n_neurons_plot):
        ax.plot(time_axis, trialdat[:, n, 0], alpha=0.7, label=f'N{n}')
    ax.axvline(0, color='k', linestyle='--', alpha=0.5)
    ax.set_title('Neural (trial 0)')
    ax.set_ylabel('FR (spk/s)')
    ax.legend(fontsize=6)

    ax = axes[0, 1]
    # Trial-averaged for one neuron
    ax.plot(time_axis, np.mean(trialdat[:, 0, :], axis=1), 'k-')
    ax.axvline(0, color='r', linestyle='--')
    ax.set_title(f'Neuron 0 mean (n={trialdat.shape[2]} trials)')

    ax = axes[0, 2]
    ax.bar(['Left', 'Right'], [np.sum(lick_dir == 0), np.sum(lick_dir == 1)])
    ax.set_title('Lick direction')

    # Row 1: Tongue velocity
    ax = axes[1, 0]
    for t in range(n_trials_plot):
        ax.plot(time_axis, tongue_vel[:, t], alpha=0.5)
    ax.axvline(0, color='k', linestyle='--')
    ax.set_title('Tongue velocity (raw)')

    ax = axes[1, 1]
    for t in range(n_trials_plot):
        ax.plot(time_axis, tongue_disc[:, t], alpha=0.5)
    ax.set_title('Tongue velocity (discretized)')

    ax = axes[1, 2]
    ax.bar(['DR', 'WC'], [np.sum(context == 1), np.sum(context == 0)])
    ax.set_title('Context')

    # Row 2: Paw velocity
    ax = axes[2, 0]
    for t in range(n_trials_plot):
        ax.plot(time_axis, paw_vel[:, t], alpha=0.5)
    ax.axvline(0, color='k', linestyle='--')
    ax.set_title('Paw velocity (raw)')

    ax = axes[2, 1]
    for t in range(n_trials_plot):
        ax.plot(time_axis, paw_disc[:, t], alpha=0.5)
    ax.set_title('Paw velocity (discretized)')

    ax = axes[2, 2]
    ax.bar(['Incorrect', 'Correct'], [np.sum(outcome == 0), np.sum(outcome == 1)])
    ax.set_title('Outcome')

    # Row 3: Motion energy
    ax = axes[3, 0]
    if me_aligned is not None:
        for t in range(n_trials_plot):
            ax.plot(time_axis, me_aligned[:, t], alpha=0.5)
    ax.axvline(0, color='k', linestyle='--')
    ax.set_title('Motion energy (raw)')

    ax = axes[3, 1]
    for t in range(n_trials_plot):
        ax.plot(time_axis, me_disc[:, t], alpha=0.5)
    ax.set_title('Motion energy (discretized)')

    ax = axes[3, 2]
    ax.text(0.5, 0.5, f'N neurons: {trialdat.shape[1]}\nN trials: {trialdat.shape[2]}',
            transform=ax.transAxes, ha='center', va='center', fontsize=12)
    ax.set_title('Summary')

    for ax_row in axes:
        for ax in ax_row:
            ax.set_xlabel('Time from goCue (s)')

    fig.tight_layout()
    fig.savefig(f'processing_{anm}_{date}.png', dpi=100)
    plt.close(fig)
    print(f"  Saved processing plot: processing_{anm}_{date}.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=str, help='Output pickle file')
    parser.add_argument('--sample', action='store_true', help='Process only 2 sessions')
    parser.add_argument('--full', action='store_true', help='Process all sessions (default)')
    parser.add_argument('--show-processing', action='store_true', help='Save processing plots')
    args = parser.parse_args()

    if args.sample:
        sessions = ALL_SESSIONS[:2]
        print(f"SAMPLE MODE: Processing {len(sessions)} sessions")
    else:
        sessions = ALL_SESSIONS
        print(f"FULL MODE: Processing {len(sessions)} sessions")

    show_processing = args.show_processing
    if show_processing and not args.sample:
        # Only show processing for first 2 sessions in full mode
        show_limit = 2
    else:
        show_limit = len(sessions)

    # Collect all session data
    all_neural = []
    all_input = []
    all_output = []
    all_animals = []
    all_brain_region_idx = []
    session_info = []

    subjects_set = set()

    t_total_start = time.time()

    for i, (anm, date, probes, data_dir) in enumerate(sessions):
        print(f"\n[{i+1}/{len(sessions)}] Processing {anm}_{date} (probes {probes})...")

        result = process_session(anm, date, probes, data_dir,
                                show_processing=(show_processing and i < show_limit),
                                session_idx=i)

        if result is None:
            continue

        all_neural.append(result['neural'])
        all_input.append(result['input'])
        all_output.append(result['output'])
        all_animals.append(anm)
        subjects_set.add(anm)

        # Brain region: all neurons are ALM
        all_brain_region_idx.append(np.zeros(result['n_neurons'], dtype=np.int64))

        session_info.append({
            'animal': anm,
            'date': date,
            'data_dir': data_dir,
            'probes': probes,
            'n_neurons': result['n_neurons'],
            'n_trials': result['n_trials'],
        })

    t_total = time.time() - t_total_start
    print(f"\n{'='*60}")
    print(f"Processing complete: {len(all_neural)} sessions in {t_total:.1f}s")

    # Build subjects list
    subjects = sorted(subjects_set)
    subject_idx = np.array([subjects.index(anm) for anm in all_animals], dtype=np.int64)

    # Build final data dict
    data = {
        'neural': all_neural,
        'input': all_input,
        'output': all_output,

        'subjects': subjects,
        'subject_idx': subject_idx,

        'brain_regions': ['ALM'],
        'brain_region_idx': all_brain_region_idx,

        'input_names': ['time_from_gocue'],
        'output_names': ['lick_direction', 'context', 'outcome', 'tongue_velocity', 'paw_velocity', 'motion_energy'],
        'output_values': [
            ['left', 'right'],           # lick_direction: 0=left, 1=right
            ['WC', 'DR'],                # context: 0=WC, 1=DR
            ['incorrect', 'correct'],    # outcome: 0=incorrect, 1=correct
            ['low', 'high'],             # tongue_velocity: 0=<50th, 1=>=50th
            ['low', 'high'],             # paw_velocity: 0=<50th, 1=>=50th
            ['low', 'high'],             # motion_energy: 0=<50th, 1=>=50th
        ],

        'metadata': {
            'task_description': 'Two-context delayed-response (DR) and water-cued (WC) directional licking task with ALM recordings',
            'time_bin_size': DT * 1000,  # 10 ms
            'temporal_alignment_event': 'Go cue onset',
            'off_start': TMIN,  # -2.5 s
            'off_end': TMAX,    # 2.5 s
            'align_event': ALIGN_EVENT,
            'smoothing': f'Causal Gaussian, window={SMOOTH_WINDOW} bins',
            'fr_threshold': LOW_FR,
            'session_info': session_info,
            'n_sessions': len(all_neural),
            'total_neurons': sum(s['n_neurons'] for s in session_info),
            'total_trials': sum(s['n_trials'] for s in session_info),
        }
    }

    # Print summary
    print(f"\nDataset Summary:")
    print(f"  Sessions: {len(all_neural)}")
    print(f"  Subjects: {len(subjects)} ({', '.join(subjects)})")
    print(f"  Total neurons: {data['metadata']['total_neurons']}")
    print(f"  Total trials: {data['metadata']['total_trials']}")
    print(f"  Time bins: {len(np.arange(TMIN, TMAX + DT, DT)) - 1}")
    print(f"  Outputs: {data['output_names']}")

    # Save
    print(f"\nSaving to {args.output}...")
    with open(args.output, 'wb') as f:
        pickle.dump(data, f, protocol=4)

    fsize = os.path.getsize(args.output) / (1024**2)
    print(f"Saved: {fsize:.1f} MB")


if __name__ == '__main__':
    main()
