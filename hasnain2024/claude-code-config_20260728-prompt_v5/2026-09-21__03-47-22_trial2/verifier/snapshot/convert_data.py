#!/usr/bin/env python3
"""
Convert neural + behavioral data from Hasnain, Birnbaum et al. (2025)
"Separating cognitive and motor processes in the behaving mouse"
into decoder-compatible format.

Usage:
    python -u convert_data.py <outpicklefile> [--sample] [--full] [--show-processing]
"""

import argparse
import os
import sys
import time
import pickle
import numpy as np
import h5py
import scipy.io
import scipy.signal
from scipy import stats as scipy_stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Parameters (matching reference code defaults)
# ============================================================
DT = 0.01            # 10 ms time bins (params.dt = 1/100)
TMIN = -2.5           # seconds before go cue
TMAX = 2.5            # seconds after go cue
SMOOTH_WIN = 15       # causal Gaussian window size (samples)
LOW_FR_THRESH = 1.0   # Hz, minimum mean firing rate
MIN_UNITS = 10        # minimum units per session
DLC_CONF_THRESH = 0.5 # DLC confidence threshold for visibility
VIDEO_FPS = 400       # video frame rate

# Quality labels to EXCLUDE (matching findClusters 'all')
EXCLUDE_QUALITY = {'garbage', 'noisy', 'gabrga', 'real?', ''}

# Time bin edges and centers
EDGES = np.arange(TMIN, TMAX + DT, DT)
TIME_CENTERS = EDGES[:-1] + DT / 2
N_TIMEBINS = len(TIME_CENTERS)

# Data directories
DATA_DIRS = {
    'Ephys': '/app/data/Ephys_Behavior',
    'RandDelay': '/app/data/RandomizedDelay_Ephys_Behavior',
}

# Duplicate files to skip (same session data saved twice with different dates/formats)
SKIP_FILES = {
    'data_structure_JEB23_2023-10-20.mat',  # duplicate of JEB23_2023-10-19
}


# ============================================================
# Causal Gaussian smoothing (matching mySmooth.m)
# ============================================================
def causal_gaussian_kernel(N):
    """Create causal Gaussian kernel matching MATLAB gausswin + causal zeroing."""
    if N <= 1:
        return np.array([1.0])
    # MATLAB gausswin(N) = exp(-0.5 * ((n - (N-1)/2) / (alpha * (N-1)/2))^2)
    # where alpha = 2.5 by default
    alpha = 2.5
    n = np.arange(N)
    w = np.exp(-0.5 * ((n - (N - 1) / 2) / (alpha * (N - 1) / 2)) ** 2)
    # Zero out first half (causal)
    w[:N // 2] = 0
    # Normalize
    w = w / w.sum()
    return w

KERNEL = causal_gaussian_kernel(SMOOTH_WIN)


def smooth_causal(x, kernel=KERNEL, bctype='reflect'):
    """Apply causal Gaussian smoothing with boundary handling.

    Matches mySmooth.m: prepend reflected/zero data, convolve 'same', trim.
    x: (n_time,) or (n_time, n_cols) array
    """
    if len(kernel) <= 1:
        return x

    was_1d = x.ndim == 1
    if was_1d:
        x = x[:, np.newaxis]

    N = len(kernel)
    if bctype == 'reflect':
        pad = x[:N, :]
        x_filt = np.concatenate([pad, x], axis=0)
        trim = N
    elif bctype == 'zeropad':
        pad = np.zeros((N, x.shape[1]))
        x_filt = np.concatenate([pad, x], axis=0)
        trim = N
    else:
        x_filt = x
        trim = 0

    out = np.zeros_like(x_filt)
    for j in range(x_filt.shape[1]):
        out[:, j] = np.convolve(x_filt[:, j], kernel, mode='same')
    out = out[trim:, :]

    if was_1d:
        out = out[:, 0]
    return out


# ============================================================
# Data loading helpers
# ============================================================
def read_h5_string(f, ref):
    """Read a string stored as uint16 array via h5py reference."""
    data = f[ref][:]
    return ''.join(chr(c) for c in data.flatten())


def load_session_h5(fpath):
    """Load session data from HDF5 MATLAB v7.3 file."""
    session = {}
    with h5py.File(fpath, 'r') as f:
        obj = f['obj']

        # ----- Behavioral data -----
        bp = obj['bp']
        session['Ntrials'] = int(np.array(bp['Ntrials']).flatten()[0])
        session['hit'] = np.array(bp['hit']).flatten().astype(bool)
        session['miss'] = np.array(bp['miss']).flatten().astype(bool)
        session['no'] = np.array(bp['no']).flatten().astype(bool)
        session['early'] = np.array(bp['early']).flatten().astype(bool)
        session['autowater'] = np.array(bp['autowater']).flatten().astype(int)
        session['L'] = np.array(bp['L']).flatten().astype(bool)
        session['R'] = np.array(bp['R']).flatten().astype(bool)

        # Stim
        session['stim_enable'] = np.array(bp['stim']['enable']).flatten().astype(bool)

        # Event times
        ev = bp['ev']
        session['goCue'] = np.array(ev['goCue']).flatten()
        session['sample'] = np.array(ev['sample']).flatten()
        session['delay'] = np.array(ev['delay']).flatten()

        # Lick times (cell arrays of variable-length arrays)
        n_trials = session['Ntrials']
        session['lickL'] = []
        session['lickR'] = []
        lickL_ds = ev['lickL']
        lickR_ds = ev['lickR']
        for t in range(n_trials):
            try:
                ref = lickL_ds[0, t] if lickL_ds.ndim > 1 else lickL_ds[t]
                licks = np.array(f[ref]).flatten()
                session['lickL'].append(licks)
            except:
                session['lickL'].append(np.array([]))
            try:
                ref = lickR_ds[0, t] if lickR_ds.ndim > 1 else lickR_ds[t]
                licks = np.array(f[ref]).flatten()
                session['lickR'].append(licks)
            except:
                session['lickR'].append(np.array([]))

        # ----- Neural data -----
        if 'clu' not in obj:
            session['neurons'] = []
            return session

        clu_ds = obj['clu']
        neurons = []
        for pi in range(clu_ds.shape[0]):
            for pj in range(clu_ds.shape[1]):
                ref = clu_ds[pi, pj]
                dereffed = f[ref]
                if not isinstance(dereffed, h5py.Group) or 'quality' not in dereffed:
                    continue

                quality_ds = dereffed['quality']
                n_neurons = quality_ds.shape[0]

                for ni in range(n_neurons):
                    # Read quality string
                    q_ref = quality_ds[ni, 0]
                    quality = read_h5_string(f, q_ref).strip().lower()

                    if quality in EXCLUDE_QUALITY:
                        continue

                    # Read spike data
                    tm_ref = dereffed['tm'][ni, 0]
                    trialtm_ref = dereffed['trialtm'][ni, 0]
                    trial_ref = dereffed['trial'][ni, 0]

                    tm = np.array(f[tm_ref]).flatten()
                    trialtm = np.array(f[trialtm_ref]).flatten()
                    trial = np.array(f[trial_ref]).flatten().astype(int)

                    neurons.append({
                        'quality': quality,
                        'trialtm': trialtm,
                        'trial': trial,
                    })

        session['neurons'] = neurons

        # ----- Video tracking data -----
        if 'traj' in obj:
            traj_ds = obj['traj']
            session['traj'] = []
            for cam_idx in range(traj_ds.shape[0]):
                cam_ref = traj_ds[cam_idx, 0]
                cam = f[cam_ref]

                cam_data = {'ts': [], 'frameTimes': [], 'featNames': []}

                # Feature names (from first trial)
                feat_refs = cam['featNames']
                feat_ref = feat_refs[0, 0]
                feat_data = f[feat_ref]
                n_features = feat_data.shape[1] if feat_data.ndim > 1 else feat_data.shape[0]
                names = []
                for i in range(n_features):
                    idx = (0, i) if feat_data.ndim > 1 else (i,)
                    name_ref = feat_data[idx]
                    name = read_h5_string(f, name_ref)
                    names.append(name)
                cam_data['featNames'] = names

                # Per-trial ts and frameTimes
                ts_ds = cam['ts']
                ft_ds = cam['frameTimes']

                for t in range(n_trials):
                    try:
                        ts_ref = ts_ds[t, 0]
                        ts = np.array(f[ts_ref])  # (nFeatures, 3, nFrames)
                        cam_data['ts'].append(ts)
                    except:
                        cam_data['ts'].append(None)

                    try:
                        ft_ref = ft_ds[t, 0]
                        ft = np.array(f[ft_ref]).flatten()
                        cam_data['frameTimes'].append(ft)
                    except:
                        cam_data['frameTimes'].append(None)

                session['traj'].append(cam_data)

        # ----- Video offset computation -----
        try:
            sglx = obj['sglx']
            bp_bitstart = np.array(ev['bitStart']).flatten()
            sglx_bitstart = np.array(sglx['bitcode']['bitstart']).flatten()
            sglx_fs = np.array(sglx['fs']).flatten()[0]

            mode_bp = scipy_stats.mode(bp_bitstart[~np.isnan(bp_bitstart)], keepdims=False).mode
            mode_sglx = scipy_stats.mode(sglx_bitstart[~np.isnan(sglx_bitstart)], keepdims=False).mode
            session['vidshift'] = mode_sglx / sglx_fs - mode_bp
        except:
            session['vidshift'] = 0.5  # default padSec

        # ----- Motion energy (if embedded in obj.me) -----
        if 'me' in obj:
            me_ds = obj['me']
            me_data = []
            if isinstance(me_ds, h5py.Dataset):
                # Array of refs
                for t in range(n_trials):
                    try:
                        ref = me_ds[0, t] if me_ds.ndim > 1 else me_ds[t]
                        me_trial = np.array(f[ref]).flatten()
                        me_data.append(me_trial)
                    except:
                        me_data.append(None)
            session['me_embedded'] = me_data

    return session


def _has_field(obj, name):
    """Check if numpy.void or structured array has a named field."""
    if isinstance(obj, np.void) and obj.dtype.names is not None:
        return name in obj.dtype.names
    if isinstance(obj, dict):
        return name in obj
    return False


def load_session_scipy(fpath):
    """Load session data from older MATLAB format file."""
    d = scipy.io.loadmat(fpath, simplify_cells=False)
    obj = d['obj'][0, 0]  # numpy.void
    bp = obj['bp'][0, 0]  # numpy.void

    session = {}
    n_trials = int(np.array(bp['Ntrials']).flatten()[0])
    session['Ntrials'] = n_trials
    session['hit'] = np.array(bp['hit']).flatten().astype(bool)
    session['miss'] = np.array(bp['miss']).flatten().astype(bool)
    session['no'] = np.array(bp['no']).flatten().astype(bool)
    session['early'] = np.array(bp['early']).flatten().astype(bool)
    session['autowater'] = np.array(bp['autowater']).flatten().astype(int)
    session['L'] = np.array(bp['L']).flatten().astype(bool)
    session['R'] = np.array(bp['R']).flatten().astype(bool)

    stim = bp['stim'][0, 0]
    session['stim_enable'] = np.array(stim['enable']).flatten().astype(bool)

    ev = bp['ev'][0, 0]
    session['goCue'] = np.array(ev['goCue']).flatten()
    session['sample'] = np.array(ev['sample']).flatten()
    session['delay'] = np.array(ev['delay']).flatten()

    # Lick times
    for side in ['lickL', 'lickR']:
        lick_list = []
        if _has_field(ev, side):
            lick_data = ev[side].flatten()
            for t in range(n_trials):
                try:
                    licks = np.array(lick_data[t]).flatten() if t < len(lick_data) else np.array([])
                    licks = licks[~np.isnan(licks)] if len(licks) > 0 else np.array([])
                    lick_list.append(licks)
                except:
                    lick_list.append(np.array([]))
        else:
            lick_list = [np.array([])] * n_trials
        session[side] = lick_list

    # Neural data - clu is (1, nProbes) object array, each element is structured array (nNeurons,)
    neurons = []
    if _has_field(obj, 'clu'):
        clu_raw = obj['clu']
        for probe_data in clu_raw.flatten():
            if probe_data is None or (isinstance(probe_data, np.ndarray) and probe_data.size == 0):
                continue
            # probe_data: structured array of neurons with fields (tm, quality, trialtm, trial, ...)
            if hasattr(probe_data, 'dtype') and probe_data.dtype.names is not None:
                neuron_arr = probe_data.flatten()
                for ni in range(len(neuron_arr)):
                    neuron = neuron_arr[ni]
                    q_raw = neuron['quality']
                    quality = str(np.array(q_raw).flat[0]).strip().lower() if np.array(q_raw).size > 0 else ''
                    if quality in EXCLUDE_QUALITY:
                        continue
                    trialtm = np.array(neuron['trialtm']).flatten()
                    trial = np.array(neuron['trial']).flatten().astype(int)
                    neurons.append({'quality': quality, 'trialtm': trialtm, 'trial': trial})

    session['neurons'] = neurons

    # Video tracking - traj is (1, nCameras) object array
    if _has_field(obj, 'traj'):
        session['traj'] = []
        traj_raw = obj['traj']
        for cam_idx in range(traj_raw.size):
            cam_arr = traj_raw.flat[cam_idx]
            cam_out = {'ts': [], 'frameTimes': [], 'featNames': []}

            if cam_arr is None or (isinstance(cam_arr, np.ndarray) and cam_arr.size == 0):
                session['traj'].append(cam_out)
                continue

            # cam_arr: structured array (1, nTrials) with fields per trial
            if hasattr(cam_arr, 'dtype') and cam_arr.dtype.names is not None:
                cam_flat = cam_arr.flatten()

                # Feature names from first trial
                for t_check in range(min(len(cam_flat), n_trials)):
                    fn_field = cam_flat[t_check]['featNames']
                    if fn_field is not None and np.array(fn_field).size > 0:
                        fn_flat = np.array(fn_field).flatten()
                        names = [str(np.array(fn_flat[i]).flat[0]).strip()
                                if np.array(fn_flat[i]).size > 0 else ''
                                for i in range(len(fn_flat))]
                        cam_out['featNames'] = names
                        break

                # Per-trial ts and frameTimes
                for t in range(min(len(cam_flat), n_trials)):
                    trial_struct = cam_flat[t]

                    # ts: (nFrames, 3, nFeatures) -> transpose to (nFeatures, 3, nFrames)
                    ts_val = trial_struct['ts']
                    if isinstance(ts_val, np.ndarray) and ts_val.ndim == 3 and ts_val.size > 0:
                        cam_out['ts'].append(ts_val.transpose(2, 1, 0))
                    else:
                        cam_out['ts'].append(None)

                    ft_val = trial_struct['frameTimes']
                    if isinstance(ft_val, np.ndarray) and ft_val.size > 0:
                        cam_out['frameTimes'].append(ft_val.flatten())
                    else:
                        cam_out['frameTimes'].append(None)

                # Pad if fewer trials in traj
                while len(cam_out['ts']) < n_trials:
                    cam_out['ts'].append(None)
                    cam_out['frameTimes'].append(None)

            session['traj'].append(cam_out)

    # Video offset
    try:
        sglx = obj['sglx'][0, 0]
        bp_bitstart = np.array(ev['bitStart']).flatten()
        bc = sglx['bitcode'][0, 0]
        sglx_bitstart = np.array(bc['bitstart']).flatten()
        sglx_fs = float(np.array(sglx['fs']).flatten()[0])
        mode_bp = scipy_stats.mode(bp_bitstart[~np.isnan(bp_bitstart)], keepdims=False).mode
        mode_sglx = scipy_stats.mode(sglx_bitstart[~np.isnan(sglx_bitstart)], keepdims=False).mode
        session['vidshift'] = mode_sglx / sglx_fs - mode_bp
    except:
        session['vidshift'] = 0.5

    # Motion energy (embedded in obj.me)
    if _has_field(obj, 'me'):
        me_raw = obj['me']
        me_data = []
        me_flat = me_raw.flatten()
        for t in range(n_trials):
            try:
                me_trial = np.array(me_flat[t]).flatten() if t < len(me_flat) else None
                me_data.append(me_trial)
            except:
                me_data.append(None)
        session['me_embedded'] = me_data

    return session


def load_motion_energy_file(me_path):
    """Load motion energy from separate .mat file."""
    d = scipy.io.loadmat(me_path, simplify_cells=True)
    me = d['me']

    # Format 1: struct with 'data' and 'moveThresh' fields
    if isinstance(me, dict) and 'data' in me:
        data = me['data']
        thresh = me.get('moveThresh', 10)
    elif isinstance(me, np.ndarray) and me.dtype == object:
        # Format 2: just an array of per-trial arrays (no struct wrapper)
        data = me
        thresh = 10  # default
    else:
        return [], 10.0

    me_trials = []
    if isinstance(data, list):
        for trial_data in data:
            me_trials.append(np.array(trial_data).flatten() if trial_data is not None else None)
    elif isinstance(data, np.ndarray):
        if data.dtype == object:
            for trial_data in data.flatten():
                me_trials.append(np.array(trial_data).flatten() if trial_data is not None else None)
        else:
            me_trials.append(data.flatten())

    return me_trials, float(thresh)


def load_session(fpath):
    """Load session from either h5py or scipy format."""
    try:
        session = load_session_h5(fpath)
        return session
    except:
        pass
    try:
        session = load_session_scipy(fpath)
        return session
    except Exception as e:
        print(f"  ERROR loading {fpath}: {e}")
        return None


# ============================================================
# Processing functions
# ============================================================
def align_and_bin_spikes(neurons, goCue, n_trials, edges, dt):
    """Align spikes to go cue and bin into time bins.

    Returns: (n_neurons, n_timebins, n_trials) array of firing rates in Hz.
    """
    n_timebins = len(edges) - 1
    n_neurons_total = len(neurons)

    # Pre-allocate
    trialdat = np.zeros((n_neurons_total, n_timebins, n_trials), dtype=np.float32)

    for ni, neuron in enumerate(neurons):
        trialtm = neuron['trialtm']
        trial = neuron['trial']

        for t in range(n_trials):
            trial_num = t + 1  # MATLAB 1-indexed
            gc = goCue[t]
            if np.isnan(gc):
                continue

            # Get spikes for this trial
            spike_mask = trial == trial_num
            if not np.any(spike_mask):
                continue

            # Align to go cue
            aligned = trialtm[spike_mask] - gc

            # Bin spikes
            counts, _ = np.histogram(aligned, bins=edges)

            # Convert to firing rate and smooth
            fr = counts.astype(np.float32) / dt
            fr = smooth_causal(fr, KERNEL, 'reflect')
            trialdat[ni, :, t] = fr

    return trialdat


def compute_mean_fr(trialdat):
    """Compute mean firing rate per neuron across all trials and time bins."""
    # trialdat: (n_neurons, n_timebins, n_trials)
    return np.mean(np.mean(trialdat, axis=2), axis=1)


def find_feature_index(feat_names, target):
    """Find index of target feature in feature name list."""
    target_lower = target.lower()
    for i, name in enumerate(feat_names):
        if name.lower().strip() == target_lower:
            return i
    return None


def compute_velocity_timeseries(session, cam_idx, feat_name, trial_indices, vidshift):
    """Compute velocity for a DLC feature, aligned to neural time bins.

    Returns: (n_timebins, n_trials) arrays of velocity and visibility.
    """
    n_trials = len(trial_indices)
    velocity = np.full((N_TIMEBINS, n_trials), np.nan, dtype=np.float32)
    visible = np.zeros((N_TIMEBINS, n_trials), dtype=bool)

    if 'traj' not in session or cam_idx >= len(session['traj']):
        return velocity, visible

    cam = session['traj'][cam_idx]
    feat_idx = find_feature_index(cam['featNames'], feat_name)
    if feat_idx is None:
        return velocity, visible

    goCue = session['goCue']
    taxis = TIME_CENTERS

    for out_idx, t in enumerate(trial_indices):
        ts = cam['ts'][t]
        ft = cam['frameTimes'][t]
        gc = goCue[t]

        if ts is None or ft is None or np.isnan(gc):
            continue
        if ft is None or len(ft) == 0:
            continue
        if np.all(np.isnan(ft)):
            continue

        # ts shape: (nFeatures, 3, nFrames)
        x = ts[feat_idx, 0, :]  # x positions
        y = ts[feat_idx, 1, :]  # y positions
        conf = ts[feat_idx, 2, :]  # confidence

        # Compute velocity at video frame rate
        dx = np.diff(x)
        dy = np.diff(y)
        v = np.sqrt(dx**2 + dy**2) * VIDEO_FPS  # pixels/second
        v = np.concatenate([[0], v])  # pad to same length

        # Set velocity to NaN where confidence is low
        low_conf = conf < DLC_CONF_THRESH
        v[low_conf] = np.nan

        # Aligned video times
        aligned_ft = ft - vidshift - gc

        # Interpolate velocity to neural time bins
        valid = ~np.isnan(v) & ~np.isnan(aligned_ft)
        if np.sum(valid) < 2:
            continue

        try:
            v_interp = np.interp(taxis, aligned_ft[valid], v[valid], left=np.nan, right=np.nan)
            conf_interp = np.interp(taxis, aligned_ft, conf, left=0, right=0)
        except:
            continue

        velocity[:, out_idx] = v_interp
        visible[:, out_idx] = conf_interp >= DLC_CONF_THRESH

    return velocity, visible


def compute_motion_energy_timeseries(session, trial_indices, me_trials, vidshift):
    """Align motion energy to neural time bins.

    Returns: (n_timebins, n_trials) array.
    """
    n_trials = len(trial_indices)
    me_aligned = np.full((N_TIMEBINS, n_trials), np.nan, dtype=np.float32)

    goCue = session['goCue']
    taxis = TIME_CENTERS

    if 'traj' not in session or len(session['traj']) == 0:
        return me_aligned

    cam = session['traj'][0]  # Use camera 0 (bottom) frame times

    for out_idx, t in enumerate(trial_indices):
        gc = goCue[t]
        if np.isnan(gc):
            continue

        if t >= len(me_trials) or me_trials[t] is None:
            continue

        me_trial = me_trials[t]

        # Get frame times for alignment
        ft = cam['frameTimes'][t]
        if ft is None or len(ft) == 0 or np.all(np.isnan(ft)):
            # Fallback: create frame times assuming 400 Hz
            n_frames = len(me_trial)
            ft = np.arange(n_frames) / VIDEO_FPS + 0.5  # 0.5s pad offset

        aligned_ft = ft - vidshift - gc

        # Trim to match me_trial length
        min_len = min(len(aligned_ft), len(me_trial))
        aligned_ft = aligned_ft[:min_len]
        me_data = me_trial[:min_len]

        valid = ~np.isnan(aligned_ft) & ~np.isnan(me_data)
        if np.sum(valid) < 2:
            continue

        try:
            me_interp = np.interp(taxis, aligned_ft[valid], me_data[valid], left=np.nan, right=np.nan)
        except:
            continue

        me_aligned[:, out_idx] = me_interp

    return me_aligned


def discretize_velocity(velocity, visible):
    """Discretize velocity per session.

    0: < 50th percentile (visible)
    1: >= 50th percentile (visible)
    2: not visible
    """
    n_time, n_trials = velocity.shape
    result = np.full((n_time, n_trials), 2, dtype=np.int32)  # default: not visible

    # Compute 50th percentile from all visible values
    vis_vals = velocity[visible]
    if len(vis_vals) == 0:
        return result

    thresh = np.nanpercentile(vis_vals, 50)

    result[visible & (velocity < thresh)] = 0
    result[visible & (velocity >= thresh)] = 1

    return result


def discretize_motion_energy(me_aligned):
    """Discretize motion energy per session.

    0: < 50th percentile
    1: >= 50th percentile
    2: no video (NaN)
    """
    n_time, n_trials = me_aligned.shape
    result = np.full((n_time, n_trials), 2, dtype=np.int32)  # default: no video

    valid = ~np.isnan(me_aligned)
    valid_vals = me_aligned[valid]
    if len(valid_vals) == 0:
        return result

    thresh = np.nanpercentile(valid_vals, 50)

    result[valid & (me_aligned < thresh)] = 0
    result[valid & (me_aligned >= thresh)] = 1

    return result


# ============================================================
# Main conversion
# ============================================================
def process_session(fpath, me_path=None, show_processing=False, session_id=''):
    """Process one session and return trial-level data."""
    t0 = time.time()

    # Load
    session = load_session(fpath)
    if session is None:
        return None

    t_load = time.time() - t0

    neurons = session['neurons']
    if len(neurons) == 0:
        print(f"  {session_id}: No neural data, skipping")
        return None

    n_trials = session['Ntrials']
    goCue = session['goCue']

    # ----- Trial filtering -----
    # Exclude: early lick, stim trials
    valid_trials = np.ones(n_trials, dtype=bool)
    valid_trials[session['early']] = False
    valid_trials[session['stim_enable']] = False
    # Also exclude trials with NaN goCue
    valid_trials[np.isnan(goCue)] = False

    trial_indices = np.where(valid_trials)[0]
    n_valid = len(trial_indices)

    if n_valid < 2:
        print(f"  {session_id}: Too few valid trials ({n_valid}), skipping")
        return None

    # ----- Neural: align, bin, smooth -----
    t1 = time.time()
    trialdat = align_and_bin_spikes(neurons, goCue, n_trials, EDGES, DT)
    t_bin = time.time() - t1

    # Filter to valid trials
    trialdat = trialdat[:, :, trial_indices]  # (n_neurons, n_timebins, n_valid)

    # Remove low FR neurons
    mean_fr = np.mean(np.mean(trialdat, axis=2), axis=1)
    keep_neurons = mean_fr >= LOW_FR_THRESH
    trialdat = trialdat[keep_neurons, :, :]

    n_units = trialdat.shape[0]
    if n_units < MIN_UNITS:
        print(f"  {session_id}: Too few units after filtering ({n_units}), skipping")
        return None

    print(f"  {session_id}: {len(neurons)} raw -> {n_units} units, {n_valid} trials "
          f"(load={t_load:.1f}s, bin={t_bin:.1f}s)")

    # ----- Outputs -----
    # Lick direction: 0=left, 1=right, 2=none
    # L/R fields indicate STIMULUS direction, not lick direction
    # hit: animal licked correct side (L->left, R->right)
    # miss: animal licked wrong side (L->right, R->left)
    # no: no lick
    lick_dir = np.full(n_valid, 2, dtype=np.int32)  # default: none
    for out_idx, t in enumerate(trial_indices):
        if session['hit'][t]:
            # Correct: lick matches stimulus
            if session['L'][t]:
                lick_dir[out_idx] = 0  # left
            elif session['R'][t]:
                lick_dir[out_idx] = 1  # right
        elif session['miss'][t]:
            # Incorrect: lick is opposite of stimulus
            if session['L'][t]:
                lick_dir[out_idx] = 1  # licked right (wrong)
            elif session['R'][t]:
                lick_dir[out_idx] = 0  # licked left (wrong)
        # 'no' trials: lick_dir stays 2 (none)

    # Behavioral context: 0=WC, 1=DR
    context = np.full(n_valid, 1, dtype=np.int32)  # default: DR
    for out_idx, t in enumerate(trial_indices):
        if session['autowater'][t] == 1:
            context[out_idx] = 0

    # Outcome: 0=incorrect, 1=correct, 2=ignore
    outcome = np.full(n_valid, 2, dtype=np.int32)  # default: ignore
    for out_idx, t in enumerate(trial_indices):
        if session['hit'][t]:
            outcome[out_idx] = 1
        elif session['miss'][t]:
            outcome[out_idx] = 0

    # Tongue velocity (bottom camera, 'tongue' feature)
    vidshift = session.get('vidshift', 0.5)
    tongue_vel, tongue_vis = compute_velocity_timeseries(
        session, cam_idx=0, feat_name='tongue',
        trial_indices=trial_indices, vidshift=vidshift)
    tongue_disc = discretize_velocity(tongue_vel, tongue_vis)

    # Paw velocity (side camera, 'top_paw' feature)
    paw_vel, paw_vis = compute_velocity_timeseries(
        session, cam_idx=1, feat_name='top_paw',
        trial_indices=trial_indices, vidshift=vidshift)
    paw_disc = discretize_velocity(paw_vel, paw_vis)

    # Motion energy
    me_trials = session.get('me_embedded', None)
    has_me_file = me_path is not None and os.path.exists(me_path)

    if has_me_file:
        me_trials_loaded, me_thresh = load_motion_energy_file(me_path)
        me_aligned = compute_motion_energy_timeseries(
            session, trial_indices, me_trials_loaded, vidshift)
    elif me_trials is not None:
        me_aligned = compute_motion_energy_timeseries(
            session, trial_indices, me_trials, vidshift)
    else:
        me_aligned = np.full((N_TIMEBINS, n_valid), np.nan, dtype=np.float32)

    me_disc = discretize_motion_energy(me_aligned)

    # ----- Build per-trial output -----
    result = {
        'neural': [],      # list of (n_neurons, n_timebins) arrays
        'input': [],       # list of (1, n_timebins) arrays (time from go cue)
        'output': [],      # list of (n_output, n_timebins) or (n_output,) arrays
        'n_neurons': n_units,
        'n_trials': n_valid,
    }

    time_input = TIME_CENTERS.astype(np.float32)  # (n_timebins,)

    for i in range(n_valid):
        # Neural: (n_neurons, n_timebins)
        result['neural'].append(trialdat[:, :, i].astype(np.float32))

        # Input: time from go cue (1, n_timebins)
        result['input'].append(time_input[np.newaxis, :])

        # Output: combine per-trial and time-varying
        # Per-trial outputs: lick_direction, behavioral_context, outcome
        # Time-varying outputs: tongue_velocity, paw_velocity, motion_energy
        out = np.zeros((6, N_TIMEBINS), dtype=np.int32)
        out[0, :] = lick_dir[i]          # lick direction (per-trial, broadcast)
        out[1, :] = context[i]           # behavioral context (per-trial, broadcast)
        out[2, :] = outcome[i]           # outcome (per-trial, broadcast)
        out[3, :] = tongue_disc[:, i]    # tongue velocity (time-varying)
        out[4, :] = paw_disc[:, i]       # paw velocity (time-varying)
        out[5, :] = me_disc[:, i]        # motion energy (time-varying)
        result['output'].append(out)

    # Visualization
    if show_processing:
        plot_processing(session_id, session, trial_indices, trialdat,
                       tongue_vel, tongue_vis, tongue_disc,
                       paw_vel, paw_vis, paw_disc,
                       me_aligned, me_disc, lick_dir, context, outcome)

    return result


def plot_processing(session_id, session, trial_indices, trialdat,
                   tongue_vel, tongue_vis, tongue_disc,
                   paw_vel, paw_vis, paw_disc,
                   me_aligned, me_disc, lick_dir, context, outcome):
    """Plot processing visualizations."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(6, 2, figsize=(20, 24))
    fig.suptitle(f'Processing: {session_id}', fontsize=16)

    taxis = TIME_CENTERS
    n_trials = trialdat.shape[2]

    # 1. Neural raster (first 20 neurons, first trial)
    ax = axes[0, 0]
    trial_idx = min(5, n_trials - 1)
    n_show = min(20, trialdat.shape[0])
    im = ax.imshow(trialdat[:n_show, :, trial_idx], aspect='auto',
                   extent=[TMIN, TMAX, n_show, 0], cmap='hot')
    ax.set_title(f'Neural activity (trial {trial_idx}, first {n_show} neurons)')
    ax.set_xlabel('Time from go cue (s)')
    ax.set_ylabel('Neuron')
    ax.axvline(0, color='white', linestyle='--', alpha=0.7)
    plt.colorbar(im, ax=ax, label='FR (Hz)')

    # 2. Mean neural activity across trials
    ax = axes[0, 1]
    mean_fr = np.mean(trialdat, axis=2)  # (neurons, time)
    pop_mean = np.mean(mean_fr, axis=0)
    ax.plot(taxis, pop_mean)
    ax.set_title('Population mean firing rate')
    ax.set_xlabel('Time from go cue (s)')
    ax.set_ylabel('Mean FR (Hz)')
    ax.axvline(0, color='r', linestyle='--', alpha=0.7)

    # 3. Tongue velocity
    ax = axes[1, 0]
    if not np.all(np.isnan(tongue_vel)):
        ax.plot(taxis, np.nanmean(tongue_vel, axis=1), label='Mean velocity')
        ax.set_title('Tongue velocity (mean across trials)')
    else:
        ax.set_title('Tongue velocity (no data)')
    ax.set_xlabel('Time from go cue (s)')
    ax.axvline(0, color='r', linestyle='--', alpha=0.7)

    # 4. Tongue discretized
    ax = axes[1, 1]
    for val, label in [(0, '<50th'), (1, '>=50th'), (2, 'not visible')]:
        frac = np.mean(tongue_disc == val, axis=1)
        ax.plot(taxis, frac, label=label)
    ax.set_title('Tongue velocity discretized')
    ax.set_xlabel('Time from go cue (s)')
    ax.legend()
    ax.axvline(0, color='r', linestyle='--', alpha=0.7)

    # 5. Paw velocity
    ax = axes[2, 0]
    if not np.all(np.isnan(paw_vel)):
        ax.plot(taxis, np.nanmean(paw_vel, axis=1))
        ax.set_title('Paw velocity (mean across trials)')
    else:
        ax.set_title('Paw velocity (no data)')
    ax.set_xlabel('Time from go cue (s)')
    ax.axvline(0, color='r', linestyle='--', alpha=0.7)

    # 6. Paw discretized
    ax = axes[2, 1]
    for val, label in [(0, '<50th'), (1, '>=50th'), (2, 'not visible')]:
        frac = np.mean(paw_disc == val, axis=1)
        ax.plot(taxis, frac, label=label)
    ax.set_title('Paw velocity discretized')
    ax.set_xlabel('Time from go cue (s)')
    ax.legend()
    ax.axvline(0, color='r', linestyle='--', alpha=0.7)

    # 7. Motion energy
    ax = axes[3, 0]
    if not np.all(np.isnan(me_aligned)):
        ax.plot(taxis, np.nanmean(me_aligned, axis=1))
        ax.set_title('Motion energy (mean across trials)')
    else:
        ax.set_title('Motion energy (no data)')
    ax.set_xlabel('Time from go cue (s)')
    ax.axvline(0, color='r', linestyle='--', alpha=0.7)

    # 8. Motion energy discretized
    ax = axes[3, 1]
    for val, label in [(0, '<50th'), (1, '>=50th'), (2, 'no video')]:
        frac = np.mean(me_disc == val, axis=1)
        ax.plot(taxis, frac, label=label)
    ax.set_title('Motion energy discretized')
    ax.set_xlabel('Time from go cue (s)')
    ax.legend()
    ax.axvline(0, color='r', linestyle='--', alpha=0.7)

    # 9. Output distributions
    ax = axes[4, 0]
    labels = ['left', 'right', 'none']
    counts = [np.sum(lick_dir == i) for i in range(3)]
    ax.bar(labels, counts)
    ax.set_title('Lick direction distribution')

    ax = axes[4, 1]
    labels = ['WC', 'DR']
    counts = [np.sum(context == i) for i in range(2)]
    ax.bar(labels, counts)
    ax.set_title('Behavioral context distribution')

    # 10. Outcome distribution
    ax = axes[5, 0]
    labels = ['incorrect', 'correct', 'ignore']
    counts = [np.sum(outcome == i) for i in range(3)]
    ax.bar(labels, counts)
    ax.set_title('Outcome distribution')

    axes[5, 1].axis('off')

    fig.tight_layout()
    safe_id = session_id.replace('/', '_').replace(' ', '_')
    fig.savefig(f'/app/processing_{safe_id}.png', dpi=100)
    plt.close(fig)
    print(f"  Saved processing plot: /app/processing_{safe_id}.png")


def find_me_file(data_dir, animal, date):
    """Find matching motionEnergy file."""
    pattern = f'motionEnergy_{animal}_{date}.mat'
    fpath = os.path.join(data_dir, pattern)
    if os.path.exists(fpath):
        return fpath
    return None


def discover_sessions(sample=False):
    """Discover all session files."""
    sessions = []

    for dir_key, data_dir in DATA_DIRS.items():
        if not os.path.exists(data_dir):
            continue

        for fn in sorted(os.listdir(data_dir)):
            if not fn.startswith('data_structure_'):
                continue
            if fn in SKIP_FILES:
                continue

            parts = fn.replace('.mat', '').split('_')
            animal = parts[2]
            date = parts[3]

            fpath = os.path.join(data_dir, fn)
            me_path = find_me_file(data_dir, animal, date)

            sessions.append({
                'fpath': fpath,
                'me_path': me_path,
                'animal': animal,
                'date': date,
                'dir': dir_key,
                'session_id': f'{animal}_{date}',
            })

    if sample:
        # Pick 2 sessions with neural data (one from each dir if possible)
        sample_sessions = []
        for dir_key in DATA_DIRS:
            dir_sessions = [s for s in sessions if s['dir'] == dir_key]
            if dir_sessions:
                # Pick a medium-sized session
                sample_sessions.append(dir_sessions[len(dir_sessions) // 2])
            if len(sample_sessions) >= 2:
                break
        return sample_sessions

    return sessions


def main():
    parser = argparse.ArgumentParser(description='Convert neural data to decoder format.')
    parser.add_argument('output', type=str, help='Output pickle file path')
    parser.add_argument('--sample', action='store_true', help='Process only 2 sessions')
    parser.add_argument('--full', action='store_true', help='Process all sessions (default)')
    parser.add_argument('--show-processing', action='store_true', help='Save processing plots')
    args = parser.parse_args()

    t_start = time.time()

    # Discover sessions
    sessions_info = discover_sessions(sample=args.sample)
    print(f"Found {len(sessions_info)} sessions to process")

    # Process sessions
    all_neural = []
    all_input = []
    all_output = []
    all_subjects = []
    subject_idx_list = []
    brain_region_idx_list = []

    subjects_set = sorted(set(s['animal'] for s in sessions_info))
    subject_to_idx = {s: i for i, s in enumerate(subjects_set)}

    session_count = 0
    for si, sess_info in enumerate(sessions_info):
        print(f"\n[{si+1}/{len(sessions_info)}] Processing {sess_info['session_id']} ({sess_info['dir']})")

        result = process_session(
            sess_info['fpath'],
            me_path=sess_info['me_path'],
            show_processing=args.show_processing and si < 2,
            session_id=sess_info['session_id'],
        )

        if result is None:
            continue

        all_neural.append(result['neural'])
        all_input.append(result['input'])
        all_output.append(result['output'])
        subject_idx_list.append(subject_to_idx[sess_info['animal']])

        # All neurons recorded from ALM
        brain_region_idx_list.append(np.zeros(result['n_neurons'], dtype=np.int64))

        session_count += 1

    if session_count == 0:
        print("ERROR: No sessions processed!")
        sys.exit(1)

    # Build output dictionary
    data = {
        'neural': all_neural,
        'input': all_input,
        'output': all_output,

        'subjects': subjects_set,
        'subject_idx': np.array(subject_idx_list, dtype=np.int64),

        'brain_regions': ['ALM'],
        'brain_region_idx': brain_region_idx_list,

        'input_names': ['time_from_go_cue'],
        'output_names': ['lick_direction', 'behavioral_context', 'outcome',
                        'tongue_velocity', 'paw_velocity', 'motion_energy'],
        'output_values': [
            ['left', 'right', 'none'],
            ['WC', 'DR'],
            ['incorrect', 'correct', 'ignore'],
            ['low', 'high', 'not_visible'],
            ['low', 'high', 'not_visible'],
            ['low', 'high', 'no_video'],
        ],

        'metadata': {
            'task_description': 'Delayed-response (DR) and water-cued (WC) licking task with auditory stimuli. '
                               'Mice report left/right lick direction after sample-delay-go cue sequence.',
            'time_bin_size': DT * 1000,  # 10 ms
            'temporal_alignment_event': 'Go cue onset (DR) / water drop (WC)',
            'off_start': TMIN,   # -2.5s
            'off_end': TMAX,     # 2.5s
            'smoothing': f'Causal Gaussian, {SMOOTH_WIN} samples ({SMOOTH_WIN * DT * 1000:.0f} ms window)',
            'quality_filter': 'all (exclude garbage, noisy)',
            'min_firing_rate': LOW_FR_THRESH,
            'min_units_per_session': MIN_UNITS,
            'brain_region': 'ALM (anterior lateral motor cortex)',
            'source_paper': 'Hasnain, Birnbaum et al., Nat Neurosci 2025',
        }
    }

    # Print summary
    total_neurons = sum(len(bi) for bi in brain_region_idx_list)
    total_trials = sum(len(s) for s in all_neural)
    n_sessions = len(all_neural)

    print(f"\n{'='*60}")
    print(f"Conversion Summary")
    print(f"{'='*60}")
    print(f"Sessions: {n_sessions}")
    print(f"Subjects: {len(subjects_set)}")
    print(f"Total neurons: {total_neurons}")
    print(f"Total trials: {total_trials}")
    print(f"Neurons/session: {total_neurons/n_sessions:.1f} mean")
    print(f"Trials/session: {total_trials/n_sessions:.1f} mean")
    print(f"Time bins: {N_TIMEBINS} ({DT*1000:.0f} ms)")
    print(f"Time window: [{TMIN}, {TMAX}] s")
    print(f"Total time: {time.time() - t_start:.1f}s")

    # Save
    print(f"\nSaving to {args.output}...")
    with open(args.output, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    fsize = os.path.getsize(args.output) / (1024**2)
    print(f"Saved ({fsize:.1f} MB)")
    print("Done!")


if __name__ == '__main__':
    main()
