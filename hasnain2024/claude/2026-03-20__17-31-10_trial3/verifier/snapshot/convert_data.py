#!/usr/bin/env python3
"""
Convert neural data from Hasnain, Birnbaum et al. (Nature Neuroscience 2024)
into decoder-compatible format.

Usage:
    python -u convert_data.py <outpicklefile> [--full | --sample] [--show-processing]
"""

import argparse
import os
import sys
import time
import pickle
import numpy as np
import h5py
import scipy.io
import scipy.ndimage
from collections import OrderedDict
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Configuration
# ============================================================

# Processing parameters (from reference code WorkingWithDataObjs.m)
ALIGN_EVENT = 'goCue'
TMIN = -2.5  # seconds
TMAX = 2.5   # seconds
DT = 1.0 / 100  # 10 ms time bins
SMOOTH_N = 15  # causal Gaussian smoothing window
SMOOTH_BC = 'reflect'  # boundary condition
LOW_FR_THRESHOLD = 1.0  # Hz, from paper
MIN_UNITS = 10  # minimum units per session (from paper)

# Quality labels to exclude (from findClusters.m)
EXCLUDE_QUALITIES = {'garbage', 'gabrga', 'noisy', 'real?'}

# Session definitions: (directory, animal, date, probe_numbers)
# From DataLoadingScripts/Recording and video/
EPHYS_SESSIONS = [
    # Ephys_Behavior
    ('data/Ephys_Behavior', 'JEB6', '2021-04-18', [2]),
    ('data/Ephys_Behavior', 'JEB7', '2021-04-29', [1]),
    ('data/Ephys_Behavior', 'JEB7', '2021-04-30', [1]),
    ('data/Ephys_Behavior', 'EKH1', '2021-08-07', [2]),
    ('data/Ephys_Behavior', 'EKH3', '2021-08-11', [2]),
    ('data/Ephys_Behavior', 'JGR2', '2021-11-16', [1]),
    ('data/Ephys_Behavior', 'JGR2', '2021-11-17', [1]),
    ('data/Ephys_Behavior', 'JGR3', '2021-11-18', [1]),
    ('data/Ephys_Behavior', 'JEB13', '2022-09-13', [2]),
    ('data/Ephys_Behavior', 'JEB13', '2022-09-14', [2]),
    ('data/Ephys_Behavior', 'JEB13', '2022-09-21', [1]),
    ('data/Ephys_Behavior', 'JEB13', '2022-09-24', [1]),
    ('data/Ephys_Behavior', 'JEB13', '2022-09-25', [1]),
    ('data/Ephys_Behavior', 'JEB14', '2022-08-22', [1]),
    ('data/Ephys_Behavior', 'JEB14', '2022-08-23', [1]),
    ('data/Ephys_Behavior', 'JEB14', '2022-08-24', [1]),
    ('data/Ephys_Behavior', 'JEB14', '2022-08-25', [1]),
    ('data/Ephys_Behavior', 'JEB15', '2022-07-26', [1, 2]),
    ('data/Ephys_Behavior', 'JEB15', '2022-07-27', [1, 2]),
    ('data/Ephys_Behavior', 'JEB15', '2022-07-28', [1, 2]),
    ('data/Ephys_Behavior', 'JEB15', '2022-07-29', [2]),
    ('data/Ephys_Behavior', 'JEB19', '2023-04-18', [1]),
    ('data/Ephys_Behavior', 'JEB19', '2023-04-19', [1]),
    ('data/Ephys_Behavior', 'JEB19', '2023-04-20', [1]),
    ('data/Ephys_Behavior', 'JEB19', '2023-04-21', [1]),
    # RandomizedDelay_Ephys_Behavior
    ('data/RandomizedDelay_Ephys_Behavior', 'JEB11', '2022-05-10', [1]),
    ('data/RandomizedDelay_Ephys_Behavior', 'JEB11', '2022-05-11', [1]),
    ('data/RandomizedDelay_Ephys_Behavior', 'JEB12', '2022-05-12', [1]),
    ('data/RandomizedDelay_Ephys_Behavior', 'JEB12', '2022-05-13', [1]),
    ('data/RandomizedDelay_Ephys_Behavior', 'JEB23', '2023-10-10', [1]),
    ('data/RandomizedDelay_Ephys_Behavior', 'JEB23', '2023-10-11', [1]),
    ('data/RandomizedDelay_Ephys_Behavior', 'JEB23', '2023-10-12', [1]),
    ('data/RandomizedDelay_Ephys_Behavior', 'JEB23', '2023-10-13', [1]),
    ('data/RandomizedDelay_Ephys_Behavior', 'JEB23', '2023-10-18', [1]),
    ('data/RandomizedDelay_Ephys_Behavior', 'JEB23', '2023-10-19', [1]),
    ('data/RandomizedDelay_Ephys_Behavior', 'JEB23', '2023-10-21', [1]),
    ('data/RandomizedDelay_Ephys_Behavior', 'JEB24', '2023-10-23', [1]),
    ('data/RandomizedDelay_Ephys_Behavior', 'JEB24', '2023-10-24', [1]),
    ('data/RandomizedDelay_Ephys_Behavior', 'JEB24', '2023-10-25', [1]),
    ('data/RandomizedDelay_Ephys_Behavior', 'JEB24', '2023-10-26', [1]),
    ('data/RandomizedDelay_Ephys_Behavior', 'JEB24', '2023-10-27', [1]),
    ('data/RandomizedDelay_Ephys_Behavior', 'JEB24', '2023-10-31', [1]),
    ('data/RandomizedDelay_Ephys_Behavior', 'JEB24', '2023-11-02', [1]),
    ('data/RandomizedDelay_Ephys_Behavior', 'JEB24', '2023-11-03', [1]),
]

# ============================================================
# Utility Functions
# ============================================================

def causal_gaussian_smooth(x, N, bctype='reflect'):
    """
    Causal Gaussian smoothing, matching mySmooth.m from reference code.
    x: 1D array
    N: window size
    bctype: 'reflect', 'zeropad', or 'none'
    """
    if N <= 1:
        return x.copy()

    # Create Gaussian kernel
    t = np.arange(N)
    mu = (N - 1) / 2
    sigma = N / 6  # approximate
    kernel = np.exp(-0.5 * ((t - mu) / sigma) ** 2)

    # Make causal: zero out first half
    kernel[:int(np.ceil(N / 2))] = 0
    kernel = kernel / kernel.sum()

    if bctype == 'reflect':
        # Pad by reflecting
        padded = np.concatenate([x[N-1:0:-1], x, x[-2:-N-1:-1]])
        smoothed = np.convolve(padded, kernel, mode='same')
        return smoothed[N-1:N-1+len(x)]
    elif bctype == 'zeropad':
        padded = np.concatenate([np.zeros(N), x, np.zeros(N)])
        smoothed = np.convolve(padded, kernel, mode='same')
        return smoothed[N:N+len(x)]
    else:  # none
        smoothed = np.convolve(x, kernel, mode='same')
        return smoothed


def compute_time_axis():
    """Compute time axis matching getSeq.m: edges + dt/2, drop last."""
    edges = np.arange(TMIN, TMAX + DT/2, DT)
    time_axis = edges[:-1] + DT / 2
    return time_axis, edges


def h5_deref_string(f, ref):
    """Dereference an HDF5 object reference to a string."""
    obj = f[ref]
    return ''.join([chr(int(c)) for c in obj[:].flatten()])


# ============================================================
# Data Loading (handles both HDF5 v7.3 and MATLAB v5 formats)
# ============================================================

def load_mat_file(filepath):
    """Load a .mat file, handling both v5 and v7.3 (HDF5) formats."""
    try:
        f = h5py.File(filepath, 'r')
        return f, 'h5'
    except Exception:
        mat = scipy.io.loadmat(filepath, squeeze_me=False)
        return mat, 'v5'


def get_bp_field(data, fmt, field_path):
    """Get a field from obj.bp, handling both formats."""
    if fmt == 'h5':
        parts = field_path.split('.')
        obj = data['obj']['bp']
        for p in parts:
            obj = obj[p]
        return obj[:].flatten()
    else:
        parts = field_path.split('.')
        obj = data['obj']['bp'][0, 0]
        for p in parts:
            obj = obj[p][0, 0] if isinstance(obj, np.ndarray) and obj.dtype.names else obj[p]
        if isinstance(obj, np.ndarray):
            return obj.flatten()
        return obj


def get_ntrials(data, fmt):
    """Get number of trials."""
    if fmt == 'h5':
        return int(data['obj']['bp']['Ntrials'][0, 0])
    else:
        return int(data['obj']['bp'][0, 0]['Ntrials'][0, 0])


def get_event_times(data, fmt, event_name):
    """Get event times for all trials."""
    return get_bp_field(data, fmt, f'ev.{event_name}')


def get_clusters(data, fmt, probe_idx):
    """
    Get cluster data for a given probe.
    Returns list of dicts with keys: trialtm, trial, quality
    """
    clusters = []

    if fmt == 'h5':
        f = data
        clu = f['obj']['clu']
        ref = clu[probe_idx, 0]
        probe_data = f[ref]

        if isinstance(probe_data, h5py.Dataset):
            # Empty or minimal probe
            return clusters

        n_clu = probe_data['quality'].shape[0]
        for i in range(n_clu):
            # Quality
            q_ref = probe_data['quality'][i, 0]
            quality = h5_deref_string(f, q_ref).strip().lower()

            # Skip excluded qualities
            if quality in EXCLUDE_QUALITIES:
                continue

            # Spike times and trials
            trialtm_ref = probe_data['trialtm'][i, 0]
            trialtm = f[trialtm_ref][:].flatten()

            trial_ref = probe_data['trial'][i, 0]
            trial = f[trial_ref][:].flatten().astype(int)

            clusters.append({
                'trialtm': trialtm,
                'trial': trial,
                'quality': quality,
            })
    else:
        clu = data['obj']['clu'][0, 0]
        probe_data = clu[0, probe_idx]

        n_clu = probe_data.shape[1]
        for i in range(n_clu):
            quality = str(probe_data['quality'][0, i][0]).strip().lower()

            if quality in EXCLUDE_QUALITIES:
                continue

            trialtm = probe_data['trialtm'][0, i].flatten()
            trial = probe_data['trial'][0, i].flatten().astype(int)

            clusters.append({
                'trialtm': trialtm,
                'trial': trial,
                'quality': quality,
            })

    return clusters


def get_traj_data(data, fmt, view, trial_idx):
    """
    Get DLC trajectory data for a given camera view and trial.
    Returns: ts (features x 3 x frames), frameTimes (frames,), featNames (list of str)
    """
    if fmt == 'h5':
        f = data
        traj = f['obj']['traj']
        cam = f[traj[view, 0]]

        # Get feature names (from first trial)
        fn_ref = cam['featNames'][0, 0]
        fn_data = f[fn_ref]
        feat_names = []
        for i in range(fn_data.shape[1]):
            ref = fn_data[0, i]
            name = h5_deref_string(f, ref)
            feat_names.append(name)

        # Get ts and frameTimes for this trial
        ts_ref = cam['ts'][trial_idx, 0]
        ts = f[ts_ref][:]  # (features, 3, frames) - note: transposed from MATLAB (frames, 3, features)
        # HDF5 stores in column-major, so shape is (features, 3, frames)
        # In MATLAB: ts is (frames, [x,y,conf], features)
        # In HDF5/numpy: ts is (features, [x,y,conf], frames)

        ft_ref = cam['frameTimes'][trial_idx, 0]
        frame_times = f[ft_ref][:].flatten()

        return ts, frame_times, feat_names
    else:
        obj = data['obj']
        traj = obj['traj'][0, 0]
        cam = traj[0, view]

        # Feature names from first trial
        feat_names_raw = cam[0, 0]['featNames']
        if feat_names_raw.ndim == 1:
            feat_names = [str(fn[0]) if hasattr(fn, '__len__') else str(fn) for fn in feat_names_raw.flatten()]
        elif feat_names_raw.ndim == 2:
            # (n_feats, 1) format
            feat_names = [str(feat_names_raw[i, 0][0]) if hasattr(feat_names_raw[i, 0], '__len__') else str(feat_names_raw[i, 0])
                         for i in range(feat_names_raw.shape[0])]
        else:
            feat_names = [str(fn) for fn in feat_names_raw.flatten()]

        ts = cam[0, trial_idx]['ts']
        frame_times = cam[0, trial_idx]['frameTimes'].flatten()

        return ts, frame_times, feat_names


def load_motion_energy(dirpath, animal, date):
    """Load motion energy from separate .mat file."""
    me_file = os.path.join(dirpath, f'motionEnergy_{animal}_{date}.mat')
    if not os.path.exists(me_file):
        return None, None

    try:
        me_mat = scipy.io.loadmat(me_file, squeeze_me=False)
        me_raw = me_mat['me']

        # Handle multiple formats:
        # Format 1: me is a struct with 'data' and 'moveThresh' fields
        # Format 2: me is directly a cell array of per-trial ME data (no struct)
        # Format 3: me.data is itself a struct with nested 'data' field

        me_thresh = None
        me_data = None

        if hasattr(me_raw, 'dtype') and me_raw.dtype.names and 'data' in me_raw.dtype.names:
            # Format 1 or 3: struct
            me_data = me_raw['data'][0, 0]
            try:
                me_thresh = float(me_raw['moveThresh'][0, 0].flatten()[0])
            except:
                me_thresh = None

            # Format 3: nested struct
            if hasattr(me_data, 'dtype') and me_data.dtype.names and 'data' in me_data.dtype.names:
                me_data = me_data['data'][0, 0]
        elif me_raw.dtype == np.dtype('O'):
            # Format 2: direct cell array
            me_data = me_raw
            me_thresh = None
        else:
            print(f"  Warning: Unknown ME format: dtype={me_raw.dtype}")
            return None, None

        # Extract per-trial data
        trial_me = []
        for i in range(me_data.shape[0]):
            elem = me_data[i, 0] if me_data.ndim == 2 else me_data[i]
            trial_me.append(elem.flatten().astype(np.float64))

        return trial_me, me_thresh
    except Exception as e:
        print(f"  Warning: Could not load motion energy: {e}")
        return None, None


def find_video_offset(data, fmt):
    """
    Find video offset (seconds) matching findVideoOffset.m.
    vidshift = mode(sglx.bitcode.bitstart) / sglx.fs - mode(bp.ev.bitStart)
    """
    try:
        if fmt == 'h5':
            f = data
            bp_bitstart = f['obj']['bp']['ev']['bitStart'][:].flatten()
            sglx = f['obj']['sglx']

            # Try to get bitcode bitstart and fs
            if 'bitcode' in sglx:
                bitcode = sglx['bitcode']
                if 'bitstart' in bitcode:
                    sglx_bitstart = bitcode['bitstart'][:].flatten()
                    fs = float(sglx['fs'][0, 0])
                    vid_offset = np.nanmedian(sglx_bitstart) / fs
                    bp_offset = np.nanmedian(bp_bitstart[bp_bitstart > 0])
                    return vid_offset - bp_offset
            return 0.5  # default offset
        else:
            obj_struct = data['obj'][0, 0] if data['obj'].ndim >= 2 else data['obj']
            bp_bitstart = get_bp_field(data, fmt, 'ev.bitStart')
            try:
                sglx = obj_struct['sglx'][0, 0]
                bitcode = sglx['bitcode'][0, 0]
                sglx_bitstart = bitcode['bitstart'].flatten()
                fs = float(sglx['fs'].flatten()[0])
                vid_offset = np.nanmedian(sglx_bitstart) / fs
                bp_offset = np.nanmedian(bp_bitstart[bp_bitstart > 0])
                return vid_offset - bp_offset
            except Exception:
                return 0.5
    except Exception:
        return 0.5


# ============================================================
# Processing Functions
# ============================================================

def bin_and_smooth_spikes(clusters, ntrials, align_times, time_axis, edges):
    """
    Bin spikes into time bins, smooth, and return firing rates.
    Matches getSeq.m logic.

    Returns: trialdat (n_timepoints, n_neurons, n_trials)
    """
    n_time = len(time_axis)
    n_neurons = len(clusters)
    trialdat = np.zeros((n_time, n_neurons, ntrials), dtype=np.float32)

    for i, clu in enumerate(clusters):
        trialtm = clu['trialtm']
        trial = clu['trial']

        # Align spike times: trialtm_aligned = trialtm - align_event_time
        for j in range(ntrials):
            trial_num = j + 1  # 1-indexed
            spike_mask = trial == trial_num
            if not np.any(spike_mask):
                continue

            aligned_times = trialtm[spike_mask] - align_times[j]

            # Bin spikes
            counts = np.histogram(aligned_times, bins=edges)[0]

            # Convert to firing rate and smooth
            rate = counts.astype(np.float64) / DT
            smoothed = causal_gaussian_smooth(rate, SMOOTH_N, SMOOTH_BC)
            trialdat[:, i, j] = smoothed.astype(np.float32)

    return trialdat


def remove_low_fr_neurons(trialdat, clusters):
    """
    Remove neurons with mean FR < LOW_FR_THRESHOLD.
    Matches removeLowFRClusters.m: mean of mean PSTH across conditions.
    """
    # Mean firing rate: average across all time bins and trials
    mean_frs = np.nanmean(np.nanmean(trialdat, axis=2), axis=0)
    keep = mean_frs > LOW_FR_THRESHOLD

    trialdat_filtered = trialdat[:, keep, :]
    clusters_filtered = [c for c, k in zip(clusters, keep) if k]

    return trialdat_filtered, clusters_filtered, keep


def extract_velocity_from_traj(data, fmt, view, feat_name, ntrials, align_times, time_axis, vidshift):
    """
    Extract velocity for a DLC feature across all trials.
    Returns: speed array (n_timepoints, n_trials)
    """
    n_time = len(time_axis)
    speed = np.full((n_time, ntrials), np.nan, dtype=np.float32)

    for trial_idx in range(ntrials):
        try:
            ts, frame_times, feat_names = get_traj_data(data, fmt, view, trial_idx)
        except Exception:
            continue

        # Find feature index
        try:
            feat_idx = feat_names.index(feat_name)
        except ValueError:
            continue

        if len(frame_times) == 0 or np.all(np.isnan(frame_times)):
            # Create synthetic frame times at 400 Hz
            if fmt == 'h5':
                n_frames = ts.shape[2]
            else:
                n_frames = ts.shape[0]
            frame_times = np.arange(n_frames) / 400.0
            ft_offset = 0.5
        else:
            ft_offset = vidshift

        # Align frame times to goCue
        aligned_ft = frame_times - ft_offset - align_times[trial_idx]

        # Extract x, y position
        if fmt == 'h5':
            # ts shape: (features, 3, frames) -> [x, y, conf]
            xpos = ts[feat_idx, 0, :]
            ypos = ts[feat_idx, 1, :]
        else:
            # ts shape: (frames, 3, features) or varies
            if ts.ndim == 3:
                if ts.shape[2] > ts.shape[0]:
                    # (features, 3, frames) format
                    xpos = ts[feat_idx, 0, :]
                    ypos = ts[feat_idx, 1, :]
                else:
                    # (frames, 3, features) format
                    xpos = ts[:, 0, feat_idx]
                    ypos = ts[:, 1, feat_idx]
            else:
                continue

        # Handle tongue: NaN means not visible -> set velocity to 0
        is_tongue = 'tongue' in feat_name.lower()

        # Smooth position (for non-tongue features)
        if not is_tongue:
            xpos_smooth = causal_gaussian_smooth(xpos, 21, 'reflect')
            ypos_smooth = causal_gaussian_smooth(ypos, 21, 'reflect')
        else:
            xpos_smooth = xpos.copy()
            ypos_smooth = ypos.copy()

        # Compute velocity using gradient (matching findVelocity.m)
        xvel = np.gradient(xpos_smooth)
        yvel = np.gradient(ypos_smooth)

        # For non-tongue: subtract baseline velocity
        if not is_tongue:
            xvel = xvel - np.nanmedian(xvel)
            yvel = yvel - np.nanmedian(yvel)

        # For tongue: set NaN velocities to 0
        if is_tongue:
            xvel[np.isnan(xvel)] = 0
            yvel[np.isnan(yvel)] = 0

        # Compute speed
        spd = np.sqrt(xvel**2 + yvel**2)

        # Fill missing with nearest
        nan_mask = np.isnan(spd)
        if np.any(nan_mask) and not np.all(nan_mask):
            valid = np.where(~nan_mask)[0]
            for idx in np.where(nan_mask)[0]:
                nearest = valid[np.argmin(np.abs(valid - idx))]
                spd[idx] = spd[nearest]

        # Interpolate to time axis
        if len(aligned_ft) > 1:
            try:
                interpolated = np.interp(time_axis, aligned_ft, spd)
                speed[:, trial_idx] = interpolated.astype(np.float32)
            except Exception:
                pass

    # Fill remaining NaN columns with zeros
    nan_cols = np.all(np.isnan(speed), axis=0)
    speed[:, nan_cols] = 0.0

    # Fill any remaining NaNs with nearest
    for t in range(ntrials):
        col = speed[:, t]
        nan_mask = np.isnan(col)
        if np.any(nan_mask) and not np.all(nan_mask):
            valid = np.where(~nan_mask)[0]
            for idx in np.where(nan_mask)[0]:
                nearest = valid[np.argmin(np.abs(valid - idx))]
                col[idx] = col[nearest]
        speed[:, t] = col

    return speed


def interpolate_motion_energy(me_data, ntrials, data, fmt, align_times, time_axis, vidshift):
    """
    Interpolate motion energy to neural time axis.
    Matches loadMotionEnergy.m logic.
    """
    n_time = len(time_axis)
    me_interp = np.full((n_time, ntrials), np.nan, dtype=np.float32)

    for trial_idx in range(ntrials):
        if trial_idx >= len(me_data):
            continue

        me_trial = me_data[trial_idx]
        if len(me_trial) == 0:
            continue

        # Get frame times for this trial to align ME
        try:
            ts, frame_times, _ = get_traj_data(data, fmt, 0, trial_idx)  # side cam

            if len(frame_times) == 0 or np.all(np.isnan(frame_times)):
                frame_times = np.arange(len(me_trial)) / 400.0
                ft_offset = 0.5
            else:
                ft_offset = vidshift

            # Align to goCue
            aligned_ft = frame_times - ft_offset - align_times[trial_idx]

            # ME might be shorter than frame times
            n_frames = min(len(me_trial), len(aligned_ft))

            me_interp[:, trial_idx] = np.interp(
                time_axis, aligned_ft[:n_frames], me_trial[:n_frames]
            ).astype(np.float32)
        except Exception:
            # Fallback: create frame times
            frame_times = np.arange(len(me_trial)) / 400.0
            aligned_ft = frame_times - 0.5 - align_times[trial_idx]
            me_interp[:, trial_idx] = np.interp(
                time_axis, aligned_ft, me_trial
            ).astype(np.float32)

    # Fill NaNs with nearest
    for t in range(ntrials):
        col = me_interp[:, t]
        nan_mask = np.isnan(col)
        if np.any(nan_mask) and not np.all(nan_mask):
            valid = np.where(~nan_mask)[0]
            for idx in np.where(nan_mask)[0]:
                nearest = valid[np.argmin(np.abs(valid - idx))]
                col[idx] = col[nearest]
        elif np.all(nan_mask):
            col[:] = 0.0
        me_interp[:, t] = col

    return me_interp


def discretize_time_series(values, threshold):
    """Discretize: 0 if < threshold, 1 if >= threshold.
    If threshold is 0 or very small (degenerate case where median is 0),
    use a small positive threshold so that exact zeros map to class 0.
    """
    if threshold < 1e-10:
        threshold = 1e-10  # Separate zero from positive values
    return (values >= threshold).astype(np.float32)


# ============================================================
# Main Processing
# ============================================================

def process_session(dirpath, animal, date, probe_nums, time_axis, edges, show_processing=False, session_idx=0):
    """
    Process a single session and return data dict, or None if session should be skipped.
    """
    t0 = time.time()
    session_id = f"{animal}_{date}"
    filepath = os.path.join(dirpath, f"data_structure_{session_id}.mat")

    if not os.path.exists(filepath):
        print(f"  WARNING: File not found: {filepath}")
        return None

    print(f"\n  Processing {session_id}...")

    # Load data
    data, fmt = load_mat_file(filepath)
    ntrials = get_ntrials(data, fmt)
    print(f"    Trials: {ntrials}")

    # Get behavior variables
    hit = get_bp_field(data, fmt, 'hit').astype(bool)
    miss = get_bp_field(data, fmt, 'miss').astype(bool)
    R = get_bp_field(data, fmt, 'R').astype(bool)
    L = get_bp_field(data, fmt, 'L').astype(bool)
    autowater = get_bp_field(data, fmt, 'autowater').astype(bool)
    early = get_bp_field(data, fmt, 'early').astype(bool)

    # Get stim.enable
    try:
        stim_enable = get_bp_field(data, fmt, 'stim.enable').astype(bool)
    except Exception:
        stim_enable = np.zeros(ntrials, dtype=bool)

    # Get alignment event times
    align_times = get_event_times(data, fmt, ALIGN_EVENT)

    # Trial filter: exclude stim and early lick trials
    valid_trials = ~stim_enable & ~early
    # Also exclude trials where align time is 0 or NaN (no go cue)
    valid_trials &= ~np.isnan(align_times) & (align_times > 0)

    valid_trial_indices = np.where(valid_trials)[0]
    print(f"    Valid trials (no stim, no early): {len(valid_trial_indices)} / {ntrials}")

    # Get clusters from all specified probes
    all_clusters = []
    for p in probe_nums:
        clusters = get_clusters(data, fmt, p - 1)  # 0-indexed
        all_clusters.extend(clusters)

    print(f"    Quality-filtered neurons: {len(all_clusters)}")

    # Find max trial with spike data (recording may end before session ends)
    if all_clusters:
        max_spike_trial = max(
            int(np.max(c['trial'])) for c in all_clusters if len(c['trial']) > 0
        )
        beyond = np.sum(valid_trial_indices >= max_spike_trial)
        if beyond > 0:
            valid_trial_indices = valid_trial_indices[valid_trial_indices < max_spike_trial]
            print(f"    Excluded {beyond} trials beyond recording (max spike trial={max_spike_trial})")

    if len(valid_trial_indices) < 2:
        print(f"    SKIP: Too few valid trials")
        if fmt == 'h5':
            data.close()
        return None

    if len(all_clusters) < MIN_UNITS:
        print(f"    SKIP: Too few neurons ({len(all_clusters)} < {MIN_UNITS})")
        if fmt == 'h5':
            data.close()
        return None

    # Bin and smooth spikes
    t1 = time.time()
    trialdat = bin_and_smooth_spikes(all_clusters, ntrials, align_times, time_axis, edges)
    print(f"    Spike binning: {time.time()-t1:.1f}s")

    # Remove low FR neurons
    trialdat, filtered_clusters, keep_mask = remove_low_fr_neurons(trialdat, all_clusters)
    n_neurons = len(filtered_clusters)
    print(f"    After low FR filter (>{LOW_FR_THRESHOLD} Hz): {n_neurons} neurons")

    if n_neurons < MIN_UNITS:
        print(f"    SKIP: Too few neurons after FR filter ({n_neurons} < {MIN_UNITS})")
        if fmt == 'h5':
            data.close()
        return None

    # Extract only valid trials from trialdat
    trialdat_valid = trialdat[:, :, valid_trial_indices]

    # Build per-trial neural data: list of (n_neurons, n_timepoints)
    neural_trials = []
    for t_idx in range(len(valid_trial_indices)):
        neural_trials.append(trialdat_valid[:, :, t_idx].T.astype(np.float32))

    # Build output variables for valid trials
    hit_valid = hit[valid_trial_indices]
    miss_valid = miss[valid_trial_indices]
    R_valid = R[valid_trial_indices]
    L_valid = L[valid_trial_indices]
    autowater_valid = autowater[valid_trial_indices]
    align_times_valid = align_times[valid_trial_indices]

    # Lick direction: R=1, L=0 (trial instruction/stimulus side)
    lick_direction = R_valid.astype(np.float32)

    # Behavioral context: WC=0, DR=1
    behavioral_context = (~autowater_valid).astype(np.float32)

    # Outcome: correct=1, incorrect=0
    outcome = hit_valid.astype(np.float32)

    # --- Extract kinematics ---
    # Video offset
    vidshift = find_video_offset(data, fmt)

    # Tongue velocity (from side cam, feature 'jaw' y-velocity as proxy,
    # but task says "tongue velocity" - use tongue from side cam)
    t2 = time.time()
    tongue_speed = extract_velocity_from_traj(
        data, fmt, view=0, feat_name='tongue',
        ntrials=ntrials, align_times=align_times,
        time_axis=time_axis, vidshift=vidshift
    )
    tongue_speed_valid = tongue_speed[:, valid_trial_indices]
    print(f"    Tongue velocity: {time.time()-t2:.1f}s")

    # Paw velocity (from bottom cam, feature 'top_paw')
    t3 = time.time()
    paw_speed = extract_velocity_from_traj(
        data, fmt, view=1, feat_name='top_paw',
        ntrials=ntrials, align_times=align_times,
        time_axis=time_axis, vidshift=vidshift
    )
    paw_speed_valid = paw_speed[:, valid_trial_indices]
    print(f"    Paw velocity: {time.time()-t3:.1f}s")

    # Motion energy
    t4 = time.time()
    me_data, me_thresh = load_motion_energy(dirpath, animal, date)
    if me_data is not None:
        me_interp = interpolate_motion_energy(
            me_data, ntrials, data, fmt, align_times, time_axis, vidshift
        )
        me_valid = me_interp[:, valid_trial_indices]
    else:
        me_valid = np.zeros((len(time_axis), len(valid_trial_indices)), dtype=np.float32)
    print(f"    Motion energy: {time.time()-t4:.1f}s")

    # Discretize continuous outputs using per-session 50th percentile
    # Threshold is computed across all valid timepoints and trials
    tongue_thresh = np.nanpercentile(tongue_speed_valid, 50)
    paw_thresh = np.nanpercentile(paw_speed_valid, 50)
    me_thresh_50 = np.nanpercentile(me_valid, 50)

    tongue_disc = discretize_time_series(tongue_speed_valid, tongue_thresh)
    paw_disc = discretize_time_series(paw_speed_valid, paw_thresh)
    me_disc = discretize_time_series(me_valid, me_thresh_50)

    # Build input and output lists
    input_trials = []
    output_trials = []

    for t_idx in range(len(valid_trial_indices)):
        # Input: time from go cue (continuous, same for all trials)
        input_data = time_axis.astype(np.float32).reshape(1, -1)
        input_trials.append(input_data)

        # Output: 6 variables
        # Per-trial: lick_direction, behavioral_context, outcome
        # Time-varying: tongue_velocity, paw_velocity, motion_energy
        out = np.zeros((6, len(time_axis)), dtype=np.int64)
        out[0, :] = int(lick_direction[t_idx])
        out[1, :] = int(behavioral_context[t_idx])
        out[2, :] = int(outcome[t_idx])
        out[3, :] = tongue_disc[:, t_idx].astype(np.int64)
        out[4, :] = paw_disc[:, t_idx].astype(np.int64)
        out[5, :] = me_disc[:, t_idx].astype(np.int64)
        output_trials.append(out)

    if fmt == 'h5':
        data.close()

    elapsed = time.time() - t0
    print(f"    Session complete: {n_neurons} neurons, {len(valid_trial_indices)} trials, {elapsed:.1f}s")

    # Generate processing plots if requested
    if show_processing and session_idx < 2:
        generate_processing_plots(
            session_id, time_axis, neural_trials, input_trials, output_trials,
            tongue_speed_valid, paw_speed_valid, me_valid,
            tongue_thresh, paw_thresh, me_thresh_50,
            lick_direction, behavioral_context, outcome
        )

    return {
        'neural': neural_trials,
        'input': input_trials,
        'output': output_trials,
        'animal': animal,
        'n_neurons': n_neurons,
        'n_trials': len(valid_trial_indices),
    }


def generate_processing_plots(session_id, time_axis, neural_trials, input_trials, output_trials,
                               tongue_speed, paw_speed, me_vals,
                               tongue_thresh, paw_thresh, me_thresh,
                               lick_dir, context, outcome):
    """Generate diagnostic plots for processing verification."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n_trials = len(neural_trials)
    n_plot = min(5, n_trials)

    fig, axes = plt.subplots(6, n_plot, figsize=(4 * n_plot, 18))
    if n_plot == 1:
        axes = axes.reshape(-1, 1)

    for i in range(n_plot):
        trial = i

        # Neural activity (first 10 neurons)
        ax = axes[0, i]
        neural = neural_trials[trial]
        n_show = min(10, neural.shape[0])
        ax.imshow(neural[:n_show, :], aspect='auto', extent=[time_axis[0], time_axis[-1], n_show, 0])
        ax.set_title(f'Trial {trial}: Neural (first {n_show})')
        ax.axvline(0, color='r', linestyle='--', alpha=0.5)

        # Tongue speed (continuous) with threshold
        ax = axes[1, i]
        ax.plot(time_axis, tongue_speed[:, trial], 'b-', linewidth=0.5)
        ax.axhline(tongue_thresh, color='r', linestyle='--', alpha=0.5)
        ax.set_title(f'Tongue speed (thresh={tongue_thresh:.1f})')
        ax.axvline(0, color='k', linestyle='--', alpha=0.3)

        # Paw speed (continuous) with threshold
        ax = axes[2, i]
        ax.plot(time_axis, paw_speed[:, trial], 'g-', linewidth=0.5)
        ax.axhline(paw_thresh, color='r', linestyle='--', alpha=0.5)
        ax.set_title(f'Paw speed (thresh={paw_thresh:.1f})')
        ax.axvline(0, color='k', linestyle='--', alpha=0.3)

        # Motion energy (continuous) with threshold
        ax = axes[3, i]
        ax.plot(time_axis, me_vals[:, trial], 'm-', linewidth=0.5)
        ax.axhline(me_thresh, color='r', linestyle='--', alpha=0.5)
        ax.set_title(f'ME (thresh={me_thresh:.1f})')
        ax.axvline(0, color='k', linestyle='--', alpha=0.3)

        # Discretized outputs
        ax = axes[4, i]
        out = output_trials[trial]
        for j, name in enumerate(['tongue_vel', 'paw_vel', 'ME']):
            ax.plot(time_axis, out[3 + j, :] + j * 1.2, label=name)
        ax.set_title('Discretized outputs')
        ax.legend(fontsize=6)
        ax.axvline(0, color='k', linestyle='--', alpha=0.3)

        # Per-trial outputs
        ax = axes[5, i]
        ax.text(0.5, 0.7, f'Lick dir: {"R" if lick_dir[trial]==1 else "L"}',
                transform=ax.transAxes, ha='center')
        ax.text(0.5, 0.5, f'Context: {"DR" if context[trial]==1 else "WC"}',
                transform=ax.transAxes, ha='center')
        ax.text(0.5, 0.3, f'Outcome: {"correct" if outcome[trial]==1 else "incorrect"}',
                transform=ax.transAxes, ha='center')
        ax.set_title('Per-trial labels')

    fig.suptitle(f'Processing: {session_id}', fontsize=14)
    fig.tight_layout()
    fig.savefig(f'processing_{session_id}.png', dpi=150)
    plt.close(fig)
    print(f"    Saved processing plot: processing_{session_id}.png")


def main():
    parser = argparse.ArgumentParser(description='Convert neural data to decoder format.')
    parser.add_argument('outfile', type=str, help='Output pickle file path')
    parser.add_argument('--full', action='store_true', default=True, help='Process all sessions (default)')
    parser.add_argument('--sample', action='store_true', help='Process only 2 sessions for testing')
    parser.add_argument('--show-processing', action='store_true', help='Generate processing plots')
    args = parser.parse_args()

    if args.sample:
        sessions = EPHYS_SESSIONS[:2]
        print(f"SAMPLE MODE: Processing {len(sessions)} sessions")
    else:
        sessions = EPHYS_SESSIONS
        print(f"FULL MODE: Processing {len(sessions)} sessions")

    # Compute time axis
    time_axis, edges = compute_time_axis()
    n_timepoints = len(time_axis)
    print(f"Time axis: {time_axis[0]:.4f} to {time_axis[-1]:.4f} s, {n_timepoints} bins, dt={DT*1000:.1f} ms")

    # Process all sessions
    all_neural = []
    all_input = []
    all_output = []
    all_animals = []
    session_info = []

    total_t0 = time.time()

    for sess_idx, (dirpath, animal, date, probes) in enumerate(sessions):
        result = process_session(
            dirpath, animal, date, probes, time_axis, edges,
            show_processing=args.show_processing, session_idx=sess_idx
        )

        if result is None:
            continue

        all_neural.append(result['neural'])
        all_input.append(result['input'])
        all_output.append(result['output'])
        all_animals.append(animal)
        session_info.append({
            'animal': animal,
            'date': date,
            'n_neurons': result['n_neurons'],
            'n_trials': result['n_trials'],
            'directory': dirpath,
        })

    total_elapsed = time.time() - total_t0
    print(f"\nTotal processing time: {total_elapsed:.1f}s")
    print(f"Sessions processed: {len(all_neural)}")

    if len(all_neural) == 0:
        print("ERROR: No sessions processed successfully")
        sys.exit(1)

    # Build subject index
    unique_subjects = sorted(set(all_animals))
    subject_idx = np.array([unique_subjects.index(a) for a in all_animals])

    # Brain regions - all ALM
    brain_regions = ['ALM']
    brain_region_idx = []
    for sess in all_neural:
        n_neurons = sess[0].shape[0]
        brain_region_idx.append(np.zeros(n_neurons, dtype=int))

    # Build final data dict
    data = {
        'neural': all_neural,
        'input': all_input,
        'output': all_output,
        'subjects': unique_subjects,
        'subject_idx': subject_idx,
        'brain_regions': brain_regions,
        'brain_region_idx': brain_region_idx,
        'input_names': ['time_from_go_cue'],
        'output_names': ['lick_direction', 'behavioral_context', 'outcome',
                         'tongue_velocity', 'paw_velocity', 'motion_energy'],
        'output_values': [
            ['left', 'right'],           # lick_direction: 0=left, 1=right
            ['WC', 'DR'],                # behavioral_context: 0=WC, 1=DR
            ['incorrect', 'correct'],    # outcome: 0=incorrect, 1=correct
            ['low', 'high'],             # tongue_velocity: 0=<50th, 1=>=50th
            ['low', 'high'],             # paw_velocity: 0=<50th, 1=>=50th
            ['low', 'high'],             # motion_energy: 0=<50th, 1=>=50th
        ],
        'metadata': {
            'task_description': 'Delayed-response and water-cued licking tasks with ALM recordings. '
                                'Mice perform directional licking in two contexts (DR and WC) that alternate block-wise.',
            'time_bin_size': DT * 1000,  # in ms
            'temporal_alignment_event': 'Go cue onset',
            'off_start': TMIN,  # -2.5 s before go cue
            'off_end': TMAX,    # 2.5 s after go cue
            'session_info': session_info,
            'paper': 'Hasnain, Birnbaum et al., Nature Neuroscience 2024',
            'processing': {
                'smoothing': f'Causal Gaussian, N={SMOOTH_N}, bc={SMOOTH_BC}',
                'low_fr_threshold': LOW_FR_THRESHOLD,
                'quality_filter': f'Exclude: {EXCLUDE_QUALITIES}',
                'trial_filter': 'Exclude stim.enable=1 and early=1 trials',
            },
        },
    }

    # Print summary statistics
    total_neurons = sum(info['n_neurons'] for info in session_info)
    total_trials = sum(info['n_trials'] for info in session_info)
    print(f"\n=== SUMMARY ===")
    print(f"Sessions: {len(session_info)}")
    print(f"Subjects: {len(unique_subjects)} ({', '.join(unique_subjects)})")
    print(f"Total neurons: {total_neurons}")
    print(f"Mean neurons/session: {total_neurons/len(session_info):.1f}")
    print(f"Total trials: {total_trials}")
    print(f"Mean trials/session: {total_trials/len(session_info):.1f}")
    print(f"Time bins: {n_timepoints}")

    # Save
    print(f"\nSaving to {args.outfile}...")
    with open(args.outfile, 'wb') as f:
        pickle.dump(data, f, protocol=4)

    fsize = os.path.getsize(args.outfile) / (1024**2)
    print(f"Saved: {fsize:.1f} MB")
    print("Done!")


if __name__ == '__main__':
    main()
