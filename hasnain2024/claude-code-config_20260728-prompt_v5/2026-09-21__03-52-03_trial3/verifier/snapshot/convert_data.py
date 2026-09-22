#!/usr/bin/env python3
"""
Convert Hasnain, Birnbaum et al. (Nat Neurosci 2024) data to decoder-compatible format.

Usage:
    python -u convert_data.py <outpicklefile> [--full | --sample] [--show-processing]
"""

import argparse
import glob
import os
import pickle
import time
import warnings

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from scipy.signal import convolve

warnings.filterwarnings('ignore')

# ============================================================
# PARAMETERS (matching reference code analysis scripts)
# ============================================================
ALIGN_EVENT = 'goCue'
TMIN = -2.5  # seconds
TMAX = 2.5   # seconds
DT = 1.0 / 100  # 10 ms bins
SMOOTH_WIN = 15  # causal Gaussian kernel window size
BC_TYPE = 'reflect'  # boundary condition for smoothing
LOW_FR = 1.0  # Hz, minimum mean firing rate threshold
MIN_UNITS = 10  # minimum number of units after filtering
EXCLUDED_QUALITIES = {'garbage', 'gabrga', 'noisy', 'real?'}

# Build time axis (matching getSeq.m: edges + dt/2, drop last)
EDGES = np.arange(TMIN, TMAX + DT, DT)
TIME_AXIS = EDGES[:-1] + DT / 2
N_TIMEBINS = len(TIME_AXIS)

# Session metadata: (animal, date, probes_matlab_indexed, data_dir_key)
# From loading scripts in code/DataLoadingScripts/Recording and video/
SESSION_META = [
    # Ephys_Behavior sessions
    ('EKH1', '2021-08-07', [2], 'ephys'),
    ('EKH3', '2021-08-11', [2], 'ephys'),
    ('JEB6', '2021-04-18', [2], 'ephys'),
    ('JEB7', '2021-04-29', [1], 'ephys'),
    ('JEB7', '2021-04-30', [1], 'ephys'),
    ('JEB13', '2022-09-13', [2], 'ephys'),
    ('JEB13', '2022-09-14', [2], 'ephys'),
    ('JEB13', '2022-09-21', [1], 'ephys'),
    ('JEB13', '2022-09-24', [1], 'ephys'),
    ('JEB13', '2022-09-25', [1], 'ephys'),
    ('JEB14', '2022-08-22', [1], 'ephys'),
    ('JEB14', '2022-08-23', [1], 'ephys'),
    ('JEB14', '2022-08-24', [1], 'ephys'),
    ('JEB14', '2022-08-25', [1], 'ephys'),
    ('JEB15', '2022-07-26', [1, 2], 'ephys'),
    ('JEB15', '2022-07-27', [1, 2], 'ephys'),
    ('JEB15', '2022-07-28', [1, 2], 'ephys'),
    ('JEB15', '2022-07-29', [2], 'ephys'),
    ('JEB19', '2023-04-21', [1], 'ephys'),
    ('JEB19', '2023-04-20', [1], 'ephys'),
    ('JEB19', '2023-04-19', [1], 'ephys'),
    ('JEB19', '2023-04-18', [1], 'ephys'),
    ('JGR2', '2021-11-16', [1], 'ephys'),
    ('JGR2', '2021-11-17', [1], 'ephys'),
    ('JGR3', '2021-11-18', [1], 'ephys'),
    # RandomizedDelay_Ephys_Behavior sessions
    ('JEB11', '2022-05-10', [1], 'randdelay'),
    ('JEB11', '2022-05-11', [1], 'randdelay'),
    ('JEB12', '2022-05-12', [1], 'randdelay'),
    ('JEB12', '2022-05-13', [1], 'randdelay'),
    ('JEB23', '2023-10-10', [1], 'randdelay'),
    ('JEB23', '2023-10-11', [1], 'randdelay'),
    ('JEB23', '2023-10-12', [1], 'randdelay'),
    ('JEB23', '2023-10-13', [1], 'randdelay'),
    ('JEB23', '2023-10-18', [1], 'randdelay'),
    ('JEB23', '2023-10-19', [1], 'randdelay'),
    # JEB23 2023-10-20 is commented out in loading script
    ('JEB23', '2023-10-21', [1], 'randdelay'),
    ('JEB24', '2023-10-23', [1], 'randdelay'),
    ('JEB24', '2023-10-24', [1], 'randdelay'),
    ('JEB24', '2023-10-25', [1], 'randdelay'),
    ('JEB24', '2023-10-26', [1], 'randdelay'),
    ('JEB24', '2023-10-27', [1], 'randdelay'),
    ('JEB24', '2023-10-31', [1], 'randdelay'),
    ('JEB24', '2023-11-02', [1], 'randdelay'),
    ('JEB24', '2023-11-03', [1], 'randdelay'),
]

DATA_DIRS = {
    'ephys': '/app/data/Ephys_Behavior/',
    'randdelay': '/app/data/RandomizedDelay_Ephys_Behavior/',
}


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def causal_gaussian_kernel(n):
    """Create causal Gaussian kernel matching mySmooth.m."""
    # gausswin equivalent
    alpha = 2.5  # MATLAB default
    half = (n - 1) / 2
    t = np.arange(n) - half
    kern = np.exp(-0.5 * (alpha * t / half) ** 2)
    # Make causal: zero out first half
    kern[:n // 2] = 0
    kern /= kern.sum()
    return kern

SMOOTH_KERNEL = causal_gaussian_kernel(SMOOTH_WIN)


def smooth_signal(x, kernel=SMOOTH_KERNEL, bctype=BC_TYPE):
    """Smooth 1D signal with causal Gaussian kernel, matching mySmooth.m."""
    n = len(kernel)
    if bctype == 'reflect':
        # Reflect first n points
        pad = x[:n][::-1]
        x_padded = np.concatenate([pad, x])
        result = np.convolve(x_padded, kernel, mode='full')
        # Trim: skip the reflected part, keep same length as original
        result = result[n:n + len(x)]
    elif bctype == 'zeropad':
        pad = np.zeros(n)
        x_padded = np.concatenate([pad, x])
        result = np.convolve(x_padded, kernel, mode='full')
        result = result[n:n + len(x)]
    else:
        result = np.convolve(x, kernel, mode='same')
    return result


def smooth_matrix(X, kernel=SMOOTH_KERNEL, bctype=BC_TYPE):
    """Smooth each column of a matrix along axis 0."""
    out = np.empty_like(X)
    for j in range(X.shape[1]):
        out[:, j] = smooth_signal(X[:, j], kernel, bctype)
    return out


# ============================================================
# DATA LOADING (handles both v7.3 HDF5 and v5.0 scipy formats)
# ============================================================

def read_h5_str(f, ref):
    """Read a MATLAB string stored as uint16 array via HDF5 reference."""
    data = f[ref]
    return ''.join(chr(int(c)) for c in np.array(data).flatten())


def load_session_h5(fpath):
    """Load a MATLAB v7.3 (HDF5) session file."""
    f = h5py.File(fpath, 'r')
    obj_grp = f['obj']
    bp = obj_grp['bp']

    session = {}
    session['_h5file'] = f
    session['ntrials'] = int(bp['Ntrials'][0, 0])

    # Trial info arrays (shape: (1, ntrials) in file)
    session['hit'] = bp['hit'][0, :].astype(bool)
    session['miss'] = bp['miss'][0, :].astype(bool)
    session['no'] = bp['no'][0, :].astype(bool)
    session['R'] = bp['R'][0, :].astype(bool)
    session['L'] = bp['L'][0, :].astype(bool)
    session['autowater'] = bp['autowater'][0, :].astype(bool)
    session['early'] = bp['early'][0, :].astype(bool)

    # Stim enable
    if 'stim' in bp and 'enable' in bp['stim']:
        session['stim_enable'] = bp['stim']['enable'][0, :].astype(bool)
    else:
        session['stim_enable'] = np.zeros(session['ntrials'], dtype=bool)

    # Event times
    ev = bp['ev']
    session['goCue'] = ev['goCue'][0, :]
    session['sample'] = ev['sample'][0, :]
    session['delay'] = ev['delay'][0, :]

    # Spike data per probe
    n_probes = obj_grp['clu'].shape[0]
    session['clu'] = []
    for p in range(n_probes):
        ref = obj_grp['clu'][p, 0]
        probe_grp = f[ref]
        if not isinstance(probe_grp, h5py.Group):
            session['clu'].append([])
            continue
        n_units = probe_grp['quality'].shape[0]
        units = []
        for i in range(n_units):
            q_str = read_h5_str(f, probe_grp['quality'][i, 0]).strip()
            trialtm = np.array(f[probe_grp['trialtm'][i, 0]]).flatten()
            trial = np.array(f[probe_grp['trial'][i, 0]]).flatten().astype(int)
            units.append({'quality': q_str, 'trialtm': trialtm, 'trial': trial})
        session['clu'].append(units)

    # Trajectory data
    session['traj'] = []
    for v in range(obj_grp['traj'].shape[0]):
        ref = obj_grp['traj'][v, 0]
        traj_grp = f[ref]
        n_trials_traj = traj_grp['ts'].shape[0]
        view_data = {'n_trials': n_trials_traj}

        # Feature names (from first trial)
        feat_ref = traj_grp['featNames']
        if feat_ref.shape[0] > 0:
            feat_names = []
            # featNames might be (n_trials, n_features) or (n_features, 1)
            # Try first trial's feature names
            try:
                first_ref = feat_ref[0, 0]
                fn_data = f[first_ref]
                if isinstance(fn_data, h5py.Dataset):
                    # It's a reference to array of feature name references
                    if fn_data.dtype == object:
                        for fi in range(fn_data.shape[1] if fn_data.ndim > 1 else fn_data.shape[0]):
                            idx = (0, fi) if fn_data.ndim > 1 else (fi,)
                            feat_names.append(read_h5_str(f, fn_data[idx]))
                    else:
                        feat_names.append(read_h5_str(f, first_ref))
                elif isinstance(fn_data, h5py.Group):
                    for fi in range(len(fn_data)):
                        feat_names.append(read_h5_str(f, fn_data[fi]))
            except:
                feat_names = []
            view_data['featNames'] = feat_names

        # We store references to load trial data on demand
        view_data['_traj_grp'] = traj_grp
        view_data['_f'] = f
        session['traj'].append(view_data)

    # Probe locations from obj.ex.probe.loc if available
    session['probe_locs'] = {}
    try:
        ex_probe = obj_grp['ex']['probe']
        for p in range(ex_probe['loc'].shape[0]):
            loc_str = read_h5_str(f, ex_probe['loc'][p, 0]).strip()
            session['probe_locs'][p] = loc_str
    except:
        pass

    # Video offset
    try:
        bit_start_bpod = float(np.nanmedian(ev['bitStart'][0, :]))
        sglx = obj_grp['sglx']
        bit_start_sglx = float(np.nanmedian(sglx['bitcode']['bitstart'][0, :])) / float(sglx['fs'][0, 0])
        session['vidshift'] = bit_start_sglx - bit_start_bpod
    except:
        session['vidshift'] = 0.5  # default: 0.5s as in code comments

    return session


def load_session_v5(fpath):
    """Load a MATLAB v5.0 session file."""
    data = sio.loadmat(fpath, squeeze_me=True, struct_as_record=False)
    obj = data['obj']
    bp = obj.bp

    session = {}
    session['ntrials'] = int(bp.Ntrials)
    session['hit'] = np.atleast_1d(bp.hit).astype(bool)
    session['miss'] = np.atleast_1d(bp.miss).astype(bool)
    session['no'] = np.atleast_1d(bp.no).astype(bool)
    session['R'] = np.atleast_1d(bp.R).astype(bool)
    session['L'] = np.atleast_1d(bp.L).astype(bool)
    session['autowater'] = np.atleast_1d(bp.autowater).astype(bool)
    session['early'] = np.atleast_1d(bp.early).astype(bool)

    if hasattr(bp, 'stim') and hasattr(bp.stim, 'enable'):
        session['stim_enable'] = np.atleast_1d(bp.stim.enable).astype(bool)
    else:
        session['stim_enable'] = np.zeros(session['ntrials'], dtype=bool)

    ev = bp.ev
    session['goCue'] = np.atleast_1d(ev.goCue).astype(float)
    session['sample'] = np.atleast_1d(ev.sample).astype(float)
    session['delay'] = np.atleast_1d(ev.delay).astype(float)

    # Spike data - v5 format: clu is flat array of units (single probe assumed)
    clu_raw = obj.clu
    if isinstance(clu_raw, np.ndarray):
        if clu_raw.ndim == 0:
            clu_raw = np.array([clu_raw.item()])
    else:
        clu_raw = [clu_raw]

    units = []
    for u in clu_raw:
        q = u.quality.strip() if hasattr(u, 'quality') else 'unknown'
        trialtm = np.atleast_1d(u.trialtm).astype(float).flatten()
        trial = np.atleast_1d(u.trial).astype(int).flatten()
        units.append({'quality': q, 'trialtm': trialtm, 'trial': trial})
    session['clu'] = [units]  # Single probe

    # Trajectory data
    session['traj'] = []
    if hasattr(obj, 'traj'):
        traj_raw = obj.traj
        if isinstance(traj_raw, np.ndarray):
            for v in range(len(traj_raw)):
                view = traj_raw[v]
                if isinstance(view, np.ndarray) and view.ndim == 0:
                    view = np.array([view.item()])
                elif not isinstance(view, np.ndarray):
                    view = np.array([view])

                n_trials_traj = len(view)
                view_data = {'n_trials': n_trials_traj, '_v5_data': view}

                # Feature names
                try:
                    first_trial = view[0] if n_trials_traj > 0 else None
                    if first_trial is not None and hasattr(first_trial, 'featNames'):
                        fnames = first_trial.featNames
                        if isinstance(fnames, np.ndarray):
                            feat_names = [str(fn).strip() for fn in fnames.flatten()]
                        else:
                            feat_names = [str(fnames).strip()]
                        view_data['featNames'] = feat_names
                except:
                    view_data['featNames'] = []

                session['traj'].append(view_data)

    # Probe locations
    session['probe_locs'] = {}
    if hasattr(obj, 'ex') and hasattr(obj.ex, 'probe') and hasattr(obj.ex.probe, 'loc'):
        locs = obj.ex.probe.loc
        if isinstance(locs, np.ndarray):
            for p in range(len(locs)):
                session['probe_locs'][p] = str(locs[p]).strip()
        else:
            session['probe_locs'][0] = str(locs).strip()

    # Video offset
    try:
        bit_start_bpod = float(np.nanmedian(ev.bitStart))
        sglx = obj.sglx
        bit_start_sglx = float(np.nanmedian(sglx.bitcode.bitstart)) / float(sglx.fs)
        session['vidshift'] = bit_start_sglx - bit_start_bpod
    except:
        session['vidshift'] = 0.5

    session['_h5file'] = None
    return session


def load_session(fpath):
    """Load session, auto-detecting format."""
    with open(fpath, 'rb') as ff:
        header = ff.read(15)
    if b'7.3' in header:
        return load_session_h5(fpath)
    else:
        return load_session_v5(fpath)


def close_session(session):
    """Close any open HDF5 file handles."""
    if session.get('_h5file') is not None:
        try:
            session['_h5file'].close()
        except:
            pass


# ============================================================
# SPIKE PROCESSING
# ============================================================

def filter_clusters(units, excluded_qualities=EXCLUDED_QUALITIES):
    """Filter clusters by quality. Returns indices of good clusters."""
    good_idx = []
    for i, u in enumerate(units):
        q = u['quality'].lower()
        if q not in excluded_qualities:
            good_idx.append(i)
    return good_idx


def align_and_bin_spikes(units, cluster_indices, go_cue_times, ntrials):
    """
    Align spikes to go cue and bin into firing rates.
    Returns: trialdat (n_timepoints, n_units, n_trials)
    """
    n_units = len(cluster_indices)
    trialdat = np.zeros((N_TIMEBINS, n_units, ntrials), dtype=np.float32)

    for ui, ci in enumerate(cluster_indices):
        u = units[ci]
        trialtm = u['trialtm']
        trial_ids = u['trial']

        for t in range(ntrials):
            trial_num = t + 1  # MATLAB 1-indexed
            spike_mask = trial_ids == trial_num
            if not np.any(spike_mask):
                continue

            # Align to go cue
            aligned_times = trialtm[spike_mask] - go_cue_times[t]

            # Bin spikes
            counts, _ = np.histogram(aligned_times, bins=EDGES)
            # Convert to firing rate and smooth
            fr = counts.astype(np.float64) / DT
            trialdat[:, ui, t] = smooth_signal(fr)

    return trialdat


def remove_low_fr_clusters(trialdat, cluster_indices, low_fr=LOW_FR):
    """Remove clusters with mean FR <= low_fr. Returns filtered data and indices."""
    # Mean FR across all timepoints and trials
    mean_frs = trialdat.mean(axis=(0, 2))
    keep = mean_frs > low_fr
    if not np.any(keep):
        return trialdat[:, keep, :], []
    return trialdat[:, keep, :], [ci for ci, k in zip(cluster_indices, keep) if k]


# ============================================================
# VIDEO / KINEMATIC PROCESSING
# ============================================================

def get_traj_trial_h5(view_data, trial_idx):
    """Get trajectory data for a specific trial from HDF5."""
    f = view_data['_f']
    traj_grp = view_data['_traj_grp']
    try:
        ts_ref = traj_grp['ts'][trial_idx, 0]
        ts = np.array(f[ts_ref])  # (n_features, 3, n_frames) or transposed
    except:
        return None, None

    try:
        ft_ref = traj_grp['frameTimes'][trial_idx, 0]
        frame_times = np.array(f[ft_ref]).flatten()
    except:
        n_frames = ts.shape[-1] if ts.ndim == 3 else ts.shape[0]
        frame_times = np.arange(n_frames) / 400.0

    return ts, frame_times


def get_traj_trial_v5(view_data, trial_idx):
    """Get trajectory data for a specific trial from v5.0 format."""
    trials = view_data['_v5_data']
    if trial_idx >= len(trials):
        return None, None
    trial = trials[trial_idx]
    ts = np.array(trial.ts) if hasattr(trial, 'ts') else None
    if ts is None:
        return None, None
    frame_times = np.array(trial.frameTimes).flatten() if hasattr(trial, 'frameTimes') else np.arange(ts.shape[0]) / 400.0
    return ts, frame_times


def compute_tongue_velocity(session, go_cue_times, vidshift):
    """
    Compute tongue speed for each trial, interpolated to neural timebase.
    Returns: (n_timepoints, n_trials) array. NaN where tongue not visible.
    """
    ntrials = session['ntrials']
    tongue_speed = np.full((N_TIMEBINS, ntrials), np.nan, dtype=np.float32)

    # Try side cam (view 0) first for tongue
    if len(session['traj']) < 1:
        return tongue_speed

    view_data = session['traj'][0]  # side cam
    feat_names = view_data.get('featNames', [])

    # Find tongue feature index
    tongue_idx = None
    for fi, fn in enumerate(feat_names):
        if fn.lower() == 'tongue':
            tongue_idx = fi
            break

    if tongue_idx is None:
        return tongue_speed

    is_h5 = '_traj_grp' in view_data

    for t in range(ntrials):
        if is_h5:
            ts, frame_times = get_traj_trial_h5(view_data, t)
        else:
            ts, frame_times = get_traj_trial_v5(view_data, t)

        if ts is None or frame_times is None:
            continue

        # ts shape: HDF5 = (n_features, 3, n_frames), v5 = (n_frames, 3, n_features)
        if is_h5:
            # (n_features, 3, n_frames) -> extract tongue x, y
            x = ts[tongue_idx, 0, :]  # x position
            y = ts[tongue_idx, 1, :]  # y position
            conf = ts[tongue_idx, 2, :] if ts.shape[1] >= 3 else np.ones_like(x)
        else:
            if ts.ndim == 3:
                x = ts[:, 0, tongue_idx]
                y = ts[:, 1, tongue_idx]
                conf = ts[:, 2, tongue_idx] if ts.shape[1] >= 3 else np.ones_like(x)
            else:
                continue

        # Low confidence = tongue not visible (DLC convention: conf < 0.5 or similar)
        # For tongue, keep NaNs as-is (no filling, per reference code)

        # Compute velocity
        xvel = np.gradient(x)
        yvel = np.gradient(y)
        speed = np.sqrt(xvel**2 + yvel**2)

        # NaN where tongue not visible (low confidence)
        speed[conf < 0.1] = np.nan

        # Align to go cue and interpolate to neural timebase
        aligned_ft = frame_times - vidshift - go_cue_times[t]
        taxis = TIME_AXIS

        # Interpolate
        try:
            interp_speed = np.interp(taxis, aligned_ft, speed, left=np.nan, right=np.nan)
            # Keep NaNs where original was NaN
            tongue_speed[:, t] = interp_speed
        except:
            pass

    return tongue_speed


def compute_paw_velocity(session, go_cue_times, vidshift):
    """
    Compute paw speed for each trial from bottom cam.
    Returns: (n_timepoints, n_trials) array. NaN where paw not visible.
    """
    ntrials = session['ntrials']
    paw_speed = np.full((N_TIMEBINS, ntrials), np.nan, dtype=np.float32)

    # Bottom cam is view 1
    if len(session['traj']) < 2:
        return paw_speed

    view_data = session['traj'][1]  # bottom cam
    feat_names = view_data.get('featNames', [])

    # Find paw feature index (try top_paw or bottom_paw)
    paw_idx = None
    for fi, fn in enumerate(feat_names):
        if 'paw' in fn.lower():
            paw_idx = fi
            break

    if paw_idx is None:
        return paw_speed

    is_h5 = '_traj_grp' in view_data

    for t in range(ntrials):
        if is_h5:
            ts, frame_times = get_traj_trial_h5(view_data, t)
        else:
            ts, frame_times = get_traj_trial_v5(view_data, t)

        if ts is None or frame_times is None:
            continue

        if is_h5:
            x = ts[paw_idx, 0, :]
            y = ts[paw_idx, 1, :]
            conf = ts[paw_idx, 2, :] if ts.shape[1] >= 3 else np.ones_like(x)
        else:
            if ts.ndim == 3:
                x = ts[:, 0, paw_idx]
                y = ts[:, 1, paw_idx]
                conf = ts[:, 2, paw_idx] if ts.shape[1] >= 3 else np.ones_like(x)
            else:
                continue

        xvel = np.gradient(x)
        yvel = np.gradient(y)
        speed = np.sqrt(xvel**2 + yvel**2)
        speed[conf < 0.1] = np.nan

        # For non-tongue features, fill missing with nearest (per reference code)
        mask = np.isnan(speed)
        if mask.any() and not mask.all():
            # Forward-fill then back-fill
            idx = np.where(~mask)[0]
            if len(idx) > 0:
                speed = np.interp(np.arange(len(speed)), idx, speed[idx])

        aligned_ft = frame_times - vidshift - go_cue_times[t]
        try:
            interp_speed = np.interp(TIME_AXIS, aligned_ft, speed, left=np.nan, right=np.nan)
            paw_speed[:, t] = interp_speed
        except:
            pass

    return paw_speed


def load_motion_energy(session, anm, date, data_dir, go_cue_times, vidshift):
    """Load motion energy and align to neural timebase."""
    ntrials = session['ntrials']
    me_data = np.full((N_TIMEBINS, ntrials), np.nan, dtype=np.float32)

    # Find motion energy file
    me_pattern = os.path.join(data_dir, f'motionEnergy_{anm}_{date}.mat')
    me_files = glob.glob(me_pattern)
    if not me_files:
        return me_data

    me_path = me_files[0]

    try:
        # Try v7.3 first
        with open(me_path, 'rb') as ff:
            header = ff.read(15)

        if b'7.3' in header:
            f = h5py.File(me_path, 'r')
            me_obj = f['me']
            me_raw_data = me_obj['data']
            n_me_trials = me_raw_data.shape[1] if me_raw_data.ndim > 1 else me_raw_data.shape[0]

            for t in range(min(ntrials, n_me_trials)):
                try:
                    if me_raw_data.dtype == object:
                        ref = me_raw_data[0, t] if me_raw_data.ndim > 1 else me_raw_data[t]
                        trial_me = np.array(f[ref]).flatten()
                    else:
                        trial_me = me_raw_data[:, t] if me_raw_data.ndim > 1 else me_raw_data[:]
                except:
                    continue

                # Get frame times from traj for alignment
                if len(session['traj']) > 0 and '_traj_grp' in session['traj'][0]:
                    view_data = session['traj'][0]
                    try:
                        ft_ref = view_data['_traj_grp']['frameTimes'][t, 0]
                        frame_times = np.array(view_data['_f'][ft_ref]).flatten()
                    except:
                        frame_times = np.arange(len(trial_me)) / 400.0
                else:
                    frame_times = np.arange(len(trial_me)) / 400.0

                # Align and interpolate
                if len(frame_times) != len(trial_me):
                    min_len = min(len(frame_times), len(trial_me))
                    frame_times = frame_times[:min_len]
                    trial_me = trial_me[:min_len]

                aligned_ft = frame_times - vidshift - go_cue_times[t]
                try:
                    me_interp = np.interp(TIME_AXIS, aligned_ft, trial_me)
                    me_data[:, t] = me_interp
                except:
                    pass

            f.close()
        else:
            me_mat = sio.loadmat(me_path, squeeze_me=False, struct_as_record=True)
            me_arr = me_mat['me']

            # Unwrap nested structs: some files have me.data.data..., others are plain arrays
            def _unwrap_me_data(arr):
                if hasattr(arr, 'dtype') and arr.dtype.names and 'data' in arr.dtype.names:
                    inner = arr['data']
                    if inner.shape == (1, 1):
                        return _unwrap_me_data(inner[0, 0])
                    return inner
                return arr

            me_top = me_arr[0, 0] if me_arr.shape == (1, 1) else me_arr
            raw = _unwrap_me_data(me_top)
            if not hasattr(raw, 'shape'):
                raw = np.asarray(raw)
            # raw should be (nTrials, 1) object array or (nTrials,) array
            if raw.ndim == 2:
                me_raw = [np.asarray(raw[t, 0]).flatten() for t in range(raw.shape[0])]
            else:
                me_raw = [np.asarray(raw[t]).flatten() for t in range(raw.shape[0])]

            if hasattr(me_raw, '__len__'):
                for t in range(min(ntrials, len(me_raw))):
                    trial_me = np.atleast_1d(me_raw[t]).astype(float).flatten()

                    # Get frame times
                    if len(session['traj']) > 0 and '_v5_data' in session['traj'][0]:
                        v5data = session['traj'][0]['_v5_data']
                        if t < len(v5data) and hasattr(v5data[t], 'frameTimes'):
                            frame_times = np.atleast_1d(v5data[t].frameTimes).flatten()
                        else:
                            frame_times = np.arange(len(trial_me)) / 400.0
                    else:
                        frame_times = np.arange(len(trial_me)) / 400.0

                    if len(frame_times) != len(trial_me):
                        min_len = min(len(frame_times), len(trial_me))
                        frame_times = frame_times[:min_len]
                        trial_me = trial_me[:min_len]

                    aligned_ft = frame_times - vidshift - go_cue_times[t]
                    try:
                        me_interp = np.interp(TIME_AXIS, aligned_ft, trial_me)
                        me_data[:, t] = me_interp
                    except:
                        pass

    except Exception as e:
        print(f"    Warning: Could not load motion energy: {e}")

    # Fill NaNs with nearest (per reference code)
    for t in range(ntrials):
        col = me_data[:, t]
        mask = np.isnan(col)
        if mask.any() and not mask.all():
            idx = np.where(~mask)[0]
            me_data[:, t] = np.interp(np.arange(len(col)), idx, col[idx])

    return me_data


# ============================================================
# DISCRETIZATION
# ============================================================

def discretize_velocity(speed_matrix):
    """
    Discretize velocity per session using 50th percentile threshold.
    speed_matrix: (n_timepoints, n_trials), NaN = not visible
    Returns: (n_timepoints, n_trials) with values 0, 1, 2
    """
    result = np.full_like(speed_matrix, 2, dtype=np.int8)  # default: not visible

    # Compute threshold from visible (non-NaN) values
    valid = speed_matrix[~np.isnan(speed_matrix)]
    if len(valid) == 0:
        return result

    threshold = np.percentile(valid, 50)

    visible = ~np.isnan(speed_matrix)
    result[visible & (speed_matrix < threshold)] = 0
    result[visible & (speed_matrix >= threshold)] = 1

    return result


def discretize_motion_energy(me_matrix):
    """
    Discretize motion energy per session using 50th percentile.
    me_matrix: (n_timepoints, n_trials), NaN = no video
    Returns: (n_timepoints, n_trials) with values 0, 1, 2
    """
    result = np.full_like(me_matrix, 2, dtype=np.int8)  # default: no video

    valid = me_matrix[~np.isnan(me_matrix)]
    if len(valid) == 0:
        return result

    threshold = np.percentile(valid, 50)

    visible = ~np.isnan(me_matrix)
    result[visible & (me_matrix < threshold)] = 0
    result[visible & (me_matrix >= threshold)] = 1

    return result


# ============================================================
# MAIN PROCESSING
# ============================================================

def process_session(anm, date, probes_matlab, data_dir_key, show_processing=False, session_idx=0):
    """Process a single session. Returns dict with neural, input, output data, or None if excluded."""
    t0 = time.time()

    data_dir = DATA_DIRS[data_dir_key]
    fpath = os.path.join(data_dir, f'data_structure_{anm}_{date}.mat')

    if not os.path.exists(fpath):
        print(f"  File not found: {fpath}")
        return None

    print(f"  Loading {anm}_{date}...", end=' ', flush=True)
    session = load_session(fpath)
    ntrials = session['ntrials']
    go_cue = session['goCue']

    # Convert MATLAB probe indices to Python (0-indexed)
    probe_indices = [p - 1 for p in probes_matlab]

    # --- 1. Filter clusters by quality and collect from specified probes ---
    all_units = []
    all_unit_probe = []  # track which probe each unit comes from
    for pi in probe_indices:
        if pi >= len(session['clu']):
            continue
        probe_units = session['clu'][pi]
        good_idx = filter_clusters(probe_units)
        for gi in good_idx:
            all_units.append(probe_units[gi])
            all_unit_probe.append(pi)

    if len(all_units) == 0:
        print(f"no good units")
        close_session(session)
        return None

    # --- 2. Align spikes and compute firing rates ---
    # We work with all_units directly
    trialdat = np.zeros((N_TIMEBINS, len(all_units), ntrials), dtype=np.float32)
    for ui, u in enumerate(all_units):
        trialtm = u['trialtm']
        trial_ids = u['trial']
        for t in range(ntrials):
            trial_num = t + 1
            spike_mask = trial_ids == trial_num
            if not np.any(spike_mask):
                continue
            aligned_times = trialtm[spike_mask] - go_cue[t]
            counts, _ = np.histogram(aligned_times, bins=EDGES)
            fr = counts.astype(np.float64) / DT
            trialdat[:, ui, t] = smooth_signal(fr)

    # --- 3. Remove low FR clusters ---
    mean_frs = trialdat.mean(axis=(0, 2))
    keep_mask = mean_frs > LOW_FR
    trialdat = trialdat[:, keep_mask, :]
    kept_probe_indices = [p for p, k in zip(all_unit_probe, keep_mask) if k]
    n_neurons = trialdat.shape[1]

    if n_neurons < MIN_UNITS:
        print(f"{n_neurons} units after filtering (< {MIN_UNITS}), skipping")
        close_session(session)
        return None

    # --- 4. Check trial inclusion criteria ---
    # Count R-hit and L-hit DR trials (no stim, no autowater, no early)
    valid_mask = ~session['stim_enable'] & ~session['autowater'] & ~session['early']
    r_hit = session['R'] & session['hit'] & valid_mask
    l_hit = session['L'] & session['hit'] & valid_mask
    n_r_hit = r_hit.sum()
    n_l_hit = l_hit.sum()

    if n_r_hit < 40 or n_l_hit < 40:
        print(f"insufficient DR trials (R-hit={n_r_hit}, L-hit={n_l_hit}), skipping")
        close_session(session)
        return None

    # --- 5. Exclude stim-enabled trials ---
    trial_mask = ~session['stim_enable']
    trial_indices = np.where(trial_mask)[0]

    if len(trial_indices) < 2:
        print(f"too few trials after stim exclusion, skipping")
        close_session(session)
        return None

    trialdat = trialdat[:, :, trial_indices]

    # --- 6. Compute output variables ---
    # Lick direction: per-trial
    lick_dir = np.zeros(len(trial_indices), dtype=np.int8)
    for i, ti in enumerate(trial_indices):
        if session['R'][ti] and (session['hit'][ti] or session['miss'][ti]):
            lick_dir[i] = 1  # right
        elif session['L'][ti] and (session['hit'][ti] or session['miss'][ti]):
            lick_dir[i] = 0  # left
        else:
            lick_dir[i] = 2  # none (no response / ignore)

    # Behavioral context: per-trial
    context = np.zeros(len(trial_indices), dtype=np.int8)
    for i, ti in enumerate(trial_indices):
        if session['autowater'][ti]:
            context[i] = 0  # WC
        else:
            context[i] = 1  # DR

    # Outcome: per-trial
    outcome = np.zeros(len(trial_indices), dtype=np.int8)
    for i, ti in enumerate(trial_indices):
        if session['hit'][ti]:
            outcome[i] = 1  # correct
        elif session['miss'][ti]:
            outcome[i] = 0  # incorrect
        else:
            outcome[i] = 2  # ignore

    # --- 7. Compute behavioral outputs (time-varying) ---
    vidshift = session['vidshift']
    go_cue_filtered = go_cue[trial_indices]

    # Tongue velocity
    tongue_speed_all = compute_tongue_velocity(session, go_cue, vidshift)
    tongue_speed = tongue_speed_all[:, trial_indices]
    tongue_disc = discretize_velocity(tongue_speed)

    # Paw velocity
    paw_speed_all = compute_paw_velocity(session, go_cue, vidshift)
    paw_speed = paw_speed_all[:, trial_indices]
    paw_disc = discretize_velocity(paw_speed)

    # Motion energy
    me_raw = load_motion_energy(session, anm, date, data_dir, go_cue, vidshift)
    me = me_raw[:, trial_indices]
    me_disc = discretize_motion_energy(me)

    # --- 8. Determine brain regions ---
    brain_regions_for_units = []
    for pi in kept_probe_indices:
        loc = session['probe_locs'].get(pi, '')
        if 'ALM' in loc.upper():
            brain_regions_for_units.append('ALM')
        elif 'M1TJ' in loc.upper() or 'TJM1' in loc.upper():
            brain_regions_for_units.append('tjM1')
        else:
            brain_regions_for_units.append('ALM')  # default

    elapsed = time.time() - t0
    print(f"{n_neurons} neurons, {len(trial_indices)} trials, {elapsed:.1f}s")

    # --- 9. Optional visualization ---
    if show_processing:
        plot_processing(anm, date, trialdat, tongue_speed, paw_speed, me,
                        tongue_disc, paw_disc, me_disc, lick_dir, context, outcome,
                        go_cue_filtered, session_idx)

    close_session(session)

    # --- 10. Package results ---
    n_trials_out = len(trial_indices)
    neural_trials = []
    input_trials = []
    output_trials = []

    for i in range(n_trials_out):
        # Neural: (n_neurons, n_timepoints)
        neural_trials.append(trialdat[:, :, i].T.astype(np.float32))

        # Input: time from go cue (1, n_timepoints)
        input_trials.append(TIME_AXIS.reshape(1, -1).astype(np.float32))

        # Output: (n_output, n_timepoints) for time-varying, (n_output,) for per-trial
        # Construct output array: 6 outputs
        # 0: lick_direction (per-trial)
        # 1: behavioral_context (per-trial)
        # 2: outcome (per-trial)
        # 3: tongue_velocity (time-varying)
        # 4: paw_velocity (time-varying)
        # 5: motion_energy (time-varying)
        out = np.zeros((6, N_TIMEBINS), dtype=np.int8)
        out[0, :] = lick_dir[i]
        out[1, :] = context[i]
        out[2, :] = outcome[i]
        out[3, :] = tongue_disc[:, i]
        out[4, :] = paw_disc[:, i]
        out[5, :] = me_disc[:, i]
        output_trials.append(out)

    return {
        'neural': neural_trials,
        'input': input_trials,
        'output': output_trials,
        'anm': anm,
        'date': date,
        'brain_regions': brain_regions_for_units,
        'n_neurons': n_neurons,
        'n_trials': n_trials_out,
    }


def plot_processing(anm, date, trialdat, tongue_speed, paw_speed, me,
                    tongue_disc, paw_disc, me_disc, lick_dir, context, outcome,
                    go_cue, session_idx):
    """Plot processing steps for visual verification."""
    fig, axes = plt.subplots(4, 3, figsize=(20, 16))
    fig.suptitle(f'Processing: {anm}_{date}', fontsize=14)

    # Pick example trial
    trial_idx = min(5, trialdat.shape[2] - 1)

    # Row 0: Neural activity (heatmap of first 20 neurons)
    n_show = min(20, trialdat.shape[1])
    axes[0, 0].imshow(trialdat[:, :n_show, trial_idx].T, aspect='auto',
                       extent=[TMIN, TMAX, 0, n_show], cmap='viridis')
    axes[0, 0].set_title(f'Neural FR trial {trial_idx}')
    axes[0, 0].set_xlabel('Time from go cue (s)')
    axes[0, 0].set_ylabel('Neuron')
    axes[0, 0].axvline(0, color='r', linestyle='--', alpha=0.5)

    # Average FR across neurons
    axes[0, 1].plot(TIME_AXIS, trialdat[:, :, trial_idx].mean(axis=1))
    axes[0, 1].set_title('Mean FR across neurons')
    axes[0, 1].axvline(0, color='r', linestyle='--', alpha=0.5)

    # FR distribution
    axes[0, 2].hist(trialdat.mean(axis=(0, 2)), bins=30)
    axes[0, 2].set_title('Mean FR distribution (neurons)')
    axes[0, 2].set_xlabel('FR (Hz)')

    # Row 1: Tongue velocity
    axes[1, 0].plot(TIME_AXIS, tongue_speed[:, trial_idx])
    axes[1, 0].set_title(f'Tongue speed trial {trial_idx}')
    axes[1, 0].axvline(0, color='r', linestyle='--', alpha=0.5)

    axes[1, 1].plot(TIME_AXIS, tongue_disc[:, trial_idx])
    axes[1, 1].set_title(f'Tongue discretized trial {trial_idx}')
    axes[1, 1].set_yticks([0, 1, 2])
    axes[1, 1].set_yticklabels(['<50th', '>=50th', 'not visible'])

    # Tongue disc distribution
    vals, counts = np.unique(tongue_disc, return_counts=True)
    axes[1, 2].bar(vals, counts / counts.sum())
    axes[1, 2].set_title('Tongue disc distribution')
    axes[1, 2].set_xticks([0, 1, 2])

    # Row 2: Paw velocity and motion energy
    axes[2, 0].plot(TIME_AXIS, paw_speed[:, trial_idx])
    axes[2, 0].set_title(f'Paw speed trial {trial_idx}')
    axes[2, 0].axvline(0, color='r', linestyle='--', alpha=0.5)

    axes[2, 1].plot(TIME_AXIS, me[:, trial_idx])
    axes[2, 1].set_title(f'Motion energy trial {trial_idx}')
    axes[2, 1].axvline(0, color='r', linestyle='--', alpha=0.5)

    # ME disc distribution
    vals, counts = np.unique(me_disc, return_counts=True)
    axes[2, 2].bar(vals, counts / counts.sum())
    axes[2, 2].set_title('ME disc distribution')
    axes[2, 2].set_xticks([0, 1, 2])

    # Row 3: Output distributions
    vals, counts = np.unique(lick_dir, return_counts=True)
    axes[3, 0].bar(vals, counts / counts.sum())
    axes[3, 0].set_title('Lick direction')
    axes[3, 0].set_xticks([0, 1, 2])
    axes[3, 0].set_xticklabels(['left', 'right', 'none'])

    vals, counts = np.unique(context, return_counts=True)
    axes[3, 1].bar(vals, counts / counts.sum())
    axes[3, 1].set_title('Context')
    axes[3, 1].set_xticks([0, 1])
    axes[3, 1].set_xticklabels(['WC', 'DR'])

    vals, counts = np.unique(outcome, return_counts=True)
    axes[3, 2].bar(vals, counts / counts.sum())
    axes[3, 2].set_title('Outcome')
    axes[3, 2].set_xticks([0, 1, 2])
    axes[3, 2].set_xticklabels(['incorrect', 'correct', 'ignore'])

    plt.tight_layout()
    plt.savefig(f'processing_{anm}_{date}.png', dpi=100)
    plt.close()
    print(f"    Saved processing_{anm}_{date}.png")


def main():
    parser = argparse.ArgumentParser(description='Convert data to decoder format')
    parser.add_argument('outfile', type=str, help='Output pickle file path')
    parser.add_argument('--full', action='store_true', default=True, help='Process all sessions (default)')
    parser.add_argument('--sample', action='store_true', help='Process only 2 sessions for testing')
    parser.add_argument('--show-processing', action='store_true', help='Plot processing visualizations')
    args = parser.parse_args()

    if args.sample:
        sessions_to_process = SESSION_META[:2]
        print(f"SAMPLE MODE: Processing {len(sessions_to_process)} sessions")
    else:
        sessions_to_process = SESSION_META
        print(f"FULL MODE: Processing {len(sessions_to_process)} sessions")

    # Process all sessions
    all_results = []
    t_total = time.time()

    for idx, (anm, date, probes, ddir) in enumerate(sessions_to_process):
        print(f"[{idx+1}/{len(sessions_to_process)}] ", end='')
        result = process_session(anm, date, probes, ddir,
                                 show_processing=args.show_processing,
                                 session_idx=idx)
        if result is not None:
            all_results.append(result)

    print(f"\nProcessed {len(all_results)} sessions in {time.time()-t_total:.1f}s")

    if len(all_results) == 0:
        print("ERROR: No sessions passed filtering. Cannot create output.")
        return

    # Build output data structure
    subjects = sorted(set(r['anm'] for r in all_results))
    subject_idx = np.array([subjects.index(r['anm']) for r in all_results])

    # Collect all brain regions
    all_br = set()
    for r in all_results:
        all_br.update(r['brain_regions'])
    brain_regions = sorted(all_br)

    brain_region_idx = []
    for r in all_results:
        idx = np.array([brain_regions.index(br) for br in r['brain_regions']])
        brain_region_idx.append(idx)

    data = {
        'neural': [r['neural'] for r in all_results],
        'input': [r['input'] for r in all_results],
        'output': [r['output'] for r in all_results],
        'subjects': subjects,
        'subject_idx': subject_idx,
        'brain_regions': brain_regions,
        'brain_region_idx': brain_region_idx,
        'input_names': ['time_from_go_cue'],
        'output_names': ['lick_direction', 'behavioral_context', 'outcome',
                         'tongue_velocity', 'paw_velocity', 'motion_energy'],
        'output_values': [
            ['left', 'right', 'none'],       # lick_direction
            ['WC', 'DR'],                     # behavioral_context
            ['incorrect', 'correct', 'ignore'],  # outcome
            ['below_50pct', 'above_50pct', 'not_visible'],  # tongue_velocity
            ['below_50pct', 'above_50pct', 'not_visible'],  # paw_velocity
            ['below_50pct', 'above_50pct', 'no_video'],     # motion_energy
        ],
        'metadata': {
            'task_description': 'Two-context task (delayed-response + water-cued) and randomized delay task. '
                                'Mice perform directional licking guided by auditory cues (DR) or water presentation (WC).',
            'time_bin_size': DT * 1000,  # in ms
            'temporal_alignment_event': 'Go cue onset',
            'off_start': TMIN,
            'off_end': TMAX,
            'align_event': ALIGN_EVENT,
            'smoothing': f'Causal Gaussian kernel, {SMOOTH_WIN}-point window, {BC_TYPE} boundary',
            'quality_filter': f'Excluded: {EXCLUDED_QUALITIES}',
            'fr_threshold': f'> {LOW_FR} Hz',
            'session_info': [
                {'animal': r['anm'], 'date': r['date'], 'n_neurons': r['n_neurons'], 'n_trials': r['n_trials']}
                for r in all_results
            ],
        }
    }

    # Print summary
    total_neurons = sum(r['n_neurons'] for r in all_results)
    total_trials = sum(r['n_trials'] for r in all_results)
    print(f"\n=== Summary ===")
    print(f"Sessions: {len(all_results)}")
    print(f"Subjects: {len(subjects)} ({subjects})")
    print(f"Total neurons: {total_neurons}")
    print(f"Total trials: {total_trials}")
    print(f"Brain regions: {brain_regions}")
    print(f"Time bins: {N_TIMEBINS} ({DT*1000:.0f} ms)")
    print(f"Outputs: {data['output_names']}")

    # Save
    print(f"\nSaving to {args.outfile}...")
    with open(args.outfile, 'wb') as f:
        pickle.dump(data, f, protocol=4)
    fsize = os.path.getsize(args.outfile) / (1024**2)
    print(f"Saved ({fsize:.1f} MB)")


if __name__ == '__main__':
    main()
