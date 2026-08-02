#!/usr/bin/env python3
"""
Convert Hasnain, Birnbaum et al. (Nat Neuro 2024) data to decoder-compatible format.
Loads MATLAB v7.3 (HDF5) data files, processes neural and behavioral data following
the reference code pipeline, and saves to pickle format.

Usage:
    python -u convert_data.py <outpicklefile> [--full | --sample] [--show-processing]
"""

import os
import sys
import time
import argparse
import pickle
import warnings
import numpy as np
import h5py
from scipy.signal import windows as sig_windows
from scipy.interpolate import interp1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# =============================================================================
# Session definitions from loading scripts
# =============================================================================
# (animal, date, probes_for_ALM, data_dir_key)
# data_dir_key: 'ephys' -> Ephys_Behavior, 'random' -> RandomizedDelay_Ephys_Behavior
SESSION_DEFS = [
    # Ephys_Behavior sessions (DR task, some with two-context)
    ('EKH1', '2021-08-07', [2], 'ephys'),
    ('EKH3', '2021-08-11', [2], 'ephys'),
    ('JEB6', '2021-04-18', [2], 'ephys'),
    ('JEB7', '2021-04-29', [1], 'ephys'),
    ('JEB7', '2021-04-30', [1], 'ephys'),
    ('JGR2', '2021-11-16', [1], 'ephys'),
    ('JGR2', '2021-11-17', [1], 'ephys'),
    ('JGR3', '2021-11-18', [1], 'ephys'),
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
    ('JEB19', '2023-04-18', [1], 'ephys'),
    ('JEB19', '2023-04-19', [1], 'ephys'),
    ('JEB19', '2023-04-20', [1], 'ephys'),
    ('JEB19', '2023-04-21', [1], 'ephys'),
    # RandomizedDelay sessions
    ('JEB11', '2022-05-10', [1], 'random'),
    ('JEB11', '2022-05-11', [1], 'random'),
    ('JEB12', '2022-05-12', [1], 'random'),
    ('JEB12', '2022-05-13', [1], 'random'),
    ('JEB23', '2023-10-10', [1], 'random'),
    ('JEB23', '2023-10-11', [1], 'random'),
    ('JEB23', '2023-10-12', [1], 'random'),
    ('JEB23', '2023-10-13', [1], 'random'),
    ('JEB23', '2023-10-18', [1], 'random'),
    ('JEB23', '2023-10-19', [1], 'random'),
    ('JEB23', '2023-10-21', [1], 'random'),
    ('JEB24', '2023-10-23', [1], 'random'),
    ('JEB24', '2023-10-24', [1], 'random'),
    ('JEB24', '2023-10-25', [1], 'random'),
    ('JEB24', '2023-10-26', [1], 'random'),
    ('JEB24', '2023-10-27', [1], 'random'),
    ('JEB24', '2023-10-31', [1], 'random'),
    ('JEB24', '2023-11-02', [1], 'random'),
    ('JEB24', '2023-11-03', [1], 'random'),
]

DATA_ROOT = '/app/data'
DATA_DIRS = {
    'ephys': os.path.join(DATA_ROOT, 'Ephys_Behavior'),
    'random': os.path.join(DATA_ROOT, 'RandomizedDelay_Ephys_Behavior'),
}

# Processing parameters (matching WorkingWithDataObjs.m tutorial)
PARAMS = {
    'align_event': 'goCue',
    'tmin': -2.5,
    'tmax': 2.5,
    'dt': 1.0 / 100,  # 10 ms bins
    'smooth_window': 15,
    'smooth_bctype': 'reflect',
    'low_fr': 1.0,  # Hz, remove neurons below this
    'min_units': 10,  # minimum units per session
    'quality_exclude': {'garbage', 'noisy', 'gabrga', 'real?'},
}


# =============================================================================
# HDF5 utility functions
# =============================================================================

def h5_read_string(f, ref_or_dataset):
    """Read a string from HDF5, handling object references and various encodings."""
    if isinstance(ref_or_dataset, h5py.h5r.Reference):
        ds = f[ref_or_dataset]
    else:
        ds = ref_or_dataset
    val = ds[()]
    if isinstance(val, bytes):
        return val.decode('utf-8', errors='replace')
    if isinstance(val, np.ndarray):
        if val.dtype.kind == 'O':
            # array of references
            return ''.join(chr(int(f[v][()].flat[0])) for v in val.flat)
        if val.dtype.kind in ('U', 'S'):
            return str(val.flat[0])
        # numeric array that represents characters (uint16)
        if val.dtype.kind in ('u', 'i') and val.size < 200:
            try:
                return ''.join(chr(int(c)) for c in val.flat)
            except (ValueError, OverflowError):
                pass
    if isinstance(val, (np.integer, np.floating)):
        return str(val)
    return str(val)


def h5_read_array(f, ref_or_dataset):
    """Read a numeric array from HDF5, handling object references."""
    if isinstance(ref_or_dataset, h5py.h5r.Reference):
        ds = f[ref_or_dataset]
    else:
        ds = ref_or_dataset
    val = ds[()]
    if isinstance(val, np.ndarray):
        return val.astype(np.float64)
    return np.array([float(val)])


def h5_read_cell_array_of_arrays(f, dataset):
    """Read a MATLAB cell array of numeric arrays from HDF5."""
    refs = dataset[()]
    result = []
    for ref in refs.flat:
        arr = h5_read_array(f, ref)
        result.append(arr)
    return result


# =============================================================================
# Signal processing functions (matching MATLAB reference code)
# =============================================================================

def causal_gaussian_smooth(x, N, bctype='reflect'):
    """
    Replicate mySmooth.m: causal Gaussian smoothing.
    x: (time,) or (time, features) array
    N: window size
    bctype: 'reflect', 'zeropad', or 'none'
    """
    if N <= 1:
        return x

    was_1d = (x.ndim == 1)
    if was_1d:
        x = x[:, np.newaxis]

    # Build kernel
    kern = np.array(sig_windows.gaussian(N, std=(N - 1) / (2 * 2.5)))
    # Make causal: zero out first half
    kern[:N // 2] = 0
    kern = kern / kern.sum()

    # Handle boundary conditions
    if bctype == 'reflect':
        x_filt = np.concatenate([x[:N, :], x], axis=0)
        trim = N
    elif bctype == 'zeropad':
        x_filt = np.concatenate([np.zeros((N, x.shape[1])), x], axis=0)
        trim = N
    else:
        x_filt = x
        trim = 0

    # Convolve each column
    out = np.zeros_like(x_filt)
    for j in range(x_filt.shape[1]):
        out[:, j] = np.convolve(x_filt[:, j], kern, mode='same')
    out = out[trim:, :]

    if was_1d:
        return out[:, 0]
    return out


# =============================================================================
# Data loading functions
# =============================================================================

def _detect_file_format(data_fn):
    """Detect if a .mat file is HDF5 (v7.3) or v5 format."""
    try:
        f = h5py.File(data_fn, 'r')
        f.close()
        return 'hdf5'
    except Exception:
        return 'v5'


def _load_raw_data_v5(data_fn, probes):
    """Load raw data from MATLAB v5 format file. Returns common dict format."""
    import scipy.io as sio
    mat = sio.loadmat(data_fn, squeeze_me=False)
    obj = mat['obj']

    # --- bp ---
    bp_raw = obj['bp'][0, 0]
    bp = {}
    bp['Ntrials'] = int(bp_raw['Ntrials'][0, 0].flat[0])
    ntrials = bp['Ntrials']

    for field in ['hit', 'miss', 'no', 'early', 'L', 'R', 'autowater']:
        bp[field] = bp_raw[field][0, 0].flatten().astype(np.float64)[:ntrials]

    stim_enable = np.zeros(ntrials)
    if 'stim' in bp_raw.dtype.names:
        stim_raw = bp_raw['stim'][0, 0]
        if 'enable' in stim_raw.dtype.names:
            stim_enable = stim_raw['enable'][0, 0].flatten().astype(np.float64)[:ntrials]
    bp['stim_enable'] = stim_enable

    ev = {}
    ev_raw = bp_raw['ev'][0, 0]
    for field in ['goCue', 'sample', 'delay']:
        ev[field] = ev_raw[field][0, 0].flatten().astype(np.float64)[:ntrials]
    bp['ev'] = ev

    # --- vidshift ---
    vidshift = 0.5
    try:
        bit_start_arr = ev_raw['bitStart'][0, 0].flatten().astype(np.float64)
        bit_start = np.nanmedian(bit_start_arr)
        sglx_raw = obj['sglx'][0, 0]
        fs = sglx_raw['fs'][0, 0].flat[0]
        bitcode_raw = sglx_raw['bitcode'][0, 0]
        bitcode_bitstart = np.nanmedian(bitcode_raw['bitstart'][0, 0].flatten().astype(np.float64))
        vid_file_offset = bitcode_bitstart / fs
        vidshift = vid_file_offset - bit_start
    except Exception:
        pass

    # --- clu ---
    clu_raw = obj['clu'][0, 0]
    all_spike_times = []
    all_spike_trials = []
    all_qualities = []

    for probe_num in probes:
        probe_idx = probe_num - 1
        probe_data = clu_raw[0, probe_idx]
        n_units = probe_data.shape[1] if probe_data.ndim > 1 else probe_data.shape[0]
        for i in range(n_units):
            unit = probe_data[0, i] if probe_data.ndim > 1 else probe_data[i]
            quality = str(unit['quality'].flat[0]).strip().lower() if unit['quality'].size > 0 else ''
            trialtm = unit['trialtm'].flatten().astype(np.float64)
            trial = unit['trial'].flatten().astype(np.float64)
            all_qualities.append(quality)
            all_spike_times.append(trialtm)
            all_spike_trials.append(trial)

    # --- traj (for velocity computation) ---
    traj_data = None
    try:
        traj_raw = obj['traj'][0, 0]
        traj_data = {'views': []}
        for view_idx in range(traj_raw.shape[1]):
            view_data = traj_raw[0, view_idx]
            view_trials = []
            feat_names = None
            for t in range(min(ntrials, view_data.shape[1] if view_data.ndim > 1 else view_data.shape[0])):
                trial_data = view_data[0, t] if view_data.ndim > 1 else view_data[t]
                ts = trial_data['ts'].astype(np.float64)  # (nFrames, 3, nFeats) in v5
                ft = trial_data['frameTimes'].flatten().astype(np.float64)
                if feat_names is None:
                    fn_raw = trial_data['featNames'].flatten()
                    feat_names = [str(x.flat[0]).strip() for x in fn_raw]
                view_trials.append({'ts': ts, 'frameTimes': ft, 'ts_format': 'v5'})
            traj_data['views'].append({'trials': view_trials, 'feat_names': feat_names})
    except Exception:
        pass

    return {
        'bp': bp, 'vidshift': vidshift, 'ntrials': ntrials,
        'all_spike_times': all_spike_times, 'all_spike_trials': all_spike_trials,
        'all_qualities': all_qualities, 'traj_data': traj_data,
        'n_units_raw': len(all_qualities),
    }


def _load_raw_data_hdf5(data_fn, probes):
    """Load raw data from HDF5 (v7.3) format file. Returns common dict format."""
    f = h5py.File(data_fn, 'r')
    obj = f['obj']

    bp = {}
    bp_group = obj['bp']
    bp['Ntrials'] = int(bp_group['Ntrials'][()].flat[0])
    ntrials = bp['Ntrials']

    for field in ['hit', 'miss', 'no', 'early', 'L', 'R', 'autowater']:
        bp[field] = bp_group[field][()].flatten().astype(np.float64)[:ntrials]

    stim_enable = np.zeros(ntrials)
    if 'stim' in bp_group:
        stim_grp = bp_group['stim']
        if 'enable' in stim_grp:
            stim_enable = stim_grp['enable'][()].flatten().astype(np.float64)[:ntrials]
    bp['stim_enable'] = stim_enable

    ev = {}
    ev_group = bp_group['ev']
    for field in ['goCue', 'sample', 'delay']:
        ev[field] = ev_group[field][()].flatten().astype(np.float64)[:ntrials]
    bp['ev'] = ev

    vidshift = 0.5
    try:
        bit_start = np.nanmedian(bp_group['ev']['bitStart'][()].flatten())
        sglx = obj['sglx']
        fs = sglx['fs'][()].flat[0]
        bitcode_bitstart = np.nanmedian(sglx['bitcode']['bitstart'][()].flatten())
        vid_file_offset = bitcode_bitstart / fs
        vidshift = vid_file_offset - bit_start
    except Exception:
        pass

    clu_refs = obj['clu'][()].flatten()
    all_spike_times = []
    all_spike_trials = []
    all_qualities = []

    for probe_num in probes:
        probe_idx = probe_num - 1
        probe_ref = clu_refs[probe_idx]
        probe_group = f[probe_ref]
        quality_refs = probe_group['quality'][()].flatten()
        trialtm_refs = probe_group['trialtm'][()].flatten()
        trial_refs = probe_group['trial'][()].flatten()

        for i in range(len(quality_refs)):
            quality = h5_read_string(f, quality_refs[i]).strip().lower()
            trialtm = f[trialtm_refs[i]][()].flatten().astype(np.float64)
            trial = f[trial_refs[i]][()].flatten().astype(np.float64)
            all_qualities.append(quality)
            all_spike_times.append(trialtm)
            all_spike_trials.append(trial)

    # --- traj ---
    traj_data = None
    try:
        traj_refs = obj['traj'][()].flatten()
        traj_data = {'views': []}
        for view_idx in range(len(traj_refs)):
            traj_group = f[traj_refs[view_idx]]
            ts_refs = traj_group['ts'][()].flatten()
            ft_refs = traj_group['frameTimes'][()].flatten()
            feat_names_refs = traj_group['featNames'][()].flatten()

            # Read feat names from first trial
            feat_refs_trial = f[feat_names_refs[0]][()].flatten()
            feat_names = [h5_read_string(f, fr) for fr in feat_refs_trial]

            view_trials = []
            for t in range(min(ntrials, len(ts_refs))):
                try:
                    ts = f[ts_refs[t]][()].astype(np.float64)  # (nFeats, 3, nFrames) in HDF5
                    ft = f[ft_refs[t]][()].flatten().astype(np.float64)
                    view_trials.append({'ts': ts, 'frameTimes': ft, 'ts_format': 'hdf5'})
                except Exception:
                    view_trials.append(None)
            traj_data['views'].append({'trials': view_trials, 'feat_names': feat_names})
    except Exception:
        pass

    f.close()

    return {
        'bp': bp, 'vidshift': vidshift, 'ntrials': ntrials,
        'all_spike_times': all_spike_times, 'all_spike_trials': all_spike_trials,
        'all_qualities': all_qualities, 'traj_data': traj_data,
        'n_units_raw': len(all_qualities),
    }


def load_session(anm, date, probes, data_dir_key, params):
    """
    Load and process one session, following the reference pipeline:
    loadObjs -> findClusters -> alignSpikes -> getSeq -> removeLowFRClusters
    Also loads motion energy and computes kinematic features.
    """
    t0 = time.time()
    data_dir = DATA_DIRS[data_dir_key]
    data_fn = os.path.join(data_dir, f'data_structure_{anm}_{date}.mat')
    me_fn = os.path.join(data_dir, f'motionEnergy_{anm}_{date}.mat')

    session_id = f'{anm}_{date}'

    if not os.path.exists(data_fn):
        print(f"  WARNING: {data_fn} not found, skipping")
        return None

    # Detect format and load raw data
    fmt = _detect_file_format(data_fn)
    try:
        if fmt == 'hdf5':
            raw = _load_raw_data_hdf5(data_fn, probes)
        else:
            raw = _load_raw_data_v5(data_fn, probes)
    except Exception as e:
        print(f"  WARNING: Could not load {session_id}: {e}")
        return None

    bp = raw['bp']
    ntrials = raw['ntrials']
    vidshift = raw['vidshift']
    all_spike_times = raw['all_spike_times']
    all_spike_trials = raw['all_spike_trials']
    all_qualities = raw['all_qualities']
    traj_data = raw['traj_data']
    n_units_before_quality = raw['n_units_raw']
    ev = bp['ev']

    # --- Filter clusters by quality ---
    quality_exclude = params['quality_exclude']
    keep_mask = np.array([q not in quality_exclude and q != '' and q != 'nan'
                          for q in all_qualities])
    good_indices = np.where(keep_mask)[0]

    if len(good_indices) == 0:
        print(f"  WARNING: {session_id} has no units after quality filter, skipping")
        return None

    # --- Trial filtering ---
    trial_mask = (bp['early'] == 0) & (bp['stim_enable'] == 0)
    valid_trials = np.where(trial_mask)[0] + 1  # 1-indexed
    if len(valid_trials) < 2:
        print(f"  WARNING: {session_id} has < 2 valid trials, skipping")
        return None

    # --- Align spikes to goCue and bin ---
    align_times = ev[params['align_event']]
    edges = np.arange(params['tmin'], params['tmax'] + params['dt'], params['dt'])
    time_axis = edges[:-1] + params['dt'] / 2
    n_timebins = len(time_axis)
    n_valid_trials = len(valid_trials)

    trialdat = np.zeros((len(good_indices), n_timebins, n_valid_trials), dtype=np.float32)

    for neuron_idx, clu_idx in enumerate(good_indices):
        spike_tm = all_spike_times[clu_idx]
        spike_trial = all_spike_trials[clu_idx]

        for t_idx, trial_num in enumerate(valid_trials):
            spk_mask = spike_trial == trial_num
            if not np.any(spk_mask):
                continue
            spk_times = spike_tm[spk_mask] - align_times[trial_num - 1]
            counts, _ = np.histogram(spk_times, bins=edges)
            fr = causal_gaussian_smooth(counts.astype(np.float64) / params['dt'],
                                       params['smooth_window'],
                                       params['smooth_bctype'])
            trialdat[neuron_idx, :, t_idx] = fr.astype(np.float32)

    # --- Remove low FR clusters ---
    mean_fr = np.mean(trialdat, axis=(1, 2))
    fr_mask = mean_fr > params['low_fr']
    trialdat = trialdat[fr_mask, :, :]
    n_neurons = trialdat.shape[0]

    if n_neurons < params['min_units']:
        print(f"  WARNING: {session_id} has {n_neurons} units (< {params['min_units']}), skipping")
        return None

    # --- Extract trial-level variables ---
    valid_trial_indices = valid_trials - 1
    lick_direction = bp['R'][valid_trial_indices].copy()
    behavioral_context = 1.0 - bp['autowater'][valid_trial_indices]
    outcome = bp['hit'][valid_trial_indices].copy()

    # --- Load Motion Energy ---
    me_data = None
    if os.path.exists(me_fn):
        try:
            me_data = _load_motion_energy_generic(me_fn, traj_data, vidshift,
                                                  align_times, time_axis, ntrials)
        except Exception as e:
            print(f"  WARNING: Could not load motion energy for {session_id}: {e}")

    # --- Compute tongue and paw velocity from DLC ---
    tongue_vel = _compute_feature_velocity_generic(
        traj_data, ntrials, view=2, feat_name='top_tongue',
        vidshift=vidshift, align_times=align_times,
        time_axis=time_axis, is_tongue=True)

    paw_vel = _compute_feature_velocity_generic(
        traj_data, ntrials, view=2, feat_name='top_paw',
        vidshift=vidshift, align_times=align_times,
        time_axis=time_axis, is_tongue=False)

    # --- Subsample to valid trials ---
    valid_0idx = valid_trial_indices  # 0-indexed

    if tongue_vel is not None:
        tongue_vel = tongue_vel[:, valid_0idx]
    if paw_vel is not None:
        paw_vel = paw_vel[:, valid_0idx]
    if me_data is not None:
        me_data = me_data[:, valid_0idx]

    # --- Discretize continuous outputs (50th percentile per session) ---
    tongue_vel_disc = _discretize_velocity(tongue_vel, n_timebins, n_valid_trials)
    paw_vel_disc = _discretize_velocity(paw_vel, n_timebins, n_valid_trials)
    me_disc = _discretize_velocity(me_data, n_timebins, n_valid_trials)

    elapsed = time.time() - t0
    print(f"  {session_id}: {n_neurons} neurons, {n_valid_trials} trials, "
          f"{n_units_before_quality} raw units -> {int(keep_mask.sum())} after quality -> "
          f"{n_neurons} after FR filter ({elapsed:.1f}s)")

    return {
        'session_id': session_id,
        'animal': anm,
        'date': date,
        'n_neurons': n_neurons,
        'n_trials': n_valid_trials,
        'time_axis': time_axis,
        'trialdat': trialdat,  # (n_neurons, n_timebins, n_trials)
        'lick_direction': lick_direction,
        'behavioral_context': behavioral_context,
        'outcome': outcome,
        'tongue_vel_disc': tongue_vel_disc,  # (n_timebins, n_trials)
        'paw_vel_disc': paw_vel_disc,
        'me_disc': me_disc,
        'tongue_vel_raw': tongue_vel,
        'paw_vel_raw': paw_vel,
        'me_raw': me_data,
    }


def _load_motion_energy_generic(me_fn, traj_data, vidshift, align_times, time_axis, ntrials):
    """Load motion energy and align to neural time axis following loadMotionEnergy.m"""
    import scipy.io as sio

    me_cell = None
    try:
        me_file = sio.loadmat(me_fn, squeeze_me=True)
        me_struct = me_file['me']
        me_cell = me_struct['data'].item()
    except Exception:
        me_h5 = h5py.File(me_fn, 'r')
        me_grp = me_h5['me']
        me_data_refs = me_grp['data'][()].flatten()
        me_cell = [me_h5[ref][()].flatten().astype(np.float64) for ref in me_data_refs]
        me_h5.close()

    side_cam = traj_data['views'][0] if traj_data is not None else None
    n_time = len(time_axis)
    me_aligned = np.full((n_time, ntrials), np.nan, dtype=np.float32)

    for trix in range(ntrials):
        try:
            me_trial = me_cell[trix].flatten().astype(np.float64) if hasattr(me_cell[trix], 'flatten') else np.array([float(me_cell[trix])])
        except (IndexError, TypeError):
            continue
        if me_trial.size == 0:
            continue

        ft = None
        if side_cam is not None and trix < len(side_cam['trials']) and side_cam['trials'][trix] is not None:
            ft = side_cam['trials'][trix]['frameTimes']
        if ft is None or ft.size == 0 or np.all(np.isnan(ft)):
            ft = np.arange(1, me_trial.size + 1) / 400.0

        ft_aligned = ft - vidshift - align_times[trix]
        min_len = min(len(ft_aligned), len(me_trial))
        ft_aligned, me_trial = ft_aligned[:min_len], me_trial[:min_len]
        valid = ~np.isnan(ft_aligned) & ~np.isnan(me_trial)
        if valid.sum() < 2:
            continue
        try:
            me_aligned[:, trix] = interp1d(ft_aligned[valid], me_trial[valid],
                                           kind='linear', bounds_error=False,
                                           fill_value=np.nan)(time_axis).astype(np.float32)
        except Exception:
            continue

    for trix in range(ntrials):
        col = me_aligned[:, trix]
        if not np.isnan(col).all() and np.isnan(col).any():
            me_aligned[:, trix] = _fill_nearest(col)

    return me_aligned


def _compute_feature_velocity_generic(traj_data, ntrials, view, feat_name, vidshift, align_times, time_axis, is_tongue=False):
    """Compute velocity of a DLC feature from traj_data (format-agnostic)."""
    n_time = len(time_axis)
    speed = np.full((n_time, ntrials), np.nan, dtype=np.float32)

    if traj_data is None:
        return None

    view_idx = view - 1
    if view_idx >= len(traj_data['views']):
        return None

    view_data = traj_data['views'][view_idx]
    feat_names = view_data['feat_names']

    feat_idx = None
    for idx, fn in enumerate(feat_names):
        if fn == feat_name:
            feat_idx = idx
            break
    if feat_idx is None:
        for idx, fn in enumerate(feat_names):
            if feat_name in fn:
                feat_idx = idx
                break
    if feat_idx is None:
        return None

    for trix in range(min(ntrials, len(view_data['trials']))):
        trial_info = view_data['trials'][trix]
        if trial_info is None:
            continue
        try:
            ts = trial_info['ts']
            ft = trial_info['frameTimes']
            ts_fmt = trial_info['ts_format']

            if ts.ndim != 3:
                continue

            if ts_fmt == 'hdf5':
                x_raw = ts[feat_idx, 0, :]
                y_raw = ts[feat_idx, 1, :]
            else:
                x_raw = ts[:, 0, feat_idx]
                y_raw = ts[:, 1, feat_idx]

            if ft.size == 0 or np.all(np.isnan(ft)):
                ft = np.arange(1, len(x_raw) + 1) / 400.0

            ft_aligned = ft - vidshift - align_times[trix]
            min_len = min(len(ft_aligned), len(x_raw))
            ft_a, x_r, y_r = ft_aligned[:min_len], x_raw[:min_len], y_raw[:min_len]

            valid = ~np.isnan(ft_a) & ~np.isnan(x_r) & ~np.isnan(y_r)
            if valid.sum() < 2:
                continue

            xpos = interp1d(ft_a[valid], x_r[valid], kind='linear',
                            bounds_error=False, fill_value=np.nan)(time_axis)
            ypos = interp1d(ft_a[valid], y_r[valid], kind='linear',
                            bounds_error=False, fill_value=np.nan)(time_axis)

            if not is_tongue:
                xpos = _fill_nearest(xpos)
                ypos = _fill_nearest(ypos)

            xvel = np.gradient(xpos)
            yvel = np.gradient(ypos)

            if is_tongue:
                xvel[np.isnan(xvel)] = 0
                yvel[np.isnan(yvel)] = 0
            else:
                base_xvel = np.nanmedian(np.diff(xpos))
                base_yvel = np.nanmedian(np.diff(ypos))
                xvel -= (base_xvel if not np.isnan(base_xvel) else 0)
                yvel -= (base_yvel if not np.isnan(base_yvel) else 0)
                xvel = _fill_nearest(xvel)
                yvel = _fill_nearest(yvel)

            speed[:, trix] = np.sqrt(xvel**2 + yvel**2).astype(np.float32)
        except Exception:
            continue

    return speed


def _fill_nearest(arr):
    """Fill NaN values with nearest non-NaN value."""
    nans = np.isnan(arr)
    if not nans.any():
        return arr
    if nans.all():
        return arr
    valid_idx = np.where(~nans)[0]
    nan_idx = np.where(nans)[0]
    nearest = np.searchsorted(valid_idx, nan_idx).clip(0, len(valid_idx) - 1)
    arr[nans] = arr[valid_idx[nearest]]
    return arr


def _discretize_velocity(data, n_timebins, n_trials):
    """Discretize continuous data into 2 bins using 50th percentile per session."""
    if data is None:
        return np.zeros((n_timebins, n_trials), dtype=np.int64)

    # Compute 50th percentile across all valid (non-NaN) values
    valid_vals = data[~np.isnan(data)]
    if valid_vals.size == 0:
        return np.zeros((n_timebins, n_trials), dtype=np.int64)

    threshold = np.percentile(valid_vals, 50)
    disc = (data >= threshold).astype(np.int64)
    # Replace NaN entries with 0
    disc[np.isnan(data)] = 0
    return disc


# =============================================================================
# Plotting functions
# =============================================================================

def plot_processing(session_data, save_path):
    """Plot processing visualizations for one session."""
    fig, axes = plt.subplots(4, 3, figsize=(20, 16))
    session_id = session_data['session_id']
    fig.suptitle(f'Processing: {session_id}', fontsize=14)

    time_axis = session_data['time_axis']
    trialdat = session_data['trialdat']
    n_neurons, n_time, n_trials = trialdat.shape

    # Pick a sample trial
    trial_idx = min(5, n_trials - 1)

    # 1. Neural activity heatmap for sample trial
    ax = axes[0, 0]
    vmax = np.percentile(trialdat[:, :, trial_idx], 95)
    ax.imshow(trialdat[:, :, trial_idx], aspect='auto', cmap='viridis',
              extent=[time_axis[0], time_axis[-1], n_neurons, 0], vmin=0, vmax=max(vmax, 1))
    ax.axvline(0, color='r', linestyle='--', alpha=0.7)
    ax.set_title(f'Neural activity (trial {trial_idx})')
    ax.set_xlabel('Time from goCue (s)')
    ax.set_ylabel('Neuron')

    # 2. Mean firing rate across trials for a few neurons
    ax = axes[0, 1]
    for i in range(min(5, n_neurons)):
        ax.plot(time_axis, np.mean(trialdat[i, :, :], axis=1), alpha=0.7)
    ax.axvline(0, color='r', linestyle='--', alpha=0.7)
    ax.set_title('Mean FR (sample neurons)')
    ax.set_xlabel('Time from goCue (s)')
    ax.set_ylabel('FR (Hz)')

    # 3. FR distribution
    ax = axes[0, 2]
    mean_frs = np.mean(trialdat, axis=(1, 2))
    ax.hist(mean_frs, bins=30)
    ax.axvline(PARAMS['low_fr'], color='r', linestyle='--')
    ax.set_title(f'Mean FR distribution (n={n_neurons})')
    ax.set_xlabel('Mean FR (Hz)')

    # 4. Output distributions
    ax = axes[1, 0]
    labels = ['Lick Dir', 'Context', 'Outcome']
    vals = [session_data['lick_direction'], session_data['behavioral_context'], session_data['outcome']]
    for i, (label, v) in enumerate(zip(labels, vals)):
        frac1 = np.mean(v)
        ax.bar(i, frac1, label=f'{label} (frac=1: {frac1:.2f})')
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels, rotation=45)
    ax.set_title('Per-trial output fractions')
    ax.set_ylabel('Fraction = 1')

    # 5. Tongue velocity
    ax = axes[1, 1]
    if session_data['tongue_vel_raw'] is not None:
        tv = session_data['tongue_vel_raw']
        ax.plot(time_axis, np.nanmean(tv, axis=1), label='mean')
        ax.fill_between(time_axis,
                       np.nanpercentile(tv, 25, axis=1),
                       np.nanpercentile(tv, 75, axis=1), alpha=0.3)
    ax.axvline(0, color='r', linestyle='--', alpha=0.7)
    ax.set_title('Tongue velocity')
    ax.set_xlabel('Time from goCue (s)')

    # 6. Paw velocity
    ax = axes[1, 2]
    if session_data['paw_vel_raw'] is not None:
        pv = session_data['paw_vel_raw']
        ax.plot(time_axis, np.nanmean(pv, axis=1), label='mean')
        ax.fill_between(time_axis,
                       np.nanpercentile(pv, 25, axis=1),
                       np.nanpercentile(pv, 75, axis=1), alpha=0.3)
    ax.axvline(0, color='r', linestyle='--', alpha=0.7)
    ax.set_title('Paw velocity')
    ax.set_xlabel('Time from goCue (s)')

    # 7. Motion energy
    ax = axes[2, 0]
    if session_data['me_raw'] is not None:
        me = session_data['me_raw']
        ax.plot(time_axis, np.nanmean(me, axis=1), label='mean')
        ax.fill_between(time_axis,
                       np.nanpercentile(me, 25, axis=1),
                       np.nanpercentile(me, 75, axis=1), alpha=0.3)
    ax.axvline(0, color='r', linestyle='--', alpha=0.7)
    ax.set_title('Motion energy')
    ax.set_xlabel('Time from goCue (s)')

    # 8. Discretized tongue velocity for sample trial
    ax = axes[2, 1]
    ax.plot(time_axis, session_data['tongue_vel_disc'][:, trial_idx], 'b-', alpha=0.7)
    ax.set_title(f'Tongue vel discretized (trial {trial_idx})')
    ax.set_xlabel('Time from goCue (s)')
    ax.set_ylim(-0.1, 1.1)

    # 9. Discretized motion energy for sample trial
    ax = axes[2, 2]
    ax.plot(time_axis, session_data['me_disc'][:, trial_idx], 'g-', alpha=0.7)
    ax.set_title(f'Motion energy discretized (trial {trial_idx})')
    ax.set_xlabel('Time from goCue (s)')
    ax.set_ylim(-0.1, 1.1)

    # 10. Neural vs tongue velocity alignment check
    ax = axes[3, 0]
    pop_rate = np.mean(trialdat[:, :, trial_idx], axis=0)
    ax.plot(time_axis, pop_rate / np.max(pop_rate + 1e-8), 'k-', alpha=0.7, label='Neural (norm)')
    if session_data['tongue_vel_raw'] is not None:
        tv_trial = session_data['tongue_vel_raw'][:, trial_idx]
        ax.plot(time_axis, tv_trial / (np.nanmax(tv_trial) + 1e-8), 'b-', alpha=0.7, label='Tongue vel (norm)')
    ax.axvline(0, color='r', linestyle='--', alpha=0.7)
    ax.set_title(f'Alignment check (trial {trial_idx})')
    ax.legend(fontsize=8)

    # 11-12. Empty
    axes[3, 1].set_visible(False)
    axes[3, 2].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close(fig)
    print(f"  Saved processing plot: {save_path}")


# =============================================================================
# Main conversion
# =============================================================================

def convert_all(session_defs, outfile, show_processing=False):
    """Convert all sessions to the target format."""
    total_t0 = time.time()

    all_sessions = []
    for i, (anm, date, probes, ddir) in enumerate(session_defs):
        print(f"[{i+1}/{len(session_defs)}] Loading {anm}_{date}...")
        result = load_session(anm, date, probes, ddir, PARAMS)
        if result is not None:
            all_sessions.append(result)
            if show_processing and len(all_sessions) <= 2:
                plot_processing(result, f"processing_{result['session_id']}.png")

    if len(all_sessions) == 0:
        print("ERROR: No sessions loaded!")
        sys.exit(1)

    print(f"\nLoaded {len(all_sessions)} sessions in {time.time() - total_t0:.1f}s")

    # --- Build target format ---
    # Subjects
    all_animals = sorted(set(s['animal'] for s in all_sessions))
    animal_to_idx = {a: i for i, a in enumerate(all_animals)}

    # Time axis (should be same for all)
    time_axis = all_sessions[0]['time_axis']
    n_timebins = len(time_axis)

    # Build data lists
    neural = []
    inputs = []
    outputs = []
    subject_idx = []
    brain_region_idx = []

    for sess in all_sessions:
        n_neurons = sess['n_neurons']
        n_trials = sess['n_trials']
        trialdat = sess['trialdat']  # (n_neurons, n_timebins, n_trials)

        # Neural: list of trials, each (n_neurons, n_timebins)
        sess_neural = []
        sess_input = []
        sess_output = []

        for t in range(n_trials):
            # Neural
            sess_neural.append(trialdat[:, :, t].astype(np.float32))

            # Input: time from goCue (continuous, time-varying)
            # Shape: (1, n_timebins)
            time_input = time_axis.astype(np.float32).reshape(1, -1)
            sess_input.append(time_input)

            # Output: per-trial variables broadcast + time-varying variables
            # Per-trial: lick_direction, behavioral_context, outcome
            # Time-varying: tongue_vel_disc, paw_vel_disc, me_disc
            lick_dir = np.full((1, n_timebins), int(sess['lick_direction'][t]), dtype=np.int64)
            context = np.full((1, n_timebins), int(sess['behavioral_context'][t]), dtype=np.int64)
            outc = np.full((1, n_timebins), int(sess['outcome'][t]), dtype=np.int64)
            tongue_v = sess['tongue_vel_disc'][:, t].astype(np.int64).reshape(1, -1)
            paw_v = sess['paw_vel_disc'][:, t].astype(np.int64).reshape(1, -1)
            me_v = sess['me_disc'][:, t].astype(np.int64).reshape(1, -1)

            trial_output = np.concatenate([lick_dir, context, outc, tongue_v, paw_v, me_v], axis=0).astype(np.int64)
            sess_output.append(trial_output)

        neural.append(sess_neural)
        inputs.append(sess_input)
        outputs.append(sess_output)
        subject_idx.append(animal_to_idx[sess['animal']])

        # Brain region: all ALM
        brain_region_idx.append(np.zeros(n_neurons, dtype=np.int64))

    data = {
        'neural': neural,
        'input': inputs,
        'output': outputs,
        'subjects': all_animals,
        'subject_idx': np.array(subject_idx, dtype=np.int64),
        'brain_regions': ['ALM'],
        'brain_region_idx': brain_region_idx,
        'input_names': ['time_from_go_cue'],
        'output_names': ['lick_direction', 'behavioral_context', 'outcome',
                         'tongue_velocity', 'paw_velocity', 'motion_energy'],
        'output_values': [
            ['left', 'right'],        # lick_direction
            ['WC', 'DR'],             # behavioral_context
            ['incorrect', 'correct'], # outcome
            ['low', 'high'],          # tongue_velocity
            ['low', 'high'],          # paw_velocity
            ['low', 'high'],          # motion_energy
        ],
        'metadata': {
            'task_description': 'Two-context delayed-response and water-cued licking task with ALM recordings',
            'time_bin_size': PARAMS['dt'] * 1000,  # in ms
            'temporal_alignment_event': 'go cue onset',
            'off_start': PARAMS['tmin'],
            'off_end': PARAMS['tmax'],
            'smooth_window': PARAMS['smooth_window'],
            'smooth_type': 'causal_gaussian',
            'low_fr_threshold': PARAMS['low_fr'],
            'min_units_per_session': PARAMS['min_units'],
            'quality_excluded': list(PARAMS['quality_exclude']),
            'source_paper': 'Hasnain, Birnbaum et al., Nature Neuroscience 2024',
            'n_sessions': len(all_sessions),
            'n_subjects': len(all_animals),
        }
    }

    # Save
    print(f"\nSaving to {outfile}...")
    with open(outfile, 'wb') as pkl:
        pickle.dump(data, pkl, protocol=4)
    file_size = os.path.getsize(outfile) / (1024 * 1024)
    print(f"Saved {outfile} ({file_size:.1f} MB)")
    print(f"Total time: {time.time() - total_t0:.1f}s")

    # Print summary
    total_neurons = sum(len(br) for br in brain_region_idx)
    total_trials = sum(len(s) for s in neural)
    print(f"\nSummary:")
    print(f"  Sessions: {len(all_sessions)}")
    print(f"  Subjects: {len(all_animals)}")
    print(f"  Total neurons: {total_neurons}")
    print(f"  Total trials: {total_trials}")
    print(f"  Mean neurons/session: {total_neurons / len(all_sessions):.1f}")
    print(f"  Mean trials/session: {total_trials / len(all_sessions):.1f}")


def main():
    parser = argparse.ArgumentParser(description='Convert Hasnain et al. data to decoder format')
    parser.add_argument('outfile', type=str, help='Output pickle file path')
    parser.add_argument('--full', action='store_true', default=True, help='Process all sessions (default)')
    parser.add_argument('--sample', action='store_true', help='Process only 2 sessions for testing')
    parser.add_argument('--show-processing', action='store_true', help='Plot processing visualizations')
    args = parser.parse_args()

    if args.sample:
        # Pick 2 sessions: one from each data directory
        session_defs = [SESSION_DEFS[0], SESSION_DEFS[25]]  # EKH1 and JEB11
        print("Running in SAMPLE mode (2 sessions)")
    else:
        session_defs = SESSION_DEFS
        print(f"Running in FULL mode ({len(SESSION_DEFS)} sessions)")

    convert_all(session_defs, args.outfile, show_processing=args.show_processing)


if __name__ == '__main__':
    main()
