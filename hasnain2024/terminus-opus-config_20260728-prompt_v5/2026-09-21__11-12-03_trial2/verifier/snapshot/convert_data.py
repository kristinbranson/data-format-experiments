#!/usr/bin/env python3
"""Convert Economo Lab data to decoder-compatible format.

Usage:
    python -u convert_data.py <outpicklefile> [--full|--sample] [--show-processing]
"""

import sys
import os
import time
import argparse
import pickle
import numpy as np
import h5py
import scipy.io as sio
from scipy.signal import convolve
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

ALIGN_EVENT = 'goCue'
TMIN = -2.5  # seconds
TMAX = 2.5   # seconds
DT = 1.0 / 100  # 10ms bins
SMOOTH_WIN = 15  # samples for causal Gaussian kernel
LOW_FR = 0.5  # Hz, minimum mean firing rate
BOUNDARY_CONDITION = 'reflect'
VIDEO_OFFSET_DEFAULT = 0.5  # seconds

# Quality labels to exclude (case-sensitive, matching MATLAB ismember)
EXCLUDED_QUALITIES = {'garbage', 'gabrga', 'noisy', 'real?'}

# Session metadata: (animal, date, dataset_dir, alm_probes)
SESSION_META = [
    # Ephys_Behavior sessions
    ('EKH1', '2021-08-07', 'Ephys_Behavior', [2]),
    ('EKH3', '2021-08-11', 'Ephys_Behavior', [2]),
    ('JEB6', '2021-04-18', 'Ephys_Behavior', [2]),
    ('JEB7', '2021-04-29', 'Ephys_Behavior', [1]),
    ('JEB7', '2021-04-30', 'Ephys_Behavior', [1]),
    ('JEB13', '2022-09-13', 'Ephys_Behavior', [2]),
    ('JEB13', '2022-09-14', 'Ephys_Behavior', [2]),
    ('JEB13', '2022-09-21', 'Ephys_Behavior', [1]),
    ('JEB13', '2022-09-24', 'Ephys_Behavior', [1]),
    ('JEB13', '2022-09-25', 'Ephys_Behavior', [1]),
    ('JEB14', '2022-08-22', 'Ephys_Behavior', [1]),
    ('JEB14', '2022-08-23', 'Ephys_Behavior', [1]),
    ('JEB14', '2022-08-24', 'Ephys_Behavior', [1]),
    ('JEB14', '2022-08-25', 'Ephys_Behavior', [1]),
    ('JEB15', '2022-07-26', 'Ephys_Behavior', [1, 2]),
    ('JEB15', '2022-07-27', 'Ephys_Behavior', [1, 2]),
    ('JEB15', '2022-07-28', 'Ephys_Behavior', [1, 2]),
    ('JEB15', '2022-07-29', 'Ephys_Behavior', [2]),
    ('JEB19', '2023-04-18', 'Ephys_Behavior', [1]),
    ('JEB19', '2023-04-19', 'Ephys_Behavior', [1]),
    ('JEB19', '2023-04-20', 'Ephys_Behavior', [1]),
    ('JEB19', '2023-04-21', 'Ephys_Behavior', [1]),
    ('JGR2', '2021-11-16', 'Ephys_Behavior', [1]),
    ('JGR2', '2021-11-17', 'Ephys_Behavior', [1]),
    ('JGR3', '2021-11-18', 'Ephys_Behavior', [1]),
    # RandomizedDelay sessions
    ('JEB11', '2022-05-10', 'RandomizedDelay_Ephys_Behavior', [1]),
    ('JEB11', '2022-05-11', 'RandomizedDelay_Ephys_Behavior', [1]),
    ('JEB12', '2022-05-12', 'RandomizedDelay_Ephys_Behavior', [1]),
    ('JEB12', '2022-05-13', 'RandomizedDelay_Ephys_Behavior', [1]),
    ('JEB23', '2023-10-10', 'RandomizedDelay_Ephys_Behavior', [1]),
    ('JEB23', '2023-10-11', 'RandomizedDelay_Ephys_Behavior', [1]),
    ('JEB23', '2023-10-12', 'RandomizedDelay_Ephys_Behavior', [1]),
    ('JEB23', '2023-10-13', 'RandomizedDelay_Ephys_Behavior', [1]),
    ('JEB23', '2023-10-18', 'RandomizedDelay_Ephys_Behavior', [1]),
    ('JEB23', '2023-10-19', 'RandomizedDelay_Ephys_Behavior', [1]),
    ('JEB23', '2023-10-21', 'RandomizedDelay_Ephys_Behavior', [1]),
    ('JEB24', '2023-10-23', 'RandomizedDelay_Ephys_Behavior', [1]),
    ('JEB24', '2023-10-24', 'RandomizedDelay_Ephys_Behavior', [1]),
    ('JEB24', '2023-10-25', 'RandomizedDelay_Ephys_Behavior', [1]),
    ('JEB24', '2023-10-26', 'RandomizedDelay_Ephys_Behavior', [1]),
    ('JEB24', '2023-10-27', 'RandomizedDelay_Ephys_Behavior', [1]),
    ('JEB24', '2023-10-31', 'RandomizedDelay_Ephys_Behavior', [1]),
    ('JEB24', '2023-11-02', 'RandomizedDelay_Ephys_Behavior', [1]),
    ('JEB24', '2023-11-03', 'RandomizedDelay_Ephys_Behavior', [1]),
]

DATA_DIR = '/app/data'


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def causal_gaussian_kernel(n):
    """Create causal Gaussian kernel matching MATLAB mySmooth."""
    # gausswin equivalent
    alpha = 2.5  # MATLAB default
    n_pts = n
    half = (n_pts - 1) / 2
    t = np.arange(n_pts) - half
    kern = np.exp(-0.5 * (alpha * t / half) ** 2)
    # Make causal: zero out first half
    kern[:n_pts // 2] = 0
    kern = kern / kern.sum()
    return kern


def smooth_signal(x, kernel):
    """Smooth signal with kernel using 'same' convolution."""
    if x.ndim == 1:
        return np.convolve(x, kernel, mode='same')
    else:
        out = np.zeros_like(x)
        for j in range(x.shape[1]):
            out[:, j] = np.convolve(x[:, j], kernel, mode='same')
        return out


def reflect_boundary(x, n):
    """Apply reflect boundary condition."""
    prefix = x[1:n+1][::-1]
    return np.concatenate([prefix, x], axis=0)


def smooth_with_bc(x, kernel, bctype='reflect'):
    """Smooth with boundary condition handling matching MATLAB mySmooth."""
    n = len(kernel)
    if bctype == 'reflect':
        if x.ndim == 1:
            prefix = x[1:n+1][::-1]
            x_ext = np.concatenate([prefix, x])
            smoothed = np.convolve(x_ext, kernel, mode='same')
            return smoothed[n:]  # trim prefix
        else:
            prefix = x[1:n+1][::-1]
            x_ext = np.concatenate([prefix, x], axis=0)
            out = np.zeros_like(x_ext)
            for j in range(x_ext.shape[1]):
                out[:, j] = np.convolve(x_ext[:, j], kernel, mode='same')
            return out[n:]  # trim prefix
    else:
        return smooth_signal(x, kernel)


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_mat_file(fpath):
    """Load .mat file, handling both HDF5 (v7.3) and v5 formats."""
    try:
        f = h5py.File(fpath, 'r')
        return f, 'h5'
    except:
        data = sio.loadmat(fpath, squeeze_me=False)
        return data, 'v5'


def safe_scalar(x):
    """Extract scalar from potentially nested arrays."""
    while hasattr(x, 'flatten') and hasattr(x, '__len__'):
        x = x.flatten()
        if len(x) == 1:
            x = x[0]
        else:
            break
    return x


def read_h5_string(f, ref):
    """Read string from HDF5 reference."""
    data = f[ref][:].flatten()
    return ''.join([chr(c) for c in data]).strip()


def get_bp_field_h5(f, field_path):
    """Get behavioral parameter field from HDF5 file."""
    return f[f'obj/bp/{field_path}'][:].flatten()


def get_bp_field_v5(obj, field_path):
    """Get behavioral parameter field from v5 mat file."""
    bp = obj['bp'][0, 0]
    parts = field_path.split('/')
    current = bp
    for part in parts:
        current = current[part][0, 0]
    return current.flatten().astype(float)


def get_event_times_h5(f, event_name):
    """Get event times from HDF5 file."""
    return f[f'obj/bp/ev/{event_name}'][:].flatten()


def get_event_times_v5(obj, event_name):
    """Get event times from v5 mat file."""
    bp = obj['bp'][0, 0]
    ev = bp['ev'][0, 0]
    return ev[event_name][0, 0].flatten().astype(float)


# ============================================================================
# SESSION PROCESSING
# ============================================================================

def process_session(anm, date, dataset_dir, alm_probes, time_edges, kernel, show_processing=False):
    """Process a single session and return data in target format."""
    sess_id = f"{anm}_{date}"
    data_fpath = os.path.join(DATA_DIR, dataset_dir, f'data_structure_{sess_id}.mat')
    me_fpath = os.path.join(DATA_DIR, dataset_dir, f'motionEnergy_{sess_id}.mat')
    
    t0 = time.time()
    
    # Time axis
    time_centers = time_edges[:-1] + DT / 2
    n_time = len(time_centers)
    
    # Load data file
    fdata, fmt = load_mat_file(data_fpath)
    
    if fmt == 'h5':
        result = process_session_h5(fdata, anm, date, sess_id, alm_probes, 
                                     time_edges, time_centers, n_time, kernel,
                                     me_fpath, show_processing)
        fdata.close()
    else:
        result = process_session_v5(fdata, anm, date, sess_id, alm_probes,
                                     time_edges, time_centers, n_time, kernel,
                                     me_fpath, show_processing)
    
    t1 = time.time()
    if result is not None:
        n_neurons = result['neural'][0].shape[0] if len(result['neural']) > 0 else 0
        n_trials = len(result['neural'])
        print(f"  {sess_id}: {n_neurons} neurons, {n_trials} trials ({t1-t0:.1f}s)")
    else:
        print(f"  {sess_id}: SKIPPED ({t1-t0:.1f}s)")
    
    return result


def process_session_h5(f, anm, date, sess_id, alm_probes, time_edges, time_centers, n_time, kernel, me_fpath, show_processing):
    """Process session from HDF5 format file."""
    
    # --- Load behavioral data ---
    ntrials = int(f['obj/bp/Ntrials'][0, 0])
    hit = f['obj/bp/hit'][:].flatten().astype(bool)
    miss = f['obj/bp/miss'][:].flatten().astype(bool)
    no_resp = f['obj/bp/no'][:].flatten().astype(bool)
    early = f['obj/bp/early'][:].flatten().astype(bool)
    autowater = f['obj/bp/autowater'][:].flatten().astype(bool)
    R = f['obj/bp/R'][:].flatten().astype(bool)
    L = f['obj/bp/L'][:].flatten().astype(bool)
    stim_enable = f['obj/bp/stim/enable'][:].flatten().astype(bool)
    
    goCue = f['obj/bp/ev/goCue'][:].flatten()
    
    # --- Trial selection ---
    # Exclude early lick and stim trials
    valid_trials_mask = ~early & ~stim_enable
    valid_trial_indices = np.where(valid_trials_mask)[0]  # 0-indexed
    
    if len(valid_trial_indices) < 2:
        print(f"  {sess_id}: Too few valid trials ({len(valid_trial_indices)})")
        return None
    
    # --- Load and filter clusters ---
    clu_data = f['obj/clu']
    all_spike_times = []  # aligned spike times per cluster
    all_spike_trials = []  # trial numbers per cluster (1-indexed)
    
    for probe_num in alm_probes:
        probe_idx = probe_num - 1  # 0-indexed for HDF5 (nProbes, 1)
        if probe_idx >= clu_data.shape[0]:
            continue
        ref = clu_data[probe_idx, 0]
        probe = f[ref]
        if not isinstance(probe, h5py.Group):
            continue
        
        quality_refs = probe['quality']
        n_clusters = quality_refs.shape[0]
        
        for i in range(n_clusters):
            # Check quality
            qref = quality_refs[i, 0]
            qdata = f[qref][:].flatten()
            label = ''.join([chr(c) for c in qdata]).strip()
            if label in EXCLUDED_QUALITIES:
                continue
            
            # Get spike data
            trialtm_ref = probe['trialtm'][i, 0]
            trial_ref = probe['trial'][i, 0]
            trialtm = f[trialtm_ref][:].flatten()
            trial = f[trial_ref][:].flatten().astype(int)  # 1-indexed
            
            # Align to goCue
            # For each spike, subtract the goCue time of its trial
            aligned_times = np.empty_like(trialtm)
            for t_idx in range(len(trialtm)):
                tr = trial[t_idx] - 1  # 0-indexed
                if 0 <= tr < ntrials:
                    aligned_times[t_idx] = trialtm[t_idx] - goCue[tr]
                else:
                    aligned_times[t_idx] = np.nan
            
            all_spike_times.append(aligned_times)
            all_spike_trials.append(trial)
    
    n_neurons_raw = len(all_spike_times)
    if n_neurons_raw == 0:
        print(f"  {sess_id}: No neurons found")
        return None
    
    # --- Bin spikes and compute firing rates ---
    # First compute single-trial firing rates for all trials (for FR filter)
    # trialdat shape: (n_time, n_neurons, n_all_trials)
    trialdat = np.zeros((n_time, n_neurons_raw, ntrials), dtype=np.float32)
    
    for i in range(n_neurons_raw):
        spk_times = all_spike_times[i]
        spk_trials = all_spike_trials[i]
        
        for tr_idx in range(ntrials):
            trial_num = tr_idx + 1  # 1-indexed
            mask = spk_trials == trial_num
            if not np.any(mask):
                continue
            
            spk_t = spk_times[mask]
            spk_t = spk_t[~np.isnan(spk_t)]
            
            counts, _ = np.histogram(spk_t, bins=time_edges)
            fr = counts.astype(np.float32) / DT
            trialdat[:, i, tr_idx] = smooth_with_bc(fr, kernel, BOUNDARY_CONDITION)
    
    # --- Remove low FR clusters ---
    # Compute mean FR across all trials and time points (matching removeLowFRClusters)
    # The reference code computes mean of mean of psth (trial-averaged by condition)
    # Simplified: mean FR across all time and trials
    mean_fr = trialdat.mean(axis=(0, 2))  # mean over time and trials
    keep_mask = mean_fr > LOW_FR
    
    if keep_mask.sum() < 10:
        print(f"  {sess_id}: Too few neurons after FR filter ({keep_mask.sum()})")
        return None
    
    trialdat = trialdat[:, keep_mask, :]
    n_neurons = trialdat.shape[1]
    
    # --- Extract trial data for valid trials ---
    neural_trials = []
    for tr_idx in valid_trial_indices:
        # shape: (n_neurons, n_time)
        neural_trials.append(trialdat[:, :, tr_idx].T.copy())
    
    # --- Compute behavioral outputs ---
    # Lick direction: 0=left, 1=right, 2=none
    # R/L indicate CORRECT direction. Need to compute ACTUAL lick direction.
    # Hit: animal licked correct side (R→right, L→left)
    # Miss: animal licked wrong side (R→left, L→right) 
    # No (ignore): no lick
    lick_dir = np.full(ntrials, 2, dtype=int)  # default: none (ignore)
    lick_dir[hit & R] = 1   # correct right → licked right
    lick_dir[hit & L] = 0   # correct left → licked left
    lick_dir[miss & R] = 0  # correct right, wrong → licked left
    lick_dir[miss & L] = 1  # correct left, wrong → licked right
    
    # Context: 0=WC, 1=DR
    context = np.zeros(ntrials, dtype=int)
    context[~autowater] = 1  # DR
    context[autowater] = 0   # WC
    
    # Outcome: 0=incorrect, 1=correct, 2=ignore
    outcome = np.full(ntrials, 2, dtype=int)  # default: ignore
    outcome[hit] = 1  # correct
    outcome[miss] = 0  # incorrect
    
    # --- Load DLC data for tongue and paw velocity ---
    tongue_vel_all, paw_vel_all, has_video = load_dlc_velocities_h5(
        f, ntrials, goCue, time_centers, sess_id)
    
    # --- Load motion energy ---
    me_all, has_me = load_motion_energy(me_fpath, f, fmt='h5', 
                                         ntrials=ntrials, goCue=goCue, 
                                         time_centers=time_centers, sess_id=sess_id)
    
    # --- Discretize continuous outputs ---
    tongue_disc = discretize_velocity(tongue_vel_all, valid_trial_indices, not_visible_val=2)
    paw_disc = discretize_velocity(paw_vel_all, valid_trial_indices, not_visible_val=2)
    me_disc = discretize_velocity(me_all, valid_trial_indices, not_visible_val=2)
    
    # --- Build output arrays ---
    input_trials = []
    output_trials = []
    
    for i, tr_idx in enumerate(valid_trial_indices):
        # Input: time from go cue (continuous, time-varying)
        inp = time_centers.astype(np.float32).reshape(1, -1)
        input_trials.append(inp)
        
        # Output: stack all output variables
        # Per-trial outputs: lick_direction, context, outcome
        # Time-varying outputs: tongue_velocity, paw_velocity, motion_energy
        out_per_trial = np.array([lick_dir[tr_idx], context[tr_idx], outcome[tr_idx]], dtype=np.float32)
        out_time_varying = np.stack([
            tongue_disc[i],
            paw_disc[i],
            me_disc[i]
        ], axis=0).astype(np.float32)  # (3, n_time)
        
        # Combine: first 3 are per-trial, last 3 are time-varying
        # Use shape (n_output,) for per-trial and (n_output, n_time) for time-varying
        # We'll make everything (n_output, n_time) by broadcasting per-trial
        out = np.zeros((6, n_time), dtype=np.int64)
        out[0, :] = lick_dir[tr_idx]  # broadcast
        out[1, :] = context[tr_idx]
        out[2, :] = outcome[tr_idx]
        out[3, :] = tongue_disc[i]
        out[4, :] = paw_disc[i]
        out[5, :] = me_disc[i]
        
        output_trials.append(out)
    
    return {
        'neural': neural_trials,
        'input': input_trials,
        'output': output_trials,
        'n_neurons': n_neurons,
        'anm': anm,
        'date': date,
        'sess_id': sess_id,
    }


def process_session_v5(data, anm, date, sess_id, alm_probes, time_edges, time_centers, n_time, kernel, me_fpath, show_processing):
    """Process session from v5 format mat file."""
    obj = data['obj']
    bp = obj['bp'][0, 0]
    
    # --- Load behavioral data ---
    ntrials = int(safe_scalar(bp['Ntrials']))
    hit = bp['hit'][0, 0].flatten().astype(bool)
    miss = bp['miss'][0, 0].flatten().astype(bool)
    no_resp = bp['no'][0, 0].flatten().astype(bool)
    early = bp['early'][0, 0].flatten().astype(bool)
    autowater = bp['autowater'][0, 0].flatten().astype(bool)
    R = bp['R'][0, 0].flatten().astype(bool)
    L = bp['L'][0, 0].flatten().astype(bool)
    stim_enable = bp['stim'][0, 0]['enable'][0, 0].flatten().astype(bool)
    
    ev = bp['ev'][0, 0]
    goCue = ev['goCue'][0, 0].flatten().astype(float)
    
    # --- Trial selection ---
    valid_trials_mask = ~early & ~stim_enable
    valid_trial_indices = np.where(valid_trials_mask)[0]
    
    if len(valid_trial_indices) < 2:
        print(f"  {sess_id}: Too few valid trials ({len(valid_trial_indices)})")
        return None
    
    # --- Load and filter clusters ---
    clu_data = obj['clu'][0, 0]
    all_spike_times = []
    all_spike_trials = []
    
    for probe_num in alm_probes:
        probe_idx = probe_num - 1
        if probe_idx >= clu_data.shape[1]:
            continue
        probe = clu_data[0, probe_idx]
        
        quality = probe['quality']
        n_clusters = quality.shape[1]
        
        for i in range(n_clusters):
            qi = quality[0, i]
            label = str(qi[0]).strip() if qi.size > 0 else ''
            if label in EXCLUDED_QUALITIES:
                continue
            
            trialtm = probe['trialtm'][0, i].flatten().astype(float)
            trial = probe['trial'][0, i].flatten().astype(int)
            
            # Align to goCue
            aligned_times = np.empty_like(trialtm)
            for t_idx in range(len(trialtm)):
                tr = trial[t_idx] - 1
                if 0 <= tr < ntrials:
                    aligned_times[t_idx] = trialtm[t_idx] - goCue[tr]
                else:
                    aligned_times[t_idx] = np.nan
            
            all_spike_times.append(aligned_times)
            all_spike_trials.append(trial)
    
    n_neurons_raw = len(all_spike_times)
    if n_neurons_raw == 0:
        print(f"  {sess_id}: No neurons found")
        return None
    
    # --- Bin spikes and compute firing rates ---
    trialdat = np.zeros((n_time, n_neurons_raw, ntrials), dtype=np.float32)
    
    for i in range(n_neurons_raw):
        spk_times = all_spike_times[i]
        spk_trials = all_spike_trials[i]
        
        for tr_idx in range(ntrials):
            trial_num = tr_idx + 1
            mask = spk_trials == trial_num
            if not np.any(mask):
                continue
            spk_t = spk_times[mask]
            spk_t = spk_t[~np.isnan(spk_t)]
            counts, _ = np.histogram(spk_t, bins=time_edges)
            fr = counts.astype(np.float32) / DT
            trialdat[:, i, tr_idx] = smooth_with_bc(fr, kernel, BOUNDARY_CONDITION)
    
    # --- Remove low FR clusters ---
    mean_fr = trialdat.mean(axis=(0, 2))
    keep_mask = mean_fr > LOW_FR
    
    if keep_mask.sum() < 10:
        print(f"  {sess_id}: Too few neurons after FR filter ({keep_mask.sum()})")
        return None
    
    trialdat = trialdat[:, keep_mask, :]
    n_neurons = trialdat.shape[1]
    
    # --- Extract trial data ---
    neural_trials = []
    for tr_idx in valid_trial_indices:
        neural_trials.append(trialdat[:, :, tr_idx].T.copy())
    
    # --- Behavioral outputs ---
    lick_dir = np.full(ntrials, 2, dtype=int)  # default: none (ignore)
    lick_dir[hit & R] = 1   # correct right → licked right
    lick_dir[hit & L] = 0   # correct left → licked left
    lick_dir[miss & R] = 0  # correct right, wrong → licked left
    lick_dir[miss & L] = 1  # correct left, wrong → licked right
    
    context = np.zeros(ntrials, dtype=int)
    context[~autowater] = 1
    context[autowater] = 0
    
    outcome = np.full(ntrials, 2, dtype=int)
    outcome[hit] = 1
    outcome[miss] = 0
    
    # --- DLC velocities ---
    tongue_vel_all, paw_vel_all, has_video = load_dlc_velocities_v5(
        obj, ntrials, goCue, time_centers, sess_id)
    
    # --- Motion energy ---
    me_all, has_me = load_motion_energy(me_fpath, data, fmt='v5',
                                         ntrials=ntrials, goCue=goCue,
                                         time_centers=time_centers, sess_id=sess_id)
    
    # --- Discretize ---
    tongue_disc = discretize_velocity(tongue_vel_all, valid_trial_indices, not_visible_val=2)
    paw_disc = discretize_velocity(paw_vel_all, valid_trial_indices, not_visible_val=2)
    me_disc = discretize_velocity(me_all, valid_trial_indices, not_visible_val=2)
    
    # --- Build output arrays ---
    input_trials = []
    output_trials = []
    
    for i, tr_idx in enumerate(valid_trial_indices):
        inp = time_centers.astype(np.float32).reshape(1, -1)
        input_trials.append(inp)
        
        out = np.zeros((6, n_time), dtype=np.int64)
        out[0, :] = lick_dir[tr_idx]
        out[1, :] = context[tr_idx]
        out[2, :] = outcome[tr_idx]
        out[3, :] = tongue_disc[i]
        out[4, :] = paw_disc[i]
        out[5, :] = me_disc[i]
        output_trials.append(out)
    
    return {
        'neural': neural_trials,
        'input': input_trials,
        'output': output_trials,
        'n_neurons': n_neurons,
        'anm': anm,
        'date': date,
        'sess_id': sess_id,
    }


# ============================================================================
# DLC VELOCITY FUNCTIONS
# ============================================================================

def compute_velocity_from_xy(x, y, frame_times, dt_frames):
    """Compute velocity magnitude from x,y coordinates."""
    dx = np.diff(x) / dt_frames
    dy = np.diff(y) / dt_frames
    vel = np.sqrt(dx**2 + dy**2)
    # Pad to same length
    vel = np.concatenate([vel, [vel[-1] if len(vel) > 0 else 0]])
    return vel


def load_dlc_velocities_h5(f, ntrials, goCue, time_centers, sess_id):
    """Load DLC tongue and paw velocities from HDF5 file."""
    n_time = len(time_centers)
    tongue_vel = np.full((n_time, ntrials), np.nan, dtype=np.float32)
    paw_vel = np.full((n_time, ntrials), np.nan, dtype=np.float32)
    has_video = True
    
    try:
        traj = f['obj/traj']
        
        # Side cam (index 0) for tongue
        side_ref = traj[0, 0]
        side_cam = f[side_ref]
        
        # Bottom cam (index 1) for paw
        bottom_ref = traj[1, 0]
        bottom_cam = f[bottom_ref]
        
        # Get feature names from first trial
        side_feat_ref = side_cam['featNames'][0, 0]
        side_feat_data = f[side_feat_ref]
        side_feats = []
        for j in range(side_feat_data.shape[1]):
            ref = side_feat_data[0, j]
            name = ''.join([chr(c) for c in f[ref][:].flatten()])
            side_feats.append(name)
        
        bottom_feat_ref = bottom_cam['featNames'][0, 0]
        bottom_feat_data = f[bottom_feat_ref]
        bottom_feats = []
        for j in range(bottom_feat_data.shape[1]):
            ref = bottom_feat_data[0, j]
            name = ''.join([chr(c) for c in f[ref][:].flatten()])
            bottom_feats.append(name)
        
        # Find tongue feature index (side cam)
        tongue_idx = None
        for j, name in enumerate(side_feats):
            if name == 'tongue':
                tongue_idx = j
                break
        
        # Find paw feature indices (bottom cam)
        paw_indices = []
        for j, name in enumerate(bottom_feats):
            if 'paw' in name.lower():
                paw_indices.append(j)
        
        # Compute video offset
        try:
            bitStart = np.nanmedian(f['obj/bp/ev/bitStart'][:].flatten())
            sglx_bitstart = np.nanmedian(f['obj/sglx/bitcode/bitstart'][:].flatten())
            sglx_fs = f['obj/sglx/fs'][0, 0]
            vidshift = sglx_bitstart / sglx_fs - bitStart
        except:
            vidshift = VIDEO_OFFSET_DEFAULT
        
        # Process each trial
        for tr_idx in range(ntrials):
            try:
                # Side cam - tongue
                if tongue_idx is not None:
                    ts_ref = side_cam['ts'][tr_idx, 0]
                    ts = f[ts_ref][:]  # (nFeats, 3, nFrames)
                    ft_ref = side_cam['frameTimes'][tr_idx, 0]
                    frame_times = f[ft_ref][:].flatten()
                    
                    # ts dimensions: (nFeats, 3=[x,y,conf], nFrames)
                    tongue_x = ts[tongue_idx, 0, :]
                    tongue_y = ts[tongue_idx, 1, :]
                    tongue_conf = ts[tongue_idx, 2, :]
                    
                    # Align frame times
                    aligned_ft = frame_times - vidshift - goCue[tr_idx]
                    
                    # Compute velocity
                    dt_frames = np.median(np.diff(aligned_ft))
                    if dt_frames > 0:
                        vel = compute_velocity_from_xy(tongue_x, tongue_y, aligned_ft, dt_frames)
                        # Set low confidence to NaN
                        vel[tongue_conf < 0.9] = np.nan
                        # Interpolate to time_centers
                        valid = ~np.isnan(vel)
                        if valid.sum() > 2:
                            tongue_vel[:, tr_idx] = np.interp(time_centers, aligned_ft[valid], vel[valid],
                                                              left=np.nan, right=np.nan)
                
                # Bottom cam - paw
                if len(paw_indices) > 0:
                    ts_ref = bottom_cam['ts'][tr_idx, 0]
                    ts = f[ts_ref][:]  # (nFeats, 3, nFrames)
                    ft_ref = bottom_cam['frameTimes'][tr_idx, 0]
                    frame_times = f[ft_ref][:].flatten()
                    
                    aligned_ft = frame_times - vidshift - goCue[tr_idx]
                    dt_frames = np.median(np.diff(aligned_ft))
                    
                    if dt_frames > 0:
                        # Average velocity across paw features
                        paw_vels = []
                        for pidx in paw_indices:
                            px = ts[pidx, 0, :]
                            py = ts[pidx, 1, :]
                            pc = ts[pidx, 2, :]
                            vel = compute_velocity_from_xy(px, py, aligned_ft, dt_frames)
                            vel[pc < 0.9] = np.nan
                            paw_vels.append(vel)
                        
                        avg_vel = np.nanmean(paw_vels, axis=0)
                        valid = ~np.isnan(avg_vel)
                        if valid.sum() > 2:
                            paw_vel[:, tr_idx] = np.interp(time_centers, aligned_ft[valid], avg_vel[valid],
                                                           left=np.nan, right=np.nan)
            except Exception as e:
                continue
    
    except Exception as e:
        print(f"  {sess_id}: DLC loading error: {e}")
        has_video = False
    
    return tongue_vel, paw_vel, has_video


def load_dlc_velocities_v5(obj, ntrials, goCue, time_centers, sess_id):
    """Load DLC tongue and paw velocities from v5 format."""
    n_time = len(time_centers)
    tongue_vel = np.full((n_time, ntrials), np.nan, dtype=np.float32)
    paw_vel = np.full((n_time, ntrials), np.nan, dtype=np.float32)
    has_video = True
    
    try:
        traj = obj['traj'][0, 0]
        # v5 format: traj shape is (1, nCams) or (nCams, 1)
        if traj.shape[0] == 1:  # (1, nCams)
            side_cam = traj[0, 0]
            bottom_cam = traj[0, 1] if traj.shape[1] > 1 else None
        else:  # (nCams, 1)
            side_cam = traj[0, 0]
            bottom_cam = traj[1, 0] if traj.shape[0] > 1 else None
        
        # Get feature names
        # v5 featNames can be (nFeats, 1) or (1, nFeats)
        feat_arr = side_cam['featNames'][0, 0]
        if feat_arr.shape[0] > feat_arr.shape[1]:
            side_feats = [str(feat_arr[j, 0][0]) for j in range(feat_arr.shape[0])]
        else:
            side_feats = [str(feat_arr[0, j][0]) for j in range(feat_arr.shape[1])]
        if bottom_cam is not None:
            feat_arr_b = bottom_cam['featNames'][0, 0]
            if feat_arr_b.shape[0] > feat_arr_b.shape[1]:
                bottom_feats = [str(feat_arr_b[j, 0][0]) for j in range(feat_arr_b.shape[0])]
            else:
                bottom_feats = [str(feat_arr_b[0, j][0]) for j in range(feat_arr_b.shape[1])]
        else:
            bottom_feats = []
        
        tongue_idx = side_feats.index('tongue') if 'tongue' in side_feats else None
        paw_indices = [j for j, name in enumerate(bottom_feats) if 'paw' in name.lower()]
        
        # Video offset
        try:
            bitStart = np.nanmedian(obj['bp'][0,0]['ev'][0,0]['bitStart'][0,0].flatten())
            sglx = obj['sglx'][0,0]
            sglx_bitstart = np.nanmedian(sglx['bitcode'][0,0]['bitstart'][0,0].flatten())
            sglx_fs = float(safe_scalar(sglx['fs']))
            vidshift = sglx_bitstart / sglx_fs - bitStart
        except:
            vidshift = VIDEO_OFFSET_DEFAULT
        
        for tr_idx in range(ntrials):
            try:
                if tongue_idx is not None:
                    ts = side_cam['ts'][0, tr_idx]  # (nFrames, 3, nFeats) or (nFeats, 3, nFrames)
                    frame_times = side_cam['frameTimes'][0, tr_idx].flatten()
                    
                    # Handle different dimension orders
                    if ts.ndim == 3:
                        if ts.shape[2] == len(side_feats):  # (nFrames, 3, nFeats)
                            tongue_x = ts[:, 0, tongue_idx]
                            tongue_y = ts[:, 1, tongue_idx]
                            tongue_conf = ts[:, 2, tongue_idx]
                        else:  # (nFeats, 3, nFrames)
                            tongue_x = ts[tongue_idx, 0, :]
                            tongue_y = ts[tongue_idx, 1, :]
                            tongue_conf = ts[tongue_idx, 2, :]
                    
                    aligned_ft = frame_times - vidshift - goCue[tr_idx]
                    dt_frames = np.median(np.diff(aligned_ft))
                    
                    if dt_frames > 0:
                        vel = compute_velocity_from_xy(tongue_x, tongue_y, aligned_ft, dt_frames)
                        vel[tongue_conf < 0.9] = np.nan
                        valid = ~np.isnan(vel)
                        if valid.sum() > 2:
                            tongue_vel[:, tr_idx] = np.interp(time_centers, aligned_ft[valid], vel[valid],
                                                              left=np.nan, right=np.nan)
                
                if len(paw_indices) > 0:
                    ts = bottom_cam['ts'][0, tr_idx]
                    frame_times = bottom_cam['frameTimes'][0, tr_idx].flatten()
                    aligned_ft = frame_times - vidshift - goCue[tr_idx]
                    dt_frames = np.median(np.diff(aligned_ft))
                    
                    if dt_frames > 0:
                        paw_vels = []
                        for pidx in paw_indices:
                            if ts.ndim == 3:
                                if ts.shape[2] == len(bottom_feats):
                                    px = ts[:, 0, pidx]
                                    py = ts[:, 1, pidx]
                                    pc = ts[:, 2, pidx]
                                else:
                                    px = ts[pidx, 0, :]
                                    py = ts[pidx, 1, :]
                                    pc = ts[pidx, 2, :]
                            vel = compute_velocity_from_xy(px, py, aligned_ft, dt_frames)
                            vel[pc < 0.9] = np.nan
                            paw_vels.append(vel)
                        
                        avg_vel = np.nanmean(paw_vels, axis=0)
                        valid = ~np.isnan(avg_vel)
                        if valid.sum() > 2:
                            paw_vel[:, tr_idx] = np.interp(time_centers, aligned_ft[valid], avg_vel[valid],
                                                           left=np.nan, right=np.nan)
            except:
                continue
    except Exception as e:
        print(f"  {sess_id}: DLC loading error: {e}")
        has_video = False
    
    return tongue_vel, paw_vel, has_video


# ============================================================================
# MOTION ENERGY
# ============================================================================

def load_motion_energy(me_fpath, fdata, fmt, ntrials, goCue, time_centers, sess_id):
    """Load and align motion energy data."""
    n_time = len(time_centers)
    me_all = np.full((n_time, ntrials), np.nan, dtype=np.float32)
    has_me = True
    
    if not os.path.exists(me_fpath):
        print(f"  {sess_id}: No motion energy file")
        return me_all, False
    
    try:
        me_data = sio.loadmat(me_fpath, squeeze_me=False)
        me_raw = me_data['me']
        
        # Handle different ME file formats
        if hasattr(me_raw, 'dtype') and me_raw.dtype.names and 'data' in me_raw.dtype.names:
            # Standard format: me is a struct with 'data' field
            me_trials = me_raw['data'][0, 0]
        elif me_raw.dtype == object:
            # Alternative format: me IS the data array directly (nTrials, 1)
            me_trials = me_raw
        else:
            print(f"  {sess_id}: Unknown ME format")
            return me_all, False
        
        # Compute video offset
        if fmt == 'h5':
            f = fdata
            try:
                bitStart = np.nanmedian(f['obj/bp/ev/bitStart'][:].flatten())
                sglx_bitstart = np.nanmedian(f['obj/sglx/bitcode/bitstart'][:].flatten())
                sglx_fs = f['obj/sglx/fs'][0, 0]
                vidshift = sglx_bitstart / sglx_fs - bitStart
            except:
                vidshift = VIDEO_OFFSET_DEFAULT
            
            # Get frame times from traj
            traj = f['obj/traj']
            side_ref = traj[0, 0]
            side_cam = f[side_ref]
        else:
            obj = fdata['obj']
            try:
                bp = obj['bp'][0,0]
                bitStart = np.nanmedian(bp['ev'][0,0]['bitStart'][0,0].flatten())
                sglx = obj['sglx'][0,0]
                sglx_bitstart = np.nanmedian(sglx['bitcode'][0,0]['bitstart'][0,0].flatten())
                sglx_fs = float(safe_scalar(sglx['fs']))
                vidshift = sglx_bitstart / sglx_fs - bitStart
            except:
                vidshift = VIDEO_OFFSET_DEFAULT
        
        for tr_idx in range(min(ntrials, me_trials.shape[0])):
            try:
                me_trial = me_trials[tr_idx, 0].flatten().astype(float)
                
                # Get frame times for this trial
                if fmt == 'h5':
                    try:
                        ft_ref = side_cam['frameTimes'][tr_idx, 0]
                        frame_times = f[ft_ref][:].flatten()
                    except:
                        frame_times = np.arange(len(me_trial)) / 400.0
                else:
                    try:
                        traj = fdata['obj']['traj'][0,0]
                        frame_times = traj[0,0]['frameTimes'][0, tr_idx].flatten()
                    except:
                        frame_times = np.arange(len(me_trial)) / 400.0
                
                # Align to goCue
                aligned_ft = frame_times - vidshift - goCue[tr_idx]
                
                # Interpolate to time_centers
                valid = ~np.isnan(me_trial) & ~np.isnan(aligned_ft)
                if valid.sum() > 2:
                    me_all[:, tr_idx] = np.interp(time_centers, aligned_ft[valid], me_trial[valid])
            except:
                continue
        
        # Fill NaNs with nearest
        for tr_idx in range(ntrials):
            col = me_all[:, tr_idx]
            nans = np.isnan(col)
            if nans.any() and not nans.all():
                col[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), col[~nans])
                me_all[:, tr_idx] = col
    
    except Exception as e:
        print(f"  {sess_id}: Motion energy loading error: {e}")
        has_me = False
    
    return me_all, has_me


# ============================================================================
# DISCRETIZATION
# ============================================================================

def discretize_velocity(vel_all, valid_trial_indices, not_visible_val=2):
    """Discretize velocity data with per-session 50th percentile threshold.
    
    Returns list of arrays, one per valid trial.
    0: < 50th percentile
    1: >= 50th percentile  
    2: not visible / no data
    """
    n_time = vel_all.shape[0]
    
    # Compute threshold from valid trials only
    valid_data = vel_all[:, valid_trial_indices]
    flat_valid = valid_data[~np.isnan(valid_data)]
    
    if len(flat_valid) == 0:
        # No valid data - all "not visible"
        return [np.full(n_time, not_visible_val, dtype=np.int64) for _ in valid_trial_indices]
    
    threshold = np.median(flat_valid)
    
    result = []
    for tr_idx in valid_trial_indices:
        trial_vel = vel_all[:, tr_idx]
        disc = np.full(n_time, not_visible_val, dtype=np.int64)
        valid_mask = ~np.isnan(trial_vel)
        disc[valid_mask & (trial_vel < threshold)] = 0
        disc[valid_mask & (trial_vel >= threshold)] = 1
        result.append(disc)
    
    return result


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Convert Economo Lab data')
    parser.add_argument('output', help='Output pickle file path')
    parser.add_argument('--full', action='store_true', default=True, help='Process all sessions (default)')
    parser.add_argument('--sample', action='store_true', help='Process only 2 sessions')
    parser.add_argument('--show-processing', action='store_true', help='Show processing plots')
    args = parser.parse_args()
    
    if args.sample:
        args.full = False
    
    print("=" * 60)
    print("Economo Lab Data Conversion")
    print("=" * 60)
    
    # Setup time axis
    time_edges = np.arange(TMIN, TMAX + DT, DT)
    time_centers = time_edges[:-1] + DT / 2
    n_time = len(time_centers)
    print(f"Time axis: {TMIN} to {TMAX}s, dt={DT*1000:.0f}ms, {n_time} bins")
    
    # Setup smoothing kernel
    kernel = causal_gaussian_kernel(SMOOTH_WIN)
    print(f"Smoothing: causal Gaussian, {SMOOTH_WIN} samples")
    
    # Select sessions
    if args.sample:
        # Pick 2 sessions: one from each dataset
        sessions_to_process = [SESSION_META[0], SESSION_META[25]]
        print(f"\nSample mode: processing {len(sessions_to_process)} sessions")
    else:
        sessions_to_process = SESSION_META
        print(f"\nFull mode: processing {len(sessions_to_process)} sessions")
    
    # Process sessions
    all_results = []
    t_start = time.time()
    
    for i, (anm, date, dataset_dir, alm_probes) in enumerate(sessions_to_process):
        print(f"\n[{i+1}/{len(sessions_to_process)}] Processing {anm}_{date}...")
        result = process_session(anm, date, dataset_dir, alm_probes, 
                                time_edges, kernel, args.show_processing)
        if result is not None:
            all_results.append(result)
    
    t_end = time.time()
    print(f"\nProcessing complete: {len(all_results)} sessions in {t_end-t_start:.1f}s")
    
    if len(all_results) == 0:
        print("ERROR: No sessions processed successfully!")
        sys.exit(1)
    
    # --- Build final data structure ---
    print("\nBuilding output data structure...")
    
    # Subjects
    all_animals = sorted(set(r['anm'] for r in all_results))
    subjects = all_animals
    subject_idx = np.array([subjects.index(r['anm']) for r in all_results])
    
    # Brain regions - all ALM
    brain_regions = ['ALM']
    brain_region_idx = [np.zeros(r['n_neurons'], dtype=int) for r in all_results]
    
    # Neural, input, output
    neural = [r['neural'] for r in all_results]
    inputs = [r['input'] for r in all_results]
    outputs = [r['output'] for r in all_results]
    
    # Names
    input_names = ['time_from_go_cue']
    output_names = ['lick_direction', 'context', 'outcome', 
                    'tongue_velocity', 'paw_velocity', 'motion_energy']
    output_values = [
        ['left', 'right', 'none'],       # lick_direction: 0=left, 1=right, 2=none
        ['WC', 'DR'],                     # context: 0=WC, 1=DR
        ['incorrect', 'correct', 'ignore'],  # outcome: 0=incorrect, 1=correct, 2=ignore
        ['below_median', 'above_median', 'not_visible'],  # tongue_velocity
        ['below_median', 'above_median', 'not_visible'],  # paw_velocity
        ['below_median', 'above_median', 'no_video'],     # motion_energy
    ]
    
    data = {
        'neural': neural,
        'input': inputs,
        'output': outputs,
        'subjects': subjects,
        'subject_idx': subject_idx,
        'brain_regions': brain_regions,
        'brain_region_idx': brain_region_idx,
        'input_names': input_names,
        'output_names': output_names,
        'output_values': output_values,
        'metadata': {
            'task_description': 'Two-context directional licking task (DR and WC) with ALM recordings. '
                               'Mice perform delayed-response and water-cued licking tasks that alternate block-wise.',
            'time_bin_size': DT * 1000,  # in ms
            'temporal_alignment_event': 'Go cue onset',
            'off_start': TMIN,
            'off_end': TMAX,
            'smooth_kernel': f'Causal Gaussian, {SMOOTH_WIN} samples',
            'low_fr_threshold': LOW_FR,
            'quality_filter': 'all (exclude garbage, gabrga, noisy, real?)',
            'trial_filter': 'Exclude early lick and stimulation trials',
            'session_info': [{'animal': r['anm'], 'date': r['date'], 
                             'session_id': r['sess_id'], 'n_neurons': r['n_neurons'],
                             'n_trials': len(r['neural'])} for r in all_results],
        }
    }
    
    # Print summary
    total_neurons = sum(r['n_neurons'] for r in all_results)
    total_trials = sum(len(r['neural']) for r in all_results)
    print(f"\nSummary:")
    print(f"  Sessions: {len(all_results)}")
    print(f"  Subjects: {len(subjects)} ({', '.join(subjects)})")
    print(f"  Total neurons: {total_neurons}")
    print(f"  Total trials: {total_trials}")
    print(f"  Time bins: {n_time}")
    print(f"  Input dims: {len(input_names)}")
    print(f"  Output dims: {len(output_names)}")
    
    # Save
    print(f"\nSaving to {args.output}...")
    with open(args.output, 'wb') as pf:
        pickle.dump(data, pf, protocol=4)
    
    fsize = os.path.getsize(args.output) / (1024**2)
    print(f"Saved: {fsize:.1f} MB")
    print("Done!")


if __name__ == '__main__':
    main()
