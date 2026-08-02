#!/usr/bin/env python3
"""Convert Economo lab electrophysiology data to decoder-compatible format.

Usage:
    python -u convert_data.py <outpicklefile> [--full|--sample] [--show-processing]
"""

import sys
import os
import argparse
import pickle
import time
import numpy as np
import h5py
from scipy.signal import windows as sig_windows
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PARAMETERS (matching reference code and paper)
# ============================================================
PARAMS = {
    'alignEvent': 'goCue',
    'tmin': -2.5,        # seconds relative to alignEvent
    'tmax': 2.5,         # seconds relative to alignEvent  
    'dt': 1/100,         # 10ms time bins
    'smooth': 15,        # smoothing window (samples)
    'bctype': 'reflect', # boundary condition for smoothing
    'lowFR': 1.0,        # minimum firing rate threshold (Hz) - paper says 1 Hz
    'quality_exclude': ['garbage', 'gabrga', 'noisy', 'real?'],  # qualities to exclude (matching MATLAB findClusters)
    'advance_movement': 0.0,  # no time shift between neural and movement data
}

# Time axis
EDGES = np.arange(PARAMS['tmin'], PARAMS['tmax'] + PARAMS['dt'], PARAMS['dt'])
TIME_AXIS = EDGES[:-1] + PARAMS['dt'] / 2
N_TIMEBINS = len(TIME_AXIS)

# Data directories
DATA_DIR = 'data/Ephys_Behavior'

# Session metadata: (animal, date, probe_numbers)
# From the loadXXX_ALMVideo.m scripts
SESSION_META = [
    ('EKH1', '2021-08-07', [1]),
    ('EKH3', '2021-08-11', [1, 2]),  # dual probe
    ('JEB6', '2021-04-18', [2]),
    ('JEB7', '2021-04-29', [2]),
    ('JEB7', '2021-04-30', [2]),
    ('JEB13', '2022-09-13', [2]),
    ('JEB13', '2022-09-14', [2]),
    ('JEB13', '2022-09-21', [1]),
    ('JEB13', '2022-09-24', [1]),
    ('JEB13', '2022-09-25', [1]),
    ('JEB14', '2022-08-22', [1]),
    ('JEB14', '2022-08-23', [1]),
    ('JEB14', '2022-08-24', [1]),
    ('JEB14', '2022-08-25', [1]),
    ('JEB15', '2022-07-26', [1]),
    ('JEB15', '2022-07-27', [1]),
    ('JEB15', '2022-07-28', [1]),
    ('JEB15', '2022-07-29', [1]),
    ('JEB19', '2023-04-18', [1]),
    ('JEB19', '2023-04-19', [1]),
    ('JEB19', '2023-04-20', [1]),
    ('JEB19', '2023-04-21', [1]),
    ('JGR2', '2021-11-16', [1]),
    ('JGR2', '2021-11-17', [1]),
    ('JGR3', '2021-11-18', [1]),
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def read_h5_string(f, ref):
    """Read a string from an HDF5 object reference."""
    data = f[ref][:].flatten()
    return ''.join(chr(c) for c in data)


def causal_gaussian_smooth(x, N, bctype='reflect'):
    """Causal Gaussian smoothing matching MATLAB mySmooth.
    
    Args:
        x: 1D or 2D array (time along first axis)
        N: window size
        bctype: boundary condition ('reflect', 'zeropad', 'none')
    
    Returns:
        Smoothed array, same shape as x
    """
    if N <= 1:
        return x
    
    was_1d = False
    if x.ndim == 1:
        x = x[:, np.newaxis]
        was_1d = True
    
    # Create causal Gaussian kernel (matching MATLAB gausswin + zero first half)
    kern = sig_windows.gaussian(N, std=(N-1)/(2*2.5))  # gausswin default
    kern[:N//2] = 0  # causal: zero out first half
    kern = kern / kern.sum()
    
    # Handle boundary conditions
    if bctype == 'reflect':
        x_padded = np.concatenate([x[:N], x], axis=0)
        trim = N
    elif bctype == 'zeropad':
        x_padded = np.concatenate([np.zeros((N, x.shape[1])), x], axis=0)
        trim = N
    else:
        x_padded = x
        trim = 0
    
    # Convolve each column
    from scipy.signal import convolve
    result = np.zeros_like(x_padded)
    for j in range(x_padded.shape[1]):
        result[:, j] = convolve(x_padded[:, j], kern, mode='same')
    
    result = result[trim:]
    
    if was_1d:
        result = result.flatten()
    
    return result


def find_video_offset(f):
    """Find offset between neural data and video file start.
    Matches findVideoOffset.m
    """
    try:
        bitStart = np.median(f['obj']['bp']['ev']['bitStart'][:].flatten())
        # mode equivalent
        from scipy import stats
        bitStart_vals = f['obj']['bp']['ev']['bitStart'][:].flatten()
        bitStart = stats.mode(bitStart_vals, keepdims=False).mode
        
        sglx_bitstart = f['obj']['sglx']['bitcode']['bitstart'][:].flatten()
        sglx_fs = f['obj']['sglx']['fs'][0, 0]
        vidFileOffset = stats.mode(sglx_bitstart, keepdims=False).mode / sglx_fs
        
        vidshift = vidFileOffset - bitStart
        return vidshift
    except Exception as e:
        print(f'    Warning: Could not compute video offset: {e}')
        return 0.0


def get_cluster_qualities(f, clu_group):
    """Get quality strings for all clusters."""
    n_clusters = clu_group['quality'].shape[0]
    qualities = []
    for i in range(n_clusters):
        ref = clu_group['quality'][i, 0]
        try:
            chars = f[ref][:].flatten()
            quality_str = ''.join(chr(c) for c in chars)
        except:
            quality_str = ''
        qualities.append(quality_str)
    return qualities


def find_clusters(qualities, exclude_list):
    """Find indices of clusters to use (matching findClusters.m with quality='all').
    Excludes garbage, gabrga, noisy, real? clusters.
    Includes all other clusters (including those with empty/null quality strings).
    """
    exclude_lower = [q.lower().strip() for q in exclude_list]
    idx = []
    for i, q in enumerate(qualities):
        q_clean = q.strip().lower()
        # Remove null characters for comparison
        q_clean = q_clean.replace('\x00', '')
        if q_clean in exclude_lower:
            continue
        idx.append(i)
    return np.array(idx, dtype=int)


def process_session(animal, date, probes, show_processing=False, session_idx=0):
    """Process a single session and return neural + behavioral data.
    
    Returns dict with keys: neural_trials, trial_info, tongue_vel, paw_vel, motion_energy,
                            n_neurons, session_id, or None if session should be skipped.
    """
    session_id = f'{animal}_{date}'
    data_file = os.path.join(DATA_DIR, f'data_structure_{session_id}.mat')
    me_file = os.path.join(DATA_DIR, f'motionEnergy_{session_id}.mat')
    
    if not os.path.exists(data_file):
        print(f'  WARNING: Data file not found: {data_file}')
        return None
    
    t_start = time.time()
    print(f'  Processing {session_id} (probes: {probes})...')
    
    f = h5py.File(data_file, 'r')
    
    # -------------------------------------------------------
    # 1. Get behavioral data
    # -------------------------------------------------------
    bp = f['obj']['bp']
    Ntrials = int(bp['Ntrials'][0, 0])
    
    L = bp['L'][:].flatten()  # left trials
    R = bp['R'][:].flatten()  # right trials
    hit = bp['hit'][:].flatten()  # correct trials
    miss = bp['miss'][:].flatten()  # error trials
    no = bp['no'][:].flatten()  # ignore/no-response trials
    autowater = bp['autowater'][:].flatten()  # WC trials
    early = bp['early'][:].flatten()  # early lick trials
    stim_enable = bp['stim']['enable'][:].flatten()  # stimulation trials
    
    # Event times
    goCue = bp['ev']['goCue'][:].flatten()
    
    # -------------------------------------------------------
    # 2. Find valid trials
    # Exclude: early lick, ignore/no-response, stimulation
    # Include: hit and miss trials
    # -------------------------------------------------------
    valid_mask = (early == 0) & (no == 0) & (stim_enable == 0) & ((hit == 1) | (miss == 1))
    valid_trials = np.where(valid_mask)[0]  # 0-indexed
    
    if len(valid_trials) < 2:
        print(f'    Skipping {session_id}: only {len(valid_trials)} valid trials')
        f.close()
        return None
    
    print(f'    Trials: {Ntrials} total, {len(valid_trials)} valid ({np.sum(hit[valid_trials]==1)} hit, {np.sum(miss[valid_trials]==1)} miss)')
    
    # -------------------------------------------------------
    # 3. Process neural data for each probe
    # -------------------------------------------------------
    all_trialdat = []  # Will be list of (n_timebins, n_neurons, n_all_trials) per probe
    all_cluid = []     # cluster indices used per probe
    
    for probe_num in probes:
        # Determine correct clu index
        # If clu has only 1 entry (shape (1,x)), always use index 0
        # If clu has multiple entries, use probe_num-1 as index
        clu_shape = f['obj']['clu'].shape
        if clu_shape[0] == 1:
            probe_idx = 0
        else:
            probe_idx = probe_num - 1
        
        # Get clu data for this probe
        clu_ref = f['obj']['clu'][probe_idx, 0]
        clu_group = f[clu_ref]
        
        # Get cluster qualities
        qualities = get_cluster_qualities(f, clu_group)
        n_total_clusters = len(qualities)
        
        # Find clusters to use (quality filter)
        cluid = find_clusters(qualities, PARAMS['quality_exclude'])
        print(f'    Probe {probe_num}: {n_total_clusters} total clusters, {len(cluid)} after quality filter')
        
        if len(cluid) == 0:
            continue
        
        # Align spikes to goCue and bin
        # For each cluster, get spike times aligned to goCue
        trialdat = np.zeros((N_TIMEBINS, len(cluid), Ntrials), dtype=np.float32)
        
        for ci, clu_idx in enumerate(cluid):
            # Get spike times and trial assignments
            tm_ref = clu_group['trialtm'][clu_idx, 0]
            trial_ref = clu_group['trial'][clu_idx, 0]
            
            spike_trialtm = f[tm_ref][:].flatten()  # spike times relative to trial start
            spike_trial = f[trial_ref][:].flatten().astype(int)  # 1-indexed trial numbers
            
            # Get absolute spike times for alignment
            tm_abs_ref = clu_group['tm'][clu_idx, 0]
            spike_tm_abs = f[tm_abs_ref][:].flatten()
            
            # Align to goCue: trialtm_aligned = trialtm - goCue(trial)
            # But we need to compute aligned times
            # In MATLAB: obj.clu{prbnum}(clu).trialtm_aligned = obj.clu{prbnum}(clu).trialtm - event
            # where event = obj.bp.ev.goCue(obj.clu{prbnum}(clu).trial)
            
            # Compute aligned spike times
            valid_spike_mask = (spike_trial >= 1) & (spike_trial <= Ntrials)
            spike_trial_valid = spike_trial[valid_spike_mask]
            spike_trialtm_valid = spike_trialtm[valid_spike_mask]
            
            # goCue times for each spike's trial
            spike_goCue = goCue[spike_trial_valid - 1]  # 0-index into goCue
            spike_aligned = spike_trialtm_valid - spike_goCue
            
            # Bin spikes for each trial
            for trial_num in range(1, Ntrials + 1):
                trial_mask = spike_trial_valid == trial_num
                if not np.any(trial_mask):
                    continue
                
                trial_spikes = spike_aligned[trial_mask]
                
                # Histogram
                counts, _ = np.histogram(trial_spikes, bins=EDGES)
                
                # Smooth: convert to firing rate then smooth
                fr = counts / PARAMS['dt']  # spks/sec
                fr_smooth = causal_gaussian_smooth(fr, PARAMS['smooth'], PARAMS['bctype'])
                
                trialdat[:, ci, trial_num - 1] = fr_smooth
        
        all_trialdat.append(trialdat)
        all_cluid.append(cluid)
    
    if len(all_trialdat) == 0:
        print(f'    Skipping {session_id}: no valid clusters')
        f.close()
        return None
    
    # Concatenate across probes
    trialdat = np.concatenate(all_trialdat, axis=1)  # (time, neurons, trials)
    n_neurons_before_fr = trialdat.shape[1]
    
    # -------------------------------------------------------
    # 4. Remove low firing rate clusters
    # Mean FR across all trials and time, threshold at lowFR
    # Matching removeLowFRClusters.m:
    #   meanFRs = mean(mean(obj.psth{prbnum},3,'omitnan'),'omitnan');
    #   use = meanFRs > lowFR;
    # psth is (time, neurons, conditions) - mean over conditions then time
    # But for simplicity, compute mean FR across all trials and time
    # Actually the MATLAB code uses psth (condition-averaged), not trialdat
    # Let's compute it the same way: mean over all time and all trials
    # -------------------------------------------------------
    # The MATLAB code computes meanFRs from psth which is condition-averaged
    # meanFRs = mean(mean(psth, dim=3), dim=1) - mean over conditions, then over time
    # This gives mean FR for each neuron averaged across conditions and time
    # For our purposes, we can compute mean FR across all trials and time
    meanFRs = np.nanmean(np.nanmean(trialdat, axis=2), axis=0)  # mean over trials, then time
    fr_mask = meanFRs > PARAMS['lowFR']
    
    trialdat = trialdat[:, fr_mask, :]
    n_neurons = trialdat.shape[1]
    
    print(f'    Neurons: {n_neurons_before_fr} after quality, {n_neurons} after FR filter (>{PARAMS["lowFR"]} Hz)')
    
    if n_neurons < 10:
        print(f'    Skipping {session_id}: only {n_neurons} neurons (need >= 10)')
        f.close()
        return None
    
    # -------------------------------------------------------
    # 5. Extract trial-level behavioral variables
    # -------------------------------------------------------
    lick_direction = R.copy()  # 1=right, 0=left
    context = 1 - autowater  # DR=1, WC=0 (autowater=1 means WC)
    outcome = hit.copy()  # 1=correct, 0=incorrect
    
    # -------------------------------------------------------
    # 6. Compute tongue velocity (time-varying)
    # From DLC side view (view 1, index 0), feature 'tongue'
    # -------------------------------------------------------
    tongue_vel = compute_tongue_velocity(f, goCue, Ntrials)
    
    # -------------------------------------------------------
    # 7. Compute paw velocity (time-varying)
    # From DLC top view (view 2, index 1), features 'top_paw' and 'bottom_paw'
    # -------------------------------------------------------
    paw_vel = compute_paw_velocity(f, goCue, Ntrials)
    
    # -------------------------------------------------------
    # 8. Load and process motion energy
    # -------------------------------------------------------
    motion_energy = load_motion_energy(f, me_file, goCue, Ntrials)
    
    f.close()
    
    # -------------------------------------------------------
    # 9. Package per-trial data for valid trials only
    # -------------------------------------------------------
    neural_trials = []
    trial_info = []
    tongue_vel_trials = []
    paw_vel_trials = []
    me_trials = []
    
    for trial_idx in valid_trials:
        # Neural: (n_neurons, n_timebins)
        neural_trials.append(trialdat[:, :, trial_idx].T.astype(np.float32))  # (neurons, time)
        
        trial_info.append({
            'lick_direction': int(lick_direction[trial_idx]),
            'context': int(context[trial_idx]),
            'outcome': int(outcome[trial_idx]),
        })
        
        if tongue_vel is not None:
            tongue_vel_trials.append(tongue_vel[:, trial_idx].astype(np.float32))
        else:
            tongue_vel_trials.append(np.zeros(N_TIMEBINS, dtype=np.float32))
        
        if paw_vel is not None:
            paw_vel_trials.append(paw_vel[:, trial_idx].astype(np.float32))
        else:
            paw_vel_trials.append(np.zeros(N_TIMEBINS, dtype=np.float32))
        
        if motion_energy is not None:
            me_trials.append(motion_energy[:, trial_idx].astype(np.float32))
        else:
            me_trials.append(np.zeros(N_TIMEBINS, dtype=np.float32))
    
    elapsed = time.time() - t_start
    print(f'    Done: {n_neurons} neurons, {len(valid_trials)} trials, {elapsed:.1f}s')
    
    result = {
        'neural_trials': neural_trials,
        'trial_info': trial_info,
        'tongue_vel': tongue_vel_trials,
        'paw_vel': paw_vel_trials,
        'motion_energy': me_trials,
        'n_neurons': n_neurons,
        'session_id': session_id,
        'animal': animal,
    }
    
    if show_processing:
        result['trialdat_full'] = trialdat  # for plotting
        result['valid_trials'] = valid_trials
        result['goCue'] = goCue
        result['lick_direction_all'] = lick_direction
        result['context_all'] = context
        result['outcome_all'] = outcome
    
    return result


def compute_tongue_velocity(f, goCue, Ntrials):
    """Compute tongue velocity from DLC side view data.
    
    Matches findPosition + findVelocity from reference code.
    Uses view 1 (index 0), feature 'tongue' (index 0).
    Returns: (n_timebins, Ntrials) array of tongue speed, or None.
    """
    try:
        traj_ref = f['obj']['traj'][0, 0]  # view 1 (side)
        traj_group = f[traj_ref]
        
        taxis = TIME_AXIS + PARAMS['advance_movement']
        vidshift = find_video_offset(f)
        
        # Find tongue feature index (should be 0 for side view)
        feat_ref = traj_group['featNames'][0, 0]
        feat_data = f[feat_ref]
        feat_names = []
        for j in range(feat_data.shape[1]):
            ref = feat_data[0, j]
            chars = f[ref][:].flatten()
            feat_names.append(''.join(chr(c) for c in chars))
        
        tongue_idx = None
        for i, name in enumerate(feat_names):
            if name == 'tongue':
                tongue_idx = i
                break
        
        if tongue_idx is None:
            print('    Warning: tongue feature not found in view 1')
            return None
        
        xpos = np.full((N_TIMEBINS, Ntrials), np.nan, dtype=np.float64)
        ypos = np.full((N_TIMEBINS, Ntrials), np.nan, dtype=np.float64)
        
        for trix in range(Ntrials):
            try:
                # Get trajectory data for this trial
                ts_ref = traj_group['ts'][trix, 0]
                ts_data = f[ts_ref][:]  # (n_features, 3, n_timepoints) or similar
                
                # Check for NdroppedFrames
                ndrop_ref = traj_group['NdroppedFrames'][trix, 0]
                ndrop = f[ndrop_ref][:].flatten()
                if np.isnan(ndrop).any():
                    continue
                
                # Get frameTimes
                try:
                    ft_ref = traj_group['frameTimes'][trix, 0]
                    frameTimes = f[ft_ref][:].flatten()
                except:
                    n_frames = ts_data.shape[-1] if ts_data.ndim == 3 else ts_data.shape[0]
                    frameTimes = np.arange(1, n_frames + 1) / 400.0
                
                if np.all(np.isnan(frameTimes)):
                    continue
                
                # Extract x, y for tongue
                # ts shape could be (n_features, 3, n_timepoints) or (n_timepoints, 3, n_features)
                if ts_data.ndim == 3:
                    if ts_data.shape[0] == len(feat_names):  # (features, 3, timepoints)
                        x = ts_data[tongue_idx, 0, :]
                        y = ts_data[tongue_idx, 1, :]
                    elif ts_data.shape[2] == len(feat_names):  # (timepoints, 3, features)
                        x = ts_data[:, 0, tongue_idx]
                        y = ts_data[:, 1, tongue_idx]
                    else:
                        # Try first interpretation
                        x = ts_data[tongue_idx, 0, :]
                        y = ts_data[tongue_idx, 1, :]
                elif ts_data.ndim == 2:
                    continue
                else:
                    continue
                
                # Align: interpolate to neural time axis
                aligned_times = frameTimes - vidshift - goCue[trix]
                
                # Interpolate
                from scipy.interpolate import interp1d
                valid = ~np.isnan(aligned_times)
                if np.sum(valid) < 2:
                    continue
                
                try:
                    fx = interp1d(aligned_times[valid], x[valid], bounds_error=False, fill_value=np.nan)
                    fy = interp1d(aligned_times[valid], y[valid], bounds_error=False, fill_value=np.nan)
                    xpos[:, trix] = fx(taxis)
                    ypos[:, trix] = fy(taxis)
                except:
                    continue
                    
            except Exception as e:
                continue
        
        # Compute velocity (matching findVelocity.m)
        # For tongue: velocity = gradient(position), NaN -> 0
        xvel = np.zeros((N_TIMEBINS, Ntrials), dtype=np.float64)
        yvel = np.zeros((N_TIMEBINS, Ntrials), dtype=np.float64)
        
        for trix in range(Ntrials):
            xv = np.gradient(xpos[:, trix])
            yv = np.gradient(ypos[:, trix])
            # For tongue: set NaN velocity to 0
            xv[np.isnan(xv)] = 0
            yv[np.isnan(yv)] = 0
            xvel[:, trix] = xv
            yvel[:, trix] = yv
        
        # Compute speed (magnitude of velocity)
        speed = np.sqrt(xvel**2 + yvel**2)
        
        return speed
        
    except Exception as e:
        print(f'    Warning: Could not compute tongue velocity: {e}')
        return None


def compute_paw_velocity(f, goCue, Ntrials):
    """Compute paw velocity from DLC top view data.
    
    Uses view 2 (index 1), features 'top_paw' and 'bottom_paw'.
    Returns: (n_timebins, Ntrials) array of paw speed, or None.
    """
    try:
        traj_ref = f['obj']['traj'][1, 0]  # view 2 (top)
        traj_group = f[traj_ref]
        
        taxis = TIME_AXIS + PARAMS['advance_movement']
        vidshift = find_video_offset(f)
        
        # Find paw feature indices
        feat_ref = traj_group['featNames'][0, 0]
        feat_data = f[feat_ref]
        feat_names = []
        for j in range(feat_data.shape[1]):
            ref = feat_data[0, j]
            chars = f[ref][:].flatten()
            feat_names.append(''.join(chr(c) for c in chars))
        
        paw_indices = []
        for i, name in enumerate(feat_names):
            if 'paw' in name.lower():
                paw_indices.append(i)
        
        if len(paw_indices) == 0:
            print('    Warning: paw features not found in view 2')
            return None
        
        # Average velocity across paw features
        all_speeds = []
        
        for paw_idx in paw_indices:
            xpos = np.full((N_TIMEBINS, Ntrials), np.nan, dtype=np.float64)
            ypos = np.full((N_TIMEBINS, Ntrials), np.nan, dtype=np.float64)
            
            for trix in range(Ntrials):
                try:
                    ts_ref = traj_group['ts'][trix, 0]
                    ts_data = f[ts_ref][:]
                    
                    ndrop_ref = traj_group['NdroppedFrames'][trix, 0]
                    ndrop = f[ndrop_ref][:].flatten()
                    if np.isnan(ndrop).any():
                        continue
                    
                    try:
                        ft_ref = traj_group['frameTimes'][trix, 0]
                        frameTimes = f[ft_ref][:].flatten()
                    except:
                        n_frames = ts_data.shape[-1] if ts_data.ndim == 3 else ts_data.shape[0]
                        frameTimes = np.arange(1, n_frames + 1) / 400.0
                    
                    if np.all(np.isnan(frameTimes)):
                        continue
                    
                    # Extract x, y for paw
                    if ts_data.ndim == 3:
                        if ts_data.shape[0] == len(feat_names):
                            x = ts_data[paw_idx, 0, :]
                            y = ts_data[paw_idx, 1, :]
                        elif ts_data.shape[2] == len(feat_names):
                            x = ts_data[:, 0, paw_idx]
                            y = ts_data[:, 1, paw_idx]
                        else:
                            x = ts_data[paw_idx, 0, :]
                            y = ts_data[paw_idx, 1, :]
                    else:
                        continue
                    
                    aligned_times = frameTimes - vidshift - goCue[trix]
                    
                    from scipy.interpolate import interp1d
                    valid = ~np.isnan(aligned_times)
                    if np.sum(valid) < 2:
                        continue
                    
                    # For non-tongue: smooth with window 1 (no-op) and fill missing
                    fx = interp1d(aligned_times[valid], x[valid], bounds_error=False, fill_value=np.nan)
                    fy = interp1d(aligned_times[valid], y[valid], bounds_error=False, fill_value=np.nan)
                    xp = fx(taxis)
                    yp = fy(taxis)
                    
                    # Fill missing with nearest
                    mask = ~np.isnan(xp)
                    if np.any(mask):
                        from scipy.interpolate import interp1d as interp1d_fill
                        indices = np.arange(len(xp))
                        xp = np.interp(indices, indices[mask], xp[mask])
                    mask = ~np.isnan(yp)
                    if np.any(mask):
                        yp = np.interp(indices, indices[mask], yp[mask])
                    
                    xpos[:, trix] = xp
                    ypos[:, trix] = yp
                    
                except:
                    continue
            
            # Compute velocity
            xvel = np.zeros((N_TIMEBINS, Ntrials), dtype=np.float64)
            yvel = np.zeros((N_TIMEBINS, Ntrials), dtype=np.float64)
            
            for trix in range(Ntrials):
                xv = np.gradient(xpos[:, trix])
                yv = np.gradient(ypos[:, trix])
                
                # For non-tongue: subtract baseline derivative and fill missing
                basederiv_x = np.nanmedian(np.diff(xpos[:, trix]))
                basederiv_y = np.nanmedian(np.diff(ypos[:, trix]))
                if not np.isnan(basederiv_x):
                    xv = xv - basederiv_x
                if not np.isnan(basederiv_y):
                    yv = yv - basederiv_y
                
                # Fill missing
                mask = ~np.isnan(xv)
                if np.any(mask):
                    indices = np.arange(len(xv))
                    xv = np.interp(indices, indices[mask], xv[mask])
                mask = ~np.isnan(yv)
                if np.any(mask):
                    yv = np.interp(indices, indices[mask], yv[mask])
                
                xvel[:, trix] = xv
                yvel[:, trix] = yv
            
            speed = np.sqrt(xvel**2 + yvel**2)
            all_speeds.append(speed)
        
        # Average across paw features
        avg_speed = np.mean(all_speeds, axis=0)
        return avg_speed
        
    except Exception as e:
        print(f'    Warning: Could not compute paw velocity: {e}')
        return None


def load_motion_energy(f, me_file, goCue, Ntrials):
    """Load and align motion energy to neural time axis.
    
    Matches loadMotionEnergy.m.
    Returns: (n_timebins, Ntrials) array, or None.
    """
    if not os.path.exists(me_file):
        print(f'    Warning: Motion energy file not found: {me_file}')
        return None
    
    try:
        import scipy.io as sio
        me_raw = sio.loadmat(me_file)
        me_struct = me_raw['me'][0, 0]
        me_data = me_struct['data']
        # Handle nested struct: if me.data is itself a struct with 'data' field
        # (matching MATLAB: if isstruct(me.data), me.data = me.data.data)
        if me_data.dtype.names is not None and 'data' in me_data.dtype.names:
            me_data = me_data[0, 0]['data']
        # me_data should now be (Ntrials, 1) object array
        
        taxis = TIME_AXIS + PARAMS['advance_movement']
        vidshift = find_video_offset(f)
        
        me_aligned = np.zeros((N_TIMEBINS, Ntrials), dtype=np.float64)
        
        for trix in range(min(Ntrials, me_data.shape[0])):
            try:
                me_trial = me_data[trix, 0].flatten()  # motion energy time series
                
                # Get frameTimes from traj
                try:
                    traj_ref = f['obj']['traj'][0, 0]
                    traj_group = f[traj_ref]
                    ft_ref = traj_group['frameTimes'][trix, 0]
                    frameTimes = f[ft_ref][:].flatten()
                except:
                    frameTimes = np.arange(1, len(me_trial) + 1) / 400.0
                
                if np.all(np.isnan(frameTimes)) or len(frameTimes) == 0:
                    frameTimes = np.arange(1, len(me_trial) + 1) / 400.0
                
                # Align times
                aligned_times = frameTimes - vidshift - goCue[trix]
                
                # Truncate to match lengths
                min_len = min(len(aligned_times), len(me_trial))
                aligned_times = aligned_times[:min_len]
                me_trial = me_trial[:min_len]
                
                # Interpolate to neural time axis
                valid = ~np.isnan(aligned_times) & ~np.isnan(me_trial)
                if np.sum(valid) < 2:
                    continue
                
                from scipy.interpolate import interp1d
                f_interp = interp1d(aligned_times[valid], me_trial[valid], 
                                   bounds_error=False, fill_value=np.nan)
                me_interp = f_interp(taxis)
                
                # Fill NaN with nearest
                mask = ~np.isnan(me_interp)
                if np.any(mask):
                    indices = np.arange(len(me_interp))
                    me_interp = np.interp(indices, indices[mask], me_interp[mask])
                
                me_aligned[:, trix] = me_interp
                
            except Exception as e:
                continue
        
        return me_aligned
        
    except Exception as e:
        print(f'    Warning: Could not load motion energy: {e}')
        return None


def discretize_continuous(values_per_trial, threshold_percentile=50):
    """Discretize continuous time-varying values into 2 bins using per-session threshold.
    
    Args:
        values_per_trial: list of 1D arrays (time-varying values per trial)
        threshold_percentile: percentile for threshold (50 = median)
    
    Returns:
        list of 1D arrays with values 0 or 1 (as float32)
    """
    # Compute threshold from all valid values across all trials
    all_values = np.concatenate([v.flatten() for v in values_per_trial])
    all_values = all_values[~np.isnan(all_values)]
    
    if len(all_values) == 0:
        return [np.zeros_like(v, dtype=np.float32) for v in values_per_trial]
    
    threshold = np.percentile(all_values, threshold_percentile)
    
    # Handle edge case: if threshold is 0 (e.g., tongue velocity is mostly 0
    # when tongue is not visible), use median of positive values instead
    if threshold == 0:
        pos_values = all_values[all_values > 0]
        if len(pos_values) > 0:
            threshold = np.percentile(pos_values, threshold_percentile)
    
    result = []
    for v in values_per_trial:
        disc = (v >= threshold).astype(np.float32)
        result.append(disc)
    
    return result


def plot_processing(session_result, session_idx):
    """Plot processing steps for visual verification."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    session_id = session_result['session_id']
    neural = session_result['neural_trials']
    info = session_result['trial_info']
    tongue = session_result['tongue_vel']
    paw = session_result['paw_vel']
    me = session_result['motion_energy']
    
    fig, axes = plt.subplots(4, 3, figsize=(20, 16))
    fig.suptitle(f'Processing: {session_id}', fontsize=16)
    
    # Plot 1: Neural activity heatmap (first trial)
    if len(neural) > 0:
        ax = axes[0, 0]
        trial_data = neural[0]  # (neurons, time)
        im = ax.imshow(trial_data, aspect='auto', cmap='viridis',
                       extent=[TIME_AXIS[0], TIME_AXIS[-1], trial_data.shape[0], 0])
        ax.set_title('Neural (Trial 1)')
        ax.set_xlabel('Time from goCue (s)')
        ax.set_ylabel('Neuron')
        ax.axvline(0, color='r', linestyle='--', alpha=0.5)
        plt.colorbar(im, ax=ax)
    
    # Plot 2: Mean neural activity across neurons
    ax = axes[0, 1]
    if len(neural) > 5:
        for i in range(min(5, len(neural))):
            mean_fr = np.mean(neural[i], axis=0)
            ax.plot(TIME_AXIS, mean_fr, alpha=0.5, label=f'Trial {i}')
        ax.set_title('Mean FR across neurons')
        ax.set_xlabel('Time from goCue (s)')
        ax.set_ylabel('Mean FR (spks/s)')
        ax.axvline(0, color='r', linestyle='--', alpha=0.5)
    
    # Plot 3: Trial type distribution
    ax = axes[0, 2]
    n_right = sum(1 for t in info if t['lick_direction'] == 1)
    n_left = sum(1 for t in info if t['lick_direction'] == 0)
    n_dr = sum(1 for t in info if t['context'] == 1)
    n_wc = sum(1 for t in info if t['context'] == 0)
    n_correct = sum(1 for t in info if t['outcome'] == 1)
    n_incorrect = sum(1 for t in info if t['outcome'] == 0)
    ax.bar(['Right', 'Left', 'DR', 'WC', 'Correct', 'Incorrect'],
           [n_right, n_left, n_dr, n_wc, n_correct, n_incorrect])
    ax.set_title('Trial Distribution')
    
    # Plot 4-6: Tongue velocity
    ax = axes[1, 0]
    for i in range(min(5, len(tongue))):
        ax.plot(TIME_AXIS, tongue[i], alpha=0.5)
    ax.set_title('Tongue velocity (raw)')
    ax.set_xlabel('Time from goCue (s)')
    ax.axvline(0, color='r', linestyle='--', alpha=0.5)
    
    # Plot 7-9: Paw velocity
    ax = axes[1, 1]
    for i in range(min(5, len(paw))):
        ax.plot(TIME_AXIS, paw[i], alpha=0.5)
    ax.set_title('Paw velocity (raw)')
    ax.set_xlabel('Time from goCue (s)')
    ax.axvline(0, color='r', linestyle='--', alpha=0.5)
    
    # Plot 10: Motion energy
    ax = axes[1, 2]
    for i in range(min(5, len(me))):
        ax.plot(TIME_AXIS, me[i], alpha=0.5)
    ax.set_title('Motion energy (raw)')
    ax.set_xlabel('Time from goCue (s)')
    ax.axvline(0, color='r', linestyle='--', alpha=0.5)
    
    # Plot discretized outputs
    tongue_disc = discretize_continuous(tongue)
    paw_disc = discretize_continuous(paw)
    me_disc = discretize_continuous(me)
    
    ax = axes[2, 0]
    for i in range(min(5, len(tongue_disc))):
        ax.plot(TIME_AXIS, tongue_disc[i] + i*1.1, alpha=0.7)
    ax.set_title('Tongue vel (discretized)')
    ax.set_xlabel('Time from goCue (s)')
    ax.axvline(0, color='r', linestyle='--', alpha=0.5)
    
    ax = axes[2, 1]
    for i in range(min(5, len(paw_disc))):
        ax.plot(TIME_AXIS, paw_disc[i] + i*1.1, alpha=0.7)
    ax.set_title('Paw vel (discretized)')
    ax.set_xlabel('Time from goCue (s)')
    ax.axvline(0, color='r', linestyle='--', alpha=0.5)
    
    ax = axes[2, 2]
    for i in range(min(5, len(me_disc))):
        ax.plot(TIME_AXIS, me_disc[i] + i*1.1, alpha=0.7)
    ax.set_title('Motion energy (discretized)')
    ax.set_xlabel('Time from goCue (s)')
    ax.axvline(0, color='r', linestyle='--', alpha=0.5)
    
    # Plot neural + output alignment check
    ax = axes[3, 0]
    if len(neural) > 0 and len(tongue) > 0:
        mean_fr = np.mean(neural[0], axis=0)
        ax.plot(TIME_AXIS, mean_fr / np.max(mean_fr), label='Neural (norm)', alpha=0.7)
        ax.plot(TIME_AXIS, tongue[0] / (np.max(tongue[0]) + 1e-10), label='Tongue vel (norm)', alpha=0.7)
        ax.set_title('Alignment check (Trial 1)')
        ax.set_xlabel('Time from goCue (s)')
        ax.legend(fontsize=8)
        ax.axvline(0, color='r', linestyle='--', alpha=0.5)
    
    ax = axes[3, 1]
    ax.text(0.5, 0.5, f'Session: {session_id}\nNeurons: {neural[0].shape[0] if neural else 0}\nTrials: {len(neural)}\nTime bins: {N_TIMEBINS}',
            transform=ax.transAxes, ha='center', va='center', fontsize=12)
    ax.set_title('Summary')
    
    axes[3, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'processing_{session_id}.png', dpi=100)
    plt.close()
    print(f'    Saved processing plot: processing_{session_id}.png')


# ============================================================
# MAIN CONVERSION
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Convert Economo lab data to decoder format')
    parser.add_argument('output', help='Output pickle file path')
    parser.add_argument('--full', action='store_true', default=True, help='Process all sessions (default)')
    parser.add_argument('--sample', action='store_true', help='Process only 2 sessions for testing')
    parser.add_argument('--show-processing', action='store_true', help='Plot processing visualizations')
    args = parser.parse_args()
    
    if args.sample:
        args.full = False
    
    print('=' * 60)
    print('Data Conversion: Economo Lab Ephys + Behavior')
    print('=' * 60)
    print(f'Output file: {args.output}')
    print(f'Mode: {"sample (2 sessions)" if args.sample else "full"}')
    print(f'Show processing: {args.show_processing}')
    print()
    
    # Select sessions
    sessions = SESSION_META.copy()
    if args.sample:
        # Pick 2 sessions from different animals
        sessions = [sessions[0], sessions[2]]  # EKH1 and JEB6
    
    print(f'Processing {len(sessions)} sessions...')
    print()
    
    # Process all sessions
    all_results = []
    t_total_start = time.time()
    
    for idx, (animal, date, probes) in enumerate(sessions):
        result = process_session(animal, date, probes, 
                               show_processing=args.show_processing,
                               session_idx=idx)
        if result is not None:
            all_results.append(result)
            if args.show_processing:
                plot_processing(result, idx)
    
    print(f'\nProcessed {len(all_results)} sessions successfully')
    
    if len(all_results) == 0:
        print('ERROR: No sessions processed successfully!')
        sys.exit(1)
    
    # -------------------------------------------------------
    # Build the output data structure
    # -------------------------------------------------------
    print('\nBuilding output data structure...')
    
    # Get unique subjects
    subjects = sorted(list(set(r['animal'] for r in all_results)))
    
    # Build data dict
    neural_list = []
    input_list = []
    output_list = []
    subject_idx_list = []
    brain_region_idx_list = []
    
    for result in all_results:
        # Neural: list of trials, each (n_neurons, n_timepoints)
        neural_list.append(result['neural_trials'])
        
        # Subject index
        subj_idx = subjects.index(result['animal'])
        subject_idx_list.append(subj_idx)
        
        # Brain region: all neurons are ALM
        n_neurons = result['n_neurons']
        brain_region_idx_list.append(np.zeros(n_neurons, dtype=np.int64))
        
        # Discretize continuous outputs per session
        tongue_disc = discretize_continuous(result['tongue_vel'])
        paw_disc = discretize_continuous(result['paw_vel'])
        me_disc = discretize_continuous(result['motion_energy'])
        
        # Build input and output for each trial
        session_inputs = []
        session_outputs = []
        
        for trial_idx in range(len(result['neural_trials'])):
            info = result['trial_info'][trial_idx]
            
            # Input: time from goCue (1, n_timepoints)
            time_input = TIME_AXIS.astype(np.float32).reshape(1, -1)
            session_inputs.append(time_input)
            
            # Output: (6, n_timepoints) for time-varying, or (6,) for per-trial
            # Per-trial outputs: lick_direction, context, outcome
            # Time-varying outputs: tongue_vel, paw_vel, motion_energy
            lick_dir = np.full(N_TIMEBINS, info['lick_direction'], dtype=np.int64)
            context = np.full(N_TIMEBINS, info['context'], dtype=np.int64)
            outcome = np.full(N_TIMEBINS, info['outcome'], dtype=np.int64)
            
            output_array = np.stack([
                lick_dir,
                context,
                outcome,
                tongue_disc[trial_idx].astype(np.int64),
                paw_disc[trial_idx].astype(np.int64),
                me_disc[trial_idx].astype(np.int64),
            ], axis=0)  # (6, n_timepoints)
            
            session_outputs.append(output_array)
        
        input_list.append(session_inputs)
        output_list.append(session_outputs)
    
    # Build final data dict
    data = {
        'neural': neural_list,
        'input': input_list,
        'output': output_list,
        
        'subjects': subjects,
        'subject_idx': np.array(subject_idx_list, dtype=np.int64),
        
        'brain_regions': ['ALM'],
        'brain_region_idx': brain_region_idx_list,
        
        'input_names': ['time_from_goCue'],
        'output_names': ['lick_direction', 'behavioral_context', 'outcome',
                        'tongue_velocity', 'paw_velocity', 'motion_energy'],
        'output_values': [
            ['left', 'right'],           # lick_direction
            ['WC', 'DR'],                # behavioral_context  
            ['incorrect', 'correct'],    # outcome
            ['low', 'high'],             # tongue_velocity
            ['low', 'high'],             # paw_velocity
            ['low', 'high'],             # motion_energy
        ],
        
        'metadata': {
            'task_description': 'Two-context delayed response (DR) and water-cued (WC) licking task. '
                               'Mice perform directional tongue movements to left or right lickport. '
                               'In DR context, auditory cue indicates reward location after delay. '
                               'In WC context, water is presented directly at random port.',
            'time_bin_size': PARAMS['dt'] * 1000,  # 10 ms
            'temporal_alignment_event': 'Go cue onset',
            'off_start': PARAMS['tmin'],  # -2.5 s
            'off_end': PARAMS['tmax'],    # 2.5 s
            'smoothing': f"Causal Gaussian kernel, window={PARAMS['smooth']} samples",
            'low_fr_threshold': PARAMS['lowFR'],
            'quality_filter': 'all (excludes garbage, noisy)',
            'brain_region': 'ALM (anterior lateral motor cortex)',
            'species': 'mouse',
            'recording_type': 'Neuropixels electrophysiology',
        }
    }
    
    # Print summary
    print('\n' + '=' * 60)
    print('CONVERSION SUMMARY')
    print('=' * 60)
    total_neurons = sum(r['n_neurons'] for r in all_results)
    total_trials = sum(len(r['neural_trials']) for r in all_results)
    print(f'Subjects: {len(subjects)}')
    print(f'Sessions: {len(all_results)}')
    print(f'Total neurons: {total_neurons}')
    print(f'Total trials: {total_trials}')
    print(f'Neurons per session: {[r["n_neurons"] for r in all_results]}')
    print(f'Trials per session: {[len(r["neural_trials"]) for r in all_results]}')
    print(f'Time bins: {N_TIMEBINS}')
    print(f'Time bin size: {PARAMS["dt"]*1000:.1f} ms')
    print(f'Time window: [{PARAMS["tmin"]}, {PARAMS["tmax"]}] s from goCue')
    
    # Output distributions
    all_lick = [info['lick_direction'] for r in all_results for info in r['trial_info']]
    all_context = [info['context'] for r in all_results for info in r['trial_info']]
    all_outcome = [info['outcome'] for r in all_results for info in r['trial_info']]
    print(f'\nLick direction: {np.mean(all_lick):.3f} right (expect ~0.5)')
    print(f'Context: {np.mean(all_context):.3f} DR (expect varies by session)')
    print(f'Outcome: {np.mean(all_outcome):.3f} correct')
    
    elapsed = time.time() - t_total_start
    print(f'\nTotal processing time: {elapsed:.1f}s')
    
    # Save
    print(f'\nSaving to {args.output}...')
    with open(args.output, 'wb') as pf:
        pickle.dump(data, pf, protocol=4)
    
    file_size = os.path.getsize(args.output) / 1e6
    print(f'Saved: {file_size:.1f} MB')
    print('Done!')


if __name__ == '__main__':
    main()
