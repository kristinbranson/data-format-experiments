#!/usr/bin/env python3
"""Convert ALM electrophysiology + behavior data to decoder-compatible format.

Usage:
    python -u convert_data.py <outpicklefile> [--sample] [--full] [--show-processing]
"""
import sys
import os
import time
import argparse
import pickle
import numpy as np
import h5py
import scipy.io as sio
from scipy import stats as sp_stats
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
ALIGN_EVENT = 'goCue'
TMIN = -2.5
TMAX = 2.5
DT = 1.0 / 100  # 10ms bins
SMOOTH_WINDOW = 15
BC_TYPE = 'reflect'
LOW_FR_THRESHOLD = 1.0  # Hz
EXCLUDED_QUALITIES = {'garbage', 'noisy', 'gabrga', 'real?'}
DLC_CONFIDENCE_THRESHOLD = 0.5

EPHYS_SESSIONS = [
    ('EKH1', '2021-08-07', [2]),
    ('EKH3', '2021-08-11', [2]),
    ('JEB6', '2021-04-18', [2]),
    ('JEB7', '2021-04-29', [1]),
    ('JEB7', '2021-04-30', [1]),
    ('JEB13', '2022-09-13', [2]),
    ('JEB13', '2022-09-14', [2]),
    ('JEB13', '2022-09-21', [1]),
    ('JEB13', '2022-09-24', [1]),
    ('JEB13', '2022-09-25', [1]),
    ('JEB14', '2022-08-22', [1]),
    ('JEB14', '2022-08-23', [1]),
    ('JEB14', '2022-08-24', [1]),
    ('JEB14', '2022-08-25', [1]),
    ('JEB15', '2022-07-26', [1, 2]),
    ('JEB15', '2022-07-27', [1, 2]),
    ('JEB15', '2022-07-28', [1, 2]),
    ('JEB15', '2022-07-29', [2]),
    ('JEB19', '2023-04-18', [1]),
    ('JEB19', '2023-04-19', [1]),
    ('JEB19', '2023-04-20', [1]),
    ('JEB19', '2023-04-21', [1]),
    ('JGR2', '2021-11-16', [1]),
    ('JGR2', '2021-11-17', [1]),
    ('JGR3', '2021-11-18', [1]),
]

RD_SESSIONS = [
    ('JEB11', '2022-05-10', [1]),
    ('JEB11', '2022-05-11', [1]),
    ('JEB12', '2022-05-12', [1]),
    ('JEB12', '2022-05-13', [1]),
    ('JEB23', '2023-10-10', [1]),
    ('JEB23', '2023-10-11', [1]),
    ('JEB23', '2023-10-12', [1]),
    ('JEB23', '2023-10-13', [1]),
    ('JEB23', '2023-10-18', [1]),
    ('JEB23', '2023-10-19', [1]),
    ('JEB23', '2023-10-21', [1]),
    ('JEB24', '2023-10-23', [1]),
    ('JEB24', '2023-10-24', [1]),
    ('JEB24', '2023-10-25', [1]),
    ('JEB24', '2023-10-26', [1]),
    ('JEB24', '2023-10-27', [1]),
    ('JEB24', '2023-10-31', [1]),
    ('JEB24', '2023-11-02', [1]),
    ('JEB24', '2023-11-03', [1]),
]

EPHYS_DATA_DIR = '/app/data/Ephys_Behavior'
RD_DATA_DIR = '/app/data/RandomizedDelay_Ephys_Behavior'

# Time axis
edges = np.arange(TMIN, TMAX + DT, DT)
time_axis = edges[:-1] + DT / 2
n_time = len(time_axis)

# ============================================================
# SMOOTHING
# ============================================================
def causal_gaussian_smooth(x, N, bctype='reflect'):
    if N <= 1:
        return x
    was_1d = (x.ndim == 1)
    if was_1d:
        x = x[:, np.newaxis]
    n = np.arange(N)
    alpha = 2.5
    center = (N - 1) / 2.0
    kern = np.exp(-0.5 * (alpha * (n - center) / center) ** 2)
    kern[:int(N // 2)] = 0
    kern = kern / kern.sum()
    if bctype == 'reflect':
        x_padded = np.concatenate([x[:N], x], axis=0)
        trim = N
    elif bctype == 'zeropad':
        x_padded = np.concatenate([np.zeros((N, x.shape[1])), x], axis=0)
        trim = N
    else:
        x_padded = x
        trim = 0
    result = np.zeros_like(x_padded)
    for j in range(x_padded.shape[1]):
        result[:, j] = np.convolve(x_padded[:, j], kern, mode='same')
    result = result[trim:]
    if was_1d:
        result = result[:, 0]
    return result

# ============================================================
# DATA LOADING HELPERS
# ============================================================
def load_mat_file(fpath):
    try:
        f = h5py.File(fpath, 'r')
        return f, 'h5py'
    except:
        data = sio.loadmat(fpath, squeeze_me=False)
        return data, 'scipy'

def read_h5_string(f, ref):
    data = f[ref][()].flatten()
    return ''.join(chr(c) for c in data).strip()

# ============================================================
# CLUSTER EXTRACTION
# ============================================================
def get_clusters_h5(f, obj, probe_nums):
    clu_dataset = obj['clu']
    clusters = []
    for probe_num in probe_nums:
        probe_idx = probe_num - 1
        if probe_idx >= clu_dataset.shape[0]:
            continue
        clu_ref = clu_dataset[probe_idx, 0]
        clu = f[clu_ref]
        if isinstance(clu, h5py.Group):
            quality_ds = clu['quality']
            n_clu = quality_ds.shape[0]
            for i in range(n_clu):
                q_str = read_h5_string(f, quality_ds[i, 0]).lower()
                if q_str in EXCLUDED_QUALITIES or q_str == '' or '\x00' in q_str:
                    continue
                trialtm = f[clu['trialtm'][i, 0]][()].flatten()
                trial = f[clu['trial'][i, 0]][()].flatten().astype(int)
                clusters.append({'quality': q_str, 'trialtm': trialtm, 'trial': trial, 'probe': probe_num})
    return clusters

def get_clusters_scipy(data, probe_nums):
    obj = data['obj'][0, 0]
    clu_cell = obj['clu']
    clusters = []
    for probe_num in probe_nums:
        probe_idx = probe_num - 1
        if probe_idx >= clu_cell.shape[1]:
            continue
        probe_clu = clu_cell[0, probe_idx]
        if not hasattr(probe_clu, 'dtype') or probe_clu.dtype.names is None:
            continue
        if 'quality' not in probe_clu.dtype.names:
            continue
        n_clu = probe_clu.shape[1] if len(probe_clu.shape) > 1 else probe_clu.shape[0]
        for i in range(n_clu):
            entry = probe_clu[0, i] if len(probe_clu.shape) > 1 else probe_clu[i]
            q_val = entry['quality'].flatten()
            q_str = str(q_val[0]).strip().lower() if len(q_val) > 0 else ''
            if q_str in EXCLUDED_QUALITIES or q_str == '' or '\x00' in q_str:
                continue
            trialtm = entry['trialtm'].flatten().astype(float)
            trial = entry['trial'].flatten().astype(int)
            clusters.append({'quality': q_str, 'trialtm': trialtm, 'trial': trial, 'probe': probe_num})
    return clusters

# ============================================================
# TRIAL INFO EXTRACTION
# ============================================================
def get_trial_info_h5(f, obj):
    bp = obj['bp']
    ev = bp['ev']
    info = {
        'n_trials': int(bp['Ntrials'][()].flatten()[0]),
        'hit': bp['hit'][()].flatten().astype(bool),
        'miss': bp['miss'][()].flatten().astype(bool),
        'R': bp['R'][()].flatten().astype(bool),
        'L': bp['L'][()].flatten().astype(bool),
        'autowater': bp['autowater'][()].flatten().astype(bool),
        'early': bp['early'][()].flatten().astype(bool),
        'goCue': ev['goCue'][()].flatten(),
        'sample': ev['sample'][()].flatten(),
        'delay': ev['delay'][()].flatten(),
    }
    if 'stim' in bp:
        stim = bp['stim']
        info['stim_enable'] = stim['enable'][()].flatten().astype(bool) if 'enable' in stim else np.zeros(info['n_trials'], dtype=bool)
    else:
        info['stim_enable'] = np.zeros(info['n_trials'], dtype=bool)
    
    # Video offset
    sglx = obj['sglx']
    fs = sglx['fs'][()].flatten()[0]
    bitstart_sglx = sglx['bitcode']['bitstart'][()].flatten()
    bitStart_bp = ev['bitStart'][()].flatten()
    info['vidshift'] = sp_stats.mode(bitstart_sglx, keepdims=False).mode / fs - sp_stats.mode(bitStart_bp, keepdims=False).mode
    return info

def get_trial_info_scipy(data):
    obj = data['obj'][0, 0]
    bp = obj['bp'][0, 0]
    ev = bp['ev'][0, 0]
    info = {
        'n_trials': int(bp['Ntrials'].flatten()[0]),
        'hit': bp['hit'].flatten().astype(bool),
        'miss': bp['miss'].flatten().astype(bool),
        'R': bp['R'].flatten().astype(bool),
        'L': bp['L'].flatten().astype(bool),
        'autowater': bp['autowater'].flatten().astype(bool),
        'early': bp['early'].flatten().astype(bool),
        'goCue': ev['goCue'].flatten(),
        'sample': ev['sample'].flatten(),
        'delay': ev['delay'].flatten(),
    }
    if 'stim' in bp.dtype.names:
        stim = bp['stim'][0, 0]
        info['stim_enable'] = stim['enable'].flatten().astype(bool) if 'enable' in stim.dtype.names else np.zeros(info['n_trials'], dtype=bool)
    else:
        info['stim_enable'] = np.zeros(info['n_trials'], dtype=bool)
    
    sglx = obj['sglx'][0, 0]
    fs = sglx['fs'].flatten()[0]
    bitcode = sglx['bitcode'][0, 0]
    bitstart_sglx = bitcode['bitstart'].flatten()
    bitStart_bp = ev['bitStart'].flatten()
    info['vidshift'] = sp_stats.mode(bitstart_sglx, keepdims=False).mode / fs - sp_stats.mode(bitStart_bp, keepdims=False).mode
    return info

# ============================================================
# TRAJECTORY (DLC) EXTRACTION
# ============================================================
def get_traj_trial_h5(f, obj, trial_idx, view_idx):
    """Get DLC data for one trial. Returns frameTimes, ts (n_feat, 3, n_frames), feat_names."""
    view_ref = obj['traj'][view_idx, 0]
    view_data = f[view_ref]
    ft = f[view_data['frameTimes'][trial_idx, 0]][()].flatten()
    ts = f[view_data['ts'][trial_idx, 0]][()]  # (n_feat, 3, n_frames)
    feat_ref = view_data['featNames'][0, 0]
    feat_data = f[feat_ref]
    feat_names = [read_h5_string(f, feat_data[0, i]) for i in range(feat_data.shape[1])]
    return ft, ts, feat_names

def get_traj_trial_scipy(data, trial_idx, view_idx):
    """Get DLC data for one trial. Returns frameTimes, ts (n_feat, 3, n_frames), feat_names."""
    obj = data['obj'][0, 0]
    traj = obj['traj']
    # scipy: traj shape is (1, 2) -> traj[0, view_idx]
    view = traj[0, view_idx]
    trial = view[0, trial_idx]
    ft = trial['frameTimes'].flatten()
    ts_raw = trial['ts']  # (n_frames, 3, n_feat) in scipy
    # Transpose to (n_feat, 3, n_frames) to match h5py convention
    ts = ts_raw.transpose(2, 1, 0)
    # Feature names
    feat_names_raw = trial['featNames']
    feat_names = []
    if feat_names_raw.shape[0] >= feat_names_raw.shape[1]:
        for i in range(feat_names_raw.shape[0]):
            feat_names.append(str(feat_names_raw[i, 0].flatten()[0]))
    else:
        for i in range(feat_names_raw.shape[1]):
            feat_names.append(str(feat_names_raw[0, i].flatten()[0]))
    return ft, ts, feat_names

# ============================================================
# MOTION ENERGY LOADING
# ============================================================
def load_motion_energy(anm, date, data_dir):
    """Load motion energy. Returns (list of per-trial arrays, moveThresh) or (None, None)."""
    me_path = os.path.join(data_dir, f'motionEnergy_{anm}_{date}.mat')
    if not os.path.exists(me_path):
        return None, None
    try:
        me_data = sio.loadmat(me_path, squeeze_me=False)
        me_raw = me_data['me']
        
        # Case 1: struct with 'data' and 'moveThresh'
        if me_raw.dtype.names and 'data' in me_raw.dtype.names:
            me_s = me_raw[0, 0]
            d = me_s['data']
            thresh = float(me_s['moveThresh'].flatten()[0]) if 'moveThresh' in me_raw.dtype.names else None
            
            # Handle nested struct: if d is (1,1) with dtype that has 'data'
            if d.shape == (1, 1) and d.dtype.names and 'data' in d.dtype.names:
                d = d[0, 0]['data']
            elif d.shape == (1, 1) and d.dtype == object:
                inner = d[0, 0]
                if inner.dtype == object:
                    d = inner  # It's a cell array wrapped in another cell
                elif inner.dtype.names and 'data' in inner.dtype.names:
                    d = inner['data']
            
            # Now d should be a cell array (n_trials, 1) of arrays
            trial_data = []
            if d.dtype == object:
                n = max(d.shape)
                for t in range(n):
                    idx = (t, 0) if d.shape[0] > d.shape[1] else (0, t)
                    trial_data.append(d[idx].flatten())
            return trial_data, thresh
        
        # Case 2: cell array directly (no struct wrapper)
        elif me_raw.dtype == object:
            n = max(me_raw.shape)
            trial_data = []
            for t in range(n):
                idx = (t, 0) if me_raw.shape[0] > me_raw.shape[1] else (0, t)
                trial_data.append(me_raw[idx].flatten())
            return trial_data, None  # No threshold available
        
        return None, None
    except Exception as e:
        print(f"    WARNING: Failed to load ME for {anm}_{date}: {e}")
        return None, None

# ============================================================
# VELOCITY COMPUTATION
# ============================================================
def compute_velocity(x, y, ft):
    """Compute speed from x,y positions and frame times."""
    dt = np.diff(ft)
    dt[dt == 0] = 1e-6  # avoid division by zero
    dx = np.diff(x)
    dy = np.diff(y)
    speed = np.sqrt(dx**2 + dy**2) / dt
    return np.concatenate([[speed[0]], speed])

# ============================================================
# MAIN PROCESSING
# ============================================================
def process_session(anm, date, probe_nums, data_dir, show_processing=False):
    t0 = time.time()
    fpath = os.path.join(data_dir, f'data_structure_{anm}_{date}.mat')
    print(f"  Loading {anm}_{date}...")
    
    file_data, fmt = load_mat_file(fpath)
    if fmt == 'h5py':
        obj = file_data['obj']
        trial_info = get_trial_info_h5(file_data, obj)
    else:
        trial_info = get_trial_info_scipy(file_data)
    
    n_trials = trial_info['n_trials']
    print(f"    Trials: {n_trials}, Format: {fmt}")
    
    # Extract clusters
    if fmt == 'h5py':
        clusters = get_clusters_h5(file_data, obj, probe_nums)
    else:
        clusters = get_clusters_scipy(file_data, probe_nums)
    print(f"    Clusters after quality filter: {len(clusters)}")
    
    if len(clusters) == 0:
        print(f"    WARNING: No clusters, skipping")
        if fmt == 'h5py': file_data.close()
        return None
    
    # Bin and smooth spikes
    goCue = trial_info['goCue']
    n_neurons = len(clusters)
    trialdat = np.zeros((n_time, n_neurons, n_trials), dtype=np.float32)
    
    for ci, clu in enumerate(clusters):
        for trial_num in range(1, n_trials + 1):
            spk_mask = clu['trial'] == trial_num
            if not np.any(spk_mask):
                continue
            aligned_times = clu['trialtm'][spk_mask] - goCue[trial_num - 1]
            N, _ = np.histogram(aligned_times, bins=edges)
            fr = N.astype(np.float64) / DT
            trialdat[:, ci, trial_num - 1] = causal_gaussian_smooth(fr, SMOOTH_WINDOW, BC_TYPE).astype(np.float32)
    
    # Remove low FR clusters
    mean_fr = trialdat.mean(axis=(0, 2))
    keep_mask = mean_fr > LOW_FR_THRESHOLD
    n_kept = int(keep_mask.sum())
    print(f"    After FR filter (>{LOW_FR_THRESHOLD} Hz): {n_kept}/{n_neurons}")
    
    if n_kept < 10:
        print(f"    WARNING: Too few neurons ({n_kept}), skipping")
        if fmt == 'h5py': file_data.close()
        return None
    
    trialdat = trialdat[:, keep_mask, :]
    n_neurons = n_kept
    
    # Trial labels
    hit = trial_info['hit']
    miss = trial_info['miss']
    R = trial_info['R']
    L = trial_info['L']
    autowater = trial_info['autowater']
    early = trial_info['early']
    
    lick_direction = np.full(n_trials, 2, dtype=int)  # 2=none
    lick_direction[hit & R] = 1  # right
    lick_direction[hit & L] = 0  # left
    lick_direction[miss & R] = 0  # licked left (wrong on right trial)
    lick_direction[miss & L] = 1  # licked right (wrong on left trial)
    # Early lick trials: override to 'none' regardless of hit/miss
    lick_direction[early] = 2
    
    context = autowater.astype(int)  # 0=DR, 1=WC
    
    outcome = np.full(n_trials, 2, dtype=int)  # 2=ignore
    outcome[hit] = 1  # correct
    outcome[miss] = 0  # incorrect
    # Early lick trials: override to 'ignore' regardless of hit/miss
    outcome[early] = 2
    
    # Load motion energy
    me_trials, me_thresh = load_motion_energy(anm, date, data_dir)
    
    # Process DLC tracking and motion energy per trial
    vidshift = trial_info['vidshift']
    tongue_vel_all = np.full((n_trials, n_time), np.nan, dtype=np.float32)
    paw_vel_all = np.full((n_trials, n_time), np.nan, dtype=np.float32)
    me_all = np.full((n_trials, n_time), np.nan, dtype=np.float32)
    
    for t in range(n_trials):
        try:
            # Get DLC tracking
            if fmt == 'h5py':
                ft_side, ts_side, feat_side = get_traj_trial_h5(file_data, obj, t, 0)
                ft_bot, ts_bot, feat_bot = get_traj_trial_h5(file_data, obj, t, 1)
            else:
                ft_side, ts_side, feat_side = get_traj_trial_scipy(file_data, t, 0)
                ft_bot, ts_bot, feat_bot = get_traj_trial_scipy(file_data, t, 1)
            
            ft_side_aligned = ft_side - vidshift - goCue[t]
            ft_bot_aligned = ft_bot - vidshift - goCue[t]
            
            # Tongue velocity (side cam)
            if 'tongue' in feat_side:
                ti = feat_side.index('tongue')
                tx, ty, tc = ts_side[ti, 0, :], ts_side[ti, 1, :], ts_side[ti, 2, :]
                tongue_visible = tc >= DLC_CONFIDENCE_THRESHOLD
                tspeed = compute_velocity(tx, ty, ft_side)
                tspeed[~tongue_visible] = np.nan
                tv_interp = np.interp(time_axis, ft_side_aligned, tspeed)
                tv_vis = np.interp(time_axis, ft_side_aligned, tongue_visible.astype(float)) >= 0.5
                tv_interp[~tv_vis] = np.nan
                tongue_vel_all[t] = tv_interp
            
            # Paw velocity (bottom cam)
            if 'top_paw' in feat_bot:
                pi = feat_bot.index('top_paw')
                px, py, pc = ts_bot[pi, 0, :], ts_bot[pi, 1, :], ts_bot[pi, 2, :]
                paw_visible = pc >= DLC_CONFIDENCE_THRESHOLD
                pspeed = compute_velocity(px, py, ft_bot)
                pspeed[~paw_visible] = np.nan
                pv_interp = np.interp(time_axis, ft_bot_aligned, pspeed)
                pv_vis = np.interp(time_axis, ft_bot_aligned, paw_visible.astype(float)) >= 0.5
                pv_interp[~pv_vis] = np.nan
                paw_vel_all[t] = pv_interp
            
            # Motion energy
            if me_trials is not None and t < len(me_trials) and len(me_trials[t]) > 0:
                me_trial = me_trials[t].astype(float)
                # ME is at same frame rate as side cam
                if len(me_trial) == len(ft_side):
                    me_interp = np.interp(time_axis, ft_side_aligned, me_trial)
                elif len(me_trial) > 0:
                    # Try to construct frame times from ME length
                    me_ft = np.linspace(ft_side[0], ft_side[-1], len(me_trial))
                    me_ft_aligned = me_ft - vidshift - goCue[t]
                    me_interp = np.interp(time_axis, me_ft_aligned, me_trial)
                else:
                    me_interp = np.full(n_time, np.nan)
                me_all[t] = me_interp
                
        except Exception as e:
            pass  # Leave as NaN
    
    # Discretize tongue velocity
    tongue_valid = ~np.isnan(tongue_vel_all)
    tongue_thresh = np.nanpercentile(tongue_vel_all[tongue_valid], 50) if tongue_valid.any() else 0
    tongue_disc = np.full(tongue_vel_all.shape, 2, dtype=int)
    tongue_disc[tongue_valid & (tongue_vel_all < tongue_thresh)] = 0
    tongue_disc[tongue_valid & (tongue_vel_all >= tongue_thresh)] = 1
    
    # Discretize paw velocity
    paw_valid = ~np.isnan(paw_vel_all)
    paw_thresh = np.nanpercentile(paw_vel_all[paw_valid], 50) if paw_valid.any() else 0
    paw_disc = np.full(paw_vel_all.shape, 2, dtype=int)
    paw_disc[paw_valid & (paw_vel_all < paw_thresh)] = 0
    paw_disc[paw_valid & (paw_vel_all >= paw_thresh)] = 1
    
    # Discretize motion energy
    me_valid = ~np.isnan(me_all)
    me_thresh_disc = np.nanpercentile(me_all[me_valid], 50) if me_valid.any() else 0
    me_disc = np.full(me_all.shape, 2, dtype=int)
    me_disc[me_valid & (me_all < me_thresh_disc)] = 0
    me_disc[me_valid & (me_all >= me_thresh_disc)] = 1
    
    # Build per-trial data
    neural_trials = []
    input_trials = []
    output_trials = []
    
    time_input = time_axis.astype(np.float32).reshape(1, -1)
    
    for t in range(n_trials):
        neural_trials.append(trialdat[:, :, t].T.astype(np.float32))
        input_trials.append(time_input.copy())
        output = np.zeros((6, n_time), dtype=int)
        output[0, :] = lick_direction[t]
        output[1, :] = context[t]
        output[2, :] = outcome[t]
        output[3, :] = tongue_disc[t]
        output[4, :] = paw_disc[t]
        output[5, :] = me_disc[t]
        output_trials.append(output)
    
    if fmt == 'h5py':
        file_data.close()
    
    elapsed = time.time() - t0
    print(f"    Processed in {elapsed:.1f}s")
    
    return {
        'neural': neural_trials, 'input': input_trials, 'output': output_trials,
        'anm': anm, 'date': date, 'n_neurons': n_neurons, 'n_trials': n_trials,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('outfile')
    parser.add_argument('--sample', action='store_true')
    parser.add_argument('--full', action='store_true')
    parser.add_argument('--show-processing', action='store_true')
    args = parser.parse_args()
    
    t_start = time.time()
    all_sessions = [(a, d, p, EPHYS_DATA_DIR) for a, d, p in EPHYS_SESSIONS] + \
                   [(a, d, p, RD_DATA_DIR) for a, d, p in RD_SESSIONS]
    
    if args.sample:
        all_sessions = [all_sessions[0], all_sessions[len(EPHYS_SESSIONS)]]
        print(f"SAMPLE MODE: {len(all_sessions)} sessions")
    else:
        print(f"FULL MODE: {len(all_sessions)} sessions")
    
    neural_all, input_all, output_all = [], [], []
    subjects, subject_idx, session_info = [], [], []
    
    for si, (anm, date, probes, ddir) in enumerate(all_sessions):
        print(f"\nSession {si+1}/{len(all_sessions)}: {anm}_{date}")
        result = process_session(anm, date, probes, ddir, args.show_processing)
        if result is None:
            continue
        neural_all.append(result['neural'])
        input_all.append(result['input'])
        output_all.append(result['output'])
        if anm not in subjects:
            subjects.append(anm)
        subject_idx.append(subjects.index(anm))
        session_info.append(f"{anm}_{date}")
    
    subject_idx = np.array(subject_idx)
    brain_regions = ['ALM']
    brain_region_idx = [np.zeros(s[0].shape[0], dtype=int) for s in neural_all if len(s) > 0]
    
    data = {
        'neural': neural_all, 'input': input_all, 'output': output_all,
        'subjects': subjects, 'subject_idx': subject_idx,
        'brain_regions': brain_regions, 'brain_region_idx': brain_region_idx,
        'input_names': ['time_from_go_cue'],
        'output_names': ['lick_direction', 'context', 'outcome', 'tongue_velocity', 'paw_velocity', 'motion_energy'],
        'output_values': [
            ['left', 'right', 'none'], ['DR', 'WC'],
            ['incorrect', 'correct', 'ignore'],
            ['low', 'high', 'not_visible'], ['low', 'high', 'not_visible'],
            ['low', 'high', 'no_video'],
        ],
        'metadata': {
            'task_description': 'Two-context delayed response and water-cued licking task with ALM recordings',
            'time_bin_size': DT * 1000, 'temporal_alignment_event': 'Go cue onset',
            'off_start': TMIN, 'off_end': TMAX, 'session_info': session_info,
            'smooth_window_ms': SMOOTH_WINDOW, 'smooth_type': 'causal_gaussian',
            'low_fr_threshold_hz': LOW_FR_THRESHOLD,
        }
    }
    
    total_neurons = sum(s[0].shape[0] for s in neural_all if len(s) > 0)
    total_trials = sum(len(s) for s in neural_all)
    print(f"\n{'='*60}")
    print(f"Sessions: {len(neural_all)}, Subjects: {len(subjects)}")
    print(f"Total neurons: {total_neurons}, Total trials: {total_trials}")
    print(f"Neurons/session: {[s[0].shape[0] for s in neural_all if len(s) > 0]}")
    
    print(f"\nSaving to {args.outfile}...")
    with open(args.outfile, 'wb') as f:
        pickle.dump(data, f, protocol=4)
    print(f"Saved ({os.path.getsize(args.outfile)/1e6:.1f} MB)")
    print(f"Total time: {time.time()-t_start:.1f}s")

if __name__ == '__main__':
    main()
