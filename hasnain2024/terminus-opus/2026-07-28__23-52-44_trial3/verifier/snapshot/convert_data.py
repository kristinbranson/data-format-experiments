#!/usr/bin/env python3
"""Convert Hasnain et al. 2024 data to decoder-compatible format.

Usage:
    python -u convert_data.py <output_pickle_file> [--sample] [--full] [--show-processing]
"""

import sys
import os
import argparse
import pickle
import time
import numpy as np
import h5py
import scipy.io as sio
from scipy.signal import windows as sig_windows
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Session metadata: which sessions to include and which probe
# Based on the loading scripts in code/DataLoadingScripts/Recording and video/
# ============================================================

SESSION_META = [
    # Standard DR task sessions (Ephys_Behavior)
    {'anm': 'EKH1', 'date': '2021-08-07', 'probe': 2, 'dir': 'Ephys_Behavior', 'task': 'DR'},
    {'anm': 'EKH3', 'date': '2021-08-11', 'probe': 2, 'dir': 'Ephys_Behavior', 'task': 'DR'},
    {'anm': 'JEB6', 'date': '2021-04-18', 'probe': 2, 'dir': 'Ephys_Behavior', 'task': 'DR'},
    {'anm': 'JEB7', 'date': '2021-04-29', 'probe': 1, 'dir': 'Ephys_Behavior', 'task': 'DR'},
    {'anm': 'JEB7', 'date': '2021-04-30', 'probe': 1, 'dir': 'Ephys_Behavior', 'task': 'DR'},
    {'anm': 'JGR2', 'date': '2021-11-16', 'probe': 1, 'dir': 'Ephys_Behavior', 'task': 'DR'},
    {'anm': 'JGR2', 'date': '2021-11-17', 'probe': 1, 'dir': 'Ephys_Behavior', 'task': 'DR'},
    {'anm': 'JGR3', 'date': '2021-11-18', 'probe': 1, 'dir': 'Ephys_Behavior', 'task': 'DR'},
    {'anm': 'JEB13', 'date': '2022-09-13', 'probe': 1, 'dir': 'Ephys_Behavior', 'task': 'DR'},
    {'anm': 'JEB13', 'date': '2022-09-14', 'probe': 1, 'dir': 'Ephys_Behavior', 'task': 'DR'},
    {'anm': 'JEB13', 'date': '2022-09-21', 'probe': 1, 'dir': 'Ephys_Behavior', 'task': 'DR'},
    {'anm': 'JEB13', 'date': '2022-09-24', 'probe': 1, 'dir': 'Ephys_Behavior', 'task': 'DR'},
    {'anm': 'JEB13', 'date': '2022-09-25', 'probe': 1, 'dir': 'Ephys_Behavior', 'task': 'DR'},
    {'anm': 'JEB14', 'date': '2022-08-22', 'probe': 1, 'dir': 'Ephys_Behavior', 'task': 'DR'},
    {'anm': 'JEB14', 'date': '2022-08-23', 'probe': 1, 'dir': 'Ephys_Behavior', 'task': 'DR'},
    {'anm': 'JEB14', 'date': '2022-08-24', 'probe': 1, 'dir': 'Ephys_Behavior', 'task': 'DR'},
    {'anm': 'JEB14', 'date': '2022-08-25', 'probe': 1, 'dir': 'Ephys_Behavior', 'task': 'DR'},
    {'anm': 'JEB15', 'date': '2022-07-26', 'probe': 1, 'dir': 'Ephys_Behavior', 'task': 'DR'},
    {'anm': 'JEB15', 'date': '2022-07-27', 'probe': 1, 'dir': 'Ephys_Behavior', 'task': 'DR'},
    {'anm': 'JEB15', 'date': '2022-07-28', 'probe': 1, 'dir': 'Ephys_Behavior', 'task': 'DR'},
    {'anm': 'JEB15', 'date': '2022-07-29', 'probe': 1, 'dir': 'Ephys_Behavior', 'task': 'DR'},
    {'anm': 'JEB19', 'date': '2023-04-18', 'probe': 1, 'dir': 'Ephys_Behavior', 'task': 'DR'},
    {'anm': 'JEB19', 'date': '2023-04-19', 'probe': 1, 'dir': 'Ephys_Behavior', 'task': 'DR'},
    {'anm': 'JEB19', 'date': '2023-04-20', 'probe': 1, 'dir': 'Ephys_Behavior', 'task': 'DR'},
    {'anm': 'JEB19', 'date': '2023-04-21', 'probe': 1, 'dir': 'Ephys_Behavior', 'task': 'DR'},
    # Randomized delay sessions (RandomizedDelay_Ephys_Behavior)
    {'anm': 'JEB11', 'date': '2022-05-10', 'probe': 1, 'dir': 'RandomizedDelay_Ephys_Behavior', 'task': 'RandDelay'},
    {'anm': 'JEB11', 'date': '2022-05-11', 'probe': 1, 'dir': 'RandomizedDelay_Ephys_Behavior', 'task': 'RandDelay'},
    {'anm': 'JEB12', 'date': '2022-05-12', 'probe': 1, 'dir': 'RandomizedDelay_Ephys_Behavior', 'task': 'RandDelay'},
    {'anm': 'JEB12', 'date': '2022-05-13', 'probe': 1, 'dir': 'RandomizedDelay_Ephys_Behavior', 'task': 'RandDelay'},
    {'anm': 'JEB23', 'date': '2023-10-10', 'probe': 1, 'dir': 'RandomizedDelay_Ephys_Behavior', 'task': 'RandDelay'},
    {'anm': 'JEB23', 'date': '2023-10-11', 'probe': 1, 'dir': 'RandomizedDelay_Ephys_Behavior', 'task': 'RandDelay'},
    {'anm': 'JEB23', 'date': '2023-10-12', 'probe': 1, 'dir': 'RandomizedDelay_Ephys_Behavior', 'task': 'RandDelay'},
    {'anm': 'JEB23', 'date': '2023-10-13', 'probe': 1, 'dir': 'RandomizedDelay_Ephys_Behavior', 'task': 'RandDelay'},
    {'anm': 'JEB23', 'date': '2023-10-18', 'probe': 1, 'dir': 'RandomizedDelay_Ephys_Behavior', 'task': 'RandDelay'},
    {'anm': 'JEB23', 'date': '2023-10-19', 'probe': 1, 'dir': 'RandomizedDelay_Ephys_Behavior', 'task': 'RandDelay'},
    # JEB23 2023-10-20 excluded (commented out in loading script)
    {'anm': 'JEB23', 'date': '2023-10-21', 'probe': 1, 'dir': 'RandomizedDelay_Ephys_Behavior', 'task': 'RandDelay'},
    {'anm': 'JEB24', 'date': '2023-10-23', 'probe': 1, 'dir': 'RandomizedDelay_Ephys_Behavior', 'task': 'RandDelay'},
    {'anm': 'JEB24', 'date': '2023-10-24', 'probe': 1, 'dir': 'RandomizedDelay_Ephys_Behavior', 'task': 'RandDelay'},
    {'anm': 'JEB24', 'date': '2023-10-25', 'probe': 1, 'dir': 'RandomizedDelay_Ephys_Behavior', 'task': 'RandDelay'},
    {'anm': 'JEB24', 'date': '2023-10-26', 'probe': 1, 'dir': 'RandomizedDelay_Ephys_Behavior', 'task': 'RandDelay'},
    {'anm': 'JEB24', 'date': '2023-10-27', 'probe': 1, 'dir': 'RandomizedDelay_Ephys_Behavior', 'task': 'RandDelay'},
    {'anm': 'JEB24', 'date': '2023-10-31', 'probe': 1, 'dir': 'RandomizedDelay_Ephys_Behavior', 'task': 'RandDelay'},
    {'anm': 'JEB24', 'date': '2023-11-02', 'probe': 1, 'dir': 'RandomizedDelay_Ephys_Behavior', 'task': 'RandDelay'},
    {'anm': 'JEB24', 'date': '2023-11-03', 'probe': 1, 'dir': 'RandomizedDelay_Ephys_Behavior', 'task': 'RandDelay'},
]

# Processing parameters (from getDefaultParams.m)
PARAMS = {
    'tmin': -2.5,
    'tmax': 2.5,
    'dt': 1.0 / 200.0,  # 5ms bins
    'smooth_window': 15,  # causal Gaussian kernel window
    'low_fr': 0.5,  # Hz, minimum firing rate
    'min_units': 10,  # minimum units per session
    'align_event': 'goCue',
    'excluded_qualities': {'garbage', 'gabrga', 'noisy', 'real?'},
}


def make_causal_gaussian_kernel(N):
    """Create causal Gaussian kernel matching mySmooth.m."""
    if N <= 1:
        return np.array([1.0])
    kern = np.array(sig_windows.gaussian(N, std=np.std(np.arange(N))))
    kern[:N // 2] = 0  # causal: zero out first half
    kern = kern / kern.sum()
    return kern


def smooth_causal(x, N, bctype='none'):
    """Smooth array x along axis 0 with causal Gaussian kernel.
    Matches mySmooth.m behavior.
    """
    if N <= 1:
        return x.copy()
    kern = make_causal_gaussian_kernel(N)
    
    if x.ndim == 1:
        if bctype == 'reflect':
            x_filt = np.concatenate([x[:N], x])
            trim = N
        elif bctype == 'zeropad':
            x_filt = np.concatenate([np.zeros(N), x])
            trim = N
        else:  # 'none'
            x_filt = x
            trim = 0
        out = np.convolve(x_filt, kern, mode='same')
        return out[trim:]
    else:
        # 2D: smooth along axis 0
        if bctype == 'reflect':
            x_filt = np.concatenate([x[:N, :], x], axis=0)
            trim = N
        elif bctype == 'zeropad':
            x_filt = np.concatenate([np.zeros((N, x.shape[1])), x], axis=0)
            trim = N
        else:
            x_filt = x
            trim = 0
        out = np.zeros_like(x_filt)
        for j in range(x_filt.shape[1]):
            out[:, j] = np.convolve(x_filt[:, j], kern, mode='same')
        return out[trim:]


def read_h5_string(f, ref):
    """Read a string from an HDF5 object reference."""
    data = f[ref][()]
    if data.dtype.kind in ['U', 'S']:
        return str(data)
    else:
        return ''.join(chr(c) for c in data.flatten())


def load_session_h5(filepath, probe_num):
    """Load session data from HDF5 (MATLAB v7.3) file."""
    data = {}
    with h5py.File(filepath, 'r') as f:
        obj = f['obj']
        
        # --- Behavioral data ---
        bp = obj['bp']
        ntrials = int(bp['Ntrials'][0, 0])
        data['ntrials'] = ntrials
        data['hit'] = bp['hit'][0, :].astype(bool)
        data['miss'] = bp['miss'][0, :].astype(bool)
        data['no'] = bp['no'][0, :].astype(bool)
        data['early'] = bp['early'][0, :].astype(bool)
        data['R'] = bp['R'][0, :].astype(bool)
        data['L'] = bp['L'][0, :].astype(bool)
        data['autowater'] = bp['autowater'][0, :]
        
        # Stim enable
        stim = bp['stim']
        if 'enable' in stim:
            stim_enable = stim['enable'][0, :]
            data['stim_enable'] = stim_enable.astype(bool)
        else:
            data['stim_enable'] = np.zeros(ntrials, dtype=bool)
        
        # Event times
        ev = bp['ev']
        data['goCue'] = ev['goCue'][0, :]
        data['sample'] = ev['sample'][0, :]
        data['delay'] = ev['delay'][0, :]
        data['bitStart'] = ev['bitStart'][0, :]
        
        # Lick times (cell arrays of object refs)
        data['lickL'] = []
        data['lickR'] = []
        lickL_refs = ev['lickL'][0, :]
        lickR_refs = ev['lickR'][0, :]
        for i in range(ntrials):
            try:
                ll = f[lickL_refs[i]][()].flatten()
                data['lickL'].append(ll)
            except:
                data['lickL'].append(np.array([]))
            try:
                lr = f[lickR_refs[i]][()].flatten()
                data['lickR'].append(lr)
            except:
                data['lickR'].append(np.array([]))
        
        # --- Cluster data ---
        clu_ds = obj['clu']
        probe_idx = probe_num - 1  # Convert 1-indexed to 0-indexed
        
        if clu_ds.shape == (1, 1):
            # Single probe
            clu_ref = clu_ds[0, 0]
            clu_data = f[clu_ref]
        else:
            # Multiple probes - select the right one
            clu_ref = clu_ds[probe_idx, 0]
            clu_data = f[clu_ref]
        
        if isinstance(clu_data, h5py.Dataset):
            # Empty probe or different format
            data['clusters'] = []
            return data
        
        n_clusters = clu_data['quality'].shape[0]
        clusters = []
        for i in range(n_clusters):
            clu_info = {}
            # Quality
            ref = clu_data['quality'][i, 0]
            clu_info['quality'] = read_h5_string(f, ref).strip()
            
            # Spike times in trial
            ref = clu_data['trialtm'][i, 0]
            clu_info['trialtm'] = f[ref][()].flatten()
            
            # Trial numbers
            ref = clu_data['trial'][i, 0]
            clu_info['trial'] = f[ref][()].flatten().astype(int)
            
            clusters.append(clu_info)
        
        data['clusters'] = clusters
        
        # --- Trajectory data (DLC) ---
        traj_ds = obj['traj']
        traj_data = []
        # traj can be (2,1) or different shapes
        n_cams = traj_ds.shape[0]
        for cam_idx in range(n_cams):
            cam_ref = traj_ds[cam_idx, 0]
            cam = f[cam_ref]
            if isinstance(cam, h5py.Group):
                cam_trials = []
                n_traj_trials = cam['ts'].shape[0]
                
                # Get feature names from first trial
                feat_ref = cam['featNames'][0, 0]
                feat_data = f[feat_ref]
                feat_names = []
                if isinstance(feat_data, h5py.Dataset):
                    for fi in range(feat_data.shape[1] if feat_data.ndim > 1 else feat_data.shape[0]):
                        idx = (0, fi) if feat_data.ndim > 1 else (fi,)
                        name_ref = feat_data[idx]
                        feat_names.append(read_h5_string(f, name_ref))
                
                for trial_idx in range(n_traj_trials):
                    trial_info = {}
                    # ts data
                    ts_ref = cam['ts'][trial_idx, 0]
                    trial_info['ts'] = f[ts_ref][()]
                    
                    # frameTimes
                    ft_ref = cam['frameTimes'][trial_idx, 0]
                    ft_data = f[ft_ref][()]
                    trial_info['frameTimes'] = ft_data.flatten()
                    
                    # NdroppedFrames
                    ndf_ref = cam['NdroppedFrames'][trial_idx, 0]
                    try:
                        ndf_data = f[ndf_ref][()]
                        trial_info['NdroppedFrames'] = ndf_data.flatten()[0] if ndf_data.size > 0 else np.nan
                    except:
                        trial_info['NdroppedFrames'] = np.nan
                    
                    cam_trials.append(trial_info)
                
                traj_data.append({'trials': cam_trials, 'featNames': feat_names})
            else:
                traj_data.append(None)
        
        data['traj'] = traj_data
        
        # --- SpikeGLX data for video offset ---
        sglx = obj['sglx']
        data['sglx_fs'] = float(sglx['fs'][0, 0])
        bitcode = sglx['bitcode']
        data['sglx_bitstart'] = bitcode['bitstart'][0, :]
    
    return data


def load_session_v5(filepath, probe_num):
    """Load session data from MATLAB v5 file."""
    d = sio.loadmat(filepath, squeeze_me=False)
    obj = d['obj'][0, 0]
    data = {}
    
    # --- Behavioral data ---
    bp = obj['bp'][0, 0]
    ntrials = int(bp['Ntrials'][0, 0])
    data['ntrials'] = ntrials
    data['hit'] = bp['hit'].flatten().astype(bool)
    data['miss'] = bp['miss'].flatten().astype(bool)
    data['no'] = bp['no'].flatten().astype(bool)
    data['early'] = bp['early'].flatten().astype(bool)
    data['R'] = bp['R'].flatten().astype(bool)
    data['L'] = bp['L'].flatten().astype(bool)
    data['autowater'] = bp['autowater'].flatten()
    
    # Stim enable
    stim = bp['stim'][0, 0]
    if 'enable' in stim.dtype.names:
        data['stim_enable'] = stim['enable'].flatten().astype(bool)
    else:
        data['stim_enable'] = np.zeros(ntrials, dtype=bool)
    
    # Event times
    ev = bp['ev'][0, 0]
    data['goCue'] = ev['goCue'].flatten()
    data['sample'] = ev['sample'].flatten()
    data['delay'] = ev['delay'].flatten()
    data['bitStart'] = ev['bitStart'].flatten()
    
    # Lick times
    data['lickL'] = []
    data['lickR'] = []
    lickL = ev['lickL'].flatten()
    lickR = ev['lickR'].flatten()
    for i in range(ntrials):
        try:
            ll = lickL[i].flatten()
            data['lickL'].append(ll)
        except:
            data['lickL'].append(np.array([]))
        try:
            lr = lickR[i].flatten()
            data['lickR'].append(lr)
        except:
            data['lickR'].append(np.array([]))
    
    # --- Cluster data ---
    clu = obj['clu']
    probe_idx = probe_num - 1
    
    if clu.dtype == object:
        probe_data = clu.flatten()[probe_idx]
    else:
        probe_data = clu
    
    if probe_data.size == 0:
        data['clusters'] = []
        return data
    
    # probe_data is a struct array (1, n_clusters)
    n_clusters = probe_data.shape[1] if probe_data.ndim > 1 and probe_data.shape[0] == 1 else probe_data.shape[0]
    clusters = []
    for i in range(n_clusters):
        clu_info = {}
        c = probe_data[0, i] if probe_data.ndim > 1 and probe_data.shape[0] == 1 else probe_data[i]
        clu_info['quality'] = str(c['quality'].flatten()[0]).strip() if c['quality'].size > 0 else ''
        clu_info['trialtm'] = c['trialtm'].flatten().astype(float)
        clu_info['trial'] = c['trial'].flatten().astype(int)
        clusters.append(clu_info)
    
    data['clusters'] = clusters
    
    # --- Trajectory data ---
    traj = obj['traj']
    traj_data = []
    n_cams = traj.size
    for cam_idx in range(n_cams):
        cam = traj.flatten()[cam_idx]
        if cam.size == 0:
            traj_data.append(None)
            continue
        
        cam_trials = []
        n_traj_trials = cam.shape[1] if cam.ndim > 1 and cam.shape[0] == 1 else cam.shape[0]
        
        # Get feature names from first trial
        c0 = cam[0, 0] if cam.ndim > 1 and cam.shape[0] == 1 else cam[0]
        feat_names = []
        if 'featNames' in c0.dtype.names:
            fn = c0['featNames'].flatten()
            for fi in range(len(fn)):
                feat_names.append(str(fn[fi].flatten()[0]).strip())
        
        for trial_idx in range(n_traj_trials):
            c = cam[0, trial_idx] if cam.ndim > 1 and cam.shape[0] == 1 else cam[trial_idx]
            trial_info = {}
            trial_info['ts'] = c['ts'] if 'ts' in c.dtype.names else np.array([])
            if 'frameTimes' in c.dtype.names:
                ft = c['frameTimes']
                trial_info['frameTimes'] = ft.flatten()
            else:
                trial_info['frameTimes'] = np.array([])
            if 'NdroppedFrames' in c.dtype.names:
                ndf = c['NdroppedFrames']
                trial_info['NdroppedFrames'] = ndf.flatten()[0] if ndf.size > 0 else np.nan
            else:
                trial_info['NdroppedFrames'] = np.nan
            cam_trials.append(trial_info)
        
        traj_data.append({'trials': cam_trials, 'featNames': feat_names})
    
    data['traj'] = traj_data
    
    # --- SpikeGLX data ---
    sglx = obj['sglx'][0, 0]
    data['sglx_fs'] = float(sglx['fs'][0, 0])
    bitcode = sglx['bitcode'][0, 0]
    data['sglx_bitstart'] = bitcode['bitstart'].flatten()
    
    return data


def load_session(filepath, probe_num):
    """Load session data, auto-detecting file format."""
    # Try h5py first (MATLAB v7.3), fall back to scipy.io (v5)
    try:
        import h5py
        f = h5py.File(filepath, 'r')
        f.close()
        return load_session_h5(filepath, probe_num)
    except:
        return load_session_v5(filepath, probe_num)


def load_motion_energy(me_filepath):
    """Load motion energy from separate file.
    Handles multiple formats:
    1. Standard: me.data is cell array, me.moveThresh is scalar
    2. Nested: me.data is struct with .data and .moveThresh fields
    3. Direct: me is cell array of trial data (no struct wrapper)
    """
    try:
        d = sio.loadmat(me_filepath, squeeze_me=False)
        me_raw = d['me']
        
        # Format 3: me is directly a cell array of trial data
        if me_raw.dtype == object and (me_raw.dtype.names is None):
            # Check if first element is a numeric array (not a struct)
            first = me_raw.flat[0]
            if isinstance(first, np.ndarray) and first.dtype.kind == 'f':
                trials = []
                for i in range(me_raw.shape[0]):
                    trial_me = me_raw[i, 0].flatten() if me_raw.ndim > 1 else me_raw[i].flatten()
                    trials.append(trial_me)
                # No moveThresh available, compute from data
                all_vals = np.concatenate([t for t in trials if t.size > 0])
                me_thresh = float(np.percentile(all_vals, 50)) if len(all_vals) > 0 else 0.0
                return {'data': trials, 'moveThresh': me_thresh}
        
        me = me_raw[0, 0]
        
        # Check if me.data is a struct (nested format)
        me_data_field = me['data']
        me_thresh = None
        
        if me_data_field.dtype.names is not None and 'data' in me_data_field.dtype.names:
            # Format 2: me.data is a struct with .data and .moveThresh
            inner = me_data_field[0, 0] if me_data_field.ndim > 1 else me_data_field.flat[0]
            me_data = inner['data']
            if 'moveThresh' in inner.dtype.names:
                me_thresh = float(inner['moveThresh'].flat[0])
        else:
            # Format 1: Standard format
            me_data = me_data_field
        
        if me_thresh is None:
            me_thresh = float(me['moveThresh'].flat[0])
        
        trials = []
        n_trials = me_data.shape[0]
        for i in range(n_trials):
            trial_me = me_data[i, 0].flatten()
            trials.append(trial_me)
        
        return {'data': trials, 'moveThresh': me_thresh}
    except Exception as e:
        print(f"  Warning: Could not load motion energy: {e}")
        return None


def find_video_offset(session_data):
    """Compute video offset matching findVideoOffset.m."""
    bitStart = np.nanmedian(session_data['bitStart'])  # mode equivalent
    # Use mode of sglx_bitstart / fs
    from scipy import stats
    bs = session_data['sglx_bitstart']
    bs = bs[~np.isnan(bs)]
    if len(bs) > 0:
        vidFileOffset = stats.mode(bs, keepdims=False).mode / session_data['sglx_fs']
    else:
        vidFileOffset = bitStart
    vidshift = vidFileOffset - bitStart
    return vidshift


def process_session(session_meta, params, show_processing=False):
    """Process a single session following the reference pipeline."""
    anm = session_meta['anm']
    date = session_meta['date']
    probe = session_meta['probe']
    data_dir = session_meta['dir']
    session_id = f"{anm}_{date}"
    
    print(f"  Processing {session_id} (probe {probe})...")
    t0 = time.time()
    
    # --- Load data ---
    data_path = os.path.join('data', data_dir, f'data_structure_{session_id}.mat')
    if not os.path.exists(data_path):
        print(f"    ERROR: File not found: {data_path}")
        return None
    
    t_load_start = time.time()
    session_data = load_session(data_path, probe)
    print(f"    Data loaded in {time.time()-t_load_start:.1f}s")
    
    if not session_data['clusters']:
        print(f"    WARNING: No clusters found for probe {probe}")
        return None
    
    ntrials = session_data['ntrials']
    
    # --- Filter clusters by quality ---
    # Following findClusters.m: exclude only garbage, gabrga, noisy, real?
    # Empty/null quality strings are kept (treated as unlabeled)
    excluded = params['excluded_qualities']
    valid_clusters = []
    for i, clu in enumerate(session_data['clusters']):
        q = clu['quality'].lower().strip().replace('\x00', '')
        if q in excluded:
            continue
        valid_clusters.append(clu)
    
    print(f"    Clusters: {len(session_data['clusters'])} total, {len(valid_clusters)} after quality filter")
    
    if len(valid_clusters) < params['min_units']:
        print(f"    WARNING: Only {len(valid_clusters)} units, skipping session (min={params['min_units']})")
        return None
    
    # --- Create time axis ---
    tmin, tmax, dt = params['tmin'], params['tmax'], params['dt']
    edges = np.arange(tmin, tmax + dt, dt)
    time_axis = edges[:-1] + dt / 2
    n_timepts = len(time_axis)
    
    # --- Align spikes to goCue and bin ---
    goCue = session_data['goCue']
    
    # Build single-trial firing rate matrix: (time, neurons, trials)
    n_neurons = len(valid_clusters)
    trialdat = np.zeros((n_timepts, n_neurons, ntrials), dtype=np.float32)
    
    for neuron_idx, clu in enumerate(valid_clusters):
        trialtm = clu['trialtm']
        trial_nums = clu['trial']
        
        for trial_idx in range(ntrials):
            trial_num = trial_idx + 1  # 1-indexed
            spike_mask = trial_nums == trial_num
            if not np.any(spike_mask):
                continue
            
            # Align to goCue
            aligned_times = trialtm[spike_mask] - goCue[trial_idx]
            
            # Bin spikes
            counts, _ = np.histogram(aligned_times, bins=edges)
            
            # Convert to firing rate and smooth
            fr = counts.astype(np.float32) / dt
            fr_smooth = smooth_causal(fr, params['smooth_window'], bctype='none')
            trialdat[:, neuron_idx, trial_idx] = fr_smooth
    
    # --- Compute PSTH for low FR removal ---
    # Use all conditions for mean FR calculation (matching removeLowFRClusters)
    # The reference code computes mean FR across conditions from psth
    # psth = trial-averaged firing rate for each condition
    # For simplicity, compute mean FR across all trials
    mean_fr = np.mean(np.mean(trialdat, axis=2), axis=0)  # mean over time and trials
    
    # Remove low FR clusters
    keep_mask = mean_fr > params['low_fr']
    n_kept = np.sum(keep_mask)
    print(f"    Neurons after low FR filter ({params['low_fr']} Hz): {n_kept} of {n_neurons}")
    
    if n_kept < params['min_units']:
        print(f"    WARNING: Only {n_kept} units after FR filter, skipping session")
        return None
    
    trialdat = trialdat[:, keep_mask, :]
    n_neurons = n_kept
    
    # --- Find valid trials ---
    # Exclude early, no-response, and stim trials
    valid_trial_mask = ~session_data['early'] & ~session_data['no'] & ~session_data['stim_enable']
    # Also need hit or miss (not ignore)
    valid_trial_mask = valid_trial_mask & (session_data['hit'] | session_data['miss'])
    
    valid_trial_indices = np.where(valid_trial_mask)[0]
    print(f"    Valid trials: {len(valid_trial_indices)} of {ntrials}")
    
    if len(valid_trial_indices) < 2:
        print(f"    WARNING: Too few valid trials, skipping session")
        return None
    
    # --- Compute behavioral outputs ---
    # Lick direction: R=1, L=0
    lick_direction = session_data['R'][valid_trial_indices].astype(int)
    
    # Context: WC=0, DR=1
    context = (session_data['autowater'][valid_trial_indices] == 0).astype(int)
    
    # Outcome: incorrect=0, correct=1
    outcome = session_data['hit'][valid_trial_indices].astype(int)
    
    # --- Load and process motion energy ---
    me_path = os.path.join('data', data_dir, f'motionEnergy_{session_id}.mat')
    me = load_motion_energy(me_path) if os.path.exists(me_path) else None
    
    # --- Compute video offset ---
    try:
        vidshift = find_video_offset(session_data)
    except:
        vidshift = 0.0
    
    # --- Process motion energy ---
    me_aligned = np.full((n_timepts, ntrials), np.nan, dtype=np.float32)
    if me is not None:
        for trial_idx in range(min(ntrials, len(me['data']))):
            trial_me = me['data'][trial_idx]
            if trial_me.size == 0:
                continue
            
            # Get frameTimes from traj for alignment
            if session_data['traj'] and len(session_data['traj']) > 0:
                cam0 = session_data['traj'][0]
                if cam0 is not None and trial_idx < len(cam0['trials']):
                    ft = cam0['trials'][trial_idx]['frameTimes']
                    if ft.size > 0 and not np.all(np.isnan(ft)):
                        # Align: frameTimes - vidshift - goCue
                        aligned_ft = ft - vidshift - goCue[trial_idx]
                        # Interpolate to neural time axis
                        if len(trial_me) == len(ft):
                            try:
                                f_interp = interp1d(aligned_ft, trial_me, 
                                                   kind='linear', bounds_error=False, fill_value=np.nan)
                                me_aligned[:, trial_idx] = f_interp(time_axis)
                            except:
                                pass
                        else:
                            # ME might have different length - create its own time axis at 400 Hz
                            me_times = np.arange(len(trial_me)) / 400.0
                            try:
                                # Try using frameTimes if available
                                aligned_me_times = me_times - 0.5 - goCue[trial_idx] + session_data['bitStart'][trial_idx]
                                f_interp = interp1d(aligned_me_times, trial_me,
                                                   kind='linear', bounds_error=False, fill_value=np.nan)
                                me_aligned[:, trial_idx] = f_interp(time_axis)
                            except:
                                pass
    
    # Fill NaN with nearest
    for trial_idx in range(ntrials):
        col = me_aligned[:, trial_idx]
        if np.any(np.isnan(col)) and np.any(~np.isnan(col)):
            # Fill with nearest non-NaN
            nans = np.isnan(col)
            not_nans = ~nans
            if np.any(not_nans):
                col[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(not_nans), col[not_nans])
                me_aligned[:, trial_idx] = col
    
    # --- Compute tongue velocity ---
    # Following reference code: interpolate positions, fill NaN with baseline, then compute velocity
    tongue_vel = np.full((n_timepts, ntrials), np.nan, dtype=np.float32)
    if session_data['traj'] and len(session_data['traj']) > 0:
        cam0 = session_data['traj'][0]  # side cam
        if cam0 is not None:
            feat_names = cam0['featNames']
            # Find tongue feature index
            tongue_idx = None
            for fi, fn in enumerate(feat_names):
                if fn.lower() == 'tongue':
                    tongue_idx = fi
                    break
            
            if tongue_idx is not None:
                # First pass: interpolate all trials to get positions
                all_x = np.full((n_timepts, ntrials), np.nan, dtype=np.float64)
                all_y = np.full((n_timepts, ntrials), np.nan, dtype=np.float64)
                
                for trial_idx in range(min(ntrials, len(cam0['trials']))):
                    trial_info = cam0['trials'][trial_idx]
                    if np.isnan(trial_info['NdroppedFrames']):
                        continue
                    ft = trial_info['frameTimes']
                    if ft.size == 0 or np.all(np.isnan(ft)):
                        continue
                    ts = trial_info['ts']
                    if ts.size == 0:
                        continue
                    
                    if ts.ndim == 3:
                        if ts.shape[0] == len(feat_names):
                            x = ts[tongue_idx, 0, :].copy()
                            y = ts[tongue_idx, 1, :].copy()
                        elif ts.shape[2] == len(feat_names):
                            x = ts[:, 0, tongue_idx].copy()
                            y = ts[:, 1, tongue_idx].copy()
                        else:
                            continue
                    else:
                        continue
                    
                    aligned_ft = ft[:len(x)] - vidshift - goCue[trial_idx]
                    
                    try:
                        fx = interp1d(aligned_ft, x, kind='linear', bounds_error=False, fill_value=np.nan)
                        fy = interp1d(aligned_ft, y, kind='linear', bounds_error=False, fill_value=np.nan)
                        all_x[:, trial_idx] = fx(time_axis)
                        all_y[:, trial_idx] = fy(time_axis)
                    except:
                        pass
                
                # Fill NaN tongue positions with session mean (baseline)
                # Following setTongueBaselinePosition: fill with mean initial position
                mean_x = np.nanmean(all_x)
                mean_y = np.nanmean(all_y)
                if np.isnan(mean_x):
                    mean_x = 0.0
                if np.isnan(mean_y):
                    mean_y = 0.0
                
                all_x_filled = all_x.copy()
                all_y_filled = all_y.copy()
                all_x_filled[np.isnan(all_x_filled)] = mean_x
                all_y_filled[np.isnan(all_y_filled)] = mean_y
                
                # Compute velocity for each trial
                for trial_idx in range(ntrials):
                    if np.all(np.isnan(all_x[:, trial_idx])):
                        tongue_vel[:, trial_idx] = 0.0
                        continue
                    
                    xvel = np.gradient(all_x_filled[:, trial_idx])
                    yvel = np.gradient(all_y_filled[:, trial_idx])
                    
                    # Set velocity to 0 where tongue was not visible (NaN in original)
                    nan_mask = np.isnan(all_x[:, trial_idx])
                    xvel[nan_mask] = 0.0
                    yvel[nan_mask] = 0.0
                    
                    speed = np.sqrt(xvel**2 + yvel**2)
                    tongue_vel[:, trial_idx] = speed.astype(np.float32)
    
    # --- Compute paw velocity ---
    paw_vel = np.full((n_timepts, ntrials), np.nan, dtype=np.float32)
    if session_data['traj'] and len(session_data['traj']) > 1:
        cam1 = session_data['traj'][1]  # top cam
        if cam1 is not None:
            feat_names = cam1['featNames']
            # Find paw feature index (top_paw or bottom_paw)
            paw_idx = None
            for fi, fn in enumerate(feat_names):
                if 'top_paw' in fn.lower():
                    paw_idx = fi
                    break
            if paw_idx is None:
                for fi, fn in enumerate(feat_names):
                    if 'paw' in fn.lower():
                        paw_idx = fi
                        break
            
            if paw_idx is not None:
                for trial_idx in range(min(ntrials, len(cam1['trials']))):
                    trial_info = cam1['trials'][trial_idx]
                    if np.isnan(trial_info['NdroppedFrames']):
                        continue
                    ft = trial_info['frameTimes']
                    if ft.size == 0 or np.all(np.isnan(ft)):
                        continue
                    ts = trial_info['ts']
                    if ts.size == 0:
                        continue
                    
                    if ts.ndim == 3:
                        if ts.shape[0] == len(feat_names):
                            x = ts[paw_idx, 0, :]
                            y = ts[paw_idx, 1, :]
                        elif ts.shape[2] == len(feat_names):
                            x = ts[:, 0, paw_idx]
                            y = ts[:, 1, paw_idx]
                        else:
                            continue
                    else:
                        continue
                    
                    aligned_ft = ft[:len(x)] - vidshift - goCue[trial_idx]
                    
                    try:
                        # Smooth positions (not tongue, so smooth)
                        # Don't smooth for now, just interpolate
                        fx = interp1d(aligned_ft, x, kind='linear', bounds_error=False, fill_value=np.nan)
                        fy = interp1d(aligned_ft, y, kind='linear', bounds_error=False, fill_value=np.nan)
                        x_interp = fx(time_axis)
                        y_interp = fy(time_axis)
                        
                        # Fill missing for non-tongue
                        for arr in [x_interp, y_interp]:
                            nans = np.isnan(arr)
                            if np.any(nans) and np.any(~nans):
                                arr[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), arr[~nans])
                        
                        # Compute velocity
                        xvel = np.gradient(x_interp)
                        yvel = np.gradient(y_interp)
                        
                        # Subtract baseline derivative (non-tongue)
                        basederiv_x = np.nanmedian(np.diff(x_interp))
                        basederiv_y = np.nanmedian(np.diff(y_interp))
                        xvel -= basederiv_x
                        yvel -= basederiv_y
                        
                        # Fill missing
                        for arr in [xvel, yvel]:
                            nans = np.isnan(arr)
                            if np.any(nans) and np.any(~nans):
                                arr[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), arr[~nans])
                        
                        speed = np.sqrt(xvel**2 + yvel**2)
                        paw_vel[:, trial_idx] = speed
                    except:
                        pass
    
    # --- Discretize continuous outputs ---
    # For valid trials only
    valid_me = me_aligned[:, valid_trial_indices]
    valid_tongue_vel = tongue_vel[:, valid_trial_indices]
    valid_paw_vel = paw_vel[:, valid_trial_indices]
    
    # Compute per-session 50th percentile thresholds
    # Only use non-NaN values
    def discretize_velocity(vel_data):
        """Discretize velocity into 2 bins using 50th percentile threshold.
        Handles edge case where median equals minimum (e.g., many zeros).
        """
        valid_vals = vel_data[~np.isnan(vel_data)]
        if len(valid_vals) == 0:
            return np.zeros_like(vel_data, dtype=int), 0.0
        threshold = np.percentile(valid_vals, 50)
        # If threshold equals minimum, use strict > to avoid all-1 output
        if threshold <= np.min(valid_vals) + 1e-10:
            discretized = (vel_data > threshold).astype(int)
        else:
            discretized = (vel_data >= threshold).astype(int)
        discretized[np.isnan(vel_data)] = 0  # default for NaN
        return discretized, threshold
    
    tongue_disc, tongue_thresh = discretize_velocity(valid_tongue_vel)
    paw_disc, paw_thresh = discretize_velocity(valid_paw_vel)
    me_disc, me_thresh = discretize_velocity(valid_me)
    
    print(f"    Thresholds - tongue: {tongue_thresh:.2f}, paw: {paw_thresh:.2f}, ME: {me_thresh:.2f}")
    
    # --- Build output ---
    neural_trials = []
    input_trials = []
    output_trials = []
    
    for i, trial_idx in enumerate(valid_trial_indices):
        # Neural: (n_neurons, n_timepoints)
        neural = trialdat[:, :, trial_idx].T.copy()  # (neurons, time)
        neural_trials.append(neural.astype(np.float32))
        
        # Input: time from goCue (continuous, time-varying)
        # Shape: (1, n_timepoints)
        time_input = time_axis.reshape(1, -1).astype(np.float32)
        input_trials.append(time_input)
        
        # Output: (n_output, n_timepoints) for time-varying, (n_output,) for per-trial
        # Per-trial outputs: lick_direction, context, outcome
        # Time-varying outputs: tongue_vel, paw_vel, me
        output = np.zeros((6, n_timepts), dtype=np.int64)
        output[0, :] = lick_direction[i]  # per-trial, broadcast
        output[1, :] = context[i]  # per-trial, broadcast
        output[2, :] = outcome[i]  # per-trial, broadcast
        output[3, :] = tongue_disc[:, i]  # time-varying
        output[4, :] = paw_disc[:, i]  # time-varying
        output[5, :] = me_disc[:, i]  # time-varying
        output_trials.append(output)
    
    elapsed = time.time() - t0
    print(f"    Session processed in {elapsed:.1f}s: {n_neurons} neurons, {len(valid_trial_indices)} trials, {n_timepts} timepoints")
    
    # Filter out trials with all-zero neural data (e.g., recording ended early)
    valid_neural = []
    valid_input = []
    valid_output = []
    n_removed = 0
    for ni, ii, oi in zip(neural_trials, input_trials, output_trials):
        if np.all(ni == 0):
            n_removed += 1
            continue
        valid_neural.append(ni)
        valid_input.append(ii)
        valid_output.append(oi)
    if n_removed > 0:
        print(f"    Removed {n_removed} trials with all-zero neural data")
    neural_trials = valid_neural
    input_trials = valid_input
    output_trials = valid_output
    
    result = {
        'neural': neural_trials,
        'input': input_trials,
        'output': output_trials,
        'subject': anm,
        'session_id': session_id,
        'n_neurons': n_neurons,
        'n_trials': len(neural_trials),
    }
    
    # --- Show processing plots ---
    if show_processing:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(4, 2, figsize=(16, 20))
            fig.suptitle(f'Session: {session_id}', fontsize=14)
            
            # Plot 1: Example neuron PSTH
            ax = axes[0, 0]
            if len(neural_trials) > 0:
                # Average across trials
                all_neural = np.stack([t for t in neural_trials], axis=0)  # (trials, neurons, time)
                mean_fr = np.mean(all_neural, axis=0)  # (neurons, time)
                for n in range(min(5, mean_fr.shape[0])):
                    ax.plot(time_axis, mean_fr[n, :], alpha=0.7, label=f'Neuron {n}')
                ax.set_xlabel('Time from goCue (s)')
                ax.set_ylabel('Firing rate (Hz)')
                ax.set_title('Example neuron PSTHs')
                ax.axvline(0, color='k', linestyle='--', alpha=0.5)
                ax.legend(fontsize=6)
            
            # Plot 2: Trial raster of neural activity
            ax = axes[0, 1]
            if len(neural_trials) > 0:
                ax.imshow(all_neural[:, 0, :], aspect='auto', extent=[tmin, tmax, len(neural_trials), 0])
                ax.set_xlabel('Time from goCue (s)')
                ax.set_ylabel('Trial')
                ax.set_title('Neuron 0 activity across trials')
            
            # Plot 3: Tongue velocity (raw)
            ax = axes[1, 0]
            valid_tv = valid_tongue_vel
            if not np.all(np.isnan(valid_tv)):
                mean_tv = np.nanmean(valid_tv, axis=1)
                ax.plot(time_axis, mean_tv)
                ax.axhline(tongue_thresh, color='r', linestyle='--', label=f'50th pct: {tongue_thresh:.2f}')
                ax.set_xlabel('Time from goCue (s)')
                ax.set_ylabel('Tongue speed')
                ax.set_title('Mean tongue velocity')
                ax.legend()
            
            # Plot 4: Paw velocity
            ax = axes[1, 1]
            valid_pv = valid_paw_vel
            if not np.all(np.isnan(valid_pv)):
                mean_pv = np.nanmean(valid_pv, axis=1)
                ax.plot(time_axis, mean_pv)
                ax.axhline(paw_thresh, color='r', linestyle='--', label=f'50th pct: {paw_thresh:.2f}')
                ax.set_xlabel('Time from goCue (s)')
                ax.set_ylabel('Paw speed')
                ax.set_title('Mean paw velocity')
                ax.legend()
            
            # Plot 5: Motion energy
            ax = axes[2, 0]
            if not np.all(np.isnan(valid_me)):
                mean_me = np.nanmean(valid_me, axis=1)
                ax.plot(time_axis, mean_me)
                ax.axhline(me_thresh, color='r', linestyle='--', label=f'50th pct: {me_thresh:.2f}')
                ax.set_xlabel('Time from goCue (s)')
                ax.set_ylabel('Motion energy')
                ax.set_title('Mean motion energy')
                ax.legend()
            
            # Plot 6: Output distributions
            ax = axes[2, 1]
            labels = ['Lick dir', 'Context', 'Outcome']
            vals = [lick_direction, context, outcome]
            x_pos = np.arange(len(labels))
            frac_1 = [np.mean(v) for v in vals]
            ax.bar(x_pos, frac_1)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(labels)
            ax.set_ylabel('Fraction = 1')
            ax.set_title('Per-trial output distributions')
            
            # Plot 7: Discretized tongue velocity example
            ax = axes[3, 0]
            if tongue_disc.shape[1] > 0:
                ax.imshow(tongue_disc.T, aspect='auto', extent=[tmin, tmax, tongue_disc.shape[1], 0],
                         cmap='binary', interpolation='nearest')
                ax.set_xlabel('Time from goCue (s)')
                ax.set_ylabel('Trial')
                ax.set_title('Discretized tongue velocity')
            
            # Plot 8: Discretized ME example
            ax = axes[3, 1]
            if me_disc.shape[1] > 0:
                ax.imshow(me_disc.T, aspect='auto', extent=[tmin, tmax, me_disc.shape[1], 0],
                         cmap='binary', interpolation='nearest')
                ax.set_xlabel('Time from goCue (s)')
                ax.set_ylabel('Trial')
                ax.set_title('Discretized motion energy')
            
            plt.tight_layout()
            plt.savefig(f'processing_{session_id}.png', dpi=100)
            plt.close()
            print(f"    Saved processing plot: processing_{session_id}.png")
        except Exception as e:
            print(f"    Warning: Could not create processing plot: {e}")
    
    return result


def main():
    parser = argparse.ArgumentParser(description='Convert data to decoder format')
    parser.add_argument('output', help='Output pickle file path')
    parser.add_argument('--sample', action='store_true', help='Process only 2 sessions')
    parser.add_argument('--full', action='store_true', help='Process all sessions (default)')
    parser.add_argument('--show-processing', action='store_true', help='Show processing plots')
    args = parser.parse_args()
    
    if not args.sample:
        args.full = True
    
    sessions_to_process = SESSION_META
    if args.sample:
        # Pick 2 sessions: one from each task type
        dr_sessions = [s for s in SESSION_META if s['task'] == 'DR']
        rd_sessions = [s for s in SESSION_META if s['task'] == 'RandDelay']
        sessions_to_process = [dr_sessions[3], rd_sessions[2]]  # JEB7 and JEB12
        print(f"Sample mode: processing {len(sessions_to_process)} sessions")
    else:
        print(f"Full mode: processing {len(sessions_to_process)} sessions")
    
    # Process all sessions
    all_neural = []
    all_input = []
    all_output = []
    all_subjects = []
    subject_idx = []
    
    total_start = time.time()
    n_processed = 0
    
    for sess_meta in sessions_to_process:
        result = process_session(sess_meta, PARAMS, show_processing=args.show_processing)
        if result is None:
            continue
        
        all_neural.append(result['neural'])
        all_input.append(result['input'])
        all_output.append(result['output'])
        
        subj = result['subject']
        if subj not in all_subjects:
            all_subjects.append(subj)
        subject_idx.append(all_subjects.index(subj))
        
        n_processed += 1
    
    total_elapsed = time.time() - total_start
    print(f"\nProcessed {n_processed} sessions in {total_elapsed:.1f}s")
    
    # Build output dictionary
    data = {
        'neural': all_neural,
        'input': all_input,
        'output': all_output,
        'subjects': all_subjects,
        'subject_idx': np.array(subject_idx),
        'brain_regions': ['ALM'],
        'brain_region_idx': [np.zeros(len(sess_neural), dtype=int) for sess_neural in all_neural],
        'input_names': ['time_from_goCue'],
        'output_names': ['lick_direction', 'context', 'outcome', 'tongue_velocity', 'paw_velocity', 'motion_energy'],
        'output_values': [
            ['left', 'right'],
            ['WC', 'DR'],
            ['incorrect', 'correct'],
            ['low', 'high'],
            ['low', 'high'],
            ['low', 'high'],
        ],
        'metadata': {
            'task_description': 'Two-context licking task: delayed-response (DR) and water-cued (WC) paradigms with ALM recordings',
            'time_bin_size': PARAMS['dt'] * 1000,  # in ms
            'temporal_alignment_event': 'Go cue onset',
            'off_start': PARAMS['tmin'],
            'off_end': PARAMS['tmax'],
            'smoothing': f"Causal Gaussian kernel, window={PARAMS['smooth_window']}",
            'low_fr_threshold': PARAMS['low_fr'],
            'recording_region': 'ALM (anterior lateral motor cortex)',
            'species': 'mouse',
        }
    }
    
    # brain_region_idx: for each session, array of zeros (all ALM)
    # Fix: should be based on n_neurons per session, not n_trials
    data['brain_region_idx'] = []
    for sess_neural in all_neural:
        if len(sess_neural) > 0:
            n_neurons = sess_neural[0].shape[0]
        else:
            n_neurons = 0
        data['brain_region_idx'].append(np.zeros(n_neurons, dtype=int))
    
    # Print summary
    print(f"\n=== Dataset Summary ===")
    print(f"Sessions: {len(all_neural)}")
    print(f"Subjects: {all_subjects}")
    total_trials = sum(len(s) for s in all_neural)
    print(f"Total trials: {total_trials}")
    total_neurons = sum(s[0].shape[0] if len(s) > 0 else 0 for s in all_neural)
    print(f"Total neurons: {total_neurons}")
    neurons_per_session = [s[0].shape[0] if len(s) > 0 else 0 for s in all_neural]
    print(f"Neurons per session: min={min(neurons_per_session)}, max={max(neurons_per_session)}, mean={np.mean(neurons_per_session):.1f}")
    trials_per_session = [len(s) for s in all_neural]
    print(f"Trials per session: min={min(trials_per_session)}, max={max(trials_per_session)}, mean={np.mean(trials_per_session):.1f}")
    
    # Save
    print(f"\nSaving to {args.output}...")
    with open(args.output, 'wb') as f:
        pickle.dump(data, f, protocol=4)
    
    file_size = os.path.getsize(args.output) / (1024**2)
    print(f"Saved {args.output} ({file_size:.1f} MB)")
    print("Done!")


if __name__ == '__main__':
    main()
