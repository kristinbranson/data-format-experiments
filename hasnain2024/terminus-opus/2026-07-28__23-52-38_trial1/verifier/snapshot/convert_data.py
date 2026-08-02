#!/usr/bin/env python3
"""Convert Hasnain et al. 2024 data to decoder-compatible format.

Usage:
    python -u convert_data.py <outpicklefile> [--sample] [--full] [--show-processing]
"""

import sys
import os
import argparse
import pickle
import time
import re
import warnings
import numpy as np
import h5py
import scipy.io as sio
from scipy.signal import windows as sig_windows
from scipy.interpolate import interp1d

warnings.filterwarnings('ignore')

# ============================================================
# Parameters (matching reference code getDefaultParams.m)
# ============================================================
ALIGN_EVENT = 'goCue'
DT = 1.0 / 200.0  # 5ms time bins
TMIN = -2.5
TMAX = 2.5
SMOOTH_N = 15  # causal gaussian smoothing window
LOW_FR = 0.5   # minimum firing rate threshold (Hz)
MIN_UNITS = 10  # minimum units per session
MIN_HIT_TRIALS = 40  # minimum hit trials per direction for session inclusion
VIDEO_FS = 400  # video frame rate


def make_time_axis():
    """Create time axis matching MATLAB code: tmin:dt:tmax."""
    time_axis = np.arange(TMIN, TMAX + DT/2, DT)
    # Ensure we don't go past TMAX due to floating point
    time_axis = time_axis[time_axis <= TMAX + 1e-10]
    return time_axis


def causal_gaussian_kernel(N):
    """Create causal gaussian smoothing kernel matching mySmooth.m."""
    if N <= 1:
        return np.array([1.0])
    kern = np.array(sig_windows.gaussian(N, std=np.std(np.arange(1, N+1))))
    # Make causal: zero the first half
    kern[:N//2] = 0
    kern = kern / kern.sum()
    return kern


def smooth_signal(x, N):
    """Apply causal gaussian smoothing matching mySmooth.m.
    x: 1D array (time,)
    Returns smoothed 1D array.
    """
    if N <= 1:
        return x
    kern = causal_gaussian_kernel(N)
    return np.convolve(x, kern, mode='same')


def read_h5_string(f, ref):
    """Read a string from an HDF5 object reference."""
    try:
        data = np.array(f[ref]).flatten()
        return ''.join(chr(int(c)) for c in data if c > 0)
    except:
        return ''



def detect_mat_format(filepath):
    """Detect if a .mat file is HDF5 (v7.3) or v5 format."""
    with open(filepath, 'rb') as f:
        header = f.read(8)
    if header[:8] == b'\x89HDF\r\n\x1a\n':
        return 'hdf5'
    elif header[:6] == b'MATLAB':
        # Check version in header
        with open(filepath, 'rb') as f:
            full_header = f.read(128)
        if b'7.3' in full_header[:116]:
            return 'hdf5'  # MATLAB 7.3 uses HDF5
        return 'v5'
    return 'unknown'


def load_session_data(filepath):
    """Load a session's data from a .mat file (auto-detect format)."""
    fmt = detect_mat_format(filepath)
    if fmt == 'hdf5':
        return load_session_data_hdf5(filepath)
    elif fmt == 'v5':
        return load_session_data_v5(filepath)
    else:
        # Try both
        try:
            return load_session_data_hdf5(filepath)
        except:
            return load_session_data_v5(filepath)


def load_session_data_v5(filepath):
    """Load session data from MATLAB v5 format using scipy.io."""
    data = sio.loadmat(filepath, squeeze_me=False)
    obj = data['obj']
    session = {}
    
    def get_field(struct, name):
        """Extract field from scipy.io structured array, handling nesting."""
        val = struct[name]
        while val.dtype == np.object_ and val.shape == (1, 1):
            val = val[0, 0]
        return val
    
    # ---- Behavioral data (bp) ----
    bp_raw = obj['bp'][0, 0]
    bp = {}
    bp['Ntrials'] = int(get_field(bp_raw, 'Ntrials').flatten()[0])
    bp['L'] = get_field(bp_raw, 'L').flatten().astype(bool)
    bp['R'] = get_field(bp_raw, 'R').flatten().astype(bool)
    bp['hit'] = get_field(bp_raw, 'hit').flatten().astype(bool)
    bp['miss'] = get_field(bp_raw, 'miss').flatten().astype(bool)
    bp['no'] = get_field(bp_raw, 'no').flatten().astype(bool)
    bp['autowater'] = get_field(bp_raw, 'autowater').flatten().astype(bool)
    bp['early'] = get_field(bp_raw, 'early').flatten().astype(bool)
    
    if 'stim' in bp_raw.dtype.names:
        stim_raw = bp_raw['stim'][0, 0]
        bp['stim_enable'] = get_field(stim_raw, 'enable').flatten().astype(bool)
    else:
        bp['stim_enable'] = np.zeros(bp['Ntrials'], dtype=bool)
    
    # Event times
    ev_raw = bp_raw['ev'][0, 0]
    bp['ev'] = {}
    bp['ev']['goCue'] = get_field(ev_raw, 'goCue').flatten().astype(float)
    bp['ev']['sample'] = get_field(ev_raw, 'sample').flatten().astype(float)
    bp['ev']['delay'] = get_field(ev_raw, 'delay').flatten().astype(float)
    try:
        bp['ev']['reward'] = get_field(ev_raw, 'reward').flatten().astype(float)
    except:
        bp['ev']['reward'] = np.full(bp['Ntrials'], np.nan)
    try:
        bp['ev']['bitStart'] = get_field(ev_raw, 'bitStart').flatten().astype(float)
    except:
        bp['ev']['bitStart'] = np.full(bp['Ntrials'], np.nan)
    
    session['bp'] = bp
    
    # ---- Cluster (spike) data ----
    clu_raw = obj['clu'][0, 0]
    # clu_raw is (1, n_probes) object array
    if clu_raw.dtype == np.object_:
        n_probes = clu_raw.shape[1] if clu_raw.ndim > 1 else 1
    else:
        n_probes = 1
    
    clu_list = []
    for p in range(n_probes):
        if clu_raw.dtype == np.object_:
            probe_data = clu_raw[0, p] if clu_raw.ndim > 1 else clu_raw[0, 0]
        else:
            probe_data = clu_raw
        
        n_units = probe_data.shape[1] if probe_data.ndim > 1 else probe_data.shape[0]
        units = []
        for u in range(n_units):
            unit_raw = probe_data[0, u] if probe_data.ndim > 1 else probe_data[u]
            unit = {}
            unit['tm'] = unit_raw['tm'].flatten().astype(float)
            unit['trial'] = unit_raw['trial'].flatten().astype(int)
            unit['trialtm'] = unit_raw['trialtm'].flatten().astype(float)
            q = unit_raw['quality']
            if hasattr(q, 'flatten'):
                q_flat = q.flatten()
                if q_flat.size > 0:
                    unit['quality'] = str(q_flat[0]).strip()
                else:
                    unit['quality'] = ''
            else:
                unit['quality'] = str(q).strip()
            units.append(unit)
        clu_list.append(units)
    session['clu'] = clu_list
    
    # ---- SpikeGLX metadata ----
    try:
        sglx_raw = obj['sglx'][0, 0]
        sglx_fs = float(get_field(sglx_raw, 'fs').flatten()[0])
        bitcode_raw = sglx_raw['bitcode'][0, 0]
        bitcode_bitstart = get_field(bitcode_raw, 'bitstart').flatten().astype(float)
        session['sglx'] = {'fs': sglx_fs, 'bitcode_bitstart': bitcode_bitstart}
    except:
        session['sglx'] = None
    
    # ---- Trajectory data ----
    traj_data = []
    traj_raw = obj['traj'][0, 0]
    n_cams = traj_raw.shape[1] if traj_raw.ndim > 1 else 1
    for cam in range(n_cams):
        cam_raw = traj_raw[0, cam] if traj_raw.ndim > 1 else traj_raw[0, 0]
        n_trials_traj = cam_raw.shape[1] if cam_raw.ndim > 1 else cam_raw.shape[0]
        
        cam_data = {'n_trials': n_trials_traj, 'trials': []}
        
        # Get feature names from first trial
        try:
            t0 = cam_raw[0, 0] if cam_raw.ndim > 1 else cam_raw[0]
            feat_names_raw = t0['featNames']
            feat_names = []
            fn_flat = feat_names_raw.flatten()
            for i in range(fn_flat.shape[0]):
                fn_item = fn_flat[i]
                if hasattr(fn_item, 'flatten'):
                    feat_names.append(str(fn_item.flatten()[0]))
                else:
                    feat_names.append(str(fn_item))
            cam_data['feat_names'] = feat_names
        except:
            cam_data['feat_names'] = []
        
        # Load per-trial data
        for trix in range(n_trials_traj):
            trial_data = {}
            try:
                t = cam_raw[0, trix] if cam_raw.ndim > 1 else cam_raw[trix]
                ts = t['ts']
                if ts.size > 0:
                    # v5 format: (timepoints, 3, features) -> transpose to (features, 3, timepoints)
                    trial_data['ts'] = np.transpose(ts, (2, 1, 0))
                else:
                    trial_data['ts'] = None
            except:
                trial_data['ts'] = None
            
            try:
                t = cam_raw[0, trix] if cam_raw.ndim > 1 else cam_raw[trix]
                ft = t['frameTimes'].flatten().astype(float)
                trial_data['frameTimes'] = ft
            except:
                trial_data['frameTimes'] = None
            
            try:
                t = cam_raw[0, trix] if cam_raw.ndim > 1 else cam_raw[trix]
                nd = t['NdroppedFrames']
                if nd.size > 0:
                    trial_data['NdroppedFrames'] = float(nd.flatten()[0])
                else:
                    trial_data['NdroppedFrames'] = np.nan
            except:
                trial_data['NdroppedFrames'] = np.nan
            
            cam_data['trials'].append(trial_data)
        
        traj_data.append(cam_data)
    
    session['traj'] = traj_data
    
    return session

def load_session_data_hdf5(filepath):
    """Load a session's data from a .mat file.
    Returns dict with all needed fields.
    """
    f = h5py.File(filepath, 'r')
    session = {}
    
    # ---- Behavioral data (bp) ----
    bp = {}
    bp['Ntrials'] = int(f['obj/bp/Ntrials'][0, 0])
    bp['L'] = np.array(f['obj/bp/L']).flatten().astype(bool)
    bp['R'] = np.array(f['obj/bp/R']).flatten().astype(bool)
    bp['hit'] = np.array(f['obj/bp/hit']).flatten().astype(bool)
    bp['miss'] = np.array(f['obj/bp/miss']).flatten().astype(bool)
    bp['no'] = np.array(f['obj/bp/no']).flatten().astype(bool)
    bp['autowater'] = np.array(f['obj/bp/autowater']).flatten().astype(bool)
    bp['early'] = np.array(f['obj/bp/early']).flatten().astype(bool)
    bp['stim_enable'] = np.array(f['obj/bp/stim/enable']).flatten().astype(bool)
    
    # Event times
    bp['ev'] = {}
    bp['ev']['goCue'] = np.array(f['obj/bp/ev/goCue']).flatten()
    bp['ev']['sample'] = np.array(f['obj/bp/ev/sample']).flatten()
    bp['ev']['delay'] = np.array(f['obj/bp/ev/delay']).flatten()
    try:
        bp['ev']['reward'] = np.array(f['obj/bp/ev/reward']).flatten()
    except:
        bp['ev']['reward'] = np.full(bp['Ntrials'], np.nan)
    try:
        bp['ev']['bitStart'] = np.array(f['obj/bp/ev/bitStart']).flatten()
    except:
        bp['ev']['bitStart'] = np.full(bp['Ntrials'], np.nan)
    
    session['bp'] = bp
    
    # ---- Cluster (spike) data ----
    n_probes = f['obj/clu'].shape[0]
    clu_list = []
    for p in range(n_probes):
        clu_ref = f['obj/clu'][p, 0]
        clu_group = f[clu_ref]
        
        # Skip probes with invalid clu data (not a group or missing 'tm')
        if not isinstance(clu_group, h5py.Group) or 'tm' not in clu_group:
            clu_list.append([])  # empty unit list for this probe
            continue
        
        n_units = clu_group['tm'].shape[0]
        units = []
        for u in range(n_units):
            unit = {}
            unit['tm'] = np.array(f[clu_group['tm'][u, 0]]).flatten()
            unit['trial'] = np.array(f[clu_group['trial'][u, 0]]).flatten().astype(int)
            unit['trialtm'] = np.array(f[clu_group['trialtm'][u, 0]]).flatten()
            unit['quality'] = read_h5_string(f, clu_group['quality'][u, 0])
            units.append(unit)
        clu_list.append(units)
    session['clu'] = clu_list
    
    # ---- SpikeGLX metadata (for video offset) ----
    try:
        sglx_fs = float(f['obj/sglx/fs'][0, 0])
        bitcode_bitstart = np.array(f['obj/sglx/bitcode/bitstart']).flatten()
        session['sglx'] = {'fs': sglx_fs, 'bitcode_bitstart': bitcode_bitstart}
    except:
        session['sglx'] = None
    
    # ---- Trajectory data ----
    traj_data = []
    n_cams = f['obj/traj'].shape[0]
    for cam in range(n_cams):
        traj_ref = f['obj/traj'][cam, 0]
        traj_group = f[traj_ref]
        
        cam_data = {
            'n_trials': traj_group['ts'].shape[0],
            'trials': []
        }
        
        # Get feature names from first trial
        try:
            fn_ref = traj_group['featNames'][0, 0]
            fn_data = f[fn_ref]
            feat_names = []
            if fn_data.dtype == np.object_:
                for i in range(fn_data.shape[1] if len(fn_data.shape) > 1 else fn_data.shape[0]):
                    idx = (0, i) if len(fn_data.shape) > 1 else (i,)
                    name = read_h5_string(f, fn_data[idx])
                    feat_names.append(name)
            cam_data['feat_names'] = feat_names
        except:
            cam_data['feat_names'] = []
        
        # Load per-trial data
        for trix in range(cam_data['n_trials']):
            trial_data = {}
            try:
                ts_ref = traj_group['ts'][trix, 0]
                ts = np.array(f[ts_ref])  # shape: (n_feats, 3, n_timepoints) in h5py
                trial_data['ts'] = ts  # Keep in h5py order
            except:
                trial_data['ts'] = None
            
            try:
                ft_ref = traj_group['frameTimes'][trix, 0]
                ft = np.array(f[ft_ref]).flatten()
                trial_data['frameTimes'] = ft
            except:
                trial_data['frameTimes'] = None
            
            try:
                nd_ref = traj_group['NdroppedFrames'][trix, 0]
                nd = np.array(f[nd_ref]).flatten()
                trial_data['NdroppedFrames'] = nd[0] if len(nd) > 0 else np.nan
            except:
                trial_data['NdroppedFrames'] = np.nan
            
            cam_data['trials'].append(trial_data)
        
        traj_data.append(cam_data)
    
    session['traj'] = traj_data
    
    f.close()
    return session


def load_motion_energy(me_filepath):
    """Load motion energy from .mat file, handling multiple format variants."""
    try:
        data = sio.loadmat(me_filepath, squeeze_me=False)
        me_raw = data['me']
        
        # Determine format
        if me_raw.dtype.names and 'data' in me_raw.dtype.names:
            # Struct format: me.data, me.moveThresh
            me_data_field = me_raw['data'][0, 0]
            
            # Check if me.data is itself a struct (double-nested)
            # MATLAB code: if isstruct(me.data); me.data = me.data.data; end
            if hasattr(me_data_field, 'dtype') and me_data_field.dtype.names and 'data' in me_data_field.dtype.names:
                # Double-nested: me.data.data contains the actual trial data
                me_data_arr = me_data_field['data'][0, 0]
                try:
                    me_thresh_inner = me_data_field['moveThresh'][0, 0]
                    me_thresh = float(me_thresh_inner.flatten()[0])
                except:
                    me_thresh = float(me_raw['moveThresh'][0, 0].flatten()[0])
            else:
                # Normal struct: me.data is the trial data
                me_data_arr = me_data_field
                me_thresh = float(me_raw['moveThresh'][0, 0].flatten()[0])
        else:
            # Direct format: me is the trial data array directly
            me_data_arr = me_raw
            me_thresh = 0  # No threshold available
        
        # Extract per-trial data
        me_trials = []
        for i in range(me_data_arr.shape[0]):
            trial_me = me_data_arr[i, 0]
            if trial_me is not None and hasattr(trial_me, 'size') and trial_me.size > 0:
                me_trials.append(trial_me.flatten())
            else:
                me_trials.append(np.array([]))
        
        return {'data': me_trials, 'moveThresh': me_thresh}
    except Exception as e:
        print(f'  Warning: Could not load motion energy: {e}')
        return None


def compute_video_offset(session):
    """Compute video-neural offset matching findVideoOffset.m."""
    if session['sglx'] is None:
        return 0.0
    
    bitStart_mode = float(np.median(session['bp']['ev']['bitStart'][~np.isnan(session['bp']['ev']['bitStart'])]))
    # Use scipy.stats.mode equivalent
    from scipy import stats
    bitStart_vals = session['bp']['ev']['bitStart'][~np.isnan(session['bp']['ev']['bitStart'])]
    if len(bitStart_vals) > 0:
        bitStart_mode = float(stats.mode(bitStart_vals, keepdims=False).mode)
    else:
        bitStart_mode = 0.0
    
    bitcode_bitstart_vals = session['sglx']['bitcode_bitstart'][~np.isnan(session['sglx']['bitcode_bitstart'])]
    if len(bitcode_bitstart_vals) > 0:
        vidFileOffset = float(stats.mode(bitcode_bitstart_vals, keepdims=False).mode) / session['sglx']['fs']
    else:
        vidFileOffset = 0.0
    
    vidshift = vidFileOffset - bitStart_mode
    return vidshift


def process_spikes(session, probe_idx, time_axis):
    """Process spikes: align to goCue, bin, smooth, filter.
    Returns: firing_rates (n_neurons, n_timepoints, n_trials), cluid_mask
    """
    units = session['clu'][probe_idx]
    goCue = session['bp']['ev']['goCue']
    Ntrials = session['bp']['Ntrials']
    
    edges = np.append(time_axis, time_axis[-1] + DT)
    n_time = len(time_axis)
    
    # Filter out garbage clusters
    valid_units = []
    for u_idx, unit in enumerate(units):
        quality = unit['quality'].lower().strip()
        if 'garbage' not in quality:
            valid_units.append(u_idx)
    
    if len(valid_units) == 0:
        return None, []
    
    # Compute firing rates for valid units
    firing_rates = np.zeros((len(valid_units), n_time, Ntrials), dtype=np.float32)
    
    for i, u_idx in enumerate(valid_units):
        unit = units[u_idx]
        trial_nums = unit['trial']
        trialtm = unit['trialtm']
        
        for j in range(Ntrials):
            trial_num = j + 1  # MATLAB 1-indexed
            spk_mask = trial_nums == trial_num
            
            if not np.any(spk_mask):
                continue
            
            # Align to goCue
            aligned_times = trialtm[spk_mask] - goCue[j]
            
            # Bin spikes
            counts, _ = np.histogram(aligned_times, bins=edges)
            
            # Convert to firing rate and smooth
            fr = counts.astype(np.float64) / DT
            fr_smooth = smooth_signal(fr, SMOOTH_N)
            firing_rates[i, :, j] = fr_smooth
    
    # Compute mean firing rate across all trials for each unit (for filtering)
    # Match removeLowFRClusters: mean of PSTH across conditions
    mean_frs = np.mean(np.mean(firing_rates, axis=2), axis=1)  # mean over time and trials
    
    # Filter low FR units
    fr_mask = mean_frs > LOW_FR
    
    if np.sum(fr_mask) == 0:
        return None, []
    
    firing_rates = firing_rates[fr_mask]
    kept_units = [valid_units[i] for i in range(len(valid_units)) if fr_mask[i]]
    
    return firing_rates, kept_units


def extract_feature_velocity(session, cam_idx, feat_name, time_axis, vidshift):
    """Extract velocity for a DLC feature, matching findPosition + findVelocity.
    Returns: velocity magnitude (n_timepoints, n_trials)
    """
    Ntrials = session['bp']['Ntrials']
    goCue = session['bp']['ev']['goCue']
    
    cam_data = session['traj'][cam_idx]
    feat_names = cam_data['feat_names']
    
    # Find feature index
    feat_idx = None
    for i, name in enumerate(feat_names):
        if name.lower() == feat_name.lower():
            feat_idx = i
            break
    
    if feat_idx is None:
        return np.full((len(time_axis), Ntrials), np.nan)
    
    is_tongue = 'tongue' in feat_name.lower()
    
    xpos = np.full((len(time_axis), Ntrials), np.nan)
    ypos = np.full((len(time_axis), Ntrials), np.nan)
    
    for trix in range(Ntrials):
        trial = cam_data['trials'][trix]
        
        if trial['ts'] is None:
            continue
        if np.isnan(trial['NdroppedFrames']):
            continue
        
        ts = trial['ts']  # (n_feats, 3, n_timepoints) in h5py
        
        # Get frame times
        if trial['frameTimes'] is not None and not np.all(np.isnan(trial['frameTimes'])):
            frame_times = trial['frameTimes']
        else:
            n_frames = ts.shape[2]
            frame_times = np.arange(1, n_frames + 1) / VIDEO_FS
        
        # Get x, y for this feature
        x = ts[feat_idx, 0, :]  # x coordinate
        y = ts[feat_idx, 1, :]  # y coordinate
        
        # Align time to goCue
        aligned_times = frame_times - vidshift - goCue[trix]
        
        # Smooth non-tongue features (mySmooth with N=1 and 'reflect')
        # In the reference code, non-tongue features are smoothed with mySmooth(ts, 1, 'reflect')
        # N=1 means no smoothing
        
        # Interpolate to time axis
        try:
            valid = ~np.isnan(x) & ~np.isnan(aligned_times)
            if np.sum(valid) < 2:
                continue
            
            f_x = interp1d(aligned_times[valid], x[valid], kind='linear',
                          bounds_error=False, fill_value=np.nan)
            f_y = interp1d(aligned_times[valid], y[valid], kind='linear',
                          bounds_error=False, fill_value=np.nan)
            
            xpos[:, trix] = f_x(time_axis)
            ypos[:, trix] = f_y(time_axis)
        except:
            continue
        
        # Fill missing for non-tongue
        if not is_tongue:
            mask_nan = np.isnan(xpos[:, trix])
            if np.any(mask_nan) and not np.all(mask_nan):
                xpos[:, trix] = _fill_nearest(xpos[:, trix])
                ypos[:, trix] = _fill_nearest(ypos[:, trix])
    
    # Compute velocity (gradient, matching findVelocity.m)
    xvel = np.full_like(xpos, np.nan)
    yvel = np.full_like(ypos, np.nan)
    
    for trix in range(Ntrials):
        if np.all(np.isnan(xpos[:, trix])):
            continue
        
        xv = np.gradient(xpos[:, trix])
        yv = np.gradient(ypos[:, trix])
        
        if not is_tongue:
            # Subtract baseline derivative
            base_x = np.nanmedian(np.diff(xpos[:, trix]))
            base_y = np.nanmedian(np.diff(ypos[:, trix]))
            xv = xv - base_x
            yv = yv - base_y
            # Fill missing
            xv = _fill_nearest(xv)
            yv = _fill_nearest(yv)
        else:
            # Set tongue velocity to 0 if not visible
            xv[np.isnan(xv)] = 0
            yv[np.isnan(yv)] = 0
        
        xvel[:, trix] = xv
        yvel[:, trix] = yv
    
    # Compute velocity magnitude
    vel_mag = np.sqrt(xvel**2 + yvel**2)
    
    return vel_mag


def _fill_nearest(arr):
    """Fill NaN values with nearest non-NaN value."""
    mask = np.isnan(arr)
    if not np.any(mask) or np.all(mask):
        return arr
    arr = arr.copy()
    # Forward fill then backward fill
    valid = np.where(~mask)[0]
    if len(valid) == 0:
        return arr
    for i in range(len(arr)):
        if mask[i]:
            # Find nearest valid
            dists = np.abs(valid - i)
            arr[i] = arr[valid[np.argmin(dists)]]
    return arr


def align_motion_energy(me_data, session, time_axis, vidshift):
    """Align motion energy to goCue, matching loadMotionEnergy.m."""
    Ntrials = session['bp']['Ntrials']
    goCue = session['bp']['ev']['goCue']
    
    me_aligned = np.full((len(time_axis), Ntrials), np.nan)
    
    n_me_trials = len(me_data['data'])
    
    for trix in range(min(Ntrials, n_me_trials)):
        trial_me = me_data['data'][trix]
        if trial_me.size == 0:
            continue
        
        # Get frame times for this trial
        cam_data = session['traj'][0]  # camera 0
        trial_traj = cam_data['trials'][trix] if trix < len(cam_data['trials']) else None
        
        if trial_traj is not None and trial_traj['frameTimes'] is not None and not np.all(np.isnan(trial_traj['frameTimes'])):
            frame_times = trial_traj['frameTimes']
            aligned_times = frame_times - vidshift - goCue[trix]
        else:
            # Fallback: create frame times
            n_frames = trial_me.shape[0]
            if trial_traj is not None and trial_traj['ts'] is not None:
                n_frames = trial_traj['ts'].shape[2]
            frame_times = np.arange(1, n_frames + 1) / VIDEO_FS
            aligned_times = frame_times - 0.5 - goCue[trix]
        
        # Trim to same length
        n = min(len(trial_me), len(aligned_times))
        if n < 2:
            continue
        
        try:
            f_me = interp1d(aligned_times[:n], trial_me[:n], kind='linear',
                           bounds_error=False, fill_value=np.nan)
            me_aligned[:, trix] = f_me(time_axis)
        except:
            continue
    
    # Fill NaN with nearest
    for trix in range(Ntrials):
        if not np.all(np.isnan(me_aligned[:, trix])):
            me_aligned[:, trix] = _fill_nearest(me_aligned[:, trix])
    
    return me_aligned


def get_available_sessions():
    """Get list of all available sessions with their metadata."""
    sessions = []
    
    # Parse loading scripts to get probe info
    script_dir = 'code/DataLoadingScripts/Recording and video'
    probe_map = {}  # (animal, date) -> probes
    
    for f in sorted(os.listdir(script_dir)):
        if f.endswith('.m'):
            with open(os.path.join(script_dir, f)) as fh:
                content = fh.read()
            anm = None
            date = None
            for line in content.split('\n'):
                line = line.strip()
                if line.startswith('%'):
                    continue
                m = re.search(r"meta\(end(?:\+1)?\)\.anm\s*=\s*'(\w+)'", line)
                if m:
                    anm = m.group(1)
                m = re.search(r"meta\(end(?:\+1)?\)\.date\s*=\s*'([\d-]+)'", line)
                if m:
                    date = m.group(1)
                m = re.search(r"meta\(end(?:\+1)?\)\.probe\s*=\s*(.+?);", line)
                if m:
                    probe_str = m.group(1).strip()
                    if '[' in probe_str:
                        probes = [int(x) for x in re.findall(r'\d+', probe_str)]
                    else:
                        probes = [int(probe_str)]
                    if anm and date:
                        probe_map[(anm, date)] = probes
    
    # Find all data files
    for data_dir in ['data/Ephys_Behavior', 'data/RandomizedDelay_Ephys_Behavior']:
        if not os.path.exists(data_dir):
            continue
        for fn in sorted(os.listdir(data_dir)):
            if not fn.startswith('data_structure_'):
                continue
            m = re.match(r'data_structure_(\w+)_(\d{4}-\d{2}-\d{2})\.mat', fn)
            if not m:
                continue
            
            animal = m.group(1)
            date = m.group(2)
            
            # Skip behavior-only sessions (MAH animals)
            if animal.startswith('MAH'):
                continue
            
            filepath = os.path.join(data_dir, fn)
            
            # Get probe info from loading scripts
            probes = probe_map.get((animal, date), [1])  # default to probe 1
            
            # Check for motion energy file
            me_fn = fn.replace('data_structure_', 'motionEnergy_')
            me_filepath = os.path.join(data_dir, me_fn)
            has_me = os.path.exists(me_filepath)
            
            sessions.append({
                'animal': animal,
                'date': date,
                'filepath': filepath,
                'probes': probes,
                'me_filepath': me_filepath if has_me else None,
                'data_dir': data_dir
            })
    
    return sessions


def check_session_inclusion(session_data, probes):
    """Check if session meets inclusion criteria.
    Matching UseInclusionCritera.m: >40 R hit DR trials AND >40 L hit DR trials.
    """
    bp = session_data['bp']
    
    # Count R hit DR trials (no stim, no autowater, no early)
    r_hit_dr = bp['R'] & bp['hit'] & ~bp['stim_enable'] & ~bp['autowater'] & ~bp['early']
    l_hit_dr = bp['L'] & bp['hit'] & ~bp['stim_enable'] & ~bp['autowater'] & ~bp['early']
    
    n_r = np.sum(r_hit_dr)
    n_l = np.sum(l_hit_dr)
    
    return n_r > MIN_HIT_TRIALS and n_l > MIN_HIT_TRIALS, n_r, n_l


def process_session(sess_info, time_axis, show_processing=False):
    """Process a single session and return formatted data."""
    animal = sess_info['animal']
    date = sess_info['date']
    probes = sess_info['probes']
    
    print(f'  Loading {animal}_{date}...')
    t0 = time.time()
    
    session = load_session_data(sess_info['filepath'])
    t_load = time.time() - t0
    print(f'    Loaded in {t_load:.1f}s')
    
    # Check inclusion criteria
    included, n_r, n_l = check_session_inclusion(session, probes)
    if not included:
        print(f'    EXCLUDED: R hit DR={n_r}, L hit DR={n_l} (need >{MIN_HIT_TRIALS} each)')
        return None
    print(f'    R hit DR={n_r}, L hit DR={n_l}')
    
    # Process spikes for each probe and combine
    t0 = time.time()
    all_firing_rates = []
    for p in probes:
        p_idx = p - 1  # Convert to 0-indexed
        if p_idx >= len(session['clu']):
            print(f'    Warning: probe {p} not found, skipping')
            continue
        fr, kept = process_spikes(session, p_idx, time_axis)
        if fr is not None and len(kept) > 0:
            all_firing_rates.append(fr)
            print(f'    Probe {p}: {fr.shape[0]} units kept')
    
    if len(all_firing_rates) == 0:
        print(f'    EXCLUDED: no valid units')
        return None
    
    # Concatenate across probes
    firing_rates = np.concatenate(all_firing_rates, axis=0)  # (n_neurons, n_time, n_trials)
    n_neurons = firing_rates.shape[0]
    t_spk = time.time() - t0
    print(f'    Spike processing: {t_spk:.1f}s, {n_neurons} total units')
    
    if n_neurons < MIN_UNITS:
        print(f'    EXCLUDED: only {n_neurons} units (need >={MIN_UNITS})')
        return None
    
    # ---- Compute video offset ----
    vidshift = compute_video_offset(session)
    
    # ---- Extract tongue velocity ----
    t0 = time.time()
    tongue_vel = extract_feature_velocity(session, 0, 'tongue', time_axis, vidshift)
    t_tongue = time.time() - t0
    print(f'    Tongue velocity: {t_tongue:.1f}s')
    
    # ---- Extract paw velocity (top_paw from camera 1) ----
    t0 = time.time()
    paw_vel = extract_feature_velocity(session, 1, 'top_paw', time_axis, vidshift)
    t_paw = time.time() - t0
    print(f'    Paw velocity: {t_paw:.1f}s')
    
    # ---- Load and align motion energy ----
    t0 = time.time()
    if sess_info['me_filepath'] is not None:
        me_data = load_motion_energy(sess_info['me_filepath'])
        if me_data is not None:
            me_aligned = align_motion_energy(me_data, session, time_axis, vidshift)
        else:
            me_aligned = np.full((len(time_axis), session['bp']['Ntrials']), np.nan)
    else:
        me_aligned = np.full((len(time_axis), session['bp']['Ntrials']), np.nan)
    t_me = time.time() - t0
    print(f'    Motion energy: {t_me:.1f}s')
    
    # ---- Select valid trials ----
    # Include all trials that are not early and not stim
    bp = session['bp']
    valid_trials = ~bp['early'] & ~bp['stim_enable']
    # Also exclude 'no' (ignore) trials
    valid_trials = valid_trials & ~bp['no']
    
    trial_indices = np.where(valid_trials)[0]
    
    if len(trial_indices) < 2:
        print(f'    EXCLUDED: only {len(trial_indices)} valid trials')
        return None
    
    print(f'    Valid trials: {len(trial_indices)} / {bp["Ntrials"]}')
    
    # ---- Compute per-session thresholds for discretization ----
    # Tongue velocity threshold: 50th percentile of all valid trial data
    tongue_valid = tongue_vel[:, trial_indices]
    tongue_thresh = np.nanpercentile(tongue_valid[~np.isnan(tongue_valid)], 50) if np.any(~np.isnan(tongue_valid)) else 0
    
    paw_valid = paw_vel[:, trial_indices]
    paw_thresh = np.nanpercentile(paw_valid[~np.isnan(paw_valid)], 50) if np.any(~np.isnan(paw_valid)) else 0
    
    me_valid = me_aligned[:, trial_indices]
    me_thresh = np.nanpercentile(me_valid[~np.isnan(me_valid)], 50) if np.any(~np.isnan(me_valid)) else 0
    
    print(f'    Thresholds: tongue={tongue_thresh:.2f}, paw={paw_thresh:.2f}, ME={me_thresh:.4f}')
    
    # ---- Build trial-level data ----
    neural_trials = []
    input_trials = []
    output_trials = []
    
    for trial_idx in trial_indices:
        # Neural: (n_neurons, n_timepoints)
        neural = firing_rates[:, :, trial_idx].astype(np.float32)
        neural_trials.append(neural)
        
        # Input: time from goCue (same for all trials)
        inp = time_axis.astype(np.float32).reshape(1, -1)  # (1, n_timepoints)
        input_trials.append(inp)
        
        # Output
        # Lick direction: R=1, L=0
        lick_dir = 1 if bp['R'][trial_idx] else 0
        
        # Behavioral context: WC=0, DR=1
        context = 0 if bp['autowater'][trial_idx] else 1
        
        # Outcome: correct=1, incorrect=0
        outcome = 1 if bp['hit'][trial_idx] else 0
        
        # Tongue velocity discretized (time-varying)
        tv = tongue_vel[:, trial_idx].copy()
        tv_disc = np.zeros(len(time_axis), dtype=np.int64)
        tv_disc[tv >= tongue_thresh] = 1
        # Handle NaN: set to 0 (below threshold)
        tv_disc[np.isnan(tv)] = 0
        
        # Paw velocity discretized (time-varying)
        pv = paw_vel[:, trial_idx].copy()
        pv_disc = np.zeros(len(time_axis), dtype=np.int64)
        pv_disc[pv >= paw_thresh] = 1
        pv_disc[np.isnan(pv)] = 0
        
        # Motion energy discretized (time-varying)
        me = me_aligned[:, trial_idx].copy()
        me_disc = np.zeros(len(time_axis), dtype=np.int64)
        me_disc[me >= me_thresh] = 1
        me_disc[np.isnan(me)] = 0
        
        # Stack outputs: (n_output, n_timepoints) for time-varying,
        # or (n_output,) for per-trial
        # Per-trial: lick_dir, context, outcome
        # Time-varying: tongue_vel, paw_vel, motion_energy
        out = np.zeros((6, len(time_axis)), dtype=np.int64)
        out[0, :] = lick_dir  # constant across time
        out[1, :] = context   # constant across time
        out[2, :] = outcome   # constant across time
        out[3, :] = tv_disc
        out[4, :] = pv_disc
        out[5, :] = me_disc
        
        output_trials.append(out)
    
    # ---- Visualization ----
    if show_processing:
        plot_processing(animal, date, session, firing_rates, tongue_vel, paw_vel,
                       me_aligned, time_axis, trial_indices, tongue_thresh, paw_thresh, me_thresh)
    
    result = {
        'neural': neural_trials,
        'input': input_trials,
        'output': output_trials,
        'animal': animal,
        'date': date,
        'n_neurons': n_neurons,
        'n_trials': len(trial_indices),
    }
    
    return result


def plot_processing(animal, date, session, firing_rates, tongue_vel, paw_vel,
                   me_aligned, time_axis, trial_indices, tongue_thresh, paw_thresh, me_thresh):
    """Plot processing steps for visual verification."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(4, 3, figsize=(20, 16))
    fig.suptitle(f'{animal}_{date}', fontsize=14)
    
    bp = session['bp']
    
    # Row 1: Neural activity examples
    n_neurons = firing_rates.shape[0]
    example_neurons = np.random.choice(n_neurons, min(3, n_neurons), replace=False)
    for col, neuron_idx in enumerate(example_neurons):
        ax = axes[0, col]
        # Plot a few trials
        n_show = min(10, len(trial_indices))
        for t in range(n_show):
            trix = trial_indices[t]
            ax.plot(time_axis, firing_rates[neuron_idx, :, trix], alpha=0.3)
        ax.axvline(0, color='k', linestyle='--', alpha=0.5, label='Go cue')
        ax.set_title(f'Neuron {neuron_idx}')
        ax.set_xlabel('Time from goCue (s)')
        ax.set_ylabel('Firing rate (Hz)')
    
    # Row 2: Tongue velocity
    ax = axes[1, 0]
    for t in range(min(10, len(trial_indices))):
        trix = trial_indices[t]
        ax.plot(time_axis, tongue_vel[:, trix], alpha=0.3)
    ax.axhline(tongue_thresh, color='r', linestyle='--', label=f'50th pct={tongue_thresh:.1f}')
    ax.axvline(0, color='k', linestyle='--', alpha=0.5)
    ax.set_title('Tongue velocity')
    ax.legend()
    
    # Row 2: Paw velocity
    ax = axes[1, 1]
    for t in range(min(10, len(trial_indices))):
        trix = trial_indices[t]
        ax.plot(time_axis, paw_vel[:, trix], alpha=0.3)
    ax.axhline(paw_thresh, color='r', linestyle='--', label=f'50th pct={paw_thresh:.1f}')
    ax.axvline(0, color='k', linestyle='--', alpha=0.5)
    ax.set_title('Paw velocity')
    ax.legend()
    
    # Row 2: Motion energy
    ax = axes[1, 2]
    for t in range(min(10, len(trial_indices))):
        trix = trial_indices[t]
        ax.plot(time_axis, me_aligned[:, trix], alpha=0.3)
    ax.axhline(me_thresh, color='r', linestyle='--', label=f'50th pct={me_thresh:.4f}')
    ax.axvline(0, color='k', linestyle='--', alpha=0.5)
    ax.set_title('Motion energy')
    ax.legend()
    
    # Row 3: Discretized outputs example trial
    ex_trix = trial_indices[0]
    ax = axes[2, 0]
    tv_disc = (tongue_vel[:, ex_trix] >= tongue_thresh).astype(int)
    ax.plot(time_axis, tv_disc, 'b-', label='Tongue vel disc')
    ax.axvline(0, color='k', linestyle='--', alpha=0.5)
    ax.set_title(f'Discretized tongue vel (trial {ex_trix})')
    ax.set_ylim(-0.1, 1.1)
    
    ax = axes[2, 1]
    pv_disc = (paw_vel[:, ex_trix] >= paw_thresh).astype(int)
    ax.plot(time_axis, pv_disc, 'g-', label='Paw vel disc')
    ax.axvline(0, color='k', linestyle='--', alpha=0.5)
    ax.set_title(f'Discretized paw vel (trial {ex_trix})')
    ax.set_ylim(-0.1, 1.1)
    
    ax = axes[2, 2]
    me_disc = (me_aligned[:, ex_trix] >= me_thresh).astype(int)
    ax.plot(time_axis, me_disc, 'r-', label='ME disc')
    ax.axvline(0, color='k', linestyle='--', alpha=0.5)
    ax.set_title(f'Discretized ME (trial {ex_trix})')
    ax.set_ylim(-0.1, 1.1)
    
    # Row 4: Trial outcome distributions
    ax = axes[3, 0]
    valid = trial_indices
    r_count = np.sum(bp['R'][valid])
    l_count = np.sum(bp['L'][valid])
    ax.bar(['Right', 'Left'], [r_count, l_count])
    ax.set_title('Lick direction')
    
    ax = axes[3, 1]
    dr_count = np.sum(~bp['autowater'][valid])
    wc_count = np.sum(bp['autowater'][valid])
    ax.bar(['DR', 'WC'], [dr_count, wc_count])
    ax.set_title('Context')
    
    ax = axes[3, 2]
    hit_count = np.sum(bp['hit'][valid])
    miss_count = np.sum(bp['miss'][valid])
    ax.bar(['Hit', 'Miss'], [hit_count, miss_count])
    ax.set_title('Outcome')
    
    plt.tight_layout()
    plt.savefig(f'processing_{animal}_{date}.png', dpi=100)
    plt.close()
    print(f'    Saved processing_{animal}_{date}.png')


def main():
    parser = argparse.ArgumentParser(description='Convert Hasnain et al. 2024 data')
    parser.add_argument('output', help='Output pickle file path')
    parser.add_argument('--sample', action='store_true', help='Process only 2 sessions')
    parser.add_argument('--full', action='store_true', help='Process all sessions (default)')
    parser.add_argument('--show-processing', action='store_true', help='Plot processing visualizations')
    args = parser.parse_args()
    
    total_start = time.time()
    
    # Get available sessions
    all_sessions = get_available_sessions()
    print(f'Found {len(all_sessions)} total sessions')
    
    if args.sample:
        # Select 2 sessions for testing
        all_sessions = all_sessions[:2]
        print(f'Sample mode: processing {len(all_sessions)} sessions')
    
    # Create time axis
    time_axis = make_time_axis()
    print(f'Time axis: {len(time_axis)} bins, [{time_axis[0]:.3f}, {time_axis[-1]:.3f}] s')
    
    # Process all sessions
    results = []
    for i, sess_info in enumerate(all_sessions):
        print(f'\nSession {i+1}/{len(all_sessions)}: {sess_info["animal"]}_{sess_info["date"]}')
        t0 = time.time()
        
        try:
            result = process_session(sess_info, time_axis, show_processing=args.show_processing)
            if result is not None:
                results.append(result)
                print(f'  -> Included: {result["n_neurons"]} neurons, {result["n_trials"]} trials')
            else:
                print(f'  -> Excluded')
        except Exception as e:
            print(f'  -> ERROR: {e}')
            import traceback
            traceback.print_exc()
        
        elapsed = time.time() - t0
        print(f'  Session time: {elapsed:.1f}s')
    
    print(f'\n{len(results)} sessions included')
    
    if len(results) == 0:
        print('ERROR: No sessions passed inclusion criteria!')
        sys.exit(1)
    
    # ---- Assemble final data structure ----
    print('\nAssembling final data structure...')
    
    # Collect all unique subjects
    subjects = sorted(set(r['animal'] for r in results))
    subject_idx = np.array([subjects.index(r['animal']) for r in results])
    
    # Brain regions - all ALM
    brain_regions = ['ALM']
    brain_region_idx = [np.zeros(r['n_neurons'], dtype=np.int64) for r in results]
    
    # Build lists
    neural = [r['neural'] for r in results]
    inputs = [r['input'] for r in results]
    outputs = [r['output'] for r in results]
    
    data = {
        'neural': neural,
        'input': inputs,
        'output': outputs,
        'subjects': subjects,
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
            'task_description': 'Two-context paradigm: delayed-response (DR) and water-cued (WC) licking tasks alternating block-wise. Mice perform directional tongue movements to left or right targets.',
            'time_bin_size': DT * 1000,  # 5 ms
            'temporal_alignment_event': 'Go cue onset',
            'off_start': TMIN,  # -2.5 seconds
            'off_end': TMAX,    # 2.5 seconds
            'smoothing': f'Causal Gaussian, window={SMOOTH_N} bins ({SMOOTH_N * DT * 1000:.0f} ms)',
            'firing_rate_threshold': LOW_FR,
            'brain_region': 'Anterior Lateral Motor cortex (ALM)',
            'species': 'Mus musculus',
            'reference': 'Hasnain, Birnbaum et al, Nature Neuroscience 2024',
            'session_info': [{'animal': r['animal'], 'date': r['date'],
                            'n_neurons': r['n_neurons'], 'n_trials': r['n_trials']}
                           for r in results],
        }
    }
    
    # Print summary
    total_neurons = sum(r['n_neurons'] for r in results)
    total_trials = sum(r['n_trials'] for r in results)
    print(f'\nSummary:')
    print(f'  Sessions: {len(results)}')
    print(f'  Subjects: {len(subjects)}')
    print(f'  Total neurons: {total_neurons}')
    print(f'  Total trials: {total_trials}')
    print(f'  Mean neurons/session: {total_neurons/len(results):.1f}')
    print(f'  Mean trials/session: {total_trials/len(results):.1f}')
    print(f'  Time bins per trial: {len(time_axis)}')
    
    # Save
    print(f'\nSaving to {args.output}...')
    with open(args.output, 'wb') as f:
        pickle.dump(data, f, protocol=4)
    
    file_size = os.path.getsize(args.output)
    print(f'Saved: {file_size / 1e6:.1f} MB')
    
    total_time = time.time() - total_start
    print(f'\nTotal time: {total_time:.1f}s ({total_time/60:.1f} min)')


if __name__ == '__main__':
    main()
