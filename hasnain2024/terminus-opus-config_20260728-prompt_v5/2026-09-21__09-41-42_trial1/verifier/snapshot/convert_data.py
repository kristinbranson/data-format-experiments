#!/usr/bin/env python3
"""Convert Hasnain, Birnbaum et al. data to decoder-compatible format.

Usage:
    python -u convert_data.py <outpicklefile> [--full|--sample] [--show-processing]
"""

import os
import sys
import time
import argparse
import pickle
import numpy as np
import h5py
import scipy.io
from scipy.signal import windows as sig_windows
from scipy import stats as scipy_stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PARAMETERS (matching reference code and paper)
# ============================================================
PARAMS = {
    'alignEvent': 'goCue',
    'tmin': -2.5,
    'tmax': 2.5,
    'dt': 1/200,  # 5ms bins
    'smooth': 15,  # causal gaussian kernel window size in samples
    'bctype': 'reflect',
    'lowFR': 1.0,  # Hz, minimum firing rate threshold (paper says 1 Hz)
    'quality_exclude': ['garbage', 'gabrga', 'noisy', 'real?'],
    'advance_movement': 0.0,
}

# Time axis
EDGES = np.arange(PARAMS['tmin'], PARAMS['tmax'] + PARAMS['dt'], PARAMS['dt'])
TIME_AXIS = EDGES[:-1] + PARAMS['dt'] / 2
N_TIMEPOINTS = len(TIME_AXIS)

# ============================================================
# SESSION METADATA
# ============================================================
SESSION_META = {
    ('EKH1', '2021-08-07'): ([2], 'Ephys_Behavior'),
    ('EKH3', '2021-08-11'): ([2], 'Ephys_Behavior'),
    ('JEB6', '2021-04-18'): ([2], 'Ephys_Behavior'),
    ('JEB7', '2021-04-29'): ([1], 'Ephys_Behavior'),
    ('JEB7', '2021-04-30'): ([1], 'Ephys_Behavior'),
    ('JGR2', '2021-11-16'): ([1], 'Ephys_Behavior'),
    ('JGR2', '2021-11-17'): ([1], 'Ephys_Behavior'),
    ('JGR3', '2021-11-18'): ([1], 'Ephys_Behavior'),
    ('JEB13', '2022-09-13'): ([2], 'Ephys_Behavior'),
    ('JEB13', '2022-09-14'): ([2], 'Ephys_Behavior'),
    ('JEB13', '2022-09-21'): ([1], 'Ephys_Behavior'),
    ('JEB13', '2022-09-24'): ([1], 'Ephys_Behavior'),
    ('JEB13', '2022-09-25'): ([1], 'Ephys_Behavior'),
    ('JEB14', '2022-08-22'): ([1], 'Ephys_Behavior'),
    ('JEB14', '2022-08-23'): ([1], 'Ephys_Behavior'),
    ('JEB14', '2022-08-24'): ([1], 'Ephys_Behavior'),
    ('JEB14', '2022-08-25'): ([1], 'Ephys_Behavior'),
    ('JEB15', '2022-07-26'): ([1, 2], 'Ephys_Behavior'),
    ('JEB15', '2022-07-27'): ([1, 2], 'Ephys_Behavior'),
    ('JEB15', '2022-07-28'): ([1, 2], 'Ephys_Behavior'),
    ('JEB15', '2022-07-29'): ([2], 'Ephys_Behavior'),
    ('JEB19', '2023-04-18'): ([1], 'Ephys_Behavior'),
    ('JEB19', '2023-04-19'): ([1], 'Ephys_Behavior'),
    ('JEB19', '2023-04-20'): ([1], 'Ephys_Behavior'),
    ('JEB19', '2023-04-21'): ([1], 'Ephys_Behavior'),
    ('JEB11', '2022-05-10'): ([1], 'RandomizedDelay_Ephys_Behavior'),
    ('JEB11', '2022-05-11'): ([1], 'RandomizedDelay_Ephys_Behavior'),
    ('JEB12', '2022-05-12'): ([1], 'RandomizedDelay_Ephys_Behavior'),
    ('JEB12', '2022-05-13'): ([1], 'RandomizedDelay_Ephys_Behavior'),
    ('JEB23', '2023-10-10'): ([1], 'RandomizedDelay_Ephys_Behavior'),
    ('JEB23', '2023-10-11'): ([1], 'RandomizedDelay_Ephys_Behavior'),
    ('JEB23', '2023-10-12'): ([1], 'RandomizedDelay_Ephys_Behavior'),
    ('JEB23', '2023-10-13'): ([1], 'RandomizedDelay_Ephys_Behavior'),
    ('JEB23', '2023-10-18'): ([1], 'RandomizedDelay_Ephys_Behavior'),
    ('JEB23', '2023-10-19'): ([1], 'RandomizedDelay_Ephys_Behavior'),
    ('JEB23', '2023-10-20'): ([1], 'RandomizedDelay_Ephys_Behavior'),
    ('JEB23', '2023-10-21'): ([1], 'RandomizedDelay_Ephys_Behavior'),
    ('JEB24', '2023-10-23'): ([1], 'RandomizedDelay_Ephys_Behavior'),
    ('JEB24', '2023-10-24'): ([1], 'RandomizedDelay_Ephys_Behavior'),
    ('JEB24', '2023-10-25'): ([1], 'RandomizedDelay_Ephys_Behavior'),
    ('JEB24', '2023-10-26'): ([1], 'RandomizedDelay_Ephys_Behavior'),
    ('JEB24', '2023-10-27'): ([1], 'RandomizedDelay_Ephys_Behavior'),
    ('JEB24', '2023-10-31'): ([1], 'RandomizedDelay_Ephys_Behavior'),
    ('JEB24', '2023-11-02'): ([1], 'RandomizedDelay_Ephys_Behavior'),
    ('JEB24', '2023-11-03'): ([1], 'RandomizedDelay_Ephys_Behavior'),
}

DATA_DIR = '/app/data'


# ============================================================
# UNIFIED DATA ACCESS LAYER
# ============================================================
class SessionData:
    """Unified interface for accessing session data from HDF5 or v5 MATLAB files."""
    
    def __init__(self, data_file):
        self.data_file = data_file
        self.is_hdf5 = None
        self.f = None  # HDF5 file handle
        self.obj = None  # v5 obj struct
        self._open()
    
    def _open(self):
        try:
            self.f = h5py.File(self.data_file, 'r')
            self.is_hdf5 = True
        except:
            data = scipy.io.loadmat(self.data_file, squeeze_me=False)
            self.obj = data['obj']
            self.is_hdf5 = False
    
    def close(self):
        if self.f is not None:
            self.f.close()
    
    def get_n_trials(self):
        if self.is_hdf5:
            return int(np.array(self.f['obj/bp/Ntrials']).flatten()[0])
        else:
            return int(self.obj['bp'][0,0]['Ntrials'][0,0].flatten()[0])
    
    def get_bp_field(self, name):
        """Get a behavioral protocol field as a 1D array."""
        if self.is_hdf5:
            return np.array(self.f[f'obj/bp/{name}']).flatten()
        else:
            return self.obj['bp'][0,0][name][0,0].flatten()
    
    def get_event(self, name):
        """Get an event time array."""
        if self.is_hdf5:
            return np.array(self.f[f'obj/bp/ev/{name}']).flatten()
        else:
            return self.obj['bp'][0,0]['ev'][0,0][name][0,0].flatten()
    
    def get_stim_enable(self):
        if self.is_hdf5:
            return np.array(self.f['obj/bp/stim/enable']).flatten()
        else:
            return self.obj['bp'][0,0]['stim'][0,0]['enable'][0,0].flatten()
    
    def get_n_probes(self):
        if self.is_hdf5:
            return self.f['obj/clu'].shape[0]
        else:
            return self.obj['clu'][0,0].shape[0]
    
    def get_n_clusters(self, probe_idx):
        """Get number of clusters for a probe (0-indexed)."""
        if self.is_hdf5:
            ref = self.f['obj/clu'][probe_idx, 0]
            clu = self.f[ref]
            return clu['quality'].shape[0]
        else:
            clu = self.obj['clu'][0,0][0, probe_idx]
            return clu.shape[1] if clu.ndim == 2 else clu.shape[0]
    
    def get_cluster_qualities(self, probe_idx):
        """Get quality labels for all clusters."""
        if self.is_hdf5:
            ref = self.f['obj/clu'][probe_idx, 0]
            clu = self.f[ref]
            qualities = []
            for i in range(clu['quality'].shape[0]):
                qref = clu['quality'][i, 0]
                try:
                    q = self.f[qref]
                    arr = np.array(q).flatten()
                    if arr.dtype.kind in ('U', 'S', 'O'):
                        qstr = str(arr[0])
                    else:
                        qstr = ''.join(chr(int(c)) for c in arr)
                    qualities.append(qstr.strip())
                except:
                    qualities.append('')
            return qualities
        else:
            clu_arr = self.obj['clu'][0,0][0, probe_idx]
            n_clu = clu_arr.shape[1] if clu_arr.ndim == 2 else clu_arr.shape[0]
            qualities = []
            for i in range(n_clu):
                if clu_arr.ndim == 2:
                    c = clu_arr[0, i]
                else:
                    c = clu_arr[i]
                q = c['quality']
                if isinstance(q, np.ndarray):
                    q = q.flatten()
                    if len(q) > 0:
                        qstr = str(q[0]).strip()
                    else:
                        qstr = ''
                else:
                    qstr = str(q).strip()
                qualities.append(qstr)
            return qualities
    
    def get_spike_data(self, probe_idx, cluster_idx):
        """Get trial and trialtm for a cluster. Returns (trial_1indexed, trialtm)."""
        if self.is_hdf5:
            ref = self.f['obj/clu'][probe_idx, 0]
            clu = self.f[ref]
            trial_ref = clu['trial'][cluster_idx, 0]
            trialtm_ref = clu['trialtm'][cluster_idx, 0]
            trial = np.array(self.f[trial_ref]).flatten()
            trialtm = np.array(self.f[trialtm_ref]).flatten()
            return trial, trialtm
        else:
            clu_arr = self.obj['clu'][0,0][0, probe_idx]
            if clu_arr.ndim == 2:
                c = clu_arr[0, cluster_idx]
            else:
                c = clu_arr[cluster_idx]
            trial = c['trial'].flatten()
            trialtm = c['trialtm'].flatten()
            return trial, trialtm
    
    def get_video_offset(self):
        """Compute video offset."""
        try:
            bitStart = self.get_event('bitStart')
            bitstart_mode = float(scipy_stats.mode(bitStart, keepdims=False).mode)
            
            if self.is_hdf5:
                sglx_bitstart = np.array(self.f['obj/sglx/bitcode/bitstart']).flatten()
                sglx_fs = np.array(self.f['obj/sglx/fs']).flatten()[0]
            else:
                sglx = self.obj['sglx'][0,0]
                sglx_bitstart = sglx['bitcode'][0,0]['bitstart'][0,0].flatten()
                sglx_fs = sglx['fs'][0,0].flatten()[0]
            
            sglx_bitstart_mode = float(scipy_stats.mode(sglx_bitstart, keepdims=False).mode)
            return sglx_bitstart_mode / sglx_fs - bitstart_mode
        except:
            return 0.0
    
    def get_traj_n_cameras(self):
        if self.is_hdf5:
            return self.f['obj/traj'].shape[0]
        else:
            return self.obj['traj'][0,0].shape[1]
    
    def get_traj_feat_names(self, cam_idx, trial_idx=0):
        """Get feature names for a camera."""
        if self.is_hdf5:
            cam = self.f[self.f['obj/traj'][cam_idx, 0]]
            feat_ds = cam['featNames']
            for ti in range(feat_ds.shape[0]):
                try:
                    ref = feat_ds[ti, 0]
                    feat_arr = self.f[ref]
                    if feat_arr.dtype == object:
                        names = []
                        for fi in range(feat_arr.shape[1]):
                            fref = feat_arr[0, fi]
                            target = self.f[fref]
                            arr = np.array(target).flatten()
                            if arr.dtype.kind in ('U', 'S', 'O'):
                                names.append(str(arr[0]))
                            else:
                                names.append(''.join(chr(int(c)) for c in arr))
                        return names
                except:
                    continue
            return []
        else:
            cam = self.obj['traj'][0,0][0, cam_idx]
            for ti in range(cam.shape[1]):
                try:
                    t = cam[0, ti]
                    fn = t['featNames']
                    if fn.dtype == object:
                        names = []
                        for fi in range(fn.shape[0]):
                            name = fn[fi, 0]
                            if isinstance(name, np.ndarray):
                                name = name.flatten()[0]
                            names.append(str(name))
                        return names
                    else:
                        return [str(fn.flatten()[0])]
                except:
                    continue
            return []
    
    def get_traj_trial(self, cam_idx, trial_idx):
        """Get tracking data for a trial.
        Returns: (ts, frameTimes, n_dropped) or (None, None, None) if invalid.
        ts shape: (n_frames, 3, n_features)
        frameTimes: (n_frames,)
        """
        if self.is_hdf5:
            cam = self.f[self.f['obj/traj'][cam_idx, 0]]
            try:
                ndf_ref = cam['NdroppedFrames'][trial_idx, 0]
                ndf = np.array(self.f[ndf_ref]).flatten()
                if np.any(np.isnan(ndf)):
                    return None, None, None
            except:
                pass
            
            try:
                ft_ref = cam['frameTimes'][trial_idx, 0]
                ft = np.array(self.f[ft_ref]).flatten()
                if np.all(np.isnan(ft)):
                    return None, None, None
                
                ts_ref = cam['ts'][trial_idx, 0]
                ts = np.array(self.f[ts_ref])
                
                # Ensure shape is (n_frames, 3, n_features)
                if ts.ndim == 3:
                    if ts.shape[0] != len(ft):
                        # (n_features, 3, n_frames) -> (n_frames, 3, n_features)
                        ts = ts.transpose(2, 1, 0)
                
                return ts, ft, 0
            except:
                return None, None, None
        else:
            cam = self.obj['traj'][0,0][0, cam_idx]
            try:
                t = cam[0, trial_idx]
                
                ndf = t['NdroppedFrames'].flatten()
                if np.any(np.isnan(ndf.astype(float))):
                    return None, None, None
                
                ft = t['frameTimes'].flatten()
                if np.all(np.isnan(ft)):
                    return None, None, None
                
                ts = t['ts']
                if isinstance(ts, np.ndarray) and ts.ndim >= 2:
                    # v5 format: (n_frames, 3, n_features)
                    if ts.ndim == 3 and ts.shape[0] == len(ft):
                        pass  # already correct
                    elif ts.ndim == 3 and ts.shape[2] == len(ft):
                        ts = ts.transpose(2, 1, 0)
                    elif ts.ndim == 2:
                        # Might be (3, n_features) for a single frame
                        return None, None, None
                else:
                    return None, None, None
                
                return ts, ft, int(ndf[0]) if len(ndf) > 0 else 0
            except:
                return None, None, None
    
    def get_motion_energy_data(self, me_file):
        """Load motion energy data. Returns (me_data_list, me_thresh) or (None, None)."""
        if not os.path.exists(me_file):
            return None, None
        try:
            me_mat = scipy.io.loadmat(me_file, squeeze_me=False)
            me_struct = me_mat['me']
            me_data_raw = me_struct['data'][0, 0]
            me_thresh = me_struct['moveThresh'][0, 0].flatten()[0]
            
            me_data = []
            for i in range(me_data_raw.shape[0]):
                d = me_data_raw[i, 0]
                if isinstance(d, np.ndarray):
                    me_data.append(d.flatten())
                else:
                    me_data.append(np.array([]))
            return me_data, float(me_thresh)
        except:
            return None, None


# ============================================================
# PROCESSING FUNCTIONS
# ============================================================

def causal_gaussian_smooth(x, N, bctype='reflect'):
    """Smooth with causal gaussian kernel, matching mySmooth.m."""
    if N <= 1:
        return x
    
    was_1d = False
    if x.ndim == 1:
        x = x[:, np.newaxis]
        was_1d = True
    
    if bctype == 'reflect':
        x_filt = np.concatenate([x[:N, :], x], axis=0)
        trim = N
    elif bctype == 'zeropad':
        x_filt = np.concatenate([np.zeros((N, x.shape[1])), x], axis=0)
        trim = N
    else:
        x_filt = x.copy()
        trim = 0
    
    # MATLAB gausswin(N) with default alpha=2.5: std = (N-1)/(2*alpha)
    kern = sig_windows.gaussian(N, std=(N-1)/(2*2.5))
    kern[:N//2] = 0  # Make causal
    kern = kern / kern.sum()
    
    out = np.zeros_like(x_filt)
    for j in range(x_filt.shape[1]):
        out[:, j] = np.convolve(x_filt[:, j], kern, mode='same')
    
    out = out[trim:, :]
    if was_1d:
        out = out[:, 0]
    return out


def find_clusters(qualities, exclude_list):
    """Find cluster indices to use."""
    exclude_lower = [e.lower() for e in exclude_list]
    indices = []
    for i, q in enumerate(qualities):
        q_clean = q.lower().strip().replace('\x00', '')
        if q_clean == '' or q_clean in exclude_lower:
            continue
        indices.append(i)
    return np.array(indices, dtype=int)


def compute_firing_rates(sd, probe_idx, cluster_indices, go_cue, n_trials):
    """Compute single-trial firing rates."""
    n_time = N_TIMEPOINTS
    n_neurons = len(cluster_indices)
    dt = PARAMS['dt']
    
    trialdat = np.zeros((n_time, n_neurons, n_trials), dtype=np.float32)
    
    for neuron_idx, clu_idx in enumerate(cluster_indices):
        trial, trialtm = sd.get_spike_data(probe_idx, clu_idx)
        
        # Align to goCue
        trial_int = trial.astype(int)
        valid_mask = (trial_int >= 1) & (trial_int <= n_trials)
        trial_int = trial_int[valid_mask]
        trialtm = trialtm[valid_mask]
        trialtm_aligned = trialtm - go_cue[trial_int - 1]
        
        for t in range(n_trials):
            spk_mask = trial_int == (t + 1)
            if not np.any(spk_mask):
                continue
            
            spk_times = trialtm_aligned[spk_mask]
            N_counts, _ = np.histogram(spk_times, bins=EDGES)
            N_counts = N_counts[:n_time].astype(np.float64)
            fr = N_counts / dt
            fr_smooth = causal_gaussian_smooth(fr, PARAMS['smooth'], PARAMS['bctype'])
            trialdat[:, neuron_idx, t] = fr_smooth
    
    return trialdat


def extract_feature_velocity(sd, cam_idx, feat_name, go_cue, n_trials, is_tongue=False):
    """Extract velocity for a tracked feature."""
    taxis = TIME_AXIS + PARAMS['advance_movement']
    n_time = len(taxis)
    
    speed = np.full((n_time, n_trials), np.nan, dtype=np.float32)
    visible = np.zeros((n_time, n_trials), dtype=bool)
    
    # Find feature index
    feat_names = sd.get_traj_feat_names(cam_idx)
    if not feat_names:
        return speed, visible
    
    feat_idx = None
    for fi, fn in enumerate(feat_names):
        if fn == feat_name:
            feat_idx = fi
            break
    
    if feat_idx is None:
        return speed, visible
    
    vidshift = sd.get_video_offset()
    
    for t in range(n_trials):
        try:
            ts, ft, ndf = sd.get_traj_trial(cam_idx, t)
            if ts is None or ft is None:
                continue
            if len(ft) < 10:  # Skip trials with too few frames
                continue
            
            # ts shape: (n_frames, 3, n_features)
            x = ts[:, 0, feat_idx].astype(float)
            y = ts[:, 1, feat_idx].astype(float)
            
            # Align frame times to goCue
            aligned_ft = ft - vidshift - go_cue[t]
            
            # Interpolate to neural time axis
            # Use NaN outside range
            x_interp = np.interp(taxis, aligned_ft, x)
            y_interp = np.interp(taxis, aligned_ft, y)
            
            # Mark as outside video range
            outside = (taxis < aligned_ft[0]) | (taxis > aligned_ft[-1])
            x_interp[outside] = np.nan
            y_interp[outside] = np.nan
            
            # For tongue: NaN in raw data means not visible
            if is_tongue:
                # DLC marks low-confidence as NaN
                vis = ~np.isnan(x_interp) & ~np.isnan(y_interp)
                visible[:, t] = vis
                
                xvel = np.gradient(x_interp)
                yvel = np.gradient(y_interp)
                xvel[~vis] = 0
                yvel[~vis] = 0
            else:
                vis = ~np.isnan(x_interp) & ~np.isnan(y_interp)
                visible[:, t] = vis
                
                xvel = np.gradient(x_interp)
                yvel = np.gradient(y_interp)
                
                basederiv_x = np.nanmedian(np.diff(x_interp))
                basederiv_y = np.nanmedian(np.diff(y_interp))
                if not np.isnan(basederiv_x):
                    xvel -= basederiv_x
                if not np.isnan(basederiv_y):
                    yvel -= basederiv_y
                
                # Fill missing with nearest
                for arr in [xvel, yvel]:
                    mask_nan = np.isnan(arr)
                    if np.any(mask_nan) and not np.all(mask_nan):
                        valid = ~mask_nan
                        arr[mask_nan] = np.interp(np.where(mask_nan)[0], np.where(valid)[0], arr[valid])
            
            spd = np.sqrt(xvel**2 + yvel**2)
            speed[:, t] = spd
            
        except Exception:
            continue
    
    return speed, visible


def extract_motion_energy(sd, me_file, go_cue, n_trials):
    """Load and align motion energy."""
    taxis = TIME_AXIS + PARAMS['advance_movement']
    n_time = len(taxis)
    
    me_aligned = np.full((n_time, n_trials), np.nan, dtype=np.float32)
    
    me_data, me_thresh = sd.get_motion_energy_data(me_file)
    if me_data is None:
        return me_aligned, False
    
    vidshift = sd.get_video_offset()
    
    for t in range(min(n_trials, len(me_data))):
        try:
            me_trial = me_data[t]
            if len(me_trial) == 0:
                continue
            
            ts, ft, ndf = sd.get_traj_trial(0, t)  # Camera 0 for frame times
            if ft is None or len(ft) < 10:
                # Try creating frame times from length
                ft = np.arange(len(me_trial)) / 400.0
                aligned_ft = ft - 0.5 - go_cue[t]  # Fallback alignment
            else:
                aligned_ft = ft - vidshift - go_cue[t]
            
            if len(me_trial) > len(aligned_ft):
                me_trial = me_trial[:len(aligned_ft)]
            elif len(me_trial) < len(aligned_ft):
                aligned_ft = aligned_ft[:len(me_trial)]
            
            me_aligned[:, t] = np.interp(taxis, aligned_ft, me_trial)
        except:
            continue
    
    # Fill NaN with nearest
    for t in range(n_trials):
        col = me_aligned[:, t]
        mask = np.isnan(col)
        if np.any(mask) and not np.all(mask):
            valid = ~mask
            col[mask] = np.interp(np.where(mask)[0], np.where(valid)[0], col[valid])
            me_aligned[:, t] = col
    
    return me_aligned, True


def discretize_velocity(speed, visible, percentile_thresh=50):
    """Discretize velocity: 0=low, 1=high, 2=not visible."""
    n_time, n_trials = speed.shape
    categories = np.full((n_time, n_trials), 2, dtype=np.int64)
    
    vis_mask = visible & ~np.isnan(speed)
    if np.sum(vis_mask) > 0:
        threshold = np.percentile(speed[vis_mask], percentile_thresh)
        categories[vis_mask & (speed < threshold)] = 0
        categories[vis_mask & (speed >= threshold)] = 1
    
    return categories


def discretize_motion_energy(me_data, has_video, percentile_thresh=50):
    """Discretize motion energy: 0=low, 1=high, 2=no video."""
    n_time, n_trials = me_data.shape
    if not has_video:
        return np.full((n_time, n_trials), 2, dtype=np.int64)
    
    categories = np.full((n_time, n_trials), 2, dtype=np.int64)
    valid_mask = ~np.isnan(me_data)
    if np.sum(valid_mask) > 0:
        threshold = np.percentile(me_data[valid_mask], percentile_thresh)
        categories[valid_mask & (me_data < threshold)] = 0
        categories[valid_mask & (me_data >= threshold)] = 1
    
    return categories


def process_session(anm, date, probes, dataset_type, show_processing=False):
    """Process a single session."""
    t_start = time.time()
    
    data_file = os.path.join(DATA_DIR, dataset_type, f'data_structure_{anm}_{date}.mat')
    me_file = os.path.join(DATA_DIR, dataset_type, f'motionEnergy_{anm}_{date}.mat')
    
    if not os.path.exists(data_file):
        print(f'  WARNING: Data file not found: {data_file}')
        return None
    
    print(f'  Loading {anm}_{date} (probes: {probes})...')
    
    sd = SessionData(data_file)
    
    # ---- Load behavioral data ----
    n_trials_total = sd.get_n_trials()
    L = sd.get_bp_field('L')
    R = sd.get_bp_field('R')
    autowater = sd.get_bp_field('autowater')
    early = sd.get_bp_field('early')
    hit = sd.get_bp_field('hit')
    miss = sd.get_bp_field('miss')
    no_resp = sd.get_bp_field('no')
    stim_enable = sd.get_stim_enable()
    go_cue = sd.get_event('goCue')
    
    # ---- Trial filtering ----
    valid_trials = (early == 0) & (no_resp == 0) & (stim_enable == 0)
    valid_trial_indices = np.where(valid_trials)[0]
    
    # Inclusion criteria: >40 right hit DR AND >40 left hit DR
    r_hit_dr = (R == 1) & (hit == 1) & (stim_enable == 0) & (autowater == 0) & (early == 0)
    l_hit_dr = (L == 1) & (hit == 1) & (stim_enable == 0) & (autowater == 0) & (early == 0)
    
    n_r_hit_dr = np.sum(r_hit_dr)
    n_l_hit_dr = np.sum(l_hit_dr)
    
    print(f'    Total trials: {n_trials_total}, Valid: {len(valid_trial_indices)}')
    print(f'    R hit DR: {n_r_hit_dr}, L hit DR: {n_l_hit_dr}')
    
    if n_r_hit_dr <= 40 or n_l_hit_dr <= 40:
        print(f'    EXCLUDED: insufficient DR hit trials (need >40 each)')
        sd.close()
        return None
    
    if len(valid_trial_indices) < 2:
        print(f'    EXCLUDED: fewer than 2 valid trials')
        sd.close()
        return None
    
    # ---- Process neural data ----
    all_trialdat = []
    
    for probe_num in probes:
        probe_idx = probe_num - 1
        
        try:
            qualities = sd.get_cluster_qualities(probe_idx)
        except Exception as e:
            print(f'    WARNING: Could not access probe {probe_num}: {e}')
            continue
        
        quality_indices = find_clusters(qualities, PARAMS['quality_exclude'])
        
        if len(quality_indices) == 0:
            print(f'    WARNING: No clusters passed quality filter for probe {probe_num}')
            continue
        
        print(f'    Probe {probe_num}: {len(quality_indices)} clusters passed quality filter (of {len(qualities)} total)')
        
        t_neural = time.time()
        trialdat = compute_firing_rates(sd, probe_idx, quality_indices, go_cue, n_trials_total)
        print(f'    Neural data computed in {time.time()-t_neural:.1f}s')
        
        # Remove low FR clusters
        mean_frs = np.mean(trialdat, axis=(0, 2))
        fr_mask = mean_frs > PARAMS['lowFR']
        
        n_before = len(quality_indices)
        trialdat = trialdat[:, fr_mask, :]
        
        print(f'    After FR filter: {np.sum(fr_mask)} clusters (removed {n_before - np.sum(fr_mask)})')
        
        all_trialdat.append(trialdat)
    
    if len(all_trialdat) == 0:
        print(f'    EXCLUDED: no neurons after filtering')
        sd.close()
        return None
    
    trialdat = np.concatenate(all_trialdat, axis=1)
    n_neurons = trialdat.shape[1]
    
    if n_neurons < 10:
        print(f'    EXCLUDED: only {n_neurons} neurons (need >= 10)')
        sd.close()
        return None
    
    print(f'    Total neurons: {n_neurons}')
    
    # ---- Trial-level variables ----
    lick_dir = np.full(n_trials_total, -1, dtype=np.int64)
    lick_dir[(hit == 1) & (L == 1)] = 0  # left
    lick_dir[(hit == 1) & (R == 1)] = 1  # right
    lick_dir[(miss == 1) & (L == 1)] = 1  # licked right (wrong)
    lick_dir[(miss == 1) & (R == 1)] = 0  # licked left (wrong)
    lick_dir[no_resp == 1] = 2  # none
    
    context = autowater.astype(np.int64)  # 0=DR, 1=WC
    
    outcome = np.full(n_trials_total, -1, dtype=np.int64)
    outcome[hit == 1] = 1  # correct
    outcome[miss == 1] = 0  # incorrect
    outcome[no_resp == 1] = 2  # ignore
    
    # ---- Video-based outputs ----
    t_video = time.time()
    
    tongue_speed, tongue_visible = extract_feature_velocity(sd, 0, 'tongue', go_cue, n_trials_total, is_tongue=True)
    paw_speed, paw_visible = extract_feature_velocity(sd, 1, 'bottom_paw', go_cue, n_trials_total, is_tongue=False)
    me_aligned, has_me = extract_motion_energy(sd, me_file, go_cue, n_trials_total)
    
    print(f'    Video data extracted in {time.time()-t_video:.1f}s')
    
    # Discretize
    tongue_cat = discretize_velocity(tongue_speed, tongue_visible, 50)
    paw_cat = discretize_velocity(paw_speed, paw_visible, 50)
    me_cat = discretize_motion_energy(me_aligned, has_me, 50)
    
    # ---- Format output ----
    neural_trials = []
    input_trials = []
    output_trials = []
    
    for t_idx in valid_trial_indices:
        neural_trials.append(trialdat[:, :, t_idx].T.astype(np.float32))
        input_trials.append(TIME_AXIS.reshape(1, -1).astype(np.float32))
        
        output_trial = np.zeros((6, N_TIMEPOINTS), dtype=np.int64)
        output_trial[0, :] = lick_dir[t_idx]
        output_trial[1, :] = context[t_idx]
        output_trial[2, :] = outcome[t_idx]
        output_trial[3, :] = tongue_cat[:, t_idx]
        output_trial[4, :] = paw_cat[:, t_idx]
        output_trial[5, :] = me_cat[:, t_idx]
        output_trials.append(output_trial)
    
    sd.close()
    
    elapsed = time.time() - t_start
    print(f'    Session processed in {elapsed:.1f}s, {len(neural_trials)} trials, {n_neurons} neurons')
    
    return {
        'neural': neural_trials,
        'input': input_trials,
        'output': output_trials,
        'animal': anm,
        'date': date,
        'n_neurons': n_neurons,
        'n_trials': len(neural_trials),
        'probes': probes,
        'dataset_type': dataset_type,
    }


def main():
    parser = argparse.ArgumentParser(description='Convert data to decoder format')
    parser.add_argument('outfile', type=str, help='Output pickle file path')
    parser.add_argument('--full', action='store_true', default=True)
    parser.add_argument('--sample', action='store_true')
    parser.add_argument('--show-processing', action='store_true')
    args = parser.parse_args()
    
    if args.sample:
        args.full = False
    
    print('='*60)
    print('Data Conversion: Hasnain, Birnbaum et al.')
    print('='*60)
    print(f'Output: {args.outfile}')
    print(f'Mode: {"sample" if args.sample else "full"}')
    print(f'Params: dt={PARAMS["dt"]*1000}ms, smooth={PARAMS["smooth"]}, lowFR={PARAMS["lowFR"]}Hz')
    print()
    
    sessions = []
    for (anm, date), (probes, dtype) in sorted(SESSION_META.items()):
        data_file = os.path.join(DATA_DIR, dtype, f'data_structure_{anm}_{date}.mat')
        if os.path.exists(data_file):
            sessions.append((anm, date, probes, dtype))
    
    print(f'Found {len(sessions)} sessions with data files')
    
    if args.sample:
        sample_sessions = []
        for s in sessions:
            if s[3] == 'Ephys_Behavior' and len(sample_sessions) < 1:
                sample_sessions.append(s)
            elif s[3] == 'RandomizedDelay_Ephys_Behavior' and len([x for x in sample_sessions if x[3] == 'RandomizedDelay_Ephys_Behavior']) < 1:
                sample_sessions.append(s)
            if len(sample_sessions) >= 2:
                break
        sessions = sample_sessions
        print(f'Sample mode: {len(sessions)} sessions')
    
    all_neural, all_input, all_output = [], [], []
    all_subjects = []
    subject_idx_list = []
    brain_region_idx_list = []
    session_info = []
    total_neurons = 0
    total_trials = 0
    
    t_total = time.time()
    
    for i, (anm, date, probes, dtype) in enumerate(sessions):
        print(f'\nSession {i+1}/{len(sessions)}: {anm}_{date}')
        result = process_session(anm, date, probes, dtype, args.show_processing)
        
        if result is None:
            continue
        
        all_neural.append(result['neural'])
        all_input.append(result['input'])
        all_output.append(result['output'])
        
        if anm not in all_subjects:
            all_subjects.append(anm)
        subject_idx_list.append(all_subjects.index(anm))
        brain_region_idx_list.append(np.zeros(result['n_neurons'], dtype=np.int64))
        
        session_info.append({
            'animal': anm, 'date': date,
            'n_neurons': result['n_neurons'], 'n_trials': result['n_trials'],
            'probes': probes, 'dataset_type': dtype,
        })
        
        total_neurons += result['n_neurons']
        total_trials += result['n_trials']
    
    elapsed_total = time.time() - t_total
    
    print(f'\n{"="*60}')
    print(f'Conversion complete in {elapsed_total:.1f}s')
    print(f'Sessions: {len(all_neural)}, Subjects: {len(all_subjects)} ({all_subjects})')
    print(f'Total neurons: {total_neurons}, Total trials: {total_trials}')
    
    if len(all_neural) == 0:
        print('ERROR: No sessions processed!')
        sys.exit(1)
    
    data = {
        'neural': all_neural,
        'input': all_input,
        'output': all_output,
        'subjects': all_subjects,
        'subject_idx': np.array(subject_idx_list, dtype=np.int64),
        'brain_regions': ['ALM'],
        'brain_region_idx': brain_region_idx_list,
        'input_names': ['time_from_go_cue'],
        'output_names': ['lick_direction', 'context', 'outcome', 'tongue_velocity', 'paw_velocity', 'motion_energy'],
        'output_values': [
            ['left', 'right', 'none'],
            ['DR', 'WC'],
            ['incorrect', 'correct', 'ignore'],
            ['low', 'high', 'not_visible'],
            ['low', 'high', 'not_visible'],
            ['low', 'high', 'no_video'],
        ],
        'metadata': {
            'task_description': 'Two-context task: delayed response (DR) with auditory cue and water-cued (WC) tasks alternating block-wise. Mice perform directional licking.',
            'time_bin_size': PARAMS['dt'] * 1000,
            'temporal_alignment_event': 'Go cue onset (auditory chirp in DR, water presentation in WC)',
            'off_start': PARAMS['tmin'],
            'off_end': PARAMS['tmax'],
            'smoothing': f'{PARAMS["smooth"]}-sample causal gaussian kernel',
            'firing_rate_threshold': PARAMS['lowFR'],
            'session_info': session_info,
            'source': 'Hasnain, Birnbaum et al. Nature Neuroscience 2024',
        }
    }
    
    print(f'\nSaving to {args.outfile}...')
    with open(args.outfile, 'wb') as fout:
        pickle.dump(data, fout, protocol=4)
    
    file_size = os.path.getsize(args.outfile) / 1e6
    print(f'Saved ({file_size:.1f} MB)')
    
    # Processing plots
    if args.show_processing and len(all_neural) > 0:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        for sess_idx in range(min(2, len(all_neural))):
            info = session_info[sess_idx]
            fig, axes = plt.subplots(4, 2, figsize=(16, 20))
            fig.suptitle(f'Processing: {info["animal"]}_{info["date"]}')
            
            trial_idx = 0
            neural = all_neural[sess_idx][trial_idx]
            
            ax = axes[0, 0]
            n_show = min(20, neural.shape[0])
            ax.imshow(neural[:n_show, :], aspect='auto', extent=[TIME_AXIS[0], TIME_AXIS[-1], n_show, 0])
            ax.set_title(f'Neural (first {n_show} neurons)'); ax.set_xlabel('Time (s)'); ax.axvline(0, color='r', ls='--', alpha=0.5)
            
            ax = axes[0, 1]
            ax.plot(TIME_AXIS, np.mean(neural, axis=0))
            ax.set_title('Mean FR'); ax.set_xlabel('Time (s)'); ax.axvline(0, color='r', ls='--', alpha=0.5)
            
            ax = axes[1, 0]
            outputs = np.array([o[0, 0] for o in all_output[sess_idx]])
            ax.bar(['left', 'right', 'none'], [np.sum(outputs==i) for i in range(3)])
            ax.set_title('Lick direction')
            
            ax = axes[1, 1]
            contexts = np.array([o[1, 0] for o in all_output[sess_idx]])
            ax.bar(['DR', 'WC'], [np.sum(contexts==i) for i in range(2)])
            ax.set_title('Context')
            
            ax = axes[2, 0]
            outcomes = np.array([o[2, 0] for o in all_output[sess_idx]])
            ax.bar(['incorrect', 'correct', 'ignore'], [np.sum(outcomes==i) for i in range(3)])
            ax.set_title('Outcome')
            
            ax = axes[2, 1]
            ax.plot(TIME_AXIS, all_output[sess_idx][trial_idx][3, :])
            ax.set_title('Tongue vel (trial 0)'); ax.set_yticks([0,1,2]); ax.set_yticklabels(['low','high','not vis'])
            
            ax = axes[3, 0]
            ax.plot(TIME_AXIS, all_output[sess_idx][trial_idx][4, :])
            ax.set_title('Paw vel (trial 0)'); ax.set_yticks([0,1,2]); ax.set_yticklabels(['low','high','not vis'])
            
            ax = axes[3, 1]
            ax.plot(TIME_AXIS, all_output[sess_idx][trial_idx][5, :])
            ax.set_title('Motion energy (trial 0)'); ax.set_yticks([0,1,2]); ax.set_yticklabels(['low','high','no vid'])
            
            plt.tight_layout()
            fname = f'processing_{info["animal"]}_{info["date"]}.png'
            plt.savefig(fname, dpi=100); plt.close()
            print(f'Saved: {fname}')
    
    print('Done!')


if __name__ == '__main__':
    main()
