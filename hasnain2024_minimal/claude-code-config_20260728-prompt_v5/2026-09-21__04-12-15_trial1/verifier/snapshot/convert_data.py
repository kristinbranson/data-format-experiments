"""
Convert neural and behavioral data from Hasnain, Birnbaum et al. (2024)
into the standardized format for training a neural decoder.

Data: ALM electrophysiology recordings during a two-context (DR + WC)
and randomized-delay task in mice.

Processing follows the reference code and paper:
- Spike data binned in 5 ms bins, smoothed with causal Gaussian (window=15)
- Aligned to go cue onset
- Units filtered by quality (exclude garbage/noisy) and firing rate (>1 Hz)
- Trials filtered to exclude photostimulation and early-lick trials
- Sessions require >= 10 units after filtering
- DLC-based tongue/paw velocity and motion energy discretized per-session
"""

import numpy as np
import h5py
import scipy.io as sio
import os
import pickle
from scipy import stats as spstats

# ============================================================
# Session metadata from loading scripts
# ============================================================
SESSION_META = [
    # Ephys_Behavior (two-context paradigm, 25 sessions)
    ('EKH1', '2021-08-07', [2], 'Ephys_Behavior'),
    ('EKH3', '2021-08-11', [2], 'Ephys_Behavior'),
    ('JEB6', '2021-04-18', [2], 'Ephys_Behavior'),
    ('JEB7', '2021-04-29', [1], 'Ephys_Behavior'),
    ('JEB7', '2021-04-30', [1], 'Ephys_Behavior'),
    ('JGR2', '2021-11-16', [1], 'Ephys_Behavior'),
    ('JGR2', '2021-11-17', [1], 'Ephys_Behavior'),
    ('JGR3', '2021-11-18', [1], 'Ephys_Behavior'),
    ('JEB13', '2022-09-13', [2], 'Ephys_Behavior'),
    ('JEB13', '2022-09-14', [2], 'Ephys_Behavior'),
    ('JEB13', '2022-09-21', [1], 'Ephys_Behavior'),
    ('JEB13', '2022-09-24', [1], 'Ephys_Behavior'),
    ('JEB13', '2022-09-25', [1], 'Ephys_Behavior'),
    ('JEB14', '2022-08-22', [1], 'Ephys_Behavior'),
    ('JEB14', '2022-08-23', [1], 'Ephys_Behavior'),
    ('JEB14', '2022-08-24', [1], 'Ephys_Behavior'),
    ('JEB14', '2022-08-25', [1], 'Ephys_Behavior'),
    ('JEB15', '2022-07-26', [1, 2], 'Ephys_Behavior'),
    ('JEB15', '2022-07-27', [1, 2], 'Ephys_Behavior'),
    ('JEB15', '2022-07-28', [1, 2], 'Ephys_Behavior'),
    ('JEB15', '2022-07-29', [2], 'Ephys_Behavior'),
    ('JEB19', '2023-04-18', [1], 'Ephys_Behavior'),
    ('JEB19', '2023-04-19', [1], 'Ephys_Behavior'),
    ('JEB19', '2023-04-20', [1], 'Ephys_Behavior'),
    ('JEB19', '2023-04-21', [1], 'Ephys_Behavior'),
    # RandomizedDelay_Ephys_Behavior (19 sessions)
    ('JEB11', '2022-05-10', [1], 'RandomizedDelay_Ephys_Behavior'),
    ('JEB11', '2022-05-11', [1], 'RandomizedDelay_Ephys_Behavior'),
    ('JEB12', '2022-05-12', [1], 'RandomizedDelay_Ephys_Behavior'),
    ('JEB12', '2022-05-13', [1], 'RandomizedDelay_Ephys_Behavior'),
    ('JEB23', '2023-10-10', [1], 'RandomizedDelay_Ephys_Behavior'),
    ('JEB23', '2023-10-11', [1], 'RandomizedDelay_Ephys_Behavior'),
    ('JEB23', '2023-10-12', [1], 'RandomizedDelay_Ephys_Behavior'),
    ('JEB23', '2023-10-13', [1], 'RandomizedDelay_Ephys_Behavior'),
    ('JEB23', '2023-10-18', [1], 'RandomizedDelay_Ephys_Behavior'),
    ('JEB23', '2023-10-19', [1], 'RandomizedDelay_Ephys_Behavior'),
    ('JEB23', '2023-10-21', [1], 'RandomizedDelay_Ephys_Behavior'),
    ('JEB24', '2023-10-23', [1], 'RandomizedDelay_Ephys_Behavior'),
    ('JEB24', '2023-10-24', [1], 'RandomizedDelay_Ephys_Behavior'),
    ('JEB24', '2023-10-25', [1], 'RandomizedDelay_Ephys_Behavior'),
    ('JEB24', '2023-10-26', [1], 'RandomizedDelay_Ephys_Behavior'),
    ('JEB24', '2023-10-27', [1], 'RandomizedDelay_Ephys_Behavior'),
    ('JEB24', '2023-10-31', [1], 'RandomizedDelay_Ephys_Behavior'),
    ('JEB24', '2023-11-02', [1], 'RandomizedDelay_Ephys_Behavior'),
    ('JEB24', '2023-11-03', [1], 'RandomizedDelay_Ephys_Behavior'),
]

# ============================================================
# Parameters (from getDefaultParams.m and paper methods)
# ============================================================
TMIN = -2.5
TMAX = 2.5
DT = 1 / 200         # 5 ms
SMOOTH_WINDOW = 15
LOW_FR = 1.0          # Hz (paper methods: "firing rates exceeding 1 Hz")
MIN_UNITS = 10        # paper: "at least 10 units"
VIDEO_FPS = 400
CONF_THRESH = 0.6

EXCLUDED_QUALITIES = {'garbage', 'gabrga', 'noisy', 'real?'}
DATA_ROOT = '/app/data'


# ============================================================
# Unified data loader for v5 and v7.3 MATLAB files
# ============================================================

def detect_mat_version(path):
    with open(path, 'rb') as f:
        header = f.read(16).decode('ascii', errors='replace')
    return '7.3' if '7.3' in header else '5.0'


class SessionData:
    """Unified interface for accessing session data from either MAT format."""

    def __init__(self, data_path):
        self.version = detect_mat_version(data_path)
        self._f = None

        if self.version == '7.3':
            self._f = h5py.File(data_path, 'r')
            self._obj = self._f['obj']
        else:
            d = sio.loadmat(data_path, squeeze_me=False)
            self._obj_v5 = d['obj'][0, 0]  # unwrap (1,1) wrapper

    def close(self):
        if self._f is not None:
            self._f.close()

    def _unwrap_v5(self, val):
        """Unwrap nested v5 scalar arrays."""
        while hasattr(val, 'shape'):
            if val.shape in [(1, 1), (1,)] and (val.dtype == object or val.dtype.names):
                val = val.flatten()[0]
            elif val.shape == () and val.dtype.names:
                break
            else:
                break
        return val

    def _v5_bp(self):
        bp_raw = self._obj_v5['bp']
        return self._unwrap_v5(bp_raw)

    def _v5_scalar(self, val):
        """Extract scalar or array from deeply nested v5 struct field."""
        while hasattr(val, 'shape') and val.shape in [(1, 1), (1,)] and val.size == 1:
            val = val.flatten()[0]
        return val

    # ---- Behavioral data ----
    def get_n_trials(self):
        if self.version == '7.3':
            return int(np.array(self._obj['bp']['Ntrials']).flatten()[0])
        bp = self._v5_bp()
        val = self._v5_scalar(bp['Ntrials'])
        return int(val)

    def get_bp_field(self, field):
        if self.version == '7.3':
            return np.array(self._obj['bp'][field]).flatten()
        bp = self._v5_bp()
        val = bp[field]
        # Unwrap extra (1,1) wrapping
        if hasattr(val, 'shape') and val.shape == (1, 1) and val.dtype == object:
            val = val[0, 0]
        return np.array(val).flatten()

    def get_event_field(self, field):
        if self.version == '7.3':
            return np.array(self._obj['bp']['ev'][field]).flatten()
        bp = self._v5_bp()
        ev = self._unwrap_v5(bp['ev'])
        val = ev[field]
        if hasattr(val, 'shape') and val.shape == (1, 1) and val.dtype == object:
            val = val[0, 0]
        return np.array(val).flatten()

    def get_stim_enable(self):
        if self.version == '7.3':
            return np.array(self._obj['bp']['stim']['enable']).flatten()
        bp = self._v5_bp()
        stim = self._unwrap_v5(bp['stim'])
        val = stim['enable']
        if hasattr(val, 'shape') and val.shape == (1, 1) and val.dtype == object:
            val = val[0, 0]
        return np.array(val).flatten()

    # ---- Cluster/neural data ----
    def get_units_for_probe(self, probe_idx):
        """Get list of unit dicts {tm, trial, trialtm, quality} for a probe."""
        units = []

        if self.version == '7.3':
            clu = self._obj['clu']
            if probe_idx >= clu.shape[0]:
                return units
            probe_ref = clu[probe_idx, 0]
            probe_group = self._f[probe_ref]
            n_units = probe_group['tm'].shape[0]

            for i in range(n_units):
                try:
                    q_data = np.array(self._f[probe_group['quality'][i, 0]]).flatten()
                    quality = ''.join(chr(int(c)) for c in q_data).strip()
                except Exception:
                    quality = ''

                tm = np.array(self._f[probe_group['tm'][i, 0]]).flatten()
                trial = np.array(self._f[probe_group['trial'][i, 0]]).flatten().astype(int)
                trialtm = np.array(self._f[probe_group['trialtm'][i, 0]]).flatten()

                units.append({
                    'tm': tm, 'trial': trial, 'trialtm': trialtm, 'quality': quality
                })
        else:
            clu_raw = self._unwrap_v5(self._obj_v5['clu'])

            # v5 clu might be: cell array (object dtype) or struct array directly
            if hasattr(clu_raw, 'dtype') and clu_raw.dtype == object:
                # Cell array of struct arrays (multi-probe)
                clu_flat = clu_raw.flatten()
                if probe_idx >= len(clu_flat):
                    return units
                probe_arr = clu_flat[probe_idx]
                if hasattr(probe_arr, 'flatten'):
                    probe_arr = probe_arr.flatten()
            elif hasattr(clu_raw, 'dtype') and clu_raw.dtype.names:
                # Direct struct array (single probe)
                if probe_idx > 0:
                    return units
                probe_arr = clu_raw.flatten() if clu_raw.ndim > 0 else np.array([clu_raw])
            else:
                return units

            for i in range(len(probe_arr)):
                u = probe_arr[i]
                # Extract fields from structured array element
                if hasattr(u, 'dtype') and u.dtype.names and 'quality' in u.dtype.names:
                    q = u['quality']
                    # Unwrap nested arrays
                    while hasattr(q, 'shape') and q.shape in [(1, 1), (1,)]:
                        q = q.flatten()[0]
                    quality = str(q).strip()
                    tm = np.array(u['tm']).flatten()
                    trial = np.array(u['trial']).flatten().astype(int)
                    trialtm = np.array(u['trialtm']).flatten()
                elif isinstance(u, np.void) and 'quality' in u.dtype.names:
                    q = u['quality']
                    while hasattr(q, 'flatten') and q.size == 1:
                        q = q.flatten()[0]
                    quality = str(q).strip()
                    tm = np.array(u['tm']).flatten()
                    trial = np.array(u['trial']).flatten().astype(int)
                    trialtm = np.array(u['trialtm']).flatten()
                else:
                    continue

                units.append({
                    'tm': tm, 'trial': trial, 'trialtm': trialtm, 'quality': quality
                })

        return units

    # ---- Video offset ----
    def get_video_offset(self):
        try:
            if self.version == '7.3':
                sglx = self._obj['sglx']
                fs = np.array(sglx['fs']).flatten()[0]
                bc_bs = np.array(sglx['bitcode']['bitstart']).flatten()
                bp_bs = np.array(self._obj['bp']['ev']['bitStart']).flatten()
            else:
                sglx = self._unwrap_v5(self._obj_v5['sglx'])
                fs = np.array(sglx['fs']).flatten()[0]
                bitcode = self._unwrap_v5(sglx['bitcode'])
                bc_bs = np.array(bitcode['bitstart']).flatten()
                bp = self._v5_bp()
                ev = self._unwrap_v5(bp['ev'])
                bp_bs = np.array(ev['bitStart']).flatten()

            valid_bc = bc_bs[bc_bs > 0]
            valid_bp = bp_bs[bp_bs > 0]
            if len(valid_bc) == 0 or len(valid_bp) == 0:
                return 0.5
            bc_mode = spstats.mode(valid_bc, keepdims=False).mode
            bp_mode = spstats.mode(valid_bp, keepdims=False).mode
            return float(bc_mode / fs - bp_mode)
        except Exception:
            return 0.5

    # ---- DLC trajectory data ----
    def _get_v5_traj_cam(self, cam_idx):
        """Get camera struct array from v5 traj."""
        traj_raw = self._unwrap_v5(self._obj_v5['traj'])
        if hasattr(traj_raw, 'flatten'):
            traj_flat = traj_raw.flatten()
        else:
            return None
        if cam_idx >= len(traj_flat):
            return None
        cam = traj_flat[cam_idx]
        if hasattr(cam, 'flatten'):
            cam = cam.flatten()
        return cam

    def get_traj_trial(self, cam_idx, trial_idx):
        """Get DLC data for a trial. Returns (ts, frame_times) or (None, None).
        ts: (n_frames, 3, n_feats) - standard convention
        """
        try:
            if self.version == '7.3':
                traj = self._obj['traj']
                cam = self._f[traj[cam_idx, 0]]
                ts_ref = cam['ts'][trial_idx, 0]
                ts_raw = np.array(self._f[ts_ref])  # (feats, 3, frames) in h5py
                ts = np.transpose(ts_raw, (2, 1, 0))  # → (frames, 3, feats)
                ft = np.array(self._f[cam['frameTimes'][trial_idx, 0]]).flatten()
                return ts, ft
            else:
                cam = self._get_v5_traj_cam(cam_idx)
                if cam is None or trial_idx >= len(cam):
                    return None, None
                trial_data = cam[trial_idx]
                if isinstance(trial_data, np.void) or (hasattr(trial_data, 'dtype') and trial_data.dtype.names):
                    ts = np.array(trial_data['ts'])
                    ft = np.array(trial_data['frameTimes']).flatten()
                    # v5 ts is already (frames, 3, feats)
                    if ts.ndim == 2:
                        ts = ts.reshape(-1, 3, 1)
                    return ts, ft
                return None, None
        except Exception:
            return None, None

    def get_n_traj_trials(self, cam_idx):
        try:
            if self.version == '7.3':
                traj = self._obj['traj']
                cam = self._f[traj[cam_idx, 0]]
                return cam['ts'].shape[0]
            else:
                cam = self._get_v5_traj_cam(cam_idx)
                if cam is None:
                    return 0
                return len(cam)
        except Exception:
            return 0


def load_motion_energy(me_path):
    """Load motion energy from .mat file. Returns list of per-trial arrays."""
    d = sio.loadmat(me_path, squeeze_me=False)
    me_raw = d['me']

    # Format 1: me is (n_trials, 1) cell array directly (no struct wrapper)
    if me_raw.dtype == object and me_raw.ndim == 2 and me_raw.shape[1] == 1:
        # Check if first element is numeric (raw data) or struct
        first = me_raw[0, 0]
        if isinstance(first, np.ndarray) and first.dtype.kind == 'f':
            return [me_raw[i, 0].flatten() for i in range(me_raw.shape[0])]

    # Format 2: me is struct with .data and .moveThresh
    me_struct = me_raw
    while me_struct.ndim > 0 and me_struct.shape == (1, 1):
        me_struct = me_struct[0, 0]

    data_field = me_struct['data']
    # Handle nested struct: me.data might itself be a struct with .data
    while hasattr(data_field, 'dtype') and data_field.dtype.names and 'data' in data_field.dtype.names:
        while data_field.ndim > 0 and data_field.shape in [(1, 1), (1,)]:
            data_field = data_field.flatten()[0]
        data_field = data_field['data']

    while data_field.ndim > 0 and data_field.shape in [(1, 1), (1,)]:
        data_field = data_field.flatten()[0]

    if data_field.dtype == object:
        n = data_field.shape[0] if data_field.ndim > 0 else 1
        me_data = []
        for i in range(n):
            if data_field.ndim == 2:
                me_data.append(np.array(data_field[i, 0]).flatten())
            elif data_field.ndim == 1:
                me_data.append(np.array(data_field[i]).flatten())
            else:
                me_data.append(np.array(data_field.item()).flatten())
        return me_data
    else:
        return [data_field.flatten()]


# ============================================================
# Processing functions
# ============================================================

def causal_gaussian_smooth(x, window, bctype='reflect'):
    """Causal Gaussian smoothing matching mySmooth.m."""
    if window <= 1:
        return x.copy()

    was_1d = x.ndim == 1
    if was_1d:
        x = x[:, np.newaxis]

    if bctype == 'reflect':
        x_filt = np.concatenate([x[:window], x], axis=0)
        trim = window
    elif bctype == 'zeropad':
        x_filt = np.concatenate([np.zeros((window, x.shape[1])), x], axis=0)
        trim = window
    else:
        x_filt = x.copy()
        trim = 0

    # Causal Gaussian kernel (matching MATLAB gausswin with alpha=2.5)
    n = np.arange(window)
    alpha = 2.5
    center = (window - 1) / 2
    kern = np.exp(-0.5 * ((n - center) / (center / alpha)) ** 2)
    kern[:int(window // 2)] = 0  # causal
    kern = kern / kern.sum()

    out = np.zeros_like(x_filt)
    for j in range(x_filt.shape[1]):
        out[:, j] = np.convolve(x_filt[:, j], kern, mode='same')
    out = out[trim:]

    if was_1d:
        out = out.flatten()
    return out


def bin_and_smooth_spikes(units, go_cue_times, n_trials, edges, time_axis):
    """Bin spikes aligned to go cue, smooth, return firing rates (n_time, n_units, n_trials)."""
    n_time = len(time_axis)
    n_units = len(units)
    trialdat = np.zeros((n_time, n_units, n_trials))

    for i, unit in enumerate(units):
        aligned = unit['trialtm'] - go_cue_times[unit['trial'] - 1]
        for j in range(n_trials):
            mask = unit['trial'] == (j + 1)
            if not np.any(mask):
                continue
            counts = np.histogram(aligned[mask], bins=edges)[0]
            fr = counts.astype(float) / DT
            trialdat[:, i, j] = causal_gaussian_smooth(fr, SMOOTH_WINDOW, 'reflect')

    return trialdat


def compute_velocity_from_dlc(ts, frame_times_aligned, feat_idx, time_axis):
    """Compute velocity of DLC feature and interpolate to time_axis.

    Args:
        ts: (n_frames, 3, n_feats) array [x, y, confidence]
        frame_times_aligned: aligned frame times
        feat_idx: feature index
        time_axis: target time axis

    Returns:
        velocity, visible arrays at time_axis points
    """
    n_time = len(time_axis)
    if ts is None or len(frame_times_aligned) < 3:
        return np.zeros(n_time), np.zeros(n_time, dtype=bool)

    x = ts[:, 0, feat_idx]
    y = ts[:, 1, feat_idx]
    conf = ts[:, 2, feat_idx]

    dt_video = 1.0 / VIDEO_FPS
    dx = np.diff(x)
    dy = np.diff(y)
    vel = np.sqrt(dx**2 + dy**2) / dt_video

    vis = (conf[:-1] >= CONF_THRESH) & (conf[1:] >= CONF_THRESH)
    ft_vel = (frame_times_aligned[:-1] + frame_times_aligned[1:]) / 2

    if len(ft_vel) < 2:
        return np.zeros(n_time), np.zeros(n_time, dtype=bool)

    vel_interp = np.interp(time_axis, ft_vel, vel, left=np.nan, right=np.nan)
    vis_interp = np.interp(time_axis, ft_vel, vis.astype(float), left=0, right=0) >= 0.5

    vis_interp[np.isnan(vel_interp)] = False
    vel_interp = np.nan_to_num(vel_interp, nan=0.0)

    return vel_interp, vis_interp


def discretize_per_session(values, valid_mask):
    """Discretize values per session: 0=<50th, 1=>=50th, 2=invalid."""
    result = np.full_like(values, 2, dtype=np.int64)
    all_valid = values[valid_mask]

    if len(all_valid) > 0:
        thresh = np.percentile(all_valid, 50)
        result[valid_mask] = np.where(all_valid >= thresh, 1, 0)

    return result


# ============================================================
# Main processing
# ============================================================

def process_all_sessions():
    n_bins = int(round((TMAX - TMIN) / DT))
    edges = np.linspace(TMIN, TMAX, n_bins + 1)
    time_axis = edges[:-1] + DT / 2

    all_neural, all_input, all_output = [], [], []
    all_subject_idx = []
    all_brain_region_idx = []
    session_info_list = []
    subjects_set = []

    for animal, date, probes, data_dir in SESSION_META:
        print(f"Processing {animal} {date} (probes {probes})...")

        data_path = os.path.join(DATA_ROOT, data_dir, f'data_structure_{animal}_{date}.mat')
        me_path = os.path.join(DATA_ROOT, data_dir, f'motionEnergy_{animal}_{date}.mat')

        if not os.path.exists(data_path):
            print(f"  WARNING: Data file not found, skipping")
            continue

        sess = SessionData(data_path)
        n_trials_total = sess.get_n_trials()

        # ---- Trial info ----
        hit = sess.get_bp_field('hit')[:n_trials_total]
        miss = sess.get_bp_field('miss')[:n_trials_total]
        no = sess.get_bp_field('no')[:n_trials_total]
        R = sess.get_bp_field('R')[:n_trials_total]
        L = sess.get_bp_field('L')[:n_trials_total]
        autowater = sess.get_bp_field('autowater')[:n_trials_total]
        early = sess.get_bp_field('early')[:n_trials_total]
        stim_enable = sess.get_stim_enable()[:n_trials_total]
        go_cue = sess.get_event_field('goCue')[:n_trials_total]

        # ---- Filter trials ----
        valid_mask = (stim_enable == 0) & (early == 0) & ((hit == 1) | (miss == 1) | (no == 1))
        valid_idx = np.where(valid_mask)[0]

        if len(valid_idx) < 2:
            print(f"  WARNING: Too few valid trials ({len(valid_idx)}), skipping")
            sess.close()
            continue

        # ---- Extract units from specified probes ----
        all_units = []
        for probe_num in probes:
            probe_idx = probe_num - 1
            units = sess.get_units_for_probe(probe_idx)
            # Quality filter
            units = [u for u in units
                     if u['quality'].lower() not in {q.lower() for q in EXCLUDED_QUALITIES}
                     and u['quality'] != '']
            all_units.extend(units)

        if len(all_units) < MIN_UNITS:
            print(f"  WARNING: Too few units after quality filter ({len(all_units)}), skipping")
            sess.close()
            continue

        # ---- Bin and smooth spikes ----
        trialdat = bin_and_smooth_spikes(all_units, go_cue, n_trials_total, edges, time_axis)

        # ---- Remove low FR units ----
        mean_frs = np.mean(trialdat, axis=(0, 2))
        fr_mask = mean_frs > LOW_FR

        if np.sum(fr_mask) < MIN_UNITS:
            print(f"  WARNING: Too few units after FR filter ({np.sum(fr_mask)}), skipping")
            sess.close()
            continue

        trialdat = trialdat[:, fr_mask, :]
        n_units = trialdat.shape[1]

        # Filter out valid trials with all-zero neural data (recording ended)
        has_spikes = np.array([np.any(trialdat[:, :, ti] != 0) for ti in valid_idx])
        valid_idx = valid_idx[has_spikes]
        if len(valid_idx) < 2:
            print(f"  WARNING: Too few trials with neural data, skipping")
            sess.close()
            continue

        print(f"  {n_units} units, {len(valid_idx)} valid trials")

        # ---- Video offset ----
        vidshift = sess.get_video_offset()

        # ---- Compute tongue and paw velocity per trial ----
        n_time = len(time_axis)
        tongue_vel = np.zeros((n_time, n_trials_total))
        tongue_vis = np.zeros((n_time, n_trials_total), dtype=bool)
        paw_vel = np.zeros((n_time, n_trials_total))
        paw_vis = np.zeros((n_time, n_trials_total), dtype=bool)

        n_traj_trials = sess.get_n_traj_trials(1)  # bottom cam
        for trix in range(min(n_trials_total, n_traj_trials)):
            ts, ft = sess.get_traj_trial(1, trix)  # bottom cam
            if ts is None or len(ft) == 0:
                continue

            aligned_ft = ft - vidshift - go_cue[trix]

            # Tongue: top_tongue = feature 0 in bottom cam
            tv, tvis = compute_velocity_from_dlc(ts, aligned_ft, 0, time_axis)
            tongue_vel[:, trix] = tv
            tongue_vis[:, trix] = tvis

            # Paw: top_paw=4, bottom_paw=5 in bottom cam - take max
            pv_top, pvis_top = compute_velocity_from_dlc(ts, aligned_ft, 4, time_axis)
            pv_bot, pvis_bot = compute_velocity_from_dlc(ts, aligned_ft, 5, time_axis)
            combined_vis = pvis_top | pvis_bot
            combined_vel = np.where(
                pvis_top & pvis_bot, np.maximum(pv_top, pv_bot),
                np.where(pvis_top, pv_top, np.where(pvis_bot, pv_bot, 0.0))
            )
            paw_vel[:, trix] = combined_vel
            paw_vis[:, trix] = combined_vis

        # ---- Motion energy ----
        me_aligned = np.full((n_time, n_trials_total), np.nan)
        if os.path.exists(me_path):
            try:
                me_data = load_motion_energy(me_path)

                # Get side cam frame times for alignment (loadMotionEnergy.m uses side cam)
                n_side_trials = sess.get_n_traj_trials(0)
                for trix in range(min(n_trials_total, len(me_data), n_side_trials)):
                    _, ft = sess.get_traj_trial(0, trix)
                    if ft is None or len(ft) < 2:
                        continue
                    aligned_ft = ft - vidshift - go_cue[trix]
                    me_trial = me_data[trix]
                    min_len = min(len(aligned_ft), len(me_trial))
                    if min_len < 2:
                        continue
                    me_aligned[:, trix] = np.interp(
                        time_axis, aligned_ft[:min_len], me_trial[:min_len],
                        left=np.nan, right=np.nan
                    )

                # Fill NaN with nearest value (matching loadMotionEnergy.m fillmissing)
                for trix in range(n_trials_total):
                    col = me_aligned[:, trix]
                    nans = np.isnan(col)
                    if np.any(nans) and not np.all(nans):
                        valid = ~nans
                        me_aligned[:, trix] = np.interp(
                            np.arange(n_time), np.where(valid)[0], col[valid]
                        )
            except Exception as e:
                print(f"  WARNING: Failed to load motion energy: {e}")

        # ---- Discretize outputs for valid trials ----
        tv_valid = tongue_vel[:, valid_idx]
        tvis_valid = tongue_vis[:, valid_idx]
        pv_valid = paw_vel[:, valid_idx]
        pvis_valid = paw_vis[:, valid_idx]
        me_valid = me_aligned[:, valid_idx]

        tongue_disc = discretize_per_session(tv_valid, tvis_valid)
        paw_disc = discretize_per_session(pv_valid, pvis_valid)
        me_disc = discretize_per_session(me_valid, ~np.isnan(me_valid))

        # ---- Per-trial labels ----
        n_valid = len(valid_idx)

        # Lick direction: 0=left, 1=right, 2=none
        lick_dir = np.full(n_valid, 2, dtype=np.int64)
        for i, ti in enumerate(valid_idx):
            if hit[ti] == 1 and R[ti] == 1:
                lick_dir[i] = 1
            elif hit[ti] == 1 and L[ti] == 1:
                lick_dir[i] = 0
            elif miss[ti] == 1 and R[ti] == 1:
                lick_dir[i] = 0  # wrong side
            elif miss[ti] == 1 and L[ti] == 1:
                lick_dir[i] = 1  # wrong side

        # Context: 0=WC, 1=DR
        context = np.where(autowater[valid_idx] == 1, 0, 1).astype(np.int64)

        # Outcome: 0=incorrect, 1=correct, 2=ignore
        outcome = np.full(n_valid, 2, dtype=np.int64)
        for i, ti in enumerate(valid_idx):
            if hit[ti] == 1:
                outcome[i] = 1
            elif miss[ti] == 1:
                outcome[i] = 0

        # ---- Build per-trial data ----
        session_neural, session_input, session_output = [], [], []

        for i, ti in enumerate(valid_idx):
            session_neural.append(trialdat[:, :, ti].T)  # (n_units, n_time)
            session_input.append(time_axis.reshape(1, -1).copy())

            out = np.zeros((6, n_time), dtype=np.int64)
            out[0, :] = lick_dir[i]
            out[1, :] = context[i]
            out[2, :] = outcome[i]
            out[3, :] = tongue_disc[:, i]
            out[4, :] = paw_disc[:, i]
            out[5, :] = me_disc[:, i]
            session_output.append(out)

        all_neural.append(session_neural)
        all_input.append(session_input)
        all_output.append(session_output)

        if animal not in subjects_set:
            subjects_set.append(animal)
        all_subject_idx.append(subjects_set.index(animal))
        all_brain_region_idx.append(np.zeros(n_units, dtype=np.int64))

        session_info_list.append({
            'animal': animal, 'date': date, 'data_dir': data_dir,
            'n_units': n_units, 'n_trials': len(valid_idx),
            'n_trials_total': n_trials_total, 'probes': probes,
        })

        sess.close()

    # ============================================================
    # Build output
    # ============================================================
    data = {
        'neural': all_neural,
        'input': all_input,
        'output': all_output,
        'subjects': subjects_set,
        'subject_idx': np.array(all_subject_idx, dtype=np.int64),
        'brain_regions': ['ALM'],
        'brain_region_idx': all_brain_region_idx,
        'input_names': ['time_from_go_cue'],
        'output_names': [
            'lick_direction', 'behavioral_context', 'outcome',
            'tongue_velocity', 'paw_velocity', 'motion_energy',
        ],
        'output_values': [
            ['left', 'right', 'none'],
            ['WC', 'DR'],
            ['incorrect', 'correct', 'ignore'],
            ['below_50th', 'above_50th', 'not_visible'],
            ['below_50th', 'above_50th', 'not_visible'],
            ['below_50th', 'above_50th', 'no_video'],
        ],
        'metadata': {
            'task_description': (
                'Two-context and randomized-delay licking task. Mice perform '
                'delayed-response (DR) and water-cued (WC) directional licking tasks '
                'alternating block-wise. In DR, an auditory cue indicates reward location; '
                'after a delay, a go cue prompts the mouse to lick. In WC, water is '
                'presented directly without auditory cues.'
            ),
            'time_bin_size': DT * 1000,
            'temporal_alignment_event': 'Go cue onset',
            'off_start': TMIN,
            'off_end': TMAX,
            'recording_region': 'Anterior lateral motor cortex (ALM)',
            'species': 'Mus musculus',
            'session_info': session_info_list,
        }
    }

    n_sessions = len(all_neural)
    n_total_trials = sum(len(s) for s in all_neural)
    print(f"\nTotal: {n_sessions} sessions, {n_total_trials} trials")
    return data


if __name__ == '__main__':
    print("Converting data...")
    data = process_all_sessions()

    output_path = '/app/converted_data.pkl'
    print(f"Saving to {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("Done.")
