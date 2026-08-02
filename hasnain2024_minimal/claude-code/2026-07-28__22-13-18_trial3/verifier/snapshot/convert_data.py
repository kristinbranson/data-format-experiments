#!/usr/bin/env python3
"""
Convert Hasnain, Birnbaum et al (Nature Neuroscience 2024) data to decoder format.

Processing matches the reference code:
- Align to go cue
- 10ms bins from -2.5 to 2.5s
- Causal Gaussian smoothing (window=15 bins, reflect boundary)
- Filter out garbage/noisy clusters, remove <1 Hz mean FR
- Filter trials: exclude stim, early lick, ignore (no response)
"""

import numpy as np
import h5py
import scipy.io as sio
from scipy import stats
from scipy.signal import windows
import pickle
import os
import sys

# ============================================================
# Parameters (matching reference code)
# ============================================================
TMIN = -2.5
TMAX = 2.5
DT = 1/100  # 10 ms
SMOOTH_WINDOW = 15
BC_TYPE = 'reflect'
LOW_FR_THRESH = 1.0
VIDEO_FPS = 400
EXCLUDE_QUALITIES = {'garbage', 'gabrga', 'noisy', 'real?'}

# Session definitions: (animal, date, probes, data_dir)
EPHYS_SESSIONS = [
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
    ('JEB23', '2023-10-20', [1], 'RandomizedDelay_Ephys_Behavior'),
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

DATA_ROOT = 'data'


# ============================================================
# Smoothing (matches mySmooth.m)
# ============================================================
def causal_gaussian_smooth(x, N, bctype='reflect'):
    if N <= 1:
        return x
    was_1d = (x.ndim == 1)
    if was_1d:
        x = x[:, None]
    if bctype == 'reflect':
        x_filt = np.concatenate([x[:N, :], x], axis=0)
        trim = N
    elif bctype == 'zeropad':
        x_filt = np.concatenate([np.zeros((N, x.shape[1])), x], axis=0)
        trim = N
    else:
        x_filt = x.copy()
        trim = 0
    kern = windows.gaussian(N, std=(N-1)/(2*2.5))
    kern[:N//2] = 0
    kern = kern / kern.sum()
    out = np.zeros_like(x_filt)
    for j in range(x_filt.shape[1]):
        out[:, j] = np.convolve(x_filt[:, j], kern, mode='same')
    out = out[trim:, :]
    if was_1d:
        out = out[:, 0]
    return out


# ============================================================
# Unified data loader (handles both v7.3 HDF5 and v5 MAT)
# ============================================================
class SessionData:
    """Unified interface for loading session data from .mat files."""

    def __init__(self, fpath):
        self.fpath = fpath
        self.is_h5 = self._check_h5(fpath)
        if self.is_h5:
            self.f = h5py.File(fpath, 'r')
            self.obj = self.f['obj']
        else:
            mat = sio.loadmat(fpath, squeeze_me=True, struct_as_record=False)
            self.obj_v5 = mat['obj']
            self.f = None

    def _check_h5(self, fpath):
        with open(fpath, 'rb') as ff:
            header = ff.read(30).decode('ascii', errors='replace')
        return '7.3' in header

    def close(self):
        if self.f is not None:
            self.f.close()

    # --- Trial info ---
    def get_ntrials(self):
        if self.is_h5:
            return int(np.array(self.obj['bp']['Ntrials']).flatten()[0])
        return int(self.obj_v5.bp.Ntrials)

    def get_trial_array(self, field):
        """Get a boolean/numeric trial array from bp (hit, miss, no, R, L, early, autowater)."""
        if self.is_h5:
            return np.array(self.obj['bp'][field]).flatten()
        return np.array(getattr(self.obj_v5.bp, field)).flatten()

    def get_stim_enable(self):
        if self.is_h5:
            return np.array(self.obj['bp']['stim']['enable']).flatten()
        return np.array(self.obj_v5.bp.stim.enable).flatten()

    def get_event_times(self, event):
        if self.is_h5:
            return np.array(self.obj['bp']['ev'][event]).flatten()
        return np.array(getattr(self.obj_v5.bp.ev, event)).flatten()

    # --- Cluster data ---
    def get_clusters(self, probe_nums):
        """Get list of (quality, trial_array, trialtm_array) for each cluster."""
        clusters = []
        if self.is_h5:
            clu_ref = self.obj['clu']
            for pnum in probe_nums:
                pidx = pnum - 1
                try:
                    ref = clu_ref[pidx, 0]
                    clu_group = self.f[ref]
                    if not isinstance(clu_group, h5py.Group):
                        continue
                except:
                    continue
                quality_refs = clu_group['quality']
                n_clu = quality_refs.shape[0]
                for i in range(n_clu):
                    q_ref = quality_refs[i, 0]
                    val = self.f[q_ref]
                    q = ''.join(chr(int(c)) for c in np.array(val).flatten()).strip().lower()

                    t_ref = clu_group['trial'][i, 0]
                    trial = np.array(self.f[t_ref]).flatten()

                    tt_ref = clu_group['trialtm'][i, 0]
                    trialtm = np.array(self.f[tt_ref]).flatten()

                    clusters.append((q, trial, trialtm))
        else:
            clu = self.obj_v5.clu
            # v5 format: clu is flat array of cluster structs (single probe)
            # or sometimes cell array {probe1_clusters, probe2_clusters}
            if isinstance(clu, np.ndarray) and len(clu) > 0:
                first = clu[0]
                if hasattr(first, 'quality'):
                    # Flat array of cluster structs (single probe)
                    for c in clu:
                        q = str(c.quality).strip().lower()
                        trial = np.array(c.trial).flatten()
                        trialtm = np.array(c.trialtm).flatten()
                        clusters.append((q, trial, trialtm))
                elif isinstance(first, np.ndarray):
                    # Cell array by probe
                    for pnum in probe_nums:
                        pidx = pnum - 1
                        if pidx < len(clu):
                            probe_clu = clu[pidx]
                            if isinstance(probe_clu, np.ndarray):
                                for c in probe_clu:
                                    q = str(c.quality).strip().lower()
                                    trial = np.array(c.trial).flatten()
                                    trialtm = np.array(c.trialtm).flatten()
                                    clusters.append((q, trial, trialtm))
        return clusters

    # --- Video offset ---
    def get_video_offset(self):
        if self.is_h5:
            bitStart = np.array(self.obj['bp']['ev']['bitStart']).flatten()
            sglx_bitstart = np.array(self.obj['sglx']['bitcode']['bitstart']).flatten()
            sglx_fs = float(np.array(self.obj['sglx']['fs']).flatten()[0])
        else:
            bitStart = np.array(self.obj_v5.bp.ev.bitStart).flatten()
            sglx_bitstart = np.array(self.obj_v5.sglx.bitcode.bitstart).flatten()
            sglx_fs = float(np.array(self.obj_v5.sglx.fs).flatten()[0])

        bs_mode = float(stats.mode(bitStart, keepdims=False).mode)
        sglx_mode = float(stats.mode(sglx_bitstart, keepdims=False).mode)
        return sglx_mode / sglx_fs - bs_mode

    # --- Trajectory data ---
    def get_traj_feature_names(self, view):
        """Get feature names for a camera view (0=side, 1=bottom)."""
        if self.is_h5:
            traj_ref = self.obj['traj'][view, 0]
            traj_g = self.f[traj_ref]
            feat_ref = traj_g['featNames'][0, 0]
            feat_data = self.f[feat_ref]
            names = []
            if feat_data.shape[0] == 1:
                for i in range(feat_data.shape[1]):
                    ref = feat_data[0, i]
                    val = self.f[ref]
                    names.append(''.join(chr(int(c)) for c in np.array(val).flatten()))
            else:
                for i in range(feat_data.shape[0]):
                    ref = feat_data[i, 0]
                    val = self.f[ref]
                    names.append(''.join(chr(int(c)) for c in np.array(val).flatten()))
            return names
        else:
            traj_view = self.obj_v5.traj[view]
            feats = traj_view[0].featNames
            if isinstance(feats, str):
                return [feats]
            return list(feats)

    def get_trial_traj(self, view, trial_idx, feat_idx):
        """Get (frame_times, x, y) for a specific trial and feature."""
        if self.is_h5:
            traj_ref = self.obj['traj'][view, 0]
            traj_g = self.f[traj_ref]

            ft_ref = traj_g['frameTimes'][trial_idx, 0]
            frame_times = np.array(self.f[ft_ref]).flatten()

            ts_ref = traj_g['ts'][trial_idx, 0]
            ts = np.array(self.f[ts_ref])
            # HDF5: (n_feats, 3, n_frames)
            x = ts[feat_idx, 0, :]
            y = ts[feat_idx, 1, :]
        else:
            traj_view = self.obj_v5.traj[view]
            trial_data = traj_view[trial_idx]
            frame_times = np.array(trial_data.frameTimes).flatten()
            ts = np.array(trial_data.ts)
            # v5: (n_frames, 3, n_feats)
            x = ts[:, 0, feat_idx]
            y = ts[:, 1, feat_idx]
        return frame_times, x, y

    def get_trial_frame_times(self, view, trial_idx):
        """Get frame times for a trial."""
        if self.is_h5:
            traj_ref = self.obj['traj'][view, 0]
            traj_g = self.f[traj_ref]
            ft_ref = traj_g['frameTimes'][trial_idx, 0]
            return np.array(self.f[ft_ref]).flatten()
        else:
            return np.array(self.obj_v5.traj[view][trial_idx].frameTimes).flatten()


def load_motion_energy(data_dir, anm, date):
    """Load motion energy data from .mat file."""
    me_fn = f'motionEnergy_{anm}_{date}.mat'
    me_path = os.path.join(DATA_ROOT, data_dir, me_fn)

    if not os.path.exists(me_path):
        return None, None

    try:
        me_mat = sio.loadmat(me_path, squeeze_me=False, struct_as_record=True)
        me_var = me_mat['me']

        # Format 1: me is a struct with data and moveThresh
        if me_var.dtype.names is not None and 'data' in me_var.dtype.names:
            me_struct = me_var[0, 0] if me_var.ndim >= 2 else me_var[0]
            thresh = float(me_struct['moveThresh'].flatten()[0])
            data_field = me_struct['data']
            # Handle nested struct (data.data)
            if data_field.dtype.names is not None and 'data' in data_field.dtype.names:
                data_field = data_field[0, 0]['data']
            data_arr = data_field.flatten()
            me_list = [data_arr[i].flatten().astype(float) for i in range(len(data_arr))]
            return me_list, thresh

        # Format 2: me is directly a cell array (no struct wrapper)
        if me_var.dtype == object:
            data_arr = me_var.flatten()
            me_list = [data_arr[i].flatten().astype(float) for i in range(len(data_arr))]
            return me_list, None

        return None, None
    except Exception as e:
        print(f'  WARNING: Cannot load motion energy: {e}')
        return None, None


# ============================================================
# Main processing
# ============================================================
def process_session(anm, date, probe_nums, data_dir, time_edges, time_axis):
    fn = f'data_structure_{anm}_{date}.mat'
    fpath = os.path.join(DATA_ROOT, data_dir, fn)

    if not os.path.exists(fpath):
        print(f'  WARNING: File not found: {fpath}')
        return None

    print(f'  Loading {anm}_{date} (probes {probe_nums})...')

    sd = SessionData(fpath)
    ntrials = sd.get_ntrials()
    n_timebins = len(time_axis)

    # Trial info
    hit = sd.get_trial_array('hit').astype(bool)
    miss = sd.get_trial_array('miss').astype(bool)
    R = sd.get_trial_array('R').astype(bool)
    L = sd.get_trial_array('L').astype(bool)
    autowater = sd.get_trial_array('autowater')
    early = sd.get_trial_array('early').astype(bool)
    stim_enable = sd.get_stim_enable().astype(bool)
    gocue = sd.get_event_times('goCue')

    # Trial selection: hit or miss, no stim, no early
    valid_trials = (hit | miss) & ~stim_enable & ~early
    valid_idx = np.where(valid_trials)[0]

    if len(valid_idx) < 5:
        print(f'  WARNING: Too few valid trials ({len(valid_idx)}), skipping')
        sd.close()
        return None

    # Get clusters
    try:
        clusters = sd.get_clusters(probe_nums)
    except (KeyError, AttributeError) as e:
        print(f'  WARNING: No cluster data found ({e}), skipping')
        sd.close()
        return None
    # Filter by quality
    filtered_clusters = [(q, t, tt) for q, t, tt in clusters if q not in EXCLUDE_QUALITIES]

    if len(filtered_clusters) == 0:
        print(f'  WARNING: No valid clusters, skipping')
        sd.close()
        return None

    n_units = len(filtered_clusters)

    # Bin spikes aligned to go cue
    trialdat = np.zeros((n_units, n_timebins, ntrials))
    for i, (q, trial_nums, spike_times) in enumerate(filtered_clusters):
        for j in range(ntrials):
            trial_mask = (trial_nums == (j + 1))
            if not np.any(trial_mask):
                continue
            aligned_times = spike_times[trial_mask] - gocue[j]
            counts, _ = np.histogram(aligned_times, bins=time_edges)
            fr = counts / DT
            trialdat[i, :, j] = causal_gaussian_smooth(fr, SMOOTH_WINDOW, BC_TYPE)

    # Remove low FR units
    mean_frs = np.mean(np.mean(trialdat[:, :, valid_idx], axis=2), axis=1)
    keep_units = mean_frs > LOW_FR_THRESH

    if keep_units.sum() < 10:
        print(f'  WARNING: Only {keep_units.sum()} units with FR>{LOW_FR_THRESH} Hz, skipping')
        sd.close()
        return None

    trialdat = trialdat[keep_units, :, :]
    n_units_final = trialdat.shape[0]
    print(f'  {n_units_final} units after filtering (from {n_units} total)')

    # Video offset
    vidshift = sd.get_video_offset()

    # Get feature indices for tongue and paw (bottom cam)
    try:
        bottom_feats = sd.get_traj_feature_names(1)
        tongue_feat_idx = None
        paw_feat_idx = None
        for idx, name in enumerate(bottom_feats):
            if name == 'top_tongue' and tongue_feat_idx is None:
                tongue_feat_idx = idx
            if name == 'bottom_paw' and paw_feat_idx is None:
                paw_feat_idx = idx
        if tongue_feat_idx is None:
            for idx, name in enumerate(bottom_feats):
                if 'tongue' in name.lower():
                    tongue_feat_idx = idx
                    break
        if paw_feat_idx is None:
            for idx, name in enumerate(bottom_feats):
                if 'paw' in name.lower():
                    paw_feat_idx = idx
                    break
    except:
        tongue_feat_idx = None
        paw_feat_idx = None

    # Load motion energy
    me_data, me_thresh = load_motion_energy(data_dir, anm, date)

    # Process each valid trial
    neural_trials = []
    input_trials = []
    raw_outputs = []

    for trial_idx in valid_idx:
        neural = trialdat[:, :, trial_idx]
        input_data = time_axis.reshape(1, -1).copy()

        lick_dir = 1 if R[trial_idx] else 0
        context = 0 if autowater[trial_idx] == 1 else 1
        outcome = 1 if hit[trial_idx] else 0

        # Tongue velocity
        tongue_vel = np.full(n_timebins, np.nan)
        if tongue_feat_idx is not None:
            try:
                ft, x, y = sd.get_trial_traj(1, trial_idx, tongue_feat_idx)
                dt_vid = 1.0 / VIDEO_FPS
                vel = np.sqrt(np.diff(x)**2 + np.diff(y)**2) / dt_vid
                vel_t = ft[:-1] + dt_vid / 2
                vel_t_aligned = vel_t - vidshift - gocue[trial_idx]
                tongue_vel = np.interp(time_axis, vel_t_aligned, vel, left=np.nan, right=np.nan)
            except:
                pass

        # Paw velocity
        paw_vel = np.full(n_timebins, np.nan)
        if paw_feat_idx is not None:
            try:
                ft, x, y = sd.get_trial_traj(1, trial_idx, paw_feat_idx)
                dt_vid = 1.0 / VIDEO_FPS
                vel = np.sqrt(np.diff(x)**2 + np.diff(y)**2) / dt_vid
                vel_t = ft[:-1] + dt_vid / 2
                vel_t_aligned = vel_t - vidshift - gocue[trial_idx]
                paw_vel = np.interp(time_axis, vel_t_aligned, vel, left=np.nan, right=np.nan)
            except:
                pass

        # Motion energy
        me_interp = np.full(n_timebins, np.nan)
        if me_data is not None and trial_idx < len(me_data):
            try:
                ft = sd.get_trial_frame_times(1, trial_idx)
                ft_aligned = ft - vidshift - gocue[trial_idx]
                me_interp = np.interp(time_axis, ft_aligned, me_data[trial_idx], left=np.nan, right=np.nan)
            except:
                pass

        neural_trials.append(neural)
        input_trials.append(input_data)
        raw_outputs.append({
            'lick_dir': lick_dir,
            'context': context,
            'outcome': outcome,
            'tongue_vel': tongue_vel,
            'paw_vel': paw_vel,
            'motion_energy': me_interp,
        })

    sd.close()

    if len(neural_trials) < 2:
        print(f'  WARNING: Only {len(neural_trials)} valid trials, skipping')
        return None

    # Discretize continuous outputs per session (50th percentile threshold)
    all_tongue = np.concatenate([t['tongue_vel'] for t in raw_outputs])
    all_paw = np.concatenate([t['paw_vel'] for t in raw_outputs])
    all_me = np.concatenate([t['motion_energy'] for t in raw_outputs])

    tongue_thresh = np.nanpercentile(all_tongue, 50) if not np.all(np.isnan(all_tongue)) else 0
    paw_thresh = np.nanpercentile(all_paw, 50) if not np.all(np.isnan(all_paw)) else 0
    me_thresh_50 = np.nanpercentile(all_me, 50) if not np.all(np.isnan(all_me)) else 0

    final_output_trials = []
    for t in raw_outputs:
        tongue_disc = np.where(np.isnan(t['tongue_vel']), 0,
                              np.where(t['tongue_vel'] >= tongue_thresh, 1, 0)).astype(np.int64)
        paw_disc = np.where(np.isnan(t['paw_vel']), 0,
                           np.where(t['paw_vel'] >= paw_thresh, 1, 0)).astype(np.int64)
        me_disc = np.where(np.isnan(t['motion_energy']), 0,
                          np.where(t['motion_energy'] >= me_thresh_50, 1, 0)).astype(np.int64)

        out = np.zeros((6, n_timebins), dtype=np.int64)
        out[0, :] = t['lick_dir']
        out[1, :] = t['context']
        out[2, :] = t['outcome']
        out[3, :] = tongue_disc
        out[4, :] = paw_disc
        out[5, :] = me_disc
        final_output_trials.append(out)

    return {
        'neural': neural_trials,
        'input': input_trials,
        'output': final_output_trials,
        'n_units': n_units_final,
        'anm': anm,
        'date': date,
        'n_trials': len(neural_trials),
    }


def main(output_file='converted_data.pkl', sample_file='sample_data.pkl'):
    time_edges = np.arange(TMIN, TMAX + DT, DT)
    time_axis = time_edges[:-1] + DT / 2
    n_timebins = len(time_axis)

    print(f'Time axis: {n_timebins} bins, {TMIN} to {TMAX} s, dt={DT*1000:.0f} ms')

    available = []
    for anm, date, probes, data_dir in EPHYS_SESSIONS:
        fn = f'data_structure_{anm}_{date}.mat'
        fpath = os.path.join(DATA_ROOT, data_dir, fn)
        if os.path.exists(fpath):
            available.append((anm, date, probes, data_dir))
        else:
            print(f'  Skipping {anm}_{date}: file not found')

    print(f'\nFound {len(available)} session files')

    all_sessions = []
    subjects_set = []

    for anm, date, probes, data_dir in available:
        result = process_session(anm, date, probes, data_dir, time_edges, time_axis)
        if result is not None:
            all_sessions.append(result)
            if anm not in subjects_set:
                subjects_set.append(anm)

    print(f'\nProcessed {len(all_sessions)} sessions from {len(subjects_set)} subjects')

    subjects = subjects_set
    subject_idx = np.array([subjects.index(s['anm']) for s in all_sessions])

    brain_regions = ['ALM']
    brain_region_idx = [np.zeros(s['n_units'], dtype=np.int64) for s in all_sessions]

    neural = [s['neural'] for s in all_sessions]
    inputs = [s['input'] for s in all_sessions]
    outputs = [s['output'] for s in all_sessions]

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
            ['left', 'right'],
            ['WC', 'DR'],
            ['incorrect', 'correct'],
            ['< 50th pctl', '>= 50th pctl'],
            ['< 50th pctl', '>= 50th pctl'],
            ['< 50th pctl', '>= 50th pctl'],
        ],
        'metadata': {
            'task_description': 'Two-context directional licking task: delayed-response (DR) and water-cued (WC) paradigms with electrophysiology in ALM',
            'time_bin_size': DT * 1000,
            'temporal_alignment_event': 'Go cue onset',
            'off_start': TMIN,
            'off_end': TMAX,
            'smoothing': f'Causal Gaussian, window={SMOOTH_WINDOW} bins',
            'boundary_condition': BC_TYPE,
            'low_fr_threshold_hz': LOW_FR_THRESH,
            'n_sessions': len(all_sessions),
            'n_subjects': len(subjects),
            'session_info': [
                {'animal': s['anm'], 'date': s['date'],
                 'n_units': s['n_units'], 'n_trials': s['n_trials']}
                for s in all_sessions
            ],
        },
    }

    total_units = sum(s['n_units'] for s in all_sessions)
    total_trials = sum(s['n_trials'] for s in all_sessions)
    print(f'\n=== Summary ===')
    print(f'Sessions: {len(all_sessions)}')
    print(f'Subjects: {len(subjects)} ({", ".join(subjects)})')
    print(f'Total units: {total_units}')
    print(f'Total trials: {total_trials}')
    print(f'Time bins per trial: {n_timebins}')
    for i, s in enumerate(all_sessions):
        print(f'  Session {i}: {s["anm"]}_{s["date"]}, {s["n_units"]} units, {s["n_trials"]} trials')

    print(f'\nSaving full dataset to {output_file}...')
    with open(output_file, 'wb') as f:
        pickle.dump(data, f, protocol=4)
    print(f'Saved ({os.path.getsize(output_file) / 1e6:.1f} MB)')

    # Sample: first 3 sessions
    n_sample = min(3, len(all_sessions))
    sample_data = {
        'neural': neural[:n_sample],
        'input': inputs[:n_sample],
        'output': outputs[:n_sample],
        'subjects': subjects,
        'subject_idx': subject_idx[:n_sample],
        'brain_regions': brain_regions,
        'brain_region_idx': brain_region_idx[:n_sample],
        'input_names': data['input_names'],
        'output_names': data['output_names'],
        'output_values': data['output_values'],
        'metadata': data['metadata'].copy(),
    }
    sample_data['metadata']['note'] = f'Sample: first {n_sample} sessions only'

    print(f'Saving sample dataset to {sample_file}...')
    with open(sample_file, 'wb') as f:
        pickle.dump(sample_data, f, protocol=4)
    print(f'Saved ({os.path.getsize(sample_file) / 1e6:.1f} MB)')

    return data


if __name__ == '__main__':
    data = main()
