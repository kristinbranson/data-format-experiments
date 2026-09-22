"""
Convert Hasnain & Birnbaum et al. (Nature Neuroscience 2024) data to decoder format.

Data: ALM electrophysiology recordings during two-context (DR + WC) and
randomized-delay DR tasks, with DeepLabCut tracking and motion energy.

Processing follows the reference MATLAB code (WorkingWithDataObjs.m,
DataLoadingScripts/, etc.):
  - Align to go cue onset
  - Time window: -2.5 to 2.5 s, dt = 10 ms (params.dt = 1/100)
  - Smoothing: causal half-Gaussian kernel, window = 15 bins
  - Boundary condition: reflect
  - Cluster quality: 'all' (exclude 'garbage', 'gabrga', 'noisy', 'real?')
  - Low firing rate threshold: 1 Hz
  - Session inclusion: >= 10 units after filtering
  - Trial filtering: exclude stim.enable and early lick trials
  - Video offset: computed from sglx.bitcode.bitstart / sglx.fs - mode(ev.bitStart)

Decoder outputs:
  - Lick direction (left, right, none) per trial
  - Behavioral context (WC, DR) per trial
  - Outcome (incorrect, correct, ignore) per trial
  - Tongue velocity (discretized 0/1/2) time-varying
  - Paw velocity (discretized 0/1/2) time-varying
  - Motion energy (discretized 0/1/2) time-varying
"""

import numpy as np
import h5py
import scipy.io as sio
from scipy.stats import mode as scipy_mode
from scipy.interpolate import interp1d
import pickle
import os
import warnings
warnings.filterwarnings('ignore')


# ─── Session metadata ───────────────────────────────────────────────────────
# Extracted from the loading scripts in code/DataLoadingScripts/Recording and video/
# Each entry: (animal, date, probe_list, data_dir)
# probe_list uses MATLAB 1-based indexing

EPHYS_DIR = '/app/data/Ephys_Behavior'
RANDDELAY_DIR = '/app/data/RandomizedDelay_Ephys_Behavior'

SESSION_META = [
    # Ephys_Behavior sessions (two-context and DR-only)
    ('EKH1', '2021-08-07', [2], EPHYS_DIR),
    ('EKH3', '2021-08-11', [2], EPHYS_DIR),
    ('JEB6', '2021-04-18', [2], EPHYS_DIR),
    ('JEB7', '2021-04-29', [1], EPHYS_DIR),
    ('JEB7', '2021-04-30', [1], EPHYS_DIR),
    ('JGR2', '2021-11-16', [1], EPHYS_DIR),
    ('JGR2', '2021-11-17', [1], EPHYS_DIR),
    ('JGR3', '2021-11-18', [1], EPHYS_DIR),
    ('JEB13', '2022-09-13', [2], EPHYS_DIR),
    ('JEB13', '2022-09-14', [2], EPHYS_DIR),
    ('JEB13', '2022-09-21', [1], EPHYS_DIR),
    ('JEB13', '2022-09-24', [1], EPHYS_DIR),
    ('JEB13', '2022-09-25', [1], EPHYS_DIR),
    ('JEB14', '2022-08-22', [1], EPHYS_DIR),
    ('JEB14', '2022-08-23', [1], EPHYS_DIR),
    ('JEB14', '2022-08-24', [1], EPHYS_DIR),
    ('JEB14', '2022-08-25', [1], EPHYS_DIR),
    ('JEB15', '2022-07-26', [1, 2], EPHYS_DIR),
    ('JEB15', '2022-07-27', [1, 2], EPHYS_DIR),
    ('JEB15', '2022-07-28', [1, 2], EPHYS_DIR),
    ('JEB15', '2022-07-29', [2], EPHYS_DIR),
    ('JEB19', '2023-04-18', [1], EPHYS_DIR),
    ('JEB19', '2023-04-19', [1], EPHYS_DIR),
    ('JEB19', '2023-04-20', [1], EPHYS_DIR),
    ('JEB19', '2023-04-21', [1], EPHYS_DIR),
    # RandomizedDelay sessions
    ('JEB11', '2022-05-10', [1], RANDDELAY_DIR),
    ('JEB11', '2022-05-11', [1], RANDDELAY_DIR),
    ('JEB12', '2022-05-12', [1], RANDDELAY_DIR),
    ('JEB12', '2022-05-13', [1], RANDDELAY_DIR),
    ('JEB23', '2023-10-10', [1], RANDDELAY_DIR),
    ('JEB23', '2023-10-11', [1], RANDDELAY_DIR),
    ('JEB23', '2023-10-12', [1], RANDDELAY_DIR),
    ('JEB23', '2023-10-13', [1], RANDDELAY_DIR),
    ('JEB23', '2023-10-18', [1], RANDDELAY_DIR),
    ('JEB23', '2023-10-19', [1], RANDDELAY_DIR),
    ('JEB23', '2023-10-21', [1], RANDDELAY_DIR),
    ('JEB24', '2023-10-23', [1], RANDDELAY_DIR),
    ('JEB24', '2023-10-24', [1], RANDDELAY_DIR),
    ('JEB24', '2023-10-25', [1], RANDDELAY_DIR),
    ('JEB24', '2023-10-26', [1], RANDDELAY_DIR),
    ('JEB24', '2023-10-27', [1], RANDDELAY_DIR),
    ('JEB24', '2023-10-31', [1], RANDDELAY_DIR),
    ('JEB24', '2023-11-02', [1], RANDDELAY_DIR),
    ('JEB24', '2023-11-03', [1], RANDDELAY_DIR),
]

# ─── Parameters (matching WorkingWithDataObjs.m) ────────────────────────────
ALIGN_EVENT = 'goCue'
TMIN = -2.5
TMAX = 2.5
DT = 1.0 / 100  # 10 ms bins
SMOOTH_WINDOW = 15  # bins for causal Gaussian kernel
BC_TYPE = 'reflect'
LOW_FR_THRESH = 1.0  # Hz
MIN_UNITS = 10  # minimum units per session
EXCLUDED_QUALITIES = {'garbage', 'gabrga', 'noisy', 'real?'}
VIDEO_FPS = 400  # Hz


# ─── Helper functions ───────────────────────────────────────────────────────

def read_h5_string(f, ref):
    """Read a MATLAB string stored as uint16 character codes via HDF5 reference."""
    data = f[ref][:]
    return ''.join(chr(int(c)) for c in data.flatten())


def causal_gaussian_kernel(N):
    """Replicate MATLAB mySmooth: gausswin(N) with first half zeroed (causal)."""
    alpha = 2.5
    n = np.arange(N)
    w = np.exp(-0.5 * (alpha * (n - (N - 1) / 2) / ((N - 1) / 2)) ** 2)
    w[:N // 2] = 0
    w /= w.sum()
    return w


def smooth_data(x, N, bc_type='reflect'):
    """Smooth 1D or 2D array along first axis using causal Gaussian kernel."""
    if N <= 1:
        return x

    kern = causal_gaussian_kernel(N)

    was_1d = x.ndim == 1
    if was_1d:
        x = x[:, None]

    if bc_type == 'reflect':
        x_padded = np.concatenate([x[:N], x], axis=0)
        trim = N
    elif bc_type == 'zeropad':
        x_padded = np.concatenate([np.zeros((N, x.shape[1])), x], axis=0)
        trim = N
    else:
        x_padded = x
        trim = 0

    out = np.zeros_like(x_padded)
    for j in range(x_padded.shape[1]):
        out[:, j] = np.convolve(x_padded[:, j], kern, mode='same')
    out = out[trim:]

    if was_1d:
        out = out[:, 0]
    return out


def fill_nearest(arr):
    """Fill NaN values with nearest non-NaN value along axis 0 for each column."""
    if arr.ndim == 1:
        nans = np.isnan(arr)
        if nans.all() or not nans.any():
            return arr
        valid = np.where(~nans)[0]
        for idx in np.where(nans)[0]:
            nearest = valid[np.argmin(np.abs(valid - idx))]
            arr[idx] = arr[nearest]
    elif arr.ndim == 2:
        for col in range(arr.shape[1]):
            fill_nearest(arr[:, col])
    return arr


DLC_CONF_THRESH = 0.9  # Confidence threshold for DLC visibility


def compute_velocity(xy_coords, confidence=None, fill_missing=True):
    """
    Compute velocity magnitude from (nframes, 2) position data at 400 Hz.
    Returns (velocity, visible_mask).
    If confidence is provided, frames below DLC_CONF_THRESH are marked not visible.
    For non-tongue features, fill missing with nearest (matching MATLAB pipeline).
    For tongue, do NOT fill missing (per methods: "except for the tongue").
    """
    if xy_coords is None or len(xy_coords) < 2:
        return None, None
    xy = xy_coords.copy().astype(float)

    # Determine visibility from DLC confidence
    if confidence is not None:
        visible = confidence >= DLC_CONF_THRESH
    else:
        visible = np.ones(len(xy), dtype=bool)

    if fill_missing:
        # Fill missing/NaN values with nearest (for non-tongue features)
        for col in range(xy.shape[1]):
            mask = np.isnan(xy[:, col])
            if mask.all():
                return None, None
            if mask.any():
                valid_idx = np.where(~mask)[0]
                for idx in np.where(mask)[0]:
                    nearest = valid_idx[np.argmin(np.abs(valid_idx - idx))]
                    xy[idx, col] = xy[nearest, col]

    dx = np.diff(xy[:, 0])
    dy = np.diff(xy[:, 1])
    vel = np.sqrt(dx**2 + dy**2) * VIDEO_FPS
    vel = np.concatenate([[vel[0]], vel])
    # Set velocity to NaN where not visible
    vel[~visible] = np.nan
    return vel, visible


# ─── Loading: HDF5 format ──────────────────────────────────────────────────

def load_session_h5(fpath, probe_list):
    """Load session from HDF5 (MATLAB v7.3) file."""
    f = h5py.File(fpath, 'r')
    obj = f['obj']
    bp = obj['bp']
    ev = bp['ev']

    ntrials = int(bp['Ntrials'][0, 0])
    hit = np.array(bp['hit']).flatten().astype(bool)
    miss = np.array(bp['miss']).flatten().astype(bool)
    no = np.array(bp['no']).flatten().astype(bool)
    R = np.array(bp['R']).flatten().astype(bool)
    L = np.array(bp['L']).flatten().astype(bool)
    autowater = np.array(bp['autowater']).flatten().astype(bool)
    early = np.array(bp['early']).flatten().astype(bool)
    stim_enable = np.array(bp['stim']['enable']).flatten().astype(bool)
    goCue = np.array(ev['goCue']).flatten()

    # Video offset
    sglx = obj['sglx']
    bitcode_bs = np.array(sglx['bitcode']['bitstart']).flatten()
    sglx_fs = np.array(sglx['fs']).flatten()[0]
    ev_bs = np.array(ev['bitStart']).flatten()
    vidshift = scipy_mode(bitcode_bs, keepdims=False).mode / sglx_fs - scipy_mode(ev_bs, keepdims=False).mode

    # Load clusters
    clu_data = obj['clu']
    all_spike_trials = []
    all_spike_trialtm = []
    all_qualities = []

    for probe_num in probe_list:
        probe_idx = probe_num - 1
        probe_ref = clu_data[probe_idx, 0]
        probe_group = f[probe_ref]

        if isinstance(probe_group, h5py.Dataset):
            continue

        quality_ds = probe_group['quality']
        trial_ds = probe_group['trial']
        trialtm_ds = probe_group['trialtm']
        n_clusters = quality_ds.shape[0]

        for ci in range(n_clusters):
            q_str = read_h5_string(f, quality_ds[ci, 0]).strip()
            if q_str.lower() in {q.lower() for q in EXCLUDED_QUALITIES}:
                continue
            spike_trials = np.array(f[trial_ds[ci, 0]]).flatten()
            spike_trialtm = np.array(f[trialtm_ds[ci, 0]]).flatten()
            all_spike_trials.append(spike_trials)
            all_spike_trialtm.append(spike_trialtm)
            all_qualities.append(q_str)

    # Load trajectory (bottom cam for tongue/paw)
    traj_data = obj['traj']
    bottom_ref = traj_data[1, 0]
    bottom = f[bottom_ref]

    # Feature names
    feat_names_ref = bottom['featNames'][0, 0]
    feat_names_ds = f[feat_names_ref]
    bottom_feat_names = []
    for j in range(feat_names_ds.shape[1]):
        bottom_feat_names.append(read_h5_string(f, feat_names_ds[0, j]))

    tongue_idx = bottom_feat_names.index('top_tongue') if 'top_tongue' in bottom_feat_names else None
    paw_idx = bottom_feat_names.index('top_paw') if 'top_paw' in bottom_feat_names else None

    frameTimes_ds = bottom['frameTimes']
    ts_ds = bottom['ts']

    trial_frame_times = []
    trial_tongue_xy = []
    trial_tongue_conf = []
    trial_paw_xy = []
    trial_paw_conf = []

    for t in range(ntrials):
        ft = np.array(f[frameTimes_ds[t, 0]]).flatten()
        trial_frame_times.append(ft)

        ts = np.array(f[ts_ds[t, 0]])  # (n_bodyparts, 3, n_frames)
        if tongue_idx is not None:
            trial_tongue_xy.append(ts[tongue_idx, :2, :].T)
            trial_tongue_conf.append(ts[tongue_idx, 2, :])
        else:
            trial_tongue_xy.append(None)
            trial_tongue_conf.append(None)
        if paw_idx is not None:
            trial_paw_xy.append(ts[paw_idx, :2, :].T)
            trial_paw_conf.append(ts[paw_idx, 2, :])
        else:
            trial_paw_xy.append(None)
            trial_paw_conf.append(None)

    f.close()

    return {
        'ntrials': ntrials, 'hit': hit, 'miss': miss, 'no': no,
        'R': R, 'L': L, 'autowater': autowater, 'early': early,
        'stim_enable': stim_enable, 'goCue': goCue, 'vidshift': vidshift,
        'spike_trials': all_spike_trials, 'spike_trialtm': all_spike_trialtm,
        'qualities': all_qualities,
        'trial_frame_times': trial_frame_times,
        'trial_tongue_xy': trial_tongue_xy, 'trial_tongue_conf': trial_tongue_conf,
        'trial_paw_xy': trial_paw_xy, 'trial_paw_conf': trial_paw_conf,
    }


# ─── Loading: scipy format (MATLAB v5) ─────────────────────────────────────

def _deep_squeeze(x):
    """Recursively unwrap nested object arrays to get the numeric array."""
    while isinstance(x, np.ndarray) and x.dtype == object:
        if x.size == 1:
            x = x.flat[0]
        else:
            # Try to convert object array of numeric arrays to a flat numeric array
            try:
                x = np.concatenate([np.asarray(item).flatten() for item in x.flat])
            except Exception:
                break
    if isinstance(x, np.ndarray):
        return x.flatten()
    return np.atleast_1d(x).flatten()


def _get_struct_field(struct, field):
    """Safely access a field from a scipy-loaded MATLAB struct."""
    val = struct[field]
    # struct fields from squeeze_me=False come as (1,1) object arrays
    if isinstance(val, np.ndarray) and val.dtype == object and val.shape == (1, 1):
        val = val[0, 0]
    return val


def _unwrap_struct(x):
    """Unwrap (1,1) struct arrays until we reach the actual struct."""
    while isinstance(x, np.ndarray) and x.dtype.names and x.shape == (1, 1):
        x = x[0, 0]
    return x


def load_session_scipy(fpath, probe_list):
    """Load session from MATLAB v5 .mat file using scipy."""
    d = sio.loadmat(fpath, squeeze_me=False)
    obj = d['obj']
    bp = _unwrap_struct(obj['bp'][0, 0])
    ev = _unwrap_struct(bp['ev'][0, 0])

    ntrials = int(_deep_squeeze(bp['Ntrials'])[0])
    hit = _deep_squeeze(bp['hit']).astype(float).astype(bool)
    miss = _deep_squeeze(bp['miss']).astype(float).astype(bool)
    no = _deep_squeeze(bp['no']).astype(float).astype(bool)
    R = _deep_squeeze(bp['R']).astype(float).astype(bool)
    L = _deep_squeeze(bp['L']).astype(float).astype(bool)
    autowater = _deep_squeeze(bp['autowater']).astype(float).astype(bool)
    early = _deep_squeeze(bp['early']).astype(float).astype(bool)
    stim_struct = _unwrap_struct(bp['stim'][0, 0])
    stim_enable = _deep_squeeze(stim_struct['enable']).astype(float).astype(bool)
    goCue = _deep_squeeze(ev['goCue']).astype(float)

    # Video offset
    sglx = _unwrap_struct(obj['sglx'][0, 0])
    bitcode = _unwrap_struct(sglx['bitcode'][0, 0])
    bitcode_bs = _deep_squeeze(bitcode['bitstart']).astype(float)
    sglx_fs = float(_deep_squeeze(sglx['fs']).astype(float)[0])
    ev_bs = _deep_squeeze(ev['bitStart']).astype(float)
    vidshift = scipy_mode(bitcode_bs, keepdims=False).mode / sglx_fs - scipy_mode(ev_bs, keepdims=False).mode

    # Load clusters
    # obj['clu'] is a MATLAB cell array {1 x nProbes}
    # In scipy: obj['clu'][0,0] is either:
    #   - (1, nProbes) object array: cell array of probes
    #   - (1, 1) object wrapping a struct array: single probe, extra wrapper
    clu_raw = obj['clu'][0, 0]
    # Remove one layer of (1,1) object wrapping if present
    if isinstance(clu_raw, np.ndarray) and clu_raw.dtype == object and clu_raw.shape == (1, 1):
        inner = clu_raw[0, 0]
        if isinstance(inner, np.ndarray) and inner.dtype.names is not None:
            # Direct struct array for single probe - wrap in cell-like form
            clu_cell_probes = [inner]
        elif isinstance(inner, np.ndarray) and inner.dtype == object:
            # It's the actual cell array
            clu_cell_probes = [inner[0, pi] for pi in range(inner.shape[1])]
        else:
            clu_cell_probes = [inner]
    elif isinstance(clu_raw, np.ndarray) and clu_raw.dtype == object:
        clu_cell_probes = [clu_raw[0, pi] for pi in range(clu_raw.shape[1])]
    elif isinstance(clu_raw, np.ndarray) and clu_raw.dtype.names is not None:
        clu_cell_probes = [clu_raw]
    else:
        clu_cell_probes = []

    all_spike_trials = []
    all_spike_trialtm = []
    all_qualities = []

    for probe_num in probe_list:
        probe_idx = probe_num - 1
        if probe_idx >= len(clu_cell_probes):
            continue
        probe_data = clu_cell_probes[probe_idx]

        if not isinstance(probe_data, np.ndarray) or probe_data.size == 0:
            continue
        if probe_data.dtype.names is None:
            continue

        # probe_data is a struct array (1, n_clusters) or (n_clusters,)
        if probe_data.ndim == 0:
            n_clusters = 1
            cluster_items = [probe_data.item()]
        elif probe_data.ndim == 1:
            n_clusters = probe_data.shape[0]
            cluster_items = [probe_data[ci] for ci in range(n_clusters)]
        else:
            n_clusters = probe_data.shape[1]
            cluster_items = [probe_data[0, ci] for ci in range(n_clusters)]

        for ci, clu_item in enumerate(cluster_items):
            q_raw = clu_item['quality']
            if isinstance(q_raw, np.ndarray):
                q_str = str(q_raw.flatten()[0]).strip() if q_raw.size > 0 else ''
            else:
                q_str = str(q_raw).strip()
            if q_str.lower() in {q.lower() for q in EXCLUDED_QUALITIES}:
                continue

            spike_trials = _deep_squeeze(clu_item['trial']).astype(float)
            spike_trialtm = _deep_squeeze(clu_item['trialtm']).astype(float)
            all_spike_trials.append(spike_trials)
            all_spike_trialtm.append(spike_trialtm)
            all_qualities.append(q_str)

    # Load trajectory (bottom cam)
    # traj is a {2,1} or {1,2} cell array: [side_cam; bottom_cam]
    traj_raw = obj['traj'][0, 0]
    # Unwrap one level of (1,1) object if needed
    if isinstance(traj_raw, np.ndarray) and traj_raw.dtype == object and traj_raw.shape == (1, 1):
        traj_raw = traj_raw[0, 0]
    # traj_raw should now be (1,2) or (2,1) object array or (1,nTrials) struct array
    if isinstance(traj_raw, np.ndarray) and traj_raw.dtype == object:
        # Find bottom cam: index 1 in whichever dimension has size 2
        if traj_raw.shape == (1, 2):
            bottom_cam = traj_raw[0, 1]
        elif traj_raw.shape == (2, 1):
            bottom_cam = traj_raw[1, 0]
        else:
            bottom_cam = traj_raw[0, 1] if traj_raw.shape[1] >= 2 else traj_raw[1, 0]
    else:
        bottom_cam = traj_raw  # Hopefully struct array directly
    # Unwrap one more level if needed
    if isinstance(bottom_cam, np.ndarray) and bottom_cam.dtype == object and bottom_cam.shape == (1, 1):
        bottom_cam = bottom_cam[0, 0]

    # Get feature names from first trial
    def _get_trial_struct(bc, idx):
        """Get the struct for trial idx from bottom_cam array."""
        if bc.dtype.names:
            return bc[0, idx] if bc.ndim > 1 else bc[idx]
        return bc[0, idx]

    first_trial = _get_trial_struct(bottom_cam, 0)
    feat_names_raw = first_trial['featNames']
    # Unwrap nested object arrays
    while isinstance(feat_names_raw, np.ndarray) and feat_names_raw.dtype == object and feat_names_raw.size == 1:
        feat_names_raw = feat_names_raw.flat[0]
    bottom_feat_names = []
    if isinstance(feat_names_raw, np.ndarray):
        for item in feat_names_raw.flat:
            if isinstance(item, np.ndarray):
                item = str(item.flat[0])
            bottom_feat_names.append(str(item).strip())

    tongue_idx = bottom_feat_names.index('top_tongue') if 'top_tongue' in bottom_feat_names else None
    paw_idx = bottom_feat_names.index('top_paw') if 'top_paw' in bottom_feat_names else None

    trial_frame_times = []
    trial_tongue_xy = []
    trial_tongue_conf = []
    trial_paw_xy = []
    trial_paw_conf = []

    for t in range(ntrials):
        trial_s = _get_trial_struct(bottom_cam, t)
        ft_raw = trial_s['frameTimes']
        ft = _deep_squeeze(ft_raw).astype(float)
        trial_frame_times.append(ft)

        ts_raw = trial_s['ts']
        while isinstance(ts_raw, np.ndarray) and ts_raw.dtype == object and ts_raw.size == 1:
            ts_raw = ts_raw.flat[0]
        ts = np.asarray(ts_raw, dtype=float)

        if ts.ndim == 3:
            n_feats = len(bottom_feat_names)
            if ts.shape[0] == n_feats:
                # (n_bodyparts, 3, n_frames)
                if tongue_idx is not None:
                    trial_tongue_xy.append(ts[tongue_idx, :2, :].T)
                    trial_tongue_conf.append(ts[tongue_idx, 2, :])
                else:
                    trial_tongue_xy.append(None)
                    trial_tongue_conf.append(None)
                if paw_idx is not None:
                    trial_paw_xy.append(ts[paw_idx, :2, :].T)
                    trial_paw_conf.append(ts[paw_idx, 2, :])
                else:
                    trial_paw_xy.append(None)
                    trial_paw_conf.append(None)
            elif ts.shape[2] == n_feats:
                # (n_frames, 3, n_bodyparts)
                if tongue_idx is not None:
                    trial_tongue_xy.append(ts[:, :2, tongue_idx])
                    trial_tongue_conf.append(ts[:, 2, tongue_idx])
                else:
                    trial_tongue_xy.append(None)
                    trial_tongue_conf.append(None)
                if paw_idx is not None:
                    trial_paw_xy.append(ts[:, :2, paw_idx])
                    trial_paw_conf.append(ts[:, 2, paw_idx])
                else:
                    trial_paw_xy.append(None)
                    trial_paw_conf.append(None)
            else:
                trial_tongue_xy.append(None)
                trial_tongue_conf.append(None)
                trial_paw_xy.append(None)
                trial_paw_conf.append(None)
        else:
            trial_tongue_xy.append(None)
            trial_tongue_conf.append(None)
            trial_paw_xy.append(None)
            trial_paw_conf.append(None)

    return {
        'ntrials': ntrials, 'hit': hit, 'miss': miss, 'no': no,
        'R': R, 'L': L, 'autowater': autowater, 'early': early,
        'stim_enable': stim_enable, 'goCue': goCue, 'vidshift': vidshift,
        'spike_trials': all_spike_trials, 'spike_trialtm': all_spike_trialtm,
        'qualities': all_qualities,
        'trial_frame_times': trial_frame_times,
        'trial_tongue_xy': trial_tongue_xy, 'trial_tongue_conf': trial_tongue_conf,
        'trial_paw_xy': trial_paw_xy, 'trial_paw_conf': trial_paw_conf,
    }


def load_session(anm, date, probe_list, data_dir):
    """Load one session, auto-detecting file format."""
    fn = f'data_structure_{anm}_{date}.mat'
    fpath = os.path.join(data_dir, fn)

    # Try HDF5 first
    try:
        sess = load_session_h5(fpath, probe_list)
    except Exception:
        # Fall back to scipy
        sess = load_session_scipy(fpath, probe_list)

    sess['anm'] = anm
    sess['date'] = date

    # Load motion energy
    me_fn = f'motionEnergy_{anm}_{date}.mat'
    me_fpath = os.path.join(data_dir, me_fn)
    sess['me_data'] = None
    sess['me_thresh'] = None

    if os.path.exists(me_fpath):
        try:
            me_mat = sio.loadmat(me_fpath, squeeze_me=True)
            me = me_mat['me']
            me_data_raw = me['data'].item()
            sess['me_thresh'] = float(me['moveThresh'].item())
            sess['me_data'] = [me_data_raw[i] for i in range(len(me_data_raw))]
        except Exception:
            try:
                mf = h5py.File(me_fpath, 'r')
                me_g = mf['me']
                sess['me_thresh'] = float(np.array(me_g['moveThresh']).flatten()[0])
                me_data_ds = me_g['data']
                sess['me_data'] = []
                for t in range(me_data_ds.shape[0]):
                    ref = me_data_ds[t, 0]
                    sess['me_data'].append(np.array(mf[ref]).flatten())
                mf.close()
            except Exception:
                pass

    return sess


def process_session(sess):
    """Process one session: bin spikes, align, smooth, filter, extract outputs."""
    edges = np.arange(TMIN, TMAX + DT, DT)
    time_centers = edges[:-1] + DT / 2
    n_timebins = len(time_centers)

    goCue = sess['goCue']
    ntrials = sess['ntrials']

    # Trial mask: exclude stim and early lick
    trial_mask = ~sess['stim_enable'] & ~sess['early']
    valid_trials = np.where(trial_mask)[0]

    if len(valid_trials) < 2:
        print(f"  Skipping {sess['anm']}_{sess['date']}: too few valid trials ({len(valid_trials)})")
        return None

    # Bin and smooth spikes for each unit, all trials
    n_units = len(sess['spike_trials'])
    trialdat = np.zeros((n_timebins, n_units, ntrials))

    for ui in range(n_units):
        spike_trials_raw = sess['spike_trials'][ui]
        spike_trialtm_raw = sess['spike_trialtm'][ui]

        for ti in range(ntrials):
            spk_mask = spike_trials_raw == (ti + 1)
            if not np.any(spk_mask):
                continue
            aligned_times = spike_trialtm_raw[spk_mask] - goCue[ti]
            counts, _ = np.histogram(aligned_times, bins=edges)
            fr = counts / DT
            trialdat[:, ui, ti] = smooth_data(fr, SMOOTH_WINDOW, BC_TYPE)

    # Remove low FR units
    mean_frs = np.mean(trialdat[:, :, valid_trials], axis=(0, 2))
    good_units = mean_frs > LOW_FR_THRESH

    if good_units.sum() < MIN_UNITS:
        print(f"  Skipping {sess['anm']}_{sess['date']}: only {good_units.sum()} units after FR filter")
        return None

    trialdat = trialdat[:, good_units, :]
    n_units_final = trialdat.shape[1]
    print(f"  {sess['anm']}_{sess['date']}: {n_units_final} units, {len(valid_trials)} trials")

    # Align video data
    vidshift = sess['vidshift']
    taxis = time_centers
    me_available = sess['me_data'] is not None

    tongue_vel_aligned = np.full((n_timebins, ntrials), np.nan)
    paw_vel_aligned = np.full((n_timebins, ntrials), np.nan)
    me_aligned = np.full((n_timebins, ntrials), np.nan)
    tongue_visible = np.zeros((n_timebins, ntrials), dtype=bool)
    paw_visible = np.zeros((n_timebins, ntrials), dtype=bool)

    for ti in range(ntrials):
        ft = sess['trial_frame_times'][ti]
        if len(ft) == 0:
            continue
        ft_aligned = ft - vidshift - goCue[ti]

        # Tongue velocity (do NOT fill missing - per methods: "except for the tongue")
        tongue_xy = sess['trial_tongue_xy'][ti]
        tongue_conf = sess['trial_tongue_conf'][ti]
        if tongue_xy is not None and len(tongue_xy) > 1:
            vel, vis = compute_velocity(tongue_xy, confidence=tongue_conf, fill_missing=False)
            if vel is not None:
                try:
                    # Interpolate velocity (NaNs where not visible)
                    f_interp = interp1d(ft_aligned, vel, kind='linear',
                                        bounds_error=False, fill_value=np.nan)
                    v = f_interp(taxis)
                    tongue_vel_aligned[:, ti] = v
                    # Interpolate visibility
                    vis_float = vis.astype(float)
                    f_vis = interp1d(ft_aligned, vis_float, kind='nearest',
                                     bounds_error=False, fill_value=0)
                    tongue_visible[:, ti] = f_vis(taxis) > 0.5
                except Exception:
                    pass

        # Paw velocity (fill missing with nearest - per methods)
        paw_xy = sess['trial_paw_xy'][ti]
        paw_conf = sess['trial_paw_conf'][ti]
        if paw_xy is not None and len(paw_xy) > 1:
            vel, vis = compute_velocity(paw_xy, confidence=paw_conf, fill_missing=True)
            if vel is not None:
                try:
                    f_interp = interp1d(ft_aligned, vel, kind='linear',
                                        bounds_error=False, fill_value=np.nan)
                    v = f_interp(taxis)
                    paw_vel_aligned[:, ti] = v
                    f_vis = interp1d(ft_aligned, vis.astype(float), kind='nearest',
                                     bounds_error=False, fill_value=0)
                    paw_visible[:, ti] = f_vis(taxis) > 0.5
                except Exception:
                    pass

        # Motion energy
        if me_available and ti < len(sess['me_data']):
            me_trial = sess['me_data'][ti]
            if len(me_trial) == len(ft):
                try:
                    f_interp = interp1d(ft_aligned, me_trial, kind='linear',
                                        bounds_error=False, fill_value=np.nan)
                    me_aligned[:, ti] = f_interp(taxis)
                except Exception:
                    pass

    # Fill NaNs with nearest for paw and ME (not tongue - per methods)
    fill_nearest(paw_vel_aligned)
    fill_nearest(me_aligned)

    # Update paw visibility after filling (tongue keeps DLC-based visibility)
    paw_visible = paw_visible | ~np.isnan(paw_vel_aligned)

    # Per-session velocity thresholds (50th percentile on valid trials)
    t_vals = tongue_vel_aligned[:, valid_trials][tongue_visible[:, valid_trials]]
    t_vals = t_vals[~np.isnan(t_vals)]
    tongue_thresh = np.percentile(t_vals, 50) if len(t_vals) > 0 else 0

    p_vals = paw_vel_aligned[:, valid_trials][paw_visible[:, valid_trials]]
    p_vals = p_vals[~np.isnan(p_vals)]
    paw_thresh = np.percentile(p_vals, 50) if len(p_vals) > 0 else 0

    me_thresh = 0
    if me_available:
        m_vals = me_aligned[:, valid_trials]
        m_vals = m_vals[~np.isnan(m_vals)]
        if len(m_vals) > 0:
            me_thresh = np.percentile(m_vals, 50)

    # Build per-trial data
    neural_trials = []
    input_trials = []
    output_trials = []

    for ti in valid_trials:
        neural_trials.append(trialdat[:, :, ti].T.astype(np.float32))
        input_trials.append(taxis.reshape(1, -1).astype(np.float32))

        # Lick direction: 0=left, 1=right, 2=none
        # R/L indicate cue direction. For hit trials, animal licked same direction.
        # For miss trials, animal licked opposite direction. For ignore, no lick.
        if sess['hit'][ti]:
            lick_dir = 1 if sess['R'][ti] else 0
        elif sess['miss'][ti]:
            lick_dir = 0 if sess['R'][ti] else 1  # opposite of cue
        else:
            lick_dir = 2  # ignore/no response
        # Context: 0=WC, 1=DR
        context = 0 if sess['autowater'][ti] else 1
        # Outcome: 0=incorrect, 1=correct, 2=ignore
        outcome = 1 if sess['hit'][ti] else (0 if sess['miss'][ti] else 2)

        # Tongue velocity discretized
        tongue_disc = np.full(n_timebins, 2, dtype=np.int64)
        vis = tongue_visible[:, ti] & ~np.isnan(tongue_vel_aligned[:, ti])
        if vis.any():
            tongue_disc[vis & (tongue_vel_aligned[:, ti] < tongue_thresh)] = 0
            tongue_disc[vis & (tongue_vel_aligned[:, ti] >= tongue_thresh)] = 1

        # Paw velocity discretized
        paw_disc = np.full(n_timebins, 2, dtype=np.int64)
        vis = paw_visible[:, ti] & ~np.isnan(paw_vel_aligned[:, ti])
        if vis.any():
            paw_disc[vis & (paw_vel_aligned[:, ti] < paw_thresh)] = 0
            paw_disc[vis & (paw_vel_aligned[:, ti] >= paw_thresh)] = 1

        # Motion energy discretized
        me_disc = np.full(n_timebins, 2, dtype=np.int64)
        if me_available:
            valid_me = ~np.isnan(me_aligned[:, ti])
            if valid_me.any():
                me_disc[valid_me & (me_aligned[:, ti] < me_thresh)] = 0
                me_disc[valid_me & (me_aligned[:, ti] >= me_thresh)] = 1

        out = np.zeros((6, n_timebins), dtype=np.int64)
        out[0, :] = lick_dir
        out[1, :] = context
        out[2, :] = outcome
        out[3, :] = tongue_disc
        out[4, :] = paw_disc
        out[5, :] = me_disc
        output_trials.append(out)

    return {
        'neural': neural_trials,
        'input': input_trials,
        'output': output_trials,
        'anm': sess['anm'],
        'date': sess['date'],
        'n_units': n_units_final,
    }


def main():
    print("Loading and processing sessions...")

    all_sessions = []
    all_subjects = []
    subject_idx_list = []

    for anm, date, probes, data_dir in SESSION_META:
        print(f"Loading {anm}_{date}...")
        try:
            sess = load_session(anm, date, probes, data_dir)
        except Exception as e:
            print(f"  ERROR loading {anm}_{date}: {e}")
            continue

        result = process_session(sess)
        if result is None:
            continue

        if anm not in all_subjects:
            all_subjects.append(anm)
        subj_idx = all_subjects.index(anm)

        all_sessions.append(result)
        subject_idx_list.append(subj_idx)

    print(f"\nTotal sessions: {len(all_sessions)}")
    print(f"Total subjects: {len(all_subjects)}")

    neural = [s['neural'] for s in all_sessions]
    inputs = [s['input'] for s in all_sessions]
    outputs = [s['output'] for s in all_sessions]

    brain_regions = ['ALM']
    brain_region_idx = [np.zeros(s['n_units'], dtype=np.int64) for s in all_sessions]

    data = {
        'neural': neural,
        'input': inputs,
        'output': outputs,
        'subjects': all_subjects,
        'subject_idx': np.array(subject_idx_list, dtype=np.int64),
        'brain_regions': brain_regions,
        'brain_region_idx': brain_region_idx,
        'input_names': ['time_from_go_cue'],
        'output_names': [
            'lick_direction', 'behavioral_context', 'outcome',
            'tongue_velocity', 'paw_velocity', 'motion_energy',
        ],
        'output_values': [
            ['left', 'right', 'none'],
            ['WC', 'DR'],
            ['incorrect', 'correct', 'ignore'],
            ['below_50pct', 'above_50pct', 'not_visible'],
            ['below_50pct', 'above_50pct', 'not_visible'],
            ['below_50pct', 'above_50pct', 'no_video'],
        ],
        'metadata': {
            'task_description': 'Two-context paradigm: delayed-response (DR) and water-cued (WC) directional licking tasks with ALM recordings',
            'time_bin_size': DT * 1000,
            'temporal_alignment_event': 'Go cue onset',
            'off_start': TMIN,
            'off_end': TMAX,
            'smooth_window_bins': SMOOTH_WINDOW,
            'smooth_type': 'causal_gaussian',
            'boundary_condition': BC_TYPE,
            'low_fr_threshold_hz': LOW_FR_THRESH,
            'min_units_per_session': MIN_UNITS,
            'video_fps': VIDEO_FPS,
            'session_info': [
                {'subject': s['anm'], 'date': s['date'],
                 'n_units': s['n_units'], 'n_trials': len(s['neural'])}
                for s in all_sessions
            ],
        },
    }

    out_path = '/app/converted_data.pkl'
    with open(out_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"\nSaved converted data to {out_path}")

    total_trials = sum(len(s['neural']) for s in all_sessions)
    total_neurons = sum(s['n_units'] for s in all_sessions)
    print(f"Total trials: {total_trials}")
    print(f"Total neurons: {total_neurons}")
    print(f"Time bins per trial: {neural[0][0].shape[1]}")


if __name__ == '__main__':
    main()
