#!/usr/bin/env python3
"""
Convert neural data from Hasnain, Birnbaum et al. (Nature Neuroscience 2024)
to decoder format.

Reference: "Separating cognitive and motor processes in the behaving mouse"
Data: ALM electrophysiology during delayed-response and water-cued licking tasks.

Usage:
    python convert_data.py --output converted_data.pkl
    python convert_data.py --output sample_data.pkl --sample 3
"""

import os
import sys
import pickle
import argparse
import numpy as np
import h5py
import scipy.io
from scipy.interpolate import interp1d

# ============================================================
# PARAMETERS (matching reference code WorkingWithDataObjs.m)
# ============================================================
DATA_DIR = 'data'
DT = 0.01           # 10ms bins (params.dt = 1/100)
TMIN = -2.5          # seconds before go cue
TMAX = 2.5           # seconds after go cue
SMOOTH_N = 15        # smoothing window (params.smooth = 15)
BC_TYPE = 'reflect'  # boundary condition (params.bctype = 'reflect')
LOW_FR = 1.0         # min firing rate Hz (params.lowFR = 1)
MIN_UNITS = 10       # min units per session (per paper methods)
MIN_TRIALS = 2       # min trials per session (for decoder)
DLC_CONF = 0.9       # DLC confidence threshold

# Quality labels to exclude (findClusters.m with 'all' mode)
EXCLUDED_QUALITIES = {'garbage', 'gabrga', 'noisy', 'real?'}

# Time axis (matching getSeq.m)
EDGES = np.arange(TMIN, TMAX + DT / 2, DT)
if len(EDGES) > 501:
    EDGES = EDGES[:501]
TIME_AXIS = EDGES[:-1] + DT / 2  # bin centers
T_BINS = len(TIME_AXIS)

# ============================================================
# SESSION DEFINITIONS (from DataLoadingScripts/Recording and video/*.m)
# ============================================================
EB = 'Ephys_Behavior'
RD = 'RandomizedDelay_Ephys_Behavior'

ALL_SESSIONS = [
    # --- Ephys_Behavior (two-context + DR-only) ---
    # EKH1
    {'dir': EB, 'anm': 'EKH1', 'date': '2021-08-07', 'probes': [2]},
    # EKH3
    {'dir': EB, 'anm': 'EKH3', 'date': '2021-08-11', 'probes': [2]},
    # JEB6
    {'dir': EB, 'anm': 'JEB6', 'date': '2021-04-18', 'probes': [2]},
    # JEB7
    {'dir': EB, 'anm': 'JEB7', 'date': '2021-04-29', 'probes': [1]},
    {'dir': EB, 'anm': 'JEB7', 'date': '2021-04-30', 'probes': [1]},
    # JGR2
    {'dir': EB, 'anm': 'JGR2', 'date': '2021-11-16', 'probes': [1]},
    {'dir': EB, 'anm': 'JGR2', 'date': '2021-11-17', 'probes': [1]},
    # JGR3
    {'dir': EB, 'anm': 'JGR3', 'date': '2021-11-18', 'probes': [1]},
    # JEB13
    {'dir': EB, 'anm': 'JEB13', 'date': '2022-09-13', 'probes': [2]},
    {'dir': EB, 'anm': 'JEB13', 'date': '2022-09-14', 'probes': [2]},
    {'dir': EB, 'anm': 'JEB13', 'date': '2022-09-21', 'probes': [1]},
    {'dir': EB, 'anm': 'JEB13', 'date': '2022-09-24', 'probes': [1]},
    {'dir': EB, 'anm': 'JEB13', 'date': '2022-09-25', 'probes': [1]},
    # JEB14
    {'dir': EB, 'anm': 'JEB14', 'date': '2022-08-22', 'probes': [1]},
    {'dir': EB, 'anm': 'JEB14', 'date': '2022-08-23', 'probes': [1]},
    {'dir': EB, 'anm': 'JEB14', 'date': '2022-08-24', 'probes': [1]},
    {'dir': EB, 'anm': 'JEB14', 'date': '2022-08-25', 'probes': [1]},
    # JEB15 - all 4 sessions per loading script (first 3 have note about
    # sensory area but are still included in the loading script)
    {'dir': EB, 'anm': 'JEB15', 'date': '2022-07-26', 'probes': [1, 2]},
    {'dir': EB, 'anm': 'JEB15', 'date': '2022-07-27', 'probes': [1, 2]},
    {'dir': EB, 'anm': 'JEB15', 'date': '2022-07-28', 'probes': [1, 2]},
    {'dir': EB, 'anm': 'JEB15', 'date': '2022-07-29', 'probes': [2]},
    # JEB19
    {'dir': EB, 'anm': 'JEB19', 'date': '2023-04-21', 'probes': [1]},
    {'dir': EB, 'anm': 'JEB19', 'date': '2023-04-20', 'probes': [1]},
    {'dir': EB, 'anm': 'JEB19', 'date': '2023-04-19', 'probes': [1]},
    {'dir': EB, 'anm': 'JEB19', 'date': '2023-04-18', 'probes': [1]},
    # --- RandomizedDelay_Ephys_Behavior ---
    # JEB11
    {'dir': RD, 'anm': 'JEB11', 'date': '2022-05-10', 'probes': [1]},
    {'dir': RD, 'anm': 'JEB11', 'date': '2022-05-11', 'probes': [1]},
    # JEB12
    {'dir': RD, 'anm': 'JEB12', 'date': '2022-05-12', 'probes': [1]},
    {'dir': RD, 'anm': 'JEB12', 'date': '2022-05-13', 'probes': [1]},
    # JEB23
    {'dir': RD, 'anm': 'JEB23', 'date': '2023-10-10', 'probes': [1]},
    {'dir': RD, 'anm': 'JEB23', 'date': '2023-10-11', 'probes': [1]},
    {'dir': RD, 'anm': 'JEB23', 'date': '2023-10-12', 'probes': [1]},
    {'dir': RD, 'anm': 'JEB23', 'date': '2023-10-13', 'probes': [1]},
    {'dir': RD, 'anm': 'JEB23', 'date': '2023-10-18', 'probes': [1]},
    {'dir': RD, 'anm': 'JEB23', 'date': '2023-10-19', 'probes': [1]},
    {'dir': RD, 'anm': 'JEB23', 'date': '2023-10-21', 'probes': [1]},
    # JEB24
    {'dir': RD, 'anm': 'JEB24', 'date': '2023-10-23', 'probes': [1]},
    {'dir': RD, 'anm': 'JEB24', 'date': '2023-10-24', 'probes': [1]},
    {'dir': RD, 'anm': 'JEB24', 'date': '2023-10-25', 'probes': [1]},
    {'dir': RD, 'anm': 'JEB24', 'date': '2023-10-26', 'probes': [1]},
    {'dir': RD, 'anm': 'JEB24', 'date': '2023-10-27', 'probes': [1]},
    {'dir': RD, 'anm': 'JEB24', 'date': '2023-10-31', 'probes': [1]},
    {'dir': RD, 'anm': 'JEB24', 'date': '2023-11-02', 'probes': [1]},
    {'dir': RD, 'anm': 'JEB24', 'date': '2023-11-03', 'probes': [1]},
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def gausswin(N, alpha=2.5):
    """MATLAB-equivalent gausswin(N, alpha)."""
    if N == 1:
        return np.array([1.0])
    n = np.arange(N)
    n_norm = 2 * n / (N - 1) - 1
    return np.exp(-0.5 * (alpha * n_norm) ** 2)


def causal_smooth(x, N=SMOOTH_N, bctype=BC_TYPE):
    """Causal Gaussian smoothing matching mySmooth.m."""
    if N <= 1:
        return x.copy()
    kern = gausswin(N)
    kern[:N // 2] = 0   # causal: zero out first half
    kern /= kern.sum()

    if bctype == 'reflect':
        x_padded = np.concatenate([x[:N], x])
        trim = N
    elif bctype == 'zeropad':
        x_padded = np.concatenate([np.zeros(N), x])
        trim = N
    else:
        x_padded = x.copy()
        trim = 0

    out = np.convolve(x_padded, kern, mode='same')
    return out[trim:]


def h5str(f, ref):
    """Read string from HDF5 object reference."""
    data = f[ref][()].flatten()
    return ''.join(chr(int(c)) for c in data)


def fill_nan_nearest(arr):
    """Fill NaN values with nearest non-NaN, matching MATLAB fillmissing('nearest')."""
    nans = np.isnan(arr)
    if not np.any(nans):
        return arr
    if np.all(nans):
        arr[:] = 0
        return arr
    valid_idx = np.where(~nans)[0]
    f_interp = interp1d(valid_idx, arr[valid_idx], kind='nearest',
                        fill_value='extrapolate', bounds_error=False)
    arr[nans] = f_interp(np.where(nans)[0])
    return arr


def compute_mode(arr):
    """Compute statistical mode."""
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return 0.0
    vals, counts = np.unique(arr, return_counts=True)
    return float(vals[np.argmax(counts)])


def fill_positions_nearest(x, y, valid_mask):
    """Fill invalid positions with nearest valid values."""
    x, y = x.copy(), y.copy()
    if not np.any(valid_mask) or np.all(valid_mask):
        return x, y
    valid_idx = np.where(valid_mask)[0]
    bad_idx = np.where(~valid_mask)[0]
    nearest = np.searchsorted(valid_idx, bad_idx).clip(0, len(valid_idx) - 1)
    # Check left neighbor too for true nearest
    left = np.clip(nearest - 1, 0, len(valid_idx) - 1)
    d_right = np.abs(bad_idx - valid_idx[nearest])
    d_left = np.abs(bad_idx - valid_idx[left])
    use_left = d_left < d_right
    final = np.where(use_left, left, nearest)
    x[bad_idx] = x[valid_idx[final]]
    y[bad_idx] = y[valid_idx[final]]
    return x, y


def compute_speed(x, y, ft, confidence=None, conf_thresh=DLC_CONF,
                  fill_missing=True):
    """Compute instantaneous speed from (x, y) trajectory.

    For all features, fills NaN/low-confidence positions with nearest valid
    position before computing velocity, matching the paper's processing for
    kinematic features. For tongue, positions are NaN when not visible; filling
    gives near-zero velocity during non-licking periods and meaningful velocity
    during licking.
    """
    x = x.copy().astype(np.float64)
    y = y.copy().astype(np.float64)

    # Determine valid frames: not NaN and (optionally) high confidence
    valid = ~np.isnan(x) & ~np.isnan(y)
    if confidence is not None:
        valid = valid & (confidence >= conf_thresh)

    if np.any(valid) and not np.all(valid):
        x, y = fill_positions_nearest(x, y, valid)
    elif not np.any(valid):
        return np.zeros(len(x))

    # Remove remaining NaN (shouldn't happen after fill, but safety)
    x = np.nan_to_num(x, nan=0.0)
    y = np.nan_to_num(y, nan=0.0)

    dt_vid = np.diff(ft)
    dt_vid[dt_vid <= 0] = 1.0 / 400.0

    vx = np.diff(x) / dt_vid
    vy = np.diff(y) / dt_vid
    speed = np.sqrt(vx**2 + vy**2)
    speed = np.concatenate([[0.0], speed])

    return speed


# ============================================================
# SESSION PROCESSING
# ============================================================

def _is_hdf5(path):
    """Check if a .mat file is HDF5 (MATLAB v7.3) format."""
    try:
        with h5py.File(path, 'r') as f:
            return True
    except Exception:
        return False


def _load_session_v5(data_path, sess):
    """Load session data from MATLAB v5 format into common dict."""
    d = scipy.io.loadmat(data_path)
    obj = d['obj']
    bp = obj['bp'][0, 0]

    n_trials = int(bp['Ntrials'][0, 0])
    hit = bp['hit'][0, 0].flatten()[:n_trials].astype(int)
    miss = bp['miss'][0, 0].flatten()[:n_trials].astype(int)
    early = bp['early'][0, 0].flatten()[:n_trials].astype(int)
    no_resp = bp['no'][0, 0].flatten()[:n_trials].astype(int)
    L = bp['L'][0, 0].flatten()[:n_trials].astype(int)
    R = bp['R'][0, 0].flatten()[:n_trials].astype(int)
    autowater = bp['autowater'][0, 0].flatten()[:n_trials].astype(int)

    stim = bp['stim'][0, 0]
    stim_en_raw = stim['enable'][0, 0]
    if stim_en_raw.ndim >= 2:
        stim_en = stim_en_raw.flatten()[:n_trials].astype(int)
    else:
        stim_en = stim_en_raw.flatten()[:n_trials].astype(int)

    ev = bp['ev'][0, 0]
    goCue = ev['goCue'][0, 0].flatten()[:n_trials].astype(float)

    # video offset
    try:
        bitStart_vals = ev['bitStart'][0, 0].flatten()[:n_trials].astype(float)
        bitStart_mode = compute_mode(bitStart_vals)
        sglx = obj['sglx'][0, 0]
        bc = sglx['bitcode'][0, 0]
        bc_bs = bc['bitstart'][0, 0].flatten().astype(float)
        bc_bs_mode = compute_mode(bc_bs)
        fs = float(sglx['fs'][0, 0])
        vidshift = bc_bs_mode / fs - bitStart_mode
    except Exception as e:
        vidshift = 0.5

    # cluster data
    clu_cell = obj['clu'][0, 0]  # (1, nProbes) object array
    spike_data = []
    for prb in sess['probes']:
        pidx = prb - 1
        try:
            probe_arr = clu_cell[0, pidx]  # struct array (1, nClusters)
        except Exception:
            print(f"  Probe {prb}: empty")
            continue
        n_clu = probe_arr.shape[1]
        qualities = []
        for i in range(n_clu):
            q = str(probe_arr['quality'][0, i].item()).strip()
            qualities.append(q)

        valid_clu = [i for i, q in enumerate(qualities)
                     if q.lower() not in EXCLUDED_QUALITIES and q != '']
        q_counts = {}
        for q in qualities:
            q_counts[q] = q_counts.get(q, 0) + 1
        print(f"  Probe {prb}: {n_clu} total, {len(valid_clu)} valid  {q_counts}")

        for ci in valid_clu:
            spike_data.append({
                'trial': probe_arr['trial'][0, ci].flatten().astype(float),
                'trialtm': probe_arr['trialtm'][0, ci].flatten().astype(float),
            })

    # DLC bottom cam
    traj_cell = obj['traj'][0, 0]  # (1, 2) for [side, bottom]
    bot_cam = None
    try:
        bot_arr = traj_cell[0, 1]  # struct array (1, nTrials)
        # feature names from first trial
        fn_raw = bot_arr[0, 0]['featNames']
        feat_names = []
        for i in range(fn_raw.shape[0]):
            try:
                feat_names.append(str(fn_raw[i, 0].item()))
            except Exception:
                feat_names.append(str(fn_raw[i].item()))
        if not feat_names:
            for i in range(fn_raw.shape[1]):
                feat_names.append(str(fn_raw[0, i].item()))

        bot_cam = {'feat_names': feat_names, 'n_cam_trials': bot_arr.shape[1]}

        def get_bot_trial(ti):
            t = bot_arr[0, ti]
            ts = t['ts'].astype(float)  # (nframes, 3, nfeat)
            ft = t['frameTimes'].flatten().astype(float)
            return ts, ft

        bot_cam['get_trial'] = get_bot_trial
    except Exception as e:
        print(f"  WARNING: DLC bottom cam: {e}")

    # side cam frameTimes
    side_ft = {}
    try:
        side_arr = traj_cell[0, 0]
        for ti in range(min(n_trials, side_arr.shape[1])):
            try:
                side_ft[ti] = side_arr[0, ti]['frameTimes'].flatten().astype(float)
            except Exception:
                pass
    except Exception as e:
        print(f"  WARNING: side cam frameTimes: {e}")

    return {
        'n_trials': n_trials, 'hit': hit, 'miss': miss, 'early': early,
        'no_resp': no_resp, 'L': L, 'R': R, 'autowater': autowater,
        'stim_en': stim_en, 'goCue': goCue, 'vidshift': vidshift,
        'spike_data': spike_data, 'bot_cam': bot_cam, 'side_ft': side_ft,
    }


def _load_session_h5(data_path, sess):
    """Load session data from HDF5 (v7.3) format into common dict."""
    spike_data = []
    bot_cam = None
    side_ft = {}

    with h5py.File(data_path, 'r') as f:
        obj = f['obj']
        bp = obj['bp']

        n_trials = int(bp['Ntrials'][0, 0])
        hit   = bp['hit'][0, :n_trials].astype(int)
        miss  = bp['miss'][0, :n_trials].astype(int)
        early = bp['early'][0, :n_trials].astype(int)
        no_resp = bp['no'][0, :n_trials].astype(int)
        L     = bp['L'][0, :n_trials].astype(int)
        R     = bp['R'][0, :n_trials].astype(int)
        autowater = bp['autowater'][0, :n_trials].astype(int)
        stim_en   = bp['stim']['enable'][0, :n_trials].astype(int)
        goCue     = bp['ev']['goCue'][0, :n_trials].copy()

        # video offset
        try:
            bitStart_mode = compute_mode(bp['ev']['bitStart'][0, :n_trials])
            bc_bs = obj['sglx']['bitcode']['bitstart'][0, :]
            bc_bs_mode = compute_mode(bc_bs)
            fs = float(obj['sglx']['fs'][0, 0])
            vidshift = bc_bs_mode / fs - bitStart_mode
        except Exception as e:
            vidshift = 0.5

        # cluster data
        clu_group = obj['clu']
        for prb in sess['probes']:
            pidx = prb - 1
            try:
                prb_ref = clu_group[pidx, 0]
                prb_g = f[prb_ref]
            except Exception:
                print(f"  Probe {prb}: empty")
                continue

            quality_ds = prb_g['quality']
            n_clu = quality_ds.shape[0]
            qualities = []
            for i in range(n_clu):
                try:
                    q = h5str(f, quality_ds[i, 0]).strip()
                except Exception:
                    q = ''
                qualities.append(q)

            valid_clu = [i for i, q in enumerate(qualities)
                         if q.lower() not in EXCLUDED_QUALITIES and q != '']
            q_counts = {}
            for q in qualities:
                q_counts[q] = q_counts.get(q, 0) + 1
            print(f"  Probe {prb}: {n_clu} total, {len(valid_clu)} valid  {q_counts}")

            for ci in valid_clu:
                spike_data.append({
                    'trial':   f[prb_g['trial'][ci, 0]][()].flatten(),
                    'trialtm': f[prb_g['trialtm'][ci, 0]][()].flatten(),
                })

        # DLC bottom cam - read all trial data into memory
        traj = obj['traj']
        bot_trials_data = {}
        try:
            vbot_ref = traj[1, 0]
            vbot = f[vbot_ref]
            fn_ref = vbot['featNames'][0, 0]
            fn_arr = f[fn_ref][:]
            feat_names = [h5str(f, fn_arr[0, i]) for i in range(fn_arr.shape[1])]

            for ti in range(n_trials):
                try:
                    ts = f[vbot['ts'][ti, 0]][()]
                    ft = f[vbot['frameTimes'][ti, 0]][()].flatten()
                    bot_trials_data[ti] = (ts, ft)
                except Exception:
                    pass

            bot_cam = {'feat_names': feat_names, 'trials_data': bot_trials_data}
        except Exception as e:
            print(f"  WARNING: DLC bottom cam: {e}")

        # side cam frameTimes
        try:
            vside_ref = traj[0, 0]
            vside = f[vside_ref]
            for ti in range(n_trials):
                try:
                    side_ft[ti] = f[vside['frameTimes'][ti, 0]][()].flatten()
                except Exception:
                    pass
        except Exception as e:
            print(f"  WARNING: side cam frameTimes: {e}")

    return {
        'n_trials': n_trials, 'hit': hit, 'miss': miss, 'early': early,
        'no_resp': no_resp, 'L': L, 'R': R, 'autowater': autowater,
        'stim_en': stim_en, 'goCue': goCue, 'vidshift': vidshift,
        'spike_data': spike_data, 'bot_cam': bot_cam, 'side_ft': side_ft,
    }


def process_session(sess, data_dir):
    """Process one recording session. Returns (result_dict, error_msg)."""

    anm, date = sess['anm'], sess['date']
    sname = f"{anm}_{date}"
    data_path = os.path.join(data_dir, sess['dir'],
                             f"data_structure_{sname}.mat")
    me_path = os.path.join(data_dir, sess['dir'],
                           f"motionEnergy_{sname}.mat")

    if not os.path.exists(data_path):
        return None, f"File not found: {data_path}"

    print(f"\n{'='*60}")
    print(f"Processing: {sname}  ({sess['dir']})")

    # ----------------------------------------------------------
    # Phase 1: Load data (auto-detect HDF5 vs v5)
    # ----------------------------------------------------------
    if _is_hdf5(data_path):
        sd = _load_session_h5(data_path, sess)
    else:
        sd = _load_session_v5(data_path, sess)

    n_trials = sd['n_trials']
    hit, miss, early = sd['hit'], sd['miss'], sd['early']
    no_resp = sd['no_resp']
    L, R = sd['L'], sd['R']
    autowater, stim_en = sd['autowater'], sd['stim_en']
    goCue = sd['goCue']
    vidshift = sd['vidshift']
    spike_data = sd['spike_data']
    bot_cam = sd['bot_cam']
    side_ft = sd['side_ft']

    print(f"  Trials: {n_trials}  hit={hit.sum()} miss={miss.sum()} "
          f"early={early.sum()} no={no_resp.sum()} stim={stim_en.sum()} "
          f"WC={autowater.sum()}")

    # trial filter: (hit|miss) & ~stim & ~early
    use = ((hit == 1) | (miss == 1)) & (stim_en == 0) & (early == 0)
    trial_idx = np.where(use)[0]

    if len(trial_idx) < MIN_TRIALS:
        return None, f"Only {len(trial_idx)} usable trials"

    n_raw = len(spike_data)
    if n_raw < MIN_UNITS:
        return None, f"Only {n_raw} neurons"

    # --- bin spikes, smooth (getSeq.m) ---
    fr = np.zeros((n_raw, T_BINS, n_trials), dtype=np.float32)
    for ni, spk in enumerate(spike_data):
        for ti in range(n_trials):
            mask = spk['trial'] == (ti + 1)
            if not np.any(mask):
                continue
            aligned = spk['trialtm'][mask] - goCue[ti]
            counts, _ = np.histogram(aligned, bins=EDGES)
            fr[ni, :, ti] = causal_smooth(counts.astype(np.float64) / DT)

    # --- remove low FR (removeLowFRClusters.m) ---
    mean_fr = np.mean(fr, axis=(1, 2))
    keep = mean_fr > LOW_FR
    fr = fr[keep]
    n_neurons = int(keep.sum())
    print(f"  Neurons: {n_raw} → {n_neurons}  (FR filter)")

    if n_neurons < MIN_UNITS:
        return None, f"Only {n_neurons} neurons after FR filter"

    # --- DLC velocities (bottom cam) ---
    tongue_vel = np.zeros((T_BINS, n_trials), dtype=np.float32)
    paw_vel    = np.zeros((T_BINS, n_trials), dtype=np.float32)

    if bot_cam is not None:
        try:
            feat_names = bot_cam['feat_names']
            tongue_fi = feat_names.index('top_tongue') if 'top_tongue' in feat_names else 0
            paw_fi = -1
            for pname in ['top_paw', 'bottom_paw']:
                if pname in feat_names:
                    paw_fi = feat_names.index(pname)
                    break

            for ti in range(n_trials):
                try:
                    if 'get_trial' in bot_cam:
                        # v5 format
                        ts, ft = bot_cam['get_trial'](ti)
                        # ts shape: (nframes, 3, nfeat)
                        t_spd = compute_speed(
                            ts[:, 0, tongue_fi], ts[:, 1, tongue_fi], ft,
                            ts[:, 2, tongue_fi], DLC_CONF, fill_missing=True)
                    else:
                        # h5 format - ts shape: (nfeat, 3, nframes)
                        if ti not in bot_cam['trials_data']:
                            continue
                        ts, ft = bot_cam['trials_data'][ti]
                        t_spd = compute_speed(
                            ts[tongue_fi, 0], ts[tongue_fi, 1], ft,
                            ts[tongue_fi, 2], DLC_CONF, fill_missing=True)

                    aln = ft - vidshift - goCue[ti]
                    interp_fn = interp1d(aln, t_spd, kind='linear',
                                         bounds_error=False, fill_value=np.nan)
                    tongue_vel[:, ti] = fill_nan_nearest(interp_fn(TIME_AXIS))

                    # paw
                    if paw_fi >= 0:
                        if 'get_trial' in bot_cam:
                            ts_p, ft_p = ts, ft
                            p_spd = compute_speed(
                                ts_p[:, 0, paw_fi], ts_p[:, 1, paw_fi], ft_p,
                                ts_p[:, 2, paw_fi], DLC_CONF, fill_missing=True)
                        else:
                            if ti not in bot_cam['trials_data']:
                                continue
                            ts_p, ft_p = bot_cam['trials_data'][ti]
                            p_spd = compute_speed(
                                ts_p[paw_fi, 0], ts_p[paw_fi, 1], ft_p,
                                ts_p[paw_fi, 2], DLC_CONF, fill_missing=True)
                        interp_fn = interp1d(aln, p_spd, kind='linear',
                                             bounds_error=False, fill_value=np.nan)
                        paw_vel[:, ti] = fill_nan_nearest(interp_fn(TIME_AXIS))
                except Exception:
                    pass
        except Exception as e:
            print(f"  WARNING: DLC processing: {e}")

    # ----------------------------------------------------------
    # Phase 2: motion energy (scipy.io, separate file)
    # ----------------------------------------------------------
    me_data = np.zeros((T_BINS, n_trials), dtype=np.float32)

    if os.path.exists(me_path):
        try:
            me_mat = scipy.io.loadmat(me_path, squeeze_me=False)
            me_struct = me_mat['me']
            me_trials = me_struct['data'][0, 0]

            for ti in range(min(n_trials, me_trials.shape[0])):
                trial_me = me_trials[ti, 0].flatten().astype(np.float64)

                if ti in side_ft:
                    aln = side_ft[ti] - vidshift - goCue[ti]
                    if len(trial_me) != len(aln):
                        aln = np.arange(len(trial_me)) / 400.0 - 0.5 - goCue[ti]
                else:
                    aln = np.arange(len(trial_me)) / 400.0 - 0.5 - goCue[ti]

                interp_fn = interp1d(aln, trial_me, kind='linear',
                                     bounds_error=False, fill_value=np.nan)
                me_data[:, ti] = fill_nan_nearest(interp_fn(TIME_AXIS))

            print(f"  Loaded motion energy")
        except Exception as e:
            print(f"  WARNING: motion energy: {e}")
    else:
        print(f"  WARNING: no motion energy file")

    # ----------------------------------------------------------
    # Phase 3: assemble trials
    # ----------------------------------------------------------
    # Per-session 50th percentile thresholds for discretization
    # For tongue: use 50th percentile of non-zero values since tongue is
    # only visible during licking (~4% of frames). Zero values (tongue not
    # visible) are automatically "low".
    tongue_usable = tongue_vel[:, trial_idx].flatten()
    tongue_nonzero = tongue_usable[tongue_usable > 0]
    if len(tongue_nonzero) > 0:
        tongue_thresh = np.percentile(tongue_nonzero, 50)
    else:
        tongue_thresh = 1.0  # no tongue data: all low
    paw_thresh    = np.percentile(paw_vel[:, trial_idx], 50)
    me_thresh     = np.percentile(me_data[:, trial_idx], 50)

    neural_list, input_list, output_list = [], [], []

    for ti in trial_idx:
        neural_list.append(fr[:, :, ti].copy())                     # (n_neurons, T)
        input_list.append(TIME_AXIS[np.newaxis, :].astype(np.float32).copy())  # (1, T)

        out = np.zeros((6, T_BINS), dtype=np.int64)
        out[0, :] = int(R[ti])                  # lick direction: L=0, R=1
        out[1, :] = 1 - int(autowater[ti])       # context: WC=0, DR=1
        out[2, :] = int(hit[ti])                 # outcome: incorrect=0, correct=1
        out[3, :] = (tongue_vel[:, ti] >= tongue_thresh).astype(np.int64)
        out[4, :] = (paw_vel[:, ti] >= paw_thresh).astype(np.int64)
        out[5, :] = (me_data[:, ti] >= me_thresh).astype(np.int64)
        output_list.append(out)

    brain_region_idx = np.zeros(n_neurons, dtype=np.int64)  # all ALM

    print(f"  DONE: {len(trial_idx)} trials, {n_neurons} neurons")

    return {
        'neural': neural_list,
        'input':  input_list,
        'output': output_list,
        'anm': anm,
        'n_neurons': n_neurons,
        'brain_region_idx': brain_region_idx,
        'info': {
            'animal': anm, 'date': date, 'dataset': sess['dir'],
            'n_trials_total': n_trials, 'n_trials_used': len(trial_idx),
            'n_neurons': n_neurons,
            'tongue_thresh': float(tongue_thresh),
            'paw_thresh': float(paw_thresh),
            'me_thresh': float(me_thresh),
        },
    }, None


# ============================================================
# MAIN
# ============================================================

def convert_all(sessions, data_dir, output_path):
    """Convert all sessions and save to pickle."""

    all_neural, all_input, all_output = [], [], []
    all_subject_idx, all_br_idx = [], []
    session_infos = []
    subjects = {}  # name → index

    for sess in sessions:
        result, err = process_session(sess, data_dir)
        if result is None:
            print(f"  SKIPPED: {err}")
            continue

        all_neural.append(result['neural'])
        all_input.append(result['input'])
        all_output.append(result['output'])

        anm = result['anm']
        if anm not in subjects:
            subjects[anm] = len(subjects)
        all_subject_idx.append(subjects[anm])
        all_br_idx.append(result['brain_region_idx'])
        session_infos.append(result['info'])

    # ordered subject list
    subj_list = [''] * len(subjects)
    for name, idx in subjects.items():
        subj_list[idx] = name

    data = {
        'neural': all_neural,
        'input':  all_input,
        'output': all_output,
        'subjects': subj_list,
        'subject_idx': np.array(all_subject_idx, dtype=np.int64),
        'brain_regions': ['ALM'],
        'brain_region_idx': all_br_idx,
        'input_names': ['time_from_gocue'],
        'output_names': [
            'lick_direction', 'behavioral_context', 'outcome',
            'tongue_velocity', 'paw_velocity', 'motion_energy',
        ],
        'output_values': [
            ['left', 'right'],
            ['WC', 'DR'],
            ['incorrect', 'correct'],
            ['low', 'high'],
            ['low', 'high'],
            ['low', 'high'],
        ],
        'metadata': {
            'task_description': (
                'Two-context delayed response (DR) and water-cued (WC) '
                'licking task with ALM electrophysiology recordings. '
                'Mice performed directional licking tasks alternating block-wise.'
            ),
            'time_bin_size': DT * 1000,   # ms
            'temporal_alignment_event': 'Go cue onset',
            'off_start': TMIN,
            'off_end': TMAX,
            'smooth_window_bins': SMOOTH_N,
            'smooth_type': 'causal_gaussian_reflect',
            'low_fr_threshold_hz': LOW_FR,
            'quality_filter': "all (exclude garbage, gabrga, noisy, real?)",
            'trial_filter': '(hit|miss) & ~stim.enable & ~early',
            'session_info': session_infos,
        },
    }

    with open(output_path, 'wb') as fp:
        pickle.dump(data, fp)

    n_sess = len(all_neural)
    n_total = sum(len(s) for s in all_neural)
    print(f"\n{'='*60}")
    print(f"Saved {n_sess} sessions ({n_total} trials) to {output_path}")
    print(f"Subjects ({len(subj_list)}): {subj_list}")
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert data for decoder.')
    parser.add_argument('--output', default='converted_data.pkl',
                        help='Output pickle path')
    parser.add_argument('--sample', type=int, default=0,
                        help='Process only first N sessions (0=all)')
    args = parser.parse_args()

    sessions = ALL_SESSIONS
    if args.sample > 0:
        sessions = sessions[:args.sample]

    convert_all(sessions, DATA_DIR, args.output)
