"""
Convert Hasnain, Birnbaum et al (Nature Neuroscience 2024) data to decoder format.

Data: ALM electrophysiology + behavior + video from delayed-response (DR) and
water-cued (WC) licking tasks in head-fixed mice.

Processing follows the reference code pipeline (WorkingWithDataObjs.m, loadSessionData.m,
getSeq.m, etc.) with parameters:
  - Alignment: go cue onset
  - Time window: -2.5 to 2.5 s
  - Bin size: 10 ms (dt = 1/100)
  - Smoothing: causal Gaussian kernel, window = 15 bins, reflect boundary
  - Cluster quality: 'all' (exclude garbage, gabrga, noisy, real?)
  - Firing rate threshold: > 1 Hz (mean across all trials)
  - Trial filtering: exclude stim trials and early-lick trials
  - Sessions: all 25 Ephys_Behavior sessions (10 animals, ALM probe only)
"""

import numpy as np
import h5py
import scipy.io
import os
import pickle
from pathlib import Path

# ─── Parameters ───────────────────────────────────────────────────────────────

TMIN = -2.5          # seconds relative to go cue
TMAX = 2.5
DT = 1 / 100         # 10 ms bins
SMOOTH_WIN = 15       # causal Gaussian kernel window (bins)
SMOOTH_BC = 'reflect' # boundary condition
LOW_FR = 1.0          # minimum mean firing rate (Hz)
PAD_SEC = 0.5         # video-neural offset (sglx padding)
EXCLUDE_QUALITIES = {'garbage', 'gabrga', 'noisy', 'real?'}

DATA_DIR = Path('/app/data/Ephys_Behavior')
OUT_PATH = Path('/app/converted_data.pkl')

# Probe mapping: (animal, date) -> list of 0-indexed ALM probe numbers
# Derived from DataLoadingScripts/Recording and video/load*_ALMVideo.m
PROBE_MAP = {
    ('EKH1', '2021-08-07'): [1],
    ('EKH3', '2021-08-11'): [1],
    ('JEB6', '2021-04-18'): [1],
    ('JEB7', '2021-04-29'): [0],
    ('JEB7', '2021-04-30'): [0],
    ('JEB13', '2022-09-13'): [1],
    ('JEB13', '2022-09-14'): [1],
    ('JEB13', '2022-09-21'): [0],
    ('JEB13', '2022-09-24'): [0],
    ('JEB13', '2022-09-25'): [0],
    ('JEB14', '2022-08-22'): [0],
    ('JEB14', '2022-08-23'): [0],
    ('JEB14', '2022-08-24'): [0],
    ('JEB14', '2022-08-25'): [0],
    ('JEB15', '2022-07-26'): [0, 1],
    ('JEB15', '2022-07-27'): [0, 1],
    ('JEB15', '2022-07-28'): [0, 1],
    ('JEB15', '2022-07-29'): [1],
    ('JEB19', '2023-04-18'): [0],
    ('JEB19', '2023-04-19'): [0],
    ('JEB19', '2023-04-20'): [0],
    ('JEB19', '2023-04-21'): [0],
    ('JGR2', '2021-11-16'): [0],
    ('JGR2', '2021-11-17'): [0],
    ('JGR3', '2021-11-18'): [0],
}


# ─── Utility functions ────────────────────────────────────────────────────────

def h5_deref(f, ref):
    """Dereference an HDF5 object reference."""
    return f[ref]


def h5_read_string(f, dataset):
    """Read a string stored as uint16 array in HDF5."""
    data = dataset[()]
    if data.dtype == np.uint16 or data.dtype == np.float64:
        chars = data.flatten().astype(int)
        return ''.join(chr(c) for c in chars if c > 0)
    if isinstance(data, bytes):
        return data.decode('utf-8')
    return str(data)


def h5_read_string_from_ref(f, ref):
    """Read a string from an HDF5 object reference."""
    obj = f[ref]
    data = obj[()]
    if data.dtype == np.uint16 or data.dtype == np.float64:
        chars = data.flatten().astype(int)
        return ''.join(chr(c) for c in chars if c > 0)
    return str(data)


def causal_gaussian_kernel(N):
    """Create a causal Gaussian kernel matching MATLAB mySmooth.m.

    gausswin(N) in MATLAB, then zero out first half, normalize.
    """
    # MATLAB gausswin(N): w(n) = exp(-0.5 * ((n - (N-1)/2) / (alpha * (N-1)/2))^2)
    # with alpha = 2.5 (default)
    alpha = 2.5
    n = np.arange(N)
    center = (N - 1) / 2.0
    sigma = center / alpha
    kern = np.exp(-0.5 * ((n - center) / sigma) ** 2)
    # Make causal: zero out first half
    kern[:int(np.floor(len(kern) / 2))] = 0
    kern = kern / kern.sum()
    return kern


def smooth_data(x, N, bctype='reflect'):
    """Smooth 1D array with causal Gaussian kernel, matching mySmooth.m."""
    if N <= 1:
        return x.copy()

    kern = causal_gaussian_kernel(N)

    if bctype == 'reflect':
        x_filt = np.concatenate([x[:N], x])
        trim = N
    elif bctype == 'zeropad':
        x_filt = np.concatenate([np.zeros(N), x])
        trim = N
    else:
        x_filt = x.copy()
        trim = 0

    out = np.convolve(x_filt, kern, mode='same')
    out = out[trim:]
    return out


def bin_and_smooth_spikes(spike_times, edges, dt, smooth_win, bctype):
    """Bin spike times into histogram and smooth to get firing rate."""
    counts, _ = np.histogram(spike_times, bins=edges)
    # Convert to firing rate (spks/sec)
    fr = counts.astype(float) / dt
    # Smooth
    fr = smooth_data(fr, smooth_win, bctype)
    return fr


def compute_speed(x, y, dt_video):
    """Compute speed from x, y position time series."""
    vx = np.gradient(x, dt_video)
    vy = np.gradient(y, dt_video)
    speed = np.sqrt(vx**2 + vy**2)
    return speed


def nearest_fill(arr):
    """Fill NaN values with nearest non-NaN value (forward then backward)."""
    out = arr.copy()
    mask = np.isnan(out)
    if mask.all():
        return out
    # Forward fill
    for i in range(1, len(out)):
        if np.isnan(out[i]) and not np.isnan(out[i-1]):
            out[i] = out[i-1]
    # Backward fill
    for i in range(len(out)-2, -1, -1):
        if np.isnan(out[i]) and not np.isnan(out[i+1]):
            out[i] = out[i+1]
    return out


# ─── Data loading functions ───────────────────────────────────────────────────

def load_session(data_path, me_path, alm_probes):
    """Load a single session's data from HDF5 .mat file.

    Returns a dict with all needed data, or None if session should be skipped.
    """
    print(f"  Loading {os.path.basename(data_path)}...")

    with h5py.File(data_path, 'r') as f:
        obj = f['obj']

        # ── Trial info ──
        bp = obj['bp']
        ntrials = int(bp['Ntrials'][0, 0])

        hit = bp['hit'][0, :].astype(bool)
        miss = bp['miss'][0, :].astype(bool)
        no = bp['no'][0, :].astype(bool)
        R = bp['R'][0, :].astype(bool)
        L = bp['L'][0, :].astype(bool)
        autowater = bp['autowater'][0, :].astype(bool)
        early = bp['early'][0, :].astype(bool)

        # Stim enable
        stim_enable = bp['stim']['enable'][0, :].astype(bool)

        # Event times
        ev = bp['ev']
        goCue = ev['goCue'][0, :]
        sample_times = ev['sample'][0, :]
        delay_times = ev['delay'][0, :]

        # Lick times (cell arrays of variable-length arrays)
        lickL_refs = ev['lickL'][0, :]
        lickR_refs = ev['lickR'][0, :]

        # ── Trial filtering ──
        # Exclude stim trials and early-lick trials
        # Include hit, miss, no (ignore) trials
        valid_mask = (hit | miss | no) & ~stim_enable & ~early
        valid_trials = np.where(valid_mask)[0]  # 0-indexed trial indices

        if len(valid_trials) < 2:
            print(f"    Skipping: only {len(valid_trials)} valid trials")
            return None

        # ── Determine lick direction per trial ──
        # R&hit or L&miss → right lick; L&hit or R&miss → left lick; no → none
        lick_direction = np.full(ntrials, -1, dtype=int)
        lick_direction[(R & hit) | (L & miss)] = 1   # right
        lick_direction[(L & hit) | (R & miss)] = 0   # left
        lick_direction[no] = 2                         # none

        # ── Context ──
        context = np.where(autowater, 0, 1).astype(int)  # 0=WC, 1=DR

        # ── Outcome ──
        outcome = np.full(ntrials, -1, dtype=int)
        outcome[miss] = 0   # incorrect
        outcome[hit] = 1    # correct
        outcome[no] = 2     # ignore

        # ── Neural data: load clusters from ALM probes ──
        clu_group = obj['clu']

        all_spike_times_aligned = []
        all_spike_trials = []
        all_qualities = []

        for prb_idx in alm_probes:
            # Dereference probe
            prb_ref = clu_group[prb_idx, 0]
            prb_data = f[prb_ref]

            # Get number of clusters
            n_clusters = prb_data['quality'].shape[0]

            for clu_i in range(n_clusters):
                # Get quality
                quality_ref = prb_data['quality'][clu_i, 0]
                try:
                    quality = h5_read_string(f, f[quality_ref])
                except:
                    quality = ''
                quality = quality.strip()

                # Filter by quality
                if quality.lower() in {q.lower() for q in EXCLUDE_QUALITIES}:
                    continue
                if quality == '':
                    continue

                # Get spike times in trial (trialtm) and trial assignments
                trialtm_ref = prb_data['trialtm'][clu_i, 0]
                trial_ref = prb_data['trial'][clu_i, 0]

                trialtm = f[trialtm_ref][()].flatten()
                trial_nums = f[trial_ref][()].flatten().astype(int)

                # Align to go cue: subtract go cue time for each spike
                aligned_times = np.empty_like(trialtm)
                for t_idx in range(len(trialtm)):
                    tr = trial_nums[t_idx] - 1  # 0-indexed
                    if 0 <= tr < ntrials:
                        aligned_times[t_idx] = trialtm[t_idx] - goCue[tr]
                    else:
                        aligned_times[t_idx] = np.nan

                all_spike_times_aligned.append(aligned_times)
                all_spike_trials.append(trial_nums)
                all_qualities.append(quality)

        n_units = len(all_spike_times_aligned)
        if n_units < 10:
            print(f"    Skipping: only {n_units} units (need >= 10)")
            return None

        # ── Bin and smooth spikes ──
        edges = np.arange(TMIN, TMAX + DT, DT)
        # Ensure exactly the right number of bins
        n_timebins = len(edges) - 1
        time_vec = edges[:-1] + DT / 2

        # Compute single-trial firing rates: (n_units, ntrials, n_timebins)
        trialdat = np.zeros((n_units, ntrials, n_timebins))
        for u_idx in range(n_units):
            stimes = all_spike_times_aligned[u_idx]
            strials = all_spike_trials[u_idx]
            for tr in range(ntrials):
                tr_mask = strials == (tr + 1)  # 1-indexed trials
                if not tr_mask.any():
                    continue
                tr_spikes = stimes[tr_mask]
                tr_spikes = tr_spikes[~np.isnan(tr_spikes)]
                trialdat[u_idx, tr, :] = bin_and_smooth_spikes(
                    tr_spikes, edges, DT, SMOOTH_WIN, SMOOTH_BC
                )

        # ── Remove low-FR units ──
        # Mean FR across all trials and time
        mean_fr = np.mean(np.mean(trialdat, axis=2), axis=1)
        fr_mask = mean_fr > LOW_FR
        trialdat = trialdat[fr_mask]
        n_units_final = trialdat.shape[0]

        if n_units_final < 10:
            print(f"    Skipping: only {n_units_final} units after FR filtering (need >= 10)")
            return None

        # ── Video data: tongue velocity ──
        traj_group = obj['traj']

        tongue_speed_trials = {}
        paw_speed_trials = {}
        side_frame_times = {}  # store aligned frame times per trial for ME

        def read_feat_names(f, view_data):
            """Read feature names from traj view (double-deref HDF5 cell array)."""
            feat_cell_ref = view_data['featNames'][0, 0]  # first trial
            feat_cell = f[feat_cell_ref][()]  # (1, n_feats) refs
            names = []
            for i in range(feat_cell.shape[1]):
                ref = feat_cell[0, i]
                chars = f[ref][()].flatten()
                names.append(''.join(chr(int(c)) for c in chars if c > 0))
            return names

        try:
            # Side cam (view 0) for tongue
            side_data = f[traj_group[0, 0]]
            side_feats = read_feat_names(f, side_data)
            tongue_idx = side_feats.index('tongue') if 'tongue' in side_feats else None

            # Bottom cam (view 1) for paw
            bot_data = f[traj_group[1, 0]]
            bot_feats = read_feat_names(f, bot_data)
            paw_idx = bot_feats.index('top_paw') if 'top_paw' in bot_feats else None

            dt_video = 1.0 / 400  # 400 Hz

            for tr in range(ntrials):
                if not valid_mask[tr]:
                    continue

                # ── Tongue from side cam ──
                if tongue_idx is not None:
                    try:
                        ft = f[side_data['frameTimes'][tr, 0]][()].flatten()
                        aligned_ft = ft - PAD_SEC - goCue[tr]
                        side_frame_times[tr] = aligned_ft

                        ts = f[side_data['ts'][tr, 0]][()]  # (n_feats, 3, n_frames)
                        tongue_x = ts[tongue_idx, 0, :]
                        tongue_y = ts[tongue_idx, 1, :]
                        tongue_nan = np.isnan(tongue_x) | np.isnan(tongue_y)

                        if tongue_nan.all():
                            tongue_speed_trials[tr] = None
                        else:
                            visible = ~tongue_nan
                            speed = np.full_like(tongue_x, np.nan)
                            if visible.sum() > 1:
                                tx_filled = nearest_fill(tongue_x)
                                ty_filled = nearest_fill(tongue_y)
                                speed_raw = compute_speed(tx_filled, ty_filled, dt_video)
                                speed[visible] = speed_raw[visible]
                            tongue_speed_trials[tr] = (aligned_ft, speed, visible)
                    except Exception:
                        tongue_speed_trials[tr] = None

                # ── Paw from bottom cam ──
                if paw_idx is not None:
                    try:
                        ft = f[bot_data['frameTimes'][tr, 0]][()].flatten()
                        aligned_ft = ft - PAD_SEC - goCue[tr]
                        ts = f[bot_data['ts'][tr, 0]][()]

                        paw_x = ts[paw_idx, 0, :]
                        paw_y = ts[paw_idx, 1, :]
                        paw_nan = np.isnan(paw_x) | np.isnan(paw_y)

                        if paw_nan.all():
                            paw_speed_trials[tr] = None
                        else:
                            px = nearest_fill(paw_x)
                            py = nearest_fill(paw_y)
                            speed_paw = compute_speed(px, py, dt_video)
                            paw_visible = ~paw_nan
                            paw_speed_trials[tr] = (aligned_ft, speed_paw, paw_visible)
                    except Exception:
                        paw_speed_trials[tr] = None

            has_video = True
        except Exception as e:
            print(f"    Warning: could not load video data: {e}")
            has_video = False

    # ── Motion energy ──
    try:
        me_data = scipy.io.loadmat(me_path)
        me_struct = me_data['me']
        me_trial_data = me_struct['data'][0, 0]  # (ntrials, 1) cell array
        me_thresh = float(me_struct['moveThresh'][0, 0].flatten()[0])
        has_me = True
    except Exception as e:
        print(f"    Warning: could not load motion energy: {e}")
        has_me = False

    # ── Interpolate video features to neural time bins ──
    neural_time = time_vec  # (n_timebins,)

    # Tongue velocity per trial, discretized
    tongue_vel_all = {}
    paw_vel_all = {}
    me_all = {}

    # First pass: collect all speeds for percentile computation
    all_tongue_speeds = []
    all_paw_speeds = []
    all_me_values = []

    for tr in valid_trials:
        # Tongue
        if has_video and tr in tongue_speed_trials and tongue_speed_trials[tr] is not None:
            ft, speed, visible = tongue_speed_trials[tr]
            # Interpolate speed and visibility to neural time bins
            # For speed: use NaN for invisible frames, interpolate visible frames
            speed_for_interp = np.where(np.isfinite(speed), speed, 0.0)
            speed_interp = np.interp(neural_time, ft, speed_for_interp,
                                     left=np.nan, right=np.nan)
            visible_interp = np.interp(neural_time, ft, visible.astype(float),
                                      left=0, right=0) > 0.5
            # Mask: only count as visible if both interpolation is valid and source was visible
            speed_interp[~visible_interp] = np.nan
            tongue_vel_all[tr] = (speed_interp, visible_interp)
            valid_speeds = speed_interp[visible_interp & np.isfinite(speed_interp)]
            all_tongue_speeds.extend(valid_speeds.tolist())

        # Paw
        if has_video and tr in paw_speed_trials and paw_speed_trials[tr] is not None:
            ft, speed, visible = paw_speed_trials[tr]
            speed_interp = np.interp(neural_time, ft, speed,
                                     left=np.nan, right=np.nan)
            visible_interp = np.interp(neural_time, ft, visible.astype(float),
                                      left=0, right=0) > 0.5
            speed_interp[~visible_interp] = np.nan
            paw_vel_all[tr] = (speed_interp, visible_interp)
            valid_speeds = speed_interp[visible_interp & np.isfinite(speed_interp)]
            all_paw_speeds.extend(valid_speeds.tolist())

        # Motion energy - use stored side cam frame times
        if has_me:
            try:
                me_trial = me_trial_data[tr, 0].flatten()
                if len(me_trial) > 1 and tr in side_frame_times:
                    ft = side_frame_times[tr]
                    if len(me_trial) == len(ft):
                        me_interp = np.interp(neural_time, ft, me_trial,
                                             left=np.nan, right=np.nan)
                        me_all[tr] = me_interp
                        all_me_values.extend(me_interp[np.isfinite(me_interp)].tolist())
                    else:
                        me_all[tr] = None
                else:
                    me_all[tr] = None
            except:
                me_all[tr] = None

    # Compute 50th percentile thresholds
    tongue_thresh = np.median(all_tongue_speeds) if len(all_tongue_speeds) > 0 else 0
    paw_thresh = np.median(all_paw_speeds) if len(all_paw_speeds) > 0 else 0
    me_thresh_50 = np.median(all_me_values) if len(all_me_values) > 0 else 0

    # ── Build output arrays ──
    session_neural = []
    session_input = []
    session_output = []

    for tr in valid_trials:
        # Neural: (n_neurons, n_timepoints)
        neural_trial = trialdat[:, tr, :]  # (n_units_final, n_timebins)
        session_neural.append(neural_trial.astype(np.float32))

        # Input: time from go cue (1, n_timepoints)
        input_trial = neural_time.reshape(1, -1).astype(np.float32)
        session_input.append(input_trial)

        # Output: (6, n_timepoints)
        output_trial = np.zeros((6, n_timebins), dtype=np.int64)

        # 0: lick direction (per-trial, broadcast)
        output_trial[0, :] = lick_direction[tr]

        # 1: behavioral context (per-trial, broadcast)
        output_trial[1, :] = context[tr]

        # 2: outcome (per-trial, broadcast)
        output_trial[2, :] = outcome[tr]

        # 3: tongue velocity (time-varying)
        if tr in tongue_vel_all and tongue_vel_all[tr] is not None:
            speed_interp, visible_interp = tongue_vel_all[tr]
            tv = np.full(n_timebins, 2, dtype=np.int64)  # default: not visible
            vis = visible_interp & np.isfinite(speed_interp)
            tv[vis & (speed_interp < tongue_thresh)] = 0
            tv[vis & (speed_interp >= tongue_thresh)] = 1
            output_trial[3, :] = tv
        else:
            output_trial[3, :] = 2  # not visible

        # 4: paw velocity (time-varying)
        if tr in paw_vel_all and paw_vel_all[tr] is not None:
            speed_interp, visible_interp = paw_vel_all[tr]
            pv = np.full(n_timebins, 2, dtype=np.int64)  # default: not visible
            vis = visible_interp & np.isfinite(speed_interp)
            pv[vis & (speed_interp < paw_thresh)] = 0
            pv[vis & (speed_interp >= paw_thresh)] = 1
            output_trial[4, :] = pv
        else:
            output_trial[4, :] = 2  # not visible

        # 5: motion energy (time-varying)
        if tr in me_all and me_all[tr] is not None:
            me_interp = me_all[tr]
            me_disc = np.full(n_timebins, 2, dtype=np.int64)  # default: no video
            vis = np.isfinite(me_interp)
            me_disc[vis & (me_interp < me_thresh_50)] = 0
            me_disc[vis & (me_interp >= me_thresh_50)] = 1
            output_trial[5, :] = me_disc
        else:
            output_trial[5, :] = 2  # no video

        session_output.append(output_trial)

    # Get animal name
    with h5py.File(data_path, 'r') as f:
        try:
            anm = h5_read_string(f, f['obj']['meta']['anm'])
        except:
            # Extract from filename
            basename = os.path.basename(data_path)
            parts = basename.replace('data_structure_', '').replace('.mat', '').split('_')
            anm = parts[0]

    print(f"    {n_units_final} units, {len(valid_trials)} trials")

    return {
        'neural': session_neural,
        'input': session_input,
        'output': session_output,
        'animal': anm,
        'n_units': n_units_final,
        'n_trials': len(valid_trials),
    }


def main():
    print("Converting Hasnain & Birnbaum et al. data...")

    # Find all session files
    data_files = sorted(DATA_DIR.glob('data_structure_*.mat'))

    all_sessions_neural = []
    all_sessions_input = []
    all_sessions_output = []
    all_animals = []
    session_animals = []

    for data_file in data_files:
        basename = data_file.stem.replace('data_structure_', '')
        parts = basename.split('_', 1)
        anm = parts[0]
        date = parts[1]

        key = (anm, date)
        if key not in PROBE_MAP:
            print(f"  Skipping {basename}: no probe mapping")
            continue

        me_file = data_file.parent / f'motionEnergy_{anm}_{date}.mat'
        me_path = str(me_file) if me_file.exists() else None

        alm_probes = PROBE_MAP[key]

        result = load_session(str(data_file), me_path, alm_probes)

        if result is None:
            continue

        all_sessions_neural.append(result['neural'])
        all_sessions_input.append(result['input'])
        all_sessions_output.append(result['output'])
        session_animals.append(result['animal'])

    # Build subjects list
    unique_subjects = sorted(set(session_animals))
    subject_idx = np.array([unique_subjects.index(a) for a in session_animals])

    # Brain regions: all ALM
    brain_regions = ['ALM']
    brain_region_idx = []
    for sess_neural in all_sessions_neural:
        n_neurons = sess_neural[0].shape[0]
        brain_region_idx.append(np.zeros(n_neurons, dtype=int))

    # Build output dict
    data = {
        'neural': all_sessions_neural,
        'input': all_sessions_input,
        'output': all_sessions_output,

        'subjects': unique_subjects,
        'subject_idx': subject_idx,

        'brain_regions': brain_regions,
        'brain_region_idx': brain_region_idx,

        'input_names': ['time_from_go_cue'],
        'output_names': [
            'lick_direction',
            'behavioral_context',
            'outcome',
            'tongue_velocity',
            'paw_velocity',
            'motion_energy',
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
                'Head-fixed mice perform two alternating licking tasks: '
                'delayed-response (DR) with auditory cue and delay, and '
                'water-cued (WC) with random water presentation. '
                'Neural activity from ALM decoded to predict lick direction, '
                'behavioral context, outcome, and movement kinematics.'
            ),
            'time_bin_size': DT * 1000,  # 10 ms
            'temporal_alignment_event': 'Go cue onset',
            'off_start': TMIN,   # -2.5 s
            'off_end': TMAX,     # 2.5 s
            'recording_region': 'ALM (anterior lateral motor cortex)',
            'species': 'mouse (Mus musculus)',
            'smoothing': f'Causal Gaussian kernel, window={SMOOTH_WIN} bins, {SMOOTH_BC} BC',
            'firing_rate_threshold': f'>{LOW_FR} Hz mean across all trials',
            'cluster_quality': f'All except {EXCLUDE_QUALITIES}',
            'trial_filter': 'Exclude stim-enabled and early-lick trials',
            'reference': 'Hasnain, Birnbaum et al., Nature Neuroscience 2024',
        },
    }

    # Summary
    n_sessions = len(all_sessions_neural)
    n_subjects = len(unique_subjects)
    total_trials = sum(len(s) for s in all_sessions_neural)
    total_neurons = sum(s[0].shape[0] for s in all_sessions_neural)

    print(f"\nConversion complete:")
    print(f"  {n_sessions} sessions from {n_subjects} subjects")
    print(f"  {total_trials} total trials")
    print(f"  {total_neurons} total neurons")
    print(f"  Time bins: {all_sessions_neural[0][0].shape[1]} ({DT*1000:.0f} ms)")

    # Save
    with open(OUT_PATH, 'wb') as f:
        pickle.dump(data, f)
    print(f"\nSaved to {OUT_PATH}")


if __name__ == '__main__':
    main()
