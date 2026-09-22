"""
Convert ALM electrophysiology + behavior data from

    Hasnain, Birnbaum et al. (2024) "Separating cognitive and motor processes
    in the behaving mouse", Nature Neuroscience

into the dictionary format expected by `train_decoder.py`.

The conversion mirrors the authors' own MATLAB pipeline
(`DataLoadingScripts/`, `funcs/kinematics/`) as closely as possible:

  * sessions / probes         -> `DataLoadingScripts/Recording and video/load*_ALMVideo.m`
  * trial selection           -> `params.condition` in the figure scripts
                                 (`~stim.enable & ~early`)
  * unit selection            -> `findClusters.m` (quality ~= garbage/noisy/...)
                                 + `removeLowFRClusters.m` (mean FR > 1 Hz)
  * alignment / binning       -> `alignSpikes.m` + `getSeq.m`
                                 (goCue aligned, -2.5..2.5 s, 10 ms bins,
                                  causal gaussian smoothing, `mySmooth.m`)
  * video kinematics          -> `funcs/kinematics/getKinematicsFromVideo.m`
                                 (`findPosition.m`, `findVelocity.m`)
  * motion energy             -> `DataLoadingScripts/loadMotionEnergy.m`

Outputs `/app/converted_data.pkl`.
"""

import os
import pickle
import warnings
from collections import Counter

import h5py
import numpy as np
import scipy.io as sio

# ---------------------------------------------------------------------------
# Parameters (values taken from the paper's analysis scripts, e.g.
# `WorkingWithDataObjs.m`, `Scripts/EDFigure 2/EDFigure2a_Left.m`)
# ---------------------------------------------------------------------------
DATA_DIR = '/app/data/Ephys_Behavior'
OUT_FILE = '/app/converted_data.pkl'

ALIGN_EVENT = 'goCue'
TMIN = -2.5          # s, relative to the alignment event
TMAX = 2.5           # s
DT = 0.01            # s (10 ms bins, params.dt = 1/100)
SMOOTH_N = 15        # params.smooth, width of the causal gaussian kernel (bins)
LOW_FR = 1.0         # Hz, params.lowFR -- "all units with firing rates
                     # exceeding 1 Hz were included in all other analyses"
MIN_UNITS = 10       # "sessions were included only if they had at least 10 units"

# `findClusters.m`: quality label 'all' keeps everything except these
BAD_QUALITY = {'garbage', 'gabrga', 'noisy', 'real?'}

# Sessions and ALM probe(s), transcribed from
# `DataLoadingScripts/Recording and video/load<ANM>_ALMVideo.m`.
# These are exactly the 25 fixed-delay ALM recording sessions analysed in the
# paper (Figs 1, 2, 4, 8); the commented-out sessions in those files are
# excluded here as well.
SESSIONS = [
    ('EKH1',  '2021-08-07', [2]),
    ('EKH3',  '2021-08-11', [2]),
    ('JEB6',  '2021-04-18', [2]),
    ('JEB7',  '2021-04-29', [1]),
    ('JEB7',  '2021-04-30', [1]),
    ('JGR2',  '2021-11-16', [1]),
    ('JGR2',  '2021-11-17', [1]),
    ('JGR3',  '2021-11-18', [1]),
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
]

# DLC features used for the two kinematic outputs.  The tongue is tracked from
# the side camera (view 1) and the two paws from the bottom camera (view 2),
# following `params.traj_features` in the paper's scripts.
TONGUE_FEATURES = [(0, 'tongue')]
PAW_FEATURES = [(1, 'top_paw'), (1, 'bottom_paw')]

# time axis, identical to `obj.time` built in `getSeq.m`
EDGES = np.arange(TMIN, TMAX + DT / 2, DT)
TAXIS = (EDGES + DT / 2)[:-1]
NT = len(TAXIS)


# ---------------------------------------------------------------------------
# small helpers for reading MATLAB v7.3 (HDF5) files
# ---------------------------------------------------------------------------
def _char(dataset):
    """MATLAB char array -> python str."""
    arr = np.array(dataset).flatten()
    if arr.dtype in (np.uint16, np.uint8):
        return ''.join(chr(c) for c in arr)
    return ''


def _vec(group, key):
    return np.array(group[key]).flatten()


def _cellstr(f, dataset):
    """MATLAB cell array of strings -> list of str."""
    return [_char(f[ref]) for ref in np.array(dataset).flatten()]


def gausswin(n, alpha=2.5):
    """MATLAB's gausswin(n)."""
    k = np.arange(n)
    return np.exp(-0.5 * (alpha * (k - (n - 1) / 2) / ((n - 1) / 2)) ** 2)


def my_smooth(x, n=SMOOTH_N):
    """Port of `utils/mySmooth.m` with bctype='reflect' (operates on axis 0)."""
    if n <= 1:
        return x
    kern = gausswin(n)
    kern[:n // 2] = 0.0          # causal
    kern = kern / kern.sum()
    x = np.atleast_2d(x.T).T
    padded = np.concatenate([x[:n], x], axis=0)
    out = np.empty_like(padded)
    for j in range(padded.shape[1]):
        out[:, j] = np.convolve(padded[:, j], kern, mode='same')
    return out[n:]


def fill_nearest(x):
    """MATLAB's fillmissing(x,'nearest') for a 1-D array."""
    x = np.asarray(x, dtype=float)
    good = ~np.isnan(x)
    if not good.any():
        return x
    idx = np.arange(x.size)
    # index of the previous / next valid sample at every position
    prev = np.maximum.accumulate(np.where(good, idx, -1))
    nxt = np.minimum.accumulate(np.where(good, idx, x.size)[::-1])[::-1]
    prev_ok, nxt_ok = prev >= 0, nxt < x.size
    dprev = np.where(prev_ok, idx - prev, x.size)
    dnxt = np.where(nxt_ok, nxt - idx, x.size)
    take = np.where(dprev <= dnxt, np.where(prev_ok, prev, 0),
                    np.where(nxt_ok, nxt, 0))
    return x[take]


def interp_nan(t_new, t_old, y):
    """
    interp1(t_old, y, t_new) with MATLAB semantics: NaN outside the support and
    NaNs in `y` propagate to the neighbouring query points.
    """
    return np.interp(t_new, t_old, y, left=np.nan, right=np.nan)


# ---------------------------------------------------------------------------
# per-session loading
# ---------------------------------------------------------------------------
def video_shift(f, obj):
    """Port of `funcs/findVideoOffset.m`."""
    bit_start = _mode(_vec(obj['bp']['ev'], 'bitStart'))
    fs = _vec(obj['sglx'], 'fs')[0]
    vid_file_offset = _mode(_vec(obj['sglx']['bitcode'], 'bitstart')) / fs
    return vid_file_offset - bit_start


def _mode(x):
    x = x[~np.isnan(x)]
    vals, counts = np.unique(x, return_counts=True)
    return vals[counts.argmax()]


def load_behavior(obj):
    bp = obj['bp']
    n = int(_vec(bp, 'Ntrials')[0])
    beh = {
        'ntrials': n,
        'R': _vec(bp, 'R') > 0.5,
        'L': _vec(bp, 'L') > 0.5,
        'hit': _vec(bp, 'hit') > 0.5,
        'miss': _vec(bp, 'miss') > 0.5,
        'no': _vec(bp, 'no') > 0.5,
        'early': _vec(bp, 'early') > 0.5,
        'autowater': _vec(bp, 'autowater') > 0.5,
        'stim': np.abs(_vec(bp['stim'], 'enable')) > 0,
        'align': _vec(bp['ev'], ALIGN_EVENT),
    }
    return beh


def load_spikes(f, obj, probes, align_times, trial_mask):
    """
    Binned, smoothed single-trial firing rates, following `alignSpikes.m` and
    `getSeq.m`: spike times are expressed relative to the alignment event,
    histogrammed in `DT` bins over [TMIN, TMAX], converted to spikes/s and
    smoothed with a causal gaussian kernel.

    Returns (rates, qualities) where rates is (nunits, ntrials_kept, NT).
    """
    clu = np.array(obj['clu'])
    keep_trials = np.flatnonzero(trial_mask)
    trial_pos = -np.ones(len(align_times), dtype=int)
    trial_pos[keep_trials] = np.arange(keep_trials.size)

    rates = []
    qualities = []
    for prb in probes:
        cl = f[clu[prb - 1, 0]]
        quality_refs = np.array(cl['quality']).flatten()
        trial_refs = np.array(cl['trial']).flatten()
        tm_refs = np.array(cl['trialtm']).flatten()
        for iclu in range(quality_refs.size):
            qual = _char(f[quality_refs[iclu]]).strip()
            if qual.lower() in BAD_QUALITY:
                continue
            trials = np.array(f[trial_refs[iclu]]).flatten().astype(int) - 1
            times = np.array(f[tm_refs[iclu]]).flatten()
            # `alignSpikes.m`: trialtm_aligned = trialtm - event(trial)
            times = times - align_times[trials]

            counts = np.zeros((keep_trials.size, NT))
            pos = trial_pos[trials]
            inwin = (pos >= 0) & (times >= TMIN) & (times < TMAX)
            if inwin.any():
                bin_idx = np.floor((times[inwin] - TMIN) / DT).astype(int)
                np.add.at(counts, (pos[inwin], bin_idx), 1.0)
            rate = my_smooth((counts / DT).T).T     # smooth over time
            rates.append(rate)
            qualities.append(qual)

    if not rates:
        return np.zeros((0, keep_trials.size, NT)), []
    return np.stack(rates, axis=0), qualities


def load_video_traces(f, obj, align_times, ntrials, vidshift):
    """
    Per-trial DLC traces interpolated onto the analysis time axis, following
    `findPosition.m`.  Returns dicts of (ntrials, NT) arrays of x/y position
    (NaN where DeepLabCut did not detect the feature) plus a per-trial flag
    saying whether video covers the analysis window at all.
    """
    traj = np.array(obj['traj'])
    views = sorted({v for v, _ in TONGUE_FEATURES + PAW_FEATURES})
    has_video = {view: np.zeros(ntrials, dtype=bool) for view in views}
    frame_times = [None] * ntrials

    pos = {}
    for view, feat in TONGUE_FEATURES + PAW_FEATURES:
        pos[(view, feat)] = (np.full((ntrials, NT), np.nan),
                             np.full((ntrials, NT), np.nan))

    view_data = {}
    for view in views:
        tv = f[traj[view, 0]]
        names = _cellstr(f, f[np.array(tv['featNames']).flatten()[0]])
        wanted = [(feat, names.index(feat))
                  for v, feat in TONGUE_FEATURES + PAW_FEATURES
                  if v == view and feat in names]
        view_data[view] = (wanted,
                           np.array(tv['frameTimes']).flatten(),
                           np.array(tv['ts']).flatten())

    for trial in range(ntrials):
        for view, (wanted, ft_refs, ts_refs) in view_data.items():
            ft = np.array(f[ft_refs[trial]]).flatten()
            if ft.size == 0 or np.all(np.isnan(ft)):
                continue
            t_rel = ft - vidshift - align_times[trial]
            if t_rel[-1] < TMIN or t_rel[0] > TMAX:
                continue          # video does not overlap the analysis window
            has_video[view][trial] = True
            if view == 0:
                frame_times[trial] = t_rel
            ts = np.array(f[ts_refs[trial]])              # (nfeat, 3, nframes)
            for feat, fi in wanted:
                pos[(view, feat)][0][trial] = interp_nan(TAXIS, t_rel, ts[fi, 0])
                pos[(view, feat)][1][trial] = interp_nan(TAXIS, t_rel, ts[fi, 1])

    return pos, has_video, frame_times


def speed_from_position(x, y, fill):
    """
    Port of `findVelocity.m`: velocity is the gradient of the interpolated
    position; for features other than the tongue the per-trial median
    frame-to-frame drift is removed and gaps are filled with the nearest
    value.  Returns (speed, visible) for one trial.
    """
    visible = ~np.isnan(x)
    if fill:
        # paws: `findPosition.m` fills missing samples with the nearest value
        xf, yf = fill_nearest(x), fill_nearest(y)
        if np.all(np.isnan(xf)):
            return np.full(NT, np.nan), visible
        vx, vy = np.gradient(xf), np.gradient(yf)
        vx = vx - np.nanmedian(np.diff(xf))
        vy = vy - np.nanmedian(np.diff(yf))
    else:
        # tongue: never filled, so the gradient is taken within each
        # contiguous stretch of frames in which the tongue was visible
        vx = np.full(NT, np.nan)
        vy = np.full(NT, np.nan)
        idx = np.flatnonzero(visible)
        if idx.size:
            splits = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
            for seg in splits:
                if seg.size == 1:
                    vx[seg] = 0.0
                    vy[seg] = 0.0
                else:
                    vx[seg] = np.gradient(x[seg])
                    vy[seg] = np.gradient(y[seg])
    return np.sqrt(vx ** 2 + vy ** 2), visible


def load_motion_energy(anm, date, align_times, frame_times, has_video, ntrials):
    """Port of `DataLoadingScripts/loadMotionEnergy.m`."""
    md = sio.loadmat(os.path.join(DATA_DIR, f'motionEnergy_{anm}_{date}.mat'))
    data = md['me'][0, 0]['data']
    if data.dtype != object:                      # me.data is itself a struct
        data = data[0, 0]['data']
    data = data.flatten()

    me = np.full((ntrials, NT), np.nan)
    for trial in range(ntrials):
        if not has_video[trial]:
            continue
        y = np.asarray(data[trial]).flatten().astype(float)
        t_rel = frame_times[trial]
        if y.size != t_rel.size:
            continue
        me[trial] = fill_nearest(interp_nan(TAXIS, t_rel, y))
    return me


# ---------------------------------------------------------------------------
def discretize(values, visible, threshold):
    """0 = below the session median, 1 = at/above it, 2 = not visible."""
    out = np.full(values.shape, 2, dtype=np.int8)
    ok = visible & ~np.isnan(values)
    out[ok] = (values[ok] >= threshold).astype(np.int8)
    return out


def process_session(anm, date, probes):
    path = os.path.join(DATA_DIR, f'data_structure_{anm}_{date}.mat')
    with h5py.File(path, 'r') as f:
        obj = f['obj']
        beh = load_behavior(obj)
        n = beh['ntrials']
        align = beh['align']

        # --- trial curation -------------------------------------------------
        # every condition in the paper's scripts is restricted to
        # `~stim.enable & ~early`: photoinactivation trials perturb ALM
        # activity, and "early lick" trials are omitted from all analyses.
        # Ignore ("no") trials are *kept*, because "ignore" / "no lick" are
        # required output categories for this decoding task.
        trial_mask = (~beh['early']) & (~beh['stim'])
        keep = np.flatnonzero(trial_mask)

        # --- neural ---------------------------------------------------------
        rates, qualities = load_spikes(f, obj, probes, align, trial_mask)

        # `removeLowFRClusters.m`: drop units below params.lowFR (1 Hz)
        mean_fr = rates.mean(axis=(1, 2)) if rates.size else np.zeros(0)
        use = mean_fr > LOW_FR
        rates = rates[use]
        qualities = [q for q, u in zip(qualities, use) if u]

        # --- video ----------------------------------------------------------
        vidshift = video_shift(f, obj)
        pos, has_video, frame_times = load_video_traces(f, obj, align, n, vidshift)

    me = load_motion_energy(anm, date, align, frame_times, has_video[0], n)

    # tongue speed / visibility
    tongue_speed = np.full((n, NT), np.nan)
    tongue_vis = np.zeros((n, NT), dtype=bool)
    tongue_view = TONGUE_FEATURES[0][0]
    for trial in range(n):
        x, y = pos[TONGUE_FEATURES[0]][0][trial], pos[TONGUE_FEATURES[0]][1][trial]
        s, v = speed_from_position(x, y, fill=False)
        tongue_speed[trial] = s
        tongue_vis[trial] = v & has_video[tongue_view][trial]

    # paw speed / visibility: averaged over the two tracked paws, "not visible"
    # only when neither paw was detected
    paw_speed_all = np.full((len(PAW_FEATURES), n, NT), np.nan)
    paw_vis_all = np.zeros((len(PAW_FEATURES), n, NT), dtype=bool)
    for i, key in enumerate(PAW_FEATURES):
        for trial in range(n):
            s, v = speed_from_position(pos[key][0][trial], pos[key][1][trial], fill=True)
            paw_speed_all[i, trial] = s
            paw_vis_all[i, trial] = v & has_video[key[0]][trial]
    paw_vis = paw_vis_all.any(axis=0)
    masked = np.where(paw_vis_all, paw_speed_all, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        paw_speed = np.nanmean(masked, axis=0)

    me_vis = has_video[0][:, None] & ~np.isnan(me)

    # --- per-session discretisation thresholds (50th percentile) ------------
    def median_of(values, visible):
        vals = values[keep][visible[keep]]
        vals = vals[~np.isnan(vals)]
        return np.median(vals) if vals.size else np.inf

    thr_tongue = median_of(tongue_speed, tongue_vis)
    thr_paw = median_of(paw_speed, paw_vis)
    thr_me = median_of(me, me_vis)

    tongue_cls = discretize(tongue_speed, tongue_vis, thr_tongue)
    paw_cls = discretize(paw_speed, paw_vis, thr_paw)
    me_cls = discretize(me, me_vis, thr_me)

    # --- per-trial task variables ------------------------------------------
    # lick direction: `funcs/getPrevChoice.m` -- the animal licked right on
    # (right trial & hit) or (left trial & miss); "no" trials have no lick.
    right = (beh['R'] & beh['hit']) | (beh['L'] & beh['miss'])
    left = (beh['L'] & beh['hit']) | (beh['R'] & beh['miss'])
    lick_dir = np.full(n, 2, dtype=np.int8)      # 2 = none
    lick_dir[left] = 0
    lick_dir[right] = 1

    # context: `obj.bp.autowater` marks water-cued (WC) trials
    context = np.where(beh['autowater'], 0, 1).astype(np.int8)

    # outcome: hit / miss / ignore (`funcs/getOutcome.m`)
    outcome = np.full(n, 2, dtype=np.int8)       # 2 = ignore
    outcome[beh['miss']] = 0
    outcome[beh['hit']] = 1

    # --- assemble ----------------------------------------------------------
    neural_trials, input_trials, output_trials = [], [], []
    time_input = TAXIS.astype(np.float32).reshape(1, NT)
    for pos_i, trial in enumerate(keep):
        neural_trials.append(np.ascontiguousarray(rates[:, pos_i, :], dtype=np.float32))
        input_trials.append(time_input.copy())
        out = np.empty((6, NT), dtype=np.int8)
        out[0] = lick_dir[trial]
        out[1] = context[trial]
        out[2] = outcome[trial]
        out[3] = tongue_cls[trial]
        out[4] = paw_cls[trial]
        out[5] = me_cls[trial]
        output_trials.append(out)

    info = {
        'animal': anm,
        'date': date,
        'probes': probes,
        'nunits': int(rates.shape[0]),
        'nsingle_units': int(sum(q.lower() in ('excellent', 'great', 'good')
                                 for q in qualities)),
        'ntrials': int(keep.size),
        'ntrials_total': int(n),
        'n_wc_trials': int(beh['autowater'][keep].sum()),
        'n_dr_trials': int((~beh['autowater'][keep]).sum()),
        'n_excluded_early': int(beh['early'].sum()),
        'n_excluded_stim': int((beh['stim'] & ~beh['early']).sum()),
        'n_trials_without_video': int((~has_video[0][keep]).sum()),
        'quality_counts': dict(Counter(q for q in qualities)),
    }
    return neural_trials, input_trials, output_trials, info


# ---------------------------------------------------------------------------
def main():
    data = {
        'neural': [], 'input': [], 'output': [],
        'subjects': [], 'subject_idx': [],
        'brain_regions': ['ALM'], 'brain_region_idx': [],
        'input_names': ['time_from_go_cue'],
        'output_names': ['lick_direction', 'context', 'outcome',
                         'tongue_velocity', 'paw_velocity', 'motion_energy'],
        'output_values': [
            ['left', 'right', 'none'],
            ['WC', 'DR'],
            ['incorrect', 'correct', 'ignore'],
            ['below_median', 'above_median', 'not_visible'],
            ['below_median', 'above_median', 'not_visible'],
            ['below_median', 'above_median', 'no_video'],
        ],
        'metadata': {},
    }

    subjects = []
    session_info = []
    for anm, date, probes in SESSIONS:
        neural, inp, out, info = process_session(anm, date, probes)
        if info['nunits'] < MIN_UNITS:
            print(f"  skipping {anm} {date}: only {info['nunits']} units")
            continue
        if len(neural) < 2:
            print(f"  skipping {anm} {date}: only {len(neural)} trials")
            continue
        if anm not in subjects:
            subjects.append(anm)
        data['neural'].append(neural)
        data['input'].append(inp)
        data['output'].append(out)
        data['subject_idx'].append(subjects.index(anm))
        data['brain_region_idx'].append(np.zeros(info['nunits'], dtype=np.int64))
        session_info.append(info)
        print(f"  {anm} {date}: {info['nunits']} units, {info['ntrials']} trials "
              f"({info['n_wc_trials']} WC)")

    data['subjects'] = subjects
    data['subject_idx'] = np.array(data['subject_idx'], dtype=np.int64)
    data['metadata'] = {
        'task_description':
            'Head-fixed mice performed two directional-licking tasks that '
            'alternated in blocks within a session. In the delayed-response '
            '(DR) task an auditory sample tone (1.3 s) indicated the rewarded '
            'lickport, a 0.9 s delay epoch followed, and an auditory go cue '
            'instructed the mouse to lick; licking the cued port was rewarded '
            'with water. In the water-cued (WC) task all auditory cues were '
            'omitted and ~3 uL of water was delivered at a random time from a '
            'randomly chosen lickport. Decoded variables are the licked '
            'direction, the behavioural context (WC/DR), the trial outcome, '
            'and three discretised movement measures (tongue speed, paw speed '
            'and whole-frame motion energy) from 400 Hz videography.',
        'time_bin_size': DT * 1000.0,
        'temporal_alignment_event':
            'go cue onset (DR trials) / equivalent response-epoch onset in the '
            'silent WC trials (obj.bp.ev.goCue)',
        'off_start': TMIN,
        'off_end': TMAX,
        'recording': 'Neuropixels 1.0 / H2 silicon probes in ALM',
        'species': 'Mus musculus',
        'neural_units': 'spikes/s (spike counts in 10 ms bins, smoothed with a '
                        'causal gaussian kernel of 15 bins, as in getSeq.m)',
        'trial_selection': 'photoinactivation ("stim.enable") and early-lick '
                           'trials excluded, as in the paper; ignore trials kept',
        'unit_selection': 'all sorted units except quality '
                          'garbage/noisy/real?, with mean firing rate > 1 Hz; '
                          'sessions with < 10 units excluded',
        'discretisation': 'tongue/paw speed and motion energy are split at the '
                          '50th percentile of all visible timepoints of that '
                          'session; timepoints where DeepLabCut did not detect '
                          'the feature (or trials without usable video) are '
                          'assigned the third category',
        'source': 'Hasnain, Birnbaum et al., Nature Neuroscience 2024 '
                  '(Zenodo DOI 10.5281/zenodo.13941415)',
        'session_info': session_info,
    }

    ntrials = sum(len(s) for s in data['neural'])
    nunits = sum(i['nunits'] for i in session_info)
    print(f"\n{len(data['neural'])} sessions, {len(subjects)} subjects, "
          f"{nunits} units, {ntrials} trials")

    with open(OUT_FILE, 'wb') as fh:
        pickle.dump(data, fh, protocol=4)
    print(f"wrote {OUT_FILE} "
          f"({os.path.getsize(OUT_FILE) / 1e9:.2f} GB)")


if __name__ == '__main__':
    main()
