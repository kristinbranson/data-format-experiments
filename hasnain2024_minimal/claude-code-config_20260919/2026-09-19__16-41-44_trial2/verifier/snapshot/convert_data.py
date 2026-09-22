"""
Convert the ALM electrophysiology + video dataset of

    Hasnain, Birnbaum et al. (2024) "Separating cognitive and motor processes
    in the behaving mouse", Nature Neuroscience

into the dictionary format expected by `train_decoder.py` / `decoder.py`.

The conversion follows the authors' own MATLAB pipeline (`DataLoadingScripts/`,
`funcs/kinematics/`) as closely as possible:

  * sessions / probes      -> DataLoadingScripts/'Recording and video'/load<ANM>_ALMVideo.m
  * alignment              -> params.alignEvent = 'goCue'  (alignSpikes.m)
  * binning + smoothing    -> getSeq.m  (tmin=-2.5, tmax=2.5, dt=1/100,
                              causal Gaussian kernel of 15 bins, 'reflect' boundary)
  * unit curation          -> findClusters.m ('all' = every quality except
                              garbage/noisy/real?) + removeLowFRClusters.m (>1 Hz)
  * trial curation         -> the `~early & ~stim.enable` masks that every
                              analysis condition in the paper's scripts uses
  * video alignment        -> findVideoOffset.m / findPosition.m / loadMotionEnergy.m
  * kinematics             -> findPosition.m + findVelocity.m
                              (tongue NaNs filled with the session's mean lick-onset
                              position, all other features filled 'nearest')

Outputs written to /app/converted_data.pkl
"""

import os
import pickle

import h5py
import numpy as np
import scipy.io
from scipy.interpolate import interp1d
from scipy.signal import lfilter

# --------------------------------------------------------------------------- #
# Parameters (mirroring params.* in the paper's analysis scripts)
# --------------------------------------------------------------------------- #
DATA_DIR = '/app/data/Ephys_Behavior'
OUT_FILE = '/app/converted_data.pkl'

ALIGN_EVENT = 'goCue'
TMIN = -2.5          # params.tmin
TMAX = 2.5           # params.tmax
DT = 1.0 / 100.0     # params.dt  (10 ms bins)
SMOOTH_N = 15        # params.smooth (bins of the causal Gaussian kernel)
BCTYPE = 'reflect'   # params.bctype
LOW_FR = 1.0         # params.lowFR (Hz)

# cluster qualities that findClusters.m rejects when params.quality = {'all'}
BAD_QUALITY = {'garbage', 'gabrga', 'noisy', 'real?'}

EDGES = np.arange(TMIN, TMAX + DT / 2, DT)          # 501 edges
TAXIS = EDGES[:-1] + DT / 2                          # obj.time, 500 bin centres
NBINS = len(TAXIS)

# Sessions and the probe(s) that recorded ALM, transcribed from
# DataLoadingScripts/'Recording and video'/load<ANM>_ALMVideo.m (1-based probe ids,
# exactly the sessions the paper analyses: 25 sessions / 10 mice / fixed 0.9 s delay).
SESSIONS = [
    ('EKH1',  '2021-08-07', [2]),
    ('EKH3',  '2021-08-11', [2]),
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
    ('JEB6',  '2021-04-18', [2]),
    ('JEB7',  '2021-04-29', [1]),
    ('JEB7',  '2021-04-30', [1]),
    ('JGR2',  '2021-11-16', [1]),
    ('JGR2',  '2021-11-17', [1]),
    ('JGR3',  '2021-11-18', [1]),
]

# DeepLabCut features used for the two kinematic outputs.
# view 1 = side camera, view 2 = bottom camera (only the bottom camera sees the paws).
TONGUE_FEATURE = ('tongue', 1)
PAW_FEATURES = [('top_paw', 2), ('bottom_paw', 2)]


# --------------------------------------------------------------------------- #
# Small helpers for reading MATLAB v7.3 (HDF5) files
# --------------------------------------------------------------------------- #
def mat_str(f, ref):
    """Decode a MATLAB char array (possibly behind an object reference)."""
    arr = np.array(f[ref]) if isinstance(ref, h5py.Reference) else np.array(ref)
    if arr.dtype not in (np.uint16, np.uint8):
        return ''
    return ''.join(chr(c) for c in arr.flatten()).strip()


def vec(group, key):
    return np.array(group[key]).flatten()


def mode_(x):
    """MATLAB mode() for a numeric vector (smallest most-frequent value)."""
    x = x[~np.isnan(x)]
    vals, counts = np.unique(x, return_counts=True)
    return vals[np.argmax(counts)]


# --------------------------------------------------------------------------- #
# Signal processing helpers (ports of utils/mySmooth.m)
# --------------------------------------------------------------------------- #
def _causal_gauss_kernel(n=SMOOTH_N):
    """gausswin(n) with the acausal half zeroed and renormalised (mySmooth.m)."""
    k = np.arange(n)
    alpha = 2.5
    w = np.exp(-0.5 * (alpha * (2 * k - (n - 1)) / (n - 1)) ** 2)
    w[: n // 2] = 0.0          # kern(1:floor(numel(kern)/2)) = 0  -> causal
    return w / w.sum()


_KERN = _causal_gauss_kernel()
# conv(x, kern, 'same') with a causal kernel is the FIR filter with these taps
_TAPS = _KERN[SMOOTH_N // 2:]


def my_smooth(x):
    """mySmooth(x, 15, 'reflect') applied along axis 0 of a 2-D array."""
    if BCTYPE == 'reflect':
        pad = x[:SMOOTH_N]
        xf = np.concatenate([pad, x], axis=0)
        return lfilter(_TAPS, [1.0], xf, axis=0)[SMOOTH_N:]
    return lfilter(_TAPS, [1.0], x, axis=0)


def interp_to_taxis(t_src, y_src, taxis):
    """MATLAB-style interp1 (linear, NaN outside the support, NaN-propagating)."""
    good = np.isfinite(t_src)
    if good.sum() < 2:
        return np.full((len(taxis),) + y_src.shape[1:], np.nan)
    fn = interp1d(t_src[good], y_src[good], axis=0, kind='linear',
                  bounds_error=False, fill_value=np.nan, assume_sorted=False)
    return fn(taxis)


def fill_nearest(y):
    """fillmissing(y, 'nearest') along axis 0 (column-wise)."""
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(y)
    if not ok.any():
        return y
    idx = np.where(ok)[0]
    out = y.copy()
    missing = np.where(~ok)[0]
    if missing.size:
        nearest = idx[np.argmin(np.abs(missing[:, None] - idx[None, :]), axis=1)]
        out[missing] = y[nearest]
    return out


def runs_of_true(mask):
    """Start indices of each run of True values (utils/ZeroOnesCount.m)."""
    m = np.concatenate([[False], mask, [False]]).astype(np.int8)
    return np.where(np.diff(m) == 1)[0]


# --------------------------------------------------------------------------- #
# Loading one session
# --------------------------------------------------------------------------- #
def clu_groups(f, obj):
    """Return a list (indexed by probe-1) of the per-probe cluster structs."""
    clu = obj['clu']
    if isinstance(clu, h5py.Group):          # single probe stored as a plain struct
        return [clu]
    out = []
    for i in range(clu.shape[0]):
        g = f[clu[i, 0]]
        out.append(g if isinstance(g, h5py.Group) else None)
    return out


def video_offset(f, obj):
    """findVideoOffset.m: offset between the ephys and the video clocks (s)."""
    try:
        bitstart = vec(obj['sglx']['bitcode'], 'bitstart')
        fs = vec(obj['sglx'], 'fs')[0]
        return mode_(bitstart) / fs - mode_(vec(obj['bp']['ev'], 'bitStart'))
    except Exception:
        return 0.5          # fallback used by the paper's code when bitcode is absent


def load_behavior(f, obj):
    bp = obj['bp']
    ev = bp['ev']
    n = int(vec(bp, 'Ntrials')[0])
    b = {
        'n': n,
        'R': np.nan_to_num(vec(bp, 'R')[:n]) > 0,
        'L': np.nan_to_num(vec(bp, 'L')[:n]) > 0,
        'hit': np.nan_to_num(vec(bp, 'hit')[:n]) > 0,
        'miss': np.nan_to_num(vec(bp, 'miss')[:n]) > 0,
        'no': np.nan_to_num(vec(bp, 'no')[:n]) > 0,
        'early': np.nan_to_num(vec(bp, 'early')[:n]) > 0,
        'autowater': np.nan_to_num(vec(bp, 'autowater')[:n]) > 0,
        'stim': np.nan_to_num(vec(bp['stim'], 'enable')[:n]) > 0,
        'goCue': vec(ev, ALIGN_EVENT)[:n],
        'sample': vec(ev, 'sample')[:n],
        'delay': vec(ev, 'delay')[:n],
    }
    return b


def neural_matrix(f, obj, probes, trials, align):
    """
    Binned, smoothed single-trial firing rates (getSeq.m) for the curated units.

    Returns (ntrials, nbins, nunits) float32 and the number of units before the
    firing-rate criterion.
    """
    groups = clu_groups(f, obj)
    keep_rates, keep_ids = [], []
    nquality = 0

    # map 1-based MATLAB trial ids onto rows of the output matrix (-1 = trial dropped)
    trial_pos = -np.ones(int(len(align)) + 2, dtype=np.int64)
    trial_pos[trials] = np.arange(len(trials))

    for p in probes:
        g = groups[p - 1]
        if g is None:
            continue
        qualities = [mat_str(f, r) for r in np.array(g['quality']).flatten()]
        for icell, q in enumerate(qualities):
            if q.lower() in BAD_QUALITY:
                continue
            nquality += 1
            tm = np.array(f[g['trialtm'][icell, 0]]).flatten()
            tr = np.array(f[g['trial'][icell, 0]]).flatten().astype(np.int64)
            if tm.size == 0:
                continue
            inrange = (tr >= 1) & (tr <= len(align))
            tm, tr = tm[inrange], tr[inrange]
            row = trial_pos[tr]
            aligned = tm - align[tr - 1]
            ok = (row >= 0) & (aligned >= TMIN) & (aligned < TMAX)
            row = row[ok]
            b = ((aligned[ok] - TMIN) / DT).astype(np.int64)
            counts = np.bincount(row * NBINS + b,
                                 minlength=len(trials) * NBINS).reshape(len(trials), NBINS)
            # condition-1 PSTH ("all trials") used by removeLowFRClusters.m
            psth = my_smooth((counts.sum(axis=0) / len(trials) / DT)[:, None])[:, 0]
            if psth.mean() > LOW_FR:
                keep_rates.append(counts)
                keep_ids.append((p, icell))

    if not keep_rates:
        return None, nquality, keep_ids

    counts = np.stack(keep_rates, axis=-1)                     # (ntrials, nbins, nunits)
    rates = counts.astype(np.float64) / DT
    flat = rates.transpose(1, 0, 2).reshape(NBINS, -1)         # time first for smoothing
    flat = my_smooth(flat)
    rates = flat.reshape(NBINS, len(trials), -1).transpose(1, 0, 2)
    return rates.astype(np.float32), nquality, keep_ids


def load_traj(f, obj, trials, align, vidshift):
    """
    Interpolate the DLC features needed for the kinematic outputs onto TAXIS.

    Returns dicts keyed by feature name holding (ntrials, nbins) arrays of
    x/y position, plus a per-trial flag saying whether the trial has usable video.
    """
    traj = obj['traj']
    views = [f[traj[0, 0]], f[traj[1, 0]]]
    featnames = [[mat_str(f, x) for x in np.array(f[v['featNames'][0, 0]]).flatten()]
                 for v in views]

    wanted = [TONGUE_FEATURE] + PAW_FEATURES
    pos = {name: np.full((len(trials), NBINS, 2), np.nan) for name, _ in wanted}
    has_video = np.zeros(len(trials), dtype=bool)

    for i, trial in enumerate(trials):
        j = trial - 1
        v0 = views[0]
        try:
            ft = np.array(f[v0['frameTimes'][j, 0]]).flatten()
        except Exception:
            ft = np.array([])
        if ft.size < 10 or not np.isfinite(ft).any():
            continue
        # findPosition.m: trials flagged with NdroppedFrames = NaN are skipped
        try:
            nd = np.array(f[v0['NdroppedFrames'][j, 0]]).flatten()
            if nd.size and np.isnan(nd[0]):
                continue
        except Exception:
            pass

        t_src = ft - vidshift - align[j]
        has_video[i] = True
        for name, view in wanted:
            v = views[view - 1]
            k = featnames[view - 1].index(name)
            ts = np.array(f[v['ts'][j, 0]])          # (nfeat, 3, nframes)
            xy = ts[k, :2, :].T                      # (nframes, 2)
            if xy.shape[0] != t_src.shape[0]:
                m = min(xy.shape[0], t_src.shape[0])
                pos[name][i] = interp_to_taxis(t_src[:m], xy[:m], TAXIS)
            else:
                pos[name][i] = interp_to_taxis(t_src, xy, TAXIS)

    return pos, has_video


def load_motion_energy(anm, date, f, obj, trials, align, vidshift):
    """loadMotionEnergy.m: interpolate motion energy onto TAXIS."""
    fn = os.path.join(DATA_DIR, f'motionEnergy_{anm}_{date}.mat')
    if not os.path.exists(fn):
        return np.full((len(trials), NBINS), np.nan)
    m = scipy.io.loadmat(fn, struct_as_record=False, squeeze_me=False)
    me = m['me'][0, 0]
    data = me.data
    # loadMotionEnergy.m: some files wrap the cell array in another struct
    while hasattr(data, '_fieldnames') or (data.dtype == object and data.size == 1
                                           and hasattr(data.flatten()[0], '_fieldnames')):
        data = data.data if hasattr(data, '_fieldnames') else data.flatten()[0].data
    v0 = f[obj['traj'][0, 0]]

    out = np.full((len(trials), NBINS), np.nan)
    for i, trial in enumerate(trials):
        j = trial - 1
        if j >= data.shape[0]:
            continue
        y = np.asarray(data[j, 0]).flatten().astype(float)
        if y.size == 0:
            continue
        try:
            ft = np.array(f[v0['frameTimes'][j, 0]]).flatten()
        except Exception:
            ft = np.array([])
        if ft.size != y.size or not np.isfinite(ft).any():
            # loadMotionEnergy.m's catch branch: assume 400 Hz frames and a 0.5 s offset
            t_src = np.arange(1, y.size + 1) / 400.0 - 0.5 - align[j]
        else:
            t_src = ft - vidshift - align[j]
        out[i] = interp_to_taxis(t_src, y, TAXIS)
        out[i] = fill_nearest(out[i])
    return out


# --------------------------------------------------------------------------- #
# Kinematic outputs
# --------------------------------------------------------------------------- #
def tongue_speed(pos):
    """
    Speed of the side-view tongue marker.

    NaNs (tongue not visible) are replaced by the session's mean lick-onset
    position, exactly as setTongueBaselinePosition() in getKinematicsFromVideo.m,
    before differentiating.  Returns (speed, visible_mask).
    """
    visible = np.isfinite(pos[..., 0]) | np.isfinite(pos[..., 1])
    starts_x, starts_y = [], []
    for i in range(pos.shape[0]):
        for s in runs_of_true(visible[i]):
            starts_x.append(pos[i, s, 0])
            starts_y.append(pos[i, s, 1])
    mux = np.nanmean(starts_x) if starts_x else 0.0
    muy = np.nanmean(starts_y) if starts_y else 0.0

    filled = pos.copy()
    filled[..., 0] = np.where(np.isfinite(filled[..., 0]), filled[..., 0], mux)
    filled[..., 1] = np.where(np.isfinite(filled[..., 1]), filled[..., 1], muy)

    vx = np.gradient(filled[..., 0], axis=1)
    vy = np.gradient(filled[..., 1], axis=1)
    return np.sqrt(vx ** 2 + vy ** 2), visible


def paw_speed(pos_list):
    """
    Mean speed of the two bottom-view paw markers.

    Positions are filled 'nearest' within each trial and the per-trial median
    frame-to-frame displacement is removed, following findPosition.m /
    findVelocity.m.  Returns (speed, visible_mask).
    """
    speeds, visibles = [], []
    for pos in pos_list:
        visible = np.isfinite(pos[..., 0]) | np.isfinite(pos[..., 1])
        sp = np.zeros(pos.shape[:2])
        for i in range(pos.shape[0]):
            if not visible[i].any():
                continue
            x = fill_nearest(pos[i, :, 0])
            y = fill_nearest(pos[i, :, 1])
            vx = np.gradient(x) - np.nanmedian(np.diff(x))
            vy = np.gradient(y) - np.nanmedian(np.diff(y))
            sp[i] = np.sqrt(vx ** 2 + vy ** 2)
        speeds.append(sp)
        visibles.append(visible)
    speeds = np.stack(speeds)
    visibles = np.stack(visibles)
    any_visible = visibles.any(axis=0)
    with np.errstate(invalid='ignore'):
        mean_speed = np.nansum(np.where(visibles, speeds, np.nan), axis=0) / \
                     np.maximum(visibles.sum(axis=0), 1)
    return mean_speed, any_visible


def discretize(values, valid, missing_code=2):
    """
    0 : below the session's 50th percentile, 1 : at or above it, 2 : not measurable.
    The percentile is taken over every valid sample in the session.
    """
    out = np.full(values.shape, missing_code, dtype=np.int8)
    v = values[valid]
    v = v[np.isfinite(v)]
    if v.size == 0:
        return out
    thresh = np.percentile(v, 50)
    finite = valid & np.isfinite(values)
    out[finite & (values < thresh)] = 0
    out[finite & (values >= thresh)] = 1
    return out


# --------------------------------------------------------------------------- #
# Main conversion
# --------------------------------------------------------------------------- #
def convert_session(anm, date, probes, verbose=True):
    path = os.path.join(DATA_DIR, f'data_structure_{anm}_{date}.mat')
    f = h5py.File(path, 'r')
    obj = f['obj']
    b = load_behavior(f, obj)

    # ---- trial curation -------------------------------------------------- #
    # `early` (lickport contact before the response epoch) and photostimulation
    # trials are excluded from every analysis in the paper.  hit / miss / no
    # (ignore) trials are all kept because outcome and lick direction are decoded.
    keep = (~b['early']) & (~b['stim']) & np.isfinite(b['goCue'])
    trials = np.where(keep)[0] + 1                  # 1-based MATLAB trial ids
    if len(trials) < 2:
        f.close()
        return None

    align = b['goCue']

    # ---- neural ---------------------------------------------------------- #
    rates, nquality, keep_ids = neural_matrix(f, obj, probes, trials, align)
    if rates is None or rates.shape[2] == 0:
        f.close()
        return None

    # ---- video ----------------------------------------------------------- #
    vidshift = video_offset(f, obj)
    pos, has_video = load_traj(f, obj, trials, align, vidshift)
    me = load_motion_energy(anm, date, f, obj, trials, align, vidshift)
    f.close()

    tspeed, tvis = tongue_speed(pos[TONGUE_FEATURE[0]])
    pspeed, pvis = paw_speed([pos[n] for n, _ in PAW_FEATURES])
    tvis &= has_video[:, None]
    pvis &= has_video[:, None]

    tongue_code = discretize(tspeed, tvis)
    paw_code = discretize(pspeed, pvis)
    me_code = discretize(me, np.isfinite(me))

    # ---- per-trial task variables ---------------------------------------- #
    idx = trials - 1
    # lick direction: R&hit or L&miss -> right, L&hit or R&miss -> left, ignore -> none
    lick = np.full(len(idx), 2, dtype=np.int8)                     # 2 = none
    licked_right = (b['R'][idx] & b['hit'][idx]) | (b['L'][idx] & b['miss'][idx])
    licked_left = (b['L'][idx] & b['hit'][idx]) | (b['R'][idx] & b['miss'][idx])
    lick[licked_left] = 0
    lick[licked_right] = 1
    context = np.where(b['autowater'][idx], 0, 1).astype(np.int8)  # 0 = WC, 1 = DR
    outcome = np.full(len(idx), 2, dtype=np.int8)                  # 2 = ignore
    outcome[b['miss'][idx]] = 0                                    # 0 = incorrect
    outcome[b['hit'][idx]] = 1                                     # 1 = correct

    # ---- assemble -------------------------------------------------------- #
    neural, inputs, outputs = [], [], []
    time_input = TAXIS.astype(np.float32)[None, :]
    for i in range(len(trials)):
        neural.append(np.ascontiguousarray(rates[i].T))            # (nunits, nbins)
        inputs.append(time_input.copy())
        out = np.empty((6, NBINS), dtype=np.int8)
        out[0] = lick[i]
        out[1] = context[i]
        out[2] = outcome[i]
        out[3] = tongue_code[i]
        out[4] = paw_code[i]
        out[5] = me_code[i]
        outputs.append(out)

    if verbose:
        print(f'{anm}_{date}: {len(trials)}/{b["n"]} trials, '
              f'{rates.shape[2]}/{nquality} units, '
              f'{int(has_video.sum())} trials with video, '
              f'{int(np.isfinite(me).any(axis=1).sum())} with motion energy, '
              f'{int((context == 0).sum())} WC trials', flush=True)

    return {
        'neural': neural,
        'input': inputs,
        'output': outputs,
        'nunits': rates.shape[2],
        'session_info': {
            'subject': anm,
            'date': date,
            'probes': probes,
            'ntrials': int(len(trials)),
            'ntrials_total': int(b['n']),
            'nunits': int(rates.shape[2]),
            'nunits_before_fr_criterion': int(nquality),
            'nWC_trials': int((context == 0).sum()),
            'nDR_trials': int((context == 1).sum()),
            'two_context_session': bool((context == 0).sum() > 0),
            'video_offset_s': float(vidshift),
        },
    }


def main():
    sessions = []
    for anm, date, probes in SESSIONS:
        s = convert_session(anm, date, probes)
        if s is None:
            print(f'skipping {anm}_{date}')
            continue
        sessions.append(s)

    subjects = sorted({s['session_info']['subject'] for s in sessions})
    subject_idx = np.array([subjects.index(s['session_info']['subject'])
                            for s in sessions], dtype=np.int64)

    data = {
        'neural': [s['neural'] for s in sessions],
        'input': [s['input'] for s in sessions],
        'output': [s['output'] for s in sessions],
        'subjects': subjects,
        'subject_idx': subject_idx,
        # every analysed probe is the ALM probe designated by the paper's
        # load<ANM>_ALMVideo.m scripts
        'brain_regions': ['ALM'],
        'brain_region_idx': [np.zeros(s['nunits'], dtype=np.int64) for s in sessions],
        'input_names': ['time_from_go_cue'],
        'output_names': ['lick_direction', 'context', 'outcome',
                         'tongue_velocity', 'paw_velocity', 'motion_energy'],
        'output_values': [
            ['left', 'right', 'none'],
            ['WC', 'DR'],
            ['incorrect', 'correct', 'ignore'],
            ['below_50th_pctile', 'at_or_above_50th_pctile', 'not_visible'],
            ['below_50th_pctile', 'at_or_above_50th_pctile', 'not_visible'],
            ['below_50th_pctile', 'at_or_above_50th_pctile', 'no_video'],
        ],
        'metadata': {
            'task_description': (
                'Head-fixed mice performed two directional-licking tasks that '
                'alternated in blocks within a session (Hasnain, Birnbaum et al., '
                'Nat Neurosci 2024). In the delayed-response (DR) context an auditory '
                'sample tone (1.3 s) cued the rewarded lickport, a 0.9 s delay followed, '
                'and an auditory go cue released the response; licking the correct port '
                'was rewarded with ~3 ul water. In the water-cued (WC, "autowater") '
                'context all auditory cues were omitted and ~3 ul of water was presented '
                'at a random time at a randomly chosen port. Mice received no cue '
                'signalling the current context. Neural activity was recorded '
                'extracellularly in ALM with H2 or Neuropixels 1.0 probes. '
                'The decoder receives ALM population activity plus the time from the go '
                'cue, and predicts the trial-level task variables (lick direction, '
                'behavioural context, outcome) and the time-varying, per-session-median '
                'thresholded movement variables (tongue speed, paw speed, motion energy).'
            ),
            'time_bin_size': DT * 1000.0,
            'temporal_alignment_event': (
                'go cue onset (DR trials); on WC trials the same event field marks the '
                'water-drop presentation, which is the analogous movement-releasing event'
            ),
            'off_start': TMIN,
            'off_end': TMAX,
            'neural_units': 'spikes/s (spike counts in 10 ms bins, smoothed with a '
                            'causal Gaussian kernel of 15 bins as in getSeq.m/mySmooth.m)',
            'epoch_times_relative_to_alignment': {
                'sample_onset': -2.2, 'delay_onset': -0.9, 'go_cue': 0.0,
            },
            'trial_curation': ('early-lick trials and optogenetic photostimulation '
                               'trials excluded; hit, miss and ignore trials retained'),
            'neuron_curation': ("cluster qualities other than garbage/noisy/real? "
                                "(findClusters.m with params.quality = 'all'), then "
                                'units with mean firing rate <= 1 Hz removed '
                                '(removeLowFRClusters.m with params.lowFR = 1)'),
            'source': 'Hasnain, Birnbaum et al., Nature Neuroscience 2024; '
                      'Zenodo DOI 10.5281/zenodo.13941415 (Ephys_Behavior)',
            'session_info': [s['session_info'] for s in sessions],
        },
    }

    ntrials = sum(len(x) for x in data['neural'])
    nunits = sum(s['nunits'] for s in sessions)
    print(f'\n{len(sessions)} sessions, {len(subjects)} subjects, '
          f'{ntrials} trials, {nunits} units')

    with open(OUT_FILE, 'wb') as fh:
        pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'wrote {OUT_FILE} ({os.path.getsize(OUT_FILE) / 1e9:.2f} GB)')


if __name__ == '__main__':
    main()
