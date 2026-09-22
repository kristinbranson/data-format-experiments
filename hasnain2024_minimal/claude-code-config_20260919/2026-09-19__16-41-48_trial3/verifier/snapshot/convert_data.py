"""Convert the Hasnain, Birnbaum et al. (Nat Neurosci 2024) ALM recordings into the
decoder dataset format described in the task instructions.

The processing follows the authors' own MATLAB pipeline (`DataLoadingScripts/`,
`WorkingWithDataObjs.m`, `Scripts/EDFigure 2/EDFigure2a_Left.m`,
`Scripts/Figure 3/Figure3h.m`):

  * sessions / probes: exactly the ones listed in `DataLoadingScripts/Recording and
    video/load<ANM>_ALMVideo.m` (commented-out entries are excluded, as in the paper)
  * alignment: spikes and video are aligned to the go cue (`obj.bp.ev.goCue`), which
    in water-cued (WC) trials is the time of the water drop
  * binning: 10 ms bins from -2.5 s to +2.5 s around the go cue, spike counts
    converted to spikes/s and smoothed with the authors' causal Gaussian kernel
    (`mySmooth`, N = 15 bins, 'reflect' boundary) -- i.e. `params.dt = 1/100`,
    `params.tmin/tmax = -/+2.5`, `params.smooth = 15`, `params.bctype = 'reflect'`
  * units: all sorted clusters except the qualities rejected by `findClusters` with
    `params.quality = {'all'}` (garbage / noisy / 'real?'), then clusters with a mean
    firing rate <= 1 Hz are dropped (`params.lowFR = 1`, `removeLowFRClusters`)
  * sessions with fewer than 10 remaining units are dropped ("Recording sessions were
    included for analysis only if they had at least 10 units", Methods)
  * trials: early-lick trials and photostimulation trials are excluded, as in every
    `params.condition` of the paper ('~early', '~stim.enable'). Hit, miss and ignore
    ('no') trials are all kept, because correct / incorrect / ignore is one of the
    variables to be decoded.

Usage: python convert_data.py [output.pkl]
"""

import os
import pickle
import sys

import numpy as np

from matio import load_obj, loadmat_var

DATA_ROOT = '/app/data'
FIXED_DELAY_DIR = os.path.join(DATA_ROOT, 'Ephys_Behavior')
RANDOM_DELAY_DIR = os.path.join(DATA_ROOT, 'RandomizedDelay_Ephys_Behavior')

# ---------------------------------------------------------------- parameters
# Values taken from the paper's analysis scripts (see module docstring).
TMIN = -2.5          # s, relative to the go cue
TMAX = 2.5           # s
DT = 0.01            # s (params.dt = 1/100)
SMOOTH_N = 15        # bins (params.smooth), causal Gaussian
LOW_FR = 1.0         # Hz (params.lowFR)
MIN_UNITS = 10       # Methods: sessions need at least 10 units
MIN_TRIALS = 2       # the decoder needs at least two trials per session

# Cluster qualities rejected by findClusters(..., {'all'}); unlabelled clusters are
# kept, as they are by that function, and are still subject to the firing-rate cut.
BAD_QUALITY = {'garbage', 'gabrga', 'noisy', 'real?'}

# Sessions and probes, copied from DataLoadingScripts/Recording and video/*.m.
# (anm, date, [1-based probe numbers]).  Entries commented out in those scripts
# (JEB13 2022-09-15, JEB23 2023-10-20, ...) are excluded here as well, as are
# JEB24 2023-10-03/04, which appear in the data folder but in no loading script
# (and have no motion-energy file).
FIXED_DELAY_SESSIONS = [
    ('JEB6',  '2021-04-18', [2]),
    ('JEB7',  '2021-04-29', [1]),
    ('JEB7',  '2021-04-30', [1]),
    ('EKH1',  '2021-08-07', [2]),
    ('EKH3',  '2021-08-11', [2]),
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

RANDOM_DELAY_SESSIONS = [
    ('JEB11', '2022-05-10', [1]),
    ('JEB11', '2022-05-11', [1]),
    ('JEB12', '2022-05-12', [1]),
    ('JEB12', '2022-05-13', [1]),
    ('JEB23', '2023-10-10', [1]),
    ('JEB23', '2023-10-11', [1]),
    ('JEB23', '2023-10-12', [1]),
    ('JEB23', '2023-10-13', [1]),
    ('JEB23', '2023-10-18', [1]),
    ('JEB23', '2023-10-19', [1]),
    ('JEB23', '2023-10-21', [1]),
    ('JEB24', '2023-10-23', [1]),
    ('JEB24', '2023-10-24', [1]),
    ('JEB24', '2023-10-25', [1]),
    ('JEB24', '2023-10-26', [1]),
    ('JEB24', '2023-10-27', [1]),
    ('JEB24', '2023-10-31', [1]),
    ('JEB24', '2023-11-02', [1]),
    ('JEB24', '2023-11-03', [1]),
]

# Video features used for the kinematic outputs.  The tongue is tracked from the
# side camera (view 1) and the two paws only from the bottom camera (view 2), as
# described in the Methods ("the paws were tracked using only the bottom view").
TONGUE_FEATURE = ('tongue', 0)          # (feature name, view index)
PAW_FEATURES = [('top_paw', 1), ('bottom_paw', 1)]

OUTPUT_NAMES = ['lick_direction', 'context', 'outcome',
                'tongue_velocity', 'paw_velocity', 'motion_energy']
OUTPUT_VALUES = [
    ['left', 'right', 'none'],
    ['WC', 'DR'],
    ['incorrect', 'correct', 'ignore'],
    ['below_median', 'above_median', 'not_visible'],
    ['below_median', 'above_median', 'not_visible'],
    ['below_median', 'above_median', 'no_video'],
]
INPUT_NAMES = ['time_from_go_cue']


# ------------------------------------------------------------------ helpers

def gausswin(N, alpha=2.5):
    """MATLAB's gausswin(N, alpha)."""
    n = np.arange(N) - (N - 1) / 2.0
    return np.exp(-0.5 * (alpha * n / ((N - 1) / 2.0)) ** 2)


def my_smooth(x, N=SMOOTH_N):
    """Port of utils/mySmooth.m with bctype = 'reflect' (causal Gaussian kernel).

    x is (time, ...); smoothing operates on the first dimension.
    """
    if N <= 1:
        return x
    kern = gausswin(N)
    kern[:N // 2] = 0.0          # causal half
    kern = kern / kern.sum()
    xp = np.concatenate([x[:N], x], axis=0)
    out = np.apply_along_axis(lambda v: np.convolve(v, kern, mode='same'), 0, xp)
    return out[N:]


def bin_spikes(spike_times, edges):
    """Spike counts per bin (histc semantics, last open edge dropped)."""
    counts, _ = np.histogram(spike_times, bins=edges)
    return counts.astype(float)


def as_bool(x, n):
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.size != n:
        x = np.resize(x, n)
    return np.nan_to_num(x, nan=0.0) > 0.5


def probe_locations(obj, nprobes):
    """Anatomical label of each probe, from obj.ex.probe.loc.

    A few of the older sessions (EKH1, EKH3, JEB7, JGR2, JGR3) have no `ex` field
    at all; they are ALM recordings (they are loaded by the authors' ALM loaders
    and "we recorded activity extracellularly in the ALM", Results), so ALM is the
    fallback label.
    """
    locs = ['ALM'] * nprobes
    ex = obj.get('ex')
    if isinstance(ex, dict) and 'probe' in ex:
        p = ex['probe']
        raw = [q.get('loc') for q in p] if isinstance(p, list) else [p.get('loc')]
        for i, v in enumerate(raw[:nprobes]):
            if isinstance(v, str) and v.strip():
                locs[i] = v.strip()
    return locs


def region_name(loc):
    """Map the recording-location string in the data object to a region name."""
    u = loc.upper()
    if 'ALM' in u:
        return 'ALM'
    if 'M1TJ' in u or 'TJM1' in u:
        return 'tjM1'
    return 'ALM'


def video_shift(obj):
    """funcs/findVideoOffset.m: offset between the ephys and video clocks (s)."""
    def mode_(v):
        v = np.asarray(v, dtype=float).reshape(-1)
        v = v[np.isfinite(v)]
        if v.size == 0:
            return 0.0
        vals, counts = np.unique(v, return_counts=True)
        return float(vals[np.argmax(counts)])
    try:
        return mode_(obj['sglx']['bitcode']['bitstart']) / float(obj['sglx']['fs']) \
            - mode_(obj['bp']['ev']['bitStart'])
    except (KeyError, TypeError):
        return 0.5


def interp_trace(t_src, y_src, t_dst):
    """Linear interpolation with NaN outside the source range (MATLAB interp1)."""
    out = np.full((t_dst.size,) + y_src.shape[1:], np.nan)
    ok = np.isfinite(t_src)
    if ok.sum() < 2:
        return out
    t_src = t_src[ok]
    y_src = y_src[ok]
    order = np.argsort(t_src)
    t_src, y_src = t_src[order], y_src[order]
    inside = (t_dst >= t_src[0]) & (t_dst <= t_src[-1])
    if y_src.ndim == 1:
        out[inside] = np.interp(t_dst[inside], t_src, y_src)
    else:
        for j in range(y_src.shape[1]):
            out[inside, j] = np.interp(t_dst[inside], t_src, y_src[:, j])
    return out


def fill_interior_nans(y):
    """Nearest-neighbour fill of NaNs that lie between valid samples.

    The authors fill every missing sample with `fillmissing(...,'nearest')`; here
    the fill is restricted to gaps inside the covered range so that timepoints the
    camera never saw stay NaN and can be labelled 'not visible' / 'no video'.
    """
    y = y.copy()
    good = np.isfinite(y)
    if not good.any():
        return y
    idx = np.flatnonzero(good)
    lo, hi = idx[0], idx[-1]
    seg = y[lo:hi + 1]
    bad = ~np.isfinite(seg)
    if bad.any():
        pos = np.flatnonzero(~bad)
        nearest = pos[np.argmin(np.abs(np.flatnonzero(bad)[:, None] - pos[None, :]), axis=1)]
        seg[bad] = seg[nearest]
        y[lo:hi + 1] = seg
    return y


def speed(x, y):
    """Speed from a 2-D position trace, NaN-aware (central / one-sided differences)."""
    def deriv(v):
        d = np.full(v.shape, np.nan)
        fwd = np.full(v.shape, np.nan)
        bwd = np.full(v.shape, np.nan)
        fwd[:-1] = v[1:] - v[:-1]
        bwd[1:] = v[1:] - v[:-1]
        both = np.isfinite(fwd) & np.isfinite(bwd)
        d[both] = (fwd[both] + bwd[both]) / 2.0
        only_f = np.isfinite(fwd) & ~np.isfinite(bwd)
        d[only_f] = fwd[only_f]
        only_b = np.isfinite(bwd) & ~np.isfinite(fwd)
        d[only_b] = bwd[only_b]
        return d
    return np.sqrt(deriv(x) ** 2 + deriv(y) ** 2)


def feature_index(traj_view, name):
    for trial in traj_view:
        feats = trial.get('featNames')
        if isinstance(feats, list) and name in feats:
            return feats.index(name)
    return None


def trial_video_times(trial, vidshift, align_time):
    """Frame times of one trial on the bpod clock, relative to the align event."""
    ft = trial.get('frameTimes')
    ts = trial.get('ts')
    if ts is None or np.size(ts) == 0:
        return None, None
    ts = np.asarray(ts, dtype=float)
    if ts.ndim != 3:
        return None, None
    if ft is None or np.size(ft) == 0 or not np.any(np.isfinite(np.asarray(ft, float))):
        # findPosition.m falls back to a nominal 400 Hz frame clock
        ft = (np.arange(ts.shape[0]) + 1) / 400.0
        vidshift = 0.5
    ft = np.asarray(ft, dtype=float).reshape(-1)
    if ft.size != ts.shape[0]:
        n = min(ft.size, ts.shape[0])
        ft, ts = ft[:n], ts[:n]
    nd = trial.get('NdroppedFrames')
    if nd is not None and np.size(nd) == 1 and not np.isfinite(float(np.asarray(nd, float))):
        return None, None      # findPosition.m skips trials with NaN NdroppedFrames
    return ft - vidshift - align_time, ts


# ------------------------------------------------------------ one session

def process_session(anm, date, probes, folder, task):
    path = os.path.join(folder, 'data_structure_%s_%s.mat' % (anm, date))
    obj = load_obj(path)
    bp = obj['bp']
    ntrials_all = int(np.asarray(bp['Ntrials']).reshape(-1)[0])

    hit = as_bool(bp['hit'], ntrials_all)
    miss = as_bool(bp['miss'], ntrials_all)
    no = as_bool(bp['no'], ntrials_all)
    R = as_bool(bp['R'], ntrials_all)
    L = as_bool(bp['L'], ntrials_all)
    early = as_bool(bp['early'], ntrials_all)
    stim = as_bool(bp['stim']['enable'], ntrials_all) if 'stim' in bp else np.zeros(ntrials_all, bool)
    aw = np.asarray(bp['autowater'], dtype=float).reshape(-1)
    if aw.size != ntrials_all:
        aw = np.resize(aw, ntrials_all)
    # older sessions code autowater as 1 = off / 2 = on, newer ones as 0/1
    autowater = (aw == 2) if np.nanmax(aw) > 1 else (np.nan_to_num(aw) > 0.5)
    gocue = np.asarray(bp['ev']['goCue'], dtype=float).reshape(-1)

    # ---- trial selection: no early licks, no photostimulation, valid go cue
    keep = (~early) & (~stim) & np.isfinite(gocue) & (hit | miss | no)
    trials = np.flatnonzero(keep)
    if trials.size < MIN_TRIALS:
        return None

    edges = np.round(np.arange(TMIN, TMAX + DT / 2, DT), 10)
    time = edges[:-1] + DT / 2
    T = time.size

    # ---- neural data ---------------------------------------------------
    clu = obj['clu']
    if isinstance(clu, dict):
        clu = [clu]
    if isinstance(clu, list) and len(clu) and isinstance(clu[0], dict):
        clu = [clu]                      # single probe -> list of one cluster list
    locs = probe_locations(obj, len(clu))

    rates, regions, qualities = [], [], []
    for prb in probes:
        if prb - 1 >= len(clu):
            continue
        region = region_name(locs[prb - 1])
        units = clu[prb - 1]
        if isinstance(units, dict):
            units = [units]
        for unit in units:
            quality = str(unit.get('quality', '')).replace('\x00', '').strip()
            if quality.lower() in BAD_QUALITY:
                continue
            utrial = np.asarray(unit['trial'], dtype=float).reshape(-1)
            utm = np.asarray(unit['trialtm'], dtype=float).reshape(-1)
            dat = np.zeros((T, trials.size))
            if utrial.size:
                order = np.argsort(utrial, kind='stable')
                utrial, utm = utrial[order], utm[order]
                starts = np.searchsorted(utrial, trials + 1, side='left')
                stops = np.searchsorted(utrial, trials + 1, side='right')
                for k in range(trials.size):
                    if stops[k] > starts[k]:
                        aligned = utm[starts[k]:stops[k]] - gocue[trials[k]]
                        dat[:, k] = bin_spikes(aligned, edges)
            dat = my_smooth(dat / DT)
            rates.append(dat.astype(np.float32))
            regions.append(region)
            qualities.append(quality)

    if not rates:
        return None
    rates = np.stack(rates, axis=1)                       # (T, nunits, ntrials)

    # remove low firing rate units (removeLowFRClusters.m, params.lowFR = 1 Hz)
    mean_fr = rates.mean(axis=(0, 2))
    use = mean_fr > LOW_FR
    if use.sum() < MIN_UNITS:
        return None
    rates = rates[:, use, :]
    regions = [regions[i] for i in np.flatnonzero(use)]
    qualities = [qualities[i] for i in np.flatnonzero(use)]

    # In two randomized-delay sessions the sorted spike data stop part way through
    # the behavioral session, leaving late trials without a single spike on any
    # unit. Those trials carry no neural data at all, so they are dropped.
    has_spikes = rates.sum(axis=(0, 1)) > 0
    n_no_spikes = int((~has_spikes).sum())
    if n_no_spikes:
        rates = rates[:, :, has_spikes]
        trials = trials[has_spikes]
        if trials.size < MIN_TRIALS:
            return None

    # ---- video: tongue / paw speed -------------------------------------
    vidshift = video_shift(obj)
    traj = obj.get('traj')
    tongue_speed = np.full((T, trials.size), np.nan)
    paw_speed = np.full((T, trials.size), np.nan)
    if isinstance(traj, list) and len(traj) >= 2:
        views = [traj[0], traj[1]]
        tongue_ix = feature_index(views[TONGUE_FEATURE[1]], TONGUE_FEATURE[0])
        paw_ix = [(feature_index(views[v], nm), v) for nm, v in PAW_FEATURES]
        for k, trix in enumerate(trials):
            # tongue (side view): NaN wherever DeepLabCut did not see the tongue
            if tongue_ix is not None and trix < len(views[TONGUE_FEATURE[1]]):
                tt, ts = trial_video_times(views[TONGUE_FEATURE[1]][trix], vidshift,
                                           gocue[trix])
                if tt is not None:
                    xy = interp_trace(tt, ts[:, 0:2, tongue_ix], time)
                    tongue_speed[:, k] = speed(xy[:, 0], xy[:, 1])
            # paws (bottom view): mean speed of the two tracked paws
            if trix < len(views[1]):
                tt, ts = trial_video_times(views[1][trix], vidshift, gocue[trix])
                if tt is not None:
                    sp = []
                    for ix, _ in paw_ix:
                        if ix is None:
                            continue
                        xy = interp_trace(tt, ts[:, 0:2, ix], time)
                        xy = np.stack([fill_interior_nans(xy[:, 0]),
                                       fill_interior_nans(xy[:, 1])], axis=1)
                        sp.append(speed(xy[:, 0], xy[:, 1]))
                    if sp:
                        paw_speed[:, k] = np.nanmean(np.stack(sp, axis=1), axis=1)

    # ---- motion energy --------------------------------------------------
    me_trace = np.full((T, trials.size), np.nan)
    me_path = os.path.join(folder, 'motionEnergy_%s_%s.mat' % (anm, date))
    me_data = None
    if os.path.exists(me_path):
        me = loadmat_var(me_path, 'me')
        me_data = me.get('data') if isinstance(me, dict) else me
        if isinstance(me_data, dict):
            me_data = me_data.get('data')
    if me_data is not None and isinstance(traj, list) and len(traj) >= 1:
        for k, trix in enumerate(trials):
            if trix >= len(me_data) or trix >= len(traj[0]):
                continue
            y = np.asarray(me_data[trix], dtype=float).reshape(-1)
            if y.size == 0:
                continue
            tt, _ = trial_video_times(traj[0][trix], vidshift, gocue[trix])
            if tt is None:
                continue
            n = min(tt.size, y.size)
            me_trace[:, k] = fill_interior_nans(interp_trace(tt[:n], y[:n], time))

    # ---- outputs ---------------------------------------------------------
    def discretize(x):
        """0 below the session median, 1 at/above it, 2 where the feature is absent."""
        out = np.full(x.shape, 2, dtype=np.int8)
        ok = np.isfinite(x)
        if ok.any():
            thresh = np.percentile(x[ok], 50)
            out[ok] = (x[ok] >= thresh).astype(np.int8)
        else:
            thresh = np.nan
        return out, thresh

    tongue_cls, tongue_thresh = discretize(tongue_speed)
    paw_cls, paw_thresh = discretize(paw_speed)
    me_cls, me_thresh = discretize(me_trace)

    # per-trial variables
    # lick direction follows the paper's choice definition (funcs/getPrevChoice.m):
    # a right lick is (R & hit) | (L & miss); ignore trials have no lick.
    right = (R & hit) | (L & miss)
    left = (L & hit) | (R & miss)
    lick_dir = np.full(ntrials_all, 2, dtype=np.int8)   # 'none'
    lick_dir[left] = 0
    lick_dir[right] = 1
    lick_dir[no] = 2
    context = np.where(autowater, 0, 1).astype(np.int8)  # 0 = WC, 1 = DR
    outcome = np.full(ntrials_all, 2, dtype=np.int8)     # 'ignore'
    outcome[miss] = 0
    outcome[hit] = 1
    outcome[no] = 2

    neural, inputs, outputs = [], [], []
    time_row = time.astype(np.float32)[None, :]
    for k, trix in enumerate(trials):
        neural.append(np.ascontiguousarray(rates[:, :, k].T))
        inputs.append(time_row.copy())
        out = np.empty((6, T), dtype=np.int8)
        out[0, :] = lick_dir[trix]
        out[1, :] = context[trix]
        out[2, :] = outcome[trix]
        out[3, :] = tongue_cls[:, k]
        out[4, :] = paw_cls[:, k]
        out[5, :] = me_cls[:, k]
        outputs.append(out)

    info = {
        'animal': anm,
        'date': date,
        'task': task,
        'probes': list(probes),
        'probe_locations': [locs[p - 1] for p in probes if p - 1 < len(locs)],
        'n_units': int(rates.shape[1]),
        # single units as counted in Scripts/EDFigure 2/EDFigure2a_Left.m
        'n_single_units': int(sum(q.lower() in ('fair', 'good', 'great', 'excellent')
                                  for q in qualities)),
        'n_trials': int(trials.size),
        'n_trials_total': ntrials_all,
        'n_trials_excluded_early': int((early & ~stim).sum()),
        'n_trials_excluded_stim': int(stim.sum()),
        'n_trials_excluded_no_spikes': n_no_spikes,
        'n_WC_trials': int(autowater[trials].sum()),
        'frac_trials_with_video': float(np.mean(np.isfinite(me_trace).any(axis=0))),
        'tongue_velocity_threshold': float(tongue_thresh),
        'paw_velocity_threshold': float(paw_thresh),
        'motion_energy_threshold': float(me_thresh),
    }
    return {
        'neural': neural,
        'input': inputs,
        'output': outputs,
        'regions': regions,
        'info': info,
    }


# ----------------------------------------------------------------- driver

def main(out_path='/app/converted_data.pkl'):
    sessions = ([(a, d, p, FIXED_DELAY_DIR, 'fixed delay (DR/WC two-context)')
                 for a, d, p in FIXED_DELAY_SESSIONS] +
                [(a, d, p, RANDOM_DELAY_DIR, 'randomized delay (DR only)')
                 for a, d, p in RANDOM_DELAY_SESSIONS])

    data = {k: [] for k in ('neural', 'input', 'output', 'brain_region_idx')}
    subjects, brain_regions = [], []
    subject_idx, session_info = [], []

    for anm, date, probes, folder, task in sessions:
        print('processing %s %s ...' % (anm, date), flush=True)
        rez = process_session(anm, date, probes, folder, task)
        if rez is None:
            print('  skipped (too few units or trials)')
            continue
        for region in rez['regions']:
            if region not in brain_regions:
                brain_regions.append(region)
        if anm not in subjects:
            subjects.append(anm)
        data['neural'].append(rez['neural'])
        data['input'].append(rez['input'])
        data['output'].append(rez['output'])
        data['brain_region_idx'].append(
            np.array([brain_regions.index(r) for r in rez['regions']], dtype=np.int64))
        subject_idx.append(subjects.index(anm))
        session_info.append(rez['info'])
        print('  %d units, %d trials (%d WC), %.0f%% of trials with video'
              % (rez['info']['n_units'], rez['info']['n_trials'],
                 rez['info']['n_WC_trials'], 100 * rez['info']['frac_trials_with_video']))

    data['subjects'] = subjects
    data['subject_idx'] = np.array(subject_idx, dtype=np.int64)
    data['brain_regions'] = brain_regions
    data['input_names'] = INPUT_NAMES
    data['output_names'] = OUTPUT_NAMES
    data['output_values'] = OUTPUT_VALUES
    data['metadata'] = {
        'task_description':
            'Head-fixed mice performed directional licking tasks in which an auditory '
            'go cue (delayed-response, DR, context) or an unsignalled water drop '
            '(water-cued, WC, context) instructed a left or right tongue movement. In '
            'DR trials an auditory sample tone (1.3 s) indicated the rewarded side and '
            'was followed by a delay epoch (0.9 s fixed, or randomized in the '
            'randomized-delay sessions) before the go cue. DR and WC trials alternated '
            'in blocks of 10-25 trials within a session in the two-context sessions. '
            'Decoded from ALM (and, in four JEB15 sessions, tjM1) spiking: the '
            'direction of the instructed lick, the behavioral context, the trial '
            'outcome, and three movement variables (tongue speed, paw speed and whole-'
            'frame motion energy), each split at its session median with a separate '
            'class for timepoints at which the feature was not visible.',
        'time_bin_size': DT * 1000.0,
        'temporal_alignment_event':
            'go cue onset (auditory go cue in DR trials, water drop in WC trials; '
            'obj.bp.ev.goCue)',
        'off_start': TMIN,
        'off_end': TMAX,
        'neural_data': (
            'spikes/s in %g ms bins, smoothed with the causal Gaussian kernel used by '
            'the authors (mySmooth, N = %d bins, reflect boundary); all sorted units '
            'except garbage/noisy, with mean firing rate > %g Hz'
            % (DT * 1000, SMOOTH_N, LOW_FR)),
        'trial_selection': (
            'early-lick trials and photostimulation trials excluded; hit, miss and '
            'ignore trials kept'),
        'session_selection': (
            'all ephys sessions listed in the paper\'s DataLoadingScripts/Recording '
            'and video loaders (25 fixed-delay + 19 randomized-delay sessions), '
            'keeping the probe(s) selected there and requiring >= %d units'
            % MIN_UNITS),
        'source': ('Hasnain, Birnbaum et al., "Separating cognitive and motor '
                   'processes in the behaving mouse", Nature Neuroscience 2024'),
        'session_info': session_info,
    }

    with open(out_path, 'wb') as f:
        pickle.dump(data, f, protocol=4)
    ntrials = sum(len(x) for x in data['neural'])
    nunits = sum(len(x) for x in data['brain_region_idx'])
    print('\nwrote %s: %d sessions, %d subjects, %d units, %d trials'
          % (out_path, len(data['neural']), len(subjects), nunits, ntrials))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else '/app/converted_data.pkl')
