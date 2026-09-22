#!/usr/bin/env python3
"""
Convert the Mesoscale Activity Map dataset (DANDI:000363) from NWB into the
decoder-training pickle format.

Usage
-----
    python -u convert_data.py <outpicklefile> [--full | --sample] [--show-processing]

Options
-------
    --full              process all sessions (default)
    --sample            process only 2 sessions (for testing)
    --show-processing   save per-step diagnostic plots for up to 2 sessions as
                        processing_<session_id>.png
    --jobs N            number of worker processes (default: min(32, cpu_count))

Design notes (see CONVERSION_NOTES.md for full justification)
-------------------------------------------------------------
* Alignment event    : go cue onset (`BehavioralEvents/go_start_times`), matching
                       the reference preprocessing.
* Window             : [-2.5, +1.5) s, 80 non-overlapping 50 ms bins.
* Neural             : firing rate in Hz (spike count / bin width), as in the
                       reference `sliding_histogram(..., rate=True)`.
* Neuron curation    : `units/classification == 'good'` (the QC-classifier "good
                       units"), a mappable CCF annotation, and non-zero activity.
* Session curation   : data-paper criteria -- control-trial performance > 65 % and
                       >= 50 correct lick-left and >= 50 correct lick-right trials.
* Trial curation     : auto-water and free-water trials dropped.  Photostim,
                       early-lick and no-response trials are *kept* because they
                       are required decoder inputs/outputs.
"""

import argparse
import glob
import json
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import h5py

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from region_map import annotation_to_region  # noqa: E402

# ----------------------------------------------------------------------------- config
DATA_DIR = '/app/data'

BIN_WIDTH = 0.05          # s, decoder spec: 50 ms bins
T_START = -2.5            # s relative to go cue, decoder spec
T_STOP = 1.5              # s relative to go cue, decoder spec
N_BINS = int(round((T_STOP - T_START) / BIN_WIDTH))   # 80

TONGUE_LIKELIHOOD_THRESH = 0.5   # DLC likelihood is bimodal (~1e-5 vs ~1.0)
TONGUE_LOW_PCT = 40.0
TONGUE_HIGH_PCT = 60.0

# Session selection (datapaper STAR Methods)
MIN_PERFORMANCE = 0.65
MIN_CORRECT_PER_DIRECTION = 50

# Video curation.  A trial needs video over its whole [-2.5, +1.5] s window for
# the tongue output to be meaningful; the reference code applies the analogous
# check in Sherlock/align_markers.get_bad_trial_inds.  The expected number of
# 300 Hz frames in the window is 4.0 / 0.0034 = 1176.
VIDEO_DT = 0.0034
MIN_VIDEO_COVERAGE = 0.9
# The tongue is only outside the mouth while licking (median 11 % of a session,
# 99th percentile 25 %).  A session whose DeepLabCut tongue likelihood is high
# almost everywhere is a tracking failure, not a licking mouse.
MAX_TONGUE_VISIBLE_FRACTION = 0.5

# hemisphere split on the CCF ML axis, as in the reference code
# (helper_get_neuron_id_area: `left` is ccf_x >= 5700)
ML_MIDLINE = 5700.0

INPUT_NAMES = ['time_from_tone_onset_s', 'photostim_on']
OUTPUT_NAMES = ['choice', 'outcome', 'early_lick', 'tongue_y_position']
OUTPUT_VALUES = [
    ['left', 'right', 'no lick'],
    ['ignore', 'miss', 'hit'],
    ['no', 'yes'],
    ['<40th pct', '40th-60th pct', '>60th pct', 'not visible'],
]

BIN_EDGES = T_START + BIN_WIDTH * np.arange(N_BINS + 1)   # (81,)
BIN_CENTERS = BIN_EDGES[:-1] + BIN_WIDTH / 2             # (80,)


# ------------------------------------------------------------------- small helpers
def _decode(arr):
    """bytes ndarray -> list of str."""
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in arr])


def session_id_from_path(path):
    base = os.path.basename(path)
    sub = base.split('_')[0].replace('sub-', '')
    ses = base.split('_')[1].replace('ses-', '')
    return f'{sub}_{ses}'


# ----------------------------------------------------------------- session loading
def read_session(path):
    """Read everything we need from one NWB file into plain numpy arrays."""
    with h5py.File(path, 'r') as f:
        trials = f['intervals/trials']
        ev = f['acquisition/BehavioralEvents']

        d = {}
        d['path'] = path
        d['session_id'] = session_id_from_path(path)
        d['subject'] = f['general/subject/subject_id'][()].decode()

        u = f['units']

        trial_start_all = trials['start_time'][:]

        # ------------------------------------------------------------------
        # Electrophysiology does not always cover the whole behavioural
        # session: in 9 of the 174 files the recording stops early and the
        # units' `obs_intervals` (and `is_good_trials`) cover only a prefix of
        # the trials table.  Spikes simply do not exist for the remaining
        # trials, so they must be dropped.  `obs_intervals` rows are exactly
        # the [start_time, stop_time] of the covered trials.
        # ------------------------------------------------------------------
        oi_index = u['obs_intervals_index'][:]
        n_int = np.diff(np.concatenate([[0], oi_index]))
        assert n_int.min() == n_int.max(), 'units disagree on observed trials'
        obs = u['obs_intervals'][0:oi_index[0]]
        ephys_trials = np.searchsorted(trial_start_all, obs[:, 0])
        assert np.allclose(trial_start_all[ephys_trials], obs[:, 0]), \
            'obs_intervals do not line up with the trials table'
        assert u['is_good_trials'].shape[1] == len(ephys_trials)
        d['ephys_trials'] = ephys_trials
        d['n_trials_behavior'] = len(trial_start_all)

        et = ephys_trials
        d['trial_start'] = trial_start_all[et]
        d['trial_stop'] = trials['stop_time'][:][et]
        d['outcome'] = _decode(trials['outcome'][:])[et]
        d['instruction'] = _decode(trials['trial_instruction'][:])[et]
        d['early_lick'] = _decode(trials['early_lick'][:])[et]
        d['auto_water'] = trials['auto_water'][:].astype(bool)[et]
        d['free_water'] = trials['free_water'][:].astype(bool)[et]
        d['photostim_power'] = _decode(trials['photostim_power'][:])[et]
        d['task'] = _decode(trials['task'][:])[et]

        d['go_time'] = ev['go_start_times']['timestamps'][:][et]
        d['sample_start'] = ev['sample_start_times']['timestamps'][:]
        d['stim_on'] = ev['photostim_start_times']['timestamps'][:] \
            if 'photostim_start_times' in ev else np.zeros(0)
        d['stim_off'] = ev['photostim_stop_times']['timestamps'][:] \
            if 'photostim_stop_times' in ev else np.zeros(0)
        d['lick_left'] = ev['left_lick_times']['timestamps'][:] \
            if 'left_lick_times' in ev else np.zeros(0)
        d['lick_right'] = ev['right_lick_times']['timestamps'][:] \
            if 'right_lick_times' in ev else np.zeros(0)

        # --- video (side-view tongue marker) ---
        bts = f['acquisition/BehavioralTimeSeries']
        tk = bts['Camera0_side_TongueTracking']
        tongue = tk['data'][:]                    # (n_frames, 3): x, y, likelihood
        d['tongue_t'] = tk['timestamps'][:]
        d['tongue_y'] = tongue[:, 1]
        d['tongue_lik'] = tongue[:, 2]

        # --- units ---
        classification = _decode(u['classification'][:])
        anno = _decode(u['anno_name'][:])
        good = classification == 'good'

        elec = u['electrodes'][:]
        etab = f['general/extracellular_ephys/electrodes']
        eid = etab['id'][:]
        # `units/electrodes` stores row indices of the electrodes table; ids are
        # 0..n-1 in every file of this dandiset, assert rather than assume.
        assert np.array_equal(eid, np.arange(len(eid))), 'unexpected electrode ids'
        ccf_x = etab['x'][:][elec]

        # hemisphere fall-back: targeted ML sign of the probe (rarely needed)
        gnames = _decode(etab['group_name'][:])[elec]
        probe_ml = {}
        for gname, grp in f['general/extracellular_ephys'].items():
            if gname == 'electrodes':
                continue
            try:
                probe_ml[gname] = float(json.loads(grp.attrs['location'])['ml_location'])
            except Exception:
                probe_ml[gname] = np.nan

        regions = np.array([annotation_to_region(a) or '' for a in anno])
        keep_unit = good & (regions != '')

        idx = np.where(keep_unit)[0]
        hemi = np.empty(len(idx), dtype=object)
        for j, i in enumerate(idx):
            x = ccf_x[i]
            if np.isfinite(x):
                hemi[j] = 'left' if x >= ML_MIDLINE else 'right'
            else:
                ml = probe_ml.get(gnames[i], np.nan)
                hemi[j] = 'left' if (np.isfinite(ml) and ml < 0) else 'right'
        d['unit_region'] = np.array([f'{h} {r}' for h, r in zip(hemi, regions[idx])])
        d['unit_anno'] = anno[idx]

        # spike times of the kept units only
        st_index = u['spike_times_index'][:]
        starts = np.concatenate([[0], st_index[:-1]])
        spike_times = u['spike_times'][:]
        unit_spikes = [spike_times[starts[i]:st_index[i]] for i in idx]
        d['unit_spikes'] = unit_spikes
        d['n_all_clusters'] = len(classification)
        d['n_good_clusters'] = int(good.sum())
    return d


# --------------------------------------------------------------- session selection
def session_performance(d, sel=slice(None)):
    """Correct rate on control trials excluding early licks (datapaper definition)."""
    ps = d['photostim_power'][sel] != 'N/A'
    ctrl = (~ps) & (d['early_lick'][sel] == 'no early')
    hit = d['outcome'][sel] == 'hit'
    miss = d['outcome'][sel] == 'miss'
    m = ctrl & (hit | miss)
    perf = hit[m].sum() / m.sum() if m.sum() > 0 else 0.0
    n_left = int((hit & ctrl & (d['instruction'][sel] == 'left')).sum())
    n_right = int((hit & ctrl & (d['instruction'][sel] == 'right')).sum())
    return float(perf), n_left, n_right


def video_coverage(d, win_start):
    """Fraction of the [-2.5, +1.5] s window covered by video frames, per trial."""
    ts = d['tongue_t']
    lo = np.searchsorted(ts, win_start)
    hi = np.searchsorted(ts, win_start + N_BINS * BIN_WIDTH)
    expected = (N_BINS * BIN_WIDTH) / VIDEO_DT
    return (hi - lo) / expected


# ----------------------------------------------------------------- core processing
def bin_spikes(unit_spikes, win_start):
    """Bin spike times of all units into (n_units, n_trials, N_BINS) firing rates.

    `win_start` is the session-time of the first bin edge for each trial
    (= go cue - 2.5 s).  Windows are guaranteed non-overlapping (asserted by the
    caller), so every spike falls in at most one window and the whole session can
    be binned with a single `np.bincount`.
    """
    n_units = len(unit_spikes)
    n_trials = len(win_start)
    win_end = win_start + N_BINS * BIN_WIDTH

    if n_units == 0:
        return np.zeros((0, n_trials, N_BINS), dtype=np.float32)

    lens = np.array([len(s) for s in unit_spikes])
    if lens.sum() == 0:
        return np.zeros((n_units, n_trials, N_BINS), dtype=np.float32)
    t = np.concatenate(unit_spikes) if n_units > 1 else unit_spikes[0]
    uidx = np.repeat(np.arange(n_units), lens)

    ti = np.searchsorted(win_start, t, side='right') - 1
    valid = (ti >= 0) & (ti < n_trials)
    ti_c = np.where(valid, ti, 0)
    valid &= t < win_end[ti_c]

    t = t[valid]
    uidx = uidx[valid]
    ti = ti[valid]
    b = ((t - win_start[ti]) / BIN_WIDTH).astype(np.int64)
    np.clip(b, 0, N_BINS - 1, out=b)

    flat = (uidx * n_trials + ti) * N_BINS + b
    counts = np.bincount(flat, minlength=n_units * n_trials * N_BINS)
    counts = counts.reshape(n_units, n_trials, N_BINS)
    return (counts / BIN_WIDTH).astype(np.float32)


def tone_onset_rel(d):
    """Time of the last sample-epoch (tone) onset at or before each go cue,
    expressed relative to the go cue (so always <= 0)."""
    si = np.searchsorted(d['sample_start'], d['go_time'], side='right') - 1
    assert (si >= 0).all(), 'a trial has no preceding sample epoch'
    return d['sample_start'][si] - d['go_time']


def photostim_windows(d):
    """Per-trial [on, off] photostimulation times relative to the go cue.

    NaN for trials without photostimulation.
    """
    n = len(d['go_time'])
    on = np.full(n, np.nan)
    off = np.full(n, np.nan)
    if len(d['stim_on']) == 0:
        return on, off
    ti = np.searchsorted(d['trial_start'], d['stim_on'], side='right') - 1
    ok = (ti >= 0) & (ti < n)
    ti, s_on, s_off = ti[ok], d['stim_on'][ok], d['stim_off'][ok]
    # keep the first stimulation event of each trial (there is exactly one)
    on[ti] = s_on - d['go_time'][ti]
    off[ti] = s_off - d['go_time'][ti]
    return on, off


def tongue_bin_position(d, win_start):
    """Mean visible tongue-y per (trial, bin), and a mask of bins where the
    tongue was visible at all.

    A video frame counts as "tongue visible" when its DeepLabCut likelihood
    exceeds `TONGUE_LIKELIHOOD_THRESH`; the likelihood distribution is strongly
    bimodal so the exact threshold is immaterial.  Frames where the tongue is
    occluded carry no position information (the reference code replaces them by
    the session mean), so they are excluded from the bin average.
    """
    n_trials = len(win_start)
    ybar = np.zeros((n_trials, N_BINS))
    has = np.zeros((n_trials, N_BINS), dtype=bool)

    visible = d['tongue_lik'] > TONGUE_LIKELIHOOD_THRESH
    if visible.sum() == 0:
        return ybar, has

    vt = d['tongue_t'][visible]
    vy = d['tongue_y'][visible]
    win_end = win_start + N_BINS * BIN_WIDTH
    ti = np.searchsorted(win_start, vt, side='right') - 1
    ok = (ti >= 0) & (ti < n_trials)
    ti_c = np.where(ok, ti, 0)
    ok &= vt < win_end[ti_c]
    vt, vy, ti = vt[ok], vy[ok], ti[ok]
    if len(vt) == 0:
        return ybar, has

    b = ((vt - win_start[ti]) / BIN_WIDTH).astype(np.int64)
    np.clip(b, 0, N_BINS - 1, out=b)
    flat = ti * N_BINS + b
    nvis = np.bincount(flat, minlength=n_trials * N_BINS).reshape(n_trials, N_BINS)
    ysum = np.bincount(flat, weights=vy,
                       minlength=n_trials * N_BINS).reshape(n_trials, N_BINS)
    has = nvis > 0
    ybar[has] = ysum[has] / nvis[has]
    return ybar, has


def tongue_classes(d, win_start):
    """(n_trials, N_BINS) int array of tongue-y classes 0/1/2/3.

    Discretisation thresholds are the 40th and 60th percentiles of the
    tongue y-position *of this session* (bins in which the tongue is visible).
    """
    ybar, has = tongue_bin_position(d, win_start)
    out = np.full(ybar.shape, 3, dtype=np.int64)
    if has.sum() < 2:
        return out, np.nan, np.nan
    p_lo, p_hi = np.percentile(ybar[has], [TONGUE_LOW_PCT, TONGUE_HIGH_PCT])
    cls = np.where(ybar < p_lo, 0, np.where(ybar > p_hi, 2, 1))
    out[has] = cls[has]
    return out, float(p_lo), float(p_hi)


def choice_from_trials(d):
    """0 = left, 1 = right, 2 = no lick."""
    ch = np.full(len(d['outcome']), 2, dtype=np.int64)
    hit = d['outcome'] == 'hit'
    miss = d['outcome'] == 'miss'
    left = d['instruction'] == 'left'
    ch[hit & left] = 0
    ch[hit & ~left] = 1
    ch[miss & left] = 1      # error on a lick-left instruction -> licked right
    ch[miss & ~left] = 0
    return ch


def process_session(path, want_debug=False):
    """Convert one NWB file.  Returns a dict (or None if the session is rejected)."""
    t0 = time.time()
    d = read_session(path)
    t_read = time.time() - t0

    go = d['go_time']
    assert len(go) == len(d['trial_start'])
    assert np.all(np.diff(go) > (T_STOP - T_START)), 'trial windows overlap'
    win_start = go + T_START

    info = dict(session_id=d['session_id'], subject=d['subject'], path=path,
                n_trials_ephys=int(len(go)),
                n_trials_behavior=int(d['n_trials_behavior']),
                n_clusters=int(d['n_all_clusters']),
                n_good_clusters=int(d['n_good_clusters']))

    # ---- trial curation -----------------------------------------------------
    # (a) reward not contingent on the animal's choice -> `outcome` is not a
    #     behavioural report on these trials (reference: get_regular_trial_mask)
    keep = (~d['auto_water']) & (~d['free_water'])
    info['n_dropped_water'] = int((~keep).sum())
    # (b) the behavioural video must cover the whole extracted window, otherwise
    #     the tongue output cannot be determined (reference: get_bad_trial_inds)
    cov = video_coverage(d, win_start)
    keep &= cov >= MIN_VIDEO_COVERAGE
    info['n_dropped_video'] = int(((cov < MIN_VIDEO_COVERAGE)).sum())
    trial_idx = np.where(keep)[0]
    info['n_trials_kept'] = int(len(trial_idx))

    # ---- session curation ---------------------------------------------------
    perf, n_left, n_right = session_performance(d, trial_idx)
    vis_frac = float((d['tongue_lik'] > TONGUE_LIKELIHOOD_THRESH).mean())
    info.update(performance=perf, n_correct_left=n_left, n_correct_right=n_right,
                tongue_visible_fraction=vis_frac)
    if perf <= MIN_PERFORMANCE:
        info['rejected'] = 'behavioural performance'
        return None, info
    if n_left < MIN_CORRECT_PER_DIRECTION or n_right < MIN_CORRECT_PER_DIRECTION:
        info['rejected'] = 'too few correct lick-left / lick-right trials'
        return None, info
    if vis_frac > MAX_TONGUE_VISIBLE_FRACTION:
        info['rejected'] = 'tongue tracking failure (implausible visible fraction)'
        return None, info

    t1 = time.time()
    # ---- neural -------------------------------------------------------------
    fr = bin_spikes(d['unit_spikes'], win_start)          # (n_units, n_trials, 80)
    fr = fr[:, trial_idx, :]
    # drop units that are completely silent in every extracted window
    active = fr.any(axis=(1, 2))
    fr = fr[active]
    unit_region = d['unit_region'][active]
    t_neural = time.time() - t1

    # ---- inputs -------------------------------------------------------------
    t2 = time.time()
    tone_rel = tone_onset_rel(d)[trial_idx]               # <= 0
    time_from_tone = BIN_CENTERS[None, :] - tone_rel[:, None]     # (n_tr, 80)

    on, off = photostim_windows(d)
    on, off = on[trial_idx], off[trial_idx]
    has_stim = np.isfinite(on)
    # A bin is "on" when the laser is on at the bin centre, i.e. the binary
    # laser signal is sampled at the same instants as the bin centres.  Using
    # interval overlap instead would light up the bin containing the go cue for
    # a third of the stimulation trials, because the recorded laser-off
    # timestamp overshoots the go cue by ~1 ms; the photoinhibition always ended
    # before the go cue (datapaper STAR Methods).  With the centre rule each
    # 0.5 s stimulation covers exactly 10 bins.
    stim = np.zeros((len(trial_idx), N_BINS), dtype=np.float32)
    if has_stim.any():
        o = np.where(has_stim, on, np.inf)[:, None]
        f_ = np.where(has_stim, off, -np.inf)[:, None]
        stim = ((BIN_CENTERS[None, :] >= o) & (BIN_CENTERS[None, :] <= f_)
                ).astype(np.float32)

    # ---- outputs ------------------------------------------------------------
    choice = choice_from_trials(d)[trial_idx]
    outcome_map = {'ignore': 0, 'miss': 1, 'hit': 2}
    outcome = np.array([outcome_map[o] for o in d['outcome']])[trial_idx]
    early = (d['early_lick'] == 'early').astype(np.int64)[trial_idx]
    tongue, p_lo, p_hi = tongue_classes(d, win_start[trial_idx])
    t_io = time.time() - t2

    # ---- pack ---------------------------------------------------------------
    t3 = time.time()
    n_tr = len(trial_idx)
    neural = [np.ascontiguousarray(fr[:, i, :]) for i in range(n_tr)]
    inputs = [np.stack([time_from_tone[i], stim[i]]).astype(np.float32)
              for i in range(n_tr)]
    outputs = [np.stack([np.full(N_BINS, choice[i]), np.full(N_BINS, outcome[i]),
                         np.full(N_BINS, early[i]), tongue[i]]).astype(np.int64)
               for i in range(n_tr)]
    t_pack = time.time() - t3

    # provenance: row indices into intervals/trials of the source NWB file, one
    # per converted trial, so every value can be traced back to the raw data
    info['raw_trial_index'] = d['ephys_trials'][trial_idx].astype(int).tolist()
    info.update(n_trials=n_tr, n_neurons=int(fr.shape[0]),
                tongue_p40=float(p_lo), tongue_p60=float(p_hi),
                n_photostim_trials=int(has_stim.sum()),
                t_read=t_read, t_neural=t_neural, t_io=t_io, t_pack=t_pack,
                t_total=time.time() - t0)

    result = dict(session_id=d['session_id'], subject=d['subject'],
                  neural=neural, input=inputs, output=outputs,
                  unit_region=unit_region, info=info)

    if want_debug:
        result['debug'] = dict(
            go=go, trial_idx=trial_idx, win_start=win_start,
            tone_rel=tone_rel, stim_on=on, stim_off=off,
            tongue_t=d['tongue_t'], tongue_y=d['tongue_y'], tongue_lik=d['tongue_lik'],
            p_lo=p_lo, p_hi=p_hi, choice=choice, outcome=outcome, early=early,
            unit_anno=d['unit_anno'][active],
            trial_start=d['trial_start'], trial_stop=d['trial_stop'],
        )
    return result, info


# ------------------------------------------------------------------------ plotting
def make_processing_plot(res, outpath):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    dbg = res['debug']
    neural = res['neural']
    inputs = res['input']
    outputs = res['output']
    n_tr = len(neural)
    n_neu = neural[0].shape[0]

    fig, ax = plt.subplots(4, 3, figsize=(22, 18))

    # (0,0) trial-averaged population firing rate + epoch markers
    mean_fr = np.mean([n.mean(axis=0) for n in neural], axis=0)
    ax[0, 0].plot(BIN_CENTERS, mean_fr, '-o', ms=3)
    ax[0, 0].axvline(0, color='r', label='go cue')
    med_tone = np.median(dbg['tone_rel'])
    ax[0, 0].axvline(med_tone, color='g', ls='--', label='median tone onset')
    ax[0, 0].axvline(med_tone + 0.65, color='g', ls=':', label='sample end')
    ax[0, 0].set_xlabel('time from go cue (s)')
    ax[0, 0].set_ylabel('mean firing rate (Hz)')
    ax[0, 0].set_title(f'{res["session_id"]}: population PSTH ({n_neu} neurons, {n_tr} trials)')
    ax[0, 0].legend(fontsize=7)

    # (0,1) fraction of bins with zero spikes across the population (coverage)
    frac_zero = np.mean([(n.sum(axis=0) == 0) for n in neural], axis=0)
    ax[0, 1].plot(BIN_CENTERS, frac_zero, '-o', ms=3)
    ax[0, 1].axvline(0, color='r')
    ax[0, 1].set_xlabel('time from go cue (s)')
    ax[0, 1].set_ylabel('fraction of trials with no spikes')
    ax[0, 1].set_title('trial-interval coverage\n(rises late: error trials end early)')

    # (0,2) raster-like heatmap of one trial
    it = int(np.argmax([o[1, 0] == 2 for o in outputs]))   # a hit trial
    im = ax[0, 2].imshow(neural[it], aspect='auto', interpolation='nearest',
                         extent=[T_START, T_STOP, n_neu, 0], cmap='magma')
    ax[0, 2].axvline(0, color='c')
    ax[0, 2].set_title(f'trial {it}: neural (Hz)')
    ax[0, 2].set_xlabel('time from go cue (s)')
    ax[0, 2].set_ylabel('neuron')
    plt.colorbar(im, ax=ax[0, 2])

    # (1,0) input 0: time from tone onset
    for i in range(min(40, n_tr)):
        ax[1, 0].plot(BIN_CENTERS, inputs[i][0], alpha=0.4)
    ax[1, 0].axvline(0, color='r')
    ax[1, 0].axhline(0, color='k', ls=':')
    ax[1, 0].set_title('input 0: time from tone onset (s)\n(0 crossing = tone onset)')
    ax[1, 0].set_xlabel('time from go cue (s)')

    # (1,1) input 1: photostim raster
    stim = np.stack([inp[1] for inp in inputs])
    ax[1, 1].imshow(stim, aspect='auto', interpolation='nearest',
                    extent=[T_START, T_STOP, n_tr, 0], cmap='Greys')
    ax[1, 1].axvline(0, color='r')
    ax[1, 1].set_title(f'input 1: photostim on ({int((stim.max(axis=1) > 0).sum())} trials)')
    ax[1, 1].set_xlabel('time from go cue (s)')
    ax[1, 1].set_ylabel('trial')

    # (1,2) check photostim against raw event times
    ok = np.isfinite(dbg['stim_on'])
    if ok.any():
        ax[1, 2].hist(dbg['stim_on'][ok], bins=40, alpha=0.6, label='laser on')
        ax[1, 2].hist(dbg['stim_off'][ok], bins=40, alpha=0.6, label='laser off')
    ax[1, 2].axvline(0, color='r', label='go cue')
    ax[1, 2].set_title('raw photostim event times re. go cue')
    ax[1, 2].set_xlabel('s'); ax[1, 2].legend(fontsize=7)

    # (2,0) tongue y trace for one trial, raw frames + derived classes
    itt = int(np.argmax([(o[3] != 3).sum() for o in outputs]))
    t0 = dbg['win_start'][dbg['trial_idx'][itt]]
    m = (dbg['tongue_t'] >= t0) & (dbg['tongue_t'] < t0 + N_BINS * BIN_WIDTH)
    tt = dbg['tongue_t'][m] - (t0 - T_START)
    yy = dbg['tongue_y'][m]
    vis = dbg['tongue_lik'][m] > TONGUE_LIKELIHOOD_THRESH
    ax[2, 0].plot(tt[vis], yy[vis], '.', ms=2, label='visible frames')
    ax[2, 0].axhline(dbg['p_lo'], color='g', ls='--', label='40th pct')
    ax[2, 0].axhline(dbg['p_hi'], color='m', ls='--', label='60th pct')
    ax2 = ax[2, 0].twinx()
    ax2.step(BIN_CENTERS, outputs[itt][3], where='mid', color='k', alpha=0.5)
    ax2.set_ylabel('class (0-3)')
    ax[2, 0].axvline(0, color='r')
    ax[2, 0].set_title(f'trial {itt}: raw tongue y vs derived class')
    ax[2, 0].set_xlabel('time from go cue (s)'); ax[2, 0].legend(fontsize=7)

    # (2,1) tongue class raster
    tong = np.stack([o[3] for o in outputs])
    im = ax[2, 1].imshow(tong, aspect='auto', interpolation='nearest',
                         extent=[T_START, T_STOP, n_tr, 0], cmap='viridis', vmin=0, vmax=3)
    ax[2, 1].axvline(0, color='r')
    ax[2, 1].set_title('output 3: tongue y class (3 = not visible)')
    ax[2, 1].set_xlabel('time from go cue (s)'); ax[2, 1].set_ylabel('trial')
    plt.colorbar(im, ax=ax[2, 1])

    # (2,2) class fractions over time
    for c, lab in enumerate(OUTPUT_VALUES[3]):
        ax[2, 2].plot(BIN_CENTERS, (tong == c).mean(axis=0), label=lab)
    ax[2, 2].axvline(0, color='r')
    ax[2, 2].set_title('tongue class fraction over time\n(tongue appears after the go cue)')
    ax[2, 2].set_xlabel('time from go cue (s)'); ax[2, 2].legend(fontsize=7)

    # (3,0) per-trial outputs
    for k, name in enumerate(OUTPUT_NAMES[:3]):
        vals = np.array([o[k, 0] for o in outputs])
        ax[3, 0].plot(vals + 0.1 * k, '.', ms=2, label=name)
    ax[3, 0].set_title('per-trial outputs across the session')
    ax[3, 0].set_xlabel('trial'); ax[3, 0].legend(fontsize=7)

    # (3,1) output distributions
    labels, fracs = [], []
    for k in range(4):
        vals = np.concatenate([o[k] for o in outputs])
        for c, lab in enumerate(OUTPUT_VALUES[k]):
            labels.append(f'{OUTPUT_NAMES[k]}={lab}')
            fracs.append((vals == c).mean())
    ax[3, 1].barh(np.arange(len(fracs)), fracs)
    ax[3, 1].set_yticks(np.arange(len(fracs)))
    ax[3, 1].set_yticklabels(labels, fontsize=7)
    ax[3, 1].set_title('output class fractions (time-point weighted)')

    # (3,2) PSTH split by choice
    ch = np.array([o[0, 0] for o in outputs])
    for c, lab in enumerate(OUTPUT_VALUES[0]):
        m = ch == c
        if m.sum() == 0:
            continue
        ax[3, 2].plot(BIN_CENTERS, np.mean([neural[i].mean(axis=0)
                                            for i in np.where(m)[0]], axis=0), label=lab)
    ax[3, 2].axvline(0, color='r')
    ax[3, 2].set_title('population PSTH by choice')
    ax[3, 2].set_xlabel('time from go cue (s)'); ax[3, 2].legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(outpath, dpi=110)
    plt.close(fig)
    print(f'  wrote {outpath}')


# ---------------------------------------------------------------------------- main
def _worker(args):
    path, want_debug = args
    try:
        return process_session(path, want_debug=want_debug)
    except Exception as e:  # keep one bad file from killing the whole run
        import traceback
        traceback.print_exc()
        return None, dict(session_id=session_id_from_path(path), path=path,
                          rejected=f'error: {e}')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('outfile')
    g = ap.add_mutually_exclusive_group()
    g.add_argument('--full', action='store_true', default=True)
    g.add_argument('--sample', action='store_true')
    ap.add_argument('--show-processing', action='store_true')
    ap.add_argument('--jobs', type=int, default=min(32, os.cpu_count() or 8))
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', '*.nwb')))
    print(f'found {len(files)} NWB files')
    n_sample = 2
    if args.sample:
        # keep scanning until `n_sample` sessions pass the session-selection
        # criteria, so the sample is a usable mini-dataset
        files = files[:8]
        print(f'--sample: scanning up to {len(files)} files for '
              f'{n_sample} accepted sessions')

    # `--show-processing` plots the first two *accepted* sessions, so debug
    # information is requested for a few more files than that.
    n_debug = 8 if args.show_processing else 0
    jobs = [(p, i < n_debug) for i, p in enumerate(files)]
    n_plots = 0

    t0 = time.time()
    results, infos = [], []
    nworkers = max(1, min(args.jobs, len(files)))
    with ProcessPoolExecutor(max_workers=nworkers) as ex:
        for k, (res, info) in enumerate(ex.map(_worker, jobs)):
            if args.sample and len(results) >= n_sample:
                break
            infos.append(info)
            if res is None:
                print(f'[{k+1}/{len(files)}] SKIP {info["session_id"]}: '
                      f'{info.get("rejected")} (perf={info.get("performance", float("nan")):.3f}, '
                      f'L={info.get("n_correct_left")}, R={info.get("n_correct_right")})',
                      flush=True)
                continue
            results.append(res)
            i = res['info']
            print(f'[{k+1}/{len(files)}] {i["session_id"]}: {i["n_neurons"]} neurons, '
                  f'{i["n_trials"]} trials, perf={i["performance"]:.3f}, '
                  f'{i["t_total"]:.1f}s (read {i["t_read"]:.1f} / bin {i["t_neural"]:.1f} '
                  f'/ io {i["t_io"]:.2f} / pack {i["t_pack"]:.2f})', flush=True)
            if 'debug' in res:
                if n_plots < 2:
                    make_processing_plot(res, f'/app/processing_{res["session_id"]}.png')
                    n_plots += 1
                del res['debug']

    t_proc = time.time() - t0
    print(f'\nprocessed {len(results)}/{len(files)} sessions in {t_proc:.1f}s '
          f'({t_proc / max(1, len(files)):.2f}s per file, {nworkers} workers)')

    # ---------------- assemble -------------------------------------------------
    results.sort(key=lambda r: r['session_id'])
    subjects = sorted({r['subject'] for r in results})
    sub_index = {s: i for i, s in enumerate(subjects)}

    all_regions = sorted({str(reg) for r in results for reg in r['unit_region']})
    reg_index = {s: i for i, s in enumerate(all_regions)}

    data = {
        'neural': [r['neural'] for r in results],
        'input': [r['input'] for r in results],
        'output': [r['output'] for r in results],
        'subjects': subjects,
        'subject_idx': np.array([sub_index[r['subject']] for r in results], dtype=np.int64),
        'brain_regions': all_regions,
        'brain_region_idx': [np.array([reg_index[str(x)] for x in r['unit_region']], dtype=np.int64)
                             for r in results],
        'input_names': INPUT_NAMES,
        'output_names': OUTPUT_NAMES,
        'output_values': OUTPUT_VALUES,
        'metadata': {
            'dataset': 'Mesoscale Activity Map (MAP), DANDI:000363 v0.230822.0128',
            'references': [
                'Chen et al., Brain-wide neural activity underlying memory-guided '
                'movement, Cell 187, 676-691 (2024)',
                'Wang, Kurgyis et al., Brain-wide analysis reveals movement encoding '
                'structured across and within brain areas, Nat Neurosci (2025)',
            ],
            'task_description': (
                'Auditory memory-guided directional licking task. A series of three '
                '150 ms pure tones (3 kHz -> lick right, 12 kHz -> lick left) is played '
                'during a 0.65 s sample epoch, followed by a 1.2 s delay epoch during '
                'which the mouse must withhold licking. An auditory go cue ends the '
                'delay, and the mouse reports its choice by licking the left or right '
                'lick port within a 1.5 s answer period. Decoded outputs are the lick '
                'direction choice (left / right / no lick), the trial outcome '
                '(ignore / miss / hit), whether the trial contained an early lick '
                '(no / yes), and the discretised side-view tongue y-position per time '
                'bin (<40th percentile / 40-60th percentile / >60th percentile of the '
                'session, or not visible). Decoder inputs are the time since the tone '
                '(sample-epoch) onset and whether ALM photoinhibition is on.'),
            'time_bin_size': BIN_WIDTH * 1000.0,   # ms
            'temporal_alignment_event': 'go cue onset (auditory Go cue ending the delay epoch)',
            'off_start': T_START,
            'off_end': T_STOP,
            'neural_units': 'firing rate, Hz (spike count per 50 ms bin / 0.05 s)',
            'n_timepoints': N_BINS,
            'species': 'Mus musculus',
            'recording': 'Neuropixels 1.0 / 2.0, 2-5 simultaneous probes, Kilosort2 spike sorting',
            'neuron_curation': (
                "units/classification == 'good' (quality-control classifier of the "
                'MAP spike-sorting white paper), a CCF annotation that maps to one of '
                'the 14 major reference regions, and non-zero spiking within the '
                'extracted windows'),
            'trial_curation': (
                'auto-water and free-water trials excluded (reward not contingent on '
                'choice). Photostimulation, early-lick and no-response trials are kept '
                'because they are required decoder inputs/outputs.'),
            'session_curation': (
                f'behavioural performance > {MIN_PERFORMANCE:.0%} on control '
                f'(no-photostim) non-early-lick trials and >= '
                f'{MIN_CORRECT_PER_DIRECTION} correct lick-left and lick-right trials '
                '(data paper STAR Methods session-selection criteria)'),
            'known_limitations': (
                'The published dataset only contains spikes inside '
                '[trial.start_time, trial.stop_time]. For error (miss) trials the '
                'exported interval ends ~0.2-0.5 s after the go cue, so the last bins '
                'of those trials necessarily contain zero spikes (15.6% of raw trials '
                'have stop_time - go_cue < 1.5 s). No imputation is possible.'),
            'input_descriptions': [
                'seconds since the onset of the instruction tone series (last sample '
                'epoch onset at or before the go cue); negative before tone onset',
                '1 while the ALM photoinhibition laser is on, else 0',
            ],
            'output_descriptions': [
                'lick direction choice: derived from outcome and trial instruction '
                '(hit -> instructed side, miss -> opposite side, ignore -> no lick)',
                'trial outcome as reported in intervals/trials/outcome',
                'whether the trial contained a lick during the sample or delay epoch',
                'side-view DeepLabCut tongue y position, averaged over the visible '
                f'frames of each 50 ms bin and discretised at the {TONGUE_LOW_PCT:.0f}th '
                f'and {TONGUE_HIGH_PCT:.0f}th percentiles of all visible frames of the '
                f'session; class 3 when no frame in the bin has DeepLabCut likelihood '
                f'> {TONGUE_LIKELIHOOD_THRESH}',
            ],
            'session_info': [
                {k: v for k, v in r['info'].items()
                 if k not in ('t_read', 't_neural', 't_io', 't_pack', 't_total',
                              'raw_trial_index')}
                for r in results
            ],
            # row indices into intervals/trials of each source NWB file, one list
            # per session, aligned with the trials in `neural`/`input`/`output`
            'raw_trial_index': [r['info']['raw_trial_index'] for r in results],
            'n_sessions_available': len(files),
            'n_sessions_kept': len(results),
            'conversion_seconds': t_proc,
        },
    }

    # ---------------- summary --------------------------------------------------
    nneu = np.array([len(b) for b in data['brain_region_idx']])
    ntr = np.array([len(s) for s in data['neural']])
    print('\n===== converted dataset summary =====')
    print(f'sessions            : {len(results)} (of {len(files)} files)')
    print(f'subjects            : {len(subjects)}')
    print(f'neurons total       : {nneu.sum()}  (median/session {np.median(nneu):.0f}, '
          f'mean {nneu.mean():.1f})')
    print(f'trials total        : {ntr.sum()}  (mean/session {ntr.mean():.1f}, '
          f'range {ntr.min()}-{ntr.max()})')
    print(f'brain regions       : {len(all_regions)}')
    for k in range(4):
        vals = np.concatenate([o[k] for r in results for o in r['output']])
        fr_ = [float((vals == c).mean()) for c in range(len(OUTPUT_VALUES[k]))]
        print(f'output {k} {OUTPUT_NAMES[k]:<20s}: ' +
              ', '.join(f'{lab}={f:.3f}' for lab, f in zip(OUTPUT_VALUES[k], fr_)))
    i0 = np.concatenate([inp[0] for r in results for inp in r['input']])
    i1 = np.concatenate([inp[1] for r in results for inp in r['input']])
    print(f'input 0 range       : [{i0.min():.3f}, {i0.max():.3f}]')
    print(f'input 1 fraction on : {i1.mean():.4f}')
    fr_all = np.concatenate([n.mean(axis=1) for r in results for n in r['neural'][:5]])
    print(f'firing rate (Hz)    : mean {fr_all.mean():.2f}, max {fr_all.max():.1f}')

    print(f'\nwriting {args.outfile} ...')
    t1 = time.time()
    with open(args.outfile, 'wb') as f:
        pickle.dump(data, f, protocol=4)
    print(f'wrote {os.path.getsize(args.outfile) / 1e9:.2f} GB in {time.time() - t1:.1f}s')

    with open(os.path.splitext(args.outfile)[0] + '_session_info.json', 'w') as f:
        json.dump(infos, f, indent=1, default=float)
    print(f'total wall time {time.time() - t0:.1f}s')


if __name__ == '__main__':
    main()
