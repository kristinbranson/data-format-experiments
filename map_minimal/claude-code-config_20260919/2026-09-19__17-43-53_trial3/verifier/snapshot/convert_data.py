#!/usr/bin/env python
"""
Convert the Mesoscale Activity Map (MAP) Neuropixels dataset (Chen et al. 2023,
DANDI:000363) into the decoder format described in the task specification.

Data source
-----------
/app/data/sub-<mouse>/sub-<mouse>_ses-<datetime>_behavior+ecephys(+ogen).nwb
174 sessions from 28 mice.  Mice performed an auditory delayed-response task:
a sample tone (3 or 12 kHz, three 150 ms pips) instructs the mouse which lick
port to lick, a delay epoch follows, and an auditory go cue releases the
response.  Silicon-probe spikes, DeepLabCut side-view marker tracking (jaw,
nose, tongue at 300 Hz) and bilateral ALM photoinhibition are stored in each
file.

Processing decisions (see README-style notes in the docstrings below) follow
Chen et al. 2023 ("Brain-wide neural activity underlying memory-guided
movement", the data paper) and Wang, Kurgyis et al. 2025 ("Brain-wide analysis
reveals movement encoding structured across and within brain areas", the method
paper) plus the accompanying code in /app/code, except where the decoding task
specification requires otherwise (those exceptions are marked DEVIATION).

Output: /app/converted_data.pkl
"""

import glob
import json
import os
import pickle
import sys
import time
import urllib.request
from collections import OrderedDict
from multiprocessing import Pool

import h5py
import numpy as np

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

DATA_DIR = '/app/data'
CACHE_DIR = '/app/cache'
OUT_FILE = '/app/converted_data.pkl'
ONTOLOGY_FILE = '/app/allen_structure_graph.json'
ONTOLOGY_URL = 'http://api.brain-map.org/api/v2/structure_graph_download/1.json'

# Temporal alignment / binning (from the task specification).
ALIGN_EVENT = 'go cue onset'
T_START = -2.5           # s relative to the go cue
T_END = 1.5              # s relative to the go cue
BIN_WIDTH = 0.05         # s.  Non-overlapping bins (stride == width).
BIN_EDGES = np.round(T_START + BIN_WIDTH * np.arange(
    int(round((T_END - T_START) / BIN_WIDTH)) + 1), 10)
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2.0
NBINS = len(BIN_CENTERS)

# Session selection (data paper, STAR Methods): "We selected experimental
# sessions for analysis based on following criteria: overall behavioral
# performance (> 65%), and at least 50 correct lick left and lick right trials
# each."  Performance is "the fraction of correct control trials (i.e. no
# photostimulation), excluding any early lick trials".  Reproducing this on the
# NWB trial tables gives 154/174 sessions with a mean performance of 83.4%
# (range 65.6-98.9%), matching the 84% (65-99%) quoted in the paper.
MIN_PERFORMANCE = 0.65
MIN_CORRECT_PER_DIRECTION = 50

# A session needs >= 2 trials for the decoder's train/test split, and at least
# one neuron to be usable at all.
MIN_TRIALS_PER_SESSION = 2
MIN_NEURONS_PER_SESSION = 1
# Minimum number of observed bins for a trial to be kept (see mask_bins()).
MIN_BINS_PER_TRIAL = 20

# DeepLabCut confidence above which the tongue is called "visible".  The
# likelihood distribution is extremely bimodal (median 6e-5 when the tongue is
# in the mouth vs > 0.999 when it is out; only 0.07% of frames fall between 0.5
# and 0.9), so the exact threshold is immaterial.
TONGUE_LIKELIHOOD_THRESH = 0.5
# Percentile cut points for the tongue y position (task specification).
TONGUE_PCTILES = (40.0, 60.0)
# Session-level quality control of the side-view tongue tracking.  Most
# sessions are clean, but a few are not usable for the tongue output:
#  * in six sessions the camera only ran from the trial start to the go cue and
#    in one it ran for 47 s in total, so the response epoch - the part of the
#    trial in which the tongue is actually out - has no video at all.  Those
#    sessions cover 1-64% of the analysis window, every other session covers
#    >= 99.97%, so the cut is not sensitive to the threshold.
#  * in one session the tracker locks onto something near the top of the frame
#    and reports the tongue as visible in 96% of frames.
# Tracking is validated against the lick-port sensors: visible tongue frames
# should coincide with recorded licks (precision) and licks recorded while the
# camera was running should coincide with a visible tongue (recall).  The
# failing session scores a precision of 0.12, the next lowest session 0.29.
MIN_VIDEO_COVERAGE = 0.95
MIN_TONGUE_LICK_PRECISION = 0.2
MIN_TONGUE_LICK_RECALL = 0.5
LICK_MATCH_TOL = 0.15     # s
# Five-sigma velocity threshold for marker outlier rejection (method paper:
# "We identified outliers by a five-sigma threshold on velocity across frames
# and imputed outliers from nearby frames.").
MARKER_OUTLIER_SIGMA = 5.0

INPUT_NAMES = ['time_from_tone_onset', 'photostim_on']
OUTPUT_NAMES = ['choice', 'outcome', 'early_lick', 'tongue_y_position']
OUTPUT_VALUES = [
    ['left', 'right', 'no lick'],
    ['ignore', 'miss', 'hit'],
    ['no', 'yes'],
    ['<40th percentile', '40th-60th percentile', '>60th percentile',
     'not visible'],
]

# Coarse brain-region grouping.  The data paper groups units into ALM, Medulla,
# Midbrain, Striatum, Thalamus, Pons, Cerebellum, Hypothalamus, Hippocampus,
# Orbital, OtherCortex, Olfactory, CorticalSubplate and Pallidum (see
# preprocessing_DJ_2022Aug.process_one_sess in /app/code).  Each unit's CCF
# annotation (`units/anno_name`) is walked up the Allen ontology and assigned to
# the first matching division below, so the order matters (MO before Isocortex,
# ORB before Isocortex).
#
# ALM is a physiological designation rather than an Allen structure; the paper
# used a CCF voxel mask that is not distributed with the code, so here ALM is
# defined as the motor cortex (MOs/MOp) units.  Every motor-cortex unit in the
# dataset was recorded on a probe whose insertion target is annotated "left
# ALM"/"right ALM" (AP +2.5, ML +/-1.5 mm), so this is a faithful stand-in; it
# yields 7,885 units against the 8,717 quoted in the paper.
REGION_RULES = [
    ('MO', 'ALM'),
    ('ORB', 'Orbital'),
    ('OLF', 'Olfactory'),
    ('HPF', 'Hippocampus'),
    ('CTXsp', 'CorticalSubplate'),
    ('Isocortex', 'OtherCortex'),
    ('STR', 'Striatum'),
    ('PAL', 'Pallidum'),
    ('TH', 'Thalamus'),
    ('HY', 'Hypothalamus'),
    ('MB', 'Midbrain'),
    ('P', 'Pons'),
    ('MY', 'Medulla'),
    ('CB', 'Cerebellum'),
]
REGION_NAMES = [name for _, name in REGION_RULES] + ['Other']


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

def _str_col(dataset):
    """Read an HDF5 column of (possibly byte) strings as a numpy array of str."""
    return np.array([x.decode() if isinstance(x, bytes) else str(x)
                     for x in dataset[:]])


def load_region_map():
    """name -> coarse region index, built from the Allen CCF structure graph."""
    if not os.path.exists(ONTOLOGY_FILE):
        urllib.request.urlretrieve(ONTOLOGY_URL, ONTOLOGY_FILE)
    with open(ONTOLOGY_FILE) as f:
        root = json.load(f)['msg'][0]

    acronym_path = {}

    def walk(node, path):
        path = path + [node['acronym']]
        acronym_path[node['name'].strip()] = path
        for child in node['children']:
            walk(child, path)

    walk(root, [])

    region_idx = {}
    for name, path in acronym_path.items():
        idx = len(REGION_NAMES) - 1  # 'Other'
        for acronym, group in REGION_RULES:
            if acronym in path:
                idx = REGION_NAMES.index(group)
                break
        region_idx[name] = idx
    return region_idx


# --------------------------------------------------------------------------
# Per-session conversion
# --------------------------------------------------------------------------

def session_trial_table(f):
    """Pull the fields of the NWB trials table that this conversion uses."""
    trials = f['intervals/trials']
    tbl = {
        'start_time': trials['start_time'][:],
        'stop_time': trials['stop_time'][:],
        'trial_id': trials['trial'][:].astype(np.int64),
        'outcome': _str_col(trials['outcome']),
        'early_lick': _str_col(trials['early_lick']),
        'instruction': _str_col(trials['trial_instruction']),
        'auto_water': trials['auto_water'][:].astype(int),
        'free_water': trials['free_water'][:].astype(int),
        'photostim_onset': _str_col(trials['photostim_onset']),
        'photostim_duration': _str_col(trials['photostim_duration']),
        'photostim_power': _str_col(trials['photostim_power']),
    }
    tbl['go_time'] = f['acquisition/BehavioralEvents/go_start_times/timestamps'][:]
    assert len(tbl['go_time']) == len(tbl['start_time'])
    return tbl


def session_performance(tbl):
    """Behavioral performance and correct-trial counts used for session QC.

    Performance = fraction correct among control trials: no photostimulation,
    no early lick, no free water, and a response given (hit or miss).
    """
    no_early = tbl['early_lick'] == 'no early'
    control = no_early & (tbl['photostim_onset'] == 'N/A') & (tbl['free_water'] == 0)
    responded = control & (tbl['outcome'] != 'ignore')
    n_resp = int(responded.sum())
    perf = float((tbl['outcome'][responded] == 'hit').sum()) / n_resp if n_resp else 0.0
    hit = tbl['outcome'] == 'hit'
    n_left = int((hit & (tbl['instruction'] == 'left') & no_early).sum())
    n_right = int((hit & (tbl['instruction'] == 'right') & no_early).sum())
    return perf, n_left, n_right


def tone_onset_times(f, tbl):
    """Time of the instruction tone (sample epoch onset) for every trial.

    Licking during the sample/delay epoch triggers a replay of the epoch, so a
    trial can contain several sample-epoch onsets.  The onset that determines
    the trial structure leading up to the go cue is the last one before the go
    cue, which is what is used here.  Trials without a recorded sample onset
    (none were found in this dataset) fall back to the session median offset.
    """
    sample = f['acquisition/BehavioralEvents/sample_start_times/timestamps'][:]
    go = tbl['go_time']
    start = tbl['start_time']
    tone = np.full(len(go), np.nan)
    # Assign each sample-epoch onset to the trial containing it, keep the last.
    trial_of = np.searchsorted(start, sample, side='right') - 1
    valid = (trial_of >= 0) & (trial_of < len(go))
    for t, s in zip(trial_of[valid], sample[valid]):
        if s <= go[t]:
            tone[t] = s
    rel = tone - go
    if np.any(np.isnan(rel)):
        rel[np.isnan(rel)] = np.nanmedian(rel)
    return rel


def photostim_intervals(tbl):
    """(onset, offset) of photostimulation relative to the go cue, per trial.

    `photostim_onset` in the trials table is given relative to the trial start
    time; this was verified against acquisition/BehavioralEvents/
    photostim_start_times (exact match for every stimulated trial).  Trials
    without photostimulation get (nan, nan).
    """
    n = len(tbl['start_time'])
    on = np.full(n, np.nan)
    off = np.full(n, np.nan)
    stim = tbl['photostim_onset'] != 'N/A'
    idx = np.flatnonzero(stim)
    for i in idx:
        onset = float(tbl['photostim_onset'][i])
        dur = float(tbl['photostim_duration'][i])
        # relative to trial start -> relative to go cue
        on[i] = tbl['start_time'][i] + onset - tbl['go_time'][i]
        off[i] = on[i] + dur
    return on, off


def _nearest_within(a, b, tol):
    """Fraction of the times in `a` that have a time in `b` within `tol`."""
    if len(a) == 0 or len(b) == 0:
        return np.nan
    i = np.clip(np.searchsorted(b, a), 1, len(b) - 1)
    d = np.minimum(np.abs(a - b[i - 1]), np.abs(a - b[i]))
    return float((d < tol).mean())


def tongue_lick_agreement(ts, visible, licks):
    """Agreement between "tongue visible" frames and lick-port events.

    Returns (precision, recall): the fraction of visible frames close to a
    lick, and the fraction of licks that happened while the camera was running
    and are close to a visible frame.
    """
    if len(ts) == 0:
        return 0.0, 0.0
    frame_dt = 0.01
    i = np.clip(np.searchsorted(ts, licks), 1, len(ts) - 1)
    covered = np.minimum(np.abs(licks - ts[i - 1]),
                         np.abs(licks - ts[i])) < frame_dt
    precision = _nearest_within(ts[visible], licks, LICK_MATCH_TOL)
    recall = _nearest_within(licks[covered], ts[visible], LICK_MATCH_TOL)
    return (0.0 if np.isnan(precision) else precision,
            0.0 if np.isnan(recall) else recall)


def lick_times(f):
    """Sorted lick times at the left and right ports."""
    be = f['acquisition/BehavioralEvents']
    return (np.sort(be['left_lick_times/timestamps'][:]),
            np.sort(be['right_lick_times/timestamps'][:]))


def lick_choice(left, right, tbl):
    """Lick direction chosen by the mouse: 0 left, 1 right, 2 no lick.

    Taken from the first lick recorded at either port during the 1.5 s answer
    period that follows the go cue (the window this conversion extracts).  This
    agrees with the label implied by outcome x instruction (hit -> instructed
    port, miss -> opposite port, ignore -> no lick) on >99.3% of trials; the
    direct read-out from the lick ports is used because "lick direction choice"
    is a behavioral measurement rather than a re-encoding of `outcome`.
    """
    go = tbl['go_time']
    choice = np.full(len(go), 2, dtype=np.int8)
    for i, g in enumerate(go):
        li = np.searchsorted(left, g)
        ri = np.searchsorted(right, g)
        tl = left[li] if li < len(left) else np.inf
        tr = right[ri] if ri < len(right) else np.inf
        end = g + T_END
        if tl >= end and tr >= end:
            continue
        choice[i] = 0 if tl <= tr else 1
    return choice


def tongue_trace(f):
    """Side-view tongue tracking: timestamps, y position, visibility mask.

    The tongue is "visible" when the DeepLabCut likelihood exceeds
    TONGUE_LIKELIHOOD_THRESH; when it is inside the mouth the tracker returns a
    meaningless position with a likelihood of ~1e-5.  Following the method
    paper, isolated outliers are identified with a five-sigma threshold on the
    frame-to-frame velocity and imputed from neighboring frames.
    """
    key = 'acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking'
    if key not in f:
        return None, None, None
    ts = f[key + '/timestamps'][:]
    data = f[key + '/data'][:]
    y = data[:, 1].astype(np.float64)
    visible = data[:, 2] > TONGUE_LIKELIHOOD_THRESH

    vis_idx = np.flatnonzero(visible)
    if len(vis_idx) > 3:
        yv = y[vis_idx]
        tv = ts[vis_idx]
        dt = np.diff(tv)
        # Velocity only between frames that are adjacent in the video; the
        # recording is split into per-trial segments and the tongue is only
        # visible in bouts, so most consecutive visible frames are not adjacent.
        adjacent = (dt > 0) & (dt < 0.01)
        vel = np.full(len(dt), np.nan)
        vel[adjacent] = np.diff(yv)[adjacent] / dt[adjacent]
        sigma = np.nanstd(vel) if adjacent.any() else np.nan
        if np.isfinite(sigma) and sigma > 0:
            jump = np.abs(vel) > MARKER_OUTLIER_SIGMA * sigma
            # A frame is an outlier when the position jumps on the way in and
            # again on the way out.
            outlier = np.zeros(len(yv), dtype=bool)
            outlier[1:-1] = jump[:-1] & jump[1:]
            if outlier.any() and (~outlier).sum() > 1:
                keep = np.flatnonzero(~outlier)
                yv = np.interp(np.arange(len(yv)), keep, yv[keep])
                y = y.copy()
                y[vis_idx] = yv
    return ts, y, visible


def mask_bins(tbl, trial):
    """Indices of the bins of the analysis window that hold observed spikes.

    The NWB files only contain spikes inside each trial's interval
    (units/obs_intervals is exactly [start_time, stop_time] for every unit, and
    100% of spikes fall inside it): nothing was exported for the inter-trial
    interval.  The go-cue-aligned window [-2.5, 1.5] s is not always contained
    in that interval - in particular an incorrect lick ends the trial
    immediately, so error ("miss") trials typically stop 0.3-1.0 s after the go
    cue, and ~4% of trials start less than 2.5 s before it.

    Bins outside the observed interval are therefore dropped rather than filled
    with zeros.  Filling them would invent "all neurons silent" samples whose
    presence is perfectly correlated with the miss outcome, which the decoder
    would read as a free answer; dropping instead keeps every trial (including
    the error trials that the `outcome` output needs) and every sample the
    probes actually recorded.  Trials keep a common bin grid and bin width, only
    their extent varies, which the decoder format allows (T[session][trial]).

    The video is recorded per trial as well, starting at the trial start; in
    the sessions kept here it outruns the spike interval (it ends >= 1.8 s after
    the go cue), so masking to the spike interval also removes essentially every
    bin without tracking (99.99% of the kept bins have video).
    """
    lo = tbl['start_time'][trial] - tbl['go_time'][trial]
    hi = tbl['stop_time'][trial] - tbl['go_time'][trial]
    keep = np.flatnonzero((BIN_EDGES[:-1] >= lo) & (BIN_EDGES[1:] <= hi))
    return keep


def convert_session(args):
    """Convert one NWB session; returns a summary dict or None if rejected."""
    path, region_map = args
    name = os.path.basename(path).replace('.nwb', '')
    subject = os.path.basename(path).split('_')[0].replace('sub-', '')
    out = {'file': path, 'session_name': name, 'subject': subject}

    with h5py.File(path, 'r') as f:
        tbl = session_trial_table(f)
        perf, n_left, n_right = session_performance(tbl)
        out.update(performance=perf, n_correct_left=n_left, n_correct_right=n_right,
                   n_trials_total=len(tbl['start_time']))

        # ---- session selection (data paper criteria) ----
        if perf <= MIN_PERFORMANCE or min(n_left, n_right) < MIN_CORRECT_PER_DIRECTION:
            out['rejected'] = 'behavior'
            return out

        # ---- neuron selection ----
        # The data paper's quality control trains a per-brain-area logistic
        # regression classifier on 15 cluster quality metrics and keeps the
        # units it labels 'good' (units/classification); 69,453 of the 272,227
        # clusters in these files pass (25.5%), matching the 69,943 "good units"
        # and "25.9% of clusters reported by Kilosort2" of the paper.  Those
        # units are exactly the ones carrying a CCF annotation.
        units = f['units']
        classification = _str_col(units['classification'])
        anno = _str_col(units['anno_name'])
        good = np.flatnonzero((classification == 'good') & (anno != ''))
        if len(good) < MIN_NEURONS_PER_SESSION:
            out['rejected'] = 'no good units'
            return out

        # ---- trial selection ----
        # Free-water trials are excluded (method paper: "trials with ...  water
        # administration regardless of the animals' choice (free water trials)
        # ... were excluded from all analyses").
        # DEVIATION: the same sentence excludes photoinhibition, early-lick and
        # ignore trials.  They are kept here because the decoding task asks for
        # photostimulation as an input and for early lick / ignore as outputs;
        # dropping them would leave those variables constant.
        bins = [mask_bins(tbl, t) for t in range(len(tbl['start_time']))]
        short = np.array([len(b) < MIN_BINS_PER_TRIAL for b in bins])
        keep_trial = (tbl['free_water'] == 0) & ~short
        trials = np.flatnonzero(keep_trial)
        out['n_trials_free_water'] = int((tbl['free_water'] == 1).sum())
        out['n_trials_short'] = int((short & (tbl['free_water'] == 0)).sum())
        if len(trials) < MIN_TRIALS_PER_SESSION:
            out['rejected'] = 'too few trials'
            return out

        go = tbl['go_time']

        # ---- neural: spike counts -> firing rate (Hz) per 50 ms bin ----
        spike_index = units['spike_times_index'][:]
        edges = (go[trials][:, None] + BIN_EDGES[None, :]).ravel()
        rates = np.zeros((len(good), len(trials), NBINS), dtype=np.float32)
        for i, u in enumerate(good):
            lo = 0 if u == 0 else spike_index[u - 1]
            st = units['spike_times'][lo:spike_index[u]]
            counts = np.diff(
                np.searchsorted(st, edges).reshape(len(trials), NBINS + 1), axis=1)
            rates[i] = counts / BIN_WIDTH

        # In eight sessions the ephys recording covers only a contiguous block
        # of the behavioral session (the probes were inserted after the mouse
        # started working, or the recording was stopped before it finished), and
        # the files simply contain no spikes for the remaining trials.  A trial
        # in which not one of several hundred units fires for four seconds is a
        # recording gap rather than a physiological observation, so those trials
        # are dropped.  The transition is clean: the trials bordering such a
        # block fire at 0.77-1.2x the session's median population rate.
        has_spikes = rates.sum(axis=(0, 2)) > 0
        out['n_trials_no_ephys'] = int((~has_spikes).sum())
        trials = trials[has_spikes]
        rates = rates[:, has_spikes]
        if len(trials) < MIN_TRIALS_PER_SESSION:
            out['rejected'] = 'no ephys'
            return out

        # Drop units without a single spike anywhere in the analysis windows:
        # they carry no information and give the per-session PCA a null
        # direction.
        alive = rates.any(axis=(1, 2))
        out['n_units_good'] = int(len(good))
        out['n_units_silent'] = int((~alive).sum())
        good = good[alive]
        rates = rates[alive]
        if len(good) < MIN_NEURONS_PER_SESSION:
            out['rejected'] = 'no active units'
            return out

        region_idx = np.array([region_map.get(a, len(REGION_NAMES) - 1)
                               for a in anno[good]], dtype=np.int64)

        # ---- inputs ----
        tone_rel = tone_onset_times(f, tbl)
        stim_on, stim_off = photostim_intervals(tbl)

        # ---- outputs ----
        left_licks, right_licks = lick_times(f)
        choice = lick_choice(left_licks, right_licks, tbl)
        outcome_code = {'ignore': 0, 'miss': 1, 'hit': 2}
        outcome = np.array([outcome_code[o] for o in tbl['outcome']], dtype=np.int8)
        early = (tbl['early_lick'] == 'early').astype(np.int8)

        ts, tongue_y, tongue_vis = tongue_trace(f)
        if ts is None:
            out['rejected'] = 'no tongue tracking'
            return out

        precision, recall = tongue_lick_agreement(
            ts, tongue_vis, np.sort(np.concatenate([left_licks, right_licks])))
        out['tongue_lick_precision'] = precision
        out['tongue_lick_recall'] = recall

        # Mean tongue y over the visible video frames of each bin; a bin with no
        # visible frame is "not visible".
        bin_y = np.full((len(trials), NBINS), np.nan)
        has_video = np.zeros((len(trials), NBINS), dtype=bool)
        for k, t in enumerate(trials):
            lo = np.searchsorted(ts, go[t] + BIN_EDGES[:-1])
            hi = np.searchsorted(ts, go[t] + BIN_EDGES[1:])
            has_video[k] = hi > lo
            for b in range(NBINS):
                if hi[b] <= lo[b]:
                    continue
                sl = slice(lo[b], hi[b])
                vis = tongue_vis[sl]
                if vis.any():
                    bin_y[k, b] = tongue_y[sl][vis].mean()

    # Percentiles of the tongue y position over the session, computed on the
    # same quantity that is discretized (per-bin position of visible bins).
    observed = np.zeros((len(trials), NBINS), dtype=bool)
    for k, t in enumerate(trials):
        observed[k, bins[t]] = True

    coverage = float(has_video[observed].mean())
    out['video_coverage'] = coverage
    if coverage < MIN_VIDEO_COVERAGE:
        out['rejected'] = 'video does not cover the analysis window'
        return out

    if precision < MIN_TONGUE_LICK_PRECISION or recall < MIN_TONGUE_LICK_RECALL:
        out['rejected'] = 'tongue tracking failed'
        return out

    vis_vals = bin_y[observed & np.isfinite(bin_y)]
    if len(vis_vals) >= 10:
        p40, p60 = np.percentile(vis_vals, TONGUE_PCTILES)
    else:
        p40 = p60 = np.inf
    tongue_class = np.full((len(trials), NBINS), 3, dtype=np.int8)
    vis_bin = np.isfinite(bin_y)
    tongue_class[vis_bin & (bin_y < p40)] = 0
    tongue_class[vis_bin & (bin_y >= p40) & (bin_y <= p60)] = 1
    tongue_class[vis_bin & (bin_y > p60)] = 2

    # ---- assemble per-trial arrays ----
    neural, inputs, outputs = [], [], []
    for k, t in enumerate(trials):
        b = bins[t]
        neural.append(np.ascontiguousarray(rates[:, k, b]))
        inp = np.empty((2, len(b)), dtype=np.float32)
        inp[0] = BIN_CENTERS[b] - tone_rel[t]
        if np.isfinite(stim_on[t]):
            inp[1] = ((BIN_CENTERS[b] >= stim_on[t]) &
                      (BIN_CENTERS[b] < stim_off[t])).astype(np.float32)
        else:
            inp[1] = 0.0
        inputs.append(inp)
        outp = np.empty((4, len(b)), dtype=np.int8)
        outp[0] = choice[t]
        outp[1] = outcome[t]
        outp[2] = early[t]
        outp[3] = tongue_class[k, b]
        outputs.append(outp)

    out.update(
        neural=neural, input=inputs, output=outputs,
        brain_region_idx=region_idx,
        n_neurons=int(len(good)), n_trials=int(len(trials)),
        trial_info={
            'trial_id': tbl['trial_id'][trials],
            'go_time': go[trials],
            'first_bin': np.array([bins[t][0] for t in trials], dtype=np.int16),
            'n_bins': np.array([len(bins[t]) for t in trials], dtype=np.int16),
            'tone_onset_rel_go': tone_rel[trials].astype(np.float32),
            'photostim_on_rel_go': stim_on[trials].astype(np.float32),
            'photostim_off_rel_go': stim_off[trials].astype(np.float32),
            'instruction': tbl['instruction'][trials],
            'auto_water': tbl['auto_water'][trials].astype(np.int8),
        },
        tongue_percentiles=(float(p40), float(p60)),
        n_photostim_trials=int(np.isfinite(stim_on[trials]).sum()),
    )
    return out


def _worker(args):
    path = args[0]
    cache = os.path.join(CACHE_DIR, os.path.basename(path).replace('.nwb', '.pkl'))
    if os.path.exists(cache):
        return cache
    t0 = time.time()
    res = convert_session(args)
    with open(cache + '.tmp', 'wb') as fh:
        pickle.dump(res, fh, protocol=4)
    os.replace(cache + '.tmp', cache)
    print(f"{os.path.basename(path)}: "
          f"{res.get('rejected', 'ok')} "
          f"({res.get('n_neurons', 0)} neurons, {res.get('n_trials', 0)} trials) "
          f"[{time.time() - t0:.0f}s]", flush=True)
    return cache


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main(n_workers=16):
    os.makedirs(CACHE_DIR, exist_ok=True)
    region_map = load_region_map()
    files = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', '*.nwb')))
    print(f"{len(files)} NWB sessions found", flush=True)

    with Pool(n_workers) as pool:
        caches = pool.map(_worker, [(p, region_map) for p in files], chunksize=1)

    data = {
        'neural': [], 'input': [], 'output': [],
        'subjects': [], 'subject_idx': [],
        'brain_regions': REGION_NAMES,
        'brain_region_idx': [],
        'input_names': INPUT_NAMES,
        'output_names': OUTPUT_NAMES,
        'output_values': OUTPUT_VALUES,
        'metadata': {},
    }
    session_info = []
    trial_metadata = []
    subjects = OrderedDict()
    rejected = []

    for cache in caches:
        with open(cache, 'rb') as fh:
            res = pickle.load(fh)
        if 'rejected' in res:
            rejected.append((res['session_name'], res['rejected'],
                             round(res['performance'], 3),
                             res['n_correct_left'], res['n_correct_right']))
            continue
        if res['subject'] not in subjects:
            subjects[res['subject']] = len(subjects)
        data['neural'].append(res['neural'])
        data['input'].append(res['input'])
        data['output'].append(res['output'])
        data['subject_idx'].append(subjects[res['subject']])
        data['brain_region_idx'].append(res['brain_region_idx'])
        trial_metadata.append(res['trial_info'])
        session_info.append({
            'session_name': res['session_name'],
            'subject': res['subject'],
            'n_trials': int(res['n_trials']),
            'n_neurons': int(res['n_neurons']),
            'performance': round(float(res['performance']), 4),
            'n_photostim_trials': int(res['n_photostim_trials']),
            'n_trials_excluded_free_water': int(res['n_trials_free_water']),
            'n_trials_excluded_short_window': int(res['n_trials_short']),
            'n_trials_excluded_no_ephys': int(res['n_trials_no_ephys']),
            'n_units_dropped_silent': int(res['n_units_silent']),
            'tongue_y_percentiles': [round(v, 2) for v in res['tongue_percentiles']],
            'video_coverage': round(float(res['video_coverage']), 4),
            'tongue_lick_precision': round(float(res['tongue_lick_precision']), 3),
            'tongue_lick_recall': round(float(res['tongue_lick_recall']), 3),
        })

    data['subjects'] = list(subjects.keys())
    data['subject_idx'] = np.array(data['subject_idx'], dtype=np.int64)
    data['trial_metadata'] = trial_metadata

    # Keep only the regions that actually have neurons, and renumber.
    used = sorted(set(int(i) for idx in data['brain_region_idx'] for i in idx))
    remap = {old: new for new, old in enumerate(used)}
    data['brain_regions'] = [REGION_NAMES[i] for i in used]
    data['brain_region_idx'] = [
        np.array([remap[int(i)] for i in idx], dtype=np.int64)
        for idx in data['brain_region_idx']]

    n_trials = sum(len(s) for s in data['neural'])
    n_neurons = sum(len(r) for r in data['brain_region_idx'])
    data['metadata'] = {
        'dataset': 'Mesoscale Activity Map (MAP), DANDI:000363',
        'references': [
            'Chen S. et al., Brain-wide neural activity underlying '
            'memory-guided movement, Cell 2024',
            'Wang Z.A., Kurgyis B. et al., Brain-wide analysis reveals movement '
            'encoding structured across and within brain areas, '
            'Nat Neurosci 2025',
        ],
        'task_description':
            'Head-fixed mice performed an auditory delayed-response task. A '
            'sample tone (3 kHz or 12 kHz, three 150 ms pips) instructed the '
            'mouse to lick the left or the right port; after a delay epoch an '
            'auditory go cue (6 kHz, 0.1 s) released the response, and licking '
            'the instructed port within 1.5 s was rewarded with water. Licking '
            'during the sample or delay epoch (early lick) replayed the epoch. '
            'On ~25% of trials, ALM was photoinhibited (left, right or both '
            'hemispheres) during the late delay epoch. Neuropixels probes '
            'recorded spikes across the brain and a 300 Hz side-view camera '
            'tracked the tongue, jaw and nose with DeepLabCut. The decoder '
            'predicts, from binned firing rates plus the time since tone onset '
            'and the photostimulation state: the licking direction chosen by '
            'the mouse (left / right / no lick), the trial outcome (ignore / '
            'miss / hit), whether the trial had an early lick (no / yes), and '
            'the discretized vertical position of the tongue in each time bin '
            '(below the 40th percentile / 40th-60th percentile / above the '
            '60th percentile of the session, or not visible).',
        'time_bin_size': BIN_WIDTH * 1000.0,
        'temporal_alignment_event':
            'Onset of the auditory go cue that ends the delay epoch',
        'off_start': T_START,
        'off_end': T_END,
        'bin_edges_relative_to_go_cue': [float(x) for x in BIN_EDGES],
        'neural_units': 'spikes/s (spike count per 50 ms bin / 0.05 s)',
        'input_descriptions': {
            'time_from_tone_onset':
                'Seconds from the onset of the instruction tone (sample epoch) '
                'to the center of the time bin; negative before the tone. The '
                'tone precedes the go cue by 1.85 s (0.65 s sample + 1.2 s '
                'delay) in most sessions and by 0.95 s in the rest, and by '
                'more on trials whose epoch was replayed after an early lick.',
            'photostim_on':
                '1 in bins whose center falls inside the ALM photoinhibition '
                'interval of the trial (0.5 s including a 100 ms ramp down, '
                'always ending before the go cue), 0 otherwise.',
        },
        'output_descriptions': {
            'choice': 'Port of the first lick during the 1.5 s answer period.',
            'outcome': 'Trial outcome from the NWB trials table: ignore (no '
                       'response), miss (incorrect port), hit (correct port).',
            'early_lick': 'Whether the mouse licked during the sample or delay '
                          'epoch, from the NWB trials table.',
            'tongue_y_position': 'Vertical position of the tongue marker in the '
                                 'side-view video, averaged over the visible '
                                 'frames of the bin and discretized at the '
                                 '40th and 60th percentile of the session; '
                                 'class 3 when the tongue is not visible '
                                 '(DeepLabCut likelihood <= 0.5) anywhere in '
                                 'the bin.',
        },
        'trial_window_note':
            'Trials share the bin grid of the [-2.5, 1.5] s window but only the '
            'bins fully inside the trial interval are kept, because the source '
            'files contain no spikes outside it (an incorrect lick ends the '
            'trial early). T therefore varies between trials; the bin index of '
            'the first kept bin is in trial_metadata.',
        'n_sessions': len(data['neural']),
        'n_trials': int(n_trials),
        'n_neurons': int(n_neurons),
        'n_subjects': len(data['subjects']),
        'session_selection':
            f'behavioral performance > {MIN_PERFORMANCE:.0%} on control trials '
            f'and >= {MIN_CORRECT_PER_DIRECTION} correct trials in each '
            'direction (Chen et al. 2023 STAR Methods); in addition the '
            'side-view video has to cover >= 95% of the analysis window and the '
            'tongue tracking has to agree with the lick-port sensors, without '
            'which the tongue output cannot be built',
        'trial_selection': 'free-water trials excluded (auto-water trials '
                           'kept: water there follows the mouse\'s own lick); '
                           'photostimulation, '
                           'early-lick and no-response trials kept because they '
                           'are decoder inputs/outputs; trials without any '
                           'recorded spikes (ephys not yet started / already '
                           'stopped) and trials with fewer than '
                           f'{MIN_BINS_PER_TRIAL} observed bins also excluded',
        'neuron_selection': "units labeled 'good' by the quality-control "
                            'classifier of the data paper, with a CCF '
                            'annotation and at least one spike in the analysis '
                            'window',
        'rejected_sessions': rejected,
        'session_info': session_info,
    }

    print(f"\nKeeping {len(data['neural'])} sessions, {n_trials} trials, "
          f"{n_neurons} neurons, {len(data['subjects'])} subjects", flush=True)
    print(f"Rejected {len(rejected)} sessions", flush=True)

    with open(OUT_FILE, 'wb') as fh:
        pickle.dump(data, fh, protocol=4)
    print(f"Wrote {OUT_FILE} "
          f"({os.path.getsize(OUT_FILE) / 1e9:.2f} GB)", flush=True)


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 16)
