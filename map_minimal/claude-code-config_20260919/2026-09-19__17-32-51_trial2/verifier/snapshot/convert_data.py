#!/usr/bin/env python
"""
Convert the Mesoscale Activity Map (MAP) dataset (DANDI:000363, Chen et al. 2024,
"Brain-wide neural activity underlying memory-guided movement") into the decoder
format described in the task description.

The processing follows the reference papers and the analysis code in /app/code
(Wang, Kurgyis et al. 2025, "Brain-wide analysis reveals movement encoding
structured across and within brain areas") wherever the decoding task allows:

  * Units      : only units that passed the region-specific quality-control
                 classifier of the accompanying spike-sorting white paper
                 (NWB ``units/classification == 'good'``) and that have a CCF
                 annotation (the reference preprocessing keeps only units with
                 both ephys and histology).
  * Regions    : units are grouped into the coarse regions used by the reference
                 preprocessing (ALM, Orbital, OtherCortex, Olfactory,
                 Hippocampus, CorticalSubplate, Striatum, Pallidum, Thalamus,
                 Hypothalamus, Midbrain, Pons, Medulla, Cerebellum), separately
                 for the left and right hemisphere (`left ALM`, `right ALM`, ...)
                 as in ``preprocessing_DJ_2022Aug.process_one_sess``.
  * Sessions   : the session-selection criteria of the data paper are applied,
                 i.e. behavioural performance > 65 % and at least 50 correct
                 lick-left and 50 correct lick-right trials.
  * Trials     : auto-water and free-water trials are dropped (as in
                 ``get_regular_trial_mask``), because on those trials reward is
                 not contingent on the animal's choice.  Unlike the reference,
                 photostimulation, early-lick and no-response ("ignore") trials
                 are KEPT, because the decoding task asks for photostimulation
                 as a decoder input and for early lick / ignore as decoder
                 outputs.
  * Alignment  : everything is aligned to the go-cue onset, the alignment event
                 used throughout both papers (spike times in the reference
                 preprocessing are relative to the go cue).
  * Binning    : spikes are binned into non-overlapping 50 ms bins covering
                 [-2.5 s, +1.5 s) around the go cue and converted to firing
                 rates in Hz (the reference also stores rates, ``rate=True`` in
                 ``sliding_histogram``).

Decoder inputs (2 x 80 per trial):
    0. time from tone (sample-epoch) onset, in seconds
    1. photostimulation on/off

Decoder outputs (4 x 80 per trial):
    0. lick direction choice          (left / right / no lick)
    1. outcome                        (ignore / miss / hit)
    2. early lick                     (no / yes)
    3. discretised tongue y-position  (<40th pct / 40-60th pct / >60th pct /
                                       not visible)

Usage:  python convert_data.py [--out /app/converted_data.pkl] [--nproc 16]
"""

import argparse
import csv
import glob
import json
import os
import pickle
import sys
import time
from multiprocessing import Pool

import h5py
import numpy as np

# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
DATA_DIR = '/app/data'
OUT_FILE = '/app/converted_data.pkl'
CACHE_DIR = '/tmp/map_converted_sessions'
STRUCTURE_CSV = '/app/allen_ccf_structures.csv'   # Allen CCFv3 structure graph
STRUCTURE_URL = ('http://api.brain-map.org/api/v2/data/Structure/query.csv?'
                 'criteria=%5Bgraph_id$eq1%5D&num_rows=all')

# temporal alignment / binning
T_START = -2.5          # s relative to go cue
T_STOP = 1.5            # s relative to go cue
BIN_WIDTH = 0.05        # s
N_BINS = int(round((T_STOP - T_START) / BIN_WIDTH))       # 80
BIN_EDGES = T_START + BIN_WIDTH * np.arange(N_BINS + 1)
BIN_CENTERS = BIN_EDGES[:-1] + BIN_WIDTH / 2

# DeepLabCut likelihood above which the tongue is considered visible.  The
# likelihood distribution is strongly bimodal (>90 % of the frames are below
# 0.01 and essentially all of the remainder are above 0.99), so the exact value
# of the threshold is inconsequential.
TONGUE_LIKELIHOOD_THRESHOLD = 0.5

# session selection (data paper, STAR methods):
#   "We selected experimental sessions for analysis based on following criteria:
#    overall behavioral performance (> 65%), and at least 50 correct lick left
#    and lick right trials each."
MIN_PERFORMANCE = 0.65
MIN_CORRECT_PER_DIRECTION = 50

# CCF midline (as in preprocessing_DJ_2022Aug.helper_get_neuron_id_area)
ML_MIDLINE = 5700.0

INPUT_NAMES = ['time_from_tone_onset', 'photostim_on']
OUTPUT_NAMES = ['choice', 'outcome', 'early_lick', 'tongue_y_position']
OUTPUT_VALUES = [
    ['left', 'right', 'no lick'],
    ['ignore', 'miss', 'hit'],
    ['no', 'yes'],
    ['<40th pct', '40-60th pct', '>60th pct', 'not visible'],
]

# coarse regions, ordered as in the reference preprocessing
REGION_ORDER = ['ALM', 'Medulla', 'Midbrain', 'Striatum', 'Thalamus', 'Pons',
                'Cerebellum', 'Hypothalamus', 'Hippocampus', 'Orbital',
                'OtherCortex', 'Olfactory', 'CorticalSubplate', 'Pallidum']


# --------------------------------------------------------------------------- #
# CCF annotation -> coarse brain region
# --------------------------------------------------------------------------- #
def _download_structures(path):
    """Fetch the Allen CCFv3 structure graph if it is not cached next to us."""
    import urllib.request
    with urllib.request.urlopen(STRUCTURE_URL, timeout=120) as response:
        body = response.read()
    with open(path, 'wb') as fh:
        fh.write(body)


def build_region_map():
    """Map every Allen CCF structure name to one of the coarse regions.

    The mapping walks the ``structure_id_path`` of the Allen ontology, so every
    annotation (e.g. "Secondary motor area, layer 5") is assigned to the major
    division it descends from.  Cortex is split into ALM (somatomotor areas
    MOs/MOp), Orbital (ORB) and OtherCortex, matching the region groups of the
    reference preprocessing.

    ALM is defined in the source papers by an inactivation-map-derived voxel
    mask that is not distributed with the NWB files; the somatomotor
    annotations are used as the closest available proxy.  This recovers 7,885
    units against the 8,717 quoted by the data paper, and 98 % of the units
    with a somatomotor annotation sit on probes that targeted ALM.
    """
    if not os.path.exists(STRUCTURE_CSV):
        _download_structures(STRUCTURE_CSV)
    rows = list(csv.DictReader(open(STRUCTURE_CSV)))
    by_acronym = {r['acronym']: r for r in rows}

    # acronym of the major division -> coarse region name.  Ordered from coarse
    # to fine so that a more specific match (e.g. ORB inside Isocortex) wins.
    divisions = [('Isocortex', 'OtherCortex'), ('OLF', 'Olfactory'),
                 ('HPF', 'Hippocampus'), ('CTXsp', 'CorticalSubplate'),
                 ('STR', 'Striatum'), ('PAL', 'Pallidum'), ('TH', 'Thalamus'),
                 ('HY', 'Hypothalamus'), ('MB', 'Midbrain'), ('P', 'Pons'),
                 ('MY', 'Medulla'), ('CB', 'Cerebellum'),
                 ('ORB', 'Orbital'), ('MO', 'ALM')]
    division_ids = [(int(by_acronym[a]['id']), name) for a, name in divisions]

    region_map = {}
    for r in rows:
        path = {int(x) for x in r['structure_id_path'].strip('/').split('/') if x}
        label = None
        for sid, name in division_ids:
            if sid in path:
                label = name
        if label is not None:
            region_map[r['name'].strip()] = label
    return region_map


# --------------------------------------------------------------------------- #
# per-session helpers
# --------------------------------------------------------------------------- #
def observed_trials(f, trial_start):
    """Trials during which all quality-controlled units were being recorded.

    ``units/obs_intervals`` holds one interval per trial the unit was observed
    in.  In most sessions that is every trial, but in nine sessions the
    electrophysiology covers only a contiguous block of the behavioural
    session; the remaining trials contain no spikes at all and are dropped
    rather than entering the dataset as silent trials.
    """
    units = f['units']
    classification = units['classification'][:].astype(str)
    good_idx = np.where(classification == 'good')[0]
    if len(good_idx) == 0:
        return np.zeros(len(trial_start), dtype=bool)

    obs = units['obs_intervals'][:]
    stop = units['obs_intervals_index'][:]
    start = np.concatenate([[0], stop[:-1]])

    observed = np.ones(len(trial_start), dtype=bool)
    previous = None
    for u in good_idx:
        interval_starts = obs[start[u]:stop[u], 0]
        if previous is not None and len(previous) == len(interval_starts) \
                and np.array_equal(previous, interval_starts):
            continue                      # same coverage as the previous unit
        previous = interval_starts
        idx = np.clip(np.searchsorted(trial_start, interval_starts + 1e-6) - 1,
                      0, len(trial_start) - 1)
        matched = np.abs(trial_start[idx] - interval_starts) < 1e-3
        unit_observed = np.zeros(len(trial_start), dtype=bool)
        unit_observed[idx[matched]] = True
        observed &= unit_observed
    return observed


def _trial_table(f):
    """Read the trial table plus the go-cue and tone-onset times of a session."""
    tr = f['intervals/trials']
    out = {
        'start_time': tr['start_time'][:],
        'stop_time': tr['stop_time'][:],
        'outcome': tr['outcome'][:].astype(str),
        'early_lick': tr['early_lick'][:].astype(str),
        'instruction': tr['trial_instruction'][:].astype(str),
        'auto_water': tr['auto_water'][:].astype(int),
        'free_water': tr['free_water'][:].astype(int),
        'photostim_onset': tr['photostim_onset'][:].astype(str),
        'photostim_duration': tr['photostim_duration'][:].astype(str),
        'photostim_power': tr['photostim_power'][:].astype(str),
    }
    out['observed'] = observed_trials(f, out['start_time'])
    events = f['acquisition/BehavioralEvents']
    out['go_time'] = events['go_start_times']['timestamps'][:]
    sample_starts = events['sample_start_times']['timestamps'][:]
    # The tone is played during the sample epoch.  On early-lick trials the
    # sample/delay epoch is replayed, so the tone the animal finally responded
    # to is the last sample onset preceding the go cue.
    idx = np.searchsorted(sample_starts, out['go_time']) - 1
    out['tone_time'] = sample_starts[np.clip(idx, 0, len(sample_starts) - 1)]
    out['tone_valid'] = (idx >= 0) & (out['tone_time'] >= out['start_time'])
    return out


def trial_mask(tt):
    """Trials that enter the converted dataset.

    Auto-water and free-water trials are dropped, following
    ``get_regular_trial_mask`` of the reference code: on those trials reward is
    delivered independently of the animal's choice, so the outcome label does
    not describe the animal's decision.  Photostimulation, early-lick and
    no-response trials are kept even though the reference excludes them,
    because the decoding task asks for photostimulation as a decoder input and
    for early lick and 'ignore' as decoder outputs.  Trials outside the
    electrophysiological observation interval, and the (rare) trials whose tone
    onset cannot be located, are dropped as well.
    """
    return ((tt['auto_water'] == 0) & (tt['free_water'] == 0)
            & tt['observed'] & tt['tone_valid'])


def session_performance(tt):
    """Behavioural performance and correct-trial counts of a session.

    Performance is the fraction of correct responses on control (no
    photostimulation) trials, excluding early-lick trials as well as
    auto-water / free-water trials.  Trials on which the animal did not respond
    ('ignore') are not counted -- with them included the mean performance over
    the dandiset would be 68 %, whereas hits / (hits + misses) gives 81 %, which
    matches the 84 % (range 65-99 %) quoted by the data paper.
    """
    base = ((tt['photostim_power'] == 'N/A') & (tt['early_lick'] == 'no early')
            & trial_mask(tt))
    hit = tt['outcome'] == 'hit'
    miss = tt['outcome'] == 'miss'
    responded = (hit | miss) & base
    performance = hit[base].sum() / max(1, responded.sum())
    n_left = int((hit & base & (tt['instruction'] == 'left')).sum())
    n_right = int((hit & base & (tt['instruction'] == 'right')).sum())
    return float(performance), n_left, n_right


def scan_session(path):
    """Metadata needed to decide whether a session is included."""
    with h5py.File(path, 'r') as f:
        tt = _trial_table(f)
        performance, n_left, n_right = session_performance(tt)
        classification = f['units/classification'][:].astype(str)
        n_good = int((classification == 'good').sum())
        subject = str(f['general/subject/subject_id'][()].decode())
        session_start = str(f['session_start_time'][()].decode())
    return {
        'path': path,
        'subject': subject,
        'session_start_time': session_start,
        'n_trials_total': int(len(tt['start_time'])),
        'performance': performance,
        'n_correct_left': n_left,
        'n_correct_right': n_right,
        'n_good_units': n_good,
    }


def unit_regions(f, good, region_map):
    """Coarse hemisphere-qualified brain region of every good unit."""
    units = f['units']
    anno = units['anno_name'][:].astype(str)[good]

    # CCF coordinates come from the unit's peak electrode.  x is the
    # medio-lateral axis (left hemisphere >= 5700 um, as in the reference
    # preprocessing); where the electrode has no registered coordinate we fall
    # back on the hemisphere the probe was targeted at.
    electrodes = f['general/extracellular_ephys/electrodes']
    ml = electrodes['x'][:]
    location = electrodes['location'][:].astype(str)
    target_side = np.array([json.loads(s)['brain_regions'].split(' ')[0]
                            for s in location])
    electrode_index = units['electrodes'][:][good]
    ml_unit = ml[electrode_index]
    side = np.where(ml_unit >= ML_MIDLINE, 'left', 'right')
    unknown = np.isnan(ml_unit)
    side[unknown] = target_side[electrode_index][unknown]

    regions = np.array([region_map.get(a.strip(), None) for a in anno],
                       dtype=object)
    keep = np.array([r is not None for r in regions])
    names = np.array(['%s %s' % (s, r) if k else ''
                      for s, r, k in zip(side, regions, keep)], dtype=object)
    return names, keep


def bin_spikes(f, good, go_time):
    """Firing rates (Hz) of every good unit in every trial: (ntrials, nbins, nunits).

    Spike times in the NWB files only exist inside the trial intervals
    (``units/obs_intervals`` == [trial start, trial stop]).  On error trials the
    interval ends with the incorrect lick, so the last part of the analysis
    window is not observed and the corresponding bins are zero.  Such trials are
    kept because 'miss' is one of the values the decoder has to predict.
    """
    units = f['units']
    spike_times = units['spike_times'][:]
    stop = units['spike_times_index'][:]
    start = np.concatenate([[0], stop[:-1]])
    unit_ids = np.where(good)[0]

    n_trials = len(go_time)
    edges = go_time[:, None] + BIN_EDGES[None, :]          # (ntrials, nbins+1)
    flat_edges = edges.ravel()

    rates = np.zeros((n_trials, N_BINS, len(unit_ids)), dtype=np.float32)
    for i, u in enumerate(unit_ids):
        st = spike_times[start[u]:stop[u]]
        counts = np.searchsorted(st, flat_edges).reshape(n_trials, N_BINS + 1)
        rates[:, :, i] = np.diff(counts, axis=1) / BIN_WIDTH
    return rates


def tongue_y_per_bin(f, go_time):
    """Mean tongue y-position per time bin; NaN when the tongue is not visible.

    The side-view camera runs at ~300 Hz and the NWB timestamps are on the same
    clock as the trials, so frames are assigned to bins directly.  A bin counts
    as visible if at least one of its frames has a DeepLabCut likelihood above
    threshold, and its value is the mean y-position over those frames.  Bins
    without any video frame (< 0.3 % of all bins) are treated as not visible.
    """
    tracking = f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking']
    ts = tracking['timestamps'][:]
    data = tracking['data'][:]
    y = data[:, 1]
    visible = data[:, 2] > TONGUE_LIKELIHOOD_THRESHOLD

    # cumulative sums let us take the per-bin mean over visible frames with two
    # lookups per bin edge instead of a python loop over frames
    csum_y = np.concatenate([[0.0], np.cumsum(np.where(visible, y, 0.0))])
    csum_n = np.concatenate([[0], np.cumsum(visible.astype(np.int64))])

    edges = go_time[:, None] + BIN_EDGES[None, :]
    idx = np.searchsorted(ts, edges)                        # (ntrials, nbins+1)
    n_visible = np.diff(csum_n[idx], axis=1)
    sum_y = np.diff(csum_y[idx], axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        mean_y = np.where(n_visible > 0, sum_y / np.maximum(n_visible, 1), np.nan)
    return mean_y


def convert_session(args):
    """Convert one NWB file into per-trial neural / input / output arrays."""
    path, region_map, cache_file = args
    if cache_file is not None and os.path.exists(cache_file):
        with open(cache_file, 'rb') as fh:
            return pickle.load(fh)

    with h5py.File(path, 'r') as f:
        tt = _trial_table(f)
        classification = f['units/classification'][:].astype(str)
        good = classification == 'good'
        region_names, has_region = unit_regions(f, good, region_map)
        good_idx = np.where(good)[0][has_region]
        good = np.zeros(len(classification), dtype=bool)
        good[good_idx] = True
        region_names = region_names[has_region]

        # ---- trial selection -------------------------------------------- #
        trials = np.where(trial_mask(tt))[0]

        go_time = tt['go_time'][trials]
        rates = bin_spikes(f, good, go_time)
        tongue_y = tongue_y_per_bin(f, go_time)

        subject = str(f['general/subject/subject_id'][()].decode())
        session_start = str(f['session_start_time'][()].decode())
        performance, n_left, n_right = session_performance(tt)

    n_trials = len(trials)

    # ---- decoder inputs ------------------------------------------------- #
    tone_rel = tt['tone_time'][trials] - go_time                    # negative
    time_from_tone = BIN_CENTERS[None, :] - tone_rel[:, None]

    photostim = np.zeros((n_trials, N_BINS), dtype=np.float32)
    onset = tt['photostim_onset'][trials]
    duration = tt['photostim_duration'][trials]
    start_time = tt['start_time'][trials]
    for i in range(n_trials):
        if onset[i] == 'N/A':
            continue
        # photostim_onset is given relative to the start of the trial
        on = start_time[i] + float(onset[i]) - go_time[i]
        off = on + float(duration[i])
        photostim[i] = ((BIN_EDGES[:-1] < off) & (BIN_EDGES[1:] > on))

    # ---- decoder outputs ------------------------------------------------ #
    outcome_str = tt['outcome'][trials]
    instruction = tt['instruction'][trials]
    early = tt['early_lick'][trials]

    outcome = np.full(n_trials, -1, dtype=np.int64)
    outcome[outcome_str == 'ignore'] = 0
    outcome[outcome_str == 'miss'] = 1
    outcome[outcome_str == 'hit'] = 2

    # 'hit' means the animal licked the instructed direction, 'miss' that it
    # licked the other one, 'ignore' that it did not lick at all.  Checked
    # against the first lick after the go cue: they agree on > 99.8 % of trials.
    choice = np.full(n_trials, 2, dtype=np.int64)                   # no lick
    licked_left = (((outcome_str == 'hit') & (instruction == 'left'))
                   | ((outcome_str == 'miss') & (instruction == 'right')))
    licked_right = (((outcome_str == 'hit') & (instruction == 'right'))
                    | ((outcome_str == 'miss') & (instruction == 'left')))
    choice[licked_left] = 0
    choice[licked_right] = 1

    early_lick = (early == 'early').astype(np.int64)

    # tongue y-position, discretised with per-session percentiles
    finite = tongue_y[np.isfinite(tongue_y)]
    tongue = np.full(tongue_y.shape, 3, dtype=np.int64)             # not visible
    if finite.size >= 10:
        p40, p60 = np.percentile(finite, [40, 60])
        visible = np.isfinite(tongue_y)
        tongue[visible & (tongue_y <= p60)] = 1
        tongue[visible & (tongue_y < p40)] = 0
        tongue[visible & (tongue_y > p60)] = 2

    # ---- assemble ------------------------------------------------------- #
    neural = [np.ascontiguousarray(rates[i].T) for i in range(n_trials)]
    inputs = [np.stack([time_from_tone[i], photostim[i]]).astype(np.float32)
              for i in range(n_trials)]
    outputs = [np.stack([np.full(N_BINS, choice[i]),
                         np.full(N_BINS, outcome[i]),
                         np.full(N_BINS, early_lick[i]),
                         tongue[i]]).astype(np.int64)
               for i in range(n_trials)]

    result = {
        'neural': neural,
        'input': inputs,
        'output': outputs,
        'region_names': list(region_names),
        'subject': subject,
        'info': {
            'session_name': os.path.basename(path),
            'subject': subject,
            'session_start_time': session_start,
            'n_trials': int(n_trials),
            'n_trials_total': int(len(tt['start_time'])),
            'n_neurons': int(len(region_names)),
            'performance': performance,
            'n_correct_left': n_left,
            'n_correct_right': n_right,
            'n_photostim_trials': int((onset != 'N/A').sum()),
            'n_early_lick_trials': int(early_lick.sum()),
            'fraction_tongue_visible': float(np.mean(tongue < 3)),
        },
    }

    if cache_file is not None:
        tmp = cache_file + '.tmp%d' % os.getpid()
        with open(tmp, 'wb') as fh:
            pickle.dump(result, fh, protocol=4)
        os.replace(tmp, cache_file)
    return result


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', default=DATA_DIR)
    parser.add_argument('--out', default=OUT_FILE)
    parser.add_argument('--nproc', type=int, default=16)
    parser.add_argument('--cache-dir', default=CACHE_DIR)
    parser.add_argument('--no-cache', action='store_true')
    parser.add_argument('--limit', type=int, default=None,
                        help='only convert this many sessions (for testing)')
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.data_dir, '*', '*.nwb')))
    print('found %d nwb files' % len(files), flush=True)

    region_map = build_region_map()

    # ---- session selection ---------------------------------------------- #
    t0 = time.time()
    with Pool(args.nproc) as pool:
        scans = pool.map(scan_session, files)
    print('scanned sessions in %.1f s' % (time.time() - t0), flush=True)

    selected, rejected = [], []
    for s in scans:
        reasons = []
        if s['performance'] <= MIN_PERFORMANCE:
            reasons.append('performance %.3f' % s['performance'])
        if min(s['n_correct_left'], s['n_correct_right']) < MIN_CORRECT_PER_DIRECTION:
            reasons.append('correct trials %d/%d'
                           % (s['n_correct_left'], s['n_correct_right']))
        if s['n_good_units'] == 0:
            reasons.append('no good units')
        if reasons:
            s['rejected_because'] = '; '.join(reasons)
            rejected.append(s)
        else:
            selected.append(s)
    print('selected %d / %d sessions (%d rejected)'
          % (len(selected), len(files), len(rejected)), flush=True)
    for s in rejected:
        print('  rejected %s: %s' % (os.path.basename(s['path']),
                                     s['rejected_because']))

    if args.limit is not None:
        selected = selected[:args.limit]

    # ---- conversion ------------------------------------------------------ #
    cache_dir = None if args.no_cache else args.cache_dir
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
    jobs = []
    for s in selected:
        name = os.path.basename(s['path']).replace('.nwb', '.pkl')
        cache = None if cache_dir is None else os.path.join(cache_dir, name)
        jobs.append((s['path'], region_map, cache))

    t0 = time.time()
    sessions = []
    with Pool(args.nproc) as pool:
        for i, result in enumerate(pool.imap(convert_session, jobs)):
            sessions.append(result)
            print('[%3d/%3d] %s: %d trials, %d neurons (%.1f s)'
                  % (i + 1, len(jobs), result['info']['session_name'],
                     result['info']['n_trials'], result['info']['n_neurons'],
                     time.time() - t0), flush=True)

    # drop sessions that ended up without neurons or with fewer than 2 trials
    sessions = [s for s in sessions
                if s['info']['n_neurons'] > 0 and s['info']['n_trials'] >= 2]

    # ---- assemble the final dictionary ---------------------------------- #
    subjects = sorted({s['subject'] for s in sessions})
    subject_idx = np.array([subjects.index(s['subject']) for s in sessions],
                           dtype=np.int64)

    brain_regions = ['%s %s' % (side, region)
                     for region in REGION_ORDER for side in ('left', 'right')]
    present = {name for s in sessions for name in s['region_names']}
    brain_regions = [r for r in brain_regions if r in present]
    region_index = {name: i for i, name in enumerate(brain_regions)}
    brain_region_idx = [np.array([region_index[n] for n in s['region_names']],
                                 dtype=np.int64) for s in sessions]

    data = {
        'neural': [s['neural'] for s in sessions],
        'input': [s['input'] for s in sessions],
        'output': [s['output'] for s in sessions],
        'subjects': subjects,
        'subject_idx': subject_idx,
        'brain_regions': brain_regions,
        'brain_region_idx': brain_region_idx,
        'input_names': INPUT_NAMES,
        'output_names': OUTPUT_NAMES,
        'output_values': OUTPUT_VALUES,
        'metadata': {
            'task_description':
                'Mice performed an auditory delayed-response task (Chen et al. '
                '2024, DANDI:000363). One of two pure tones (3 or 12 kHz, three '
                '150 ms pips) played during the 0.65 s sample epoch instructed '
                'the animal to lick the left or the right lick port after a '
                '1.2 s delay epoch; an auditory go cue (6 kHz, 0.1 s) ended the '
                'delay and the animal reported its choice during a 1.5 s '
                'response epoch. Licking during the sample or delay epoch '
                '(early lick) triggered a replay of the epoch. On ~25 % of the '
                'trials ALM was photoinhibited (473 nm, 40 Hz sinusoidal, 5 mW '
                'per hemisphere, left, right or both hemispheres) during the '
                'delay epoch. The decoder predicts the lick direction, the '
                'trial outcome, whether the trial contained an early lick, and '
                'the discretised y-position of the tongue, from brain-wide '
                'Neuropixels recordings plus the time from tone onset and the '
                'photostimulation state.',
            'time_bin_size': BIN_WIDTH * 1000.0,
            'temporal_alignment_event':
                'onset of the auditory go cue that ends the delay epoch',
            'off_start': T_START,
            'off_end': T_STOP,
            'neural_units': 'firing rate in spikes/s (spike count / 50 ms bin)',
            'input_descriptions': [
                'time from the onset of the instruction tone (sample epoch) in '
                'seconds; negative before tone onset. On early-lick trials the '
                'delay epoch is replayed, so the last sample onset before the '
                'go cue is used.',
                '1 while ALM photoinhibition is on, 0 otherwise',
            ],
            'output_descriptions': [
                'lick direction reported by the animal after the go cue '
                '(derived from outcome and instructed direction)',
                'trial outcome: ignore (no response), miss (licked the wrong '
                'port), hit (licked the correct port)',
                'whether the animal licked during the sample or delay epoch',
                'tongue y-position (side-view camera, DeepLabCut) averaged over '
                'the frames of each 50 ms bin and discretised at the 40th and '
                '60th percentile of the visible values of that session; bins '
                'without a visible tongue (DeepLabCut likelihood <= %.2f) are '
                'labelled "not visible"' % TONGUE_LIKELIHOOD_THRESHOLD,
            ],
            'unit_selection':
                "units labelled 'good' by the region-specific quality-control "
                'classifiers of the spike-sorting white paper '
                '(doi:10.25378/janelia.24066108.v1) and carrying a CCF '
                'annotation',
            'session_selection':
                'behavioural performance > %d %% and at least %d correct '
                'lick-left and lick-right trials (data paper criteria); '
                '%d of %d sessions selected'
                % (MIN_PERFORMANCE * 100, MIN_CORRECT_PER_DIRECTION,
                   len(sessions), len(files)),
            'trial_selection':
                'auto-water and free-water trials excluded (reward not '
                'contingent on the choice) as well as trials outside the '
                'electrophysiological observation interval '
                '(units/obs_intervals); photostimulation, early-lick and '
                'no-response trials kept because they are decoder inputs or '
                'outputs',
            'known_limitation':
                'spike times in the NWB files are only stored inside the trial '
                'intervals, which end with the incorrect lick on error trials, '
                'so bins of the analysis window that fall outside the trial '
                'interval contain no spikes (~15 % of trials are affected at '
                'the end of the window, ~3 % at the beginning)',
            'tongue_tracking_note':
                'the fraction of bins with a visible tongue is 0.24 in the '
                'median session; the sessions listed here are far outside that '
                'range, i.e. their DeepLabCut tongue trace is unreliable. They '
                'are kept (their neural data and the other three outputs are '
                'unaffected) but their tongue labels should be treated with '
                'care: %s'
                % ', '.join('%s (%.3f visible)'
                            % (s['info']['session_name'],
                               s['info']['fraction_tongue_visible'])
                            for s in sessions
                            if not 0.02 < s['info']['fraction_tongue_visible'] < 0.9),
            'time_bin_centers': BIN_CENTERS.tolist(),
            'n_sessions': len(sessions),
            'n_trials': sum(len(s['neural']) for s in sessions),
            'n_neurons': sum(len(s['region_names']) for s in sessions),
            'data_source': 'DANDI:000363 (Mesoscale Activity Map Dataset), '
                           'Chen, Nguyen, Li & Svoboda 2023',
            'session_info': [s['info'] for s in sessions],
            'rejected_sessions': [
                {'session_name': os.path.basename(s['path']),
                 'subject': s['subject'],
                 'reason': s['rejected_because']} for s in rejected],
        },
    }

    n_trials = sum(len(s) for s in data['neural'])
    n_neurons = sum(len(idx) for idx in brain_region_idx)
    print('assembled %d sessions, %d subjects, %d trials, %d neurons'
          % (len(sessions), len(subjects), n_trials, n_neurons), flush=True)

    t0 = time.time()
    with open(args.out, 'wb') as fh:
        pickle.dump(data, fh, protocol=4)
    print('wrote %s (%.1f GB) in %.1f s'
          % (args.out, os.path.getsize(args.out) / 1e9, time.time() - t0))


if __name__ == '__main__':
    sys.exit(main())
