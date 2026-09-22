#!/usr/bin/env python3
"""Convert DANDI:000363 (Mesoscale Activity Map, Chen et al. 2023) NWB files into the
decoder-ready pickle format.

Usage
-----
    python -u convert_data.py <outpicklefile> [--full | --sample] [--show-processing]

Options
-------
    --full             process every session that passes curation (default)
    --sample           process only the first 2 curated sessions
    --show-processing  save a diagnostic figure of every processing step for up to
                       2 sessions as ``processing_<session_id>.png``
    --workers N        number of worker processes (default: min(24, cpu_count))

What it produces
----------------
Trials are aligned to **go-cue onset** and cut from -2.5 s to +1.5 s, binned into 80
non-overlapping 50 ms bins.

    neural[session][trial]  (n_neurons, 80) float32  firing rate in Hz
    input [session][trial]  (2, 80)        float32   [time since tone onset (s),
                                                      photostimulation on (0/1)]
    output[session][trial]  (4, 80)        int8      [lick direction choice,
                                                      outcome, early lick,
                                                      tongue y-position class]

See CONVERSION_NOTES.md for the provenance of every curation and processing decision.
"""

import argparse
import glob
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from pynwb import NWBHDF5IO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ccf_regions import REGION_NAMES, annotation_to_region  # noqa: E402

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------
DATA_DIR = '/app/data'

OFF_START = -2.5          # s, signed offset of the window start from the go cue
OFF_END = 1.5             # s, signed offset of the window end from the go cue
BIN_WIDTH = 0.05          # s
N_BINS = int(round((OFF_END - OFF_START) / BIN_WIDTH))          # 80
BIN_EDGES = OFF_START + BIN_WIDTH * np.arange(N_BINS + 1)       # (81,)
BIN_CENTERS = 0.5 * (BIN_EDGES[:-1] + BIN_EDGES[1:])            # (80,)

# DeepLabCut confidence above which the tongue counts as visible.  The likelihood is
# strongly bimodal (87% of frames < 1e-4, 11% > 0.999) so any cut in [0.01, 0.999]
# gives the same answer to within 0.5% of frames.
TONGUE_LIKELIHOOD_THRESHOLD = 0.9
TONGUE_LOW_PCT, TONGUE_HIGH_PCT = 40.0, 60.0

# Session curation thresholds, from the data paper STAR Methods:
# "overall behavioral performance (> 65%), and at least 50 correct lick left and lick
#  right trials each", where performance is the fraction correct of control (no
#  photostimulation) trials excluding early-lick trials.
MIN_PERFORMANCE = 0.65
MIN_CORRECT_PER_DIRECTION = 50

INPUT_NAMES = ['time_from_tone_onset_s', 'photostim_on']
OUTPUT_NAMES = ['lick_direction_choice', 'outcome', 'early_lick', 'tongue_y_position']
OUTPUT_VALUES = [
    ['left', 'right', 'no lick'],
    ['ignore', 'miss', 'hit'],
    ['no', 'yes'],
    ['y < 40th pct', 'y 40th-60th pct', 'y > 60th pct', 'not visible'],
]
CHOICE_LEFT, CHOICE_RIGHT, CHOICE_NOLICK = 0, 1, 2
OUTCOME_CODE = {'ignore': 0, 'miss': 1, 'hit': 2}
TONGUE_INVISIBLE = 3


# --------------------------------------------------------------------------------------
# Low-level helpers
# --------------------------------------------------------------------------------------
def window_edges(go_times):
    """Absolute times of the 81 bin edges of every trial.

    Args:
        go_times: (n_trials,) absolute go-cue times, seconds.

    Returns:
        (n_trials, 81) float64 array of absolute bin-edge times.
    """
    return go_times[:, None] + BIN_EDGES[None, :]


def bin_spike_times(spike_times, edges_abs):
    """Spike counts of one unit in every (trial, bin).

    `np.searchsorted` needs only `spike_times` to be sorted, so the trials' edge
    arrays can be passed in one flat call even where consecutive trial windows are
    not themselves monotone.

    Args:
        spike_times: (n_spikes,) sorted absolute spike times.
        edges_abs: (n_trials, 81) absolute bin edges.

    Returns:
        (n_trials, 80) int32 spike counts.
    """
    pos = np.searchsorted(spike_times, edges_abs.ravel()).reshape(edges_abs.shape)
    return np.diff(pos, axis=1).astype(np.int32)


def segment_sums(values, mask, edges_abs, sample_times):
    """Per-(trial, bin) count of samples, count of masked samples and sum of masked values.

    Used to reduce the 300 Hz tracking trace onto the 50 ms grid without a Python loop.

    Args:
        values: (n_frames,) per-frame value (tongue y).
        mask: (n_frames,) bool, frames that count (tongue visible).
        edges_abs: (n_trials, 81) absolute bin edges.
        sample_times: (n_frames,) sorted absolute frame times.

    Returns:
        n_frames_bin, n_masked_bin: (n_trials, 80) int arrays
        masked_sum: (n_trials, 80) float array
    """
    pos = np.searchsorted(sample_times, edges_abs.ravel()).reshape(edges_abs.shape)
    cum_n = np.concatenate([[0], np.cumsum(mask)])
    cum_v = np.concatenate([[0.0], np.cumsum(np.where(mask, values, 0.0))])
    n_frames_bin = np.diff(pos, axis=1)
    n_masked_bin = cum_n[pos[:, 1:]] - cum_n[pos[:, :-1]]
    masked_sum = cum_v[pos[:, 1:]] - cum_v[pos[:, :-1]]
    return n_frames_bin, n_masked_bin, masked_sum


def recorded_trials(units, good, start_time, stop_time, identifier):
    """Boolean mask of trials for which the electrophysiology was actually recorded.

    A unit's ``obs_intervals`` are exactly the ``[start_time, stop_time]`` intervals of
    the trials during which it was observed: the NWB release stores no spikes outside
    them.  In 8 of the 174 sessions the ephys covers only a contiguous *subset* of the
    behavioural trials (e.g. 160 of 480), and the remaining trials would otherwise be
    converted into all-zero firing rates.  Every good unit of a session shares the same
    observation block (verified across the whole release), so the first and last good
    unit are enough, and they are cross-checked against each other.

    Args:
        units: `nwb.units`.
        good: indices of the units to consider.
        start_time, stop_time: (n_trials,) trial bounds from the trials table.
        identifier: session name, for error messages.

    Returns:
        (n_trials,) bool array.
    """
    mask = np.zeros(len(start_time), dtype=bool)
    if len(good) == 0:
        return mask
    for unit in (good[0], good[-1]):
        obs = np.asarray(units['obs_intervals'][int(unit)])
        idx = np.searchsorted(start_time, obs[:, 0])
        idx = np.clip(idx, 0, len(start_time) - 1)
        ok = (np.abs(start_time[idx] - obs[:, 0]) < 1e-6) & \
             (np.abs(stop_time[idx] - obs[:, 1]) < 1e-6)
        assert ok.all(), (f'{identifier}: {int((~ok).sum())} observation intervals do '
                          'not correspond to a trial')
        unit_mask = np.zeros(len(start_time), dtype=bool)
        unit_mask[idx] = True
        if not mask.any():
            mask = unit_mask
        else:
            assert np.array_equal(mask, unit_mask), (
                f'{identifier}: units disagree about which trials were recorded')
    return mask


# --------------------------------------------------------------------------------------
# Reading one NWB file through pynwb
# --------------------------------------------------------------------------------------
def read_session(nwb):
    """Pull every variable this conversion needs out of an open NWBFile.

    Args:
        nwb: NWBFile returned by `NWBHDF5IO.read()`.

    Returns:
        dict of raw per-session arrays (no curation applied yet).
    """
    trials = nwb.intervals['trials']
    events = nwb.acquisition['BehavioralEvents'].time_series
    tracking = nwb.acquisition['BehavioralTimeSeries'].time_series

    out = {
        'identifier': nwb.identifier,
        'mouse': nwb.subject.description,
        'subject_id': str(nwb.subject.subject_id),
        'session_start_time': str(nwb.session_start_time),
        'start_time': np.asarray(trials['start_time'][:], dtype=np.float64),
        'stop_time': np.asarray(trials['stop_time'][:], dtype=np.float64),
        'instruction': np.asarray(trials['trial_instruction'][:]),
        'outcome': np.asarray(trials['outcome'][:]),
        'early_lick': np.asarray(trials['early_lick'][:]),
        'auto_water': np.asarray(trials['auto_water'][:], dtype=np.int64),
        'free_water': np.asarray(trials['free_water'][:], dtype=np.int64),
        'photostim_onset_str': np.asarray(trials['photostim_onset'][:]),
        'go': np.asarray(events['go_start_times'].timestamps[:], dtype=np.float64),
        'sample_start': np.asarray(events['sample_start_times'].timestamps[:], dtype=np.float64),
        'photostim_start': np.asarray(events['photostim_start_times'].timestamps[:], dtype=np.float64),
        'photostim_stop': np.asarray(events['photostim_stop_times'].timestamps[:], dtype=np.float64),
    }

    tongue = tracking['Camera0_side_TongueTracking']
    out['video_t'] = np.asarray(tongue.timestamps[:], dtype=np.float64)
    tongue_data = np.asarray(tongue.data[:], dtype=np.float64)
    out['tongue_y'] = tongue_data[:, 1]
    out['tongue_likelihood'] = tongue_data[:, 2]

    units = nwb.units
    if len(units) == 0:
        out['good_units'] = np.zeros(0, dtype=np.int64)
        out['anno'] = np.zeros(0, dtype=object)
        out['ccf'] = np.zeros((0, 3), dtype=np.float64)
        out['recorded'] = np.zeros(len(out['start_time']), dtype=bool)
        return out

    classification = np.asarray(units['classification'][:])
    good = np.flatnonzero(classification == 'good')
    out['recorded'] = recorded_trials(units, good, out['start_time'], out['stop_time'],
                                      out['identifier'])
    anno = np.asarray(units['anno_name'][:])[good]
    electrode_idx = np.asarray(units['electrodes'].target.data[:])[good]
    electrodes = nwb.electrodes
    ccf = np.stack([np.asarray(electrodes['x'].data[:]),
                    np.asarray(electrodes['y'].data[:]),
                    np.asarray(electrodes['z'].data[:])], axis=1)[electrode_idx]
    out['good_units'] = good
    out['anno'] = anno
    out['ccf'] = ccf
    return out


def session_performance(raw, keep):
    """Behavioural statistics used for session curation.

    Performance is the fraction of `hit` among control (no photostim), non-early-lick
    trials with a lick response, following the data paper's definition.

    Args:
        raw: output of `read_session`.
        keep: (n_trials,) bool mask of the trials that survive trial curation.

    Returns:
        (performance, n_correct_left, n_correct_right)
    """
    control = (raw['photostim_onset_str'] == 'N/A') & (raw['early_lick'] == 'no early') \
        & keep
    hit = raw['outcome'] == 'hit'
    miss = raw['outcome'] == 'miss'
    n_hit = int(np.sum(hit & control))
    n_miss = int(np.sum(miss & control))
    performance = n_hit / (n_hit + n_miss) if (n_hit + n_miss) else 0.0
    n_left = int(np.sum(hit & control & (raw['instruction'] == 'left')))
    n_right = int(np.sum(hit & control & (raw['instruction'] == 'right')))
    return performance, n_left, n_right


# --------------------------------------------------------------------------------------
# Per-session conversion
# --------------------------------------------------------------------------------------
def convert_session(path, collect_debug=False):
    """Convert one NWB session file.

    Args:
        path: path to the .nwb file.
        collect_debug: also return raw traces needed by `--show-processing`.

    Returns:
        dict with the per-session converted arrays, or None if the session fails
        curation.  Keys: 'neural' (n_trials, n_neurons, 80) float32,
        'input' (n_trials, 2, 80) float32, 'output' (n_trials, 4, 80) int8,
        plus identifiers and per-neuron metadata.
    """
    t0 = time.time()
    with NWBHDF5IO(path, 'r', load_namespaces=True) as io:
        nwb = io.read()
        raw = read_session(nwb)

        n_good = len(raw['good_units'])
        go = raw['go']
        n_trials_all = len(raw['start_time'])
        assert len(go) == n_trials_all, (
            f"{raw['identifier']}: {len(go)} go cues for {n_trials_all} trials")

        # ---- trial curation -------------------------------------------------------
        # Keep early-lick / no-response / photostim trials: they are the decoder's
        # outputs and input.  Drop auto-water and free-water trials (reference
        # `get_regular_trial_mask`), whose `outcome` is not a behavioural report.
        # Trials outside the electrophysiological recording carry no spikes at all.
        not_recorded = ~raw['recorded']
        keep = raw['recorded'] & (raw['auto_water'] == 0) & (raw['free_water'] == 0)

        # Drop trials with no video frame in the analysis window; every bin would
        # otherwise be labelled "tongue not visible" when the truth is "not measured".
        edges_all = window_edges(go)
        n_frames_all = (np.searchsorted(raw['video_t'], edges_all[:, -1])
                        - np.searchsorted(raw['video_t'], edges_all[:, 0]))
        no_video = keep & (n_frames_all == 0)
        keep &= n_frames_all > 0
        trial_idx = np.flatnonzero(keep)

        # ---- session curation -----------------------------------------------------
        # The data paper's criteria, evaluated on the trials that actually survive
        # curation, so that a session whose video or ephys covers only a handful of
        # trials also fails here (the data paper likewise excluded sessions with video
        # artifacts).
        performance, n_left, n_right = session_performance(raw, keep)
        reject = None
        if n_good == 0:
            reject = 'no good units'
        elif len(trial_idx) < 2:
            reject = f'only {len(trial_idx)} usable trials'
        elif performance <= MIN_PERFORMANCE:
            reject = f'performance {performance:.3f} <= {MIN_PERFORMANCE}'
        elif n_left < MIN_CORRECT_PER_DIRECTION or n_right < MIN_CORRECT_PER_DIRECTION:
            reject = f'correct trials L={n_left} R={n_right} < {MIN_CORRECT_PER_DIRECTION}'
        if reject is not None:
            return {'identifier': raw['identifier'], 'path': path, 'rejected': reject,
                    'performance': performance, 'n_good_units': n_good}

        n_trials = len(trial_idx)
        go = go[trial_idx]
        edges = window_edges(go)

        # ---- neural ---------------------------------------------------------------
        units = nwb.units
        spike_index = units['spike_times']
        counts = np.empty((n_good, n_trials, N_BINS), dtype=np.int32)
        for i, unit in enumerate(raw['good_units']):
            counts[i] = bin_spike_times(np.asarray(spike_index[int(unit)]), edges)
        neural = np.ascontiguousarray(
            counts.transpose(1, 0, 2).astype(np.float32) / BIN_WIDTH)

        # ---- inputs ---------------------------------------------------------------
        # Tone onset = last sample-epoch onset at or before the go cue.  Early licking
        # replays the sample/delay epoch, so a trial can contain several onsets; the
        # most recent one is the tone the animal is acting on.
        tone_idx = np.searchsorted(raw['sample_start'], go, side='right') - 1
        assert np.all(tone_idx >= 0), f"{raw['identifier']}: go cue before any tone"
        tone = raw['sample_start'][tone_idx]
        assert np.all(tone >= raw['start_time'][trial_idx] - 1e-9), (
            f"{raw['identifier']}: tone onset outside its trial")
        time_from_tone = (BIN_CENTERS[None, :] + (go - tone)[:, None]).astype(np.float32)

        photostim = np.zeros((n_trials, N_BINS), dtype=np.float32)
        if len(raw['photostim_start']):
            # A bin [t0, t1) is "on" if it intersects any [stim_start, stim_stop).
            stim_trial = np.searchsorted(raw['start_time'], raw['photostim_start'],
                                         side='right') - 1
            position = np.searchsorted(trial_idx, stim_trial)
            in_kept = (position < n_trials) & (trial_idx[np.clip(position, 0, n_trials - 1)]
                                               == stim_trial)
            for row, on, off in zip(position[in_kept], raw['photostim_start'][in_kept],
                                    raw['photostim_stop'][in_kept]):
                overlap = (edges[row, :-1] < off) & (edges[row, 1:] > on)
                photostim[row, overlap] = 1.0

        inputs = np.stack([time_from_tone, photostim], axis=1)  # (n_trials, 2, 80)

        # ---- outputs --------------------------------------------------------------
        instruction = raw['instruction'][trial_idx]
        outcome = raw['outcome'][trial_idx]
        early = raw['early_lick'][trial_idx]

        choice = np.where(instruction == 'left', CHOICE_LEFT, CHOICE_RIGHT)
        # An error trial means the mouse licked the port opposite the instruction.
        choice = np.where(outcome == 'miss', 1 - choice, choice)
        choice = np.where(outcome == 'ignore', CHOICE_NOLICK, choice)
        outcome_code = np.array([OUTCOME_CODE[o] for o in outcome], dtype=np.int8)
        early_code = (early == 'early').astype(np.int8)

        visible = raw['tongue_likelihood'] > TONGUE_LIKELIHOOD_THRESHOLD
        _, n_visible_bin, y_sum_bin = segment_sums(
            raw['tongue_y'], visible, edges, raw['video_t'])
        bin_visible = n_visible_bin > 0
        y_mean = np.where(bin_visible, y_sum_bin / np.maximum(n_visible_bin, 1), np.nan)

        tongue_class = np.full((n_trials, N_BINS), TONGUE_INVISIBLE, dtype=np.int8)
        percentiles = (np.nan, np.nan)
        if bin_visible.any():
            visible_values = y_mean[bin_visible]
            p_low, p_high = np.percentile(visible_values, [TONGUE_LOW_PCT, TONGUE_HIGH_PCT])
            percentiles = (float(p_low), float(p_high))
            cls = np.where(y_mean < p_low, 0, np.where(y_mean > p_high, 2, 1))
            tongue_class[bin_visible] = cls[bin_visible].astype(np.int8)

        outputs = np.empty((n_trials, 4, N_BINS), dtype=np.int8)
        outputs[:, 0, :] = choice[:, None]
        outputs[:, 1, :] = outcome_code[:, None]
        outputs[:, 2, :] = early_code[:, None]
        outputs[:, 3, :] = tongue_class

        # ---- per-neuron metadata --------------------------------------------------
        regions = np.array([annotation_to_region(a, ccf[2], ccf[0])
                            for a, ccf in zip(raw['anno'], raw['ccf'])])
        region_idx = np.array([REGION_NAMES.index(r) if r in REGION_NAMES else -1
                               for r in regions], dtype=np.int64)
        # Hemisphere from the CCF ML coordinate, midline 5700 um (reference code
        # `helper_get_neuron_id_area`).
        hemisphere = np.where(raw['ccf'][:, 0] >= 5700.0, 'left', 'right')

        result = {
            'identifier': raw['identifier'],
            'path': path,
            'mouse': raw['mouse'],
            'subject_id': raw['subject_id'],
            'session_start_time': raw['session_start_time'],
            'rejected': None,
            'performance': performance,
            'n_correct_left': n_left,
            'n_correct_right': n_right,
            'neural': neural,
            'input': inputs,
            'output': outputs,
            'brain_region_idx': region_idx,
            'neuron_index_in_file': raw['good_units'].astype(np.int32),
            'neuron_annotation': raw['anno'],
            'neuron_hemisphere': hemisphere,
            'neuron_ccf': raw['ccf'].astype(np.float32),
            'trial_index': trial_idx,
            'n_trials_in_file': n_trials_all,
            'n_trials_dropped': int(n_trials_all - n_trials),
            'n_trials_not_recorded': int(not_recorded.sum()),
            'n_trials_no_video': int(no_video.sum()),
            'n_trials_auto_or_free_water': int(np.sum(
                raw['recorded'] & ((raw['auto_water'] != 0) | (raw['free_water'] != 0)))),
            'tongue_percentiles': percentiles,
            'frac_bins_tongue_visible': float(bin_visible.mean()),
            'n_silent_trials': int(np.sum(counts.sum(axis=(0, 2)) == 0)),
            'frac_bins_no_spikes_in_trial': float(np.mean(
                (edges[:, 1:] > raw['stop_time'][trial_idx][:, None])
                | (edges[:, :-1] < raw['start_time'][trial_idx][:, None]))),
            'elapsed': time.time() - t0,
        }
        if collect_debug:
            result['debug'] = {
                'go': go, 'tone': tone, 'edges': edges,
                'start_time': raw['start_time'][trial_idx],
                'stop_time': raw['stop_time'][trial_idx],
                'video_t': raw['video_t'], 'tongue_y': raw['tongue_y'],
                'tongue_likelihood': raw['tongue_likelihood'],
                'y_mean': y_mean, 'bin_visible': bin_visible,
                'photostim_start': raw['photostim_start'],
                'photostim_stop': raw['photostim_stop'],
                'spike_times': [np.asarray(spike_index[int(u)])
                                for u in raw['good_units'][:60]],
            }
        return result


# --------------------------------------------------------------------------------------
# Diagnostic plots
# --------------------------------------------------------------------------------------
def plot_processing(result, outfile):
    """Save a figure showing every processing step for one session."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    dbg = result['debug']
    neural, inputs, outputs = result['neural'], result['input'], result['output']
    n_trials = neural.shape[0]
    go, tone, edges = dbg['go'], dbg['tone'], dbg['edges']

    # Prefer a trial that has photostimulation and some visible tongue, for a busy plot.
    score = inputs[:, 1, :].sum(1) * 100 + (outputs[:, 3, :] != TONGUE_INVISIBLE).sum(1)
    trial = int(np.argmax(score))

    fig, axes = plt.subplots(4, 2, figsize=(20, 22))

    # (1) raw spike raster + binned rates for the example trial
    ax = axes[0, 0]
    for i, st in enumerate(dbg['spike_times']):
        rel = st[(st >= go[trial] + OFF_START) & (st < go[trial] + OFF_END)] - go[trial]
        ax.plot(rel, np.full(len(rel), i), '|', color='k', ms=3, mew=0.5)
    ax.axvline(0, color='r', lw=2, label='go cue')
    ax.axvline(tone[trial] - go[trial], color='g', lw=2, label='tone onset')
    on = inputs[trial, 1] > 0
    if on.any():
        ax.axvspan(BIN_EDGES[np.flatnonzero(on)[0]], BIN_EDGES[np.flatnonzero(on)[-1] + 1],
                   color='c', alpha=0.3, label='photostim')
    ax.axvline(dbg['start_time'][trial] - go[trial], color='b', ls='--', label='trial start')
    ax.axvline(dbg['stop_time'][trial] - go[trial], color='b', ls=':', label='trial stop')
    ax.set(xlim=(OFF_START, OFF_END), xlabel='time from go cue (s)', ylabel='neuron',
           title=f'(1) raw spike times, trial {trial} (first 60 good units)')
    ax.legend(loc='upper left', fontsize=8)

    # (2) the same trial after binning
    ax = axes[0, 1]
    show = neural[trial][:60]
    ax.imshow(show, aspect='auto', origin='lower', interpolation='nearest',
              extent=[BIN_EDGES[0], BIN_EDGES[-1], -0.5, show.shape[0] - 0.5],
              cmap='magma')
    ax.axvline(0, color='r', lw=2)
    ax.axvline(tone[trial] - go[trial], color='g', lw=2)
    ax.set(xlabel='time from go cue (s)', ylabel='neuron',
           title='(2) binned firing rate, 50 ms bins (same neurons, same trial)')

    # (3) population PSTH split by choice -- alignment check
    ax = axes[1, 0]
    choice = outputs[:, 0, 0]
    for code, name, color in [(0, 'lick left', 'b'), (1, 'lick right', 'r'),
                              (2, 'no lick', 'gray')]:
        m = choice == code
        if m.sum() > 1:
            ax.plot(BIN_CENTERS, neural[m].mean(axis=(0, 1)), color=color,
                    label=f'{name} (n={int(m.sum())})')
    ax.axvline(0, color='r', lw=1, ls='--')
    ax.axvline(np.median(tone - go), color='g', lw=1, ls='--')
    ax.set(xlabel='time from go cue (s)', ylabel='mean firing rate (Hz)',
           title='(3) population PSTH by choice (go cue red, median tone green)')
    ax.legend(fontsize=8)

    # (4) inputs for the example trial
    ax = axes[1, 1]
    ax.plot(BIN_CENTERS, inputs[trial, 0], 'g.-', label=INPUT_NAMES[0])
    ax.plot(BIN_CENTERS, inputs[trial, 1], 'c.-', label=INPUT_NAMES[1])
    ax.axvline(0, color='r', lw=1, ls='--')
    ax.axhline(0, color='k', lw=0.5)
    ax.plot(tone[trial] - go[trial], 0, 'g*', ms=18, label='true tone onset (input=0)')
    ax.set(xlabel='time from go cue (s)', ylabel='value',
           title=f'(4) decoder inputs, trial {trial}')
    ax.legend(fontsize=8)

    # (5) raw tongue trace vs binned class for the example trial
    ax = axes[2, 0]
    t = dbg['video_t']
    sel = (t >= edges[trial, 0]) & (t < edges[trial, -1])
    rel = t[sel] - go[trial]
    vis = dbg['tongue_likelihood'][sel] > TONGUE_LIKELIHOOD_THRESHOLD
    ax.plot(rel[~vis], dbg['tongue_y'][sel][~vis], '.', color='0.8', ms=2,
            label='raw y (occluded)')
    ax.plot(rel[vis], dbg['tongue_y'][sel][vis], '.', color='k', ms=3, label='raw y (visible)')
    ax.plot(BIN_CENTERS, dbg['y_mean'][trial], 'o-', color='tab:orange', ms=4,
            label='per-bin mean y (visible)')
    p_low, p_high = result['tongue_percentiles']
    ax.axhline(p_low, color='b', ls='--', label='40th pct (session)')
    ax.axhline(p_high, color='m', ls='--', label='60th pct (session)')
    ax2 = ax.twinx()
    ax2.step(BIN_CENTERS, outputs[trial, 3], where='mid', color='tab:red', lw=2,
             label='tongue class')
    ax2.set_ylabel('tongue class (0/1/2 = low/mid/high, 3 = not visible)', color='tab:red')
    ax2.set_ylim(-0.2, 3.2)
    ax.axvline(0, color='r', lw=1, ls='--')
    ax.set(xlabel='time from go cue (s)', ylabel='tongue y (px)',
           title=f'(5) tongue tracking -> discretised output, trial {trial}')
    ax.legend(fontsize=7, loc='upper left')

    # (6) session histogram of per-bin visible y with the percentile cuts
    ax = axes[2, 1]
    vals = dbg['y_mean'][dbg['bin_visible']]
    ax.hist(vals, bins=80, color='0.6')
    ax.axvline(p_low, color='b', ls='--', label=f'40th pct = {p_low:.1f}')
    ax.axvline(p_high, color='m', ls='--', label=f'60th pct = {p_high:.1f}')
    frac = [float(np.mean(outputs[:, 3] == c)) for c in range(4)]
    ax.set(xlabel='per-bin mean tongue y (px, visible bins)', ylabel='# bins',
           title='(6) session discretisation; class fractions = '
                 + ', '.join(f'{f:.3f}' for f in frac))
    ax.legend(fontsize=8)

    # (7) tongue class over all trials, sorted by choice
    ax = axes[3, 0]
    order = np.argsort(choice, kind='stable')
    im = ax.imshow(outputs[order, 3], aspect='auto', origin='lower',
                   interpolation='nearest', cmap='viridis',
                   extent=[BIN_EDGES[0], BIN_EDGES[-1], -0.5, n_trials - 0.5])
    ax.axvline(0, color='r', lw=1)
    boundaries = np.flatnonzero(np.diff(choice[order])) + 0.5
    for b in boundaries:
        ax.axhline(b, color='w', lw=1)
    fig.colorbar(im, ax=ax, label='tongue class')
    ax.set(xlabel='time from go cue (s)', ylabel='trial (sorted by choice)',
           title='(7) tongue y class, all trials (white lines separate choices)')

    # (8) outputs and photostim across trials
    ax = axes[3, 1]
    ax.plot(outputs[:, 0, 0], '.', label='choice')
    ax.plot(outputs[:, 1, 0] + 0.15, '.', label='outcome')
    ax.plot(outputs[:, 2, 0] + 0.3, '.', label='early lick')
    ax.plot(inputs[:, 1, :].max(axis=1) + 0.45, '.', label='any photostim')
    ax.set(xlabel='trial', ylabel='value (offset for clarity)',
           title='(8) per-trial outputs and photostim across the session')
    ax.legend(fontsize=8)

    fig.suptitle(f"{result['identifier']}  —  {n_trials} trials, "
                 f"{neural.shape[1]} good units, performance {result['performance']:.3f}",
                 fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    fig.savefig(outfile, dpi=110)
    plt.close(fig)
    print(f'  wrote {outfile}')


# --------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------
def _worker(args):
    path, collect_debug = args
    try:
        return convert_session(path, collect_debug=collect_debug)
    except Exception as exc:  # keep one bad file from killing the whole run
        import traceback
        traceback.print_exc()
        return {'identifier': os.path.basename(path), 'path': path,
                'rejected': f'exception: {exc!r}'}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('outfile', help='output pickle path')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--full', action='store_true', default=True,
                       help='process all sessions (default)')
    group.add_argument('--sample', action='store_true',
                       help='process only the first 2 curated sessions')
    parser.add_argument('--show-processing', action='store_true',
                        help='save processing_<session>.png for up to 2 sessions')
    parser.add_argument('--workers', type=int,
                        default=min(24, os.cpu_count() or 1))
    args = parser.parse_args()

    t_start = time.time()
    files = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', '*.nwb')))
    print(f'found {len(files)} NWB files in {DATA_DIR}')

    if args.sample:
        # Curation is cheap compared with conversion, but not free; in sample mode just
        # walk the list until 2 sessions have been converted.
        files = files[:8]
        print(f'--sample: trying the first {len(files)} files')

    # Debug traces are only kept for the first few files, since they hold the raw
    # spike/video arrays; enough of them that at least two curated sessions have them.
    n_debug = 8 if args.show_processing else 0
    jobs = [(p, i < n_debug) for i, p in enumerate(files)]

    results = []
    if args.workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for res in pool.map(_worker, jobs):
                results.append(res)
                _report(res)
    else:
        for job in jobs:
            res = _worker(job)
            results.append(res)
            _report(res)

    kept = [r for r in results if r.get('rejected') is None]
    rejected = [r for r in results if r.get('rejected') is not None]
    if args.sample:
        kept = kept[:2]

    print(f'\ncurated {len(kept)} sessions, rejected {len(rejected)}')
    for r in rejected:
        print(f'  REJECTED {r["identifier"]}: {r["rejected"]}')

    if args.show_processing:
        for r in [x for x in kept if 'debug' in x][:2]:
            plot_processing(r, f'processing_{r["identifier"]}.png')

    data = assemble(kept)
    print(f'\nwriting {args.outfile} ...')
    t_write = time.time()
    with open(args.outfile, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    size = os.path.getsize(args.outfile)
    print(f'wrote {size / 1e9:.2f} GB in {time.time() - t_write:.1f} s')
    print(f'total elapsed {time.time() - t_start:.1f} s')
    summarise(data, kept)


def _report(res):
    if res.get('rejected') is not None:
        print(f'  - {res["identifier"]}: rejected ({res["rejected"]})')
    else:
        print(f'  + {res["identifier"]}: {res["neural"].shape[0]} trials x '
              f'{res["neural"].shape[1]} units  ({res["elapsed"]:.1f}s)')
    sys.stdout.flush()


def assemble(sessions):
    """Build the final dictionary from the per-session results."""
    sessions = sorted(sessions, key=lambda r: r['identifier'])
    mice = sorted({r['mouse'] for r in sessions})
    data = {
        'neural': [], 'input': [], 'output': [],
        'subjects': mice,
        'subject_idx': np.array([mice.index(r['mouse']) for r in sessions], dtype=np.int64),
        'brain_regions': list(REGION_NAMES),
        'brain_region_idx': [r['brain_region_idx'] for r in sessions],
        'input_names': list(INPUT_NAMES),
        'output_names': list(OUTPUT_NAMES),
        'output_values': [list(v) for v in OUTPUT_VALUES],
    }
    for r in sessions:
        # Split the (n_trials, ...) blocks into the per-trial list the format wants and
        # drop the block immediately: keeping both alive doubles peak memory on the
        # full dataset (~10 GB of neural data).
        data['neural'].append([np.ascontiguousarray(x) for x in r['neural']])
        r['neural_shape'] = r['neural'].shape
        r['neural'] = None
        data['input'].append([np.ascontiguousarray(x) for x in r['input']])
        r['input'] = None
        data['output'].append([np.ascontiguousarray(x) for x in r['output']])
        r['output'] = None

    data['metadata'] = {
        'dataset': 'DANDI:000363 Mesoscale Activity Map Dataset (Chen et al. 2023), '
                   'version 0.230822.0128',
        'task_description':
            'Head-fixed mice perform an auditory delayed-response task. A series of pure '
            'tones during the sample epoch (3 kHz instructs lick-right, 12 kHz instructs '
            'lick-left) is followed by a 1.2 s delay in which licking must be withheld, '
            'then an auditory go cue opens a 1.5 s answer epoch in which the mouse licks '
            'the left or right port; a correct lick is rewarded with water. Licking during '
            'the sample or delay epoch is an "early lick" and replays the epoch. On ~25% of '
            'randomly interleaved trials, ALM was photoinhibited (473 nm, 40 Hz sinusoid, '
            '5 mW/hemisphere) for 0.5 s during the delay, always ending at or before the go '
            'cue. Decoded outputs are the lick direction choice (left/right/no lick), the '
            'trial outcome (ignore/miss/hit), whether the trial had an early lick (no/yes), '
            'and the discretised vertical position of the tongue in the side-view video.',
        'time_bin_size': BIN_WIDTH * 1000.0,
        'temporal_alignment_event': 'go cue onset (auditory go cue, BehavioralEvents/'
                                    'go_start_times; one per trial)',
        'off_start': OFF_START,
        'off_end': OFF_END,
        'bin_centers_s': BIN_CENTERS.astype(np.float32),
        'neural_units': 'firing rate in spikes/s (spike count per 50 ms bin / 0.05 s)',
        'input_descriptions': [
            'seconds elapsed since the onset of the instruction tone (the last sample-epoch '
            'onset at or before the go cue; negative before the tone). Continuous.',
            '1 while ALM photostimulation is on in that 50 ms bin, else 0.',
        ],
        'output_descriptions': [
            'lick direction the mouse reported: 0 left, 1 right, 2 no lick. Derived from '
            'trial_instruction and outcome (hit -> instructed side, miss -> opposite side, '
            'ignore -> no lick). Constant within a trial.',
            'trial outcome: 0 ignore (no response), 1 miss (incorrect lick), 2 hit (correct '
            'lick). Constant within a trial.',
            'early lick during the sample or delay epoch: 0 no, 1 yes. Constant within a trial.',
            'vertical position of the tongue in the side-view video, per 50 ms bin: 3 if the '
            'tongue is not visible (DeepLabCut likelihood <= '
            f'{TONGUE_LIKELIHOOD_THRESHOLD} in every frame of the bin), otherwise 0/1/2 for '
            'below the 40th percentile / between the 40th and 60th percentile / above the '
            '60th percentile of the visible per-bin y values of that session.',
        ],
        'neuron_curation': "units['classification'] == 'good' (per-region quality-control "
                           'classifier of Chen, Liu et al. 2023); all such units carry a CCF '
                           'annotation.',
        'trial_curation': 'trials outside the electrophysiological recording (units\' '
                          'obs_intervals) removed; auto-water and free-water trials '
                          'removed (reference get_regular_trial_mask); trials without '
                          'video coverage of the analysis window removed. Early-lick, '
                          'no-response and photostimulation trials are kept because they '
                          'are decoder outputs/inputs.',
        'session_curation': f'>=1 good unit, control non-early-lick performance > '
                            f'{MIN_PERFORMANCE}, and >= {MIN_CORRECT_PER_DIRECTION} correct '
                            'lick-left and lick-right control trials (data paper STAR '
                            'Methods), all evaluated on the trials that survive trial '
                            'curation, which also removes sessions whose video or ephys '
                            'covers too few trials.',
        'known_limitation': 'The NWB release stores spikes only within each trial\'s '
                            '[start_time, stop_time] interval (0 spikes fall outside; trials '
                            'cover 66% of session wall-clock). 3.1% of trials begin less than '
                            '2.5 s before the go cue and 15.6% end less than 1.5 s after it - '
                            'error (miss) trials are terminated a few hundred ms after the go '
                            'cue by the time-out - so those bins contain no spikes and have a '
                            'firing rate of 0.',
        'tongue_likelihood_threshold': TONGUE_LIKELIHOOD_THRESHOLD,
        'video_frame_rate_hz': 1.0 / 0.0034,
        'session_info': [
            {'identifier': r['identifier'], 'mouse': r['mouse'],
             'subject_id': r['subject_id'], 'session_start_time': r['session_start_time'],
             'file': os.path.basename(r['path']),
             'n_trials': int(r['neural_shape'][0]),
             'n_trials_in_file': int(r['n_trials_in_file']),
             'n_trials_dropped': int(r['n_trials_dropped']),
             'n_trials_not_recorded': int(r['n_trials_not_recorded']),
             'n_trials_no_video': int(r['n_trials_no_video']),
             'n_trials_auto_or_free_water': int(r['n_trials_auto_or_free_water']),
             'n_neurons': int(r['neural_shape'][1]),
             'n_silent_trials': int(r['n_silent_trials']),
             'frac_bins_outside_trial': float(r['frac_bins_no_spikes_in_trial']),
             'performance': float(r['performance']),
             'n_correct_left': int(r['n_correct_left']),
             'n_correct_right': int(r['n_correct_right']),
             'trial_index_in_file': r['trial_index'].astype(np.int32),
             'neuron_index_in_file': r['neuron_index_in_file'],
             'tongue_y_percentiles_40_60': r['tongue_percentiles'],
             'frac_bins_tongue_visible': r['frac_bins_tongue_visible']}
            for r in sessions],
        'neuron_annotation': [r['neuron_annotation'] for r in sessions],
        'neuron_hemisphere': [r['neuron_hemisphere'] for r in sessions],
        'neuron_ccf_coordinates': [r['neuron_ccf'] for r in sessions],
        'neuron_ccf_coordinate_axes': 'columns are CCF ML (x), DV (y), AP (z) in um; '
                                      'midline ML = 5700 um',
    }
    return data


def summarise(data, sessions):
    """Print the headline statistics of the converted dataset."""
    n_sessions = len(data['neural'])
    n_trials = sum(len(s) for s in data['neural'])
    n_neurons = sum(s[0].shape[0] for s in data['neural'] if s)
    print('\n================ converted dataset ================')
    print(f'sessions           : {n_sessions}')
    print(f'subjects           : {len(data["subjects"])}')
    print(f'trials             : {n_trials} '
          f'(mean {n_trials / max(n_sessions, 1):.1f} per session)')
    print(f'neurons            : {n_neurons} '
          f'(mean {n_neurons / max(n_sessions, 1):.1f} per session)')
    print(f'timepoints / trial : {data["neural"][0][0].shape[1]}')
    counts = np.zeros(len(REGION_NAMES), dtype=np.int64)
    for idx in data['brain_region_idx']:
        counts += np.bincount(idx, minlength=len(REGION_NAMES))
    print('neurons per region :')
    for name, c in zip(REGION_NAMES, counts):
        print(f'    {name:18s} {c}')
    for d, name in enumerate(data['input_names']):
        lo = min(float(t[d].min()) for s in data['input'] for t in s)
        hi = max(float(t[d].max()) for s in data['input'] for t in s)
        print(f'input  {d} {name:26s} range [{lo:.3f}, {hi:.3f}]')
    for d, name in enumerate(data['output_names']):
        n_cat = len(data['output_values'][d])
        hist = np.zeros(n_cat, dtype=np.int64)
        for s in data['output']:
            for t in s:
                hist += np.bincount(t[d], minlength=n_cat)
        frac = hist / hist.sum()
        print(f'output {d} {name:26s} fractions '
              + ', '.join(f'{v}={f:.4f}' for v, f in zip(data['output_values'][d], frac)))
    perf = np.array([s['performance'] for s in sessions])
    print(f'performance        : mean {perf.mean():.3f}, range '
          f'[{perf.min():.3f}, {perf.max():.3f}]')


if __name__ == '__main__':
    main()
