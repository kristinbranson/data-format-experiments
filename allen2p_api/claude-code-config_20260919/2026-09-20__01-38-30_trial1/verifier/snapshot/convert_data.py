"""
Convert the Allen Brain Observatory Visual Behavior 2P dataset into the decoder format.

Usage:
    python -u convert_data.py <outpicklefile> [--full | --sample] [--show-processing]

The conversion, in one sentence: for every *active behaviour* ophys session we take the go and catch
trials, cut a fixed [-2.25, +3.0] s window around the stimulus change, bin every data stream into
250 ms bins on the shared session clock, and store the binned calcium-event matrix together with five
categorical output variables (image identity, image change, running-speed quintile, pupil-diameter
quintile, trial outcome).

All data are read through the AllenSDK `VisualBehaviorOphysProjectCache` API; no NWB file is opened
directly.
"""

import argparse
import os
import pickle
import re
import sys
import time
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

from allensdk.brain_observatory.behavior.behavior_project_cache import (  # noqa: E402
    VisualBehaviorOphysProjectCache,
)

# --------------------------------------------------------------------------------------
# Conversion parameters (see CONVERSION_NOTES.md Step 5 for the justification of each)
# --------------------------------------------------------------------------------------
CACHE_DIR = '/app/data'

BIN_SIZE = 0.25          # s, 1/3 of the 750 ms flash cycle, >= 2 ophys frames even at 11 Hz
OFF_START = -2.25        # s relative to the change: 3 flash cycles before the change
OFF_END = 3.00           # s relative to the change: the change flash + 3 flash cycles after
NBINS = int(round((OFF_END - OFF_START) / BIN_SIZE))   # 21
FLASH_CYCLE = 0.75       # s, image (250 ms) + gray (500 ms)
CHANGE_WINDOW = 0.5      # s after the change that image_change marks as 1 (see notes Step 5/12)

NEURAL_SIGNAL = 'dff'  # 'dff' (default, see CONVERSION_NOTES Step 5), 'events' or 'filtered_events'
NQUANTILES = 5            # five equal percentile bins for running speed and pupil diameter
MAX_SENSOR_GAP = 2.0      # s; behavioural bins with no sample are interpolated across gaps up to
                          # this length, trials with longer sensor dropouts are dropped

STIM_BLOCK = 'change_detection_behavior'

SIGNAL_DESCRIPTION = {
    'dff': ('dff: mean per 250 ms bin of the neuropil-corrected, demixed, detrended dF/F trace produced '
            'by the Allen pipeline (allensdk BehaviorOphysExperiment.dff_traces); ophys frames are '
            'assigned to bins by ophys_timestamps'),
    'events': ('events: mean per 250 ms bin of the FastLZero detected calcium event magnitude trace '
               '(allensdk BehaviorOphysExperiment.events); ophys frames are assigned to bins by '
               'ophys_timestamps'),
    'filtered_events': ('filtered_events: mean per 250 ms bin of the half-normal-filtered detected '
                        'calcium event trace (allensdk BehaviorOphysExperiment.events)'),
}

OUTPUT_NAMES = ['image_identity', 'image_change', 'running_speed_bin',
                'pupil_diameter_bin', 'trial_outcome']
OUTCOMES = ['hit', 'miss', 'false_alarm', 'correct_reject']
QUANTILE_NAMES = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']


# --------------------------------------------------------------------------------------
# Loading / session selection
# --------------------------------------------------------------------------------------
def get_cache():
    """Open the local AllenSDK cache (offline)."""
    return VisualBehaviorOphysProjectCache.from_s3_cache(cache_dir=CACHE_DIR)


def local_experiment_ids():
    """ophys_experiment_ids whose NWB file is present in the local cache."""
    d = os.path.join(CACHE_DIR, 'visual-behavior-ophys-1.1.0', 'behavior_ophys_experiments')
    return sorted(int(re.findall(r'(\d+)', f)[0]) for f in os.listdir(d))


def select_sessions(cache):
    """Select the ophys sessions to convert.

    - only experiments whose NWB file is available locally
    - only *active behaviour* sessions: passive-viewing sessions have the lick spout retracted, so
      there is no choice and trial outcome would be degenerate. Those are not the Visual Behavior task.

    Returns a DataFrame of the selected experiments (indexed by ophys_experiment_id) and the list of
    ophys_session_ids, sorted.
    """
    et = cache.get_ophys_experiment_table()
    ids = local_experiment_ids()
    sel = et[et.index.isin(ids) & (~et.passive)].copy()
    sel = sel.sort_values(['ophys_session_id', 'ophys_experiment_id'])
    session_ids = sorted(sel.ophys_session_id.unique().tolist())
    return sel, session_ids


# --------------------------------------------------------------------------------------
# Binning helpers
# --------------------------------------------------------------------------------------
def bin_edges_for_trials(change_times):
    """(n_trials, NBINS+1) array of absolute bin edges, and the flattened, strictly increasing version."""
    offsets = OFF_START + BIN_SIZE * np.arange(NBINS + 1)
    edges = change_times[:, None] + offsets[None, :]
    return edges


def bin_mean(values, timestamps, edges_flat):
    """Mean of `values` over each bin, for bins defined by consecutive entries of `edges_flat`.

    values:      (n_signals, T) or (T,)
    timestamps:  (T,) increasing sample times of `values`
    edges_flat:  (K,) increasing bin edges (all trials concatenated)

    Returns (n_signals, K-1) [or (K-1,)] of means, and (K-1,) sample counts. Segments with no sample
    return NaN; the caller decides what to do with them.
    """
    one_d = values.ndim == 1
    v = values[None, :] if one_d else values
    idx = np.searchsorted(timestamps, edges_flat, side='left')
    counts = np.diff(idx)
    # np.add.reduceat's last segment always runs to the end of the array, so the array is truncated
    # at the final bin edge; that makes the last segment stop at that edge like every other one.
    stop = max(int(idx[-1]), 1)
    starts = np.minimum(idx[:-1], stop - 1)   # only ever clamps indices of empty (NaN'd) segments
    sums = np.add.reduceat(v[:, :stop], starts, axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        means = sums / np.maximum(counts, 1)[None, :]
    empty = counts <= 0
    if empty.any():
        means[:, empty] = np.nan
    return (means[0] if one_d else means), counts


def keep_index(n_trials):
    """Indices, within the flattened segment array, of the NBINS real bins of each trial.

    The flattened edge array has n_trials*(NBINS+1) entries, so reduceat produces
    n_trials*(NBINS+1)-1 segments; per trial the first NBINS are real bins and the last is the gap to
    the next trial (dropped).
    """
    base = np.arange(n_trials)[:, None] * (NBINS + 1)
    return (base + np.arange(NBINS)[None, :]).ravel()


def interpolate_nans(x):
    """Linear interpolation over NaNs in a 1-D array; edges are held constant. Returns None if all NaN."""
    x = np.asarray(x, dtype=np.float64)
    good = np.isfinite(x)
    if not good.any():
        return None
    if good.all():
        return x
    idx = np.arange(len(x))
    return np.interp(idx, idx[good], x[good])


def fill_short_gaps(binned, centers, max_gap):
    """Fill empty (NaN) bins of a behavioural stream by linear interpolation in time.

    binned, centers: (n_trials, NBINS); centers are absolute bin-centre times.
    Returns (filled, trial_ok). A trial is marked not-ok if any of its empty bins is farther than
    `max_gap` seconds from a bin that does have data (i.e. the sensor was down for a long stretch,
    where interpolation would be fiction rather than a fix).
    """
    flat = binned.ravel().astype(np.float64)
    ct = centers.ravel()
    good = np.isfinite(flat)
    trial_ok = np.ones(binned.shape[0], dtype=bool)
    if good.all():
        return binned, trial_ok
    if not good.any():
        return binned, np.zeros(binned.shape[0], dtype=bool)
    filled = np.interp(ct, ct[good], flat[good])
    prev_t = np.maximum.accumulate(np.where(good, ct, -np.inf))
    next_t = np.minimum.accumulate(np.where(good, ct, np.inf)[::-1])[::-1]
    ok = good | ((ct - prev_t <= max_gap) & (next_t - ct <= max_gap))
    trial_ok = ok.reshape(binned.shape).all(axis=1)
    return filled.reshape(binned.shape), trial_ok


def within_range(binned, raw, tol=1e-6):
    """A bin mean can never leave the range of the samples it averages — cheap alignment/binning check."""
    return (np.nanmin(binned) >= np.nanmin(raw) - tol) and (np.nanmax(binned) <= np.nanmax(raw) + tol)


def quantile_bins(values, nq=NQUANTILES):
    """Discretize into `nq` equal-percentile bins. Returns (labels, edges)."""
    edges = np.percentile(values, np.linspace(0, 100, nq + 1)[1:-1])
    labels = np.searchsorted(edges, values, side='right').astype(np.int64)
    return np.clip(labels, 0, nq - 1), edges


# --------------------------------------------------------------------------------------
# Per-session conversion
# --------------------------------------------------------------------------------------
def select_trials(trials):
    """Go and catch trials, excluding aborted and auto-rewarded ones (decoder-task specification).

    `go`/`catch` are already mutually exclusive with `aborted`/`auto_rewarded` in this dataset, but the
    explicit mask documents the intent and protects against sessions where that does not hold.
    """
    m = (trials.go | trials.catch) & (~trials.aborted) & (~trials.auto_rewarded)
    m &= trials.change_time.notna()
    return trials[m]


def outcome_labels(sel):
    """Integer trial-outcome label (index into OUTCOMES) for each selected trial."""
    lab = np.full(len(sel), -1, dtype=np.int64)
    for i, name in enumerate(OUTCOMES):
        lab[sel[name].values.astype(bool)] = i
    return lab


def convert_session(args):
    """Convert one ophys session. Returns a dict (or dict with 'skip' explaining why it was dropped)."""
    session_id, oeids, show_processing, signal_name, neural_lag = args
    t_start = time.time()
    cache = get_cache()
    timing = {}

    t0 = time.time()
    datasets = [cache.get_behavior_ophys_experiment(int(o)) for o in oeids]
    timing['load'] = time.time() - t0
    ds0 = datasets[0]

    # ---- trials -----------------------------------------------------------------------
    trials = ds0.trials
    sel = select_trials(trials)
    if len(sel) < 2:
        return {'session_id': session_id, 'skip': f'only {len(sel)} usable trials'}
    change_times = sel.change_time.values.astype(np.float64)
    edges = bin_edges_for_trials(change_times)
    edges_flat = edges.ravel()
    if np.any(np.diff(edges_flat) <= 0):
        return {'session_id': session_id, 'skip': 'overlapping trial windows'}
    centers_flat = (edges_flat[:-1] + edges_flat[1:]) / 2.0
    keep = keep_index(len(sel))
    centers = centers_flat[keep].reshape(len(sel), NBINS)

    # ---- neural ------------------------------------------------------------------------
    t0 = time.time()
    neural_blocks, region_per_neuron, cell_ids = [], [], []
    for ds in datasets:
        ts = np.asarray(ds.ophys_timestamps, dtype=np.float64)
        if signal_name == 'dff':
            traces = np.vstack(ds.dff_traces.dff.values).astype(np.float64)
            index = ds.dff_traces.index.values
        else:
            traces = np.vstack(ds.events[signal_name].values).astype(np.float64)
            index = ds.events.index.values
        assert traces.shape[1] == len(ts), 'neural trace length != ophys_timestamps'
        means, counts = bin_mean(traces, ts - neural_lag, edges_flat)
        block = means[:, keep].reshape(traces.shape[0], len(sel), NBINS)
        if not within_range(block, traces):
            return {'session_id': session_id, 'skip': 'binned neural outside raw trace range'}
        neural_blocks.append(block)
        region_per_neuron += [ds.metadata['targeted_structure']] * traces.shape[0]
        cell_ids += list(index)
    neural = np.concatenate(neural_blocks, axis=0)   # (n_neurons, n_trials, NBINS)
    timing['neural'] = time.time() - t0
    # a trial whose window is not fully covered by 2p frames (never seen in this dataset) is dropped
    trial_ok = np.isfinite(neural).all(axis=(0, 2))

    # ---- running speed ------------------------------------------------------------------
    t0 = time.time()
    rs = ds0.running_speed
    run_ts = rs.timestamps.values.astype(np.float64)
    run_v = interpolate_nans(rs.speed.values)
    if run_v is None:
        return {'session_id': session_id, 'skip': 'no running speed'}
    run_binned, _ = bin_mean(run_v, run_ts, edges_flat)
    run_binned = run_binned[keep].reshape(len(sel), NBINS)
    run_binned, ok = fill_short_gaps(run_binned, centers, MAX_SENSOR_GAP)
    trial_ok &= ok
    if not within_range(run_binned, run_v):
        return {'session_id': session_id, 'skip': 'binned running speed outside raw range'}

    # ---- pupil diameter ------------------------------------------------------------------
    et = ds0.eye_tracking
    if len(et) == 0:
        return {'session_id': session_id, 'skip': 'no eye tracking'}
    eye_ts = et.timestamps.values.astype(np.float64)
    # whitepaper: pupil_area is the area of a circle whose diameter is the ellipse major axis, and is
    # NaN on likely_blink frames -> diameter in the same pixel units, blinks interpolated over.
    pupil_area = et.pupil_area.values.astype(np.float64)
    pupil_diam_raw = 2.0 * np.sqrt(pupil_area / np.pi)
    pupil_diam = interpolate_nans(pupil_diam_raw)
    if pupil_diam is None:
        return {'session_id': session_id, 'skip': 'pupil all NaN'}
    pupil_binned, _ = bin_mean(pupil_diam, eye_ts, edges_flat)
    pupil_binned = pupil_binned[keep].reshape(len(sel), NBINS)
    pupil_binned, ok = fill_short_gaps(pupil_binned, centers, MAX_SENSOR_GAP)
    trial_ok &= ok
    if not within_range(pupil_binned, pupil_diam):
        return {'session_id': session_id, 'skip': 'binned pupil outside raw range'}
    timing['behavior'] = time.time() - t0

    # ---- stimulus: image identity per flash interval ----------------------------------------
    t0 = time.time()
    sp = ds0.stimulus_presentations
    spa = sp[sp.stimulus_block_name == STIM_BLOCK]
    flash_start = spa.start_time.values.astype(np.float64)
    flash_name = spa.image_name.values.astype(object)
    # the flash interval is [start_time, start_time + 750 ms): assign every bin centre to the last
    # flash that started before it (the paper's "image presentation interval" convention).
    j = np.searchsorted(flash_start, centers_flat, side='right') - 1
    valid = (j >= 0) & (j < len(flash_start))
    j_clipped = np.clip(j, 0, len(flash_start) - 1)
    within = valid & (centers_flat - flash_start[j_clipped] < FLASH_CYCLE + 0.05)
    image_name_flat = np.where(within, flash_name[j_clipped], 'none')
    image_names = image_name_flat[keep].reshape(len(sel), NBINS)
    trial_ok &= ~(image_names == 'none').any(axis=1)

    # ---- image change --------------------------------------------------------------------
    is_change = sel.is_change.values.astype(bool)
    rel_centers = centers - change_times[:, None]
    image_change = ((rel_centers >= 0) & (rel_centers < CHANGE_WINDOW)
                    & is_change[:, None]).astype(np.int64)
    timing['stimulus'] = time.time() - t0

    # ---- drop trials with missing data before anything session-level is computed ---------------
    n_dropped = int((~trial_ok).sum())
    if trial_ok.sum() < 2:
        return {'session_id': session_id,
                'skip': f'only {int(trial_ok.sum())} trials with complete data'}
    if n_dropped:
        sel = sel[trial_ok]
        change_times = change_times[trial_ok]
        edges = edges[trial_ok]
        centers = centers[trial_ok]
        neural = neural[:, trial_ok, :]
        run_binned = run_binned[trial_ok]
        pupil_binned = pupil_binned[trial_ok]
        image_names = image_names[trial_ok]
        image_change = image_change[trial_ok]

    # ---- discretize running / pupil (per session, over exactly the exported timepoints) --------
    run_bin, run_edges = quantile_bins(run_binned.ravel())
    pupil_bin, pupil_edges = quantile_bins(pupil_binned.ravel())
    run_bin = run_bin.reshape(len(sel), NBINS)
    pupil_bin = pupil_bin.reshape(len(sel), NBINS)

    # ---- trial outcome ---------------------------------------------------------------------
    outcome = outcome_labels(sel)
    if (outcome < 0).any():
        return {'session_id': session_id, 'skip': 'trial without an outcome label'}

    meta = ds0.metadata
    result = {
        'session_id': session_id,
        'oeids': [int(o) for o in oeids],
        'skip': None,
        'neural': neural.astype(np.float32),                 # (n_neurons, n_trials, NBINS)
        'image_names': image_names,                          # (n_trials, NBINS) str
        'image_change': image_change,
        'run_bin': run_bin,
        'pupil_bin': pupil_bin,
        'outcome': outcome,
        'region_per_neuron': region_per_neuron,
        'cell_ids': cell_ids,
        'mouse_id': str(meta['mouse_id']),
        'session_type': meta['session_type'],
        'cre_line': meta['cre_line'],
        'equipment_name': meta['equipment_name'],
        'project_code': meta['project_code'],
        'imaging_depths': [int(d.metadata['imaging_depth']) for d in datasets],
        'frame_rate': float(meta['ophys_frame_rate']),
        'change_times': change_times,
        'trial_ids': sel.index.values,
        'ntrials': len(sel),
        'n_dropped_trials': n_dropped,
        'run_edges': run_edges,
        'pupil_edges': pupil_edges,
        'run_binned': run_binned,
        'pupil_binned': pupil_binned,
        'timing': timing,
        'total_time': time.time() - t_start,
    }

    if show_processing:
        try:
            plot_processing(result, datasets, sel, spa, run_ts, run_v, eye_ts,
                            pupil_diam_raw, pupil_diam, edges, signal_name)
        except Exception as e:      # plotting must never break the conversion
            print(f'  [warn] plotting failed for session {session_id}: {e!r}')
    return result


# --------------------------------------------------------------------------------------
# Visualisation of every processing step (--show-processing)
# --------------------------------------------------------------------------------------
def plot_processing(res, datasets, sel, spa, run_ts, run_v, eye_ts, pupil_raw, pupil_interp,
                    edges, signal_name):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    sid = res['session_id']
    # pick a go trial (a change is visible) that is not the first one
    go_idx = int(np.nonzero(sel.is_change.values.astype(bool))[0][min(5, len(sel) - 1)])
    ct = res['change_times'][go_idx]
    t0, t1 = ct + OFF_START, ct + OFF_END
    tr_edges = edges[go_idx]
    centers = (tr_edges[:-1] + tr_edges[1:]) / 2

    fig, ax = plt.subplots(7, 1, figsize=(16, 22), sharex=False)

    # 1: raw neural traces + bin edges
    ds = datasets[0]
    ts = np.asarray(ds.ophys_timestamps)
    if signal_name == 'dff':
        traces = np.vstack(ds.dff_traces.dff.values)
    else:
        traces = np.vstack(ds.events[signal_name].values)
    m = (ts >= t0 - 1) & (ts <= t1 + 1)
    nshow = min(8, traces.shape[0])
    for i in range(nshow):
        ax[0].plot(ts[m], traces[i, m] + i * (np.nanmax(traces[:nshow, m]) + 1e-9), lw=0.8)
    for e in tr_edges:
        ax[0].axvline(e, color='k', lw=0.3, alpha=0.4)
    ax[0].axvline(ct, color='r', lw=2)
    for _, f in spa[(spa.end_time >= t0 - 1) & (spa.start_time <= t1 + 1)].iterrows():
        ax[0].axvspan(f.start_time, f.end_time,
                      color='none' if f.image_name == 'omitted' else 'C0', alpha=0.15)
    ax[0].set_title(f'session {sid} trial {go_idx}: raw {signal_name} ({nshow} cells), '
                    f'250 ms bin edges (black), change time (red), flashes (blue)')

    # 2: binned neural for the same trial
    im = ax[1].imshow(res['neural'][:, go_idx, :], aspect='auto', interpolation='nearest',
                      extent=[OFF_START, OFF_END, res['neural'].shape[0], 0])
    ax[1].axvline(0, color='r')
    ax[1].set_title('binned neural (neurons x 21 bins), x = time re: change (s)')
    plt.colorbar(im, ax=ax[1])

    # 3: running speed raw vs binned
    m = (run_ts >= t0 - 1) & (run_ts <= t1 + 1)
    ax[2].plot(run_ts[m] - ct, run_v[m], color='gray', label='raw (60 Hz)')
    ax[2].step(centers - ct, res['run_binned'][go_idx], where='mid', color='C1', label='binned')
    for e in tr_edges:
        ax[2].axvline(e - ct, color='k', lw=0.3, alpha=0.4)
    ax[2].axvline(0, color='r')
    ax[2].legend(); ax[2].set_title('running speed (cm/s)')

    # 4: running quantile labels + edges
    ax[3].step(centers - ct, res['run_bin'][go_idx], where='mid', color='C1')
    ax[3].set_yticks(range(NQUANTILES))
    ax[3].set_title('running speed quintile label; session edges = '
                    + ', '.join(f'{e:.2f}' for e in res['run_edges']))

    # 5: pupil raw (with blink NaNs), interpolated, binned
    m = (eye_ts >= t0 - 1) & (eye_ts <= t1 + 1)
    ax[4].plot(eye_ts[m] - ct, pupil_interp[m], color='C2', label='blink-interpolated')
    ax[4].plot(eye_ts[m] - ct, pupil_raw[m], 'k.', ms=3, label='raw (NaN on blinks)')
    ax[4].step(centers - ct, res['pupil_binned'][go_idx], where='mid', color='C3', label='binned')
    ax[4].axvline(0, color='r'); ax[4].legend()
    ax[4].set_title('pupil diameter (px)')

    # 6: image identity + image change
    names = res['image_names'][go_idx]
    uniq = sorted(set(res['image_names'].ravel().tolist()))
    ax[5].step(centers - ct, [uniq.index(n) for n in names], where='mid', color='C4')
    ax[5].set_yticks(range(len(uniq))); ax[5].set_yticklabels(uniq)
    for _, f in spa[(spa.end_time >= t0) & (spa.start_time <= t1)].iterrows():
        ax[5].axvspan(f.start_time - ct, f.end_time - ct, color='C0', alpha=0.2)
        ax[5].text(f.start_time - ct, len(uniq) - 0.5, str(f.image_name), fontsize=6, rotation=90)
    ax[5].axvline(0, color='r')
    ax[5].set_title('image identity per bin (line) vs. actual flashes (shaded, labelled)')

    ax[6].step(centers - ct, res['image_change'][go_idx], where='mid', color='C5', label='image_change')
    ax[6].axvline(CHANGE_WINDOW, color='g', ls='--')
    ax[6].axvline(0, color='r')
    ax[6].set_yticks([0, 1])
    ax[6].set_title(f"image_change; trial outcome = {OUTCOMES[res['outcome'][go_idx]]}, "
                    f"is_change = {bool(sel.is_change.values[go_idx])}")

    # session-level distributions on a second figure
    fig.tight_layout()
    fig.savefig(f'processing_{sid}.png', dpi=110)
    plt.close(fig)

    fig2, ax2 = plt.subplots(2, 3, figsize=(18, 9))
    ax2[0, 0].hist(res['run_binned'].ravel(), bins=60)
    for e in res['run_edges']:
        ax2[0, 0].axvline(e, color='r')
    ax2[0, 0].set_title('binned running speed + quintile edges')
    ax2[0, 1].hist(res['pupil_binned'].ravel(), bins=60)
    for e in res['pupil_edges']:
        ax2[0, 1].axvline(e, color='r')
    ax2[0, 1].set_title('binned pupil diameter + quintile edges')
    ax2[0, 2].bar(range(NQUANTILES), np.bincount(res['run_bin'].ravel(), minlength=NQUANTILES))
    ax2[0, 2].bar(np.arange(NQUANTILES) + 0.3,
                  np.bincount(res['pupil_bin'].ravel(), minlength=NQUANTILES), width=0.3)
    ax2[0, 2].set_title('quintile occupancy (run, pupil) — should be flat')
    u, c = np.unique(res['image_names'], return_counts=True)
    ax2[1, 0].bar(range(len(u)), c); ax2[1, 0].set_xticks(range(len(u)))
    ax2[1, 0].set_xticklabels(u, rotation=90); ax2[1, 0].set_title('image identity counts')
    ax2[1, 1].plot(res['image_change'].mean(axis=0))
    ax2[1, 1].set_title('P(image_change) per bin (should be flat 0 except bins 9-11)')
    oc = np.bincount(res['outcome'], minlength=len(OUTCOMES))
    ax2[1, 2].bar(range(len(OUTCOMES)), oc); ax2[1, 2].set_xticks(range(len(OUTCOMES)))
    ax2[1, 2].set_xticklabels(OUTCOMES, rotation=45); ax2[1, 2].set_title('trial outcome counts')
    fig2.suptitle(f'session {sid} summary')
    fig2.tight_layout()
    fig2.savefig(f'processing_{sid}_summary.png', dpi=110)
    plt.close(fig2)
    print(f'  wrote processing_{sid}.png and processing_{sid}_summary.png')


# --------------------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------------------
def assemble(results, signal_name):
    """Assemble per-session results into the target dictionary."""
    results = [r for r in results if r.get('skip') is None]
    results.sort(key=lambda r: r['session_id'])

    image_values = sorted({n for r in results for n in np.unique(r['image_names']) if n != 'omitted'})
    image_values = image_values + ['omitted']
    image_to_idx = {n: i for i, n in enumerate(image_values)}

    regions = sorted({reg for r in results for reg in r['region_per_neuron']})
    region_to_idx = {reg: i for i, reg in enumerate(regions)}
    subjects = sorted({r['mouse_id'] for r in results})
    subject_to_idx = {s: i for i, s in enumerate(subjects)}

    data = {
        'neural': [], 'input': [], 'output': [],
        'subjects': subjects,
        'subject_idx': np.array([subject_to_idx[r['mouse_id']] for r in results], dtype=np.int64),
        'brain_regions': regions,
        'brain_region_idx': [np.array([region_to_idx[x] for x in r['region_per_neuron']],
                                      dtype=np.int64) for r in results],
        'input_names': [],
        'output_names': OUTPUT_NAMES,
        'output_values': [image_values, ['no_change', 'change'],
                          QUANTILE_NAMES, QUANTILE_NAMES, OUTCOMES],
    }

    session_info = []
    for r in results:
        n_neurons, n_trials, _ = r['neural'].shape
        img = np.vectorize(image_to_idx.get)(r['image_names']).astype(np.int64)
        neural_trials, input_trials, output_trials = [], [], []
        for t in range(n_trials):
            neural_trials.append(np.ascontiguousarray(r['neural'][:, t, :]))
            input_trials.append(np.zeros((0, NBINS), dtype=np.float32))
            out = np.stack([
                img[t],
                r['image_change'][t],
                r['run_bin'][t],
                r['pupil_bin'][t],
                np.full(NBINS, r['outcome'][t], dtype=np.int64),
            ], axis=0)
            output_trials.append(out)
        data['neural'].append(neural_trials)
        data['input'].append(input_trials)
        data['output'].append(output_trials)
        session_info.append({
            'ophys_session_id': int(r['session_id']),
            'ophys_experiment_ids': r['oeids'],
            'mouse_id': r['mouse_id'],
            'session_type': r['session_type'],
            'cre_line': r['cre_line'],
            'project_code': r['project_code'],
            'equipment_name': r['equipment_name'],
            'imaging_depths': r['imaging_depths'],
            'ophys_frame_rate': r['frame_rate'],
            'n_neurons': int(n_neurons),
            'n_trials': int(n_trials),
            'brain_regions': sorted(set(r['region_per_neuron'])),
            'running_speed_quintile_edges_cm_s': [float(x) for x in r['run_edges']],
            'pupil_diameter_quintile_edges_px': [float(x) for x in r['pupil_edges']],
        })

    data['metadata'] = {
        'task_description': (
            'Allen Brain Observatory Visual Behavior 2P: head-fixed mice perform a go/no-go visual '
            'change-detection task while 2-photon calcium imaging is performed in visual cortex. '
            'Natural images are flashed for 250 ms every 750 ms (500 ms gray inter-stimulus interval, '
            '5% of flashes omitted); mice earn water by licking within 150-750 ms of a change in image '
            'identity. Trials are go (image change) or catch (sham change); aborted (premature lick) and '
            'auto-rewarded trials are excluded. The decoder predicts, from the binned dF/F activity '
            'alone (no decoder inputs): (0) image identity of the flash interval, (1) image change, '
            '(2) running-speed quintile, (3) pupil-diameter quintile, all time-varying, and '
            '(4) trial outcome (hit/miss/false alarm/correct reject), static per trial.'),
        'time_bin_size': BIN_SIZE * 1000.0,
        'temporal_alignment_event': (
            'stimulus change time (trials.change_time): the onset of the changed image on go trials and '
            'of the sham change on catch trials; equal to the start_time of that stimulus flash'),
        'off_start': OFF_START,
        'off_end': OFF_END,
        'image_change_definition': (
            f'1 in the {CHANGE_WINDOW * 1000:.0f} ms following the change of image identity on go '
            'trials (the two bins covering the presentation of the new image and the start of the '
            'following gray period), 0 elsewhere and everywhere on catch trials (sham change, the '
            'image identity does not change)'),
        'image_identity_definition': (
            'the image presented in the 750 ms flash interval containing the bin centre (the '
            'reference paper\'s "image presentation interval"); omitted flashes are their own '
            'category'),
        'n_timepoints_per_trial': NBINS,
        'neural_signal': SIGNAL_DESCRIPTION[signal_name],
        'neural_units': ('mean dF/F per 250 ms bin' if signal_name == 'dff'
                         else 'mean detected-calcium-event magnitude per 250 ms bin'),
        'trial_selection': 'trials.go | trials.catch, excluding aborted and auto-rewarded trials',
        'session_selection': (
            'all locally available active-behaviour (passive == False) ophys sessions of the '
            'Visual Behavior 2P release; experiments (imaging planes) recorded simultaneously in the '
            'same ophys session are concatenated along the neuron axis; sessions without eye tracking '
            'are dropped because pupil diameter is a required output'),
        'neuron_selection': (
            'all cells in the released NWB files (cell_specimen_table.valid_roi is True for every cell; '
            'ROI filtering, demixing and neuropil correction were applied by the Allen pipeline)'),
        'output_discretization': (
            'running speed and pupil diameter are discretized into five equal-percentile bins computed '
            'per session over exactly the exported timepoints; pupil diameter = 2*sqrt(pupil_area/pi) '
            'with blink NaNs linearly interpolated over time'),
        'dataset': 'visual-behavior-ophys-1.1.0 (AllenSDK VisualBehaviorOphysProjectCache)',
        'session_info': session_info,
    }
    return data


def summarize(data, skipped, elapsed, results=None):
    n_sessions = len(data['neural'])
    n_trials = sum(len(s) for s in data['neural'])
    n_neurons = sum(len(b) for b in data['brain_region_idx'])
    print('\n' + '=' * 70)
    print(f'sessions:        {n_sessions}')
    print(f'trials:          {n_trials} (mean {n_trials / max(n_sessions, 1):.1f} per session)')
    print(f'neurons:         {n_neurons} (mean {n_neurons / max(n_sessions, 1):.1f} per session)')
    print(f'subjects:        {len(data["subjects"])}')
    print(f'brain regions:   {data["brain_regions"]}')
    print(f'timepoints:      {NBINS} x {BIN_SIZE * 1000:.0f} ms')
    for i, name in enumerate(data['output_names']):
        vals = np.concatenate([np.concatenate([o[i] for o in sess]) for sess in data['output']])
        cnt = np.bincount(vals, minlength=len(data['output_values'][i]))
        frac = cnt / cnt.sum()
        print(f'  {name}: ' + ', '.join(f'{v}={f:.3f}' for v, f in
                                        zip(data['output_values'][i], frac)))
    if results is not None:
        nd = sum(r['n_dropped_trials'] for r in results if r.get('skip') is None)
        print(f'trials dropped for missing data: {nd}')
    if skipped:
        print('skipped sessions:')
        for s in skipped:
            print(f'  {s["session_id"]}: {s["skip"]}')
    print(f'total conversion time: {elapsed:.1f} s')
    print('=' * 70)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('outfile', type=str)
    ap.add_argument('--full', action='store_true', help='process all sessions (default)')
    ap.add_argument('--sample', action='store_true', help='process only 2 sessions')
    ap.add_argument('--show-processing', action='store_true',
                    help='plot every processing step for up to 2 sessions')
    ap.add_argument('--nproc', type=int, default=8)
    ap.add_argument('--neural-lag', type=float, default=0.0,
                    help='testing only: shift the neural bin grid later by this many seconds')
    ap.add_argument('--nsessions', type=int, default=0,
                    help='testing only: convert N sessions spread evenly over the dataset')
    ap.add_argument('--signal', type=str, default=NEURAL_SIGNAL,
                    choices=['events', 'filtered_events', 'dff'])
    args = ap.parse_args()

    t_start = time.time()
    cache = get_cache()
    exps, session_ids = select_sessions(cache)
    print(f'{len(exps)} active-behaviour experiments in {len(session_ids)} ophys sessions')

    if args.sample:
        # one single-plane session and one Multiscope session, to exercise both paths
        single = exps[exps.project_code == 'VisualBehavior'].ophys_session_id.iloc[0]
        multi = exps[exps.project_code == 'VisualBehaviorMultiscope'].ophys_session_id.iloc[0]
        session_ids = [int(single), int(multi)]
        print(f'--sample: sessions {session_ids}')

    if args.nsessions:
        pick = np.linspace(0, len(session_ids) - 1, args.nsessions).astype(int)
        session_ids = [session_ids[i] for i in sorted(set(pick))]
        print(f'--nsessions: {len(session_ids)} sessions')

    show = args.show_processing
    jobs = []
    for i, sid in enumerate(session_ids):
        oeids = exps[exps.ophys_session_id == sid].index.tolist()
        jobs.append((int(sid), oeids, show and i < 2, args.signal, args.neural_lag))

    results = []
    if args.nproc > 1 and len(jobs) > 1:
        with Pool(args.nproc) as pool:
            for k, r in enumerate(pool.imap_unordered(convert_session, jobs, chunksize=1)):
                results.append(r)
                el = time.time() - t_start
                print(f'[{k + 1}/{len(jobs)}] session {r["session_id"]}: '
                      + (f'SKIPPED ({r["skip"]})' if r.get('skip') else
                         f'{r["neural"].shape[0]} neurons x {r["ntrials"]} trials '
                         f'({r["n_dropped_trials"]} dropped), '
                         f'{r["total_time"]:.1f}s (load {r["timing"]["load"]:.1f}s, '
                         f'neural {r["timing"]["neural"]:.1f}s)')
                      + f' | elapsed {el:.0f}s, eta {el / (k + 1) * (len(jobs) - k - 1):.0f}s',
                      flush=True)
    else:
        for k, job in enumerate(jobs):
            r = convert_session(job)
            results.append(r)
            print(f'[{k + 1}/{len(jobs)}] session {r["session_id"]}: '
                  + (f'SKIPPED ({r["skip"]})' if r.get('skip') else
                     f'{r["neural"].shape[0]} neurons x {r["ntrials"]} trials '
                     f'({r["n_dropped_trials"]} dropped), {r["total_time"]:.1f}s'), flush=True)

    skipped = [r for r in results if r.get('skip')]
    data = assemble(results, args.signal)
    with open(args.outfile, 'wb') as f:
        pickle.dump(data, f, protocol=4)
    print(f'\nwrote {args.outfile} ({os.path.getsize(args.outfile) / 1e6:.1f} MB)')
    summarize(data, skipped, time.time() - t_start, results)


if __name__ == '__main__':
    main()
