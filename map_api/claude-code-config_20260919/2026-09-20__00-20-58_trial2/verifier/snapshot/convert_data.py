#!/usr/bin/env python3
"""
Convert the MAP brain-wide Neuropixels dataset (DANDI:000363, NWB) into the decoder
format described in the task specification.

Usage
-----
    python -u /app/convert_data.py <outpicklefile> [--full | --sample] [--show-processing]

Options
-------
    --full             process all sessions (default)
    --sample           process only 2 sessions (for testing)
    --show-processing  plot every processing step for up to 2 sessions,
                       saved as processing_<session_id>.png
    --nproc N          number of worker processes (default: 16)

Design decisions (see /app/CONVERSION_NOTES.md for the full rationale)
---------------------------------------------------------------------
* Alignment event: the auditory **go cue** (`BehavioralEvents['go_start_times']`).
* Window: [-2.5, +1.5] s around the go cue, 80 non-overlapping 50 ms bins.
* Neural data: firing **rate** in Hz (spike count / 0.05 s), as in the reference
  `sliding_histogram(..., rate=True)`.
* Units: only `classification == 'good'` (the region-specific QC classifiers of the
  data paper / QC white paper).
* Trials: `auto_water` and `free_water` trials are dropped (reference
  `get_regular_trial_mask`), as are trials not annotated good in `is_good_trials`.
  Early-lick / `ignore` / photostim trials are KEPT because they are required by the
  decoder input & output specification.
* Sessions: dropped if they contain no good units or fewer than 2 usable trials.
"""

import argparse
import json
import os
import pickle
import sys
import time
import warnings
from collections import OrderedDict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ccf_regions import REGIONS, annotation_to_region  # noqa: E402

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------
DATA_DIR = '/app/data'

OFF_START = -2.5          # s, signed time from the go cue to the start of the trial window
OFF_END = 1.5             # s, signed time from the go cue to the end of the trial window
BIN_SIZE = 0.05           # s
NBINS = int(round((OFF_END - OFF_START) / BIN_SIZE))          # 80
BIN_EDGES = OFF_START + BIN_SIZE * np.arange(NBINS + 1)       # (81,)
BIN_CENTERS = BIN_EDGES[:-1] + BIN_SIZE / 2                   # (80,)

TONGUE_LIKELIHOOD_THRESH = 0.9   # DeepLabCut likelihood above which the tongue is "visible"
TONGUE_LOW_PCT = 40              # percentile boundaries for the tongue-y discretisation
TONGUE_HIGH_PCT = 60

ML_MIDLINE_UM = 5700             # reference rule: CCF x >= 5700 um => left hemisphere
OVERLAP_TOL = 1e-3               # s, minimum overlap for a bin to count as photostim-on

INPUT_NAMES = ['time_from_tone_onset', 'photostim_on']
OUTPUT_NAMES = ['choice', 'outcome', 'early_lick', 'tongue_y_position']
OUTPUT_VALUES = [
    ['left', 'right', 'no lick'],
    ['ignore', 'miss', 'hit'],
    ['no', 'yes'],
    ['low (<40th pct)', 'mid (40-60th pct)', 'high (>60th pct)', 'not visible'],
]

CHOICE_LEFT, CHOICE_RIGHT, CHOICE_NOLICK = 0, 1, 2
OUTCOME_CODE = {'ignore': 0, 'miss': 1, 'hit': 2}


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def list_session_files(data_dir=DATA_DIR):
    """Return the sorted list of NWB session files."""
    files = []
    for sub in sorted(os.listdir(data_dir)):
        d = os.path.join(data_dir, sub)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.endswith('.nwb'):
                files.append(os.path.join(d, f))
    return files


def bin_spikes_rate(spike_times, go_times):
    """Spike counts of one unit in the fixed window around every go cue, as a rate.

    Args:
        spike_times: (n_spikes,) sorted absolute spike times of one unit, in seconds.
        go_times: (n_trials,) absolute go-cue times, in seconds.

    Returns:
        (n_trials, NBINS) float32 array of firing rates in Hz.

    Notes:
        `np.searchsorted` is used on the (sorted) spike train, so the cost is
        O(n_trials * NBINS * log n_spikes) and no per-trial Python loop is needed.
        Bin k covers [go + BIN_EDGES[k], go + BIN_EDGES[k+1]).
    """
    edges = go_times[:, None] + BIN_EDGES[None, :]           # (n_trials, NBINS+1)
    idx = np.searchsorted(spike_times, edges.ravel(), side='left')
    idx = idx.reshape(edges.shape)
    counts = np.diff(idx, axis=1)
    return (counts / BIN_SIZE).astype(np.float32)


def interval_overlap_bins(starts, stops, go_time):
    """Binary (NBINS,) indicator: 1 where a [start, stop] interval overlaps the bin.

    Args:
        starts, stops: absolute start/stop times of the intervals (may be empty).
        go_time: absolute time of this trial's go cue.
    """
    on = np.zeros(NBINS, dtype=np.float32)
    if len(starts) == 0:
        return on
    for a, b in zip(starts - go_time, stops - go_time):
        if b <= BIN_EDGES[0] or a >= BIN_EDGES[-1]:
            continue
        # Overlap duration between [a, b] and every bin.  A plain "does it touch the bin"
        # test would mark a bin that the interval overlaps by a fraction of a millisecond:
        # photoinhibition ends at the go cue to within +-0.5 ms, which would otherwise
        # switch the first post-go bin on for about half of the stimulated trials.
        ov = np.minimum(b, BIN_EDGES[1:]) - np.maximum(a, BIN_EDGES[:-1])
        on[ov > OVERLAP_TOL] = 1.0
    return on


def _events(nwb, name):
    """Timestamps of a `BehavioralEvents` TimeSeries, as a numpy array."""
    ts = nwb.acquisition['BehavioralEvents'].time_series[name]
    return np.asarray(ts.timestamps[:], dtype=np.float64)


# --------------------------------------------------------------------------------------
# Per-session conversion
# --------------------------------------------------------------------------------------
def process_session(path, collect_debug=False):
    """Convert one NWB session.

    Args:
        path: path to the NWB file.
        collect_debug: if True, also return intermediate quantities for plotting.

    Returns:
        dict with the converted session, or None if the session has to be dropped.
    """
    from pynwb import NWBHDF5IO

    t0 = time.time()
    timing = {}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        io = NWBHDF5IO(path, 'r', load_namespaces=True)
        nwb = io.read()

        session_id = nwb.identifier
        subject = str(nwb.subject.subject_id)

        # ---------------- trials table -------------------------------------------
        trials = nwb.trials
        n_trials_raw = len(trials)
        instruction = np.asarray(trials['trial_instruction'][:])
        outcome = np.asarray(trials['outcome'][:])
        early_lick = np.asarray(trials['early_lick'][:])
        auto_water = np.asarray(trials['auto_water'][:]).astype(bool)
        free_water = np.asarray(trials['free_water'][:]).astype(bool)
        trial_start = np.asarray(trials['start_time'][:], dtype=np.float64)
        trial_stop = np.asarray(trials['stop_time'][:], dtype=np.float64)

        go_times = _events(nwb, 'go_start_times')
        if len(go_times) != n_trials_raw:
            io.close()
            raise RuntimeError(f'{session_id}: {len(go_times)} go cues for {n_trials_raw} trials')

        # ---------------- units ---------------------------------------------------
        units = nwb.units
        classification = np.asarray(units['classification'][:])
        good = np.flatnonzero(classification == 'good')
        if len(good) == 0:
            io.close()
            return None

        anno = np.asarray(units['anno_name'][:])[good]
        egroups = np.asarray(units['electrode_group'].data[:])[good]
        probe_target = np.array([json.loads(g.location)['brain_regions'] for g in egroups])

        # unit -> electrode -> CCF x, for the hemisphere rule of the reference code
        er = units['electrodes']
        cum = np.asarray(er.data[:])
        elec_idx = np.asarray(er.target.data[:])[cum - 1][good]
        elec_x = np.asarray(nwb.electrodes['x'][:], dtype=np.float64)[elec_idx]
        hemisphere = np.where(elec_x >= ML_MIDLINE_UM, 'left', 'right')
        # fall back on the insertion's targeted hemisphere where CCF x is missing
        missing = np.isnan(elec_x)
        if missing.any():
            hemisphere[missing] = np.array([t.split(' ')[0] for t in probe_target])[missing]

        region_name = np.array([annotation_to_region(a, t) for a, t in zip(anno, probe_target)])
        unknown = region_name == None  # noqa: E711
        if unknown.any():
            print(f'  WARNING {session_id}: {unknown.sum()} units with unmapped annotation '
                  f'{set(anno[unknown])}', flush=True)
            region_name[unknown] = 'OtherCortex'
        region_idx = np.array([REGIONS.index(r) for r in region_name], dtype=np.int64)

        # ---------------- trial curation -----------------------------------------
        # In 9 sessions the ephys recording covers only a subset of the behavioural
        # trials, so the observed trials have to be located in the trials table first.
        # units.obs_intervals is identical for every unit of a session and its rows
        # match the [start_time, stop_time] of the observed trials exactly (verified
        # for all 174 files).
        obs = np.asarray(units['obs_intervals'][int(good[0])], dtype=np.float64)
        obs_trial = np.searchsorted(trial_start, obs[:, 0] - 1e-6)
        if not np.allclose(trial_start[obs_trial], obs[:, 0]):
            raise RuntimeError(f'{session_id}: obs_intervals do not match the trials table')

        # (a) reference `get_regular_trial_mask`: no auto-water, no free-water trials
        keep = (~auto_water[obs_trial]) & (~free_water[obs_trial])
        # (b) recording-stability annotation: keep only trials annotated 'good' for
        #     every retained unit (i.e. for every probe insertion contributing units)
        is_good_trials = np.asarray(units['is_good_trials'][:])[good]   # (n_good, n_obs)
        keep &= is_good_trials.all(axis=0)
        keep_idx = obs_trial[keep]
        n_trials = len(keep_idx)
        n_trials_obs = len(obs_trial)
        if n_trials < 2:
            io.close()
            return None

        go = go_times[keep_idx]
        timing['setup'] = time.time() - t0

        # ---------------- neural ---------------------------------------------------
        t1 = time.time()
        sv = units['spike_times']
        ends = np.asarray(sv.data[:])
        starts = np.concatenate([[0], ends[:-1]])
        flat_spikes = np.asarray(sv.target.data[:], dtype=np.float64)

        n_neurons = len(good)
        fr = np.empty((n_neurons, n_trials, NBINS), dtype=np.float32)
        for i, u in enumerate(good):
            st = flat_spikes[starts[u]:ends[u]]
            fr[i] = bin_spikes_rate(st, go)

        # A trial in which not one of the (hundreds of) good units fired a single spike
        # means the amplifiers were not running: the recording can stop part-way through
        # the last annotated trial. Such trials carry no neural information at all.
        nonempty = fr.sum(axis=(0, 2)) > 0
        if not nonempty.all():
            n_empty = int((~nonempty).sum())
            print(f'  {session_id}: dropping {n_empty} trial(s) with no spikes at all',
                  flush=True)
            fr = fr[:, nonempty, :]
            keep_idx = keep_idx[nonempty]
            go = go[nonempty]
            n_trials = len(keep_idx)
            if n_trials < 2:
                io.close()
                return None
        timing['neural'] = time.time() - t1

        # ---------------- inputs ---------------------------------------------------
        t1 = time.time()
        # tone onset := last sample-epoch start before the go cue
        sample_start = _events(nwb, 'sample_start_times')
        tone_pos = np.searchsorted(sample_start, go, side='right') - 1
        if np.any(tone_pos < 0):
            raise RuntimeError(f'{session_id}: trial without a preceding sample epoch')
        tone_time = sample_start[tone_pos]
        # time from tone onset at each bin centre (seconds, continuous)
        time_from_tone = BIN_CENTERS[None, :] + (go - tone_time)[:, None]   # (n_trials, NBINS)

        photostim_on = np.zeros((n_trials, NBINS), dtype=np.float32)
        try:
            ps_start = _events(nwb, 'photostim_start_times')
            ps_stop = _events(nwb, 'photostim_stop_times')
        except KeyError:
            ps_start = ps_stop = np.zeros(0)
        if len(ps_start) and len(ps_start) == len(ps_stop):
            for j, g in enumerate(go):
                m = (ps_stop > g + OFF_START) & (ps_start < g + OFF_END)
                if m.any():
                    photostim_on[j] = interval_overlap_bins(ps_start[m], ps_stop[m], g)
        elif len(ps_start) != len(ps_stop):
            print(f'  WARNING {session_id}: photostim start/stop counts differ '
                  f'({len(ps_start)}/{len(ps_stop)}); photostim input left at 0', flush=True)
        timing['input'] = time.time() - t1

        # ---------------- outputs --------------------------------------------------
        t1 = time.time()
        instr_k = instruction[keep_idx]
        out_k = outcome[keep_idx]
        early_k = early_lick[keep_idx]

        # choice: hit -> instructed side, miss -> opposite side, ignore -> no lick
        choice = np.full(n_trials, CHOICE_NOLICK, dtype=np.int64)
        hit = out_k == 'hit'
        miss = out_k == 'miss'
        choice[hit & (instr_k == 'left')] = CHOICE_LEFT
        choice[hit & (instr_k == 'right')] = CHOICE_RIGHT
        choice[miss & (instr_k == 'left')] = CHOICE_RIGHT
        choice[miss & (instr_k == 'right')] = CHOICE_LEFT

        outcome_code = np.array([OUTCOME_CODE[o] for o in out_k], dtype=np.int64)
        early_code = (early_k == 'early').astype(np.int64)

        tongue_y_binned, tongue_visible_frac = bin_tongue(nwb, trial_start, go, keep_idx)
        tongue_class, pct = discretise_tongue(tongue_y_binned)
        timing['output'] = time.time() - t1

        io.close()

    # ---------------- assemble ----------------------------------------------------
    neural = [fr[:, j, :] for j in range(n_trials)]
    inputs = [np.stack([time_from_tone[j], photostim_on[j]]).astype(np.float32)
              for j in range(n_trials)]
    outputs = [np.stack([np.full(NBINS, choice[j]),
                         np.full(NBINS, outcome_code[j]),
                         np.full(NBINS, early_code[j]),
                         tongue_class[j]]).astype(np.int64)
               for j in range(n_trials)]

    result = dict(
        session_id=session_id,
        subject=subject,
        file=os.path.basename(path),
        neural=neural,
        input=inputs,
        output=outputs,
        brain_region_idx=region_idx,
        hemisphere=np.where(hemisphere == 'left', 0, 1).astype(np.int8),
        n_trials_raw=n_trials_raw,
        n_trials_observed=n_trials_obs,
        n_trials=n_trials,
        n_neurons=n_neurons,
        trial_index=keep_idx,                     # index into the raw trials table
        tongue_percentiles=pct,
        tongue_visible_frac=float(tongue_visible_frac),
        timing=timing,
        total_time=time.time() - t0,
    )
    if collect_debug:
        result['debug'] = dict(
            go=go, tone_time=tone_time, trial_start=trial_start[keep_idx],
            trial_stop=trial_stop[keep_idx], time_from_tone=time_from_tone,
            photostim_on=photostim_on, tongue_y=tongue_y_binned, tongue_class=tongue_class,
            fr=fr, instruction=instr_k, outcome=out_k, early=early_k,
            choice=choice, anno=anno, region_name=region_name, hemisphere=hemisphere,
        )
    return result


def bin_tongue(nwb, trial_start, go, keep_idx):
    """Mean tongue y-position per 50 ms bin, NaN where the tongue is not visible.

    Video frames are first assigned to the trial they were acquired in (the camera only
    runs during a trial), so a window that extends outside its own trial cannot pick up
    frames belonging to a neighbouring trial.

    Args:
        nwb: open NWBFile.
        trial_start: (n_trials_raw,) absolute trial start times.
        go: (n_trials,) absolute go-cue times of the retained trials.
        keep_idx: (n_trials,) indices of the retained trials in the raw trials table.

    Returns:
        (n_trials, NBINS) float array with the mean visible tongue y per bin (NaN when
        the tongue was never visible in that bin), and the fraction of visible frames.
    """
    n_trials = len(keep_idx)
    out = np.full((n_trials, NBINS), np.nan, dtype=np.float64)

    bts = nwb.acquisition.get('BehavioralTimeSeries', None)
    if bts is None or 'Camera0_side_TongueTracking' not in bts.time_series:
        return out, 0.0

    ts_obj = bts.time_series['Camera0_side_TongueTracking']
    frame_t = np.asarray(ts_obj.timestamps[:], dtype=np.float64)
    data = np.asarray(ts_obj.data[:, 1:3], dtype=np.float64)     # y, likelihood
    y = data[:, 0]
    visible = data[:, 1] > TONGUE_LIKELIHOOD_THRESH
    visible_frac = float(visible.mean()) if len(visible) else 0.0

    # frame -> trial (video only runs within a trial; frames start at trial_start)
    frame_trial = np.searchsorted(trial_start, frame_t, side='right') - 1

    # only visible frames matter for the value; keep them and their trial/bin index
    vis = np.flatnonzero(visible & (frame_trial >= 0))
    if len(vis) == 0:
        return out, visible_frac
    ft = frame_trial[vis]

    # map raw trial index -> position in keep_idx (-1 when the trial was dropped)
    n_raw = len(trial_start)
    pos_of_trial = np.full(n_raw, -1, dtype=np.int64)
    pos_of_trial[keep_idx] = np.arange(n_trials)
    pos = pos_of_trial[ft]
    sel = pos >= 0
    vis, pos = vis[sel], pos[sel]
    if len(vis) == 0:
        return out, visible_frac

    rel = frame_t[vis] - go[pos]
    inwin = (rel >= OFF_START) & (rel < OFF_END)
    vis, pos, rel = vis[inwin], pos[inwin], rel[inwin]
    if len(vis) == 0:
        return out, visible_frac

    bin_idx = np.floor((rel - OFF_START) / BIN_SIZE).astype(np.int64)
    np.clip(bin_idx, 0, NBINS - 1, out=bin_idx)
    flat = pos * NBINS + bin_idx
    n_cells = n_trials * NBINS
    sums = np.bincount(flat, weights=y[vis], minlength=n_cells)
    cnts = np.bincount(flat, minlength=n_cells)
    with np.errstate(invalid='ignore', divide='ignore'):
        mean = np.where(cnts > 0, sums / np.maximum(cnts, 1), np.nan)
    out = mean.reshape(n_trials, NBINS)
    return out, visible_frac


def discretise_tongue(tongue_y):
    """Discretise tongue y into 0/1/2 by session percentiles, 3 where not visible."""
    cls = np.full(tongue_y.shape, 3, dtype=np.int64)
    vis = ~np.isnan(tongue_y)
    if vis.sum() == 0:
        return cls, (np.nan, np.nan)
    vals = tongue_y[vis]
    p_lo = np.percentile(vals, TONGUE_LOW_PCT)
    p_hi = np.percentile(vals, TONGUE_HIGH_PCT)
    v = tongue_y[vis]
    c = np.ones(v.shape, dtype=np.int64)
    c[v < p_lo] = 0
    c[v > p_hi] = 2
    cls[vis] = c
    return cls, (float(p_lo), float(p_hi))


# --------------------------------------------------------------------------------------
# Plotting (--show-processing)
# --------------------------------------------------------------------------------------
def plot_processing(res, path):
    """Plot every processing step for one session so it can be checked visually."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    dbg = res['debug']
    fr = dbg['fr']
    n_trials = res['n_trials']
    t = BIN_CENTERS

    fig, ax = plt.subplots(4, 2, figsize=(20, 20))

    # (1) trial structure relative to the go cue
    a = ax[0, 0]
    rel_start = dbg['trial_start'] - dbg['go']
    rel_stop = dbg['trial_stop'] - dbg['go']
    rel_tone = dbg['tone_time'] - dbg['go']
    a.plot(rel_start, np.arange(n_trials), '.', ms=2, label='trial start')
    a.plot(rel_stop, np.arange(n_trials), '.', ms=2, label='trial stop (end of recording)')
    a.plot(rel_tone, np.arange(n_trials), '.', ms=2, label='tone (sample) onset')
    a.axvline(0, color='k', label='go cue')
    a.axvline(OFF_START, color='r', ls='--'), a.axvline(OFF_END, color='r', ls='--')
    a.set_xlabel('time from go cue (s)'), a.set_ylabel('trial'), a.legend(fontsize=7)
    a.set_title('Step 1: temporal alignment (red = extraction window)')

    # (2) population PSTH, sorted by choice
    a = ax[0, 1]
    order = np.argsort(dbg['choice'], kind='stable')
    pop = fr.mean(axis=0)[order]
    im = a.imshow(pop, aspect='auto', origin='lower',
                  extent=[OFF_START, OFF_END, 0, n_trials], cmap='magma')
    a.axvline(0, color='w')
    plt.colorbar(im, ax=a, label='pop. mean rate (Hz)')
    a.set_xlabel('time from go cue (s)'), a.set_ylabel('trial (sorted by choice)')
    a.set_title('Step 2: binned firing rates (50 ms bins)')

    # (3) mean rate of the 40 most active neurons
    a = ax[1, 0]
    mean_fr = fr.mean(axis=(1, 2))
    top = np.argsort(-mean_fr)[:40]
    for i in top:
        a.plot(t, fr[i].mean(axis=0), lw=0.7)
    a.axvline(0, color='k')
    a.set_xlabel('time from go cue (s)'), a.set_ylabel('mean rate (Hz)')
    a.set_title('Step 2b: trial-averaged rates, 40 most active neurons')

    # (4) input 0: time from tone onset
    a = ax[1, 1]
    for j in range(0, n_trials, max(1, n_trials // 40)):
        a.plot(t, dbg['time_from_tone'][j], lw=0.7)
    a.axvline(0, color='k'), a.axhline(0, color='r', ls='--')
    a.set_xlabel('time from go cue (s)'), a.set_ylabel('time from tone onset (s)')
    a.set_title('Step 3: input 0 (red dashed = tone onset)')

    # (5) input 1: photostim
    a = ax[2, 0]
    im = a.imshow(dbg['photostim_on'], aspect='auto', origin='lower',
                  extent=[OFF_START, OFF_END, 0, n_trials], cmap='gray_r')
    a.axvline(0, color='r')
    a.set_xlabel('time from go cue (s)'), a.set_ylabel('trial')
    a.set_title('Step 3b: input 1, photostim on (%d stim trials)'
                % int((dbg['photostim_on'].max(axis=1) > 0).sum()))

    # (6) tongue y + discretisation
    a = ax[2, 1]
    ty = dbg['tongue_y']
    vis = ~np.isnan(ty)
    a.hist(ty[vis], bins=80, color='0.6')
    for p, c in zip(res['tongue_percentiles'], ['b', 'g']):
        a.axvline(p, color=c, lw=2)
    a.set_xlabel('tongue y (px)'), a.set_ylabel('# bins')
    a.set_title('Step 4: tongue-y distribution (visible bins) + 40/60 pct')

    # (7) tongue class raster
    a = ax[3, 0]
    im = a.imshow(dbg['tongue_class'], aspect='auto', origin='lower',
                  extent=[OFF_START, OFF_END, 0, n_trials], cmap='viridis',
                  vmin=-0.5, vmax=3.5, interpolation='nearest')
    a.axvline(0, color='r')
    plt.colorbar(im, ax=a, ticks=[0, 1, 2, 3], label='tongue class')
    a.set_xlabel('time from go cue (s)'), a.set_ylabel('trial')
    a.set_title('Step 4b: output 3, tongue class (3 = not visible)')

    # (8) per-trial outputs
    a = ax[3, 1]
    a.plot(dbg['choice'], np.arange(n_trials), '.', ms=3, label='choice (0=L,1=R,2=none)')
    a.plot(np.array([OUTCOME_CODE[o] for o in dbg['outcome']]) + 0.15, np.arange(n_trials),
           '.', ms=3, label='outcome (0=ign,1=miss,2=hit)')
    a.plot((dbg['early'] == 'early').astype(int) + 0.3, np.arange(n_trials), '.', ms=3,
           label='early lick')
    a.set_xlabel('value'), a.set_ylabel('trial'), a.legend(fontsize=7)
    a.set_title('Step 4c: outputs 0-2 (per trial)')

    fig.suptitle('%s  (%d neurons, %d trials)' % (res['session_id'], res['n_neurons'], n_trials))
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f'  wrote {path}', flush=True)


# --------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------
def _worker(args):
    path, debug = args
    try:
        return process_session(path, collect_debug=debug)
    except Exception as exc:          # keep one bad file from killing the whole run
        import traceback
        traceback.print_exc()
        print(f'  ERROR processing {path}: {exc}', flush=True)
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('outfile')
    ap.add_argument('--full', action='store_true', default=True)
    ap.add_argument('--sample', action='store_true')
    ap.add_argument('--show-processing', action='store_true')
    ap.add_argument('--nproc', type=int, default=16)
    args = ap.parse_args()

    files = list_session_files()
    if args.sample:
        files = files[:2]
    print(f'Found {len(files)} session files', flush=True)

    t_start = time.time()
    debug_for = set(files[:2]) if args.show_processing else set()

    jobs = [(f, f in debug_for) for f in files]
    results = []
    if args.nproc > 1 and len(files) > 1:
        import multiprocessing as mp
        ctx = mp.get_context('spawn')
        with ctx.Pool(min(args.nproc, len(files))) as pool:
            for i, res in enumerate(pool.imap(_worker, jobs)):
                if res is None:
                    print(f'[{i + 1}/{len(files)}] dropped {os.path.basename(jobs[i][0])}',
                          flush=True)
                else:
                    print('[%d/%d] %s: %d neurons, %d/%d trials (%.1fs: %s)'
                          % (i + 1, len(files), res['session_id'], res['n_neurons'],
                             res['n_trials'], res['n_trials_raw'], res['total_time'],
                             ', '.join(f'{k}={v:.1f}' for k, v in res['timing'].items())),
                          flush=True)
                    results.append(res)
    else:
        for i, job in enumerate(jobs):
            res = _worker(job)
            if res is None:
                print(f'[{i + 1}/{len(files)}] dropped {os.path.basename(job[0])}', flush=True)
                continue
            print('[%d/%d] %s: %d neurons, %d/%d trials (%.1fs: %s)'
                  % (i + 1, len(files), res['session_id'], res['n_neurons'], res['n_trials'],
                     res['n_trials_raw'], res['total_time'],
                     ', '.join(f'{k}={v:.1f}' for k, v in res['timing'].items())), flush=True)
            results.append(res)

    t_convert = time.time() - t_start
    print(f'\nConverted {len(results)} sessions in {t_convert:.1f}s '
          f'({t_convert / max(1, len(results)):.2f}s per session)', flush=True)

    if args.show_processing:
        for res in results:
            if 'debug' in res:
                plot_processing(res, f'processing_{res["session_id"]}.png')

    # ---------------- assemble the final dictionary -------------------------------
    subjects = sorted({r['subject'] for r in results})
    subject_idx = np.array([subjects.index(r['subject']) for r in results], dtype=np.int64)

    data = {
        'neural': [r['neural'] for r in results],
        'input': [r['input'] for r in results],
        'output': [r['output'] for r in results],
        'subjects': subjects,
        'subject_idx': subject_idx,
        'brain_regions': list(REGIONS),
        'brain_region_idx': [r['brain_region_idx'] for r in results],
        'input_names': list(INPUT_NAMES),
        'output_names': list(OUTPUT_NAMES),
        'output_values': [list(v) for v in OUTPUT_VALUES],
        'metadata': {
            'dataset': 'MAP multi-regional Neuropixels dataset (DANDI:000363)',
            'papers': [
                'Chen, Liu et al. (2024) Brain-wide neural activity underlying '
                'memory-guided movement. Cell 187:676-691',
                'Wang, Kurgyis et al. (2025) Brain-wide analysis reveals movement '
                'encoding structured across and within brain areas. Nat Neurosci',
            ],
            'task_description': (
                'Head-fixed mice perform an auditory delayed-response (memory-guided '
                'directional licking) task. During the sample epoch one of two pure tones '
                '(3 kHz or 12 kHz, played 3x150 ms with 100 ms gaps, 0.65 s total) '
                'instructs the mouse to lick left or right. After a 1.2 s delay epoch an '
                'auditory go cue (6 kHz, 0.1 s) opens a 1.5 s answer period in which the '
                'mouse reports its choice by licking one of two lick ports; licking the '
                'correct port yields a water reward. Licking during the sample/delay epoch '
                '(early lick) triggers a replay of the epoch. On ~20% of trials one or both '
                'ALM hemispheres were photoinhibited during the delay epoch. '
                'Decoder outputs: the lick-direction choice (left / right / no lick), the '
                'trial outcome (ignore / miss / hit), whether the trial contained an early '
                'lick (no / yes) and the discretised side-view tongue y-position '
                '(<40th pct / 40-60th pct / >60th pct of the session, or not visible).'),
            'temporal_alignment_event': 'onset of the auditory go cue (go_start_times)',
            'off_start': OFF_START,
            'off_end': OFF_END,
            'time_bin_size': BIN_SIZE * 1000.0,       # ms
            'n_time_bins': NBINS,
            'bin_centers_s': BIN_CENTERS.tolist(),
            'neural_units': 'spikes/s (firing rate = spike count / 50 ms bin)',
            'input_descriptions': [
                'time from tone (sample-epoch) onset in seconds at each bin centre; the '
                'tone onset used is the last sample-epoch start preceding the go cue '
                '(early-lick trials replay the sample epoch)',
                'binary indicator, 1 if ALM photostimulation was on during the bin',
            ],
            'output_descriptions': [
                'lick direction chosen by the animal, derived from trial_instruction and '
                'outcome (hit -> instructed side, miss -> opposite side, ignore -> no lick)',
                'trial outcome from the trials table (ignore = no lick, miss = licked the '
                'wrong port, hit = licked the correct port)',
                'whether the animal licked during the sample/delay epoch',
                'side-view (Camera0) DeepLabCut tongue y-position, averaged over the frames '
                'of each 50 ms bin in which the tongue was visible (likelihood > %.2f), then '
                'discretised with the 40th/60th percentiles of all visible bins of that '
                'session; bins with no visible frame are class 3 (not visible)'
                % TONGUE_LIKELIHOOD_THRESH,
            ],
            'neuron_curation': (
                "units.classification == 'good' (region-specific logistic-regression QC "
                'classifiers of Chen, Liu et al. 2023 white paper); no firing-rate threshold'),
            'trial_curation': (
                'auto_water and free_water trials removed (as in the reference '
                'get_regular_trial_mask); trials not annotated good in units.is_good_trials '
                'for every retained unit removed. Early-lick, ignore and photostimulation '
                'trials are RETAINED because they are required decoder outputs/inputs.'),
            'session_curation': ('sessions with no good units or fewer than 2 usable trials '
                                 'removed'),
            'session_info': [
                {'session_id': r['session_id'], 'subject': r['subject'], 'file': r['file'],
                 'n_neurons': int(r['n_neurons']), 'n_trials': int(r['n_trials']),
                 'n_trials_raw': int(r['n_trials_raw']), 'n_trials_observed': int(r['n_trials_observed']),
                 'tongue_pct_40_60': list(r['tongue_percentiles']),
                 'tongue_visible_frame_frac': r['tongue_visible_frac']}
                for r in results],
            # plain python ints so that the whole metadata dict stays JSON-serialisable
            # (train_decoder.py --stats-json dumps metadata verbatim)
            'neuron_hemisphere': [[int(v) for v in r['hemisphere']] for r in results],
            'hemisphere_values': ['left', 'right'],
            'caveat': (
                'Spikes exist only inside each trial\'s recorded interval '
                '(units.obs_intervals == trials [start_time, stop_time]). That interval '
                'covers the go cue by about -3.15 s / +1.80 s (medians), so 3% of trials are '
                'zero-padded before -2.5 s and 16% after +1.5 s. On error (miss) trials the '
                'recording stops at the error lick, so only ~8% of them reach go+1.5 s. As in '
                'the reference preprocessing (process_one_area), the full fixed window is '
                'binned regardless, which yields zeros outside the recorded interval.'),
        },
    }

    print(f'Writing {args.outfile} ...', flush=True)
    t1 = time.time()
    with open(args.outfile, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print('Wrote %s (%.2f GB) in %.1fs'
          % (args.outfile, os.path.getsize(args.outfile) / 1e9, time.time() - t1), flush=True)

    # ---------------- summary ------------------------------------------------------
    n_neurons = [r['n_neurons'] for r in results]
    n_tr = [r['n_trials'] for r in results]
    print('\n=== conversion summary ===')
    print('sessions           : %d' % len(results))
    print('subjects           : %d' % len(subjects))
    print('neurons (total)    : %d' % int(np.sum(n_neurons)))
    print('neurons / session  : mean %.1f, range %d-%d'
          % (np.mean(n_neurons), np.min(n_neurons), np.max(n_neurons)))
    print('trials (total)     : %d' % int(np.sum(n_tr)))
    print('trials / session   : mean %.1f, range %d-%d'
          % (np.mean(n_tr), np.min(n_tr), np.max(n_tr)))
    region_counts = np.zeros(len(REGIONS), dtype=int)
    for r in results:
        region_counts += np.bincount(r['brain_region_idx'], minlength=len(REGIONS))
    print('neurons per region :')
    for name, c in zip(REGIONS, region_counts):
        print('    %-18s %6d' % (name, c))
    allout = np.concatenate([np.concatenate([o[:, None, :] for o in r['output']], axis=1)
                             .reshape(len(OUTPUT_NAMES), -1) for r in results], axis=1)
    for i, name in enumerate(OUTPUT_NAMES):
        cnt = np.bincount(allout[i], minlength=len(OUTPUT_VALUES[i]))
        frac = cnt / cnt.sum()
        print('output %-18s %s' % (name, ', '.join(
            f'{v}={f:.3f}' for v, f in zip(OUTPUT_VALUES[i], frac))))
    print('total wall time    : %.1fs' % (time.time() - t_start))


if __name__ == '__main__':
    main()
