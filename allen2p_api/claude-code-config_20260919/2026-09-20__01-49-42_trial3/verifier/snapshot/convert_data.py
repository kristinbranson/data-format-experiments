#!/usr/bin/env python3
"""
Convert the Allen Brain Observatory **Visual Behavior 2P** dataset into the
decoder-compatible pickle format described in the task specification.

Usage
-----
    python -u /app/convert_data.py <outpicklefile> [--full | --sample] [--show-processing]

    --full             process every selected session (default)
    --sample           process only the first 2 selected sessions (quick test)
    --show-processing  render step-by-step verification plots for up to 2 sessions
                       into processing_<ophys_experiment_id>.png

What is converted
-----------------
* Sessions   : `project_code == 'VisualBehavior'`, active behaviour (not passive),
               with a non-empty eye-tracking table.  One session == one
               BehaviorOphysExperiment (a single imaging plane; the VisualBehavior
               project is single-plane, so session == experiment == NWB file).
* Trials     : rows of `ds.trials` with `go | catch` (this automatically excludes
               aborted and auto-rewarded trials -- see allensdk
               `data_objects/trials/trial.py:_get_trial_data`).  A trial spans
               [start_time, stop_time) exactly as the experiment defines it.
* Time base  : `ds.ophys_timestamps` (2-photon frame times, 31 Hz).  Every other
               stream is resampled onto those timestamps.
* Neural     : `ds.dff_traces.dff` -- the detrended dF/F traces produced by the
               Allen processing pipeline (already computed in the released data).
* Inputs     : none (the task specifies no decoder inputs).
* Outputs    : image identity, image change, running-speed quintile,
               pupil-diameter quintile (all time-varying), trial outcome (static).

All data access goes through the AllenSDK `VisualBehaviorOphysProjectCache`; the
NWB files are never opened directly.
"""

import argparse
import os
import pickle
import re
import sys
import time
import warnings
from glob import glob
from multiprocessing import Pool

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

import allensdk.brain_observatory.behavior.behavior_project_cache as bpc

CACHE_DIR = '/app/data'
NWB_DIR = '/app/data/visual-behavior-ophys-1.1.0/behavior_ophys_experiments'
METADATA_DIR = '/app/data/visual-behavior-ophys-1.1.0/project_metadata'

PROJECT_CODE = 'VisualBehavior'          # the "Visual Behavior" dataset variant
NEURAL_SIGNAL = 'dff'                    # detrended dF/F (Allen pipeline output)
N_QUANTILE_BINS = 5                      # "five equal percentile bins"
OUTCOME_NAMES = ['hit', 'miss', 'false_alarm', 'correct_reject']
GREY_LABEL = 'grey'                      # image identity when no image is on screen
CHANGE_WINDOW = 'flash'                  # 'flash' = 250 ms image, 'presentation_interval' = 750 ms

NEURAL_SIGNAL_DESCRIPTION = {
    'dff': ('ds.dff_traces["dff"] - detrended dF/F per cell, the output of the Allen '
            'two-photon processing pipeline (motion correction, segmentation, ROI filtering, '
            'demixing, neuropil subtraction, dF/F with a 600 s median-filter baseline, '
            'detrending). Already computed in the released data; one value per cell per '
            'two-photon frame.'),
    'filtered_events': ('ds.events["filtered_events"] - Allen-detected calcium events '
                        'convolved with the AllenSDK half-normal smoothing kernel.'),
    'events': ('ds.events["events"] - Allen-detected (deconvolved) calcium events, '
               'unsmoothed.'),
}

# --------------------------------------------------------------------------- #
# cache handling (one cache object per process)
# --------------------------------------------------------------------------- #
_CACHE = None


def get_cache():
    """Return this process's AllenSDK project cache (created lazily)."""
    global _CACHE
    if _CACHE is None:
        _CACHE = bpc.VisualBehaviorOphysProjectCache.from_local_cache(
            cache_dir=CACHE_DIR)
    return _CACHE


# --------------------------------------------------------------------------- #
# session selection
# --------------------------------------------------------------------------- #
def local_experiment_ids():
    """ophys_experiment_ids whose NWB file is present in the local cache."""
    files = glob(os.path.join(NWB_DIR, '*.nwb'))
    return sorted(int(re.search(r'(\d+)\.nwb', f).group(1)) for f in files)


def select_experiments():
    """Table of the ophys experiments to convert, sorted by experiment id.

    Curation (see CONVERSION_NOTES.md Step 5):
      * present locally,
      * project_code == 'VisualBehavior'  (the named dataset variant; the only
        project completely present in this cache),
      * active behaviour (passive sessions have a retracted lick spout and hence
        no go/catch trial outcomes).
    Sessions without eye tracking are dropped later, once the NWB is open.
    """
    et = get_cache().get_ophys_experiment_table()
    ids = local_experiment_ids()
    sel = et[(et.index.isin(ids)) &
             (et.project_code == PROJECT_CODE) &
             (~et.passive)].copy()
    return sel.sort_index()


# --------------------------------------------------------------------------- #
# per-session extraction
# --------------------------------------------------------------------------- #
def frame_slice(ophys_ts, t_start, t_stop):
    """Indices of ophys frames with t_start <= t < t_stop."""
    i0 = int(np.searchsorted(ophys_ts, t_start, side='left'))
    i1 = int(np.searchsorted(ophys_ts, t_stop, side='left'))
    return i0, i1


def build_stimulus_timeseries(stim, ophys_ts):
    """Per-ophys-frame image identity and image-change timeseries.

    Returns
    -------
    img_local : int16 (n_frames,)   0 == grey, 1..n == index into `image_names`
    is_change : uint8 (n_frames,)   1 during the 250 ms presentation of a changed image
    image_names : list of str       the session's images, sorted
    """
    n = len(ophys_ts)
    img_local = np.zeros(n, dtype=np.int16)
    is_change = np.zeros(n, dtype=np.uint8)

    # the change-detection block only (the SDK table also holds the 5-min grey
    # screens and the 5-min natural movie -- see the AllenSDK tutorial)
    block = stim[stim.stimulus_block_name.str.contains('change_detection', na=False)]
    shown = block[~block.omitted.astype(bool)]          # omitted flash == grey screen

    image_names = sorted(set(shown.image_name.unique()))
    code = {name: i + 1 for i, name in enumerate(image_names)}

    starts = shown.start_time.values
    ends = shown.end_time.values
    # window used for the image_change label
    if CHANGE_WINDOW == 'presentation_interval':
        # the 750 ms interval beginning with the flash (Piet et al.'s
        # "image presentation interval"): ends at the onset of the *next*
        # presentation, omitted ones included, so an omission after a change
        # does not stretch the window to 1.5 s.
        all_starts = block.start_time.values
        nxt = np.searchsorted(all_starts, starts, side='right')
        chg_ends = np.where(nxt < len(all_starts),
                            all_starts[np.minimum(nxt, len(all_starts) - 1)],
                            ends)
    else:
        chg_ends = ends
    i0 = np.searchsorted(ophys_ts, starts, side='left')
    i1 = np.searchsorted(ophys_ts, ends, side='left')
    i1c = np.searchsorted(ophys_ts, chg_ends, side='left')
    codes = shown.image_name.map(code).values.astype(np.int16)
    changes = shown.is_change.values.astype(bool)

    for a, b, bc, c, ch in zip(i0, i1, i1c, codes, changes):
        img_local[a:b] = c
        if ch:
            is_change[a:bc] = 1

    return img_local, is_change, image_names


def resample_running(ds, ophys_ts):
    """Running speed (cm/s) linearly interpolated onto the ophys timestamps."""
    rs = ds.running_speed
    t = rs.timestamps.values.astype(np.float64)
    v = rs.speed.values.astype(np.float64)
    good = np.isfinite(v)
    if not good.all():                      # not observed in this dataset, but be safe
        t, v = t[good], v[good]
    return np.interp(ophys_ts, t, v).astype(np.float32)


def resample_pupil(ds, ophys_ts):
    """Pupil diameter (pixels) interpolated onto the ophys timestamps.

    `pupil_area` is NaN exactly where `likely_blink`; those gaps are filled by
    linear interpolation from the surrounding valid samples (edges held constant),
    so every ophys frame gets a value and the trial keeps the same length as the
    neural data.  Diameter = 2*sqrt(area/pi).  Returns None when the session has
    no usable eye tracking.
    """
    try:
        eye = ds.eye_tracking
    except Exception:
        return None
    if eye is None or len(eye) == 0:
        return None
    t = eye.timestamps.values.astype(np.float64)
    area = eye.pupil_area.values.astype(np.float64)
    good = np.isfinite(area) & (area > 0)
    if good.sum() < 2:
        return None
    diam = 2.0 * np.sqrt(area[good] / np.pi)
    return np.interp(ophys_ts, t[good], diam).astype(np.float32)


def extract_session(oeid):
    """Load one experiment and return everything needed for the final dict.

    Returns a dict (or {'oeid':..., 'skip': reason}).  Continuous behavioural
    variables are returned unbinned; the quintile edges are global and can only
    be computed once every session has been read.
    """
    t0 = time.time()
    ds = get_cache().get_behavior_ophys_experiment(oeid)
    t_load = time.time() - t0

    ophys_ts = np.asarray(ds.ophys_timestamps, dtype=np.float64)

    # ---- neural ---------------------------------------------------------- #
    ev = ds.dff_traces if NEURAL_SIGNAL == 'dff' else ds.events
    if len(ev) == 0:
        return {'oeid': oeid, 'skip': 'no cells'}
    activity = np.vstack(ev[NEURAL_SIGNAL].values).astype(np.float32)
    assert activity.shape[1] == len(ophys_ts), \
        f'{oeid}: {NEURAL_SIGNAL} length {activity.shape[1]} != {len(ophys_ts)} ophys frames'
    cell_specimen_ids = np.asarray(ev.index.values)

    # ---- behaviour / stimulus timeseries --------------------------------- #
    img_local, is_change_ts, image_names = build_stimulus_timeseries(
        ds.stimulus_presentations, ophys_ts)
    running = resample_running(ds, ophys_ts)
    pupil = resample_pupil(ds, ophys_ts)
    if pupil is None:
        return {'oeid': oeid, 'skip': 'no eye tracking'}

    # ---- trials ----------------------------------------------------------- #
    trials = ds.trials
    sel = trials[(trials.go.astype(bool)) | (trials.catch.astype(bool))]
    # `go`/`catch` are mutually exclusive with aborted/auto_rewarded in the SDK;
    # assert it rather than trusting it.
    assert not sel.aborted.any(), f'{oeid}: aborted trial selected'
    assert not sel.auto_rewarded.any(), f'{oeid}: auto-rewarded trial selected'
    outcome_flags = sel[OUTCOME_NAMES].values.astype(bool)
    assert (outcome_flags.sum(axis=1) == 1).all(), \
        f'{oeid}: trials without exactly one outcome'
    outcomes = np.argmax(outcome_flags, axis=1).astype(np.int64)

    neural, img_tr, chg_tr, run_tr, pup_tr, keep = [], [], [], [], [], []
    for k, (_, tr) in enumerate(sel.iterrows()):
        i0, i1 = frame_slice(ophys_ts, tr.start_time, tr.stop_time)
        if i1 - i0 < 2:                    # no ophys coverage -> unusable trial
            continue
        keep.append(k)
        neural.append(activity[:, i0:i1].copy())
        img_tr.append(img_local[i0:i1].copy())
        chg_tr.append(is_change_ts[i0:i1].copy())
        run_tr.append(running[i0:i1].copy())
        pup_tr.append(pupil[i0:i1].copy())
    keep = np.asarray(keep, dtype=int)

    md = ds.metadata
    info = {
        'oeid': oeid,
        'skip': None,
        'neural': neural,
        'image_local': img_tr,
        'is_change': chg_tr,
        'running': run_tr,
        'pupil': pup_tr,
        'outcome': outcomes[keep] if len(keep) else np.zeros(0, dtype=np.int64),
        'image_names': image_names,
        'n_neurons': activity.shape[0],
        'cell_specimen_ids': cell_specimen_ids,
        'n_trials_selected': len(sel),
        'n_trials_kept': len(keep),
        'n_go': int(sel.go.sum()),
        'n_catch': int(sel.catch.sum()),
        'dt': float(np.median(np.diff(ophys_ts))),
        'mouse_id': str(md['mouse_id']),
        'targeted_structure': str(md['targeted_structure']),
        'session_type': str(md['session_type']),
        'cre_line': str(md['cre_line']),
        'imaging_depth': int(md['imaging_depth']),
        'ophys_session_id': int(md['ophys_session_id']),
        'behavior_session_id': int(md['behavior_session_id']),
        'ophys_frame_rate': float(md['ophys_frame_rate']),
        'equipment_name': str(md['equipment_name']),
        'load_seconds': t_load,
        'total_seconds': time.time() - t0,
    }
    return info


def extract_session_safe(oeid):
    try:
        return extract_session(oeid)
    except Exception as exc:                # keep the pool alive, report at the end
        import traceback
        return {'oeid': oeid, 'skip': f'ERROR {exc!r}', 'traceback': traceback.format_exc()}


# --------------------------------------------------------------------------- #
# binning
# --------------------------------------------------------------------------- #
def quantile_edges(values, nbins=N_QUANTILE_BINS):
    """Interior edges of `nbins` equal-percentile bins."""
    qs = np.arange(1, nbins) / nbins
    return np.quantile(values, qs)


def digitize(x, edges):
    """Bin index in 0..len(edges); edges are the interior quantile edges."""
    return np.digitize(x, edges, right=False).astype(np.int64)


# --------------------------------------------------------------------------- #
# plotting (--show-processing)
# --------------------------------------------------------------------------- #
def show_processing(info, run_edges, pup_edges, image_names_global, outfile):
    """Render a multi-panel figure convincing the reader every step is correct."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    oeid = info['oeid']
    ds = get_cache().get_behavior_ophys_experiment(oeid)
    ophys_ts = np.asarray(ds.ophys_timestamps, dtype=np.float64)
    stim = ds.stimulus_presentations
    block = stim[stim.stimulus_block_name.str.contains('change_detection', na=False)]
    trials = ds.trials
    sel = trials[(trials.go.astype(bool)) | (trials.catch.astype(bool))]

    fig = plt.figure(figsize=(26, 30))
    gs = fig.add_gridspec(7, 2, height_ratios=[1.3, 1.3, 1, 1, 1.3, 1, 1], hspace=.45, wspace=.18)

    # ---------- row 0-1: two example trials, raw streams vs converted -------- #
    # pick one go and one catch trial that are long enough
    go_idx = np.nonzero(sel.go.values)[0]
    catch_idx = np.nonzero(sel.catch.values)[0]
    picks = []
    if len(go_idx):
        picks.append(('go', int(go_idx[len(go_idx) // 2])))
    if len(catch_idx):
        picks.append(('catch', int(catch_idx[len(catch_idx) // 2])))
    while len(picks) < 2:
        picks.append(picks[-1])

    # map the position in `sel` to the position in the kept-trial lists
    # (trials are only dropped when they have <2 ophys frames, which does not
    #  happen in this dataset; guard anyway)
    n_kept = info['n_trials_kept']
    for col, (kind, si) in enumerate(picks):
        if si >= n_kept:
            si = n_kept - 1
        tr = sel.iloc[si]
        i0, i1 = frame_slice(ophys_ts, tr.start_time, tr.stop_time)
        t = ophys_ts[i0:i1]

        ax = fig.add_subplot(gs[0, col])
        neural = info['neural'][si]
        ax.imshow(neural, aspect='auto', interpolation='nearest',
                  extent=[t[0], t[-1], neural.shape[0], 0],
                  cmap='magma', vmin=0, vmax=np.percentile(neural, 99.9) + 1e-9)
        for _, f in block[(block.end_time > t[0]) & (block.start_time < t[-1])].iterrows():
            if f.omitted:
                continue
            ax.axvspan(f.start_time, f.end_time, color='cyan', alpha=.18)
        ax.axvline(tr.change_time, color='lime', lw=2)
        ax.set_title(f'{kind} trial #{si} (oeid {oeid})  neural = {NEURAL_SIGNAL}\n'
                     f'cyan = image flash, green = change_time, T={i1 - i0}')
        ax.set_ylabel('neuron')

        ax = fig.add_subplot(gs[1, col])
        ax.step(t, info['image_local'][si], where='post', label='image identity code', lw=2)
        ax.step(t, info['is_change'][si] * 3, where='post', label='image change x3',
                color='crimson', lw=2)
        for _, f in block[(block.end_time > t[0]) & (block.start_time < t[-1])].iterrows():
            ax.axvspan(f.start_time, f.end_time,
                       color=('0.8' if f.omitted else 'cyan'), alpha=.25)
        ax.axvline(tr.change_time, color='lime', lw=2)
        ax.set_ylabel('code (0 = grey)')
        ax.set_xlabel('time (s)')
        ax.legend(loc='upper left', fontsize=8)
        ax.set_title('image identity / change vs. the SDK flash table '
                     '(cyan = flash, grey = omitted)')

        ax = fig.add_subplot(gs[2, col])
        rs = ds.running_speed
        m = (rs.timestamps.values > t[0] - .5) & (rs.timestamps.values < t[-1] + .5)
        ax.plot(rs.timestamps.values[m], rs.speed.values[m], '-', color='0.6',
                label='raw 60 Hz running speed')
        ax.plot(t, info['running'][si], '.-', color='tab:blue', ms=3,
                label='resampled onto ophys frames')
        for e in run_edges:
            ax.axhline(e, ls=':', color='tab:orange', lw=1)
        ax.set_ylabel('cm/s')
        ax.legend(loc='upper left', fontsize=8)
        ax.set_title('running speed: raw vs resampled (dotted = global quintile edges)')

        ax = fig.add_subplot(gs[3, col])
        eye = ds.eye_tracking
        m = (eye.timestamps.values > t[0] - .5) & (eye.timestamps.values < t[-1] + .5)
        raw_d = 2 * np.sqrt(eye.pupil_area.values[m] / np.pi)
        ax.plot(eye.timestamps.values[m], raw_d, '-', color='0.6',
                label='raw pupil diameter (NaN at blinks)')
        ax.plot(t, info['pupil'][si], '.-', color='tab:green', ms=3,
                label='blink-interpolated, resampled')
        for e in pup_edges:
            ax.axhline(e, ls=':', color='tab:orange', lw=1)
        ax.set_ylabel('pixels')
        ax.set_xlabel('time (s)')
        ax.legend(loc='upper left', fontsize=8)
        ax.set_title('pupil diameter: raw vs resampled')

    # ---------- row 4: session-level alignment check ------------------------ #
    ax = fig.add_subplot(gs[4, :])
    t_mid = float(sel.iloc[len(sel) // 2].start_time)
    win = (t_mid, t_mid + 40)
    m = (ophys_ts >= win[0]) & (ophys_ts < win[1])
    tbl = ds.dff_traces if NEURAL_SIGNAL == 'dff' else ds.events
    pop = np.vstack(tbl[NEURAL_SIGNAL].values).astype(np.float32)[:, m].mean(axis=0)
    ax.plot(ophys_ts[m], pop / (pop.max() + 1e-9), color='k', label='population mean activity')
    for _, f in block[(block.end_time > win[0]) & (block.start_time < win[1])].iterrows():
        ax.axvspan(f.start_time, f.end_time,
                   color=('0.85' if f.omitted else 'cyan'), alpha=.3)
    for _, tr in sel[(sel.stop_time > win[0]) & (sel.start_time < win[1])].iterrows():
        ax.axvspan(tr.start_time, tr.stop_time, color='tab:red', alpha=.06)
        ax.axvline(tr.start_time, color='tab:red', ls='--', lw=1)
        ax.axvline(tr.change_time, color='lime', lw=1.5)
    ax.set_xlim(win)
    ax.set_title('40 s of session: flashes (cyan), go/catch trial windows (red shading, '
                 'dashed = start), change times (green)')
    ax.set_xlabel('time (s)')

    # ---------- row 5: discretisation ---------------------------------------- #
    allrun = np.concatenate(info['running'])
    allpup = np.concatenate(info['pupil'])
    ax = fig.add_subplot(gs[5, 0])
    ax.hist(allrun, bins=200, color='tab:blue')
    for e in run_edges:
        ax.axvline(e, color='tab:orange', ls='--')
    ax.set_yscale('log')
    ax.set_title('running speed distribution (this session) with global quintile edges')
    ax.set_xlabel('cm/s')

    ax = fig.add_subplot(gs[5, 1])
    ax.hist(allpup, bins=200, color='tab:green')
    for e in pup_edges:
        ax.axvline(e, color='tab:orange', ls='--')
    ax.set_yscale('log')
    ax.set_title('pupil diameter distribution (this session) with global quintile edges')
    ax.set_xlabel('pixels')

    # ---------- row 6: binning correctness + output distributions ----------- #
    ax = fig.add_subplot(gs[6, 0])
    sub = np.random.default_rng(0).choice(len(allrun), size=min(20000, len(allrun)),
                                          replace=False)
    ax.plot(allrun[sub], digitize(allrun[sub], run_edges), '.', ms=2, alpha=.3,
            label='running')
    ax.plot(allpup[sub], digitize(allpup[sub], pup_edges), '.', ms=2, alpha=.3,
            label='pupil')
    for e in run_edges:
        ax.axvline(e, color='tab:blue', ls=':', lw=1)
    for e in pup_edges:
        ax.axvline(e, color='tab:green', ls=':', lw=1)
    ax.set_xlabel('continuous value')
    ax.set_ylabel('assigned bin')
    ax.legend(fontsize=8)
    ax.set_title('value -> bin is monotone and changes exactly at the quintile edges')

    ax = fig.add_subplot(gs[6, 1])
    img_all = np.concatenate(info['image_local'])
    names = [GREY_LABEL] + list(info['image_names'])
    frac = [np.mean(img_all == i) for i in range(len(names))]
    ax.bar(np.arange(len(names)), frac, color='tab:purple')
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=60, ha='right')
    ax.axhline(2 / 3, color='k', ls=':', label='expected grey fraction (500/750 ms)')
    ax.set_ylabel('fraction of timepoints')
    ax.legend(fontsize=8)
    ax.set_title('image identity distribution in this session')

    fig.suptitle(f'Processing verification - ophys_experiment_id {oeid} '
                 f'({info["cre_line"]}, {info["session_type"]}, '
                 f'{info["n_neurons"]} cells, {info["n_trials_kept"]} trials)',
                 fontsize=16)
    fig.savefig(outfile, dpi=110, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {outfile}')


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    global NEURAL_SIGNAL, CHANGE_WINDOW
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('outfile', type=str, help='output pickle path')
    ap.add_argument('--full', action='store_true', help='process all sessions (default)')
    ap.add_argument('--sample', action='store_true', help='process only 2 sessions')
    ap.add_argument('--show-processing', action='store_true',
                    help='write processing_<oeid>.png for up to 2 sessions')
    ap.add_argument('--workers', type=int, default=16, help='parallel worker processes')
    ap.add_argument('--nsessions', type=int, default=None,
                    help='debug: process only the first N sessions')
    ap.add_argument('--change-window', type=str, default=CHANGE_WINDOW,
                    choices=['flash', 'presentation_interval'],
                    help='debug: window labelled as image_change')
    ap.add_argument('--neural-signal', type=str, default=NEURAL_SIGNAL,
                    choices=['dff', 'filtered_events', 'events'],
                    help='debug: which neural signal to convert (default dff)')
    args = ap.parse_args()
    NEURAL_SIGNAL = args.neural_signal
    CHANGE_WINDOW = args.change_window

    t_start = time.time()
    sel = select_experiments()
    oeids = list(sel.index.values)
    if args.sample:
        oeids = oeids[:2]
    if args.nsessions is not None:
        oeids = oeids[:args.nsessions]
    print(f'Selected {len(oeids)} ophys experiments '
          f'(project_code == {PROJECT_CODE!r}, active behaviour)')
    print(f'  mice: {sel.loc[oeids].mouse_id.nunique()}, '
          f'structures: {sorted(sel.loc[oeids].targeted_structure.unique())}')

    # ---- phase 1: read every session (parallel) ---------------------------- #
    t0 = time.time()
    if args.workers > 1 and len(oeids) > 1:
        with Pool(min(args.workers, len(oeids))) as pool:
            results = []
            for i, info in enumerate(pool.imap(extract_session_safe, oeids, chunksize=1)):
                results.append(info)
                if (i + 1) % 10 == 0 or i == len(oeids) - 1:
                    print(f'  read {i + 1}/{len(oeids)} sessions '
                          f'({time.time() - t0:.1f}s elapsed)', flush=True)
    else:
        results = [extract_session_safe(o) for o in oeids]
    t_read = time.time() - t0
    print(f'Phase 1 (read+extract): {t_read:.1f}s total, '
          f'{t_read / max(1, len(oeids)):.2f}s per session (wall, {args.workers} workers)')

    errors = [r for r in results if r.get('skip', '').__class__ is str
              and str(r.get('skip')).startswith('ERROR')]
    for r in errors:
        print(f'ERROR on {r["oeid"]}:\n{r.get("traceback")}')
    if errors:
        sys.exit(1)

    skipped = [r for r in results if r['skip']]
    for r in skipped:
        print(f'  SKIPPED {r["oeid"]}: {r["skip"]}')
    sessions = [r for r in results if not r['skip'] and r['n_trials_kept'] >= 2]
    dropped_few = [r for r in results if not r['skip'] and r['n_trials_kept'] < 2]
    for r in dropped_few:
        print(f'  SKIPPED {r["oeid"]}: only {r["n_trials_kept"]} usable trials')
    print(f'Kept {len(sessions)} sessions')

    # ---- phase 2: global quantile bins ------------------------------------- #
    t0 = time.time()
    all_run = np.concatenate([np.concatenate(s['running']) for s in sessions])
    all_pup = np.concatenate([np.concatenate(s['pupil']) for s in sessions])
    run_edges = quantile_edges(all_run)
    pup_edges = quantile_edges(all_pup)
    print(f'Running speed (cm/s): quintile edges {np.round(run_edges, 3).tolist()}, '
          f'range [{all_run.min():.2f}, {all_run.max():.2f}]')
    print(f'Pupil diameter (px):  quintile edges {np.round(pup_edges, 3).tolist()}, '
          f'range [{all_pup.min():.2f}, {all_pup.max():.2f}]')
    assert np.all(np.diff(run_edges) > 0), 'degenerate running-speed quantiles'
    assert np.all(np.diff(pup_edges) > 0), 'degenerate pupil quantiles'
    del all_run, all_pup

    # global image vocabulary: grey + every image seen in any session
    image_names_global = sorted({n for s in sessions for n in s['image_names']})
    image_values = [GREY_LABEL] + image_names_global
    global_code = {n: i + 1 for i, n in enumerate(image_names_global)}
    print(f'Images across dataset ({len(image_names_global)}): {image_names_global}')

    # ---- phase 3: assemble the output dict --------------------------------- #
    neural, inputs, outputs = [], [], []
    subjects = sorted({s['mouse_id'] for s in sessions})
    subject_idx = np.array([subjects.index(s['mouse_id']) for s in sessions], dtype=np.int64)
    brain_regions = sorted({s['targeted_structure'] for s in sessions})
    brain_region_idx = [np.full(s['n_neurons'], brain_regions.index(s['targeted_structure']),
                                dtype=np.int64) for s in sessions]

    session_info = []
    for s in sessions:
        # session-local image code -> global image code
        lut = np.zeros(len(s['image_names']) + 1, dtype=np.int64)   # 0 stays grey
        for i, name in enumerate(s['image_names']):
            lut[i + 1] = global_code[name]

        sess_neural, sess_in, sess_out = [], [], []
        for k in range(s['n_trials_kept']):
            act = s['neural'][k]
            T = act.shape[1]
            out = np.empty((5, T), dtype=np.int64)
            out[0] = lut[s['image_local'][k]]
            out[1] = s['is_change'][k]
            out[2] = digitize(s['running'][k], run_edges)
            out[3] = digitize(s['pupil'][k], pup_edges)
            out[4] = s['outcome'][k]                    # static, broadcast over T
            sess_neural.append(np.ascontiguousarray(act, dtype=np.float32))
            sess_in.append(np.zeros((0, T), dtype=np.float32))
            sess_out.append(out)
        neural.append(sess_neural)
        inputs.append(sess_in)
        outputs.append(sess_out)

        session_info.append({
            'ophys_experiment_id': int(s['oeid']),
            'ophys_session_id': s['ophys_session_id'],
            'behavior_session_id': s['behavior_session_id'],
            'mouse_id': s['mouse_id'],
            'cre_line': s['cre_line'],
            'session_type': s['session_type'],
            'experience_level': str(sel.loc[s['oeid'], 'experience_level']),
            'image_set': str(sel.loc[s['oeid'], 'image_set']),
            'targeted_structure': s['targeted_structure'],
            'imaging_depth': s['imaging_depth'],
            'equipment_name': s['equipment_name'],
            'n_neurons': s['n_neurons'],
            'n_trials': s['n_trials_kept'],
            'n_go_trials': s['n_go'],
            'n_catch_trials': s['n_catch'],
            'ophys_frame_rate_hz': s['ophys_frame_rate'],
            'median_frame_interval_s': s['dt'],
            'cell_specimen_ids': s['cell_specimen_ids'],
        })

    dt_all = np.array([s['dt'] for s in sessions])
    data = {
        'neural': neural,
        'input': inputs,
        'output': outputs,
        'subjects': subjects,
        'subject_idx': subject_idx,
        'brain_regions': brain_regions,
        'brain_region_idx': brain_region_idx,
        'input_names': [],
        'output_names': ['image_identity', 'image_change',
                         'running_speed_bin', 'pupil_diameter_bin', 'trial_outcome'],
        'output_values': [
            image_values,
            ['no_change', 'change'],
            [f'run_q{i + 1}' for i in range(N_QUANTILE_BINS)],
            [f'pupil_q{i + 1}' for i in range(N_QUANTILE_BINS)],
            OUTCOME_NAMES,
        ],
        'metadata': {
            'task_description': (
                'Allen Brain Observatory Visual Behavior 2P (project_code "VisualBehavior"). '
                'Head-fixed mice perform a go/no-go visual change-detection task: natural images '
                'are flashed for 250 ms every 750 ms (500 ms grey inter-stimulus interval, 5% of '
                'repeats omitted) and the mouse earns water by licking within 150-750 ms of a '
                'change in image identity. Two-photon calcium imaging of VISp at 31 Hz. '
                'The decoder has no inputs and predicts, from population activity alone: '
                '(0) the identity of the image on screen (17 classes: grey + 16 natural images), '
                '(1) whether an image change is currently on screen (binary), '
                '(2) running speed in global quintile bins, '
                '(3) pupil diameter in global quintile bins, and '
                '(4) the trial outcome (hit / miss / false_alarm / correct_reject, constant '
                'within a trial). Outputs 0-3 are time-varying, output 4 is static per trial.'),
            'time_bin_size': float(np.mean(dt_all) * 1000.0),   # ms
            'time_bin_size_range_ms': [float(dt_all.min() * 1000), float(dt_all.max() * 1000)],
            'temporal_alignment_event': (
                'Trial start (allensdk trials.start_time) of each go/catch trial; all data '
                'streams are resampled onto the two-photon ophys frame timestamps '
                '(ds.ophys_timestamps), which is the common clock for this dataset.'),
            'off_start': 0.0,
            'off_end': None,
            'trial_definition': (
                'One row of allensdk ds.trials with go==True or catch==True, spanning '
                '[start_time, stop_time). Aborted and auto-rewarded trials are excluded (the SDK '
                'already makes go/catch mutually exclusive with both). Trials have variable '
                'length (~7.0-12.6 s; the change occurs 2.25-8.25 s after trial start and the '
                'trial ends 4.24 s after the change), hence off_end is None.'),
            'neural_signal': NEURAL_SIGNAL_DESCRIPTION[NEURAL_SIGNAL],
            'sampling_rate_hz': float(1.0 / np.mean(dt_all)),
            'brain_region_note': 'targeted_structure of the imaging plane',
            'running_speed_bin_edges_cm_s': run_edges.tolist(),
            'pupil_diameter_bin_edges_px': pup_edges.tolist(),
            'pupil_note': ('pupil diameter = 2*sqrt(pupil_area/pi) from ds.eye_tracking; '
                           'blink frames (pupil_area NaN where likely_blink) are filled by '
                           'linear interpolation before resampling'),
            'curation': (
                f'project_code == "{PROJECT_CODE}"; active behaviour sessions only (passive '
                'sessions have a retracted lick spout and no trial outcomes); sessions without '
                'eye tracking dropped; no additional neuron filtering (the Allen pipeline\'s ROI '
                'filtering is already applied upstream and cell_specimen_table.valid_roi is '
                'all True in the released data).'),
            'n_sessions': len(sessions),
            'n_subjects': len(subjects),
            'n_neurons_total': int(sum(s['n_neurons'] for s in sessions)),
            'n_trials_total': int(sum(s['n_trials_kept'] for s in sessions)),
            'source': ('AllenSDK VisualBehaviorOphysProjectCache.from_local_cache('
                       f'cache_dir="{CACHE_DIR}"), visual-behavior-ophys-1.1.0'),
            'session_info': session_info,
        },
    }
    print(f'Phase 2+3 (bin+assemble): {time.time() - t0:.1f}s')

    # ---- quick self-checks -------------------------------------------------- #
    run_summary(data)

    # ---- optional plots ----------------------------------------------------- #
    if args.show_processing:
        for s in sessions[:2]:
            show_processing(s, run_edges, pup_edges, image_values,
                            f'processing_{s["oeid"]}.png')

    # ---- write -------------------------------------------------------------- #
    t0 = time.time()
    with open(args.outfile, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'Wrote {args.outfile} '
          f'({os.path.getsize(args.outfile) / 1e9:.2f} GB) in {time.time() - t0:.1f}s')
    print(f'TOTAL {time.time() - t_start:.1f}s')


def run_summary(data):
    """Print conversion statistics and assert the invariants the format requires."""
    print('\n--- conversion summary -------------------------------------------')
    nsess = len(data['neural'])
    ntr = [len(x) for x in data['neural']]
    nneu = [x[0].shape[0] for x in data['neural']]
    Ts = np.concatenate([[t.shape[1] for t in sess] for sess in data['neural']])
    print(f'sessions {nsess}, subjects {len(data["subjects"])}, '
          f'trials {sum(ntr)} (per session {np.mean(ntr):.1f} '
          f'[{min(ntr)}, {max(ntr)}])')
    print(f'neurons {sum(nneu)} (per session {np.mean(nneu):.1f} [{min(nneu)}, {max(nneu)}])')
    print(f'T per trial: mean {Ts.mean():.1f}, min {Ts.min()}, max {Ts.max()}; '
          f'total timepoints {Ts.sum()}')
    print(f'bin size {data["metadata"]["time_bin_size"]:.3f} ms')

    for d, name in enumerate(data['output_names']):
        vals = np.concatenate([np.concatenate([t[d] for t in sess])
                               for sess in data['output']])
        counts = np.bincount(vals, minlength=len(data['output_values'][d]))
        frac = counts / counts.sum()
        assert counts.sum() == Ts.sum()
        print(f'output {d} {name}: ' +
              ', '.join(f'{v}={f:.4f}' for v, f in zip(data['output_values'][d], frac)))

    # trial-level outcome distribution (each trial counted once)
    trial_outcomes = np.array([t[4, 0] for sess in data['output'] for t in sess])
    c = np.bincount(trial_outcomes, minlength=4)
    print('trial-level outcome counts: ' +
          ', '.join(f'{n}={v} ({v / c.sum():.3f})'
                    for n, v in zip(OUTCOME_NAMES, c)))
    print(f'  go trials (hit+miss) = {c[0] + c[1]}, catch trials (fa+cr) = {c[2] + c[3]}, '
          f'catch fraction = {(c[2] + c[3]) / c.sum():.4f}')

    # structural assertions
    for s in range(nsess):
        assert len(data['input'][s]) == ntr[s] and len(data['output'][s]) == ntr[s]
        assert len(data['brain_region_idx'][s]) == nneu[s]
        for k in range(ntr[s]):
            T = data['neural'][s][k].shape[1]
            assert data['input'][s][k].shape == (0, T)
            assert data['output'][s][k].shape == (5, T)
            assert data['neural'][s][k].shape[0] == nneu[s]
            assert np.isfinite(data['neural'][s][k]).all()
    print('structural assertions passed')
    print('------------------------------------------------------------------\n')


if __name__ == '__main__':
    main()
