#!/usr/bin/env python3
"""
Convert the Allen Brain Observatory "Visual Behavior 2P" dataset into the
decoder-compatible pickle format described in the task specification.

Usage
-----
    python -u /app/convert_data.py <outpicklefile> [--full | --sample] [--show-processing]

Options
-------
    --full             process all selected sessions (default)
    --sample           process only the first 2 selected sessions
    --show-processing  write processing_<ophys_experiment_id>.png diagnostic plots
                       for up to 2 sessions

What is converted (see CONVERSION_NOTES.md for the full rationale)
------------------------------------------------------------------
sessions   : project_code == 'VisualBehavior' (the single-plane project, 30.9406 Hz),
             active behaviour only (passive == False), eye tracking present.
trials     : rows of `BehaviorOphysExperiment.trials` with go | catch
             (this excludes aborted and auto_rewarded trials).
window     : [trials.start_time, trials.stop_time), sampled at the ophys frame
             timestamps that fall inside it.
neural     : `dff_traces.dff` -- the detrended dF/F traces released with the
             dataset (whitepaper section "DF/F CALCULATION").  `events` and
             `events.filtered_events` were also converted and benchmarked; dF/F
             decoded better on every output (see CONVERSION_NOTES.md Step 7).
input      : none (shape (0, T)).
outputs    : image_identity (17), image_change (2), running_speed_bin (5),
             pupil_diameter_bin (5), trial_outcome (4).
"""

import argparse
import os
import pickle
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

DATA_ROOT = '/app/data/visual-behavior-ophys-1.1.0'
NWB_DIR = os.path.join(DATA_ROOT, 'behavior_ophys_experiments')
META_DIR = os.path.join(DATA_ROOT, 'project_metadata')

PROJECT_CODE = 'VisualBehavior'          # single-plane Visual Behavior project
N_QUANTILE_BINS = 5                      # "five equal percentile bins"
MIN_TRIALS_PER_SESSION = 2               # decoder needs >= 2 trials per session
MIN_FRAMES_PER_TRIAL = 2

# The 8 + 8 natural images of image sets A and B (verified against the data).
IMAGE_NAMES = ['im000', 'im031', 'im035', 'im045', 'im054', 'im061', 'im062',
               'im063', 'im065', 'im066', 'im069', 'im073', 'im075', 'im077',
               'im085', 'im106']
# 0 == grey screen (inter-stimulus interval, and omitted flashes which are a
# continuation of the grey screen).
IMAGE_VALUES = ['grey'] + IMAGE_NAMES
IMAGE_CODE = {name: i + 1 for i, name in enumerate(IMAGE_NAMES)}

OUTCOME_VALUES = ['hit', 'miss', 'false_alarm', 'correct_reject']

OUTPUT_NAMES = ['image_identity', 'image_change', 'running_speed_bin',
                'pupil_diameter_bin', 'trial_outcome']
OUTPUT_VALUES = [
    IMAGE_VALUES,
    ['no_change', 'change'],
    [f'running_quintile_{i}' for i in range(N_QUANTILE_BINS)],
    [f'pupil_quintile_{i}' for i in range(N_QUANTILE_BINS)],
    OUTCOME_VALUES,
]


# ---------------------------------------------------------------------------
# session selection
# ---------------------------------------------------------------------------
def select_experiments():
    """Return the ophys_experiment_table rows that should be converted.

    Selection (Step 5 of CONVERSION_NOTES.md):
      * the NWB file is present locally,
      * project_code == 'VisualBehavior'  -> one imaging plane per session and a
        single uniform ophys frame rate (30.9406 Hz) across every session,
      * passive == False -> active change-detection behaviour only (the paper
        does not analyse passive sessions and their trial outcome is degenerate).
    Sessions without eye tracking are dropped later, when the file is opened.
    """
    et = pd.read_csv(os.path.join(META_DIR, 'ophys_experiment_table.csv'))
    present = set()
    for fn in os.listdir(NWB_DIR):
        if fn.endswith('.nwb'):
            present.add(int(fn.split('_')[-1].split('.')[0]))
    sel = et[et.ophys_experiment_id.isin(present)
             & (et.project_code == PROJECT_CODE)
             & (~et.passive.astype(bool))].copy()
    sel = sel.sort_values('ophys_experiment_id').reset_index(drop=True)
    return sel


# ---------------------------------------------------------------------------
# per-frame resampling helpers
# ---------------------------------------------------------------------------
def resample_running(running_speed, ts):
    """Linear interpolation of the 60 Hz running speed onto ophys frame times.

    The SDK already returns the unwrapped / transient-corrected / 10 Hz
    low-passed speed, so nothing but resampling is required.
    """
    rt = running_speed['timestamps'].values.astype(np.float64)
    rv = running_speed['speed'].values.astype(np.float64)
    good = np.isfinite(rt) & np.isfinite(rv)
    rt, rv = rt[good], rv[good]
    order = np.argsort(rt)
    return np.interp(ts, rt[order], rv[order])


def resample_pupil(eye_tracking, ts):
    """Blink-free pupil *diameter* interpolated onto ophys frame times.

    `pupil_area` is NaN exactly where `likely_blink` is True.  Those samples are
    dropped and the remaining ones linearly interpolated (standard blink
    handling).  The effective diameter 2*sqrt(area/pi) is reported; it is a
    monotonic function of area so the quintile bins are unaffected by the choice.

    Returns (pupil_diameter_at_ts, valid_sample_times) where valid_sample_times
    are the timestamps of the non-blink samples (used to reject trials that
    contain no measurement at all).
    """
    t = eye_tracking['timestamps'].values.astype(np.float64)
    a = eye_tracking['pupil_area'].values.astype(np.float64)
    good = np.isfinite(t) & np.isfinite(a) & (a > 0)
    t, a = t[good], a[good]
    order = np.argsort(t)
    t, a = t[order], a[order]
    diam = 2.0 * np.sqrt(a / np.pi)
    return np.interp(ts, t, diam), t


def stimulus_traces(stimulus_presentations, ts, change_window='flash'):
    """Per-ophys-frame image identity code and image-change indicator.

    A frame is assigned an image only while a (non-omitted) flash is actually on
    the screen, i.e. for t in [start_time, end_time) of that presentation
    (250 ms).  Everything else -- the 500 ms grey inter-stimulus interval and
    omitted flashes, which are a continuation of the grey screen -- is code 0
    ('grey').

    image_change is 1 on the frames of a flash whose `is_change` is True, i.e.
    the flash at which the image identity actually changed.  Catch trials carry
    `is_sham_change` (the same image is re-presented) and therefore correctly
    get 0.
    """
    sp = stimulus_presentations
    sp = sp[sp['stimulus_block_name'].astype(str).str.contains('change_detection')]
    start = sp['start_time'].values.astype(np.float64)
    end = sp['end_time'].values.astype(np.float64)
    names = sp['image_name'].astype(str).values
    is_change = sp['is_change'].values.astype(bool)
    order = np.argsort(start)
    start, end, names, is_change = start[order], end[order], names[order], is_change[order]

    codes = np.array([IMAGE_CODE.get(n, 0) for n in names], dtype=np.int16)

    idx = np.searchsorted(start, ts, side='right') - 1
    valid = idx >= 0
    idx_c = np.clip(idx, 0, len(start) - 1)
    on_screen = valid & (ts < np.nan_to_num(end[idx_c], nan=-np.inf))

    image = np.where(on_screen, codes[idx_c], 0).astype(np.int16)
    if change_window == 'interval':
        # the whole 750 ms image-presentation interval that starts at the change
        # (the unit the reference paper assigns events to)
        change = (valid & is_change[idx_c]).astype(np.int16)
    else:
        # only the 250 ms during which the changed image is actually on screen
        change = (on_screen & is_change[idx_c]).astype(np.int16)
    return image, change, sp


def quantile_bins(values, nbins=N_QUANTILE_BINS):
    """Assign `values` to `nbins` equal-percentile bins computed from `values`."""
    edges = np.quantile(values, np.arange(1, nbins) / nbins)
    return np.searchsorted(edges, values, side='right').astype(np.int16), edges


# ---------------------------------------------------------------------------
# one session
# ---------------------------------------------------------------------------
def process_experiment(job):
    """Convert one ophys experiment (== one session in this project).

    Returns a dict with the per-session pieces of the target format, or a dict
    with 'skip' explaining why the session was dropped.
    """
    oeid = job['ophys_experiment_id']
    show = job.get('show', False)
    t_start = time.time()
    timing = {}

    from allensdk.brain_observatory.behavior.behavior_ophys_experiment import (
        BehaviorOphysExperiment)

    path = os.path.join(NWB_DIR, f'behavior_ophys_experiment_{oeid}.nwb')
    t0 = time.time()
    ds = BehaviorOphysExperiment.from_nwb_path(path)
    ts = np.asarray(ds.ophys_timestamps, dtype=np.float64)
    timing['open'] = time.time() - t0

    # -- eye tracking must exist (pupil diameter is a required output) --------
    t0 = time.time()
    try:
        eye = ds.eye_tracking
    except Exception:
        eye = None
    if eye is None or len(eye) == 0 or not np.isfinite(eye['pupil_area'].values).any():
        return {'oeid': int(oeid), 'skip': 'no eye tracking data'}

    # -- neurons -------------------------------------------------------------
    cst = ds.cell_specimen_table
    valid_roi = cst['valid_roi'].values.astype(bool)
    cell_ids = cst.index.values[valid_roi]
    if len(cell_ids) == 0:
        return {'oeid': int(oeid), 'skip': 'no valid ROIs'}

    signal = job.get('neural_signal', 'dff')
    if signal == 'dff':
        col = ds.dff_traces['dff']
    else:
        col = ds.events[signal]
    traces = np.empty((len(cell_ids), len(ts)), dtype=np.float32)
    for i, cid in enumerate(cell_ids):
        traces[i] = col.loc[cid]
    if job.get('zscore', False):
        sd = traces.std(axis=1, keepdims=True)
        traces = (traces - traces.mean(axis=1, keepdims=True)) / np.maximum(sd, 1e-6)
    timing['neural'] = time.time() - t0

    # -- behaviour / stimulus streams resampled onto the ophys clock ---------
    t0 = time.time()
    run_all = resample_running(ds.running_speed, ts)
    pupil_all, pupil_valid_t = resample_pupil(eye, ts)
    image_all, change_all, sp = stimulus_traces(
        ds.stimulus_presentations, ts, job.get('change_window', 'flash'))
    timing['behavior'] = time.time() - t0

    # -- trials --------------------------------------------------------------
    trials = ds.trials
    go = trials['go'].values.astype(bool)
    catch = trials['catch'].values.astype(bool)
    keep = go | catch
    tr = trials[keep]

    outcome_cols = np.stack([tr['hit'].values.astype(bool),
                             tr['miss'].values.astype(bool),
                             tr['false_alarm'].values.astype(bool),
                             tr['correct_reject'].values.astype(bool)], axis=1)
    # every go/catch trial must have exactly one outcome
    assert outcome_cols.sum(axis=1).min() == 1 and outcome_cols.sum(axis=1).max() == 1, \
        f'{oeid}: go/catch trial without a unique outcome'
    outcome = np.argmax(outcome_cols, axis=1).astype(np.int16)

    starts = tr['start_time'].values.astype(np.float64)
    stops = tr['stop_time'].values.astype(np.float64)
    i0 = np.searchsorted(ts, starts, side='left')
    i1 = np.searchsorted(ts, stops, side='left')

    # a trial must contain at least one real (non-blink) pupil measurement,
    # otherwise its pupil trace would be pure extrapolation
    j0 = np.searchsorted(pupil_valid_t, starts, side='left')
    j1 = np.searchsorted(pupil_valid_t, stops, side='left')

    n_drop_short, n_drop_pupil = 0, 0
    sel = []
    for k in range(len(tr)):
        if i1[k] - i0[k] < MIN_FRAMES_PER_TRIAL:
            n_drop_short += 1
            continue
        if j1[k] - j0[k] < 1:
            n_drop_pupil += 1
            continue
        sel.append(k)
    sel = np.asarray(sel, dtype=int)
    if len(sel) < MIN_TRIALS_PER_SESSION:
        return {'oeid': int(oeid), 'skip': f'only {len(sel)} usable trials'}

    # -- per-session quintile bins over exactly the timepoints we keep -------
    frame_idx = np.concatenate([np.arange(i0[k], i1[k]) for k in sel])
    run_bin_all = np.zeros(len(ts), dtype=np.int16)
    pup_bin_all = np.zeros(len(ts), dtype=np.int16)
    rb, run_edges = quantile_bins(run_all[frame_idx])
    pb, pup_edges = quantile_bins(pupil_all[frame_idx])
    run_bin_all[frame_idx] = rb
    pup_bin_all[frame_idx] = pb

    # -- assemble trials -----------------------------------------------------
    t0 = time.time()
    neural, inputs, outputs = [], [], []
    for k in sel:
        a, b = i0[k], i1[k]
        T = b - a
        neural.append(np.ascontiguousarray(traces[:, a:b]))
        inputs.append(np.zeros((0, T), dtype=np.float32))
        out = np.empty((5, T), dtype=np.int16)
        out[0] = image_all[a:b]
        out[1] = change_all[a:b]
        out[2] = run_bin_all[a:b]
        out[3] = pup_bin_all[a:b]
        out[4] = outcome[k]
        outputs.append(out)
    timing['assemble'] = time.time() - t0

    md = ds.metadata
    region = str(md['targeted_structure'])
    brain_region_name = [region] * len(cell_ids)

    info = {
        'ophys_experiment_id': int(oeid),
        'ophys_session_id': int(md['ophys_session_id']),
        'behavior_session_id': int(md['behavior_session_id']),
        'mouse_id': str(md['mouse_id']),
        'cre_line': str(md['cre_line']),
        'session_type': str(md['session_type']),
        'targeted_structure': region,
        'imaging_depth': int(md['imaging_depth']),
        'equipment_name': str(md['equipment_name']),
        'ophys_frame_rate': float(md['ophys_frame_rate']),
        'frame_period_s': float(np.median(np.diff(ts))),
        'cell_specimen_ids': [int(c) for c in cell_ids],
        'trial_ids': [int(x) for x in tr.index.values[sel]],
        'n_neurons': int(len(cell_ids)),
        'n_trials': int(len(sel)),
        'n_trials_go_catch': int(len(tr)),
        'n_trials_dropped_short': int(n_drop_short),
        'n_trials_dropped_no_pupil': int(n_drop_pupil),
        'n_trials_all': int(len(trials)),
        'n_aborted': int(trials['aborted'].values.astype(bool).sum()),
        'n_auto_rewarded': int(trials['auto_rewarded'].values.astype(bool).sum()),
        'n_catch': int(catch[keep][sel].sum()),
        'n_go': int(go[keep][sel].sum()),
        'running_quintile_edges': run_edges.tolist(),
        'pupil_quintile_edges': pup_edges.tolist(),
        'pupil_blink_fraction': float(np.mean(~np.isfinite(eye['pupil_area'].values))),
        'mean_change_latency_s': float(np.nanmean(
            tr['change_time'].values[sel] - starts[sel])),
        'total_timepoints': int(sum(x.shape[1] for x in neural)),
    }

    if show:
        try:
            make_processing_plot(oeid, ds, ts, traces, run_all, pupil_all, image_all,
                                 change_all, run_bin_all, pup_bin_all, run_edges,
                                 pup_edges, tr, sel, i0, i1, outcome, sp, frame_idx)
        except Exception as exc:                      # plotting must not kill a run
            print(f'  [warn] plotting failed for {oeid}: {exc!r}', flush=True)

    timing['total'] = time.time() - t_start
    return {
        'oeid': int(oeid),
        'neural': neural,
        'input': inputs,
        'output': outputs,
        'brain_region_name': brain_region_name,
        'subject': str(md['mouse_id']),
        'info': info,
        'timing': timing,
    }


# ---------------------------------------------------------------------------
# diagnostic plots
# ---------------------------------------------------------------------------
def make_processing_plot(oeid, ds, ts, traces, run_all, pupil_all, image_all,
                         change_all, run_bin_all, pup_bin_all, run_edges,
                         pup_edges, tr, sel, i0, i1, outcome, sp, frame_idx):
    """Visualise every processing step for one session."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(26, 30))
    gs = fig.add_gridspec(7, 2, height_ratios=[1.2, 1.2, 1.2, 1.2, 1.0, 1.0, 1.4])

    # pick an example trial that is a hit if possible
    k_ex = sel[np.argmax(outcome[sel] == 0)] if np.any(outcome[sel] == 0) else sel[0]
    a, b = i0[k_ex], i1[k_ex]
    t_lo, t_hi = ts[a] - 1.0, ts[b - 1] + 1.0

    raw_run_t = ds.running_speed['timestamps'].values
    raw_run_v = ds.running_speed['speed'].values
    eye = ds.eye_tracking
    raw_eye_t = eye['timestamps'].values
    raw_eye_d = 2.0 * np.sqrt(eye['pupil_area'].values / np.pi)

    def shade_stim(ax):
        m = (sp['end_time'].values > t_lo) & (sp['start_time'].values < t_hi)
        for s, e, nm, om in zip(sp['start_time'].values[m], sp['end_time'].values[m],
                                sp['image_name'].astype(str).values[m],
                                sp['omitted'].values.astype(bool)[m]):
            if om:
                ax.axvspan(s, e, color='k', alpha=0.06, hatch='//')
            else:
                ax.axvspan(s, e, color=plt.cm.tab20(IMAGE_CODE.get(nm, 0) % 20), alpha=0.30)
        ax.axvline(tr['change_time'].values[k_ex], color='r', lw=2, ls='--')
        ax.axvline(ts[a], color='k', lw=1.5)
        ax.axvline(ts[b - 1], color='k', lw=1.5)

    # 1. running speed: raw 60 Hz vs resampled to ophys frames
    ax = fig.add_subplot(gs[0, :])
    m = (raw_run_t >= t_lo) & (raw_run_t <= t_hi)
    ax.plot(raw_run_t[m], raw_run_v[m], '-', color='gray', lw=2, label='raw 60 Hz running speed')
    mo = (ts >= t_lo) & (ts <= t_hi)
    ax.plot(ts[mo], run_all[mo], '.-', color='C0', ms=6, lw=0.8,
            label='resampled onto ophys frames')
    shade_stim(ax)
    ax.legend(loc='upper right')
    ax.set_ylabel('cm/s')
    ax.set_title(f'{oeid}: running speed alignment (shaded = image flashes, '
                 f'hatched = omitted, red dashed = change time, black = trial bounds)')

    # 2. pupil: raw 30 Hz samples (with blinks as gaps) vs interpolated
    ax = fig.add_subplot(gs[1, :])
    m = (raw_eye_t >= t_lo) & (raw_eye_t <= t_hi)
    ax.plot(raw_eye_t[m], raw_eye_d[m], 'o', color='gray', ms=5,
            label='raw 30 Hz pupil diameter (gaps = blinks)')
    ax.plot(ts[mo], pupil_all[mo], '.-', color='C1', ms=6, lw=0.8,
            label='blink-interpolated, resampled onto ophys frames')
    shade_stim(ax)
    ax.legend(loc='upper right')
    ax.set_ylabel('pupil diameter (px)')
    ax.set_title('pupil alignment')

    # 3. image identity trace vs the stimulus table
    ax = fig.add_subplot(gs[2, :])
    ax.step(ts[mo], image_all[mo], where='post', color='C2', lw=2, label='image_identity code')
    ax.step(ts[mo], change_all[mo] * 17, where='post', color='r', lw=2, label='image_change x17')
    shade_stim(ax)
    ax.set_yticks(range(len(IMAGE_VALUES)))
    ax.set_yticklabels(IMAGE_VALUES, fontsize=7)
    ax.legend(loc='upper right')
    ax.set_title('image identity / change vs. stimulus_presentations spans '
                 '(code must be non-grey exactly inside a shaded, non-hatched flash)')

    # 4. neural activity for the same window
    ax = fig.add_subplot(gs[3, :])
    nsh = min(40, traces.shape[0])
    sub = traces[:nsh][:, mo]
    ax.imshow(sub, aspect='auto', origin='lower', interpolation='nearest',
              extent=[ts[mo][0], ts[mo][-1], 0, nsh],
              cmap='magma', vmax=np.percentile(sub, 99.5) if sub.size else 1)
    ax.axvline(tr['change_time'].values[k_ex], color='c', lw=2, ls='--')
    ax.axvline(ts[a], color='w', lw=1.5)
    ax.axvline(ts[b - 1], color='w', lw=1.5)
    ax.set_ylabel('neuron')
    ax.set_xlabel('session time (s)')
    ax.set_title('neural activity (dF/F) on the ophys clock (same window)')

    # 5. quintile discretisation
    ax = fig.add_subplot(gs[4, 0])
    ax.hist(run_all[frame_idx], bins=200, color='C0')
    for e in run_edges:
        ax.axvline(e, color='k', ls='--')
    ax.set_yscale('log')
    ax.set_title('running speed distribution + quintile edges')
    ax.set_xlabel('cm/s')

    ax = fig.add_subplot(gs[4, 1])
    ax.hist(pupil_all[frame_idx], bins=200, color='C1')
    for e in pup_edges:
        ax.axvline(e, color='k', ls='--')
    ax.set_yscale('log')
    ax.set_title('pupil diameter distribution + quintile edges')
    ax.set_xlabel('px')

    ax = fig.add_subplot(gs[5, 0])
    cnt = np.bincount(run_bin_all[frame_idx], minlength=N_QUANTILE_BINS) / len(frame_idx)
    ax.bar(np.arange(N_QUANTILE_BINS), cnt, color='C0')
    ax.axhline(1 / N_QUANTILE_BINS, color='k', ls='--')
    ax.set_ylim(0, 0.35)
    ax.set_title('fraction of timepoints per running quintile (target 0.2)')

    ax = fig.add_subplot(gs[5, 1])
    cnt = np.bincount(pup_bin_all[frame_idx], minlength=N_QUANTILE_BINS) / len(frame_idx)
    ax.bar(np.arange(N_QUANTILE_BINS), cnt, color='C1')
    ax.axhline(1 / N_QUANTILE_BINS, color='k', ls='--')
    ax.set_ylim(0, 0.35)
    ax.set_title('fraction of timepoints per pupil quintile (target 0.2)')

    # 6. the five converted outputs for three consecutive kept trials
    ax = fig.add_subplot(gs[6, :])
    off = 0
    for k in sel[:3]:
        aa, bb = i0[k], i1[k]
        tt = np.arange(bb - aa) + off
        ax.plot(tt, image_all[aa:bb] / 16.0 + 4.2, color='C2')
        ax.plot(tt, change_all[aa:bb] + 3.1, color='r')
        ax.plot(tt, run_bin_all[aa:bb] / 4.0 + 2.1, color='C0')
        ax.plot(tt, pup_bin_all[aa:bb] / 4.0 + 1.1, color='C1')
        ax.plot(tt, np.full(bb - aa, outcome[k] / 3.0 + 0.1), color='C4')
        ax.axvline(off, color='k', lw=1)
        # change time in samples
        ct = tr['change_time'].values[k]
        ax.axvline(off + np.searchsorted(ts[aa:bb], ct), color='r', ls='--', lw=1)
        off += bb - aa
    ax.set_yticks([0.35, 1.35, 2.35, 3.35, 4.45])
    ax.set_yticklabels(OUTPUT_NAMES[::-1][::-1][4::-1])
    ax.set_yticklabels(['trial_outcome', 'pupil_bin', 'running_bin',
                        'image_change', 'image_identity'])
    ax.set_xlabel('timepoint (concatenated over 3 kept trials)')
    ax.set_title('converted outputs (red dashed = change time from the trials table)')

    fig.tight_layout()
    fig.savefig(f'/app/processing_{oeid}.png', dpi=90)
    plt.close(fig)
    print(f'  wrote /app/processing_{oeid}.png', flush=True)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('outfile', type=str)
    ap.add_argument('--full', action='store_true', default=True)
    ap.add_argument('--sample', action='store_true')
    ap.add_argument('--show-processing', action='store_true')
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--neural-signal', type=str, default='dff',
                    choices=['dff', 'events', 'filtered_events'])
    ap.add_argument('--zscore', action='store_true')
    ap.add_argument('--change-window', type=str, default='flash',
                    choices=['flash', 'interval'])
    ap.add_argument('--limit', type=int, default=0,
                    help='debug: process only the first N selected sessions')
    args = ap.parse_args()

    t_all = time.time()
    sel = select_experiments()
    print(f'Selected {len(sel)} active {PROJECT_CODE} experiments '
          f'from {sel.mouse_id.nunique()} mice', flush=True)
    if args.sample:
        sel = sel.iloc[:2]
        print(f'--sample: restricting to {len(sel)} sessions', flush=True)
    elif args.limit:
        sel = sel.iloc[::max(1, len(sel) // args.limit)].iloc[:args.limit]
        print(f'--limit: restricting to {len(sel)} sessions', flush=True)

    jobs = []
    for i, row in sel.iterrows():
        jobs.append({'ophys_experiment_id': int(row.ophys_experiment_id),
                     'neural_signal': args.neural_signal,
                     'zscore': bool(args.zscore),
                     'change_window': args.change_window,
                     'show': bool(args.show_processing and i < 2)})

    results = []
    nworkers = max(1, min(args.workers, len(jobs)))
    t0 = time.time()
    with ProcessPoolExecutor(nworkers) as ex:
        for n, res in enumerate(ex.map(process_experiment, jobs), start=1):
            results.append(res)
            if 'skip' in res:
                print(f'[{n}/{len(jobs)}] {res["oeid"]}: SKIPPED ({res["skip"]})', flush=True)
            else:
                tm = res['timing']
                print(f'[{n}/{len(jobs)}] {res["oeid"]}: '
                      f'{res["info"]["n_neurons"]} neurons, {res["info"]["n_trials"]} trials, '
                      f'{res["info"]["total_timepoints"]} timepoints  '
                      f'(open {tm["open"]:.1f}s neural {tm["neural"]:.1f}s '
                      f'behav {tm["behavior"]:.1f}s asm {tm["assemble"]:.1f}s '
                      f'tot {tm["total"]:.1f}s)', flush=True)
    t_proc = time.time() - t0
    print(f'\nProcessing wall time: {t_proc:.1f}s for {len(jobs)} sessions '
          f'({t_proc / max(1, len(jobs)):.2f}s/session with {nworkers} workers)', flush=True)

    kept = [r for r in results if 'skip' not in r]
    skipped = [r for r in results if 'skip' in r]
    print(f'Kept {len(kept)} sessions, skipped {len(skipped)}', flush=True)

    # ---- assemble the target structure ------------------------------------
    subjects = sorted({r['subject'] for r in kept})
    subject_index = {s: i for i, s in enumerate(subjects)}
    regions = sorted({n for r in kept for n in set(r['brain_region_name'])})
    region_index = {s: i for i, s in enumerate(regions)}

    data = {
        'neural': [r['neural'] for r in kept],
        'input': [r['input'] for r in kept],
        'output': [r['output'] for r in kept],
        'subjects': subjects,
        'subject_idx': np.array([subject_index[r['subject']] for r in kept], dtype=np.int64),
        'brain_regions': regions,
        'brain_region_idx': [np.array([region_index[n] for n in r['brain_region_name']],
                                      dtype=np.int64) for r in kept],
        'input_names': [],
        'output_names': OUTPUT_NAMES,
        'output_values': OUTPUT_VALUES,
    }

    frame_periods = np.array([r['info']['frame_period_s'] for r in kept])
    n_trials = sum(len(r['neural']) for r in kept)
    n_tp = sum(r['info']['total_timepoints'] for r in kept)
    n_neurons = sum(r['info']['n_neurons'] for r in kept)

    data['metadata'] = {
        'dataset': 'Allen Brain Observatory - Visual Behavior 2P (visual-behavior-ophys-1.1.0)',
        'project_code': PROJECT_CODE,
        'task_description':
            'Head-fixed mice perform a go/no-go visual change-detection task. Natural images '
            'are flashed for 250 ms every 750 ms (500 ms grey inter-stimulus interval, 5% of '
            'flashes omitted); the mouse earns water by licking within 150-750 ms of a change '
            'in image identity. Each trial is either a GO trial (the image changes) or a CATCH '
            'trial (a sham change: the same image is re-presented); aborted trials (licks '
            'before the change) and auto-rewarded (free-reward) trials are excluded. Decoded '
            'from 2-photon calcium activity in visual cortex: (0) identity of the image on '
            'screen (grey screen = its own class), (1) whether the current flash is the image '
            'change, (2) running speed quintile, (3) pupil diameter quintile, (4) the trial '
            'outcome (hit / miss / false alarm / correct reject).',
        'time_bin_size': float(np.median(frame_periods) * 1000.0),   # ms
        'time_bin_size_range_ms': [float(frame_periods.min() * 1000),
                                   float(frame_periods.max() * 1000)],
        'temporal_alignment_event':
            'Start of each change-detection trial (trials.start_time). All data streams are '
            'resampled onto the 2-photon frame times (ophys_timestamps) that fall inside '
            '[trials.start_time, trials.stop_time).',
        'off_start': 0.0,
        'off_end': None,
        'trial_duration_s_note':
            'Trials keep their natural, experiment-defined length (7.26-12.56 s); the image '
            'change occurs 3.0-8.3 s (mean 4.3 s) after trial start and the trial ends 4.23 s '
            'after the change, so off_end is not a single number.',
        'neural_signal': args.neural_signal,
        'neural_signal_description':
            'AllenSDK dff_traces.dff: neuropil-subtracted, demixed, baseline-normalised and '
            'detrended dF/F for every valid ROI, as released with the dataset (whitepaper '
            '"DF/F CALCULATION"). The detected calcium events used by the reference paper '
            '(events / filtered_events) were benchmarked against dF/F and decoded worse on '
            'every output, so dF/F was kept for the decoding task.',
        'sampling_rate_hz': float(1.0 / np.median(frame_periods)),
        'n_sessions': len(kept),
        'n_subjects': len(subjects),
        'n_trials': int(n_trials),
        'n_neurons': int(n_neurons),
        'n_timepoints': int(n_tp),
        'running_speed_units': 'cm/s (quintile-binned per session)',
        'pupil_units': 'effective pupil diameter 2*sqrt(pupil_area/pi) in camera pixels '
                       '(quintile-binned per session)',
        'exclusions':
            'Excluded: the VisualBehaviorMultiscope experiments (different ophys frame rate, '
            '10.7 Hz vs 30.94 Hz, would break the uniform time bin; only 1 mouse); passive '
            'sessions (not analysed in the reference paper, and their trial outcome is '
            'degenerate because the lick spout is retracted); sessions with no eye-tracking '
            'data; trials with no valid (non-blink) pupil sample.',
        'session_info': [r['info'] for r in kept],
        'skipped_sessions': [{'oeid': r['oeid'], 'reason': r['skip']} for r in skipped],
    }

    # ---- summary + sanity checks ------------------------------------------
    print('\n--- converted dataset ---')
    print(f'sessions      : {len(kept)}')
    print(f'subjects      : {len(subjects)}')
    print(f'brain regions : {regions}')
    print(f'neurons       : {n_neurons} (mean {n_neurons / len(kept):.1f}/session)')
    print(f'trials        : {n_trials} (mean {n_trials / len(kept):.1f}/session)')
    print(f'timepoints    : {n_tp}')
    print(f'time bin      : {data["metadata"]["time_bin_size"]:.4f} ms '
          f'(range {data["metadata"]["time_bin_size_range_ms"]})')
    dropped_pupil = sum(r['info']['n_trials_dropped_no_pupil'] for r in kept)
    dropped_short = sum(r['info']['n_trials_dropped_short'] for r in kept)
    print(f'trials dropped: {dropped_pupil} (no pupil sample), {dropped_short} (too short)')

    allout = np.concatenate([o for r in kept for o in r['output']], axis=1)
    for i, name in enumerate(OUTPUT_NAMES):
        cnt = np.bincount(allout[i], minlength=len(OUTPUT_VALUES[i]))
        frac = cnt / cnt.sum()
        print(f'  {name}: ' + ', '.join(f'{v}={f:.4f}'
                                        for v, f in zip(OUTPUT_VALUES[i], frac)))
    # per-trial outcome fractions
    tro = np.array([r['output'][t][4, 0] for r in kept for t in range(len(r['output']))])
    print('  trial_outcome (per trial): ' +
          ', '.join(f'{v}={np.mean(tro == i):.4f}' for i, v in enumerate(OUTCOME_VALUES)))
    ncatch = sum(r['info']['n_catch'] for r in kept)
    print(f'  catch fraction of kept trials: {ncatch / n_trials:.4f} (whitepaper: ~0.125)')
    mcl = np.mean([r['info']['mean_change_latency_s'] for r in kept])
    print(f'  mean change latency after trial start: {mcl:.3f}s (whitepaper: ~4.2s)')

    assert np.isclose(frame_periods.min(), frame_periods.max(), atol=1e-4), \
        'frame period is not uniform across sessions'

    # ---- write -------------------------------------------------------------
    t0 = time.time()
    with open(args.outfile, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'\nWrote {args.outfile} '
          f'({os.path.getsize(args.outfile) / 1e9:.2f} GB) in {time.time() - t0:.1f}s')
    print(f'Total wall time: {time.time() - t_all:.1f}s')


if __name__ == '__main__':
    main()
