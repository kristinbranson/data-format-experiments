#!/usr/bin/env python3
"""
Convert the Allen Brain Observatory *Visual Behavior 2P* dataset into the
decoder-training format described in the task specification.

Usage
-----
    python -u /app/convert_data.py <outpicklefile> [--full | --sample] [--show-processing]

Options
-------
    --full             process every selected session (default)
    --sample           process only the first 2 selected sessions
    --show-processing  save diagnostic plots (`processing_<session_id>.png`) for up to
                       2 sessions, showing every step of the conversion
    --workers N        number of parallel worker processes (default 24)

Data selection / processing decisions are documented in /app/CONVERSION_NOTES.md.
Everything is read through the AllenSDK `VisualBehaviorOphysProjectCache`; no NWB file
is ever opened directly.
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

warnings.filterwarnings('ignore')

CACHE_DIR = '/app/data'
NWB_DIR = os.path.join(CACHE_DIR, 'visual-behavior-ophys-1.1.0',
                       'behavior_ophys_experiments')

# ----------------------------------------------------------------------------------
# Fixed category vocabularies (global across sessions)
# ----------------------------------------------------------------------------------
# The 16 natural images used by the single-plane Visual Behavior variant:
# image set A = {im061, im062, im063, im065, im066, im069, im077, im085}
# image set B = {im000, im031, im035, im045, im054, im073, im075, im106}
IMAGE_NAMES = ['im000', 'im031', 'im035', 'im045', 'im054', 'im061', 'im062', 'im063',
               'im065', 'im066', 'im069', 'im073', 'im075', 'im077', 'im085', 'im106']
IMAGE_TO_IDX = {n: i for i, n in enumerate(IMAGE_NAMES)}

OUTCOME_NAMES = ['hit', 'miss', 'false_alarm', 'correct_reject']
CHANGE_NAMES = ['no_change', 'change']
QUINTILE_NAMES = ['Q1_lowest', 'Q2', 'Q3', 'Q4', 'Q5_highest']

OUTPUT_NAMES = ['image_identity', 'image_change', 'running_speed_bin',
                'pupil_diameter_bin', 'trial_outcome']
OUTPUT_VALUES = [IMAGE_NAMES, CHANGE_NAMES, QUINTILE_NAMES, QUINTILE_NAMES,
                 OUTCOME_NAMES]

N_QUANTILES = 5
QUANTILE_PCTS = [20., 40., 60., 80.]
STIM_BLOCK = 'change_detection_behavior'


# ----------------------------------------------------------------------------------
# Cache helpers
# ----------------------------------------------------------------------------------
def get_cache():
    """Open the local AllenSDK cache (no network access)."""
    from allensdk.brain_observatory.behavior.behavior_project_cache import (
        VisualBehaviorOphysProjectCache)
    return VisualBehaviorOphysProjectCache.from_local_cache(
        cache_dir=CACHE_DIR, use_static_cache=False)


def locally_available_experiment_ids():
    """ophys_experiment_ids whose NWB file is present in the local cache."""
    files = os.listdir(NWB_DIR)
    return sorted(int(re.findall(r'(\d+)', f)[0]) for f in files
                  if f.endswith('.nwb'))


def select_experiments(cache):
    """Apply the session-level selection criteria (see CONVERSION_NOTES.md Step 5).

    Returns the metadata rows of the selected experiments, ordered by experiment id.
    """
    et = cache.get_ophys_experiment_table()
    sub = et.loc[locally_available_experiment_ids()]
    # D1: single-plane "VisualBehavior" project variant only -> a single, constant
    #     ophys frame interval (30.95 Hz) for every session, as the target format
    #     requires, and one imaging plane (= one experiment) per session.
    # D2: active behaviour only -- passive sessions have the lick spout retracted, so
    #     go/catch/hit/miss trial structure does not exist there.
    sel = sub[(~sub.passive) & (sub.project_code == 'VisualBehavior')]
    return sel.sort_index()


# ----------------------------------------------------------------------------------
# Per-experiment conversion
# ----------------------------------------------------------------------------------
def build_stimulus_series(stimulus_presentations, ophys_timestamps):
    """Map every ophys frame onto the image-presentation interval that contains it.

    The Neuron reference paper assigns behavioural events to "the 750 ms interval
    beginning with each image presentation" (and, for omissions, the 750 ms following
    the time at which the image would have been shown).  We use exactly those
    intervals: interval i spans [start_time_i, start_time_{i+1}).

    Returns
    -------
    image_idx : (n_frames,) int64   index into IMAGE_NAMES of the image currently being
                                    shown / most recently shown.  Omitted flashes carry
                                    the identity of the preceding image forward, because
                                    an omission is a continuation of the gray screen in
                                    an otherwise unchanged image sequence.
    is_change : (n_frames,) int64   1 for every frame inside the presentation interval of
                                    a flash whose image identity differs from the previous
                                    one, 0 otherwise.
    interval  : (n_frames,) int64   index of the containing presentation interval.
    """
    sp = stimulus_presentations
    sp = sp[sp['stimulus_block_name'] == STIM_BLOCK].sort_values('start_time')
    start = sp['start_time'].to_numpy(dtype=float)
    names = sp['image_name'].to_numpy()
    change = sp['is_change'].to_numpy().astype(bool)

    raw = np.array([IMAGE_TO_IDX.get(n, -1) if isinstance(n, str) else -1
                    for n in names], dtype=np.int64)
    if raw[0] < 0:
        raise ValueError('first flash of the change-detection block is not an image')
    # forward-fill the omitted (-1) entries with the previous image identity
    valid_pos = np.where(raw >= 0)[0]
    fill_src = valid_pos[np.searchsorted(valid_pos,
                                         np.arange(len(raw)), side='right') - 1]
    filled = raw[fill_src]

    # index of the presentation interval containing each ophys frame.  Frames that
    # precede the very first flash (at most one frame: the first trial of a session can
    # begin a few ms before the first flash) are clamped onto interval 0.
    k = np.searchsorted(start, ophys_timestamps, side='right') - 1
    k = np.clip(k, 0, len(start) - 1)
    return filled[k], change[k].astype(np.int64), k


def pupil_diameter_series(eye_tracking):
    """Pupil diameter in pixels, with blink/outlier frames removed.

    AllenSDK computes `pupil_area = pi * max(pupil_width, pupil_height)**2`
    (`eye_tracking_processing.compute_circular_area`), i.e. it treats
    `max(width, height)` as the pupil *radius*.  The matching diameter is therefore
    `2 * max(width, height)`.  Frames flagged `likely_blink` (blinks and |z| > 3 area
    outliers, dilated by 2 frames) are NaN in the SDK output and are dropped here; they
    are filled back in by linear interpolation when the trace is resampled onto the
    ophys grid.
    """
    diam = 2.0 * np.maximum(eye_tracking['pupil_width'].to_numpy(dtype=float),
                            eye_tracking['pupil_height'].to_numpy(dtype=float))
    t = eye_tracking['timestamps'].to_numpy(dtype=float)
    good = np.isfinite(diam)
    return t, diam, good


def convert_experiment(ophys_experiment_id, collect_debug=False):
    """Convert one ophys experiment (= one session) into the target structure."""
    t_start = time.time()
    cache = get_cache()
    ds = cache.get_behavior_ophys_experiment(ophys_experiment_id)

    ts = np.asarray(ds.ophys_timestamps, dtype=float)      # (F,) session clock, s
    n_frames = ts.size

    # ---------------- neural ----------------------------------------------------
    # dF/F traces from the standard Allen pipeline (already neuropil-subtracted,
    # demixed, baseline-normalised and detrended -- see methods.txt "DF/F CALCULATION").
    # `dff_traces` contains one row per *valid* ROI only; no further neuron filtering is
    # available or needed (cell_specimen_table.valid_roi is True for every row).
    dff = ds.dff_traces
    cell_ids = dff.index.to_numpy()
    traces = np.vstack(dff['dff'].to_numpy()).astype(np.float32)   # (N, F)
    if traces.shape[1] != n_frames:
        raise ValueError(f'{ophys_experiment_id}: dff has {traces.shape[1]} samples '
                         f'but there are {n_frames} ophys timestamps')
    n_neurons = traces.shape[0]

    cst = ds.cell_specimen_table
    if not bool(cst['valid_roi'].all()):
        keep = cst['valid_roi'].to_numpy().astype(bool)
        traces = traces[keep]
        cell_ids = cell_ids[keep]
        n_neurons = traces.shape[0]

    # ---------------- trials ----------------------------------------------------
    trials = ds.trials
    # `go`, `catch`, `aborted` and `auto_rewarded` are mutually exclusive in the SDK
    # (allensdk/brain_observatory/behavior/data_objects/trials/trial.py), so `go | catch`
    # is exactly "Go and Catch trials, excluding Aborted and Auto-rewarded".
    sel = trials[trials['go'].to_numpy().astype(bool)
                 | trials['catch'].to_numpy().astype(bool)].sort_values('start_time')
    if len(sel) == 0:
        raise ValueError(f'{ophys_experiment_id}: no go/catch trials')

    outcome = np.full(len(sel), -1, dtype=np.int64)
    for j, name in enumerate(OUTCOME_NAMES):
        outcome[sel[name].to_numpy().astype(bool)] = j
    if np.any(outcome < 0):
        raise ValueError(f'{ophys_experiment_id}: unclassified go/catch trial')

    # ---------------- stimulus ---------------------------------------------------
    image_idx, is_change, interval = build_stimulus_series(ds.stimulus_presentations, ts)

    # ---------------- running speed ---------------------------------------------
    # 60 Hz encoder trace (already de-wrapped, transient-corrected and 10 Hz low-pass
    # filtered by the SDK) linearly resampled onto the ophys frame times.
    rs = ds.running_speed
    run_t = rs['timestamps'].to_numpy(dtype=float)
    run_v = rs['speed'].to_numpy(dtype=float)
    run_ok = np.isfinite(run_v)
    run_frames = np.interp(ts, run_t[run_ok], run_v[run_ok])

    # ---------------- pupil diameter ---------------------------------------------
    eye = ds.eye_tracking
    if eye is None or len(eye) == 0:
        raise ValueError(f'{ophys_experiment_id}: no eye tracking data')
    eye_t, eye_d, eye_ok = pupil_diameter_series(eye)
    if eye_ok.sum() < 100:
        raise ValueError(f'{ophys_experiment_id}: too few valid pupil frames')
    pupil_frames = np.interp(ts, eye_t[eye_ok], eye_d[eye_ok])

    # ---------------- trial windows ----------------------------------------------
    # A trial owns every ophys frame with start_time <= t < stop_time.
    a_idx = np.searchsorted(ts, sel['start_time'].to_numpy(dtype=float), side='left')
    b_idx = np.searchsorted(ts, sel['stop_time'].to_numpy(dtype=float), side='left')
    keep = (b_idx - a_idx) >= 2
    n_dropped_short = int((~keep).sum())
    a_idx, b_idx = a_idx[keep], b_idx[keep]
    outcome = outcome[keep]
    sel_kept = sel[keep]

    # ---------------- discretisation of the continuous behavioural outputs -------
    # Quintile edges are computed per session over exactly the frames that end up in
    # the converted data, so each session contributes ~20 % of its samples to each bin.
    # Per-session (rather than global) edges are required because pupil diameter is
    # measured in camera pixels and is not comparable across sessions/rigs; the same
    # convention is used for running speed so that the two discretisations match.
    frame_mask = np.zeros(n_frames, dtype=bool)
    for a, b in zip(a_idx, b_idx):
        frame_mask[a:b] = True
    run_edges = np.percentile(run_frames[frame_mask], QUANTILE_PCTS)
    pupil_edges = np.percentile(pupil_frames[frame_mask], QUANTILE_PCTS)
    run_bin = np.searchsorted(run_edges, run_frames, side='right').astype(np.int64)
    pupil_bin = np.searchsorted(pupil_edges, pupil_frames, side='right').astype(np.int64)

    # ---------------- assemble trials --------------------------------------------
    neural, output, inputs = [], [], []
    for i, (a, b) in enumerate(zip(a_idx, b_idx)):
        T = b - a
        neural.append(np.ascontiguousarray(traces[:, a:b]))
        out = np.empty((len(OUTPUT_NAMES), T), dtype=np.int64)
        out[0] = image_idx[a:b]
        out[1] = is_change[a:b]
        out[2] = run_bin[a:b]
        out[3] = pupil_bin[a:b]
        out[4] = outcome[i]
        output.append(out)
        inputs.append(np.zeros((0, T), dtype=np.float32))

    md = ds.metadata
    info = {
        'ophys_experiment_id': int(ophys_experiment_id),
        'ophys_session_id': int(md['ophys_session_id']),
        'behavior_session_id': int(md['behavior_session_id']),
        'mouse_id': str(md['mouse_id']),
        'cre_line': str(md['cre_line']),
        'targeted_structure': str(md['targeted_structure']),
        'imaging_depth': int(md['imaging_depth']),
        'session_type': str(md['session_type']),
        'equipment_name': str(md['equipment_name']),
        'project_code': str(md['project_code']),
        'n_neurons': int(n_neurons),
        'n_trials': int(len(neural)),
        'n_trials_dropped_short': n_dropped_short,
        'ophys_frame_interval_s': float(np.median(np.diff(ts))),
        'running_speed_quintile_edges': run_edges.tolist(),
        'pupil_diameter_quintile_edges': pupil_edges.tolist(),
        'pupil_blink_fraction': float(1.0 - eye_ok.mean()),
        'frac_hit': float(np.mean(outcome == 0)),
        'frac_miss': float(np.mean(outcome == 1)),
        'frac_false_alarm': float(np.mean(outcome == 2)),
        'frac_correct_reject': float(np.mean(outcome == 3)),
        'convert_seconds': time.time() - t_start,
    }

    result = {
        'neural': neural,
        'input': inputs,
        'output': output,
        'brain_region': str(md['targeted_structure']),
        'mouse_id': str(md['mouse_id']),
        'n_neurons': n_neurons,
        'cell_specimen_ids': cell_ids,
        'info': info,
    }

    if collect_debug:
        result['debug'] = {
            'ts': ts, 'traces_mean': traces.mean(axis=0),
            'traces': traces[:min(30, n_neurons)],
            'image_idx': image_idx, 'is_change': is_change,
            'run_frames': run_frames, 'pupil_frames': pupil_frames,
            'run_bin': run_bin, 'pupil_bin': pupil_bin,
            'run_edges': run_edges, 'pupil_edges': pupil_edges,
            'run_t': run_t, 'run_v': run_v,
            'eye_t': eye_t, 'eye_d': eye_d, 'eye_ok': eye_ok,
            'a_idx': a_idx, 'b_idx': b_idx, 'outcome': outcome,
            'trial_start': sel_kept['start_time'].to_numpy(dtype=float),
            'trial_stop': sel_kept['stop_time'].to_numpy(dtype=float),
            'change_time': sel_kept['change_time'].to_numpy(dtype=float),
            'is_go': sel_kept['go'].to_numpy().astype(bool),
            'frame_mask': frame_mask,
            'stim': ds.stimulus_presentations[
                ds.stimulus_presentations['stimulus_block_name'] == STIM_BLOCK][
                    ['start_time', 'end_time', 'image_name', 'omitted', 'is_change']],
            'licks': ds.licks['timestamps'].to_numpy(dtype=float),
            'rewards': ds.rewards['timestamps'].to_numpy(dtype=float),
        }
    return result


def _worker(args):
    eid, collect_debug = args
    try:
        return convert_experiment(eid, collect_debug=collect_debug)
    except Exception as exc:                                   # noqa: BLE001
        return {'error': f'{type(exc).__name__}: {exc}', 'eid': int(eid)}


# ----------------------------------------------------------------------------------
# Diagnostic plots
# ----------------------------------------------------------------------------------
def plot_processing(res, out_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    d = res['debug']
    ts = d['ts']
    a_idx, b_idx = d['a_idx'], d['b_idx']
    eid = res['info']['ophys_experiment_id']

    fig = plt.figure(figsize=(26, 30))
    gs = fig.add_gridspec(7, 3, height_ratios=[1, 1, 1.4, 1.4, 1, 1, 1], hspace=0.45,
                          wspace=0.22)

    # ---- row 0: whole-session raw streams -------------------------------------
    ax = fig.add_subplot(gs[0, :])
    ax.plot(ts, d['traces_mean'], lw=0.3, color='k')
    for a, b in zip(a_idx[:400], b_idx[:400]):
        ax.axvspan(ts[a], ts[b - 1], color='orange', alpha=0.15, lw=0)
    ax.set_title(f'[1] Session {eid}: population-mean dF/F (black) with selected '
                 f'go/catch trial windows (orange)')
    ax.set_xlabel('session time (s)')
    ax.set_ylabel('mean dF/F')

    ax = fig.add_subplot(gs[1, :])
    ax.plot(d['run_t'], d['run_v'], lw=0.3, color='0.6', label='raw 60 Hz running speed')
    ax.plot(ts, d['run_frames'], lw=0.3, color='C0',
            label='resampled onto ophys frames')
    ax.set_ylabel('running speed (cm/s)')
    ax.set_xlabel('session time (s)')
    ax.legend(loc='upper right')
    ax.set_title('[2] Running speed: raw stream vs. resampled onto the ophys timestamps')

    # ---- row 2: zoom on one trial ---------------------------------------------
    # pick a hit trial and a miss/catch trial to show
    def zoom_axes(row, trial_i, title):
        a, b = a_idx[trial_i], b_idx[trial_i]
        t0, t1 = ts[a], ts[b - 1]
        axn = fig.add_subplot(gs[row, 0])
        sub = d['traces'][:, a:b]
        axn.imshow(sub, aspect='auto', interpolation='nearest',
                   extent=[t0, t1, sub.shape[0], 0], cmap='magma')
        axn.set_title(f'{title}\nneural (dF/F, first {sub.shape[0]} cells)')
        axn.set_ylabel('cell')
        axn.set_xlabel('time (s)')

        axo = fig.add_subplot(gs[row, 1])
        stim = d['stim']
        m = (stim['end_time'] >= t0 - 1) & (stim['start_time'] <= t1 + 1)
        for _, s in stim[m].iterrows():
            if bool(s['omitted']):
                continue
            col = plt.cm.tab20(IMAGE_TO_IDX.get(s['image_name'], 0) % 20)
            axo.axvspan(s['start_time'], s['end_time'], color=col, alpha=0.6, lw=0)
            if bool(s['is_change']):
                axo.axvline(s['start_time'], color='r', lw=2)
        axo.step(ts[a:b], d['image_idx'][a:b], where='post', color='k', lw=1.5,
                 label='output image_identity')
        axo.step(ts[a:b], d['is_change'][a:b] * 15, where='post', color='r', lw=1.5,
                 label='output image_change (x15)')
        for lk in d['licks'][(d['licks'] >= t0) & (d['licks'] <= t1)]:
            axo.plot(lk, -1, 'k|', ms=8)
        for rw in d['rewards'][(d['rewards'] >= t0) & (d['rewards'] <= t1)]:
            axo.plot(rw, -1.5, 'bd', ms=6)
        axo.set_ylim(-2, 16)
        axo.set_xlim(t0, t1)
        axo.legend(loc='upper left', fontsize=8)
        axo.set_title('stimulus flashes (colour = image) vs. converted outputs\n'
                      'red line = image change; ticks = licks, diamonds = rewards')
        axo.set_xlabel('time (s)')

        axb = fig.add_subplot(gs[row, 2])
        axb.plot(ts[a:b], d['run_frames'][a:b], color='C0', label='running speed (cm/s)')
        axb.step(ts[a:b], d['run_bin'][a:b] * 5, where='post', color='C0', ls='--',
                 label='running quintile bin (x5)')
        axb2 = axb.twinx()
        axb2.plot(ts[a:b], d['pupil_frames'][a:b], color='C3',
                  label='pupil diameter (px)')
        axb2.step(ts[a:b], d['pupil_bin'][a:b] * 10 + 10, where='post', color='C3',
                  ls='--', label='pupil quintile bin')
        axb.legend(loc='upper left', fontsize=8)
        axb2.legend(loc='upper right', fontsize=8)
        axb.set_xlim(t0, t1)
        axb.set_title(f'behaviour + discretisation  (outcome = '
                      f'{OUTCOME_NAMES[d["outcome"][trial_i]]})')
        axb.set_xlabel('time (s)')

    go_trials = np.where(d['is_go'])[0]
    catch_trials = np.where(~d['is_go'])[0]
    zoom_axes(2, int(go_trials[len(go_trials) // 3]), '[3] Example GO trial')
    zoom_axes(3, int(catch_trials[len(catch_trials) // 2]) if len(catch_trials)
              else int(go_trials[0]), '[4] Example CATCH trial')

    # ---- row 4: discretisation --------------------------------------------------
    mask = d['frame_mask']
    ax = fig.add_subplot(gs[4, 0])
    ax.hist(d['run_frames'][mask], bins=200, color='C0')
    for e in d['run_edges']:
        ax.axvline(e, color='k', ls='--')
    ax.set_yscale('log')
    ax.set_title('[5] running speed within selected trials\n(dashed = quintile edges)')
    ax.set_xlabel('cm/s')

    ax = fig.add_subplot(gs[4, 1])
    ax.hist(d['pupil_frames'][mask], bins=200, color='C3')
    for e in d['pupil_edges']:
        ax.axvline(e, color='k', ls='--')
    ax.set_title('[6] pupil diameter within selected trials\n(dashed = quintile edges)')
    ax.set_xlabel('pixels')

    ax = fig.add_subplot(gs[4, 2])
    counts = [np.sum(d['run_bin'][mask] == i) for i in range(N_QUANTILES)]
    counts2 = [np.sum(d['pupil_bin'][mask] == i) for i in range(N_QUANTILES)]
    x = np.arange(N_QUANTILES)
    ax.bar(x - 0.2, np.array(counts) / mask.sum(), 0.4, label='running')
    ax.bar(x + 0.2, np.array(counts2) / mask.sum(), 0.4, label='pupil')
    ax.axhline(0.2, color='k', ls='--')
    ax.set_xticks(x)
    ax.set_xticklabels(QUINTILE_NAMES, rotation=30)
    ax.legend()
    ax.set_title('[7] fraction of converted samples per quintile bin\n'
                 '(should be 0.2 each)')

    # ---- row 5: blink handling and eye resampling ------------------------------
    ax = fig.add_subplot(gs[5, :])
    seg = (d['eye_t'] > d['trial_start'][0] - 5) & (d['eye_t'] < d['trial_start'][0] + 60)
    ax.plot(d['eye_t'][seg], d['eye_d'][seg], '.', ms=3, color='0.6',
            label='raw pupil diameter (NaN on likely_blink frames)')
    bad = seg & ~d['eye_ok']
    ax.plot(d['eye_t'][bad], np.zeros(bad.sum()), 'rx', ms=5,
            label='blink / outlier frames (dropped)')
    seg2 = (ts > d['trial_start'][0] - 5) & (ts < d['trial_start'][0] + 60)
    ax.plot(ts[seg2], d['pupil_frames'][seg2], '-', lw=1, color='C3',
            label='interpolated onto ophys frames')
    ax.legend(loc='upper right')
    ax.set_title('[8] pupil: blink removal + interpolation onto the ophys timestamps '
                 '(60 s excerpt)')
    ax.set_xlabel('session time (s)')
    ax.set_ylabel('diameter (px)')

    # ---- row 6: change-triggered averages (alignment sanity check) -------------
    ax = fig.add_subplot(gs[6, 0])
    dt = res['info']['ophys_frame_interval_s']
    win = int(round(2.0 / dt))
    chg_times = d['change_time'][d['is_go']]
    chg_times = chg_times[np.isfinite(chg_times)]
    k = np.searchsorted(ts, chg_times)
    k = k[(k > win) & (k < len(ts) - win)]
    if len(k):
        seg = np.stack([d['traces_mean'][i - win:i + win] for i in k])
        tt = (np.arange(-win, win)) * dt
        ax.plot(tt, seg.mean(axis=0), color='k')
        ax.axvline(0, color='r')
    ax.set_title('[9] change-triggered mean dF/F\n(peak must follow t=0)')
    ax.set_xlabel('time from image change (s)')

    ax = fig.add_subplot(gs[6, 1])
    allout = np.concatenate(res['output'], axis=1)
    ax.bar(np.arange(len(IMAGE_NAMES)),
           [np.mean(allout[0] == i) for i in range(len(IMAGE_NAMES))])
    ax.set_xticks(np.arange(len(IMAGE_NAMES)))
    ax.set_xticklabels(IMAGE_NAMES, rotation=90)
    ax.set_title('[10] image_identity distribution in converted data')

    ax = fig.add_subplot(gs[6, 2])
    ax.bar(np.arange(4), [np.mean(allout[4] == i) for i in range(4)])
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(OUTCOME_NAMES, rotation=30)
    ax.set_title(f'[11] trial_outcome distribution (frac change='
                 f'{allout[1].mean():.3f})')

    fig.suptitle(f'Conversion diagnostics: ophys_experiment_id {eid} '
                 f'({res["info"]["session_type"]}, {res["info"]["cre_line"]}, '
                 f'{res["info"]["targeted_structure"]}, mouse {res["info"]["mouse_id"]})',
                 fontsize=16)
    fig.savefig(out_path, dpi=90, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {out_path}', flush=True)


# ----------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('outfile')
    ap.add_argument('--full', action='store_true', default=True)
    ap.add_argument('--sample', action='store_true')
    ap.add_argument('--show-processing', action='store_true')
    ap.add_argument('--workers', type=int, default=24)
    args = ap.parse_args()

    t_all = time.time()
    cache = get_cache()
    meta = select_experiments(cache)
    eids = list(meta.index)
    print(f'{len(eids)} experiments selected '
          f'(active behaviour, project_code=VisualBehavior, locally available)')
    if args.sample:
        eids = eids[:2]
        print(f'--sample: restricting to {eids}')
    n_debug = 2 if args.show_processing else 0
    print(f'loading with {args.workers} workers ...', flush=True)

    t0 = time.time()
    jobs = [(eid, i < n_debug) for i, eid in enumerate(eids)]
    results = []
    with Pool(min(args.workers, max(1, len(jobs)))) as pool:
        for i, res in enumerate(pool.imap(_worker, jobs, chunksize=1)):
            if 'error' in res:
                print(f'  [{i + 1}/{len(jobs)}] SKIP {res["eid"]}: {res["error"]}',
                      flush=True)
            else:
                nfo = res['info']
                print(f'  [{i + 1}/{len(jobs)}] {nfo["ophys_experiment_id"]} '
                      f'{nfo["session_type"]:<18s} mouse {nfo["mouse_id"]:>7s} '
                      f'neurons {nfo["n_neurons"]:>4d} trials {nfo["n_trials"]:>4d} '
                      f'({nfo["convert_seconds"]:.1f}s)', flush=True)
            results.append(res)
    t_load = time.time() - t0
    print(f'conversion of {len(jobs)} sessions took {t_load:.1f}s '
          f'({t_load / max(1, len(jobs)):.2f}s/session wall clock)')

    ok = [r for r in results if 'error' not in r]
    skipped = [r for r in results if 'error' in r]

    if args.show_processing:
        for res in ok[:n_debug]:
            plot_processing(
                res, f'processing_{res["info"]["ophys_experiment_id"]}.png')

    # ---- assemble the final dictionary -----------------------------------------
    subjects, brain_regions = [], []
    for r in ok:
        if r['mouse_id'] not in subjects:
            subjects.append(r['mouse_id'])
    for r in ok:
        if r['brain_region'] not in brain_regions:
            brain_regions.append(r['brain_region'])

    data = {
        'neural': [r['neural'] for r in ok],
        'input': [r['input'] for r in ok],
        'output': [r['output'] for r in ok],
        'subjects': subjects,
        'subject_idx': np.array([subjects.index(r['mouse_id']) for r in ok],
                                dtype=np.int64),
        'brain_regions': brain_regions,
        'brain_region_idx': [
            np.full(r['n_neurons'], brain_regions.index(r['brain_region']),
                    dtype=np.int64) for r in ok],
        'input_names': [],
        'output_names': OUTPUT_NAMES,
        'output_values': OUTPUT_VALUES,
    }

    dts = np.array([r['info']['ophys_frame_interval_s'] for r in ok])
    n_trials = sum(len(r['neural']) for r in ok)
    n_neurons = sum(r['n_neurons'] for r in ok)
    n_timepoints = sum(int(t.shape[1]) for r in ok for t in r['neural'])

    data['metadata'] = {
        'task_description': (
            'Allen Brain Observatory Visual Behavior 2P: head-fixed mice perform a '
            'go/no-go visual change-detection task while 2-photon calcium imaging is '
            'performed in visual cortex. A series of natural images is flashed (250 ms '
            'on / 500 ms gray, 5% of non-change flashes omitted) and the mouse earns a '
            'water reward by licking within 150-750 ms of a change in image identity. '
            'Each session is segmented into the experiment-defined Go (real change) and '
            'Catch (sham change) trials; Aborted and Auto-rewarded trials are excluded. '
            'The decoder receives no external inputs and predicts, from population dF/F '
            'alone, five categorical variables: (0) image_identity -- which of the 16 '
            'natural images is currently being shown, held through the gray screen and '
            'through omitted flashes; (1) image_change -- 1 for every frame in the '
            '750 ms image-presentation interval that begins with a change in image '
            'identity; (2) running_speed_bin -- running speed discretised into 5 equal '
            'percentile (quintile) bins; (3) pupil_diameter_bin -- pupil diameter '
            'discretised into 5 equal percentile bins; (4) trial_outcome -- '
            'hit/miss/false_alarm/correct_reject, constant within a trial.'),
        'time_bin_size': float(np.mean(dts) * 1000.0),
        'temporal_alignment_event': (
            'Ophys (2-photon) frame timestamps. Every data stream is placed on the '
            'native ophys timestamp grid of its session: dF/F and its timestamps are '
            'used as recorded, the 60 Hz running trace and the ~30 Hz pupil trace are '
            'linearly interpolated onto the ophys frame times, and the stimulus and '
            'trial variables are evaluated at those same frame times. A trial contains '
            'every ophys frame t with trials.start_time <= t < trials.stop_time.'),
        'off_start': 0.0,
        'off_end': None,
        'trial_definition': (
            'trials table rows with go==True or catch==True (aborted and auto_rewarded '
            'excluded); trial window = [start_time, stop_time).'),
        'neural_signal': 'dff_traces (detrended dF/F, AllenSDK/Allen pipeline)',
        'neural_units': 'dF/F (fraction)',
        'time_bin_size_std_ms': float(np.std(dts) * 1000.0),
        'time_bin_size_range_ms': [float(dts.min() * 1000.0),
                                   float(dts.max() * 1000.0)],
        'n_sessions': len(ok),
        'n_subjects': len(subjects),
        'n_trials': int(n_trials),
        'n_neurons': int(n_neurons),
        'n_timepoints': int(n_timepoints),
        'trial_duration_s': 'variable (mean ~8.5 s, range ~7.3-12.6 s)',
        'dataset': 'Allen Brain Observatory Visual Behavior 2P, visual-behavior-ophys-1.1.0',
        'selection_criteria': (
            'locally available NWB experiments with project_code == "VisualBehavior" '
            '(single-plane Scientifica rigs, 30.95 Hz, VISp) and passive == False '
            '(active behaviour). Sessions without eye-tracking data are dropped.'),
        'session_info': [r['info'] for r in ok],
        'skipped_sessions': skipped,
    }

    print('\n--- summary ---')
    print(f'sessions            : {len(ok)} (skipped {len(skipped)})')
    for s in skipped:
        print(f'   skipped {s["eid"]}: {s["error"]}')
    print(f'subjects            : {len(subjects)}')
    print(f'brain regions       : {brain_regions}')
    print(f'neurons (total)     : {n_neurons}')
    print(f'trials (total)      : {n_trials}')
    print(f'timepoints (total)  : {n_timepoints}')
    print(f'time bin size (ms)  : {data["metadata"]["time_bin_size"]:.4f} '
          f'(sd {data["metadata"]["time_bin_size_std_ms"]:.5f})')

    allout = np.concatenate([o for r in ok for o in r['output']], axis=1)
    for i, name in enumerate(OUTPUT_NAMES):
        vals, counts = np.unique(allout[i], return_counts=True)
        frac = counts / counts.sum()
        print(f'output {i} {name}: ' +
              ', '.join(f'{OUTPUT_VALUES[i][v]}={f:.4f}' for v, f in zip(vals, frac)))

    t0 = time.time()
    with open(args.outfile, 'wb') as fh:
        pickle.dump(data, fh, protocol=4)
    print(f'\nwrote {args.outfile} '
          f'({os.path.getsize(args.outfile) / 1e9:.2f} GB) in {time.time() - t0:.1f}s')
    print(f'total elapsed {time.time() - t_all:.1f}s')


if __name__ == '__main__':
    main()
