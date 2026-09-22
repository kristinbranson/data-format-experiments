#!/usr/bin/env python3
"""
Convert the Allen Brain Observatory Visual Behavior 2P dataset into the
decoder-ready dictionary format described in the task specification.

Usage
-----
    python -u convert_data.py <outpicklefile> [--full | --sample] [--show-processing]

Options
-------
    --full             process every selected session (default)
    --sample           process only 2 sessions (one per image set) for testing
    --show-processing  save diagnostic plots (processing_<ophys_experiment_id>.png)
                       for up to 2 sessions, illustrating every processing step
    --workers N        number of parallel worker processes (default 16)

What is converted (see CONVERSION_NOTES.md for full justification)
------------------------------------------------------------------
sessions : every *active behaviour* single-plane ("VisualBehavior" project code,
           Scientifica rigs, 31 Hz) ophys experiment available locally that has
           eye-tracking data.  Passive sessions have no trials; the Multiscope
           sessions are excluded because their 11 Hz frame rate is incompatible
           with a single common time-bin size (and 34 of their planes come from
           only 6 behaviour sessions of a single mouse).
trials   : trials from the SDK `trials` table with go == True or catch == True
           (i.e. excluding aborted and auto-rewarded trials), spanning
           [start_time, stop_time).
neural   : dF/F traces (`dff_traces`, the normalised + detrended traces produced
           by the Allen pipeline) sampled on the native ophys frame times.
           `--neural events|filtered_events` selects the detected calcium events
           instead (see CONVERSION_NOTES.md for the comparison).
input    : none (dimension 0), as specified by the decoder task.
output   : 0 image identity (categorical, held over the 750 ms image
             presentation interval, carried through omissions)
           1 image change (1 during the 750 ms interval starting at a change)
           2 running speed, 5 per-session equal-percentile bins
           3 pupil diameter, 5 per-session equal-percentile bins
           4 trial outcome (hit / miss / false_alarm / correct_reject), constant
             within a trial
"""

import argparse
import os
import re
import sys
import time
import pickle
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

DATA_ROOT = '/app/data/visual-behavior-ophys-1.1.0'
NWB_DIR = os.path.join(DATA_ROOT, 'behavior_ophys_experiments')
META_DIR = os.path.join(DATA_ROOT, 'project_metadata')

# --- fixed analysis constants -------------------------------------------------
NEURAL_SIGNAL = 'dff'              # see CONVERSION_NOTES.md, Step 5 decision 3
NEURAL_NORMALIZE = 'none'          # see CONVERSION_NOTES.md, Step 5 decision 4
N_BEHAVIOR_BINS = 5                 # "five equal percentile bins"
OUTCOME_NAMES = ['hit', 'miss', 'false_alarm', 'correct_reject']
OUTPUT_NAMES = ['image_identity', 'image_change', 'running_speed_bin',
                'pupil_diameter_bin', 'trial_outcome']


# ------------------------------------------------------------------ selection
def select_experiments():
    """Return the experiment-table rows for the sessions we convert.

    Selection (documented in CONVERSION_NOTES.md Step 5):
      * the NWB file is present locally
      * active behaviour (session_type does not contain 'passive')
      * project_code == 'VisualBehavior' (single-plane Scientifica rigs, 31 Hz)
    """
    et = pd.read_csv(os.path.join(META_DIR, 'ophys_experiment_table.csv'))
    local = {int(re.findall(r'(\d+)', f)[0]): os.path.join(NWB_DIR, f)
             for f in os.listdir(NWB_DIR) if f.endswith('.nwb')}
    et = et[et.ophys_experiment_id.isin(local.keys())].copy()
    et['path'] = et.ophys_experiment_id.map(local)
    n_local = len(et)
    et = et[~et.session_type.str.contains('passive')]
    n_active = len(et)
    et = et[et.project_code == 'VisualBehavior']
    print(f"  local NWB files: {n_local}; active behaviour: {n_active}; "
          f"active single-plane (VisualBehavior): {len(et)}")
    # deterministic order: by mouse, then acquisition date
    et = et.sort_values(['mouse_id', 'date_of_acquisition']).reset_index(drop=True)
    return et


# ------------------------------------------------------------- per-session job
def process_session(args):
    """Load one NWB file and cut it into trials.  Runs in a worker process.

    Returns a dict with per-trial neural / output arrays and session metadata,
    or a dict with 'skip' explaining why the session was dropped.
    """
    path, want_raw, signal, normalize = args
    from allensdk.brain_observatory.behavior.behavior_ophys_experiment import (
        BehaviorOphysExperiment)

    t_start = time.time()
    eid = int(re.findall(r'(\d+)', os.path.basename(path))[0])
    ds = BehaviorOphysExperiment.from_nwb_path(path)
    md = ds.metadata
    t_load = time.time() - t_start

    res = {'eid': eid, 'mouse': str(md['mouse_id']),
           'session_type': md['session_type'], 'cre_line': md['cre_line'],
           'structure': md['targeted_structure'], 'depth': int(md['imaging_depth']),
           'rig': md['equipment_name'], 'frame_rate': float(md['ophys_frame_rate']),
           'ophys_session_id': int(md['ophys_session_id']),
           'container': int(md['ophys_container_id'])}

    # ---- eye tracking is required (one of the decoder outputs) --------------
    try:
        eye = ds.eye_tracking
        if eye is None or len(eye) == 0:
            raise ValueError('empty')
    except Exception as exc:
        res['skip'] = f'no eye tracking ({exc})'
        return res

    # ---- neural ------------------------------------------------------------
    ots = np.asarray(ds.ophys_timestamps, dtype=np.float64)
    if signal == 'dff':
        neural_full = np.vstack(ds.dff_traces.dff.values).astype(np.float32)
    else:
        neural_full = np.vstack(ds.events[signal].values).astype(np.float32)
    assert neural_full.shape[1] == ots.size, 'trace length != ophys timestamps'
    if normalize == 'zscore':
        # unit variance per neuron; the decoder's SVD-initialised projection and
        # its fixed learning rate need a well-conditioned input scale
        sd = neural_full.std(axis=1, keepdims=True)
        neural_full = neural_full / np.maximum(sd, 1e-6)
    elif normalize == 'noise_std':
        sd = ds.events['noise_std'].values.astype(np.float32)[:, None]
        neural_full = neural_full / np.maximum(sd, 1e-6)
    assert np.all(np.diff(ots) > 0), 'ophys timestamps not increasing'
    cst = ds.cell_specimen_table
    assert bool(cst.valid_roi.all()), 'invalid ROIs present in released data'
    assert len(cst) == neural_full.shape[0]
    res['n_neurons'] = neural_full.shape[0]
    res['dt_median'] = float(np.median(np.diff(ots)))
    res['cell_specimen_ids'] = cst.index.values.astype(np.int64)

    # ---- stimulus timeline -------------------------------------------------
    stim = ds.stimulus_presentations
    flashes = stim[stim.stimulus_block_name == 'change_detection_behavior'].copy()
    flashes = flashes.sort_values('start_time')
    f_start = flashes.start_time.values.astype(np.float64)
    f_omitted = flashes.omitted.values.astype(bool)
    f_change = flashes.is_change.values.astype(bool)
    f_name = flashes.image_name.values.astype(str)

    # per-session image vocabulary (8 images, alphabetically sorted)
    images = sorted(str(x) for x in set(f_name[~f_omitted]))
    res['images'] = images
    img_to_local = {n: i for i, n in enumerate(images)}

    # image identity per *flash interval*: the most recent non-omitted flash
    # (omissions carry the ongoing image forward, so that the image identity
    # signal changes exactly at image changes)
    local_idx = np.array([img_to_local.get(n, -1) for n in f_name], dtype=np.int16)
    carry = local_idx.copy()
    for i in range(1, carry.size):          # forward-fill through omissions
        if f_omitted[i]:
            carry[i] = carry[i - 1]
    assert np.all(carry >= 0), 'image identity undefined at start of block'

    # map every ophys frame onto its flash interval [start_i, start_{i+1})
    fi = np.searchsorted(f_start, ots, side='right') - 1
    fi_clipped = np.clip(fi, 0, f_start.size - 1)
    image_per_frame = carry[fi_clipped]
    change_per_frame = f_change[fi_clipped].astype(np.int8)
    # frames before the first flash of the behaviour block carry no stimulus
    change_per_frame[fi < 0] = 0

    # ---- behaviour streams resampled onto the ophys frame times ------------
    run = ds.running_speed
    run_t = run.timestamps.values.astype(np.float64)
    run_v = run.speed.values.astype(np.float64)
    good = np.isfinite(run_v)
    speed_per_frame = np.interp(ots, run_t[good], run_v[good])

    eye_t = eye.timestamps.values.astype(np.float64)
    pupil_area = eye.pupil_area.values.astype(np.float64)
    good_eye = np.isfinite(pupil_area)
    res['blink_fraction'] = float(1.0 - good_eye.mean())
    if good_eye.sum() < 100:
        res['skip'] = 'pupil data all NaN'
        return res
    # pupil diameter of the fitted ellipse, in pixels (area -> diameter is a
    # monotone transform, so the percentile bins are unaffected by the choice)
    pupil_diam = 2.0 * np.sqrt(pupil_area / np.pi)
    pupil_per_frame = np.interp(ots, eye_t[good_eye], pupil_diam[good_eye])

    # ---- trials ------------------------------------------------------------
    trials = ds.trials
    res['n_trials_all'] = int(len(trials))
    res['n_aborted'] = int(trials.aborted.sum())
    res['n_auto_rewarded'] = int(trials.auto_rewarded.sum())
    sel = trials[(trials.go.values | trials.catch.values)].copy()
    # the four outcomes are mutually exclusive and exhaustive on go|catch trials
    oc = np.full(len(sel), -1, dtype=np.int8)
    for k, name in enumerate(OUTCOME_NAMES):
        oc[sel[name].values.astype(bool)] = k
    assert np.all(oc >= 0), 'go/catch trial without an outcome'
    assert (sel[OUTCOME_NAMES].values.astype(int).sum(axis=1) == 1).all(), \
        'trial with more than one outcome'

    i0 = np.searchsorted(ots, sel.start_time.values, side='left')
    i1 = np.searchsorted(ots, sel.stop_time.values, side='left')
    keep = (i1 - i0) > 1
    if not keep.all():
        print(f'    [{eid}] dropping {int((~keep).sum())} trials with <2 ophys frames')
    # a trial must not start before the first flash of the behaviour block by
    # more than one flash cycle (trials start ~21 ms before a flash onset)
    assert np.all(sel.start_time.values[keep] > f_start[0] - 0.1), \
        'trial starts before the stimulus block'

    sel = sel[keep]
    oc = oc[keep]
    i0 = i0[keep]
    i1 = i1[keep]
    n_trials = len(sel)
    res['n_trials'] = n_trials
    res['n_go'] = int(sel.go.sum())
    res['n_catch'] = int(sel.catch.sum())
    if n_trials < 2:
        res['skip'] = f'only {n_trials} usable trials'
        return res

    frame_idx = np.concatenate([np.arange(a, b) for a, b in zip(i0, i1)])
    assert np.all(np.diff(frame_idx) > 0), 'trials overlap in time'

    # ---- percentile discretisation, per session over the retained frames ----
    def percentile_bins(x):
        edges = np.percentile(x, np.linspace(0, 100, N_BEHAVIOR_BINS + 1)[1:-1])
        return np.digitize(x, edges).astype(np.int8), edges

    speed_bin_all, speed_edges = percentile_bins(speed_per_frame[frame_idx])
    pupil_bin_all, pupil_edges = percentile_bins(pupil_per_frame[frame_idx])
    res['speed_edges'] = speed_edges
    res['pupil_edges'] = pupil_edges
    res['speed_range'] = (float(speed_per_frame[frame_idx].min()),
                          float(speed_per_frame[frame_idx].max()))
    res['pupil_range'] = (float(pupil_per_frame[frame_idx].min()),
                          float(pupil_per_frame[frame_idx].max()))

    # scatter the per-frame bins back onto a session-length array for slicing
    speed_bin = np.zeros(ots.size, dtype=np.int8)
    pupil_bin = np.zeros(ots.size, dtype=np.int8)
    speed_bin[frame_idx] = speed_bin_all
    pupil_bin[frame_idx] = pupil_bin_all

    # ---- cut trials --------------------------------------------------------
    neural, output = [], []
    for k in range(n_trials):
        a, b = i0[k], i1[k]
        neural.append(np.ascontiguousarray(neural_full[:, a:b]))
        out = np.empty((5, b - a), dtype=np.int8)
        out[0] = image_per_frame[a:b]
        out[1] = change_per_frame[a:b]
        out[2] = speed_bin[a:b]
        out[3] = pupil_bin[a:b]
        out[4] = oc[k]
        output.append(out)

    # ---- per-session sanity checks ----------------------------------------
    _check_session(res, sel, oc, i0, i1, ots, output, images)

    res['neural'] = neural
    res['output'] = output
    res['T'] = np.array([o.shape[1] for o in output])
    res['trial_start'] = sel.start_time.values
    res['trial_stop'] = sel.stop_time.values
    res['change_time'] = sel.change_time.values
    res['outcome'] = oc
    res['t_load'] = t_load
    res['t_total'] = time.time() - t_start

    if want_raw:   # extra material for the --show-processing plots
        res['raw'] = dict(
            ots=ots, neural_full=neural_full, f_start=f_start, f_omitted=f_omitted,
            f_change=f_change, f_name=f_name, image_per_frame=image_per_frame,
            change_per_frame=change_per_frame, speed_per_frame=speed_per_frame,
            pupil_per_frame=pupil_per_frame, run_t=run_t, run_v=run_v,
            eye_t=eye_t[good_eye], pupil_diam=pupil_diam[good_eye],
            eye_t_all=eye_t, pupil_diam_all=pupil_diam,
            i0=i0, i1=i1, frame_idx=frame_idx,
            speed_bin_all=speed_bin_all, pupil_bin_all=pupil_bin_all)
    return res


def _check_session(res, sel, oc, i0, i1, ots, output, images):
    """Internal consistency checks of one converted session (raise on failure)."""
    eid = res['eid']
    img_to_local = {n: i for i, n in enumerate(images)}
    go = sel.go.values.astype(bool)
    change_time = sel.change_time.values
    init_img = sel.initial_image_name.values.astype(str)
    chg_img = sel.change_image_name.values.astype(str)

    n_bad_id, n_bad_change = 0, 0
    for k in range(len(sel)):
        t = ots[i0[k]:i1[k]]
        ident = output[k][0]
        chg = output[k][1]
        pre = t < change_time[k]
        post = ~pre
        # before the change the trial shows `initial_image_name`; after a real
        # change it shows `change_image_name`; catch trials never change
        if pre.any() and not np.all(ident[pre] == img_to_local[init_img[k]]):
            n_bad_id += 1
        if post.any():
            want = img_to_local[chg_img[k] if go[k] else init_img[k]]
            if not np.all(ident[post] == want):
                n_bad_id += 1
        # image_change must be 1 only in the 750 ms interval after a real change
        if go[k]:
            if chg.sum() == 0 or not np.all(t[chg == 1] >= change_time[k] - 1e-9):
                n_bad_change += 1
            dur = t[chg == 1].max() - t[chg == 1].min() if chg.sum() else 0
            if dur > 0.8:
                n_bad_change += 1
        else:
            if chg.sum() != 0:
                n_bad_change += 1
    assert n_bad_id == 0, f'[{eid}] image identity mismatch in {n_bad_id} trials'
    assert n_bad_change == 0, f'[{eid}] image change mismatch in {n_bad_change} trials'
    # outcome consistency with the trial type
    assert np.all((oc < 2) == sel.go.values.astype(bool)), \
        f'[{eid}] hit/miss must be go trials, FA/CR must be catch trials'


# --------------------------------------------------------------------- plotting
def show_processing(res, outdir='/app'):
    """Diagnostic figure covering every processing step for one session."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    r = res['raw']
    eid = res['eid']
    ots = r['ots']
    fig = plt.figure(figsize=(26, 30))
    gs = fig.add_gridspec(7, 3, height_ratios=[1.2, 1.2, 1, 1, 1.4, 1, 1.2],
                          hspace=0.45, wspace=0.2)
    cmap = plt.get_cmap('tab10')

    # pick a 30 s window inside the 3rd retained trial for the zoomed panels
    k0 = min(3, len(res['T']) - 1)
    t_lo = res['trial_start'][k0] - 2
    t_hi = t_lo + 30

    def stim_bars(ax):
        m = (r['f_start'] >= t_lo - 1) & (r['f_start'] <= t_hi)
        for s, om, nm, ch in zip(r['f_start'][m], r['f_omitted'][m],
                                 r['f_name'][m], r['f_change'][m]):
            if om:
                ax.axvspan(s, s + 0.25, color='k', alpha=0.08, hatch='//')
            else:
                ci = res['images'].index(nm)
                ax.axvspan(s, s + 0.25, color=cmap(ci), alpha=0.3)
            if ch:
                ax.axvline(s, color='r', lw=1.5)

    def trial_bounds(ax):
        for a, b in zip(res['trial_start'], res['trial_stop']):
            if b > t_lo and a < t_hi:
                ax.axvline(a, color='g', ls='--', lw=1)
                ax.axvline(b, color='m', ls=':', lw=1)

    # (1) raw neural traces on the ophys grid ---------------------------------
    ax = fig.add_subplot(gs[0, :])
    m = (ots >= t_lo) & (ots <= t_hi)
    nsh = min(12, res['n_neurons'])
    for i in range(nsh):
        tr = r['neural_full'][i, m]
        mx = tr.max() if tr.max() > 0 else 1
        ax.plot(ots[m], tr / mx + i, lw=0.8)
    stim_bars(ax); trial_bounds(ax)
    ax.set_title(f'{eid}: step 1 — neural trace on native ophys timestamps '
                 f'(shading = image flashes, red = change, green/magenta = trial start/stop)')
    ax.set_xlim(t_lo, t_hi); ax.set_ylabel('neuron (norm.)')

    # (2) population mean + stimulus/behaviour overlay ------------------------
    ax = fig.add_subplot(gs[1, :])
    ax.plot(ots[m], r['neural_full'][:, m].mean(0), 'k', lw=1.2, label='pop. mean')
    stim_bars(ax); trial_bounds(ax)
    ax.set_xlim(t_lo, t_hi); ax.legend(loc='upper right')
    ax.set_title('step 2 — population mean activity (check: transients follow flashes/changes)')

    # (3) running speed: native vs resampled ----------------------------------
    ax = fig.add_subplot(gs[2, :])
    mr = (r['run_t'] >= t_lo) & (r['run_t'] <= t_hi)
    ax.plot(r['run_t'][mr], r['run_v'][mr], '.-', color='0.6', ms=3, lw=0.8,
            label='running speed, native ~60 Hz')
    ax.plot(ots[m], r['speed_per_frame'][m], 'b.-', ms=4, lw=1,
            label='resampled onto ophys frames')
    stim_bars(ax); ax.set_xlim(t_lo, t_hi); ax.legend(loc='upper right')
    ax.set_ylabel('cm/s'); ax.set_title('step 3 — running speed alignment')

    # (4) pupil: native (with blinks) vs interpolated & resampled -------------
    ax = fig.add_subplot(gs[3, :])
    me = (r['eye_t_all'] >= t_lo) & (r['eye_t_all'] <= t_hi)
    ax.plot(r['eye_t_all'][me], r['pupil_diam_all'][me], '.-', color='0.6', ms=3,
            lw=0.8, label='pupil diameter, native (NaN during blinks)')
    ax.plot(ots[m], r['pupil_per_frame'][m], 'r.-', ms=4, lw=1,
            label='blink-interpolated, resampled onto ophys frames')
    ax.set_xlim(t_lo, t_hi); ax.legend(loc='upper right')
    ax.set_ylabel('pixels'); ax.set_title('step 4 — pupil diameter alignment')

    # (5) outputs for the same window -----------------------------------------
    ax = fig.add_subplot(gs[4, :])
    fidx = np.concatenate([np.arange(a, b) for a, b in zip(r['i0'], r['i1'])])
    inwin = fidx[(ots[fidx] >= t_lo) & (ots[fidx] <= t_hi)]
    # rebuild the per-frame bins for plotting
    sb = np.full(ots.size, np.nan); sb[r['frame_idx']] = r['speed_bin_all']
    pb = np.full(ots.size, np.nan); pb[r['frame_idx']] = r['pupil_bin_all']
    ax.step(ots[inwin], r['image_per_frame'][inwin], where='post', label='image identity (0-7)')
    ax.step(ots[inwin], r['change_per_frame'][inwin] * 8, where='post', label='image change x8')
    ax.step(ots[inwin], sb[inwin], where='post', label='running speed bin')
    ax.step(ots[inwin], pb[inwin], where='post', label='pupil bin')
    stim_bars(ax); trial_bounds(ax)
    ax.set_xlim(t_lo, t_hi); ax.legend(loc='upper right', ncol=4)
    ax.set_title('step 5 — decoder outputs (only frames inside retained trials are drawn)')

    # (6) discretisation histograms -------------------------------------------
    ax = fig.add_subplot(gs[5, 0])
    ax.hist(r['speed_per_frame'][r['frame_idx']], bins=100)
    for e in res['speed_edges']:
        ax.axvline(e, color='r')
    ax.set_yscale('log'); ax.set_xlabel('running speed (cm/s)')
    ax.set_title('step 6 — running speed percentile edges')
    ax = fig.add_subplot(gs[5, 1])
    ax.hist(r['pupil_per_frame'][r['frame_idx']], bins=100)
    for e in res['pupil_edges']:
        ax.axvline(e, color='r')
    ax.set_yscale('log'); ax.set_xlabel('pupil diameter (px)')
    ax.set_title('pupil percentile edges')
    ax = fig.add_subplot(gs[5, 2])
    counts = [np.mean(r['speed_bin_all'] == i) for i in range(N_BEHAVIOR_BINS)]
    countp = [np.mean(r['pupil_bin_all'] == i) for i in range(N_BEHAVIOR_BINS)]
    ax.bar(np.arange(5) - 0.2, counts, 0.4, label='running')
    ax.bar(np.arange(5) + 0.2, countp, 0.4, label='pupil')
    ax.axhline(0.2, color='k', ls='--'); ax.legend()
    ax.set_title('bin occupancy (should be 0.2 each)')

    # (7) change-triggered average: the temporal-alignment check --------------
    ax = fig.add_subplot(gs[6, 0])
    win = np.arange(-int(1.0 / res['dt_median']), int(2.0 / res['dt_median']))
    ct = res['change_time'][~np.isnan(res['change_time'])]
    ci = np.searchsorted(ots, ct)
    ci = ci[(ci + win[0] >= 0) & (ci + win[-1] < ots.size)]
    pop = r['neural_full'].mean(0)
    sta = np.stack([pop[c + win] for c in ci]).mean(0)
    ax.plot(win * res['dt_median'], sta)
    ax.axvline(0, color='r')
    ax.set_xlabel('time from change (s)'); ax.set_ylabel('pop. mean')
    ax.set_title('step 7 — change-triggered average (peak must follow t=0)')

    ax = fig.add_subplot(gs[6, 1])
    # flash-triggered average, non-change flashes
    fi = np.searchsorted(ots, r['f_start'][~r['f_change'] & ~r['f_omitted']])
    win2 = np.arange(-int(0.3 / res['dt_median']), int(0.75 / res['dt_median']))
    fi = fi[(fi + win2[0] >= 0) & (fi + win2[-1] < ots.size)]
    ax.plot(win2 * res['dt_median'], np.stack([pop[c + win2] for c in fi]).mean(0))
    ax.axvline(0, color='r')
    ax.set_xlabel('time from flash (s)'); ax.set_title('flash-triggered average')

    ax = fig.add_subplot(gs[6, 2])
    ax.hist(res['T'], bins=40)
    ax.set_xlabel('trial length (ophys frames)')
    ax.set_title(f"trial lengths: n={len(res['T'])}, "
                 f"{res['T'].min()}-{res['T'].max()} frames")

    fig.suptitle(f"Processing checks — experiment {eid} "
                 f"({res['session_type']}, {res['cre_line']}, mouse {res['mouse']})",
                 fontsize=16)
    fn = os.path.join(outdir, f'processing_{eid}.png')
    fig.savefig(fn, dpi=80, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {fn}')


# ------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('outfile')
    g = ap.add_mutually_exclusive_group()
    g.add_argument('--full', action='store_true', help='process all sessions (default)')
    g.add_argument('--sample', action='store_true', help='process 2 sessions only')
    ap.add_argument('--show-processing', action='store_true')
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--neural', default=NEURAL_SIGNAL,
                    choices=['filtered_events', 'events', 'dff'],
                    help='which neural trace to store (default: %(default)s)')
    ap.add_argument('--normalize', default=NEURAL_NORMALIZE,
                    choices=['none', 'zscore', 'noise_std'],
                    help='per-neuron scaling of the neural trace (default: %(default)s)')
    args = ap.parse_args()

    t0 = time.time()
    print('Selecting experiments...')
    et = select_experiments()

    n_keep = None
    if args.sample:
        # candidates alternating between the two image sets and coming from
        # different mice, so that the global image vocabulary logic is
        # exercised; the first two that survive the quality checks are kept
        cand = []
        for i in range(3):
            for s in ('A', 'B'):
                sub = et[(et.image_set == s) & (~et.mouse_id.isin(
                    [c.mouse_id for c in cand]))]
                if len(sub):
                    cand.append(sub.iloc[i % len(sub)])
        et = pd.DataFrame(cand)
        n_keep = 2
        print(f'  --sample: candidate experiments {list(et.ophys_experiment_id)}')

    paths = list(et.path)
    n_plot = 2 if args.show_processing else 0
    # in sample mode any candidate may end up being the one we plot
    jobs = [(p, args.show_processing and (args.sample or i < n_plot),
             args.neural, args.normalize) for i, p in enumerate(paths)]

    print(f'Processing {len(jobs)} sessions with {args.workers} workers...')
    results = []
    t_proc = time.time()
    if args.workers > 1 and len(jobs) > 1:
        with Pool(min(args.workers, len(jobs))) as pool:
            for i, r in enumerate(pool.imap(process_session, jobs)):
                _report(i, len(jobs), r, t_proc)
                results.append(r)
    else:
        for i, j in enumerate(jobs):
            r = process_session(j)
            _report(i, len(jobs), r, t_proc)
            results.append(r)
    print(f'  session processing took {time.time() - t_proc:.1f} s '
          f'({(time.time() - t_proc) / max(1, len(jobs)):.2f} s/session)')

    good = [r for r in results if 'skip' not in r]
    skipped = [r for r in results if 'skip' in r]
    if n_keep is not None:
        # prefer one session per image set among the surviving candidates
        picked, seen = [], set()
        for r in good:
            key = tuple(r['images'])
            if key not in seen:
                picked.append(r)
                seen.add(key)
        good = (picked + [r for r in good if r not in picked])[:n_keep]

    if args.show_processing:
        for r in good[:n_plot]:
            if 'raw' in r:
                show_processing(r)
    for r in results:
        r.pop('raw', None)

    for r in skipped:
        print(f"  SKIPPED {r['eid']} ({r.get('session_type')}): {r['skip']}")
    print(f'Kept {len(good)} / {len(results)} sessions')

    # ---- assemble the global structure -------------------------------------
    t_asm = time.time()
    image_vocab = sorted(set(sum([r['images'] for r in good], [])))
    print(f'Image vocabulary ({len(image_vocab)}): {image_vocab}')
    subjects = sorted(set(r['mouse'] for r in good))
    subj_to_idx = {s: i for i, s in enumerate(subjects)}
    regions = sorted(set(r['structure'] for r in good))
    reg_to_idx = {s: i for i, s in enumerate(regions)}

    neural, inputs, outputs, subject_idx, brain_region_idx, session_info = \
        [], [], [], [], [], []
    for r in good:
        # remap the per-session image indices onto the global vocabulary
        remap = np.array([image_vocab.index(n) for n in r['images']], dtype=np.int8)
        for o in r['output']:
            o[0] = remap[o[0]]
        neural.append(r['neural'])
        outputs.append(r['output'])
        inputs.append([np.zeros((0, o.shape[1]), dtype=np.float32) for o in r['output']])
        subject_idx.append(subj_to_idx[r['mouse']])
        brain_region_idx.append(np.full(r['n_neurons'], reg_to_idx[r['structure']],
                                        dtype=np.int64))
        session_info.append({
            'ophys_experiment_id': r['eid'], 'ophys_session_id': r['ophys_session_id'],
            'ophys_container_id': r['container'], 'mouse_id': r['mouse'],
            'session_type': r['session_type'], 'cre_line': r['cre_line'],
            'targeted_structure': r['structure'], 'imaging_depth': r['depth'],
            'equipment_name': r['rig'], 'ophys_frame_rate': r['frame_rate'],
            'median_frame_interval_s': r['dt_median'],
            'n_neurons': r['n_neurons'], 'n_trials': r['n_trials'],
            'n_go': r['n_go'], 'n_catch': r['n_catch'],
            'n_trials_in_session': r['n_trials_all'],
            'n_aborted': r['n_aborted'], 'n_auto_rewarded': r['n_auto_rewarded'],
            'images': r['images'],
            'blink_fraction': r['blink_fraction'],
            'running_speed_bin_edges': r['speed_edges'].tolist(),
            'pupil_diameter_bin_edges': r['pupil_edges'].tolist(),
            'cell_specimen_ids': r['cell_specimen_ids'],
            'trial_start_times': r['trial_start'], 'trial_stop_times': r['trial_stop'],
            'change_times': r['change_time'],
        })

    dt = float(np.mean([r['dt_median'] for r in good]))
    data = {
        'neural': neural,
        'input': inputs,
        'output': outputs,
        'subjects': subjects,
        'subject_idx': np.array(subject_idx, dtype=np.int64),
        'brain_regions': regions,
        'brain_region_idx': brain_region_idx,
        'input_names': [],
        'output_names': OUTPUT_NAMES,
        'output_values': [
            image_vocab,
            ['no_change', 'change'],
            [f'speed_pct_{i*20}_{(i+1)*20}' for i in range(N_BEHAVIOR_BINS)],
            [f'pupil_pct_{i*20}_{(i+1)*20}' for i in range(N_BEHAVIOR_BINS)],
            OUTCOME_NAMES,
        ],
        'metadata': {
            'task_description':
                'Allen Brain Observatory Visual Behavior 2P: head-fixed mice perform a '
                'go/no-go visual change-detection task while 2-photon calcium imaging is '
                'performed in VISp. Natural images are flashed for 250 ms every 750 ms '
                '(500 ms grey inter-stimulus interval, 5% of flashes omitted); mice are '
                'rewarded for licking within 150-750 ms of a change in image identity. '
                'Trials are go (real change) or catch (sham change); aborted and '
                'auto-rewarded trials are excluded. Decoded from neural activity: image '
                'identity, image change, running-speed quintile, pupil-diameter quintile '
                '(all time-varying) and trial outcome (hit/miss/false alarm/correct '
                'reject, constant within a trial). There are no decoder inputs.',
            'time_bin_size': dt * 1000.0,
            'temporal_alignment_event':
                'trial start (trials.start_time from the AllenSDK trials table); each '
                'trial spans [start_time, stop_time) and is sampled on the native ophys '
                'frame times (ophys_timestamps), so trial length varies',
            'off_start': 0.0,
            'off_end': None,
            'neural_signal': args.neural,
            'neural_normalization': args.normalize,
            'neural_signal_description': {
                'dff': 'baseline-normalised, detrended dF/F traces as produced by the '
                       'Allen Visual Behavior 2P pipeline (demixed, neuropil-subtracted '
                       'fluorescence; 600 s median-filter baseline; 3.33 s median-filter '
                       'detrend), sampled at the ophys frame rate',
                'events': 'detected calcium events (Allen event detection)',
                'filtered_events': 'detected calcium events convolved with the AllenSDK '
                                   'causal half-normal filter (scale 2/31 s, 20 taps)',
            }[args.neural],
            'dataset': 'visual-behavior-ophys-1.1.0',
            'project_codes': sorted(set(et.project_code)),
            'session_types': sorted(set(r['session_type'] for r in good)),
            'trial_selection': 'trials.go or trials.catch (aborted and auto-rewarded excluded)',
            'behavior_discretization':
                'running speed and pupil diameter are binned into five equal-percentile '
                'bins computed separately for each session over the timepoints retained '
                'in that session',
            'image_identity_definition':
                'identity of the most recent presented (non-omitted) image, held over the '
                '750 ms image presentation interval; omitted flashes carry the ongoing '
                'image forward so that image identity changes exactly at image changes',
            'image_change_definition':
                '1 during the 750 ms image presentation interval that starts at a real '
                'image change, 0 otherwise (always 0 on catch/sham-change trials)',
            'mean_frame_interval_s': dt,
            'n_sessions': len(good),
            'n_subjects': len(subjects),
            'n_neurons_total': int(sum(r['n_neurons'] for r in good)),
            'n_trials_total': int(sum(r['n_trials'] for r in good)),
            'session_info': session_info,
        },
    }
    print(f'  assembly took {time.time() - t_asm:.1f} s')

    # ---- summary -----------------------------------------------------------
    nT = sum(int(r['T'].sum()) for r in good)
    print(f"\nSummary: {len(good)} sessions, {len(subjects)} mice, "
          f"{data['metadata']['n_neurons_total']} neurons, "
          f"{data['metadata']['n_trials_total']} trials, {nT} timepoints, "
          f"time bin {data['metadata']['time_bin_size']:.3f} ms")
    allout = np.concatenate([o for r in good for o in r['output']], axis=1)
    for i, name in enumerate(OUTPUT_NAMES):
        vals, cnt = np.unique(allout[i], return_counts=True)
        frac = cnt / cnt.sum()
        print(f'  {name}: ' + ', '.join(
            f'{data["output_values"][i][v]}={f:.3f}' for v, f in zip(vals, frac)))

    t_save = time.time()
    with open(args.outfile, 'wb') as f:
        pickle.dump(data, f, protocol=4)
    print(f'Wrote {args.outfile} '
          f'({os.path.getsize(args.outfile) / 1e9:.2f} GB) in {time.time() - t_save:.1f} s')
    print(f'Total time {time.time() - t0:.1f} s')


def _report(i, n, r, t_proc):
    el = time.time() - t_proc
    msg = (f"  [{i+1}/{n}] {r['eid']} {r.get('session_type','?')} "
           f"{r.get('n_neurons','?')} cells, {r.get('n_trials','?')} trials, "
           f"{r.get('t_total', float('nan')):.1f}s "
           f"(elapsed {el:.0f}s, eta {el/(i+1)*(n-i-1):.0f}s)")
    if 'skip' in r:
        msg += f"  SKIP: {r['skip']}"
    print(msg, flush=True)


if __name__ == '__main__':
    main()
