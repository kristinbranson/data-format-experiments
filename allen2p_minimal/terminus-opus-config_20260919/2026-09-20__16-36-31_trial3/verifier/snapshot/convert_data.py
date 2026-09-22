"""Convert Allen Visual Behavior 2P data into decoder format.

Design decisions (see README notes in the task):
 - Data: Allen Brain Observatory Visual Behavior 2P (visual-behavior-ophys-1.1.0),
   loaded from the local NWB files with the AllenSDK BehaviorOphysExperiment.
 - Session selection: active behavior sessions (passive == False) with the FAMILIAR
   image set, matching the Vip-Sst paper, which restricted neural analysis to
   familiar image-set sessions during active behavior.  All locally available
   familiar active sessions use image set A, so image identity labels are shared
   across sessions.
 - A 'session' is one ophys session (ophys_session_id).  For the multi-plane
   (Mesoscope) sessions the simultaneously recorded imaging planes (experiments)
   are concatenated into a single population, since they are one recording.
 - Neural data: detected calcium events, as used by the paper.  We use the
   AllenSDK 'filtered_events' (the same detected events convolved with a
   half-normal filter, as recommended in the SDK tutorials), summed into 100 ms
   bins on each plane's own ophys timestamps, and then scaled per neuron by its
   standard deviation across the session so that a few very high-amplitude cells
   do not dominate the decoder's projection.
 - Trials: 'go' and 'catch' trials (aborted and auto-rewarded excluded), aligned to
   the (sham) change time, window [-2.0, +4.0] s, which fits inside every trial.
"""

import os
import glob
import pickle
import argparse
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
from allensdk.brain_observatory.behavior.behavior_ophys_experiment import (
    BehaviorOphysExperiment)

DATA_DIR = '/app/data/visual-behavior-ophys-1.1.0'
EXP_DIR = os.path.join(DATA_DIR, 'behavior_ophys_experiments')
META_DIR = os.path.join(DATA_DIR, 'project_metadata')

BIN_SIZE = 0.1          # seconds
OFF_START = -2.0        # seconds relative to change time
OFF_END = 4.0           # seconds relative to change time
FLASH_INTERVAL = 0.75   # image presentation interval (s), as in the paper
NQUANTILES = 5
EVENTS_COL = 'filtered_events'  # AllenSDK detected calcium events, half-Gaussian filtered


def bin_edges():
    n = int(round((OFF_END - OFF_START) / BIN_SIZE))
    edges = OFF_START + BIN_SIZE * np.arange(n + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return edges, centers


def bin_events_matrix(traces, timestamps, t_edges):
    """Sum of calcium event magnitudes in each bin, for all cells at once."""
    idx = np.clip(np.searchsorted(timestamps, t_edges), 0, traces.shape[1])
    cs = np.concatenate([np.zeros((traces.shape[0], 1)), np.cumsum(traces, axis=1)],
                        axis=1)
    return cs[:, idx[1:]] - cs[:, idx[:-1]]


def bin_mean(values, timestamps, t_edges):
    """Mean of a continuous signal within each bin (NaN if no samples)."""
    idx = np.searchsorted(timestamps, t_edges)
    vals = np.concatenate([values, [0.0]])
    sums = np.add.reduceat(vals, idx[:-1])
    counts = (idx[1:] - idx[:-1]).astype(float)
    out = np.full(len(counts), np.nan)
    good = counts > 0
    out[good] = sums[good] / counts[good]
    return out


def fill_nans(x):
    """Linear interpolation of NaNs, extended at the edges."""
    x = np.asarray(x, dtype=float)
    good = np.isfinite(x)
    if not np.any(good):
        return None
    if np.all(good):
        return x
    idx = np.arange(len(x))
    return np.interp(idx, idx[good], x[good])


def quantize(values, nq=NQUANTILES):
    """Discretize into nq equal-percentile bins (quantile edges from the data)."""
    v = np.concatenate([np.ravel(a) for a in values])
    edges = np.quantile(v, np.linspace(0, 1, nq + 1)[1:-1])
    return [np.searchsorted(edges, np.ravel(a), side='right').astype(np.int64)
            for a in values]


def get_experiment_table():
    tab = pd.read_csv(os.path.join(META_DIR, 'ophys_experiment_table.csv'))
    local = [int(os.path.basename(f).split('_')[-1].split('.')[0])
             for f in glob.glob(os.path.join(EXP_DIR, '*.nwb'))]
    tab = tab[tab.ophys_experiment_id.isin(local)]
    # active behavior (mouse performing the change detection task) with familiar images
    tab = tab[(~tab.passive) & (tab.experience_level == 'Familiar')]
    return tab.sort_values(['ophys_session_id', 'ophys_experiment_id'])


def process_session(session_id, exp_rows, image_names):
    """Returns dict with neural/input/output lists for one ophys session."""
    edges_rel, centers_rel = bin_edges()
    T = len(centers_rel)

    neural_planes = []   # per plane: (ncells, ntrials, T)
    regions = []
    exps = []
    for _, row in exp_rows.iterrows():
        path = os.path.join(
            EXP_DIR, f'behavior_ophys_experiment_{row.ophys_experiment_id}.nwb')
        exps.append((row, BehaviorOphysExperiment.from_nwb_path(path)))

    ex0 = exps[0][1]

    # ---- trials: go and catch only (excludes aborted and auto-rewarded) ----
    trials = ex0.trials
    sel = trials[(trials.go | trials.catch) & trials.change_time.notna()]
    if len(sel) < 2:
        return None
    change_times = sel.change_time.values.astype(float)

    # ---- stimulus presentations for image identity / change ----
    sp = ex0.stimulus_presentations
    sp = sp[sp.stimulus_block_name.str.contains('change_detection', na=False)]
    flash = sp[~sp.omitted.astype(bool)]
    flash_start = flash.start_time.values.astype(float)
    flash_img = np.array([image_names.index(n) for n in flash.image_name.values])
    chg = sp[sp.is_change.astype(bool)]
    change_starts = chg.start_time.values.astype(float)

    # ---- behavior streams ----
    run = ex0.running_speed
    run_t = run.timestamps.values.astype(float)
    run_v = run.speed.values.astype(float)

    eye = ex0.eye_tracking
    pupil_t, pupil_v = None, None
    if len(eye) > 0:
        area = eye.pupil_area.values.astype(float)
        blink = eye.likely_blink.values.astype(bool)
        area = np.where(blink, np.nan, area)
        diam = 2.0 * np.sqrt(area / np.pi)   # pupil diameter from area
        diam = fill_nans(diam)
        if diam is not None:
            pupil_t = eye.timestamps.values.astype(float)
            pupil_v = diam
    if pupil_v is None:
        return None   # pupil diameter is a required decoder output

    ntrials = len(change_times)

    # ---- neural ----
    for row, ex in exps:
        ts = np.asarray(ex.ophys_timestamps, dtype=float)
        ev = ex.events
        traces = np.stack([np.asarray(e, dtype=float) for e in ev[EVENTS_COL].values])
        # guard against length mismatch
        n = min(traces.shape[1], len(ts))
        traces, tss = traces[:, :n], ts[:n]
        mat = np.zeros((traces.shape[0], ntrials, T), dtype=np.float32)
        for k, ct in enumerate(change_times):
            mat[:, k, :] = bin_events_matrix(traces, tss, ct + edges_rel)
        neural_planes.append(mat)
        regions += [row.targeted_structure] * traces.shape[0]

    neural_all = np.concatenate(neural_planes, axis=0)  # (ncells, ntrials, T)

    # Normalize each neuron by its standard deviation across all extracted bins of
    # the session.  Event magnitudes vary by orders of magnitude across cells; without
    # this a handful of high-variance cells would dominate the decoder's PCA
    # projection.  The scaling is per neuron and constant in time, so it does not mix
    # information across time or cells.
    sd = neural_all.reshape(neural_all.shape[0], -1).std(axis=1)
    sd[sd == 0] = 1.0
    neural_all = (neural_all / sd[:, None, None]).astype(np.float32)

    # ---- outputs ----
    img_trials, chg_trials, run_trials, pup_trials, out_trials = [], [], [], [], []
    for k, ct in enumerate(change_times):
        te = ct + edges_rel
        tc = ct + centers_rel

        # image identity: identity of the image presented in the ongoing 750 ms
        # image presentation interval, held through the gray screen and omissions
        j = np.searchsorted(flash_start, tc, side='right') - 1
        j = np.clip(j, 0, len(flash_start) - 1)
        img_trials.append(flash_img[j].astype(np.int64))

        # image change: 1 during the 750 ms interval starting at an actual image
        # change (catch/sham trials contain no change)
        i = np.searchsorted(change_starts, tc, side='right') - 1
        is_chg = np.zeros(T, dtype=np.int64)
        valid = i >= 0
        is_chg[valid] = (tc[valid] - change_starts[i[valid]] < FLASH_INTERVAL).astype(np.int64)
        chg_trials.append(is_chg)

        r = fill_nans(bin_mean(run_v, run_t, te))
        p = fill_nans(bin_mean(pupil_v, pupil_t, te))
        if r is None or p is None:
            return None
        run_trials.append(r)
        pup_trials.append(p)

        tr = sel.iloc[k]
        if tr.hit:
            o = 0
        elif tr.miss:
            o = 1
        elif tr.false_alarm:
            o = 2
        elif tr.correct_reject:
            o = 3
        else:
            o = -1
        out_trials.append(o)

    keep = [k for k in range(ntrials) if out_trials[k] >= 0]
    if len(keep) < 2:
        return None

    run_q = quantize([run_trials[k] for k in keep])
    pup_q = quantize([pup_trials[k] for k in keep])

    neural, inputs, outputs = [], [], []
    for n_, k in enumerate(keep):
        neural.append(np.ascontiguousarray(neural_all[:, k, :], dtype=np.float32))
        inputs.append(np.zeros((0, T), dtype=np.float32))
        outputs.append(np.stack([
            img_trials[k],
            chg_trials[k],
            run_q[n_],
            pup_q[n_],
            np.full(T, out_trials[k], dtype=np.int64),
        ]).astype(np.int64))

    row0 = exp_rows.iloc[0]
    info = {
        'ophys_session_id': int(session_id),
        'ophys_experiment_ids': [int(e) for e in exp_rows.ophys_experiment_id],
        'mouse_id': str(row0.mouse_id),
        'cre_line': str(row0.cre_line),
        'session_type': str(row0.session_type),
        'equipment_name': str(row0.equipment_name),
        'project_code': str(row0.project_code),
        'n_neurons': int(neural_all.shape[0]),
        'n_trials': len(keep),
    }
    return dict(neural=neural, input=inputs, output=outputs, regions=regions,
                mouse=str(row0.mouse_id), info=info)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='/app/converted_data.pkl')
    ap.add_argument('--limit', type=int, default=None)
    args = ap.parse_args()

    tab = get_experiment_table()
    print(f'{len(tab)} experiments, {tab.ophys_session_id.nunique()} sessions, '
          f'{tab.mouse_id.nunique()} mice')

    # global list of image names (familiar image set A)
    image_names = ['im061', 'im062', 'im063', 'im065',
                   'im066', 'im069', 'im077', 'im085']

    sessions = list(tab.groupby('ophys_session_id'))
    if args.limit:
        sessions = sessions[:args.limit]

    data = {'neural': [], 'input': [], 'output': [],
            'subjects': [], 'subject_idx': [],
            'brain_regions': ['VISp', 'VISl'], 'brain_region_idx': [],
            'input_names': [],
            'output_names': ['image_identity', 'image_change',
                             'running_speed_quintile', 'pupil_diameter_quintile',
                             'trial_outcome'],
            'output_values': [image_names,
                              ['no_change', 'change'],
                              ['Q1_0-20%', 'Q2_20-40%', 'Q3_40-60%',
                               'Q4_60-80%', 'Q5_80-100%'],
                              ['Q1_0-20%', 'Q2_20-40%', 'Q3_40-60%',
                               'Q4_60-80%', 'Q5_80-100%'],
                              ['hit', 'miss', 'false_alarm', 'correct_reject']],
            'metadata': {}}
    session_info = []

    for i, (sid, rows) in enumerate(sessions):
        try:
            res = process_session(sid, rows, image_names)
        except Exception as e:
            print(f'session {sid} failed: {e}')
            continue
        if res is None:
            print(f'session {sid} skipped')
            continue
        if res['mouse'] not in data['subjects']:
            data['subjects'].append(res['mouse'])
        data['neural'].append(res['neural'])
        data['input'].append(res['input'])
        data['output'].append(res['output'])
        data['subject_idx'].append(data['subjects'].index(res['mouse']))
        data['brain_region_idx'].append(
            np.array([data['brain_regions'].index(r) for r in res['regions']],
                     dtype=np.int64))
        session_info.append(res['info'])
        print(f"[{i+1}/{len(sessions)}] session {sid}: "
              f"{res['info']['n_neurons']} neurons, {res['info']['n_trials']} trials",
              flush=True)

    data['subject_idx'] = np.array(data['subject_idx'], dtype=np.int64)
    data['metadata'] = {
        'task_description': (
            'Allen Brain Observatory Visual Behavior 2P: head-fixed mice perform a '
            'go/no-go visual change detection task with flashed natural images '
            '(250 ms image, 500 ms gray, 5% omissions) while 2-photon calcium '
            'imaging is performed in visual cortex.  Decoded outputs are the '
            'identity of the currently presented image (8 familiar images), whether '
            'an image change just occurred, running speed and pupil diameter '
            '(each discretized into 5 equal-percentile bins), and the trial outcome '
            '(hit/miss/false alarm/correct reject, constant within a trial).'),
        'time_bin_size': BIN_SIZE * 1000.0,
        'temporal_alignment_event': (
            'image change time on go trials (sham change time on catch trials)'),
        'off_start': OFF_START,
        'off_end': OFF_END,
        'neural_data': (
            'detected calcium events (AllenSDK filtered_events: detected events '
            'convolved with a half-normal filter) summed within 100 ms bins on each '
            'imaging plane\'s own ophys timestamps, then divided per neuron by its '
            'standard deviation across the session'),
        'trial_selection': (
            'go and catch trials; aborted and auto-rewarded trials excluded'),
        'session_selection': (
            'active behavior sessions (passive == False) with the familiar image '
            'set (experience_level == Familiar; all are image set A); planes of a '
            'multi-plane session are merged into a single population'),
        'dataset': 'visual-behavior-ophys-1.1.0',
        'session_info': session_info,
    }

    with open(args.out, 'wb') as f:
        pickle.dump(data, f, protocol=4)
    print('saved', args.out)


if __name__ == '__main__':
    main()
