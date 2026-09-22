#!/usr/bin/env python3
import argparse
import pickle
import time
from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
from pynwb import NWBHDF5IO

BIN = 0.05
T_START = -2.5
T_END = 1.5
BIN_EDGES = np.arange(T_START, T_END + BIN + 1e-9, BIN)
BIN_CENTERS = BIN_EDGES[:-1] + BIN / 2


def find_first(cols, candidates):
    lower = {c.lower(): c for c in cols}
    for cand in candidates:
        for c in cols:
            if c.lower() == cand.lower():
                return c
        for c in cols:
            if cand.lower() in c.lower():
                return c
    return None


def get_subject_name(nwb, path):
    sid = getattr(getattr(nwb, 'subject', None), 'subject_id', None)
    return sid if sid is not None else path.parent.name


def get_event_times(nwb, key):
    ts = nwb.acquisition['BehavioralEvents'].time_series[key]
    return np.asarray(ts.timestamps[:], dtype=float)


def get_trial_table_dict(nwb):
    cols = list(nwb.trials.colnames)
    out = {}
    for c in cols:
        out[c] = np.asarray(nwb.trials[c][:])
    return out


def get_unit_mask_and_regions(nwb):
    cols = list(nwb.units.colnames)
    good_col = find_first(cols, ['good', 'quality', 'label', 'unit_quality'])
    region_col = find_first(cols, ['location', 'brain_region', 'structure', 'ccf_acronym', 'acronym'])
    n_units = len(nwb.units.id[:])
    mask = np.ones(n_units, dtype=bool)
    if good_col is not None:
        vals = np.asarray(nwb.units[good_col][:])
        if vals.dtype.kind in 'OUS':
            sval = np.array([str(v).lower() for v in vals])
            mask = np.array([('good' in v) or (v == '1') or (v == 'true') for v in sval], dtype=bool)
        else:
            mask = vals.astype(bool)
    regions = np.array(['unknown'] * n_units, dtype=object)
    if region_col is not None:
        regions = np.array([str(v) for v in nwb.units[region_col][:]], dtype=object)
    return mask, regions, good_col, region_col


def spike_times_list(nwb, unit_mask):
    idx = np.where(unit_mask)[0]
    spikes = []
    for i in idx:
        spikes.append(np.asarray(nwb.units['spike_times'][i], dtype=float))
    return idx, spikes


def infer_trial_intervals(nwb):
    td = get_trial_table_dict(nwb)
    cols = list(td.keys())
    start_col = find_first(cols, ['start_time', 'start'])
    stop_col = find_first(cols, ['stop_time', 'stop'])
    if start_col is None or stop_col is None:
        raise RuntimeError('Could not find trial start/stop columns')
    return np.asarray(td[start_col], float), np.asarray(td[stop_col], float), td


def assign_events_to_trials(event_times, trial_starts, trial_stops):
    out = np.full(len(trial_starts), np.nan, dtype=float)
    for i, (a, b) in enumerate(zip(trial_starts, trial_stops)):
        m = (event_times >= a) & (event_times <= b)
        if np.any(m):
            out[i] = event_times[m][0]
    return out


def bin_spikes_for_trial(spike_times, align_time):
    arr = np.zeros((len(spike_times), len(BIN_CENTERS)), dtype=np.float32)
    rel_edges = align_time + BIN_EDGES
    for i, st in enumerate(spike_times):
        arr[i] = np.histogram(st, bins=rel_edges)[0].astype(np.float32) / BIN
    return arr


def build_inputs(go_time, sample_start, photostim_intervals):
    tone_time = sample_start
    time_from_tone = (go_time + BIN_CENTERS) - tone_time
    phot = np.zeros(len(BIN_CENTERS), dtype=np.float32)
    abs_centers = go_time + BIN_CENTERS
    for a, b in photostim_intervals:
        phot[(abs_centers >= a) & (abs_centers <= b)] = 1.0
    return np.vstack([time_from_tone.astype(np.float32), phot])


def get_photostim_intervals(nwb):
    keys = list(nwb.acquisition['BehavioralEvents'].time_series.keys())
    if 'photostim_start_times' not in keys or 'photostim_stop_times' not in keys:
        return np.empty((0, 2), dtype=float)
    s = get_event_times(nwb, 'photostim_start_times')
    e = get_event_times(nwb, 'photostim_stop_times')
    n = min(len(s), len(e))
    return np.c_[s[:n], e[:n]] if n else np.empty((0, 2), dtype=float)


def tongue_series(nwb):
    ts = nwb.acquisition['BehavioralTimeSeries'].time_series['Camera0_side_TongueTracking']
    data = np.asarray(ts.data[:])
    t = np.asarray(ts.timestamps[:], dtype=float)
    return t, data


def interpolate_tongue_y(nwb, go_time):
    t, data = tongue_series(nwb)
    # assume columns include x,y,likelihood or similar; use second column as y when available
    y = data[:, 1] if data.ndim > 1 and data.shape[1] > 1 else data[:, 0]
    vis = np.ones_like(y, dtype=bool)
    if data.ndim > 1 and data.shape[1] > 2:
        vis = np.asarray(data[:, 2] > 0.5)
    sample_t = go_time + BIN_CENTERS
    y_interp = np.interp(sample_t, t, y, left=np.nan, right=np.nan)
    vis_interp = np.interp(sample_t, t, vis.astype(float), left=0, right=0) > 0.5
    y_interp[~vis_interp] = np.nan
    return y_interp


def discretize_tongue_session(trial_y_list):
    all_y = np.concatenate([y[np.isfinite(y)] for y in trial_y_list if np.any(np.isfinite(y))]) if trial_y_list else np.array([])
    if len(all_y) == 0:
        return [np.full(len(BIN_CENTERS), 3, dtype=np.int64) for _ in trial_y_list], (np.nan, np.nan)
    q40, q60 = np.percentile(all_y, [40, 60])
    out = []
    for y in trial_y_list:
        d = np.full(len(y), 3, dtype=np.int64)
        m = np.isfinite(y)
        d[m & (y < q40)] = 0
        d[m & (y >= q40) & (y <= q60)] = 1
        d[m & (y > q60)] = 2
        out.append(d)
    return out, (float(q40), float(q60))


def infer_trial_labels(td):
    cols = list(td.keys())
    choice_col = find_first(cols, ['choice', 'lick_direction', 'response_side'])
    outcome_col = find_first(cols, ['outcome', 'trial_outcome', 'result', 'correctness'])
    early_col = find_first(cols, ['early', 'early_lick'])
    return choice_col, outcome_col, early_col


def map_choice(v):
    s = str(v).lower()
    if 'left' in s:
        return 0
    if 'right' in s:
        return 1
    if 'no' in s or 'ignore' in s or 'miss' in s or s in ('nan', ''):
        return 2
    try:
        iv = int(v)
        if iv == 0:
            return 0
        if iv == 1:
            return 1
    except Exception:
        pass
    return 2


def map_outcome(v):
    s = str(v).lower()
    if 'ignore' in s:
        return 0
    if 'miss' in s:
        return 1
    if 'hit' in s or 'correct' in s or s == '1':
        return 2
    try:
        iv = int(v)
        return 2 if iv == 1 else 1
    except Exception:
        return 0


def map_early(v):
    s = str(v).lower()
    if 'true' in s or 'yes' in s or s == '1' or 'early' in s:
        return 1
    try:
        return int(bool(int(v)))
    except Exception:
        return 0


def process_session(path, show_processing=False):
    io = NWBHDF5IO(str(path), 'r', load_namespaces=True)
    nwb = io.read()
    subj = get_subject_name(nwb, path)
    trial_starts, trial_stops, td = infer_trial_intervals(nwb)
    go_times = assign_events_to_trials(get_event_times(nwb, 'go_start_times'), trial_starts, trial_stops)
    sample_times = assign_events_to_trials(get_event_times(nwb, 'sample_start_times'), trial_starts, trial_stops)
    phot_int = get_photostim_intervals(nwb)
    unit_mask, regions, good_col, region_col = get_unit_mask_and_regions(nwb)
    kept_unit_idx, spikes = spike_times_list(nwb, unit_mask)
    choice_col, outcome_col, early_col = infer_trial_labels(td)

    neural_trials, input_trials, output_trials = [], [], []
    tongue_y_trials = []
    kept_trial_idx = []
    for i, go in enumerate(go_times):
        if not np.isfinite(go):
            continue
        neural_trials.append(bin_spikes_for_trial(spikes, go).astype(np.float32))
        input_trials.append(build_inputs(go, sample_times[i] if np.isfinite(sample_times[i]) else go - 1.5, phot_int).astype(np.float32))
        tongue_y_trials.append(interpolate_tongue_y(nwb, go))
        kept_trial_idx.append(i)

    tongue_disc, tongue_q = discretize_tongue_session(tongue_y_trials)
    for j, i in enumerate(kept_trial_idx):
        choice = map_choice(td[choice_col][i]) if choice_col is not None else 2
        outcome = map_outcome(td[outcome_col][i]) if outcome_col is not None else 0
        early = map_early(td[early_col][i]) if early_col is not None else 0
        out = np.vstack([
            np.full(len(BIN_CENTERS), choice, dtype=np.uint8),
            np.full(len(BIN_CENTERS), outcome, dtype=np.uint8),
            np.full(len(BIN_CENTERS), early, dtype=np.uint8),
            tongue_disc[j].astype(np.uint8),
        ])
        output_trials.append(out)

    neural_trials = [x.T.astype(np.float32, copy=False) for x in neural_trials]
    reg_kept = regions[kept_unit_idx] if len(kept_unit_idx) else np.array([], dtype=object)
    io.close()

    info = {
        'session_id': path.stem,
        'subject': subj,
        'n_trials': len(neural_trials),
        'n_units_kept': len(kept_unit_idx),
        'good_col': good_col,
        'region_col': region_col,
        'choice_col': choice_col,
        'outcome_col': outcome_col,
        'early_col': early_col,
        'tongue_quantiles': tongue_q,
    }

    if show_processing and len(neural_trials) > 0:
        fig, axs = plt.subplots(3, 1, figsize=(10, 8), constrained_layout=True)
        axs[0].imshow(neural_trials[0], aspect='auto', origin='lower')
        axs[0].set_title(f"{path.stem} neural trial0")
        axs[1].plot(BIN_CENTERS, input_trials[0][0], label='time_from_tone')
        axs[1].plot(BIN_CENTERS, input_trials[0][1], label='photostim')
        axs[1].legend()
        axs[2].plot(BIN_CENTERS, output_trials[0][3], label='tongue_y_disc')
        axs[2].plot(BIN_CENTERS, output_trials[0][0], label='choice')
        axs[2].legend()
        fig.savefig(f"/app/processing_{path.stem}.png", dpi=150)
        plt.close(fig)

    return neural_trials, input_trials, output_trials, subj, reg_kept, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('outpicklefile')
    g = ap.add_mutually_exclusive_group()
    g.add_argument('--full', action='store_true')
    g.add_argument('--sample', action='store_true')
    ap.add_argument('--show-processing', action='store_true')
    args = ap.parse_args()

    t0 = time.time()
    files = sorted(Path('/app/data').rglob('*.nwb'))
    if args.sample:
        files = files[:2]
    subjects = []
    subject_to_idx = {}
    brain_regions = []
    region_to_idx = {}

    data = {
        'neural': [],
        'input': [],
        'output': [],
        'subjects': subjects,
        'subject_idx': [],
        'brain_regions': brain_regions,
        'brain_region_idx': [],
        'input_names': ['time_from_tone_onset_sec', 'photostimulation_on'],
        'output_names': ['choice', 'outcome', 'early_lick', 'tongue_y_position'],
        'output_values': [
            ['left', 'right', 'no_lick'],
            ['ignore', 'miss', 'hit'],
            ['no', 'yes'],
            ['lt_40pct', '40_to_60pct', 'gt_60pct', 'not_visible'],
        ],
        'metadata': {
            'task_description': 'Auditory delayed-response task with Neuropixels ephys, licking choice, photoinhibition, and tongue tracking.',
            'time_bin_size': 50.0,
            'temporal_alignment_event': 'Go cue onset',
            'off_start': T_START,
            'off_end': T_END,
            'n_timepoints': len(BIN_CENTERS),
            'bin_centers_sec': BIN_CENTERS.astype(np.float32),
            'source_data_dir': '/app/data',
            'notes': 'Initial conversion implementation; field mappings may require refinement after sample validation.',
            'session_info': [],
        },
    }

    for path in files:
        print('PROCESS', path)
        sess = process_session(path, show_processing=args.show_processing)
        neural_trials, input_trials, output_trials, subj, reg_kept, info = sess
        if len(neural_trials) < 2 or len(reg_kept) == 0:
            print('SKIP insufficient trials or units', path)
            continue
        if subj not in subject_to_idx:
            subject_to_idx[subj] = len(subjects)
            subjects.append(subj)
        data['subject_idx'].append(subject_to_idx[subj])
        data['neural'].append(neural_trials)
        data['input'].append(input_trials)
        data['output'].append(output_trials)
        sess_region_idx = np.zeros(len(reg_kept), dtype=np.int64)
        for i, r in enumerate(reg_kept):
            if r not in region_to_idx:
                region_to_idx[r] = len(brain_regions)
                brain_regions.append(r)
            sess_region_idx[i] = region_to_idx[r]
        data['brain_region_idx'].append(sess_region_idx)
        data['metadata']['session_info'].append(info)

    data['subject_idx'] = np.asarray(data['subject_idx'], dtype=np.int64)
    with open(args.outpicklefile, 'wb') as f:
        pickle.dump(data, f)
    print('WROTE', args.outpicklefile)
    print('sessions', len(data['neural']), 'subjects', len(data['subjects']), 'regions', len(data['brain_regions']))
    print('elapsed_sec', time.time() - t0)

if __name__ == '__main__':
    main()
