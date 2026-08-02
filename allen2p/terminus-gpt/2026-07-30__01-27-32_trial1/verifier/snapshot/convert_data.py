#!/usr/bin/env python3
import argparse
import pickle
import time
from pathlib import Path

import warnings
warnings.filterwarnings('ignore', message='Ignoring the following cached namespace\(s\) because another version is already loaded:')
warnings.filterwarnings('ignore', message='Downcasting object dtype arrays on \.fillna, \.ffill, \.bfill is deprecated.*')

import numpy as np
import pandas as pd

from allensdk.brain_observatory.behavior.behavior_ophys_experiment import BehaviorOphysExperiment

DATASET_DIR = Path('data/visual-behavior-ophys-1.1.0')
META_DIR = DATASET_DIR / 'project_metadata'
EXPT_TABLE = META_DIR / 'ophys_experiment_table.csv'
EXPT_DIR = DATASET_DIR / 'behavior_ophys_experiments'


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('outpicklefile')
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument('--full', action='store_true', help='Process all sessions')
    mode.add_argument('--sample', action='store_true', help='Process only 2 sessions for testing')
    ap.add_argument('--show-processing', action='store_true', help='Plot visualizations for up to 2 sessions')
    return ap.parse_args()


def load_experiment_table():
    return pd.read_csv(EXPT_TABLE)


def get_nwb_path(exp_id: int) -> Path:
    return EXPT_DIR / f'behavior_ophys_experiment_{int(exp_id)}.nwb'


def load_experiment(exp_id: int) -> BehaviorOphysExperiment:
    return BehaviorOphysExperiment.from_nwb_path(str(get_nwb_path(exp_id)))


def choose_signal(exp):
    dff = exp.dff_traces.copy()
    if isinstance(dff, pd.DataFrame) and len(dff) > 0:
        return 'dff', dff
    events = exp.events.copy()
    return 'events', events


def extract_neural_matrix(signal_df, signal_kind):
    preferred = ['events', 'filtered_events', 'dff']
    col = None
    for c in preferred:
        if c in signal_df.columns:
            col = c
            break
    if col is None:
        for c in signal_df.columns:
            v = signal_df.iloc[0][c]
            if isinstance(v, (np.ndarray, list, tuple)):
                col = c
                break
    if col is None:
        raise RuntimeError(f'Could not find neural trace column in {list(signal_df.columns)}')
    arrays = [np.asarray(v, dtype=np.float32) for v in signal_df[col].values]
    n_t = min(len(a) for a in arrays)
    return np.stack([a[:n_t] for a in arrays], axis=0)


def pick_pupil_column(eye_tracking):
    candidates = ['pupil_diameter', 'pupil_area', 'pupil_width', 'pupil_radius']
    for c in candidates:
        if c in eye_tracking.columns:
            return c
    for c in eye_tracking.columns:
        if 'pupil' in c.lower() and pd.api.types.is_numeric_dtype(eye_tracking[c]):
            return c
    return None


def make_percentile_bins(values, n_bins=5):
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.array([0, 1, 2, 3], dtype=float)
    qs = np.linspace(0, 100, n_bins + 1)[1:-1]
    edges = np.percentile(vals, qs)
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = np.nextafter(edges[i - 1], np.inf)
    return edges


def digitize(values, edges):
    vals = np.asarray(values, dtype=float)
    out = np.digitize(vals, edges, right=False).astype(np.int64)
    out[~np.isfinite(vals)] = 0
    return out


def nearest_assign(sample_times, source_times, source_values):
    source_times = np.asarray(source_times, dtype=float)
    source_values = np.asarray(source_values)
    idx = np.searchsorted(source_times, sample_times, side='left')
    idx = np.clip(idx, 0, len(source_times) - 1)
    prev_idx = np.clip(idx - 1, 0, len(source_times) - 1)
    choose_prev = np.abs(sample_times - source_times[prev_idx]) < np.abs(sample_times - source_times[idx])
    idx[choose_prev] = prev_idx[choose_prev]
    return source_values[idx]


def interval_assign(sample_times, starts, stops, values, default=-1):
    out = np.full(sample_times.shape, default, dtype=np.int64)
    for s, e, v in zip(np.asarray(starts, float), np.asarray(stops, float), np.asarray(values, np.int64)):
        m = (sample_times >= s) & (sample_times < e)
        out[m] = v
    return out


def ensure_stim_stop_time(stim, use_full_interval=True):
    stim = stim.sort_values('start_time').copy()
    starts = stim['start_time'].to_numpy(dtype=float) if 'start_time' in stim.columns else np.array([], dtype=float)
    if len(starts) == 0:
        stim['stop_time'] = np.array([], dtype=float)
        return stim
    diffs = np.diff(starts)
    default_interval = float(np.nanmedian(diffs)) if len(diffs) else 0.75
    if use_full_interval:
        next_starts = np.r_[starts[1:], starts[-1] + default_interval]
        stim['stop_time'] = next_starts
        return stim
    if 'stop_time' in stim.columns:
        return stim
    if 'duration' in stim.columns:
        stim['stop_time'] = stim['start_time'] + pd.to_numeric(stim['duration'], errors='coerce').fillna(0.25)
        return stim
    stim['stop_time'] = np.r_[starts[1:], starts[-1] + default_interval]
    return stim


def get_running_df(exp):
    run_df = exp.running_speed.copy()
    cols = {c.lower(): c for c in run_df.columns}
    tcol = cols.get('timestamps', 'timestamps')
    scol = cols.get('speed', None)
    if scol is None:
        for c in run_df.columns:
            if 'speed' in c.lower():
                scol = c
                break
    return run_df.rename(columns={tcol: 'timestamps', scol: 'speed'})[['timestamps', 'speed']]


def get_eye_df(exp):
    eye_df = exp.eye_tracking.copy()
    if 'timestamps' not in eye_df.columns:
        for c in eye_df.columns:
            if 'time' in c.lower():
                eye_df = eye_df.rename(columns={c: 'timestamps'})
                break
    return eye_df


def build_trial_output(trial_row, sample_times, stim_df, run_df, eye_df, run_edges, pupil_edges, image_to_idx, blank_idx):
    stim = ensure_stim_stop_time(stim_df)
    if 'omitted' in stim.columns:
        stim_non_omitted = stim[~stim['omitted'].fillna(False)].copy()
    else:
        stim_non_omitted = stim

    image_vals = [image_to_idx.get(str(x), blank_idx) for x in stim_non_omitted['image_name'].astype(str).values]
    image_identity = interval_assign(sample_times, stim_non_omitted['start_time'].values, stim_non_omitted['stop_time'].values, image_vals, default=blank_idx)

    image_change = np.zeros(sample_times.shape, dtype=np.int64)
    if 'is_change' in stim_non_omitted.columns:
        for ct in stim_non_omitted.loc[stim_non_omitted['is_change'].fillna(False), 'start_time'].values:
            idx = np.searchsorted(sample_times, ct, side='left')
            if idx < len(image_change):
                image_change[idx] = 1

    run_vals = nearest_assign(sample_times, run_df['timestamps'].values, run_df['speed'].values)
    run_bins = digitize(run_vals, run_edges)

    pupil_col = pick_pupil_column(eye_df)
    if pupil_col is None:
        pupil_bins = np.zeros(sample_times.shape, dtype=np.int64)
    else:
        eye_valid = eye_df[['timestamps', pupil_col]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(eye_valid) == 0:
            pupil_bins = np.zeros(sample_times.shape, dtype=np.int64)
        else:
            pupil_vals = nearest_assign(sample_times, eye_valid['timestamps'].values, eye_valid[pupil_col].values)
            pupil_bins = digitize(pupil_vals, pupil_edges)

    if bool(trial_row.get('hit', False)):
        outcome = 0
    elif bool(trial_row.get('miss', False)):
        outcome = 1
    elif bool(trial_row.get('false_alarm', False)):
        outcome = 2
    elif bool(trial_row.get('correct_reject', False)):
        outcome = 3
    else:
        outcome = 0
    trial_outcome = np.full(sample_times.shape, outcome, dtype=np.int64)

    return np.stack([image_identity, image_change, run_bins, pupil_bins, trial_outcome], axis=0)


def process_experiment(exp_id, meta_row, run_edges, pupil_edges, image_to_idx, blank_idx):
    exp = load_experiment(int(exp_id))
    signal_kind, signal_df = choose_signal(exp)
    neural = extract_neural_matrix(signal_df, signal_kind)
    ophys_timestamps = np.asarray(exp.ophys_timestamps, dtype=float)
    n_t = min(neural.shape[1], len(ophys_timestamps))
    neural = neural[:, :n_t]
    ophys_timestamps = ophys_timestamps[:n_t]

    trials = exp.trials.copy()
    keep = trials['go'].fillna(False) | trials['catch'].fillna(False)
    keep &= ~trials['aborted'].fillna(False)
    keep &= ~trials['auto_rewarded'].fillna(False)
    trials = trials[keep].copy()

    stim = ensure_stim_stop_time(exp.stimulus_presentations.copy())
    run_df = get_running_df(exp)
    eye_df = get_eye_df(exp)

    sess_neural, sess_input, sess_output = [], [], []
    for _, tr in trials.iterrows():
        start = float(tr['start_time'])
        stop = float(tr['stop_time'])
        m = (ophys_timestamps >= start) & (ophys_timestamps < stop)
        if m.sum() < 2:
            continue
        sample_times = ophys_timestamps[m]
        trial_neural = neural[:, m].astype(np.float32)
        trial_stim = stim[(stim['start_time'] < stop) & (stim['stop_time'] > start)].copy()
        trial_output = build_trial_output(tr, sample_times, trial_stim, run_df, eye_df, run_edges, pupil_edges, image_to_idx, blank_idx)
        trial_input = np.zeros((0, len(sample_times)), dtype=np.float32)
        sess_neural.append(trial_neural)
        sess_input.append(trial_input)
        sess_output.append(trial_output.astype(np.int64))

    region = str(meta_row.get('targeted_structure', 'unknown'))
    subject = str(meta_row.get('mouse_id', 'unknown'))
    brain_region_idx = np.zeros((neural.shape[0],), dtype=np.int64)
    return {
        'exp_id': int(exp_id),
        'subject': subject,
        'region': region,
        'neural': sess_neural,
        'input': sess_input,
        'output': sess_output,
        'brain_region_idx': brain_region_idx,
        'signal_kind': signal_kind,
        'n_trials': len(sess_neural),
        'dt_ms': float(np.median(np.diff(ophys_timestamps)) * 1000.0),
    }


def collect_global_info(exp_table):
    all_images = set()
    run_vals = []
    pupil_vals = []
    chosen_ids = []
    for _, row in exp_table.iterrows():
        exp_id = int(row['ophys_experiment_id'])
        nwb_path = get_nwb_path(exp_id)
        if not nwb_path.exists():
            continue
        exp = load_experiment(exp_id)
        trials = exp.trials.copy()
        keep = trials['go'].fillna(False) | trials['catch'].fillna(False)
        keep &= ~trials['aborted'].fillna(False)
        keep &= ~trials['auto_rewarded'].fillna(False)
        if keep.sum() < 2:
            continue
        chosen_ids.append(exp_id)
        stim = ensure_stim_stop_time(exp.stimulus_presentations.copy())
        if 'omitted' in stim.columns:
            stim = stim[~stim['omitted'].fillna(False)]
        if 'image_name' in stim.columns:
            all_images.update(stim['image_name'].astype(str).unique().tolist())
        run_df = get_running_df(exp)
        run_vals.append(pd.to_numeric(run_df['speed'], errors='coerce').values)
        eye_df = get_eye_df(exp)
        pupil_col = pick_pupil_column(eye_df)
        if pupil_col is not None:
            pupil_vals.append(pd.to_numeric(eye_df[pupil_col], errors='coerce').values)
    image_names = ['blank'] + sorted(all_images)
    image_to_idx = {name: i for i, name in enumerate(image_names)}
    run_edges = make_percentile_bins(np.concatenate(run_vals) if run_vals else np.array([0.0]))
    pupil_edges = make_percentile_bins(np.concatenate(pupil_vals) if pupil_vals else np.array([0.0]))
    return chosen_ids, image_names, image_to_idx, run_edges, pupil_edges


def main():
    args = parse_args()
    t0 = time.time()
    exp_table = load_experiment_table()
    exp_table = exp_table[~exp_table['session_type'].astype(str).str.contains('passive', case=False, na=False)].copy()
    if args.sample:
        exp_table = exp_table.head(2).copy()

    chosen_ids, image_names, image_to_idx, run_edges, pupil_edges = collect_global_info(exp_table)
    blank_idx = image_to_idx['blank']
    if args.sample:
        chosen_ids = chosen_ids[:2]
    row_lookup = {int(r['ophys_experiment_id']): r for _, r in exp_table.iterrows()}

    sessions = []
    for i, exp_id in enumerate(chosen_ids, 1):
        t1 = time.time()
        sess = process_experiment(exp_id, row_lookup[exp_id], run_edges, pupil_edges, image_to_idx, blank_idx)
        if sess['n_trials'] >= 2:
            sessions.append(sess)
        print(f'processed {i}/{len(chosen_ids)} exp_id={exp_id} kept_trials={sess["n_trials"]} signal={sess["signal_kind"]} dt={time.time()-t1:.2f}s')

    subjects = sorted({s['subject'] for s in sessions})
    subject_to_idx = {s: i for i, s in enumerate(subjects)}
    brain_regions = sorted({s['region'] for s in sessions})
    region_to_idx = {r: i for i, r in enumerate(brain_regions)}
    time_bin_size = float(np.median([s['dt_ms'] for s in sessions])) if sessions else np.nan

    data = {
        'neural': [s['neural'] for s in sessions],
        'input': [s['input'] for s in sessions],
        'output': [s['output'] for s in sessions],
        'subjects': subjects,
        'subject_idx': np.asarray([subject_to_idx[s['subject']] for s in sessions], dtype=np.int64),
        'brain_regions': brain_regions,
        'brain_region_idx': [np.full((len(s['brain_region_idx']),), region_to_idx[s['region']], dtype=np.int64) for s in sessions],
        'input_names': [],
        'output_names': ['image_identity', 'image_change', 'running_speed_bin', 'pupil_diameter_bin', 'trial_outcome'],
        'output_values': [
            image_names,
            ['no_change', 'change'],
            [f'bin_{i}' for i in range(5)],
            [f'bin_{i}' for i in range(5)],
            ['hit', 'miss', 'false_alarm', 'correct_reject'],
        ],
        'metadata': {
            'task_description': 'Allen Visual Behavior change-detection task; predict image identity, image change, running bin, pupil bin, and trial outcome from ophys activity.',
            'time_bin_size': time_bin_size,
            'temporal_alignment_event': 'ophys timestamps within each behaviorally defined trial',
            'off_start': 0.0,
            'off_end': None,
            'signal_type': 'dff_preferred_else_events',
            'excluded_sessions': 'passive sessions excluded; aborted and auto_rewarded trials excluded',
            'n_sessions': len(sessions),
        }
    }

    with open(args.outpicklefile, 'wb') as f:
        pickle.dump(data, f)
    print(f'saved {args.outpicklefile} with {len(sessions)} sessions in {time.time()-t0:.2f}s')


if __name__ == '__main__':
    main()
