#!/usr/bin/env python3
import argparse
import sys
import time
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from pynwb import NWBHDF5IO

sys.path.insert(0, 'code')
from allensdk.brain_observatory.behavior.behavior_ophys_experiment import BehaviorOphysExperiment

DATA_ROOT = Path('data/visual-behavior-ophys-1.1.0')
NWB_DIR = DATA_ROOT / 'behavior_ophys_experiments'
META_DIR = DATA_ROOT / 'project_metadata'
TRIAL_OUTCOME_VALUES = ['hit', 'miss', 'false_alarm', 'correct_reject', 'other']


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('outpicklefile')
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument('--full', action='store_true', help='Process all sessions')
    mode.add_argument('--sample', action='store_true', help='Process only 2 sessions for testing')
    ap.add_argument('--show-processing', action='store_true', help='Save processing plots for up to 2 sessions')
    return ap.parse_args()


def list_session_files(sample=False):
    files = sorted(NWB_DIR.glob('*.nwb'))
    return files[:2] if sample else files


def load_experiment(nwb_path):
    io = NWBHDF5IO(str(nwb_path), 'r', load_namespaces=True)
    nwbfile = io.read()
    exp = BehaviorOphysExperiment.from_nwb(nwbfile)
    exp._nwb_io = io
    return exp


def get_cell_matrix(exp):
    dff = exp.dff_traces.copy()
    if isinstance(dff, pd.DataFrame) and 'dff' in dff.columns:
        cell_ids = np.asarray(dff.index)
        traces = [np.asarray(v, dtype=np.float32) for v in dff['dff'].values]
        mat = np.stack(traces, axis=0)
        return mat, cell_ids, 'dff'
    raise RuntimeError('Could not access dff traces')


def pick_pupil_series(eye_tracking):
    if eye_tracking is None or len(eye_tracking) == 0:
        return None, None, None
    cols = list(eye_tracking.columns)
    time_col = 'timestamps' if 'timestamps' in cols else None
    candidate_cols = ['pupil_diameter', 'pupil_area', 'pupil_width', 'pupil_height', 'pupil_radius', 'pupil_size']
    value_col = next((c for c in candidate_cols if c in cols), None)
    if value_col is None:
        value_col = next((c for c in cols if 'pupil' in c.lower() and np.issubdtype(eye_tracking[c].dtype, np.number)), None)
    if time_col is None or value_col is None:
        return None, None, None
    return np.asarray(eye_tracking[time_col], dtype=float), np.asarray(eye_tracking[value_col], dtype=float), value_col


def infer_trial_outcome(row):
    for key in ['hit', 'miss', 'false_alarm', 'correct_reject']:
        if key in row.index and bool(row[key]):
            return key
    return 'other'


def valid_trials_table(trials):
    df = trials.copy()
    for col in ['aborted', 'auto_rewarded', 'auto_rewarded_trial', 'is_auto_rewarded']:
        if col in df.columns:
            df = df[~df[col].fillna(False)]
    keep = None
    if 'go' in df.columns and 'catch' in df.columns:
        keep = df['go'].fillna(False) | df['catch'].fillna(False)
    elif 'trial_type' in df.columns:
        keep = df['trial_type'].astype(str).str.lower().isin(['go', 'catch'])
    if keep is not None:
        df = df[keep]
    return df.reset_index(drop=True)


def get_running_series(run_df):
    cols = list(run_df.columns)
    tcol = 'timestamps' if 'timestamps' in cols else cols[0]
    vcol = 'speed' if 'speed' in cols else cols[1]
    return np.asarray(run_df[tcol], dtype=float), np.asarray(run_df[vcol], dtype=float)


def collect_global_info(files):
    # Efficiency optimization: estimate bin edges from a small subset to avoid loading all NWB files twice.
    image_names = set(['blank', 'omitted'])
    run_vals = []
    pupil_vals = []
    probe_files = files[:min(8, len(files))]
    for p in probe_files:
        exp = load_experiment(p)
        stim = exp.stimulus_presentations.copy()
        if 'image_name' in stim.columns:
            names = stim['image_name'].dropna().astype(str).unique().tolist()
            image_names.update(names)
        rt, rv = get_running_series(exp.running_speed.copy())
        run_vals.append(rv[np.isfinite(rv)])
        try:
            eye = exp.eye_tracking.copy()
            pt, pv, _ = pick_pupil_series(eye)
            if pv is not None:
                pupil_vals.append(pv[np.isfinite(pv)])
        except Exception:
            pass
    run_all = np.concatenate(run_vals) if run_vals else np.array([0.0])
    pupil_all = np.concatenate(pupil_vals) if pupil_vals else np.array([0.0])
    run_edges = np.quantile(run_all, [0, .2, .4, .6, .8, 1])
    pupil_edges = np.quantile(pupil_all, [0, .2, .4, .6, .8, 1])
    image_values = sorted(image_names)
    image_to_idx = {name: i for i, name in enumerate(image_values)}
    outcome_to_idx = {name: i for i, name in enumerate(TRIAL_OUTCOME_VALUES)}
    return image_values, image_to_idx, run_edges, pupil_edges, outcome_to_idx


def build_session(exp, image_to_idx, run_edges, pupil_edges, outcome_to_idx):
    ts = np.asarray(exp.ophys_timestamps, dtype=float)
    neural_mat, cell_ids, neural_source = get_cell_matrix(exp)
    trials = valid_trials_table(exp.trials)
    stim = exp.stimulus_presentations.copy().sort_values('start_time')
    stim_start = np.asarray(stim['start_time'], dtype=float)
    stim_end = stim_start + 0.75
    stim = stim.copy()
    stim['_end_time'] = stim_end
    run_t, run_v = get_running_series(exp.running_speed.copy())
    try:
        eye = exp.eye_tracking.copy()
    except Exception:
        eye = None
    pupil_t, pupil_v, pupil_col = pick_pupil_series(eye)

    session_neural, session_input, session_output = [], [], []
    image_names_seen = []
    trial_outcomes_seen = []

    for _, tr in trials.iterrows():
        start = float(tr['start_time'])
        stop = float(tr['stop_time']) if 'stop_time' in tr.index else float(tr['end_time'])
        idx = np.flatnonzero((ts >= start) & (ts < stop))
        if idx.size < 2:
            continue
        trial_ts = ts[idx]
        trial_neural = neural_mat[:, idx].astype(np.float32, copy=False)

        image_series = np.full(len(trial_ts), image_to_idx['blank'], dtype=np.int64)
        change_series = np.zeros(len(trial_ts), dtype=np.int64)
        stim_trial = stim[(stim['start_time'] < stop) & (stim['_end_time'] > start)]
        prev_name = None
        for _, srow in stim_trial.iterrows():
            s0 = max(start, float(srow['start_time']))
            s1 = min(stop, float(srow['_end_time']))
            smask = (trial_ts >= s0) & (trial_ts < s1)
            name = str(srow['image_name']) if 'image_name' in srow.index else 'blank'
            image_series[smask] = image_to_idx.get(name, image_to_idx['blank'])
            if prev_name is not None and name != prev_name:
                hit = np.flatnonzero(smask)
                if hit.size:
                    change_series[hit[0]] = 1
            prev_name = name
            image_names_seen.append(name)

        run_interp = np.interp(trial_ts, run_t, run_v).astype(np.float32)
        run_bins = np.digitize(run_interp, run_edges[1:-1], right=False).astype(np.int64)

        if pupil_t is not None and pupil_v is not None:
            valid = np.isfinite(pupil_t) & np.isfinite(pupil_v)
            if valid.sum() >= 2:
                pupil_interp = np.interp(trial_ts, pupil_t[valid], pupil_v[valid]).astype(np.float32)
                pupil_bins = np.digitize(pupil_interp, pupil_edges[1:-1], right=False).astype(np.int64)
            else:
                pupil_bins = np.zeros(len(trial_ts), dtype=np.int64)
        else:
            pupil_bins = np.zeros(len(trial_ts), dtype=np.int64)

        outcome = infer_trial_outcome(tr)
        trial_outcomes_seen.append(outcome)
        outcome_idx = outcome_to_idx.get(outcome, outcome_to_idx['other'])
        outcome_series = np.full(len(trial_ts), outcome_idx, dtype=np.int64)

        output = np.vstack([image_series, change_series, run_bins, pupil_bins, outcome_series]).astype(np.int64)
        session_neural.append(trial_neural)
        session_input.append(np.zeros((0, len(trial_ts)), dtype=np.float32))
        session_output.append(output)

    meta = exp.metadata if isinstance(exp.metadata, dict) else {}
    targeted_structure = str(meta.get('targeted_structure', 'unknown'))
    mouse_id = str(meta.get('mouse_id', 'unknown'))
    return {
        'neural': session_neural,
        'input': session_input,
        'output': session_output,
        'cell_ids': cell_ids,
        'neural_source': neural_source,
        'image_names_seen': sorted(set(image_names_seen)),
        'trial_outcomes_seen': sorted(set(trial_outcomes_seen)),
        'targeted_structure': targeted_structure,
        'mouse_id': mouse_id,
    }


def main():
    args = parse_args()
    sample = args.sample or not args.full
    files = list_session_files(sample=sample)
    print(f'processing {len(files)} sessions')
    t0 = time.time()
    image_values, image_to_idx, run_edges, pupil_edges, outcome_to_idx = collect_global_info(files)
    print('n_image_values', len(image_values))
    print('run_edges', run_edges)
    print('pupil_edges', pupil_edges)

    sessions = []
    for i, p in enumerate(files, 1):
        s0 = time.time()
        exp = load_experiment(p)
        sess = build_session(exp, image_to_idx, run_edges, pupil_edges, outcome_to_idx)
        sessions.append(sess)
        print(f'processed {i}/{len(files)} {p.name} trials={len(sess["neural"])} neurons={len(sess["cell_ids"])} source={sess["neural_source"]} time={time.time()-s0:.2f}s')

    subjects = sorted({s['mouse_id'] for s in sessions})
    subject_to_idx = {s: i for i, s in enumerate(subjects)}
    brain_regions = sorted({s['targeted_structure'] for s in sessions})
    region_to_idx = {r: i for i, r in enumerate(brain_regions)}
    dt_ms = float(np.median(np.diff(load_experiment(files[0]).ophys_timestamps)) * 1000.0)

    data = {
        'neural': [s['neural'] for s in sessions],
        'input': [s['input'] for s in sessions],
        'output': [s['output'] for s in sessions],
        'subjects': subjects,
        'subject_idx': np.array([subject_to_idx[s['mouse_id']] for s in sessions], dtype=np.int64),
        'brain_regions': brain_regions,
        'brain_region_idx': [np.full(len(s['cell_ids']), region_to_idx[s['targeted_structure']], dtype=np.int64) for s in sessions],
        'input_names': [],
        'output_names': ['image_identity', 'image_change', 'running_speed_bin', 'pupil_diameter_bin', 'trial_outcome'],
        'output_values': [image_values, ['no_change', 'change'], [f'bin_{i}' for i in range(5)], [f'bin_{i}' for i in range(5)], TRIAL_OUTCOME_VALUES],
        'metadata': {
            'task_description': 'Allen Visual Behavior 2P task; decode image identity, image change, running bin, pupil bin, and trial outcome from neural activity.',
            'time_bin_size': dt_ms,
            'temporal_alignment_event': 'ophys timestamps within experiment-defined trial window',
            'off_start': None,
            'off_end': None,
            'neural_source_preference': 'events_then_dff_fallback',
            'n_sessions': len(sessions),
            'local_subset_only': True,
        }
    }
    with open(args.outpicklefile, 'wb') as f:
        pickle.dump(data, f)
    print(f'saved {args.outpicklefile} in {time.time()-t0:.2f}s')


if __name__ == '__main__':
    main()
