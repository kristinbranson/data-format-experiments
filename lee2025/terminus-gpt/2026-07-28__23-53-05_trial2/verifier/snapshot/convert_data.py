import argparse
import math
import time
import pickle
from pathlib import Path

import joblib
import numpy as np

ANIMALS = ['QLAK-CA1-08','QLAK-CA1-30','QLAK-CA1-50','QLAK-CA1-51','QLAK-CA1-56','QLAK-CA1-74','QLAK-CA1-75']
FPS = 30
SESSION_SECONDS = 40 * 60
TRIAL_SECONDS = 60
FRAMES_PER_TRIAL = FPS * TRIAL_SECONDS
BRAIN_REGION = 'CA1'


def get_env_mat(env):
    if env == 'square':
        return np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.float32)
    elif env == 'o':
        return np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.float32)
    elif env == 't':
        return np.array([[0, 1, 0], [0, 1, 0], [1, 1, 1]], dtype=np.float32)
    elif env == 'u':
        return np.array([[1, 1, 1], [1, 0, 0], [1, 1, 1]], dtype=np.float32)
    elif env == 'rectangle':
        return np.array([[0, 1, 1], [0, 1, 1], [0, 1, 1]], dtype=np.float32)
    elif env == '+':
        return np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.float32)
    elif env == 'i':
        return np.array([[1, 1, 1], [0, 1, 0], [1, 1, 1]], dtype=np.float32)
    elif env == 'l':
        return np.array([[1, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=np.float32)
    elif env == 'bit donut':
        return np.array([[1, 1, 1], [1, 0, 1], [0, 1, 1]], dtype=np.float32)
    elif env == 'glenn':
        return np.array([[1, 1, 1], [1, 0, 0], [1, 0, 1]], dtype=np.float32)
    raise ValueError(f'Unknown environment: {env}')


def infer_valid_frames(position, trace):
    pos_valid = np.all(np.isfinite(position), axis=0)
    neural_valid = np.any(np.isfinite(trace), axis=0)
    valid = pos_valid & neural_valid
    idx = np.where(valid)[0]
    if len(idx) == 0:
        return 0
    return int(idx[-1] + 1)


def discretize_position_3x3(position_xy, arena_size=75.0):
    x = np.clip(position_xy[0], 0, arena_size - 1e-6)
    y = np.clip(position_xy[1], 0, arena_size - 1e-6)
    xb = np.floor((x / arena_size) * 3).astype(int)
    yb = np.floor((y / arena_size) * 3).astype(int)
    xb = np.clip(xb, 0, 2)
    yb = np.clip(yb, 0, 2)
    return (yb * 3 + xb).astype(np.int64)


def session_to_trials(trace_sxn, pos_sxn, env_name, min_trials=2):
    valid_frames = infer_valid_frames(pos_sxn, trace_sxn)
    n_trials = valid_frames // FRAMES_PER_TRIAL
    if n_trials < min_trials:
        return [], [], []
    n_keep = n_trials * FRAMES_PER_TRIAL
    trace_sxn = trace_sxn[:, :n_keep].astype(np.float32, copy=False)
    pos_sxn = pos_sxn[:, :n_keep].astype(np.float32, copy=False)
    finite_neurons = np.all(np.isfinite(trace_sxn), axis=1)
    trace_sxn = trace_sxn[finite_neurons]
    if trace_sxn.shape[0] == 0:
        return [], [], []
    pos_bins = discretize_position_3x3(pos_sxn)
    env_vec = get_env_mat(env_name).reshape(-1).astype(np.float32)
    neural_trials, input_trials, output_trials = [], [], []
    for t in range(n_trials):
        sl = slice(t * FRAMES_PER_TRIAL, (t + 1) * FRAMES_PER_TRIAL)
        neural_trials.append(trace_sxn[:, sl])
        input_trials.append(env_vec.copy())
        output_trials.append(pos_bins[sl][None, :])
    return neural_trials, input_trials, output_trials


def convert_dataset(sample=False):
    animals = ANIMALS[:2] if sample else ANIMALS
    subjects = animals.copy()
    subject_to_idx = {a: i for i, a in enumerate(subjects)}
    neural, inp, out = [], [], []
    subject_idx, brain_region_idx = [], []
    for animal in animals:
        rec = joblib.load(Path('data') / animal)[animal]
        envs = rec['envs'].ravel().tolist()
        traces = rec['trace']
        positions = rec['position']
        for s, env_name in enumerate(envs):
            nt, it, ot = session_to_trials(traces[s], positions[s], env_name)
            if len(nt) < 2:
                continue
            neural.append(nt)
            inp.append(it)
            out.append(ot)
            subject_idx.append(subject_to_idx[animal])
            brain_region_idx.append(np.zeros(nt[0].shape[0], dtype=np.int64))
    data = {
        'neural': neural,
        'input': inp,
        'output': out,
        'subjects': subjects,
        'subject_idx': np.asarray(subject_idx, dtype=np.int64),
        'brain_regions': [BRAIN_REGION],
        'brain_region_idx': brain_region_idx,
        'input_names': [f'geom_bin_{i}' for i in range(9)],
        'output_names': ['position_bin'],
        'output_values': [[f'bin_{i}' for i in range(9)]],
        'metadata': {
            'task_description': 'Decode mouse position (3x3 spatial bin) from CA1 calcium-event activity with environment geometry as static trial input.',
            'time_bin_size': 1000.0 / FPS,
            'temporal_alignment_event': 'session time; sessions split into consecutive 1-minute chunks',
            'off_start': 0.0,
            'off_end': float(TRIAL_SECONDS),
            'session_duration_s': SESSION_SECONDS,
            'trial_duration_s': TRIAL_SECONDS,
            'fps': FPS,
            'source_trace_representation': 'binary rising-phase vectors treated as firing rates',
        }
    }
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('outpicklefile')
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument('--full', action='store_true')
    mode.add_argument('--sample', action='store_true')
    ap.add_argument('--show-processing', action='store_true')
    args = ap.parse_args()
    sample = args.sample and not args.full
    t0 = time.time()
    data = convert_dataset(sample=sample)
    with open(args.outpicklefile, 'wb') as f:
        pickle.dump(data, f)
    print('saved', args.outpicklefile)
    print('sessions', len(data['neural']))
    print('trials_total', sum(len(s) for s in data['neural']))
    print('subjects', len(data['subjects']))
    print('elapsed_sec', round(time.time() - t0, 3))
    if args.show_processing:
        print('show-processing requested; plotting not yet implemented')


if __name__ == '__main__':
    main()
