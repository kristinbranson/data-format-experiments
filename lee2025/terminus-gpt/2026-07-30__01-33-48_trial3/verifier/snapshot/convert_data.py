import argparse
import os
import time
import pickle
from pathlib import Path

import joblib
import numpy as np
import matplotlib.pyplot as plt

FPS = 30
TRIAL_SECONDS = 60
FRAMES_PER_TRIAL = FPS * TRIAL_SECONDS
ARENA_SIZE_CM = 75.0
N_POS_BINS = 3

ANIMALS = [
    'QLAK-CA1-08', 'QLAK-CA1-30', 'QLAK-CA1-50',
    'QLAK-CA1-51', 'QLAK-CA1-56', 'QLAK-CA1-74', 'QLAK-CA1-75'
]


def load_animal(data_dir, animal):
    return joblib.load(Path(data_dir) / animal)[animal]


def flatten_env_name(x):
    arr = np.array(x)
    if arr.size == 1:
        return str(arr.reshape(-1)[0])
    return str(x)


def blocked_to_vec(blocked_entry):
    vec = np.zeros(9, dtype=np.float32)
    flat = np.array(blocked_entry, dtype=object).reshape(-1)
    vals = []
    for item in flat:
        arr = np.array(item).reshape(-1)
        for v in arr:
            if np.isnan(v):
                continue
            vals.append(int(v))
    if len(vals) == 1 and vals[0] == -1:
        return vec
    for v in vals:
        if 0 <= v <= 8:
            vec[v] = 1.0
    return vec


def position_to_bins_3x3(position_xy):
    pos = np.asarray(position_xy, dtype=np.float32)
    x = pos[0]
    y = pos[1]
    eps = 1e-6
    xbin = np.floor(np.clip(x, 0, ARENA_SIZE_CM - eps) / (ARENA_SIZE_CM / N_POS_BINS)).astype(int)
    ybin = np.floor(np.clip(y, 0, ARENA_SIZE_CM - eps) / (ARENA_SIZE_CM / N_POS_BINS)).astype(int)
    xbin = np.clip(xbin, 0, N_POS_BINS - 1)
    ybin = np.clip(ybin, 0, N_POS_BINS - 1)
    return (ybin * N_POS_BINS + xbin).astype(np.int64)


def split_session_into_trials(trace_day, pos_day, blocked_vec):
    n_frames = min(trace_day.shape[1], pos_day.shape[1])
    n_trials = n_frames // FRAMES_PER_TRIAL
    used = n_trials * FRAMES_PER_TRIAL
    trace_day = trace_day[:, :used]
    pos_day = pos_day[:, :used]
    pos_bins = position_to_bins_3x3(pos_day)

    neural_trials = []
    input_trials = []
    output_trials = []
    for i in range(n_trials):
        s = i * FRAMES_PER_TRIAL
        e = (i + 1) * FRAMES_PER_TRIAL
        neural_trials.append(trace_day[:, s:e].astype(np.uint8, copy=False))
        input_trials.append(blocked_vec.copy())
        output_trials.append(pos_bins[s:e][None, :].astype(np.uint8, copy=False))
    return neural_trials, input_trials, output_trials, n_frames, used, n_trials


def make_processing_plot(session_id, pos_day, pos_bins, blocked_vec, neural_trial0, out_path):
    fig, ax = plt.subplots(2, 2, figsize=(10, 8))
    ax[0, 0].plot(pos_day[0, :3000], pos_day[1, :3000], lw=0.5)
    ax[0, 0].set_title(f'{session_id}: trajectory (first 3000 frames)')
    ax[0, 0].set_xlabel('x (cm)')
    ax[0, 0].set_ylabel('y (cm)')
    ax[0, 0].set_xlim(0, ARENA_SIZE_CM)
    ax[0, 0].set_ylim(0, ARENA_SIZE_CM)

    h = np.bincount(pos_bins, minlength=9).reshape(3, 3)
    im = ax[0, 1].imshow(h, origin='lower')
    ax[0, 1].set_title('3x3 position-bin occupancy')
    plt.colorbar(im, ax=ax[0, 1], fraction=0.046, pad=0.04)

    ax[1, 0].bar(np.arange(9), blocked_vec)
    ax[1, 0].set_title('Blocked partition input vector')
    ax[1, 0].set_xlabel('partition idx')
    ax[1, 0].set_ylim(0, 1.1)

    show_neurons = min(30, neural_trial0.shape[0])
    ax[1, 1].imshow(neural_trial0[:show_neurons], aspect='auto', interpolation='nearest')
    ax[1, 1].set_title(f'Neural events first trial ({show_neurons} neurons)')
    ax[1, 1].set_xlabel('time (frames)')
    ax[1, 1].set_ylabel('neurons')

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def convert_dataset(data_dir, sample=False, show_processing=False):
    animals = ANIMALS[:2] if sample else ANIMALS
    data = {
        'neural': [],
        'input': [],
        'output': [],
        'subjects': animals.copy(),
        'subject_idx': [],
        'brain_regions': ['CA1'],
        'brain_region_idx': [],
        'input_names': [f'blocked_partition_{i}' for i in range(9)],
        'output_names': ['position_bin_3x3'],
        'output_values': [[f'bin_{i}' for i in range(9)]],
        'metadata': {
            'task_description': 'Decode mouse position (3x3 spatial bins) from CA1 neural activity with blocked-geometry context.',
            'time_bin_size': 1000.0 / FPS,
            'temporal_alignment_event': 'Continuous session split into contiguous 1-minute windows from session start.',
            'off_start': 0.0,
            'off_end': float(TRIAL_SECONDS),
            'fps': FPS,
            'trial_seconds': TRIAL_SECONDS,
            'source_signal': 'Binarized rising-phase calcium event vector treated as firing rate',
            'source_sessions_are_continuous': True,
        }
    }

    session_info = []
    plotted = 0
    t0 = time.time()
    for subj_idx, animal in enumerate(animals):
        print(f'Loading {animal}...')
        rec = load_animal(data_dir, animal)
        envs = np.array(rec['envs']).reshape(-1)
        trace = np.asarray(rec['trace'])
        position = np.asarray(rec['position'])
        blocked = rec['blocked']
        n_days = len(envs)

        for day in range(n_days):
            session_id = f'{animal}_day{day:02d}_{flatten_env_name(envs[day])}'
            blocked_vec = blocked_to_vec(blocked[day])
            trace_day = np.asarray(trace[day], dtype=np.float32)
            pos_day = np.asarray(position[day], dtype=np.float32)
            valid_neurons = np.all(np.isfinite(trace_day), axis=1)
            trace_day = trace_day[valid_neurons]
            trace_day = np.nan_to_num(trace_day, nan=0.0, posinf=0.0, neginf=0.0)
            pos_day = np.nan_to_num(pos_day, nan=0.0, posinf=0.0, neginf=0.0)
            neural_trials, input_trials, output_trials, n_frames, used, n_trials = split_session_into_trials(
                trace_day, pos_day, blocked_vec
            )
            if n_trials < 2:
                print(f'Skipping {session_id}: only {n_trials} full 1-minute trial(s)')
                continue

            data['neural'].append(neural_trials)
            data['input'].append(input_trials)
            data['output'].append(output_trials)
            data['subject_idx'].append(subj_idx)
            data['brain_region_idx'].append(np.zeros(trace_day.shape[0], dtype=np.int64))
            session_info.append({
                'session_id': session_id,
                'animal': animal,
                'day_index': int(day),
                'environment': flatten_env_name(envs[day]),
                'n_neurons': int(trace_day.shape[0]),
                'n_neurons_dropped_nonfinite': int((~valid_neurons).sum()),
                'n_frames_raw': int(n_frames),
                'n_frames_used': int(used),
                'n_trials': int(n_trials),
                'blocked_vector': blocked_vec.astype(int).tolist(),
            })

            if show_processing and plotted < 2:
                pos_bins = position_to_bins_3x3(pos_day[:, :used])
                make_processing_plot(
                    session_id=session_id,
                    pos_day=pos_day[:, :used],
                    pos_bins=pos_bins,
                    blocked_vec=blocked_vec,
                    neural_trial0=neural_trials[0],
                    out_path=f'processing_{session_id}.png'
                )
                plotted += 1

    data['subject_idx'] = np.asarray(data['subject_idx'], dtype=np.int64)
    data['metadata']['session_info'] = session_info
    data['metadata']['n_sessions'] = len(data['neural'])
    data['metadata']['conversion_runtime_sec'] = time.time() - t0

    assert len(data['neural']) == len(data['input']) == len(data['output']) == len(data['subject_idx'])
    for s in range(len(data['neural'])):
        assert len(data['neural'][s]) == len(data['input'][s]) == len(data['output'][s])
        for tr in range(len(data['neural'][s])):
            assert data['neural'][s][tr].shape[1] == data['output'][s][tr].shape[1]

    print(f'Converted {len(data["neural"])} sessions in {data["metadata"]["conversion_runtime_sec"]:.2f}s')
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('outpicklefile')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--full', action='store_true', help='Process all sessions')
    mode.add_argument('--sample', action='store_true', help='Process only 2 animals for testing')
    parser.add_argument('--show-processing', action='store_true', help='Save processing plots for up to 2 sessions')
    args = parser.parse_args()

    sample = bool(args.sample)
    data = convert_dataset(data_dir='data', sample=sample, show_processing=args.show_processing)
    with open(args.outpicklefile, 'wb') as f:
        pickle.dump(data, f)
    print(f'Saved converted dataset to {args.outpicklefile}')


if __name__ == '__main__':
    main()
