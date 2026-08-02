#!/usr/bin/env python3
import argparse
import pickle
import time
from pathlib import Path

import joblib
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except Exception:
    plt = None

ANIMAL_IDS = [
    'QLAK-CA1-08', 'QLAK-CA1-30', 'QLAK-CA1-50',
    'QLAK-CA1-51', 'QLAK-CA1-56', 'QLAK-CA1-74', 'QLAK-CA1-75'
]
FRAME_RATE_HZ = 30.0
TRIAL_SECONDS = 60.0
FRAMES_PER_TRIAL = int(FRAME_RATE_HZ * TRIAL_SECONDS)
BRAIN_REGION = 'CA1'


def env_to_mask(env_name):
    env = str(env_name)
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
        return np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=np.float32)
    raise ValueError(f'Unknown environment name: {env!r}')


def normalize_blocked_entry(entry):
    cur = entry
    while isinstance(cur, list) and len(cur) == 1:
        cur = cur[0]
    if isinstance(cur, np.ndarray):
        cur = cur.tolist()
    if isinstance(cur, tuple):
        cur = list(cur)
    return cur


def blocked_to_mask(entry):
    cur = normalize_blocked_entry(entry)
    if isinstance(cur, list):
        vals = [int(x) for x in np.array(cur, dtype=object).reshape(-1).tolist()]
    else:
        vals = [int(cur)]
    mask = np.ones((3, 3), dtype=np.float32)
    if len(vals) == 1 and vals[0] == -1:
        return mask
    for v in vals:
        if v == -1:
            continue
        r, c = divmod(int(v), 3)
        mask[r, c] = 0.0
    return mask


def combined_env_mask(env_name, blocked_entry):
    return env_to_mask(env_name) * blocked_to_mask(blocked_entry)


def discretize_position_to_3x3(pos_xy):
    x = pos_xy[0]
    y = pos_xy[1]
    valid = np.isfinite(x) & np.isfinite(y)
    x_out = np.zeros_like(x, dtype=np.int64)
    y_out = np.zeros_like(y, dtype=np.int64)
    if np.any(valid):
        xv = x[valid]
        yv = y[valid]
        xmin, xmax = np.min(xv), np.max(xv)
        ymin, ymax = np.min(yv), np.max(yv)
        if xmax == xmin:
            xbins = np.zeros(np.sum(valid), dtype=np.int64)
        else:
            xbins = np.clip(np.floor((xv - xmin) / (xmax - xmin + 1e-12) * 3), 0, 2).astype(np.int64)
        if ymax == ymin:
            ybins = np.zeros(np.sum(valid), dtype=np.int64)
        else:
            ybins = np.clip(np.floor((yv - ymin) / (ymax - ymin + 1e-12) * 3), 0, 2).astype(np.int64)
        x_out[valid] = xbins
        y_out[valid] = ybins
    labels = y_out * 3 + x_out
    labels[~valid] = -1
    return labels[np.newaxis, :]


def load_animal(animal_id, data_dir):
    return joblib.load(Path(data_dir) / animal_id)[animal_id]


def process_day(day_trace, day_pos, env_name, blocked_entry, show_processing=False, session_tag=''):
    valid_neurons = ~np.all(np.isnan(day_trace), axis=1)
    trace = np.nan_to_num(day_trace[valid_neurons], nan=0.0).astype(np.float32)
    pos = np.asarray(day_pos, dtype=np.float32)
    n_frames = min(trace.shape[1], pos.shape[1])
    trace = trace[:, :n_frames]
    pos = pos[:, :n_frames]
    n_full_trials = n_frames // FRAMES_PER_TRIAL
    mask = combined_env_mask(env_name, blocked_entry).reshape(-1).astype(np.float32)
    neural_trials = []
    input_trials = []
    output_trials = []
    for ti in range(n_full_trials):
        s = ti * FRAMES_PER_TRIAL
        e = s + FRAMES_PER_TRIAL
        neural_trials.append(trace[:, s:e])
        input_trials.append(mask.copy())
        output_trials.append(discretize_position_to_3x3(pos[:, s:e]).astype(np.int64))
    if show_processing and plt is not None:
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        axs[0, 0].imshow(mask.reshape(3, 3), cmap='gray_r', vmin=0, vmax=1)
        axs[0, 0].set_title(f'Env mask {env_name}')
        axs[0, 1].plot(pos[0, :min(n_frames, 3000)], pos[1, :min(n_frames, 3000)], lw=0.5)
        axs[0, 1].set_title('Position trajectory sample')
        axs[1, 0].imshow(trace[:min(50, trace.shape[0]), :min(n_frames, 1000)], aspect='auto', cmap='viridis')
        axs[1, 0].set_title('Neural trace sample')
        if output_trials:
            axs[1, 1].plot(output_trials[0][0, :300])
            axs[1, 1].set_title('First trial 3x3 position labels')
        else:
            axs[1, 1].text(0.5, 0.5, 'No full trials', ha='center', va='center')
        for ax in axs.ravel():
            ax.set_xticks([])
            ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(f'processing_{session_tag}.png', dpi=150)
        plt.close(fig)
    return neural_trials, input_trials, output_trials, valid_neurons


def convert_dataset(data_dir='data', mode='full', show_processing=False):
    animal_ids = ANIMAL_IDS if mode == 'full' else ANIMAL_IDS[:2]
    subjects = list(animal_ids)
    subject_to_idx = {s: i for i, s in enumerate(subjects)}
    data = {
        'neural': [],
        'input': [],
        'output': [],
        'subjects': subjects,
        'subject_idx': [],
        'brain_regions': [BRAIN_REGION],
        'brain_region_idx': [],
        'input_names': [f'geom_bin_{i}' for i in range(9)],
        'output_names': ['position_bin'],
        'output_values': [[f'bin_{i}' for i in range(9)]],
        'metadata': {
            'task_description': 'Decode mouse position in a 3x3 spatial grid from CA1 calcium-event activity using static per-trial environment geometry/block mask.',
            'time_bin_size': 1000.0 / FRAME_RATE_HZ,
            'temporal_alignment_event': 'Session/day start; sessions split into contiguous 1-minute trials',
            'off_start': 0.0,
            'off_end': TRIAL_SECONDS,
            'frame_rate_hz': FRAME_RATE_HZ,
            'trial_seconds': TRIAL_SECONDS,
            'source_signal': 'Binary rising-phase calcium trace events',
            'notes': 'Trailing partial-minute frames are dropped. Position discretized independently within each session/day into 3x3 bins using observed x/y range.'
        }
    }
    total_sessions = 0
    total_trials = 0
    t0 = time.time()
    for animal_id in animal_ids:
        animal = load_animal(animal_id, data_dir)
        envs = np.array(animal['envs']).squeeze()
        traces = np.asarray(animal['trace'])
        positions = np.asarray(animal['position'])
        blocked = animal['blocked']
        n_days = traces.shape[0]
        for day_idx in range(n_days):
            neural_trials, input_trials, output_trials, valid_neurons = process_day(
                traces[day_idx], positions[day_idx], envs[day_idx], blocked[day_idx],
                show_processing=show_processing and total_sessions < 2,
                session_tag=f'{animal_id}_day{day_idx+1:02d}'
            )
            if len(neural_trials) < 2:
                continue
            data['neural'].append(neural_trials)
            data['input'].append(input_trials)
            data['output'].append(output_trials)
            data['subject_idx'].append(subject_to_idx[animal_id])
            data['brain_region_idx'].append(np.zeros(int(np.sum(valid_neurons)), dtype=np.int64))
            total_sessions += 1
            total_trials += len(neural_trials)
        print(f'processed {animal_id}: {n_days} days')
    data['subject_idx'] = np.asarray(data['subject_idx'], dtype=np.int64)
    data['metadata']['n_sessions'] = total_sessions
    data['metadata']['n_trials'] = total_trials
    data['metadata']['conversion_runtime_sec'] = time.time() - t0
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('outpicklefile')
    parser.add_argument('--full', action='store_true', help='Process all sessions')
    parser.add_argument('--sample', action='store_true', help='Process only 2 animals for testing')
    parser.add_argument('--show-processing', action='store_true', help='Save processing plots for up to 2 sessions')
    args = parser.parse_args()
    mode = 'sample' if args.sample and not args.full else 'full'
    data = convert_dataset(data_dir='data', mode=mode, show_processing=args.show_processing)
    with open(args.outpicklefile, 'wb') as f:
        pickle.dump(data, f)
    print(f'saved {args.outpicklefile}')
    print(f"sessions={len(data['neural'])} trials={sum(len(x) for x in data['neural'])} subjects={len(data['subjects'])}")


if __name__ == '__main__':
    main()
