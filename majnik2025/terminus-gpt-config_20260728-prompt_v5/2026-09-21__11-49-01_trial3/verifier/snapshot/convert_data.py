#!/usr/bin/env python3
import argparse
import math
import os
import pickle
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


BRAIN_REGION = 'barrel cortex'
INPUT_NAMES = ['session_time_seconds']
OUTPUT_NAMES = ['motion_energy_bin']
OUTPUT_VALUES = [[f'bin_{i}' for i in range(5)]]


def list_sessions(data_root: Path):
    sessions = []
    for subj_dir in sorted([p for p in data_root.iterdir() if p.is_dir()]):
        for sess_dir in sorted([p for p in subj_dir.iterdir() if p.is_dir()]):
            s2p = sess_dir / 'suite2p' / 'plane0'
            mov = sess_dir / 'move_deve'
            if s2p.exists() and mov.exists():
                sessions.append((subj_dir.name, sess_dir.name, sess_dir))
    return sessions


def moving_average_bin(x: np.ndarray, bin_size: int):
    n = x.shape[-1] // bin_size
    if n <= 0:
        raise ValueError('time series too short for chosen bin size')
    trimmed = x[..., : n * bin_size]
    new_shape = x.shape[:-1] + (n, bin_size)
    return trimmed.reshape(new_shape).mean(axis=-1)


def robust_dff(Fcorr: np.ndarray):
    baseline = np.percentile(Fcorr, 20, axis=1, keepdims=True)
    baseline = np.where(np.abs(baseline) < 1e-6, 1e-6, baseline)
    return (Fcorr - baseline) / baseline


def compute_quantile_bins(values: np.ndarray, n_bins: int = 5):
    edges = np.quantile(values, np.linspace(0, 1, n_bins + 1))
    edges = np.asarray(edges, dtype=np.float64)
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-9
    bins = np.digitize(values, edges[1:-1], right=False)
    return bins.astype(np.int64), edges


def load_session(sess_dir: Path):
    s2p = sess_dir / 'suite2p' / 'plane0'
    mov = sess_dir / 'move_deve'
    ops = np.load(s2p / 'ops.npy', allow_pickle=True).item()
    F = np.load(s2p / 'F.npy').astype(np.float32)
    Fneu = np.load(s2p / 'Fneu.npy').astype(np.float32)
    iscell = np.load(s2p / 'iscell.npy')
    motion = np.load(mov / 'motion_energy_glob.npy').astype(np.float32)
    tstamps = np.load(mov / 'tstamps.npy').astype(np.float64)
    return ops, F, Fneu, iscell, motion, tstamps


def process_session(sess_dir: Path, bin_size_frames: int = 10, trial_seconds: float = 60.0, show_processing: bool = False):
    ops, F, Fneu, iscell, motion, tstamps = load_session(sess_dir)
    fs = float(ops.get('fs', 30.0))
    neucoeff = float(ops.get('neucoeff', 0.7))

    cell_mask = iscell[:, 0].astype(bool)
    F = F[cell_mask]
    Fneu = Fneu[cell_mask]

    Fcorr = F - neucoeff * Fneu
    neural = robust_dff(Fcorr)

    n_overlap = min(neural.shape[1], motion.shape[0], tstamps.shape[0])
    neural = neural[:, :n_overlap]
    motion = motion[:n_overlap]
    tstamps = tstamps[:n_overlap]

    neural_b = moving_average_bin(neural, bin_size_frames).astype(np.float32)
    motion_b = moving_average_bin(motion[None, :], bin_size_frames)[0].astype(np.float32)
    # Use imaging frame rate for elapsed session time because tstamps units may not be seconds.
    raw_time = np.arange(n_overlap, dtype=np.float32) / fs
    time_b = moving_average_bin(raw_time[None, :], bin_size_frames)[0].astype(np.float32)

    motion_bins, edges = compute_quantile_bins(motion_b, n_bins=5)

    dt = float(bin_size_frames / fs)
    trial_len = int(round(trial_seconds / dt))
    n_trials = neural_b.shape[1] // trial_len
    if n_trials < 2:
        raise ValueError(f'Not enough 60 s trials in session {sess_dir.name}: {n_trials}')

    usable = n_trials * trial_len
    neural_b = neural_b[:, :usable]
    motion_bins = motion_bins[:usable]
    time_b = time_b[:usable]
    motion_b = motion_b[:usable]

    neural_trials = []
    input_trials = []
    output_trials = []
    for i in range(n_trials):
        sl = slice(i * trial_len, (i + 1) * trial_len)
        neural_trials.append(neural_b[:, sl].astype(np.float32))
        input_trials.append(time_b[sl][None, :].astype(np.float32))
        output_trials.append(motion_bins[sl][None, :].astype(np.int64))

    info = {
        'fs': fs,
        'neucoeff': neucoeff,
        'bin_size_frames': bin_size_frames,
        'dt_seconds': dt,
        'trial_len_bins': trial_len,
        'n_trials': n_trials,
        'n_neurons': int(neural_b.shape[0]),
        'quantile_edges': edges.tolist(),
        'raw_frames_overlap': int(n_overlap),
        'duration_seconds_overlap': float(n_overlap / fs),
    }

    if show_processing:
        fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=False)
        axes[0].plot((F[:3, :1000]).T)
        axes[0].set_title(f'{sess_dir.name}: raw F (first 3 cells, first 1000 frames)')
        axes[1].plot((neural_b[:3, :500]).T)
        axes[1].set_title('binned dF/F-like neural traces')
        axes[2].plot(time_b[:500], motion_b[:500])
        axes[2].set_title('binned motion energy')
        axes[3].plot(time_b[:500], motion_bins[:500])
        axes[3].set_title('discretized motion energy bins')
        for ax in axes:
            ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(f'/app/processing_{sess_dir.parent.name}_{sess_dir.name}.png', dpi=150)
        plt.close(fig)

    return neural_trials, input_trials, output_trials, info


def build_dataset(data_root: Path, mode: str = 'full', show_processing: bool = False):
    sessions = list_sessions(data_root)
    if mode == 'sample':
        sessions = sessions[:2]

    subjects = sorted({s[0] for s in sessions})
    subject_to_idx = {s: i for i, s in enumerate(subjects)}
    brain_regions = [BRAIN_REGION]

    data = {
        'neural': [],
        'input': [],
        'output': [],
        'subjects': subjects,
        'subject_idx': [],
        'brain_regions': brain_regions,
        'brain_region_idx': [],
        'input_names': INPUT_NAMES,
        'output_names': OUTPUT_NAMES,
        'output_values': OUTPUT_VALUES,
        'metadata': {
            'task_description': 'Decode per-session motion energy quintile from barrel cortex calcium activity in 60-second pseudo-trials.',
            'time_bin_size': None,
            'temporal_alignment_event': 'session start',
            'off_start': 0.0,
            'off_end': 60.0,
            'source_dataset': 'Track2p longitudinal barrel cortex imaging sessions',
            'neural_signal': 'neuropil-corrected dF/F-like fluorescence, averaged in non-overlapping 10-frame bins',
            'behavior_signal': 'motion_energy_glob averaged in non-overlapping 10-sample bins and discretized into 5 session-wise quantile bins',
            'trial_definition': 'consecutive non-overlapping 60-second windows within session',
            'session_info': [],
        },
    }

    dt_values = []
    t0 = time.time()
    for idx, (subject, session_name, sess_dir) in enumerate(sessions):
        st = time.time()
        neural_trials, input_trials, output_trials, info = process_session(sess_dir, show_processing=show_processing and idx < 2)
        data['neural'].append(neural_trials)
        data['input'].append(input_trials)
        data['output'].append(output_trials)
        data['subject_idx'].append(subject_to_idx[subject])
        data['brain_region_idx'].append(np.zeros(neural_trials[0].shape[0], dtype=np.int64))
        data['metadata']['session_info'].append({
            'subject': subject,
            'session': session_name,
            **info,
        })
        dt_values.append(info['dt_seconds'])
        print(f'processed {subject}/{session_name}: neurons={info["n_neurons"]} trials={info["n_trials"]} dt={info["dt_seconds"]:.4f}s time={time.time()-st:.2f}s')

    data['subject_idx'] = np.asarray(data['subject_idx'], dtype=np.int64)
    data['metadata']['time_bin_size'] = float(np.median(dt_values) * 1000.0)
    print(f'total sessions={len(data["neural"])} total time={time.time()-t0:.2f}s')
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('outpicklefile')
    g = ap.add_mutually_exclusive_group()
    g.add_argument('--full', action='store_true', help='Process all sessions')
    g.add_argument('--sample', action='store_true', help='Process only 2 sessions for testing')
    ap.add_argument('--show-processing', action='store_true', help='Save processing plots for up to 2 sessions')
    args = ap.parse_args()

    mode = 'sample' if args.sample else 'full'
    data = build_dataset(Path('/app/data'), mode=mode, show_processing=args.show_processing)
    with open(args.outpicklefile, 'wb') as f:
        pickle.dump(data, f)
    print(f'saved {args.outpicklefile}')


if __name__ == '__main__':
    main()
