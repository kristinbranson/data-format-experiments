#!/usr/bin/env python3
import argparse
import math
import os
import pickle
import time
from pathlib import Path

import numpy as np


def discover_sessions(data_root):
    data_root = Path(data_root)
    sessions = []
    for subj_dir in sorted([p for p in data_root.iterdir() if p.is_dir() and p.name.startswith('jm')]):
        for sess_dir in sorted([p for p in subj_dir.iterdir() if p.is_dir()]):
            plane = sess_dir / 'suite2p' / 'plane0'
            move = sess_dir / 'move_deve'
            if plane.exists() and move.exists():
                sessions.append({
                    'subject': subj_dir.name,
                    'session': sess_dir.name,
                    'session_dir': sess_dir,
                    'plane_dir': plane,
                    'move_dir': move,
                })
    return sessions


def load_session_arrays(sess):
    plane = sess['plane_dir']
    move = sess['move_dir']
    F = np.load(plane / 'F.npy').astype(np.float32)
    Fneu = np.load(plane / 'Fneu.npy').astype(np.float32)
    iscell = np.load(plane / 'iscell.npy')
    ops = np.load(plane / 'ops.npy', allow_pickle=True).item()
    motion = np.load(move / 'motion_energy_glob.npy').astype(np.float32)
    tstamps = np.load(move / 'tstamps.npy').astype(np.float64)
    interframe = np.load(move / 'interframe_int.npy').astype(np.float64)
    return F, Fneu, iscell, ops, motion, tstamps, interframe


def get_neural_frame_times_sec(n_frames, ops, tstamps=None):
    fs = float(ops.get('fs', 30.0))
    if tstamps is not None and len(tstamps) >= n_frames:
        dt = np.diff(tstamps[:n_frames])
        if len(dt) > 0:
            dt_sec = float(np.median(dt) * 86400.0)
            if 0.5 / fs < dt_sec < 1.5 / fs:
                t = (tstamps[:n_frames] - tstamps[0]) * 86400.0
                return np.asarray(t, dtype=np.float32)
    return (np.arange(n_frames, dtype=np.float32) / fs).astype(np.float32)


def align_motion_to_neural(motion, tstamps, neural_times_sec, ops):
    fs = float(ops.get('fs', 30.0))
    n_motion = len(motion)
    if len(tstamps) >= n_motion:
        mt = (tstamps[:n_motion] - tstamps[0]) * 86400.0
        good = np.isfinite(mt) & np.isfinite(motion)
        mt = mt[good]
        mv = motion[good]
        if len(mt) >= 2 and np.all(np.diff(mt) > 0):
            return np.interp(neural_times_sec, mt.astype(np.float32), mv.astype(np.float32)).astype(np.float32)
    motion = motion[: min(len(motion), len(neural_times_sec))]
    if len(motion) == len(neural_times_sec):
        return motion.astype(np.float32)
    x_old = np.linspace(0, 1, num=len(motion), dtype=np.float32)
    x_new = np.linspace(0, 1, num=len(neural_times_sec), dtype=np.float32)
    return np.interp(x_new, x_old, motion.astype(np.float32)).astype(np.float32)


def compute_dff(F, Fneu, neuropil_coeff=0.7, baseline_percentile=8.0):
    Fc = F - neuropil_coeff * Fneu
    baseline = np.percentile(Fc, baseline_percentile, axis=1, keepdims=True).astype(np.float32)
    baseline = np.where(np.abs(baseline) < 1e-3, 1e-3, baseline)
    dff = (Fc - baseline) / np.abs(baseline)
    return dff.astype(np.float32)


def bin_time_series(arr, bin_size, axis=-1, reducer='mean'):
    n = arr.shape[axis]
    n_bins = n // bin_size
    if n_bins <= 0:
        raise ValueError('Not enough samples to bin')
    slicer = [slice(None)] * arr.ndim
    slicer[axis] = slice(0, n_bins * bin_size)
    arr = arr[tuple(slicer)]
    new_shape = list(arr.shape)
    new_shape[axis] = n_bins
    new_shape.insert(axis + 1, bin_size)
    arr = arr.reshape(new_shape)
    if reducer == 'mean':
        return arr.mean(axis=axis + 1)
    raise ValueError(f'Unsupported reducer: {reducer}')


def discretize_into_quantile_bins(values, n_bins=5):
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(values, qs)
    edges[0] = -np.inf
    edges[-1] = np.inf
    for i in range(1, len(edges) - 1):
        if edges[i] <= edges[i - 1]:
            edges[i] = np.nextafter(edges[i - 1], np.inf)
    labels = np.digitize(values, edges[1:-1], right=False).astype(np.int64)
    return labels, edges


def split_into_trials(neural, inp, out, trial_len_bins):
    n_time = neural.shape[1]
    n_trials = n_time // trial_len_bins
    if n_trials < 2:
        return [], [], []
    keep = n_trials * trial_len_bins
    neural = neural[:, :keep]
    inp = inp[:, :keep]
    out = out[:, :keep]
    neural_trials = [neural[:, i * trial_len_bins:(i + 1) * trial_len_bins].astype(np.float32) for i in range(n_trials)]
    input_trials = [inp[:, i * trial_len_bins:(i + 1) * trial_len_bins].astype(np.float32) for i in range(n_trials)]
    output_trials = [out[:, i * trial_len_bins:(i + 1) * trial_len_bins].astype(np.int64) for i in range(n_trials)]
    return neural_trials, input_trials, output_trials


def process_session(sess, show_processing=False):
    t0 = time.time()
    F, Fneu, iscell, ops, motion, tstamps, interframe = load_session_arrays(sess)
    n_frames = F.shape[1]
    neural_times_sec = get_neural_frame_times_sec(n_frames, ops, tstamps)
    motion_aligned = align_motion_to_neural(motion, tstamps, neural_times_sec, ops)
    neural = compute_dff(F, Fneu, neuropil_coeff=float(ops.get('neucoeff', 0.7)), baseline_percentile=float(ops.get('prctile_baseline', 8.0)))

    bin_size = 10
    neural_b = bin_time_series(neural, bin_size, axis=1, reducer='mean')
    motion_b = bin_time_series(motion_aligned[None, :], bin_size, axis=1, reducer='mean')[0]
    time_b = bin_time_series(neural_times_sec[None, :], bin_size, axis=1, reducer='mean')[0]

    motion_disc, edges = discretize_into_quantile_bins(motion_b, n_bins=5)
    inp = time_b[None, :].astype(np.float32)
    out = motion_disc[None, :].astype(np.int64)

    trial_len_bins = int(round(60.0 / (bin_size / float(ops.get('fs', 30.0)))))
    neural_trials, input_trials, output_trials = split_into_trials(neural_b, inp, out, trial_len_bins)

    info = {
        'subject': sess['subject'],
        'session': sess['session'],
        'n_neurons': int(neural_b.shape[0]),
        'n_frames_raw': int(n_frames),
        'n_bins': int(neural_b.shape[1]),
        'n_trials': int(len(neural_trials)),
        'motion_edges': edges.tolist(),
        'elapsed_sec_end': float(time_b[-1]),
        'process_time_sec': time.time() - t0,
    }

    if show_processing:
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(4, 1, figsize=(12, 10), constrained_layout=True)
        axs[0].plot(neural_times_sec[:2000], motion_aligned[:2000], lw=1)
        axs[0].set_title(f"{sess['subject']} {sess['session']} aligned motion (first 2000 frames)")
        axs[1].imshow(neural_b[: min(30, neural_b.shape[0]), : min(500, neural_b.shape[1])], aspect='auto', cmap='viridis')
        axs[1].set_title('Binned neural activity (subset)')
        axs[2].plot(time_b[:500], motion_b[:500], lw=1)
        axs[2].set_title('Binned motion energy (first 500 bins)')
        axs[3].hist(motion_disc, bins=np.arange(6)-0.5, rwidth=0.8)
        axs[3].set_title('Motion quantile bins')
        outpng = f"/app/processing_{sess['subject']}_{sess['session']}.png"
        fig.savefig(outpng, dpi=150)
        plt.close(fig)
        info['plot_file'] = outpng

    return neural_trials, input_trials, output_trials, info


def build_dataset(sessions, show_processing=False):
    subjects = sorted({s['subject'] for s in sessions})
    subject_to_idx = {s: i for i, s in enumerate(subjects)}
    brain_regions = ['barrel cortex']

    data = {
        'neural': [],
        'input': [],
        'output': [],
        'subjects': subjects,
        'subject_idx': [],
        'brain_regions': brain_regions,
        'brain_region_idx': [],
        'input_names': ['time_elapsed_sec'],
        'output_names': ['motion_energy_quantile'],
        'output_values': [[f'bin_{i}' for i in range(5)]],
        'metadata': {
            'task_description': 'Decode spontaneous animal motion energy from barrel cortex calcium activity in continuous sessions split into 60-second trials.',
            'time_bin_size': 1000.0 * (10.0 / 30.0),
            'temporal_alignment_event': 'session start',
            'off_start': 0.0,
            'off_end': 60.0,
            'neural_signal': 'approximate_dff_from_F_and_Fneu',
            'binning': 'non-overlapping 10-frame means',
            'motion_discretization': '5 equal-percentile bins per session after alignment and 10-frame binning',
            'session_info': [],
        },
    }

    for i, sess in enumerate(sessions):
        neural_trials, input_trials, output_trials, info = process_session(sess, show_processing=sess.get('show_processing', False))
        if len(neural_trials) < 2:
            continue
        data['neural'].append(neural_trials)
        data['input'].append(input_trials)
        data['output'].append(output_trials)
        data['subject_idx'].append(subject_to_idx[sess['subject']])
        data['brain_region_idx'].append(np.zeros(neural_trials[0].shape[0], dtype=np.int64))
        data['metadata']['session_info'].append(info)
        print(f"processed {i+1}/{len(sessions)} {sess['subject']} {sess['session']} -> {info['n_trials']} trials, {info['n_neurons']} neurons in {info['process_time_sec']:.2f}s")

    data['subject_idx'] = np.asarray(data['subject_idx'], dtype=np.int64)
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('outpicklefile')
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument('--full', action='store_true', help='Process all sessions (default)')
    mode.add_argument('--sample', action='store_true', help='Process only 2 sessions for testing')
    ap.add_argument('--show-processing', action='store_true', help='Save processing visualizations for up to 2 sessions')
    args = ap.parse_args()

    sessions = discover_sessions('/app/data')
    if args.sample:
        sessions = sessions[:2]
    if args.show_processing and len(sessions) > 2:
        show_ids = {0, 1}
    else:
        show_ids = set(range(len(sessions))) if args.show_processing else set()

    sessions_for_build = []
    for idx, s in enumerate(sessions):
        s = dict(s)
        s['show_processing'] = idx in show_ids
        sessions_for_build.append(s)

    data = build_dataset(sessions_for_build, show_processing=args.show_processing)

    with open(args.outpicklefile, 'wb') as f:
        pickle.dump(data, f)
    print(f"saved dataset to {args.outpicklefile}")
    print(f"n_sessions={len(data['neural'])}, n_subjects={len(data['subjects'])}")


if __name__ == '__main__':
    main()
