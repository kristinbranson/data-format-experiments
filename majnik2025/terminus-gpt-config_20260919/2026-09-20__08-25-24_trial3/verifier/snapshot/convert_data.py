#!/usr/bin/env python3
"""Convert Track2p longitudinal barrel-cortex data for neural decoding."""
import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, minimum_filter1d, maximum_filter1d

DATA_ROOT = Path('/app/data')
RAW_FRAMES_PER_TRIAL = 1800  # 60 s * 30 Hz
TEMPORAL_AVG = 10
PROCESSED_POINTS_PER_TRIAL = RAW_FRAMES_PER_TRIAL // TEMPORAL_AVG


def discover_sessions(root=DATA_ROOT):
    """Return sorted (subject, session_id, session_path) tuples."""
    out = []
    for subject_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for session_dir in sorted(p for p in subject_dir.iterdir() if p.is_dir()):
            out.append((subject_dir.name, session_dir.name, session_dir))
    return out


def baseline_correct(F, Fneu, ops):
    """Reproduce default Suite2p/Track2p maximin baseline subtraction."""
    neucoeff = float(ops.get('neucoeff', 0.7))
    fs = float(ops['fs'])
    sig = float(ops.get('sig_baseline', 10.0))
    win = int(float(ops.get('win_baseline', 60.0)) * fs)
    baseline_mode = ops.get('baseline', 'maximin')
    Fc = np.asarray(F, dtype=np.float32) - np.float32(neucoeff) * np.asarray(Fneu, dtype=np.float32)
    if baseline_mode == 'maximin':
        Flow = gaussian_filter(Fc, [0.0, sig])
        Flow = minimum_filter1d(Flow, win, axis=1)
        Flow = maximum_filter1d(Flow, win, axis=1)
    elif baseline_mode == 'constant':
        Flow = np.amin(gaussian_filter(Fc, [0.0, sig]))
    elif baseline_mode in ('constant_prctile', 'prctile'):
        Flow = np.percentile(Fc, float(ops.get('prctile_baseline', 8.0)), axis=1)[:, None]
    else:
        Flow = 0.0
    return (Fc - Flow).astype(np.float32, copy=False), Fc, Flow


def mean_bins_2d(x, n_frames):
    """Average a neurons/features x time array in nonoverlapping 10-frame bins."""
    x = np.asarray(x[:, :n_frames])
    return x.reshape(x.shape[0], n_frames // TEMPORAL_AVG, TEMPORAL_AVG).mean(axis=2, dtype=np.float32)


def mean_bins_1d(x, n_frames):
    """Average a time vector in nonoverlapping 10-frame bins."""
    x = np.asarray(x[:n_frames], dtype=np.float64)
    return x.reshape(n_frames // TEMPORAL_AVG, TEMPORAL_AVG).mean(axis=1)


def make_processing_plot(outpath, session_label, raw_F, raw_Fneu, Fc, Flow, neural_binned,
                         motion_raw, motion_binned, labels, quantile_edges, fs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    nshow = min(4, raw_F.shape[0])
    raw_n = min(raw_F.shape[1], int(fs * 180))
    bin_n = min(neural_binned.shape[1], int(fs / TEMPORAL_AVG * 180))
    t_raw = np.arange(raw_n) / fs
    t_bin = (np.arange(bin_n) * TEMPORAL_AVG + (TEMPORAL_AVG - 1) / 2) / fs
    fig, axes = plt.subplots(4, 2, figsize=(16, 13), constrained_layout=True)
    axes[0, 0].plot(t_raw, raw_F[:nshow, :raw_n].T, lw=.5)
    axes[0, 0].set(title='Raw ROI fluorescence F', ylabel='F')
    axes[0, 1].plot(t_raw, raw_Fneu[:nshow, :raw_n].T, lw=.5)
    axes[0, 1].set(title='Raw neuropil fluorescence Fneu', ylabel='Fneu')
    axes[1, 0].plot(t_raw, Fc[:nshow, :raw_n].T, lw=.5)
    axes[1, 0].set(title='Neuropil corrected F - neucoeff*Fneu', ylabel='Fc')
    flow_arr = np.asarray(Flow)
    if flow_arr.ndim == 0:
        flow_arr = np.full_like(Fc[:nshow, :raw_n], flow_arr)
    axes[1, 1].plot(t_raw, flow_arr[:nshow, :raw_n].T, lw=.5)
    axes[1, 1].set(title='Maximin baseline', ylabel='baseline')
    axes[2, 0].plot(t_bin, neural_binned[:nshow, :bin_n].T, lw=.7)
    axes[2, 0].set(title='Baseline-subtracted neural, 10-frame means', ylabel='activity')
    axes[2, 1].plot(np.arange(raw_n) / fs, motion_raw[:raw_n], lw=.5)
    axes[2, 1].set(title='Raw motion energy', ylabel='motion')
    axes[3, 0].plot(t_bin, motion_binned[:bin_n], lw=.7)
    for edge in quantile_edges:
        axes[3, 0].axhline(edge, color='k', alpha=.25, lw=.7)
    axes[3, 0].set(title='10-frame mean motion and quintile thresholds', xlabel='session time (s)', ylabel='motion')
    axes[3, 1].hist(labels, bins=np.arange(6)-.5, rwidth=.8)
    axes[3, 1].set(title='Saved output class distribution', xlabel='class', ylabel='samples', xticks=range(5))
    for ax in axes.flat:
        ax.grid(alpha=.15)
    fig.suptitle(f'Processing audit: {session_label}')
    fig.savefig(outpath, dpi=140)
    plt.close(fig)


def process_session(subject, session_id, path, show_processing=False):
    t0 = time.perf_counter()
    plane = path / 'suite2p' / 'plane0'
    move = path / 'move_deve'
    F = np.load(plane / 'F.npy', mmap_mode='r')
    Fneu = np.load(plane / 'Fneu.npy', mmap_mode='r')
    iscell = np.load(plane / 'iscell.npy', mmap_mode='r')
    ops = np.load(plane / 'ops.npy', allow_pickle=True).item()
    motion = np.load(move / 'motion_energy_glob.npy', mmap_mode='r')

    if F.shape != Fneu.shape:
        raise ValueError(f'{subject}/{session_id}: F and Fneu shapes differ')
    if F.shape[0] != iscell.shape[0] or not np.all(iscell[:, 0] == 1):
        raise ValueError(f'{subject}/{session_id}: supplied matched-cell curation is inconsistent')
    fs = float(ops['fs'])
    if not np.isclose(fs, 30.0):
        raise ValueError(f'{subject}/{session_id}: expected 30 Hz, found {fs}')

    shared_frames = min(F.shape[1], motion.shape[0])
    n_trials = shared_frames // RAW_FRAMES_PER_TRIAL
    if n_trials < 2:
        raise ValueError(f'{subject}/{session_id}: fewer than two complete 60-s trials')
    used_frames = n_trials * RAW_FRAMES_PER_TRIAL

    # Baseline estimation uses the complete neural recording, matching Suite2p processing.
    neural_bc, Fc, Flow = baseline_correct(F, Fneu, ops)
    neural_binned = mean_bins_2d(neural_bc, used_frames)
    motion_binned = mean_bins_1d(motion, used_frames)

    edges = np.quantile(motion_binned, [0.2, 0.4, 0.6, 0.8])
    labels = np.searchsorted(edges, motion_binned, side='right').astype(np.int64)
    elapsed = ((np.arange(len(motion_binned), dtype=np.float64) * TEMPORAL_AVG
                + (TEMPORAL_AVG - 1) / 2) / fs).astype(np.float32)

    neural_trials, input_trials, output_trials = [], [], []
    for trial in range(n_trials):
        a = trial * PROCESSED_POINTS_PER_TRIAL
        b = a + PROCESSED_POINTS_PER_TRIAL
        neural_trials.append(np.ascontiguousarray(neural_binned[:, a:b], dtype=np.float32))
        input_trials.append(np.ascontiguousarray(elapsed[None, a:b], dtype=np.float32))
        output_trials.append(np.ascontiguousarray(labels[None, a:b], dtype=np.int64))

    counts = np.bincount(labels, minlength=5)
    info = {
        'subject': subject, 'session_id': session_id, 'n_neurons': int(F.shape[0]),
        'neural_source_frames': int(F.shape[1]), 'motion_source_frames': int(motion.shape[0]),
        'shared_frames': int(shared_frames), 'used_frames': int(used_frames),
        'unused_shared_terminal_frames': int(shared_frames-used_frames),
        'motion_frames_missing_vs_neural': int(F.shape[1]-motion.shape[0]),
        'n_trials': int(n_trials), 'fs_hz': fs,
        'motion_quintile_edges': edges.tolist(), 'output_class_counts': counts.tolist(),
        'neucoeff': float(ops.get('neucoeff', .7)), 'baseline': str(ops.get('baseline', 'maximin')),
        'sig_baseline_frames': float(ops.get('sig_baseline', 10.0)),
        'win_baseline_seconds': float(ops.get('win_baseline', 60.0)),
    }
    if show_processing:
        safe = f'{subject}_{session_id}'.replace('/', '_')
        make_processing_plot(Path('/app') / f'processing_{safe}.png', f'{subject}/{session_id}',
                             F, Fneu, Fc, Flow, neural_binned, motion, motion_binned,
                             labels, edges, fs)
    dt = time.perf_counter() - t0
    print(f'Processed {subject}/{session_id}: {F.shape[0]} neurons, {n_trials} trials, '
          f'class counts {counts.tolist()}, {dt:.2f}s', flush=True)
    return neural_trials, input_trials, output_trials, info, dt


def validate_converted(data):
    ns = len(data['neural'])
    assert ns == len(data['input']) == len(data['output']) == len(data['subject_idx'])
    assert ns == len(data['brain_region_idx'])
    for s in range(ns):
        assert len(data['neural'][s]) == len(data['input'][s]) == len(data['output'][s]) >= 2
        assert len(data['brain_region_idx'][s]) == data['neural'][s][0].shape[0]
        for n, x, y in zip(data['neural'][s], data['input'][s], data['output'][s]):
            assert n.shape[1] == x.shape[1] == y.shape[1] == PROCESSED_POINTS_PER_TRIAL
            assert n.dtype == np.float32 and x.dtype == np.float32 and y.dtype == np.int64
            assert np.isfinite(n).all() and np.isfinite(x).all() and np.isfinite(y).all()
            assert y.min() >= 0 and y.max() <= 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('outpicklefile')
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument('--full', action='store_true', help='process all sessions (default)')
    mode.add_argument('--sample', action='store_true', help='process first two sessions')
    ap.add_argument('--show-processing', action='store_true')
    args = ap.parse_args()

    sessions = discover_sessions()
    if args.sample:
        sessions = sessions[:2]
    subjects = sorted({x[0] for x in sessions})
    subject_lookup = {x: i for i, x in enumerate(subjects)}
    neural, inputs, outputs, infos, times = [], [], [], [], []
    print(f'Converting {len(sessions)} sessions to {args.outpicklefile}', flush=True)
    wall0 = time.perf_counter()
    for i, (subject, sid, path) in enumerate(sessions):
        show = args.show_processing and i < 2
        n, x, y, info, dt = process_session(subject, sid, path, show)
        neural.append(n); inputs.append(x); outputs.append(y); infos.append(info); times.append(dt)

    data = {
        'neural': neural,
        'input': inputs,
        'output': outputs,
        'subjects': subjects,
        'subject_idx': np.asarray([subject_lookup[x[0]] for x in sessions], dtype=np.int64),
        'brain_regions': ['barrel cortex L2/3'],
        'brain_region_idx': [np.zeros(n[0].shape[0], dtype=np.int64) for n in neural],
        'input_names': ['time elapsed from session start (s)'],
        'output_names': ['motion energy quintile'],
        'output_values': [['quintile 1 (lowest)', 'quintile 2', 'quintile 3',
                           'quintile 4', 'quintile 5 (highest)']],
        'metadata': {
            'task_description': 'Predict per-session motion-energy quintile from barrel-cortex neural activity.',
            'time_bin_size': 1000.0 * TEMPORAL_AVG / 30.0,
            'temporal_alignment_event': 'start of each non-overlapping 60-second trial; source video was hardware-triggered by microscope',
            'off_start': 0.0,
            'off_end': 60.0,
            'trial_duration_seconds': 60.0,
            'native_sampling_rate_hz': 30.0,
            'temporal_averaging_frames': TEMPORAL_AVG,
            'neural_signal': 'neuropil-corrected Suite2p fluorescence with maximin baseline subtracted',
            'motion_processing': 'global squared frame-difference energy; 10-frame means; per-session quintiles',
            'trial_policy': 'non-overlapping complete 60-s trials from shared neural/behavior prefix',
            'session_info': infos,
        }
    }
    validate_converted(data)
    with open(args.outpicklefile, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    wall = time.perf_counter() - wall0
    print(f'Saved {args.outpicklefile}: {len(sessions)} sessions, '
          f'{sum(len(x) for x in neural)} trials in {wall:.2f}s; '
          f'mean processing {np.mean(times):.2f}s/session', flush=True)

if __name__ == '__main__':
    main()
