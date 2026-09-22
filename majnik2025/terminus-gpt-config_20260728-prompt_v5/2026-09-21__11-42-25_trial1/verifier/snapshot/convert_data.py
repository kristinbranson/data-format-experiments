#!/usr/bin/env python3
import argparse
import math
import os
import pickle
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description='Convert Track2p/Suite2p dataset to decoder format.')
    p.add_argument('outpicklefile', type=str)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--full', action='store_true', help='Process all sessions (default).')
    mode.add_argument('--sample', action='store_true', help='Process only 2 sessions for testing.')
    p.add_argument('--show-processing', action='store_true', help='Plot visualizations for up to 2 sessions.')
    return p.parse_args()


@dataclass
class SessionResult:
    session_id: str
    subject: str
    neural_trials: list
    input_trials: list
    output_trials: list
    brain_region_idx: np.ndarray
    metadata: dict


def discover_sessions(data_root: Path):
    sessions = []
    for subj_dir in sorted([p for p in data_root.iterdir() if p.is_dir() and p.name.startswith('jm')]):
        for sess_dir in sorted([p for p in subj_dir.iterdir() if p.is_dir()]):
            if (sess_dir / 'suite2p/plane0/F.npy').exists() and (sess_dir / 'move_deve/motion_energy_glob.npy').exists():
                sessions.append(sess_dir)
    return sessions


def moving_average_nonoverlap(arr, bin_size):
    n = arr.shape[-1] // bin_size
    trimmed = arr[..., : n * bin_size]
    new_shape = arr.shape[:-1] + (n, bin_size)
    return trimmed.reshape(new_shape).mean(axis=-1)


def robust_df_over_f(fcorr, fs, win_baseline_sec=60.0, prctile_baseline=8.0, eps=1e-3):
    # Approximate Suite2p-style low-percentile running baseline, while ensuring positivity.
    # Shift each neuron upward if neuropil correction makes the trace non-positive.
    min_per_neuron = fcorr.min(axis=1, keepdims=True)
    shift = np.maximum(0.0, 1.0 - min_per_neuron)
    fc = fcorr + shift

    win = max(1, int(round(fs * win_baseline_sec)))
    n = fc.shape[1]
    baseline = np.empty_like(fc, dtype=np.float32)
    for start in range(0, n, win):
        end = min(n, start + win)
        chunk = fc[:, start:end]
        b = np.percentile(chunk, prctile_baseline, axis=1, keepdims=True)
        baseline[:, start:end] = b.astype(np.float32)

    # Avoid unstable division by very small baselines.
    med = np.median(fc, axis=1, keepdims=True)
    floor = np.maximum(eps, 0.05 * med)
    baseline = np.maximum(baseline, floor).astype(np.float32)
    dff = (fc - baseline) / baseline
    return dff.astype(np.float32), baseline.astype(np.float32), shift.astype(np.float32)


def digitize_equal_percentile(x, n_bins=5):
    edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    # enforce monotonicity to avoid numerical issues when values repeat heavily
    edges = np.maximum.accumulate(edges)
    labels = np.digitize(x, edges[1:-1], right=False).astype(np.int64)
    return labels, edges.astype(np.float32)


def compute_time_binned_from_tstamps(tstamps, bin_size, fs_fallback):
    n = len(tstamps) // bin_size
    trimmed = np.asarray(tstamps[: n * bin_size], dtype=np.float64).reshape(n, bin_size)
    t = trimmed.mean(axis=1)
    t = t - t[0]
    sec_per_bin = float(np.median(np.diff(t))) if len(t) > 1 else np.nan

    # Heuristic unit handling: if raw timestamps look like ms, convert to seconds.
    if np.isfinite(sec_per_bin) and sec_per_bin > 1.0:
        t = t / 1000.0
        sec_per_bin = sec_per_bin / 1000.0

    # Fallback if timestamps are missing or have implausible scale.
    expected = bin_size / fs_fallback
    if (not np.isfinite(sec_per_bin)) or sec_per_bin <= 0 or sec_per_bin > 10 * expected or sec_per_bin < 0.1 * expected:
        t = np.arange(n, dtype=np.float64) * expected
        sec_per_bin = expected

    return t[None, :].astype(np.float32), float(sec_per_bin)


def split_trials(neural, inp, out, fs_binned, trial_sec=60.0):
    bins_per_trial = int(round(trial_sec * fs_binned))
    n_trials = neural.shape[1] // bins_per_trial
    usable = n_trials * bins_per_trial
    neural = neural[:, :usable]
    inp = inp[:, :usable]
    out = out[:, :usable]
    neural_trials = [np.asarray(neural[:, i*bins_per_trial:(i+1)*bins_per_trial], dtype=np.float32) for i in range(n_trials)]
    input_trials = [np.asarray(inp[:, i*bins_per_trial:(i+1)*bins_per_trial], dtype=np.float32) for i in range(n_trials)]
    output_trials = [np.asarray(out[:, i*bins_per_trial:(i+1)*bins_per_trial], dtype=np.int64) for i in range(n_trials)]
    return neural_trials, input_trials, output_trials, bins_per_trial


def load_session(sess_dir: Path, bin_size=10, neuropil_coeff=0.7, iscell_thr=0.5, trial_sec=60.0):
    plane = sess_dir / 'suite2p/plane0'
    move = sess_dir / 'move_deve'

    iscell = np.load(plane / 'iscell.npy', allow_pickle=True)
    keep = iscell[:, 0] > iscell_thr

    F = np.load(plane / 'F.npy').astype(np.float32)
    Fneu = np.load(plane / 'Fneu.npy').astype(np.float32)
    ops = np.load(plane / 'ops.npy', allow_pickle=True).item()
    motion = np.load(move / 'motion_energy_glob.npy').astype(np.float32)
    tstamps = np.load(move / 'tstamps.npy').astype(np.float32)

    fs = float(ops['fs'])
    n_common = min(F.shape[1], motion.shape[0], tstamps.shape[0])

    F = F[keep, :n_common]
    Fneu = Fneu[keep, :n_common]
    motion = motion[:n_common]
    tstamps = tstamps[:n_common]

    fcorr = F - neuropil_coeff * Fneu
    dff, baseline, shift = robust_df_over_f(
        fcorr,
        fs=fs,
        win_baseline_sec=float(ops.get('win_baseline', 60.0)),
        prctile_baseline=float(ops.get('prctile_baseline', 8.0)),
    )

    neural_b = moving_average_nonoverlap(dff, bin_size)
    motion_b = moving_average_nonoverlap(motion[None, :], bin_size)[0]

    n_bins = neural_b.shape[1]
    motion_b = motion_b[:n_bins]
    time_b, sec_per_bin = compute_time_binned_from_tstamps(tstamps, bin_size, fs_fallback=fs)
    time_b = time_b[:, :n_bins]
    fs_binned = 1.0 / sec_per_bin

    motion_labels, motion_edges = digitize_equal_percentile(motion_b, n_bins=5)
    output_b = motion_labels[None, :]

    neural_trials, input_trials, output_trials, bins_per_trial = split_trials(
        neural_b, time_b, output_b, fs_binned=fs_binned, trial_sec=trial_sec
    )

    meta = {
        'session_id': f'{sess_dir.parent.name}/{sess_dir.name}',
        'subject': sess_dir.parent.name,
        'fs_raw_hz': fs,
        'time_bin_size_frames': bin_size,
        'time_bin_size_sec': sec_per_bin,
        'n_raw_frames_common': int(n_common),
        'n_neurons': int(neural_b.shape[0]),
        'n_binned_timepoints': int(n_bins),
        'bins_per_trial': int(bins_per_trial),
        'n_trials': int(len(neural_trials)),
        'motion_bin_edges': motion_edges,
        'baseline_median': float(np.median(baseline)),
        'shift_median': float(np.median(shift)),
        'source_session_path': str(sess_dir),
        'tstamp_start_sec': float(tstamps[0]) if len(tstamps) else 0.0,
        'tstamp_end_sec': float(tstamps[-1]) if len(tstamps) else 0.0,
    }

    return SessionResult(
        session_id=meta['session_id'],
        subject=meta['subject'],
        neural_trials=neural_trials,
        input_trials=input_trials,
        output_trials=output_trials,
        brain_region_idx=np.zeros(neural_b.shape[0], dtype=np.int64),
        metadata=meta,
    )


def maybe_plot_session(sess_result: SessionResult, outdir: Path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n_trials = min(2, len(sess_result.neural_trials))
    if n_trials <= 0:
        return
    fig, axes = plt.subplots(3, n_trials, figsize=(6 * n_trials, 8), squeeze=False)
    for i in range(n_trials):
        neural = sess_result.neural_trials[i]
        inp = sess_result.input_trials[i][0]
        out = sess_result.output_trials[i][0]
        axes[0, i].imshow(neural[:min(50, neural.shape[0])], aspect='auto', cmap='viridis')
        axes[0, i].set_title(f'{sess_result.session_id} trial {i} neural')
        axes[1, i].plot(inp)
        axes[1, i].set_title('time elapsed (s)')
        axes[2, i].plot(out)
        axes[2, i].set_title('motion energy bin')
    plt.tight_layout()
    outpath = outdir / f"processing_{sess_result.session_id.replace('/', '_')}.png"
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def build_dataset(results):
    subjects = sorted({r.subject for r in results})
    subject_to_idx = {s: i for i, s in enumerate(subjects)}
    data = {
        'neural': [r.neural_trials for r in results],
        'input': [r.input_trials for r in results],
        'output': [r.output_trials for r in results],
        'subjects': subjects,
        'subject_idx': np.asarray([subject_to_idx[r.subject] for r in results], dtype=np.int64),
        'brain_regions': ['barrel cortex'],
        'brain_region_idx': [r.brain_region_idx for r in results],
        'input_names': ['time_from_session_start_sec'],
        'output_names': ['motion_energy_bin'],
        'output_values': [[f'bin_{i}' for i in range(5)]],
        'metadata': {
            'task_description': 'Decode session-wise motion energy from barrel-cortex calcium activity in longitudinal mouse recordings.',
            'time_bin_size': float(results[0].metadata['time_bin_size_sec'] * 1000.0) if results else None,
            'temporal_alignment_event': 'session start; continuous session split into contiguous 60-second windows',
            'off_start': 0.0,
            'off_end': 60.0,
            'preprocessing': {
                'cell_filter': 'iscell[:,0] > 0.5',
                'neuropil_correction': 'F - 0.7 * Fneu',
                'dff_baseline': 'per-neuron session 10th percentile of neuropil-corrected fluorescence',
                'smoothing': 'non-overlapping mean over 10 frames for neural and motion energy',
                'output_discretization': '5 equal-percentile bins computed separately within each session after smoothing',
            },
            'session_info': [r.metadata for r in results],
        },
    }
    return data


def validate_dataset(data):
    assert len(data['neural']) == len(data['input']) == len(data['output']) == len(data['subject_idx'])
    for s in range(len(data['neural'])):
        assert len(data['neural'][s]) == len(data['input'][s]) == len(data['output'][s])
        assert len(data['neural'][s]) >= 2
        n_neurons = data['neural'][s][0].shape[0]
        assert data['brain_region_idx'][s].shape == (n_neurons,)
        for tr_n, tr_i, tr_o in zip(data['neural'][s], data['input'][s], data['output'][s]):
            assert tr_n.ndim == 2 and tr_i.ndim == 2 and tr_o.ndim == 2
            T = tr_n.shape[1]
            assert tr_i.shape[1] == T and tr_o.shape[1] == T
            assert tr_i.shape[0] == 1 and tr_o.shape[0] == 1


def main():
    args = parse_args()
    t0 = time.time()
    data_root = Path('/app/data')
    outpath = Path(args.outpicklefile)
    sessions = discover_sessions(data_root)
    if not args.full and not args.sample:
        args.full = True
    if args.sample:
        sessions = sessions[:2]

    print(f'Found {len(sessions)} sessions to process')
    results = []
    plot_budget = 2
    for i, sess_dir in enumerate(sessions, start=1):
        ts = time.time()
        res = load_session(sess_dir)
        results.append(res)
        dt = time.time() - ts
        print(f'[{i}/{len(sessions)}] {res.session_id}: neurons={res.metadata["n_neurons"]} trials={res.metadata["n_trials"]} binned_T={res.metadata["n_binned_timepoints"]} in {dt:.2f}s')
        if args.show_processing and plot_budget > 0:
            maybe_plot_session(res, outpath.parent)
            plot_budget -= 1

    data = build_dataset(results)
    validate_dataset(data)

    with open(outpath, 'wb') as f:
        pickle.dump(data, f)

    n_sessions = len(data['neural'])
    n_trials = sum(len(x) for x in data['neural'])
    n_neurons_total = sum(arr.shape[0] for arr in data['brain_region_idx'])
    print(f'Saved {outpath}')
    print(f'Summary: sessions={n_sessions}, trials={n_trials}, total_neurons={n_neurons_total}, subjects={len(data["subjects"])}')
    print(f'Total elapsed: {time.time() - t0:.2f}s')


if __name__ == '__main__':
    main()
