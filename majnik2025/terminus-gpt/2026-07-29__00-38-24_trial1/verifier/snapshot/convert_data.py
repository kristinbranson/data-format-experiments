import argparse
import math
import pickle
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path('data')
BRAIN_REGION = 'barrel cortex'
CELL_THRESHOLD = 0.5
BIN_FRAMES = 10
WINDOW_SECONDS = 120.0


def discover_sessions(root=ROOT):
    sessions = []
    for subj_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        for sess_dir in sorted([p for p in subj_dir.iterdir() if p.is_dir()]):
            suite = sess_dir / 'suite2p' / 'plane0'
            move = sess_dir / 'move_deve'
            req = [suite/'F.npy', suite/'Fneu.npy', suite/'iscell.npy', suite/'ops.npy', suite/'spks.npy', move/'motion_energy_glob.npy', move/'tstamps.npy']
            if all(p.exists() for p in req):
                sessions.append((subj_dir.name, sess_dir.name, sess_dir))
    return sessions


def load_session(sess_dir):
    suite = sess_dir / 'suite2p' / 'plane0'
    move = sess_dir / 'move_deve'
    F = np.load(suite / 'F.npy').astype(np.float32)
    Fneu = np.load(suite / 'Fneu.npy').astype(np.float32)
    iscell = np.load(suite / 'iscell.npy')
    ops = np.load(suite / 'ops.npy', allow_pickle=True).item()
    motion = np.load(move / 'motion_energy_glob.npy').astype(np.float32)
    tstamps = np.load(move / 'tstamps.npy').astype(np.float64)
    spks = np.load(suite / 'spks.npy').astype(np.float32)
    return F, Fneu, spks, iscell, ops, motion, tstamps


def compute_df_f(F, Fneu, ops):
    neucoeff = float(ops.get('neucoeff', 0.7))
    Fc = F - neucoeff * Fneu
    prctile = float(ops.get('prctile_baseline', 8.0))
    baseline = np.percentile(Fc, prctile, axis=1, keepdims=True).astype(np.float32)
    baseline = np.maximum(baseline, 1e-3)
    dff = (Fc - baseline) / baseline
    return dff.astype(np.float32)


def trim_to_common_length(neural, motion, tstamps):
    n = min(neural.shape[1], motion.shape[0], tstamps.shape[0])
    return neural[:, :n], motion[:n], tstamps[:n]


def bin_array_2d(x, bin_frames):
    n_bins = x.shape[1] // bin_frames
    x = x[:, :n_bins * bin_frames]
    return x.reshape(x.shape[0], n_bins, bin_frames).mean(axis=2)


def bin_array_1d(x, bin_frames):
    n_bins = x.shape[0] // bin_frames
    x = x[:n_bins * bin_frames]
    return x.reshape(n_bins, bin_frames).mean(axis=1)


def make_time_input(t_binned):
    return t_binned[None, :].astype(np.float32)


def segment_session(neural_b, motion_b, time_b, fs_binned, window_seconds=WINDOW_SECONDS):
    bins_per_window = max(1, int(round(window_seconds * fs_binned)))
    n_windows = neural_b.shape[1] // bins_per_window
    neural_trials, input_trials, motion_trials = [], [], []
    for i in range(n_windows):
        sl = slice(i * bins_per_window, (i + 1) * bins_per_window)
        neural_trials.append(neural_b[:, sl].astype(np.float32))
        input_trials.append(make_time_input(time_b[sl]))
        motion_trials.append(motion_b[sl].astype(np.float32)[None, :])
    return neural_trials, input_trials, motion_trials, bins_per_window


def percentile_bin_outputs(all_motion_trials, n_classes=5):
    vals = np.concatenate([m.ravel() for sess in all_motion_trials for m in sess])
    edges = np.quantile(vals, np.linspace(0, 1, n_classes + 1))
    edges[0] = -np.inf
    edges[-1] = np.inf
    out = []
    for sess in all_motion_trials:
        sess_out = []
        for m in sess:
            b = np.digitize(m.ravel(), edges[1:-1], right=False).astype(np.int64)
            sess_out.append(b[None, :])
        out.append(sess_out)
    return out, edges


def plot_processing(session_id, raw_neural, raw_motion, raw_t, binned_neural, binned_motion, binned_t, out_path):
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), constrained_layout=True)
    axes[0].plot(raw_t[:2000], raw_motion[:2000], lw=0.8)
    axes[0].set_title(f'{session_id}: raw motion energy (first 2000 frames)')
    axes[1].plot(raw_t[:2000], raw_neural[0, :2000], lw=0.8)
    axes[1].set_title('raw neural dF/F example cell (first 2000 frames)')
    axes[2].plot(binned_t[:500], binned_motion[:500], lw=0.8)
    axes[2].set_title('binned motion energy')
    axes[3].plot(binned_t[:500], binned_neural[0, :500], lw=0.8)
    axes[3].set_title('binned neural dF/F example cell')
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def convert(sample=False, show_processing=False):
    t0 = time.time()
    sessions = discover_sessions()
    if sample:
        sessions = sessions[:2]
    subjects = sorted({s[0] for s in sessions})
    subject_to_idx = {s: i for i, s in enumerate(subjects)}

    neural_all = []
    input_all = []
    motion_all = []
    subject_idx = []
    brain_region_idx = []
    session_info = []
    fs_binned_values = []

    for subj, sess_name, sess_dir in sessions:
        st = time.time()
        F, Fneu, spks, iscell, ops, motion, tstamps = load_session(sess_dir)
        keep = iscell[:, 0] > CELL_THRESHOLD if iscell.ndim == 2 else iscell > CELL_THRESHOLD
        neural = spks[keep].astype(np.float32)
        neural, motion, tstamps = trim_to_common_length(neural, motion, tstamps)
        neural_b = bin_array_2d(neural, BIN_FRAMES)
        motion_b = bin_array_1d(motion, BIN_FRAMES)
        _t_b_raw = bin_array_1d(tstamps, BIN_FRAMES)
        fs_binned = float(ops.get('fs', 30.0)) / BIN_FRAMES
        t_b = (np.arange(neural_b.shape[1], dtype=np.float32) / fs_binned).astype(np.float32)
        fs_binned_values.append(fs_binned)
        neural_trials, input_trials, motion_trials, bins_per_window = segment_session(neural_b, motion_b, t_b, fs_binned)
        if len(neural_trials) < 2:
            continue
        neural_all.append(neural_trials)
        input_all.append(input_trials)
        motion_all.append(motion_trials)
        subject_idx.append(subject_to_idx[subj])
        brain_region_idx.append(np.zeros(neural_trials[0].shape[0], dtype=np.int64))
        session_info.append({
            'subject': subj,
            'session': sess_name,
            'raw_frames_common': int(neural.shape[1]),
            'binned_timepoints': int(neural_b.shape[1]),
            'n_trials': len(neural_trials),
            'n_neurons': int(neural.shape[0]),
            'fs_binned_hz': fs_binned,
            'bins_per_window': bins_per_window,
        })
        if show_processing and len(session_info) <= 2:
            plot_processing(f'{subj}_{sess_name}', neural, motion, tstamps, neural_b, motion_b, t_b, f'processing_{subj}_{sess_name}.png')
        print(f'processed {subj}/{sess_name} in {time.time()-st:.2f}s: neurons={neural.shape[0]}, trials={len(neural_trials)}')

    if not motion_all:
        raise RuntimeError('No sessions produced at least two windows; check binning/window logic.')
    output_all, edges = percentile_bin_outputs(motion_all, n_classes=5)
    data = {
        'neural': neural_all,
        'input': input_all,
        'output': output_all,
        'subjects': subjects,
        'subject_idx': np.asarray(subject_idx, dtype=np.int64),
        'brain_regions': [BRAIN_REGION],
        'brain_region_idx': brain_region_idx,
        'input_names': ['time_from_session_start_s'],
        'output_names': ['motion_energy_bin'],
        'output_values': [[f'bin_{i}' for i in range(5)]],
        'metadata': {
            'task_description': 'Decode discretized global motion energy from barrel cortex calcium activity.',
            'time_bin_size': float(1000.0 * np.median([1.0 / fs for fs in fs_binned_values])) if fs_binned_values else float('nan'),
            'temporal_alignment_event': 'session start (continuous recording segmented into consecutive windows)',
            'off_start': 0.0,
            'off_end': WINDOW_SECONDS,
            'source_neural_signal': 'Suite2p deconvolved spikes from spks.npy (iscell-filtered)',
            'cell_inclusion_rule': 'iscell probability > 0.5',
            'behavior_source': 'global motion energy from move_deve/motion_energy_glob.npy',
            'binning_frames': BIN_FRAMES,
            'window_seconds': WINDOW_SECONDS,
            'motion_percentile_edges': edges.tolist(),
            'session_info': session_info,
        }
    }
    print(f'converted {len(neural_all)} sessions in {time.time()-t0:.2f}s')
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('outpicklefile')
    g = ap.add_mutually_exclusive_group()
    g.add_argument('--full', action='store_true')
    g.add_argument('--sample', action='store_true')
    ap.add_argument('--show-processing', action='store_true')
    args = ap.parse_args()
    sample = args.sample
    data = convert(sample=sample, show_processing=args.show_processing)
    with open(args.outpicklefile, 'wb') as f:
        pickle.dump(data, f)
    print(f'saved to {args.outpicklefile}')


if __name__ == '__main__':
    main()
