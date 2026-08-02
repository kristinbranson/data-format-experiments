#!/usr/bin/env python3
import argparse
import math
import os
import pickle
import time
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description='Convert Track2p/Suite2p dataset to decoder format')
    p.add_argument('outpicklefile')
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--full', action='store_true', help='Process all sessions (default)')
    mode.add_argument('--sample', action='store_true', help='Process only 2 sessions for testing')
    p.add_argument('--show-processing', action='store_true', help='Save processing plots for up to 2 sessions')
    return p.parse_args()


def compute_dff(F, Fneu, ops):
    neucoeff = float(ops.get('neucoeff', 0.7))
    Fc = F - neucoeff * Fneu
    prctile_baseline = float(ops.get('prctile_baseline', 8.0))
    # Fast approximation to suite2p baseline-corrected fluorescence:
    # per-neuron low-percentile baseline after neuropil correction.
    F0 = np.percentile(Fc, prctile_baseline, axis=1, keepdims=True).astype(np.float32)
    F0 = np.maximum(F0, 1e-3)
    dff = (Fc - F0) / F0
    return dff.astype(np.float32)


def bin_time_series(x, bin_size):
    # x shape (..., T)
    T = x.shape[-1]
    T2 = (T // bin_size) * bin_size
    x = x[..., :T2]
    new_shape = x.shape[:-1] + (T2 // bin_size, bin_size)
    return x.reshape(new_shape).mean(axis=-1)


def load_session(sess_path):
    suite = sess_path / 'suite2p' / 'plane0'
    move = sess_path / 'move_deve'
    F = np.load(suite / 'F.npy', mmap_mode='r')
    Fneu = np.load(suite / 'Fneu.npy', mmap_mode='r')
    iscell = np.load(suite / 'iscell.npy')
    ops = np.load(suite / 'ops.npy', allow_pickle=True).item()
    motion = np.load(move / 'motion_energy_glob.npy', mmap_mode='r').astype(np.float32)
    tstamps = np.load(move / 'tstamps.npy', mmap_mode='r').astype(np.float64)
    keep = iscell[:, 1] > 0.5
    dff = compute_dff(np.asarray(F[keep], dtype=np.float32), np.asarray(Fneu[keep], dtype=np.float32), ops)
    T = min(dff.shape[1], motion.shape[0], tstamps.shape[0])
    dff = dff[:, :T]
    motion = motion[:T]
    tstamps = tstamps[:T]
    return {
        'dff': dff,
        'motion': motion,
        'tstamps': tstamps,
        'keep': keep,
        'ops': ops,
    }


def make_blocks(dff_binned, motion_binned, time_binned, block_bins):
    T = dff_binned.shape[1]
    n_blocks = T // block_bins
    T2 = n_blocks * block_bins
    dff_binned = dff_binned[:, :T2]
    motion_binned = motion_binned[:T2]
    time_binned = time_binned[:T2]
    neural_trials, input_trials, output_cont = [], [], []
    for i in range(n_blocks):
        sl = slice(i * block_bins, (i + 1) * block_bins)
        neural_trials.append(dff_binned[:, sl].astype(np.float32))
        input_trials.append(time_binned[sl][None, :].astype(np.float32))
        output_cont.append(motion_binned[sl][None, :].astype(np.float32))
    return neural_trials, input_trials, output_cont


def save_processing_plot(sess_id, raw_motion, binned_motion, dff, dff_binned):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), constrained_layout=True)
    axes[0].plot(raw_motion[:2000])
    axes[0].set_title(f'{sess_id}: raw motion energy (first 2000 frames)')
    axes[1].plot(binned_motion[:300])
    axes[1].set_title('binned motion energy (first 300 bins)')
    axes[2].imshow(dff[: min(50, dff.shape[0]), : min(1000, dff.shape[1])], aspect='auto', cmap='viridis')
    axes[2].set_title('raw dF/F sample neurons x time')
    axes[3].imshow(dff_binned[: min(50, dff_binned.shape[0]), : min(300, dff_binned.shape[1])], aspect='auto', cmap='viridis')
    axes[3].set_title('binned dF/F sample neurons x time')
    fig.savefig(f'processing_{sess_id}.png', dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    t0 = time.time()
    root = Path('data')
    subjects = sorted([p.name for p in root.iterdir() if p.is_dir()])
    session_paths = []
    session_subjects = []
    for subj in subjects:
        subj_path = root / subj
        for sess in sorted([p for p in subj_path.iterdir() if p.is_dir()]):
            session_paths.append(sess)
            session_subjects.append(subj)
    if args.sample:
        session_paths = session_paths[:2]
        session_subjects = session_subjects[:2]

    print(f'Found {len(session_paths)} sessions from {len(set(session_subjects))} subjects')

    bin_size_frames = 10
    raw_fs = 30.0
    time_bin_size_ms = 1000.0 * bin_size_frames / raw_fs
    block_bins = int((2 * 60 * raw_fs) / bin_size_frames)  # 2 minutes after binning

    all_neural = []
    all_input = []
    all_output_cont = []
    brain_region_idx = []
    subject_idx = []
    session_ids = []
    all_motion_values = []

    for i, (sess_path, subj) in enumerate(zip(session_paths, session_subjects)):
        st = time.time()
        sess_id = f'{sess_path.parent.name}__{sess_path.name}'
        loaded = load_session(sess_path)
        dff = loaded['dff']
        motion = loaded['motion']
        tstamps = loaded['tstamps']
        dff_binned = bin_time_series(dff, bin_size_frames)
        motion_binned = bin_time_series(motion[None, :], bin_size_frames).squeeze(0)
        frame_time = (np.arange(dff.shape[1], dtype=np.float32) / raw_fs)
        time_binned = bin_time_series(frame_time[None, :], bin_size_frames).squeeze(0)
        neural_trials, input_trials, output_cont = make_blocks(dff_binned, motion_binned, time_binned, block_bins)
        if len(neural_trials) < 2:
            print(f'Skipping {sess_id}: fewer than 2 blocks')
            continue
        all_neural.append(neural_trials)
        all_input.append(input_trials)
        all_output_cont.append(output_cont)
        all_motion_values.append(np.concatenate([x.ravel() for x in output_cont]))
        brain_region_idx.append(np.zeros(neural_trials[0].shape[0], dtype=np.int64))
        subject_idx.append(subjects.index(subj))
        session_ids.append(sess_id)
        if args.show_processing and i < 2:
            save_processing_plot(sess_id, motion, motion_binned, dff, dff_binned)
        print(f'Processed {sess_id}: neurons={dff.shape[0]} raw_frames={dff.shape[1]} bins={dff_binned.shape[1]} blocks={len(neural_trials)} time={time.time()-st:.2f}s')

    all_motion_values = np.concatenate(all_motion_values)
    edges = np.percentile(all_motion_values, [20, 40, 60, 80]).astype(np.float32)
    print('Motion percentile edges:', edges)

    all_output = []
    for sess_trials in all_output_cont:
        out_trials = []
        for arr in sess_trials:
            x = arr.squeeze(0)
            labels = np.digitize(x, edges, right=False).astype(np.int64)
            labels = np.clip(labels, 0, 4)
            out_trials.append(labels[None, :])
        all_output.append(out_trials)

    data = {
        'neural': all_neural,
        'input': all_input,
        'output': all_output,
        'subjects': subjects,
        'subject_idx': np.asarray(subject_idx, dtype=np.int64),
        'brain_regions': ['barrel_cortex_L2_3'],
        'brain_region_idx': brain_region_idx,
        'input_names': ['time_from_session_start_s'],
        'output_names': ['motion_energy_bin'],
        'output_values': [[f'bin_{i}' for i in range(5)]],
        'metadata': {
            'task_description': 'Decode discretized global motion energy from spontaneous barrel-cortex calcium activity during developmental longitudinal imaging.',
            'time_bin_size': float(time_bin_size_ms),
            'temporal_alignment_event': 'session start',
            'off_start': 0.0,
            'off_end': 120.0,
            'raw_frame_rate_hz': raw_fs,
            'bin_size_frames': bin_size_frames,
            'block_duration_s': 120.0,
            'cell_filter': 'iscell_probability_gt_0.5',
            'motion_bin_edges_percentiles_20_40_60_80': edges.tolist(),
            'session_ids': session_ids,
            'notes': 'Neural traces computed from suite2p F/Fneu using neuropil correction and baseline normalization to approximate suite2p baseline-corrected dF/F; sessions split into consecutive 2-minute blocks after 10-frame averaging.',
        }
    }

    with open(args.outpicklefile, 'wb') as f:
        pickle.dump(data, f)
    print(f'Saved {args.outpicklefile}')
    print(f'Total time: {time.time()-t0:.2f}s')


if __name__ == '__main__':
    main()
