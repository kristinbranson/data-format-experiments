#!/usr/bin/env python3
import argparse
import math
import os
import pickle
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


FS = 30.0
BIN_FRAMES = 10
BLOCK_SECONDS = 120.0
BLOCK_FRAMES = int(FS * BLOCK_SECONDS)
BLOCK_BINS = BLOCK_FRAMES // BIN_FRAMES
NEUCOEFF_DEFAULT = 0.7


def discover_subjects(data_root: Path):
    subjects = sorted([p for p in data_root.iterdir() if p.is_dir() and p.name.startswith('jm')])
    sessions_by_subject = {}
    for subj in subjects:
        sessions = sorted([p for p in subj.iterdir() if p.is_dir() and p.name[:4].isdigit()])
        sessions_by_subject[subj.name] = sessions
    return subjects, sessions_by_subject


def moving_average_reflect(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x.astype(np.float32, copy=True)
    pad = win // 2
    xp = np.pad(x.astype(np.float32), (pad, pad), mode='reflect')
    kernel = np.ones(win, dtype=np.float32) / win
    y = np.convolve(xp, kernel, mode='valid')
    return y[: x.shape[0]].astype(np.float32)


def compute_dff(F: np.ndarray, Fneu: np.ndarray, ops: dict) -> np.ndarray:
    neucoeff = float(ops.get('neucoeff', NEUCOEFF_DEFAULT))
    Fcorr = F.astype(np.float32) - neucoeff * Fneu.astype(np.float32)
    win_seconds = float(ops.get('win_baseline', 60.0))
    win = max(1, int(round(win_seconds * float(ops.get('fs', FS)))))
    prct = float(ops.get('prctile_baseline', 8.0))
    baseline = np.empty_like(Fcorr, dtype=np.float32)
    for i in range(Fcorr.shape[0]):
        smooth = moving_average_reflect(Fcorr[i], win)
        base = np.percentile(smooth, prct)
        baseline[i] = base
    baseline = np.maximum(baseline, 1e-3)
    dff = (Fcorr - baseline) / baseline
    return dff.astype(np.float32)


def align_motion_to_frames(motion: np.ndarray, tstamps: np.ndarray, nframes: int) -> np.ndarray:
    motion = np.asarray(motion, dtype=np.float32)
    aligned = np.full(nframes, np.nan, dtype=np.float32)
    # Data README states motion_energy_glob is framewise with occasional missing camera frames.
    # `tstamps.npy` stores timestamps (seconds), not frame indices, so alignment should preserve
    # sample order and pad/truncate to imaging length.
    m = min(nframes, motion.shape[0])
    aligned[:m] = motion[:m]
    return aligned


def nanbin_mean_1d(x: np.ndarray, bin_frames: int) -> np.ndarray:
    n = (len(x) // bin_frames) * bin_frames
    x = x[:n].reshape(-1, bin_frames)
    valid = np.isfinite(x)
    sums = np.where(valid, x, 0.0).sum(axis=1)
    counts = valid.sum(axis=1)
    out = np.full(x.shape[0], np.nan, dtype=np.float32)
    nz = counts > 0
    out[nz] = (sums[nz] / counts[nz]).astype(np.float32)
    return out


def bin_neural(x: np.ndarray, bin_frames: int) -> np.ndarray:
    n = (x.shape[1] // bin_frames) * bin_frames
    x = x[:, :n]
    return x.reshape(x.shape[0], -1, bin_frames).mean(axis=2).astype(np.float32)


def session_to_blocks(neural_binned: np.ndarray, motion_binned: np.ndarray):
    n_bins = min(neural_binned.shape[1], motion_binned.shape[0])
    neural_binned = neural_binned[:, :n_bins]
    motion_binned = motion_binned[:n_bins]
    n_blocks = n_bins // BLOCK_BINS
    neural_blocks = []
    motion_blocks = []
    time_blocks = []
    for b in range(n_blocks):
        s = b * BLOCK_BINS
        e = (b + 1) * BLOCK_BINS
        neural_blocks.append(neural_binned[:, s:e].astype(np.float32))
        motion_blocks.append(motion_binned[s:e].astype(np.float32))
        time_blocks.append((np.arange(BLOCK_BINS, dtype=np.float32) * (BIN_FRAMES / FS))[None, :])
    return neural_blocks, motion_blocks, time_blocks


def plot_processing(session_id: str, raw_motion: np.ndarray, aligned_motion: np.ndarray, binned_motion: np.ndarray,
                    raw_neural: np.ndarray, binned_neural: np.ndarray):
    fig, ax = plt.subplots(4, 1, figsize=(14, 10), constrained_layout=True)
    ax[0].plot(raw_motion[:2000])
    ax[0].set_title(f'{session_id}: raw motion energy')
    ax[1].plot(aligned_motion[:2000])
    ax[1].set_title('aligned motion energy (NaNs indicate missing camera frames)')
    ax[2].plot(binned_motion[:300])
    ax[2].set_title('10-frame binned motion energy')
    im0 = ax[3].imshow(raw_neural[: min(40, raw_neural.shape[0]), : min(2000, raw_neural.shape[1])], aspect='auto', cmap='viridis')
    ax[3].set_title('raw neural signal (subset neurons/time)')
    fig.colorbar(im0, ax=ax[3], shrink=0.6)
    fig.savefig(f'processing_{session_id}.png', dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(14, 4), constrained_layout=True)
    im1 = ax.imshow(binned_neural[: min(40, binned_neural.shape[0]), : min(300, binned_neural.shape[1])], aspect='auto', cmap='viridis')
    ax.set_title(f'{session_id}: 10-frame binned neural signal')
    fig.colorbar(im1, ax=ax, shrink=0.7)
    fig.savefig(f'processing_{session_id}_binned.png', dpi=150)
    plt.close(fig)


def load_session(session_dir: Path):
    pl0 = session_dir / 'suite2p' / 'plane0'
    move = session_dir / 'move_deve'
    F = np.load(pl0 / 'F.npy', allow_pickle=True)
    Fneu = np.load(pl0 / 'Fneu.npy', allow_pickle=True)
    ops = np.load(pl0 / 'ops.npy', allow_pickle=True).item()
    stat = np.load(pl0 / 'stat.npy', allow_pickle=True)
    motion = np.load(move / 'motion_energy_glob.npy', allow_pickle=True)
    tstamps = np.load(move / 'tstamps.npy', allow_pickle=True)
    interframe = np.load(move / 'interframe_int.npy', allow_pickle=True)
    return F, Fneu, ops, stat, motion, tstamps, interframe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('outpicklefile')
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument('--full', action='store_true', help='Process all sessions')
    mode.add_argument('--sample', action='store_true', help='Process only 2 sessions for testing')
    ap.add_argument('--show-processing', action='store_true', help='Save processing plots for up to 2 sessions')
    args = ap.parse_args()

    data_root = Path('data')
    subjects, sessions_by_subject = discover_subjects(data_root)
    subject_names = [p.name for p in subjects]

    selected = []
    for subj in subject_names:
        for sess in sessions_by_subject[subj]:
            selected.append((subj, sess))
    if args.sample:
        selected = selected[:2]

    all_motion_values = []
    prepared = []
    t0 = time.time()
    for idx, (subj, sessdir) in enumerate(selected):
        F, Fneu, ops, stat, motion, tstamps, interframe = load_session(sessdir)
        dff = compute_dff(F, Fneu, ops)
        aligned_motion = align_motion_to_frames(motion, tstamps, dff.shape[1])
        neural_binned = bin_neural(dff, BIN_FRAMES)
        motion_binned = nanbin_mean_1d(aligned_motion, BIN_FRAMES)
        neural_blocks, motion_blocks, time_blocks = session_to_blocks(neural_binned, motion_binned)
        prepared.append({
            'subject': subj,
            'session': sessdir.name,
            'neural_blocks': neural_blocks,
            'motion_blocks': motion_blocks,
            'time_blocks': time_blocks,
            'n_neurons': dff.shape[0],
        })
        for mb in motion_blocks:
            all_motion_values.append(mb[np.isfinite(mb)])
        if args.show_processing and idx < 2:
            plot_processing(f'{subj}_{sessdir.name}', motion, aligned_motion, motion_binned, dff, neural_binned)
        print(f'prepared {subj} {sessdir.name}: neurons={dff.shape[0]} blocks={len(neural_blocks)} frames={dff.shape[1]}')

    all_motion_values = np.concatenate([x for x in all_motion_values if x.size > 0])
    edges = np.quantile(all_motion_values, [0.2, 0.4, 0.6, 0.8]).astype(np.float32)
    print('motion bin edges:', edges)

    data = {
        'neural': [],
        'input': [],
        'output': [],
        'subjects': subject_names,
        'subject_idx': [],
        'brain_regions': ['barrel cortex L2/3'],
        'brain_region_idx': [],
        'input_names': ['time_elapsed_s'],
        'output_names': ['motion_energy_bin'],
        'output_values': [[f'bin_{i}' for i in range(5)]],
        'metadata': {
            'task_description': 'Decode spontaneous-motion energy from longitudinal barrel-cortex calcium imaging using continuous recordings segmented into 2-minute blocks.',
            'time_bin_size': 1000.0 * BIN_FRAMES / FS,
            'temporal_alignment_event': 'start of each 2-minute continuous recording block',
            'off_start': 0.0,
            'off_end': BLOCK_SECONDS,
            'fs_hz': FS,
            'bin_frames': BIN_FRAMES,
            'block_seconds': BLOCK_SECONDS,
            'notes': 'Motion energy aligned to imaging frames using tstamps; missing camera frames remain NaN before binning. Neural signal computed from F and Fneu using neuropil subtraction and percentile baseline normalization.',
        }
    }

    for item in prepared:
        sess_neural = []
        sess_input = []
        sess_output = []
        for nb, mb, tb in zip(item['neural_blocks'], item['motion_blocks'], item['time_blocks']):
            y = np.digitize(mb, edges, right=False).astype(np.int64)
            y[np.isnan(mb)] = -1
            valid = y >= 0
            if valid.sum() < max(10, int(0.8 * len(y))):
                continue
            # fill missing labels by nearest valid previous then next if needed
            if not np.all(valid):
                yy = y.copy()
                last = None
                for i in range(len(yy)):
                    if yy[i] >= 0:
                        last = yy[i]
                    elif last is not None:
                        yy[i] = last
                nxt = None
                for i in range(len(yy)-1, -1, -1):
                    if yy[i] >= 0:
                        nxt = yy[i]
                    elif nxt is not None:
                        yy[i] = nxt
                yy[yy < 0] = 0
                y = yy
            sess_neural.append(nb.astype(np.float32))
            sess_input.append(tb.astype(np.float32))
            sess_output.append(y[None, :].astype(np.int64))
        if len(sess_neural) >= 2:
            data['neural'].append(sess_neural)
            data['input'].append(sess_input)
            data['output'].append(sess_output)
            data['subject_idx'].append(subject_names.index(item['subject']))
            data['brain_region_idx'].append(np.zeros(item['n_neurons'], dtype=np.int64))

    data['subject_idx'] = np.asarray(data['subject_idx'], dtype=np.int64)

    with open(args.outpicklefile, 'wb') as f:
        pickle.dump(data, f)
    print(f'saved {args.outpicklefile}')
    print(f'n_sessions={len(data["neural"])} total_trials={sum(len(x) for x in data["neural"])} elapsed={time.time()-t0:.2f}s')


if __name__ == '__main__':
    main()
