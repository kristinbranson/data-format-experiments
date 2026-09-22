#!/usr/bin/env python3
"""Convert the supplied Track2p barrel-cortex dataset to decoder format.

Usage: python -u /app/convert_data.py OUT.pkl [--full|--sample] [--show-processing]
"""
from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter1d, minimum_filter1d

DATA_ROOT = Path('/app/data')
NATIVE_FRAMES_PER_TRIAL = 1800  # 60 s * 30 Hz
AVERAGE_FRAMES = 10
N_BINS_PER_TRIAL = NATIVE_FRAMES_PER_TRIAL // AVERAGE_FRAMES


def suite2p_preprocess(F: np.ndarray, ops: dict) -> np.ndarray:
    """Reproduce suite2p.extraction.dcnv.preprocess using session ops.

    Input must already be neuropil corrected. A new float32 array is returned.
    Suite2p calls this baseline-corrected fluorescence; the paper calls it dF/F.
    """
    x = np.asarray(F, dtype=np.float32).copy()
    baseline = ops.get('baseline', 'maximin')
    sig = float(ops.get('sig_baseline', 10.0))
    fs = float(ops['fs'])
    win = max(1, int(float(ops.get('win_baseline', 60.0)) * fs))
    prct = float(ops.get('prctile_baseline', 8.0))
    if baseline == 'maximin':
        flow = gaussian_filter(x, sigma=(0.0, sig), mode='reflect')
        flow = minimum_filter1d(flow, size=win, axis=1, mode='reflect')
        flow = maximum_filter1d(flow, size=win, axis=1, mode='reflect')
    elif baseline == 'constant':
        flow = gaussian_filter(x, sigma=(0.0, sig), mode='reflect')
        flow = np.amin(flow, axis=1, keepdims=True)
    elif baseline == 'constant_prctile':
        flow = np.percentile(x, prct, axis=1, keepdims=True)
    else:
        flow = 0.0
    x -= flow
    return x


def average_blocks_2d(x: np.ndarray, block: int = AVERAGE_FRAMES) -> np.ndarray:
    """Average final dimension in non-overlapping blocks."""
    if x.ndim != 2 or x.shape[1] % block:
        raise ValueError(f'Expected 2D array with time divisible by {block}, got {x.shape}')
    return x.reshape(x.shape[0], -1, block).mean(axis=2, dtype=np.float32)


def average_blocks_1d(x: np.ndarray, block: int = AVERAGE_FRAMES) -> np.ndarray:
    if x.ndim != 1 or x.size % block:
        raise ValueError(f'Expected 1D array divisible by {block}, got {x.shape}')
    return x.reshape(-1, block).mean(axis=1, dtype=np.float64)


def discover_sessions() -> list[Path]:
    return sorted(DATA_ROOT.glob('*/*/suite2p/plane0'))


def plot_processing(session_id: str, raw_corr: np.ndarray, corrected: np.ndarray,
                    neural_binned: np.ndarray, motion_raw: np.ndarray,
                    motion_binned: np.ndarray, labels: np.ndarray,
                    edges: np.ndarray) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    nshow = min(3, raw_corr.shape[0])
    native_n = min(raw_corr.shape[1], 30 * 120)
    binned_n = native_n // AVERAGE_FRAMES
    t_native = np.arange(native_n) / 30.0
    t_bin = (np.arange(binned_n) * AVERAGE_FRAMES + 4.5) / 30.0
    fig, ax = plt.subplots(4, 1, figsize=(14, 12), constrained_layout=True)
    for i in range(nshow):
        ax[0].plot(t_native, raw_corr[i, :native_n], lw=.45, alpha=.75, label=f'cell {i} raw neuropil-corrected')
        ax[0].plot(t_native, corrected[i, :native_n], lw=.7, label=f'cell {i} baseline-corrected')
    ax[0].set(title='Neural processing: neuropil and Suite2p baseline correction', ylabel='fluorescence')
    ax[0].legend(ncol=2, fontsize=7)
    for i in range(nshow):
        ax[1].plot(t_bin, neural_binned[i, :binned_n], lw=.8, label=f'cell {i}')
    ax[1].set(title='10-frame averaged neural activity (3 Hz)', ylabel='corrected fluorescence')
    ax[1].legend(fontsize=7)
    ax[2].plot(t_native, motion_raw[:native_n], color='0.7', lw=.5, label='native 30 Hz')
    ax[2].plot(t_bin, motion_binned[:binned_n], color='tab:blue', lw=1, label='10-frame mean')
    ax[2].set(title='Synchronized motion-energy processing', ylabel='motion energy')
    ax[2].legend(fontsize=8)
    full_t = (np.arange(labels.size) * AVERAGE_FRAMES + 4.5) / 30.0
    ax[3].plot(full_t, labels, lw=.55, color='tab:purple')
    counts = np.bincount(labels, minlength=5)
    ax[3].set(title=f'Per-session quintiles; edges={np.array2string(edges, precision=2)}; counts={counts.tolist()}',
              xlabel='elapsed session time (s)', ylabel='class', yticks=range(5))
    safe = session_id.replace('/', '_')
    out = Path('/app') / f'processing_{safe}.png'
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f'  saved processing plot: {out}', flush=True)


def process_session(sp: Path, show_processing: bool = False) -> tuple[list, list, list, np.ndarray, dict]:
    start = time.perf_counter()
    session_dir = sp.parent.parent
    subject = session_dir.parent.name
    session_name = session_dir.name
    sid = f'{subject}/{session_name}'

    F = np.load(sp / 'F.npy', mmap_mode='r')
    Fneu = np.load(sp / 'Fneu.npy', mmap_mode='r')
    iscell = np.load(sp / 'iscell.npy')
    ops = np.load(sp / 'ops.npy', allow_pickle=True).item()
    motion = np.load(session_dir / 'move_deve' / 'motion_energy_glob.npy', mmap_mode='r')
    tstamps = np.load(session_dir / 'move_deve' / 'tstamps.npy', mmap_mode='r')

    fs = float(ops['fs'])
    if not np.isclose(fs, 30.0):
        raise ValueError(f'{sid}: expected 30 Hz, found {fs}')
    if F.shape != Fneu.shape or F.ndim != 2:
        raise ValueError(f'{sid}: incompatible F/Fneu shapes {F.shape}, {Fneu.shape}')
    cell_mask = iscell[:, 1] > 0.5
    if len(cell_mask) != F.shape[0]:
        raise ValueError(f'{sid}: iscell/F neuron mismatch')
    # Distributed data are already curated; still enforce the paper criterion.
    if not np.all(cell_mask):
        print(f'  {sid}: filtering {np.sum(~cell_mask)} ROIs below iscell threshold', flush=True)

    common = min(F.shape[1], Fneu.shape[1], motion.size, tstamps.size)
    keep_native = (common // NATIVE_FRAMES_PER_TRIAL) * NATIVE_FRAMES_PER_TRIAL
    if keep_native < 2 * NATIVE_FRAMES_PER_TRIAL:
        raise ValueError(f'{sid}: fewer than two complete 60-s trials')
    dropped_neural = F.shape[1] - keep_native
    dropped_motion = motion.size - keep_native

    neucoeff = float(ops.get('neucoeff', 0.7))
    # Match Suite2p/reference ordering: estimate the baseline on the complete
    # neural recording, then restrict to synchronized complete trials. Filtering
    # an already truncated prefix would introduce an artificial endpoint baseline.
    raw_corr_full = (np.asarray(F[cell_mask, :], dtype=np.float32)
                     - neucoeff * np.asarray(Fneu[cell_mask, :], dtype=np.float32))
    corrected_full = suite2p_preprocess(raw_corr_full, ops)
    raw_corr = raw_corr_full[:, :keep_native]
    corrected = corrected_full[:, :keep_native]
    neural_binned = average_blocks_2d(corrected)
    motion_native = np.asarray(motion[:keep_native], dtype=np.float64)
    motion_binned = average_blocks_1d(motion_native)

    edges = np.quantile(motion_binned, [0.2, 0.4, 0.6, 0.8])
    if np.unique(edges).size != 4:
        raise ValueError(f'{sid}: non-distinct quintile edges {edges}')
    labels = np.searchsorted(edges, motion_binned, side='right').astype(np.int64)
    elapsed = ((np.arange(keep_native // AVERAGE_FRAMES, dtype=np.float64) * AVERAGE_FRAMES
                + (AVERAGE_FRAMES - 1) / 2) / fs).astype(np.float32)

    n_trials = keep_native // NATIVE_FRAMES_PER_TRIAL
    neural_trials = [np.ascontiguousarray(x, dtype=np.float32)
                     for x in neural_binned.reshape(neural_binned.shape[0], n_trials, N_BINS_PER_TRIAL).transpose(1, 0, 2)]
    input_trials = [np.ascontiguousarray(x[None, :], dtype=np.float32)
                    for x in elapsed.reshape(n_trials, N_BINS_PER_TRIAL)]
    output_trials = [np.ascontiguousarray(x[None, :], dtype=np.int64)
                     for x in labels.reshape(n_trials, N_BINS_PER_TRIAL)]

    for ni, ii, oo in zip(neural_trials, input_trials, output_trials):
        assert ni.shape == (int(cell_mask.sum()), N_BINS_PER_TRIAL)
        assert ii.shape == oo.shape == (1, N_BINS_PER_TRIAL)
        assert np.isfinite(ni).all() and np.isfinite(ii).all() and np.isfinite(oo).all()
    assert labels.min() == 0 and labels.max() == 4

    if show_processing:
        plot_processing(sid, raw_corr, corrected, neural_binned, motion_native,
                        motion_binned, labels, edges)

    info = {
        'session_id': sid,
        'subject': subject,
        'native_neural_frames': int(F.shape[1]),
        'native_motion_frames': int(motion.size),
        'retained_native_frames': int(keep_native),
        'discarded_neural_endpoint_frames': int(dropped_neural),
        'discarded_motion_endpoint_frames': int(dropped_motion),
        'n_neurons': int(cell_mask.sum()),
        'n_trials': int(n_trials),
        'quintile_edges': edges.tolist(),
        'class_counts': np.bincount(labels, minlength=5).tolist(),
    }
    region_idx = np.zeros(int(cell_mask.sum()), dtype=np.int64)
    elapsed_s = time.perf_counter() - start
    print(f'  {sid}: {region_idx.size} neurons, {n_trials} trials, '
          f'discarded endpoint frames neural={dropped_neural}, motion={dropped_motion}, '
          f'{elapsed_s:.2f}s', flush=True)
    return neural_trials, input_trials, output_trials, region_idx, info


def convert(out_path: Path, sample: bool, show_processing: bool) -> None:
    overall = time.perf_counter()
    sessions = discover_sessions()
    if sample:
        sessions = sessions[:2]
    if not sessions:
        raise FileNotFoundError(f'No sessions found under {DATA_ROOT}')
    subjects = sorted({p.parent.parent.parent.name for p in sessions})
    subject_to_idx = {s: i for i, s in enumerate(subjects)}
    print(f'Converting {len(sessions)} sessions from {len(subjects)} subjects', flush=True)

    neural, inputs, outputs, regions, infos, subject_idx = [], [], [], [], [], []
    for i, sp in enumerate(sessions):
        show = show_processing and i < 2
        n, x, y, r, info = process_session(sp, show)
        neural.append(n); inputs.append(x); outputs.append(y); regions.append(r); infos.append(info)
        subject_idx.append(subject_to_idx[info['subject']])

    data = {
        'neural': neural,
        'input': inputs,
        'output': outputs,
        'subjects': subjects,
        'subject_idx': np.asarray(subject_idx, dtype=np.int64),
        'brain_regions': ['barrel cortex L2/3'],
        'brain_region_idx': regions,
        'input_names': ['elapsed_time_from_session_start_s'],
        'output_names': ['motion_energy_quintile'],
        'output_values': [[
            'lowest motion energy', 'low motion energy', 'middle motion energy',
            'high motion energy', 'highest motion energy'
        ]],
        'metadata': {
            'task_description': 'Decode within-session motion-energy quintile from baseline-corrected calcium fluorescence.',
            'time_bin_size': 1000.0 * AVERAGE_FRAMES / 30.0,
            'temporal_alignment_event': 'Start of each consecutive non-overlapping 60-second window in a continuous session',
            'off_start': 0.0,
            'off_end': 60.0,
            'native_sampling_rate_hz': 30.0,
            'temporal_averaging_frames': AVERAGE_FRAMES,
            'trial_duration_s': 60.0,
            'neural_processing': 'F - 0.7*Fneu (or saved neucoeff), Suite2p saved maximin baseline correction, then 10-frame mean',
            'behavior_processing': 'Global squared frame-difference motion energy, 10-frame mean, session-specific quintiles',
            'alignment_note': 'Imaging-triggered video aligned by frame ordinal; incomplete endpoint windows discarded without interpolation.',
            'session_info': infos,
        },
    }
    if not (len(neural) == len(inputs) == len(outputs) == len(regions) == len(subject_idx)):
        raise AssertionError('Session-list lengths differ')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = out_path.stat().st_size / 1024**2
    total_trials = sum(map(len, neural))
    print(f'Saved {out_path} ({size_mb:.1f} MiB): {len(neural)} sessions, '
          f'{total_trials} trials in {time.perf_counter()-overall:.2f}s', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('outpicklefile', type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--full', action='store_true', help='process all sessions (default)')
    mode.add_argument('--sample', action='store_true', help='process only first 2 sessions')
    parser.add_argument('--show-processing', action='store_true',
                        help='save processing plots for up to 2 sessions')
    args = parser.parse_args()
    convert(args.outpicklefile, sample=args.sample, show_processing=args.show_processing)


if __name__ == '__main__':
    main()
