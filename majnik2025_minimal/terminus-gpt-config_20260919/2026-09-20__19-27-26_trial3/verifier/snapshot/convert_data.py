#!/usr/bin/env python3
"""Convert the Majnik et al. Track2p developmental dataset for decoding.

Decisions and provenance
------------------------
* Sessions and subjects are sorted chronologically/alphabetically.  Each Suite2p
  output contains only neurons successfully tracked on every day of that mouse;
  these are already reindexed identically across days.  iscell is nevertheless
  checked against the paper's default 0.5 cell-probability threshold.
* Neural activity is Suite2p's processed/deconvolved ``spks.npy``.  The supplied
  loading notebook explicitly recommends this as an analysis-ready alternative
  to deriving dF/F from raw F.  This avoids inventing a dF/F normalization from
  F without the neuropil trace while retaining the paper's Suite2p processing.
* The paper averages neural and behavioral traces in non-overlapping groups of
  10 timestamps before decoding; the same operation is used here (30 Hz -> 3 Hz).
* A few videos dropped frames.  As documented in data/README.md, gaps are found
  in camera timestamps and missing motion values are linearly interpolated onto
  the neural-frame timeline.  No session or trial is discarded for these sparse
  camera losses.
* The requested trials are consecutive, complete 60-s blocks. Motion quintile
  boundaries are estimated separately for each session after temporal averaging.
"""
from pathlib import Path
import pickle
import numpy as np

ROOT = Path('/app/data')
OUT = Path('/app/converted_data.pkl')
FS = 30.0
AVERAGE_FRAMES = 10
TRIAL_SECONDS = 60
BIN_SECONDS = AVERAGE_FRAMES / FS
BINS_PER_TRIAL = int(round(TRIAL_SECONDS / BIN_SECONDS))


def restore_motion(motion, timestamps, n_frames):
    """Align camera motion to neural frames, interpolating dropped frames."""
    motion = np.asarray(motion, dtype=np.float64)
    timestamps = np.asarray(timestamps, dtype=np.float64)
    if len(motion) != len(timestamps):
        raise ValueError('motion and camera timestamp lengths differ')
    if len(motion) == n_frames:
        return motion
    if len(motion) > n_frames:
        raise ValueError(f'camera has {len(motion)} samples but neural has {n_frames}')
    # Camera timestamps are monotonic but use acquisition-clock units. Their
    # median interval defines neural-frame positions; gaps therefore leave the
    # appropriate integer slots in x. Rescale the small accumulated clock drift
    # so that the nominal complete recording spans exactly n_frames positions.
    intervals = np.diff(timestamps)
    dt = np.median(intervals)
    # Infer how many nominal camera periods elapsed at each interval. Rounding
    # intervals (rather than absolute positions) avoids accumulated clock drift.
    steps = np.maximum(1, np.rint(intervals / dt).astype(np.int64))
    x = np.concatenate(([0], np.cumsum(steps)))
    if x[-1] != n_frames - 1:
        raise ValueError(f'timestamp gaps imply {x[-1] + 1} frames, expected {n_frames}')
    return np.interp(np.arange(n_frames, dtype=np.float64), x, motion)


def mean_blocks(a, block=AVERAGE_FRAMES):
    n = a.shape[-1] // block * block
    return a[..., :n].reshape(*a.shape[:-1], n // block, block).mean(axis=-1)


def quintile_labels(x):
    # Internal percentile cut points; digitize returns exactly labels 0..4.
    edges = np.percentile(x, [20, 40, 60, 80])
    return np.digitize(x, edges, right=False).astype(np.int64), edges


def convert():
    subjects = sorted(p.name for p in ROOT.glob('jm*') if p.is_dir() and any(p.glob('20*')))
    neural, inputs, outputs = [], [], []
    subject_idx, brain_region_idx, session_info = [], [], []

    for subj_i, subject in enumerate(subjects):
        session_dirs = sorted(p for p in (ROOT / subject).glob('20*') if p.is_dir())
        for session_dir in session_dirs:
            p2 = session_dir / 'suite2p' / 'plane0'
            move = session_dir / 'move_deve'
            spks = np.load(p2 / 'spks.npy')
            iscell = np.load(p2 / 'iscell.npy')
            if spks.ndim != 2 or iscell.shape != (spks.shape[0], 2):
                raise ValueError(f'inconsistent Suite2p arrays in {session_dir}')
            keep = (iscell[:, 0] == 1) & (iscell[:, 1] >= 0.5)
            if not np.all(keep):
                spks = spks[keep]

            motion_raw = np.load(move / 'motion_energy_glob.npy')
            timestamps = np.load(move / 'tstamps.npy')
            motion = restore_motion(motion_raw, timestamps, spks.shape[1])

            # Identical paper processing for activity and behavior.
            activity_binned = mean_blocks(np.asarray(spks, dtype=np.float32))
            motion_binned = mean_blocks(motion).astype(np.float32)
            n_bins = min(activity_binned.shape[1], len(motion_binned))
            n_trials = n_bins // BINS_PER_TRIAL
            n_use = n_trials * BINS_PER_TRIAL
            if n_trials < 2:
                raise ValueError(f'fewer than two complete trials in {session_dir}')
            activity_binned = activity_binned[:, :n_use]
            motion_binned = motion_binned[:n_use]
            labels, edges = quintile_labels(motion_binned)
            # Mean elapsed time of each underlying group of ten frames.
            elapsed = (np.arange(n_use, dtype=np.float32) * AVERAGE_FRAMES +
                       (AVERAGE_FRAMES - 1) / 2) / FS

            ns, ins, outs = [], [], []
            for tr in range(n_trials):
                sl = slice(tr * BINS_PER_TRIAL, (tr + 1) * BINS_PER_TRIAL)
                ns.append(np.ascontiguousarray(activity_binned[:, sl], dtype=np.float32))
                ins.append(np.ascontiguousarray(elapsed[sl][None, :], dtype=np.float32))
                outs.append(np.ascontiguousarray(labels[sl][None, :], dtype=np.int64))
            neural.append(ns); inputs.append(ins); outputs.append(outs)
            subject_idx.append(subj_i)
            brain_region_idx.append(np.zeros(spks.shape[0], dtype=np.int64))
            session_info.append({
                'subject': subject, 'session': session_dir.name,
                'n_neurons': int(spks.shape[0]), 'n_trials': n_trials,
                'original_neural_frames': int(spks.shape[1]),
                'original_motion_frames': int(len(motion_raw)),
                'interpolated_motion_frames': int(spks.shape[1] - len(motion_raw)),
                'motion_quintile_edges': edges.tolist(),
            })
            print(subject, session_dir.name, spks.shape, '->', n_trials, 'trials', flush=True)

    data = {
        'neural': neural,
        'input': inputs,
        'output': outputs,
        'subjects': subjects,
        'subject_idx': np.asarray(subject_idx, dtype=np.int64),
        'brain_regions': ['barrel cortex'],
        'brain_region_idx': brain_region_idx,
        'input_names': ['time elapsed from session start (s)'],
        'output_names': ['motion energy quintile'],
        'output_values': [['quintile 1 (lowest)', 'quintile 2', 'quintile 3',
                           'quintile 4', 'quintile 5 (highest)']],
        'metadata': {
            'task_description': 'Decode spontaneous mouse motion-energy quintile from barrel-cortex neural activity.',
            'time_bin_size': BIN_SECONDS * 1000.0,
            'temporal_alignment_event': 'start of each consecutive 60-second session block',
            'off_start': 0.0,
            'off_end': float(TRIAL_SECONDS),
            'trial_duration_seconds': float(TRIAL_SECONDS),
            'native_sampling_rate_hz': FS,
            'temporal_averaging_frames': AVERAGE_FRAMES,
            'neural_signal': 'Suite2p deconvolved calcium activity (spks.npy), averaged over 10 frames',
            'motion_processing': 'global squared frame-difference energy; missing camera frames interpolated; averaged over 10 frames; per-session percentile bins',
            'session_info': session_info,
        },
    }
    with OUT.open('wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'Saved {OUT} ({OUT.stat().st_size / 2**20:.1f} MiB), {len(neural)} sessions')

if __name__ == '__main__':
    convert()
