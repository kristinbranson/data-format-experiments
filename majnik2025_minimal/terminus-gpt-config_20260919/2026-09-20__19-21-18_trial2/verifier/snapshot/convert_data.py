#!/usr/bin/env python3
"""Convert the Majnik et al. Track2p dataset for motion decoding.

Processing choices follow the paper and supplied data documentation:
* Every supplied subject/session is retained. Track2p's Suite2p exports already
  contain only cells tracked across every day of a subject. Their iscell scores
  exceed the paper's default 0.5 threshold, so no second ROI filter is applied.
* The paper used Suite2p-default baseline-corrected fluorescence. We form the
  pipeline's neuropil-corrected trace F - 0.7 Fneu and reproduce its maximin
  baseline (Gaussian sigma 10 frames, rolling min then max over 60 s).
* Camera frames occasionally dropped. Camera timestamps identify their exact
  locations, so motion energy is linearly interpolated to the regular 30 Hz
  two-photon frame grid rather than shifted or merely padded at the end.
* As in the paper's decoding analysis, neural and behavioral traces are averaged
  over non-overlapping groups of 10 timestamps (30 Hz -> 3 Hz).
* The requested categorical target is formed after averaging, using four
  per-session percentile boundaries to produce five motion-energy quintiles.
* Sessions divide exactly into complete, consecutive 60 s trials. Elapsed-time
  input remains relative to session start (rather than resetting each trial).
"""
from pathlib import Path
import pickle
import numpy as np
from scipy.ndimage import gaussian_filter1d, minimum_filter1d, maximum_filter1d

DATA_ROOT = Path('/app/data')
OUT = Path('/app/converted_data.pkl')
FS = 30.0
AVERAGE_FRAMES = 10
BIN_FS = FS / AVERAGE_FRAMES
TRIAL_SECONDS = 60
TRIAL_SAMPLES = int(TRIAL_SECONDS * BIN_FS)


def suite2p_baseline_corrected(session: Path) -> np.ndarray:
    """Paper's Suite2p-default neuropil and maximin baseline correction."""
    p = session / 'suite2p' / 'plane0'
    F = np.load(p / 'F.npy').astype(np.float32, copy=False)
    Fneu = np.load(p / 'Fneu.npy').astype(np.float32, copy=False)
    ops = np.load(p / 'ops.npy', allow_pickle=True).item()
    neucoeff = float(ops.get('neucoeff', 0.7))
    corrected = F - neucoeff * Fneu
    # Suite2p maximin: smooth, rolling minimum, rolling maximum, subtract.
    smooth = gaussian_filter1d(corrected, float(ops.get('sig_baseline', 10.0)),
                               axis=1, mode='reflect')
    window = int(float(ops.get('win_baseline', 60.0)) * float(ops.get('fs', FS)))
    base = minimum_filter1d(smooth, size=window, axis=1, mode='reflect')
    base = maximum_filter1d(base, size=window, axis=1, mode='reflect')
    corrected -= base
    return corrected.astype(np.float32, copy=False)


def aligned_motion(session: Path, nframes: int) -> np.ndarray:
    """Interpolate camera motion samples onto the regular imaging-frame grid."""
    p = session / 'move_deve'
    motion = np.load(p / 'motion_energy_glob.npy').astype(np.float64)
    stamps = np.load(p / 'tstamps.npy').astype(np.float64)
    if len(motion) != len(stamps):
        raise ValueError(f'motion/timestamp mismatch in {session}')
    # The timestamp clock has arbitrary units; its median camera-frame interval
    # defines one regular 30 Hz step and preserves the observed dropped-frame gaps.
    positive_dt = np.diff(stamps)
    step = np.median(positive_dt[positive_dt > 0])
    target = stamps[0] + np.arange(nframes, dtype=np.float64) * step
    return np.interp(target, stamps, motion, left=motion[0], right=motion[-1])


def average_ten(x: np.ndarray) -> np.ndarray:
    n = x.shape[-1] // AVERAGE_FRAMES
    x = x[..., :n * AVERAGE_FRAMES]
    return x.reshape(*x.shape[:-1], n, AVERAGE_FRAMES).mean(axis=-1)


def quintiles(x: np.ndarray) -> np.ndarray:
    edges = np.quantile(x, [0.2, 0.4, 0.6, 0.8])
    # right=False gives classes 0..4 and is equivalent to cutting at percentile
    # boundaries. The data have no problematic percentile ties.
    return np.searchsorted(edges, x, side='right').astype(np.int64)


def main() -> None:
    subjects = sorted(p.name for p in DATA_ROOT.iterdir() if p.is_dir() and p.name.startswith('jm'))
    neural, inputs, outputs = [], [], []
    subject_idx, region_idx, session_info = [], [], []

    for si, subject in enumerate(subjects):
        sessions = sorted(p for p in (DATA_ROOT / subject).iterdir() if p.is_dir())
        for session in sessions:
            print(f'Processing {subject}/{session.name}', flush=True)
            activity = suite2p_baseline_corrected(session)
            motion = aligned_motion(session, activity.shape[1])
            activity = average_ten(activity).astype(np.float32)
            motion = average_ten(motion)
            labels = quintiles(motion)
            elapsed = (np.arange(activity.shape[1], dtype=np.float32) / BIN_FS)

            ntrials = activity.shape[1] // TRIAL_SAMPLES
            usable = ntrials * TRIAL_SAMPLES
            activity, labels, elapsed = activity[:, :usable], labels[:usable], elapsed[:usable]
            neural.append([np.ascontiguousarray(x, dtype=np.float32)
                           for x in np.split(activity, ntrials, axis=1)])
            inputs.append([np.ascontiguousarray(x[None, :], dtype=np.float32)
                           for x in np.split(elapsed, ntrials)])
            outputs.append([np.ascontiguousarray(x[None, :], dtype=np.int64)
                            for x in np.split(labels, ntrials)])
            subject_idx.append(si)
            region_idx.append(np.zeros(activity.shape[0], dtype=np.int64))
            session_info.append({
                'subject': subject, 'session': session.name,
                'n_neurons': int(activity.shape[0]), 'n_trials': ntrials,
                'duration_seconds': float(usable / BIN_FS),
                'source_imaging_rate_hz': FS,
            })

    data = {
        'neural': neural,
        'input': inputs,
        'output': outputs,
        'subjects': subjects,
        'subject_idx': np.asarray(subject_idx, dtype=np.int64),
        'brain_regions': ['barrel cortex'],
        'brain_region_idx': region_idx,
        'input_names': ['elapsed time from session start (s)'],
        'output_names': ['motion energy quintile'],
        'output_values': [['lowest', 'low', 'middle', 'high', 'highest']],
        'metadata': {
            'task_description': 'Decode per-session motion-energy quintile from barrel-cortex calcium activity.',
            'time_bin_size': 1000.0 / BIN_FS,
            'temporal_alignment_event': 'session start; consecutive 60-second trial segmentation',
            'off_start': 0.0,
            'off_end': float(TRIAL_SECONDS),
            'source_frame_rate_hz': FS,
            'temporal_averaging_frames': AVERAGE_FRAMES,
            'trial_duration_seconds': TRIAL_SECONDS,
            'neural_processing': 'F - 0.7*Fneu; Suite2p maximin baseline subtraction; mean of 10 frames',
            'behavior_processing': 'timestamp interpolation to imaging frames; mean of 10 frames; per-session percentile bins',
            'session_info': session_info,
        },
    }
    with OUT.open('wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'Saved {len(neural)} sessions to {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)')


if __name__ == '__main__':
    main()
