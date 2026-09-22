#!/usr/bin/env python3
"""Convert the Majnik et al. Track2p developmental barrel-cortex dataset.

Processing choices follow the supplied paper/methods and data README:
* use every supplied mouse/session and the supplied Track2p-curated population
  (these are cells detected at Suite2p probability > .5 and tracked on all days);
* calculate neuropil-corrected dF/F with each session's Suite2p parameters;
* interpolate motion energy only at camera frames identified as missing by its
  timestamps, onto the complete imaging-frame index;
* average both dF/F and behavior over 10 frames, as in the paper's decoding;
* split each recording into consecutive, complete 60 s trials;
* assign motion energy to session-specific equal-percentile (quintile) bins.
"""
from pathlib import Path
import pickle
import numpy as np
from scipy.ndimage import gaussian_filter1d, minimum_filter1d, maximum_filter1d

ROOT = Path('/app/data')
OUT = Path('/app/converted_data.pkl')
AVG_FRAMES = 10
TRIAL_SECONDS = 60


def maximin_dff(F, Fneu, ops):
    """Suite2p-style maximin-baseline dF/F from raw supplied traces."""
    x = np.asarray(F, dtype=np.float32) - np.float32(ops.get('neucoeff', .7)) * np.asarray(Fneu, dtype=np.float32)
    # Suite2p defaults: Gaussian sigma is in frames; min then max over window.
    smooth = gaussian_filter1d(x, float(ops.get('sig_baseline', 10.0)), axis=1, mode='reflect')
    win = int(round(float(ops.get('win_baseline', 60.0)) * float(ops['fs'])))
    base = minimum_filter1d(smooth, size=win, axis=1, mode='reflect')
    base = maximum_filter1d(base, size=win, axis=1, mode='reflect')
    # The paper calls the baseline-corrected fluorescence dF/F. Avoid only
    # numerical zero denominators; normal session baselines are positive.
    eps = np.finfo(np.float32).eps
    denom = np.where(np.abs(base) > eps, base, np.where(base < 0, -eps, eps))
    x = (x - base) / denom
    return x.astype(np.float32, copy=False)


def aligned_motion(session, nframes):
    """Insert/interpolate camera frames missing according to timestamps."""
    md = session / 'move_deve'
    motion = np.load(md / 'motion_energy_glob.npy').astype(np.float64)
    stamps = np.load(md / 'tstamps.npy').astype(np.float64)
    if len(motion) != len(stamps):
        raise ValueError(f'motion/timestamp mismatch in {session}')
    if len(motion) == nframes and not np.any(np.diff(stamps) > 1.5 * np.median(np.diff(stamps))):
        return motion
    # Timestamp increments are nominally constant; gaps are integer multiples
    # and identify omitted camera-frame slots (as described by data README).
    dt = np.median(np.diff(stamps))
    slots = np.rint((stamps - stamps[0]) / dt).astype(np.int64)
    # Tiny clock drift can shift the last rounded slot. Scale to the known
    # imaging index range while preserving detected gaps.
    if slots[-1] not in (nframes - 1, nframes):
        slots = np.rint((stamps - stamps[0]) * (nframes - 1) / (stamps[-1] - stamps[0])).astype(np.int64)
    slots = np.clip(slots, 0, nframes - 1)
    unique, idx = np.unique(slots, return_index=True)
    return np.interp(np.arange(nframes), unique, motion[idx])


def quintile_labels(x):
    """Five session-wise equal-percentile bins, labels 0..4."""
    edges = np.quantile(x, [.2, .4, .6, .8])
    return np.searchsorted(edges, x, side='right').astype(np.int64)


def main():
    subjects = sorted(p.name for p in ROOT.glob('jm*') if p.is_dir())
    neural, inputs, outputs = [], [], []
    subject_idx, region_idx, session_info = [], [], []

    for si, subject in enumerate(subjects):
        sessions = sorted(p for p in (ROOT / subject).iterdir() if p.is_dir())
        for session in sessions:
            plane = session / 'suite2p' / 'plane0'
            F = np.load(plane / 'F.npy', mmap_mode='r')
            Fneu = np.load(plane / 'Fneu.npy', mmap_mode='r')
            ops = np.load(plane / 'ops.npy', allow_pickle=True).item()
            fs = float(ops['fs'])
            if fs != 30 or F.shape != Fneu.shape:
                raise ValueError(f'unexpected imaging data in {session}: {F.shape}, fs={fs}')
            dff = maximin_dff(F, Fneu, ops)
            motion = aligned_motion(session, F.shape[1])

            # Paper decoding averages 10 consecutive timestamps.
            usable = (F.shape[1] // AVG_FRAMES) * AVG_FRAMES
            dff = dff[:, :usable].reshape(F.shape[0], -1, AVG_FRAMES).mean(2, dtype=np.float32)
            motion = motion[:usable].reshape(-1, AVG_FRAMES).mean(1)
            labels = quintile_labels(motion)
            out_fs = fs / AVG_FRAMES
            trial_bins = int(round(TRIAL_SECONDS * out_fs))
            ntrials = dff.shape[1] // trial_bins

            ns, ins, outs = [], [], []
            elapsed = np.arange(dff.shape[1], dtype=np.float32) / np.float32(out_fs)
            for tr in range(ntrials):
                a, b = tr * trial_bins, (tr + 1) * trial_bins
                ns.append(np.ascontiguousarray(dff[:, a:b], dtype=np.float32))
                ins.append(elapsed[None, a:b].copy())
                outs.append(labels[None, a:b].copy())
            neural.append(ns); inputs.append(ins); outputs.append(outs)
            subject_idx.append(si)
            region_idx.append(np.zeros(F.shape[0], dtype=np.int64))
            session_info.append({
                'subject': subject, 'session': session.name,
                'source_frames': int(F.shape[1]), 'motion_frames': int(np.load(session/'move_deve'/'motion_energy_glob.npy', mmap_mode='r').shape[0]),
                'n_neurons': int(F.shape[0]), 'n_trials': ntrials,
                'suite2p_fs_hz': fs
            })
            print(subject, session.name, F.shape[0], 'neurons,', ntrials, 'trials', flush=True)

    data = {
        'neural': neural, 'input': inputs, 'output': outputs,
        'subjects': subjects, 'subject_idx': np.asarray(subject_idx, dtype=np.int64),
        'brain_regions': ['barrel cortex L2/3'], 'brain_region_idx': region_idx,
        'input_names': ['time elapsed from session start (s)'],
        'output_names': ['motion energy quintile'],
        'output_values': [[
            'lowest motion (0-20%)', 'low motion (20-40%)',
            'medium motion (40-60%)', 'high motion (60-80%)',
            'highest motion (80-100%)']],
        'metadata': {
            'task_description': 'Decode session-wise motion-energy quintile from barrel-cortex calcium activity.',
            'time_bin_size': 1000.0 * AVG_FRAMES / 30.0,
            'temporal_alignment_event': 'start of each consecutive 60-second recording block',
            'off_start': 0.0, 'off_end': 60.0,
            'neural_measure': 'neuropil-corrected, Suite2p maximin-baseline dF/F averaged over 10 frames',
            'motion_processing': 'global squared frame-difference energy; missing camera frames linearly interpolated; 10-frame means; session quintiles',
            'trial_definition': 'consecutive non-overlapping 60-second blocks',
            'session_info': session_info,
            'source': 'Majnik et al. (2025), Track2p developmental barrel-cortex dataset'
        }
    }
    with OUT.open('wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'Wrote {OUT}: {len(neural)} sessions, {sum(map(len, neural))} trials, {OUT.stat().st_size/1e9:.3f} GB')

if __name__ == '__main__':
    main()
