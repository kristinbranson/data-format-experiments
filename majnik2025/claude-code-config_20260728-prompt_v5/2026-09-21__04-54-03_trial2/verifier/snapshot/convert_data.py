#!/usr/bin/env python3
"""
Convert Track2p longitudinal calcium imaging data to decoder-compatible format.

Usage:
    python -u convert_data.py <output_pickle> [--full|--sample] [--show-processing]
"""

import os
import sys
import time
import argparse
import pickle
import numpy as np
from scipy.ndimage import gaussian_filter1d, minimum_filter1d, maximum_filter1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_ROOT = '/app/data'
FRAME_RATE = 30  # Hz
BIN_SIZE = 10    # frames per bin
TRIAL_DURATION = 60  # seconds
N_BINS = 5       # output discretization bins
NEUCOEFF = 0.7   # neuropil coefficient
WIN_BASELINE = 60.0  # seconds
SIG_BASELINE = 10.0  # frames (sigma for gaussian smoothing)


def get_subjects_and_sessions():
    """Get all subjects and their session directories."""
    subjects = sorted([d for d in os.listdir(DATA_ROOT)
                      if os.path.isdir(os.path.join(DATA_ROOT, d))])
    result = []
    for subj in subjects:
        subj_dir = os.path.join(DATA_ROOT, subj)
        sessions = sorted([d for d in os.listdir(subj_dir)
                          if os.path.isdir(os.path.join(subj_dir, d))])
        for sess in sessions:
            result.append((subj, sess))
    return result


def compute_baseline_maximin(Fc, win_baseline=WIN_BASELINE, sig_baseline=SIG_BASELINE, fs=FRAME_RATE):
    """
    Compute maximin baseline following Suite2p's method.
    sig_baseline is in frames (as in Suite2p).
    win_baseline is in seconds.
    """
    win = int(win_baseline * fs)
    if win % 2 == 0:
        win += 1

    n_neurons, n_frames = Fc.shape
    Flow = np.zeros_like(Fc, dtype=np.float32)

    for i in range(n_neurons):
        trace = Fc[i].astype(np.float32)
        # Gaussian smoothing
        smoothed = gaussian_filter1d(trace, sig_baseline)
        # Running minimum
        smoothed = minimum_filter1d(smoothed, win)
        # Running maximum
        smoothed = maximum_filter1d(smoothed, win)
        Flow[i] = smoothed

    return Flow


def compute_dff(F, Fneu):
    """
    Compute dF/F using Suite2p default parameters.
    1. Neuropil correction: Fc = F - 0.7 * Fneu
    2. Maximin baseline estimation
    3. dF/F = (Fc - F0) / F0
    """
    Fc = F.astype(np.float32) - NEUCOEFF * Fneu.astype(np.float32)
    F0 = compute_baseline_maximin(Fc)
    # Avoid division by zero
    F0_safe = np.maximum(F0, 1e-6)
    dff = (Fc - F0) / F0_safe
    return dff


def interpolate_motion_energy(me, n_target_frames):
    """Interpolate motion energy to match neural frame count when frames are missing."""
    if len(me) == n_target_frames:
        return me
    # Linear interpolation
    x_orig = np.linspace(0, 1, len(me))
    x_target = np.linspace(0, 1, n_target_frames)
    me_interp = np.interp(x_target, x_orig, me)
    return me_interp


def bin_data(data, bin_size):
    """Bin data by averaging consecutive frames. data shape: (..., n_frames)."""
    n_frames = data.shape[-1]
    n_bins = n_frames // bin_size
    trimmed = data[..., :n_bins * bin_size]
    if data.ndim == 1:
        return trimmed.reshape(n_bins, bin_size).mean(axis=1)
    else:
        return trimmed.reshape(*data.shape[:-1], n_bins, bin_size).mean(axis=-1)


def discretize_motion_energy(me_binned_trials, n_bins=N_BINS):
    """
    Discretize motion energy into n_bins equal-percentile bins per session.
    Returns bin indices (0 to n_bins-1) and bin edges.
    """
    # Concatenate all trials to compute session-wide percentiles
    all_me = np.concatenate(me_binned_trials)

    # Compute percentile boundaries
    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(all_me, percentiles)

    # Make edges slightly wider to include all values
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    # Digitize each trial
    discretized = []
    for me_trial in me_binned_trials:
        binned = np.digitize(me_trial, bin_edges[1:-1])  # 0 to n_bins-1
        discretized.append(binned)

    return discretized, bin_edges


def process_session(subj, sess, show_processing=False, session_idx=0):
    """Process a single session and return trial data."""
    t0 = time.time()
    sess_dir = os.path.join(DATA_ROOT, subj, sess)
    s2p_dir = os.path.join(sess_dir, 'suite2p', 'plane0')
    move_dir = os.path.join(sess_dir, 'move_deve')

    # Load neural data
    F = np.load(os.path.join(s2p_dir, 'F.npy'))
    Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
    n_neurons, n_frames = F.shape

    # Load motion energy
    me = np.load(os.path.join(move_dir, 'motion_energy_glob.npy'))

    print(f"  {subj}/{sess}: {n_neurons} neurons, {n_frames} frames, ME={len(me)} frames", flush=True)

    # Interpolate motion energy if needed
    me = interpolate_motion_energy(me, n_frames)

    # Compute dF/F
    t1 = time.time()
    dff = compute_dff(F, Fneu)
    t_dff = time.time() - t1

    # Bin neural data and motion energy
    t1 = time.time()
    dff_binned = bin_data(dff, BIN_SIZE)
    me_binned = bin_data(me, BIN_SIZE)
    t_bin = time.time() - t1

    binned_rate = FRAME_RATE / BIN_SIZE  # 3 Hz
    timepoints_per_trial = int(TRIAL_DURATION * binned_rate)  # 180
    n_binned_frames = dff_binned.shape[1]
    n_trials = n_binned_frames // timepoints_per_trial

    # Split into trials
    neural_trials = []
    me_trials = []
    input_trials = []

    for t in range(n_trials):
        start = t * timepoints_per_trial
        end = start + timepoints_per_trial

        neural_trials.append(dff_binned[:, start:end].astype(np.float32))
        me_trials.append(me_binned[start:end])

        # Time input: elapsed seconds from session start
        time_seconds = (np.arange(start, end) / binned_rate).astype(np.float32)
        input_trials.append(time_seconds.reshape(1, -1))

    # Discretize motion energy per session
    me_discretized, bin_edges = discretize_motion_energy(me_trials, N_BINS)

    output_trials = [me_d.reshape(1, -1).astype(np.int64) for me_d in me_discretized]

    t_total = time.time() - t0
    print(f"    dF/F: {t_dff:.1f}s, binning: {t_bin:.1f}s, total: {t_total:.1f}s, "
          f"{n_trials} trials x {timepoints_per_trial} timepoints", flush=True)

    # Plotting for show-processing mode
    if show_processing:
        plot_processing(subj, sess, F, Fneu, dff, me, dff_binned, me_binned,
                       neural_trials, me_trials, me_discretized, bin_edges,
                       input_trials, session_idx)

    return neural_trials, input_trials, output_trials, n_neurons


def plot_processing(subj, sess, F, Fneu, dff, me_raw, dff_binned, me_binned,
                   neural_trials, me_trials, me_discretized, bin_edges,
                   input_trials, session_idx):
    """Plot processing steps for visual verification."""
    fig, axes = plt.subplots(6, 1, figsize=(16, 20))
    fig.suptitle(f'{subj}/{sess}', fontsize=14)

    neuron_idx = 0
    n_show = min(3000, F.shape[1])  # Show first 100 seconds at 30Hz

    # 1. Raw F and Fneu
    axes[0].plot(F[neuron_idx, :n_show], label='F', alpha=0.7)
    axes[0].plot(Fneu[neuron_idx, :n_show] * NEUCOEFF, label=f'Fneu*{NEUCOEFF}', alpha=0.7)
    axes[0].set_title('Raw fluorescence (neuron 0)')
    axes[0].legend()

    # 2. dF/F
    axes[1].plot(dff[neuron_idx, :n_show])
    axes[1].set_title('dF/F (neuron 0)')

    # 3. Binned dF/F (show same time window)
    n_show_binned = n_show // BIN_SIZE
    axes[2].plot(dff_binned[neuron_idx, :n_show_binned])
    axes[2].set_title(f'Binned dF/F (neuron 0, bin={BIN_SIZE} frames)')

    # 4. Motion energy raw and binned
    axes[3].plot(me_raw[:n_show], alpha=0.5, label='Raw ME')
    me_binned_x = np.arange(n_show_binned) * BIN_SIZE
    axes[3].plot(me_binned_x, me_binned[:n_show_binned], label='Binned ME')
    axes[3].set_title('Motion energy')
    axes[3].legend()

    # 5. Discretized ME for first trial
    if len(me_discretized) > 0:
        axes[4].plot(me_trials[0], label='Continuous ME')
        axes[4].plot(me_discretized[0], label='Discretized ME', alpha=0.7)
        axes[4].set_title('Trial 0: Motion energy discretization')
        axes[4].legend()

    # 6. Time input for first trial
    if len(input_trials) > 0:
        axes[5].plot(input_trials[0][0])
        axes[5].set_title('Trial 0: Time input (seconds)')
        axes[5].set_xlabel('Timepoint')

    plt.tight_layout()
    plt.savefig(f'/app/processing_{subj}_{sess}.png', dpi=100)
    plt.close()
    print(f"    Saved processing plot: processing_{subj}_{sess}.png")


def main():
    parser = argparse.ArgumentParser(description='Convert Track2p data to decoder format')
    parser.add_argument('output', type=str, help='Output pickle file path')
    parser.add_argument('--full', action='store_true', default=True, help='Process all sessions (default)')
    parser.add_argument('--sample', action='store_true', help='Process only 2 sessions for testing')
    parser.add_argument('--show-processing', action='store_true', help='Plot processing visualizations')
    args = parser.parse_args()

    t_start = time.time()

    all_sessions = get_subjects_and_sessions()
    print(f"Found {len(all_sessions)} total sessions across {len(set(s[0] for s in all_sessions))} subjects")

    if args.sample:
        # Pick 2 sessions from different subjects
        all_sessions = [all_sessions[0], all_sessions[-1]]
        print(f"Sample mode: processing {len(all_sessions)} sessions")

    # Organize data
    neural_all = []
    input_all = []
    output_all = []
    subject_idx_list = []
    brain_region_idx_all = []

    subjects_list = sorted(set(s[0] for s in all_sessions))
    subjects_map = {s: i for i, s in enumerate(subjects_list)}

    session_count = 0
    for subj, sess in all_sessions:
        t_sess = time.time()
        neural_trials, input_trials, output_trials, n_neurons = process_session(
            subj, sess,
            show_processing=args.show_processing and session_count < 2,
            session_idx=session_count
        )

        neural_all.append(neural_trials)
        input_all.append(input_trials)
        output_all.append(output_trials)
        subject_idx_list.append(subjects_map[subj])
        brain_region_idx_all.append(np.zeros(n_neurons, dtype=np.int64))

        session_count += 1

    # Build output structure
    bin_size_ms = (BIN_SIZE / FRAME_RATE) * 1000  # 333.33 ms

    output_bin_names = [f'bin_{i}' for i in range(N_BINS)]

    data = {
        'neural': neural_all,
        'input': input_all,
        'output': output_all,

        'subjects': subjects_list,
        'subject_idx': np.array(subject_idx_list, dtype=np.int64),

        'brain_regions': ['S1BF'],  # barrel cortex (primary somatosensory, barrel field)
        'brain_region_idx': brain_region_idx_all,

        'input_names': ['time_elapsed_s'],
        'output_names': ['motion_energy_bin'],
        'output_values': [output_bin_names],

        'metadata': {
            'task_description': 'Decode motion energy (5 quintile bins) from barrel cortex calcium imaging during spontaneous behavior in developing mouse pups',
            'time_bin_size': bin_size_ms,
            'temporal_alignment_event': 'Session start (beginning of recording)',
            'off_start': 0.0,
            'off_end': float(TRIAL_DURATION),
            'frame_rate': float(FRAME_RATE),
            'bin_size_frames': BIN_SIZE,
            'trial_duration_s': TRIAL_DURATION,
            'neural_signal': 'dF/F (Suite2p baseline-corrected, neucoeff=0.7, maximin baseline)',
            'session_info': {f'{subj}/{sess}': {'subject': subj, 'session': sess}
                           for subj, sess in all_sessions},
        }
    }

    # Save
    print(f"\nSaving to {args.output}...")
    with open(args.output, 'wb') as f:
        pickle.dump(data, f, protocol=4)

    file_size = os.path.getsize(args.output) / (1024**2)
    t_total = time.time() - t_start

    # Print summary
    print(f"\n=== Summary ===")
    print(f"Subjects: {len(subjects_list)} ({', '.join(subjects_list)})")
    print(f"Sessions: {len(neural_all)}")
    print(f"Trials per session: {[len(s) for s in neural_all]}")
    print(f"Neurons per session: {[s[0].shape[0] for s in neural_all]}")
    print(f"Timepoints per trial: {neural_all[0][0].shape[1]}")
    print(f"Time bin size: {bin_size_ms:.2f} ms")
    print(f"Output bins: {N_BINS}")
    print(f"File size: {file_size:.1f} MB")
    print(f"Total time: {t_total:.1f}s ({t_total/len(neural_all):.1f}s/session)")


if __name__ == '__main__':
    main()
