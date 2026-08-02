#!/usr/bin/env python3
"""
Convert Track2p longitudinal calcium imaging data to decoder format.

Reference: Majnik et al. 2025, "Longitudinal tracking of neuronal activity from
the same cells in the developing brain using Track2p", eLife.

Data: 6 mice imaged daily in barrel cortex (layer 2/3) during P7-P14.
Neural: dF/F computed from Suite2p outputs (F, Fneu) with standard baseline correction.
Behavioral: Motion energy from videography, normalized and discretized into 5 bins.

Processing follows the paper's methods:
- dF/F: neuropil-corrected (F - 0.7*Fneu), baseline = Suite2p default maximin over 60s window
- Binning: average over 10 consecutive frames (30Hz -> 3Hz, ~333ms bins)
- Trials: consecutive 2-minute blocks of the recording
- Motion energy: squared pixel-wise frame differences, normalized, 5 equal-percentile bins
"""

import os
import sys
import numpy as np
import pickle
from scipy.ndimage import minimum_filter1d, maximum_filter1d, gaussian_filter1d

# ============ PARAMETERS ============
DATA_DIR = 'data'
OUTPUT_FILE = 'converted_data.pkl'
SAMPLE_OUTPUT_FILE = 'sample_data.pkl'

MICE = ['jm031', 'jm032', 'jm038', 'jm039', 'jm040', 'jm046']

BIN_SIZE = 10           # temporal binning factor (10 frames averaged)
FS = 30                 # imaging frame rate (Hz)
NEUCOEFF = 0.7          # neuropil correction coefficient (Suite2p default)
WIN_BASELINE_SEC = 60   # baseline window in seconds
PRCTILE_BASELINE = 8    # baseline percentile (Suite2p default)
TRIAL_DURATION_SEC = 120  # 2-minute trial blocks (as in paper's decoding)
N_OUTPUT_BINS = 5       # number of motion energy bins (quintiles)

# Derived
BIN_DURATION_MS = (BIN_SIZE / FS) * 1000  # ~333.33 ms
TRIAL_BINS = int(TRIAL_DURATION_SEC * FS / BIN_SIZE)  # 360 bins per trial


def get_sessions(mouse_dir):
    """Get sorted session directories for a mouse."""
    sessions = [d for d in os.listdir(mouse_dir)
                if os.path.isdir(os.path.join(mouse_dir, d)) and d[0] == '2']
    sessions.sort()
    return sessions


def compute_dff(F, Fneu):
    """
    Compute dF/F using Suite2p default parameters.
    1. Neuropil correction: Fc = F - 0.7 * Fneu
    2. Baseline: Suite2p default 'maximin' method
       - Gaussian smooth, then min filter, then max filter over 60s window
    3. dF/F = (Fc - F0) / F0
    """
    Fc = F - NEUCOEFF * Fneu
    win = int(WIN_BASELINE_SEC * FS)
    # Suite2p default 'maximin' baseline
    Flow = gaussian_filter1d(Fc.astype(np.float64), sigma=win / 6, axis=1)
    Flow = minimum_filter1d(Flow, size=win, axis=1)
    F0 = maximum_filter1d(Flow, size=win, axis=1)
    # Avoid division by zero/near-zero
    F0_safe = np.where(np.abs(F0) < 1e-6, 1e-6, F0)
    dff = (Fc - F0) / F0_safe
    return dff


def align_motion_energy(me, interframe_int, n_neural_frames):
    """
    Align motion energy to neural frames by detecting and interpolating dropped video frames.
    Uses interframe_int to detect gaps (> 1.5x median interval = dropped frame).
    """
    if len(me) == n_neural_frames:
        return me.astype(np.float64)

    median_ifi = np.median(interframe_int)

    # Build aligned array by inserting NaN at dropped frame locations
    aligned = np.full(n_neural_frames, np.nan)
    neural_idx = 0
    for i in range(len(me)):
        if neural_idx < n_neural_frames:
            aligned[neural_idx] = me[i]
        neural_idx += 1
        # Check if next interval indicates dropped frames
        if i < len(interframe_int):
            n_dropped = int(np.round(interframe_int[i] / median_ifi - 1))
            neural_idx += n_dropped

    # Interpolate NaN values
    nans = np.isnan(aligned)
    if np.any(nans) and not np.all(nans):
        x = np.arange(n_neural_frames)
        aligned[nans] = np.interp(x[nans], x[~nans], aligned[~nans])

    return aligned


def bin_array(data, bin_size):
    """Average data over consecutive bins along the last axis."""
    if data.ndim == 1:
        n_bins = len(data) // bin_size
        return data[:n_bins * bin_size].reshape(n_bins, bin_size).mean(axis=1)
    else:
        n_bins = data.shape[1] // bin_size
        return data[:, :n_bins * bin_size].reshape(data.shape[0], n_bins, bin_size).mean(axis=2)


def split_into_trials(data, trial_length):
    """Split data into non-overlapping trials of fixed length along last axis."""
    if data.ndim == 1:
        n_trials = len(data) // trial_length
        return [data[i*trial_length:(i+1)*trial_length] for i in range(n_trials)]
    else:
        n_trials = data.shape[1] // trial_length
        return [data[:, i*trial_length:(i+1)*trial_length] for i in range(n_trials)]


def convert_data(data_dir=DATA_DIR, output_file=OUTPUT_FILE, sample_output_file=SAMPLE_OUTPUT_FILE,
                 sample_only=False):
    """Main conversion function."""

    print("=" * 60)
    print("Converting Track2p data to decoder format")
    print("=" * 60)

    neural_all = []
    input_all = []
    output_all_raw = []  # store raw ME values before discretization
    subjects = MICE[:]
    subject_idx = []
    brain_region_idx_all = []

    # First pass: load, process, bin all data
    all_me_values = []  # collect all binned ME for global percentile computation

    for mouse_i, mouse in enumerate(MICE):
        mouse_dir = os.path.join(data_dir, mouse)
        sessions = get_sessions(mouse_dir)
        print(f"\n--- Mouse {mouse} ({len(sessions)} sessions) ---")

        for sess_name in sessions:
            sess_dir = os.path.join(mouse_dir, sess_name)
            suite2p_dir = os.path.join(sess_dir, 'suite2p', 'plane0')
            move_dir = os.path.join(sess_dir, 'move_deve')

            # Load neural data
            F = np.load(os.path.join(suite2p_dir, 'F.npy'))
            Fneu = np.load(os.path.join(suite2p_dir, 'Fneu.npy'))
            n_neurons, n_frames = F.shape

            # Compute dF/F
            dff = compute_dff(F, Fneu)

            # Load behavioral data
            me_raw = np.load(os.path.join(move_dir, 'motion_energy_glob.npy')).astype(np.float64)
            ifi = np.load(os.path.join(move_dir, 'interframe_int.npy'))

            # Align ME to neural frames (handle dropped video frames)
            me_aligned = align_motion_energy(me_raw, ifi, n_frames)

            # Bin both neural and ME by BIN_SIZE frames
            dff_binned = bin_array(dff, BIN_SIZE)
            me_binned = bin_array(me_aligned, BIN_SIZE)

            n_bins = dff_binned.shape[1]
            n_trials = n_bins // TRIAL_BINS

            print(f"  {sess_name}: {n_neurons} neurons, {n_frames} frames -> "
                  f"{n_bins} bins -> {n_trials} trials of {TRIAL_BINS} bins")

            # Split into trials
            neural_trials = split_into_trials(dff_binned, TRIAL_BINS)
            me_trials = split_into_trials(me_binned, TRIAL_BINS)

            # Create time input for each trial (time elapsed from start of session, in seconds)
            input_trials = []
            for t_i in range(n_trials):
                start_sec = t_i * TRIAL_DURATION_SEC
                time_bins = np.linspace(
                    start_sec + BIN_DURATION_MS / 2000,  # center of first bin
                    start_sec + TRIAL_DURATION_SEC - BIN_DURATION_MS / 2000,  # center of last bin
                    TRIAL_BINS
                )
                input_trials.append(time_bins.reshape(1, -1).astype(np.float32))

            neural_all.append(neural_trials)
            input_all.append(input_trials)
            output_all_raw.append(me_trials)
            subject_idx.append(mouse_i)
            brain_region_idx_all.append(np.zeros(n_neurons, dtype=np.int64))

            # Collect ME values for global percentile computation
            for mt in me_trials:
                all_me_values.append(mt)

    # Compute global percentile bin edges for motion energy
    all_me_concat = np.concatenate(all_me_values)
    percentiles = np.linspace(0, 100, N_OUTPUT_BINS + 1)
    bin_edges = np.percentile(all_me_concat, percentiles)
    print(f"\nMotion energy percentile bin edges: {bin_edges}")
    print(f"Motion energy stats: min={all_me_concat.min():.2f}, "
          f"max={all_me_concat.max():.2f}, mean={all_me_concat.mean():.2f}")

    # Discretize ME into bins (0 to N_OUTPUT_BINS-1)
    output_all = []
    for session_me_trials in output_all_raw:
        session_output = []
        for me_trial in session_me_trials:
            # np.digitize with bin_edges gives 1-based indices, subtract 1 for 0-based
            binned = np.digitize(me_trial, bin_edges[1:-1])  # 0 to N_OUTPUT_BINS-1
            session_output.append(binned.reshape(1, -1).astype(np.int64))
        output_all.append(session_output)

    # Verify bin distribution
    all_binned = np.concatenate([np.concatenate([t.flatten() for t in s]) for s in output_all])
    for b in range(N_OUTPUT_BINS):
        frac = np.mean(all_binned == b)
        print(f"  Bin {b}: {frac:.3f} ({frac*100:.1f}%)")

    # Sanity checks
    print(f"\n{'='*60}")
    print("SANITY CHECKS")
    print(f"{'='*60}")

    n_sessions = len(neural_all)
    print(f"Total sessions: {n_sessions}")
    print(f"Total mice: {len(subjects)}")

    neurons_per_mouse = {}
    sessions_per_mouse = {}
    for i, mouse_idx in enumerate(subject_idx):
        mouse = subjects[mouse_idx]
        n_neur = neural_all[i][0].shape[0]
        neurons_per_mouse.setdefault(mouse, []).append(n_neur)
        sessions_per_mouse[mouse] = sessions_per_mouse.get(mouse, 0) + 1

    print("\nNeurons per mouse (should be consistent across sessions):")
    for mouse, counts in neurons_per_mouse.items():
        consistent = "OK" if len(set(counts)) == 1 else "MISMATCH!"
        print(f"  {mouse}: {counts[0]} neurons ({consistent})")

    all_neuron_counts = [counts[0] for counts in neurons_per_mouse.values()]
    mean_neurons = np.mean(all_neuron_counts)
    std_neurons = np.std(all_neuron_counts, ddof=1)
    print(f"\nMean neurons per mouse: {mean_neurons:.0f} +/- {std_neurons:.0f}")
    print(f"Paper reports: 526 +/- 190")

    print(f"\nSessions per mouse:")
    for mouse, count in sessions_per_mouse.items():
        print(f"  {mouse}: {count}")

    # Check for NaN/Inf in neural data
    nan_count = sum(np.any(np.isnan(t)) for s in neural_all for t in s)
    inf_count = sum(np.any(np.isinf(t)) for s in neural_all for t in s)
    print(f"\nTrials with NaN: {nan_count}, with Inf: {inf_count}")

    # Build final data dict
    subject_idx_arr = np.array(subject_idx, dtype=np.int64)

    output_bin_names = [f"Q{i+1}" for i in range(N_OUTPUT_BINS)]

    data = {
        'neural': neural_all,
        'input': input_all,
        'output': output_all,

        'subjects': subjects,
        'subject_idx': subject_idx_arr,

        'brain_regions': ['barrel_cortex'],
        'brain_region_idx': brain_region_idx_all,

        'input_names': ['time_elapsed_s'],
        'output_names': ['motion_energy_bin'],
        'output_values': [output_bin_names],

        'metadata': {
            'task_description': 'Decode motion energy (5 equal-percentile bins) from barrel cortex calcium imaging during spontaneous behavior in developing mice (P7-P14)',
            'time_bin_size': BIN_DURATION_MS,
            'temporal_alignment_event': 'Start of recording session',
            'off_start': 0.0,
            'off_end': None,
            'imaging_rate_hz': FS,
            'bin_size_frames': BIN_SIZE,
            'trial_duration_sec': TRIAL_DURATION_SEC,
            'neuropil_coefficient': NEUCOEFF,
            'baseline_percentile': PRCTILE_BASELINE,
            'baseline_window_sec': WIN_BASELINE_SEC,
            'n_output_bins': N_OUTPUT_BINS,
            'motion_energy_bin_edges': bin_edges.tolist(),
            'session_info': {
                mouse: {
                    'sessions': get_sessions(os.path.join(data_dir, mouse)),
                    'n_neurons': neurons_per_mouse[mouse][0],
                    'n_sessions': sessions_per_mouse[mouse]
                }
                for mouse in MICE
            }
        }
    }

    # Convert neural data to float32
    for i in range(len(data['neural'])):
        data['neural'][i] = [t.astype(np.float32) for t in data['neural'][i]]

    # Save full dataset
    if not sample_only:
        print(f"\nSaving full dataset to {output_file}...")
        with open(output_file, 'wb') as f:
            pickle.dump(data, f)
        print(f"  File size: {os.path.getsize(output_file) / 1e6:.1f} MB")

    # Create sample dataset (first 2 sessions)
    n_sample = 2
    sample_data = {
        'neural': data['neural'][:n_sample],
        'input': data['input'][:n_sample],
        'output': data['output'][:n_sample],
        'subjects': data['subjects'],
        'subject_idx': data['subject_idx'][:n_sample],
        'brain_regions': data['brain_regions'],
        'brain_region_idx': data['brain_region_idx'][:n_sample],
        'input_names': data['input_names'],
        'output_names': data['output_names'],
        'output_values': data['output_values'],
        'metadata': data['metadata'],
    }

    print(f"Saving sample dataset ({n_sample} sessions) to {sample_output_file}...")
    with open(sample_output_file, 'wb') as f:
        pickle.dump(sample_data, f)
    print(f"  File size: {os.path.getsize(sample_output_file) / 1e6:.1f} MB")

    print(f"\n{'='*60}")
    print("Conversion complete!")
    print(f"{'='*60}")

    return data


if __name__ == '__main__':
    sample_only = '--sample-only' in sys.argv
    data = convert_data(sample_only=sample_only)
