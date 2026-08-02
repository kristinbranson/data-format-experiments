"""
Convert Track2p longitudinal calcium imaging data to decoder format.

Reference: Majnik et al. 2025 - "Longitudinal tracking of neuronal activity from the
same cells in the developing brain using Track2p"

Processing follows the paper's methods:
- dF/F computed using Suite2p defaults (neuropil subtraction + maximin baseline)
- Neural and behavioral data binned by 10 frames (as in paper's decoding analysis)
- Sessions split into 2-minute blocks (trials) matching paper's cross-validation scheme
- Motion energy interpolated for missing video frames, then normalized and discretized
"""

import os
import sys
import numpy as np
import pickle
from suite2p.extraction.dcnv import preprocess
import torch


# ============================================================
# Suite2p default dF/F computation (maximin baseline)
# ============================================================

def compute_dff_suite2p(F, Fneu, fs=30.0, neucoeff=0.7,
                        win_baseline=60.0, sig_baseline=10.0):
    """
    Compute baseline-corrected fluorescence using Suite2p's default method.

    Paper: "We used baseline corrected fluorescence traces as our dF/F
    (using the default Suite2p parameters)"

    1. Neuropil correction: Fc = F - neucoeff * Fneu
    2. Suite2p preprocess (maximin): Gaussian smooth -> running min -> running max
    3. Returns Fc - baseline (as Suite2p's preprocess does)
    """
    # Neuropil correction
    Fc = (F - neucoeff * Fneu).astype(np.float32)

    # Use Suite2p's preprocess: returns F - baseline
    dff = preprocess(Fc.copy(), 'maximin', win_baseline, sig_baseline, fs,
                     device=torch.device('cpu'))

    return dff


# ============================================================
# Motion energy processing
# ============================================================

def interpolate_missing_frames(me, ifi, n_neural_frames):
    """
    Interpolate motion energy to match neural frame count when video frames are missing.

    Uses interframe intervals to detect where frames were dropped,
    then inserts interpolated values at those positions.
    """
    if len(me) == n_neural_frames:
        return me

    n_missing = n_neural_frames - len(me)
    if n_missing <= 0:
        # ME is longer or equal, just truncate
        return me[:n_neural_frames]

    # Detect gaps from interframe intervals
    median_ifi = np.median(ifi)
    threshold = median_ifi * 1.5
    gap_indices = np.where(ifi > threshold)[0]

    # Estimate missing frames per gap
    missing_per_gap = np.round(ifi[gap_indices] / median_ifi).astype(int) - 1

    # Build full-length ME by inserting interpolated values at gap locations
    result = np.zeros(n_neural_frames, dtype=float)
    src_idx = 0  # index into me
    dst_idx = 0  # index into result

    gap_pos = 0  # index into gap_indices

    for src_idx in range(len(me)):
        result[dst_idx] = me[src_idx]
        dst_idx += 1

        # Check if there's a gap after this frame
        if gap_pos < len(gap_indices) and src_idx == gap_indices[gap_pos]:
            n_insert = missing_per_gap[gap_pos]
            # Interpolate between current and next frame
            if src_idx + 1 < len(me):
                for k in range(n_insert):
                    alpha = (k + 1) / (n_insert + 1)
                    if dst_idx < n_neural_frames:
                        result[dst_idx] = me[src_idx] * (1 - alpha) + me[src_idx + 1] * alpha
                        dst_idx += 1
            else:
                for k in range(n_insert):
                    if dst_idx < n_neural_frames:
                        result[dst_idx] = me[src_idx]
                        dst_idx += 1
            gap_pos += 1

    # If we still haven't filled all frames, pad with last value
    while dst_idx < n_neural_frames:
        result[dst_idx] = result[dst_idx - 1]
        dst_idx += 1

    return result[:n_neural_frames]


def bin_data(data, bin_size):
    """
    Bin data by averaging consecutive frames.

    For 1D: (n_time,) -> (n_bins,)
    For 2D: (n_neurons, n_time) -> (n_neurons, n_bins)
    """
    if data.ndim == 1:
        n = len(data)
        n_bins = n // bin_size
        trimmed = data[:n_bins * bin_size]
        return trimmed.reshape(n_bins, bin_size).mean(axis=1)
    elif data.ndim == 2:
        n_neurons, n_time = data.shape
        n_bins = n_time // bin_size
        trimmed = data[:, :n_bins * bin_size]
        return trimmed.reshape(n_neurons, n_bins, bin_size).mean(axis=2)


def discretize_percentile_bins(values, n_bins=5):
    """
    Discretize values into n_bins equal-percentile bins.
    Returns bin indices (0 to n_bins-1) and bin edges.
    """
    percentiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(values, percentiles)
    # Make edges strictly increasing to handle ties
    edges[-1] = edges[-1] + 1e-10
    bins = np.digitize(values, edges[1:-1])  # 0 to n_bins-1
    return bins, edges


# ============================================================
# Main conversion
# ============================================================

def convert_data(data_dir, output_path, sample_output_path=None):
    """Convert Track2p data to decoder format."""

    BIN_SIZE = 10  # frames per bin (paper: "averaging in bins of 10 consecutive timestamps")
    FS = 30.0  # imaging rate in Hz
    TRIAL_DURATION_S = 120  # 2-minute blocks (paper: "splits were done on consecutive 2 minute blocks")
    TRIAL_BINS = int(TRIAL_DURATION_S * FS / BIN_SIZE)  # 360 bins per trial
    N_OUTPUT_BINS = 5  # discretize into 5 equal-percentile bins

    time_bin_size_ms = (BIN_SIZE / FS) * 1000  # ~333.33 ms

    mice = sorted([d for d in os.listdir(data_dir)
                   if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('jm')])

    print(f"Found {len(mice)} mice: {mice}")

    all_neural = []
    all_input = []
    all_output = []
    all_subject_idx = []
    all_brain_region_idx = []

    subjects = mice

    for mouse_idx, mouse in enumerate(mice):
        mouse_dir = os.path.join(data_dir, mouse)
        sessions = sorted([d for d in os.listdir(mouse_dir)
                          if os.path.isdir(os.path.join(mouse_dir, d))])

        print(f"\n{'='*60}")
        print(f"Processing {mouse} ({len(sessions)} sessions)")

        for sess_name in sessions:
            sess_dir = os.path.join(mouse_dir, sess_name)
            s2p_dir = os.path.join(sess_dir, 'suite2p', 'plane0')
            move_dir = os.path.join(sess_dir, 'move_deve')

            # Load neural data
            F = np.load(os.path.join(s2p_dir, 'F.npy'))
            Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
            n_neurons, n_frames = F.shape

            # Load motion energy
            me_raw = np.load(os.path.join(move_dir, 'motion_energy_glob.npy')).astype(float)
            ifi = np.load(os.path.join(move_dir, 'interframe_int.npy'))

            print(f"  {sess_name}: {n_neurons} neurons, {n_frames} frames, ME={len(me_raw)}")

            # Interpolate missing video frames
            me = interpolate_missing_frames(me_raw, ifi, n_frames)

            # Compute dF/F (Suite2p default method)
            dff = compute_dff_suite2p(F, Fneu, fs=FS)

            # Bin neural data and motion energy by 10 frames
            dff_binned = bin_data(dff, BIN_SIZE)
            me_binned = bin_data(me, BIN_SIZE)

            n_total_bins = dff_binned.shape[1]
            n_trials = n_total_bins // TRIAL_BINS

            print(f"    Binned: {dff_binned.shape}, {n_total_bins} bins -> {n_trials} trials of {TRIAL_BINS} bins")

            # Discretize motion energy into 5 equal-percentile bins (per session)
            me_discrete, me_edges = discretize_percentile_bins(me_binned, N_OUTPUT_BINS)

            # Split into trials
            session_neural = []
            session_input = []
            session_output = []

            for t in range(n_trials):
                start = t * TRIAL_BINS
                end = (t + 1) * TRIAL_BINS

                # Neural: (n_neurons, n_timepoints)
                trial_neural = dff_binned[:, start:end]

                # Input: time elapsed from beginning of experiment (in seconds)
                # Time of each bin center from session start
                bin_times = (np.arange(start, end) + 0.5) * (BIN_SIZE / FS)
                trial_input = bin_times.reshape(1, -1)  # (1, n_timepoints)

                # Output: discretized motion energy (0-4)
                trial_output = me_discrete[start:end].reshape(1, -1)  # (1, n_timepoints)

                session_neural.append(trial_neural.astype(np.float32))
                session_input.append(trial_input.astype(np.float32))
                session_output.append(trial_output.astype(np.int64))

            all_neural.append(session_neural)
            all_input.append(session_input)
            all_output.append(session_output)
            all_subject_idx.append(mouse_idx)
            all_brain_region_idx.append(np.zeros(n_neurons, dtype=int))

    # Build output values labels
    output_values = [
        [f'bin_{i}' for i in range(N_OUTPUT_BINS)]
    ]

    data = {
        'neural': all_neural,
        'input': all_input,
        'output': all_output,

        'subjects': subjects,
        'subject_idx': np.array(all_subject_idx, dtype=int),

        'brain_regions': ['barrel_cortex'],
        'brain_region_idx': all_brain_region_idx,

        'input_names': ['time_s'],
        'output_names': ['motion_energy'],
        'output_values': output_values,

        'metadata': {
            'task_description': 'Decode motion energy (5 percentile bins) from barrel cortex neural activity during spontaneous behavior in developing mouse pups (P7-P14)',
            'time_bin_size': time_bin_size_ms,
            'temporal_alignment_event': 'Start of imaging session',
            'off_start': 0.0,
            'off_end': None,
            'session_info': {
                'imaging_rate_hz': FS,
                'bin_size_frames': BIN_SIZE,
                'trial_duration_s': TRIAL_DURATION_S,
                'trial_bins': TRIAL_BINS,
                'n_output_bins': N_OUTPUT_BINS,
                'neural_signal': 'dF/F (Suite2p default: neuropil subtraction + maximin baseline)',
                'brain_region': 'barrel cortex layer 2/3',
                'fov_size_um': '720x720',
                'depth_um': '100-200',
            }
        }
    }

    # Print summary statistics
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Subjects: {subjects}")
    print(f"Total sessions: {len(all_neural)}")
    print(f"Sessions per subject: {[np.sum(np.array(all_subject_idx) == i) for i in range(len(subjects))]}")

    total_neurons = []
    for i, mouse in enumerate(subjects):
        sess_mask = np.array(all_subject_idx) == i
        sess_indices = np.where(sess_mask)[0]
        if len(sess_indices) > 0:
            nn = all_neural[sess_indices[0]][0].shape[0]
            total_neurons.append(nn)
            print(f"  {mouse}: {nn} tracked neurons, {len(all_neural[sess_indices[0]])} trials/session")

    print(f"\nMean neurons per mouse: {np.mean(total_neurons):.0f} (± {np.std(total_neurons):.0f} std)")
    print(f"Time bin size: {time_bin_size_ms:.2f} ms")
    print(f"Trial duration: {TRIAL_DURATION_S}s ({TRIAL_BINS} bins)")

    # Save full dataset
    print(f"\nSaving full dataset to {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved ({os.path.getsize(output_path) / 1e6:.1f} MB)")

    # Save sample dataset (subset of 2 mice)
    if sample_output_path:
        sample_mice = [0, 1]  # first two mice
        sample_sessions = []
        for i in range(len(all_neural)):
            if all_subject_idx[i] in sample_mice:
                sample_sessions.append(i)

        sample_data = {
            'neural': [all_neural[i] for i in sample_sessions],
            'input': [all_input[i] for i in sample_sessions],
            'output': [all_output[i] for i in sample_sessions],
            'subjects': [subjects[i] for i in sample_mice],
            'subject_idx': np.array([sample_mice.index(all_subject_idx[i]) for i in sample_sessions], dtype=int),
            'brain_regions': data['brain_regions'],
            'brain_region_idx': [all_brain_region_idx[i] for i in sample_sessions],
            'input_names': data['input_names'],
            'output_names': data['output_names'],
            'output_values': data['output_values'],
            'metadata': data['metadata'],
        }

        print(f"Saving sample dataset to {sample_output_path}...")
        with open(sample_output_path, 'wb') as f:
            pickle.dump(sample_data, f)
        print(f"Saved ({os.path.getsize(sample_output_path) / 1e6:.1f} MB)")

    return data


if __name__ == '__main__':
    data_dir = '/app/data'
    output_path = '/app/converted_data.pkl'
    sample_output_path = '/app/sample_data.pkl'

    data = convert_data(data_dir, output_path, sample_output_path)
