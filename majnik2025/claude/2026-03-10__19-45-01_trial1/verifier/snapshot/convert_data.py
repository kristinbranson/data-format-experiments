#!/usr/bin/env python3
"""
Convert Track2p barrel cortex calcium imaging data to decoder-compatible format.

Usage:
    python -u convert_data.py <outpicklefile> [--sample] [--full] [--show-processing]

Reference: Majnik et al. 2025 - "Longitudinal tracking of neuronal activity from
the same cells in the developing brain using Track2p"
"""

import os
import sys
import time
import argparse
import pickle
import numpy as np
from scipy.ndimage import uniform_filter1d

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# =============================================================================
# Constants
# =============================================================================
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
FS = 30.0  # imaging frame rate (Hz)
BIN_SIZE = 10  # number of frames per bin (paper: "bins of 10 consecutive timestamps")
TRIAL_DURATION_SEC = 120  # 2 minutes per trial (paper: "consecutive 2 minute blocks")
NEUCOEFF = 0.7  # neuropil correction coefficient (Suite2p default)
N_OUTPUT_BINS = 5  # quintile discretization of motion energy


# =============================================================================
# Suite2p dF/F computation (matching default Suite2p parameters)
# =============================================================================
def compute_dff(F, Fneu, fs=30.0, neucoeff=0.7, win_baseline=60.0, sig_baseline=10.0,
                prctile_baseline=8.0):
    """
    Compute dF/F using Suite2p's default 'maximin' baseline method.

    Matches Suite2p's dcnv.preprocess():
    1. Neuropil correction: Fc = F - neucoeff * Fneu
    2. Baseline estimation using maximin method
    3. dF/F = Fc - F0 (baseline subtraction, as in Suite2p)

    Note: sig_baseline is in frames (not seconds), matching Suite2p's implementation.
    """
    # Step 1: Neuropil correction
    Fc = F - neucoeff * Fneu

    # Step 2: Baseline estimation (maximin method)
    win_frames = int(win_baseline * fs)  # window in frames
    # sig_baseline is used directly as sigma in frames (Suite2p convention)
    F0 = _maximin_baseline(Fc, win_frames, sig_baseline)

    # Step 3: Baseline subtraction (Suite2p's "baseline corrected fluorescence")
    dff = Fc - F0

    return dff


def _maximin_baseline(Fc, win_frames, sig_frames):
    """
    Suite2p's maximin baseline:
    1. Compute running percentile (min) over a window
    2. Smooth with a Gaussian-like filter

    This is a simplified but faithful implementation of Suite2p's baseline.
    We use rolling min then rolling max (maximin), followed by Gaussian smoothing.
    """
    from scipy.ndimage import minimum_filter1d, maximum_filter1d, gaussian_filter1d

    n_neurons, n_frames = Fc.shape

    # Rolling minimum over win_frames
    Flow = minimum_filter1d(Fc, size=win_frames, axis=1)
    # Rolling maximum of the minimum (maximin)
    Flow = maximum_filter1d(Flow, size=win_frames, axis=1)
    # Gaussian smoothing
    Flow = gaussian_filter1d(Flow, sigma=sig_frames, axis=1)

    return Flow


# =============================================================================
# Missing frame interpolation
# =============================================================================
def interpolate_missing_frames(motion_energy, interframe_int, n_neural_frames):
    """
    Interpolate motion energy to match neural frame count when video frames are missing.

    Uses interframe intervals to identify where frames were dropped and inserts
    interpolated values at those positions.
    """
    n_me = len(motion_energy)
    if n_me == n_neural_frames:
        return motion_energy.astype(np.float64)

    n_missing = n_neural_frames - n_me
    if n_missing < 0:
        # More ME frames than neural - truncate
        return motion_energy[:n_neural_frames].astype(np.float64)

    # Use interframe intervals to find missing frame positions
    median_ifi = np.median(interframe_int)
    ratios = interframe_int / median_ifi
    missed_counts = np.round(ratios).astype(int) - 1  # 0 = no miss, 1 = 1 miss, etc.

    # Build mapping from ME frames to neural frames
    # ME frame i corresponds to neural frame at position sum of (1 + missed_counts[:i])
    me_positions = np.zeros(n_me, dtype=int)
    me_positions[0] = 0
    for i in range(1, n_me):
        me_positions[i] = me_positions[i-1] + 1 + missed_counts[i-1]

    # Interpolate to fill all neural frames
    neural_positions = np.arange(n_neural_frames)
    me_interp = np.interp(neural_positions, me_positions, motion_energy.astype(np.float64))

    return me_interp


# =============================================================================
# Binning
# =============================================================================
def bin_data(data, bin_size):
    """
    Bin data by averaging consecutive frames.
    data: array of shape (..., n_frames) or (n_frames,)
    Returns: array with last dimension reduced by factor bin_size
    """
    if data.ndim == 1:
        n = len(data)
        n_bins = n // bin_size
        return data[:n_bins * bin_size].reshape(n_bins, bin_size).mean(axis=1)
    else:
        # For 2D: (n_neurons, n_frames)
        n = data.shape[-1]
        n_bins = n // bin_size
        trimmed = data[..., :n_bins * bin_size]
        new_shape = trimmed.shape[:-1] + (n_bins, bin_size)
        return trimmed.reshape(new_shape).mean(axis=-1)


# =============================================================================
# Session loading and processing
# =============================================================================
def load_session(subject_dir, session_name):
    """Load all data for a single session."""
    session_dir = os.path.join(subject_dir, session_name)
    s2p_dir = os.path.join(session_dir, 'suite2p', 'plane0')
    move_dir = os.path.join(session_dir, 'move_deve')

    F = np.load(os.path.join(s2p_dir, 'F.npy'))
    Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
    me = np.load(os.path.join(move_dir, 'motion_energy_glob.npy'))
    interframe = np.load(os.path.join(move_dir, 'interframe_int.npy'))

    return F, Fneu, me, interframe


def process_session(F, Fneu, me, interframe, bin_size=BIN_SIZE):
    """
    Process a single session:
    1. Compute dF/F
    2. Interpolate missing ME frames
    3. Bin both neural and behavioral data
    """
    n_neurons, n_frames = F.shape

    # Compute dF/F
    dff = compute_dff(F, Fneu, fs=FS, neucoeff=NEUCOEFF)

    # Interpolate missing video frames
    me_interp = interpolate_missing_frames(me, interframe, n_frames)

    # Bin both signals
    dff_binned = bin_data(dff, bin_size)  # (n_neurons, n_bins)
    me_binned = bin_data(me_interp, bin_size)  # (n_bins,)

    return dff_binned, me_binned


def split_into_trials(neural_binned, me_binned, trial_frames):
    """
    Split continuous recording into fixed-length trials.

    neural_binned: (n_neurons, n_bins)
    me_binned: (n_bins,)
    trial_frames: number of binned frames per trial

    Returns lists of arrays for neural and me.
    """
    n_bins = neural_binned.shape[1]
    n_trials = n_bins // trial_frames

    neural_trials = []
    me_trials = []

    for t in range(n_trials):
        start = t * trial_frames
        end = start + trial_frames
        neural_trials.append(neural_binned[:, start:end].astype(np.float32))
        me_trials.append(me_binned[start:end])

    return neural_trials, me_trials


# =============================================================================
# Discretization
# =============================================================================
def discretize_motion_energy(all_me_trials, n_bins=N_OUTPUT_BINS):
    """
    Discretize motion energy into equal-percentile bins across all data.

    Returns:
        discretized trials (list of lists of arrays)
        bin_edges (array of percentile boundaries)
        bin_labels (list of str descriptions)
    """
    # Collect all ME values
    all_values = np.concatenate([me for session_trials in all_me_trials for me in session_trials])

    # Compute percentile bin edges
    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(all_values, percentiles)

    # Make bin edges strictly increasing (handle duplicate edges)
    for i in range(1, len(bin_edges)):
        if bin_edges[i] <= bin_edges[i-1]:
            bin_edges[i] = bin_edges[i-1] + 1e-10

    # Discretize each trial
    discretized = []
    for session_trials in all_me_trials:
        session_disc = []
        for me in session_trials:
            # np.digitize returns bin indices (1-based), clip to valid range
            binned = np.digitize(me, bin_edges[1:-1])  # 0 to n_bins-1
            binned = np.clip(binned, 0, n_bins - 1)
            session_disc.append(binned.reshape(1, -1).astype(np.int64))
        discretized.append(session_disc)

    # Create bin labels
    bin_labels = [f'{percentiles[i]:.0f}-{percentiles[i+1]:.0f}%ile' for i in range(n_bins)]

    return discretized, bin_edges, bin_labels


# =============================================================================
# Main conversion
# =============================================================================
def convert_data(output_path, sample=False, show_processing=False):
    """Main conversion function."""

    total_start = time.time()

    # Discover subjects and sessions
    subjects = sorted([d for d in os.listdir(DATA_DIR)
                       if os.path.isdir(os.path.join(DATA_DIR, d))])

    print(f"Found {len(subjects)} subjects: {subjects}")

    # Build session list
    all_sessions = []
    for subj in subjects:
        subj_dir = os.path.join(DATA_DIR, subj)
        sessions = sorted([d for d in os.listdir(subj_dir)
                          if os.path.isdir(os.path.join(subj_dir, d))])
        for sess in sessions:
            all_sessions.append((subj, sess))

    print(f"Total sessions: {len(all_sessions)}")

    if sample:
        # Select 2 sessions from different subjects for testing
        selected = []
        seen_subjects = set()
        for subj, sess in all_sessions:
            if subj not in seen_subjects and len(selected) < 2:
                selected.append((subj, sess))
                seen_subjects.add(subj)
        all_sessions = selected
        print(f"Sample mode: using {len(all_sessions)} sessions")

    # Process all sessions
    neural_all = []
    me_all_raw = []  # raw binned ME for later discretization
    subject_idx = []
    subject_list = list(dict.fromkeys([s[0] for s in all_sessions]))  # unique, ordered

    trial_frames = int(TRIAL_DURATION_SEC * FS / BIN_SIZE)  # 120 * 30 / 10 = 360
    print(f"Trial length: {trial_frames} bins ({TRIAL_DURATION_SEC}s at {FS/BIN_SIZE:.1f} Hz)")

    for i, (subj, sess) in enumerate(all_sessions):
        t0 = time.time()
        print(f"Processing session {i+1}/{len(all_sessions)}: {subj}/{sess}...", end=" ")

        subj_dir = os.path.join(DATA_DIR, subj)
        F, Fneu, me, interframe = load_session(subj_dir, sess)

        dff_binned, me_binned = process_session(F, Fneu, me, interframe)
        neural_trials, me_trials = split_into_trials(dff_binned, me_binned, trial_frames)

        neural_all.append(neural_trials)
        me_all_raw.append(me_trials)
        subject_idx.append(subject_list.index(subj))

        dt = time.time() - t0
        print(f"{len(neural_trials)} trials, {F.shape[0]} neurons, {dt:.1f}s")

        # Show processing plots for first 2 sessions
        if show_processing and i < 2:
            _plot_processing(subj, sess, F, Fneu, me, interframe,
                           dff_binned, me_binned, neural_trials, me_trials,
                           trial_frames)

    # Discretize motion energy across all sessions
    print("Discretizing motion energy into quintile bins...")
    output_all, bin_edges, bin_labels = discretize_motion_energy(me_all_raw, N_OUTPUT_BINS)

    # Create time input for each trial
    input_all = []
    for session_trials in neural_all:
        session_inputs = []
        for trial in session_trials:
            n_timepoints = trial.shape[1]
            # Time elapsed from start of trial in seconds
            time_input = np.arange(n_timepoints) * (BIN_SIZE / FS)
            session_inputs.append(time_input.reshape(1, -1).astype(np.float32))
        input_all.append(session_inputs)

    # Build output data structure
    data = {
        'neural': neural_all,
        'input': input_all,
        'output': output_all,

        'subjects': subject_list,
        'subject_idx': np.array(subject_idx, dtype=np.int64),

        'brain_regions': ['barrel_cortex'],
        'brain_region_idx': [
            np.zeros(neural_all[s][0].shape[0], dtype=np.int64)
            for s in range(len(neural_all))
        ],

        'input_names': ['time_elapsed_s'],
        'output_names': ['motion_energy_bin'],
        'output_values': [bin_labels],

        'metadata': {
            'task_description': 'Decode motion energy (discretized into 5 quintile bins) from barrel cortex calcium imaging dF/F',
            'time_bin_size': BIN_SIZE / FS * 1000,  # in ms = 333.33
            'temporal_alignment_event': 'Start of 2-minute recording block',
            'off_start': 0.0,
            'off_end': TRIAL_DURATION_SEC,
            'imaging_rate_hz': FS,
            'bin_size_frames': BIN_SIZE,
            'trial_duration_s': TRIAL_DURATION_SEC,
            'neucoeff': NEUCOEFF,
            'baseline_method': 'maximin',
            'win_baseline_s': 60.0,
            'sig_baseline_frames': int(10.0 * FS),
            'source': 'Majnik et al. 2025 - Track2p barrel cortex development',
        }
    }

    # Print summary statistics
    total_trials = sum(len(s) for s in neural_all)
    total_neurons = sum(neural_all[s][0].shape[0] for s in range(len(neural_all)))
    print(f"\n=== Conversion Summary ===")
    print(f"Subjects: {len(subject_list)}")
    print(f"Sessions: {len(neural_all)}")
    print(f"Total trials: {total_trials}")
    print(f"Neurons per session: {[neural_all[s][0].shape[0] for s in range(len(neural_all))]}")
    print(f"Trials per session: {[len(s) for s in neural_all]}")
    print(f"Time bin size: {BIN_SIZE / FS * 1000:.1f} ms")
    print(f"Trial duration: {TRIAL_DURATION_SEC}s ({trial_frames} bins)")
    print(f"Output bins: {bin_labels}")

    # Check output distribution
    all_outputs = np.concatenate([o for sess in output_all for o in sess], axis=1).flatten()
    unique, counts = np.unique(all_outputs, return_counts=True)
    print(f"Output distribution: {dict(zip(unique, counts/counts.sum()))}")

    # Save
    print(f"\nSaving to {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump(data, f, protocol=4)

    total_time = time.time() - total_start
    print(f"Total conversion time: {total_time:.1f}s")

    return data


def _plot_processing(subj, sess, F, Fneu, me, interframe,
                     dff_binned, me_binned, neural_trials, me_trials,
                     trial_frames):
    """Plot processing steps for visual inspection."""
    fig, axes = plt.subplots(5, 1, figsize=(16, 20))

    n_neurons = F.shape[0]
    example_neuron = min(5, n_neurons - 1)
    n_frames = min(3000, F.shape[1])  # Show first 100 seconds

    # 1. Raw F and Fneu
    ax = axes[0]
    ax.plot(F[example_neuron, :n_frames], label='F', alpha=0.7)
    ax.plot(Fneu[example_neuron, :n_frames], label='Fneu', alpha=0.7)
    ax.set_title(f'{subj}/{sess} - Raw fluorescence (neuron {example_neuron})')
    ax.set_xlabel('Frame')
    ax.legend()

    # 2. dF/F (binned)
    ax = axes[1]
    n_show = min(300, dff_binned.shape[1])
    ax.plot(dff_binned[example_neuron, :n_show])
    ax.set_title(f'dF/F (binned, neuron {example_neuron})')
    ax.set_xlabel('Bin')

    # 3. Motion energy (raw vs binned)
    ax = axes[2]
    me_show = min(3000, len(me))
    ax.plot(me[:me_show].astype(float), alpha=0.5, label='Raw ME')
    me_binned_show = min(300, len(me_binned))
    ax.plot(np.arange(me_binned_show) * BIN_SIZE, me_binned[:me_binned_show],
            label='Binned ME', linewidth=2)
    ax.set_title('Motion energy')
    ax.set_xlabel('Frame')
    ax.legend()

    # 4. Neural raster (first trial, subset of neurons)
    ax = axes[3]
    if neural_trials:
        n_show_neurons = min(50, neural_trials[0].shape[0])
        ax.imshow(neural_trials[0][:n_show_neurons, :], aspect='auto',
                  cmap='gray_r', vmin=0, vmax=3)
        ax.set_title(f'Neural raster (trial 1, first {n_show_neurons} neurons)')
        ax.set_xlabel('Bin')
        ax.set_ylabel('Neuron')

    # 5. Neural + ME alignment check
    ax = axes[4]
    if neural_trials and me_trials:
        # Show mean neural activity and ME for trial 1
        ax2 = ax.twinx()
        mean_neural = neural_trials[0].mean(axis=0)
        ax.plot(mean_neural, 'b-', alpha=0.7, label='Mean dF/F')
        ax2.plot(me_trials[0], 'r-', alpha=0.7, label='Motion energy')
        ax.set_title('Alignment check: mean dF/F vs motion energy (trial 1)')
        ax.set_xlabel('Bin')
        ax.set_ylabel('Mean dF/F', color='b')
        ax2.set_ylabel('Motion energy', color='r')

    fig.tight_layout()
    session_id = f"{subj}_{sess}"
    fig.savefig(f'processing_{session_id}.png', dpi=100)
    plt.close(fig)
    print(f"  Saved processing_{session_id}.png")


# =============================================================================
# Entry point
# =============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert Track2p data to decoder format.')
    parser.add_argument('output', type=str, help='Output pickle file path')
    parser.add_argument('--sample', action='store_true', help='Process only 2 sessions for testing')
    parser.add_argument('--full', action='store_true', help='Process all sessions (default)')
    parser.add_argument('--show-processing', action='store_true', help='Plot processing steps')

    args = parser.parse_args()

    convert_data(args.output, sample=args.sample, show_processing=args.show_processing)
