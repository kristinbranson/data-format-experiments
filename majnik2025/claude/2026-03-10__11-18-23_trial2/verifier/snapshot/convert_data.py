#!/usr/bin/env python3
"""
Convert Track2p longitudinal calcium imaging data to decoder-compatible format.

Usage:
    python -u convert_data.py <outpicklefile> [--full | --sample] [--show-processing]

Reference: Majnik et al. 2025 - "Longitudinal tracking of neuronal activity from
the same cells in the developing brain using Track2p"
"""

import os
import sys
import time
import argparse
import pickle
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from suite2p.extraction import preprocess

# ============================================================================
# Constants from paper / Suite2p ops
# ============================================================================
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
FS = 30.0           # imaging frame rate (Hz)
NEUCOEFF = 0.7      # neuropil coefficient (Suite2p default)
BASELINE = 'maximin' # baseline method
WIN_BASELINE = 60.0  # baseline window (seconds)
SIG_BASELINE = 10.0  # baseline Gaussian sigma (frames)
PRCTILE_BASELINE = 8.0
BIN_SIZE = 10        # number of frames per bin (paper: "bins of 10 consecutive timestamps")
TRIAL_DURATION_S = 120.0  # 2 minutes per trial (paper: "consecutive 2 minute blocks")
N_OUTPUT_BINS = 5    # number of motion energy bins
BRAIN_REGION = 'barrel cortex'

# Subjects in order (alphabetical = mouse A-F)
SUBJECTS = ['jm031', 'jm032', 'jm038', 'jm039', 'jm040', 'jm046']


def get_sessions(subject_dir):
    """Get sorted list of session directories for a subject."""
    sessions = []
    for d in sorted(os.listdir(subject_dir)):
        full = os.path.join(subject_dir, d)
        if os.path.isdir(full) and not d.startswith('.'):
            sessions.append(full)
    return sessions


def compute_dff(F, Fneu, device=None):
    """
    Compute dF/F using Suite2p default parameters.

    1. Neuropil subtraction: F_corr = F - 0.7 * Fneu
    2. Baseline correction via maximin filter
    3. dF/F = (F_corr - F0) / F0

    Paper: "We used baseline corrected fluorescence traces as our dF/F
    (using the default Suite2p parameters)"
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Neuropil subtraction
    F_corr = F - NEUCOEFF * Fneu

    # Compute baseline using Suite2p's preprocess (returns F_corr - baseline)
    # We need baseline itself for dF/F = (F_corr - F0) / F0
    # preprocess returns F - baseline, so baseline = F_corr - preprocess(F_corr)
    F_corr_copy = F_corr.copy()
    F_subtracted = preprocess(
        F_corr_copy, BASELINE, WIN_BASELINE, SIG_BASELINE, FS,
        prctile_baseline=PRCTILE_BASELINE, device=device
    )
    # F_subtracted = F_corr - baseline => baseline = F_corr - F_subtracted
    baseline = F_corr - F_subtracted

    # Avoid division by zero (clip baseline to small positive value)
    baseline_safe = np.clip(baseline, 1e-6, None)

    # dF/F = (F_corr - baseline) / baseline = F_subtracted / baseline
    dff = F_subtracted / baseline_safe

    return dff.astype(np.float32)


def interpolate_motion_energy(me, n_neural_frames, tstamps):
    """
    Handle frame mismatches between motion energy and neural data.

    From data README: "In some recordings there might be some missing frames
    from the camera... The indices of missing frames can be obtained by looking
    at 'tstamps.npy' or 'interframe_int.npy' and treated as missing values for
    motion energy or they can be interpolated over"
    """
    n_me = len(me)
    if n_me == n_neural_frames:
        return me.astype(np.float64)

    # Missing frames: ME has fewer frames than neural
    # tstamps has same length as ME, values are timestamps for each ME frame
    # We need to interpolate ME to match neural frames
    if n_me < n_neural_frames:
        # Create neural frame indices
        neural_indices = np.arange(n_neural_frames)
        # ME frame indices based on tstamps (map to neural frame space)
        # tstamps values are inter-frame intervals in seconds
        # Actually tstamps appears to be cumulative or relative timestamps
        # Since camera is triggered by microscope, ME frames map 1:1 but some are missing
        # Use linear interpolation to fill in missing frames
        me_indices = np.linspace(0, n_neural_frames - 1, n_me)
        me_interp = np.interp(neural_indices, me_indices, me.astype(np.float64))
        return me_interp
    else:
        # ME has more frames than neural (shouldn't happen, but truncate)
        return me[:n_neural_frames].astype(np.float64)


def bin_traces(data, bin_size):
    """
    Average data in non-overlapping bins.

    Paper: "slightly denoised the dF/F as well as the behaviour traces
    by averaging in bins of 10 consecutive timestamps"

    Parameters:
        data: (n_features, n_timepoints) or (n_timepoints,)
        bin_size: number of frames per bin
    """
    if data.ndim == 1:
        n_frames = len(data)
        n_bins = n_frames // bin_size
        trimmed = data[:n_bins * bin_size]
        return trimmed.reshape(n_bins, bin_size).mean(axis=1)
    else:
        n_features, n_frames = data.shape
        n_bins = n_frames // bin_size
        trimmed = data[:, :n_bins * bin_size]
        return trimmed.reshape(n_features, n_bins, bin_size).mean(axis=2)


def process_session(session_dir, device=None):
    """
    Process a single session: load data, compute dF/F, handle ME mismatches, bin.

    Returns:
        dff_binned: (n_neurons, n_timebins) binned dF/F
        me_binned: (n_timebins,) binned motion energy
        n_neurons: int
        n_frames_raw: int (before binning)
    """
    s2p_dir = os.path.join(session_dir, 'suite2p', 'plane0')
    me_dir = os.path.join(session_dir, 'move_deve')

    # Load neural data
    F = np.load(os.path.join(s2p_dir, 'F.npy'))
    Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))

    # Load motion energy and timestamps
    me = np.load(os.path.join(me_dir, 'motion_energy_glob.npy'))
    tstamps = np.load(os.path.join(me_dir, 'tstamps.npy'))

    n_neurons, n_frames = F.shape

    # Compute dF/F
    dff = compute_dff(F, Fneu, device=device)

    # Interpolate motion energy to match neural frames
    me_interp = interpolate_motion_energy(me, n_frames, tstamps)

    # Bin both signals by 10 frames
    dff_binned = bin_traces(dff, BIN_SIZE)
    me_binned = bin_traces(me_interp, BIN_SIZE)

    return dff_binned, me_binned, n_neurons, n_frames


def split_into_trials(dff_binned, me_binned):
    """
    Split binned session data into 2-minute trials.

    Paper: "splits were done on consecutive 2 minute blocks of the recording"

    Trial length in bins: 2 min * 60 s/min * 30 Hz / 10 frames/bin = 360 bins
    """
    bins_per_trial = int(TRIAL_DURATION_S * FS / BIN_SIZE)  # 360
    n_bins = dff_binned.shape[1]
    n_trials = n_bins // bins_per_trial

    neural_trials = []
    me_trials = []

    for t in range(n_trials):
        start = t * bins_per_trial
        end = start + bins_per_trial
        neural_trials.append(dff_binned[:, start:end].astype(np.float32))
        me_trials.append(me_binned[start:end])

    return neural_trials, me_trials


def compute_me_percentile_bins(all_me_values, n_bins=N_OUTPUT_BINS):
    """
    Compute percentile bin edges for motion energy discretization.

    Task: "normalized and discretized into five equal-percentile bins"
    """
    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(all_me_values, percentiles)
    return bin_edges


def discretize_me(me_values, bin_edges):
    """
    Discretize motion energy into bins using precomputed percentile edges.
    Returns integer labels 0 to n_bins-1.
    """
    n_bins = len(bin_edges) - 1
    labels = np.digitize(me_values, bin_edges[1:-1])  # 0 to n_bins-1
    labels = np.clip(labels, 0, n_bins - 1)
    return labels.astype(np.int64)


def make_time_input(n_timebins):
    """
    Create time-elapsed input for a trial.

    Task: "Time elapsed from the beginning of the experiment. Time-varying."
    Time in seconds from start of trial.
    """
    time_bin_s = BIN_SIZE / FS  # seconds per bin
    return (np.arange(n_timebins) * time_bin_s).astype(np.float32).reshape(1, -1)


def plot_processing(session_dir, dff_binned, me_binned, me_disc_trials,
                    neural_trials, n_frames_raw, save_path):
    """Plot processing steps for visual verification."""
    s2p_dir = os.path.join(session_dir, 'suite2p', 'plane0')
    me_dir = os.path.join(session_dir, 'move_deve')

    F = np.load(os.path.join(s2p_dir, 'F.npy'))
    Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
    me_raw = np.load(os.path.join(me_dir, 'motion_energy_glob.npy'))

    fig, axes = plt.subplots(5, 1, figsize=(16, 20))
    session_name = os.path.basename(session_dir)
    fig.suptitle(f'Processing: {session_name}', fontsize=14)

    # 1. Raw F vs neuropil-corrected
    neuron_idx = 0
    ax = axes[0]
    t_raw = np.arange(min(3000, F.shape[1])) / FS
    ax.plot(t_raw, F[neuron_idx, :len(t_raw)], alpha=0.5, label='F (raw)')
    ax.plot(t_raw, (F[neuron_idx, :len(t_raw)] - NEUCOEFF * Fneu[neuron_idx, :len(t_raw)]),
            alpha=0.5, label='F - 0.7*Fneu')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Fluorescence')
    ax.set_title(f'Neuron {neuron_idx}: Raw vs Neuropil-corrected (first 100s)')
    ax.legend()

    # 2. dF/F binned (raster of first 10 neurons)
    ax = axes[1]
    n_show = min(20, dff_binned.shape[0])
    t_binned = np.arange(dff_binned.shape[1]) * BIN_SIZE / FS
    im = ax.imshow(dff_binned[:n_show, :], aspect='auto', cmap='RdBu_r',
                   extent=[0, t_binned[-1], n_show, 0],
                   vmin=-2, vmax=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Neuron')
    ax.set_title(f'Binned dF/F (first {n_show} neurons)')
    plt.colorbar(im, ax=ax)

    # 3. Motion energy: raw vs binned
    ax = axes[2]
    t_me_raw = np.arange(len(me_raw)) / FS
    ax.plot(t_me_raw, me_raw, alpha=0.3, label='ME raw')
    ax.plot(t_binned, me_binned, label='ME binned', linewidth=1)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Motion Energy')
    ax.set_title('Motion Energy: Raw vs Binned')
    ax.legend()

    # 4. Discretized motion energy
    ax = axes[3]
    all_disc = np.concatenate(me_disc_trials)
    t_disc = np.arange(len(all_disc)) * BIN_SIZE / FS
    ax.plot(t_disc, all_disc, '.', markersize=1)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('ME Bin (0-4)')
    ax.set_title('Discretized Motion Energy (5 bins)')
    ax.set_yticks(range(N_OUTPUT_BINS))

    # 5. Trial structure verification (overlay trial boundaries on dF/F neuron 0)
    ax = axes[4]
    ax.plot(t_binned, dff_binned[0, :], alpha=0.7, label='dF/F neuron 0')
    bins_per_trial = int(TRIAL_DURATION_S * FS / BIN_SIZE)
    for i in range(len(neural_trials) + 1):
        ax.axvline(x=i * bins_per_trial * BIN_SIZE / FS, color='r',
                   linestyle='--', alpha=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('dF/F')
    ax.set_title(f'Trial boundaries ({len(neural_trials)} trials)')
    ax.legend()

    plt.tight_layout()
    fig.savefig(save_path, dpi=100)
    plt.close(fig)
    print(f"  Saved processing plot: {save_path}")


def convert_data(output_path, sample=False, show_processing=False):
    """Main conversion function."""
    t_start = time.time()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Collect all session data
    all_neural = []      # list of sessions, each session is list of trial arrays
    all_input = []       # list of sessions
    all_output = []      # list of sessions
    subject_idx_list = []
    brain_region_idx_list = []

    # First pass: process all sessions and collect raw binned ME for global percentile computation
    session_data = []  # (subject_idx, session_dir, dff_binned, me_binned, n_neurons)
    all_me_binned_values = []

    subjects_to_process = SUBJECTS
    if sample:
        # Process only 2 sessions total (from first subject)
        subjects_to_process = SUBJECTS[:1]

    session_count = 0
    for subj_idx, subject in enumerate(subjects_to_process):
        subject_dir = os.path.join(DATA_DIR, subject)
        if not os.path.isdir(subject_dir):
            print(f"Warning: Subject directory not found: {subject_dir}")
            continue

        sessions = get_sessions(subject_dir)
        if sample:
            sessions = sessions[:2]  # Only 2 sessions for sample

        print(f"\nProcessing subject {subject} ({len(sessions)} sessions)...")

        for sess_dir in sessions:
            t_sess = time.time()
            sess_name = os.path.basename(sess_dir)

            dff_binned, me_binned, n_neurons, n_frames_raw = process_session(
                sess_dir, device=device
            )

            session_data.append((subj_idx, sess_dir, dff_binned, me_binned, n_neurons))
            all_me_binned_values.append(me_binned)
            session_count += 1

            elapsed = time.time() - t_sess
            print(f"  {sess_name}: {n_neurons} neurons, {n_frames_raw} frames -> "
                  f"{dff_binned.shape[1]} bins ({elapsed:.1f}s)")

    # Compute global percentile bin edges for motion energy
    print(f"\nComputing global ME percentile bins...")
    all_me_concat = np.concatenate(all_me_binned_values)
    bin_edges = compute_me_percentile_bins(all_me_concat, N_OUTPUT_BINS)
    print(f"  ME bin edges: {bin_edges}")
    print(f"  ME range: [{all_me_concat.min():.2f}, {all_me_concat.max():.2f}]")

    # Generate bin labels for output_values
    output_value_labels = []
    for i in range(N_OUTPUT_BINS):
        if i == 0:
            output_value_labels.append(f"ME<{bin_edges[1]:.0f}")
        elif i == N_OUTPUT_BINS - 1:
            output_value_labels.append(f"ME>={bin_edges[-2]:.0f}")
        else:
            output_value_labels.append(f"{bin_edges[i]:.0f}<=ME<{bin_edges[i+1]:.0f}")

    # Second pass: split into trials and discretize ME
    print(f"\nSplitting into trials and discretizing...")
    plot_count = 0

    for subj_idx, sess_dir, dff_binned, me_binned, n_neurons in session_data:
        # Split into trials
        neural_trials, me_trials = split_into_trials(dff_binned, me_binned)

        # Discretize ME for each trial
        input_trials = []
        output_trials = []
        me_disc_trials = []

        for neural_t, me_t in zip(neural_trials, me_trials):
            n_timebins = neural_t.shape[1]

            # Input: time elapsed
            time_input = make_time_input(n_timebins)
            input_trials.append(time_input)

            # Output: discretized ME
            me_disc = discretize_me(me_t, bin_edges)
            output_trials.append(me_disc.reshape(1, -1))
            me_disc_trials.append(me_disc)

        all_neural.append(neural_trials)
        all_input.append(input_trials)
        all_output.append(output_trials)
        subject_idx_list.append(subj_idx)
        brain_region_idx_list.append(np.zeros(n_neurons, dtype=np.int64))

        sess_name = os.path.basename(sess_dir)
        print(f"  {sess_name}: {len(neural_trials)} trials, "
              f"{neural_trials[0].shape if neural_trials else 'N/A'}")

        # Processing plots
        if show_processing and plot_count < 2:
            plot_path = f"processing_{sess_name}.png"
            plot_processing(sess_dir, dff_binned, me_binned, me_disc_trials,
                          neural_trials, dff_binned.shape[1] * BIN_SIZE, plot_path)
            plot_count += 1

    # Build final data dict
    time_bin_ms = BIN_SIZE / FS * 1000  # ms

    data = {
        'neural': all_neural,
        'input': all_input,
        'output': all_output,

        'subjects': SUBJECTS if not sample else SUBJECTS[:1],
        'subject_idx': np.array(subject_idx_list, dtype=np.int64),

        'brain_regions': [BRAIN_REGION],
        'brain_region_idx': brain_region_idx_list,

        'input_names': ['time_elapsed_s'],
        'output_names': ['motion_energy_bin'],
        'output_values': [output_value_labels],

        'metadata': {
            'task_description': 'Decode motion energy from barrel cortex neural activity during spontaneous behavior in developing mice',
            'time_bin_size': time_bin_ms,
            'temporal_alignment_event': 'Start of recording session',
            'off_start': 0.0,
            'off_end': None,
            'imaging_rate_hz': FS,
            'bin_size_frames': BIN_SIZE,
            'trial_duration_s': TRIAL_DURATION_S,
            'n_output_bins': N_OUTPUT_BINS,
            'me_bin_edges': bin_edges.tolist(),
            'neuropil_coefficient': NEUCOEFF,
            'baseline_method': BASELINE,
            'source': 'Majnik et al. 2025 - Track2p',
        }
    }

    # Save
    print(f"\nSaving to {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)

    file_size = os.path.getsize(output_path) / (1024 * 1024)
    total_time = time.time() - t_start

    # Print summary
    n_sessions = len(all_neural)
    n_trials_total = sum(len(s) for s in all_neural)
    n_neurons_list = [all_neural[s][0].shape[0] for s in range(n_sessions) if all_neural[s]]

    print(f"\n=== Conversion Summary ===")
    print(f"Sessions: {n_sessions}")
    print(f"Trials total: {n_trials_total}")
    print(f"Neurons per session: {n_neurons_list}")
    print(f"Subjects: {data['subjects']}")
    print(f"Time bin size: {time_bin_ms:.2f} ms")
    print(f"Output bins: {N_OUTPUT_BINS}")
    print(f"File size: {file_size:.1f} MB")
    print(f"Total time: {total_time:.1f}s")

    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert Track2p data to decoder format')
    parser.add_argument('output', type=str, help='Output pickle file path')
    parser.add_argument('--full', action='store_true', default=True,
                       help='Process all sessions (default)')
    parser.add_argument('--sample', action='store_true',
                       help='Process only 2 sessions for testing')
    parser.add_argument('--show-processing', action='store_true',
                       help='Plot processing visualizations')

    args = parser.parse_args()

    sample_mode = args.sample
    if sample_mode:
        args.full = False

    convert_data(args.output, sample=sample_mode, show_processing=args.show_processing)
