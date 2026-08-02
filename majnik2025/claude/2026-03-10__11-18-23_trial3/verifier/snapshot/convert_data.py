#!/usr/bin/env python3
"""
Convert Track2p longitudinal calcium imaging data to decoder-compatible format.

Usage:
    python -u convert_data.py <outpicklefile> [--full|--sample] [--show-processing]

The data consists of 2-photon calcium imaging from layer 2/3 barrel cortex of
developing mice (P7-P14), with simultaneous videography-derived motion energy.

Processing follows the reference paper (Majnik et al. 2025, eLife):
- dF/F computed using Suite2p's baseline correction (maximin method)
- Neural and behavioral data binned by 10 frames (~333ms)
- Motion energy normalized per session and discretized into 5 equal-percentile bins
- Continuous 20-30 min recordings split into 2-minute trials
"""

import os
import sys
import time
import argparse
import pickle
import numpy as np
import warnings

# Suite2p baseline correction
from suite2p.extraction.dcnv import preprocess as s2p_preprocess
import torch

# ============================================================
# Constants matching paper's methods
# ============================================================
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
FRAME_RATE = 30.0          # Hz
NEUROPIL_COEFF = 0.7       # Suite2p default
BIN_SIZE = 10              # frames per bin (paper: "bins of 10 consecutive timestamps")
TRIAL_DURATION_SEC = 120   # 2 minutes (paper: "consecutive 2 minute blocks")
TRIAL_BINS = int(TRIAL_DURATION_SEC * FRAME_RATE / BIN_SIZE)  # 360 bins per trial
N_OUTPUT_BINS = 5          # quintile discretization
TIME_BIN_MS = BIN_SIZE / FRAME_RATE * 1000  # ~333.33 ms

# Suite2p baseline correction parameters (from ops.npy)
BASELINE_METHOD = 'maximin'
WIN_BASELINE = 60.0        # seconds
SIG_BASELINE = 10.0        # frames
PRCTILE_BASELINE = 8.0

# Subject to paper name mapping
SUBJECT_MAP = {
    'jm031': 'Mouse_A', 'jm032': 'Mouse_B', 'jm038': 'Mouse_C',
    'jm039': 'Mouse_D', 'jm040': 'Mouse_E', 'jm046': 'Mouse_F'
}


def get_subjects_and_sessions():
    """Discover all subjects and their sessions from the data directory."""
    subjects = sorted([d for d in os.listdir(DATA_DIR)
                       if os.path.isdir(os.path.join(DATA_DIR, d)) and d.startswith('jm')])
    all_sessions = {}
    for subj in subjects:
        subj_dir = os.path.join(DATA_DIR, subj)
        sessions = sorted([d for d in os.listdir(subj_dir)
                          if os.path.isdir(os.path.join(subj_dir, d))])
        all_sessions[subj] = sessions
    return subjects, all_sessions


def compute_dfof(F, Fneu, fs=FRAME_RATE):
    """
    Compute dF/F following the paper's method:
    1. Neuropil correction: Fc = F - 0.7 * Fneu
    2. Baseline correction using Suite2p's maximin method

    The paper states: "We used baseline corrected fluorescence traces as our dF/F
    (using the default Suite2p parameters)". Suite2p's preprocess function returns
    the baseline-subtracted signal (Fc - F0), which is the "baseline corrected"
    fluorescence used for all subsequent analyses.
    """
    # Neuropil correction
    Fc = F - NEUROPIL_COEFF * Fneu

    # Suite2p baseline correction: returns Fc - F0 (baseline-subtracted)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dfof = s2p_preprocess(
        Fc.copy().astype(np.float32), BASELINE_METHOD, WIN_BASELINE,
        SIG_BASELINE, fs, PRCTILE_BASELINE, device=device
    )

    return dfof.astype(np.float32)


def interpolate_motion_energy(me, n_frames):
    """
    Interpolate motion energy to match neural frame count.
    Handles missing camera frames as described in data README.
    """
    if len(me) == n_frames:
        return me.astype(np.float64)

    # Linear interpolation to match neural frame count
    x_orig = np.linspace(0, 1, len(me))
    x_new = np.linspace(0, 1, n_frames)
    me_interp = np.interp(x_new, x_orig, me.astype(np.float64))
    return me_interp


def bin_timeseries(data, bin_size):
    """
    Bin a time series by averaging consecutive frames.
    data: (..., n_frames) array
    Returns: (..., n_bins) array where n_bins = n_frames // bin_size
    """
    n_frames = data.shape[-1]
    n_bins = n_frames // bin_size
    # Truncate to exact multiple of bin_size
    truncated = data[..., :n_bins * bin_size]
    # Reshape and mean
    new_shape = truncated.shape[:-1] + (n_bins, bin_size)
    return truncated.reshape(new_shape).mean(axis=-1)


def discretize_motion_energy(me_binned, n_bins=N_OUTPUT_BINS):
    """
    Discretize motion energy into equal-percentile bins per session.
    Returns integer labels 0..n_bins-1.
    """
    # Compute percentile boundaries
    percentiles = np.linspace(0, 100, n_bins + 1)[1:-1]  # e.g., [20, 40, 60, 80]
    thresholds = np.percentile(me_binned, percentiles)

    # Assign bins using digitize
    labels = np.digitize(me_binned, thresholds).astype(np.int64)
    return labels


def process_session(subj, session, show_processing=False, ax_list=None):
    """
    Process a single session: load data, compute dF/F, bin, discretize.

    Returns:
        neural_trials: list of (n_neurons, n_bins_per_trial) arrays
        input_trials: list of (1, n_bins_per_trial) arrays
        output_trials: list of (1, n_bins_per_trial) arrays
        n_neurons: int
    """
    sess_dir = os.path.join(DATA_DIR, subj, session)

    # Load neural data
    F = np.load(os.path.join(sess_dir, 'suite2p', 'plane0', 'F.npy'))
    Fneu = np.load(os.path.join(sess_dir, 'suite2p', 'plane0', 'Fneu.npy'))
    n_neurons, n_frames = F.shape

    # Load motion energy
    me_raw = np.load(os.path.join(sess_dir, 'move_deve', 'motion_energy_glob.npy'))

    # Interpolate motion energy to match neural frames
    me = interpolate_motion_energy(me_raw, n_frames)

    # Compute dF/F
    dfof = compute_dfof(F, Fneu)

    # Bin neural data and motion energy by 10 frames
    dfof_binned = bin_timeseries(dfof, BIN_SIZE)  # (n_neurons, n_bins)
    me_binned = bin_timeseries(me.reshape(1, -1), BIN_SIZE).squeeze()  # (n_bins,)

    # Discretize motion energy into 5 equal-percentile bins
    me_discrete = discretize_motion_energy(me_binned, N_OUTPUT_BINS)

    # Split into 2-minute trials
    n_total_bins = dfof_binned.shape[1]
    n_trials = n_total_bins // TRIAL_BINS

    neural_trials = []
    input_trials = []
    output_trials = []

    for t in range(n_trials):
        start = t * TRIAL_BINS
        end = (t + 1) * TRIAL_BINS

        # Neural: (n_neurons, TRIAL_BINS)
        neural_trials.append(dfof_binned[:, start:end].astype(np.float32))

        # Input: time elapsed from start of recording in seconds
        # Each bin covers BIN_SIZE/FRAME_RATE seconds
        time_sec = np.arange(start, end) * (BIN_SIZE / FRAME_RATE)
        input_trials.append(time_sec.reshape(1, -1).astype(np.float32))

        # Output: discretized motion energy (1, TRIAL_BINS)
        output_trials.append(me_discrete[start:end].reshape(1, -1).astype(np.int64))

    if show_processing and ax_list is not None:
        plot_processing(ax_list, subj, session, F, Fneu, dfof, me, me_binned,
                       me_discrete, dfof_binned, n_frames)

    return neural_trials, input_trials, output_trials, n_neurons


def plot_processing(ax_list, subj, session, F, Fneu, dfof, me, me_binned,
                    me_discrete, dfof_binned, n_frames):
    """Plot processing steps for visual verification."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = ax_list
    neuron_idx = min(5, dfof.shape[0] - 1)
    time_frames = np.arange(min(3000, n_frames)) / FRAME_RATE  # First 100s
    time_bins = np.arange(min(300, dfof_binned.shape[1])) * BIN_SIZE / FRAME_RATE
    n_show_frames = len(time_frames)
    n_show_bins = len(time_bins)

    # Row 0: Raw F and Fneu
    axes[0].plot(time_frames, F[neuron_idx, :n_show_frames], alpha=0.7, label='F')
    axes[0].plot(time_frames, Fneu[neuron_idx, :n_show_frames], alpha=0.7, label='Fneu')
    axes[0].set_title(f'{subj}/{session} - Raw fluorescence (neuron {neuron_idx})')
    axes[0].legend()
    axes[0].set_ylabel('Fluorescence')

    # Row 1: dF/F
    axes[1].plot(time_frames, dfof[neuron_idx, :n_show_frames], alpha=0.7)
    axes[1].set_title('dF/F (neuropil corrected + baseline)')
    axes[1].set_ylabel('dF/F')

    # Row 2: Binned dF/F
    axes[2].plot(time_bins, dfof_binned[neuron_idx, :n_show_bins], alpha=0.7)
    axes[2].set_title(f'Binned dF/F (bin={BIN_SIZE} frames)')
    axes[2].set_ylabel('dF/F')

    # Row 3: Raw motion energy
    axes[3].plot(time_frames, me[:n_show_frames], alpha=0.7)
    axes[3].set_title('Raw motion energy')
    axes[3].set_ylabel('Motion energy')

    # Row 4: Binned ME + discretized overlay
    ax4 = axes[4]
    ax4.plot(time_bins, me_binned[:n_show_bins], alpha=0.7, label='Binned ME')
    ax4.set_ylabel('Binned ME')
    ax4.set_title('Binned motion energy + discretized')
    ax4b = ax4.twinx()
    ax4b.plot(time_bins, me_discrete[:n_show_bins], 'r-', alpha=0.5, label='Discrete')
    ax4b.set_ylabel('Bin label')
    ax4b.set_ylim(-0.5, N_OUTPUT_BINS - 0.5)

    # Row 5: Distribution of discrete bins
    unique, counts = np.unique(me_discrete, return_counts=True)
    axes[5].bar(unique, counts / len(me_discrete))
    axes[5].set_title('Motion energy bin distribution')
    axes[5].set_xlabel('Bin')
    axes[5].set_ylabel('Fraction')

    for ax in axes[:-1]:
        ax.set_xlabel('')
    axes[-1].set_xlabel('Time (s)')


def convert_data(output_file, sample_mode=False, show_processing=False):
    """Main conversion function."""
    t_start = time.time()

    subjects, all_sessions = get_subjects_and_sessions()
    print(f"Found {len(subjects)} subjects: {subjects}")
    for subj in subjects:
        print(f"  {subj}: {len(all_sessions[subj])} sessions")

    # In sample mode, use only 2 sessions from first subject
    if sample_mode:
        subjects_to_process = [subjects[0]]
        sessions_to_process = {subjects[0]: all_sessions[subjects[0]][:2]}
        print(f"\nSAMPLE MODE: Processing {subjects_to_process[0]}, "
              f"sessions {sessions_to_process[subjects_to_process[0]]}")
    else:
        subjects_to_process = subjects
        sessions_to_process = all_sessions

    # Setup processing plots
    if show_processing:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

    # Build data structure
    neural_all = []
    input_all = []
    output_all = []
    subject_idx_list = []
    brain_region_idx_all = []
    subject_names = list(SUBJECT_MAP.values()) if not sample_mode else [SUBJECT_MAP[subjects_to_process[0]]]

    session_count = 0
    for subj in subjects_to_process:
        subj_idx = subject_names.index(SUBJECT_MAP[subj])
        sessions = sessions_to_process[subj]

        for i, session in enumerate(sessions):
            t_sess = time.time()
            print(f"\nProcessing {subj}/{session} ({session_count + 1})...")

            # Setup plotting
            fig_ax = None
            if show_processing and session_count < 2:
                import matplotlib.pyplot as plt
                fig, axes = plt.subplots(6, 1, figsize=(16, 20))
                fig_ax = (fig, axes)

            neural_trials, input_trials, output_trials, n_neurons = process_session(
                subj, session, show_processing=(show_processing and session_count < 2),
                ax_list=fig_ax
            )

            neural_all.append(neural_trials)
            input_all.append(input_trials)
            output_all.append(output_trials)
            subject_idx_list.append(subj_idx)
            brain_region_idx_all.append(np.zeros(n_neurons, dtype=np.int64))

            elapsed = time.time() - t_sess
            print(f"  {n_neurons} neurons, {len(neural_trials)} trials, "
                  f"trial shape: ({n_neurons}, {TRIAL_BINS}), "
                  f"time: {elapsed:.1f}s")

            # Save processing plot
            if show_processing and fig_ax is not None:
                fig, axes = fig_ax
                fig.suptitle(f'Processing: {subj}/{session}', fontsize=14)
                fig.tight_layout()
                fig.savefig(f'processing_{subj}_{session}.png', dpi=100)
                plt.close(fig)

            session_count += 1

    # Assemble final data dict
    data = {
        'neural': neural_all,
        'input': input_all,
        'output': output_all,
        'subjects': subject_names,
        'subject_idx': np.array(subject_idx_list, dtype=np.int64),
        'brain_regions': ['barrel_cortex'],
        'brain_region_idx': brain_region_idx_all,
        'input_names': ['time_elapsed_s'],
        'output_names': ['motion_energy'],
        'output_values': [
            [f'quintile_{i+1}' for i in range(N_OUTPUT_BINS)]
        ],
        'metadata': {
            'task_description': 'Decode motion energy from spontaneous neural activity in developing mouse barrel cortex',
            'time_bin_size': TIME_BIN_MS,
            'temporal_alignment_event': 'start of continuous recording session',
            'off_start': 0.0,
            'off_end': None,
            'frame_rate': FRAME_RATE,
            'bin_size_frames': BIN_SIZE,
            'trial_duration_s': TRIAL_DURATION_SEC,
            'neuropil_coefficient': NEUROPIL_COEFF,
            'baseline_method': BASELINE_METHOD,
            'n_output_bins': N_OUTPUT_BINS,
            'paper': 'Majnik et al. 2025, eLife - Track2p',
            'brain_region': 'Layer 2/3 barrel cortex',
        }
    }

    # Save
    print(f"\nSaving to {output_file}...")
    with open(output_file, 'wb') as f:
        pickle.dump(data, f, protocol=4)

    total_time = time.time() - t_start
    print(f"\nConversion complete in {total_time:.1f}s")
    print(f"  Sessions: {session_count}")
    print(f"  Subjects: {len(subject_names)}")
    total_trials = sum(len(s) for s in neural_all)
    print(f"  Total trials: {total_trials}")
    print(f"  Output file: {output_file} ({os.path.getsize(output_file) / 1e6:.1f} MB)")

    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert Track2p data to decoder format.')
    parser.add_argument('output_file', type=str, help='Output pickle file path')
    parser.add_argument('--full', action='store_true', default=True,
                        help='Process all sessions (default)')
    parser.add_argument('--sample', action='store_true',
                        help='Process only 2 sessions for testing')
    parser.add_argument('--show-processing', action='store_true',
                        help='Plot processing steps for up to 2 sessions')

    args = parser.parse_args()
    if args.sample:
        args.full = False

    convert_data(args.output_file, sample_mode=args.sample,
                 show_processing=args.show_processing)
