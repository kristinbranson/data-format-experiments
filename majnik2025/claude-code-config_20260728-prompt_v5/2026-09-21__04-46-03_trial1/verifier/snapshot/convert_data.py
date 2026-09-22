#!/usr/bin/env python3
"""
Convert Track2p longitudinal calcium imaging data to decoder-compatible format.

Usage:
    python -u convert_data.py <outpicklefile> [--full|--sample] [--show-processing]

Data source: Majnik et al. 2025 - Track2p barrel cortex development dataset.
"""

import os
import sys
import time
import argparse
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d, minimum_filter1d, maximum_filter1d

# ============================================================================
# Constants
# ============================================================================
DATA_DIR = '/app/data'
FRAME_RATE = 30.0  # Hz (from paper: "Imaging rate was 30 Hz")
BIN_SIZE = 10  # frames (from paper: "averaging in bins of 10 consecutive timestamps")
BIN_DURATION_S = BIN_SIZE / FRAME_RATE  # ~0.333 seconds
TRIAL_DURATION_S = 60.0  # seconds per trial (from decoder task spec)
BINS_PER_TRIAL = int(TRIAL_DURATION_S / BIN_DURATION_S)  # 180 bins
NEUCOEFF = 0.7  # Suite2p default neuropil coefficient
N_OUTPUT_BINS = 5  # Number of percentile bins for motion energy

SUBJECTS = sorted([d for d in os.listdir(DATA_DIR)
                    if os.path.isdir(os.path.join(DATA_DIR, d)) and d.startswith('jm')])

# Subject name to letter mapping (from data README)
SUBJECT_MAP = {
    'jm031': 'Mouse A', 'jm032': 'Mouse B', 'jm038': 'Mouse C',
    'jm039': 'Mouse D', 'jm040': 'Mouse E', 'jm046': 'Mouse F'
}


# ============================================================================
# Processing functions
# ============================================================================

def get_sessions(subject):
    """Get sorted list of session directories for a subject."""
    subj_dir = os.path.join(DATA_DIR, subject)
    sessions = sorted([d for d in os.listdir(subj_dir)
                       if os.path.isdir(os.path.join(subj_dir, d)) and d[0] == '2'])
    return sessions


def compute_dff(F, Fneu, ops):
    """
    Compute dF/F using Suite2p's baseline correction method.

    As described in the paper: "We used baseline corrected fluorescence traces
    as our dF/F (using the default Suite2p parameters)"

    Suite2p default:
    1. Neuropil correction: Fc = F - 0.7 * Fneu
    2. Maximin baseline: Gaussian smooth -> running min -> running max
    3. dF/F = (Fc - baseline) / baseline
    """
    # Neuropil correction
    Fc = F - NEUCOEFF * Fneu

    # Baseline correction using Suite2p's maximin method
    sig_baseline = ops.get('sig_baseline', 10.0)  # in frames
    win_baseline = ops.get('win_baseline', 60.0)  # in seconds
    fs = ops.get('fs', FRAME_RATE)

    win = int(win_baseline * fs)  # window in frames

    # Maximin baseline: Gaussian smooth -> running min -> running max
    smoothed = gaussian_filter1d(Fc, sigma=sig_baseline, axis=1)
    flow_min = minimum_filter1d(smoothed, size=win, axis=1)
    baseline = maximum_filter1d(flow_min, size=win, axis=1)

    # dF/F with floor to avoid division by near-zero or negative baselines
    baseline_safe = np.maximum(baseline, 10.0)
    dff = (Fc - baseline) / baseline_safe

    return dff


def align_motion_energy(me, me_timestamps, n_neural_frames):
    """
    Align motion energy to neural frames, handling missing camera frames.

    From data README: "In some recordings there might be some missing frames
    from the camera... can be interpolated over"

    Camera is triggered by microscope, so frame indices should be 1:1.
    When ME has fewer frames, we interpolate to fill in missing frames.
    """
    if len(me) == n_neural_frames:
        return me.copy()

    # Camera timestamps are in kiloseconds, convert to seconds
    cam_times_s = me_timestamps * 1000.0

    # Neural frame times in seconds
    neural_times_s = np.arange(n_neural_frames) / FRAME_RATE

    # Interpolate ME to neural frame times
    me_aligned = np.interp(neural_times_s, cam_times_s, me)

    return me_aligned


def bin_data(data, bin_size):
    """
    Bin data by averaging consecutive frames.

    From paper: "we slightly denoised the dF/F as well as the behaviour traces
    by averaging in bins of 10 consecutive timestamps"

    For 2D: data shape (n_neurons, n_frames) -> (n_neurons, n_bins)
    For 1D: data shape (n_frames,) -> (n_bins,)
    """
    if data.ndim == 2:
        n_neurons, n_frames = data.shape
        n_bins = n_frames // bin_size
        trimmed = data[:, :n_bins * bin_size]
        return trimmed.reshape(n_neurons, n_bins, bin_size).mean(axis=2)
    else:
        n_frames = len(data)
        n_bins = n_frames // bin_size
        trimmed = data[:n_bins * bin_size]
        return trimmed.reshape(n_bins, bin_size).mean(axis=1)


def discretize_motion_energy(me_binned, n_bins=N_OUTPUT_BINS):
    """
    Discretize motion energy into equal-percentile bins per session.

    Returns integer labels 0..n_bins-1 and the bin edges.
    """
    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(me_binned, percentiles)

    # Make bin edges strictly increasing to handle ties
    # Use digitize with right=False
    labels = np.digitize(me_binned, bin_edges[1:-1], right=False)
    # Clip to valid range
    labels = np.clip(labels, 0, n_bins - 1)

    return labels, bin_edges


def split_into_trials(data, bins_per_trial):
    """
    Split time-series data into fixed-length trials.

    Drops incomplete last trial.

    For 2D: (n_neurons, n_bins) -> list of (n_neurons, bins_per_trial)
    For 1D: (n_bins,) -> list of (bins_per_trial,)
    """
    if data.ndim == 2:
        n_neurons, n_bins = data.shape
        n_trials = n_bins // bins_per_trial
        trials = []
        for t in range(n_trials):
            start = t * bins_per_trial
            end = start + bins_per_trial
            trials.append(data[:, start:end])
        return trials
    else:
        n_bins = len(data)
        n_trials = n_bins // bins_per_trial
        trials = []
        for t in range(n_trials):
            start = t * bins_per_trial
            end = start + bins_per_trial
            trials.append(data[start:end])
        return trials


def process_session(subject, session, show_processing=False, session_idx=0):
    """Process a single session and return neural, input, output data."""
    t0 = time.time()
    sess_dir = os.path.join(DATA_DIR, subject, session)

    # Load neural data
    F = np.load(os.path.join(sess_dir, 'suite2p', 'plane0', 'F.npy'))
    Fneu = np.load(os.path.join(sess_dir, 'suite2p', 'plane0', 'Fneu.npy'))
    ops = np.load(os.path.join(sess_dir, 'suite2p', 'plane0', 'ops.npy'),
                  allow_pickle=True).item()

    n_neurons, n_frames = F.shape
    print(f"  {subject}/{session}: {n_neurons} neurons, {n_frames} frames "
          f"({n_frames/FRAME_RATE:.0f}s)")

    # Load behavioral data
    me = np.load(os.path.join(sess_dir, 'move_deve', 'motion_energy_glob.npy')).astype(np.float64)
    ts = np.load(os.path.join(sess_dir, 'move_deve', 'tstamps.npy'))

    if len(me) != n_frames:
        print(f"    Camera frames: {len(me)} (missing {n_frames - len(me)}), interpolating...")

    # Step 1: Align motion energy to neural frames
    me_aligned = align_motion_energy(me, ts, n_frames)

    # Step 2: Compute dF/F
    dff = compute_dff(F, Fneu, ops)

    # Step 3: Bin data (10 frames)
    dff_binned = bin_data(dff, BIN_SIZE)
    me_binned = bin_data(me_aligned, BIN_SIZE)

    n_bins = dff_binned.shape[1]

    # Step 4: Discretize motion energy into 5 percentile bins
    me_discrete, bin_edges = discretize_motion_energy(me_binned)

    # Step 5: Create time input (seconds from session start)
    time_input = np.arange(n_bins) * BIN_DURATION_S

    # Step 6: Split into 60-second trials
    neural_trials = split_into_trials(dff_binned, BINS_PER_TRIAL)
    output_trials = split_into_trials(me_discrete, BINS_PER_TRIAL)
    input_trials = split_into_trials(time_input, BINS_PER_TRIAL)

    n_trials = len(neural_trials)
    print(f"    After binning: {n_bins} bins, {n_trials} trials of {TRIAL_DURATION_S}s "
          f"({BINS_PER_TRIAL} bins each)")

    # Check output distribution
    all_outputs = np.concatenate(output_trials)
    unique, counts = np.unique(all_outputs, return_counts=True)
    fracs = counts / counts.sum()
    print(f"    ME bin distribution: {dict(zip(unique.astype(int), np.round(fracs, 3)))}")

    elapsed = time.time() - t0
    print(f"    Processing time: {elapsed:.2f}s")

    # Plotting for --show-processing
    if show_processing:
        _plot_processing(subject, session, F, Fneu, dff, me, me_aligned,
                        dff_binned, me_binned, me_discrete, time_input,
                        neural_trials, output_trials, bin_edges, session_idx)

    # Format output trials as (1, n_timepoints) arrays
    output_trials_formatted = [o.reshape(1, -1).astype(np.int64) for o in output_trials]
    input_trials_formatted = [t_in.reshape(1, -1).astype(np.float32) for t_in in input_trials]
    neural_trials_formatted = [n.astype(np.float32) for n in neural_trials]

    return neural_trials_formatted, input_trials_formatted, output_trials_formatted, n_neurons


def _plot_processing(subject, session, F, Fneu, dff, me_raw, me_aligned,
                     dff_binned, me_binned, me_discrete, time_input,
                     neural_trials, output_trials, bin_edges, session_idx):
    """Plot processing steps for visual verification."""
    fig, axes = plt.subplots(6, 1, figsize=(16, 20))

    # 1. Raw F and Fneu for example neuron
    nrn = 0
    t_range = slice(0, 3000)  # first 100s
    axes[0].plot(np.arange(3000)/FRAME_RATE, F[nrn, t_range], alpha=0.7, label='F')
    axes[0].plot(np.arange(3000)/FRAME_RATE, Fneu[nrn, t_range], alpha=0.7, label='Fneu')
    axes[0].set_title(f'{subject}/{session} - Raw F and Fneu (neuron 0)')
    axes[0].legend()
    axes[0].set_xlabel('Time (s)')

    # 2. dF/F for example neuron
    axes[1].plot(np.arange(3000)/FRAME_RATE, dff[nrn, t_range], color='green')
    axes[1].set_title('dF/F (neuron 0)')
    axes[1].set_xlabel('Time (s)')

    # 3. Motion energy raw vs aligned
    n_show = min(3000, len(me_raw))
    axes[2].plot(np.arange(n_show)/FRAME_RATE, me_raw[:n_show], alpha=0.7, label='Raw ME')
    axes[2].plot(np.arange(3000)/FRAME_RATE, me_aligned[:3000], alpha=0.7, label='Aligned ME')
    axes[2].set_title('Motion Energy (raw vs aligned)')
    axes[2].legend()
    axes[2].set_xlabel('Time (s)')

    # 4. Binned dF/F and ME
    n_bins_show = 300  # first 100s
    bin_times = np.arange(n_bins_show) * BIN_DURATION_S
    ax4 = axes[3]
    ax4.plot(bin_times, dff_binned[nrn, :n_bins_show], color='green', label='dF/F binned')
    ax4b = ax4.twinx()
    ax4b.plot(bin_times, me_binned[:n_bins_show], color='gray', alpha=0.5, label='ME binned')
    ax4.set_title('Binned dF/F and Motion Energy')
    ax4.set_xlabel('Time (s)')
    ax4.legend(loc='upper left')
    ax4b.legend(loc='upper right')

    # 5. Discretized ME with bin edges
    axes[4].plot(bin_times, me_discrete[:n_bins_show], '.', markersize=1)
    axes[4].set_title(f'Discretized ME (5 bins, edges: {np.round(bin_edges, 1)})')
    axes[4].set_xlabel('Time (s)')
    axes[4].set_ylabel('Bin label')

    # 6. Example trial - neural activity raster
    if len(neural_trials) > 0:
        trial_data = neural_trials[0]
        n_show_neurons = min(50, trial_data.shape[0])
        im = axes[5].imshow(trial_data[:n_show_neurons], aspect='auto', cmap='RdBu_r',
                           vmin=-2, vmax=2, extent=[0, TRIAL_DURATION_S, n_show_neurons, 0])
        axes[5].set_title(f'Trial 0 - Neural activity (first {n_show_neurons} neurons)')
        axes[5].set_xlabel('Time (s)')
        axes[5].set_ylabel('Neuron')
        plt.colorbar(im, ax=axes[5], label='dF/F')

    fig.suptitle(f'Processing: {subject}/{session}', fontsize=14)
    fig.tight_layout()
    fig.savefig(f'processing_{subject}_{session}.png', dpi=100)
    plt.close(fig)
    print(f"    Saved processing plot: processing_{subject}_{session}.png")


# ============================================================================
# Main conversion
# ============================================================================

def convert_all(output_file, sample=False, show_processing=False):
    """Convert all data to decoder format."""
    t_start = time.time()

    all_neural = []
    all_input = []
    all_output = []
    all_subject_idx = []
    all_brain_region_idx = []

    subjects_list = SUBJECTS

    session_count = 0
    plot_count = 0
    max_plot_sessions = 2

    for subj_i, subject in enumerate(subjects_list):
        sessions = get_sessions(subject)
        print(f"\nSubject {subject} ({SUBJECT_MAP.get(subject, '')}): {len(sessions)} sessions")

        for sess_j, session in enumerate(sessions):
            if sample and session_count >= 2:
                break

            do_plot = show_processing and plot_count < max_plot_sessions

            neural, inp, out, n_neurons = process_session(
                subject, session, show_processing=do_plot, session_idx=session_count)

            if do_plot:
                plot_count += 1

            all_neural.append(neural)
            all_input.append(inp)
            all_output.append(out)
            all_subject_idx.append(subj_i)
            all_brain_region_idx.append(np.zeros(n_neurons, dtype=np.int64))

            session_count += 1

        if sample and session_count >= 2:
            break

    # Build output dictionary
    data = {
        'neural': all_neural,
        'input': all_input,
        'output': all_output,

        'subjects': subjects_list,
        'subject_idx': np.array(all_subject_idx, dtype=np.int64),

        'brain_regions': ['barrel_cortex'],
        'brain_region_idx': all_brain_region_idx,

        'input_names': ['time_s'],
        'output_names': ['motion_energy'],
        'output_values': [
            [f'ME_bin{i}' for i in range(N_OUTPUT_BINS)]
        ],

        'metadata': {
            'task_description': 'Decode motion energy (5 percentile bins) from barrel cortex neural activity during spontaneous behavior in developing mouse pups',
            'time_bin_size': BIN_DURATION_S * 1000,  # in ms
            'temporal_alignment_event': 'Session start (beginning of recording)',
            'off_start': 0.0,
            'off_end': TRIAL_DURATION_S,
            'frame_rate': FRAME_RATE,
            'bin_size_frames': BIN_SIZE,
            'trial_duration_s': TRIAL_DURATION_S,
            'bins_per_trial': BINS_PER_TRIAL,
            'neucoeff': NEUCOEFF,
            'n_output_bins': N_OUTPUT_BINS,
            'session_info': {
                'n_sessions': session_count,
                'n_subjects': len(set(all_subject_idx)),
            }
        }
    }

    # Print summary
    total_neurons = sum(len(br) for br in all_brain_region_idx)
    total_trials = sum(len(s) for s in all_neural)
    print(f"\n{'='*60}")
    print(f"Conversion complete:")
    print(f"  Subjects: {len(set(all_subject_idx))}")
    print(f"  Sessions: {session_count}")
    print(f"  Total trials: {total_trials}")
    print(f"  Total neurons (sum across sessions): {total_neurons}")
    print(f"  Time bin: {BIN_DURATION_S*1000:.1f} ms")
    print(f"  Bins per trial: {BINS_PER_TRIAL}")

    # Save
    print(f"\nSaving to {output_file}...")
    with open(output_file, 'wb') as f:
        pickle.dump(data, f, protocol=4)

    file_size = os.path.getsize(output_file) / (1024 * 1024)
    print(f"Saved: {file_size:.1f} MB")

    elapsed = time.time() - t_start
    print(f"Total time: {elapsed:.1f}s")

    return data


# ============================================================================
# Entry point
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert Track2p data to decoder format')
    parser.add_argument('output', type=str, help='Output pickle file path')
    parser.add_argument('--full', action='store_true', default=True,
                        help='Process all sessions (default)')
    parser.add_argument('--sample', action='store_true',
                        help='Process only 2 sessions for testing')
    parser.add_argument('--show-processing', action='store_true',
                        help='Plot processing steps for up to 2 sessions')

    args = parser.parse_args()

    sample_mode = args.sample
    if sample_mode:
        print("Running in SAMPLE mode (2 sessions only)")
    else:
        print("Running in FULL mode (all sessions)")

    convert_all(args.output, sample=sample_mode, show_processing=args.show_processing)
