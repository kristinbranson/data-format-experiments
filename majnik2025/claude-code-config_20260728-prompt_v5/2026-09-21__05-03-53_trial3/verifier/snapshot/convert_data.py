#!/usr/bin/env python3
"""
Convert Track2p longitudinal calcium imaging data to decoder-compatible format.

Usage:
    python -u convert_data.py <output.pkl> [--sample] [--full] [--show-processing]

Data from: Majnik et al. 2025 - "Longitudinal tracking of neuronal activity
from the same cells in the developing brain using Track2p"
"""

import os
import sys
import time
import argparse
import pickle
import numpy as np
import warnings

DATA_DIR = '/app/data'
FRAME_RATE = 30  # Hz
BIN_SIZE = 10    # frames per bin (as in paper's decoding analysis)
TIME_BIN_MS = BIN_SIZE / FRAME_RATE * 1000  # 333.33 ms
TRIAL_DURATION_S = 60  # seconds per trial
TRIAL_BINS = int(TRIAL_DURATION_S / (BIN_SIZE / FRAME_RATE))  # 180 bins per trial
N_ME_BINS = 5  # number of motion energy percentile bins

# Suite2p default parameters (from ops.npy)
NEUCOEFF = 0.7
WIN_BASELINE = 60.0  # seconds
SIG_BASELINE = 10.0  # frames (std of Gaussian)
BASELINE_METHOD = 'maximin'


def get_subjects_and_sessions(data_dir):
    """Get all subjects and their session directories."""
    subjects = sorted([d for d in os.listdir(data_dir)
                       if os.path.isdir(os.path.join(data_dir, d))])
    subject_sessions = {}
    for subj in subjects:
        subj_dir = os.path.join(data_dir, subj)
        sessions = sorted([d for d in os.listdir(subj_dir)
                          if os.path.isdir(os.path.join(subj_dir, d))])
        subject_sessions[subj] = sessions
    return subjects, subject_sessions


def compute_dfof(F, Fneu, fs=FRAME_RATE):
    """
    Compute dF/F using Suite2p default parameters.

    1. Neuropil subtraction: Fc = F - 0.7 * Fneu
    2. Baseline correction using maximin filter (Suite2p preprocess)

    Paper: "We used baseline corrected fluorescence traces as our dF/F
    (using the default Suite2p parameters)"
    """
    import torch
    from torch.nn.functional import conv1d, max_pool1d, pad

    # Neuropil subtraction
    Fc = F - NEUCOEFF * Fneu
    Fc = Fc.astype(np.float32)

    # Maximin baseline (Suite2p default)
    win = int(WIN_BASELINE * fs)
    if win % 2 == 0:
        win += 1

    ncells, n_frames = Fc.shape
    device = torch.device('cpu')  # CPU for reliability

    gwid = int(np.round(SIG_BASELINE * 3))
    gaussian = torch.exp(-torch.arange(-gwid, gwid + 1, 1, device=device, dtype=torch.float32)**2 /
                         (2 * SIG_BASELINE**2))
    gaussian /= gaussian.sum()

    batch_size = 100
    Flow = np.zeros_like(Fc)
    n_batches = int(np.ceil(ncells / batch_size))

    for n in range(n_batches):
        nstart = n * batch_size
        nend = min((n + 1) * batch_size, ncells)
        data = torch.from_numpy(Fc[nstart:nend]).to(device, dtype=torch.float32)

        # Gaussian smoothing
        data = pad(data, (gwid, gwid), 'replicate')
        data = conv1d(data.unsqueeze(1), gaussian.unsqueeze(0).unsqueeze(0), padding=0)

        # Min filter then max filter
        data = pad(data, (win // 2, win // 2), 'replicate')
        data = -max_pool1d(-data, kernel_size=win, stride=1, padding=0)

        data = pad(data, (win // 2, win // 2), 'replicate')
        data = max_pool1d(data, kernel_size=win, stride=1, padding=0)

        Flow[nstart:nend] = data.squeeze(1).cpu().numpy()

    # Baseline subtraction
    dff = Fc - Flow
    return dff


def bin_data(data, bin_size=BIN_SIZE, axis=-1):
    """Average data in bins along specified axis."""
    n = data.shape[axis]
    n_bins = n // bin_size
    # Truncate to exact multiple of bin_size
    slices = [slice(None)] * data.ndim
    slices[axis] = slice(0, n_bins * bin_size)
    data_trunc = data[tuple(slices)]

    if axis == -1 or axis == data.ndim - 1:
        new_shape = data_trunc.shape[:-1] + (n_bins, bin_size)
        return data_trunc.reshape(new_shape).mean(axis=-1)
    elif axis == 0:
        new_shape = (n_bins, bin_size) + data_trunc.shape[1:]
        return data_trunc.reshape(new_shape).mean(axis=1)
    else:
        raise ValueError(f"Unsupported axis {axis}")


def interpolate_motion_energy(me, tstamps, n_neural_frames):
    """
    Interpolate motion energy to match neural frame count.
    Handles missing camera frames by linear interpolation.
    """
    if len(me) == n_neural_frames:
        return me.astype(np.float64)

    # Create indices for the available ME frames
    # tstamps contains timestamps for each camera frame
    # We need to interpolate to get n_neural_frames values
    me_float = me.astype(np.float64)

    # Map camera frames to neural frame indices using tstamps
    # tstamps are cumulative times; neural frames are at regular intervals
    # Since camera is triggered by microscope, frame indices should be close to 1:1
    # but some frames may be missing

    # Simple approach: use numpy interpolation
    # Camera frame indices map to neural indices approximately 1:1
    # Create target indices (0 to n_neural_frames-1)
    neural_indices = np.arange(n_neural_frames)

    # Source indices: evenly spaced from 0 to n_neural_frames-1
    # (camera frames should map roughly linearly to neural frames)
    camera_indices = np.linspace(0, n_neural_frames - 1, len(me))

    me_interp = np.interp(neural_indices, camera_indices, me_float)
    return me_interp


def discretize_motion_energy(me_binned, n_bins=N_ME_BINS):
    """
    Discretize motion energy into equal-percentile bins per session.
    Returns bin indices (0 to n_bins-1) and bin edges.
    """
    # Compute percentile edges
    percentiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(me_binned, percentiles)

    # Handle duplicate edges (when many values are the same)
    # np.digitize with right=False: edges[i-1] <= x < edges[i]
    binned = np.digitize(me_binned, edges[1:-1], right=False)
    # Clip to valid range
    binned = np.clip(binned, 0, n_bins - 1)

    return binned, edges


def split_into_trials(data, trial_length, axis=-1):
    """Split data into trials of given length along axis."""
    n = data.shape[axis]
    n_trials = n // trial_length
    trials = []
    for i in range(n_trials):
        slices = [slice(None)] * data.ndim
        slices[axis] = slice(i * trial_length, (i + 1) * trial_length)
        trials.append(data[tuple(slices)])
    return trials


def process_session(subj, sess_name, data_dir, show_processing=False, session_idx=0):
    """Process a single session and return neural, input, output data."""
    sess_dir = os.path.join(data_dir, subj, sess_name)
    s2p_dir = os.path.join(sess_dir, 'suite2p', 'plane0')
    move_dir = os.path.join(sess_dir, 'move_deve')

    t0 = time.time()

    # Load neural data
    F = np.load(os.path.join(s2p_dir, 'F.npy'))
    Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
    n_neurons, n_frames = F.shape

    # Load motion energy
    me = np.load(os.path.join(move_dir, 'motion_energy_glob.npy'))
    tstamps = np.load(os.path.join(move_dir, 'tstamps.npy'))

    t_load = time.time() - t0

    # Interpolate ME if needed
    t0 = time.time()
    me_interp = interpolate_motion_energy(me, tstamps, n_frames)
    t_interp = time.time() - t0

    # Compute dF/F
    t0 = time.time()
    dff = compute_dfof(F, Fneu)
    t_dfof = time.time() - t0

    # Bin neural data and ME by 10 frames
    t0 = time.time()
    dff_binned = bin_data(dff, BIN_SIZE, axis=1)  # (n_neurons, n_timebins)
    me_binned = bin_data(me_interp.reshape(1, -1), BIN_SIZE, axis=1).flatten()  # (n_timebins,)
    t_bin = time.time() - t0

    n_timebins = dff_binned.shape[1]

    # Discretize motion energy into 5 equal-percentile bins
    me_discrete, me_edges = discretize_motion_energy(me_binned)

    # Create time input (seconds from session start)
    time_s = np.arange(n_timebins) * (BIN_SIZE / FRAME_RATE)  # seconds

    # Split into 60-second trials
    n_trials = n_timebins // TRIAL_BINS

    neural_trials = split_into_trials(dff_binned, TRIAL_BINS, axis=1)
    me_trials = split_into_trials(me_discrete.reshape(1, -1), TRIAL_BINS, axis=1)
    time_trials = split_into_trials(time_s.reshape(1, -1), TRIAL_BINS, axis=1)

    print(f"  {subj}/{sess_name}: {n_neurons} neurons, {n_frames} frames -> "
          f"{n_timebins} bins -> {n_trials} trials | "
          f"load={t_load:.1f}s dfof={t_dfof:.1f}s bin={t_bin:.2f}s")

    if show_processing and session_idx < 2:
        plot_processing(subj, sess_name, F, Fneu, dff, me_interp, me_binned,
                       me_discrete, me_edges, dff_binned, time_s, session_idx)

    return {
        'neural': neural_trials,
        'input': [t.astype(np.float32) for t in time_trials],
        'output': [m.astype(np.int64) for m in me_trials],
        'n_neurons': n_neurons,
        'n_trials': n_trials,
    }


def plot_processing(subj, sess_name, F, Fneu, dff, me_interp, me_binned,
                   me_discrete, me_edges, dff_binned, time_s, session_idx):
    """Plot processing steps for visual verification."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(6, 1, figsize=(16, 20))

    # 1. Raw F for example neuron
    nrn = 0
    axes[0].plot(F[nrn, :3000], 'b', alpha=0.7, label='F')
    axes[0].plot(Fneu[nrn, :3000], 'r', alpha=0.5, label='Fneu')
    axes[0].set_title(f'{subj}/{sess_name} - Raw F and Fneu (neuron {nrn}, first 3000 frames)')
    axes[0].legend()

    # 2. dF/F for example neuron
    axes[1].plot(dff[nrn, :3000], 'g')
    axes[1].set_title(f'dF/F (neuron {nrn}, first 3000 frames)')

    # 3. Motion energy (continuous)
    axes[2].plot(me_interp[:3000], 'gray')
    axes[2].set_title('Motion energy (first 3000 frames)')

    # 4. Binned dF/F and ME overlay
    n_show = 300  # first 300 bins = 100s
    ax4 = axes[3]
    ax4.plot(time_s[:n_show], dff_binned[nrn, :n_show], 'g', alpha=0.7, label='dF/F (binned)')
    ax4b = ax4.twinx()
    ax4b.plot(time_s[:n_show], me_binned[:n_show], 'gray', alpha=0.5, label='ME (binned)')
    ax4.set_title(f'Binned dF/F and ME (first {n_show} bins)')
    ax4.set_xlabel('Time (s)')

    # 5. ME discretization
    axes[4].plot(time_s[:n_show], me_discrete[:n_show], 'k', alpha=0.7)
    for i, edge in enumerate(me_edges):
        axes[4].axhline(y=i - 0.5, color='r', linestyle='--', alpha=0.3)
    axes[4].set_title(f'ME discretized ({N_ME_BINS} bins)')
    axes[4].set_xlabel('Time (s)')
    axes[4].set_ylabel('Bin')

    # 6. ME bin distribution
    unique, counts = np.unique(me_discrete, return_counts=True)
    axes[5].bar(unique, counts / counts.sum())
    axes[5].set_title('ME bin distribution (should be ~0.2 each)')
    axes[5].set_xlabel('Bin')
    axes[5].set_ylabel('Fraction')

    plt.tight_layout()
    plt.savefig(f'/app/processing_{subj}_{sess_name}.png', dpi=100)
    plt.close()
    print(f"  Saved processing plot: processing_{subj}_{sess_name}.png")


def convert_data(data_dir, output_path, sample=False, show_processing=False):
    """Main conversion function."""
    t_total_start = time.time()

    subjects, subject_sessions = get_subjects_and_sessions(data_dir)
    print(f"Found {len(subjects)} subjects: {subjects}")

    if sample:
        # Process only 2 sessions for testing
        subjects_to_process = subjects[:1]  # first subject
        sessions_limit = 2
        print(f"SAMPLE MODE: Processing {sessions_limit} sessions from {subjects_to_process}")
    else:
        subjects_to_process = subjects
        sessions_limit = None

    all_neural = []
    all_input = []
    all_output = []
    all_subject_idx = []
    all_brain_region_idx = []

    session_count = 0

    for subj in subjects_to_process:
        sessions = subject_sessions[subj]
        if sessions_limit:
            sessions = sessions[:sessions_limit]

        subj_idx = subjects.index(subj)

        for sess_name in sessions:
            result = process_session(subj, sess_name, data_dir,
                                    show_processing=show_processing,
                                    session_idx=session_count)

            all_neural.append(result['neural'])
            all_input.append(result['input'])
            all_output.append(result['output'])
            all_subject_idx.append(subj_idx)
            all_brain_region_idx.append(np.zeros(result['n_neurons'], dtype=np.int64))

            session_count += 1

    # Build output dictionary
    data = {
        'neural': all_neural,
        'input': all_input,
        'output': all_output,
        'subjects': subjects,
        'subject_idx': np.array(all_subject_idx, dtype=np.int64),
        'brain_regions': ['barrel_cortex'],
        'brain_region_idx': all_brain_region_idx,
        'input_names': ['time_s'],
        'output_names': ['motion_energy'],
        'output_values': [
            [f'ME_bin_{i}' for i in range(N_ME_BINS)]
        ],
        'metadata': {
            'task_description': 'Decode motion energy from neural activity in developing mouse barrel cortex',
            'time_bin_size': TIME_BIN_MS,
            'temporal_alignment_event': 'Session start (spontaneous activity, no task events)',
            'off_start': None,
            'off_end': None,
            'frame_rate': FRAME_RATE,
            'bin_size_frames': BIN_SIZE,
            'trial_duration_s': TRIAL_DURATION_S,
            'n_me_bins': N_ME_BINS,
            'neural_signal': 'dF/F (Suite2p baseline corrected, neucoeff=0.7, maximin baseline)',
            'source': 'Majnik et al. 2025 - Track2p',
        }
    }

    # Print summary
    total_neurons = sum(len(br) for br in all_brain_region_idx)
    total_trials = sum(len(s) for s in all_neural)
    print(f"\n=== Summary ===")
    print(f"Sessions: {session_count}")
    print(f"Total trials: {total_trials}")
    print(f"Subjects: {len(subjects)}")
    print(f"Brain regions: {data['brain_regions']}")

    for i, subj in enumerate(subjects):
        sess_indices = [j for j, si in enumerate(all_subject_idx) if si == i]
        if sess_indices:
            n_neurons = all_brain_region_idx[sess_indices[0]].shape[0]
            n_trials_subj = sum(len(all_neural[j]) for j in sess_indices)
            print(f"  {subj}: {len(sess_indices)} sessions, {n_neurons} neurons, {n_trials_subj} trials")

    # Save
    t0 = time.time()
    with open(output_path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    t_save = time.time() - t0

    file_size = os.path.getsize(output_path) / (1024 * 1024)
    t_total = time.time() - t_total_start
    print(f"\nSaved to {output_path} ({file_size:.1f} MB)")
    print(f"Total time: {t_total:.1f}s (save: {t_save:.1f}s)")

    return data


def main():
    parser = argparse.ArgumentParser(description='Convert Track2p data to decoder format')
    parser.add_argument('output', help='Output pickle file path')
    parser.add_argument('--sample', action='store_true', help='Process only 2 sessions')
    parser.add_argument('--full', action='store_true', help='Process all sessions (default)')
    parser.add_argument('--show-processing', action='store_true', help='Save processing plots')
    args = parser.parse_args()

    convert_data(DATA_DIR, args.output, sample=args.sample,
                 show_processing=args.show_processing)


if __name__ == '__main__':
    main()
