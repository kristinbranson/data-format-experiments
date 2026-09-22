#!/usr/bin/env python3
"""Convert Track2p data to decoder-compatible format.

Usage:
    python -u convert_data.py <outpicklefile> [--full|--sample] [--show-processing]
"""

import os
import sys
import time
import argparse
import pickle
import numpy as np
import torch
from suite2p.extraction.dcnv import preprocess as s2p_preprocess

# ============================================================
# Constants
# ============================================================
DATA_DIR = '/app/data'
BIN_SIZE = 10          # frames to average per bin
FS = 30.0              # imaging frame rate (Hz)
TIME_PER_BIN = BIN_SIZE / FS  # seconds per bin (0.3333s)
TRIAL_DURATION = 60.0  # seconds per trial
BINS_PER_TRIAL = int(TRIAL_DURATION / TIME_PER_BIN)  # 180
N_ME_BINS = 5          # number of motion energy percentile bins
NEUCOEFF = 0.7         # neuropil coefficient
BRAIN_REGION = 'barrel_cortex'

SUBJECTS = ['jm031', 'jm032', 'jm038', 'jm039', 'jm040', 'jm046']


def get_sessions(subject_dir):
    """Get sorted list of session directories for a subject."""
    sessions = sorted([d for d in os.listdir(subject_dir)
                       if os.path.isdir(os.path.join(subject_dir, d))])
    return sessions


def load_session_data(subject_dir, session_name):
    """Load neural and behavioral data for one session."""
    session_dir = os.path.join(subject_dir, session_name)
    s2p_dir = os.path.join(session_dir, 'suite2p', 'plane0')
    move_dir = os.path.join(session_dir, 'move_deve')
    
    F = np.load(os.path.join(s2p_dir, 'F.npy'))        # (n_neurons, n_frames)
    Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))  # (n_neurons, n_frames)
    ops = np.load(os.path.join(s2p_dir, 'ops.npy'), allow_pickle=True).item()
    
    me = np.load(os.path.join(move_dir, 'motion_energy_glob.npy'))  # (n_frames_me,)
    
    return F, Fneu, ops, me


def compute_dff(F, Fneu, ops):
    """Compute dF/F using Suite2p's baseline correction.
    
    Following the paper: neuropil correction then maximin baseline subtraction.
    """
    neucoeff = ops.get('neucoeff', NEUCOEFF)
    baseline = ops.get('baseline', 'maximin')
    win_baseline = ops.get('win_baseline', 60.0)
    sig_baseline = ops.get('sig_baseline', 10.0)
    fs = ops.get('fs', FS)
    
    # Step 1: Neuropil correction
    Fc = F - neucoeff * Fneu
    Fc = Fc.astype(np.float32)
    
    # Step 2: Baseline correction using Suite2p's preprocess
    device = torch.device('cpu')
    dFF = s2p_preprocess(Fc, baseline, win_baseline, sig_baseline, fs, device=device)
    
    return dFF


def align_me_to_neural(me, n_neural_frames):
    """Align motion energy to neural frame count.
    
    ME and neural data are synchronized (camera triggered by microscope).
    Handle minor length mismatches by padding or truncating.
    """
    n_me = len(me)
    if n_me == n_neural_frames:
        return me.astype(np.float64)
    elif n_me < n_neural_frames:
        # Pad ME with last value (or 0)
        padded = np.zeros(n_neural_frames, dtype=np.float64)
        padded[:n_me] = me.astype(np.float64)
        padded[n_me:] = me[-1]  # repeat last value
        return padded
    else:
        # Truncate ME
        return me[:n_neural_frames].astype(np.float64)


def bin_data(data, bin_size):
    """Average data in non-overlapping bins along the last axis.
    
    Args:
        data: array of shape (..., n_frames)
        bin_size: number of frames per bin
    Returns:
        binned: array of shape (..., n_bins)
    """
    n_frames = data.shape[-1]
    n_bins = n_frames // bin_size
    # Truncate to exact multiple of bin_size
    truncated = data[..., :n_bins * bin_size]
    # Reshape and average
    new_shape = truncated.shape[:-1] + (n_bins, bin_size)
    binned = truncated.reshape(new_shape).mean(axis=-1)
    return binned


def discretize_me(me_binned, n_bins=N_ME_BINS):
    """Discretize motion energy into equal-percentile bins.
    
    Args:
        me_binned: 1D array of binned motion energy for one session
        n_bins: number of bins
    Returns:
        me_discrete: 1D array of bin indices (0 to n_bins-1)
        bin_edges: percentile edges used
    """
    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(me_binned, percentiles)
    
    # Use digitize to assign bins
    # np.digitize returns 1-based indices; we want 0-based
    me_discrete = np.digitize(me_binned, bin_edges[1:-1], right=False)
    # Clip to valid range [0, n_bins-1]
    me_discrete = np.clip(me_discrete, 0, n_bins - 1)
    
    return me_discrete, bin_edges


def split_into_trials(data, bins_per_trial):
    """Split binned data into trials.
    
    Args:
        data: array of shape (..., n_total_bins)
        bins_per_trial: number of bins per trial
    Returns:
        list of arrays, each of shape (..., bins_per_trial)
    """
    n_total_bins = data.shape[-1]
    n_trials = n_total_bins // bins_per_trial
    trials = []
    for t in range(n_trials):
        start = t * bins_per_trial
        end = start + bins_per_trial
        trials.append(data[..., start:end])
    return trials


def process_session(subject, subject_dir, session_name, session_idx_global, show_processing=False):
    """Process a single session and return trial data."""
    t0 = time.time()
    
    # Load data
    F, Fneu, ops, me = load_session_data(subject_dir, session_name)
    n_neurons, n_frames = F.shape
    t_load = time.time() - t0
    
    # Compute dF/F
    t1 = time.time()
    dFF = compute_dff(F, Fneu, ops)
    t_dff = time.time() - t1
    
    # Align ME to neural frames
    me_aligned = align_me_to_neural(me, n_frames)
    
    # Bin neural data and ME (10 frames per bin)
    t2 = time.time()
    dFF_binned = bin_data(dFF, BIN_SIZE)  # (n_neurons, n_bins)
    me_binned = bin_data(me_aligned.reshape(1, -1), BIN_SIZE).flatten()  # (n_bins,)
    t_bin = time.time() - t2
    
    # Create time axis (seconds from session start)
    n_bins = dFF_binned.shape[1]
    # Time at center of each bin
    time_axis = (np.arange(n_bins) + 0.5) * TIME_PER_BIN  # seconds
    
    # Discretize ME into 5 equal-percentile bins for this session
    me_discrete, bin_edges = discretize_me(me_binned, N_ME_BINS)
    
    # Split into 60-second trials
    neural_trials = split_into_trials(dFF_binned, BINS_PER_TRIAL)
    n_trials = len(neural_trials)
    
    # For each trial, compute input (time from session start) and output (discretized ME)
    input_trials = []
    output_trials = []
    for t in range(n_trials):
        start_bin = t * BINS_PER_TRIAL
        end_bin = start_bin + BINS_PER_TRIAL
        
        # Input: time elapsed from session start (seconds)
        trial_time = time_axis[start_bin:end_bin]  # (180,)
        input_trials.append(trial_time.reshape(1, -1))  # (1, 180)
        
        # Output: discretized motion energy
        trial_me = me_discrete[start_bin:end_bin]  # (180,)
        output_trials.append(trial_me.reshape(1, -1))  # (1, 180)
    
    t_total = time.time() - t0
    print(f"  {subject}/{session_name}: {n_neurons} neurons, {n_frames} frames, "
          f"{n_bins} bins, {n_trials} trials | "
          f"load={t_load:.1f}s dff={t_dff:.1f}s bin={t_bin:.2f}s total={t_total:.1f}s")
    
    if show_processing:
        plot_processing(subject, session_name, F, Fneu, dFF, me_aligned, 
                       dFF_binned, me_binned, me_discrete, time_axis, bin_edges,
                       neural_trials, input_trials, output_trials)
    
    return neural_trials, input_trials, output_trials, n_neurons, me_binned, bin_edges


def plot_processing(subject, session_name, F, Fneu, dFF, me_aligned,
                   dFF_binned, me_binned, me_discrete, time_axis, bin_edges,
                   neural_trials, input_trials, output_trials):
    """Plot processing steps for visual verification."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    session_id = f"{subject}_{session_name}"
    fig, axes = plt.subplots(5, 2, figsize=(20, 20))
    fig.suptitle(f"Processing: {session_id}", fontsize=16)
    
    n_neurons = F.shape[0]
    example_neuron = min(5, n_neurons - 1)
    
    # Row 1: Raw F and Fneu for example neuron
    ax = axes[0, 0]
    frames = np.arange(min(3000, F.shape[1]))
    ax.plot(frames/FS, F[example_neuron, frames], label='F', alpha=0.7)
    ax.plot(frames/FS, Fneu[example_neuron, frames], label='Fneu', alpha=0.7)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Fluorescence')
    ax.set_title(f'Raw F and Fneu (neuron {example_neuron}, first 100s)')
    ax.legend()
    
    # Row 1: dF/F for example neuron
    ax = axes[0, 1]
    ax.plot(frames/FS, dFF[example_neuron, frames], color='green', alpha=0.7)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('dF/F')
    ax.set_title(f'dF/F (neuron {example_neuron}, first 100s)')
    
    # Row 2: Motion energy raw and binned
    ax = axes[1, 0]
    t_raw = np.arange(len(me_aligned)) / FS
    ax.plot(t_raw[:3000], me_aligned[:3000], alpha=0.7, label='raw ME')
    bins_100s = int(100 / TIME_PER_BIN)
    ax.plot(time_axis[:bins_100s], me_binned[:bins_100s], 'r-', linewidth=2, label='binned ME')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Motion Energy')
    ax.set_title('Motion Energy (first 100s)')
    ax.legend()
    
    # Row 2: ME histogram and percentile bins
    ax = axes[1, 1]
    ax.hist(me_binned, bins=100, alpha=0.7, density=True)
    for edge in bin_edges:
        ax.axvline(edge, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Motion Energy')
    ax.set_ylabel('Density')
    ax.set_title(f'ME Distribution with {N_ME_BINS} percentile bin edges')
    
    # Row 3: Discretized ME
    ax = axes[2, 0]
    ax.plot(time_axis[:bins_100s], me_discrete[:bins_100s], 'k-', alpha=0.7)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('ME Bin')
    ax.set_title('Discretized Motion Energy (first 100s)')
    ax.set_yticks(range(N_ME_BINS))
    
    # Row 3: ME bin distribution
    ax = axes[2, 1]
    unique, counts = np.unique(me_discrete, return_counts=True)
    ax.bar(unique, counts / len(me_discrete))
    ax.set_xlabel('ME Bin')
    ax.set_ylabel('Fraction')
    ax.set_title('ME Bin Distribution (should be ~uniform)')
    ax.axhline(1/N_ME_BINS, color='red', linestyle='--', label='expected')
    ax.legend()
    
    # Row 4: Neural raster (binned dF/F)
    ax = axes[3, 0]
    from scipy.stats import zscore
    dFF_z = zscore(dFF_binned, axis=1)
    ax.imshow(dFF_z[:, :bins_100s], aspect='auto', cmap='Greys', vmin=0, vmax=2,
             extent=[0, 100, dFF_z.shape[0], 0])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Neuron')
    ax.set_title('Binned dF/F (z-scored, first 100s)')
    
    # Row 4: Trial structure visualization
    ax = axes[3, 1]
    n_trials = len(neural_trials)
    for t in range(min(3, n_trials)):
        trial_time = input_trials[t].flatten()
        trial_me = output_trials[t].flatten()
        ax.plot(trial_time, trial_me + t * 6, label=f'Trial {t}')
    ax.set_xlabel('Time from session start (s)')
    ax.set_ylabel('ME Bin (offset)')
    ax.set_title('Trial structure: input (time) vs output (ME bin)')
    ax.legend()
    
    # Row 5: Neural activity aligned with ME for one trial
    ax = axes[4, 0]
    trial_neural = neural_trials[0]  # (n_neurons, 180)
    trial_me_vals = output_trials[0].flatten()
    trial_time_vals = input_trials[0].flatten()
    # Plot mean neural activity and ME
    ax.plot(trial_time_vals, np.mean(trial_neural, axis=0), 'b-', label='mean dF/F')
    ax2 = ax.twinx()
    ax2.plot(trial_time_vals, trial_me_vals, 'r-', label='ME bin', alpha=0.7)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Mean dF/F', color='b')
    ax2.set_ylabel('ME Bin', color='r')
    ax.set_title('Trial 0: Neural + ME alignment')
    
    # Row 5: Summary stats
    ax = axes[4, 1]
    ax.axis('off')
    stats_text = (
        f"Session: {session_id}\n"
        f"Neurons: {F.shape[0]}\n"
        f"Frames: {F.shape[1]}\n"
        f"Bins: {dFF_binned.shape[1]}\n"
        f"Trials: {n_trials}\n"
        f"Bins/trial: {BINS_PER_TRIAL}\n"
        f"Time/bin: {TIME_PER_BIN*1000:.1f} ms\n"
        f"ME bin edges: {[f'{e:.1f}' for e in bin_edges]}\n"
        f"dF/F range: [{dFF.min():.1f}, {dFF.max():.1f}]\n"
        f"dF/F mean: {dFF.mean():.2f}"
    )
    ax.text(0.1, 0.5, stats_text, transform=ax.transAxes, fontsize=12,
           verticalalignment='center', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig(f'/app/processing_{session_id}.png', dpi=100, bbox_inches='tight')
    plt.close()
    print(f"    Saved processing plot: /app/processing_{session_id}.png")


def main():
    parser = argparse.ArgumentParser(description='Convert Track2p data to decoder format')
    parser.add_argument('outfile', help='Output pickle file path')
    parser.add_argument('--full', action='store_true', default=True, help='Process all sessions (default)')
    parser.add_argument('--sample', action='store_true', help='Process only 2 sessions for testing')
    parser.add_argument('--show-processing', action='store_true', help='Plot processing visualizations')
    args = parser.parse_args()
    
    t_start = time.time()
    
    # Collect all sessions
    all_sessions = []  # list of (subject, session_name)
    for subject in SUBJECTS:
        subject_dir = os.path.join(DATA_DIR, subject)
        sessions = get_sessions(subject_dir)
        for s in sessions:
            all_sessions.append((subject, s))
    
    print(f"Found {len(all_sessions)} sessions across {len(SUBJECTS)} subjects")
    
    if args.sample:
        # Select 2 sessions from different subjects
        sample_sessions = [all_sessions[0], all_sessions[14]]  # jm031 first, jm038 first
        all_sessions = sample_sessions
        print(f"Sample mode: processing {len(all_sessions)} sessions")
    
    # Initialize data structure
    neural_all = []  # list of sessions, each is list of trials
    input_all = []
    output_all = []
    subject_idx_list = []
    brain_region_idx_all = []
    
    subjects_seen = []
    
    for sess_idx, (subject, session_name) in enumerate(all_sessions):
        subject_dir = os.path.join(DATA_DIR, subject)
        
        if subject not in subjects_seen:
            subjects_seen.append(subject)
        subj_idx = subjects_seen.index(subject)
        
        neural_trials, input_trials, output_trials, n_neurons, me_binned, bin_edges = \
            process_session(subject, subject_dir, session_name, sess_idx, 
                          show_processing=args.show_processing)
        
        neural_all.append(neural_trials)
        input_all.append(input_trials)
        output_all.append(output_trials)
        subject_idx_list.append(subj_idx)
        brain_region_idx_all.append(np.zeros(n_neurons, dtype=np.int64))  # all barrel cortex
    
    # Build output dictionary
    data = {
        'neural': neural_all,
        'input': input_all,
        'output': output_all,
        'subjects': subjects_seen,
        'subject_idx': np.array(subject_idx_list, dtype=np.int64),
        'brain_regions': [BRAIN_REGION],
        'brain_region_idx': brain_region_idx_all,
        'input_names': ['time_in_session'],
        'output_names': ['motion_energy'],
        'output_values': [
            ['ME_bin_0', 'ME_bin_1', 'ME_bin_2', 'ME_bin_3', 'ME_bin_4']
        ],
        'metadata': {
            'task_description': 'Decode motion energy from barrel cortex neural activity in developing mice',
            'time_bin_size': TIME_PER_BIN * 1000,  # in ms (333.33 ms)
            'temporal_alignment_event': 'session_start',
            'off_start': 0.0,
            'off_end': None,
            'trial_duration_s': TRIAL_DURATION,
            'bin_size_frames': BIN_SIZE,
            'frame_rate_hz': FS,
            'n_me_bins': N_ME_BINS,
            'neural_processing': 'dF/F via Suite2p neuropil correction (neucoeff=0.7) + maximin baseline subtraction, binned by averaging 10 frames',
            'me_processing': 'Global motion energy (sum of squared pixel-wise frame differences), binned by averaging 10 frames, discretized into 5 equal-percentile bins per session',
            'reference': 'Majnik et al. 2025, Track2p, eLife',
        }
    }
    
    # Print summary
    print(f"\n=== Summary ===")
    print(f"Subjects: {subjects_seen}")
    print(f"Sessions: {len(neural_all)}")
    total_trials = sum(len(s) for s in neural_all)
    print(f"Total trials: {total_trials}")
    for i, (subject, session_name) in enumerate(all_sessions):
        nt = len(neural_all[i])
        nn = neural_all[i][0].shape[0] if nt > 0 else 0
        print(f"  Session {i} ({subject}/{session_name}): {nn} neurons, {nt} trials")
    
    # Save
    print(f"\nSaving to {args.outfile}...")
    with open(args.outfile, 'wb') as f:
        pickle.dump(data, f, protocol=4)
    
    file_size = os.path.getsize(args.outfile) / (1024 * 1024)
    print(f"Saved: {file_size:.1f} MB")
    
    t_total = time.time() - t_start
    print(f"Total time: {t_total:.1f}s")


if __name__ == '__main__':
    main()
