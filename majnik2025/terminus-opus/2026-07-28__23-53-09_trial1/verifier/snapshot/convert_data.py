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
from suite2p.extraction.dcnv import preprocess

# Constants
DATA_DIR = 'data'
FS = 30.0  # imaging rate in Hz
BIN_SIZE = 10  # number of frames to average
TRIAL_DURATION_SEC = 120  # 2 minutes per trial
TRIAL_FRAMES_RAW = int(TRIAL_DURATION_SEC * FS)  # 3600 raw frames per trial
TRIAL_FRAMES_BINNED = TRIAL_FRAMES_RAW // BIN_SIZE  # 360 binned frames per trial
NEUCOEFF = 0.7
WIN_BASELINE = 60.0
SIG_BASELINE = 10.0
N_BINS = 5  # number of percentile bins for motion energy
TIME_BIN_MS = (BIN_SIZE / FS) * 1000  # 333.33 ms


def get_subjects_and_sessions(data_dir):
    """Get all subjects and their sessions."""
    subjects = sorted([d for d in os.listdir(data_dir) 
                       if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('jm')])
    sessions = {}
    for subj in subjects:
        subj_path = os.path.join(data_dir, subj)
        sess_list = sorted([s for s in os.listdir(subj_path) 
                           if os.path.isdir(os.path.join(subj_path, s))])
        sessions[subj] = sess_list
    return subjects, sessions


def compute_dff(F, Fneu, neucoeff=NEUCOEFF, win_baseline=WIN_BASELINE, 
                sig_baseline=SIG_BASELINE, fs=FS):
    """Compute dF/F using Suite2p default method.
    
    Fc = F - neucoeff * Fneu, then maximin baseline subtraction.
    """
    Fc = F - neucoeff * Fneu
    device = torch.device('cpu')
    dff = preprocess(Fc.copy(), 'maximin', win_baseline, sig_baseline, fs, device=device)
    return dff


def align_motion_energy(me, n_neural_frames, tstamps):
    """Align motion energy to neural frames, handling missing camera frames.
    
    When ME has fewer frames than neural data, use inter-frame intervals
    to detect dropped camera frames and map ME to correct neural frames.
    tstamps are in kiloseconds.
    """
    if len(me) == n_neural_frames:
        return me.astype(np.float64)
    
    # Use inter-frame intervals to detect dropped frames
    # Each gap > 1.5x median interval indicates dropped frame(s)
    ifi = np.diff(tstamps)  # in kiloseconds
    median_ifi = np.median(ifi)
    
    # Build mapping from camera frame index to neural frame index
    neural_idx = np.zeros(len(me), dtype=int)
    neural_idx[0] = 0
    for i in range(1, len(me)):
        n_dropped = max(0, round(ifi[i-1] / median_ifi) - 1)
        neural_idx[i] = neural_idx[i-1] + 1 + n_dropped
    
    # Clip to valid range
    neural_idx = np.clip(neural_idx, 0, n_neural_frames - 1)
    
    # Create full ME array
    me_full = np.full(n_neural_frames, np.nan, dtype=np.float64)
    me_full[neural_idx] = me.astype(np.float64)
    
    # Interpolate missing values
    valid = ~np.isnan(me_full)
    if not valid.all():
        indices = np.arange(n_neural_frames)
        me_full = np.interp(indices, indices[valid], me_full[valid])
    
    return me_full


def bin_data(data, bin_size=BIN_SIZE):
    """Bin data by averaging consecutive frames.
    
    For 2D data (neurons x time): bin along time axis.
    For 1D data (time,): bin along the single axis.
    """
    if data.ndim == 2:
        n_neurons, n_frames = data.shape
        n_bins = n_frames // bin_size
        data_trimmed = data[:, :n_bins * bin_size]
        return data_trimmed.reshape(n_neurons, n_bins, bin_size).mean(axis=2)
    elif data.ndim == 1:
        n_frames = len(data)
        n_bins = n_frames // bin_size
        data_trimmed = data[:n_bins * bin_size]
        return data_trimmed.reshape(n_bins, bin_size).mean(axis=1)
    else:
        raise ValueError(f"Unexpected data dimensions: {data.ndim}")


def split_into_trials(data, trial_length):
    """Split data into trials of fixed length.
    
    For 2D data (neurons x time): split along time axis.
    For 1D data (time,): split along the single axis.
    Returns list of arrays.
    """
    if data.ndim == 2:
        n_neurons, n_timepoints = data.shape
        n_trials = n_timepoints // trial_length
        trials = []
        for i in range(n_trials):
            start = i * trial_length
            end = start + trial_length
            trials.append(data[:, start:end].astype(np.float32))
        return trials
    elif data.ndim == 1:
        n_timepoints = len(data)
        n_trials = n_timepoints // trial_length
        trials = []
        for i in range(n_trials):
            start = i * trial_length
            end = start + trial_length
            trials.append(data[start:end].astype(np.float32))
        return trials
    else:
        raise ValueError(f"Unexpected data dimensions: {data.ndim}")


def discretize_motion_energy(me_binned, n_bins=N_BINS):
    """Discretize motion energy into equal-percentile bins.
    
    Computes percentile bin edges per session, assigns each timepoint to a bin.
    Returns integer bin labels (0 to n_bins-1).
    """
    percentiles = np.linspace(0, 100, n_bins + 1)[1:-1]  # [20, 40, 60, 80]
    bin_edges = np.percentile(me_binned, percentiles)
    
    # Digitize: assigns values to bins
    # np.digitize returns 0 for values below first edge, n_bins for above last
    binned = np.digitize(me_binned, bin_edges)
    # binned is now 0, 1, 2, 3, 4 (5 bins)
    return binned.astype(np.int64)


def process_session(subj, sess_name, data_dir, show_processing=False, sess_idx=0):
    """Process a single session.
    
    Returns:
        neural_trials: list of (n_neurons, n_timepoints) arrays
        input_trials: list of (1, n_timepoints) arrays
        output_trials: list of (1, n_timepoints) arrays
        n_neurons: int
    """
    t0 = time.time()
    sess_path = os.path.join(data_dir, subj, sess_name)
    
    # Load neural data
    F = np.load(os.path.join(sess_path, 'suite2p', 'plane0', 'F.npy'))
    Fneu = np.load(os.path.join(sess_path, 'suite2p', 'plane0', 'Fneu.npy'))
    n_neurons, n_neural_frames = F.shape
    print(f"  Loading {subj}/{sess_name}: {n_neurons} neurons, {n_neural_frames} frames")
    
    # Compute dF/F
    t1 = time.time()
    dff = compute_dff(F, Fneu)
    print(f"    dF/F computation: {time.time()-t1:.2f}s")
    
    # Load behavioral data
    me = np.load(os.path.join(sess_path, 'move_deve', 'motion_energy_glob.npy'))
    tstamps = np.load(os.path.join(sess_path, 'move_deve', 'tstamps.npy'))
    
    # Align motion energy to neural frames
    me_aligned = align_motion_energy(me, n_neural_frames, tstamps)
    print(f"    ME aligned: {len(me)} -> {len(me_aligned)} frames (missing: {n_neural_frames - len(me)})")
    
    # Bin by 10 frames
    t2 = time.time()
    dff_binned = bin_data(dff, BIN_SIZE)
    me_binned = bin_data(me_aligned, BIN_SIZE)
    n_binned = dff_binned.shape[1]
    print(f"    Binning: {n_neural_frames} -> {n_binned} timepoints ({time.time()-t2:.2f}s)")
    
    # Create time input (time elapsed from session start in seconds)
    # Each binned timepoint represents the center of the bin
    time_input = (np.arange(n_binned) * BIN_SIZE + BIN_SIZE / 2) / FS
    
    # Discretize motion energy into 5 equal-percentile bins (per session)
    me_discrete = discretize_motion_energy(me_binned, N_BINS)
    
    # Check bin distribution
    bin_counts = np.bincount(me_discrete, minlength=N_BINS)
    print(f"    ME bin distribution: {bin_counts} (total: {bin_counts.sum()})")
    
    # Split into 2-minute trials
    neural_trials = split_into_trials(dff_binned, TRIAL_FRAMES_BINNED)
    
    # For input: time within each trial (reset per trial)
    # Actually, the task says "time elapsed from beginning of experiment"
    # So we use absolute time from session start
    input_trials_list = []
    for i in range(len(neural_trials)):
        start = i * TRIAL_FRAMES_BINNED
        end = start + TRIAL_FRAMES_BINNED
        trial_time = time_input[start:end].astype(np.float32)
        input_trials_list.append(trial_time.reshape(1, -1))  # (1, n_timepoints)
    
    output_trials_list = []
    for i in range(len(neural_trials)):
        start = i * TRIAL_FRAMES_BINNED
        end = start + TRIAL_FRAMES_BINNED
        trial_me = me_discrete[start:end].astype(np.int64)
        output_trials_list.append(trial_me.reshape(1, -1))  # (1, n_timepoints)
    
    n_trials = len(neural_trials)
    print(f"    Split into {n_trials} trials of {TRIAL_FRAMES_BINNED} timepoints")
    print(f"    Session processing time: {time.time()-t0:.2f}s")
    
    if show_processing and sess_idx < 2:
        plot_processing(subj, sess_name, dff, me_aligned, dff_binned, me_binned, 
                       me_discrete, time_input, neural_trials, output_trials_list,
                       sess_idx)
    
    return neural_trials, input_trials_list, output_trials_list, n_neurons


def plot_processing(subj, sess_name, dff, me_aligned, dff_binned, me_binned,
                    me_discrete, time_input, neural_trials, output_trials,
                    sess_idx):
    """Plot processing steps for visual verification."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(5, 1, figsize=(16, 20))
    fig.suptitle(f'{subj}/{sess_name}', fontsize=14)
    
    # 1. Raw dF/F raster (first 5 neurons)
    ax = axes[0]
    n_show = min(5, dff.shape[0])
    for i in range(n_show):
        ax.plot(dff[i, :1000] + i * 200, alpha=0.7, label=f'Neuron {i}')
    ax.set_title('Raw dF/F (first 1000 frames, 5 neurons)')
    ax.set_xlabel('Frame')
    ax.legend(fontsize=8)
    
    # 2. Binned dF/F raster
    ax = axes[1]
    for i in range(n_show):
        ax.plot(dff_binned[i, :100] + i * 200, alpha=0.7, label=f'Neuron {i}')
    ax.set_title('Binned dF/F (first 100 bins, 5 neurons)')
    ax.set_xlabel('Bin')
    ax.legend(fontsize=8)
    
    # 3. Motion energy: raw vs binned
    ax = axes[2]
    ax.plot(me_aligned[:3600], alpha=0.5, label='Raw ME (first trial)')
    ax.plot(np.arange(0, 3600, BIN_SIZE) + BIN_SIZE/2, me_binned[:360], 
            alpha=0.8, label='Binned ME')
    ax.set_title('Motion Energy: Raw vs Binned (first trial)')
    ax.set_xlabel('Frame')
    ax.legend()
    
    # 4. Discretized motion energy
    ax = axes[3]
    ax.plot(me_discrete[:360], '.', markersize=2, alpha=0.5)
    ax.set_title('Discretized ME (first trial, 5 bins)')
    ax.set_xlabel('Bin')
    ax.set_ylabel('Bin label')
    ax.set_yticks(range(N_BINS))
    
    # 5. Neural + output alignment check (first trial)
    ax = axes[4]
    trial_neural = neural_trials[0]
    trial_output = output_trials[0]
    ax2 = ax.twinx()
    ax.plot(trial_neural[0], alpha=0.7, label='Neuron 0 dF/F', color='blue')
    ax2.plot(trial_output[0], alpha=0.7, label='ME bin', color='red')
    ax.set_title('Neural + Output Alignment (Trial 0)')
    ax.set_xlabel('Binned timepoint')
    ax.set_ylabel('dF/F', color='blue')
    ax2.set_ylabel('ME bin', color='red')
    
    plt.tight_layout()
    fname = f'processing_{subj}_{sess_name}.png'
    plt.savefig(fname, dpi=100)
    plt.close()
    print(f"    Saved processing plot: {fname}")


def main():
    parser = argparse.ArgumentParser(description='Convert Track2p data to decoder format')
    parser.add_argument('output', help='Output pickle file path')
    parser.add_argument('--full', action='store_true', default=True, help='Process all sessions (default)')
    parser.add_argument('--sample', action='store_true', help='Process only 2 sessions for testing')
    parser.add_argument('--show-processing', action='store_true', help='Plot processing visualizations')
    args = parser.parse_args()
    
    if args.sample:
        args.full = False
    
    total_start = time.time()
    
    subjects, sessions = get_subjects_and_sessions(DATA_DIR)
    print(f"Found {len(subjects)} subjects: {subjects}")
    for subj in subjects:
        print(f"  {subj}: {len(sessions[subj])} sessions")
    
    # Build session list
    session_list = []  # (subject, session_name)
    for subj in subjects:
        for sess in sessions[subj]:
            session_list.append((subj, sess))
    
    if args.sample:
        # Take 2 sessions from different subjects
        session_list = [session_list[0], session_list[7]]  # jm031 first, jm032 first
        print(f"\nSample mode: processing {len(session_list)} sessions")
    else:
        print(f"\nFull mode: processing {len(session_list)} sessions")
    
    # Process all sessions
    all_neural = []
    all_input = []
    all_output = []
    subject_idx_list = []
    brain_region_idx_list = []
    
    # Track unique subjects used
    used_subjects = []
    
    for sess_i, (subj, sess_name) in enumerate(session_list):
        print(f"\nSession {sess_i+1}/{len(session_list)}: {subj}/{sess_name}")
        
        neural_trials, input_trials, output_trials, n_neurons = process_session(
            subj, sess_name, DATA_DIR, 
            show_processing=args.show_processing,
            sess_idx=sess_i
        )
        
        all_neural.append(neural_trials)
        all_input.append(input_trials)
        all_output.append(output_trials)
        
        if subj not in used_subjects:
            used_subjects.append(subj)
        subject_idx_list.append(used_subjects.index(subj))
        
        # All neurons from barrel cortex
        brain_region_idx_list.append(np.zeros(n_neurons, dtype=np.int64))
    
    # Build output dictionary
    data = {
        'neural': all_neural,
        'input': all_input,
        'output': all_output,
        
        'subjects': used_subjects,
        'subject_idx': np.array(subject_idx_list, dtype=np.int64),
        
        'brain_regions': ['S1BF'],  # barrel cortex
        'brain_region_idx': brain_region_idx_list,
        
        'input_names': ['time_elapsed'],
        'output_names': ['motion_energy_bin'],
        'output_values': [
            ['bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4']
        ],
        
        'metadata': {
            'task_description': 'Decode motion energy (5 equal-percentile bins) from barrel cortex calcium imaging during spontaneous behavior',
            'time_bin_size': TIME_BIN_MS,
            'temporal_alignment_event': 'session_start',
            'off_start': 0.0,
            'off_end': None,
            'imaging_rate_hz': FS,
            'bin_size_frames': BIN_SIZE,
            'trial_duration_sec': TRIAL_DURATION_SEC,
            'neural_data_type': 'dF/F (Suite2p baseline-subtracted neuropil-corrected fluorescence)',
            'neucoeff': NEUCOEFF,
            'baseline_method': 'maximin',
            'win_baseline_sec': WIN_BASELINE,
            'sig_baseline_frames': SIG_BASELINE,
            'motion_energy_discretization': '5 equal-percentile bins per session',
            'paper': 'Majnik et al. 2025, Track2p',
        }
    }
    
    # Print summary
    print(f"\n{'='*60}")
    print("CONVERSION SUMMARY")
    print(f"{'='*60}")
    print(f"Subjects: {data['subjects']}")
    print(f"Sessions: {len(all_neural)}")
    total_trials = sum(len(s) for s in all_neural)
    print(f"Total trials: {total_trials}")
    for i, (subj, sess) in enumerate(session_list):
        n_trials = len(all_neural[i])
        n_neur = all_neural[i][0].shape[0]
        n_tp = all_neural[i][0].shape[1]
        print(f"  Session {i}: {subj}/{sess} - {n_trials} trials, {n_neur} neurons, {n_tp} timepoints/trial")
    
    # Save
    print(f"\nSaving to {args.output}...")
    with open(args.output, 'wb') as f:
        pickle.dump(data, f)
    
    file_size = os.path.getsize(args.output) / (1024**2)
    print(f"Saved {args.output} ({file_size:.1f} MB)")
    print(f"Total time: {time.time()-total_start:.1f}s")


if __name__ == '__main__':
    main()
