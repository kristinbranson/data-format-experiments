#!/usr/bin/env python3
"""Convert Track2p data to decoder-compatible format.

Usage:
    python -u convert_data.py <output_file> [--full|--sample] [--show-processing]
"""

import os
import sys
import time
import argparse
import pickle
import numpy as np
from scipy.ndimage import gaussian_filter, minimum_filter1d, maximum_filter1d

# ============================================================
# Configuration
# ============================================================
DATA_DIR = 'data'
BIN_SIZE = 10           # frames to average for denoising
TRIAL_DURATION_SEC = 120  # 2-minute blocks
FRAME_RATE = 30         # Hz
FRAMES_PER_TRIAL = TRIAL_DURATION_SEC * FRAME_RATE  # 3600
BINNED_PER_TRIAL = FRAMES_PER_TRIAL // BIN_SIZE     # 360
TIME_BIN_MS = 1000.0 * BIN_SIZE / FRAME_RATE        # 333.33 ms
N_OUTPUT_BINS = 5       # number of equal-percentile bins for motion energy

# Suite2p default parameters for dF/F
NEUCOEFF = 0.7
BASELINE_METHOD = 'maximin'
SIG_BASELINE = 10.0
WIN_BASELINE = 60.0  # in seconds

MICE = ['jm031', 'jm032', 'jm038', 'jm039', 'jm040', 'jm046']
BRAIN_REGION = 'barrel_cortex_L2/3'


def compute_dff(F, Fneu, fs=30.0, neucoeff=0.7, sig_baseline=10.0, win_baseline=60.0):
    """Compute dF/F using Suite2p default baseline correction.
    
    Steps:
    1. Neuropil correction: Fc = F - neucoeff * Fneu
    2. Baseline estimation using maximin method
    3. dF/F = (Fc - baseline) / baseline
    """
    # Neuropil correction
    Fc = F - neucoeff * Fneu
    
    # Baseline estimation (maximin method)
    win = int(win_baseline * fs)  # window in frames
    Flow = gaussian_filter(Fc.astype(np.float64), [0., sig_baseline])
    Flow = minimum_filter1d(Flow, win, axis=1)
    Flow = maximum_filter1d(Flow, win, axis=1)
    
    # Baseline correction (subtraction only, as in Suite2p)
    # Suite2p's baseline_maximin does F = F - Flow (no division)
    dff = Fc - Flow
    
    return dff.astype(np.float32)


def bin_data(data, bin_size, axis=-1):
    """Average data in non-overlapping bins along specified axis.
    
    Truncates to largest multiple of bin_size.
    """
    n = data.shape[axis]
    n_bins = n // bin_size
    n_use = n_bins * bin_size
    
    if axis == -1 or axis == len(data.shape) - 1:
        data_trunc = data[..., :n_use]
        new_shape = data.shape[:-1] + (n_bins, bin_size)
        return data_trunc.reshape(new_shape).mean(axis=-1)
    elif axis == 0:
        data_trunc = data[:n_use]
        new_shape = (n_bins, bin_size) + data.shape[1:]
        return data_trunc.reshape(new_shape).mean(axis=1)
    else:
        raise ValueError(f"Unsupported axis {axis}")


def align_motion_energy(me, n_neural_frames):
    """Align motion energy to neural frames.
    
    Motion energy may have fewer frames due to missing camera frames.
    Pad with NaN or interpolate to match neural frame count.
    """
    if len(me) == n_neural_frames:
        return me.astype(np.float64)
    elif len(me) < n_neural_frames:
        # Interpolate missing frames
        me_aligned = np.full(n_neural_frames, np.nan, dtype=np.float64)
        me_aligned[:len(me)] = me.astype(np.float64)
        # Simple linear interpolation for missing end frames
        if np.any(np.isnan(me_aligned)):
            valid = ~np.isnan(me_aligned)
            indices = np.arange(n_neural_frames)
            me_aligned = np.interp(indices, indices[valid], me_aligned[valid])
        return me_aligned
    else:
        # Truncate if somehow longer
        return me[:n_neural_frames].astype(np.float64)


def normalize_motion_energy(me):
    """Normalize motion energy to [0, 1] range."""
    me_min = np.nanmin(me)
    me_max = np.nanmax(me)
    if me_max - me_min < 1e-10:
        return np.zeros_like(me)
    return (me - me_min) / (me_max - me_min)


def discretize_to_bins(values, n_bins=5):
    """Discretize continuous values into equal-percentile bins.
    
    Returns integer bin labels 0..n_bins-1.
    Percentile edges are computed to create equal-frequency bins.
    """
    percentiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(values, percentiles)
    # Use digitize to assign bins
    bin_labels = np.digitize(values, edges[1:-1])  # n_bins-1 edges -> n_bins bins (0..n_bins-1)
    return bin_labels.astype(np.int64)


def get_sessions(mouse_dir):
    """Get sorted list of session directories for a mouse."""
    sessions = [d for d in os.listdir(mouse_dir) 
                if os.path.isdir(os.path.join(mouse_dir, d)) and d[0].isdigit()]
    sessions.sort()
    return sessions


def process_session(mouse, session, session_dir, show_processing=False, fig_data=None):
    """Process a single session and return trial-segmented data.
    
    Returns:
        neural_trials: list of (n_neurons, n_timepoints) arrays
        input_trials: list of (1, n_timepoints) arrays (time elapsed)
        output_trials: list of (1, n_timepoints) arrays (discretized motion energy)
        me_binned_raw: binned motion energy before discretization (for computing percentiles)
    """
    t0 = time.time()
    
    # Load neural data
    F = np.load(os.path.join(session_dir, 'suite2p', 'plane0', 'F.npy'))
    Fneu = np.load(os.path.join(session_dir, 'suite2p', 'plane0', 'Fneu.npy'))
    ops = np.load(os.path.join(session_dir, 'suite2p', 'plane0', 'ops.npy'), allow_pickle=True).item()
    
    n_neurons, n_frames = F.shape
    fs = ops.get('fs', FRAME_RATE)
    
    # Load motion energy
    me = np.load(os.path.join(session_dir, 'move_deve', 'motion_energy_glob.npy'))
    
    # Align motion energy to neural frames
    me_aligned = align_motion_energy(me, n_frames)
    
    # Compute dF/F
    dff = compute_dff(F, Fneu, fs=fs)
    
    # Bin both neural and behavioral data by 10 frames
    dff_binned = bin_data(dff, BIN_SIZE, axis=-1)  # (n_neurons, n_binned_frames)
    me_binned = bin_data(me_aligned, BIN_SIZE, axis=0)  # (n_binned_frames,)
    
    n_binned = dff_binned.shape[1]
    
    # Segment into 2-minute trials
    n_trials = n_binned // BINNED_PER_TRIAL
    
    neural_trials = []
    input_trials = []
    me_binned_values = []  # collect all ME values for percentile computation
    
    for t in range(n_trials):
        start = t * BINNED_PER_TRIAL
        end = (t + 1) * BINNED_PER_TRIAL
        
        # Neural data
        neural_trial = dff_binned[:, start:end].astype(np.float32)
        neural_trials.append(neural_trial)
        
        # Input: time elapsed from beginning of experiment (in seconds)
        time_bins = (np.arange(start, end) * TIME_BIN_MS / 1000.0).astype(np.float32)
        input_trials.append(time_bins.reshape(1, -1))
        
        # Collect ME values
        me_binned_values.append(me_binned[start:end])
    
    t1 = time.time()
    print(f'  {mouse}/{session}: {n_neurons} neurons, {n_frames} frames, '
          f'{n_binned} binned frames, {n_trials} trials, {t1-t0:.2f}s')
    
    return neural_trials, input_trials, me_binned_values


def convert_data(output_file, sample=False, show_processing=False):
    """Main conversion function."""
    t_start = time.time()
    
    mice_to_process = MICE[:2] if sample else MICE
    
    all_neural = []
    all_input = []
    all_output = []
    all_subject_idx = []
    all_brain_region_idx = []
    
    # Collect all motion energy values across all sessions for global percentile computation
    # Actually, per the task: "normalized and discretized into five equal-percentile bins"
    # We need to decide: global percentiles or per-session percentiles?
    # The paper normalizes per-session (each session has its own motion energy scale)
    # Using per-session percentiles makes more sense for equal-frequency bins
    
    session_data = []  # (mouse_idx, mouse, session, neural_trials, input_trials, me_values)
    
    for mouse_idx, mouse in enumerate(mice_to_process):
        mouse_dir = os.path.join(DATA_DIR, mouse)
        sessions = get_sessions(mouse_dir)
        
        if sample:
            sessions = sessions[:2]  # Only 2 sessions for sample
        
        print(f'\nProcessing {mouse} ({len(sessions)} sessions)...')
        
        for session in sessions:
            session_dir = os.path.join(mouse_dir, session)
            neural_trials, input_trials, me_values = process_session(
                mouse, session, session_dir, show_processing=show_processing
            )
            session_data.append((mouse_idx, mouse, session, neural_trials, input_trials, me_values))
    
    # Now discretize motion energy per session
    print('\nDiscretizing motion energy...')
    for mouse_idx, mouse, session, neural_trials, input_trials, me_values in session_data:
        # Concatenate all ME values for this session
        me_all = np.concatenate(me_values)
        
        # Normalize to [0, 1]
        me_norm = normalize_motion_energy(me_all)
        
        # Compute percentile edges for this session
        percentiles = np.linspace(0, 100, N_OUTPUT_BINS + 1)
        edges = np.percentile(me_norm, percentiles)
        
        # Discretize each trial
        output_trials = []
        offset = 0
        for me_trial in me_values:
            n_t = len(me_trial)
            me_trial_norm = me_norm[offset:offset + n_t]
            bin_labels = np.digitize(me_trial_norm, edges[1:-1]).astype(np.int64)
            output_trials.append(bin_labels.reshape(1, -1))
            offset += n_t
            
            # Print distribution
        dist = np.bincount(np.concatenate([o.flatten() for o in output_trials]), minlength=N_OUTPUT_BINS)
        print(f'  {mouse}/{session}: ME bin distribution: {dist / dist.sum()}')
        
        all_neural.append(neural_trials)
        all_input.append(input_trials)
        all_output.append(output_trials)
        all_subject_idx.append(mouse_idx)
        
        n_neurons = neural_trials[0].shape[0]
        all_brain_region_idx.append(np.zeros(n_neurons, dtype=np.int64))
    
    # Build the data dictionary
    subjects = mice_to_process
    
    data = {
        'neural': all_neural,
        'input': all_input,
        'output': all_output,
        'subjects': subjects,
        'subject_idx': np.array(all_subject_idx, dtype=np.int64),
        'brain_regions': [BRAIN_REGION],
        'brain_region_idx': all_brain_region_idx,
        'input_names': ['time_elapsed_s'],
        'output_names': ['motion_energy_bin'],
        'output_values': [
            ['bin_0 (lowest)', 'bin_1', 'bin_2', 'bin_3', 'bin_4 (highest)']
        ],
        'metadata': {
            'task_description': 'Decode motion energy (behavioral state proxy) from barrel cortex neural activity in developing mice (P7-P14)',
            'time_bin_size': TIME_BIN_MS,
            'temporal_alignment_event': 'start of recording session',
            'off_start': 0.0,
            'off_end': None,
            'bin_size_frames': BIN_SIZE,
            'frame_rate_hz': FRAME_RATE,
            'trial_duration_s': TRIAL_DURATION_SEC,
            'n_output_bins': N_OUTPUT_BINS,
            'dff_method': 'Suite2p default (neucoeff=0.7, baseline=maximin, sig_baseline=10, win_baseline=60)',
            'motion_energy_method': 'Global pixel-wise squared difference of consecutive video frames',
            'normalization': 'Per-session min-max normalization before percentile binning',
            'reference': 'Majnik et al. 2025 eLife - Track2p',
        }
    }
    
    # Save
    print(f'\nSaving to {output_file}...')
    with open(output_file, 'wb') as f:
        pickle.dump(data, f)
    
    file_size = os.path.getsize(output_file) / (1024 * 1024)
    t_total = time.time() - t_start
    
    print(f'\nConversion complete!')
    print(f'  Output file: {output_file} ({file_size:.1f} MB)')
    print(f'  Total time: {t_total:.1f}s')
    print(f'  Sessions: {len(all_neural)}')
    print(f'  Total trials: {sum(len(s) for s in all_neural)}')
    print(f'  Subjects: {len(subjects)}')
    
    # Print summary statistics
    print(f'\n--- Summary Statistics ---')
    for i, (neural_sess, input_sess, output_sess) in enumerate(zip(all_neural, all_input, all_output)):
        n_trials = len(neural_sess)
        n_neurons = neural_sess[0].shape[0]
        n_timepoints = neural_sess[0].shape[1]
        print(f'  Session {i}: {n_neurons} neurons, {n_trials} trials, {n_timepoints} timepoints/trial')
    
    if show_processing:
        plot_processing(data, mice_to_process)
    
    return data


def plot_processing(data, mice):
    """Plot processing visualizations for up to 2 sessions."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib not available, skipping plots')
        return
    
    n_sessions_to_plot = min(2, len(data['neural']))
    
    for sess_idx in range(n_sessions_to_plot):
        neural_sess = data['neural'][sess_idx]
        input_sess = data['input'][sess_idx]
        output_sess = data['output'][sess_idx]
        
        mouse_idx = data['subject_idx'][sess_idx]
        mouse = data['subjects'][mouse_idx]
        
        fig, axes = plt.subplots(5, 1, figsize=(16, 20))
        fig.suptitle(f'Session {sess_idx}: {mouse}', fontsize=14)
        
        # 1. Raw dF/F heatmap (first trial)
        ax = axes[0]
        neural_trial = neural_sess[0]
        im = ax.imshow(neural_trial[:50], aspect='auto', cmap='viridis', vmin=-1, vmax=3)
        ax.set_title('dF/F (first trial, first 50 neurons)')
        ax.set_xlabel('Time bin')
        ax.set_ylabel('Neuron')
        plt.colorbar(im, ax=ax)
        
        # 2. Example neuron trace
        ax = axes[1]
        # Concatenate all trials for one neuron
        all_neural = np.concatenate([t[0, :] for t in neural_sess])
        ax.plot(all_neural, 'k', linewidth=0.5)
        ax.set_title('Example neuron (neuron 0) dF/F across all trials')
        ax.set_xlabel('Time bin')
        ax.set_ylabel('dF/F')
        # Mark trial boundaries
        for t in range(len(neural_sess)):
            ax.axvline(t * BINNED_PER_TRIAL, color='r', alpha=0.3, linewidth=0.5)
        
        # 3. Input (time elapsed)
        ax = axes[2]
        all_time = np.concatenate([t[0, :] for t in input_sess])
        ax.plot(all_time, 'b', linewidth=0.5)
        ax.set_title('Input: Time elapsed (s)')
        ax.set_xlabel('Time bin')
        ax.set_ylabel('Time (s)')
        
        # 4. Output (motion energy bins)
        ax = axes[3]
        all_output = np.concatenate([t[0, :] for t in output_sess])
        ax.plot(all_output, 'g', linewidth=0.5)
        ax.set_title('Output: Motion energy bins (0-4)')
        ax.set_xlabel('Time bin')
        ax.set_ylabel('Bin')
        
        # 5. Output distribution
        ax = axes[4]
        counts = np.bincount(all_output.astype(int), minlength=N_OUTPUT_BINS)
        ax.bar(range(N_OUTPUT_BINS), counts / counts.sum())
        ax.set_title('Output bin distribution')
        ax.set_xlabel('Bin')
        ax.set_ylabel('Fraction')
        ax.set_xticks(range(N_OUTPUT_BINS))
        
        plt.tight_layout()
        plt.savefig(f'processing_session_{sess_idx}.png', dpi=150)
        plt.close()
        print(f'  Saved processing_session_{sess_idx}.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert Track2p data to decoder format')
    parser.add_argument('output', help='Output pickle file path')
    parser.add_argument('--full', action='store_true', default=True, help='Process all sessions (default)')
    parser.add_argument('--sample', action='store_true', help='Process only 2 sessions for testing')
    parser.add_argument('--show-processing', action='store_true', help='Plot processing visualizations')
    
    args = parser.parse_args()
    
    if args.sample:
        args.full = False
    
    convert_data(args.output, sample=args.sample, show_processing=args.show_processing)
