#!/usr/bin/env python3
"""Convert Track2p longitudinal calcium imaging data to decoder-compatible format.

Usage:
    python -u convert_data.py <outpicklefile> [--full|--sample] [--show-processing]
"""

import os
import sys
import time
import argparse
import pickle
import numpy as np
from scipy.ndimage import gaussian_filter1d, minimum_filter1d, maximum_filter1d
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Constants
# ============================================================
DATA_DIR = '/app/data'
SUBJECTS = ['jm031', 'jm032', 'jm038', 'jm039', 'jm040', 'jm046']
FS = 30.0  # imaging frame rate in Hz
BIN_SIZE = 10  # number of frames to average for binning
TRIAL_DURATION_SEC = 60  # trial duration in seconds
NEUCOEFF = 0.7  # neuropil correction coefficient (Suite2p default)
SIG_BASELINE = 10.0  # baseline smoothing sigma
WIN_BASELINE = 60.0  # baseline window in seconds
N_BINS_OUTPUT = 5  # number of output bins for motion energy
BRAIN_REGION = 'barrel_cortex'

# Derived constants
BINNED_FS = FS / BIN_SIZE  # 3 Hz
TIME_BIN_MS = (BIN_SIZE / FS) * 1000  # ~333.33 ms
TRIAL_BINS = int(TRIAL_DURATION_SEC * BINNED_FS)  # 180 bins per trial


def compute_dff(F, Fneu, fs=FS, neucoeff=NEUCOEFF, 
                sig_baseline=SIG_BASELINE, win_baseline=WIN_BASELINE):
    """Compute baseline-corrected fluorescence (dF/F) using Suite2p default parameters.
    
    Following the paper: "baseline corrected fluorescence traces as our dF/F 
    (using the default Suite2p parameters)"
    
    Steps:
    1. Neuropil subtraction: Fc = F - neucoeff * Fneu
    2. Baseline estimation using maximin filter
    3. Baseline subtraction: dF = Fc - F0 (as implemented in reference code)
    """
    # Neuropil correction
    Fc = F - neucoeff * Fneu
    
    # Baseline estimation (maximin method - Suite2p default)
    win = int(win_baseline * fs)  # window in frames
    Flow = gaussian_filter1d(Fc, sigma=sig_baseline, axis=1)
    Flow = minimum_filter1d(Flow, size=win, axis=1)
    Flow = maximum_filter1d(Flow, size=win, axis=1)
    
    # Baseline subtraction (matching reference code: Fc - Flow)
    # The reference code does NOT divide by baseline, just subtracts it
    dff = Fc - Flow
    
    return dff.astype(np.float32)


def interpolate_missing_frames(me, n_neural_frames, tstamps, ifi):
    """Interpolate motion energy to match neural frame count when camera frames are missing.
    
    Args:
        me: motion energy array (n_camera_frames,)
        n_neural_frames: expected number of frames
        tstamps: camera timestamps
        ifi: inter-frame intervals
    
    Returns:
        me_full: motion energy array of length n_neural_frames
    """
    if len(me) == n_neural_frames:
        return me.astype(np.float64)
    
    # Identify which neural frames have camera data
    # Camera frames are triggered by the microscope at 30 Hz
    # Missing frames show up as gaps in the ifi (>1.5x median)
    median_ifi = np.median(ifi)
    
    # Build mapping from camera frame index to neural frame index
    neural_indices = np.zeros(len(me), dtype=int)
    neural_idx = 0
    neural_indices[0] = 0
    
    for i in range(len(ifi)):
        # How many neural frames this gap spans
        n_skip = max(1, round(ifi[i] / median_ifi))
        neural_idx += n_skip
        if i + 1 < len(me):
            neural_indices[i + 1] = neural_idx
    
    # Interpolate to fill all neural frames
    me_full = np.interp(
        np.arange(n_neural_frames),
        neural_indices,
        me.astype(np.float64)
    )
    
    return me_full


def bin_data(data, bin_size, axis=-1):
    """Average data in bins along specified axis.
    
    Args:
        data: array to bin
        bin_size: number of frames per bin
        axis: axis along which to bin
    
    Returns:
        binned: binned array
    """
    if axis == -1:
        axis = data.ndim - 1
    
    n = data.shape[axis]
    n_bins = n // bin_size
    
    # Trim to multiple of bin_size
    slices = [slice(None)] * data.ndim
    slices[axis] = slice(0, n_bins * bin_size)
    data_trimmed = data[tuple(slices)]
    
    # Reshape and average
    new_shape = list(data_trimmed.shape)
    new_shape[axis] = n_bins
    new_shape.insert(axis + 1, bin_size)
    
    return data_trimmed.reshape(new_shape).mean(axis=axis + 1).astype(np.float32)


def discretize_motion_energy(me_binned, n_bins=N_BINS_OUTPUT):
    """Discretize motion energy into equal-percentile bins.
    
    Args:
        me_binned: binned motion energy for a full session
        n_bins: number of bins
    
    Returns:
        me_discrete: integer array of bin indices (0 to n_bins-1)
        bin_edges: percentile boundaries
    """
    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(me_binned, percentiles)
    
    # Make bin edges unique to handle ties
    # Use digitize with right=False for [edge_i, edge_i+1) bins
    me_discrete = np.digitize(me_binned, bin_edges[1:-1], right=False)
    
    # Clip to valid range
    me_discrete = np.clip(me_discrete, 0, n_bins - 1)
    
    return me_discrete.astype(np.int64), bin_edges


def get_session_dirs(subject):
    """Get sorted list of session directories for a subject."""
    subj_dir = os.path.join(DATA_DIR, subject)
    sessions = sorted([d for d in os.listdir(subj_dir) 
                       if os.path.isdir(os.path.join(subj_dir, d))])
    return [os.path.join(subj_dir, s) for s in sessions]


def process_session(sess_dir, show_processing=False, session_label=''):
    """Process a single session.
    
    Returns:
        neural_trials: list of (n_neurons, n_timebins) arrays
        input_trials: list of (1, n_timebins) arrays  
        output_trials: list of (1, n_timebins) arrays
        n_neurons: number of neurons
    """
    t0 = time.time()
    
    # Load data
    s2p_dir = os.path.join(sess_dir, 'suite2p', 'plane0')
    move_dir = os.path.join(sess_dir, 'move_deve')
    
    F = np.load(os.path.join(s2p_dir, 'F.npy'))
    Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
    me = np.load(os.path.join(move_dir, 'motion_energy_glob.npy'))
    tstamps = np.load(os.path.join(move_dir, 'tstamps.npy'))
    ifi = np.load(os.path.join(move_dir, 'interframe_int.npy'))
    
    n_neurons, n_frames = F.shape
    t_load = time.time() - t0
    
    # Step 1: Compute dF/F
    t1 = time.time()
    dff = compute_dff(F, Fneu)
    t_dff = time.time() - t1
    
    # Step 2: Handle missing camera frames and align ME
    t2 = time.time()
    me_full = interpolate_missing_frames(me, n_frames, tstamps, ifi)
    t_interp = time.time() - t2
    
    # Step 3: Bin both neural and behavioral data by 10 frames
    t3 = time.time()
    dff_binned = bin_data(dff, BIN_SIZE, axis=1)  # (n_neurons, n_bins)
    me_binned = bin_data(me_full.reshape(1, -1), BIN_SIZE, axis=1).squeeze()  # (n_bins,)
    t_bin = time.time() - t3
    
    n_total_bins = dff_binned.shape[1]
    
    # Step 4: Discretize motion energy into 5 equal-percentile bins (per session)
    t4 = time.time()
    me_discrete, bin_edges = discretize_motion_energy(me_binned, N_BINS_OUTPUT)
    t_disc = time.time() - t4
    
    # Step 5: Create time input (seconds from session start)
    time_seconds = (np.arange(n_total_bins) * BIN_SIZE / FS).astype(np.float32)
    
    # Step 6: Split into 60-second trials
    n_trials = n_total_bins // TRIAL_BINS
    
    neural_trials = []
    input_trials = []
    output_trials = []
    
    for t in range(n_trials):
        start = t * TRIAL_BINS
        end = (t + 1) * TRIAL_BINS
        
        neural_trials.append(dff_binned[:, start:end].astype(np.float32))
        input_trials.append(time_seconds[start:end].reshape(1, -1).astype(np.float32))
        output_trials.append(me_discrete[start:end].reshape(1, -1).astype(np.int64))
    
    t_total = time.time() - t0
    print(f'  {session_label}: {n_neurons} neurons, {n_frames} frames, '
          f'{n_total_bins} bins, {n_trials} trials, '
          f'ME missing={n_frames - len(me)} frames, '
          f'time: load={t_load:.2f}s dff={t_dff:.2f}s bin={t_bin:.2f}s total={t_total:.2f}s')
    
    # Check output distribution
    counts = np.bincount(me_discrete, minlength=N_BINS_OUTPUT)
    fracs = counts / counts.sum()
    print(f'    Output bin distribution: {fracs}')
    
    if show_processing:
        plot_processing(sess_dir, session_label, F, Fneu, dff, me, me_full, 
                       dff_binned, me_binned, me_discrete, time_seconds, bin_edges)
    
    return neural_trials, input_trials, output_trials, n_neurons


def plot_processing(sess_dir, session_label, F, Fneu, dff, me_raw, me_full,
                   dff_binned, me_binned, me_discrete, time_seconds, bin_edges):
    """Plot processing steps for visual verification."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(6, 1, figsize=(20, 24))
    
    # Pick a sample neuron
    neuron_idx = 0
    n_show = min(3000, F.shape[1])  # Show first 100 seconds at 30 Hz
    n_show_binned = n_show // BIN_SIZE
    
    # 1. Raw F and Fneu
    ax = axes[0]
    ax.plot(np.arange(n_show) / FS, F[neuron_idx, :n_show], label='F', alpha=0.7)
    ax.plot(np.arange(n_show) / FS, Fneu[neuron_idx, :n_show], label='Fneu', alpha=0.7)
    ax.set_title(f'{session_label} - Raw fluorescence (neuron {neuron_idx})')
    ax.set_xlabel('Time (s)')
    ax.legend()
    
    # 2. dF/F
    ax = axes[1]
    ax.plot(np.arange(n_show) / FS, dff[neuron_idx, :n_show])
    ax.set_title(f'dF/F (neuron {neuron_idx})')
    ax.set_xlabel('Time (s)')
    
    # 3. Binned dF/F
    ax = axes[2]
    ax.plot(np.arange(n_show_binned) / BINNED_FS, dff_binned[neuron_idx, :n_show_binned])
    ax.set_title(f'Binned dF/F (neuron {neuron_idx}, bin={BIN_SIZE} frames)')
    ax.set_xlabel('Time (s)')
    
    # 4. Motion energy (raw vs interpolated)
    ax = axes[3]
    n_me_show = min(n_show, len(me_raw))
    ax.plot(np.arange(n_me_show) / FS, me_raw[:n_me_show], label='Raw ME', alpha=0.7)
    ax.plot(np.arange(n_show) / FS, me_full[:n_show], label='Interpolated ME', alpha=0.7, linestyle='--')
    ax.set_title('Motion Energy (raw vs interpolated)')
    ax.set_xlabel('Time (s)')
    ax.legend()
    
    # 5. Binned motion energy
    ax = axes[4]
    ax.plot(np.arange(n_show_binned) / BINNED_FS, me_binned[:n_show_binned])
    ax.set_title(f'Binned Motion Energy (bin={BIN_SIZE} frames)')
    ax.set_xlabel('Time (s)')
    
    # 6. Discretized motion energy
    ax = axes[5]
    ax.plot(np.arange(n_show_binned) / BINNED_FS, me_discrete[:n_show_binned], '.', markersize=2)
    for i, edge in enumerate(bin_edges):
        ax.axhline(y=i-0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_title(f'Discretized Motion Energy ({N_BINS_OUTPUT} bins)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Bin index')
    
    fig.suptitle(f'Processing: {session_label}', fontsize=16)
    fig.tight_layout()
    safe_label = session_label.replace('/', '_').replace(' ', '_')
    fig.savefig(f'/app/processing_{safe_label}.png', dpi=100)
    plt.close(fig)
    print(f'    Saved processing plot: /app/processing_{safe_label}.png')


def main():
    parser = argparse.ArgumentParser(description='Convert Track2p data to decoder format')
    parser.add_argument('outfile', type=str, help='Output pickle file path')
    parser.add_argument('--full', action='store_true', default=True,
                       help='Process all sessions (default)')
    parser.add_argument('--sample', action='store_true',
                       help='Process only 2 sessions for testing')
    parser.add_argument('--show-processing', action='store_true',
                       help='Plot processing visualizations')
    args = parser.parse_args()
    
    if args.sample:
        args.full = False
    
    print('=' * 60)
    print('Track2p Data Conversion')
    print('=' * 60)
    print(f'Output file: {args.outfile}')
    print(f'Mode: {"sample" if args.sample else "full"}')
    print(f'Show processing: {args.show_processing}')
    print(f'Binning: {BIN_SIZE} frames ({TIME_BIN_MS:.1f} ms)')
    print(f'Trial duration: {TRIAL_DURATION_SEC}s ({TRIAL_BINS} bins)')
    print(f'Output bins: {N_BINS_OUTPUT}')
    print()
    
    t_start = time.time()
    
    # Initialize data structure
    all_neural = []
    all_input = []
    all_output = []
    all_subject_idx = []
    all_brain_region_idx = []
    
    session_count = 0
    
    for subj_i, subject in enumerate(SUBJECTS):
        sess_dirs = get_session_dirs(subject)
        print(f'\nSubject {subject} ({len(sess_dirs)} sessions):')
        
        for sess_dir in sess_dirs:
            session_name = os.path.basename(sess_dir)
            session_label = f'{subject}/{session_name}'
            
            if args.sample and session_count >= 2:
                break
            
            show = args.show_processing and session_count < 2
            
            neural_trials, input_trials, output_trials, n_neurons = process_session(
                sess_dir, show_processing=show, session_label=session_label
            )
            
            all_neural.append(neural_trials)
            all_input.append(input_trials)
            all_output.append(output_trials)
            all_subject_idx.append(subj_i)
            all_brain_region_idx.append(np.zeros(n_neurons, dtype=np.int64))
            
            session_count += 1
        
        if args.sample and session_count >= 2:
            break
    
    # Build final data dictionary
    data = {
        'neural': all_neural,
        'input': all_input,
        'output': all_output,
        
        'subjects': SUBJECTS,
        'subject_idx': np.array(all_subject_idx, dtype=np.int64),
        
        'brain_regions': [BRAIN_REGION],
        'brain_region_idx': all_brain_region_idx,
        
        'input_names': ['time_in_session'],
        'output_names': ['motion_energy'],
        'output_values': [
            [f'bin_{i}' for i in range(N_BINS_OUTPUT)]
        ],
        
        'metadata': {
            'task_description': 'Decode motion energy (5 equal-percentile bins) from barrel cortex calcium activity during spontaneous behavior in developing mice',
            'time_bin_size': TIME_BIN_MS,
            'temporal_alignment_event': 'session_start',
            'off_start': 0.0,
            'off_end': float(TRIAL_DURATION_SEC),
            'imaging_rate_hz': FS,
            'bin_size_frames': BIN_SIZE,
            'binned_rate_hz': BINNED_FS,
            'trial_duration_sec': TRIAL_DURATION_SEC,
            'neucoeff': NEUCOEFF,
            'sig_baseline': SIG_BASELINE,
            'win_baseline_sec': WIN_BASELINE,
            'n_output_bins': N_BINS_OUTPUT,
            'brain_region_full': 'barrel cortex layer 2/3',
            'calcium_indicator': 'GCaMP8m',
            'species': 'mouse',
        }
    }
    
    # Save
    print(f'\nSaving to {args.outfile}...')
    with open(args.outfile, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    file_size = os.path.getsize(args.outfile) / (1024 * 1024)
    print(f'Saved ({file_size:.1f} MB)')
    
    # Print summary
    print(f'\n{"=" * 60}')
    print('Summary:')
    print(f'  Subjects: {len(SUBJECTS)}')
    print(f'  Sessions: {len(all_neural)}')
    total_trials = sum(len(s) for s in all_neural)
    print(f'  Total trials: {total_trials}')
    total_neurons = sum(all_neural[i][0].shape[0] for i in range(len(all_neural)))
    print(f'  Total neurons (across sessions): {total_neurons}')
    print(f'  Time bins per trial: {TRIAL_BINS}')
    print(f'  Total time: {time.time() - t_start:.1f}s')
    print(f'{"=" * 60}')


if __name__ == '__main__':
    main()
