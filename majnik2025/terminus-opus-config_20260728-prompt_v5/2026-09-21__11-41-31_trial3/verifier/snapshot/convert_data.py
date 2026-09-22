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
import warnings

# Suppress warnings during suite2p import
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from suite2p.extraction.dcnv import preprocess

DATA_DIR = '/app/data'
BIN_SIZE = 10  # frames per bin (paper: "averaging using a bin size of 10 frames")
FRAME_RATE = 30  # Hz
NEUCOEFF = 0.7  # neuropil coefficient (Suite2p default)
TRIAL_DURATION_SEC = 60  # seconds per trial
N_ME_BINS = 5  # number of motion energy percentile bins
BINNED_RATE = FRAME_RATE / BIN_SIZE  # 3 Hz
TIMEPOINTS_PER_TRIAL = int(TRIAL_DURATION_SEC * BINNED_RATE)  # 180


def get_subjects_and_sessions(data_dir):
    """Get sorted list of subjects and their sessions."""
    subjects = sorted([d for d in os.listdir(data_dir)
                       if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('jm')])
    all_sessions = {}
    for subj in subjects:
        subj_dir = os.path.join(data_dir, subj)
        sessions = sorted([d for d in os.listdir(subj_dir)
                          if os.path.isdir(os.path.join(subj_dir, d))])
        all_sessions[subj] = sessions
    return subjects, all_sessions


def load_session_data(data_dir, subject, session):
    """Load all data for a single session."""
    sess_dir = os.path.join(data_dir, subject, session)
    s2p_dir = os.path.join(sess_dir, 'suite2p', 'plane0')
    move_dir = os.path.join(sess_dir, 'move_deve')

    F = np.load(os.path.join(s2p_dir, 'F.npy'))  # (n_neurons, n_frames)
    Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))  # (n_neurons, n_frames)
    ops = np.load(os.path.join(s2p_dir, 'ops.npy'), allow_pickle=True).item()
    me = np.load(os.path.join(move_dir, 'motion_energy_glob.npy'))  # (n_cam_frames,)
    tstamps = np.load(os.path.join(move_dir, 'tstamps.npy'))  # (n_cam_frames,)

    return F, Fneu, ops, me, tstamps


def compute_dff(F, Fneu, ops):
    """Compute dF/F using Suite2p's baseline correction.
    
    Following the paper: "We used baseline corrected fluorescence traces as our dF/F
    (using the default Suite2p parameters)"
    
    Steps:
    1. Neuropil subtraction: Fc = F - 0.7 * Fneu
    2. Baseline correction using maximin method (Suite2p default)
    """
    # Step 1: Neuropil subtraction
    Fc = F.copy() - NEUCOEFF * Fneu
    
    # Step 2: Baseline correction using Suite2p's preprocess function
    # This applies the maximin baseline estimation and subtraction
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dff = preprocess(
        F=Fc.copy(),
        baseline=ops.get('baseline', 'maximin'),
        win_baseline=ops.get('win_baseline', 60.0),
        sig_baseline=ops.get('sig_baseline', 10.0),
        fs=ops.get('fs', 30.0),
        prctile_baseline=ops.get('prctile_baseline', 8.0),
        device=device
    )
    
    return dff


def align_motion_energy(me, tstamps, n_neural_frames):
    """Align motion energy to neural frames, handling missing camera frames.
    
    When there are missing camera frames (me has fewer frames than neural),
    we interpolate the motion energy to match the neural frame count.
    
    The tstamps array tells us which camera frames correspond to which timepoints.
    We use the frame indices implied by tstamps to interpolate.
    """
    n_cam_frames = len(me)
    
    if n_cam_frames == n_neural_frames:
        # No missing frames
        return me.astype(np.float64)
    
    # Missing frames case: interpolate motion energy to neural frame count
    # Camera timestamps tell us when each camera frame was acquired
    # Neural frames are at regular intervals (1/30 Hz)
    # We need to map camera frames to neural frame indices
    
    # Create neural frame timestamps (regular spacing)
    neural_times = np.arange(n_neural_frames) / FRAME_RATE  # in seconds
    
    # Camera timestamps are in kiloseconds, convert to seconds
    cam_times = tstamps * 1000.0  # convert to seconds
    
    # Interpolate motion energy to neural frame times
    me_aligned = np.interp(neural_times, cam_times, me.astype(np.float64))
    
    return me_aligned


def bin_data(data, bin_size, axis=-1):
    """Average data in non-overlapping bins along specified axis.
    
    Following the paper: "slightly denoised the dF/F as well as the behaviour traces
    by averaging in bins of 10 consecutive timestamps"
    """
    if axis == -1:
        axis = data.ndim - 1
    
    n = data.shape[axis]
    n_bins = n // bin_size
    # Truncate to multiple of bin_size
    slices = [slice(None)] * data.ndim
    slices[axis] = slice(0, n_bins * bin_size)
    data_trunc = data[tuple(slices)]
    
    # Reshape and average
    new_shape = list(data_trunc.shape)
    new_shape[axis] = n_bins
    new_shape.insert(axis + 1, bin_size)
    data_reshaped = data_trunc.reshape(new_shape)
    data_binned = data_reshaped.mean(axis=axis + 1)
    
    return data_binned


def split_into_trials(data_2d, timepoints_per_trial):
    """Split a 2D array (n_features x n_timepoints) into trials.
    
    Returns list of arrays, each (n_features x timepoints_per_trial).
    Drops incomplete last trial.
    """
    n_timepoints = data_2d.shape[-1]
    n_trials = n_timepoints // timepoints_per_trial
    
    trials = []
    for t in range(n_trials):
        start = t * timepoints_per_trial
        end = start + timepoints_per_trial
        if data_2d.ndim == 2:
            trials.append(data_2d[:, start:end].copy())
        else:  # 1D
            trials.append(data_2d[start:end].copy())
    
    return trials


def discretize_motion_energy(me_binned_trials, n_bins=5):
    """Discretize motion energy into equal-percentile bins per session.
    
    Computes percentile boundaries from ALL trials in the session,
    then applies to each trial.
    
    Returns discretized trials and bin edge info.
    """
    # Concatenate all motion energy values from all trials in this session
    all_me = np.concatenate(me_binned_trials)
    
    # Compute percentile boundaries for equal-frequency bins
    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(all_me, percentiles)
    
    # Make bin edges unique to handle ties
    # Use np.digitize with right=False to get bin indices 1..n_bins
    # Then subtract 1 to get 0-indexed
    discretized_trials = []
    for me_trial in me_binned_trials:
        # np.digitize returns indices such that bin_edges[i-1] <= x < bin_edges[i]
        bin_indices = np.digitize(me_trial, bin_edges[1:-1], right=False)
        # Clip to valid range [0, n_bins-1]
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        discretized_trials.append(bin_indices.astype(np.int64))
    
    return discretized_trials, bin_edges


def process_session(data_dir, subject, session, show_processing=False, session_idx=0):
    """Process a single session and return trial data."""
    t0 = time.time()
    
    # Load data
    F, Fneu, ops, me, tstamps = load_session_data(data_dir, subject, session)
    n_neurons, n_neural_frames = F.shape
    n_cam_frames = len(me)
    t_load = time.time() - t0
    
    print(f"  [{subject}/{session}] Loaded: {n_neurons} neurons, {n_neural_frames} neural frames, "
          f"{n_cam_frames} camera frames (load: {t_load:.2f}s)")
    
    # Compute dF/F
    t1 = time.time()
    dff = compute_dff(F, Fneu, ops)
    t_dff = time.time() - t1
    print(f"  [{subject}/{session}] dF/F computed ({t_dff:.2f}s), range: [{dff.min():.2f}, {dff.max():.2f}]")
    
    # Align motion energy to neural frames
    me_aligned = align_motion_energy(me, tstamps, n_neural_frames)
    
    if n_cam_frames != n_neural_frames:
        print(f"  [{subject}/{session}] Motion energy interpolated: {n_cam_frames} -> {n_neural_frames} frames")
    
    # Bin data (10 frames per bin)
    t2 = time.time()
    dff_binned = bin_data(dff, BIN_SIZE, axis=1)  # (n_neurons, n_bins)
    me_binned = bin_data(me_aligned.reshape(1, -1), BIN_SIZE, axis=1).flatten()  # (n_bins,)
    t_bin = time.time() - t2
    
    n_binned = dff_binned.shape[1]
    print(f"  [{subject}/{session}] Binned: {n_binned} timepoints ({t_bin:.2f}s)")
    
    # Split into 60-second trials
    neural_trials = split_into_trials(dff_binned, TIMEPOINTS_PER_TRIAL)
    me_trials = split_into_trials(me_binned, TIMEPOINTS_PER_TRIAL)
    n_trials = len(neural_trials)
    
    print(f"  [{subject}/{session}] Split into {n_trials} trials of {TRIAL_DURATION_SEC}s "
          f"({TIMEPOINTS_PER_TRIAL} timepoints each)")
    
    # Discretize motion energy (per session)
    me_discrete_trials, bin_edges = discretize_motion_energy(me_trials, N_ME_BINS)
    
    # Create time input for each trial (time elapsed from session start in seconds)
    input_trials = []
    for t in range(n_trials):
        # Time from beginning of session
        trial_start_sec = t * TRIAL_DURATION_SEC
        time_in_trial = trial_start_sec + np.arange(TIMEPOINTS_PER_TRIAL) / BINNED_RATE
        input_trials.append(time_in_trial.astype(np.float32).reshape(1, -1))  # (1, n_timepoints)
    
    # Format output trials as (1, n_timepoints)
    output_trials = [me.astype(np.int64).reshape(1, -1) for me in me_discrete_trials]
    
    # Convert neural to float32
    neural_trials = [n.astype(np.float32) for n in neural_trials]
    
    t_total = time.time() - t0
    print(f"  [{subject}/{session}] Total processing time: {t_total:.2f}s")
    
    # Plotting for --show-processing
    if show_processing:
        plot_processing(subject, session, session_idx,
                       F, Fneu, dff, me, me_aligned, me_binned,
                       dff_binned, neural_trials, me_trials, me_discrete_trials,
                       input_trials, bin_edges)
    
    return neural_trials, input_trials, output_trials, n_neurons, bin_edges


def plot_processing(subject, session, session_idx,
                   F, Fneu, dff, me_raw, me_aligned, me_binned,
                   dff_binned, neural_trials, me_trials, me_discrete_trials,
                   input_trials, bin_edges):
    """Plot processing steps for visual verification."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(6, 2, figsize=(20, 24))
    fig.suptitle(f'Processing: {subject}/{session}', fontsize=16)
    
    # Sample neuron for plotting
    nrn = 0
    
    # Row 0: Raw F and Fneu for sample neuron
    ax = axes[0, 0]
    ax.plot(F[nrn, :1000], label='F', alpha=0.7)
    ax.plot(Fneu[nrn, :1000], label='Fneu', alpha=0.7)
    ax.set_title(f'Raw fluorescence (neuron {nrn}, first 1000 frames)')
    ax.legend()
    ax.set_xlabel('Frame')
    
    # Row 0: dF/F for sample neuron
    ax = axes[0, 1]
    ax.plot(dff[nrn, :1000])
    ax.set_title(f'dF/F (neuron {nrn}, first 1000 frames)')
    ax.set_xlabel('Frame')
    
    # Row 1: Motion energy raw vs aligned
    ax = axes[1, 0]
    ax.plot(me_raw[:1000], label='Raw ME', alpha=0.7)
    ax.plot(me_aligned[:1000], label='Aligned ME', alpha=0.7)
    ax.set_title('Motion energy: raw vs aligned (first 1000 frames)')
    ax.legend()
    ax.set_xlabel('Frame')
    
    # Row 1: Binned motion energy
    ax = axes[1, 1]
    ax.plot(me_binned[:300])
    ax.set_title('Binned motion energy (first 300 bins)')
    ax.set_xlabel('Bin')
    
    # Row 2: dF/F binned - raster
    ax = axes[2, 0]
    n_show = min(50, dff_binned.shape[0])
    ax.imshow(dff_binned[:n_show, :300], aspect='auto', cmap='gray_r', vmin=0, vmax=np.percentile(dff_binned[:n_show], 95))
    ax.set_title(f'Binned dF/F raster (first {n_show} neurons, 300 bins)')
    ax.set_xlabel('Bin')
    ax.set_ylabel('Neuron')
    
    # Row 2: Histogram of motion energy with bin edges
    ax = axes[2, 1]
    ax.hist(np.concatenate(me_trials), bins=100, density=True, alpha=0.7)
    for edge in bin_edges:
        ax.axvline(edge, color='r', linestyle='--', alpha=0.5)
    ax.set_title('Motion energy distribution with percentile bin edges')
    ax.set_xlabel('Motion energy')
    
    # Row 3: Sample trial neural data
    if len(neural_trials) > 0:
        ax = axes[3, 0]
        trial_idx = 0
        ax.imshow(neural_trials[trial_idx][:n_show], aspect='auto', cmap='gray_r',
                  vmin=0, vmax=np.percentile(neural_trials[trial_idx][:n_show], 95))
        ax.set_title(f'Trial {trial_idx} neural data (first {n_show} neurons)')
        ax.set_xlabel('Timepoint')
        ax.set_ylabel('Neuron')
    
    # Row 3: Sample trial output
    if len(me_discrete_trials) > 0:
        ax = axes[3, 1]
        ax.plot(me_discrete_trials[0].flatten())
        ax.set_title(f'Trial 0 discretized motion energy')
        ax.set_xlabel('Timepoint')
        ax.set_ylabel('Bin')
    
    # Row 4: Input (time) for first two trials
    ax = axes[4, 0]
    if len(input_trials) > 1:
        ax.plot(input_trials[0].flatten(), label='Trial 0')
        ax.plot(input_trials[1].flatten(), label='Trial 1')
        ax.legend()
    ax.set_title('Input: time elapsed (seconds)')
    ax.set_xlabel('Timepoint')
    ax.set_ylabel('Time (s)')
    
    # Row 4: Distribution of discretized output across trials
    ax = axes[4, 1]
    all_discrete = np.concatenate([m.flatten() for m in me_discrete_trials])
    counts = np.bincount(all_discrete.astype(int), minlength=N_ME_BINS)
    ax.bar(range(N_ME_BINS), counts / counts.sum())
    ax.set_title('Distribution of discretized motion energy bins')
    ax.set_xlabel('Bin')
    ax.set_ylabel('Fraction')
    ax.axhline(1.0/N_ME_BINS, color='r', linestyle='--', label='Uniform')
    ax.legend()
    
    # Row 5: Temporal alignment check - overlay neural and ME for one trial
    if len(neural_trials) > 0 and len(me_trials) > 0:
        ax = axes[5, 0]
        trial_idx = 0
        # Normalize both for overlay
        neural_mean = neural_trials[trial_idx].mean(axis=0)
        neural_norm = (neural_mean - neural_mean.mean()) / (neural_mean.std() + 1e-8)
        me_norm = (me_trials[trial_idx] - me_trials[trial_idx].mean()) / (me_trials[trial_idx].std() + 1e-8)
        ax.plot(neural_norm, label='Mean neural (z-scored)', alpha=0.7)
        ax.plot(me_norm, label='ME (z-scored)', alpha=0.7)
        ax.set_title(f'Trial {trial_idx}: Neural vs ME alignment check')
        ax.legend()
        ax.set_xlabel('Timepoint')
    
    # Row 5: Continuous ME vs discretized
    if len(me_trials) > 0 and len(me_discrete_trials) > 0:
        ax = axes[5, 1]
        ax.plot(me_trials[0], label='Continuous ME', alpha=0.7)
        ax2 = ax.twinx()
        ax2.plot(me_discrete_trials[0].flatten(), color='r', label='Discretized', alpha=0.7)
        ax.set_title('Trial 0: Continuous vs Discretized ME')
        ax.set_xlabel('Timepoint')
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(f'processing_{subject}_{session}.png', dpi=150)
    plt.close()
    print(f"  Saved processing plot: processing_{subject}_{session}.png")


def main():
    parser = argparse.ArgumentParser(description='Convert Track2p data to decoder format.')
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
    
    print("="*60)
    print("Track2p Data Conversion")
    print("="*60)
    
    t_start = time.time()
    
    # Get subjects and sessions
    subjects, all_sessions = get_subjects_and_sessions(DATA_DIR)
    print(f"Found {len(subjects)} subjects: {subjects}")
    for subj in subjects:
        print(f"  {subj}: {len(all_sessions[subj])} sessions")
    
    # Select sessions to process
    session_list = []  # (subject, session) tuples
    for subj in subjects:
        for sess in all_sessions[subj]:
            session_list.append((subj, sess))
    
    if args.sample:
        # Select 2 sessions from different subjects
        session_list = [session_list[0], session_list[len(session_list)//2]]
        print(f"\nSample mode: processing {len(session_list)} sessions")
    
    print(f"\nTotal sessions to process: {len(session_list)}")
    
    # Process all sessions
    neural_all = []
    input_all = []
    output_all = []
    subject_idx_list = []
    brain_region_idx_all = []
    session_info = []
    
    for i, (subj, sess) in enumerate(session_list):
        print(f"\nProcessing session {i+1}/{len(session_list)}: {subj}/{sess}")
        
        subj_idx = subjects.index(subj)
        
        show = args.show_processing and i < 2  # Only show first 2 sessions
        
        neural_trials, input_trials, output_trials, n_neurons, bin_edges = \
            process_session(DATA_DIR, subj, sess, show_processing=show, session_idx=i)
        
        if len(neural_trials) < 2:
            print(f"  WARNING: Session {subj}/{sess} has only {len(neural_trials)} trials, skipping")
            continue
        
        neural_all.append(neural_trials)
        input_all.append(input_trials)
        output_all.append(output_trials)
        subject_idx_list.append(subj_idx)
        brain_region_idx_all.append(np.zeros(n_neurons, dtype=np.int64))  # all barrel cortex
        session_info.append({
            'subject': subj,
            'session': sess,
            'n_neurons': n_neurons,
            'n_trials': len(neural_trials),
            'me_bin_edges': bin_edges.tolist()
        })
    
    # Build output dictionary
    # Filter subjects list to only include subjects with sessions
    used_subjects = sorted(set(subjects[idx] for idx in subject_idx_list))
    # Remap subject indices if in sample mode
    if args.sample:
        subject_idx_remapped = []
        for idx in subject_idx_list:
            subject_idx_remapped.append(used_subjects.index(subjects[idx]))
        subject_idx_array = np.array(subject_idx_remapped, dtype=np.int64)
        final_subjects = used_subjects
    else:
        subject_idx_array = np.array(subject_idx_list, dtype=np.int64)
        final_subjects = subjects
    
    # ME bin labels
    output_values = [['bin_0 (lowest)', 'bin_1', 'bin_2', 'bin_3', 'bin_4 (highest)']]
    
    data = {
        'neural': neural_all,
        'input': input_all,
        'output': output_all,
        'subjects': final_subjects,
        'subject_idx': subject_idx_array,
        'brain_regions': ['barrel_cortex'],
        'brain_region_idx': brain_region_idx_all,
        'input_names': ['time_seconds'],
        'output_names': ['motion_energy'],
        'output_values': output_values,
        'metadata': {
            'task_description': 'Decode motion energy from neural activity in developing mouse barrel cortex',
            'time_bin_size': 1000.0 / BINNED_RATE,  # 333.33 ms
            'temporal_alignment_event': 'session_start',
            'off_start': 0.0,
            'off_end': float(TRIAL_DURATION_SEC),
            'frame_rate': FRAME_RATE,
            'bin_size_frames': BIN_SIZE,
            'binned_rate_hz': BINNED_RATE,
            'trial_duration_sec': TRIAL_DURATION_SEC,
            'timepoints_per_trial': TIMEPOINTS_PER_TRIAL,
            'n_me_bins': N_ME_BINS,
            'neuropil_coefficient': NEUCOEFF,
            'neural_data_type': 'dF/F (Suite2p baseline corrected)',
            'session_info': session_info,
        }
    }
    
    # Print summary
    print("\n" + "="*60)
    print("Conversion Summary")
    print("="*60)
    n_sessions = len(neural_all)
    total_trials = sum(len(s) for s in neural_all)
    neurons_per_session = [neural_all[s][0].shape[0] for s in range(n_sessions)]
    print(f"Sessions: {n_sessions}")
    print(f"Total trials: {total_trials}")
    print(f"Subjects: {final_subjects}")
    print(f"Neurons per session: {neurons_per_session}")
    print(f"Mean neurons: {np.mean(neurons_per_session):.1f} ± {np.std(neurons_per_session):.1f}")
    print(f"Timepoints per trial: {TIMEPOINTS_PER_TRIAL}")
    print(f"Time bin size: {1000.0/BINNED_RATE:.2f} ms")
    
    # Save
    print(f"\nSaving to {args.outfile}...")
    with open(args.outfile, 'wb') as f:
        pickle.dump(data, f)
    
    file_size = os.path.getsize(args.outfile) / (1024*1024)
    print(f"Saved ({file_size:.1f} MB)")
    
    t_total = time.time() - t_start
    print(f"\nTotal conversion time: {t_total:.1f}s")


if __name__ == '__main__':
    main()
