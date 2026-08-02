#!/usr/bin/env python3
"""Convert Track2p longitudinal calcium imaging data to decoder format.

Usage:
    python -u convert_data.py <outpicklefile> [--full|--sample] [--show-processing]
"""

import os
import sys
import time
import argparse
import numpy as np
import pickle
from scipy.ndimage import gaussian_filter1d, maximum_filter1d, minimum_filter1d

# ============================================================
# Suite2p-style baseline correction (maximin filter)
# Matches Suite2p's baseline_maximin function exactly
# ============================================================

def compute_dff(F, Fneu, neucoeff=0.7, win_baseline=60.0, sig_baseline=10.0, fs=30.0):
    """Compute baseline-corrected fluorescence using Suite2p's default parameters.
    
    Suite2p 'maximin' baseline:
    1. Neuropil correction: Fc = F - neucoeff * Fneu
    2. Gaussian smooth Fc (sigma = sig_baseline frames)
    3. Minimum filter (window = win_baseline * fs frames)
    4. Maximum filter (window = win_baseline * fs frames)
    5. Result = Fc - Flow (baseline subtraction)
    
    This matches Suite2p's preprocess() and baseline_maximin() functions.
    The paper calls this "baseline corrected fluorescence traces as our dF/F".
    
    Args:
        F: raw fluorescence, shape (n_neurons, n_timepoints)
        Fneu: neuropil fluorescence, shape (n_neurons, n_timepoints)
        neucoeff: neuropil coefficient (default 0.7)
        win_baseline: baseline window in seconds (default 60.0)
        sig_baseline: Gaussian sigma in frames (default 10.0)
        fs: frame rate in Hz (default 30.0)
    
    Returns:
        dff: baseline-corrected fluorescence, shape (n_neurons, n_timepoints)
    """
    # Neuropil correction
    Fc = F - neucoeff * Fneu
    
    # Compute window in frames
    win = int(win_baseline * fs)  # 60 * 30 = 1800 frames
    
    # Gaussian smoothing
    Flow = gaussian_filter1d(Fc, sigma=sig_baseline, axis=1)
    
    # Minimum filter then maximum filter
    Flow = minimum_filter1d(Flow, size=win, axis=1)
    Flow = maximum_filter1d(Flow, size=win, axis=1)
    
    # Baseline subtraction (NOT division - matches Suite2p)
    dff = Fc - Flow
    
    return dff


def bin_data(data, bin_size=10, axis=-1):
    """Bin data by averaging consecutive timepoints."""
    if axis == -1:
        axis = data.ndim - 1
    
    n = data.shape[axis]
    n_bins = n // bin_size
    slices = [slice(None)] * data.ndim
    slices[axis] = slice(0, n_bins * bin_size)
    data_trimmed = data[tuple(slices)]
    
    new_shape = list(data_trimmed.shape)
    new_shape[axis] = n_bins
    new_shape.insert(axis + 1, bin_size)
    
    return data_trimmed.reshape(new_shape).mean(axis=axis + 1)


def interpolate_missing_frames(me, n_target):
    """Interpolate motion energy to match neural frame count."""
    if len(me) == n_target:
        return me
    x_orig = np.linspace(0, 1, len(me))
    x_target = np.linspace(0, 1, n_target)
    me_interp = np.interp(x_target, x_orig, me.astype(float))
    return me_interp


def load_session_data(session_dir):
    """Load neural and behavioral data for one session."""
    s2p_dir = os.path.join(session_dir, 'suite2p', 'plane0')
    move_dir = os.path.join(session_dir, 'move_deve')
    
    F = np.load(os.path.join(s2p_dir, 'F.npy'))
    Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
    ops = np.load(os.path.join(s2p_dir, 'ops.npy'), allow_pickle=True).item()
    
    me = np.load(os.path.join(move_dir, 'motion_energy_glob.npy'))
    
    n_neurons, n_frames = F.shape
    
    return {
        'F': F,
        'Fneu': Fneu,
        'motion_energy': me,
        'ops': ops,
        'n_neurons': n_neurons,
        'n_frames': n_frames,
    }


def process_session(session_dir, bin_size=10, trial_duration_sec=120):
    """Process one session: compute dF/F, bin, split into trials."""
    data = load_session_data(session_dir)
    F = data['F']
    Fneu = data['Fneu']
    me = data['motion_energy']
    n_neurons = data['n_neurons']
    n_frames = data['n_frames']
    
    # Get ops parameters
    ops = data['ops']
    fs = ops.get('fs', 30.0)
    neucoeff = ops.get('neucoeff', 0.7)
    win_baseline = ops.get('win_baseline', 60.0)
    sig_baseline = ops.get('sig_baseline', 10.0)
    
    # Compute dF/F (baseline-corrected fluorescence)
    dff = compute_dff(F, Fneu, 
                      neucoeff=neucoeff,
                      win_baseline=win_baseline,
                      sig_baseline=sig_baseline,
                      fs=fs)
    
    # Handle missing ME frames by interpolation
    me = interpolate_missing_frames(me, n_frames)
    
    # Bin both neural and behavioral data
    dff_binned = bin_data(dff, bin_size=bin_size, axis=1)
    me_binned = bin_data(me.reshape(1, -1), bin_size=bin_size, axis=1).flatten()
    
    n_bins = dff_binned.shape[1]
    
    # Create time vector (in seconds from start of session)
    effective_fs = fs / bin_size
    time_vec = np.arange(n_bins) / effective_fs
    
    # Split into trials of trial_duration_sec
    trial_bins = int(trial_duration_sec * effective_fs)
    n_trials = n_bins // trial_bins
    
    neural_trials = []
    input_trials = []
    me_trials = []
    
    for t in range(n_trials):
        start = t * trial_bins
        end = (t + 1) * trial_bins
        
        neural_trial = dff_binned[:, start:end].astype(np.float32)
        time_trial = time_vec[start:end].reshape(1, -1).astype(np.float32)
        me_trial = me_binned[start:end].reshape(1, -1).astype(np.float32)
        
        neural_trials.append(neural_trial)
        input_trials.append(time_trial)
        me_trials.append(me_trial)
    
    return neural_trials, input_trials, me_trials


def discretize_output(all_me_trials, n_bins=5):
    """Discretize motion energy into equal-percentile bins."""
    all_values = []
    for session_trials in all_me_trials:
        for trial in session_trials:
            all_values.append(trial.flatten())
    all_values = np.concatenate(all_values)
    
    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(all_values, percentiles)
    
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf
    
    all_output_trials = []
    for session_trials in all_me_trials:
        session_output = []
        for trial in session_trials:
            binned = np.digitize(trial.flatten(), bin_edges[1:-1])
            binned = np.clip(binned, 0, n_bins - 1).astype(np.int64)
            session_output.append(binned.reshape(1, -1).astype(np.int64))
        all_output_trials.append(session_output)
    
    return all_output_trials, bin_edges


def main():
    parser = argparse.ArgumentParser(description='Convert Track2p data to decoder format.')
    parser.add_argument('output_file', type=str, help='Output pickle file path')
    parser.add_argument('--full', action='store_true', default=True, help='Process all sessions (default)')
    parser.add_argument('--sample', action='store_true', help='Process only 2 sessions for testing')
    parser.add_argument('--show-processing', action='store_true', help='Plot processing visualizations')
    args = parser.parse_args()
    
    if args.sample:
        args.full = False
    
    data_dir = 'data'
    bin_size = 10
    trial_duration_sec = 120
    n_output_bins = 5
    
    mice = sorted([d for d in os.listdir(data_dir) 
                   if os.path.isdir(os.path.join(data_dir, d))])
    
    print(f"Found {len(mice)} mice: {mice}")
    
    all_sessions = []
    for mouse_idx, mouse in enumerate(mice):
        mouse_dir = os.path.join(data_dir, mouse)
        sessions = sorted([d for d in os.listdir(mouse_dir) 
                          if os.path.isdir(os.path.join(mouse_dir, d))])
        for sess in sessions:
            sess_dir = os.path.join(mouse_dir, sess)
            all_sessions.append((mouse, mouse_idx, sess_dir, sess))
    
    print(f"Total sessions: {len(all_sessions)}")
    
    if args.sample:
        all_sessions = all_sessions[:2]
        print(f"Sample mode: processing {len(all_sessions)} sessions")
    
    all_neural = []
    all_input = []
    all_me_raw = []
    all_subject_idx = []
    all_brain_region_idx = []
    session_info = []
    
    t_start = time.time()
    
    for i, (mouse, mouse_idx, sess_dir, sess_name) in enumerate(all_sessions):
        t_sess = time.time()
        print(f"Processing session {i+1}/{len(all_sessions)}: {mouse}/{sess_name}...", end=' ')
        
        neural_trials, input_trials, me_trials = process_session(
            sess_dir, bin_size=bin_size, trial_duration_sec=trial_duration_sec
        )
        
        n_neurons = neural_trials[0].shape[0] if neural_trials else 0
        n_trials = len(neural_trials)
        
        all_neural.append(neural_trials)
        all_input.append(input_trials)
        all_me_raw.append(me_trials)
        
        if args.sample:
            unique_mice_so_far = list(dict.fromkeys(s[0] for s in all_sessions[:i+1]))
            actual_mouse_idx = unique_mice_so_far.index(mouse)
        else:
            actual_mouse_idx = mouse_idx
        
        all_subject_idx.append(actual_mouse_idx)
        all_brain_region_idx.append(np.zeros(n_neurons, dtype=np.int64))
        
        session_info.append({
            'mouse': mouse,
            'session': sess_name,
            'n_neurons': n_neurons,
            'n_trials': n_trials,
        })
        
        dt = time.time() - t_sess
        print(f"{n_neurons} neurons, {n_trials} trials, {dt:.1f}s")
    
    t_process = time.time() - t_start
    print(f"\nTotal processing time: {t_process:.1f}s")
    
    print("Discretizing motion energy into 5 equal-percentile bins...")
    all_output, bin_edges = discretize_output(all_me_raw, n_bins=n_output_bins)
    
    output_values = []
    for i in range(n_output_bins):
        if i == 0:
            output_values.append(f"bin{i}_lowest")
        elif i == n_output_bins - 1:
            output_values.append(f"bin{i}_highest")
        else:
            output_values.append(f"bin{i}")
    
    if args.sample:
        unique_mice = list(dict.fromkeys(s[0] for s in all_sessions))
    else:
        unique_mice = mice
    
    if args.sample:
        subject_idx_arr = np.array([unique_mice.index(s[0]) for s in all_sessions], dtype=np.int64)
    else:
        subject_idx_arr = np.array(all_subject_idx, dtype=np.int64)
    
    result = {
        'neural': all_neural,
        'input': all_input,
        'output': all_output,
        
        'subjects': unique_mice,
        'subject_idx': subject_idx_arr,
        
        'brain_regions': ['S1BF'],
        'brain_region_idx': all_brain_region_idx,
        
        'input_names': ['time_seconds'],
        'output_names': ['motion_energy'],
        'output_values': [output_values],
        
        'metadata': {
            'task_description': 'Decode motion energy (5 equal-percentile bins) from barrel cortex calcium imaging dF/F',
            'time_bin_size': 1000.0 * bin_size / 30.0,
            'temporal_alignment_event': 'start of recording session',
            'off_start': 0.0,
            'off_end': None,
            'bin_size_frames': bin_size,
            'original_frame_rate_hz': 30.0,
            'effective_frame_rate_hz': 3.0,
            'trial_duration_sec': trial_duration_sec,
            'n_output_bins': n_output_bins,
            'neucoeff': 0.7,
            'baseline_method': 'maximin (gaussian smooth -> min filter -> max filter, window=60s)',
            'session_info': session_info,
            'me_bin_edges': bin_edges.tolist(),
        }
    }
    
    print(f"\n=== Data Summary ===")
    print(f"Subjects: {result['subjects']}")
    print(f"Sessions: {len(result['neural'])}")
    total_trials = sum(len(s) for s in result['neural'])
    print(f"Total trials: {total_trials}")
    for i, sess in enumerate(session_info):
        print(f"  Session {i}: {sess['mouse']}/{sess['session']} - {sess['n_neurons']} neurons, {sess['n_trials']} trials")
    
    all_out_vals = np.concatenate([t.flatten() for s in result['output'] for t in s])
    unique, counts = np.unique(all_out_vals, return_counts=True)
    print(f"\nOutput distribution:")
    for u, c in zip(unique, counts):
        print(f"  Bin {int(u)}: {c} ({c/len(all_out_vals)*100:.1f}%)")
    
    print(f"\nNeural data ranges (baseline-corrected F):")
    for i in range(len(result['neural'])):
        all_n = np.concatenate([t for t in result['neural'][i]], axis=1)
        info = session_info[i]
        print(f"  Session {i} ({info['mouse']}/{info['session']}): [{all_n.min():.2f}, {all_n.max():.2f}], mean={all_n.mean():.4f}")
    
    print(f"\nSaving to {args.output_file}...")
    with open(args.output_file, 'wb') as f:
        pickle.dump(result, f)
    
    file_size = os.path.getsize(args.output_file) / (1024 * 1024)
    print(f"Saved {args.output_file} ({file_size:.1f} MB)")
    
    if args.show_processing:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        n_plot_sessions = min(2, len(all_sessions))
        
        for si in range(n_plot_sessions):
            mouse, mouse_idx, sess_dir, sess_name = all_sessions[si]
            sess_data = load_session_data(sess_dir)
            ops = sess_data['ops']
            
            fig, axes = plt.subplots(6, 1, figsize=(20, 24))
            fig.suptitle(f'{mouse}/{sess_name}', fontsize=16)
            
            n_show = min(10, sess_data['n_neurons'])
            
            ax = axes[0]
            for ni in range(n_show):
                ax.plot(sess_data['F'][ni, :1000] + ni * 500, alpha=0.7)
            ax.set_title('Raw F (first 1000 frames, 10 neurons)')
            ax.set_xlabel('Frame')
            
            dff = compute_dff(sess_data['F'], sess_data['Fneu'],
                            neucoeff=ops.get('neucoeff', 0.7),
                            win_baseline=ops.get('win_baseline', 60.0),
                            sig_baseline=ops.get('sig_baseline', 10.0),
                            fs=ops.get('fs', 30.0))
            ax = axes[1]
            for ni in range(n_show):
                ax.plot(dff[ni, :1000] + ni * 100, alpha=0.7)
            ax.set_title('Baseline-corrected F (first 1000 frames, 10 neurons)')
            ax.set_xlabel('Frame')
            
            dff_binned = bin_data(dff, bin_size=bin_size, axis=1)
            ax = axes[2]
            for ni in range(n_show):
                ax.plot(dff_binned[ni, :100] + ni * 100, alpha=0.7)
            ax.set_title(f'Binned baseline-corrected F (first 100 bins, bin_size={bin_size})')
            ax.set_xlabel('Bin')
            
            ax = axes[3]
            me = sess_data['motion_energy']
            ax.plot(me[:3000], alpha=0.7)
            ax.set_title('Raw motion energy (first 3000 frames)')
            ax.set_xlabel('Frame')
            
            me_interp = interpolate_missing_frames(me, sess_data['n_frames'])
            me_binned_plot = bin_data(me_interp.reshape(1, -1), bin_size=bin_size, axis=1).flatten()
            ax = axes[4]
            ax.plot(me_binned_plot[:300], alpha=0.7)
            ax.set_title(f'Binned motion energy (first 300 bins)')
            ax.set_xlabel('Bin')
            
            ax = axes[5]
            output_vals = result['output'][si][0].flatten()
            ax.plot(output_vals, 'o-', markersize=2, alpha=0.7)
            ax.set_title('Discretized motion energy (trial 0)')
            ax.set_xlabel('Bin')
            ax.set_ylabel('Category')
            ax.set_yticks(range(n_output_bins))
            
            plt.tight_layout()
            plt.savefig(f'processing_{mouse}_{sess_name}.png', dpi=100)
            plt.close()
            print(f"Saved processing_{mouse}_{sess_name}.png")
    
    print("\nDone!")


if __name__ == '__main__':
    main()
