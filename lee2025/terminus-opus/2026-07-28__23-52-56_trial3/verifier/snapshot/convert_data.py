#!/usr/bin/env python3
"""Convert georepca1 CA1 calcium imaging data to decoder-compatible format.

Usage:
    python -u convert_data.py <output_pickle_file> [--sample] [--full] [--show-processing]
"""
import sys
import os
import argparse
import time
import pickle
import numpy as np
import joblib

# ============================================================
# Constants
# ============================================================
ANIMALS = [
    "QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50",
    "QLAK-CA1-51", "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"
]
DATA_DIR = "data"
FPS = 30  # frames per second
TRIAL_DURATION_SEC = 60  # 1 minute trials
FRAMES_PER_TRIAL = FPS * TRIAL_DURATION_SEC  # 1800 frames
N_SPATIAL_BINS = 3  # 3x3 grid
ENV_SIZE_CM = 75.0  # 75cm x 75cm arena
BIN_SIZE_CM = ENV_SIZE_CM / N_SPATIAL_BINS  # 25cm per bin


def get_env_mat(env):
    """Get binary 3x3 matrix for environment geometry.
    Copied from reference code utils.py."""
    envs = {
        'square':    [[1,1,1],[1,1,1],[1,1,1]],
        'o':         [[1,1,1],[1,0,1],[1,1,1]],
        't':         [[0,1,0],[0,1,0],[1,1,1]],
        'u':         [[1,1,1],[1,0,0],[1,1,1]],
        'rectangle': [[0,1,1],[0,1,1],[0,1,1]],
        '+':         [[0,1,0],[1,1,1],[0,1,0]],
        'i':         [[1,1,1],[0,1,0],[1,1,1]],
        'l':         [[1,1,1],[1,0,0],[1,0,0]],
        'bit donut': [[1,1,1],[1,0,1],[0,1,1]],
        'glenn':     [[1,1,0],[1,1,1],[0,1,1]],
    }
    return np.array(envs.get(env, [[0,0,0],[0,0,0],[0,0,0]])).astype(float)


def position_to_bin(pos_xy, env_size=ENV_SIZE_CM, n_bins=N_SPATIAL_BINS):
    """Convert continuous x,y position to discrete 3x3 bin index (0-8).
    
    Args:
        pos_xy: (2, n_timepoints) array of x,y positions in cm
        env_size: size of environment in cm
        n_bins: number of bins per dimension
    
    Returns:
        bin_indices: (n_timepoints,) array of bin indices 0-8
    """
    bin_size = env_size / n_bins
    # Clip to valid range and compute bin indices
    x_bins = np.clip(np.floor(pos_xy[0] / bin_size).astype(int), 0, n_bins - 1)
    y_bins = np.clip(np.floor(pos_xy[1] / bin_size).astype(int), 0, n_bins - 1)
    # Combine into single index: row * n_cols + col
    bin_indices = x_bins * n_bins + y_bins
    return bin_indices


def process_animal(animal, data_dir=DATA_DIR, show_processing=False):
    """Process one animal's data into sessions of trials.
    
    Returns:
        sessions: list of dicts, each with 'neural', 'input', 'output', 'env_name', 'n_active'
    """
    t0 = time.time()
    print(f"  Loading {animal}...")
    dat = joblib.load(os.path.join(data_dir, animal))
    d = dat[animal]
    t_load = time.time() - t0
    print(f"  Loaded in {t_load:.1f}s")
    
    trace = d['trace']       # (n_sessions, n_neurons, n_timepoints)
    position = d['position'] # (n_sessions, 2, n_timepoints)
    envs = d['envs']         # (n_sessions, 1)
    
    n_sessions = trace.shape[0]
    n_neurons_total = trace.shape[1]
    n_timepoints = trace.shape[2]
    
    sessions = []
    
    for day in range(n_sessions):
        t_sess = time.time()
        env_name = envs[day, 0]
        
        # Get active neurons (non-NaN)
        tr = trace[day]  # (n_neurons, n_timepoints)
        active_mask = ~np.all(np.isnan(tr), axis=1)  # neurons that are not all-NaN
        active_neurons = tr[active_mask]  # (n_active, n_timepoints)
        n_active = active_neurons.shape[0]
        
        # Replace any remaining NaN with 0 (shouldn't happen but safety)
        active_neurons = np.nan_to_num(active_neurons, nan=0.0)
        
        # Get position data
        pos = position[day]  # (2, n_timepoints)
        
        # Compute position bins (3x3 = 9 categories)
        pos_bins = position_to_bin(pos)  # (n_timepoints,)
        
        # Get environment geometry as input
        env_mat = get_env_mat(env_name).flatten()  # (9,)
        
        # Split into 1-minute trials
        n_trials = n_timepoints // FRAMES_PER_TRIAL
        
        trial_neural = []
        trial_input = []
        trial_output = []
        
        for t in range(n_trials):
            start = t * FRAMES_PER_TRIAL
            end = (t + 1) * FRAMES_PER_TRIAL
            
            # Neural: (n_active, FRAMES_PER_TRIAL)
            trial_neural.append(active_neurons[:, start:end].astype(np.float32))
            
            # Input: environment geometry, static per trial (9,)
            trial_input.append(env_mat.astype(np.float32))
            
            # Output: position bin, time-varying (1, FRAMES_PER_TRIAL)
            # Reshape to (1, n_timepoints) for single output variable
            trial_output.append(pos_bins[start:end].reshape(1, -1).astype(np.int64))
        
        sessions.append({
            'neural': trial_neural,
            'input': trial_input,
            'output': trial_output,
            'env_name': env_name,
            'n_active': n_active,
            'animal': animal,
        })
        
        dt = time.time() - t_sess
        if day == 0 or (day + 1) % 10 == 0 or day == n_sessions - 1:
            print(f"    Session {day+1}/{n_sessions}: env={env_name}, "
                  f"n_active={n_active}, n_trials={n_trials}, dt={dt:.2f}s")
    
    # Show processing plots if requested
    if show_processing:
        plot_processing(d, animal, sessions)
    
    del dat  # free memory
    return sessions


def plot_processing(d, animal, sessions):
    """Plot processing visualizations for verification."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(4, 3, figsize=(20, 16))
    fig.suptitle(f'Processing Verification: {animal}', fontsize=14)
    
    # Plot first 3 sessions
    for col, sess_idx in enumerate(range(min(3, len(sessions)))):
        sess = sessions[sess_idx]
        env_name = sess['env_name']
        
        # Plot 1: Raw neural activity (first trial, first 10 neurons)
        ax = axes[0, col]
        trial_neural = sess['neural'][0]  # first trial
        n_show = min(10, trial_neural.shape[0])
        t_show = min(300, trial_neural.shape[1])  # first 10 seconds
        for i in range(n_show):
            ax.plot(np.arange(t_show)/FPS, trial_neural[i, :t_show] + i*1.5, 'k', linewidth=0.5)
        ax.set_title(f'Session {sess_idx}: {env_name}\nNeural (first trial)')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Neuron')
        
        # Plot 2: Position trajectory (first trial)
        ax = axes[1, col]
        trial_output = sess['output'][0]  # first trial
        ax.plot(np.arange(trial_output.shape[1])/FPS, trial_output[0], '.', markersize=1)
        ax.set_title(f'Position bin (first trial)')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Bin (0-8)')
        ax.set_ylim(-0.5, 8.5)
        
        # Plot 3: Environment geometry input
        ax = axes[2, col]
        env_mat = sess['input'][0].reshape(3, 3)
        ax.imshow(env_mat, cmap='binary', vmin=0, vmax=1)
        ax.set_title(f'Environment: {env_name}')
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f'{int(env_mat[i,j])}', ha='center', va='center', fontsize=12)
        
        # Plot 4: Position bin distribution (all trials)
        ax = axes[3, col]
        all_bins = np.concatenate([o[0] for o in sess['output']])
        counts = np.bincount(all_bins.astype(int), minlength=9)
        ax.bar(range(9), counts / counts.sum())
        ax.set_title(f'Position bin distribution')
        ax.set_xlabel('Bin')
        ax.set_ylabel('Fraction')
    
    plt.tight_layout()
    plt.savefig(f'processing_{animal}.png', dpi=150)
    plt.close()
    print(f"  Saved processing_{animal}.png")


def main():
    parser = argparse.ArgumentParser(description='Convert georepca1 data')
    parser.add_argument('output', type=str, help='Output pickle file path')
    parser.add_argument('--sample', action='store_true', help='Process only 2 sessions for testing')
    parser.add_argument('--full', action='store_true', help='Process all sessions (default)')
    parser.add_argument('--show-processing', action='store_true', help='Plot processing visualizations')
    args = parser.parse_args()
    
    t_start = time.time()
    
    # Determine which animals/sessions to process
    if args.sample:
        # Process 2 animals for testing
        animals_to_process = ANIMALS[:2]
        print(f"SAMPLE mode: processing {len(animals_to_process)} animals")
    else:
        animals_to_process = ANIMALS
        print(f"FULL mode: processing all {len(animals_to_process)} animals")
    
    # Process all animals
    all_neural = []
    all_input = []
    all_output = []
    all_subject_idx = []
    all_brain_region_idx = []
    session_info = []
    
    for a_idx, animal in enumerate(animals_to_process):
        print(f"\nProcessing animal {a_idx+1}/{len(animals_to_process)}: {animal}")
        t_animal = time.time()
        
        sessions = process_animal(
            animal, 
            data_dir=DATA_DIR,
            show_processing=args.show_processing and a_idx < 2
        )
        
        for sess in sessions:
            all_neural.append(sess['neural'])
            all_input.append(sess['input'])
            all_output.append(sess['output'])
            all_subject_idx.append(a_idx)
            # All neurons are from CA1
            all_brain_region_idx.append(np.zeros(sess['n_active'], dtype=int))
            session_info.append({
                'animal': sess['animal'],
                'env_name': sess['env_name'],
                'n_active': sess['n_active'],
                'n_trials': len(sess['neural']),
            })
        
        dt = time.time() - t_animal
        print(f"  Animal {animal} done in {dt:.1f}s ({len(sessions)} sessions)")
    
    # Build output dictionary
    data = {
        'neural': all_neural,
        'input': all_input,
        'output': all_output,
        
        'subjects': [a for a in animals_to_process],
        'subject_idx': np.array(all_subject_idx, dtype=int),
        
        'brain_regions': ['CA1'],
        'brain_region_idx': all_brain_region_idx,
        
        'input_names': [
            'env_grid_0', 'env_grid_1', 'env_grid_2',
            'env_grid_3', 'env_grid_4', 'env_grid_5',
            'env_grid_6', 'env_grid_7', 'env_grid_8',
        ],
        'output_names': ['position_bin'],
        'output_values': [
            ['bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4',
             'bin_5', 'bin_6', 'bin_7', 'bin_8'],
        ],
        
        'metadata': {
            'task_description': 'Decode mouse position (3x3 spatial bins) from CA1 neural activity during free exploration of geometric environments',
            'time_bin_size': 1000.0 / FPS,  # ~33.33 ms
            'temporal_alignment_event': 'Start of recording session',
            'off_start': 0.0,
            'off_end': TRIAL_DURATION_SEC,
            'session_info': session_info,
            'fps': FPS,
            'trial_duration_sec': TRIAL_DURATION_SEC,
            'frames_per_trial': FRAMES_PER_TRIAL,
            'env_size_cm': ENV_SIZE_CM,
            'n_spatial_bins': N_SPATIAL_BINS,
            'bin_size_cm': BIN_SIZE_CM,
            'neural_data_type': 'binary rising-phase calcium transients',
            'position_tracking': 'DeepLabCut',
            'recording_device': 'UCLA miniscope v3',
        },
    }
    
    # Print summary statistics
    total_sessions = len(all_neural)
    total_trials = sum(len(s) for s in all_neural)
    total_neurons = sum(info['n_active'] for info in session_info)
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Subjects: {len(animals_to_process)}")
    print(f"  Sessions: {total_sessions}")
    print(f"  Total trials: {total_trials}")
    print(f"  Total active neuron-sessions: {total_neurons}")
    print(f"  Mean neurons/session: {total_neurons/total_sessions:.1f}")
    print(f"  Mean trials/session: {total_trials/total_sessions:.1f}")
    print(f"  Time bin size: {data['metadata']['time_bin_size']:.2f} ms")
    print(f"  Frames per trial: {FRAMES_PER_TRIAL}")
    
    # Check output distribution
    all_bins = []
    for sess_outputs in all_output:
        for trial_out in sess_outputs:
            all_bins.append(trial_out[0])
    all_bins = np.concatenate(all_bins)
    counts = np.bincount(all_bins.astype(int), minlength=9)
    print(f"  Output distribution (9 bins): {counts / counts.sum()}")
    print(f"{'='*60}")
    
    # Save
    print(f"\nSaving to {args.output}...")
    t_save = time.time()
    with open(args.output, 'wb') as f:
        pickle.dump(data, f, protocol=4)
    print(f"Saved in {time.time() - t_save:.1f}s")
    
    file_size = os.path.getsize(args.output)
    print(f"File size: {file_size / 1e9:.2f} GB")
    
    total_time = time.time() - t_start
    print(f"\nTotal time: {total_time:.1f}s ({total_time/60:.1f} min)")


if __name__ == '__main__':
    main()
