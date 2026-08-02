#!/usr/bin/env python3
"""Convert Lee et al. (2025) CA1 calcium imaging data to decoder-compatible format.

Usage:
    python -u convert_data.py <output_pickle_file> [--sample] [--full] [--show-processing]
"""
import os
import sys
import time
import argparse
import pickle
import numpy as np
import joblib
from scipy.ndimage import gaussian_filter1d
import torch
from torch.nn import AvgPool1d

# ---- Constants ----
DATA_DIR = 'data'
ANIMALS = ["QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
           "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"]
FPS = 30  # frames per second
TEMPORAL_BIN_SIZE = 3  # frames per temporal bin (matches reference fit_decoder)
GAUSS_SIGMA = 3  # Gaussian smoothing sigma in frames (matches reference fit_decoder)
TRIAL_DURATION_SEC = 60  # 1 minute trials
FRAMES_PER_TRIAL = TRIAL_DURATION_SEC * FPS  # 1800 frames
TIME_BINS_PER_TRIAL = FRAMES_PER_TRIAL // TEMPORAL_BIN_SIZE  # 600 time bins
TIME_BIN_MS = 1000.0 * TEMPORAL_BIN_SIZE / FPS  # 100 ms
N_SPATIAL_BINS = 3  # 3x3 position grid
POSITION_MAX = 75.0  # cm, maximum position value
BUFFER = 1e-5  # small buffer for binning


def get_env_mat(env):
    """Get binary 3x3 matrix for environment name. From reference code utils.py."""
    env_mats = {
        'square':    np.array([[1,1,1],[1,1,1],[1,1,1]]),
        'o':         np.array([[1,1,1],[1,0,1],[1,1,1]]),
        't':         np.array([[0,1,0],[0,1,0],[1,1,1]]),
        'u':         np.array([[1,1,1],[1,0,0],[1,1,1]]),
        'rectangle': np.array([[0,1,1],[0,1,1],[0,1,1]]),
        '+':         np.array([[0,1,0],[1,1,1],[0,1,0]]),
        'i':         np.array([[1,1,1],[0,1,0],[1,1,1]]),
        'l':         np.array([[1,1,1],[1,0,0],[1,0,0]]),
        'bit donut': np.array([[1,1,1],[1,0,1],[0,1,1]]),
        'glenn':     np.array([[1,1,0],[1,1,1],[0,1,1]]),
    }
    return env_mats.get(env, np.full((3,3), np.nan)).astype(float)


def process_session(trace_day, position_day, env_name):
    """Process a single session (day) of data.
    
    Args:
        trace_day: (n_cells, n_frames) binary trace, may contain NaN for unregistered cells
        position_day: (2, n_frames) x,y position in cm
        env_name: environment name string
    
    Returns:
        neural_trials: list of (n_neurons, n_timebins) arrays
        input_trials: list of (9,) arrays (environment geometry)
        output_trials: list of (1, n_timebins) arrays (position bin 0-8)
        n_registered: number of registered neurons
    """
    n_cells, n_frames = trace_day.shape
    
    # Identify registered cells (not NaN on this day)
    # A cell is registered if its trace is not all NaN
    registered_mask = ~np.isnan(trace_day[:, 0])
    registered_trace = trace_day[registered_mask]  # (n_registered, n_frames)
    n_registered = registered_trace.shape[0]
    
    if n_registered == 0:
        return [], [], [], 0
    
    # Replace any remaining NaN with 0 (shouldn't happen but be safe)
    registered_trace = np.nan_to_num(registered_trace, nan=0.0)
    
    # Process neural data: Gaussian smooth then temporal bin
    # Reference: fit_decoder applies gaussian_filter1d(traces, sigma=temporal_bin_size, axis=0)
    # then AvgPool1d(kernel_size=temporal_bin_size)
    # Note: in fit_decoder, traces are (n_frames, n_cells), sigma=temporal_bin_size=3
    # gaussian_filter1d operates on axis=0 (time axis)
    # Here trace is (n_registered, n_frames), so smooth along axis=1
    smoothed_trace = gaussian_filter1d(registered_trace.astype(np.float32), 
                                        sigma=GAUSS_SIGMA, axis=1)
    
    # Temporal binning with AvgPool1d
    # Reference: pooling(torch.tensor(smoothed.T)).numpy().T
    # smoothed.T is (n_cells, n_frames) -> after pool -> (n_cells, n_timebins)
    # Actually in fit_decoder: traces are (n_frames, n_cells)
    # pooling = AvgPool1d(kernel_size=3, stride=3)
    # pooling(torch.tensor(smoothed.T)) -> input is (n_cells, n_frames) -> output (n_cells, n_timebins)
    # .numpy().T -> (n_timebins, n_cells)
    # So for us: trace is (n_registered, n_frames), apply pool directly
    pooling = AvgPool1d(kernel_size=TEMPORAL_BIN_SIZE, stride=TEMPORAL_BIN_SIZE)
    neural_binned = pooling(torch.tensor(smoothed_trace)).numpy()  # (n_registered, n_timebins)
    
    # Process position data: temporal bin then discretize
    # Reference: pooling(torch.tensor(behav.T)).numpy().astype(int).T
    # behav is (n_frames, 2), behav.T is (2, n_frames)
    # After pool: (2, n_timebins), .astype(int).T -> (n_timebins, 2)
    # But in decode_position_within, position is first binned to 15x15
    # For our 3x3 binning:
    pos_binned_temporal = pooling(torch.tensor(position_day.astype(np.float32))).numpy()  # (2, n_timebins)
    
    # Discretize position into 3x3 bins
    # Each bin is 25 cm wide: [0-25), [25-50), [50-75]
    bin_size = (POSITION_MAX + BUFFER) / N_SPATIAL_BINS
    pos_bins = (pos_binned_temporal / bin_size).astype(int)  # (2, n_timebins)
    pos_bins = np.clip(pos_bins, 0, N_SPATIAL_BINS - 1)  # ensure valid range
    
    # Convert to single category: x_bin * 3 + y_bin
    pos_category = pos_bins[0] * N_SPATIAL_BINS + pos_bins[1]  # (n_timebins,)
    
    # Get environment geometry
    env_mat = get_env_mat(env_name)
    env_flat = env_mat.flatten()  # (9,)
    
    # Split into 1-minute trials
    n_timebins_total = neural_binned.shape[1]
    n_trials = n_timebins_total // TIME_BINS_PER_TRIAL
    
    neural_trials = []
    input_trials = []
    output_trials = []
    
    for t in range(n_trials):
        t_start = t * TIME_BINS_PER_TRIAL
        t_end = (t + 1) * TIME_BINS_PER_TRIAL
        
        neural_trial = neural_binned[:, t_start:t_end].astype(np.float32)  # (n_registered, 600)
        input_trial = env_flat.astype(np.float32)  # (9,) static per trial
        output_trial = pos_category[t_start:t_end].astype(np.int64).reshape(1, -1)  # (1, 600)
        
        neural_trials.append(neural_trial)
        input_trials.append(input_trial)
        output_trials.append(output_trial)
    
    return neural_trials, input_trials, output_trials, n_registered


def convert_data(animals, data_dir, sample=False, show_processing=False):
    """Convert all animal data to decoder format.
    
    Args:
        animals: list of animal IDs
        data_dir: path to data directory
        sample: if True, only process 2 sessions total
        show_processing: if True, save processing visualization plots
    """
    all_neural = []
    all_input = []
    all_output = []
    all_subject_idx = []
    all_brain_region_idx = []
    
    subjects = list(animals)  # animal IDs as subject names
    
    total_sessions = 0
    total_trials = 0
    sessions_processed = 0
    
    for animal_idx, animal in enumerate(animals):
        t_animal_start = time.time()
        print(f"\nLoading {animal}...")
        t0 = time.time()
        dat = joblib.load(os.path.join(data_dir, animal))
        print(f"  Loaded in {time.time()-t0:.1f}s")
        
        trace_all = dat[animal]['trace']      # (n_days, n_cells, n_frames)
        position_all = dat[animal]['position']  # (n_days, 2, n_frames)
        envs_all = dat[animal]['envs']          # (n_days, 1)
        
        n_days = trace_all.shape[0]
        n_cells = trace_all.shape[1]
        
        print(f"  {n_days} days, {n_cells} cells, {trace_all.shape[2]} frames/day")
        
        for day in range(n_days):
            if sample and sessions_processed >= 2:
                break
            
            t_day_start = time.time()
            env_name = envs_all[day, 0] if envs_all.ndim > 1 else envs_all[day]
            
            neural_trials, input_trials, output_trials, n_registered = process_session(
                trace_all[day], position_all[day], env_name
            )
            
            if len(neural_trials) == 0:
                print(f"  Day {day}: skipped (no registered cells)")
                continue
            
            all_neural.append(neural_trials)
            all_input.append(input_trials)
            all_output.append(output_trials)
            all_subject_idx.append(animal_idx)
            all_brain_region_idx.append(np.zeros(n_registered, dtype=np.int64))  # all CA1
            
            n_trials_day = len(neural_trials)
            total_trials += n_trials_day
            total_sessions += 1
            sessions_processed += 1
            
            print(f"  Day {day} ({env_name}): {n_registered} neurons, {n_trials_day} trials, "
                  f"{time.time()-t_day_start:.2f}s")
            
            # Show processing visualization
            if show_processing and sessions_processed <= 2:
                save_processing_plot(trace_all[day], position_all[day], env_name,
                                    neural_trials, input_trials, output_trials,
                                    animal, day)
        
        if sample and sessions_processed >= 2:
            print(f"  Sample mode: stopping after {sessions_processed} sessions")
            break
        
        print(f"  Animal {animal} done in {time.time()-t_animal_start:.1f}s")
    
    # Build output names and values
    output_names = ['position']
    output_values = [[
        'top-left', 'top-center', 'top-right',
        'middle-left', 'middle-center', 'middle-right',
        'bottom-left', 'bottom-center', 'bottom-right'
    ]]
    
    input_names = [
        'env_00', 'env_01', 'env_02',
        'env_10', 'env_11', 'env_12',
        'env_20', 'env_21', 'env_22'
    ]
    
    data = {
        'neural': all_neural,
        'input': all_input,
        'output': all_output,
        'subjects': subjects,
        'subject_idx': np.array(all_subject_idx, dtype=np.int64),
        'brain_regions': ['CA1'],
        'brain_region_idx': all_brain_region_idx,
        'input_names': input_names,
        'output_names': output_names,
        'output_values': output_values,
        'metadata': {
            'task_description': 'Decode mouse position (3x3 spatial bins) from CA1 calcium imaging during geometric environment exploration',
            'time_bin_size': TIME_BIN_MS,
            'temporal_alignment_event': 'Start of 1-minute trial segment within 40-minute recording session',
            'off_start': 0.0,
            'off_end': float(TRIAL_DURATION_SEC),
            'fps': FPS,
            'temporal_bin_frames': TEMPORAL_BIN_SIZE,
            'gauss_sigma_frames': GAUSS_SIGMA,
            'position_max_cm': POSITION_MAX,
            'n_spatial_bins': N_SPATIAL_BINS,
            'environment_size_cm': '75x75',
            'recording_duration_min': 40,
            'trial_duration_sec': TRIAL_DURATION_SEC,
            'source_paper': 'Lee et al. (2025) Neuron 113(2): 307-320',
        }
    }
    
    print(f"\n=== Conversion Summary ===")
    print(f"Sessions: {total_sessions}")
    print(f"Trials: {total_trials}")
    print(f"Subjects: {len(subjects)}")
    print(f"Time bins per trial: {TIME_BINS_PER_TRIAL}")
    print(f"Time bin size: {TIME_BIN_MS} ms")
    
    return data


def save_processing_plot(trace_day, position_day, env_name, 
                         neural_trials, input_trials, output_trials,
                         animal, day):
    """Save visualization of processing steps for a single session."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(4, 2, figsize=(20, 16))
    fig.suptitle(f'{animal} Day {day} ({env_name})', fontsize=14)
    
    # Get registered cells
    registered_mask = ~np.isnan(trace_day[:, 0])
    n_registered = registered_mask.sum()
    
    # Plot 1: Raw trace (first 5 registered cells, first 3000 frames)
    ax = axes[0, 0]
    reg_indices = np.where(registered_mask)[0][:5]
    t_show = min(3000, trace_day.shape[1])
    for i, ci in enumerate(reg_indices):
        ax.plot(np.arange(t_show)/FPS, trace_day[ci, :t_show] + i*1.5, alpha=0.7)
    ax.set_title(f'Raw binary trace (first 5/{n_registered} cells)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Cell (offset)')
    
    # Plot 2: Raw position
    ax = axes[0, 1]
    ax.plot(position_day[0, :t_show], position_day[1, :t_show], alpha=0.3, linewidth=0.5)
    ax.set_title('Raw position')
    ax.set_xlabel('X (cm)')
    ax.set_ylabel('Y (cm)')
    ax.set_xlim(0, 75)
    ax.set_ylim(0, 75)
    ax.set_aspect('equal')
    
    # Plot 3: Processed neural (first trial, first 5 cells)
    ax = axes[1, 0]
    if len(neural_trials) > 0:
        trial0 = neural_trials[0]
        t_bins = np.arange(trial0.shape[1]) * TIME_BIN_MS / 1000
        for i in range(min(5, trial0.shape[0])):
            ax.plot(t_bins, trial0[i] + i*0.5, alpha=0.7)
        ax.set_title(f'Processed neural (trial 0, first 5 cells)')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Firing rate (offset)')
    
    # Plot 4: Output position bins (first trial)
    ax = axes[1, 1]
    if len(output_trials) > 0:
        out0 = output_trials[0][0]  # (n_timebins,)
        ax.plot(t_bins, out0, '.', markersize=1)
        ax.set_title('Output position bin (trial 0)')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Position bin (0-8)')
        ax.set_yticks(range(9))
    
    # Plot 5: Environment geometry
    ax = axes[2, 0]
    env_mat = get_env_mat(env_name)
    ax.imshow(env_mat, cmap='binary', vmin=0, vmax=1)
    ax.set_title(f'Environment: {env_name}')
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f'{int(env_mat[i,j])}', ha='center', va='center', fontsize=14)
    
    # Plot 6: Position bin distribution (all trials)
    ax = axes[2, 1]
    all_pos = np.concatenate([ot[0] for ot in output_trials])
    counts = np.bincount(all_pos.astype(int), minlength=9)
    ax.bar(range(9), counts / counts.sum())
    ax.set_title('Position bin distribution (all trials)')
    ax.set_xlabel('Position bin')
    ax.set_ylabel('Fraction')
    ax.set_xticks(range(9))
    
    # Plot 7: Neural activity histogram
    ax = axes[3, 0]
    all_neural = np.concatenate([nt.flatten() for nt in neural_trials])
    ax.hist(all_neural[all_neural > 0], bins=50, alpha=0.7)
    ax.set_title('Neural activity distribution (non-zero values)')
    ax.set_xlabel('Activity')
    ax.set_ylabel('Count')
    
    # Plot 8: Trials per session info
    ax = axes[3, 1]
    info_text = (
        f'Animal: {animal}\n'
        f'Day: {day}\n'
        f'Environment: {env_name}\n'
        f'Registered cells: {n_registered}\n'
        f'Trials: {len(neural_trials)}\n'
        f'Time bins/trial: {TIME_BINS_PER_TRIAL}\n'
        f'Time bin size: {TIME_BIN_MS} ms\n'
        f'Input shape: {input_trials[0].shape if input_trials else "N/A"}\n'
        f'Neural shape: {neural_trials[0].shape if neural_trials else "N/A"}\n'
        f'Output shape: {output_trials[0].shape if output_trials else "N/A"}'
    )
    ax.text(0.1, 0.5, info_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='center', fontfamily='monospace')
    ax.axis('off')
    ax.set_title('Session info')
    
    plt.tight_layout()
    plt.savefig(f'processing_{animal}_day{day}.png', dpi=100)
    plt.close()
    print(f"    Saved processing plot: processing_{animal}_day{day}.png")


def main():
    parser = argparse.ArgumentParser(description='Convert CA1 data to decoder format')
    parser.add_argument('output', help='Output pickle file path')
    parser.add_argument('--sample', action='store_true', help='Process only 2 sessions')
    parser.add_argument('--full', action='store_true', help='Process all sessions (default)')
    parser.add_argument('--show-processing', action='store_true', help='Save processing plots')
    args = parser.parse_args()
    
    t_total_start = time.time()
    
    print("=== CA1 Data Conversion ===")
    print(f"Output: {args.output}")
    print(f"Mode: {'sample (2 sessions)' if args.sample else 'full'}")
    print(f"Show processing: {args.show_processing}")
    print(f"Temporal bin: {TEMPORAL_BIN_SIZE} frames = {TIME_BIN_MS} ms")
    print(f"Gauss sigma: {GAUSS_SIGMA} frames")
    print(f"Trial duration: {TRIAL_DURATION_SEC} s = {FRAMES_PER_TRIAL} frames = {TIME_BINS_PER_TRIAL} time bins")
    print(f"Spatial bins: {N_SPATIAL_BINS}x{N_SPATIAL_BINS} = {N_SPATIAL_BINS**2} bins")
    
    data = convert_data(
        ANIMALS, DATA_DIR,
        sample=args.sample,
        show_processing=args.show_processing
    )
    
    print(f"\nSaving to {args.output}...")
    t0 = time.time()
    with open(args.output, 'wb') as f:
        pickle.dump(data, f, protocol=4)
    print(f"Saved in {time.time()-t0:.1f}s")
    
    file_size = os.path.getsize(args.output)
    print(f"File size: {file_size / 1024**2:.1f} MB")
    print(f"\nTotal time: {time.time()-t_total_start:.1f}s")


if __name__ == '__main__':
    main()
