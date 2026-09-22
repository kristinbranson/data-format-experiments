#!/usr/bin/env python3
"""
Convert Sosa et al. 2025 hippocampal calcium imaging data to decoder format.

Usage:
    python -u convert_data.py <outpicklefile> [--full|--sample] [--show-processing]
"""

import argparse
import os
import pickle
import time
import sys
import warnings

import h5py
import numpy as np
import scipy.ndimage as ndi
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─── Constants ────────────────────────────────────────────────────────────────
DATA_DIR = '/app/data'
TRACK_LENGTH = 450.0  # cm
REWARD_ZONES = {
    'A': (80, 130),
    'B': (200, 250),
    'C': (320, 370),
}
ZONE_LABELS = ['A', 'B', 'C']
NEUROPIL_COEF = 0.7
BASELINE_SMOOTH_SIGMA = 15    # frames
BASELINE_MINMAX_WINDOW = 300  # frames (~20 s at 15.5 Hz)
DFF_SMOOTH_SIGMA = 2          # frames
INTERNEURON_R_THRESH = 0.5    # Pearson r threshold (paper)
LICK_ERROR_THRESH = 0.3       # fraction of frames with lick>2 (paper: >30%)
TARGET_RATE = 15.5078125       # Hz, canonical frame rate


# ─── Helper functions ─────────────────────────────────────────────────────────

def nansmooth(a, sigma, axis=-1):
    """Gaussian smoothing that handles NaNs (matching reference code)."""
    nan_inds = np.isnan(a)
    a_clean = np.copy(a)
    a_clean[nan_inds] = 0
    weights = np.ones(a.shape)
    weights[nan_inds] = 0.001
    a_smooth = ndi.gaussian_filter1d(a_clean, sigma, axis=axis)
    w_smooth = ndi.gaussian_filter1d(weights, sigma, axis=axis)
    return a_smooth / w_smooth


def compute_dff(f_raw, f_neu, trial_starts, trial_ends):
    """
    Compute dF/F following reference preprocessing.py:dff().

    Args:
        f_raw: (n_cells, n_timepoints) raw fluorescence
        f_neu: (n_cells, n_timepoints) neuropil fluorescence
        trial_starts: list of trial start frame indices (0-based)
        trial_ends: list of trial end frame indices (0-based, exclusive)

    Returns:
        dff: (n_cells, n_timepoints) with NaN outside trials
    """
    n_cells, n_time = f_raw.shape

    # Initialize with NaN
    f_ = np.full_like(f_raw, np.nan)
    f_neu_ = np.full_like(f_neu, np.nan)

    # Keep only trial data (exclude ITI)
    for start, end in zip(trial_starts, trial_ends):
        f_[:, start:end] = f_raw[:, start:end]
        f_neu_[:, start:end] = f_neu[:, start:end]

    # Neuropil subtraction
    f_ = f_ - NEUROPIL_COEF * f_neu_

    # Baseline computation per trial
    baseline = np.full_like(f_, np.nan)
    dff = np.full_like(f_, np.nan)

    for start, end in zip(trial_starts, trial_ends):
        trial_data = f_[:, start:end]

        # Add back neuropil mean per trial (reference: lines 433-435)
        neuropil_mean = np.nanmean(f_neu_[:, start:end], axis=1, keepdims=True)
        trial_data = trial_data + NEUROPIL_COEF * neuropil_mean
        f_[:, start:end] = trial_data

        # Maximin baseline
        # Step 1: Smooth with sigma=15
        smoothed = nansmooth(trial_data, BASELINE_SMOOTH_SIGMA, axis=1)
        # Step 2: Minimum filter (300 frames = ~20s)
        bl = ndi.minimum_filter1d(smoothed, BASELINE_MINMAX_WINDOW, axis=-1)
        # Step 3: Maximum filter (dilation)
        bl = ndi.maximum_filter1d(bl, BASELINE_MINMAX_WINDOW, axis=-1)
        baseline[:, start:end] = bl

    # Compute dF/F
    nanmask = ~np.isnan(f_[0, :])
    dff[:, nanmask] = (f_[:, nanmask] - baseline[:, nanmask]) / np.abs(baseline[:, nanmask])

    # Smooth dF/F per trial
    for start, end in zip(trial_starts, trial_ends):
        dff[:, start:end] = nansmooth(dff[:, start:end], DFF_SMOOTH_SIGMA, axis=1)

    return dff


def detect_interneurons(dff, speed, nanmask=None):
    """Detect putative interneurons by speed-dFF correlation (paper: r > 0.5)."""
    n_cells = dff.shape[0]
    if nanmask is None:
        nanmask = ~np.isnan(dff[0, :])

    is_interneuron = np.zeros(n_cells, dtype=bool)
    for c in range(n_cells):
        valid = nanmask & ~np.isnan(dff[c, :]) & ~np.isnan(speed)
        if valid.sum() < 10:
            continue
        r = np.corrcoef(dff[c, valid], speed[valid])[0, 1]
        if r > INTERNEURON_R_THRESH:
            is_interneuron[c] = True
    return is_interneuron


def detect_lick_errors(lick, trial_starts, trial_ends, threshold=LICK_ERROR_THRESH):
    """Detect trials with stuck lick sensor (paper: >30% frames with lick>2)."""
    error_trials = []
    for i, (start, end) in enumerate(zip(trial_starts, trial_ends)):
        trial_licks = lick[start:end]
        n_frames = len(trial_licks)
        if n_frames == 0:
            continue
        frac_high = np.sum(trial_licks > 2) / n_frames
        if frac_high > threshold:
            error_trials.append(i)
    return error_trials


def determine_reward_zone(position, reward_zone_signal, trial_starts, trial_ends):
    """
    Determine reward zone label (A, B, or C) for each trial.
    Uses mean position where reward_zone > 0.
    """
    n_trials = len(trial_starts)
    zone_labels = [None] * n_trials

    for i, (start, end) in enumerate(zip(trial_starts, trial_ends)):
        rz = reward_zone_signal[start:end]
        pos = position[start:end]
        mask = rz > 0
        if mask.sum() > 0:
            mean_pos = pos[mask].mean()
            # Assign to closest zone center
            best_zone = None
            best_dist = float('inf')
            for label, (zs, ze) in REWARD_ZONES.items():
                center = (zs + ze) / 2
                dist = abs(mean_pos - center)
                if dist < best_dist:
                    best_dist = dist
                    best_zone = label
            zone_labels[i] = best_zone

    # Fill missing labels by inheriting from nearest neighbor
    for i in range(n_trials):
        if zone_labels[i] is None:
            # Search forward and backward
            for d in range(1, n_trials):
                if i - d >= 0 and zone_labels[i - d] is not None:
                    zone_labels[i] = zone_labels[i - d]
                    break
                if i + d < n_trials and zone_labels[i + d] is not None:
                    zone_labels[i] = zone_labels[i + d]
                    break

    return zone_labels


def determine_reward_per_trial(reward_timestamps, behavior_timestamps, trial_number_signal,
                                trial_starts, trial_ends):
    """Determine which trials received reward from reward event timestamps."""
    n_trials = len(trial_starts)
    rewarded = np.zeros(n_trials, dtype=int)

    if len(reward_timestamps) == 0:
        return rewarded

    # Match each reward timestamp to closest behavior frame
    for rt in reward_timestamps:
        frame_idx = np.argmin(np.abs(behavior_timestamps - rt))
        trial_num = int(trial_number_signal[frame_idx])
        if trial_num >= 0 and trial_num < n_trials:
            rewarded[trial_num] = 1

    return rewarded


def compute_distance_to_reward_zone(position, zone_start, zone_end):
    """Compute signed distance from position to nearest point in reward zone."""
    distance = np.zeros_like(position)
    before = position < zone_start
    inside = (position >= zone_start) & (position <= zone_end)
    after = position > zone_end
    distance[before] = position[before] - zone_start
    distance[inside] = 0
    distance[after] = position[after] - zone_end
    return distance


def discretize_distance(distance):
    """Discretize distance to reward zone into 7 bins."""
    bins = np.zeros(len(distance), dtype=int)
    bins[distance < -50] = 0
    bins[(distance >= -50) & (distance < -10)] = 1
    bins[(distance >= -10) & (distance < 0)] = 2
    bins[distance == 0] = 3  # inside reward zone or exactly at boundary
    bins[(distance > 0) & (distance <= 10)] = 4
    bins[(distance > 10) & (distance <= 50)] = 5
    bins[distance > 50] = 6
    return bins


def discretize_position(position):
    """Discretize position into 5 equal bins (90 cm each)."""
    bins = np.zeros(len(position), dtype=int)
    bins[position < 90] = 0
    bins[(position >= 90) & (position < 180)] = 1
    bins[(position >= 180) & (position < 270)] = 2
    bins[(position >= 270) & (position < 360)] = 3
    bins[position >= 360] = 4
    return bins


def discretize_speed(speed):
    """Discretize speed into 5 bins."""
    bins = np.zeros(len(speed), dtype=int)
    bins[speed < 2] = 0
    bins[(speed >= 2) & (speed < 10)] = 1
    bins[(speed >= 10) & (speed < 20)] = 2
    bins[(speed >= 20) & (speed < 40)] = 3
    bins[speed >= 40] = 4
    return bins


def load_nwb_session(filepath):
    """Load all needed data from an NWB file."""
    data = {}
    with h5py.File(filepath, 'r') as f:
        ophys = f['processing']['ophys']
        bts = f['processing']['behavior']['BehavioralTimeSeries']

        # Image segmentation
        seg = ophys['ImageSegmentation']['PlaneSegmentation']
        iscell = seg['iscell'][:]
        plane_idx = seg['planeIdx'][:]

        # Determine planes
        fluor_grp = ophys['Fluorescence']
        plane_keys = sorted(fluor_grp.keys())

        # Load fluorescence and neuropil for all planes
        fluor_list = []
        neuropil_list = []
        cell_mask_list = []

        offset = 0
        for pk in plane_keys:
            n_rois = fluor_grp[pk]['data'].shape[1]
            plane_num = int(pk.replace('plane', ''))

            # Get iscell mask for this plane
            plane_mask = plane_idx == plane_num
            plane_iscell = iscell[plane_mask, 0].astype(bool)

            # Load fluorescence (timepoints x ROIs) -> transpose to (ROIs x timepoints)
            fl = fluor_grp[pk]['data'][:].T  # (n_rois, T)

            # Load neuropil
            neu_key = pk  # neuropil keys match plane keys
            neu = ophys['Neuropil'][pk]['data'][:].T  # (n_rois, T)

            fluor_list.append(fl[plane_iscell])
            neuropil_list.append(neu[plane_iscell])
            cell_mask_list.append(plane_iscell)
            offset += n_rois

        data['fluorescence'] = np.concatenate(fluor_list, axis=0).astype(np.float64)
        data['neuropil'] = np.concatenate(neuropil_list, axis=0).astype(np.float64)
        data['n_cells'] = data['fluorescence'].shape[0]

        # Frame rate
        rate = fluor_grp[plane_keys[0]]['starting_time'].attrs['rate']
        data['rate'] = float(rate)
        data['is_dual_plane'] = len(plane_keys) > 1

        # Behavioral data
        data['position'] = bts['position']['data'][:].astype(np.float64)
        data['speed'] = bts['speed']['data'][:].astype(np.float64)
        data['lick'] = bts['lick']['data'][:].astype(np.float64)
        data['environment'] = bts['environment']['data'][:].astype(np.float64)
        data['reward_zone'] = bts['reward_zone']['data'][:].astype(np.float64)
        data['trial_number'] = bts['trial number']['data'][:].astype(np.float64)
        data['trial_start'] = bts['trial_start']['data'][:].astype(np.float64)
        data['teleport'] = bts['teleport']['data'][:].astype(np.float64)
        data['behavior_timestamps'] = bts['position']['timestamps'][:]

        # Truncate to minimum length across neural and behavioral data
        n_neural = data['fluorescence'].shape[1]
        n_behav = len(data['position'])
        n_min = min(n_neural, n_behav)
        if n_neural != n_behav:
            data['fluorescence'] = data['fluorescence'][:, :n_min]
            data['neuropil'] = data['neuropil'][:, :n_min]
            data['position'] = data['position'][:n_min]
            data['speed'] = data['speed'][:n_min]
            data['lick'] = data['lick'][:n_min]
            data['environment'] = data['environment'][:n_min]
            data['reward_zone'] = data['reward_zone'][:n_min]
            data['trial_number'] = data['trial_number'][:n_min]
            data['trial_start'] = data['trial_start'][:n_min]
            data['teleport'] = data['teleport'][:n_min]
            data['behavior_timestamps'] = data['behavior_timestamps'][:n_min]

        # Reward events
        reward_ts_data = bts['Reward']
        data['reward_timestamps'] = reward_ts_data['timestamps'][:]

        # Subject info
        data['subject_id'] = f['general']['subject']['subject_id'][()].decode()
        data['session_id'] = f['general']['session_id'][()].decode()

    return data


def get_trial_boundaries(trial_start_signal, teleport_signal, trial_number_signal):
    """Extract trial start and end frame indices."""
    starts = np.where(trial_start_signal == 1)[0]
    teleports = np.where(teleport_signal == 1)[0]

    trial_starts = []
    trial_ends = []

    for s in starts:
        # Find next teleport after this start
        next_teleports = teleports[teleports > s]
        if len(next_teleports) > 0:
            trial_starts.append(s)
            trial_ends.append(next_teleports[0])  # teleport frame (exclusive end)

    return trial_starts, trial_ends


def process_session(filepath, show_processing=False, session_label=""):
    """Process one NWB file into decoder format."""
    t0 = time.time()

    # Load data
    raw = load_nwb_session(filepath)
    t_load = time.time() - t0
    print(f"  [{session_label}] Loaded: {raw['n_cells']} cells, rate={raw['rate']:.2f} Hz, "
          f"dual_plane={raw['is_dual_plane']}, time={t_load:.1f}s")

    # Get trial boundaries
    trial_starts, trial_ends = get_trial_boundaries(
        raw['trial_start'], raw['teleport'], raw['trial_number'])
    n_trials = len(trial_starts)
    print(f"  [{session_label}] Trials: {n_trials}")

    if n_trials < 2:
        print(f"  [{session_label}] SKIP: fewer than 2 trials")
        return None

    # Compute dF/F
    t1 = time.time()
    dff = compute_dff(raw['fluorescence'], raw['neuropil'], trial_starts, trial_ends)
    t_dff = time.time() - t1
    print(f"  [{session_label}] dF/F computed: time={t_dff:.1f}s")

    # Detect and remove interneurons
    speed_for_corr = raw['speed'].copy()
    nanmask = ~np.isnan(dff[0, :])
    is_interneuron = detect_interneurons(dff, speed_for_corr, nanmask)
    n_interneurons = is_interneuron.sum()
    print(f"  [{session_label}] Interneurons: {n_interneurons}/{raw['n_cells']} "
          f"({100*n_interneurons/max(1,raw['n_cells']):.2f}%)")

    # Filter neurons
    keep_mask = ~is_interneuron
    dff = dff[keep_mask]
    n_cells = dff.shape[0]

    if n_cells == 0:
        print(f"  [{session_label}] SKIP: no cells after filtering")
        return None

    # Detect lick errors
    lick_error_trials = detect_lick_errors(raw['lick'], trial_starts, trial_ends)
    print(f"  [{session_label}] Lick error trials: {len(lick_error_trials)}")

    # Determine reward zone per trial
    zone_labels = determine_reward_zone(raw['position'], raw['reward_zone'],
                                         trial_starts, trial_ends)

    # Determine reward per trial
    rewarded = determine_reward_per_trial(
        raw['reward_timestamps'], raw['behavior_timestamps'],
        raw['trial_number'], trial_starts, trial_ends)

    # Determine environment per trial
    environments = []
    for start, end in zip(trial_starts, trial_ends):
        env_vals = raw['environment'][start:end]
        active_env = env_vals[env_vals >= 0]
        if len(active_env) > 0:
            environments.append(int(round(active_env[0])))
        else:
            environments.append(0)

    # Downsample dual-plane data
    downsample = 2 if raw['is_dual_plane'] else 1
    effective_rate = raw['rate'] / downsample

    # Build trial arrays
    neural_trials = []
    input_trials = []
    output_trials = []
    valid_trial_indices = []

    for i in range(n_trials):
        if i in lick_error_trials:
            continue

        start = trial_starts[i]
        end = trial_ends[i]

        # Extract trial data
        trial_neural = dff[:, start:end]
        trial_pos = raw['position'][start:end]
        trial_speed = raw['speed'][start:end]
        trial_lick = raw['lick'][start:end]

        # Skip trials with all-NaN neural data
        if np.all(np.isnan(trial_neural)):
            continue

        # Downsample if dual-plane
        if downsample > 1:
            T = trial_neural.shape[1]
            T_new = T // downsample
            if T_new < 2:
                continue
            trial_neural = trial_neural[:, :T_new * downsample].reshape(n_cells, T_new, downsample).mean(axis=2)
            trial_pos = trial_pos[:T_new * downsample].reshape(T_new, downsample).mean(axis=1)
            trial_speed = trial_speed[:T_new * downsample].reshape(T_new, downsample).mean(axis=1)
            trial_lick = trial_lick[:T_new * downsample].reshape(T_new, downsample).max(axis=1)

        T = trial_neural.shape[1]
        if T < 2:
            continue

        # Replace NaN in neural with 0
        trial_neural = np.nan_to_num(trial_neural, nan=0.0).astype(np.float32)

        # --- INPUTS (4, T) ---
        time_from_start = np.arange(T, dtype=np.float32) / effective_rate
        env = np.full(T, environments[i], dtype=np.float32)
        trial_num = np.full(T, i, dtype=np.float32)
        prev_outcome = np.full(T, rewarded[i - 1] if i > 0 else 0, dtype=np.float32)
        inputs = np.stack([time_from_start, env, trial_num, prev_outcome], axis=0)

        # --- OUTPUTS (6, T) ---
        # Reward zone for this trial
        zone_label = zone_labels[i]
        zone_idx = ZONE_LABELS.index(zone_label) if zone_label in ZONE_LABELS else 0
        zone_start, zone_end = REWARD_ZONES[zone_label] if zone_label else (80, 130)

        # Distance to reward zone
        distance = compute_distance_to_reward_zone(trial_pos, zone_start, zone_end)
        dist_bins = discretize_distance(distance)

        # Position bins
        pos_bins = discretize_position(trial_pos)

        # Speed bins
        speed_bins = discretize_speed(np.abs(trial_speed))

        # Lick binary
        lick_binary = (trial_lick > 0).astype(int)

        # Per-trial outputs broadcast
        rz_loc = np.full(T, zone_idx, dtype=int)
        reward_out = np.full(T, rewarded[i], dtype=int)

        outputs = np.stack([dist_bins, pos_bins, speed_bins, lick_binary, rz_loc, reward_out], axis=0).astype(int)

        neural_trials.append(trial_neural)
        input_trials.append(inputs)
        output_trials.append(outputs)
        valid_trial_indices.append(i)

    if len(neural_trials) < 2:
        print(f"  [{session_label}] SKIP: fewer than 2 valid trials")
        return None

    t_total = time.time() - t0
    print(f"  [{session_label}] Done: {len(neural_trials)} valid trials, "
          f"{n_cells} cells, time={t_total:.1f}s")

    # Show processing plots
    if show_processing and len(neural_trials) > 0:
        plot_processing(raw, dff, trial_starts, trial_ends, zone_labels, rewarded,
                       neural_trials, input_trials, output_trials, valid_trial_indices,
                       session_label, effective_rate)

    result = {
        'neural': neural_trials,
        'input': input_trials,
        'output': output_trials,
        'n_cells': n_cells,
        'n_interneurons': n_interneurons,
        'n_lick_errors': len(lick_error_trials),
        'subject_id': raw['subject_id'],
        'session_id': raw['session_id'],
        'rate': effective_rate,
    }
    return result


def plot_processing(raw, dff, trial_starts, trial_ends, zone_labels, rewarded,
                   neural_trials, input_trials, output_trials, valid_trial_indices,
                   session_label, effective_rate):
    """Plot processing steps for visual verification."""
    fig, axes = plt.subplots(5, 2, figsize=(20, 20))
    fig.suptitle(f'Processing: {session_label}', fontsize=14)

    # Pick 2 example trials
    example_indices = [0, min(len(neural_trials) - 1, len(neural_trials) // 2)]

    for col, eidx in enumerate(example_indices):
        trial_idx = valid_trial_indices[eidx]
        start, end = trial_starts[trial_idx], trial_ends[trial_idx]

        # Row 0: Raw fluorescence (first 5 cells)
        ax = axes[0, col]
        n_show = min(5, raw['fluorescence'].shape[0])
        for c in range(n_show):
            ax.plot(raw['fluorescence'][c, start:end], alpha=0.5, label=f'Cell {c}')
        ax.set_title(f'Trial {trial_idx}: Raw Fluorescence')
        ax.set_ylabel('F')

        # Row 1: dF/F (first 5 cells)
        ax = axes[1, col]
        for c in range(min(5, dff.shape[0])):
            ax.plot(dff[c, start:end], alpha=0.5)
        ax.set_title(f'dF/F (after processing)')
        ax.set_ylabel('dF/F')

        # Row 2: Position and reward zone
        ax = axes[2, col]
        pos = raw['position'][start:end]
        ax.plot(pos, 'b-', label='Position')
        zl = zone_labels[trial_idx]
        if zl and zl in REWARD_ZONES:
            zs, ze = REWARD_ZONES[zl]
            ax.axhspan(zs, ze, alpha=0.2, color='green', label=f'Zone {zl}')
        ax.set_title(f'Position & RZ (zone={zl}, rew={rewarded[trial_idx]})')
        ax.set_ylabel('Position (cm)')
        ax.legend(fontsize=8)

        # Row 3: Decoder outputs
        ax = axes[3, col]
        T = output_trials[eidx].shape[1]
        t = np.arange(T) / effective_rate
        ax.plot(t, output_trials[eidx][0], label='dist_rz_bin')
        ax.plot(t, output_trials[eidx][1], label='pos_bin')
        ax.plot(t, output_trials[eidx][2], label='speed_bin')
        ax.plot(t, output_trials[eidx][3], label='lick')
        ax.set_title('Decoder Outputs')
        ax.legend(fontsize=8)

        # Row 4: Speed and lick
        ax = axes[4, col]
        ax.plot(raw['speed'][start:end], 'g-', label='Speed')
        ax2 = ax.twinx()
        ax2.plot(raw['lick'][start:end], 'r-', alpha=0.5, label='Lick')
        ax.set_title('Speed & Lick (raw)')
        ax.set_xlabel('Frame')
        ax.set_ylabel('Speed (cm/s)')
        ax2.set_ylabel('Lick count')

    fig.tight_layout()
    safe_label = session_label.replace('/', '_').replace(' ', '_')
    fig.savefig(f'/app/processing_{safe_label}.png', dpi=100)
    plt.close(fig)
    print(f"  Saved processing plot: processing_{safe_label}.png")


def collect_nwb_files(data_dir, sample=False):
    """Collect all NWB files, sorted by subject then session."""
    files = []
    for subj_dir in sorted(os.listdir(data_dir)):
        if not subj_dir.startswith('sub-'):
            continue
        subj_path = os.path.join(data_dir, subj_dir)
        for nwb_file in sorted(os.listdir(subj_path)):
            if nwb_file.endswith('.nwb'):
                files.append(os.path.join(subj_path, nwb_file))

    if sample:
        # Return 2 sessions from different subjects (pick small ones)
        # Pick m11 ses-03 (small) and m19 ses-01 (medium)
        sample_files = []
        for f in files:
            if 'sub-m11_ses-03' in f or 'sub-m19_ses-01' in f:
                sample_files.append(f)
        return sample_files if sample_files else files[:2]

    return files


def main():
    parser = argparse.ArgumentParser(description='Convert NWB data to decoder format.')
    parser.add_argument('outfile', type=str, help='Output pickle file path')
    parser.add_argument('--full', action='store_true', default=True, help='Process all sessions (default)')
    parser.add_argument('--sample', action='store_true', help='Process only 2 sessions for testing')
    parser.add_argument('--show-processing', action='store_true', help='Plot processing visualizations')
    args = parser.parse_args()

    if args.sample:
        args.full = False

    nwb_files = collect_nwb_files(DATA_DIR, sample=args.sample)
    print(f"Found {len(nwb_files)} NWB files to process")

    # Process all sessions
    all_results = []
    total_t0 = time.time()

    for i, filepath in enumerate(nwb_files):
        basename = os.path.basename(filepath)
        label = basename.replace('_behavior+ophys.nwb', '')
        print(f"\n[{i+1}/{len(nwb_files)}] Processing {label}")

        try:
            result = process_session(filepath, show_processing=args.show_processing,
                                    session_label=label)
            if result is not None:
                all_results.append(result)
        except Exception as e:
            print(f"  ERROR processing {label}: {e}")
            import traceback
            traceback.print_exc()

    total_time = time.time() - total_t0
    print(f"\n{'='*60}")
    print(f"Processed {len(all_results)} sessions in {total_time:.1f}s")

    if len(all_results) == 0:
        print("ERROR: No valid sessions!")
        sys.exit(1)

    # Build final data structure
    data = build_data_dict(all_results)

    # Print summary
    print_summary(data)

    # Save
    with open(args.outfile, 'wb') as f:
        pickle.dump(data, f)
    print(f"\nSaved to {args.outfile} ({os.path.getsize(args.outfile) / 1e6:.1f} MB)")


def build_data_dict(results):
    """Build the final data dictionary from processed sessions."""
    # Collect unique subjects
    subjects = sorted(set(r['subject_id'] for r in results))
    subject_to_idx = {s: i for i, s in enumerate(subjects)}

    neural = []
    inputs = []
    outputs = []
    subject_idx = []
    brain_region_idx_list = []

    for r in results:
        neural.append(r['neural'])
        inputs.append(r['input'])
        outputs.append(r['output'])
        subject_idx.append(subject_to_idx[r['subject_id']])
        # All neurons are from CA1
        brain_region_idx_list.append(np.zeros(r['n_cells'], dtype=int))

    data = {
        'neural': neural,
        'input': inputs,
        'output': outputs,
        'subjects': subjects,
        'subject_idx': np.array(subject_idx, dtype=int),
        'brain_regions': ['CA1'],
        'brain_region_idx': brain_region_idx_list,
        'input_names': [
            'time_from_trial_start',
            'environment',
            'trial_number',
            'previous_trial_outcome',
        ],
        'output_names': [
            'distance_to_reward_zone',
            'absolute_position',
            'speed',
            'lick',
            'reward_zone_location',
            'reward_outcome',
        ],
        'output_values': [
            ['< -50cm', '-50 to -10cm', '-10 to 0cm', '0cm (in zone)', '>0 to +10cm', '+10 to +50cm', '> +50cm'],
            ['0-90cm', '90-180cm', '180-270cm', '270-360cm', '360-450cm'],
            ['< 2cm/s', '2-10cm/s', '10-20cm/s', '20-40cm/s', '> 40cm/s'],
            ['no lick', 'lick'],
            ['Zone A', 'Zone B', 'Zone C'],
            ['no reward', 'reward'],
        ],
        'metadata': {
            'task_description': ('Hidden reward zone navigation task on 450cm virtual linear track. '
                                'Mice navigate to find hidden 50cm reward zone (A/B/C) across two '
                                'environments with reward zone switches every few days.'),
            'time_bin_size': 1000.0 / TARGET_RATE,  # ~64.48 ms
            'temporal_alignment_event': 'Start of trial (entry onto linear track)',
            'off_start': 0.0,
            'off_end': None,  # Variable trial length
            'track_length_cm': TRACK_LENGTH,
            'imaging_rate_hz': TARGET_RATE,
            'reward_zones': REWARD_ZONES,
            'n_sessions': len(results),
            'session_info': [
                {'subject': r['subject_id'], 'session': r['session_id'],
                 'n_cells': r['n_cells'], 'n_trials': len(r['neural']),
                 'n_interneurons': r['n_interneurons'],
                 'n_lick_errors': r['n_lick_errors']}
                for r in results
            ],
        },
    }
    return data


def print_summary(data):
    """Print summary statistics."""
    n_sessions = len(data['neural'])
    n_subjects = len(data['subjects'])
    total_trials = sum(len(s) for s in data['neural'])
    cells_per_session = [s[0].shape[0] for s in data['neural'] if len(s) > 0]
    trials_per_session = [len(s) for s in data['neural']]

    print(f"\n{'='*60}")
    print(f"DATASET SUMMARY")
    print(f"{'='*60}")
    print(f"Subjects: {n_subjects} ({', '.join(data['subjects'])})")
    print(f"Sessions: {n_sessions}")
    print(f"Total trials: {total_trials}")
    print(f"Trials/session: mean={np.mean(trials_per_session):.1f} +/- {np.std(trials_per_session):.1f}")
    print(f"Cells/session: mean={np.mean(cells_per_session):.1f} +/- {np.std(cells_per_session):.1f}")
    print(f"Time bin: {data['metadata']['time_bin_size']:.2f} ms")

    # Output distributions
    all_outputs = []
    for sess in data['output']:
        for trial in sess:
            all_outputs.append(trial)

    if all_outputs:
        # Concatenate time-varying outputs
        all_out = np.concatenate([o for o in all_outputs], axis=1)
        for dim, name in enumerate(data['output_names']):
            values, counts = np.unique(all_out[dim].astype(int), return_counts=True)
            fracs = counts / counts.sum()
            dist_str = ', '.join([f'{v}:{f:.3f}' for v, f in zip(values, fracs)])
            print(f"  {name}: {dist_str}")


if __name__ == '__main__':
    main()
