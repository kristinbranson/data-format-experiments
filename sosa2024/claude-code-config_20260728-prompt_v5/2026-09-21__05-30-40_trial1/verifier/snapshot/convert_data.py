#!/usr/bin/env python3
"""
Convert Sosa et al. 2025 NWB data to decoder-compatible format.

Usage:
    python -u convert_data.py <output_pickle> [--full|--sample] [--show-processing]
"""

import argparse
import glob
import os
import pickle
import time
import warnings

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter1d, minimum_filter1d, maximum_filter1d

# OASIS deconvolution from suite2p
from suite2p.extraction.dcnv import oasis

# ============================================================
# Constants
# ============================================================
TRACK_LENGTH = 450.0  # cm
REWARD_ZONES = {
    'A': (80, 130),
    'B': (200, 250),
    'C': (320, 370),
}
ZONE_CENTERS = {k: (v[0] + v[1]) / 2 for k, v in REWARD_ZONES.items()}
NEU_COEF = 0.7
TAU = 0.7  # calcium indicator time constant for OASIS
LICK_ERROR_THR = 0.35  # fraction of frames with cumsum lick > 2
BASELINE_WINDOW = 300  # samples for maximin baseline (~20s at 15.5 Hz)
DFF_SMOOTH_SIGMA = 2  # samples for Gaussian smoothing of dF/F

DATA_DIR = '/app/data'

# ============================================================
# Utility functions
# ============================================================

def nansmooth(a, sigma, axis=-1):
    """Gaussian smooth ignoring NaNs (from reference utilities.py)."""
    nan_inds = np.isnan(a)
    a_nanless = np.copy(a)
    a_nanless[nan_inds] = 0
    one = np.ones(a.shape)
    one[nan_inds] = 0.001
    a_nanless = gaussian_filter1d(a_nanless, sigma, axis=axis)
    one = gaussian_filter1d(one, sigma, axis=axis)
    return a_nanless / one


def compute_dff_and_deconvolve(F, Fneu, trial_starts, teleports, frame_rate, n_planes=1):
    """
    Compute dF/F and deconvolved events following reference preprocessing.py.

    Steps:
    1. Extract F within trial boundaries, set rest to NaN
    2. Neuropil subtraction: F -= NEU_COEF * Fneu
    3. Add back neuropil mean per trial
    4. Maximin baseline per trial
    5. dF/F = (F - baseline) / |baseline|
    6. Smooth dF/F with 2-sample Gaussian
    7. Deconvolve with OASIS

    Args:
        F: (n_timepoints, n_neurons) raw fluorescence
        Fneu: (n_timepoints, n_neurons) neuropil fluorescence
        trial_starts: array of trial start indices (1-indexed from NWB)
        teleports: array of teleport indices (1-indexed from NWB)
        frame_rate: imaging frame rate
        n_planes: number of imaging planes

    Returns:
        events: (n_neurons, n_timepoints) deconvolved events
        dff: (n_neurons, n_timepoints) dF/F
    """
    # Transpose to (n_neurons, n_timepoints) for processing
    F = F.T.astype(np.float64)
    Fneu = Fneu.T.astype(np.float64)
    n_neurons, n_timepoints = F.shape

    # Step 1: Extract within trials only
    f_ = np.full_like(F, np.nan)
    f_neu_ = np.full_like(Fneu, np.nan)

    start_inds = trial_starts.tolist()
    stop_inds = teleports.tolist()

    for start, stop in zip(start_inds, stop_inds):
        # Reference uses start-1:stop-1 (1-indexed to 0-indexed)
        s = start - 1
        e = stop - 1
        if s < 0:
            s = 0
        if e > n_timepoints:
            e = n_timepoints
        f_[:, s:e] = F[:, s:e]
        f_neu_[:, s:e] = Fneu[:, s:e]

    # Step 2: Neuropil subtraction
    f_ -= NEU_COEF * f_neu_

    # Step 3-4: Baseline computation and dF/F per trial
    nanmask = ~np.isnan(f_[0, :])
    flow = np.full_like(f_, np.nan)
    dff = np.full_like(f_, np.nan)
    spks = np.full_like(f_, np.nan)

    for start, stop in zip(start_inds, stop_inds):
        s = start - 1
        e = stop - 1
        if s < 0:
            s = 0
        if e > n_timepoints:
            e = n_timepoints

        # Add back neuropil mean per trial after subtraction
        f_[:, s:e] = f_[:, s:e] + NEU_COEF * np.nanmean(f_neu_[:, s:e], axis=1, keepdims=True)

        # Maximin baseline: smooth, min filter, max filter
        flow[:, s:e] = nansmooth(f_[:, s:e], 15, axis=1)
        flow[:, s:e] = minimum_filter1d(flow[:, s:e], BASELINE_WINDOW, axis=-1)
        flow[:, s:e] = maximum_filter1d(flow[:, s:e], BASELINE_WINDOW, axis=-1)

    # dF/F = (F - baseline) / |baseline|
    dff[:, nanmask] = (f_[:, nanmask] - flow[:, nanmask]) / np.abs(flow[:, nanmask])

    # Step 6: Smooth dF/F with 2-sample Gaussian per trial
    for start, stop in zip(start_inds, stop_inds):
        s = start - 1
        e = stop - 1
        if s < 0:
            s = 0
        if e > n_timepoints:
            e = n_timepoints
        dff[:, s:e] = nansmooth(dff[:, s:e], DFF_SMOOTH_SIGMA, axis=1)

        # Step 7: Deconvolve with OASIS
        trial_dff = dff[:, s:e].copy()
        trial_dff[np.isnan(trial_dff)] = 0
        spks[:, s:e] = oasis(trial_dff, 2000, TAU, frame_rate / n_planes)

    return spks, dff


def determine_reward_zone_per_trial(position, rz_signal, trial_starts, teleports):
    """
    Determine reward zone label (A, B, C) for each trial based on
    where the reward_zone signal is nonzero.

    Returns:
        zone_labels: list of str ('A', 'B', 'C') per trial
        zone_coords: list of (start, end) tuples per trial
    """
    n_trials = len(trial_starts)
    zone_labels = []
    zone_coords = []

    for i in range(n_trials):
        s = trial_starts[i] - 1  # convert to 0-indexed
        e = teleports[i] - 1

        trial_pos = position[s:e]
        trial_rz = rz_signal[s:e]

        # Find positions where reward zone signal is nonzero
        rz_active = trial_rz > 0
        if np.any(rz_active):
            rz_positions = trial_pos[rz_active]
            rz_center = np.mean(rz_positions)
        else:
            # No reward zone signal - use previous trial's zone
            if len(zone_labels) > 0:
                zone_labels.append(zone_labels[-1])
                zone_coords.append(zone_coords[-1])
                continue
            else:
                # Default to zone A
                zone_labels.append('A')
                zone_coords.append(REWARD_ZONES['A'])
                continue

        # Classify zone by center position
        dists = {k: abs(rz_center - c) for k, c in ZONE_CENTERS.items()}
        zone = min(dists, key=dists.get)
        zone_labels.append(zone)
        zone_coords.append(REWARD_ZONES[zone])

    return zone_labels, zone_coords


def compute_distance_to_reward_zone(position, zone_start, zone_end):
    """
    Compute signed distance from position to reward zone.
    Negative = before zone, 0 = in zone, positive = past zone.
    """
    dist = np.zeros_like(position)
    before = position < zone_start
    inside = (position >= zone_start) & (position <= zone_end)
    after = position > zone_end

    dist[before] = position[before] - zone_start  # negative
    dist[inside] = 0.0
    dist[after] = position[after] - zone_end  # positive

    return dist


def discretize_distance_to_rz(distance):
    """Discretize distance to reward zone into 7 bins per decoder spec."""
    bins = np.zeros(len(distance), dtype=np.int64)
    bins[distance < -50] = 0
    bins[(distance >= -50) & (distance < -10)] = 1
    bins[(distance >= -10) & (distance < 0)] = 2
    bins[distance == 0] = 3  # in zone
    bins[(distance > 0) & (distance <= 10)] = 4
    bins[(distance > 10) & (distance <= 50)] = 5
    bins[distance > 50] = 6
    return bins


def discretize_position(position):
    """Discretize position into 5 bins of 90cm each."""
    bins = np.zeros(len(position), dtype=np.int64)
    bins[position < 90] = 0
    bins[(position >= 90) & (position < 180)] = 1
    bins[(position >= 180) & (position < 270)] = 2
    bins[(position >= 270) & (position < 360)] = 3
    bins[position >= 360] = 4
    return bins


def discretize_speed(speed):
    """Discretize speed into 5 bins per decoder spec."""
    bins = np.zeros(len(speed), dtype=np.int64)
    speed_abs = np.abs(speed)
    bins[speed_abs < 2] = 0
    bins[(speed_abs >= 2) & (speed_abs < 10)] = 1
    bins[(speed_abs >= 10) & (speed_abs < 20)] = 2
    bins[(speed_abs >= 20) & (speed_abs < 40)] = 3
    bins[speed_abs >= 40] = 4
    return bins


def determine_reward_per_trial(reward_timestamps, behav_timestamps, trial_starts, teleports):
    """
    Determine if reward was delivered on each trial.
    Uses the Reward timestamps to check if any reward event falls within trial.
    """
    n_trials = len(trial_starts)
    rewarded = np.zeros(n_trials, dtype=np.int64)

    for i in range(n_trials):
        s = trial_starts[i] - 1
        e = teleports[i] - 1
        t_start = behav_timestamps[s]
        t_end = behav_timestamps[min(e, len(behav_timestamps) - 1)]

        # Check if any reward timestamp falls within trial
        in_trial = (reward_timestamps >= t_start) & (reward_timestamps <= t_end)
        if np.any(in_trial):
            rewarded[i] = 1

    return rewarded


# ============================================================
# Main processing function
# ============================================================

def process_session(nwb_path, show_processing=False, session_idx=0):
    """Process a single NWB file into decoder format."""
    t0 = time.time()

    subject = os.path.basename(os.path.dirname(nwb_path))
    ses_id = os.path.basename(nwb_path).split('_')[1]
    print(f"  Processing {subject} {ses_id}...")

    with h5py.File(nwb_path, 'r') as f:
        # ---- Load metadata ----
        imaging_rate = f['acquisition/TwoPhotonSeries/imaging_plane/imaging_rate'][()]

        # ---- Load behavioral data ----
        bts = f['processing/behavior/BehavioralTimeSeries']
        position = bts['position/data'][:]
        speed = bts['speed/data'][:]
        lick = bts['lick/data'][:]
        trial_num = bts['trial number/data'][:]
        trial_start_signal = bts['trial_start/data'][:]
        teleport_signal = bts['teleport/data'][:]
        environment = bts['environment/data'][:]
        rz_signal = bts['reward_zone/data'][:]
        behav_timestamps = bts['position/timestamps'][:]

        # Reward events
        reward_data = bts['Reward/data'][:]
        reward_timestamps = bts['Reward/timestamps'][:]

        # Autoreward
        autoreward = bts['autoreward/data'][:]

        # ---- Load neural data ----
        ophys = f['processing/ophys']
        iscell_full = ophys['ImageSegmentation/PlaneSegmentation/iscell'][:]
        plane_idx = ophys['ImageSegmentation/PlaneSegmentation/planeIdx'][:]

        # Get available planes
        planes = sorted([k for k in ophys['Fluorescence'].keys() if k.startswith('plane')])

        # Load and concatenate across planes
        F_all = []
        Fneu_all = []
        cell_mask_all = []

        for plane in planes:
            F_plane = ophys[f'Fluorescence/{plane}/data'][:]
            Fneu_plane = ophys[f'Neuropil/{plane}/data'][:]

            # Get plane number
            pnum = int(plane.replace('plane', ''))

            # iscell for this plane's ROIs
            plane_roi_mask = plane_idx == pnum
            iscell_plane = iscell_full[plane_roi_mask, 0]

            # Verify shape consistency
            assert F_plane.shape[1] == np.sum(plane_roi_mask), \
                f"Shape mismatch: F has {F_plane.shape[1]} cols but {np.sum(plane_roi_mask)} ROIs in plane {pnum}"

            F_all.append(F_plane)
            Fneu_all.append(Fneu_plane)
            cell_mask_all.append(iscell_plane > 0)

        # Concatenate across planes
        F_concat = np.concatenate(F_all, axis=1)
        Fneu_concat = np.concatenate(Fneu_all, axis=1)
        cell_mask = np.concatenate(cell_mask_all)

        n_planes = len(planes)

    # ---- Get trial indices ----
    trial_start_inds = np.where(trial_start_signal > 0)[0] + 1  # convert to 1-indexed
    teleport_inds = np.where(teleport_signal > 0)[0] + 1  # convert to 1-indexed

    if len(trial_start_inds) != len(teleport_inds):
        # Handle mismatch
        min_len = min(len(trial_start_inds), len(teleport_inds))
        trial_start_inds = trial_start_inds[:min_len]
        teleport_inds = teleport_inds[:min_len]

    n_trials = len(trial_start_inds)

    # ---- Compute dF/F and deconvolved events ----
    # Effective frame rate per plane
    effective_rate = imaging_rate  # Already per-plane for multi-plane
    if n_planes > 1:
        # For multi-plane, imaging_rate is the combined rate
        # Per-plane rate is imaging_rate / n_planes
        effective_rate = imaging_rate / n_planes

    t_dff = time.time()
    events, dff = compute_dff_and_deconvolve(
        F_concat, Fneu_concat, trial_start_inds, teleport_inds,
        imaging_rate, n_planes
    )
    print(f"    dF/F + deconv: {time.time() - t_dff:.1f}s")

    # Apply iscell filter
    cell_indices = np.where(cell_mask)[0]
    events_cells = events[cell_indices, :]  # (n_cells, n_timepoints)
    n_neurons = len(cell_indices)

    # ---- Determine reward zones ----
    zone_labels, zone_coords = determine_reward_zone_per_trial(
        position, rz_signal, trial_start_inds, teleport_inds
    )

    # ---- Determine reward per trial ----
    rewarded = determine_reward_per_trial(
        reward_timestamps, behav_timestamps, trial_start_inds, teleport_inds
    )

    # Also check autoreward
    for i in range(n_trials):
        s = trial_start_inds[i] - 1
        e = teleport_inds[i] - 1
        if np.any(autoreward[s:e] > 0):
            rewarded[i] = 1

    # ---- Determine environment per trial ----
    env_per_trial = np.zeros(n_trials, dtype=np.int64)
    for i in range(n_trials):
        s = trial_start_inds[i] - 1
        e = teleport_inds[i] - 1
        env_vals = environment[s:e]
        valid_env = env_vals[env_vals >= 0]
        if len(valid_env) > 0:
            env_per_trial[i] = int(np.median(valid_env))
        elif i > 0:
            env_per_trial[i] = env_per_trial[i - 1]

    # ---- Lick sensor error correction ----
    lick_corrected = lick.copy()
    for i in range(n_trials):
        s = trial_start_inds[i] - 1
        e = teleport_inds[i] - 1
        trial_lick = lick_corrected[s:e]
        n_frames = len(trial_lick)
        if n_frames > 0 and np.sum(trial_lick > 2) / n_frames > LICK_ERROR_THR:
            lick_corrected[s:e] = 0

    # Convert lick to binary
    lick_binary = (lick_corrected > 0).astype(np.int64)

    # ---- Build trial-level data ----
    neural_trials = []
    input_trials = []
    output_trials = []

    for i in range(n_trials):
        s = trial_start_inds[i] - 1  # 0-indexed start
        e = teleport_inds[i] - 1      # 0-indexed end

        if e <= s:
            continue

        n_tp = e - s

        # Neural: (n_neurons, n_timepoints)
        trial_neural = events_cells[:, s:e].copy()
        # Replace NaN with 0
        trial_neural[np.isnan(trial_neural)] = 0
        neural_trials.append(trial_neural.astype(np.float32))

        # ---- Inputs ----
        # Time from trial start (seconds)
        time_from_start = np.arange(n_tp, dtype=np.float32) / effective_rate

        # Per-trial inputs (broadcast to match time-varying shape)
        env_type = np.float32(env_per_trial[i])
        trial_number = np.float32(i)

        # Previous trial outcome
        if i == 0:
            prev_outcome = np.float32(0)  # no previous trial
        else:
            prev_outcome = np.float32(rewarded[i - 1])

        # input shape: (n_input, n_timepoints) for time-varying, (n_input,) for per-trial
        # Mix of time-varying and per-trial
        input_tv = time_from_start.reshape(1, -1)  # (1, n_tp)
        input_pt = np.array([env_type, trial_number, prev_outcome], dtype=np.float32)  # (3,)

        # Combine: first row is time-varying, rest are per-trial
        # Per spec: (n_input, n_timepoints) or (n_input,)
        # We'll use a mixed approach: time-varying inputs get full shape, per-trial get scalar
        inputs = np.vstack([
            input_tv,  # (1, n_tp) - time from start
            np.full((1, n_tp), env_type, dtype=np.float32),  # (1, n_tp) - env type
            np.full((1, n_tp), trial_number, dtype=np.float32),  # (1, n_tp) - trial number
            np.full((1, n_tp), prev_outcome, dtype=np.float32),  # (1, n_tp) - prev outcome
        ]).astype(np.float32)
        input_trials.append(inputs)

        # ---- Outputs ----
        trial_pos = position[s:e]
        trial_speed = speed[s:e]
        trial_lick = lick_binary[s:e]

        # Distance to reward zone
        zs, ze = zone_coords[i]
        dist = compute_distance_to_reward_zone(trial_pos, zs, ze)
        dist_binned = discretize_distance_to_rz(dist)

        # Absolute position
        pos_binned = discretize_position(trial_pos)

        # Speed
        speed_binned = discretize_speed(trial_speed)

        # Per-trial outputs
        zone_label = zone_labels[i]
        rz_loc = {'A': 0, 'B': 1, 'C': 2}[zone_label]
        reward_out = int(rewarded[i])

        outputs = np.vstack([
            dist_binned.reshape(1, -1),    # (1, n_tp) - distance to RZ
            pos_binned.reshape(1, -1),      # (1, n_tp) - absolute position
            speed_binned.reshape(1, -1),    # (1, n_tp) - speed
            trial_lick.reshape(1, -1),      # (1, n_tp) - lick
            np.full((1, n_tp), rz_loc, dtype=np.int64),      # (1, n_tp) - RZ location
            np.full((1, n_tp), reward_out, dtype=np.int64),   # (1, n_tp) - reward outcome
        ]).astype(np.int64)
        output_trials.append(outputs)

    elapsed = time.time() - t0
    print(f"    {subject} {ses_id}: {n_neurons} neurons, {len(neural_trials)} trials, {elapsed:.1f}s")

    # ---- Processing visualization ----
    if show_processing and session_idx < 2:
        plot_processing(
            subject, ses_id, position, speed, lick_binary,
            events_cells, dff[cell_indices, :], trial_start_inds, teleport_inds,
            zone_labels, zone_coords, rewarded, effective_rate,
            neural_trials, input_trials, output_trials
        )

    session_info = {
        'subject': subject.replace('sub-', ''),
        'session': ses_id,
        'n_neurons': n_neurons,
        'n_trials': len(neural_trials),
        'imaging_rate': effective_rate,
        'zone_labels': zone_labels,
        'rewarded': rewarded,
        'env': env_per_trial,
    }

    return neural_trials, input_trials, output_trials, session_info


def plot_processing(subject, ses_id, position, speed, lick, events, dff,
                    trial_starts, teleports, zone_labels, zone_coords,
                    rewarded, frame_rate, neural_trials, input_trials, output_trials):
    """Plot processing visualizations for sanity checking."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(6, 2, figsize=(20, 24))
    fig.suptitle(f'{subject} {ses_id}', fontsize=16)

    # Plot 1: Raw position and reward zone across trials
    ax = axes[0, 0]
    for i in range(min(len(trial_starts), 10)):
        s = trial_starts[i] - 1
        e = teleports[i] - 1
        t = np.arange(s, e) / frame_rate
        ax.plot(t, position[s:e], alpha=0.5)
        zs, ze = zone_coords[i]
        ax.axhspan(zs, ze, alpha=0.1, color='green')
    ax.set_title('Position (first 10 trials)')
    ax.set_ylabel('Position (cm)')
    ax.set_xlabel('Time (s)')

    # Plot 2: Neural activity example (first 3 cells)
    ax = axes[0, 1]
    n_cells_plot = min(3, events.shape[0])
    for c in range(n_cells_plot):
        s = trial_starts[0] - 1
        e = teleports[0] - 1
        t = np.arange(s, e) / frame_rate
        trace = events[c, s:e]
        trace[np.isnan(trace)] = 0
        ax.plot(t, trace + c * np.nanmax(trace) * 1.5, alpha=0.7)
    ax.set_title('Deconvolved events (trial 0, first 3 cells)')
    ax.set_xlabel('Time (s)')

    # Plot 3: dF/F example
    ax = axes[1, 0]
    for c in range(n_cells_plot):
        s = trial_starts[0] - 1
        e = teleports[0] - 1
        t = np.arange(s, e) / frame_rate
        trace = dff[c, s:e]
        trace[np.isnan(trace)] = 0
        ax.plot(t, trace + c * 2, alpha=0.7)
    ax.set_title('dF/F (trial 0, first 3 cells)')
    ax.set_xlabel('Time (s)')

    # Plot 4: Speed distribution
    ax = axes[1, 1]
    valid_speed = speed[speed > -100]
    ax.hist(valid_speed, bins=50, alpha=0.7)
    ax.axvline(2, color='r', linestyle='--', label='2 cm/s threshold')
    ax.set_title('Speed distribution')
    ax.set_xlabel('Speed (cm/s)')
    ax.legend()

    # Plot 5: Output discretization - distance to RZ
    ax = axes[2, 0]
    trial_idx = 5
    if trial_idx < len(output_trials):
        out = output_trials[trial_idx]
        t = np.arange(out.shape[1]) / frame_rate
        ax.plot(t, out[0, :], '.', markersize=2)
        ax.set_title(f'Distance to RZ (trial {trial_idx})')
        ax.set_ylabel('Bin')
        ax.set_yticks(range(7))

    # Plot 6: Output discretization - position
    ax = axes[2, 1]
    if trial_idx < len(output_trials):
        ax.plot(t, out[1, :], '.', markersize=2)
        ax.set_title(f'Absolute position (trial {trial_idx})')
        ax.set_ylabel('Bin')
        ax.set_yticks(range(5))

    # Plot 7: Output - speed
    ax = axes[3, 0]
    if trial_idx < len(output_trials):
        ax.plot(t, out[2, :], '.', markersize=2)
        ax.set_title(f'Speed (trial {trial_idx})')
        ax.set_ylabel('Bin')
        ax.set_yticks(range(5))

    # Plot 8: Output - lick
    ax = axes[3, 1]
    if trial_idx < len(output_trials):
        ax.plot(t, out[3, :], '.', markersize=2)
        ax.set_title(f'Lick (trial {trial_idx})')
        ax.set_ylabel('0/1')

    # Plot 9: Reward zone per trial
    ax = axes[4, 0]
    zone_numeric = [{'A': 0, 'B': 1, 'C': 2}[z] for z in zone_labels]
    ax.plot(zone_numeric, 'o-', markersize=3)
    ax.set_title('Reward zone per trial')
    ax.set_ylabel('Zone (A=0, B=1, C=2)')
    ax.set_xlabel('Trial')
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(['A', 'B', 'C'])

    # Plot 10: Reward outcome per trial
    ax = axes[4, 1]
    ax.plot(rewarded, 'o', markersize=3)
    reward_rate = np.mean(rewarded)
    ax.set_title(f'Reward outcome (rate={reward_rate:.2f})')
    ax.set_ylabel('Rewarded')
    ax.set_xlabel('Trial')

    # Plot 11: Input time from start
    ax = axes[5, 0]
    if trial_idx < len(input_trials):
        inp = input_trials[trial_idx]
        t = np.arange(inp.shape[1]) / frame_rate
        ax.plot(t, inp[0, :])
        ax.set_title(f'Input: time from trial start (trial {trial_idx})')
        ax.set_xlabel('Time (s)')

    # Plot 12: Neural activity heatmap for one trial
    ax = axes[5, 1]
    if trial_idx < len(neural_trials):
        neural = neural_trials[trial_idx]
        n_show = min(50, neural.shape[0])
        ax.imshow(neural[:n_show, :], aspect='auto', cmap='hot', interpolation='none')
        ax.set_title(f'Neural activity (trial {trial_idx}, first {n_show} neurons)')
        ax.set_xlabel('Time (frames)')
        ax.set_ylabel('Neuron')

    plt.tight_layout()
    fname = f'/app/processing_{subject}_{ses_id}.png'
    plt.savefig(fname, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"    Saved processing plot: {fname}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', help='Output pickle file path')
    parser.add_argument('--full', action='store_true', default=True, help='Process all sessions')
    parser.add_argument('--sample', action='store_true', help='Process only 2 sessions')
    parser.add_argument('--show-processing', action='store_true', help='Plot processing visualizations')
    args = parser.parse_args()

    if args.sample:
        args.full = False

    # ---- Discover all NWB files ----
    subjects = sorted([d for d in os.listdir(DATA_DIR) if d.startswith('sub-')])

    all_nwb_files = []
    for subj in subjects:
        subj_dir = os.path.join(DATA_DIR, subj)
        nwb_files = sorted(glob.glob(os.path.join(subj_dir, '*.nwb')))
        all_nwb_files.extend(nwb_files)

    if args.sample:
        # Pick 2 sessions from different subjects for testing
        all_nwb_files = [all_nwb_files[0], all_nwb_files[14]]  # m11-ses03, m12-ses01
        print(f"Sample mode: processing {len(all_nwb_files)} sessions")
    else:
        print(f"Full mode: processing {len(all_nwb_files)} sessions")

    # ---- Process sessions ----
    neural_all = []
    input_all = []
    output_all = []
    subject_list = []
    session_infos = []

    total_start = time.time()

    for idx, nwb_path in enumerate(all_nwb_files):
        t_sess = time.time()
        neural_trials, input_trials, output_trials, session_info = process_session(
            nwb_path, show_processing=args.show_processing, session_idx=idx
        )

        if len(neural_trials) < 2:
            print(f"    WARNING: Skipping session with {len(neural_trials)} trials")
            continue

        neural_all.append(neural_trials)
        input_all.append(input_trials)
        output_all.append(output_trials)
        session_infos.append(session_info)

        elapsed_total = time.time() - total_start
        sessions_done = idx + 1
        est_remaining = elapsed_total / sessions_done * (len(all_nwb_files) - sessions_done)
        print(f"    [{sessions_done}/{len(all_nwb_files)}] "
              f"Elapsed: {elapsed_total:.0f}s, Est remaining: {est_remaining:.0f}s")

    # ---- Build subject index ----
    unique_subjects = sorted(set(si['subject'] for si in session_infos))
    subject_idx = np.array([unique_subjects.index(si['subject']) for si in session_infos])

    # ---- Build brain region index ----
    brain_regions = ['CA1']
    brain_region_idx = []
    for si in session_infos:
        brain_region_idx.append(np.zeros(si['n_neurons'], dtype=np.int64))

    # ---- Build metadata ----
    data = {
        'neural': neural_all,
        'input': input_all,
        'output': output_all,

        'subjects': unique_subjects,
        'subject_idx': subject_idx,

        'brain_regions': brain_regions,
        'brain_region_idx': brain_region_idx,

        'input_names': [
            'time_from_trial_start',
            'environment_type',
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
            ['< -50cm', '-50 to -10cm', '-10 to 0cm', '0cm (in zone)', '0 to +10cm', '+10 to +50cm', '> +50cm'],
            ['0-90cm', '90-180cm', '180-270cm', '270-360cm', '360-450cm'],
            ['< 2 cm/s', '2-10 cm/s', '10-20 cm/s', '20-40 cm/s', '> 40 cm/s'],
            ['no lick', 'lick'],
            ['Zone A', 'Zone B', 'Zone C'],
            ['no reward', 'reward'],
        ],

        'metadata': {
            'task_description': 'Virtual reality navigation with hidden reward zone switches. '
                               'Mice run on 450cm linear track, lick to receive reward in hidden 50cm zone. '
                               'Zone location switches after 30 trials on switch days.',
            'time_bin_size': 1000.0 / 15.5078125,  # ms per frame
            'temporal_alignment_event': 'start of trial (trial_start signal)',
            'off_start': 0.0,  # trial starts at alignment event
            'off_end': None,  # variable trial lengths
            'imaging_rate_hz': 15.5078125,
            'track_length_cm': 450.0,
            'reward_zones': {'A': [80, 130], 'B': [200, 250], 'C': [320, 370]},
            'session_info': session_infos,
            'n_sessions': len(session_infos),
            'n_subjects': len(unique_subjects),
        }
    }

    # ---- Print summary ----
    total_neurons = sum(si['n_neurons'] for si in session_infos)
    total_trials = sum(si['n_trials'] for si in session_infos)
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Sessions: {len(session_infos)}")
    print(f"  Subjects: {len(unique_subjects)} ({', '.join(unique_subjects)})")
    print(f"  Total neurons: {total_neurons}")
    print(f"  Total trials: {total_trials}")
    print(f"  Mean neurons/session: {total_neurons/len(session_infos):.1f}")
    print(f"  Mean trials/session: {total_trials/len(session_infos):.1f}")
    print(f"  Total time: {time.time() - total_start:.1f}s")

    # ---- Save ----
    print(f"\nSaving to {args.output}...")
    with open(args.output, 'wb') as f:
        pickle.dump(data, f, protocol=4)

    file_size = os.path.getsize(args.output)
    print(f"Saved ({file_size / 1e9:.2f} GB)")


if __name__ == '__main__':
    main()
