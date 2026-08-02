#!/usr/bin/env python3
"""
Convert NWB data from the Mesoscale Activity Map dataset to decoder-compatible format.

Usage:
    python -u convert_data.py <outpicklefile> [--sample] [--full] [--show-processing]
"""

import os
import sys
import time
import pickle
import argparse
import warnings
import numpy as np
from collections import Counter

import pynwb
from pynwb import NWBHDF5IO

warnings.filterwarnings('ignore', category=UserWarning)

# ============================================================================
# Constants
# ============================================================================
DATA_DIR = '/app/data'
BIN_SIZE_S = 0.05  # 50 ms bins
T_START = -2.5     # seconds before go cue
T_END = 1.5        # seconds after go cue
N_TIMEBINS = int((T_END - T_START) / BIN_SIZE_S)  # 80

# 14 coarse brain regions from the reference code
COARSE_REGIONS = [
    'ALM', 'Cerebellum', 'CorticalSubplate', 'Hippocampus', 'Hypothalamus',
    'Medulla', 'Midbrain', 'Olfactory', 'Orbital', 'OtherCortex',
    'Pallidum', 'Pons', 'Striatum', 'Thalamus'
]

# ============================================================================
# Brain region mapping
# ============================================================================
def map_anno_to_region(anno):
    """Map fine CCF annotation to one of 14 coarse regions."""
    a = anno.lower().strip()
    if not a:
        return None

    # ALM: Secondary motor area
    if 'secondary motor area' in a:
        return 'ALM'

    # Orbital
    if 'orbital area' in a:
        return 'Orbital'

    # Olfactory
    if any(x in a for x in ['anterior olfactory', 'piriform', 'olfactory area',
                              'olfactory tubercle', 'taenia tecta', 'accessory olfactory',
                              'dorsal peduncular area']):
        return 'Olfactory'

    # Hippocampus
    if any(x in a for x in ['field ca', 'dentate gyrus', 'postsubiculum',
                              'subiculum', 'hippocampal formation', 'entorhinal area']):
        return 'Hippocampus'

    # CorticalSubplate
    if any(x in a for x in ['endopiriform', 'claustrum', 'cortical subplate',
                              'amygdal', 'substantia innominata',
                              'bed nuclei of the stria terminalis', 'nucleus sagulum',
                              'lateral septal', 'septofimbrial',
                              'triangular nucleus of septum']):
        return 'CorticalSubplate'

    # Pallidum
    if any(x in a for x in ['globus pallidus', 'pallidum']):
        return 'Pallidum'

    # Striatum
    if any(x in a for x in ['caudoputamen', 'striatum', 'nucleus accumbens',
                              'fundus of striatum']):
        return 'Striatum'

    # Cerebellum
    if any(x in a for x in ['cerebellum', 'lobule', 'simple lobule', 'copula pyramidis',
                              'nodulus', 'uvula', 'paramedian lobule', 'declive',
                              'crus ', 'pyramus', 'lingula', 'interposed nucleus',
                              'fastigial nucleus', 'infracerebellar']):
        return 'Cerebellum'

    # Thalamus (must be before Hypothalamus since some share substrings)
    if any(x in a for x in ['thalamus', 'habenula', 'geniculate',
                              'paracentral nucleus', 'parafascicular',
                              'anteromedial nucleus', 'rhomboid nucleus',
                              'anterodorsal nucleus', 'perireunensis']):
        return 'Thalamus'

    # Hypothalamus
    if any(x in a for x in ['hypothalamus', 'hypothalamic', 'zona incerta',
                              'fields of forel', 'subthalamic', 'tuberomammillary',
                              'parasubthalamic', 'lateral preoptic',
                              'lateral hypothalamic']):
        return 'Hypothalamus'

    # Pons
    if any(x in a for x in ['pons', 'pontine reticular', 'pedunculopontine',
                              'parabrachial', 'tegmental reticular', 'locus ceruleus',
                              'koelliker-fuse', 'nucleus of the lateral lemniscus']):
        return 'Pons'

    # Medulla
    if any(x in a for x in ['medulla', 'gigantocellular', 'intermediate reticular',
                              'magnocellular reticular', 'paragigantocellular',
                              'spinal nucleus of the trigeminal', 'inferior olivary',
                              'medullary reticular', 'vestibular', 'hypoglossal',
                              'dorsal motor nucleus of the vagus', 'facial motor',
                              'external cuneate', 'nucleus of the solitary',
                              'raphe magnus', 'raphe obscurus',
                              'lateral reticular nucleus', 'parasolitary',
                              'parapyramidal', 'nucleus of roller', 'nucleus x',
                              'parvicellular reticular']):
        return 'Medulla'

    # Midbrain
    if any(x in a for x in ['midbrain', 'superior colliculus', 'substantia nigra',
                              'red nucleus', 'ventral tegmental', 'inferior colliculus',
                              'periaqueductal', 'pretectal', 'peripeduncular',
                              'nucleus of the brachium', 'nucleus of the optic tract',
                              'dorsal terminal nucleus', 'medial terminal nucleus',
                              'subparafascicular']):
        return 'Midbrain'

    # OtherCortex: everything else that's cortical
    if any(x in a for x in ['primary motor', 'primary somatosensory', 'visual area',
                              'auditory area', 'retrosplenial', 'supplemental somatosensory',
                              'agranular insular', 'anterior cingulate', 'prelimbic',
                              'infralimbic', 'gustatory', 'visceral area',
                              'temporal association', 'perirhinal', 'ectorhinal',
                              'frontal pole', 'primary auditory']):
        return 'OtherCortex'

    # If nothing matched, print warning and return None
    return None


# ============================================================================
# Data loading and processing functions
# ============================================================================

def get_nwb_files(data_dir):
    """Get list of all NWB files organized by subject."""
    subjects = sorted([d for d in os.listdir(data_dir) if d.startswith('sub-')])
    nwb_files = []
    for subj in subjects:
        subj_dir = os.path.join(data_dir, subj)
        files = sorted([f for f in os.listdir(subj_dir) if f.endswith('.nwb')])
        for fname in files:
            nwb_files.append((subj, os.path.join(subj_dir, fname)))
    return nwb_files


def compute_firing_rates(spike_times_list, go_cue_time, t_start, t_end, bin_size):
    """
    Compute firing rates for a set of neurons in a single trial.

    Args:
        spike_times_list: list of arrays, one per neuron (absolute times)
        go_cue_time: float, absolute time of go cue
        t_start, t_end: float, time window relative to go cue
        bin_size: float, bin width in seconds

    Returns:
        fr: array (n_neurons, n_timebins), firing rates in Hz
    """
    n_neurons = len(spike_times_list)
    n_bins = int((t_end - t_start) / bin_size)
    fr = np.zeros((n_neurons, n_bins), dtype=np.float32)

    abs_start = go_cue_time + t_start
    abs_end = go_cue_time + t_end

    for i, st in enumerate(spike_times_list):
        if len(st) == 0:
            continue
        # Filter spikes in window
        mask = (st >= abs_start) & (st < abs_end)
        spikes_in_window = st[mask]
        if len(spikes_in_window) == 0:
            continue
        # Assign spikes to bins
        bin_indices = ((spikes_in_window - abs_start) / bin_size).astype(int)
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        np.add.at(fr[i], bin_indices, 1.0)

    # Convert counts to rates
    fr /= bin_size
    return fr


def find_last_sample_before_go(sample_start_times, go_cue_time):
    """Find the last sample onset time before a given go cue time."""
    # Sample starts that are before the go cue
    valid = sample_start_times[sample_start_times < go_cue_time]
    if len(valid) == 0:
        return None
    return valid[-1]  # Last one before go cue


def compute_time_from_tone(go_cue_time, tone_onset_time, t_start, t_end, bin_size):
    """
    Compute time from tone onset at each time bin center.

    Returns array of shape (n_timebins,) with values in seconds.
    """
    n_bins = int((t_end - t_start) / bin_size)
    bin_centers = np.arange(n_bins) * bin_size + t_start + bin_size / 2
    # bin_centers are relative to go cue
    # tone_onset is absolute, go_cue is absolute
    tone_rel = tone_onset_time - go_cue_time  # tone onset relative to go cue
    time_from_tone = bin_centers - tone_rel  # time since tone onset
    return time_from_tone.astype(np.float32)


def compute_photostim_timeseries(go_cue_time, photostim_starts, photostim_stops,
                                  t_start, t_end, bin_size):
    """
    Compute binary photostim time series (1 if stim on, 0 if off).

    Returns array of shape (n_timebins,).
    """
    n_bins = int((t_end - t_start) / bin_size)
    bin_centers = np.arange(n_bins) * bin_size + t_start + bin_size / 2
    # bin_centers are relative to go cue; convert to absolute
    abs_centers = bin_centers + go_cue_time

    stim = np.zeros(n_bins, dtype=np.float32)
    for start, stop in zip(photostim_starts, photostim_stops):
        mask = (abs_centers >= start) & (abs_centers <= stop)
        stim[mask] = 1.0
    return stim


def get_tongue_y_for_trial(tongue_data, tongue_timestamps, tongue_likelihood,
                            go_cue_time, t_start, t_end, bin_size):
    """
    Get tongue y-position for each time bin in a trial.

    Returns array of shape (n_timebins,) or None if no data.
    """
    n_bins = int((t_end - t_start) / bin_size)
    bin_edges = np.arange(n_bins + 1) * bin_size + t_start + go_cue_time

    # Find tongue data in the trial window
    abs_start = go_cue_time + t_start
    abs_end = go_cue_time + t_end

    mask = (tongue_timestamps >= abs_start) & (tongue_timestamps < abs_end)
    if mask.sum() == 0:
        return np.full(n_bins, np.nan, dtype=np.float32)

    t_in_window = tongue_timestamps[mask]
    y_in_window = tongue_data[mask]
    like_in_window = tongue_likelihood[mask]

    # Low likelihood → set to NaN
    y_in_window = y_in_window.copy()
    y_in_window[like_in_window < 0.9] = np.nan

    # Bin the tongue y-position (vectorized)
    tongue_y_binned = np.full(n_bins, np.nan, dtype=np.float32)
    bin_indices = np.digitize(t_in_window, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    # Use only non-NaN values
    valid_mask = ~np.isnan(y_in_window)
    if valid_mask.any():
        valid_bins = bin_indices[valid_mask]
        valid_vals = y_in_window[valid_mask]
        sums = np.bincount(valid_bins, weights=valid_vals, minlength=n_bins)
        counts = np.bincount(valid_bins, minlength=n_bins)
        has_data = counts > 0
        tongue_y_binned[has_data] = (sums[has_data] / counts[has_data]).astype(np.float32)

    return tongue_y_binned


def get_valid_trial_indices(nwb, good_unit_indices, n_trials):
    """
    Determine which trials have valid recording for ALL good units.
    Uses obs_intervals to match trials with recording periods.

    Returns array of valid trial indices (0-based).
    """
    trial_starts = nwb.trials['start_time'][:]

    # Get obs_intervals for all good units and find common valid trials
    # Group units by their recording period (obs_intervals length)
    obs_sets = {}
    for idx in good_unit_indices:
        obs = nwb.units['obs_intervals'][idx]
        n_obs = len(obs)
        if n_obs not in obs_sets:
            obs_sets[n_obs] = obs

    # Find trials covered by ALL recording periods (vectorized)
    valid_trials = np.ones(n_trials, dtype=bool)

    for n_obs, obs in obs_sets.items():
        obs_starts = obs[:, 0]
        # Vectorized: compute min distance from each trial to any obs_start
        # Shape: (n_trials, n_obs)
        diffs = np.abs(trial_starts[:, None] - obs_starts[None, :])
        min_diffs = diffs.min(axis=1)
        covered = min_diffs < 1.0
        valid_trials &= covered

    return np.where(valid_trials)[0]


def process_session(nwb_path, subject_id):
    """
    Process a single NWB session file.

    Returns:
        session_data dict or None if session should be skipped.
    """
    io = NWBHDF5IO(nwb_path, 'r')
    nwb = io.read()

    # ---- Check for good units ----
    classification = nwb.units['classification'][:]
    good_mask = np.array([c == 'good' for c in classification])
    n_good = good_mask.sum()

    if n_good == 0:
        io.close()
        return None

    # ---- Get neuron data ----
    anno_names = nwb.units['anno_name'][:][good_mask]

    # Map to coarse brain regions
    region_indices = []
    valid_neuron_mask = np.ones(n_good, dtype=bool)
    for i, anno in enumerate(anno_names):
        region = map_anno_to_region(str(anno))
        if region is None:
            valid_neuron_mask[i] = False
            region_indices.append(-1)
        else:
            region_indices.append(COARSE_REGIONS.index(region))
    region_indices = np.array(region_indices)

    # Apply valid neuron mask
    good_unit_indices = np.where(good_mask)[0]
    good_unit_indices = good_unit_indices[valid_neuron_mask]
    region_indices = region_indices[valid_neuron_mask]
    n_neurons = len(good_unit_indices)

    if n_neurons == 0:
        io.close()
        return None

    # ---- Determine valid trials (based on obs_intervals) ----
    n_trials_total = len(nwb.trials)
    valid_trial_idx = get_valid_trial_indices(nwb, good_unit_indices, n_trials_total)

    # ---- Get spike times for good units ----
    spike_times_per_unit = []
    for idx in good_unit_indices:
        st = nwb.units['spike_times'][idx]
        spike_times_per_unit.append(st)

    # ---- Get trial data ----
    trials = nwb.trials
    outcomes = trials['outcome'][:]
    instructions = trials['trial_instruction'][:]
    early_lick = trials['early_lick'][:]
    auto_water = trials['auto_water'][:]
    free_water = trials['free_water'][:]

    # ---- Get behavioral event times ----
    events = nwb.acquisition['BehavioralEvents']
    go_start_times = events.time_series['go_start_times'].timestamps[:]
    sample_start_times = events.time_series['sample_start_times'].timestamps[:]

    # Photostim event times (absolute)
    photostim_start_abs = events.time_series['photostim_start_times'].timestamps[:]
    photostim_stop_abs = events.time_series['photostim_stop_times'].timestamps[:]

    # ---- Get tongue tracking data ----
    tongue_ts = nwb.acquisition['BehavioralTimeSeries'].time_series['Camera0_side_TongueTracking']
    tongue_data_all = tongue_ts.data[:]  # (n_frames, 3): x, y, likelihood
    tongue_timestamps = tongue_ts.timestamps[:]
    tongue_y_all = tongue_data_all[:, 1].astype(np.float32)
    tongue_likelihood = tongue_data_all[:, 2].astype(np.float32)

    # ---- Filter trials: exclude auto_water, free_water, and trials without recording ----
    trial_mask = (auto_water == 0) & (free_water == 0)
    # Intersect with valid recording trials
    recording_mask = np.zeros(n_trials_total, dtype=bool)
    recording_mask[valid_trial_idx] = True
    trial_mask = trial_mask & recording_mask
    trial_indices = np.where(trial_mask)[0]

    if len(trial_indices) < 2:
        io.close()
        return None

    # ---- Process each trial ----
    neural_trials = []
    input_trials = []
    output_trials_raw = []
    tongue_y_session = []  # for percentile computation

    for ti in trial_indices:
        go_cue = go_start_times[ti]

        # Compute firing rates
        fr = compute_firing_rates(spike_times_per_unit, go_cue, T_START, T_END, BIN_SIZE_S)
        neural_trials.append(fr)

        # Input 0: time from tone onset
        tone_onset = find_last_sample_before_go(sample_start_times, go_cue)
        if tone_onset is None:
            tone_onset = go_cue - 1.85
        time_from_tone = compute_time_from_tone(go_cue, tone_onset, T_START, T_END, BIN_SIZE_S)

        # Input 1: photostim on/off
        trial_abs_start = go_cue + T_START
        trial_abs_end = go_cue + T_END
        stim_mask = (photostim_stop_abs > trial_abs_start) & (photostim_start_abs < trial_abs_end)
        ps_starts = photostim_start_abs[stim_mask]
        ps_stops = photostim_stop_abs[stim_mask]
        photostim_ts = compute_photostim_timeseries(go_cue, ps_starts, ps_stops,
                                                      T_START, T_END, BIN_SIZE_S)

        input_data = np.stack([time_from_tone, photostim_ts], axis=0)
        input_trials.append(input_data)

        # Output 0: choice (left=0, right=1)
        choice = 0 if instructions[ti] == 'left' else 1

        # Output 1: outcome (ignore=0, miss=1, hit=2)
        outcome_str = outcomes[ti]
        outcome_val = {'ignore': 0, 'miss': 1, 'hit': 2}.get(outcome_str, 0)

        # Output 2: early lick (no=0, yes=1)
        early_val = 0 if early_lick[ti] == 'no early' else 1

        # Output 3: tongue y-position (discretized later)
        tongue_y_trial = get_tongue_y_for_trial(tongue_y_all, tongue_timestamps,
                                                  tongue_likelihood, go_cue,
                                                  T_START, T_END, BIN_SIZE_S)

        output_trials_raw.append({
            'choice': choice,
            'outcome': outcome_val,
            'early_lick': early_val,
            'tongue_y_raw': tongue_y_trial
        })

        valid_y = tongue_y_trial[~np.isnan(tongue_y_trial)]
        if len(valid_y) > 0:
            tongue_y_session.extend(valid_y.tolist())

    io.close()

    # ---- Discretize tongue y-position per session ----
    if len(tongue_y_session) > 0:
        tongue_y_arr = np.array(tongue_y_session)
        p40 = np.percentile(tongue_y_arr, 40)
        p60 = np.percentile(tongue_y_arr, 60)
    else:
        p40, p60 = 0, 0

    # Build final output arrays
    final_outputs = []
    for out_dict in output_trials_raw:
        tongue_y_raw = out_dict['tongue_y_raw']
        tongue_y_disc = np.zeros(N_TIMEBINS, dtype=np.float32)
        valid = ~np.isnan(tongue_y_raw)
        tongue_y_disc[valid & (tongue_y_raw < p40)] = 0
        tongue_y_disc[valid & (tongue_y_raw >= p40) & (tongue_y_raw <= p60)] = 1
        tongue_y_disc[valid & (tongue_y_raw > p60)] = 2

        # Output shape: (4, n_timebins) - all time-varying
        # Per-trial outputs are replicated across time
        full_output = np.zeros((4, N_TIMEBINS), dtype=np.int64)
        full_output[0, :] = out_dict['choice']
        full_output[1, :] = out_dict['outcome']
        full_output[2, :] = out_dict['early_lick']
        full_output[3, :] = tongue_y_disc.astype(np.int64)

        final_outputs.append(full_output)

    return {
        'neural': neural_trials,
        'input': input_trials,
        'output': final_outputs,
        'subject_id': subject_id,
        'n_neurons': n_neurons,
        'n_trials': len(neural_trials),
        'region_indices': region_indices,
        'session_name': os.path.basename(nwb_path),
    }


# ============================================================================
# Visualization
# ============================================================================

def plot_processing(session_data, session_idx, nwb_path):
    """Plot visualizations of processing steps for a session."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    neural = session_data['neural']
    inputs = session_data['input']
    outputs = session_data['output']
    n_trials = len(neural)

    fig, axes = plt.subplots(4, 4, figsize=(20, 16))
    fig.suptitle(f'Session: {session_data["session_name"]}\n'
                 f'Neurons: {session_data["n_neurons"]}, Trials: {n_trials}',
                 fontsize=14)

    time_axis = np.arange(N_TIMEBINS) * BIN_SIZE_S + T_START + BIN_SIZE_S / 2

    # Pick a few example trials
    trial_indices = np.linspace(0, n_trials - 1, min(4, n_trials), dtype=int)

    for col, ti in enumerate(trial_indices):
        # Row 0: Neural activity (subset of neurons)
        ax = axes[0, col]
        fr = neural[ti]
        n_show = min(20, fr.shape[0])
        ax.imshow(fr[:n_show], aspect='auto', extent=[T_START, T_END, n_show, 0])
        ax.set_title(f'Trial {ti}')
        ax.axvline(0, color='r', linestyle='--', alpha=0.5, label='Go cue')
        if col == 0:
            ax.set_ylabel('Neural (neurons)')

        # Row 1: Inputs
        ax = axes[1, col]
        ax.plot(time_axis, inputs[ti][0], label='Time from tone', color='blue')
        ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
        ax2 = ax.twinx()
        ax2.fill_between(time_axis, 0, inputs[ti][1], alpha=0.3, color='red', label='Photostim')
        ax2.set_ylim(-0.1, 1.5)
        if col == 0:
            ax.set_ylabel('Time from tone (s)')

        # Row 2: Outputs (per-trial)
        ax = axes[2, col]
        out = outputs[ti]
        choice_str = 'Left' if out[0, 0] == 0 else 'Right'
        outcome_map = {0: 'Ignore', 1: 'Miss', 2: 'Hit'}
        outcome_str = outcome_map.get(int(out[1, 0]), '?')
        early_str = 'Yes' if out[2, 0] == 1 else 'No'
        ax.text(0.5, 0.5, f'Choice: {choice_str}\nOutcome: {outcome_str}\nEarly lick: {early_str}',
                transform=ax.transAxes, ha='center', va='center', fontsize=12)
        ax.set_xlim(T_START, T_END)
        if col == 0:
            ax.set_ylabel('Trial outputs')

        # Row 3: Tongue y-position
        ax = axes[3, col]
        ax.plot(time_axis, out[3], color='green')
        ax.set_ylim(-0.5, 2.5)
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(['<40th', '40-60th', '>60th'])
        ax.set_xlabel('Time (s)')
        if col == 0:
            ax.set_ylabel('Tongue y class')

    plt.tight_layout()
    session_name = os.path.splitext(os.path.basename(nwb_path))[0]
    plt.savefig(f'processing_{session_name}.png', dpi=100)
    plt.close()


# ============================================================================
# Main conversion
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Convert NWB data to decoder format.')
    parser.add_argument('output_file', type=str, help='Output pickle file path')
    parser.add_argument('--sample', action='store_true', help='Process only 2 sessions')
    parser.add_argument('--full', action='store_true', help='Process all sessions (default)')
    parser.add_argument('--show-processing', action='store_true', help='Plot processing visualizations')
    args = parser.parse_args()

    if not args.sample:
        args.full = True

    print(f"=== NWB to Decoder Format Conversion ===")
    print(f"Output: {args.output_file}")
    print(f"Mode: {'sample (2 sessions)' if args.sample else 'full'}")
    print(f"Bin size: {BIN_SIZE_S*1000:.0f} ms")
    print(f"Time window: {T_START} to {T_END} s (relative to go cue)")
    print(f"Time bins: {N_TIMEBINS}")
    print()

    # Get all NWB files
    nwb_files = get_nwb_files(DATA_DIR)
    print(f"Found {len(nwb_files)} NWB files from {len(set(s for s, _ in nwb_files))} subjects")

    if args.sample:
        # Pick 2 sessions from different subjects
        nwb_files = nwb_files[:2]
        print(f"Sample mode: processing {len(nwb_files)} sessions")

    # Process sessions
    all_sessions = []
    subjects_list = []
    subject_idx_list = []
    brain_region_idx_list = []

    subjects_seen = {}
    total_neurons = 0
    total_trials = 0

    overall_start = time.time()

    for i, (subject_id, nwb_path) in enumerate(nwb_files):
        sess_start = time.time()
        print(f"[{i+1}/{len(nwb_files)}] Processing {os.path.basename(nwb_path)}...", end=' ', flush=True)

        result = process_session(nwb_path, subject_id)

        if result is None:
            print("SKIPPED (no good units or trials)")
            continue

        # Track subjects
        if subject_id not in subjects_seen:
            subjects_seen[subject_id] = len(subjects_seen)
            subjects_list.append(subject_id)
        subj_idx = subjects_seen[subject_id]

        all_sessions.append(result)
        subject_idx_list.append(subj_idx)
        brain_region_idx_list.append(result['region_indices'])

        total_neurons += result['n_neurons']
        total_trials += result['n_trials']

        elapsed = time.time() - sess_start
        print(f"Done ({result['n_neurons']} neurons, {result['n_trials']} trials, {elapsed:.1f}s)")

        # Show processing plots
        if args.show_processing and i < 2:
            plot_processing(result, i, nwb_path)
            print(f"  Saved processing plot")

    total_time = time.time() - overall_start
    print(f"\nProcessing complete: {len(all_sessions)} sessions, "
          f"{total_neurons} neurons, {total_trials} trials in {total_time:.1f}s")

    # ---- Assemble final data structure ----
    print("\nAssembling output data structure...")

    data = {
        'neural': [s['neural'] for s in all_sessions],
        'input': [s['input'] for s in all_sessions],
        'output': [s['output'] for s in all_sessions],

        'subjects': subjects_list,
        'subject_idx': np.array(subject_idx_list, dtype=np.int64),

        'brain_regions': COARSE_REGIONS,
        'brain_region_idx': brain_region_idx_list,

        'input_names': ['time_from_tone_onset', 'photostim_on'],
        'output_names': ['choice', 'outcome', 'early_lick', 'tongue_y_position'],
        'output_values': [
            ['left', 'right'],
            ['ignore', 'miss', 'hit'],
            ['no', 'yes'],
            ['below_40th', '40th_to_60th', 'above_60th'],
        ],

        'metadata': {
            'task_description': 'Auditory delayed response task: mice hear tone, wait through delay, then lick left/right to report instruction',
            'time_bin_size': BIN_SIZE_S * 1000,  # in ms
            'temporal_alignment_event': 'Go cue onset',
            'off_start': T_START,
            'off_end': T_END,
            'dataset': 'Mesoscale Activity Map (DANDI:000363)',
            'n_sessions': len(all_sessions),
            'n_subjects': len(subjects_list),
            'total_neurons': total_neurons,
            'total_trials': total_trials,
            'neuron_filter': 'classification == good (QC classifier)',
            'trial_filter': 'excluded auto_water and free_water trials',
        }
    }

    # Save
    print(f"Saving to {args.output_file}...")
    with open(args.output_file, 'wb') as f:
        pickle.dump(data, f, protocol=4)

    file_size = os.path.getsize(args.output_file) / (1024**2)
    print(f"Saved ({file_size:.1f} MB)")

    # Print summary statistics
    print(f"\n=== Summary ===")
    print(f"Sessions: {len(all_sessions)}")
    print(f"Subjects: {len(subjects_list)}")
    print(f"Total neurons: {total_neurons}")
    print(f"Total trials: {total_trials}")
    print(f"Mean neurons/session: {total_neurons/len(all_sessions):.1f}")
    print(f"Mean trials/session: {total_trials/len(all_sessions):.1f}")

    # Output distributions
    all_choices = []
    all_outcomes = []
    all_early = []
    for session_outputs in data['output']:
        for trial_out in session_outputs:
            all_choices.append(int(trial_out[0, 0]))
            all_outcomes.append(int(trial_out[1, 0]))
            all_early.append(int(trial_out[2, 0]))

    print(f"\nOutput distributions:")
    print(f"  Choice: {Counter(all_choices)}")
    print(f"  Outcome: {Counter(all_outcomes)}")
    print(f"  Early lick: {Counter(all_early)}")

    # Brain region counts
    region_counts = Counter()
    for ridx in brain_region_idx_list:
        for r in ridx:
            region_counts[COARSE_REGIONS[r]] += 1
    print(f"\nBrain regions:")
    for r, c in sorted(region_counts.items(), key=lambda x: -x[1]):
        print(f"  {r}: {c}")


if __name__ == '__main__':
    main()
