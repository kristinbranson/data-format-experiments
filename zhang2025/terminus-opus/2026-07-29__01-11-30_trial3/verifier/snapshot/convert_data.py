#!/usr/bin/env python3
"""Convert IBL Brain-Wide Map data to decoder-compatible format.

Usage:
    python -u convert_data.py <outpicklefile> [--sample] [--full] [--show-processing]
"""
import os
import sys
import time
import argparse
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import interp1d
from iblatlas.regions import BrainRegions
import sys
sys.path.insert(0, 'code/ibllib')
from brainbox.behavior.wheel import velocity_filtered
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Configuration
# ============================================================
BINSIZE = 0.02  # 20ms bins
TIME_WINDOW = (-0.5, 1.5)  # relative to stimOn_times
ALIGN_TIME = 'stimOn_times'
INTERVAL_LEN = TIME_WINDOW[1] - TIME_WINDOW[0]  # 2.0 seconds
N_BINS = int(np.ceil(INTERVAL_LEN / BINSIZE))  # 100 time bins

# Trial filtering parameters (matching reference code defaults)
MIN_RT = 0.08
MAX_RT = 2.0
MAX_TRIAL_LEN = 10.0
EXCLUDE_UNBIASED = False  # Keep unbiased trials for prior decoding
EXCLUDE_NOCHOICE = True   # Exclude no-choice trials
NAN_EXCLUDE = ['stimOn_times', 'choice', 'feedback_times', 
               'probabilityLeft', 'firstMovement_times', 'feedbackType']

DATA_DIR = 'data/one_cache'
BWM_CSV = 'code/code_zhang2025/data/bwm_release.csv'


def find_file(base_dir, pattern):
    """Find a file matching pattern in base_dir, handling versioned directories."""
    import glob
    # Try direct match first
    matches = glob.glob(os.path.join(base_dir, pattern))
    if matches:
        return sorted(matches)[-1]  # Return latest version
    # Try with versioned subdirectories
    matches = glob.glob(os.path.join(base_dir, '#*#', pattern))
    if matches:
        return sorted(matches)[-1]
    return None


def build_session_map(data_dir):
    """Build mapping from (subject, date) to session directory path."""
    session_map = {}
    for lab_dir in os.listdir(data_dir):
        lab_path = os.path.join(data_dir, lab_dir, 'Subjects')
        if not os.path.isdir(lab_path):
            continue
        for subject in os.listdir(lab_path):
            subj_path = os.path.join(lab_path, subject)
            if not os.path.isdir(subj_path):
                continue
            for date in os.listdir(subj_path):
                date_path = os.path.join(subj_path, date)
                if not os.path.isdir(date_path):
                    continue
                for sess_num in os.listdir(date_path):
                    sess_path = os.path.join(date_path, sess_num)
                    if os.path.isdir(sess_path):
                        key = (subject, date)
                        session_map[key] = sess_path
    return session_map


def load_trials(alf_dir):
    """Load trials table from parquet file."""
    trials_file = find_file(alf_dir, '_ibl_trials.table.pqt')
    if trials_file is None:
        return None
    trials_df = pd.read_parquet(trials_file)
    return trials_df


def create_trials_mask(trials_df):
    """Create mask for valid trials, matching reference code load_trials_and_mask."""
    query_parts = []
    
    # Reaction time filtering
    if MIN_RT is not None:
        query_parts.append(f'(firstMovement_times - stimOn_times < {MIN_RT})')
    if MAX_RT is not None:
        query_parts.append(f'(firstMovement_times - stimOn_times > {MAX_RT})')
    
    # Trial duration filtering  
    if MAX_TRIAL_LEN is not None:
        query_parts.append(f'(feedback_times - goCue_times > {MAX_TRIAL_LEN})')
    
    # NaN exclusion
    for event in NAN_EXCLUDE:
        query_parts.append(f'{event}.isnull()')
    
    # Exclude unbiased block
    if EXCLUDE_UNBIASED:
        query_parts.append('(probabilityLeft == 0.5)')
    
    # Exclude no-choice trials
    if EXCLUDE_NOCHOICE:
        query_parts.append('(choice == 0)')
    
    query = ' | '.join(query_parts)
    mask = ~trials_df.eval(query)
    return mask


def load_spikes(alf_dir, probe_name):
    """Load spike times and clusters for a probe."""
    probe_dir = os.path.join(alf_dir, probe_name)
    if not os.path.isdir(probe_dir):
        return None, None, None, None
    
    # Find pykilosort directory
    pyks_dir = os.path.join(probe_dir, 'pykilosort')
    if not os.path.isdir(pyks_dir):
        return None, None, None, None
    
    # Find versioned subdirectory
    versions = [d for d in os.listdir(pyks_dir) if os.path.isdir(os.path.join(pyks_dir, d))]
    if not versions:
        return None, None, None, None
    version_dir = os.path.join(pyks_dir, sorted(versions)[-1])
    
    try:
        spike_times = np.load(os.path.join(version_dir, 'spikes.times.npy')).flatten()
        spike_clusters = np.load(os.path.join(version_dir, 'spikes.clusters.npy')).flatten()
        cluster_channels = np.load(os.path.join(version_dir, 'clusters.channels.npy')).flatten()
        channel_brain_ids = np.load(os.path.join(version_dir, 'channels.brainLocationIds_ccf_2017.npy')).flatten()
        
        return spike_times, spike_clusters, cluster_channels, channel_brain_ids
    except Exception as e:
        print(f'  Error loading spikes from {version_dir}: {e}')
        return None, None, None, None


def merge_probes_data(probes_data):
    """Merge spike data from multiple probes."""
    all_spike_times = []
    all_spike_clusters = []
    all_cluster_regions = []
    cluster_offset = 0
    
    for spike_times, spike_clusters, cluster_channels, channel_brain_ids in probes_data:
        n_clusters = len(cluster_channels)
        all_spike_times.append(spike_times)
        all_spike_clusters.append(spike_clusters + cluster_offset)
        
        # Map clusters to brain regions
        cluster_brain_ids = channel_brain_ids[cluster_channels]
        all_cluster_regions.append(cluster_brain_ids)
        
        cluster_offset += n_clusters
    
    merged_times = np.concatenate(all_spike_times)
    merged_clusters = np.concatenate(all_spike_clusters)
    merged_regions = np.concatenate(all_cluster_regions)  # brain IDs
    
    # Sort by time
    sort_idx = np.argsort(merged_times)
    merged_times = merged_times[sort_idx]
    merged_clusters = merged_clusters[sort_idx]
    
    return merged_times, merged_clusters, merged_regions


def bin_spikes_vectorized(spike_times, spike_clusters, n_clusters, 
                          interval_begs, interval_ends, binsize, n_bins):
    """Bin spikes into time bins for all trials efficiently.
    
    Returns: array of shape (n_trials, n_clusters, n_bins)
    """
    n_trials = len(interval_begs)
    binned = np.zeros((n_trials, n_clusters, n_bins), dtype=np.float32)
    
    # Get unique cluster IDs and create mapping
    unique_clusters = np.arange(n_clusters)
    
    for trial_idx in range(n_trials):
        t_beg = interval_begs[trial_idx]
        t_end = interval_ends[trial_idx]
        
        if np.isnan(t_beg) or np.isnan(t_end):
            continue
        
        # Select spikes in this interval
        mask = (spike_times >= t_beg) & (spike_times < t_end)
        trial_times = spike_times[mask]
        trial_clusters = spike_clusters[mask]
        
        if len(trial_times) == 0:
            continue
        
        # Compute bin indices
        bin_idx = np.minimum(
            ((trial_times - t_beg) / binsize).astype(int),
            n_bins - 1
        )
        
        # Count spikes per cluster per bin
        for spike_i in range(len(trial_times)):
            c = trial_clusters[spike_i]
            b = bin_idx[spike_i]
            if c < n_clusters:
                binned[trial_idx, c, b] += 1
    
    return binned


def bin_spikes_fast(spike_times, spike_clusters, n_clusters,
                    interval_begs, interval_ends, binsize, n_bins):
    """Fast vectorized spike binning using searchsorted and histogram."""
    n_trials = len(interval_begs)
    binned = np.zeros((n_trials, n_clusters, n_bins), dtype=np.float32)
    
    for trial_idx in range(n_trials):
        t_beg = interval_begs[trial_idx]
        t_end = interval_ends[trial_idx]
        
        if np.isnan(t_beg) or np.isnan(t_end):
            continue
        
        # Binary search for spikes in interval
        i_start = np.searchsorted(spike_times, t_beg, side='left')
        i_end = np.searchsorted(spike_times, t_end, side='left')
        
        if i_start >= i_end:
            continue
        
        trial_times = spike_times[i_start:i_end]
        trial_clusters = spike_clusters[i_start:i_end]
        
        # Compute bin indices
        bin_idx = np.minimum(
            ((trial_times - t_beg) / binsize).astype(np.int32),
            n_bins - 1
        )
        
        # Use linear indexing for fast accumulation
        linear_idx = trial_clusters * n_bins + bin_idx
        np.add.at(binned[trial_idx].ravel(), linear_idx, 1)
    
    return binned


def load_wheel_data(alf_dir):
    """Load wheel position and timestamps, compute velocity and speed.
    Uses Butterworth-filtered velocity matching the reference code (SessionLoader.load_wheel)."""
    pos_file = find_file(alf_dir, '_ibl_wheel.position.npy')
    ts_file = find_file(alf_dir, '_ibl_wheel.timestamps.npy')
    
    if pos_file is None or ts_file is None:
        return None, None
    
    position = np.load(pos_file).flatten()
    timestamps = np.load(ts_file).flatten()
    
    # Compute sampling frequency
    dt_median = np.median(np.diff(timestamps))
    if dt_median <= 0:
        dt_median = 0.001  # fallback
    fs = 1.0 / dt_median
    
    # Use Butterworth-filtered velocity (matching reference code SessionLoader.load_wheel)
    try:
        velocity, _ = velocity_filtered(position, fs)
    except Exception:
        # Fallback to simple diff if filter fails
        dt = np.diff(timestamps)
        dt[dt == 0] = dt_median
        velocity = np.zeros_like(position)
        velocity[1:] = np.diff(position) / dt
        velocity[0] = velocity[1]
    
    # Speed is absolute velocity
    speed = np.abs(velocity)
    
    return timestamps, speed


def load_whisker_me(alf_dir):
    """Load whisker motion energy data."""
    # Try left camera first (matching reference code)
    me_file = find_file(alf_dir, 'leftCamera.ROIMotionEnergy.npy')
    times_file = find_file(alf_dir, '_ibl_leftCamera.times.npy')
    
    if me_file is None or times_file is None:
        # Fall back to right camera
        me_file = find_file(alf_dir, 'rightCamera.ROIMotionEnergy.npy')
        times_file = find_file(alf_dir, '_ibl_rightCamera.times.npy')
    
    if me_file is None or times_file is None:
        return None, None
    
    me_values = np.load(me_file).flatten()
    me_times = np.load(times_file).flatten()
    
    # Handle length mismatches
    min_len = min(len(me_values), len(me_times))
    me_values = me_values[:min_len]
    me_times = me_times[:min_len]
    
    # Remove NaN values
    valid = ~(np.isnan(me_values) | np.isnan(me_times))
    me_values = me_values[valid]
    me_times = me_times[valid]
    
    return me_times, me_values


def interpolate_behavior_to_bins(beh_times, beh_values, interval_begs, interval_ends, 
                                  binsize, n_bins):
    """Interpolate behavioral data to trial time bins.
    
    Matches reference code: x_interp = np.linspace(interval_beg + binsize, interval_end, n_bins)
    """
    n_trials = len(interval_begs)
    result = np.full((n_trials, n_bins), np.nan, dtype=np.float32)
    valid_mask = np.ones(n_trials, dtype=bool)
    
    for trial_idx in range(n_trials):
        t_beg = interval_begs[trial_idx]
        t_end = interval_ends[trial_idx]
        
        if np.isnan(t_beg) or np.isnan(t_end):
            valid_mask[trial_idx] = False
            continue
        
        # Match reference code interpolation points
        x_interp = np.linspace(t_beg + binsize, t_end, n_bins)
        
        # Find behavioral data in this interval (with some padding)
        pad = 1.0  # 1 second padding for interpolation
        i_start = np.searchsorted(beh_times, t_beg - pad, side='left')
        i_end = np.searchsorted(beh_times, t_end + pad, side='right')
        
        if i_end - i_start < 2:
            valid_mask[trial_idx] = False
            continue
        
        local_times = beh_times[i_start:i_end]
        local_vals = beh_values[i_start:i_end]
        
        try:
            interp_func = interp1d(local_times, local_vals, kind='linear',
                                   fill_value='extrapolate')
            result[trial_idx] = interp_func(x_interp).astype(np.float32)
        except Exception:
            valid_mask[trial_idx] = False
    
    return result, valid_mask


def compute_trial_number_in_block(prob_left):
    """Compute trial number within each block.
    
    A block change occurs when probabilityLeft changes value.
    Trial number starts at 1 for the first trial in each block.
    """
    trial_nums = np.zeros(len(prob_left), dtype=np.float32)
    current_block_start = 0
    current_prob = prob_left[0]
    
    for i in range(len(prob_left)):
        if prob_left[i] != current_prob:
            current_block_start = i
            current_prob = prob_left[i]
        trial_nums[i] = i - current_block_start + 1
    
    return trial_nums


def discretize_continuous(values, n_bins=3):
    """Discretize continuous values into n_bins equal-frequency bins.
    
    Computes bin edges from the data, then assigns each value to a bin.
    Returns discretized values (0, 1, ..., n_bins-1) and bin edges.
    """
    # Flatten all values to compute global bin edges
    all_vals = values[~np.isnan(values)].flatten()
    
    # Compute quantile-based bin edges
    percentiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(all_vals, percentiles)
    edges[0] = -np.inf
    edges[-1] = np.inf
    
    # Discretize
    discretized = np.digitize(values, edges[1:-1]).astype(np.float32)
    
    return discretized, edges


def process_session(eid, session_path, probe_names, br, show_processing=False, session_idx=0):
    """Process a single session and return formatted data.
    
    Returns None if session cannot be processed.
    """
    alf_dir = os.path.join(session_path, 'alf')
    if not os.path.isdir(alf_dir):
        print(f'  No alf directory found')
        return None
    
    # 1. Load trials
    trials_df = load_trials(alf_dir)
    if trials_df is None:
        print(f'  No trials data found')
        return None
    
    # 2. Create trial mask
    mask = create_trials_mask(trials_df)
    n_valid = mask.sum()
    print(f'  Trials: {len(trials_df)} total, {n_valid} valid')
    
    if n_valid < 2:
        print(f'  Too few valid trials')
        return None
    
    # 3. Load spike data from all probes
    probes_data = []
    for probe_name in probe_names:
        result = load_spikes(alf_dir, probe_name)
        if result[0] is not None:
            probes_data.append(result)
    
    if len(probes_data) == 0:
        print(f'  No spike data found')
        return None
    
    # 4. Merge probes
    spike_times, spike_clusters, cluster_brain_ids = merge_probes_data(probes_data)
    n_clusters = len(cluster_brain_ids)
    print(f'  Merged {len(probes_data)} probes: {n_clusters} clusters, {len(spike_times)} spikes')
    
    # 5. Map brain regions using Beryl mapping
    acronyms = br.id2acronym(cluster_brain_ids)
    beryl_regions = br.acronym2acronym(acronyms, mapping='Beryl')
    
    # 6. Get valid trials
    valid_trials_df = trials_df[mask].reset_index(drop=True)
    
    # 7. Compute trial intervals
    stim_on = valid_trials_df[ALIGN_TIME].values
    interval_begs = stim_on + TIME_WINDOW[0]
    interval_ends = stim_on + TIME_WINDOW[1]
    
    # 8. Bin spikes
    t0 = time.time()
    binned_spikes = bin_spikes_fast(
        spike_times, spike_clusters, n_clusters,
        interval_begs, interval_ends, BINSIZE, N_BINS
    )
    print(f'  Spike binning: {time.time()-t0:.1f}s, shape={binned_spikes.shape}')
    
    # 9. Load and process wheel speed
    wheel_times, wheel_speed = load_wheel_data(alf_dir)
    if wheel_times is None:
        print(f'  No wheel data found')
        return None
    
    wheel_binned, wheel_valid = interpolate_behavior_to_bins(
        wheel_times, wheel_speed, interval_begs, interval_ends, BINSIZE, N_BINS
    )
    
    # 10. Load and process whisker motion energy
    me_times, me_values = load_whisker_me(alf_dir)
    if me_times is None:
        print(f'  No whisker motion energy data found')
        return None
    
    me_binned, me_valid = interpolate_behavior_to_bins(
        me_times, me_values, interval_begs, interval_ends, BINSIZE, N_BINS
    )
    
    # 11. Combined validity mask
    combined_valid = wheel_valid & me_valid
    n_final = combined_valid.sum()
    print(f'  After behavior filtering: {n_final} trials')
    
    if n_final < 2:
        print(f'  Too few valid trials after behavior filtering')
        return None
    
    # Apply combined mask
    final_spikes = binned_spikes[combined_valid]  # (n_trials, n_clusters, n_bins)
    final_wheel = wheel_binned[combined_valid]    # (n_trials, n_bins)
    final_me = me_binned[combined_valid]           # (n_trials, n_bins)
    final_trials = valid_trials_df[combined_valid].reset_index(drop=True)
    
    # 12. Extract per-trial variables
    # Choice: -1 (left) -> 0, 1 (right) -> 1
    choice = final_trials['choice'].values.copy()
    choice[choice == -1] = 0
    choice = choice.astype(np.float32)
    
    # Prior (probabilityLeft): 0.2 -> 0, 0.5 -> 1, 0.8 -> 2
    prob_left = final_trials['probabilityLeft'].values.copy()
    prior = np.zeros(len(prob_left), dtype=np.float32)
    prior[np.isclose(prob_left, 0.2)] = 0
    prior[np.isclose(prob_left, 0.5)] = 1
    prior[np.isclose(prob_left, 0.8)] = 2
    
    # Trial number in block - compute from ALL trials, then select valid ones
    all_prob_left = trials_df['probabilityLeft'].values
    all_trial_nums = compute_trial_number_in_block(all_prob_left)
    # Select only valid trials
    valid_indices = np.where(mask.values)[0]
    trial_nums_valid = all_trial_nums[valid_indices]
    trial_nums_final = trial_nums_valid[combined_valid]
    
    # 13. Construct time since stimulus onset (input)
    # Time points: center of each bin
    time_since_stim = np.linspace(
        TIME_WINDOW[0] + BINSIZE/2, 
        TIME_WINDOW[1] - BINSIZE/2, 
        N_BINS
    ).astype(np.float32)
    
    # 14. Build per-trial lists
    neural_list = []
    input_list = []
    output_list_wheel = []
    output_list_me = []
    
    n_final_trials = len(final_trials)
    for t in range(n_final_trials):
        # Neural: (n_clusters, n_bins)
        neural_list.append(final_spikes[t].astype(np.float32))
        
        # Input: (2, n_bins) for time-varying, or (2,) for per-trial
        # input[0] = time since stim onset (time-varying)
        # input[1] = trial number in block (per-trial, broadcast to time)
        inp = np.stack([
            time_since_stim,
            np.full(N_BINS, trial_nums_final[t], dtype=np.float32)
        ], axis=0)  # (2, n_bins)
        input_list.append(inp)
    
    # Return raw continuous values for wheel/ME - will be discretized globally
    return {
        'neural': neural_list,
        'input': input_list,
        'choice': choice,
        'prior': prior,
        'wheel_speed_raw': final_wheel,  # (n_trials, n_bins)
        'me_raw': final_me,              # (n_trials, n_bins)
        'beryl_regions': beryl_regions,   # (n_clusters,)
        'n_clusters': n_clusters,
        'n_trials': n_final_trials,
        'eid': eid,
    }


def main():
    parser = argparse.ArgumentParser(description='Convert IBL BWM data')
    parser.add_argument('output', type=str, help='Output pickle file path')
    parser.add_argument('--sample', action='store_true', help='Process only 2 sessions')
    parser.add_argument('--full', action='store_true', help='Process all sessions (default)')
    parser.add_argument('--show-processing', action='store_true', help='Show processing plots')
    args = parser.parse_args()
    
    if not args.sample:
        args.full = True
    
    total_start = time.time()
    
    # Initialize brain regions mapper
    br = BrainRegions()
    
    # Load session info
    bwm_df = pd.read_csv(BWM_CSV, index_col=0)
    session_map = build_session_map(DATA_DIR)
    
    # Group by eid to get unique sessions with their probes
    sessions_info = []
    for eid, group in bwm_df.groupby('eid'):
        subject = group['subject'].iloc[0]
        date = group['date'].iloc[0]
        lab = group['lab'].iloc[0]
        probe_names = list(group['probe_name'].unique())
        key = (subject, date)
        if key in session_map:
            sessions_info.append({
                'eid': eid,
                'subject': subject,
                'date': date,
                'lab': lab,
                'probe_names': probe_names,
                'path': session_map[key]
            })
    
    print(f'Found {len(sessions_info)} sessions to process')
    
    if args.sample:
        # Select 2 sessions with different subjects for testing
        np.random.seed(42)
        # Pick sessions that are likely to have good data
        sessions_info = sessions_info[:2]
        print(f'Sample mode: processing {len(sessions_info)} sessions')
    
    # Process all sessions
    all_neural = []
    all_input = []
    all_output = []
    all_subjects = []
    all_subject_idx = []
    all_brain_region_idx = []
    all_wheel_raw = []  # for global discretization
    all_me_raw = []     # for global discretization
    session_metadata = []
    
    subject_to_idx = {}
    
    for sess_i, sess_info in enumerate(sessions_info):
        sess_start = time.time()
        eid = sess_info['eid']
        subject = sess_info['subject']
        print(f'\n[{sess_i+1}/{len(sessions_info)}] Processing {subject} / {sess_info["date"]} (eid={eid})')
        
        result = process_session(
            eid, sess_info['path'], sess_info['probe_names'], br,
            show_processing=args.show_processing, session_idx=sess_i
        )
        
        if result is None:
            print(f'  SKIPPED')
            continue
        
        # Track subject
        if subject not in subject_to_idx:
            subject_to_idx[subject] = len(all_subjects)
            all_subjects.append(subject)
        
        all_neural.append(result['neural'])
        all_input.append(result['input'])
        all_subject_idx.append(subject_to_idx[subject])
        all_brain_region_idx.append(result['beryl_regions'])
        all_wheel_raw.append(result['wheel_speed_raw'])
        all_me_raw.append(result['me_raw'])
        
        # Store per-trial outputs temporarily
        session_metadata.append({
            'choice': result['choice'],
            'prior': result['prior'],
            'n_trials': result['n_trials'],
            'eid': eid,
        })
        
        elapsed = time.time() - sess_start
        print(f'  Done in {elapsed:.1f}s')
    
    print(f'\n=== Processed {len(all_neural)} sessions ===')
    
    if len(all_neural) == 0:
        print('ERROR: No sessions processed successfully')
        sys.exit(1)
    
    # Global discretization of wheel speed and whisker ME
    print('\nComputing global discretization bins...')
    
    # Collect all wheel speed values
    all_wheel_vals = np.concatenate([w.flatten() for w in all_wheel_raw])
    all_me_vals = np.concatenate([m.flatten() for m in all_me_raw])
    
    # Compute global bin edges (quantile-based, 3 bins)
    wheel_percentiles = np.percentile(all_wheel_vals[~np.isnan(all_wheel_vals)], [33.33, 66.67])
    me_percentiles = np.percentile(all_me_vals[~np.isnan(all_me_vals)], [33.33, 66.67])
    
    print(f'  Wheel speed bin edges: {wheel_percentiles}')
    print(f'  Whisker ME bin edges: {me_percentiles}')
    
    # Build brain_regions list and remap indices
    all_unique_regions = sorted(set(r for regions in all_brain_region_idx for r in regions))
    region_to_idx = {r: i for i, r in enumerate(all_unique_regions)}
    
    brain_region_idx_mapped = []
    for regions in all_brain_region_idx:
        idx = np.array([region_to_idx[r] for r in regions], dtype=np.int64)
        brain_region_idx_mapped.append(idx)
    
    # Build output lists with discretized wheel/ME
    for sess_i in range(len(all_neural)):
        n_trials = session_metadata[sess_i]['n_trials']
        choice = session_metadata[sess_i]['choice']
        prior = session_metadata[sess_i]['prior']
        wheel_raw = all_wheel_raw[sess_i]
        me_raw = all_me_raw[sess_i]
        
        # Discretize wheel speed
        wheel_disc = np.digitize(wheel_raw, wheel_percentiles).astype(np.int64)
        me_disc = np.digitize(me_raw, me_percentiles).astype(np.int64)
        
        output_list = []
        for t in range(n_trials):
            # Output: (4, n_bins) or mixed
            # output[0] = choice (per-trial, broadcast)
            # output[1] = prior (per-trial, broadcast) 
            # output[2] = wheel speed discretized (time-varying)
            # output[3] = whisker ME discretized (time-varying)
            out = np.stack([
                np.full(N_BINS, int(choice[t]), dtype=np.int64),
                np.full(N_BINS, int(prior[t]), dtype=np.int64),
                wheel_disc[t],
                me_disc[t],
            ], axis=0).astype(np.int64)  # (4, n_bins)
            output_list.append(out)
        
        all_output.append(output_list)
    
    # Build final data structure
    data = {
        'neural': all_neural,
        'input': all_input,
        'output': all_output,
        
        'subjects': all_subjects,
        'subject_idx': np.array(all_subject_idx, dtype=np.int64),
        
        'brain_regions': all_unique_regions,
        'brain_region_idx': brain_region_idx_mapped,
        
        'input_names': ['time_since_stim_onset', 'trial_number_in_block'],
        'output_names': ['choice', 'prior', 'wheel_speed', 'whisker_motion_energy'],
        'output_values': [
            ['left', 'right'],           # choice: 0=left, 1=right
            ['0.2', '0.5', '0.8'],       # prior: 0=0.2, 1=0.5, 2=0.8
            ['low', 'medium', 'high'],   # wheel speed: 3 bins
            ['low', 'medium', 'high'],   # whisker ME: 3 bins
        ],
        
        'metadata': {
            'task_description': 'IBL decision-making task: mice rotate wheel to indicate location of visual stimulus',
            'time_bin_size': BINSIZE * 1000,  # in ms
            'temporal_alignment_event': 'stimulus onset (stimOn_times)',
            'off_start': TIME_WINDOW[0],  # -0.5s before stim onset
            'off_end': TIME_WINDOW[1],    # 1.5s after stim onset
            'bin_size_s': BINSIZE,
            'n_time_bins': N_BINS,
            'wheel_speed_bin_edges': wheel_percentiles.tolist(),
            'whisker_me_bin_edges': me_percentiles.tolist(),
            'source': 'IBL Brain-Wide Map',
            'n_sessions_processed': len(all_neural),
            'n_sessions_total': len(sessions_info),
        }
    }
    
    # Print summary statistics
    total_trials = sum(len(s) for s in all_neural)
    total_neurons = sum(s[0].shape[0] for s in all_neural if len(s) > 0)
    print(f'\n=== Summary ===')
    print(f'Sessions: {len(all_neural)}')
    print(f'Subjects: {len(all_subjects)}')
    print(f'Total trials: {total_trials}')
    print(f'Total neurons: {total_neurons}')
    print(f'Brain regions: {len(all_unique_regions)}')
    print(f'Mean neurons/session: {total_neurons/len(all_neural):.1f}')
    print(f'Mean trials/session: {total_trials/len(all_neural):.1f}')
    
    # Save
    print(f'\nSaving to {args.output}...')
    with open(args.output, 'wb') as f:
        pickle.dump(data, f)
    
    file_size = os.path.getsize(args.output) / (1024**3)
    total_time = time.time() - total_start
    print(f'Saved ({file_size:.2f} GB) in {total_time:.1f}s total')
    
    # Show processing plots if requested
    if args.show_processing and len(all_neural) > 0:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            for sess_i in range(min(2, len(all_neural))):
                fig, axes = plt.subplots(4, 2, figsize=(16, 20))
                fig.suptitle(f'Session {sess_i}: {session_metadata[sess_i]["eid"]}')
                
                # Pick a random trial
                n_t = len(all_neural[sess_i])
                trial_idx = min(5, n_t - 1)
                
                neural_trial = all_neural[sess_i][trial_idx]
                input_trial = all_input[sess_i][trial_idx]
                output_trial = all_output[sess_i][trial_idx]
                
                time_axis = np.linspace(TIME_WINDOW[0], TIME_WINDOW[1], N_BINS)
                
                # Neural activity (first 50 neurons)
                n_show = min(50, neural_trial.shape[0])
                axes[0, 0].imshow(neural_trial[:n_show], aspect='auto', 
                                  extent=[TIME_WINDOW[0], TIME_WINDOW[1], n_show, 0])
                axes[0, 0].set_title(f'Neural activity (first {n_show} neurons)')
                axes[0, 0].set_xlabel('Time (s)')
                axes[0, 0].set_ylabel('Neuron')
                
                # Mean firing rate
                axes[0, 1].plot(time_axis, neural_trial.mean(axis=0))
                axes[0, 1].set_title('Mean firing rate')
                axes[0, 1].set_xlabel('Time (s)')
                
                # Inputs
                axes[1, 0].plot(time_axis, input_trial[0], label='Time since stim')
                axes[1, 0].set_title('Input: Time since stimulus onset')
                axes[1, 0].set_xlabel('Time (s)')
                
                axes[1, 1].plot(time_axis, input_trial[1], label='Trial # in block')
                axes[1, 1].set_title(f'Input: Trial number in block = {input_trial[1, 0]:.0f}')
                axes[1, 1].set_xlabel('Time (s)')
                
                # Outputs
                axes[2, 0].plot(time_axis, output_trial[0], label='Choice')
                axes[2, 0].set_title(f'Output: Choice = {output_trial[0, 0]:.0f}')
                axes[2, 0].set_xlabel('Time (s)')
                
                axes[2, 1].plot(time_axis, output_trial[1], label='Prior')
                axes[2, 1].set_title(f'Output: Prior = {output_trial[1, 0]:.0f}')
                axes[2, 1].set_xlabel('Time (s)')
                
                axes[3, 0].plot(time_axis, output_trial[2], label='Wheel speed')
                axes[3, 0].set_title('Output: Wheel speed (discretized)')
                axes[3, 0].set_xlabel('Time (s)')
                
                axes[3, 1].plot(time_axis, output_trial[3], label='Whisker ME')
                axes[3, 1].set_title('Output: Whisker ME (discretized)')
                axes[3, 1].set_xlabel('Time (s)')
                
                fig.tight_layout()
                fig.savefig(f'processing_session_{sess_i}.png', dpi=150)
                plt.close(fig)
                print(f'Saved processing_session_{sess_i}.png')
        except Exception as e:
            print(f'Error creating plots: {e}')


if __name__ == '__main__':
    main()
