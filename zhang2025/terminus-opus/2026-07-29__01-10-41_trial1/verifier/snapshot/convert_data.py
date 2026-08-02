#!/usr/bin/env python3
"""Convert IBL Brain-Wide Map data to decoder-compatible format.

Usage:
    python -u convert_data.py <outpicklefile> [--full|--sample] [--show-processing]
"""

import os
import sys
import time
import glob
import argparse
import pickle
import numpy as np
import pandas as pd
from scipy import signal, interpolate
from iblatlas.regions import BrainRegions
from iblutil.numerical import bincount2D
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Constants
# ============================================================
BINSIZE = 0.02  # 20 ms
TIME_WINDOW = (-0.5, 1.5)  # seconds relative to stimOn_times
INTERVAL_LEN = TIME_WINDOW[1] - TIME_WINDOW[0]  # 2.0 seconds
N_BINS = int(np.ceil(INTERVAL_LEN / BINSIZE))  # 100
ALIGN_TIME = 'stimOn_times'

# Trial filtering params (from reference code load_trials_and_mask)
MIN_RT = 0.08
MAX_RT = 2.0
MAX_TRIAL_LEN = 10.0

# Wheel processing params (from brainbox wheel.py)
WHEEL_FS = 1000  # Hz interpolation frequency
WHEEL_CORNER_FREQ = 20  # Hz
WHEEL_FILTER_ORDER = 8

# Number of discretization bins for continuous outputs
N_DISC_BINS = 3

# Brain region mapper
br = BrainRegions()


# ============================================================
# Data Loading Functions
# ============================================================

def find_session_dir(lab, subject, date, base_dir='data/one_cache'):
    """Find session directory in ONE cache."""
    session_dir = os.path.join(base_dir, lab, 'Subjects', subject, date, '001')
    if os.path.exists(session_dir):
        return session_dir
    return None


def load_trials(session_dir):
    """Load trials table from parquet file."""
    trial_files = glob.glob(os.path.join(session_dir, 'alf', '*', '_ibl_trials.table.pqt'))
    if not trial_files:
        return None
    trials_df = pd.read_parquet(trial_files[0])
    return trials_df


def create_trials_mask(trials_df):
    """Create mask to filter trials based on reference code criteria.
    
    Matches load_trials_and_mask() from ibl_data_utils.py:
    - min_rt=0.08, max_rt=2.0
    - max_trial_len=10.0
    - exclude no-choice trials (choice == 0)
    - exclude NaN in key fields
    """
    nan_exclude = [
        'stimOn_times', 'choice', 'feedback_times',
        'probabilityLeft', 'firstMovement_times', 'feedbackType'
    ]
    
    mask = pd.Series(True, index=trials_df.index)
    
    # Reaction time filter
    rt = trials_df['firstMovement_times'] - trials_df['stimOn_times']
    if MIN_RT is not None:
        mask &= (rt >= MIN_RT)
    if MAX_RT is not None:
        mask &= (rt <= MAX_RT)
    
    # Trial length filter
    if MAX_TRIAL_LEN is not None:
        trial_len = trials_df['feedback_times'] - trials_df['goCue_times']
        mask &= (trial_len <= MAX_TRIAL_LEN)
    
    # NaN exclusion
    for event in nan_exclude:
        if event in trials_df.columns:
            mask &= ~trials_df[event].isna()
    
    # Exclude no-choice trials
    mask &= (trials_df['choice'] != 0)
    
    return mask


def load_spikes(session_dir, probe_names):
    """Load and merge spikes from all probes.
    
    Returns merged spike times, clusters, and cluster info.
    """
    all_spike_times = []
    all_spike_clusters = []
    all_cluster_regions = []
    cluster_offset = 0
    
    for pname in probe_names:
        # Find spike sorting directory
        spike_files = glob.glob(os.path.join(
            session_dir, 'alf', pname, 'pykilosort', '*', 'spikes.times.npy'))
        if not spike_files:
            continue
        
        ks_dir = os.path.dirname(spike_files[0])
        
        # Load spikes
        spike_times = np.load(os.path.join(ks_dir, 'spikes.times.npy')).flatten()
        spike_clusters = np.load(os.path.join(ks_dir, 'spikes.clusters.npy')).flatten()
        
        # Load cluster-to-channel mapping
        clusters_channels = np.load(os.path.join(ks_dir, 'clusters.channels.npy')).flatten()
        n_clusters = len(clusters_channels)
        
        # Load channel brain location IDs
        brain_id_files = glob.glob(os.path.join(
            session_dir, 'alf', pname, 'pykilosort', '*', 'channels.brainLocationIds_ccf_2017.npy'))
        if brain_id_files:
            brain_ids = np.load(brain_id_files[0]).flatten()
            # Map clusters to brain regions
            cluster_brain_ids = brain_ids[clusters_channels]
            acronyms = br.id2acronym(cluster_brain_ids)
            beryl = br.acronym2acronym(acronyms, mapping='Beryl')
        else:
            beryl = np.array(['unknown'] * n_clusters)
        
        # Offset cluster IDs for merging
        spike_clusters = spike_clusters + cluster_offset
        cluster_offset += n_clusters
        
        all_spike_times.append(spike_times)
        all_spike_clusters.append(spike_clusters)
        all_cluster_regions.extend(beryl)
    
    if not all_spike_times:
        return None, None, None
    
    # Merge and sort by time
    merged_times = np.concatenate(all_spike_times)
    merged_clusters = np.concatenate(all_spike_clusters)
    sort_idx = np.argsort(merged_times, kind='stable')
    merged_times = merged_times[sort_idx]
    merged_clusters = merged_clusters[sort_idx]
    
    return merged_times, merged_clusters, np.array(all_cluster_regions)


def load_wheel_speed(session_dir):
    """Load wheel data and compute speed.
    
    Matches the reference code:
    1. Load raw wheel position and timestamps
    2. Interpolate to 1000 Hz
    3. Apply Butterworth lowpass filter
    4. Compute velocity as filtered diff
    5. Speed = abs(velocity)
    """
    pos_file = os.path.join(session_dir, 'alf', '_ibl_wheel.position.npy')
    ts_file = os.path.join(session_dir, 'alf', '_ibl_wheel.timestamps.npy')
    
    if not os.path.exists(pos_file) or not os.path.exists(ts_file):
        return None, None
    
    wheel_pos = np.load(pos_file).flatten()
    wheel_ts = np.load(ts_file).flatten()
    
    if len(wheel_pos) != len(wheel_ts):
        return None, None
    
    if len(wheel_pos) < 10:
        return None, None
    
    # Interpolate to uniform sampling rate (1000 Hz)
    t_interp = np.arange(wheel_ts[0], wheel_ts[-1], 1.0 / WHEEL_FS)
    if len(t_interp) == 0:
        return None, None
    if t_interp[-1] > wheel_ts[-1]:
        t_interp = t_interp[:-1]
    
    pos_interp = interpolate.interp1d(wheel_ts, wheel_pos, kind='linear')(t_interp)
    
    # Butterworth lowpass filter and compute velocity
    sos = signal.butter(N=WHEEL_FILTER_ORDER, Wn=WHEEL_CORNER_FREQ / WHEEL_FS * 2,
                        btype='lowpass', output='sos')
    vel = np.insert(np.diff(signal.sosfiltfilt(sos, pos_interp)), 0, 0) * WHEEL_FS
    speed = np.abs(vel)
    
    return t_interp, speed


def find_file(session_dir, filename):
    """Find a file in alf/ or alf/*/  (handles hash-dated subdirectories)."""
    # Try alf/*/ first (hash-dated dirs)
    matches = glob.glob(os.path.join(session_dir, 'alf', '*', filename))
    if matches:
        return matches[0]
    # Try alf/ directly
    direct = os.path.join(session_dir, 'alf', filename)
    if os.path.exists(direct):
        return direct
    return None


def load_whisker_motion_energy(session_dir):
    """Load whisker motion energy data.
    
    Try left camera first, then right camera (matching reference code).
    """
    # Try left camera
    me_file = find_file(session_dir, 'leftCamera.ROIMotionEnergy.npy')
    cam_file = find_file(session_dir, '_ibl_leftCamera.times.npy')
    
    if me_file and cam_file:
        me = np.load(me_file).flatten()
        cam_times = np.load(cam_file).flatten()
        if len(me) == len(cam_times) and len(me) > 0:
            return cam_times, me
    
    # Try right camera
    me_file = find_file(session_dir, 'rightCamera.ROIMotionEnergy.npy')
    cam_file = find_file(session_dir, '_ibl_rightCamera.times.npy')
    
    if me_file and cam_file:
        me = np.load(me_file).flatten()
        cam_times = np.load(cam_file).flatten()
        if len(me) == len(cam_times) and len(me) > 0:
            return cam_times, me
    
    return None, None


# ============================================================
# Processing Functions
# ============================================================

def bin_spikes_per_trial(spike_times, spike_clusters, n_clusters, trial_starts, trial_ends):
    """Bin spikes into time bins for each trial.
    
    Returns array of shape (n_trials, n_clusters, n_bins).
    Optimized vectorized version.
    """
    n_trials = len(trial_starts)
    binned_all = np.zeros((n_trials, n_clusters, N_BINS), dtype=np.float32)
    
    # Process all trials at once using searchsorted
    valid_mask = ~(np.isnan(trial_starts) | np.isnan(trial_ends))
    valid_indices = np.where(valid_mask)[0]
    
    if len(valid_indices) == 0:
        return binned_all
    
    for trial_idx in valid_indices:
        t_beg = trial_starts[trial_idx]
        t_end = trial_ends[trial_idx]
        
        # Use searchsorted for fast spike selection
        i_start = np.searchsorted(spike_times, t_beg, side='left')
        i_end = np.searchsorted(spike_times, t_end, side='left')
        
        if i_start >= i_end:
            continue
        
        times_curr = spike_times[i_start:i_end]
        clust_curr = spike_clusters[i_start:i_end]
        
        # Compute time bin indices
        time_bin_idx = np.floor((times_curr - t_beg) / BINSIZE).astype(np.int32)
        time_bin_idx = np.clip(time_bin_idx, 0, N_BINS - 1)
        
        # Use np.add.at for fast binning
        np.add.at(binned_all[trial_idx], (clust_curr, time_bin_idx), 1)
    
    return binned_all


def interpolate_behavior_to_bins(beh_times, beh_values, trial_starts, trial_ends):
    """Interpolate continuous behavior signal to trial time bins.
    
    Returns array of shape (n_trials, n_bins).
    Matches get_behavior_per_interval from reference code.
    """
    n_trials = len(trial_starts)
    result = np.full((n_trials, N_BINS), np.nan, dtype=np.float32)
    valid = np.ones(n_trials, dtype=bool)
    
    for trial_idx in range(n_trials):
        t_beg = trial_starts[trial_idx]
        t_end = trial_ends[trial_idx]
        
        if np.isnan(t_beg) or np.isnan(t_end):
            valid[trial_idx] = False
            continue
        
        # Find behavior data in this interval
        idx_beg = np.searchsorted(beh_times, t_beg, side='right')
        idx_end = np.searchsorted(beh_times, t_end, side='left')
        
        beh_t = beh_times[idx_beg:idx_end]
        beh_v = beh_values[idx_beg:idx_end]
        
        if len(beh_v) < 2:
            valid[trial_idx] = False
            continue
        
        # Check for NaNs
        nan_mask = ~np.isnan(beh_v)
        if nan_mask.sum() < 2:
            valid[trial_idx] = False
            continue
        
        beh_t_clean = beh_t[nan_mask]
        beh_v_clean = beh_v[nan_mask]
        
        # Interpolate to uniform bins
        x_interp = np.linspace(t_beg, t_end, N_BINS, endpoint=False) + BINSIZE / 2
        
        try:
            y_interp = interpolate.interp1d(
                beh_t_clean, beh_v_clean, kind='linear',
                fill_value='extrapolate')(x_interp)
            result[trial_idx] = y_interp
        except Exception:
            valid[trial_idx] = False
    
    return result, valid


def discretize_time_varying(values, n_bins=N_DISC_BINS):
    """Discretize continuous time-varying values into n_bins categories.
    
    Uses quantile-based binning across all timepoints in all trials.
    Returns integer categories 0, 1, ..., n_bins-1.
    """
    # Flatten all values to compute quantiles
    flat = values[~np.isnan(values)].flatten()
    if len(flat) == 0:
        return np.zeros_like(values, dtype=np.int64)
    
    # Compute quantile boundaries
    quantiles = np.linspace(0, 1, n_bins + 1)[1:-1]
    boundaries = np.quantile(flat, quantiles)
    
    # Digitize
    result = np.digitize(values, boundaries).astype(np.int64)
    
    return result


def compute_trial_number_in_block(prob_left):
    """Compute trial number within each block.
    
    Block changes when probabilityLeft changes.
    Trial number resets to 1 at each block change.
    """
    trial_nums = np.zeros(len(prob_left), dtype=np.float32)
    counter = 1
    for i in range(len(prob_left)):
        if i > 0 and prob_left[i] != prob_left[i-1]:
            counter = 1
        trial_nums[i] = counter
        counter += 1
    return trial_nums


# ============================================================
# Main Processing
# ============================================================

def process_session(session_info, session_dir, show_processing=False):
    """Process a single session and return formatted data."""
    eid = session_info['eid']
    lab = session_info['lab']
    subject = session_info['subject']
    date = session_info['date']
    probe_names = session_info['probe_names']
    
    t0 = time.time()
    
    # 1. Load trials
    trials_df = load_trials(session_dir)
    if trials_df is None:
        print(f"  Skipping {eid}: no trials data")
        return None
    
    # 2. Create trial mask
    mask = create_trials_mask(trials_df)
    
    # 3. Load spikes
    spike_times, spike_clusters, cluster_regions = load_spikes(session_dir, probe_names)
    if spike_times is None:
        print(f"  Skipping {eid}: no spike data")
        return None
    
    n_clusters = len(cluster_regions)
    
    # 4. Load wheel speed
    wheel_times, wheel_speed = load_wheel_speed(session_dir)
    has_wheel = wheel_times is not None
    
    # 5. Load whisker motion energy
    me_times, me_values = load_whisker_motion_energy(session_dir)
    has_me = me_times is not None
    
    # 6. Compute trial intervals (aligned to stimOn)
    stim_on = trials_df[ALIGN_TIME].values
    trial_starts = stim_on + TIME_WINDOW[0]
    trial_ends = stim_on + TIME_WINDOW[1]
    
    # 7. Bin spikes per trial (for ALL trials first, then filter)
    print(f"  Binning spikes ({n_clusters} clusters, {len(trials_df)} trials)...", end=' ')
    t_bin = time.time()
    binned_spikes = bin_spikes_per_trial(
        spike_times, spike_clusters, n_clusters, trial_starts, trial_ends)
    print(f"{time.time()-t_bin:.1f}s")
    
    # 8. Interpolate behaviors per trial
    if has_wheel:
        wheel_binned, wheel_valid = interpolate_behavior_to_bins(
            wheel_times, wheel_speed, trial_starts, trial_ends)
    else:
        wheel_valid = np.zeros(len(trials_df), dtype=bool)
    
    if has_me:
        me_binned, me_valid = interpolate_behavior_to_bins(
            me_times, me_values, trial_starts, trial_ends)
    else:
        me_valid = np.zeros(len(trials_df), dtype=bool)
    
    # 9. Combine masks: trial quality + behavior availability
    combined_mask = mask.values & wheel_valid & me_valid
    
    # Check we have enough trials
    n_valid = combined_mask.sum()
    if n_valid < 2:
        print(f"  Skipping {eid}: only {n_valid} valid trials")
        return None
    
    print(f"  Valid trials: {n_valid}/{len(trials_df)} "
          f"(mask: {mask.sum()}, wheel: {wheel_valid.sum()}, me: {me_valid.sum()})")
    
    # 10. Apply mask
    valid_idx = np.where(combined_mask)[0]
    neural_data = binned_spikes[valid_idx]  # (n_valid, n_clusters, N_BINS)
    
    # Get trial-level variables
    valid_trials = trials_df.iloc[valid_idx]
    
    # Choice: -1 -> 0 (left), 1 -> 1 (right)
    choice = valid_trials['choice'].values.copy()
    choice_mapped = np.where(choice == -1, 0, 1).astype(np.int64)
    
    # Prior: probabilityLeft 0.2->0, 0.5->1, 0.8->2
    prob_left = valid_trials['probabilityLeft'].values
    prior_map = {0.2: 0, 0.5: 1, 0.8: 2}
    prior_mapped = np.array([prior_map.get(p, 1) for p in prob_left], dtype=np.int64)
    
    # Trial number in block
    # Need to compute on ALL trials first, then select valid ones
    all_prob_left = trials_df['probabilityLeft'].values
    all_trial_nums = compute_trial_number_in_block(all_prob_left)
    trial_num_in_block = all_trial_nums[valid_idx]
    
    # Time since stimulus onset (same for all trials)
    time_since_stim = np.arange(N_BINS) * BINSIZE + TIME_WINDOW[0] + BINSIZE / 2
    time_since_stim = time_since_stim.astype(np.float32)
    
    # Wheel speed and whisker ME (already interpolated)
    wheel_data = wheel_binned[valid_idx]  # (n_valid, N_BINS)
    me_data = me_binned[valid_idx]  # (n_valid, N_BINS)
    
    # Discretize continuous outputs
    wheel_disc = discretize_time_varying(wheel_data, N_DISC_BINS)
    me_disc = discretize_time_varying(me_data, N_DISC_BINS)
    
    # Format output
    n_valid_trials = len(valid_idx)
    
    # Neural: list of (n_neurons, n_timepoints) arrays
    neural_list = [neural_data[i].astype(np.float32) for i in range(n_valid_trials)]
    
    # Input: list of (n_input, n_timepoints) or (n_input,) arrays
    # input[0] = time since stim onset (time-varying, same for all trials)
    # input[1] = trial number in block (per-trial)
    input_list = []
    for i in range(n_valid_trials):
        inp = np.vstack([
            time_since_stim[np.newaxis, :],  # (1, N_BINS)
            np.full((1, N_BINS), trial_num_in_block[i], dtype=np.float32)  # (1, N_BINS)
        ])  # (2, N_BINS)
        input_list.append(inp)
    
    # Output: list of (n_output, n_timepoints) or (n_output,) arrays
    # output[0] = choice (per-trial)
    # output[1] = prior (per-trial)
    # output[2] = wheel speed discretized (time-varying)
    # output[3] = whisker ME discretized (time-varying)
    output_list = []
    for i in range(n_valid_trials):
        out = np.vstack([
            np.full((1, N_BINS), choice_mapped[i], dtype=np.int64),  # (1, N_BINS)
            np.full((1, N_BINS), prior_mapped[i], dtype=np.int64),  # (1, N_BINS)
            wheel_disc[i:i+1, :],  # (1, N_BINS)
            me_disc[i:i+1, :],  # (1, N_BINS)
        ])  # (4, N_BINS)
        output_list.append(out)
    
    elapsed = time.time() - t0
    print(f"  Session processed in {elapsed:.1f}s")
    
    # Visualization
    if show_processing:
        plot_processing(session_dir, eid, trials_df, valid_idx, neural_data,
                       wheel_data, me_data, wheel_disc, me_disc,
                       choice_mapped, prior_mapped, time_since_stim)
    
    return {
        'neural': neural_list,
        'input': input_list,
        'output': output_list,
        'subject': subject,
        'cluster_regions': cluster_regions,
        'eid': eid,
        'n_trials': n_valid_trials,
        'n_neurons': n_clusters,
    }


def plot_processing(session_dir, eid, trials_df, valid_idx, neural_data,
                   wheel_data, me_data, wheel_disc, me_disc,
                   choice_mapped, prior_mapped, time_since_stim):
    """Plot processing steps for visual verification."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(4, 3, figsize=(18, 16))
    fig.suptitle(f'Processing: {eid}', fontsize=14)
    
    # Select a few example trials
    n_examples = min(3, len(valid_idx))
    example_trials = np.random.choice(len(valid_idx), n_examples, replace=False)
    
    for col, trial_idx in enumerate(example_trials):
        # Row 0: Neural activity (raster of top neurons)
        ax = axes[0, col]
        n_show = min(50, neural_data.shape[1])
        ax.imshow(neural_data[trial_idx, :n_show, :], aspect='auto',
                  extent=[time_since_stim[0], time_since_stim[-1], n_show, 0])
        ax.set_title(f'Trial {valid_idx[trial_idx]}')
        ax.set_ylabel('Neuron' if col == 0 else '')
        ax.axvline(0, color='r', linestyle='--', alpha=0.5)
        
        # Row 1: Wheel speed (continuous and discretized)
        ax = axes[1, col]
        ax.plot(time_since_stim, wheel_data[trial_idx], 'b-', alpha=0.7, label='continuous')
        ax2 = ax.twinx()
        ax2.plot(time_since_stim, wheel_disc[trial_idx], 'r-', alpha=0.5, label='discretized')
        ax.set_ylabel('Wheel speed' if col == 0 else '')
        ax.axvline(0, color='r', linestyle='--', alpha=0.5)
        
        # Row 2: Whisker ME (continuous and discretized)
        ax = axes[2, col]
        ax.plot(time_since_stim, me_data[trial_idx], 'g-', alpha=0.7, label='continuous')
        ax2 = ax.twinx()
        ax2.plot(time_since_stim, me_disc[trial_idx], 'r-', alpha=0.5, label='discretized')
        ax.set_ylabel('Whisker ME' if col == 0 else '')
        ax.axvline(0, color='r', linestyle='--', alpha=0.5)
        
        # Row 3: Choice and Prior
        ax = axes[3, col]
        ax.text(0.5, 0.7, f'Choice: {"left" if choice_mapped[trial_idx]==0 else "right"}',
                transform=ax.transAxes, ha='center', fontsize=12)
        ax.text(0.5, 0.3, f'Prior: {["0.2","0.5","0.8"][prior_mapped[trial_idx]]}',
                transform=ax.transAxes, ha='center', fontsize=12)
        ax.set_xlabel('Time since stim onset (s)')
    
    plt.tight_layout()
    safe_eid = eid.replace('/', '_')
    plt.savefig(f'processing_{safe_eid}.png', dpi=150)
    plt.close()
    print(f"  Saved processing plot: processing_{safe_eid}.png")


def main():
    parser = argparse.ArgumentParser(description='Convert IBL BWM data to decoder format.')
    parser.add_argument('output', type=str, help='Output pickle file path')
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
    print("IBL BWM Data Conversion")
    print("="*60)
    
    t_total = time.time()
    
    # Load session info from bwm_release.csv
    bwm_df = pd.read_csv('code/code_zhang2025/data/bwm_release.csv', index_col=0)
    
    # Group by eid to get session-level info with probe names
    session_groups = bwm_df.groupby('eid').agg({
        'lab': 'first',
        'subject': 'first',
        'date': 'first',
        'probe_name': list,
        'pid': list
    }).reset_index()
    session_groups.rename(columns={'probe_name': 'probe_names'}, inplace=True)
    
    print(f"Total sessions in bwm_release: {len(session_groups)}")
    
    if args.sample:
        # Select 2 sessions with complete data
        valid_sessions = []
        for _, row in session_groups.iterrows():
            session_dir = find_session_dir(row['lab'], row['subject'], row['date'])
            if session_dir is None:
                continue
            has_spikes = len(glob.glob(os.path.join(
                session_dir, 'alf', 'probe*', 'pykilosort', '*', 'spikes.times.npy'))) > 0
            has_trials = len(glob.glob(os.path.join(
                session_dir, 'alf', '*', '_ibl_trials.table.pqt'))) > 0
            has_wheel = os.path.exists(os.path.join(
                session_dir, 'alf', '_ibl_wheel.position.npy'))
            has_me = (find_file(session_dir, 'leftCamera.ROIMotionEnergy.npy') is not None and
                (find_file(session_dir, '_ibl_leftCamera.times.npy') is not None or
                 find_file(session_dir, '_ibl_rightCamera.times.npy') is not None)) or \
                (find_file(session_dir, 'rightCamera.ROIMotionEnergy.npy') is not None and
                (find_file(session_dir, '_ibl_leftCamera.times.npy') is not None or
                 find_file(session_dir, '_ibl_rightCamera.times.npy') is not None))
            if has_spikes and has_trials and has_wheel and has_me:
                valid_sessions.append(row)
            if len(valid_sessions) >= 2:
                break
        session_groups = pd.DataFrame(valid_sessions)
        print(f"Sample mode: processing {len(session_groups)} sessions")
    
    # Process sessions
    all_neural = []
    all_input = []
    all_output = []
    all_subjects = []
    subject_idx_list = []
    all_brain_regions = []
    brain_region_idx_list = []
    session_eids = []
    
    subject_map = {}  # subject name -> index
    region_map = {}   # region name -> index
    
    n_processed = 0
    n_skipped = 0
    
    for sess_idx, (_, row) in enumerate(session_groups.iterrows()):
        eid = row['eid']
        lab = row['lab']
        subject = row['subject']
        date = row['date']
        probe_names = row['probe_names']
        
        print(f"\n[{sess_idx+1}/{len(session_groups)}] Processing {subject}/{date} ({eid})...")
        
        session_dir = find_session_dir(lab, subject, date)
        if session_dir is None:
            print(f"  Skipping: session directory not found")
            n_skipped += 1
            continue
        
        session_info = {
            'eid': eid, 'lab': lab, 'subject': subject,
            'date': date, 'probe_names': probe_names
        }
        
        try:
            result = process_session(session_info, session_dir,
                                   show_processing=args.show_processing)
        except Exception as e:
            print(f"  Error processing session: {e}")
            import traceback
            traceback.print_exc()
            n_skipped += 1
            continue
        
        if result is None:
            n_skipped += 1
            continue
        
        # Add to lists
        all_neural.append(result['neural'])
        all_input.append(result['input'])
        all_output.append(result['output'])
        session_eids.append(result['eid'])
        
        # Subject mapping
        subj = result['subject']
        if subj not in subject_map:
            subject_map[subj] = len(all_subjects)
            all_subjects.append(subj)
        subject_idx_list.append(subject_map[subj])
        
        # Brain region mapping
        regions = result['cluster_regions']
        neuron_region_idx = np.zeros(len(regions), dtype=np.int64)
        for i, r in enumerate(regions):
            if r not in region_map:
                region_map[r] = len(all_brain_regions)
                all_brain_regions.append(r)
            neuron_region_idx[i] = region_map[r]
        brain_region_idx_list.append(neuron_region_idx)
        
        n_processed += 1
    
    print(f"\n{'='*60}")
    print(f"Processed: {n_processed} sessions, Skipped: {n_skipped}")
    print(f"Subjects: {len(all_subjects)}, Brain regions: {len(all_brain_regions)}")
    
    if n_processed == 0:
        print("ERROR: No sessions processed!")
        sys.exit(1)
    
    # Build final data dictionary
    data = {
        'neural': all_neural,
        'input': all_input,
        'output': all_output,
        'subjects': all_subjects,
        'subject_idx': np.array(subject_idx_list),
        'brain_regions': all_brain_regions,
        'brain_region_idx': brain_region_idx_list,
        'input_names': ['time_since_stim_onset', 'trial_number_in_block'],
        'output_names': ['choice', 'prior', 'wheel_speed', 'whisker_motion_energy'],
        'output_values': [
            ['left', 'right'],  # choice
            ['0.2', '0.5', '0.8'],  # prior (probabilityLeft)
            ['low', 'medium', 'high'],  # wheel speed bins
            ['low', 'medium', 'high'],  # whisker ME bins
        ],
        'metadata': {
            'task_description': 'IBL decision-making task: mice rotate wheel to indicate location of visual stimulus',
            'time_bin_size': BINSIZE * 1000,  # in ms
            'temporal_alignment_event': 'stimulus onset (stimOn_times)',
            'off_start': TIME_WINDOW[0],  # -0.5s
            'off_end': TIME_WINDOW[1],  # 1.5s
            'binsize_s': BINSIZE,
            'n_bins': N_BINS,
            'session_eids': session_eids,
            'n_sessions': n_processed,
            'n_subjects': len(all_subjects),
            'trial_filtering': {
                'min_rt': MIN_RT,
                'max_rt': MAX_RT,
                'max_trial_len': MAX_TRIAL_LEN,
                'exclude_nochoice': True,
                'nan_exclude': ['stimOn_times', 'choice', 'feedback_times',
                               'probabilityLeft', 'firstMovement_times', 'feedbackType']
            }
        }
    }
    
    # Save
    print(f"\nSaving to {args.output}...")
    with open(args.output, 'wb') as f:
        pickle.dump(data, f)
    
    file_size = os.path.getsize(args.output) / (1024**2)
    print(f"Saved: {file_size:.1f} MB")
    
    total_time = time.time() - t_total
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    
    # Print summary statistics
    total_trials = sum(len(s) for s in all_neural)
    total_neurons = sum(len(s[0]) for s in all_neural)
    print(f"\nSummary:")
    print(f"  Sessions: {n_processed}")
    print(f"  Subjects: {len(all_subjects)}")
    print(f"  Total trials: {total_trials}")
    print(f"  Brain regions: {len(all_brain_regions)}")
    print(f"  Neurons/session: {[len(s[0]) for s in all_neural[:5]]}...")
    print(f"  Trials/session: {[len(s) for s in all_neural[:5]]}...")


if __name__ == '__main__':
    main()
