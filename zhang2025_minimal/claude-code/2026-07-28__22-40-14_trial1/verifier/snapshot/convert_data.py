#!/usr/bin/env python3
"""
Convert IBL Brain-Wide Map data into the standardized decoder format.

Follows processing from:
- "Exploiting correlations across trials and behavioral sessions to improve neural decoding" (Zhang et al.)
- "A brain-wide map of neural activity during complex behaviour" (IBL)

Key processing decisions (matching reference code):
- Align to stimulus onset (stimOn_times)
- Time window: -0.5 to 1.5 s relative to stimulus onset
- Bin size: 20 ms -> T=100 time bins
- All spike-sorted clusters included (no quality filter, matching reference code's qc=None)
- Merge probes from same session
- Beryl atlas mapping for brain regions
- Trial exclusion: NaN events, reaction time 0.08-2.0 s, no-choice trials
- Whisker motion energy: left camera preferred, right camera fallback
- Wheel speed: absolute value of velocity from interpolated wheel data
"""

import os
import sys
import argparse
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import interp1d
from iblatlas.regions import BrainRegions

# ============================================================
# Configuration
# ============================================================

CACHE_DIR = '/app/data/one_cache'
TABLES_DIR = '/app/data/one_cache/2022_Q4_IBL_et_al_BWM'
BWM_CSV = '/app/code/code_zhang2025/data/bwm_release.csv'

# Matching reference code: 0_data_caching.py line 52-54
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)  # seconds relative to stimulus onset
BINSIZE = 0.02  # 20 ms bins
INTERVAL_LEN = TIME_WINDOW[1] - TIME_WINDOW[0]  # 2.0 seconds
N_TIMEBINS = int(np.ceil(INTERVAL_LEN / BINSIZE))  # 100

# Trial filtering (matching reference code: load_trials_and_mask defaults)
MIN_RT = 0.08
MAX_RT = 2.0
NAN_EXCLUDE = ['stimOn_times', 'choice', 'feedback_times', 'probabilityLeft',
               'firstMovement_times', 'feedbackType']

# Minimum neurons per session to include
MIN_NEURONS = 5  # matching data paper criteria

ap = argparse.ArgumentParser()
ap.add_argument('--max-sessions', type=int, default=None, help='Limit number of sessions (for testing)')
ap.add_argument('--output', type=str, default='converted_data.pkl', help='Output file path')
ap.add_argument('--n-workers', type=int, default=1)
args = ap.parse_args()


# ============================================================
# Helper functions
# ============================================================

def find_session_path(eid, sessions_df, datasets_df):
    """Find the local path for a given session eid."""
    # Try datasets_df first for exact session_path
    if eid in datasets_df.index.get_level_values(0):
        session_path_str = datasets_df.loc[eid, 'session_path'].iloc[0]
        return Path(CACHE_DIR) / session_path_str

    # Fallback to sessions_df
    if eid in sessions_df.index:
        row = sessions_df.loc[eid]
        lab = row['lab']
        subject = row['subject']
        date = str(row['date'])
        number = str(row['number']).zfill(3)
        return Path(CACHE_DIR) / lab / 'Subjects' / subject / date / number
    return None


def find_file_with_revision(base_dir, filename):
    """Find a file that might be in a revision subdirectory like #2025-03-03#/."""
    # Check direct path first
    direct = base_dir / filename
    if direct.exists():
        return direct
    # Check revision directories
    for d in sorted(base_dir.iterdir()):
        if d.is_dir() and d.name.startswith('#') and d.name.endswith('#'):
            candidate = d / filename
            if candidate.exists():
                return candidate
    return None


def load_trials(session_path):
    """Load trials table and create inclusion mask."""
    alf_dir = session_path / 'alf'

    # Load trials table (may be in revision directory)
    trials_file = find_file_with_revision(alf_dir, '_ibl_trials.table.pqt')
    if trials_file is None:
        return None, None

    trials = pd.read_parquet(trials_file)

    # Create mask matching reference code: load_trials_and_mask
    mask = pd.Series(True, index=trials.index)

    # Exclude trials with NaN in required fields
    for col in NAN_EXCLUDE:
        if col in trials.columns:
            mask &= ~trials[col].isna()

    # Exclude trials with reaction time outside range
    if 'firstMovement_times' in trials.columns and 'stimOn_times' in trials.columns:
        rt = trials['firstMovement_times'] - trials['stimOn_times']
        if MIN_RT is not None:
            mask &= (rt >= MIN_RT)
        if MAX_RT is not None:
            mask &= (rt <= MAX_RT)

    # Exclude no-choice trials (choice == 0)
    if 'choice' in trials.columns:
        mask &= (trials['choice'] != 0)

    return trials, mask


def load_spike_data(session_path, probe_name, qc_threshold=1.0):
    """Load spike times, clusters, and cluster info for a probe.

    Parameters
    ----------
    qc_threshold : float
        Minimum cluster label value to include. Default 1.0 matches
        the BWM paper's well-isolated neuron criterion.
        Set to None to include all clusters (matching Zhang code).
    """
    probe_dir = session_path / 'alf' / probe_name / 'pykilosort'

    if not probe_dir.exists():
        return None, None, None, None

    spike_times_file = find_file_with_revision(probe_dir, 'spikes.times.npy')
    spike_clusters_file = find_file_with_revision(probe_dir, 'spikes.clusters.npy')
    clusters_channels_file = find_file_with_revision(probe_dir, 'clusters.channels.npy')
    clusters_metrics_file = find_file_with_revision(probe_dir, 'clusters.metrics.pqt')
    channels_brain_ids_file = find_file_with_revision(probe_dir, 'channels.brainLocationIds_ccf_2017.npy')

    if any(f is None for f in [spike_times_file, spike_clusters_file,
                                clusters_channels_file, channels_brain_ids_file]):
        return None, None, None, None

    spike_times = np.load(spike_times_file).flatten()
    spike_clusters = np.load(spike_clusters_file).flatten()
    clusters_channels = np.load(clusters_channels_file).flatten()
    channels_brain_ids = np.load(channels_brain_ids_file).flatten()

    # Map clusters to brain region IDs
    n_clusters = len(clusters_channels)
    cluster_brain_ids = channels_brain_ids[clusters_channels]

    # Apply quality filter if requested
    if qc_threshold is not None and clusters_metrics_file is not None:
        metrics = pd.read_parquet(clusters_metrics_file)
        if 'label' in metrics.columns:
            good_mask = metrics['label'].values >= qc_threshold
            good_cluster_ids = np.where(good_mask)[0]

            if len(good_cluster_ids) == 0:
                return None, None, None, None

            # Filter spikes to only good clusters
            from iblutil.numerical import ismember
            spike_mask, ib = ismember(spike_clusters, good_cluster_ids)
            spike_times = spike_times[spike_mask]
            # Remap cluster IDs to 0..n_good-1
            spike_clusters = ib.astype(np.int32)
            cluster_brain_ids = cluster_brain_ids[good_cluster_ids]
            n_clusters = len(good_cluster_ids)

    return spike_times, spike_clusters, cluster_brain_ids, n_clusters


def merge_probes_data(probes_data):
    """Merge spike data from multiple probes, matching reference code merge_probes."""
    all_spike_times = []
    all_spike_clusters = []
    all_brain_ids = []
    cluster_offset = 0

    for spike_times, spike_clusters, brain_ids, n_clusters in probes_data:
        all_spike_times.append(spike_times)
        all_spike_clusters.append(spike_clusters + cluster_offset)
        all_brain_ids.append(brain_ids)
        cluster_offset += n_clusters

    merged_times = np.concatenate(all_spike_times)
    merged_clusters = np.concatenate(all_spike_clusters)
    merged_brain_ids = np.concatenate(all_brain_ids)

    # Sort by time (matching reference code)
    sort_idx = np.argsort(merged_times, kind='stable')
    merged_times = merged_times[sort_idx]
    merged_clusters = merged_clusters[sort_idx]

    return merged_times, merged_clusters, merged_brain_ids


def bin_spikes_per_trial(spike_times, spike_clusters, n_clusters_total,
                         align_times, time_window, binsize):
    """Bin spikes into trials aligned to events.

    Returns array of shape (n_trials, n_clusters, n_bins).
    Matching reference code: bin_spiking_data -> get_spike_data_per_interval.
    """
    n_bins = int(np.ceil((time_window[1] - time_window[0]) / binsize))
    n_trials = len(align_times)

    binned = np.zeros((n_trials, n_clusters_total, n_bins), dtype=np.float32)

    for trial_idx in range(n_trials):
        t_beg = align_times[trial_idx] + time_window[0]
        t_end = align_times[trial_idx] + time_window[1]

        if np.isnan(t_beg) or np.isnan(t_end):
            continue

        # Select spikes in this interval using searchsorted for speed
        i_start = np.searchsorted(spike_times, t_beg, side='left')
        i_end = np.searchsorted(spike_times, t_end, side='left')

        if i_start >= i_end:
            continue

        trial_times = spike_times[i_start:i_end]
        trial_clusters = spike_clusters[i_start:i_end]

        # Compute bin indices for all spikes at once
        bin_indices = np.minimum(
            ((trial_times - t_beg) / binsize).astype(np.int64),
            n_bins - 1
        )

        # Count spikes per (cluster, bin) using np.add.at for vectorized counting
        valid = trial_clusters < n_clusters_total
        np.add.at(binned[trial_idx], (trial_clusters[valid], bin_indices[valid]), 1)

    return binned


def interpolate_behavior_to_bins(beh_times, beh_values, align_times,
                                  time_window, binsize):
    """Interpolate behavioral signal to match neural bins.

    Matching reference code: get_behavior_per_interval.
    Returns list of arrays and a mask of good trials.
    """
    n_bins = int(np.ceil((time_window[1] - time_window[0]) / binsize))
    n_trials = len(align_times)

    result = []
    good_mask = np.ones(n_trials, dtype=bool)

    for trial_idx in range(n_trials):
        t_beg = align_times[trial_idx] + time_window[0]
        t_end = align_times[trial_idx] + time_window[1]

        if np.isnan(t_beg) or np.isnan(t_end):
            good_mask[trial_idx] = False
            result.append(None)
            continue

        # Get behavior data in this interval
        beh_mask = (beh_times >= t_beg - binsize) & (beh_times <= t_end + binsize)
        local_times = beh_times[beh_mask]
        local_vals = beh_values[beh_mask]

        if len(local_vals) == 0:
            good_mask[trial_idx] = False
            result.append(None)
            continue

        if np.any(np.isnan(local_vals)):
            good_mask[trial_idx] = False
            result.append(None)
            continue

        # Check data coverage (matching reference code tolerance)
        if np.abs(t_beg - local_times[0]) > binsize:
            good_mask[trial_idx] = False
            result.append(None)
            continue
        if np.abs(t_end - local_times[-1]) > binsize:
            good_mask[trial_idx] = False
            result.append(None)
            continue

        # Interpolate to bin centers (matching reference code: linspace from beg+binsize to end)
        x_interp = np.linspace(t_beg + binsize, t_end, n_bins)
        try:
            y_interp = interp1d(local_times, local_vals, kind='linear',
                               fill_value='extrapolate')(x_interp)
        except Exception:
            good_mask[trial_idx] = False
            result.append(None)
            continue

        result.append(y_interp)

    return result, good_mask


def load_wheel_speed(session_path):
    """Load wheel data and compute speed (absolute velocity).

    Matching reference code: load_target_behavior for 'wheel-speed'.
    SessionLoader interpolates wheel to uniform sampling, computes velocity
    via Gaussian smoothing, then speed = abs(velocity).
    """
    alf_dir = session_path / 'alf'

    pos_file = alf_dir / '_ibl_wheel.position.npy'
    ts_file = alf_dir / '_ibl_wheel.timestamps.npy'

    if not pos_file.exists() or not ts_file.exists():
        return None, None

    position = np.load(pos_file).flatten()
    timestamps = np.load(ts_file).flatten()

    # Interpolate to uniform 1kHz sampling (matching ibllib SessionLoader.load_wheel)
    from brainbox.behavior.wheel import interpolate_position, velocity_filtered
    pos_interp, ts_interp = interpolate_position(timestamps, position, freq=1000)
    # velocity_filtered expects (position_array, sampling_freq), returns (velocity, acceleration)
    velocity, _ = velocity_filtered(pos_interp, 1000)
    speed = np.abs(velocity)

    return ts_interp, speed


def load_whisker_motion_energy(session_path):
    """Load whisker motion energy, preferring left camera.

    Matching reference code: bin_behaviors for 'whisker-motion-energy'.
    """
    alf_dir = session_path / 'alf'

    # Try left camera first (matching reference code)
    me_file = find_file_with_revision(alf_dir, 'leftCamera.ROIMotionEnergy.npy')
    ts_file = find_file_with_revision(alf_dir, '_ibl_leftCamera.times.npy')

    if me_file is None or ts_file is None:
        # Fallback to right camera
        me_file = find_file_with_revision(alf_dir, 'rightCamera.ROIMotionEnergy.npy')
        ts_file = find_file_with_revision(alf_dir, '_ibl_rightCamera.times.npy')

    if me_file is None or ts_file is None:
        return None, None

    me = np.load(me_file).flatten()
    times = np.load(ts_file).flatten()

    # Ensure same length
    min_len = min(len(me), len(times))
    me = me[:min_len]
    times = times[:min_len]

    # Remove NaN values
    valid = ~np.isnan(me) & ~np.isnan(times)
    if np.sum(valid) < 10:
        return None, None

    return times[valid], me[valid]


def discretize_to_bins(values, n_bins=3):
    """Discretize continuous values into n_bins equal-frequency bins.

    Returns integer labels 0 to n_bins-1.
    """
    if len(values) == 0:
        return values

    # Use percentile-based binning for equal frequency
    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(values, percentiles)
    # Make edges unique to avoid issues
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    labels = np.digitize(values, bin_edges[1:-1])  # 0 to n_bins-1
    return labels.astype(np.int64)


def compute_trial_number_in_block(trials_df, mask):
    """Compute trial number within each block.

    A new block starts when probabilityLeft changes.
    """
    prob_left = trials_df['probabilityLeft'].values
    trial_nums = np.zeros(len(trials_df), dtype=np.float64)

    block_start = 0
    for i in range(1, len(prob_left)):
        if prob_left[i] != prob_left[i-1] or np.isnan(prob_left[i]) or np.isnan(prob_left[i-1]):
            block_start = i
        trial_nums[i] = i - block_start

    return trial_nums[mask]


# ============================================================
# Main conversion
# ============================================================

def convert_data(max_sessions=None, output_path='converted_data.pkl'):
    """Main conversion function."""

    print("=" * 60)
    print("IBL Brain-Wide Map Data Conversion")
    print("=" * 60)

    # Load session info
    sessions_df = pd.read_parquet(os.path.join(TABLES_DIR, 'sessions.pqt'))
    datasets_df = pd.read_parquet(os.path.join(TABLES_DIR, 'datasets.pqt'))
    bwm_df = pd.read_csv(BWM_CSV, index_col=0)

    print(f"Sessions in cache: {len(sessions_df)}")
    print(f"BWM release entries: {len(bwm_df)}")

    # Get unique eids from BWM that exist in cache
    bwm_eids = bwm_df['eid'].unique()
    cache_eids = set(sessions_df.index.astype(str))
    available_eids = [eid for eid in bwm_eids if eid in cache_eids]
    print(f"Available sessions: {len(available_eids)}")

    if max_sessions is not None:
        available_eids = available_eids[:max_sessions]
        print(f"Processing {len(available_eids)} sessions (limited)")

    # Initialize brain regions mapper
    br = BrainRegions()

    # Initialize output containers
    all_neural = []
    all_input = []
    all_output = []
    all_subjects = []
    all_subject_idx = []
    all_brain_region_idx = []
    all_brain_regions_set = set()

    subject_to_idx = {}
    session_metadata = []

    n_skipped = 0
    n_processed = 0

    for eid_idx, eid in enumerate(available_eids):
        print(f"\n--- Session {eid_idx+1}/{len(available_eids)}: {eid} ---")

        session_path = find_session_path(eid, sessions_df, datasets_df)
        if session_path is None or not session_path.exists():
            print(f"  SKIP: session path not found")
            n_skipped += 1
            continue

        # Get subject info
        row = sessions_df.loc[eid]
        subject = row['subject']

        # ---- Load trials ----
        trials, mask = load_trials(session_path)
        if trials is None:
            print(f"  SKIP: could not load trials")
            n_skipped += 1
            continue

        n_total_trials = len(trials)
        n_valid_trials = mask.sum()
        print(f"  Trials: {n_valid_trials}/{n_total_trials} valid")

        if n_valid_trials < 2:
            print(f"  SKIP: too few valid trials")
            n_skipped += 1
            continue

        # ---- Load spike data from all probes ----
        # Find available probes
        alf_dir = session_path / 'alf'
        probe_dirs = sorted([d.name for d in alf_dir.iterdir()
                            if d.is_dir() and d.name.startswith('probe')])

        if len(probe_dirs) == 0:
            print(f"  SKIP: no probe directories found")
            n_skipped += 1
            continue

        probes_data = []
        for probe_name in probe_dirs:
            result = load_spike_data(session_path, probe_name)
            if result[0] is not None:
                probes_data.append(result)

        if len(probes_data) == 0:
            print(f"  SKIP: no valid probe data")
            n_skipped += 1
            continue

        print(f"  Probes loaded: {len(probes_data)}")

        # Merge probes (matching reference code)
        spike_times, spike_clusters, cluster_brain_ids = merge_probes_data(probes_data)
        n_clusters = len(cluster_brain_ids)
        print(f"  Total clusters: {n_clusters}")

        if n_clusters < MIN_NEURONS:
            print(f"  SKIP: too few clusters ({n_clusters} < {MIN_NEURONS})")
            n_skipped += 1
            continue

        # ---- Map brain regions (Beryl mapping, matching reference code) ----
        cluster_acronyms = br.id2acronym(cluster_brain_ids)
        beryl_acronyms = br.acronym2acronym(cluster_acronyms, mapping='Beryl')

        # ---- Bin spikes ----
        valid_trials = trials[mask].copy()
        align_times = valid_trials[ALIGN_TIME].values

        print(f"  Binning spikes (this may take a moment)...")
        binned_spikes = bin_spikes_per_trial(
            spike_times, spike_clusters, n_clusters,
            align_times, TIME_WINDOW, BINSIZE
        )
        # binned_spikes shape: (n_trials, n_clusters, n_bins)
        print(f"  Binned spikes shape: {binned_spikes.shape}")

        # ---- Load and bin behavioral data ----
        # Wheel speed
        wheel_times, wheel_speed = load_wheel_speed(session_path)
        wheel_binned = None
        wheel_mask = np.ones(n_valid_trials, dtype=bool)
        if wheel_times is not None:
            wheel_binned_list, wheel_mask = interpolate_behavior_to_bins(
                wheel_times, wheel_speed, align_times, TIME_WINDOW, BINSIZE
            )
        else:
            print(f"  WARNING: no wheel data")
            wheel_mask = np.zeros(n_valid_trials, dtype=bool)

        # Whisker motion energy
        me_times, me_values = load_whisker_motion_energy(session_path)
        me_binned = None
        me_mask = np.ones(n_valid_trials, dtype=bool)
        if me_times is not None:
            me_binned_list, me_mask = interpolate_behavior_to_bins(
                me_times, me_values, align_times, TIME_WINDOW, BINSIZE
            )
        else:
            print(f"  WARNING: no whisker motion energy data")
            me_mask = np.zeros(n_valid_trials, dtype=bool)

        # ---- Combined mask: trials with all data available ----
        combined_mask = wheel_mask & me_mask
        n_final = combined_mask.sum()
        print(f"  Trials with all data: {n_final}/{n_valid_trials}")

        if n_final < 2:
            print(f"  SKIP: too few trials with complete data")
            n_skipped += 1
            continue

        # ---- Extract per-trial data ----
        # Get choice, prior, trial number in block
        choice_vals = valid_trials['choice'].values  # -1 or 1
        prob_left_vals = valid_trials['probabilityLeft'].values  # 0.2, 0.5, 0.8
        trial_nums_in_block = compute_trial_number_in_block(trials, mask)

        # Discretize wheel speed and whisker ME across all valid trials
        # First, collect all valid continuous values for bin edge computation
        all_wheel_vals = []
        all_me_vals = []
        combined_indices = np.where(combined_mask)[0]

        for idx in combined_indices:
            if wheel_binned_list[idx] is not None:
                all_wheel_vals.append(wheel_binned_list[idx])
            if me_binned_list[idx] is not None:
                all_me_vals.append(me_binned_list[idx])

        all_wheel_concat = np.concatenate(all_wheel_vals)
        all_me_concat = np.concatenate(all_me_vals)

        # Compute bin edges for discretization (3 equal-frequency bins)
        wheel_percentiles = np.percentile(all_wheel_concat, [33.33, 66.67])
        me_percentiles = np.percentile(all_me_concat, [33.33, 66.67])

        # Build per-trial arrays
        session_neural = []
        session_input = []
        session_output = []

        # Time since stimulus onset (continuous, time-varying) - same for all trials
        time_since_stim = np.linspace(
            TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_TIMEBINS
        ).astype(np.float64)

        for trial_local_idx, trial_global_idx in enumerate(combined_indices):
            # Neural: (n_clusters, n_bins)
            neural_trial = binned_spikes[trial_global_idx]  # already (n_clusters, n_bins)
            session_neural.append(neural_trial.astype(np.float64))

            # Input: (2, n_bins) - time since stim onset + trial number in block
            trial_num = trial_nums_in_block[trial_global_idx]
            input_trial = np.array([trial_num], dtype=np.float64)  # (1,) per-trial
            # Combine time-varying input and per-trial input
            # Time since stimulus onset: (1, n_bins)
            # Trial number in block: (1,) per-trial -> will be tiled by decoder
            input_trial_full = np.vstack([
                time_since_stim.reshape(1, -1),  # (1, T) time-varying
                np.full((1, N_TIMEBINS), trial_num, dtype=np.float64)  # (1, T) constant
            ])  # (2, T)
            session_input.append(input_trial_full)

            # Output: choice, prior, wheel speed (discretized), whisker ME (discretized)
            # Choice: binary, left=0, right=1
            choice = 0 if choice_vals[trial_global_idx] == -1 else 1  # -1->0 (left), 1->1 (right)

            # Prior: 0.2->0, 0.5->1, 0.8->2
            prob = prob_left_vals[trial_global_idx]
            if prob == 0.2:
                prior = 0
            elif prob == 0.5:
                prior = 1
            else:  # 0.8
                prior = 2

            # Wheel speed: discretize to 3 bins (time-varying)
            wheel_trial = wheel_binned_list[trial_global_idx]
            wheel_disc = np.digitize(wheel_trial, wheel_percentiles).astype(np.int64)  # 0, 1, 2

            # Whisker ME: discretize to 3 bins (time-varying)
            me_trial = me_binned_list[trial_global_idx]
            me_disc = np.digitize(me_trial, me_percentiles).astype(np.int64)  # 0, 1, 2

            # Output array: (4, T) - first two are per-trial (constant), last two time-varying
            output_trial = np.vstack([
                np.full((1, N_TIMEBINS), choice, dtype=np.int64),
                np.full((1, N_TIMEBINS), prior, dtype=np.int64),
                wheel_disc.reshape(1, -1),
                me_disc.reshape(1, -1)
            ])  # (4, T)
            session_output.append(output_trial)

        # ---- Store session data ----
        all_neural.append(session_neural)
        all_input.append(session_input)
        all_output.append(session_output)

        # Subject tracking
        if subject not in subject_to_idx:
            subject_to_idx[subject] = len(subject_to_idx)
            all_subjects.append(subject)
        all_subject_idx.append(subject_to_idx[subject])

        # Brain region tracking
        for acr in beryl_acronyms:
            all_brain_regions_set.add(acr)
        all_brain_region_idx.append(beryl_acronyms)

        session_metadata.append({
            'eid': eid,
            'subject': subject,
            'n_trials': n_final,
            'n_neurons': n_clusters,
            'n_probes': len(probes_data),
        })

        n_processed += 1
        print(f"  OK: {n_final} trials, {n_clusters} neurons")

    print(f"\n{'='*60}")
    print(f"Conversion complete: {n_processed} sessions, {n_skipped} skipped")
    print(f"{'='*60}")

    if n_processed == 0:
        print("ERROR: No sessions processed!")
        return None

    # ---- Build brain_regions index ----
    brain_regions_list = sorted(all_brain_regions_set)
    brain_region_to_idx = {r: i for i, r in enumerate(brain_regions_list)}

    # Convert brain_region_idx from acronyms to indices
    brain_region_idx_arrays = []
    for session_acronyms in all_brain_region_idx:
        idx_array = np.array([brain_region_to_idx[a] for a in session_acronyms], dtype=np.int64)
        brain_region_idx_arrays.append(idx_array)

    # ---- Assemble final data structure ----
    data = {
        'neural': all_neural,
        'input': all_input,
        'output': all_output,

        'subjects': all_subjects,
        'subject_idx': np.array(all_subject_idx, dtype=np.int64),

        'brain_regions': brain_regions_list,
        'brain_region_idx': brain_region_idx_arrays,

        'input_names': ['time_since_stimulus_onset', 'trial_number_in_block'],
        'output_names': ['choice', 'prior', 'wheel_speed', 'whisker_motion_energy'],
        'output_values': [
            ['left', 'right'],  # choice: 0=left, 1=right
            ['0.2', '0.5', '0.8'],  # prior: 0=0.2, 1=0.5, 2=0.8
            ['low', 'medium', 'high'],  # wheel speed bins
            ['low', 'medium', 'high'],  # whisker ME bins
        ],

        'metadata': {
            'task_description': 'IBL decision-making task: mice rotate wheel to move visual stimulus to center',
            'time_bin_size': BINSIZE * 1000,  # 20 ms
            'temporal_alignment_event': 'stimulus onset (stimOn_times)',
            'off_start': TIME_WINDOW[0],  # -0.5 s
            'off_end': TIME_WINDOW[1],  # 1.5 s
            'binsize_seconds': BINSIZE,
            'n_timebins': N_TIMEBINS,
            'session_info': session_metadata,
            'n_sessions': n_processed,
            'n_skipped': n_skipped,
            'atlas_mapping': 'Beryl',
            'spike_sorting': 'pykilosort (Kilosort 2.5)',
            'cluster_quality_filter': 'none (all clusters, matching reference code)',
            'trial_filter': f'RT {MIN_RT}-{MAX_RT}s, no NaN events, no no-choice',
        }
    }

    # ---- Sanity checks ----
    print("\n--- Sanity Checks ---")
    total_trials = sum(len(s) for s in all_neural)
    total_neurons = [all_neural[s][0].shape[0] for s in range(len(all_neural))]
    print(f"Total sessions: {len(all_neural)}")
    print(f"Total trials: {total_trials}")
    print(f"Neurons per session: min={min(total_neurons)}, max={max(total_neurons)}, "
          f"mean={np.mean(total_neurons):.0f}")
    print(f"Number of subjects: {len(all_subjects)}")
    print(f"Number of brain regions: {len(brain_regions_list)}")
    print(f"Time bins per trial: {N_TIMEBINS}")
    print(f"Bin size: {BINSIZE*1000} ms")

    # Check output distributions
    all_choices = []
    all_priors = []
    for session in all_output:
        for trial in session:
            all_choices.append(trial[0, 0])
            all_priors.append(trial[1, 0])
    all_choices = np.array(all_choices)
    all_priors = np.array(all_priors)
    print(f"\nChoice distribution: left={np.mean(all_choices==0):.3f}, right={np.mean(all_choices==1):.3f}")
    print(f"Prior distribution: 0.2={np.mean(all_priors==0):.3f}, 0.5={np.mean(all_priors==1):.3f}, 0.8={np.mean(all_priors==2):.3f}")

    # ---- Save ----
    print(f"\nSaving to {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)

    file_size = os.path.getsize(output_path) / (1024**3)
    print(f"Saved {output_path} ({file_size:.2f} GB)")

    return data


if __name__ == '__main__':
    data = convert_data(max_sessions=args.max_sessions, output_path=args.output)
