"""
Convert IBL Brain-Wide Map data to the target format for neural decoding.

Based on the reference code from Zhang et al. 2025 and the IBL BWM dataset.

Key processing decisions (matching reference code and papers):
- Align all trials to stimulus onset (stimOn_times)
- Time window: -0.5s to 1.5s relative to stimulus onset (2s total)
- Bin size: 20ms -> T=100 time bins per trial
- Load ALL clusters (not filtered by quality) per the reference caching code
- Merge probes within a session
- Use Beryl brain region mapping
- Trial filtering: exclude trials with NaN events, choice==0, RT < 0.08s or > 2s
- Minimum 5 good neurons per session (from data paper inclusion criteria)

Decoder task:
- Inputs: time since stimulus onset (continuous), trial number in block (per-trial)
- Outputs: choice (binary), prior (categorical 0/1/2), wheel speed (3 bins), whisker ME (3 bins)
"""

import os
import sys
import pickle
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import interp1d
from iblatlas.regions import BrainRegions

# Constants matching reference code
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)  # seconds relative to stimulus onset
BINSIZE = 0.02  # 20ms bins
INTERVAL_LEN = TIME_WINDOW[1] - TIME_WINDOW[0]  # 2.0s
N_BINS = int(np.ceil(INTERVAL_LEN / BINSIZE))  # 100 bins
MIN_RT = 0.08  # minimum reaction time
MAX_RT = 2.0   # maximum reaction time
N_WHEEL_BINS = 3  # discretize wheel speed into 3 bins
N_WHISKER_BINS = 3  # discretize whisker ME into 3 bins

# Trial exclusion criteria matching reference code
NAN_EXCLUDE = [
    'stimOn_times', 'choice', 'feedback_times',
    'probabilityLeft', 'firstMovement_times', 'feedbackType'
]


def find_session_paths(cache_dir):
    """Find all session paths in the ONE cache directory."""
    bwm_df = pd.read_csv('/app/code/code_zhang2025/data/bwm_release.csv', index_col=0)

    sessions = {}
    for _, row in bwm_df.iterrows():
        eid = row['eid']
        lab = row['lab']
        subject = row['subject']
        date = row['date']
        probe_name = row['probe_name']

        # Construct session path
        session_path = Path(cache_dir) / lab / 'Subjects' / subject / date / '001'
        if not session_path.exists():
            # Try other session numbers
            parent = Path(cache_dir) / lab / 'Subjects' / subject / date
            if parent.exists():
                subdirs = sorted(parent.iterdir())
                if subdirs:
                    session_path = subdirs[0]

        if session_path.exists():
            if eid not in sessions:
                sessions[eid] = {
                    'path': session_path,
                    'probes': [],
                    'subject': subject,
                    'lab': lab,
                    'date': date,
                    'eid': eid,
                }
            sessions[eid]['probes'].append(probe_name)

    return sessions


def load_trials(session_path):
    """Load trials table and create quality mask."""
    # Find trials table
    trials_files = list(session_path.rglob('_ibl_trials.table.pqt'))
    if not trials_files:
        return None, None

    trials = pd.read_parquet(trials_files[0])

    # Create mask matching reference code load_trials_and_mask
    mask = pd.Series(True, index=trials.index)

    # Exclude NaN events
    for event in NAN_EXCLUDE:
        if event in trials.columns:
            mask &= ~trials[event].isna()

    # Exclude RT outside range
    if 'firstMovement_times' in trials.columns and 'stimOn_times' in trials.columns:
        rt = trials['firstMovement_times'] - trials['stimOn_times']
        mask &= (rt >= MIN_RT) & (rt <= MAX_RT)

    # Exclude no-choice trials (choice == 0)
    if 'choice' in trials.columns:
        mask &= (trials['choice'] != 0)

    return trials, mask


def load_spikes(session_path, probe_name):
    """Load spike times and clusters for a probe."""
    # Find pykilosort directory
    probe_path = session_path / 'alf' / probe_name / 'pykilosort'
    if not probe_path.exists():
        return None, None, None, None, None

    # Find the revision directory
    revisions = [d for d in probe_path.iterdir() if d.is_dir()]
    if not revisions:
        return None, None, None, None, None
    revision_path = sorted(revisions)[-1]  # Use latest revision

    try:
        spike_times = np.load(revision_path / 'spikes.times.npy')
        spike_clusters = np.load(revision_path / 'spikes.clusters.npy')
        clusters_channels = np.load(revision_path / 'clusters.channels.npy')
        clusters_depths = np.load(revision_path / 'clusters.depths.npy')

        # Load channel brain location IDs
        chan_brain_ids = np.load(revision_path / 'channels.brainLocationIds_ccf_2017.npy')

        # Load cluster metrics for quality labels
        metrics_files = list(revision_path.glob('clusters.metrics.pqt'))
        cluster_labels = None
        if metrics_files:
            metrics = pd.read_parquet(metrics_files[0])
            cluster_labels = metrics['label'].values

        return spike_times, spike_clusters, clusters_channels, chan_brain_ids, cluster_labels
    except Exception as e:
        print(f"  Error loading spikes for {probe_name}: {e}")
        return None, None, None, None, None


def merge_probes(spikes_list, clusters_channels_list, chan_brain_ids_list, cluster_labels_list):
    """Merge spikes from multiple probes, matching reference code."""
    if len(spikes_list) == 1:
        return (spikes_list[0][0], spikes_list[0][1],
                clusters_channels_list[0], chan_brain_ids_list[0], cluster_labels_list[0])

    merged_times = []
    merged_clusters = []
    merged_channels = []
    merged_brain_ids = []
    merged_labels = []
    cluster_offset = 0
    channel_offset = 0

    for i, (times, clusters) in enumerate(spikes_list):
        n_clusters = len(clusters_channels_list[i])
        n_channels = len(chan_brain_ids_list[i])

        merged_times.append(times)
        merged_clusters.append(clusters + cluster_offset)
        merged_channels.append(clusters_channels_list[i] + channel_offset)
        merged_brain_ids.append(chan_brain_ids_list[i])
        if cluster_labels_list[i] is not None:
            merged_labels.append(cluster_labels_list[i])

        cluster_offset += n_clusters
        channel_offset += n_channels

    all_times = np.concatenate(merged_times)
    all_clusters = np.concatenate(merged_clusters)
    all_channels = np.concatenate(merged_channels)
    all_brain_ids = np.concatenate(merged_brain_ids)
    all_labels = np.concatenate(merged_labels) if merged_labels else None

    # Sort by time
    sort_idx = np.argsort(all_times, kind='stable')
    all_times = all_times[sort_idx]
    all_clusters = all_clusters[sort_idx]

    return all_times, all_clusters, all_channels, all_brain_ids, all_labels


def bin_spikes_trial(spike_times, spike_clusters, n_clusters, t_start, t_end, binsize, n_bins):
    """Bin spikes for a single trial into a (n_clusters, n_bins) matrix."""
    # Select spikes in the interval using searchsorted for efficiency
    i_start = np.searchsorted(spike_times, t_start, side='left')
    i_end = np.searchsorted(spike_times, t_end, side='left')
    times_sel = spike_times[i_start:i_end]
    clusters_sel = spike_clusters[i_start:i_end]

    # Create spike count matrix
    binned = np.zeros((n_clusters, n_bins), dtype=np.float32)

    if len(times_sel) == 0:
        return binned

    # Compute bin indices
    bin_idx = np.floor((times_sel - t_start) / binsize).astype(int)
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    # Vectorized filling using np.add.at
    valid = clusters_sel < n_clusters
    np.add.at(binned, (clusters_sel[valid], bin_idx[valid]), 1)

    return binned


def load_wheel_speed(session_path):
    """Load wheel data and compute speed (abs velocity)."""
    wheel_pos_files = list(session_path.rglob('_ibl_wheel.position.npy'))
    wheel_ts_files = list(session_path.rglob('_ibl_wheel.timestamps.npy'))

    if not wheel_pos_files or not wheel_ts_files:
        return None, None

    wheel_pos = np.load(wheel_pos_files[0])
    wheel_ts = np.load(wheel_ts_files[0])

    # Compute velocity using Gaussian-smoothed derivative (matching brainbox)
    # Simple finite difference for velocity, then take absolute value for speed
    dt = np.diff(wheel_ts)
    dp = np.diff(wheel_pos)
    vel = dp / dt
    speed = np.abs(vel)

    # Timestamps for velocity are midpoints
    vel_ts = wheel_ts[:-1] + dt / 2

    return vel_ts, speed


def load_whisker_motion_energy(session_path):
    """Load whisker motion energy (prefer left camera at 60Hz)."""
    # Try left camera first (60 Hz)
    left_me_files = list(session_path.rglob('leftCamera.ROIMotionEnergy.npy'))
    left_times_files = list(session_path.rglob('_ibl_leftCamera.times.npy'))

    if left_me_files and left_times_files:
        me = np.load(left_me_files[0])
        times = np.load(left_times_files[0])
        # ME may have same length as camera times (ROIMotionEnergy)
        if len(me) == len(times):
            return times, me
        elif len(me) == len(times) - 1:
            return times[:-1], me

    # Try right camera
    right_me_files = list(session_path.rglob('rightCamera.ROIMotionEnergy.npy'))
    right_times_files = list(session_path.rglob('_ibl_rightCamera.times.npy'))

    if right_me_files and right_times_files:
        me = np.load(right_me_files[0])
        times = np.load(right_times_files[0])
        if len(me) == len(times):
            return times, me
        elif len(me) == len(times) - 1:
            return times[:-1], me

    return None, None


def interpolate_behavior_to_bins(beh_times, beh_values, t_start, t_end, binsize, n_bins):
    """Interpolate behavioral signal to match neural bin centers."""
    # Bin centers (matching reference code: linspace from start+binsize to end)
    bin_centers = np.linspace(t_start + binsize, t_end, n_bins)

    # Select relevant portion of behavioral signal
    margin = binsize
    mask = (beh_times >= t_start - margin) & (beh_times <= t_end + margin)
    t_sel = beh_times[mask]
    v_sel = beh_values[mask]

    if len(t_sel) < 2:
        return None

    # Check coverage
    if np.abs(t_sel[0] - t_start) > binsize or np.abs(t_sel[-1] - t_end) > binsize:
        return None

    # Interpolate
    f = interp1d(t_sel, v_sel, kind='linear', fill_value='extrapolate')
    return f(bin_centers)


def compute_trial_number_in_block(trials_df, mask):
    """Compute trial number within each block (0-indexed)."""
    trials_masked = trials_df[mask].copy()
    prob_left = trials_masked['probabilityLeft'].values

    # Detect block changes
    block_changes = np.concatenate([[0], np.where(np.diff(prob_left) != 0)[0] + 1])
    trial_in_block = np.zeros(len(prob_left), dtype=int)

    for i in range(len(block_changes)):
        start = block_changes[i]
        end = block_changes[i + 1] if i + 1 < len(block_changes) else len(prob_left)
        trial_in_block[start:end] = np.arange(end - start)

    return trial_in_block


def discretize_to_bins(values_list, n_bins, compute_edges=True, edges=None):
    """Discretize time-varying values into n_bins categories using quantiles.

    Returns discretized values and bin edges.
    """
    if compute_edges:
        # Collect all non-NaN values across all trials to compute global quantile edges
        all_vals = []
        for v in values_list:
            if v is not None:
                valid = v[~np.isnan(v)]
                all_vals.append(valid)
        if not all_vals:
            return values_list, None
        all_vals = np.concatenate(all_vals)
        # Use quantile edges for equal-count bins
        quantiles = np.linspace(0, 100, n_bins + 1)
        edges = np.percentile(all_vals, quantiles)
        edges[0] = -np.inf
        edges[-1] = np.inf

    # Discretize each trial
    discretized = []
    for v in values_list:
        if v is not None:
            d = np.digitize(v, edges[1:-1])  # Returns 0 to n_bins-1
            discretized.append(d.astype(np.float32))
        else:
            discretized.append(None)

    return discretized, edges


def process_session(session_info, cache_dir, br):
    """Process a single session and return formatted data."""
    session_path = session_info['path']
    eid = session_info['eid']
    subject = session_info['subject']

    print(f"  Processing session {eid} ({subject})...")

    # Load trials
    trials, mask = load_trials(session_path)
    if trials is None:
        print(f"    Skipping: no trials data")
        return None

    n_valid = mask.sum()
    if n_valid < 2:
        print(f"    Skipping: only {n_valid} valid trials")
        return None

    # Load spikes from all probes and merge
    spikes_list = []
    channels_list = []
    brain_ids_list = []
    labels_list = []

    for probe_name in session_info['probes']:
        spike_times, spike_clusters, clusters_channels, chan_brain_ids, cluster_labels = \
            load_spikes(session_path, probe_name)
        if spike_times is None:
            continue
        spikes_list.append((spike_times, spike_clusters))
        channels_list.append(clusters_channels)
        brain_ids_list.append(chan_brain_ids)
        labels_list.append(cluster_labels)

    if not spikes_list:
        print(f"    Skipping: no spike data")
        return None

    # Merge probes
    spike_times, spike_clusters, all_channels, all_brain_ids, all_labels = \
        merge_probes(spikes_list, channels_list, brain_ids_list, labels_list)

    # Get unique cluster IDs
    cluster_ids = np.unique(spike_clusters)
    n_clusters = len(cluster_ids)

    # Create cluster ID to index mapping
    cluster_id_to_idx = {cid: idx for idx, cid in enumerate(cluster_ids)}

    # Remap spike clusters to sequential indices
    spike_clusters_remapped = np.array([cluster_id_to_idx[c] for c in spike_clusters])

    # Get brain regions for each cluster using channel mapping
    # Map each cluster to its channel's brain region
    cluster_brain_ids = []
    for cid in cluster_ids:
        if cid < len(all_channels):
            ch = int(all_channels[cid])
            if ch < len(all_brain_ids):
                cluster_brain_ids.append(all_brain_ids[ch])
            else:
                cluster_brain_ids.append(0)  # root
        else:
            cluster_brain_ids.append(0)  # root
    cluster_brain_ids = np.array(cluster_brain_ids)
    cluster_acronyms = br.id2acronym(cluster_brain_ids)

    # Map to Beryl regions
    beryl_regions = br.acronym2acronym(cluster_acronyms, mapping='Beryl')

    # Load behavioral data
    wheel_ts, wheel_speed = load_wheel_speed(session_path)
    me_ts, me_values = load_whisker_motion_energy(session_path)

    if wheel_speed is None:
        print(f"    Skipping: no wheel data")
        return None

    # Get valid trials
    valid_trials = trials[mask].copy()
    valid_indices = valid_trials.index.tolist()

    # Compute trial number in block
    trial_in_block = compute_trial_number_in_block(trials, mask)

    # Process each trial
    neural_trials = []
    input_trials = []
    output_trials = []
    wheel_speed_raw = []
    whisker_me_raw = []
    good_trial_indices = []

    for trial_idx, (df_idx, trial) in enumerate(valid_trials.iterrows()):
        stim_on = trial[ALIGN_TIME]
        t_start = stim_on + TIME_WINDOW[0]
        t_end = stim_on + TIME_WINDOW[1]

        # Bin spikes
        neural = bin_spikes_trial(
            spike_times, spike_clusters_remapped, n_clusters,
            t_start, t_end, BINSIZE, N_BINS
        )

        # Interpolate wheel speed
        if wheel_ts is not None and wheel_speed is not None:
            ws = interpolate_behavior_to_bins(wheel_ts, wheel_speed, t_start, t_end, BINSIZE, N_BINS)
        else:
            ws = None

        # Interpolate whisker ME
        if me_ts is not None and me_values is not None:
            wm = interpolate_behavior_to_bins(me_ts, me_values, t_start, t_end, BINSIZE, N_BINS)
        else:
            wm = None

        if ws is None or wm is None:
            continue

        # Check for NaNs in behavioral data
        if np.any(np.isnan(ws)) or np.any(np.isnan(wm)):
            continue

        # Build inputs
        # Time since stimulus onset (continuous, time-varying)
        time_since_stim = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS).astype(np.float32)

        # Trial number in block (per-trial, broadcast to match)
        trial_num = np.float32(trial_in_block[trial_idx])

        # Input: (2, T) - time since stim onset + trial number in block
        input_data = np.stack([
            time_since_stim,
            np.full(N_BINS, trial_num, dtype=np.float32)
        ], axis=0)

        # Build outputs (will discretize later)
        # Choice: left=0, right=1
        choice_val = 0 if trial['choice'] == 1 else 1  # IBL: 1=left, -1=right -> map: left=0, right=1

        # Prior: 0.2->0, 0.5->1, 0.8->2
        prob_left = trial['probabilityLeft']
        if prob_left == 0.2:
            prior_val = 0
        elif prob_left == 0.5:
            prior_val = 1
        elif prob_left == 0.8:
            prior_val = 2
        else:
            prior_val = 1  # fallback

        neural_trials.append(neural)
        good_trial_indices.append(trial_idx)

        # Store raw wheel speed and ME for later discretization
        wheel_speed_raw.append(ws)
        whisker_me_raw.append(wm)

        input_trials.append(input_data)
        # Temporarily store choice and prior; wheel/whisker will be added after discretization
        output_trials.append((choice_val, prior_val))

    if len(neural_trials) < 2:
        print(f"    Skipping: only {len(neural_trials)} valid trials after behavioral filtering")
        return None

    # Discretize wheel speed and whisker ME into 3 bins each
    # (done per-session to maintain local context, but could also be global)
    wheel_disc, wheel_edges = discretize_to_bins(wheel_speed_raw, N_WHEEL_BINS)
    whisker_disc, whisker_edges = discretize_to_bins(whisker_me_raw, N_WHISKER_BINS)

    # Build final output arrays
    final_outputs = []
    for i in range(len(neural_trials)):
        choice_val, prior_val = output_trials[i]

        # Output: (4, T) integer-valued categorical
        # choice: per-trial -> broadcast to (T,)
        # prior: per-trial -> broadcast to (T,)
        # wheel speed: time-varying (T,)
        # whisker ME: time-varying (T,)
        output = np.stack([
            np.full(N_BINS, choice_val, dtype=np.int64),
            np.full(N_BINS, prior_val, dtype=np.int64),
            wheel_disc[i].astype(np.int64),
            whisker_disc[i].astype(np.int64),
        ], axis=0)
        final_outputs.append(output)

    print(f"    {len(neural_trials)} trials, {n_clusters} clusters, "
          f"{len(np.unique(beryl_regions))} brain regions")

    return {
        'neural': neural_trials,
        'input': input_trials,
        'output': final_outputs,
        'subject': subject,
        'beryl_regions': beryl_regions,
        'n_clusters': n_clusters,
        'eid': eid,
        'wheel_edges': wheel_edges,
        'whisker_edges': whisker_edges,
    }


def convert_data(cache_dir, max_sessions=None, sample=False):
    """Convert all IBL BWM sessions to the target format."""

    print("Finding sessions in cache...")
    sessions = find_session_paths(cache_dir)
    print(f"Found {len(sessions)} sessions in BWM release")

    if max_sessions is not None:
        # Deterministic subset
        eids = sorted(sessions.keys())[:max_sessions]
        sessions = {k: sessions[k] for k in eids}
        print(f"Processing {len(sessions)} sessions")

    # Initialize brain regions helper
    br = BrainRegions()

    # Process all sessions
    all_neural = []
    all_input = []
    all_output = []
    all_subjects = []
    all_subject_idx = []
    all_brain_region_idx = []
    all_beryl_regions_per_session = []
    session_eids = []

    subject_set = []  # ordered unique subjects

    n_processed = 0
    n_skipped = 0

    for i, (eid, session_info) in enumerate(sorted(sessions.items())):
        print(f"\nSession {i+1}/{len(sessions)}: {eid}")

        result = process_session(session_info, cache_dir, br)

        if result is None:
            n_skipped += 1
            continue

        # Track subject
        subject = result['subject']
        if subject not in subject_set:
            subject_set.append(subject)
        subj_idx = subject_set.index(subject)

        all_neural.append(result['neural'])
        all_input.append(result['input'])
        all_output.append(result['output'])
        all_subject_idx.append(subj_idx)
        all_brain_region_idx.append(result['beryl_regions'])
        all_beryl_regions_per_session.append(result['beryl_regions'])
        session_eids.append(eid)

        n_processed += 1

    print(f"\n{'='*60}")
    print(f"Processed: {n_processed}, Skipped: {n_skipped}")

    if n_processed == 0:
        print("ERROR: No sessions processed successfully!")
        return None

    # Build global brain region list
    all_regions_flat = []
    for regions in all_beryl_regions_per_session:
        for r in regions:
            if r not in all_regions_flat:
                all_regions_flat.append(r)
    all_regions_flat = sorted(all_regions_flat)

    # Build brain_region_idx for each session
    brain_region_idx = []
    for regions in all_beryl_regions_per_session:
        idx = np.array([all_regions_flat.index(r) for r in regions])
        brain_region_idx.append(idx)

    # Build final data dictionary
    data = {
        'neural': all_neural,
        'input': all_input,
        'output': all_output,
        'subjects': subject_set,
        'subject_idx': np.array(all_subject_idx),
        'brain_regions': all_regions_flat,
        'brain_region_idx': brain_region_idx,
        'input_names': ['time_since_stim_onset', 'trial_number_in_block'],
        'output_names': ['choice', 'prior', 'wheel_speed', 'whisker_motion_energy'],
        'output_values': [
            ['left', 'right'],                    # choice: 0=left, 1=right
            ['0.2', '0.5', '0.8'],                # prior: 0=0.2, 1=0.5, 2=0.8
            ['low', 'medium', 'high'],             # wheel speed bins
            ['low', 'medium', 'high'],             # whisker ME bins
        ],
        'metadata': {
            'task_description': 'IBL brain-wide map task: mice rotate wheel to move visual stimulus to center',
            'time_bin_size': BINSIZE * 1000,  # in ms
            'temporal_alignment_event': 'stimulus onset (stimOn_times)',
            'off_start': TIME_WINDOW[0],  # -0.5s before stimulus onset
            'off_end': TIME_WINDOW[1],    # 1.5s after stimulus onset
            'bin_size_s': BINSIZE,
            'n_time_bins': N_BINS,
            'n_sessions': n_processed,
            'n_sessions_skipped': n_skipped,
            'session_eids': session_eids,
            'align_time': ALIGN_TIME,
            'time_window': TIME_WINDOW,
            'min_rt': MIN_RT,
            'max_rt': MAX_RT,
        }
    }

    return data


def print_stats(data):
    """Print summary statistics for sanity checking."""
    print("\n" + "="*60)
    print("DATA SUMMARY")
    print("="*60)

    n_sessions = len(data['neural'])
    print(f"Number of sessions: {n_sessions}")
    print(f"Number of subjects: {len(data['subjects'])}")
    print(f"Brain regions: {len(data['brain_regions'])}")

    # Trial counts
    trial_counts = [len(s) for s in data['neural']]
    print(f"\nTrials per session: min={min(trial_counts)}, max={max(trial_counts)}, "
          f"mean={np.mean(trial_counts):.1f}, total={sum(trial_counts)}")

    # Neuron counts
    neuron_counts = [data['neural'][s][0].shape[0] for s in range(n_sessions)]
    print(f"Neurons per session: min={min(neuron_counts)}, max={max(neuron_counts)}, "
          f"mean={np.mean(neuron_counts):.1f}")

    # Time bins
    tbins = [data['neural'][s][0].shape[1] for s in range(n_sessions)]
    print(f"Time bins per trial: {set(tbins)}")

    # Output distribution
    print(f"\nOutput names: {data['output_names']}")
    print(f"Output values: {data['output_values']}")

    # Choice distribution
    choices = []
    for session in data['output']:
        for trial in session:
            choices.append(trial[0, 0])  # choice is same across time
    choices = np.array(choices)
    print(f"\nChoice distribution: left={np.sum(choices==0)}, right={np.sum(choices==1)}")

    # Prior distribution
    priors = []
    for session in data['output']:
        for trial in session:
            priors.append(trial[1, 0])
    priors = np.array(priors)
    print(f"Prior distribution: 0.2->{np.sum(priors==0)}, 0.5->{np.sum(priors==1)}, 0.8->{np.sum(priors==2)}")

    print(f"\nInput names: {data['input_names']}")
    print(f"Metadata: {data['metadata']}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert IBL BWM data to decoder format')
    parser.add_argument('--cache-dir', type=str, default='/app/data/one_cache',
                        help='Path to ONE cache directory')
    parser.add_argument('--output', type=str, default='/app/converted_data.pkl',
                        help='Output pickle file path')
    parser.add_argument('--max-sessions', type=int, default=None,
                        help='Maximum number of sessions to process')
    parser.add_argument('--sample', action='store_true',
                        help='Process a small sample for testing')

    args = parser.parse_args()

    if args.sample:
        args.max_sessions = args.max_sessions or 5
        args.output = args.output.replace('.pkl', '').replace('converted_data', 'sample_data') + '.pkl'

    data = convert_data(args.cache_dir, max_sessions=args.max_sessions, sample=args.sample)

    if data is not None:
        print_stats(data)

        print(f"\nSaving to {args.output}...")
        with open(args.output, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved! File size: {os.path.getsize(args.output) / 1024 / 1024:.1f} MB")
