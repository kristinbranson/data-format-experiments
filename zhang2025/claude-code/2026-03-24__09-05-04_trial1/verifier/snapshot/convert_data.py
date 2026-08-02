#!/usr/bin/env python3
"""
Convert IBL Brain-Wide Map data to decoder-compatible format.

Usage:
    python -u convert_data.py <outpicklefile> [--full|--sample] [--show-processing]
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
from scipy import signal
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# Add ibllib to path for brain atlas
sys.path.insert(0, '/app/code/ibllib')
from iblatlas.regions import BrainRegions

# ============================================================
# Constants
# ============================================================
BINSIZE = 0.02  # 20 ms time bins
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)  # 2s window around stimOn
N_BINS = int(np.ceil((TIME_WINDOW[1] - TIME_WINDOW[0]) / BINSIZE))  # 100
MIN_RT = 0.08
MAX_RT = 2.0
MAX_TRIAL_LEN = 10.0
WHEEL_FS = 1000  # Wheel interpolation frequency
WHEEL_CORNER_FREQ = 20
WHEEL_FILTER_ORDER = 8
N_DISCRETE_BINS = 3  # For discretizing wheel speed and whisker ME

DATA_DIR = Path('/app/data/one_cache')
BWM_CSV = Path('/app/code/code_zhang2025/data/bwm_release.csv')

NAN_EXCLUDE = ['stimOn_times', 'choice', 'feedback_times',
               'probabilityLeft', 'firstMovement_times', 'feedbackType']


def find_session_path(lab, subject, date, number=1):
    """Find the session directory in the ONE cache."""
    sess_path = DATA_DIR / lab / 'Subjects' / subject / date / f'{number:03d}' / 'alf'
    if sess_path.exists():
        return sess_path
    return None


def find_latest_revision(base_path, filename_pattern):
    """Find the latest revision of a file, checking revision directories first."""
    # Check revision directories (format: #YYYY-MM-DD#)
    candidates = []
    if base_path.exists():
        for item in base_path.iterdir():
            if item.is_dir() and item.name.startswith('#') and item.name.endswith('#'):
                for f in item.iterdir():
                    if filename_pattern in f.name:
                        candidates.append(f)
        # Also check base path directly
        for f in base_path.iterdir():
            if not f.is_dir() and filename_pattern in f.name:
                candidates.append(f)

    if not candidates:
        return None
    # Sort by revision date (newest first), files without revision come last
    def sort_key(p):
        parts = p.parts
        for part in parts:
            if part.startswith('#') and part.endswith('#'):
                return part
        return '#0000-00-00#'
    candidates.sort(key=sort_key, reverse=True)
    return candidates[0]


def load_trials(alf_path):
    """Load trials table from session."""
    trials_file = find_latest_revision(alf_path, '_ibl_trials.table.pqt')
    if trials_file is None:
        raise FileNotFoundError(f"No trials table found in {alf_path}")
    trials = pd.read_parquet(trials_file)
    return trials


def create_trial_mask(trials):
    """Create boolean mask for trial inclusion following reference code."""
    mask = pd.Series(True, index=trials.index)

    # Exclude NaN in key events
    for event in NAN_EXCLUDE:
        if event in trials.columns:
            mask &= ~trials[event].isna()

    # Reaction time filter
    rt = trials['firstMovement_times'] - trials['stimOn_times']
    mask &= (rt >= MIN_RT) & (rt <= MAX_RT)

    # Exclude no-choice trials
    mask &= (trials['choice'] != 0)

    # Max trial length filter
    if 'goCue_times' in trials.columns:
        trial_len = trials['feedback_times'] - trials['goCue_times']
        mask &= (trial_len <= MAX_TRIAL_LEN) | trial_len.isna()

    return mask


def load_spike_sorting(alf_path, probe_name):
    """Load spike sorting data for a probe."""
    probe_path = alf_path / probe_name / 'pykilosort'

    # Find latest revision
    spike_times_file = find_latest_revision(probe_path, 'spikes.times.npy')
    if spike_times_file is None:
        raise FileNotFoundError(f"No spike times found in {probe_path}")

    rev_dir = spike_times_file.parent

    spikes = {
        'times': np.load(rev_dir / 'spikes.times.npy').flatten(),
        'clusters': np.load(rev_dir / 'spikes.clusters.npy').flatten(),
    }

    # Load cluster info
    clusters_channels = np.load(rev_dir / 'clusters.channels.npy').flatten()
    clusters_depths = np.load(rev_dir / 'clusters.depths.npy').flatten()

    # Load cluster metrics
    metrics_file = rev_dir / 'clusters.metrics.pqt'
    if metrics_file.exists():
        metrics = pd.read_parquet(metrics_file)
    else:
        metrics = None

    # Load channel brain region IDs
    chan_brain_ids = np.load(rev_dir / 'channels.brainLocationIds_ccf_2017.npy').flatten()

    clusters = {
        'channels': clusters_channels,
        'depths': clusters_depths,
        'metrics': metrics,
        'chan_brain_ids': chan_brain_ids,
    }

    return spikes, clusters


def merge_probes(spikes_list, clusters_list):
    """Merge spike data from multiple probes."""
    if len(spikes_list) == 1:
        return spikes_list[0], clusters_list[0]

    merged_spikes_times = []
    merged_spikes_clusters = []
    merged_channels = []
    merged_depths = []
    merged_chan_brain_ids_list = []
    merged_metrics_list = []

    cluster_offset = 0
    chan_offset = 0

    for spikes, clusters in zip(spikes_list, clusters_list):
        n_clusters = len(clusters['channels'])

        merged_spikes_times.append(spikes['times'])
        merged_spikes_clusters.append(spikes['clusters'] + cluster_offset)
        merged_channels.append(clusters['channels'] + chan_offset)
        merged_depths.append(clusters['depths'])
        merged_chan_brain_ids_list.append(clusters['chan_brain_ids'])

        if clusters['metrics'] is not None:
            m = clusters['metrics'].copy()
            if 'cluster_id' in m.columns:
                m['cluster_id'] += cluster_offset
            merged_metrics_list.append(m)

        cluster_offset += n_clusters
        chan_offset += len(clusters['chan_brain_ids'])

    # Concatenate
    all_times = np.concatenate(merged_spikes_times)
    all_clusters = np.concatenate(merged_spikes_clusters)

    # Sort by time
    sort_idx = np.argsort(all_times, kind='stable')

    merged_spikes = {
        'times': all_times[sort_idx],
        'clusters': all_clusters[sort_idx],
    }

    all_chan_brain_ids = np.concatenate(merged_chan_brain_ids_list)

    merged_clusters = {
        'channels': np.concatenate(merged_channels),
        'depths': np.concatenate(merged_depths),
        'chan_brain_ids': all_chan_brain_ids,
        'metrics': pd.concat(merged_metrics_list, ignore_index=True) if merged_metrics_list else None,
    }

    return merged_spikes, merged_clusters


def get_brain_regions(clusters, br):
    """Get Beryl brain region acronyms for each cluster."""
    chan_ids = clusters['chan_brain_ids']
    chan_acronyms = br.id2acronym(chan_ids)
    chan_beryl = br.acronym2acronym(chan_acronyms, mapping='Beryl')

    # Map cluster to region via channel assignment
    cluster_regions = chan_beryl[clusters['channels']]
    return cluster_regions


def bin_spikes_vectorized(spike_times, spike_clusters, n_clusters, interval_starts, interval_ends, binsize, n_bins):
    """Bin spikes into trial-aligned time windows. Optimized implementation."""
    n_trials = len(interval_starts)
    binned = np.zeros((n_trials, n_clusters, n_bins), dtype=np.float32)

    # Pre-sort spikes are already time-sorted after merge
    for trial_idx in range(n_trials):
        t_start = interval_starts[trial_idx]
        t_end = interval_ends[trial_idx]

        if np.isnan(t_start) or np.isnan(t_end):
            continue

        # Use searchsorted for fast interval selection
        idx_beg = np.searchsorted(spike_times, t_start, side='left')
        idx_end = np.searchsorted(spike_times, t_end, side='left')

        if idx_beg >= idx_end:
            continue

        t_sel = spike_times[idx_beg:idx_end]
        c_sel = spike_clusters[idx_beg:idx_end]

        # Compute bin indices
        bin_idx = np.minimum(((t_sel - t_start) / binsize).astype(np.int32), n_bins - 1)

        # Use flat indexing for efficient accumulation
        valid = (c_sel >= 0) & (c_sel < n_clusters)
        flat_idx = c_sel[valid] * n_bins + bin_idx[valid]
        np.add.at(binned[trial_idx].ravel(), flat_idx, 1)

    return binned


def interpolate_wheel(timestamps, position, fs=WHEEL_FS, corner_freq=WHEEL_CORNER_FREQ, order=WHEEL_FILTER_ORDER):
    """Interpolate wheel position and compute velocity (matching ibllib)."""
    t = np.arange(timestamps[0], timestamps[-1], 1.0 / fs)
    if len(t) > 0 and t[-1] > timestamps[-1]:
        t = t[:-1]

    pos_interp = interp1d(timestamps, position, kind='linear')(t)

    # Butterworth low-pass filter for velocity
    sos = signal.butter(N=order, Wn=corner_freq / fs * 2, btype='lowpass', output='sos')
    vel = np.insert(np.diff(signal.sosfiltfilt(sos, pos_interp)), 0, 0) * fs

    return t, np.abs(vel).astype(np.float32)  # wheel speed = |velocity|


def load_motion_energy(alf_path, side='left'):
    """Load whisker motion energy from camera data."""
    if side == 'left':
        me_file = find_latest_revision(alf_path, 'leftCamera.ROIMotionEnergy.npy')
        times_file = find_latest_revision(alf_path, '_ibl_leftCamera.times.npy')
    else:
        me_file = find_latest_revision(alf_path, 'rightCamera.ROIMotionEnergy.npy')
        times_file = find_latest_revision(alf_path, '_ibl_rightCamera.times.npy')

    if me_file is None or times_file is None:
        return None, None

    me = np.load(me_file).flatten()
    times = np.load(times_file).flatten()

    # Handle length mismatch (common in IBL data)
    min_len = min(len(me), len(times))
    return times[:min_len], me[:min_len].astype(np.float32)


def interpolate_behavior_to_bins(beh_times, beh_values, interval_starts, interval_ends, binsize, n_bins):
    """Interpolate behavior signal to trial-aligned bins (matching reference code)."""
    n_trials = len(interval_starts)
    result = np.full((n_trials, n_bins), np.nan, dtype=np.float32)
    good_mask = np.zeros(n_trials, dtype=bool)

    for trial_idx in range(n_trials):
        t_start = interval_starts[trial_idx]
        t_end = interval_ends[trial_idx]

        if np.isnan(t_start) or np.isnan(t_end):
            continue

        # Find behavior data in this interval
        idx_beg = np.searchsorted(beh_times, t_start, side='right')
        idx_end = np.searchsorted(beh_times, t_end, side='left')

        t_sel = beh_times[idx_beg:idx_end]
        v_sel = beh_values[idx_beg:idx_end]

        if len(v_sel) == 0:
            continue

        # Check data availability (matching reference code)
        if np.abs(t_start - t_sel[0]) > binsize:
            continue
        if np.abs(t_end - t_sel[-1]) > binsize:
            continue

        # Interpolate to uniform bins (matching reference code)
        x_interp = np.linspace(t_start + binsize, t_end, n_bins)
        try:
            y_interp = interp1d(t_sel, v_sel, kind='linear', fill_value='extrapolate')(x_interp)
            result[trial_idx] = y_interp
            good_mask[trial_idx] = True
        except Exception:
            continue

    return result, good_mask


def discretize_to_bins(values, n_bins=N_DISCRETE_BINS):
    """Discretize continuous values into n_bins equal-frequency bins using quantiles.
    Computed across all valid (non-NaN) values."""
    valid = values[~np.isnan(values)]
    if len(valid) == 0:
        return np.zeros_like(values, dtype=int), np.array([])

    # Compute quantile boundaries
    quantiles = np.linspace(0, 100, n_bins + 1)
    boundaries = np.percentile(valid, quantiles)

    # Digitize - np.digitize returns values 0..n_bins, clip to 0..n_bins-1
    result = np.digitize(values, boundaries[1:-1])  # 0 to n_bins-1
    result = np.clip(result, 0, n_bins - 1)

    return result.astype(int), boundaries


def compute_trial_in_block(prob_left):
    """Compute trial number within each block.
    Block boundaries are where probabilityLeft changes."""
    trial_in_block = np.zeros(len(prob_left), dtype=np.float32)
    count = 1
    for i in range(len(prob_left)):
        if i > 0 and prob_left[i] != prob_left[i-1]:
            count = 1
        trial_in_block[i] = count
        count += 1
    return trial_in_block


def process_session(session_info, br, show_processing=False):
    """Process a single session and return data dict."""
    eid = session_info['eid']
    subject = session_info['subject']
    lab = session_info['lab']
    date = session_info['date']

    t0 = time.time()

    # Find session path
    alf_path = find_session_path(lab, subject, date)
    if alf_path is None:
        print(f"  Session {eid} ({subject}/{date}): path not found, skipping")
        return None

    # Load trials
    try:
        trials = load_trials(alf_path)
    except Exception as e:
        print(f"  Session {eid}: error loading trials: {e}")
        return None

    # Create trial mask
    mask = create_trial_mask(trials)

    # Find probes
    probe_dirs = sorted([d.name for d in alf_path.iterdir() if d.is_dir() and d.name.startswith('probe')])
    if not probe_dirs:
        print(f"  Session {eid}: no probes found, skipping")
        return None

    # Load and merge spike sorting from all probes
    spikes_list = []
    clusters_list = []
    for probe_name in probe_dirs:
        try:
            spk, clu = load_spike_sorting(alf_path, probe_name)
            spikes_list.append(spk)
            clusters_list.append(clu)
        except Exception as e:
            print(f"  Session {eid}, {probe_name}: error loading spikes: {e}")
            continue

    if not spikes_list:
        print(f"  Session {eid}: no valid probes, skipping")
        return None

    spikes, clusters = merge_probes(spikes_list, clusters_list)
    n_clusters = len(clusters['channels'])

    # Get brain regions
    cluster_regions = get_brain_regions(clusters, br)

    # Apply trial mask
    valid_trials = trials[mask].reset_index(drop=True)

    if len(valid_trials) < 2:
        print(f"  Session {eid}: fewer than 2 valid trials, skipping")
        return None

    # Compute intervals aligned to stimOn_times
    align_times = valid_trials[ALIGN_TIME].values
    interval_starts = align_times + TIME_WINDOW[0]
    interval_ends = align_times + TIME_WINDOW[1]

    # Bin spikes
    binned_spikes = bin_spikes_vectorized(
        spikes['times'], spikes['clusters'], n_clusters,
        interval_starts, interval_ends, BINSIZE, N_BINS
    )
    # binned_spikes shape: (n_trials, n_clusters, n_bins)

    # Load wheel data
    wheel_speed_binned = None
    try:
        wheel_pos_raw = np.load(find_latest_revision(alf_path, '_ibl_wheel.position.npy')).flatten()
        wheel_ts_raw = np.load(find_latest_revision(alf_path, '_ibl_wheel.timestamps.npy')).flatten()
        wheel_times, wheel_speed = interpolate_wheel(wheel_ts_raw, wheel_pos_raw)
        wheel_speed_binned, wheel_good = interpolate_behavior_to_bins(
            wheel_times, wheel_speed, interval_starts, interval_ends, BINSIZE, N_BINS
        )
    except Exception as e:
        print(f"  Session {eid}: error loading wheel: {e}")

    # Load whisker motion energy
    me_binned = None
    try:
        me_times, me_values = load_motion_energy(alf_path, side='left')
        if me_times is None:
            me_times, me_values = load_motion_energy(alf_path, side='right')
        if me_times is not None:
            me_binned, me_good = interpolate_behavior_to_bins(
                me_times, me_values, interval_starts, interval_ends, BINSIZE, N_BINS
            )
    except Exception as e:
        print(f"  Session {eid}: error loading motion energy: {e}")

    # Build combined mask (trials where all data is available)
    combined_mask = np.ones(len(valid_trials), dtype=bool)
    if wheel_speed_binned is not None:
        combined_mask &= ~np.any(np.isnan(wheel_speed_binned), axis=1)
    else:
        # If no wheel data, mark all as bad
        combined_mask[:] = False

    if me_binned is not None:
        combined_mask &= ~np.any(np.isnan(me_binned), axis=1)
    else:
        combined_mask[:] = False

    # Apply combined mask
    n_valid = combined_mask.sum()
    if n_valid < 2:
        print(f"  Session {eid}: fewer than 2 trials with all data, skipping")
        return None

    final_spikes = binned_spikes[combined_mask]  # (n_trials, n_clusters, n_bins)
    final_wheel = wheel_speed_binned[combined_mask]  # (n_trials, n_bins)
    final_me = me_binned[combined_mask]  # (n_trials, n_bins)
    final_trials = valid_trials[combined_mask].reset_index(drop=True)

    # Build neural data: list of (n_neurons, n_timepoints) per trial
    # Use uint8 to save memory (spike counts per 20ms bin are small integers, max ~12)
    neural_trials = []
    for t in range(len(final_spikes)):
        neural_trials.append(final_spikes[t].astype(np.uint8))  # (n_clusters, n_bins)

    # Build inputs
    # Input 0: time since stimulus onset (continuous, time-varying)
    time_since_stim = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS).astype(np.float32)

    # Input 1: trial number in block (continuous, per-trial)
    prob_left_all = trials['probabilityLeft'].values
    trial_in_block_all = compute_trial_in_block(prob_left_all)
    # Apply masks to get trial_in_block for valid+combined trials
    trial_in_block_valid = trial_in_block_all[mask.values]
    trial_in_block_final = trial_in_block_valid[combined_mask]

    input_trials = []
    for t in range(len(final_trials)):
        inp = np.array([
            time_since_stim,  # time-varying
            np.full(N_BINS, trial_in_block_final[t], dtype=np.float32),  # per-trial, broadcast to time
        ], dtype=np.float32)  # (2, n_bins)
        input_trials.append(inp)

    # Build outputs
    # Output 0: choice (binary, per-trial) - left=0, right=1
    choice = final_trials['choice'].values.copy()
    # IBL: -1=left, 1=right -> convert to 0=left, 1=right
    choice_encoded = ((choice + 1) // 2).astype(int)  # -1->0, 1->1

    # Output 1: prior probability of left (per-trial, categorical)
    prob_left = final_trials['probabilityLeft'].values
    prior_encoded = np.zeros(len(prob_left), dtype=int)
    prior_encoded[prob_left == 0.2] = 0
    prior_encoded[prob_left == 0.5] = 1
    prior_encoded[prob_left == 0.8] = 2

    # Output 2: wheel speed discretized into 3 bins (time-varying)
    # Discretize using quantiles across ALL timepoints in this session
    all_wheel_flat = final_wheel.flatten()
    wheel_disc, wheel_boundaries = discretize_to_bins(all_wheel_flat, N_DISCRETE_BINS)
    wheel_disc_2d = wheel_disc.reshape(final_wheel.shape).astype(int)

    # Output 3: whisker motion energy discretized into 3 bins (time-varying)
    all_me_flat = final_me.flatten()
    me_disc, me_boundaries = discretize_to_bins(all_me_flat, N_DISCRETE_BINS)
    me_disc_2d = me_disc.reshape(final_me.shape).astype(int)

    output_trials = []
    for t in range(len(final_trials)):
        out = np.array([
            np.full(N_BINS, choice_encoded[t], dtype=int),  # per-trial, broadcast
            np.full(N_BINS, prior_encoded[t], dtype=int),  # per-trial, broadcast
            wheel_disc_2d[t],  # time-varying
            me_disc_2d[t],  # time-varying
        ], dtype=int)  # (4, n_bins)
        output_trials.append(out)

    elapsed = time.time() - t0
    print(f"  Session {eid} ({subject}/{date}): {n_valid} trials, {n_clusters} neurons, {elapsed:.1f}s")

    # Processing visualization
    if show_processing:
        try:
            plot_processing(eid, subject, date, final_spikes, final_wheel, final_me,
                          wheel_disc_2d, me_disc_2d, choice_encoded, prior_encoded,
                          time_since_stim, cluster_regions)
        except Exception as e:
            print(f"  Warning: could not create processing plots: {e}")

    return {
        'neural': neural_trials,
        'input': input_trials,
        'output': output_trials,
        'subject': subject,
        'cluster_regions': cluster_regions,
        'n_clusters': n_clusters,
        'n_trials': n_valid,
        'eid': eid,
    }


def plot_processing(eid, subject, date, spikes, wheel, me, wheel_disc, me_disc,
                   choice, prior, time_axis, cluster_regions):
    """Plot processing steps for visual verification."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 3, figsize=(18, 16))
    fig.suptitle(f'Processing: {subject}/{date} ({eid[:8]}...)', fontsize=14)

    # Trial 0 and 1
    for col, trial_idx in enumerate([0, min(1, len(spikes)-1)]):
        # Neural activity (first 50 neurons)
        n_show = min(50, spikes.shape[1])
        axes[0, col].imshow(spikes[trial_idx, :n_show, :], aspect='auto', cmap='hot',
                           extent=[time_axis[0], time_axis[-1], n_show, 0])
        axes[0, col].set_title(f'Trial {trial_idx}: Neural ({n_show}/{spikes.shape[1]} neurons)')
        axes[0, col].set_xlabel('Time from stimOn (s)')

        # Wheel speed (raw vs discretized)
        axes[1, col].plot(time_axis, wheel[trial_idx], 'b-', alpha=0.5, label='Raw wheel speed')
        axes[1, col].plot(time_axis, wheel_disc[trial_idx], 'r-', label='Discretized')
        axes[1, col].set_title(f'Trial {trial_idx}: Wheel speed')
        axes[1, col].legend(fontsize=8)

        # Whisker ME (raw vs discretized)
        axes[2, col].plot(time_axis, me[trial_idx], 'b-', alpha=0.5, label='Raw ME')
        axes[2, col].plot(time_axis, me_disc[trial_idx], 'r-', label='Discretized')
        axes[2, col].set_title(f'Trial {trial_idx}: Whisker ME')
        axes[2, col].legend(fontsize=8)

    # Summary statistics
    # Choice distribution
    axes[0, 2].bar(['Left (0)', 'Right (1)'],
                   [np.sum(choice == 0), np.sum(choice == 1)])
    axes[0, 2].set_title('Choice distribution')

    # Prior distribution
    axes[1, 2].bar(['0.2 (0)', '0.5 (1)', '0.8 (2)'],
                   [np.sum(prior == 0), np.sum(prior == 1), np.sum(prior == 2)])
    axes[1, 2].set_title('Prior distribution')

    # Wheel speed histogram
    axes[2, 2].hist(wheel.flatten(), bins=50, alpha=0.7)
    axes[2, 2].set_title('Wheel speed distribution')

    # Brain regions
    unique_regions, counts = np.unique(cluster_regions, return_counts=True)
    sort_idx = np.argsort(-counts)[:15]  # Top 15
    axes[3, 0].barh(range(len(sort_idx)), counts[sort_idx])
    axes[3, 0].set_yticks(range(len(sort_idx)))
    axes[3, 0].set_yticklabels(unique_regions[sort_idx], fontsize=7)
    axes[3, 0].set_title('Top brain regions')

    # ME distribution
    axes[3, 1].hist(me.flatten(), bins=50, alpha=0.7)
    axes[3, 1].set_title('Whisker ME distribution')

    # Mean neural activity across trials
    axes[3, 2].plot(time_axis, spikes.mean(axis=(0, 1)), 'k-')
    axes[3, 2].set_title('Mean firing rate')
    axes[3, 2].set_xlabel('Time from stimOn (s)')

    fig.tight_layout()
    safe_eid = eid[:8]
    fig.savefig(f'/app/processing_{safe_eid}.png', dpi=150)
    plt.close(fig)
    print(f"  Saved processing plot: processing_{safe_eid}.png")


def get_session_list(bwm_df):
    """Get unique sessions from BWM release CSV."""
    # Group by eid to get one row per session
    sessions = bwm_df.groupby('eid').first().reset_index()
    return sessions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=str, help='Output pickle file path')
    parser.add_argument('--full', action='store_true', default=True, help='Process all sessions (default)')
    parser.add_argument('--sample', action='store_true', help='Process only 2 sessions for testing')
    parser.add_argument('--show-processing', action='store_true', help='Plot processing visualizations')
    args = parser.parse_args()

    if args.sample:
        args.full = False

    print("=" * 60)
    print("IBL Brain-Wide Map Data Conversion")
    print("=" * 60)

    t_start = time.time()

    # Load BWM release info
    bwm_df = pd.read_csv(BWM_CSV, index_col=0)
    sessions = get_session_list(bwm_df)
    print(f"Total sessions in BWM release: {len(sessions)}")

    if args.sample:
        # Pick 2 sessions for testing
        np.random.seed(42)
        sample_idx = np.random.choice(len(sessions), min(2, len(sessions)), replace=False)
        sessions = sessions.iloc[sample_idx].reset_index(drop=True)
        print(f"Sample mode: processing {len(sessions)} sessions")

    # Initialize brain atlas
    br = BrainRegions()

    # Process sessions
    all_results = []
    for idx in range(len(sessions)):
        sess = sessions.iloc[idx]
        print(f"\nProcessing session {idx+1}/{len(sessions)}: {sess['subject']}/{sess['date']}")
        result = process_session(sess, br, show_processing=args.show_processing)
        if result is not None:
            all_results.append(result)

        if (idx + 1) % 10 == 0:
            elapsed = time.time() - t_start
            rate = elapsed / (idx + 1)
            remaining = rate * (len(sessions) - idx - 1)
            print(f"\n  Progress: {idx+1}/{len(sessions)} sessions, "
                  f"{elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining")

    print(f"\n{'='*60}")
    print(f"Successfully processed {len(all_results)} / {len(sessions)} sessions")

    if len(all_results) == 0:
        print("ERROR: No sessions processed successfully!")
        sys.exit(1)

    # Build final data structure
    print("\nBuilding final data structure...")

    # Collect all unique subjects
    all_subjects = sorted(set(r['subject'] for r in all_results))
    subject_to_idx = {s: i for i, s in enumerate(all_subjects)}

    # Collect all unique brain regions
    all_regions_set = set()
    for r in all_results:
        all_regions_set.update(r['cluster_regions'])
    all_regions = sorted(all_regions_set)
    region_to_idx = {r: i for i, r in enumerate(all_regions)}

    # Build lists
    neural_list = []
    input_list = []
    output_list = []
    subject_idx = []
    brain_region_idx = []

    for r in all_results:
        neural_list.append(r['neural'])
        input_list.append(r['input'])
        output_list.append(r['output'])
        subject_idx.append(subject_to_idx[r['subject']])
        brain_region_idx.append(np.array([region_to_idx[reg] for reg in r['cluster_regions']]))

    data = {
        'neural': neural_list,
        'input': input_list,
        'output': output_list,
        'subjects': all_subjects,
        'subject_idx': np.array(subject_idx),
        'brain_regions': all_regions,
        'brain_region_idx': brain_region_idx,
        'input_names': ['time_since_stimulus_onset', 'trial_number_in_block'],
        'output_names': ['choice', 'prior_probability_left', 'wheel_speed', 'whisker_motion_energy'],
        'output_values': [
            ['left', 'right'],                  # choice: 0=left, 1=right
            ['0.2', '0.5', '0.8'],              # prior: 0=0.2, 1=0.5, 2=0.8
            ['low', 'medium', 'high'],           # wheel speed: 3 quantile bins
            ['low', 'medium', 'high'],           # whisker ME: 3 quantile bins
        ],
        'metadata': {
            'task_description': 'IBL decision-making task: mice rotate wheel to indicate location of visual stimulus',
            'time_bin_size': BINSIZE * 1000,  # in ms
            'temporal_alignment_event': 'stimulus onset (stimOn_times)',
            'off_start': TIME_WINDOW[0],
            'off_end': TIME_WINDOW[1],
            'n_timepoints': N_BINS,
            'dataset': 'IBL Brain-Wide Map',
            'n_sessions': len(all_results),
            'n_subjects': len(all_subjects),
            'total_trials': sum(r['n_trials'] for r in all_results),
            'total_neurons': sum(r['n_clusters'] for r in all_results),
        }
    }

    # Print summary
    print(f"\nDataset Summary:")
    print(f"  Sessions: {len(all_results)}")
    print(f"  Subjects: {len(all_subjects)}")
    print(f"  Total trials: {data['metadata']['total_trials']}")
    print(f"  Total neurons: {data['metadata']['total_neurons']}")
    print(f"  Brain regions: {len(all_regions)}")
    print(f"  Time bins: {N_BINS} ({BINSIZE*1000:.0f} ms each)")
    print(f"  Inputs: {data['input_names']}")
    print(f"  Outputs: {data['output_names']}")

    # Save
    print(f"\nSaving to {args.output}...")
    with open(args.output, 'wb') as f:
        pickle.dump(data, f, protocol=4)

    file_size = os.path.getsize(args.output) / (1024**2)
    print(f"Saved {args.output} ({file_size:.1f} MB)")

    total_time = time.time() - t_start
    print(f"\nTotal time: {total_time:.1f}s ({total_time/60:.1f} min)")


if __name__ == '__main__':
    main()
