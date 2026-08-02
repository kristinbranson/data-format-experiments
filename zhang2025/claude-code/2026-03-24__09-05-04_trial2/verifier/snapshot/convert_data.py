#!/usr/bin/env python3
"""
Convert IBL Brain-wide Map data to decoder-compatible format.

Usage:
    python -u convert_data.py <outpicklefile> [--sample] [--full] [--show-processing]

Follows processing from Zhang et al. (2025) code (code_zhang2025/src/0_data_caching.py):
- Align to stimOn_times, window (-0.5, 1.5)s
- 20ms bins -> 100 time steps per trial
- All neurons (no QC filtering)
- Trial filtering: RT 0.08-2.0s, exclude no-choice, exclude NaN key events
"""

import sys
import os
import argparse
import time
import pickle
import glob
import warnings
from pathlib import Path
from collections import defaultdict
import gc

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from iblatlas.regions import BrainRegions

warnings.filterwarnings('ignore')

# ============================================================
# Constants matching reference code
# ============================================================
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)  # 2s trial
BINSIZE = 0.02  # 20ms
INTERVAL_LEN = TIME_WINDOW[1] - TIME_WINDOW[0]  # 2.0s
N_BINS = int(np.ceil(INTERVAL_LEN / BINSIZE))  # 100

MIN_RT = 0.08
MAX_RT = 2.0
MAX_TRIAL_LEN = 10.0

NAN_EXCLUDE = [
    'stimOn_times', 'choice', 'feedback_times',
    'probabilityLeft', 'firstMovement_times', 'feedbackType'
]

DATA_ROOT = Path('data/one_cache')


# ============================================================
# Helper functions
# ============================================================

def find_session_dirs():
    """Find all session directories with required data."""
    session_dirs = sorted(glob.glob(str(DATA_ROOT / '*/Subjects/*/*/001')))
    valid = []
    for sdir in session_dirs:
        has_spikes = len(glob.glob(os.path.join(sdir, 'alf/probe*/pykilosort/*/spikes.times.npy'))) > 0
        has_trials = len(glob.glob(os.path.join(sdir, 'alf/*/_ibl_trials.table.pqt'))) > 0
        has_wheel = os.path.exists(os.path.join(sdir, 'alf/_ibl_wheel.timestamps.npy'))
        # Check for ME files in dated subdirectories (alf/#date#/)
        has_me = (len(glob.glob(os.path.join(sdir, 'alf/*/leftCamera.ROIMotionEnergy.npy'))) > 0 or
                  len(glob.glob(os.path.join(sdir, 'alf/*/rightCamera.ROIMotionEnergy.npy'))) > 0)
        # Check for camera times in alf/ or alf/#date#/
        has_me_times = (os.path.exists(os.path.join(sdir, 'alf/_ibl_leftCamera.times.npy')) or
                       os.path.exists(os.path.join(sdir, 'alf/_ibl_rightCamera.times.npy')) or
                       len(glob.glob(os.path.join(sdir, 'alf/*/_ibl_leftCamera.times.npy'))) > 0 or
                       len(glob.glob(os.path.join(sdir, 'alf/*/_ibl_rightCamera.times.npy'))) > 0)
        has_me = has_me and has_me_times
        if has_spikes and has_trials and has_wheel and has_me:
            valid.append(sdir)
    return valid


def parse_session_info(sdir):
    """Extract lab, subject, date from session directory path."""
    parts = Path(sdir).parts
    sub_idx = parts.index('Subjects')
    lab = parts[sub_idx - 1]
    subject = parts[sub_idx + 1]
    date = parts[sub_idx + 2]
    return lab, subject, date


def load_trials(sdir):
    """Load trials table from parquet file."""
    trial_files = sorted(glob.glob(os.path.join(sdir, 'alf/*/_ibl_trials.table.pqt')))
    if not trial_files:
        raise FileNotFoundError(f"No trials table found in {sdir}")
    # Use the most recent revision
    trials = pd.read_parquet(trial_files[-1])
    return trials


def create_trial_mask(trials):
    """Create trial mask following reference code load_trials_and_mask.

    Excludes:
    - RT < 0.08 or RT > 2.0
    - No-choice trials (choice == 0)
    - Trials with NaN in key events
    - Trial length > 10s
    """
    mask = pd.Series(True, index=trials.index)

    # RT filter
    rt = trials['firstMovement_times'] - trials['stimOn_times']
    mask &= (rt >= MIN_RT) & (rt <= MAX_RT)

    # Trial length filter (goCue to feedback)
    if 'goCue_times' in trials.columns and 'feedback_times' in trials.columns:
        trial_len = trials['feedback_times'] - trials['goCue_times']
        mask &= (trial_len <= MAX_TRIAL_LEN) | trial_len.isna()

    # No-choice exclusion
    mask &= (trials['choice'] != 0)

    # NaN exclusion
    for col in NAN_EXCLUDE:
        if col in trials.columns:
            mask &= ~trials[col].isna()

    return mask


def load_spikes(sdir):
    """Load and merge spikes from all probes in a session.

    Following reference code: no QC filtering (qc=None).
    """
    probe_dirs = sorted(glob.glob(os.path.join(sdir, 'alf/probe*/pykilosort/*')))

    all_spike_times = []
    all_spike_clusters = []
    all_cluster_regions = []
    cluster_offset = 0

    br = BrainRegions()

    for pdir in probe_dirs:
        st_file = os.path.join(pdir, 'spikes.times.npy')
        sc_file = os.path.join(pdir, 'spikes.clusters.npy')
        cc_file = os.path.join(pdir, 'clusters.channels.npy')
        cb_file = os.path.join(pdir, 'channels.brainLocationIds_ccf_2017.npy')

        if not all(os.path.exists(f) for f in [st_file, sc_file, cc_file, cb_file]):
            continue

        spike_times = np.load(st_file).flatten()
        spike_clusters = np.load(sc_file).flatten()
        cluster_channels = np.load(cc_file).flatten()
        channel_brain_ids = np.load(cb_file).flatten()

        n_clusters = len(cluster_channels)

        # Map clusters to brain regions
        # Clip channel indices to valid range
        valid_channels = np.clip(cluster_channels, 0, len(channel_brain_ids) - 1)
        cluster_brain_ids = channel_brain_ids[valid_channels]
        cluster_acronyms = br.id2acronym(cluster_brain_ids)
        beryl_regions = br.acronym2acronym(cluster_acronyms, mapping='Beryl')

        # Offset cluster IDs for merging
        spike_clusters_offset = spike_clusters + cluster_offset
        cluster_offset += n_clusters

        all_spike_times.append(spike_times)
        all_spike_clusters.append(spike_clusters_offset)
        all_cluster_regions.extend(beryl_regions)

    if not all_spike_times:
        raise ValueError(f"No spike data found in {sdir}")

    # Merge and sort by time
    merged_times = np.concatenate(all_spike_times)
    del all_spike_times
    merged_clusters = np.concatenate(all_spike_clusters)
    del all_spike_clusters
    sort_idx = np.argsort(merged_times, kind='stable')
    merged_times = merged_times[sort_idx]
    merged_clusters = merged_clusters[sort_idx].astype(np.int32)
    del sort_idx

    return merged_times, merged_clusters, np.array(all_cluster_regions)


def bin_spikes_vectorized(spike_times, spike_clusters, n_clusters, interval_begs, interval_ends):
    """Bin spikes into time bins for all trials.

    Returns array of shape (n_trials, n_clusters, n_bins).
    Matches reference code bin_spiking_data / get_spike_data_per_interval.

    Uses searchsorted for fast trial assignment and flat indexing for vectorized counting.
    """
    n_trials = len(interval_begs)
    binned = np.zeros((n_trials, n_clusters, N_BINS), dtype=np.float32)

    # Process each trial using searchsorted for fast spike selection
    for trial_idx in range(n_trials):
        t_beg = interval_begs[trial_idx]
        t_end = interval_ends[trial_idx]

        if np.isnan(t_beg) or np.isnan(t_end):
            continue

        # Use searchsorted on sorted spike_times for O(log n) lookup
        i_start = np.searchsorted(spike_times, t_beg, side='left')
        i_end = np.searchsorted(spike_times, t_end, side='left')

        if i_start >= i_end:
            continue

        times_trial = spike_times[i_start:i_end]
        clusters_trial = spike_clusters[i_start:i_end]

        # Compute bin indices for each spike
        bin_idx = np.minimum(
            ((times_trial - t_beg) / BINSIZE).astype(np.int32),
            N_BINS - 1
        )

        # Use flat index: (cluster * N_BINS + bin) and bincount
        flat_idx = clusters_trial * N_BINS + bin_idx
        counts = np.bincount(flat_idx, minlength=n_clusters * N_BINS)
        binned[trial_idx] = counts[:n_clusters * N_BINS].reshape(n_clusters, N_BINS)

    return binned


def load_wheel_speed(sdir):
    """Load wheel speed (absolute velocity) following reference code.

    Reference: load_target_behavior for 'wheel-speed' uses SessionLoader.load_wheel()
    which returns interpolated wheel with velocity computed via Gaussian smoothing.
    We need to replicate this from raw position + timestamps.
    """
    wh_pos = np.load(os.path.join(sdir, 'alf/_ibl_wheel.position.npy')).flatten()
    wh_times = np.load(os.path.join(sdir, 'alf/_ibl_wheel.timestamps.npy')).flatten()

    # The reference code uses SessionLoader.load_wheel() which interpolates to 1kHz
    # and computes velocity via Gaussian smoothing. We replicate this:
    # Interpolate position to uniform 1kHz sampling
    dt = 0.001  # 1kHz
    t_uniform = np.arange(wh_times[0], wh_times[-1], dt)
    pos_interp = np.interp(t_uniform, wh_times, wh_pos)

    # Compute velocity via finite differences (matching brainbox wheel processing)
    velocity = np.gradient(pos_interp, dt)
    speed = np.abs(velocity)

    return t_uniform, speed


def load_whisker_me(sdir):
    """Load whisker motion energy. Try left camera first, then right.

    Matches reference code bin_behaviors / load_target_behavior for 'whisker-motion-energy'.
    """
    def _find_files(sdir, me_pattern, time_pattern):
        """Search for ME and time files in dated subdirs (alf/#date#/) and alf/ root."""
        me_files = sorted(glob.glob(os.path.join(sdir, 'alf', '*', me_pattern)))
        time_files = sorted(glob.glob(os.path.join(sdir, 'alf', '*', time_pattern)))
        # Also check directly in alf/
        direct_time = os.path.join(sdir, 'alf', time_pattern)
        if os.path.exists(direct_time) and direct_time not in time_files:
            time_files.append(direct_time)
        return me_files, sorted(time_files)

    # Try left camera first (matching reference code)
    left_me_files, left_time_files = _find_files(
        sdir, 'leftCamera.ROIMotionEnergy.npy', '_ibl_leftCamera.times.npy')

    if left_me_files and left_time_files:
        me = np.load(left_me_files[-1]).flatten()
        times = np.load(left_time_files[-1]).flatten()
        min_len = min(len(me), len(times))
        return times[:min_len], me[:min_len]

    # Fall back to right camera
    right_me_files, right_time_files = _find_files(
        sdir, 'rightCamera.ROIMotionEnergy.npy', '_ibl_rightCamera.times.npy')

    if right_me_files and right_time_files:
        me = np.load(right_me_files[-1]).flatten()
        times = np.load(right_time_files[-1]).flatten()
        min_len = min(len(me), len(times))
        return times[:min_len], me[:min_len]

    raise FileNotFoundError(f"No whisker motion energy data found in {sdir}")


def interpolate_behavior_to_bins(beh_times, beh_vals, interval_begs, interval_ends):
    """Interpolate behavioral signal to trial time bins.

    Matches reference code get_behavior_per_interval:
    - x_interp = linspace(interval_beg + binsize, interval_end, n_bins)
    - Interpolate using linear interp1d with extrapolation

    Returns:
        values: array (n_trials, n_bins)
        mask: boolean array (n_trials,) - True for good trials
    """
    n_trials = len(interval_begs)
    values = np.full((n_trials, N_BINS), np.nan, dtype=np.float32)
    mask = np.ones(n_trials, dtype=bool)

    for trial_idx in range(n_trials):
        t_beg = interval_begs[trial_idx]
        t_end = interval_ends[trial_idx]

        if np.isnan(t_beg) or np.isnan(t_end):
            mask[trial_idx] = False
            continue

        # Find behavior samples in this interval
        idx_beg = np.searchsorted(beh_times, t_beg, side='right')
        idx_end = np.searchsorted(beh_times, t_end, side='left')

        beh_t = beh_times[idx_beg:idx_end]
        beh_v = beh_vals[idx_beg:idx_end]

        if len(beh_v) == 0:
            mask[trial_idx] = False
            continue

        # Check coverage (matching reference code)
        if np.abs(t_beg - beh_t[0]) > BINSIZE:
            mask[trial_idx] = False
            continue
        if np.abs(t_end - beh_t[-1]) > BINSIZE:
            mask[trial_idx] = False
            continue

        # Interpolation points matching reference: linspace(beg + binsize, end, n_bins)
        x_interp = np.linspace(t_beg + BINSIZE, t_end, N_BINS)

        try:
            interp_func = interp1d(beh_t, beh_v, kind='linear', fill_value='extrapolate')
            values[trial_idx] = interp_func(x_interp).astype(np.float32)
        except Exception:
            mask[trial_idx] = False

    return values, mask


def compute_trial_num_in_block(prob_left):
    """Compute trial number within block.

    A block is a contiguous sequence of trials with the same probabilityLeft.
    Trial number resets to 1 at each block boundary.
    """
    trial_nums = np.ones(len(prob_left), dtype=np.int32)
    for i in range(1, len(prob_left)):
        if prob_left[i] == prob_left[i - 1]:
            trial_nums[i] = trial_nums[i - 1] + 1
        else:
            trial_nums[i] = 1
    return trial_nums


def discretize_to_bins(values, n_bins=3):
    """Discretize continuous values into n_bins categories using quantiles.

    Computes quantile boundaries from non-NaN values and assigns each value
    to a bin. Returns integer bin indices.
    """
    flat = values[~np.isnan(values)].flatten()
    if len(flat) == 0:
        return np.zeros_like(values, dtype=np.int32)

    quantiles = np.linspace(0, 100, n_bins + 1)[1:-1]
    boundaries = np.percentile(flat, quantiles)

    result = np.digitize(values, boundaries).astype(np.int32)
    return result


def process_session(sdir, br, show_processing=False, session_idx=0):
    """Process a single session and return data dict.

    Returns None if session cannot be processed.
    """
    lab, subject, date = parse_session_info(sdir)
    session_id = f"{subject}_{date}"
    t0 = time.time()

    try:
        # 1. Load trials
        trials = load_trials(sdir)
        mask = create_trial_mask(trials)

        # 2. Load spikes
        spike_times, spike_clusters, cluster_regions = load_spikes(sdir)
        n_clusters = len(cluster_regions)

        # 3. Compute trial intervals aligned to stimOn_times
        stim_on = trials[ALIGN_TIME].values
        interval_begs = stim_on + TIME_WINDOW[0]
        interval_ends = stim_on + TIME_WINDOW[1]

        # 4. Bin spikes for ALL trials first (before masking)
        print(f"  Binning spikes ({n_clusters} neurons, {len(trials)} trials)...", flush=True)
        t_bin = time.time()
        binned_spikes = bin_spikes_vectorized(
            spike_times, spike_clusters, n_clusters, interval_begs, interval_ends
        )
        print(f"  Spike binning: {time.time() - t_bin:.1f}s", flush=True)

        # 5. Load and bin behavioral data
        print(f"  Loading wheel speed...", flush=True)
        wh_times, wh_speed = load_wheel_speed(sdir)
        wheel_vals, wheel_mask = interpolate_behavior_to_bins(
            wh_times, wh_speed, interval_begs, interval_ends
        )

        print(f"  Loading whisker ME...", flush=True)
        me_times, me_vals_raw = load_whisker_me(sdir)
        whisker_vals, whisker_mask = interpolate_behavior_to_bins(
            me_times, me_vals_raw, interval_begs, interval_ends
        )

        # 6. Combine masks: trial quality + behavior availability
        # Following reference code align_spike_behavior
        combined_mask = mask.values & wheel_mask & whisker_mask

        # Apply mask
        good_indices = np.where(combined_mask)[0]
        if len(good_indices) < 2:
            print(f"  WARNING: Only {len(good_indices)} valid trials, skipping session", flush=True)
            return None

        neural_trials = binned_spikes[good_indices]  # (n_trials, n_clusters, n_bins)
        wheel_trials = wheel_vals[good_indices]  # (n_trials, n_bins)
        whisker_trials = whisker_vals[good_indices]  # (n_trials, n_bins)

        # 7. Extract per-trial variables
        trials_good = trials.iloc[good_indices]

        # Choice: -1 (left) -> 0, 1 (right) -> 1
        choice = trials_good['choice'].values.copy()
        choice_binary = np.where(choice == 1, 1, 0).astype(np.int32)

        # Prior: probabilityLeft -> 0.2->0, 0.5->1, 0.8->2
        prob_left = trials_good['probabilityLeft'].values
        prior = np.full(len(prob_left), 1, dtype=np.int32)  # default 0.5->1
        prior[np.isclose(prob_left, 0.2, atol=0.05)] = 0
        prior[np.isclose(prob_left, 0.8, atol=0.05)] = 2

        # Trial number in block
        trial_num_in_block = compute_trial_num_in_block(prob_left)

        # 8. Discretize continuous outputs
        wheel_discrete = discretize_to_bins(wheel_trials, n_bins=3)
        whisker_discrete = discretize_to_bins(whisker_trials, n_bins=3)

        # 9. Build time input
        # Time since stimulus onset: center of each bin
        time_input = np.linspace(
            TIME_WINDOW[0] + BINSIZE / 2,
            TIME_WINDOW[1] - BINSIZE / 2,
            N_BINS
        ).astype(np.float32)

        # 10. Format as lists of per-trial arrays
        n_trials = len(good_indices)

        # Store as uint8 during accumulation to save memory (spike counts rarely exceed 255)
        # Will be converted to float32 when decoder loads it
        neural_list = [np.clip(neural_trials[i], 0, 255).astype(np.uint8) for i in range(n_trials)]

        input_list = []
        for i in range(n_trials):
            inp = np.stack([
                time_input,
                np.full(N_BINS, trial_num_in_block[i], dtype=np.float32)
            ], axis=0)  # (2, 100)
            input_list.append(inp)

        output_list = []
        for i in range(n_trials):
            out = np.stack([
                np.full(N_BINS, choice_binary[i], dtype=np.int64),
                np.full(N_BINS, prior[i], dtype=np.int64),
                wheel_discrete[i].astype(np.int64),
                whisker_discrete[i].astype(np.int64)
            ], axis=0)  # (4, 100)
            output_list.append(out)

        elapsed = time.time() - t0
        print(f"  Session {session_id}: {n_trials} trials, {n_clusters} neurons, {elapsed:.1f}s", flush=True)

        # Show processing plots if requested
        if show_processing:
            plot_processing(
                session_id, trials_good, neural_trials, wheel_trials, whisker_trials,
                choice_binary, prior, wheel_discrete, whisker_discrete,
                time_input, interval_begs[good_indices], interval_ends[good_indices],
                session_idx
            )

        return {
            'neural': neural_list,
            'input': input_list,
            'output': output_list,
            'subject': subject,
            'cluster_regions': cluster_regions,
            'n_trials': n_trials,
            'n_neurons': n_clusters,
            'session_id': session_id,
        }

    except Exception as e:
        print(f"  ERROR processing {session_id}: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return None


def plot_processing(session_id, trials, neural, wheel, whisker,
                    choice, prior, wheel_disc, whisker_disc,
                    time_input, interval_begs, interval_ends, session_idx):
    """Plot processing visualizations for a session."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n_trials_plot = min(3, len(trials))
    fig, axes = plt.subplots(6, n_trials_plot, figsize=(5 * n_trials_plot, 20))
    if n_trials_plot == 1:
        axes = axes[:, np.newaxis]

    for i in range(n_trials_plot):
        t = time_input

        # Neural activity (mean across neurons)
        ax = axes[0, i]
        mean_rate = neural[i].mean(axis=0)
        ax.plot(t, mean_rate)
        ax.set_title(f'Trial {i}: Mean firing rate')
        ax.axvline(0, color='r', linestyle='--', alpha=0.5, label='stim onset')
        ax.set_ylabel('Spike count')

        # Wheel speed continuous
        ax = axes[1, i]
        ax.plot(t, wheel[i])
        ax.set_title(f'Wheel speed (continuous)')
        ax.axvline(0, color='r', linestyle='--', alpha=0.5)

        # Wheel speed discretized
        ax = axes[2, i]
        ax.plot(t, wheel_disc[i])
        ax.set_title(f'Wheel speed (discretized)')
        ax.set_yticks([0, 1, 2])
        ax.axvline(0, color='r', linestyle='--', alpha=0.5)

        # Whisker ME continuous
        ax = axes[3, i]
        ax.plot(t, whisker[i])
        ax.set_title(f'Whisker ME (continuous)')
        ax.axvline(0, color='r', linestyle='--', alpha=0.5)

        # Whisker ME discretized
        ax = axes[4, i]
        ax.plot(t, whisker_disc[i])
        ax.set_title(f'Whisker ME (discretized)')
        ax.set_yticks([0, 1, 2])
        ax.axvline(0, color='r', linestyle='--', alpha=0.5)

        # Per-trial outputs
        ax = axes[5, i]
        ax.text(0.5, 0.7, f'Choice: {choice[i]}', transform=ax.transAxes, ha='center')
        ax.text(0.5, 0.4, f'Prior: {prior[i]}', transform=ax.transAxes, ha='center')
        ax.set_title('Per-trial outputs')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    fig.suptitle(f'Processing: {session_id}', fontsize=14)
    fig.tight_layout()
    fig.savefig(f'processing_{session_idx}.png', dpi=100)
    plt.close(fig)
    print(f"  Saved processing plot: processing_{session_idx}.png", flush=True)


# ============================================================
# Main conversion function
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Convert IBL data to decoder format')
    parser.add_argument('outfile', type=str, help='Output pickle file path')
    parser.add_argument('--sample', action='store_true', help='Process only 2 sessions')
    parser.add_argument('--full', action='store_true', help='Process all sessions (default)')
    parser.add_argument('--show-processing', action='store_true', help='Plot processing visualizations')
    args = parser.parse_args()

    t_start = time.time()

    # Find session directories
    print("Finding session directories...", flush=True)
    session_dirs = find_session_dirs()
    print(f"Found {len(session_dirs)} sessions with complete data", flush=True)

    if args.sample:
        session_dirs = session_dirs[:2]
        print(f"Sample mode: processing {len(session_dirs)} sessions", flush=True)

    # Initialize brain regions mapper
    br = BrainRegions()

    # Process sessions - build output incrementally to save memory
    # Pass 1: process all sessions, collect data and region names
    neural = []
    input_data = []
    output_data = []
    all_subjects = []
    subject_per_session = []
    cluster_regions_per_session = []
    all_brain_regions = set()
    n_success = 0

    for i, sdir in enumerate(session_dirs):
        lab, subject, date = parse_session_info(sdir)
        print(f"\n[{i+1}/{len(session_dirs)}] Processing {subject}/{date}...", flush=True)

        result = process_session(
            sdir, br,
            show_processing=(args.show_processing and i < 2),
            session_idx=i
        )

        if result is not None:
            neural.append(result['neural'])
            input_data.append(result['input'])
            output_data.append(result['output'])
            if subject not in all_subjects:
                all_subjects.append(subject)
            subject_per_session.append(subject)
            cluster_regions_per_session.append(result['cluster_regions'])
            all_brain_regions.update(result['cluster_regions'])
            n_success += 1
            del result  # free memory immediately
        gc.collect()
        # Print memory usage every 50 sessions
        if (i + 1) % 50 == 0:
            import resource
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # MB
            print(f"  [Memory] RSS: {rss:.0f} MB after {n_success} sessions", flush=True)

    print(f"\nSuccessfully processed {n_success} sessions", flush=True)

    if n_success == 0:
        print("ERROR: No sessions were processed successfully!", flush=True)
        sys.exit(1)

    # Build brain regions list (exclude 'root' and 'void')
    brain_regions = sorted([r for r in all_brain_regions if r not in ('root', 'void')])
    # Add root and void at end if present
    if 'void' in all_brain_regions:
        brain_regions.append('void')
    if 'root' in all_brain_regions:
        brain_regions.append('root')

    region_to_idx = {r: i for i, r in enumerate(brain_regions)}

    # Build index arrays
    subject_idx = np.array([all_subjects.index(s) for s in subject_per_session], dtype=np.int32)
    brain_region_idx = []
    for regions in cluster_regions_per_session:
        region_indices = np.array([
            region_to_idx.get(r, region_to_idx.get('root', 0))
            for r in regions
        ], dtype=np.int32)
        brain_region_idx.append(region_indices)
    del cluster_regions_per_session  # free memory

    data = {
        'neural': neural,
        'input': input_data,
        'output': output_data,
        'subjects': all_subjects,
        'subject_idx': subject_idx,
        'brain_regions': brain_regions,
        'brain_region_idx': brain_region_idx,
        'input_names': ['time_since_stim_onset', 'trial_num_in_block'],
        'output_names': ['choice', 'prior', 'wheel_speed', 'whisker_motion_energy'],
        'output_values': [
            ['left', 'right'],           # choice: 0=left, 1=right
            ['0.2', '0.5', '0.8'],       # prior: 0=0.2, 1=0.5, 2=0.8
            ['low', 'medium', 'high'],    # wheel_speed bins
            ['low', 'medium', 'high'],    # whisker_motion_energy bins
        ],
        'metadata': {
            'task_description': 'IBL visual decision-making task: mice rotate wheel to indicate stimulus location',
            'time_bin_size': BINSIZE * 1000,  # 20ms
            'temporal_alignment_event': 'stimulus onset (stimOn_times)',
            'off_start': TIME_WINDOW[0],  # -0.5s
            'off_end': TIME_WINDOW[1],    # 1.5s
            'bin_size_seconds': BINSIZE,
            'n_time_bins': N_BINS,
            'trial_filter': f'RT {MIN_RT}-{MAX_RT}s, exclude no-choice, NaN exclusion',
            'neuron_filter': 'All neurons (no QC filtering), matching reference code',
            'source': 'IBL Brain-wide Map dataset',
            'reference_code': 'Zhang et al. (2025) code_zhang2025',
        }
    }

    # Save
    print(f"\nSaving to {args.outfile}...", flush=True)
    with open(args.outfile, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    file_size = os.path.getsize(args.outfile) / (1024 * 1024)
    elapsed = time.time() - t_start

    # Summary
    total_trials = sum(len(s) for s in neural)
    total_neurons = sum(len(s[0]) for s in neural)
    print(f"\n{'='*60}", flush=True)
    print(f"Conversion complete!", flush=True)
    print(f"Sessions: {len(neural)}", flush=True)
    print(f"Subjects: {len(all_subjects)}", flush=True)
    print(f"Total trials: {total_trials}", flush=True)
    print(f"Brain regions: {len(brain_regions)}", flush=True)
    print(f"File size: {file_size:.1f} MB", flush=True)
    print(f"Time elapsed: {elapsed:.1f}s", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == '__main__':
    main()
