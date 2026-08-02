"""
Convert IBL Brain Wide Map data into the standardized decoder format.

This script loads spike-sorted neural data, trial information, wheel speed,
and whisker motion energy from the IBL Brain Wide Map dataset, processes them
following the methods in Zhang et al. (2025) and IBL et al. (2024), and saves
them in the standardized dictionary format for decoder training.

Key processing decisions (matching reference code and papers):
- Temporal alignment: stimulus onset (stimOn_times)
- Time window: -0.5s to +1.5s relative to stimulus onset (2s total)
- Bin size: 20ms non-overlapping bins -> T=100 time steps per trial
- Trial filtering: exclude trials with RT < 0.08s or > 2s, NaN events, no-choice
- Neuron quality: use all clusters (no QC filter), matching reference code's default qc=None
  in prepare_data -> load_spiking_data; good cluster filtering happens at region level in BWM paper
- Brain regions: Beryl mapping
- Merge probes within session
- Wheel speed: absolute velocity, interpolated to 20ms bins in trial window
- Whisker motion energy: interpolated to 20ms bins in trial window
- Both wheel speed and whisker ME discretized into 3 bins (terciles)

Decoder task specification:
- Inputs: time since stimulus onset (continuous), trial number in block (continuous)
- Outputs: choice (binary), prior (categorical 3-class), wheel speed (3 bins),
           whisker motion energy (3 bins)
"""

import os
import sys
import argparse
import pickle
import traceback
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import interp1d

from one.api import ONE
from brainbox.io.one import SpikeSortingLoader, SessionLoader
from iblatlas.regions import BrainRegions
from iblutil.numerical import ismember


def load_spiking_data(one, pid, eid='', pname=''):
    """Load spike sorting data for a probe insertion. Uses all clusters (no QC filter)
    matching the reference code's default behavior in prepare_data."""
    ssl = SpikeSortingLoader(pid=pid, one=one, eid=eid, pname=pname)
    spikes, clusters, channels = ssl.load_spike_sorting()
    clusters_labeled = SpikeSortingLoader.merge_clusters(spikes, clusters, channels).to_df()
    return spikes, clusters_labeled


def merge_probes(spikes_list, clusters_list):
    """Merge spikes and clusters from multiple probes in same session.
    Matches reference code merge_probes function."""
    merged_spikes = []
    merged_clusters = []
    cluster_max = 0

    for clusters, spikes in zip(clusters_list, spikes_list):
        spikes_copy = {k: v.copy() for k, v in spikes.items()}
        spikes_copy['clusters'] = spikes_copy['clusters'] + cluster_max
        cluster_max = clusters.index.max() + 1
        merged_spikes.append(spikes_copy)
        merged_clusters.append(clusters)

    merged_clusters = pd.concat(merged_clusters, ignore_index=True)
    merged_spikes_dict = {k: np.concatenate([s[k] for s in merged_spikes]) for k in merged_spikes[0].keys()}
    sort_idx = np.argsort(merged_spikes_dict['times'], kind='stable')
    merged_spikes_dict = {k: v[sort_idx] for k, v in merged_spikes_dict.items()}

    return merged_spikes_dict, merged_clusters


def load_trials_and_mask(one, eid, min_rt=0.08, max_rt=2.0):
    """Load trials and create mask following reference code.
    Excludes: RT outside [0.08, 2.0]s, NaN key events, no-choice trials."""
    sl = SessionLoader(one=one, eid=eid)
    sl.load_trials()
    trials = sl.trials

    nan_exclude = [
        'stimOn_times', 'choice', 'feedback_times',
        'probabilityLeft', 'firstMovement_times', 'feedbackType'
    ]

    # Build mask
    mask = pd.Series(True, index=trials.index)

    # RT filter
    rt = trials['firstMovement_times'] - trials['stimOn_times']
    if min_rt is not None:
        mask &= rt >= min_rt
    if max_rt is not None:
        mask &= rt <= max_rt

    # NaN exclusion
    for event in nan_exclude:
        mask &= ~trials[event].isnull()

    # Exclude no-choice trials (choice == 0)
    mask &= trials['choice'] != 0

    return trials, mask, sl


def bin_spikes_in_window(spike_times, spike_clusters, cluster_ids,
                          align_times, window, binsize):
    """Bin spikes into time bins for each trial.

    Uses vectorized numpy operations for speed.
    Returns: array of shape (n_trials, n_clusters, n_bins)
    """
    n_trials = len(align_times)
    n_clusters = len(cluster_ids)
    n_bins = int(np.round((window[1] - window[0]) / binsize))

    # Create cluster id to index mapping
    cluster_to_idx = np.full(int(cluster_ids.max()) + 1, -1, dtype=np.int32)
    for idx, cid in enumerate(cluster_ids):
        cluster_to_idx[cid] = idx

    binned = np.zeros((n_trials, n_clusters, n_bins), dtype=np.float32)

    # Sort spike times for fast searchsorted
    sort_idx = np.argsort(spike_times)
    sorted_times = spike_times[sort_idx]
    sorted_clusters = spike_clusters[sort_idx]

    for trial_idx in range(n_trials):
        t_start = align_times[trial_idx] + window[0]
        t_end = align_times[trial_idx] + window[1]

        if np.isnan(t_start) or np.isnan(t_end):
            continue

        # Use searchsorted for fast spike selection
        i_start = np.searchsorted(sorted_times, t_start, side='left')
        i_end = np.searchsorted(sorted_times, t_end, side='left')

        if i_start >= i_end:
            continue

        times_in_window = sorted_times[i_start:i_end]
        clusters_in_window = sorted_clusters[i_start:i_end]

        # Vectorized binning
        bin_indices = np.floor((times_in_window - t_start) / binsize).astype(np.int32)
        np.clip(bin_indices, 0, n_bins - 1, out=bin_indices)

        # Map cluster ids to indices (vectorized)
        valid_mask = clusters_in_window < len(cluster_to_idx)
        cluster_indices = np.where(valid_mask,
                                    cluster_to_idx[clusters_in_window[valid_mask] if valid_mask.all() else clusters_in_window],
                                    -1)
        if not valid_mask.all():
            ci = np.full(len(clusters_in_window), -1, dtype=np.int32)
            ci[valid_mask] = cluster_to_idx[clusters_in_window[valid_mask]]
            cluster_indices = ci
        else:
            cluster_indices = cluster_to_idx[clusters_in_window]

        # Use np.add.at for accumulation
        valid = cluster_indices >= 0
        if valid.any():
            np.add.at(binned[trial_idx], (cluster_indices[valid], bin_indices[valid]), 1)

    return binned


def interpolate_behavior_to_bins(beh_times, beh_values, align_times, window, binsize):
    """Interpolate behavioral time series to match neural bins.

    Following reference code: bin centers are at
    np.linspace(align + window[0] + binsize, align + window[1], n_bins)

    Returns: array of shape (n_trials,n_bins), mask of good trials
    """
    n_bins = int(np.round((window[1] - window[0]) / binsize))
    n_trials = len(align_times)
    binned_beh = np.zeros((n_trials, n_bins), dtype=np.float32)
    good_mask = np.ones(n_trials, dtype=bool)

    for trial_idx in range(n_trials):
        t_start = align_times[trial_idx] + window[0]
        t_end = align_times[trial_idx] + window[1]

        if np.isnan(t_start) or np.isnan(t_end):
            good_mask[trial_idx] = False
            continue

        # Get behavior data in window (with small buffer)
        beh_mask = (beh_times >= t_start - binsize) & (beh_times <= t_end + binsize)
        bt = beh_times[beh_mask]
        bv = beh_values[beh_mask]

        if len(bt) < 2:
            good_mask[trial_idx] = False
            continue

        if np.any(np.isnan(bv)):
            good_mask[trial_idx] = False
            continue

        # Check coverage
        if np.abs(t_start - bt[0]) > binsize:
            good_mask[trial_idx] = False
            continue
        if np.abs(t_end - bt[-1]) > binsize:
            good_mask[trial_idx] = False
            continue

        # Interpolate to bin centers (matching reference code)
        x_interp = np.linspace(t_start + binsize, t_end, n_bins)
        try:
            f = interp1d(bt, bv, kind='linear', fill_value='extrapolate')
            binned_beh[trial_idx] = f(x_interp).astype(np.float32)
        except Exception:
            good_mask[trial_idx] = False

    return binned_beh, good_mask


def compute_trial_number_in_block(trials_df, mask):
    """Compute trial number within each block of constant probabilityLeft.

    A block change occurs when probabilityLeft changes from one trial to the next.
    Trial number is 1-indexed within each block.
    """
    pLeft = trials_df['probabilityLeft'].values
    trial_num = np.zeros(len(pLeft), dtype=np.float32)

    block_counter = 1
    for i in range(len(pLeft)):
        if i == 0 or pLeft[i] != pLeft[i-1]:
            block_counter = 1
        trial_num[i] = block_counter
        block_counter += 1

    # Return only masked trials
    return trial_num[mask]


def discretize_to_bins(values, n_bins=3):
    """Discretize continuous values into n_bins categories using quantile-based binning.

    Computes bin edges from the data (across all trials/timepoints), then assigns
    each value to a bin (0 to n_bins-1).
    """
    # Flatten all values to compute global quantiles
    flat = values.flatten()
    flat = flat[~np.isnan(flat)]

    # Compute quantile edges
    quantiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(flat, quantiles)

    # Make edges unique to handle ties
    edges[0] = -np.inf
    edges[-1] = np.inf

    # Digitize: returns bin indices 1..n_bins, subtract 1 for 0-indexed
    discretized = np.digitize(values, edges[1:-1]).astype(np.int64)

    return discretized


def convert_ibl_data(one, bwm_df, max_sessions=None, sample_mode=False, verbose=True):
    """Main conversion function.

    Parameters
    ----------
    one : ONE instance
    bwm_df : DataFrame with BWM release info
    max_sessions : int or None, limit number of sessions
    sample_mode : bool, if True process only first 5 sessions
    verbose : bool

    Returns
    -------
    data : dict in target format
    """

    # Parameters matching reference code
    WINDOW = (-0.5, 1.5)  # seconds relative to stimOn_times
    BINSIZE = 0.02  # 20ms bins
    N_BINS = int(np.round((WINDOW[1] - WINDOW[0]) / BINSIZE))  # = 100

    brainreg = BrainRegions()

    # Get unique eids
    eids = bwm_df['eid'].unique()
    if sample_mode:
        eids = eids[:5]
    elif max_sessions is not None:
        eids = eids[:max_sessions]

    # Storage
    all_neural = []
    all_input = []
    all_output = []
    all_subject_names = []
    all_subject_idx = []
    all_brain_region_idx = []
    all_brain_regions_set = set()

    # Track all wheel speed and whisker ME values for global discretization
    all_wheel_speed_raw = []
    all_whisker_me_raw = []
    session_meta = []

    # First pass: collect all data and raw behavior values
    subjects_list = []
    subject_to_idx = {}

    processed_sessions = 0
    skipped_sessions = 0

    for eid_idx, eid in enumerate(eids):
        try:
            rows = bwm_df[bwm_df['eid'] == eid]
            lab = rows.iloc[0]['lab']
            subject = rows.iloc[0]['subject']

            if verbose:
                print(f'\n[{eid_idx+1}/{len(eids)}] Processing session {eid} ({subject}, {lab})')

            # Track subject
            if subject not in subject_to_idx:
                subject_to_idx[subject] = len(subjects_list)
                subjects_list.append(subject)
            subj_idx = subject_to_idx[subject]

            # Load spike sorting - merge probes
            pids = rows['pid'].values
            probe_names = rows['probe_name'].values

            spikes_list = []
            clusters_list = []
            for pid, pname in zip(pids, probe_names):
                try:
                    spks, clust = load_spiking_data(one, pid, eid=eid, pname=pname)
                    spikes_list.append(spks)
                    clusters_list.append(clust)
                except Exception as e:
                    print(f'  Warning: Failed to load probe {pid}: {e}')

            if len(spikes_list) == 0:
                print(f'  Skipping: no probes loaded')
                skipped_sessions += 1
                continue

            if len(spikes_list) > 1:
                spikes, clusters = merge_probes(spikes_list, clusters_list)
            else:
                spikes = spikes_list[0]
                clusters = clusters_list[0]

            # Get brain regions (Beryl mapping)
            beryl_reg = brainreg.acronym2acronym(clusters['acronym'].values, mapping='Beryl')

            # Load trials and mask
            trials, mask, sl = load_trials_and_mask(one, eid)

            if mask.sum() < 10:
                print(f'  Skipping: only {mask.sum()} valid trials')
                skipped_sessions += 1
                continue

            masked_trials = trials[mask].reset_index(drop=True)
            align_times = masked_trials['stimOn_times'].values

            if verbose:
                print(f'  Trials: {len(trials)} total, {mask.sum()} valid')
                print(f'  Neurons: {len(clusters)} total')
                print(f'  Probes: {len(pids)}')

            # Bin spikes
            cluster_ids = np.arange(len(clusters))
            binned_spikes = bin_spikes_in_window(
                spikes['times'], spikes['clusters'], cluster_ids,
                align_times, WINDOW, BINSIZE
            )
            # binned_spikes shape: (n_trials, n_clusters, n_bins)

            # Load wheel speed
            sl.load_wheel()
            wheel_times = sl.wheel['times'].values
            wheel_speed = np.abs(sl.wheel['velocity'].values)

            wheel_binned, wheel_mask = interpolate_behavior_to_bins(
                wheel_times, wheel_speed, align_times, WINDOW, BINSIZE
            )

            # Load whisker motion energy (try left first, then right)
            whisker_binned = None
            whisker_mask = None
            try:
                sl.load_motion_energy(views=['left'])
                me_times = sl.motion_energy['leftCamera']['times'].values
                me_values = sl.motion_energy['leftCamera']['whiskerMotionEnergy'].values
                whisker_binned, whisker_mask = interpolate_behavior_to_bins(
                    me_times, me_values, align_times, WINDOW, BINSIZE
                )
            except Exception:
                try:
                    sl.load_motion_energy(views=['right'])
                    me_times = sl.motion_energy['rightCamera']['times'].values
                    me_values = sl.motion_energy['rightCamera']['whiskerMotionEnergy'].values
                    whisker_binned, whisker_mask = interpolate_behavior_to_bins(
                        me_times, me_values, align_times, WINDOW, BINSIZE
                    )
                except Exception as e:
                    print(f'  Warning: Could not load motion energy: {e}')

            if whisker_binned is None:
                print(f'  Skipping: no whisker motion energy')
                skipped_sessions += 1
                continue

            # Combined mask: trials where both wheel and whisker data are good
            combined_mask = wheel_mask & whisker_mask

            if combined_mask.sum() < 10:
                print(f'  Skipping: only {combined_mask.sum()} trials with valid behavior')
                skipped_sessions += 1
                continue

            # Apply combined mask
            binned_spikes = binned_spikes[combined_mask]
            wheel_binned = wheel_binned[combined_mask]
            whisker_binned = whisker_binned[combined_mask]

            # Extract trial variables for masked trials
            masked_idx = np.where(combined_mask)[0]

            # Choice: left=-1 -> 0, right=1 -> 1 (as per decoder spec)
            choice_vals = masked_trials.iloc[masked_idx]['choice'].values.copy()
            choice = np.where(choice_vals == 1, 1, 0).astype(np.int64)

            # Prior: probabilityLeft -> 0.2->0, 0.5->1, 0.8->2
            pLeft = masked_trials.iloc[masked_idx]['probabilityLeft'].values
            prior = np.zeros(len(pLeft), dtype=np.int64)
            prior[pLeft == 0.2] = 0
            prior[pLeft == 0.5] = 1
            prior[pLeft == 0.8] = 2

            # Trial number in block
            trial_num_in_block = compute_trial_number_in_block(trials, mask.values)
            trial_num_in_block = trial_num_in_block[combined_mask]

            # Time since stimulus onset (same for all trials: relative time in window)
            time_since_stim = np.linspace(
                WINDOW[0] + BINSIZE, WINDOW[1], N_BINS
            ).astype(np.float32)

            # Store raw behavior for later discretization
            all_wheel_speed_raw.append(wheel_binned)
            all_whisker_me_raw.append(whisker_binned)

            # Store session data
            session_meta.append({
                'eid': eid,
                'subject': subject,
                'lab': lab,
                'subject_idx': subj_idx,
                'beryl_regions': beryl_reg,
                'n_trials': combined_mask.sum(),
                'n_neurons': len(clusters),
                'binned_spikes': binned_spikes,
                'choice': choice,
                'prior': prior,
                'trial_num_in_block': trial_num_in_block,
                'time_since_stim': time_since_stim,
                'wheel_binned': wheel_binned,
                'whisker_binned': whisker_binned,
            })

            processed_sessions += 1

            if verbose:
                print(f'  Valid trials after behavior mask: {combined_mask.sum()}')
                print(f'  Brain regions: {np.unique(beryl_reg)}')

        except Exception as e:
            print(f'  ERROR processing session {eid}: {e}')
            traceback.print_exc()
            skipped_sessions += 1
            continue

    print(f'\n{"="*60}')
    print(f'Processed {processed_sessions} sessions, skipped {skipped_sessions}')
    print(f'{"="*60}')

    if processed_sessions == 0:
        raise ValueError("No sessions were successfully processed!")

    # Compute global discretization thresholds for wheel speed and whisker ME
    all_wheel_flat = np.concatenate([w.flatten() for w in all_wheel_speed_raw])
    all_whisker_flat = np.concatenate([w.flatten() for w in all_whisker_me_raw])

    wheel_edges = np.percentile(all_wheel_flat[~np.isnan(all_wheel_flat)], [100/3, 200/3])
    whisker_edges = np.percentile(all_whisker_flat[~np.isnan(all_whisker_flat)], [100/3, 200/3])

    print(f'Wheel speed tercile edges: {wheel_edges}')
    print(f'Whisker ME tercile edges: {whisker_edges}')

    # Build all brain regions list
    all_regions = set()
    for sm in session_meta:
        all_regions.update(sm['beryl_regions'])
    all_regions = sorted(all_regions)
    region_to_idx = {r: i for i, r in enumerate(all_regions)}

    # Now build the final data structure
    neural_list = []
    input_list = []
    output_list = []
    subject_idx_list = []
    brain_region_idx_list = []

    for sm in session_meta:
        n_trials = sm['n_trials']
        n_neurons = sm['n_neurons']

        # Neural: list of trials, each (n_neurons, n_bins)
        session_neural = []
        session_input = []
        session_output = []

        # Brain region index for this session
        region_idx = np.array([region_to_idx[r] for r in sm['beryl_regions']], dtype=np.int64)

        # Discretize wheel speed and whisker ME using global edges
        wheel_disc = np.digitize(sm['wheel_binned'], wheel_edges).astype(np.int64)
        whisker_disc = np.digitize(sm['whisker_binned'], whisker_edges).astype(np.int64)

        for trial in range(n_trials):
            # Neural: (n_neurons, T)
            session_neural.append(sm['binned_spikes'][trial].astype(np.float32))

            # Input: (2, T) - time since stimulus onset + trial number in block
            time_input = sm['time_since_stim'].copy()  # (T,)
            trial_num = np.full(N_BINS, sm['trial_num_in_block'][trial], dtype=np.float32)
            inp = np.stack([time_input, trial_num], axis=0)  # (2, T)
            session_input.append(inp.astype(np.float32))

            # Output: (4, T) for time-varying, (4,) for per-trial
            # Choice and prior are per-trial, wheel speed and whisker ME are time-varying
            # Make all time-varying for consistency
            choice_arr = np.full(N_BINS, sm['choice'][trial], dtype=np.int64)
            prior_arr = np.full(N_BINS, sm['prior'][trial], dtype=np.int64)
            wheel_arr = wheel_disc[trial].astype(np.int64)
            whisker_arr = whisker_disc[trial].astype(np.int64)

            out = np.stack([choice_arr, prior_arr, wheel_arr, whisker_arr], axis=0)
            session_output.append(out)

        neural_list.append(session_neural)
        input_list.append(session_input)
        output_list.append(session_output)
        subject_idx_list.append(sm['subject_idx'])
        brain_region_idx_list.append(region_idx)

    # Build final data dict
    data = {
        'neural': neural_list,
        'input': input_list,
        'output': output_list,

        'subjects': subjects_list,
        'subject_idx': np.array(subject_idx_list, dtype=np.int64),

        'brain_regions': all_regions,
        'brain_region_idx': brain_region_idx_list,

        'input_names': ['time_since_stimulus_onset', 'trial_number_in_block'],
        'output_names': ['choice', 'prior', 'wheel_speed', 'whisker_motion_energy'],
        'output_values': [
            ['left', 'right'],  # choice: 0=left, 1=right
            ['0.2', '0.5', '0.8'],  # prior: 0=0.2, 1=0.5, 2=0.8
            ['low', 'medium', 'high'],  # wheel speed bins
            ['low', 'medium', 'high'],  # whisker ME bins
        ],

        'metadata': {
            'task_description': 'IBL Brain Wide Map: mice rotate a wheel to indicate '
                               'the location of a visual stimulus (left/right)',
            'time_bin_size': BINSIZE * 1000,  # 20ms
            'temporal_alignment_event': 'stimulus onset (stimOn_times)',
            'off_start': WINDOW[0],  # -0.5s before stimulus onset
            'off_end': WINDOW[1],  # 1.5s after stimulus onset
            'window': WINDOW,
            'binsize_s': BINSIZE,
            'n_sessions': len(neural_list),
            'n_subjects': len(subjects_list),
            'wheel_speed_tercile_edges': wheel_edges.tolist(),
            'whisker_me_tercile_edges': whisker_edges.tolist(),
            'trial_filtering': 'RT in [0.08, 2.0]s, no NaN events, no no-choice trials',
            'neuron_selection': 'All spike-sorted clusters (matching reference code default)',
            'brain_region_mapping': 'Beryl',
            'reference_paper': 'Zhang et al. 2025, IBL et al. 2024',
        }
    }

    return data


def print_sanity_checks(data):
    """Print sanity checks comparing to reference paper statistics."""
    print('\n' + '='*60)
    print('SANITY CHECKS')
    print('='*60)

    n_sessions = len(data['neural'])
    n_trials = sum(len(s) for s in data['neural'])
    n_subjects = len(data['subjects'])

    print(f'\nDataset size:')
    print(f'  Sessions: {n_sessions} (paper: 433 sessions used in Zhang et al.)')
    print(f'  Total trials: {n_trials}')
    print(f'  Subjects: {n_subjects} (paper: 139 mice)')
    print(f'  Brain regions: {len(data["brain_regions"])} (paper: 270 regions)')

    # Neuron counts
    nneurons = [data['neural'][s][0].shape[0] for s in range(n_sessions)]
    print(f'\nNeurons per session:')
    print(f'  Mean: {np.mean(nneurons):.1f}')
    print(f'  Min: {np.min(nneurons)}, Max: {np.max(nneurons)}')
    print(f'  Total: {sum(nneurons)}')

    # Time bins
    T_values = [data['neural'][s][0].shape[1] for s in range(n_sessions)]
    print(f'\nTime bins per trial: {T_values[0]} (expected: 100 = 2s / 20ms)')

    # Trials per session
    trials_per = [len(data['neural'][s]) for s in range(n_sessions)]
    print(f'\nTrials per session:')
    print(f'  Mean: {np.mean(trials_per):.1f}')
    print(f'  Min: {np.min(trials_per)}, Max: {np.max(trials_per)}')

    # Output distributions
    print(f'\nOutput distributions:')
    for out_idx, out_name in enumerate(data['output_names']):
        all_vals = []
        for s in range(n_sessions):
            for t in range(len(data['output'][s])):
                trial_out = data['output'][s][t]
                if trial_out.ndim == 1:
                    all_vals.append(trial_out[out_idx])
                else:
                    all_vals.extend(trial_out[out_idx].tolist())
        all_vals = np.array(all_vals)
        unique, counts = np.unique(all_vals, return_counts=True)
        fracs = counts / counts.sum()
        print(f'  {out_name}: ' + ', '.join(f'{v:.0f}:{f:.3f}' for v, f in zip(unique, fracs)))

    # Input ranges
    print(f'\nInput ranges:')
    for inp_idx, inp_name in enumerate(data['input_names']):
        all_vals = []
        for s in range(n_sessions):
            for t in range(len(data['input'][s])):
                trial_in = data['input'][s][t]
                if trial_in.ndim == 1:
                    all_vals.append(trial_in[inp_idx])
                else:
                    all_vals.extend(trial_in[inp_idx].tolist())
        all_vals = np.array(all_vals)
        print(f'  {inp_name}: [{all_vals.min():.3f}, {all_vals.max():.3f}]')

    # Sessions per subject
    unique_subj, subj_counts = np.unique(data['subject_idx'], return_counts=True)
    print(f'\nSessions per subject:')
    print(f'  Mean: {np.mean(subj_counts):.1f}')
    print(f'  Min: {np.min(subj_counts)}, Max: {np.max(subj_counts)}')


def main():
    parser = argparse.ArgumentParser(description='Convert IBL BWM data to decoder format')
    parser.add_argument('--output', type=str, default='converted_data.pkl',
                        help='Output pickle file path')
    parser.add_argument('--sample', action='store_true',
                        help='Process only first 5 sessions (for testing)')
    parser.add_argument('--max-sessions', type=int, default=None,
                        help='Maximum number of sessions to process')
    parser.add_argument('--cache-dir', type=str, default='/app/data/one_cache',
                        help='ONE cache directory')
    args = parser.parse_args()

    print(f'Initializing ONE API...')
    one = ONE(
        base_url='https://openalyx.internationalbrainlab.org',
        password='international', silent=True,
        cache_dir=args.cache_dir
    )

    bwm_df = pd.read_csv('/app/code/code_zhang2025/data/bwm_release.csv', index_col=0)
    print(f'BWM release: {len(bwm_df)} probes, {bwm_df.eid.nunique()} sessions')

    # Convert
    data = convert_ibl_data(
        one, bwm_df,
        max_sessions=args.max_sessions,
        sample_mode=args.sample,
        verbose=True
    )

    # Sanity checks
    print_sanity_checks(data)

    # Save
    print(f'\nSaving to {args.output}...')
    with open(args.output, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    file_size = os.path.getsize(args.output) / (1024**3)
    print(f'Saved {args.output} ({file_size:.2f} GB)')

    return data


if __name__ == '__main__':
    main()
