#!/usr/bin/env python3
"""
Convert MAP dataset NWB files to decoder-compatible format.

Usage:
    python -u convert_data.py <outpicklefile> [--full|--sample] [--show-processing]
"""

import os
import sys
import glob
import time
import pickle
import argparse
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ============================================================
# Constants
# ============================================================
BIN_WIDTH = 0.050  # 50 ms bins for firing rates
WINDOW_START = -2.5  # seconds before go cue
WINDOW_END = 1.5    # seconds after go cue
TONE_ONSET_REL_GO = -1.85  # tone onset relative to go cue (seconds)
N_TIMEBINS = int((WINDOW_END - WINDOW_START) / BIN_WIDTH)  # 80 bins
BIN_CENTERS = WINDOW_START + BIN_WIDTH/2 + np.arange(N_TIMEBINS) * BIN_WIDTH

# Output mappings
CHOICE_MAP = {'left': 0, 'right': 1}
OUTCOME_MAP = {'ignore': 0, 'miss': 1, 'hit': 2}
EARLY_LICK_MAP = {'no early': 0, 'early': 1}


def compute_firing_rates_session(spike_times_list, go_times, bin_width=BIN_WIDTH,
                                  window_start=WINDOW_START, window_end=WINDOW_END):
    """Compute firing rates for all neurons across all trials."""
    n_neurons = len(spike_times_list)
    n_trials = len(go_times)
    n_bins = int((window_end - window_start) / bin_width)
    
    fr_all = []
    for trial_idx in range(n_trials):
        go_time = go_times[trial_idx]
        bin_edges = go_time + window_start + np.arange(n_bins + 1) * bin_width
        
        fr = np.zeros((n_neurons, n_bins), dtype=np.float32)
        for i, spikes in enumerate(spike_times_list):
            if len(spikes) > 0:
                left = np.searchsorted(spikes, bin_edges[0])
                right = np.searchsorted(spikes, bin_edges[-1])
                if left < right:
                    counts, _ = np.histogram(spikes[left:right], bins=bin_edges)
                    fr[i, :] = counts / bin_width
        fr_all.append(fr)
    
    return fr_all


def get_photostim_timeseries(photostim_start_times, photostim_stop_times,
                              go_time, bin_centers=BIN_CENTERS):
    """Create binary photostimulation time series for a trial."""
    n_bins = len(bin_centers)
    photostim = np.zeros(n_bins, dtype=np.float32)
    abs_bin_centers = go_time + bin_centers
    for start, stop in zip(photostim_start_times, photostim_stop_times):
        mask = (abs_bin_centers >= start) & (abs_bin_centers <= stop)
        photostim[mask] = 1.0
    return photostim


def get_tongue_y_for_trial(tongue_ts, tongue_data, go_time,
                           bin_centers=BIN_CENTERS, bin_width=BIN_WIDTH):
    """Extract tongue y-position for a trial, averaged within each time bin."""
    n_bins = len(bin_centers)
    tongue_y = np.full(n_bins, np.nan, dtype=np.float32)
    
    t_start = go_time + bin_centers[0] - bin_width / 2
    t_end = go_time + bin_centers[-1] + bin_width / 2
    
    idx_start = np.searchsorted(tongue_ts, t_start, side='left')
    idx_end = np.searchsorted(tongue_ts, t_end, side='right')
    
    if idx_start >= idx_end:
        return tongue_y
    
    local_ts = tongue_ts[idx_start:idx_end]
    local_y = tongue_data[idx_start:idx_end, 1]
    
    abs_bin_edges = go_time + bin_centers[0] - bin_width/2 + np.arange(n_bins + 1) * bin_width
    bin_indices = np.digitize(local_ts, abs_bin_edges) - 1
    
    for b in range(n_bins):
        mask = bin_indices == b
        if np.any(mask):
            tongue_y[b] = np.mean(local_y[mask])
    
    return tongue_y


def discretize_tongue_y(tongue_y_session, percentile_low=40, percentile_high=60):
    """Discretize tongue y-position per session."""
    all_values = []
    for y in tongue_y_session:
        valid = y[~np.isnan(y)]
        if len(valid) > 0:
            all_values.append(valid)
    
    if len(all_values) == 0:
        return [np.ones(len(y), dtype=np.int64) for y in tongue_y_session]
    
    all_values = np.concatenate(all_values)
    p_low = np.percentile(all_values, percentile_low)
    p_high = np.percentile(all_values, percentile_high)
    
    tongue_y_discrete = []
    for y in tongue_y_session:
        discrete = np.ones(len(y), dtype=np.int64)  # default middle
        valid = ~np.isnan(y)
        if np.any(valid):
            discrete[valid & (y < p_low)] = 0
            discrete[valid & (y >= p_low) & (y <= p_high)] = 1
            discrete[valid & (y > p_high)] = 2
        tongue_y_discrete.append(discrete)
    
    return tongue_y_discrete


def simplify_brain_region(anno_name):
    """Simplify detailed CCF annotation to broader brain region."""
    if not anno_name or anno_name.strip() == '':
        return 'unknown'
    
    name = anno_name.strip()
    
    region_map = {
        'Secondary motor area': 'MOs',
        'Primary motor area': 'MOp',
        'Anterior cingulate area': 'ACA',
        'Prelimbic area': 'PL',
        'Infralimbic area': 'ILA',
        'Orbital area': 'ORB',
        'Frontal pole': 'FRP',
        'Agranular insular area': 'AI',
        'Gustatory areas': 'GU',
        'Visceral area': 'VISC',
        'Primary somatosensory area': 'SSp',
        'Supplemental somatosensory area': 'SSs',
        'Primary auditory area': 'AUDp',
        'Posterior parietal association': 'PTLp',
        'Visual area': 'VIS',
        'Primary visual area': 'VISp',
        'Retrosplenial area': 'RSP',
        'Temporal association': 'TEa',
        'Perirhinal area': 'PERI',
        'Ectorhinal area': 'ECT',
        'Entorhinal area': 'ENT',
        'Caudoputamen': 'CP',
        'Striatum': 'STR',
        'Nucleus accumbens': 'ACB',
        'Pallidum': 'PAL',
        'Globus pallidus': 'GP',
        'Thalamus': 'TH',
        'Ventral posteromedial': 'VPM',
        'Ventral posterolateral': 'VPL',
        'Posterior complex': 'PO',
        'Lateral posterior': 'LP',
        'Ventral anterior-lateral': 'VAL',
        'Ventral medial': 'VM',
        'Mediodorsal': 'MD',
        'Reticular nucleus': 'RT',
        'Zona incerta': 'ZI',
        'Hypothalamus': 'HY',
        'Substantia nigra': 'SNr',
        'Midbrain reticular': 'MRN',
        'Superior colliculus': 'SC',
        'Inferior colliculus': 'IC',
        'Periaqueductal gray': 'PAG',
        'Red nucleus': 'RN',
        'Pontine gray': 'PG',
        'Medulla': 'MY',
        'Cerebellum': 'CB',
        'Hippocampal': 'HPC',
        'Subiculum': 'SUB',
        'Dentate gyrus': 'DG',
        'Ammon\'s horn': 'CA',
        'Bed nuclei of the stria terminalis': 'BST',
        'Olfactory areas': 'OLF',
        'Olfactory tubercle': 'OT',
        'Lateral septal nucleus': 'LS',
        'Medial septal nucleus': 'MS',
        'Central amygdalar nucleus': 'CEA',
        'Basolateral amygdalar nucleus': 'BLA',
        'Intercalated amygdalar nucleus': 'IA',
        'Medial amygdalar nucleus': 'MEA',
        'Endopiriform nucleus': 'EP',
        'Claustrum': 'CLA',
        'Lateral habenula': 'LH',
        'Medial habenula': 'MH',
        'Subthalamic nucleus': 'STN',
        'Substantia innominata': 'SI',
        'Magnocellular nucleus': 'MA',
        'Pedunculopontine nucleus': 'PPN',
        'Parabrachial nucleus': 'PB',
        'Dorsal column nuclei': 'DCN',
        'Vestibular nuclei': 'VN',
        'Cochlear nuclei': 'CN',
        'Spinal nucleus of the trigeminal': 'SPV',
        'Pontine reticular nucleus': 'PRNr',
        'Tegmental reticular nucleus': 'TRN',
        'Gigantocellular reticular nucleus': 'GRN',
        'Paragigantocellular reticular nucleus': 'PGRN',
        'Intermediate reticular nucleus': 'IRN',
        'Parvicellular reticular nucleus': 'PARN',
        'Lateral reticular nucleus': 'LRN',
        'Facial motor nucleus': 'VII',
        'Hypoglossal nucleus': 'XII',
        'Inferior olivary complex': 'IO',
        'Dentate nucleus': 'DN',
        'Fastigial nucleus': 'FN',
        'Interposed nucleus': 'IP',
    }
    
    for key, abbrev in region_map.items():
        if key.lower() in name.lower():
            return abbrev
    
    return name


def get_recording_range(nwb, good_indices):
    """Get the recording time range for good units using actual spike times."""
    spike_times_all = nwb.units['spike_times'][:]
    
    rec_start = np.inf
    rec_end = -np.inf
    
    for i in good_indices:
        st = np.array(spike_times_all[i])
        if len(st) > 0:
            rec_start = min(rec_start, st.min())
            rec_end = max(rec_end, st.max())
    
    if rec_start == np.inf:
        rec_start = 0
    if rec_end == -np.inf:
        rec_end = 0
    
    return rec_start, rec_end


def process_session(nwb_file, show_processing=False, session_idx=0):
    """Process a single NWB file and extract all required data."""
    import pynwb
    
    t0 = time.time()
    
    io = pynwb.NWBHDF5IO(nwb_file, 'r')
    nwb = io.read()
    
    # Extract subject info
    subject_id = nwb.subject.subject_id
    subject_desc = nwb.subject.description
    
    # Get trials info
    n_trials_total = len(nwb.trials)
    trial_instruction = nwb.trials['trial_instruction'][:]
    outcome = nwb.trials['outcome'][:]
    early_lick = nwb.trials['early_lick'][:]
    auto_water = nwb.trials['auto_water'][:]
    free_water = nwb.trials['free_water'][:]
    photostim_power = nwb.trials['photostim_power'][:]
    
    # Get units info - filter by classifier QC
    classification = nwb.units['classification'][:]
    good_mask = classification == 'good'
    n_good = np.sum(good_mask)
    
    if n_good < 2:
        print(f'  Skipping: only {n_good} good units')
        io.close()
        return None
    
    good_indices = np.where(good_mask)[0]
    
    # Get recording range from obs_intervals
    rec_start, rec_end = get_recording_range(nwb, good_indices)
    
    # Get go cue times
    beh_events = nwb.acquisition['BehavioralEvents']
    go_times = beh_events.time_series['go_start_times'].timestamps[:]
    
    # Filter trials by recording coverage
    valid_trial_mask = (go_times + WINDOW_START >= rec_start) & (go_times + WINDOW_END <= rec_end)
    valid_trial_indices = np.where(valid_trial_mask)[0]
    
    if len(valid_trial_indices) < 2:
        print(f'  Skipping: only {len(valid_trial_indices)} trials with neural coverage')
        io.close()
        return None
    
    # Session selection criteria (computed on ALL control trials, not just covered ones)
    is_control = np.array([str(p) == 'N/A' for p in photostim_power])
    is_no_early = np.array([str(e) == 'no early' for e in early_lick])
    control_no_early = is_control & is_no_early
    is_not_ignore = outcome != 'ignore'
    
    denom = np.sum(control_no_early & is_not_ignore)
    if denom == 0:
        print(f'  Skipping: no valid control trials')
        io.close()
        return None
    
    hits = np.sum(control_no_early & (outcome == 'hit'))
    correct_rate = hits / denom
    correct_left = np.sum(control_no_early & (outcome == 'hit') & (trial_instruction == 'left'))
    correct_right = np.sum(control_no_early & (outcome == 'hit') & (trial_instruction == 'right'))
    
    if correct_rate < 0.65:
        print(f'  Skipping: correct rate {correct_rate:.3f} < 0.65')
        io.close()
        return None
    
    if correct_left < 50 or correct_right < 50:
        print(f'  Skipping: correct_left={correct_left}, correct_right={correct_right} (need >=50)')
        io.close()
        return None
    
    # Get brain region annotations for good units
    anno_names = nwb.units['anno_name'][:]
    brain_regions_raw = [anno_names[i] for i in good_indices]
    brain_regions_simplified = [simplify_brain_region(r) for r in brain_regions_raw]
    
    # Get spike times for good units (sorted)
    spike_times_all = nwb.units['spike_times'][:]
    spike_times_good = []
    for i in good_indices:
        st = np.array(spike_times_all[i], dtype=np.float64)
        spike_times_good.append(np.sort(st))
    
    # Get photostim start/stop times
    photostim_starts_all = beh_events.time_series['photostim_start_times'].timestamps[:]
    photostim_stops_all = beh_events.time_series['photostim_stop_times'].timestamps[:]
    
    # Get tongue tracking data
    tongue_ts_obj = nwb.acquisition['BehavioralTimeSeries'].time_series['Camera0_side_TongueTracking']
    tongue_data_all = tongue_ts_obj.data[:]
    tongue_ts_all = tongue_ts_obj.timestamps[:]
    
    t_load = time.time()
    print(f'  Data loaded in {t_load - t0:.1f}s ({n_good} good units, {len(valid_trial_indices)}/{n_trials_total} valid trials, rec=[{rec_start:.0f},{rec_end:.0f}]s)')
    
    # Compute firing rates only for valid trials
    valid_go_times = go_times[valid_trial_indices]
    fr_all = compute_firing_rates_session(spike_times_good, valid_go_times)
    
    t_fr = time.time()
    print(f'  Firing rates computed in {t_fr - t_load:.1f}s')
    
    # Build inputs and outputs for each valid trial
    neural_trials = fr_all  # Already computed for valid trials only
    input_trials = []
    output_choice = []
    output_outcome = []
    output_early_lick = []
    tongue_y_trials = []
    
    # Time from tone onset (same for all trials)
    time_from_tone = (BIN_CENTERS - TONE_ONSET_REL_GO).astype(np.float32)
    
    for local_idx, trial_idx in enumerate(valid_trial_indices):
        go_time = go_times[trial_idx]
        
        # Input 2: Photostimulation on/off
        trial_window_start = go_time + WINDOW_START
        trial_window_end = go_time + WINDOW_END
        
        stim_mask = (photostim_stops_all > trial_window_start) & (photostim_starts_all < trial_window_end)
        if np.any(stim_mask):
            trial_stim_starts = photostim_starts_all[stim_mask]
            trial_stim_stops = photostim_stops_all[stim_mask]
            photostim_ts = get_photostim_timeseries(trial_stim_starts, trial_stim_stops, go_time)
        else:
            photostim_ts = np.zeros(N_TIMEBINS, dtype=np.float32)
        
        input_data = np.stack([time_from_tone, photostim_ts], axis=0)
        input_trials.append(input_data)
        
        # Outputs
        choice = CHOICE_MAP.get(str(trial_instruction[trial_idx]), 0)
        out = OUTCOME_MAP.get(str(outcome[trial_idx]), 0)
        el = EARLY_LICK_MAP.get(str(early_lick[trial_idx]), 0)
        
        output_choice.append(choice)
        output_outcome.append(out)
        output_early_lick.append(el)
        
        # Tongue y-position
        tongue_y = get_tongue_y_for_trial(tongue_ts_all, tongue_data_all, go_time)
        tongue_y_trials.append(tongue_y)
    
    t_io = time.time()
    print(f'  Inputs/outputs built in {t_io - t_fr:.1f}s')
    
    # Discretize tongue y-position per session
    tongue_y_discrete = discretize_tongue_y(tongue_y_trials)
    
    # Build output arrays
    n_valid = len(valid_trial_indices)
    output_trials = []
    for t_idx in range(n_valid):
        out_arr = np.stack([
            np.full(N_TIMEBINS, output_choice[t_idx], dtype=np.int64),
            np.full(N_TIMEBINS, output_outcome[t_idx], dtype=np.int64),
            np.full(N_TIMEBINS, output_early_lick[t_idx], dtype=np.int64),
            tongue_y_discrete[t_idx].astype(np.int64),
        ], axis=0)
        output_trials.append(out_arr)
    
    io.close()
    
    t1 = time.time()
    print(f'  Total: {n_good} good units, {n_valid} trials, {t1-t0:.1f}s')
    
    if show_processing:
        plot_processing(nwb_file, neural_trials, input_trials, output_trials, session_idx)
    
    return {
        'neural': neural_trials,
        'input': input_trials,
        'output': output_trials,
        'subject_id': subject_id,
        'subject_desc': subject_desc,
        'brain_regions': brain_regions_simplified,
        'n_good': n_good,
        'n_trials_total': n_trials_total,
        'n_trials_valid': n_valid,
        'correct_rate': correct_rate,
        'nwb_file': nwb_file,
    }


def plot_processing(nwb_file, neural_trials, input_trials, output_trials, session_idx):
    """Plot processing steps for visual verification."""
    fig, axes = plt.subplots(5, 3, figsize=(20, 16))
    fig.suptitle(f'Processing: {os.path.basename(nwb_file)}', fontsize=14)
    
    n_show = min(3, len(neural_trials))
    trial_indices = np.linspace(0, len(neural_trials)-1, n_show, dtype=int)
    
    for col, t_idx in enumerate(trial_indices):
        ax = axes[0, col]
        neural = neural_trials[t_idx]
        n_show_neurons = min(50, neural.shape[0])
        ax.imshow(neural[:n_show_neurons, :], aspect='auto', cmap='viridis',
                  extent=[WINDOW_START, WINDOW_END, n_show_neurons, 0])
        ax.set_title(f'Trial {t_idx}: Neural ({neural.shape[0]} neurons)')
        ax.axvline(0, color='r', linestyle='--', alpha=0.7)
        ax.axvline(TONE_ONSET_REL_GO, color='b', linestyle='--', alpha=0.7)
        if col == 0: ax.set_ylabel('Neuron #')
        
        ax = axes[1, col]
        ax.plot(BIN_CENTERS, input_trials[t_idx][0, :], 'b-')
        ax.set_title('Input: Time from tone onset')
        ax.axvline(0, color='r', linestyle='--', alpha=0.7)
        if col == 0: ax.set_ylabel('Time (s)')
        
        ax = axes[2, col]
        ax.plot(BIN_CENTERS, input_trials[t_idx][1, :], 'g-')
        ax.set_title('Input: Photostim')
        ax.axvline(0, color='r', linestyle='--', alpha=0.7)
        ax.set_ylim(-0.1, 1.1)
        if col == 0: ax.set_ylabel('On/Off')
        
        ax = axes[3, col]
        ax.plot(BIN_CENTERS, output_trials[t_idx][0, :], 'r-', label='Choice')
        ax.plot(BIN_CENTERS, output_trials[t_idx][1, :], 'b-', label='Outcome')
        ax.plot(BIN_CENTERS, output_trials[t_idx][2, :], 'g-', label='Early lick')
        ax.set_title(f'C={output_trials[t_idx][0,0]:.0f}, O={output_trials[t_idx][1,0]:.0f}, EL={output_trials[t_idx][2,0]:.0f}')
        ax.legend(fontsize=8)
        if col == 0: ax.set_ylabel('Value')
        
        ax = axes[4, col]
        ax.plot(BIN_CENTERS, output_trials[t_idx][3, :], 'm-')
        ax.set_title('Output: Tongue Y (discretized)')
        ax.axvline(0, color='r', linestyle='--', alpha=0.7)
        ax.set_xlabel('Time from Go cue (s)')
        if col == 0: ax.set_ylabel('Category')
    
    plt.tight_layout()
    plt.savefig(f'processing_session_{session_idx}.png', dpi=150)
    plt.close()
    print(f'  Saved processing plot: processing_session_{session_idx}.png')


def main():
    parser = argparse.ArgumentParser(description='Convert MAP dataset to decoder format')
    parser.add_argument('outfile', type=str, help='Output pickle file path')
    parser.add_argument('--full', action='store_true', default=True)
    parser.add_argument('--sample', action='store_true', help='Process only 2 sessions')
    parser.add_argument('--show-processing', action='store_true')
    args = parser.parse_args()
    
    if args.sample:
        args.full = False
    
    print(f'=== MAP Dataset Conversion ===')
    print(f'Output: {args.outfile}, Mode: {"sample" if args.sample else "full"}')
    print(f'Bin: {BIN_WIDTH*1000:.0f}ms, Window: [{WINDOW_START}, {WINDOW_END}]s, Bins: {N_TIMEBINS}')
    print()
    
    nwb_files = sorted(glob.glob('data/sub-*/sub-*.nwb'))
    print(f'Found {len(nwb_files)} NWB files')
    
    if args.sample:
        nwb_files = [nwb_files[1], nwb_files[4]]
        print(f'Sample mode: processing {len(nwb_files)} files')
    
    all_sessions = []
    total_t0 = time.time()
    
    for i, nwb_file in enumerate(nwb_files):
        print(f'[{i+1}/{len(nwb_files)}] {nwb_file}')
        
        session_data = process_session(
            nwb_file,
            show_processing=args.show_processing and i < 2,
            session_idx=i
        )
        
        if session_data is not None:
            all_sessions.append(session_data)
        
        elapsed = time.time() - total_t0
        if i > 0:
            rate = elapsed / (i + 1)
            remaining = rate * (len(nwb_files) - i - 1)
            print(f'  Elapsed: {elapsed:.1f}s, Est. remaining: {remaining:.1f}s ({remaining/60:.1f}min)')
    
    print(f'\nTotal time: {time.time() - total_t0:.1f}s, Sessions: {len(all_sessions)}/{len(nwb_files)}')
    
    if len(all_sessions) == 0:
        print('ERROR: No sessions processed!')
        sys.exit(1)
    
    # Build output data structure
    print('\nBuilding output data structure...')
    
    all_subject_ids = [s['subject_id'] for s in all_sessions]
    unique_subjects = sorted(set(all_subject_ids))
    subject_to_idx = {s: i for i, s in enumerate(unique_subjects)}
    
    all_brain_regions_set = set()
    for s in all_sessions:
        all_brain_regions_set.update(s['brain_regions'])
    all_brain_regions = sorted(all_brain_regions_set)
    region_to_idx = {r: i for i, r in enumerate(all_brain_regions)}
    
    neural_list = [s['neural'] for s in all_sessions]
    input_list = [s['input'] for s in all_sessions]
    output_list = [s['output'] for s in all_sessions]
    subject_idx_list = [subject_to_idx[s['subject_id']] for s in all_sessions]
    brain_region_idx_list = [
        np.array([region_to_idx[r] for r in s['brain_regions']], dtype=np.int64)
        for s in all_sessions
    ]
    
    data = {
        'neural': neural_list,
        'input': input_list,
        'output': output_list,
        'subjects': unique_subjects,
        'subject_idx': np.array(subject_idx_list, dtype=np.int64),
        'brain_regions': all_brain_regions,
        'brain_region_idx': brain_region_idx_list,
        'input_names': ['time_from_tone_onset', 'photostimulation'],
        'output_names': ['choice', 'outcome', 'early_lick', 'tongue_y_position'],
        'output_values': [
            ['left', 'right'],
            ['ignore', 'miss', 'hit'],
            ['no_early', 'early'],
            ['low', 'mid', 'high'],
        ],
        'metadata': {
            'task_description': 'Auditory delayed response task: mice report instruction tone (3kHz or 12kHz) by licking left or right port after a 1.2s delay and go cue',
            'time_bin_size': BIN_WIDTH * 1000,
            'temporal_alignment_event': 'Go cue onset',
            'off_start': WINDOW_START,
            'off_end': WINDOW_END,
            'tone_onset_relative_to_go': TONE_ONSET_REL_GO,
            'n_sessions': len(all_sessions),
            'n_subjects': len(unique_subjects),
            'total_trials': sum(len(s['neural']) for s in all_sessions),
            'total_neurons': sum(s['n_good'] for s in all_sessions),
            'session_info': [
                {
                    'nwb_file': s['nwb_file'],
                    'subject_id': s['subject_id'],
                    'n_neurons': s['n_good'],
                    'n_trials': s['n_trials_valid'],
                    'n_trials_total': s['n_trials_total'],
                    'correct_rate': float(s['correct_rate']),
                }
                for s in all_sessions
            ],
        }
    }
    
    print(f'Saving to {args.outfile}...')
    with open(args.outfile, 'wb') as f:
        pickle.dump(data, f)
    
    file_size = os.path.getsize(args.outfile) / (1024**2)
    print(f'Saved {args.outfile} ({file_size:.1f} MB)')
    
    # Summary
    print(f'\n=== Summary ===')
    total_trials = sum(len(s['neural']) for s in all_sessions)
    total_neurons = sum(s['n_good'] for s in all_sessions)
    print(f'Sessions: {len(all_sessions)}, Subjects: {len(unique_subjects)}')
    print(f'Total trials: {total_trials}, Total neurons: {total_neurons}')
    print(f'Brain regions ({len(all_brain_regions)}): {all_brain_regions}')
    
    tc = [len(s['neural']) for s in all_sessions]
    nc = [s['n_good'] for s in all_sessions]
    print(f'Trials/session: mean={np.mean(tc):.1f}, min={np.min(tc)}, max={np.max(tc)}')
    print(f'Neurons/session: mean={np.mean(nc):.1f}, min={np.min(nc)}, max={np.max(nc)}')
    
    # Output distributions
    all_c, all_o, all_e = [], [], []
    for s_idx in range(len(all_sessions)):
        for t_idx in range(len(output_list[s_idx])):
            all_c.append(output_list[s_idx][t_idx][0, 0])
            all_o.append(output_list[s_idx][t_idx][1, 0])
            all_e.append(output_list[s_idx][t_idx][2, 0])
    all_c, all_o, all_e = np.array(all_c), np.array(all_o), np.array(all_e)
    
    print(f'\nOutput distributions:')
    for v, n in zip([0,1], ['left','right']): print(f'  Choice {n}: {np.mean(all_c==v):.3f}')
    for v, n in zip([0,1,2], ['ignore','miss','hit']): print(f'  Outcome {n}: {np.mean(all_o==v):.3f}')
    for v, n in zip([0,1], ['no_early','early']): print(f'  Early lick {n}: {np.mean(all_e==v):.3f}')


if __name__ == '__main__':
    main()
