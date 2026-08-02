#!/usr/bin/env python3
"""
Convert MAP dataset NWB files to decoder-compatible format.

Usage:
    python -u convert_data.py <outpicklefile> [--sample] [--full] [--show-processing]
"""

import os
import sys
import time
import json
import argparse
import pickle
import numpy as np
import h5py
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Constants
# ============================================================
BIN_WIDTH = 0.05  # 50 ms bins
WINDOW_START = -2.5  # seconds relative to go cue
WINDOW_END = 1.5  # seconds relative to go cue
N_TIMEBINS = int(round((WINDOW_END - WINDOW_START) / BIN_WIDTH))  # 80 bins

DATA_DIR = 'data'

# Session selection criteria from methods
MIN_CORRECT_RATE = 0.65
MIN_CORRECT_TRIALS_PER_SIDE = 50


def parse_args():
    parser = argparse.ArgumentParser(description='Convert MAP dataset to decoder format')
    parser.add_argument('output', type=str, help='Output pickle file path')
    parser.add_argument('--sample', action='store_true', help='Process only 2 sessions')
    parser.add_argument('--full', action='store_true', help='Process all sessions (default)')
    parser.add_argument('--show-processing', action='store_true', help='Show processing plots')
    return parser.parse_args()


def get_nwb_files():
    """Get all NWB file paths organized by subject."""
    subjects = sorted([d for d in os.listdir(DATA_DIR) if d.startswith('sub-')])
    all_files = []
    for subj in subjects:
        subj_dir = os.path.join(DATA_DIR, subj)
        nwb_files = sorted([f for f in os.listdir(subj_dir) if f.endswith('.nwb')])
        for nwb_file in nwb_files:
            all_files.append({
                'subject': subj,
                'path': os.path.join(subj_dir, nwb_file),
                'filename': nwb_file
            })
    return all_files


def extract_brain_region(location_json_bytes):
    """Extract brain region from electrode location JSON string."""
    try:
        loc_str = location_json_bytes.decode('utf-8') if isinstance(location_json_bytes, bytes) else location_json_bytes
        loc_dict = json.loads(loc_str)
        return loc_dict.get('brain_regions', 'unknown')
    except (json.JSONDecodeError, AttributeError):
        return 'unknown'


def compute_firing_rates_fast(spike_times_flat, spike_times_index, go_cue_times,
                               good_indices, trial_indices,
                               window_start=WINDOW_START, window_end=WINDOW_END,
                               bin_width=BIN_WIDTH):
    """
    Compute firing rates for good units across selected trials.
    Optimized: pre-sort spike times per unit, use searchsorted.
    """
    n_bins = int(round((window_end - window_start) / bin_width))
    bin_edges = np.linspace(window_start, window_end, n_bins + 1)
    n_good = len(good_indices)
    
    # Pre-compute spike time ranges for each unit
    starts = np.zeros(len(spike_times_index), dtype=np.int64)
    starts[1:] = spike_times_index[:-1]
    ends = spike_times_index.astype(np.int64)
    
    # Pre-extract spike times for good units
    good_spike_times = []
    for unit_idx in good_indices:
        good_spike_times.append(spike_times_flat[starts[unit_idx]:ends[unit_idx]])
    
    firing_rates_list = []
    
    for trial_idx in trial_indices:
        go_time = go_cue_times[trial_idx]
        fr_trial = np.zeros((n_good, n_bins), dtype=np.float32)
        
        abs_start = go_time + window_start
        abs_end = go_time + window_end
        
        for i, unit_spikes in enumerate(good_spike_times):
            # Use searchsorted to find spikes in window
            idx_lo = np.searchsorted(unit_spikes, abs_start)
            idx_hi = np.searchsorted(unit_spikes, abs_end)
            
            if idx_hi > idx_lo:
                aligned = unit_spikes[idx_lo:idx_hi] - go_time
                counts = np.histogram(aligned, bins=bin_edges)[0]
                fr_trial[i, :] = counts / bin_width
        
        firing_rates_list.append(fr_trial)
    
    return firing_rates_list


def get_tongue_y_for_trials(tongue_data, tongue_timestamps, go_times, trial_indices,
                            window_start=WINDOW_START, window_end=WINDOW_END,
                            bin_width=BIN_WIDTH):
    """
    Get tongue y-position for multiple trials, binned to match neural data.
    """
    n_bins = int(round((window_end - window_start) / bin_width))
    bin_edges = np.linspace(window_start, window_end, n_bins + 1)
    
    tongue_y_trials = []
    all_tongue_y = []
    
    for trial_idx in trial_indices:
        go_time = go_times[trial_idx]
        abs_start = go_time + window_start - bin_width
        abs_end = go_time + window_end + bin_width
        
        idx_start = np.searchsorted(tongue_timestamps, abs_start)
        idx_end = np.searchsorted(tongue_timestamps, abs_end)
        
        tongue_y_binned = np.full(n_bins, np.nan)
        
        if idx_start < idx_end:
            ts_slice = tongue_timestamps[idx_start:idx_end] - go_time
            y_slice = tongue_data[idx_start:idx_end, 1]
            lk_slice = tongue_data[idx_start:idx_end, 2]
            high_conf = lk_slice > 0.1
            
            for b in range(n_bins):
                bin_mask = (ts_slice >= bin_edges[b]) & (ts_slice < bin_edges[b+1]) & high_conf
                n_in_bin = np.sum(bin_mask)
                if n_in_bin > 0:
                    tongue_y_binned[b] = np.mean(y_slice[bin_mask])
        
        tongue_y_trials.append(tongue_y_binned)
        valid_y = tongue_y_binned[~np.isnan(tongue_y_binned)]
        if len(valid_y) > 0:
            all_tongue_y.extend(valid_y.tolist())
    
    return tongue_y_trials, all_tongue_y


def process_session(nwb_path, subject_id, show_processing=False, session_idx=0):
    """
    Process a single NWB session file.
    """
    t0 = time.time()
    print(f"  Loading {os.path.basename(nwb_path)}...")
    
    f = h5py.File(nwb_path, 'r')
    
    # ---- Extract trial info ----
    trials = f['intervals']['trials']
    n_trials_total = len(trials['id'])
    
    trial_instruction = trials['trial_instruction'][:]
    outcome = trials['outcome'][:]
    early_lick = trials['early_lick'][:]
    auto_water = trials['auto_water'][:]
    free_water = trials['free_water'][:]
    photostim_onset_trial = trials['photostim_onset'][:]
    photostim_duration_trial = trials['photostim_duration'][:]
    trial_start_times = trials['start_time'][:]
    
    go_times = f['acquisition']['BehavioralEvents']['go_start_times']['timestamps'][:]
    sample_starts_all = f['acquisition']['BehavioralEvents']['sample_start_times']['timestamps'][:]
    
    # ---- Determine valid neural trials ----
    n_recorded_trials = f['units']['is_good_trials'].shape[1]
    
    spike_times_flat = f['units']['spike_times'][:]
    spike_times_index = f['units']['spike_times_index'][:]
    
    if len(spike_times_flat) > 0:
        max_spike_time = spike_times_flat.max()
        min_spike_time = spike_times_flat.min()
    else:
        f.close()
        return None
    
    # Trial valid if within recorded trials AND window fits in recording
    neural_valid = np.zeros(n_trials_total, dtype=bool)
    for t_idx in range(min(n_trials_total, n_recorded_trials)):
        go = go_times[t_idx]
        if (go + WINDOW_START >= min_spike_time - 1.0 and
            go + WINDOW_END <= max_spike_time + 1.0):
            neural_valid[t_idx] = True
    
    # ---- Trial filtering ----
    trial_mask = (auto_water == 0) & (free_water == 0) & neural_valid
    trial_indices = np.where(trial_mask)[0]
    n_selected = len(trial_indices)
    
    print(f"    Trials: {n_trials_total} total, {n_recorded_trials} recorded, {n_selected} selected")
    
    if n_selected < 2:
        print(f"    Skipping: too few valid trials")
        f.close()
        return None
    
    # ---- Session selection criteria ----
    is_control = (photostim_onset_trial == b'N/A') & trial_mask
    is_not_early = early_lick == b'no early'
    control_non_early = is_control & is_not_early
    
    hits = np.sum(outcome[control_non_early] == b'hit')
    misses = np.sum(outcome[control_non_early] == b'miss')
    
    if (hits + misses) == 0:
        print(f"    Skipping: no valid control trials")
        f.close()
        return None
    
    correct_rate = hits / (hits + misses)
    correct_left = np.sum((outcome == b'hit') & (trial_instruction == b'left') & control_non_early)
    correct_right = np.sum((outcome == b'hit') & (trial_instruction == b'right') & control_non_early)
    
    print(f"    Correct rate: {correct_rate:.2%}, correct L: {correct_left}, correct R: {correct_right}")
    
    if correct_rate < MIN_CORRECT_RATE:
        print(f"    Skipping: correct rate too low")
        f.close()
        return None
    
    if correct_left < MIN_CORRECT_TRIALS_PER_SIDE or correct_right < MIN_CORRECT_TRIALS_PER_SIDE:
        print(f"    Skipping: insufficient correct trials per side")
        f.close()
        return None
    
    # ---- Unit filtering ----
    classification = f['units']['classification'][:]
    good_mask = classification == b'good'
    n_good = int(np.sum(good_mask))
    n_total = len(classification)
    good_indices = np.where(good_mask)[0]
    
    print(f"    Units: {n_total} total, {n_good} good")
    
    if n_good == 0:
        print(f"    Skipping: no good units")
        f.close()
        return None
    
    # ---- Brain regions ----
    electrodes_idx = f['units']['electrodes'][:]
    electrodes_index = f['units']['electrodes_index'][:]
    electrode_locations = f['general']['extracellular_ephys']['electrodes']['location'][:]
    
    unit_regions = []
    for i in range(n_total):
        start = 0 if i == 0 else int(electrodes_index[i-1])
        elec_idx = electrodes_idx[start]
        region = extract_brain_region(electrode_locations[elec_idx])
        unit_regions.append(region)
    
    good_regions = [unit_regions[i] for i in range(n_total) if good_mask[i]]
    
    t1 = time.time()
    print(f"    Data loading: {t1-t0:.1f}s")
    
    # ---- Compute firing rates ----
    firing_rates = compute_firing_rates_fast(
        spike_times_flat, spike_times_index, go_times,
        good_indices, trial_indices
    )
    
    t2 = time.time()
    print(f"    Firing rate computation: {t2-t1:.1f}s")
    
    # ---- Construct inputs ----
    n_bins = N_TIMEBINS
    bin_centers = np.linspace(WINDOW_START + BIN_WIDTH/2, WINDOW_END - BIN_WIDTH/2, n_bins)
    
    inputs_list = []
    for trial_idx in trial_indices:
        go_time = go_times[trial_idx]
        
        # Input 0: Time from tone onset
        ss_idx = np.searchsorted(sample_starts_all, go_time, side='right') - 1
        if ss_idx >= 0:
            tone_onset_rel = sample_starts_all[ss_idx] - go_time
        else:
            tone_onset_rel = -1.85
        time_from_tone = bin_centers - tone_onset_rel
        
        # Input 1: Photostim on/off
        photostim_binary = np.zeros(n_bins, dtype=np.float32)
        ps_onset_val = photostim_onset_trial[trial_idx]
        if ps_onset_val != b'N/A':
            ps_onset_float = float(ps_onset_val)
            ps_dur_float = float(photostim_duration_trial[trial_idx])
            trial_start = trial_start_times[trial_idx]
            ps_rel_start = (trial_start + ps_onset_float) - go_time
            ps_rel_end = ps_rel_start + ps_dur_float
            photostim_binary = ((bin_centers >= ps_rel_start) & (bin_centers < ps_rel_end)).astype(np.float32)
        
        input_trial = np.stack([time_from_tone.astype(np.float32), photostim_binary], axis=0)
        inputs_list.append(input_trial)
    
    t3 = time.time()
    print(f"    Input construction: {t3-t2:.1f}s")
    
    # ---- Construct outputs ----
    has_tongue = 'Camera0_side_TongueTracking' in f['acquisition']['BehavioralTimeSeries']
    
    if has_tongue:
        tongue_data = f['acquisition']['BehavioralTimeSeries']['Camera0_side_TongueTracking']['data'][:]
        tongue_timestamps = f['acquisition']['BehavioralTimeSeries']['Camera0_side_TongueTracking']['timestamps'][:]
        tongue_y_trials, all_tongue_y = get_tongue_y_for_trials(
            tongue_data, tongue_timestamps, go_times, trial_indices
        )
    else:
        tongue_y_trials = [np.full(n_bins, np.nan) for _ in trial_indices]
        all_tongue_y = []
    
    if len(all_tongue_y) > 0:
        all_tongue_y_arr = np.array(all_tongue_y)
        p40 = np.percentile(all_tongue_y_arr, 40)
        p60 = np.percentile(all_tongue_y_arr, 60)
    else:
        p40, p60 = 0, 0
    
    t4 = time.time()
    print(f"    Tongue tracking: {t4-t3:.1f}s")
    
    outputs_list = []
    for i, trial_idx in enumerate(trial_indices):
        choice = 0 if trial_instruction[trial_idx] == b'left' else 1
        
        out = outcome[trial_idx]
        if out == b'ignore':
            outcome_val = 0
        elif out == b'miss':
            outcome_val = 1
        elif out == b'hit':
            outcome_val = 2
        else:
            outcome_val = 0
        
        early = 1 if early_lick[trial_idx] == b'early' else 0
        
        tongue_y = tongue_y_trials[i]
        tongue_y_disc = np.ones(n_bins, dtype=np.int64)
        valid_mask = ~np.isnan(tongue_y)
        if np.any(valid_mask):
            tongue_y_disc[valid_mask & (tongue_y < p40)] = 0
            tongue_y_disc[valid_mask & (tongue_y >= p40) & (tongue_y <= p60)] = 1
            tongue_y_disc[valid_mask & (tongue_y > p60)] = 2
        
        output_trial = np.zeros((4, n_bins), dtype=np.int64)
        output_trial[0, :] = choice
        output_trial[1, :] = outcome_val
        output_trial[2, :] = early
        output_trial[3, :] = tongue_y_disc
        
        outputs_list.append(output_trial)
    
    t5 = time.time()
    print(f"    Output construction: {t5-t4:.1f}s")
    
    # ---- Plotting ----
    if show_processing and session_idx < 2:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            sess_name = os.path.basename(nwb_path).replace('.nwb', '')
            fig, axes = plt.subplots(4, 2, figsize=(16, 16))
            fig.suptitle(f'Processing: {sess_name}\n'
                        f'{n_good} good units, {n_selected} trials, correct={correct_rate:.1%}', fontsize=14)
            
            bc = bin_centers
            
            if len(firing_rates) > 0:
                ax = axes[0, 0]
                fr0 = firing_rates[0]
                n_show = min(50, fr0.shape[0])
                ax.imshow(fr0[:n_show, :], aspect='auto',
                         extent=[WINDOW_START, WINDOW_END, n_show, 0], cmap='hot')
                ax.axvline(0, color='cyan', linestyle='--', label='Go cue')
                ax.set_xlabel('Time from go cue (s)')
                ax.set_ylabel('Neuron')
                ax.set_title(f'FR (trial 0, first {n_show} neurons)')
                ax.legend()
            
            if len(firing_rates) > 0:
                ax = axes[0, 1]
                n_pt = min(20, len(firing_rates))
                mean_fr = np.mean(np.stack([fr.mean(axis=0) for fr in firing_rates[:n_pt]]), axis=0)
                ax.plot(bc, mean_fr)
                ax.axvline(0, color='r', linestyle='--', label='Go cue')
                ax.axvline(-1.85, color='g', linestyle='--', label='~Tone onset')
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Mean FR (Hz)')
                ax.set_title('Mean firing rate')
                ax.legend()
            
            ax = axes[1, 0]
            for t_idx in range(min(5, len(inputs_list))):
                ax.plot(bc, inputs_list[t_idx][0, :], alpha=0.5)
            ax.axvline(0, color='r', linestyle='--')
            ax.set_title('Input: Time from tone onset')
            
            ax = axes[1, 1]
            sc = 0
            for t_idx in range(len(inputs_list)):
                if np.any(inputs_list[t_idx][1, :] > 0):
                    ax.plot(bc, inputs_list[t_idx][1, :] + sc * 0.05, alpha=0.5)
                    sc += 1
                    if sc >= 10: break
            ax.axvline(0, color='r', linestyle='--')
            ax.set_title(f'Input: Photostim ({sc} stim trials)')
            
            ax = axes[2, 0]
            choices = [int(o[0, 0]) for o in outputs_list]
            ax.bar(['Left', 'Right'], [choices.count(0), choices.count(1)])
            ax.set_title('Choice distribution')
            
            ax = axes[2, 1]
            ov = [int(o[1, 0]) for o in outputs_list]
            ax.bar(['Ignore', 'Miss', 'Hit'], [ov.count(0), ov.count(1), ov.count(2)])
            ax.set_title('Outcome distribution')
            
            ax = axes[3, 0]
            for t_idx in range(min(10, len(tongue_y_trials))):
                ax.plot(bc, tongue_y_trials[t_idx], alpha=0.3)
            ax.axhline(p40, color='r', linestyle='--', label=f'p40={p40:.1f}')
            ax.axhline(p60, color='b', linestyle='--', label=f'p60={p60:.1f}')
            ax.set_title('Tongue y-position')
            ax.legend()
            
            ax = axes[3, 1]
            for t_idx in range(min(10, len(outputs_list))):
                ax.plot(bc, outputs_list[t_idx][3, :], alpha=0.3)
            ax.set_title('Output: Tongue y discretized')
            
            plt.tight_layout()
            plt.savefig(f'processing_{sess_name}.png', dpi=100)
            plt.close()
            print(f"    Saved processing plot")
        except Exception as e:
            print(f"    Warning: Could not create plot: {e}")
    
    f.close()
    
    t6 = time.time()
    print(f"    Total session time: {t6-t0:.1f}s")
    
    return {
        'neural': firing_rates,
        'input': inputs_list,
        'output': outputs_list,
        'brain_regions': good_regions,
        'subject': subject_id,
        'n_good': n_good,
        'n_trials': len(firing_rates),
        'correct_rate': correct_rate,
    }


def main():
    args = parse_args()
    t_start = time.time()
    
    nwb_files = get_nwb_files()
    print(f"Found {len(nwb_files)} NWB files from {len(set(f['subject'] for f in nwb_files))} subjects")
    
    if args.sample:
        selected = []
        seen_subjects = set()
        for nf in nwb_files:
            if nf['subject'] not in seen_subjects:
                seen_subjects.add(nf['subject'])
                subj_files = [x for x in nwb_files if x['subject'] == nf['subject']]
                selected.append(subj_files[-1])
                if len(selected) >= 2:
                    break
        nwb_files = selected
        print(f"Sample mode: processing {len(nwb_files)} sessions")
    
    all_neural = []
    all_input = []
    all_output = []
    all_brain_region_idx_raw = []
    all_subjects = []
    all_subject_idx = []
    all_brain_regions_set = set()
    
    session_count = 0
    total_neurons = 0
    total_trials = 0
    skipped = 0
    
    for i, nwb_info in enumerate(nwb_files):
        print(f"\nProcessing session {i+1}/{len(nwb_files)}: {nwb_info['filename']}")
        
        result = process_session(
            nwb_info['path'],
            nwb_info['subject'],
            show_processing=args.show_processing,
            session_idx=session_count
        )
        
        if result is None:
            skipped += 1
            continue
        
        if result['n_trials'] < 2:
            print(f"    Skipping: fewer than 2 trials")
            skipped += 1
            continue
        
        all_neural.append(result['neural'])
        all_input.append(result['input'])
        all_output.append(result['output'])
        
        for r in result['brain_regions']:
            all_brain_regions_set.add(r)
        all_brain_region_idx_raw.append(result['brain_regions'])
        
        subj = result['subject']
        if subj not in all_subjects:
            all_subjects.append(subj)
        all_subject_idx.append(all_subjects.index(subj))
        
        session_count += 1
        total_neurons += result['n_good']
        total_trials += result['n_trials']
        
        elapsed = time.time() - t_start
        est_total = elapsed / (i + 1) * len(nwb_files)
        print(f"    Session {session_count}: {result['n_good']} neurons, {result['n_trials']} trials  [{elapsed:.0f}s elapsed, ~{est_total:.0f}s total est]")
    
    print(f"\n{'='*60}")
    print(f"Processed {session_count} sessions, skipped {skipped}")
    print(f"Total neurons: {total_neurons}")
    print(f"Total trials: {total_trials}")
    print(f"Subjects: {len(all_subjects)}")
    
    if session_count == 0:
        print("ERROR: No sessions passed criteria!")
        sys.exit(1)
    
    brain_regions = sorted(list(all_brain_regions_set))
    brain_region_idx = []
    for session_regions in all_brain_region_idx_raw:
        idx = np.array([brain_regions.index(r) for r in session_regions], dtype=np.int64)
        brain_region_idx.append(idx)
    
    data = {
        'neural': all_neural,
        'input': all_input,
        'output': all_output,
        
        'subjects': all_subjects,
        'subject_idx': np.array(all_subject_idx, dtype=np.int64),
        
        'brain_regions': brain_regions,
        'brain_region_idx': brain_region_idx,
        
        'input_names': ['time_from_tone_onset', 'photostim_on'],
        'output_names': ['choice', 'outcome', 'early_lick', 'tongue_y_position'],
        'output_values': [
            ['left', 'right'],
            ['ignore', 'miss', 'hit'],
            ['no', 'yes'],
            ['below_p40', 'p40_to_p60', 'above_p60'],
        ],
        
        'metadata': {
            'task_description': 'Auditory delayed response task: mice hear tones during sample epoch, maintain memory during delay, then lick left or right after go cue',
            'time_bin_size': BIN_WIDTH * 1000,
            'temporal_alignment_event': 'Go cue onset',
            'off_start': WINDOW_START,
            'off_end': WINDOW_END,
            'dataset': 'MAP (Mesoscale Activity Project) - DANDI:000363',
            'n_sessions': session_count,
            'n_subjects': len(all_subjects),
            'total_neurons': total_neurons,
            'total_trials': total_trials,
        }
    }
    
    print(f"\nSaving to {args.output}...")
    with open(args.output, 'wb') as f:
        pickle.dump(data, f)
    
    file_size = os.path.getsize(args.output) / (1024**2)
    print(f"Saved {args.output} ({file_size:.1f} MB)")
    
    t_end = time.time()
    print(f"Total time: {t_end - t_start:.1f}s")


if __name__ == '__main__':
    main()
