#!/usr/bin/env python3
"""Convert MAP dataset NWB files to decoder-compatible format.

Usage:
    python -u convert_data.py <output_pickle> [--sample] [--full] [--show-processing]
"""

import os
import sys
import time
import glob
import pickle
import argparse
import numpy as np
import warnings
warnings.filterwarnings('ignore')

try:
    import pynwb
except ImportError:
    os.system('pip install pynwb')
    import pynwb

# ============================================================
# Constants
# ============================================================
DATA_DIR = '/app/data'
BIN_WIDTH = 0.050  # 50 ms bins
ALIGN_START = -2.5  # seconds before go cue
ALIGN_END = 1.5    # seconds after go cue
N_TIMEBINS = int((ALIGN_END - ALIGN_START) / BIN_WIDTH)  # 80 bins
BIN_CENTERS = ALIGN_START + BIN_WIDTH/2 + np.arange(N_TIMEBINS) * BIN_WIDTH

# Tongue tracking
TONGUE_LIKELIHOOD_THRESHOLD = 0.9  # DLC confidence threshold


def get_nwb_files(data_dir):
    """Get all NWB files organized by subject."""
    subjects = sorted([d for d in os.listdir(data_dir) if d.startswith('sub-')])
    all_files = []
    for subj in subjects:
        subj_dir = os.path.join(data_dir, subj)
        nwb_files = sorted(glob.glob(os.path.join(subj_dir, '*.nwb')))
        for f in nwb_files:
            all_files.append((subj, f))
    return all_files


def compute_firing_rates_vectorized(spike_times_list, go_cue_times, n_neurons,
                                     bin_width=BIN_WIDTH, align_start=ALIGN_START,
                                     align_end=ALIGN_END):
    """Compute firing rates for all neurons across all trials."""
    n_trials = len(go_cue_times)
    n_bins = int((align_end - align_start) / bin_width)
    bin_offsets = align_start + np.arange(n_bins + 1) * bin_width
    
    fr_all = []
    for t in range(n_trials):
        fr = np.zeros((n_neurons, n_bins), dtype=np.float32)
        bin_edges = go_cue_times[t] + bin_offsets
        
        for i, spikes in enumerate(spike_times_list):
            if len(spikes) == 0:
                continue
            left = np.searchsorted(spikes, bin_edges[0])
            right = np.searchsorted(spikes, bin_edges[-1])
            if left >= right:
                continue
            spikes_window = spikes[left:right]
            counts, _ = np.histogram(spikes_window, bins=bin_edges)
            fr[i] = counts / bin_width
        
        fr_all.append(fr)
    
    return fr_all


def get_tongue_y_for_trial(tongue_ts, tongue_data, go_cue_time,
                           bin_width=BIN_WIDTH, align_start=ALIGN_START,
                           align_end=ALIGN_END):
    """Extract tongue y position for a trial, binned to match neural data."""
    n_bins = int((align_end - align_start) / bin_width)
    bin_edges = go_cue_time + align_start + np.arange(n_bins + 1) * bin_width
    
    tongue_y_binned = np.full(n_bins, np.nan, dtype=np.float32)
    visible_binned = np.zeros(n_bins, dtype=np.float32)
    
    left_idx = np.searchsorted(tongue_ts, bin_edges[0])
    right_idx = np.searchsorted(tongue_ts, bin_edges[-1])
    
    if left_idx >= right_idx:
        return tongue_y_binned, visible_binned
    
    ts_window = tongue_ts[left_idx:right_idx]
    data_window = tongue_data[left_idx:right_idx]
    
    bin_idx = np.digitize(ts_window, bin_edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        frames = data_window[mask]
        visible = frames[:, 2] >= TONGUE_LIKELIHOOD_THRESHOLD
        visible_binned[b] = visible.mean()
        if visible.sum() > 0:
            tongue_y_binned[b] = frames[visible, 1].mean()
    
    return tongue_y_binned, visible_binned


def determine_choice(trial_instruction, outcome):
    """Determine lick direction choice.
    Returns: 0=left, 1=right, 2=no_lick
    """
    if outcome == 'ignore':
        return 2  # no lick
    elif outcome == 'hit':
        return 0 if trial_instruction == 'left' else 1
    elif outcome == 'miss':
        return 1 if trial_instruction == 'left' else 0
    return 2


def get_recording_time_range(units, good_indices):
    """Get the recording time range from obs_intervals of good units."""
    obs = np.array(units['obs_intervals'][good_indices[0]])
    min_time = obs[0, 0]
    max_time = obs[-1, 1]
    return min_time, max_time


def process_session(nwb_path, show_processing=False, session_idx=0):
    """Process a single NWB session file."""
    t0 = time.time()
    
    with pynwb.NWBHDF5IO(nwb_path, 'r') as io:
        nwb = io.read()
        
        # === Extract basic info ===
        subject_id = nwb.subject.subject_id
        session_id = nwb.identifier
        
        # === Get trials info ===
        trials = nwb.trials
        n_trials = len(trials)
        trial_instruction = trials['trial_instruction'][:]
        outcome = trials['outcome'][:]
        early_lick = trials['early_lick'][:]
        auto_water = trials['auto_water'][:]
        free_water = trials['free_water'][:]
        photostim_power_str = trials['photostim_power'][:]
        
        # === Get behavioral events ===
        be = nwb.acquisition['BehavioralEvents']
        go_times = be.time_series['go_start_times'].timestamps[:]
        sample_start_times = be.time_series['sample_start_times'].timestamps[:]
        
        has_photostim_events = 'photostim_start_times' in be.time_series
        if has_photostim_events:
            photostim_event_starts = be.time_series['photostim_start_times'].timestamps[:]
            photostim_event_stops = be.time_series['photostim_stop_times'].timestamps[:]
        
        # === Get units (good only) ===
        t_units_start = time.time()
        units = nwb.units
        classification = units['classification'][:]
        good_mask = classification == 'good'
        good_indices = np.where(good_mask)[0]
        n_good = len(good_indices)
        
        if n_good == 0:
            print(f"  Session {session_id}: No good units, skipping")
            return None
        
        # === Get recording time range from obs_intervals ===
        rec_min, rec_max = get_recording_time_range(units, good_indices)
        
        # Brain region annotations
        anno_names = units['anno_name'][:]
        good_anno_names = anno_names[good_indices]
        
        # Read spike times for good units
        good_spike_times = []
        for idx in good_indices:
            st = units['spike_times'][idx]
            good_spike_times.append(np.sort(st))
        t_units_end = time.time()
        
        # === Filter trials for data extraction ===
        # 1. Exclude auto_water and free_water
        # 2. Exclude trials outside recording range
        valid_trial_mask = (auto_water == 0) & (free_water == 0)
        for i in range(n_trials):
            if valid_trial_mask[i]:
                window_start = go_times[i] + ALIGN_START
                window_end = go_times[i] + ALIGN_END
                if window_start < rec_min or window_end > rec_max:
                    valid_trial_mask[i] = False
        
        valid_trial_indices = np.where(valid_trial_mask)[0]
        n_valid_trials = len(valid_trial_indices)
        
        if n_valid_trials < 2:
            print(f"  Session {session_id}: Only {n_valid_trials} valid trials within recording, skipping")
            return None
        
        # Compute correct rate for reporting (not filtering)
        control_mask_all = (photostim_power_str == 'N/A') & (early_lick == 'no early')
        ctrl_out = outcome[control_mask_all]
        n_hit = (ctrl_out == 'hit').sum()
        n_miss = (ctrl_out == 'miss').sum()
        correct_rate = n_hit / (n_hit + n_miss) if (n_hit + n_miss) > 0 else 0
        
        print(f"    Reading units: {t_units_end - t_units_start:.1f}s")
        
        # === Get tongue tracking data ===
        t_tongue_start = time.time()
        bts = nwb.acquisition['BehavioralTimeSeries']
        has_tongue = 'Camera0_side_TongueTracking' in bts.time_series
        if has_tongue:
            tongue_ts_obj = bts.time_series['Camera0_side_TongueTracking']
            tongue_data_all = tongue_ts_obj.data[:]
            tongue_ts_all = tongue_ts_obj.timestamps[:]
        t_tongue_end = time.time()
        print(f"    Reading tongue: {t_tongue_end - t_tongue_start:.1f}s")
        
        # === Compute firing rates ===
        t_fr_start = time.time()
        valid_go_times = go_times[valid_trial_indices]
        neural_trials = compute_firing_rates_vectorized(
            good_spike_times, valid_go_times, n_good)
        t_fr_end = time.time()
        print(f"    Computing firing rates: {t_fr_end - t_fr_start:.1f}s")
        
        # === Process inputs and outputs ===
        t_io_start = time.time()
        input_trials = []
        output_per_trial = []
        tongue_y_per_trial = []
        tongue_visible_per_trial = []
        tongue_y_all_visible = []
        
        for i, trial_idx in enumerate(valid_trial_indices):
            go_time = go_times[trial_idx]
            
            # --- Input 0: Time from tone onset ---
            prev_samples = sample_start_times[sample_start_times < go_time + 0.01]
            if len(prev_samples) > 0:
                tone_onset = prev_samples[-1]
            else:
                tone_onset = go_time - 1.85
            time_from_tone = (go_time + BIN_CENTERS) - tone_onset
            
            # --- Input 1: Photostim on/off ---
            photostim_binary = np.zeros(N_TIMEBINS, dtype=np.float32)
            if has_photostim_events and photostim_power_str[trial_idx] != 'N/A':
                trial_window_start = go_time + ALIGN_START
                trial_window_end = go_time + ALIGN_END
                for ps_idx in range(len(photostim_event_starts)):
                    ps_start = photostim_event_starts[ps_idx]
                    ps_stop = photostim_event_stops[ps_idx]
                    if ps_stop < trial_window_start or ps_start > trial_window_end:
                        continue
                    for b in range(N_TIMEBINS):
                        bin_start_abs = go_time + ALIGN_START + b * BIN_WIDTH
                        bin_end_abs = bin_start_abs + BIN_WIDTH
                        if bin_start_abs < ps_stop and bin_end_abs > ps_start:
                            photostim_binary[b] = 1.0
            
            input_data = np.stack([time_from_tone.astype(np.float32), photostim_binary], axis=0)
            input_trials.append(input_data)
            
            # --- Outputs ---
            choice = determine_choice(trial_instruction[trial_idx], outcome[trial_idx])
            outcome_map = {'ignore': 0, 'miss': 1, 'hit': 2}
            outcome_val = outcome_map.get(outcome[trial_idx], 0)
            early_lick_val = 1 if early_lick[trial_idx] == 'early' else 0
            output_per_trial.append(np.array([choice, outcome_val, early_lick_val], dtype=np.int64))
            
            # --- Tongue y ---
            if has_tongue:
                tongue_y, tongue_vis = get_tongue_y_for_trial(
                    tongue_ts_all, tongue_data_all, go_time)
                tongue_y_per_trial.append(tongue_y)
                tongue_visible_per_trial.append(tongue_vis)
                visible_mask = tongue_vis >= 0.5
                if visible_mask.sum() > 0:
                    tongue_y_all_visible.extend(tongue_y[visible_mask].tolist())
            else:
                tongue_y_per_trial.append(np.full(N_TIMEBINS, np.nan))
                tongue_visible_per_trial.append(np.zeros(N_TIMEBINS))
        
        # === Discretize tongue y position ===
        if len(tongue_y_all_visible) > 0:
            tongue_y_arr = np.array(tongue_y_all_visible)
            p40 = np.percentile(tongue_y_arr, 40)
            p60 = np.percentile(tongue_y_arr, 60)
        else:
            p40 = 0
            p60 = 0
        
        output_trials = []
        for i in range(len(output_per_trial)):
            tongue_y = tongue_y_per_trial[i]
            tongue_vis = tongue_visible_per_trial[i]
            
            tongue_discrete = np.full(N_TIMEBINS, 3, dtype=np.int64)
            visible_mask = tongue_vis >= 0.5
            if visible_mask.sum() > 0 and len(tongue_y_all_visible) > 0:
                ty = tongue_y[visible_mask]
                tongue_discrete[visible_mask] = np.where(ty < p40, 0,
                                                         np.where(ty <= p60, 1, 2))
            
            per_trial = output_per_trial[i]
            per_trial_expanded = np.repeat(per_trial[:, np.newaxis], N_TIMEBINS, axis=1)
            tongue_expanded = tongue_discrete[np.newaxis, :]
            output_trials.append(
                np.concatenate([per_trial_expanded, tongue_expanded], axis=0).astype(np.int64)
            )
        
        t_io_end = time.time()
        print(f"    Processing I/O: {t_io_end - t_io_start:.1f}s")
    
    t1 = time.time()
    print(f"  Session {session_id}: {n_good} good units, {n_valid_trials} valid trials "
          f"(rec: {rec_min:.0f}-{rec_max:.0f}s), correct_rate={correct_rate:.2%}, time={t1-t0:.1f}s")
    
    if show_processing:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(4, 2, figsize=(16, 16))
            fig.suptitle(f'Session: {session_id}', fontsize=14)
            
            ax = axes[0, 0]
            for t_idx in range(min(5, len(neural_trials))):
                ax.plot(BIN_CENTERS, neural_trials[t_idx].mean(axis=0), alpha=0.5)
            ax.set_xlabel('Time from go cue (s)')
            ax.set_ylabel('Mean firing rate (Hz)')
            ax.set_title('Mean firing rate across neurons')
            ax.axvline(0, color='r', linestyle='--', label='Go cue')
            ax.legend()
            
            ax = axes[0, 1]
            for t_idx in range(min(5, len(input_trials))):
                ax.plot(BIN_CENTERS, input_trials[t_idx][0], alpha=0.5)
            ax.set_xlabel('Time from go cue (s)')
            ax.set_ylabel('Time from tone onset (s)')
            ax.set_title('Input: Time from tone onset')
            ax.axvline(0, color='r', linestyle='--')
            
            ax = axes[1, 0]
            ps_trials = [i for i in range(len(input_trials)) if input_trials[i][1].max() > 0]
            if ps_trials:
                for t_idx in ps_trials[:3]:
                    ax.plot(BIN_CENTERS, input_trials[t_idx][1], alpha=0.7)
            ax.set_xlabel('Time from go cue (s)')
            ax.set_ylabel('Photostim on')
            ax.set_title(f'Input: Photostim ({len(ps_trials)} trials with stim)')
            ax.axvline(0, color='r', linestyle='--')
            
            ax = axes[1, 1]
            choices = [output_trials[i][0, 0] for i in range(len(output_trials))]
            ax.bar(['L', 'R', 'NoLick'], [choices.count(0), choices.count(1), choices.count(2)])
            ax.set_title('Choice distribution')
            
            ax = axes[2, 0]
            outcomes_list = [output_trials[i][1, 0] for i in range(len(output_trials))]
            ax.bar(['ignore', 'miss', 'hit'],
                   [outcomes_list.count(0), outcomes_list.count(1), outcomes_list.count(2)])
            ax.set_title('Outcome distribution')
            
            ax = axes[2, 1]
            early_list = [output_trials[i][2, 0] for i in range(len(output_trials))]
            ax.bar(['no', 'yes'], [early_list.count(0), early_list.count(1)])
            ax.set_title('Early lick distribution')
            
            ax = axes[3, 0]
            for t_idx in range(min(5, len(output_trials))):
                ax.plot(BIN_CENTERS, output_trials[t_idx][3], alpha=0.5)
            ax.set_xlabel('Time from go cue (s)')
            ax.set_ylabel('Tongue y category')
            ax.set_title('Output: Tongue y position (discretized)')
            ax.set_yticks([0, 1, 2, 3])
            ax.set_yticklabels(['<40th', '40-60th', '>60th', 'not visible'])
            
            ax = axes[3, 1]
            if len(neural_trials) > 0:
                mean_fr = np.mean([t for t in neural_trials], axis=0)
                peak_times = np.argmax(mean_fr, axis=1)
                sort_idx = np.argsort(peak_times)
                im = ax.imshow(mean_fr[sort_idx], aspect='auto',
                              extent=[ALIGN_START, ALIGN_END, 0, n_good],
                              cmap='viridis')
                ax.set_xlabel('Time from go cue (s)')
                ax.set_ylabel('Neuron')
                ax.set_title('Mean neural activity')
                plt.colorbar(im, ax=ax, label='Hz')
            
            plt.tight_layout()
            safe_id = session_id.replace('/', '_')
            plt.savefig(f'/app/processing_{safe_id}.png', dpi=100)
            plt.close()
            print(f"  Saved processing plot: /app/processing_{safe_id}.png")
        except Exception as e:
            print(f"  Warning: Could not create processing plot: {e}")
    
    return {
        'subject_id': subject_id,
        'session_id': session_id,
        'neural': neural_trials,
        'input': input_trials,
        'output': output_trials,
        'brain_regions': good_anno_names.tolist(),
        'n_good_units': n_good,
        'n_valid_trials': n_valid_trials,
        'correct_rate': correct_rate,
        'n_total_trials': n_trials,
        'rec_range': (rec_min, rec_max),
    }


def main():
    parser = argparse.ArgumentParser(description='Convert MAP dataset to decoder format')
    parser.add_argument('output', help='Output pickle file path')
    parser.add_argument('--sample', action='store_true', help='Process only 2 sessions')
    parser.add_argument('--full', action='store_true', help='Process all sessions (default)')
    parser.add_argument('--show-processing', action='store_true', help='Show processing plots')
    args = parser.parse_args()
    
    if not args.sample:
        args.full = True
    
    print("="*60)
    print("MAP Dataset Conversion")
    print("="*60)
    
    all_files = get_nwb_files(DATA_DIR)
    print(f"Found {len(all_files)} NWB files from {len(set(s for s,_ in all_files))} subjects")
    
    if args.sample:
        selected = []
        seen_subjects = set()
        for subj, fpath in all_files:
            if subj not in seen_subjects and len(selected) < 2:
                if len(seen_subjects) >= 1 or subj != 'sub-440956':
                    selected.append((subj, fpath))
                seen_subjects.add(subj)
        all_files = selected
        print(f"Sample mode: processing {len(all_files)} sessions")
    
    all_neural = []
    all_input = []
    all_output = []
    all_subject_ids = []
    all_session_brain_regions = []
    all_session_ids = []
    
    total_t0 = time.time()
    n_skipped = 0
    
    for i, (subj, fpath) in enumerate(all_files):
        fname = os.path.basename(fpath)
        print(f"\n[{i+1}/{len(all_files)}] Processing {fname}...")
        
        show = args.show_processing and i < 2
        result = process_session(fpath, show_processing=show, session_idx=i)
        
        if result is None:
            n_skipped += 1
            continue
        
        all_neural.append(result['neural'])
        all_input.append(result['input'])
        all_output.append(result['output'])
        all_subject_ids.append(result['subject_id'])
        all_session_brain_regions.append(result['brain_regions'])
        all_session_ids.append(result['session_id'])
    
    total_t1 = time.time()
    print(f"\nTotal processing time: {total_t1 - total_t0:.1f}s")
    print(f"Sessions processed: {len(all_neural)}, skipped: {n_skipped}")
    
    if len(all_neural) == 0:
        print("ERROR: No valid sessions found!")
        sys.exit(1)
    
    # === Build output data structure ===
    print("\nBuilding output data structure...")
    
    unique_subjects = sorted(set(all_subject_ids))
    subject_idx = np.array([unique_subjects.index(s) for s in all_subject_ids])
    
    all_region_names = set()
    for regions in all_session_brain_regions:
        all_region_names.update(regions)
    all_region_names = sorted(all_region_names)
    
    brain_region_idx = []
    for regions in all_session_brain_regions:
        idx = np.array([all_region_names.index(r) for r in regions])
        brain_region_idx.append(idx)
    
    data = {
        'neural': all_neural,
        'input': all_input,
        'output': all_output,
        
        'subjects': unique_subjects,
        'subject_idx': subject_idx,
        
        'brain_regions': all_region_names,
        'brain_region_idx': brain_region_idx,
        
        'input_names': ['time_from_tone_onset', 'photostim_on'],
        'output_names': ['choice', 'outcome', 'early_lick', 'tongue_y_position'],
        'output_values': [
            ['left', 'right', 'no_lick'],
            ['ignore', 'miss', 'hit'],
            ['no', 'yes'],
            ['below_40th', '40th_to_60th', 'above_60th', 'not_visible'],
        ],
        
        'metadata': {
            'task_description': 'Auditory delayed response task: mice hear 3kHz or 12kHz tones, '
                               'wait through a 1.2s delay, then lick left or right after go cue',
            'time_bin_size': BIN_WIDTH * 1000,
            'temporal_alignment_event': 'Go cue onset',
            'off_start': ALIGN_START,
            'off_end': ALIGN_END,
            'session_ids': all_session_ids,
            'n_sessions': len(all_neural),
            'n_subjects': len(unique_subjects),
            'bin_centers': BIN_CENTERS.tolist(),
        }
    }
    
    # === Print summary ===
    print("\n" + "="*60)
    print("CONVERSION SUMMARY")
    print("="*60)
    n_sessions = len(data['neural'])
    total_trials = sum(len(s) for s in data['neural'])
    total_neurons = sum(data['neural'][s][0].shape[0] if len(data['neural'][s]) > 0 else 0 
                       for s in range(n_sessions))
    trials_per_session = [len(s) for s in data['neural']]
    neurons_per_session = [data['neural'][s][0].shape[0] if len(data['neural'][s]) > 0 else 0 
                          for s in range(n_sessions)]
    
    print(f"Sessions: {n_sessions}")
    print(f"Subjects: {len(unique_subjects)}")
    print(f"Total trials: {total_trials}")
    print(f"Total neurons: {total_neurons}")
    print(f"Mean trials/session: {np.mean(trials_per_session):.1f} (range: {min(trials_per_session)}-{max(trials_per_session)})")
    print(f"Mean neurons/session: {np.mean(neurons_per_session):.1f} (range: {min(neurons_per_session)}-{max(neurons_per_session)})")
    print(f"Brain regions: {len(all_region_names)}")
    print(f"Time bins: {N_TIMEBINS}")
    print(f"Bin width: {BIN_WIDTH*1000:.0f} ms")
    print(f"Window: [{ALIGN_START}, {ALIGN_END}] s")
    
    all_choices = []
    all_outcomes = []
    all_early = []
    for sess_outputs in data['output']:
        for trial_out in sess_outputs:
            all_choices.append(trial_out[0, 0])
            all_outcomes.append(trial_out[1, 0])
            all_early.append(trial_out[2, 0])
    
    all_choices = np.array(all_choices)
    all_outcomes = np.array(all_outcomes)
    all_early = np.array(all_early)
    
    print(f"\nChoice dist: left={np.mean(all_choices==0):.3f}, right={np.mean(all_choices==1):.3f}, no_lick={np.mean(all_choices==2):.3f}")
    print(f"Outcome dist: ignore={np.mean(all_outcomes==0):.3f}, miss={np.mean(all_outcomes==1):.3f}, hit={np.mean(all_outcomes==2):.3f}")
    print(f"Early lick dist: no={np.mean(all_early==0):.3f}, yes={np.mean(all_early==1):.3f}")
    
    # === Save ===
    print(f"\nSaving to {args.output}...")
    with open(args.output, 'wb') as f:
        pickle.dump(data, f)
    
    file_size = os.path.getsize(args.output) / (1024**2)
    print(f"Saved! File size: {file_size:.1f} MB")
    print("Done!")


if __name__ == '__main__':
    main()
