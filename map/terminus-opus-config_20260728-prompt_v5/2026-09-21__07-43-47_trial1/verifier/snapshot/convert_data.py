#!/usr/bin/env python3
"""Convert MAP dataset NWB files to decoder-compatible format.

Usage:
    python -u convert_data.py <outpicklefile> [--full|--sample] [--show-processing]
"""

import os
import sys
import time
import argparse
import pickle
import numpy as np
import pynwb
import warnings
warnings.filterwarnings('ignore')


def get_nwb_files(data_dir='/app/data'):
    """Get sorted list of all NWB file paths."""
    nwb_files = []
    for subdir in sorted(os.listdir(data_dir)):
        sub_path = os.path.join(data_dir, subdir)
        if not os.path.isdir(sub_path) or not subdir.startswith('sub-'):
            continue
        for f in sorted(os.listdir(sub_path)):
            if f.endswith('.nwb'):
                nwb_files.append(os.path.join(sub_path, f))
    return nwb_files


def determine_lick_choice(go_time, trial_stop, left_lick_times, right_lick_times):
    """Determine lick choice based on first lick after go cue.
    Returns: 0='left', 1='right', 2='no_lick'
    """
    left_after = left_lick_times[(left_lick_times > go_time) & (left_lick_times < trial_stop)]
    right_after = right_lick_times[(right_lick_times > go_time) & (right_lick_times < trial_stop)]
    first_left = left_after[0] if len(left_after) > 0 else np.inf
    first_right = right_after[0] if len(right_after) > 0 else np.inf
    if first_left < first_right:
        return 0
    elif first_right < first_left:
        return 1
    else:
        return 2


def compute_firing_rates_vectorized(spike_times_list, go_times, t_start, t_end, bin_width):
    """Compute firing rates for all neurons across all trials."""
    n_bins = int(round((t_end - t_start) / bin_width))
    bin_edges = t_start + np.arange(n_bins + 1) * bin_width
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    n_neurons = len(spike_times_list)
    n_trials = len(go_times)
    
    fr_all = np.zeros((n_trials, n_neurons, n_bins), dtype=np.float32)
    
    for j, spk_times in enumerate(spike_times_list):
        if len(spk_times) == 0:
            continue
        for i in range(n_trials):
            go = go_times[i]
            lo = np.searchsorted(spk_times, go + t_start)
            hi = np.searchsorted(spk_times, go + t_end)
            if hi > lo:
                relative_spikes = spk_times[lo:hi] - go
                counts, _ = np.histogram(relative_spikes, bins=bin_edges)
                fr_all[i, j, :] = counts / bin_width
    
    fr_list = [fr_all[i] for i in range(n_trials)]
    return fr_list, bin_centers


def get_tone_onset_per_trial(go_times, trial_starts, sample_start_times):
    """Find the tone (sample) onset time for each trial."""
    tone_onsets = np.full(len(go_times), np.nan)
    for i in range(len(go_times)):
        go = go_times[i]
        ts = trial_starts[i]
        samp_in_trial = sample_start_times[(sample_start_times >= ts) & (sample_start_times < go)]
        if len(samp_in_trial) > 0:
            tone_onsets[i] = samp_in_trial[-1]
        else:
            tone_onsets[i] = go - 1.85
    return tone_onsets


def compute_tongue_y_vectorized(tongue_data, tongue_timestamps, go_times,
                                 t_start, t_end, bin_width,
                                 p40, p60, likelihood_threshold=0.9):
    """Compute discretized tongue y-position aligned to go cue."""
    n_bins = int(round((t_end - t_start) / bin_width))
    bin_edges = t_start + np.arange(n_bins + 1) * bin_width
    n_trials = len(go_times)
    
    tongue_y = tongue_data[:, 1]
    tongue_lk = tongue_data[:, 2]
    
    result = np.full((n_trials, n_bins), 3, dtype=np.int64)
    
    for i in range(n_trials):
        go = go_times[i]
        abs_edges = go + bin_edges
        
        idx_start = np.searchsorted(tongue_timestamps, abs_edges[0])
        idx_end = np.searchsorted(tongue_timestamps, abs_edges[-1])
        
        if idx_start >= idx_end:
            continue
        
        trial_ts = tongue_timestamps[idx_start:idx_end]
        trial_y = tongue_y[idx_start:idx_end]
        trial_lk = tongue_lk[idx_start:idx_end]
        
        bin_indices = np.searchsorted(abs_edges, trial_ts, side='right') - 1
        valid = (bin_indices >= 0) & (bin_indices < n_bins)
        
        if not np.any(valid):
            continue
        
        bin_indices_v = bin_indices[valid]
        trial_y_v = trial_y[valid]
        trial_lk_v = trial_lk[valid]
        
        for b in range(n_bins):
            mask = bin_indices_v == b
            if not np.any(mask):
                continue
            visible = trial_lk_v[mask] >= likelihood_threshold
            if np.any(visible):
                mean_y = np.mean(trial_y_v[mask][visible])
                if mean_y < p40:
                    result[i, b] = 0
                elif mean_y <= p60:
                    result[i, b] = 1
                else:
                    result[i, b] = 2
    
    return [result[i] for i in range(n_trials)]


def get_recording_time_range(nwb, good_indices):
    """Get the common recording time range across all good units.
    
    Returns (rec_start, rec_end) - the intersection of all units' recording periods.
    """
    rec_start = -np.inf
    rec_end = np.inf
    
    for idx in good_indices:
        obs = nwb.units['obs_intervals'][idx]
        unit_start = obs[0, 0]
        unit_end = obs[-1, 1]
        rec_start = max(rec_start, unit_start)
        rec_end = min(rec_end, unit_end)
    
    return rec_start, rec_end


def process_session(nwb_path, t_start=-2.5, t_end=1.5, bin_width=0.05,
                    show_processing=False, session_idx=0):
    """Process a single NWB session file."""
    t0 = time.time()
    
    with pynwb.NWBHDF5IO(nwb_path, 'r') as io:
        nwb = io.read()
        
        subject_id = nwb_path.split('/')[-2]
        session_id = os.path.basename(nwb_path).replace('.nwb', '')
        
        # === Filter units ===
        units = nwb.units
        classifications = units['classification'][:]
        good_mask = classifications == 'good'
        good_indices = np.where(good_mask)[0]
        n_good = len(good_indices)
        
        if n_good == 0:
            print(f"  Skipping {session_id}: no good units")
            return None
        
        # === Get brain region labels ===
        anno_names = units['anno_name'][:]
        good_anno_names = anno_names[good_indices]
        
        # === Get recording time range ===
        rec_start, rec_end = get_recording_time_range(nwb, good_indices)
        
        # === Get spike times for good units ===
        spike_times_list = []
        for idx in good_indices:
            spike_times_list.append(units['spike_times'][idx])
        
        t_load_spikes = time.time()
        print(f"  Loaded {n_good} good units in {t_load_spikes - t0:.1f}s (rec: {rec_start:.0f}-{rec_end:.0f}s)")
        
        # === Get trial info ===
        trials = nwb.intervals['trials']
        n_trials_total = len(trials)
        trial_starts = trials['start_time'][:]
        trial_stops = trials['stop_time'][:]
        outcomes = trials['outcome'][:]
        instructions = trials['trial_instruction'][:]
        early_lick = trials['early_lick'][:]
        auto_water = trials['auto_water'][:]
        free_water = trials['free_water'][:]
        photostim_onset_str = trials['photostim_onset'][:]
        photostim_duration_str = trials['photostim_duration'][:]
        
        # === Get behavioral events ===
        be = nwb.acquisition['BehavioralEvents']
        go_times = be.time_series['go_start_times'].timestamps[:]
        sample_start_times = be.time_series['sample_start_times'].timestamps[:]
        left_lick_times = be.time_series['left_lick_times'].timestamps[:]
        right_lick_times = be.time_series['right_lick_times'].timestamps[:]
        
        # === Get tongue tracking data ===
        bts = nwb.acquisition['BehavioralTimeSeries']
        has_tongue = 'Camera0_side_TongueTracking' in bts.time_series
        if has_tongue:
            tongue_ts_obj = bts.time_series['Camera0_side_TongueTracking']
            tongue_data = tongue_ts_obj.data[:]
            tongue_timestamps = tongue_ts_obj.timestamps[:]
        
        t_load = time.time()
        
        # === Filter trials ===
        # 1. Remove auto_water and free_water trials
        valid_trials = (auto_water == 0) & (free_water == 0)
        
        # 2. Only include trials where the analysis window falls within the recording
        recording_covered = (go_times + t_start >= rec_start) & (go_times + t_end <= rec_end)
        valid_trials = valid_trials & recording_covered
        
        valid_indices = np.where(valid_trials)[0]
        n_valid = len(valid_indices)
        
        n_auto_free = np.sum((auto_water == 1) | (free_water == 1))
        n_not_covered = np.sum(~recording_covered & (auto_water == 0) & (free_water == 0))
        
        if n_valid < 2:
            print(f"  Skipping {session_id}: only {n_valid} valid trials")
            return None
        
        print(f"  {n_valid}/{n_trials_total} valid trials (removed {n_auto_free} auto/free water, {n_not_covered} outside recording)")
        
        # Filter arrays
        go_times_valid = go_times[valid_indices]
        trial_starts_valid = trial_starts[valid_indices]
        trial_stops_valid = trial_stops[valid_indices]
        outcomes_valid = outcomes[valid_indices]
        instructions_valid = instructions[valid_indices]
        early_lick_valid = early_lick[valid_indices]
        photostim_onset_valid = photostim_onset_str[valid_indices]
        photostim_duration_valid = photostim_duration_str[valid_indices]
        
        # === Compute firing rates ===
        fr_list, bin_centers = compute_firing_rates_vectorized(
            spike_times_list, go_times_valid, t_start, t_end, bin_width
        )
        n_bins = len(bin_centers)
        
        t_fr = time.time()
        print(f"  Computed firing rates in {t_fr - t_load:.1f}s")
        
        # === Compute tone onset per trial ===
        tone_onsets = get_tone_onset_per_trial(go_times_valid, trial_starts_valid, sample_start_times)
        
        # === Compute inputs ===
        input_list = []
        for i in range(n_valid):
            inputs = np.zeros((2, n_bins), dtype=np.float32)
            
            # Time from tone onset (continuous)
            tone_onset_rel = tone_onsets[i] - go_times_valid[i]
            inputs[0, :] = bin_centers - tone_onset_rel
            
            # Photostim on/off
            if photostim_onset_valid[i] != 'N/A':
                stim_onset_abs = trial_starts_valid[i] + float(photostim_onset_valid[i])
                stim_onset_go_rel = stim_onset_abs - go_times_valid[i]
                stim_dur = float(photostim_duration_valid[i])
                stim_end_go_rel = stim_onset_go_rel + stim_dur
                inputs[1, :] = ((bin_centers >= stim_onset_go_rel) & (bin_centers < stim_end_go_rel)).astype(np.float32)
            
            input_list.append(inputs)
        
        # === Compute outputs ===
        p40, p60 = None, None
        if has_tongue:
            visible_mask = tongue_data[:, 2] >= 0.9
            if np.any(visible_mask):
                y_visible = tongue_data[visible_mask, 1]
                p40 = np.percentile(y_visible, 40)
                p60 = np.percentile(y_visible, 60)
        
        if has_tongue and p40 is not None:
            tongue_y_list = compute_tongue_y_vectorized(
                tongue_data, tongue_timestamps, go_times_valid,
                t_start, t_end, bin_width, p40, p60
            )
        else:
            tongue_y_list = [np.full(n_bins, 3, dtype=np.int64) for _ in range(n_valid)]
        
        t_tongue = time.time()
        print(f"  Computed tongue y in {t_tongue - t_fr:.1f}s")
        
        output_list = []
        for i in range(n_valid):
            choice = determine_lick_choice(
                go_times_valid[i], trial_stops_valid[i],
                left_lick_times, right_lick_times
            )
            
            outcome_map = {'hit': 0, 'miss': 1, 'ignore': 2}
            outcome = outcome_map.get(outcomes_valid[i], 2)
            early = 1 if early_lick_valid[i] == 'early' else 0
            
            outputs = np.zeros((4, n_bins), dtype=np.int64)
            outputs[0, :] = choice
            outputs[1, :] = outcome
            outputs[2, :] = early
            outputs[3, :] = tongue_y_list[i]
            
            output_list.append(outputs)
        
        t_end_proc = time.time()
        print(f"  Total processing time: {t_end_proc - t0:.1f}s")
        
        if show_processing:
            plot_processing(session_id, bin_centers, fr_list, input_list, output_list,
                           go_times_valid, outcomes_valid, instructions_valid,
                           early_lick_valid, tongue_y_list, session_idx)
        
        return {
            'neural': fr_list,
            'input': input_list,
            'output': output_list,
            'subject_id': subject_id,
            'session_id': session_id,
            'brain_regions': list(good_anno_names),
            'n_good_units': n_good,
            'n_trials': n_valid,
            'bin_centers': bin_centers,
        }


def plot_processing(session_id, bin_centers, fr_list, input_list, output_list,
                    go_times, outcomes, instructions, early_lick, tongue_y_list, session_idx):
    """Plot processing visualizations for a session."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(4, 2, figsize=(16, 16))
    fig.suptitle(f'Session: {session_id}', fontsize=14)
    
    n_trials = len(fr_list)
    n_neurons = fr_list[0].shape[0] if n_trials > 0 else 0
    
    ax = axes[0, 0]
    if n_trials > 0 and n_neurons > 0:
        mean_fr = np.mean([fr.mean(axis=0) for fr in fr_list], axis=0)
        ax.plot(bin_centers, mean_fr)
        ax.axvline(0, color='r', linestyle='--', label='Go cue')
        ax.set_xlabel('Time from go cue (s)')
        ax.set_ylabel('Mean firing rate (Hz)')
        ax.set_title(f'Mean FR (n={n_neurons} neurons, {n_trials} trials)')
        ax.legend()
    
    ax = axes[0, 1]
    if n_trials > 0 and n_neurons > 0:
        neuron_fr = np.array([fr[0, :] for fr in fr_list])
        ax.imshow(neuron_fr, aspect='auto', extent=[bin_centers[0], bin_centers[-1], n_trials, 0])
        ax.axvline(0, color='r', linestyle='--')
        ax.set_xlabel('Time from go cue (s)')
        ax.set_ylabel('Trial')
        ax.set_title('Neuron 0 firing rate')
    
    ax = axes[1, 0]
    if n_trials > 0:
        ax.plot(bin_centers, input_list[0][0, :], label='Trial 0')
        ax.plot(bin_centers, input_list[min(5, n_trials-1)][0, :], label=f'Trial {min(5, n_trials-1)}')
        ax.axvline(0, color='r', linestyle='--', label='Go cue')
        ax.set_xlabel('Time from go cue (s)')
        ax.set_ylabel('Time from tone onset (s)')
        ax.set_title('Input 0: Time from tone onset')
        ax.legend()
    
    ax = axes[1, 1]
    if n_trials > 0:
        stim_trials = [i for i in range(n_trials) if np.any(input_list[i][1, :] > 0)]
        if stim_trials:
            ax.plot(bin_centers, input_list[stim_trials[0]][1, :], label=f'Stim trial {stim_trials[0]}')
        non_stim = [i for i in range(n_trials) if not np.any(input_list[i][1, :] > 0)]
        if non_stim:
            ax.plot(bin_centers, input_list[non_stim[0]][1, :], label=f'Non-stim trial {non_stim[0]}')
        ax.axvline(0, color='r', linestyle='--')
        ax.set_xlabel('Time from go cue (s)')
        ax.set_ylabel('Photostim on')
        ax.set_title('Input 1: Photostim')
        ax.legend()
    
    ax = axes[2, 0]
    choices = [out[0, 0] for out in output_list]
    choice_labels = ['left', 'right', 'no_lick']
    choice_counts = [choices.count(i) for i in range(3)]
    ax.bar(choice_labels, choice_counts)
    ax.set_title('Output 0: Lick choice')
    
    ax = axes[2, 1]
    outcomes_arr = [out[1, 0] for out in output_list]
    outcome_labels = ['hit', 'miss', 'ignore']
    outcome_counts = [outcomes_arr.count(i) for i in range(3)]
    ax.bar(outcome_labels, outcome_counts)
    ax.set_title('Output 1: Outcome')
    
    ax = axes[3, 0]
    if n_trials > 0:
        tongue_img = np.array([out[3, :] for out in output_list])
        ax.imshow(tongue_img, aspect='auto', extent=[bin_centers[0], bin_centers[-1], n_trials, 0],
                  cmap='viridis', vmin=0, vmax=3)
        ax.axvline(0, color='r', linestyle='--')
        ax.set_xlabel('Time from go cue (s)')
        ax.set_ylabel('Trial')
        ax.set_title('Output 3: Tongue y-position')
    
    ax = axes[3, 1]
    early_arr = [out[2, 0] for out in output_list]
    early_labels = ['no early', 'early']
    early_counts = [early_arr.count(i) for i in range(2)]
    ax.bar(early_labels, early_counts)
    ax.set_title('Output 2: Early lick')
    
    plt.tight_layout()
    plt.savefig(f'/app/processing_session{session_idx}.png', dpi=100)
    plt.close()
    print(f"  Saved processing plot to /app/processing_session{session_idx}.png")


def build_brain_region_mapping(all_sessions_data):
    """Build a consistent brain region mapping across all sessions."""
    all_regions = set()
    for sess in all_sessions_data:
        all_regions.update(sess['brain_regions'])
    brain_regions = sorted(all_regions)
    region_to_idx = {r: i for i, r in enumerate(brain_regions)}
    return brain_regions, region_to_idx


def assemble_dataset(all_sessions_data, t_start=-2.5, t_end=1.5, bin_width=0.05):
    """Assemble the final dataset dictionary."""
    brain_regions, region_to_idx = build_brain_region_mapping(all_sessions_data)
    all_subjects = sorted(set(s['subject_id'] for s in all_sessions_data))
    subject_to_idx = {s: i for i, s in enumerate(all_subjects)}
    
    neural = []
    inputs = []
    outputs = []
    subject_idx = []
    brain_region_idx = []
    
    for sess in all_sessions_data:
        neural.append(sess['neural'])
        inputs.append(sess['input'])
        outputs.append(sess['output'])
        subject_idx.append(subject_to_idx[sess['subject_id']])
        region_indices = np.array([region_to_idx.get(r, 0) for r in sess['brain_regions']])
        brain_region_idx.append(region_indices)
    
    data = {
        'neural': neural,
        'input': inputs,
        'output': outputs,
        'subjects': all_subjects,
        'subject_idx': np.array(subject_idx),
        'brain_regions': brain_regions,
        'brain_region_idx': brain_region_idx,
        'input_names': ['time_from_tone_onset', 'photostim_on'],
        'output_names': ['lick_choice', 'outcome', 'early_lick', 'tongue_y_position'],
        'output_values': [
            ['left', 'right', 'no_lick'],
            ['hit', 'miss', 'ignore'],
            ['no_early', 'early'],
            ['below_p40', 'p40_to_p60', 'above_p60', 'not_visible'],
        ],
        'metadata': {
            'task_description': 'Audio delay task: mice discriminate between two tones and report by directional licking after a delay period',
            'time_bin_size': bin_width * 1000,
            'temporal_alignment_event': 'Go cue onset',
            'off_start': t_start,
            'off_end': t_end,
            'dataset': 'MAP (Mesoscale Activity Project)',
            'paper': 'Li et al. - Brain-wide neural activity underlying memory-guided movement',
            'n_sessions': len(all_sessions_data),
            'n_subjects': len(all_subjects),
            'total_good_neurons': sum(s['n_good_units'] for s in all_sessions_data),
            'total_trials': sum(s['n_trials'] for s in all_sessions_data),
        }
    }
    
    return data


def main():
    parser = argparse.ArgumentParser(description='Convert MAP NWB data to decoder format')
    parser.add_argument('output', type=str, help='Output pickle file path')
    parser.add_argument('--full', action='store_true', default=True, help='Process all sessions (default)')
    parser.add_argument('--sample', action='store_true', help='Process only 2 sessions for testing')
    parser.add_argument('--show-processing', action='store_true', help='Plot processing visualizations')
    args = parser.parse_args()
    
    t_start = -2.5
    t_end = 1.5
    bin_width = 0.05
    
    print("=== MAP Dataset Conversion ===")
    print(f"Output: {args.output}")
    print(f"Mode: {'sample' if args.sample else 'full'}")
    print(f"Time window: [{t_start}, {t_end}] relative to go cue")
    print(f"Bin width: {bin_width}s ({bin_width*1000}ms)")
    print()
    
    nwb_files = get_nwb_files()
    print(f"Found {len(nwb_files)} NWB files")
    
    if args.sample:
        nwb_files = [nwb_files[0], nwb_files[4]]
        print(f"Sample mode: processing {len(nwb_files)} sessions")
    
    print()
    
    all_sessions = []
    total_start = time.time()
    
    for i, nwb_path in enumerate(nwb_files):
        print(f"[{i+1}/{len(nwb_files)}] Processing {os.path.basename(nwb_path)}")
        
        show = args.show_processing and i < 2
        result = process_session(
            nwb_path, t_start=t_start, t_end=t_end, bin_width=bin_width,
            show_processing=show, session_idx=i
        )
        
        if result is not None:
            all_sessions.append(result)
            print(f"  -> {result['n_good_units']} neurons, {result['n_trials']} trials")
        print()
    
    total_time = time.time() - total_start
    print(f"\nProcessed {len(all_sessions)} sessions in {total_time:.1f}s")
    print(f"Average time per session: {total_time/max(len(all_sessions),1):.1f}s")
    
    print("\nAssembling dataset...")
    data = assemble_dataset(all_sessions, t_start=t_start, t_end=t_end, bin_width=bin_width)
    
    print(f"\n=== Dataset Summary ===")
    print(f"Sessions: {len(data['neural'])}")
    print(f"Subjects: {len(data['subjects'])}")
    print(f"Total neurons: {data['metadata']['total_good_neurons']}")
    print(f"Total trials: {data['metadata']['total_trials']}")
    print(f"Brain regions: {len(data['brain_regions'])}")
    print(f"Input names: {data['input_names']}")
    print(f"Output names: {data['output_names']}")
    
    print(f"\nSaving to {args.output}...")
    with open(args.output, 'wb') as f:
        pickle.dump(data, f)
    
    file_size = os.path.getsize(args.output) / (1024**2)
    print(f"Saved ({file_size:.1f} MB)")
    print("Done!")


if __name__ == '__main__':
    main()
