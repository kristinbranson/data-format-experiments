#!/usr/bin/env python3
"""Convert Allen Visual Behavior 2P data to decoder-compatible format.

Usage:
    python -u convert_data.py <output.pkl> [--sample] [--full] [--show-processing]
"""

import argparse
import os
import sys
import time
import pickle
import warnings
import numpy as np
import pandas as pd
import h5py
from scipy import interpolate

warnings.filterwarnings('ignore')

# Constants
DATA_DIR = 'data/visual-behavior-ophys-1.1.0'
EXPT_DIR = os.path.join(DATA_DIR, 'behavior_ophys_experiments')
META_DIR = os.path.join(DATA_DIR, 'project_metadata')

# Common time bin: use ~93ms (MESO rate) to accommodate all experiments
COMMON_DT = 0.09323  # seconds, ~10.73 Hz


def get_experiment_list(sample=False):
    """Get list of active experiment IDs to process."""
    expt_table = pd.read_csv(os.path.join(META_DIR, 'ophys_experiment_table.csv'))
    downloaded_ids = set()
    for f in os.listdir(EXPT_DIR):
        if f.endswith('.nwb'):
            eid = int(f.split('_')[-1].replace('.nwb', ''))
            downloaded_ids.add(eid)
    expt_table = expt_table[expt_table.ophys_experiment_id.isin(downloaded_ids)]
    expt_table = expt_table[expt_table.passive == False]
    
    if sample:
        sample_expts = []
        for equip in ['CAM2P.3', 'MESO.1']:
            eq_expts = expt_table[expt_table.equipment_name == equip]
            if len(eq_expts) > 0:
                sample_expts.append(eq_expts.iloc[0])
        if len(sample_expts) < 2:
            sample_expts = [expt_table.iloc[i] for i in range(min(2, len(expt_table)))]
        expt_table = pd.DataFrame(sample_expts)
    
    return expt_table


def load_experiment_data(expt_id):
    """Load data from a single NWB experiment file."""
    fname = os.path.join(EXPT_DIR, f'behavior_ophys_experiment_{expt_id}.nwb')
    f = h5py.File(fname, 'r')
    data = {}
    
    data['subject_id'] = f['general/subject/subject_id'][()]
    if isinstance(data['subject_id'], bytes):
        data['subject_id'] = data['subject_id'].decode()
    
    data['events'] = f['processing/ophys/event_detection/data'][:].T  # (n_neurons, n_timepoints)
    data['ophys_timestamps'] = f['processing/ophys/dff/traces/timestamps'][:]
    data['valid_roi'] = f['processing/ophys/image_segmentation/cell_specimen_table/valid_roi'][:]
    data['cell_specimen_ids'] = f['processing/ophys/image_segmentation/cell_specimen_table/cell_specimen_id'][:]
    data['running_speed'] = f['processing/running/speed/data'][:]
    data['running_timestamps'] = f['processing/running/speed/timestamps'][:]
    
    try:
        data['pupil_area'] = f['acquisition/EyeTracking/pupil_tracking/area'][:]
        data['pupil_timestamps'] = f['acquisition/EyeTracking/pupil_tracking/timestamps'][:]
        data['has_pupil'] = True
    except KeyError:
        data['has_pupil'] = False
        data['pupil_area'] = None
        data['pupil_timestamps'] = None
    
    trials = f['intervals/trials']
    trial_data = {}
    for key in ['go', 'catch', 'aborted', 'auto_rewarded', 'hit', 'miss',
                'correct_reject', 'false_alarm', 'start_time', 'stop_time',
                'change_time', 'change_image_name', 'initial_image_name', 'is_change']:
        vals = trials[key][:]
        if vals.dtype == object:
            vals = np.array([v.decode() if isinstance(v, bytes) else v for v in vals])
        trial_data[key] = vals
    data['trials'] = trial_data
    
    stim_keys = [k for k in f['intervals'].keys() if 'Natural_Images' in k]
    if not stim_keys:
        stim_keys = [k for k in f['intervals'].keys() 
                     if k != 'trials' and 'spontaneous' not in k and 'natural_movie' not in k.lower()]
    stim_key = stim_keys[0] if stim_keys else None
    
    if stim_key:
        stim = f[f'intervals/{stim_key}']
        stim_data = {}
        for key in ['start_time', 'stop_time', 'image_name', 'is_change', 'omitted']:
            try:
                vals = stim[key][:]
                if vals.dtype == object:
                    vals = np.array([v.decode() if isinstance(v, bytes) else v for v in vals])
                stim_data[key] = vals
            except KeyError:
                pass
        data['stimulus_presentations'] = stim_data
    else:
        data['stimulus_presentations'] = None
    
    f.close()
    return data


def resample_events_to_common_bins(events, ophys_ts, trial_start, trial_stop):
    """Resample events to common time bins within a trial.
    
    For each common bin, sum events from ophys frames that fall within it.
    Vectorized for speed.
    """
    common_ts = np.arange(trial_start, trial_stop, COMMON_DT)
    n_common = len(common_ts)
    if n_common < 2:
        return None, None
    
    n_neurons = events.shape[0]
    
    # Find which ophys frames fall in the trial window (with some margin)
    trial_mask = (ophys_ts >= trial_start - COMMON_DT) & (ophys_ts < trial_stop + COMMON_DT)
    trial_ophys_ts = ophys_ts[trial_mask]
    trial_events = events[:, trial_mask]
    
    if len(trial_ophys_ts) == 0:
        return np.zeros((n_neurons, n_common), dtype=np.float32), common_ts
    
    # Assign each ophys frame to a common bin using digitize
    bin_edges = np.concatenate([
        [common_ts[0] - COMMON_DT/2],
        (common_ts[:-1] + common_ts[1:]) / 2,
        [common_ts[-1] + COMMON_DT/2]
    ])
    bin_assignments = np.digitize(trial_ophys_ts, bin_edges) - 1  # 0-indexed
    
    # Sum events in each bin
    resampled = np.zeros((n_neurons, n_common), dtype=np.float32)
    for b in range(n_common):
        mask = bin_assignments == b
        if mask.any():
            resampled[:, b] = trial_events[:, mask].sum(axis=1)
    
    return resampled, common_ts


def interpolate_to_common(data, source_ts, common_ts):
    """Interpolate continuous data to common timestamps."""
    if data is None or source_ts is None:
        return np.full(len(common_ts), np.nan)
    
    valid = ~np.isnan(data)
    if valid.sum() < 2:
        return np.full(len(common_ts), np.nan)
    
    f_interp = interpolate.interp1d(
        source_ts[valid], data[valid],
        kind='linear', bounds_error=False, fill_value=np.nan
    )
    return f_interp(common_ts)


def get_image_at_timepoints(common_ts, stim_data, image_to_idx):
    """Get image identity at each timepoint."""
    n_tp = len(common_ts)
    image_id = np.zeros(n_tp, dtype=np.int64)
    
    start_times = stim_data['start_time']
    image_names = stim_data['image_name']
    
    # For each timepoint, find the most recent stimulus
    indices = np.searchsorted(start_times, common_ts, side='right') - 1
    
    last_valid = 0  # default to first image
    for t in range(n_tp):
        idx = indices[t]
        if idx < 0:
            image_id[t] = last_valid
            continue
        img = image_names[idx]
        if img == 'omitted':
            image_id[t] = last_valid
        elif img in image_to_idx:
            image_id[t] = image_to_idx[img]
            last_valid = image_to_idx[img]
        else:
            image_id[t] = last_valid
    
    return image_id


def get_image_change_at_timepoints(common_ts, stim_data):
    """Binary: 1 during change stimulus, 0 otherwise."""
    n_tp = len(common_ts)
    change = np.zeros(n_tp, dtype=np.int64)
    
    is_change = stim_data.get('is_change', np.zeros(len(stim_data['start_time'])))
    start_times = stim_data['start_time']
    stop_times = stim_data['stop_time']
    
    change_indices = np.where(np.array(is_change) == 1)[0]
    for ci in change_indices:
        mask = (common_ts >= start_times[ci]) & (common_ts < stop_times[ci])
        change[mask] = 1
    
    return change


def get_trial_outcome(trial_data, trial_idx):
    """Get trial outcome as categorical value."""
    if trial_data['hit'][trial_idx]: return 0
    elif trial_data['miss'][trial_idx]: return 1
    elif trial_data['correct_reject'][trial_idx]: return 2
    elif trial_data['false_alarm'][trial_idx]: return 3
    else: return -1


def process_experiment(expt_id, expt_info, all_image_names, image_to_idx,
                       running_percentiles, pupil_percentiles,
                       show_processing=False):
    """Process a single experiment."""
    t0 = time.time()
    raw = load_experiment_data(expt_id)
    t_load = time.time() - t0
    
    valid_mask = raw['valid_roi']
    events = raw['events'][valid_mask]
    ophys_ts = raw['ophys_timestamps']
    n_neurons = events.shape[0]
    
    if n_neurons == 0:
        print(f'  WARNING: No valid neurons in experiment {expt_id}')
        return None
    if raw['stimulus_presentations'] is None:
        print(f'  WARNING: No stimulus presentations in experiment {expt_id}')
        return None
    
    trial_data = raw['trials']
    include_mask = trial_data['go'] | trial_data['catch']
    include_indices = np.where(include_mask)[0]
    
    if len(include_indices) < 2:
        print(f'  WARNING: Too few trials in experiment {expt_id}')
        return None
    
    neural_trials = []
    output_trials = []
    
    for trial_idx in include_indices:
        trial_start = trial_data['start_time'][trial_idx]
        trial_stop = trial_data['stop_time'][trial_idx]
        
        # Resample neural events
        trial_neural, common_ts = resample_events_to_common_bins(
            events, ophys_ts, trial_start, trial_stop
        )
        if trial_neural is None:
            continue
        n_common = len(common_ts)
        
        # Image identity
        img_id = get_image_at_timepoints(common_ts, raw['stimulus_presentations'], image_to_idx)
        
        # Image change
        img_change = get_image_change_at_timepoints(common_ts, raw['stimulus_presentations'])
        
        # Running speed - interpolate and bin
        run_interp = interpolate_to_common(
            raw['running_speed'], raw['running_timestamps'], common_ts
        )
        run_binned = np.digitize(run_interp, running_percentiles[1:-1])
        run_binned = np.clip(run_binned, 0, 4).astype(np.int64)
        run_binned[np.isnan(run_interp)] = 2
        
        # Pupil area - interpolate and bin
        if raw['has_pupil']:
            pupil_interp = interpolate_to_common(
                raw['pupil_area'], raw['pupil_timestamps'], common_ts
            )
        else:
            pupil_interp = np.full(n_common, np.nan)
        
        if pupil_percentiles is not None:
            pupil_binned = np.digitize(pupil_interp, pupil_percentiles[1:-1])
            pupil_binned = np.clip(pupil_binned, 0, 4).astype(np.int64)
            pupil_binned[np.isnan(pupil_interp)] = 2
        else:
            pupil_binned = np.full(n_common, 2, dtype=np.int64)
        
        # Trial outcome
        outcome = get_trial_outcome(trial_data, trial_idx)
        if outcome == -1:
            continue
        
        # Stack outputs: (5, n_timepoints)
        trial_output = np.stack([
            img_id, img_change, run_binned, pupil_binned,
            np.full(n_common, outcome, dtype=np.int64)
        ], axis=0)
        
        neural_trials.append(trial_neural)
        output_trials.append(trial_output)
    
    t_process = time.time() - t0
    print(f'  Experiment {expt_id}: {n_neurons} neurons, {len(neural_trials)} trials, '
          f'load={t_load:.1f}s, total={t_process:.1f}s')
    
    if len(neural_trials) < 2:
        print(f'  WARNING: Too few valid trials in experiment {expt_id}')
        return None
    
    result = {
        'neural': neural_trials,
        'output': output_trials,
        'subject_id': raw['subject_id'],
        'targeted_structure': expt_info['targeted_structure'],
        'n_neurons': n_neurons,
        'n_trials': len(neural_trials),
        'expt_id': expt_id,
    }
    
    if show_processing:
        result['raw'] = raw
        result['ophys_ts'] = ophys_ts
    
    return result


def collect_global_stats(expt_table):
    """Collect global statistics for percentile binning and image names."""
    print('Collecting global statistics...')
    all_running = []
    all_pupil = []
    all_images = set()
    
    expt_ids = expt_table.ophys_experiment_id.values
    for i, expt_id in enumerate(expt_ids):
        if i % 20 == 0:
            print(f'  Scanning experiment {i+1}/{len(expt_ids)}...')
        try:
            fname = os.path.join(EXPT_DIR, f'behavior_ophys_experiment_{expt_id}.nwb')
            with h5py.File(fname, 'r') as f:
                run_data = f['processing/running/speed/data'][:]
                valid_run = run_data[~np.isnan(run_data)]
                if len(valid_run) > 2000:
                    rng = np.random.RandomState(int(expt_id) % (2**31))
                    valid_run = rng.choice(valid_run, 2000, replace=False)
                all_running.append(valid_run)
                
                try:
                    pupil_data = f['acquisition/EyeTracking/pupil_tracking/area'][:]
                    valid_pupil = pupil_data[~np.isnan(pupil_data)]
                    if len(valid_pupil) > 2000:
                        rng = np.random.RandomState(int(expt_id) % (2**31))
                        valid_pupil = rng.choice(valid_pupil, 2000, replace=False)
                    all_pupil.append(valid_pupil)
                except KeyError:
                    pass
                
                stim_keys = [k for k in f['intervals'].keys() if 'Natural_Images' in k]
                if stim_keys:
                    imgs = f[f'intervals/{stim_keys[0]}/image_name'][:]
                    for img in imgs:
                        if isinstance(img, bytes): img = img.decode()
                        if img != 'omitted': all_images.add(img)
        except Exception as e:
            print(f'  Error scanning {expt_id}: {e}')
    
    all_running = np.concatenate(all_running)
    running_percentiles = np.percentile(all_running, np.linspace(0, 100, 6))
    print(f'  Running percentiles: {running_percentiles}')
    
    if all_pupil:
        all_pupil = np.concatenate(all_pupil)
        pupil_percentiles = np.percentile(all_pupil, np.linspace(0, 100, 6))
        print(f'  Pupil percentiles: {pupil_percentiles}')
    else:
        pupil_percentiles = None
    
    all_images = sorted(all_images)
    print(f'  Unique images ({len(all_images)}): {all_images}')
    
    return {
        'running_percentiles': running_percentiles,
        'pupil_percentiles': pupil_percentiles,
        'all_images': all_images
    }


def create_processing_plots(result, expt_id, all_image_names, image_to_idx,
                            running_percentiles, pupil_percentiles):
    """Create processing visualization plots."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    raw = result['raw']
    ophys_ts = result['ophys_ts']
    trial_data = raw['trials']
    include_mask = trial_data['go'] | trial_data['catch']
    first_trial_idx = np.where(include_mask)[0][0]
    
    # Time window: first few included trials
    t_start = trial_data['start_time'][first_trial_idx] - 2
    t_end = min(t_start + 60, trial_data['stop_time'][np.where(include_mask)[0][-1]])
    
    fig, axes = plt.subplots(6, 1, figsize=(20, 24), sharex=True)
    
    # 1. Neural events at native rate
    events = raw['events'][raw['valid_roi']]
    mask = (ophys_ts >= t_start) & (ophys_ts < t_end)
    t = ophys_ts[mask]
    n_plot = min(3, events.shape[0])
    for i in range(n_plot):
        axes[0].plot(t, events[i, mask] + i*0.3, alpha=0.7, label=f'Neuron {i}')
    axes[0].set_ylabel('Events (native)')
    axes[0].set_title(f'Experiment {expt_id}')
    axes[0].legend(fontsize=8)
    
    # 2. Resampled neural events (first trial)
    trial_neural = result['neural'][0]
    ts0 = trial_data['start_time'][first_trial_idx]
    te0 = trial_data['stop_time'][first_trial_idx]
    cts = np.arange(ts0, te0, COMMON_DT)
    for i in range(n_plot):
        axes[1].plot(cts, trial_neural[i, :len(cts)] + i*0.3, 'o-', markersize=2, alpha=0.7)
    axes[1].set_ylabel('Events (resampled)')
    
    # 3. Image identity (first trial)
    trial_output = result['output'][0]
    axes[2].plot(cts, trial_output[0, :len(cts)], 'b-', alpha=0.7)
    axes[2].set_ylabel('Image ID')
    
    # 4. Image change (first trial)
    axes[3].plot(cts, trial_output[1, :len(cts)], 'r-', alpha=0.7)
    axes[3].set_ylabel('Image Change')
    
    # 5. Running speed
    run_mask = (raw['running_timestamps'] >= t_start) & (raw['running_timestamps'] < t_end)
    axes[4].plot(raw['running_timestamps'][run_mask], raw['running_speed'][run_mask], 
                 'b-', alpha=0.3, label='Raw')
    axes[4].set_ylabel('Running Speed')
    axes[4].legend(fontsize=8)
    
    # 6. Running binned + pupil binned
    axes[5].plot(cts, trial_output[2, :len(cts)], 'g-', alpha=0.7, label='Running bin')
    axes[5].plot(cts, trial_output[3, :len(cts)], 'm-', alpha=0.7, label='Pupil bin')
    axes[5].set_ylabel('Binned outputs')
    axes[5].set_xlabel('Time (s)')
    axes[5].legend(fontsize=8)
    
    # Add trial boundaries
    for ax in axes:
        for i in np.where(include_mask)[0][:10]:  # first 10 trials
            ts = trial_data['start_time'][i]
            te = trial_data['stop_time'][i]
            if ts >= t_start and ts < t_end:
                color = 'green' if trial_data['go'][i] else 'orange'
                ax.axvspan(ts, min(te, t_end), alpha=0.1, color=color)
                ct = trial_data['change_time'][i]
                if not np.isnan(ct) and ct >= t_start and ct < t_end:
                    ax.axvline(ct, color='red', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(f'processing_{expt_id}.png', dpi=100, bbox_inches='tight')
    plt.close()
    print(f'  Saved processing_{expt_id}.png')


def main():
    parser = argparse.ArgumentParser(description='Convert Visual Behavior 2P data')
    parser.add_argument('output', help='Output pickle file path')
    parser.add_argument('--sample', action='store_true', help='Process only 2 sessions')
    parser.add_argument('--full', action='store_true', help='Process all sessions (default)')
    parser.add_argument('--show-processing', action='store_true', help='Save processing plots')
    args = parser.parse_args()
    
    t_total_start = time.time()
    
    expt_table = get_experiment_list(sample=args.sample)
    print(f'Processing {len(expt_table)} experiments...')
    print(f'Unique mice: {expt_table.mouse_id.nunique()}')
    print(f'Unique sessions: {expt_table.ophys_session_id.nunique()}')
    print(f'Common time bin: {COMMON_DT*1000:.2f} ms ({1/COMMON_DT:.2f} Hz)')
    
    global_stats = collect_global_stats(expt_table)
    all_image_names = global_stats['all_images']
    image_to_idx = {img: i for i, img in enumerate(all_image_names)}
    running_percentiles = global_stats['running_percentiles']
    pupil_percentiles = global_stats['pupil_percentiles']
    
    all_neural = []
    all_input = []
    all_output = []
    all_brain_region_idx = []
    unique_subjects = []
    subject_idx_list = []
    unique_regions = []
    
    for i, (_, expt_info) in enumerate(expt_table.iterrows()):
        expt_id = expt_info['ophys_experiment_id']
        print(f'\n[{i+1}/{len(expt_table)}] Processing experiment {expt_id} '
              f'({expt_info["equipment_name"]}, {expt_info["cre_line"]})...')
        
        try:
            result = process_experiment(
                expt_id, expt_info, all_image_names, image_to_idx,
                running_percentiles, pupil_percentiles,
                show_processing=args.show_processing
            )
        except Exception as e:
            print(f'  ERROR: {e}')
            import traceback; traceback.print_exc()
            continue
        
        if result is None:
            continue
        
        all_neural.append(result['neural'])
        all_input.append([np.zeros((0, t.shape[1]), dtype=np.float32) for t in result['neural']])
        all_output.append(result['output'])
        
        subj = str(result['subject_id'])
        if subj not in unique_subjects:
            unique_subjects.append(subj)
        subject_idx_list.append(unique_subjects.index(subj))
        
        region = result['targeted_structure']
        if region not in unique_regions:
            unique_regions.append(region)
        all_brain_region_idx.append(
            np.full(result['n_neurons'], unique_regions.index(region), dtype=np.int64)
        )
        
        if args.show_processing:
            try:
                create_processing_plots(result, expt_id, all_image_names, image_to_idx,
                                        running_percentiles, pupil_percentiles)
            except Exception as e:
                print(f'  Plot error: {e}')
    
    if not all_neural:
        print('ERROR: No experiments processed!'); sys.exit(1)
    
    output_names = ['image_identity', 'image_change', 'running_speed', 'pupil_diameter', 'trial_outcome']
    output_values = [
        all_image_names,
        ['no_change', 'change'],
        [f'bin_{i}' for i in range(5)],
        [f'bin_{i}' for i in range(5)],
        ['hit', 'miss', 'correct_reject', 'false_alarm'],
    ]
    
    data = {
        'neural': all_neural,
        'input': all_input,
        'output': all_output,
        'subjects': unique_subjects,
        'subject_idx': np.array(subject_idx_list, dtype=np.int64),
        'brain_regions': unique_regions,
        'brain_region_idx': all_brain_region_idx,
        'input_names': [],
        'output_names': output_names,
        'output_values': output_values,
        'metadata': {
            'task_description': 'Visual change detection task: mice detect changes in natural image identity. Go trials have real image change, Catch trials have sham change (same image).',
            'time_bin_size': COMMON_DT * 1000,
            'temporal_alignment_event': 'Trial start time',
            'off_start': 0.0,
            'off_end': None,
            'dataset': 'Allen Brain Observatory Visual Behavior 2P',
            'neural_data_type': 'detected calcium events (raw, unfiltered)',
            'n_sessions': len(all_neural),
            'n_subjects': len(unique_subjects),
            'running_percentiles': running_percentiles.tolist(),
            'pupil_percentiles': pupil_percentiles.tolist() if pupil_percentiles is not None else None,
            'image_names': all_image_names,
        }
    }
    
    total_trials = sum(len(s) for s in all_neural)
    total_neurons = sum(len(r) for r in all_brain_region_idx)
    print(f'\n=== Conversion Summary ===')
    print(f'Sessions: {len(all_neural)}')
    print(f'Subjects: {len(unique_subjects)}')
    print(f'Brain regions: {unique_regions}')
    print(f'Total trials: {total_trials}')
    print(f'Total neurons: {total_neurons}')
    print(f'Time bin size: {COMMON_DT*1000:.2f} ms ({1/COMMON_DT:.2f} Hz)')
    print(f'Output names: {output_names}')
    print(f'Image names ({len(all_image_names)}): {all_image_names}')
    
    print(f'\nSaving to {args.output}...')
    with open(args.output, 'wb') as f:
        pickle.dump(data, f, protocol=4)
    
    file_size = os.path.getsize(args.output) / (1024**2)
    print(f'Saved {args.output} ({file_size:.1f} MB)')
    print(f'Total time: {time.time() - t_total_start:.1f}s')


if __name__ == '__main__':
    main()
