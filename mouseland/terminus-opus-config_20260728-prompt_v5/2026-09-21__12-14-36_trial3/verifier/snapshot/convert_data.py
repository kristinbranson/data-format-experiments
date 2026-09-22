#!/usr/bin/env python3
"""Convert Zhong et al. 2025 data to decoder-compatible format.

Usage:
    python -u convert_data.py <outpicklefile> [--full|--sample] [--show-processing]
"""

import sys
import os
import time
import argparse
import pickle
import numpy as np
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

DATA_ROOT = '/app/data'

# Brain region mapping (from utils.py neu_area_ID)
BRAIN_REGION_NAMES = ['V1', 'mHV', 'aHV', 'lHV', 'unassigned']
IAREA_TO_REGION = {8: 0}  # V1
for ia in [0, 1, 2, 9]: IAREA_TO_REGION[ia] = 1  # mHV
for ia in [3, 4]: IAREA_TO_REGION[ia] = 2  # aHV
for ia in [5, 6]: IAREA_TO_REGION[ia] = 3  # lHV

# Global stimulus mapping
ALL_STIMULI = sorted(['circle1', 'circle2', 'circle3', 'leaf1', 'leaf1_swap1', 'leaf1_swap2',
                      'leaf2', 'leaf3', 'rock1', 'rock2', 'wood1', 'wood1_swap1',
                      'wood1_swap2', 'wood2', 'wood5'])
STIM_TO_IDX = {s: i for i, s in enumerate(ALL_STIMULI)}

_beh_cache = {}


def load_spk(mname, datexp, blk):
    fn = os.path.join(DATA_ROOT, 'spk', f'{mname}_{datexp}_{blk}_neural_data.npy')
    data = np.load(fn, allow_pickle=True).item()
    return np.concatenate(data['spks'], 0)


def load_retino(mname, datexp):
    fn = os.path.join(DATA_ROOT, 'retinotopy', f'{mname}_{datexp}_trans.npz')
    if not os.path.exists(fn):
        return None
    return np.load(fn, allow_pickle=True)['iarea']


def iarea_to_region_idx(iarea):
    region_idx = np.full(len(iarea), 4, dtype=np.int64)
    for ia_code, reg_idx in IAREA_TO_REGION.items():
        region_idx[iarea == ia_code] = reg_idx
    return region_idx


def get_all_unique_recordings():
    exp_info = np.load(os.path.join(DATA_ROOT, 'beh', 'Imaging_Exp_info.npy'), allow_pickle=True).item()
    seen = set()
    recordings = []
    for exp_type, sessions in exp_info.items():
        for s in sessions:
            key = f"{s['mname']}_{s['datexp']}_{s['blk']}"
            if key not in seen:
                seen.add(key)
                recordings.append((key, s, exp_type))
    return recordings


def load_beh_for_recording(key, exp_type):
    beh_file = f'Beh_{exp_type}.npy'
    if beh_file not in _beh_cache:
        _beh_cache[beh_file] = np.load(
            os.path.join(DATA_ROOT, 'beh', beh_file), allow_pickle=True).item()
    beh_dict = _beh_cache[beh_file]
    if key in beh_dict:
        return beh_dict[key]
    for beh_key in beh_dict:
        if beh_key.startswith(key):
            return beh_dict[beh_key]
    return None


def get_training_day(db_info):
    if 'days' in db_info:
        return float(db_info['days'])
    if 'sess#' in db_info:
        return float(db_info['sess#'])
    return 0.0


def compute_running_speed_quartiles(recordings):
    print('Computing running speed quartiles...')
    t0 = time.time()
    all_speeds = []
    for key, db_info, exp_type in recordings:
        beh = load_beh_for_recording(key, exp_type)
        if beh is None:
            continue
        n = min(len(beh['ft_RunSpeed']), len(beh['ft_CorrSpc']), len(beh['ft_move']))
        running_corridor = beh['ft_CorrSpc'][:n] & (beh['ft_move'][:n] > 0)
        corridor_speeds = beh['ft_RunSpeed'][:n][running_corridor]
        all_speeds.append(corridor_speeds)
    all_speeds = np.concatenate(all_speeds)
    bins = np.percentile(all_speeds, [25, 50, 75])
    print(f'  Q25={bins[0]:.2f}, Q50={bins[1]:.2f}, Q75={bins[2]:.2f}')
    print(f'  Range: [{all_speeds.min():.2f}, {all_speeds.max():.2f}], computed in {time.time()-t0:.1f}s')
    return bins


def process_session(key, db_info, exp_type, speed_bins):
    """Process a single recording session efficiently."""
    mname, datexp, blk = db_info['mname'], db_info['datexp'], db_info['blk']
    
    beh = load_beh_for_recording(key, exp_type)
    if beh is None:
        return None
    
    t0 = time.time()
    spk = load_spk(mname, datexp, blk)
    n_neurons, n_frames = spk.shape
    t_load = time.time() - t0
    
    iarea = load_retino(mname, datexp)
    brain_region_idx = iarea_to_region_idx(iarea) if iarea is not None else np.full(n_neurons, 4, dtype=np.int64)
    
    ntrials = beh['ntrials']
    n_use = min(n_frames, len(beh['ft']))
    
    ft_Pos = beh['ft_Pos'][:n_use]
    ft_CorrSpc = beh['ft_CorrSpc'][:n_use]
    ft_RunSpeed = beh['ft_RunSpeed'][:n_use]
    ft_trInd = beh['ft_trInd'][:n_use]
    ft = beh['ft'][:n_use]
    
    sound_fr = beh['SoundFr']
    is_rew = beh['isRew']
    wall_name = beh['WallName']
    lick_fr = beh['LickFr']
    lick_trind = beh['LickTrind']
    
    training_day = get_training_day(db_info)
    dt = float(np.median(np.diff(ft)) * 86400)
    
    # Pre-compute lick lookup: trial -> list of frame indices
    lick_by_trial = defaultdict(list)
    for li in range(len(lick_fr)):
        lick_by_trial[int(lick_trind[li])].append(lick_fr[li])
    
    # Pre-compute masks
    corridor_mask = ft_CorrSpc.copy()
    corridor_mask[n_use:] = False  # safety
    running_mask = beh['ft_move'][:n_use] > 0  # only running frames, matching reference paper
    
    neural_trials = []
    input_trials = []
    output_trials = []
    
    t1 = time.time()
    
    for trial in range(ntrials):
        trial_mask = (ft_trInd == trial) & corridor_mask & running_mask
        trial_frames = np.where(trial_mask)[0]
        
        if len(trial_frames) < 2:
            continue
        
        n_t = len(trial_frames)
        
        # Neural data as float32
        neural = spk[:, trial_frames].astype(np.float32)  # keep float32 for decoder compatibility
        
        # Inputs
        time_to_cue = ((trial_frames - sound_fr[trial]) * dt).astype(np.float32)
        day_arr = np.full(n_t, training_day, dtype=np.float32)
        time_since_start = ((trial_frames - trial_frames[0]) * dt).astype(np.float32)
        reward_arr = np.full(n_t, float(is_rew[trial]), dtype=np.float32)
        inp = np.stack([time_to_cue, day_arr, time_since_start, reward_arr], axis=0)
        
        # Outputs (int64)
        stim_idx = STIM_TO_IDX.get(str(wall_name[trial]), 0)
        stim_arr = np.full(n_t, stim_idx, dtype=np.int64)
        
        lick_binary = np.zeros(n_t, dtype=np.int64)
        if trial in lick_by_trial:
            for lf in lick_by_trial[trial]:
                diffs = np.abs(trial_frames - lf)
                closest = np.argmin(diffs)
                if diffs[closest] < 1.0:
                    lick_binary[closest] = 1
        
        pos = ft_Pos[trial_frames]
        pos_binned = np.clip(np.floor(pos / 10.0).astype(np.int64), 0, 3)
        
        speed = ft_RunSpeed[trial_frames]
        speed_binned = np.clip(np.digitize(speed, speed_bins), 0, 3).astype(np.int64)
        
        out = np.stack([stim_arr, lick_binary, pos_binned, speed_binned], axis=0)
        
        neural_trials.append(neural)
        input_trials.append(inp)
        output_trials.append(out)
    
    t_proc = time.time() - t1
    
    if len(neural_trials) < 2:
        return None
    
    return {
        'neural': neural_trials,
        'input': input_trials,
        'output': output_trials,
        'brain_region_idx': brain_region_idx,
        'n_neurons': n_neurons,
        'mname': mname,
        'key': key,
        'exp_type': exp_type,
        'ntrials_original': ntrials,
        'ntrials_valid': len(neural_trials),
        'dt': dt,
        't_load': t_load,
        't_proc': t_proc,
    }


def convert_data(output_file, sample=False, show_processing=False):
    t_start = time.time()
    
    all_recordings = get_all_unique_recordings()
    print(f'Total unique recordings: {len(all_recordings)}')
    
    speed_bins = compute_running_speed_quartiles(all_recordings)
    _beh_cache.clear()
    
    if sample:
        # Pick diverse sessions
        sup_rec, unsup_rec = None, None
        for rec in all_recordings:
            key, db_info, exp_type = rec
            if 'sup_test' in exp_type and sup_rec is None:
                sup_rec = rec
            elif 'unsup_train1_after' in exp_type and unsup_rec is None:
                unsup_rec = rec
            if sup_rec and unsup_rec:
                break
        recordings = [r for r in [unsup_rec, sup_rec] if r is not None]
        if len(recordings) < 2:
            recordings = all_recordings[:2]
        print(f'Sample mode: {len(recordings)} sessions')
    else:
        recordings = all_recordings
    
    all_neural, all_input, all_output = [], [], []
    all_brain_region_idx, all_subject_idx = [], []
    subjects, subject_to_idx = [], {}
    session_info, dt_values = [], []
    
    for i, (key, db_info, exp_type) in enumerate(recordings):
        t0 = time.time()
        
        result = process_session(key, db_info, exp_type, speed_bins)
        
        if result is None:
            print(f'[{i+1}/{len(recordings)}] {key} SKIPPED')
            continue
        
        mname = result['mname']
        if mname not in subject_to_idx:
            subject_to_idx[mname] = len(subjects)
            subjects.append(mname)
        
        all_neural.append(result['neural'])
        all_input.append(result['input'])
        all_output.append(result['output'])
        all_brain_region_idx.append(result['brain_region_idx'])
        all_subject_idx.append(subject_to_idx[mname])
        dt_values.append(result['dt'])
        
        session_info.append({
            'key': key, 'exp_type': exp_type, 'mname': mname,
            'n_neurons': result['n_neurons'],
            'ntrials_original': result['ntrials_original'],
            'ntrials_valid': result['ntrials_valid'],
        })
        
        elapsed = time.time() - t0
        print(f'[{i+1}/{len(recordings)}] {key} ({exp_type}) '
              f'{result["ntrials_valid"]}/{result["ntrials_original"]} trials, '
              f'{result["n_neurons"]} neurons, '
              f'load={result["t_load"]:.1f}s proc={result["t_proc"]:.1f}s total={elapsed:.1f}s')
        
        if (i + 1) % 10 == 0:
            _beh_cache.clear()
    
    _beh_cache.clear()
    
    median_dt_ms = float(np.median(dt_values) * 1000)
    
    data = {
        'neural': all_neural,
        'input': all_input,
        'output': all_output,
        'subjects': subjects,
        'subject_idx': np.array(all_subject_idx, dtype=np.int64),
        'brain_regions': BRAIN_REGION_NAMES,
        'brain_region_idx': all_brain_region_idx,
        'input_names': ['time_to_sound_cue', 'day_of_training',
                       'time_since_trial_start', 'reward_availability'],
        'output_names': ['visual_stimulus', 'licking', 'position', 'running_speed'],
        'output_values': [
            ALL_STIMULI,
            ['no_lick', 'lick'],
            ['0-1m', '1-2m', '2-3m', '3-4m'],
            ['Q1', 'Q2', 'Q3', 'Q4'],
        ],
        'metadata': {
            'task_description': 'Visual discrimination in virtual reality corridors. '
                'Mice run through 4m corridors with naturalistic texture patterns. '
                'Sound cue at random position (0.5-3.5m) signals reward availability.',
            'time_bin_size': median_dt_ms,
            'temporal_alignment_event': 'Trial start (corridor entry)',
            'off_start': 0.0,
            'off_end': None,
            'session_info': session_info,
            'speed_quartile_boundaries': speed_bins.tolist(),
            'stimulus_mapping': STIM_TO_IDX,
            'n_sessions': len(all_neural),
            'n_subjects': len(subjects),
            'source': 'Zhong et al. 2025',
        }
    }
    
    print(f'\nSaving to {output_file}...')
    with open(output_file, 'wb') as f:
        pickle.dump(data, f, protocol=4)
    
    file_size = os.path.getsize(output_file) / (1024**3)
    total_trials = sum(len(s) for s in all_neural)
    total_neurons = sum(si['n_neurons'] for si in session_info)
    
    print(f'Saved: {file_size:.2f} GB')
    print(f'\n=== Summary ===')
    print(f'Sessions: {len(all_neural)}')
    print(f'Subjects: {len(subjects)}')
    print(f'Total trials: {total_trials}')
    print(f'Total neurons (sum): {total_neurons}')
    print(f'Mean neurons/session: {total_neurons/max(len(all_neural),1):.0f}')
    print(f'Mean trials/session: {total_trials/max(len(all_neural),1):.1f}')
    print(f'Time bin: {median_dt_ms:.2f} ms')
    print(f'Total time: {time.time()-t_start:.1f}s')
    
    if show_processing:
        create_processing_plots(data)
    
    return data


def create_processing_plots(data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    for sess_idx in range(min(2, len(data['neural']))):
        si = data['metadata']['session_info'][sess_idx]
        key = si['key']
        neural = data['neural'][sess_idx]
        inputs = data['input'][sess_idx]
        outputs = data['output'][sess_idx]
        
        fig, axes = plt.subplots(4, 2, figsize=(16, 20))
        fig.suptitle(f'Processing: {key} ({si["exp_type"]})', fontsize=14)
        
        for col, trial_idx in enumerate([0, min(len(neural)//2, len(neural)-1)]):
            n = neural[trial_idx]
            inp = inputs[trial_idx]
            out = outputs[trial_idx]
            
            ax = axes[0, col]
            n_show = min(50, n.shape[0])
            ax.imshow(n[:n_show], aspect='auto', cmap='hot')
            ax.set_title(f'Trial {trial_idx}: Neural ({n_show}/{n.shape[0]} neurons)')
            ax.set_xlabel('Time bins'); ax.set_ylabel('Neurons')
            
            ax = axes[1, col]
            for j in range(inp.shape[0]):
                ax.plot(inp[j], label=data['input_names'][j])
            ax.set_title(f'Trial {trial_idx}: Inputs')
            ax.legend(fontsize=6); ax.set_xlabel('Time bins')
            
            ax = axes[2, col]
            for j in range(out.shape[0]):
                ax.plot(out[j], label=data['output_names'][j], alpha=0.7)
            ax.set_title(f'Trial {trial_idx}: Outputs')
            ax.legend(fontsize=6); ax.set_xlabel('Time bins')
            
            ax = axes[3, col]
            ax.plot(out[2], label='Position bin', color='blue')
            lick_t = np.where(out[1] > 0)[0]
            if len(lick_t) > 0:
                ax.scatter(lick_t, np.ones_like(lick_t)*3.5, c='red', s=10, label='Licks')
            ax.plot(out[3], label='Speed bin', color='green', alpha=0.5)
            ax.set_title(f'Trial {trial_idx}: Position, Licking & Speed')
            ax.legend(fontsize=6); ax.set_xlabel('Time bins')
        
        plt.tight_layout()
        plt.savefig(f'processing_{key}.png', dpi=100)
        plt.close(fig)
        print(f'  Saved processing_{key}.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=str)
    parser.add_argument('--full', action='store_true', default=True)
    parser.add_argument('--sample', action='store_true')
    parser.add_argument('--show-processing', action='store_true')
    args = parser.parse_args()
    convert_data(args.output, sample=args.sample, show_processing=args.show_processing)
