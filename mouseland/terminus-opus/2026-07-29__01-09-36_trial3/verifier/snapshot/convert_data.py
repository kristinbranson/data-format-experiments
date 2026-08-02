#!/usr/bin/env python3
"""Convert Zhong et al. 2025 data to decoder-compatible format.

Usage:
    python -u convert_data.py <output_file> [--full|--sample] [--show-processing]
"""

import sys
import os
import time
import argparse
import pickle
import numpy as np
from pathlib import Path

DATA_ROOT = 'data'
FS = 3.17  # Frame rate in Hz
TIME_BIN_MS = 1000.0 / FS  # ~315.5 ms
MAX_NEURONS = 2000  # Max neurons per session

def neu_area_ID(iarea):
    idx = {}
    idx['V1'] = (iarea == 8)
    idx['mHV'] = (iarea == 0) | (iarea == 1) | (iarea == 2) | (iarea == 9)
    idx['lHV'] = (iarea == 5) | (iarea == 6)
    idx['aHV'] = (iarea == 3) | (iarea == 4)
    return idx

BRAIN_REGIONS = ['V1', 'mHV', 'lHV', 'aHV']
STIM_NAMES = {0: 'circle1', 1: 'circle2', 2: 'leaf1', 3: 'leaf2', 4: 'leaf3', 5: 'leaf1_swap1', 6: 'leaf1_swap2'}

def load_experiment_info():
    """Load experiment info and build session metadata."""
    exp_info = np.load(os.path.join(DATA_ROOT, 'beh', 'Imaging_Exp_info.npy'),
                       allow_pickle=True).item()
    session_meta = {}
    for exp_type, sessions in exp_info.items():
        for db in sessions:
            sid = f"{db['mname']}_{db['datexp']}_{db['blk']}"
            day_key = 'days' if 'days' in db else 'sess#'
            day_val = db.get(day_key, 0)
            stimtype = db.get('stimtype', None)
            
            if sid not in session_meta:
                session_meta[sid] = {
                    'mname': db['mname'],
                    'datexp': db['datexp'],
                    'blk': db['blk'],
                    'exptype': db.get('exptype', ''),
                    'rewType': db.get('rewType', ''),
                    'day_of_training': day_val,
                    'exp_types': [],
                    'beh_files': [],
                    'beh_keys': [],  # possible keys in beh files
                    'stim_id': db.get('stim_id', None),
                }
            session_meta[sid]['exp_types'].append(exp_type)
            beh_file = f'Beh_{exp_type}.npy'
            
            # Build possible behavioral data keys
            if stimtype:
                beh_key = f"{sid}_{stimtype}"
            else:
                beh_key = sid
            
            if beh_file not in session_meta[sid]['beh_files']:
                session_meta[sid]['beh_files'].append(beh_file)
            if beh_key not in session_meta[sid]['beh_keys']:
                session_meta[sid]['beh_keys'].append(beh_key)
    
    return session_meta

def load_spk(mname, datexp, blk):
    fn = f'{mname}_{datexp}_{blk}_neural_data.npy'
    spk_path = os.path.join(DATA_ROOT, 'spk', fn)
    spk_data = np.load(spk_path, allow_pickle=True).item()
    spk = np.concatenate(spk_data['spks'], axis=0)
    return spk

def load_beh(session_id, beh_files, beh_keys):
    """Load behavioral data for a session, trying multiple key formats."""
    # Try each beh file with each possible key
    for beh_file in beh_files:
        beh_path = os.path.join(DATA_ROOT, 'beh', beh_file)
        if not os.path.exists(beh_path):
            continue
        beh_all = np.load(beh_path, allow_pickle=True).item()
        
        # Try the session_id directly first
        if session_id in beh_all:
            return beh_all[session_id]
        
        # Try all possible keys
        for key in beh_keys:
            if key in beh_all:
                return beh_all[key]
    
    raise ValueError(f"Session {session_id} not found. Tried keys: {beh_keys} in files: {beh_files}")

def load_retinotopy(mname, datexp):
    fn = f'{mname}_{datexp}_trans.npz'
    ret_path = os.path.join(DATA_ROOT, 'retinotopy', fn)
    dtrans = np.load(ret_path, allow_pickle=True)
    return dtrans['iarea']

def filter_and_subsample_neurons(spk, iarea, max_neurons=MAX_NEURONS, rng=None):
    """Filter neurons to visual areas and subsample if needed."""
    if rng is None:
        rng = np.random.RandomState(42)
    
    valid_mask = (iarea != -1) & (iarea != 7)
    valid_indices = np.where(valid_mask)[0]
    n_valid = len(valid_indices)
    
    if n_valid <= max_neurons:
        selected_indices = valid_indices
    else:
        region_masks = neu_area_ID(iarea[valid_indices])
        selected_local = []
        region_counts = {r: mask.sum() for r, mask in region_masks.items()}
        total_in_regions = sum(region_counts.values())
        
        for region in BRAIN_REGIONS:
            region_local_idx = np.where(region_masks[region])[0]
            n_region = len(region_local_idx)
            if n_region == 0:
                continue
            n_sample = max(1, int(np.round(max_neurons * n_region / total_in_regions)))
            n_sample = min(n_sample, n_region)
            sampled = rng.choice(region_local_idx, size=n_sample, replace=False)
            selected_local.extend(sampled)
        
        selected_local = np.array(selected_local)
        if len(selected_local) > max_neurons:
            selected_local = rng.choice(selected_local, size=max_neurons, replace=False)
        
        selected_indices = valid_indices[selected_local]
    
    selected_indices = np.sort(selected_indices)
    return spk[selected_indices], iarea[selected_indices], selected_indices

def get_brain_region_idx(iarea):
    region_masks = neu_area_ID(iarea)
    n_neurons = len(iarea)
    region_idx = np.zeros(n_neurons, dtype=np.int64)
    for i, region in enumerate(BRAIN_REGIONS):
        region_idx[region_masks[region]] = i
    return region_idx

def compute_running_speed_quartiles(session_meta, session_ids):
    """Compute global running speed quartiles."""
    print("Computing global running speed quartiles...")
    all_speeds = []
    for sid in session_ids:
        meta = session_meta[sid]
        beh = load_beh(sid, meta['beh_files'], meta['beh_keys'])
        speeds = beh['ft_RunSpeed']
        all_speeds.append(speeds)
    all_speeds = np.concatenate(all_speeds)
    quartiles = np.percentile(all_speeds, [25, 50, 75])
    print(f"  Speed quartiles: {quartiles}")
    print(f"  Speed range: [{all_speeds.min():.2f}, {all_speeds.max():.2f}]")
    return quartiles

def process_session(sid, meta, speed_quartiles, show_processing=False):
    """Process a single session."""
    t0 = time.time()
    mname, datexp, blk = meta['mname'], meta['datexp'], meta['blk']
    
    # Load neural data
    print(f"  Loading neural data...")
    t_load = time.time()
    spk = load_spk(mname, datexp, blk)
    n_neurons_raw, n_frames_spk = spk.shape
    print(f"    Raw: {n_neurons_raw} neurons x {n_frames_spk} frames ({time.time()-t_load:.1f}s)")
    
    # Load retinotopy and filter/subsample
    iarea = load_retinotopy(mname, datexp)
    assert len(iarea) == n_neurons_raw, f"Neuron count mismatch: iarea={len(iarea)}, spk={n_neurons_raw}"
    
    spk, iarea_filtered, selected_idx = filter_and_subsample_neurons(spk, iarea)
    n_neurons = spk.shape[0]
    region_idx = get_brain_region_idx(iarea_filtered)
    print(f"    After filtering: {n_neurons} neurons")
    
    # Load behavioral data
    print(f"  Loading behavioral data...")
    beh = load_beh(sid, meta['beh_files'], meta['beh_keys'])
    
    ntrials = beh['ntrials']
    n_frames_beh = len(beh['ft'])
    n_frames = min(n_frames_spk, n_frames_beh)
    
    ft_trInd = beh['ft_trInd'][:n_frames]
    ft_Pos = beh['ft_Pos'][:n_frames]
    ft_RunSpeed = beh['ft_RunSpeed'][:n_frames]
    ft_CorrSpc = beh['ft_CorrSpc'][:n_frames]
    
    sound_fr = beh['SoundFr']
    start_fr = beh['StartFr']
    is_rew = beh['isRew']
    wall_name = beh['WallName']
    
    lick_fr = beh['LickFr']
    lick_trind = beh['LickTrind']
    
    day_of_training = float(meta['day_of_training'])
    
    # Build stimulus name mapping
    uniq_walls = beh['UniqWalls']
    stim_id_map = beh.get('stim_id', None)
    wall_to_stim = {}
    if stim_id_map is not None:
        for i, wall in enumerate(uniq_walls):
            if i < len(stim_id_map):
                sid_val = stim_id_map[i]
                if not np.isnan(sid_val):
                    wall_to_stim[wall] = STIM_NAMES.get(int(sid_val), wall)
                else:
                    wall_to_stim[wall] = wall
            else:
                wall_to_stim[wall] = wall
    else:
        for wall in uniq_walls:
            wall_to_stim[wall] = wall
    
    # Process trials
    neural_trials = []
    input_trials = []
    output_trials = []
    skipped = 0
    
    for trial_idx in range(ntrials):
        trial_mask = (ft_trInd == trial_idx)
        frame_indices = np.where(trial_mask)[0]
        
        if len(frame_indices) < 2:
            skipped += 1
            continue
        
        # Ensure frame indices are within neural data range
        frame_indices = frame_indices[frame_indices < n_frames_spk]
        if len(frame_indices) < 2:
            skipped += 1
            continue
        
        n_tp = len(frame_indices)
        
        # Neural data
        trial_neural = spk[:, frame_indices].astype(np.float32)
        
        # --- INPUTS ---
        trial_sound_fr = sound_fr[trial_idx]
        time_to_sound = (trial_sound_fr - frame_indices.astype(np.float64)) / FS
        
        trial_start = start_fr[trial_idx]
        time_since_start = (frame_indices.astype(np.float64) - trial_start) / FS
        
        trial_input = np.zeros((4, n_tp), dtype=np.float32)
        trial_input[0, :] = time_to_sound.astype(np.float32)
        trial_input[1, :] = day_of_training
        trial_input[2, :] = time_since_start.astype(np.float32)
        trial_input[3, :] = float(is_rew[trial_idx])
        
        # --- OUTPUTS ---
        stim_name = wall_to_stim.get(wall_name[trial_idx], wall_name[trial_idx])
        
        # Licking binary
        trial_lick_mask = (lick_trind == trial_idx)
        trial_lick_frames = lick_fr[trial_lick_mask]
        lick_binary = np.zeros(n_tp, dtype=np.int64)
        for lf in trial_lick_frames:
            lf_int = int(np.round(lf))
            pos = np.searchsorted(frame_indices, lf_int)
            if pos < n_tp and frame_indices[pos] == lf_int:
                lick_binary[pos] = 1
            elif pos > 0 and frame_indices[pos-1] == lf_int:
                lick_binary[pos-1] = 1
        
        # Position bin
        trial_pos = ft_Pos[frame_indices]
        pos_clipped = np.clip(trial_pos, 0, 39.999)
        pos_bin = np.floor(pos_clipped / 10.0).astype(np.int64)
        pos_bin = np.clip(pos_bin, 0, 3)
        
        # Speed bin
        trial_speed = ft_RunSpeed[frame_indices]
        speed_bin = np.digitize(trial_speed, speed_quartiles).astype(np.int64)
        speed_bin = np.clip(speed_bin, 0, 3)
        
        trial_output = np.zeros((4, n_tp), dtype=np.int64)
        trial_output[0, :] = -1  # placeholder
        trial_output[1, :] = lick_binary
        trial_output[2, :] = pos_bin
        trial_output[3, :] = speed_bin
        
        neural_trials.append(trial_neural)
        input_trials.append(trial_input)
        output_trials.append((trial_output, stim_name))
    
    if skipped > 0:
        print(f"    Skipped {skipped} trials with < 2 frames")
    print(f"  {sid}: {len(neural_trials)} trials, {n_neurons} neurons ({time.time()-t0:.1f}s)")
    
    return neural_trials, input_trials, output_trials, region_idx

def build_stimulus_mapping(all_output_trials):
    all_stim_names = set()
    for session_outputs in all_output_trials:
        for _, stim_name in session_outputs:
            all_stim_names.add(stim_name)
    sorted_stim = sorted(all_stim_names)
    stim_to_idx = {name: idx for idx, name in enumerate(sorted_stim)}
    return stim_to_idx, sorted_stim

def main():
    parser = argparse.ArgumentParser(description='Convert Zhong et al. 2025 data')
    parser.add_argument('output', help='Output pickle file path')
    parser.add_argument('--full', action='store_true', default=True)
    parser.add_argument('--sample', action='store_true')
    parser.add_argument('--show-processing', action='store_true')
    args = parser.parse_args()
    if args.sample:
        args.full = False
    
    print("=" * 60)
    print("Zhong et al. 2025 Data Conversion")
    print("=" * 60)
    t_start = time.time()
    
    print("\nLoading experiment info...")
    session_meta = load_experiment_info()
    all_session_ids = sorted(session_meta.keys())
    print(f"  Found {len(all_session_ids)} unique sessions")
    
    if args.sample:
        sample_ids = []
        for sid in all_session_ids:
            meta = session_meta[sid]
            if meta['exptype'] == 'sup' and len(sample_ids) == 0:
                sample_ids.append(sid)
            elif meta['exptype'] == 'unsup' and len(sample_ids) == 1:
                sample_ids.append(sid)
            if len(sample_ids) == 2:
                break
        session_ids = sample_ids
        print(f"  Sample mode: {session_ids}")
    else:
        session_ids = all_session_ids
        print(f"  Full mode: {len(session_ids)} sessions")
    
    print("\nComputing running speed quartiles...")
    speed_quartiles = compute_running_speed_quartiles(session_meta, session_ids)
    
    subjects_set = set()
    for sid in session_ids:
        subjects_set.add(session_meta[sid]['mname'])
    subjects_list = sorted(subjects_set)
    subject_to_idx = {name: idx for idx, name in enumerate(subjects_list)}
    
    print("\nProcessing sessions...")
    all_neural, all_input, all_output_raw = [], [], []
    all_region_idx, session_subject_map = [], []
    
    for i, sid in enumerate(session_ids):
        print(f"\n[{i+1}/{len(session_ids)}] {sid}")
        meta = session_meta[sid]
        neural_trials, input_trials, output_trials, region_idx = process_session(
            sid, meta, speed_quartiles, show_processing=args.show_processing
        )
        all_neural.append(neural_trials)
        all_input.append(input_trials)
        all_output_raw.append(output_trials)
        all_region_idx.append(region_idx)
        session_subject_map.append(subject_to_idx[meta['mname']])
    
    print("\nBuilding stimulus mapping...")
    stim_to_idx, stim_names_sorted = build_stimulus_mapping(all_output_raw)
    print(f"  Stimuli: {stim_names_sorted}")
    
    all_output = []
    for session_outputs in all_output_raw:
        session_out = []
        for output_arr, stim_name in session_outputs:
            stim_idx = stim_to_idx[stim_name]
            output_arr[0, :] = stim_idx
            session_out.append(output_arr)
        all_output.append(session_out)
    
    pos_bin_names = ['0-1m', '1-2m', '2-3m', '3-4m']
    speed_bin_names = ['Q1', 'Q2', 'Q3', 'Q4']
    output_values = [
        stim_names_sorted,
        ['no_lick', 'lick'],
        pos_bin_names,
        speed_bin_names,
    ]
    
    data = {
        'neural': all_neural,
        'input': all_input,
        'output': all_output,
        'subjects': subjects_list,
        'subject_idx': np.array(session_subject_map, dtype=np.int64),
        'brain_regions': list(BRAIN_REGIONS),
        'brain_region_idx': all_region_idx,
        'input_names': ['time_to_sound_cue', 'day_of_training', 'time_since_trial_start', 'reward_availability'],
        'output_names': ['visual_stimulus_category', 'licking', 'position_bin', 'running_speed_bin'],
        'output_values': output_values,
        'metadata': {
            'task_description': 'Visual discrimination in VR corridors with naturalistic textures. Mice discriminate leaf vs circle patterns.',
            'time_bin_size': TIME_BIN_MS,
            'temporal_alignment_event': 'Trial start (corridor entry)',
            'off_start': 0.0,
            'off_end': None,
            'frame_rate_hz': FS,
            'corridor_length_m': 4.0,
            'gray_space_length_m': 2.0,
            'vr_speed_cm_s': 60.0,
            'running_threshold_cm_s': 6.0,
            'speed_quartile_boundaries': speed_quartiles.tolist(),
            'n_sessions': len(session_ids),
            'n_subjects': len(subjects_list),
            'max_neurons_per_session': MAX_NEURONS,
            'neuron_filtering': 'Excluded neurons outside visual cortex (iarea==-1 or iarea==7). Subsampled to max 2000 per session, stratified by brain region.',
        }
    }
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    total_trials = sum(len(s) for s in all_neural)
    print(f"Sessions: {len(all_neural)}")
    print(f"Subjects: {len(subjects_list)} - {subjects_list}")
    print(f"Total trials: {total_trials}")
    print(f"Trials/session: min={min(len(s) for s in all_neural)}, max={max(len(s) for s in all_neural)}, mean={total_trials/len(all_neural):.0f}")
    neurons_per_session = [s[0].shape[0] if len(s) > 0 else 0 for s in all_neural]
    print(f"Neurons/session: min={min(neurons_per_session)}, max={max(neurons_per_session)}, mean={np.mean(neurons_per_session):.0f}")
    print(f"Brain regions: {BRAIN_REGIONS}")
    print(f"Stimuli: {stim_names_sorted}")
    print(f"Speed quartiles: {speed_quartiles}")
    
    print("\nOutput distributions:")
    for out_idx, out_name in enumerate(data['output_names']):
        all_vals = np.concatenate([trial[out_idx] for session in all_output for trial in session])
        unique, counts = np.unique(all_vals, return_counts=True)
        total = len(all_vals)
        print(f"  {out_name}: " + ", ".join(f"{data['output_values'][out_idx][int(u)]}={c/total*100:.1f}%" for u, c in zip(unique, counts)))
    
    print(f"\nSaving to {args.output}...")
    with open(args.output, 'wb') as f:
        pickle.dump(data, f, protocol=4)
    file_size = os.path.getsize(args.output)
    print(f"  File size: {file_size / 1e9:.2f} GB")
    print(f"Total time: {time.time() - t_start:.1f}s")
    
    if args.show_processing:
        plot_processing(data, session_ids[:2])

def plot_processing(data, session_ids):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    for s_idx in range(min(2, len(data['neural']))):
        sid = session_ids[s_idx]
        fig, axes = plt.subplots(7, 1, figsize=(16, 24), sharex=True)
        fig.suptitle(f'Processing: {sid}', fontsize=14)
        
        n_show = min(10, len(data['neural'][s_idx]))
        neural_cat, input_cat, output_cat = [], [], []
        boundaries = [0]
        
        for t in range(n_show):
            neural_cat.append(data['neural'][s_idx][t])
            input_cat.append(data['input'][s_idx][t])
            output_cat.append(data['output'][s_idx][t])
            boundaries.append(boundaries[-1] + data['neural'][s_idx][t].shape[1])
        
        nc = np.concatenate(neural_cat, axis=1)
        ic = np.concatenate(input_cat, axis=1)
        oc = np.concatenate(output_cat, axis=1)
        t_ax = np.arange(nc.shape[1]) * TIME_BIN_MS / 1000
        
        labels = ['Neural\n(mean 50 neu)', 'Time to\nsound (s)', 'Day of\ntraining',
                  'Time since\nstart (s)', 'Licking', 'Position\nbin', 'Speed\nbin']
        colors = ['k', 'b', 'orange', 'g', 'r', 'm', 'c']
        data_arrays = [
            np.mean(nc[:min(50, nc.shape[0])], axis=0),
            ic[0], ic[1], ic[2], oc[1].astype(float), oc[2].astype(float), oc[3].astype(float)
        ]
        
        for idx, (ax, label, color, arr) in enumerate(zip(axes, labels, colors, data_arrays)):
            ax.plot(t_ax, arr, color, lw=0.5)
            ax.set_ylabel(label)
            for b in boundaries:
                ax.axvline(b*TIME_BIN_MS/1000, color='r', ls='--', alpha=0.3)
            if idx == 1:
                ax.axhline(0, color='gray', ls=':')
            if idx == 4:
                ax.set_ylim(-0.1, 1.1)
            if idx in [5, 6]:
                ax.set_yticks([0,1,2,3])
        
        axes[-1].set_xlabel('Time (s)')
        plt.tight_layout()
        plt.savefig(f'processing_{sid}.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved processing_{sid}.png")

if __name__ == '__main__':
    main()
