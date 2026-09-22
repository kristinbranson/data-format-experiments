#!/usr/bin/env python3
"""Convert Zhong et al. 2025 data to decoder-compatible format.

Usage:
    python -u convert_data.py <outpicklefile> [--sample] [--show-processing]
"""

import numpy as np
import os
import sys
import pickle
import time
import argparse
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Constants
FS = 3.17  # Frame rate in Hz
TIME_BIN_MS = 1000.0 / FS  # ~315.5 ms
DATA_ROOT = '/app/data'
CORRIDOR_LENGTH_DM = 60  # decimeters (6 meters)
TEXTURE_LENGTH_DM = 40  # decimeters (4 meters)
GRAY_LENGTH_DM = 20  # decimeters (2 meters)
N_POS_BINS = 4  # 4 bins of 1 meter each
N_SPEED_BINS = 4  # 4 quartile bins
MAX_NEURONS = 5000  # Subsample neurons for manageable file size; decoder uses PCA to 100 components
POS_BIN_SIZE_DM = TEXTURE_LENGTH_DM / N_POS_BINS  # 10 decimeters = 1 meter


def load_exp_info():
    """Load experiment information for all sessions."""
    info = np.load(os.path.join(DATA_ROOT, 'beh', 'Imaging_Exp_info.npy'), 
                   allow_pickle=True).item()
    return info


def load_spk(db):
    """Load neural data for a session. Matches reference code utils.load_spk."""
    fn = '%s_%s_%s_neural_data.npy' % (db['mname'], db['datexp'], db['blk'])
    spk_path = os.path.join(DATA_ROOT, 'spk', fn)
    spk = np.concatenate([nspk for nspk in 
                          np.load(spk_path, allow_pickle=True).item()['spks']], 0)
    return spk


def load_retino(db):
    """Load retinotopy data. Matches reference code utils.load_retino."""
    dtrans = np.load(os.path.join(DATA_ROOT, 'retinotopy', 
                     '%s_%s_trans.npz' % (db['mname'], db['datexp'])), 
                     allow_pickle=True)
    return dtrans


def neu_area_ID(iarea):
    """Map area codes to brain region names. Matches reference code utils.neu_area_ID."""
    area_map = {
        8: 'V1',
        0: 'mHV', 1: 'mHV', 2: 'mHV', 9: 'mHV',
        5: 'lHV', 6: 'lHV',
        3: 'aHV', 4: 'aHV',
    }
    regions = []
    for a in iarea:
        a_int = int(a)
        regions.append(area_map.get(a_int, 'unassigned'))
    return regions


def get_brain_region_idx(iarea, brain_regions):
    """Convert iarea to indices into brain_regions list."""
    region_names = neu_area_ID(iarea)
    region_to_idx = {r: i for i, r in enumerate(brain_regions)}
    return np.array([region_to_idx[r] for r in region_names], dtype=np.int64)


def load_beh(exp_type):
    """Load behavior data for an experiment type."""
    beh = np.load(os.path.join(DATA_ROOT, 'beh', 'Beh_' + exp_type + '.npy'),
                  allow_pickle=True).item()
    return beh


def get_session_key(db):
    """Get behavior data key for a session."""
    if 'stimtype' in db:
        return '%s_%s_%s_%s' % (db['mname'], db['datexp'], db['blk'], db['stimtype'])
    else:
        return '%s_%s_%s' % (db['mname'], db['datexp'], db['blk'])


def get_day_of_training(db):
    """Get day of training from session metadata."""
    if 'sess#' in db:
        return float(db['sess#'])
    elif 'days' in db:
        return float(db['days'])
    else:
        return 0.0


def compute_global_speed_quartiles(info):
    """Compute running speed quartiles across all sessions."""
    print("Computing global speed quartiles...")
    all_speeds = []
    
    seen_keys = set()
    beh_cache = {}
    
    for exp_type, sessions in info.items():
        for db in sessions:
            key = f"{db['mname']}_{db['datexp']}_{db['blk']}"
            if key in seen_keys:
                continue
            seen_keys.add(key)
            
            if exp_type not in beh_cache:
                beh_cache[exp_type] = load_beh(exp_type)
            
            beh_key = get_session_key(db)
            if beh_key not in beh_cache[exp_type]:
                continue
            
            beh = beh_cache[exp_type][beh_key]
            speeds = np.array(beh['ft_RunSpeed'])
            all_speeds.append(speeds)
    
    all_speeds = np.concatenate(all_speeds)
    quartiles = np.percentile(all_speeds, [25, 50, 75])
    print(f"  Speed quartile boundaries: {quartiles}")
    print(f"  Speed range: [{all_speeds.min():.2f}, {all_speeds.max():.2f}]")
    print(f"  Speed mean: {all_speeds.mean():.2f}")
    
    del beh_cache
    return quartiles


def discretize_speed(speed, quartiles):
    """Discretize speed into 4 quartile bins. Returns integer values 0-3."""
    bins = np.digitize(speed, quartiles)  # Returns 0,1,2,3
    return bins.astype(np.int64)


def discretize_position(position_dm):
    """Discretize position into 4 bins of 1m (10 dm) each. Returns integer values 0-3.
    Bin 0: 0-10 dm (0-1m)
    Bin 1: 10-20 dm (1-2m)
    Bin 2: 20-30 dm (2-3m)
    Bin 3: 30+ dm (3m+, including gray space)
    """
    bins = np.clip(position_dm // POS_BIN_SIZE_DM, 0, N_POS_BINS - 1).astype(np.int64)
    return bins


def process_session(db, beh, spk, retino, stim_list, brain_regions, speed_quartiles, 
                    exp_type, show_processing=False, session_idx=0):
    """Process a single session and return trial-level data."""
    nfr = spk.shape[1]
    nneu = spk.shape[0]
    ntrials = beh['ntrials']
    
    # Get frame-time aligned variables (truncate to neural data length)
    ft_pos = np.array(beh['ft_Pos'])[:nfr]
    ft_run_speed = np.array(beh['ft_RunSpeed'])[:nfr]
    
    # Get trial boundaries
    start_fr = np.array(beh['StartFr'])
    end_fr = np.array(beh['EndFr'])
    sound_fr = np.array(beh['SoundFr'])
    
    # Convert to integer frame indices
    start_fr_int = np.floor(start_fr).astype(int)
    end_fr_int = np.floor(end_fr).astype(int)
    
    # Clip to valid range
    start_fr_int = np.clip(start_fr_int, 0, nfr - 1)
    end_fr_int = np.clip(end_fr_int, 0, nfr)
    
    # Get per-trial variables
    is_rew = np.array(beh['isRew']).astype(float)
    wall_name = np.array(beh['WallName'])
    
    # Map stimulus names to indices
    stim_to_idx = {s: i for i, s in enumerate(stim_list)}
    
    # Get lick data
    lick_fr = np.array(beh['LickFr']).astype(float) if len(beh['LickFr']) > 0 else np.array([])
    lick_trind = np.array(beh['LickTrind']).astype(int) if len(beh['LickTrind']) > 0 else np.array([], dtype=int)
    
    # Get day of training
    day_of_training = get_day_of_training(db)
    
    # Get brain region indices
    iarea = retino['iarea']
    brain_region_idx = get_brain_region_idx(iarea, brain_regions)
    
    # Process each trial
    neural_trials = []
    input_trials = []
    output_trials = []
    
    skipped_trials = 0
    
    for tr in range(ntrials):
        s_fr = start_fr_int[tr]
        e_fr = end_fr_int[tr]
        n_trial_frames = e_fr - s_fr
        
        if n_trial_frames < 2:
            skipped_trials += 1
            continue
        
        # Neural data: (n_neurons, n_timepoints) - store as float16
        neural = spk[:, s_fr:e_fr].astype(np.float16)
        
        # --- INPUTS ---
        frame_indices = np.arange(s_fr, e_fr, dtype=np.float64)
        
        # Input 0: Time to sound cue (seconds) - negative before, positive after
        time_to_cue = (frame_indices - sound_fr[tr]) / FS
        
        # Input 1: Day of training (per-trial)
        day_input = np.full(n_trial_frames, day_of_training, dtype=np.float64)
        
        # Input 2: Time since trial start (seconds)
        time_since_start = (frame_indices - start_fr[tr]) / FS
        
        # Input 3: Reward availability
        reward_input = np.full(n_trial_frames, is_rew[tr], dtype=np.float64)
        
        # Stack inputs: (4, n_timepoints)
        inputs = np.stack([time_to_cue, day_input, time_since_start, reward_input], 
                          axis=0).astype(np.float32)
        
        # --- OUTPUTS (all integer-valued) ---
        # Output 0: Visual stimulus category (per-trial)
        stim_name = str(wall_name[tr])
        stim_idx = stim_to_idx.get(stim_name, 0)
        stim_output = np.full(n_trial_frames, stim_idx, dtype=np.int64)
        
        # Output 1: Licking (binary, time-varying)
        lick_binary = np.zeros(n_trial_frames, dtype=np.int64)
        if len(lick_trind) > 0:
            trial_lick_mask = lick_trind == tr
            if np.any(trial_lick_mask):
                trial_lick_frames = np.round(lick_fr[trial_lick_mask]).astype(int)
                trial_lick_frames = trial_lick_frames - s_fr
                valid = (trial_lick_frames >= 0) & (trial_lick_frames < n_trial_frames)
                if np.any(valid):
                    lick_binary[trial_lick_frames[valid]] = 1
        
        # Output 2: Position (4 bins)
        pos = ft_pos[s_fr:e_fr]
        pos_binned = discretize_position(pos)
        
        # Output 3: Running speed (4 quartile bins)
        speed = ft_run_speed[s_fr:e_fr]
        speed_binned = discretize_speed(speed, speed_quartiles)
        
        # Stack outputs: (4, n_timepoints) as int64 (categorical values)
        outputs = np.stack([stim_output, lick_binary, pos_binned, speed_binned],
                           axis=0).astype(np.int64)
        
        neural_trials.append(neural)
        input_trials.append(inputs)
        output_trials.append(outputs)
    
    if skipped_trials > 0:
        print(f"    Skipped {skipped_trials} trials with < 2 frames")
    
    # Visualization for --show-processing
    if show_processing and len(neural_trials) > 0:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            session_id = f"{db['mname']}_{db['datexp']}_{db['blk']}"
            fig, axes = plt.subplots(6, 2, figsize=(20, 24))
            fig.suptitle(f'Session: {session_id} ({exp_type})', fontsize=14)
            
            for col, tr_idx in enumerate([0, min(5, len(neural_trials)-1)]):
                neural_tr = neural_trials[tr_idx].astype(np.float32)
                input_tr = input_trials[tr_idx]
                output_tr = output_trials[tr_idx]
                n_t = neural_tr.shape[1]
                t = np.arange(n_t) / FS
                
                ax = axes[0, col]
                n_show = min(50, neural_tr.shape[0])
                ax.imshow(neural_tr[:n_show], aspect='auto', cmap='gray_r',
                         extent=[0, t[-1], n_show, 0])
                ax.set_title(f'Trial {tr_idx}: Neural (first {n_show} neurons)')
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Neuron')
                
                ax = axes[1, col]
                ax.plot(t, input_tr[0], label='Time to cue')
                ax.plot(t, input_tr[2], label='Time since start')
                ax.axhline(0, color='k', linestyle='--', alpha=0.3)
                ax.set_title('Inputs: Time variables')
                ax.set_xlabel('Time (s)')
                ax.legend(fontsize=8)
                
                ax = axes[2, col]
                ax.plot(t, input_tr[3], label=f'Reward={input_tr[3,0]:.0f}')
                ax.set_title(f'Inputs: Reward={input_tr[3,0]:.0f}, Day={input_tr[1,0]:.0f}')
                ax.set_xlabel('Time (s)')
                ax.set_ylim(-0.1, 1.1)
                
                ax = axes[3, col]
                ax.plot(t, output_tr[0], label=f'Stim={stim_list[int(output_tr[0,0])]}')
                ax.set_title(f'Output: Stimulus = {stim_list[int(output_tr[0,0])]}')
                ax.set_xlabel('Time (s)')
                
                ax = axes[4, col]
                ax.plot(t, output_tr[1], label='Licking')
                ax.set_title('Output: Licking')
                ax.set_xlabel('Time (s)')
                ax.set_ylim(-0.1, 1.1)
                
                ax = axes[5, col]
                ax.plot(t, output_tr[2], label='Position bin', alpha=0.7)
                ax.plot(t, output_tr[3], label='Speed bin', alpha=0.7)
                ax.set_title('Output: Position & Speed bins')
                ax.set_xlabel('Time (s)')
                ax.legend(fontsize=8)
            
            plt.tight_layout()
            plt.savefig(f'/app/processing_{session_id}.png', dpi=100)
            plt.close()
            print(f"    Saved processing plot: /app/processing_{session_id}.png")
        except Exception as e:
            print(f"    Warning: Could not create processing plot: {e}")
    
    return neural_trials, input_trials, output_trials, brain_region_idx


def main():
    parser = argparse.ArgumentParser(description='Convert Zhong et al. 2025 data')
    parser.add_argument('outfile', type=str, help='Output pickle file path')
    parser.add_argument('--sample', action='store_true', help='Process only 2 sessions')
    parser.add_argument('--full', action='store_true', help='Process all sessions (default)')
    parser.add_argument('--show-processing', action='store_true', help='Save processing plots')
    args = parser.parse_args()
    
    total_start = time.time()
    
    print("Loading experiment info...")
    info = load_exp_info()
    
    # Build list of all unique sessions
    all_sessions = []
    seen_keys = set()
    
    for exp_type, sessions in info.items():
        for db in sessions:
            key = f"{db['mname']}_{db['datexp']}_{db['blk']}"
            if key not in seen_keys:
                seen_keys.add(key)
                all_sessions.append((exp_type, db, key))
    
    print(f"Total unique sessions: {len(all_sessions)}")
    
    if args.sample:
        # Select 2 sessions: one supervised with licking, one unsupervised
        sample_sessions = []
        found_sup = False
        found_unsup = False
        for exp_type, db, key in all_sessions:
            if not found_sup and 'sup_train1_after' in exp_type and 'unsup' not in exp_type:
                sample_sessions.append((exp_type, db, key))
                found_sup = True
            elif not found_unsup and 'unsup_test1' in exp_type:
                sample_sessions.append((exp_type, db, key))
                found_unsup = True
            if found_sup and found_unsup:
                break
        if len(sample_sessions) < 2:
            sample_sessions = all_sessions[:2]
        all_sessions = sample_sessions
        print(f"Sample mode: processing {len(all_sessions)} sessions")
        for _, _, k in all_sessions:
            print(f"  {k}")
    
    # Get all unique stimuli
    print("Getting unique stimuli...")
    all_stim = set()
    beh_cache = {}
    for exp_type, sessions in info.items():
        if exp_type not in beh_cache:
            beh_cache[exp_type] = load_beh(exp_type)
        for db in sessions:
            beh_key = get_session_key(db)
            if beh_key in beh_cache[exp_type]:
                for w in beh_cache[exp_type][beh_key]['UniqWalls']:
                    all_stim.add(str(w))
    stim_list = sorted(all_stim)
    print(f"  Stimuli: {stim_list}")
    
    brain_regions = ['V1', 'mHV', 'lHV', 'aHV', 'unassigned']
    
    # Compute global speed quartiles
    speed_quartiles = compute_global_speed_quartiles(info)
    
    del beh_cache
    
    # Process all sessions
    neural_all = []
    input_all = []
    output_all = []
    brain_region_idx_all = []
    subject_idx_list = []
    subjects_set = {}
    
    beh_cache = {}
    
    for sess_idx, (exp_type, db, key) in enumerate(all_sessions):
        sess_start = time.time()
        print(f"\nProcessing session {sess_idx+1}/{len(all_sessions)}: {key} ({exp_type})")
        
        if exp_type not in beh_cache:
            print(f"  Loading behavior data for {exp_type}...")
            beh_cache[exp_type] = load_beh(exp_type)
        
        beh_key = get_session_key(db)
        if beh_key not in beh_cache[exp_type]:
            print(f"  WARNING: Session key {beh_key} not found in behavior data")
            continue
        beh = beh_cache[exp_type][beh_key]
        
        print(f"  Loading neural data...")
        t0 = time.time()
        spk = load_spk(db)
        print(f"  Neural data loaded: {spk.shape} in {time.time()-t0:.1f}s")
        
        retino = load_retino(db)
        
        n_neu_spk = spk.shape[0]
        n_neu_ret = len(retino['iarea'])
        if n_neu_spk != n_neu_ret:
            print(f"  WARNING: Neuron count mismatch: spk={n_neu_spk}, retino={n_neu_ret}")
            min_n = min(n_neu_spk, n_neu_ret)
            spk = spk[:min_n]
        
        # Subsample neurons if too many (decoder uses PCA to 100 components)
        iarea_arr = np.array(retino['iarea'])
        if n_neu_spk != n_neu_ret:
            iarea_arr = iarea_arr[:min_n]
        n_total = spk.shape[0]
        if n_total > MAX_NEURONS:
            rng = np.random.RandomState(42)  # Fixed seed for reproducibility
            neuron_idx = np.sort(rng.choice(n_total, MAX_NEURONS, replace=False))
            spk = spk[neuron_idx]
            iarea_arr = iarea_arr[neuron_idx]
            print(f"  Subsampled neurons: {n_total} -> {MAX_NEURONS}")
        
        # Create a modified retino dict with subsampled iarea
        retino_sub = {'iarea': iarea_arr}
        
        show = args.show_processing and sess_idx < 2
        neural_trials, input_trials, output_trials, brain_reg_idx = process_session(
            db, beh, spk, retino_sub, stim_list, brain_regions, speed_quartiles,
            exp_type, show_processing=show, session_idx=sess_idx
        )
        
        if len(neural_trials) < 2:
            print(f"  WARNING: Session has fewer than 2 valid trials, skipping")
            continue
        
        neural_all.append(neural_trials)
        input_all.append(input_trials)
        output_all.append(output_trials)
        brain_region_idx_all.append(brain_reg_idx)
        
        mname = db['mname']
        if mname not in subjects_set:
            subjects_set[mname] = len(subjects_set)
        subject_idx_list.append(subjects_set[mname])
        
        sess_time = time.time() - sess_start
        print(f"  Session done: {len(neural_trials)} trials, {spk.shape[0]} neurons, {sess_time:.1f}s")
        
        del spk
    
    subjects = list(subjects_set.keys())
    subject_idx = np.array(subject_idx_list, dtype=np.int64)
    
    input_names = ['time_to_sound_cue', 'day_of_training', 'time_since_trial_start', 'reward_availability']
    output_names = ['visual_stimulus', 'licking', 'position_bin', 'running_speed_bin']
    
    output_values = [
        stim_list,
        ['not_licking', 'licking'],
        ['0-1m', '1-2m', '2-3m', '3-4m'],
        ['Q1_slowest', 'Q2', 'Q3', 'Q4_fastest'],
    ]
    
    data = {
        'neural': neural_all,
        'input': input_all,
        'output': output_all,
        'subjects': subjects,
        'subject_idx': subject_idx,
        'brain_regions': brain_regions,
        'brain_region_idx': brain_region_idx_all,
        'input_names': input_names,
        'output_names': output_names,
        'output_values': output_values,
        'metadata': {
            'task_description': 'Visual discrimination task in virtual reality corridors with naturalistic textures. Mice discriminate between visual texture patterns (leaf vs circle) with reward association.',
            'time_bin_size': TIME_BIN_MS,
            'temporal_alignment_event': 'Trial start (corridor entry)',
            'off_start': 0.0,
            'off_end': None,
            'frame_rate_hz': FS,
            'corridor_length_m': CORRIDOR_LENGTH_DM / 10,
            'texture_length_m': TEXTURE_LENGTH_DM / 10,
            'gray_space_length_m': GRAY_LENGTH_DM / 10,
            'speed_quartile_boundaries': speed_quartiles.tolist(),
            'stimulus_list': stim_list,
            'n_sessions': len(neural_all),
        }
    }
    
    total_trials = sum(len(s) for s in neural_all)
    total_neurons = sum(br.shape[0] for br in brain_region_idx_all)
    neurons_per_session = [br.shape[0] for br in brain_region_idx_all]
    trials_per_session = [len(s) for s in neural_all]
    
    print(f"\n{'='*60}")
    print(f"CONVERSION SUMMARY")
    print(f"{'='*60}")
    print(f"Sessions: {len(neural_all)}")
    print(f"Subjects: {len(subjects)}")
    print(f"Total trials: {total_trials}")
    print(f"Trials/session: mean={np.mean(trials_per_session):.1f}, min={np.min(trials_per_session)}, max={np.max(trials_per_session)}")
    print(f"Neurons/session: mean={np.mean(neurons_per_session):.1f}, min={np.min(neurons_per_session)}, max={np.max(neurons_per_session)}")
    print(f"Total neurons: {total_neurons}")
    print(f"Brain regions: {brain_regions}")
    print(f"Stimuli: {stim_list}")
    print(f"Input names: {input_names}")
    print(f"Output names: {output_names}")
    
    print(f"\nSaving to {args.outfile}...")
    with open(args.outfile, 'wb') as f:
        pickle.dump(data, f, protocol=4)
    
    file_size = os.path.getsize(args.outfile)
    print(f"Saved: {file_size / 1e9:.2f} GB")
    
    total_time = time.time() - total_start
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Time per session: {total_time/len(neural_all):.1f}s")


if __name__ == '__main__':
    main()
