#!/usr/bin/env python3
"""Convert Zhong et al. 2025 data to decoder-compatible format."""
import numpy as np
import os
import sys
import pickle
import time
import argparse
from datetime import datetime
from collections import defaultdict, Counter
import gc
import tempfile

DATA_ROOT = 'data'
FRAME_RATE = 3.17  # Hz
TIME_BIN_MS = 1000.0 / FRAME_RATE
DAYS_TO_SEC = 24 * 3600

def neu_area_ID(iarea):
    idx = {}
    idx['V1'] = iarea == 8
    idx['mHV'] = (iarea == 0) | (iarea == 1) | (iarea == 2) | (iarea == 9)
    idx['lHV'] = (iarea == 5) | (iarea == 6)
    idx['aHV'] = (iarea == 3) | (iarea == 4)
    return idx

def load_spk(mname, datexp, blk, root=''):
    fn = f'{mname}_{datexp}_{blk}_neural_data.npy'
    spk = np.concatenate(
        [nspk for nspk in np.load(os.path.join(root, fn), allow_pickle=True).item()['spks']], 0
    )
    return spk

def load_retino(mname, datexp, root=''):
    dtrans = np.load(os.path.join(root, f'{mname}_{datexp}_trans.npz'), allow_pickle=True)
    return dtrans['iarea']

def get_all_sessions():
    info = np.load(os.path.join(DATA_ROOT, 'beh', 'Imaging_Exp_info.npy'), allow_pickle=True).item()
    session_map = {}
    for exp_type in info.keys():
        for entry in info[exp_type]:
            session_id = f"{entry['mname']}_{entry['datexp']}_{entry['blk']}"
            beh_key = session_id if 'stimtype' not in entry else f"{session_id}_{entry['stimtype']}"
            if session_id not in session_map:
                session_map[session_id] = (exp_type, entry, beh_key)
    sessions = []
    for session_id in sorted(session_map.keys()):
        exp_type, entry, beh_key = session_map[session_id]
        sessions.append({'session_id': session_id, 'mname': entry['mname'],
                         'datexp': entry['datexp'], 'blk': entry['blk'],
                         'exp_type': exp_type, 'beh_key': beh_key, 'entry': entry})
    return sessions

def compute_day_of_training(sessions):
    mouse_sessions = defaultdict(list)
    for i, s in enumerate(sessions):
        mouse_sessions[s['mname']].append((i, s))
    day_of_training = np.zeros(len(sessions))
    for mname, msessions in mouse_sessions.items():
        dates = [(idx, datetime(int(s['datexp'].split('_')[0]), int(s['datexp'].split('_')[1]), int(s['datexp'].split('_')[2]))) for idx, s in msessions]
        dates.sort(key=lambda x: x[1])
        first_date = dates[0][1]
        for idx, date in dates:
            day_of_training[idx] = (date - first_date).days
    return day_of_training

def build_lick_per_frame(n_frames, lick_fr):
    lick_binary = np.zeros(n_frames, dtype=np.int32)
    if len(lick_fr) == 0:
        return lick_binary
    idx = np.round(lick_fr).astype(int)
    idx = idx[(idx >= 0) & (idx < n_frames)]
    lick_binary[idx] = 1
    return lick_binary

def compute_global_speed_quartiles(sessions):
    print('Computing global running speed quartiles...')
    all_speeds = []
    beh_cache = {}
    for i, session in enumerate(sessions):
        exp_type, beh_key = session['exp_type'], session['beh_key']
        if exp_type not in beh_cache:
            beh_cache[exp_type] = np.load(os.path.join(DATA_ROOT, 'beh', f'Beh_{exp_type}.npy'), allow_pickle=True).item()
        beh = beh_cache[exp_type][beh_key]
        vr_move = beh['ft_move'] > 0
        speeds = beh['ft_RunSpeed'][vr_move]
        if len(speeds) > 5000:
            speeds = np.random.RandomState(42+i).choice(speeds, 5000, replace=False)
        all_speeds.append(speeds)
    del beh_cache
    all_speeds = np.concatenate(all_speeds)
    quartiles = np.percentile(all_speeds, [25, 50, 75])
    print(f'  Speed quartiles: {quartiles}')
    return quartiles

def get_all_stim_names(sessions):
    all_stim = set()
    beh_cache = {}
    for session in sessions:
        exp_type, beh_key = session['exp_type'], session['beh_key']
        if exp_type not in beh_cache:
            beh_cache[exp_type] = np.load(os.path.join(DATA_ROOT, 'beh', f'Beh_{exp_type}.npy'), allow_pickle=True).item()
        for wn in beh_cache[exp_type][beh_key]['UniqWalls']:
            all_stim.add(str(wn))
    del beh_cache
    return sorted(all_stim)

def process_session(session, day_val, speed_quartiles, stim_to_idx):
    session_id = session['session_id']
    mname, datexp, blk = session['mname'], session['datexp'], session['blk']
    exp_type, beh_key = session['exp_type'], session['beh_key']
    
    t0 = time.time()
    
    # Load neural data
    spk = load_spk(mname, datexp, blk, root=os.path.join(DATA_ROOT, 'spk'))
    n_neurons_total, n_frames_neural = spk.shape
    
    # Load retinotopy and filter
    iarea = load_retino(mname, datexp, root=os.path.join(DATA_ROOT, 'retinotopy'))
    neuron_mask = (iarea != -1) & (iarea != 7)
    spk = spk[neuron_mask]
    iarea_filtered = iarea[neuron_mask]
    n_neurons = spk.shape[0]
    t_load = time.time()
    print(f'  Neurons: {n_neurons_total} -> {n_neurons} ({t_load-t0:.1f}s load)')
    
    # Load behavior
    beh_all = np.load(os.path.join(DATA_ROOT, 'beh', f'Beh_{exp_type}.npy'), allow_pickle=True).item()
    beh = beh_all[beh_key]
    del beh_all
    
    ntrials = beh['ntrials']
    ft = beh['ft'][:n_frames_neural]
    ft_trInd = beh['ft_trInd'][:n_frames_neural]
    ft_Pos = beh['ft_Pos'][:n_frames_neural]
    ft_move = beh['ft_move'][:n_frames_neural]
    ft_RunSpeed = beh['ft_RunSpeed'][:n_frames_neural]
    lick_binary = build_lick_per_frame(n_frames_neural, beh.get('LickFr', np.array([])))
    sound_fr = beh['SoundFr']
    is_rew = beh['isRew']
    wall_name = beh['WallName']
    vr_move = ft_move > 0
    
    neural_trials, input_trials, output_trials = [], [], []
    
    for trial_idx in range(ntrials):
        valid_mask = (ft_trInd == trial_idx) & vr_move
        valid_frame_indices = np.where(valid_mask)[0]
        if len(valid_frame_indices) < 2:
            continue
        n_t = len(valid_frame_indices)
        
        neural_trial = spk[:, valid_frame_indices].astype(np.float32)
        ft_trial = ft[valid_frame_indices]
        time_since_start = ((ft_trial - ft_trial[0]) * DAYS_TO_SEC).astype(np.float32)
        
        s_fr = sound_fr[trial_idx]
        if np.isnan(s_fr):
            time_to_sound = np.zeros(n_t, dtype=np.float32)
        else:
            s_fr_int = int(np.floor(s_fr))
            s_fr_frac = s_fr - s_fr_int
            if 0 <= s_fr_int < len(ft) - 1:
                sound_time = ft[s_fr_int] + s_fr_frac * (ft[s_fr_int + 1] - ft[s_fr_int])
            elif s_fr_int >= len(ft) - 1:
                sound_time = ft[-1]
            else:
                sound_time = ft[0]
            time_to_sound = ((ft_trial - sound_time) * DAYS_TO_SEC).astype(np.float32)
        
        input_trial = np.zeros((4, n_t), dtype=np.float32)
        input_trial[0] = time_to_sound
        input_trial[1] = np.float32(day_val)
        input_trial[2] = time_since_start
        input_trial[3] = np.float32(1.0 if is_rew[trial_idx] else 0.0)
        
        stim_idx = stim_to_idx[str(wall_name[trial_idx])]
        lick_trial = lick_binary[valid_frame_indices]
        pos_bins = np.clip(np.floor(ft_Pos[valid_frame_indices] / 15.0), 0, 3).astype(np.int32)
        speed_bins = np.clip(np.digitize(ft_RunSpeed[valid_frame_indices], speed_quartiles), 0, 3).astype(np.int32)
        
        output_trial = np.zeros((4, n_t), dtype=np.int32)
        output_trial[0] = stim_idx
        output_trial[1] = lick_trial
        output_trial[2] = pos_bins
        output_trial[3] = speed_bins
        
        neural_trials.append(neural_trial)
        input_trials.append(input_trial)
        output_trials.append(output_trial)
    
    del spk
    gc.collect()
    
    valid_count = len(neural_trials)
    t1 = time.time()
    print(f'  Trials: {valid_count}/{ntrials} ({t1-t_load:.1f}s process)')
    
    if valid_count < 2:
        return None
    
    area_map = neu_area_ID(iarea_filtered)
    region_idx = np.full(n_neurons, 0, dtype=np.int32)
    for r_idx, r_name in enumerate(['V1', 'mHV', 'lHV', 'aHV']):
        region_idx[area_map[r_name]] = r_idx
    
    return {'neural': neural_trials, 'input': input_trials, 'output': output_trials,
            'n_neurons': n_neurons, 'region_idx': region_idx,
            'session_id': session_id, 'mname': mname, 'exp_type': exp_type}

def plot_processing(result, session, speed_quartiles, stim_to_idx, idx):
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(4, 3, figsize=(18, 16))
        fig.suptitle(f'{session["session_id"]} ({session["exp_type"]})', fontsize=14)
        n_trials = len(result['neural'])
        ax = axes[0,0]; t0 = result['neural'][0]; n_show = min(50, t0.shape[0])
        ax.imshow(t0[:n_show], aspect='auto', cmap='hot'); ax.set_title(f'Neural ({n_show}n)')
        ax = axes[0,1]; inp = result['input'][0]
        for k, l in enumerate(['tts','day','tss','rew']): ax.plot(inp[k], label=l)
        ax.legend(fontsize=6); ax.set_title('Inputs')
        ax = axes[0,2]; out = result['output'][0]
        for k, l in enumerate(['stim','lick','pos','spd']): ax.plot(out[k]+k*5, label=l)
        ax.legend(fontsize=6); ax.set_title('Outputs')
        ax = axes[1,0]; ax.hist([t.shape[1] for t in result['neural']], bins=30); ax.set_title(f'Trial lens (n={n_trials})')
        ax = axes[1,1]; ax.bar(['V1','mHV','lHV','aHV'], [np.sum(result['region_idx']==i) for i in range(4)]); ax.set_title('Brain regions')
        ax = axes[1,2]; ax.hist(np.concatenate([result['output'][t][2] for t in range(n_trials)]), bins=4, range=(-0.5,3.5)); ax.set_title('Position bins')
        ax = axes[2,0]; ax.hist(np.concatenate([result['output'][t][3] for t in range(n_trials)]), bins=4, range=(-0.5,3.5)); ax.set_title('Speed bins')
        ax = axes[2,1]; ax.hist(np.concatenate([result['output'][t][1] for t in range(n_trials)]), bins=2); ax.set_title('Licking')
        ax = axes[2,2]; ax.hist(np.concatenate([result['input'][t][0] for t in range(n_trials)]), bins=50); ax.set_title('Time to sound')
        ax = axes[3,0]; ax.plot([result['neural'][t].mean() for t in range(n_trials)]); ax.set_title('Mean neural')
        ax = axes[3,1]; ax.plot([result['output'][t][1].mean() for t in range(n_trials)]); ax.set_title('Lick rate')
        ax = axes[3,2]; ax.hist(np.concatenate([result['input'][t][2] for t in range(n_trials)]), bins=50); ax.set_title('Time since start')
        plt.tight_layout(); plt.savefig(f'processing_{session["session_id"]}.png', dpi=150); plt.close()
        print(f'  Saved plot: processing_{session["session_id"]}.png')
    except Exception as e:
        print(f'  Plot error: {e}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output'); parser.add_argument('--full', action='store_true', default=True)
    parser.add_argument('--sample', action='store_true'); parser.add_argument('--show-processing', action='store_true')
    args = parser.parse_args()
    if args.sample: args.full = False
    
    total_start = time.time()
    print('Getting session list...')
    all_sessions = get_all_sessions()
    print(f'Total unique sessions: {len(all_sessions)}')
    
    if args.sample:
        sup_s, unsup_s = None, None
        for s in all_sessions:
            et = s['exp_type']
            if sup_s is None and et.startswith('sup_'): sup_s = s
            elif unsup_s is None and (et.startswith('unsup_') or et.startswith('naive_')): unsup_s = s
            if sup_s and unsup_s: break
        sessions = [s for s in [sup_s, unsup_s] if s]
        print(f'Sample: {[s["session_id"] for s in sessions]}')
    else:
        sessions = all_sessions
    
    day_of_training = compute_day_of_training(all_sessions)
    day_map = {s['session_id']: day_of_training[i] for i, s in enumerate(all_sessions)}
    
    print('Getting stimulus names...')
    all_stim_names = get_all_stim_names(all_sessions)
    print(f'Stimuli: {all_stim_names}')
    stim_to_idx = {name: i for i, name in enumerate(all_stim_names)}
    
    speed_quartiles = compute_global_speed_quartiles(all_sessions)
    
    print(f'\nProcessing {len(sessions)} sessions...')
    
    # Process sessions and save to temp files to avoid memory accumulation
    os.makedirs('cache', exist_ok=True)
    temp_files = []
    session_meta = []
    brain_regions = ['V1', 'mHV', 'lHV', 'aHV']
    all_subjects = []
    
    for i, session in enumerate(sessions):
        print(f'\n[{i+1}/{len(sessions)}] {session["session_id"]} ({session["exp_type"]})')
        result = process_session(session, day_map[session['session_id']], speed_quartiles, stim_to_idx)
        if result is None: continue
        
        mname = result['mname']
        if mname not in all_subjects: all_subjects.append(mname)
        subject_idx = all_subjects.index(mname)
        
        # Save session data to temp file
        temp_fn = f'cache/session_{i:03d}.pkl'
        with open(temp_fn, 'wb') as f:
            pickle.dump({
                'neural': result['neural'], 'input': result['input'], 'output': result['output'],
                'region_idx': result['region_idx'], 'subject_idx': subject_idx,
                'session_info': {'session_id': result['session_id'], 'exp_type': result['exp_type'],
                                 'n_neurons': result['n_neurons'], 'n_trials': len(result['neural']), 'mname': mname}
            }, f, protocol=4)
        temp_files.append(temp_fn)
        
        if args.show_processing and i < 2:
            plot_processing(result, session, speed_quartiles, stim_to_idx, i)
        
        del result
        gc.collect()
    
    # Combine all temp files
    print(f'\nCombining {len(temp_files)} session files...')
    all_neural, all_input, all_output = [], [], []
    all_subject_idx, all_brain_region_idx = [], []
    session_infos = []
    
    for tf in temp_files:
        with open(tf, 'rb') as f:
            sd = pickle.load(f)
        all_neural.append(sd['neural'])
        all_input.append(sd['input'])
        all_output.append(sd['output'])
        all_subject_idx.append(sd['subject_idx'])
        all_brain_region_idx.append(sd['region_idx'])
        session_infos.append(sd['session_info'])
        del sd
    
    data = {
        'neural': all_neural, 'input': all_input, 'output': all_output,
        'subjects': all_subjects, 'subject_idx': np.array(all_subject_idx),
        'brain_regions': brain_regions, 'brain_region_idx': all_brain_region_idx,
        'input_names': ['time_to_sound_cue', 'day_of_training', 'time_since_trial_start', 'reward_availability'],
        'output_names': ['visual_stimulus_category', 'licking', 'position_bin', 'running_speed_bin'],
        'output_values': [all_stim_names, ['no_lick', 'lick'], ['0-1.5m', '1.5-3m', '3-4.5m', '4.5-6m'], ['Q1', 'Q2', 'Q3', 'Q4']],
        'metadata': {
            'task_description': 'Visual discrimination in VR corridors with naturalistic textures.',
            'time_bin_size': TIME_BIN_MS, 'temporal_alignment_event': 'Trial start (corridor entry)',
            'off_start': 0.0, 'off_end': None,
            'frame_rate_hz': FRAME_RATE, 'corridor_length_m': 6.0, 'texture_length_m': 4.0,
            'speed_quartile_boundaries': speed_quartiles.tolist(),
            'session_info': session_infos, 'stim_to_idx': stim_to_idx,
            'neural_data_type': 'Suite2p deconvolved fluorescence traces',
            'deconvolution_decay_s': 0.75,
            'neuron_filter': 'Excluded neurons outside visual cortex (iarea==-1 or iarea==7)',
            'frame_filter': 'Only VR-moving frames (ft_move > 0)',
            'n_sessions': len(all_neural), 'n_subjects': len(all_subjects),
        }
    }
    
    print(f'Saving to {args.output}...')
    with open(args.output, 'wb') as f:
        pickle.dump(data, f, protocol=4)
    
    sz = os.path.getsize(args.output) / (1024**3)
    tt = time.time() - total_start
    print(f'Saved {args.output} ({sz:.2f} GB) in {tt:.1f}s ({tt/60:.1f} min)')
    print(f'\n=== Summary ===')
    print(f'Sessions: {len(all_neural)}, Subjects: {len(all_subjects)}')
    print(f'Total trials: {sum(len(t) for t in all_neural)}')
    for si in session_infos:
        print(f'  {si["session_id"]}: {si["n_neurons"]}n, {si["n_trials"]}t')
    
    # Cleanup temp files
    for tf in temp_files:
        os.remove(tf)

if __name__ == '__main__':
    main()
