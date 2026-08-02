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
from scipy import interpolate
import warnings
warnings.filterwarnings('ignore')

# Add code directory to path
sys.path.insert(0, 'code')

# Constants
FRAME_RATE = 3.17  # Hz
TIME_BIN_MS = 1000.0 / FRAME_RATE  # ~315.5 ms
CORRIDOR_LENGTH_VR = 40.0  # texture corridor in VR units (4m)
GRAY_LENGTH_VR = 20.0  # grey space in VR units (2m)
TOTAL_LENGTH_VR = 60.0  # total corridor + grey
VR_UNIT_TO_METERS = 0.1  # 1 VR unit = 0.1m
POSITION_BIN_EDGES = [0, 10, 20, 30, 40]  # 4 bins of 1m each in VR units
N_POSITION_BINS = 4
N_SPEED_BINS = 4


def neu_area_ID(iarea):
    """Map iarea indices to brain region groups. From reference code utils.py."""
    area_name = ['V1', 'mHV', 'lHV', 'aHV']
    idx = {}
    for ar in area_name:
        if ar == 'V1':
            idx[ar] = iarea == 8
        elif ar == 'mHV':
            idx[ar] = (iarea == 0) | (iarea == 1) | (iarea == 2) | (iarea == 9)
        elif ar == 'lHV':
            idx[ar] = (iarea == 5) | (iarea == 6)
        elif ar == 'aHV':
            idx[ar] = (iarea == 3) | (iarea == 4)
    return idx


def load_spk(mname, datexp, blk, root='data/spk'):
    """Load neural data for a session. From reference code utils.py."""
    fn = f'{mname}_{datexp}_{blk}_neural_data.npy'
    spk_path = os.path.join(root, fn)
    dat = np.load(spk_path, allow_pickle=True).item()
    spk = np.concatenate([nspk for nspk in dat['spks']], 0)
    return spk


def load_retino(mname, datexp, root='data/retinotopy'):
    """Load retinotopy data for a session. From reference code utils.py."""
    fn = f'{mname}_{datexp}_trans.npz'
    dtrans = np.load(os.path.join(root, fn), allow_pickle=True)
    ix = neu_area_ID(dtrans['iarea'])
    return {'iarea': dtrans['iarea'], 'neu_ar_idx': ix}


def get_brain_region_indices(iarea, brain_regions):
    """Map each neuron to a brain region index.
    
    brain_regions: list of region names
    Returns: array of indices into brain_regions for each neuron
    """
    region_map = neu_area_ID(iarea)
    n_neurons = len(iarea)
    region_idx = np.zeros(n_neurons, dtype=int)
    
    for i, region in enumerate(brain_regions):
        if region in region_map:
            region_idx[region_map[region]] = i
    
    return region_idx


def build_session_to_beh_map(exp_info):
    """Build a mapping from session_id to (beh_file, session_key).
    
    Returns dict: session_id -> list of (exp_type, session_key)
    """
    session_map = {}
    
    for exp_type, sessions in exp_info.items():
        for s in sessions:
            sess_id = f"{s['mname']}_{s['datexp']}_{s['blk']}"
            if sess_id not in session_map:
                session_map[sess_id] = []
            session_map[sess_id].append(exp_type)
    
    return session_map


def load_beh_for_session(sess_id, exp_types):
    """Load behavioral data for a session from the appropriate Beh file.
    
    Try each experiment type until we find the session.
    Returns list of (beh_data, session_key) tuples for all variants.
    """
    results = []
    for exp_type in exp_types:
        beh_path = f'data/beh/Beh_{exp_type}.npy'
        if os.path.exists(beh_path):
            beh_all = np.load(beh_path, allow_pickle=True).item()
            if sess_id in beh_all:
                results.append((beh_all[sess_id], sess_id))
            # Check for swap variants
            for key in sorted(beh_all.keys()):
                if key.startswith(sess_id + '_swap'):
                    results.append((beh_all[key], key))
    
    # Remove duplicates by session key
    seen = set()
    unique_results = []
    for beh, key in results:
        if key not in seen:
            seen.add(key)
            unique_results.append((beh, key))
    
    return unique_results if unique_results else None


def get_training_day(mname, datexp, exp_info):
    """Compute the training day index for a session.
    
    Returns the chronological day index (0-based) for this mouse.
    """
    # Collect all dates for this mouse
    dates = set()
    for exp_type, sessions in exp_info.items():
        for s in sessions:
            if s['mname'] == mname:
                dates.add(s['datexp'])
    
    sorted_dates = sorted(dates)
    return sorted_dates.index(datexp)


def extract_trial_data(spk, beh, trial_idx, n_timepoints):
    """Extract neural and behavioral data for a single trial.
    
    Args:
        spk: (n_neurons, n_total_frames) neural data
        beh: behavioral data dict
        trial_idx: trial index
        n_timepoints: fixed number of timepoints to extract
    
    Returns:
        neural: (n_neurons, n_timepoints) or None if trial too short
        position: (n_timepoints,) position in VR units
        speed: (n_timepoints,) running speed
        licking: (n_timepoints,) binary lick indicator
        sound_frame_offset: frame offset of sound cue from trial start
    """
    start_fr = int(np.round(beh['StartFr'][trial_idx]))
    gray_fr = int(np.round(beh['GrayFr'][trial_idx]))
    sound_fr = beh['SoundFr'][trial_idx]  # keep as float for precision
    
    # Available frames in corridor
    avail_frames = gray_fr - start_fr
    
    if avail_frames < 2:  # need at least 2 frames
        return None
    
    # Extract frames from start to start + n_timepoints (or available)
    end_fr = min(start_fr + n_timepoints, gray_fr)
    actual_frames = end_fr - start_fr
    
    if actual_frames < 2:
        return None
    
    # Clip to valid range
    if start_fr < 0 or end_fr > spk.shape[1]:
        return None
    
    # Neural data
    neural = spk[:, start_fr:end_fr].astype(np.float16)
    
    # Position
    position = beh['ft_Pos'][start_fr:end_fr].copy()
    
    # Running speed
    speed = beh['ft_RunSpeed'][start_fr:end_fr].copy()
    
    # Licking - binary per frame
    lick_frs = beh['LickFr']
    lick_trinds = beh['LickTrind']
    trial_lick_frs = lick_frs[lick_trinds == trial_idx]
    licking = np.zeros(actual_frames, dtype=np.float32)
    for lf in trial_lick_frs:
        lf_int = int(np.round(lf))
        frame_offset = lf_int - start_fr
        if 0 <= frame_offset < actual_frames:
            licking[frame_offset] = 1.0
    
    # Sound frame offset from trial start
    sound_frame_offset = sound_fr - start_fr
    
    return {
        'neural': neural,
        'position': position,
        'speed': speed,
        'licking': licking,
        'sound_frame_offset': sound_frame_offset,
        'actual_frames': actual_frames,
    }


def compute_speed_bin_edges(all_speeds):
    """Compute quartile bin edges for running speed."""
    # Flatten all speeds and remove NaN
    flat_speeds = np.concatenate(all_speeds)
    flat_speeds = flat_speeds[~np.isnan(flat_speeds)]
    
    # Compute quartile edges
    edges = np.percentile(flat_speeds, [0, 25, 50, 75, 100])
    return edges


def discretize_position(position, bin_edges=POSITION_BIN_EDGES):
    """Discretize position into bins.
    
    4 bins of 1m each: [0-10), [10-20), [20-30), [30-40]
    Returns bin indices 0-3.
    """
    bins = np.digitize(position, bin_edges[1:])  # 0,1,2,3
    bins = np.clip(bins, 0, N_POSITION_BINS - 1)
    return bins.astype(np.int64)


def discretize_speed(speed, bin_edges):
    """Discretize speed into quartile bins.
    
    Returns bin indices 0-3.
    """
    bins = np.digitize(speed, bin_edges[1:-1])  # 0,1,2,3
    bins = np.clip(bins, 0, N_SPEED_BINS - 1)
    return bins.astype(np.int64)


def get_stimulus_category(wall_name):
    """Get the visual stimulus category name."""
    return wall_name


def process_session(sess_id, sess_info, exp_info, speed_bin_edges=None, collect_speeds=False):
    """Process a single session.
    
    Args:
        sess_id: session identifier string
        sess_info: dict with session metadata
        exp_info: full experiment info dict
        speed_bin_edges: precomputed speed bin edges (None if collecting)
        collect_speeds: if True, just collect speeds for quartile computation
    
    Returns:
        dict with processed trial data, or list of speeds if collect_speeds
    """
    mname = sess_info['mname']
    datexp = sess_info['datexp']
    blk = sess_info['blk']
    
    # Load behavioral data
    exp_types = sess_info['exp_types']
    beh_results = load_beh_for_session(sess_id, exp_types)
    if beh_results is None:
        print(f'  WARNING: No behavioral data found for {sess_id}')
        return None
    
    # Use the first (primary) behavioral data
    beh, beh_key = beh_results[0]
    ntrials = beh['ntrials']
    
    if collect_speeds:
        # Just collect running speeds from corridor frames
        speeds = []
        for t in range(ntrials):
            start_fr = int(np.round(beh['StartFr'][t]))
            gray_fr = int(np.round(beh['GrayFr'][t]))
            if start_fr < len(beh['ft_RunSpeed']) and gray_fr <= len(beh['ft_RunSpeed']):
                trial_speed = beh['ft_RunSpeed'][start_fr:gray_fr]
                speeds.append(trial_speed)
        return speeds
    
    # Load neural data
    t0 = time.time()
    spk = load_spk(mname, datexp, blk)
    t_load = time.time() - t0
    
    # Load retinotopy
    retino = load_retino(mname, datexp)
    
    n_neurons = spk.shape[0]
    n_total_frames = spk.shape[1]
    
    # Get training day
    training_day = get_training_day(mname, datexp, exp_info)
    
    # Brain region assignment
    brain_regions = ['V1', 'mHV', 'lHV', 'aHV', 'unassigned']
    iarea = retino['iarea']
    region_map = neu_area_ID(iarea)
    
    # Assign each neuron to a region index
    neuron_region_idx = np.full(n_neurons, len(brain_regions) - 1, dtype=int)  # default: unassigned
    for i, region in enumerate(brain_regions[:-1]):
        if region in region_map:
            neuron_region_idx[region_map[region]] = i
    
    # Process trials
    trial_neural = []
    trial_inputs = []
    trial_outputs = []
    
    # Get all unique stimulus names across the session
    wall_names = beh['WallName']
    
    for t in range(ntrials):
        trial_data = extract_trial_data(spk, beh, t, n_timepoints=1000)  # large max
        
        if trial_data is None:
            continue
        
        actual_frames = trial_data['actual_frames']
        
        # === NEURAL DATA ===
        neural = trial_data['neural']  # (n_neurons, actual_frames)
        
        # === INPUT DATA ===
        # Input 0: time_to_sound_cue (continuous, time-varying)
        # Signed time from current frame to sound cue (negative = before cue, positive = after)
        sound_offset = trial_data['sound_frame_offset']  # frame offset of sound from trial start
        time_to_cue = np.array([(i - sound_offset) / FRAME_RATE for i in range(actual_frames)], dtype=np.float32)
        
        # Input 1: day_of_training (continuous, per-trial)
        day_val = np.full(actual_frames, training_day, dtype=np.float32)
        
        # Input 2: time_since_trial_start (continuous, time-varying)
        time_since_start = np.array([i / FRAME_RATE for i in range(actual_frames)], dtype=np.float32)
        
        # Input 3: reward_availability (discrete, per-trial)
        reward_avail = np.full(actual_frames, float(beh['isRew'][t]), dtype=np.float32)
        
        inputs = np.stack([time_to_cue, day_val, time_since_start, reward_avail], axis=0)  # (4, actual_frames)
        
        # === OUTPUT DATA ===
        # Output 0: visual_stimulus_category (per-trial)
        stim_name = str(wall_names[t])
        
        # Output 1: licking (binary, time-varying)
        licking = trial_data['licking']  # (actual_frames,)
        
        # Output 2: position_bin (4 bins, time-varying)
        pos_bins = discretize_position(trial_data['position'])
        
        # Output 3: speed_bin (4 quartile bins, time-varying)
        spd_bins = discretize_speed(trial_data['speed'], speed_bin_edges)
        
        # Stack outputs: (4, actual_frames)
        # For per-trial outputs, repeat across time
        # stim_category needs to be encoded as integer
        # We'll handle this in the final assembly
        
        trial_neural.append(neural)
        trial_inputs.append(inputs)
        trial_outputs.append({
            'stim_name': stim_name,
            'licking': licking,
            'position_bin': pos_bins,
            'speed_bin': spd_bins,
        })
    
    if len(trial_neural) < 2:
        print(f'  WARNING: Session {sess_id} has fewer than 2 valid trials, skipping')
        return None
    
    return {
        'neural': trial_neural,
        'inputs': trial_inputs,
        'outputs': trial_outputs,
        'neuron_region_idx': neuron_region_idx,
        'n_neurons': n_neurons,
        'mname': mname,
        'sess_id': sess_id,
        'ntrials_original': ntrials,
        'ntrials_valid': len(trial_neural),
        'load_time': t_load,
    }


def build_all_sessions_info(exp_info):
    """Build a list of all unique sessions with their metadata."""
    sessions = {}
    for exp_type, sess_list in exp_info.items():
        for s in sess_list:
            sess_id = f"{s['mname']}_{s['datexp']}_{s['blk']}"
            if sess_id not in sessions:
                sessions[sess_id] = {
                    'mname': s['mname'],
                    'datexp': s['datexp'],
                    'blk': s['blk'],
                    'exp_types': [],
                    'exptype': s.get('exptype', 'unknown'),
                    'rewType': s.get('rewType', 'None'),
                }
            sessions[sess_id]['exp_types'].append(exp_type)
    
    return sessions


def main():
    parser = argparse.ArgumentParser(description='Convert Zhong et al. 2025 data')
    parser.add_argument('output', help='Output pickle file path')
    parser.add_argument('--full', action='store_true', default=True, help='Process all sessions (default)')
    parser.add_argument('--sample', action='store_true', help='Process only 2 sessions')
    parser.add_argument('--show-processing', action='store_true', help='Show processing plots')
    args = parser.parse_args()
    
    if args.sample:
        args.full = False
    
    total_start = time.time()
    
    # Load experiment info
    print('Loading experiment info...')
    exp_info = np.load('data/beh/Imaging_Exp_info.npy', allow_pickle=True).item()
    
    # Build session list
    all_sessions = build_all_sessions_info(exp_info)
    print(f'Total unique sessions: {len(all_sessions)}')
    
    # Select sessions to process
    session_ids = sorted(all_sessions.keys())
    if args.sample:
        # Pick 2 sessions - one supervised, one unsupervised
        sup_sessions = [s for s in session_ids if all_sessions[s]['exptype'] == 'sup']
        unsup_sessions = [s for s in session_ids if all_sessions[s]['exptype'] == 'unsup']
        selected = []
        if sup_sessions:
            selected.append(sup_sessions[0])
        if unsup_sessions:
            selected.append(unsup_sessions[0])
        if len(selected) < 2 and len(session_ids) >= 2:
            for s in session_ids:
                if s not in selected:
                    selected.append(s)
                    if len(selected) >= 2:
                        break
        session_ids = selected
    
    print(f'Processing {len(session_ids)} sessions...')
    
    # === PASS 1: Collect running speeds for quartile computation ===
    print('\n=== Pass 1: Collecting running speeds for quartile computation ===')
    t0 = time.time()
    all_speeds = []
    for i, sess_id in enumerate(session_ids):
        sess_info = all_sessions[sess_id]
        speeds = process_session(sess_id, sess_info, exp_info, collect_speeds=True)
        if speeds is not None:
            all_speeds.extend(speeds)
        if (i + 1) % 10 == 0:
            print(f'  Collected speeds from {i+1}/{len(session_ids)} sessions')
    
    speed_bin_edges = compute_speed_bin_edges(all_speeds)
    print(f'Speed quartile edges: {speed_bin_edges}')
    print(f'Pass 1 took {time.time()-t0:.1f}s')
    del all_speeds  # free memory
    
    # === PASS 2: Process all sessions ===
    print('\n=== Pass 2: Processing sessions ===')
    
    # Collect all unique stimulus names
    all_stim_names = set()
    all_results = []
    
    for i, sess_id in enumerate(session_ids):
        t0 = time.time()
        sess_info = all_sessions[sess_id]
        print(f'\n[{i+1}/{len(session_ids)}] Processing {sess_id} (exptype={sess_info["exptype"]})...')
        
        result = process_session(sess_id, sess_info, exp_info, speed_bin_edges=speed_bin_edges)
        
        if result is None:
            print(f'  SKIPPED')
            continue
        
        # Collect stimulus names
        for trial_out in result['outputs']:
            all_stim_names.add(trial_out['stim_name'])
        
        all_results.append(result)
        t_total = time.time() - t0
        print(f'  Neurons: {result["n_neurons"]}, Trials: {result["ntrials_valid"]}/{result["ntrials_original"]}, '
              f'Load: {result["load_time"]:.1f}s, Total: {t_total:.1f}s')
    
    print(f'\nProcessed {len(all_results)} sessions successfully')
    
    # === Build final data structure ===
    print('\n=== Building final data structure ===')
    
    # Sort stimulus names
    all_stim_names = sorted(all_stim_names)
    stim_to_idx = {name: i for i, name in enumerate(all_stim_names)}
    print(f'Stimulus categories: {all_stim_names}')
    
    # Brain regions
    brain_regions = ['V1', 'mHV', 'lHV', 'aHV', 'unassigned']
    
    # Build data dict
    neural_list = []
    input_list = []
    output_list = []
    subjects = []
    subject_idx = []
    brain_region_idx_list = []
    
    # Collect unique subjects
    all_mice = sorted(set(r['mname'] for r in all_results))
    mouse_to_idx = {m: i for i, m in enumerate(all_mice)}
    
    # Input names
    input_names = ['time_to_sound_cue', 'day_of_training', 'time_since_trial_start', 'reward_availability']
    
    # Output names and values
    output_names = ['visual_stimulus', 'licking', 'position_bin', 'speed_bin']
    output_values = [
        all_stim_names,  # visual stimulus categories
        ['no_lick', 'lick'],  # licking
        ['0-1m', '1-2m', '2-3m', '3-4m'],  # position bins
        ['Q1_slowest', 'Q2', 'Q3', 'Q4_fastest'],  # speed bins
    ]
    
    for result in all_results:
        # Neural
        neural_list.append(result['neural'])
        
        # Brain region idx
        brain_region_idx_list.append(result['neuron_region_idx'])
        
        # Subject
        subject_idx.append(mouse_to_idx[result['mname']])
        
        # Inputs
        session_inputs = []
        for trial_inp in result['inputs']:
            session_inputs.append(trial_inp)  # (4, n_timepoints)
        input_list.append(session_inputs)
        
        # Outputs
        session_outputs = []
        for trial_out in result['outputs']:
            stim_idx = stim_to_idx[trial_out['stim_name']]
            n_t = len(trial_out['licking'])
            
            # Build output array: (4, n_timepoints)
            out = np.zeros((4, n_t), dtype=np.int64)
            out[0, :] = stim_idx  # visual stimulus (per-trial, repeated)
            out[1, :] = trial_out['licking'].astype(np.int64)  # licking
            out[2, :] = trial_out['position_bin']  # position bin
            out[3, :] = trial_out['speed_bin']  # speed bin
            
            session_outputs.append(out)
        output_list.append(session_outputs)
    
    data = {
        'neural': neural_list,
        'input': input_list,
        'output': output_list,
        'subjects': all_mice,
        'subject_idx': np.array(subject_idx),
        'brain_regions': brain_regions,
        'brain_region_idx': brain_region_idx_list,
        'input_names': input_names,
        'output_names': output_names,
        'output_values': output_values,
        'metadata': {
            'task_description': 'Visual discrimination task in head-fixed mice running through virtual reality corridors with naturalistic textures. Mice discriminate between visual patterns (leaf/circle/rock/brick) with reward delivery in one corridor type.',
            'time_bin_size': TIME_BIN_MS,
            'temporal_alignment_event': 'Trial start (corridor entry)',
            'off_start': 0.0,
            'off_end': None,  # variable trial length
            'frame_rate_hz': FRAME_RATE,
            'speed_bin_edges': speed_bin_edges.tolist(),
            'position_bin_edges_vr_units': POSITION_BIN_EDGES,
            'position_bin_edges_meters': [e * VR_UNIT_TO_METERS for e in POSITION_BIN_EDGES],
            'corridor_length_m': CORRIDOR_LENGTH_VR * VR_UNIT_TO_METERS,
            'n_sessions': len(all_results),
            'n_subjects': len(all_mice),
            'session_ids': [r['sess_id'] for r in all_results],
        },
    }
    
    # Print summary
    print('\n=== Data Summary ===')
    print(f'Sessions: {len(neural_list)}')
    print(f'Subjects: {len(all_mice)} - {all_mice}')
    total_trials = sum(len(s) for s in neural_list)
    print(f'Total trials: {total_trials}')
    total_neurons = sum(r['n_neurons'] for r in all_results)
    print(f'Total neurons: {total_neurons}')
    print(f'Neurons per session: {[r["n_neurons"] for r in all_results]}')
    print(f'Trials per session: {[len(s) for s in neural_list]}')
    print(f'Brain regions: {brain_regions}')
    print(f'Input names: {input_names}')
    print(f'Output names: {output_names}')
    print(f'Output values: {output_values}')
    print(f'Speed bin edges: {speed_bin_edges}')
    
    # Show processing plots if requested
    if args.show_processing:
        show_processing_plots(data, all_results, speed_bin_edges)
    
    # Save
    print(f'\nSaving to {args.output}...')
    with open(args.output, 'wb') as f:
        pickle.dump(data, f, protocol=4)
    
    file_size = os.path.getsize(args.output)
    print(f'Saved {args.output} ({file_size/1e9:.2f} GB)')
    
    total_time = time.time() - total_start
    print(f'\nTotal conversion time: {total_time:.1f}s ({total_time/60:.1f}min)')


def show_processing_plots(data, all_results, speed_bin_edges):
    """Generate processing visualization plots."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib not available, skipping plots')
        return
    
    n_sessions_to_plot = min(2, len(all_results))
    
    for s_idx in range(n_sessions_to_plot):
        result = all_results[s_idx]
        sess_id = result['sess_id']
        
        fig, axes = plt.subplots(4, 2, figsize=(16, 20))
        fig.suptitle(f'Processing: {sess_id}', fontsize=14)
        
        # Pick a sample trial
        trial_idx = min(5, len(result['neural']) - 1)
        neural = result['neural'][trial_idx]
        inputs = result['inputs'][trial_idx]
        outputs = result['outputs'][trial_idx]
        
        n_t = neural.shape[1]
        time_axis = np.arange(n_t) / FRAME_RATE
        
        # Plot 1: Neural activity (first 20 neurons)
        ax = axes[0, 0]
        n_show = min(20, neural.shape[0])
        im = ax.imshow(neural[:n_show], aspect='auto', interpolation='none')
        ax.set_title(f'Neural activity (first {n_show} neurons)')
        ax.set_xlabel('Frame')
        ax.set_ylabel('Neuron')
        plt.colorbar(im, ax=ax)
        
        # Plot 2: Input - time to sound cue
        ax = axes[0, 1]
        ax.plot(time_axis, inputs[0], 'b-')
        ax.axhline(0, color='r', linestyle='--', alpha=0.5)
        ax.set_title('Input: Time to sound cue (s)')
        ax.set_xlabel('Time (s)')
        
        # Plot 3: Input - time since trial start
        ax = axes[1, 0]
        ax.plot(time_axis, inputs[2], 'g-')
        ax.set_title('Input: Time since trial start (s)')
        ax.set_xlabel('Time (s)')
        
        # Plot 4: Input - reward availability
        ax = axes[1, 1]
        ax.plot(time_axis, inputs[3], 'r-')
        ax.set_title(f'Input: Reward availability ({inputs[3][0]:.0f})')
        ax.set_xlabel('Time (s)')
        ax.set_ylim(-0.1, 1.1)
        
        # Plot 5: Output - licking
        ax = axes[2, 0]
        licking = outputs['licking']
        ax.plot(time_axis, licking, 'k-')
        ax.set_title(f'Output: Licking (stim={outputs["stim_name"]})')
        ax.set_xlabel('Time (s)')
        ax.set_ylim(-0.1, 1.1)
        
        # Plot 6: Output - position bin
        ax = axes[2, 1]
        ax.plot(time_axis, outputs['position_bin'], 'b-')
        ax.set_title('Output: Position bin (0-3)')
        ax.set_xlabel('Time (s)')
        ax.set_yticks([0, 1, 2, 3])
        ax.set_yticklabels(['0-1m', '1-2m', '2-3m', '3-4m'])
        
        # Plot 7: Output - speed bin
        ax = axes[3, 0]
        ax.plot(time_axis, outputs['speed_bin'], 'r-')
        ax.set_title('Output: Speed bin (0-3)')
        ax.set_xlabel('Time (s)')
        ax.set_yticks([0, 1, 2, 3])
        ax.set_yticklabels(['Q1', 'Q2', 'Q3', 'Q4'])
        
        # Plot 8: Brain region distribution
        ax = axes[3, 1]
        brain_regions = ['V1', 'mHV', 'lHV', 'aHV', 'unassigned']
        region_counts = [np.sum(result['neuron_region_idx'] == i) for i in range(len(brain_regions))]
        ax.bar(brain_regions, region_counts)
        ax.set_title('Brain region distribution')
        ax.set_ylabel('Number of neurons')
        
        plt.tight_layout()
        plt.savefig(f'processing_{sess_id}.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f'Saved processing_{sess_id}.png')


if __name__ == '__main__':
    main()
