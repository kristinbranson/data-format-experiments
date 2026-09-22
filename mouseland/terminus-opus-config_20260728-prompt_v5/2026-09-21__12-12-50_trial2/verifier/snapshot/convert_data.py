#!/usr/bin/env python3
"""Convert Zhong et al. 2025 data to decoder-compatible format.

Usage:
    python -u convert_data.py <outpicklefile> [--sample] [--full] [--show-processing]
"""

import os
import sys
import time
import argparse
import pickle
import numpy as np
from scipy import interpolate
from collections import defaultdict

# ============================================================
# Constants
# ============================================================
DATA_ROOT = '/app/data'
FS = 3.17  # Hz, calcium imaging frame rate
VR_SPEED = 0.6  # m/s = 60 cm/s constant VR speed
CORRIDOR_LENGTH_M = 4.0  # meters of texture corridor
GREY_LENGTH_M = 2.0  # meters of grey space
TOTAL_LENGTH_M = CORRIDOR_LENGTH_M + GREY_LENGTH_M  # 6m total
N_POS_BINS = 60  # position bins per trial (corridor + grey)
N_CORRIDOR_BINS = 40  # position bins in corridor only
N_GREY_BINS = 20  # position bins in grey space
BIN_SIZE_M = TOTAL_LENGTH_M / N_POS_BINS  # 0.1m per bin
TIME_PER_BIN = BIN_SIZE_M / VR_SPEED  # ~0.1667s per bin
TIME_BIN_MS = TIME_PER_BIN * 1000  # ~166.67 ms

# Brain area mapping from retinotopy iarea values
# From utils.py neu_area_ID function
AREA_MAPPING = {
    'V1': [8],
    'mHV': [0, 1, 2, 9],
    'lHV': [5, 6],
    'aHV': [3, 4],
}
EXCLUDED_AREAS = [-1, 7]  # outside visual cortex


def load_exp_info():
    """Load experiment info."""
    return np.load(os.path.join(DATA_ROOT, 'beh', 'Imaging_Exp_info.npy'),
                   allow_pickle=True).item()


def get_unique_sessions(exp_info):
    """Get list of unique sessions with metadata."""
    sessions = {}
    for exp_type, sess_list in exp_info.items():
        for s in sess_list:
            key = f"{s['mname']}_{s['datexp']}_{s['blk']}"
            if key not in sessions:
                sessions[key] = {
                    'mname': s['mname'],
                    'datexp': s['datexp'],
                    'blk': s['blk'],
                    'key': key,
                    'exp_types': [],
                    'sess_num': s.get('sess#', 0),
                    'exptype': s.get('exptype', ''),
                    'rewType': s.get('rewType', ''),
                    'stim_id': s.get('stim_id', None),
                }
            sessions[key]['exp_types'].append(exp_type)
    return sessions


def load_spk(mname, datexp, blk):
    """Load spike data, concatenating across imaging planes."""
    fn = f'{mname}_{datexp}_{blk}_neural_data.npy'
    spk_path = os.path.join(DATA_ROOT, 'spk', fn)
    spk_data = np.load(spk_path, allow_pickle=True).item()
    spk = np.concatenate([nspk for nspk in spk_data['spks']], 0)
    return spk


def load_retino(mname, datexp):
    """Load retinotopy data for brain area assignment."""
    fn = f'{mname}_{datexp}_trans.npz'
    ret_path = os.path.join(DATA_ROOT, 'retinotopy', fn)
    dtrans = np.load(ret_path, allow_pickle=True)
    return dtrans['iarea']


def get_brain_region_idx(iarea):
    """Map iarea values to brain region indices.
    Returns mask of valid neurons and their region indices.
    Brain regions: ['V1', 'mHV', 'lHV', 'aHV']
    """
    region_names = ['V1', 'mHV', 'lHV', 'aHV']
    n_neurons = len(iarea)
    region_idx = np.full(n_neurons, -1, dtype=int)
    
    for r_idx, (region, areas) in enumerate(AREA_MAPPING.items()):
        for area_val in areas:
            mask = iarea == area_val
            region_idx[mask] = r_idx
    
    # Valid neurons are those assigned to a region (not -1)
    valid_mask = region_idx >= 0
    return valid_mask, region_idx


# Global cache for behavioral data
_beh_cache = {}

def load_beh_for_session(session_key, exp_info):
    """Load behavioral data for a session.
    Uses caching to avoid reloading large files.
    """
    # Check cache first
    if session_key in _beh_cache:
        return _beh_cache[session_key]
    
    # Get the experiment types this session belongs to
    sessions = get_unique_sessions(exp_info)
    sess = sessions[session_key]
    
    # Try each experiment type until we find the behavioral data
    for exp_type in sess['exp_types']:
        beh_file = os.path.join(DATA_ROOT, 'beh', f'Beh_{exp_type}.npy')
        if os.path.exists(beh_file):
            beh = np.load(beh_file, allow_pickle=True).item()
            # Cache all sessions from this file
            for k, v in beh.items():
                if k not in _beh_cache:
                    _beh_cache[k] = v
            if session_key in _beh_cache:
                return _beh_cache[session_key]
            # Check for keys with stimtype suffix
            for k in beh.keys():
                if k.startswith(session_key):
                    _beh_cache[session_key] = beh[k]
                    return beh[k]
    
    raise ValueError(f"Could not find behavioral data for session {session_key}")


def interp_value(v, vind, tind):
    """Interpolate values. From utils.py."""
    Model_ = interpolate.interp1d(vind, v, fill_value='extrapolate')
    return Model_(tind)


def position_interpolate_spk(raw_spk, accum_pos, corridor_len, n_trials, n_bins=60):
    """Interpolate spike data to position bins.
    
    Uses np.interp for fast vectorized interpolation.
    
    Args:
        raw_spk: (n_neurons, n_valid_frames) - spike data for moving frames only
        accum_pos: (n_valid_frames,) - cumulative position for moving frames
        corridor_len: float - corridor length in position units
        n_trials: int - number of trials
        n_bins: int - bins per trial
    
    Returns:
        interp_spk: (n_neurons, n_trials, n_bins)
    """
    # Target positions in normalized units (trial number)
    lin_pos = np.arange(0, n_trials, 1.0 / n_bins)
    # Source positions in normalized units
    src_pos = accum_pos / corridor_len
    
    n_neurons = raw_spk.shape[0]
    interp_spk = np.zeros((n_neurons, n_trials * n_bins), dtype=np.float32)
    
    # Process in batches for memory efficiency
    batch_size = 2000
    for i in range(0, n_neurons, batch_size):
        end_i = min(i + batch_size, n_neurons)
        batch = raw_spk[i:end_i]
        for s in range(batch.shape[0]):
            interp_spk[i + s] = np.interp(lin_pos, src_pos, batch[s])
    
    return interp_spk.reshape(n_neurons, n_trials, n_bins)


def make_lick_raster(lick_pos, lick_trind, n_trials, n_bins=60, corridor_len=60.0):
    """Convert lick events to binary raster at position bins.
    
    Args:
        lick_pos: position of each lick event
        lick_trind: trial index of each lick event
        n_trials: number of trials
        n_bins: number of position bins
        corridor_len: corridor length in position units
    
    Returns:
        lick_raster: (n_trials, n_bins) binary array
    """
    lick_raster = np.zeros((n_trials, n_bins), dtype=np.float32)
    bin_edges = np.linspace(0, corridor_len, n_bins + 1)
    
    for i in range(len(lick_pos)):
        tr = int(lick_trind[i])
        pos = lick_pos[i]
        if tr < 0 or tr >= n_trials:
            continue
        # Find which bin this lick falls in
        bin_idx = np.searchsorted(bin_edges, pos, side='right') - 1
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)
        lick_raster[tr, bin_idx] = 1.0
    
    return lick_raster


def make_position_output(n_trials, n_bins=60):
    """Create discretized position output.
    Position in corridor discretized into 4 equal-length, 1-m-long spatial bins.
    Corridor is 4m (bins 0-40), grey space is 2m (bins 40-60).
    
    Bin 0: 0-1m (position bins 0-10)
    Bin 1: 1-2m (position bins 10-20)  
    Bin 2: 2-3m (position bins 20-30)
    Bin 3: 3-4m (position bins 30-40)
    Grey space (bins 40-60): assign to bin 3 (last corridor bin)
    
    Returns:
        pos_output: (n_trials, n_bins) with values 0-3
    """
    pos_output = np.zeros((n_trials, n_bins), dtype=np.float32)
    for b in range(n_bins):
        if b < 10:
            pos_output[:, b] = 0
        elif b < 20:
            pos_output[:, b] = 1
        elif b < 30:
            pos_output[:, b] = 2
        else:  # bins 30-59 (3-4m corridor + grey space)
            pos_output[:, b] = 3
    return pos_output


def make_speed_output(run_pos, speed_bin_edges):
    """Discretize running speed into 4 quartile bins.
    
    Args:
        run_pos: (n_trials, n_bins) running speed at each position
        speed_bin_edges: edges for 4 speed bins (from global quartiles)
    
    Returns:
        speed_output: (n_trials, n_bins) with values 0-3
    """
    speed_output = np.digitize(run_pos, speed_bin_edges[1:-1]).astype(np.float32)
    return speed_output


def make_time_to_sound_cue(sound_pos, n_bins=60):
    """Create time-to-sound-cue input for each position bin.
    
    Args:
        sound_pos: (n_trials,) sound cue position in position units (0-60)
        n_bins: number of position bins
    
    Returns:
        time_to_cue: (n_trials, n_bins) time to sound cue in seconds
    """
    positions = np.arange(n_bins) + 0.5  # center of each bin
    # Distance in position units, then convert to time
    dist = sound_pos[:, np.newaxis] - positions[np.newaxis, :]  # positive = cue ahead
    time_to_cue = (dist * BIN_SIZE_M / VR_SPEED).astype(np.float32)  # convert to seconds
    return time_to_cue


def make_time_since_trial_start(n_trials, n_bins=60):
    """Create time-since-trial-start input.
    
    Returns:
        time_since_start: (n_trials, n_bins) time in seconds from trial start
    """
    time_per_bin = BIN_SIZE_M / VR_SPEED
    times = (np.arange(n_bins) * time_per_bin).astype(np.float32)
    return np.tile(times, (n_trials, 1))


def get_stimulus_category(wall_name, uniq_walls):
    """Map wall names to stimulus category indices.
    
    Returns:
        categories: (n_trials,) integer category indices
        category_names: list of category name strings
    """
    # Use unique wall names as categories
    category_names = list(uniq_walls)
    categories = np.array([category_names.index(wn) for wn in wall_name], dtype=np.int64)
    return categories, category_names


def compute_global_speed_quartiles(exp_info):
    """Compute global speed quartiles across all sessions.
    Loads each beh file only once for efficiency."""
    print("Computing global speed quartiles...")
    all_speeds = []
    loaded_sessions = set()
    
    for exp_type in exp_info.keys():
        beh_file = os.path.join(DATA_ROOT, 'beh', f'Beh_{exp_type}.npy')
        if not os.path.exists(beh_file):
            continue
        beh = np.load(beh_file, allow_pickle=True).item()
        for session_key, dat in beh.items():
            if session_key in loaded_sessions:
                continue
            loaded_sessions.add(session_key)
            if 'run_pos' in dat:
                run_pos = dat['run_pos']  # (n_trials, 60)
                all_speeds.append(run_pos.ravel())
    
    all_speeds = np.concatenate(all_speeds)
    # Remove NaN and zero/negative speeds
    valid = np.isfinite(all_speeds) & (all_speeds > 0)
    all_speeds = all_speeds[valid]
    
    quartiles = np.percentile(all_speeds, [25, 50, 75])
    bin_edges = np.array([0, quartiles[0], quartiles[1], quartiles[2], np.inf])
    print(f"  Speed quartile edges: {bin_edges}")
    print(f"  Loaded {len(loaded_sessions)} sessions for speed computation")
    return bin_edges


def process_session(session_key, sess_meta, exp_info, speed_bin_edges, 
                    show_processing=False, session_idx=0):
    """Process a single session.
    
    Returns dict with neural, input, output data for this session.
    """
    t0 = time.time()
    mname = sess_meta['mname']
    datexp = sess_meta['datexp']
    blk = sess_meta['blk']
    
    print(f"  Loading data for {session_key}...")
    
    # Load spike data
    spk = load_spk(mname, datexp, blk)
    n_neurons_total, n_frames = spk.shape
    print(f"    Spikes: {n_neurons_total} neurons, {n_frames} frames")
    
    # Load retinotopy for brain region assignment
    iarea = load_retino(mname, datexp)
    assert len(iarea) == n_neurons_total, \
        f"Retinotopy mismatch: {len(iarea)} vs {n_neurons_total} neurons"
    
    # Filter neurons to visual cortex only
    valid_mask, region_idx = get_brain_region_idx(iarea)
    spk = spk[valid_mask]
    region_idx = region_idx[valid_mask]
    n_neurons = spk.shape[0]
    print(f"    After filtering: {n_neurons} neurons in visual cortex")
    
    # Load behavioral data
    beh = load_beh_for_session(session_key, exp_info)
    n_trials = beh['ntrials']
    corridor_len = beh['Corridor_Length']
    print(f"    Trials: {n_trials}, Corridor length: {corridor_len}")
    
    # Get frame-level data (truncate to match spike frames)
    ft_move = beh['ft_move'][:n_frames]
    ft_pos_cum = beh['ft_PosCum'][:n_frames]
    vr_moving = ft_move > 0
    
    # Position-interpolate neural activity
    print(f"    Interpolating neural activity...")
    t_interp = time.time()
    interp_spk = position_interpolate_spk(
        spk[:, vr_moving], ft_pos_cum[vr_moving], 
        corridor_len, n_trials, N_POS_BINS
    )
    print(f"    Interpolation took {time.time()-t_interp:.1f}s")
    
    # === NEURAL DATA ===
    # interp_spk shape: (n_neurons, n_trials, n_bins)
    neural_trials = [interp_spk[:, t, :].astype(np.float32) for t in range(n_trials)]
    
    # === INPUTS ===
    # Input 0: Time to sound cue (continuous, time-varying)
    sound_pos = beh['SoundPos']  # position units
    time_to_cue = make_time_to_sound_cue(sound_pos, N_POS_BINS)
    
    # Input 1: Day of training (continuous, per-trial)
    day_of_training = np.full(n_trials, sess_meta['sess_num'], dtype=np.float32)
    
    # Input 2: Time since trial start (continuous, time-varying)
    time_since_start = make_time_since_trial_start(n_trials, N_POS_BINS)
    
    # Input 3: Reward availability (discrete, per-trial)
    reward_avail = beh['isRew'].astype(np.float32)
    
    # Stack inputs: shape (n_input, n_timepoints) or (n_input,) per trial
    input_trials = []
    for t in range(n_trials):
        inp = np.stack([
            time_to_cue[t],          # (n_bins,)
            np.full(N_POS_BINS, day_of_training[t], dtype=np.float32),  # (n_bins,)
            time_since_start[t],     # (n_bins,)
            np.full(N_POS_BINS, reward_avail[t], dtype=np.float32),     # (n_bins,)
        ], axis=0)  # (4, n_bins)
        input_trials.append(inp)
    
    # === OUTPUTS ===
    # Output 0: Visual stimulus category (per-trial)
    wall_name = beh['WallName']
    uniq_walls = beh['UniqWalls']
    stim_categories, stim_names = get_stimulus_category(wall_name, uniq_walls)
    
    # Output 1: Licking (binary, time-varying)
    lick_raster = make_lick_raster(
        beh['LickPos'], beh['LickTrind'], n_trials, N_POS_BINS, corridor_len
    )
    
    # Output 2: Position in corridor (4 bins, time-varying)
    pos_output = make_position_output(n_trials, N_POS_BINS)
    
    # Output 3: Running speed (4 bins, time-varying)
    run_pos = beh['run_pos']  # (n_trials, 60)
    speed_output = make_speed_output(run_pos, speed_bin_edges)
    
    # Stack outputs per trial
    output_trials = []
    for t in range(n_trials):
        out = np.stack([
            np.full(N_POS_BINS, stim_categories[t], dtype=np.int64),  # (n_bins,) - per trial but broadcast
            lick_raster[t].astype(np.int64),   # (n_bins,)
            pos_output[t].astype(np.int64),    # (n_bins,)
            speed_output[t].astype(np.int64),  # (n_bins,)
        ], axis=0)  # (4, n_bins)
        output_trials.append(out)
    
    elapsed = time.time() - t0
    print(f"    Session processed in {elapsed:.1f}s")
    
    if show_processing and session_idx < 2:
        plot_processing(session_key, interp_spk, input_trials, output_trials,
                       beh, stim_names, speed_bin_edges, session_idx)
    
    return {
        'neural': neural_trials,
        'input': input_trials,
        'output': output_trials,
        'region_idx': region_idx,
        'n_neurons': n_neurons,
        'n_trials': n_trials,
        'stim_names': stim_names,
    }


def plot_processing(session_key, interp_spk, input_trials, output_trials,
                   beh, stim_names, speed_bin_edges, session_idx):
    """Plot processing steps for visual verification."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(4, 3, figsize=(18, 16))
    fig.suptitle(f'Processing: {session_key}', fontsize=14)
    
    n_trials = interp_spk.shape[1]
    trial_idx = min(10, n_trials - 1)  # Pick a trial to visualize
    
    # Row 0: Neural activity
    ax = axes[0, 0]
    ax.imshow(interp_spk[:50, trial_idx, :], aspect='auto', cmap='gray_r')
    ax.set_title(f'Neural (first 50 neurons, trial {trial_idx})')
    ax.set_xlabel('Position bin')
    ax.set_ylabel('Neuron')
    
    ax = axes[0, 1]
    ax.plot(interp_spk[:, trial_idx, :].mean(0))
    ax.set_title(f'Mean neural activity (trial {trial_idx})')
    ax.set_xlabel('Position bin')
    ax.axvline(40, color='r', linestyle='--', label='Grey space')
    ax.legend()
    
    ax = axes[0, 2]
    ax.hist(interp_spk[:, trial_idx, :].ravel(), bins=50)
    ax.set_title('Neural activity distribution')
    
    # Row 1: Inputs
    inp = input_trials[trial_idx]
    input_names = ['time_to_sound_cue', 'day_of_training', 'time_since_trial_start', 'reward_availability']
    for i in range(min(3, inp.shape[0])):
        ax = axes[1, i]
        ax.plot(inp[i])
        ax.set_title(f'Input: {input_names[i]}')
        ax.set_xlabel('Position bin')
        ax.axvline(40, color='r', linestyle='--')
    
    # Row 2: Outputs
    out = output_trials[trial_idx]
    output_names = ['stimulus_category', 'licking', 'position_bin', 'running_speed_bin']
    for i in range(min(3, out.shape[0])):
        ax = axes[2, i]
        ax.plot(out[i])
        ax.set_title(f'Output: {output_names[i]}')
        ax.set_xlabel('Position bin')
        ax.axvline(40, color='r', linestyle='--')
    
    # Row 3: Summary stats
    ax = axes[2, 2] if out.shape[0] > 3 else axes[3, 0]
    if out.shape[0] > 3:
        ax = axes[3, 0]
    ax.plot(out[3] if out.shape[0] > 3 else out[0])
    ax.set_title(f'Output: {output_names[3] if out.shape[0] > 3 else output_names[0]}')
    ax.set_xlabel('Position bin')
    
    ax = axes[3, 1]
    # Lick raster across trials
    lick_data = np.array([output_trials[t][1] for t in range(n_trials)])
    ax.imshow(lick_data[:50], aspect='auto', cmap='gray_r')
    ax.set_title('Lick raster (first 50 trials)')
    ax.set_xlabel('Position bin')
    ax.set_ylabel('Trial')
    
    ax = axes[3, 2]
    # Speed distribution
    speed_data = np.array([output_trials[t][3] for t in range(n_trials)])
    ax.hist(speed_data.ravel(), bins=4, range=(-0.5, 3.5))
    ax.set_title('Speed bin distribution')
    ax.set_xticks([0, 1, 2, 3])
    
    plt.tight_layout()
    plt.savefig(f'/app/processing_{session_key}.png', dpi=100)
    plt.close()
    print(f"    Saved processing plot to /app/processing_{session_key}.png")


def collect_all_stim_names(exp_info):
    """Collect all unique stimulus names across all sessions.
    Loads each beh file only once."""
    all_stim = set()
    loaded_files = set()
    for exp_type in exp_info.keys():
        beh_file = os.path.join(DATA_ROOT, 'beh', f'Beh_{exp_type}.npy')
        if beh_file in loaded_files:
            continue
        loaded_files.add(beh_file)
        if os.path.exists(beh_file):
            beh = np.load(beh_file, allow_pickle=True).item()
            for session_key, dat in beh.items():
                for wn in dat['UniqWalls']:
                    all_stim.add(str(wn))
    return sorted(list(all_stim))


def main():
    parser = argparse.ArgumentParser(description='Convert Zhong et al. 2025 data')
    parser.add_argument('output', help='Output pickle file path')
    parser.add_argument('--sample', action='store_true', help='Process only 2 sessions')
    parser.add_argument('--full', action='store_true', help='Process all sessions (default)')
    parser.add_argument('--show-processing', action='store_true', help='Plot processing steps')
    args = parser.parse_args()
    
    t_total = time.time()
    
    # Load experiment info
    print("Loading experiment info...")
    exp_info = load_exp_info()
    sessions = get_unique_sessions(exp_info)
    session_keys = sorted(sessions.keys())
    
    if args.sample:
        session_keys = session_keys[:2]
        print(f"Sample mode: processing {len(session_keys)} sessions")
    else:
        print(f"Full mode: processing {len(session_keys)} sessions")
    
    # Collect all unique stimulus names across ALL sessions (even in sample mode)
    print("Collecting stimulus names...")
    all_stim_names = collect_all_stim_names(exp_info)
    print(f"  All stimuli: {all_stim_names}")
    
    # Compute global speed quartiles
    speed_bin_edges = compute_global_speed_quartiles(exp_info)
    
    # Process sessions
    neural_all = []
    input_all = []
    output_all = []
    subjects_list = []
    subject_idx_list = []
    brain_region_idx_all = []
    
    # Track unique subjects
    unique_subjects = []
    subject_map = {}
    
    # Cache loaded behavioral data to avoid reloading
    beh_cache = {}
    
    for s_idx, session_key in enumerate(session_keys):
        sess = sessions[session_key]
        mname = sess['mname']
        
        print(f"\nProcessing session {s_idx+1}/{len(session_keys)}: {session_key}")
        
        # Track subjects
        if mname not in subject_map:
            subject_map[mname] = len(unique_subjects)
            unique_subjects.append(mname)
        
        # Process session
        result = process_session(
            session_key, sess, exp_info, speed_bin_edges,
            show_processing=args.show_processing, session_idx=s_idx
        )
        
        # Remap stimulus categories to global indices
        local_stim_names = result['stim_names']
        global_stim_map = {name: all_stim_names.index(name) for name in local_stim_names}
        
        # Update output stimulus categories to global indices
        for t in range(result['n_trials']):
            local_cat = int(result['output'][t][0, 0])  # per-trial, same across bins
            local_name = local_stim_names[local_cat]
            global_cat = global_stim_map[local_name]
            result['output'][t][0, :] = global_cat
        
        neural_all.append(result['neural'])
        input_all.append(result['input'])
        output_all.append(result['output'])
        subject_idx_list.append(subject_map[mname])
        brain_region_idx_all.append(result['region_idx'])
    
    # Build output dictionary
    brain_regions = ['V1', 'mHV', 'lHV', 'aHV']
    
    # Output names and values
    output_names = ['visual_stimulus', 'licking', 'position', 'running_speed']
    output_values = [
        [str(s) for s in all_stim_names],  # stimulus categories
        ['not_licking', 'licking'],  # binary
        ['0-1m', '1-2m', '2-3m', '3-4m'],  # position bins
        [f'Q{i+1}' for i in range(4)],  # speed quartiles
    ]
    
    input_names = ['time_to_sound_cue', 'day_of_training', 'time_since_trial_start', 'reward_availability']
    
    data = {
        'neural': neural_all,
        'input': input_all,
        'output': output_all,
        'subjects': unique_subjects,
        'subject_idx': np.array(subject_idx_list),
        'brain_regions': brain_regions,
        'brain_region_idx': brain_region_idx_all,
        'input_names': input_names,
        'output_names': output_names,
        'output_values': output_values,
        'metadata': {
            'task_description': 'Visual discrimination task in head-fixed mice running through virtual reality corridors with naturalistic texture patterns. Mice discriminate between visual stimuli (e.g., leaf vs circle) with reward delivery in one corridor type.',
            'time_bin_size': TIME_BIN_MS,
            'temporal_alignment_event': 'Trial start (corridor entry)',
            'off_start': 0.0,  # Trial starts at corridor entry
            'off_end': TOTAL_LENGTH_M / VR_SPEED,  # ~10s for full corridor + grey
            'n_position_bins': N_POS_BINS,
            'corridor_length_m': CORRIDOR_LENGTH_M,
            'grey_space_length_m': GREY_LENGTH_M,
            'vr_speed_m_per_s': VR_SPEED,
            'frame_rate_hz': FS,
            'position_bin_size_m': BIN_SIZE_M,
            'speed_bin_edges': speed_bin_edges.tolist(),
            'source': 'Zhong et al. 2025, Unsupervised pretraining in biological neural networks',
        }
    }
    
    # Save
    print(f"\nSaving to {args.output}...")
    with open(args.output, 'wb') as f:
        pickle.dump(data, f, protocol=4)
    
    file_size = os.path.getsize(args.output) / 1e9
    print(f"Saved {args.output} ({file_size:.2f} GB)")
    
    # Print summary
    print(f"\n=== Summary ===")
    print(f"Sessions: {len(neural_all)}")
    print(f"Subjects: {len(unique_subjects)}")
    print(f"Brain regions: {brain_regions}")
    print(f"Input names: {input_names}")
    print(f"Output names: {output_names}")
    print(f"Output values: {output_values}")
    total_trials = sum(len(s) for s in neural_all)
    print(f"Total trials: {total_trials}")
    neurons_per_session = [neural_all[i][0].shape[0] for i in range(len(neural_all))]
    print(f"Neurons per session: min={min(neurons_per_session)}, max={max(neurons_per_session)}, mean={np.mean(neurons_per_session):.0f}")
    print(f"Time bins per trial: {N_POS_BINS}")
    print(f"Time bin size: {TIME_BIN_MS:.2f} ms")
    print(f"Total time: {time.time()-t_total:.1f}s")


if __name__ == '__main__':
    main()
