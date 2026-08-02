"""
Convert neural data from Zhong et al. 2025 "Unsupervised pretraining in biological neural networks"
into the standardized decoder format.

Processing follows the reference paper and code:
- Neural data: deconvolved calcium traces from Suite2p, position-interpolated to 60 bins per corridor
- Only texture area bins (0-39) are used (40 bins = 4m corridor at 1 dm resolution)
- Only visual cortex neurons are included (iarea != -1 and iarea != 7)
- Only running frames are used for interpolation (VR moving, ft_move > 0)
- Brain regions: V1 (iarea=8), mHV (iarea in 0,1,2,9), lHV (iarea in 5,6), aHV (iarea in 3,4)
"""

import numpy as np
import os
import pickle
import sys
from datetime import datetime
import gc

ROOT = 'data'
N_POS_BINS = 60       # Total position bins per corridor (paper convention)
N_TEXTURE_BINS = 40   # Texture area bins (0-39, = 4m corridor)
VR_SPEED = 6.0        # dm/s (60 cm/s)
BIN_SIZE_SEC = 1.0 / VR_SPEED  # seconds per position bin = 1/6 s ≈ 0.1667s
BIN_SIZE_MS = BIN_SIZE_SEC * 1000  # ≈ 166.67 ms


def position_interpolate_spk(raw_spk, accum_pos, corridor_len, n_trials, n_bins=60):
    """Position-interpolate spike data to evenly spaced position bins.

    Matches utils.spk_pos_interp and utils.get_interpPos_spk from the paper code.
    Uses numpy.interp for speed (equivalent to scipy interp1d with extrapolation).

    Args:
        raw_spk: (n_neurons, n_frames) - neural data for running frames only
        accum_pos: (n_frames,) - cumulative position for running frames
        corridor_len: corridor length in dm
        n_trials: number of trials
        n_bins: bins per corridor

    Returns:
        interp_spk: (n_neurons, n_trials, n_bins)
    """
    lin_pos = np.arange(0, n_trials, 1.0 / n_bins)
    norm_pos = accum_pos / corridor_len
    n_neurons = raw_spk.shape[0]
    n_target = len(lin_pos)
    interp_spk = np.zeros((n_neurons, n_target), dtype=np.float32)

    # Use numpy.interp which is much faster than scipy for 1D interpolation
    for s in range(n_neurons):
        interp_spk[s, :] = np.interp(lin_pos, norm_pos, raw_spk[s, :])

    return interp_spk.reshape(n_neurons, n_trials, n_bins)


def get_brain_region_idx(iarea):
    """Map iarea values to brain region indices.

    Brain regions (from paper's neu_area_ID):
        V1: iarea == 8
        mHV: iarea in {0, 1, 2, 9}
        lHV: iarea in {5, 6}
        aHV: iarea in {3, 4}

    Returns:
        region_idx: array of region indices (0=V1, 1=mHV, 2=lHV, 3=aHV)
        valid_mask: boolean mask for neurons in visual cortex
    """
    valid_mask = (iarea != -1) & (iarea != 7)
    region_idx = np.full(len(iarea), -1, dtype=int)
    region_idx[iarea == 8] = 0  # V1
    region_idx[(iarea == 0) | (iarea == 1) | (iarea == 2) | (iarea == 9)] = 1  # mHV
    region_idx[(iarea == 5) | (iarea == 6)] = 2  # lHV
    region_idx[(iarea == 3) | (iarea == 4)] = 3  # aHV
    return region_idx, valid_mask


def lick_to_position_bins(lick_pos, lick_trind, n_trials, n_bins=40):
    """Convert lick events to binary per-position-bin per-trial.

    Args:
        lick_pos: position of each lick event (in dm)
        lick_trind: trial index of each lick event
        n_trials: number of trials
        n_bins: number of position bins (texture area)

    Returns:
        lick_binary: (n_trials, n_bins) binary array
    """
    lick_binary = np.zeros((n_trials, n_bins), dtype=int)
    for i in range(len(lick_pos)):
        tr = int(lick_trind[i])
        pos = lick_pos[i]
        bin_idx = int(pos)
        if 0 <= bin_idx < n_bins and 0 <= tr < n_trials:
            lick_binary[tr, bin_idx] = 1
    return lick_binary


def compute_day_of_training(date_str):
    """Parse date string like '2022_08_17' to datetime."""
    return datetime.strptime(date_str, '%Y_%m_%d')


def collect_all_sessions(exp_info, root):
    """Collect all unique recording sessions with their metadata.

    For recordings appearing in multiple experiment types, use the first one found.
    Returns list of session dicts with all needed info.
    """
    seen_recordings = set()
    sessions = []

    # Process experiment types in a specific order to prefer more complete annotations
    # Prefer test sessions (more stimuli) over train sessions (fewer stimuli)
    exp_order = sorted(exp_info.keys(), key=lambda x: (
        0 if 'test' in x else 1,  # test first
        0 if 'sup_' in x else 1,  # supervised first (has rewards)
        0 if 'after' in x else 1,  # after learning first
        x  # alphabetical tiebreak
    ))

    for exp_type in exp_order:
        db_list = exp_info[exp_type]
        # Load behavior data
        beh_path = os.path.join(root, 'beh', f'Beh_{exp_type}.npy')
        if not os.path.exists(beh_path):
            print(f"  Warning: behavior file not found for {exp_type}")
            continue
        beh_data = np.load(beh_path, allow_pickle=True).item()

        for ndb in db_list:
            rec_key = (ndb['mname'], ndb['datexp'], ndb['blk'])
            if rec_key in seen_recordings:
                continue
            seen_recordings.add(rec_key)

            # Build behavior key
            stimtype = ndb.get('stimtype', '')
            if stimtype:
                beh_key = f"{ndb['mname']}_{ndb['datexp']}_{ndb['blk']}_{stimtype}"
            else:
                beh_key = f"{ndb['mname']}_{ndb['datexp']}_{ndb['blk']}"

            if beh_key not in beh_data:
                print(f"  Warning: behavior key {beh_key} not found in {exp_type}")
                continue

            sessions.append({
                'mname': ndb['mname'],
                'datexp': ndb['datexp'],
                'blk': ndb['blk'],
                'exp_type': exp_type,
                'beh_key': beh_key,
                'beh': beh_data[beh_key],
                'ndb': ndb,
            })

    return sessions


def process_session(session_info, root, all_stim_names, speed_quantile_edges):
    """Process a single recording session into decoder format.

    Args:
        session_info: dict with session metadata and behavior
        root: data root directory
        all_stim_names: list of all unique stimulus names (for consistent labeling)
        speed_quantile_edges: pre-computed quartile edges for running speed

    Returns:
        dict with neural, input, output, brain_region_idx for this session
    """
    ndb = session_info['ndb']
    beh = session_info['beh']
    mname = ndb['mname']
    datexp = ndb['datexp']
    blk = ndb['blk']

    # Load neural data
    spk_fn = f"{mname}_{datexp}_{blk}_neural_data.npy"
    spk_data = np.load(os.path.join(root, 'spk', spk_fn), allow_pickle=True).item()
    spk = np.concatenate([s for s in spk_data['spks']], axis=0)
    del spk_data
    n_neurons, n_frames = spk.shape

    # Load retinotopy
    ret_fn = f"{mname}_{datexp}_trans.npz"
    ret_data = np.load(os.path.join(root, 'retinotopy', ret_fn), allow_pickle=True)
    iarea = ret_data['iarea']

    # Get valid neuron mask (visual cortex only)
    region_idx, valid_mask = get_brain_region_idx(iarea)
    valid_neurons = valid_mask

    # Get behavior variables
    n_trials = beh['ntrials']
    corridor_len = beh['Corridor_Length']  # typically 60 dm

    # Filter for running frames (VR moving)
    ft_move = beh['ft_move'][:n_frames]
    vr_moving = ft_move > 0
    ft_pos_cum = beh['ft_PosCum'][:n_frames]

    # Position-interpolate neural data (matching paper's processing)
    # Only pass valid (visual cortex) neurons and running frames
    valid_spk = spk[valid_neurons][:, vr_moving]
    del spk  # free memory
    interp_spk = position_interpolate_spk(
        valid_spk,
        ft_pos_cum[vr_moving],
        corridor_len,
        n_trials,
        n_bins=N_POS_BINS
    )
    del valid_spk  # free memory
    # Keep only texture area (first 40 bins)
    interp_spk = interp_spk[:, :, :N_TEXTURE_BINS]

    # Get running speed per position bin (from behavior data)
    run_pos = beh['run_pos']  # (n_trials, 60) - already position-interpolated
    run_speed = run_pos[:, :N_TEXTURE_BINS]  # (n_trials, 40)

    # Discretize running speed into 4 quartile bins
    # np.digitize with 3 edges (Q25, Q50, Q75) returns 0-3 directly
    speed_bins = np.digitize(run_speed, speed_quantile_edges)
    speed_bins = np.clip(speed_bins, 0, 3)

    # Get lick data per position bin
    lick_pos = beh['LickPos']
    lick_trind = beh['LickTrind']
    lick_binary = lick_to_position_bins(lick_pos, lick_trind, n_trials, N_TEXTURE_BINS)

    # Get stimulus names per trial
    wall_names = beh['WallName']

    # Map stimulus names to indices
    stim_indices = np.array([all_stim_names.index(wn) for wn in wall_names])

    # Get sound cue position (in dm within corridor)
    sound_pos = beh['SoundPos']  # position where sound cue was delivered

    # Reward availability
    is_rew = beh['isRew'].astype(float)

    # Position bins for corridor (4 equal 1m bins)
    pos_bins = np.zeros(N_TEXTURE_BINS, dtype=int)
    for i in range(N_TEXTURE_BINS):
        pos_bins[i] = min(i // 10, 3)  # 0-9 -> 0, 10-19 -> 1, 20-29 -> 2, 30-39 -> 3

    # Compute day of training for this session
    session_date = compute_day_of_training(datexp)

    # Build per-trial data
    neural_trials = []
    input_trials = []
    output_trials = []

    for tr in range(n_trials):
        # Neural: (n_valid_neurons, 40)
        neural_trial = interp_spk[:, tr, :].astype(np.float32)

        # Input: (4, 40)
        # 1. Time to sound cue (seconds) - negative before cue, positive after
        time_to_cue = np.array([(i - sound_pos[tr]) * BIN_SIZE_SEC
                                 for i in range(N_TEXTURE_BINS)], dtype=np.float32)

        # 2. Day of training - broadcast to all time bins
        # Will be filled in later with per-mouse relative days
        day_val = 0.0  # placeholder
        day_of_training = np.full(N_TEXTURE_BINS, day_val, dtype=np.float32)

        # 3. Time since trial start (seconds)
        time_since_start = np.array([i * BIN_SIZE_SEC for i in range(N_TEXTURE_BINS)],
                                     dtype=np.float32)

        # 4. Reward availability
        reward_avail = np.full(N_TEXTURE_BINS, is_rew[tr], dtype=np.float32)

        input_trial = np.stack([time_to_cue, day_of_training, time_since_start, reward_avail],
                                axis=0)  # (4, 40)

        # Output: (4, 40) - integer categorical values
        # 1. Visual stimulus category - broadcast to all time bins
        stim_cat = np.full(N_TEXTURE_BINS, stim_indices[tr], dtype=np.int64)

        # 2. Licking binary
        lick_tr = lick_binary[tr, :].astype(np.int64)

        # 3. Position bin
        pos_tr = pos_bins.astype(np.int64)

        # 4. Running speed bin
        speed_tr = speed_bins[tr, :].astype(np.int64)

        output_trial = np.stack([stim_cat, lick_tr, pos_tr, speed_tr], axis=0)  # (4, 40)

        neural_trials.append(neural_trial)
        input_trials.append(input_trial)
        output_trials.append(output_trial)

    return {
        'neural': neural_trials,
        'input': input_trials,
        'output': output_trials,
        'brain_region_idx': region_idx[valid_mask].astype(int),
        'n_neurons': valid_mask.sum(),
        'n_trials': n_trials,
        'mname': mname,
        'datexp': datexp,
        'session_date': session_date,
    }


def main(sample_only=False):
    print("Loading experiment info...")
    exp_info = np.load(os.path.join(ROOT, 'beh', 'Imaging_Exp_info.npy'),
                       allow_pickle=True).item()

    # Collect all unique sessions
    print("Collecting sessions...")
    sessions = collect_all_sessions(exp_info, ROOT)
    print(f"Found {len(sessions)} unique recording sessions")

    # Collect all unique stimulus names across all sessions
    all_stim_set = set()
    for s in sessions:
        for wn in s['beh']['UniqWalls']:
            all_stim_set.add(str(wn))
    all_stim_names = sorted(all_stim_set)
    print(f"Unique stimuli: {all_stim_names}")

    # Collect all unique subjects
    all_subjects = sorted(set(s['mname'] for s in sessions))
    print(f"Unique subjects: {all_subjects}")

    # Compute running speed quantile edges across all sessions
    print("Computing running speed quantiles...")
    all_speeds = []
    for s in sessions:
        run_pos = s['beh']['run_pos'][:, :N_TEXTURE_BINS]
        all_speeds.append(run_pos.ravel())
    all_speeds = np.concatenate(all_speeds)
    speed_quantile_edges = np.percentile(all_speeds, [25, 50, 75])
    print(f"Running speed quartile edges: {speed_quantile_edges}")
    del all_speeds

    # Compute day of training for each session (relative to first recording per mouse)
    mouse_first_date = {}
    for s in sessions:
        d = compute_day_of_training(s['datexp'])
        if s['mname'] not in mouse_first_date or d < mouse_first_date[s['mname']]:
            mouse_first_date[s['mname']] = d

    session_days = {}
    for s in sessions:
        d = compute_day_of_training(s['datexp'])
        days = (d - mouse_first_date[s['mname']]).days
        key = (s['mname'], s['datexp'], s['blk'])
        session_days[key] = float(days)

    if sample_only:
        # Use first 3 sessions for sample
        sessions = sessions[:3]
        print(f"Sample mode: using {len(sessions)} sessions")

    # Process each session
    brain_regions = ['V1', 'mHV', 'lHV', 'aHV']

    all_neural = []
    all_input = []
    all_output = []
    all_brain_region_idx = []
    all_subject_idx = []
    session_info_list = []

    for i, s in enumerate(sessions):
        rec_key = (s['mname'], s['datexp'], s['blk'])
        print(f"\nProcessing session {i+1}/{len(sessions)}: {rec_key} ({s['exp_type']})")

        try:
            result = process_session(s, ROOT, all_stim_names, speed_quantile_edges)
        except Exception as e:
            print(f"  ERROR processing session: {e}")
            import traceback
            traceback.print_exc()
            continue

        # Fill in day_of_training
        day_val = session_days[rec_key]
        for trial_input in result['input']:
            trial_input[1, :] = day_val  # row 1 = day_of_training

        # Check minimum trials
        if result['n_trials'] < 2:
            print(f"  Skipping: only {result['n_trials']} trials")
            continue

        all_neural.append(result['neural'])
        all_input.append(result['input'])
        all_output.append(result['output'])
        all_brain_region_idx.append(result['brain_region_idx'])
        all_subject_idx.append(all_subjects.index(result['mname']))

        session_info_list.append({
            'mname': s['mname'],
            'datexp': s['datexp'],
            'blk': s['blk'],
            'exp_type': s['exp_type'],
            'n_neurons': result['n_neurons'],
            'n_trials': result['n_trials'],
            'day_of_training': day_val,
        })

        print(f"  Neurons: {result['n_neurons']}, Trials: {result['n_trials']}, "
              f"Day: {day_val}")
        gc.collect()

    print(f"\n{'='*60}")
    print(f"Total sessions processed: {len(all_neural)}")
    print(f"Total subjects: {len(all_subjects)}")

    # Build output_values
    output_values = [
        all_stim_names,  # visual stimulus categories
        ['no_lick', 'lick'],  # licking
        ['0-1m', '1-2m', '2-3m', '3-4m'],  # position bins
        ['Q1', 'Q2', 'Q3', 'Q4'],  # running speed quartiles
    ]

    # Build the data dict
    data = {
        'neural': all_neural,
        'input': all_input,
        'output': all_output,

        'subjects': all_subjects,
        'subject_idx': np.array(all_subject_idx, dtype=int),

        'brain_regions': brain_regions,
        'brain_region_idx': all_brain_region_idx,

        'input_names': ['time_to_sound_cue', 'day_of_training', 'time_since_trial_start',
                        'reward_availability'],
        'output_names': ['visual_stimulus', 'licking', 'position', 'running_speed'],
        'output_values': output_values,

        'metadata': {
            'task_description': ('Visual discrimination task in head-fixed mice running through '
                                 'linear virtual reality corridors. Mice discriminate between '
                                 'visual texture patterns. Sound cue indicates reward zone in '
                                 'rewarded corridor.'),
            'time_bin_size': BIN_SIZE_MS,
            'temporal_alignment_event': 'Trial start (corridor entry)',
            'off_start': 0.0,
            'off_end': N_TEXTURE_BINS * BIN_SIZE_SEC,
            'vr_speed_cm_s': 60.0,
            'corridor_length_m': 4.0,
            'position_bin_size_dm': 1.0,
            'n_position_bins': N_TEXTURE_BINS,
            'calcium_indicator': 'GCaMP6s',
            'neural_data_type': 'deconvolved fluorescence (Suite2p)',
            'frame_rate_hz': 3.17,
            'speed_quantile_edges': speed_quantile_edges.tolist(),
            'stimulus_names': all_stim_names,
            'session_info': session_info_list,
            'n_mice': len(all_subjects),
            'n_sessions': len(all_neural),
            'reference': 'Zhong et al. 2025, Unsupervised pretraining in biological neural networks',
        }
    }

    return data


if __name__ == '__main__':
    sample_mode = '--sample' in sys.argv

    data = main(sample_only=sample_mode)

    if sample_mode:
        out_path = 'sample_data.pkl'
    else:
        out_path = 'converted_data.pkl'

    print(f"\nSaving to {out_path}...")
    with open(out_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved successfully.")

    # Print summary statistics
    n_sessions = len(data['neural'])
    total_trials = sum(len(data['neural'][i]) for i in range(n_sessions))
    total_neurons = sum(data['neural'][i][0].shape[0] for i in range(n_sessions) if len(data['neural'][i]) > 0)
    print(f"\nSummary:")
    print(f"  Sessions: {n_sessions}")
    print(f"  Total trials: {total_trials}")
    print(f"  Total neurons (across sessions): {total_neurons}")
    print(f"  Subjects: {len(data['subjects'])}")
    print(f"  Brain regions: {data['brain_regions']}")
    print(f"  Input variables: {data['input_names']}")
    print(f"  Output variables: {data['output_names']}")
    print(f"  Time bin size: {data['metadata']['time_bin_size']:.2f} ms")
    print(f"  Timepoints per trial: {N_TEXTURE_BINS}")
