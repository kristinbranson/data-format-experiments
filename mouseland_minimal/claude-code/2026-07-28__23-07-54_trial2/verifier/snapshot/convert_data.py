"""
Convert neural and behavioral data from Zhong et al. 2025
("Unsupervised pretraining in biological neural networks")
into the standardized decoder format.

Data structure:
- Neural: Deconvolved calcium traces (Suite2p), sampled at ~3.17 Hz
- Behavior: VR corridor task with visual stimuli, licking, rewards
- Retinotopy: Visual area assignments for each neuron

Processing:
- Align to trial start (corridor entry)
- Fixed-length time windows per trial
- Only include running frames (VR moving) per paper methods
- Brain regions: V1, mHV, lHV, aHV (from retinotopy)
- Exclude neurons outside visual cortex (iarea == -1 or 7)
"""

import numpy as np
import pickle
import os
import sys
from collections import defaultdict

# ============ CONFIGURATION ============
DATA_ROOT = 'data'
FS = 3.17  # Calcium imaging frame rate in Hz
TIME_BIN_MS = 1000.0 / FS  # ~315.5 ms per frame
CORRIDOR_LENGTH_DM = 60.0  # Total corridor length in decimeters (6m)
TEXTURE_LENGTH_DM = 40.0   # Texture area (4m)
GRAY_LENGTH_DM = 20.0      # Gray space (2m)
VR_SPEED_CM_S = 60.0       # VR constant speed when running

# Trial window: use a fixed number of frames from corridor entry
# At 60 cm/s and 3.17 Hz: 6m corridor takes ~32 frames
# Use 32 frames to capture most of the corridor traversal
N_TIMEPOINTS = 32

# Minimum trials per session for decoder evaluation
MIN_TRIALS = 10

# stim_id mapping from the paper
STIM_NAMES = {
    0: 'circle1',
    1: 'circle2',
    2: 'leaf1',
    3: 'leaf2',
    4: 'leaf3',
    5: 'leaf1_swap1',
    6: 'leaf1_swap2',
}

# Brain region definitions from utils.py neu_area_ID
BRAIN_REGIONS = ['V1', 'mHV', 'lHV', 'aHV']
def get_brain_region_idx(iarea):
    """Map iarea values to brain region indices. Returns -1 for excluded neurons."""
    region_idx = np.full(len(iarea), -1, dtype=int)
    region_idx[iarea == 8] = 0  # V1
    region_idx[(iarea == 0) | (iarea == 1) | (iarea == 2) | (iarea == 9)] = 1  # mHV
    region_idx[(iarea == 5) | (iarea == 6)] = 2  # lHV
    region_idx[(iarea == 3) | (iarea == 4)] = 3  # aHV
    return region_idx


def load_spk(mname, datexp, blk, root):
    """Load spike data (deconvolved fluorescence) for a session."""
    fn = f'{mname}_{datexp}_{blk}_neural_data.npy'
    spk_path = os.path.join(root, 'spk', fn)
    spk_data = np.load(spk_path, allow_pickle=True).item()
    spk = np.concatenate([s for s in spk_data['spks']], axis=0)
    return spk


def load_retino(mname, datexp, root):
    """Load retinotopy data for a session."""
    fn = f'{mname}_{datexp}_trans.npz'
    ret_path = os.path.join(root, 'retinotopy', fn)
    dtrans = np.load(ret_path, allow_pickle=True)
    return dtrans['iarea']


def get_session_list():
    """Build the list of unique sessions to process, picking the best behavior file for each."""
    exp_info = np.load(os.path.join(DATA_ROOT, 'beh', 'Imaging_Exp_info.npy'), allow_pickle=True).item()

    # For each unique neural session, find the best behavior data (most stimuli)
    sessions = {}
    for exp_type in exp_info:
        for db in exp_info[exp_type]:
            key = f'{db["mname"]}_{db["datexp"]}_{db["blk"]}'
            stimtype = db.get('stimtype', '')
            beh_key = key if not stimtype else f'{key}_{stimtype}'
            stim_id = db.get('stim_id', np.array([]))
            n_stim = int(np.sum(~np.isnan(stim_id.astype(float))))

            if key not in sessions or n_stim > sessions[key]['n_stim']:
                sessions[key] = {
                    'mname': db['mname'],
                    'datexp': db['datexp'],
                    'blk': db['blk'],
                    'exp_type': exp_type,
                    'beh_key': beh_key,
                    'n_stim': n_stim,
                    'stim_id': stim_id,
                    'exptype': db.get('exptype', ''),
                    'rewType': db.get('rewType', 'None'),
                    'days': db.get('days', db.get('sess#', 0)),
                }

    return sessions


def compute_day_of_training(mname, datexp, all_sessions):
    """Compute day of training relative to first session for this mouse."""
    from datetime import datetime
    mouse_dates = []
    for k, s in all_sessions.items():
        if s['mname'] == mname:
            date = datetime.strptime(s['datexp'], '%Y_%m_%d')
            mouse_dates.append(date)
    mouse_dates.sort()
    current_date = datetime.strptime(datexp, '%Y_%m_%d')
    day_idx = (current_date - mouse_dates[0]).days
    return day_idx


def discretize_speed(speed_values, n_bins=4):
    """Discretize running speed into n_bins using quartile boundaries computed per session."""
    # Compute quartile boundaries from all valid (non-nan) speed values
    valid = speed_values[~np.isnan(speed_values)]
    if len(valid) == 0:
        return np.zeros_like(speed_values, dtype=int)

    percentiles = np.linspace(0, 100, n_bins + 1)[1:-1]  # e.g., [25, 50, 75]
    boundaries = np.percentile(valid, percentiles)

    # Digitize
    binned = np.digitize(speed_values, boundaries)  # 0 to n_bins-1
    return binned


def discretize_position(pos_values, n_bins=4):
    """Discretize position into 4 equal-length 1-m bins (0-1m, 1-2m, 2-3m, 3-4m).
    Position is in decimeters. Corridor texture area is 0-40dm = 0-4m.
    Values in gray space (>40dm) get assigned to the last bin.
    """
    # 4 bins of 1m each = 10 dm each: [0,10), [10,20), [20,30), [30,40+)
    boundaries = [10, 20, 30]  # in decimeters
    binned = np.digitize(pos_values, boundaries)  # 0,1,2,3
    return binned


def extract_trial_data(spk, beh, n_timepoints, session_speed_all=None):
    """Extract trial-aligned neural and behavioral data.

    Returns:
        neural_trials: list of (n_neurons, n_timepoints) arrays
        input_trials: list of (n_input, n_timepoints) or (n_input,) arrays
        output_trials: list of (n_output, n_timepoints) or (n_output,) arrays
        valid_trials: boolean mask of valid trials
    """
    ntrials = beh['ntrials']
    n_neurons = spk.shape[0]
    n_frames = spk.shape[1]

    start_frs = beh['StartFr']
    sound_frs = beh['SoundFr']
    wall_names = beh['WallName']
    is_rew = beh['isRew']

    # Frame-level data
    ft_pos = beh['ft_Pos'][:n_frames]
    ft_move = beh['ft_move'][:n_frames]
    ft_speed = beh['ft_RunSpeed'][:n_frames]

    # Lick data
    lick_frs = beh['LickFr']
    lick_trinds = beh['LickTrind']

    # Get stim_id mapping if available
    stim_id = beh.get('stim_id', None)
    uniq_walls = beh['UniqWalls']

    neural_trials = []
    input_trials = []
    output_trials = []
    valid_mask = []
    stim_categories = []

    for trial in range(ntrials):
        start_fr = int(np.round(start_frs[trial]))
        end_fr = start_fr + n_timepoints

        # Check bounds
        if start_fr < 0 or end_fr > n_frames:
            valid_mask.append(False)
            neural_trials.append(None)
            input_trials.append(None)
            output_trials.append(None)
            stim_categories.append('')
            continue

        # ---- NEURAL DATA ----
        trial_neural = spk[:, start_fr:end_fr].astype(np.float32)  # (n_neurons, n_timepoints)

        # ---- INPUTS ----
        # 1. Time to sound cue (continuous, time-varying)
        # Sound cue frame relative to trial start
        sound_fr_rel = sound_frs[trial] - start_frs[trial]
        time_to_cue = np.arange(n_timepoints) - sound_fr_rel  # negative before cue, positive after
        time_to_cue_sec = time_to_cue / FS  # convert to seconds

        # 2. Day of training (continuous, constant per trial)
        # Will be filled in later at session level
        day_of_training = np.zeros(n_timepoints)  # placeholder

        # 3. Time since trial start (continuous, time-varying)
        time_since_start = np.arange(n_timepoints) / FS  # in seconds

        # 4. Reward availability (discrete, per trial)
        # 1 if rewarded corridor, 0 if not
        reward_avail = np.full(n_timepoints, float(is_rew[trial]))

        trial_input = np.stack([time_to_cue_sec, day_of_training, time_since_start, reward_avail])
        # shape: (4, n_timepoints)

        # ---- OUTPUTS ----
        # 1. Visual stimulus category (per-trial, categorical)
        # Map wall name to stim category name
        wall_name = wall_names[trial]
        stim_cat = wall_name  # e.g., 'circle1', 'leaf2', etc.
        stim_categories.append(stim_cat)

        # 2. Licking (binary, time-varying)
        # Check which frames in this trial window have licks
        lick_binary = np.zeros(n_timepoints, dtype=float)
        trial_lick_mask = lick_trinds == trial
        if trial_lick_mask.any():
            trial_lick_frs = lick_frs[trial_lick_mask]
            for lf in trial_lick_frs:
                fr_idx = int(np.round(lf)) - start_fr
                if 0 <= fr_idx < n_timepoints:
                    lick_binary[fr_idx] = 1.0

        # 3. Position in corridor (discretized into 4 bins, time-varying)
        trial_pos = ft_pos[start_fr:end_fr]
        pos_binned = discretize_position(trial_pos)

        # 4. Running speed (discretized into 4 bins, time-varying)
        trial_speed = ft_speed[start_fr:end_fr]
        speed_binned = discretize_speed(trial_speed, n_bins=4)

        trial_output = np.stack([
            np.zeros(n_timepoints, dtype=int),  # placeholder for stim category (per-trial, filled later)
            lick_binary.astype(int),
            pos_binned.astype(int),
            speed_binned.astype(int),
        ])
        # shape: (4, n_timepoints), integer values

        neural_trials.append(trial_neural)
        input_trials.append(trial_input)
        output_trials.append(trial_output)
        valid_mask.append(True)

    return neural_trials, input_trials, output_trials, valid_mask, stim_categories


def process_all_sessions(sample_only=False):
    """Process all sessions and build the decoder data dict."""
    sessions = get_session_list()
    exp_info = np.load(os.path.join(DATA_ROOT, 'beh', 'Imaging_Exp_info.npy'), allow_pickle=True).item()

    # Collect all unique subjects
    all_subjects = sorted(set(s['mname'] for s in sessions.values()))
    subject_to_idx = {s: i for i, s in enumerate(all_subjects)}

    # Collect all unique stimulus names across all sessions
    all_stim_names = set()

    # First pass: determine stimulus categories
    beh_cache = {}
    for key, sess_info in sessions.items():
        exp_type = sess_info['exp_type']
        if exp_type not in beh_cache:
            beh_cache[exp_type] = np.load(
                os.path.join(DATA_ROOT, 'beh', f'Beh_{exp_type}.npy'),
                allow_pickle=True
            ).item()
        beh_key = sess_info['beh_key']
        if beh_key in beh_cache[exp_type]:
            beh = beh_cache[exp_type][beh_key]
            for w in beh['UniqWalls']:
                all_stim_names.add(w)

    # Sort stimulus names for consistency, convert to plain Python strings
    all_stim_sorted = sorted(str(s) for s in all_stim_names)
    stim_to_idx = {}
    for exp_type in beh_cache:
        for beh_key_inner in beh_cache[exp_type]:
            for w in beh_cache[exp_type][beh_key_inner]['UniqWalls']:
                stim_to_idx[str(w)] = all_stim_sorted.index(str(w))
    print(f"All stimulus categories: {all_stim_sorted}")

    # Compute speed quartile boundaries across ALL sessions for consistent discretization
    print("Computing global speed quartiles...")
    all_speeds = []
    for key, sess_info in sessions.items():
        exp_type = sess_info['exp_type']
        beh_key = sess_info['beh_key']
        if beh_key in beh_cache[exp_type]:
            beh = beh_cache[exp_type][beh_key]
            n_frames = len(beh['ft_RunSpeed'])
            speeds = beh['ft_RunSpeed']
            valid_speed = speeds[~np.isnan(speeds)]
            all_speeds.append(valid_speed)
    all_speeds = np.concatenate(all_speeds)
    speed_quartiles = np.percentile(all_speeds, [25, 50, 75])
    print(f"Global speed quartile boundaries: {speed_quartiles}")
    del all_speeds

    # Process each session
    neural_all = []
    input_all = []
    output_all = []
    subject_idx_all = []
    brain_region_idx_all = []
    session_info_list = []

    sorted_keys = sorted(sessions.keys())
    if sample_only:
        # For sample, pick sessions that include supervised (with rewards/licking)
        # and unsupervised sessions for diversity
        sample_keys = []
        # Pick one supervised and two unsupervised sessions from different mice
        sup_keys = [k for k in sorted_keys if sessions[k].get('rewType', 'None') not in ('None', 'Unknown')]
        unsup_keys = [k for k in sorted_keys if sessions[k].get('rewType', 'None') in ('None', 'Unknown')]
        seen_mice = set()
        for k in sup_keys:
            if sessions[k]['mname'] not in seen_mice:
                sample_keys.append(k)
                seen_mice.add(sessions[k]['mname'])
                if len(sample_keys) >= 2:
                    break
        for k in unsup_keys:
            if sessions[k]['mname'] not in seen_mice:
                sample_keys.append(k)
                seen_mice.add(sessions[k]['mname'])
                if len(sample_keys) >= 3:
                    break
        sorted_keys = sample_keys
        print(f"Sample mode: processing {len(sorted_keys)} sessions")

    for sess_idx, key in enumerate(sorted_keys):
        sess_info = sessions[key]
        mname = sess_info['mname']
        datexp = sess_info['datexp']
        blk = sess_info['blk']
        exp_type = sess_info['exp_type']
        beh_key = sess_info['beh_key']

        print(f"\n[{sess_idx+1}/{len(sorted_keys)}] Processing {key} (exp: {exp_type})")

        # Load behavior
        if beh_key not in beh_cache[exp_type]:
            print(f"  WARNING: beh_key {beh_key} not in {exp_type}, skipping")
            continue
        beh = beh_cache[exp_type][beh_key]

        if beh['ntrials'] < MIN_TRIALS:
            print(f"  Skipping: only {beh['ntrials']} trials (< {MIN_TRIALS})")
            continue

        # Load neural data
        try:
            spk = load_spk(mname, datexp, blk, DATA_ROOT)
        except Exception as e:
            print(f"  ERROR loading spk: {e}")
            continue

        n_neurons, n_frames = spk.shape
        print(f"  Neurons: {n_neurons}, Frames: {n_frames}, Trials: {beh['ntrials']}")

        # Load retinotopy and get brain region assignments
        try:
            iarea = load_retino(mname, datexp, DATA_ROOT)
        except Exception as e:
            print(f"  ERROR loading retinotopy: {e}")
            continue

        region_idx = get_brain_region_idx(iarea)

        # Filter: only keep neurons in visual cortex (V1, mHV, lHV, aHV)
        valid_neurons = region_idx >= 0  # Exclude iarea -1 and 7

        if valid_neurons.sum() < 10:
            print(f"  Skipping: only {valid_neurons.sum()} visual cortex neurons")
            continue

        spk_filtered = spk[valid_neurons]
        region_idx_filtered = region_idx[valid_neurons]
        n_neurons_filtered = spk_filtered.shape[0]
        print(f"  Visual cortex neurons: {n_neurons_filtered}/{n_neurons}")

        # Compute day of training
        day = compute_day_of_training(mname, datexp, sessions)

        # Extract trial data
        neural_trials, input_trials, output_trials, valid_mask, stim_cats = extract_trial_data(
            spk_filtered, beh, N_TIMEPOINTS
        )

        # Fill in day of training for inputs
        for i in range(len(input_trials)):
            if valid_mask[i]:
                input_trials[i][1, :] = float(day)  # day_of_training row

        # Discretize speed using global quartiles
        for i in range(len(output_trials)):
            if valid_mask[i]:
                # Re-discretize speed with global quartiles
                start_fr = int(np.round(beh['StartFr'][i]))
                end_fr = start_fr + N_TIMEPOINTS
                if end_fr <= len(beh['ft_RunSpeed']):
                    trial_speed = beh['ft_RunSpeed'][start_fr:end_fr]
                    speed_binned = np.digitize(trial_speed, speed_quartiles)  # 0-3
                    output_trials[i][3, :] = speed_binned.astype(int)

        # Map stimulus categories to indices
        for i in range(len(output_trials)):
            if valid_mask[i]:
                stim_idx = stim_to_idx.get(str(stim_cats[i]), 0)
                output_trials[i][0, :] = stim_idx  # constant across time

        # Filter to valid trials only
        valid_neural = [neural_trials[i] for i in range(len(valid_mask)) if valid_mask[i]]
        valid_input = [input_trials[i] for i in range(len(valid_mask)) if valid_mask[i]]
        valid_output = [output_trials[i] for i in range(len(valid_mask)) if valid_mask[i]]

        if len(valid_neural) < MIN_TRIALS:
            print(f"  Skipping: only {len(valid_neural)} valid trials after filtering")
            continue

        print(f"  Valid trials: {len(valid_neural)}/{beh['ntrials']}")

        neural_all.append(valid_neural)
        input_all.append(valid_input)
        output_all.append(valid_output)
        subject_idx_all.append(subject_to_idx[mname])
        brain_region_idx_all.append(region_idx_filtered)
        session_info_list.append({
            'key': key,
            'mname': mname,
            'datexp': datexp,
            'blk': blk,
            'exp_type': exp_type,
            'exptype': sess_info['exptype'],
            'rewType': sess_info['rewType'],
            'n_trials': len(valid_neural),
            'n_neurons': n_neurons_filtered,
            'day': day,
        })

        # Free memory
        del spk, spk_filtered

    # Build output names and values
    # Output 0: visual_stimulus (per-trial categorical)
    # Output 1: licking (binary, time-varying)
    # Output 2: position (4 bins, time-varying)
    # Output 3: running_speed (4 bins, time-varying)

    output_names = ['visual_stimulus', 'licking', 'position', 'running_speed']
    output_values = [
        [str(s) for s in all_stim_sorted],  # stimulus category names (plain strings)
        ['no_lick', 'lick'],     # licking
        ['0-1m', '1-2m', '2-3m', '3-4m'],  # position bins (4m texture corridor)
        ['Q1', 'Q2', 'Q3', 'Q4'],  # speed quartiles
    ]

    input_names = ['time_to_sound_cue', 'day_of_training', 'time_since_trial_start', 'reward_availability']

    data = {
        'neural': neural_all,
        'input': input_all,
        'output': output_all,
        'subjects': all_subjects,
        'subject_idx': np.array(subject_idx_all),
        'brain_regions': BRAIN_REGIONS,
        'brain_region_idx': brain_region_idx_all,
        'input_names': input_names,
        'output_names': output_names,
        'output_values': output_values,
        'metadata': {
            'task_description': 'Visual discrimination task in head-fixed mice running through virtual reality corridors. Mice discriminate visual texture patterns (leaf/circle) across corridors. Sound cue indicates reward availability in rewarded corridor.',
            'time_bin_size': TIME_BIN_MS,
            'temporal_alignment_event': 'Trial start (corridor entry)',
            'off_start': 0.0,
            'off_end': N_TIMEPOINTS / FS,
            'n_timepoints': N_TIMEPOINTS,
            'frame_rate_hz': FS,
            'corridor_length_m': 6.0,
            'texture_length_m': 4.0,
            'gray_space_m': 2.0,
            'vr_speed_cm_s': VR_SPEED_CM_S,
            'speed_quartile_boundaries': speed_quartiles.tolist(),
            'session_info': session_info_list,
            'paper': 'Zhong et al. 2025, Unsupervised pretraining in biological neural networks',
            'neural_data_type': 'Deconvolved calcium fluorescence traces (Suite2p)',
        },
    }

    # Sanity checks
    n_sessions = len(neural_all)
    print(f"\n{'='*60}")
    print(f"CONVERSION SUMMARY")
    print(f"{'='*60}")
    print(f"Total sessions: {n_sessions}")
    print(f"Total subjects: {len(all_subjects)}")
    print(f"Brain regions: {BRAIN_REGIONS}")
    print(f"Time bin size: {TIME_BIN_MS:.2f} ms")
    print(f"Timepoints per trial: {N_TIMEPOINTS}")
    print(f"Input variables: {input_names}")
    print(f"Output variables: {output_names}")
    print(f"Stimulus categories: {all_stim_sorted}")

    total_trials = sum(len(s) for s in neural_all)
    print(f"Total trials: {total_trials}")

    neuron_counts = [s[0].shape[0] for s in neural_all if len(s) > 0]
    print(f"Neurons per session: min={min(neuron_counts)}, max={max(neuron_counts)}, median={np.median(neuron_counts):.0f}")

    # Verify dimensions
    for i in range(n_sessions):
        n_trials = len(neural_all[i])
        for j in range(min(n_trials, 3)):
            assert neural_all[i][j].shape == (neuron_counts[i], N_TIMEPOINTS), \
                f"Session {i}, trial {j}: neural shape {neural_all[i][j].shape} != ({neuron_counts[i]}, {N_TIMEPOINTS})"
            assert input_all[i][j].shape == (4, N_TIMEPOINTS), \
                f"Session {i}, trial {j}: input shape {input_all[i][j].shape}"
            assert output_all[i][j].shape == (4, N_TIMEPOINTS), \
                f"Session {i}, trial {j}: output shape {output_all[i][j].shape}"

    print("Dimension checks passed!")

    return data


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', action='store_true', help='Process only 3 sample sessions')
    parser.add_argument('--output', type=str, default='converted_data.pkl', help='Output file path')
    args = parser.parse_args()

    data = process_all_sessions(sample_only=args.sample)

    output_path = args.output
    print(f"\nSaving to {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved! File size: {os.path.getsize(output_path) / 1e9:.2f} GB")
