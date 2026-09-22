#!/usr/bin/env python3
"""
Convert Zhong et al. 2025 calcium imaging + VR corridor data to decoder-compatible format.

Usage:
    python -u convert_data.py <output_pickle> [--full|--sample] [--show-processing]
"""

import argparse
import os
import sys
import time
import pickle
import numpy as np
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# Constants
# ============================================================
DATA_ROOT = '/app/data'
CORRIDOR_LENGTH_DM = 40  # texture corridor = 4m = 40 decimeters
TOTAL_CORRIDOR_DM = 60   # full corridor = 6m = 60 decimeters
RUNNING_THRESHOLD = 0     # ft_move > 0 means VR is moving
N_POSITION_BINS = 4       # 4 bins of 1m each (0-10, 10-20, 20-30, 30-40 dm)
N_SPEED_BINS = 4          # 4 quartile bins
POSITION_BIN_EDGES_DM = np.array([0, 10, 20, 30, 40])  # decimeter edges

# Brain area mapping from iarea codes
BRAIN_REGION_MAP = {
    8: 'V1',
    0: 'mHV', 1: 'mHV', 2: 'mHV', 9: 'mHV',
    5: 'lHV', 6: 'lHV',
    3: 'aHV', 4: 'aHV',
}
BRAIN_REGIONS = ['V1', 'mHV', 'lHV', 'aHV']
EXCLUDED_IAREA = {-1, 7}  # outside visual cortex / unassigned

# Stimulus category mapping
# Note: paper calls 4 categories "circle, leaf, rock, brick" but actual data labels
# use "wood" instead of "brick". We use data labels.
STIM_CATEGORY_MAP = {}
for s in ['circle1', 'circle2', 'circle3']:
    STIM_CATEGORY_MAP[s] = 'circle'
for s in ['leaf1', 'leaf2', 'leaf3', 'leaf1_swap1', 'leaf1_swap2']:
    STIM_CATEGORY_MAP[s] = 'leaf'
for s in ['rock1', 'rock2']:
    STIM_CATEGORY_MAP[s] = 'rock'
for s in ['wood1', 'wood2', 'wood5', 'wood1_swap1', 'wood1_swap2']:
    STIM_CATEGORY_MAP[s] = 'wood'
STIM_CATEGORIES = ['circle', 'leaf', 'rock', 'wood']


def load_spk(mname, datexp, blk, root=DATA_ROOT):
    """Load neural data, concatenate across imaging planes (matches reference utils.load_spk)."""
    fn = os.path.join(root, 'spk', f'{mname}_{datexp}_{blk}_neural_data.npy')
    dat = np.load(fn, allow_pickle=True).item()
    spk = np.concatenate(dat['spks'], axis=0)  # (n_neurons, n_frames)
    return spk


def load_retino(mname, datexp, root=DATA_ROOT):
    """Load retinotopy data and return iarea array."""
    fn = os.path.join(root, 'retinotopy', f'{mname}_{datexp}_trans.npz')
    dat = np.load(fn, allow_pickle=True)
    return dat['iarea']


def get_neuron_mask(iarea):
    """Return boolean mask for neurons in visual cortex (exclude iarea=-1 and iarea=7)."""
    mask = np.ones(len(iarea), dtype=bool)
    for exc in EXCLUDED_IAREA:
        mask &= (iarea != exc)
    return mask


def get_brain_region_indices(iarea, neuron_mask):
    """Map each included neuron to a brain region index."""
    filtered_iarea = iarea[neuron_mask]
    region_idx = np.zeros(len(filtered_iarea), dtype=np.int64)
    for i, ia in enumerate(filtered_iarea):
        region_name = BRAIN_REGION_MAP.get(int(ia), None)
        if region_name is not None:
            region_idx[i] = BRAIN_REGIONS.index(region_name)
        else:
            region_idx[i] = 0  # shouldn't happen after masking
    return region_idx


def get_session_beh(session_key, exp_info):
    """Find behavioral data for a session key across all experiment types."""
    mname, datexp, blk = session_key.split('_', 2)
    datexp_parts = session_key.split('_')
    mname = datexp_parts[0]
    datexp = '_'.join(datexp_parts[1:4])
    blk = datexp_parts[4]

    for exp_type, sessions in exp_info.items():
        for s in sessions:
            if s['mname'] == mname and s['datexp'] == datexp and s['blk'] == blk:
                beh_path = os.path.join(DATA_ROOT, 'beh', f'Beh_{exp_type}.npy')
                beh_all = np.load(beh_path, allow_pickle=True).item()
                # Try different key formats
                for key_fmt in [
                    f'{mname}_{datexp}_{blk}',
                    f'{mname}_{datexp}_{blk}_{s.get("stimtype", "")}',
                ]:
                    if key_fmt in beh_all:
                        return beh_all[key_fmt], exp_type, s
                # Also try without stimtype
                base_key = f'{mname}_{datexp}_{blk}'
                if base_key in beh_all:
                    return beh_all[base_key], exp_type, s
    return None, None, None


def get_all_unique_sessions(exp_info):
    """Get list of all unique physical recordings with metadata."""
    seen = set()
    sessions = []
    for exp_type, session_list in exp_info.items():
        for s in session_list:
            key = f"{s['mname']}_{s['datexp']}_{s['blk']}"
            if key not in seen:
                seen.add(key)
                sessions.append({
                    'key': key,
                    'mname': s['mname'],
                    'datexp': s['datexp'],
                    'blk': s['blk'],
                    'exp_type': exp_type,
                    'db': s,
                })
    return sessions


def compute_day_of_training(sessions):
    """Compute day of training for each session (days since first session for each mouse)."""
    mouse_dates = {}
    for s in sessions:
        mname = s['mname']
        if mname not in mouse_dates:
            mouse_dates[mname] = []
        date = datetime.strptime(s['datexp'], '%Y_%m_%d')
        mouse_dates[mname].append((date, s['key']))

    day_map = {}
    for mname, dates in mouse_dates.items():
        dates.sort(key=lambda x: x[0])
        first_date = dates[0][0]
        for date, key in dates:
            day_map[key] = (date - first_date).days
    return day_map


def compute_speed_quartile_edges(sessions, exp_info):
    """Compute running speed quartile bin edges across all sessions."""
    print("Computing running speed quartile edges across all sessions...")
    t0 = time.time()
    all_speeds = []
    for s in sessions:
        beh, _, _ = get_session_beh(s['key'], exp_info)
        if beh is None:
            continue
        nfr_beh = len(beh['ft_trInd'])
        ft_CorrSpc = beh['ft_CorrSpc'][:nfr_beh]
        ft_move = beh['ft_move'][:nfr_beh]
        ft_RunSpeed = beh['ft_RunSpeed'][:nfr_beh]
        mask = ft_CorrSpc & (ft_move > RUNNING_THRESHOLD)
        speeds = ft_RunSpeed[mask]
        # Subsample if too many
        if len(speeds) > 10000:
            rng = np.random.default_rng(42)
            speeds = rng.choice(speeds, 10000, replace=False)
        all_speeds.append(speeds)

    all_speeds = np.concatenate(all_speeds)
    edges = np.percentile(all_speeds, [0, 25, 50, 75, 100])
    print(f"  Speed quartile edges: {edges}")
    print(f"  Total speed samples: {len(all_speeds)}")
    print(f"  Computed in {time.time()-t0:.1f}s")
    return edges


def process_session(session_info, exp_info, day_map, speed_edges, frame_dt_s=None):
    """Process a single session and return per-trial data."""
    key = session_info['key']
    mname = session_info['mname']
    datexp = session_info['datexp']
    blk = session_info['blk']

    t0 = time.time()

    # Load behavioral data
    beh, exp_type, db_entry = get_session_beh(key, exp_info)
    if beh is None:
        print(f"  WARNING: No behavioral data found for {key}, skipping")
        return None

    ntrials = beh['ntrials']
    nfr_beh = len(beh['ft_trInd'])

    # Load neural data
    spk = load_spk(mname, datexp, blk)
    n_neurons_raw, n_frames = spk.shape
    t_load = time.time()

    # Load retinotopy and filter neurons
    iarea = load_retino(mname, datexp)
    if len(iarea) != n_neurons_raw:
        print(f"  WARNING: neuron count mismatch for {key}: spk={n_neurons_raw}, retino={len(iarea)}")
        min_n = min(n_neurons_raw, len(iarea))
        spk = spk[:min_n]
        iarea = iarea[:min_n]
        n_neurons_raw = min_n

    neuron_mask = get_neuron_mask(iarea)
    brain_region_idx = get_brain_region_indices(iarea, neuron_mask)
    spk = spk[neuron_mask]  # (n_neurons_filtered, n_frames)
    n_neurons = spk.shape[0]

    t_filter = time.time()

    # Frame-level data
    nfr = min(n_frames, nfr_beh)
    ft_trInd = beh['ft_trInd'][:nfr]
    ft_CorrSpc = beh['ft_CorrSpc'][:nfr]
    ft_move = beh['ft_move'][:nfr]
    ft_Pos = beh['ft_Pos'][:nfr]
    ft_RunSpeed = beh['ft_RunSpeed'][:nfr]

    # Frame timestamps (in days, convert to seconds)
    ft = beh['ft'][:nfr]
    if frame_dt_s is None:
        frame_dt_s = float(np.median(np.diff(ft)) * 86400)  # days to seconds

    # Sound cue frame for each trial
    SoundFr = beh['SoundFr']

    # Licking data
    LickFr = beh['LickFr']
    LickTrind = beh['LickTrind']

    # Day of training
    day_of_training = float(day_map.get(key, 0))

    # Reward availability
    isRew = beh['isRew']

    # Stimulus names
    WallName = beh['WallName']

    # Process each trial
    neural_trials = []
    input_trials = []
    output_trials = []
    valid_trial_indices = []

    for trial_idx in range(ntrials):
        # Find valid frames: in corridor, running, and matching trial
        trial_mask = (ft_trInd == trial_idx) & ft_CorrSpc & (ft_move > RUNNING_THRESHOLD)
        frame_indices = np.where(trial_mask)[0]

        if len(frame_indices) < 2:
            continue  # skip trials with too few frames

        # Neural activity for this trial
        neural = spk[:, frame_indices].astype(np.float32)  # (n_neurons, n_timepoints)

        # --- INPUTS ---
        n_tp = len(frame_indices)

        # Use actual timestamps (converted from MATLAB datenum to seconds)
        frame_times = ft[frame_indices] * 86400  # convert days to seconds

        # Time to sound cue (seconds) - use actual timestamps
        sound_fr = SoundFr[trial_idx]
        # Interpolate sound time from nearest frames
        sound_time_s = np.interp(sound_fr, np.arange(nfr), ft[:nfr]) * 86400
        time_to_sound = sound_time_s - frame_times  # positive before cue, negative after

        # Time since trial start (seconds) - use actual timestamps
        time_since_start = frame_times - frame_times[0]

        # Reward availability (per-trial scalar)
        reward_avail = np.full(1, float(isRew[trial_idx]), dtype=np.float32)

        # Construct input array: mix of time-varying and per-trial
        # time_to_sound_cue: (1, n_tp)
        # day_of_training: (1,) per-trial
        # time_since_trial_start: (1, n_tp)
        # reward_availability: (1,) per-trial
        input_data = np.vstack([
            time_to_sound.reshape(1, -1).astype(np.float32),
            np.full((1, n_tp), day_of_training, dtype=np.float32),
            time_since_start.reshape(1, -1).astype(np.float32),
            np.full((1, n_tp), float(isRew[trial_idx]), dtype=np.float32),
        ])  # (4, n_tp)

        # --- OUTPUTS ---
        # Stimulus category (per-trial)
        wall_name = str(WallName[trial_idx])
        stim_cat = STIM_CATEGORY_MAP.get(wall_name, None)
        if stim_cat is None:
            print(f"  WARNING: Unknown stimulus '{wall_name}' in trial {trial_idx}, skipping")
            continue
        stim_cat_idx = STIM_CATEGORIES.index(stim_cat)

        # Licking (binary, time-varying) - vectorized
        trial_lick_mask = (LickTrind == trial_idx)
        trial_lick_frs = LickFr[trial_lick_mask]
        lick_binary = np.zeros(n_tp, dtype=np.float32)
        if len(trial_lick_frs) > 0:
            # Assign each lick to nearest frame
            # Use searchsorted to find nearest frame for each lick
            fi_sorted = frame_indices.astype(float)
            lick_bin_idx = np.searchsorted(fi_sorted, trial_lick_frs)
            # For each lick, choose the closer of the two neighboring frames
            for li in range(len(trial_lick_frs)):
                idx = lick_bin_idx[li]
                if idx >= n_tp:
                    idx = n_tp - 1
                elif idx > 0:
                    if abs(trial_lick_frs[li] - fi_sorted[idx-1]) < abs(trial_lick_frs[li] - fi_sorted[idx]):
                        idx = idx - 1
                lick_binary[idx] = 1.0

        # Position (4 bins, time-varying)
        positions = ft_Pos[frame_indices]
        # Clip to [0, 40) and bin
        positions = np.clip(positions, 0, CORRIDOR_LENGTH_DM - 0.001)
        pos_bin = np.digitize(positions, POSITION_BIN_EDGES_DM[1:])  # 0,1,2,3
        pos_bin = np.clip(pos_bin, 0, N_POSITION_BINS - 1)

        # Running speed (4 quartile bins, time-varying)
        speeds = ft_RunSpeed[frame_indices]
        speed_bin = np.digitize(speeds, speed_edges[1:-1])  # 0,1,2,3
        speed_bin = np.clip(speed_bin, 0, N_SPEED_BINS - 1)

        # Construct output array (integer-valued for categorical outputs)
        output_data = np.vstack([
            np.full((1, n_tp), stim_cat_idx, dtype=np.int64),
            lick_binary.reshape(1, -1).astype(np.int64),
            pos_bin.reshape(1, -1).astype(np.int64),
            speed_bin.reshape(1, -1).astype(np.int64),
        ])  # (4, n_tp)

        neural_trials.append(neural)
        input_trials.append(input_data)
        output_trials.append(output_data)
        valid_trial_indices.append(trial_idx)

    t_process = time.time()

    if len(neural_trials) < 2:
        print(f"  WARNING: {key} has {len(neural_trials)} valid trials, skipping")
        return None

    result = {
        'neural': neural_trials,
        'input': input_trials,
        'output': output_trials,
        'n_neurons': n_neurons,
        'brain_region_idx': brain_region_idx,
        'mname': mname,
        'key': key,
        'n_trials': len(neural_trials),
        'n_neurons_raw': n_neurons_raw,  # total before filter (already correct)
    }

    print(f"  {key}: {n_neurons} neurons, {len(neural_trials)}/{ntrials} trials, "
          f"load={t_load-t0:.1f}s, filter={t_filter-t_load:.1f}s, process={t_process-t_filter:.1f}s")

    # Clean up large arrays
    del spk
    return result


def plot_processing(session_result, session_info, exp_info, speed_edges, day_map, save_path):
    """Plot processing visualizations for a session."""
    key = session_info['key']
    mname = session_info['mname']
    datexp = session_info['datexp']
    blk = session_info['blk']

    # Reload behavioral data for plotting
    beh, _, _ = get_session_beh(key, exp_info)
    nfr_beh = len(beh['ft_trInd'])

    fig, axes = plt.subplots(4, 3, figsize=(18, 16))
    fig.suptitle(f'Processing: {key}', fontsize=14)

    neural_trials = session_result['neural']
    input_trials = session_result['input']
    output_trials = session_result['output']

    # Plot 1: Neural activity heatmap for first trial
    ax = axes[0, 0]
    if len(neural_trials) > 0:
        trial_data = neural_trials[0]
        n_show = min(50, trial_data.shape[0])
        ax.imshow(trial_data[:n_show], aspect='auto', cmap='hot')
        ax.set_title(f'Neural activity (trial 0, {n_show}/{trial_data.shape[0]} neurons)')
        ax.set_xlabel('Time bin')
        ax.set_ylabel('Neuron')

    # Plot 2: Trial length distribution
    ax = axes[0, 1]
    trial_lengths = [t.shape[1] for t in neural_trials]
    ax.hist(trial_lengths, bins=30, edgecolor='black')
    ax.set_title(f'Trial lengths (n={len(trial_lengths)})')
    ax.set_xlabel('Frames per trial')
    ax.set_ylabel('Count')

    # Plot 3: Neuron count per brain region
    ax = axes[0, 2]
    br_idx = session_result['brain_region_idx']
    for i, name in enumerate(BRAIN_REGIONS):
        count = (br_idx == i).sum()
        ax.bar(i, count, label=f'{name}: {count}')
    ax.set_xticks(range(len(BRAIN_REGIONS)))
    ax.set_xticklabels(BRAIN_REGIONS)
    ax.set_title('Neurons per brain region')

    # Plot 4: Time to sound cue for first 5 trials
    ax = axes[1, 0]
    for i in range(min(5, len(input_trials))):
        ax.plot(input_trials[i][0], label=f'Trial {i}')
    ax.set_title('Time to sound cue (s)')
    ax.set_xlabel('Time bin')
    ax.axhline(0, color='k', linestyle='--', alpha=0.5)
    ax.legend(fontsize=6)

    # Plot 5: Time since trial start for first 5 trials
    ax = axes[1, 1]
    for i in range(min(5, len(input_trials))):
        ax.plot(input_trials[i][2], label=f'Trial {i}')
    ax.set_title('Time since trial start (s)')
    ax.set_xlabel('Time bin')

    # Plot 6: Reward availability distribution
    ax = axes[1, 2]
    reward_vals = [input_trials[i][3, 0] for i in range(len(input_trials))]
    ax.bar(['Non-rewarded', 'Rewarded'],
           [reward_vals.count(0.0), reward_vals.count(1.0)])
    ax.set_title('Reward availability')

    # Plot 7: Stimulus category distribution
    ax = axes[2, 0]
    stim_vals = [int(output_trials[i][0, 0]) for i in range(len(output_trials))]
    cats_present = sorted(set(stim_vals))
    cat_counts = [stim_vals.count(c) for c in cats_present]
    cat_names = [STIM_CATEGORIES[c] for c in cats_present]
    ax.bar(cat_names, cat_counts)
    ax.set_title('Stimulus category')

    # Plot 8: Licking rate across trials
    ax = axes[2, 1]
    lick_rates = [output_trials[i][1].mean() for i in range(len(output_trials))]
    ax.plot(lick_rates, '.', markersize=2)
    ax.set_title('Licking rate per trial')
    ax.set_xlabel('Trial')
    ax.set_ylabel('Fraction of frames with lick')

    # Plot 9: Position distribution
    ax = axes[2, 2]
    all_pos = np.concatenate([output_trials[i][2].ravel() for i in range(len(output_trials))])
    ax.hist(all_pos, bins=N_POSITION_BINS, range=(-0.5, N_POSITION_BINS - 0.5), edgecolor='black')
    ax.set_title('Position bin distribution')
    ax.set_xticks(range(N_POSITION_BINS))
    ax.set_xticklabels(['0-1m', '1-2m', '2-3m', '3-4m'])

    # Plot 10: Running speed distribution
    ax = axes[3, 0]
    all_speed = np.concatenate([output_trials[i][3].ravel() for i in range(len(output_trials))])
    ax.hist(all_speed, bins=N_SPEED_BINS, range=(-0.5, N_SPEED_BINS - 0.5), edgecolor='black')
    ax.set_title('Speed bin distribution')
    ax.set_xlabel('Speed quartile')

    # Plot 11: Example trial - neural + licking + position overlay
    ax = axes[3, 1]
    if len(neural_trials) > 5:
        tidx = 5
        t_bins = np.arange(neural_trials[tidx].shape[1])
        ax.plot(t_bins, output_trials[tidx][2].ravel() / 3.0, label='Position (norm)', alpha=0.7)
        ax.plot(t_bins, output_trials[tidx][1].ravel(), label='Licking', alpha=0.7)
        ax.plot(t_bins, output_trials[tidx][3].ravel() / 3.0, label='Speed (norm)', alpha=0.7)
        ax.set_title(f'Trial {tidx} outputs')
        ax.legend(fontsize=6)
        ax.set_xlabel('Time bin')

    # Plot 12: Speed quartile edges
    ax = axes[3, 2]
    ax.bar(range(len(speed_edges)), speed_edges)
    ax.set_title('Speed bin edges')
    ax.set_xlabel('Edge index')
    ax.set_ylabel('Speed (cm/s)')

    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()
    print(f"  Saved processing plot: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', help='Output pickle file path')
    parser.add_argument('--full', action='store_true', default=True, help='Process all sessions (default)')
    parser.add_argument('--sample', action='store_true', help='Process only 2 sessions')
    parser.add_argument('--show-processing', action='store_true', help='Plot processing visualizations')
    args = parser.parse_args()

    if args.sample:
        args.full = False

    total_start = time.time()

    # Load experiment info
    print("Loading experiment info...")
    exp_info = np.load(os.path.join(DATA_ROOT, 'beh', 'Imaging_Exp_info.npy'), allow_pickle=True).item()

    # Get all unique sessions
    all_sessions = get_all_unique_sessions(exp_info)
    print(f"Total unique sessions: {len(all_sessions)}")

    # Compute day of training for each session
    day_map = compute_day_of_training(all_sessions)

    if args.sample:
        # Pick 2 sessions from different mice with reward
        # Use one sup (has reward) and one unsup (no reward) for diversity
        sample_keys = []
        for s in all_sessions:
            if s['mname'] == 'TX108' and s['datexp'] == '2023_03_25':
                sample_keys.append(s)
            elif s['mname'] == 'TX88' and s['datexp'] == '2022_06_20':
                sample_keys.append(s)
            if len(sample_keys) == 2:
                break
        if len(sample_keys) < 2:
            sample_keys = all_sessions[:2]
        sessions = sample_keys
        print(f"Sample mode: processing {len(sessions)} sessions")
    else:
        sessions = all_sessions
        print(f"Full mode: processing {len(sessions)} sessions")

    # Compute speed quartile edges across all sessions (or sample subset)
    speed_sessions = all_sessions if args.full else sessions
    speed_edges = compute_speed_quartile_edges(speed_sessions, exp_info)

    # Process sessions
    all_neural = []
    all_input = []
    all_output = []
    all_subjects = []
    subject_idx_list = []
    all_brain_region_idx = []

    subjects_seen = {}

    for i, session in enumerate(sessions):
        print(f"\n[{i+1}/{len(sessions)}] Processing {session['key']}...")
        t_sess = time.time()

        result = process_session(session, exp_info, day_map, speed_edges)
        if result is None:
            continue

        # Track subjects
        mname = result['mname']
        if mname not in subjects_seen:
            subjects_seen[mname] = len(subjects_seen)
        subj_idx = subjects_seen[mname]

        all_neural.append(result['neural'])
        all_input.append(result['input'])
        all_output.append(result['output'])
        subject_idx_list.append(subj_idx)
        all_brain_region_idx.append(result['brain_region_idx'])

        # Plot processing if requested
        if args.show_processing and i < 2:
            plot_path = f'/app/processing_{session["key"]}.png'
            plot_processing(result, session, exp_info, speed_edges, day_map, plot_path)

        print(f"  Session time: {time.time()-t_sess:.1f}s")

    # Build final data structure
    subjects = list(subjects_seen.keys())
    subject_idx = np.array(subject_idx_list, dtype=np.int64)

    # Speed bin labels
    speed_labels = []
    for i in range(N_SPEED_BINS):
        low = speed_edges[i]
        high = speed_edges[i + 1]
        speed_labels.append(f'{low:.1f}-{high:.1f} cm/s')

    data = {
        'neural': all_neural,
        'input': all_input,
        'output': all_output,

        'subjects': subjects,
        'subject_idx': subject_idx,

        'brain_regions': BRAIN_REGIONS,
        'brain_region_idx': all_brain_region_idx,

        'input_names': ['time_to_sound_cue', 'day_of_training', 'time_since_trial_start', 'reward_availability'],
        'output_names': ['stimulus_category', 'licking', 'position', 'running_speed'],
        'output_values': [
            STIM_CATEGORIES,   # stimulus category values
            ['no_lick', 'lick'],  # licking values
            ['0-1m', '1-2m', '2-3m', '3-4m'],  # position bin labels
            speed_labels,  # speed bin labels
        ],

        'metadata': {
            'task_description': 'Visual discrimination in head-fixed mice running through virtual reality corridors with naturalistic textures. Mice discriminate between texture categories (circle, leaf, rock, brick). Sound cue indicates reward zone in rewarded trials.',
            'time_bin_size': 315.0,  # approximate, in ms (1/3.178 Hz * 1000)
            'temporal_alignment_event': 'Trial start (corridor entry, first running frame in texture corridor)',
            'off_start': 0.0,  # alignment is at trial start
            'off_end': None,  # variable trial lengths
            'frame_rate_hz': 3.178,
            'corridor_length_m': 4.0,
            'grey_space_length_m': 2.0,
            'running_threshold_cm_s': 6.0,
            'vr_speed_cm_s': 60.0,
            'calcium_indicator': 'GCaMP6s',
            'deconvolution_timescale_s': 0.75,
            'speed_quartile_edges': speed_edges.tolist(),
            'n_sessions': len(all_neural),
            'n_subjects': len(subjects),
        }
    }

    # Print summary
    total_trials = sum(len(s) for s in all_neural)
    total_neurons = sum(br.shape[0] for br in all_brain_region_idx)
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Sessions: {len(all_neural)}")
    print(f"  Subjects: {len(subjects)}")
    print(f"  Total trials: {total_trials}")
    print(f"  Total neurons (filtered): {total_neurons}")
    print(f"  Mean trials/session: {total_trials/len(all_neural):.1f}")
    print(f"  Mean neurons/session: {total_neurons/len(all_neural):.0f}")

    # Save
    print(f"\nSaving to {args.output}...")
    with open(args.output, 'wb') as f:
        pickle.dump(data, f, protocol=4)
    file_size = os.path.getsize(args.output)
    print(f"Saved: {file_size/1e9:.2f} GB")
    print(f"Total time: {time.time()-total_start:.1f}s")


if __name__ == '__main__':
    main()
