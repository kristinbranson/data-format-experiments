#!/usr/bin/env python3
"""
Convert Zhong et al. (2025) data to decoder-compatible format.

Usage:
    python -u convert_data.py <outpicklefile> [--full|--sample] [--show-processing]
"""

import argparse
import os
import pickle
import sys
import time
from collections import defaultdict
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ==============================================================================
# Configuration
# ==============================================================================
DATA_ROOT = '/app/data'
CORRIDOR_LENGTH = 60  # decimeters (6m)
TEXTURE_LENGTH = 40   # decimeters (4m)
GREY_LENGTH = 20      # decimeters (2m)
N_POS_BINS = 4        # 4 bins of 1m each (10 decimeters)
N_SPEED_BINS = 4      # 4 quartile bins
POS_BIN_EDGES = np.array([0, 10, 20, 30, 40])  # bin edges in decimeters

# Brain region mapping from iarea values
# V1: iarea==8, mHV: iarea in {0,1,2,9}, lHV: iarea in {5,6}, aHV: iarea in {3,4}
# Exclude: iarea==-1, iarea==7
AREA_MAP = {
    8: 'V1',
    0: 'mHV', 1: 'mHV', 2: 'mHV', 9: 'mHV',
    5: 'lHV', 6: 'lHV',
    3: 'aHV', 4: 'aHV',
}
BRAIN_REGIONS = ['V1', 'mHV', 'lHV', 'aHV']

# ==============================================================================
# Helper functions
# ==============================================================================

def parse_date(datexp):
    """Parse date string like '2022_07_12' to datetime."""
    return datetime.strptime(datexp, '%Y_%m_%d')


def build_session_list():
    """Build list of unique sessions with metadata from Imaging_Exp_info."""
    exp_info = np.load(os.path.join(DATA_ROOT, 'beh', 'Imaging_Exp_info.npy'),
                       allow_pickle=True).item()

    # Collect all unique sessions
    sessions = {}  # key -> {'mname', 'datexp', 'blk', 'exp_types': [...], 'ndb': first_ndb}
    for exp_type, db_list in exp_info.items():
        for ndb in db_list:
            key = f"{ndb['mname']}_{ndb['datexp']}_{ndb['blk']}"
            if key not in sessions:
                sessions[key] = {
                    'mname': ndb['mname'],
                    'datexp': ndb['datexp'],
                    'blk': ndb['blk'],
                    'exp_types': [],
                    'ndb_list': [],
                }
            sessions[key]['exp_types'].append(exp_type)
            sessions[key]['ndb_list'].append(ndb)

    return sessions, exp_info


def find_behavior_for_session(session_key, sessions_info, exp_info):
    """Find and load behavior data for a given session.

    Some sessions appear in multiple experiment types with the same behavior data.
    For test3 sessions with 'stimtype', behavior keys have a suffix.
    We prefer experiment types without stimtype (simpler key).
    """
    info = sessions_info[session_key]

    # Try experiment types without stimtype first
    for exp_type, ndb in zip(info['exp_types'], info['ndb_list']):
        stimtype = ndb.get('stimtype', None)
        if stimtype is not None:
            continue
        beh_file = os.path.join(DATA_ROOT, 'beh', f'Beh_{exp_type}.npy')
        if os.path.exists(beh_file):
            beh_data = np.load(beh_file, allow_pickle=True).item()
            beh_key = session_key
            if beh_key in beh_data:
                return beh_data[beh_key], exp_type

    # Fall back to experiment types with stimtype
    for exp_type, ndb in zip(info['exp_types'], info['ndb_list']):
        stimtype = ndb.get('stimtype', None)
        beh_file = os.path.join(DATA_ROOT, 'beh', f'Beh_{exp_type}.npy')
        if os.path.exists(beh_file):
            beh_data = np.load(beh_file, allow_pickle=True).item()
            if stimtype is not None:
                beh_key = f'{session_key}_{stimtype}'
            else:
                beh_key = session_key
            if beh_key in beh_data:
                return beh_data[beh_key], exp_type

    return None, None


def load_neural_data(mname, datexp, blk):
    """Load and concatenate neural data from all planes."""
    fn = f'{mname}_{datexp}_{blk}_neural_data.npy'
    path = os.path.join(DATA_ROOT, 'spk', fn)
    data = np.load(path, allow_pickle=True).item()
    spk = np.concatenate(data['spks'], axis=0)  # (n_neurons, n_frames)
    return spk


def load_retinotopy(mname, datexp):
    """Load retinotopy data to get brain region assignments."""
    fn = f'{mname}_{datexp}_trans.npz'
    path = os.path.join(DATA_ROOT, 'retinotopy', fn)
    data = np.load(path, allow_pickle=True)
    return data['iarea']


def get_neuron_mask_and_regions(iarea):
    """Get mask for valid neurons and their brain region indices.

    Returns:
        mask: boolean array, True for neurons in visual cortex
        region_idx: integer array, index into BRAIN_REGIONS for each valid neuron
    """
    # Exclude iarea == -1 (unassigned) and iarea == 7
    mask = (iarea != -1) & (iarea != 7)

    region_idx = np.zeros(mask.sum(), dtype=np.int64)
    valid_iarea = iarea[mask]
    for iarea_val, region_name in AREA_MAP.items():
        region_idx[valid_iarea == iarea_val] = BRAIN_REGIONS.index(region_name)

    return mask, region_idx


def compute_day_of_training(sessions_info):
    """Compute day of training for each session based on dates.

    For each mouse, day 0 = first session date.
    """
    # Group sessions by mouse
    mouse_dates = defaultdict(list)
    for key, info in sessions_info.items():
        mouse_dates[info['mname']].append((key, parse_date(info['datexp'])))

    day_map = {}
    for mname, date_list in mouse_dates.items():
        date_list.sort(key=lambda x: x[1])
        first_date = date_list[0][1]
        for key, dt in date_list:
            day_map[key] = (dt - first_date).days

    return day_map


def extract_trial_frames(beh, trial_idx, n_neural_frames):
    """Get frame indices for a trial where mouse is running in corridor.

    Returns integer frame indices where:
    - ft_trInd == trial_idx
    - ft_CorrSpc == True
    - ft_move > 0
    """
    n_beh_frames = len(beh['ft_trInd'])
    n_frames = min(n_neural_frames, n_beh_frames)

    ft_trInd = beh['ft_trInd'][:n_frames]
    ft_CorrSpc = beh['ft_CorrSpc'][:n_frames]
    ft_move = beh['ft_move'][:n_frames]

    # ft_trInd can be float with NaN for frames outside behavior
    # Need to handle NaN comparison
    valid = np.isfinite(ft_trInd)
    trial_mask = valid & (ft_trInd.astype(float) == trial_idx)
    corr_mask = ft_CorrSpc.astype(bool)
    move_mask = ft_move > 0

    frame_mask = trial_mask & corr_mask & move_mask
    return np.where(frame_mask)[0]


def make_lick_binary(beh, frame_indices, trial_idx):
    """Create binary licking signal for given frames of a trial."""
    lick_fr = beh['LickFr']
    lick_trind = beh['LickTrind']

    if len(lick_fr) == 0:
        return np.zeros(len(frame_indices), dtype=np.float32)

    # Get lick frames for this trial
    trial_lick_mask = lick_trind == trial_idx
    trial_lick_fr = lick_fr[trial_lick_mask]

    # For each frame, check if any lick is closest to it
    # Lick frames are float (interpolated). Round to nearest int frame.
    if len(trial_lick_fr) == 0:
        return np.zeros(len(frame_indices), dtype=np.float32)

    lick_int_frames = np.round(trial_lick_fr).astype(int)
    lick_set = set(lick_int_frames)

    binary = np.array([1.0 if fi in lick_set else 0.0 for fi in frame_indices],
                      dtype=np.float32)
    return binary


def discretize_position(positions, bin_edges=POS_BIN_EDGES):
    """Discretize continuous position (0-40 dm) into bins.

    Bins: [0,10), [10,20), [20,30), [30,40]
    Returns integer bin indices 0-3.
    """
    # Clip to valid range
    pos_clipped = np.clip(positions, 0, TEXTURE_LENGTH - 1e-6)
    bins = np.digitize(pos_clipped, bin_edges) - 1  # 0-indexed
    bins = np.clip(bins, 0, N_POS_BINS - 1)
    return bins.astype(np.int64)


def collect_all_running_speeds(sessions_info, exp_info, session_keys):
    """Collect all running speeds from corridor+moving frames to compute quartiles."""
    print("Collecting running speeds for quartile computation...")
    t0 = time.time()
    all_speeds = []

    for session_key in session_keys:
        info = sessions_info[session_key]
        beh, _ = find_behavior_for_session(session_key, sessions_info, exp_info)
        if beh is None:
            continue

        n_frames = len(beh['ft_trInd'])
        ft_CorrSpc = beh['ft_CorrSpc'][:n_frames].astype(bool)
        ft_move = beh['ft_move'][:n_frames]
        ft_RunSpeed = beh['ft_RunSpeed'][:n_frames]

        mask = ft_CorrSpc & (ft_move > 0)
        speeds = ft_RunSpeed[mask]
        # Subsample for efficiency if too many frames
        if len(speeds) > 10000:
            rng = np.random.RandomState(42)
            idx = rng.choice(len(speeds), 10000, replace=False)
            speeds = speeds[idx]
        all_speeds.append(speeds)

    all_speeds = np.concatenate(all_speeds)
    quartiles = np.percentile(all_speeds, [25, 50, 75])
    print(f"  Speed quartiles: {quartiles}")
    print(f"  Speed range: [{all_speeds.min():.2f}, {all_speeds.max():.2f}]")
    print(f"  Collected in {time.time()-t0:.1f}s")
    return quartiles


def discretize_speed(speeds, quartile_edges):
    """Discretize running speed into 4 quartile bins."""
    bin_edges = np.array([-np.inf, quartile_edges[0], quartile_edges[1], quartile_edges[2], np.inf])
    bins = np.digitize(speeds, quartile_edges)  # 0,1,2,3
    return bins.astype(np.int64)


def process_session(session_key, sessions_info, exp_info, day_map,
                    speed_quartiles, neuron_mask_cache=None):
    """Process a single session and return trial data."""
    info = sessions_info[session_key]
    mname, datexp, blk = info['mname'], info['datexp'], info['blk']

    # Load behavior
    beh, exp_type = find_behavior_for_session(session_key, sessions_info, exp_info)
    if beh is None:
        print(f"  WARNING: No behavior found for {session_key}, skipping")
        return None

    # Load neural data
    spk = load_neural_data(mname, datexp, blk)
    n_neurons_total, n_neural_frames = spk.shape

    # Load retinotopy and filter neurons
    iarea = load_retinotopy(mname, datexp)

    # Sanity check: neuron counts should match
    if len(iarea) != n_neurons_total:
        print(f"  WARNING: Neuron count mismatch for {session_key}: "
              f"spk={n_neurons_total}, retino={len(iarea)}")
        # Use minimum
        min_n = min(len(iarea), n_neurons_total)
        iarea = iarea[:min_n]
        spk = spk[:min_n]
        n_neurons_total = min_n

    neuron_mask, region_idx = get_neuron_mask_and_regions(iarea)
    spk_filtered = spk[neuron_mask]  # (n_valid_neurons, n_frames)
    n_neurons = spk_filtered.shape[0]

    # Get trial info
    ntrials = beh['ntrials']
    day = day_map.get(session_key, 0)

    # Compute frame dt for time calculations
    ft = beh['ft']
    frame_dt_days = np.nanmedian(np.diff(ft))
    frame_dt_sec = frame_dt_days * 24 * 3600

    # Process trials
    neural_trials = []
    input_trials = []
    output_trials = []

    # Get unique stimuli for this session
    wall_names = beh['WallName']

    for t in range(ntrials):
        # Get valid frame indices for this trial
        frame_idx = extract_trial_frames(beh, t, n_neural_frames)

        if len(frame_idx) < 2:
            continue  # Skip trials with too few frames

        # Neural data: (n_neurons, n_timepoints)
        neural = spk_filtered[:, frame_idx].astype(np.float32)

        # --- INPUTS ---
        # Time to sound cue (seconds): negative before, positive after
        sound_fr = beh['SoundFr'][t]
        time_to_cue = (frame_idx - sound_fr) * frame_dt_sec

        # Day of training (scalar, repeated for all timepoints)
        day_val = float(day)

        # Time since trial start (seconds)
        start_fr = beh['StartFr'][t]
        time_since_start = (frame_idx - start_fr) * frame_dt_sec

        # Reward availability (0 or 1)
        reward_avail = float(beh['isRew'][t])

        # Construct input array: (4, n_timepoints) for time-varying, (4,) for per-trial
        n_tp = len(frame_idx)
        inp = np.zeros((4, n_tp), dtype=np.float32)
        inp[0, :] = time_to_cue.astype(np.float32)
        inp[1, :] = day_val  # per-trial, broadcast
        inp[2, :] = time_since_start.astype(np.float32)
        inp[3, :] = reward_avail  # per-trial, broadcast

        # --- OUTPUTS ---
        # Visual stimulus category (per-trial)
        stim_name = wall_names[t]

        # Licking (binary, time-varying)
        licking = make_lick_binary(beh, frame_idx, t)

        # Position (discretized, time-varying)
        n_beh_frames = len(beh['ft_Pos'])
        n_use = min(n_neural_frames, n_beh_frames)
        positions = beh['ft_Pos'][:n_use]
        trial_positions = positions[frame_idx[frame_idx < n_use]]
        if len(trial_positions) < len(frame_idx):
            # Pad with last valid position
            pad_len = len(frame_idx) - len(trial_positions)
            trial_positions = np.concatenate([
                trial_positions,
                np.full(pad_len, trial_positions[-1] if len(trial_positions) > 0 else 0)
            ])
        pos_bins = discretize_position(trial_positions)

        # Running speed (discretized, time-varying)
        n_speed_frames = len(beh['ft_RunSpeed'])
        n_use_speed = min(n_neural_frames, n_speed_frames)
        run_speeds = beh['ft_RunSpeed'][:n_use_speed]
        trial_speeds = run_speeds[frame_idx[frame_idx < n_use_speed]]
        if len(trial_speeds) < len(frame_idx):
            pad_len = len(frame_idx) - len(trial_speeds)
            trial_speeds = np.concatenate([
                trial_speeds,
                np.full(pad_len, trial_speeds[-1] if len(trial_speeds) > 0 else 0)
            ])
        speed_bins = discretize_speed(trial_speeds, speed_quartiles)

        # Construct output array: (4, n_timepoints)
        # output[0] = stimulus (per-trial, same for all timepoints)
        # output[1] = licking (time-varying)
        # output[2] = position (time-varying)
        # output[3] = running speed (time-varying)
        out = np.zeros((4, n_tp), dtype=np.int64)
        out[0, :] = -1  # placeholder, will be filled with stimulus index
        out[1, :] = licking.astype(np.int64)
        out[2, :] = pos_bins
        out[3, :] = speed_bins

        neural_trials.append(neural)
        input_trials.append(inp)
        output_trials.append((out, stim_name))

    if len(neural_trials) < 2:
        print(f"  WARNING: Session {session_key} has {len(neural_trials)} valid trials, skipping")
        return None

    return {
        'neural': neural_trials,
        'input': input_trials,
        'output': output_trials,
        'region_idx': region_idx,
        'n_neurons': n_neurons,
        'mname': mname,
        'session_key': session_key,
        'day': day_map.get(session_key, 0),
        'exp_type': exp_type,
        'frame_dt_sec': frame_dt_sec,
    }


def show_processing_plots(session_key, sessions_info, exp_info, day_map,
                          speed_quartiles, save_path):
    """Generate detailed processing visualization for a session."""
    info = sessions_info[session_key]
    mname, datexp, blk = info['mname'], info['datexp'], info['blk']

    beh, _ = find_behavior_for_session(session_key, sessions_info, exp_info)
    if beh is None:
        return

    spk = load_neural_data(mname, datexp, blk)
    n_neurons_total, n_neural_frames = spk.shape
    iarea = load_retinotopy(mname, datexp)
    min_n = min(len(iarea), n_neurons_total)
    iarea = iarea[:min_n]
    spk = spk[:min_n]
    neuron_mask, region_idx = get_neuron_mask_and_regions(iarea)
    spk_filtered = spk[neuron_mask]

    ft = beh['ft']
    frame_dt_sec = np.nanmedian(np.diff(ft)) * 24 * 3600

    fig, axes = plt.subplots(5, 3, figsize=(20, 18))
    fig.suptitle(f'Processing: {session_key}', fontsize=14)

    # Pick 3 example trials
    trial_indices = [0, beh['ntrials']//3, 2*beh['ntrials']//3]

    for col, t in enumerate(trial_indices):
        frame_idx = extract_trial_frames(beh, t, n_neural_frames)
        if len(frame_idx) < 2:
            continue

        time_axis = (frame_idx - beh['StartFr'][t]) * frame_dt_sec

        # Row 0: Neural activity (first 5 neurons)
        ax = axes[0, col]
        n_show = min(5, spk_filtered.shape[0])
        for i in range(n_show):
            ax.plot(time_axis, spk_filtered[i, frame_idx] + i*2, alpha=0.7)
        ax.set_title(f'Trial {t}: {beh["WallName"][t]}')
        ax.set_ylabel('Neural (offset)')

        # Row 1: Position
        ax = axes[1, col]
        n_beh = min(n_neural_frames, len(beh['ft_Pos']))
        pos = beh['ft_Pos'][:n_beh][frame_idx[frame_idx < n_beh]]
        t_pos = time_axis[:len(pos)]
        ax.plot(t_pos, pos, 'b-')
        for edge in POS_BIN_EDGES[1:-1]:
            ax.axhline(edge, color='gray', linestyle='--', alpha=0.5)
        ax.set_ylabel('Position (dm)')

        # Row 2: Running speed
        ax = axes[2, col]
        n_spd = min(n_neural_frames, len(beh['ft_RunSpeed']))
        spd = beh['ft_RunSpeed'][:n_spd][frame_idx[frame_idx < n_spd]]
        t_spd = time_axis[:len(spd)]
        ax.plot(t_spd, spd, 'g-')
        for q in speed_quartiles:
            ax.axhline(q, color='gray', linestyle='--', alpha=0.5)
        ax.set_ylabel('Speed')

        # Row 3: Licking
        ax = axes[3, col]
        lick = make_lick_binary(beh, frame_idx, t)
        ax.plot(time_axis, lick, 'r-', alpha=0.7)
        ax.set_ylabel('Licking')
        ax.set_ylim(-0.1, 1.1)

        # Row 4: Time to sound cue
        ax = axes[4, col]
        sound_fr = beh['SoundFr'][t]
        time_to_cue = (frame_idx - sound_fr) * frame_dt_sec
        ax.plot(time_axis, time_to_cue, 'm-')
        ax.axhline(0, color='k', linestyle='--')
        ax.set_ylabel('Time to cue (s)')
        ax.set_xlabel('Time since trial start (s)')

    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()
    print(f"  Saved processing plot: {save_path}")


def build_output(session_results, speed_quartiles):
    """Build the final output dictionary from processed session results."""

    # Collect all unique stimuli across sessions
    all_stimuli = set()
    for result in session_results:
        for _, stim_name in [r for r in [(out[1]) for out in result['output']]
                             if isinstance(r, str)] if False else []:
            pass
        for out_data, stim_name in result['output']:
            all_stimuli.add(stim_name)

    all_stimuli = sorted(all_stimuli)
    stim_to_idx = {s: i for i, s in enumerate(all_stimuli)}

    # Collect subjects
    all_subjects = sorted(set(r['mname'] for r in session_results))
    subject_to_idx = {s: i for i, s in enumerate(all_subjects)}

    # Build arrays
    neural = []
    inputs = []
    outputs = []
    subject_idx = []
    brain_region_idx = []

    for result in session_results:
        # Neural
        neural.append(result['neural'])

        # Inputs
        inputs.append(result['input'])

        # Outputs: fill in stimulus index
        session_outputs = []
        for out_data, stim_name in result['output']:
            out_filled = out_data.copy()
            out_filled[0, :] = stim_to_idx[stim_name]
            session_outputs.append(out_filled)
        outputs.append(session_outputs)

        # Subject index
        subject_idx.append(subject_to_idx[result['mname']])

        # Brain region index
        brain_region_idx.append(result['region_idx'])

    # Compute speed bin labels
    speed_labels = [
        f'<{speed_quartiles[0]:.1f}',
        f'{speed_quartiles[0]:.1f}-{speed_quartiles[1]:.1f}',
        f'{speed_quartiles[1]:.1f}-{speed_quartiles[2]:.1f}',
        f'>{speed_quartiles[2]:.1f}',
    ]

    # Compute frame dt
    frame_dts = [r['frame_dt_sec'] for r in session_results]
    mean_frame_dt = np.mean(frame_dts)

    data = {
        'neural': neural,
        'input': inputs,
        'output': outputs,

        'subjects': all_subjects,
        'subject_idx': np.array(subject_idx, dtype=np.int64),

        'brain_regions': BRAIN_REGIONS,
        'brain_region_idx': brain_region_idx,

        'input_names': [
            'time_to_sound_cue',
            'day_of_training',
            'time_since_trial_start',
            'reward_availability',
        ],
        'output_names': [
            'visual_stimulus',
            'licking',
            'position',
            'running_speed',
        ],
        'output_values': [
            all_stimuli,                      # visual stimulus categories
            ['not_licking', 'licking'],        # licking
            ['0-1m', '1-2m', '2-3m', '3-4m'], # position bins
            speed_labels,                      # running speed bins
        ],

        'metadata': {
            'task_description': 'Visual discrimination in VR corridors. Mice run through '
                              'linear corridors with naturalistic texture patterns (leaf, circle, etc). '
                              'Sound cue presented at random position; water reward in rewarded corridor.',
            'time_bin_size': mean_frame_dt * 1000,  # in ms
            'temporal_alignment_event': 'Corridor entry (trial start)',
            'off_start': 0.0,  # trial starts at corridor entry
            'off_end': None,  # variable trial length
            'frame_rate_hz': 1.0 / mean_frame_dt,
            'corridor_length_m': 4.0,
            'grey_space_length_m': 2.0,
            'speed_quartile_edges': speed_quartiles.tolist(),
            'neural_data_type': 'Suite2p deconvolved fluorescence (decay timescale 0.75s)',
            'only_running_frames': True,
            'only_corridor_frames': True,
            'neuron_filtering': 'Excluded neurons with iarea==-1 or iarea==7 (outside visual cortex)',
            'source_paper': 'Zhong et al. (2025) Unsupervised pretraining in biological neural networks',
        }
    }

    return data


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description='Convert Zhong et al. data to decoder format')
    parser.add_argument('outfile', type=str, help='Output pickle file path')
    parser.add_argument('--full', action='store_true', default=True, help='Process all sessions (default)')
    parser.add_argument('--sample', action='store_true', help='Process only 2 sessions')
    parser.add_argument('--show-processing', action='store_true', help='Plot processing visualizations')
    args = parser.parse_args()

    t_start = time.time()

    print("=" * 60)
    print("Building session list...")
    sessions_info, exp_info = build_session_list()
    all_session_keys = sorted(sessions_info.keys())
    print(f"  Found {len(all_session_keys)} unique sessions")

    # Select sessions
    if args.sample:
        # Pick 2 sessions: one with rewards, one without
        # Find one supervised and one unsupervised
        sample_keys = []
        for key in all_session_keys:
            info = sessions_info[key]
            if 'sup_test1' in info['exp_types'] and len(sample_keys) == 0:
                sample_keys.append(key)
            elif 'unsup_test1' in info['exp_types'] and len(sample_keys) == 1:
                sample_keys.append(key)
            if len(sample_keys) == 2:
                break
        if len(sample_keys) < 2:
            sample_keys = all_session_keys[:2]
        session_keys = sample_keys
        print(f"  Sample mode: processing {len(session_keys)} sessions: {session_keys}")
    else:
        session_keys = all_session_keys
        print(f"  Full mode: processing {len(session_keys)} sessions")

    # Compute day of training for all sessions
    print("Computing day of training...")
    day_map = compute_day_of_training(sessions_info)

    # Compute speed quartiles from all sessions (use all for consistent binning)
    speed_quartiles = collect_all_running_speeds(sessions_info, exp_info, all_session_keys)

    # Process sessions
    print("=" * 60)
    print("Processing sessions...")
    session_results = []

    for i, session_key in enumerate(session_keys):
        t_sess = time.time()
        print(f"\n[{i+1}/{len(session_keys)}] Processing {session_key}...")

        result = process_session(session_key, sessions_info, exp_info,
                                day_map, speed_quartiles)

        if result is not None:
            n_trials = len(result['neural'])
            total_frames = sum(n.shape[1] for n in result['neural'])
            print(f"  {result['n_neurons']} neurons, {n_trials} trials, "
                  f"{total_frames} total frames, {time.time()-t_sess:.1f}s")
            session_results.append(result)

        # Show processing plots for first 2 sessions
        if args.show_processing and i < 2:
            plot_path = f'processing_{session_key}.png'
            show_processing_plots(session_key, sessions_info, exp_info,
                                day_map, speed_quartiles, plot_path)

    # Build output
    print("\n" + "=" * 60)
    print("Building output dictionary...")
    data = build_output(session_results, speed_quartiles)

    # Print summary
    n_sessions = len(data['neural'])
    n_subjects = len(data['subjects'])
    total_trials = sum(len(s) for s in data['neural'])
    total_neurons = sum(data['brain_region_idx'][i].shape[0] for i in range(n_sessions))
    print(f"\nSummary:")
    print(f"  Sessions: {n_sessions}")
    print(f"  Subjects: {n_subjects}")
    print(f"  Total trials: {total_trials}")
    print(f"  Mean trials/session: {total_trials/n_sessions:.1f}")
    print(f"  Mean neurons/session: {total_neurons/n_sessions:.0f}")
    print(f"  Stimuli: {data['output_values'][0]}")
    print(f"  Time bin: {data['metadata']['time_bin_size']:.1f} ms")

    # Save
    print(f"\nSaving to {args.outfile}...")
    with open(args.outfile, 'wb') as f:
        pickle.dump(data, f, protocol=4)

    file_size = os.path.getsize(args.outfile) / (1024**2)
    print(f"  File size: {file_size:.1f} MB")

    total_time = time.time() - t_start
    print(f"\nTotal time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print("Done!")


if __name__ == '__main__':
    main()
