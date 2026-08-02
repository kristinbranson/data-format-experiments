#!/usr/bin/env python3
"""
Convert Zhong et al. 2025 data to decoder-compatible format.

Usage:
    python -u convert_data.py <outpicklefile> [--full | --sample] [--show-processing]
"""

import argparse
import os
import sys
import gc
import time
import pickle
import numpy as np
from datetime import datetime
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# =============================================================================
# Constants
# =============================================================================
DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
FRAME_RATE_HZ = 3.17  # approximate, computed per-session from timestamps
CORRIDOR_LENGTH_DM = 40  # 4m texture area in decimeters
GREY_SPACE_DM = 20  # 2m grey space
TOTAL_LENGTH_DM = 60  # 6m total
POSITION_BINS = 4  # 4 equal 1m bins
SPEED_BINS = 4  # 4 quartile bins

# Brain region mapping from iarea values
BRAIN_REGIONS = ['V1', 'mHV', 'lHV', 'aHV', 'other']


def iarea_to_region_idx(iarea):
    """Map iarea values to brain region indices."""
    region_idx = np.full(len(iarea), 4, dtype=int)  # default: 'other'
    region_idx[iarea == 8] = 0  # V1
    region_idx[(iarea == 0) | (iarea == 1) | (iarea == 2) | (iarea == 9)] = 1  # mHV
    region_idx[(iarea == 5) | (iarea == 6)] = 2  # lHV
    region_idx[(iarea == 3) | (iarea == 4)] = 3  # aHV
    # iarea == -1 or 7 remain as 4 ('other')
    return region_idx


def load_spk(mname, datexp, blk, root=''):
    """Load neural data, concatenating across planes. Matches reference code."""
    fn = f'{mname}_{datexp}_{blk}_neural_data.npy'
    spk_path = os.path.join(root, fn)
    spk = np.concatenate(
        [nspk for nspk in np.load(spk_path, allow_pickle=True).item()['spks']], 0
    )
    return spk


def load_retino(mname, datexp, root=''):
    """Load retinotopy data."""
    fn = f'{mname}_{datexp}_trans.npz'
    dtrans = np.load(os.path.join(root, fn), allow_pickle=True)
    return dtrans['iarea']


def get_unique_sessions():
    """Build list of unique sessions with metadata from exp_info."""
    exp_info = np.load(
        os.path.join(DATA_ROOT, 'beh', 'Imaging_Exp_info.npy'),
        allow_pickle=True
    ).item()

    # Track unique sessions and their metadata
    session_map = {}  # session_key -> (exp_type, db_entry)

    for exp_type in exp_info:
        for ndb in exp_info[exp_type]:
            key = f"{ndb['mname']}_{ndb['datexp']}_{ndb['blk']}"
            if key not in session_map:
                session_map[key] = (exp_type, ndb)

    return session_map


def load_behavior_for_session(session_key, exp_type, ndb):
    """Load behavior data for a session from the appropriate file."""
    beh_file = os.path.join(DATA_ROOT, 'beh', f'Beh_{exp_type}.npy')
    beh_data = np.load(beh_file, allow_pickle=True).item()

    # Try standard key first
    if session_key in beh_data:
        return beh_data[session_key]

    # Try with stimtype appended
    stimtype = ndb.get('stimtype', None)
    if stimtype:
        key_with_stim = f"{session_key}_{stimtype}"
        if key_with_stim in beh_data:
            return beh_data[key_with_stim]

    raise KeyError(f"Could not find behavior for {session_key} in {beh_file}")


def compute_day_of_training(session_map):
    """Compute day of training for each session relative to mouse's first recording."""
    # Group sessions by mouse
    mouse_sessions = defaultdict(list)
    for sess_key, (exp_type, ndb) in session_map.items():
        mname = ndb['mname']
        datexp = ndb['datexp']
        # Parse date
        date = datetime.strptime(datexp, '%Y_%m_%d')
        mouse_sessions[mname].append((sess_key, date))

    # Compute day offset from first session per mouse
    day_map = {}
    for mname, sessions in mouse_sessions.items():
        sessions.sort(key=lambda x: x[1])
        first_date = sessions[0][1]
        for sess_key, date in sessions:
            day_map[sess_key] = (date - first_date).days

    return day_map


def discretize_position(pos, n_bins=4):
    """Discretize position (0-40 dm) into n_bins equal 1m bins."""
    bin_edges = np.linspace(0, CORRIDOR_LENGTH_DM, n_bins + 1)
    binned = np.digitize(pos, bin_edges) - 1
    binned = np.clip(binned, 0, n_bins - 1)
    return binned.astype(int)


def compute_speed_bin_edges(all_speeds):
    """Compute quartile bin edges for running speed across all data."""
    # Remove NaN/inf
    valid = all_speeds[np.isfinite(all_speeds)]
    quartiles = np.percentile(valid, [25, 50, 75])
    return quartiles


def discretize_speed(speed, quartiles):
    """Discretize speed into 4 quartile bins."""
    binned = np.digitize(speed, quartiles, right=True)
    binned = np.clip(binned, 0, 3)
    return binned


def build_lick_binary(beh, start_fr, end_fr):
    """Build binary licking array for frames between start_fr and end_fr."""
    n_frames = end_fr - start_fr
    lick_binary = np.zeros(n_frames, dtype=np.float32)

    lick_frs = beh['LickFr']
    if len(lick_frs) == 0:
        return lick_binary

    lick_trinds = beh['LickTrind']

    # Round lick frames to nearest int
    lick_frs_int = np.round(lick_frs).astype(int)

    # Find licks in this frame range
    mask = (lick_frs_int >= start_fr) & (lick_frs_int < end_fr)
    if mask.any():
        frame_offsets = lick_frs_int[mask] - start_fr
        frame_offsets = np.clip(frame_offsets, 0, n_frames - 1)
        lick_binary[frame_offsets] = 1.0

    return lick_binary


def process_session(session_key, exp_type, ndb, day_of_training, speed_quartiles,
                    show_processing=False):
    """Process a single session and return trial-level data."""
    t0 = time.time()

    mname = ndb['mname']
    datexp = ndb['datexp']
    blk = ndb['blk']

    # Load neural data
    spk = load_spk(mname, datexp, blk, root=os.path.join(DATA_ROOT, 'spk'))
    n_neurons, n_frames_neural = spk.shape

    # Load behavior
    beh = load_behavior_for_session(session_key, exp_type, ndb)
    n_frames_beh = len(beh['ft'])

    # Compute actual frame duration from timestamps
    ft = beh['ft']
    dt_days = np.nanmedian(np.diff(ft))
    dt_sec = dt_days * 24 * 3600  # convert days to seconds

    # Use minimum of neural and behavior frames
    n_frames = min(n_frames_neural, n_frames_beh)

    # Load retinotopy for brain region mapping
    iarea = load_retino(mname, datexp, root=os.path.join(DATA_ROOT, 'retinotopy'))
    region_idx = iarea_to_region_idx(iarea)

    # Verify neuron count matches
    assert len(region_idx) == n_neurons, \
        f"Retinotopy ({len(region_idx)}) != neural ({n_neurons}) for {session_key}"

    ntrials = beh['ntrials']

    # Get frame indices for each trial
    start_frs = np.round(beh['StartFr']).astype(int)
    gray_frs = np.round(beh['GrayFr']).astype(int)
    sound_frs = beh['SoundFr']  # keep as float for precise time computation

    # Build trial data
    neural_trials = []
    input_trials = []
    output_trials = []

    # Collect all stimulus names across sessions for consistent encoding
    wall_names = beh['WallName']
    is_rew = beh['isRew']

    # Get position and speed for each frame
    ft_pos = beh['ft_Pos'][:n_frames]
    ft_run_speed = beh['ft_RunSpeed'][:n_frames]

    skipped = 0
    for trial_idx in range(ntrials):
        sfr = start_frs[trial_idx]
        gfr = gray_frs[trial_idx]

        # Validate frame range
        if sfr < 0 or gfr > n_frames or sfr >= gfr:
            skipped += 1
            continue

        n_trial_frames = gfr - sfr
        if n_trial_frames < 2:
            skipped += 1
            continue

        # Neural data: (n_neurons, n_timepoints)
        trial_neural = spk[:, sfr:gfr].astype(np.float16)

        # === INPUTS ===
        # Time to sound cue (positive before cue, negative after)
        frame_indices = np.arange(sfr, gfr, dtype=np.float64)
        time_to_sound = (sound_frs[trial_idx] - frame_indices) * dt_sec

        # Day of training (scalar, broadcast to per-trial)
        day_val = np.float32(day_of_training)

        # Time since trial start
        time_since_start = (frame_indices - sfr) * dt_sec

        # Reward availability
        rew_val = np.float32(1.0 if is_rew[trial_idx] else 0.0)

        # Stack inputs: (4, n_timepoints) for time-varying, or (4,) for mixed
        # time_to_sound and time_since_start are time-varying
        # day_of_training and reward_availability are per-trial (scalar)
        # Use (4, n_timepoints) with broadcast for per-trial values
        trial_input = np.stack([
            time_to_sound.astype(np.float32),
            np.full(n_trial_frames, day_val, dtype=np.float32),
            time_since_start.astype(np.float32),
            np.full(n_trial_frames, rew_val, dtype=np.float32),
        ], axis=0)  # shape: (4, n_timepoints)

        # === OUTPUTS ===
        # Visual stimulus category (per-trial) -> index
        stim_name = wall_names[trial_idx]

        # Licking (binary, time-varying)
        lick_binary = build_lick_binary(beh, sfr, gfr)

        # Position (4 bins, time-varying)
        trial_pos = ft_pos[sfr:gfr]
        pos_binned = discretize_position(trial_pos, POSITION_BINS).astype(np.float32)

        # Running speed (4 quartile bins, time-varying)
        trial_speed = ft_run_speed[sfr:gfr]
        speed_binned = discretize_speed(trial_speed, speed_quartiles).astype(np.float32)

        # Stack outputs: (4, n_timepoints) as int
        trial_output = np.stack([
            np.full(n_trial_frames, -1, dtype=int),  # placeholder for stim, filled later
            lick_binary.astype(int),
            pos_binned.astype(int),
            speed_binned.astype(int),
        ], axis=0)  # shape: (4, n_timepoints)

        neural_trials.append(trial_neural)
        input_trials.append(trial_input)
        output_trials.append((trial_output, stim_name))

    t1 = time.time()
    if skipped > 0:
        print(f"  {session_key}: skipped {skipped}/{ntrials} trials")
    print(f"  {session_key}: {len(neural_trials)} trials, {n_neurons} neurons, {t1-t0:.1f}s")

    result = {
        'neural': neural_trials,
        'input': input_trials,
        'output': output_trials,
        'region_idx': region_idx,
        'subject': mname,
        'session_key': session_key,
        'dt_sec': dt_sec,
        'n_neurons': n_neurons,
    }
    if show_processing:
        result['beh'] = beh
        result['start_frs'] = start_frs
        result['gray_frs'] = gray_frs

    # Free large arrays
    del spk
    gc.collect()

    return result


def collect_all_corridor_speeds(session_map, max_sessions=None):
    """Collect all running speeds from corridor frames across sessions for quartile computation."""
    print("Collecting running speeds for quartile computation...")
    all_speeds = []

    sessions = list(session_map.items())
    if max_sessions:
        sessions = sessions[:max_sessions]

    for sess_key, (exp_type, ndb) in sessions:
        beh = load_behavior_for_session(sess_key, exp_type, ndb)
        n_frames = len(beh['ft'])

        # Get corridor frames
        corr_mask = beh['ft_CorrSpc'][:n_frames]
        speeds = beh['ft_RunSpeed'][:n_frames]
        all_speeds.append(speeds[corr_mask])

    all_speeds = np.concatenate(all_speeds)
    return all_speeds


def build_stimulus_mapping(session_results):
    """Build consistent stimulus name to index mapping across all sessions."""
    all_stim_names = set()
    for result in session_results:
        for (_, stim_name) in result['output']:
            all_stim_names.add(stim_name)

    # Sort for consistency
    stim_names = sorted(all_stim_names)
    stim_to_idx = {name: idx for idx, name in enumerate(stim_names)}
    return stim_names, stim_to_idx


def plot_processing(result, session_idx, speed_quartiles, stim_to_idx):
    """Plot processing steps for visual verification."""
    beh = result['beh']
    if beh is None:
        return

    session_key = result['session_key']
    fig, axes = plt.subplots(4, 3, figsize=(20, 16))
    fig.suptitle(f'Processing: {session_key}', fontsize=14)

    # Pick 3 sample trials
    n_trials = len(result['neural'])
    sample_indices = np.linspace(0, n_trials - 1, min(3, n_trials)).astype(int)

    for col, trial_idx in enumerate(sample_indices):
        neural = result['neural'][trial_idx]
        inp = result['input'][trial_idx]
        out_data = result['output'][trial_idx]
        stim_idx_val = int(out_data[0, 0])
        stim_name = [k for k, v in stim_to_idx.items() if v == stim_idx_val][0] if stim_to_idx else f"stim_{stim_idx_val}"

        n_frames = neural.shape[1]
        time_axis = np.arange(n_frames) * result['dt_sec']

        # Row 0: Neural activity (sample neurons)
        ax = axes[0, col]
        n_show = min(20, neural.shape[0])
        ax.imshow(neural[:n_show], aspect='auto', interpolation='none',
                  extent=[0, time_axis[-1], n_show, 0])
        ax.set_title(f'Trial {trial_idx}: {stim_name}')
        ax.set_ylabel('Neurons' if col == 0 else '')

        # Row 1: Inputs
        ax = axes[1, col]
        ax.plot(time_axis, inp[0], label='time_to_sound')
        ax.plot(time_axis, inp[2], label='time_since_start')
        ax.axhline(y=inp[1, 0], color='g', ls='--', label=f'day={inp[1,0]:.0f}')
        ax.axhline(y=inp[3, 0], color='r', ls='--', label=f'rew={inp[3,0]:.0f}')
        ax.set_ylabel('Input values' if col == 0 else '')
        ax.legend(fontsize=6)

        # Row 2: Outputs (position and speed)
        ax = axes[2, col]
        ax.plot(time_axis, out_data[2], label='position_bin', alpha=0.7)
        ax.plot(time_axis, out_data[3], label='speed_bin', alpha=0.7)
        ax.plot(time_axis, out_data[1], label='licking', alpha=0.7)
        ax.set_ylabel('Output values' if col == 0 else '')
        ax.legend(fontsize=6)

        # Row 3: Raw position and speed
        ax = axes[3, col]
        sfr = result['start_frs'][trial_idx] if trial_idx < len(result['start_frs']) else 0
        gfr = result['gray_frs'][trial_idx] if trial_idx < len(result['gray_frs']) else 0
        if sfr < len(beh['ft_Pos']) and gfr <= len(beh['ft_Pos']):
            raw_pos = beh['ft_Pos'][sfr:gfr]
            raw_speed = beh['ft_RunSpeed'][sfr:gfr]
            ax.plot(time_axis[:len(raw_pos)], raw_pos, label='raw_pos (dm)')
            ax2 = ax.twinx()
            ax2.plot(time_axis[:len(raw_speed)], raw_speed, color='orange', label='raw_speed')
            ax2.set_ylabel('Speed')
        ax.set_ylabel('Position (dm)' if col == 0 else '')
        ax.set_xlabel('Time (s)')

    plt.tight_layout()
    plt.savefig(f'processing_{session_key}.png', dpi=100)
    plt.close()
    print(f"  Saved processing_{session_key}.png")


def convert_data(output_file, sample_mode=False, show_processing=False):
    """Main conversion function."""
    total_start = time.time()

    print("=" * 60)
    print("Data Conversion: Zhong et al. 2025")
    print("=" * 60)

    # Get unique sessions
    session_map = get_unique_sessions()
    print(f"Found {len(session_map)} unique sessions")

    if sample_mode:
        # Take first 2 sessions
        keys = sorted(session_map.keys())[:2]
        session_map = {k: session_map[k] for k in keys}
        print(f"Sample mode: using {len(session_map)} sessions")

    # Compute day of training for each session
    all_session_map = get_unique_sessions()  # need full map for day computation
    day_map = compute_day_of_training(all_session_map)

    # Compute speed quartiles across all sessions (or sample)
    t0 = time.time()
    all_speeds = collect_all_corridor_speeds(
        session_map if sample_mode else all_session_map,
        max_sessions=None
    )
    speed_quartiles = compute_speed_bin_edges(all_speeds)
    print(f"Speed quartiles: {speed_quartiles} (computed in {time.time()-t0:.1f}s)")
    print(f"Speed range: [{all_speeds.min():.2f}, {all_speeds.max():.2f}]")

    # Process each session
    print("\nProcessing sessions...")
    session_results = []

    for sess_key, (exp_type, ndb) in sorted(session_map.items()):
        t0 = time.time()
        result = process_session(
            sess_key, exp_type, ndb,
            day_of_training=day_map[sess_key],
            speed_quartiles=speed_quartiles,
            show_processing=show_processing
        )
        session_results.append(result)
        gc.collect()
        sys.stdout.flush()

    # Build stimulus mapping
    stim_names, stim_to_idx = build_stimulus_mapping(session_results)
    print(f"\nStimulus categories ({len(stim_names)}): {stim_names}")

    # Fill in stimulus indices in output data
    for result in session_results:
        for i, (out_data, stim_name) in enumerate(result['output']):
            stim_idx = stim_to_idx[stim_name]
            out_data[0, :] = stim_idx
            result['output'][i] = out_data  # replace tuple with just the array

    # Generate processing plots if requested
    if show_processing:
        for i, result in enumerate(session_results[:2]):
            plot_processing(result, i, speed_quartiles, stim_to_idx)

    # Build subject list
    all_subjects = sorted(set(r['subject'] for r in session_results))
    subject_to_idx = {s: i for i, s in enumerate(all_subjects)}

    # Assemble final data structure
    print("\nAssembling final data structure...")

    neural_all = []
    input_all = []
    output_all = []
    subject_idx = []
    brain_region_idx = []

    for result in session_results:
        neural_all.append(result['neural'])
        input_all.append(result['input'])
        output_all.append(result['output'])
        subject_idx.append(subject_to_idx[result['subject']])
        brain_region_idx.append(result['region_idx'])

    # Define output values
    # Position bins: 0-1m, 1-2m, 2-3m, 3-4m
    pos_values = ['0-1m', '1-2m', '2-3m', '3-4m']

    # Speed bins: Q1, Q2, Q3, Q4
    speed_values = ['Q1_slowest', 'Q2', 'Q3', 'Q4_fastest']

    # Lick values: no_lick, lick
    lick_values = ['no_lick', 'lick']

    # Compute median dt across sessions
    median_dt = np.median([r['dt_sec'] for r in session_results])

    data = {
        'neural': neural_all,
        'input': input_all,
        'output': output_all,

        'subjects': all_subjects,
        'subject_idx': np.array(subject_idx, dtype=int),

        'brain_regions': BRAIN_REGIONS,
        'brain_region_idx': brain_region_idx,

        'input_names': ['time_to_sound_cue', 'day_of_training', 'time_since_trial_start', 'reward_availability'],
        'output_names': ['visual_stimulus', 'licking', 'position', 'running_speed'],
        'output_values': [
            stim_names,       # visual_stimulus values
            lick_values,      # licking values
            pos_values,       # position values
            speed_values,     # running_speed values
        ],

        'metadata': {
            'task_description': 'Visual discrimination task in virtual reality corridors with naturalistic textures. Mice discriminate between visual patterns (leaf/circle) and lick for water reward in the correct corridor.',
            'time_bin_size': float(median_dt * 1000),  # in ms
            'temporal_alignment_event': 'Trial start (corridor entry)',
            'off_start': 0.0,  # alignment is at corridor entry
            'off_end': None,  # variable trial length
            'frame_rate_hz': 1.0 / median_dt,
            'corridor_length_m': 4.0,
            'grey_space_m': 2.0,
            'speed_quartiles': speed_quartiles.tolist(),
            'stimulus_categories': stim_names,
            'n_sessions': len(session_results),
            'n_subjects': len(all_subjects),
            'source': 'Zhong et al. 2025, Nature',
        }
    }

    # Print summary statistics
    total_trials = sum(len(s) for s in data['neural'])
    total_neurons = sum(r['n_neurons'] for r in session_results)
    neurons_per_session = [r['n_neurons'] for r in session_results]
    trials_per_session = [len(s) for s in data['neural']]

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Sessions: {len(data['neural'])}")
    print(f"Subjects: {len(all_subjects)}")
    print(f"Total trials: {total_trials}")
    print(f"Trials/session: {np.mean(trials_per_session):.1f} (range: {min(trials_per_session)}-{max(trials_per_session)})")
    print(f"Total neurons: {total_neurons}")
    print(f"Neurons/session: {np.mean(neurons_per_session):.0f} (range: {min(neurons_per_session)}-{max(neurons_per_session)})")
    print(f"Time bin: {median_dt*1000:.1f} ms")
    print(f"Speed quartiles: {speed_quartiles}")
    print(f"Stimulus categories: {stim_names}")

    # Free intermediate results to reduce memory before saving
    del session_results
    gc.collect()

    # Save
    print(f"\nSaving to {output_file}...")
    sys.stdout.flush()
    t0 = time.time()
    with open(output_file, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    file_size = os.path.getsize(output_file)
    print(f"Saved {file_size / 1e9:.2f} GB in {time.time()-t0:.1f}s")

    total_time = time.time() - total_start
    print(f"\nTotal conversion time: {total_time:.1f}s ({total_time/60:.1f} min)")

    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert Zhong et al. 2025 data')
    parser.add_argument('output', type=str, help='Output pickle file path')
    parser.add_argument('--full', action='store_true', default=True, help='Process all sessions (default)')
    parser.add_argument('--sample', action='store_true', help='Process only 2 sessions for testing')
    parser.add_argument('--show-processing', action='store_true', help='Plot processing visualizations')

    args = parser.parse_args()

    sample_mode = args.sample
    if sample_mode:
        args.full = False

    convert_data(args.output, sample_mode=sample_mode, show_processing=args.show_processing)
