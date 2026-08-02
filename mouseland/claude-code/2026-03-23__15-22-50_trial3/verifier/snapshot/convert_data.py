#!/usr/bin/env python3
"""
Convert Zhong et al. 2025 data to decoder-compatible format.

Usage:
    python -u convert_data.py <outpicklefile> [--full|--sample] [--show-processing]

The data consists of 2-photon calcium imaging recordings from mouse visual cortex
during a VR corridor navigation task. Mice ran through corridors with different
visual textures and received water rewards in one corridor type.
"""

import os
import sys
import time
import argparse
import pickle
import numpy as np
from collections import defaultdict
import gc
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Constants and configuration
# =============================================================================
DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
CODE_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'code')

# Brain region mapping from iarea values (from utils.py: neu_area_ID)
AREA_MAP = {
    8: 'V1',
    0: 'mHV', 1: 'mHV', 2: 'mHV', 9: 'mHV',
    5: 'lHV', 6: 'lHV',
    3: 'aHV', 4: 'aHV',
    -1: 'other', 7: 'other',
}

BRAIN_REGIONS = ['V1', 'mHV', 'lHV', 'aHV', 'other']

# Position binning: 4 bins of 1m (10 dm) each over [0, 40) dm corridor
# Gray space (40-60 dm) goes to last bin
POS_BIN_EDGES = [0, 10, 20, 30, 60.01]  # last bin includes gray space
POS_BIN_LABELS = ['0-1m', '1-2m', '2-3m', '3m+']

# =============================================================================
# Data loading functions (adapted from reference code utils.py)
# =============================================================================

def load_spk(db):
    """Load neural data and concatenate planes. Returns (n_neurons, n_frames)."""
    fn = '%s_%s_%s_neural_data.npy' % (db['mname'], db['datexp'], db['blk'])
    spk_path = os.path.join(DATA_ROOT, 'spk', fn)
    spk = np.concatenate(
        [nspk for nspk in np.load(spk_path, allow_pickle=True).item()['spks']], 0
    )
    return spk


def load_retino(db):
    """Load retinotopy data. Returns iarea array."""
    fn = '%s_%s_trans.npz' % (db['mname'], db['datexp'])
    dtrans = np.load(os.path.join(DATA_ROOT, 'retinotopy', fn), allow_pickle=True)
    return dtrans['iarea']


def load_beh(exp_type):
    """Load behavior data for an experiment type."""
    fn = os.path.join(DATA_ROOT, 'beh', 'Beh_%s.npy' % exp_type)
    return np.load(fn, allow_pickle=True).item()


def get_beh_key(db):
    """Get the behavior dictionary key for a session."""
    if 'stimtype' in db and db['stimtype']:
        return '%s_%s_%s_%s' % (db['mname'], db['datexp'], db['blk'], db['stimtype'])
    return '%s_%s_%s' % (db['mname'], db['datexp'], db['blk'])


# =============================================================================
# Build session list
# =============================================================================

def build_session_list():
    """Build list of unique sessions with their metadata.

    Returns list of dicts with keys: mname, datexp, blk, exp_type, db_entry, beh_key
    """
    exp_info = np.load(
        os.path.join(DATA_ROOT, 'beh', 'Imaging_Exp_info.npy'), allow_pickle=True
    ).item()

    seen = set()
    sessions = []

    for exp_type in exp_info:
        for db in exp_info[exp_type]:
            key = (db['mname'], db['datexp'], db['blk'])
            if key in seen:
                continue
            seen.add(key)

            beh_key = get_beh_key(db)
            sessions.append({
                'mname': db['mname'],
                'datexp': db['datexp'],
                'blk': db['blk'],
                'exp_type': exp_type,
                'db_entry': db,
                'beh_key': beh_key,
                'session_key': key,
            })

    # Sort by mouse name, then date
    sessions.sort(key=lambda s: (s['mname'], s['datexp']))
    return sessions


def compute_training_days(sessions):
    """Compute ordinal training day for each session within each mouse."""
    mouse_sessions = defaultdict(list)
    for i, s in enumerate(sessions):
        mouse_sessions[s['mname']].append((s['datexp'], i))

    days = np.zeros(len(sessions), dtype=np.float32)
    for mname, sess_list in mouse_sessions.items():
        sess_list.sort(key=lambda x: x[0])  # sort by date
        for day_idx, (datexp, global_idx) in enumerate(sess_list):
            days[global_idx] = float(day_idx)

    return days


# =============================================================================
# Processing functions
# =============================================================================

def compute_speed_quartiles(sessions, beh_cache):
    """Compute global speed quartiles across all sessions."""
    print("Computing global speed quartiles...")
    all_speeds = []
    for s in sessions:
        beh = beh_cache[s['exp_type']][s['beh_key']]
        speeds = beh['ft_RunSpeed']
        all_speeds.append(speeds)

    all_speeds = np.concatenate(all_speeds)
    quartiles = np.percentile(all_speeds, [25, 50, 75])
    print(f"  Speed quartiles: {quartiles}")
    print(f"  Speed range: [{all_speeds.min():.1f}, {all_speeds.max():.1f}]")
    return quartiles


def get_all_stimuli(sessions, beh_cache):
    """Get sorted list of all unique stimulus names."""
    all_stim = set()
    for s in sessions:
        beh = beh_cache[s['exp_type']][s['beh_key']]
        for wn in beh['UniqWalls']:
            all_stim.add(str(wn))
    return sorted(all_stim)


def digitize_position(pos):
    """Discretize position into 4 bins of 1m (10dm) each.
    Bins: [0,10), [10,20), [20,30), [30,60+)
    """
    bins = np.digitize(pos, [10, 20, 30])
    return bins.astype(np.int64)


def digitize_speed(speed, quartiles):
    """Discretize running speed into 4 quartile bins."""
    bins = np.digitize(speed, quartiles)
    return bins.astype(np.int64)


def make_lick_vector(beh, start_fr, end_fr):
    """Create binary lick vector for frames [start_fr, end_fr).

    LickFr values are fractional frame indices.
    """
    n_frames = end_fr - start_fr
    lick_vec = np.zeros(n_frames, dtype=np.int64)

    lick_frs = beh['LickFr']
    # Round to nearest integer frame
    lick_frs_int = np.round(lick_frs).astype(int)
    # Filter to frames within this trial
    mask = (lick_frs_int >= start_fr) & (lick_frs_int < end_fr)
    if mask.any():
        trial_lick_frs = lick_frs_int[mask] - start_fr
        # Clip to valid range
        trial_lick_frs = np.clip(trial_lick_frs, 0, n_frames - 1)
        lick_vec[trial_lick_frs] = 1

    return lick_vec


def get_brain_region_idx(iarea):
    """Map iarea values to brain_region indices.

    BRAIN_REGIONS = ['V1', 'mHV', 'lHV', 'aHV', 'other']
    """
    region_to_idx = {r: i for i, r in enumerate(BRAIN_REGIONS)}
    # Vectorized mapping
    idx = np.full(len(iarea), region_to_idx['other'], dtype=np.int64)
    for area_val, region_name in AREA_MAP.items():
        mask = iarea == area_val
        idx[mask] = region_to_idx[region_name]
    return idx


def process_session(session, beh, training_day, speed_quartiles, stim_to_idx, frame_period):
    """Process a single session into decoder format.

    Returns:
        neural_trials: list of (n_neurons, n_timepoints) arrays
        input_trials: list of (4, n_timepoints) arrays
        output_trials: list of (4, n_timepoints) arrays
        brain_region_idx: (n_neurons,) array
    """
    t0 = time.time()

    # Load neural data
    spk = load_spk(session['db_entry'])
    n_neurons, n_total_frames = spk.shape

    # Load retinotopy
    iarea = load_retino(session['db_entry'])
    brain_reg_idx = get_brain_region_idx(iarea)

    # Verify consistency
    assert len(iarea) == n_neurons, \
        f"Neuron count mismatch: spk={n_neurons}, retinotopy={len(iarea)}"

    ntrials = beh['ntrials']
    StartFr = beh['StartFr'].astype(int)
    EndFr = beh['EndFr'].astype(int)
    SoundFr = beh['SoundFr']
    WallName = beh['WallName']
    isRew = beh['isRew']
    ft_Pos = beh['ft_Pos']
    ft_RunSpeed = beh['ft_RunSpeed']

    neural_trials = []
    input_trials = []
    output_trials = []

    skipped = 0
    for t in range(ntrials):
        start = StartFr[t]
        end = EndFr[t]

        # Skip if frames out of bounds
        if start < 0 or end > n_total_frames or end <= start:
            skipped += 1
            continue

        # Also clip end to available frames in beh arrays
        end = min(end, len(ft_Pos), len(ft_RunSpeed))
        if end <= start:
            skipped += 1
            continue

        n_tp = end - start
        if n_tp < 2:
            skipped += 1
            continue

        # --- Neural ---
        neural = spk[:, start:end].astype(np.float16)
        # Using float16 to reduce memory (decoder uses PCA, so precision is fine)

        # --- Inputs ---
        # 1. Time to sound cue (seconds): positive = time until cue, negative = time since cue
        sound_fr = SoundFr[t]
        frame_indices = np.arange(start, end, dtype=np.float64)
        time_to_sound = (sound_fr - frame_indices) * frame_period  # positive before, negative after

        # 2. Day of training (constant for trial)
        day = np.full(n_tp, training_day, dtype=np.float32)

        # 3. Time since trial start (seconds)
        time_since_start = (frame_indices - start) * frame_period

        # 4. Reward availability
        rew = np.full(n_tp, float(isRew[t]), dtype=np.float32)

        inp = np.stack([time_to_sound, day, time_since_start, rew], axis=0).astype(np.float32)

        # --- Outputs ---
        # 1. Visual stimulus category (per-trial, broadcast)
        stim_name = str(WallName[t])
        stim_idx = stim_to_idx[stim_name]
        stim_out = np.full(n_tp, stim_idx, dtype=np.int64)

        # 2. Licking (binary, time-varying)
        lick = make_lick_vector(beh, start, end)

        # 3. Position (4 bins, time-varying)
        pos = ft_Pos[start:end]
        pos_bin = digitize_position(pos)

        # 4. Running speed (4 bins, time-varying)
        speed = ft_RunSpeed[start:end]
        speed_bin = digitize_speed(speed, speed_quartiles)

        out = np.stack([stim_out, lick, pos_bin, speed_bin], axis=0).astype(np.int64)

        neural_trials.append(neural)
        input_trials.append(inp)
        output_trials.append(out)

    elapsed = time.time() - t0
    if skipped > 0:
        print(f"    Skipped {skipped}/{ntrials} trials")

    return neural_trials, input_trials, output_trials, brain_reg_idx, elapsed


# =============================================================================
# Visualization
# =============================================================================

def plot_processing(session, beh, neural_trials, input_trials, output_trials,
                    stim_names, speed_quartiles, session_idx):
    """Plot processing steps for visual verification."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n_trials_plot = min(5, len(neural_trials))
    fig, axes = plt.subplots(6, n_trials_plot, figsize=(4 * n_trials_plot, 18))
    if n_trials_plot == 1:
        axes = axes[:, np.newaxis]

    for t in range(n_trials_plot):
        neural = neural_trials[t]
        inp = input_trials[t]
        out = output_trials[t]
        n_tp = neural.shape[1]
        time_axis = np.arange(n_tp)

        # 1. Neural activity (mean across neurons)
        ax = axes[0, t]
        ax.plot(time_axis, neural.mean(axis=0), 'k', linewidth=0.5)
        ax.set_title(f'Trial {t}: {stim_names[int(out[0,0])]}')
        ax.set_ylabel('Mean neural')

        # 2. Input: time to sound cue
        ax = axes[1, t]
        ax.plot(time_axis, inp[0], 'b')
        ax.axhline(0, color='r', linestyle='--', alpha=0.5)
        ax.set_ylabel('Time to sound (s)')

        # 3. Input: time since start + reward availability
        ax = axes[2, t]
        ax.plot(time_axis, inp[2], 'g', label='Time since start')
        ax.axhline(inp[3, 0], color='orange', linestyle='--',
                   label=f'Reward={inp[3,0]:.0f}')
        ax.set_ylabel('Time (s)')
        ax.legend(fontsize=6)

        # 4. Output: licking
        ax = axes[3, t]
        ax.plot(time_axis, out[1], 'r', linewidth=0.8)
        ax.set_ylabel('Licking')
        ax.set_ylim(-0.1, 1.1)

        # 5. Output: position
        ax = axes[4, t]
        ax.plot(time_axis, out[2], 'm')
        ax.set_ylabel('Position bin')
        ax.set_ylim(-0.5, 3.5)

        # 6. Output: speed
        ax = axes[5, t]
        ax.plot(time_axis, out[3], 'c')
        ax.set_ylabel('Speed bin')
        ax.set_xlabel('Frame')
        ax.set_ylim(-0.5, 3.5)

    sid = f"{session['mname']}_{session['datexp']}_{session['blk']}"
    fig.suptitle(f'Session: {sid}', fontsize=14)
    fig.tight_layout()
    fn = f'processing_{sid}.png'
    fig.savefig(fn, dpi=100)
    plt.close(fig)
    print(f"  Saved {fn}")


# =============================================================================
# Main conversion
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Convert data to decoder format.')
    parser.add_argument('outfile', type=str, help='Output pickle file path')
    parser.add_argument('--full', action='store_true', default=True,
                       help='Process all sessions (default)')
    parser.add_argument('--sample', action='store_true',
                       help='Process only 2 sessions for testing')
    parser.add_argument('--show-processing', action='store_true',
                       help='Plot processing steps')
    args = parser.parse_args()

    if args.sample:
        args.full = False

    total_start = time.time()

    # Build session list
    print("Building session list...")
    sessions = build_session_list()
    print(f"  Found {len(sessions)} unique sessions")

    if args.sample:
        # Pick 2 sessions: one with rewards (sup), one without (unsup/naive)
        # to ensure we have licking and reward data in sample
        sup_sess = [s for s in sessions if 'sup' in s['exp_type'] and 'unsup' not in s['exp_type']]
        unsup_sess = [s for s in sessions if s not in sup_sess]
        sample = []
        if sup_sess:
            sample.append(sup_sess[0])
        if unsup_sess:
            sample.append(unsup_sess[0])
        sessions = sample[:2]
        print(f"  Sample mode: using {len(sessions)} sessions")

    # Compute training days
    training_days = compute_training_days(sessions)

    # Load all needed behavior files
    print("Loading behavior data...")
    beh_cache = {}
    exp_types_needed = set(s['exp_type'] for s in sessions)
    for et in exp_types_needed:
        beh_cache[et] = load_beh(et)
        print(f"  Loaded Beh_{et}.npy")

    # Get all stimuli
    all_stimuli = get_all_stimuli(sessions, beh_cache)
    stim_to_idx = {s: i for i, s in enumerate(all_stimuli)}
    print(f"  Stimuli ({len(all_stimuli)}): {all_stimuli}")

    # Compute speed quartiles
    speed_quartiles = compute_speed_quartiles(sessions, beh_cache)

    # Compute mean frame period
    print("Computing frame period...")
    sample_beh = beh_cache[sessions[0]['exp_type']][sessions[0]['beh_key']]
    ft = sample_beh['ft']
    dt = np.diff(ft) * 24 * 3600  # datenum to seconds
    frame_period = float(np.nanmedian(dt))
    print(f"  Frame period: {frame_period:.4f} s ({1/frame_period:.2f} Hz)")

    # Process sessions
    print(f"\nProcessing {len(sessions)} sessions...")
    neural_all = []
    input_all = []
    output_all = []
    subjects_list = []
    subject_idx_list = []
    brain_region_idx_all = []

    subjects = sorted(set(s['mname'] for s in sessions))
    subject_to_idx = {s: i for i, s in enumerate(subjects)}

    for i, session in enumerate(sessions):
        sid = f"{session['mname']}_{session['datexp']}_{session['blk']}"
        print(f"  [{i+1}/{len(sessions)}] {sid} ({session['exp_type']})...")

        beh = beh_cache[session['exp_type']][session['beh_key']]
        day = training_days[i]

        neural_trials, input_trials, output_trials, brain_reg_idx, elapsed = \
            process_session(session, beh, day, speed_quartiles, stim_to_idx, frame_period)

        if len(neural_trials) < 2:
            print(f"    WARNING: Only {len(neural_trials)} trials, skipping session")
            continue

        neural_all.append(neural_trials)
        input_all.append(input_trials)
        output_all.append(output_trials)
        subject_idx_list.append(subject_to_idx[session['mname']])
        brain_region_idx_all.append(brain_reg_idx)

        n_neurons = neural_trials[0].shape[0]
        n_trials = len(neural_trials)
        mean_tp = np.mean([t.shape[1] for t in neural_trials])
        neural_mb = sum(t.nbytes for t in neural_trials) / 1e6
        print(f"    {n_trials} trials, {n_neurons} neurons, mean {mean_tp:.0f} frames/trial, {neural_mb:.0f}MB, {elapsed:.1f}s")

        if args.show_processing and i < 2:
            plot_processing(session, beh, neural_trials, input_trials, output_trials,
                          all_stimuli, speed_quartiles, i)

        # Force garbage collection periodically
        if (i + 1) % 10 == 0:
            gc.collect()
            total_neural_mb = sum(sum(t.nbytes for t in s) for s in neural_all) / 1e6
            print(f"    [Memory] Total neural so far: {total_neural_mb/1000:.1f} GB")

    # Build output dictionary
    print("\nBuilding output dictionary...")

    # Speed bin labels
    speed_labels = ['Q1_slow', 'Q2', 'Q3', 'Q4_fast']

    data = {
        'neural': neural_all,
        'input': input_all,
        'output': output_all,

        'subjects': subjects,
        'subject_idx': np.array(subject_idx_list, dtype=np.int64),

        'brain_regions': BRAIN_REGIONS,
        'brain_region_idx': brain_region_idx_all,

        'input_names': ['time_to_sound_cue', 'day_of_training',
                       'time_since_trial_start', 'reward_availability'],
        'output_names': ['visual_stimulus', 'licking', 'position', 'running_speed'],
        'output_values': [
            all_stimuli,                           # visual stimulus categories
            ['no_lick', 'lick'],                   # licking
            POS_BIN_LABELS,                        # position bins
            speed_labels,                          # speed bins
        ],

        'metadata': {
            'task_description': 'VR corridor navigation with visual texture discrimination and reward learning. '
                              'Mice ran through corridors with different textures (leaf, circle, rock, wood). '
                              'Sound cue indicated reward zone in rewarded corridor.',
            'time_bin_size': frame_period * 1000,  # in ms
            'temporal_alignment_event': 'Trial start (corridor entry)',
            'off_start': 0.0,  # trial starts at alignment event
            'off_end': None,   # variable trial length
            'frame_rate_hz': 1.0 / frame_period,
            'corridor_length_m': 6.0,
            'texture_length_m': 4.0,
            'gray_space_length_m': 2.0,
            'vr_speed_cm_s': 60.0,
            'speed_quartiles': speed_quartiles.tolist(),
            'position_bin_edges_dm': POS_BIN_EDGES,
            'n_sessions': len(neural_all),
            'n_subjects': len(subjects),
            'total_trials': sum(len(s) for s in neural_all),
            'total_neurons_per_session': [s[0].shape[0] for s in neural_all],
            'data_source': 'Zhong et al. 2025 - Unsupervised pretraining in biological neural networks',
            'neural_data_type': 'Suite2p deconvolved calcium traces (0.75s decay timescale)',
        },
    }

    # Print summary
    print(f"\n{'='*60}")
    print(f"CONVERSION SUMMARY")
    print(f"{'='*60}")
    print(f"Sessions: {len(neural_all)}")
    print(f"Subjects: {len(subjects)} ({subjects})")
    print(f"Total trials: {sum(len(s) for s in neural_all)}")
    total_neurons = sum(s[0].shape[0] for s in neural_all)
    print(f"Total neurons (sum across sessions): {total_neurons}")
    print(f"Mean neurons/session: {total_neurons/len(neural_all):.0f}")
    print(f"Mean trials/session: {sum(len(s) for s in neural_all)/len(neural_all):.0f}")
    print(f"Frame period: {frame_period*1000:.1f} ms")
    print(f"Stimuli: {all_stimuli}")
    print(f"Speed quartiles: {speed_quartiles}")

    # Free behavior cache before saving
    del beh_cache
    gc.collect()

    # Save
    print(f"\nSaving to {args.outfile}...", flush=True)
    with open(args.outfile, 'wb') as f:
        pickle.dump(data, f, protocol=5)

    file_size = os.path.getsize(args.outfile) / (1024**3)
    total_elapsed = time.time() - total_start
    print(f"  File size: {file_size:.2f} GB")
    print(f"  Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")


if __name__ == '__main__':
    main()
