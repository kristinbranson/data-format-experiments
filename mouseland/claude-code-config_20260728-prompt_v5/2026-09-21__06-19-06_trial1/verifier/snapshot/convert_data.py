#!/usr/bin/env python3
"""
Convert Zhong et al. 2025 VR corridor calcium imaging data to decoder-compatible format.

Usage:
    python -u convert_data.py <outfile> [--sample] [--full] [--show-processing]
"""

import numpy as np
import pickle
import os
import sys
import argparse
import time
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_ROOT = '/app/data'
FRAME_RATE = 3.17  # Hz, calcium imaging frame rate

# ---- Brain region mapping (from reference utils.py: neu_area_ID) ----
# iarea codes: V1=8, mHV={0,1,2,9}, lHV={5,6}, aHV={3,4}, excluded={-1,7}
AREA_MAP = {
    8: 0,   # V1
    0: 1, 1: 1, 2: 1, 9: 1,  # mHV
    5: 2, 6: 2,  # lHV
    3: 3, 4: 3,  # aHV
}
BRAIN_REGIONS = ['V1', 'mHV', 'lHV', 'aHV']
EXCLUDED_AREAS = {-1, 7}

# ---- Stimulus mapping ----
ALL_STIMULI = sorted([
    'circle1', 'circle2', 'circle3',
    'leaf1', 'leaf1_swap1', 'leaf1_swap2', 'leaf2', 'leaf3',
    'rock1', 'rock2',
    'wood1', 'wood1_swap1', 'wood1_swap2', 'wood2', 'wood5',
])
STIM_TO_IDX = {s: i for i, s in enumerate(ALL_STIMULI)}


def load_spk(mname, datexp, blk, root=DATA_ROOT):
    """Load neural data (deconvolved traces), concatenate across planes."""
    fn = f'{mname}_{datexp}_{blk}_neural_data.npy'
    path = os.path.join(root, 'spk', fn)
    data = np.load(path, allow_pickle=True).item()
    spk = np.concatenate([nspk for nspk in data['spks']], 0)
    return spk


def load_area_ids(mname, datexp, root=DATA_ROOT):
    """Load brain area assignment for each neuron."""
    fn = f'{mname}_{datexp}_trans.npz'
    path = os.path.join(root, 'retinotopy', fn)
    trans = np.load(path, allow_pickle=True)
    return trans['iarea']


def get_neuron_mask_and_regions(iarea):
    """Return boolean mask of included neurons and their brain region indices."""
    iarea_int = iarea.astype(int)
    mask = np.array([a not in EXCLUDED_AREAS for a in iarea_int])
    region_idx = np.array([AREA_MAP[int(a)] for a in iarea_int[mask]], dtype=np.int64)
    return mask, region_idx


def build_session_list(exp_info):
    """Build deduplicated list of unique sessions with metadata."""
    session_dict = {}
    for exp_type, db_list in exp_info.items():
        for ndb in db_list:
            key = f"{ndb['mname']}_{ndb['datexp']}_{ndb['blk']}"
            if key not in session_dict:
                session_dict[key] = {
                    'key': key,
                    'mname': ndb['mname'],
                    'datexp': ndb['datexp'],
                    'blk': ndb['blk'],
                    'exp_types': [],
                }
            session_dict[key]['exp_types'].append(exp_type)

    sessions = list(session_dict.values())
    sessions.sort(key=lambda s: (s['mname'], s['datexp']))

    # Compute chronological day index per mouse
    mouse_sessions = {}
    for s in sessions:
        m = s['mname']
        if m not in mouse_sessions:
            mouse_sessions[m] = []
        mouse_sessions[m].append(s)
    for m, sess_list in mouse_sessions.items():
        sess_list.sort(key=lambda s: s['datexp'])
        for i, s in enumerate(sess_list):
            s['day_index'] = i

    return sessions


def load_all_behavior(exp_info):
    """Load all behavior files, deduplicate by session key.

    Some experiment types (test3) use suffixed keys like 'mouse_date_blk_swap1'.
    We store these under the base key (mouse_date_blk) since the behavior data
    is identical between swap variants.
    """
    all_beh = {}
    for exp_type in exp_info.keys():
        beh_path = os.path.join(DATA_ROOT, 'beh', f'Beh_{exp_type}.npy')
        if not os.path.exists(beh_path):
            continue
        beh = np.load(beh_path, allow_pickle=True).item()
        for k, v in beh.items():
            # Strip swap suffixes to get base session key
            base_key = k
            for suffix in ('_swap1', '_swap2'):
                if base_key.endswith(suffix):
                    base_key = base_key[:-len(suffix)]
                    break
            if base_key not in all_beh:
                all_beh[base_key] = v
    return all_beh


def extract_trial_frames(beh, nfr):
    """For each trial, get the corridor frame indices."""
    ft_trInd = beh['ft_trInd'][:nfr]
    ft_CorrSpc = beh['ft_CorrSpc'][:nfr]
    ntrials = beh['ntrials']

    trials = []
    for n in range(ntrials):
        frames = np.where((ft_trInd == n) & ft_CorrSpc)[0]
        if len(frames) >= 2:
            trials.append(frames)
        else:
            trials.append(None)
    return trials


def build_lick_array_vectorized(beh, trial_frames_list, nfr):
    """Build binary lick array for each trial using vectorized lookup."""
    lick_fr = beh['LickFr'].astype(int) if len(beh['LickFr']) > 0 else np.array([], dtype=int)
    lick_trind = beh['LickTrind'].astype(int) if len(beh['LickTrind']) > 0 else np.array([], dtype=int)

    # Build frame-level lick indicator
    lick_indicator = np.zeros(nfr, dtype=np.int64)
    valid_mask = (lick_fr >= 0) & (lick_fr < nfr)
    if valid_mask.any():
        lick_indicator[lick_fr[valid_mask]] = 1

    lick_arrays = []
    for n, frames in enumerate(trial_frames_list):
        if frames is None:
            lick_arrays.append(None)
        else:
            lick_arrays.append(lick_indicator[frames].copy())
    return lick_arrays


def collect_speeds_from_behavior(sessions, all_beh):
    """Pass 1: Collect corridor running speeds from behavior data only (no neural loading)."""
    all_speeds = []
    session_info = []
    for i, sess in enumerate(sessions):
        key = sess['key']
        beh = all_beh[key]
        # We need nfr to truncate. Estimate from ft length or use a large value.
        # Actually we need to match the neural frame count. Load just the shape.
        mname, datexp, blk = sess['mname'], sess['datexp'], sess['blk']
        fn = f'{mname}_{datexp}_{blk}_neural_data.npy'
        path = os.path.join(DATA_ROOT, 'spk', fn)
        # Load just to get shape - this is fast since we only need spks[0].shape
        data = np.load(path, allow_pickle=True).item()
        nfr = data['spks'][0].shape[1]
        nneu_total = sum(s.shape[0] for s in data['spks'])
        del data

        ft_RunSpeed = beh['ft_RunSpeed'][:nfr]
        ft_trInd = beh['ft_trInd'][:nfr]
        ft_CorrSpc = beh['ft_CorrSpc'][:nfr]

        # Collect all corridor speeds
        corr_mask = ft_CorrSpc
        if corr_mask.any():
            all_speeds.append(ft_RunSpeed[corr_mask])

        # Compute dt
        ft = beh['ft'][:nfr]
        dt_sec = np.median(np.diff(ft)) * 24 * 3600 if len(ft) > 1 else 1.0/FRAME_RATE

        # Count valid trials
        ntrials = beh['ntrials']
        n_valid = 0
        for n in range(ntrials):
            frames = np.where((ft_trInd == n) & ft_CorrSpc)[0]
            if len(frames) >= 2:
                n_valid += 1

        # Get iarea to count included neurons
        iarea = load_area_ids(mname, datexp)
        n_included = sum(1 for a in iarea.astype(int) if a not in EXCLUDED_AREAS)

        session_info.append({
            'n_included': n_included,
            'nneu_total': nneu_total,
            'ntrials': ntrials,
            'ntrials_valid': n_valid,
            'nfr': nfr,
            'dt_sec': dt_sec,
        })
        print(f"  [{i+1}/{len(sessions)}] {key}: {n_included}/{nneu_total} neurons, "
              f"{n_valid}/{ntrials} trials, dt={dt_sec:.4f}s")

    return all_speeds, session_info


def process_session_full(sess, all_beh, speed_quantiles):
    """Process one session: load neural data, build trial arrays."""
    key = sess['key']
    mname, datexp, blk = sess['mname'], sess['datexp'], sess['blk']

    # Load neural data
    spk = load_spk(mname, datexp, blk)
    nneu, nfr = spk.shape

    # Filter neurons by brain area
    iarea = load_area_ids(mname, datexp)
    neuron_mask, region_idx = get_neuron_mask_and_regions(iarea)
    spk_filtered = spk[neuron_mask]
    n_included = neuron_mask.sum()
    del spk  # free memory

    # Load behavior
    beh = all_beh[key]
    ntrials = beh['ntrials']
    ft_Pos = beh['ft_Pos'][:nfr]
    ft_RunSpeed = beh['ft_RunSpeed'][:nfr]
    SoundFr = beh['SoundFr']
    isRew = beh['isRew']
    WallName = beh['WallName']

    # Frame interval
    ft = beh['ft'][:nfr]
    dt_sec = np.median(np.diff(ft)) * 24 * 3600 if len(ft) > 1 else 1.0/FRAME_RATE

    # Extract trial frames
    trial_frames_list = extract_trial_frames(beh, nfr)

    # Build lick arrays (vectorized)
    lick_arrays = build_lick_array_vectorized(beh, trial_frames_list, nfr)

    # Build per-trial data
    neural_trials = []
    input_trials = []
    output_trials = []

    for n in range(ntrials):
        frames = trial_frames_list[n]
        if frames is None:
            continue

        T = len(frames)

        # Neural: (n_neurons, T), float32
        neural = spk_filtered[:, frames].astype(np.float32)

        # --- Inputs (4, T) ---
        input_arr = np.empty((4, T), dtype=np.float32)
        input_arr[0, :] = (frames - SoundFr[n]) * dt_sec     # time to sound cue
        input_arr[1, :] = float(sess['day_index'])             # day of training
        input_arr[2, :] = (frames - frames[0]) * dt_sec       # time since trial start
        input_arr[3, :] = 1.0 if isRew[n] else 0.0            # reward availability

        # --- Outputs (4, T) ---
        output_arr = np.empty((4, T), dtype=np.int64)
        output_arr[0, :] = STIM_TO_IDX[str(WallName[n])]      # visual stimulus
        output_arr[1, :] = lick_arrays[n]                      # licking
        output_arr[2, :] = np.clip(ft_Pos[frames] // 10, 0, 3).astype(np.int64)  # position bin
        output_arr[3, :] = np.digitize(ft_RunSpeed[frames], speed_quantiles).astype(np.int64)  # speed bin

        neural_trials.append(neural)
        input_trials.append(input_arr)
        output_trials.append(output_arr)

    return {
        'neural': neural_trials,
        'input': input_trials,
        'output': output_trials,
        'region_idx': region_idx,
        'n_included': n_included,
        'ntrials_valid': len(neural_trials),
        'dt_sec': dt_sec,
    }


def plot_processing(sess, all_beh, speed_quantiles, save_path):
    """Plot processing steps for visual verification."""
    key = sess['key']
    mname, datexp, blk = sess['mname'], sess['datexp'], sess['blk']

    spk = load_spk(mname, datexp, blk)
    nneu, nfr = spk.shape
    iarea = load_area_ids(mname, datexp)
    neuron_mask, _ = get_neuron_mask_and_regions(iarea)
    spk_filtered = spk[neuron_mask]
    del spk

    beh = all_beh[key]
    ft_Pos = beh['ft_Pos'][:nfr]
    ft_RunSpeed = beh['ft_RunSpeed'][:nfr]
    SoundFr = beh['SoundFr']
    isRew = beh['isRew']
    WallName = beh['WallName']

    trial_frames_list = extract_trial_frames(beh, nfr)
    lick_arrays = build_lick_array_vectorized(beh, trial_frames_list, nfr)

    ft = beh['ft'][:nfr]
    dt_sec = np.median(np.diff(ft)) * 24 * 3600 if len(ft) > 1 else 1.0/FRAME_RATE

    valid_trials = [n for n in range(beh['ntrials']) if trial_frames_list[n] is not None]
    plot_trials = valid_trials[:3]

    fig, axes = plt.subplots(len(plot_trials), 5, figsize=(25, 4*len(plot_trials)))
    if len(plot_trials) == 1:
        axes = axes[np.newaxis, :]

    for row, trial_n in enumerate(plot_trials):
        frames = trial_frames_list[trial_n]
        T = len(frames)
        t = np.arange(T) * dt_sec

        # Neural activity (5 random neurons)
        ax = axes[row, 0]
        n_plot = min(5, spk_filtered.shape[0])
        idx = np.random.choice(spk_filtered.shape[0], n_plot, replace=False)
        for i, ni in enumerate(idx):
            ax.plot(t, spk_filtered[ni, frames] + i*2, linewidth=0.5)
        ax.set_title(f'Trial {trial_n}: Neural ({spk_filtered.shape[0]} neu)')
        ax.set_xlabel('Time (s)')

        # Position
        ax = axes[row, 1]
        pos = ft_Pos[frames]
        pos_bin = np.clip(pos // 10, 0, 3).astype(int)
        ax.plot(t, pos, 'b-')
        ax2 = ax.twinx()
        ax2.plot(t, pos_bin, 'r-', alpha=0.5)
        ax2.set_ylabel('Bin', color='r')
        ax.set_title(f'Position (stim={WallName[trial_n]})')

        # Speed
        ax = axes[row, 2]
        speed = ft_RunSpeed[frames]
        speed_bin = np.digitize(speed, speed_quantiles).astype(int)
        ax.plot(t, speed, 'b-')
        ax2 = ax.twinx()
        ax2.plot(t, speed_bin, 'r-', alpha=0.5)
        for q in speed_quantiles:
            ax.axhline(q, color='gray', linestyle='--', alpha=0.3)
        ax.set_title(f'Speed (rew={isRew[trial_n]})')

        # Licking
        ax = axes[row, 3]
        ax.plot(t, lick_arrays[trial_n], 'k-', linewidth=0.5)
        ax.set_title('Licking')
        ax.set_ylim(-0.1, 1.1)

        # Time to sound cue
        ax = axes[row, 4]
        ax.plot(t, (frames - SoundFr[trial_n]) * dt_sec)
        ax.axhline(0, color='r', linestyle='--', alpha=0.5)
        ax.set_title('Time to cue (s)')

    fig.suptitle(f'Session: {key}', fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=100)
    plt.close(fig)
    print(f'  Processing plot saved to {save_path}')


def main():
    parser = argparse.ArgumentParser(description='Convert VR corridor data to decoder format')
    parser.add_argument('outfile', help='Output pickle file path')
    parser.add_argument('--sample', action='store_true', help='Process only 2 sessions')
    parser.add_argument('--full', action='store_true', help='Process all sessions (default)')
    parser.add_argument('--show-processing', action='store_true', help='Generate processing plots')
    args = parser.parse_args()

    t0 = time.time()

    print("Loading experiment info and behavior data...")
    exp_info = np.load(os.path.join(DATA_ROOT, 'beh', 'Imaging_Exp_info.npy'), allow_pickle=True).item()
    all_beh = load_all_behavior(exp_info)
    sessions = build_session_list(exp_info)

    if args.sample:
        sessions = sessions[:2]

    n_mice = len(set(s['mname'] for s in sessions))
    print(f"Processing {len(sessions)} sessions across {n_mice} mice")

    # --- Pass 1: Collect corridor speeds for quartile computation ---
    print("\n--- Pass 1: Collecting corridor speeds ---")
    t1 = time.time()
    all_speeds, session_info = collect_speeds_from_behavior(sessions, all_beh)

    all_speeds_flat = np.concatenate(all_speeds)
    speed_quantiles = np.percentile(all_speeds_flat, [25, 50, 75])
    del all_speeds, all_speeds_flat
    print(f"\nSpeed quartile boundaries: {speed_quantiles}")
    print(f"Pass 1 time: {time.time()-t1:.1f}s")

    # --- Pass 2: Build full dataset ---
    print("\n--- Pass 2: Building dataset ---")
    t2 = time.time()

    neural_all = []
    input_all = []
    output_all = []
    brain_region_idx_all = []
    subject_list = sorted(set(s['mname'] for s in sessions))
    subject_idx_list = []

    for i, sess in enumerate(sessions):
        ts = time.time()
        result = process_session_full(sess, all_beh, speed_quantiles)

        neural_all.append(result['neural'])
        input_all.append(result['input'])
        output_all.append(result['output'])
        brain_region_idx_all.append(result['region_idx'])
        subject_idx_list.append(subject_list.index(sess['mname']))

        te = time.time()
        print(f"  [{i+1}/{len(sessions)}] {sess['key']}: {result['ntrials_valid']} trials, "
              f"{result['n_included']} neurons, time={te-ts:.1f}s")

        if args.show_processing and i < 2:
            plot_processing(sess, all_beh, speed_quantiles,
                          f'processing_{sess["key"]}.png')

    print(f"Pass 2 time: {time.time()-t2:.1f}s")

    # Compute median time bin
    dt_secs = [si['dt_sec'] for si in session_info]
    median_dt = np.median(dt_secs)

    # --- Build output dictionary ---
    input_names = ['time_to_sound_cue', 'day_of_training', 'time_since_trial_start', 'reward_availability']
    output_names = ['visual_stimulus', 'licking', 'position_bin', 'speed_bin']
    output_values = [
        ALL_STIMULI,
        ['no_lick', 'lick'],
        ['0-1m', '1-2m', '2-3m', '3-4m'],
        ['Q1', 'Q2', 'Q3', 'Q4'],
    ]

    data = {
        'neural': neural_all,
        'input': input_all,
        'output': output_all,
        'subjects': subject_list,
        'subject_idx': np.array(subject_idx_list, dtype=np.int64),
        'brain_regions': BRAIN_REGIONS,
        'brain_region_idx': brain_region_idx_all,
        'input_names': input_names,
        'output_names': output_names,
        'output_values': output_values,
        'metadata': {
            'task_description': 'Visual discrimination in VR corridor: mice discriminate textures (leaf/circle/rock/wood) with reward conditioning. Decode stimulus, licking, position, speed from V1+HVA calcium imaging.',
            'time_bin_size': median_dt * 1000,  # in ms
            'temporal_alignment_event': 'Corridor entry (trial start)',
            'off_start': 0.0,
            'off_end': None,
            'frame_rate_hz': FRAME_RATE,
            'speed_quartile_boundaries': speed_quantiles.tolist(),
            'n_sessions': len(sessions),
            'session_keys': [s['key'] for s in sessions],
        }
    }

    # --- Save ---
    print(f"\nSaving to {args.outfile}...")
    with open(args.outfile, 'wb') as f:
        pickle.dump(data, f, protocol=4)

    file_size = os.path.getsize(args.outfile) / (1024**3)
    print(f"Saved: {file_size:.2f} GB")

    # --- Summary ---
    total_trials = sum(len(s) for s in neural_all)
    neurons_per_session = [neural_all[i][0].shape[0] if len(neural_all[i]) > 0 else 0 for i in range(len(neural_all))]
    print(f"\n--- Summary ---")
    print(f"Sessions: {len(sessions)}")
    print(f"Subjects: {len(subject_list)}")
    print(f"Total trials: {total_trials}")
    print(f"Neurons per session: min={min(neurons_per_session)}, max={max(neurons_per_session)}, mean={np.mean(neurons_per_session):.0f}")
    print(f"Time bin: {median_dt*1000:.1f} ms")
    print(f"Total time: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
