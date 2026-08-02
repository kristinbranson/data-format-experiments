#!/usr/bin/env python3
"""
Convert calcium imaging data from Zhong et al. 2025
("Unsupervised pretraining in biological neural networks")
to decoder-compatible format.

Usage:
    python -u convert_data.py <outpicklefile> [--full|--sample] [--show-processing]
"""

import numpy as np
import os
import sys
import pickle
import argparse
import time

# ============================================================
# Constants
# ============================================================
DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

# Brain area mapping from iarea codes (from utils.py neu_area_ID)
AREA_MAP = {
    8: 'V1',
    0: 'mHV', 1: 'mHV', 2: 'mHV', 9: 'mHV',
    5: 'lHV', 6: 'lHV',
    3: 'aHV', 4: 'aHV',
}
EXCLUDED_AREAS = {-1, 7}  # Outside visual cortex
BRAIN_REGIONS = ['V1', 'mHV', 'lHV', 'aHV']

FRAME_RATE = 3.17  # Hz
TIME_BIN_MS = 1000.0 / FRAME_RATE  # ~315.5 ms

# Stimulus category mapping - rock/wood/brick are equivalent to circle/leaf
STIM_CATEGORY_MAP = {
    'circle1': 'circle1', 'circle2': 'circle2',
    'leaf1': 'leaf1', 'leaf2': 'leaf2', 'leaf3': 'leaf3',
    'leaf1_swap1': 'leaf1_swap1', 'leaf1_swap2': 'leaf1_swap2',
    'rock1': 'circle1', 'rock2': 'circle2',
    'wood1': 'leaf1', 'wood2': 'leaf2',
    'wood1_swap1': 'leaf1_swap1', 'wood1_swap2': 'leaf1_swap2',
    'brick1': 'circle1', 'brick2': 'circle2',
    'brick5': 'leaf3',
    'wood5': 'leaf3',
    'rock5': 'circle3',
}


# ============================================================
# Loading Functions (matching utils.py)
# ============================================================

def load_spk_filtered(mname, datexp, blk, valid_mask):
    """Load neural data, filter by valid_mask, and return as float16.
    Filters per-plane before concatenation to reduce peak memory.
    """
    fn = f'{mname}_{datexp}_{blk}_neural_data.npy'
    spk_data = np.load(os.path.join(DATA_ROOT, 'spk', fn), allow_pickle=True).item()
    planes = spk_data['spks']
    # Split valid_mask across planes
    filtered = []
    offset = 0
    for plane in planes:
        n = plane.shape[0]
        plane_mask = valid_mask[offset:offset+n]
        filtered.append(plane[plane_mask].astype(np.float16))
        offset += n
    return np.concatenate(filtered, 0)


def load_retino(mname, datexp):
    """Load retinotopy iarea assignments."""
    fn = f'{mname}_{datexp}_trans.npz'
    dtrans = np.load(os.path.join(DATA_ROOT, 'retinotopy', fn), allow_pickle=True)
    return dtrans['iarea']


def get_brain_region_idx(iarea):
    """Get brain region index for neurons in visual cortex.
    Returns: (valid_mask, region_idx for valid neurons)
    """
    valid_mask = np.array([int(ia) not in EXCLUDED_AREAS for ia in iarea])
    region_idx = np.array([
        BRAIN_REGIONS.index(AREA_MAP.get(int(ia), 'V1'))
        for ia in iarea[valid_mask]
    ], dtype=np.int64)
    return valid_mask, region_idx


def build_session_map():
    """Build mapping: spk_key -> {exp_type, beh_key, db}"""
    exp_info = np.load(os.path.join(DATA_ROOT, 'beh', 'Imaging_Exp_info.npy'),
                       allow_pickle=True).item()
    session_map = {}
    for exp_type, db_list in exp_info.items():
        for ndb in db_list:
            spk_key = f"{ndb['mname']}_{ndb['datexp']}_{ndb['blk']}"
            beh_key = f"{spk_key}_{ndb['stimtype']}" if 'stimtype' in ndb else spk_key
            # Prefer entries without stimtype
            if spk_key not in session_map or 'stimtype' not in ndb:
                session_map[spk_key] = {
                    'exp_type': exp_type, 'beh_key': beh_key, 'db': dict(ndb)
                }
    return session_map


def load_beh(session_info):
    """Load behavior data for a session."""
    beh_all = np.load(
        os.path.join(DATA_ROOT, 'beh', f"Beh_{session_info['exp_type']}.npy"),
        allow_pickle=True
    ).item()
    return beh_all[session_info['beh_key']]


def get_session_day(db):
    """Extract day of training from session metadata."""
    for key in ['days', 'sess#']:
        if key in db:
            return int(db[key])
    return 0


def standardize_stim_name(name):
    """Map stimulus names to standard categories."""
    return STIM_CATEGORY_MAP.get(name, name)


# ============================================================
# Processing Functions
# ============================================================

def collect_speed_quartiles(session_map, keys):
    """Collect running speed quartiles from behavioral data only (no neural loading)."""
    all_speeds = []
    for spk_key in keys:
        info = session_map[spk_key]
        beh = load_beh(info)
        speeds = beh['ft_RunSpeed']
        all_speeds.append(speeds)
    all_speeds = np.concatenate(all_speeds)
    valid = all_speeds > 0
    if valid.sum() == 0:
        return np.array([1.0, 2.0, 3.0])
    return np.percentile(all_speeds[valid], [25, 50, 75])


def process_session(spk_key, session_info, speed_quartiles):
    """Process a single session. Returns dict or None if invalid."""
    db = session_info['db']
    mname, datexp, blk = db['mname'], db['datexp'], db['blk']

    # Load retinotopy first to build mask before loading large neural data
    iarea = load_retino(mname, datexp)
    valid_mask, region_idx = get_brain_region_idx(iarea)
    nneu_total = len(iarea)

    # Load neural data with filtering (reduces memory, avoids large intermediate array)
    spk = load_spk_filtered(mname, datexp, blk, valid_mask)
    nneu, nfr = spk.shape

    # Load behavior
    beh = load_beh(session_info)
    ntrials = beh['ntrials']
    nfr_use = min(nfr, len(beh['ft']))
    spk = spk[:, :nfr_use]

    # Frame-level arrays
    ft_Pos = beh['ft_Pos'][:nfr_use]
    ft_CorrSpc = beh['ft_CorrSpc'][:nfr_use].astype(bool)
    ft_RunSpeed = beh['ft_RunSpeed'][:nfr_use]

    # Per-trial arrays
    StartFr = beh['StartFr'].astype(int)
    SoundFr = beh['SoundFr']
    WallName = beh['WallName']
    isRew = beh['isRew']

    # Compute frame-level outputs
    # Licking: binary per frame
    lick_binary = np.zeros(nfr_use, dtype=np.int64)
    if 'LickFr' in beh and len(beh['LickFr']) > 0:
        lick_fr = beh['LickFr'].astype(int)
        valid_lick = (lick_fr >= 0) & (lick_fr < nfr_use)
        lick_binary[lick_fr[valid_lick]] = 1

    # Position bins: 4 texture bins (1m each) + gray space
    pos_bins = np.full(nfr_use, 4, dtype=np.int64)  # default: gray
    for b in range(4):
        mask = ft_CorrSpc & (ft_Pos >= b * 10) & (ft_Pos < (b + 1) * 10)
        pos_bins[mask] = b
    pos_bins[ft_CorrSpc & (ft_Pos >= 40)] = 3  # edge case

    # Speed bins: quartiles
    speed_bins = np.zeros(nfr_use, dtype=np.int64)
    speed_bins[ft_RunSpeed >= speed_quartiles[0]] = 1
    speed_bins[ft_RunSpeed >= speed_quartiles[1]] = 2
    speed_bins[ft_RunSpeed >= speed_quartiles[2]] = 3

    # Frame rate for this session
    ft = beh['ft']
    dt = np.median(np.diff(ft[:min(1000, len(ft))])) * 86400  # days->seconds
    fs = 1.0 / dt if dt > 0 else FRAME_RATE

    # Day of training
    day = np.float32(get_session_day(db))

    # Build trial data
    neural_trials = []
    input_trials = []
    output_trials = []
    stim_names = []

    for i in range(ntrials):
        # Trial frames: StartFr[i] to StartFr[i+1] or end
        start = StartFr[i]
        end = StartFr[i + 1] if i < ntrials - 1 else nfr_use
        start = max(0, start)
        end = min(nfr_use, end)
        n_frames = end - start
        if n_frames < 2:
            continue

        # Neural: (n_neurons, n_timepoints), already float16
        trial_spk = spk[:, start:end].copy()  # copy to avoid referencing large array

        # Inputs: (4, n_timepoints)
        frame_idx = np.arange(start, end)
        sound_fr = SoundFr[i]
        if np.isnan(sound_fr):
            time_to_sound = np.zeros(n_frames, dtype=np.float32)
        else:
            time_to_sound = ((sound_fr - frame_idx) / fs).astype(np.float32)

        inp = np.stack([
            time_to_sound,
            np.full(n_frames, day, dtype=np.float32),
            (np.arange(n_frames) / fs).astype(np.float32),
            np.full(n_frames, float(isRew[i]), dtype=np.float32),
        ], axis=0)

        # Outputs: (4, n_timepoints)
        stim = standardize_stim_name(str(WallName[i]))
        out = np.stack([
            np.full(n_frames, 0, dtype=np.int64),  # placeholder for stim idx
            lick_binary[start:end],
            pos_bins[start:end],
            speed_bins[start:end],
        ], axis=0)

        neural_trials.append(trial_spk)
        input_trials.append(inp)
        output_trials.append(out)
        stim_names.append(stim)

    if len(neural_trials) < 2:
        print(f"WARNING: <2 valid trials for {spk_key}")
        return None

    return {
        'neural': neural_trials,
        'input': input_trials,
        'output': output_trials,
        'stim_names': stim_names,
        'region_idx': region_idx,
        'mname': mname,
        'nneu': nneu,
        'nneu_total': nneu_total,
        'ntrials': len(neural_trials),
    }


# ============================================================
# Main Conversion
# ============================================================

def convert_data(session_map, output_file, sample_keys=None, show_processing=False):
    """Main conversion function."""
    keys = sample_keys if sample_keys else sorted(session_map.keys())
    t0 = time.time()

    print(f"Processing {len(keys)} sessions...")

    # Step 1: Speed quartiles (from behavior only - fast)
    print("Computing speed quartiles...")
    t1 = time.time()
    speed_quartiles = collect_speed_quartiles(session_map, keys)
    print(f"  Quartiles: {speed_quartiles}, time: {time.time()-t1:.1f}s")

    # Step 2: Process sessions
    neural_all, input_all, output_all = [], [], []
    subject_idx_list = []
    brain_region_idx_all = []
    all_stim_per_session = []
    all_stim_names = set()
    subjects_seen = {}

    for idx, spk_key in enumerate(keys):
        t_s = time.time()
        info = session_map[spk_key]
        mname = info['db']['mname']
        print(f"  [{idx+1}/{len(keys)}] {spk_key}...", end=' ', flush=True)

        result = process_session(spk_key, info, speed_quartiles)
        if result is None:
            print("SKIPPED")
            continue

        if mname not in subjects_seen:
            subjects_seen[mname] = len(subjects_seen)

        neural_all.append(result['neural'])
        input_all.append(result['input'])
        output_all.append(result['output'])
        subject_idx_list.append(subjects_seen[mname])
        brain_region_idx_all.append(result['region_idx'])
        all_stim_per_session.append(result['stim_names'])
        all_stim_names.update(result['stim_names'])

        print(f"{result['nneu']}({result['nneu_total']}) neurons, {result['ntrials']} trials, {time.time()-t_s:.1f}s")

    # Step 3: Encode stimulus indices
    all_stim_sorted = sorted(all_stim_names)
    stim_to_idx = {s: i for i, s in enumerate(all_stim_sorted)}
    for si in range(len(output_all)):
        for ti in range(len(output_all[si])):
            output_all[si][ti][0, :] = stim_to_idx[all_stim_per_session[si][ti]]

    # Build data structure
    subjects = sorted(subjects_seen.keys(), key=lambda x: subjects_seen[x])

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
            all_stim_sorted,
            ['no_lick', 'lick'],
            ['0-1m', '1-2m', '2-3m', '3-4m', 'gray'],
            ['Q1_slow', 'Q2', 'Q3', 'Q4_fast'],
        ],
        'metadata': {
            'task_description': 'Visual discrimination in VR corridors with naturalistic textures. '
                                'Sound cue signals reward availability. Decode stimulus, licking, position, speed.',
            'time_bin_size': TIME_BIN_MS,
            'temporal_alignment_event': 'corridor entry (trial start)',
            'off_start': 0.0,
            'off_end': None,
            'frame_rate_hz': FRAME_RATE,
            'corridor_length_m': 6.0,
            'texture_length_m': 4.0,
            'vr_speed_cm_s': 60.0,
            'speed_quartiles': speed_quartiles.tolist(),
            'neural_data_type': 'Suite2p deconvolved calcium traces (tau=0.75s), stored as float16',
            'neuron_filtering': 'Excluded neurons outside visual cortex (iarea==-1 or iarea==7)',
        }
    }

    # Summary
    total_trials = sum(len(s) for s in neural_all)
    total_neurons = sum(len(br) for br in brain_region_idx_all)
    print(f"\nSummary: {len(neural_all)} sessions, {len(subjects)} subjects, "
          f"{total_trials} trials, {total_neurons} neurons (across sessions)")
    print(f"Stimuli: {all_stim_sorted}")

    # Save
    print(f"Saving to {output_file}...", flush=True)
    with open(output_file, 'wb') as f:
        pickle.dump(data, f, protocol=4)
        f.flush()
        os.fsync(f.fileno())
    fsize = os.path.getsize(output_file) / (1024**3)
    print(f"  Size: {fsize:.2f} GB, total time: {time.time()-t0:.1f}s", flush=True)

    if show_processing:
        plot_processing(data)

    return data


def plot_processing(data):
    """Plot processing verification for first 2 sessions."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    for si in range(min(2, len(data['neural']))):
        trials = data['neural'][si]
        inputs = data['input'][si]
        outputs = data['output'][si]
        if not trials:
            continue

        fig, axes = plt.subplots(4, 2, figsize=(20, 16))
        fig.suptitle(f'Session {si} Processing Verification')

        # Trial 0: neural heatmap
        ax = axes[0, 0]
        n_show = min(50, trials[0].shape[0])
        ax.imshow(trials[0][:n_show].astype(float), aspect='auto', cmap='viridis')
        ax.set_title(f'Trial 0: Neural ({n_show} neurons)')
        ax.set_xlabel('Frame'); ax.set_ylabel('Neuron')

        # Trial 0: inputs
        ax = axes[0, 1]
        for i, name in enumerate(data['input_names']):
            ax.plot(inputs[0][i], label=name, alpha=0.7)
        ax.legend(fontsize=7); ax.set_title('Trial 0: Inputs')

        # Trial 0: outputs
        ax = axes[1, 0]
        for i, name in enumerate(data['output_names']):
            ax.plot(outputs[0][i], label=name, alpha=0.7)
        ax.legend(fontsize=7); ax.set_title('Trial 0: Outputs')

        # Licking raster
        ax = axes[1, 1]
        for ti in range(min(50, len(trials))):
            lf = np.where(outputs[ti][1] > 0)[0]
            if len(lf):
                ax.scatter(lf, np.full_like(lf, ti), s=1, c='k')
        ax.set_title('Licking raster (50 trials)')
        ax.set_xlabel('Frame'); ax.set_ylabel('Trial')

        # Position distribution
        ax = axes[2, 0]
        all_pos = np.concatenate([outputs[ti][2] for ti in range(len(trials))])
        ax.hist(all_pos, bins=np.arange(6)-0.5, density=True)
        ax.set_xticks(range(5))
        ax.set_xticklabels(['0-1m', '1-2m', '2-3m', '3-4m', 'gray'])
        ax.set_title('Position distribution')

        # Speed distribution
        ax = axes[2, 1]
        all_spd = np.concatenate([outputs[ti][3] for ti in range(len(trials))])
        ax.hist(all_spd, bins=np.arange(5)-0.5, density=True)
        ax.set_xticks(range(4))
        ax.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4'])
        ax.set_title('Speed distribution')

        # Stim distribution
        ax = axes[3, 0]
        all_stim = [outputs[ti][0, 0] for ti in range(len(trials))]
        vals, counts = np.unique(all_stim, return_counts=True)
        ax.bar(vals, counts / len(trials))
        ax.set_xticks(range(len(data['output_values'][0])))
        ax.set_xticklabels(data['output_values'][0], rotation=45, fontsize=7)
        ax.set_title('Stimulus distribution')

        # Trial duration
        ax = axes[3, 1]
        durs = [trials[ti].shape[1] for ti in range(len(trials))]
        ax.hist(durs, bins=30)
        ax.set_title(f'Trial duration (median={np.median(durs):.0f} frames)')

        plt.tight_layout()
        fn = f'processing_session_{si}.png'
        plt.savefig(fn, dpi=150); plt.close()
        print(f"  Saved {fn}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=str)
    parser.add_argument('--full', action='store_true', default=True)
    parser.add_argument('--sample', action='store_true')
    parser.add_argument('--show-processing', action='store_true')
    args = parser.parse_args()

    if args.sample:
        args.full = False

    print("Building session map...")
    session_map = build_session_map()
    print(f"  {len(session_map)} sessions found")

    all_keys = sorted(session_map.keys())
    if args.sample:
        # Pick one unsupervised and one supervised session for diverse testing
        sample_keys = ['TX108_2023_03_25_1', 'DR10_2022_07_12_1']
        # Verify they exist
        sample_keys = [k for k in sample_keys if k in session_map]
        if len(sample_keys) < 2:
            sample_keys = [all_keys[0], all_keys[len(all_keys)//2]]
        print(f"  Sample: {sample_keys}")
    else:
        sample_keys = None

    convert_data(session_map, args.output, sample_keys=sample_keys,
                 show_processing=args.show_processing)


if __name__ == '__main__':
    main()
