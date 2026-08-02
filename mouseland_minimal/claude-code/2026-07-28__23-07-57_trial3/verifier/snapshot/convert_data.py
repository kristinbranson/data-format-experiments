"""
Convert Zhong et al. 2025 data to decoder format.

Reference: "Unsupervised pretraining in biological neural networks"
Paper: Zhong, Baptista, Gattoni et al., Nature 2025

Data: 89 recordings in 19 mice, two-photon calcium imaging of visual cortex.
Mice ran through virtual reality corridors (4m texture + 2m gray space) with
visual stimuli (leaf, circle, rock, brick patterns). VR speed constant at
60 cm/s when mouse running > 6 cm/s threshold.

Processing follows reference code:
- Neural data: deconvolved fluorescence traces (Suite2p), concatenated across planes
- Spatial interpolation: 60 bins per trial (40 corridor + 20 gray), each 1 dm
- Only running frames included (ft_move > 0), consistent with paper
- Brain regions: V1, mHV, lHV, aHV based on retinotopy (iarea mapping)
- Neurons outside visual cortex (iarea == -1 or 7) excluded

Time bin size: 166.67 ms (1 dm / 6 dm/s at constant VR speed)
Temporal alignment: trial start (corridor entry)
"""

import numpy as np
import pickle
import os
import sys
from scipy import interpolate
from collections import defaultdict
from datetime import datetime


def interp_value(v, vind, tind):
    """Interpolate value v at positions vind to target positions tind."""
    Model_ = interpolate.interp1d(vind, v, fill_value='extrapolate')
    return Model_(tind)


def spk_pos_interp(raw_spk, accum_pos, corridorLen, new_shape):
    """Interpolate spike data into position bins.
    raw_spk: neurons x frames
    accum_pos: accumulated position
    corridorLen: length of corridor
    new_shape: [n_trials, n_bins_per_trial]
    """
    if len(new_shape) == 2:
        if new_shape[1] == 0:
            new_shape[1] = corridorLen
    linPos = np.arange(0, new_shape[0], 1 / new_shape[1])
    spk_resh = []
    for s in range(raw_spk.shape[0]):
        spk_resh.append(np.reshape(
            interp_value(raw_spk[s, :], accum_pos / corridorLen, linPos),
            (int(new_shape[0]), int(new_shape[1]))
        ))
    return np.array(spk_resh)


def get_interpPos_spk(spk, spk_culm_pos, ntrial, n_bins=60, lengths=60):
    """Get position-interpolated neural activity (neurons x trials x positions)."""
    interp_spk = np.zeros((spk.shape[0], ntrial, n_bins))
    step_size = 10000
    i = 0
    while i <= spk.shape[0]:
        interp_spk[i:i+step_size, :] = spk_pos_interp(
            raw_spk=spk[i:i+step_size, :],
            accum_pos=spk_culm_pos,
            corridorLen=lengths,
            new_shape=[ntrial, 0]
        )
        i += step_size
    return interp_spk


def neu_area_ID(iarea):
    """Map iarea codes to brain region names. From reference utils.py."""
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
    """Load neural data (deconvolved fluorescence) and concatenate planes."""
    fn = f'{mname}_{datexp}_{blk}_neural_data.npy'
    spk_path = os.path.join(root, fn)
    spk = np.concatenate(
        [nspk for nspk in np.load(spk_path, allow_pickle=True).item()['spks']], 0
    )
    return spk


def load_retino(mname, datexp, root='data/retinotopy'):
    """Load retinotopy data for brain region assignment."""
    fn = f'{mname}_{datexp}_trans.npz'
    dtrans = np.load(os.path.join(root, fn), allow_pickle=True)
    return dtrans['iarea']


def get_session_beh_mapping(exp_info, data_root='data/beh'):
    """Build mapping from session key (mname_datexp_blk) to (exp_type, beh_key).
    For sessions in multiple experiment types, prefer one with most stimuli."""
    session_map = {}

    for exp_type, db_list in exp_info.items():
        for ndb in db_list:
            key = f'{ndb["mname"]}_{ndb["datexp"]}_{ndb["blk"]}'
            stimtype = ndb.get('stimtype', '')
            if stimtype:
                beh_key = f'{key}_{stimtype}'
            else:
                beh_key = key

            if key not in session_map:
                session_map[key] = (exp_type, beh_key, ndb)
            else:
                # Prefer experiment type with more stimuli (more complete data)
                old_ndb = session_map[key][2]
                old_nstim = np.sum(~np.isnan(old_ndb['stim_id'].astype(float)))
                new_nstim = np.sum(~np.isnan(ndb['stim_id'].astype(float)))
                if new_nstim > old_nstim:
                    session_map[key] = (exp_type, beh_key, ndb)

    return session_map


def compute_day_of_training(exp_info):
    """Compute day of training for each session relative to first session of each mouse."""
    mouse_dates = defaultdict(list)
    session_dates = {}

    for exp_type, db_list in exp_info.items():
        for ndb in db_list:
            key = f'{ndb["mname"]}_{ndb["datexp"]}_{ndb["blk"]}'
            date = datetime.strptime(ndb['datexp'], '%Y_%m_%d')
            mouse_dates[ndb['mname']].append(date)
            session_dates[key] = (ndb['mname'], date)

    # Get first date for each mouse
    mouse_first_date = {m: min(dates) for m, dates in mouse_dates.items()}

    # Compute day offset for each session
    session_days = {}
    for key, (mname, date) in session_dates.items():
        session_days[key] = (date - mouse_first_date[mname]).days

    return session_days


def make_lick_spatial_bins(lick_pos, lick_trind, ntrials, n_bins=60):
    """Create binary lick array in spatial bins for each trial.
    Returns: (ntrials, n_bins) binary array.
    """
    lick_arr = np.zeros((ntrials, n_bins), dtype=np.int64)
    for i in range(len(lick_pos)):
        tr = int(lick_trind[i])
        pos = lick_pos[i]
        bin_idx = int(np.clip(np.floor(pos), 0, n_bins - 1))
        if 0 <= tr < ntrials:
            lick_arr[tr, bin_idx] = 1
    return lick_arr


def discretize_speed_quartiles(run_pos_all):
    """Compute global quartile edges for running speed discretization.
    run_pos_all: list of (ntrials, 60) arrays from all sessions.
    Returns: quartile edges array.
    """
    all_speeds = np.concatenate([rp.ravel() for rp in run_pos_all])
    # Remove any NaN
    all_speeds = all_speeds[~np.isnan(all_speeds)]
    edges = np.percentile(all_speeds, [25, 50, 75])
    return edges


def speed_to_bins(run_pos, edges):
    """Discretize running speed into 4 quartile bins.
    Returns: (ntrials, n_bins) array with values 0-3.
    """
    result = np.digitize(run_pos, edges)  # 0, 1, 2, 3
    return result.astype(np.int64)


def convert_data(data_root='data', output_file='converted_data.pkl', sample_file=None, max_sessions=None):
    """Main conversion function."""

    print("Loading experiment info...")
    exp_info = np.load(
        os.path.join(data_root, 'beh', 'Imaging_Exp_info.npy'),
        allow_pickle=True
    ).item()

    # Build session-to-behavior mapping
    session_map = get_session_beh_mapping(exp_info, os.path.join(data_root, 'beh'))

    # Get all unique sessions sorted for reproducibility
    all_sessions = sorted(session_map.keys())
    print(f"Total unique recording sessions: {len(all_sessions)}")

    if max_sessions is not None:
        all_sessions = all_sessions[:max_sessions]
        print(f"Limiting to {max_sessions} sessions")

    # Compute day of training per session
    session_days = compute_day_of_training(exp_info)

    # Collect all subjects
    all_mice = sorted(set(k.split('_')[0] for k in all_sessions))
    mouse_to_idx = {m: i for i, m in enumerate(all_mice)}

    # Brain regions
    brain_regions = ['V1', 'mHV', 'lHV', 'aHV']
    brain_region_map = {name: i for i, name in enumerate(brain_regions)}

    # First pass: collect all running speeds for global quartile computation
    # Use ALL sessions (not limited by max_sessions) for global quartiles
    print("First pass: collecting running speed data for quartile computation...")
    beh_cache = {}
    run_pos_all = []

    all_sessions_full = sorted(session_map.keys())
    for sess_key in all_sessions_full:
        exp_type, beh_key, ndb = session_map[sess_key]

        # Load behavior if not cached
        if exp_type not in beh_cache:
            beh_path = os.path.join(data_root, 'beh', f'Beh_{exp_type}.npy')
            beh_cache[exp_type] = np.load(beh_path, allow_pickle=True).item()

        beh = beh_cache[exp_type][beh_key]
        run_pos_all.append(beh['run_pos'])

    speed_edges = discretize_speed_quartiles(run_pos_all)
    print(f"Running speed quartile edges: {speed_edges}")

    # Collect all unique stimulus names across ALL sessions (not just the subset)
    all_stim_names_set = set()
    for exp_type_key in exp_info.keys():
        beh_path = os.path.join(data_root, 'beh', f'Beh_{exp_type_key}.npy')
        if os.path.exists(beh_path):
            beh_data = np.load(beh_path, allow_pickle=True).item()
            for beh_key_inner in beh_data:
                for name in np.unique(beh_data[beh_key_inner]['WallName']):
                    all_stim_names_set.add(str(name))
    all_stim_names = sorted(all_stim_names_set)
    stim_to_idx = {name: i for i, name in enumerate(all_stim_names)}
    print(f"Stimulus categories: {all_stim_names}")

    # Second pass: build the dataset
    print("\nSecond pass: building dataset...")
    neural_all = []
    input_all = []
    output_all = []
    subject_idx_list = []
    brain_region_idx_all = []

    n_bins = 60  # spatial bins per trial (40 corridor + 20 gray)
    time_bin_sec = 1.0 / 6.0  # 1 dm at 6 dm/s = 166.67 ms

    # Position categories: 4 bins of 1m (10 dm) each in corridor
    # Bins: [0-10) -> 0, [10-20) -> 1, [20-30) -> 2, [30-40) -> 3, [40-60) -> 4 (gray)
    position_bins_edges = [10, 20, 30, 40]  # gray space is everything >= 40
    n_position_categories = 5  # 4 corridor bins + gray space

    position_values = ['0-1m', '1-2m', '2-3m', '3-4m', 'gray_space']

    skipped_sessions = []

    for sess_idx, sess_key in enumerate(all_sessions):
        exp_type, beh_key, ndb = session_map[sess_key]
        mname = ndb['mname']
        datexp = ndb['datexp']
        blk = ndb['blk']

        print(f"  [{sess_idx+1}/{len(all_sessions)}] Processing {sess_key}...")

        # Load behavior
        beh = beh_cache[exp_type][beh_key]
        ntrials = beh['ntrials']
        CL = beh['Corridor_Length']  # typically 60

        if ntrials < 2:
            print(f"    Skipping: only {ntrials} trials")
            skipped_sessions.append(sess_key)
            continue

        # Load neural data
        try:
            spk = load_spk(mname, datexp, blk, root=os.path.join(data_root, 'spk'))
        except Exception as e:
            print(f"    Skipping: could not load neural data: {e}")
            skipped_sessions.append(sess_key)
            continue

        nneu, nfr = spk.shape

        # Load retinotopy for brain region assignment
        try:
            iarea = load_retino(mname, datexp, root=os.path.join(data_root, 'retinotopy'))
        except Exception as e:
            print(f"    Skipping: could not load retinotopy: {e}")
            skipped_sessions.append(sess_key)
            continue

        # Filter neurons: exclude those outside visual cortex (iarea == -1 or 7)
        valid_neuron_mask = (iarea != -1) & (iarea != 7)
        n_valid = valid_neuron_mask.sum()

        if n_valid < 10:
            print(f"    Skipping: only {n_valid} valid neurons")
            skipped_sessions.append(sess_key)
            continue

        # Get brain region index for valid neurons
        area_idx = neu_area_ID(iarea)
        neuron_region_idx = np.full(nneu, -1, dtype=np.int64)
        for region_name, region_i in brain_region_map.items():
            neuron_region_idx[area_idx[region_name]] = region_i

        # Apply valid neuron filter
        spk_filtered = spk[valid_neuron_mask]
        neuron_region_idx_filtered = neuron_region_idx[valid_neuron_mask]

        # Sanity check: all valid neurons should have a region
        assert np.all(neuron_region_idx_filtered >= 0), \
            f"Session {sess_key}: some valid neurons have no region assignment"

        print(f"    Neurons: {nneu} total, {n_valid} valid ({n_valid/nneu*100:.1f}%)")

        # Interpolate neural data into spatial bins using only running frames
        # Following reference: spk[:, VRmove], ft_AcumPos[VRmove], ntrials, n_bins=60, lengths=CL
        VRmove = beh['ft_move'][:nfr] > 0
        ft_AcumPos = beh['ft_PosCum'][:nfr]

        interp_spk = get_interpPos_spk(
            spk_filtered[:, VRmove],
            ft_AcumPos[VRmove],
            ntrials,
            n_bins=int(CL),
            lengths=CL
        )
        # interp_spk shape: (n_valid_neurons, ntrials, n_bins)

        # Build trial-level data
        neural_trials = []
        input_trials = []
        output_trials = []

        # Get trial-level variables
        WallName = beh['WallName']
        isRew = beh['isRew']
        SoundPos = beh['SoundPos']  # sound cue position in dm
        run_pos = beh['run_pos']    # (ntrials, 60) running speed at each position

        # Licking in spatial bins
        lick_spatial = make_lick_spatial_bins(
            beh['LickPos'], beh['LickTrind'], ntrials, n_bins=int(CL)
        )

        # Speed discretized into quartile bins
        speed_bins = speed_to_bins(run_pos, speed_edges)  # (ntrials, 60)

        # Day of training for this session
        day = session_days.get(sess_key, 0)

        # Position array: which spatial bin each position falls in
        positions = np.arange(n_bins)
        pos_category = np.digitize(positions, position_bins_edges).astype(np.int64)
        # pos_category: 0 for [0-10), 1 for [10-20), 2 for [20-30), 3 for [30-40), 4 for [40-60)

        for trial in range(ntrials):
            # Neural: (n_neurons, n_timepoints)
            neural_trial = interp_spk[:, trial, :].astype(np.float32)

            # Check for NaN/Inf in neural data
            if np.any(np.isnan(neural_trial)) or np.any(np.isinf(neural_trial)):
                neural_trial = np.nan_to_num(neural_trial, nan=0.0, posinf=0.0, neginf=0.0)

            # === INPUTS (4 variables) ===
            # 1. Time to sound cue (seconds): (SoundPos - position) * time_bin_sec
            #    Positive before cue, negative after
            sound_pos = SoundPos[trial]
            time_to_cue = (sound_pos - positions) * time_bin_sec  # shape (60,)

            # 2. Day of training (constant across timepoints)
            day_array = np.full(n_bins, day, dtype=np.float32)

            # 3. Time since trial start (seconds)
            time_since_start = positions * time_bin_sec  # shape (60,)

            # 4. Reward availability (binary, constant per trial)
            reward_avail = np.full(n_bins, float(isRew[trial]), dtype=np.float32)

            input_trial = np.stack([
                time_to_cue.astype(np.float32),
                day_array,
                time_since_start.astype(np.float32),
                reward_avail
            ], axis=0)  # shape (4, 60)

            # === OUTPUTS (4 variables) ===
            # 1. Visual stimulus category (per-trial, constant across time)
            stim_idx = stim_to_idx[str(WallName[trial])]
            stim_category = np.full(n_bins, stim_idx, dtype=np.int64)

            # 2. Licking (binary, time-varying)
            lick_trial = lick_spatial[trial, :].astype(np.int64)  # shape (60,)

            # 3. Position in corridor (4 bins + gray, time-varying)
            pos_trial = pos_category.copy()  # shape (60,)

            # 4. Running speed (4 quartile bins, time-varying)
            speed_trial = speed_bins[trial, :].astype(np.int64)  # shape (60,)

            output_trial = np.stack([
                stim_category,
                lick_trial,
                pos_trial,
                speed_trial
            ], axis=0)  # shape (4, 60)

            neural_trials.append(neural_trial)
            input_trials.append(input_trial)
            output_trials.append(output_trial)

        neural_all.append(neural_trials)
        input_all.append(input_trials)
        output_all.append(output_trials)
        subject_idx_list.append(mouse_to_idx[mname])
        brain_region_idx_all.append(neuron_region_idx_filtered)

    # Build speed bin labels
    speed_labels = [
        f'Q1(<{speed_edges[0]:.1f})',
        f'Q2({speed_edges[0]:.1f}-{speed_edges[1]:.1f})',
        f'Q3({speed_edges[1]:.1f}-{speed_edges[2]:.1f})',
        f'Q4(>{speed_edges[2]:.1f})'
    ]

    # Build output
    data = {
        'neural': neural_all,
        'input': input_all,
        'output': output_all,

        'subjects': [str(m) for m in all_mice],
        'subject_idx': np.array(subject_idx_list, dtype=np.int64),

        'brain_regions': [str(r) for r in brain_regions],
        'brain_region_idx': brain_region_idx_all,

        'input_names': [
            'time_to_sound_cue',
            'day_of_training',
            'time_since_trial_start',
            'reward_availability'
        ],

        'output_names': [
            'visual_stimulus',
            'licking',
            'corridor_position',
            'running_speed'
        ],

        'output_values': [
            [str(s) for s in all_stim_names],     # stimulus categories
            ['no_lick', 'lick'],                   # licking
            [str(p) for p in position_values],     # position bins
            [str(s) for s in speed_labels]         # speed quartile bins
        ],

        'metadata': {
            'task_description': 'Visual discrimination in virtual reality corridors. Mice discriminate naturalistic texture patterns (leaf vs circle) in 4m corridors. Sound cue indicates reward availability in rewarded corridor.',
            'time_bin_size': time_bin_sec * 1000,  # 166.67 ms
            'temporal_alignment_event': 'Trial start (corridor entry)',
            'off_start': 0.0,  # alignment is at corridor entry
            'off_end': n_bins * time_bin_sec,  # ~10 seconds
            'n_spatial_bins': n_bins,
            'corridor_length_m': 4.0,
            'gray_space_length_m': 2.0,
            'vr_speed_cm_per_s': 60.0,
            'spatial_bin_size_dm': 1.0,
            'frame_rate_hz': 3.17,
            'calcium_indicator': 'GCaMP6s',
            'neural_data_type': 'deconvolved fluorescence (Suite2p)',
            'recording_method': 'two-photon mesoscope',
            'n_sessions': len(neural_all),
            'n_sessions_skipped': len(skipped_sessions),
            'skipped_sessions': skipped_sessions,
            'speed_quartile_edges': speed_edges.tolist(),
            'brain_region_mapping': {
                'V1': 'iarea==8',
                'mHV': 'iarea in {0,1,2,9} (PM, AM, MMA, retrosplenial)',
                'lHV': 'iarea in {5,6}',
                'aHV': 'iarea in {3,4}'
            },
            'neuron_filtering': 'Excluded neurons with iarea==-1 or iarea==7 (outside visual cortex)',
            'reference': 'Zhong et al., Unsupervised pretraining in biological neural networks, Nature 2025'
        }
    }

    # Print summary statistics
    total_trials = sum(len(s) for s in neural_all)
    total_neurons = sum(brain_region_idx_all[i].shape[0] for i in range(len(neural_all)))
    neurons_per_session = [brain_region_idx_all[i].shape[0] for i in range(len(neural_all))]

    print(f"\n=== Conversion Summary ===")
    print(f"Sessions: {len(neural_all)}")
    print(f"Total trials: {total_trials}")
    print(f"Total neurons (across sessions): {total_neurons}")
    print(f"Neurons per session: min={min(neurons_per_session)}, max={max(neurons_per_session)}, mean={np.mean(neurons_per_session):.0f}")
    print(f"Subjects: {len(all_mice)} ({all_mice})")
    print(f"Brain regions: {brain_regions}")
    print(f"Stimuli: {all_stim_names}")
    print(f"Time bins per trial: {n_bins}")
    print(f"Time bin size: {time_bin_sec*1000:.2f} ms")
    print(f"Speed quartile edges: {speed_edges}")
    print(f"Skipped sessions: {len(skipped_sessions)}")

    # Region distribution
    for region_name in brain_regions:
        region_idx = brain_region_map[region_name]
        count = sum(np.sum(br == region_idx) for br in brain_region_idx_all)
        print(f"  {region_name}: {count} neurons total")

    # Save
    print(f"\nSaving to {output_file}...")
    with open(output_file, 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved successfully ({os.path.getsize(output_file) / 1e9:.2f} GB)")

    # Optionally save a sample (first few sessions)
    if sample_file is not None:
        n_sample = min(5, len(neural_all))
        sample_data = {
            'neural': neural_all[:n_sample],
            'input': input_all[:n_sample],
            'output': output_all[:n_sample],
            'subjects': data['subjects'],
            'subject_idx': np.array(subject_idx_list[:n_sample], dtype=np.int64),
            'brain_regions': data['brain_regions'],
            'brain_region_idx': brain_region_idx_all[:n_sample],
            'input_names': data['input_names'],
            'output_names': data['output_names'],
            'output_values': data['output_values'],
            'metadata': data['metadata'],
        }
        print(f"Saving sample ({n_sample} sessions) to {sample_file}...")
        with open(sample_file, 'wb') as f:
            pickle.dump(sample_data, f)
        print(f"Sample saved ({os.path.getsize(sample_file) / 1e6:.1f} MB)")

    return data


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Convert Zhong et al. 2025 data to decoder format')
    parser.add_argument('--data-root', type=str, default='data', help='Root data directory')
    parser.add_argument('--output', type=str, default='converted_data.pkl', help='Output pickle file')
    parser.add_argument('--sample', type=str, default=None, help='Output sample pickle file')
    parser.add_argument('--max-sessions', type=int, default=None, help='Max sessions to process')
    args = parser.parse_args()

    convert_data(
        data_root=args.data_root,
        output_file=args.output,
        sample_file=args.sample,
        max_sessions=args.max_sessions
    )
