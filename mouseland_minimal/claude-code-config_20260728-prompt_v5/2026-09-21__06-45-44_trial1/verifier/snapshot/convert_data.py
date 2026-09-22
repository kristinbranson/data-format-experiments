#!/usr/bin/env python3
"""
Convert data from Zhong et al. (2025) "Unsupervised pretraining in biological neural
networks" into a standardized format for neural decoder training.

Processing decisions and rationale:
1. Neural data: Deconvolved spikes from Suite2p, position-interpolated to uniform
   position bins following the reference code's `get_interpPos_spk`. Only frames where
   the VR is moving (ft_move > 0) are used, matching the paper: "We only considered
   timepoints during running for analysis."
2. Position interpolation: 60 bins per full corridor (6m), then only the first 40 bins
   (texture area, 4m) are kept. This matches the paper's standard processing and gives
   uniform trial lengths.
3. Neuron filtering: Neurons with iarea == -1 (unassigned) or iarea == 7 (non-visual)
   are excluded, matching the paper's convention (e.g., `idx_neu = (arid!=-1) & (arid != 7)`
   in Get_density_map).
4. Brain regions: V1 (iarea=8), mHV (0,1,2,9), lHV (5,6), aHV (3,4) per neu_area_ID.
5. All 89 unique recording sessions across all experiment types are included.
6. Trials with NaN/Inf in neural data after interpolation are excluded.
7. Time bin interpretation: VR moves at constant 60 cm/s when running, so each 1-dm
   position bin = 1/6 second = 166.67 ms.
8. Day of training: computed as days from first recording for each mouse.
9. Running speed discretized into quartiles computed across all sessions' texture area.
10. Licking binarized per position bin using LickPos and LickTrind from behavior data.
"""

import numpy as np
import os
import pickle
import gc
from datetime import datetime

DATA_ROOT = '/app/data'
OUTPUT_PATH = '/app/converted_data.pkl'

# Processing parameters matching the paper
CORRIDOR_LENGTH = 60    # decimeters (6m total corridor)
TEXTURE_LENGTH = 40     # decimeters (4m texture area)
N_BINS_FULL = CORRIDOR_LENGTH  # 60 position bins per corridor
N_BINS = TEXTURE_LENGTH  # 40 bins for texture area (used for output)
VR_SPEED_DM_S = 6.0     # VR speed in dm/s (60 cm/s)
TIME_PER_BIN = 1.0 / VR_SPEED_DM_S  # seconds per position bin (1/6 s)
TIME_PER_BIN_MS = TIME_PER_BIN * 1000  # ~166.67 ms
POSITION_N_CATS = 4      # 4 spatial bins of 1m each
SPEED_N_CATS = 4          # 4 quartile bins

BRAIN_REGIONS = ['V1', 'mHV', 'lHV', 'aHV']


def get_brain_region_idx(iarea):
    """Map iarea values to brain region indices, following neu_area_ID in utils.py."""
    region = np.empty(len(iarea), dtype=np.int32)
    region[:] = -1  # temporary
    region[iarea == 8] = 0  # V1
    for a in [0, 1, 2, 9]:
        region[iarea == a] = 1  # mHV (medial higher visual)
    for a in [5, 6]:
        region[iarea == a] = 2  # lHV (lateral higher visual)
    for a in [3, 4]:
        region[iarea == a] = 3  # aHV (anterior higher visual)
    return region


def load_spk(mname, datexp, blk):
    """Load and concatenate spike data across planes, matching utils.load_spk."""
    fn = os.path.join(DATA_ROOT, 'spk', f'{mname}_{datexp}_{blk}_neural_data.npy')
    d = np.load(fn, allow_pickle=True).item()
    return np.concatenate(d['spks'], axis=0)


def load_iarea(mname, datexp):
    """Load area assignments from retinotopy data."""
    fn = os.path.join(DATA_ROOT, 'retinotopy', f'{mname}_{datexp}_trans.npz')
    return np.load(fn, allow_pickle=True)['iarea']


def position_interpolate(spk_running, pos_cum_running, ntrials, n_bins=N_BINS):
    """Position-interpolate neural data to uniform bins, matching spk_pos_interp.

    Computes only the first `n_bins` position bins per trial (texture area).
    Uses the same interpolation grid as the paper: positions normalized by
    corridor length, with 1/CORRIDOR_LENGTH spacing between bins.

    Args:
        spk_running: (n_neurons, n_running_frames) spike data for running frames
        pos_cum_running: (n_running_frames,) cumulative position for running frames
        ntrials: number of trials
        n_bins: number of position bins per trial (default: texture area only)

    Returns: (n_neurons, ntrials, n_bins) position-interpolated activity
    """
    # Target: first n_bins positions within each corridor
    # Matches the paper's linPos = np.arange(0, ntrials, 1/corridor_length)
    # but only the first n_bins of each trial are computed.
    bin_offsets = np.arange(n_bins, dtype=np.float64) / CORRIDOR_LENGTH
    trial_starts = np.arange(ntrials, dtype=np.float64)
    target = (trial_starts[:, None] + bin_offsets[None, :]).ravel()

    # Source: cumulative position normalized by corridor length
    source = pos_cum_running / CORRIDOR_LENGTH

    n_neurons = spk_running.shape[0]
    result = np.empty((n_neurons, len(target)), dtype=np.float32)

    for i in range(n_neurons):
        result[i] = np.interp(target, source, spk_running[i])

    return result.reshape(n_neurons, ntrials, n_bins)


def main():
    # Load experiment info
    exp_info = np.load(
        os.path.join(DATA_ROOT, 'beh', 'Imaging_Exp_info.npy'),
        allow_pickle=True
    ).item()

    # Collect unique sessions: (mname, datexp, blk) -> (exp_type, ndb)
    # First occurrence is kept for each unique recording session.
    unique_sessions = {}
    for exp_type in exp_info:
        for ndb in exp_info[exp_type]:
            key = (ndb['mname'], ndb['datexp'], ndb['blk'])
            if key not in unique_sessions:
                unique_sessions[key] = (exp_type, ndb)

    session_keys = sorted(unique_sessions.keys())
    print(f"Found {len(session_keys)} unique sessions")

    # ---- First pass: collect global statistics ----
    print("First pass: collecting global statistics...")

    all_stimuli = set()
    all_speed_values = []
    mouse_dates = {}
    beh_cache = {}

    for key in session_keys:
        mname, datexp, blk = key
        exp_type, ndb = unique_sessions[key]

        if exp_type not in beh_cache:
            beh_cache[exp_type] = np.load(
                os.path.join(DATA_ROOT, 'beh', f'Beh_{exp_type}.npy'),
                allow_pickle=True
            ).item()

        beh_key = f'{mname}_{datexp}_{blk}'
        if 'stimtype' in ndb:
            beh_key += f'_{ndb["stimtype"]}'

        beh = beh_cache[exp_type].get(beh_key)
        if beh is None:
            print(f"  WARNING: {beh_key} not found in Beh_{exp_type}.npy")
            continue

        for wn in beh['UniqWalls']:
            all_stimuli.add(str(wn))

        speed = beh['run_pos'][:, :N_BINS]
        valid = ~np.isnan(speed)
        all_speed_values.append(speed[valid].ravel())

        date = datetime.strptime(datexp, '%Y_%m_%d')
        mouse_dates.setdefault(mname, []).append(date)

    del beh_cache

    # Speed quartiles (25th, 50th, 75th percentiles) across all sessions
    all_speed_arr = np.concatenate(all_speed_values)
    speed_quartiles = np.percentile(all_speed_arr, [25, 50, 75])
    print(f"Speed quartiles: {speed_quartiles}")
    del all_speed_values, all_speed_arr

    # Stimulus category mapping
    stim_list = sorted(all_stimuli)
    stim_to_idx = {s: i for i, s in enumerate(stim_list)}
    print(f"Stimuli ({len(stim_list)}): {stim_list}")

    # Subjects and day-of-training reference dates
    mouse_first_date = {m: min(dates) for m, dates in mouse_dates.items()}
    subjects = sorted(mouse_first_date.keys())
    subject_to_idx = {s: i for i, s in enumerate(subjects)}
    print(f"Subjects ({len(subjects)}): {subjects}")

    # ---- Second pass: process all sessions ----
    print(f"\nSecond pass: processing {len(session_keys)} sessions...")

    data_neural = []
    data_input = []
    data_output = []
    data_subject_idx = []
    data_brain_region_idx = []

    beh_cache = {}

    # Pre-compute static arrays
    position_bins = np.repeat(np.arange(POSITION_N_CATS), N_BINS // POSITION_N_CATS).astype(np.int32)
    time_since_start = (np.arange(N_BINS, dtype=np.float32) * TIME_PER_BIN)

    for sess_i, key in enumerate(session_keys):
        mname, datexp, blk = key
        exp_type, ndb = unique_sessions[key]

        print(f"  [{sess_i+1}/{len(session_keys)}] {mname}_{datexp}_{blk} ({exp_type})",
              end="", flush=True)

        # Load behavior
        if exp_type not in beh_cache:
            beh_cache[exp_type] = np.load(
                os.path.join(DATA_ROOT, 'beh', f'Beh_{exp_type}.npy'),
                allow_pickle=True
            ).item()

        beh_key = f'{mname}_{datexp}_{blk}'
        if 'stimtype' in ndb:
            beh_key += f'_{ndb["stimtype"]}'

        beh = beh_cache[exp_type].get(beh_key)
        if beh is None:
            print(" -> SKIP: behavior not found")
            continue

        ntrials = int(beh['ntrials'])
        if ntrials < 2:
            print(f" -> SKIP: {ntrials} trials")
            continue

        # Load neural data (concatenated across planes)
        spk = load_spk(mname, datexp, blk)
        n_neurons_total, n_frames_spk = spk.shape

        # Load retinotopy and filter to visual cortex neurons
        iarea = load_iarea(mname, datexp)
        vc_mask = (iarea != -1) & (iarea != 7)  # visual cortex mask
        brain_region = get_brain_region_idx(iarea[vc_mask])
        spk = spk[vc_mask]
        n_neurons = spk.shape[0]

        # Position-interpolate running frames
        n_fr = min(n_frames_spk, len(beh['ft_move']))
        VRmove = beh['ft_move'][:n_fr] > 0
        pos_cum = beh['ft_PosCum'][:n_fr]

        # Apply running mask to neural and position data
        spk_running = spk[:, :n_fr][:, VRmove]
        pos_cum_running = pos_cum[VRmove]
        del spk

        if len(pos_cum_running) < 2:
            print(" -> SKIP: insufficient running frames")
            del spk_running
            continue

        # Interpolate to uniform position bins (texture area only: 40 bins)
        texture_spk = position_interpolate(spk_running, pos_cum_running, ntrials,
                                           n_bins=N_BINS)
        del spk_running, pos_cum_running
        gc.collect()

        # Compute day of training
        session_date = datetime.strptime(datexp, '%Y_%m_%d')
        day_of_training = float((session_date - mouse_first_date[mname]).days)

        # Precompute licking info
        lick_pos = beh['LickPos']
        lick_trind = beh['LickTrind']

        # Build per-trial data
        session_neural = []
        session_input = []
        session_output = []

        for t in range(ntrials):
            neural_t = texture_spk[:, t, :]  # (n_neurons, 40)

            # Skip trials with NaN/Inf
            if np.any(np.isnan(neural_t)) or np.any(np.isinf(neural_t)):
                continue

            # --- Inputs (4, 40) ---
            sound_pos = beh['SoundPos'][t]  # in decimeters
            time_to_cue = ((sound_pos - np.arange(N_BINS)) * TIME_PER_BIN).astype(np.float32)

            day_arr = np.full(N_BINS, day_of_training, dtype=np.float32)
            reward_arr = np.full(N_BINS, float(beh['isRew'][t]), dtype=np.float32)

            input_t = np.stack([time_to_cue, day_arr, time_since_start, reward_arr], axis=0)
            session_input.append(input_t)

            # --- Outputs (4, 40) ---
            # Visual stimulus category (per-trial, repeated across position bins)
            stim_name = str(beh['WallName'][t])
            stim_cat = stim_to_idx[stim_name]
            stim_arr = np.full(N_BINS, stim_cat, dtype=np.int32)

            # Licking: binary per position bin
            lick_binary = np.zeros(N_BINS, dtype=np.int32)
            trial_mask = lick_trind == t
            trial_lick_pos = lick_pos[trial_mask]
            valid_licks = trial_lick_pos[(trial_lick_pos >= 0) & (trial_lick_pos < N_BINS)]
            if len(valid_licks) > 0:
                lick_bins = np.clip(np.floor(valid_licks).astype(int), 0, N_BINS - 1)
                lick_binary[lick_bins] = 1

            # Position: 4 bins of 1m each (deterministic from position index)
            pos_arr = position_bins.copy()

            # Running speed: 4 quartile bins
            speed = beh['run_pos'][t, :N_BINS].copy()
            speed[np.isnan(speed)] = 0.0
            speed_cat = np.digitize(speed, speed_quartiles).astype(np.int32)

            output_t = np.stack([stim_arr, lick_binary, pos_arr, speed_cat], axis=0)

            session_neural.append(neural_t)
            session_output.append(output_t)

        del texture_spk
        gc.collect()

        if len(session_neural) < 2:
            print(f" -> SKIP: {len(session_neural)} valid trials")
            continue

        data_neural.append(session_neural)
        data_input.append(session_input)
        data_output.append(session_output)
        data_subject_idx.append(subject_to_idx[mname])
        data_brain_region_idx.append(brain_region)

        print(f" -> {len(session_neural)} trials, {n_neurons} neurons")

    del beh_cache
    gc.collect()

    # ---- Assemble final data dictionary ----
    print(f"\nAssembling data: {len(data_neural)} sessions, "
          f"{sum(len(s) for s in data_neural)} total trials")

    output_values = [
        stim_list,                              # visual_stimulus categories
        ['no_lick', 'lick'],                    # licking categories
        ['0-1m', '1-2m', '2-3m', '3-4m'],      # position categories
        ['Q1', 'Q2', 'Q3', 'Q4'],              # running speed quartiles
    ]

    data = {
        'neural': data_neural,
        'input': data_input,
        'output': data_output,

        'subjects': subjects,
        'subject_idx': np.array(data_subject_idx, dtype=np.int64),

        'brain_regions': BRAIN_REGIONS,
        'brain_region_idx': data_brain_region_idx,

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
        'output_values': output_values,

        'metadata': {
            'task_description': (
                'Visual discrimination task in head-fixed mice running through '
                'virtual reality corridors with naturalistic texture stimuli. '
                'Decoder predicts stimulus category, licking, corridor position, '
                'and running speed from visual cortex neural activity.'
            ),
            'time_bin_size': TIME_PER_BIN_MS,
            'temporal_alignment_event': 'corridor entry (start of texture area)',
            'off_start': 0.0,
            'off_end': N_BINS * TIME_PER_BIN,
            'n_position_bins': N_BINS,
            'corridor_texture_length_m': 4.0,
            'vr_speed_m_s': 0.6,
            'calcium_imaging_frame_rate_hz': 3.17,
            'processing_notes': (
                'Position-interpolated deconvolved spikes (Suite2p with 0.75s decay). '
                '60 bins per 6m corridor, first 40 bins (texture area) retained. '
                'Only running frames (VR moving) used for interpolation. '
                'Neurons filtered to visual cortex (iarea != -1, != 7).'
            ),
            'speed_quartile_thresholds': speed_quartiles.tolist(),
        }
    }

    # Save
    print(f"Saving to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, 'wb') as f:
        pickle.dump(data, f, protocol=4)

    file_size = os.path.getsize(OUTPUT_PATH)
    print(f"Done! File size: {file_size / 1e9:.2f} GB")
    print(f"Sessions: {len(data_neural)}")
    print(f"Total trials: {sum(len(s) for s in data_neural)}")


if __name__ == '__main__':
    main()
