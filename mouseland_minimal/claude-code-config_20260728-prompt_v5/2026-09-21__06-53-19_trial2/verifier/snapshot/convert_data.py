"""
Convert neural data from Zhong et al. (2025) "Unsupervised pretraining in biological neural networks"
into the standardized format for neural decoder training.

Key decisions:
- Include ALL 89 unique recording sessions across all experiment types (task, unsupervised, naive, grating).
  This maximizes data and the decoder inputs (reward availability, day of training) capture condition differences.
- Use deconvolved fluorescence traces as stated in the paper ("All our analyses were based on
  deconvolved fluorescence traces").
- Trial = texture corridor traversal only (corridor entry to gray space entry), covering the 4m
  texture area. This matches the paper's focus on corridor activity and aligns with the 4 x 1m
  position bins specified for the decoder.
- Include all frames within the corridor period (both running and non-running). The paper filters
  non-running frames for d-prime/statistics, but the decoder needs contiguous temporal data.
  Running speed is captured as an output variable.
- Temporal alignment: trial start = corridor entry (StartFr). Time bins = imaging frames (~315 ms).
- Brain regions from retinotopy data: V1, mHV, lHV, aHV, plus "unassigned" for neurons outside
  identified visual areas (iarea == -1 or 7).
- Running speed quartiles computed across all corridor frames in all sessions.
- Licking derived per frame from LickFr/LickTrind.
- Day of training: calendar days from each mouse's first recording date.
"""

import numpy as np
import pickle
import os
import gc
from datetime import datetime

# Paths
DATA_ROOT = 'data'
SPK_ROOT = os.path.join(DATA_ROOT, 'spk')
BEH_ROOT = os.path.join(DATA_ROOT, 'beh')
RET_ROOT = os.path.join(DATA_ROOT, 'retinotopy')
OUTPUT_PATH = 'converted_data.pkl'

# All possible stimulus names across the dataset
ALL_STIMULI = sorted([
    'circle1', 'circle2', 'circle3',
    'leaf1', 'leaf1_swap1', 'leaf1_swap2', 'leaf2', 'leaf3',
    'rock1', 'rock2',
    'wood1', 'wood1_swap1', 'wood1_swap2', 'wood2', 'wood5',
])
STIM_TO_IDX = {s: i for i, s in enumerate(ALL_STIMULI)}

# Brain region mapping from iarea values (from utils.py neu_area_ID)
BRAIN_REGIONS = ['V1', 'mHV', 'lHV', 'aHV', 'unassigned']
BRAIN_REGION_MAP = {
    8: 0,                  # V1
    0: 1, 1: 1, 2: 1, 9: 1,  # mHV (medial higher visual)
    5: 2, 6: 2,            # lHV (lateral higher visual)
    3: 3, 4: 3,            # aHV (anterior higher visual)
    -1: 4, 7: 4,           # unassigned
}


def load_spk(ndb):
    """Load and concatenate neural data across imaging planes."""
    fn = f"{ndb['mname']}_{ndb['datexp']}_{ndb['blk']}_neural_data.npy"
    spk_data = np.load(os.path.join(SPK_ROOT, fn), allow_pickle=True).item()
    spk = np.concatenate(spk_data['spks'], axis=0)
    del spk_data
    return spk


def get_nfr(ndb):
    """Get number of neural frames without loading full data."""
    fn = f"{ndb['mname']}_{ndb['datexp']}_{ndb['blk']}_neural_data.npy"
    spk_data = np.load(os.path.join(SPK_ROOT, fn), allow_pickle=True).item()
    nfr = spk_data['spks'][0].shape[1]
    del spk_data
    return nfr


def map_brain_regions(iarea):
    """Map iarea values to brain region indices."""
    return np.array([BRAIN_REGION_MAP.get(int(ia), 4) for ia in iarea], dtype=np.int64)


def parse_date(datexp):
    return datetime.strptime(datexp, '%Y_%m_%d')


def collect_sessions():
    """Collect all unique recording sessions across all experiment types."""
    exp_info = np.load(os.path.join(BEH_ROOT, 'Imaging_Exp_info.npy'), allow_pickle=True).item()

    sessions = {}
    for exp_type in exp_info:
        beh_file = os.path.join(BEH_ROOT, f'Beh_{exp_type}.npy')
        Beh = np.load(beh_file, allow_pickle=True).item()

        for ndb in exp_info[exp_type]:
            session_key = f"{ndb['mname']}_{ndb['datexp']}_{ndb['blk']}"
            if session_key in sessions:
                continue

            if 'stimtype' in ndb:
                beh_key = f"{session_key}_{ndb['stimtype']}"
            else:
                beh_key = session_key

            if beh_key in Beh:
                sessions[session_key] = {
                    'ndb': ndb,
                    'beh': Beh[beh_key],
                    'exp_type': exp_type,
                }
        del Beh
    gc.collect()
    return sessions


def compute_days_of_training(sessions):
    """For each mouse, compute calendar days from first recording."""
    mouse_dates = {}
    for session_key, info in sessions.items():
        mname = info['ndb']['mname']
        dt = parse_date(info['ndb']['datexp'])
        mouse_dates.setdefault(mname, []).append(dt)

    mouse_first_date = {m: min(dates) for m, dates in mouse_dates.items()}

    return {
        sk: (parse_date(info['ndb']['datexp']) - mouse_first_date[info['ndb']['mname']]).days
        for sk, info in sessions.items()
    }


def compute_lick_per_frame(beh, trial_idx, frame_indices):
    """Compute binary licking array for given frames of a trial."""
    lick_mask = beh['LickTrind'] == trial_idx
    if not np.any(lick_mask):
        return np.zeros(len(frame_indices), dtype=np.float32)

    lick_frames_set = set(np.round(beh['LickFr'][lick_mask]).astype(int))
    return np.array([1.0 if f in lick_frames_set else 0.0 for f in frame_indices], dtype=np.float32)


def compute_speed_quartiles(sessions):
    """Compute running speed quartile edges across all corridor frames."""
    print("Computing running speed quartiles...", flush=True)
    all_speeds = []
    for session_key in sorted(sessions.keys()):
        info = sessions[session_key]
        beh = info['beh']
        ndb = info['ndb']

        nfr = get_nfr(ndb)
        ft_trInd = beh['ft_trInd'][:nfr]
        ft_CorrSpc = beh['ft_CorrSpc'][:nfr]
        ft_RunSpeed = beh['ft_RunSpeed'][:nfr]

        corridor_mask = ft_CorrSpc & ~np.isnan(ft_trInd)
        all_speeds.append(ft_RunSpeed[corridor_mask])

    all_speeds = np.concatenate(all_speeds)
    q25, q50, q75 = np.percentile(all_speeds, [25, 50, 75])
    print(f"  Speed quartiles: Q25={q25:.2f}, Q50={q50:.2f}, Q75={q75:.2f}", flush=True)
    del all_speeds
    gc.collect()
    return np.array([q25, q50, q75])


def process_session(session_key, info, day_of_training, speed_quartiles, dt_seconds):
    """Process a single recording session. Returns None if < 2 valid trials."""
    ndb = info['ndb']
    beh = info['beh']

    # Load neural data
    spk = load_spk(ndb)
    n_neurons, nfr = spk.shape

    # Load retinotopy and map brain regions
    fn = f"{ndb['mname']}_{ndb['datexp']}_trans.npz"
    ret = np.load(os.path.join(RET_ROOT, fn))
    brain_reg_idx = map_brain_regions(ret['iarea'])

    # Get per-frame behavioral variables (trim to neural frame count)
    ft_trInd = beh['ft_trInd'][:nfr]
    ft_CorrSpc = beh['ft_CorrSpc'][:nfr]
    ft_Pos = beh['ft_Pos'][:nfr]
    ft_RunSpeed = beh['ft_RunSpeed'][:nfr]

    ntrials = beh['ntrials']
    SoundFr = beh['SoundFr']
    WallName = beh['WallName']
    isRew = beh['isRew']

    neural_trials = []
    input_trials = []
    output_trials = []

    for trial_idx in range(ntrials):
        mask = (ft_trInd == trial_idx) & ft_CorrSpc
        frame_indices = np.where(mask)[0]

        if len(frame_indices) < 2:
            continue

        T = len(frame_indices)

        # Neural data: (n_neurons, T) as float32
        neural_trial = spk[:, frame_indices].astype(np.float32)

        # --- INPUTS (4, T) ---
        time_to_sound = ((SoundFr[trial_idx] - frame_indices) * dt_seconds).astype(np.float32)
        day_train = np.full(T, day_of_training, dtype=np.float32)
        time_since_start = ((frame_indices - frame_indices[0]) * dt_seconds).astype(np.float32)
        reward_avail = np.full(T, float(isRew[trial_idx]), dtype=np.float32)

        input_trial = np.stack([time_to_sound, day_train, time_since_start, reward_avail], axis=0)

        # --- OUTPUTS (4, T) --- categorical integer values
        stim_idx = STIM_TO_IDX[str(WallName[trial_idx])]
        stim_out = np.full(T, stim_idx, dtype=np.int64)
        lick_out = compute_lick_per_frame(beh, trial_idx, frame_indices).astype(np.int64)
        pos_out = np.clip(ft_Pos[frame_indices] // 10, 0, 3).astype(np.int64)
        speed_out = np.digitize(ft_RunSpeed[frame_indices], speed_quartiles).astype(np.int64)

        output_trial = np.stack([stim_out, lick_out, pos_out, speed_out], axis=0)

        neural_trials.append(neural_trial)
        input_trials.append(input_trial)
        output_trials.append(output_trial)

    del spk
    gc.collect()

    if len(neural_trials) < 2:
        return None

    return neural_trials, input_trials, output_trials, brain_reg_idx


def main():
    print("=" * 60, flush=True)
    print("Converting data for neural decoder training", flush=True)
    print("=" * 60, flush=True)

    sessions = collect_sessions()
    print(f"Found {len(sessions)} unique recording sessions", flush=True)

    days_map = compute_days_of_training(sessions)

    # Compute dt from first session
    first_key = sorted(sessions.keys())[0]
    ft = sessions[first_key]['beh']['ft']
    dt_seconds = float(np.median(np.diff(ft)) * 86400)
    dt_ms = dt_seconds * 1000
    print(f"Time bin size: {dt_ms:.1f} ms (frame rate: {1/dt_seconds:.2f} Hz)", flush=True)

    speed_quartiles = compute_speed_quartiles(sessions)

    # Process each session
    all_neural = []
    all_input = []
    all_output = []
    all_brain_region_idx = []
    session_subject_idx = []
    session_keys_ordered = []
    subjects_list = []
    subject_to_idx = {}

    sorted_keys = sorted(sessions.keys())
    for i, session_key in enumerate(sorted_keys):
        info = sessions[session_key]
        mname = info['ndb']['mname']

        print(f"[{i+1}/{len(sorted_keys)}] {session_key} (mouse={mname}, exp={info['exp_type']})",
              flush=True)

        result = process_session(session_key, info, days_map[session_key],
                                 speed_quartiles, dt_seconds)

        if result is None:
            print(f"  Skipped: < 2 valid trials", flush=True)
            continue

        neural_trials, input_trials, output_trials, brain_reg_idx = result
        print(f"  {len(neural_trials)} trials, {neural_trials[0].shape[0]} neurons", flush=True)

        all_neural.append(neural_trials)
        all_input.append(input_trials)
        all_output.append(output_trials)
        all_brain_region_idx.append(brain_reg_idx)
        session_keys_ordered.append(session_key)

        if mname not in subject_to_idx:
            subject_to_idx[mname] = len(subjects_list)
            subjects_list.append(mname)
        session_subject_idx.append(subject_to_idx[mname])

    n_sessions = len(all_neural)
    print(f"\nTotal sessions: {n_sessions}", flush=True)
    print(f"Total subjects: {len(subjects_list)}", flush=True)

    output_values = [
        ALL_STIMULI,
        ['no_lick', 'lick'],
        ['0-1m', '1-2m', '2-3m', '3-4m'],
        ['Q1 (slowest)', 'Q2', 'Q3', 'Q4 (fastest)'],
    ]

    data = {
        'neural': all_neural,
        'input': all_input,
        'output': all_output,
        'subjects': subjects_list,
        'subject_idx': np.array(session_subject_idx, dtype=np.int64),
        'brain_regions': BRAIN_REGIONS,
        'brain_region_idx': all_brain_region_idx,
        'input_names': [
            'time_to_sound_cue',
            'day_of_training',
            'time_since_trial_start',
            'reward_availability',
        ],
        'output_names': [
            'visual_stimulus',
            'licking',
            'position_bin',
            'running_speed_bin',
        ],
        'output_values': output_values,
        'metadata': {
            'task_description': (
                'Mice run through virtual reality corridors with naturalistic texture patterns. '
                'In the supervised (task) condition, one corridor is rewarded; a sound cue '
                'indicates reward availability. The decoder predicts stimulus identity, licking, '
                'position, and running speed from neural activity in visual cortex.'
            ),
            'time_bin_size': dt_ms,
            'temporal_alignment_event': 'Corridor entry (trial start)',
            'off_start': 0.0,
            'off_end': None,
            'frame_rate_hz': 1.0 / dt_seconds,
            'corridor_length_m': 4.0,
            'n_sessions': n_sessions,
            'n_subjects': len(subjects_list),
            'session_keys': session_keys_ordered,
            'speed_quartile_edges': speed_quartiles.tolist(),
            'stimulus_names': ALL_STIMULI,
            'source': 'Zhong et al. (2025) Unsupervised pretraining in biological neural networks',
        }
    }

    print(f"\nSaving to {OUTPUT_PATH}...", flush=True)
    with open(OUTPUT_PATH, 'wb') as f:
        pickle.dump(data, f, protocol=4)

    total_trials = sum(len(s) for s in all_neural)
    total_timepoints = sum(t.shape[1] for s in all_neural for t in s)
    print(f"\nSummary:", flush=True)
    print(f"  Sessions: {n_sessions}", flush=True)
    print(f"  Subjects: {len(subjects_list)}", flush=True)
    print(f"  Total trials: {total_trials}", flush=True)
    print(f"  Total timepoints: {total_timepoints}", flush=True)
    print("Done!", flush=True)


if __name__ == '__main__':
    main()
