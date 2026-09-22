"""
Convert neural and behavioral data from Zhong et al. 2025 ("Unsupervised pretraining
in biological neural networks") into a standardized format for neural decoder training.

Key decisions:
- Include all sessions from all experiment types, deduplicated by (mname, datexp, blk).
- Filter neurons to visual cortex areas: V1, mHV (medial higher visual), lHV (lateral HV),
  aHV (anterior HV), matching the reference code's neu_area_ID function.
  Neurons with iarea==-1 (outside visual cortex) or iarea==7 are excluded.
- Temporal alignment: trial start = corridor entry. Extract all frames in the texture
  corridor (ft_CorrSpc == True) for each trial, regardless of running state.
  Rationale: the decoder needs uniform time bins and running speed is an output variable.
  The paper's running-only filter was for position-interpolated analyses.
- Neural data: deconvolved calcium traces (Suite2p output), concatenated across planes.
- Licking: binary per-frame, derived from LickFr rounded to nearest integer frame.
- Position: ft_Pos in decimeters [0,40], discretized into 4 bins of 10 dm (1 m) each.
- Running speed: ft_RunSpeed, discretized into 4 quartile bins computed across all
  corridor frames from all sessions.
- Day of training: computed as calendar days since the mouse's first recording session.
- Minimum 3 corridor frames per trial to be included.
- Sessions with fewer than 2 valid trials are excluded.
"""

import numpy as np
import pickle
import os
from datetime import datetime


DATA_ROOT = '/app/data'
OUTPUT_PATH = '/app/converted_data.pkl'

# Minimum number of corridor frames per trial
MIN_FRAMES_PER_TRIAL = 3


def neu_area_ID(iarea):
    """Assign neurons to visual cortex areas, matching reference code."""
    idx = {}
    idx['V1'] = iarea == 8
    idx['mHV'] = (iarea == 0) | (iarea == 1) | (iarea == 2) | (iarea == 9)
    idx['lHV'] = (iarea == 5) | (iarea == 6)
    idx['aHV'] = (iarea == 3) | (iarea == 4)
    return idx


def load_spk(mname, datexp, blk, root=''):
    """Load and concatenate neural data across planes."""
    fn = f'{mname}_{datexp}_{blk}_neural_data.npy'
    spk_path = os.path.join(root, fn)
    spk = np.concatenate(
        [nspk for nspk in np.load(spk_path, allow_pickle=True).item()['spks']], 0
    )
    return spk


def load_retino(mname, datexp, root=''):
    """Load retinotopy data and return iarea."""
    fn = f'{mname}_{datexp}_trans.npz'
    dtrans = np.load(os.path.join(root, fn), allow_pickle=True)
    return dtrans['iarea']


def collect_unique_sessions():
    """Collect all unique sessions across experiment types, dedup by (mname, datexp, blk).

    For each unique physical recording, we need one behavior data source. When a session
    appears in multiple experiment types, the behavior data is the same (same trials,
    stimuli, etc.) - only the stim_id mapping differs for specific analyses.
    We pick the first encountered exp_type and behavior key for each unique session.
    """
    exp_info = np.load(
        os.path.join(DATA_ROOT, 'beh', 'Imaging_Exp_info.npy'), allow_pickle=True
    ).item()

    unique_sessions = {}  # key: mname_datexp_blk -> session info

    for exp_type, sessions in exp_info.items():
        for s in sessions:
            key = f"{s['mname']}_{s['datexp']}_{s['blk']}"
            if key not in unique_sessions:
                # Determine behavior key
                if 'stimtype' in s:
                    beh_key = f"{s['mname']}_{s['datexp']}_{s['blk']}_{s['stimtype']}"
                else:
                    beh_key = key
                unique_sessions[key] = {
                    'mname': s['mname'],
                    'datexp': s['datexp'],
                    'blk': s['blk'],
                    'exp_type': exp_type,
                    'beh_key': beh_key,
                    'exp_info_entry': s,
                }

    return unique_sessions


def get_date(datexp):
    """Parse date string like '2022_08_17' into datetime."""
    return datetime.strptime(datexp, '%Y_%m_%d')


def compute_days_per_mouse(unique_sessions):
    """Compute calendar days since the mouse's first session for each session."""
    mouse_dates = {}
    for key, info in unique_sessions.items():
        mname = info['mname']
        d = get_date(info['datexp'])
        if mname not in mouse_dates:
            mouse_dates[mname] = []
        mouse_dates[mname].append(d)

    mouse_first_date = {m: min(dates) for m, dates in mouse_dates.items()}

    days = {}
    for key, info in unique_sessions.items():
        d = get_date(info['datexp'])
        first_d = mouse_first_date[info['mname']]
        days[key] = (d - first_d).days
    return days


def compute_speed_quartiles(unique_sessions):
    """First pass: collect running speeds from all corridor frames to compute quartiles."""
    print("Computing speed quartiles across all sessions...")
    all_speeds = []

    beh_cache = {}
    for key, info in unique_sessions.items():
        exp_type = info['exp_type']
        beh_key = info['beh_key']

        if exp_type not in beh_cache:
            beh_cache[exp_type] = np.load(
                os.path.join(DATA_ROOT, 'beh', f'Beh_{exp_type}.npy'), allow_pickle=True
            ).item()
        beh = beh_cache[exp_type][beh_key]

        spk_fn = f"{info['mname']}_{info['datexp']}_{info['blk']}_neural_data.npy"
        spk_path = os.path.join(DATA_ROOT, 'spk', spk_fn)
        spk_data = np.load(spk_path, allow_pickle=True).item()['spks']
        nfr = spk_data[0].shape[1]

        ft_CorrSpc = beh['ft_CorrSpc'][:nfr]
        ft_RunSpeed = beh['ft_RunSpeed'][:nfr]

        corridor_speeds = ft_RunSpeed[ft_CorrSpc]
        all_speeds.append(corridor_speeds)

    all_speeds = np.concatenate(all_speeds)
    quartiles = np.percentile(all_speeds, [25, 50, 75])
    print(f"  Speed quartile boundaries: {quartiles}")
    return quartiles


def discretize_speed(speeds, quartiles):
    """Discretize running speed into 4 quartile bins (0, 1, 2, 3)."""
    bins = np.digitize(speeds, quartiles)  # Returns 0, 1, 2, 3
    return bins


def discretize_position(positions):
    """Discretize position into 4 equal bins of 10 dm (1 m) each.

    Position range in corridor: 0-40 dm (0-4 m).
    Bins: [0,10) -> 0, [10,20) -> 1, [20,30) -> 2, [30,40] -> 3
    """
    bins = np.clip(positions // 10, 0, 3).astype(int)
    return bins


def main():
    unique_sessions = collect_unique_sessions()
    print(f"Found {len(unique_sessions)} unique sessions")

    # Compute days since first session per mouse
    days_map = compute_days_per_mouse(unique_sessions)

    # Compute speed quartiles across all data
    speed_quartiles = compute_speed_quartiles(unique_sessions)

    # Define brain regions
    brain_regions = ['V1', 'mHV', 'lHV', 'aHV']

    # Collect all unique stimulus names across all sessions
    print("Collecting stimulus names...")
    all_stim_names = set()
    beh_cache = {}
    for key, info in unique_sessions.items():
        exp_type = info['exp_type']
        beh_key = info['beh_key']
        if exp_type not in beh_cache:
            beh_cache[exp_type] = np.load(
                os.path.join(DATA_ROOT, 'beh', f'Beh_{exp_type}.npy'), allow_pickle=True
            ).item()
        beh = beh_cache[exp_type][beh_key]
        for name in beh['UniqWalls']:
            all_stim_names.add(name)

    stim_names = sorted(all_stim_names)
    stim_to_idx = {name: i for i, name in enumerate(stim_names)}
    print(f"  Stimulus names: {stim_names}")

    # Process each session
    neural_all = []
    input_all = []
    output_all = []
    subject_idx_all = []
    brain_region_idx_all = []
    subjects_set = []
    session_info = []

    # Compute time bin size from first session
    first_key = list(unique_sessions.keys())[0]
    first_info = unique_sessions[first_key]
    first_beh_file = np.load(
        os.path.join(DATA_ROOT, 'beh', f'Beh_{first_info["exp_type"]}.npy'),
        allow_pickle=True
    ).item()
    first_beh = first_beh_file[first_info['beh_key']]
    ft = first_beh['ft']
    dt_days = np.median(np.diff(ft))
    time_bin_ms = dt_days * 24 * 3600 * 1000  # convert days to ms
    time_bin_s = time_bin_ms / 1000
    print(f"Time bin size: {time_bin_ms:.1f} ms ({1000/time_bin_ms:.2f} Hz)")

    skipped_sessions = 0
    for sess_idx, (key, info) in enumerate(unique_sessions.items()):
        mname = info['mname']
        datexp = info['datexp']
        blk = info['blk']
        exp_type = info['exp_type']
        beh_key = info['beh_key']

        print(f"\nProcessing {key} ({sess_idx+1}/{len(unique_sessions)})...")

        # Load behavior
        if exp_type not in beh_cache:
            beh_cache[exp_type] = np.load(
                os.path.join(DATA_ROOT, 'beh', f'Beh_{exp_type}.npy'), allow_pickle=True
            ).item()
        beh = beh_cache[exp_type][beh_key]

        # Load neural data
        try:
            spk = load_spk(mname, datexp, blk, root=os.path.join(DATA_ROOT, 'spk'))
        except Exception as e:
            print(f"  Error loading neural data: {e}, skipping")
            skipped_sessions += 1
            continue

        nneu, nfr = spk.shape

        # Load retinotopy and filter neurons to visual cortex
        try:
            iarea = load_retino(mname, datexp, root=os.path.join(DATA_ROOT, 'retinotopy'))
        except Exception as e:
            print(f"  Error loading retinotopy: {e}, skipping")
            skipped_sessions += 1
            continue

        area_idx = neu_area_ID(iarea)
        # Include neurons in any of the four visual areas
        valid_neurons = area_idx['V1'] | area_idx['mHV'] | area_idx['lHV'] | area_idx['aHV']
        neuron_indices = np.where(valid_neurons)[0]

        if len(neuron_indices) == 0:
            print(f"  No valid neurons, skipping")
            skipped_sessions += 1
            continue

        # Filter neural data to valid neurons
        spk_filtered = spk[neuron_indices, :]

        # Build brain region index for filtered neurons
        region_idx = np.zeros(len(neuron_indices), dtype=int)
        for r_idx, region in enumerate(brain_regions):
            mask = area_idx[region][neuron_indices]
            region_idx[mask] = r_idx

        # Get frame-level behavior data, truncated to neural frames
        ft_trInd = beh['ft_trInd'][:nfr]
        ft_CorrSpc = beh['ft_CorrSpc'][:nfr]
        ft_Pos = beh['ft_Pos'][:nfr]
        ft_RunSpeed = beh['ft_RunSpeed'][:nfr]

        ntrials = beh['ntrials']
        WallName = beh['WallName']
        isRew = beh['isRew']
        SoundFr = beh['SoundFr']

        # Lick data
        LickFr = beh['LickFr']
        LickTrind = beh['LickTrind']

        # Day of training
        day_of_training = days_map[key]

        # Process each trial
        neural_trials = []
        input_trials = []
        output_trials = []

        for t in range(ntrials):
            # Find corridor frames for this trial
            mask = (ft_trInd == t) & ft_CorrSpc
            frames = np.where(mask)[0]

            if len(frames) < MIN_FRAMES_PER_TRIAL:
                continue

            n_frames = len(frames)

            # Neural data: (n_neurons, n_timepoints)
            neural_trial = spk_filtered[:, frames]

            # --- Inputs ---
            # Time to sound cue (seconds): positive before cue, negative after
            sound_fr = SoundFr[t]
            time_to_cue = (sound_fr - frames) * time_bin_s

            # Day of training (per-trial, repeated)
            day_arr = np.full(n_frames, day_of_training, dtype=np.float32)

            # Time since trial start (seconds)
            time_since_start = (frames - frames[0]) * time_bin_s

            # Reward availability (per-trial, repeated)
            reward_avail = np.full(n_frames, float(isRew[t]), dtype=np.float32)

            input_trial = np.stack([time_to_cue, day_arr, time_since_start, reward_avail], axis=0)

            # --- Outputs ---
            # Visual stimulus category (per-trial, repeated as time-varying)
            stim_cat = stim_to_idx[WallName[t]]
            stim_arr = np.full(n_frames, stim_cat, dtype=int)

            # Licking: binary per frame
            lick_arr = np.zeros(n_frames, dtype=int)
            # Find licks in this trial
            trial_lick_mask = (LickTrind.astype(int) == t)
            trial_lick_frames = np.round(LickFr[trial_lick_mask]).astype(int)
            # Mark frames that have a lick
            frame_set = set(frames.tolist())
            frame_to_local = {f: i for i, f in enumerate(frames)}
            for lf in trial_lick_frames:
                if lf in frame_to_local:
                    lick_arr[frame_to_local[lf]] = 1

            # Position in corridor: 4 bins
            pos_bins = discretize_position(ft_Pos[frames])

            # Running speed: 4 quartile bins
            speed_bins = discretize_speed(ft_RunSpeed[frames], speed_quartiles)

            output_trial = np.stack([stim_arr, lick_arr, pos_bins, speed_bins], axis=0)

            neural_trials.append(neural_trial)
            input_trials.append(input_trial.astype(np.float32))
            output_trials.append(output_trial.astype(int))

        if len(neural_trials) < 2:
            print(f"  Only {len(neural_trials)} valid trials, skipping session")
            skipped_sessions += 1
            continue

        # Add subject
        if mname not in subjects_set:
            subjects_set.append(mname)

        neural_all.append(neural_trials)
        input_all.append(input_trials)
        output_all.append(output_trials)
        subject_idx_all.append(subjects_set.index(mname))
        brain_region_idx_all.append(region_idx)
        session_info.append({
            'session_key': key,
            'exp_type': exp_type,
            'mname': mname,
            'datexp': datexp,
            'blk': blk,
            'day_of_training': day_of_training,
            'n_trials': len(neural_trials),
            'n_neurons': len(neuron_indices),
        })

        print(f"  {len(neural_trials)} trials, {len(neuron_indices)} neurons, day={day_of_training}")

    print(f"\n{'='*60}")
    print(f"Total sessions: {len(neural_all)} (skipped {skipped_sessions})")
    print(f"Total subjects: {len(subjects_set)}")

    # Build output_values
    # Stimulus category
    stim_values = stim_names

    # Licking
    lick_values = ['no_lick', 'lick']

    # Position bins
    pos_values = ['0-1m', '1-2m', '2-3m', '3-4m']

    # Speed bins
    speed_values = ['Q1 (slowest)', 'Q2', 'Q3', 'Q4 (fastest)']

    # Build data dictionary
    data = {
        'neural': neural_all,
        'input': input_all,
        'output': output_all,

        'subjects': subjects_set,
        'subject_idx': np.array(subject_idx_all, dtype=int),

        'brain_regions': brain_regions,
        'brain_region_idx': brain_region_idx_all,

        'input_names': ['time_to_sound_cue', 'day_of_training', 'time_since_trial_start', 'reward_availability'],
        'output_names': ['visual_stimulus', 'licking', 'position', 'running_speed'],
        'output_values': [stim_values, lick_values, pos_values, speed_values],

        'metadata': {
            'task_description': (
                'Visual discrimination task in head-fixed mice running through linear '
                'virtual reality corridors. Mice discriminate between visual texture patterns '
                '(e.g., leaf vs circle). A sound cue indicates reward availability in '
                'rewarded corridors. Neural activity from visual cortex (two-photon calcium '
                'imaging, deconvolved traces) is used to decode stimulus identity, licking, '
                'position, and running speed.'
            ),
            'time_bin_size': float(time_bin_ms),
            'temporal_alignment_event': 'Trial start (corridor entry)',
            'off_start': 0.0,
            'off_end': None,
            'calcium_frame_rate_hz': float(1000 / time_bin_ms),
            'corridor_length_m': 4.0,
            'speed_quartile_boundaries': speed_quartiles.tolist(),
            'session_info': session_info,
            'source': 'Zhong et al. 2025, Unsupervised pretraining in biological neural networks',
        }
    }

    # Save
    print(f"\nSaving to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, 'wb') as f:
        pickle.dump(data, f)
    print("Done!")

    # Print summary statistics
    total_trials = sum(len(s) for s in neural_all)
    total_timepoints = sum(trial.shape[1] for s in neural_all for trial in s)
    print(f"\nSummary:")
    print(f"  Sessions: {len(neural_all)}")
    print(f"  Subjects: {len(subjects_set)}")
    print(f"  Total trials: {total_trials}")
    print(f"  Total timepoints: {total_timepoints}")
    print(f"  Brain regions: {brain_regions}")
    print(f"  Stimulus categories: {stim_names}")


if __name__ == '__main__':
    main()
