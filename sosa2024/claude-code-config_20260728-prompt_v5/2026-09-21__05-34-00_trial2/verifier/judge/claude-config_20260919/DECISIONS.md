# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Every `.nwb` file under `/app/data/sub-*/` is discovered with a single sorted `glob` and processed in one pass (152 files; no second "survey" pass). Files are opened directly with `h5py` rather than `pynwb`, and only the specific datasets needed are read: `identifier`, `general/subject/subject_id`, `general/session_id`, `acquisition/TwoPhotonSeries/imaging_plane/imaging_rate`, the `iscell` table, the per-plane `processing/ophys/Deconvolved/plane*/data` arrays, and the `processing/behavior/BehavioralTimeSeries` series (`position`, `speed`, `lick`, `trial_start`, `teleport`, `trial number`, `environment`, `position/timestamps`, `Reward/timestamps`). Fluorescence/Neuropil are *not* read. The file is closed before any processing.

ii.
```python
DATA_DIR = '/app/data'

def get_all_nwb_files():
    """Get all NWB file paths, sorted by subject and session."""
    files = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', '*.nwb')))
    return files
```
```python
    with h5py.File(nwb_path, 'r') as f:
        identifier = f['identifier'][()].decode() ...
        iscell = f['processing/ophys/ImageSegmentation/PlaneSegmentation/iscell'][:, 0]
        deconv_group = f['processing/ophys/Deconvolved']
        plane_keys = sorted([k for k in deconv_group.keys() if k.startswith('plane')])
        plane_data = [deconv_group[pk]['data'][:] for pk in plane_keys]
        deconv_all = np.concatenate(plane_data, axis=1)  # (T, total_ROIs)
        bts = f['processing/behavior/BehavioralTimeSeries']
        position = bts['position/data'][:]
        ...
        timestamps = bts['position/timestamps'][:]
        reward_timestamps = bts['Reward/timestamps'][:]
```

iii. From CONVERSION_NOTES Step 2 the AI documented the directory layout as one level deep (`sub-mX/sub-mX_ses-YY_behavior+ophys.nwb`) and cross-checked the resulting counts against the paper: 11 mice, 14 sessions each except m11 (12, "imaging started on day 3"), 152 sessions total, 80.4 ± 6.1 trials/session vs. the paper's 80.5 ± 7.4, and 155–2341 cells/session vs. the paper's "155–2,172 putative pyramidal neurons per session". `h5py` was used instead of `pynwb` for speed; the whole conversion runs in 99.8 s.

## 1-b. How are the data split into subjects?

i. Subject identity is read from inside each file (`general/subject/subject_id`), not parsed from the path. A running `unique_subjects` list is built in order of first appearance, and `subject_idx` for each session is the index into that list. This yields 11 subjects in the order m11, m12, m13, m14, m15, m17, m18, m19, m3, m4, m7 (sorted-glob order).

ii.
```python
        subject_id = f['general/subject/subject_id'][()].decode() ...
```
```python
        subj = result['subject']
        if subj not in unique_subjects:
            unique_subjects.append(subj)
        subj_idx = unique_subjects.index(subj)
        ...
        subject_idx_list.append(subj_idx)
```
```python
        'subjects': unique_subjects,
        'subject_idx': np.array(subject_idx_list, dtype=np.int64),
```

iii. The AI verified the subject list against the paper: "11 switch-task mice (m3, m4, m7, m11-m15, m17-m19)" and noted that the three fixed-condition mice (m2, m6, m10) of the paper are absent from the released dataset, so no filtering of animals was needed.

## 1-c. How are the data split into sessions?

i. One NWB file = one session. The 152 files are processed in sorted order, so sessions are grouped by subject and ordered by session number within subject. `session_id` is read from `general/session_id` and stored in `metadata['session_info']`. No merging or alignment of cells across days is attempted.

ii.
```python
    for fpath in all_files:
        fname = os.path.basename(fpath)
        session_label = fname.replace('_behavior+ophys.nwb', '')
        result = process_session(fpath, show_processing=args.show_processing,
                                 session_label=session_label)
        if result is None:
            continue
        neural_all.append(result['neural'])
        ...
        session_info.append({'subject': subj, 'session_id': result['session_id'],
                             'scene': result['scene'], 'n_neurons': result['n_neurons'], ...})
```

iii. CONVERSION_NOTES Key Decision 3: "Each NWB session = one session in output: Each session is one continuous recording with aligned neural + behavioral data." The AI checked that this gives 152 sessions with 12–14 sessions per mouse, matching the paper's day 1–14 design and m11's late start.

## 1-d. How are the data split into trials?

i. Trial onsets are the indices where the `trial_start` behavioral time series is non-zero; trial offsets are the indices where `teleport` is non-zero. The i-th trial is the half-open slice `[trial_start_inds[i], teleport_inds[i])` of every behavioral and neural array, so the inter-trial/teleport period is excluded. If the numbers of starts and teleports disagree, both arrays are truncated from the front to the shorter length (this branch never fires — I confirmed the counts match in all 152 sessions, and that starts and teleports strictly interleave).

ii.
```python
    trial_start_inds = np.where(trial_start_data > 0)[0]
    teleport_inds = np.where(teleport_data > 0)[0]
    n_trials = len(trial_start_inds)

    if len(teleport_inds) != n_trials:
        min_len = min(len(trial_start_inds), len(teleport_inds))
        trial_start_inds = trial_start_inds[:min_len]
        teleport_inds = teleport_inds[:min_len]
        n_trials = min_len
    ...
    for i in range(n_trials):
        ts = trial_start_inds[i]
        te = teleport_inds[i]
        if te <= ts:
            continue
        trial_pos = position[ts:te]
        ...
```

iii. CONVERSION_NOTES Key Decision 5: "Trial segmentation: Use trial_start and teleport signals. Only include data between trial_start and teleport (excluding inter-trial intervals)." Step 3 records the same rule from the paper/reference code, and the resulting counts (mostly 80 trials/session, range 41–100, 12,216 total) were checked against the paper's 80.5 ± 7.4 trials/session.

## 1-e. How are trials filtered based on quality controls?

i. Very little trial-level curation. A trial is dropped only if (a) `teleport_inds[i] <= trial_start_inds[i]`, (b) it has fewer than 2 raw timepoints, or (c) fewer than 2 timepoints survive the speed ≥ 2 cm/s + non-NaN mask. A whole session is dropped if it has no `iscell` ROIs, fewer than 2 trials, or fewer than 2 surviving trials. No minimum trial-length criterion is applied. In practice none of these fired: all 12,216 trials and all 152 sessions were kept. Lick-sensor-error trials are *not* dropped — only their lick channel is zeroed (see 9-b).

ii.
```python
    if n_trials < 2:
        print(f"  WARNING: Only {n_trials} trials in {session_label}, skipping")
        return None
    ...
        if te <= ts:
            continue
        ...
        n_trial_tp = len(trial_pos)
        if n_trial_tp < 2:
            continue
        ...
        if valid_mask.sum() < 2:
            continue
    ...
    if len(neural_trials) < 2:
        print(f"  WARNING: Only {len(neural_trials)} valid trials in {session_label}, skipping")
        return None
```

iii. CONVERSION_NOTES Step 3, "Trial curation rules": "1. Lick sensor error: Trials with >35% of frames having cumulative lick count >2 have licks set to NaN. 2. All trials are included (no trial exclusion beyond the lick correction)." The ≥2-trial and ≥2-timepoint guards were added to satisfy the target-format requirement that each session contain at least two usable trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The `neural` array is taken **directly from the NWB `processing/ophys/Deconvolved/plane*/data` arrays** (suite2p's own `spks`), concatenated across planes and masked by `iscell[:,0] == 1`. The `Fluorescence` (F) and `Neuropil` (Fneu) arrays present in the same files are read by neither the loader nor any processing step.

ii.
```python
        iscell = f['processing/ophys/ImageSegmentation/PlaneSegmentation/iscell'][:, 0]
        cell_mask = iscell == 1
        n_neurons = int(cell_mask.sum())

        # Load deconvolved events from all planes and concatenate
        deconv_group = f['processing/ophys/Deconvolved']
        plane_keys = sorted([k for k in deconv_group.keys() if k.startswith('plane')])
        plane_data = [deconv_group[pk]['data'][:] for pk in plane_keys]
        deconv_all = np.concatenate(plane_data, axis=1)  # (T, total_ROIs)

        # Apply iscell mask
        deconv = deconv_all[:, cell_mask]  # (T, N_cells)
```

iii. CONVERSION_NOTES Step 1: "**dFF is NOT in the NWB** - the NWB contains raw F, Fneu, and deconvolved events. The reference code uses deconvolved events for the decoder, NOT dFF." Step 4: "Deconvolved data | events timeseries used | NWB has Deconvolved data already | deconvolution from dFF | **NWB deconvolved data is ready to use**." Key Decision 1: "Use deconvolved events (not dFF) as neural data: Consistent with reference decoder code which uses `sess.timeseries['events']`."

## 2-b. How is the `neural` data processed?

i. Essentially no signal processing is applied. The per-plane `Deconvolved` arrays are concatenated along the ROI axis (planes sorted `plane0`, `plane1`, matching the order of `PlaneSegmentation/planeIdx`), subset to `iscell` ROIs, sliced per trial, subset to the speed ≥ 2 cm/s timepoints, `nan_to_num`'d, transposed to (n_neurons, n_timepoints) and cast to `float32`. None of the paper's neural pipeline — neuropil subtraction with `neu_coef = 0.7`, per-trial maximin baseline over a 20 s sliding window, `(F − baseline)/|baseline|`, 2-sample Gaussian smoothing, OASIS deconvolution at `tau = 0.7` and `frame_rate/n_planes` — is reproduced.

ii.
```python
        deconv = deconv_all[:, cell_mask]  # (T, N_cells)
        ...
        trial_neural = deconv[ts:te, :]  # (T_trial, N)
        ...
        trial_neural_valid = trial_neural[valid_mask, :]  # (T_valid, N)
        # Replace NaN in neural data with 0 (as in reference code: X[np.isnan(X)] = 0)
        trial_neural_valid = np.nan_to_num(trial_neural_valid, nan=0.0)
        # --- Neural: (n_neurons, n_timepoints) ---
        neural_matrix = trial_neural_valid.T.astype(np.float32)  # (N, T)
```

iii. The stated rationale is that the stored `Deconvolved` array is already the signal the reference decoder consumes, so no further processing is needed ("NWB deconvolved data is ready to use"). The AI documented all of the paper's dF/F parameters in its Step 3 table (maximin, 20 s window, 2-sample Gaussian, OASIS) but did not implement any of them.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters, one at the cell level and one at the timepoint level:
- **Cells**: keep only ROIs with `iscell[:,0] == 1` (suite2p manual curation). The paper's second neuron filter — dropping putative interneurons whose dF/F correlates with running speed at r > 0.5 — is explicitly **skipped**.
- **Timepoints**: within every trial, all timepoints with `speed < 2 cm/s` are **deleted** from the neural matrix and from every input and output stream. A `~np.isnan(trial_neural[:, 0])` mask is also applied (checks only the first neuron).

This removes ~12.6% of all in-trial samples (mean T per trial 189.6 vs. 216.8 in the reference) and leaves the `speed` output with an empty class 0 (`speed` range reported as [1.0, 4.0] in `verification_full_out.txt`), even though the Decoder Task spec defines class 0 as "< 2 cm/s".

ii.
```python
        iscell = f['processing/ophys/ImageSegmentation/PlaneSegmentation/iscell'][:, 0]
        cell_mask = iscell == 1
```
```python
SPEED_THRESHOLD = 2.0  # cm/s  (exclude below this)
...
        # Speed threshold mask: keep timepoints with speed >= 2 cm/s
        speed_mask = trial_speed >= SPEED_THRESHOLD
        # Also exclude NaN neural data
        neural_nan_mask = ~np.isnan(trial_neural[:, 0])
        # Combined valid mask
        valid_mask = speed_mask & neural_nan_mask
        if valid_mask.sum() < 2:
            continue
        # Apply mask
        trial_pos_valid = trial_pos[valid_mask]
        trial_speed_valid = trial_speed[valid_mask]
        trial_lick_valid = trial_lick[valid_mask]
        trial_neural_valid = trial_neural[valid_mask, :]  # (T_valid, N)
        trial_times_valid = trial_timestamps[valid_mask]
```

iii. Key Decision 2: "Apply speed >= 2 cm/s threshold: Exclude timepoints below threshold by setting neural data to NaN, then removing NaN timepoints. This matches the reference code's `use_speed_thr=2`", backed by the Methods quote "excluded activity when the animal was moving at <2 cm/s". Key Decision 7: "Interneuron exclusion: Skip this filtering step. It affects <0.5% of cells and requires dFF computation which is not in the NWB. The iscell filtering already removes most non-pyramidal cells." In Step 10 the AI listed "Speed <2 cm/s properly filtered (0 timepoints in bin 0)" as a *passed* sanity check, and in the trajectory (step 69) wrote "bin 0 (<2 cm/s) is absent because we excluded those timepoints with the speed threshold. This is correct behavior since the reference code filters out speed < 2 cm/s."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment to trial start requires nothing beyond the trial slice: the neural array is indexed with the same `[trial_start_inds[i], teleport_inds[i])` window used for all behavioral streams, so sample 0 of every trial's neural matrix is the `trial_start` frame. `off_start = 0.0`, `off_end = None` (variable trial length), `temporal_alignment_event = 'start of trial (entry to virtual track)'`. There is no pre-trial window. No explicit check is made that the neural frame index and the behavioral sample index correspond (10 sessions in fact have one more neural frame than behavioral sample; because the surplus is at the end this is harmless).

ii.
```python
        trial_neural = deconv[ts:te, :]  # (T_trial, N)
        trial_timestamps = timestamps[ts:te]
        ...
        time_from_start = trial_times_valid - trial_timestamps[0]  # seconds
```
```python
            'temporal_alignment_event': 'start of trial (entry to virtual track)',
            'off_start': 0.0,  # trial starts at alignment event
            'off_end': None,  # variable trial length
```

iii. CONVERSION_NOTES Step 3: "Temporal alignment: VR behavior data is already aligned to imaging frames in the NWB." Key Decision 10: "Temporal alignment: Align to trial start (first imaging frame of the trial). Time = 0 at trial start."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning or resampling. Data are kept at the native imaging frame rate; `metadata['time_bin_size']` is hard-coded to 64.5 ms (the true per-sample interval is 64.4836 ms = 1/15.5078 s, identical in every session including the two-plane m17/m18 sessions where the scanner rate is 31 Hz but the per-plane rate is 15.5 Hz). Caveat: because low-speed samples are deleted (2-c), the retained samples within a trial are *not* uniformly spaced — the 64.5 ms figure describes the raw sampling grid, not the converted trial matrices.

ii.
```python
        imaging_rate = float(f['acquisition/TwoPhotonSeries/imaging_plane/imaging_rate'][()])
    ...
    frame_period = 1.0 / imaging_rate  # seconds per frame   (computed but never used)
```
```python
            'time_bin_size': 64.5,  # ms (approximate, ~1/15.5 Hz)
            ...
            'imaging_rate_hz': 15.5,
```

iii. Key Decision 4: "Time bin = one imaging frame (~64.5 ms): No additional temporal binning needed; data is already at imaging frame rate", supported by the Methods quote "imaging FOV collected at ~15.5 Hz".

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. From `processing/behavior/BehavioralTimeSeries/position/timestamps` (the behavioral timestamp vector, which is identical for every behavioral series in the file).

ii.
```python
        timestamps = bts['position/timestamps'][:]
    ...
        trial_timestamps = timestamps[ts:te]
        ...
        trial_times_valid = trial_timestamps[valid_mask]
```

iii. Not separately justified in CONVERSION_NOTES beyond the mapping-table entry "Time from trial start | input[0] | Compute from timestamps within each trial | Continuous, time-varying". The AI's Step 3 notes record that behavior is already aligned to imaging frames, so any of the per-series timestamp vectors is equivalent.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The timestamp of the **unmasked** first sample of the trial is subtracted from the retained timestamps, so t = 0 is the true `trial_start` frame even when the first samples were removed by the speed mask. The result is stored as `input[0]` (float32 seconds). Because of the speed mask the series is monotonically increasing but has gaps; per-session maxima range from ~14 s to 216.5 s.

ii.
```python
        # Time from start of trial in seconds
        time_from_start = trial_times_valid - trial_timestamps[0]  # seconds
        ...
        input_data = np.zeros((4, n_tp), dtype=np.float32)
        input_data[0, :] = time_from_start
```

iii. No explicit justification given; it is the direct reading of the decoder input spec "Time from start of trial in seconds (continuous, time-varying)".

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. By construction: `trial_times_valid` and `trial_neural_valid` are produced from the same `[ts:te]` slice and the same `valid_mask`, so element k of the time vector corresponds to column k of the neural matrix. No assertion compares the behavioral timestamp vector to the neural series' `starting_time`/`rate`, and no check is made that the neural and behavioral arrays have the same length (they differ by one sample in 10 sessions).

ii.
```python
        trial_neural = deconv[ts:te, :]
        trial_timestamps = timestamps[ts:te]
        ...
        trial_neural_valid = trial_neural[valid_mask, :]
        trial_times_valid = trial_timestamps[valid_mask]
```

iii. CONVERSION_NOTES Step 3/Step 4: "VR behavior data is already aligned to imaging frames in the NWB"; "Speed in NWB | Already computed and aligned | Confirmed | Use as-is."

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. From the **`identifier` string** of the NWB file (e.g. `/data/InVivoDA/GCAMP11/23_02_2023/Env1_LocationB_to_A`), whose last path component is the VR scene name. The `environment` behavioral time series *is* read into `environment_data` but is never used. On cross-environment scenes (`Env1_X_to_Env2_Y`) the environment is taken to change at trial 30.

ii.
```python
def parse_scene(identifier):
    """Extract scene name from NWB identifier field."""
    if '/' in identifier:
        return identifier.split('/')[-1]
    return identifier


def get_environment_from_scene(scene, n_trials, change_trial=SWITCH_TRIAL):
    env = np.zeros(n_trials, dtype=np.int64)
    if '_to_Env' in scene:
        parts = scene.split('_to_')
        env1_num = int(parts[0][3])  # "Env1..." -> 1
        env2_num = int(parts[1][3])  # "Env2..." -> 2
        env[:change_trial] = env1_num - 1  # 0-indexed
        env[change_trial:] = env2_num - 1
    elif scene.startswith('Env1'):
        env[:] = 0
    elif scene.startswith('Env2'):
        env[:] = 1
    else:
        raise ValueError(f"Cannot determine environment from scene: {scene}")
    return env
```
```python
        environment_data = bts['environment/data'][:]   # loaded, never used
    ...
    env_per_trial = get_environment_from_scene(scene, n_trials)
```

iii. The AI's mapping table (Step 5) states the source as the `environment` field — "Environment type | input[1] | From `environment` field: 0=ENV1, 1=ENV2" — which is not what the code does. The scene-based route is justified in Step 4 ("Session 08 for m11: scene=Env1_B_to_Env2_C. Environment switches from 0 to 1 at trial 30. Confirmed.") and Step 10 ("All environment switches at trial 30", "11 sessions with env switch").

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The scene string is parsed to `Env1` → 0 and `Env2` → 1; on the 11 cross-environment scenes, trials 0–29 get the first environment and trials 30+ the second. The per-trial value is broadcast across all timepoints of the trial into `input[1]` (float32).

ii.
```python
SWITCH_TRIAL = 30  # reward zone changes after this many trials on switch days
...
        env_type = env_per_trial[i]
        ...
        input_data[1, :] = env_type
```

iii. Step 1 notes: "Switch occurs at trial 30 (first 30 trials = pre-switch, remaining = post-switch)"; Step 10 sanity check: "All environment switches at trial 30". (I independently confirmed this reconstruction equals the per-trial median of the `environment` time series in all 12,216 trials of all 152 sessions.)

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. From the position of the trial in the `trial_start`/`teleport` sequence — i.e. the loop counter `i`, 0-indexed within the session. The stored `trial number` behavioral series is read into `trial_number_data` but never used.

ii.
```python
        trial_number_data = bts['trial number/data'][:]   # loaded, never used
    ...
    for i in range(n_trials):
        ...
        trial_num = float(i)  # 0-indexed trial number
        ...
        input_data[2, :] = trial_num
```

iii. The mapping table states "Trial number | input[2] | From `trial number` field (0-indexed within session) | Continuous, per-trial", which does not match the implementation. (The two are numerically identical here: the stored `trial number` is exactly 0…n−1 within the `trial_start`→`teleport` windows.)

## 5-b. What processing is involved in computing `input` *Trial number*?

i. None beyond casting the loop index to float and broadcasting it constant across the trial's timepoints. Ranges are [0, 79] for 80-trial sessions, up to [0, 99].

ii.
```python
        trial_num = float(i)  # 0-indexed trial number
        ...
        input_data[2, :] = trial_num
```

iii. Follows directly from the decoder input spec "Trial number (continuous, per trial)".

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. From `processing/behavior/BehavioralTimeSeries/Reward/timestamps` (the reward-delivery event times, which have their own timestamp vector), combined with the per-trial time windows given by the behavioral timestamps at `trial_start` and `teleport`.

ii.
```python
        # Reward events
        reward_timestamps = bts['Reward/timestamps'][:]
    ...
    reward_per_trial = np.zeros(n_trials, dtype=np.int64)
    for i in range(n_trials):
        ts = trial_start_inds[i]
        te = teleport_inds[i]
        trial_time_start = timestamps[ts]
        trial_time_end = timestamps[te] if te < len(timestamps) else timestamps[-1]
        rewards_in_trial = np.sum((reward_timestamps >= trial_time_start) &
                                   (reward_timestamps <= trial_time_end))
        reward_per_trial[i] = 1 if rewards_in_trial > 0 else 0
```

iii. Mapping table: "Previous trial outcome | input[3] | Derived from reward delivery: previous trial rewarded=1, omitted=0 | Binary, per-trial. First trial = 0 (no previous)". The AI cross-checked the resulting reward rate against the paper's "reward was randomly omitted on ~15% of trials": it obtained 84.7% rewarded per trial.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. The per-trial reward indicator is shifted forward by one trial; the first trial of each session is assigned 0. The value is broadcast constant across the trial's timepoints.

ii.
```python
    # --- Previous trial outcome ---
    prev_outcome = np.zeros(n_trials, dtype=np.int64)
    prev_outcome[1:] = reward_per_trial[:-1]  # First trial: 0 (no previous)
    ...
        prev_out = float(prev_outcome[i])
        ...
        input_data[3, :] = prev_out
```

iii. Direct implementation of the spec "Previous trial outcome (binary, omitted = 0, rewarded = 1, per trial)", with the first trial defaulted to 0 because there is no preceding trial.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. From two sources: the `position` behavioral time series, and the reward-zone boundaries for that trial, which are obtained by **parsing the VR scene name out of the NWB `identifier`** (not from the `reward_zone` time series, which is never read). Scene names of the form `EnvN_LocationX` give a constant zone; `EnvN_LocationX_to_Y` and `EnvN_X_to_EnvM_Y` give zone X for trials 0–29 and zone Y for trials 30+. Zone coordinates come from a hard-coded dictionary A = [80, 130], B = [200, 250], C = [320, 370] cm.

ii.
```python
REWARD_ZONE_DICT = {'A': (80, 130), 'B': (200, 250), 'C': (320, 370)}
SWITCH_TRIAL = 30

def get_reward_zones_from_scene(scene, n_trials, change_trial=SWITCH_TRIAL):
    rz_coords = np.zeros((n_trials, 2))
    rz_labels = np.empty(n_trials, dtype='U1')

    # Single-location scenes
    for loc in ['A', 'B', 'C']:
        if scene.endswith(f'Location{loc}') and '_to_' not in scene:
            rz_coords[:] = REWARD_ZONE_DICT[loc]
            rz_labels[:] = loc
            return rz_coords, rz_labels

    # Switch scenes within same environment: X_to_Y
    if '_to_' in scene and 'Env' not in scene.split('_to_')[1].split('_')[0] if '_' in scene.split('_to_')[1] else True:
        parts = scene.split('_to_')
        loc1 = parts[0][-1]
        loc2 = parts[1][-1]
        rz_coords[:change_trial] = REWARD_ZONE_DICT[loc1]
        rz_labels[:change_trial] = loc1
        rz_coords[change_trial:] = REWARD_ZONE_DICT[loc2]
        rz_labels[change_trial:] = loc2
        return rz_coords, rz_labels

    # Cross-environment switch
    if '_to_Env' in scene:
        ...
    raise ValueError(f"Cannot parse scene: {scene}")
```
```python
        position = bts['position/data'][:]
    ...
    rz_coords, rz_labels = get_reward_zones_from_scene(scene, n_trials)
```

iii. Step 1 identified the reference function `get_reward_zones` in `behavior.py` as "Gets reward zone coordinates and labels per trial from scene name", and Step 4 records "Reward zone location | From scene name in identifier | Confirmed via position check | A=[80,130], B=[200,250], C=[320,370] | Parse from identifier/scene". The `reward_zone` NWB field was examined and rejected as a source: "Reward_zone NWB field | Cumulative count (like lick) | rzone>0 indicates in reward zone; not needed for reward zone location". Step 10 reports the class balance check: 33.0 / 33.8 / 33.2 % for zones A/B/C.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. For each retained timepoint, the signed distance to the nearest point of the active reward zone is computed: `position − rz_start` when before the zone, `position − rz_end` when past it, and exactly 0.0 while inside. The continuous distance is then discretized (7-c) and written to `output[0]`.

ii.
```python
def compute_distance_to_reward_zone(position, rz_start, rz_end):
    distance = np.zeros_like(position)
    before = position < rz_start
    after = position > rz_end
    inside = ~before & ~after

    distance[before] = position[before] - rz_start
    distance[after] = position[after] - rz_end
    distance[inside] = 0.0
    return distance
```
```python
        rz_start = rz_coords[i, 0]
        rz_end = rz_coords[i, 1]
        dist = compute_distance_to_reward_zone(trial_pos_valid, rz_start, rz_end)
        dist_disc = discretize_distance(dist)
        ...
        output_data[0, :] = dist_disc
```

iii. Key Decision 9: "Distance to reward zone: Signed linear distance from animal position to nearest edge of the 50 cm reward zone active on that trial", matching the paper's 50 cm hidden zone and the reference `reward_zone_dict`.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Seven explicit boolean masks, exactly the bin edges in the Decoder Task spec: 0 = d < −50; 1 = −50 ≤ d < −10; 2 = −10 ≤ d < 0; 3 = d == 0 (in zone); 4 = 0 < d ≤ 10; 5 = 10 < d ≤ 50; 6 = d > 50. (The reference puts the ±10 and ±50 edges on the lower side, i.e. `d < 10`/`d < 50`; the difference affects only samples landing exactly on an edge.)

ii.
```python
def discretize_distance(distance):
    """Discretize distance to reward zone into 7 bins."""
    out = np.zeros(len(distance), dtype=np.int64)
    out[distance < -50] = 0
    out[(distance >= -50) & (distance < -10)] = 1
    out[(distance >= -10) & (distance < 0)] = 2
    out[distance == 0] = 3  # in zone
    out[(distance > 0) & (distance <= 10)] = 4
    out[(distance > 10) & (distance <= 50)] = 5
    out[distance > 50] = 6
    return out
```

iii. CONVERSION_NOTES "Output Discretization": "Distance to reward zone (7 bins): Compute signed distance to nearest point in reward zone. If position is within the zone, distance=0. Negative = before zone, positive = after zone. 0: < -50 cm, 1: -50 to -10 cm, 2: -10 to <0 cm, 3: 0 cm (in zone), 4: >0 to +10 cm, 5: +10 to +50 cm, 6: >+50 cm" — i.e. copied from the instructions. All 7 classes are populated in the full dataset (fractions 0.275 / 0.115 / 0.046 / 0.199 / 0.022 / 0.078 / 0.265).

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. It is computed from `trial_pos_valid`, which is the same `[ts:te]` slice and the same `valid_mask` used for the neural matrix, so output column k corresponds to neural column k. No lag or shift is introduced.

ii.
```python
        trial_pos = position[ts:te]
        trial_neural = deconv[ts:te, :]
        ...
        trial_pos_valid = trial_pos[valid_mask]
        trial_neural_valid = trial_neural[valid_mask, :]
```

iii. Relies on the documented fact that behavior and imaging frames are one-to-one in the NWB ("VR behavior data is already aligned to imaging frames in the NWB").

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. From the `position` behavioral time series (`processing/behavior/BehavioralTimeSeries/position/data`), in cm along the 450 cm virtual track.

ii.
```python
        position = bts['position/data'][:]
    ...
        trial_pos = position[ts:te]
        trial_pos_valid = trial_pos[valid_mask]
```

iii. Mapping table: "Absolute position | output[1] | From `position`, discretize into 5 bins of 90 cm | Time-varying, 5 classes." Step 5 sanity check: "Verify position range is [0, 450] cm".

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. None beyond the per-trial slice, the speed mask, and discretization; the raw cm values are used directly.

ii.
```python
        pos_disc = discretize_position(trial_pos_valid)
        ...
        output_data[1, :] = pos_disc
```

iii. "Absolute position (5 bins): 0: <90, 1: 90-180, 2: 180-270, 3: 270-360, 4: >360", with the 450 cm track length taken from the Methods.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Five 90 cm bins with open first and last edges: 0 = p < 90; 1 = 90 ≤ p < 180; 2 = 180 ≤ p < 270; 3 = 270 ≤ p < 360; 4 = p ≥ 360. The open ends absorb the handful of samples marginally outside [0, 450].

ii.
```python
def discretize_position(position):
    """Discretize absolute position into 5 equal bins spanning 450 cm."""
    out = np.zeros(len(position), dtype=np.int64)
    out[position < 90] = 0
    out[(position >= 90) & (position < 180)] = 1
    out[(position >= 180) & (position < 270)] = 2
    out[(position >= 270) & (position < 360)] = 3
    out[position >= 360] = 4
    return out
```

iii. Directly from the Decoder Task spec "Discretized into 5 equal-sized bins spanning the 450 cm track". Resulting fractions are 0.199 / 0.194 / 0.224 / 0.217 / 0.166, which the AI noted as consistent with roughly uniform track coverage.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same index slice and same `valid_mask` as the neural matrix; no further alignment.

ii.
```python
        trial_pos = position[ts:te]
        trial_pos_valid = trial_pos[valid_mask]
        trial_neural_valid = trial_neural[valid_mask, :]
```

iii. Same justification as 7-d: behavior and imaging frames are one-to-one in the NWB.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. From the `lick` behavioral time series (`processing/behavior/BehavioralTimeSeries/lick/data`), a per-frame lick count taking integer values 0–7.

ii.
```python
        lick_raw = bts['lick/data'][:]
```

iii. Mapping table: "Lick | output[3] | From `lick`, binarize (>0 = 1) | Time-varying, binary."

## 9-b. What processing is involved in computing `output` *Lick*?

i. Two steps. First, the paper's **lick-sensor-error correction**: for each trial, if more than 35% of its frames have a lick value > 2, the whole trial's lick trace is set to NaN. Second, binarization: any value > 0 → 1; NaN → 0. So flagged trials are emitted as all-zero lick traces rather than being dropped.

ii.
```python
LICK_ERROR_THRESHOLD = 0.35  # fraction of frames with cumulative lick >2
...
    # --- Lick sensor error correction ---
    lick_corrected = lick_raw.copy()
    for i in range(n_trials):
        ts = trial_start_inds[i]
        te = teleport_inds[i]
        trial_licks = lick_corrected[ts:te]
        if len(trial_licks) > 0:
            frac_error = np.sum(trial_licks > 2) / len(trial_licks)
            if frac_error > LICK_ERROR_THRESHOLD:
                lick_corrected[ts:te] = np.nan

    # Binarize licks
    lick_binary = np.zeros_like(lick_corrected)
    lick_binary[lick_corrected > 0] = 1
    lick_binary[np.isnan(lick_corrected)] = 0  # NaN licks -> 0
```

iii. Key Decision 6: "Lick sensor error correction: Apply the 0.35 threshold from the code (>35% of frames with cumulative lick >2). Set licks to NaN on those trials. Binarize remaining licks (>0 -> 1)." Step 4 records the code-vs-methods discrepancy explicitly: "Lick error threshold | 0.35 (glmUtils.py line 112) | N/A | 0.30 (methods.txt) | Use 0.35 from code - this is the actual implementation", and Step 3 notes the paper's expected incidence "~0.65% of all imaged trials, n=81 out of 12,376 trials".

## 9-c. How is `output` *Lick* aligned with the neural data?

i. The binarized lick trace is sliced with the same `[ts:te]` window and masked with the same `valid_mask` as the neural matrix, so it is sample-for-sample aligned. Note that the speed mask preferentially removes low-speed frames, which is when the animals lick most: the converted lick-positive fraction is 0.172, versus 0.230 in the reference conversion.

ii.
```python
        trial_lick = lick_binary[ts:te]
        ...
        trial_lick_valid = trial_lick[valid_mask]
        lick_disc = trial_lick_valid.astype(np.int64)
        output_data[3, :] = lick_disc
```

iii. Same justification as 7-d/8-d.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Same source as 7-a: the scene name parsed from the NWB `identifier`, mapped through `REWARD_ZONE_LABEL_MAP` to A = 0, B = 1, C = 2, with the switch at trial 30 on switch days. The `reward_zone` behavioral time series is not used.

ii.
```python
REWARD_ZONE_LABEL_MAP = {'A': 0, 'B': 1, 'C': 2}
...
    rz_coords, rz_labels = get_reward_zones_from_scene(scene, n_trials)
    ...
        rz_label = REWARD_ZONE_LABEL_MAP[rz_labels[i]]
```

iii. See 7-a: the AI followed the reference code's `get_reward_zones` (which reads the scene name) and verified the assignment against animal position, e.g. Step 4: "m11 ses-03 scene=Env1_LocationB_to_A: reward zone at ~200-210 cm (zone B) -> switches to zone A. This matches the expected pattern."

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. String parsing of the scene name into one or two zone letters, the letter→integer map, and broadcasting the per-trial integer across all timepoints of the trial into `output[4]`.

ii.
```python
        reward_out = reward_per_trial[i]
        ...
        output_data[4, :] = rz_label  # broadcast per-trial
```

iii. The format spec asks for per-trial variables to be made time-varying where possible; the AI therefore emits a constant time series rather than a scalar. Class balance was checked: 33.0 / 33.8 / 33.2 %.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. From `processing/behavior/BehavioralTimeSeries/Reward/timestamps` — the same array used for 6-a.

ii.
```python
        reward_timestamps = bts['Reward/timestamps'][:]
```

iii. "Reward outcome | output[5] | From reward delivery within trial | Per-trial, binary."

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, the number of reward event timestamps falling in the closed interval `[timestamps[trial_start], timestamps[teleport]]` is counted; the outcome is 1 if that count is > 0 and 0 otherwise. The value is broadcast constant across all timepoints of the trial. Reward events are compared in *time*, so no index-snapping/`searchsorted` step is needed. The resulting per-trial reward rate is 84.7%.

ii.
```python
    reward_per_trial = np.zeros(n_trials, dtype=np.int64)
    for i in range(n_trials):
        ts = trial_start_inds[i]
        te = teleport_inds[i]
        trial_time_start = timestamps[ts]
        trial_time_end = timestamps[te] if te < len(timestamps) else timestamps[-1]
        rewards_in_trial = np.sum((reward_timestamps >= trial_time_start) &
                                   (reward_timestamps <= trial_time_end))
        reward_per_trial[i] = 1 if rewards_in_trial > 0 else 0
    ...
        output_data[5, :] = reward_out  # broadcast per-trial
```

iii. Checked against the paper's "reward was randomly omitted on ~15% of trials": Step 5 sanity check "Verify reward rate matches (~85% rewarded) -> 82.8% overall, individual sessions 74-95%" (82.8% is the timepoint-weighted figure; the per-trial figure printed by the conversion is 84.7%).

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The cases handled are:
- **Sessions with no curated cells** → session skipped with a warning (never fired).
- **Mismatched numbers of `trial_start` and `teleport` pulses** → both index arrays silently truncated from the front to the shorter length (never fired; I verified all 152 sessions have equal counts and strictly interleaved pulses).
- **Degenerate trials** (`teleport <= trial_start`, < 2 raw samples, < 2 samples surviving the speed/NaN mask) → trial skipped (never fired).
- **Sessions left with < 2 trials** → session skipped (never fired).
- **NaNs in the neural data** → timepoints where neuron 0 is NaN are dropped, and any remaining NaNs are replaced with 0 via `np.nan_to_num`.
- **Stuck lick sensor** → trial's lick trace set to NaN then emitted as 0 (see 9-b).
- **Multi-plane sessions** (m17, m18) → per-plane arrays concatenated in sorted plane order before applying the ROI-level `iscell` mask (this ordering does match `PlaneSegmentation/planeIdx`; an earlier version crashed here and was fixed).

Not handled explicitly: the neural and behavioral arrays differ in length by one sample in 10 sessions (m17 ses-04/06, m18 ses-01/05/07/10/11/12/13/14). Because the extra sample is always at the *end* of the neural array and all indexing is driven by the behavioral arrays, this is harmless, but the script neither detects nor reports it.

ii.
```python
        if n_neurons == 0:
            print(f"  WARNING: No cells in {session_label}, skipping")
            return None
    ...
    if len(teleport_inds) != n_trials:
        # Trim to match
        min_len = min(len(trial_start_inds), len(teleport_inds))
        trial_start_inds = trial_start_inds[:min_len]
        teleport_inds = teleport_inds[:min_len]
        n_trials = min_len
    ...
        neural_nan_mask = ~np.isnan(trial_neural[:, 0])
        valid_mask = speed_mask & neural_nan_mask
        if valid_mask.sum() < 2:
            continue
        ...
        trial_neural_valid = np.nan_to_num(trial_neural_valid, nan=0.0)
```

iii. Step 10 records the corresponding checks: "No NaN in neural data (0/2.1B values)"; the multi-plane `iscell` bug is documented in the trajectory ("Cause: `iscell` had shape (2468, 2) covering all ROIs across both planes, but `Deconvolved/plane0/data` only had 1135 ROIs for plane0") and fixed by concatenating planes before masking. The trim-to-shortest fallback is an undocumented defensive guard.

## 13-a. What are the most time-consuming steps of the code?

i. The conversion is I/O dominated and fast: 99.8 s total for all 152 sessions, of which 8.6 s is the final `pickle.dump` of the 8.2 GB output. Per-session wall time (printed for every session) ranges from 0.1 s to ~2 s and tracks the size of the `Deconvolved` array being read — i.e. the dominant cost is reading (T × n_ROIs) float32 arrays out of HDF5, followed by the boolean-mask copies (`deconv_all[:, cell_mask]`, then per-trial `trial_neural[valid_mask, :]`) and the pickling/serialisation. The step that dominates the reference implementation — the dF/F + OASIS deconvolution — is absent here, which is the main reason this script is fast.

ii.
```python
    t0 = time.time()
    ...
    t1 = time.time()
    print(f"  {session_label}: {n_neurons} cells, {len(neural_trials)} trials, {t1-t0:.1f}s")
```
```python
    t_save = time.time()
    with open(args.outfile, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    ...
    print(f"Total time: {t_end - t_start:.1f}s (save: {t_end - t_save:.1f}s)")
```

iii. CONVERSION_NOTES has no written timing analysis (Steps 6 and 7 are left empty apart from "Status: COMPLETE"), but the script itself instruments per-session and save timing, and the totals are recorded in `conversion_full_out.txt`. The instruction's 15-minute budget is met by a wide margin.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. Three separate `for i in range(n_trials)` loops run over the same trial boundaries:
1. the reward-per-trial loop — fully vectorizable with `np.searchsorted(reward_timestamps, timestamps[trial_start_inds])` vs. `timestamps[teleport_inds]`;
2. the lick-sensor-error loop — vectorizable with `np.add.reduceat` over `(lick_raw > 2)`;
3. the main per-trial extraction loop — intrinsically ragged, but the position/speed/distance discretization inside it could be done once on the whole-session arrays before slicing.

In addition, the three `discretize_*` helpers each allocate and evaluate 5–7 boolean masks over the array where a single `np.digitize` call (as the reference uses) would do one pass. None of this matters in practice at 100 s total runtime.

ii.
```python
    for i in range(n_trials):          # loop 1: reward per trial
        ...
        rewards_in_trial = np.sum((reward_timestamps >= trial_time_start) &
                                   (reward_timestamps <= trial_time_end))
    ...
    for i in range(n_trials):          # loop 2: lick sensor error
        ...
            frac_error = np.sum(trial_licks > 2) / len(trial_licks)
    ...
    for i in range(n_trials):          # loop 3: main extraction
        ...
        dist_disc = discretize_distance(dist)
        pos_disc = discretize_position(trial_pos_valid)
        speed_disc = discretize_speed(trial_speed_valid)
```

iii. Not discussed in CONVERSION_NOTES. The ragged per-trial structure is the natural one given variable trial lengths, and the script already meets the runtime requirement, so no vectorization work was done.

## 13-c. What processing does the code repeat multiple times?

i. Each NWB file is opened and read exactly **once** — there is no separate survey/statistics pass, so the reference's duplicated read of every file is avoided. What is repeated within a session is the per-trial windowing: the `[trial_start_inds[i], teleport_inds[i])` window is recomputed and re-scanned three times (once for reward, once for lick correction, once for extraction), and `timestamps[ts]`/`timestamps[te]` lookups are repeated. `np.isnan` is effectively evaluated twice on the neural slice (`neural_nan_mask` and then `nan_to_num`). In `--show-processing` mode, the plotting function recomputes several quantities already present in `result`.

ii.
```python
    for i in range(n_trials):
        ts = trial_start_inds[i]; te = teleport_inds[i]     # reward
    for i in range(n_trials):
        ts = trial_start_inds[i]; te = teleport_inds[i]     # lick
    for i in range(n_trials):
        ts = trial_start_inds[i]; te = teleport_inds[i]     # extraction
```
```python
        neural_nan_mask = ~np.isnan(trial_neural[:, 0])
        ...
        trial_neural_valid = np.nan_to_num(trial_neural_valid, nan=0.0)
```

iii. Not discussed in CONVERSION_NOTES. The single-pass design is a deliberate simplification enabled by deriving the reward zone from the scene name, which removes the need for a dataset-wide survey before conversion.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Small amounts:
- `trial_number_data = bts['trial number/data'][:]` and `environment_data = bts['environment/data'][:]` are read from disk on every session and **never used** (both quantities are reconstructed from the loop index and the scene name instead).
- `frame_period = 1.0 / imaging_rate` and `n_timepoints = len(position)` are computed and never used.
- `rz_coords` is computed for every trial even in single-location sessions where it is constant.
- The two per-trial outputs (`reward_zone_location`, `reward_outcome`) and the three per-trial inputs are materialised as full-length time series, and all outputs are stored as `int64`; this inflates the pickle (8.2 GB) well beyond what the information content requires. This is arguably required by the target format's preference for time-varying representations, but `int8`/`int16` would have carried the same information.
- Conversely, the `Fluorescence`, `Neuropil` and `reward_zone` arrays are never read at all, so no time is wasted on them.

ii.
```python
        trial_number_data = bts['trial number/data'][:]      # never used
        environment_data = bts['environment/data'][:]        # never used
    ...
    n_timepoints = len(position)                              # never used
    ...
    frame_period = 1.0 / imaging_rate  # seconds per frame    # never used
    ...
        output_data = np.zeros((n_output_tv + n_output_pt, n_tp), dtype=np.int64)
        ...
        output_data[4, :] = rz_label  # broadcast per-trial
        output_data[5, :] = reward_out  # broadcast per-trial
```

iii. Not discussed in CONVERSION_NOTES. The broadcasting of per-trial variables is justified by the target-format instruction "If at all possible, make it time-varying"; the unused reads appear to be leftovers from the earlier plan (recorded in the Step 5 mapping table) to take environment and trial number from those behavioural series.
