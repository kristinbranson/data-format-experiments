# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI globs every NWB file under `data/sub-*/` (relative to the working directory `/app`), sorts them, and treats each file as one session. 152 files are found, matching the number of session files in `/app/data`. Files are opened directly with `h5py` rather than with `pynwb`, and the needed HDF5 datasets are read eagerly into memory (`[()]`): the eleven `BehavioralTimeSeries` streams, the `Deconvolved` ROI response series for every imaging plane, and the `iscell` table from `ImageSegmentation/PlaneSegmentation`. Each file is opened exactly once; there is no separate survey pass.

ii.
```python
    # Find all NWB files
    all_files = sorted(glob.glob('data/sub-*/*.nwb'))
    print(f"Found {len(all_files)} NWB files")
    ...
        files_to_process = all_files
        print(f"Full mode: processing {len(files_to_process)} sessions")
```
```python
    with h5py.File(nwb_path, 'r') as f:
        # --- Metadata ---
        identifier = f['identifier'][()].decode()
        scene = identifier.split('/')[-1]
        sess_id = f['general/session_id'][()].decode()
        subj_name = os.path.basename(nwb_path).split('_')[0]  # e.g., sub-m11
        ...
        behav = f['processing/behavior/BehavioralTimeSeries']
        position = behav['position/data'][()]
        speed = behav['speed/data'][()]
        lick = behav['lick/data'][()]
        reward_zone_signal = behav['reward_zone/data'][()]
        environment = behav['environment/data'][()]
        trial_start_signal = behav['trial_start/data'][()]
        teleport_signal = behav['teleport/data'][()]
        trial_number = behav['trial number/data'][()]
        timestamps = behav['position/timestamps'][()]
        reward_timestamps = behav['Reward/timestamps'][()]
        iscell = f['processing/ophys/ImageSegmentation/PlaneSegmentation/iscell'][()]
        planes = sorted(f['processing/ophys/Deconvolved'].keys())
        deconv_planes = []
        for plane in planes:
            deconv_planes.append(f[f'processing/ophys/Deconvolved/{plane}/data'][()])
        deconv_data = np.concatenate(deconv_planes, axis=1)  # (n_timepoints, total_rois)
```

iii. From CONVERSION_NOTES.md Step 2: "NWB files: `data/sub-{mouse}/sub-{mouse}_ses-{session}_behavior+ophys.nwb`; 11 subjects ... 152 sessions". The AI cross-checked the resulting counts (11 subjects, 152 sessions, 12,216 trials, 138,678 `iscell` neurons) against the paper (11 switch mice, 12,376 trials) in Step 9 and judged them consistent. The trajectory shows it chose raw `h5py` after exploring the NWB hierarchy, for speed (~1.3 s/session, 200 s total).

## 1-b. How are the data split into subjects?

i. Subject identity is taken from the first underscore-delimited field of the NWB file name (e.g. `sub-m11`). A `subject_list` is built incrementally in the order subjects are first encountered while iterating the sorted file list, and `subject_idx` records the index of that subject for each session. The `sub-` prefix is kept as part of the subject name. Eleven subjects result (`sub-m11 … sub-m7`, in glob order).

ii.
```python
        subj_name = os.path.basename(nwb_path).split('_')[0]  # e.g., sub-m11
```
```python
        # Track subjects
        if subj_name not in subject_list:
            subject_list.append(subj_name)
        subj_idx = subject_list.index(subj_name)
        ...
        all_subject_idx.append(subj_idx)
```

iii. CONVERSION_NOTES.md Step 2: "11 subjects: m3, m4, m7, m11, m12, m13, m14, m15, m17, m18, m19", cross-checked in Step 3/Step 9 against the paper's "n = 11 mice". The file-name prefix is identical to the containing directory name, so this reproduces the directory-based split.

## 1-c. How are the data split into sessions?

i. One NWB file = one session. Sessions are processed in sorted file order (which is `ses-01 … ses-14` within each subject). The session identity is carried in `session_info['sess_id']` (read from `general/session_id`) and in the scene string from `identifier`. A session is dropped only if it yields fewer than 2 usable trials (this never happens; all 152 sessions are kept). No cross-session cell registration is attempted.

ii.
```python
    for file_idx, nwb_path in enumerate(files_to_process):
        print(f"\nProcessing {file_idx + 1}/{len(files_to_process)}: {os.path.basename(nwb_path)}")
        ...
        result = process_session(nwb_path, show_processing=show, session_idx=file_idx)
        neural_trials, input_trials, output_trials, subj_name, n_cells, session_info = result

        if len(neural_trials) < 2:
            print(f"  SKIPPING: only {len(neural_trials)} trials")
            continue
```

iii. CONVERSION_NOTES.md Step 2 documents the file naming convention `sub-{mouse}_ses-{session}`, and Step 9 reports 152 sessions with 12–14 sessions per subject (m11 has 12 because imaging started on day 3), consistent with the paper's "up to 14 days". The ≥2-trial rule comes from the target-format requirement that "There needs to be at least two trials within each session in order to evaluate the decoder performance."

## 1-d. Are the data correctly split into trials?

i. Trials run from a `trial_start` event to the following `teleport` event. The AI takes *all* sample indices where `trial_start > 0` as starts and *all* sample indices where `teleport > 0` as ends (not rising-edge onsets), pairs them positionally, and slices `[start, stop)`. If the two counts disagree it truncates both lists to the shorter one and prints a warning. Trials with `stop <= start` are skipped. Inter-trial/teleport samples are therefore excluded from every trial.

ii.
```python
    # --- Trial boundaries ---
    trial_starts = np.where(trial_start_signal > 0)[0]
    teleports = np.where(teleport_signal > 0)[0]
    n_trials = len(trial_starts)

    if len(teleports) != n_trials:
        print(f"  WARNING: {nwb_path} has {n_trials} trial starts but {len(teleports)} teleports")
        n_trials = min(n_trials, len(teleports))
        trial_starts = trial_starts[:n_trials]
        teleports = teleports[:n_trials]
```
```python
    for i in range(n_trials):
        start = trial_starts[i]
        stop = teleports[i]

        if stop <= start:
            continue

        n_tp = stop - start
```

iii. CONVERSION_NOTES.md Step 1 records that the reference code uses `sess.trial_start_inds` and `sess.teleport_inds` for trial boundaries, and Step 5 Key Decision 3 states "Trial boundaries: trial_start to teleport signals". Step 10 check 3(c) claims this "matches reference code's trial_start_inds to teleport_inds". The resulting 12,216 trials (mean 80.4/session, range 41–100) were compared to the paper's 12,376.

## 1-e. How are trials filtered based on quality controls?

i. There is no minimum-trial-length filter and no behavioural performance filter. The only trial-level exclusions are structural: a trial with `stop <= start` is dropped, and a session with fewer than two surviving trials is dropped (neither case occurs). One trial-level *curation* step is applied but does not remove the trial: the reference code's lick-sensor-error rule — if more than 35% of a trial's samples have a cumulative lick count > 2, that trial's lick trace is set to NaN and subsequently to 0. This fires on 69 of 12,216 trials. All 12,216 trials are retained in the output.

ii.
```python
    # --- Lick sensor error correction ---
    # Following the reference code: if >35% of samples have cumulative lick count >2, set to NaN
    # (Paper says >30%, code uses 0.35)
    lick_corrected = lick.copy()
    for i in range(n_trials):
        start = trial_starts[i]
        stop = teleports[i]
        trial_licks = lick_corrected[start:stop]
        if len(trial_licks) > 0:
            frac_high = np.sum(trial_licks > 2) / len(trial_licks)
            if frac_high > 0.35:
                lick_corrected[start:stop] = np.nan
```
```python
        if len(neural_trials) < 2:
            print(f"  SKIPPING: only {len(neural_trials)} trials")
            continue
```

iii. CONVERSION_NOTES.md Step 3 "Curation Steps — Trial curation: Lick sensor error correction"; Step 4 records the discrepancy "Lick error threshold | code 0.35 | paper >30% | Used code value 0.35". Step 5 Key Decision 6: "Lick sensor error: Following code (>35% threshold), set erroneous lick data to 0". Step 10 check 5 lists the edge cases considered (multi-plane sessions, length mismatch, very long trials up to 3,359 samples, variable trial counts) and concludes no further trial exclusion is needed.

## 2-a. What variables in the raw data is the final `neural` data derived from?

i. The neural data is the NWB `processing/ophys/Deconvolved` ROI response series, read per plane and concatenated along the ROI axis for the two-plane sessions (m17, m18). The `Fluorescence` (F) and `Neuropil` (Fneu) series present in the same files are never read.

ii.
```python
        # Handle multi-plane sessions (m17, m18 have 2 planes)
        planes = sorted(f['processing/ophys/Deconvolved'].keys())
        deconv_planes = []
        for plane in planes:
            deconv_planes.append(f[f'processing/ophys/Deconvolved/{plane}/data'][()])
        deconv_data = np.concatenate(deconv_planes, axis=1)  # (n_timepoints, total_rois)

        # Verify dimensions match
        assert deconv_data.shape[1] == iscell.shape[0], \
            f"Mismatch: deconv has {deconv_data.shape[1]} ROIs but iscell has {iscell.shape[0]}"
```
```python
    # Filter neural data to only cells
    neural_data = deconv_data[:, cell_indices]  # (n_timepoints, n_cells_final)
    ...
        trial_neural = neural_data[start:stop, :].T.astype(np.float32)
```

iii. Trajectory step 29: "The NWB files already contain deconvolved events, so I don't need to compute dF/F." Step 31 repeats: "The NWB files contain deconvolved events already, so I don't need to compute dF/F and deconvolve." CONVERSION_NOTES.md Step 5 Key Decision list: "Neural data: Use deconvolved events, filtered by iscell", and the metadata field `neural_data_type: 'deconvolved_calcium_events'`. The AI equated the NWB `Deconvolved` array with the paper's `sess.timeseries['events']`.

## 2-b. How is the `neural` data processed?

i. Essentially no processing is applied. The stored `Deconvolved` values are taken as-is: no neuropil subtraction, no maximin baseline, no dF/F normalisation, no Gaussian smoothing, no OASIS deconvolution, no per-cell normalisation. The only transformations are plane concatenation, truncation to the shorter of the neural/behaviour lengths, selection of curated cells, per-trial slicing, transposition to `(n_neurons, n_timepoints)` and a cast to `float32`.

ii.
```python
    # Handle length mismatch between neural and behavioral data
    n_behav = len(position)
    n_neural = deconv_data.shape[0]
    if n_behav != n_neural:
        min_len = min(n_behav, n_neural)
        ...
        deconv_data = deconv_data[:min_len, :]
```
```python
        # --- Neural data ---
        # (n_neurons, n_timepoints)
        trial_neural = neural_data[start:stop, :].T.astype(np.float32)
```

iii. CONVERSION_NOTES.md Step 3 accurately records the paper's pipeline ("dF/F: Baseline via maximin with 20 s sliding window, per trial; Deconvolution: OASIS algorithm (Suite2p) on dF/F") but the AI concluded from the presence of the `Deconvolved` field that this work was already done in the released files (trajectory steps 29 and 31). Step 10 check 3(a) asserts "Data loading: NWB fields match sess object fields used in reference code", and 3(d) "Using raw imaging frames (~64.48 ms), no additional temporal binning". No comparison of the stored `Deconvolved` magnitudes against a dF/F-derived signal was attempted.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters, applied per session. (1) Suite2p curation: keep ROIs with `iscell[:,0] == 1`. (2) Putative-interneuron exclusion: for each `iscell` ROI, the Pearson correlation between its **deconvolved event trace** and the (negative-clipped) running speed is computed over samples with `trial number >= 0`, and cells with r > 0.5 are dropped. The AI substituted deconvolved events for dF/F because it considered computing dF/F "complex". In the delivered dataset this second filter removes **zero** cells in all 152 sessions (`sum(n_interneurons_excluded) == 0`), so the effective curation is `iscell` alone: 138,678 neurons, 155–2,341 per session.

ii.
```python
    # --- Cell filtering ---
    cell_mask = iscell[:, 0] == 1
    n_cells = int(np.sum(cell_mask))

    # Compute dF/F for interneuron exclusion
    # Paper: Pearson r > 0.5 between dF/F and speed -> interneuron
    # Since computing full dF/F with maximin baseline is complex,
    # we use a simpler approach: correlate deconvolved events with speed
    # This is a reasonable approximation since deconvolved events are derived from dF/F
    speed_valid = speed.copy()
    speed_valid[speed_valid < 0] = 0  # clip negative speeds

    # Only use valid timepoints (during trials)
    valid_mask = trial_number >= 0

    if np.sum(valid_mask) > 100:
        speed_corr = np.zeros(deconv_data.shape[1])
        for c in range(deconv_data.shape[1]):
            if cell_mask[c]:
                valid_neural = deconv_data[valid_mask, c]
                valid_speed = speed_valid[valid_mask]
                both_valid = ~np.isnan(valid_neural) & ~np.isnan(valid_speed)
                if np.sum(both_valid) > 100:
                    r = np.corrcoef(valid_neural[both_valid], valid_speed[both_valid])[0, 1]
                    speed_corr[c] = r if not np.isnan(r) else 0

        # Exclude putative interneurons (r > 0.5 as per paper)
        interneuron_mask = speed_corr > 0.5
        n_interneurons = int(np.sum(cell_mask & interneuron_mask))
        cell_mask = cell_mask & ~interneuron_mask
```

iii. CONVERSION_NOTES.md Step 3: "Neuron curation: Suite2p iscell + interneuron exclusion (speed corr > 0.5)", with the expected effect recorded as "Interneuron excl. ~0.42% — 'excluding 0.42 ± 0.85% of cells'". Step 4 records the threshold discrepancy ("code default 0.3 vs paper 0.5 → used paper value 0.5"). Step 5 Key Decision 2: "applied using deconvolved events correlated with speed". Step 4 also notes "Max neurons/session: data 2341 vs paper 2172 — Minor difference, likely due to iscell threshold"; the AI did not connect this to the interneuron filter having removed nothing.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to trial start, achieved purely by slicing: each trial's neural matrix starts at the `trial_start` sample index and ends just before the `teleport` sample index. No pre-trial window is included (`off_start = 0.0`) and trial length is variable (`off_end = None`). Neural and behavioural samples share a single index axis, so no resampling or lag correction is applied; the 10 sessions whose neural array is one sample longer than the behaviour arrays are truncated to the common length before slicing.

ii.
```python
        start = trial_starts[i]
        stop = teleports[i]
        ...
        trial_neural = neural_data[start:stop, :].T.astype(np.float32)
        ...
        time_from_start = (timestamps[start:stop] - timestamps[start]).astype(np.float32)
```
```python
        'metadata': {
            ...
            'temporal_alignment_event': 'trial_start',
            'off_start': 0.0,
            'off_end': None,  # variable trial length
```

iii. CONVERSION_NOTES.md Step 5 Key Decision 3 and Step 10 check 3(c): "Temporal alignment: trial_start to teleport matches reference code's trial_start_inds to teleport_inds". Step 10 check 2 reports a sanity check that raw NWB deconvolved events match the converted neural data for session 0, trial 5, neuron 3, and that raw timestamps match the converted `time_from_trial_start`.

## 2-e. How is the `neural` data temporally binned/resampled?

i. No rebinning or resampling. The native imaging frame is kept as the time bin. The bin size is computed per session as the median difference of the behaviour timestamps and reported as 64.48 ms (15.51 Hz); the value from the first session is written to `metadata['time_bin_size']` and used for the whole dataset.

ii.
```python
    # --- Compute time bin size ---
    dt = np.median(np.diff(timestamps))
```
```python
    # Get time bin size from first session
    dt_ms = session_infos[0]['dt'] * 1000  # convert to ms
    ...
            'time_bin_size': dt_ms,
            ...
            'imaging_rate_hz': 1000.0 / dt_ms,
```

iii. CONVERSION_NOTES.md Step 2 "Sampling rate ~15.5 Hz (64.48 ms)"; Step 3 lists the paper's "~15.5 Hz" imaging rate; Step 9 consistency table "Imaging rate | ~15.5 Hz | 15.51 Hz | ✓"; Step 10 check 3(d): "Binning: Using raw imaging frames (~64.48 ms), no additional temporal binning."

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. The `position` time series' `timestamps` array (`processing/behavior/BehavioralTimeSeries/position/timestamps`), in seconds.

ii.
```python
        timestamps = behav['position/timestamps'][()]
```
```python
        # --- Time from trial start ---
        time_from_start = (timestamps[start:stop] - timestamps[start]).astype(np.float32)
```

iii. CONVERSION_NOTES.md Step 5 mapping table: "timestamps - trial_start_time → input[0]: time_from_trial_start, float seconds, time-varying". Trajectory step 15: "All behavioral timeseries have 19818 timepoints with consistent timestamps (~64 ms interval = ~15.5 Hz, matching 2-photon imaging)" — i.e. the AI checked that the behavioural streams share one timestamp vector before picking one of them.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The trial's first timestamp is subtracted from the trial's timestamp slice, and the result is cast to `float32`. Nothing else. Values therefore start at 0 for every trial and run up to 216.5 s for the longest trial.

ii.
```python
        time_from_start = (timestamps[start:stop] - timestamps[start]).astype(np.float32)
```
```python
        input_arr = np.array([
            time_from_start,
            ...
        ], dtype=np.float32)  # (4, n_tp)
```

iii. Straightforward implementation of the decoder-input specification "Time from start of trial in seconds (continuous, time-varying)". CONVERSION_NOTES.md Step 10 check 2 lists "Time from trial start: Raw timestamps match converted time values (np.allclose = True)" as a sanity check against the original NWB files.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Both are indexed by the same `[start:stop]` slice of the same sample axis, so alignment is implicit. Before slicing, if the neural array and the behaviour arrays differ in length (10 sessions, off by one sample) every stream including the timestamps is truncated to the common minimum, which keeps the shared indexing valid. The AI did not assert that the individual behavioural streams' timestamp vectors are identical to each other.

ii.
```python
    n_behav = len(position)
    n_neural = deconv_data.shape[0]
    if n_behav != n_neural:
        min_len = min(n_behav, n_neural)
        position = position[:min_len]
        ...
        timestamps = timestamps[:min_len]
        deconv_data = deconv_data[:min_len, :]
```
```python
        trial_neural = neural_data[start:stop, :].T.astype(np.float32)
        time_from_start = (timestamps[start:stop] - timestamps[start]).astype(np.float32)
```

iii. CONVERSION_NOTES.md Step 2: "Some sessions have neural/behavioral length mismatch (off by 1)"; Step 5 Key Decision 5: "Length mismatch: Truncate to minimum of neural/behavioral lengths"; Step 10: "Neural/behavioral length mismatch (off by 1): Fixed by truncating to minimum length".

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. **Not** the `environment` behavioural time series (which is loaded and truncated but then never used). Instead the environment is parsed out of the NWB `identifier` string — e.g. `/data/InVivoDA/GCAMP11/23_02_2023/Env1_C_to_Env2_B` — by `parse_scene()`, and assigned per trial by `get_env_per_trial()`, which switches from the "before" environment to the "after" environment at the hard-coded trial index `CHANGE_TRIAL = 30`.

ii.
```python
    scene = identifier.split('/')[-1]

    # Environment switch: e.g., "Env1_C_to_Env2_B"
    if '_to_Env' in scene:
        parts = scene.split('_to_')
        before_part = parts[0]  # e.g., Env1_C
        after_part = parts[1]   # e.g., Env2_B
        env_before = before_part.split('_')[0]  # Env1
        rz_before_letter = before_part.split('_')[1]  # C
        env_after = after_part.split('_')[0]  # Env2
        rz_after_letter = after_part.split('_')[1]  # B
        return env_before, env_after, rz_before_letter, rz_after_letter, True, True
```
```python
def get_env_per_trial(scene_info, n_trials, change_trial=CHANGE_TRIAL):
    """Get environment type (0=Env1, 1=Env2) for each trial."""
    env_before, env_after, rz_before, rz_after, is_switch, is_env_switch = scene_info
    env_map = {'Env1': 0, 'Env2': 1}
    env_vals = []
    for i in range(n_trials):
        if is_env_switch and i >= change_trial:
            env_vals.append(env_map.get(env_after, 0))
        else:
            env_vals.append(env_map.get(env_before, 0))
    return env_vals
```

iii. Trajectory step 42: "env=0: Env1, env=1: Env2, env=-1: inter-trial/teleport period. The environment switch happens at trial 30 in the Env1_C_to_Env2_B session. This matches the scene name perfectly." CONVERSION_NOTES.md Step 1 records "Environment Encoding: Env1 = 0, Env2 = 1"; Step 5's mapping table nevertheless lists the source as "environment signal", which is not what the code uses.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. `Env1 → 0`, `Env2 → 1` via `env_map`; the per-trial scalar is broadcast across all timepoints of the trial into row 1 of the input array, as a `float32`. Trials before index 30 get the pre-switch environment, trials from index 30 on get the post-switch environment on the 7 environment-switch sessions; on all other sessions the single environment is used throughout.

ii.
```python
        input_arr = np.array([
            time_from_start,
            np.full(n_tp, env_per_trial[i], dtype=np.float32),
            np.full(n_tp, i, dtype=np.float32),  # trial number within session
            np.full(n_tp, prev_outcome[i], dtype=np.float32),
        ], dtype=np.float32)  # (4, n_tp)
```

iii. CONVERSION_NOTES.md Step 5 mapping table: "environment → input[1]: environment_type, 0=Env1, 1=Env2, per-trial, broadcast". The decoder-input spec calls for "Environment type (binary, ENV1 vs ENV2, per trial)", and the target format requires `(d_input, n_timepoints)`, hence the broadcast.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. No raw variable: it is the loop index `i` over the trial boundaries derived from `trial_start`/`teleport`, i.e. the within-session sequential trial index starting at 0. The NWB `trial number` time series is loaded but used only to build the `valid_mask` for the speed-correlation step, never as the trial-number input.

ii.
```python
    for i in range(n_trials):
        start = trial_starts[i]
        stop = teleports[i]
        ...
            np.full(n_tp, i, dtype=np.float32),  # trial number within session
```

iii. CONVERSION_NOTES.md Step 5 mapping table: "trial index → input[2]: trial_number, integer, per-trial, broadcast". The verification log confirms the range `trial_number: [0.0, 99.0]`, consistent with the maximum of 100 trials per session.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. None beyond broadcasting the integer loop index across the trial's timepoints as a `float32`. It is not normalised, not made session-relative in any other way, and not reset at the switch trial.

ii.
```python
            np.full(n_tp, i, dtype=np.float32),  # trial number within session
```

iii. The decoder-input spec asks for "Trial number (continuous, per trial)". CONVERSION_NOTES.md gives no further rationale.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. The `Reward` behavioural time series' `timestamps` (event times in seconds), compared against each trial's start and end timestamps. The `Reward/data` amounts are not read.

ii.
```python
        # Reward events
        reward_timestamps = behav['Reward/timestamps'][()]
```
```python
    # --- Determine reward per trial ---
    is_rewarded = np.zeros(n_trials, dtype=int)
    for i in range(n_trials):
        start = trial_starts[i]
        stop = teleports[i]
        trial_start_time = timestamps[start]
        trial_end_time = timestamps[stop - 1] if stop > start else timestamps[start]
        # Check if any reward event falls within this trial's time window
        reward_in_trial = np.any(
            (reward_timestamps >= trial_start_time) &
            (reward_timestamps <= trial_end_time)
        )
        is_rewarded[i] = int(reward_in_trial)
```

iii. CONVERSION_NOTES.md Step 1 lists "Rewards: `sess.timeseries['rewards']`" as the reference source and Step 5's mapping table maps "previous trial reward → input[3]: previous_trial_outcome, 0=omitted, 1=rewarded". Trajectory step 17 notes "74 reward events out of 80 trials (6 omissions)" for the first session, matching the paper's ~15% omission rate.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. `prev_outcome[i] = is_rewarded[i-1]` for `i >= 1`; for the first trial of each session the value is 0 (treated as "not rewarded"). The per-trial scalar is broadcast across the trial's timepoints. The reward window is closed on both ends and uses the timestamp of the last in-trial sample as the end.

ii.
```python
    # Previous trial outcome (for first trial, use 0 = no previous)
    prev_outcome = np.zeros(n_trials, dtype=int)
    for i in range(1, n_trials):
        prev_outcome[i] = is_rewarded[i - 1]
```
```python
            np.full(n_tp, prev_outcome[i], dtype=np.float32),
```

iii. Directly implements the decoder-input spec "Previous trial outcome (binary, omitted = 0, rewarded = 1, per trial)". CONVERSION_NOTES.md does not separately justify the first-trial convention.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Two things: the `position` behavioural time series, and the per-trial reward-zone identity. The reward-zone identity is **not** taken from the `reward_zone` time series (loaded but unused); it is parsed from the session's scene name in `identifier` (e.g. `Env1_LocationB_to_A`) and switched at the hard-coded trial 30, then mapped to fixed coordinate ranges A = [80, 130], B = [200, 250], C = [320, 370] taken from the reference code's `behavior.py`.

ii.
```python
# Reward zone dictionary (from behavior.py)
REWARD_ZONE_DICT = {
    'A': [80, 130],   # X in code
    'B': [200, 250],  # Y in code
    'C': [320, 370],  # Z in code
}
...
CHANGE_TRIAL = 30   # default switch trial
```
```python
def get_reward_zone_per_trial(scene_info, n_trials, change_trial=CHANGE_TRIAL):
    env_before, env_after, rz_before, rz_after, is_switch, is_env_switch = scene_info
    rz_labels = []
    rz_coords = []
    for i in range(n_trials):
        if is_switch and i >= change_trial:
            label = rz_after
        else:
            label = rz_before
        rz_labels.append(label)
        rz_coords.append(REWARD_ZONE_DICT[label])
    return rz_labels, rz_coords
```
```python
        trial_pos = position[start:stop].astype(np.float32)
        trial_pos = np.clip(trial_pos, TRACK_START, TRACK_LENGTH)
        ...
        rz_start, rz_end = rz_coords[i]
        trial_dist = distance_to_reward_zone(trial_pos, rz_start, rz_end)
```

iii. Trajectory step 23: the AI inspected the NWB `reward_zone` signal (values 0–6) and concluded "the reward_zone field encodes whether the animal is in the reward zone, and the actual zone location comes from the scene identifier". Trajectory step 36 then verified on a `B_to_A` session that "Trials 0-29: reward zone centered around 200-220 cm (Location B) ... Trials 30+: reward zone centered around 80-108 cm (Location A). The change_trial=30 default in the code matches what we see." CONVERSION_NOTES.md Step 3 cites the paper's zone coordinates and the "30 warm-up trials".

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Position is first clipped to [0, 450] cm. Then the signed distance to the nearest point of the reward zone is computed: `position - rz_start` when before the zone (negative), `position - rz_end` when past it (positive), and exactly 0 anywhere inside `[rz_start, rz_end]`. The continuous distance is then discretised (see 7-c). No smoothing or interpolation.

ii.
```python
def distance_to_reward_zone(position, rz_start, rz_end):
    """Compute signed distance from position to nearest point in reward zone.

    Negative = before reward zone, positive = after reward zone, 0 = in reward zone.
    """
    dist = np.zeros_like(position, dtype=float)

    before_mask = position < rz_start
    in_mask = (position >= rz_start) & (position <= rz_end)
    after_mask = position > rz_end

    dist[before_mask] = position[before_mask] - rz_start
    dist[in_mask] = 0.0
    dist[after_mask] = position[after_mask] - rz_end

    return dist
```

iii. CONVERSION_NOTES.md Step 5 mapping table: "position - reward zone → output[0]: distance_to_reward_zone, 7 bins, time-varying". The instruction's bin edges include an explicit "0 cm" category, which only makes sense if the whole 50 cm zone maps to distance 0, i.e. distance to *any* location in the reward zone; the AI's `in_mask` implements exactly that.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Seven categories assigned with explicit boolean masks, transcribing the instruction bins: `< -50 → 0`, `[-50, -10) → 1`, `[-10, 0) → 2`, `== 0 → 3`, `(0, 10] → 4`, `(10, 50] → 5`, `> 50 → 6`. Note the closed upper edges on bins 4 and 5 (a distance of exactly +10 falls in bin 4, exactly +50 in bin 5). The resulting full-dataset distribution is [0.253, 0.102, 0.074, 0.237, 0.021, 0.072, 0.242].

ii.
```python
def discretize_distance(dist):
    """Discretize distance to reward zone.
    Bins:
        0: < -50 cm
        1: -50 to -10 cm
        2: -10 cm to < 0 cm
        3: 0 cm
        4: >0 cm to +10 cm
        5: +10 to +50 cm
        6: > +50 cm
    """
    bins = np.zeros(len(dist), dtype=int)
    bins[dist < -50] = 0
    bins[(dist >= -50) & (dist < -10)] = 1
    bins[(dist >= -10) & (dist < 0)] = 2
    bins[dist == 0] = 3
    bins[(dist > 0) & (dist <= 10)] = 4
    bins[(dist > 10) & (dist <= 50)] = 5
    bins[dist > 50] = 6
    return bins
```

iii. The docstring is a verbatim copy of the Decoder Task specification. The AI's `--show-processing` plots (`processing_session0/1.png`, moved to `/app/cache/`) plot the discretised distance per trial; CONVERSION_NOTES.md Step 7 records "No anomalies observed."

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. It is computed from the same `[start:stop]` position slice used for the neural slice, on the shared sample axis, after the neural/behaviour length equalisation. No lag or offset is introduced.

ii.
```python
        trial_neural = neural_data[start:stop, :].T.astype(np.float32)
        ...
        trial_pos = position[start:stop].astype(np.float32)
        ...
        trial_dist = distance_to_reward_zone(trial_pos, rz_start, rz_end)
        dist_disc = discretize_distance(trial_dist)
```

iii. CONVERSION_NOTES.md Step 10 check 3(c) on temporal alignment, and check 2's sanity checks confirming that raw NWB position maps to the converted discretised bins and that raw timestamps match the converted time axis.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. The `position` behavioural time series (`processing/behavior/BehavioralTimeSeries/position/data`), in cm.

ii.
```python
        position = behav['position/data'][()]
```
```python
        trial_pos = position[start:stop].astype(np.float32)
```

iii. CONVERSION_NOTES.md Step 1 lists the reference source as `sess.vr_data['pos']`; Step 5's mapping table maps "position → output[1]: absolute_position, 5 equal bins (90 cm each), time-varying". Trajectory step 17 notes "Position: 0-450 cm corridor" within trials and −500 during the inter-trial period.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Cast to `float32` and clipped to the track range [0, 450] cm, then discretised. The clip is a no-op for binning purposes (in-trial positions only stray marginally outside the track) but is also applied before the distance-to-reward-zone computation.

ii.
```python
TRACK_LENGTH = 450  # cm
TRACK_START = 0     # cm
...
        trial_pos = position[start:stop].astype(np.float32)
        # Clip position to valid range
        trial_pos = np.clip(trial_pos, TRACK_START, TRACK_LENGTH)
        ...
        pos_disc = discretize_position(trial_pos)
```

iii. CONVERSION_NOTES.md Step 3 records the paper's "Track length 450 cm" and trajectory step 27 "Track is 0-450 cm". The clip is the AI's guard against the out-of-track samples it saw in the raw trace (which belong to the teleport period).

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Five equal 90 cm bins over [0, 450] built with `np.linspace`, applied with `np.digitize` on the interior edges [90, 180, 270, 360] and clipped to [0, 4]. Resulting distribution: [0.211, 0.178, 0.231, 0.227, 0.154].

ii.
```python
def discretize_position(position, n_bins=5):
    """Discretize absolute position into n_bins equal-sized bins.
    Track is 0-450 cm, so bins are 90 cm each.
    """
    bin_edges = np.linspace(TRACK_START, TRACK_LENGTH, n_bins + 1)
    bins = np.digitize(position, bin_edges[1:-1])  # 0 to n_bins-1
    bins = np.clip(bins, 0, n_bins - 1)
    return bins
```

iii. Directly implements the specification "Discretized into 5 equal-sized bins spanning the 450 cm track". `output_values[1]` is documented as `['0-90cm', '90-180cm', '180-270cm', '270-360cm', '360-450cm']`.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same `[start:stop]` slice on the shared sample axis as the neural data; no additional alignment.

ii.
```python
        trial_neural = neural_data[start:stop, :].T.astype(np.float32)
        trial_pos = position[start:stop].astype(np.float32)
```

iii. As in 2-d/7-d: the behaviour and imaging streams share one sample grid at 64.48 ms, verified by the AI's Step 10 sanity checks against the raw NWB files.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. The `lick` behavioural time series (`processing/behavior/BehavioralTimeSeries/lick/data`), which holds a cumulative lick count per trial (observed values 0–6).

ii.
```python
        lick = behav['lick/data'][()]
```

iii. CONVERSION_NOTES.md Step 1: "Licks: `sess.timeseries['licks']`, binary, sensor error correction"; trajectory step 17: "Lick: cumulative count per trial (0-6)".

## 9-b. What processing is involved in computing `output` *Lick*?

i. Three steps. (1) Lick-sensor-error correction from the reference code: within each trial, if more than 35% of samples have a cumulative count > 2, the whole trial's lick trace is set to NaN (69 trials dataset-wide). (2) Binarisation: values > 1 set to 1, NaNs set to 0 — so the flagged trials become all-zero ("no lick") rather than being masked out or dropped. (3) A final `> 0` threshold when writing the output row. Overall lick fraction: 22.0%.

ii.
```python
    lick_corrected = lick.copy()
    for i in range(n_trials):
        start = trial_starts[i]
        stop = teleports[i]
        trial_licks = lick_corrected[start:stop]
        if len(trial_licks) > 0:
            frac_high = np.sum(trial_licks > 2) / len(trial_licks)
            if frac_high > 0.35:
                lick_corrected[start:stop] = np.nan

    # Convert licks to binary (as in reference code)
    lick_binary = lick_corrected.copy()
    lick_binary[lick_binary > 1] = 1
    lick_binary[np.isnan(lick_binary)] = 0  # treat NaN licks as 0
```
```python
        trial_lick = lick_binary[start:stop].astype(np.float32)
        ...
        lick_disc = (trial_lick > 0).astype(int)
```

iii. CONVERSION_NOTES.md Step 1 "Key Processing in get_timeseries_data(): Lick sensor error: >35% samples with cumulative lick >2 → NaN; Licks capped at 1 (binary)"; Step 4 records the paper-vs-code threshold discrepancy and the choice of 0.35; Step 5 Key Decision 6: "Lick sensor error: Following code (>35% threshold), set erroneous lick data to 0." The NaN→0 imputation is justified implicitly by the decoder needing a label at every timepoint.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same `[start:stop]` slice on the shared sample axis; no additional alignment. The lick series carries its own timestamps in the NWB file but these are not checked against the position timestamps.

ii.
```python
        trial_lick = lick_binary[start:stop].astype(np.float32)
```

iii. Same rationale as the other behavioural streams: trajectory step 15 established that all `BehavioralTimeSeries` share one timestamp grid.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. The NWB `identifier` string (scene name), parsed by `parse_scene()` into a pre-switch and post-switch zone letter, with the switch applied at trial 30 — the same mechanism as 7-a. The `reward_zone` behavioural time series is not used.

ii.
```python
        identifier = f['identifier'][()].decode()
        ...
        scene_info = parse_scene(identifier)
    ...
    rz_labels, rz_coords = get_reward_zone_per_trial(scene_info, n_trials)
```
```python
    # Reward zone switch: e.g., "Env1_LocationA_to_B"
    elif '_to_' in scene:
        parts = scene.split('_to_')
        before_part = parts[0]  # e.g., Env1_LocationA
        after_letter = parts[1]  # e.g., B
        env = before_part.split('_')[0]  # Env1
        before_letter = before_part.split('_')[1].replace('Location', '')  # A
        return env, None, before_letter, after_letter, True, False

    # No switch: e.g., "Env1_LocationC"
    else:
        env = scene.split('_')[0]  # Env1
        rz_letter = scene.split('_')[1].replace('Location', '')  # C
        return env, None, rz_letter, None, False, False
```

iii. Trajectory steps 22–23 and 36, as quoted in 7-a: the AI inspected the positions at which the `reward_zone` signal is non-zero, matched them against the A/B/C coordinate ranges, confirmed the scene name predicts them, and confirmed the switch occurs at trial 30. CONVERSION_NOTES.md Step 5 mapping table: "scene name → output[4]: reward_zone_location, A=0, B=1, C=2, per-trial."

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The letter is mapped `A→0, B→1, C→2` and broadcast across all timepoints of the trial into row 4 of the output array. The full-dataset distribution is A 32.9%, B 33.7%, C 33.5% (by timepoint).

ii.
```python
        # Reward zone location: A=0, B=1, C=2
        rz_label_map = {'A': 0, 'B': 1, 'C': 2}
        rz_loc = rz_label_map[rz_labels[i]]
        ...
        output_arr = np.array([
            dist_disc,
            pos_disc,
            speed_disc,
            lick_disc,
            np.full(n_tp, rz_loc, dtype=int),
            np.full(n_tp, reward_out, dtype=int),
        ], dtype=int)  # (6, n_tp)
```

iii. The Decoder Task specifies "Reward zone location, per-trial. 0 = A, 1 = B, 2 = C"; the target format prefers time-varying arrays "if at all possible", hence the broadcast. CONVERSION_NOTES.md Step 9 checks the class balance: "RZ A 34.3% / B 32.8% / C 32.9% vs ~33% expected ✓" (per-trial figures).

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. The `Reward` time series' `timestamps` (the same source as the previous-trial-outcome input); `is_rewarded[i]` is reused for both.

ii.
```python
        reward_timestamps = behav['Reward/timestamps'][()]
```
```python
        reward_in_trial = np.any(
            (reward_timestamps >= trial_start_time) &
            (reward_timestamps <= trial_end_time)
        )
        is_rewarded[i] = int(reward_in_trial)
```

iii. CONVERSION_NOTES.md Step 5 mapping table: "reward events → output[5]: reward_outcome, 0=no, 1=yes, per-trial". Step 9 checks the resulting omission rate (15.3%) against the paper's "~15%".

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. A trial is labelled 1 if any reward timestamp falls in the closed interval between the timestamp of the trial's first sample and the timestamp of its last sample, else 0. The reward event times are compared in seconds directly, without being snapped to a sample index. The per-trial scalar is broadcast across the trial's timepoints.

ii.
```python
    is_rewarded = np.zeros(n_trials, dtype=int)
    for i in range(n_trials):
        start = trial_starts[i]
        stop = teleports[i]
        trial_start_time = timestamps[start]
        trial_end_time = timestamps[stop - 1] if stop > start else timestamps[start]
        reward_in_trial = np.any(
            (reward_timestamps >= trial_start_time) &
            (reward_timestamps <= trial_end_time)
        )
        is_rewarded[i] = int(reward_in_trial)
```
```python
        reward_out = is_rewarded[i]
        ...
            np.full(n_tp, reward_out, dtype=int),
```

iii. The Decoder Task specifies "Reward outcome, per-trial. 0 = no, 1 = yes". CONVERSION_NOTES.md Step 9's consistency table uses the resulting omission rate as the validation: "Omission rate | ~15% | 15.3% | ✓".

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Five cases are handled:
- **Neural/behaviour length mismatch**: every stream (including the neural array) is truncated to the shorter length. This fires on 10 sessions (all m17/m18), each off by exactly one sample.
- **Trial-start/teleport count mismatch**: a warning is printed and both index lists are truncated to the shorter one (never triggered in this dataset).
- **Degenerate trials** (`stop <= start`) are skipped; sessions left with fewer than 2 trials are dropped (neither occurs).
- **Lick sensor errors**: trials failing the >35% rule have their lick trace set to NaN and then imputed to 0.
- **Out-of-range behaviour values**: position clipped to [0, 450] cm, speed clipped at 0 (raw speed goes slightly negative, min −1.39 cm/s); NaNs in the neural/speed traces are excluded from the speed-correlation step.

No handling exists for missing/NaN neural samples in the exported trials (there are none, because the stored `Deconvolved` array is defined everywhere), and there is no assertion that the several behavioural streams share identical timestamps.

ii.
```python
    if n_behav != n_neural:
        min_len = min(n_behav, n_neural)
        position = position[:min_len]
        ...
        deconv_data = deconv_data[:min_len, :]
```
```python
    if len(teleports) != n_trials:
        print(f"  WARNING: {nwb_path} has {n_trials} trial starts but {len(teleports)} teleports")
        n_trials = min(n_trials, len(teleports))
        trial_starts = trial_starts[:n_trials]
        teleports = teleports[:n_trials]
```
```python
        if stop <= start:
            continue
```
```python
    speed_valid = speed.copy()
    speed_valid[speed_valid < 0] = 0  # clip negative speeds
    ...
        trial_pos = np.clip(trial_pos, TRACK_START, TRACK_LENGTH)
        trial_speed = np.clip(trial_speed, 0, None)  # clip negative speeds
```

iii. CONVERSION_NOTES.md Step 5 Key Decisions 4–6 and Step 10 "Issues Found and Resolved": "Multi-plane sessions caused IndexError: Fixed by loading all planes and concatenating"; "Neural/behavioral length mismatch (off by 1): Fixed by truncating to minimum length"; "Lick sensor errors: Handled by setting erroneous lick data to 0"; "Very long trials (max 3359 timepoints = 216 s): Valid data, mouse was slow/stopped".

## 13-a. What are the most time-consuming steps of the code?

i. The AI instrumented the code with per-session load and total timers and reported the results. The dominant costs are (1) reading the NWB/HDF5 arrays into memory — `load` is roughly a third to a half of each session's time and scales with neuron count — and (2) the per-cell speed-correlation loop, which runs `np.corrcoef` over ~20–30k samples once per ROI and therefore dominates the post-load time on the large sessions (up to 2,341 cells). Pickling the 9.4 GB result is the other large cost. Total: 199.6 s for 152 sessions (~1.3 s/session, 0.2 s on the smallest, 2.5 s on the largest). The design decision that saves the most time is reading the HDF5 file directly with `h5py` and making only one pass over each file.

ii.
```python
def process_session(nwb_path, show_processing=False, session_idx=0):
    t0 = time.time()
    ...
    t_load = time.time() - t0
    ...
    t_process = time.time() - t0
    ...
    print(f"  {subj_name} ses-{sess_id} ({scene}): {n_cells_final} cells, {len(neural_trials)} trials, "
          f"load={t_load:.1f}s, total={t_process:.1f}s")
```
```python
    total_time = time.time() - total_start
    print(f"\nTotal processing time: {total_time:.1f}s")
    print(f"Average per session: {total_time / len(files_to_process):.1f}s")
```

iii. CONVERSION_NOTES.md Step 6: "Processing time: ~1.3 s per session, ~200 s total"; Step 7 "Run Time Estimates | Full conversion | ~1.3 s | ~200 s (3.3 min)". The instructions required the full conversion to finish in under 15 minutes, which this comfortably does, so the AI did not pursue further optimisation.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI does not discuss vectorisation anywhere in CONVERSION_NOTES.md. Four Python loops remain:
- The **per-cell speed-correlation loop** — the clearest candidate. It recomputes `valid_speed` and the NaN mask inside the loop for every ROI and calls `np.corrcoef` per cell; a single centred matrix multiply computes all cells' correlations at once (a few ms instead of seconds on a 2,000-cell session).
- The **per-trial reward loop** — replaceable by one `np.searchsorted` of the reward times into the trial boundaries.
- The **per-trial lick-error loop** — replaceable with `np.add.reduceat` over the trial boundaries.
- The **main per-trial loop**, which slices and discretises each trial separately. The discretisations (`discretize_distance`, `discretize_position`, `discretize_speed`, lick binarisation) could all be applied once to the whole session array before slicing, since they are elementwise; only the reward-zone-dependent distance varies per trial, and even that only through two scalars.

ii.
```python
        speed_corr = np.zeros(deconv_data.shape[1])
        for c in range(deconv_data.shape[1]):
            if cell_mask[c]:
                valid_neural = deconv_data[valid_mask, c]
                valid_speed = speed_valid[valid_mask]
                both_valid = ~np.isnan(valid_neural) & ~np.isnan(valid_speed)
                if np.sum(both_valid) > 100:
                    r = np.corrcoef(valid_neural[both_valid], valid_speed[both_valid])[0, 1]
```
```python
    for i in range(n_trials):
        ...
        reward_in_trial = np.any(
            (reward_timestamps >= trial_start_time) &
            (reward_timestamps <= trial_end_time)
        )
```
```python
    for i in range(n_trials):
        ...
        dist_disc = discretize_distance(trial_dist)
        pos_disc = discretize_position(trial_pos)
        speed_disc = discretize_speed(trial_speed)
```

iii. No justification is given; CONVERSION_NOTES.md Step 6's "Code inefficiencies identified" / "Code speedups added" template fields were replaced with a bullet list that does not mention loops. The implicit rationale is that at ~200 s total the conversion already met the 15-minute budget.

## 13-c. What processing does the code repeat multiple times?

i. Little is repeated at the file level: each NWB file is opened once and every array is read once, with no separate survey pass. Within a session there are three separate sweeps over the trial boundaries (reward determination, lick-error correction, and the main conversion loop) that could be a single sweep, and the speed-correlation loop recomputes `valid_speed = speed_valid[valid_mask]` and its NaN mask once per ROI instead of once per session. `n_cells_final` is recomputed as both `np.sum(cell_mask)` and `len(cell_indices)`.

ii.
```python
    for i in range(n_trials):          # sweep 1: reward per trial
        ...
    for i in range(n_trials):          # sweep 2: lick sensor error
        ...
    for i in range(n_trials):          # sweep 3: build trial arrays
```
```python
            if cell_mask[c]:
                valid_neural = deconv_data[valid_mask, c]
                valid_speed = speed_valid[valid_mask]      # loop-invariant
                both_valid = ~np.isnan(valid_neural) & ~np.isnan(valid_speed)
```

iii. Not discussed in CONVERSION_NOTES.md. The single-pass file design follows the instruction to "Avoid unnecessary file I/O"; the intra-session repetition is small relative to I/O.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Three kinds:
- **Loaded-but-unused variables**: `reward_zone_signal` and `environment` are read from every NWB file, truncated on the length-mismatch path, and then never referenced again (the environment and reward-zone labels both come from the scene name instead). `scene`/`sess_id`/`n_rois` are stored in metadata only.
- **A computation whose result never changes anything**: the per-cell speed-correlation / interneuron filter. Because it correlates *deconvolved events* rather than dF/F with speed, the maximum r observed is ~0.24, well below the 0.5 threshold, and it removes 0 cells in all 152 sessions. It is nonetheless the most expensive non-I/O step in the script.
- **Redundant guards**: `np.clip(trial_pos, 0, 450)` before `discretize_position`, which itself already clips bin indices to [0, 4]; `np.clip(speed, 0, None)` before a discretisation whose lowest bin is `< 2`; the double NaN-masking inside the correlation loop (the arrays contain no NaNs).

Additionally, the whole `session_info` list (including per-session `n_rois`, `n_cells_iscell`, `n_interneurons_excluded`) is embedded in the pickle metadata; this is documentation rather than waste.

ii.
```python
        reward_zone_signal = behav['reward_zone/data'][()]
        environment = behav['environment/data'][()]
        ...
        reward_zone_signal = reward_zone_signal[:min_len]
        environment = environment[:min_len]
        # ... neither is used again
```
```python
        interneuron_mask = speed_corr > 0.5
        n_interneurons = int(np.sum(cell_mask & interneuron_mask))
        cell_mask = cell_mask & ~interneuron_mask   # removes 0 cells in every session
```
```python
        trial_pos = np.clip(trial_pos, TRACK_START, TRACK_LENGTH)
        ...
    bins = np.digitize(position, bin_edges[1:-1])  # 0 to n_bins-1
    bins = np.clip(bins, 0, n_bins - 1)
```

iii. Not identified in CONVERSION_NOTES.md; on the contrary, Step 9/README report the neuron counts as being "after iscell filtering + interneuron exclusion", and Step 3 records the paper's expectation that ~0.42% of cells should be excluded — a check that would have exposed the no-op had it been run.
