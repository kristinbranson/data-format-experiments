# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI walks `/app/data`, treating every sub-directory whose name starts with `sub-` as a subject and every `*.nwb` file inside it as one session. The session number is parsed out of the file name (`..._ses-NN_...`). Files are read directly with `h5py` (not `pynwb`), reading whole arrays into memory with `[()]`: the eleven behavioural time series (`position`, `speed`, `lick`, `trial number`, `trial_start`, `teleport`, `environment`, `reward_zone`, `scanning`, `autoreward`, `Reward`), the position timestamps, the per-plane `Deconvolved` neural arrays, `iscell`, and `general/optophysiology/ImagingPlane/imaging_rate`. All 152 NWB files (11 subjects) are processed in one pass; nothing is sub-sampled in `--full` mode. `--sample` mode takes files `[0]` and `[len//2]`.

ii.
```python
    data_dir = 'data'
    all_files = []
    for sub_dir in sorted(os.listdir(data_dir)):
        if not sub_dir.startswith('sub-'):
            continue
        subject_id = sub_dir.replace('sub-m', '')
        sub_path = os.path.join(data_dir, sub_dir)
        for fname in sorted(os.listdir(sub_path)):
            if not fname.endswith('.nwb'):
                continue
            ses_num = int(fname.split('_ses-')[1].split('_')[0])
            all_files.append({'path': os.path.join(sub_path, fname),
                              'subject_id': subject_id, 'session_num': ses_num})
```
```python
    f = h5py.File(nwb_path, 'r')
    behav = f['processing']['behavior']['BehavioralTimeSeries']
    ophys = f['processing']['ophys']
    position = behav['position']['data'][()]
    ...
    planes = sorted(ophys['Deconvolved'].keys())
    deconv_parts = []
    for plane in planes:
        deconv_parts.append(ophys['Deconvolved'][plane]['data'][()])
    deconv = np.concatenate(deconv_parts, axis=1)  # (n_timepoints, total_rois)
    iscell = ophys['ImageSegmentation']['PlaneSegmentation']['iscell'][()]
    imaging_rate = f['general']['optophysiology']['ImagingPlane']['imaging_rate'][()]
    f.close()
```

iii. From CONVERSION_NOTES.md Step 2: "11 subjects (m3, m4, m7, m11-m15, m17-m19); NWB files: `data/sub-mX/sub-mX_ses-YY_behavior+ophys.nwb`; Sessions total 152". The AI cross-checked the file inventory against the paper ("n = 11 mice", 12 sessions for m11 which is missing ses-01/02, 14 for the rest) and against `sessions_dict.py` in the reference repo, and against the total trial count (12,216 from `trial_start` signals vs 12,376 reported in the paper, attributed to the 3 fixed-condition mice that are not in the NWB release). `h5py` was chosen over `pynwb` without explicit comment; the trajectory shows the AI explored the HDF5 hierarchy directly from the start.

## 1-b. How are the data split into subjects?

i. One subject per `sub-m<N>` directory; the subject name is the directory name with `sub-` stripped (`m11`, `m3`, ...). `subjects` is built in first-encountered order over the sorted directory listing, and `subject_idx` records, for each session, the index into that list. Subject id is also used to look up `GCAMP<N>` in the reference `sessions_dict` (the AI verified the `GCAMP_N -> m_N` naming convention).

ii.
```python
        subject_id = sub_dir.replace('sub-m', '')
        ...
        sub_name = f'm{subject_id}'
        if sub_name not in subject_map:
            subject_map[sub_name] = len(subjects)
            subjects.append(sub_name)
        ...
        subject_idx.append(subject_map[sub_name])
```
```python
def get_scene_for_session(subject_id, session_num):
    gcamp_name = f'GCAMP{subject_id}'
```

iii. CONVERSION_NOTES.md Step 1: "GCAMP_N -> m_N naming convention; Session numbers in NWB = exp_day in sessions_dict". Step 9 consistency check: "Subjects: 11 (switch) vs 11 converted ✓". The trajectory (steps 29–31) shows the AI explicitly confirming that all 11 NWB subjects have an entry in the reference `sessions_dict` and that GCAMP2/6/10 (m2/m6/m10) simply have no NWB files.

## 1-c. How are the data split into sessions?

i. One session per NWB file. The session number is parsed from the file name and used directly as `exp_day` when looking up the scene in `sessions_dict`. No cross-session neuron alignment is attempted; each session keeps its own neuron set. Sessions are ordered by `sorted()` over subject directories and then file names, so sessions appear grouped by subject and in ascending day order.

ii.
```python
            ses_num = int(fname.split('_ses-')[1].split('_')[0])
```
```python
    scene = get_scene_for_session(subject_id, session_num)
```
```python
    sessions = all_sessions_dict[gcamp_name]
    for s in sessions:
        if s['exp_day'] == session_num:
            return s['scene']
```

iii. CONVERSION_NOTES.md Step 1: "Session numbers in NWB = exp_day in sessions_dict"; Step 2 reports 152 sessions, 12–14 per subject. The trajectory confirms the AI verified this mapping against the reward positions actually observed in the data (e.g. m3 ses-03 = `Env1_LocationC_to_A`, rewards at ~320 cm then ~80–135 cm).

## 1-d. How are the data split into trials?

i. Trial starts are all indices where the `trial_start` time series is positive; trial ends are the first index where `teleport` is positive *after* that start. Each `trial_start` that has no subsequent teleport (i.e. a truncated final trial) is discarded. The trial slice is `[start, stop)`, so the teleport sample itself is excluded and the inter-trial teleport period is not included. The stored `trial number` time series is loaded but never used for segmentation.

ii.
```python
    trial_start_inds = np.where(trial_start_signal > 0)[0]
    teleport_inds = np.where(teleport_signal > 0)[0]
    matched_starts = []
    matched_teleports = []
    for i in range(len(trial_start_inds)):
        start = trial_start_inds[i]
        future_teleports = teleport_inds[teleport_inds > start]
        if len(future_teleports) > 0:
            matched_starts.append(start)
            matched_teleports.append(future_teleports[0])
    trial_start_inds = np.array(matched_starts)
    teleport_inds = np.array(matched_teleports)
    n_trials = len(trial_start_inds)
```
```python
        start = trial_start_inds[i]
        stop = teleport_inds[i]
        trial_len = stop - start
        trial_neural = neural_all[start:stop, :].T.astype(np.float32)
```

iii. CONVERSION_NOTES.md Step 1 records that the reference `glmUtils.get_timeseries_data` "uses trial_start_inds and teleport_inds" for trial boundaries, and Step 10 Check 3(c): "Temporal alignment: trial_start to teleport matches reference code". In the trajectory (step 67) the AI explicitly checked m4 ses-04 (a session with unusually long trials) and confirmed the `trial_start`/`teleport` pairs line up and that the long trials are genuine. The resulting counts (12,216 trials; 80.4 ± 6.1 per session) were checked against the paper's "80.5 ± 7.4 trials".

## 1-e. How are trials filtered based on quality controls?

i. Very little trial-level rejection:
- A `trial_start` with no following teleport is dropped (truncated last trial).
- Trials shorter than 3 samples are skipped.
- Whole sessions are skipped if `iscell` yields fewer than 5 cells or if fewer than 2 valid trials survive.
- A lick-sensor-error rule is applied *within* a trial rather than as a trial rejection: if more than 35% of samples in a trial have cumulative lick > 2, the trial's lick trace is set to all zeros (the trial is kept, with its other variables intact).
In practice none of these drop any data: all 152 sessions and all 12,216 trials are retained (minimum trial length in the converted data is 96 samples). The lick-zeroing fires on 69 trials (0.56%).

ii.
```python
        if len(future_teleports) > 0:
            matched_starts.append(start)
```
```python
    if n_cells < 5:
        print(f"  Skipping: only {n_cells} cells")
        return None
    ...
    if n_trials < 2:
        print(f"  Skipping: only {n_trials} trials")
        return None
```
```python
        trial_len = stop - start
        if trial_len < 3:
            # Very short trial, skip
            continue
```
```python
        if np.sum(trial_lick > 2) / len(trial_lick) > 0.35:
            trial_lick[:] = 0  # Set to 0 for error trials
        else:
            trial_lick[trial_lick > 1] = 1
```
```python
    if len(neural_trials) < 2:
        print(f"  Skipping: only {len(neural_trials)} valid trials")
        return None
```

iii. CONVERSION_NOTES.md Step 1 lists `correct_lick_sensor_error` (behavior.py) as the reference curation function and Step 4 records the discrepancy "Lick error threshold: code says 0.35, paper says 0.30 → use code value 0.35". Step 3 notes the paper's "Lick error trials ~0.65% (n=81 out of 12,376)". Step 10 Check 5 lists "Short trials (<3 timepoints): Skipped" and "Lick sensor errors: Detected and set to 0". The `<2 trials` guard is motivated by the format requirement that "there needs to be at least two trials within each session".

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The NWB `processing/ophys/Deconvolved/plane<k>` arrays only. For the two-plane animals (m17, m18) the per-plane arrays are concatenated along the ROI axis in `sorted()` plane order, and the `iscell` mask from `ImageSegmentation/PlaneSegmentation` is applied to the concatenation (the AI verified that `planeIdx` is ordered plane0-then-plane1, so the mask lines up). The `Fluorescence` (F) and `Neuropil` (Fneu) arrays present in the same files are never read.

ii.
```python
    planes = sorted(ophys['Deconvolved'].keys())
    deconv_parts = []
    for plane in planes:
        deconv_parts.append(ophys['Deconvolved'][plane]['data'][()])
    deconv = np.concatenate(deconv_parts, axis=1)  # (n_timepoints, total_rois)
    iscell = ophys['ImageSegmentation']['PlaneSegmentation']['iscell'][()]
    ...
    cell_mask = iscell[:, 0] == 1
    neural_all = deconv[:, cell_mask]  # (n_timepoints, n_cells)
```

iii. CONVERSION_NOTES.md Step 5 Key Decision 1: "Use deconvolved events from NWB (matches reference code `sess.timeseries['events']`)", and Step 3 Processing Details: "Neural data: Deconvolved calcium events (already in NWB)". The trajectory shows the AI first read the reference `preprocessing.dff` and correctly summarised the paper's pipeline (step 49: "raw fluorescence → neuropil subtraction → maximin baseline (20 s window, per trial) → (F − baseline)/|baseline| → OASIS deconvolution → events"), then in step 50 inspected the NWB arrays and concluded without further verification: "**Deconvolved**: Already processed deconvolved events. ~71% zeros, values range 0–15054. These are the 'events' used in the reference code." Multi-plane handling is justified in Step 10 Check 5 and trajectory step 58 ("planeIdx ordering matches the concatenation order").

## 2-b. How is the `neural` data processed?

i. No signal processing at all. The stored `Deconvolved` array is masked to `iscell == 1` cells, sliced per trial, transposed to (n_neurons, n_timepoints) and cast to `float32`. No neuropil subtraction, no per-trial maximin baseline, no dF/F normalisation, no Gaussian smoothing, no OASIS deconvolution, and no z-scoring or normalisation of any kind.

ii.
```python
    neural_all = deconv[:, cell_mask]  # (n_timepoints, n_cells)
    ...
        trial_neural = neural_all[start:stop, :].T.astype(np.float32)  # (n_cells, trial_len)
```

iii. CONVERSION_NOTES.md Step 5 Key Decision 1 and Step 10 Check 3(d): "Binning: Native imaging rate (~64.5 ms) used, no additional binning". The AI's premise is that the NWB `Deconvolved` field is already the paper's `events` signal, so re-running `preprocessing.dff` would be redundant. The notes record the paper's pipeline (Step 1 lists `dff | preprocessing.py | PROCESSING | Computes dF/F with maximin baseline + deconvolution`) but do not document any test of whether the stored array is that output.

## 2-c. How is the `neural` data filtered based on quality controls?

i. A single filter: suite2p's `iscell[:, 0] == 1` manual-curation flag. Sessions with fewer than 5 such cells would be dropped (never triggers). The paper's additional exclusion of putative interneurons (cells whose dF/F correlates with running speed at r > 0.5) is *not* applied. Result: 138,678 neurons over 152 sessions, 912.4 ± 448.7 per session, range 155–2341.

ii.
```python
    cell_mask = iscell[:, 0] == 1
    n_cells = cell_mask.sum()
    if n_cells < 5:
        print(f"  Skipping: only {n_cells} cells")
        return None
    neural_all = deconv[:, cell_mask]
```
```python
        brain_region_idx_list.append(np.zeros(result['n_neurons'], dtype=int))  # All CA1
```

iii. CONVERSION_NOTES.md Step 3: "Cell filtering: iscell=1 from Suite2P"; Step 3 also records the paper's "Interneuron exclusion 0.42 ± 0.85%". Step 4 discrepancy table: "Neuron range — data shows 155–2341, paper says 155–2172 → Max slightly higher, likely pre-interneuron exclusion". Trajectory steps 102/104: "The interneuron exclusion (speed correlation > 0.5) was not implemented — this only affects 0.42% of cells, very minor impact." So the omission was recognised and consciously accepted as negligible, but never implemented or quantified on the actual data.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to trial start, which requires nothing beyond the slicing already done: the neural array is indexed with exactly the same sample indices `[start, stop)` derived from the behavioural `trial_start`/`teleport` signals, so sample 0 of every trial is the trial-start frame. Metadata records `temporal_alignment_event = 'trial_start'`, `off_start = 0.0`, `off_end = None`. No pre-trial baseline window is included.

ii.
```python
        start = trial_start_inds[i]
        stop = teleport_inds[i]
        trial_neural = neural_all[start:stop, :].T.astype(np.float32)
        ...
        trial_pos = position[start:stop]
```
```python
            'temporal_alignment_event': 'trial_start',
            'off_start': 0.0,
            'off_end': None,
```

iii. Implicit: the neural and behavioural streams in the NWB file share one sample grid (the AI verified in the trajectory that all behavioural series have the same length and timestamps as the imaging frames, ~15.5 Hz), so a common index range aligns them. CONVERSION_NOTES.md Step 10, sanity check 1: "Loaded m11 ses-03, trial 5, compared first 3 neurons × 5 timepoints between converted data and original NWB. Result: EXACT MATCH (np.allclose = True)."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning or resampling: the data are kept at the native imaging frame rate, one column per stored frame. The declared bin size is taken from the first processed session's `imaging_rate` attribute: `dt_ms = 1000 / imaging_rate = 64.48 ms` (15.5078 Hz), written once into `metadata['time_bin_size']` for the whole dataset. However, the per-session `dt` used to build the *time* input is computed from each file's own `imaging_rate` attribute, and for the two-plane animals (m17, m18) that attribute stores the scanner rate (31.0156 Hz) rather than the per-plane frame rate; for those 28 of 152 sessions the code therefore uses `dt = 32.24 ms` even though their frames are 64.48 ms apart (verified: median `diff(timestamps) = 0.06448 s` in those files).

ii.
```python
    imaging_rate = f['general']['optophysiology']['ImagingPlane']['imaging_rate'][()]
    dt = 1.0 / imaging_rate  # seconds per frame
```
```python
    imaging_rate = session_infos[0]['imaging_rate']
    dt_ms = 1000.0 / imaging_rate
    ...
            'time_bin_size': dt_ms,
```

iii. CONVERSION_NOTES.md Step 5 Key Decision 3: "Time bin: native imaging rate (~64.5 ms)"; Step 2: "Behavioral time series (all same length, ~15.5 Hz)"; Step 10 Check 3(d): "Native imaging rate (~64.5ms) used, no additional binning". Step 7 reports "Time bin 64.48 ms" for the sample. The AI never revisited the fact that its sample set (m11 ses-03 and m17 ses-09) contained a two-plane session with a different `imaging_rate` attribute.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Not from any stored time variable: it is synthesised from the frame index within the trial and the session's `imaging_rate` attribute (`general/optophysiology/ImagingPlane/imaging_rate`). The NWB `timestamps` array *is* loaded (from `position`) but is used only to window reward events, never for the time input.

ii.
```python
    imaging_rate = f['general']['optophysiology']['ImagingPlane']['imaging_rate'][()]
    dt = 1.0 / imaging_rate  # seconds per frame
    ...
        time_from_start = np.arange(trial_len) * dt  # (trial_len,)
```

iii. CONVERSION_NOTES.md Step 5 Variable Mapping: "Time from trial start | input[0] | frame_index * dt | Time-varying". The AI's exploration (trajectory steps 17, 22–24) established that behaviour timestamps are regularly spaced at 0.06448 s and equal to the imaging frame period, so it treated a synthetic uniform grid as equivalent to the stored timestamps. Step 10 sanity check 4: "Input time check: compared time_from_trial_start for m11 ses-03 trial 5. Result: EXACT MATCH" — a single-plane session.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. `np.arange(trial_len) * dt`, i.e. it starts at exactly 0.0 at the trial-start frame and increases linearly at the nominal frame period. No jitter from real timestamps is preserved and no offset is subtracted (none is needed). The value is stored as row 0 of a `(4, trial_len)` float32 input array.

ii.
```python
        time_from_start = np.arange(trial_len) * dt  # (trial_len,)
        ...
        trial_input = np.zeros((4, trial_len), dtype=np.float32)
        trial_input[0, :] = time_from_start
        trial_input[1, :] = float(env_type)
        trial_input[2, :] = float(trial_number)
        trial_input[3, :] = float(prev_out)
```

iii. CONVERSION_NOTES.md Step 5 ("frame_index * dt") and Step 10 sanity check 4. The design rationale (also visible in the code comments) is that only `time_from_start` is genuinely time-varying, but all four inputs are broadcast to `(4, trial_len)` so that every trial has a uniform 2-D input array.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. By construction: the input array is built with `trial_len = stop - start`, the same length as the neural slice, and starts at 0 on the same frame that starts the neural slice. No interpolation or shifting is performed. There is no explicit assertion that the behavioural and neural arrays have the same total length (in 10 of the 152 sessions the neural arrays are one sample longer than the behavioural ones; because indices are derived from behaviour this never overruns).

ii.
```python
        trial_len = stop - start
        trial_neural = neural_all[start:stop, :].T.astype(np.float32)
        time_from_start = np.arange(trial_len) * dt
        trial_input = np.zeros((4, trial_len), dtype=np.float32)
        trial_input[0, :] = time_from_start
```

iii. CONVERSION_NOTES.md Step 2 states that all behavioural series and the imaging frames share one sample grid ("all same length, ~15.5 Hz"), so sharing `start`/`stop` indices is sufficient. Step 10 Check 3(c) "Temporal alignment: trial_start to teleport matches reference code", plus the per-trial spot checks in Step 10.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Not from the NWB `environment` time series (which is loaded into `env_data` but never used). Environment is taken from the scene string in the reference repo's `sessions_dict`, auto-generated into the conversion script: `Env1 → 0`, `Env2 → 1`, looked up by (subject, session/exp_day).

ii.
```python
    env_data = behav['environment']['data'][()]   # loaded, never used
    ...
def get_environment_from_scene(scene, n_trials, change_trial=CHANGE_TRIAL):
    """Get environment type (0=ENV1, 1=ENV2) for each trial."""
    env = np.zeros(n_trials, dtype=int)
    if '_to_Env' in scene:
        parts = scene.split('_to_')
        before_env = 0 if 'Env1' in parts[0] else 1
        after_env = 0 if 'Env1' in parts[1] else 1
        env[:change_trial] = before_env
        env[change_trial:] = after_env
    else:
        env[:] = 0 if 'Env1' in scene else 1
    return env
```

iii. CONVERSION_NOTES.md Step 1: "Environment mapping: Env1=0, Env2=1"; Step 5 Variable Mapping: "Environment (0/1) | input[1] | From scene name | Per trial"; Step 5 Key Decision 5: "Reward zone from sessions_dict scene mapping (imported from reference code)". The AI treats the scene string as the authoritative experimental record (it is what the reference `behavior.get_reward_zones`/`get_trial_types` use) and, after discovering a transcription error, auto-generated the dictionary from the reference module rather than retyping it.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. Parse the scene name. For a day-8 cross-environment switch scene (e.g. `Env1_C_to_Env2_B`) the first `CHANGE_TRIAL = 30` trials get the "before" environment and the remainder the "after" environment; otherwise a single value is broadcast to all trials. The per-trial value is then broadcast across all timepoints of the trial as row 1 of the input array.

ii.
```python
CHANGE_TRIAL = 30   # reward switches after this trial
...
    env_per_trial = get_environment_from_scene(scene, n_trials)
    ...
        env_type = env_per_trial[i]
        trial_input[1, :] = float(env_type)
```

iii. CONVERSION_NOTES.md Step 3: "Switch trial 30 — 'Each switch occurred after 30 trials'" (paper quote). The binary ENV1/ENV2 coding follows the decoder spec and the reference code convention noted in Step 1.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. No raw variable: it is the within-session sequential index of the (trial_start, teleport) pair, i.e. the loop counter. The NWB `trial number` time series is loaded into `trial_num` but never used.

ii.
```python
    for i in range(n_trials):
        ...
        # Trial number (per trial)
        trial_number = i
```

iii. CONVERSION_NOTES.md Step 5 Variable Mapping: "Trial number | input[2] | 0-indexed | Per trial". Using the loop index keeps the trial number consistent with the trial segmentation actually used (and with `CHANGE_TRIAL = 30`, which is defined on the same index). The trajectory (step 67) notes the AI compared the count from `trial_start` signals (12,216) with `max(trial number)+1` (12,217) and stayed with the `trial_start`-derived segmentation.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. None beyond casting to float and broadcasting the constant across every timepoint of the trial. Numbering starts at 0 for each session and is not re-indexed if a trial were skipped (indices come from the enumeration of matched start/teleport pairs, so the values run 0…n_trials−1; the observed range is [0, 99]).

ii.
```python
        trial_number = i
        ...
        trial_input[2, :] = float(trial_number)
```

iii. Same as 5-a; no separate justification is given in CONVERSION_NOTES.md.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. From the `Reward` event series (its `timestamps`) together with the `reward_zone` time series, via the per-trial `isreward` vector: a trial counts as rewarded if at least one reward timestamp falls between the trial's start and teleport timestamps **and** `reward_zone` was active (>0) at some point in the trial. `previous trial outcome` is then `isreward` shifted by one trial.

ii.
```python
    reward_data = behav['Reward']['data'][()]
    reward_ts = behav['Reward']['timestamps'][()]
    timestamps = behav['position']['timestamps'][()]
    rzone_data = behav['reward_zone']['data'][()]
    ...
    isreward = np.zeros(n_trials, dtype=int)
    for i in range(n_trials):
        start = trial_start_inds[i]
        stop = teleport_inds[i]
        trial_rewards = np.sum((reward_ts >= timestamps[start]) & (reward_ts <= timestamps[stop]))
        rzone_active = np.any(rzone_data[start:stop+1] > 0)
        isreward[i] = 1 if (trial_rewards > 0 and rzone_active) else 0
```

iii. CONVERSION_NOTES.md Step 1 lists `get_omission_trials` (rewardAnalysis.py) as the reference function. Trajectory step 52: "Now I understand the omission trial logic: `isreward` is determined by checking if any reward was delivered in the trial AND if rzone > 0; omission trials are unrewarded trials where the reward zone was NOT active." The AI also established (step 43) that `Reward` is an event series with its own timestamps rather than a per-frame signal, which is why timestamp windowing is used instead of index masking.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. A one-trial shift of `isreward` within the session, with 0 assigned to the first trial of each session (no carry-over across sessions). The per-trial value is broadcast across all timepoints as row 3 of the input.

ii.
```python
    prev_outcome = np.zeros(n_trials, dtype=int)
    prev_outcome[1:] = isreward[:-1]  # First trial has no previous, default to 0
    ...
        prev_out = prev_outcome[i]
        trial_input[3, :] = float(prev_out)
```

iii. CONVERSION_NOTES.md Step 5: "Previous trial outcome | input[3] | 0=omission, 1=rewarded | Per trial", matching the decoder spec. Step 10 sanity check 7: "Verified prev_outcome[i] = reward_outcome[i−1]. Result: CORRECT."

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. From the `position` time series plus a per-trial reward-zone interval that is *not* read from the data but looked up from the scene string in the reference `sessions_dict`: zone A = [80, 130], B = [200, 250], C = [320, 370] cm (the reference repo's X/Y/Z dictionary, remapped A/B/C). On switch days the first 30 trials use the "before" zone and the rest the "after" zone. The `reward_zone` time series is used only as a gate on reward outcome, not to locate the zone.

ii.
```python
REWARD_ZONE_DICT = {'X': [80, 130], 'Y': [200, 250], 'Z': [320, 370]}
LOCATION_TO_ZONE = {'A': 'X', 'B': 'Y', 'C': 'Z'}
ZONE_TO_LABEL = {'X': 'A', 'Y': 'B', 'Z': 'C'}
...
def get_reward_zones_from_scene(scene, n_trials, change_trial=CHANGE_TRIAL):
    rz_coords = np.zeros((n_trials, 2))
    rz_labels = np.empty(n_trials, dtype='U1')
    if '_to_' in scene or '_to_Env' in scene:
        parts = scene.split('_to_')
        before_loc = parts[0][-1]
        after_loc = parts[1][-1]
        before_zone = LOCATION_TO_ZONE[before_loc]
        after_zone = LOCATION_TO_ZONE[after_loc]
        rz_coords[:change_trial] = REWARD_ZONE_DICT[before_zone]
        rz_labels[:change_trial] = ZONE_TO_LABEL[before_zone]
        rz_coords[change_trial:] = REWARD_ZONE_DICT[after_zone]
        rz_labels[change_trial:] = ZONE_TO_LABEL[after_zone]
    else:
        loc = scene[-1]
        zone = LOCATION_TO_ZONE[loc]
        rz_coords[:] = REWARD_ZONE_DICT[zone]
        rz_labels[:] = ZONE_TO_LABEL[zone]
    return rz_coords, rz_labels
```

iii. CONVERSION_NOTES.md Step 1: "reward_zone_dict: A/X=[80,130], B/Y=[200,250], C/Z=[320,370]"; Step 3 quotes the paper's zone coordinates and "Each switch occurred after 30 trials"; Step 5 Key Decision 5: "Reward zone from sessions_dict scene mapping (imported from reference code)". Trajectory step 25 records that the NWB `reward_zone` field takes values 0–6 and "is NOT the reward zone label (A/B/C)", which is why the metadata route was chosen. Critically, the AI found and fixed a real bug here: it had hand-copied `sessions_dict` and got GCAMP11 day 3 backwards; it caught this by comparing assigned zones against the positions at which rewards were actually delivered (trajectory steps 78–84), then auto-generated the dictionary from the reference module and re-verified.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance to the nearest edge of the current trial's reward zone: negative before the zone (`position − rz_start`), exactly 0 anywhere inside the zone, positive after it (`position − rz_end`). Computed on the raw (unclipped) position trace for the trial, then discretised.

ii.
```python
def compute_distance_to_reward_zone(position, rz_start, rz_end):
    distance = np.zeros_like(position)
    before = position < rz_start
    distance[before] = position[before] - rz_start
    in_zone = (position >= rz_start) & (position <= rz_end)
    distance[in_zone] = 0
    after = position > rz_end
    distance[after] = position[after] - rz_end
    return distance
...
        rz_start = rz_coords[i, 0]
        rz_end = rz_coords[i, 1]
        dist_to_rz = compute_distance_to_reward_zone(trial_pos, rz_start, rz_end)
        dist_bins = discretize_distance_to_reward(dist_to_rz)
```

iii. CONVERSION_NOTES.md Step 5 Variable Mapping: "Distance to reward zone | output[0] | Signed distance, 7 bins | Time-varying". The convention "0 = anywhere in the zone" comes straight from the decoder spec, which defines bin 3 as "0 cm" and separates "−10 to < 0" from "> 0 to +10".

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Seven categories assigned by explicit boolean masks, exactly reproducing the spec's edges: `< −50 → 0`, `[−50, −10) → 1`, `[−10, 0) → 2`, `== 0 → 3`, `(0, 10] → 4`, `(10, 50] → 5`, `> 50 → 6`.

ii.
```python
def discretize_distance_to_reward(distances):
    bins = np.zeros(len(distances), dtype=int)
    bins[distances < -50] = 0
    bins[(distances >= -50) & (distances < -10)] = 1
    bins[(distances >= -10) & (distances < 0)] = 2
    bins[distances == 0] = 3
    bins[(distances > 0) & (distances <= 10)] = 4
    bins[(distances > 10) & (distances <= 50)] = 5
    bins[distances > 50] = 6
    return bins
```
```python
        'output_values': [
            ['< -50cm', '-50 to -10cm', '-10 to 0cm', '0cm (in zone)', '>0 to +10cm', '+10 to +50cm', '> +50cm'],
```

iii. Directly transcribed from the "Decoder Outputs" section of the instructions (docstring of `discretize_distance_to_reward` repeats the spec verbatim). Resulting distribution (CONVERSION_NOTES.md Step 9 / verification output): {0: 0.253, 1: 0.102, 2: 0.074, 3: 0.237, 4: 0.021, 5: 0.072, 6: 0.242}; the AI used the rise in the "in zone" fraction from 0.204 to 0.237 as confirmation that the reward-zone bug fix was correct.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Same sample indices as the neural slice — `position[start:stop]` — so no additional alignment step. Distance is stored as row 0 of a `(6, trial_len)` int64 output array whose length equals the neural trial length.

ii.
```python
        trial_pos = position[start:stop]
        ...
        trial_output = np.zeros((n_outputs, trial_len), dtype=np.int64)
        trial_output[0, :] = dist_bins
```

iii. CONVERSION_NOTES.md Step 2 (all behavioural series share the imaging sample grid) and Step 10 sanity checks 2–5, which compared converted position/speed/lick/time for m11 ses-03 trial 5 against values recomputed from the raw NWB file ("EXACT MATCH").

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. The `position` behavioural time series (cm along the 450 cm virtual corridor), sliced to the trial.

ii.
```python
    position = behav['position']['data'][()]
    ...
        trial_pos = position[start:stop]
```

iii. CONVERSION_NOTES.md Step 2 lists `position` among the behavioural series; Step 3 records "Track length 450 cm — '450 cm linear track'". No alternative source was considered.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The trial's position trace is clipped to `[0, 450]` and then discretised into 5 equal 90 cm bins. Clipping is the only numerical processing; it exists to keep the handful of samples marginally outside the track (the raw range within trials runs slightly below 0 and slightly above 450) inside the end bins.

ii.
```python
        pos_clipped = np.clip(trial_pos, 0, TRACK_LENGTH)
        pos_bins = discretize_position(pos_clipped)
```
```python
def discretize_position(positions, n_bins=5):
    bin_edges = np.linspace(0, TRACK_LENGTH, n_bins + 1)
    bins = np.digitize(positions, bin_edges[1:])  # 0 to n_bins-1
    bins = np.clip(bins, 0, n_bins - 1)
    return bins
```

iii. CONVERSION_NOTES.md Step 5: "Absolute position | output[1] | 5 equal bins | Time-varying" and Step 3's 450 cm track length from the paper. The double safety (clip the values *and* clip the bin indices) is not separately justified in the notes.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Five equal bins over the 450 cm track using `np.digitize` against the interior edges [90, 180, 270, 360, 450], with the index clipped to [0, 4]: `<90 → 0`, `90–180 → 1`, `180–270 → 2`, `270–360 → 3`, `≥360 → 4`.

ii.
```python
    bin_edges = np.linspace(0, TRACK_LENGTH, n_bins + 1)
    bins = np.digitize(positions, bin_edges[1:])
    bins = np.clip(bins, 0, n_bins - 1)
```
```python
            ['0-90cm', '90-180cm', '180-270cm', '270-360cm', '360-450cm'],
```

iii. Straight from the decoder spec ("Discretized into 5 equal-sized bins spanning the 450 cm track"). Resulting distribution {0: 0.211, 1: 0.178, 2: 0.231, 3: 0.227, 4: 0.154} is reported in Step 9.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same trial sample indices as the neural data; no resampling or shifting.

ii.
```python
        trial_pos = position[start:stop]
        ...
        trial_output[1, :] = pos_bins
```

iii. As in 7-d: one shared sample grid, verified by the Step 10 spot checks ("Position bin check: compared position discretization for m11 ses-03 trial 5. Result: EXACT MATCH").

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. The `lick` behavioural time series (a per-frame lick count that takes integer values 0–6 in these files).

ii.
```python
    lick_raw = behav['lick']['data'][()]
    ...
        trial_lick = lick_raw[start:stop].copy()
```

iii. CONVERSION_NOTES.md Step 1: "Licks clipped to binary (0/1)" (reference code behaviour); Step 2 lists `lick` among the behavioural series.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Two steps. First, the reference repo's lick-sensor-error rule is applied per trial: if more than 35% of the trial's samples have a cumulative lick value > 2, the entire trial's lick trace is set to 0 (rather than the trial being excluded, as the paper does). Otherwise the trace is binarised by clipping values > 1 to 1 (equivalent to `lick > 0` for these integer-valued data). The result is cast to int.

ii.
```python
        trial_lick = lick_raw[start:stop].copy()
        # Lick processing: clip to binary
        # First check for lick sensor error (>35% samples with cumulative lick > 2)
        if np.sum(trial_lick > 2) / len(trial_lick) > 0.35:
            trial_lick[:] = 0  # Set to 0 for error trials
        else:
            trial_lick[trial_lick > 1] = 1
        trial_lick = trial_lick.astype(int)
        ...
        trial_output[3, :] = trial_lick
```

iii. CONVERSION_NOTES.md Step 1 records `correct_lick_sensor_error` from the reference `behavior.py` and its rule; Step 4 explicitly resolves the threshold discrepancy in favour of the code's 0.35 over the paper's 0.30; Step 3 notes the paper's "n=81 out of 12,376" error trials. Step 10 Check 5 lists "Lick sensor errors: Detected and set to 0". The rule fires on 69/12,216 trials (0.56%), close to the paper's 0.65%.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same trial sample indices as the neural data; no shifting, smoothing or event-time matching (the lick series is already per-frame, unlike `Reward`).

ii.
```python
        trial_lick = lick_raw[start:stop].copy()
        ...
        trial_output[3, :] = trial_lick
```

iii. As in 7-d/8-d; Step 10 sanity check 5: "Lick data check: compared lick processing for m11 ses-03 trial 5. Result: EXACT MATCH".

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Same source as 7-a: the scene string from the reference `sessions_dict`, keyed by subject and session, with the switch at trial 30 on switch days. Not derived from any time series in the NWB file.

ii.
```python
    scene = get_scene_for_session(subject_id, session_num)
    rz_coords, rz_labels = get_reward_zones_from_scene(scene, n_trials)
```

iii. See 7-a. The AI validated the assignment against the data by checking where rewards were actually delivered (trajectory steps 78–84 found and fixed the transcription bug; step 84 re-verified that m11 ses-03 = B for trials 0–29 and A for trials 30+). Step 9 reports the resulting balance A = 32.9%, B = 33.7%, C = 33.5%, "well balanced as expected".

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The letter label is mapped to an integer (A = 0, B = 1, C = 2) with an explicit per-trial loop, then broadcast across all timepoints of the trial as row 4 of the output array.

ii.
```python
    rz_label_idx = np.zeros(n_trials, dtype=int)
    for i in range(n_trials):
        if rz_labels[i] == 'A':
            rz_label_idx[i] = 0
        elif rz_labels[i] == 'B':
            rz_label_idx[i] = 1
        elif rz_labels[i] == 'C':
            rz_label_idx[i] = 2
    ...
        rz_loc = rz_label_idx[i]
        trial_output[4, :] = rz_loc  # per-trial, broadcast
```
```python
            ['A', 'B', 'C'],
```

iii. The 0/1/2 coding follows the decoder spec ("Reward zone location, per-trial. 0 = A, 1 = B, 2 = C"). Broadcasting a per-trial constant over time follows the format guidance that outputs should be time-varying where possible.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. The `Reward` event series timestamps, gated by the `reward_zone` time series being active somewhere in the trial (the same `isreward` vector used for 6-a). The `Reward` *data* (reward amounts) is loaded but unused.

ii.
```python
    reward_data = behav['Reward']['data'][()]
    reward_ts = behav['Reward']['timestamps'][()]
    ...
        trial_rewards = np.sum((reward_ts >= timestamps[start]) & (reward_ts <= timestamps[stop]))
        rzone_active = np.any(rzone_data[start:stop+1] > 0)
        isreward[i] = 1 if (trial_rewards > 0 and rzone_active) else 0
```

iii. Trajectory steps 43 and 51–52: the AI determined that `Reward` is event-based with its own timestamps (e.g. 71 events in a 33,913-sample session) and that the reference `rewardAnalysis` code combines reward delivery with `rzone > 0` to distinguish true omissions. CONVERSION_NOTES.md Step 1 lists `get_omission_trials` and `get_omission_inds` as the corresponding reference functions.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Per trial, count the reward timestamps falling in the closed interval between the trial-start and teleport timestamps; the outcome is 1 if that count is non-zero and the reward zone was active, else 0. The per-trial binary value is broadcast across all timepoints as row 5 of the output.

ii.
```python
    isreward = np.zeros(n_trials, dtype=int)
    for i in range(n_trials):
        start = trial_start_inds[i]
        stop = teleport_inds[i]
        trial_rewards = np.sum((reward_ts >= timestamps[start]) & (reward_ts <= timestamps[stop]))
        rzone_active = np.any(rzone_data[start:stop+1] > 0)
        isreward[i] = 1 if (trial_rewards > 0 and rzone_active) else 0
    ...
        rew_out = isreward[i]
        trial_output[5, :] = rew_out  # per-trial, broadcast
```

iii. CONVERSION_NOTES.md Step 9 consistency check: "Reward omission ~15% (paper) vs 15.7% (converted) ✓"; Step 12: the resulting 84.3% reward rate is "consistent" with the paper's ~85%.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Handling is mostly defensive-by-construction rather than explicit:
- Each session is wrapped in a bare `try/except Exception`; on any error the traceback is printed and the session is silently dropped from the dataset (all 152 sessions in fact succeeded).
- Sessions with < 5 cells or < 2 usable trials return `None` and are dropped.
- Trials shorter than 3 samples are skipped; trailing `trial_start` events with no teleport are skipped.
- Multi-plane sessions are handled by concatenating planes (this was an actual crash, found and fixed during sample testing).
- Lick-sensor-error trials are zeroed rather than dropped.
- Neural/behaviour length mismatches are **not** checked. In 10 sessions (m17 ses-04/06, m18 ses-01/05/07/10/11/12/13/14) the neural arrays are one sample longer than the behavioural arrays; because all indices are derived from the behavioural signals this is harmless here, but a mismatch in the opposite direction would silently produce a neural trial shorter than its input/output arrays.
- No NaN/Inf checks are performed on any stream.

ii.
```python
        try:
            result = process_session(nwb_path, subject_id, session_num,
                                     show_processing=args.show_processing)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue
        if result is None:
            continue
```
```python
    if n_cells < 5:
        print(f"  Skipping: only {n_cells} cells")
        return None
    ...
        if trial_len < 3:
            # Very short trial, skip
            continue
```

iii. CONVERSION_NOTES.md Step 10 Check 5 (Edge cases): "Multi-plane animals (m17, m18): correctly handled by concatenating plane0 and plane1 data; m11 starts from session 03: correctly handled; Short trials (<3 timepoints): Skipped; Lick sensor errors: Detected and set to 0". Step 10 "Issues Found and Resolved" documents the two real bugs that were caught (wrong hand-copied `sessions_dict`, and the multi-plane `IndexError`). Nothing in the notes addresses length mismatches or the silent session-skip path.

## 13-a. What are the most time-consuming steps of the code?

i. The whole full conversion takes 72 s for 152 sessions (`conversion_full_out.txt`), with per-session timing printed. The dominant costs are (1) HDF5 I/O — reading whole `Deconvolved` arrays and eleven behavioural arrays into memory with `[()]` for every file, and (2) writing the 9.8 GB output pickle at the end (plus the peak memory of holding the entire dataset before writing). The per-trial Python loop and the discretisation are negligible by comparison. The code is fast mainly because it skips the expensive part of the paper's pipeline (dF/F + OASIS deconvolution) by reusing the stored `Deconvolved` array.

ii.
```python
    t0 = time.time()
    ...
    elapsed = time.time() - t0
    print(f"  Processed {len(neural_trials)} trials, {n_cells} neurons in {elapsed:.1f}s")
```
```python
    total_elapsed = time.time() - total_t0
    print(f"\n\nTotal processing time: {total_elapsed:.1f}s")
    ...
    with open(args.output, 'wb') as f:
        pickle.dump(data, f)
```

iii. CONVERSION_NOTES.md Step 7 Run Time Estimates: "Sample (2 sessions) 3.6 s; Full (152 sessions) ~90 s", i.e. the AI extrapolated linearly from the sample and judged no optimisation necessary (the instructions' threshold was 15 minutes). No bottleneck breakdown beyond the per-session timer is documented.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. Several small per-trial Python loops remain, all cheap relative to I/O:
- the trial-matching loop, which does an `O(n_teleports)` boolean scan per trial (`teleport_inds[teleport_inds > start]`) and could be a single `np.searchsorted`;
- the `isreward` loop, which rescans all reward timestamps per trial and could be vectorised with `np.searchsorted` plus `np.add.reduceat`;
- the `rz_label_idx` letter→integer loop, which is a pure element-wise map over an array of 80 labels;
- the main per-trial loop itself: distance-to-zone, position/speed digitisation and lick binarisation could be computed once for the whole session and then sliced (variable trial length makes the neural slicing loop unavoidable, but the behavioural derivations do not need it).
The AI does not discuss vectorisation anywhere in CONVERSION_NOTES.md.

ii.
```python
    for i in range(len(trial_start_inds)):
        start = trial_start_inds[i]
        future_teleports = teleport_inds[teleport_inds > start]
        if len(future_teleports) > 0:
            matched_starts.append(start)
            matched_teleports.append(future_teleports[0])
```
```python
    for i in range(n_trials):
        ...
        trial_rewards = np.sum((reward_ts >= timestamps[start]) & (reward_ts <= timestamps[stop]))
```
```python
    for i in range(n_trials):
        if rz_labels[i] == 'A':
            rz_label_idx[i] = 0
        elif rz_labels[i] == 'B':
            ...
```

iii. No justification given; the instructions asked for vectorised loops, and the AI's only stated efficiency argument is the 72 s total runtime (CONVERSION_NOTES.md Step 7), which made further optimisation unnecessary in practice.

## 13-c. What processing does the code repeat multiple times?

i. Little is repeated at the file level: each NWB file is opened exactly once and every session is processed in a single pass (there is no separate survey/statistics pass). What is repeated within a session is minor: the set of trials is iterated three times (trial matching, `isreward`, then the main conversion loop), and `rz_labels` is converted to indices in a loop of its own instead of inside the main loop. `time_from_start` is also effectively computed into a throwaway array before being recomputed into the final input array (see 13-d). Across runs, the entire conversion was re-executed twice end-to-end after the `sessions_dict` bug was fixed, and the sample conversion duplicates work already covered by the full run.

ii.
```python
    for i in range(n_trials):      # pass 1: isreward
    ...
    for i in range(n_trials):      # pass 2: rz_label_idx
    ...
    for i in range(n_trials):      # pass 3: main conversion
```

iii. Not discussed in CONVERSION_NOTES.md. The single-pass file structure is implicit in the script's design: everything needed per session (scene, reward zones, environment) comes either from the file itself or from the static `sessions_dict`, so no pre-pass over the dataset is required.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. A few pieces of dead or unused work, all cheap:
- five behavioural arrays are read in full and never used: `trial_num`, `scanning`, `autoreward`, `reward_data` (reward amounts), and `env_data` (the `environment` series — environment is taken from the scene name instead);
- `trial_input` is built once as a 4-element array and then immediately overwritten by the `(4, trial_len)` array, so the first construction (and its `time_from_start[0]` lookup) is discarded;
- `pos_clipped` makes a copy of the position trace, and `discretize_position` then clips the bin indices again, so the clipping is done twice;
- per-trial constants (environment, trial number, previous outcome, reward-zone location, reward outcome) are materialised as full-length rows for every trial, which inflates the pickle; this is deliberate, following the instruction to make outputs time-varying where possible.
- the neural data is stored as dense `float32` even though the deconvolved array is ~72% zeros, producing a 9.8 GB pickle.

ii.
```python
    trial_num = behav['trial number']['data'][()]
    scanning = behav['scanning']['data'][()]
    autoreward = behav['autoreward']['data'][()]
    reward_data = behav['Reward']['data'][()]
    env_data = behav['environment']['data'][()]
```
```python
        trial_input = np.array([time_from_start[0], float(env_type), float(trial_number), float(prev_out)], dtype=np.float32)
        # Actually, time_from_start is time-varying, so we need shape (4, trial_len) or mixed
        trial_input = np.zeros((4, trial_len), dtype=np.float32)
```
```python
        pos_clipped = np.clip(trial_pos, 0, TRACK_LENGTH)
        pos_bins = discretize_position(pos_clipped)   # which clips the indices again
```

iii. Not discussed in CONVERSION_NOTES.md. The trajectory (step 63) shows the AI noticed the 9.8 GB file size and the sparsity of the deconvolved data ("the deconvolved events are very sparse (~70% zeros)") but chose not to change the representation, since the target format specifies dense `(n_neurons, n_timepoints)` matrices.
