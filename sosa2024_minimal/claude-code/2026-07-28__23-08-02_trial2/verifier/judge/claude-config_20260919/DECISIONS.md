# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI walks `/app/data`, taking every sub-directory whose name starts with `sub-` as a subject and every `*.nwb` file inside it as a session (152 files, 11 mice). Files are opened directly with `h5py` (not `pynwb`, although `pynwb` was available) and the arrays are read eagerly into memory: all behaviour time series (`position`, `speed`, `lick`, `trial_start`, `teleport`, `trial number`, `environment`, `reward_zone`, `autoreward`, `Reward` data+timestamps, `position/timestamps`), the ROI table (`iscell`, `planeIdx`), and the full `Deconvolved` and `Fluorescence` matrices for every imaging plane (planes concatenated along the neuron axis), plus `imaging_rate`, `subject_id`, `session_id`. A session is only processed if its subject is in a hard-coded `SUBJECT_MAP` and its day is a key of a hard-coded `SESSIONS_INFO` table (transcribed from the paper repo's `sessions_dict.py`); otherwise it is skipped with a warning. All 152 sessions / 12,216 trials in the release pass this gate, identical to the reference.

ii.
```python
def load_nwb_session(filepath):
    """Load data from a single NWB file."""
    with h5py.File(filepath, 'r') as f:
        bts = f['processing/behavior/BehavioralTimeSeries']
        ophys = f['processing/ophys']
        seg = f['processing/ophys/ImageSegmentation/PlaneSegmentation']
        position = bts['position/data'][:]
        ...
        deconv_list = []
        for pk in sorted(deconv_keys):
            deconv_list.append(ophys['Deconvolved'][pk]['data'][:])
        deconvolved = np.concatenate(deconv_list, axis=1)
```
```python
    subjects = sorted([d for d in os.listdir(data_dir) if d.startswith('sub-')])
    ...
        session_files = sorted([f for f in os.listdir(subj_path) if f.endswith('.nwb')])
        for sess_file in session_files:
            ses_part = sess_file.split('_')[1]  # 'ses-03'
            exp_day = int(ses_part.split('-')[1])
            if gcamp_name not in SESSIONS_INFO: ... continue
            if exp_day not in SESSIONS_INFO[gcamp_name]: ... continue
```

iii. From the trajectory (steps 24–27) the AI first dumped the HDF5 tree of one file, confirmed the field names and that every subject directory holds `ses-01 … ses-14` files (m11 starting at ses-03), and then treated "one `.nwb` file = one session". Its notes state the dataset is "11 mice (switch task), each with 12-14 sessions" and that the 3 fixed-condition mice are not in the release. `h5py` was chosen because the whole file is read as plain arrays; the AI never justified preferring it over `pynwb`, which it had checked was installed (step 13).

## 1-b. How are the data split into subjects?

i. One subject per `sub-*` directory; the directory name minus the `sub-` prefix (`m3`, `m11`, …) is used as the subject id in `data['subjects']`, and `subject_idx` records the subject of each session. A parallel `SUBJECT_MAP` maps `sub-mN` to the paper's internal name `GCAMPN`, which is only used to look up the session-scene table.

ii.
```python
SUBJECT_MAP = {'sub-m3': 'GCAMP3', 'sub-m4': 'GCAMP4', ..., 'sub-m19': 'GCAMP19'}
...
        gcamp_name = SUBJECT_MAP[subj_id]
        mouse_name = subj_id.replace('sub-', '')  # e.g., 'm11'
        if mouse_name not in subject_names:
            subject_names.append(mouse_name)
        subj_idx = subject_names.index(mouse_name)
...
        'subjects': subject_names,
        'subject_idx': np.array(all_subject_idx, dtype=int),
```

iii. The AI verified (step 26) that `general/subject/subject_id` inside the file equals the directory-derived name (`m11`), and reasoned from the paper that the 11 released mice are the "switch" cohort (GCAMP2/6/10, the fixed-condition mice, are absent). The `sub-mN → GCAMPN` identification was needed to index the repo's `sessions_dict.py` metadata.

## 1-c. How are the data split into sessions?

i. Each `.nwb` file is one session. The experiment day is parsed from the `ses-NN` field of the filename and used both as the session key and as the lookup into the scene table. Sessions are emitted in sorted filename order within each subject; no cross-session neuron alignment is attempted.

ii.
```python
            ses_part = sess_file.split('_')[1]  # 'ses-03'
            exp_day = int(ses_part.split('-')[1])
            scene = SESSIONS_INFO[gcamp_name][exp_day]
```

iii. Step 27 of the trajectory lists the per-subject file ranges and the AI concluded session number = experiment day (noting m11 starts at day 3, consistent with the paper's statement that imaging for that mouse began on day 3). It needed the day number anyway to know the session's scene (reward-zone / environment schedule).

## 1-d. How are the data split into trials?

i. A trial runs from a `trial_start` impulse to the next `teleport` impulse, i.e. the on-track lap only, excluding the grey teleport/ITI zone. The two event channels are converted to index lists with `np.where(x > 0)` and paired positionally; if the counts differ the lists are truncated to the shorter one. The resulting trial count (12,216) and duration statistics (T mean 216.78, median 197.5, min 96, max 3359 samples) are identical to the reference solution.

ii.
```python
    tstart_idx = np.where(nwb_data['trial_start'] > 0)[0]
    teleport_idx = np.where(nwb_data['teleport'] > 0)[0]

    n_trials = min(len(tstart_idx), len(teleport_idx))
    tstart_idx = tstart_idx[:n_trials]
    teleport_idx = teleport_idx[:n_trials]
    ...
    for i in range(n_trials):
        start = tstart_idx[i]
        end = teleport_idx[i]
```

iii. The AI checked in step 26 that `trial_start` and `teleport` each contain exactly 80 impulses in the inspected session and that `trial number` runs `-1 … 80` (i.e. the `-1` ITI values make the stored trial counter awkward), and its notes state "Each trial spans from trial_start to teleport (the on-track portion only, excluding the teleport zone)", matching the paper's definition of a lap.

## 1-e. How are trials filtered based on quality controls?

i. Almost no trial-level QC. A trial is dropped only if it is degenerate: `end <= start` or fewer than 2 samples. A whole session is dropped if it yields fewer than 2 trials or fewer than 1 neuron (needed by the decoder format). There is no minimum-duration threshold, no exclusion of trials by behaviour, and lick-error trials are kept (their lick trace is zeroed, see 9-b). Empirically nothing was removed: 12,216 trials in, 12,216 out.

ii.
```python
    if n_trials < 2:
        print(f"  Skipping session: only {n_trials} trials")
        return None
    ...
        if end <= start:
            continue
        n_timepoints = end - start
        if n_timepoints < 2:
            continue
    ...
    if len(trial_neural) < 2:
        print(f"  Skipping session: only {len(trial_neural)} valid trials")
        return None
```

iii. The AI gives no explicit justification for the absence of a duration filter; the guards it does add are motivated by the format requirement that "There needs to be at least two trials within each session". Its sanity check compares the resulting trial counts to the paper ("2. Trial counts: mean=80.4 +/- 6.1 (paper: 80.5 +/- 7.4)") and treats the agreement as evidence that no further curation is needed.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The `neural` matrices are the NWB `processing/ophys/Deconvolved` traces, taken as stored, with planes concatenated along the neuron axis. `Fluorescence` is also loaded, but only as the signal for the interneuron-correlation filter; `Neuropil` is never read and no dF/F is computed. The AI's notes assert that the stored `Deconvolved` array "corresponds to the OASIS-deconvolved signal described in the paper".

ii.
```python
        deconv_keys = list(ophys['Deconvolved'].keys())
        fluor_keys = list(ophys['Fluorescence'].keys())
        for pk in sorted(deconv_keys):
            deconv_list.append(ophys['Deconvolved'][pk]['data'][:])
        for fk in sorted(fluor_keys):
            fluor_list.append(ophys['Fluorescence'][fk]['data'][:])
        deconvolved = np.concatenate(deconv_list, axis=1)
        fluorescence = np.concatenate(fluor_list, axis=1)
...
    deconvolved = nwb_data['deconvolved'][:, cell_mask]
    ...
    neural = deconvolved[:, neuron_mask]  # (timepoints, n_neurons)
```

iii. Step 32 reasoning: "Use deconvolved activity ('events') for neural data" — the AI equated the NWB `Deconvolved` field with the paper's `events` timeseries. It had read `preprocessing.py` (step 17), which contains the paper's `dff()` (neuropil subtraction, maximin baseline, Gaussian smoothing, OASIS), but chose not to run it, writing in `identify_interneurons` that "we don't have separate dF/F". For the multi-plane mice it justified pooling planes with the paper's statement that "ROIs were identified separately per plane, but planes were pooled for all analyses".

## 2-b. How is the `neural` data processed?

i. No processing at all beyond ROI selection, plane pooling and a cast to `float32`: the stored deconvolved values are sliced per trial and transposed to (n_neurons, n_timepoints). There is no neuropil subtraction with `neu_coef = 0.7`, no per-trial maximin baseline over a 20 s window, no dF/F normalisation, no 2-sample Gaussian smoothing, and no OASIS deconvolution at `tau = 0.7` and the per-plane frame rate — i.e. none of the Methods' calcium pipeline is reproduced.

ii.
```python
    neural = deconvolved[:, neuron_mask]  # (timepoints, n_neurons)
    ...
        trial_n = neural[start:end, :].T.astype(np.float32)
```

iii. The AI's stated rationale (CONVERSION_NOTES.md) is that the NWB deconvolved trace already "corresponds to the OASIS-deconvolved signal described in the paper", so no further processing was thought necessary. It supported this only with the per-session neuron counts and trial counts matching the paper, not with any comparison of the signal itself to the `dff()` output it had read.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters, in the paper's spirit. (1) ROIs are restricted to Suite2p's manual curation, `iscell[:, 0] == 1`. (2) Putative interneurons are dropped: per-cell Pearson correlation with running speed `> 0.5`, computed over on-track samples only (`position > 0`, which removes the teleport period where position is −500). The correlation is computed on the raw `Fluorescence` trace rather than dF/F. Across the dataset this removed 138 cells, leaving 138,540 neurons (the reference keeps 138,298), and per-session counts of 155–2337 neurons.

ii.
```python
    cell_mask = nwb_data['iscell'][:, 0] == 1
    deconvolved = nwb_data['deconvolved'][:, cell_mask]
    fluorescence = nwb_data['fluorescence'][:, cell_mask]

    valid_mask = nwb_data['position'] > 0  # exclude teleport period (pos = -500)
    is_interneuron = identify_interneurons(
        fluorescence, nwb_data['speed'], valid_mask,
        threshold=SPEED_CORR_THRESHOLD)
    neuron_mask = ~is_interneuron
```
```python
    for c in range(n_neurons):
        neural_ts = neural_data[valid_mask, c]
        speed_ts = speed_data[valid_mask]
        if np.std(neural_ts) == 0 or np.std(speed_ts) == 0:
            continue
        r = np.corrcoef(neural_ts, speed_ts)[0, 1]
        if r > threshold:
            is_interneuron[c] = True
```

iii. Notes: "Only ROIs with `iscell[:, 0] == 1` are included (manual curation)" and "ROIs with Pearson correlation > 0.5 between fluorescence (dF/F) and running speed are excluded as putative interneurons (per Methods: 'Pearson correlation of >0.5 between dF/F timeseries and the animal's running speed')". The `0.5` threshold comes from the paper's `dayData` setting (the library default in `spatial.py` is 0.3). The substitution of raw fluorescence for dF/F is acknowledged in the docstring as a proxy because "we don't have separate dF/F". The resulting neuron counts were checked against the paper's reported 155–2172 range.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment to trial start requires nothing beyond the trial slicing: sample `tstart_idx[i]` is the first column of the trial's neural matrix, so t = 0 is the `trial_start` event. `metadata['temporal_alignment_event'] = 'start of trial (trial_start event)'`, `off_start = 0.0`, `off_end = None` (variable trial length). No pre-trial baseline is included.

ii.
```python
        trial_n = neural[start:end, :].T.astype(np.float32)
...
            'temporal_alignment_event': 'start of trial (trial_start event)',
            'off_start': 0.0,
            'off_end': None,
```

iii. The AI listed "Temporal alignment: Align to trial start" as design decision 4 (step 40) and notes that `off_end` is `None` because trial length varies "due to running speed and teleport jitter".

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning or resampling: data stay on the native acquisition grid, one column per imaging frame. `metadata['time_bin_size']` is hard-coded to `1000 / 15.5078125 = 64.48 ms`, and `imaging_rate_hz` to 15.5078125. That constant is in fact the true per-plane sample period for all 152 sessions (all behaviour timestamp steps are 0.0644836 s, and each plane's neural array has exactly one sample per behaviour sample). Note the tension with 3-b: the per-trial *time* input is built from the per-session `imaging_rate` attribute instead, which is 31.0156 Hz on the 28 two-plane sessions.

ii.
```python
            'time_bin_size': 1000.0 / 15.5078125,  # ~64.5 ms
            ...
            'imaging_rate_hz': 15.5078125,
```

iii. The AI read `imaging_rate = 15.5078125` and `Time step: 0.064484` from a single-plane file (step 26) and adopted that as the global bin; its notes record "Time bin size: ~64.48 ms (1/15.5078125 Hz imaging rate)" and "Imaging rate | 15.5 Hz | ~15.5 Hz" as a match to the paper. Keeping the native resolution avoids discarding information and keeps neural and behaviour on one common index.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Not from any stored timestamp. It is synthesised from the sample index within the trial multiplied by a frame period taken from `acquisition/TwoPhotonSeries/imaging_plane/imaging_rate`. The behaviour `position/timestamps` array *is* loaded, but is used only for the reward-event matching.

ii.
```python
    # Frame time
    frame_time = 1.0 / nwb_data['imaging_rate']
    ...
        time_from_start = (np.arange(n_timepoints) * frame_time).astype(np.float32)
        input_arr[0, :] = time_from_start
```

iii. No explicit justification is given beyond design decision 5 in step 40 ("Time bin: ~64.48 ms (1/15.5 Hz)") and the observation in step 26 that the behaviour time step (0.064484 s) equals `1 / imaging_rate` for that single-plane file. The AI generalised that equality to all sessions.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. `t = arange(n_timepoints) / imaging_rate`, so the first sample of every trial is 0 s. Because the stored `imaging_rate` is the *scanner* rate, this is correct only for the 124 single-plane sessions. For the 28 two-plane sessions (all of m17 and m18) `imaging_rate = 31.015625` while the true per-plane/behaviour sample period is 0.0644836 s, so the emitted time axis runs at half real time (a 20 s lap is labelled 10 s). This also contradicts the file's own `time_bin_size` of 64.48 ms for ~18% of sessions. The data-wide maximum (216.5 s) still matches the reference because the longest trial happens to be in a single-plane session.

ii.
```python
    frame_time = 1.0 / nwb_data['imaging_rate']
    ...
        time_from_start = (np.arange(n_timepoints) * frame_time).astype(np.float32)
```

iii. The AI assumed one global imaging rate (notes: "Imaging rate 15.5 Hz"). It had seen both the Methods sentence that the two-plane mice were "imaged at ~31 Hz interleaved in the scan for a sampling rate of ~15.5 Hz per plane" and the two-plane file structure (step 35), but did not connect the stored `imaging_rate` attribute of those files to the scanner rate and never checked the behaviour timestamps of a two-plane session.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. By construction, sample-for-sample: the time vector is generated from the same `n_timepoints = end - start` as the neural slice, so column k of `input` is column k of `neural`. Before trial splitting, behaviour and neural arrays are truncated to a common length when they differ (10 sessions differ by one sample), which keeps the two streams index-aligned.

ii.
```python
        n_behavior = len(position)
        n_neural = deconvolved.shape[0]
        if n_behavior != n_neural:
            min_len = min(n_behavior, n_neural)
            position = position[:min_len]
            ...
            deconvolved = deconvolved[:min_len]
            fluorescence = fluorescence[:min_len]
...
        n_timepoints = end - start
        time_from_start = (np.arange(n_timepoints) * frame_time).astype(np.float32)
```

iii. The AI found the mismatch empirically when the sample run crashed on m18 ses-01 (steps 43–47: "The neural data has one more timepoint than the behavioral data. Let me fix the code to handle this by truncating to the minimum length"), and its sanity check then verifies that neural, input and output have equal `n_timepoints` for every trial.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Not from the `environment` time series (which is loaded and then never used). Environment is read off the session's scene name in the hard-coded `SESSIONS_INFO` table transcribed from the repo's `sessions_dict.py`: a name containing `Env1` gives 0, `Env2` gives 1, and a name containing both (the day-8 novel-environment switch) gives a pre/post pair.

ii.
```python
def parse_scene_environment(scene):
    if 'Env1' in scene and 'Env2' not in scene:
        return 0, None
    elif 'Env2' in scene and 'Env1' not in scene:
        return 1, None
    elif 'Env1' in scene and 'Env2' in scene:
        parts = scene.split('_to_')
        env_before = 0 if 'Env1' in parts[0] else 1
        env_after = 0 if 'Env1' in parts[1] else 1
        return env_before, env_after
```

iii. Notes: "Determined from scene name: sessions with 'Env1' only: environment = 0; 'Env2' only: environment = 1; day 8 sessions (environment switch): first 30 trials = original env, remaining = new env." The AI had observed the raw `environment` channel takes values {−1, 0} for the Env1 mice and {−1, 1} for m17 (steps 35, 39), i.e. −1 during the ITI, which is presumably why it preferred the metadata table; it never states this explicitly. (Checked against the raw channel here: the scene-derived label agrees with `environment` on all 12,216 trials.)

## 4-b. What processing is involved in computing `input` *Environment type*?

i. A per-trial integer (0/1) tiled across all timepoints of the trial. On day-8 sessions the label switches from the pre- to the post-switch environment at trial index 30 (`min(30, n_trials)`).

ii.
```python
    env_before, env_after = parse_scene_environment(scene)
    env_per_trial = np.full(n_trials, env_before, dtype=int)
    if env_after is not None:
        ct = min(SWITCH_TRIAL, n_trials)
        env_per_trial[ct:] = env_after
    ...
        input_arr[1, :] = env_per_trial[i]
```

iii. The 30-trial switch point comes from the paper ("On day 3, the reward zone was moved after 30 trials" as quoted in the AI's notes) and from the AI's inspection of m11 ses-03, where the rewarded position jumps between trial 29 and trial 30 (step 31). Tiling per-trial values across time was chosen because the format allows `(n_input, n_timepoints)` for all inputs.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Nothing in the file: it is the 0-based loop index of the trial within the session, i.e. derived indirectly from the `trial_start`/`teleport` segmentation. The stored `trial number` channel is loaded but unused.

ii.
```python
    for i in range(n_trials):
        ...
        input_arr[2, :] = float(i)
```

iii. Not explicitly justified. The AI had seen that the stored counter runs from −1 (ITI) to 80 (step 26), so the loop index gives a clean 0-based within-session number consistent with its own trial segmentation.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. None — the index is cast to float and tiled across the trial's timepoints. Note that the index counts trials the loop emitted, and since no trial is ever dropped in practice it is also the raw lap index.

ii.
```python
        input_arr[2, :] = float(i)
```

iii. None given; it is the natural sequential encoding of "trial number" required by the decoder spec (verified range 0–99 across the dataset, matching the reference).

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. From the `Reward` behavioural time series, which is an event list (one 0.004 ml entry per delivery) with its own timestamps. A trial is rewarded if any reward timestamp falls inside the trial's time window (`position/timestamps` at trial start and at teleport); the previous-trial input is that per-trial flag shifted by one.

ii.
```python
    reward_per_trial = np.zeros(n_trials, dtype=int)
    timestamps = nwb_data['timestamps']
    reward_ts = nwb_data['reward_ts']
    for i in range(n_trials):
        start_t = timestamps[tstart_idx[i]]
        end_t = timestamps[teleport_idx[i]]
        if np.any((reward_ts >= start_t) & (reward_ts <= end_t)):
            reward_per_trial[i] = 1
```

iii. Step 37 of the trajectory is an explicit check of this construction: the AI counted rewards per trial, confirmed values are all 0.004 (volume, not a per-sample channel), and found 74/80 rewarded (7.5% omission in that session), then confirmed the whole-dataset omission rate of 15.3% against the paper's "~15%".

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. A one-trial shift of the reward flag, with 0 for the first trial of each session, tiled across the trial's timepoints.

ii.
```python
    # Previous trial outcome (first trial has no previous, set to 0)
    prev_outcome = np.zeros(n_trials, dtype=int)
    prev_outcome[1:] = reward_per_trial[:-1]
    ...
        input_arr[3, :] = float(prev_outcome[i])
```

iii. Notes: "previous_trial_outcome: 0=omitted, 1=rewarded, constant within trial" and "Previous trial outcome for the first trial of each session is set to 0", following the instruction's binary coding.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. From the `position` time series plus a per-trial reward-zone label that is **not** read from the data: the label comes from the scene name in the hard-coded `SESSIONS_INFO` table (pre-switch zone for trials 0–29, post-switch zone from trial 30), and the zone's physical extent from the hard-coded `REWARD_ZONES = {A: (80,130), B: (200,250), C: (320,370)}`. The `reward_zone` channel is loaded but never used in the conversion.

ii.
```python
REWARD_ZONES = {'A': (80, 130), 'B': (200, 250), 'C': (320, 370)}
SWITCH_TRIAL = 30
...
def parse_scene_reward_zones(scene, n_trials, change_trial=SWITCH_TRIAL):
    rz_labels = np.empty(n_trials, dtype='U1')
    if scene.endswith('_LocationA') or scene.endswith('LocationA'):
        rz_labels[:] = 'A'
    ... 
    else:
        zone_before, zone_after = parse_switch_zones(scene)
        ct = min(change_trial, n_trials)
        rz_labels[:ct] = zone_before
        rz_labels[ct:] = zone_after
    return rz_labels
...
        rz_start, rz_end = get_reward_zone_coords(rz_labels[i])
        dist = compute_distance_to_reward_zone(pos, rz_start, rz_end)
```

iii. The AI explored the `reward_zone` channel first (steps 28–31): it found the channel takes values 0–6, is non-zero only while the animal is in the zone, is entirely absent on some trials, and that the positions at which it is non-zero jump from ~200 cm to ~80 cm exactly at trial 30 of the m11 day-3 switch session. Because the channel is missing on omission trials it decided instead to take the zone from the session metadata it had read in `sessions_dict.py` (step 18) combined with the paper's 30-trial switch, which is always defined. (Cross-checked here against the raw channel: the metadata-derived label agrees on all 10,394 trials where `reward_zone` is ever non-zero.)

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance in cm to the nearest edge of the trial's zone: negative before the zone, exactly 0 anywhere inside it, positive after it. Computed on the raw (unclipped) position trace.

ii.
```python
def compute_distance_to_reward_zone(position, rz_start, rz_end):
    dist = np.zeros_like(position, dtype=float)
    before = position < rz_start
    inside = (position >= rz_start) & (position <= rz_end)
    after = position > rz_end
    dist[before] = position[before] - rz_start
    dist[inside] = 0.0
    dist[after] = position[after] - rz_end
    return dist
```

iii. The AI's notes describe the output as "distance_to_reward_zone ... 3: 0 cm (inside zone)", i.e. the instruction's bin 3 is read as "anywhere within the 50 cm zone", which is what dictates the zero-inside convention.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Seven categories assigned with explicit boolean masks: `< -50 → 0`, `[-50, -10) → 1`, `[-10, 0) → 2`, `== 0 → 3`, `(0, 10] → 4`, `(10, 50] → 5`, `> 50 → 6`. The only difference from the reference's `np.digitize` edges is which side the ±50 boundary falls on; the resulting class fractions (0.253 / 0.102 / 0.074 / 0.237 / 0.021 / 0.072 / 0.242) match the reference to three decimals.

ii.
```python
def discretize_distance_to_reward(distance):
    out = np.zeros_like(distance, dtype=int)
    out[distance < -50] = 0
    out[(distance >= -50) & (distance < -10)] = 1
    out[(distance >= -10) & (distance < 0)] = 2
    out[distance == 0] = 3
    out[(distance > 0) & (distance <= 10)] = 4
    out[(distance > 10) & (distance <= 50)] = 5
    out[distance > 50] = 6
    return out
```

iii. The bin edges are copied verbatim from the Decoder Task specification (reproduced in the AI's notes), with bin 3 reserved for being inside the zone.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position is sliced with the same `[start:end]` indices as the neural matrix, so the output is sample-aligned with neural activity by construction; the whole-session arrays were already truncated to a common length.

ii.
```python
        pos = nwb_data['position'][start:end]
        ...
        output_arr = np.zeros((6, n_timepoints), dtype=np.int64)
        output_arr[0, :] = dist_disc
```

iii. The AI treats the NWB behaviour and ophys streams as already resampled onto the same imaging-frame grid (step 32: "the NWB data is already aligned to the imaging frames"), which it confirmed by the equal array lengths (and fixed where they differed by one sample).

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. The `position` behavioural time series (cm along the 450 cm corridor; −500 during the teleport period, which never enters a trial slice).

ii.
```python
        position = bts['position/data'][:]
        ...
        pos = nwb_data['position'][start:end]
```

iii. Step 26 established the range ("Position range: -500.0 to 450.8") and the AI noted the −500 values mark the teleport zone, which is excluded both from trials and from the interneuron correlation mask.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Position is clipped to [0, 450] and then binned by integer division into 90 cm bins, with the bin index clipped to 0–4. Clipping means the handful of samples slightly outside the track (e.g. 450.8, or small negatives) fall in the end bins rather than creating extra classes.

ii.
```python
        pos_clipped = np.clip(pos, 0, TRACK_LENGTH)
        pos_disc = discretize_position(pos_clipped)
...
def discretize_position(position, n_bins=5):
    bin_size = TRACK_LENGTH / n_bins
    out = np.clip(np.floor(position / bin_size).astype(int), 0, n_bins - 1)
    return out
```

iii. Notes: "absolute_position (5 bins, time-varying): 0: 0-90 cm, 1: 90-180, 2: 180-270, 3: 270-360, 4: 360-450", i.e. the instruction's five equal bins over the paper's 450 cm track (`TRACK_LENGTH = 450` is listed among "Constants from the paper and code").

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Five equal 90 cm bins via `floor(position / 90)` with clipping at both ends — equivalent to the reference's open-ended `digitize` edges `[-inf, 90, 180, 270, 360, inf]`. Resulting class fractions 0.211 / 0.178 / 0.231 / 0.227 / 0.154 match the reference (0.2107 / 0.1777 / 0.2310 / 0.2265 / 0.1540).

ii.
```python
    bin_size = TRACK_LENGTH / n_bins
    out = np.clip(np.floor(position / bin_size).astype(int), 0, n_bins - 1)
```

iii. Same as 8-b: the instruction demands "5 equal-sized bins spanning the 450 cm track".

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same `[start:end]` slice as the neural data; no resampling or shifting.

ii.
```python
        pos = nwb_data['position'][start:end]
        output_arr[1, :] = pos_disc
```

iii. As in 7-d, the streams are taken to be on a common imaging-frame index after the length-truncation fix.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. The `lick` behavioural time series (per-frame lick counts, range 0–6 in the inspected session).

ii.
```python
        lick = bts['lick/data'][:]
        ...
    licks_corrected, error_trials = correct_lick_sensor_errors(
        nwb_data['lick'], tstart_idx, teleport_idx, threshold=LICK_ERROR_THRESHOLD)
        ...
        lck = licks_corrected[start:end]
```

iii. Step 26 showed "Lick range: 0 to 6", i.e. counts rather than a binary flag, which motivated both the >0 binarisation and the sensor-error check.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Two steps. (1) The paper's lick-sensor-error detection: for each trial, if more than 30% of samples have a lick count > 2, the whole trial's lick trace is set to NaN. 81 trials were flagged across the dataset — exactly the "n = 81 out of 12,376 trials" the Methods report. (2) The NaNs are then replaced by **0** rather than being propagated or excluded, and the remaining counts are binarised at > 0. Net effect: 81 trials are labelled "no lick" everywhere. Dataset lick fraction is 0.219 versus 0.230 for the reference, which applies no sensor correction.

ii.
```python
def correct_lick_sensor_errors(lick_data, tstart_indices, teleport_indices, threshold=LICK_ERROR_THRESHOLD):
    """Trials where >threshold fraction of samples have cumulative lick count >2 are set to NaN."""
    licks = np.copy(lick_data)
    for i, (start, end) in enumerate(zip(tstart_indices, teleport_indices)):
        trial_licks = licks[start:end]
        frac_high = np.sum(trial_licks > 2) / len(trial_licks)
        if frac_high > threshold:
            licks[start:end] = np.nan
            error_trials.append(i)
    return licks, error_trials
```
```python
        lck = np.nan_to_num(lck, nan=0.0)
        lck_binary = (lck > 0).astype(int)
        output_arr[3, :] = lck_binary
```

iii. Notes: "Per the paper: trials where >30% of imaging frame samples have cumulative lick count >2 are flagged. Lick data for error trials is set to 0 (not NaN) after correction. Remaining lick counts binarized: >0 = lick, 0 = no lick." The detection rule and the 0.3 threshold are taken from the Methods sentence the AI read in `methods.txt`; the substitution of 0 for NaN is justified only by the format's requirement that outputs be categorical (NaN is not a valid class).

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Sliced with the same trial indices as the neural data; no lag or smoothing is applied.

ii.
```python
        lck = licks_corrected[start:end]
        output_arr[3, :] = lck_binary
```

iii. Same rationale as 7-d/8-d: behaviour and imaging share one sample index in the NWB files.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Same source as 7-a: the hard-coded scene table (`SESSIONS_INFO`, transcribed from `sessions_dict.py`) plus the assumed switch at trial 30. No raw channel is consulted.

ii.
```python
    rz_labels = parse_scene_reward_zones(scene, n_trials)
    ...
        rz_loc = {'A': 0, 'B': 1, 'C': 2}[rz_labels[i]]
        output_arr[4, :] = rz_loc
```

iii. See 7-a: the AI rejected the `reward_zone` channel because it is empty on trials where the animal did not enter/lick in the zone, and preferred the always-defined session metadata.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The scene string is parsed into either a single zone letter (stay sessions) or a before/after pair (switch sessions, including the day-8 environment switch), the letter is mapped A→0, B→1, C→2, and the value is tiled across the trial's timepoints. Class counts are near-balanced (A 4186, B 4010, C 4020 trials; time-fractions 0.329 / 0.337 / 0.335, matching the reference).

ii.
```python
def parse_switch_zones(scene):
    if '_TO_' in scene_upper:
        parts = scene.split('_to_')
        before_part, after_part = parts[0], parts[1]
        for z in ['A', 'B', 'C']:
            if before_part.endswith(z) or f'Location{z}' in before_part or f'_{z}_' in before_part:
                zone_before = z
                break
        ...
        return zone_before, zone_after
    raise ValueError(f"Cannot parse switch zones from scene: {scene}")
```

iii. Notes: "Reward zone locations per trial are determined from the session scene metadata (from `sessions_dict.py`) — Zone A: 80-130 cm, Zone B: 200-250 cm, Zone C: 320-370 cm. On switch days, trials 0-29 use the pre-switch zone and trials 30+ use the post-switch zone (per paper: 'On day 3, the reward zone was moved after 30 trials')." The zone coordinates match the reference's `reward_zone_dict`, and the AI's step-31 inspection of a switch session confirmed the trial-30 boundary in that session.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. The `Reward` event series (`Reward/timestamps`), compared against the behavioural timestamps at trial start and teleport.

ii.
```python
        reward_data = bts['Reward/data'][:]
        reward_ts = bts['Reward/timestamps'][:]
...
        start_t = timestamps[tstart_idx[i]]
        end_t = timestamps[teleport_idx[i]]
        if np.any((reward_ts >= start_t) & (reward_ts <= end_t)):
            reward_per_trial[i] = 1
```

iii. Step 37: the AI verified that `Reward/data` holds only the constant volume 0.004 and that the timestamps are sparse events, so the outcome must be derived from timestamp membership rather than from a per-sample channel.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Per-trial binary "any reward delivered between trial start and teleport", tiled across the trial's timepoints. Rewards falling in the inter-trial teleport interval do not count. Dataset omission rate 15.3% of trials, matching the paper's ~15%.

ii.
```python
        output_arr[5, :] = reward_per_trial[i]
```
```python
    omission_rate = 1.0 - total_rewarded / total_trial_count
    print(f"\n4. Reward omission rate: {omission_rate*100:.1f}% (paper: ~15%)")
```

iii. Notes: "A trial is 'rewarded' if any reward timestamp falls within the trial boundaries." The AI used the resulting omission rate as an explicit sanity check against the paper's stated ~15% random omission.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Four mechanisms:
- **Neural/behaviour length mismatch** (10 sessions differ by one sample): every array is truncated to the shorter length before trial splitting.
- **Unequal trial-start/teleport counts**: both index lists are truncated to the shorter one.
- **Degenerate trials/sessions**: trials with `end <= start` or < 2 samples are skipped; sessions with < 2 trials or no surviving neurons return `None` and are omitted.
- **Any other per-session failure**: `process_session` is wrapped in a bare `try/except` that prints a traceback and continues to the next file, so a broken session is silently dropped from the dataset rather than aborting the run.
- Missing metadata (unknown subject or unknown experiment day) also causes the session to be skipped with a warning.
There are no assertions checking that behaviour and neural timestamps actually agree, and no handling of NaNs in the source traces (only the NaNs the lick correction itself introduces).

ii.
```python
        if n_behavior != n_neural:
            min_len = min(n_behavior, n_neural)
            position = position[:min_len]
            ...
            deconvolved = deconvolved[:min_len]
```
```python
            try:
                nwb_data = load_nwb_session(filepath)
                result = process_session(nwb_data, scene, exp_day)
            except Exception as e:
                print(f"  Error: {e}")
                import traceback
                traceback.print_exc()
                continue
            if result is None:
                continue
```

iii. The truncation was added reactively after the sample run crashed on m18 ses-01 (steps 44–47); the AI checked the file and found "Position shape: (22794,), Deconv plane0: (22795, ...)", i.e. one extra imaging sample, and decided truncating to the minimum was the safe fix. The remaining guards are defensive so that a single bad file cannot abort a multi-hour conversion, and the run log confirms that in the end no session or trial was lost.

## 13-a. What are the most time-consuming steps of the code?

i. (1) NWB I/O: for each of the 152 files the code reads the full `Deconvolved` **and** the full `Fluorescence` matrix (T × all ROIs, up to ~22,000 × 4,857 per plane) into memory — by far the dominant cost. (2) `identify_interneurons`, a Python loop that calls `np.corrcoef` once per curated cell (up to ~4,800 cells/session over ~20,000 samples). (3) Writing the 9.8 GB output pickle (plus, in the sample run, a second 1.3 GB file). The per-trial conversion loop itself is cheap. Unlike the reference, the code makes only one pass over the dataset.

ii. N/A (observational)

iii. The AI did not profile or discuss runtime; it ran the conversion once in sample mode and once in full (steps 43, 54) and reported only the resulting file size.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The clearest one is the per-cell loop in `identify_interneurons`: the speed correlation for all cells is a single vectorised expression (z-score the masked matrix once and take a matrix-vector product), avoiding both the Python loop and the repeated `speed_data[valid_mask]` fancy-indexing inside it. `correct_lick_sensor_errors` likewise re-slices per trial where a single `np.add.reduceat` over trial boundaries would do. The per-trial conversion loop (`discretize_*`, `compute_distance_to_reward_zone`) could be applied once to the whole session and then sliced, but the variable trial lengths make that awkward — the same trade-off the reference makes.

ii.
```python
    for c in range(n_neurons):
        neural_ts = neural_data[valid_mask, c]
        speed_ts = speed_data[valid_mask]
        if np.std(neural_ts) == 0 or np.std(speed_ts) == 0:
            continue
        r = np.corrcoef(neural_ts, speed_ts)[0, 1]
```

iii. Not discussed by the AI. (The reference's `is_putative_interneuron` contains the same per-cell loop.)

## 13-c. What processing does the code repeat multiple times?

i. Little: the conversion is a single pass, one open/read/process per NWB file, with no separate survey stage. The repetitions that do exist are small: the full `Deconvolved` and `Fluorescence` arrays are both read and then both masked by `cell_mask`; `speed_data[valid_mask]` is recomputed inside every iteration of the per-cell correlation loop; and `run_sanity_checks` re-walks the entire in-memory dataset after conversion. During development the sample run also re-processed 22 of the sessions that the full run later processed again.

ii.
```python
    deconvolved = nwb_data['deconvolved'][:, cell_mask]
    fluorescence = nwb_data['fluorescence'][:, cell_mask]
```

iii. Not discussed. The single-pass structure follows from the AI's decision to take reward zone and environment from a static metadata table, so nothing has to be measured from the data before conversion can begin.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several small wastes:
- Eight arrays/fields are read from every file and never used: `trial number`, `environment`, `reward_zone`, `autoreward`, `Reward/data`, `planeIdx`, `session_id`, `subject_id` (the last two are re-derived from the filename). `environment` and `reward_zone` are exactly the variables the reference uses; here they are loaded and then ignored in favour of the hard-coded tables.
- The complete `Fluorescence` matrix for **all** ROIs is loaded and truncated although only the `iscell` subset is used, and only to compute one correlation per cell.
- Dead code in the trial loop: a placeholder `input_arr = np.array([time_from_start[0] if False else 0])` and an `input_per_trial` vector are built and then immediately overwritten by the real `input_arr`.
- Per-trial constant variables (environment, trial number, previous outcome, reward-zone location, reward outcome) are tiled across every timepoint, although the format allows a length-`n` per-trial vector; this inflates the pickle (the reference does the same).
- `run_sanity_checks` and the 1.3 GB `sample_data.pkl` are development artefacts that play no part in the output.

ii.
```python
        input_arr = np.array([
            time_from_start[0] if False else 0,  # placeholder
        ])
        ...
        input_per_trial = np.array([
            float(env_per_trial[i]), float(i), float(prev_outcome[i])], dtype=np.float32)
        # Combine: first row is time-varying, rest are repeated per trial
        input_arr = np.zeros((4, n_timepoints), dtype=np.float32)
```

iii. Not discussed by the AI. Storing the neural data as `float32` was a deliberate size choice and keeps the output at 9.8 GB.
