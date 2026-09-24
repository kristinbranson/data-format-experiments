# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI enumerates every sub-directory of `/app/data` whose name starts with `sub-`, and within each, every `*.nwb` file (sorted). Each `.nwb` file is one session. Files are opened directly with `h5py` (not `pynwb`) and the needed datasets are read by HDF5 path: the whole `processing/behavior/BehavioralTimeSeries` group (`position`, `speed`, `lick`, `trial number`, `trial_start`, `teleport`, `Reward`) and `processing/ophys` (`Deconvolved`, `Fluorescence`, `Neuropil`, `ImageSegmentation/PlaneSegmentation/iscell`, `planeIdx`), plus the top-level `identifier`. A single pass over the files does everything (there is no separate survey pass). A session is dropped if the scene name parsed from `identifier` matches none of three regexes ("training sessions - skip these"), if it has < 2 trials, if it has 0 good cells, or if < 2 trials survive. In practice none of these branches ever fire: all 11 subjects, all 152 NWB files and all 12,216 trials are loaded (verified against the raw data).

ii.
```python
DATA_DIR = '/app/data'
...
    subjects_dirs = sorted([
        d for d in os.listdir(DATA_DIR)
        if d.startswith('sub-') and os.path.isdir(os.path.join(DATA_DIR, d))
    ])
    for subj_dir in subjects_dirs:
        subject_name = subj_dir.replace('sub-', '')
        subj_path = os.path.join(DATA_DIR, subj_dir)
        nwb_files = sorted([f for f in os.listdir(subj_path) if f.endswith('.nwb')])
        for nwb_file in nwb_files:
            nwb_path = os.path.join(subj_path, nwb_file)
            result = load_and_process_session(nwb_path, subject_name)
```
```python
    with h5py.File(nwb_path, 'r') as f:
        identifier = f['identifier'][()]
        ...
        beh = f['processing']['behavior']['BehavioralTimeSeries']
        position = beh['position']['data'][:]
        speed = beh['speed']['data'][:]
        lick = beh['lick']['data'][:]
        trial_number = beh['trial number']['data'][:]
        trial_start_sig = beh['trial_start']['data'][:]
        teleport_sig = beh['teleport']['data'][:]
        beh_timestamps = beh['position']['timestamps'][:]
        reward_timestamps = beh['Reward']['timestamps'][:]
        ophys = f['processing']['ophys']
```

iii. The AI first ran `find /app/data -name "*.nwb" | wc -l` and confirmed 152 files, and mapped the directory names to the paper's animals ("m3=GCAMP3, ... m19=GCAMP19"), explicitly noting that the fixed-condition control mice (GCAMP2, 6, 10) are absent, "leaving 11 switch-task mice to work with". It used `h5py` rather than `pynwb` because it only needs a handful of arrays by path and did not want the NWB object overhead; it verified group/dataset layout by walking the HDF5 tree before writing the loader.

## 1-b. How are the data split into subjects?

i. One subject per `sub-*` directory; the subject id is the directory name with the `sub-` prefix stripped (`m11`, `m12`, ..., `m7`). Subjects are registered lazily, in the order their first successfully-converted session is encountered, into `data['subjects']`, and each session records an index into that list in `data['subject_idx']`. Result: 11 subjects, with 12 sessions for m11 and 14 for each of the other 10.

ii.
```python
    subject_name = subj_dir.replace('sub-', '')
...
            if subject_name not in subject_to_idx:
                subject_to_idx[subject_name] = len(subjects)
                subjects.append(subject_name)
            all_subject_idx.append(subject_to_idx[subject_name])
```

iii. The AI treated the DANDI-style `sub-<id>` directory layout as authoritative and cross-checked the count (11) against the paper's "n = 11 mice" for the switch task.

## 1-c. How are the data split into sessions?

i. One session per NWB file. Files are processed in sorted filename order, i.e. `ses-01` ... `ses-14`, so session order within a subject is chronological. The session number itself is never parsed out; instead the AI parses the recording's *scene* from the top-level `identifier` (e.g. `/data/InVivoDA/GCAMP11/23_02_2023/Env1_LocationB_to_A` → `Env1_LocationB_to_A`), which is used downstream for environment and reward-zone identity. Session identity is recorded in `metadata['session_info']` as subject + scene + cell/trial counts. No across-day ROI alignment is attempted (each session's neurons are independent).

ii.
```python
def parse_scene_from_identifier(identifier):
    """Parse the scene name from NWB identifier path."""
    scene = identifier.strip('/').split('/')[-1]
    return scene
...
            session_metadata.append({
                'subject': subject_name,
                'scene': result['scene'],
                'n_cells': result['n_cells'],
                'n_trials': len(result['neural']),
            })
```

iii. From the trajectory: "the NWB identifier field itself contains path info revealing the scene directly, like 'Env1_LocationB_to_A' ... so I can extract the scene from there rather than relying solely on sessions_dict". The AI cross-checked the identifier-derived scene sequence for m3, m11 and m17 against `sessions_dict.py` in the paper's code and found them to "match the sessions_dict entries exactly".

## 1-d. How are the data split into trials?

i. A trial starts at each frame where the `trial_start` behavioural signal is > 0.5 and ends at (and includes) the first subsequent frame where `teleport` > 0.5; if no teleport follows, the next trial start or the end of the recording is used. Within those bounds the AI then keeps only the **on-track** frames, defined as `0 <= position <= 455` cm, which strips the teleport/inter-trial samples (position goes to −500 during the jitter period and to −50…0 in the gray tunnel). This yields 12,216 trials, exactly the same number as the reference, with essentially the same lengths (mean T 215.4 vs 216.8 reference; the on-track mask removes only ~0.02% of within-bound frames, and the AI keeps one extra frame at teleport onset relative to the reference's convention).

ii.
```python
def get_trial_boundaries(trial_number, trial_start_signal, teleport_signal):
    start_frames = np.where(trial_start_signal > 0.5)[0]
    teleport_frames = np.where(teleport_signal > 0.5)[0]
    for sf in start_frames:
        tid = int(trial_number[sf])
        if tid < 0:
            continue
        future_teleports = teleport_frames[teleport_frames > sf]
        if len(future_teleports) == 0:
            next_starts = start_frames[start_frames > sf]
            ef = next_starts[0] if len(next_starts) > 0 else len(trial_number)
        else:
            ef = future_teleports[0] + 1  # include the teleport frame
        trial_starts.append(sf); trial_ends.append(ef); trial_ids.append(tid)
```
```python
def extract_on_track_indices(position, start_idx, end_idx):
    trial_pos = position[start_idx:end_idx]
    on_track = (trial_pos >= 0) & (trial_pos <= TRACK_LENGTH + 5)  # small buffer
    indices = np.where(on_track)[0] + start_idx
    return indices
```

iii. The AI worked out the trial structure empirically: "it seems to start at trial_start=1 and run until teleport=1, with the ITI marked by trial_number=-1 and position=-500 ... meaning I should extract timepoints from trial start through track exit, excluding the unreliable teleport segment." It justified dropping the teleport segment on imaging grounds as well: "the paper notes laser power was reduced during that phase in most sessions."

## 1-e. How are trials filtered based on quality controls?

i. Three thresholds, all very permissive: a trial is skipped if it has fewer than 5 on-track timepoints; a session is skipped if it yields fewer than 2 trial boundaries, has 0 surviving cells, or ends up with fewer than 2 valid trials. There is no speed threshold, no engagement/lick-based trial exclusion, and no exclusion of the post-switch "automatic reward" trials. Empirically none of these filters ever removes anything (the shortest trial in the dataset is 96 bins), so all 12,216 trials are kept — the same total as the reference.

ii.
```python
        on_track_idx = extract_on_track_indices(position, ts, te)
        if len(on_track_idx) < 5:  # Skip very short trials
            continue
...
    if n_trials < 2:
        print(f"  Skipping session with < 2 trials: {scene}")
        return None
...
    if len(neural_trials) < 2:
        print(f"  Skipping session with < 2 valid trials: {scene}")
        return None
```

iii. The AI's explicit reasoning was "No speed threshold applied (decoder predicts at all timepoints including stationary)" (module docstring). It investigated the longest trials (up to 3,360 bins) and concluded they were genuine: "The long trials are mostly from m4 and they show the mouse spending most of the time at position bin 0 ... or at very low speed. These are real trials but with a very slow/stopped mouse", and kept them. The `< 2 trials` guards exist to satisfy the format requirement that every session have at least two trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The emitted `neural` array is the NWB `processing/ophys/Deconvolved` `RoiResponseSeries` (per plane), subset to ROIs with `iscell == 1` and then to non-interneurons. `Fluorescence` (F) and `Neuropil` (Fneu) are also read, but **only** to compute a dF/F that is used for the interneuron-detection correlation; that dF/F is thrown away and never contributes to the saved neural data. For the two 2-plane animals (m17, m18) `plane0` and `plane1` are concatenated along the ROI axis (verified correct: both planes have identical frame counts and `planeIdx` is ordered plane0-then-plane1).

ii.
```python
        has_plane1 = 'plane1' in ophys['Deconvolved']
        if has_plane1:
            deconv0 = ophys['Deconvolved']['plane0']['data'][:]
            deconv1 = ophys['Deconvolved']['plane1']['data'][:]
            ...
            deconvolved = np.concatenate([deconv0, deconv1], axis=1)
        else:
            deconvolved = ophys['Deconvolved']['plane0']['data'][:]
...
    deconvolved_cells = deconvolved[:, cell_mask]
    neural_data = deconvolved_cells[:, good_cells]
```

iii. The AI recognised the issue explicitly and then decided against resolving it: "The NWB file has Fluorescence, Deconvolved, and Neuropil fields, but the Deconvolved data is likely Suite2p's default output rather than the paper's custom pipeline, which recomputes dF/F with neuropil subtraction, a maximin baseline, and OASIS deconvolution — so I need to decide whether the NWB's stored deconvolution is suitable or whether I should reprocess from raw fluorescence to match the paper's methodology. Reprocessing from raw fluorescence would be complex and error-prone, so I'll settle on using the deconvolved data directly from the NWB file." The module docstring nevertheless asserts the opposite, describing the saved signal as "deconvolved calcium activity from NWB (OASIS deconvolution of custom dF/F with maximin baseline, neuropil subtraction coef=0.7)".

## 2-b. How is the `neural` data processed?

i. No processing at all is applied to the saved neural signal beyond ROI selection, length-cropping, a `nan_to_num` guard and a cast to `float32`. It is stored in the raw suite2p `spks` scale (values up to ~1.5e4 in this dataset; non-zero during the inter-trial period, i.e. no per-trial baselining). The paper's pipeline (`preprocessing.dff`: per-trial restriction, `F - 0.7*Fneu` with the trial's mean neuropil added back, maximin baseline = Gaussian σ=15 then 300-sample running min then running max, `(F - baseline)/|baseline|`, σ=2 Gaussian smoothing, OASIS deconvolution at τ=0.7 and `rate/n_planes`) *is* implemented in `compute_dff_for_interneuron_detection`, but it stops before deconvolution and its output is used only for the speed correlation. The per-mouse/per-day `keep_teleports` distinction from `teleport_metadata.py` is not implemented.

ii.
```python
def compute_dff_for_interneuron_detection(fluorescence, neuropil, trial_starts, trial_ends):
    """... Used only for interneuron detection (speed correlation)."""
    f_corr = fluorescence - NEUROPIL_COEF * neuropil
    dff = np.full_like(f_corr, np.nan)
    for t_start, t_end in zip(trial_starts, trial_ends):
        trial_f = f_corr[t_start:t_end, :]
        smoothed = gaussian_filter1d(trial_f, sigma=BASELINE_SMOOTH_SIGMA, axis=0)
        baseline = minimum_filter1d(smoothed, size=BASELINE_WINDOW, axis=0)
        baseline = maximum_filter1d(baseline, size=BASELINE_WINDOW, axis=0)
        abs_baseline = np.abs(baseline); abs_baseline[abs_baseline < 1e-6] = 1e-6
        trial_dff = (trial_f - baseline) / abs_baseline
        trial_dff = gaussian_filter1d(trial_dff, sigma=DFF_SMOOTH_SIGMA, axis=0)
        dff[t_start:t_end, :] = trial_dff
    return dff
```
```python
        trial_neural = neural_data[on_track_idx, :].T  # (n_good_cells, n_timepoints)
        if np.any(np.isnan(trial_neural)) or np.any(np.isinf(trial_neural)):
            trial_neural = np.nan_to_num(trial_neural, nan=0.0, posinf=0.0, neginf=0.0)
        neural_trials.append(trial_neural.astype(np.float32))
```

iii. Same justification as 2-a: the AI judged re-deriving the paper's `events` from F/Fneu to be "complex and error-prone" and assumed the stored `Deconvolved` array was an acceptable stand-in. It did implement the paper's maximin dF/F recipe (with the paper's constants: neuropil coefficient 0.7, σ=15 baseline smoothing, 300-frame ≈ 20 s window, σ=2 dF/F smoothing) because the interneuron filter in the Methods is defined on dF/F.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters, matching the paper. (1) Suite2p curation: only ROIs with `iscell[:,0] == 1` are kept. (2) Putative interneurons are removed: for each surviving cell the Pearson correlation between its (recomputed) dF/F and running speed is taken over all within-trial frames, and cells with r > 0.5 are dropped. Result: 138,336 neurons over 152 sessions (mean 910, min 155, max 2,323), i.e. 342 of the 138,678 `iscell` ROIs removed as interneurons — closely comparable to the reference (138,298 kept, 380 removed).

ii.
```python
    cell_mask = iscell == 1
    deconvolved_cells = deconvolved[:, cell_mask]
    fluorescence_cells = fluorescence[:, cell_mask]
    neuropil_cells = neuropil_data[:, cell_mask]
...
    dff = compute_dff_for_interneuron_detection(fluorescence_cells, neuropil_cells,
                                                trial_starts, trial_ends)
    is_interneuron = detect_interneurons(dff, speed, cell_mask)
    good_cells = ~is_interneuron
    neural_data = deconvolved_cells[:, good_cells]
```
```python
def detect_interneurons(dff, speed, iscell_mask, threshold=INTERNEURON_SPEED_CORR_THR):
    valid = ~np.isnan(dff[:, 0]) & ~np.isnan(speed)
    speed_valid = speed[valid]
    for i in range(n_cells):
        cell_dff = dff[valid, i]
        if np.std(cell_dff) < 1e-10 or np.std(speed_valid) < 1e-10:
            continue
        r, _ = pearsonr(cell_dff, speed_valid)
        if r > threshold:
            is_interneuron[i] = True
```

iii. Module docstring: "Cell filtering: Suite2p iscell classification, then exclude putative interneurons (Pearson r > 0.5 between dF/F and running speed, per paper methods)." The r > 0.5 threshold was taken from the paper's `dayData` default (`int_method = 'speed'`, `int_thresh = 0.5`), which the AI read out of the code repository.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is by construction: every trial's arrays begin at the `trial_start` frame, so sample 0 of each neural matrix *is* the alignment event. Neural and behavioural streams share one frame index (the ophys series has no timestamps, only a `starting_time`/`rate`, and the behavioural timestamps are one per imaging frame), so the same `on_track_idx` index vector slices neural, position, speed, lick and time together. `metadata['temporal_alignment_event'] = 'Start of trial (mouse enters track at position 0 cm)'`, `off_start = 0.0`, `off_end = None`. No pre-event window is included and no interpolation/resampling between streams is performed.

ii.
```python
        on_track_idx = extract_on_track_indices(position, ts, te)
        trial_neural = neural_data[on_track_idx, :].T
        trial_times = beh_timestamps[on_track_idx] - beh_timestamps[on_track_idx[0]]
        trial_pos = position[on_track_idx]
        trial_speed = speed[on_track_idx]
        trial_lick = lick[on_track_idx]
```
```python
            'temporal_alignment_event': 'Start of trial (mouse enters track at position 0 cm)',
            'off_start': 0.0,
            'off_end': None,
```

iii. The AI verified from the HDF5 layout that behaviour and ophys have one sample per imaging frame in the same order and the same length (up to a 1-frame difference in 10 sessions, handled by cropping), so a shared integer index is sufficient and no resampling is required.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning or resampling: data are kept at the native per-plane imaging rate. The bin size is measured post hoc as the median first difference of the per-trial time vector over the whole dataset and written to `metadata['time_bin_size']` in ms; it comes out at 64.4836 ms (≈15.5 Hz), identical to the reference's 64.4836 ms. The AI deliberately did not use the `rate` attribute stored on the `RoiResponseSeries` (which is the 31.0156 Hz scanner rate on the 2-plane animals, not the per-plane rate).

ii.
```python
    all_dt = []
    for sess_inputs in data['input']:
        for trial_inp in sess_inputs:
            if trial_inp.ndim == 2 and trial_inp.shape[1] > 1:
                dt = np.diff(trial_inp[0, :])
                all_dt.extend(dt.tolist())
    if all_dt:
        median_dt_ms = np.median(all_dt) * 1000  # convert to ms
        data['metadata']['time_bin_size'] = float(median_dt_ms)
```
```python
- Time bin size: Native imaging frame rate (~15.5 Hz, ~64.5 ms per bin).
```

iii. Module docstring records the decision as "Native imaging frame rate (~15.5 Hz, ~64.5 ms per bin)". Deriving the value empirically from the behavioural timestamps (which are one per imaging frame) sidesteps the scanner-rate-vs-plane-rate trap that the multi-plane sessions create, and confirms the bin size is common to all sessions, as the format requires.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. The `timestamps` of the `position` behavioural time series (checked to be identical, to floating-point tolerance, to the timestamps of every other behavioural series including `trial number`, which is the one the reference used).

ii.
```python
        beh_timestamps = beh['position']['timestamps'][:]
...
        trial_times = beh_timestamps[on_track_idx] - beh_timestamps[on_track_idx[0]]
```

iii. The AI read the behavioural group and found all series carry one timestamp per imaging frame on a common clock, so any of them serves; it took `position`'s because position is loaded first and is the series it relies on most.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The timestamp of the trial's first retained (on-track) frame is subtracted from the trial's timestamps, so every trial starts at exactly 0 s and the vector is in seconds. Nothing else — no rounding to a nominal bin grid, no clipping. Because the on-track mask is applied before the subtraction, t=0 corresponds to the first on-track frame rather than to the `trial_start` frame itself; in practice these coincide. Stored as row 0 of the `(4, n_timepoints)` `float32` input array. Observed range across the dataset: 0 to 216.6 s (reference: 0 to 216.5 s).

ii.
```python
        trial_times = beh_timestamps[on_track_idx] - beh_timestamps[on_track_idx[0]]
...
        inp = np.zeros((4, n_tp), dtype=np.float32)
        inp[0, :] = trial_times                # time from trial start (s)
```

iii. Straightforward: the spec asks for "time from start of trial in seconds", and the trial's own first timestamp is the natural zero.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It is sliced with the identical index vector (`on_track_idx`) used for the neural matrix, so element *k* of the time vector is the timestamp of column *k* of the neural matrix by construction. Before any slicing, every array (neural, all behavioural series, timestamps) is truncated to their common minimum length, which handles the 10 sessions where the ophys series has exactly one frame more than the behavioural series.

ii.
```python
    n_time = min(
        deconvolved.shape[0], fluorescence.shape[0], neuropil_data.shape[0],
        len(position), len(speed), len(lick), len(trial_number),
        len(trial_start_sig), len(teleport_sig), len(beh_timestamps)
    )
    deconvolved = deconvolved[:n_time]
    ...
    beh_timestamps = beh_timestamps[:n_time]
```

iii. The AI hit this as a runtime error on the multi-plane animals ("Shape mismatch between neural data and behavioral data for multi-plane sessions") and resolved it by cropping every stream to the common length, which both fixes the crash and guarantees index-level alignment thereafter.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Not from the `environment` behavioural time series. The environment is parsed out of the session's *scene name*, taken from the top-level NWB `identifier` string, by three regexes: a cross-environment switch (`Env1_B_to_Env2_C`), a within-environment reward switch (`Env1_LocationA_to_B`), or a stable session (`Env1_LocationA`). For cross-environment switch sessions the environment changes from `env_before` to `env_after` at the hard-coded trial index `CHANGE_TRIAL = 30`. `Env1 → 0`, `Env2 → 1`. All 152 scene strings in the dataset match one of the three patterns, so the "training session" fall-through is never used. I verified this construction against the NWB `environment` signal: it agrees on 12,216 / 12,216 trials.

ii.
```python
    env_map = {'Env1': 0, 'Env2': 1}
    m = re.match(r'(Env[12])_([ABC])_to_(Env[12])_([ABC])', scene)
    if m:
        return {'is_switch': True, 'env_before': env_map[m.group(1)],
                'env_after': env_map[m.group(3)],
                'zone_before': m.group(2), 'zone_after': m.group(4)}
    m = re.match(r'(Env[12])_Location([ABC])_to_([ABC])', scene)
    ...
    m = re.match(r'(Env[12])_Location([ABC])', scene)
```
```python
CHANGE_TRIAL = 30  # Trial index where reward zone switches
...
def get_per_trial_info(session_info, n_trials):
    for t in range(n_trials):
        if session_info['is_switch'] and t >= CHANGE_TRIAL:
            zones.append(session_info['zone_after']); envs.append(session_info['env_after'])
        else:
            zones.append(session_info['zone_before']); envs.append(session_info['env_before'])
```

iii. The AI's reasoning: "For the environment signal per trial, stable and within-environment switch sessions keep the same environment across all trials, while cross-environment switch sessions have the first 30 trials in one environment and the rest in the other — something the NWB's environment signal should encode directly." It had noticed the raw `environment` series takes values 0/1 with −1 during the ITI, but still chose the scene-name route so that environment and reward zone come from a single, consistent source. The trial-30 boundary comes from the Methods ("On day 3 (switch one), the zone was moved after 30 trials"). The AI did not verify the scene-derived labels against the `environment` series.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. A dictionary lookup `{'Env1': 0, 'Env2': 1}` and broadcast of the per-trial scalar across all timepoints of the trial, stored as row 1 of the input array (as `float32`). Observed range 0–1, matching the reference.

ii.
```python
        trial_env = envs[i]
...
        inp[1, :] = float(trial_env)           # environment type
```

iii. The spec asks for a binary ENV1/ENV2 indicator per trial; the format requires `(d_input, n_timepoints)`, so the constant is tiled across the trial.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. The `trial number` behavioural time series, sampled at the trial's start frame (`trial_number[sf]`). Trials whose sampled id is negative are skipped (never happens). I checked this against the raw data: for all 152 sessions `trial number` at the `trial_start` frames is exactly `0, 1, 2, ...`, so this is numerically identical to the reference's use of the loop index. Observed range 0–99, same as the reference.

ii.
```python
    for sf in start_frames:
        tid = int(trial_number[sf])
        if tid < 0:
            continue
        ...
        trial_ids.append(tid)
...
        trial_num = trial_ids[i]
```

iii. The AI had inspected the `trial number` series and found it counts laps and drops to −1 during the inter-trial interval, so reading it at the trial-start frame gives the within-session lap index directly. (Note the AI does *not* use `trial number` to define trial boundaries — those come from `trial_start`/`teleport` — so the disagreement the reference author found between the two is avoided.)

## 5-b. What processing is involved in computing `input` *Trial number*?

i. None beyond the cast to float and broadcasting the per-trial scalar across all timepoints; it is the raw within-session index, not normalised, not made cumulative across sessions. Stored as row 2 of the input array.

ii.
```python
        inp[2, :] = float(trial_num)           # trial number
```

iii. The spec lists trial number as a continuous per-trial input; the AI left it as the raw index and let the decoder handle scaling.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. The `Reward` behavioural time series' `timestamps` (the reward event times; its `data` values, the reward amounts, are not read). Each trial is marked rewarded if any reward timestamp falls inside the closed interval spanned by that trial's first and last behavioural timestamps; the previous-trial input is then the preceding trial's flag.

ii.
```python
        reward_timestamps = beh['Reward']['timestamps'][:]
...
    is_rewarded = np.zeros(n_trials, dtype=int)
    for i, (ts, te) in enumerate(zip(trial_starts, trial_ends)):
        t_start_time = beh_timestamps[ts]
        t_end_time = beh_timestamps[min(te - 1, len(beh_timestamps) - 1)]
        if np.any((reward_timestamps >= t_start_time) & (reward_timestamps <= t_end_time)):
            is_rewarded[i] = 1
```

iii. The AI noted that `Reward` is an event series on its own clock rather than a per-frame signal, so it compared event times against the trial's time window instead of trying to index it by frame. It also reasoned that operant rewards, auto-rewards and omissions cannot and need not be distinguished here: "omissions ... and non-rewarded trials ... both look identical — no Reward event — but for the decoder's purposes I only care about the actual outcome."

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. `prev_outcome = is_rewarded[i-1]` for `i > 0`, and 0 for the first trial of the session (previous trial unknown; note it is coded the same as "omitted"). Indexing is over the full pre-filter trial list, so a skipped trial would not shift the lookup. The scalar is broadcast across the trial as row 3 of the input array. Range 0–1.

ii.
```python
        if i == 0:
            prev_outcome = 0
        else:
            prev_outcome = is_rewarded[i - 1]
...
        inp[3, :] = float(prev_outcome)        # previous trial outcome
```

iii. Directly follows the spec ("Previous trial outcome (binary, omitted = 0, rewarded = 1, per trial)"). No justification is offered for coding the undefined first trial as 0 rather than excluding it.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Two things: the `position` behavioural time series, and the trial's reward-zone identity (A/B/C), which is taken from the scene name parsed from the NWB `identifier` (switching at trial 30 on switch sessions) rather than from the `reward_zone` time series. The zone edges are hard-coded from the Methods: A = 80–130 cm, B = 200–250 cm, C = 320–370 cm.

ii.
```python
REWARD_ZONES = {'A': (80.0, 130.0), 'B': (200.0, 250.0), 'C': (320.0, 370.0)}
...
        trial_pos = position[on_track_idx]
        trial_pos = np.clip(trial_pos, 0, TRACK_LENGTH)
        zone_label = zones[i]
        rz = REWARD_ZONES[zone_label]
        out[0, :] = discretize_distance_to_reward(trial_pos, rz)
```

iii. The AI dispatched a sub-agent to work out what the `reward_zone` series encodes; the sub-agent reported that its non-zero values 1–6 "appear to encode something like the number of licks or a reward-related event counter, not the zone identity itself" and that the zone must be inferred otherwise. The AI concluded: "the reward zone identity needs to be inferred from the session identifier (scene name), not from the `reward_zone` signal in the NWB", and confirmed against the paper's `sessions_dict.py` that the identifier scenes match the documented per-mouse switch sequences. Zone boundaries come straight from the Methods text.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Position is first clipped to [0, 450] cm, then the signed distance to the nearest zone edge is computed: `position − zone_start` when before the zone (negative), `position − zone_end` when past it (positive), and exactly 0 anywhere inside the zone. This is the same definition as the reference. The result is then discretised (see 7-c).

ii.
```python
def discretize_distance_to_reward(position, reward_zone):
    rz_start, rz_end = reward_zone
    distance = np.where(
        position < rz_start,
        position - rz_start,  # negative (approaching)
        np.where(position > rz_end,
                 position - rz_end,  # positive (past)
                 0.0))  # inside zone
```

iii. Trajectory: "For the reward zone distance, I'm computing a signed distance based on position relative to the zone boundaries - negative when approaching, zero inside the zone, positive when past it - and then discretizing that into bins." This is the paper's notion of "distance relative to reward" and matches the instruction's bin 3 = "0 cm" (i.e. in-zone).

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Seven categories assigned by explicit boolean masks: 0 for d < −50; 1 for −50 ≤ d < −10; 2 for −10 ≤ d < 0; 3 for d == 0 (in zone); 4 for 0 < d ≤ 10; 5 for 10 < d ≤ 50; 6 for d > 50. These reproduce the instruction's bins; the only differences from the reference (which uses `np.digitize` with edges `[-inf,-50,-10,0,1e-6,10,50,inf]`) are the closed/open convention at exactly ±10 and +50, which affects a negligible number of samples. Class fractions: 0.253/0.101/0.073/0.237/0.021/0.071/0.244 vs reference 0.253/0.102/0.074/0.237/0.021/0.072/0.243 — essentially identical.

ii.
```python
    bins = np.zeros_like(distance, dtype=np.int64)
    bins[distance < -50] = 0
    bins[(distance >= -50) & (distance < -10)] = 1
    bins[(distance >= -10) & (distance < 0)] = 2
    bins[distance == 0] = 3
    bins[(distance > 0) & (distance <= 10)] = 4
    bins[(distance > 10) & (distance <= 50)] = 5
    bins[distance > 50] = 6
```

iii. The bin edges are transcribed directly from the "Decoder Outputs" section of the instructions; the `distance == 0` test is exact because the in-zone case is assigned a literal 0.0 rather than computed.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position is sliced with the same `on_track_idx` vector as the neural matrix, so the output row is sample-for-sample aligned with the neural columns; no shift, interpolation or causal offset is applied.

ii.
```python
        trial_pos = position[on_track_idx]
...
        out = np.zeros((6, n_tp), dtype=np.int64)
        out[0, :] = discretize_distance_to_reward(trial_pos, rz)
```

iii. Behaviour and imaging share one sample index in these files (one behavioural sample per imaging frame), so identical indexing is the alignment.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. The `position` behavioural time series alone (cm along the 450 cm virtual corridor).

ii.
```python
        position = beh['position']['data'][:]
...
        trial_pos = position[on_track_idx]
```

iii. `position` is the direct VR position readout; no derivation is needed.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The trial's positions are clipped to [0, 450] cm (recall the on-track mask has already restricted them to [0, 455]), then binned. The clip is what keeps the handful of samples that overshoot 450 cm inside the last bin. No smoothing, no unwrapping.

ii.
```python
        trial_pos = position[on_track_idx]
        trial_pos = np.clip(trial_pos, 0, TRACK_LENGTH)
...
        out[1, :] = discretize_position(trial_pos)
```

iii. Track length 450 cm is from the Methods; the clip is a defensive guard so that the floor-divide discretisation cannot emit an out-of-range class.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Five equal 90 cm bins by floor division, clipped to 0–4: `bin = clip(floor(position / 90), 0, 4)`. This gives 0: <90, 1: 90–180, 2: 180–270, 3: 270–360, 4: ≥360, as the instructions specify. Class fractions 0.210/0.178/0.231/0.226/0.156 vs reference 0.211/0.178/0.231/0.227/0.154.

ii.
```python
def discretize_position(position):
    """Discretize position into 5 equal bins spanning 450 cm."""
    bin_size = TRACK_LENGTH / 5  # 90 cm
    bins = np.clip(np.floor(position / bin_size).astype(np.int64), 0, 4)
    return bins
```

iii. Directly from the instruction ("5 equal-sized bins spanning the 450 cm track"); 450/5 = 90 cm is computed rather than hard-coded.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same `on_track_idx` slice as the neural matrix — sample-for-sample, no offset.

ii.
```python
        trial_pos = position[on_track_idx]
        out[1, :] = discretize_position(trial_pos)
```

iii. As in 7-d: one behavioural sample per imaging frame, so shared indexing is the alignment.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. The `lick` behavioural time series (a per-frame lick count, integer 0–6 in these files).

ii.
```python
        lick = beh['lick']['data'][:]
...
        trial_lick = lick[on_track_idx]
```

iii. The AI inspected the series and found "the NWB stores lick counts per frame ranging 0-6", i.e. counts rather than a binary flag, so it needs thresholding.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Two steps. (1) A stuck-lick-sensor correction taken from the paper's code: if more than 35% of a trial's frames have a lick count > 2, the whole trial's lick trace is set to 0. (2) Binarisation `lick > 0 → 1`. The correction fires on 69 of 12,216 trials (17 sessions) and lowers the overall lick fraction from 0.230 (reference, no correction) to 0.219. Note the paper's own routine sets such licks to NaN (i.e. treats them as missing); the AI sets them to 0, which labels them as "no lick".

ii.
```python
LICK_CORRECTION_THR = 0.35  # Fraction of frames with lick > 2 to flag bad sensor
...
def check_lick_sensor_error(lick_data, threshold=LICK_CORRECTION_THR):
    frac_high = np.mean(lick_data > 2)
    return frac_high > threshold
...
        if check_lick_sensor_error(trial_lick):
            trial_lick = np.zeros_like(trial_lick)
        trial_lick_binary = (trial_lick > 0).astype(np.int64)
...
        out[3, :] = trial_lick_binary
```

iii. The AI found `lick_correction_thr = 0.35` among the `dayData` defaults in the paper's code ("threshold for detecting stuck lick sensors") and noted the Methods text instead says 30%: "This discrepancy between the paper's stated 30% and the code's actual 35% threshold is worth flagging ... Since the instructions emphasize matching implemented code, I'll go with 0.35." It also sanity-checked the physiology: "at 15.5 Hz, a per-frame lick count over 2 seems implausibly high, so this likely signals a stuck sensor rather than genuine licking." It recorded the NaN→0 substitution in the module docstring ("lick set to 0 for those trials") without arguing for it.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same `on_track_idx` slice as the neural matrix — sample-for-sample, no offset or smoothing, so the binary lick series marks the imaging frames during which licks were counted.

ii.
```python
        trial_lick = lick[on_track_idx]
        ...
        out[3, :] = trial_lick_binary
```

iii. As in 7-d/8-d.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. The NWB top-level `identifier` string only: the scene name (e.g. `Env1_LocationC_to_B`, `Env2_B_to_Env1_A`, `Env1_LocationA`) is regex-parsed into `zone_before`/`zone_after`, and switch sessions take `zone_after` from trial index 30 onward. Neither `reward_zone` nor `position` at reward delivery is consulted. I validated this against the data (zone inferred from the position at which `reward_zone` first goes positive): 0 mismatches on the 10,394 trials where the animal entered the zone, and it also labels the 1,822 trials where the zone was never entered. Class fractions 0.329/0.337/0.334 vs reference 0.329/0.337/0.334.

ii.
```python
    m = re.match(r'(Env[12])_([ABC])_to_(Env[12])_([ABC])', scene)   # cross-env switch
    m = re.match(r'(Env[12])_Location([ABC])_to_([ABC])', scene)      # within-env switch
    m = re.match(r'(Env[12])_Location([ABC])', scene)                 # stable
...
REWARD_ZONE_LABELS = {'A': 0, 'B': 1, 'C': 2}
...
        out[4, :] = REWARD_ZONE_LABELS[zone_label]  # reward zone location
```

iii. See 7-a: after the sub-agent concluded the `reward_zone` series does not encode zone identity, the AI decided the scene name is the authoritative source, and cross-validated the identifier-derived scene sequences for m3/m11/m17 against `sessions_dict.py` ("This matches the sessions_dict entries exactly"). The trial-30 switch point comes from the Methods.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. A lookup `{'A': 0, 'B': 1, 'C': 2}` applied to the per-trial zone label, broadcast across all timepoints of the trial (time-varying representation of a per-trial constant), as row 4 of the output array. `output_values[4] = ['A','B','C']`.

ii.
```python
    zones, envs = get_per_trial_info(session_info, n_trials)
...
        zone_label = zones[i]
        out[4, :] = REWARD_ZONE_LABELS[zone_label]
```

iii. The instruction specifies "Reward zone location, per-trial. 0 = A, 1 = B, 2 = C"; the format note "If at all possible, make it time-varying" motivates the broadcast.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. The `Reward` behavioural time series' event `timestamps`, compared against each trial's time window (same `is_rewarded` array used for the previous-trial input). Auto-rewards and operant rewards are not distinguished; the `autoreward` series is not read.

ii.
```python
        reward_timestamps = beh['Reward']['timestamps'][:]
...
        if np.any((reward_timestamps >= t_start_time) & (reward_timestamps <= t_end_time)):
            is_rewarded[i] = 1
```

iii. The AI treated the presence of a `Reward` event within the lap as the definition of a rewarded trial, explicitly accepting that omission trials and no-lick trials are indistinguishable and that this is the correct behaviour for a binary "reward outcome" target.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. The per-trial binary flag is broadcast across all timepoints as row 5 of the output array. The trial window is `[beh_timestamps[trial_start], beh_timestamps[trial_end-1]]`, i.e. the full lap including the teleport-onset frame (note the window uses the *unfiltered* trial bounds, so a reward delivered in a frame later dropped by the on-track mask would still count). Class fractions 0.157/0.843 vs reference 0.157/0.843.

ii.
```python
        out[5, :] = is_rewarded[i]                    # reward outcome
```

iii. The spec asks for a per-trial binary reward outcome; broadcasting follows the "make it time-varying if at all possible" guidance.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Five defensive measures, all silent except the session-skip prints:
- **Length mismatch between streams**: everything (all ophys arrays, all behavioural arrays, timestamps) is truncated to the common minimum length. This is what handles the 10 sessions (m17 ses-04/06, m18 ses-01/05/07/10/11/12/13/14) where the ophys series has exactly one frame more than the behavioural series. No warning is printed and the number of dropped frames is not reported.
- **Degenerate trials/sessions**: trials with < 5 on-track frames are skipped; sessions with < 2 trials, 0 surviving cells, or < 2 valid trials are dropped entirely (none of these fire in practice).
- **Non-finite neural values**: any NaN/Inf in a trial's neural matrix is replaced with 0.
- **Division by ~0 in the dF/F baseline**: `|baseline|` is floored at 1e-6.
- **Degenerate correlations**: interneuron detection is skipped if fewer than 10 valid frames exist or if a cell's (or speed's) standard deviation is < 1e-10.
There is no cross-check that the different behavioural series share timestamps, and no tolerance check on reward-event alignment.

ii.
```python
    n_time = min(deconvolved.shape[0], fluorescence.shape[0], neuropil_data.shape[0],
                 len(position), len(speed), len(lick), len(trial_number),
                 len(trial_start_sig), len(teleport_sig), len(beh_timestamps))
```
```python
        if len(on_track_idx) < 5:  # Skip very short trials
            continue
...
    if n_good_cells < 1:
        print(f"  Skipping session with 0 good cells: {scene}")
        return None
```
```python
        if np.any(np.isnan(trial_neural)) or np.any(np.isinf(trial_neural)):
            trial_neural = np.nan_to_num(trial_neural, nan=0.0, posinf=0.0, neginf=0.0)
```
```python
        abs_baseline = np.abs(baseline)
        abs_baseline[abs_baseline < 1e-6] = 1e-6  # avoid division by zero
```

iii. These were added reactively while debugging: the length crop was introduced after the multi-plane sessions crashed ("Shape mismatch between neural data and behavioral data for multi-plane sessions"), and the rest are generic guards. The AI also verified the output with `train_decoder.py --verify-only` and `--plot-samples` and inspected the longest trials before accepting them as real.

## 13-a. What are the most time-consuming steps of the code?

i. In order: (1) reading the ophys arrays out of the 152 NWB files — for every session it pulls the *full* `Deconvolved`, `Fluorescence` and `Neuropil` matrices (three `(T, n_roi)` float arrays, all ROIs, before any `iscell` masking), which is I/O- and memory-bound; (2) `compute_dff_for_interneuron_detection`, which per trial runs a Gaussian filter plus a 300-sample minimum filter plus a 300-sample maximum filter plus a second Gaussian filter over an `(n_trial_frames, n_cells)` block — ~12,000 trials × up to 2,300 cells; (3) the per-cell `scipy.stats.pearsonr` loop (138k calls in total, each also computing an unused p-value); (4) pickling the 9.85 GB result in one `pickle.dump`. The whole conversion nevertheless fits comfortably inside the 10-minute command budget the AI gave it.

ii. N/A — this is a property of the code as a whole; the relevant snippets appear under 2-b and 2-c.

iii. Not discussed by the AI. It did structure the script as a single pass over the files (no separate survey stage), which avoids reading every NWB file twice.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. Four:
- `detect_interneurons`: the `for i in range(n_cells)` loop over `pearsonr`. The whole correlation vector can be computed in one matrix expression (centre `dff[valid]` and `speed_valid`, then a single dot product / `np.corrcoef` call), as would dropping the discarded p-value.
- `get_trial_boundaries`: the `for sf in start_frames` loop re-scans the entire `teleport_frames` array per trial (`teleport_frames[teleport_frames > sf]`), i.e. O(n_trials × n_teleports). A single `np.searchsorted(teleport_frames, start_frames)` would do it in one call.
- `compute_dff_for_interneuron_detection`: the per-trial filtering loop. Harder to remove because the maximin baseline is defined *within* trial, but it could at least be skipped entirely for the many sessions where no cell is near threshold, or restricted to a subsample of frames.
- The main `for i in range(n_trials)` extraction loop: `discretize_distance_to_reward`, `discretize_position` and `discretize_speed` are pure elementwise functions and could be applied once to the whole session array before splitting, as could the `on_track` mask.

ii. N/A

iii. Not discussed by the AI.

## 13-c. What processing does the code repeat multiple times?

i. Little. The script makes a single pass over each NWB file, so unlike a survey-then-convert design it does not read the same arrays twice. What *is* repeated is small: `position` is sliced once for the `on_track` mask inside `extract_on_track_indices` and then indexed again in the trial body; `is_rewarded[i]` is used for both the `reward_outcome` output and the next trial's `previous_trial_outcome` input (computed once, so this is fine); and the per-trial time vector is built during conversion and then differenced again at the end of `main()` over every trial in the dataset just to recover the median bin size.

ii.
```python
    all_dt = []
    for sess_inputs in data['input']:
        for trial_inp in sess_inputs:
            if trial_inp.ndim == 2 and trial_inp.shape[1] > 1:
                dt = np.diff(trial_inp[0, :])
                all_dt.extend(dt.tolist())
```

iii. Not discussed by the AI. (The final `all_dt` pass materialises a Python list of ~2.6 million floats purely to take one median; the same number is available from any single session.)

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several small items, none of which corrupt the output:
- `planeIdx` is read from the file on multi-plane sessions and then never used; `n_cells_raw` is computed and never used; `detect_interneurons`'s `iscell_mask` parameter is accepted and never used.
- The entire maximin dF/F pipeline (`Fluorescence` + `Neuropil` read in full, four filter passes per trial) exists solely to remove 342 of 138,678 cells (0.25%) — the dF/F itself is discarded and never reaches the saved data. Within it, the σ=2 Gaussian smoothing of dF/F is irrelevant to a Pearson correlation against speed and could be dropped.
- `pearsonr` returns a p-value that is discarded.
- `is_rewarded` is computed for every trial including ones later skipped.
- The full-dataset `all_dt` list (see 13-c) is built only to take a median.
The largest genuine waste is the dF/F computation: it reproduces most of the paper's neural pipeline at full cost but its product is thrown away, while the saved neural data comes from a different source.

ii.
```python
            plane_idx = ophys['ImageSegmentation']['PlaneSegmentation']['planeIdx'][:]
```
```python
    n_cells_raw = int(np.sum(cell_mask))
```
```python
def detect_interneurons(dff, speed, iscell_mask, threshold=INTERNEURON_SPEED_CORR_THR):
```
```python
        r, _ = pearsonr(cell_dff, speed_valid)
```

iii. Not discussed by the AI.
