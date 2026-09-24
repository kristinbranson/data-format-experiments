# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI discovers data by listing every sub-directory of `data/` whose name starts with `sub-`, then globbing every `*.nwb` file inside each of those directories. Each NWB file becomes one session record `{subject, filepath, filename}`. All 152 files (11 subjects) are found and processed in a single pass; no subject/session/file is excluded a priori. Files are opened directly with `h5py` (not `pynwb`), reading the HDF5 paths `processing/ophys/...` and `processing/behavior/BehavioralTimeSeries/...`. Each session is read exactly once (no separate survey pass). In `--sample` mode only two sessions (index 0 and the middle one) are processed.

ii.
```python
def find_nwb_files(data_dir='data'):
    """Find all NWB files organized by subject."""
    subjects = sorted([d for d in os.listdir(data_dir) if d.startswith('sub-')])
    sessions = []
    for subj in subjects:
        subj_dir = os.path.join(data_dir, subj)
        files = sorted(glob(os.path.join(subj_dir, '*.nwb')))
        for fpath in files:
            sessions.append({
                'subject': subj.replace('sub-', ''),
                'filepath': fpath,
                'filename': os.path.basename(fpath),
            })
    return sessions
```
```python
    with h5py.File(filepath, 'r') as f:
        subject_id = f['general/subject/subject_id'][()].decode()
        identifier = f['identifier'][()].decode()
        ...
        position = f['processing/behavior/BehavioralTimeSeries/position/data'][:]
```
```python
    sessions_info = find_nwb_files('data')
    print(f"Found {len(sessions_info)} NWB files")
    ...
    for i, sess_info in enumerate(sessions_info):
        result = process_session(sess_info['filepath'], ...)
```

iii. From CONVERSION_NOTES.md Step 2: "NWB files organized: `data/sub-{id}/sub-{id}_ses-{ses}_behavior+ophys.nwb`", giving "11 subjects, 152 sessions", which the AI cross-checked against the paper ("Sessions 152 (10*14 + 1*12) — YES"). The AI chose `h5py` over `pynwb` for speed and because it only needs a handful of datasets; it documents the exact HDF5 paths it reads in Step 2. The conversion log confirms "Found 152 NWB files" and the final summary reports all 152 sessions / 12,216 trials retained.

## 1-b. How are the data split into subjects (mice)?

i. One subject per `sub-*` directory. The subject label actually written into the output is not the directory name but the `general/subject/subject_id` field read from inside each NWB file. After all sessions are processed, the unique subject ids are sorted to build `subjects`, and `subject_idx` maps each session to its index in that list. Result: 11 subjects (m3, m4, m7, m11–m15, m17–m19), 14 sessions each except m11 with 12.

ii.
```python
        subject_id = f['general/subject/subject_id'][()].decode()
        ...
        return {..., 'subject': subject_id, ...}
```
```python
    unique_subjects = sorted(set(all_subject_ids))
    subject_idx = np.array([unique_subjects.index(s) for s in all_subject_ids])
    ...
    data = {..., 'subjects': unique_subjects, 'subject_idx': subject_idx, ...}
```

iii. CONVERSION_NOTES.md Step 2/3: "Subjects | 11 (m3, m4, m7, m11-m15, m17-m19)" and "Sessions/subject | 12-14 (m11 has 12, rest 14)", checked against the paper's 11 mice ("Subjects | 11 mice | 11 | YES"). No explicit rationale is given for preferring the in-file `subject_id` over the directory name; they agree.

## 1-c. How are the data split into sessions?

i. One session = one NWB file. Sessions are kept in sorted filename order within each subject, and subjects are processed in sorted directory order, so the session ordering in `neural`/`input`/`output`/`subject_idx` is deterministic. No merging or alignment of cells across days is attempted. A session is dropped only if it yields fewer than 2 trials (never triggered).

ii.
```python
        files = sorted(glob(os.path.join(subj_dir, '*.nwb')))
        for fpath in files:
            sessions.append({...})
```
```python
    if n_trials < 2:
        print(f"  WARNING: Only {n_trials} trials, skipping session")
        return None
    ...
        if result['n_trials'] < 2:
            print(f"  Skipping: fewer than 2 valid trials")
            continue
```

iii. CONVERSION_NOTES.md Step 2 documents the `ses-{ses}` naming and the 152-file total; Step 9 reports "All 152/152 sessions processed successfully". The `< 2 trials` guard exists because the target format requires "at least two trials within each session in order to evaluate the decoder performance".

## 1-d. How are the data split into trials?

i. Trials are delimited by the behavior time series `trial_start` and `teleport`: a trial starts at the first sample where `trial_start > 0` and ends (exclusive) at the sample where `teleport > 0`. If the number of teleport samples does not equal the number of trial starts, the AI re-derives the ends by taking, for each trial start, the first teleport sample after it, and truncates to the matched count. The inter-trial / teleport period is therefore excluded from every trial. Trials are indexed `si:ei` on all streams (neural and behavior alike).

ii.
```python
    # === Trial boundaries ===
    trial_start_inds = np.where(trial_start > 0)[0]
    teleport_inds = np.where(teleport_sig > 0)[0]
    n_trials = len(trial_start_inds)
    ...
    # Match teleport to trial start (ensure same count)
    if len(teleport_inds) != n_trials:
        matched_teleports = []
        for ts in trial_start_inds:
            tp_after = teleport_inds[teleport_inds > ts]
            if len(tp_after) > 0:
                matched_teleports.append(tp_after[0])
        teleport_inds = np.array(matched_teleports)
        n_trials = min(n_trials, len(teleport_inds))
        trial_start_inds = trial_start_inds[:n_trials]
```
```python
    for t in range(n_trials):
        si = trial_start_inds[t]
        ei = teleport_inds[t]
        n_tp = ei - si
```

iii. CONVERSION_NOTES.md Step 1: "Trial boundaries: trial_start and teleport signals in behavior timeseries". The AI reports "Mean trials/session 80.4 +/- 6.1" against the paper's "80.5 +/- 7.4" and calls it a match. (Independent check by this reviewer: the resulting trial-length statistics — 12,216 trials, T_mean 216.783, T_min 96, T_max 3359 — are bit-for-bit identical to the human reference solution, so the segmentation is in fact exactly the reference segmentation.)

## 1-e. How are trials filtered based on quality controls?

i. Essentially no quality filtering of trials. The only rejections are degenerate ones: a trial with fewer than 2 time points is skipped, and a session with fewer than 2 trials is dropped. There is no minimum trial-duration criterion, no speed/engagement criterion, and no rejection of trials with missing reward-zone information. Separately (see 9-b) trials whose lick sensor appears stuck are *kept* but have their lick trace zeroed.

ii.
```python
    for t in range(n_trials):
        si = trial_start_inds[t]
        ei = teleport_inds[t]
        n_tp = ei - si

        if n_tp < 2:
            continue
```
```python
    if n_trials < 2:
        print(f"  WARNING: Only {n_trials} trials, skipping session")
        return None
```
```python
        # Lick sensor error correction per trial (>30% frames with count>2 -> NaN)
        if len(trial_lick) > 0 and (np.sum(trial_lick > 2) / len(trial_lick)) > 0.30:
            lick_binary[si:ei] = 0
```

iii. CONVERSION_NOTES.md never states an explicit trial-curation rule beyond the format requirement of ≥2 trials per session; Step 1 lists `correct_lick_sensor_error()` from the reference code under "CURATION" and it is the only curation rule carried over at trial level. The implicit justification is that no trials in this dataset are pathologically short (the shortest retained trial is 96 samples ≈ 6.2 s), so no length threshold was needed.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The `neural` matrices are taken **directly** from the NWB `processing/ophys/Deconvolved/<plane>/data` arrays, concatenated across imaging planes along the ROI axis. `Fluorescence` (F) and `Neuropil` (Fneu) are also read, but only to build an approximate dF/F used by the interneuron filter — they never enter the saved `neural` data.

ii.
```python
        deconv_group = f['processing/ophys/Deconvolved']
        planes = sorted(deconv_group.keys())  # e.g. ['plane0'] or ['plane0', 'plane1']
        for plane in planes:
            d = f[f'processing/ophys/Deconvolved/{plane}/data'][:]
            fl = f[f'processing/ophys/Fluorescence/{plane}/data'][:]
            ne = f[f'processing/ophys/Neuropil/{plane}/data'][:]
            deconv_list.append(d); flu_list.append(fl); neu_list.append(ne)
        deconv = np.concatenate(deconv_list, axis=1)
```
```python
    neural_all = deconv[:, final_cell_mask].T  # (n_neurons, n_timepoints)
    neural_all = neural_all.astype(np.float32)
```

iii. CONVERSION_NOTES.md Step 1 asserts: "**Key**: NWB `Deconvolved` data is the deconvolved events AFTER full dF/F processing pipeline (computed from the multi_anim_sess notebook)" and "**Important**: The NWB files contain the ALREADY PROCESSED deconvolved events. We should use these directly." Step 4 records the same resolution: "Deconvolved data | Use NWB data directly - already fully processed". Step 13 lists it as key design decision #1. The claim was never tested against the raw F/Fneu.

## 2-b. How is the `neural` data processed?

i. No processing at all is applied to the neural signal: the stored `Deconvolved` array is transposed to (n_neurons, n_timepoints), cast to `float32`, sliced per trial, and `np.nan_to_num`-ed. None of the paper's pipeline — neuropil subtraction with coefficient 0.7 and per-trial neuropil add-back, maximin baseline over a 20 s (300-sample) window, dF/F = (F−baseline)/|baseline|, 2-sample Gaussian smoothing, OASIS deconvolution at tau = 0.7 with frame_rate/n_planes, and NaN-ing of the inter-trial interval — is reproduced. A separate, much cruder signal is computed solely for the interneuron test: `dff_simple = (F − 0.7·Fneu − median) / |median|` with a single whole-session median as the baseline.

ii.
```python
        trial_neural = neural_all[:, si:ei].copy()
        # Replace NaN with 0 in neural data
        trial_neural = np.nan_to_num(trial_neural, nan=0.0)
```
```python
    # Step 2: Interneuron exclusion - compute dF/F correlation with speed
    f_corrected = fluorescence - 0.7 * neuropil_data
    f_median = np.median(f_corrected, axis=0, keepdims=True)
    f_median[f_median == 0] = 1
    dff_simple = (f_corrected - f_median) / np.abs(f_median)
```

iii. Justification is the (incorrect) premise recorded in Step 1 — that the stored `Deconvolved` array is already the paper's `events` signal — so that recomputing it would be redundant. CONVERSION_NOTES.md nevertheless transcribes the paper's true pipeline in full ("dF/F Processing (from preprocessing.py): 1. NaN out inter-trial (teleport) periods 2. Neuropil subtraction ... 6. Deconvolve with OASIS (tau=0.7, frame_rate ~15.5 Hz)") and then declines to run it. Step 13 metadata labels the result "deconvolved calcium events".

*Reviewer note:* the stored `Deconvolved` arrays contain no NaNs anywhere (including the laser-blanked inter-trial periods) and span 0 to ~13,771 in raw-fluorescence units (session-mean F ≈ 1,371), i.e. they are Suite2p's own deconvolution of raw F, not a deconvolution of the paper's dF/F.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters, matching the paper's stated rules. (1) Suite2p curation: only ROIs with `iscell[:,0] == 1` are kept; for two-plane sessions the concatenated column order is mapped back to `iscell` rows via `planeIdx`. (2) Putative-interneuron exclusion: for each accepted ROI, the Pearson correlation between the crude `dff_simple` trace and running speed is computed over frames with `speed > 0 & position >= 0 & ~isnan(speed)`, and cells with r > 0.5 are dropped. The correlation is vectorised as a z-scored dot product. Result: 138,603 neurons over 152 sessions (mean 911.9, range 155–2339).

ii.
```python
        iscell = f['processing/ophys/ImageSegmentation/PlaneSegmentation/iscell'][:]
        planeIdx = f['processing/ophys/ImageSegmentation/PlaneSegmentation/planeIdx'][:]
        ...
            plane_mask = planeIdx == plane_num
            plane_offsets[plane_num] = np.where(plane_mask)[0]
        ...
        concat_to_iscell = np.concatenate(concat_to_iscell)
    cell_mask_concat = np.array([iscell[concat_to_iscell[i], 0] == 1 for i in range(n_rois_concat)])
```
```python
    valid_mask = (speed > 0) & (position >= 0) & ~np.isnan(speed)
    if valid_mask.sum() > 100:
        speed_z = (speed_valid - speed_valid.mean()) / (speed_valid.std() + 1e-10)
        dff_accepted = dff_simple[valid_mask][:, accepted_cols]
        dff_z = (dff_accepted - dff_mean) / dff_std
        corrs = (speed_z @ dff_z) / len(speed_z)
        for i, col_idx in enumerate(accepted_cols):
            if corrs[i] > 0.5:
                interneuron_mask[col_idx] = True
    final_cell_mask = cell_mask_concat & ~interneuron_mask
```

iii. CONVERSION_NOTES.md Step 1: "`iscell`: Suite2p cell curation stored in NWB; iscell[:,0]==1 means accepted cell" and lists "Interneuron exclusion via speed-dFF Pearson correlation > 0.5" as a key concept; Step 3 compares "Interneuron exclusion | 0.42 +/- 0.85% | Low rate observed | YES"; Step 13 design decision #2. The multi-plane index mapping is justified in Step 4: "Multi-plane data (m17/m18) | Concatenate plane0+plane1, map iscell indices via planeIdx".

*Reviewer note:* the per-session conversion log shows "0 interneurons removed" for almost every session (5 cells out of 56,695 accepted across the 71 logged sessions, ≈0.009%), i.e. ~50× fewer than the paper's 0.42 ± 0.85%, so in practice the second filter is inert.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment to trial start requires no extra work: the neural array is indexed with exactly the same sample range `si:ei` (trial_start sample to teleport sample) as every behavioural stream, so sample 0 of every trial is the trial-start frame. `off_start` is recorded as 0.0 and `off_end` as None (variable-length trials). Before slicing, the neural and behaviour arrays are truncated to a common length when they differ (10 multi-plane sessions were off by one).

ii.
```python
    if n_timepoints != n_behav:
        min_len = min(n_timepoints, n_behav)
        deconv = deconv[:min_len]; position = position[:min_len]; ... ; n_timepoints = min_len
```
```python
        si = trial_start_inds[t]; ei = teleport_inds[t]
        trial_neural = neural_all[:, si:ei].copy()
        trial_pos = position[si:ei]
```
```python
            'temporal_alignment_event': 'start of trial (entry to linear track)',
            'off_start': 0.0,
            'off_end': None,  # variable trial length
```

iii. CONVERSION_NOTES.md Step 1: "Behavior aligned to imaging: All behavior timeseries at ~15.5 Hz imaging frame rate", so the two streams share a sample index and no resampling or shifting is needed. Step 4/9: "Off-by-one behavior/neural length mismatch in some sessions (truncated to shorter)" — "10 sessions had neural/behavior length alignment (off-by-one, all multi-plane)".

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning or resampling — data are kept at the native per-plane imaging rate. `metadata['time_bin_size']` is set from a hard-coded nominal rate constant `IMAGING_RATE_NOMINAL = 15.5078125` Hz, giving 64.4836 ms, and `metadata['imaging_rate_hz']` is set to the same constant for every session.

ii.
```python
IMAGING_RATE_NOMINAL = 15.5078125  # Hz
...
    # Compute time bin size
    time_bin_ms = 1000.0 / IMAGING_RATE_NOMINAL
...
            'time_bin_size': time_bin_ms,
            'imaging_rate_hz': IMAGING_RATE_NOMINAL,
```

iii. CONVERSION_NOTES.md Step 2/3: "Imaging rate | ~15.5 Hz | 15.5078125 Hz | YES" and "~15.5 Hz per plane, data concatenated across planes"; Step 9 reports "Time bin | 64.48 ms". Because every session's behaviour sampling interval is 64.4836 ms, a single constant is used for all sessions rather than reading `rate` per file.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is **not** derived from any stored timestamp. It is synthesised from the within-trial frame index multiplied by a frame period taken from the NWB metadata field `general/optophysiology/ImagingPlane/imaging_rate`. The behaviour timestamps (`.../position/timestamps`) are loaded but used only for mapping reward events to frames, never for this input.

ii.
```python
        imaging_rate = f['general/optophysiology/ImagingPlane/imaging_rate'][()]
    ...
    frame_time = 1.0 / imaging_rate  # seconds per frame
    ...
        time_from_start = np.arange(n_tp, dtype=np.float32) * frame_time
```

iii. CONVERSION_NOTES.md Step 5 variable mapping: "Time from trial start | input[0] | Frame index * time_bin_size, in seconds". The AI's premise is that the sampling rate is constant and identical for behaviour and imaging ("Behavior aligned to imaging: All behavior timeseries at ~15.5 Hz imaging frame rate"), so a regular grid reproduces the timestamps.

*Reviewer note:* `ImagingPlane/imaging_rate` is the **scanner** rate. In the 28 two-plane sessions (m17, m18) it is 31.015625 Hz while the per-plane/behaviour rate is 15.5078125 Hz, so the time axis in those sessions runs at half speed.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. `np.arange(n_tp) * frame_time`, i.e. a uniform ramp starting at exactly 0.0 for every trial (equivalent to subtracting the trial's first timestamp). It is stored as `input[0]`, the only genuinely time-varying input.

ii.
```python
        time_from_start = np.arange(n_tp, dtype=np.float32) * frame_time
        ...
        trial_input_tv = time_from_start.reshape(1, -1)  # (1, n_tp)
        trial_input = np.vstack([
            trial_input_tv,
            np.full((1, n_tp), env_type, dtype=np.float32),
            np.full((1, n_tp), trial_num, dtype=np.float32),
            np.full((1, n_tp), prev_out, dtype=np.float32),
        ])  # (4, n_tp)
```

iii. Step 5 mapping table ("Frame index * time_bin_size, in seconds"); Step 10 range check: "time_from_trial_start | [0.0, 216.5] seconds | GOOD (most trials <30s)".

*Reviewer note:* the realised per-sample step is 0.032242 s in two-plane sessions and 0.064484 s elsewhere, contradicting the documented `time_bin_size` of 64.48 ms; 2,240 of 12,216 trials (18%) carry a halved time axis.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. By construction: the ramp has exactly `n_tp = ei - si` samples, the same length as the neural slice, and both begin at the trial-start frame. No offset or interpolation is applied.

ii.
```python
        n_tp = ei - si
        trial_neural = neural_all[:, si:ei].copy()
        time_from_start = np.arange(n_tp, dtype=np.float32) * frame_time
```

iii. Implicit in Step 1's finding that behaviour and imaging share a sample grid; the AI also relies on the up-front length truncation (see 2-d) to guarantee the two streams have a common index.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. From the **session identifier string**, not from the `environment` behaviour time series. `identifier` (e.g. `/data/InVivoDA/GCAMP11/23_02_2023/Env1_LocationB_to_A`) is split on `/` to get a "scene" name, which is parsed for `EnvN` tokens; the environment is assigned per trial (trials 0–29 from the "from" side, trials 30+ from the "to" side on switch days). The `environment` dataset is loaded from the NWB file and truncated, but is then never used.

ii.
```python
def parse_scene(identifier):
    return identifier.split('/')[-1]
```
```python
    if '_to_' in scene:
        parts = scene.split('_'); to_idx = parts.index('to')
        from_zone, from_env = _parse_zone_and_env(parts[:to_idx])
        to_zone, to_env = _parse_zone_and_env(parts[to_idx+1:])
        ...
        env_per_trial[:change_trial] = from_env
        env_per_trial[change_trial:] = to_env
```
```python
def _parse_zone_and_env(parts):
    env = 0
    zone = 'A'
    for p in parts:
        if p.startswith('Env'):
            env = int(p.replace('Env', '')) - 1
        elif p.startswith('Location'):
            zone = p.replace('Location', '')
        elif len(p) == 1 and p in 'ABC':
            zone = p
    return zone, env
```
```python
    # Use env_per_trial from scene parsing (more reliable for cross-env switches)
    # Fall back to NWB environment field where scene doesn't specify
    trial_env = env_per_trial.copy()
```

iii. The AI states the rationale inline and in the trajectory (step 100): "update the environment assignment to use env_per_trial from the scene parser instead of the NWB field (which is more reliable for cross-env switch sessions)". CONVERSION_NOTES.md Step 4 lists "Cross-env scene names | Handle `Env1_B_to_Env2_C` format in addition to `Env1_LocationB_to_A`". The comment promises a fallback to the NWB `environment` field, but no fallback is implemented.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. `_parse_zone_and_env` returns `env = EnvN − 1` (0 for Env1, 1 for Env2) when an `EnvN` token is present, and **defaults to `env = 0`** otherwise. The per-trial value is then broadcast across all timepoints of the trial as `input[1]`.

ii.
```python
    env = 0
    zone = 'A'
    for p in parts:
        if p.startswith('Env'):
            env = int(p.replace('Env', '')) - 1
```
```python
        env_type = np.float32(trial_env[t])
        ...
            np.full((1, n_tp), env_type, dtype=np.float32),
```

iii. Step 6–8 notes: "Scene parsing: handles all 26 unique scene name formats (single zone, within-env switch, cross-env switch)". Step 10 range check: "environment_type | [0.0, 1.0] | CORRECT (binary)".

*Reviewer note:* for within-environment switch scenes in Env2 (e.g. `Env2_LocationC_to_A`), the "to" side is the bare token `A`, which carries no `Env` token, so `_parse_zone_and_env` falls back to `env = 0`. 33 sessions / 1,720 trials (14% of all trials) are therefore labelled Env1 when the `environment` time series says Env2. The conversion log prints `env=[0 1]` for these single-environment sessions, but the discrepancy was not investigated.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. It is the within-session sequential trial index — the loop counter `t` over `trial_start_inds` — not the NWB `trial number` time series.

ii.
```python
    for t in range(n_trials):
        ...
        trial_num = np.float32(t)
```

iii. Not separately argued in CONVERSION_NOTES.md; the Step 5 mapping table simply says "Trial number | input[2] | Integer per trial". Step 10 range check: "trial_number | [0.0, 99.0] | CORRECT" (max 99 corresponds to the longest, 100-trial sessions).

## 5-b. What processing is involved in computing `input` *Trial number*?

i. None beyond casting the loop index to `float32` and broadcasting it constant across the trial's timepoints.

ii.
```python
        trial_num = np.float32(t)
        ...
            np.full((1, n_tp), trial_num, dtype=np.float32),
```

iii. Same as 5-a; the index is used as a continuous per-trial covariate as the Decoder Task requires.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. From the `Reward` behaviour time series, which stores sparse event timestamps rather than a frame-aligned trace. Each reward timestamp is snapped to the nearest behaviour timestamp to build a binary frame-aligned `reward_frames` vector; a trial is "rewarded" if any reward frame falls in `[si, ei)`. The previous trial's value is then used as the input.

ii.
```python
        reward_timestamps = f['processing/behavior/BehavioralTimeSeries/Reward/timestamps'][:]
    ...
    reward_frames = np.zeros(n_timepoints, dtype=np.float32)
    for rt in reward_timestamps:
        frame_idx = np.argmin(np.abs(behav_timestamps - rt))
        reward_frames[frame_idx] = 1.0

    trial_rewarded = np.zeros(n_trials, dtype=np.int64)
    for t in range(n_trials):
        si = trial_start_inds[t]; ei = teleport_inds[t]
        if np.any(reward_frames[si:ei] > 0):
            trial_rewarded[t] = 1
```

iii. CONVERSION_NOTES.md Step 2: "`Reward/data` and `Reward/timestamps`: Event-based (sparse), not frame-aligned. Shape (n_reward_events,). Reward timestamps matched to behavior timestamps to create frame-aligned reward signal." Step 4: "Reward field | Convert sparse timestamps to frame-aligned binary per trial"; Step 13 design decision #4.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. A one-trial lag of `trial_rewarded`, with trial 0 set to 0 (treated as omission), broadcast constant across the trial as `input[3]`.

ii.
```python
    prev_outcome = np.zeros(n_trials, dtype=np.int64)
    for t in range(1, n_trials):
        prev_outcome[t] = trial_rewarded[t - 1]
    # Trial 0: no previous, use 0 (omission)
```
```python
        prev_out = np.float32(prev_outcome[t])
        ...
            np.full((1, n_tp), prev_out, dtype=np.float32),
```

iii. Step 5 mapping: "Previous trial outcome | input[3] | Binary: 0=omission, 1=rewarded", following the Decoder Task spec. Step 10: "previous_trial_outcome | [0.0, 1.0] | CORRECT (binary)".

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. From the `position` behaviour time series together with reward-zone boundaries that are looked up from a hard-coded dictionary (A = [80,130], B = [200,250], C = [320,370] cm) indexed by the zone letter parsed out of the session `identifier`/scene name. The `reward_zone` behaviour time series is **not** used to determine the zone; on switch sessions the zone is assumed to change at trial 30.

ii.
```python
# Reward zone coordinates from behavior.py in reference code
REWARD_ZONE_DICT = {
    'A': [80, 130],   # mapped from 'X' in code
    'B': [200, 250],  # mapped from 'Y' in code
    'C': [320, 370],  # mapped from 'Z' in code
}
```
```python
def get_reward_zone_labels(scene, n_trials, change_trial=30):
    ...
        labels[:change_trial] = from_zone
        labels[change_trial:] = to_zone
        coords[:change_trial] = REWARD_ZONE_DICT[from_zone]
        coords[change_trial:] = REWARD_ZONE_DICT[to_zone]
```
```python
    rz_labels, rz_coords, env_per_trial = get_reward_zone_labels(scene, n_trials)
    ...
        rz_start = rz_coords[t, 0]; rz_end = rz_coords[t, 1]
        signed_dist = compute_distance_to_reward_zone(trial_pos, rz_start, rz_end)
```

iii. CONVERSION_NOTES.md Step 3 verifies "Reward zones | A:80-130, B:200-250, C:320-370 | Confirmed | YES", and Step 5 uses the scene name as the source of the per-trial zone. Trajectory step 65 records the check that motivated `change_trial=30`: "For the first 30 trials, rzone entry at ~200-250 (zone B), and after trial 30, at ~80-130 (zone A). The `rzone` field is a cumulative count of zone entry events, like `lick`."

*Reviewer note:* independently recomputing the zone from `reward_zone > 0` and `position` for every trial in the dataset gives 0 disagreements with the scene-derived labels across all 10,394 trials in which the animal entered a zone, so the hard-coded switch at trial 30 does hold throughout this dataset.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance to the nearest edge of that trial's reward zone: negative before the zone (`position − zone_start`), positive after it (`position − zone_end`), exactly 0 while inside. Computed vectorised over the trial's position samples, then discretised (7-c).

ii.
```python
def compute_distance_to_reward_zone(position, rz_start, rz_end):
    dist = np.zeros_like(position)
    before = position < rz_start
    after = position > rz_end
    inside = ~before & ~after
    dist[before] = position[before] - rz_start  # negative
    dist[after] = position[after] - rz_end       # positive
    dist[inside] = 0.0
    return dist
```

iii. Step 5 mapping: "Distance to reward zone | output[0] | 7 bins based on signed distance", following the Decoder Task's definition "distance to any location in the reward zone" (hence 0 anywhere inside the zone). Step 10: "distance_to_reward_zone | 7 bins, range [0,6], well-distributed | GOOD".

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Seven categories assigned by explicit boolean masks: `<−50 → 0`, `[−50,−10) → 1`, `[−10,0) → 2`, `==0 → 3`, `(0,10] → 4`, `(10,50] → 5`, `>50 → 6`. The exactly-zero case gets its own class, which is what makes "in the zone" a distinct label.

ii.
```python
def discretize_distance(distance):
    bins = np.zeros(len(distance), dtype=np.int64)
    bins[distance < -50] = 0
    bins[(distance >= -50) & (distance < -10)] = 1
    bins[(distance >= -10) & (distance < 0)] = 2
    bins[distance == 0] = 3
    bins[(distance > 0) & (distance <= 10)] = 4
    bins[(distance > 10) & (distance <= 50)] = 5
    bins[distance > 50] = 6
    return bins
```
```python
        'output_values': [
            ['< -50cm', '-50 to -10cm', '-10 to 0cm', '0cm (in zone)', '0 to +10cm', '+10 to +50cm', '> +50cm'],
```

iii. Bin edges are copied from the Decoder Task specification. Step 10 reports the realised distribution as "well-distributed"; the resulting class fractions (0.253, 0.102, 0.074, 0.237, 0.021, 0.072, 0.242) are essentially identical to the human reference's.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. The position slice uses the identical `si:ei` index range as the neural slice, so no separate alignment step exists.

ii.
```python
        trial_pos = position[si:ei]
        ...
        signed_dist = compute_distance_to_reward_zone(trial_pos, rz_start, rz_end)
        dist_bins = discretize_distance(signed_dist)
        ...
        trial_output = np.vstack([dist_bins.reshape(1, -1), ...])
```

iii. Rests on the Step 1 finding that all behaviour series share the imaging frame grid, plus the up-front truncation of neural and behaviour arrays to a common length (2-d).

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Directly from the `position` behaviour time series (cm along the VR corridor), sliced per trial.

ii.
```python
        position = f['processing/behavior/BehavioralTimeSeries/position/data'][:]
    ...
        trial_pos = position[si:ei]
```

iii. Step 2 lists `position` among the behaviour time series; Step 5 maps it to `output[1]`; Step 3 confirms "Track length | 450 cm | 450 cm | YES".

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. None beyond the per-trial slice and the discretisation: `floor(position / 90)` clipped into `[0, 4]`, so samples slightly below 0 cm or above 450 cm are absorbed into the end bins rather than creating extra classes.

ii.
```python
def discretize_position(position):
    """Discretize absolute position into 5 equal bins (0-90, 90-180, 180-270, 270-360, 360-450)."""
    bins = np.clip(np.floor(position / 90.0).astype(np.int64), 0, 4)
    return bins
```

iii. Step 5 mapping: "Absolute position | output[1] | 5 equal bins (90cm each)". Step 10: "absolute_position | 5 bins, roughly equal (~15-23% each) | GOOD".

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Five equal 90 cm bins over the 450 cm track (0–90, 90–180, 180–270, 270–360, 360–450), implemented by integer division and clipping; the lowest and highest bins are effectively open-ended.

ii.
```python
    bins = np.clip(np.floor(position / 90.0).astype(np.int64), 0, 4)
```
```python
            ['0-90cm', '90-180cm', '180-270cm', '270-360cm', '360-450cm'],
```

iii. Bin edges come straight from the Decoder Task ("5 equal-sized bins spanning the 450 cm track"), with `TRACK_LENGTH = 450` documented from the Methods.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same `si:ei` slice as the neural data; no additional alignment.

ii.
```python
        trial_pos = position[si:ei]
        pos_bins = discretize_position(trial_pos)
```

iii. Same rationale as 7-d (shared frame grid, common-length truncation).

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. From the `lick` behaviour time series, which stores a per-frame lick count rather than a binary flag.

ii.
```python
        lick = f['processing/behavior/BehavioralTimeSeries/lick/data'][:]
```

iii. Step 1 notes the reference function `correct_lick_sensor_error()` and that lick is "cumulative per frame"; Step 5 maps it to `output[3]` as "Binary 0/1 per frame".

## 9-b. What processing is involved in computing `output` *Lick*?

i. Two steps. (1) Binarisation: any count > 0 becomes 1. (2) A stuck-sensor correction ported from the reference codebase: within each trial, if more than 30% of frames have a lick count > 2, the whole trial's lick trace is overwritten with **0** (the reference function sets it to NaN; the AI substitutes 0 "for cleaner output"). A redundant second binarisation `(trial_lick > 0)` is applied when the trial is assembled.

ii.
```python
    # Following reference code: lick is cumulative per frame, set >1 to 1
    lick_binary = lick.copy()
    lick_binary[lick_binary > 0] = 1
    # Lick sensor error correction per trial (>30% frames with count>2 -> NaN)
    for t in range(n_trials):
        si = trial_start_inds[t]; ei = teleport_inds[t]
        trial_lick = lick[si:ei]
        if len(trial_lick) > 0 and (np.sum(trial_lick > 2) / len(trial_lick)) > 0.30:
            lick_binary[si:ei] = 0  # Set to 0 instead of NaN for cleaner output
```
```python
        lick_out = (trial_lick > 0).astype(np.int64)
```

iii. Step 1 lists `correct_lick_sensor_error()` under CURATION: "Fix stuck lick sensor: if >30-50% frames have cumcount>2, set to NaN". The 0 substitution is justified in the inline comment only ("cleaner output"), since the target format has no NaN representation. Step 10 reports the resulting distribution: "lick | 78.1% no lick, 21.9% lick | REASONABLE" (the human reference, which applies no sensor correction, gets 23.0% lick).

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same `si:ei` slice as the neural data; no additional alignment.

ii.
```python
        trial_lick = lick_binary[si:ei]
        ...
        lick_out = (trial_lick > 0).astype(np.int64)
```

iii. Same rationale as 7-d/8-d.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. From the zone letter parsed out of the session `identifier`/scene name (with the trial-30 switch rule), i.e. the same source as 7-a. The `reward_zone` behaviour time series is not used.

ii.
```python
    rz_labels, rz_coords, env_per_trial = get_reward_zone_labels(scene, n_trials)
```
```python
        labels[:change_trial] = from_zone
        labels[change_trial:] = to_zone
```

iii. See 7-a: the zone identity comes from the scene name, whose zone assignment the AI checked against `reward_zone`-entry positions in one session (trajectory step 65) and whose bin fractions it checked globally in Step 10: "reward_zone_location | A=32.9%, B=33.7%, C=33.5% | EXCELLENT (balanced)".

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. A letter→integer map (A→0, B→1, C→2), broadcast constant across the trial's timepoints as `output[4]`.

ii.
```python
        zone_map = {'A': 0, 'B': 1, 'C': 2}
        rz_loc = zone_map.get(rz_labels[t], 0)
        ...
            np.full((1, n_tp), rz_loc, dtype=np.int64),
```
```python
            ['A', 'B', 'C'],
```

iii. Encoding follows the Decoder Task ("0 = A, 1 = B, 2 = C"). The `.get(..., 0)` default silently maps an unrecognised label to A; in practice the parser always yields A/B/C.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. From the sparse `Reward` event timestamps, snapped to the nearest behaviour timestamp (the same `reward_frames` vector used for the previous-trial-outcome input).

ii.
```python
    for rt in reward_timestamps:
        frame_idx = np.argmin(np.abs(behav_timestamps - rt))
        reward_frames[frame_idx] = 1.0
```

iii. Step 2: "`Reward/data` and `Reward/timestamps`: Event-based (sparse), not frame-aligned"; Step 13 design decision #4: "Reward determined from sparse timestamp matching to frame times".

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. A trial is labelled 1 if any reward frame falls within `[si, ei)` (trial start to teleport), else 0; the scalar is broadcast constant across the trial as `output[5]`.

ii.
```python
    trial_rewarded = np.zeros(n_trials, dtype=np.int64)
    for t in range(n_trials):
        si = trial_start_inds[t]; ei = teleport_inds[t]
        if np.any(reward_frames[si:ei] > 0):
            trial_rewarded[t] = 1
```
```python
        rew_outcome = int(trial_rewarded[t])
        ...
            np.full((1, n_tp), rew_outcome, dtype=np.int64),
```

iii. Step 3 cross-check: "Reward omission rate | ~15% | 15.7% | YES"; Step 10: "reward_outcome | 84.3% reward, 15.7% no reward | MATCHES paper (~15% omission)". The realised fraction (0.157177) is identical to the human reference's.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several defensive cases:
- **Neural/behaviour length mismatch**: all neural and behaviour arrays are truncated to the shorter length with a printed note (affected 10 multi-plane sessions, off by one).
- **Teleport/trial-start count mismatch**: the trial ends are re-derived by taking the first teleport sample after each trial start, and the trial list is truncated to the matched count.
- **Degenerate trials/sessions**: trials with `< 2` samples are skipped; sessions with `< 2` trials are dropped.
- **Stuck lick sensor**: trials with >30% of frames at lick count > 2 have their lick trace zeroed.
- **Zero fluorescence median**: guarded to 1 before dividing, to avoid divide-by-zero in the interneuron dF/F proxy.
- **NaNs in neural data**: replaced with 0 per trial.
- **Unknown zone label**: `zone_map.get(label, 0)` defaults to A.

ii.
```python
    if n_timepoints != n_behav:
        min_len = min(n_timepoints, n_behav)
        print(f"  NOTE: Aligned neural ({n_timepoints}) and behavior ({n_behav}) to {min_len} timepoints")
        deconv = deconv[:min_len]; ...; n_timepoints = min_len
```
```python
    f_median[f_median == 0] = 1
```
```python
        trial_neural = np.nan_to_num(trial_neural, nan=0.0)
```
```python
    if valid_mask.sum() > 100:   # skip interneuron test if too little running data
```

iii. Step 4 "Discrepancies Found and Resolved" lists the multi-plane concatenation, the behaviour/neural length mismatch ("Off-by-one in some multi-plane sessions, truncate to min length") and the sparse reward field. The length mismatch was discovered as a crash during the first full run (`IndexError: boolean index did not match indexed array ... 22791 vs 22790`) and fixed by moving the truncation before the interneuron computation; Step 13 records it as "Bug Fixed This Session".

*Reviewer note:* unlike the human reference, there is no assertion that the different behaviour series share timestamps, and no check that reward events snap to within half a time bin — the nearest-frame `argmin` will silently assign an out-of-range reward timestamp to the first or last frame.

## 13-a. What are the most time-consuming steps of the code?

i. The AI instruments per-session timing and prints it in the conversion log (0.6 s for the smallest single-plane session up to ~7 s for the largest two-plane session), and reports 762.5 s total for the full 152-session conversion. It identified the interneuron speed-correlation as the bottleneck for large sessions and vectorised it. The dominant remaining costs are HDF5 reads — the code reads three full (n_timepoints × n_ROIs) arrays per plane (`Deconvolved`, `Fluorescence`, `Neuropil`) — and pickling the 9.4 GB output at the end.

ii.
```python
    t0 = time.time()
    ...
    t1 = time.time()
    print(f"  {os.path.basename(filepath)}: {n_final} neurons (...), time={t1-t0:.1f}s")
```
```python
    t_end = time.time()
    print(f"Saved {args.output} ({file_size:.1f} MB)")
    print(f"Total time: {t_end - t_start:.1f}s")
```

iii. Trajectory step 89: "The sample took 10.4s for 2 sessions ... I estimate: average ~4s/session * 152 = ~10 minutes. The interneuron correlation check is the bottleneck for large sessions. Let me optimize it and run the full conversion." Step 9 of CONVERSION_NOTES.md: "Total processing time: 762.5 seconds", within the instructions' 15-minute budget.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI vectorised the one loop it identified as costly (the per-cell speed correlation, rewritten as a z-scored matrix product). Loops that remain and could be vectorised:
- `cell_mask_concat` is built with a Python list comprehension over every ROI, where `iscell[concat_to_iscell, 0] == 1` would do.
- the `for i, col_idx in enumerate(accepted_cols)` loop that writes the interneuron mask, replaceable by a boolean assignment.
- the reward-event loop calling `np.argmin(np.abs(behav_timestamps - rt))` per event, which is O(n_events × T) and could be a single `np.searchsorted`.
- the three separate per-trial loops (reward outcome, lick correction, previous outcome) plus the main segmentation loop, which could be merged or computed with `np.add.reduceat`-style segment reductions.

ii.
```python
    cell_mask_concat = np.array([iscell[concat_to_iscell[i], 0] == 1 for i in range(n_rois_concat)])
```
```python
    for rt in reward_timestamps:
        frame_idx = np.argmin(np.abs(behav_timestamps - rt))
        reward_frames[frame_idx] = 1.0
```
```python
            for i, col_idx in enumerate(accepted_cols):
                if corrs[i] > 0.5:
                    interneuron_mask[col_idx] = True
```

iii. The AI documents only the one optimisation it made (trajectory step 91: "Let me vectorize the interneuron correlation computation"; Step 6–8 notes: "Interneuron exclusion: vectorized z-scored dot product for speed-dFF correlation"). The remaining loops are short relative to I/O, so they were left alone; the per-trial loop is the natural structure for variable-length trials.

## 13-c. What processing does the code repeat multiple times?

i. Each NWB file is opened and read exactly once — there is no second survey pass. Repetition that does occur is small:
- lick is binarised twice (`lick_binary[lick_binary > 0] = 1`, then `(trial_lick > 0)` per trial);
- the trial `si`/`ei` boundaries are recomputed in four separate loops (reward, lick, plotting, main segmentation);
- `trial_env = env_per_trial.copy()` duplicates an array that is never modified;
- `concat_to_iscell` re-walks the plane list that was already walked when loading the data.

ii.
```python
    lick_binary = lick.copy()
    lick_binary[lick_binary > 0] = 1
    ...
        lick_out = (trial_lick > 0).astype(np.int64)
```
```python
    trial_env = env_per_trial.copy()
```
```python
        concat_to_iscell = []
        for plane in planes:
            plane_num = int(plane.replace('plane', ''))
            concat_to_iscell.append(plane_offsets[plane_num])
```

iii. Not discussed in CONVERSION_NOTES.md. The single-pass design is implied by the Step 6 instruction to "avoid unnecessary file I/O", and the conversion completes in 762 s.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Three items:
- The largest one: `Fluorescence` and `Neuropil` are read in full for every plane of every session — doubling/tripling the HDF5 read volume — and `dff_simple` is computed for **all** ROIs (including the rejected non-cells) even though only the `iscell`-accepted columns are ever correlated with speed, and the resulting dF/F never enters the saved data. Given the filter removes ~5 cells in total, nearly all of this work is discarded.
- `trial_input_pt` is assembled and never used (the per-trial inputs are re-created inline in the `np.vstack`).
- `plane_cell_offset = 0` is initialised and never read; `environment`, `speed` timestamps etc. are loaded and (for `environment`) never used, since environment comes from the scene name.

ii.
```python
    f_corrected = fluorescence - 0.7 * neuropil_data
    f_median = np.median(f_corrected, axis=0, keepdims=True)
    dff_simple = (f_corrected - f_median) / np.abs(f_median)   # computed for all ROIs
    ...
            dff_accepted = dff_simple[valid_mask][:, accepted_cols]   # only accepted columns used
```
```python
        trial_input_pt = np.array([env_type, trial_num, prev_out], dtype=np.float32)  # (3,)
        # ... never used
        trial_input = np.vstack([
            trial_input_tv,
            np.full((1, n_tp), env_type, dtype=np.float32), ...])
```
```python
        environment = f['processing/behavior/BehavioralTimeSeries/environment/data'][:]
        ...
    # Use env_per_trial from scene parsing (more reliable for cross-env switches)
    trial_env = env_per_trial.copy()
```

iii. No justification is offered; these are leftovers from the development iterations (the `trial_input_pt`/`trial_input_tv` pair from the initial attempt at mixed per-trial/time-varying inputs, and the `environment` read from before the switch to scene-name parsing). The F/Fneu read is genuinely needed for the interneuron filter as designed, but only for the accepted columns.
