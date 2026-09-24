# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. All data live in `/app/data/sub-<mouse>/sub-<mouse>_ses-<day>_behavior+ophys.nwb`. The AI lists every directory in `/app/data` starting with `sub-`, then globs every `*.nwb` file inside each of them, sorts both levels, and processes each file as one session. Each file is opened once with `pynwb.NWBHDF5IO` and both the `ophys` (Fluorescence, Neuropil, ImageSegmentation/PlaneSegmentation) and `behavior` (BehavioralTimeSeries) processing modules are read from that single open handle. 152 files from 11 subjects are found and all 152 are converted (no session is skipped). `--sample` restricts processing to 2 files (the first, and the middle one) purely for testing.

ii.
```python
    data_dir = '/app/data'
    subjects = sorted([s for s in os.listdir(data_dir) if s.startswith('sub-')])

    all_nwb_files = []
    for sub in subjects:
        files = sorted(glob.glob(os.path.join(data_dir, sub, '*.nwb')))
        all_nwb_files.extend(files)

    print(f"Found {len(all_nwb_files)} NWB files from {len(subjects)} subjects")
```
```python
    with NWBHDF5IO(nwb_file, 'r') as io:
        nwb = io.read()

        # === Extract metadata ===
        subject_id = nwb.subject.subject_id  # e.g., 'm3'
        session_id = nwb.session_id  # e.g., '03'
        identifier = nwb.identifier
        scene = parse_scene_name(identifier)
```

iii. From CONVERSION_NOTES.md Step 2: "NWB files: `/app/data/sub-{mouse}/sub-{mouse}_ses-{day}_behavior+ophys.nwb`", 11 subjects, 152 sessions. The AI cross-checked this against the paper ("n = 11 mice", "14 days", m11 starting on day 3 → 152 sessions) and reports the match in Step 9's consistency table. The trajectory shows the AI first explored the reference repo (`preprocessing.py`, `utilities.py`, `behavior.py`) and the NWB layout before settling on this loader.

## 1-b. How are the data split into subjects?

i. Subject identity is taken from the NWB metadata field `nwb.subject.subject_id` (e.g. `'m11'`), not from the directory name. A `unique_subjects` list is built in first-encountered order, and `subject_idx` for each session is the index of that session's subject into that list. Because files are iterated in sorted directory order, the list ends up as `['m11','m12','m13','m14','m15','m17','m18','m19','m3','m4','m7']`.

ii.
```python
        subject_id = nwb.subject.subject_id  # e.g., 'm3'
```
```python
        # Track subjects
        subj = result['subject']
        if subj not in unique_subjects:
            unique_subjects.append(subj)
        subj_idx = unique_subjects.index(subj)
        all_subject_idx.append(subj_idx)
```
```python
        'subjects': unique_subjects,
        'subject_idx': np.array(all_subject_idx),
```

iii. CONVERSION_NOTES.md Step 2 lists "Subjects | 11 (m3, m4, m7, m11-m15, m17-m19)" and Step 9 checks this against the paper's "n = 11 mice". Using the in-file `subject_id` rather than the folder name is a redundancy check that the folder and the file metadata agree. The verification log confirms 11 subjects with 12 (m11) or 14 sessions each.

## 1-c. How are the data split into sessions?

i. One NWB file = one session. No further splitting or merging is done, and sessions are not aligned/registered across days (the paper's cross-day cell registration is not reproduced). The session label is `nwb.session_id`, and the "scene" (environment + reward-zone condition) is parsed from `nwb.identifier`.

ii.
```python
    for si, nwb_file in enumerate(nwb_files):
        print(f"\n[{si+1}/{len(nwb_files)}] {os.path.basename(nwb_file)}")
        result = process_session(nwb_file, show_processing=args.show_processing, session_idx=si)
```
```python
def parse_scene_name(identifier):
    """Extract scene name from NWB identifier.

    Example identifiers:
    /data/InVivoDA/GCAMP3/03_10_2022/Env1_LocationC_to_A
    """
    scene = identifier.split('/')[-1]
    return scene
```

iii. CONVERSION_NOTES.md Step 2: filenames contain a parsable `ses-<day>`, and Step 9 checks 152 sessions against the paper's 14 days × 11 mice (m11 = 12). The trajectory (step 40) notes that "Scene names are extractable from NWB identifier - they encode environment and reward zone location", which the AI then relies on heavily for the per-trial environment and reward-zone labels.

## 1-d. How are the data split into trials?

i. A trial runs from a frame where the `trial_start` behavioural time series equals 1, up to (but **not** including) the first subsequent frame where `teleport` equals 1. The inter-trial/teleport period (the grey tunnel, position ≈ −50) is therefore excluded. Trial ends are found by searching, for each start, the first teleport index greater than it; if a trial start has no following teleport, the end of the recording is used. The end index is additionally clipped by `min(end, dff.shape[1], len(position))` to protect against neural/behaviour length mismatches.

ii.
```python
        # === Find trial boundaries ===
        trial_start_inds = np.where(tstart == 1)[0]
        teleport_inds = np.where(teleport == 1)[0]

        n_trials = len(trial_start_inds)
        ...
        # Match trial starts with teleports
        # Each trial goes from trial_start to the next teleport
        trial_ends = np.zeros(n_trials, dtype=int)
        for i in range(n_trials):
            # Find first teleport after this trial start
            later_teleports = teleport_inds[teleport_inds > trial_start_inds[i]]
            if len(later_teleports) > 0:
                trial_ends[i] = later_teleports[0]
            else:
                # Last trial - use end of data
                trial_ends[i] = len(position) - 1
```
```python
        for i in range(n_trials):
            start = trial_start_inds[i]
            end = trial_ends[i]  # teleport index
            end = min(end, dff.shape[1], len(position))
            n_frames = end - start  # don't include teleport frame itself
```

iii. CONVERSION_NOTES.md Step 5, Key Decision 3: "Trial boundaries: trial_start to teleport (excluding teleport/ITI period)". The trajectory (step 44) documents the reasoning: "Pre-sync period: frames before first trial start have trial_num=-1, scanning=-1, pos=-500 ... Trial period: from trial_start to teleport ... Teleport period: after teleport, position goes to −50 (gray tunnel) ... the trial boundaries should be defined by trial_start indices, not by trial_num changes", because `trial number` already increments at/near the teleport. Total trials recovered = 12,216, matching the raw count exactly.

## 1-e. How are trials filtered based on quality controls?

i. Essentially no quality filtering of trials. Only two degenerate cases are dropped: (a) a trial shorter than 2 frames is skipped, and (b) a whole session is skipped if it has fewer than 2 trial starts, if fewer than 2 valid trials survive, or if the reward zone cannot be parsed from the scene name. No session, mouse, or trial was in fact excluded — all 152 sessions and all 12,216 trials are retained (verification log: min trial length = 96 frames).

ii.
```python
        if n_trials < 2:
            print(f"    WARNING: Only {n_trials} trials, skipping session")
            return None
```
```python
            if n_frames < 2:
                continue
```
```python
        if len(neural_trials) < 2:
            print(f"    WARNING: Less than 2 valid trials, skipping session")
            return None
```

iii. CONVERSION_NOTES.md Step 3, "Trial curation rules": "No explicit trial exclusion beyond standard data quality. Sessions terminated early if mouse stopped (<41 trials in some sessions)". Step 10 Check 5 adds: "Sessions with fewer trials (m4 ses-04: 41 trials) handled correctly ... Very long trials (up to 3359 frames = 216s) preserved as-is (legitimate data)". The ≥2-trial rule is there to satisfy the target-format requirement that each session have at least two trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The raw suite2p traces: `nwb.processing['ophys']['Fluorescence'][planeN].data` (raw F) and `nwb.processing['ophys']['Neuropil'][planeN].data` (Fneu), per imaging plane, with ROI curation taken from `ImageSegmentation/PlaneSegmentation['iscell']` and plane membership from `PlaneSegmentation['planeIdx']`. The NWB `Deconvolved` field is explicitly **not** used. For the two 2-plane mice (m17, m18) both planes are loaded and pooled.

ii.
```python
        ps = nwb.processing['ophys']['ImageSegmentation']['PlaneSegmentation']
        iscell = ps['iscell'][:]  # (n_rois, 2)
        plane_idx = ps['planeIdx'][:]  # (n_rois,)

        planes = sorted(list(nwb.processing['ophys']['Fluorescence'].roi_response_series.keys()))
        n_planes = len(planes)

        for plane_name in planes:
            plane_num = int(plane_name.replace('plane', ''))
            plane_mask = plane_idx == plane_num
            plane_iscell = iscell[plane_mask, 0] == 1

            F_plane = nwb.processing['ophys']['Fluorescence'][plane_name].data[:]
            Fneu_plane = nwb.processing['ophys']['Neuropil'][plane_name].data[:]

            F_list.append(F_plane[:, plane_iscell].T.astype(np.float64))
            Fneu_list.append(Fneu_plane[:, plane_iscell].T.astype(np.float64))

        F = np.concatenate(F_list, axis=0)
        Fneu = np.concatenate(Fneu_list, axis=0)
```

iii. CONVERSION_NOTES.md Step 4: "The NWB Deconvolved data is in the raw fluorescence scale (values in thousands), suggesting it was computed from raw F rather than from dFF. The reference code computes dFF from F and Fneu using a specific pipeline ... Decision: Compute dFF from raw F and Fneu following the reference code pipeline." Trajectory step 45 reaches the same conclusion: "The NWB Deconvolved data appears to be suite2p's default deconvolution applied to the raw F (not the custom dFF pipeline). This is different from what the reference code does." Trajectory step 49 documents the multi-plane handling with the paper's quote: "ROIs were identified separately per plane, but planes were pooled for all analyses."

## 2-b. How is the `neural` data processed?

i. A re-implementation of the reference `preprocessing.dff` pipeline, **stopping short of deconvolution**: (1) neuropil subtraction `F − 0.7·Fneu`; (2) per-trial (trial_start→teleport), add back `0.7 × mean(Fneu)` of that trial so the ratio is a true dF/F; (3) maximin baseline within each trial: Gaussian smooth with σ = 15 samples, then a running minimum, then a running maximum, each with a window of `int(20 × frame_rate)` samples; (4) `dFF = (F_corr − baseline)/|baseline|`; (5) per-trial Gaussian smoothing with σ = 2 samples. The resulting dF/F (float32, NaNs → 0) is what is stored in `neural`. OASIS deconvolution into "events" is **not** performed.

ii.
```python
    # Step 1: Neuropil subtraction
    f_ = F - neu_coef * Fneu
    ...
    window_size = int(20 * frame_rate)  # 20 seconds in frames (~300 frames)

    for i in range(n_trials):
        start = start_inds[i]; stop = stop_inds[i]
        if stop <= start: continue
        # Add back neuropil mean per trial (as in reference code)
        f_[:, start:stop] = f_[:, start:stop] + neu_coef * np.nanmean(
            Fneu[:, start:stop], axis=1, keepdims=True)
        # Compute baseline using maximin method
        flow[:, start:stop] = nansmooth(f_[:, start:stop], 15, axis=1)
        flow[:, start:stop] = ndimage.minimum_filter1d(flow[:, start:stop], window_size, axis=1)
        flow[:, start:stop] = ndimage.maximum_filter1d(flow[:, start:stop], window_size, axis=1)

    valid_mask = ~np.isnan(flow) & (np.abs(flow) > 0)
    dff[valid_mask] = (f_[valid_mask] - flow[valid_mask]) / np.abs(flow[valid_mask])

    for i in range(n_trials):
        ...
        dff[:, start:stop] = nansmooth(dff[:, start:stop], 2, axis=1)
```
```python
        frame_rate = nwb.processing['ophys']['Fluorescence']['plane0'].rate
        ...
        dff = compute_dff(F, Fneu, trial_start_inds, trial_ends,
                         neu_coef=0.7, frame_rate=frame_rate)
```
```python
            trial_neural = dff[:, start:end].astype(np.float32)
            trial_neural = np.nan_to_num(trial_neural, nan=0.0)
```

iii. CONVERSION_NOTES.md Step 1 records the reference pipeline exactly ("F − 0.7*Fneu → add back Fneu mean per trial → maximin baseline (smooth σ=15, min filter ~300 frames (20 s), max filter ~300 frames) → dFF = (F−baseline)/|baseline| → smooth σ=2"), and Step 10 Check 3 tabulates each step as matching the reference. For skipping deconvolution, Step 10 Check 3 says: "We use dFF as neural activity instead of deconvolved events. Rationale: The NWB deconvolved data was computed from raw F (not dFF), which doesn't match the reference code's pipeline. Computing dFF from scratch is more faithful to the reference processing." Step 4 adds "dFF preserves more information than deconvolved events". The trajectory (step 45) shows the AI knew that "the paper's GLM uses 'deconvolved calcium event time series' and place cell analyses use deconvolved activity" but still chose dF/F, and never ran OASIS.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters, both from the paper. (1) ROIs are restricted to suite2p's manual curation, `iscell[:,0] == 1`, applied per plane before concatenation. (2) Putative interneurons are removed: any surviving cell whose dF/F has Pearson r > 0.5 with the running-speed trace is dropped. Correlations are computed over the frames where dF/F is defined (i.e. in-trial frames, determined from cell 0's NaN mask) and over the common length of the neural and behavioural arrays. Across the full dataset this removed 360 cells (0.32% per session on average), leaving 138,318 neurons (154–2,328 per session).

ii.
```python
            plane_iscell = iscell[plane_mask, 0] == 1
```
```python
        # === Interneuron exclusion ===
        # Exclude cells with Pearson correlation > 0.5 between dFF and speed
        n_neural_frames = dff.shape[1]
        n_behav_frames = len(speed)
        n_common = min(n_neural_frames, n_behav_frames)

        valid_frames = ~np.isnan(dff[0, :n_common])  # frames where dFF is valid
        interneuron_mask = np.ones(n_cells, dtype=bool)  # True = keep

        if np.sum(valid_frames) > 100:
            speed_valid = speed[:n_common][valid_frames]
            for c in range(n_cells):
                dff_valid = dff[c, :n_common][valid_frames]
                if np.std(dff_valid) > 0 and np.std(speed_valid) > 0:
                    corr = np.corrcoef(dff_valid, speed_valid)[0, 1]
                    if not np.isnan(corr) and corr > 0.5:
                        interneuron_mask[c] = False

        dff = dff[interneuron_mask, :]
```

iii. CONVERSION_NOTES.md Step 3 "Neuron curation rules": "1. Manual suite2p curation: iscell[:,0] == 1. 2. Interneuron exclusion: Pearson correlation > 0.5 between dFF and running speed", cited to the paper. The trajectory (steps 41–42) records the AI initially misreading `iscell` as probabilities and then correcting: "iscell is (n_rois, 2) where column 0 is the manual curation binary label ... m11 ses-03: 349 ROIs, 155 manually curated cells", which it cross-checks against the paper's stated minimum of 155 neurons/session. Step 10 Check 4 compares the exclusion rate to the paper's "0.42 ± 0.85% of cells". No place-cell or spatial-information filter is applied ("Place cell identification is not needed for the decoder task"), and no 2 cm/s speed threshold is applied ("only used for spatial analyses ... in the paper").

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The instructions ask for alignment to trial start, which requires no extra work beyond the trial segmentation: each trial's neural matrix is `dff[:, trial_start_index : teleport_index]`, so sample 0 of every trial is the trial-start frame. Neural frames and behavioural samples share the same index grid (both at 15.5078 Hz, one behavioural sample per imaging frame), so no resampling or shifting is done. `off_start = 0.0` and `off_end = None` (variable trial length) are recorded in the metadata.

ii.
```python
            start = trial_start_inds[i]
            end = trial_ends[i]  # teleport index
            end = min(end, dff.shape[1], len(position))
            n_frames = end - start
            trial_neural = dff[:, start:end].astype(np.float32)
```
```python
            'temporal_alignment_event': 'Trial start (entry to linear track)',
            'off_start': 0.0,  # Trial starts at alignment event
            'off_end': None,  # Variable trial length
```

iii. CONVERSION_NOTES.md Step 3 "Temporal alignment: Data aligned to imaging frames at ~15.5 Hz"; Step 5 decision 3 sets trial boundaries as trial_start→teleport. Step 10 Check 2 reports the sanity check "Verified time_from_trial_start at frame 10 of trial 5: expected 0.6448 s, got 0.6448 s", i.e. the first sample of a trial is t = 0 at the trial-start frame. Step 10 Check 5 notes the 1-frame neural/behaviour length mismatch found in some sessions and the `min()` clipping used to handle it.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning, resampling, or spatial binning. Data are kept at the native imaging frame rate of 15.5078125 Hz, i.e. a bin of 64.4836 ms, identical for every trial and session. `metadata['time_bin_size']` is hard-coded as `1000.0 / 15.5078125` ms and `metadata['frame_rate']` as 15.5078125 Hz. (Note that within `process_session` the AI reads `frame_rate` from `Fluorescence['plane0'].rate`, which for the 2-plane mice m17/m18 is the 31.015625 Hz scanner rate rather than the 15.5078125 Hz per-plane rate; this variable is used for the maximin window but not for the stored bin size, and the per-frame `dt` it computes is never used.)

ii.
```python
        dt = 1.0 / frame_rate  # time per frame in seconds
```
```python
            'time_bin_size': 1000.0 / 15.5078125,  # ~64.5 ms
            ...
            'frame_rate': 15.5078125,
```

iii. CONVERSION_NOTES.md Step 5 Key Decision 4: "Time bin: Native frame rate (~64.5 ms = 1/15.5078125 s)", cross-checked in Step 3 against the paper's "~15.5 Hz" and in Step 9's consistency table ("Frame rate 15.5 Hz → 15.5078125 Hz ✓"). The paper's 10 cm spatial binning is deliberately not used because the decoder requires time-resolved data. I independently confirmed from the raw NWB files that the behavioural timestamp spacing is 64.484 ms in every one of the 152 sessions, so the stored bin size is right for all sessions including the 2-plane ones.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. The `timestamps` attribute of the `position` behavioural time series (`behavior/BehavioralTimeSeries/position.timestamps`), which is the common behavioural/imaging time base.

ii.
```python
        bts = nwb.processing['behavior']['BehavioralTimeSeries']
        ...
        timestamps = bts.time_series['position'].timestamps[:].astype(np.float64)
```

iii. CONVERSION_NOTES.md Step 5 variable-mapping table: "timestamps → input[0]: time_from_trial_start, transform t − t_start (seconds), time-varying". All behavioural series in these files share one timestamp vector, so the choice of which series to read it from is immaterial.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. Subtract the timestamp of the trial's first frame from the trial's timestamps, then cast to float32. No smoothing, clipping, or normalisation.

ii.
```python
            # Time from trial start (seconds)
            time_from_start = (timestamps[start:end] - timestamps[start]).astype(np.float32)
            ...
            input_data[0, :] = time_from_start
```

iii. CONVERSION_NOTES.md Step 10 Check 2 (input sanity check): "Verified time_from_trial_start at frame 10 of trial 5: expected 0.6448 s, got 0.6448 s ✓" (10 × 64.48 ms). The verification log reports the input range `time_from_trial_start: [0.0, 216.5]` s, consistent with the longest trial of 3,359 frames.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It is sliced with exactly the same `[start:end]` index range as the neural matrix, so the two are aligned by construction. The shared end index is clipped by `min(end, dff.shape[1], len(position))` so that neural and behavioural slices always have identical length even in the 10 sessions where the imaging array is one frame longer than the behavioural arrays.

ii.
```python
            end = min(end, dff.shape[1], len(position))
            n_frames = end - start
            trial_neural = dff[:, start:end].astype(np.float32)
            ...
            time_from_start = (timestamps[start:end] - timestamps[start]).astype(np.float32)
            ...
            input_data = np.zeros((4, n_frames), dtype=np.float32)
            input_data[0, :] = time_from_start
```

iii. CONVERSION_NOTES.md Step 6: "Handles length mismatches between neural and behavioral data"; Step 10 Check 5: "Handled length mismatch between neural and behavioral data (1 frame difference in some sessions)". The trajectory (step 56) shows the AI hit an index error on exactly this mismatch and added the clipping. I verified independently that the mismatch is +1 imaging frame in 10 m17/m18 sessions and 0 elsewhere.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. **Not** from the `environment` behavioural time series. The AI parses the "scene" string from `nwb.identifier` (e.g. `Env1_LocationB_to_A`, `Env1_B_to_Env2_C`) and derives ENV1 = 0 / ENV2 = 1 per trial from that string, using a hard-coded switch at trial index 30 for the cross-environment sessions.

ii.
```python
def get_env_per_trial(scene, n_trials, change_trial=30):
    env_labels = np.zeros(n_trials, dtype=int)

    if '_to_' in scene and 'Env2' in scene.split('_to_')[1] and 'Env1' in scene.split('_to_')[0]:
        # Cross-env switch: Env1 before, Env2 after
        env_labels[:change_trial] = 0
        env_labels[change_trial:] = 1
    elif '_to_' in scene and 'Env1' in scene.split('_to_')[1] and 'Env2' in scene.split('_to_')[0]:
        env_labels[:change_trial] = 1
        env_labels[change_trial:] = 0
    elif 'Env2' in scene:
        env_labels[:] = 1
    else:
        env_labels[:] = 0

    return env_labels
```
```python
        env_per_trial = get_env_per_trial(scene, n_trials)
        ...
            env_val = float(env_per_trial[i])
            input_data[1, :] = env_val
```

iii. CONVERSION_NOTES.md Step 5 mapping: "scene name (NWB identifier) → input[1]: environment, Env1→0, Env2→1, per-trial binary". The trajectory (step 34) shows the AI inspected the `environment` behavioural variable ("per-frame, −1 (pre-sync) or 0 (ENV1)") but then found (step 40) that the identifier string encodes environment and reward zone directly and switched to parsing it. Step 26 of the trajectory records "Each switch after 30 trials" from the paper/reference code, which is where `change_trial=30` comes from. Step 10 Check 2 claims "Verified environment labels: ENV1→0, ENV2→1, cross-env switch correct ✓" — though that check is against the scene name itself, not against the recorded `environment` variable.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. String parsing of the scene name into a per-trial 0/1 vector, then the per-trial scalar is broadcast across all timepoints of the trial (so `input[1]` is stored as a constant row of the `(4, n_timepoints)` input matrix).

ii.
```python
            input_data = np.zeros((4, n_frames), dtype=np.float32)
            input_data[0, :] = time_from_start
            input_data[1, :] = env_val
            input_data[2, :] = trial_number
            input_data[3, :] = prev_outcome
```

iii. CONVERSION_NOTES.md Step 5 Key Decision 8: "Per-trial inputs/outputs: Broadcast to all timepoints in the trial" — this keeps all four inputs in one `(n_input, n_timepoints)` array as the target format prefers. The verification log confirms `environment: [0.0, 1.0]`.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Nothing in the raw data directly: it is the 0-based loop index of the trial within the session, i.e. the index into the list of `trial_start` events. The stored `trial number` behavioural time series is loaded but never used for this (it is read into a variable that is dead code).

ii.
```python
        for i in range(n_trials):
            ...
            # Trial number
            trial_number = float(i)
            ...
            input_data[2, :] = trial_number
```
```python
        trial_num = bts.time_series['trial number'].data[:].astype(np.float64)   # loaded, never used
```

iii. The trajectory (step 44) explains why the stored variable was rejected: "trial_num at start=0.0 and at teleport=1.0 for trial 0. This means trial_num increments during the trial or at teleport. So the trial boundaries should be defined by trial_start indices, not by trial_num changes." CONVERSION_NOTES.md Step 5 maps "trial index → input[2]: trial_number, 0-indexed trial number, per-trial, continuous". The verification log shows `trial_number: [0.0, 99.0]`, consistent with the longest sessions having 100 trials.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. None beyond `float(i)` and broadcasting the constant across the trial's timepoints. Numbering restarts at 0 in each session and is not normalised or rescaled.

ii.
```python
            trial_number = float(i)
            ...
            input_data[2, :] = trial_number
```

iii. No specific justification is given in CONVERSION_NOTES.md beyond the mapping table entry; the choice follows the instruction "Trial number (continuous, per trial)". Because no trial is actually dropped in this dataset, the index is contiguous 0…n_trials−1 in every session.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. The `Reward` entry of `BehavioralTimeSeries`, specifically its **timestamps** (one entry per reward delivery, not per frame). The trial boundary times come from `timestamps[trial_start_inds]` and `timestamps[trial_ends]`.

ii.
```python
        # Reward delivery timestamps
        reward_ts_obj = bts.time_series['Reward']
        reward_timestamps = reward_ts_obj.timestamps[:]
        ...
        trial_start_times = timestamps[trial_start_inds]
        trial_end_times = timestamps[trial_ends]
        rewarded = determine_reward_per_trial(reward_timestamps, trial_start_times, trial_end_times)
```

iii. The trajectory (step 34) records the AI's investigation: "Reward: per-trial (shape matches number of rewards), timestamps indicate when reward was delivered, value is 0.004 (likely reward amount in mL)". CONVERSION_NOTES.md Step 5 maps "previous reward → input[3]: previous_trial_outcome, from reward delivery timestamps, 0=omission, 1=rewarded".

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For each trial, a per-trial binary `rewarded` flag is set to 1 if any reward timestamp falls in the closed interval `[t_start, t_end]` of that trial. `prev_reward` is then that vector shifted by one trial, with the first trial of each session set to 0. The scalar is broadcast across the trial's timepoints.

ii.
```python
def determine_reward_per_trial(reward_timestamps, trial_start_times, trial_end_times):
    n_trials = len(trial_start_times)
    rewarded = np.zeros(n_trials, dtype=int)
    for i in range(n_trials):
        t_start = trial_start_times[i]
        t_end = trial_end_times[i]
        mask = (reward_timestamps >= t_start) & (reward_timestamps <= t_end)
        if np.any(mask):
            rewarded[i] = 1
    return rewarded
```
```python
        # === Determine previous trial outcome ===
        prev_reward = np.zeros(n_trials, dtype=int)
        prev_reward[1:] = rewarded[:-1]  # First trial has no previous, default to 0
        ...
            prev_outcome = float(prev_reward[i])
            input_data[3, :] = prev_outcome
```

iii. CONVERSION_NOTES.md Step 10 Check 2: "Verified previous trial outcome: trial 0 = 0, subsequent trials match previous reward ✓". Setting trial 0 to 0 follows the instruction's binary coding (omitted = 0). I reproduced this independently from the raw NWB for m11 ses-03 and the stored values match exactly for all 80 trials.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. The `position` behavioural time series, combined with reward-zone edge coordinates that come from the scene name in `nwb.identifier` (not from the `reward_zone` / `reward zone start` time series). Zone coordinates are hard-coded as A = [80, 130], B = [200, 250], C = [320, 370] cm, and per-trial zone identity is `label1` for trials 0–29 and `label2` for trials ≥ 30 on switch days.

ii.
```python
def get_reward_zone_info(scene):
    rz_dict = {'A': [80, 130], 'B': [200, 250], 'C': [320, 370]}
    loc_to_label = {'LocationA': 'A', 'LocationB': 'B', 'LocationC': 'C'}

    if '_to_' in scene:
        if 'Env' in scene.split('_to_')[1] and 'Location' not in scene.split('_to_')[1].split('Env')[0]:
            parts = scene.split('_to_')
            before_loc = parts[0].split('_')[-1]
            after_loc = parts[1].split('_')[-1]
        else:
            parts = scene.split('_to_')
            for loc_name, label in loc_to_label.items():
                if loc_name in parts[0]:
                    before_loc = label; break
            after_loc = parts[1]
        return [(rz_dict[before_loc], before_loc, 30), (rz_dict[after_loc], after_loc, None)]
    else:
        for loc_name, label in loc_to_label.items():
            if loc_name in scene:
                return [(rz_dict[label], label, None)]
        return [(None, None, None)]
```
```python
        rz_labels, rz_coords = get_reward_zone_label_per_trial(scene, n_trials)
        ...
            rz_start_cm = rz_coords[i, 0]
            rz_end_cm = rz_coords[i, 1]
            dist_to_rz = compute_distance_to_reward_zone(trial_pos, rz_start_cm, rz_end_cm)
```

iii. CONVERSION_NOTES.md Step 1: "Reward zones: A=[80,130], B=[200,250], C=[320,370] cm (mapped from X, Y, Z in code)" and "Switch after trial 30 (change_trial=30)", both taken from the reference repo's `behavior.get_reward_zones` / `define_trial_subsets`; Step 3 quotes the paper: "zone A, 80–130 cm; zone B, 200–250 cm; zone C, 320–370 cm". Step 5 Key Decision 6: "Reward zone: Determined from scene name in NWB identifier, following reference code logic". Step 10 Check 2: "Verified reward zone labels: trials 0-29 → B, trials 30-79 → A (for LocationB_to_A) ✓". I independently checked the scene-derived label against the position at which the recorded `reward_zone` signal first goes positive, for every trial in all 152 sessions: 0 mismatches out of 10,394 trials that carry a reward-zone signal.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance to the nearest point of the zone: `position − rz_start` when before the zone (negative), exactly 0.0 while inside `[rz_start, rz_end]`, and `position − rz_end` when past it (positive). Computed on the raw, unsmoothed per-trial position trace, then discretised (see 7-c).

ii.
```python
def compute_distance_to_reward_zone(position, rz_start, rz_end):
    """Compute signed distance from position to nearest point in reward zone.
    Negative = before reward zone, Positive = past reward zone, 0 = inside zone.
    """
    distance = np.zeros_like(position)
    before_zone = position < rz_start
    in_zone = (position >= rz_start) & (position <= rz_end)
    after_zone = position > rz_end
    distance[before_zone] = position[before_zone] - rz_start
    distance[in_zone] = 0.0
    distance[after_zone] = position[after_zone] - rz_end
    return distance
```

iii. CONVERSION_NOTES.md Step 10 Check 2: "Verified distance to reward zone: pos=201.4 in zone [200,250] → dist=0 → bin 3 ✓". This matches the instruction's "Distance to any location in the reward zone" (hence 0 anywhere inside the 50 cm zone) and the paper's reward-relative position framing. The verification log gives a sensible distribution across all 7 classes.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Seven classes assigned by explicit boolean masks: `< −50` → 0; `[−50, −10)` → 1; `[−10, 0)` → 2; `== 0` → 3; `(0, 10]` → 4; `(10, 50]` → 5; `> 50` → 6. (Endpoints +10 and +50 are assigned to the lower of the two adjoining bins, a measure-zero difference from a `np.digitize`-based implementation.)

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
            ['< -50cm', '-50 to -10cm', '-10 to 0cm', '0cm (in zone)', '0 to +10cm', '+10 to +50cm', '> +50cm'],
```

iii. The edges are copied verbatim from the Decoder Task specification in the instructions; `output_values` names them in the same order. Verification log: `distance_to_reward_zone: {< -50cm (0.253), -50 to -10cm (0.102), -10 to 0cm (0.074), 0cm (in zone) (0.237), 0 to +10cm (0.021), +10 to +50cm (0.072), > +50cm (0.242)}` — all seven classes populated, with the "in zone" class enriched because the animals slow down and dwell there.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Same `[start:end]` slice as the neural matrix — position is sampled on the same frame grid as the imaging data, so no shifting or interpolation is applied.

ii.
```python
            trial_pos = position[start:end]
            ...
            dist_to_rz = compute_distance_to_reward_zone(trial_pos, rz_start_cm, rz_end_cm)
            dist_bins = discretize_distance(dist_to_rz)
            ...
            output_data = np.zeros((6, n_frames), dtype=np.int64)
            output_data[0, :] = dist_bins
```

iii. Same rationale as 2-d/3-c: one behavioural sample per imaging frame; the shared `end = min(end, dff.shape[1], len(position))` guarantees equal lengths. I verified bin-for-bin agreement against the raw NWB file for trials 0 and 40 of m11 ses-03.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. The `position` behavioural time series (cm along the 450 cm virtual corridor), used as-is.

ii.
```python
        position = bts.time_series['position'].data[:].astype(np.float64)
        ...
            trial_pos = position[start:end]
```

iii. CONVERSION_NOTES.md Step 2 lists `position` among the behavioural series and the trajectory (step 34) notes its range "−500 to ~450 cm", with −500/−50 being pre-sync and grey-tunnel values that lie outside the trial windows actually converted.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. None: the per-trial slice of the raw position trace is discretised directly, with no smoothing, offsetting, or clipping.

ii.
```python
            pos_bins = discretize_position(trial_pos)
            ...
            output_data[1, :] = pos_bins
```

iii. CONVERSION_NOTES.md Step 5 mapping: "position → output[1]: absolute_position, 5 bins of 90 cm, time-varying". Step 9 checks "Track length 450 cm → position range 0-450 ✓".

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Five 90 cm bins with open ends: `< 90` → 0, `[90, 180)` → 1, `[180, 270)` → 2, `[270, 360)` → 3, `≥ 360` → 4. The open first/last bins absorb the handful of samples marginally outside [0, 450].

ii.
```python
def discretize_position(position):
    bins = np.zeros(len(position), dtype=np.int64)
    bins[position < 90] = 0
    bins[(position >= 90) & (position < 180)] = 1
    bins[(position >= 180) & (position < 270)] = 2
    bins[(position >= 270) & (position < 360)] = 3
    bins[position >= 360] = 4
    return bins
```

iii. The edges come straight from the Decoder Task spec ("5 equal-sized bins spanning the 450 cm track"). Step 10 Check 2: "Verified position bin at trial start: pos=1.71 → bin 0 ✓". Verification log: `absolute_position: {0-90cm (0.211), 90-180cm (0.178), 180-270cm (0.231), 270-360cm (0.227), 360-450cm (0.154)}`, i.e. close to uniform as expected for a track traversal.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Identical `[start:end]` slice as the neural matrix; no shift or resampling.

ii.
```python
            trial_pos = position[start:end]
            ...
            output_data[1, :] = pos_bins
```

iii. Same as 7-d. I verified `np.array_equal` of the stored position bins against bins recomputed from the raw NWB for several trials of m11 ses-03.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. The `lick` behavioural time series, which holds a per-frame lick count (values 0–6+, not binary).

ii.
```python
        lick = bts.time_series['lick'].data[:].astype(np.float64)
        ...
            trial_lick = lick[start:end]
```

iii. Trajectory step 34: "lick: per-frame, cumulative lick count per frame (0-6+)". CONVERSION_NOTES.md Step 5 maps "lick → output[3]: lick, Binary (>0 → 1), time-varying".

## 9-b. What processing is involved in computing `output` *Lick*?

i. Binarisation only: any frame with a lick count > 0 becomes 1, otherwise 0. No smoothing or temporal dilation.

ii.
```python
            lick_binary = (trial_lick > 0).astype(np.int64)
            ...
            output_data[3, :] = lick_binary
```

iii. The instruction specifies "Lick, time-varying. 0 = no, 1 = yes", and the raw counts exceed 1, so thresholding at > 0 is required. Step 10 Check 2: "Verified lick at frame 10: lick=0.0 → bin 0 ✓". Verification log: `lick: {no_lick (0.770), lick (0.230)}`.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same `[start:end]` slice as the neural matrix; licks are recorded on the same frame grid as the imaging data.

ii.
```python
            trial_lick = lick[start:end]
            lick_binary = (trial_lick > 0).astype(np.int64)
```

iii. Same as 7-d/8-d. I confirmed exact equality against the raw NWB lick trace for four trials of m11 ses-03.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. The scene string in `nwb.identifier`, mapped through the same `get_reward_zone_info` / `get_reward_zone_label_per_trial` logic used for the distance output (see 7-a), then encoded A = 0, B = 1, C = 2. The `reward_zone` behavioural time series is not used.

ii.
```python
def get_reward_zone_label_per_trial(scene, n_trials, change_trial=30):
    rz_info = get_reward_zone_info(scene)
    if rz_info[0][0] is None:
        return None, None
    rz_labels = np.empty(n_trials, dtype='U1')
    rz_coords = np.zeros((n_trials, 2))
    if len(rz_info) == 1:
        coords, label, _ = rz_info[0]
        rz_labels[:] = label
        rz_coords[:] = coords
    else:
        coords1, label1, ct = rz_info[0]
        coords2, label2, _ = rz_info[1]
        ct = ct if ct is not None else change_trial
        rz_labels[:ct] = label1;  rz_labels[ct:] = label2
        rz_coords[:ct] = coords1; rz_coords[ct:] = coords2
    return rz_labels, rz_coords
```

iii. See 7-a. CONVERSION_NOTES.md Step 5 mapping: "scene name → output[4]: reward_zone_location, A=0, B=1, C=2, per-trial".

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. A per-trial letter label is turned into an integer via a dict lookup and broadcast across the trial's timepoints. Sessions whose scene name contains no `Location`/zone letter would be skipped entirely (this never triggers in this dataset).

ii.
```python
            rz_label = rz_labels[i]
            rz_loc = {'A': 0, 'B': 1, 'C': 2}.get(rz_label, 0)
            ...
            output_data[4, :] = rz_loc  # per-trial, broadcast
```
```python
        rz_labels, rz_coords = get_reward_zone_label_per_trial(scene, n_trials)
        if rz_labels is None:
            print(f"    WARNING: Could not determine reward zone, skipping session")
            return None
```

iii. Verification log: `reward_zone_location: {A (0.329), B (0.337), C (0.335)}` — a near-perfect three-way split, as expected from the counterbalanced design, which the AI cites in Step 9 as evidence the scene parsing is right. My independent check against the recorded `reward_zone` signal found 0/10,394 mismatches.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. The timestamps of the `Reward` time series in `BehavioralTimeSeries`, compared against the trial's start and end times taken from the behavioural timestamp vector.

ii.
```python
        reward_ts_obj = bts.time_series['Reward']
        reward_timestamps = reward_ts_obj.timestamps[:]
        ...
        trial_start_times = timestamps[trial_start_inds]
        trial_end_times = timestamps[trial_ends]
        rewarded = determine_reward_per_trial(reward_timestamps, trial_start_times, trial_end_times)
```

iii. CONVERSION_NOTES.md Step 5 mapping: "reward timestamps → output[5]: reward_outcome, 0=no, 1=yes, per-trial". Trajectory step 34 identifies `Reward` as an event series with its own timestamps and a 0.004 mL amount.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. 1 if at least one reward timestamp falls within `[t_start, t_end]` of the trial (t_end being the timestamp of the teleport frame), else 0; broadcast across the trial's timepoints. No per-timepoint reward event series is produced.

ii.
```python
    for i in range(n_trials):
        t_start = trial_start_times[i]
        t_end = trial_end_times[i]
        mask = (reward_timestamps >= t_start) & (reward_timestamps <= t_end)
        if np.any(mask):
            rewarded[i] = 1
```
```python
            reward_outcome = rewarded[i]
            output_data[5, :] = reward_outcome  # per-trial, broadcast
```

iii. CONVERSION_NOTES.md Step 10 Check 2: "Verified reward outcome matches reward delivery timestamps ✓"; Step 9 consistency table: "Omission rate ~15% (paper) vs 15.3% (converted) ✓". Verification log: `reward_outcome: {no_reward (0.157), reward (0.843)}`. I reproduced the per-trial reward vector from the raw NWB for all 80 trials of m11 ses-03 and it matches exactly.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several defensive cases:
- **Neural/behaviour length mismatch** (10 sessions where the imaging array is 1 frame longer): every trial's end index is clipped to `min(teleport_idx, n_neural_frames, n_position_samples)`; the interneuron correlation is computed over `n_common = min(n_neural, n_behav)` frames.
- **Trial start with no following teleport**: the trial end falls back to the last sample of the session.
- **Degenerate trials/sessions**: trials with < 2 frames are skipped; a session is dropped if it has < 2 trial starts or < 2 surviving trials.
- **Undefined dF/F** (baseline NaN or exactly 0, and any frame outside the maximin windows): masked out of the dF/F computation, then replaced by 0 in the stored neural matrix.
- **Unparseable reward zone** (e.g. a pure training session): the session is skipped.
- **First trial of a session** has no previous trial, so previous outcome is set to 0.
- NaNs inside the smoothing kernels are handled by a weight-renormalising `nansmooth`.

ii.
```python
        n_common = min(n_neural_frames, n_behav_frames)
        valid_frames = ~np.isnan(dff[0, :n_common])
```
```python
            end = min(end, dff.shape[1], len(position))
            n_frames = end - start
            if n_frames < 2:
                continue
```
```python
    valid_mask = ~np.isnan(flow) & (np.abs(flow) > 0)
    dff[valid_mask] = (f_[valid_mask] - flow[valid_mask]) / np.abs(flow[valid_mask])
```
```python
            trial_neural = np.nan_to_num(trial_neural, nan=0.0)
```
```python
    nan_mask = np.isnan(data)
    data_filled = np.copy(data); data_filled[nan_mask] = 0
    smoothed = ndimage.gaussian_filter1d(data_filled, sigma=float(sigma_val), axis=axis)
    weight = np.ones_like(data, dtype=np.float64); weight[nan_mask] = 0
    weight_smooth = ndimage.gaussian_filter1d(weight, sigma=float(sigma_val), axis=axis)
    weight_smooth[weight_smooth == 0] = 1
    result = smoothed / weight_smooth
```

iii. CONVERSION_NOTES.md Step 6: "Handles length mismatches between neural and behavioral data"; Step 10 Check 5 lists the edge cases checked ("1 frame difference in some sessions", multi-plane animals, first trial previous-outcome, 41-trial and 100-trial sessions, 3,359-frame trials). The trajectory (steps 47, 48, 56) shows each of these guards being added in response to an actual crash or anomaly during development. The verification log reports zero format errors and zero warnings, and no trials or sessions were ultimately lost (12,216 trials in = 12,216 trials out).

## 13-a. What are the most time-consuming steps of the code?

i. From the timing instrumentation printed by the script itself over the full 152-session run (428 s total): dF/F computation 283 s (66%), reading the fluorescence/neuropil arrays out of HDF5 89 s (21%), interneuron correlation 24 s (6%), trial segmentation 5 s (1%). On top of that is the (untimed) pickling of the 9.4 GB output file. Per session the worst case was 8.4 s.

ii.
```python
        t1 = time.time()
        dff = compute_dff(F, Fneu, trial_start_inds, trial_ends,
                         neu_coef=0.7, frame_rate=frame_rate)
        print(f"    dFF computed: {time.time()-t1:.1f}s")
```
```python
        print(f"    Neural data loaded: {time.time()-t1:.1f}s")
        ...
        print(f"    Interneuron filtering: {time.time()-t1:.1f}s")
        ...
        print(f"    Trial segmentation: {time.time()-t1:.1f}s")
        ...
    print(f"    Total time: {time.time()-t0:.1f}s")
```

iii. CONVERSION_NOTES.md Step 6: "Code inefficiencies identified: Loading entire NWB file for each session (unavoidable with pynwb); Per-cell interneuron correlation check". Step 7 gives an extrapolated budget ("Load data ~100 s, Compute dFF ~250 s, Interneuron filter ~30 s, Trial segmentation ~15 s, Total ~7 min") and Step 7 closes with "Actual full conversion time: 428 seconds (~7.1 minutes)" — the estimate was accurate and well under the 15-minute bar in the instructions, so no further optimisation was done.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. Three loops are vectorizable but were left as Python loops:
- The **per-cell interneuron correlation loop** (`for c in range(n_cells): np.corrcoef(...)`), ~24 s in total. This is a single matrix operation (centre both arrays, one dot product); CONVERSION_NOTES.md claims this was "vectorized where possible", but the shipped code is a plain per-cell loop calling `np.corrcoef` 138,678 times.
- The **per-trial dF/F loops** in `compute_dff` (two separate passes over trials, one for the maximin baseline and one for the σ = 2 smoothing). These could be merged into one pass, and the min/max filtering could be run once over the whole session with trial boundaries enforced by NaN padding.
- The **trial-end matching loop**, which re-scans the whole `teleport_inds` array for every trial (O(n_trials × n_teleports)); `np.searchsorted` does this in one call.
- Minor: `n_rois = sum(np.sum(plane_idx == ...) for p in planes)`, computed only to be printed.

ii.
```python
            for c in range(n_cells):
                dff_valid = dff[c, :n_common][valid_frames]
                if np.std(dff_valid) > 0 and np.std(speed_valid) > 0:
                    corr = np.corrcoef(dff_valid, speed_valid)[0, 1]
```
```python
        for i in range(n_trials):
            later_teleports = teleport_inds[teleport_inds > trial_start_inds[i]]
            if len(later_teleports) > 0:
                trial_ends[i] = later_teleports[0]
```
```python
    for i in range(n_trials):
        ...
        flow[:, start:stop] = nansmooth(f_[:, start:stop], 15, axis=1)
        ...
    for i in range(n_trials):
        ...
        dff[:, start:stop] = nansmooth(dff[:, start:stop], 2, axis=1)
```

iii. CONVERSION_NOTES.md Step 6 states "Code speedups added: Vectorized discretization functions; Efficient trial boundary matching" and lists the per-cell correlation as "(vectorized where possible)". The discretisation functions are genuinely vectorized; the trial-boundary matching and the interneuron loop are not. Because the total runtime (7 min) was inside the instructions' 15-minute budget, the AI explicitly decided not to optimise further.

## 13-c. What processing does the code repeat multiple times?

i. Very little at the file level: each NWB file is opened exactly once and each array is read once, so unlike a survey-then-convert design there is no second pass over the 88 GB of source data. The repeats that do exist are small:
- `nansmooth` is called twice per trial (baseline, then dF/F) and each call recomputes `np.isnan(data)`, builds a full-size weight array, and runs a second Gaussian filter on it — doubling the smoothing cost even though, in practice, the trial slices passed to it contain no NaNs.
- The trial-end search re-scans `teleport_inds` once per trial.
- `np.isnan(flow)` is evaluated over the whole session array to build `valid_mask`, after the same NaN structure was already implied by the per-trial windows.
- In `--show-processing` mode, per-trial values already stored in the output arrays are re-extracted in list comprehensions to make the plots.

ii.
```python
    nan_mask = np.isnan(data)
    ...
    weight = np.ones_like(data, dtype=np.float64)
    weight[nan_mask] = 0
    weight_smooth = ndimage.gaussian_filter1d(weight, sigma=float(sigma_val), axis=axis)
```
```python
        result = process_session(nwb_file, show_processing=args.show_processing, session_idx=si)
```

iii. CONVERSION_NOTES.md Step 6 names the only repeat the AI considered important — "Loading entire NWB file for each session (unavoidable with pynwb)" — and treats the single-pass design as sufficient. The redundant NaN bookkeeping in `nansmooth` is a faithful port of the reference's NaN-safe smoother rather than a deliberate choice.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. A handful of small items, none of them documented by the AI:
- `dt = 1.0 / frame_rate` is computed for every session and never used (time-from-start comes from the timestamps instead).
- The `trial number` behavioural array is read out of HDF5 in full and never used.
- `n_rois` is recomputed from `planeIdx` purely to print a log line.
- `process_session` takes `show_processing` and `session_idx` arguments that it ignores; the returned `result` carries `env_per_trial`, `rz_labels` and `rewarded`, which are only consumed by the optional plotting function.
- `flow` and `dff` are allocated at full session length even though only the in-trial windows are ever filled or used, and dF/F is computed for the ~360 cells that the interneuron filter then discards (this one is unavoidable, since the filter needs dF/F).
- The output is stored as float32 dF/F at full temporal resolution, producing a 9.4 GB pickle; nothing in the pipeline downsamples or sparsifies it.

ii.
```python
        dt = 1.0 / frame_rate  # time per frame in seconds
```
```python
        trial_num = bts.time_series['trial number'].data[:].astype(np.float64)
```
```python
        n_rois = sum(np.sum(plane_idx == int(p.replace('plane', ''))) for p in planes)
        n_cells = F.shape[0]
        print(f"    ROIs: {n_rois}, Cells (iscell): {n_cells}, Planes: {n_planes}")
```
```python
def process_session(nwb_file, show_processing=False, session_idx=0):
```

iii. CONVERSION_NOTES.md does not discuss discarded computation; Step 6 only claims the script is efficient enough ("~7 minutes for full dataset"). Notably, the AI deliberately avoided the genuinely expensive unnecessary work in this domain — it does not compute place cells or spatial tuning ("Place cell identification is not needed for the decoder task", Step 10 Check 3) and does not spatially bin the data.
