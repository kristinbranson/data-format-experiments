# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. <Decisions>

The AI bypassed the AllenSDK entirely and read the released NWB (HDF5) files directly with `h5py`. Data discovery is done from the project metadata CSV: `project_metadata/ophys_experiment_table.csv` is read with pandas, then intersected with the set of experiment IDs that actually have an NWB file on disk in `data/visual-behavior-ophys-1.1.0/behavior_ophys_experiments/` (parsed from the filenames). That intersection (284 experiments) is then filtered to remove any `session_type` containing "passive" (leaving 202 experiments) and sorted by `ophys_experiment_id` for reproducibility.

Notably, the AI applied **no `project_code` filter**. The downloaded set contains 239 `VisualBehavior` experiments plus 45 `VisualBehaviorMultiscope` experiments; after the passive filter, 168 `VisualBehavior` and 34 `VisualBehaviorMultiscope` experiments are retained. Each retained experiment is then opened once by `load_nwb_data()`, which pulls dF/F traces + timestamps, running speed + timestamps, eye tracking (pupil area, timestamps, `likely_blink`), the `intervals/trials` table, and the `intervals/Natural_Images_*_presentations` stimulus table. A separate earlier pass (`get_all_image_names`) opens every NWB file a second time just to collect the union of image names.

ii. <Code snippets>

```python
def load_experiment_table():
    """Load the ophys experiment table and filter to active sessions with downloaded NWB files."""
    exp_table = pd.read_csv(os.path.join(METADATA_DIR, 'ophys_experiment_table.csv'))

    # Get list of downloaded NWB experiment IDs
    nwb_files = glob.glob(os.path.join(NWB_DIR, '*.nwb'))
    downloaded_ids = set()
    for f in nwb_files:
        eid = int(os.path.basename(f).replace('behavior_ophys_experiment_', '').replace('.nwb', ''))
        downloaded_ids.add(eid)

    # Filter to downloaded experiments
    exp_table = exp_table[exp_table['ophys_experiment_id'].isin(downloaded_ids)].copy()

    # Filter to active sessions only (exclude passive)
    exp_table = exp_table[~exp_table['session_type'].str.contains('passive', case=False)].copy()

    # Sort by experiment ID for reproducibility
    exp_table = exp_table.sort_values('ophys_experiment_id').reset_index(drop=True)

    return exp_table
```

```python
def load_nwb_data(nwb_path):
    """Load all relevant data from a single NWB file."""
    data = {}
    with h5py.File(nwb_path, 'r') as f:
        # Ophys timestamps
        data['ophys_timestamps'] = f['processing']['ophys']['dff']['traces']['timestamps'][:]

        # DFF traces: shape (n_frames, n_cells) -> transpose to (n_cells, n_frames)
        dff_raw = f['processing']['ophys']['dff']['traces']['data'][:]
        data['dff_traces'] = dff_raw.T  # (n_cells, n_frames)
        ...
        data['running_timestamps'] = f['processing']['running']['speed']['timestamps'][:]
        data['running_speed'] = f['processing']['running']['speed']['data'][:]
        ...
        trials = f['intervals']['trials']
        trial_data = {}
        for key in ['start_time', 'stop_time', 'go', 'catch', 'aborted', 'auto_rewarded',
                     'hit', 'miss', 'false_alarm', 'correct_reject', 'change_time',
                     'initial_image_name', 'change_image_name', 'is_change']:
            if key in trials:
                trial_data[key] = trials[key][:]
        data['trials'] = trial_data
```

```python
for idx, (_, row) in enumerate(exp_table.iterrows()):
    exp_id = row['ophys_experiment_id']
    result = process_experiment(
        exp_id, row, image_names_list, outcome_names,
        show_processing=args.show_processing
    )
```

iii. <Justification>

From CONVERSION_NOTES.md Step 6: *"Loads NWB files directly via h5py (fast, no AllenSDK overhead)"*. Step 1 documents that the AI read the SDK code and concluded dF/F and events are already pre-computed in the NWB files, so the SDK adds no processing beyond file access. Step 10 Check 3 asserts direct h5py reads are *"Equivalent"* to `BehaviorOphysExperiment.from_nwb()`. Passive exclusion is justified in Step 2 (*"We should only use active sessions for the decoder task"*) and Step 4 (*"Passive sessions ... Exclude passive sessions"*). The AI explicitly noted in Step 4 that the download contains a mix of VisualBehavior and VisualBehaviorMultiscope, and chose to keep both: *"We have a mix of VisualBehavior (single-plane) and VisualBehaviorMultiscope (multi-plane) data ... Need to handle both ~31 Hz and ~11 Hz ophys frame rates."*

## 1-b. How are the data split into subjects?

i. <Decisions>

Subjects are the unique `mouse_id` values from the experiment table. The AI does not pre-enumerate the mouse list; it builds `subjects`/`subject_idx` incrementally as experiments are processed, registering a new subject index the first time a `mouse_id` is seen. `mouse_id` is cast to `str`. Result: 38 subjects (37 from the VisualBehavior project + 1 extra mouse, 457841, contributed only by the Multiscope experiments).

ii. <Code snippets>

```python
    brain_region = exp_row['targeted_structure']
    mouse_id = str(exp_row['mouse_id'])
```

```python
    subject_map = {}  # mouse_id -> index
    ...
        # Map subject
        mouse_id = result['mouse_id']
        if mouse_id not in subject_map:
            subject_map[mouse_id] = len(all_subjects)
            all_subjects.append(mouse_id)
        ...
        all_subject_idx.append(subject_map[mouse_id])
```

```python
    'subjects': all_subjects,
    'subject_idx': np.array(all_subject_idx, dtype=np.int64),
```

iii. <Justification>

CONVERSION_NOTES.md Step 5 variable mapping: *"`mouse_id` → `subjects`, `subject_idx` — Map unique mice to indices"*. Step 2 records 38 subjects in the downloaded subset and Step 4 reconciles this against the 82 mice in the full published dataset (*"Only subset downloaded; proceed with 38"*). No further rationale was given; `mouse_id` is the obvious canonical animal identifier.

## 1-c. How are the data split into sessions?

i. <Decisions>

The AI defines **one output "session" = one ophys experiment (one imaging plane)**. There is no grouping by `ophys_session_id`; the experiment table is simply iterated row by row and each row appends one entry to `neural`/`output`/`subject_idx`/`brain_region_idx`. Sessions are ordered by ascending `ophys_experiment_id`, not by acquisition date. The true `ophys_session_id` is preserved only as descriptive metadata in `metadata['session_info']`.

For the 168 single-plane `VisualBehavior` experiments this is equivalent to one session (1 experiment = 1 session). For the 34 `VisualBehaviorMultiscope` experiments, however, it splits each of 6 real recording sessions into 7–8 separate output "sessions" that share identical behavior/stimulus/trial data but carry different neurons. This is directly visible in the verification log, where trial counts repeat in blocks (`209, 209, 209, 209, 209, 209, 209, ... 287 ×7, 309 ×7, 239 ×5, 196 ×5`). These Multiscope planes also run at ~11 Hz (93.2 ms bins) rather than ~31 Hz.

Passive sessions (OPHYS_2, OPHYS_5) are excluded at the table-filtering stage. Final count: 202 sessions.

ii. <Code snippets>

```python
    # Filter to active sessions only (exclude passive)
    exp_table = exp_table[~exp_table['session_type'].str.contains('passive', case=False)].copy()
    exp_table = exp_table.sort_values('ophys_experiment_id').reset_index(drop=True)
```

```python
    for idx, (_, row) in enumerate(exp_table.iterrows()):
        exp_id = row['ophys_experiment_id']
        ...
        result = process_experiment(...)
        if result is None:
            continue
        ...
        # Neural and output data
        all_neural.append(result['neural'])
        all_output.append(result['output'])
```

```python
        session_metadata.append({
            'exp_id': result['exp_id'],
            'ophys_session_id': result['ophys_session_id'],
            'session_type': result['session_type'],
            ...
        })
```

iii. <Justification>

CONVERSION_NOTES.md Step 5, Key Decision 9: *"Multiscope handling: Each experiment (plane) is a separate 'session' in the output, since they have different neurons but share the same behavioral data."* Key Decision 10: *"Passive sessions excluded: Only use active behavior sessions."* Step 4 resolution notes: *"For Multiscope sessions, multiple experiments (planes) per session share the same trials/behavior."* The AI observed the consequence in the trajectory (step 52: *"the Multiscope sessions have mean T around 91-92 (vs ~260 for Scientifica), because Multiscope records at ~11 Hz instead of ~31 Hz"*) and in Step 10 Check 5 listed it as a known edge case rather than a problem: *"Multiscope sessions have ~11 Hz frame rate (vs ~31 Hz Scientifica) - different T per trial."*

## 1-d. How are the data split into trials?

i. <Decisions>

Trials come from the NWB `intervals/trials` table. A boolean mask selects `(go | catch) & ~aborted & ~auto_rewarded`, and `np.where` gives the valid trial indices. For each valid trial the window is the half-open interval `[start_time, stop_time)` on the ophys timebase, selected by a boolean mask over `ophys_timestamps`. Trials are therefore variable length (verification reports T from 77 to 389 frames; ~260 for 31 Hz sessions, ~90 for 11 Hz Multiscope planes). Any trial yielding fewer than 2 ophys frames is dropped.

ii. <Code snippets>

```python
def get_valid_trials(trial_data):
    """Get indices of valid trials (Go + Catch, excluding Aborted and Auto-rewarded)."""
    go = trial_data['go'].astype(bool)
    catch = trial_data['catch'].astype(bool)
    aborted = trial_data['aborted'].astype(bool)
    auto_rewarded = trial_data['auto_rewarded'].astype(bool)

    valid = (go | catch) & ~aborted & ~auto_rewarded
    return np.where(valid)[0]
```

```python
    for trial_idx in valid_trial_idx:
        t_start = trial_data['start_time'][trial_idx]
        t_stop = trial_data['stop_time'][trial_idx]

        # Get ophys frames for this trial
        frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_stop)
        trial_ts = ophys_ts[frame_mask]
        n_trial_frames = frame_mask.sum()

        if n_trial_frames < 2:
            continue

        # Neural data: dF/F for this trial
        neural = dff[:, frame_mask].astype(np.float32)  # (n_cells, n_trial_frames)
```

iii. <Justification>

CONVERSION_NOTES.md Step 5, Key Decision 3: *"Trial definition: Use `start_time` and `stop_time` from trials table for Go and Catch trials only."* Step 3 curation rules: *"Include: Go trials and Catch trials; Exclude: Aborted trials and Auto-rewarded trials"* — a direct transcription of the task instruction. Step 10 Check 3 records the trial filtering as matching the paper.

## 1-e. How are trials filtered based on quality controls?

i. <Decisions>

Four filters are applied, all inside `process_experiment`:
1. Trial-type filter: `(go | catch) & ~aborted & ~auto_rewarded` (see 1-d).
2. Pre-check: if fewer than 2 valid trials are in the table, the whole experiment is dropped with a warning and returns `None`.
3. Per-trial frame check: trials with `< 2` ophys frames in `[start_time, stop_time)` are silently skipped (this also handles trials whose window falls off the end of the recording, since the boolean mask simply returns fewer frames).
4. Post-check: if fewer than 2 trials survive processing, the whole experiment is dropped with a warning.

No behavioral-engagement / reward-rate filter, no d-prime filter, and no per-trial motion or signal-quality filter is applied. The AI also relies on the session-level QC already baked into the released dataset.

ii. <Code snippets>

```python
    # Get valid trial indices
    valid_trial_idx = get_valid_trials(trial_data)

    if len(valid_trial_idx) < 2:
        print(f"  WARNING: Only {len(valid_trial_idx)} valid trials in experiment {exp_id}")
        return None
```

```python
        if n_trial_frames < 2:
            continue
```

```python
    if len(neural_trials) < 2:
        print(f"  WARNING: Only {len(neural_trials)} valid trials after processing for experiment {exp_id}")
        return None
```

iii. <Justification>

The ≥2-trial requirement follows the target-format requirement quoted in the instructions (*"There needs to be at least two trials within each session in order to evaluate the decoder performance"*). CONVERSION_NOTES.md Step 3 records the session-level QC that the Allen pipeline already applied (*"<1000 saturated pixels, <20% photobleaching, correct targeting, <10 um z-drift, peak d-prime >= 1.0, temporal sync confirmed"*) under the heading *"Session-level QC (already applied to data in table)"*, i.e. the AI decided no further session QC was needed. Step 10 Check 5 explicitly accepts poorly-performing sessions: *"Sessions with very few valid trials (e.g., 39) are retained if >=2 trials."* In the trajectory (step 41) the AI investigated the 39-trial session and concluded *"The first session had 1078 aborted trials (mouse licked prematurely most of the time) and only 39 valid Go+Catch trials. This is legitimate behavior - the mouse was poorly performing."*

## 2-a. What variables in the raw data is the `neural` data derived from?

i. <Decisions>

`neural` is the dF/F calcium trace matrix, read directly from the NWB path `processing/ophys/dff/traces/data`, with the corresponding frame times from `processing/ophys/dff/traces/timestamps`. The stored array is `(n_frames, n_cells)` and is transposed to `(n_cells, n_frames)`. Deconvolved events (`processing/ophys/event_detection`) were explicitly considered and rejected.

ii. <Code snippets>

```python
        # Ophys timestamps
        data['ophys_timestamps'] = f['processing']['ophys']['dff']['traces']['timestamps'][:]

        # DFF traces: shape (n_frames, n_cells) -> transpose to (n_cells, n_frames)
        dff_raw = f['processing']['ophys']['dff']['traces']['data'][:]
        data['dff_traces'] = dff_raw.T  # (n_cells, n_frames)
```

```python
    ophys_ts = nwb_data['ophys_timestamps']
    dff = nwb_data['dff_traces']  # (n_cells, n_frames)
    n_cells, n_frames = dff.shape
```

iii. <Justification>

CONVERSION_NOTES.md Step 5, Key Decision 1: *"Neural data: Use dF/F traces (not events/deconvolved). dF/F is the standard calcium imaging signal."* Step 1 notes that *"dF/F is pre-computed in NWB files (no need to compute from scratch)"* and that events are also available via FastLZeroSpikeInference. Step 3 records the SDK's dF/F recipe (*"600s median filter baseline, detrending with 3.33s median filter"*), and Step 10 Check 3 confirms the pre-computed dF/F matches the reference pipeline.

## 2-b. How is the `neural` data processed?

i. <Decisions>

Essentially no processing. The dF/F array is transposed to `(n_cells, n_frames)`, sliced by the per-trial ophys frame mask, and cast to `float32`. There is no normalization, z-scoring, smoothing, detrending, baseline subtraction, or across-plane merging (planes are kept as separate sessions — see 1-c). Brain region is assigned per session from `targeted_structure` alone (`VISp` or `VISl`), and every neuron of a session gets that single region index.

ii. <Code snippets>

```python
        neural = dff[:, frame_mask].astype(np.float32)  # (n_cells, n_trial_frames)
        ...
        neural_trials.append(neural)
```

```python
        # Map brain region
        brain_region = result['brain_region']
        if brain_region not in region_map:
            region_map[brain_region] = len(all_brain_regions)
            all_brain_regions.append(brain_region)
        ...
        all_brain_region_idx.append(
            np.full(result['n_cells'], region_map[brain_region], dtype=np.int64)
        )
```

iii. <Justification>

CONVERSION_NOTES.md Step 1/Step 3 establish that the released dF/F is already fully processed by the Allen pipeline (motion correction, neuropil correction, 600 s median-filter baseline, detrending), so no additional processing is warranted. Step 10 Check 3 records *"dF/F computation | Pre-computed in NWB | Pre-computed (600s median filter + detrending) | Yes"*. The region mapping is justified in Step 5: *"`targeted_structure` → `brain_regions`, `brain_region_idx` — Map VISp, VISl to indices"*.

## 2-c. How is the `neural` data filtered based on quality controls?

i. <Decisions>

No neuron-level filtering is applied. Every ROI present in the NWB `dff` table is retained. The AI considered adding a `valid_roi == True` filter (the SDK's `exclude_invalid_rois=True` default) and decided it was unnecessary because the released NWB files already contain only valid ROIs. The only neuron-level guard is that experiments with `n_cells == 0` are dropped.

ii. <Code snippets>

```python
    if n_cells == 0:
        print(f"  WARNING: No cells in experiment {exp_id}")
        return None
```

(No `valid_roi` filtering appears anywhere in `convert_data.py`. The only segmentation read is unused:)
```python
        if 'image_segmentation' in f['processing']['ophys']:
            seg = f['processing']['ophys']['image_segmentation']
            for key in seg.keys():
                if 'id' in seg[key]:
                    data['cell_roi_ids'] = seg[key]['id'][:]
                    break
```

iii. <Justification>

CONVERSION_NOTES.md Step 1 lists `exclude_invalid_rois` in `cell_specimens.py` as the SDK's CURATION step, and Step 3 records the neuron curation rule *"`valid_roi == True` (exclude non-cell ROIs, duplicates, edge ROIs, dendrites, too small/dim)"*. Step 10 Check 3 then resolves it: *"ROI filtering | No explicit valid_roi filter | exclude_invalid_rois=True (default) | OK - all ROIs in downloaded NWB files are valid."* The trajectory (step 60) shows the AI intending to *"add the valid_roi filter for safety"*, but no such filter was added to the final script. I independently confirmed the claim: in the released NWB files every entry of `cell_specimen_table/valid_roi` is `True`, so the omission has no effect on the output.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. <Decisions>

Alignment is to the **ophys timestamps**, with each trial anchored at the trials-table `start_time`. For each trial, a boolean mask `(ophys_ts >= start_time) & (ophys_ts < stop_time)` selects the ophys frames, and the same mask is used to slice dF/F, running speed, and pupil diameter, and the same `trial_ts` is used to build the image-identity and image-change traces. No resampling to a fixed grid and no fixed pre/post window around the change is used, so trial lengths vary. `metadata['off_start']` and `metadata['off_end']` are both `None`.

ii. <Code snippets>

```python
        frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_stop)
        trial_ts = ophys_ts[frame_mask]
        n_trial_frames = frame_mask.sum()
        ...
        neural = dff[:, frame_mask].astype(np.float32)
        ...
        running_trial = running_at_ophys[frame_mask]
        ...
        pupil_trial = pupil_at_ophys[frame_mask]
```

```python
            'temporal_alignment_event': 'Aligned to ophys (2-photon imaging) timestamps. Each trial spans from trial start_time to stop_time.',
            'off_start': None,
            'off_end': None,
```

iii. <Justification>

The task instructions say *"Temporally align based on ophys timestamp"*, and CONVERSION_NOTES.md Step 5, Key Decision 4 restates this: *"Temporal alignment: Align to ophys timestamps. For each trial, extract the ophys frames between trial start_time and stop_time."* Step 10 Check 2 documents a spot-check of the neural slice against the raw NWB (`np.allclose=True, max_diff=0.0` for trial 5 of session 803736273) and of image identity at stimulus onset.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. <Decisions>

**No rebinning or resampling of the neural data is applied.** Data are kept at each experiment's native ophys frame rate. `dt` is computed per experiment as `np.median(np.diff(ophys_ts))`, and the single reported `metadata['time_bin_size']` is the **median across sessions** of those per-session values, i.e. 32.32 ms (~30.94 Hz).

This is where the decision to include Multiscope planes bites: 168 sessions are at ~32.3 ms while 34 sessions (the Multiscope planes) are at ~93.2 ms (~10.7 Hz), a ~2.9× mismatch. The reported `time_bin_size` of 32.32 ms is therefore wrong for 34 of 202 sessions, and the target-format requirement *"Time bins should be the same size for all trials and sessions"* is not met. The per-session true `dt_ms` is preserved in `metadata['session_info']`, but nothing in the top-level format signals the mixture.

ii. <Code snippets>

```python
    dt = np.median(np.diff(ophys_ts))  # time bin size
    ...
    result = {
        ...
        'dt': float(dt),
```

```python
    # Compute median time bin size across sessions
    all_dts = [m['dt_ms'] for m in session_metadata]
    median_dt = np.median(all_dts)
```

```python
            'time_bin_size': median_dt,
            ...
            'ophys_frame_rate_hz': 1000.0 / median_dt,
```

iii. <Justification>

CONVERSION_NOTES.md Step 5, Key Decision 2: *"Time bin: Use native ophys timestamps (~31 Hz for Scientifica, ~11 Hz for Multiscope). Each session's time bin is consistent within itself."* The AI was aware of the mixture — trajectory step 52: *"the Multiscope sessions have mean T around 91-92 (vs ~260 for Scientifica), because Multiscope records at ~11 Hz instead of ~31 Hz"* — and Step 10 Check 5 lists it as an accepted edge case (*"Multiscope sessions have ~11 Hz frame rate (vs ~31 Hz Scientifica) - different T per trial"*). No justification is offered for why a single `time_bin_size` value is adequate, nor for why the differing rates were not resolved by either excluding Multiscope or resampling to a common grid.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. <Decisions>

Image identity is derived from the **stimulus presentations table**, not the trials table. `load_nwb_data` searches `f['intervals']` for a key matching `Natural_Images*`/`natural_images*` and reads `start_time`, `stop_time`, `image_name`, `is_change`, `omitted`. The identity trace is built flash-by-flash from `image_name` and each flash's `[start_time, stop_time)`. Timepoints not covered by any (non-omitted) flash — the 500 ms inter-stimulus grey and the 5% omitted flashes — get an explicit extra category `'gray'`. The trials-table columns `initial_image_name` / `change_image_name` / `change_time` are read from the NWB but never used.

ii. <Code snippets>

```python
        # Stimulus presentations - find the image presentations interval
        stim_key = None
        for key in f['intervals'].keys():
            if 'Natural_Images' in key or 'natural_images' in key:
                stim_key = key
                break
        ...
        if stim_key is not None:
            stim = f['intervals'][stim_key]
            stim_data = {}
            for key in ['start_time', 'stop_time', 'image_name', 'is_change', 'omitted']:
                if key in stim:
                    stim_data[key] = stim[key][:]
            data['stimulus'] = stim_data
```

```python
    # Initialize to gray (ISI)
    gray_idx = image_names_list.index(GRAY_LABEL)
    trace = np.full(n_frames, gray_idx, dtype=np.int64)

    # Map stimulus presentations onto ophys frames
    stim_starts = stim_data['start_time']
    stim_stops = stim_data['stop_time']
    stim_names = stim_data['image_name']
```

iii. <Justification>

CONVERSION_NOTES.md Step 5, Key Decision 5: *"Image identity: Map stimulus presentations to ophys timepoints. During gray screen (ISI), use a 'gray' category."* The Step 5 mapping table lists *"`image_name` from stimulus presentations → `output[0]`: image_identity — Map to categorical integer, time-varying at ophys rate — 8 images + gray screen"*. The AI validated the choice with a distributional sanity check in Step 9/Step 10 Check 4: *"Gray screen fraction | 66.9% | 66.7% (500/750ms) | PASS"*, i.e. the grey fraction matches the known 250 ms-on / 500 ms-off flash cycle.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. <Decisions>

A single global category list is built before any conversion: `get_all_image_names()` opens every NWB file in scope, collects the union of `image_name` values from the stimulus table, drops the literal `'omitted'`, and sorts them; `'gray'` is then prepended so `gray` is always code 0. Across the full dataset this yields 17 categories (`gray` + 16 natural images from image sets A and B).

Per trial, the trace is initialised to the gray code, then for every stimulus presentation overlapping the trial window the frames satisfying `(trial_ts >= s_start) & (trial_ts < s_stop)` are overwritten with `image_names_list.index(name)`. Byte strings are decoded to UTF-8. `'omitted'` presentations are skipped (so they remain grey), and any name not in the global list is skipped. Result is a per-frame `int64` row that steps image → gray → image → ... within each trial, including the mid-trial identity change.

ii. <Code snippets>

```python
def get_all_image_names(exp_table):
    """Scan a few NWB files to collect all unique image names across sessions."""
    all_names = set()
    for _, row in exp_table.iterrows():
        eid = row['ophys_experiment_id']
        nwb_path = os.path.join(NWB_DIR, f'behavior_ophys_experiment_{eid}.nwb')
        try:
            with h5py.File(nwb_path, 'r') as f:
                for key in f['intervals'].keys():
                    if 'Natural_Images' in key or 'natural_images' in key:
                        names = f['intervals'][key]['image_name'][:]
                        for n in names:
                            if isinstance(n, bytes):
                                n = n.decode('utf-8')
                            if n != 'omitted':
                                all_names.add(n)
                        break
        except Exception as e:
            print(f"  WARNING: Could not read images from {eid}: {e}")

    return sorted(all_names)
```

```python
    all_image_names = get_all_image_names(exp_table)
    # Add gray screen label
    image_names_list = [GRAY_LABEL] + all_image_names
```

```python
    for si in range(len(stim_starts)):
        s_start = stim_starts[si]
        s_stop = stim_stops[si]

        if s_stop < trial_start:
            continue
        if s_start >= trial_stop:
            break

        name = stim_names[si]
        if isinstance(name, bytes):
            name = name.decode('utf-8')

        # Skip omitted stimuli (they're gray screen)
        if name == 'omitted':
            continue

        if name in image_names_list:
            img_idx = image_names_list.index(name)
        else:
            continue

        # Find ophys frames during this stimulus
        frame_mask = (trial_ts >= s_start) & (trial_ts < s_stop)
        trace[frame_mask] = img_idx
```

iii. <Justification>

Building the mapping globally (and sorted) is needed so image codes are comparable across sessions that use different image sets — Step 9 notes *"Images/session | 8 | 8 per session (16 total across sets) | Yes"*. Treating omitted flashes as grey follows Step 3's note that omissions are 5% of flashes and that the screen is grey during them. Step 10 Check 2 records a sanity check: *"Image identity trace | 803736273 | PASS - np.array_equal=True, verified at stimulus onset"*, and Step 7 states the processing plots show *"Image identity correctly shows gray during ISI, images during flashes."*

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. <Decisions>

Perfectly co-indexed with the neural data by construction: `build_image_identity_trace` recomputes the identical boolean mask `(ophys_ts >= trial_start) & (ophys_ts < trial_stop)` and works on the resulting `trial_ts`, so the returned trace has exactly the same length and the same frame ordering as the `neural` slice. A frame is assigned an image if its ophys timestamp falls inside that flash's `[start_time, stop_time)`; because the ophys frame period (~32 ms) is much shorter than the 250 ms flash, the worst-case labelling error is one frame at each flash boundary.

ii. <Code snippets>

```python
def build_image_identity_trace(ophys_ts, stim_data, trial_start, trial_stop, image_names_list):
    ...
    # Get ophys frame indices within trial
    trial_mask = (ophys_ts >= trial_start) & (ophys_ts < trial_stop)
    trial_ts = ophys_ts[trial_mask]
    n_frames = len(trial_ts)
```

```python
        img_trace, _ = build_image_identity_trace(
            ophys_ts, stim_data, t_start, t_stop, image_names_list
        )
        ...
        output_full = np.zeros((5, n_trial_frames), dtype=np.int64)
        output_full[0] = img_trace
```

iii. <Justification>

CONVERSION_NOTES.md Step 5, Key Decision 4 (align everything to ophys timestamps) plus the Step 5 planned sanity check *"Verify no temporal misalignment by plotting neural + stimulus for sample trials"*, delivered via `--show-processing` plots in Step 7 (*"Change events aligned to stimulus onset"*) and the Step 10 Check 2 raw-NWB spot check (*"verified at stimulus onset"*).

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. <Decisions>

From the stimulus presentations table's boolean `is_change` column together with that presentation's `start_time`. The trials-table `change_time`/`go` columns are read into memory but not used for this variable. Because the SDK sets `is_change = True` only when the displayed image actually differs from the previous flash, catch (sham-change) trials carry no `is_change` flash and therefore get an all-zero change trace — the same effective behaviour as gating on the trials-table `go` flag.

ii. <Code snippets>

```python
def build_image_change_trace(ophys_ts, stim_data, trial_start, trial_stop):
    """Build a binary trace that is 1 at the first frame after an image change."""
    ...
    stim_starts = stim_data['start_time']
    is_change = stim_data['is_change']

    for si in range(len(stim_starts)):
        if not is_change[si]:
            continue
        s_start = stim_starts[si]
```

iii. <Justification>

CONVERSION_NOTES.md Step 5 mapping table: *"`is_change` from stimulus presentations → `output[1]`: image_change — Binary, 1 at change onset frame, 0 otherwise — Time-varying"*. Step 5 planned sanity check: *"Check that image_change=1 aligns with actual change in image_identity"*, which the Step 7 processing plots were used to confirm.

## 4-b. What processing is involved in computing `output` *Image change*?

i. <Decisions>

The trace is initialised to zeros over the trial's ophys frames. For each change presentation whose `start_time` lies inside `[trial_start, trial_stop)`, `np.searchsorted(trial_ts, s_start)` gives the first ophys frame at or after the change onset and **that single frame** is set to 1. Everything else stays 0.

The consequence is an extremely sparse label: verification reports `image_change: {no_change (0.996), change (0.004)}`, i.e. ~1 positive frame in ~250 (about one frame per go trial). The human reference instead marks the whole 750 ms post-change flash + grey window (~23 frames at 31 Hz), giving 7.65% positives.

ii. <Code snippets>

```python
    trace = np.zeros(n_frames, dtype=np.int64)

    stim_starts = stim_data['start_time']
    is_change = stim_data['is_change']

    for si in range(len(stim_starts)):
        if not is_change[si]:
            continue

        s_start = stim_starts[si]
        if s_start < trial_start or s_start >= trial_stop:
            continue

        # Find the first ophys frame at or after the change onset
        frame_idx = np.searchsorted(trial_ts, s_start)
        if frame_idx < n_frames:
            trace[frame_idx] = 1

    return trace
```

iii. <Justification>

This is a literal implementation of the task instruction *"Image change, binary variable. Have value of 1 right after a change in image identity, otherwise 0."* CONVERSION_NOTES.md states the intent as *"1 at change onset frame, 0 otherwise"*. The AI noticed the resulting imbalance and explicitly defended it rather than widening the window — Step 10 Check 4: *"Image change fraction | 0.372% | ~0.3-0.4% | PASS"*; Step 12: *"image_change (1.27x): Extreme class imbalance (99.6% no_change vs 0.4% change). Balanced accuracy penalizes heavily. The decoder correctly identifies change timepoints but the signal is very sparse. Not a conversion bug."*

## 4-c. How is `output` *Image change* thresholded into categories?

i. <Decisions>

No thresholding of a continuous quantity is involved — the variable is natively binary. Categories are `['no_change', 'change']` = `[0, 1]`, declared in `output_values[1]`. The only implicit "threshold" is the temporal one described in 4-b: exactly one frame (the first frame at or after change onset) is labelled `change`.

ii. <Code snippets>

```python
    output_values = [
        image_names_list,          # image identity categories
        ['no_change', 'change'],   # image change: 0=no change, 1=change
        [f'bin_{i}' for i in range(5)],  # running speed percentile bins
        [f'bin_{i}' for i in range(5)],  # pupil diameter percentile bins
        outcome_names,             # trial outcomes
    ]
```

```python
        frame_idx = np.searchsorted(trial_ts, s_start)
        if frame_idx < n_frames:
            trace[frame_idx] = 1
```

iii. <Justification>

The instruction specifies image change as a binary variable, so no discretisation rule was needed. The AI's notes only record the category names and the 0.4% positive rate as a consistency check.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. <Decisions>

Same mechanism as image identity: `build_image_change_trace` independently recomputes the identical `(ophys_ts >= trial_start) & (ophys_ts < trial_stop)` mask, so the returned trace is frame-for-frame co-indexed with the `neural` slice and is written into row 1 of the `(5, T)` output array. `np.searchsorted` on `trial_ts` picks the first ophys frame at or after the change onset, so the label lands within one frame (~32 ms at 31 Hz, ~93 ms at 11 Hz) of the true change.

ii. <Code snippets>

```python
    trial_mask = (ophys_ts >= trial_start) & (ophys_ts < trial_stop)
    trial_ts = ophys_ts[trial_mask]
    n_frames = len(trial_ts)
    ...
        frame_idx = np.searchsorted(trial_ts, s_start)
        if frame_idx < n_frames:
            trace[frame_idx] = 1
```

```python
        change_trace = build_image_change_trace(ophys_ts, stim_data, t_start, t_stop)
        ...
        output_full[1] = change_trace
```

iii. <Justification>

Same as 3-c: everything is expressed on the ophys timebase (Step 5, Key Decision 4). Step 7 processing plots were used to confirm *"Change events aligned to stimulus onset"*, and Step 5's planned check *"Check that image_change=1 aligns with actual change in image_identity"* targets exactly this.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. <Decisions>

From the NWB running module: `processing/running/speed/data` (speed in cm/s, ~60 Hz) and `processing/running/speed/timestamps`. This is the SDK's filtered running speed (10 Hz low-pass Butterworth applied upstream by the Allen pipeline); no raw encoder signal is used.

ii. <Code snippets>

```python
        # Running speed
        data['running_timestamps'] = f['processing']['running']['speed']['timestamps'][:]
        data['running_speed'] = f['processing']['running']['speed']['data'][:]
```

```python
    running_at_ophys = interpolate_to_ophys(
        nwb_data['running_speed'], nwb_data['running_timestamps'], ophys_ts
    )
```

iii. <Justification>

CONVERSION_NOTES.md Step 2 documents the NWB location (*"Running: `processing/running/speed` (270K samples @ 60 Hz)"*) and Step 3 records that the pipeline already applies a *"10 Hz lowpass Butterworth filter"*, so the stored speed is the appropriate, already-processed signal. Step 5 mapping: *"`running_speed` → `output[2]`: running_speed_bin"*.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. <Decisions>

Two steps: (1) resample and (2) discretise.
1. `scipy.interpolate.interp1d(..., kind='linear', bounds_error=False, fill_value=np.nan)` is fit once per experiment on the full 60 Hz trace and evaluated at every ophys timestamp, producing a session-length `running_at_ophys` array. Timestamps outside the running-signal range become NaN.
2. Percentile bin edges are computed **once per experiment** over all non-NaN ophys-resampled values, then applied to each trial's slice.

No smoothing, clipping, absolute-value, or sign handling is applied; negative speeds (backwards wheel motion) are kept as-is.

ii. <Code snippets>

```python
def interpolate_to_ophys(signal, signal_ts, ophys_ts_trial):
    """Interpolate a behavioral signal to ophys timestamps using linear interpolation."""
    if len(signal) == 0 or len(ophys_ts_trial) == 0:
        return np.full(len(ophys_ts_trial), np.nan)

    f = interpolate.interp1d(signal_ts, signal, kind='linear',
                             bounds_error=False, fill_value=np.nan)
    return f(ophys_ts_trial)
```

```python
    # Compute session-wide percentile bin edges
    running_edges = compute_session_percentile_edges(running_at_ophys, n_bins=5)
```

```python
        running_trial = running_at_ophys[frame_mask]
        running_binned = apply_percentile_bins(running_trial, running_edges, n_bins=5)
```

iii. <Justification>

CONVERSION_NOTES.md Step 5, Key Decision 7: *"Running speed: Interpolate from 60 Hz to ophys timestamps using linear interpolation."* Step 10 Check 3: *"Running speed | Interpolated to ophys timestamps | SDK also interpolates | Yes."* Step 10 Check 2 records a raw-data sanity check on the binned values: *"Running speed bins | 803736273 | PASS - np.array_equal=True."*

## 5-c. How is `output` *Running speed* thresholded into categories?

i. <Decisions>

Five **equal-count percentile bins, computed independently per experiment/session**. `compute_session_percentile_edges` takes `np.percentile(valid, [0, 20, 40, 60, 80, 100])`, then replaces the outermost edges with `-inf`/`+inf` so boundary values cannot fall outside. `apply_percentile_bins` does `np.digitize(values, edges[1:-1])` clipped to `[0, 4]`; NaN values are forced to bin 0.

This differs from the human reference, which pools all trials from all sessions and computes **one global** set of bin edges. Per-session binning makes the labels session-relative (each session contributes ~20% to each bin) rather than an absolute speed scale. Pooled across the whole dataset the distribution is still near-uniform: `{bin_0 0.190, bin_1 0.199, bin_2 0.204, bin_3 0.208, bin_4 0.200}`.

ii. <Code snippets>

```python
def compute_session_percentile_edges(values, n_bins=5):
    """Compute percentile bin edges from session-wide values (excluding NaN)."""
    valid = values[~np.isnan(values)]
    if len(valid) == 0:
        return np.linspace(0, 1, n_bins + 1)

    percentiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(valid, percentiles)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def apply_percentile_bins(values, edges, n_bins=5):
    """Apply pre-computed percentile bin edges to values."""
    valid = ~np.isnan(values)
    result = np.zeros(len(values), dtype=np.int64)
    if valid.sum() > 0:
        result[valid] = np.clip(np.digitize(values[valid], edges[1:-1]), 0, n_bins - 1)
    result[~valid] = 0
    return result
```

iii. <Justification>

CONVERSION_NOTES.md Step 5, Key Decision 8: *"Percentile bins for running/pupil: Compute percentiles across the entire session (all valid timepoints), then apply per-trial. Use 5 equal bins (0-20th, 20-40th, ..., 80-100th percentile)."* This follows the instruction *"Running speed, discretized into five equal percentile bins"*. The AI verified balance in Step 12: *"running_speed: Well-balanced (19-21% per bin)."* No explicit rationale is given for choosing per-session over global edges.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. <Decisions>

Running speed is interpolated onto the **full session's** ophys timestamp vector up front, so it lives on exactly the same timebase as dF/F. Per trial, the *same* `frame_mask` object used to slice `dff` is used to slice `running_at_ophys`, making alignment exact by construction. The binned result becomes row 2 of the `(5, T)` output array.

ii. <Code snippets>

```python
    running_at_ophys = interpolate_to_ophys(
        nwb_data['running_speed'], nwb_data['running_timestamps'], ophys_ts
    )
```

```python
        frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_stop)
        ...
        neural = dff[:, frame_mask].astype(np.float32)
        ...
        running_trial = running_at_ophys[frame_mask]
        running_binned = apply_percentile_bins(running_trial, running_edges, n_bins=5)
        ...
        output_full[2] = running_binned
```

iii. <Justification>

Step 5, Key Decision 4 (align everything to ophys timestamps). The Allen pipeline hardware-synchronises all streams (Step 3: *"Synchronization: All streams synced via NI PCI-6612 at 100 kHz"*), so resampling onto the ophys clock is valid. Step 7's `--show-processing` plots overlay the raw cm/s trace against the binned trace per trial to demonstrate no misalignment.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. <Decisions>

From `acquisition/EyeTracking/pupil_tracking` — specifically the **`area`** field and its `timestamps` — together with `acquisition/EyeTracking/likely_blink/data`. The AI converts the ellipse-fit pupil area to an equivalent diameter rather than using a stored width/height field (the human reference used `pupil_width`). If the `EyeTracking` group or `pupil_tracking` is absent, `pupil_area` is set to `None`.

ii. <Code snippets>

```python
        # Eye tracking / pupil
        if 'EyeTracking' in f.get('acquisition', {}):
            et = f['acquisition']['EyeTracking']
            if 'pupil_tracking' in et:
                pt = et['pupil_tracking']
                data['pupil_area'] = pt['area']['data'][:] if 'data' in pt['area'] else pt['area'][:]
                data['pupil_timestamps'] = pt['timestamps'][:]
                data['likely_blink'] = et['likely_blink']['data'][:]
            else:
                data['pupil_area'] = None
        else:
            data['pupil_area'] = None
```

iii. <Justification>

CONVERSION_NOTES.md Step 2 documents the NWB layout (*"Eye tracking: `acquisition/EyeTracking/pupil_tracking` (area, height, width @ 30 Hz)"*, *"Blinks: `acquisition/EyeTracking/likely_blink`"*). Step 3 records the upstream processing (*"Pupil: DeepLabCut tracking, ellipse fit, blinks detected and set to NaN"*), which motivates both the use of the ellipse area and the blink masking. Step 10 Check 3 claims the area→diameter formula is *"Same formula in whitepaper"*.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. <Decisions>

Four steps:
1. Frames flagged `likely_blink` are set to NaN **in the native 30 Hz series**.
2. Area is converted to an effective diameter, `d = 2*sqrt(area/pi)`, for entries that are non-NaN and strictly positive; all others stay NaN.
3. The NaN-containing series is passed to `interp1d` and evaluated at the ophys timestamps. Because the NaNs are left *inside* the interpolator's `y`, they propagate: blink frames and the interpolation intervals adjacent to them come out NaN rather than being bridged. (The human reference instead **drops** blink rows before constructing the interpolator, so it interpolates *across* blinks.)
4. NaN values are then mapped to bin 0 by `apply_percentile_bins` (see 6-c). Since blinks are ~3.1% of eye-tracking frames, this systematically re-labels blink periods as "smallest pupil" rather than as missing.

For experiments with no eye tracking at all, `pupil_at_ophys` is an all-NaN array and the entire session's pupil output is constant bin 0.

ii. <Code snippets>

```python
    if nwb_data['pupil_area'] is not None:
        pupil_area = nwb_data['pupil_area'].copy()
        likely_blink = nwb_data['likely_blink']
        pupil_ts = nwb_data['pupil_timestamps']

        # Set blink frames to NaN
        pupil_area[likely_blink] = np.nan

        # Compute diameter from area: diameter = 2 * sqrt(area / pi)
        pupil_diameter = np.full_like(pupil_area, np.nan)
        valid_pupil = ~np.isnan(pupil_area) & (pupil_area > 0)
        pupil_diameter[valid_pupil] = 2.0 * np.sqrt(pupil_area[valid_pupil] / np.pi)

        # Interpolate to ophys timestamps
        pupil_at_ophys = interpolate_to_ophys(pupil_diameter, pupil_ts, ophys_ts)
    else:
        pupil_at_ophys = np.full(len(ophys_ts), np.nan)
```

```python
    result[~valid] = 0  # NaN gets bin 0 (lowest bin)
```

iii. <Justification>

CONVERSION_NOTES.md Step 5, Key Decision 6: *"Pupil diameter: Compute from pupil area as `2*sqrt(area/pi)`. Set blink frames to NaN, then interpolate. Discretize non-NaN values."* The NaN→bin 0 choice was deliberated in the trajectory (step 62): *"The NaN pupil -> bin 0 issue: since pupil_diameter is specified as 'discretized into five equal percentile bins', mapping NaN to bin 0 is a reasonable choice (lowest bin). A separate blink category would add a 6th class which isn't specified. I'll leave it as is since it's a valid design choice."* Step 10 Check 5 restates it: *"NaN pupil values (blinks) mapped to bin 0 - design choice, not bug (5 bins specified)."* Step 10 Check 2 records a sanity check: *"Pupil diameter bins | 775614751 | PASS - np.array_equal=True, blinks=NaN confirmed."*

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. <Decisions>

Identical machinery to running speed: five equal-count percentile bins with edges computed **per experiment** over the non-NaN ophys-resampled diameters, outer edges replaced by `±inf`, `np.digitize` clipped to `[0, 4]`, NaN → bin 0. Because ~3.1% of frames are blinks (plus 3 sessions with no eye tracking at all), bin 0 absorbs those, and the realised global distribution is somewhat less uniform than for running: `{bin_0 0.200, bin_1 0.199, bin_2 0.214, bin_3 0.221, bin_4 0.166}`.

ii. <Code snippets>

```python
    pupil_edges = compute_session_percentile_edges(pupil_at_ophys, n_bins=5)
```

```python
        pupil_trial = pupil_at_ophys[frame_mask]
        pupil_binned = apply_percentile_bins(pupil_trial, pupil_edges, n_bins=5)
        ...
        output_full[3] = pupil_binned
```

```python
def compute_session_percentile_edges(values, n_bins=5):
    valid = values[~np.isnan(values)]
    if len(valid) == 0:
        return np.linspace(0, 1, n_bins + 1)
    percentiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(valid, percentiles)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges
```

iii. <Justification>

Same as 5-c — Step 5, Key Decision 8 applies to both running and pupil. The task instruction is *"Pupil diameter, discretized into five equal percentile bins."* Step 12 acknowledges the resulting imbalance: *"pupil_diameter: Slightly unbalanced due to blink frames mapped to bin 0."*

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. <Decisions>

Exactly as for running speed: the diameter is interpolated onto the full session ophys timestamp vector before trial segmentation, and each trial is extracted with the same `frame_mask` used for `dff`, guaranteeing frame-for-frame correspondence. Row 3 of the `(5, T)` output array.

ii. <Code snippets>

```python
        pupil_at_ophys = interpolate_to_ophys(pupil_diameter, pupil_ts, ophys_ts)
```

```python
        frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_stop)
        ...
        neural = dff[:, frame_mask].astype(np.float32)
        ...
        pupil_trial = pupil_at_ophys[frame_mask]
        pupil_binned = apply_percentile_bins(pupil_trial, pupil_edges, n_bins=5)
```

iii. <Justification>

Step 5, Key Decision 4 (ophys timebase for everything); Step 3's note that all streams are hardware-synced at 100 kHz makes cross-stream interpolation valid. The `--show-processing` plots overlay the raw pupil trace and the binned trace per trial (Step 7).

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. <Decisions>

From the four mutually-exclusive boolean columns of the NWB `intervals/trials` table: `hit`, `miss`, `false_alarm`, `correct_reject`. They are tested in that fixed order and the first `True` wins; if none is set, the string `'unknown'` is returned.

ii. <Code snippets>

```python
def get_trial_outcome(trial_data, idx):
    """Get the trial outcome for a given trial index."""
    if trial_data['hit'][idx]:
        return 'hit'
    elif trial_data['miss'][idx]:
        return 'miss'
    elif trial_data['false_alarm'][idx]:
        return 'false_alarm'
    elif trial_data['correct_reject'][idx]:
        return 'correct_reject'
    else:
        return 'unknown'
```

iii. <Justification>

CONVERSION_NOTES.md Step 5 mapping: *"Trial outcome (hit/miss/FA/CR) → `output[4]`: trial_outcome — Categorical, static per trial — 4 categories."* These are the SDK's canonical change-detection outcome labels, and since aborted and auto-rewarded trials are already excluded, exactly one of the four is set for every retained trial (I verified this holds — 0 `'unknown'` cases across a 25-experiment, 8057-trial sample). Step 10 Check 2 records *"Trial outcomes (first 20 trials) | 803736273 | PASS - all match NWB hit/miss/FA/CR."*

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. <Decisions>

The outcome string is mapped to an integer via `outcome_names.index(outcome)` with the fixed order `['hit', 'miss', 'false_alarm', 'correct_reject']` → `[0, 1, 2, 3]`. Although the outcome is conceptually static per trial, the AI broadcasts the scalar across every timepoint so that the whole `output` block is a single homogeneous `(5, T)` `int64` array. A `(1,)` static array (`output_static`) is also constructed but never used — dead code left behind by the AI's deliberation about how to represent mixed static/time-varying outputs (visible in the comment block).

The fallback for an unmatched outcome is index **0**, i.e. `'hit'`, rather than a sentinel. This never triggers on the actual data but would silently mislabel rather than flag.

Realised distribution: `{hit 0.310, miss 0.565, false_alarm 0.018, correct_reject 0.107}`.

ii. <Code snippets>

```python
        # Output 4: Trial outcome (static per trial)
        outcome = get_trial_outcome(trial_data, trial_idx)
        outcome_idx = outcome_names.index(outcome) if outcome in outcome_names else 0
```

```python
        # Stack outputs
        # Time-varying outputs: (4, n_trial_frames)
        # Static output: (1,)
        output_tv = np.stack([img_trace, change_trace, running_binned, pupil_binned], axis=0).astype(np.int64)
        output_static = np.array([outcome_idx], dtype=np.int64)

        # Combine: (5, n_trial_frames) for time-varying, last row repeated
        # Actually, static outputs should be (n_output,) shape
        # ...
        output_full = np.zeros((5, n_trial_frames), dtype=np.int64)
        output_full[0] = img_trace
        output_full[1] = change_trace
        output_full[2] = running_binned
        output_full[3] = pupil_binned
        output_full[4] = outcome_idx  # broadcast scalar to all timepoints
```

```python
    outcome_names = ['hit', 'miss', 'false_alarm', 'correct_reject']
```

iii. <Justification>

Broadcasting is required by the format: the target spec allows `(n_output, n_timepoints)` or `(n_output,)` per trial, and mixing a time-varying block with a static scalar in one array forces the static value to be repeated. The AI's in-code comments show it reasoned through this explicitly. Step 9 validates the resulting distribution against expectations: *"Outcome: hit rate | ~30% typical | 30.7% | Yes"*, *"miss rate | ~57% | 56.8% | Yes"*, *"FA rate | ~2% | 1.8% | Yes"*, *"CR rate | ~11% | 10.7% | Yes"*.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. <Decisions>

The script handles several classes of problems, mostly by skipping or by defaulting:

- **Missing NWB file**: warn and return `None` (experiment dropped).
- **Zero cells**: warn and drop the experiment.
- **Missing stimulus presentations table**: a two-tier key search (`Natural_Images*`, then any key containing `presentation`); if still not found, warn and drop the experiment.
- **Missing `image_segmentation`**: guarded with `if 'image_segmentation' in ...` (the value is unused anyway).
- **Missing trials columns**: each column is copied only `if key in trials`.
- **Missing eye tracking / pupil_tracking**: `pupil_area = None` → `pupil_at_ophys` is all NaN → the whole session's pupil output is constant bin 0. This applies to exactly 3 active experiments (795953296, 806456687, 833631914), which are silently retained with a degenerate pupil label rather than dropped.
- **Blinks**: set to NaN, then → bin 0 (see 6-b).
- **Behavioral samples outside the signal's time range**: `fill_value=np.nan` → bin 0.
- **All-NaN behavioral signal**: `compute_session_percentile_edges` returns `np.linspace(0, 1, 6)` placeholder edges (which are then never meaningfully used because all values are NaN → bin 0).
- **Byte-string image names**: decoded to UTF-8; unrecognised names skipped.
- **Trials running past the end of the recording**: handled implicitly, because the boolean `frame_mask` simply yields fewer frames; `< 2` frames → trial skipped.
- **Unrecognised trial outcome**: defaults to index 0 (`'hit'`) — a silent mislabel, though it never triggers on this data.
- **Errors reading image names**: caught with a broad `try/except` and a warning.

There is **no** top-level `try/except` around `process_experiment`, so an unanticipated exception in any one experiment would abort the whole conversion (the human reference wraps each session in `try/except` and continues).

ii. <Code snippets>

```python
    nwb_path = os.path.join(NWB_DIR, f'behavior_ophys_experiment_{exp_id}.nwb')
    if not os.path.exists(nwb_path):
        print(f"  WARNING: NWB file not found for experiment {exp_id}")
        return None
    ...
    if n_cells == 0:
        print(f"  WARNING: No cells in experiment {exp_id}")
        return None
    ...
    if stim_data is None:
        print(f"  WARNING: No stimulus data in experiment {exp_id}")
        return None
```

```python
    else:
        pupil_at_ophys = np.full(len(ophys_ts), np.nan)
```

```python
def compute_session_percentile_edges(values, n_bins=5):
    valid = values[~np.isnan(values)]
    if len(valid) == 0:
        return np.linspace(0, 1, n_bins + 1)
```

```python
def interpolate_to_ophys(signal, signal_ts, ophys_ts_trial):
    if len(signal) == 0 or len(ophys_ts_trial) == 0:
        return np.full(len(ophys_ts_trial), np.nan)
    f = interpolate.interp1d(signal_ts, signal, kind='linear',
                             bounds_error=False, fill_value=np.nan)
    return f(ophys_ts_trial)
```

```python
        name = stim_names[si]
        if isinstance(name, bytes):
            name = name.decode('utf-8')
        ...
        if name in image_names_list:
            img_idx = image_names_list.index(name)
        else:
            continue
```

iii. <Justification>

CONVERSION_NOTES.md Step 10 Check 5 (Edge Cases) enumerates the accepted cases: *"Sessions with very few valid trials (e.g., 39) are retained if >=2 trials"*, *"Multiscope sessions have ~11 Hz frame rate ... different T per trial"*, *"NaN pupil values (blinks) mapped to bin 0 - design choice, not bug (5 bins specified)"*. The trajectory (step 62) gives the fuller rationale for NaN→bin 0: adding a sixth "blink/missing" class would violate the instruction's "five equal percentile bins". The missing-eye-tracking case is not called out anywhere in the notes.

## 9-a. What are the most time-consuming steps of the code?

i. <Decisions>

The AI instrumented the code with per-experiment timers and split them into `t_load` (NWB read) and `t_process` (trial segmentation + output construction), printing both for every experiment along with a running total. The measured split was ~1.7 s loading vs ~0.4 s processing per experiment, i.e. **NWB file I/O dominates** — specifically reading the full `(n_frames, n_cells)` dF/F array (~140k frames per session) plus the 270k-sample running trace and the ~4800-row stimulus table. The extra `get_all_image_names` pre-pass added ~14 s.

The full conversion took ~9 minutes for 202 experiments, inside the 15-minute budget in the instructions, so the AI did no further optimisation.

ii. <Code snippets>

```python
def process_experiment(exp_id, exp_row, image_names_list, outcome_names, show_processing=False):
    """Process a single experiment and return trial-level data."""
    t0 = time.time()
    ...
    nwb_data = load_nwb_data(nwb_path)
    t_load = time.time() - t0
    ...
    t_process = time.time() - t0 - t_load
```

```python
        print(f"  Cells: {result['n_cells']}, Trials: {result['n_trials']}, "
              f"dt: {result['dt']*1000:.1f}ms, "
              f"Time: {t_total:.1f}s (load={result['t_load']:.1f}s, process={result['t_process']:.1f}s)")
```

```python
    t_total = time.time() - t_start_total
    print(f"\nTotal conversion time: {t_total:.1f}s ({t_total/60:.1f} min)")
    print(f"Average time per session: {t_total/len(all_neural):.1f}s")
```

iii. <Justification>

CONVERSION_NOTES.md Step 7 gives the measured breakdown and extrapolation:

| Step | Time/Session | Estimated Total |
|------|-------------|----------------|
| Load NWB | ~1.7s | ~340s |
| Process trials | ~0.4s | ~80s |
| Total | ~3.7s | ~750s (~12.5 min) |

plus *"Image name collection adds ~14s overhead."* Step 9 reports the realised *"Full conversion completed in ~9 minutes"*. The choice to read NWB with `h5py` instead of the SDK was itself motivated by load speed (Step 6: *"fast, no AllenSDK overhead"*).

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. <Decisions>

The AI vectorised the behavioural resampling and the discretisation (single `interp1d` call per session, single `np.digitize` per trial) but left several loops un-vectorised, and the notes do not identify them. The main candidates:

1. **`build_image_change_trace`** — for every trial it iterates over the *entire* session stimulus table (~4800 rows) with **no early exit**, so the cost is O(n_trials × n_stim) ≈ 300 × 4800 ≈ 1.4 M Python-level iterations per session. This could be done once per session with a single `np.searchsorted(ophys_ts, stim_starts[is_change])`.
2. **`build_image_identity_trace`** — the same nested scan, restarting from row 0 for every trial; it does `break` once `s_start >= trial_stop`, but still rescans the whole prefix each time. A session-wide identity trace could be built once with two `np.searchsorted` calls on the flash start/stop times and then simply sliced per trial.
3. **`image_names_list.index(name)`** — a linear list scan inside the innermost stimulus loop; should be a dict lookup.
4. **The per-trial boolean masks** — `(ophys_ts >= t_start) & (ophys_ts < t_stop)` is an O(n_frames) scan over the full ~140k-sample timestamp vector, executed for every trial (and three times per trial, see 9-c). `np.searchsorted` would make this O(log n), as the human reference does.
5. **The outer per-experiment loop** — fully independent across experiments and therefore trivially parallelisable with `multiprocessing`, which the instructions explicitly suggested; the AI did not use it.

ii. <Code snippets>

```python
    for si in range(len(stim_starts)):
        if not is_change[si]:
            continue

        s_start = stim_starts[si]
        if s_start < trial_start or s_start >= trial_stop:
            continue
        # NOTE: no `break` -- full table scanned for every trial
```

```python
    for si in range(len(stim_starts)):
        s_start = stim_starts[si]
        s_stop = stim_stops[si]

        if s_stop < trial_start:
            continue
        if s_start >= trial_stop:
            break
        ...
        if name in image_names_list:
            img_idx = image_names_list.index(name)
```

```python
        frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_stop)
```

iii. <Justification>

CONVERSION_NOTES.md Step 6 records no inefficiency list (the template's *"Code inefficiencies identified"* field is not filled in for this script), and Step 7 concludes the runtime is acceptable (*"~12.5 min"* estimate, ~9 min realised), so no vectorisation work was undertaken. The implicit justification is that file I/O dominates (9-a): at ~0.4 s of processing per experiment against ~1.7 s of loading, vectorising these loops would cut at most ~20% of total wall-clock.

## 9-c. What processing does the code repeat multiple times?

i. <Decisions>

Several computations are performed redundantly:

1. **Every NWB file is opened twice.** `get_all_image_names()` opens all 202 files up front purely to read the `image_name` column, and then `process_experiment` → `load_nwb_data` opens each one again. (~14 s of the runtime.)
2. **The trial frame mask is recomputed three times per trial.** `process_experiment` computes `frame_mask`, then `build_image_identity_trace` recomputes the identical `trial_mask`, then `build_image_change_trace` recomputes it a third time — each a full pass over the ~140k-element `ophys_ts` array. Passing `trial_ts` down instead would eliminate two of the three.
3. **The stimulus table is rescanned per trial** in both trace builders (see 9-b), instead of being converted once per session into a frame-indexed trace.
4. **`image_names_list.index(...)`** re-scans the category list on every stimulus presentation of every trial.
5. `build_image_identity_trace` returns `trial_mask` but the caller discards it (`img_trace, _ = ...`) and uses its own copy.

ii. <Code snippets>

```python
    all_image_names = get_all_image_names(exp_table)   # opens every NWB file
    ...
    for idx, (_, row) in enumerate(exp_table.iterrows()):
        result = process_experiment(...)               # opens every NWB file again
```

```python
        frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_stop)   # pass 1
        ...
        img_trace, _ = build_image_identity_trace(
            ophys_ts, stim_data, t_start, t_stop, image_names_list)  # pass 2 inside
        change_trace = build_image_change_trace(ophys_ts, stim_data, t_start, t_stop)  # pass 3 inside
```

```python
def build_image_identity_trace(ophys_ts, stim_data, trial_start, trial_stop, image_names_list):
    trial_mask = (ophys_ts >= trial_start) & (ophys_ts < trial_stop)   # recomputed
    trial_ts = ophys_ts[trial_mask]
```

```python
def build_image_change_trace(ophys_ts, stim_data, trial_start, trial_stop):
    trial_mask = (ophys_ts >= trial_start) & (ophys_ts < trial_stop)   # recomputed again
    trial_ts = ophys_ts[trial_mask]
```

iii. <Justification>

No justification is given in CONVERSION_NOTES.md — these repetitions are not identified anywhere in the notes or the trajectory. The two trace-builder functions were written as self-contained helpers taking `(ophys_ts, stim_data, trial_start, trial_stop)`, which is a clean interface but forces each to rederive the window. The `get_all_image_names` pre-pass is necessary in principle (a global, session-independent category mapping requires knowing all image names before any trial is encoded), and its cost is explicitly budgeted in Step 7 (*"Image name collection adds ~14s overhead"*); it could have been folded into the main pass with a two-phase assembly instead.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. <Decisions>

Modest but real amounts of wasted work and dead code:

- **Unused trials columns read from disk**: `change_time`, `initial_image_name`, `change_image_name`, and the trials-table `is_change` are all loaded in `load_nwb_data` but never referenced (image identity and change are taken from the stimulus table instead).
- **Unused stimulus column**: `stim_data['omitted']` is loaded but never used (omissions are detected from the `image_name == 'omitted'` string instead).
- **Unused ROI ids**: the `image_segmentation` group is walked and `cell_roi_ids` is read, but the value is never used anywhere.
- **Dead variable**: `output_tv` and `output_static` are both constructed each trial and immediately discarded in favour of `output_full`.
- **Dead function**: `discretize_percentile()` is defined but never called (superseded by `compute_session_percentile_edges` + `apply_percentile_bins`).
- **Full-session dF/F read**: the entire `(140k, n_cells)` dF/F array is read into memory, although only the frames inside valid trials (roughly half a session) are ever written out. HDF5 hyperslab reads restricted to the trial windows would cut the dominant I/O cost.
- **Empty input arrays**: a `(0, T)` float32 array is allocated per trial for `input`, even though `input_names` is empty. Harmless but ~52k pointless allocations.
- Not strictly "unnecessary", but note the flipside: the ~8.5 GB output is dominated by storing full-length float32 dF/F per trial.

ii. <Code snippets>

```python
        for key in ['start_time', 'stop_time', 'go', 'catch', 'aborted', 'auto_rewarded',
                     'hit', 'miss', 'false_alarm', 'correct_reject', 'change_time',
                     'initial_image_name', 'change_image_name', 'is_change']:
            if key in trials:
                trial_data[key] = trials[key][:]
```

```python
            for key in ['start_time', 'stop_time', 'image_name', 'is_change', 'omitted']:
                if key in stim:
                    stim_data[key] = stim[key][:]
```

```python
        if 'image_segmentation' in f['processing']['ophys']:
            seg = f['processing']['ophys']['image_segmentation']
            for key in seg.keys():
                if 'id' in seg[key]:
                    data['cell_roi_ids'] = seg[key]['id'][:]
                    break
```

```python
        output_tv = np.stack([img_trace, change_trace, running_binned, pupil_binned], axis=0).astype(np.int64)
        output_static = np.array([outcome_idx], dtype=np.int64)
        # ... neither is used; output_full is rebuilt from scratch below
        output_full = np.zeros((5, n_trial_frames), dtype=np.int64)
```

```python
def discretize_percentile(values, n_bins=5):
    """Discretize values into n_bins equal percentile bins.
    ...
    """
    # never called
```

```python
        all_input.append([np.zeros((0, trial.shape[1]), dtype=np.float32) for trial in result['neural']])
```

iii. <Justification>

None of these are acknowledged in CONVERSION_NOTES.md; Step 10 concludes *"No critical issues found. All checks pass."* The residue is consistent with an exploratory-then-settled implementation: several fields (`change_time`, `initial_image_name`, `change_image_name`, trials `is_change`) look like leftovers from an earlier plan to derive image identity/change from the trials table (the approach the human reference took) before the AI switched to the stimulus-presentations table; `output_tv`/`output_static` are leftovers from the in-code deliberation about mixed static/time-varying output shapes, which the surviving comment block documents. The impact is small relative to the ~1.7 s/experiment dF/F read that dominates runtime.
