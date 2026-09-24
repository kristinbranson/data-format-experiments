# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI enumerates every `*.nwb` file on disk in `/app/data/.../behavior_ophys_experiments`, parses the experiment id out of each filename, and intersects that list with the project metadata table `project_metadata/ophys_experiment_table.csv`. It then applies two filters: `behavior_type == 'active_behavior'` (drops the 71 passive-viewing experiments) and `equipment_name != 'MESO.1'` (drops the 34 Multiscope/11 Hz experiments). The surviving 168 experiments are loaded one at a time with the AllenSDK's `BehaviorOphysExperiment.from_nwb_path()`. Loading is done in a single "Pass 1" loop that extracts everything needed (events, trials, stimulus presentations, running speed, eye tracking) and caches it in memory; a second pass re-derives image indices and segments trials without touching disk again. Load failures are caught per-experiment and the session is skipped. Note the AI does *not* filter on `project_code`; in this data subset `active_behavior & non-MESO` happens to select exactly the 168 active experiments of the `VisualBehavior` project code.

ii.
```python
def get_experiment_table():
    """Get filtered experiment table for active behavior experiments."""
    nwb_files = os.listdir(NWB_DIR)
    all_exp_ids = [int(re.search(r'(\d+)', f).group(1)) for f in nwb_files if f.endswith('.nwb')]
    exp_table = pd.read_csv(os.path.join(METADATA_DIR, 'ophys_experiment_table.csv'))
    exp_table = exp_table[exp_table['ophys_experiment_id'].isin(all_exp_ids)]
    exp_table = exp_table[exp_table['behavior_type'] == 'active_behavior']
    # Exclude Multiscope (11 Hz) sessions - different frame rate than Scientifica (31 Hz)
    # The format requires consistent time bins across all sessions
    exp_table = exp_table[exp_table['equipment_name'] != 'MESO.1']
    return exp_table


def load_experiment(exp_id):
    """Load a single experiment from NWB file."""
    nwb_path = os.path.join(NWB_DIR, f'behavior_ophys_experiment_{exp_id}.nwb')
    return BehaviorOphysExperiment.from_nwb_path(nwb_path)
```
```python
    for i, exp_id in enumerate(exp_ids):
        try:
            dataset = load_experiment(exp_id)
        except Exception as e:
            print(f'  ERROR loading experiment {exp_id}: {e}')
            all_session_data.append(None)
            continue
```

iii. From CONVERSION_NOTES.md Step 1/Step 5 and the trajectory: the AI followed the provided tutorials, which load data with `BehaviorOphysExperiment.from_nwb_path()`; since the data are local NWB files there is no S3 cache to use. It filtered to `active_behavior` because "the task says to include only active behavior sessions under the 'Visual Behavior' task" (trajectory step 11) — in passive-viewing sessions the mouse does not perform the change-detection task. It excluded MESO.1 late in the process (trajectory steps 123–124) after discovering mixed 11 Hz/31 Hz frame rates, arguing that the target format requires "Time bins … the same size for all trials and sessions", that the 11 Hz sessions all came from a single mouse, and that they contained only ~10 neurons/plane on average. Loading was restructured into a two-pass scheme specifically so each NWB file is opened only once (trajectory step 69).

## 1-b. How are the data split into subjects?

i. Subjects are the unique `mouse_id` values read from each loaded experiment's `dataset.metadata['mouse_id']` (string). A dict `subject_set` maps mouse id → index in order of first appearance; `subject_idx` holds one entry per emitted session. This yields 37 subjects, matching the reference.

ii.
```python
        meta = session_data['metadata']
        mouse_id = str(meta['mouse_id'])
        if mouse_id not in subject_set:
            subject_set[mouse_id] = len(subject_set)
        all_subject_idx.append(subject_set[mouse_id])
...
    subjects = [''] * len(subject_set)
    for mouse_id, idx in subject_set.items():
        subjects[idx] = mouse_id
```

iii. CONVERSION_NOTES Step 2 identifies `mouse_id` as the animal identifier in both the experiment table and per-experiment metadata; no justification beyond "mouse_id is the subject id" is given. The AI cross-checked the count against the metadata table (38 mice available, 37 after the Multiscope-only mouse 457841 was dropped) and noted this in Step 9.

## 1-c. How are the data split into sessions?

i. Each *experiment* (one NWB file = one imaging plane) is treated as one output "session" — no grouping by `ophys_session_id`. Sessions are emitted in the order in which the experiment ids appear in the filtered metadata table (not sorted by acquisition date). Because all multi-plane (MESO.1) experiments were excluded, the 168 remaining experiments map 1:1 onto 168 distinct `ophys_session_id` values (verified against the metadata table), so in practice this is equivalent to grouping by session. `ophys_session_id` is nevertheless recorded per session in `metadata['session_info']`.

ii.
```python
    exp_ids = exp_table['ophys_experiment_id'].values
    ...
    for i, exp_id in enumerate(exp_ids):
        ...
        session_info.append({
            'exp_id': exp_id,
            'session_id': int(meta['ophys_session_id']),
            'mouse_id': mouse_id,
            'cre_line': meta['cre_line'],
            'session_type': meta['session_type'],
            'targeted_structure': region,
            'n_neurons': session_data['n_neurons'],
            'n_trials': len(neural_trials),
            'ophys_frame_rate': float(1.0 / session_data['dt']),
        })
```

iii. CONVERSION_NOTES Step 5, Key Decision 1: "**Each experiment = one session**: Each imaging plane treated as independent session." The AI had observed (Step 2) that only Multiscope sessions contain multiple experiments per session ("Experiments per session 1–7, mean 1.16"), and after excluding Multiscope every session contains exactly one plane, so the simplification is exact for the retained data. Passive-viewing sessions were excluded as described in 1-a.

## 1-d. How are the data split into trials?

i. Trials come from the SDK's built-in `dataset.trials` table. A trial is kept if it is a go **or** catch trial and is neither aborted nor auto-rewarded. The trial window is the full `[start_time, stop_time)` interval — variable length, typically ~7–12.5 s (224–388 ophys frames in session 0). Frames are selected by a boolean mask on `ophys_timestamps`. Trials yielding fewer than 5 ophys frames are dropped.

ii.
```python
    trials = dataset.trials
    valid_trials = trials[(trials['go'] | trials['catch']) & ~trials['aborted'] & ~trials['auto_rewarded']]
    if len(valid_trials) < 2:
        return None
```
```python
    for _, trial_row in valid_trials.iterrows():
        start_time = trial_row['start_time']
        stop_time = trial_row['stop_time']

        frame_mask = (ophys_ts >= start_time) & (ophys_ts < stop_time)
        frame_indices = np.where(frame_mask)[0]

        if len(frame_indices) < 5:
            continue
```

iii. CONVERSION_NOTES Step 3 ("Trial curation rules: Include Go and Catch trials, exclude Aborted and Auto-rewarded") follows the Decoder Task instruction directly. The AI verified in the trajectory (steps 17–18, 57) that every retained trial has exactly one of hit/miss/false_alarm/correct_reject, that trials contain ~9.6 stimulus flashes, and that go+catch = total − aborted − auto_rewarded. Using the full start→stop window (rather than a fixed window around the change) is what makes image identity and image change time-varying within the trial.

## 1-e. How are trials filtered based on quality controls?

i. Four filters: (1) trial type — go/catch only, aborted and auto-rewarded excluded; (2) a trial must contain ≥ 5 ophys frames; (3) a session must contain ≥ 2 valid trials (checked twice: before segmentation on the trials table, and after segmentation on the produced trial list); (4) a session must contain ≥ 2 neurons. Sessions that raise an exception on load are skipped entirely. No filtering on behavioral performance, engagement, lick contamination, or reward rate is applied. In the full run no sessions were dropped by these rules (168 in, 168 out; 43,387 trials).

ii.
```python
    if n_neurons < 2:
        return None
    ...
    valid_trials = trials[(trials['go'] | trials['catch']) & ~trials['aborted'] & ~trials['auto_rewarded']]
    if len(valid_trials) < 2:
        return None
```
```python
        if len(frame_indices) < 5:
            continue
...
        if len(neural_trials) < 2:
            print(f'  Skipping experiment {exp_id}: only {len(neural_trials)} trials after segmentation')
            continue
```

iii. The trial-type filter is taken verbatim from the instructions (aborted = mouse licked before the change, so no change was shown; auto-rewarded = free reward biases behavior). The ≥ 2 trials and ≥ 2 neurons thresholds come from the target-format requirement that "There needs to be at least two trials within each session in order to evaluate the decoder performance" (CONVERSION_NOTES Step 5). The ≥ 5 frame minimum is not explicitly justified in the notes; it guards against degenerate near-empty trial windows.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is `dataset.events['events']` — the deconvolved calcium event traces (FastLZeroSpikeInference applied to dF/F by the Allen pipeline), stacked into an (n_neurons, n_ophys_frames) float32 array. `dff_traces` is explicitly *not* used, and neither is `filtered_events` (the Gaussian-smoothed version also present in the NWB file). I verified on experiment 792815735 that the converted trial-0 matrix is `np.allclose` to the raw `events` array sliced by the trial frames (and is *not* close to dF/F).

ii.
```python
    # Neural data (events)
    events = dataset.events
    events_array = np.vstack(events['events'].values).astype(np.float32)
    n_neurons = events_array.shape[0]
```

iii. CONVERSION_NOTES Step 1 Notes: "Neural data: Use `events` (calcium events from FastLZeroSpikeInference) NOT `dff_traces`. Per methods.txt: 'For all analysis of neural data we used the detected calcium events'. Events approximate firing rate with ~200ms resolution." The whitepaper reading (trajectory steps 50–55) confirmed the pipeline ROI-detection → demixing → neuropil subtraction → dF/F → event detection, and that events use multiplicative factor 2.0 at 31 Hz.

## 2-b. How is the `neural` data processed?

i. No processing at all beyond `np.vstack` of the per-cell event arrays and a cast to float32, then per-trial column slicing. No smoothing/filtering, no normalization or z-scoring, no baseline subtraction, no temporal rebinning, and no merging across imaging planes (each experiment is a single plane). The resulting signal is extremely sparse: in experiment 792815735, 99.69 % of the raw `events` entries are exactly zero (the available `filtered_events` are 96.3 % zero). The verification log consequently reports 1,860 trials whose entire neural matrix is zero.

ii.
```python
    events_array = np.vstack(events['events'].values).astype(np.float32)
...
        # Neural
        neural_trial = events_array[:, frame_indices]
```

iii. CONVERSION_NOTES Step 3/Step 5: the Allen pipeline already performs motion correction, demixing, neuropil subtraction, dF/F and event detection, so the AI regarded the stored events as analysis-ready. On the all-zero-trial warnings, CONVERSION_NOTES Step 10 Check 1 states: "Warnings: all-zero neural data in some trials (expected for calcium events). Cannot fix: some trials genuinely have no detected calcium events." Low decoder accuracy was attributed in Step 12 to decoder-architecture differences and class imbalance rather than to the neural representation.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neuron-level quality filtering is applied: every cell in `dataset.events` (i.e. every ROI that survived the Allen pipeline's classifier-based valid-ROI filtering) is kept. The only related rule is a session-level one — an experiment with fewer than 2 neurons is dropped. Final dataset: 29,097 neurons over 168 sessions (mean 173.2, min 6, max 666).

ii.
```python
    if n_neurons < 2:
        return None
```
(no other neuron filtering exists in the script)

iii. CONVERSION_NOTES Step 3 "Neuron curation rules: ROI filtering already applied by Allen pipeline. No additional filtering needed" and Step 5 Key Decision 5: "**No neuron filtering**: ROIs already filtered by Allen pipeline". This was checked against the whitepaper's ROI-filtering section (trajectory steps 50–51), which describes the classifier that removes invalid ROIs before release, and against `cell_specimen_table`, which contains only valid cells.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to **trial start**: for each trial the ophys frames satisfying `start_time <= ophys_timestamp < stop_time` are selected, and the neural matrix is those columns of the session-wide event array. Everything else (image identity, change, running, pupil, outcome) is computed on the same ophys timebase and indexed with the identical `frame_indices`, so all streams are aligned by construction. `metadata['temporal_alignment_event'] = 'Trial start time'`, `off_start = 0.0`, `off_end = None` (variable-length trials). Maximum alignment error is one frame (~32 ms).

ii.
```python
        frame_mask = (ophys_ts >= start_time) & (ophys_ts < stop_time)
        frame_indices = np.where(frame_mask)[0]
        ...
        neural_trial = events_array[:, frame_indices]
        img_trial = image_idx_full[frame_indices]
        change_trial = change_full[frame_indices]
        running_trial = running_binned[frame_indices]
        pupil_trial = pupil_binned[frame_indices]
```

iii. The Decoder Task says "Temporally align based on ophys timestamp" and "Segment each recording session into individual trials based on how they are defined in the experiment". CONVERSION_NOTES Step 5 therefore aligns everything onto `ophys_timestamps` and slices by trial boundaries; behavioral streams are resampled onto that same timebase (see 5-b/6-b) rather than the other way around, so no interpolation of the neural data is needed. The `--show-processing` plots overlay trial start/stop/change markers on every stream to check visually for misalignment (CONVERSION_NOTES Step 7: "No temporal misalignment visible").

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning, resampling or smoothing is applied to the neural data: bins are the native ophys frames of the Scientifica rigs, nominally 31 Hz. The script computes the per-session median inter-frame interval (`dt`) and stores the derived frame rate in `session_info` (measured 30.93–30.95 Hz, i.e. ~32.32 ms), but the global `metadata['time_bin_size']` is hard-coded as `1000/31 = 32.258` ms. Consistent bin size across sessions was the explicit reason for excluding the 11 Hz Multiscope experiments. Trial lengths therefore vary (217–389 frames, mean 262).

ii.
```python
    ophys_ts = dataset.ophys_timestamps
    dt = np.median(np.diff(ophys_ts))
...
        'metadata': {
            ...
            'time_bin_size': 1000.0 / 31.0,  # ~32.3 ms
            'temporal_alignment_event': 'Trial start time',
            'off_start': 0.0,
            'off_end': None,
```

iii. CONVERSION_NOTES Step 10 Check 5 / trajectory steps 123–124: the AI discovered two frame rates in the data (31 Hz Scientifica, 11 Hz Multiscope), noted that the format spec requires "Time bins … the same size for all trials and sessions", and chose to exclude the 11 Hz sessions rather than resample, because they came from a single mouse with ~10 neurons per plane while downsampling the 31 Hz majority would throw away temporal resolution. Keeping the native frame rate avoids introducing interpolation artifacts into the event traces.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. From `dataset.stimulus_presentations` restricted to the `change_detection_behavior` stimulus block — specifically the `start_time`, `end_time` and `image_name` columns. The trials table's `initial_image_name` / `change_image_name` are *not* used. Flashes whose `image_name` is `'omitted'` (≈2.9 % of flashes, the intentional stimulus omissions) are skipped and covered by forward-fill.

ii.
```python
    sp = dataset.stimulus_presentations
    sp_active = sp[sp['stimulus_block_name'] == 'change_detection_behavior'].copy()

    # Image identity at ophys timestamps
    image_idx_full = get_image_at_ophys(sp_active, ophys_ts, image_to_idx)
```
```python
    starts = sp_active['start_time'].values
    ends = sp_active['end_time'].values
    names = sp_active['image_name'].values

    for j in range(len(starts)):
        img_name = names[j]
        if not isinstance(img_name, str) or img_name == 'omitted':
            continue
        if img_name not in image_to_idx:
            continue
        mask = (ophys_ts >= starts[j]) & (ophys_ts < ends[j])
        image_idx[mask] = image_to_idx[img_name]
```

iii. CONVERSION_NOTES Step 5 variable mapping: `stimulus_presentations.image_name → output[0]`, "Map to index, forward-fill". The AI inspected the stimulus table in the trajectory (steps 18, 57) and confirmed ~250 ms flashes with 500 ms grey gaps and ~9.6 flashes per trial, so the presentation table is the direct per-frame record of what was on the screen.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Three steps. (1) A global, sorted name→integer mapping is built from the union of image names seen across all experiments (16 names: im000…im106); each session uses only its own 8-image set. (2) A full-session integer trace is built at ophys resolution: frames inside a flash get that flash's code, frames in the grey inter-stimulus interval are **forward-filled** with the last shown image, frames before the first flash stay −1. (3) After slicing a trial, any remaining −1 entries (only possible at the very start of a recording) are back-filled with the first valid code in that trial. Values are stored as row 0 of the (5, n_timepoints) output array.

ii.
```python
    # Forward-fill
    last_img = -1
    for i in range(n_tp):
        if image_idx[i] >= 0:
            last_img = image_idx[i]
        elif last_img >= 0:
            image_idx[i] = last_img
```
```python
    all_image_names = sorted(list(all_image_names))
    image_to_idx = {name: idx for idx, name in enumerate(all_image_names)}
```
```python
        # Fix any -1 image indices (before first stimulus)
        neg_mask = img_trial < 0
        if neg_mask.any():
            first_valid = img_trial[~neg_mask][0] if (~neg_mask).any() else 0
            img_trial[neg_mask] = first_valid
```

iii. CONVERSION_NOTES Step 5 Key Decision 3: "**Image identity during gray screen**: Forward-filled with last shown image" — the grey period is part of the 750 ms presentation cycle and carries the same stimulus context, and the instruction defines the variable as "the image presented during the non-grey screen". A single global sorted mapping keeps codes consistent across sessions and image sets (Step 9 consistency check: "16 image names (8 per image set, 2 sets)"). The −1 backfill was added in Step 10 Check 5 after finding "4 trials had −1 image identity at first frame (before first stimulus)".

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. The image trace is built on the full session's `ophys_timestamps` vector and then indexed with exactly the same `frame_indices` used for the neural matrix, so alignment is identical by construction. A frame is assigned an image if its timestamp falls in `[flash_start, flash_end)`. Spot-check on experiment 792815735, trial 0: the converted row 0 contains only codes 8 (`im065`, the initial image) and 14 (`im085`, the change image), and switches at frame 117, exactly `np.searchsorted(ophys_ts[idx], change_time)`.

ii.
```python
        img_trial = image_idx_full[frame_indices]
```

iii. CONVERSION_NOTES Step 5/Step 10 Check 3: "Temporal alignment: Based on ophys timestamps ✓". Since every stream is first placed on the ophys timebase and then all sliced by one index array, no per-stream alignment logic can drift. The `--show-processing` figure plots image identity together with neural events and trial/change markers on a shared time axis.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. From `stimulus_presentations.is_change` (plus `start_time`/`end_time`) within the `change_detection_behavior` block. The trials table's `change_time`/`go` columns are not used for this output. I confirmed on experiment 792815735 that `is_change` is True for exactly 164 flashes = the 164 go trials, and is False at the sham "change" of every catch trial — so catch trials correctly get an all-zero change row, the same semantics as the reference's go-only rule.

ii.
```python
def get_change_at_ophys(sp_active, ophys_ts):
    """Get binary change signal at ophys timestamps."""
    n_tp = len(ophys_ts)
    change = np.zeros(n_tp, dtype=np.int32)

    change_sp = sp_active[sp_active['is_change'] == True]
    for _, row in change_sp.iterrows():
        mask = (ophys_ts >= row['start_time']) & (ophys_ts < row['end_time'])
        change[mask] = 1

    return change
```

iii. CONVERSION_NOTES Step 5 variable mapping: `stimulus_presentations.is_change → output[1]`, "Binary at ophys timestamps, 1 during change presentation". The AI treated `is_change` as the SDK's canonical flag for "this flash differs from the previous one", which is exactly the event the instruction describes ("value of 1 right after a change in image identity").

## 4-b. What processing is involved in computing `output` *Image change*?

i. Essentially none: a zero vector of session length is set to 1 on the frames covered by each changed flash, and the trial slice of that vector becomes row 1 of the output. No smoothing, no extension into the following grey period, no per-trial special-casing. In the full dataset 2.6 % of timepoints are labelled `change` (8 frames ≈ 250 ms per go trial, ~9.6 flashes per trial).

ii. See the snippet in 4-a; the trial-level use is:
```python
        change_trial = change_full[frame_indices]
        ...
        output_trial = np.stack([img_trial, change_trial, running_trial, pupil_trial, outcome_trial], axis=0)
```

iii. CONVERSION_NOTES Step 5: the change indicator is meant to mark the transient change event itself, which the AI equated with the duration of the changed image flash. Step 7/Step 9 report the resulting rate (2.5–2.6 % of frames) and the AI accepted it as sensible given that a change occupies one of ~9.6 flashes per trial.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is natively binary, so no thresholding of a continuous quantity is needed. The category boundary is temporal: 1 for the ~8 ophys frames spanning the 250 ms presentation of the changed image (`[start_time, end_time)` of that flash), 0 everywhere else — including the 500 ms grey period that follows the change. `output_values[1] = ['no_change', 'change']`. Verified on experiment 792815735 trial 0: frames 117–124 are 1 (8 frames ≈ 258 ms), starting exactly at the change frame.

ii.
```python
        mask = (ophys_ts >= row['start_time']) & (ophys_ts < row['end_time'])
        change[mask] = 1
```
```python
    output_values = [
        all_image_names,
        ['no_change', 'change'],
        ...
```

iii. CONVERSION_NOTES Step 5 ("1 during change presentation") and README ("1 during change image presentation"). The AI's rationale is that the change is a transient stimulus event whose natural extent is the flash in which it occurred; it noted the resulting strong class imbalance (97.4 % no_change) in Step 12 and attributed the modest change-decoding accuracy to that imbalance rather than to the window length.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Identically to image identity: the binary trace is computed over the whole session on the ophys timebase and sliced with the same `frame_indices` as the neural matrix, so the change onset coincides with the first ophys frame at or after the change flash onset (≤ 1 frame, ~32 ms, of quantization error).

ii.
```python
        change_trial = change_full[frame_indices]
```

iii. Same reasoning as 3-c: all streams share one timebase and one index array (CONVERSION_NOTES Step 10 Check 3). The processing plots draw the change signal against the trial's `change_time` marker to make the alignment visually checkable.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. `dataset.running_speed`, using its `timestamps` and `speed` columns (running-disc encoder, ~60 Hz).

ii.
```python
    # Running speed resampled to ophys timestamps
    running = dataset.running_speed
    running_full = resample_to_ophys(running['timestamps'].values, running['speed'].values, ophys_ts)
```

iii. CONVERSION_NOTES Step 1 lists `dataset.running_speed` as the SDK accessor for locomotion ("Running speed and pupil at ~60 Hz, need resampling to ophys timestamps"); the tutorials use the same attribute.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. (1) Linear interpolation onto the ophys timestamps with `np.interp` (which clamps rather than extrapolating outside the recorded range, so no NaNs are produced). (2) Global percentile edges at 0/20/40/60/80/100 % are computed once from the concatenation of every session's *full-session* interpolated trace. (3) `np.digitize` against the four interior edges, clipped to 0–4. No smoothing or unit conversion; negative speeds (backwards wheel motion) are retained (edge 0 = −23.98 cm/s).

ii.
```python
def resample_to_ophys(signal_timestamps, signal_values, ophys_timestamps):
    """Resample a signal to ophys timestamps using linear interpolation."""
    return np.interp(ophys_timestamps, signal_timestamps, signal_values).astype(np.float32)
```
```python
    all_running_concat = np.concatenate(all_running_values)
    running_edges = np.percentile(all_running_concat, np.linspace(0, 100, 6))
```
```python
    # Bin running speed
    running_binned = np.clip(np.digitize(running_full, running_edges[1:-1]), 0, 4).astype(np.int32)
```

iii. CONVERSION_NOTES Step 5 Key Decision 2: "**Global percentile binning**: Running speed and pupil diameter binned using percentiles computed across all experiments", so that bin labels mean the same thing in every session and the classes are ~balanced for the decoder. Resampling to the ophys timebase (rather than the reverse) is justified in Step 5 by the requirement that all streams share the neural time bins.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Five bins defined by the global 0/20/40/60/80/100-th percentiles of the pooled interpolated running speed: edges `[-23.98, 0.0368, 3.746, 21.35, 36.14, 99.92]` cm/s. Values are assigned with `np.digitize` on the interior edges and clipped into [0, 4]; `output_values[2] = ['bin_0' … 'bin_4']`. Because percentiles were computed over whole sessions but only in-trial frames are exported, the within-dataset class fractions are approximately but not exactly uniform: 0.193 / 0.195 / 0.194 / 0.215 / 0.203.

ii.
```python
    running_edges = np.percentile(all_running_concat, np.linspace(0, 100, 6))
    ...
    running_binned = np.clip(np.digitize(running_full, running_edges[1:-1]), 0, 4).astype(np.int32)
```

iii. Directly from the Decoder Task instruction: "Running speed, discretized into five equal percentile bins." CONVERSION_NOTES Step 9 reports the resulting near-uniform distribution as a consistency check, and Step 10 Check 2 records a `np.allclose` sanity check recomputing the bins from the raw NWB file.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. The speed trace is interpolated onto `ophys_timestamps` *before* trial segmentation and binned at full-session level, then the trial slice uses the same `frame_indices` as the neural matrix — so alignment is exact by construction. I reproduced row 2 of session 0 / trial 0 from the raw NWB file (interp + digitize with the stored edges) and got an exact match.

ii.
```python
    running_full = resample_to_ophys(running['timestamps'].values, running['speed'].values, ophys_ts)
    ...
    running_binned = np.clip(np.digitize(running_full, running_edges[1:-1]), 0, 4).astype(np.int32)
    ...
        running_trial = running_binned[frame_indices]
```

iii. Same rationale as 2-d/3-c: the AllenSDK NWB files carry all streams on a common hardware-synced clock, so linear interpolation onto ophys timestamps is valid and guarantees alignment (CONVERSION_NOTES Step 10 Check 3).

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. `dataset.eye_tracking['pupil_width']` with `eye_tracking['timestamps']`. Frames whose `pupil_width` is NaN are dropped before interpolation. The SDK already sets `pupil_width` to NaN exactly on `likely_blink` frames (verified on experiment 792815735: 3,073 blink frames, 3,073 NaNs, zero NaNs among non-blink frames), so this NaN mask is equivalent to explicit blink removal. If a session has ≤ 10 valid samples, or eye tracking raises, pupil is set to all-NaN for that session.

ii.
```python
    # Pupil diameter resampled to ophys timestamps
    try:
        eye = dataset.eye_tracking
        pupil_raw = eye['pupil_width'].values
        pupil_ts = eye['timestamps'].values
        valid_mask = ~np.isnan(pupil_raw)
        if valid_mask.sum() > 10:
            pupil_full = resample_to_ophys(pupil_ts[valid_mask], pupil_raw[valid_mask], ophys_ts)
        else:
            pupil_full = np.full(len(ophys_ts), np.nan, dtype=np.float32)
    except:
        pupil_full = np.full(len(ophys_ts), np.nan, dtype=np.float32)
```

iii. CONVERSION_NOTES Step 5 maps `dataset.eye_tracking.pupil_width → output[3]` ("Resample to ophys, 5 percentile bins"). Step 1 notes eye tracking is sampled at ~60 Hz and must be resampled. The AI read the whitepaper's eye-tracking section (trajectory step 47) before choosing `pupil_width`; dropping NaNs is its blink/dropout handling.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Same pipeline as running speed: drop NaN (blink) samples → `np.interp` onto ophys timestamps (edge-clamped) → global 5-quantile edges computed from the pooled *valid* full-session traces of all sessions → `np.digitize` + clip to 0–4. NaN frames (only possible when a whole session has no usable eye data) are left at bin 0. No unit conversion or smoothing; the units are the raw ellipse-fit pupil width in camera pixels (edges span 4.81–252.2).

ii.
```python
    if len(all_pupil_values) > 0:
        all_pupil_concat = np.concatenate(all_pupil_values)
        pupil_edges = np.percentile(all_pupil_concat, np.linspace(0, 100, 6))
```
```python
    valid_pupil_mask = ~np.isnan(pupil_full)
    pupil_binned = np.zeros(len(ophys_ts), dtype=np.int32)
    if valid_pupil_mask.any() and pupil_edges is not None:
        pupil_binned[valid_pupil_mask] = np.clip(
            np.digitize(pupil_full[valid_pupil_mask], pupil_edges[1:-1]), 0, 4
        ).astype(np.int32)
```

iii. CONVERSION_NOTES Step 5 Key Decision 2 (global percentile binning, for cross-session comparability and balanced classes) plus the instruction "Pupil diameter, discretized into five equal percentile bins". Step 10 Check 2 records a `np.allclose` sanity check of the pupil bins against a recomputation from the raw NWB file.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Five global percentile bins with edges `[4.81, 36.88, 41.48, 46.26, 53.04, 252.21]`, assigned by `np.digitize` on the interior edges and clipped to [0, 4]; `output_values[3] = ['bin_0' … 'bin_4']`. Missing/blink-only sessions fall into bin 0. Realized class fractions in the exported data: 0.202 / 0.184 / 0.205 / 0.212 / 0.196 — close to uniform, deviating because the percentiles were taken over whole sessions rather than over in-trial frames only.

ii.
```python
    pupil_edges = np.percentile(all_pupil_concat, np.linspace(0, 100, 6))
    ...
        pupil_binned[valid_pupil_mask] = np.clip(
            np.digitize(pupil_full[valid_pupil_mask], pupil_edges[1:-1]), 0, 4
        ).astype(np.int32)
```

iii. As for running speed: the Decoder Task requires five equal-percentile bins, and CONVERSION_NOTES Step 9 uses the near-uniform realized distribution as the consistency check that the discretization worked.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Interpolated onto `ophys_timestamps` before segmentation and sliced with the same `frame_indices` as the neural matrix. I reproduced row 3 of session 0 / trial 0 from the raw NWB file and obtained an exact match.

ii.
```python
    pupil_full = resample_to_ophys(pupil_ts[valid_mask], pupil_raw[valid_mask], ophys_ts)
    ...
        pupil_trial = pupil_binned[frame_indices]
```

iii. Same justification as running speed — one shared, hardware-synced timebase, one index array per trial (CONVERSION_NOTES Step 5, Step 10 Check 3). Note that because blink samples are removed *before* interpolation, the interpolation bridges blink gaps rather than propagating artifacts.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. The four mutually exclusive boolean columns of the trials table — `hit`, `miss`, `false_alarm`, `correct_reject` — checked in that order, mapping to codes 0–3 (−1 if none match, which cannot occur for go/catch non-auto-rewarded trials).

ii.
```python
def get_trial_outcome(trial_row):
    """Get trial outcome as integer: 0=hit, 1=miss, 2=false_alarm, 3=correct_reject."""
    if trial_row['hit']: return 0
    if trial_row['miss']: return 1
    if trial_row['false_alarm']: return 2
    if trial_row['correct_reject']: return 3
    return -1
```

iii. CONVERSION_NOTES Step 3 identifies these as the SDK's canonical outcome labels for the change-detection task, and the trajectory (step 57) verified that "Each valid trial has exactly one outcome (hit/miss/FA/CR)" before the mapping was written.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The integer code is broadcast to a constant vector over the trial's timepoints and stored as row 4 of the output array, i.e. a static per-trial label represented in time-varying form. `output_values[4] = ['hit', 'miss', 'false_alarm', 'correct_reject']`. Realized distribution over the full dataset: hit 0.315, miss 0.560, false_alarm 0.018, correct_reject 0.107.

ii.
```python
        outcome = get_trial_outcome(trial_row)
        outcome_trial = np.full(len(frame_indices), outcome, dtype=np.int32)
```

iii. CONVERSION_NOTES Step 5 Key Decision 4: "**Trial outcome as time-varying**: Repeated constant value across trial timepoints" — the target format asks for time-varying outputs "if at all possible" and requires all output rows to share one shape, so the per-trial scalar is tiled. Step 12 notes the decoder does comparatively poorly on this output precisely because it is constant within a trial.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Six mechanisms: (1) per-experiment `try/except` around NWB loading — a failing experiment is recorded as `None` and skipped with a printed error; (2) a bare `try/except` around eye tracking plus a "> 10 valid samples" guard — a session without usable pupil data gets an all-NaN trace, which then bins to a constant 0; (3) NaN (blink) pupil samples are dropped before interpolation and NaN frames after it are left at bin 0; (4) `np.interp` edge-clamps rather than extrapolating, so behavioral frames outside the recorded range get the nearest value instead of NaN; (5) image indices of −1 at the very start of a recording are back-filled with the first valid image of the trial; (6) degenerate units are dropped — sessions with < 2 neurons or < 2 valid trials, and trials with < 5 ophys frames. Trials whose `stop_time` runs past the recording simply pick up fewer frames (the boolean mask cannot over-run the array). All-zero neural trials (1,860 of them) are left in the data and reported as warnings.

ii.
```python
        try:
            dataset = load_experiment(exp_id)
        except Exception as e:
            print(f'  ERROR loading experiment {exp_id}: {e}')
            all_session_data.append(None)
            continue
```
```python
        if valid_mask.sum() > 10:
            pupil_full = resample_to_ophys(pupil_ts[valid_mask], pupil_raw[valid_mask], ophys_ts)
        else:
            pupil_full = np.full(len(ophys_ts), np.nan, dtype=np.float32)
    except:
        pupil_full = np.full(len(ophys_ts), np.nan, dtype=np.float32)
```
```python
        neg_mask = img_trial < 0
        if neg_mask.any():
            first_valid = img_trial[~neg_mask][0] if (~neg_mask).any() else 0
            img_trial[neg_mask] = first_valid
```

iii. CONVERSION_NOTES Step 10 Check 5 ("Fixed: 4 trials had −1 image identity at first frame (before first stimulus). Resolution: backward-fill with next valid image index") and Check 1 ("Warnings: all-zero neural data in some trials (expected for calcium events) … some trials genuinely have no detected calcium events"). The general principle stated in Step 6/Step 10 is that a single bad session or trial must not abort the pipeline, and that missing behavioral data should fall back to a defined discrete label rather than propagate NaN into the outputs.

## 9-a. What are the most time-consuming steps of the code?

i. Reading the NWB files dominates. The full run took 1,475 s (24.6 min), of which Pass 1 — opening each of the 168 NWB files with `BehaviorOphysExperiment.from_nwb_path()` and pulling events/trials/stimulus/running/eye tables — took 1,277 s (87 %), at 4.5–16 s per experiment scaling with neuron count. Pass 2 (recomputing image indices, binning, trial slicing, assembly) costs ~1.0–1.2 s per session, ~180 s in total. Pickling the 8.2 GB output is the remaining minutes. The AI reduced runtime earlier by restructuring so each NWB file is opened only once instead of twice.

ii.
```python
    for i, exp_id in enumerate(exp_ids):
        t0 = time.time()
        ...
        t_elapsed = time.time() - t0
        if (i + 1) % 10 == 0 or i == 0:
            print(f'  [{i+1}/{n_exp}] Experiment {exp_id}: {session_data["n_neurons"]} neurons, '
                  f'{len(session_data["valid_trials"])} trials, {t_elapsed:.1f}s')
    ...
    print(f'Pass 1 completed in {time.time() - total_start:.1f}s')
```

iii. CONVERSION_NOTES Step 6: "two-pass approach … Processing time: ~24 min for full dataset". Trajectory steps 68–69 record the AI measuring ~3.5 s/experiment on the sample, estimating ~17 min for the full set, identifying double-loading of NWB files as the avoidable cost, and rewriting the script to cache Pass-1 extractions in memory so Pass 2 needs no file I/O.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Four:
- `get_image_at_ophys` loops over every stimulus presentation (~4,800/session) and builds a full-length boolean mask over all ~140,000 ophys timestamps each time — O(n_flashes × T) ≈ 7×10⁸ comparisons per session. Two `np.searchsorted` calls on the flash start/end times would be O(T log n).
- The forward-fill inside the same function is a pure-Python per-sample loop over all ~140,000 timestamps; `np.maximum.accumulate` on a masked index array does this in one vectorized pass.
- `get_change_at_ophys` uses `DataFrame.iterrows()` plus a full-length mask per change flash (same pattern as above).
- The per-trial loop in `segment_trials` computes `(ophys_ts >= start) & (ophys_ts < stop)` over the whole session for each of ~260 trials — O(n_trials × T); `np.searchsorted` on the sorted timestamp vector gives the same indices directly.

None of these is the dominant cost (NWB I/O is), but together they account for essentially all of Pass 2 and part of Pass 1, and `get_image_at_ophys` is run twice per session. The AI also left the whole pipeline single-threaded despite 32 available cores and an instruction to consider parallel processing.

ii.
```python
    for j in range(len(starts)):
        ...
        mask = (ophys_ts >= starts[j]) & (ophys_ts < ends[j])
        image_idx[mask] = image_to_idx[img_name]

    # Forward-fill
    last_img = -1
    for i in range(n_tp):
        if image_idx[i] >= 0:
            last_img = image_idx[i]
        elif last_img >= 0:
            image_idx[i] = last_img
```
```python
    for _, row in change_sp.iterrows():
        mask = (ophys_ts >= row['start_time']) & (ophys_ts < row['end_time'])
        change[mask] = 1
```
```python
        frame_mask = (ophys_ts >= start_time) & (ophys_ts < stop_time)
        frame_indices = np.where(frame_mask)[0]
```

iii. The AI labels `get_image_at_ophys` "Vectorized: for each stimulus presentation, find ophys frames" in a comment, indicating it believed the mask-based assignment was already the vectorized form. CONVERSION_NOTES Step 6 only reports the I/O-level speed-up (single load per file) and does not identify any remaining loop as a bottleneck; the AI accepted the 24.6 min runtime even though the instructions ask for optimization beyond 15 min.

## 9-c. What processing does the code repeat multiple times?

i. Three repeats:
- **Image identity is computed twice per session.** Pass 1 calls `get_image_at_ophys` with a *provisional* mapping built from the image names seen so far; Pass 2 throws that result away and recomputes it with the final global mapping. This is the single most expensive redundant computation in the script.
- **`stimulus_presentations` is filtered to the `change_detection_behavior` block twice** per experiment in Pass 1 (once inline to harvest image names, once inside `extract_session_data`).
- **`plot_processing` re-derives the running and pupil binning** from the full-session traces instead of reusing `segment_trials`' arrays (only in `--show-processing` mode, for ≤ 2 sessions).

ii.
```python
        # Build image_to_idx (temporary, will rebuild after collecting all names)
        temp_image_to_idx = {name: idx for idx, name in enumerate(sorted(all_image_names))}

        # Extract session data
        session_data = extract_session_data(dataset, temp_image_to_idx)
```
```python
        # Re-compute image indices with global image_to_idx
        session_data['image_idx_full'] = get_image_at_ophys(session_data['sp_active'], session_data['ophys_ts'], image_to_idx)
```
```python
    running_binned = np.clip(np.digitize(session_data['running_full'], running_edges[1:-1]), 0, 4)
```

iii. The two-pass design exists because the global image mapping and the global percentile edges are only known after all sessions have been seen (CONVERSION_NOTES Step 6). The AI kept `sp_active` in memory specifically so Pass 2 would not have to reopen the NWB file (trajectory step 69), but did not notice that it could equally have stored image *names* per frame once and mapped them to codes at the end, avoiding the recomputation entirely.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several items, mostly memory rather than CPU:
- **Full-session data for every experiment is retained in RAM for the whole run.** `all_session_data` accumulates each session's complete `events_array` ((n_neurons × ~140,000) float32), plus the full-session image/change/running/pupil traces and the `sp_active` table, for all 168 sessions — roughly 16 GB of event data alone — even though only the ~30 % of frames that fall inside trials are ever exported. The reference explicitly deletes the full-session arrays after segmentation.
- **Pass 1's `image_idx_full`** (a 140,000-element trace per session) is computed and then discarded (see 9-c).
- **Running and pupil are interpolated and binned over the entire session**, including all inter-trial and aborted-trial periods that are never written out.
- **Per-session `dt`** is computed but unused for the exported `time_bin_size` (which is hard-coded to 1000/31); it only appears in `session_info`.
- **Empty input arrays** of shape `(0, n_timepoints)` are allocated per trial although `input_names` is empty and the decoder takes no inputs — harmless but pointless.
- The trial-outcome row is tiled across ~262 timepoints per trial although it is constant; that is required by the output format, not waste.

ii.
```python
        all_session_data.append(session_data)   # never trimmed; holds full-session arrays
```
```python
        input_trials = [np.zeros((0, t.shape[1]), dtype=np.float32) for t in neural_trials]
```
```python
    dt = np.median(np.diff(ophys_ts))
    ...
            'time_bin_size': 1000.0 / 31.0,  # ~32.3 ms
```

iii. The AI's stated reason for caching everything from Pass 1 is to avoid reopening NWB files in Pass 2 (trajectory step 69, CONVERSION_NOTES Step 6). It did notice the size of the final artifact ("The file is 8.2 GB which is very large", trajectory step 95) but never revisited peak memory or the retention of full-session arrays, and CONVERSION_NOTES contains no discussion of discarded intermediate computation.
