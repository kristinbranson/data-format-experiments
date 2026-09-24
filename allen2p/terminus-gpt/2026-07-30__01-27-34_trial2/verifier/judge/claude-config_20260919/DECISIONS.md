# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI ignores the Allen SDK project cache / metadata tables entirely and instead globs every NWB file under the local data directory (`data/visual-behavior-ophys-1.1.0/**/*.nwb`, which resolves to the 284 files in `data/behavior_ophys_experiments/`). Each file is opened with `BehaviorOphysExperiment.from_nwb_path()`, which yields the full SDK experiment object (metadata, `ophys_timestamps`, `events`, `dff_traces`, `trials`, `stimulus_presentations`, `running_speed`, `eye_tracking`). One NWB file = one ophys **experiment** = one decoder "session". Files are processed with a `multiprocessing.Pool` of up to 8 workers in `--full` mode. No filtering on `project_code`, `session_type`, or `experience_level` is applied, so the run includes all 239 `VisualBehavior` (single-plane, ~31 Hz) experiments **and** all 45 `VisualBehaviorMultiscope` (8-plane, ~11 Hz) experiments. The `project_metadata/*.csv` tables were inspected during Steps 2–3 but are not read by `convert_data.py`.

ii.
```python
DATASET_ROOT = Path('data/visual-behavior-ophys-1.1.0')

def list_nwb_files():
    files = sorted(DATASET_ROOT.rglob('*.nwb'))
    if not files:
        raise FileNotFoundError(f'No NWB files found under {DATASET_ROOT}')
    return files

def choose_files(sample=False):
    files = list_nwb_files()
    return files[:2] if sample else files

def process_experiment(nwb_path):
    exp = BehaviorOphysExperiment.from_nwb_path(str(nwb_path))
    meta = exp.metadata
    ophys_t = np.asarray(exp.ophys_timestamps, dtype=np.float64)
    neural_full = extract_events_matrix(exp.events)
    ...

    use_parallel = (not sample) and len(files) > 4
    if use_parallel:
        n_workers = min(max((os.cpu_count() or 2) // 2, 2), 8)
        with Pool(processes=n_workers) as pool:
            for i, sess in enumerate(pool.imap_unordered(process_experiment, files), start=1):
                ...
```

iii. From CONVERSION_NOTES.md Step 4/Step 5: "Treat each ophys experiment NWB as one decoder session because neural populations are experiment-specific" and "each NWB/experiment has its own neural population and aligned behavior tables; this matches decoder session semantics." The AI reasoned that `BehaviorOphysExperiment` already encodes sync/alignment between behavior and ophys streams, so loading it directly from NWB is equivalent to going through the cache. No justification is given anywhere for including the Multiscope project code; the notes acknowledge the paper used "familiar image set sessions from the multiplane calcium imaging rig" and then state "for this conversion we will include all available Visual Behavior task sessions unless later reference constraints indicate stronger filtering is required."

## 1-b. How are the data split into subjects?

i. One subject per `metadata['mouse_id']` of each experiment, stored as a string. The unique set is sorted and `subject_idx` is the index of the owning mouse for each session. Result: 38 subjects.

ii.
```python
session = {
    'session_id': int(meta['ophys_experiment_id']),
    'subject': str(meta['mouse_id']),
    'region': str(meta['targeted_structure']),
    ...
}

subjects = sorted({s['subject'] for s in processed_sessions})
subject_to_idx = {s: i for i, s in enumerate(subjects)}
...
'subject_idx': np.array([subject_to_idx[s['subject']] for s in processed_sessions], dtype=np.int64),
```

iii. Step 5 mapping table: "`metadata['mouse_id']` → subjects / subject_idx — Unique mouse IDs mapped to subject index per experiment/session. One subject per experiment session." `mouse_id` is the SDK's canonical animal identifier.

## 1-c. How are the data split into sessions?

i. Each NWB file — i.e. each **ophys experiment** (one imaging plane) — becomes one decoder session, keyed by `ophys_experiment_id`. Planes belonging to the same `ophys_session_id` are never merged. For the 239 single-plane `VisualBehavior` experiments this is equivalent to one session per recording; for the 45 Multiscope experiments, one physical recording session is emitted as up to 8 separate "sessions" that share the same behavior. Total: 284 sessions, with one mouse (457841) contributing 45 of them. Sessions with fewer than 2 kept trials would be dropped, but none are.

ii.
```python
    for i, sess in enumerate(pool.imap_unordered(process_experiment, files), start=1):
        print(f'[{i}/{len(files)}] finished session {sess["session_id"]} trials={len(sess["trials"])}')
        if len(sess['trials']) >= 2:
            processed.append(sess)
            ...
        else:
            print(f'skipping session {sess["session_id"]}: <2 valid trials')
```
Session ordering is whatever `imap_unordered` returns (non-deterministic across runs).

iii. Step 4 discrepancy table: "Session granularity | Metadata distinguishes behavior sessions, ophys sessions, and ophys experiments | Example NWB is per ophys experiment with ROI-level data | Paper discusses imaging planes/experiments within sessions | **Resolution: Treat each ophys experiment NWB as one decoder session because neural populations are experiment-specific**."

## 1-d. Are the data correctly split into trials?

i. **This is the AI's largest deviation.** A decoder "trial" is *not* an experimental trial: it is a single **image flash** (one row of `stimulus_presentations`). The SDK `trials` table is used only to decide *which* flashes are eligible (flashes whose `trials_id` points at a Go/Catch, non-aborted, non-auto-rewarded trial) and to supply the trial outcome label. Each kept flash window runs from the presentation's `start_time` to its `end_time`, i.e. the 250 ms image-on period — the 500 ms grey inter-stimulus interval is discarded. At the single-plane frame rate (~31 Hz) this is 7–8 ophys frames; at the Multiscope rate (~11 Hz) it is 2–3 frames. Overall `T`: mean 6.94, min 2, max 11 frames. This produced 963,758 "trials" (vs. the reference's 71,242 trials of ~264 frames each).

The AI's first implementation did use the SDK trial window (`start_time`→`stop_time`); it was abandoned after the decoder crashed on `-1` image-identity labels during grey periods.

ii.
```python
def valid_trials_df(trials):
    trials = trials.copy()
    keep = (trials['go'].fillna(False) | trials['catch'].fillna(False))
    keep &= ~trials['aborted'].fillna(False)
    keep &= ~trials['auto_rewarded'].fillna(False)
    trials = trials.loc[keep].copy()
    trials['trial_outcome_idx'] = trials.apply(get_trial_outcome, axis=1)
    trials = trials.loc[trials['trial_outcome_idx'].notna()].copy()
    return trials

    stim = exp.stimulus_presentations.copy().sort_values('start_time')
    stim = stim[stim['trials_id'].notna()].copy()
    stim = stim[stim['trials_id'].isin(valid_trial_ids)].copy()
    if 'active' in stim.columns:
        stim = stim[stim['active'].fillna(False)].copy()
    if 'omitted' in stim.columns:
        stim = stim[~stim['omitted'].fillna(False)].copy()
    stim = stim[stim['image_name'].notna()].copy()
    stim = stim[stim['start_time'].notna() & stim['end_time'].notna()].copy()

    for stim_id, parent_id, start, stop, img_name, is_change in zip(...):
        left = np.searchsorted(ophys_t, start, side='left')
        right = np.searchsorted(ophys_t, stop, side='left')
        if right - left < 2:
            continue
        trial_t = ophys_t[left:right]
        neural = neural_full[:, left:right].astype(np.float32, copy=False)
```

iii. Trajectory step 21: "The root issue is that trial windows include gray/ITI periods where no non-grey image is presented, but image_identity is required only during non-grey screens. We need to redefine trials or labels so all timepoints have valid categories. The most consistent fix is to segment into image-presentation intervals within valid Go/Catch trials… This aligns better with the paper's image-by-image analyses and the 750 ms image presentation interval." CONVERSION_NOTES.md Step 7 repeats this: "Initial full-trial segmentation produced invalid unlabeled image-identity periods during gray screens. Revised conversion uses image-presentation intervals as trials, eliminating invalid labels and matching the paper's image-by-image analysis," and claims "trial lengths are 7-8 ophys frames, which matches 750 ms image presentation intervals at ~10 Hz."

## 1-e. How are trials filtered based on quality controls?

i. Filters applied, in order: (1) parent SDK trial must be `go` or `catch`; (2) not `aborted`; (3) not `auto_rewarded`; (4) parent trial must carry exactly one of `hit/miss/false_alarm/correct_reject` (otherwise dropped); (5) the flash must have a non-null `trials_id` pointing at a kept trial; (6) `active` stimulus block only (passive/grey blocks dropped); (7) `omitted` flashes dropped; (8) non-null `image_name`, `start_time`, `end_time`; (9) the flash window must contain at least 2 ophys frames; (10) sessions with <2 kept flashes are dropped (none were). No filtering is applied on missing behavioural data, and — crucially — no filter or flag is applied to the 448,413 kept trials (46.5% of the dataset) that the verifier reports as containing **entirely zero** neural data.

ii.
```python
    keep = (trials['go'].fillna(False) | trials['catch'].fillna(False))
    keep &= ~trials['aborted'].fillna(False)
    keep &= ~trials['auto_rewarded'].fillna(False)
    ...
    trials = trials.loc[trials['trial_outcome_idx'].notna()].copy()
    ...
        if right - left < 2:
            continue
    ...
        if len(sess['trials']) >= 2:
            processed.append(sess)
        else:
            print(f'skipping session {sess["session_id"]}: <2 valid trials')
```

iii. CONVERSION_NOTES.md Step 3/Step 5: "Include Go and Catch trials. Exclude Aborted and Auto-rewarded trials, per task instructions." Omitted flashes and inactive blocks are excluded so that every timepoint carries a valid non-grey image label. The ≥2-frame rule and ≥2-trial rule are not discussed in the notes; the ≥2-trial rule follows the format requirement "There needs to be at least two trials within each session." The 448k all-zero-neural warnings in `verification_full_out.txt` were never mentioned — Step 10 (Critical Review 1) was left with template placeholders.

## 2-a. What variables in the raw data is the final `neural` data derived from?

i. The SDK **detected calcium events** table, `exp.events['events']` (unfiltered deconvolved event magnitudes per ROI, one trace per cell on the ophys timebase). `dff_traces` is available in the same NWB and was deliberately not used. The `filtered_events` column (half-gaussian smoothed) was also not used.

ii.
```python
def extract_events_matrix(events_df):
    event_col = 'events'
    if event_col not in events_df.columns:
        raise KeyError(f'Expected {event_col} column in events dataframe')
    arrs = [np.asarray(x, dtype=np.float32) for x in events_df[event_col].values]
    lengths = {a.shape[0] for a in arrs}
    if len(lengths) != 1:
        raise ValueError(f'Inconsistent event trace lengths: {lengths}')
    return np.stack(arrs, axis=0)

    neural_full = extract_events_matrix(exp.events)
```

iii. CONVERSION_NOTES.md Step 4/Step 5 Key Decision 1: "**Neural signal = detected events**: methods.txt explicitly states neural analyses used detected calcium events, so use `events` rather than dF/F," backed by the quote "For all analysis of neural data we used the detected calcium events…" and recorded in metadata as `'neural_signal': 'detected calcium events'`.

## 2-b. How is the `neural` data processed?

i. Essentially none. The per-ROI event traces are stacked into a single `(n_neurons, T_session)` float32 matrix in the order they appear in `exp.events`, then sliced per trial. No smoothing, normalisation, z-scoring, baseline subtraction, rebinning, or cross-plane merging. `float32` is used throughout.

ii.
```python
    arrs = [np.asarray(x, dtype=np.float32) for x in events_df[event_col].values]
    return np.stack(arrs, axis=0)
    ...
    neural = neural_full[:, left:right].astype(np.float32, copy=False)
```

iii. Step 5 mapping table: "Stack per-cell event traces into `(n_neurons, n_timepoints)` trial matrices by slicing on ophys timestamps between trial start/stop." The AI treats the SDK event traces as already fully processed; "Code speedups added: … Stores neural traces as float32."

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neuron-level quality control at all. Every ROI present in the NWB `events` table is kept (the SDK has already restricted this to valid cell specimens). The AI also validates that all ROI traces in a file have the same length and raises if not. Resulting counts: 42,147 neurons, mean 148.4/session, min 4, max 666. No trial- or neuron-level check for empty/zero activity is performed despite 46.5% of trials being all-zero.

ii.
```python
    lengths = {a.shape[0] for a in arrs}
    if len(lengths) != 1:
        raise ValueError(f'Inconsistent event trace lengths: {lengths}')
```
(There is no other neural QC code.)

iii. CONVERSION_NOTES.md Step 3 "Neuron curation rules": "Use SDK-provided valid cell/ROI tables and whichever neural signal the reference analysis used (detected calcium events per methods text). Additional filtering rules still need confirmation from code/data inspection." That confirmation step (Step 10) was never completed.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to **image-flash onset**. For each kept flash, `np.searchsorted(ophys_t, start, 'left')` and `np.searchsorted(ophys_t, stop, 'left')` give the half-open ophys frame range `[left, right)`, and the neural matrix is sliced with those indices. All outputs use the same index range, so neural and output rows are index-aligned by construction. Every trial therefore begins at the first ophys frame at/after image onset and ends just before the image goes off (~250 ms later); no pre-stimulus baseline is included and no response lag is allowed for. The written metadata, however, describes a different alignment: `'temporal_alignment_event': 'native ophys timestamps within each trial defined by SDK trial start/stop times'`, with `off_start = 0.0` and `off_end = None`.

ii.
```python
        left = np.searchsorted(ophys_t, start, side='left')
        right = np.searchsorted(ophys_t, stop, side='left')
        if right - left < 2:
            continue
        trial_t = ophys_t[left:right]
        neural = neural_full[:, left:right].astype(np.float32, copy=False)
        run_aligned = interp_to_ophys(run_t, run_v, trial_t)
        pupil_aligned = interp_to_ophys(pupil_t, pupil_v, trial_t)
...
        'temporal_alignment_event': 'native ophys timestamps within each trial defined by SDK trial start/stop times',
        'off_start': 0.0,
        'off_end': None,
```

iii. Step 4/Step 5 Key Decision 3: "**Temporal basis = ophys timestamps**: all trial matrices will be sampled on native ophys timestamps as required by the task," and "Resample/assign all outputs onto ophys timestamps within each trial." The choice of flash onset as the zero point is a consequence of the image-presentation trial definition (1-d); it is not separately justified.

## 2-e. How is the `neural` data temporally binned/resampled?

i. No rebinning or resampling of the neural data: it stays on the native ophys frame grid. Because both project codes are included, the bin size is **not uniform across sessions** — ~32.3 ms for the 239 single-plane `VisualBehavior` experiments and ~93 ms for the 45 Multiscope experiments. The format's `metadata['time_bin_size']` field is hard-coded to `None` rather than a float in ms. The notes state trial length "7-8 ophys frames … matches 750 ms image presentation intervals at ~10 Hz", which is arithmetically wrong: the single-plane rate is ~31 Hz, so 7–8 frames is the 250 ms flash, not 750 ms.

ii.
```python
    ophys_t = np.asarray(exp.ophys_timestamps, dtype=np.float64)
    ...
        trial_t = ophys_t[left:right]
    ...
        'metadata': {
            ...
            'time_bin_size': None,
```

iii. Step 5 Key Decision 3: the task instruction "Temporally align based on ophys timestamp" is read as "keep the native ophys sampling," so no binning code was written. No rationale is given for leaving `time_bin_size` as `None`, and the mixed frame rates are never acknowledged.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. `exp.stimulus_presentations['image_name']` for the flash that defines the trial (after dropping `omitted` flashes and inactive blocks). The `trials` table's `initial_image_name` / `change_image_name` columns are not used. 16 unique image names are found (`im000 … im106`), matching the reference's set exactly.

ii.
```python
    stim = stim[~stim['omitted'].fillna(False)].copy()
    stim = stim[stim['image_name'].notna()].copy()
    stim_image_name = stim['image_name'].astype(str).to_numpy()
    ...
        session['trials'].append({
            ...
            'image_name': img_name,
```

iii. Step 4/Step 5: "Use stimulus presentations to construct time-varying image identity and image-change labels within trials"; "Time-varying; use non-grey image presented during each interval." The instruction's parenthetical "(of the image presented during the non-grey screen)" is read as licensing the restriction to image-on periods.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. A single global, deterministic mapping from sorted unique image name → integer 0–15 is built across all sessions after conversion, and each trial's scalar image code is broadcast to every timepoint of the trial with `np.full`. `image_values` (the `output_values` entry) is the sorted name list, so codes are interpretable. An earlier `-1` "no image" sentinel was removed when trialing changed.

ii.
```python
    image_names = sorted({tr['image_name'] for s in processed_sessions for tr in s['trials']})
    image_to_idx = {name: i for i, name in enumerate(image_names)}
    image_values = image_names
    ...
            out = np.vstack([
                np.full(T, image_to_idx[tr['image_name']], dtype=np.int64),
                ...
```

iii. Trajectory step 21: outputs "must contain only valid categorical indices"; restricting trials to image-on periods "removes invalid -1 labels." A global sorted mapping keeps codes consistent across sessions (the verifier reports each of the 16 images at ≈6.1–6.4% of timepoints, matching the reference).

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. The image code is written to the same `T = right - left` frames used to slice the neural matrix, so it is index-aligned by construction. Because a trial *is* one flash, the label is **constant within every trial** — the nominally time-varying variable carries no within-trial temporal information, and the 500 ms grey period after each flash (where the calcium response to that image actually peaks) is not part of the trial.

ii.
```python
        for tr in s['trials']:
            T = tr['neural'].shape[1]
            out = np.vstack([
                np.full(T, image_to_idx[tr['image_name']], dtype=np.int64),
                ...
            ]).astype(np.int64)
            sess_neural.append(tr['neural'])
            sess_output.append(out)
```

iii. Step 7: "Revised conversion uses image-presentation intervals as trials, eliminating invalid labels"; each ophys timestamp within the window falls inside `[start_time, end_time)` of exactly that presentation, so the label is correct at every timepoint by construction.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. `exp.stimulus_presentations['is_change']` — the SDK's per-flash flag that is True on the first presentation of a new image. `trials['change_time']` / `trials['go']` are not used. Because `is_change` is False for sham (catch) changes, catch trials correctly receive 0 throughout, matching the reference.

ii.
```python
    if 'is_change' in stim.columns:
        stim_is_change = stim['is_change'].fillna(False).astype(bool).to_numpy()
    else:
        stim_is_change = np.zeros(len(stim), dtype=bool)
```

iii. Step 5 mapping table: "`stimulus_presentations.is_change` → output[1] image change | Binary time series on ophys timestamps, 1 during image presentation intervals immediately after an image change, else 0 | `Presentations.from_nwb` | Time-varying."

## 4-b. What processing is involved in computing `output` *Image change*?

i. None beyond casting the boolean to `int` and broadcasting it over the trial's frames with `np.full`. The 1 therefore covers exactly the 250 ms change flash (not the following grey period, and not any longer post-change window). Resulting global rate: 7.65% of timepoints are `change`, essentially identical to the reference's 7.65%.

ii.
```python
            'image_change': np.full(right - left, int(is_change), dtype=np.int64),
    ...
            out = np.vstack([
                np.full(T, image_to_idx[tr['image_name']], dtype=np.int64),
                tr['image_change'],
                ...
```

iii. Per Step 5, `is_change` is taken as already encoding "1 right after a change in image identity" at the flash level, so no further processing was considered necessary.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is natively binary: `{0: 'no_change', 1: 'change'}`, obtained by `int()` of the boolean flag. No threshold is involved.

ii.
```python
IMAGE_CHANGE_VALUES = ['no_change', 'change']
...
        'output_values': [
            image_values,
            IMAGE_CHANGE_VALUES,
            ...
```

iii. The instruction specifies a binary variable and `is_change` is already boolean.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same frame range as the neural slice (`right - left` frames starting at flash onset), so it is index-aligned. As with image identity, the value is **constant within each trial**: the variable degenerates from "1 right after the change, 0 otherwise" to a per-trial change/no-change label, and the post-change grey period — where the change response is largest in calcium — is excluded from the trial.

ii.
```python
            'image_change': np.full(right - left, int(is_change), dtype=np.int64),
```

iii. Same as 3-c: labels derive from the presentation row that defines the trial window, so alignment is exact by construction.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. `exp.running_speed`, using its `speed` and `timestamps` columns — the SDK's filtered running speed (cm/s) on the stimulus/behavior clock. This is the same source the reference uses.

ii.
```python
    run_t = exp.running_speed['timestamps'].to_numpy(dtype=np.float64)
    run_v = exp.running_speed['speed'].to_numpy(dtype=np.float32)
```

iii. Step 1 notes: "Running speed loading explicitly supports filtered and unfiltered versions; likely use filtered running_speed for decoder output unless references indicate otherwise." Step 5: "`running_speed.speed` with `running_speed.timestamps` → output[2] running speed bin."

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Linear interpolation onto the trial's ophys timestamps with `np.interp` (non-finite source samples dropped, source sorted by time, out-of-range set to NaN), then global percentile discretization. The interpolation is performed **per trial**, passing the full-session running trace each time.

ii.
```python
def interp_to_ophys(src_t, src_v, dst_t):
    src_t = np.asarray(src_t, dtype=np.float64)
    src_v = np.asarray(src_v, dtype=np.float64)
    dst_t = np.asarray(dst_t, dtype=np.float64)
    good = np.isfinite(src_t) & np.isfinite(src_v)
    if good.sum() < 2:
        return np.full(dst_t.shape, np.nan, dtype=np.float32)
    src_t = src_t[good]; src_v = src_v[good]
    order = np.argsort(src_t)
    src_t = src_t[order]; src_v = src_v[order]
    return np.interp(dst_t, src_t, src_v, left=np.nan, right=np.nan).astype(np.float32)

        run_aligned = interp_to_ophys(run_t, run_v, trial_t)
```

iii. Step 5: "Interpolate or assign running speed onto ophys timestamps, then discretize globally into 5 equal-frequency percentile bins." Linear interpolation is the standard way to bring a ~60 Hz behavioural stream onto the ophys grid.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Five equal-frequency bins. Bin edges are computed **once, globally**, from the concatenation of every kept trial's aligned running values across all retained sessions, using `np.quantile` at 0, .2, .4, .6, .8, 1. Degenerate (non-increasing) edges are nudged by 1e-6. `np.digitize(values, edges[1:-1])` assigns bins 0–4; non-finite values are replaced by the median bin of the finite values *of that same trial* (0 if the trial has none), and the result is clipped to [0, 4]. Realised distribution is exactly 20% per bin. Edges: `[-23.98, -0.0085, 0.217, 15.81, 33.17, 110.44]`, close to the reference's `[-23.98, -0.00063, 0.386, 17.93, 34.41, 110.44]`.

ii.
```python
def compute_bin_edges(values, n_bins=5):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.linspace(0.0, 1.0, n_bins + 1)
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(values, qs)
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-6
    return edges

def digitize_with_edges(values, edges):
    out = np.digitize(values, edges[1:-1], right=False).astype(np.int64)
    bad = ~np.isfinite(values)
    if np.any(bad):
        finite = np.where(np.isfinite(values))[0]
        fill = int(np.median(out[finite])) if finite.size else 0
        out[bad] = fill
    out = np.clip(out, 0, len(edges) - 2)
    return out

    run_edges = compute_bin_edges(np.concatenate(all_run))
```

iii. Step 5 Key Decision 6: "**Running/pupil binning global across included data**: compute percentile bin edges over all valid timepoints in processed sessions to ensure comparable categorical outputs across sessions," which follows the instruction "discretized into five equal percentile bins."

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. It is interpolated directly onto `trial_t = ophys_t[left:right]`, i.e. exactly the timestamps of the neural columns, so alignment is exact with no extra indexing step. Discretization happens later in `assemble_dataset` on the stored aligned array, preserving alignment.

ii.
```python
        trial_t = ophys_t[left:right]
        neural = neural_full[:, left:right].astype(np.float32, copy=False)
        run_aligned = interp_to_ophys(run_t, run_v, trial_t)
        ...
            'running_raw': run_aligned,
    ...
            run_bin = digitize_with_edges(tr['running_raw'], run_edges)
```

iii. Step 4: "Temporal alignment basis … User task explicitly says align based on ophys timestamp → Resample/assign all outputs onto ophys timestamps within each trial." The SDK streams are hardware-synced to a common clock, so interpolation across streams is valid.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. `exp.eye_tracking`, preferring the `pupil_area` column, from which an equivalent-circle diameter is computed as `2*sqrt(area/pi)`; if `pupil_area` is absent it falls back to `sqrt(pupil_width * pupil_height)`. `likely_blink` frames are masked to NaN before interpolation. The reference instead uses `pupil_width` directly.

ii.
```python
def pupil_diameter_series(eye_tracking_df):
    et = eye_tracking_df.copy()
    blink = et['likely_blink'].fillna(False).to_numpy(dtype=bool) if 'likely_blink' in et.columns else np.zeros(len(et), dtype=bool)
    if 'pupil_area' in et.columns:
        area = et['pupil_area'].to_numpy(dtype=np.float64)
        diam = 2.0 * np.sqrt(area / math.pi)
    elif 'pupil_width' in et.columns and 'pupil_height' in et.columns:
        diam = np.sqrt(et['pupil_width'].to_numpy(dtype=np.float64) * et['pupil_height'].to_numpy(dtype=np.float64))
    else:
        raise KeyError('No pupil area/width-height columns available')
    diam[blink] = np.nan
    return et['timestamps'].to_numpy(dtype=np.float64), diam.astype(np.float32)
```

iii. Step 5 Key Decision 7: "**Pupil variable choice**: derive diameter from pupil area if no direct diameter column exists; mask likely blinks and missing values before binning/alignment," with the mapping-table note "prefer geometric diameter from area: `2*sqrt(area/pi)` if direct diameter absent."

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Blink frames → NaN; the NaN samples are then dropped inside `interp_to_ophys` (`good = isfinite(...)`), so the interpolation bridges across blinks rather than propagating them; linear interpolation onto each trial's ophys timestamps; then the same global percentile discretization as running speed. As with running speed, interpolation is redone per trial over the full-session trace.

ii.
```python
    pupil_t, pupil_v = pupil_diameter_series(exp.eye_tracking)
    ...
        pupil_aligned = interp_to_ophys(pupil_t, pupil_v, trial_t)
    ...
    pupil_edges = compute_bin_edges(np.concatenate(all_pupil))
    ...
            pupil_bin = digitize_with_edges(tr['pupil_raw'], pupil_edges)
```

iii. Same rationale as running speed plus blink exclusion (Key Decision 7). The SDK's `likely_blink` flag is taken as the pre-computed blink detector.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Identical machinery to running speed: 5 global equal-frequency bins from `np.quantile` over all kept-trial aligned pupil values, `np.digitize` with the interior edges, non-finite values filled with the trial's median bin, clipped to [0, 4]. Edges: `[17.57, 73.22, 84.02, 94.20, 107.33, 410.63]` (equivalent-circle diameter in pixels; the reference's `pupil_width` edges are on a different scale, `[5.28, 36.28, 41.69, 46.78, 53.27, 240.34]`). Realised distribution 0.210/0.198/0.198/0.198/0.198 — slightly off 20% because NaN frames are median-filled after the edges are computed.

ii.
```python
    pupil_edges = compute_bin_edges(np.concatenate(all_pupil))
    print('pupil bin edges:', pupil_edges)
    ...
            pupil_bin = digitize_with_edges(tr['pupil_raw'], pupil_edges)
```

iii. Same as 5-c (Key Decision 6): global percentile binning for comparability across sessions, per the instruction "discretized into five equal percentile bins."

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Interpolated straight onto `trial_t = ophys_t[left:right]`, the same timestamps as the neural columns, so alignment is exact and shares the trial's frame count.

ii.
```python
        pupil_aligned = interp_to_ophys(pupil_t, pupil_v, trial_t)
        session['pupil_raw_all'].append(pupil_aligned)
        session['trials'].append({
            ...
            'pupil_raw': pupil_aligned,
```

iii. Same as running speed: eye-tracking timestamps are on the same synced clock as the ophys timestamps, so interpolation onto the ophys grid is the correct alignment operation.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. The four mutually exclusive boolean columns of the SDK `trials` table — `hit`, `miss`, `false_alarm`, `correct_reject` — read from the *parent* trial of each flash. Same source and same 0–3 code order as the reference.

ii.
```python
TRIAL_OUTCOME_VALUES = ['hit', 'miss', 'false_alarm', 'correct_reject']

def get_trial_outcome(row):
    if bool(row.get('hit', False)):
        return 0
    if bool(row.get('miss', False)):
        return 1
    if bool(row.get('false_alarm', False)):
        return 2
    if bool(row.get('correct_reject', False)):
        return 3
    return None
```

iii. Step 5 mapping table: "`trials` outcome flags (`hit`, `miss`, `false_alarm`, `correct_reject`) → output[4] trial outcome | Static per-trial categorical label derived from mutually exclusive outcome flags | Include only Go/Catch, exclude Aborted/Auto-rewarded."

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Priority-ordered scan of the four flags → integer 0–3; parent trials matching none of them are **excluded** (rather than given a sentinel). The outcome is looked up per flash through `trials_id` and broadcast constant across the flash's frames, so it is stored as a time-varying row with a single repeated value. Realised distribution — hit 0.187, miss 0.687, false_alarm 0.011, correct_reject 0.115 — matches the reference to three decimals.

ii.
```python
    trials['trial_outcome_idx'] = trials.apply(get_trial_outcome, axis=1)
    trials = trials.loc[trials['trial_outcome_idx'].notna()].copy()
    trial_outcome_map = trials['trial_outcome_idx'].to_dict()
    ...
        outcome = int(trial_outcome_map[parent_id])
    ...
            out = np.vstack([
                ...
                np.full(T, tr['trial_outcome'], dtype=np.int64),
            ]).astype(np.int64)
```

iii. Step 5 Key Decision 8: "**Static trial outcome representation**: store outcome as a per-trial scalar categorical output; decoder format permits static outputs." The value is nonetheless broadcast over time so that all five output rows share one `(5, T)` matrix, and dropping unlabelled trials keeps all class indices valid for the NLL loss.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Handled: boolean SDK columns are `fillna(False)` before use (`go`, `catch`, `aborted`, `auto_rewarded`, `omitted`, `is_change`, `likely_blink`); flashes with null `trials_id`, `image_name`, `start_time` or `end_time` are dropped; blink frames are NaN-masked; non-finite behavioural samples are dropped before interpolation and a stream with <2 usable samples returns all-NaN; out-of-range interpolation returns NaN, and NaN bins are filled with the trial's median bin (0 if the whole trial is NaN); degenerate quantile edges are nudged apart; flash windows with <2 frames and sessions with <2 trials are skipped; ROI traces of inconsistent length raise.

Not handled: there is **no try/except around per-file processing** (unlike the reference, which skips failed sessions), so a single unreadable NWB would abort the entire pooled run; the 46.5% of trials with all-zero neural data are neither flagged nor dropped; and `subject_idx` is truncated from the tail rather than filtered, which silently misaligns subjects with sessions if `assemble_dataset` ever drops a session (currently masked by the upstream ≥2-trial filter, so it is latent rather than active).

ii.
```python
    keep = (trials['go'].fillna(False) | trials['catch'].fillna(False))
    ...
        omitted = False if pd.isna(omitted_val) else bool(omitted_val)
    ...
    diam[blink] = np.nan
    ...
    good = np.isfinite(src_t) & np.isfinite(src_v)
    if good.sum() < 2:
        return np.full(dst_t.shape, np.nan, dtype=np.float32)
    ...
    bad = ~np.isfinite(values)
    if np.any(bad):
        finite = np.where(np.isfinite(values))[0]
        fill = int(np.median(out[finite])) if finite.size else 0
        out[bad] = fill
    ...
        for i in range(1, len(edges)):
            if edges[i] <= edges[i - 1]:
                edges[i] = edges[i - 1] + 1e-6
    ...
    for s in processed_sessions:
        if not s['trials']:
            continue
        ...
        if len(sess_neural) >= 2:
            ...
        else:
            data['brain_region_idx'].pop()
    data['subject_idx'] = data['subject_idx'][:len(data['neural'])]
```

iii. CONVERSION_NOTES.md Step 6: "Handle missing data appropriately" is claimed but the notes give no per-case rationale. Step 10 ("Critical Review 1"), which was supposed to hunt edge cases and run raw-data sanity checks with `np.allclose`, was left at `IN PROGRESS` with the template placeholders `1. [Check]: [Result]` and `- [Issue]: [Resolution]`, so none of the five planned sanity checks from Step 5 were executed.

## 9-a. What are the most time-consuming steps of the code?

i. `process_experiment` dominates: 50–80 s per NWB file, logged per experiment. Two costs are inside it — (1) `BehaviorOphysExperiment.from_nwb_path` + materialising the full event matrix (I/O bound), and (2) the ~2,000–4,600 iteration per-flash loop, each iteration of which calls `interp_to_ophys` twice over the *entire* session running/pupil trace (an `isfinite` pass plus an `argsort` of ~10⁵ samples per call, ~6,000–9,000 times per session). The AI mitigated wall-clock with an 8-worker process pool; the full run over 284 files took roughly 35–45 minutes. Assembly, discretization and pickling (a 4.96 GB file) are secondary.

ii.
```python
def process_experiment(nwb_path):
    t0 = time.time()
    exp = BehaviorOphysExperiment.from_nwb_path(str(nwb_path))
    ...
    dt = time.time() - t0
    print(f'processed experiment {session["session_id"]} with {len(session["trials"])} image-presentation trials in {dt:.2f}s')

    use_parallel = (not sample) and len(files) > 4
    if use_parallel:
        n_workers = min(max((os.cpu_count() or 2) // 2, 2), 8)
```

iii. CONVERSION_NOTES.md Step 6: "Code inefficiencies identified: Current implementation iterates over stimulus presentations per trial and loads experiments serially." Step 7 estimated "~22 s/session … ~104 minutes for 284 sessions before optimization", which motivated adding the pool. Only whole-experiment timing is printed; no sub-step profiling was done, so the repeated interpolation was never measured.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. (1) The per-flash loop's two `interp_to_ophys` calls: the running and pupil traces could be interpolated once per session onto the full `ophys_timestamps` vector and then sliced with `[left:right]` (exactly what the reference does), removing thousands of redundant sorts. (2) The `left`/`right` boundaries could be obtained for all flashes at once with a single vectorised `np.searchsorted(ophys_t, stim_start)` / `np.searchsorted(ophys_t, stim_end)`. (3) `trials.apply(get_trial_outcome, axis=1)` is a row-wise Python apply that could be a vectorised `np.select` / `argmax` over the four boolean columns. (4) The unused `build_image_labels_for_trial` uses `stim_df.iterrows()`. None of these were vectorised.

ii.
```python
    for stim_id, parent_id, start, stop, img_name, is_change in zip(...):
        left = np.searchsorted(ophys_t, start, side='left')
        right = np.searchsorted(ophys_t, stop, side='left')
        ...
        run_aligned = interp_to_ophys(run_t, run_v, trial_t)
        pupil_aligned = interp_to_ophys(pupil_t, pupil_v, trial_t)
    ...
    trials['trial_outcome_idx'] = trials.apply(get_trial_outcome, axis=1)
    ...
    for _, srow in stim_df.iterrows():   # build_image_labels_for_trial (never called)
```

iii. The AI listed "iterates over stimulus presentations per trial" as a known inefficiency in Step 6 but chose process-level parallelism instead of vectorisation ("Code speedups added: Uses direct slicing on native ophys timestamps; Stores neural traces as float32; Reuses one pass over sessions to collect global binning statistics"). No justification is given for leaving the loop in place.

## 9-c. What processing does the code repeat multiple times?

i. (1) Re-interpolation of the whole-session running and pupil traces once per flash (thousands of times per session) — the `isfinite` filter and `argsort` of the source arrays are recomputed identically every time. (2) The aligned running/pupil arrays are stored twice per trial — once in `session['trials'][i]['running_raw'/'pupil_raw']` and again in `session['running_raw_all'/'pupil_raw_all']` — doubling the behavioural memory footprint. (3) `np.asarray`/`float64` casts of the same source vectors on every call. (4) `sorted(...)` over all trials of all sessions is re-run to build the image name set. Files themselves are read only once, which is correct.

ii.
```python
    session = {..., 'running_raw_all': [], 'pupil_raw_all': []}
    ...
        session['running_raw_all'].append(run_aligned)
        session['pupil_raw_all'].append(pupil_aligned)
        session['trials'].append({
            ...
            'running_raw': run_aligned,
            'pupil_raw': pupil_aligned,
```

iii. The duplicate storage is intentional per Step 6 ("Reuses one pass over sessions to collect global binning statistics") — it lets bin edges be computed without a second read of the raw data. The repeated interpolation is not justified anywhere; it appears to be an oversight from moving alignment inside the trial loop.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. (1) `build_image_labels_for_trial` is fully implemented (per-frame image identity and change labels over a stimulus dataframe) but **never called** — dead code left in the deliverable, and it is the routine that would have been needed for genuinely time-varying labels. (2) Each trial dict carries `ophys_timestamps` (a float32 copy of the frame times), `start_time`, `stop_time`, `trial_id`, `go`, `catch`, none of which reach the output pickle. (3) The entire `exp.metadata` dict is copied into `session['meta']` and never used. (4) `session['running_raw_all']`/`pupil_raw_all` duplicate data already held per trial. (5) An empty `np.zeros((0, T))` input array is allocated for each of 963,758 trials although `input_names` is empty. (6) `exp.dff_traces` is loaded as part of the NWB object but never used. (7) `interp_to_ophys` upcasts everything to float64 before downcasting to float32. These items are also why the intermediate in-memory session objects (and the 4.96 GB pickle) are larger than necessary.

ii.
```python
def build_image_labels_for_trial(trial_timestamps, stim_df, image_to_idx):   # never called
    ...

        session['trials'].append({
            'trial_id': ...,
            'start_time': float(start),
            'stop_time': float(stop),
            'ophys_timestamps': trial_t.astype(np.float32),
            ...
            'go': bool(trial_go_map[parent_id]),
            'catch': bool(trial_catch_map[parent_id]),
        })
    session = {..., 'meta': dict(meta), ...}
    ...
            sess_input.append(np.zeros((0, T), dtype=np.float32))
```

iii. No justification is offered; the notes' Step 13 (Documentation and Cleanup) is marked `NOT STARTED`, and the trajectory shows the AI stopped after Step 11 with "some later documentation/cleanup steps may be lighter than ideal." The extra per-trial fields appear to be leftovers from the original SDK-trial implementation that were never pruned when the trialing scheme changed.
