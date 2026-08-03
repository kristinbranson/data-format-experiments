# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI walks the local dataset tree under `data/visual-behavior-ophys-1.1.0`, finds every `.nwb` file, and loads each file directly with `BehaviorOphysExperiment.from_nwb_path`. In full mode it processes all 284 NWB files, optionally in parallel with `multiprocessing.Pool`.

ii. ```python
def list_nwb_files():
    files = sorted(DATASET_ROOT.rglob('*.nwb'))
    if not files:
        raise FileNotFoundError(f'No NWB files found under {DATASET_ROOT}')
    return files

def process_experiment(nwb_path):
    exp = BehaviorOphysExperiment.from_nwb_path(str(nwb_path))
```

iii. In `CONVERSION_NOTES.md` Step 2, the agent says the supplied data are organized as local per-experiment NWB files. In Step 4-6, it treats those NWB files as the primary loading unit and later adds multiprocessing because the per-file load/process pass was the main runtime cost.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are taken from each experiment’s `metadata['mouse_id']`. After processing, the AI builds a sorted unique subject list and maps each kept session to a subject index.

ii. ```python
session = {
    'session_id': int(meta['ophys_experiment_id']),
    'subject': str(meta['mouse_id']),
    'region': str(meta['targeted_structure']),
    ...
}

subjects = sorted({s['subject'] for s in processed_sessions})
subject_to_idx = {s: i for i, s in enumerate(subjects)}
```

iii. The notes’ Step 5 mapping explicitly says `metadata['mouse_id']` should drive `subjects` and `subject_idx`.

## 1-c. How are the data split into sessions?

i. Each NWB file, i.e. each `ophys_experiment_id`, is treated as one decoder session. The AI does not group multiple experiments by `ophys_session_id`.

ii. ```python
session = {
    'session_id': int(meta['ophys_experiment_id']),
    'subject': str(meta['mouse_id']),
    'region': str(meta['targeted_structure']),
    ...
}
```

iii. In `CONVERSION_NOTES.md` Step 4 and Step 5, the agent explicitly justifies this as “Treat each ophys experiment NWB as one decoder session because neural populations are experiment-specific.”

## 1-d. How are the data split into trials?

i. Trials are not taken from the Allen SDK `trials` table window. Instead, the AI uses `stimulus_presentations` rows linked to valid parent trial IDs and treats each image-presentation interval (`start_time` to `end_time`) as a trial.

ii. ```python
stim = exp.stimulus_presentations.copy().sort_values('start_time')
stim = stim[stim['trials_id'].notna()].copy()
stim = stim[stim['trials_id'].isin(valid_trial_ids)].copy()
...
for stim_id, parent_id, start, stop, img_name, is_change in zip(
    stim_ids, stim_trial_ids, stim_start, stim_end, stim_image_name, stim_is_change
):
    left = np.searchsorted(ophys_t, start, side='left')
    right = np.searchsorted(ophys_t, stop, side='left')
```

iii. The notes’ Step 3-4 say the paper’s analyses are “image-by-image” and use 750 ms image-presentation intervals. In Step 7 the agent says it revised away from full trial segmentation because grey periods produced unlabeled image-identity bins, so it switched to image-presentation trialing.

## 1-e. How are trials filtered based on quality controls?

i. The AI first filters parent SDK trials to `go` or `catch`, excluding `aborted` and `auto_rewarded`, and requiring a recognizable trial outcome. It then filters `stimulus_presentations` to those attached to the surviving parent trials, and further keeps only active, non-omitted stimuli with non-null `image_name`, `start_time`, and `end_time`. Stimulus intervals shorter than 2 ophys frames are skipped, and sessions with fewer than 2 kept intervals are dropped.

ii. ```python
def valid_trials_df(trials):
    keep = (trials['go'].fillna(False) | trials['catch'].fillna(False))
    keep &= ~trials['aborted'].fillna(False)
    keep &= ~trials['auto_rewarded'].fillna(False)
    trials = trials.loc[keep].copy()
    trials['trial_outcome_idx'] = trials.apply(get_trial_outcome, axis=1)
    trials = trials.loc[trials['trial_outcome_idx'].notna()].copy()
    return trials

stim = stim[stim['trials_id'].isin(valid_trial_ids)].copy()
if 'active' in stim.columns:
    stim = stim[stim['active'].fillna(False)].copy()
if 'omitted' in stim.columns:
    stim = stim[~stim['omitted'].fillna(False)].copy()
stim = stim[stim['image_name'].notna()].copy()
stim = stim[stim['start_time'].notna() & stim['end_time'].notna()].copy()
...
if right - left < 2:
    continue
```

iii. The notes say Go/Catch should be included and Aborted/Auto-rewarded excluded. They also say the final switch to image-presentation trials was motivated by wanting every kept interval to have a valid non-grey image label.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from `exp.events['events']`, i.e. calcium event traces, not from `dff_traces`.

ii. ```python
def extract_events_matrix(events_df):
    event_col = 'events'
    arrs = [np.asarray(x, dtype=np.float32) for x in events_df[event_col].values]
    return np.stack(arrs, axis=0)

neural_full = extract_events_matrix(exp.events)
```

iii. In Step 3-5 of the notes, the agent repeatedly cites `methods.txt` saying “For all analysis of neural data we used the detected calcium events,” and uses that to justify preferring events over dF/F.

## 2-b. How is the `neural` data processed?

i. The AI stacks the per-ROI event traces into a neuron-by-time matrix for one experiment, then slices that matrix by each kept stimulus interval. It casts slices to `float32`. It does not merge multiple imaging planes into one session because each experiment is already treated as a separate session.

ii. ```python
neural_full = extract_events_matrix(exp.events)
...
left = np.searchsorted(ophys_t, start, side='left')
right = np.searchsorted(ophys_t, stop, side='left')
...
neural = neural_full[:, left:right].astype(np.float32, copy=False)
```

iii. The notes say the signal should be kept on ophys timestamps and that “session unit = ophys experiment.” That combination leads directly to stack-then-slice processing.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No explicit neuron-level quality filter is applied in `convert_data.py`. The script uses whatever ROIs are present in `exp.events`.

ii. ```python
neural_full = extract_events_matrix(exp.events)
...
neural = neural_full[:, left:right].astype(np.float32, copy=False)
```

iii. The notes mention relying on the SDK-provided experiment object and “SDK-provided valid cell/ROI tables,” but the final script itself does not perform an extra neuron-selection step.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to the ophys timestamps that fall inside each kept image-presentation interval. The alignment event is therefore stimulus interval onset, not parent trial start.

ii. ```python
left = np.searchsorted(ophys_t, start, side='left')
right = np.searchsorted(ophys_t, stop, side='left')
trial_t = ophys_t[left:right]
neural = neural_full[:, left:right].astype(np.float32, copy=False)
```

iii. The notes say the task should align on ophys timestamps, and after the agent switched to image-presentation trialing it used each presentation’s `start_time`/`end_time` as the window boundaries.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI keeps the native ophys frame sampling within each image-presentation interval and does not explicitly rebin. However, it leaves `metadata['time_bin_size']` as `None` instead of recording the actual frame interval.

ii. ```python
trial_t = ophys_t[left:right]
...
'metadata': {
    ...
    'time_bin_size': None,
    'temporal_alignment_event': 'native ophys timestamps within each trial defined by SDK trial start/stop times',
    ...
}
```

iii. The notes say “Temporal basis = ophys timestamps” and frame the output as native-time ophys slices. There is no note justifying why `time_bin_size` was left unset; that appears to be an omission in the final code.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from `stimulus_presentations.image_name`.

ii. ```python
stim_image_name = stim['image_name'].astype(str).to_numpy()
...
session['trials'].append({
    ...
    'image_name': img_name,
    ...
})
```

iii. The notes’ Step 4-5 state that `stimulus_presentations` should be the source for time-varying image labels, especially after the switch to image-presentation trials.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. The AI collects all unique image names across processed sessions, sorts them, maps them to integer category IDs, and then fills each kept trial with one constant image ID for the full interval.

ii. ```python
image_names = sorted({tr['image_name'] for s in processed_sessions for tr in s['trials']})
image_to_idx = {name: i for i, name in enumerate(image_names)}
...
out = np.vstack([
    np.full(T, image_to_idx[tr['image_name']], dtype=np.int64),
    tr['image_change'],
    ...
])
```

iii. The trajectory and Step 7 notes say the AI deliberately redefined trials to be single image-presentation intervals so that image identity would never pass through unlabeled grey-screen periods.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is aligned by using the exact same ophys slice as the neural data for each stimulus interval, then assigning one categorical image label across that whole slice.

ii. ```python
trial_t = ophys_t[left:right]
neural = neural_full[:, left:right].astype(np.float32, copy=False)
...
np.full(T, image_to_idx[tr['image_name']], dtype=np.int64)
```

iii. The AI’s justification is the same image-presentation-trial rationale: once the trial is one non-grey stimulus interval, a single image label aligned over the same ophys timestamps is sufficient.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from `stimulus_presentations.is_change`.

ii. ```python
if 'is_change' in stim.columns:
    stim_is_change = stim['is_change'].fillna(False).astype(bool).to_numpy()
...
'image_change': np.full(right - left, int(is_change), dtype=np.int64),
```

iii. The notes say `stimulus_presentations.is_change` should provide the per-image interval change label for image-by-image analyses.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The AI performs no extra temporal construction around `change_time`. It simply turns the `is_change` flag for a stimulus presentation into an integer and repeats that value across all ophys frames in the kept interval.

ii. ```python
session['trials'].append({
    ...
    'image_change': np.full(right - left, int(is_change), dtype=np.int64),
    ...
})
```

iii. The justification comes from the same Step 3-5 image-presentation framing: if trials are already single image intervals, the per-interval `is_change` flag is treated as the correct change label.

## 4-c. How is `output` *Image change* thresholded into categories?

i. There is no continuous thresholding step. The boolean `is_change` value is cast directly into the two categories `0/1`, with `output_values` naming them `no_change` and `change`.

ii. ```python
IMAGE_CHANGE_VALUES = ['no_change', 'change']
...
'image_change': np.full(right - left, int(is_change), dtype=np.int64),
```

iii. The AI treats image change as an already discrete field coming from the stimulus table, so no additional thresholding is documented.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Image change is aligned over the same ophys frames used for the trial’s neural slice, with one constant 0/1 label per kept image-presentation interval.

ii. ```python
trial_t = ophys_t[left:right]
neural = neural_full[:, left:right].astype(np.float32, copy=False)
...
'image_change': np.full(right - left, int(is_change), dtype=np.int64),
```

iii. Again, the notes justify this by treating the image-presentation interval itself as the trial/alignment window.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed comes from `exp.running_speed['timestamps']` and `exp.running_speed['speed']`.

ii. ```python
run_t = exp.running_speed['timestamps'].to_numpy(dtype=np.float64)
run_v = exp.running_speed['speed'].to_numpy(dtype=np.float32)
```

iii. The notes’ Step 5 mapping explicitly maps the experiment object’s running speed timestamps and speed values to the running-speed decoder output.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. For each kept interval, the AI linearly interpolates running speed onto that interval’s ophys timestamps. After all sessions are processed, it computes global 5-quantile bin edges from all aligned running values and digitizes each trial’s running trace with those edges.

ii. ```python
run_aligned = interp_to_ophys(run_t, run_v, trial_t)
...
run_edges = compute_bin_edges(np.concatenate(all_run))
...
run_bin = digitize_with_edges(tr['running_raw'], run_edges)
```

iii. The notes say the outputs should be aligned to ophys timestamps and that running speed should be globally discretized into five equal-frequency bins across the included data.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. The AI computes five quantile bin edges over all finite running samples, then applies `np.digitize`. Non-finite values are filled with the median occupied bin for that trial if possible, otherwise 0.

ii. ```python
def compute_bin_edges(values, n_bins=5):
    values = values[np.isfinite(values)]
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(values, qs)
    ...

def digitize_with_edges(values, edges):
    out = np.digitize(values, edges[1:-1], right=False).astype(np.int64)
    bad = ~np.isfinite(values)
    if np.any(bad):
        finite = np.where(np.isfinite(values))[0]
        fill = int(np.median(out[finite])) if finite.size else 0
        out[bad] = fill
```

iii. The notes describe “5 equal-frequency percentile bins.” They do not specifically justify the median-bin fill for missing values; that choice is only visible in the code.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is aligned by interpolation onto the exact per-trial ophys timestamps used to slice the neural data.

ii. ```python
trial_t = ophys_t[left:right]
neural = neural_full[:, left:right].astype(np.float32, copy=False)
run_aligned = interp_to_ophys(run_t, run_v, trial_t)
```

iii. The notes repeatedly state that all outputs should be aligned to the ophys time base.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `exp.eye_tracking`. The AI prefers `pupil_area` and converts it to a diameter with `2*sqrt(area/pi)`. If `pupil_area` is unavailable, it falls back to `sqrt(pupil_width * pupil_height)`. Frames flagged by `likely_blink` are set to `NaN`.

ii. ```python
def pupil_diameter_series(eye_tracking_df):
    blink = et['likely_blink'].fillna(False).to_numpy(dtype=bool)
    if 'pupil_area' in et.columns:
        area = et['pupil_area'].to_numpy(dtype=np.float64)
        diam = 2.0 * np.sqrt(area / math.pi)
    elif 'pupil_width' in et.columns and 'pupil_height' in et.columns:
        diam = np.sqrt(et['pupil_width'].to_numpy(dtype=np.float64) *
                       et['pupil_height'].to_numpy(dtype=np.float64))
    ...
    diam[blink] = np.nan
```

iii. In Step 5 the notes explicitly say “prefer geometric diameter from area: `2*sqrt(area/pi)` if direct diameter absent,” because the task asked for pupil diameter rather than a generic eye-tracking scalar.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. The AI computes a diameter proxy from eye-tracking columns, masks blink frames, linearly interpolates the result onto each trial’s ophys timestamps, then computes global 5-quantile bin edges and digitizes the aligned signal.

ii. ```python
pupil_t, pupil_v = pupil_diameter_series(exp.eye_tracking)
...
pupil_aligned = interp_to_ophys(pupil_t, pupil_v, trial_t)
...
pupil_edges = compute_bin_edges(np.concatenate(all_pupil))
...
pupil_bin = digitize_with_edges(tr['pupil_raw'], pupil_edges)
```

iii. The notes justify this as the pupil analogue of the running-speed pipeline, with added blink masking and a diameter conversion step.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Pupil diameter is binned exactly like running speed: five global quantile bins with `np.digitize`, and non-finite values are filled with the median occupied bin for that trial if available.

ii. ```python
pupil_edges = compute_bin_edges(np.concatenate(all_pupil))
...
pupil_bin = digitize_with_edges(tr['pupil_raw'], pupil_edges)
```

iii. The notes describe global five-bin percentile discretization but do not separately justify the median-bin fill behavior for missing pupil values.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is aligned by interpolation onto the same per-trial ophys timestamps as the neural slice.

ii. ```python
trial_t = ophys_t[left:right]
neural = neural_full[:, left:right].astype(np.float32, copy=False)
pupil_aligned = interp_to_ophys(pupil_t, pupil_v, trial_t)
```

iii. The notes say all continuous behavioral streams should be resampled or assigned onto ophys timestamps before forming decoder outputs.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the SDK trial-table booleans `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii. ```python
def get_trial_outcome(row):
    if bool(row.get('hit', False)):
        return 0
    if bool(row.get('miss', False)):
        return 1
    if bool(row.get('false_alarm', False)):
        return 2
    if bool(row.get('correct_reject', False)):
        return 3
```

iii. The notes’ Step 5 mapping explicitly identifies those four trial-table outcome flags as the source for the static trial-outcome label.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The AI maps the mutually exclusive outcome booleans to integer codes 0-3 during trial filtering, stores the code per kept trial, and then repeats that code across all time bins when assembling the final output matrix.

ii. ```python
trials['trial_outcome_idx'] = trials.apply(get_trial_outcome, axis=1)
...
'trial_outcome': outcome,
...
np.full(T, tr['trial_outcome'], dtype=np.int64),
```

iii. The notes say trial outcome should be a static per-trial categorical output, and the code implements that as a constant row over time.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing or invalid behavioral samples are mostly handled through interpolation and fallback categories. `interp_to_ophys` drops non-finite source samples and returns all-`NaN` if too little source data exist; later `digitize_with_edges` replaces non-finite outputs with a trial-local median bin (or 0 if no finite values exist). Blink frames are masked to `NaN` before interpolation. Stimulus rows with missing `trials_id`, `image_name`, `start_time`, or `end_time` are dropped, and intervals shorter than two ophys frames are skipped. Sessions with fewer than two kept trials are excluded.

ii. ```python
good = np.isfinite(src_t) & np.isfinite(src_v)
if good.sum() < 2:
    return np.full(dst_t.shape, np.nan, dtype=np.float32)
...
diam[blink] = np.nan
...
stim = stim[stim['trials_id'].notna()].copy()
stim = stim[stim['image_name'].notna()].copy()
stim = stim[stim['start_time'].notna() & stim['end_time'].notna()].copy()
...
if right - left < 2:
    continue
...
if np.any(bad):
    finite = np.where(np.isfinite(values))[0]
    fill = int(np.median(out[finite])) if finite.size else 0
```

iii. The notes emphasize sensible handling of blink/missing data and avoiding unlabeled intervals. They do not mention a try/except session-level recovery path; the final script instead assumes each NWB file can be loaded successfully.

## 9-a. What are the most time-consuming steps of the code?

i. The expensive path is loading and processing each NWB experiment with `BehaviorOphysExperiment.from_nwb_path`, then iterating through thousands of image-presentation intervals per experiment. The full-run log shows roughly 15-27 seconds per experiment before multiprocessing.

ii. ```python
def process_experiment(nwb_path):
    t0 = time.time()
    exp = BehaviorOphysExperiment.from_nwb_path(str(nwb_path))
    ...
    for stim_id, parent_id, start, stop, img_name, is_change in zip(...):
        ...
    dt = time.time() - t0
    print(f'processed experiment {session["session_id"]} with {len(session["trials"])} image-presentation trials in {dt:.2f}s')
```

iii. In Step 6-7 the notes identify iteration over stimulus presentations and full-session loading as the runtime bottlenecks, and the agent added multiprocessing in response.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The clearest candidates are the `trials.apply(get_trial_outcome, axis=1)` pass and the per-stimulus loop in `process_experiment`, which repeatedly does `searchsorted`, interpolation, and dictionary assembly once per image interval. If the unused helper `build_image_labels_for_trial` were active, its row-wise `iterrows()` loop would be another candidate.

ii. ```python
trials['trial_outcome_idx'] = trials.apply(get_trial_outcome, axis=1)
...
for stim_id, parent_id, start, stop, img_name, is_change in zip(
    stim_ids, stim_trial_ids, stim_start, stim_end, stim_image_name, stim_is_change
):
    left = np.searchsorted(ophys_t, start, side='left')
    right = np.searchsorted(ophys_t, stop, side='left')
    run_aligned = interp_to_ophys(run_t, run_v, trial_t)
    pupil_aligned = interp_to_ophys(pupil_t, pupil_v, trial_t)
```

iii. The notes explicitly say “Current implementation iterates over stimulus presentations per trial,” and list that as an inefficiency.

## 9-c. What processing does the code repeat multiple times?

i. The code repeats file discovery (`list_nwb_files()` is called once inside `choose_files()` and again for the status print), repeats DataFrame copying/filtering over `stimulus_presentations`, and most importantly re-interpolates running speed and pupil separately for every image-presentation interval rather than interpolating once for the full experiment and then slicing. It also keeps raw aligned running/pupil traces, then digitizes them later in a second pass during dataset assembly.

ii. ```python
files = choose_files(sample=sample)
print(f'found {len(list_nwb_files())} nwb files; processing {len(files)}')
...
stim = exp.stimulus_presentations.copy().sort_values('start_time')
stim = stim[stim['trials_id'].notna()].copy()
stim = stim[stim['trials_id'].isin(valid_trial_ids)].copy()
...
run_aligned = interp_to_ophys(run_t, run_v, trial_t)
pupil_aligned = interp_to_ophys(pupil_t, pupil_v, trial_t)
...
run_bin = digitize_with_edges(tr['running_raw'], run_edges)
pupil_bin = digitize_with_edges(tr['pupil_raw'], pupil_edges)
```

iii. The notes highlight that one-pass global bin-edge collection avoided a full reload, but the final code still repeats several smaller processing steps inside each experiment.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script stores a lot of intermediate per-trial and per-session state that never reaches the saved dataset: `meta`, `start_time`, `stop_time`, `ophys_timestamps`, `go`, `catch`, and the raw aligned running/pupil arrays are all kept in `processed` session dictionaries but discarded when `assemble_dataset` builds the final pickle. The module also contains unused objects (`PROJECT_METADATA`, `build_image_labels_for_trial`) that do not contribute to the output.

ii. ```python
session = {
    'session_id': int(meta['ophys_experiment_id']),
    'subject': str(meta['mouse_id']),
    'region': str(meta['targeted_structure']),
    'meta': dict(meta),
    'trials': [],
    'running_raw_all': [],
    'pupil_raw_all': [],
}
...
session['trials'].append({
    'start_time': float(start),
    'stop_time': float(stop),
    'ophys_timestamps': trial_t.astype(np.float32),
    ...
    'go': bool(trial_go_map[parent_id]),
    'catch': bool(trial_catch_map[parent_id]),
})
...
with open(args.outpicklefile, 'wb') as f:
    pickle.dump(data, f)
```

iii. The notes focus on correctness and plotting rather than memory minimization. The final code keeps these intermediates long enough for bin-edge computation and optional plotting, then drops them implicitly when only `data` is pickled.
