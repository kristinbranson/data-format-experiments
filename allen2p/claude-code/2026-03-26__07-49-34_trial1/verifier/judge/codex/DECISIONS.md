# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data through the Allen SDK S3 cache, not from the local NWB files. It creates a `VisualBehaviorOphysProjectCache`, reads the full ophys experiment table, filters to `project_code == 'VisualBehavior'`, and then loads each experiment in a session with `get_behavior_ophys_experiment()`.

ii.
```python
def get_cache(cache_dir):
    return bpc.VisualBehaviorOphysProjectCache.from_s3_cache(cache_dir=cache_dir)
...
bc = get_cache(args.datadir)
experiment_table = bc.get_ophys_experiment_table()
vb_experiments = experiment_table[experiment_table.project_code == PROJECT_CODE]
...
for exp_id in session_experiments.index:
    datasets[exp_id] = bc.get_behavior_ophys_experiment(exp_id)
```

iii. The justification is only partly explicit. [CONVERSION_NOTES.md] says the SDK cache is the "main entry point for data access" and treats the experiment table as the canonical source; the trajectory also shows the agent intentionally pivoted to the Allen SDK workflow rather than loading local NWBs directly.

## 1-b. How are the data split into subjects?

i. Subjects are unique `mouse_id` values from the Allen SDK experiment table.

ii.
```python
all_mouse_ids = sorted(vb_experiments.mouse_id.unique())
...
if mouse_id not in mouse_to_subj_idx:
    mouse_to_subj_idx[mouse_id] = len(subjects)
    subjects.append(str(mouse_id))
```

iii. The justification is explicit in the notes: `mouse_id` is treated as the SDK’s subject identifier for each animal.

## 1-c. How are the data split into sessions?

i. Sessions are grouped by `ophys_session_id`. For each mouse, experiments sharing the same `ophys_session_id` are combined into one session, sorted by `date_of_acquisition`.

ii.
```python
mouse_sessions = mouse_exps.drop_duplicates(subset='ophys_session_id')[
    ['ophys_session_id', 'session_type', 'date_of_acquisition']
].sort_values(by='date_of_acquisition')
...
sess_exps = mouse_exps[mouse_exps.ophys_session_id == sid]
session_data = extract_session_data(bc, sess_exps)
```

iii. The notes justify this by saying multiple experiments can be different imaging planes from the same behavioral session and therefore should be reconstructed via shared `ophys_session_id`.

## 1-d. How are the data split into trials?

i. Trials come from the Allen SDK `trials` table. For each valid row, the AI takes all ophys frames from `start_time` to `stop_time`, producing variable-length trials.

ii.
```python
trials_table = ref_ds.trials
...
for _, row in valid_trials.iterrows():
    start_idx = np.searchsorted(ophys_ts, row['start_time'])
    end_idx = np.searchsorted(ophys_ts, row['stop_time'])
    ...
    idx = np.arange(start_idx, end_idx)
```

iii. The justification in [CONVERSION_NOTES.md] is that the built-in trials table already contains the needed behavioral segmentation and that using the full trial window preserves pre-change and post-change periods for time-varying outputs.

## 1-e. How are trials filtered based on quality controls?

i. The AI excludes aborted trials, auto-rewarded trials, and trials with missing `change_time`. It skips empty frame windows and drops sessions with fewer than two retained trials.

ii.
```python
valid_trials = trials_table[
    (~trials_table['aborted']) &
    (~trials_table['auto_rewarded']) &
    (trials_table['change_time'].notna())
]
...
if end_idx > T:
    end_idx = T
if end_idx <= start_idx:
    continue
...
if len(trials) < 2:
    continue
```

iii. The notes explicitly justify excluding aborted and auto-rewarded trials per task instructions. The `change_time.notna()` requirement is not separately justified in the notes; it appears to be an implementation choice so that every retained trial has a defined change alignment point.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derives neural data from each experiment’s `dff_traces.dff` values.

ii.
```python
for exp_id, ds in datasets.items():
    dff_list.append(np.vstack(ds.dff_traces.dff.values))
```

iii. The notes explicitly justify using dF/F because it is the standard precomputed calcium-imaging signal provided by Allen.

## 2-b. How is the `neural` data processed?

i. The AI vertically stacks dF/F traces from all experiments/planes belonging to the same `ophys_session_id`. It also creates per-neuron plane labels using area and imaging depth.

ii.
```python
dff_list = []
plane_labels = []
for exp_id, ds in datasets.items():
    dff_list.append(np.vstack(ds.dff_traces.dff.values))
    area = session_experiments.loc[exp_id, 'targeted_structure']
    depth = session_experiments.loc[exp_id, 'imaging_depth']
    plane_labels.extend([f'{area}_{depth}um'] * len(ds.dff_traces))
neural_data = np.vstack(dff_list)
```

iii. The notes justify this as reconstructing a session from multiple imaging planes while preserving anatomical identity in `plane_labels`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional neural QC is applied in `convert_data.py`. Every ROI present in `ds.dff_traces` is included.

ii.
```python
for exp_id, ds in datasets.items():
    dff_list.append(np.vstack(ds.dff_traces.dff.values))
```

iii. The notes say the Allen pipeline has already applied its own preprocessing/QC and that no extra filtering was deemed necessary.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial’s neural data is aligned to trial start in the sense that frames are sliced from `start_time` to `stop_time` on the ophys timestamp axis.

ii.
```python
start_idx = np.searchsorted(ophys_ts, row['start_time'])
end_idx = np.searchsorted(ophys_ts, row['stop_time'])
idx = np.arange(start_idx, end_idx)
...
'neural': neural_data[:, idx].astype(np.float32),
```

iii. The notes justify using the ophys timestamps directly and retaining the full trial window so outputs can vary across the trial.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No temporal rebinning is applied. The data remain on the native ophys frame grid, and `time_bin_size` is taken from the median frame interval of the first session.

ii.
```python
first_ophys_ts = session_results[0][1]['ophys_ts']
time_bin_size_ms = float(np.median(np.diff(first_ophys_ts)) * 1000)
```

iii. The notes justify this by treating the ophys frame times as the native synchronized clock for all outputs after interpolation.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from `initial_image_name`, `change_image_name`, and `change_time` in the trials table.

ii.
```python
change_idx = np.searchsorted(ophys_ts[idx], row['change_time'])
image_names = np.empty(n_frames, dtype=object)
image_names[:change_idx] = row['initial_image_name']
image_names[change_idx:] = row['change_image_name']
```

iii. The notes justify this as a pre-change versus post-change representation within each trial; catch trials are assumed to have the same image before and after the sham change.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. The AI constructs a per-frame string array within each trial, then later converts all image names to integer codes using a global sorted mapping across sessions.

ii.
```python
image_names = np.empty(n_frames, dtype=object)
image_names[:change_idx] = row['initial_image_name']
image_names[change_idx:] = row['change_image_name']
...
all_image_names = sorted(all_image_names)
image_to_code = {name: i for i, name in enumerate(all_image_names)}
...
image_row = np.array([image_to_code[name] for name in t['image_names']], dtype=np.int8)
```

iii. The notes explicitly justify the deterministic global mapping so the same image has the same category in every session.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is assigned on the same trial frame index array `idx` used for neural slicing. The switch from initial image to changed image occurs at the first trial frame at or after `change_time`.

ii.
```python
change_idx = np.searchsorted(ophys_ts[idx], row['change_time'])
image_names[:change_idx] = row['initial_image_name']
image_names[change_idx:] = row['change_image_name']
```

iii. The justification is implicit in the code and notes: because the same ophys-frame indices are used for neural and output arrays, they are treated as aligned.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from `change_time` plus the `go` flag in the trials table.

ii.
```python
image_change = np.zeros(n_frames, dtype=np.int8)
if row['go']:
    change_end_idx = np.searchsorted(ophys_ts[idx], row['change_time'] + 0.75)
    image_change[change_idx:change_end_idx] = 1
```

iii. The notes justify this as marking real image changes only on go trials and leaving catch trials at zero.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The AI creates a binary vector and sets it to 1 from `change_time` through the next 750 ms, corresponding to one 250 ms flash plus the following 500 ms gray interval.

ii.
```python
image_change = np.zeros(n_frames, dtype=np.int8)
if row['go']:
    change_end_idx = np.searchsorted(ophys_ts[idx], row['change_time'] + 0.75)
    image_change[change_idx:change_end_idx] = 1
```

iii. [CONVERSION_NOTES.md] explicitly explains the 750 ms window as flash plus inter-stimulus gray period.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is converted directly into binary categories: `0` for no change and `1` for change. There is no further thresholding.

ii.
```python
image_change = np.zeros(n_frames, dtype=np.int8)
...
output_values': [run_value_names, pupil_value_names,
                 image_value_names, image_change_value_names,
                 outcome_value_names],
...
image_change_value_names = ['no_change', 'change']
```

iii. The justification is implicit in the task itself, which requested a binary image-change output.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. The image-change vector is built on the same ophys trial index array `idx` used for the neural slices, with onset anchored by `change_time`.

ii.
```python
change_idx = np.searchsorted(ophys_ts[idx], row['change_time'])
...
'neural': neural_data[:, idx].astype(np.float32),
'image_change': image_change,
```

iii. The notes justify alignment by interpolation/slicing to the ophys clock across all variables.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed comes from `ref_ds.running_speed`, using its `timestamps` and `speed` columns.

ii.
```python
run = ref_ds.running_speed
f_run = interp1d(run['timestamps'].values, run['speed'].values,
                 kind='linear', bounds_error=False, fill_value=np.nan)
```

iii. The notes explicitly identify Allen’s running-speed stream as the locomotion source.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is linearly interpolated to the ophys timestamps, saved per trial as a continuous signal, and then discretized into five percentile bins using bin edges computed globally across all retained sessions/trials.

ii.
```python
running_speed = f_run(ophys_ts)
...
all_running = np.concatenate([
    np.concatenate([t['running'] for t in trials])
    for _, _, trials in session_results if len(trials) > 0
])
run_edges = discretize(all_running, N_LEVELS)
...
run_disc = apply_discretize(t['running'], run_edges)
```

iii. The notes justify interpolation to the ophys clock and percentile bins as a way to create balanced discrete decoder targets.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. The AI uses five equal-percentile bins computed from the pooled running-speed values across all sessions, with NaNs mapped to bin 0 during application.

ii.
```python
def discretize(all_values_flat, n_levels=N_LEVELS):
    valid = all_values_flat[~np.isnan(all_values_flat)]
    percentiles = np.linspace(0, 100, n_levels + 1)
    bin_edges = np.percentile(valid, percentiles)
    return bin_edges
...
out = np.digitize(values, bin_edges[1:-1]).astype(np.int8)
out[np.isnan(values)] = 0
```

iii. The notes explicitly justify percentile binning to approximate equal class counts.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is first interpolated onto `ophys_ts`, then the same trial frame indices `idx` are used to slice both running and neural data.

ii.
```python
running_speed = f_run(ophys_ts)
...
'neural': neural_data[:, idx].astype(np.float32),
'running': running_speed[idx].astype(np.float32),
```

iii. The notes explicitly justify interpolation onto the ophys timebase as the alignment mechanism.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. The AI uses `ref_ds.eye_tracking`, specifically the `pupil_width` column after removing rows with `likely_blink == True`.

ii.
```python
eye = ref_ds.eye_tracking
eye_clean = eye[~eye['likely_blink']]
f_pupil = interp1d(eye_clean['timestamps'].values,
                   eye_clean['pupil_width'].values,
                   kind='linear', bounds_error=False, fill_value=np.nan)
```

iii. The notes explicitly say blink frames are excluded and that `pupil_width` is used as the pupil-size measure.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. The AI removes blink frames, linearly interpolates `pupil_width` to the ophys timestamps, stores the continuous trial traces, then discretizes them into five global percentile bins with NaNs sent to bin 0.

ii.
```python
eye_clean = eye[~eye['likely_blink']]
f_pupil = interp1d(eye_clean['timestamps'].values,
                   eye_clean['pupil_width'].values,
                   kind='linear', bounds_error=False, fill_value=np.nan)
pupil_diameter = f_pupil(ophys_ts)
...
all_pupil = np.concatenate([
    np.concatenate([t['pupil'] for t in trials])
    for _, _, trials in session_results if len(trials) > 0
])
pupil_edges = discretize(all_pupil, N_LEVELS)
...
pup_disc = apply_discretize(t['pupil'], pupil_edges)
```

iii. The notes justify blink exclusion before interpolation and global percentile binning for categorical decoding.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. It is discretized into five equal-percentile bins using pooled pupil values across all sessions/trials; NaNs are assigned to bin 0.

ii.
```python
pupil_edges = discretize(all_pupil, N_LEVELS)
...
pup_disc = apply_discretize(t['pupil'], pupil_edges)
```

iii. The notes explicitly describe this as the same discretization strategy used for running speed.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil values are interpolated to `ophys_ts` first, then sliced with the same per-trial index array used for neural data.

ii.
```python
pupil_diameter = f_pupil(ophys_ts)
...
'neural': neural_data[:, idx].astype(np.float32),
'pupil': pupil_diameter[idx].astype(np.float32),
```

iii. The notes explicitly justify interpolation to the ophys timebase as the alignment step.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean columns `hit`, `miss`, `false_alarm`, and `correct_reject` in the SDK trials table.

ii.
```python
TRIAL_OUTCOMES = ['hit', 'miss', 'false_alarm', 'correct_reject']
...
outcome = 'other'
for label in TRIAL_OUTCOMES:
    if row[label]:
        outcome = label
        break
```

iii. The notes justify these as the canonical task outcome labels.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The AI picks the first true label from `TRIAL_OUTCOMES`, maps it to an integer code, and repeats that code across all time bins in the trial output matrix.

ii.
```python
outcome_to_code = {name: i for i, name in enumerate(TRIAL_OUTCOMES)}
...
outcome_code = outcome_to_code.get(t['trial_outcome'], -1)
outcome_row = np.full(n_frames, outcome_code, dtype=np.int8)
```

iii. The notes explicitly justify using a fixed categorical mapping for the four trial-outcome classes.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several edge cases pragmatically: it catches session-level exceptions and skips failed sessions; clips trial end indices to the recording length; skips zero-frame trials; maps interpolated NaNs in running/pupil to discrete bin 0; and drops sessions with fewer than two trials.

ii.
```python
try:
    session_data = extract_session_data(bc, sess_exps)
    trials = segment_trials(session_data)
except Exception as e:
    print(f'FAILED: {e}')
    continue
...
if end_idx > T:
    end_idx = T
if end_idx <= start_idx:
    continue
...
out[np.isnan(values)] = 0
...
if len(trials) < 2:
    continue
```

iii. The trajectory explicitly mentions concern about NaN pupil handling and concludes that mapping NaNs to bin 0 is an acceptable compromise; the remaining handling is only implicit from the code.

## 9-a. What are the most time-consuming steps of the code?

i. The dominant cost is loading experiments through the Allen SDK cache for every experiment in every session. The script also does a full pass over all trials afterward for discretization and assembly.

ii.
```python
for exp_id in session_experiments.index:
    datasets[exp_id] = bc.get_behavior_ophys_experiment(exp_id)
...
all_running = np.concatenate([...])
all_pupil = np.concatenate([...])
```

iii. The justification is implicit in the code structure and explicit in the notes’ runtime discussion: data loading dominates per-session runtime.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `segment_trials()` is the clearest candidate. Within it, image-name conversion to codes is also done with a Python list comprehension during assembly.

ii.
```python
for _, row in valid_trials.iterrows():
    ...
for t in trials:
    ...
    image_row = np.array([image_to_code[name] for name in t['image_names']], dtype=np.int8)
```

iii. The notes do not discuss vectorization directly. This answer is inferred from the code and from the agent’s runtime focus on data loading rather than CPU-side loop optimization.

## 9-c. What processing does the code repeat multiple times?

i. The code repeats multiple full-data traversals: one pass to load and segment trials, another pooled pass to compute running/pupil bin edges, another pooled pass to collect all image names from trials, and another pass to assemble final outputs.

ii.
```python
session_results.append((mouse_id, session_meta, trials))
...
all_running = np.concatenate([... for _, _, trials in session_results ...])
all_pupil = np.concatenate([... for _, _, trials in session_results ...])
...
for _, _, trials in session_results:
    for t in trials:
        all_image_names.update(t['image_names'])
...
for i, (mouse_id, session_meta, trials) in enumerate(session_results):
    for t in trials:
        ...
```

iii. There is no explicit justification in the notes. The repeated passes appear to be a straightforward implementation choice rather than a documented optimization.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several computations are only used for logging/debugging rather than the saved dataset: `global_session_count` is incremented but never used; `session_images` and `trial_lens` are computed only for prints; summary-time `output_dist` concatenation is only for console output.

ii.
```python
global_session_count += 1
...
session_images = set()
for t in trials:
    session_images.update(t['image_names'])
trial_lens = [t['neural'].shape[1] for t in trials]
print(...)
...
all_o = np.concatenate([data['output'][si][ti].ravel() for ti in range(n_trials)])
levels, counts = np.unique(all_o, return_counts=True)
print(...)
```

iii. The notes do not justify these computations; they appear to exist only for progress reporting and post-hoc inspection.
