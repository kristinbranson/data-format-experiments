# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI uses the Allen SDK's `VisualBehaviorOphysProjectCache` to load data from a local S3-style cache. It retrieves the experiment table via `get_ophys_experiment_table()`, filters to `project_code == 'VisualBehavior'`, and optionally restricts to a subset of experiments listed in a `DATALIMIT_SUBSET.csv` file. For each experiment, it calls `bc.get_behavior_ophys_experiment(exp_id)` to load the full dataset (neural, behavioral, trial data).

ii.
```python
bc = bpc.VisualBehaviorOphysProjectCache.from_s3_cache(cache_dir=cache_dir)
experiment_table = bc.get_ophys_experiment_table()
vb_experiments = experiment_table[experiment_table.project_code == PROJECT_CODE]

subset_csv = f"{args.datadir}/DATALIMIT_SUBSET.csv"
if Path(subset_csv).is_file():
    subset_ids = pd.read_csv(subset_csv).ophys_experiment_id
    vb_experiments = vb_experiments[vb_experiments.index.isin(subset_ids)]

for exp_id in session_experiments.index:
    datasets[exp_id] = bc.get_behavior_ophys_experiment(exp_id)
```

iii. The CONVERSION_NOTES document that the experiment table is the SDK's canonical listing of all experiments, and filtering by `project_code == 'VisualBehavior'` selects the single-plane ophys experiments. The AI chose to use the SDK's high-level API rather than direct NWB/HDF5 reads, and added DATALIMIT_SUBSET filtering to handle cases where only a subset of experiments are locally available.

## 1-b. How are the data split into subjects?

i. Subjects are identified by unique `mouse_id` values from the experiment table. All unique mouse IDs are sorted to create a deterministic ordering.

ii.
```python
all_mouse_ids = sorted(vb_experiments.mouse_id.unique())
```

iii. The AI notes that `mouse_id` is the SDK's unique identifier for each animal, and the count can be cross-checked against project documentation.

## 1-c. How are the data split into sessions?

i. Sessions are defined by unique `ophys_session_id` values. Multiple experiments (imaging planes) sharing the same `ophys_session_id` are grouped into a single session. Within each mouse, sessions are sorted by `date_of_acquisition`. Neural data from multiple planes are vertically stacked into one neuron-by-time matrix per session.

ii.
```python
mouse_sessions = mouse_exps.drop_duplicates(subset='ophys_session_id')[
    ['ophys_session_id', 'session_type', 'date_of_acquisition']
].sort_values(by='date_of_acquisition')
session_ids = mouse_sessions['ophys_session_id'].values
sess_exps = mouse_exps[mouse_exps.ophys_session_id == sid]
# In extract_session_data:
for exp_id, ds in datasets.items():
    dff_list.append(np.vstack(ds.dff_traces.dff.values))
neural_data = np.vstack(dff_list)  # (N_neurons, T)
```

iii. The AI's CONVERSION_NOTES (Step 5 Decision 1) state: "One NWB ophys experiment/imaging plane is one target session, matching SDK and paper imaging-plane decoding." However, the code actually groups by `ophys_session_id`, combining multiple planes. The notes and code are inconsistent on this point.

## 1-d. How are the data split into trials?

i. Trials are defined using the SDK's built-in `trials` table. Each valid trial spans from `start_time` to `stop_time` using native ophys timestamps (variable length per trial). Ophys frame indices are found via `np.searchsorted`. No temporal rebinning is applied — the native ophys frame times are used directly.

ii.
```python
trials_table = ref_ds.trials
valid_trials = trials_table[
    (~trials_table['aborted']) &
    (~trials_table['auto_rewarded']) &
    (trials_table['change_time'].notna())
]
for _, row in valid_trials.iterrows():
    start_idx = np.searchsorted(ophys_ts, row['start_time'])
    end_idx = np.searchsorted(ophys_ts, row['stop_time'])
    idx = np.arange(start_idx, end_idx)
```

iii. The CONVERSION_NOTES state the SDK's built-in trials table provides pre-computed trial metadata and the full trial window is used to capture both pre-change and post-change periods.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered to exclude aborted trials, auto-rewarded trials, and trials without a valid `change_time`. Trials with zero frames (`end_idx <= start_idx`) are skipped. Trials extending past the recording end are clipped. Sessions with fewer than 2 valid trials are excluded. No trials are dropped for having non-finite behavioral values — NaN values are instead mapped to bin 0 during discretization.

ii.
```python
valid_trials = trials_table[
    (~trials_table['aborted']) &
    (~trials_table['auto_rewarded']) &
    (trials_table['change_time'].notna())
]
if end_idx > T:
    end_idx = T
if end_idx <= start_idx:
    continue
# ... later ...
if len(trials) < 2:
    continue
# In apply_discretize:
out[np.isnan(values)] = 0
```

iii. Per the instructions, aborted and auto-rewarded trials are excluded. The AI also required a valid `change_time`. The NaN-to-0 mapping is described as a "conservative default" in the CONVERSION_NOTES.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `dff_traces` (delta F/F calcium fluorescence traces) from each experiment, accessed via the SDK's `dataset.dff_traces.dff` property.

ii.
```python
for exp_id, ds in datasets.items():
    dff_list.append(np.vstack(ds.dff_traces.dff.values))
neural_data = np.vstack(dff_list)  # (N_neurons, T)
```

iii. The AI notes that dF/F is the standard measure for two-photon calcium imaging and that the Allen SDK provides it pre-computed with neuropil correction and baseline normalization.

## 2-b. How is the `neural` data processed?

i. The only processing is combining neurons from multiple imaging planes within a session by vertically stacking their dF/F arrays. No interpolation, rebinning, normalization, or other processing is applied. The data is kept at native ophys timestamps and cast to float32 when extracting per-trial segments.

ii.
```python
dff_list = []
for exp_id, ds in datasets.items():
    dff_list.append(np.vstack(ds.dff_traces.dff.values))
neural_data = np.vstack(dff_list)
# Per trial:
'neural': neural_data[:, idx].astype(np.float32),
```

iii. The CONVERSION_NOTES state that dF/F traces are already processed by the Allen SDK pipeline (motion correction, neuropil subtraction, dF/F normalization) and no additional processing was deemed necessary.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional quality filtering is applied to the neural data. All neurons present in the SDK's `dff_traces` are included.

ii. N/A — no filtering code exists.

iii. The AI states that the Allen SDK pipeline already applies its own quality control (cell segmentation, neuropil correction) and no further filtering was needed.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial's neural data is aligned to the trial's `start_time`. The ophys frames between `start_time` and `stop_time` are extracted using `np.searchsorted` on the ophys timestamps array.

ii.
```python
start_idx = np.searchsorted(ophys_ts, row['start_time'])
end_idx = np.searchsorted(ophys_ts, row['stop_time'])
idx = np.arange(start_idx, end_idx)
'neural': neural_data[:, idx].astype(np.float32),
```

iii. The AI notes that `np.searchsorted` finds the first ophys frame at or after each boundary time, giving at most ~45ms alignment error (half a frame at ~11 Hz).

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No temporal rebinning is applied. The data is kept at the native ophys frame rate. The `time_bin_size` in metadata is computed from the median inter-frame interval of the first session's ophys timestamps. Native rates are approximately 31 Hz (~32 ms) for single-plane sessions and 11 Hz (~93 ms) for multi-plane sessions, meaning the bin size varies across sessions.

ii.
```python
first_ophys_ts = session_results[0][1]['ophys_ts']
time_bin_size_ms = float(np.median(np.diff(first_ophys_ts)) * 1000)
```

iii. The AI's CONVERSION_NOTES (Step 5 Decision 3) actually planned to use a 100 ms uniform grid: "Use 100 ms (`metadata.time_bin_size=100.0`). This is close to but not finer than the slowest 93.23 ms native interval." However, the code does NOT implement this — it keeps native timestamps without rebinning. The CONVERSION_NOTES and code are inconsistent.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the `initial_image_name` and `change_image_name` columns in the SDK's trials table, combined with `change_time` to determine when the image switches. Every frame is labeled with either the initial or change image name — there is no "gray" class for inter-stimulus intervals.

ii.
```python
change_idx = np.searchsorted(ophys_ts[idx], row['change_time'])
image_names = np.empty(n_frames, dtype=object)
image_names[:change_idx] = row['initial_image_name']
image_names[change_idx:] = row['change_image_name']
```

iii. The CONVERSION_NOTES (Step 5 Decision 6) planned to use stimulus presentation timestamps and include a "gray" class: "Mark the actual 250 ms image display after each presentation timestamp, gray otherwise." However, the code uses trial-level image names without a gray class. The CONVERSION_NOTES and code are inconsistent.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Image names are mapped to integer codes via a global mapping built from all unique image names across all sessions. The names are sorted alphabetically and assigned sequential indices.

ii.
```python
all_image_names = sorted(all_image_names)
image_to_code = {name: i for i, name in enumerate(all_image_names)}
image_row = np.array(
    [image_to_code[name] for name in t['image_names']],
    dtype=np.int8)
```

iii. The AI states that a global mapping ensures consistent integer codes across sessions, and sorting makes the mapping deterministic.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is computed per ophys frame within the trial window, using the same `idx` array as the neural data. Both share the same frame indices.

ii.
```python
change_idx = np.searchsorted(ophys_ts[idx], row['change_time'])
image_names[:change_idx] = row['initial_image_name']
image_names[change_idx:] = row['change_image_name']
```

iii. The image identity at each frame is determined by whether that frame falls before or after `change_time`, using the same ophys frame indices as the neural data.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from `change_time` and the `go` column in the trials table. It is set to 1 during a 750 ms window starting at `change_time` for go trials only. For catch trials, image_change is 0 throughout.

ii.
```python
image_change = np.zeros(n_frames, dtype=np.int8)
if row['go']:
    change_end_idx = np.searchsorted(ophys_ts[idx], row['change_time'] + 0.75)
    image_change[change_idx:change_end_idx] = 1
```

iii. The AI states the 750 ms window corresponds to one stimulus flash (250 ms) plus the following grey inter-stimulus interval (500 ms), marking the transient change event.

## 4-b. What processing is involved in computing `output` *Image change*?

i. No processing beyond computing the binary indicator from `change_time` and `go` flag via `np.searchsorted`. The 750 ms window is hardcoded.

ii. See 4-a code.

iii. N/A.

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is inherently binary (0 or 1). No thresholding is needed.

ii. N/A — it is constructed as a binary variable.

iii. N/A.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same as image identity and neural data — computed per ophys frame using the same `idx` array.

ii. See 4-a code.

iii. Same frame-level alignment as all other variables.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `dataset.running_speed`, which provides speed values and timestamps from the running wheel encoder, accessed via the SDK.

ii.
```python
run = ref_ds.running_speed
```

iii. The AI notes that `running_speed` is the SDK's standard interface for locomotion data.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is linearly interpolated from its native timestamps to the ophys timebase using `scipy.interpolate.interp1d`. It is then discretized into 5 equal-percentile bins computed globally across all sessions. NaN values from extrapolation are mapped to bin 0 during discretization.

ii.
```python
f_run = interp1d(run['timestamps'].values, run['speed'].values,
                 kind='linear', bounds_error=False, fill_value=np.nan)
running_speed = f_run(ophys_ts)
# Global discretization:
all_running = np.concatenate([
    np.concatenate([t['running'] for t in trials])
    for _, _, trials in session_results if len(trials) > 0
])
run_edges = discretize(all_running, N_LEVELS)
run_disc = apply_discretize(t['running'], run_edges)
```

iii. The AI states that linear interpolation preserves the signal shape while resampling, and percentile-based binning ensures roughly equal class counts. The CONVERSION_NOTES (Step 5 Decision 8) planned session-level ("per experiment") quintile edges, but the code computes them globally across all sessions.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 bins using percentile-based edges computed globally from all finite running speed values across all sessions. `np.digitize` assigns each value to a bin, and NaN values are mapped to bin 0.

ii.
```python
def discretize(all_values_flat, n_levels=N_LEVELS):
    valid = all_values_flat[~np.isnan(all_values_flat)]
    percentiles = np.linspace(0, 100, n_levels + 1)
    bin_edges = np.percentile(valid, percentiles)
    return bin_edges

def apply_discretize(values, bin_edges):
    out = np.digitize(values, bin_edges[1:-1]).astype(np.int8)
    out[np.isnan(values)] = 0
    return out
```

iii. The AI chose global percentile bins to maintain consistent categories across all sessions.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated to the ophys timebase before trial segmentation, so it shares the same time indices as the neural data. The same `idx` array is used to extract both.

ii.
```python
running_speed = f_run(ophys_ts)
'running': running_speed[idx].astype(np.float32),
'neural': neural_data[:, idx].astype(np.float32),
```

iii. By interpolating running speed onto `ophys_ts` upfront, alignment is guaranteed.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from the `pupil_width` column of the SDK's `eye_tracking` table. Blink frames (where `likely_blink` is True) are excluded before interpolation.

ii.
```python
eye = ref_ds.eye_tracking
eye_clean = eye[~eye['likely_blink']]
f_pupil = interp1d(eye_clean['timestamps'].values,
                   eye_clean['pupil_width'].values,
                   kind='linear', bounds_error=False, fill_value=np.nan)
pupil_diameter = f_pupil(ophys_ts)
```

iii. The CONVERSION_NOTES (Step 5 Decision 9) planned to use processed pupil `area` with equivalent-circle diameter conversion `2*sqrt(area/pi)`: "Processed area is preferred because reference blink handling has already set bad frames to NaN." However, the code uses `pupil_width` directly, not area converted to diameter. The CONVERSION_NOTES and code are inconsistent.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. After blink removal, pupil width is linearly interpolated from its native timestamps to the ophys timebase. It is then discretized into 5 equal-percentile bins computed globally across all sessions, with NaN values mapped to bin 0.

ii.
```python
f_pupil = interp1d(eye_clean['timestamps'].values,
                   eye_clean['pupil_width'].values,
                   kind='linear', bounds_error=False, fill_value=np.nan)
pupil_diameter = f_pupil(ophys_ts)
# ... global discretization same as running speed ...
pupil_edges = discretize(all_pupil, N_LEVELS)
pup_disc = apply_discretize(t['pupil'], pupil_edges)
```

iii. Same approach as running speed: linear interpolation to ophys timebase, then global percentile-based discretization.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same as running speed: 5 equal-percentile bins computed globally across all sessions, with NaN mapped to bin 0.

ii. See 5-c and 6-b code.

iii. Same justification as running speed discretization.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Same as running speed — pupil is interpolated to the ophys timebase before trial segmentation, using the same `idx` array as neural data.

ii.
```python
'pupil': pupil_diameter[idx].astype(np.float32),
'neural': neural_data[:, idx].astype(np.float32),
```

iii. Same alignment approach as running speed.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean columns `hit`, `miss`, `false_alarm`, and `correct_reject` in the trials table. The first matching outcome is used.

ii.
```python
TRIAL_OUTCOMES = ['hit', 'miss', 'false_alarm', 'correct_reject']
outcome = 'other'
for label in TRIAL_OUTCOMES:
    if row[label]:
        outcome = label
        break
```

iii. These four columns are the SDK's canonical trial outcome labels for the change detection task, and are mutually exclusive for valid (non-aborted, non-auto-rewarded) trials.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Trial outcomes are mapped to integer codes (0-3) and replicated across all time bins within a trial to create a uniform output matrix shape.

ii.
```python
outcome_to_code = {name: i for i, name in enumerate(TRIAL_OUTCOMES)}
outcome_code = outcome_to_code.get(t['trial_outcome'], -1)
outcome_row = np.full(n_frames, outcome_code, dtype=np.int8)
```

iii. The mapping order follows `TRIAL_OUTCOMES = ['hit', 'miss', 'false_alarm', 'correct_reject']` with hit=0, miss=1, false_alarm=2, correct_reject=3.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **Failed sessions**: If `extract_session_data` or `segment_trials` throws an exception, the session is skipped with a warning (try/except).
- **Truncated trials**: If `stop_time` extends past the recording, the trial is clipped. Trials with no frames are skipped.
- **Missing behavioral data**: NaN values from interpolation (running speed or pupil diameter outside the recorded range) are mapped to bin 0 during discretization.
- **Few trials**: Sessions with fewer than 2 valid trials are excluded.
- **Missing pupil**: No explicit exclusion of experiments without pupil data; the try/except would catch failures from missing eye tracking.

ii.
```python
try:
    session_data = extract_session_data(bc, sess_exps)
    trials = segment_trials(session_data)
except Exception as e:
    print(f'FAILED: {e}')
    continue
# ...
out[np.isnan(values)] = 0
```

iii. The AI's CONVERSION_NOTES (Step 5 Decision 10) planned to explicitly exclude three experiments without pupil data and drop individual trials if interpolation cannot produce finite values. However, the code relies on a generic try/except and NaN-to-0 mapping instead.

## 9-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading each experiment via `bc.get_behavior_ophys_experiment()`, which reads large neural and behavioral data arrays from the local cache. This is I/O bound. Each experiment loads full-session dF/F traces for all neurons plus running, eye tracking, and trials data.

ii. N/A — this is about the SDK data loading calls.

iii. The AI notes that the SDK caches files locally but reading them is still the bottleneck.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `segment_trials` iterates over each valid trial sequentially, performing `np.searchsorted` and array slicing for each trial. Additionally, in the assembly phase, the list comprehension `[image_to_code[name] for name in t['image_names']]` converts image names to codes element-by-element. The trial outcome iteration also loops over TRIAL_OUTCOMES for each trial.

ii.
```python
for _, row in valid_trials.iterrows():
    start_idx = np.searchsorted(ophys_ts, row['start_time'])
    end_idx = np.searchsorted(ophys_ts, row['stop_time'])
    # ...
    image_row = np.array(
        [image_to_code[name] for name in t['image_names']],
        dtype=np.int8)
```

iii. The AI notes the per-trial loop is not a bottleneck compared to data loading.

## 9-c. What processing does the code repeat multiple times?

i. The code iterates over trials twice: once in `segment_trials` (to extract raw data) and again in the assembly phase (to apply discretization and build output arrays). The collection of unique image names across all sessions also requires a separate pass over all trial data. Session images are collected again per-session for printing.

ii.
```python
# Pass 1: segment_trials extracts raw data
for _, row in valid_trials.iterrows():
    trials.append({...})

# Pass 2: assembly applies discretization
for t in trials:
    run_disc = apply_discretize(t['running'], run_edges)
    # ...

# Separate pass for image names
all_image_names = set()
for _, _, trials in session_results:
    for t in trials:
        all_image_names.update(t['image_names'])
```

iii. The AI notes that each session is loaded once and extracted trial data is reused, avoiding redundant file I/O.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code stores per-trial `image_names` as object arrays containing string values, which are then converted to integer codes in the assembly phase. These intermediate string arrays are not needed in the final output and consume extra memory. The plane labels (`f'{area}_{depth}um'`) include imaging depth, which is more granular than just the brain region name and may not be needed for the final `brain_regions` list (though it is used for region indexing).

ii.
```python
image_names = np.empty(n_frames, dtype=object)
image_names[:change_idx] = row['initial_image_name']
image_names[change_idx:] = row['change_image_name']
# Later converted:
image_row = np.array([image_to_code[name] for name in t['image_names']], dtype=np.int8)
```

iii. N/A.
