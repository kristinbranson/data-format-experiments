# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent hard-codes seven animal IDs, loads one extensionless joblib file per animal, and extracts the animal-keyed dictionary. Full mode processes every listed animal; its “sample” mode actually selects two animals, not two sessions.

ii.
```python
ALL_ANIMALS = ["QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
               "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"]
...
dat = joblib.load(os.path.join(data_dir, animal))
d = dat[animal]
```

iii. The notes say this follows the paper repository's `load_dat(..., format="joblib")` path and report checks recovering 7 subjects, 207 sessions, 5,413 unique cells, and 69,744 valid cell-days.

## 1-b. How are the data split into subjects?

i. Each joblib file/animal ID is one subject. `subjects` is the sorted unique ID list, and each emitted daily session gets that animal's index.

ii.
```python
all_subjects = sorted(set(animals))
...
subject_id = animal
subj_idx = all_subjects.index(subject_id)
...
subject_idx_list.append(subj_idx)
```

iii. The agent states that the seven animal IDs are the subject identifiers and validates their session counts as 31, 31, 31, 21, 31, 31, and 31.

## 1-c. How are the data split into sessions?

i. Each day (axis 0 of `trace`) is treated as one session. A session is appended after all full one-minute trials for that day are built.

ii.
```python
n_days = d['trace'].shape[0]
...
for day in range(n_days):
    trace_day = d['trace'][day]
    pos_day = d['position'][day]
...
    neural_sessions.append(session_neural)
```

iii. The notes explicitly establish “Session = Day” and verify 207 days/sessions against the paper.

## 1-d. How are the data split into trials?

i. After 1-second temporal aggregation, each day is cut into consecutive, non-overlapping 60-bin (60-second) trials. Any final partial minute is dropped.

ii.
```python
TRIAL_DURATION_BINS = int(TRIAL_DURATION_SEC / TIME_BIN_SEC)
...
n_trials = n_timebins // trial_length
for t in range(n_trials):
    start = t * trial_length
    end = start + trial_length
    trials.append(data[..., start:end])
```

iii. The requested trials are one minute long. The notes report 39–40 trials per approximately 40-minute session and acknowledge that roughly 55 seconds can be discarded at a session end.

## 1-e. How are trials filtered based on quality controls?

i. There is no trial-level quality filter. The only session guard skips an entire day if fewer than two complete trials remain; this condition does not affect the full dataset.

ii.
```python
n_trials = len(neural_trials)
if n_trials < 2:
    print(f"  WARNING: Day {day} ({env_name}) has only {n_trials} trials, skipping")
    continue
```

iii. The notes say the reference has no explicit trial curation and that all days/sessions are used. The two-trial guard enforces the downstream format requirement.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from the joblib dataset's `trace` array, indexed by recording day.

ii.
```python
trace_day = d['trace'][day]  # (n_cells, n_timepoints)
```

iii. The agent identified `trace` as already preprocessed, binary rising-phase calcium events, so it did not recompute fluorescence or deconvolution.

## 2-b. How is the `neural` data processed?

i. The agent removes unregistered cells and averages each cell's binary event trace over non-overlapping 30-frame windows, yielding float32 event rates at 1 Hz. It truncates frames that do not complete a 30-frame bin before trial splitting.

ii.
```python
valid_trace = trace_day[valid_mask]
...
trace_truncated = trace[:, :n_bins * time_bin_frames]
trace_binned = trace_truncated.reshape(n_cells, n_bins, time_bin_frames).mean(axis=2)
return trace_binned.astype(np.float32)
```

iii. The notes justify one-second bins as a practical reduction in data size that retains behavior, while acknowledging that the paper decoder uses 3-frame (100 ms) pooling and that the raw data are at 30 Hz.

## 2-c. How is the `neural` data filtered based on quality controls?

i. A cell is retained for a day when its first trace sample is not NaN. No place-cell, movement-speed, or minimum-activity threshold is applied.

ii.
```python
valid_mask = ~np.isnan(trace_day[:, 0])
valid_trace = trace_day[valid_mask]
```

iii. The agent interpreted NaN rows as cells not registered that day. It deliberately kept every registered cell, noting that the paper's decoder applies activity/velocity filters internally but that the conversion would include all cells and time points.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no experimental-event alignment. Windows begin at the start of the continuous session and successive trials follow consecutively. Metadata nevertheless labels the alignment event “Start of recording session” and gives every trial offsets 0–60 seconds.

ii.
```python
start = t * trial_length
end = start + trial_length
trials.append(data[..., start:end])
...
'temporal_alignment_event': 'Start of recording session',
'off_start': 0.0,
'off_end': TRIAL_DURATION_SEC,
```

iii. The notes describe temporal alignment as the start of recording. They do not address that this metadata is literally true only for the first trial of a session.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Resolution is 1,000 ms. Thirty native 30 Hz frames are averaged for neural activity; position uses the modal spatial class over the same 30 frames.

ii.
```python
TIME_BIN_SEC = 1.0
TIME_BIN_FRAMES = int(FPS * TIME_BIN_SEC)
...
'time_bin_size': TIME_BIN_SEC * 1000,
```

iii. The agent chose 1 second for file size and practicality. It explicitly records that this differs from both the native 30 Hz data and the reference decoder's 100 ms binning.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. Geometry is derived from the day's categorical `envs` name, not from the raw `blocked` indices.

ii.
```python
env_name = str(d['envs'][day, 0])
...
env_mat = get_env_mat(env_name).flatten()
```

iii. The trajectory says the agent found apparent orientation/index discrepancies between `blocked` and named masks and selected `get_env_mat(env_name)` as the representation it believed matched trajectories and arena shape.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. `get_env_mat` maps each of ten names to a hard-coded 3×3 binary accessible-space mask, with 1 meaning open and 0 blocked. It is flattened to nine float32 values and repeated as a static vector for each trial.

ii.
```python
env_mats = {
    'square': [[1,1,1],[1,1,1],[1,1,1]],
    'o': [[1,1,1],[1,0,1],[1,1,1]],
    ...
}
...
session_input.append(env_mat.astype(np.float32))
```

iii. The notes say this copies the repository's `get_env_mat` logic and validate, for example, that the center of the `o` arena has zero occupancy.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. Output comes from the two-coordinate `position` array for each day.

ii.
```python
pos_day = d['position'][day]  # (2, n_timepoints)
```

iii. The notes identify these as DeepLabCut-derived x/y positions recorded at the same 30 Hz rate as neural traces.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. Each frame is assigned to a 3×3 spatial class. Then every 30-frame window is reduced to its most frequent class, reshaped to `(1, 60)` per trial, and cast to integer.

ii.
```python
bin_idx = bin_position_to_grid(position, n_spatial_bins)
...
for i in range(n_bins):
    values, counts = np.unique(bin_idx_reshaped[i], return_counts=True)
    result[i] = values[np.argmax(counts)]
...
session_output.append(output_trials_raw[trial_idx].reshape(1, -1).astype(int))
```

iii. The agent chose the modal position as the representative location in each one-second neural bin and reports exact spot-checks against that intended calculation.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. A single scalar maximum across both coordinate axes and the full day defines three equal-width bins. Values are floored, clipped to 0–2, and encoded as `x_bin * 3 + y_bin`.

ii.
```python
pos_max = np.nanmax(position) + buffer
bin_size = pos_max / n_spatial_bins
pos_binned = np.floor(position / bin_size).astype(int)
pos_binned = np.clip(pos_binned, 0, n_spatial_bins - 1)
bin_idx = pos_binned[0] * n_spatial_bins + pos_binned[1]
```

iii. The agent says this follows the repository's floor-based binning. It uses the observed daily maximum rather than the known fixed 75 cm arena extent.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Neural and position streams use matching non-overlapping 30-frame windows, truncate to complete windows, and are separately split with the same 60-bin boundaries.

ii.
```python
neural_binned = bin_trace_temporal(valid_trace, TIME_BIN_FRAMES)
pos_binned = bin_position_temporal(pos_day, TIME_BIN_FRAMES, N_SPATIAL_BINS)
neural_trials = split_into_trials(neural_binned, TRIAL_DURATION_BINS)
output_trials_raw = split_into_trials(pos_binned, TRIAL_DURATION_BINS)
```

iii. The agent treats the raw streams as frame-synchronous and reports boundary and spot checks confirming matching converted indices.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. NaN-padded unregistered cells are removed. Incomplete temporal bins and incomplete final trials are discarded; spatial indices are clipped to valid classes. Unknown environment names raise an error rather than being imputed. There is no special repair for sporadic within-cell or position NaNs.

ii.
```python
valid_mask = ~np.isnan(trace_day[:, 0])
...
trace_truncated = trace[:, :n_bins * time_bin_frames]
...
n_trials = n_timebins // trial_length
...
pos_binned = np.clip(pos_binned, 0, n_spatial_bins - 1)
```

iii. The notes characterize NaNs as registration padding and explicitly accept loss of the short session remainder. Sanity checks found finite, correctly shaped final trials.

## 6-a. What are the most time-consuming steps of the code?

i. Loading each large joblib animal file and processing/buffering its full neural arrays dominate conversion; pickle serialization and, if requested, plotting add I/O. The notes measured roughly 23 seconds per animal and about three minutes total.

ii.
```python
dat = joblib.load(os.path.join(data_dir, animal))
...
neural_binned = bin_trace_temporal(valid_trace, TIME_BIN_FRAMES)
...
pickle.dump(data, f, protocol=4)
```

iii. The notes identify loading as sequential and estimate loading plus processing per animal; they did not provide a finer profiler breakdown.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-time-bin mode loop and the trial-slicing/list-building loops could be vectorized or reshaped in bulk. Subject/day iteration is appropriate because cell counts and session metadata vary.

ii.
```python
for i in range(n_bins):
    values, counts = np.unique(bin_idx_reshaped[i], return_counts=True)
    result[i] = values[np.argmax(counts)]
...
for trial_idx in range(n_trials):
    session_neural.append(neural_trials[trial_idx])
```

iii. The notes specifically flag modal position computation as vectorizable (for example with `scipy.stats.mode`) and say parallel loading was unnecessary at the observed runtime.

## 6-c. What processing does the code repeat multiple times?

i. Trial splitting traverses neural and position arrays separately, and each trial repeatedly casts the same static environment vector. With `--show-processing`, the first three days' neural and position binning are recomputed after conversion solely for plots.

ii.
```python
neural_trials = split_into_trials(neural_binned, TRIAL_DURATION_BINS)
output_trials_raw = split_into_trials(pos_binned, TRIAL_DURATION_BINS)
...
session_input.append(env_mat.astype(np.float32))
...
neural_binned = bin_trace_temporal(valid_trace, TIME_BIN_FRAMES)  # in plot_processing
```

iii. The agent documented the mode loop but did not explicitly discuss these repeated operations; plotting is optional and intended as a sanity check.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several computed values are unused (`subjects_list`, `n_cells_total`, `total_neurons_sum`), and timing/progress values serve only logging. Optional plotting recomputes converted views that are not stored in the pickle. The conversion itself otherwise stores its neural, input, and output transformations for downstream use.

ii.
```python
subjects_list = []
n_cells_total = d['trace'].shape[1]
...
total_neurons_sum += n_valid
...
if show_processing:
    plot_processing(d, animal, data_dir)
```

iii. The notes present plots and progress statistics as validation aids. They do not identify the unused variables, and the plots are deliberately optional rather than part of downstream decoder data.
