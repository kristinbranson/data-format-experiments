# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script uses a fixed list of seven animal IDs. For each ID it loads the corresponding extensionless joblib file, selects the dictionary entry keyed by that ID, and iterates over every day in its `trace` array. Full mode processes all seven animals; the nominal sample mode processes the first two animals (all their days, despite the help text saying two sessions).

ii.
```python
ANIMALS = ["QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
           "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"]

for a_idx, animal in enumerate(animals_to_process):
    dat = joblib.load(os.path.join(data_dir, animal))
    animal_data = dat[animal]
    n_days = animal_data['trace'].shape[0]
    for day in range(n_days):
        ...
```

iii. The notes say this matches the repository's `load_dat(..., format="joblib")` route and report that all 7 subjects, 207 sessions, 5,413 unique cells, and 69,744 cell-days agree with the paper and source data.

## 1-b. How are the data split into subjects?

i. Each joblib file/ID in `ANIMALS` is one subject. `subjects` preserves that list, while each retained session receives the current animal's integer index.

ii.
```python
subjects = list(animals_to_process)
for a_idx, animal in enumerate(animals_to_process):
    ...
    subject_idx_list.append(a_idx)
```

iii. The notes identify seven per-animal files and describe their cell/day counts; the fixed ordering makes session-to-subject mapping deterministic.

## 1-c. How are the data split into sessions?

i. One recording day is treated as one session. The day dimension of each animal's `trace` determines how many sessions are visited, and each accepted day is appended separately to the output session lists.

ii.
```python
n_days = animal_data['trace'].shape[0]
for day in range(n_days):
    neural_trials, input_trials, output_trials, n_valid = process_session(animal_data, day, ...)
    ...
    all_neural.append(neural_trials)
```

iii. The notes explicitly state “Session = day” and validate the resulting 207 sessions against the paper (31 days for six animals and 21 for CA1-51).

## 1-d. How are the data split into trials?

i. After 3-frame temporal pooling, each session is cut into consecutive, non-overlapping 600-bin chunks, corresponding to 1 minute (1,800 original frames). Any incomplete final chunk is dropped.

ii.
```python
BINS_PER_TRIAL = FRAMES_PER_TRIAL // TEMPORAL_BIN_SIZE  # 600
n_trials = n_total_bins // BINS_PER_TRIAL
for t in range(n_trials):
    start = t * BINS_PER_TRIAL
    end = (t + 1) * BINS_PER_TRIAL
    trial_neural = binned_trace[:, start:end]
```

iii. The task explicitly calls for 1-minute trials. The notes report 39–40 complete trials per approximately 40-minute session and explain the behavior for different frame counts.

## 1-e. How are trials filtered based on quality controls?

i. Individual complete trials receive no behavioral or trial-quality filtering. Incomplete tails are discarded by floor division. A day is excluded if it yields no valid cells, and the outer conversion also excludes any session with fewer than two complete trials.

ii.
```python
if n_valid == 0:
    return [], [], [], 0
...
if len(neural_trials) < 2:
    print(f"  Day {day}: skipped (< 2 trials)")
    continue
```

iii. The notes say the source analysis has no explicit trial filtering and intentionally omit velocity filtering because it was viewed as decoder-specific. The two-trial session rule enforces the target format requirement.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from the per-day `trace` array in each animal's joblib data. These are described as already-preprocessed binary calcium-event traces.

ii.
```python
trace = animal_data['trace'][day_idx]  # (n_cells, n_frames)
```

iii. The notes say no dF/F computation is needed because `trace` already represents thresholded rising calcium-transient events.

## 2-b. How is the `neural` data processed?

i. Cells deemed registered are selected; the trace is converted to float64, Gaussian-smoothed along time with sigma 3 frames, trimmed to a multiple of three, averaged over non-overlapping 3-frame groups, and converted to float32.

ii.
```python
valid_trace = trace[valid_mask]
smoothed = gaussian_filter1d(trace_2d.astype(np.float64), sigma=sigma, axis=1)
trimmed = smoothed[:, :n_bins * bin_size]
binned = trimmed.reshape(n_cells, n_bins, bin_size).mean(axis=2)
return binned.astype(np.float32)
```

iii. The agent justified this as matching the repository's `fit_decoder`: Gaussian smoothing with sigma equal to the 3-frame temporal bin, followed by average pooling. Its spot checks reproduced its intended transform exactly.

## 2-c. How is the `neural` data filtered based on quality controls?

i. A neuron is retained for a day when its first trace sample is not NaN; days with zero retained cells are dropped. No activity-count, movement, reliability, or place-cell threshold is applied.

ii.
```python
valid_mask = ~np.isnan(trace[:, 0])
valid_trace = trace[valid_mask]
if n_valid == 0:
    return [], [], [], 0
```

iii. The notes interpret NaNs as unregistered cells and say reference position decoding uses registered cells rather than only place cells. They deliberately omit the `>5` moving-period event and velocity filters as decoder-specific.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no experimental-event alignment. Artificial trials are consecutive windows aligned to the start of the session; metadata names “Start of recording session” and gives offsets 0 to 60 seconds.

ii.
```python
start = t * BINS_PER_TRIAL
end = (t + 1) * BINS_PER_TRIAL
...
'temporal_alignment_event': 'Start of recording session',
'off_start': 0.0,
'off_end': float(TRIAL_DURATION_SEC),
```

iii. The notes explain that the recordings are continuous and that starting trial segmentation at the beginning of the recording is consistent with processing the full session.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted resolution is 100 ms. Native 30 Hz frames are Gaussian-smoothed and downsampled by averaging each non-overlapping group of three frames.

ii.
```python
TEMPORAL_BIN_SIZE = 3
TIME_BIN_MS = (TEMPORAL_BIN_SIZE / FPS) * 1000  # 100 ms
...
'time_bin_size': TIME_BIN_MS,
```

iii. The agent chose 3-frame bins because the repository's decoder uses `temporal_bin_size=3`, and documented this as matching that downstream decoder processing.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The input is derived from the per-day `envs` string, not directly from `blocked`. The string selects one of ten hard-coded 3×3 accessibility matrices.

ii.
```python
env_name = str(animal_data['envs'][day_idx].item() ...)
...
mat = ENV_MATRICES.get(env_name)
return mat.flatten().astype(np.float32)
```

iii. The notes say these matrices reproduce the reference `get_env_mat` function and represent the known ten geometries.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The selected 3×3 matrix uses 1 for accessible partitions and 0 for blocked partitions, is flattened in row-major order to nine float32 values, and the same static array is appended to every trial in that session.

ii.
```python
ENV_MATRICES = {
    'square': np.array([[1,1,1],[1,1,1],[1,1,1]]),
    'o':      np.array([[1,1,1],[1,0,1],[1,1,1]]),
    ...
}
env_input = get_env_input(env_name)
input_trials.append(env_input)
```

iii. The agent says flattening the reference environment matrix captures which partitions are available and is appropriate as a static per-trial decoder input.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. Output is derived from each day's two-channel `position` array containing x and y coordinates in centimeters.

ii.
```python
position = animal_data['position'][day_idx]  # (2, n_frames)
```

iii. The notes identify the arena as 75×75 cm with behavior sampled alongside calcium imaging at 30 Hz.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. Position is first trimmed and averaged over each non-overlapping group of three frames. The averaged x and y values are then discretized into a single 0–8 category and stored as an int64 row vector for each trial.

ii.
```python
trimmed = position_2d[:, :n_bins * bin_size]
binned = trimmed.reshape(2, n_bins, bin_size).mean(axis=2)
...
pos_bins = discretize_position(binned_pos)
trial_output = pos_bins[start:end].reshape(1, -1).astype(np.int64)
```

iii. The notes claim mean binning matches reference behavior binning; the output dtype was changed to int64 after validation exposed a decoder error.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Each averaged coordinate is divided by 25 cm, floored, and clipped to 0–2. The class is `x_bin * 3 + y_bin`, so x is the major index and y the minor index.

ii.
```python
x_bin = np.clip(np.floor(position_2d[0] / bin_size).astype(int), 0, n_bins - 1)
y_bin = np.clip(np.floor(position_2d[1] / bin_size).astype(int), 0, n_bins - 1)
return x_bin * n_bins + y_bin
```

iii. The task requires a 3×3 grid, and the notes justify equal 25 cm bins spanning the 75 cm arena and clipping the upper boundary into the last bin.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Neural and position streams are independently transformed into the same number of 3-frame bins, then sliced with identical 600-bin trial boundaries. Thus each output category corresponds to the same 100 ms interval as its neural column.

ii.
```python
binned_trace = temporal_bin_trace(valid_trace)
binned_pos = bin_position(position)
...
trial_neural = binned_trace[:, start:end]
trial_output = pos_bins[start:end].reshape(1, -1).astype(np.int64)
```

iii. The notes state that behavior and imaging were acquired synchronously at 30 Hz and report exact converted/raw-transform spot checks for both streams.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. NaN-padded unregistered cells are removed per day using the first sample; a day with no valid cells is skipped. Unknown environment names raise an error. Data at the end that cannot fill a 3-frame bin or full minute is discarded, and positions outside the nominal range are clipped to valid categories.

ii.
```python
valid_mask = ~np.isnan(trace[:, 0])
if n_valid == 0:
    return [], [], [], 0
if mat is None:
    raise ValueError(f"Unknown environment: {env_name}")
trimmed = smoothed[:, :n_bins * bin_size]
n_trials = n_total_bins // BINS_PER_TRIAL
```

iii. The notes treat NaNs as intentional registration padding, validate variable session lengths, and explicitly document clipping positions at 75 cm and dropping incomplete tails.

## 6-a. What are the most time-consuming steps of the code?

i. Loading large per-animal joblib objects and, especially, Gaussian filtering every valid cell across roughly 72,000 frames dominate conversion. Pickling the approximately 6.3 GB expanded trial structure is also substantial. Optional plotting and decoder training are outside the core conversion path.

ii.
```python
dat = joblib.load(os.path.join(data_dir, animal))
smoothed = gaussian_filter1d(trace_2d.astype(np.float64), sigma=sigma, axis=1)
...
pickle.dump(data, f, protocol=4)
```

iii. The notes measured roughly 37 seconds per animal and about 4.3 minutes for full conversion; the final file size was 6,353.3 MB.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. Animal/day iteration is needed for variable cell registration and session output, but the Python loop that appends trial slices could be replaced by reshaping whole complete-session arrays and constructing repeated inputs in bulk. The two nested loops that create nine output labels are also trivially replaceable, though negligible. Plot annotation loops only affect optional diagnostics.

ii.
```python
for t in range(n_trials):
    start = t * BINS_PER_TRIAL
    end = (t + 1) * BINS_PER_TRIAL
    neural_trials.append(binned_trace[:, start:end])
    input_trials.append(env_input)
...
for i in range(N_SPATIAL_BINS):
    for j in range(N_SPATIAL_BINS):
        output_values_position.append(f"row{i}_col{j}")
```

iii. The agent did not explicitly discuss vectorization. Its expensive smoothing and pooling operations are already vectorized across cells and time; remaining loops primarily build the required nested list format.

## 6-c. What processing does the code repeat multiple times?

i. It extracts/converts the environment name twice per day (once inside `process_session` and once for logging), recreates the identical static input reference once per trial, and runs similar trim/reshape/mean logic separately for neural and position arrays. It also calculates plotting-only quantities when plotting is enabled.

ii.
```python
env_name = str(animal_data['envs'][day_idx].item() ...)
...
env_name = str(animal_data['envs'][day].squeeze())
...
input_trials.append(env_input)
```

iii. No repetition analysis appears in the notes. The repeated operations are mostly cheap relative to trace smoothing, although repeated trial references contribute to the serialized representation.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Full conversion computes logging counters/timings and an environment name outside `process_session` that is used only for progress text. With `--show-processing`, it creates figures for two sessions that are saved but not used in the dataset or decoder. More importantly relative to the human solution, Gaussian smoothing and 3-frame pooling are extra transforms rather than necessary format conversion, and pooling discards native temporal samples.

ii.
```python
t1 = time.time()
env_name = str(animal_data['envs'][day].squeeze())
elapsed = time.time() - t1
...
if do_plot:
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
```

iii. The agent regarded smoothing/pooling as necessary to match `fit_decoder`, not as unnecessary work. It presents optional figures as visualization diagnostics and reports timing/statistics for validation and auditability.
