# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. A fixed list of seven animal IDs is traversed. One extensionless joblib file per animal is loaded, and its complete `trace`, `position`, and `envs` day arrays are processed (except sample mode, which stops after two sessions).

ii.
```python
for animal_idx, animal in enumerate(animals):
    dat = joblib.load(os.path.join(data_dir, animal))
    trace_all = dat[animal]['trace']
    position_all = dat[animal]['position']
    envs_all = dat[animal]['envs']
```

iii. The agent says joblib loading matches repository `load_dat()`. Its checks found all 7 subjects, 207 sessions, and 69,744 registered cell-days.

## 1-b. How are the data split into subjects?

i. Each animal ID is one subject; its list index is assigned to every retained day-session.

ii.
```python
subjects = list(animals)
for animal_idx, animal in enumerate(animals):
    ...
    all_subject_idx.append(animal_idx)
```

iii. The notes identify the seven animal files as seven subjects and validate their session counts.

## 1-c. How are the data split into sessions?

i. Each recording day is an output session. A day with no registered cell would be skipped.

ii.
```python
for day in range(n_days):
    neural_trials, input_trials, output_trials, n_registered = process_session(
        trace_all[day], position_all[day], env_name)
    if len(neural_trials) == 0:
        continue
```

iii. The notes explicitly decide one animal-day per session, totaling 207.

## 1-d. How are the data split into trials?

i. After temporal pooling, sessions are split into consecutive non-overlapping one-minute trials of 600 100-ms bins. An incomplete final minute is dropped.

ii.
```python
n_trials = n_timebins_total // TIME_BINS_PER_TRIAL
for t in range(n_trials):
    t_start = t * TIME_BINS_PER_TRIAL
    t_end = (t + 1) * TIME_BINS_PER_TRIAL
```

iii. The requested trial duration is one minute; the notes state that 1,800 native frames become 600 pooled bins.

## 1-e. How are trials filtered based on quality controls?

i. There is no trial-quality filter. Every complete minute is used; only an incomplete remainder is omitted.

ii.
```python
n_trials = n_timebins_total // TIME_BINS_PER_TRIAL
for t in range(n_trials):
```

iii. The notes found no paper-level trial curation and omit the reference decoder's velocity filter because the new task decodes all times.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from `dat[animal]['trace']`, a `(days, cells, frames)` binary calcium-event array with NaN registration padding.

ii.
```python
trace_all = dat[animal]['trace']
process_session(trace_all[day], position_all[day], env_name)
```

iii. The notes say fluorescence was already converted to binarized rising-phase events.

## 2-b. How is the `neural` data processed?

i. Unregistered cells are removed, residual NaNs become zero, and float32 traces are Gaussian-smoothed along time (sigma 3 frames) and average-pooled in groups of three.

ii.
```python
registered_mask = ~np.isnan(trace_day[:, 0])
registered_trace = np.nan_to_num(trace_day[registered_mask], nan=0.0)
smoothed_trace = gaussian_filter1d(registered_trace.astype(np.float32), sigma=3, axis=1)
pooling = AvgPool1d(kernel_size=3, stride=3)
neural_binned = pooling(torch.tensor(smoothed_trace)).numpy()
```

iii. The agent justifies this as matching repository `fit_decoder` and producing 10-Hz continuous activity.

## 2-c. How is the `neural` data filtered based on quality controls?

i. A cell is retained when its first day sample is non-NaN. Empty-cell sessions are skipped. No place-cell, reliability, event-count, or movement filter is applied.

ii.
```python
registered_mask = ~np.isnan(trace_day[:, 0])
if n_registered == 0:
    return [], [], [], 0
```

iii. NaNs are registration padding. The agent deliberately uses all registered cells and regards the `>5` event rule as specific to the old Bayesian decoder.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no biological event. Each artificial trial is aligned to its one-minute segment start.

ii.
```python
t_start = t * TIME_BINS_PER_TRIAL
neural_trial = neural_binned[:, t_start:t_end]
```

iii. Continuous arena exploration has no stimulus onset; metadata describes the imposed segment start.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Resolution is 100 ms. Native 30-Hz samples are smoothed then pooled three at a time.

ii.
```python
TEMPORAL_BIN_SIZE = 3
TIME_BIN_MS = 1000.0 * TEMPORAL_BIN_SIZE / FPS
```

iii. The agent says three-frame pooling matches `fit_decoder`; validation confirms 600 bins per minute.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. Geometry comes from the day's raw `envs` name, not `blocked` indices.

ii.
```python
envs_all = dat[animal]['envs']
env_name = envs_all[day, 0] if envs_all.ndim > 1 else envs_all[day]
env_mat = get_env_mat(env_name)
```

iii. The notes describe `envs -> get_env_mat()` as the repository mapping for static environment geometry.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. A lookup maps ten environment names to binary 3×3 occupancy matrices (one=open, zero=blocked), then flattens row-major to a static length-nine float32 trial vector. Unknown names yield NaNs.

ii.
```python
env_mat = get_env_mat(env_name)
env_flat = env_mat.flatten()
input_trial = env_flat.astype(np.float32)
```

iii. The mapping was copied from repository `utils.py`; the agent checked `square` and center-blocked `o` examples.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. Output comes from `dat[animal]['position']`, the x/y trajectory in centimeters.

ii.
```python
position_all = dat[animal]['position']
process_session(trace_all[day], position_all[day], env_name)
```

iii. The notes describe DeepLabCut tracking in a 75×75-cm arena.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. Coordinates are averaged over three-frame windows, divided into three bins per axis, clipped, combined into one class, and reshaped to `(1, time)` int64.

ii.
```python
pos_binned_temporal = pooling(torch.tensor(position_day.astype(np.float32))).numpy()
pos_bins = (pos_binned_temporal / bin_size).astype(int)
pos_bins = np.clip(pos_bins, 0, 2)
pos_category = pos_bins[0] * 3 + pos_bins[1]
```

iii. The agent wanted position to share neural time bins and changed the repository's 15×15 output to the requested 3×3 output.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Each averaged coordinate uses approximately `[0,25)`, `[25,50)`, and `[50,75]`; clipping guarantees 0–2. Category is `x_bin * 3 + y_bin`, producing 0–8.

ii.
```python
bin_size = (POSITION_MAX + BUFFER) / N_SPATIAL_BINS
pos_bins = np.clip((pos_binned_temporal / bin_size).astype(int), 0, 2)
pos_category = pos_bins[0] * 3 + pos_bins[1]
```

iii. Three equal bins per dimension satisfy the requested 3×3 categorization and match the partition scale.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position and neural data use identical three-frame pooling windows and identical 600-bin trial slices.

ii.
```python
neural_binned = pooling(torch.tensor(smoothed_trace)).numpy()
pos_binned_temporal = pooling(torch.tensor(position_day.astype(np.float32))).numpy()
output_trial = pos_category[t_start:t_end].reshape(1, -1)
```

iii. Shared pooling and slice indices preserve sample correspondence; the agent reports spot-checking it.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. NaN-padded cells are removed, residual neural NaNs become zero, empty-cell days are skipped, position bins are clipped, and incomplete final trials are dropped. Unknown environments silently produce NaN input.

ii.
```python
registered_trace = np.nan_to_num(trace_day[registered_mask], nan=0.0)
if n_registered == 0: return [], [], [], 0
pos_bins = np.clip(pos_bins, 0, 2)
```

iii. The notes treat NaNs as registration padding and variable final frame counts as expected; no empty sessions were found.

## 6-a. What are the most time-consuming steps of the code?

i. Joblib loading dominates, followed by Gaussian smoothing/pooling; optional plots also add work. Notes report 20–80 seconds loading per animal, 0.7–1.0 seconds processing per session, and 427 seconds total.

ii.
```python
dat = joblib.load(os.path.join(data_dir, animal))
smoothed_trace = gaussian_filter1d(...)
neural_binned = pooling(torch.tensor(smoothed_trace)).numpy()
```

iii. This is supported by timing calls and the notes' runtime table.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. Trial slicing/appending could use reshaping/batching. Optional plot annotation loops are also vectorizable but negligible. Expensive smoothing/pooling is already vectorized.

ii.
```python
for t in range(n_trials):
    neural_trials.append(neural_binned[:, t_start:t_end])
```

iii. The agent did not discuss this; it is inferred from the implementation.

## 6-c. What processing does the code repeat multiple times?

i. It recreates `AvgPool1d` and the environment vector per session, recasts the same static vector per trial, and in plot mode concatenates trial data again for distributions.

ii.
```python
pooling = AvgPool1d(kernel_size=3, stride=3)
input_trial = env_flat.astype(np.float32)
all_pos = np.concatenate([ot[0] for ot in output_trials])
```

iii. The notes do not discuss repeated work; these minor repetitions are visible in code.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Normal conversion discards only pooled remainder frames and incomplete minutes. With `--show-processing`, plots, histograms, concatenations, and distributions are diagnostic only and never enter the pickle.

ii.
```python
if show_processing and sessions_processed <= 2:
    save_processing_plot(...)
all_neural = np.concatenate([nt.flatten() for nt in neural_trials])
```

iii. The agent intentionally uses these optional diagnostics for sanity checking, although downstream training does not consume them.
