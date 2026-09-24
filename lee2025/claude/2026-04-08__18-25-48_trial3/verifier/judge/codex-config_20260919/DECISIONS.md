# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent hard-codes the seven animal IDs and loads each animal's extensionless joblib file. Each file is a nested dictionary keyed by animal ID and contains all days, cells, frames, positions, and environment names. Full mode iterates over all seven animals; sample mode loads the first animal.

ii.
```python
animals = ['QLAK-CA1-08', 'QLAK-CA1-30', 'QLAK-CA1-50', 'QLAK-CA1-51',
           'QLAK-CA1-56', 'QLAK-CA1-74', 'QLAK-CA1-75']
...
dat = joblib.load(os.path.join(data_dir, animal_name))
d = dat[animal_name]
trace = d['trace']
position = d['position']
envs = d['envs'].flatten()
```

iii. The notes state that the extensionless files are the repository's joblib representation and that this matches the reference repository's `load_dat(..., format='joblib')`. The agent validated 7 animals, 207 sessions, and 69,744 neuron-sessions against the paper.

## 1-b. How are the data split into subjects?

i. One hard-coded animal ID is one subject. Its index in `animals` is stored once for every session belonging to it.

ii.
```python
subjects = animals
for animal in animals_to_process:
    subject_id = animals.index(animal)
    ...
    subject_idx_list.append(subject_id)
```

iii. The notes identify seven joblib files/animals and confirm the expected per-animal session counts. This follows the source organization.

## 1-c. How are the data split into sessions?

i. Each recording day along axis 0 of the animal arrays becomes a session. The agent processes every `day` and later appends each day's trial lists as one output session.

ii.
```python
n_days, n_cells_total, n_frames_total = trace.shape
for day in range(n_days):
    trace_day = trace[day]
    pos_day = position[day]
    env_name = str(envs[day])
...
for s in range(n_sessions):
    all_neural.append(result['neural'][s])
```

iii. The notes say one recording day is one decoder session and verify 207 total sessions, matching the paper and reference data.

## 1-d. How are the data split into trials?

i. Each session is divided into consecutive, non-overlapping 1-minute windows: 1,800 frames at 30 Hz. Integer division drops the final incomplete window.

ii.
```python
trial_duration_frames = fps * trial_duration_sec  # 1800 frames
n_trials = n_frames_total // trial_duration_frames
for trial_idx in range(n_trials):
    start = trial_idx * trial_duration_frames
    end = start + trial_duration_frames
    trial_neural = active_trace[:, start:end].astype(np.float32)
```

iii. The task explicitly requires 1-minute trials. The notes report 39–40 trials per session and acknowledge that about 1,666 remainder frames per typical session are discarded.

## 1-e. How are trials filtered based on quality controls?

i. No complete 1-minute trial is quality-filtered. Only an incomplete trailing segment is omitted.

ii.
```python
n_trials = n_frames_total // trial_duration_frames
```

iii. The notes say there was no original trial curation and that artificial one-minute segmentation is required by the task.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes directly from `d['trace']`, the cross-day registered, binary calcium-event array.

ii.
```python
trace = d['trace']       # (n_days, n_cells, n_frames)
trace_day = trace[day]   # (n_cells, n_frames)
```

iii. The agent determined that `trace` already contains binarized significant rising-phase calcium events, so raw fluorescence or delta-F/F need not be recomputed.

## 2-b. How is the `neural` data processed?

i. For each day, the agent selects registered cells, replaces any residual NaNs with zero, slices the time axis into trials, and casts each trial to `float32`. It does not smooth, calculate rate maps, or temporally rebin.

ii.
```python
active_mask = ~np.all(np.isnan(trace_day), axis=1)
active_trace = trace_day[active_mask]
active_trace = np.nan_to_num(active_trace, nan=0.0)
trial_neural = active_trace[:, start:end].astype(np.float32)
```

iii. The notes justify direct use because the traces are already binary events and say all registered cells should be included; place-cell, speed, rate-map, and smoothing operations were judged irrelevant to this decoder conversion.

## 2-c. How is the `neural` data filtered based on quality controls?

i. A cell is retained in a session if its entire daily trace is not NaN. Thus cross-day registered cells absent on that day are removed. There is no place-cell, event-count, or velocity filter.

ii.
```python
active_mask = ~np.all(np.isnan(trace_day), axis=1)
active_trace = trace_day[active_mask]
```

iii. The agent interpreted all-NaN rows as unregistered cells and cited the paper's motivation for including all recorded cells. It explicitly chose not to apply the reference decoder's speed/event threshold, saying the downstream decoder would handle the data.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no biological event alignment. Time zero is the start of each consecutive 1-minute segment, and metadata names that artificial boundary as the alignment event.

ii.
```python
start = trial_idx * trial_duration_frames
end = start + trial_duration_frames
trial_neural = active_trace[:, start:end].astype(np.float32)
...
'temporal_alignment_event': 'Start of 1-minute trial segment within recording session',
```

iii. The recordings are continuous and the task defines trials by duration, so the agent used segment onset rather than a stimulus event. Boundary spot-checks were documented.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The native 30 Hz resolution is retained, giving 33.333 ms bins. No temporal rebinning is applied.

ii.
```python
fps = 30
...
'time_bin_size': 1000.0 / fps,
```

iii. The notes say neural and position streams are synchronous at 30 Hz and that native binary traces are used directly.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The input is derived from the per-day string in `d['envs']`, not directly from `d['blocked']`. The string is looked up in a hard-coded map of ten environment names to 3×3 accessibility matrices.

ii.
```python
envs = d['envs'].flatten()
env_name = str(envs[day])
env_mat = get_env_mat(env_name).flatten()
```

iii. The agent chose the repository's `get_env_mat()` representation and verified several shapes against named environments. The notes describe it as the reference-code geometry representation.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. A named geometry is converted to a 3×3 matrix where 1 means accessible and 0 means blocked, flattened row-major to nine values, cast to `float32`, and copied as a static vector into every trial of that session.

ii.
```python
return env_mats.get(env, np.full((3,3), np.nan)).astype(float)
...
env_mat = get_env_mat(env_name).flatten()
trial_input = env_mat.astype(np.float32)
```

iii. The notes say a flattened binary matrix directly captures which partitions are accessible and matches `get_env_mat()`. This is the inverse polarity of the human reference's blocked-position indicator.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. Output is derived from `d['position']`, specifically the two synchronous x/y coordinate streams for each day.

ii.
```python
position = d['position'] # (n_days, 2, n_frames)
pos_day = position[day]  # (2, n_frames)
```

iii. The notes identify these as DeepLabCut x/y coordinates in a 75×75 cm arena sampled at 30 Hz.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. Coordinates are clipped into the arena, divided by 25 cm, floored to integer x/y bins, clamped to 0–2, and combined into one of nine labels using `x_bin * 3 + y_bin`. Each trial output is reshaped to `(1, 1800)` and stored as `int64`.

ii.
```python
x = np.clip(position[0], 0, env_size - 1e-10)
y = np.clip(position[1], 0, env_size - 1e-10)
x_bin = np.floor(x / bin_size).astype(int)
y_bin = np.floor(y / bin_size).astype(int)
bin_ids = x_bin * 3 + y_bin
...
trial_output = bin_ids[start:end].reshape(1, -1).astype(np.int64)
```

iii. The agent says this implements the requested 3×3 partition structure. Its class numbering is x-major; the human reference uses `y_bin * 3 + x_bin`.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Both axes use equal 25 cm categories corresponding to `[0,25)`, `[25,50)`, and `[50,75]` after clipping. The nine category IDs are the Cartesian combination of the two axis bins.

ii.
```python
bin_size = env_size / 3.0
x_bin = np.floor(x / bin_size).astype(int)
y_bin = np.floor(y / bin_size).astype(int)
x_bin = np.clip(x_bin, 0, 2)
y_bin = np.clip(y_bin, 0, 2)
```

iii. The task requires a 3×3 output; the notes specify edges at 0, 25, 50, and 75 cm. Clipping handles boundary or slightly out-of-range values.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position and trace are assumed frame-synchronous and are sliced with identical `start:end` indices for each trial.

ii.
```python
trial_neural = active_trace[:, start:end].astype(np.float32)
trial_output = bin_ids[start:end].reshape(1, -1).astype(np.int64)
```

iii. The notes state both streams are sampled at 30 Hz and document spot-checks at ordinary frames and trial boundaries.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. All-NaN neuron rows are excluded; residual NaNs in otherwise retained neural rows are silently changed to zero. Position is clipped to valid arena bounds. An unknown environment produces nine NaNs rather than an exception. Incomplete trial tails are dropped.

ii.
```python
active_mask = ~np.all(np.isnan(trace_day), axis=1)
active_trace = np.nan_to_num(active_trace, nan=0.0)
...
return env_mats.get(env, np.full((3,3), np.nan)).astype(float)
...
x = np.clip(position[0], 0, env_size - 1e-10)
```

iii. The agent describes zero-filling as a safety measure that should not normally be exercised, and treats incomplete-tail loss as acceptable. It validated known environments and observed no format warnings.

## 6-a. What are the most time-consuming steps of the code?

i. Loading each large joblib animal and serializing the roughly 20 GB converted pickle dominate. Per-day conversion is comparatively fast; optional plotting also adds work when requested.

ii.
```python
t0 = time.time()
dat = joblib.load(os.path.join(data_dir, animal_name))
...
with open(args.outfile, 'wb') as f:
    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The notes measured about 13 seconds to load an animal, about 9 seconds to process 31 sessions, and about 215 seconds for full conversion. They identify loading as the expensive conversion stage.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. Position binning and operations within frames are already vectorized. The per-trial loop could be replaced by reshape/split operations, but the required output is ultimately a list of trial arrays. Animal/day loops are appropriate because files and neuron counts vary. Plot annotation and raster loops are optional and could be vectorized/batched only marginally.

ii.
```python
for day in range(n_days):
    ...
    for trial_idx in range(n_trials):
        start = trial_idx * trial_duration_frames
        end = start + trial_duration_frames
```

iii. The agent specifically highlighted vectorized position discretization and direct NumPy trial slicing with no loop over frames. It reported no optimization need because runtime was well below the limit.

## 6-c. What processing does the code repeat multiple times?

i. It casts the same session-static environment vector separately for every trial. Trial slicing and dtype conversion are also repeated for every trial; `animals.index(animal)` repeatedly searches a seven-item list. Summary computation traverses session/trial lists again.

ii.
```python
for trial_idx in range(n_trials):
    trial_neural = active_trace[:, start:end].astype(np.float32)
    trial_input = env_mat.astype(np.float32)
    trial_output = bin_ids[start:end].reshape(1, -1).astype(np.int64)
```

iii. The agent did not explicitly identify repeated work in its notes. Its efficiency discussion instead emphasizes vectorized frame operations and acceptable overall runtime.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. In sample mode, `process_animal` converts all days of the first animal before `main` retains only two sessions. `sessions_env` is built and returned but not stored in the final dataset. `--show-processing` generates diagnostic plots that are not decoder inputs. Full mode otherwise performs little discarded scientific processing: it deliberately avoids rate maps, smoothing, speed filters, and place-cell analysis.

ii.
```python
result = process_animal(animal, data_dir, trial_duration_frames,
                        show_processing=show)
...
if max_sessions is not None:
    n_sessions = min(n_sessions, max_sessions)
...
return {..., 'envs': sessions_env, ...}
```

iii. The agent says visualization is optional and that irrelevant reference analyses were intentionally omitted. It did not note that sample mode converts and then discards the remaining days.
