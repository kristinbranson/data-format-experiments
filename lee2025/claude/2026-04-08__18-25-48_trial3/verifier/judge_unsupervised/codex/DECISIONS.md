# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent hard-codes the seven animal IDs, loads one extensionless `joblib` file per animal from `data/`, pulls `trace`, `position`, and `envs` out of the nested dict, then iterates over all days and all fixed-length trial segments. It does not load `.mat` files, `blocked`, `maps`, `SFPs`, or `centroids` for conversion.

ii. 
```python
data_dir = 'data'
animals = ['QLAK-CA1-08', 'QLAK-CA1-30', 'QLAK-CA1-50', 'QLAK-CA1-51',
           'QLAK-CA1-56', 'QLAK-CA1-74', 'QLAK-CA1-75']

for animal in animals_to_process:
    subject_id = animals.index(animal)
    result = process_animal(animal, data_dir, trial_duration_frames,
                           show_processing=show)
```

```python
dat = joblib.load(os.path.join(data_dir, animal_name))
d = dat[animal_name]

trace = d['trace']       # (n_days, n_cells, n_frames)
position = d['position'] # (n_days, 2, n_frames)
envs = d['envs'].flatten()
```

iii. In `CONVERSION_NOTES.md`, the agent says the extensionless joblib files contain the needed fields and that `trace`, `position`, and `envs` are sufficient for its mapping. The trajectory shows it explicitly chose the joblib files after inspecting the directory and data structure.

## 1-b. How are the data split into subjects?

i. Subjects are the seven named mice in the hard-coded `animals` list. `subjects` is set to that full list, and each session gets a `subject_idx` equal to the mouse’s index in that list.

ii.
```python
subjects = animals  # All 7 animals are subjects
subject_idx_list = []

for animal in animals_to_process:
    subject_id = animals.index(animal)
    ...
    for s in range(n_sessions):
        ...
        subject_idx_list.append(subject_id)
```

iii. The agent’s notes say “All 7 animals are subjects” and “One session = one decoder session,” so it used the animal file boundary as the subject boundary.

## 1-c. How are the data split into sessions?

i. Each recording day within one animal file is treated as one session. The script iterates `for day in range(n_days)` and appends one session-level list of trials for each day.

ii.
```python
n_days, n_cells_total, n_frames_total = trace.shape

for day in range(n_days):
    trace_day = trace[day]
    pos_day = position[day]
    env_name = str(envs[day])
    ...
    sessions_neural.append(trials_neural)
    sessions_input.append(trials_input)
    sessions_output.append(trials_output)
```

iii. In `CONVERSION_NOTES.md`, the agent states “One session = one decoder session” and that each recording day is a separate session.

## 1-d. How are the data split into trials?

i. Within each day/session, the agent creates contiguous fixed-length 1-minute trials at 30 Hz, so each trial is 1800 frames. It uses floor division and slices consecutive frame blocks.

ii.
```python
fps = 30
trial_duration_sec = 60
trial_duration_frames = fps * trial_duration_sec  # 1800 frames
...
n_trials = n_frames_total // trial_duration_frames

for trial_idx in range(n_trials):
    start = trial_idx * trial_duration_frames
    end = start + trial_duration_frames
    trial_neural = active_trace[:, start:end].astype(np.float32)
    trial_input = env_mat.astype(np.float32)
    trial_output = bin_ids[start:end].reshape(1, -1).astype(np.int64)
```

iii. The justification in the notes is direct: the instructions requested “1-minute trials within each session,” so the agent implemented fixed 60 s chunks.

## 1-e. How are trials filtered based on quality controls?

i. There is effectively no trial-quality filtering. The only implicit filtering is that the final short remainder of a session is discarded because `n_trials` uses floor division. Trials are not filtered for movement, occupancy, missing behavior, or neural quality.

ii.
```python
n_trials = n_frames_total // trial_duration_frames

for trial_idx in range(n_trials):
    start = trial_idx * trial_duration_frames
    end = start + trial_duration_frames
```

iii. In the notes, the agent says “Trial curation: None in original (1 session = 1 day). We split into 1-min trials.” It also says “No velocity filtering.”

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived only from the raw `trace` array, using one day at a time and then subselecting rows whose entire trace is not NaN.

ii.
```python
trace = d['trace']       # (n_days, n_cells, n_frames)
...
trace_day = trace[day]
active_mask = ~np.all(np.isnan(trace_day), axis=1)
active_trace = trace_day[active_mask]
```

iii. The notes repeatedly describe the neural source as the already-binarized calcium `trace` and say no dF/F computation is needed.

## 2-b. How is the `neural` data processed?

i. Processing is minimal: select the day, drop cells that are all NaN, replace any remaining NaNs with zero, slice by trial, and cast to `float32`. No temporal smoothing, deconvolution, event thresholding, velocity gating, or time rebinning is applied.

ii.
```python
active_mask = ~np.all(np.isnan(trace_day), axis=1)
active_trace = trace_day[active_mask]
active_trace = np.nan_to_num(active_trace, nan=0.0)
...
trial_neural = active_trace[:, start:end].astype(np.float32)
```

iii. The agent’s notes say “Use raw binary trace directly,” “NO additional processing needed,” and “No delta F/F needed.”

## 2-c. How is the `neural` data filtered based on quality controls?

i. The only explicit neural QC is excluding cells whose full-day trace is all NaN, treating those as unregistered on that day. Partially missing values within retained cells are zero-filled. There is no filtering on event count, place-cell status, reliability, or movement periods.

ii.
```python
# A cell is registered if its trace is not all NaN
active_mask = ~np.all(np.isnan(trace_day), axis=1)
active_trace = trace_day[active_mask]
...
active_trace = np.nan_to_num(active_trace, nan=0.0)
```

iii. The notes justify this as “All registered cells included” and say the paper motivated inclusion of all cells in subsequent analyses.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to the start of each arbitrary 1-minute segment, not to a behavioral or experimental event from the original experiment. The metadata explicitly labels the alignment event as the segment start.

ii.
```python
'metadata': {
    ...
    'temporal_alignment_event': 'Start of 1-minute trial segment within recording session',
    'off_start': 0.0,
    'off_end': float(trial_duration_sec),
}
```

iii. The agent’s notes say the instructions required 1-minute trials, so it treated each segment onset as the alignment event.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data stay at the native 30 Hz frame rate, with a time bin size of `1000 / 30 ≈ 33.33 ms`. No temporal rebinning is applied.

ii.
```python
fps = 30  # recording frame rate
...
'time_bin_size': 1000.0 / fps,  # ~33.33 ms
```

iii. The notes say “Use raw binary trace at native 30 Hz” and “Time bin size = 1/30 s.”

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The environment input is derived from the per-day `envs` string label. The script does not use the raw `blocked` field directly.

ii.
```python
envs = d['envs'].flatten()
...
env_name = str(envs[day])
env_mat = get_env_mat(env_name).flatten()
```

iii. The notes say the decoder input should be the environment geometry and point to `get_env_mat()` as the reference-code function for that mapping.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The script converts each environment name into a hard-coded binary 3x3 accessibility matrix, then flattens it to a 9-element vector and casts it to `float32`.

ii.
```python
def get_env_mat(env):
    env_mats = {
        'square':    np.array([[1,1,1],[1,1,1],[1,1,1]]),
        'o':         np.array([[1,1,1],[1,0,1],[1,1,1]]),
        ...
    }
    return env_mats.get(env, np.full((3,3), np.nan)).astype(float)
...
env_mat = get_env_mat(env_name).flatten()
trial_input = env_mat.astype(np.float32)
```

iii. The notes say this reproduces the reference-code environment representation and matches the instruction to encode which partitions are blocked.

## 3-c. How is the `input` *Environment geometry* aligned with the neural data?

i. The environment vector is static within a day, so the exact same 9-element vector is attached to every 1-minute trial from that session. It is aligned at the trial level, not per frame.

ii.
```python
for trial_idx in range(n_trials):
    ...
    trial_input = env_mat.astype(np.float32)
    trials_input.append(trial_input)
```

iii. The notes justify this by saying environment geometry is static per trial and does not change within a session.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The output is derived from the raw `position` array, specifically `position[day]` with x and y coordinates over frames.

ii.
```python
position = d['position'] # (n_days, 2, n_frames)
...
pos_day = position[day]
bin_ids = discretize_position_3x3(pos_day)
```

iii. The notes identify `position` as the source variable and describe it as DeepLabCut-tracked x-y position.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The x and y coordinates are clipped into `[0, env_size)`, divided into thirds of the 75 cm environment, floored to integer bin indices, clamped to `[0, 2]`, then combined into one categorical bin ID per frame.

ii.
```python
bin_size = env_size / 3.0
x = np.clip(position[0], 0, env_size - 1e-10)
y = np.clip(position[1], 0, env_size - 1e-10)
x_bin = np.floor(x / bin_size).astype(int)
y_bin = np.floor(y / bin_size).astype(int)
x_bin = np.clip(x_bin, 0, 2)
y_bin = np.clip(y_bin, 0, 2)
bin_ids = x_bin * 3 + y_bin
```

iii. The notes say the task required “3 x 3 = 9 spatial bins,” so the agent used direct 25 cm spatial binning of the 75 cm arena.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. The position categories are nine 25 cm by 25 cm bins defined by edges `[0, 25, 50, 75]` along x and y. The single category ID is `x_bin * 3 + y_bin` in row-major order.

ii.
```python
bin_size = env_size / 3.0
...
bin_ids = x_bin * 3 + y_bin
...
output_bin_names.append(f"x[{x_start}-{x_end}]_y[{y_start}-{y_end}]")
```

iii. The notes explicitly describe the bin edges and the `x_bin * 3 + y_bin` combination rule.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Output uses the same frame indices as neural data inside each trial slice, then is reshaped to `(1, n_timepoints)`. Alignment is therefore frame-synchronous within each 1-minute segment.

ii.
```python
start = trial_idx * trial_duration_frames
end = start + trial_duration_frames
trial_neural = active_trace[:, start:end].astype(np.float32)
trial_output = bin_ids[start:end].reshape(1, -1).astype(np.int64)
```

iii. The notes say all streams are synchronous at 30 Hz, so the agent just slices the same frame ranges from neural and behavior arrays.

## 5-a. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Same as 2-e: 33.33 ms bins at 30 Hz, with no rebinning or pooling.

ii.
```python
fps = 30
'time_bin_size': 1000.0 / fps
```

iii. The notes say the raw frame rate is kept.

## 5-b. How are the neural, input, and output data temporally aligned?

i. Neural and output are aligned by slicing the same frame interval for each trial. Input is a session-static vector copied into each trial, so it is only trial-aligned, not frame-varying. All alignment is relative to the start of the 1-minute segment.

ii.
```python
start = trial_idx * trial_duration_frames
end = start + trial_duration_frames
trial_neural = active_trace[:, start:end].astype(np.float32)
trial_input = env_mat.astype(np.float32)
trial_output = bin_ids[start:end].reshape(1, -1).astype(np.int64)
```

iii. The notes say “All streams synchronous at 30 Hz” and “Environment geometry as input: static per trial.”

## 6. How are minor issues in the data (e.g., missing data, malformed entries) handled?

i. The script drops cells that are entirely NaN on a day, replaces remaining NaNs in retained neural traces with zeros, clips out-of-range positions into valid bounds, returns an all-NaN matrix for unknown environment names, and discards incomplete final trial remainders by floor division.

ii.
```python
active_mask = ~np.all(np.isnan(trace_day), axis=1)
active_trace = np.nan_to_num(active_trace, nan=0.0)
...
return env_mats.get(env, np.full((3,3), np.nan)).astype(float)
...
x = np.clip(position[0], 0, env_size - 1e-10)
y = np.clip(position[1], 0, env_size - 1e-10)
...
n_trials = n_frames_total // trial_duration_frames
```

iii. The notes describe NaN rows as unregistered cells and remaining NaNs as “shouldn’t happen... but safety.” Other handling is implicit from the code rather than explicitly justified.

## 7-a. What are the most time-consuming steps of the code?

i. The expensive steps are loading each large animal file with `joblib.load`, iterating through every day and trial to slice arrays and append Python objects, and writing the final ~20 GB pickle.

ii.
```python
dat = joblib.load(os.path.join(data_dir, animal_name))
...
for day in range(n_days):
    ...
    for trial_idx in range(n_trials):
        ...
with open(args.outfile, 'wb') as f:
    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The runtime logs in `conversion_full_out.txt` show per-animal load times of roughly 9-23 s and a final save time of 14 s, which matches the agent’s own timing printouts.

## 7-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial Python loop could have been vectorized or at least reduced by reshaping the session arrays into `(n_trials, ...)` blocks. The output-bin-name loop and the plotting loops are also trivial Python loops.

ii.
```python
for trial_idx in range(n_trials):
    start = trial_idx * trial_duration_frames
    end = start + trial_duration_frames
    trial_neural = active_trace[:, start:end].astype(np.float32)
    ...
for i in range(3):
    for j in range(3):
        output_bin_names.append(...)
```

iii. The notes mention “Direct numpy array slicing for trial splitting,” but they still leave the trial packing itself as a Python loop.

## 7-c. What processing does the code repeat multiple times?

i. It repeatedly casts the same session-level environment vector to `float32` once per trial, repeatedly computes start/end indices in the trial loop, and repeats the same per-day trial-splitting logic for every session.

ii.
```python
for trial_idx in range(n_trials):
    ...
    trial_input = env_mat.astype(np.float32)
    ...
    trials_input.append(trial_input)
```

iii. There is no explicit justification beyond simplicity. The notes frame the implementation as straightforward and readable.

## 7-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Optional processing plots and raster visualizations are generated only for inspection and are not used by the final dataset. The script also computes and prints summary statistics only for logging.

ii.
```python
if show_processing:
    plot_processing(...)
...
print(f"Total sessions: {total_sessions}")
print(f"Total trials: {total_trials}")
```

iii. The notes explicitly label the plots as “processing visualizations” and “sanity checks,” not part of the saved decoder data.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Same as 6: all-NaN cells are dropped, remaining NaNs in retained traces are zero-filled, positions are clipped into the valid arena range, unknown environments become NaN geometry matrices, and partial trailing data are dropped.

ii.
```python
active_mask = ~np.all(np.isnan(trace_day), axis=1)
active_trace = np.nan_to_num(active_trace, nan=0.0)
return env_mats.get(env, np.full((3,3), np.nan)).astype(float)
x = np.clip(position[0], 0, env_size - 1e-10)
n_trials = n_frames_total // trial_duration_frames
```

iii. The agent’s explicit rationale only covers NaN traces; the rest is implicit defensive handling in the code.

## 9-a. What are the most time-consuming steps of the code?

i. Same as 7-a: loading large raw files, looping over days and trials, and serializing the giant pickle dominate runtime.

ii.
```python
dat = joblib.load(os.path.join(data_dir, animal_name))
...
for day in range(n_days):
    ...
    for trial_idx in range(n_trials):
        ...
pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The conversion log directly shows these phases taking most of the wall-clock time.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Same as 7-b: the nested day/trial packaging loop is the main target for vectorization; plotting and bin-name generation are minor secondary targets.

ii.
```python
for day in range(n_days):
    ...
    for trial_idx in range(n_trials):
        ...
```

iii. The agent did not justify keeping these loops beyond implementation simplicity.

## 9-c. What processing does the code repeat multiple times?

i. Same as 7-c: repeated `astype` calls for static inputs, repeated per-trial list appends, and repeated index arithmetic.

ii.
```python
trial_input = env_mat.astype(np.float32)
trials_input.append(trial_input)
```

iii. The notes do not defend this repetition; it is just how the implementation is structured.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Same as 7-d: optional plotting, summary logging, and several metadata/statistics calculations are not consumed by downstream decoding once the pickle is written.

ii.
```python
if show_processing:
    plot_processing(...)
...
print(f"File size: {file_size / 1e6:.1f} MB")
```

iii. The agent treated these as validation artifacts and documentation rather than core conversion outputs.
