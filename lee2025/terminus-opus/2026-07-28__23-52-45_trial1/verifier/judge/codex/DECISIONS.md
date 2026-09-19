# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI hard-codes the seven animal IDs in `ALL_ANIMALS`, loads one `joblib` file per animal from `data/`, and then iterates over all days inside each animal object. It does not discover `.mat` files dynamically.

ii.
```python
ALL_ANIMALS = ["QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
               "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"]
...
for animal in animals:
    dat = joblib.load(os.path.join(data_dir, animal))
    d = dat[animal]
    n_days = d['trace'].shape[0]
```

iii. In `CONVERSION_NOTES.md`, the AI justified this by citing the paper code path `load_dat(animal, p, format="joblib")` and claiming its `joblib.load` path matched the reference loading route.

## 1-b. How are the data split into subjects?

i. Each animal ID is treated as one subject. The output `subjects` list is the sorted set of the requested animal IDs, and each session gets the index of its animal.

ii.
```python
all_subjects = sorted(set(animals))
...
subject_id = animal
subj_idx = all_subjects.index(subject_id)
...
subject_idx_list.append(subj_idx)
```

iii. The AI explicitly stated in `CONVERSION_NOTES.md` that subject identification should use the animal ID strings.

## 1-c. How are the data split into sessions?

i. The AI treats each day within an animal's `trace`/`position` arrays as a separate session and appends one session entry per day.

ii.
```python
n_days = d['trace'].shape[0]
...
for day in range(n_days):
    trace_day = d['trace'][day]
    pos_day = d['position'][day]
    ...
    neural_sessions.append(session_neural)
    input_sessions.append(session_input)
    output_sessions.append(session_output)
```

iii. In `CONVERSION_NOTES.md`, the AI wrote "Session = Day" and said this matched the paper's count of 207 sessions.

## 1-d. How are the data split into trials?

i. The AI first temporally rebins the continuous session into 1-second bins, then splits the rebinned stream into non-overlapping 60-bin trials, corresponding to 60-second trials. Any remainder shorter than a full trial is dropped.

ii.
```python
TIME_BIN_SEC = 1.0
TRIAL_DURATION_SEC = 60
TRIAL_DURATION_BINS = int(TRIAL_DURATION_SEC / TIME_BIN_SEC)
...
neural_binned = bin_trace_temporal(valid_trace, TIME_BIN_FRAMES)
pos_binned = bin_position_temporal(pos_day, TIME_BIN_FRAMES, N_SPATIAL_BINS)
...
neural_trials = split_into_trials(neural_binned, TRIAL_DURATION_BINS)
output_trials_raw = split_into_trials(pos_binned, TRIAL_DURATION_BINS)
```

iii. The AI justified this in `CONVERSION_NOTES.md` as following the task instruction to split each roughly 40-minute session into 1-minute trials.

## 1-e. How are trials filtered based on quality controls?

i. There is no explicit per-trial quality-control filter. The only related filter is that a session is skipped if trialization would produce fewer than two trials.

ii.
```python
n_trials = len(neural_trials)

if n_trials < 2:
    print(f"  WARNING: Day {day} ({env_name}) has only {n_trials} trials, skipping")
    continue
```

iii. The AI's apparent justification is the format requirement from the instructions that each session should have at least two trials for decoder evaluation.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derives neural data from the `trace` array in the per-animal joblib structure, taking one day at a time.

ii.
```python
trace_day = d['trace'][day]  # (n_cells, n_timepoints)
```

iii. In `CONVERSION_NOTES.md`, the AI described `trace` as binary calcium-event data already preprocessed by the original authors, so no dF/F computation was needed.

## 2-b. How is the `neural` data processed?

i. The AI keeps the day-by-cell trace orientation, removes cells it deems invalid, then averages the binary trace within 30-frame windows to produce 1-second firing-rate-like values as `float32`.

ii.
```python
def bin_trace_temporal(trace, time_bin_frames):
    n_cells, n_timepoints = trace.shape
    n_bins = n_timepoints // time_bin_frames
    trace_truncated = trace[:, :n_bins * time_bin_frames]
    trace_binned = trace_truncated.reshape(n_cells, n_bins, time_bin_frames).mean(axis=2)
    return trace_binned.astype(np.float32)
...
valid_trace = trace_day[valid_mask]
neural_binned = bin_trace_temporal(valid_trace, TIME_BIN_FRAMES)
```

iii. `CONVERSION_NOTES.md` says the AI chose to "use the binary trace data directly" but average within 1-second bins because that was "more practical" for the decoder format.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI filters neurons by keeping only cells whose first frame is not `NaN`, interpreting those as the cells registered on that day. It does not apply any activity threshold.

ii.
```python
valid_mask = ~np.isnan(trace_day[:, 0])
valid_trace = trace_day[valid_mask]  # (n_valid_cells, n_timepoints)
```

iii. The AI justified this in `CONVERSION_NOTES.md` by saying cells not registered on a given day have `NaN` traces and that it intentionally did not apply the reference decoder's activity-threshold filter.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no event-locked alignment. The AI aligns everything to the start of the recording session and then uses consecutive 1-minute windows as trials.

ii.
```python
'metadata': {
    ...
    'temporal_alignment_event': 'Start of recording session',
    'off_start': 0.0,
    'off_end': TRIAL_DURATION_SEC,
    ...
}
```

iii. The AI stated in `CONVERSION_NOTES.md` that the data are continuous exploration recordings, so it used the start of the recording as the effective alignment point.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 1-second bins. The AI rebins both neural and behavioral streams by grouping 30 native frames per bin.

ii.
```python
FPS = 30
TIME_BIN_SEC = 1.0
TIME_BIN_FRAMES = int(FPS * TIME_BIN_SEC)
...
'time_bin_size': TIME_BIN_SEC * 1000,
```

iii. The AI explicitly justified 1-second bins in `CONVERSION_NOTES.md` as a practical compromise that reduced data size while still "captur[ing] spatial behavior well."

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The AI derives environment geometry from the categorical environment name in `d['envs'][day, 0]`, not from the raw `blocked` arrays.

ii.
```python
env_name = str(d['envs'][day, 0])
...
env_mat = get_env_mat(env_name).flatten()
```

iii. In `CONVERSION_NOTES.md`, the AI explicitly argued for using `get_env_mat(env_name)` rather than `blocked`, because it considered the task input to be full environment geometry.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The AI maps each environment name through a hard-coded 3x3 binary template, flattens that matrix to length 9, casts it to `float32`, and repeats the same static vector for every trial in the session.

ii.
```python
def get_env_mat(env):
    env_mats = {
        'square': [[1,1,1],[1,1,1],[1,1,1]],
        'o': [[1,1,1],[1,0,1],[1,1,1]],
        ...
    }
    return np.array(env_mats[env], dtype=float)
...
env_mat = get_env_mat(env_name).flatten()  # (9,)
...
session_input.append(env_mat.astype(np.float32))
```

iii. The AI said in `CONVERSION_NOTES.md` that this logic was copied from the reference `get_env_mat` helper and better reflected the decoder input specification.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The AI derives the decoder output from the `position` array for each day, using the x/y coordinate stream.

ii.
```python
pos_day = d['position'][day]  # (2, n_timepoints)
```

iii. The AI described `position` in `CONVERSION_NOTES.md` as 2D position in a 75 x 75 cm arena sampled at 30 Hz.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The AI first bins each frame's x/y coordinates into a 3x3 spatial grid using a bin size derived from the session-wide maximum observed coordinate, then temporally bins those framewise class labels into 1-second windows by taking the within-window mode.

ii.
```python
def bin_position_to_grid(position, n_spatial_bins=3):
    buffer = 1e-5
    pos_max = np.nanmax(position) + buffer
    bin_size = pos_max / n_spatial_bins
    pos_binned = np.floor(position / bin_size).astype(int)
    pos_binned = np.clip(pos_binned, 0, n_spatial_bins - 1)
    bin_idx = pos_binned[0] * n_spatial_bins + pos_binned[1]
    return bin_idx

def bin_position_temporal(position, time_bin_frames, n_spatial_bins=3):
    bin_idx = bin_position_to_grid(position, n_spatial_bins)
    ...
    for i in range(n_bins):
        values, counts = np.unique(bin_idx_reshaped[i], return_counts=True)
        result[i] = values[np.argmax(counts)]
```

iii. The AI justified this in `CONVERSION_NOTES.md` as using the "same binning approach as the reference code" while reducing the decoder target to 3 x 3 bins and choosing the mode within each 1-second window.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Each coordinate is divided by a session-dependent `bin_size`, floored to `0, 1, 2`, clipped to the valid range, and then combined into a single category with `x_bin * 3 + y_bin`.

ii.
```python
pos_max = np.nanmax(position) + buffer
bin_size = pos_max / n_spatial_bins
pos_binned = np.floor(position / bin_size).astype(int)
pos_binned = np.clip(pos_binned, 0, n_spatial_bins - 1)
bin_idx = pos_binned[0] * n_spatial_bins + pos_binned[1]
```

iii. In `CONVERSION_NOTES.md`, the AI justified the 3x3 categorization as matching the decoder specification and explicitly described the category formula as `x_bin * 3 + y_bin`.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. The AI aligns output to neural data by applying the same 30-frame temporal binning and the same trial boundaries to both streams. Within each 1-second neural bin, the output label is the modal spatial bin.

ii.
```python
neural_binned = bin_trace_temporal(valid_trace, TIME_BIN_FRAMES)
pos_binned = bin_position_temporal(pos_day, TIME_BIN_FRAMES, N_SPATIAL_BINS)
...
neural_trials = split_into_trials(neural_binned, TRIAL_DURATION_BINS)
output_trials_raw = split_into_trials(pos_binned, TRIAL_DURATION_BINS)
...
session_output.append(output_trials_raw[trial_idx].reshape(1, -1).astype(int))
```

iii. The AI's stated rationale was that both streams should use the same 1-second bins and trial boundaries so the decoder sees one output label per neural time bin.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles missing or awkward data by dropping cells whose first sample is `NaN`, truncating any leftover partial temporal bin or partial 60-second trial, and skipping sessions that would have fewer than two full trials.

ii.
```python
valid_mask = ~np.isnan(trace_day[:, 0])
...
n_bins = n_timepoints // time_bin_frames
trace_truncated = trace[:, :n_bins * time_bin_frames]
...
n_trials = n_timebins // trial_length
...
if n_trials < 2:
    ...
    continue
```

iii. The AI justified the NaN handling in `CONVERSION_NOTES.md` as removing unregistered cells and justified the session skip with the decoder-format requirement for at least two trials per session.

## 6-a. What are the most time-consuming steps of the code?

i. The AI identified loading the per-animal files as the main expensive step. In the code itself, the largest work is the per-animal `joblib.load` plus per-day rebinned processing over large `trace` and `position` arrays.

ii.
```python
for animal in animals:
    dat = joblib.load(os.path.join(data_dir, animal))
    ...
    for day in range(n_days):
        ...
        neural_binned = bin_trace_temporal(valid_trace, TIME_BIN_FRAMES)
        pos_binned = bin_position_temporal(pos_day, TIME_BIN_FRAMES, N_SPATIAL_BINS)
```

iii. `CONVERSION_NOTES.md` says the loading is sequential and gives a runtime estimate of roughly 23 seconds per animal for load-plus-process.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI explicitly called out the temporal position-binning loop as vectorizable. The code also contains trial-splitting and per-trial append loops that could be reduced, but those were not explicitly discussed.

ii.
```python
result = np.zeros(n_bins, dtype=int)
for i in range(n_bins):
    values, counts = np.unique(bin_idx_reshaped[i], return_counts=True)
    result[i] = values[np.argmax(counts)]
```

iii. In `CONVERSION_NOTES.md`, the AI specifically said the mode computation in `bin_position_temporal` could be vectorized, suggesting `scipy.stats.mode`.

## 6-c. What processing does the code repeat multiple times?

i. The AI did not explicitly document this, but the code repeats some work: it casts the same static environment vector for every trial in a session, and if `--show-processing` is enabled it recomputes the same temporal binning again inside `plot_processing`.

ii.
```python
for trial_idx in range(n_trials):
    ...
    session_input.append(env_mat.astype(np.float32))
...
neural_binned = bin_trace_temporal(valid_trace, TIME_BIN_FRAMES)
...
def plot_processing(d, animal, data_dir):
    ...
    neural_binned = bin_trace_temporal(valid_trace, TIME_BIN_FRAMES)
    ...
    pos_binned = bin_position_temporal(d['position'][day], TIME_BIN_FRAMES, N_SPATIAL_BINS)
```

iii. No explicit justification for this repeated work was found in `CONVERSION_NOTES.md`; it appears to follow from the implementation style and the optional plotting path.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code performs some bookkeeping and visualization that are not used in the saved dataset: `subjects_list` is created but never populated or used, `total_neurons_sum` is accumulated but never written, and the optional `plot_processing` pipeline only creates diagnostic PNGs.

ii.
```python
subjects_list = []
...
total_neurons_sum = 0
...
total_neurons_sum += n_valid
...
if show_processing:
    plot_processing(d, animal, data_dir)
```

iii. The AI did not explicitly justify these as necessary for downstream analysis; they appear to have been added for logging, diagnostics, or development convenience.
