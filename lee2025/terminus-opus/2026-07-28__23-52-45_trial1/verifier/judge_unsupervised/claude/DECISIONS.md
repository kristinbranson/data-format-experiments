# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from joblib files, one per animal (7 animals total). Each file contains a dictionary keyed by the animal ID, with fields including `trace`, `position`, `envs`, etc. It iterates through the `ALL_ANIMALS` list and calls `joblib.load()` for each animal file, then accesses `dat[animal]` to get the data dictionary.

ii.
```python
ALL_ANIMALS = ["QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
               "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"]

for animal in animals:
    dat = joblib.load(os.path.join(data_dir, animal))
    d = dat[animal]
```

iii. The AI examined the reference code's `load_dat` function, which loads data via `joblib.load` from the `data` directory with the animal ID as the filename. The AI replicated this approach directly. The CONVERSION_NOTES.md confirms: "Data is loaded via `load_dat(animal, p, format='joblib')` which returns `{animal: dataset_dict}`".

## 1-b. How are the data split into subjects (mice)?

i. Each animal file corresponds to one subject. The AI uses the 7 animal IDs as subject identifiers, maintaining a list of unique subjects and assigning a subject index to each session.

ii.
```python
all_subjects = sorted(set(animals))

for animal in animals:
    subject_id = animal
    subj_idx = all_subjects.index(subject_id)
    # ...
    subject_idx_list.append(subj_idx)

data = {
    'subjects': all_subjects,
    'subject_idx': np.array(subject_idx_list),
    # ...
}
```

iii. The AI noted 7 subjects in the data, matching the paper's description and the `ALL_ANIMALS` list from the reference code's main.py.

## 1-c. How are the data split into sessions?

i. Each "day" within an animal's data is treated as one session. The data structure has `trace` with shape `(n_days, n_cells, n_timepoints)`, and the AI iterates over `range(n_days)` for each animal, treating each day as a separate session. This yields 207 total sessions across all animals.

ii.
```python
n_days = d['trace'].shape[0]
for day in range(n_days):
    trace_day = d['trace'][day]  # (n_cells, n_timepoints)
    pos_day = d['position'][day]  # (2, n_timepoints)
    env_name = str(d['envs'][day, 0])
    # ... process and append as one session
    neural_sessions.append(session_neural)
```

iii. The AI confirmed in CONVERSION_NOTES.md Step 4: "Each 'session' = one day of recording (~40 min) in one environment. Confirmed: 1 session per day."

## 1-d. How are the data split into trials?

i. Each ~40-minute session is split into fixed-length 1-minute trials (60 time bins at 1-second resolution). The `split_into_trials` function divides the time series by simply taking consecutive non-overlapping windows. This gives ~39-40 trials per session. Any remainder at the end of a session is discarded.

ii.
```python
TRIAL_DURATION_SEC = 60
TRIAL_DURATION_BINS = int(TRIAL_DURATION_SEC / TIME_BIN_SEC)  # 60

def split_into_trials(data, trial_length):
    n_timebins = data.shape[-1]
    n_trials = n_timebins // trial_length
    trials = []
    for t in range(n_trials):
        start = t * trial_length
        end = start + trial_length
        trials.append(data[..., start:end])
    return trials

neural_trials = split_into_trials(neural_binned, TRIAL_DURATION_BINS)
```

iii. The task instructions specified: "The experiment consists of long recording sessions, which will be split into 1-minute trials within each session." The AI followed this instruction.

## 1-e. How are trials filtered based on quality controls?

i. The only filtering is that sessions producing fewer than 2 trials are skipped. No velocity-based filtering of timepoints within trials is applied, and no other trial quality controls are implemented.

ii.
```python
if n_trials < 2:
    print(f"  WARNING: Day {day} ({env_name}) has only {n_trials} trials, skipping")
    continue
```

iii. The CONVERSION_NOTES.md states under Step 5 Key Decisions: "No velocity filtering: The reference code applies velocity filtering within the decoding function, but for our format we include all timepoints. The decoder can learn to handle stationary periods." The instructions require >=2 trials per session.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from the `trace` field in each animal's data dictionary. This contains binary calcium event data (0/1 values) representing the rising phase of calcium transients, with shape `(n_days, n_cells, n_timepoints)`.

ii.
```python
trace_day = d['trace'][day]  # (n_cells, n_timepoints)
valid_mask = ~np.isnan(trace_day[:, 0])
valid_trace = trace_day[valid_mask]
```

iii. The AI noted in CONVERSION_NOTES.md Step 1: "The trace data is BINARY (0/1) - rising phase of calcium transients. No additional dF/F computation needed - data is already preprocessed."

## 2-b. How is the `neural` data processed?

i. The binary trace data is temporally binned by averaging within 1-second windows (30 frames at 30 Hz). No Gaussian smoothing is applied before binning. The result is a firing rate estimate per time bin. No velocity filtering of timepoints is applied.

ii.
```python
FPS = 30
TIME_BIN_SEC = 1.0
TIME_BIN_FRAMES = int(FPS * TIME_BIN_SEC)  # 30

def bin_trace_temporal(trace, time_bin_frames):
    n_cells, n_timepoints = trace.shape
    n_bins = n_timepoints // time_bin_frames
    trace_truncated = trace[:, :n_bins * time_bin_frames]
    trace_binned = trace_truncated.reshape(n_cells, n_bins, time_bin_frames).mean(axis=2)
    return trace_binned.astype(np.float32)

neural_binned = bin_trace_temporal(valid_trace, TIME_BIN_FRAMES)
```

iii. The CONVERSION_NOTES.md Step 5 states: "Use the binary trace data directly. Average within each 1-second time bin to get firing rates." The reference code applies `gaussian_filter1d(traces, sigma=temporal_bin_size, axis=0)` before temporal binning in `fit_decoder`, but the AI did not replicate this smoothing step.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only cells that are registered (non-NaN) on a given day are included. The AI checks for NaN in the first timepoint of each cell's trace. No additional filtering based on cell activity threshold is applied. The reference code filters cells with >5 calcium events during moving periods (`cell_threshold=5`), but the AI does not implement this.

ii.
```python
valid_mask = ~np.isnan(trace_day[:, 0])
valid_trace = trace_day[valid_mask]
n_valid = valid_mask.sum()
```

iii. CONVERSION_NOTES.md Step 5: "No cell activity filtering: Include all registered cells. The reference code filters by activity threshold within the decoding function, but we include all cells and let the decoder handle it."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the start of the recording session. No specific stimulus or behavioral event is used for alignment. Trials are simply consecutive 1-minute windows from the start of the session.

ii.
```python
# No alignment code - data starts from index 0 of the session
neural_binned = bin_trace_temporal(valid_trace, TIME_BIN_FRAMES)
neural_trials = split_into_trials(neural_binned, TRIAL_DURATION_BINS)
```

iii. CONVERSION_NOTES.md and metadata confirm: `'temporal_alignment_event': 'Start of recording session'`, `'off_start': 0.0`, `'off_end': TRIAL_DURATION_SEC`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is 1 second (30 frames at 30 Hz). This is a rebinning from the native 30 Hz frame rate. The reference code uses 3-frame temporal bins (100 ms) in the `fit_decoder` function. The AI chose 1-second bins instead.

ii.
```python
TIME_BIN_SEC = 1.0
TIME_BIN_FRAMES = int(FPS * TIME_BIN_SEC)  # 30 frames per time bin
```

iii. CONVERSION_NOTES.md Step 5: "Time bin size: Use 1 second (30 frames) bins. This provides reasonable temporal resolution while reducing data size. The reference code uses temporal_bin_size=3 (100ms) for decoding, but for our decoder format, 1-second bins are more practical and still capture spatial behavior well."

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The input is derived from the `envs` field in each animal's data dictionary, which contains environment name strings (e.g., 'square', 'o', 't', etc.) for each day. The environment name is then mapped to a 3x3 binary matrix using the `get_env_mat` function.

ii.
```python
env_name = str(d['envs'][day, 0])
env_mat = get_env_mat(env_name).flatten()  # (9,)
session_input.append(env_mat.astype(np.float32))
```

iii. The AI copied the `get_env_mat` function directly from the reference code (`utils.py:215`), using the same 3x3 binary matrix definitions for all 10 environments.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The environment name string is looked up in a dictionary to get a 3x3 binary matrix, where 1 indicates an accessible partition and 0 indicates a blocked partition. This matrix is flattened to a 9-element vector and used as a static (non-time-varying) input per trial.

ii.
```python
def get_env_mat(env):
    env_mats = {
        'square': [[1,1,1],[1,1,1],[1,1,1]],
        'o': [[1,1,1],[1,0,1],[1,1,1]],
        't': [[0,1,0],[0,1,0],[1,1,1]],
        # ... etc
    }
    return np.array(env_mats[env], dtype=float)

env_mat = get_env_mat(env_name).flatten()  # (9,)
```

iii. The AI noted this was copied from the reference code. CONVERSION_NOTES.md: "Use get_env_mat(env_name).flatten() -> 9 binary values. Static per trial."

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The output is derived from the `position` field in each animal's data dictionary, which contains x,y coordinates of the mouse at 30 Hz with shape `(n_days, 2, n_timepoints)`.

ii.
```python
pos_day = d['position'][day]  # (2, n_timepoints)
pos_binned = bin_position_temporal(pos_day, TIME_BIN_FRAMES, N_SPATIAL_BINS)
```

iii. CONVERSION_NOTES.md Step 2 notes position data is at 30 Hz from DeepLabCut tracking.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The continuous x,y position is first discretized into a 3x3 spatial grid (9 bins), then temporally binned into 1-second windows using the mode (most frequent spatial bin) within each window.

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
    n_bins = len(bin_idx) // time_bin_frames
    bin_idx_reshaped = bin_idx[:n_bins * time_bin_frames].reshape(n_bins, time_bin_frames)
    result = np.zeros(n_bins, dtype=int)
    for i in range(n_bins):
        values, counts = np.unique(bin_idx_reshaped[i], return_counts=True)
        result[i] = values[np.argmax(counts)]
    return result
```

iii. CONVERSION_NOTES.md Step 5: "Bin position into 3x3 grid (9 bins). Use the same binning approach as the reference code: pos_binned = floor(pos / bin_size) where bin_size = (max_pos + buffer) / 3. Output = x_bin * 3 + y_bin."

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Position is discretized by dividing the arena into a 3x3 grid. The bin size is computed as `(max_position + buffer) / 3`, and each coordinate is assigned to a bin via `floor(pos / bin_size)`. The x and y bin indices are combined into a single category index: `bin_idx = x_bin * 3 + y_bin`, giving 9 categories (0-8).

ii.
```python
N_SPATIAL_BINS = 3
N_OUTPUT_CLASSES = N_SPATIAL_BINS ** 2  # 9

pos_max = np.nanmax(position) + buffer  # buffer = 1e-5
bin_size = pos_max / n_spatial_bins
pos_binned = np.floor(position / bin_size).astype(int)
pos_binned = np.clip(pos_binned, 0, n_spatial_bins - 1)
bin_idx = pos_binned[0] * n_spatial_bins + pos_binned[1]
```

iii. The task instructions specify: "Mouse position discretized into 3 x 3 = 9 spatial bins. Time-varying." The AI followed this specification. The binning logic parallels the reference code's approach but uses 3 bins instead of 15.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position and neural data are aligned by sharing the same temporal binning. Both are binned from the same 30 Hz timepoints into the same 1-second time bins, then split into the same 60-bin trials. This ensures position bin `t` corresponds to neural activity at bin `t`.

ii.
```python
# Both use the same time bins and trial splitting
neural_binned = bin_trace_temporal(valid_trace, TIME_BIN_FRAMES)  # (n_cells, n_timebins)
pos_binned = bin_position_temporal(pos_day, TIME_BIN_FRAMES, N_SPATIAL_BINS)  # (n_timebins,)

neural_trials = split_into_trials(neural_binned, TRIAL_DURATION_BINS)
output_trials_raw = split_into_trials(pos_binned, TRIAL_DURATION_BINS)
```

iii. The AI aligned position and neural data by processing both from the same raw timepoints with the same temporal binning, ensuring correspondence.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing data is handled in two ways: (1) Cells not registered on a given day have NaN traces and are excluded via `~np.isnan(trace_day[:, 0])`. (2) Sessions with fewer than 2 trials are skipped. No handling of NaN in position data is explicitly implemented beyond the use of `np.nanmax`.

ii.
```python
valid_mask = ~np.isnan(trace_day[:, 0])
valid_trace = trace_day[valid_mask]

if n_trials < 2:
    print(f"  WARNING: Day {day} ({env_name}) has only {n_trials} trials, skipping")
    continue
```

iii. CONVERSION_NOTES.md Step 2: "Cells not registered on a given day have NaN traces." Step 10 Check 5: "NaN cells: Properly excluded per session."

## 6-a. What are the most time-consuming steps of the code?

i. The AI identified data loading as the dominant cost (~23s per animal), with total conversion taking ~3 minutes for all 7 animals. The mode computation in `bin_position_temporal` is also noted as relatively slow due to the per-bin loop.

ii.
```python
# Each animal load takes ~23s
dat = joblib.load(os.path.join(data_dir, animal))
```

iii. CONVERSION_NOTES.md Step 7: "Load + process: ~23s/animal, estimated total ~160s for 7 animals."

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The mode computation in `bin_position_temporal` iterates over each time bin to compute `np.unique` and `argmax`. This could be vectorized using `scipy.stats.mode`. The trial-building loop also iterates per trial, though this is less significant.

ii.
```python
# Loop that could be vectorized:
result = np.zeros(n_bins, dtype=int)
for i in range(n_bins):
    values, counts = np.unique(bin_idx_reshaped[i], return_counts=True)
    result[i] = values[np.argmax(counts)]
```

iii. CONVERSION_NOTES.md Step 6: "Mode computation in bin_position_temporal uses a loop (could vectorize with scipy.stats.mode)."

## 6-c. What processing does the code repeat multiple times?

i. The position max computation (`np.nanmax(position)`) is computed independently for each day/session within `bin_position_to_grid`, rather than being computed once globally. Environment geometry is also recomputed for each trial within a session, though it's the same for all trials in that session.

ii.
```python
# Position max computed per session inside bin_position_to_grid:
pos_max = np.nanmax(position) + buffer  # computed for each day

# env_mat computed once per session but same for all trials:
env_mat = get_env_mat(env_name).flatten()
session_input.append(env_mat.astype(np.float32))  # same object appended per trial
```

iii. No explicit justification for this repetition was documented.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code processes all timepoints including when the mouse is stationary. The reference code filters out timepoints where mouse velocity is below 5 cm/s, but the AI includes all timepoints. This means the decoder receives neural activity during stationary periods that the reference analysis would have excluded. The `--show-processing` mode generates visualization plots that are not needed for the final conversion.

ii.
```python
# All timepoints included without velocity filtering:
neural_binned = bin_trace_temporal(valid_trace, TIME_BIN_FRAMES)
# No velocity-based exclusion of timepoints
```

iii. CONVERSION_NOTES.md Step 5: "No velocity filtering: The reference code applies velocity filtering within the decoding function, but for our format we include all timepoints."
