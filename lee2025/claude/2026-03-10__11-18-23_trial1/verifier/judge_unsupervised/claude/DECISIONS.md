# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data for each animal from pre-saved joblib files in the `data/` directory. Each animal's file contains a dictionary keyed by the animal ID, with fields `trace`, `position`, `envs`, etc. The AI iterates over a hardcoded list of 7 animal IDs and loads each file with `joblib.load`.

ii.
```python
ANIMALS = [
    "QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
    "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"
]
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

def load_animal_data(animal):
    filepath = os.path.join(DATA_DIR, animal)
    dat = joblib.load(filepath)
    return dat[animal]
```

iii. The AI documented this in CONVERSION_NOTES.md Step 1, noting that `load_dat` in the reference code loads via `joblib.load` for the joblib format. The AI correctly matched this approach.

## 1-b. How are the data split into subjects (mice)?

i. Each animal corresponds to a separate joblib data file. The AI iterates over the `ANIMALS` list and processes each animal independently in `process_animal()`. A `subject_idx` array maps each session to its animal's index in the `ANIMALS` list.

ii.
```python
for animal in animals:
    neural, inp, out, sess_info = process_animal(animal, show_processing=show_processing)
    subject_id = ANIMALS.index(animal)
    for s_idx in range(len(neural)):
        subject_idx_list.append(subject_id)
```

iii. The AI noted in CONVERSION_NOTES.md Step 2 that each animal is stored as a separate file. The split is inherent to the data organization.

## 1-c. How are the data split into sessions?

i. Each animal's data has a `trace` array of shape `(n_days, n_cells, n_frames)`. Each day (index along the first axis) becomes a separate session. Sessions with 0 registered cells or fewer than 2 valid trials are skipped.

ii.
```python
n_days = d['trace'].shape[0]
for day in range(n_days):
    trace = d['trace'][day]
    # ... process day ...
    if n_trials < 2:
        print(f"  Day {day} ({env_name}): Only {n_trials} trial(s), skipping (need >=2)")
        continue
    all_neural.append(neural_trials)
```

iii. The AI documented in CONVERSION_NOTES.md Step 2 that sessions are days (31 days per animal, except 21 for QLAK-CA1-51), yielding 207 total sessions matching the paper.

## 1-d. How are the data split into trials?

i. Each session (~40 min, ~72000 frames at 30Hz) is split into 1-minute segments of 1800 frames each. Trial starts are generated with `range(0, n_frames, 1800)`. Partial trials at the end shorter than 30 seconds (900 frames) are dropped.

ii.
```python
TRIAL_DURATION_S = 60
FRAMES_PER_TRIAL = FPS * TRIAL_DURATION_S  # 1800 frames
MIN_TRIAL_FRAMES = FPS * 30  # 900 frames

trial_starts = list(range(0, n_frames, FRAMES_PER_TRIAL))
for start in trial_starts:
    end = min(start + FRAMES_PER_TRIAL, n_frames)
    trial_len = end - start
    if trial_len < MIN_TRIAL_FRAMES:
        continue
```

iii. The AI's justification comes from the instructions: "The experiment consists of long recording sessions, which will be split into 1-minute trials within each session." The 30-second minimum threshold for partial trials is a reasonable design choice.

## 1-e. How are trials filtered based on quality controls?

i. The only trial filtering is the minimum duration check: partial trials at the end of a session that are less than 30 seconds long are dropped. No other quality-based trial filtering (e.g., based on velocity, coverage of the arena, or other behavioral metrics) is applied.

ii.
```python
if trial_len < MIN_TRIAL_FRAMES:
    continue
```

iii. The AI stated in CONVERSION_NOTES.md Step 3 under "Trial curation rules": "No trial filtering. Each session (day) is one continuous 40-min recording, split into 1-minute segments." The reference code does not have trial-level filtering either (the 1-minute segmentation is a requirement from the instructions, not from the reference).

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from the `trace` field of each animal's dataset. This is a 3D array of shape `(n_days, n_cells, n_frames)` containing binarized calcium events (0 or 1) from rising-phase extraction.

ii.
```python
trace = d['trace'][day]  # shape: (n_cells, n_frames) for this day
```

iii. The AI documented in CONVERSION_NOTES.md Step 1: "Data is already preprocessed: binary calcium trace (0/1 for significant events from rising-phase extraction)" and "No delta F/F computation needed - trace is already binarized."

## 2-b. How is the `neural` data processed?

i. The neural data undergoes minimal processing: (1) non-registered cells (those with NaN values at frame 0) are filtered out, (2) the trace is sliced into 1-minute trial windows, and (3) cast to float32. No additional processing such as smoothing, rate computation, z-scoring, or velocity-based filtering is applied.

ii.
```python
registered_mask = ~np.isnan(trace[:, 0])
trace_registered = trace[registered_mask]
# ...
neural_trial = trace_registered[:, start:end].astype(np.float32)
```

iii. The AI noted in CONVERSION_NOTES.md: "Binary trace (0/1) - Used as-is" and stated "No velocity filtering: Not specified in decoder task" in the Key Decisions section.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The only filtering is exclusion of non-registered cells. A cell is considered non-registered on a given day if its trace value at frame 0 is NaN. No further quality filtering is applied (no place cell identification, no minimum activity threshold, no split-half reliability filtering).

ii.
```python
registered_mask = ~np.isnan(trace[:, 0])
n_registered = np.sum(registered_mask)
if n_registered == 0:
    print(f"  Day {day} ({env_name}): No registered cells, skipping")
    continue
trace_registered = trace[registered_mask]
```

iii. The AI justified using all registered cells by citing the paper: "motivated the inclusion of all cells in subsequent analyses." However, the reference `decode_position_within` function applies cell filtering (`cell_threshold=5`: cells must have >5 events during high-velocity periods).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the start of each 1-minute trial segment. There is no event-based alignment (e.g., stimulus onset, reward delivery). The alignment is purely temporal: trials start at frame 0, 1800, 3600, etc. within each session recording.

ii.
```python
trial_starts = list(range(0, n_frames, FRAMES_PER_TRIAL))
neural_trial = trace_registered[:, start:end].astype(np.float32)
```

iii. The AI's metadata reflects this: `'temporal_alignment_event': 'Start of each 1-minute trial segment within a 40-minute recording session'`, with `off_start=0.0` and `off_end=60.0`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is the native recording frame rate of 30 Hz (~33.33 ms per frame). No temporal rebinning is applied; raw frames are used directly.

ii.
```python
FPS = 30
# In metadata:
'time_bin_size': 1000.0 / FPS,  # ~33.33 ms
```

iii. The AI documented this in CONVERSION_NOTES.md Step 5: "Time bin: 30Hz native sampling (33.33ms)." The reference code also uses 30 Hz native data for its analyses.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The input is derived from the `envs` field of each animal's data, which contains environment name strings (e.g., 'square', 'o', 't', 'u', etc.) for each day/session.

ii.
```python
envs = d['envs'].squeeze()
env_name = envs[day]
env_mat = get_env_mat(env_name).flatten()  # (9,)
```

iii. The AI correctly identified the `envs` field and used the `get_env_mat` function from the reference code to convert environment names to binary 3x3 matrices.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The environment name string is mapped to a binary 3x3 matrix using the `get_env_mat` function (copied directly from the reference code). Each cell in the matrix is 1 if that partition is accessible and 0 if blocked. The matrix is flattened to a 9-element vector and is static per trial (same value for all trials in a session).

ii.
```python
def get_env_mat(env):
    if env == 'square':
        return np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]).astype(float)
    elif env == 'o':
        return np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]]).astype(float)
    # ... (all 10 environments)

# Usage:
env_mat = get_env_mat(env_name).flatten()  # (9,)
input_trial = env_mat.astype(np.float32)
```

iii. The AI documented this as: "get_env_mat(env_name) -> flatten to 9 values, static per trial" and confirmed the function is identical to the reference code.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The output is derived from the `position` field of each animal's data, which has shape `(n_days, 2, n_frames)` containing x,y coordinates in centimeters (range 0-75 cm).

ii.
```python
position = d['position'][day]  # (2, n_frames)
pos_bins = bin_position_3x3(position)  # (n_frames,)
```

iii. The AI documented this in CONVERSION_NOTES.md Step 2: "position: shape (n_days, 2, n_frames) - x,y position in cm (0-75 range)."

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The continuous x,y position is discretized into a 3x3 spatial grid (9 bins). Each spatial dimension is divided into 3 equal bins of ~25 cm each (75 cm / 3). The position is binned using `np.floor(position / bin_size)` with a small buffer to avoid edge effects.

ii.
```python
def bin_position_3x3(position, env_size=ENV_SIZE_CM):
    buffer = 1e-5
    bin_size = (env_size + buffer) / N_POS_BINS
    x_bin = np.clip(np.floor(position[0] / bin_size).astype(int), 0, N_POS_BINS - 1)
    y_bin = np.clip(np.floor(position[1] / bin_size).astype(int), 0, N_POS_BINS - 1)
    bin_idx = x_bin * N_POS_BINS + y_bin
    return bin_idx
```

iii. The AI's binning approach is similar to the reference code's `get_rate_maps` function, which uses `position // ((np.nanmax(position) + buffer) / n_bins)`. The AI uses a fixed `env_size=75` rather than `np.nanmax(position)`.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. The continuous x,y position is discretized into 9 categories (0-8) using a 3x3 spatial grid. Each bin is assigned a row-major index: `bin_idx = x_bin * 3 + y_bin`. Categories are named `x0y0` through `x2y2`.

ii.
```python
bin_idx = x_bin * N_POS_BINS + y_bin  # 0-8 integer categories
output_trial = pos_bins[start:end].astype(np.int64).reshape(1, -1)

# Output values:
output_values = [[f"x{r}y{c}" for r in range(N_POS_BINS) for c in range(N_POS_BINS)]]
```

iii. The instructions specify "Mouse position discretized into 3 x 3 = 9 spatial bins. Time-varying." The AI follows this exactly.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. The position and neural data are inherently aligned because they share the same time axis within each session. Both are sliced with the same `start:end` indices for each trial, ensuring frame-by-frame alignment.

ii.
```python
neural_trial = trace_registered[:, start:end].astype(np.float32)
output_trial = pos_bins[start:end].astype(np.int64).reshape(1, -1)
```

iii. The data was already recorded synchronously at 30 Hz, so no additional alignment is needed. The AI confirmed this in CONVERSION_NOTES.md Step 10: "temporal alignment - the neural trace and position are already in the same reference frame."

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles missing data in the following ways: (1) Non-registered cells (NaN traces) are excluded per session. (2) Sessions with 0 registered cells are skipped. (3) Sessions with fewer than 2 valid trials are skipped. (4) Partial trials shorter than 30 seconds are dropped. (5) The last trial in each session may have fewer than 1800 frames (minimum 1666 frames observed).

ii.
```python
registered_mask = ~np.isnan(trace[:, 0])
if n_registered == 0:
    continue
if trial_len < MIN_TRIAL_FRAMES:
    continue
if n_trials < 2:
    continue
```

iii. The AI documented in CONVERSION_NOTES.md Step 10: "Last trial has fewer frames when session length isn't divisible by 1800 (min T=1666)" and "Partial trials <900 frames (30s) are dropped."

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading data from the joblib files. Each animal's data file is large (containing trace, position, maps, SFPs, etc.). The total conversion took ~160 seconds for all 7 animals, with individual animal processing times ranging from 17-25 seconds. The actual data processing (slicing, binning) is fast since it's vectorized.

ii.
```python
# Timing is printed per animal:
dt_total = time.time() - t0
print(f"  {animal} done: {len(all_neural)} sessions, {dt_total:.1f}s total")
```

iii. From `conversion_full_out.txt`, per-animal times were: CA1-08: 17.0s, CA1-30: 25.2s, CA1-50: ~25s, etc. The AI estimated and confirmed that full conversion would take ~160s.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The main loop iterates over trial start indices within each session. However, since each trial may have different lengths (last trial), and the operations within the loop are simple array slicing (already vectorized), there is limited room for further vectorization. The position binning function `bin_position_3x3` is already fully vectorized.

ii.
```python
# This loop could potentially be replaced with np.split or similar:
for start in trial_starts:
    end = min(start + FRAMES_PER_TRIAL, n_frames)
    neural_trial = trace_registered[:, start:end].astype(np.float32)
```

iii. The AI noted in CONVERSION_NOTES.md Step 6: "Efficient vectorized operations (no inner loops for trace/position)." The inner operations are vectorized; only the trial-splitting loop remains.

## 6-c. What processing does the code repeat multiple times?

i. The code does not repeat any significant processing. Each animal is loaded once, each session is processed once, and within each session, position binning is done once for the full session before splitting into trials. Environment geometry is computed once per session and reused across trials.

ii.
```python
# Position binning done once per session, then sliced per trial:
pos_bins = bin_position_3x3(position)  # once per session
# ...
output_trial = pos_bins[start:end]  # sliced per trial
```

iii. The design avoids redundant computation efficiently.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code loads the entire animal data dictionary including fields like `maps`, `SFPs`, `centroids`, and `blocked` that are never used. Only `trace`, `position`, and `envs` are needed. The `del d` at the end of `process_animal` frees this memory, but the initial load is wasteful.

ii.
```python
d = load_animal_data(animal)  # loads entire dict including maps, SFPs, etc.
# Only uses:
trace = d['trace'][day]
position = d['position'][day]
envs = d['envs'].squeeze()
del d  # Free memory
```

iii. The AI did not explicitly document this inefficiency. Loading only the needed fields from the joblib file is not straightforward without modifying the data format, so this is a minor issue.
