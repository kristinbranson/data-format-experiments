# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads a hardcoded list of seven animal-level joblib files from `data/`, not the `.mat` files. It assumes each file contains all days/sessions for one animal, then iterates over days and later splits each day into trials.

ii.
```python
ANIMALS = ["QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
           "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"]
...
for animal_idx, animal in enumerate(animals):
    dat = joblib.load(os.path.join(data_dir, animal))
    trace_all = dat[animal]['trace']
    position_all = dat[animal]['position']
    envs_all = dat[animal]['envs']
```

iii. In `CONVERSION_NOTES.md`, the AI states that the reference repo uses `load_dat(..., format="joblib")` and describes the data files as “7 animals as joblib + .mat,” then chooses the joblib representation as its working source because it already exposes `trace`, `position`, `envs`, and related fields.

## 1-b. How are the data split into subjects?

i. Subjects are the seven hardcoded animal IDs. The AI uses the position of each animal in `ANIMALS` as the subject index.

ii.
```python
subjects = list(animals)  # animal IDs as subject names
...
for animal_idx, animal in enumerate(animals):
    ...
    all_subject_idx.append(animal_idx)
```

iii. The notes say “Each animal file (joblib) contains a dict with animal ID as key” and “Sessions: Each day for each animal = 1 session. Total 207 sessions,” so the AI treated animal identity as the subject split.

## 1-c. How are the data split into sessions?

i. Each day in the per-animal arrays is treated as one session. The AI iterates over the first dimension of `trace`, `position`, and `envs`.

ii.
```python
trace_all = dat[animal]['trace']      # (n_days, n_cells, n_frames)
position_all = dat[animal]['position']  # (n_days, 2, n_frames)
envs_all = dat[animal]['envs']          # (n_days, 1)
...
n_days = trace_all.shape[0]
...
for day in range(n_days):
    env_name = envs_all[day, 0] if envs_all.ndim > 1 else envs_all[day]
    neural_trials, input_trials, output_trials, n_registered = process_session(
        trace_all[day], position_all[day], env_name
    )
```

iii. The notes explicitly record the decision “Sessions: Each day for each animal = 1 session. Total 207 sessions.”

## 1-d. How are the data split into trials?

i. Each session is split into non-overlapping 1-minute trials after temporal rebinning. With 30 Hz data and a 3-frame bin, each trial is 600 time bins. Any leftover bins at the end are dropped.

ii.
```python
FPS = 30
TEMPORAL_BIN_SIZE = 3
TRIAL_DURATION_SEC = 60
FRAMES_PER_TRIAL = TRIAL_DURATION_SEC * FPS
TIME_BINS_PER_TRIAL = FRAMES_PER_TRIAL // TEMPORAL_BIN_SIZE  # 600
...
n_timebins_total = neural_binned.shape[1]
n_trials = n_timebins_total // TIME_BINS_PER_TRIAL
...
for t in range(n_trials):
    t_start = t * TIME_BINS_PER_TRIAL
    t_end = (t + 1) * TIME_BINS_PER_TRIAL
    neural_trial = neural_binned[:, t_start:t_end].astype(np.float32)
    output_trial = pos_category[t_start:t_end].astype(np.int64).reshape(1, -1)
```

iii. The notes say “Split each 40-min session into 1-minute trials” and explain that trial boundaries are artificial segments of the continuous recording.

## 1-e. How are trials filtered based on quality controls?

i. There is no explicit per-trial quality-control filter. Trials are generated for every complete 1-minute chunk in sessions that have at least one registered cell. Incomplete trailing data are discarded implicitly by floor division.

ii.
```python
if n_registered == 0:
    return [], [], [], 0
...
n_trials = n_timebins_total // TIME_BINS_PER_TRIAL
```

iii. The notes say “No velocity filtering” and “No explicit trial-level curation,” indicating that the AI intentionally did not apply trial-wise filtering.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from the per-day `trace` array in each animal’s joblib file.

ii.
```python
trace_all = dat[animal]['trace']      # (n_days, n_cells, n_frames)
...
def process_session(trace_day, position_day, env_name):
    n_cells, n_frames = trace_day.shape
```

iii. The notes describe `trace` as “binary calcium events (0/1), NaN for unregistered cells” and map `trace` directly to the target `neural` field.

## 2-b. How is the `neural` data processed?

i. The AI removes unregistered cells, replaces remaining NaNs with zero, Gaussian-smooths each cell’s trace with sigma 3 frames, then averages in non-overlapping 3-frame windows using `AvgPool1d`, yielding continuous-valued 100 ms bins.

ii.
```python
registered_mask = ~np.isnan(trace_day[:, 0])
registered_trace = trace_day[registered_mask]
...
registered_trace = np.nan_to_num(registered_trace, nan=0.0)
...
smoothed_trace = gaussian_filter1d(registered_trace.astype(np.float32),
                                    sigma=GAUSS_SIGMA, axis=1)
pooling = AvgPool1d(kernel_size=TEMPORAL_BIN_SIZE, stride=TEMPORAL_BIN_SIZE)
neural_binned = pooling(torch.tensor(smoothed_trace)).numpy()
```

iii. The notes justify this as matching the decoder utilities: “fit_decoder applies: `gaussian_filter1d(..., sigma=3)` then `AvgPool1d(kernel_size=3)`.” The trajectory also shows the AI reacting to the verifier’s warning that purely binary 0/1 neural values would fail format checks.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI keeps all registered cells in a session and excludes cells that appear unregistered on that day. It does not filter to place cells and does not apply the decoder code’s `>5` activity threshold or velocity-based filtering.

ii.
```python
registered_mask = ~np.isnan(trace_day[:, 0])
registered_trace = trace_day[registered_mask]
...
if n_registered == 0:
    return [], [], [], 0
```

iii. The notes state: “Use ALL registered cells per session (not just place cells)” and “No velocity filtering.” They explicitly distinguish these from curation steps used inside `decode_position_within`.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI does not align to an experimental event. It treats the start of each artificial 1-minute segment as the alignment point and records that in metadata.

ii.
```python
'metadata': {
    'temporal_alignment_event': 'Start of 1-minute trial segment within 40-minute recording session',
    'off_start': 0.0,
    'off_end': float(TRIAL_DURATION_SEC),
```

iii. The notes frame the trials as segments of continuous recording rather than event-locked trials, so the segment start is the only alignment event the AI defines.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI uses 100 ms bins. Yes: it rebins from 30 Hz frames to 3-frame windows after Gaussian smoothing.

ii.
```python
TEMPORAL_BIN_SIZE = 3
TIME_BIN_MS = 1000.0 * TEMPORAL_BIN_SIZE / FPS  # 100 ms
...
pooling = AvgPool1d(kernel_size=TEMPORAL_BIN_SIZE, stride=TEMPORAL_BIN_SIZE)
```

iii. The notes repeatedly justify this as copying the temporal binning inside the paper’s decoder functions rather than preserving the native frame rate.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The AI derives the decoder input from `envs` session labels, not from the raw `blocked` field.

ii.
```python
envs_all = dat[animal]['envs']
...
env_name = envs_all[day, 0] if envs_all.ndim > 1 else envs_all[day]
...
env_mat = get_env_mat(env_name)
env_flat = env_mat.flatten()
```

iii. The notes say “`envs → get_env_mat()`” and justify the input as environment geometry rather than blocked locations.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The AI maps the environment name string to a 3x3 binary matrix describing present vs omitted partitions, then flattens that matrix to a length-9 vector and reuses the same vector for every trial in the session.

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
env_mat = get_env_mat(env_name)
env_flat = env_mat.flatten()  # (9,)
...
input_trial = env_flat.astype(np.float32)
```

iii. The notes call this “Environment input: Use `get_env_mat()` to convert environment name to 3x3 binary matrix. Flatten to 9 values.”

## 3-c. How is `input` *Environment geometry* aligned with the neural data?

i. It is static within a session and copied once per trial, so alignment is by session/trial identity rather than by timepoint.

ii.
```python
for t in range(n_trials):
    ...
    input_trial = env_flat.astype(np.float32)  # (9,) static per trial
    input_trials.append(input_trial)
```

iii. The notes explicitly describe the environment geometry input as “Static per trial (1D array of 9 values).”

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. `output` is derived from the per-day `position` array in each animal’s joblib file.

ii.
```python
position_all = dat[animal]['position']  # (n_days, 2, n_frames)
...
def process_session(trace_day, position_day, env_name):
```

iii. The notes describe `position` as raw x,y location in centimeters and map it directly to the decoder output.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The AI first temporally averages x and y with the same 3-frame pooling used for neural data, then discretizes the pooled positions onto a 3x3 grid over the 75 cm arena.

ii.
```python
pos_binned_temporal = pooling(torch.tensor(position_day.astype(np.float32))).numpy()
...
bin_size = (POSITION_MAX + BUFFER) / N_SPATIAL_BINS
pos_bins = (pos_binned_temporal / bin_size).astype(int)
pos_bins = np.clip(pos_bins, 0, N_SPATIAL_BINS - 1)
```

iii. The notes say “Position discretization: Bin x,y position into 3x3 grid” and “Applied to both neural and position data” when describing temporal binning.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Each pooled x and y coordinate is converted to integer bin indices 0, 1, or 2 by dividing by the 25 cm bin width and clipping to valid range. The AI combines them as `x_bin * 3 + y_bin`, yielding categories 0-8.

ii.
```python
bin_size = (POSITION_MAX + BUFFER) / N_SPATIAL_BINS
pos_bins = (pos_binned_temporal / bin_size).astype(int)
pos_bins = np.clip(pos_bins, 0, N_SPATIAL_BINS - 1)
pos_category = pos_bins[0] * N_SPATIAL_BINS + pos_bins[1]
```

iii. The notes state the bins are `[0-25), [25-50), [50-75]` cm and category is `bin_x * 3 + bin_y = 0-8`.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. The AI aligns neural and position data by applying the same 3-frame temporal pooling and then slicing both with the same trial start and end bin indices.

ii.
```python
neural_binned = pooling(torch.tensor(smoothed_trace)).numpy()
pos_binned_temporal = pooling(torch.tensor(position_day.astype(np.float32))).numpy()
...
for t in range(n_trials):
    t_start = t * TIME_BINS_PER_TRIAL
    t_end = (t + 1) * TIME_BINS_PER_TRIAL
    neural_trial = neural_binned[:, t_start:t_end].astype(np.float32)
    output_trial = pos_category[t_start:t_end].astype(np.int64).reshape(1, -1)
```

iii. The notes repeatedly say that position is “temporally bin[ned] same as neural.”

## 5-a. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data are 100 ms per time bin, produced by 3-frame temporal averaging after smoothing for neural data and by 3-frame temporal averaging for position.

ii.
```python
TIME_BIN_MS = 1000.0 * TEMPORAL_BIN_SIZE / FPS  # 100 ms
...
pooling = AvgPool1d(kernel_size=TEMPORAL_BIN_SIZE, stride=TEMPORAL_BIN_SIZE)
```

iii. The notes justify this as copying the reference decoder’s internal temporal preprocessing.

## 5-b. How are the neural, input, and output data temporally aligned?

i. Neural and output position are rebinned on the same 3-frame windows and cut into the same 1-minute trial windows. Input environment geometry is static and attached once per trial, so it is aligned at the trial/session level rather than per timepoint.

ii.
```python
neural_binned = pooling(torch.tensor(smoothed_trace)).numpy()
pos_binned_temporal = pooling(torch.tensor(position_day.astype(np.float32))).numpy()
...
for t in range(n_trials):
    ...
    input_trial = env_flat.astype(np.float32)
    output_trial = pos_category[t_start:t_end].astype(np.int64).reshape(1, -1)
```

iii. The notes describe the neural and position streams as using the same temporal binning and the input as static per trial.

## 6. How are minor issues in the data (e.g., missing data, malformed entries) handled?

i. Unregistered cells are removed via a NaN-based mask; sessions with zero registered cells are skipped; any remaining NaNs in retained cells are replaced with zero; incomplete trailing data are dropped by floor division; and unknown environment labels produce a 3x3 matrix of NaNs.

ii.
```python
registered_mask = ~np.isnan(trace_day[:, 0])
...
if n_registered == 0:
    return [], [], [], 0
...
registered_trace = np.nan_to_num(registered_trace, nan=0.0)
...
return env_mats.get(env, np.full((3,3), np.nan)).astype(float)
...
n_trials = n_timebins_total // TIME_BINS_PER_TRIAL
```

iii. The notes emphasize NaN-based registration and state that there were no major anomalies in spot checks. The code itself contains the more specific fallback behavior for remaining NaNs and unknown environments.

## 7-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading the large per-animal joblib datasets. The AI’s notes estimate loading at roughly 20-80 seconds per animal, much larger than per-session processing time.

ii.
```python
t0 = time.time()
dat = joblib.load(os.path.join(data_dir, animal))
print(f"  Loaded in {time.time()-t0:.1f}s")
```

iii. `CONVERSION_NOTES.md` includes a runtime table with “Loading: ~20-80s per animal” and “Processing: ~0.7-1.0s per session.”

## 7-b. What loops in the code could have been vectorized to improve efficiency?

i. The main vectorizable loop is the per-trial append loop inside `process_session`, which could have been replaced by reshaping/splitting precomputed arrays once. The animal/day loops are structural, but the trial construction loop is Python overhead.

ii.
```python
neural_trials = []
input_trials = []
output_trials = []

for t in range(n_trials):
    t_start = t * TIME_BINS_PER_TRIAL
    t_end = (t + 1) * TIME_BINS_PER_TRIAL
    neural_trial = neural_binned[:, t_start:t_end].astype(np.float32)
    input_trial = env_flat.astype(np.float32)
    output_trial = pos_category[t_start:t_end].astype(np.int64).reshape(1, -1)
    neural_trials.append(neural_trial)
    input_trials.append(input_trial)
    output_trials.append(output_trial)
```

iii. The AI did not explicitly discuss vectorization in its notes. This follows from the implemented code structure.

## 7-c. What processing does the code repeat multiple times?

i. It repeatedly constructs `AvgPool1d` for each session, repeatedly casts the same flattened environment vector to `float32` once per trial, and repeatedly performs trial slicing in Python lists instead of batching.

ii.
```python
pooling = AvgPool1d(kernel_size=TEMPORAL_BIN_SIZE, stride=TEMPORAL_BIN_SIZE)
...
for t in range(n_trials):
    ...
    input_trial = env_flat.astype(np.float32)
```

iii. The AI’s notes focus on scientific choices, not these micro-efficiency costs; the repetition is evident from the code.

## 7-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. There is no major discarded scientific transform in the main conversion path, but the script includes optional plotting and extra timing/summary bookkeeping that are not used downstream by the decoder.

ii.
```python
if show_processing and sessions_processed <= 2:
    save_processing_plot(...)
...
print(f"\n=== Conversion Summary ===")
print(f"Sessions: {total_sessions}")
```

iii. The notes describe these as sanity checks and validation aids rather than part of the target dataset itself.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Same as section 6: NaN-marked unregistered cells are removed, sessions with no registered cells are skipped, remaining NaNs are filled with zero, unknown environments become all-NaN geometry matrices, and incomplete trailing data are discarded.

ii.
```python
registered_mask = ~np.isnan(trace_day[:, 0])
if n_registered == 0:
    return [], [], [], 0
registered_trace = np.nan_to_num(registered_trace, nan=0.0)
return env_mats.get(env, np.full((3,3), np.nan)).astype(float)
n_trials = n_timebins_total // TIME_BINS_PER_TRIAL
```

iii. This is the same handling described in section 6; the AI’s notes emphasize NaN-based registration and successful validation.

## 9-a. What are the most time-consuming steps of the code?

i. Same as section 7-a: the dominant cost is loading large joblib files for each animal; per-session computation is comparatively smaller.

ii.
```python
t0 = time.time()
dat = joblib.load(os.path.join(data_dir, animal))
print(f"  Loaded in {time.time()-t0:.1f}s")
```

iii. The runtime estimates in `CONVERSION_NOTES.md` attribute most wall-clock time to loading.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Same as section 7-b: the per-trial loop inside `process_session` is the clearest vectorization target.

ii.
```python
for t in range(n_trials):
    t_start = t * TIME_BINS_PER_TRIAL
    t_end = (t + 1) * TIME_BINS_PER_TRIAL
    neural_trials.append(neural_binned[:, t_start:t_end].astype(np.float32))
    input_trials.append(env_flat.astype(np.float32))
    output_trials.append(pos_category[t_start:t_end].astype(np.int64).reshape(1, -1))
```

iii. The AI did not document this explicitly; it is inferred from the code path.

## 9-c. What processing does the code repeat multiple times?

i. Same as section 7-c: repeated pooling-module construction, repeated `astype(np.float32)` on the same environment vector, and repeated Python slicing/append work for each trial.

ii.
```python
pooling = AvgPool1d(kernel_size=TEMPORAL_BIN_SIZE, stride=TEMPORAL_BIN_SIZE)
...
input_trial = env_flat.astype(np.float32)
```

iii. The repetition is implicit in the implementation rather than stated in the notes.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Same as section 7-d: there is no major discarded scientific transform in the main conversion path, but optional plotting and diagnostic logging are extra work outside the saved dataset.

ii.
```python
if show_processing and sessions_processed <= 2:
    save_processing_plot(...)
...
print(f"\n=== Conversion Summary ===")
```

iii. The AI treated these as validation/sanity-check support rather than decoder inputs.
