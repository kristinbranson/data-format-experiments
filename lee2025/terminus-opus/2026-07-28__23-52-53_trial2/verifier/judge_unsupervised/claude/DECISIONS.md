# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data by iterating over a hardcoded list of 7 animal IDs (`ANIMALS`). For each animal, it calls `joblib.load(os.path.join(data_dir, animal))` to load the joblib file, then accesses `dat[animal]['trace']` (shape: n_days, n_cells, n_frames), `dat[animal]['position']` (shape: n_days, 2, n_frames), and `dat[animal]['envs']` (shape: n_days, 1). This mirrors the reference code's `load_dat` function.

ii.
```python
dat = joblib.load(os.path.join(data_dir, animal))
trace_all = dat[animal]['trace']      # (n_days, n_cells, n_frames)
position_all = dat[animal]['position']  # (n_days, 2, n_frames)
envs_all = dat[animal]['envs']          # (n_days, 1)
```

iii. The AI identified from reading the reference code's `load_dat` function (utils.py:61) that data is stored in joblib format with animal ID as the key. The AI confirmed the data structure by exploring the data files directly and verified that 7 animal files exist in the data directory.

## 1-b. How are the data split into subjects?

i. Each animal file corresponds to one subject. The 7 animal IDs are hardcoded in the `ANIMALS` list. Each animal's data is processed separately, and a `subject_idx` array maps sessions to their corresponding animal index.

ii.
```python
ANIMALS = ["QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
           "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"]
subjects = list(animals)
# ...
all_subject_idx.append(animal_idx)
```

iii. The AI identified 7 animal data files in the data directory and hardcoded them. The subject list matches the paper's dataset description.

## 1-c. How are the data split into sessions?

i. Each day (recording session) within each animal is treated as one session. The AI iterates over `n_days` for each animal. One animal (QLAK-CA1-51) has 21 days; the other 6 have 31 days each, giving 207 total sessions, which matches the paper.

ii.
```python
n_days = trace_all.shape[0]
for day in range(n_days):
    # ... process each day as a session
    neural_trials, input_trials, output_trials, n_registered = process_session(
        trace_all[day], position_all[day], env_name
    )
```

iii. The AI verified that total sessions = 207, matching the paper's statement of "207 sessions." Sessions per subject: 31/31/31/21/31/31/31.

## 1-d. How are the data split into trials?

i. Each ~40-minute recording session is split into 1-minute trials. At 30 Hz with temporal binning of 3 frames, each trial is 1800 frames = 600 time bins. The number of complete trials per session is `n_timebins_total // TIME_BINS_PER_TRIAL`, and remainder frames are discarded.

ii.
```python
TRIAL_DURATION_SEC = 60  # 1 minute trials
FRAMES_PER_TRIAL = TRIAL_DURATION_SEC * FPS  # 1800 frames
TIME_BINS_PER_TRIAL = FRAMES_PER_TRIAL // TEMPORAL_BIN_SIZE  # 600 time bins

n_timebins_total = neural_binned.shape[1]
n_trials = n_timebins_total // TIME_BINS_PER_TRIAL

for t in range(n_trials):
    t_start = t * TIME_BINS_PER_TRIAL
    t_end = (t + 1) * TIME_BINS_PER_TRIAL
    neural_trial = neural_binned[:, t_start:t_end]
```

iii. The AI followed the task instruction: "split into 1-minute trials within each session." Sessions are ~40 minutes long (71866-72219 frames), yielding 39-40 trials per session, totaling 8187 trials across all 207 sessions.

## 1-e. How are trials filtered based on quality controls?

i. No explicit trial-level quality filtering is performed. Sessions with zero registered neurons are skipped (though none exist in practice). The only filtering is discarding remainder frames at the end of each session that don't form a complete 1-minute trial.

ii.
```python
if len(neural_trials) == 0:
    print(f"  Day {day}: skipped (no registered cells)")
    continue
```

iii. The AI noted in CONVERSION_NOTES.md: "No explicit trial-level curation in the paper (sessions are continuous recordings)." The reference code's velocity filtering is per-timepoint, not per-trial, so there is no trial-level curation to match.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from the `trace` field in the raw data, which contains binarized rising-phase calcium events (0/1 values). Cells not registered on a given day have NaN values.

ii.
```python
trace_all = dat[animal]['trace']  # (n_days, n_cells, n_frames)
# In process_session:
registered_mask = ~np.isnan(trace_day[:, 0])
registered_trace = trace_day[registered_mask]  # (n_registered, n_frames)
```

iii. The AI identified from the reference code and paper that the trace data is "already binarized rising-phase calcium events (0/1)" and that "Binary vector treated as firing rate in all analyses." This is pre-processed data — no delta F/F computation is needed.

## 2-b. How is the `neural` data processed?

i. The neural data is processed in two steps matching the reference `fit_decoder` function: (1) Gaussian smoothing with sigma=3 frames along the time axis, then (2) temporal binning via AvgPool1d with kernel_size=3 and stride=3. This converts the binary trace into continuous firing rate estimates at 100ms resolution (10 Hz).

ii.
```python
smoothed_trace = gaussian_filter1d(registered_trace.astype(np.float32),
                                    sigma=GAUSS_SIGMA, axis=1)  # GAUSS_SIGMA=3
pooling = AvgPool1d(kernel_size=TEMPORAL_BIN_SIZE, stride=TEMPORAL_BIN_SIZE)  # k=3
neural_binned = pooling(torch.tensor(smoothed_trace)).numpy()  # (n_registered, n_timebins)
```

iii. The AI documented: "Reference: fit_decoder applies gaussian_filter1d(traces, sigma=temporal_bin_size, axis=0) then AvgPool1d(kernel_size=temporal_bin_size)." The sigma=3 and kernel_size=3 parameters were directly copied from the reference `fit_decoder` function.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI includes ALL registered cells per session — any cell whose trace is not all NaN on that day. The reference code's `decode_position_within` applies two additional filters: (1) velocity filtering (exclude non-moving timepoints, >5 cm/s), and (2) cell activity threshold (>5 events during movement periods). The AI deliberately chose NOT to apply either of these filters.

ii.
```python
# Only filter: registered cells (non-NaN)
registered_mask = ~np.isnan(trace_day[:, 0])
registered_trace = trace_day[registered_mask]
# No velocity filtering, no cell activity threshold
```

iii. From CONVERSION_NOTES.md Step 10, Check 3: "Neuron filtering: Including all registered cells (non-NaN trace). Reference decode_position_within uses cell_threshold>5 events, but that's specific to Bayesian decoder. We include all cells for neural network decoder. Justified difference." The AI reasoned that these filters were specific to the Bayesian decoder pipeline and not general preprocessing steps.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no explicit alignment event. Sessions are continuous 40-minute recordings without discrete trial events. The AI simply splits the processed data into consecutive 1-minute segments starting from the beginning of each session. The `temporal_alignment_event` is set to "Start of 1-minute trial segment within 40-minute recording session."

ii.
```python
'temporal_alignment_event': 'Start of 1-minute trial segment within 40-minute recording session',
'off_start': 0.0,
'off_end': float(TRIAL_DURATION_SEC),  # 60.0
```

iii. The AI correctly identified that this continuous recording paradigm has no natural alignment event (like stimulus onset). The trial boundaries are artificial 1-minute segments, with off_start=0 and off_end=60 seconds.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is 100 ms (3 frames at 30 Hz). Temporal rebinning IS applied: the raw 30 Hz data (33.3 ms per frame) is averaged into 100 ms bins using AvgPool1d with kernel_size=3, stride=3. This matches the reference `fit_decoder` function's `temporal_bin_size=3`.

ii.
```python
FPS = 30  # frames per second
TEMPORAL_BIN_SIZE = 3  # frames per temporal bin
TIME_BIN_MS = 1000.0 * TEMPORAL_BIN_SIZE / FPS  # 100 ms
```

iii. The AI documented: "Temporal binning: Use reference code approach: Gaussian smooth trace with sigma=3, then AvgPool1d with kernel_size=3. This gives 100ms time bins at ~10 Hz."

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The input is derived from the `envs` field in the raw data, which contains environment name strings (e.g., 'square', 'o', 't', etc.) for each session day.

ii.
```python
envs_all = dat[animal]['envs']  # (n_days, 1)
env_name = envs_all[day, 0] if envs_all.ndim > 1 else envs_all[day]
env_mat = get_env_mat(env_name)
env_flat = env_mat.flatten()  # (9,)
```

iii. The AI identified 10 unique environment geometries from the data and reference code. The environment name is used to look up the corresponding 3x3 binary matrix via `get_env_mat()`, which was copied from the reference code's `utils.py`.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The environment name string is converted to a 3x3 binary matrix using `get_env_mat()` (copied from the reference code). The matrix is then flattened to a 9-element vector. A value of 1 indicates an accessible region, 0 indicates a blocked region. This is static per trial (same for all timepoints within a session).

ii.
```python
def get_env_mat(env):
    env_mats = {
        'square':    np.array([[1,1,1],[1,1,1],[1,1,1]]),
        'o':         np.array([[1,1,1],[1,0,1],[1,1,1]]),
        # ... 10 environments total
    }
    return env_mats.get(env, np.full((3,3), np.nan)).astype(float)

# Per trial:
input_trial = env_flat.astype(np.float32)  # (9,) static per trial
```

iii. The AI copied `get_env_mat()` directly from the reference code (utils.py:215). The 9 input features represent the flattened 3x3 environment geometry grid.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The output is derived from the `position` field in the raw data, which contains x,y coordinates in cm (range 0-75) at the original 30 Hz frame rate.

ii.
```python
position_all = dat[animal]['position']  # (n_days, 2, n_frames)
# In process_session:
pos_binned_temporal = pooling(torch.tensor(position_day.astype(np.float32))).numpy()  # (2, n_timebins)
```

iii. The AI identified from the data exploration and reference code that position data is raw x,y coordinates tracked via DeepLabCut, ranging from 0 to 75 cm in both dimensions (75x75 cm arena).

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. Position is first temporally binned using the same AvgPool1d as the neural data (kernel_size=3, stride=3), then discretized into a 3x3 spatial grid. No Gaussian smoothing is applied to position before binning.

ii.
```python
pos_binned_temporal = pooling(torch.tensor(position_day.astype(np.float32))).numpy()  # (2, n_timebins)
bin_size = (POSITION_MAX + BUFFER) / N_SPATIAL_BINS  # ~25 cm per bin
pos_bins = (pos_binned_temporal / bin_size).astype(int)
pos_bins = np.clip(pos_bins, 0, N_SPATIAL_BINS - 1)
pos_category = pos_bins[0] * N_SPATIAL_BINS + pos_bins[1]  # (n_timebins,)
```

iii. The AI followed the reference code's pattern of applying the same temporal pooling to position data as to neural data. The 3x3 discretization was mandated by the task instructions ("Mouse position discretized into 3 x 3 = 9 spatial bins").

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Position is divided into a 3x3 grid where each bin spans 25 cm (position range 0-75 cm). The x and y coordinates are independently binned into 3 bins: [0-25), [25-50), [50-75]. The two bin indices are combined into a single category: `category = x_bin * 3 + y_bin`, yielding values 0-8.

ii.
```python
N_SPATIAL_BINS = 3
POSITION_MAX = 75.0
BUFFER = 1e-5
bin_size = (POSITION_MAX + BUFFER) / N_SPATIAL_BINS  # ~25.000003 cm
pos_bins = (pos_binned_temporal / bin_size).astype(int)
pos_bins = np.clip(pos_bins, 0, N_SPATIAL_BINS - 1)
pos_category = pos_bins[0] * N_SPATIAL_BINS + pos_bins[1]
```

iii. The AI chose equal-width bins (25 cm each) to divide the 75x75 cm arena, noting "The 3x3 grid matches the environment partition structure." A small buffer (1e-5) is added to prevent edge-case binning issues at position=75.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position and neural data are aligned by applying the same temporal binning (AvgPool1d with kernel_size=3, stride=3) to both streams before splitting into trials. Both are split at the same trial boundaries. This ensures the output has the same number of time bins (600) as the neural data per trial.

ii.
```python
# Same pooling applied to both neural and position:
pooling = AvgPool1d(kernel_size=TEMPORAL_BIN_SIZE, stride=TEMPORAL_BIN_SIZE)
neural_binned = pooling(torch.tensor(smoothed_trace)).numpy()
pos_binned_temporal = pooling(torch.tensor(position_day.astype(np.float32))).numpy()

# Same trial boundaries for both:
neural_trial = neural_binned[:, t_start:t_end]
output_trial = pos_category[t_start:t_end]
```

iii. The AI ensured temporal alignment by processing both streams identically in time, using the same pooling kernel and trial segmentation indices.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several data issues: (1) NaN values in the trace (unregistered cells on a given day) are excluded by checking the first frame for NaN. (2) Any remaining NaN values in registered cells are replaced with 0. (3) Sessions with no registered cells are skipped. (4) Position values are clipped to valid range [0, N_SPATIAL_BINS-1]. (5) Remainder frames at end of sessions (that don't form a complete 1-minute trial) are discarded. (6) Variable session lengths (71866-72219 frames) are handled by computing trial count from actual frame count.

ii.
```python
registered_mask = ~np.isnan(trace_day[:, 0])
registered_trace = trace_day[registered_mask]
registered_trace = np.nan_to_num(registered_trace, nan=0.0)
# ...
pos_bins = np.clip(pos_bins, 0, N_SPATIAL_BINS - 1)
n_trials = n_timebins_total // TIME_BINS_PER_TRIAL  # discards remainder
```

iii. The AI documented in CONVERSION_NOTES.md Step 10, Check 5: "Frames per session varies (71866-72219). Handled by discarding remainder after last complete trial." and "Position range [0, 75] handled with buffer for binning."

## 6-a. What are the most time-consuming steps of the code?

i. Data loading (joblib.load) is the dominant cost, taking 9-80 seconds per animal depending on file size. Processing per session is relatively fast at ~0.7-1.0 seconds. Total conversion time for all 207 sessions was ~427 seconds (~7 minutes). Saving the final pickle file (6.35 GB) takes ~12 seconds.

ii.
```python
t0 = time.time()
dat = joblib.load(os.path.join(data_dir, animal))
print(f"  Loaded in {time.time()-t0:.1f}s")
```

iii. From conversion_full_out.txt: loading times ranged from 9.3s (QLAK-CA1-51, smallest) to 79.7s (QLAK-CA1-50). The AI estimated total time in Step 7 as ~380s and actual was 427s.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial-splitting loop (lines 128-138) iterates over trials one at a time to slice neural, input, and output data. This could be vectorized using `np.split` or array reshaping instead of a Python loop. However, since the loop is simple array slicing and the bottleneck is I/O, the performance impact is negligible.

ii.
```python
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

iii. The AI noted in CONVERSION_NOTES.md that per-session processing time was ~0.7-1s, which is small relative to loading time. The trial loop could be replaced with `np.array_split` or `reshape` but the gains would be minimal.

## 6-c. What processing does the code repeat multiple times?

i. The `AvgPool1d` pooling object is recreated inside `process_session` for every session call (line 96), though it could be created once and reused. The `env_flat.astype(np.float32)` conversion is repeated for every trial in a session even though the environment is the same for all trials in a session (lines 133). The `get_env_mat` dictionary lookup is done per session but results could be cached.

ii.
```python
# Recreated every session call:
pooling = AvgPool1d(kernel_size=TEMPORAL_BIN_SIZE, stride=TEMPORAL_BIN_SIZE)

# Repeated per trial:
input_trial = env_flat.astype(np.float32)  # same value every trial in session
```

iii. The AI did not explicitly document these repetitions. The pooling object recreation is trivially cheap. The env_flat conversion is also cheap but unnecessary to repeat per trial.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code processes the full session's data (all frames) before splitting into trials, meaning the remainder frames at the end of each session (frames beyond the last complete trial) are smoothed, temporally binned, and position-discretized, but then discarded. For sessions with 71866 frames, 71866 mod 1800 = 1666 frames (almost one extra trial's worth) are processed and thrown away. Similarly, Gaussian smoothing is applied to all neurons including those that may be completely inactive (all zeros), which contributes no information.

ii.
```python
# Full session processing (including remainder frames):
smoothed_trace = gaussian_filter1d(registered_trace.astype(np.float32), sigma=GAUSS_SIGMA, axis=1)
neural_binned = pooling(torch.tensor(smoothed_trace)).numpy()  # includes remainder bins

# Only complete trials are kept:
n_trials = n_timebins_total // TIME_BINS_PER_TRIAL  # truncates
```

iii. The AI did not discuss this inefficiency. Processing the full session before truncation is simpler and has negligible performance impact compared to the I/O cost, but it is technically unnecessary work.
