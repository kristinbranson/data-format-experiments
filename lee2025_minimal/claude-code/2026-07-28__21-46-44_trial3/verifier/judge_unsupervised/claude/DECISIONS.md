# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data for each of the 7 animals individually using `joblib.load()`, which reads pre-saved compressed joblib files from `/app/data/`. Each file contains a dictionary keyed by the animal ID, with sub-fields including `trace` (neural data), `position` (x-y coordinates), and `envs` (environment labels). This matches the reference code's `load_dat()` function which also uses `joblib.load()` for the "joblib" format.

ii.
```python
for animal_idx, animal in enumerate(animals):
    dat = joblib.load(os.path.join(data_dir, animal))
    d = dat[animal]
```

iii. The AI's CONVERSION_NOTES.md states: "Each animal has multiple recording sessions (days), each ~40 min at 30 Hz." The agent read the reference `load_dat()` function in `utils.py` and replicated the same loading pattern (joblib format, nested dictionary with animal ID as key).

## 1-b. How are the data split into subjects?

i. Subjects are the 7 mice. The AI hardcodes the list of all 7 animal IDs and iterates over them. Each animal's data is loaded separately, and a subject index is maintained to map sessions back to subjects.

ii.
```python
ANIMALS = [
    "QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
    "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"
]
# ...
subjects = list(animals)
# ...
subject_idx_list.append(animal_idx)
```

iii. The AI verified against the paper: "7 animals (mice): QLAK-CA1-08, QLAK-CA1-30, QLAK-CA1-50, QLAK-CA1-51, QLAK-CA1-56, QLAK-CA1-74, QLAK-CA1-75." The animal IDs match the data files in `/app/data/`.

## 1-c. How are the data split into sessions?

i. Each recording day is treated as one session. The AI iterates over the day dimension (axis 0) of the `trace` array, which has shape `(n_days, n_cells, n_timepoints)`. One session = one day = one environment geometry.

ii.
```python
n_days = d['trace'].shape[0]
for day in range(n_days):
    trace_day = d['trace'][day]  # (n_cells, n_timepoints)
    pos_day = d['position'][day]  # (2, n_timepoints)
    env_name = str(d['envs'][day, 0])
```

iii. The CONVERSION_NOTES state: "Each session = one recording day (one environment geometry per day)." The paper states "one session was recorded per day." This yields 207 total sessions matching the paper's count.

## 1-d. How are the data split into trials?

i. Each session is split into 1-minute segments (trials) of 1800 frames (30 Hz * 60 seconds). The number of complete trials per session is computed by integer division of total timepoints by frames-per-trial. Remainder frames at the end of a session are discarded.

ii.
```python
FRAMES_PER_TRIAL = FPS * TRIAL_DURATION_S  # 1800 frames per trial
n_trials = n_timepoints // FRAMES_PER_TRIAL

for t in range(n_trials):
    start = t * FRAMES_PER_TRIAL
    end = (t + 1) * FRAMES_PER_TRIAL
    trial_traces = traces[:, start:end]
    trial_pos = pos_bins[start:end]
```

iii. The instructions specify "1-minute trials within each session." The AI notes that 40-minute sessions at 30 Hz yield ~71,866 frames, giving 39-40 trials per session with a small remainder discarded.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies two session-level filters: (1) sessions with fewer than 5 registered cells are skipped, and (2) sessions with fewer than 2 possible trials are skipped. No trial-level quality filtering is applied (e.g., no speed filtering, no filtering for time spent in blocked regions).

ii.
```python
if n_registered < 5:
    print(f"  Day {day}: skipping, only {n_registered} registered cells")
    continue

if n_trials < 2:
    print(f"  Day {day}: skipping, only {n_trials} possible trials")
    continue
```

iii. The CONVERSION_NOTES state: "No additional filtering applied - paper states 'motivated the inclusion of all cells in subsequent analyses.'" The minimum 2 trials requirement comes from the instructions ("at least two trials within each session"). The 5-cell threshold is the AI's own safety check; in practice, no sessions are dropped by either filter (all 207 sessions are retained). Notably, the reference code's `decode_position_within()` function applies velocity filtering (v_thresh=5, v_filt_size=5) to exclude timepoints when the mouse is stationary, but the AI did not implement or discuss this filtering.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from the `trace` field in the raw data, which contains binary calcium transient events (0 or 1). These are the binarized rising-phase vectors from the calcium imaging preprocessing pipeline described in the paper.

ii.
```python
trace_day = d['trace'][day]  # (n_cells, n_timepoints)
```

iii. The CONVERSION_NOTES state: "Binary calcium transient events from rise-phase extraction (z-score > 2.5 threshold). Values: 0 (no event) or 1 (significant transient event)." The paper confirms: "The final binarized rising-phase vector was then set to 1 whenever this z-scored vector exceeded 2.5, and 0 otherwise."

## 2-b. How is the `neural` data processed?

i. The AI selects only registered cells (those with non-NaN traces for that day), replaces any remaining NaN values with 0 as a safety measure, and casts to float32. No additional processing (smoothing, normalization, etc.) is applied to the binary trace data.

ii.
```python
registered = ~np.all(np.isnan(trace_day), axis=1)
traces = trace_day[registered]  # (n_registered, n_timepoints)
traces = np.nan_to_num(traces, nan=0.0)
# ...
neural_trials.append(trial_traces.astype(np.float32))
```

iii. The CONVERSION_NOTES state: "Only registered cells (non-NaN traces) included per session." The paper says "All analyses were conducted using the binary vector of the rising phases of transients," so no further processing is needed beyond using the binary traces.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI includes ALL registered cells without any place cell filtering or other quality-based neuron selection. A cell is considered "registered" for a given day if its trace is not entirely NaN for that day (cells tracked via CellReg across sessions may not be present every day).

ii.
```python
registered = ~np.all(np.isnan(trace_day), axis=1)
n_registered = registered.sum()
```

iii. The CONVERSION_NOTES state: "No additional filtering applied - paper states 'motivated the inclusion of all cells in subsequent analyses.'" The paper indeed mentions including all cells for its broader analyses. However, the reference code's `decode_position_within()` function does apply a `cell_threshold=5` filter, excluding cells with fewer than 5 active frames during velocity-filtered time periods. The AI did not consider or discuss this activity-based cell filtering.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to the start of each 1-minute segment from the beginning of the recording session. Since the DAQ simultaneously acquired behavioral and cellular imaging streams at 30 Hz, neural and behavioral data are natively aligned frame-by-frame. There is no explicit alignment to a specific behavioral event.

ii.
```python
'temporal_alignment_event': 'Start of recording session',
'off_start': 0.0,
'off_end': TRIAL_DURATION_S,
```

iii. The CONVERSION_NOTES state: "All time series (neural, position) are natively aligned at 30 Hz recording rate. DAQ simultaneously acquired behavioral and cellular imaging streams at 30 Hz."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is 33.33 ms (1000/30 Hz), the native recording rate. No temporal rebinning is applied.

ii.
```python
FPS = 30  # recording frame rate
# ...
'time_bin_size': 1000.0 / FPS,  # ~33.33 ms
```

iii. The CONVERSION_NOTES state: "Time bin size: 33.33 ms (1000/30)." The data is used at its native acquisition rate without downsampling or rebinning.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The input is derived from the `envs` field in the raw data, which contains the environment name (string) for each recording day. The environment name is then mapped to a 3x3 binary matrix via the `get_env_mat()` function.

ii.
```python
env_name = str(d['envs'][day, 0])
env_mat = get_env_mat(env_name)
env_input = env_mat.flatten()  # shape (9,)
```

iii. The AI directly replicated the `get_env_mat()` function from the reference code (`utils.py`), mapping each of 10 environment names to its corresponding 3x3 binary matrix.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The environment name string is mapped to a 3x3 binary matrix (1=open, 0=blocked partition), then flattened to a 9-element vector in row-major order. The input is static per trial (same environment for all timepoints within a session).

ii.
```python
def get_env_mat(env):
    env_mats = {
        'square':    [[1,1,1],[1,1,1],[1,1,1]],
        'o':         [[1,1,1],[1,0,1],[1,1,1]],
        # ... (10 environments total)
    }
    return np.array(env_mats[env], dtype=float)

# Usage:
env_mat = get_env_mat(env_name)
env_input = env_mat.flatten()  # shape (9,)
input_trials.append(env_input.astype(np.float32))
```

iii. The 3x3 matrices match the reference code's `get_env_mat()` function exactly. The input is stored as a 1D vector of length 9, static per trial.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The output is derived from the `position` field in the raw data, which contains x-y coordinates of the mouse's position at each timepoint, shape `(2, n_timepoints)` per day.

ii.
```python
pos_day = d['position'][day]  # (2, n_timepoints)
```

iii. The paper states: "Position data were generated from tracking the head with DeepLabCut pose-estimation software." The AI correctly uses the `position` field from the dataset.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The continuous x-y position is discretized into a 3x3 grid of spatial bins. Each dimension is divided into 3 equal bins of 25 cm each (75 cm arena / 3 bins). The x-y position is clipped to the arena bounds, then assigned to bins using floor division by the bin size.

ii.
```python
def discretize_position(position, n_bins=N_SPATIAL_BINS, arena_size=ARENA_SIZE):
    x = np.clip(position[0], 0, arena_size - 1e-10)
    y = np.clip(position[1], 0, arena_size - 1e-10)
    bin_size = arena_size / n_bins
    x_bin = np.floor(x / bin_size).astype(int)
    y_bin = np.floor(y / bin_size).astype(int)
    x_bin = np.clip(x_bin, 0, n_bins - 1)
    y_bin = np.clip(y_bin, 0, n_bins - 1)
    bin_idx = x_bin * n_bins + y_bin
    return bin_idx
```

iii. The instructions specify "Mouse position discretized into 3 x 3 = 9 spatial bins." The AI uses bins of 25 cm each which corresponds to the physical grid partition size (75/3 = 25 cm).

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Position is converted to categorical bin indices (0-8) using row-major indexing: `bin = row * 3 + col`. Position is clipped to [0, arena_size - epsilon] before binning, then bins are clamped to [0, n_bins-1]. No additional thresholding or filtering of position values is applied (e.g., no masking of timepoints when the mouse is in blocked regions, no speed threshold).

ii.
```python
x_bin = np.floor(x / bin_size).astype(int)
y_bin = np.floor(y / bin_size).astype(int)
x_bin = np.clip(x_bin, 0, n_bins - 1)
y_bin = np.clip(y_bin, 0, n_bins - 1)
bin_idx = x_bin * n_bins + y_bin
```

iii. The output bin names are: `row0_col0` (0) through `row2_col2` (8). The verification output shows the position distribution is non-uniform across bins, with `row2_col2` being most common (20%) and `row1_col1` least common (5.7%).

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position and neural data share the same timepoints at 30 Hz, since both were simultaneously acquired by the DAQ. The position is first discretized for the entire session, then sliced into 1-minute trials using the same frame indices as the neural data.

ii.
```python
pos_bins = discretize_position(pos_day)  # shape (n_timepoints,)

for t in range(n_trials):
    start = t * FRAMES_PER_TRIAL
    end = (t + 1) * FRAMES_PER_TRIAL
    trial_traces = traces[:, start:end]
    trial_pos = pos_bins[start:end]
```

iii. The AI notes: "DAQ simultaneously acquired behavioral and cellular imaging streams at 30 Hz," confirming native alignment.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles missing data in two ways: (1) Cells with entirely NaN traces for a day are excluded as "unregistered" for that session. (2) Any remaining partial NaN values in registered cells' traces are replaced with 0 using `np.nan_to_num()`. No handling is applied for missing or out-of-bounds position data beyond clipping to the arena bounds.

ii.
```python
registered = ~np.all(np.isnan(trace_day), axis=1)
traces = trace_day[registered]
traces = np.nan_to_num(traces, nan=0.0)
```

iii. The CONVERSION_NOTES state: "Replace any remaining NaN with 0 (shouldn't happen but safety)." The AI treats this as a defensive measure rather than an expected case.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are: (1) Loading each animal's data via `joblib.load()`, as each file is a large compressed dataset containing trace, position, and other fields for all days. (2) The conversion loop that iterates over all days, selects registered cells, discretizes position, and splits into trials. The full conversion output file (19.98 GB pickle) also indicates significant I/O time for saving.

ii.
```python
dat = joblib.load(os.path.join(data_dir, animal))
```

iii. The conversion log shows 7 animals loaded sequentially, each with 21-31 days of data. The output file is ~20 GB, indicating significant data volume.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial-splitting loop iterates over trials sequentially using Python for-loops, appending to lists. This could be vectorized using `np.reshape()` or array slicing to split the full session data into fixed-size trial blocks in one operation.

ii.
```python
for t in range(n_trials):
    start = t * FRAMES_PER_TRIAL
    end = (t + 1) * FRAMES_PER_TRIAL
    trial_traces = traces[:, start:end]
    trial_pos = pos_bins[start:end]
    neural_trials.append(trial_traces.astype(np.float32))
```

iii. The loop body is simple array slicing, so the overhead is primarily Python loop iteration and list appending. Vectorizing with reshape (e.g., `traces[:, :n_trials*FRAMES_PER_TRIAL].reshape(n_registered, n_trials, FRAMES_PER_TRIAL)`) would be more efficient.

## 6-c. What processing does the code repeat multiple times?

i. The environment geometry input (`env_input`) is computed once per session but then copied identically for every trial within that session. The `get_env_mat()` function could also be cached since the same environment names recur across sessions and animals.

ii.
```python
env_mat = get_env_mat(env_name)
env_input = env_mat.flatten()

for t in range(n_trials):
    # Same env_input appended for every trial
    input_trials.append(env_input.astype(np.float32))
```

iii. The environment input is static per session, so the `.astype(np.float32)` conversion is repeated ~39 times per session unnecessarily. However, this is a very minor efficiency concern.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code performs position clipping twice: first to `[0, arena_size - 1e-10]` and then `np.clip(x_bin, 0, n_bins-1)` on the bin indices. The second clip is redundant given the first. Additionally, the code processes all timepoints within each trial at 30 Hz, whereas downstream decoder analysis may use dimensionality reduction (SVD) that subsamples timepoints. The full 30 Hz resolution may be unnecessarily fine-grained for the decoder, but this is consistent with the instructions which say to keep the native temporal resolution.

ii.
```python
x = np.clip(position[0], 0, arena_size - 1e-10)
# ...
x_bin = np.clip(x_bin, 0, n_bins - 1)  # redundant given the above clip
```

iii. The redundant clipping is a minor inefficiency. The 20 GB output file size suggests no compression or data reduction is applied, which may be unnecessary for downstream use but preserves all information as instructed.
