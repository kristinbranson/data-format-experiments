# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from preprocessed joblib files (one per animal) in the `data/` directory, using `joblib.load()`. Each file contains a dictionary keyed by the animal name, with arrays for `trace`, `position`, `envs`, `blocked`, etc. The AI hardcodes the list of 7 animal names in the `ANIMALS` constant.

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

iii. The AI identified both `.mat` files and joblib files in the data directory and chose to use the joblib files because they contain preprocessed Python data that is easier to work with. The reference code's `load_dat` function also supports loading from joblib files.

## 1-b. How are the data split into subjects?

i. Each joblib file corresponds to one subject (animal). The AI iterates over the hardcoded `ANIMALS` list, loading one file per animal. The subject ID is the animal name string.

ii.
```python
for animal in animals:
    neural, inp, out, sess_info = process_animal(
        animal, show_processing=show_processing
    )
    subject_id = ANIMALS.index(animal)
```

iii. The AI uses the same animal names as the reference paper (7 mice). Each file is identified by the animal name.

## 1-c. How are the data split into sessions?

i. Each animal's data has a `trace` array with shape `(n_days, n_cells, n_frames)`. The first dimension indexes recording sessions (days). Each day becomes a separate session in the output.

ii.
```python
n_days = d['trace'].shape[0]
for day in range(n_days):
    trace = d['trace'][day]
    position = d['position'][day]
```

iii. Each day is a separate recording session with its own environment geometry, neural recordings, and behavioral data.

## 1-d. How are the data split into trials?

i. Each session (~40 min recording) is split into 1-minute (1800-frame) non-overlapping segments. Unlike the reference, the AI also keeps partial trials at the end of a session if they are at least 30 seconds (900 frames) long, resulting in some trials with fewer than 1800 timepoints.

ii.
```python
FRAMES_PER_TRIAL = FPS * TRIAL_DURATION_S  # 1800 frames
MIN_TRIAL_FRAMES = FPS * 30  # Minimum 30s for a partial trial at end

trial_starts = list(range(0, n_frames, FRAMES_PER_TRIAL))
for start in trial_starts:
    end = min(start + FRAMES_PER_TRIAL, n_frames)
    trial_len = end - start
    if trial_len < MIN_TRIAL_FRAMES:
        continue
```

iii. The AI chose to keep partial trials of at least 30 seconds to avoid wasting data at the end of each session.

## 1-e. How are trials filtered based on quality controls?

i. Sessions with fewer than 2 trials are skipped (to allow decoder evaluation). Partial trials shorter than 30 seconds are dropped. No other trial-level filtering is applied.

ii.
```python
if n_trials < 2:
    print(f"  Day {day} ({env_name}): Only {n_trials} trial(s), skipping (need >=2)")
    continue
```

iii. The decoder requires at least 2 trials per session for train/validation splitting.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `trace` variable, which is a 3D array of shape `(n_days, n_cells, n_frames)` containing binarized calcium events (0/1 values).

ii.
```python
trace = d['trace'][day]  # (n_cells, n_frames) for this day
```

iii. The trace contains pre-processed binary calcium events extracted from calcium imaging using rising-phase extraction with a z-score threshold of 2.5.

## 2-b. How is the `neural` data processed?

i. Only registered (non-NaN) neurons are kept for each session. The data is cast to float32. No additional processing (smoothing, rate computation, etc.) is applied.

ii.
```python
registered_mask = ~np.isnan(trace[:, 0])
n_registered = np.sum(registered_mask)
trace_registered = trace[registered_mask]  # (n_registered, n_frames)
neural_trial = trace_registered[:, start:end].astype(np.float32)
```

iii. The AI noted that the trace is already binarized and treated as firing rate, so no further processing was needed. The paper states data was "treated as firing rate."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons that are NaN at the first timepoint are removed (assumed unregistered for that session). No place-cell filtering or activity threshold is applied.

ii.
```python
registered_mask = ~np.isnan(trace[:, 0])
```

iii. The AI noted that the paper uses ALL cells (not just place cells) for its analyses, quoting "motivated the inclusion of all cells in subsequent analyses."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. No event-based alignment is performed. The recording is continuous, and trials are simply sequential segments of the recording. Each trial is aligned to its own start (time 0 = start of the 1-minute segment).

ii. N/A (no alignment code; trials are just sequential slices of the continuous recording)

iii. There is no stimulus event to align to in this free-exploration paradigm.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The data is kept at the native 30 Hz frame rate (~33.33 ms per bin). No temporal rebinning is applied.

ii.
```python
FPS = 30  # Recording frame rate (Hz)
'time_bin_size': 1000.0 / FPS,  # ~33.33 ms
```

iii. The data is already at a consistent 30 Hz sampling rate, matching the recording frame rate described in the paper.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The input is derived from the `envs` variable (environment name string per session, e.g., 'square', 'o', 't'), which is then converted to a 3x3 binary matrix using the `get_env_mat` function from the reference code.

ii.
```python
env_name = envs[day]
env_mat = get_env_mat(env_name).flatten()  # (9,)
```

iii. The AI adopted the `get_env_mat` function directly from the reference code (`utils.py`), which maps environment names to 3x3 binary matrices where 1 = accessible and 0 = blocked.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The environment name string is mapped to a 3x3 binary matrix via `get_env_mat`, then flattened to a 9-element vector. The input is static per trial (same for all timepoints and all trials within a session).

ii.
```python
def get_env_mat(env):
    if env == 'square':
        return np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]).astype(float)
    elif env == 'o':
        return np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]]).astype(float)
    # ... etc for all 10 environments
input_trial = env_mat.astype(np.float32)  # (9,)
```

iii. This uses the reference code's environment geometry representation. Each of the 9 values represents whether a 25x25 cm partition of the arena is accessible (1) or blocked (0).

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. Output is derived from the `position` variable with shape `(2, n_frames)` containing x,y coordinates in cm within the 75x75 cm arena.

ii.
```python
position = d['position'][day]  # (2, n_frames)
```

iii. Position was tracked using DeepLabCut at the recording frame rate.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The continuous x,y position is discretized into a 3x3 grid (9 bins) using `np.floor` with a bin size of ~25 cm. The bin index is computed as `x_bin * 3 + y_bin` (row-major with x as the first dimension).

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

iii. The 3x3 grid matches the instruction requirement of 9 spatial bins. The small buffer (1e-5) prevents edge cases where position = 75.0 exactly would produce an out-of-bounds bin.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Position is divided into 3 equal bins along each axis (0-25, 25-50, 50-75 cm approximately), producing 9 categories (0-8). The bin index formula is `x_bin * 3 + y_bin`.

ii.
```python
bin_size = (env_size + buffer) / N_POS_BINS  # ~25.00 cm
x_bin = np.clip(np.floor(position[0] / bin_size).astype(int), 0, N_POS_BINS - 1)
y_bin = np.clip(np.floor(position[1] / bin_size).astype(int), 0, N_POS_BINS - 1)
bin_idx = x_bin * N_POS_BINS + y_bin
```

iii. Equal-width bins across the 75 cm arena. The AI uses `x_bin * 3 + y_bin` ordering, while the reference uses `y_bin * 3 + x_bin`.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position and neural data are recorded at the same 30 Hz frame rate and are frame-aligned in the source data. Both are sliced using the same start:end indices for each trial.

ii.
```python
pos_bins = bin_position_3x3(position)  # (n_frames,)
# Same start:end used for both:
neural_trial = trace_registered[:, start:end].astype(np.float32)
output_trial = pos_bins[start:end].astype(np.int64).reshape(1, -1)
```

iii. Since both signals share the same time axis, no interpolation or resampling is needed.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Neurons with NaN at the first timepoint (unregistered for that session) are excluded. Sessions with zero registered cells are skipped. Sessions producing fewer than 2 trials are skipped. Partial trials shorter than 30 seconds are dropped. No handling of NaN values within otherwise valid traces.

ii.
```python
registered_mask = ~np.isnan(trace[:, 0])
if n_registered == 0:
    print(f"  Day {day} ({env_name}): No registered cells, skipping")
    continue
if trial_len < MIN_TRIAL_FRAMES:
    continue
if n_trials < 2:
    continue
```

iii. The AI's approach assumes that if a neuron is non-NaN at the first frame, it is valid for the entire session. This is generally true for this dataset since NaN indicates unregistered cells.

## 6-a. What are the most time-consuming steps of the code?

i. Loading the large joblib files is the primary bottleneck. Each animal's data contains large 3D arrays (n_days x n_cells x n_frames). The full conversion takes ~160 seconds, dominated by I/O.

ii. N/A (timing is printed via `time.time()` calls throughout)

iii. The conversion output shows ~17-29 seconds per animal, with most time in the `joblib.load` call.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial splitting loop iterates over trial starts and creates individual trial arrays. This could potentially be vectorized using `np.split` or array reshaping, though the savings would be minor since the loop body is simple slicing.

ii.
```python
for start in trial_starts:
    end = min(start + FRAMES_PER_TRIAL, n_frames)
    trial_len = end - start
    if trial_len < MIN_TRIAL_FRAMES:
        continue
    neural_trial = trace_registered[:, start:end].astype(np.float32)
```

iii. The AI's code is already reasonably efficient with vectorized position binning. The trial-splitting loop is simple and not a significant bottleneck.

## 6-c. What processing does the code repeat multiple times?

i. The position binning and environment geometry encoding are computed once per session and reused across trials, so there is no redundant processing. However, the full `trace` array for each day is loaded even though only registered cells are used.

ii. N/A

iii. The code is reasonably efficient in this regard.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code loads the entire animal dictionary (including `maps`, `SFPs`, `centroids`, etc.) when only `trace`, `position`, and `envs` are needed. This wastes memory on unused fields.

ii.
```python
d = load_animal_data(animal)  # Loads entire dict including maps, SFPs, centroids, etc.
```

iii. The AI loads the full preprocessed data structure but only uses a small subset of the variables. The unused data is freed later with `del d`.
