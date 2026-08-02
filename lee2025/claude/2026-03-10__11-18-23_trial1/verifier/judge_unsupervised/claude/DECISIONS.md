# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data per-animal from joblib files in the `data/` directory. Each animal's file is a dictionary keyed by the animal ID, containing arrays for `trace`, `position`, `envs`, `blocked`, `maps`, `SFPs`, and `centroids`. The code iterates over a hardcoded list of 7 animal IDs (ANIMALS), loads each file with `joblib.load`, and processes all days (sessions) and frames within each animal sequentially.

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

iii. The AI identified from the reference code (`load_dat` in utils.py) that data is stored as joblib files named by animal ID. The CONVERSION_NOTES confirm: "Each animal stored as joblib file: `data/{animal_id}` -> dict with animal_id key."

## 1-b. How are the data split into subjects (mice)?

i. Subjects are defined by the hardcoded `ANIMALS` list of 7 mouse IDs. Each animal is processed independently in a loop. A `subject_idx` array maps each session to its animal via the index in the ANIMALS list.

ii.
```python
for animal in animals:
    neural, inp, out, sess_info = process_animal(animal, show_processing=show_processing)
    subject_id = ANIMALS.index(animal)
    ...
    subject_idx_list.append(subject_id)
```

iii. The AI noted from the paper: "Naive male (4) and female (3) mice" totaling 7 subjects with 5,413 unique neurons across 207 sessions. The subject list matches the data files found in the `data/` directory.

## 1-c. How are the data split into sessions?

i. Each day within an animal constitutes one session. The `trace` array has shape `(n_days, n_cells, n_frames)`, so the first dimension indexes days/sessions. All days for all animals are included (no session filtering), yielding 207 total sessions.

ii.
```python
n_days = d['trace'].shape[0]
for day in range(n_days):
    env_name = envs[day]
    trace = d['trace'][day]
    position = d['position'][day]
    ...
```

iii. The AI verified: "Sessions total: 207" matching the paper's "207 sessions". Sessions per subject are 31, 31, 31, 21, 31, 31, 31 (QLAK-CA1-51 has only 21).

## 1-d. How are the data split into trials?

i. Each ~40-minute session is split into 1-minute (1800-frame) contiguous, non-overlapping segments. Trial boundaries start at frame indices 0, 1800, 3600, etc. The last trial in a session may be shorter if the session length is not an exact multiple of 1800 frames.

ii.
```python
TRIAL_DURATION_S = 60  # 1-minute trials
FRAMES_PER_TRIAL = FPS * TRIAL_DURATION_S  # 1800 frames
trial_starts = list(range(0, n_frames, FRAMES_PER_TRIAL))
for start in trial_starts:
    end = min(start + FRAMES_PER_TRIAL, n_frames)
    trial_len = end - start
```

iii. The instructions specify: "The experiment consists of long recording sessions, which will be split into 1-minute trials within each session." The AI followed this directly. All sessions yield 40 trials, with some sessions having a shorter last trial (min T=1666 frames).

## 1-e. How are trials filtered based on quality controls?

i. Two quality filters are applied: (1) Partial trials shorter than 30 seconds (900 frames) are dropped. (2) Sessions with fewer than 2 valid trials are skipped entirely. No other trial-level quality filtering is performed.

ii.
```python
MIN_TRIAL_FRAMES = FPS * 30  # Minimum 30s for a partial trial at end
if trial_len < MIN_TRIAL_FRAMES:
    continue
...
if n_trials < 2:
    print(f"  Day {day} ({env_name}): Only {n_trials} trial(s), skipping (need >=2)")
    continue
```

iii. The AI's CONVERSION_NOTES state: "Partial trials <900 frames (30s) are dropped" and "All sessions produce >=2 trials (required for decoder evaluation)." The 30s threshold is an arbitrary choice by the AI; the instructions don't specify a minimum trial length. The >=2 trial requirement comes from the instruction: "There needs to be at least two trials within each session."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from the `trace` variable in the raw data, which has shape `(n_days, n_cells, n_frames)`. This contains binarized calcium events (0/1 values) representing significant calcium transients detected via rising-phase extraction with z-score > 2.5 on the derivative.

ii.
```python
trace = d['trace'][day]  # (n_cells, n_frames) for one day
```

iii. The AI documented: "Data is already preprocessed: binary calcium trace (0/1 for significant events from rising-phase extraction)" and "No delta F/F computation needed - trace is already binarized."

## 2-b. How is the `neural` data processed?

i. Minimal processing is applied. The binary trace for registered cells is extracted for each trial segment, converted to float32, and stored directly. No smoothing, rate map computation, normalization, or delta F/F computation is performed. The data is used as-is from the preprocessed files.

ii.
```python
trace_registered = trace[registered_mask]  # (n_registered, n_frames)
neural_trial = trace_registered[:, start:end].astype(np.float32)
```

iii. The AI justified this by noting: "Neural data: already binarized rising-phase calcium events" and "Binary trace: Used as-is (0/1), 'treated as firing rate'." The reference code treats the binary trace directly in its analyses.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The only filtering is based on cell registration: cells that are not registered (have NaN values) on a given day are excluded. This is checked by testing if the first frame's value is NaN. No place-cell filtering, activity threshold, or velocity filtering is applied.

ii.
```python
registered_mask = ~np.isnan(trace[:, 0])
n_registered = np.sum(registered_mask)
if n_registered == 0:
    continue
trace_registered = trace[registered_mask]
```

iii. The AI noted from the paper: "motivated the inclusion of all cells in subsequent analyses" and decided: "Use ALL registered cells on each day (non-NaN trace). No place-cell filtering." The reference code's `decode_position_within` function uses a `cell_threshold > 5` (cells must have >5 events when animal is moving), but the AI did not implement this filter.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the start of each 1-minute trial segment. The first trial starts at frame 0 of the session, subsequent trials start at multiples of 1800 frames. There is no alignment to a behavioral event (e.g., stimulus onset) since the recording is continuous free exploration.

ii.
```python
trial_starts = list(range(0, n_frames, FRAMES_PER_TRIAL))
for start in trial_starts:
    end = min(start + FRAMES_PER_TRIAL, n_frames)
    neural_trial = trace_registered[:, start:end].astype(np.float32)
```

iii. The AI set `temporal_alignment_event` to "Start of each 1-minute trial segment within a 40-minute recording session" and `off_start=0.0`, `off_end=60.0`. This is appropriate since the task is continuous free foraging with no discrete trial events.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is the native recording frame rate of 30 Hz, corresponding to ~33.33 ms per time bin. No temporal rebinning is applied.

ii.
```python
FPS = 30  # Recording frame rate (Hz)
# In metadata:
'time_bin_size': 1000.0 / FPS,  # ~33.33 ms
```

iii. The AI used the native sampling rate directly: "Time bin: 30Hz native sampling (33.33ms)." The paper confirms: "acquired at 30 Hz."

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The environment geometry input is derived from the `envs` variable in the raw data, which stores the environment name (string) for each day/session (e.g., 'square', 'o', 't', 'u', etc.).

ii.
```python
envs = d['envs'].squeeze()
env_name = envs[day]
env_mat = get_env_mat(env_name).flatten()  # (9,)
```

iii. The AI identified `envs` from the data structure: "envs: shape (n_days, 1) - environment name strings." The `get_env_mat` function was copied from the reference code (utils.py).

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The environment name string is mapped to a 3x3 binary matrix via the `get_env_mat` function (from the reference code), where 1 indicates accessible regions and 0 indicates blocked regions. The 3x3 matrix is then flattened to a 9-element vector and cast to float32. This input is static (same for all time points within a trial).

ii.
```python
def get_env_mat(env):
    if env == 'square':
        return np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]).astype(float)
    elif env == 'o':
        return np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]]).astype(float)
    # ... (10 environments total)

env_mat = get_env_mat(env_name).flatten()
input_trial = env_mat.astype(np.float32)
```

iii. The AI documented: "get_env_mat: Get binary 3x3 matrix for environment geometry. From reference code." The function was directly adapted from the reference code's `utils.py`.

## 3-c. How is the `input` *Environment geometry* aligned with the neural data?

i. The environment geometry is static per trial (shape `(9,)` rather than `(9, n_timepoints)`), so no temporal alignment is needed. The same geometry vector applies to all time points within a trial. The correct environment is selected based on the session's day index.

ii.
```python
input_trial = env_mat.astype(np.float32)  # shape: (9,)
```

iii. The instructions specify input can be "(n_input, n_timepoints) or (n_input)" for static per-trial inputs. The AI correctly uses the static form since environment geometry doesn't change within a trial.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The mouse position output is derived from the `position` variable in the raw data, which has shape `(n_days, 2, n_frames)` containing x,y coordinates in centimeters (0-75 range) tracked via DeepLabCut.

ii.
```python
position = d['position'][day]  # (2, n_frames)
pos_bins = bin_position_3x3(position)
```

iii. The AI documented: "position: shape (n_days, 2, n_frames) - x,y position in cm (0-75 range)" and "Position tracked with DeepLabCut."

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The continuous x,y position is discretized into a 3x3 spatial grid. Each spatial dimension (0-75 cm) is divided into 3 equal bins of ~25 cm each. The x and y coordinates are independently binned using floor division, then combined into a single bin index using row-major ordering.

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

iii. The instructions specify: "Mouse position discretized into 3 x 3 = 9 spatial bins." The AI implemented this with a small buffer (1e-5) to handle edge cases where position equals exactly 75 cm.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. The continuous position is thresholded by dividing each dimension (0-75 cm) into 3 equal bins with boundaries at 0, 25, 50, 75 cm (approximately). Floor division is used, so bin 0 = [0, 25), bin 1 = [25, 50), bin 2 = [50, 75]. Values are clipped to [0, 2] range. The 2D bins are combined into 9 categories (0-8) via `x_bin * 3 + y_bin`.

ii.
```python
bin_size = (env_size + buffer) / N_POS_BINS  # ~25.000003 cm
x_bin = np.clip(np.floor(position[0] / bin_size).astype(int), 0, N_POS_BINS - 1)
y_bin = np.clip(np.floor(position[1] / bin_size).astype(int), 0, N_POS_BINS - 1)
bin_idx = x_bin * N_POS_BINS + y_bin
output_trial = pos_bins[start:end].astype(np.int64).reshape(1, -1)
```

iii. The AI's output values are labeled `["x0y0", "x0y1", "x0y2", "x1y0", "x1y1", "x1y2", "x2y0", "x2y1", "x2y2"]`, corresponding to the 9 bins. The output distribution shows uneven coverage (bin 8/x2y2 at 20.1%, bin 4/x1y1 at 5.7%).

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. The position is recorded at the same 30 Hz frame rate as the neural data, so they are inherently aligned frame-by-frame. The same frame indices used to slice the neural data are used to slice the position data for each trial.

ii.
```python
pos_bins = bin_position_3x3(position)  # (n_frames,) - full session
# Same start:end slice for both:
neural_trial = trace_registered[:, start:end]
output_trial = pos_bins[start:end].astype(np.int64).reshape(1, -1)
```

iii. The AI confirmed in CONVERSION_NOTES: "Position bins: Same session/trial -> `np.allclose` returns True (exact match)." Neural and position share the same frame indices, ensuring perfect temporal alignment.

## 5-a. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is 33.33 ms (1/30 Hz), the native recording frame rate. No temporal rebinning is applied; each frame in the original data maps to one time bin in the converted data.

ii.
```python
FPS = 30
'time_bin_size': 1000.0 / FPS,  # ~33.33 ms
```

iii. Same as 2-e. The AI preserved the native temporal resolution.

## 5-b. How are the neural, input, and output data temporally aligned?

i. Neural and output (position) data are inherently aligned because they share the same frame rate (30 Hz) and frame indices. Both are sliced using the same `start:end` indices for each trial. The input (environment geometry) is static per trial and does not require temporal alignment.

ii.
```python
for start in trial_starts:
    end = min(start + FRAMES_PER_TRIAL, n_frames)
    neural_trial = trace_registered[:, start:end].astype(np.float32)
    input_trial = env_mat.astype(np.float32)  # static
    output_trial = pos_bins[start:end].astype(np.int64).reshape(1, -1)
```

iii. The AI's sanity checks verified alignment: "Neural data: QLAK-CA1-50, day 0, trial 5 -> np.allclose returns True" and "Position bins: Same session/trial -> np.allclose returns True."

## 6. How are minor issues in the data (e.g., missing data, malformed entries) handled?

i. Missing data is handled in two ways: (1) NaN values in the trace matrix indicate unregistered cells on a given day; these cells are excluded from that session. (2) Sessions with zero registered cells are skipped entirely. (3) Short partial trials at the end of sessions (< 30s) are dropped. The code does not handle other potential issues like NaN position values or anomalous trace values.

ii.
```python
registered_mask = ~np.isnan(trace[:, 0])
if n_registered == 0:
    continue
if trial_len < MIN_TRIAL_FRAMES:
    continue
```

iii. The AI documented: "Last trial has fewer frames when session length isn't divisible by 1800 (min T=1666)" and "QLAK-CA1-51 has only 21 sessions (2 sequences) - correctly handled."

## 7-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading the animal data files from disk using `joblib.load`. Each animal file contains large arrays (n_days x n_cells x n_frames), and the 7 files total approximately 19 GB when loaded. The full conversion takes ~160 seconds, with individual animals taking 11-29 seconds depending on the number of cells.

ii.
```python
dat = joblib.load(filepath)  # Loading large joblib files
```

iii. The AI's timing shows: per-animal processing ranges from 11.0s (QLAK-CA1-51, 21 days, 554 cells) to 29.2s (QLAK-CA1-50, 31 days, 942 cells). The conversion output also notes "Saved 19255.4 MB in 20.1s" for the pickle write, which is also substantial.

## 7-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial-splitting loop iterates over trial start positions within each session, performing array slicing. This could potentially be vectorized using `np.reshape` to split the entire session trace into trial-sized chunks at once (for full-length trials). However, the current implementation is already efficient since array slicing in NumPy is O(1) (creates views).

ii.
```python
for start in trial_starts:
    end = min(start + FRAMES_PER_TRIAL, n_frames)
    neural_trial = trace_registered[:, start:end].astype(np.float32)
    output_trial = pos_bins[start:end].astype(np.int64).reshape(1, -1)
```

iii. The AI noted in CONVERSION_NOTES: "Efficient vectorized operations (no inner loops for trace/position)." The position binning is already fully vectorized. The remaining loops (over animals, days, trials) are structural and not easily vectorizable.

## 7-c. What processing does the code repeat multiple times?

i. The `get_env_mat` function is called once per session, but returns the same result for sessions sharing the same environment. Since each environment appears multiple times across sessions (each of 10 environments appears ~20 times across 207 sessions), this computation is repeated unnecessarily. However, this is a trivial computation (creating a small array).

ii.
```python
env_mat = get_env_mat(env_name).flatten()  # Called every session
```

iii. No explicit justification for this repetition was given. The overhead is negligible since the function just creates a 3x3 array.

## 7-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code loads the entire animal data dictionary (including `maps`, `SFPs`, `centroids`, `blocked`) but only uses `trace`, `position`, and `envs`. The unused fields consume memory unnecessarily. The `del d` statement at the end of `process_animal` frees this memory, but it was held during the entire processing of that animal.

ii.
```python
d = load_animal_data(animal)  # Loads ALL fields
# Only uses: d['trace'], d['position'], d['envs']
del d  # Free memory after processing
```

iii. The AI did not document this inefficiency. The `load_animal_data` function loads the complete dictionary from the joblib file.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Same as question 6. NaN values in the trace matrix are used to identify and exclude unregistered cells. The last trial in each session may be shorter than 60 seconds due to the session not being an exact multiple of 1800 frames; these partial trials are included as long as they are at least 30 seconds (900 frames). No explicit handling for other data anomalies (e.g., NaN positions, out-of-range values).

ii.
```python
registered_mask = ~np.isnan(trace[:, 0])
if trial_len < MIN_TRIAL_FRAMES:
    continue
```

iii. The AI's CONVERSION_NOTES document edge cases: "Last trial has fewer frames when session length isn't divisible by 1800 (min T=1666)" and checks that "All sessions produce >=2 trials."

## 9-a. What are the most time-consuming steps of the code?

i. Same as 7-a. The dominant costs are: (1) Loading animal data from joblib files (~60-70% of time based on per-animal timings), (2) Writing the 19.3 GB pickle output file (20.1s), (3) Array slicing and type conversion within the trial loop.

ii.
```python
dat = joblib.load(filepath)  # ~15-25s per animal for large animals
pickle.dump(data, f, protocol=4)  # 20.1s for 19.3 GB
```

iii. The AI estimated "Full run estimated: ~160s" and achieved 159.8s actual, well under the 15-minute target. No further optimization was pursued.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Same as 7-b. The main loop structure (animals -> days -> trials) is inherently sequential for memory reasons (loading one animal at a time). The trial-splitting loop could theoretically be replaced with array reshaping for full-length trials, but the current slicing approach is already efficient.

ii. See 7-b code snippets.

iii. The AI's code is already largely vectorized for the core computations (position binning, cell masking). The remaining loops handle structural iteration that cannot be easily vectorized.

## 9-c. What processing does the code repeat multiple times?

i. Same as 7-c. The `get_env_mat` function is called redundantly for repeated environments. Additionally, the `bin_position_3x3` function is called once per session, computing bins for the entire session even though bins for the last partial trial may be discarded.

ii.
```python
pos_bins = bin_position_3x3(position)  # Full session binning
# Some bins at the end may be unused if last trial is dropped
```

iii. These redundancies have negligible performance impact relative to the I/O-bound data loading.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Same as 7-d. The code loads unused data fields (`maps`, `SFPs`, `centroids`, `blocked`) from the joblib files. Position bins are computed for the entire session including any frames that fall in dropped partial trials. The code also computes and prints output distribution statistics that are informational only.

ii.
```python
d = load_animal_data(animal)  # Loads maps, SFPs, centroids, blocked (unused)
pos_bins = bin_position_3x3(position)  # Bins frames that may be in dropped trials
```

iii. The AI did not explicitly discuss these inefficiencies but noted "del d  # Free memory" to mitigate the memory impact.
