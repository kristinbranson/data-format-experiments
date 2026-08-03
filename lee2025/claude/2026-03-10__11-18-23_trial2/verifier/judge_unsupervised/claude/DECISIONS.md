# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads each animal's data from joblib files in the `data/` directory using `joblib.load()`. It iterates over a hardcoded list of 7 animal names (`ANIMALS`), loading one file per animal. Each file contains a dictionary keyed by animal name, with fields `trace`, `position`, `envs`, `blocked`, etc. The AI accesses `dat[animal]` to get the animal's data dictionary. This matches the reference code's `load_dat` function which also uses `joblib.load`.

ii.
```python
ANIMALS = ["QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
           "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"]
# ...
dat = joblib.load(os.path.join(data_dir, animal))
animal_data = dat[animal]
```

iii. The AI identified that the reference code's `load_dat` function uses `joblib.load` for the joblib format. The AI's CONVERSION_NOTES.md states: "Each animal has a joblib file in `data/` containing a dict keyed by animal name." The agent verified the data structure by exploring the files and matching against the reference code.

## 1-b. How are the data split into subjects?

i. Each animal (mouse) is treated as a separate subject. The AI hardcodes the list of 7 animal names in the `ANIMALS` constant. A `subject_idx` array maps each session to its subject index. The `subjects` list in the output is the list of animal names.

ii.
```python
ANIMALS = ["QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
           "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"]
# ...
subjects = list(animals_to_process)
# ...
subject_idx_list.append(a_idx)
```

iii. The AI noted from the paper: "7 mice, mean number of cells per animal = 773 +/- 68 SE, min=515." The animal list matches the data files in the `data/` directory. The agent verified 515+875+942+554+862+713+952 = 5,413 unique neurons matching the paper.

## 1-c. How are the data split into sessions?

i. Each recording day for each animal is treated as a separate session. The AI iterates over `range(n_days)` for each animal, where `n_days` comes from `animal_data['trace'].shape[0]`. This yields 207 total sessions (31 days each for 6 animals + 21 days for QLAK-CA1-51).

ii.
```python
n_days = animal_data['trace'].shape[0]
for day in range(n_days):
    # ... process each day as a session
    neural_trials, input_trials, output_trials, n_valid = process_session(
        animal_data, day, ...)
```

iii. The AI verified: "Total sessions = 207, matching 31*6 + 21 = 207 from the paper." The CONVERSION_NOTES document: "Session = day: Each day of recording is one session."

## 1-d. How are the data split into trials?

i. Each ~40-minute session is split into 1-minute trials. At 30 Hz with 3-frame temporal bins, each trial contains 600 time bins (1800 frames / 3 = 600 bins). The number of complete trials per session is `n_total_bins // BINS_PER_TRIAL`. Sessions with ~71,866 frames yield 39 trials; sessions with ~72,219 frames yield 40 trials. Remainder frames at the end of sessions are discarded.

ii.
```python
TRIAL_DURATION_SEC = 60  # 1 minute trials
FRAMES_PER_TRIAL = TRIAL_DURATION_SEC * FPS  # 1800 frames per trial
BINS_PER_TRIAL = FRAMES_PER_TRIAL // TEMPORAL_BIN_SIZE  # 600 time bins per trial
# ...
n_trials = n_total_bins // BINS_PER_TRIAL
for t in range(n_trials):
    start = t * BINS_PER_TRIAL
    end = (t + 1) * BINS_PER_TRIAL
    trial_neural = binned_trace[:, start:end]
```

iii. The task instructions specify "1-minute trials within each session." The AI confirmed: "~40 trials per session (40 min / 1 min)" and verified 39-40 trials per session in the output.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies minimal trial-level filtering: sessions with fewer than 2 trials are skipped (to allow decoder train/test splitting). No individual trial quality filtering is applied. No velocity-based filtering of timepoints within trials is applied. The reference code does not have trial-level quality controls either — the reference's `decode_position_within` filters individual timepoints by velocity, not entire trials.

ii.
```python
if len(neural_trials) < 2:
    print(f"  Day {day}: skipped (< 2 trials)")
    continue
```

iii. The AI's CONVERSION_NOTES state: "No explicit trial filtering in the reference (entire sessions used)." The minimum-2-trials check ensures the decoder can evaluate performance. In practice, all sessions produce 39-40 trials, so this filter never triggers.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from the `trace` field in each animal's data dictionary. This contains binary calcium event data (0/1 values) representing the rising phase of calcium transients, already preprocessed by z-scoring > 2.5 of the derivative of the calcium signal. Shape: `(n_days, n_cells, n_frames)`.

ii.
```python
trace = animal_data['trace'][day_idx]  # (n_cells, n_frames)
```

iii. The AI documented: "Neural data: `trace` is already binarized (0/1) - rising phase of calcium transients, z-scored > 2.5" and "No dF/F needed - data is already preprocessed binary events." This matches the reference code which uses `trace` directly.

## 2-b. How is the `neural` data processed?

i. The neural trace data undergoes two processing steps matching the reference `fit_decoder` function:
1. Gaussian smoothing along the time axis with `sigma=3` frames using `scipy.ndimage.gaussian_filter1d`
2. Average pooling with kernel size 3 and stride 3 (non-overlapping), implemented as reshape + mean

The data is converted from binary (0/1) to float64 for smoothing, then to float32 for output. This produces temporally binned firing rate estimates at 100ms resolution.

ii.
```python
def temporal_bin_trace(trace_2d, sigma=TEMPORAL_BIN_SIZE, bin_size=TEMPORAL_BIN_SIZE):
    smoothed = gaussian_filter1d(trace_2d.astype(np.float64), sigma=sigma, axis=1)
    n_cells, n_frames = smoothed.shape
    n_bins = n_frames // bin_size
    trimmed = smoothed[:, :n_bins * bin_size]
    binned = trimmed.reshape(n_cells, n_bins, bin_size).mean(axis=2)
    return binned.astype(np.float32)
```

iii. The AI documented: "Gaussian smoothing of trace: Reference `fit_decoder` applies `gaussian_filter1d(traces, sigma=temporal_bin_size, axis=0)` before binning. We should do the same." The reference code applies `gaussian_filter1d(traces, sigma=temporal_bin_size, axis=0)` then `AvgPool1d(kernel_size=3, stride=3)`. The AI's implementation matches this exactly (with transposed axes since AI uses cells x time rather than time x cells).

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI filters cells by checking for NaN values: cells with NaN at the first timepoint are excluded (these are cells not registered on that particular day). No further quality filtering is applied. Specifically, the reference code's velocity-based cell filtering (cells with >5 events during moving periods) is NOT applied. The AI argues this is "decoder-specific" preprocessing.

ii.
```python
valid_mask = ~np.isnan(trace[:, 0])
valid_trace = trace[valid_mask]  # (n_valid, n_frames)
n_valid = valid_mask.sum()
```

iii. The AI's CONVERSION_NOTES explain: "Cell filtering: Include only registered cells per session (not NaN). No velocity filtering (that's decoder-specific)." And: "For decoding: cells with >5 events during moving periods included (per-session)" — acknowledged but not applied. The reference `decode_position_within` applies `cell_idx[d] = np.sum(traces[:, :, d][vel_idx[d]], axis=0) > cell_threshold` (cell_threshold=5), but the AI chose to defer this to downstream processing.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to the start of the recording session. The first trial starts at frame 0, the second at frame 1800 (60s * 30Hz), etc. There is no event-based alignment (no stimulus onset, no movement onset). The metadata records `temporal_alignment_event: 'Start of recording session'` and `off_start: 0.0`.

ii.
```python
for t in range(n_trials):
    start = t * BINS_PER_TRIAL  # 0, 600, 1200, ...
    end = (t + 1) * BINS_PER_TRIAL
    trial_neural = binned_trace[:, start:end]
```

iii. The AI documented: "Temporal alignment: Trials start at beginning of session recording. Reference processes full session. CONSISTENT." Since the reference processes entire sessions without trial segmentation, and the task requires 1-minute trials, aligning to session start is the natural choice.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is 100 ms (3 frames at 30 Hz). Temporal rebinning IS applied: the raw 30 Hz data (33.3 ms per frame) is rebinned into 100 ms bins through Gaussian smoothing followed by average pooling with a factor of 3. This matches the reference `fit_decoder`'s `temporal_bin_size=3` parameter.

ii.
```python
FPS = 30  # recording frame rate (Hz)
TEMPORAL_BIN_SIZE = 3  # frames per time bin (from reference fit_decoder)
TIME_BIN_MS = (TEMPORAL_BIN_SIZE / FPS) * 1000  # 100 ms
```

iii. The AI's CONVERSION_NOTES note: "Decoder temporal bin: 3 frames (100ms) — From code: temporal_bin_size=3." This directly matches the reference `fit_decoder` function's default parameter.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The environment geometry input is derived from the `envs` field in each animal's data dictionary. This field contains environment name strings (e.g., 'square', 'o', 't', 'u', etc.) for each recording day. The name is then mapped to a 3x3 binary matrix using the `get_env_mat` lookup.

ii.
```python
env_name = str(animal_data['envs'][day_idx].item() if hasattr(animal_data['envs'][day_idx], 'item')
               else animal_data['envs'][day_idx])
# ...
env_input = get_env_input(env_name)  # (9,)
```

iii. The AI identified the `envs` field during data exploration and matched it to the reference code's `get_env_mat` function. The CONVERSION_NOTES document: "`blocked` field indicates which partitions of 3x3 grid are blocked. Layout: [[0,1,2],[3,4,5],[6,7,8]]. -1 means no blocks (square)."

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The environment name string is looked up in a dictionary (`ENV_MATRICES`) that maps each of the 10 environment names to a 3x3 binary numpy array (1=accessible, 0=blocked). The matrix is then flattened to a 9-element vector of float32 values. This input is static per trial (same for all timepoints within a trial). The matrices are hardcoded to match the reference `get_env_mat` function exactly.

ii.
```python
ENV_MATRICES = {
    'square':    np.array([[1,1,1],[1,1,1],[1,1,1]]),
    'o':         np.array([[1,1,1],[1,0,1],[1,1,1]]),
    # ... (all 10 environments)
}

def get_env_input(env_name):
    mat = ENV_MATRICES.get(env_name)
    return mat.flatten().astype(np.float32)
```

iii. The AI copied the environment matrices directly from the reference code's `get_env_mat` function. The CONVERSION_NOTES confirm: "Input construction: `get_env_mat` 3x3 binary matrix from env name. Directly from reference code. MATCH."

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The mouse position output is derived from the `position` field in each animal's data dictionary. This contains x,y coordinates tracked by DeepLabCut head tracking, with shape `(2, n_frames)` representing (x, y) position in centimeters. The position range is [0, 75] cm for the 75x75 cm arena.

ii.
```python
position = animal_data['position'][day_idx]  # (2, n_frames)
```

iii. The AI documented: "Position tracking: DeepLabCut head tracking" and "Arena is 75x75 cm." This matches the paper's description of the arena size and tracking method.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The position data undergoes two processing steps:
1. Temporal binning: raw position (2, n_frames) is averaged over non-overlapping 3-frame bins, yielding (2, n_bins) values in cm. No Gaussian smoothing is applied (matching the reference which only smooths traces, not behavior).
2. Spatial discretization: the temporally binned x,y positions are discretized into a 3x3 grid (25 cm per bin) using `floor(pos / 25)`, clamped to [0, 2], then combined into a single index `x_bin * 3 + y_bin` (values 0-8).

ii.
```python
def bin_position(position_2d, bin_size=TEMPORAL_BIN_SIZE):
    n_frames = position_2d.shape[1]
    n_bins = n_frames // bin_size
    trimmed = position_2d[:, :n_bins * bin_size]
    binned = trimmed.reshape(2, n_bins, bin_size).mean(axis=2)
    return binned

def discretize_position(position_2d, bin_size=SPATIAL_BIN_SIZE, n_bins=N_SPATIAL_BINS):
    x_bin = np.clip(np.floor(position_2d[0] / bin_size).astype(int), 0, n_bins - 1)
    y_bin = np.clip(np.floor(position_2d[1] / bin_size).astype(int), 0, n_bins - 1)
    return x_bin * n_bins + y_bin
```

iii. The AI noted that the reference `fit_decoder` applies `pooling(behav.T).astype(int).T` for behavior — simple average pooling without Gaussian smoothing. The AI's approach (average then discretize) matches this pipeline, adapted from 15x15 to 3x3 bins as required by the task.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Position is discretized into 9 categories (3x3 grid). Each x and y coordinate is divided by the spatial bin size (25 cm), floored, and clamped to [0, 2]. The combined category is `x_bin * 3 + y_bin`, yielding values 0-8. Output values are labeled as "row{i}_col{j}" for i,j in [0,1,2]. This produces row-major indexing consistent with the reference code's `empty_map.flatten()` (C-order).

ii.
```python
ARENA_SIZE_CM = 75.0
N_SPATIAL_BINS = 3
SPATIAL_BIN_SIZE = ARENA_SIZE_CM / N_SPATIAL_BINS  # 25 cm per bin

x_bin = np.clip(np.floor(position_2d[0] / bin_size).astype(int), 0, n_bins - 1)
y_bin = np.clip(np.floor(position_2d[1] / bin_size).astype(int), 0, n_bins - 1)
return x_bin * n_bins + y_bin
```

iii. The task specifies "Mouse position discretized into 3 x 3 = 9 spatial bins." The AI correctly uses 25 cm bins for the 75 cm arena, with floor + clamp ensuring values at the boundary (75 cm) map to bin 2 rather than going out of range.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Both neural and position data use the same temporal binning scheme: non-overlapping 3-frame bins starting from frame 0 of the session. Both are then split into 1-minute trials (600 bins each) using the same indices. This ensures perfect temporal alignment — time bin `t` in the neural data corresponds to the same 3-frame window as time bin `t` in the position data.

ii.
```python
# Same binning for both:
binned_trace = temporal_bin_trace(valid_trace)  # (n_valid, n_total_bins)
binned_pos = bin_position(position)  # (2, n_total_bins)
pos_bins = discretize_position(binned_pos)  # (n_total_bins,)
# Same trial splitting:
trial_neural = binned_trace[:, start:end]
trial_output = pos_bins[start:end].reshape(1, -1)
```

iii. The AI verified alignment through processing plots (--show-processing mode) and spot-checked specific timepoints against raw data to confirm neural and position data are synchronized.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several types of missing/edge-case data:
1. **Unregistered cells**: Cells not registered on a given day have NaN values in the trace. These are excluded per session via `~np.isnan(trace[:, 0])`.
2. **Incomplete trials**: Frames remaining after the last complete 1-minute trial are discarded (e.g., for 71,866 frames: 71,866/3 = 23,955 bins, 23,955 mod 600 = 555 bins = ~55 seconds discarded).
3. **Empty sessions**: Sessions with 0 valid cells return empty lists.
4. **Minimum trial count**: Sessions with fewer than 2 trials are skipped.
5. **Environment name extraction**: Handles various numpy array types via `.item()` and `.squeeze()` calls.

ii.
```python
valid_mask = ~np.isnan(trace[:, 0])
valid_trace = trace[valid_mask]
if n_valid == 0:
    return [], [], [], 0
# ...
if len(neural_trials) < 2:
    print(f"  Day {day}: skipped (< 2 trials)")
    continue
# ...
env_name = str(animal_data['envs'][day_idx].item() if hasattr(animal_data['envs'][day_idx], 'item')
               else animal_data['envs'][day_idx])
```

iii. The AI's CONVERSION_NOTES document edge case handling: "Sessions with different frame counts handled: 71866/3=23955 bins → 39 trials; 72219/3=24073 bins → 40 trials. Correct." And: "Position at exactly 75cm → clipped to bin 2. Correct."

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are:
1. **Loading joblib files**: Each animal's file is large (contains trace, position, maps, SFPs, centroids for all days and cells).
2. **Gaussian smoothing**: `gaussian_filter1d` on the full trace array (n_cells x n_frames) for each session.
3. **Sequential animal processing**: Animals are processed one at a time in a loop, with data loaded and freed per animal.

ii.
```python
dat = joblib.load(os.path.join(data_dir, animal))  # Large file load
# ...
smoothed = gaussian_filter1d(trace_2d.astype(np.float64), sigma=sigma, axis=1)  # Full trace smoothing
```

iii. The AI reported: "Per animal (avg) ~37s" and "Full (7 animals) ~4.3 min" — well under the 15-minute limit. No further optimization was needed. The `del dat` at the end of each animal frees memory.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The main loop that could be vectorized is the trial splitting loop. Instead of iterating over trials with a Python for-loop, the trace and position data could be reshaped directly into (n_trials, n_neurons/dims, BINS_PER_TRIAL) using a single reshape operation. The output value label generation loop (lines 305-308) could also be replaced with a list comprehension or itertools.product.

ii.
```python
# Current loop-based approach (could be vectorized):
for t in range(n_trials):
    start = t * BINS_PER_TRIAL
    end = (t + 1) * BINS_PER_TRIAL
    trial_neural = binned_trace[:, start:end]
    trial_output = pos_bins[start:end].reshape(1, -1).astype(np.int64)
    neural_trials.append(trial_neural)
    input_trials.append(env_input)
    output_trials.append(trial_output)
```

iii. The AI acknowledged that vectorization could improve speed but found the total runtime acceptable (~4.3 min) without it. The trial count per session (39-40) makes this loop relatively lightweight.

## 6-c. What processing does the code repeat multiple times?

i. The environment name is extracted twice: once in `convert_dataset` (line 265: `env_name = str(animal_data['envs'][day].squeeze())`) for logging, and again inside `process_session` (line 130: `env_name = str(animal_data['envs'][day_idx].item() ...)`). The environment matrix lookup also happens for every trial even though it's the same for all trials in a session (though this is trivial cost).

ii.
```python
# In convert_dataset:
env_name = str(animal_data['envs'][day].squeeze())  # First extraction

# In process_session:
env_name = str(animal_data['envs'][day_idx].item() if hasattr(animal_data['envs'][day_idx], 'item')
               else animal_data['envs'][day_idx])  # Second extraction
```

iii. This duplication is minor and doesn't significantly impact performance. The environment name is a simple string extraction. The AI's code is structured for clarity with `process_session` being self-contained.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several processing steps produce results that are partially or fully discarded:
1. **Remainder frames**: The Gaussian smoothing and temporal binning process the full session (~71,866-72,219 frames), but only complete 1-minute trials are kept. The last ~55-73 seconds of each session (555-73 bins) are smoothed and binned but then discarded.
2. **Loading excess data**: The entire animal data dictionary is loaded (including `maps`, `SFPs`, `centroids`, `blocked`) but only `trace`, `position`, and `envs` are used. The spatial footprints and rate maps are never accessed.
3. **Position data for discarded frames**: Position is binned and discretized for the full session, but only complete trial segments are retained.

ii.
```python
# Full session is loaded and processed, but only complete trials are kept:
dat = joblib.load(os.path.join(data_dir, animal))  # Loads maps, SFPs, etc.
# ...
n_bins = n_frames // bin_size  # Trims to complete bins
# ...
n_trials = n_total_bins // BINS_PER_TRIAL  # Only complete trials kept
```

iii. The AI did not explicitly discuss optimizing the data loading to skip unused fields. Joblib loads the entire dictionary including rate maps, spatial footprints, and centroids — data that could be significant in size but is never used for conversion. This is a trade-off of using the simple `joblib.load` interface.
