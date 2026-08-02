# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads the `bwm_release.csv` file to identify all sessions (459 sessions, 699 probe insertions, 139 subjects). It groups by `eid` to get unique sessions and resolves session paths from the local ONE cache using the pattern `lab/Subjects/subject/date/session_number`. For each session, it loads trials from `_ibl_trials.table.pqt`, spike data from `spikes.times.npy`/`spikes.clusters.npy`, wheel data from `_ibl_wheel.timestamps.npy`/`_ibl_wheel.position.npy`, and whisker motion energy from camera files. Sessions that are missing required files are skipped.

ii.
```python
RELEASE_CSV = ROOT / "code" / "code_zhang2025" / "data" / "bwm_release.csv"
DATA_ROOT = ROOT / "data" / "one_cache"

release = pd.read_csv(RELEASE_CSV, index_col=0)
sessions = (
    release.groupby("eid", sort=False)
    .agg({
        "subject": "first", "date": "first",
        "session_number": "first", "lab": "first",
        "probe_name": lambda x: tuple(sorted(x)),
    })
    .reset_index()
)
```

iii. The AI states it used the full provided `bwm_release.csv` release table, matching the reference code's approach in `0_data_caching.py`. Session paths are resolved from the local ONE cache since the ONE API is not available.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified from the `subject` column of `bwm_release.csv`, grouped per session. A unique subject list is built in encounter order as sessions are processed. Each session is assigned a `subject_idx` mapping into this list.

ii.
```python
subjects = []
subject_to_idx = {}
for rec in records:
    if rec["subject"] not in subject_to_idx:
        subject_to_idx[rec["subject"]] = len(subjects)
        subjects.append(rec["subject"])
```

iii. The AI followed the standard approach of collecting unique subjects from successfully processed sessions.

## 1-c. How are the data split into sessions?

i. Each unique `eid` in `bwm_release.csv` represents one session. The AI groups probe insertions by `eid` and processes each session independently, merging probes from the same session into a single neural population.

ii.
```python
sessions = (
    release.groupby("eid", sort=False)
    .agg({...
        "probe_name": lambda x: tuple(sorted(x)),
    })
    .reset_index()
)
for idx, row in enumerate(sessions.to_dict("records"), start=1):
    session_path = find_session_path(row)
    ...
```

iii. This matches the reference code's approach of iterating over eids and merging probes per session.

## 1-d. How are the data split into trials?

i. Trials are loaded from `_ibl_trials.table.pqt` for each session. Each trial is defined by the trial events in this table. Neural and behavioral data are segmented into per-trial arrays using time intervals defined by `stimOn_times + [-0.5, 1.5]`.

ii.
```python
trials = load_trials_table(session_path)
# Returns pd.read_parquet(pick_latest(session_path, "alf/**/_ibl_trials.table.pqt"))

stim_on = trials[ALIGN_EVENT].to_numpy(dtype=np.float32)
intervals = np.c_[stim_on + WINDOW[0], stim_on + WINDOW[1]].astype(np.float32)
```

iii. The AI used the standard IBL trials table, consistent with the reference code.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies a trial mask matching the reference `load_trials_and_mask()` function: (1) required non-NaN fields: stimOn_times, choice, feedback_times, probabilityLeft, firstMovement_times, feedbackType; (2) reaction time 0.08-2.0s; (3) trial duration (feedback_times - goCue_times) <= 10s; (4) choice != 0 (no-response excluded). Additionally, trials are excluded if wheel or whisker behavioral streams don't cover the full window, or if no spikes are detected in the binned neural data.

ii.
```python
def make_trial_mask(trials: pd.DataFrame) -> np.ndarray:
    required = ["stimOn_times", "choice", "feedback_times",
                 "probabilityLeft", "firstMovement_times", "feedbackType"]
    mask = np.ones(len(trials), dtype=bool)
    rt = trials["firstMovement_times"].to_numpy() - trials["stimOn_times"].to_numpy()
    trial_len = trials["feedback_times"].to_numpy() - trials["goCue_times"].to_numpy()
    mask &= rt >= 0.08
    mask &= rt <= 2.0
    mask &= trial_len <= 10.0
    mask &= trials["choice"].to_numpy() != 0
    for col in required:
        mask &= ~trials[col].isna().to_numpy()
    return mask

# Additional filters:
final_mask = trial_mask & wheel_valid & whisker_valid & neural_valid
```

iii. The AI documented that this matches the reference `load_trials_and_mask(...)` logic. The additional filters (behavior coverage, non-zero neural) are practical additions for data quality.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `spikes.times.npy` (spike timestamps) and `spikes.clusters.npy` (cluster assignments) from the pykilosort output directory for each probe.

ii.
```python
spikes_times = np.load(probe_base / "spikes.times.npy").astype(np.float32)
spikes_clusters = np.load(probe_base / "spikes.clusters.npy").astype(np.int64)
```

iii. These are the standard IBL spike-sorted data files, consistent with the reference code's use of `SpikeSortingLoader`.

## 2-b. How is the `neural` data processed?

i. Spikes are binned into 20ms non-overlapping time bins within each trial's [-0.5, 1.5]s window (100 bins total). For each bin, spike counts per neuron are computed. Multiple probes from the same session are merged into a single neural population. The result is stored as float16.

ii.
```python
def bin_spikes(spike_times, spike_clusters, intervals, n_neurons):
    ...
    for lo, hi, start in zip(start_idx, end_idx, starts, strict=True):
        times = spike_times[lo:hi] - start
        bins = np.floor(times / BIN_SIZE_S).astype(np.int64)
        valid = (bins >= 0) & (bins < NBINS)
        flat = spike_clusters[lo:hi][valid] * NBINS + bins[valid]
        counts = np.bincount(flat, minlength=n_neurons * NBINS).reshape(n_neurons, NBINS)
        trials.append(counts.astype(np.float16))
```

iii. The AI matched the reference code's 20ms bin size and 2-second window. The binning approach (floor division) is equivalent to the reference's `bincount2D`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI filters neurons using `clusters.metrics.label >= 1.0`, retaining only well-isolated units. This yields 75,708 good units across the full release, matching the data paper's reported number. The reference caching code (`0_data_caching.py`) does NOT apply this QC filter -- it loads ALL units.

ii.
```python
metrics = pd.read_parquet(probe_base / "clusters.metrics.pqt", columns=["cluster_id", "label"])
cluster_ids = metrics["cluster_id"].to_numpy(dtype=np.int64)
good_mask = metrics["label"].to_numpy(dtype=float) >= 1.0
good_cluster_ids = cluster_ids[good_mask]
```

iii. The AI justified this by referencing the data paper's criterion for well-isolated neurons and verified the total matches 75,708. However, the reference code (`0_data_caching.py`) calls `load_spiking_data(one, pid, eid=eid, pname=probe_name)` without a `qc` parameter, meaning ALL units are included in the caching pipeline.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to `stimOn_times`. For each trial, a time window of [-0.5, +1.5] seconds relative to stimulus onset is used. Spikes within this window are binned into 20ms bins.

ii.
```python
ALIGN_EVENT = "stimOn_times"
WINDOW = (-0.5, 1.5)
stim_on = trials[ALIGN_EVENT].to_numpy(dtype=np.float32)
intervals = np.c_[stim_on + WINDOW[0], stim_on + WINDOW[1]].astype(np.float32)
neural_trials = bin_spikes(spike_times, spike_clusters, intervals, len(region_labels))
```

iii. This matches the instructions' requirement for stimulus onset alignment and the reference code's params: `'align_time': 'stimOn_times', 'time_window': (-.5, 1.5)`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is 20ms (0.02s), yielding 100 bins per 2-second trial. No rebinning is applied; spikes are binned directly at this resolution.

ii.
```python
BIN_SIZE_S = 0.02
NBINS = int(round((WINDOW[1] - WINDOW[0]) / BIN_SIZE_S))  # = 100
```

iii. The 20ms bin size matches the reference code's `'binsize': 0.02` and the methods paper's description for dynamic behavior decoding. The metadata reports `time_bin_size: 20.0` (in ms).

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. This input is not derived from raw data variables. It is computed analytically from the window definition and bin size as the end time of each bin relative to stimulus onset.

ii.
```python
time_input = (WINDOW[0] + BIN_SIZE_S * np.arange(1, NBINS + 1)).astype(np.float32)
# = [-0.48, -0.46, ..., 1.50]
```

iii. This provides the bin end times relative to stimulus onset, matching the reference code's `x_interp = np.linspace(interval_begs + binsize, interval_ends, n_bins)` but expressed relative to stim onset.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. No processing of raw data is needed. The values are deterministic: `WINDOW[0] + BIN_SIZE_S * k` for `k = 1, ..., 100`, giving bin end times from -0.48s to +1.50s relative to stimulus onset.

ii.
```python
time_input = (WINDOW[0] + BIN_SIZE_S * np.arange(1, NBINS + 1)).astype(np.float32)
```

iii. These represent the right edge of each 20ms bin, consistent with how the reference code computes interpolation time points.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The time input values are the same for every trial and represent bin end times relative to stimulus onset. Since neural bins are also defined relative to stimulus onset with the same bin size and window, the alignment is one-to-one: bin index i of the time input corresponds to bin index i of the neural data.

ii.
```python
input_trials = [
    np.vstack([
        time_input,  # same for all trials
        np.full(NBINS, trial_number[i], dtype=np.float32),
    ]).astype(np.float32)
    for i in selected_idx
]
```

iii. The time input is inherently aligned since it's derived from the same window/bin parameters as the neural data.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. Trial number in block is derived from the `probabilityLeft` column of the trials table.

ii.
```python
trial_number = compute_trial_number_in_block(trials["probabilityLeft"].to_numpy(dtype=np.float32))
```

iii. The AI uses changes in `probabilityLeft` to detect block boundaries, since `probabilityLeft` is the variable that defines blocks in the IBL task.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The AI iterates through all trials (before filtering) and tracks consecutive runs of the same `probabilityLeft` value. When `probabilityLeft` changes, the counter resets to 1. The resulting trial number is a 1-indexed count within each block. This is computed on ALL trials, then indexed by the kept trial indices.

ii.
```python
def compute_trial_number_in_block(prob_left: np.ndarray) -> np.ndarray:
    trial_num = np.ones(prob_left.shape[0], dtype=np.float32)
    curr = 1.0
    for i in range(1, prob_left.shape[0]):
        if np.isclose(prob_left[i], prob_left[i - 1]):
            curr += 1.0
        else:
            curr = 1.0
        trial_num[i] = curr
    return trial_num
```

iii. This is a reasonable approach to compute trial-within-block position. The instructions specify "Trial number in block, continuous, per-trial" without further detail on how to compute it. Using `probabilityLeft` changes as block boundaries aligns with the IBL task structure.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. Choice is derived from the `choice` column of `_ibl_trials.table.pqt`.

ii.
```python
choice = map_choice(trials["choice"].to_numpy(dtype=np.float32)[selected_idx])
```

iii. This is the standard IBL choice variable.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The IBL `choice` values (-1 for left, +1 for right) are mapped to the decoder format: left=-1 -> 0, right=+1 -> 1. The per-trial value is repeated across all 100 time bins.

ii.
```python
def map_choice(values: np.ndarray) -> np.ndarray:
    out = np.full(values.shape[0], -1, dtype=np.int8)
    out[np.isclose(values, -1.0)] = 0  # left
    out[np.isclose(values, 1.0)] = 1   # right
    if np.any(out < 0):
        raise ValueError("Unexpected choice values encountered")
    return out
```

iii. This matches the instructions: "Choice, binary, per-trial, left = 0, right = 1". The choice=0 (no-response) trials are already excluded by the trial mask.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. Prior probability of left is derived from the `probabilityLeft` column of `_ibl_trials.table.pqt`.

ii.
```python
prior = map_probability_left(trials["probabilityLeft"].to_numpy(dtype=np.float32)[selected_idx])
```

iii. This is the standard IBL block probability variable.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The continuous `probabilityLeft` values (0.2, 0.5, 0.8) are mapped to categorical labels: 0.2 -> 0, 0.5 -> 1, 0.8 -> 2. The per-trial value is repeated across all 100 time bins.

ii.
```python
def map_probability_left(values: np.ndarray) -> np.ndarray:
    rounded = np.round(values.astype(np.float64), 1)
    out = np.full(values.shape[0], -1, dtype=np.int8)
    out[np.isclose(rounded, 0.2)] = 0
    out[np.isclose(rounded, 0.5)] = 1
    out[np.isclose(rounded, 0.8)] = 2
    if np.any(out < 0):
        raise ValueError("Unexpected probabilityLeft values encountered")
    return out
```

iii. This matches the instructions: "Prior probability of left, per-trial, 0.2 -> 0, 0.5 -> 1, 0.8 -> 2".

## 9-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. Wheel speed is derived from `_ibl_wheel.timestamps.npy` (wheel encoder timestamps) and `_ibl_wheel.position.npy` (wheel encoder position).

ii.
```python
def load_wheel_speed(session_path: Path) -> tuple[np.ndarray, np.ndarray]:
    timestamps = np.load(pick_latest(session_path, "alf/**/_ibl_wheel.timestamps.npy"))
    position = np.load(pick_latest(session_path, "alf/**/_ibl_wheel.position.npy"))
```

iii. These are the standard IBL wheel data files, matching what the reference `SessionLoader.load_wheel()` loads.

## 9-b. What processing is involved in computing `output` *Wheel speed*?

i. The raw wheel position is interpolated to 1000 Hz using linear interpolation (`interpolate_position`), then a Butterworth low-pass filtered velocity is computed using `velocity_filtered` with corner frequency 20 Hz and order 8. The absolute value of velocity gives wheel speed.

ii.
```python
interp_pos, interp_t = interpolate_position(timestamps, position, freq=WHEEL_FS)
velocity, _ = velocity_filtered(interp_pos, fs=WHEEL_FS, corner_frequency=20, order=8)
return interp_t.astype(np.float32), np.abs(velocity).astype(np.float32)
```

iii. This exactly matches the reference `SessionLoader.load_wheel()` which uses the same `interpolate_position` and `velocity_filtered` functions with the same default parameters (fs=1000, corner_frequency=20, order=8), followed by `np.abs()` for speed.

## 9-c. How is `output` *Wheel speed* thresholded into categories?

i. Wheel speed is discretized into 3 categories using global tertile thresholds computed across all included time bins from all sessions. Thresholds at the 1/3 and 2/3 quantiles are computed, then `np.digitize` maps values to categories 0 (low), 1 (medium), 2 (high).

ii.
```python
wheel_thresholds = np.quantile(np.concatenate(wheel_pool), [1 / 3, 2 / 3]).astype(np.float32)
# ...
def digitize_tertiles(values: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    return np.digitize(values, thresholds, right=False).astype(np.int8)
```

iii. The instructions specify "Wheel speed discretized into 3 bins, time-varying" without specifying the discretization method. The reference code does not discretize (uses continuous values). The AI chose global tertiles to ensure balanced classes.

## 9-d. How is `output` *Wheel speed* aligned with the neural data?

i. The AI's stated approach is to interpolate the 1000 Hz wheel speed signal into the same 100 time bins as the neural data, using bin end times as interpolation points. However, the implementation has a coordinate frame mismatch.

ii.
```python
def interpolate_behavior_into_trials(times, values, intervals):
    ...
    x_interp = (WINDOW[0] + BIN_SIZE_S * np.arange(1, NBINS + 1)).astype(np.float32)
    # x_interp = [-0.48, -0.46, ..., 1.50] (time relative to stimulus onset)
    ...
    for i, (ib, ie, beg, end) in enumerate(...):
        t = times[ib:ie]
        y = values[ib:ie]
        rel_t = t - beg  # time relative to interval START (stim_on + WINDOW[0])
        interp = np.interp(x_interp, rel_t, y)
```

The bug: `x_interp` values are in the range [-0.48, 1.50] (time relative to stimulus onset), but `rel_t` values are in the range [~0, ~2.0] (time relative to interval start). These differ by WINDOW[0] = -0.5s. The result is that the interpolated behavior is shifted by ~25 bins (0.5s) relative to the neural data. The correct code should use `x_interp = BIN_SIZE_S * np.arange(1, NBINS + 1)` for consistency with `rel_t`.

iii. The AI intended to align behavior to the same bin times as neural data, but the coordinate mismatch introduces a 0.5-second temporal shift in the time-varying behavioral outputs.

## 10-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. Whisker motion energy is derived from `{view}Camera.ROIMotionEnergy.npy` (motion energy values) and `_ibl_{view}Camera.times.npy` (camera timestamps), where `view` is 'left' (preferred) or 'right' (fallback).

ii.
```python
def _load_camera_stream(session_path: Path, view: str):
    times = np.load(pick_latest(session_path, f"alf/**/*_ibl_{view}Camera.times.npy"))
    values = np.load(pick_latest(session_path, f"alf/**/{view}Camera.ROIMotionEnergy.npy"))
```

iii. These are the standard IBL motion energy files, matching the reference `SessionLoader.load_motion_energy()`.

## 10-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The raw whisker motion energy signal is loaded directly (no additional signal processing). If camera timestamps are longer than the motion energy array, leading timestamps are trimmed to match (consistent with `_check_video_timestamps` in the reference code). Left camera is preferred; right camera is used as fallback if left fails to load.

ii.
```python
def _load_camera_stream(session_path, view):
    ...
    if times.shape[0] > values.shape[0]:
        times = times[-values.shape[0]:]  # trim leading timestamps
    return times, values

def load_whisker_motion_energy(session_path):
    try:
        times, values = _load_camera_stream(session_path, "left")
        return times, values, "left"
    except Exception:
        times, values = _load_camera_stream(session_path, "right")
        return times, values, "right"
```

iii. This matches the reference code's `_check_video_timestamps` (trim leading timestamps when times > data) and `bin_behaviors` (left-first fallback for whisker-motion-energy).

## 10-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Same as wheel speed: global tertile thresholds computed across all included time bins from all sessions, then `np.digitize` maps to 3 categories.

ii.
```python
whisker_thresholds = np.quantile(np.concatenate(whisker_pool), [1 / 3, 2 / 3]).astype(np.float32)
```

iii. Same justification as wheel speed discretization. Global tertiles ensure balanced classes without session-specific drift.

## 10-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. The same `interpolate_behavior_into_trials` function is used for whisker motion energy as for wheel speed. It suffers from the same coordinate frame mismatch bug described in 9-d, resulting in a ~0.5s temporal shift of the whisker motion energy relative to the neural data.

ii.
```python
whisker_trials, whisker_valid = interpolate_behavior_into_trials(whisker_times, whisker_me, intervals)
```

iii. Same coordinate bug as 9-d. The whisker ME signal at ~60 Hz (left) or ~150 Hz (right) is interpolated to 100 bins, but the interpolation coordinates are shifted.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Multiple strategies handle missing or problematic data:
- Sessions with missing files (wheel, whisker, spike data) are skipped entirely with `FileNotFoundError` caught.
- Sessions with unexpected errors are skipped with the exception type logged.
- Trials with NaN in required fields are excluded by the trial mask.
- Trials where behavior streams don't cover the full window (checked via gap > BIN_SIZE_S) are excluded.
- Trials with NaN in interpolated behavior are excluded.
- Trials with zero spikes across all neurons are excluded.
- Camera timestamp length mismatches are handled by trimming leading timestamps (if times > values) or raising an error (if times < values).
- Sessions with fewer than 2 valid trials after all filtering are skipped.

ii.
```python
except FileNotFoundError:
    skip_reasons["missing_required_file"] += 1
except Exception as exc:
    skip_reasons[type(exc).__name__] += 1

# Behavior coverage check:
if np.abs(beg - t[0]) > BIN_SIZE_S or np.abs(end - t[-1]) > BIN_SIZE_S:
    outputs.append(None)

# Zero-spike trial check:
neural_valid = np.array([np.any(trial > 0) for trial in neural_trials], dtype=bool)
```

iii. The AI documented these as practical data quality measures. The approach of skipping sessions with missing files and filtering trials with incomplete data coverage is consistent with the reference code's error handling patterns.

## 12-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are: (1) loading spike data from disk for each session (reading multiple large numpy files per probe), (2) binning spikes into trials in `bin_spikes` (iterating over all trials per session), and (3) loading and processing wheel data (interpolation to 1000 Hz and Butterworth filtering). The session loop itself processes 459 sessions sequentially.

ii.
```python
# Loading spike data (large files per probe):
spikes_times = np.load(probe_base / "spikes.times.npy").astype(np.float32)
spikes_clusters = np.load(probe_base / "spikes.clusters.npy").astype(np.int64)

# Binning spikes (loop over trials):
for lo, hi, start in zip(start_idx, end_idx, starts, strict=True):
    ...
```

iii. The AI did not explicitly discuss time-consuming steps in CONVERSION_NOTES.md but included timing output (elapsed time printed every 25 sessions).

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. Three loops could be vectorized: (1) `compute_trial_number_in_block` uses a Python for-loop over all trials to detect block changes; this could use `np.diff` and `np.cumsum`. (2) `bin_spikes` loops over trials to compute per-trial spike counts; this could potentially use vectorized binning. (3) `interpolate_behavior_into_trials` loops over trials for interpolation.

ii.
```python
# Python for-loop in compute_trial_number_in_block:
for i in range(1, prob_left.shape[0]):
    if np.isclose(prob_left[i], prob_left[i - 1]):
        curr += 1.0
    else:
        curr = 1.0
    trial_num[i] = curr

# Trial loop in bin_spikes:
for lo, hi, start in zip(start_idx, end_idx, starts, strict=True):
    ...

# Trial loop in interpolate_behavior_into_trials:
for i, (ib, ie, beg, end) in enumerate(...):
    ...
```

iii. The Python for-loop in `compute_trial_number_in_block` is the most obvious candidate for vectorization. The trial loops are harder to vectorize due to variable-length data per trial but could benefit from parallelization (the reference code uses multiprocessing).

## 12-c. What processing does the code repeat multiple times?

i. The code does not obviously repeat major processing steps. The `x_interp` array is computed once outside the loop in `interpolate_behavior_into_trials`. Each session's data is processed once. The brain region mapping (`id2acronym`, `acronym2acronym`) is done per-session but this is unavoidable. A `BrainRegions()` object is created once and reused across sessions.

ii.
```python
br = BrainRegions()  # created once
# x_interp computed once:
x_interp = (WINDOW[0] + BIN_SIZE_S * np.arange(1, NBINS + 1)).astype(np.float32)
```

iii. The code is reasonably structured to avoid redundant computation.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several outputs are computed but not used by the downstream decoder: (1) Continuous wheel speed and whisker motion energy arrays are stored in `wheel_pool`/`whisker_pool` solely for computing global tertile thresholds, then the continuous values are discarded. (2) Extensive metadata (session paths, whisker view names, per-session trial counts, excluded session IDs) is stored in the pickle file but not needed by the decoder. (3) Brain region labels include non-grey-matter regions (root, void, etc.) that would be filtered in the reference paper's analyses.

ii.
```python
# Continuous values collected only for threshold computation:
wheel_pool.append(np.concatenate(wheel_cont))
whisker_pool.append(np.concatenate(whisker_cont))
wheel_thresholds = np.quantile(np.concatenate(wheel_pool), [1 / 3, 2 / 3])

# Extensive metadata stored:
"metadata": {
    ...
    "session_info": [{"eid": ..., "session_path": ..., ...} for rec in records],
    ...
}
```

iii. The continuous behavior pools are a necessary intermediate for computing thresholds. The metadata overhead is minimal in terms of processing time and provides useful provenance information.
