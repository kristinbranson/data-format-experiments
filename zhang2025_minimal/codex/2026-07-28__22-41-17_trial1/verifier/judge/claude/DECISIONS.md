# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads `bwm_release.csv` (the brain-wide map release table) to get the list of sessions and probe insertions, then resolves file paths directly on disk using the `lab/Subjects/subject/date/session_number` convention in the ONE cache. It does NOT use the ONE API; instead, it reads `.npy`, `.pqt`, and other files directly using numpy/pandas. The release CSV provides all session metadata (eid, subject, date, lab, probe_name).

ii.
```python
release = pd.read_csv(RELEASE_CSV, index_col=0)
sessions = (
    release.groupby("eid", sort=False)
    .agg({
        "subject": "first",
        "date": "first",
        "session_number": "first",
        "lab": "first",
        "probe_name": lambda x: tuple(sorted(x)),
    })
    .reset_index()
)
```

```python
def find_session_path(row: pd.Series) -> Path:
    return DATA_ROOT / row["lab"] / "Subjects" / row["subject"] / row["date"] / f"{int(row['session_number']):03d}"
```

iii. The AI chose to read the release CSV and resolve paths manually rather than using the ONE API. This avoids dependencies on the ONE client but bypasses the session availability checks that the ONE API provides (e.g., checking which datasets actually exist locally).

## 1-b. How are the data split into subjects?

i. Subjects are identified from the `subject` column of `bwm_release.csv`. After processing, unique subjects are collected in encounter order and assigned indices. The `subject_idx` array maps each session to its subject index.

ii.
```python
subjects = []
subject_to_idx = {}
for rec in records:
    if rec["subject"] not in subject_to_idx:
        subject_to_idx[rec["subject"]] = len(subjects)
        subjects.append(rec["subject"])
```

iii. Subjects come directly from the release table metadata; no parsing of paths or filenames is needed.

## 1-c. How are the data split into sessions?

i. Sessions are identified by unique `eid` values in the release CSV. The CSV is grouped by `eid` to get one row per session with aggregated probe names. Each session is processed independently.

ii.
```python
sessions = (
    release.groupby("eid", sort=False)
    .agg({...})
    .reset_index()
)
```

iii. Each eid represents a unique session, so the groupby gives one entry per session.

## 1-d. How are the data split into trials?

i. Trials come from the `_ibl_trials.table.pqt` parquet file loaded per session. Each row is one trial.

ii.
```python
def load_trials_table(session_path: Path) -> pd.DataFrame:
    return pd.read_parquet(pick_latest(session_path, "alf/**/_ibl_trials.table.pqt"))
```

iii. The trials table is already one row per trial; no splitting is needed.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered by multiple criteria: (1) reaction time between 0.08 s and 2.0 s, (2) trial length (feedback_times - goCue_times) <= 10 s, (3) no-choice trials excluded (choice != 0), (4) required fields must not be NaN (stimOn_times, choice, feedback_times, probabilityLeft, firstMovement_times, feedbackType), (5) wheel and whisker streams must cover the full trial window, (6) trials with zero spikes across all neurons are dropped.

ii.
```python
def make_trial_mask(trials: pd.DataFrame) -> np.ndarray:
    required = [
        "stimOn_times", "choice", "feedback_times",
        "probabilityLeft", "firstMovement_times", "feedbackType",
    ]
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
```

```python
neural_valid = np.array([np.any(trial > 0) for trial in neural_trials], dtype=bool)
final_mask = trial_mask & wheel_valid & whisker_valid & neural_valid
```

iii. The AI based its trial mask on the reference code's `load_trials_and_mask()` in `ibl_data_utils.py`. It added a trial length constraint (`<= 10 s`) and zero-spike trial exclusion beyond the reference solution's filtering.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Two arrays per probe: `spikes.times.npy` (spike timestamps) and `spikes.clusters.npy` (cluster assignment for each spike). Cluster quality labels come from `clusters.metrics.pqt`, and anatomical regions come from `clusters.channels.npy` and `channels.brainLocationIds_ccf_2017.npy`.

ii.
```python
spikes_times = np.load(probe_base / "spikes.times.npy").astype(np.float32)
spikes_clusters = np.load(probe_base / "spikes.clusters.npy").astype(np.int64)
```

iii. These are the standard IBL spike sorting outputs read directly from the file system.

## 2-b. How is the `neural` data processed?

i. Spikes are binned into 20 ms bins over the 2 s trial window (-0.5 to 1.5 s relative to stimulus onset), giving 100 bins. The raw spike counts are stored directly as float16 -- they are NOT divided by the bin width to convert to firing rates. When a session has multiple probes, their units are pooled into a single population with continuous cluster numbering.

ii.
```python
bins = np.floor(times / BIN_SIZE_S).astype(np.int64)
valid = (bins >= 0) & (bins < NBINS)
flat = spike_clusters[lo:hi][valid] * NBINS + bins[valid]
counts = np.bincount(flat, minlength=n_neurons * NBINS).reshape(n_neurons, NBINS)
trials.append(counts.astype(np.float16))
```

iii. The AI bins spikes into 20 ms bins matching the reference specification, but stores raw counts as float16 rather than converting to firing rates (Hz) by dividing by the bin size. The reference code divides by BIN (0.02) to get Hz. The AI also uses float16 precision instead of float32.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only clusters with `label >= 1.0` in `clusters.metrics.pqt` are kept. This corresponds to the IBL's stringent quality control criterion. The AI reports this reproduces the paper's 75,708 well-isolated units.

ii.
```python
metrics = pd.read_parquet(probe_base / "clusters.metrics.pqt", columns=["cluster_id", "label"])
good_mask = metrics["label"].to_numpy(dtype=float) >= 1.0
good_cluster_ids = cluster_ids[good_mask]
```

iii. Matches the data paper's quality control criterion.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial's spikes are aligned to `stimOn_times` (stimulus onset). The trial window runs from -0.5 s to +1.5 s relative to stimulus onset. Spike times within each trial are expressed relative to the window start (stimulus onset + T_START).

ii.
```python
stim_on = trials[ALIGN_EVENT].to_numpy(dtype=np.float32)
intervals = np.c_[stim_on + WINDOW[0], stim_on + WINDOW[1]].astype(np.float32)
```

```python
times = spike_times[lo:hi] - start  # start = stim_on + WINDOW[0]
bins = np.floor(times / BIN_SIZE_S).astype(np.int64)
```

iii. The alignment event matches the instructions ("Temporally align based on stimulus onset").

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The bin size is 20 ms, producing 100 bins over the 2 s window. No rebinning or interpolation is applied.

ii.
```python
BIN_SIZE_S = 0.02
NBINS = int(round((WINDOW[1] - WINDOW[0]) / BIN_SIZE_S))  # 100
```

iii. Matches the reference code's `binsize: 0.02` and the paper's specification of 20 ms bins.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. Derived from `stimOn_times` in the trials table, which defines the alignment event. The input is a time axis constructed from the bin edges of the trial window.

ii.
```python
time_input = (WINDOW[0] + BIN_SIZE_S * np.arange(1, NBINS + 1)).astype(np.float32)
```

iii. The time values represent the position within each trial's window relative to stimulus onset.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The time input is computed as bin END times rather than bin CENTERS. The values are `[-0.48, -0.46, ..., 1.50]`, which are `T_START + BIN * np.arange(1, NBINS+1)`. The reference code uses bin centers: `EDGES[:-1] + BIN/2` = `[-0.49, -0.47, ..., 1.49]`.

ii.
```python
# AI code: bin end times
time_input = (WINDOW[0] + BIN_SIZE_S * np.arange(1, NBINS + 1)).astype(np.float32)
# = [-0.48, -0.46, ..., 1.50]

# Reference code: bin centers
# TIME = EDGES[:-1] + BIN / 2
# = [-0.49, -0.47, ..., 1.49]
```

iii. The AI's CONVERSION_NOTES.md acknowledges this: "bin end times: `[-0.48, -0.46, ..., 1.50]`". The difference is 10 ms (half a bin) relative to the reference's bin centers.

## 3-c. How is `input` *Time since stimulus onset* aligned with the neural data?

i. The time input uses the same window definition as the neural binning, but offset by half a bin. The neural data is binned using `np.floor(times / BIN_SIZE_S)` starting from the window start, while the time input uses bin end times. So there is a half-bin (10 ms) misalignment between the time labels and the actual bin centers.

ii.
```python
# Neural bins: floor division from window start
bins = np.floor(times / BIN_SIZE_S).astype(np.int64)
# Time input: bin end times
time_input = (WINDOW[0] + BIN_SIZE_S * np.arange(1, NBINS + 1)).astype(np.float32)
```

iii. The AI uses bin end times as the time axis while binning spikes from the left edge of each bin, creating a systematic 10 ms offset.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. Derived from `probabilityLeft` in the trials table. Blocks are identified by detecting changes in `probabilityLeft` across consecutive trials.

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

iii. The same approach as the reference: blocks are recovered from transitions in `probabilityLeft`.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The trial number starts at 1 for the first trial in each block and increments by 1 for each subsequent trial in the same block (1-indexed). The computation is done on ALL trials before any filtering, so filtered-out trials still advance the count.

ii.
```python
trial_number = compute_trial_number_in_block(trials["probabilityLeft"].to_numpy(dtype=np.float32))
```

iii. The reference uses `groupby(block).cumcount()` which is 0-indexed (starts at 0). The AI starts at 1. This is a difference of 1 in all trial-in-block values.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. From the `choice` column in the trials table, which takes values +1 (left), -1 (right), and 0 (no response).

ii.
```python
def map_choice(values: np.ndarray) -> np.ndarray:
    out = np.full(values.shape[0], -1, dtype=np.int8)
    out[np.isclose(values, -1.0)] = 0
    out[np.isclose(values, 1.0)] = 1
    if np.any(out < 0):
        raise ValueError("Unexpected choice values encountered")
    return out
```

iii. The AI reads choice from the trials table, same as the reference.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The AI maps IBL choice values to decoder labels: -1 (right) -> 0 and +1 (left) -> 1. This is REVERSED from the instructions which specify "left = 0, right = 1". The reference correctly maps +1 (left) -> 0 and -1 (right) -> 1.

ii.
```python
# AI mapping (REVERSED):
out[np.isclose(values, -1.0)] = 0   # right -> 0
out[np.isclose(values, 1.0)] = 1    # left -> 1

# Reference mapping (CORRECT):
CHOICE = {1.0: 0, -1.0: 1}  # left -> 0, right -> 1
```

iii. The instructions explicitly state "left = 0, right = 1". The IBL convention is +1 = left, -1 = right. The AI has the mapping backwards.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. From `probabilityLeft` in the trials table, which takes values 0.2, 0.5, and 0.8.

ii.
```python
def map_probability_left(values: np.ndarray) -> np.ndarray:
    rounded = np.round(values.astype(np.float64), 1)
    out = np.full(values.shape[0], -1, dtype=np.int8)
    out[np.isclose(rounded, 0.2)] = 0
    out[np.isclose(rounded, 0.5)] = 1
    out[np.isclose(rounded, 0.8)] = 2
    return out
```

iii. The mapping 0.2->0, 0.5->1, 0.8->2 matches the instructions exactly.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Values are rounded to 1 decimal place and mapped using `np.isclose` comparisons: 0.2->0, 0.5->1, 0.8->2. The rounding step is an extra precaution not in the reference.

ii.
```python
rounded = np.round(values.astype(np.float64), 1)
out[np.isclose(rounded, 0.2)] = 0
out[np.isclose(rounded, 0.5)] = 1
out[np.isclose(rounded, 0.8)] = 2
```

iii. The reference uses a simple dictionary map: `trials.probabilityLeft.map(PRIOR)`. The AI's approach is functionally equivalent but more defensive.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. From `_ibl_wheel.position.npy` and `_ibl_wheel.timestamps.npy`. These are loaded directly and processed through `interpolate_position` and `velocity_filtered` from brainbox.

ii.
```python
timestamps = np.load(pick_latest(session_path, "alf/**/_ibl_wheel.timestamps.npy"))
position = np.load(pick_latest(session_path, "alf/**/_ibl_wheel.position.npy"))
interp_pos, interp_t = interpolate_position(timestamps, position, freq=WHEEL_FS)
velocity, _ = velocity_filtered(interp_pos, fs=WHEEL_FS, corner_frequency=20, order=8)
```

iii. Same raw data and processing pipeline as the reference, using the same brainbox functions.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Three steps: (1) wheel position is interpolated to 1000 Hz, (2) velocity is computed with a 20 Hz Butterworth low-pass filter (order 8), (3) absolute value is taken to get speed. The continuous speed is then interpolated onto trial time bins and discretized into 3 categories using GLOBAL tertile thresholds across all sessions.

ii.
```python
velocity, _ = velocity_filtered(interp_pos, fs=WHEEL_FS, corner_frequency=20, order=8)
return interp_t.astype(np.float32), np.abs(velocity).astype(np.float32)
```

```python
wheel_thresholds = np.quantile(np.concatenate(wheel_pool), [1 / 3, 2 / 3]).astype(np.float32)
```

iii. The wheel processing matches the reference's use of `SessionLoader.load_wheel()`. However, the discretization uses GLOBAL tertiles (across all sessions) rather than PER-SESSION percentiles as in the reference.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. The AI computes global tertile thresholds across ALL included sessions and time bins, then applies `np.digitize` to categorize each time bin into one of 3 classes (0=low, 1=medium, 2=high). The reference computes percentiles PER SESSION.

ii.
```python
# Global thresholds across all sessions:
wheel_pool.append(np.concatenate(wheel_cont))
wheel_thresholds = np.quantile(np.concatenate(wheel_pool), [1 / 3, 2 / 3]).astype(np.float32)

# Applied to each trial:
def digitize_tertiles(values, thresholds):
    return np.digitize(values, thresholds, right=False).astype(np.int8)
```

```python
# Reference: per-session percentiles
def discretize(trace):
    return np.digitize(trace, np.percentile(trace, SPLIT))
```

iii. The AI explicitly chose global tertiles "to avoid session-specific label drift" and "keep class balance near uniform for decoder training." The reference uses per-session percentiles at 33.3% and 66.7%.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The wheel speed is interpolated onto the same time points as the neural bins using `np.interp`. However, the AI uses bin END times for interpolation (`WINDOW[0] + BIN_SIZE_S * np.arange(1, NBINS + 1)`) rather than bin CENTERS as in the reference.

ii.
```python
x_interp = (WINDOW[0] + BIN_SIZE_S * np.arange(1, NBINS + 1)).astype(np.float32)
interp = np.interp(x_interp, rel_t, y).astype(np.float32)
```

iii. The interpolation approach is similar to the reference but offset by half a bin (10 ms).

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. From `<side>Camera.ROIMotionEnergy.npy` and `_ibl_<side>Camera.times.npy`. The left camera is preferred; if unavailable, the right camera is used as fallback.

ii.
```python
def load_whisker_motion_energy(session_path: Path) -> tuple:
    try:
        times, values = _load_camera_stream(session_path, "left")
        return times, values, "left"
    except Exception:
        times, values = _load_camera_stream(session_path, "right")
        return times, values, "right"
```

iii. Matches the reference's camera fallback logic.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The released motion energy trace is used as-is (no filtering or normalization). It is interpolated onto trial time bins (bin end times) and discretized using GLOBAL tertile thresholds, same approach as wheel speed.

ii.
```python
whisker_trials, whisker_valid = interpolate_behavior_into_trials(whisker_times, whisker_me, intervals)
whisker_thresholds = np.quantile(np.concatenate(whisker_pool), [1 / 3, 2 / 3]).astype(np.float32)
```

iii. Same global discretization approach as wheel speed.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Global tertile thresholds computed across all sessions, then `np.digitize` applied to each trial. Same method as wheel speed.

ii.
```python
whisker_thresholds = np.quantile(np.concatenate(whisker_pool), [1 / 3, 2 / 3]).astype(np.float32)
whisker_disc = digitize_tertiles(whisker, whisker_thresholds)
```

iii. The reference uses per-session percentiles. The AI uses global tertiles.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Same approach as wheel speed: interpolated onto bin end times using `np.interp`, offset by half a bin from the reference's bin centers.

ii.
```python
x_interp = (WINDOW[0] + BIN_SIZE_S * np.arange(1, NBINS + 1)).astype(np.float32)
interp = np.interp(x_interp, rel_t, y).astype(np.float32)
```

iii. Same half-bin offset issue as wheel speed alignment.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Multiple strategies: (1) Sessions with missing required files (wheel, whisker, trials, spikes) are skipped via exception handling, (2) Probes with no good units are skipped, (3) Sessions with < 2 trials after filtering are skipped, (4) Trials where wheel or whisker streams don't cover the full window are marked invalid, (5) Trials with zero spikes are excluded, (6) When camera timestamps are longer than motion energy values, leading timestamps are trimmed.

ii.
```python
except FileNotFoundError:
    skip_reasons["missing_required_file"] += 1
except Exception as exc:
    skip_reasons[type(exc).__name__] += 1
```

```python
if times.shape[0] > values.shape[0]:
    times = times[-values.shape[0]:]
```

iii. The approach is generally robust, handling missing files, mismatched array lengths, and insufficient data gracefully.

## 10-a. What are the most time-consuming steps of the code?

i. Loading spike sorting data from disk is the most expensive step, as each probe's `spikes.times.npy` and `spikes.clusters.npy` files can be hundreds of megabytes. The code processes sessions sequentially (no parallelization), which makes this even more impactful.

ii.
```python
spikes_times = np.load(probe_base / "spikes.times.npy").astype(np.float32)
spikes_clusters = np.load(probe_base / "spikes.clusters.npy").astype(np.int64)
```

iii. File I/O dominates the runtime. The reference code uses `ProcessPoolExecutor` for parallel processing; the AI processes sessions sequentially.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Two per-trial loops exist: (1) the spike binning loop in `bin_spikes` iterates over each trial, and (2) the behavior interpolation loop in `interpolate_behavior_into_trials` iterates over each trial. Both could potentially be vectorized.

ii.
```python
for lo, hi, start in zip(start_idx, end_idx, starts, strict=True):
    ...
```

```python
for i, (ib, ie, beg, end) in enumerate(zip(idx_beg, idx_end, starts, ends, strict=True)):
    ...
```

iii. These are standard per-trial loops similar to the reference code's approach.

## 10-c. What processing does the code repeat multiple times?

i. The behavioral interpolation function `interpolate_behavior_into_trials` is called twice with essentially the same interval computation -- once for wheel and once for whisker data. The interval calculation (`np.c_[stim_on + WINDOW[0], stim_on + WINDOW[1]]`) and searchsorted operations are duplicated.

ii.
```python
wheel_trials, wheel_valid = interpolate_behavior_into_trials(wheel_times, wheel_speed, intervals)
whisker_trials, whisker_valid = interpolate_behavior_into_trials(whisker_times, whisker_me, intervals)
```

iii. Minor redundancy; the interval computation is shared but the searchsorted must differ per stream since wheel and whisker have different time bases.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code collects extensive metadata including per-session info, skip reasons, and sanity statistics that are stored in the pickle but not used by the decoder. It also stores continuous wheel and whisker values temporarily in `wheel_pool` and `whisker_pool` for global threshold computation, which consumes significant memory.

ii.
```python
"metadata": {
    ...
    "trial_exclusion": {...},
    "continuous_output_discretization": {...},
    "session_info": [...],
    "conversion_summary": summarize_records(records, skip_reasons),
}
```

iii. The extra metadata is useful for documentation but is not used by downstream decoder training.
