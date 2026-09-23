# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads sessions using the freeze CSV file (`code_zhang2025/data/bwm_release.csv`) rather than the ONE API. It reads the CSV to enumerate all 459 sessions/699 probes/139 subjects, constructs file paths directly from lab/subject/date/session_number columns, and loads data files (trials parquet, spike npy, wheel npy, camera npy) from the staged ALF directory tree. It does not use `SessionLoader` or `SpikeSortingLoader` from the IBL libraries for data loading.

ii.
```python
FREEZE_CSV = APP / "code" / "code_zhang2025" / "data" / "bwm_release.csv"

def load_freeze() -> tuple[pd.DataFrame, list[dict]]:
    freeze = pd.read_csv(FREEZE_CSV)
    ...
    for eid, group in freeze.groupby("eid", sort=False):
        first = group.iloc[0]
        sessions.append({
            "eid": str(eid),
            "lab": str(first["lab"]),
            "subject": str(first["subject"]),
            ...
        })
    return freeze, sessions
```

```python
def session_path(row: pd.Series) -> Path:
    return (
        DATA_ROOT / str(row["lab"]) / "Subjects" / str(row["subject"])
        / str(row["date"]) / f"{int(row['session_number']):03d}"
    )
```

iii. The AI chose direct file access over the ONE API because the data is staged locally and the ONE client had issues with offline resolution. The freeze CSV provides an authoritative list of sessions from the paper.

## 1-b. How are the data split into subjects?

i. Subjects are identified from the `subject` column of the freeze CSV. After processing, subjects are the sorted unique names from successful sessions, and `subject_idx` maps each session to its index.

ii.
```python
subjects = sorted({r["info"]["subject"] for r in results})
subject_lookup = {subject: i for i, subject in enumerate(subjects)}
subject_idx = np.asarray(
    [subject_lookup[r["info"]["subject"]] for r in results], dtype=np.int32,
)
```

iii. Subject identity comes directly from the freeze CSV metadata; no parsing of paths is needed.

## 1-c. How are the data split into sessions?

i. Sessions are the unique EIDs from the freeze CSV. Each EID corresponds to one session. Sessions are processed independently and assembled in freeze-CSV order.

ii.
```python
for eid, group in freeze.groupby("eid", sort=False):
    ...
    sessions.append({...})
```

iii. The freeze CSV already lists sessions individually; no splitting is needed.

## 1-d. How are the data split into trials?

i. Trials are rows in the `_ibl_trials.table.pqt` parquet file for each session. Each row is one trial.

ii.
```python
def load_trials(info: dict) -> tuple[pd.DataFrame, Path]:
    alf = session_path(pd.Series(info)) / "alf"
    path = newest_file(alf, "_ibl_trials.table.pqt", "#2025-03-03#")
    trials = pd.read_parquet(path)
    return trials, path
```

iii. The trials table is already one row per trial.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies several filters: (1) all required columns must be finite (stimOn_times, choice, feedback_times, probabilityLeft, firstMovement_times, feedbackType); (2) reaction time between 0.08-2.0s; (3) choice != 0 (no-go excluded); (4) goCue-to-feedback duration <= 10s; (5) wheel and whisker motion energy stream coverage for the trial window; (6) neural recording coverage (trial window within probe recording bounds).

ii.
```python
def trial_mask(trials: pd.DataFrame) -> np.ndarray:
    required = [
        "stimOn_times", "choice", "feedback_times", "probabilityLeft",
        "firstMovement_times", "feedbackType",
    ]
    good = np.ones(len(trials), dtype=bool)
    for column in required:
        good &= trials[column].notna().to_numpy()
    reaction_time = (
        trials["firstMovement_times"].to_numpy(dtype=float)
        - trials["stimOn_times"].to_numpy(dtype=float)
    )
    good &= reaction_time >= 0.08
    good &= reaction_time <= 2.0
    good &= trials["choice"].to_numpy() != 0
    if "goCue_times" in trials:
        duration = (
            trials["feedback_times"].to_numpy(dtype=float)
            - trials["goCue_times"].to_numpy(dtype=float)
        )
        good &= ~(duration > 10.0)
    return good
```

Neural coverage check:
```python
neural_good = (
    (stim_code + OFF_START >= neural_coverage_start)
    & (stim_code + OFF_END <= neural_coverage_end)
)
```

iii. The AI says it reproduces `load_trials_and_mask(..., max_trial_len=10)` from the reference code. The goCue-to-feedback <=10s filter and finite-value checks on feedbackType/feedback_times are additional compared to the human reference, which only checks RT bounds, valid choice, and valid probabilityLeft plus wheel/camera coverage.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from `spikes.times.npy` and `spikes.clusters.npy` for each probe, plus `clusters.metrics.pqt` for cluster metadata and `channels.brainLocationIds_ccf_2017.npy` for brain region mapping.

ii.
```python
spikes_times_path = directory / "spikes.times.npy"
spikes_clusters_path = directory / "spikes.clusters.npy"
...
spike_times = np.load(probe["spikes_times_path"], mmap_mode="r")
spike_clusters = np.load(probe["spikes_clusters_path"], mmap_mode="r")
```

iii. These are the standard IBL spike sorting outputs.

## 2-b. How is the `neural` data processed?

i. Spikes are counted into 100 half-open 20ms bins over the [-0.5, 1.5) s window around stimulus onset. The AI stores **raw spike counts** (not firing rates). When a session has multiple probes, their clusters are concatenated with disjoint offsets. ALL Kilosort clusters are retained (no quality filtering).

ii.
```python
bins = np.floor((times - starts[trial]) / BIN_SIZE).astype(np.int64)
keep = (bins >= 0) & (bins < N_BINS)
...
code = ((local_trial * n_clusters + clusters) * N_BINS + bins).astype(np.int64)
...
counts = np.bincount(flat, minlength=(last - first) * n_clusters * N_BINS)
```

```python
output = np.zeros((n_trials, n_clusters, N_BINS), dtype=np.float32)
```

iii. The AI's CONVERSION_NOTES state: "Spike values are raw counts per 20 ms bin, not rates" and "The caching script requests all clusters (qc=None); quality labels are saved but not applied." The AI justifies keeping all clusters because the Zhang et al. decoder reference code loads `qc=None`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI does **not** filter neurons by quality label. All Kilosort-sorted clusters are retained, including those with label < 1 and those in `void`/`root` brain regions. The AI records quality counts (`n_good_label_clusters`) as metadata but does not use them for filtering.

ii.
```python
def load_probe_metadata(alf: Path, probe_name: str) -> dict:
    ...
    metrics = pd.read_parquet(metrics_path)
    cluster_ids = metrics["cluster_id"].to_numpy(dtype=np.int64)
    n_clusters = len(metrics)
    ...
    "n_good": int((metrics["label"].to_numpy() >= 1).sum()),
    ...
```

No filtering is applied -- all `n_clusters` are used.

iii. The AI states in CONVERSION_NOTES: "The requested neural-decoder conversion follows Zhang/caching code: retain all 621,733 sorted units. Quality filtering would change the reference decoder input and discard multiunit information."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to stimulus onset (`stimOn_times`). Each trial's spike window starts at `stimOn - 0.5s` and ends at `stimOn + 1.5s`.

ii.
```python
stim_times = trials["stimOn_times"].to_numpy(dtype=float)[source_indices]
starts = stim_times + OFF_START
ends = stim_times + OFF_END
...
bins = np.floor((times - starts[trial]) / BIN_SIZE).astype(np.int64)
```

iii. This matches the instructions which specify temporal alignment based on stimulus onset.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is 20ms, producing 100 bins over the 2s window. No rebinning or interpolation is applied to the neural data.

ii.
```python
BIN_SIZE = 0.020
OFF_START = -0.5
OFF_END = 1.5
N_BINS = 100
```

iii. This matches the reference code's 20ms bins and 100 time steps.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is derived from the fixed temporal grid, not from any raw data variable per se. The grid is defined by the bin parameters and is the same for every trial.

ii.
```python
REL_SAMPLE_TIMES = np.linspace(OFF_START + BIN_SIZE, OFF_END, N_BINS).astype(np.float32)
```

This produces values `[-0.48, -0.46, ..., 1.48, 1.50]` -- the right edges of each 20ms bin.

iii. The time input is defined by the temporal grid parameters.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The AI computes the time input as the right edge of each 20ms bin: `np.linspace(-0.5 + 0.02, 1.5, 100)`. This produces values from -0.48 to 1.50 in 0.02 steps.

ii.
```python
REL_SAMPLE_TIMES = np.linspace(OFF_START + BIN_SIZE, OFF_END, N_BINS).astype(np.float32)
...
time_input = np.broadcast_to(REL_SAMPLE_TIMES, (len(source_indices), N_BINS))
```

iii. The AI uses right-edge times to match the behavior sampling convention. The reference uses bin centres (`EDGES[:-1] + BIN/2` = `[-0.49, -0.47, ..., 1.49]`). These differ by 0.01s (half a bin).

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The time input values are the right edges of each neural bin, so each time value corresponds to the end of the neural bin it labels.

ii.
```python
time_input = np.broadcast_to(REL_SAMPLE_TIMES, (len(source_indices), N_BINS))
```

iii. The AI states behavior samples label "the preceding spike-count bin end." This is slightly different from the reference which uses bin centres.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. From the `probabilityLeft` column of the trials table. Blocks are identified by detecting changes in `probabilityLeft`.

ii.
```python
def trial_number_in_block(probability_left: np.ndarray) -> np.ndarray:
    probability_left = np.asarray(probability_left, dtype=float)
    changes = np.ones(len(probability_left), dtype=bool)
    if len(probability_left) > 1:
        changes[1:] = ~np.isclose(
            probability_left[1:], probability_left[:-1], rtol=0.0, atol=1e-8,
            equal_nan=False,
        )
    starts = np.maximum.accumulate(np.where(changes, np.arange(len(probability_left)), 0))
    return (np.arange(len(probability_left)) - starts).astype(np.float32)
```

iii. Same approach as the reference: detect block boundaries from changes in probabilityLeft.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. Block boundaries are detected where `probabilityLeft` changes. The trial number within each block is the zero-based index from the start of each block. This is computed on the full (unfiltered) trial table, so filtered trials preserve their original position in the block.

ii.
```python
raw_block_numbers = trial_number_in_block(trials["probabilityLeft"].to_numpy())
...
block_numbers = raw_block_numbers[source_indices]
```

iii. Computing before filtering preserves the real experimental trial position, matching the reference approach.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. From the `choice` column of the trials table, where IBL convention uses +1 for left, -1 for right, and 0 for no-go.

ii.
```python
raw_choice = trials["choice"].to_numpy()[source_indices]
choices = (raw_choice == -1).astype(np.int8)
```

iii. The raw choice values are +1 (left), -1 (right), 0 (no-go).

## 5-b. What processing is involved in computing `output` *Choice*?

i. The AI maps -1 (right) to 1, and everything else (which after filtering is only +1/left) to 0. So left=0, right=1 as required.

ii.
```python
choices = (raw_choice == -1).astype(np.int8)
```

iii. This is equivalent to the reference mapping `{1.0: 0, -1.0: 1}`, producing the same result.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. From the `probabilityLeft` column of the trials table, which takes values 0.2, 0.5, and 0.8.

ii.
```python
raw_prior = trials["probabilityLeft"].to_numpy(dtype=float)[source_indices]
priors = np.full(len(raw_prior), -1, dtype=np.int8)
for value, category in ((0.2, 0), (0.5, 1), (0.8, 2)):
    priors[np.isclose(raw_prior, value, rtol=0.0, atol=1e-8)] = category
```

iii. The mapping matches the instructions: 0.2 -> 0, 0.5 -> 1, 0.8 -> 2.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. A direct categorical mapping using `np.isclose` for floating-point comparison. Uses tolerance `atol=1e-8` rather than exact equality.

ii. Same as 6-a.

iii. Functionally equivalent to the reference's `PRIOR = {0.2: 0, 0.5: 1, 0.8: 2}` mapping.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. From `_ibl_wheel.timestamps.npy` and `_ibl_wheel.position.npy`.

ii.
```python
raw_times = np.load(timestamp_path, mmap_mode="r")
raw_position = np.load(position_path, mmap_mode="r")
```

iii. Standard IBL wheel data files.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. (1) Wheel position is interpolated to 1kHz using `interpolate_position`; (2) velocity is computed with 20Hz Butterworth low-pass filtering using `velocity_filtered`; (3) absolute value gives speed; (4) speed is linearly interpolated/extrapolated to the bin right-edge sample times; (5) discretized into 3 categories using session-specific 1/3 and 2/3 quantiles.

ii.
```python
position, times = interpolate_position(raw_times, raw_position, freq=1000)
velocity, _ = velocity_filtered(position, fs=1000, corner_frequency=20, order=8)
speed = np.abs(velocity)
sampled, good = sample_behavior(times, speed, stim_times)
```

```python
def discretize_tertiles(values: np.ndarray) -> tuple[...]:
    thresholds = np.quantile(values[finite], [1 / 3, 2 / 3]).astype(np.float64)
    ...
    categories = np.digitize(clean, thresholds, right=False).astype(np.int8)
```

iii. The wheel processing pipeline uses the same IBL functions as the reference. The AI directly imports `interpolate_position` and `velocity_filtered` rather than going through `SessionLoader`.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Session-specific empirical 1/3 and 2/3 quantiles are computed from all finite values across all retained trials. `np.digitize` with `right=False` assigns values to three bins (0=low, 1=medium, 2=high). Non-finite values are imputed with the session median before categorization.

ii.
```python
thresholds = np.quantile(values[finite], [1 / 3, 2 / 3]).astype(np.float64)
median = float(np.median(values[finite]))
clean = np.where(finite, values, median)
categories = np.digitize(clean, thresholds, right=False).astype(np.int8)
```

iii. The reference uses `np.percentile(trace, [100/3, 200/3])` and `np.digitize(trace, thresholds)`. These are mathematically equivalent (1/3 quantile = 33.33 percentile). The AI additionally handles NaN imputation which the reference does not explicitly do.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. Wheel speed is sampled at the right edges of the neural time bins (`stim_time + REL_SAMPLE_TIMES`), which are the same times as the behavior sampling grid.

ii.
```python
query = stim_times[trial] + REL_SAMPLE_TIMES.astype(np.float64)
sampled[trial] = _linear_interp_extrapolate(tx, vy, query).astype(np.float32)
```

iii. The AI samples at bin right edges while the reference samples at bin centres. The difference is 0.01s (half a bin width).

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. From `leftCamera.ROIMotionEnergy.npy` (preferred) or `rightCamera.ROIMotionEnergy.npy` (fallback), along with corresponding `_ibl_leftCamera.times.npy` or `_ibl_rightCamera.times.npy`.

ii.
```python
for view in ("left", "right"):
    value_path = newest_file(alf, f"{view}Camera.ROIMotionEnergy.npy")
    time_path = newest_file(alf, f"_ibl_{view}Camera.times.npy")
    ...
    values = np.load(value_path, mmap_mode="r")
    times = np.load(time_path, mmap_mode="r")
```

iii. Left camera preferred with right fallback matches the reference approach.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The raw motion energy is used as-is (no additional filtering). It is linearly interpolated/extrapolated to the bin right-edge sample times, then discretized into 3 categories using session-specific 1/3 and 2/3 quantiles. A timestamp-length fix is applied when `len(times) > len(values)` by trimming leading timestamps.

ii.
```python
if len(times) > len(values):
    times = times[-len(values):]
sampled, good = sample_behavior(times, values, stim_times)
```

Discretization same as wheel:
```python
categories = np.digitize(clean, thresholds, right=False).astype(np.int8)
```

iii. No additional smoothing or normalization is applied, consistent with the reference.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Same method as wheel speed: session-specific 1/3 and 2/3 quantiles from finite values, with NaN imputation using session median.

ii. Same as 7-c discretization code.

iii. Same approach as wheel speed discretization.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Whisker motion energy is sampled at the same right-edge sample times as wheel speed, which correspond to the right edges of the neural time bins.

ii.
```python
query = stim_times[trial] + REL_SAMPLE_TIMES.astype(np.float64)
sampled[trial] = _linear_interp_extrapolate(tx, vy, query).astype(np.float32)
```

iii. Same alignment convention as wheel speed. The reference uses bin centres instead.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Multiple strategies: (1) Trials with NaN in required columns are filtered out; (2) Sessions without usable whisker motion energy are skipped; (3) Sessions with <2 valid trials are skipped; (4) Trials outside neural recording bounds are excluded; (5) Non-finite behavior samples are imputed with the session median; (6) Camera timestamp/value length mismatches are fixed by trimming leading timestamps.

ii.
```python
# NaN imputation for behavior
clean = np.where(finite, values, median)
```

```python
# Camera timestamp fix
if len(times) > len(values):
    times = times[-len(values):]
```

```python
# Session skip
if len(source_indices) < 2:
    raise ValueError(f"Only {len(source_indices)} trials after stream coverage")
```

iii. The AI documents that 15 sessions are skipped (14 lacking whisker stream, 1 with insufficient coverage). Non-finite values in continuous behavior are imputed rather than excluding the entire trial.

## 10-a. What are the most time-consuming steps of the code?

i. Loading and binning spike data. The AI uses memory-mapped file access and batched bincount operations to mitigate this.

ii.
```python
spike_times = np.load(probe["spikes_times_path"], mmap_mode="r")
spike_clusters = np.load(probe["spikes_clusters_path"], mmap_mode="r")
```

iii. The AI reports full conversion in ~209s with 24 workers, with neural processing dominating per-session time.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial spike binning loop inside `bin_probe` iterates over trials within chunks. The behavior sampling loop in `sample_behavior` also iterates per trial. Both could potentially be vectorized by constructing offset indices across all trials.

ii.
```python
for local_trial, trial in enumerate(range(first, last)):
    lo = int(np.searchsorted(spike_times, starts[trial], side="left"))
    hi = int(np.searchsorted(spike_times, ends[trial], side="left"))
    ...
```

```python
for trial in range(len(stim_times)):
    ...
    sampled[trial] = _linear_interp_extrapolate(tx, vy, query).astype(np.float32)
```

iii. The AI partially batches trials within chunks for spike binning, but the inner loop remains per-trial.

## 10-c. What processing does the code repeat multiple times?

i. The code loads probe spike times twice: once in `process_session` to check neural coverage bounds (`probe_starts`/`probe_ends`), and again in `bin_probe` for actual binning. This is a minor redundancy mitigated by memory mapping.

ii.
```python
# First load in process_session for coverage check:
spike_times = np.load(probe["spikes_times_path"], mmap_mode="r")
...
probe_starts.append(float(spike_times[0]))
probe_ends.append(float(spike_times[-1]))

# Second load in bin_probe:
spike_times = np.load(probe["spikes_times_path"], mmap_mode="r")
```

iii. The double load is mitigated by memory mapping, so the OS cache likely handles the second access efficiently.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes and stores detailed session metadata (source trial indices, tertile thresholds, imputation counts, processing times, neural coverage bounds) that the decoder does not use. The code also loads and processes behavioral data for trials that may later be excluded by the neural coverage check, since behavior is processed before neural coverage is verified.

ii.
```python
session_info = {
    ...
    "source_trial_indices": source_indices.astype(int).tolist(),
    "wheel_tertiles": wheel_thresholds.tolist(),
    "whisker_tertiles": motion_thresholds.tolist(),
    "wheel_imputed_samples": wheel_imputed,
    ...
}
```

iii. The metadata is useful for debugging and reproducibility even though the decoder doesn't use it directly.
