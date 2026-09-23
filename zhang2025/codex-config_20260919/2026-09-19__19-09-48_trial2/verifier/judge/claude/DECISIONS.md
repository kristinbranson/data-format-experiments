# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads data directly from disk using file paths derived from the publication freeze CSV (`bwm_release.csv`), rather than using the ONE API. It constructs session paths from lab/subject/date/session_number columns and uses `pathlib.Path.glob` with a `preferred()` helper to resolve ALF revisions. Spike data is memory-mapped via `np.load(..., mmap_mode="r")`, trials are read from Parquet, and wheel/camera data from `.npy` files.

ii.
```python
FREEZE = APP / "code" / "code_zhang2025" / "data" / "bwm_release.csv"
freeze = pd.read_csv(FREEZE, index_col=0)
groups = list(freeze.groupby("eid", sort=False))

def session_path(row: pd.Series) -> Path:
    return (DATA_ROOT / str(row.lab) / "Subjects" / str(row.subject)
            / str(row.date) / f"{int(row.session_number):03d}")

def load_trials(path: Path) -> tuple[pd.DataFrame, Path]:
    trial_file = preferred(path.glob("alf/**/_ibl_trials.table.pqt"))
    return pd.read_parquet(trial_file), trial_file
```

iii. The AI chose to use `bwm_release.csv` as the authoritative session/probe list rather than the ONE API search, stating this exactly reproduces published totals (459 sessions, 699 probes). This avoids non-release sessions that exist on disk but aren't part of the release.

## 1-b. How are the data split into subjects?

i. Subjects are identified from the `subject` column in `bwm_release.csv`. The final subject list preserves first-appearance order (via `dict.fromkeys`), not sorted order.

ii.
```python
subjects = list(dict.fromkeys(x["session_info"]["subject"] for x in converted))
subject_lookup = {name: i for i, name in enumerate(subjects)}
```

iii. The subject information comes directly from the release CSV. The ordering difference (first-appearance vs sorted) does not affect correctness.

## 1-c. How are the data split into sessions?

i. Sessions are identified by their `eid` in the release CSV. The CSV is grouped by `eid`, and each group (potentially with multiple probe rows) represents one session.

ii.
```python
groups = list(freeze.groupby("eid", sort=False))
for eid, rows in groups:
    result = process_session(str(eid), rows, brain_atlas, want_plot)
```

iii. The release CSV already lists sessions; no additional splitting is needed.

## 1-d. How are the data split into trials?

i. Trials come from the Parquet trial table loaded per session. Each row is one trial.

ii.
```python
trials, trial_file = load_trials(path)
```

iii. The trial table is already one row per trial; no additional splitting is needed.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies a `reference_trial_mask` that checks: (1) required event columns are not NaN (stimOn_times, choice, feedback_times, probabilityLeft, firstMovement_times, feedbackType), (2) reaction time between 0.08 and 2.0 s, (3) trial duration (feedback_times - goCue_times) <= 10 s, (4) choice != 0. Additionally, coverage masks require both wheel and whisker motion energy to span the full trial window. Sessions with fewer than 2 jointly valid trials are skipped.

ii.
```python
def reference_trial_mask(trials: pd.DataFrame):
    required = ["stimOn_times", "choice", "feedback_times", "probabilityLeft",
                "firstMovement_times", "feedbackType"]
    present = np.ones(len(trials), dtype=bool)
    for name in required:
        present &= trials[name].notna().to_numpy()
    rt = (trials["firstMovement_times"] - trials["stimOn_times"]).to_numpy()
    rt_ok = (rt >= 0.08) & (rt <= 2.00)
    duration = (trials["feedback_times"] - trials["goCue_times"]).to_numpy()
    duration_ok = duration <= 10.0
    choice_ok = trials["choice"].to_numpy() != 0
    mask = present & rt_ok & duration_ok & choice_ok
    return mask, audit

valid = base_mask & motion_ok & wheel_ok
```

iii. The AI states this reproduces the reference code's `load_trials_and_mask` function with `min_rt=0.08, max_rt=2, max_trial_len=10, exclude_nochoice=True`. The duration check comes from the reference method code but is not in the human reference solution.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `spikes.times.npy` and `spikes.clusters.npy` per probe, along with `clusters.metrics.pqt` for quality labels and `clusters.channels.npy` plus atlas ID arrays for brain region annotation.

ii.
```python
spike_times_path = preferred(probe_path.glob("**/spikes.times.npy"))
spike_clusters_path = preferred(probe_path.glob("**/spikes.clusters.npy"))
times = np.load(info["times_path"], mmap_mode="r")
clusters = np.load(info["clusters_path"], mmap_mode="r")
```

iii. The AI loads spike arrays directly as memory maps rather than through the SpikeSortingLoader API, but accesses the same underlying data files.

## 2-b. How is the `neural` data processed?

i. Spikes from good clusters (label >= 1) are counted into 100 half-open 20 ms bins spanning [-0.5, 1.5) s around stimulus onset. Counts are stored as uint16 during construction, then cast to float32. **Critically, the neural data is stored as raw spike counts, NOT firing rates.** Multiple probes within a session are merged by concatenating units.

ii.
```python
counts = np.zeros((len(trial_indices), n_units, N_BINS), dtype=np.uint16)
# ...
flat = mapped[keep][valid].astype(np.int64) * N_BINS + bins[valid]
destination[trial] = np.bincount(flat, minlength=width).reshape(destination.shape[1], N_BINS)
# ...
neural_trials.append(counts[i].astype(np.float32))
```

iii. The AI's CONVERSION_NOTES.md states: "Neural representation is raw spike counts, not firing rates and not fluorescence." The metadata confirms: `'neural_representation': 'raw spike counts from label>=1 well-isolated units'`. This differs from the reference which divides by the bin width to produce firing rates in Hz.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only clusters with `label >= 1` in `clusters.metrics.pqt` are retained. However, the AI does **not** filter out "void" regions (units whose Beryl-mapped region is "void", indicating channels placed outside the brain).

ii.
```python
metrics = pd.read_parquet(metrics_path)
good_rows = np.flatnonzero(metrics["label"].to_numpy() >= 1)
cluster_ids = metrics["cluster_id"].to_numpy(dtype=np.int64)[good_rows]
# ... no void filter applied
allen = brain_regions.id2acronym(np.asarray(atlas_ids[peak_channels], dtype=np.int64))
beryl = brain_regions.acronym2acronym(allen, mapping="Beryl").astype(str)
```

iii. The AI's CONVERSION_NOTES notes that `label >= 1` follows the data paper's explicit curation (75,708 well-isolated neurons). However, it does not mention or implement the void-region exclusion that the reference applies.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial's spike window is defined as `[stimOn_times + OFF_START, stimOn_times + OFF_END)` = `[stimOn - 0.5, stimOn + 1.5)`. Spike times relative to the window start are binned: `floor((spike_time - begin) / BIN_SIZE)`.

ii.
```python
begins_all = trials["stimOn_times"].to_numpy(dtype=float) + OFF_START
ends_all = trials["stimOn_times"].to_numpy(dtype=float) + OFF_END
# ...
rel_t = np.asarray(times[lo:hi], dtype=np.float64)[keep] - beg
bins = np.floor(rel_t / BIN_SIZE).astype(np.int64)
```

iii. Alignment is to stimulus onset as required by the instructions. The binning formula is mathematically equivalent to the reference's `floor((spike_time - onset - T_START) / BIN)`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 20 ms bins, 100 bins total over the 2 s window. No rebinning or smoothing is applied.

ii.
```python
BIN_SIZE = 0.020
OFF_START = -0.5
OFF_END = 1.5
N_BINS = int(round((OFF_END - OFF_START) / BIN_SIZE))  # 100
```

iii. This matches the reference code's 20 ms bin size and the method paper's description.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is derived from the bin grid definition, specifically the right edges of the 100 bins. The values are `OFF_START + BIN_SIZE * arange(1, N_BINS+1)` = `[-0.48, -0.46, ..., 1.50]`.

ii.
```python
TIME_GRID = (OFF_START + BIN_SIZE * np.arange(1, N_BINS + 1)).astype(np.float32)
```

iii. The AI chose right-edge bin times rather than bin centers. The CONVERSION_NOTES states this follows the reference code's behavior interpolation convention.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. No processing; the time grid is a fixed array computed from the bin parameters.

ii.
```python
TIME_GRID = (OFF_START + BIN_SIZE * np.arange(1, N_BINS + 1)).astype(np.float32)
inp = np.vstack((TIME_GRID, np.full(N_BINS, block_number[i], dtype=np.float32)))
```

iii. The grid is defined once and reused for every trial.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The time grid values are the right edges of the neural spike-count bins. The neural bins are half-open intervals `[edge_k, edge_{k+1})` and the time input is `edge_{k+1}` for each bin.

ii.
```python
TIME_GRID = (OFF_START + BIN_SIZE * np.arange(1, N_BINS + 1)).astype(np.float32)
# Neural bins: floor((spike_time - begin) / BIN_SIZE), begin = stimOn + OFF_START
```

iii. The AI explicitly documents this as "right edge of each half-open neural spike-count bin." The reference uses bin centers instead.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. From `probabilityLeft` in the trials table. Block boundaries are detected when consecutive `probabilityLeft` values differ.

ii.
```python
def block_trial_numbers(probability_left: np.ndarray) -> np.ndarray:
    out = np.zeros(len(probability_left), dtype=np.float32)
    count = 0
    for i in range(1, len(probability_left)):
        same = np.isfinite(probability_left[i]) and np.isfinite(probability_left[i - 1])
        same = same and np.isclose(probability_left[i], probability_left[i - 1])
        count = count + 1 if same else 0
        out[i] = count
    return out
```

iii. The AI computes block boundaries from native `probabilityLeft` changes before trial filtering, preserving the animal's real position in the block.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. A zero-based counter increments for consecutive trials with the same `probabilityLeft` and resets to 0 when the value changes. This is computed on all original trials before filtering. The per-trial value is then broadcast to all 100 time bins.

ii.
```python
native_prob = trials["probabilityLeft"].to_numpy(dtype=float)
block_number = block_trial_numbers(native_prob)[trial_indices]
inp = np.vstack((TIME_GRID, np.full(N_BINS, block_number[i], dtype=np.float32)))
```

iii. The AI uses `np.isfinite` and `np.isclose` for robustness against NaN values, which is a minor implementation difference from the reference's pandas-based approach but produces the same result.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. From the `choice` column of the trials table, which takes values +1 (left), -1 (right), and 0 (no-go).

ii.
```python
choice_native = trials["choice"].to_numpy(dtype=float)[trial_indices]
choice = np.where(choice_native == -1, 0, 1).astype(np.int64)
```

iii. No-go trials (choice == 0) are excluded by the trial filter.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The AI maps IBL choice -1 (right) to 0 and +1 (left) to 1. This is **reversed** relative to the instructions which specify "left = 0, right = 1". The output_values labels `["left", "right"]` imply value 0 = "left" and value 1 = "right", but the code assigns right choices to 0 and left choices to 1.

ii.
```python
choice = np.where(choice_native == -1, 0, 1).astype(np.int64)
# Output values: ["left", "right"] implies 0="left", 1="right"
# But code maps: -1 (right) -> 0, +1 (left) -> 1
```

iii. The AI's CONVERSION_NOTES does not discuss this mapping in detail. The mapping contradicts the explicit instruction "left = 0, right = 1".

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. From `probabilityLeft` in the trials table, which takes values 0.2, 0.5, and 0.8.

ii.
```python
prior_native = native_prob[trial_indices]
prior = np.full(len(trial_indices), -1, dtype=np.int64)
for value, label in ((0.2, 0), (0.5, 1), (0.8, 2)):
    prior[np.isclose(prior_native, value)] = label
```

iii. The mapping follows the instructions: 0.2 -> 0, 0.5 -> 1, 0.8 -> 2.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The three prior values are mapped to integers 0, 1, 2 using `np.isclose` for floating-point comparison. A validation check raises an error if any unmapped values remain.

ii.
```python
if np.any(prior < 0):
    raise ValueError(f"Unexpected probabilityLeft in {eid}")
```

iii. The approach is equivalent to the reference's dictionary mapping.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. From `_ibl_wheel.timestamps.npy` and `_ibl_wheel.position.npy`.

ii.
```python
raw_times = np.asarray(np.load(times_path, mmap_mode="r"), dtype=np.float64)
raw_pos = np.asarray(np.load(pos_path, mmap_mode="r"), dtype=np.float64)
```

iii. Same raw data as the reference.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Position is interpolated to 1 kHz and velocity computed with a 20 Hz order-8 Butterworth filter (using `interpolate_position` and `velocity_filtered` from brainbox). Speed is the absolute value of velocity. The speed trace is then linearly interpolated to the right-edge time grid for each trial and discretized into 3 classes using session-wise tertile thresholds.

ii.
```python
pos_1khz, times_1khz = interpolate_position(raw_times, raw_pos, freq=1000)
velocity, _ = velocity_filtered(pos_1khz, fs=1000, corner_frequency=20, order=8)
BehaviorStream(times_1khz, np.abs(velocity), "wheel")
# ...
wheel_cont = interpolate_trials(wheel, begins, ends)
wheel_labels, wheel_q, wheel_method = discretize_tertiles(wheel_cont)
```

iii. The wheel processing pipeline matches the reference's use of SessionLoader internals.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Session-wise empirical tertiles: the 1/3 and 2/3 quantiles of all retained wheel speed samples in the session serve as thresholds. `np.searchsorted(..., side='right')` assigns values to classes 0 (low), 1 (medium), 2 (high). A stable-rank fallback handles tied quantiles.

ii.
```python
def discretize_tertiles(values: np.ndarray):
    flat = values.ravel()
    thresholds = np.quantile(flat, [1/3, 2/3])
    if thresholds[0] < thresholds[1]:
        labels = np.searchsorted(thresholds, values, side="right").astype(np.int64)
        return labels, thresholds, "value_quantiles"
    # Deterministic equal-frequency fallback
    order = np.argsort(flat, kind="stable")
    ranked = np.empty(flat.size, dtype=np.int64)
    ranked[order] = np.minimum(2, (np.arange(flat.size) * 3) // flat.size)
    return ranked.reshape(values.shape), thresholds, "stable_rank_quantiles"
```

iii. The discretization is functionally equivalent to the reference's `np.digitize(trace, np.percentile(trace, [100/3, 200/3]))`.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The wheel speed trace is linearly interpolated onto the right-edge time grid (bin right edges relative to stimulus onset), matching the neural bin boundaries. The reference interpolates at bin centers instead.

ii.
```python
steps = BIN_SIZE * np.arange(1, N_BINS + 1)
x = beg + steps  # beg = stimOn + OFF_START
y = np.interp(x, t, v)
```

iii. The AI states this follows the reference code's behavior interpolation convention of sampling at the right edge of each spike-count bin.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. From `leftCamera.ROIMotionEnergy.npy` (preferred) or `rightCamera.ROIMotionEnergy.npy` (fallback), with corresponding `_ibl_<side>Camera.times.npy`.

ii.
```python
def load_motion_energy(path: Path) -> BehaviorStream | None:
    for side in ("left", "right"):
        values_path = preferred(path.glob(f"alf/**/{side}Camera.ROIMotionEnergy.npy"))
        times_path = preferred(path.glob(f"alf/**/_ibl_{side}Camera.times.npy"))
```

iii. The left-first/right-fallback selection matches the reference.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The motion energy trace is used as-is (no filtering or normalization). When timestamps are longer than values, leading timestamps are trimmed. The trace is linearly interpolated to the right-edge time grid, then discretized into 3 classes using session-wise tertiles (same method as wheel speed).

ii.
```python
if len(times) > len(values):
    times = times[-len(values):]
# ...
motion_cont = interpolate_trials(motion, begins, ends)
motion_labels, motion_q, motion_method = discretize_tertiles(motion_cont)
```

iii. The timestamp correction (trimming leading timestamps) mirrors what SessionLoader does internally.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Same session-wise tertile method as wheel speed: 1/3 and 2/3 quantiles, `searchsorted` assignment, stable-rank fallback.

ii. Same `discretize_tertiles` function as for wheel speed.

iii. Functionally equivalent to the reference's approach.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Same right-edge interpolation as wheel speed: the motion energy trace is sampled at bin right edges relative to stimulus onset.

ii.
```python
motion_cont = interpolate_trials(motion, begins, ends)
# interpolate_trials uses x = beg + BIN_SIZE * arange(1, N_BINS+1)
```

iii. Same alignment convention as all other time-varying variables in the AI's code.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Multiple safeguards: (1) trials with NaN in required columns are excluded, (2) trials without full wheel/whisker coverage are excluded, (3) sessions with no label>=1 units are skipped, (4) sessions with <2 valid trials are skipped, (5) camera timestamps longer than values are handled by trimming leading timestamps, (6) non-monotonic camera timestamps cause fallback to the other camera, (7) a `validate_before_save` function checks all shapes and value ranges before serialization.

ii.
```python
if len(times) > len(values):
    times = times[-len(values):]
if not np.all(np.diff(times) > 0):
    continue
# ...
if n_units == 0:
    print(f"SKIP {eid}: no label>=1 units", flush=True)
    return None
if len(trial_indices) < 2:
    print(f"SKIP {eid}: {len(trial_indices)} jointly valid trials", flush=True)
    return None
```

iii. The AI handles edge cases robustly, with clear skip messages for excluded sessions.

## 10-a. What are the most time-consuming steps of the code?

i. Loading spike sorting data from disk (memory-mapped reads of large spike arrays). The AI identified that I/O was the bottleneck and added a ThreadPoolExecutor for parallel session processing.

ii.
```python
times = np.load(info["times_path"], mmap_mode="r")
clusters = np.load(info["clusters_path"], mmap_mode="r")
# ...
with ThreadPoolExecutor(max_workers=4) as pool:
    for result in pool.map(run_group, groups):
```

iii. CONVERSION_NOTES documents that the initial sequential run projected ~19 minutes, so threading was added to overlap I/O, reducing total time to ~349 s.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Two per-trial loops: (1) `bin_probe` iterates over trials to bin spikes, (2) `interpolate_trials` iterates over trials to interpolate behavior traces. Both could potentially be vectorized.

ii.
```python
# bin_probe loop
for trial, (beg, lo, hi) in enumerate(zip(begins, left, right)):
    # ...

# interpolate_trials loop
for i, (beg, end, ib, ie) in enumerate(zip(begins, ends, ibs, ies)):
    # ...
```

iii. The AI did not discuss vectorization of these loops, though each trial processes a different time window, making full vectorization non-trivial.

## 10-c. What processing does the code repeat multiple times?

i. The `release_preflight` function reads all cluster metrics files to verify release totals, then each session's processing reads the same metrics files again during `load_probe_units`. This is a repeated read of all 699 probe metric files.

ii.
```python
def release_preflight(freeze: pd.DataFrame) -> dict[str, int]:
    for _, row in freeze.iterrows():
        metrics = pd.read_parquet(metrics_path, columns=["label"])
        # ...

# Later, during session processing:
def load_probe_units(probe_path: Path, brain_regions: BrainRegions) -> dict:
    metrics = pd.read_parquet(metrics_path)
```

iii. The preflight is a validation step that the AI deemed necessary for data integrity verification.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The `release_preflight` counts all raw and good units across all 699 probes, but this information is only used for an assertion check and not in the converted data. Additionally, the per-session `audit` dictionary with detailed trial-filter stage counts is stored in metadata but not used by the decoder.

ii.
```python
release_stats = release_preflight(freeze)  # counts all units, only used for assertion
# ...
"trial_filter_counts": audit,  # stored in metadata, not used by decoder
```

iii. These are validation/documentation features, not processing errors.
