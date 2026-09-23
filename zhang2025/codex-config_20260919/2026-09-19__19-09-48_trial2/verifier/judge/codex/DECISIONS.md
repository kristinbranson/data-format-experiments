# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent does not use the ONE API. It loads a fixed publication-freeze table from `bwm_release.csv`, groups rows by `eid`, constructs filesystem paths directly from `lab/subject/date/session_number`, then reads trial parquet files and ALF `.npy`/`.pqt` files with pandas/NumPy. When multiple physical revisions exist, it picks the lexicographically newest explicit revision with `preferred(...)`.

ii.
```python
FREEZE = APP / "code" / "code_zhang2025" / "data" / "bwm_release.csv"
DATA_ROOT = APP / "data" / "one_cache"
```

```python
def session_path(row: pd.Series) -> Path:
    return (
        DATA_ROOT
        / str(row.lab)
        / "Subjects"
        / str(row.subject)
        / str(row.date)
        / f"{int(row.session_number):03d}"
    )
```

```python
freeze = pd.read_csv(FREEZE, index_col=0)
groups = list(freeze.groupby("eid", sort=False))
```

iii. In `CONVERSION_NOTES.md`, the agent justifies this as using the supplied publication freeze exactly, avoiding two extra cached sessions not in the release, and manually reproducing ONE’s revision resolution by selecting the newest revised file.

## 1-b. How are the data split into subjects?

i. Subjects are taken from the release-freeze metadata, not inferred from paths at assembly time. Each converted session stores its subject string, and `subjects` is built from first appearance order with `dict.fromkeys(...)`; `subject_idx` indexes into that order.

ii.
```python
subjects = list(dict.fromkeys(x["session_info"]["subject"] for x in converted))
subject_lookup = {name: i for i, name in enumerate(subjects)}
```

```python
"subject_idx": np.asarray(
    [subject_lookup[x["session_info"]["subject"]] for x in converted], dtype=np.int64
),
```

iii. The notes say the release table is authoritative for subject identity and that stable first-appearance ordering is sufficient as long as `subject_idx` is consistent.

## 1-c. How are the data split into sessions?

i. Sessions are defined by unique `eid` values from the release freeze. The script groups the release table by `eid`, and each group corresponds to one converted session, potentially with multiple probes.

ii.
```python
groups = list(freeze.groupby("eid", sort=False))
```

```python
for eid, rows in groups:
    result = process_session(str(eid), rows, brain_atlas, want_plot)
```

iii. The agent’s justification is that the publication freeze already specifies the released session set and associated probes, so grouping by `eid` is the release-consistent session unit.

## 1-d. How are the data split into trials?

i. Trials are taken directly from the rows of the session trial table `_ibl_trials.table.pqt`. After filtering, `trial_indices = np.flatnonzero(valid)` selects the surviving trial rows; each retained row becomes one trial in `neural`, `input`, and `output`.

ii.
```python
def load_trials(path: Path) -> tuple[pd.DataFrame, Path]:
    trial_file = preferred(path.glob("alf/**/_ibl_trials.table.pqt"))
    ...
    return pd.read_parquet(trial_file), trial_file
```

```python
valid = base_mask & motion_ok & wheel_ok
trial_indices = np.flatnonzero(valid)
```

iii. The agent treats the native trial table as already trialized and only applies filtering plus index selection.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered with a mask that requires non-missing `stimOn_times`, `choice`, `feedback_times`, `probabilityLeft`, `firstMovement_times`, and `feedbackType`; reaction time in `[0.08, 2.0]` s; duration `feedback_times - goCue_times <= 10` s; nonzero choice; and full wheel and whisker-motion coverage across the whole `[-0.5, 1.5]` s window.

ii.
```python
required = [
    "stimOn_times",
    "choice",
    "feedback_times",
    "probabilityLeft",
    "firstMovement_times",
    "feedbackType",
]
...
rt = (trials["firstMovement_times"] - trials["stimOn_times"]).to_numpy()
rt_ok = (rt >= 0.08) & (rt <= 2.00)
duration = (trials["feedback_times"] - trials["goCue_times"]).to_numpy()
duration_ok = duration <= 10.0
choice_ok = trials["choice"].to_numpy() != 0
mask = present & rt_ok & duration_ok & choice_ok
```

```python
motion_ok = coverage_mask(motion, begins_all, ends_all)
wheel_ok = coverage_mask(wheel, begins_all, ends_all)
valid = base_mask & motion_ok & wheel_ok
```

iii. The notes say this reproduces `load_trials_and_mask` from the method code and then adds a joint coverage requirement for the two continuous outputs so the final decoder data have complete aligned windows.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural array is built from `spikes.times.npy` and `spikes.clusters.npy`, with `clusters.metrics.pqt`, `clusters.channels.npy`, and channel/electrode atlas IDs used to choose good units and assign brain regions.

ii.
```python
metrics_path = preferred(probe_path.glob("**/clusters.metrics.pqt"))
channels_path = preferred(probe_path.glob("**/clusters.channels.npy"))
atlas_path = preferred(probe_path.glob("electrodeSites.brainLocationIds_ccf_2017.npy"))
...
spike_times_path = preferred(probe_path.glob("**/spikes.times.npy"))
spike_clusters_path = preferred(probe_path.glob("**/spikes.clusters.npy"))
```

iii. The notes explicitly map `spikes.times`, `spikes.clusters`, cluster labels, and peak-channel atlas IDs onto the retained neural representation and region annotations.

## 2-b. How is the `neural` data processed?

i. The agent merges all release probes within a session, bins spikes into 100 non-overlapping 20 ms bins over `[-0.5, 1.5]` s around stimulus onset, and stores raw spike counts as `float32`. It does not divide by bin width to convert counts into Hz.

ii.
```python
BIN_SIZE = 0.020
OFF_START = -0.5
OFF_END = 1.5
N_BINS = int(round((OFF_END - OFF_START) / BIN_SIZE))
```

```python
bins = np.floor(rel_t / BIN_SIZE).astype(np.int64)
...
destination[trial] = np.bincount(flat, minlength=width).reshape(
    destination.shape[1], N_BINS
)
```

```python
neural_trials.append(counts[i].astype(np.float32))
```

iii. In the notes, the agent says raw counts match the reference spike binning but intentionally keep the dense target representation tractable, and it labels the metadata as `"raw spike counts from label>=1 well-isolated units"`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units are filtered to `clusters.metrics.label >= 1`, using the release metrics parquet. The agent does not explicitly remove Beryl `void` units, although it does assign every retained unit a Beryl label from its peak-channel atlas ID.

ii.
```python
metrics = pd.read_parquet(metrics_path)
good_rows = np.flatnonzero(metrics["label"].to_numpy() >= 1)
cluster_ids = metrics["cluster_id"].to_numpy(dtype=np.int64)[good_rows]
```

```python
allen = brain_regions.id2acronym(np.asarray(atlas_ids[peak_channels], dtype=np.int64))
beryl = brain_regions.acronym2acronym(allen, mapping="Beryl").astype(str)
```

iii. The notes justify `label >= 1` as following the data paper’s well-isolated-neuron curation and as reducing the size of the dense decoder dataset; they acknowledge this intentionally differs from the method cache’s `qc=None`.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to `stimOn_times`. For each retained trial, the binning window starts at `stimOn_times - 0.5` s and ends at `stimOn_times + 1.5` s; spike times are converted to offsets from the window start before binning.

ii.
```python
begins_all = trials["stimOn_times"].to_numpy(dtype=float) + OFF_START
ends_all = trials["stimOn_times"].to_numpy(dtype=float) + OFF_END
```

```python
rel_t = np.asarray(times[lo:hi], dtype=np.float64)[keep] - beg
bins = np.floor(rel_t / BIN_SIZE).astype(np.int64)
```

iii. The notes repeatedly state that all requested streams are aligned to stimulus onset on a common `[-0.5, +1.5)` window.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The neural data use 20 ms bins, 100 bins per 2 s trial window. No additional temporal rebinning is applied beyond this binning.

ii.
```python
BIN_SIZE = 0.020
N_BINS = int(round((OFF_END - OFF_START) / BIN_SIZE))
```

iii. The notes justify 20 ms as matching the method code and the requested common alignment/binning across streams.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. The input time coordinate is defined relative to `stimOn_times`. The raw variable that anchors it is the trial’s `stimOn_times`, but the actual values stored are from a fixed synthetic time grid.

ii.
```python
TIME_GRID = (OFF_START + BIN_SIZE * np.arange(1, N_BINS + 1)).astype(np.float32)
```

```python
begins_all = trials["stimOn_times"].to_numpy(dtype=float) + OFF_START
ends_all = trials["stimOn_times"].to_numpy(dtype=float) + OFF_END
```

iii. The notes describe this input as “time since stimulus onset” on the reference/common grid, repeated for every trial.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. No raw measurement is transformed. The script creates a fixed 100-point grid from `-0.48` to `1.50` s in 20 ms steps and repeats that row for every trial.

ii.
```python
TIME_GRID = (OFF_START + BIN_SIZE * np.arange(1, N_BINS + 1)).astype(np.float32)
```

```python
inp = np.vstack(
    (TIME_GRID, np.full(N_BINS, block_number[i], dtype=np.float32))
).astype(np.float32, copy=False)
```

iii. The notes justify this as using the same right-edge coordinate as the behavioral interpolation grid.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. It is aligned bin-for-bin to the neural representation by using the right edge of each half-open neural spike-count bin. The first time point corresponds to `-0.48` s and the last to `1.50` s.

ii.
```python
TIME_GRID = (OFF_START + BIN_SIZE * np.arange(1, N_BINS + 1)).astype(np.float32)
```

```python
bins = np.floor(rel_t / BIN_SIZE).astype(np.int64)
```

iii. The notes explicitly claim the time coordinate should be the “right edge of each half-open neural spike-count bin.”

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from the raw `probabilityLeft` sequence in the trial table.

ii.
```python
native_prob = trials["probabilityLeft"].to_numpy(dtype=float)
block_number = block_trial_numbers(native_prob)[trial_indices]
```

iii. The agent’s notes say blocks are recovered from changes in `probabilityLeft` because no direct block counter exists in the raw data.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The script computes a zero-based trial counter within each native probability block before any trial filtering. Consecutive equal `probabilityLeft` values increment the counter; any change resets it to zero. The retained per-trial scalar is then repeated across all 100 time bins.

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

```python
np.full(N_BINS, block_number[i], dtype=np.float32)
```

iii. The notes justify computing it before filtering so dropped trials still advance later retained trials’ within-block position.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. Choice is derived from the raw `choice` column in the trial table. The agent interprets native `-1` as left and `+1` as right, then maps them to `0` and `1` respectively.

ii.
```python
choice_native = trials["choice"].to_numpy(dtype=float)[trial_indices]
choice = np.where(choice_native == -1, 0, 1).astype(np.int64)
```

iii. The notes explicitly document the same mapping: “Native −1 (left) → 0; +1 (right) → 1.”

## 5-b. What processing is involved in computing `output` *Choice*?

i. After filtering out `choice == 0` trials, the script recodes the remaining `choice` values with the mapping above and repeats the categorical value across all 100 time bins for that trial.

ii.
```python
choice_ok = trials["choice"].to_numpy() != 0
mask = present & rt_ok & duration_ok & choice_ok
```

```python
np.full(N_BINS, choice[i], dtype=np.int64)
```

iii. The agent’s notes justify the time repetition as a way to keep every output trial as a 2D array with the same time dimension as the neural data.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It is derived from `probabilityLeft` in the trial table.

ii.
```python
prior_native = native_prob[trial_indices]
prior = np.full(len(trial_indices), -1, dtype=np.int64)
for value, label in ((0.2, 0), (0.5, 1), (0.8, 2)):
    prior[np.isclose(prior_native, value)] = label
```

iii. The notes say the task requested actual block prior categories, not the method paper’s model-derived subjective prior.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The script recodes raw `probabilityLeft` values from `0.2/0.5/0.8` into class labels `0/1/2`, raises an error if any retained trial has a different value, and repeats the class across all 100 bins of that trial.

ii.
```python
for value, label in ((0.2, 0), (0.5, 1), (0.8, 2)):
    prior[np.isclose(prior_native, value)] = label
if np.any(prior < 0):
    raise ValueError(f"Unexpected probabilityLeft in {eid}")
```

```python
np.full(N_BINS, prior[i], dtype=np.int64)
```

iii. The notes justify this as a direct task-mandated categorical conversion.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. Wheel speed is derived from `_ibl_wheel.timestamps.npy` and `_ibl_wheel.position.npy`.

ii.
```python
times_path = preferred(path.glob("alf/**/_ibl_wheel.timestamps.npy"))
pos_path = preferred(path.glob("alf/**/_ibl_wheel.position.npy"))
```

iii. The notes cite the IBL/SessionLoader wheel-processing path as the intended source processing.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The script interpolates wheel position to a 1 kHz grid, computes filtered velocity with `velocity_filtered(..., corner_frequency=20, order=8)`, takes the absolute value to obtain speed, interpolates that continuous signal onto the trial-aligned 20 ms grid, and discretizes all retained session samples into tertiles.

ii.
```python
pos_1khz, times_1khz = interpolate_position(raw_times, raw_pos, freq=1000)
velocity, _ = velocity_filtered(pos_1khz, fs=1000, corner_frequency=20, order=8)
return (
    BehaviorStream(times_1khz, np.abs(velocity), "wheel"),
    raw_times,
    raw_pos,
)
```

```python
wheel_cont = interpolate_trials(wheel, begins, ends)
wheel_labels, wheel_q, wheel_method = discretize_tertiles(wheel_cont)
```

iii. The notes justify this as reproducing SessionLoader’s wheel processing and then applying the task-required 3-class discretization.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. The agent computes session-specific thresholds at the 1/3 and 2/3 quantiles over all retained interpolated wheel-speed samples. If the quantiles are distinct, it assigns classes with `np.searchsorted(..., side="right")`; if the quantiles tie, it falls back to a stable-rank equal-frequency split.

ii.
```python
def discretize_tertiles(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    flat = values.ravel()
    thresholds = np.quantile(flat, [1 / 3, 2 / 3])
    if thresholds[0] < thresholds[1]:
        labels = np.searchsorted(thresholds, values, side="right").astype(np.int64)
        return labels, thresholds.astype(float), "value_quantiles"
    order = np.argsort(flat, kind="stable")
    ranked = np.empty(flat.size, dtype=np.int64)
    ranked[order] = np.minimum(2, (np.arange(flat.size) * 3) // flat.size)
    return ranked.reshape(values.shape), thresholds.astype(float), "stable_rank_quantiles"
```

iii. The notes justify session-wise tertiles as mirroring the reference’s per-session scaling, and the stable-rank fallback as a deterministic safeguard when thresholds collapse.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. Wheel speed is aligned by interpolating it onto the same 100-point right-edge time grid used for the neural trial window. The wheel trace is therefore sampled once per neural bin, but at bin right edges rather than bin centers.

ii.
```python
steps = BIN_SIZE * np.arange(1, N_BINS + 1)
...
x = beg + steps
y = np.interp(x, t, v)
```

iii. The notes repeatedly justify this as matching the agent’s interpretation of the reference behavior-sampling implementation.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It is derived from either `leftCamera.ROIMotionEnergy.npy` plus `_ibl_leftCamera.times.npy`, or, if unavailable, the corresponding right-camera files. Left camera is preferred.

ii.
```python
for side in ("left", "right"):
    values_path = preferred(path.glob(f"alf/**/{side}Camera.ROIMotionEnergy.npy"))
    times_path = preferred(path.glob(f"alf/**/_ibl_{side}Camera.times.npy"))
    if values_path is None or times_path is None:
        continue
```

iii. The notes explicitly say the conversion should “prefer left, fall back right,” following the reference behavior loader.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The script loads the released motion-energy trace as-is, trims leading camera timestamps if there are more timestamps than frames, rejects malformed/non-monotonic streams, interpolates the retained stream onto the session’s trial-aligned 20 ms grid, and discretizes the resulting values into session-wise tertiles.

ii.
```python
values = np.asarray(np.load(values_path, mmap_mode="r"), dtype=np.float64)
times = np.asarray(np.load(times_path, mmap_mode="r"), dtype=np.float64)
if len(times) > len(values):
    times = times[-len(values) :]
if not np.all(np.diff(times) > 0):
    continue
return BehaviorStream(times, values, side)
```

```python
motion_cont = interpolate_trials(motion, begins, ends)
motion_labels, motion_q, motion_method = discretize_tertiles(motion_cont)
```

iii. The notes justify the timestamp trimming as matching SessionLoader’s camera-timestamp correction and otherwise using the released motion-energy values unchanged apart from alignment/discretization.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. It is thresholded exactly like wheel speed: session-level 1/3 and 2/3 quantiles over all retained aligned samples, with `searchsorted(..., side="right")` when thresholds are distinct and a stable-rank fallback when they tie.

ii.
```python
motion_labels, motion_q, motion_method = discretize_tertiles(motion_cont)
```

```python
return ranked.reshape(values.shape), thresholds.astype(float), "stable_rank_quantiles"
```

iii. The notes justify using the same discretization rule for both continuous outputs so the decoder sees comparable low/medium/high categorical targets.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Whisker motion energy is aligned by interpolating it onto the same 100-point right-edge time grid used for the neural bins.

ii.
```python
steps = BIN_SIZE * np.arange(1, N_BINS + 1)
...
x = beg + steps
y = np.interp(x, t, v)
```

iii. The notes justify this with the same right-edge alignment claim used for wheel speed and the time input.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The agent mostly drops unusable data. It chooses a preferred revision when multiple files exist, trims excess leading camera timestamps, skips sessions with no valid whisker stream, drops trials failing wheel or motion coverage, skips sessions with fewer than two valid trials or zero retained units, and raises hard errors for structurally inconsistent files or unexpected `probabilityLeft` values.

ii.
```python
def preferred(paths) -> Path | None:
    paths = list(paths)
    if not paths:
        return None
    return sorted(paths, key=lambda p: ("#" in str(p), str(p)))[-1]
```

```python
if len(times) > len(values):
    times = times[-len(values) :]
if not np.all(np.diff(times) > 0):
    continue
```

```python
if motion is None:
    print(f"SKIP {eid}: no valid left/right whisker motion-energy stream", flush=True)
    return None
...
if len(trial_indices) < 2:
    print(f"SKIP {eid}: {len(trial_indices)} jointly valid trials", flush=True)
    return None
...
if n_units == 0:
    print(f"SKIP {eid}: no label>=1 units", flush=True)
    return None
```

iii. The notes justify these checks as enforcing complete aligned data windows, exact release membership, and early failure on structural inconsistencies rather than silent corruption.

## 10-a. What are the most time-consuming steps of the code?

i. The agent identifies spike-array I/O and spike binning as the dominant cost, especially if all spikes were materialized at once. It therefore treats spike memmap access and trial-window binning as the main performance concern.

ii.
```python
times = np.load(info["times_path"], mmap_mode="r")
clusters = np.load(info["clusters_path"], mmap_mode="r")
```

```python
for trial, (beg, lo, hi) in enumerate(zip(begins, left, right)):
    ...
    destination[trial] = np.bincount(flat, minlength=width).reshape(
        destination.shape[1], N_BINS
    )
```

iii. The notes say reading billions of spike events and binning them is the expensive part, while behavior processing is secondary.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The agent explicitly targets per-spike/per-trial processing as the place where vectorization matters most. Its notes point to per-trial behavioral interpolation and spike binning as loops that would be slow in pure Python, and the final code partly vectorizes them but still leaves trial loops in `interpolate_trials` and `bin_probe`.

ii.
```python
for i, (beg, end, ib, ie) in enumerate(zip(begins, ends, ibs, ies)):
    ...
    y = np.interp(x, t, v)
```

```python
for trial, (beg, lo, hi) in enumerate(zip(begins, left, right)):
    ...
    destination[trial] = np.bincount(flat, minlength=width).reshape(
        destination.shape[1], N_BINS
    )
```

iii. The notes justify the implemented compromise as keeping the logic clear while eliminating the most expensive unnecessary object creation and full-session materialization.

## 10-c. What processing does the code repeat multiple times?

i. The code repeats some metadata/file processing. It performs a global `release_preflight(...)` pass over every probe’s metrics file to count raw and good units, and then later re-reads each probe’s metrics again during actual session conversion in `load_probe_units(...)`. It also repeatedly constructs full per-trial input/output arrays in Python loops after already having session-wide aligned continuous traces.

ii.
```python
def release_preflight(freeze: pd.DataFrame) -> dict[str, int]:
    raw = 0
    good = 0
    for _, row in freeze.iterrows():
        ...
        metrics = pd.read_parquet(metrics_path, columns=["label"])
        raw += len(metrics)
        good += int((metrics["label"] >= 1).sum())
```

```python
for _, row in rows.sort_values("probe_name").iterrows():
    ...
    info = load_probe_units(probe_path, brain_regions)
```

iii. The notes justify the preflight reread as a release-integrity check, not as part of the minimal conversion path.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes and briefly keeps more information than the downstream decoder consumes. It constructs continuous `wheel_cont` and `motion_cont` only to discretize them, keeps raw wheel timestamps/positions mainly for optional diagnostic plots, performs release-wide preflight counting, and stores extensive per-session provenance/threshold metadata that the decoder itself does not use.

ii.
```python
wheel_cont = interpolate_trials(wheel, begins, ends)
motion_cont = interpolate_trials(motion, begins, ends)
wheel_labels, wheel_q, wheel_method = discretize_tertiles(wheel_cont)
motion_labels, motion_q, motion_method = discretize_tertiles(motion_cont)
```

```python
wheel, raw_wheel_times, raw_wheel_pos = load_wheel_speed(path)
```

```python
"wheel_tertile_thresholds": wheel_q.tolist(),
"wheel_discretization": wheel_method,
"whisker_motion_energy_tertile_thresholds": motion_q.tolist(),
"whisker_motion_energy_discretization": motion_method,
```

iii. The notes justify these extras as validation/audit support rather than strict decoder requirements.
