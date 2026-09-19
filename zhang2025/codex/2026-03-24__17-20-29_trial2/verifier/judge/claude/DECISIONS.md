# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads `code/code_zhang2025/data/bwm_release.csv` to get the list of 459 release sessions with their eids, subjects, labs, dates, session numbers, and probe names. It then constructs file paths from this metadata to directly load ALF files from `data/one_cache/<lab>/Subjects/<subject>/<date>/<number>/alf/`. It does NOT use the ONE API or `SessionLoader`/`SpikeSortingLoader` because those dependencies were not available in its environment.

ii.
```python
def load_release_sessions() -> list[SessionSpec]:
    bwm = pd.read_csv(RELEASE_CSV, index_col=0)
    grouped = bwm.groupby("eid", sort=False)
    sessions: list[SessionSpec] = []
    for eid, df in grouped:
        row = df.iloc[0]
        probe_names = tuple(df["probe_name"].tolist())
        sessions.append(
            SessionSpec(
                eid=eid,
                subject=str(row["subject"]),
                lab=str(row["lab"]),
                date=str(row["date"]),
                session_number=int(row["session_number"]),
                probe_names=probe_names,
            )
        )
    return sessions
```

```python
@property
def session_path(self) -> Path:
    return DATA_ROOT / self.lab / "Subjects" / self.subject / self.date / f"{self.session_number:03d}"
```

iii. The AI noted that `SessionLoader` and `SpikeSortingLoader` could not be used due to import/dependency issues in its environment, so it implemented direct ALF file loading that accesses the same underlying data files.

## 1-b. How are the data split into subjects?

i. Subject names come from the `subject` column of `bwm_release.csv`. Sessions are grouped by subject. At assembly, a sorted unique subject list is created and each session gets a `subject_idx`.

ii.
```python
subject_names = sorted({ps.spec.subject for ps in prepared_sessions})
subject_to_idx = {subject: i for i, subject in enumerate(subject_names)}
```

iii. The subject identity is directly available in the release CSV metadata.

## 1-c. How are the data split into sessions?

i. Each unique `eid` in the release CSV corresponds to one session. Sessions are the natural unit of the release.

ii.
```python
grouped = bwm.groupby("eid", sort=False)
```

iii. No additional splitting is needed; the release CSV already lists sessions individually.

## 1-d. How are the data split into trials?

i. The trials table (one parquet file per session) has one row per trial. Trials are loaded from `_ibl_trials.table.pqt`.

ii.
```python
def load_trials_table(session_path: Path) -> pd.DataFrame:
    path = resolve_latest(session_path / "alf", "**/_ibl_trials.table.pqt")
    return pd.read_parquet(path)
```

iii. No decision to make; the trials table is already one row per trial.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered based on multiple criteria: (1) reaction time between 0.08 and 2.0 s, (2) non-null values for `stimOn_times`, `choice`, `feedback_times`, `probabilityLeft`, `firstMovement_times`, `feedbackType`, (3) no-choice trials excluded (`choice != 0`), (4) trial duration (`feedback_times - goCue_times`) <= 10.0 s, (5) wheel and whisker coverage must span the trial window.

ii.
```python
def compute_trial_mask(trials: pd.DataFrame) -> np.ndarray:
    rt = trials["firstMovement_times"] - trials["stimOn_times"]
    mask = (
        ~trials["stimOn_times"].isnull()
        & ~trials["choice"].isnull()
        & ~trials["feedback_times"].isnull()
        & ~trials["probabilityLeft"].isnull()
        & ~trials["firstMovement_times"].isnull()
        & ~trials["feedbackType"].isnull()
        & (rt >= 0.08)
        & (rt <= 2.0)
        & ((trials["feedback_times"] - trials["goCue_times"]) <= 10.0)
        & (trials["choice"] != 0)
    )
    return mask.to_numpy(dtype=bool)
```

Additional behavior coverage filtering:
```python
keep_mask = trial_mask & wheel_mask & whisker_mask
```

iii. The AI documented this as matching the reference code's `load_trials_and_mask` function, including the `max_trial_len=10.0` criterion and the behavior coverage checks.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Two arrays per probe: `spikes.times.npy` and `spikes.clusters.npy`, plus `clusters.metrics.pqt` for QC labels and `clusters.channels.npy` / `channels.brainLocationIds_ccf_2017.npy` for region assignment.

ii.
```python
spikes_times = np.load(spikes_times_path, mmap_mode="r")
spikes_clusters = np.load(spikes_clusters_path, mmap_mode="r")
```

iii. The AI loads the same spike sorting outputs as the reference, just via direct file access rather than through the ONE API.

## 2-b. How is the `neural` data processed?

i. Spikes are counted into 20 ms bins over the trial window [-0.5, 1.5] s, giving 100 bins per trial. When a session has multiple probes, units are merged with continuous renumbering. The spike counts are stored as **float16** and are NOT divided by the bin width (i.e., they are raw counts, not firing rates in Hz).

ii.
```python
rel = spike_times[i0:i1] - interval_begs[i]
bin_idx = np.floor(rel / binsize).astype(np.int64)
valid = (bin_idx >= 0) & (bin_idx < n_bins)
flat = spike_clusters[i0:i1][valid] * n_bins + bin_idx[valid]
counts = np.bincount(flat, minlength=n_units * n_bins).reshape(n_units, n_bins)
out.append(counts.astype(np.float16))
```

iii. The AI chose float16 for memory efficiency but did not convert counts to firing rates. The CONVERSION_NOTES describe this as "spike counts per bin" without mentioning a division by bin width.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only clusters with `label >= 1` in `clusters.metrics.pqt` are kept. The AI does NOT explicitly filter out `void` regions (clusters outside the brain). It maps cluster locations to Beryl acronyms via `brain_regions.id2acronym(region_ids, mapping="Beryl")` but does not exclude any regions based on the resulting acronym.

ii.
```python
metrics = pd.read_parquet(metrics_path, columns=["label"])
good_rows = metrics["label"].to_numpy(copy=False) >= 1
```

```python
cluster_regions = brain_regions.id2acronym(region_ids, mapping="Beryl").astype(str)
```

iii. The AI verified that `label >= 1` across the 699 frozen-release insertions yields exactly 75,708 units, matching the paper. It did not discuss void filtering.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to stimulus onset (`stimOn_times`). For each trial, spikes are binned relative to `stimOn_times + TIME_WINDOW[0]` (-0.5 s before stimulus onset) over a 2 s window.

ii.
```python
interval_begs = align_times + time_window[0]
interval_ends = align_times + time_window[1]
...
rel = spike_times[i0:i1] - interval_begs[i]
bin_idx = np.floor(rel / binsize).astype(np.int64)
```

iii. The alignment event is stimulus onset as specified in the instructions and matching the reference code.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The bin size is 20 ms, yielding 100 bins over the 2 s window. No rebinning or interpolation is applied to the neural data.

ii.
```python
BINSIZE = 0.02
NBINS = int(round((TIME_WINDOW[1] - TIME_WINDOW[0]) / BINSIZE))  # 100
```

iii. This matches the reference code's `binsize=0.02` and the papers' description of "20-ms bins, producing T = 100 time steps."

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. Derived from `stimOn_times` in the trials table, which defines the alignment event. The time input is a fixed vector of relative times computed as `np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], NBINS)` = [-0.48, -0.46, ..., 1.48, 1.50].

ii.
```python
COMMON_RELATIVE_TIMES = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], NBINS, dtype=np.float32)
```

iii. The time grid represents the right edges of the bins, following the reference code's `get_behavior_per_interval` interpolation grid.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. No per-trial processing; the same fixed vector of 100 relative time values is used for every trial.

ii.
```python
COMMON_RELATIVE_TIMES = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], NBINS, dtype=np.float32)
```

iii. The time input is defined by the binning grid itself.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The time input values represent the right edges of the same bins used for neural spike counting. The neural bins cover [T_START + k*BIN, T_START + (k+1)*BIN) for k=0..99, and the time input at index k is T_START + (k+1)*BIN.

ii.
```python
COMMON_RELATIVE_TIMES = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], NBINS, dtype=np.float32)
# Gives [-0.48, -0.46, ..., 1.50]
```

iii. The AI followed the reference code's `get_behavior_per_interval` which uses `np.linspace(interval_beg + binsize, interval_end, n_bins)` for behavior interpolation.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. Derived from `probabilityLeft` in the trials table. A block boundary is detected whenever `probabilityLeft` changes value.

ii.
```python
def compute_trial_number_in_block(prob_left: np.ndarray) -> np.ndarray:
    ...
    for i, val in enumerate(prob_left):
        current = None if pd.isna(val) else float(val)
        if i == 0 or current != prev:
            counter = 1
        else:
            counter += 1
        out[i] = counter
        prev = current
    return out
```

iii. The trial table has no explicit block identifier, so blocks are recovered from changes in `probabilityLeft`.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The trial's position within its block is counted starting from 1 (not 0). The count is computed on the original unfiltered trial table before quality filtering, so dropped trials still advance the counter. The value is broadcast across all 100 time bins for each trial.

ii.
```python
if i == 0 or current != prev:
    counter = 1
else:
    counter += 1
out[i] = counter
```

```python
np.full(NBINS, prepared.trial_number_in_block[trial_idx], dtype=np.float32)
```

iii. The AI noted that the count is taken before filtering so the number reflects the animal's real position in the block.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. From the `choice` column in the trials table, which takes values +1 (left), -1 (right), or 0 (no response).

ii.
```python
def map_choice(raw_choice: np.ndarray) -> np.ndarray:
    mapped = np.empty(raw_choice.shape[0], dtype=np.int8)
    mapped[raw_choice == 1] = 0
    mapped[raw_choice == -1] = 1
    ...
    return mapped
```

iii. No-response trials (choice=0) are excluded by the trial mask.

## 5-b. What processing is involved in computing `output` *Choice*?

i. Left choice (+1) is mapped to 0, right choice (-1) is mapped to 1. The scalar value is broadcast across all 100 time bins.

ii.
```python
np.full(NBINS, choice[trial_idx], dtype=np.int8)
```

iii. Matches the instruction specification: left = 0, right = 1.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. From `probabilityLeft` in the trials table, which takes values 0.2, 0.5, and 0.8.

ii.
```python
def map_prior(raw_prior: np.ndarray) -> np.ndarray:
    ...
    mapper = {0.2: 0, 0.5: 1, 0.8: 2}
    ...
```

iii. The three values correspond to the block structure of the task.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The values are mapped: 0.2 -> 0, 0.5 -> 1, 0.8 -> 2. The scalar is broadcast across all 100 time bins.

ii.
```python
np.full(NBINS, prior[trial_idx], dtype=np.int8)
```

iii. Matches the instruction specification.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. From `_ibl_wheel.timestamps.npy` and `_ibl_wheel.position.npy`. The position is interpolated to 1000 Hz and differentiated with a 20 Hz Butterworth lowpass filter, then the absolute value gives wheel speed.

ii.
```python
def load_wheel_speed(session_path: Path) -> tuple[np.ndarray, np.ndarray]:
    ...
    interp_pos, interp_times = interpolate_position(timestamps, position, freq=1000)
    velocity, _ = velocity_filtered(interp_pos, fs=1000, corner_frequency=20, order=8)
    return interp_times.astype(np.float64), np.abs(velocity).astype(np.float32)
```

iii. The AI imported and used the bundled `brainbox.behavior.wheel` functions (`interpolate_position`, `velocity_filtered`) which are the same functions `SessionLoader` uses internally.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Three steps: (1) Wheel position is interpolated to 1000 Hz and filtered to get velocity, speed = |velocity|. (2) The speed trace is linearly interpolated onto the trial time grid (right edges). (3) It is discretized into 3 bins using **global** tertile thresholds computed across all kept timepoints from all sessions.

ii.
```python
all_wheel = np.concatenate([ps.wheel_cont.reshape(-1) for ps in prepared_sessions]).astype(np.float32)
wheel_edges = robust_tertile_edges(all_wheel)
```

```python
def robust_tertile_edges(values: np.ndarray) -> tuple[float, float]:
    ...
    q1, q2 = np.quantile(finite, [1 / 3, 2 / 3])
    ...
```

```python
def digitize_three_bins(values: np.ndarray, edges: tuple[float, float]) -> np.ndarray:
    ...
    return np.digitize(values, bins=np.array([low, high], dtype=np.float32), right=False).astype(np.int8)
```

iii. The AI chose global thresholds so that categories are consistent across sessions, noting that session-specific thresholds would make class labels inconsistent.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Global tertile edges are computed from all kept wheel speed values across all sessions (1/3 and 2/3 quantiles). Values are digitized into 3 bins: 0 (low), 1 (medium), 2 (high).

ii.
```python
def robust_tertile_edges(values: np.ndarray) -> tuple[float, float]:
    ...
    q1, q2 = np.quantile(finite, [1 / 3, 2 / 3])
    ...
```

iii. The AI explicitly chose global thresholds for cross-session consistency.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The wheel speed trace is linearly interpolated onto the same time grid as the neural data (right bin edges at `np.linspace(t_beg + binsize, t_end, n_bins)`).

ii.
```python
x_interp = np.linspace(t_beg + binsize, t_end, n_bins)
y_interp = np.interp(x_interp, curr_times, curr_vals)
```

iii. The wheel is on the same session clock as the spikes, so interpolating at the bin times provides alignment.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. From `<side>Camera.ROIMotionEnergy.npy` with timestamps `_ibl_<side>Camera.times.npy`. Left camera is preferred; right is used as fallback.

ii.
```python
def load_whisker_motion_energy(session_path: Path) -> tuple[np.ndarray, np.ndarray, str]:
    ...
    for view in ("left", "right"):
        me_candidates = sorted(alf_path.glob(f"**/{view}Camera.ROIMotionEnergy.npy"))
        ts_candidates = sorted(alf_path.glob(f"**/_ibl_{view}Camera.times.npy"))
        ...
```

iii. This follows the reference code's preference for left camera.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The motion energy trace is used as-is (no filtering or normalization). It is linearly interpolated onto the trial time grid and discretized into 3 bins using global tertile thresholds, same as wheel speed.

ii.
```python
whisker_interp, whisker_mask = interpolate_behavior_per_trial(whisker_times, whisker_motion, align_times)
...
all_whisker = np.concatenate([ps.whisker_cont.reshape(-1) for ps in prepared_sessions]).astype(np.float32)
whisker_edges = robust_tertile_edges(all_whisker)
```

iii. No additional processing beyond interpolation and discretization.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Same method as wheel speed: global tertile edges from all kept whisker motion energy values across all sessions.

ii.
```python
whisker_edges = robust_tertile_edges(all_whisker)
whisker_bins = digitize_three_bins(prepared.whisker_cont, whisker_edges)
```

iii. Global thresholds for cross-session consistency.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Same method as wheel speed: linearly interpolated onto the same time grid (right bin edges).

ii.
```python
x_interp = np.linspace(t_beg + binsize, t_end, n_bins)
y_interp = np.interp(x_interp, curr_times, curr_vals)
```

iii. Camera timestamps are on the same session clock as neural data.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Several strategies: (1) Sessions missing required data streams (whisker motion energy) are excluded (20 sessions). (2) Sessions with no good units or fewer than 2 valid trials are excluded. (3) Camera timestamps longer than motion energy arrays are trimmed from the front. (4) Trials without full behavior coverage are excluded via the behavior interpolation mask. (5) NaN values in behavior streams cause trial exclusion.

ii.
```python
def check_video_timestamps(view, video_timestamps, video_data):
    ...
    if video_timestamps.shape[0] > video_data.shape[0]:
        video_timestamps = video_timestamps[-video_data.shape[0]:]
    ...
```

```python
if np.isnan(curr_vals).any():
    continue
```

iii. The AI documented 21 excluded sessions (20 missing whisker streams, 1 with insufficient data) and preserved zero-spike trials as legitimate data.

## 10-a. What are the most time-consuming steps of the code?

i. Loading spike sorting data from disk (memory-mapped reads of large spike arrays) was identified as the primary bottleneck.

ii.
```python
spikes_times = np.load(spikes_times_path, mmap_mode="r")
spikes_clusters = np.load(spikes_clusters_path, mmap_mode="r")
```

iii. The AI profiled and optimized this, switching to memory-mapped loading and increasing parallelism.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Two per-trial loops: (1) the spike binning loop in `bin_spikes_by_trial` that processes each trial sequentially, and (2) the behavior interpolation loop in `interpolate_behavior_per_trial`. Both iterate over trials one at a time.

ii.
```python
for i in range(len(align_times)):
    ...
    rel = spike_times[i0:i1] - interval_begs[i]
    bin_idx = np.floor(rel / binsize).astype(np.int64)
    ...
```

```python
for i in range(len(align_times)):
    ...
    y_interp = np.interp(x_interp, curr_times, curr_vals)
    ...
```

iii. These could be vectorized by offsetting spike indices by trial, but the AI kept the loops for clarity since they were not the dominant bottleneck.

## 10-c. What processing does the code repeat multiple times?

i. The code uses a two-pass architecture: pass 1 loads trials, wheel, and whisker data to compute behavior and global discretization thresholds; pass 2 reloads spike data for neural binning. The trial table, wheel, and whisker data are effectively loaded once in pass 1 and cached in `PreparedSession` objects. However, `BrainRegions()` is instantiated once per session in `build_session_payload`.

ii.
```python
def build_session_payload(...):
    ...
    brain_regions = BrainRegions()  # instantiated per session
    ...
```

iii. The two-pass design is intentional: behavior data must be loaded first to compute global discretization thresholds before neural data is binned and outputs are assigned.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The `order=8` Butterworth filter parameter in `velocity_filtered` is harder than necessary (the default is 3). The code also computes `acceleration` from `velocity_filtered` which is never used. The `robust_tertile_edges` function has a complex fallback for degenerate distributions that is unlikely to be exercised.

ii.
```python
velocity, _ = velocity_filtered(interp_pos, fs=1000, corner_frequency=20, order=8)
```

iii. These are minor inefficiencies that don't significantly affect runtime or correctness.
