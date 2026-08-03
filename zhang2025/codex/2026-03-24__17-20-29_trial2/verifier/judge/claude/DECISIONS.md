# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads a CSV file (`code/code_zhang2025/data/bwm_release.csv`) to enumerate the 459 release sessions and their probe insertions. It then constructs file paths directly from session metadata (lab, subject, date, session number) to locate ALF-format files on disk under `data/one_cache/`. It does NOT use the ONE API or `SessionLoader`/`SpikeSortingLoader`; instead it reads `.npy` and `.pqt` files directly. This is because the AI found that `SessionLoader` was broken in its environment due to missing dependencies.

ii.
```python
RELEASE_CSV = Path("code/code_zhang2025/data/bwm_release.csv")
DATA_ROOT = Path("data/one_cache")

def load_release_sessions() -> list[SessionSpec]:
    bwm = pd.read_csv(RELEASE_CSV, index_col=0)
    grouped = bwm.groupby("eid", sort=False)
    sessions: list[SessionSpec] = []
    for eid, df in grouped:
        row = df.iloc[0]
        probe_names = tuple(df["probe_name"].tolist())
        sessions.append(SessionSpec(eid=eid, subject=str(row["subject"]), ...))
    return sessions
```

```python
@property
def session_path(self) -> Path:
    return DATA_ROOT / self.lab / "Subjects" / self.subject / self.date / f"{self.session_number:03d}"
```

iii. The AI noted that direct use of `brainbox.io.one.SessionLoader` was not viable because it imports an unavailable `neuropixel` package and `ONE.load_object(...)` hits `.rest` permission issues. Using `bwm_release.csv` was chosen because it matches both the data paper and the reference code's session universe of 459 sessions.

## 1-b. How are the data split into subjects?

i. Subjects are identified from the `subject` column in `bwm_release.csv`. The AI groups sessions by subject and builds a sorted unique subject list. `subject_idx` maps each session to its index in this sorted list.

ii.
```python
subject_names = sorted({ps.spec.subject for ps in prepared_sessions})
subject_to_idx = {subject: i for i, subject in enumerate(subject_names)}
```

iii. The subject name comes directly from the release CSV metadata, consistent with the reference approach of deriving subjects from session metadata.

## 1-c. How are the data split into sessions?

i. Each row (grouped by `eid`) in `bwm_release.csv` corresponds to one session. The AI groups the CSV by `eid` and creates one `SessionSpec` per unique session.

ii.
```python
grouped = bwm.groupby("eid", sort=False)
for eid, df in grouped:
    sessions.append(SessionSpec(eid=eid, ...))
```

iii. Sessions are already the unit of organization in the release CSV; no splitting is needed.

## 1-d. How are the data split into trials?

i. The trials table (`_ibl_trials.table.pqt`) has one row per trial; this defines the trial split.

ii.
```python
def load_trials_table(session_path: Path) -> pd.DataFrame:
    path = resolve_latest(session_path / "alf", "**/_ibl_trials.table.pqt")
    return pd.read_parquet(path)
```

iii. No decision to make; trials are already one row per trial in the data.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies several filters: (1) non-null `stimOn_times`, `choice`, `feedback_times`, `probabilityLeft`, `firstMovement_times`, `feedbackType`; (2) reaction time between 0.08 and 2.0 s; (3) trial duration (`feedback_times - goCue_times`) <= 10 s; (4) `choice != 0` (exclude no-choice); (5) wheel and whisker coverage of the trial window. This is more filters than the reference, which does not check `feedbackType` non-null or `max_trial_len`.

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

iii. The AI derived its trial mask from the reference code's `load_trials_and_mask` function, which includes the `max_trial_len=10.0` and `feedbackType` non-null checks. The AI documented this as "keeping `exclude_nochoice=True` and `max_trial_len=10.0` to remain consistent with the provided code path."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Two arrays per probe: `spikes.times.npy` and `spikes.clusters.npy`. The cluster quality label comes from `clusters.metrics.pqt` and brain regions from `clusters.channels.npy` and `channels.brainLocationIds_ccf_2017.npy`.

ii.
```python
spikes_times = np.load(spikes_times_path, mmap_mode="r")
spikes_clusters = np.load(spikes_clusters_path, mmap_mode="r")
```

iii. These are the standard IBL spike sorting outputs, matching the reference.

## 2-b. How is the `neural` data processed?

i. Spikes are counted into 20 ms bins over the [-0.5, 1.5] s trial window relative to stimulus onset. However, the AI stores **raw spike counts** (as float16), NOT firing rates. The reference divides by BIN (0.02) to convert to Hz. The AI also uses float16 precision rather than float32.

ii.
```python
flat = spike_clusters[i0:i1][valid] * n_bins + bin_idx[valid]
counts = np.bincount(flat, minlength=n_units * n_bins).reshape(n_units, n_bins)
out.append(counts.astype(np.float16))
```

iii. The AI's CONVERSION_NOTES mention "no firing-rate smoothing; data are spike counts per bin" from the reference code analysis, but the AI appears to have missed that the reference actually converts counts to rates by dividing by the bin width.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only clusters with `label >= 1` in `clusters.metrics.pqt` are kept. The AI verified this yields exactly 75,708 units across the full 459-session release, matching the data paper.

ii.
```python
metrics = pd.read_parquet(metrics_path, columns=["label"])
good_rows = metrics["label"].to_numpy(copy=False) >= 1
```

iii. This matches the data paper's definition of "well-isolated neurons" and reproduces the reported count of 75,708.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to stimulus onset (`stimOn_times`). Spike times are binned relative to the start of the trial window (`align_time + T_START`), which puts zero at stimulus onset time.

ii.
```python
ALIGN_EVENT = "stimOn_times"
...
interval_begs = align_times + time_window[0]  # align_time - 0.5
...
rel = spike_times[i0:i1] - interval_begs[i]
bin_idx = np.floor(rel / binsize).astype(np.int64)
```

iii. The AI noted the user explicitly requires "Temporally align based on stimulus onset."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 20 ms bins, 100 bins per trial over the 2 s window. No rebinning is applied.

ii.
```python
BINSIZE = 0.02
NBINS = int(round((TIME_WINDOW[1] - TIME_WINDOW[0]) / BINSIZE))  # 100
```

iii. Matches the reference code's `binsize=0.02` and the method paper's description of "20-ms bins, producing T = 100 time steps."

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. From `stimOn_times` in the trials table, which defines the alignment event. The time input is a fixed vector of time points relative to stimulus onset.

ii.
```python
COMMON_RELATIVE_TIMES = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], NBINS, dtype=np.float32)
# = np.linspace(-0.48, 1.5, 100)
```

iii. The AI uses the same time grid construction as the reference code's `get_behavior_per_interval` function.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The time vector is computed as `np.linspace(-0.48, 1.5, 100)`, which produces right bin edges rather than bin centers. The reference uses bin centers: `EDGES[:-1] + BIN/2 = [-0.49, -0.47, ..., 1.49]`.

ii.
```python
COMMON_RELATIVE_TIMES = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], NBINS, dtype=np.float32)
```

Reference uses:
```python
TIME = EDGES[:-1] + BIN / 2  # bin centres
```

iii. The AI followed the reference code's `get_behavior_per_interval` which uses `np.linspace(interval_beg + binsize, interval_end, n_bins)` for interpolation points. This is slightly shifted from the bin centers used by the reference human solution.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The same `COMMON_RELATIVE_TIMES` vector is used as both the time input and the interpolation grid for behavioral signals. However, neural spike binning uses bins defined by `np.floor(rel / binsize)`, which defines bins from their left edges. The time input values are the right edges of these bins, not their centers.

ii.
```python
# Neural binning: floor division gives bin index from left edge
bin_idx = np.floor(rel / binsize).astype(np.int64)

# Time input: right edges
COMMON_RELATIVE_TIMES = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], NBINS)
```

iii. There is a subtle misalignment: the time input represents right bin edges while the neural bins are defined by left edges. The bin center would be more appropriate. However, this is consistent with how the reference code's `get_behavior_per_interval` defines its interpolation points.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. From `probabilityLeft` in the trials table. Block boundaries are detected where `probabilityLeft` changes value.

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

iii. The blocks are recovered from changes in `probabilityLeft`, consistent with the reference approach.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The AI counts trials within each block starting from **1**, whereas the reference starts from **0** (using `pandas.groupby.cumcount()`). The count is computed on the unfiltered trial table before trial quality filtering, so excluded trials still advance the counter.

ii.
```python
# AI: starts at 1
counter = 1
...
counter += 1

# Reference: starts at 0
trial_in_block = trials.groupby(block).cumcount()
```

iii. The AI's CONVERSION_NOTES do not explicitly discuss the choice to start from 1 vs 0. The reference uses `cumcount()` which produces 0-indexed values.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. From the `choice` column in the trials table, which takes values +1 (left), -1 (right), and 0 (no response).

ii.
```python
choice_raw = trials.loc[keep_mask, "choice"].to_numpy()
```

iii. Standard IBL choice encoding.

## 5-b. What processing is involved in computing `output` *Choice*?

i. Choice is mapped: left (+1) -> 0, right (-1) -> 1. No-choice trials (0) are excluded by the trial mask.

ii.
```python
def map_choice(raw_choice: np.ndarray) -> np.ndarray:
    mapped = np.empty(raw_choice.shape[0], dtype=np.int8)
    mapped[raw_choice == 1] = 0
    mapped[raw_choice == -1] = 1
    return mapped
```

iii. Matches the instruction specification: "left = 0, right = 1."

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. From `probabilityLeft` in the trials table, which takes values 0.2, 0.5, and 0.8.

ii.
```python
prior_raw = trials.loc[keep_mask, "probabilityLeft"].to_numpy()
```

iii. Standard IBL block prior variable.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Mapped to categorical: 0.2 -> 0, 0.5 -> 1, 0.8 -> 2. Values are rounded to 1 decimal place before mapping.

ii.
```python
def map_prior(raw_prior: np.ndarray) -> np.ndarray:
    mapper = {0.2: 0, 0.5: 1, 0.8: 2}
    for i, val in enumerate(raw_prior):
        key = round(float(val), 1)
        out[i] = mapper[key]
    return out
```

iii. Matches the instruction specification: "0.2 -> 0, 0.5 -> 1, 0.8 -> 2."

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. From `_ibl_wheel.timestamps.npy` and `_ibl_wheel.position.npy`. The position is interpolated to 1000 Hz and differentiated with a 20 Hz Butterworth low-pass filter to get velocity; speed is the absolute value.

ii.
```python
def load_wheel_speed(session_path: Path) -> tuple[np.ndarray, np.ndarray]:
    timestamps = np.load(resolve_latest(alf_path, "**/_ibl_wheel.timestamps.npy"))
    position = np.load(resolve_latest(alf_path, "**/_ibl_wheel.position.npy"))
    interp_pos, interp_times = interpolate_position(timestamps, position, freq=1000)
    velocity, _ = velocity_filtered(interp_pos, fs=1000, corner_frequency=20, order=8)
    return interp_times.astype(np.float64), np.abs(velocity).astype(np.float32)
```

iii. The AI directly imports and calls the same `interpolate_position` and `velocity_filtered` functions from `brainbox.behavior.wheel`, matching the reference processing. The filter order is 8 (matching the default in `velocity_filtered`).

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The continuous wheel speed trace is interpolated onto the trial-aligned time grid using `np.interp`. It is then discretized into 3 bins. However, the AI uses **global** tertile edges (computed across all sessions' kept timepoints) rather than **per-session** percentiles as the reference does.

ii.
```python
# Global edges computed across all sessions
all_wheel = np.concatenate([ps.wheel_cont.reshape(-1) for ps in prepared_sessions]).astype(np.float32)
wheel_edges = robust_tertile_edges(all_wheel)
```

```python
def digitize_three_bins(values, edges):
    return np.digitize(values, bins=np.array([low, high]), right=False).astype(np.int8)
```

iii. The AI's CONVERSION_NOTES state: "session-specific thresholds would make class labels inconsistent across sessions; global thresholds keep categories comparable." This is a deliberate design choice that differs from the reference's per-session approach.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Global tertile edges are computed using `np.quantile` at [1/3, 2/3] across all kept timepoints from all sessions. Values are then digitized into 3 bins (0, 1, 2) using `np.digitize`. The AI also includes a `robust_tertile_edges` fallback for degenerate cases.

ii.
```python
def robust_tertile_edges(values):
    q1, q2 = np.quantile(finite, [1 / 3, 2 / 3])
    ...
    return float(q1), float(q2)

def digitize_three_bins(values, edges):
    return np.digitize(values, bins=np.array([low, high]), right=False).astype(np.int8)
```

iii. The reference uses per-session percentiles at [100/3, 200/3] which guarantees roughly equal-sized classes within each session. The AI's global approach does not guarantee equal class sizes per session.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The wheel speed is interpolated to the same `COMMON_RELATIVE_TIMES` grid as the neural data, using `np.interp` within each trial window.

ii.
```python
x_interp = np.linspace(t_beg + binsize, t_end, n_bins)
y_interp = np.interp(x_interp, curr_times, curr_vals)
```

iii. The interpolation uses the same time grid as all other signals, ensuring alignment. However, as noted in 3-b, these are right bin edges, not bin centers.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. From `leftCamera.ROIMotionEnergy.npy` (preferred) or `rightCamera.ROIMotionEnergy.npy`, along with corresponding `_ibl_<side>Camera.times.npy` timestamps.

ii.
```python
def load_whisker_motion_energy(session_path):
    for view in ("left", "right"):
        me_candidates = sorted(alf_path.glob(f"**/{view}Camera.ROIMotionEnergy.npy"))
        ts_candidates = sorted(alf_path.glob(f"**/_ibl_{view}Camera.times.npy"))
        ...
```

iii. Left camera preferred, falling back to right. Matches the reference approach.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The raw motion energy trace is used as-is (no filtering or normalization). It is interpolated onto the trial-aligned time grid and then discretized into 3 bins using global tertile edges, same approach as wheel speed. Camera timestamps longer than motion energy arrays are trimmed from the front.

ii.
```python
def check_video_timestamps(view, video_timestamps, video_data):
    if video_timestamps.shape[0] > video_data.shape[0]:
        video_timestamps = video_timestamps[-video_data.shape[0]:]
    return video_timestamps, video_data
```

iii. The timestamp trimming from the front matches the IBL convention for camera timestamp alignment.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Same as wheel speed: global tertile edges computed across all sessions using `np.quantile` at [1/3, 2/3], then `np.digitize` into 3 bins.

ii.
```python
all_whisker = np.concatenate([ps.whisker_cont.reshape(-1) for ps in prepared_sessions]).astype(np.float32)
whisker_edges = robust_tertile_edges(all_whisker)
whisker_bins = digitize_three_bins(prepared.whisker_cont, whisker_edges)
```

iii. Same rationale as wheel speed - global thresholds for cross-session consistency. Differs from reference's per-session approach.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Same as wheel speed: interpolated to the `COMMON_RELATIVE_TIMES` grid using `np.interp` within each trial window.

ii.
```python
x_interp = np.linspace(t_beg + binsize, t_end, n_bins)
y_interp = np.interp(x_interp, curr_times, curr_vals)
```

iii. Shared time grid ensures alignment across all signals.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Several strategies: (1) Sessions missing required streams (whisker motion energy) are excluded entirely (20 sessions). (2) Trials with NaN in key fields are excluded by the trial mask. (3) Camera timestamps longer than motion energy arrays are trimmed from the front. (4) Sessions with no good units or fewer than 2 valid trials are excluded. (5) Invalid spike cluster indices (beyond cluster count) are handled defensively. (6) Zero-spike trials are preserved as all-zero matrices.

ii.
```python
# Camera timestamp repair
if video_timestamps.shape[0] > video_data.shape[0]:
    video_timestamps = video_timestamps[-video_data.shape[0]:]

# Invalid cluster indices
valid_spikes = spikes_clusters < n_clusters
if np.all(valid_spikes):
    spike_mask = good_rows[spikes_clusters]
else:
    spike_mask = np.zeros(spikes_clusters.shape[0], dtype=bool)
    spike_mask[valid_spikes] = good_rows[spikes_clusters[valid_spikes]]
```

iii. The AI documented investigating zero-spike trials and confirmed they represent true zero-spike windows in the raw data rather than conversion bugs.

## 10-a. What are the most time-consuming steps of the code?

i. The AI identified two main bottlenecks: (1) Loading spike sorting data from disk (memory-mapped reads of large spike arrays), and (2) the trial-by-trial spike binning loop. The AI documented that pass-2 spike loading was initially the dominant cost and was optimized with memory-mapped reads and no-copy dtype handling.

ii.
```python
spikes_times = np.load(spikes_times_path, mmap_mode="r")
spikes_clusters = np.load(spikes_clusters_path, mmap_mode="r")
```

iii. The AI noted the full conversion took about 10 minutes after optimization (pass 1: 144s, pass 2 build: 456s).

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Two per-trial loops: (1) `bin_spikes_by_trial` iterates over trials to bin spikes, and (2) `interpolate_behavior_per_trial` iterates over trials to interpolate behavioral signals. Both could potentially be vectorized. Additionally, `map_prior` uses a Python loop over trials that could be vectorized with numpy.

ii.
```python
# Per-trial spike binning loop
for i in range(len(align_times)):
    ...
    flat = spike_clusters[i0:i1][valid] * n_bins + bin_idx[valid]
    counts = np.bincount(flat, minlength=n_units * n_bins).reshape(n_units, n_bins)

# Per-trial behavior interpolation loop
for i in range(len(align_times)):
    ...
    y_interp = np.interp(x_interp, curr_times, curr_vals)

# Per-value prior mapping loop
for i, val in enumerate(raw_prior):
    key = round(float(val), 1)
    out[i] = mapper[key]
```

iii. The AI noted these loops but chose to keep them for clarity and because session-level parallelism provides sufficient throughput.

## 10-c. What processing does the code repeat multiple times?

i. The code uses a two-pass architecture: pass 1 loads trials, wheel, and whisker data to compute global tertile edges; pass 2 reloads spike data and rebuilds the final dataset. The behavioral data loaded in pass 1 is retained in memory for pass 2, but spike loading happens only in pass 2. The `BrainRegions()` object is instantiated once per session in `build_session_payload` rather than once globally.

ii.
```python
# BrainRegions instantiated per session in pass 2
def build_session_payload(prepared, wheel_edges, whisker_edges):
    brain_regions = BrainRegions()  # repeated per session
```

iii. The two-pass design is necessary because global tertile edges must be known before outputs can be constructed. The repeated `BrainRegions()` instantiation is minor overhead.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI adds a `max_trial_len` filter (`(feedback_times - goCue_times) <= 10.0`) and checks for non-null `feedbackType`, which the reference solution does not use. These extra filters may exclude trials that would otherwise be valid. The `robust_tertile_edges` fallback logic handles degenerate cases that likely never occur in practice.

ii.
```python
& ((trials["feedback_times"] - trials["goCue_times"]) <= 10.0)
& ~trials["feedbackType"].isnull()
```

iii. The AI adopted these from the reference code's `load_trials_and_mask` function, which includes them, but the human reference solution's simpler filter omits them.
