# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads session metadata from `code/code_zhang2025/data/bwm_release.csv`, which lists all 459 sessions with probe insertions. Sessions are grouped by `eid` (session ID). For each session, data is loaded directly from ALF (Analysis-Level Files) on disk under `data/one_cache/<lab>/Subjects/<subject>/<date>/<session_number>/alf/`. The AI bypassed the ONE API (`one.api.ONE`) and `SessionLoader` used in the reference code because these were not functional in the offline environment; instead it reads `.npy`, `.pqt`, and `.parquet` files directly.

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
        sessions.append(
            SessionSpec(
                eid=eid, subject=str(row["subject"]), lab=str(row["lab"]),
                date=str(row["date"]), session_number=int(row["session_number"]),
                probe_names=probe_names,
            )
        )
    return sessions
```

iii. The agent documented that it used the frozen BWM release CSV as the authoritative session list and bypassed the ONE API due to environment constraints, loading ALF files directly from disk. This is consistent with the reference code's use of `bwm_release.csv` as the session index.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified from the `subject` column in `bwm_release.csv`. A sorted list of unique subject names is built, and each session is mapped to its subject via `subject_to_idx`. The final data dictionary contains `subjects` (sorted list of unique names) and `subject_idx` (per-session index into that list).

ii.
```python
subject_names = sorted({ps.spec.subject for ps in prepared_sessions})
subject_to_idx = {subject: i for i, subject in enumerate(subject_names)}
subject_idx_list.append(subject_to_idx[prepared.spec.subject])
```

iii. The agent followed the target data format specification which requires `subjects` as a list of strings and `subject_idx` as a per-session index array.

## 1-c. How are the data split into sessions?

i. Each unique `eid` in the release CSV corresponds to one session. Sessions may have multiple probes (rows in the CSV), which are grouped together. Each session becomes one entry in the `neural`, `input`, and `output` lists.

ii.
```python
@dataclass(frozen=True)
class SessionSpec:
    eid: str
    subject: str
    lab: str
    date: str
    session_number: int
    probe_names: tuple[str, ...]
```
Sessions are grouped by eid: `grouped = bwm.groupby("eid", sort=False)`

iii. The agent documented that probe data is merged within sessions (consistent with the reference code's `merge_probes()` function), with each eid treated as one session.

## 1-d. How are the data split into trials?

i. Trials are loaded from `_ibl_trials.table.pqt` in the session's ALF directory. Each row in the parquet table is one trial. After applying quality filters and behavioral coverage masks, the kept trials for each session form the trial-level entries in the data lists.

ii.
```python
def load_trials_table(session_path: Path) -> pd.DataFrame:
    path = resolve_latest(session_path / "alf", "**/_ibl_trials.table.pqt")
    return pd.read_parquet(path)
```

iii. The agent loads the standard IBL trials table, which is the same data source used by the reference code via `SessionLoader.load_trials()`.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies a trial mask requiring: (1) non-null `stimOn_times`, `choice`, `feedback_times`, `probabilityLeft`, `firstMovement_times`, `feedbackType`; (2) reaction time (firstMovement - stimOn) between 0.08 and 2.0 s; (3) trial duration (feedback - goCue) <= 10.0 s; (4) choice != 0 (exclude no-go trials). Additionally, trials are excluded if wheel or whisker motion energy interpolation fails for that trial.

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
Then combined: `keep_mask = trial_mask & wheel_mask & whisker_mask`

iii. The agent documented that these filters match the reference code's `load_trials_and_mask()` function parameters: `min_rt=0.08`, `max_rt=2.0`, `max_trial_len=10.0`, `exclude_nochoice=True`, and the default `nan_exclude` list. The additional behavioral coverage masks mirror the reference code's `align_spike_behavior()` step.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from: `spikes.times.npy` (spike timestamps), `spikes.clusters.npy` (cluster IDs per spike), `clusters.metrics.pqt` (quality labels), `clusters.channels.npy` (channel assignments), and `channels.brainLocationIds_ccf_2017.npy` (brain region IDs). These are loaded from each probe's pykilosort directory.

ii.
```python
spikes_times_path = resolve_latest(probe_dir, "**/spikes.times.npy")
spikes_clusters_path = resolve_latest(probe_dir, "**/spikes.clusters.npy")
metrics_path = resolve_latest(probe_dir, "**/clusters.metrics.pqt")
clusters_channels_path = resolve_latest(probe_dir, "**/clusters.channels.npy")
channels_region_ids_path = resolve_latest(probe_dir, "**/channels.brainLocationIds_ccf_2017.npy")
```

iii. The agent documented that these are the standard IBL ALF spike-sorting outputs from Kilosort, matching the reference code's use of `SpikeSortingLoader.load_spike_sorting()`.

## 2-b. How is the `neural` data processed?

i. Spikes from multiple probes are merged into a single session-wide spike train (sorted by time). Spike counts are binned into 20ms non-overlapping bins within each trial's time window ([-0.5, 1.5]s relative to stimulus onset). The output is a (n_neurons, n_bins) matrix per trial stored as float16.

ii.
```python
def bin_spikes_by_trial(spike_times, spike_clusters, align_times, n_units, binsize=BINSIZE, time_window=TIME_WINDOW):
    interval_begs = align_times + time_window[0]
    interval_ends = align_times + time_window[1]
    n_bins = int(np.ceil((time_window[1] - time_window[0]) / binsize))
    ...
    rel = spike_times[i0:i1] - interval_begs[i]
    bin_idx = np.floor(rel / binsize).astype(np.int64)
    ...
    flat = spike_clusters[i0:i1][valid] * n_bins + bin_idx[valid]
    counts = np.bincount(flat, minlength=n_units * n_bins).reshape(n_units, n_bins)
    out.append(counts.astype(np.float16))
```

iii. The agent documented that spike binning follows the reference code's approach in `bin_spiking_data()`, using 20ms bins and the same time window.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only clusters with `label >= 1` in `clusters.metrics.pqt` are kept (well-isolated neurons). Spikes belonging to excluded clusters are discarded. This filtering is done upfront per-probe before merging.

ii.
```python
metrics = pd.read_parquet(metrics_path, columns=["label"])
good_rows = metrics["label"].to_numpy(copy=False) >= 1
if not np.any(good_rows):
    continue
...
spike_mask = good_rows[spikes_clusters]
```

iii. The agent documented that `label >= 1` reproduces the paper's definition of 75,708 well-isolated neurons and matches the reference code's QC criterion (`clusters['label'] >= 1`). The reference caching code loads all clusters but stores `good_clusters` metadata; the AI applies the filter upfront, which is a valid approach since only well-isolated neurons are used in the paper's analyses.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to stimulus onset (`stimOn_times`). For each trial, the time window is [stimOn - 0.5s, stimOn + 1.5s]. Spikes within this window are binned.

ii.
```python
ALIGN_EVENT = "stimOn_times"
TIME_WINDOW = (-0.5, 1.5)
...
kept_align_times = prepared.trials.loc[prepared.keep_mask, ALIGN_EVENT].to_numpy(dtype=np.float64)
neural_trials = bin_spikes_by_trial(
    spike_times=spike_times, spike_clusters=spike_clusters,
    align_times=kept_align_times, n_units=cluster_regions.shape[0],
)
```

iii. The instructions explicitly state "Temporally align based on stimulus onset". The reference code also uses `stimOn_times` with `time_window=(-0.5, 1.5)` in `0_data_caching.py`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is 20ms (0.02s), producing 100 bins per trial over the 2s window. No temporal rebinning is applied; the 20ms bin size is used directly for both spike counting and behavioral signal interpolation.

ii.
```python
BINSIZE = 0.02
NBINS = int(round((TIME_WINDOW[1] - TIME_WINDOW[0]) / BINSIZE))  # = 100
```

iii. The 20ms bin size matches the reference code parameter `binsize=0.02` in `0_data_caching.py`. The methods paper describes 20ms bins for dynamic behaviors (wheel speed, whisker ME), though it uses 50ms bins for prior decoding. The AI uses a uniform 20ms bin size for all variables, which is consistent with the task requirement of a single time grid.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is not derived from any raw data variable. It is a computed time vector based on the alignment parameters: `TIME_WINDOW = (-0.5, 1.5)` and `BINSIZE = 0.02`.

ii.
```python
COMMON_RELATIVE_TIMES = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], NBINS, dtype=np.float32)
```

iii. The agent uses `np.linspace(-0.5 + 0.02, 1.5, 100)` = `np.linspace(-0.48, 1.5, 100)`, producing 100 evenly-spaced time points from -0.48s to 1.50s. This represents the center/right-edge of each 20ms bin relative to stimulus onset.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The time vector is precomputed once as `COMMON_RELATIVE_TIMES` and reused identically for every trial. Each value represents the time of a bin relative to stimulus onset. The first bin is at -0.48s (center of the [-0.5, -0.48]s bin) and the last at 1.50s.

ii.
```python
inp = np.vstack([
    COMMON_RELATIVE_TIMES,
    np.full(NBINS, prepared.trial_number_in_block[trial_idx], dtype=np.float32),
]).astype(np.float32)
```

iii. This matches the reference code's approach where behavior bin times are computed as `np.linspace(interval_beg + binsize, interval_end, n_bins)`.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The same time grid (100 bins of 20ms from -0.5 to 1.5s) is used for both neural spike binning and the time-since-onset input. Since the time vector is defined relative to stimulus onset and neural activity is also binned relative to stimulus onset, they are inherently aligned.

ii. Both use `TIME_WINDOW = (-0.5, 1.5)` and `BINSIZE = 0.02`, producing the same 100-bin structure.

iii. No additional alignment step is needed because the same temporal parameters define both.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from the `probabilityLeft` column in the trials table (`_ibl_trials.table.pqt`). Block boundaries are detected by changes in `probabilityLeft`.

ii.
```python
def compute_trial_number_in_block(prob_left: np.ndarray) -> np.ndarray:
    out = np.zeros(len(prob_left), dtype=np.int16)
    prev = None
    counter = 0
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

iii. The agent documented that blocks are defined by the value of `probabilityLeft` (0.2, 0.5, or 0.8), with block boundaries occurring when this value changes. The counter resets to 1 at each boundary. This is consistent with the IBL task structure where blocks change between biased and unbiased conditions.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The code iterates through all trials sequentially, tracking the current `probabilityLeft` value. When the value changes (or at trial 0), the counter resets to 1. Otherwise it increments. The result is a per-trial integer that is then broadcast across all 100 time bins for that trial. The computation is done on ALL trials first, then indexed to kept trials only.

ii.
```python
trial_number_in_block = compute_trial_number_in_block(trials["probabilityLeft"].to_numpy())
# Later, indexed to kept trials:
trial_number_in_block=trial_number_in_block[keep_mask],
# In output construction:
np.full(NBINS, prepared.trial_number_in_block[trial_idx], dtype=np.float32),
```

iii. The agent computed trial_number_in_block on all trials before applying the trial mask, ensuring the counter reflects the true block position. The value is then broadcast as a constant across all time bins for each trial.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. Choice is derived from the `choice` column in the trials table. In IBL convention: 1 = left, -1 = right, 0 = no-go (excluded by trial filtering).

ii.
```python
choice_raw=trials.loc[keep_mask, "choice"].to_numpy(),
```

iii. The agent correctly identified `choice` from the trials table as the source variable.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The raw choice values are mapped to binary categories: IBL `1` (left) -> `0`, IBL `-1` (right) -> `1`. This is broadcast as a constant across all 100 time bins. The mapping matches the instructions: "left = 0, right = 1".

ii.
```python
def map_choice(raw_choice: np.ndarray) -> np.ndarray:
    mapped = np.empty(raw_choice.shape[0], dtype=np.int8)
    mapped[raw_choice == 1] = 0
    mapped[raw_choice == -1] = 1
    if not np.all(np.isin(raw_choice, [-1, 1])):
        raise ValueError("Unexpected choice values encountered after filtering.")
    return mapped
```

iii. The instructions specify "left = 0, right = 1". The agent correctly maps IBL choice convention to this encoding.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It is derived from the `probabilityLeft` column in the trials table, which takes values 0.2, 0.5, or 0.8.

ii.
```python
prior_raw=trials.loc[keep_mask, "probabilityLeft"].to_numpy(),
```

iii. The agent correctly identified `probabilityLeft` as the source for the prior.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The raw probability values are mapped to categorical indices: 0.2 -> 0, 0.5 -> 1, 0.8 -> 2. This value is broadcast as a constant across all 100 time bins per trial.

ii.
```python
def map_prior(raw_prior: np.ndarray) -> np.ndarray:
    out = np.empty(raw_prior.shape[0], dtype=np.int8)
    mapper = {0.2: 0, 0.5: 1, 0.8: 2}
    for i, val in enumerate(raw_prior):
        key = round(float(val), 1)
        if key not in mapper:
            raise ValueError(f"Unexpected probabilityLeft value {val}")
        out[i] = mapper[key]
    return out
```

iii. The instructions specify "0.2 -> 0, 0.5 -> 1, 0.8 -> 2". The agent implements this exactly.

## 9-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. Wheel speed is derived from `_ibl_wheel.timestamps.npy` (wheel encoder timestamps) and `_ibl_wheel.position.npy` (wheel position in radians). These are loaded from the session's ALF directory.

ii.
```python
def load_wheel_speed(session_path: Path) -> tuple[np.ndarray, np.ndarray]:
    alf_path = session_path / "alf"
    timestamps = np.load(resolve_latest(alf_path, "**/_ibl_wheel.timestamps.npy"))
    position = np.load(resolve_latest(alf_path, "**/_ibl_wheel.position.npy"))
```

iii. The agent documented these as the standard IBL wheel data files.

## 9-b. What processing is involved in computing `output` *Wheel speed*?

i. Raw wheel position is interpolated to a uniform 1000 Hz sampling rate using `brainbox.behavior.wheel.interpolate_position`. Velocity is computed using a Butterworth low-pass filter (`velocity_filtered` with corner_frequency=20Hz, order=8). The absolute value of velocity gives speed.

ii.
```python
interp_pos, interp_times = interpolate_position(timestamps, position, freq=1000)
velocity, _ = velocity_filtered(interp_pos, fs=1000, corner_frequency=20, order=8)
return interp_times.astype(np.float64), np.abs(velocity).astype(np.float32)
```

iii. The agent reused the `brainbox.behavior.wheel` library functions. The reference code uses `SessionLoader.load_wheel()` which internally calls these same functions. Both take the absolute value of velocity to get speed.

## 9-c. How is `output` *Wheel speed* thresholded into categories?

i. Wheel speed is discretized into 3 bins (low=0, medium=1, high=2) using global tertile thresholds. The thresholds are computed from the 1/3 and 2/3 quantiles of ALL wheel speed values across ALL kept sessions and time points. A robust fallback handles degenerate cases where quantile values are tied.

ii.
```python
all_wheel = np.concatenate([ps.wheel_cont.reshape(-1) for ps in prepared_sessions]).astype(np.float32)
wheel_edges = robust_tertile_edges(all_wheel)

def robust_tertile_edges(values: np.ndarray) -> tuple[float, float]:
    ...
    q1, q2 = np.quantile(finite, [1 / 3, 2 / 3])
    ...

def digitize_three_bins(values: np.ndarray, edges: tuple[float, float]) -> np.ndarray:
    return np.digitize(values, bins=np.array([low, high], dtype=np.float32), right=False).astype(np.int8)
```

iii. The task instructions require "Wheel speed discretized into 3 bins, time-varying". The agent chose global tertile thresholds to ensure consistent categories across sessions. This produces approximately equal-sized bins by design.

## 9-d. How is `output` *Wheel speed* aligned with the neural data?

i. For each trial, wheel speed is interpolated onto the same 100-bin time grid as the neural data ([-0.5, 1.5]s relative to stimulus onset, 20ms bins). The interpolation uses `np.interp` (linear interpolation) from the 1000Hz wheel speed signal to the 100 target time points.

ii.
```python
def interpolate_behavior_per_trial(target_times, target_vals, align_times, ...):
    interval_begs = align_times + time_window[0]
    interval_ends = align_times + time_window[1]
    ...
    x_interp = np.linspace(t_beg + binsize, t_end, n_bins)
    y_interp = np.interp(x_interp, curr_times, curr_vals)
    outputs[i] = y_interp.astype(np.float32)
```

iii. The agent aligns wheel speed to stimulus onset (same as neural data), matching the task instruction. The reference code also uses `np.linspace(interval_beg + binsize, interval_end, n_bins)` for interpolation grid, though it uses `scipy.interpolate.interp1d` with extrapolation rather than `np.interp`.

## 10-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. Whisker motion energy is derived from `leftCamera.ROIMotionEnergy.npy` (or `rightCamera.ROIMotionEnergy.npy` as fallback) and the corresponding camera timestamps `_ibl_leftCamera.times.npy`.

ii.
```python
def load_whisker_motion_energy(session_path: Path) -> tuple[np.ndarray, np.ndarray, str]:
    alf_path = session_path / "alf"
    for view in ("left", "right"):
        me_candidates = sorted(alf_path.glob(f"**/{view}Camera.ROIMotionEnergy.npy"))
        ts_candidates = sorted(alf_path.glob(f"**/_ibl_{view}Camera.times.npy"))
        ...
        motion_energy = np.load(me_candidates[-1])
        timestamps = np.load(ts_candidates[-1])
```

iii. The agent tries left camera first (matching the reference code which loads `left-whisker-motion-energy` first), falling back to right camera if unavailable.

## 10-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The raw motion energy values are loaded directly (no additional filtering or smoothing). Camera timestamps that are longer than the motion energy array are trimmed from the front to match lengths. The signal is then interpolated per-trial onto the 100-bin time grid.

ii.
```python
def check_video_timestamps(view, video_timestamps, video_data):
    if video_timestamps.shape[0] > video_data.shape[0]:
        video_timestamps = video_timestamps[-video_data.shape[0]:]
    return video_timestamps, video_data
```

iii. The agent documented that camera timestamps can be longer than data arrays, and trimming from the front is the correct approach per IBL data architecture conventions.

## 10-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Same approach as wheel speed: global tertile thresholds computed from the 1/3 and 2/3 quantiles of all whisker ME values across all kept sessions and time points. Produces 3 bins (low=0, medium=1, high=2).

ii.
```python
all_whisker = np.concatenate([ps.whisker_cont.reshape(-1) for ps in prepared_sessions]).astype(np.float32)
whisker_edges = robust_tertile_edges(all_whisker)
whisker_bins = digitize_three_bins(prepared.whisker_cont, whisker_edges)
```

iii. Same justification as wheel speed - global tertile thresholds for consistent cross-session categories.

## 10-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Identical approach to wheel speed: whisker motion energy is interpolated onto the same 100-bin stimulus-onset-aligned time grid using `interpolate_behavior_per_trial()`.

ii.
```python
whisker_interp, whisker_mask = interpolate_behavior_per_trial(whisker_times, whisker_motion, align_times)
```

iii. The alignment matches the neural data grid, consistent with the task instruction for stimulus-onset alignment.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Multiple mechanisms handle data issues:
- Sessions missing whisker ME streams are excluded entirely (20 sessions).
- Sessions with zero good units are skipped.
- Sessions with fewer than 2 valid trials after all filtering are skipped.
- Trials with NaN alignment times are skipped during behavior interpolation.
- Trials where behavior data has NaN values, gaps exceeding one bin width, or insufficient temporal coverage are excluded via the behavior masks.
- Camera timestamps longer than motion energy arrays are trimmed from the front.
- Trials with zero spikes in the neural window are preserved (not treated as errors).
- The `build_session_behavior_safe()` wrapper catches FileNotFoundError and other exceptions per session.

ii.
```python
def build_session_behavior_safe(spec):
    try:
        prepared = build_session_behavior(spec)
        if prepared is None:
            return spec.eid, None, "no_good_units_or_too_few_valid_trials"
        return spec.eid, prepared, None
    except FileNotFoundError:
        return spec.eid, None, "missing_required_stream"
    except Exception as exc:
        return spec.eid, None, f"{type(exc).__name__}: {exc}"
```

iii. The agent documented that 21 sessions were excluded (20 missing whisker ME, 1 with no good units/too few trials). The 16 zero-spike trial warnings were investigated and found to be legitimate raw-data edge cases, not conversion bugs.

## 12-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are: (1) neural spike loading and binning in `build_session_payload()` (pass 2), which involves loading large spike arrays from disk and computing per-trial binned counts for ~438 sessions; (2) the first pass (`build_session_behavior()`) which loads trials, wheel, and whisker data and performs interpolation for all sessions. The agent reported pass 1 took ~144s and pass 2 (build) took ~456s for the full dataset.

ii. The code uses `ThreadPoolExecutor` for parallelism:
```python
with ThreadPoolExecutor(max_workers=n_workers) as pool:
    future_to_index = {
        pool.submit(build_session_payload, prepared, wheel_edges, whisker_edges): idx
        for idx, prepared in enumerate(prepared_sessions)
    }
```

iii. The agent documented performance optimizations including memory-mapped spike loading, capping BLAS threads, and using up to 32 workers.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops could be vectorized:
- `compute_trial_number_in_block()`: iterates through all trials sequentially to count block position. Could use `np.diff` and `np.cumsum` on the `probabilityLeft` changes.
- `map_prior()`: iterates element-by-element with a dictionary lookup. Could use vectorized `np.searchsorted` or direct array indexing.
- `bin_spikes_by_trial()`: loops over trials. The inner binning per trial is vectorized, but the outer loop could potentially be batched.
- `interpolate_behavior_per_trial()`: loops over trials for interpolation.

ii.
```python
# compute_trial_number_in_block - sequential loop
for i, val in enumerate(prob_left):
    current = None if pd.isna(val) else float(val)
    if i == 0 or current != prev:
        counter = 1
    else:
        counter += 1
    out[i] = counter

# map_prior - sequential loop
for i, val in enumerate(raw_prior):
    key = round(float(val), 1)
    if key not in mapper:
        raise ValueError(...)
    out[i] = mapper[key]
```

iii. These loops are relatively minor bottlenecks compared to spike loading and binning, but represent missed vectorization opportunities.

## 12-c. What processing does the code repeat multiple times?

i.
- `BrainRegions()` is instantiated once per session inside `build_session_payload()`, rather than being created once and shared. This involves loading the brain atlas each time.
- The wheel and whisker interpolation is done in pass 1 (`build_session_behavior`) and the results are stored, but the behavioral data (timestamps, values) is loaded once and not re-read in pass 2 - this is efficient.
- `resolve_latest()` calls `sorted(base.glob(pattern))` repeatedly for each file type in each probe directory.

ii.
```python
def build_session_payload(prepared, wheel_edges, whisker_edges):
    ...
    brain_regions = BrainRegions()  # Re-instantiated for every session
```

iii. The BrainRegions instantiation is the most notable repeated computation; it could be cached or passed as a parameter.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i.
- The code computes and stores `excluded_session_notes` (an empty list in the metadata), which adds no information.
- Processing plots (`make_processing_plot`) are generated only in `--show-processing` mode, so this is not wasted in normal runs.
- The continuous wheel and whisker values (`wheel_cont`, `whisker_cont`) are computed in pass 1 and used for threshold computation, then the discretized versions are computed in pass 2. The continuous values are not stored in the final output.
- The `whisker_source` (which camera was used) is tracked per session but only stored in `PreparedSession`, not in the final pickle.

ii.
```python
data["metadata"]["excluded_session_notes"] = excluded_session_notes  # always empty list
```

iii. The main unnecessary computation is relatively minor. The two-pass architecture (pass 1 for behavior/thresholds, pass 2 for neural + final assembly) is a design choice that requires storing intermediate behavior results in memory.
