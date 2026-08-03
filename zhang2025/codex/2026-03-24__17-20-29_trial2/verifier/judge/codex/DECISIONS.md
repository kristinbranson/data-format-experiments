# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does not use `ONE`, `SessionLoader`, or `SpikeSortingLoader`. It reads `code/code_zhang2025/data/bwm_release.csv`, groups rows by `eid`, builds session paths under `data/one_cache`, and loads trials, wheel, camera, cluster-metrics, and spike files directly from parquet and `.npy` files. It uses a two-pass workflow: pass 1 prepares behavior and masks; pass 2 reloads spike data and builds the final payload.

ii. ```python
def load_release_sessions() -> list[SessionSpec]:
    bwm = pd.read_csv(RELEASE_CSV, index_col=0)
    grouped = bwm.groupby("eid", sort=False)
    ...

def load_trials_table(session_path: Path) -> pd.DataFrame:
    path = resolve_latest(session_path / "alf", "**/_ibl_trials.table.pqt")
    return pd.read_parquet(path)
```

```python
metrics_path = resolve_latest(probe_dir, "**/clusters.metrics.pqt")
spikes_times_path = resolve_latest(probe_dir, "**/spikes.times.npy")
spikes_clusters_path = resolve_latest(probe_dir, "**/spikes.clusters.npy")
metrics = pd.read_parquet(metrics_path, columns=["label"])
spikes_times = np.load(spikes_times_path, mmap_mode="r")
spikes_clusters = np.load(spikes_clusters_path, mmap_mode="r")
```

iii. `CONVERSION_NOTES.md` says the agent chose direct ALF loading because `brainbox.io.one.SessionLoader` was unusable in this environment due to an unavailable `neuropixel` dependency and `.rest` permission issues when trying `ONE.load_object(...)`.

## 1-b. How are the data split into subjects (mice)?

i. Subjects come from the `subject` column in the release CSV. Each `SessionSpec` stores a subject, and the final dataset builds `subjects` as the sorted unique subject names and `subject_idx` as the per-session index into that list.

ii. ```python
sessions.append(
    SessionSpec(
        eid=eid,
        subject=str(row["subject"]),
        lab=str(row["lab"]),
        ...
    )
)
```

```python
subject_names = sorted({ps.spec.subject for ps in prepared_sessions})
subject_to_idx = {subject: i for i, subject in enumerate(subject_names)}
...
subject_idx_list.append(subject_to_idx[prepared.spec.subject])
```

iii. The notes treat the frozen release CSV as the authoritative source of session metadata, so subject IDs are taken directly from that table.

## 1-c. How are the data split into sessions?

i. The AI treats each unique `eid` in `bwm_release.csv` as one session. It groups the CSV by `eid`, creates one `SessionSpec` per group, and processes the corresponding session directory under `data/one_cache/<lab>/Subjects/<subject>/<date>/<number>/`.

ii. ```python
def load_release_sessions() -> list[SessionSpec]:
    bwm = pd.read_csv(RELEASE_CSV, index_col=0)
    grouped = bwm.groupby("eid", sort=False)
    for eid, df in grouped:
        sessions.append(SessionSpec(...))
```

```python
@property
def session_path(self) -> Path:
    return DATA_ROOT / self.lab / "Subjects" / self.subject / self.date / f"{self.session_number:03d}"
```

iii. The notes say the agent chose the frozen 459-session release in `bwm_release.csv` because it best matched the paper-level release counts.

## 1-d. How are the data split into trials?

i. Trials come directly from the rows of `_ibl_trials.table.pqt`. The AI loads the full trial table and uses boolean masks to keep or drop rows; the surviving rows become the converted trials.

ii. ```python
def load_trials_table(session_path: Path) -> pd.DataFrame:
    path = resolve_latest(session_path / "alf", "**/_ibl_trials.table.pqt")
    return pd.read_parquet(path)
```

```python
trials = load_trials_table(spec.session_path)
trial_mask = compute_trial_mask(trials)
...
keep_mask = trial_mask & wheel_mask & whisker_mask
```

iii. No extra trial-segmentation heuristic is documented. The agent treats the trial table as authoritative, as the reference code does.

## 1-e. How are trials filtered based on quality controls?

i. The AI uses a stricter mask than the human reference. It requires non-null `stimOn_times`, `choice`, `feedback_times`, `probabilityLeft`, `firstMovement_times`, and `feedbackType`; reaction time in `[0.08, 2.0]`; `feedback_times - goCue_times <= 10.0`; and `choice != 0`. It then intersects this with wheel and whisker coverage masks, and drops sessions with fewer than two valid trials.

ii. ```python
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
```

```python
keep_mask = trial_mask & wheel_mask & whisker_mask
if keep_mask.sum() < 2:
    return None
```

iii. The notes explicitly justify keeping `exclude_nochoice=True`, `max_trial_len=10.0`, and the behavior-coverage requirement as part of staying close to the reference code path.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural arrays are built from `spikes.times.npy` and `spikes.clusters.npy`, after filtering units with `clusters.metrics.pqt` and mapping unit anatomy with `clusters.channels.npy` plus `channels.brainLocationIds_ccf_2017.npy`. The actual neural tensor is produced from spike times and cluster assignments.

ii. ```python
metrics_path = resolve_latest(probe_dir, "**/clusters.metrics.pqt")
clusters_channels_path = resolve_latest(probe_dir, "**/clusters.channels.npy")
channels_region_ids_path = resolve_latest(probe_dir, "**/channels.brainLocationIds_ccf_2017.npy")
spikes_times_path = resolve_latest(probe_dir, "**/spikes.times.npy")
spikes_clusters_path = resolve_latest(probe_dir, "**/spikes.clusters.npy")
```

```python
good_spike_times = np.asarray(spikes_times[spike_mask], dtype=np.float64)
good_spike_clusters = remap[spikes_clusters[spike_mask]] + cluster_offset
```

iii. The notes say the conversion uses direct ALF readers but the same source streams as the reference: spike times, spike clusters, and cluster QC/region metadata.

## 2-b. How is the `neural` data processed?

i. The AI merges good units across probes, sorts spikes by time, bins them into 20 ms stimulus-aligned bins for each trial, and stores dense `(n_neurons, 100)` spike-count arrays as `float16`. It does not divide by bin width to convert counts to firing rates.

ii. ```python
order = np.argsort(merged_spike_times)
merged_spike_times = merged_spike_times[order]
merged_spike_clusters = merged_spike_clusters[order]
```

```python
rel = spike_times[i0:i1] - interval_begs[i]
bin_idx = np.floor(rel / binsize).astype(np.int64)
flat = spike_clusters[i0:i1][valid] * n_bins + bin_idx[valid]
counts = np.bincount(flat, minlength=n_units * n_bins).reshape(n_units, n_bins)
out.append(counts.astype(np.float16))
```

iii. `CONVERSION_NOTES.md` says the agent kept low-precision dense neural arrays to make the pickle tractable, and describes pass 2 as binning stimulus-aligned spikes into 20 ms bins. The final code never converts those counts to Hz.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Clusters are filtered to `label >= 1` from `clusters.metrics.pqt`. Only spikes assigned to those clusters are retained, and sessions with no such units are dropped.

ii. ```python
metrics = pd.read_parquet(metrics_path, columns=["label"])
good_rows = metrics["label"].to_numpy(copy=False) >= 1
if not np.any(good_rows):
    continue
...
spike_mask = good_rows[spikes_clusters]
```

```python
if cluster_offset == 0:
    raise ValueError(f"No good units remained for {spec.eid}")
```

iii. The notes justify this by saying it reproduces the paper’s 75,708 well-isolated neurons exactly and keeps the dense decoder dataset computationally feasible.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial is aligned to `stimOn_times`. The code defines the interval `[stimOn_times - 0.5, stimOn_times + 1.5]`, slices spikes in that window, and bins them relative to the interval start.

ii. ```python
ALIGN_EVENT = "stimOn_times"
TIME_WINDOW = (-0.5, 1.5)
```

```python
kept_align_times = prepared.trials.loc[prepared.keep_mask, ALIGN_EVENT].to_numpy(dtype=np.float64)
interval_begs = align_times + time_window[0]
interval_ends = align_times + time_window[1]
rel = spike_times[i0:i1] - interval_begs[i]
```

iii. The notes say a common stimulus-onset-aligned grid was chosen because the decoder task explicitly required stimulus-onset alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted dataset uses 20 ms bins and 100 time bins per trial. No later temporal rebinning or smoothing is applied.

ii. ```python
BINSIZE = 0.02
NBINS = int(round((TIME_WINDOW[1] - TIME_WINDOW[0]) / BINSIZE))
```

```python
n_bins = int(np.ceil((time_window[1] - time_window[0]) / binsize))
```

iii. The constants and notes both state that the conversion follows the 20 ms common decoding grid.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. The AI represents this input with a fixed relative-time vector tied to the stimulus-onset alignment event. It is defined from `stimOn_times` and the chosen decoding window rather than from a separate measured raw signal.

ii. ```python
ALIGN_EVENT = "stimOn_times"
TIME_WINDOW = (-0.5, 1.5)
BINSIZE = 0.02
COMMON_RELATIVE_TIMES = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], NBINS, dtype=np.float32)
```

iii. The notes say the agent used one common stimulus-onset-aligned grid for every variable because that was required by the decoder task.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The AI creates a fixed 100-point vector from `-0.48` to `1.50` s with `np.linspace`, then reuses it for every trial. That is effectively a right-edge grid, not the bin-center grid used by the human reference.

ii. ```python
COMMON_RELATIVE_TIMES = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], NBINS, dtype=np.float32)
...
inp = np.vstack(
    [
        COMMON_RELATIVE_TIMES,
        np.full(NBINS, prepared.trial_number_in_block[trial_idx], dtype=np.float32),
    ]
).astype(np.float32)
```

iii. The notes document the resulting range as `[-0.48, 1.50]` and describe it as the common time grid.

## 3-c. How is `input` *Time since stimulus onset* aligned with the neural data?

i. The AI aligns this input by using the same stimulus-onset trial window everywhere and by evaluating behavior on the same 100-point common grid. However, that grid is offset by half a bin from the neural bin centers used in the human reference.

ii. ```python
COMMON_RELATIVE_TIMES = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], NBINS, dtype=np.float32)
```

```python
x_interp = np.linspace(t_beg + binsize, t_end, n_bins)
y_interp = np.interp(x_interp, curr_times, curr_vals)
```

iii. The notes repeatedly describe a single common stimulus-onset grid for neural, input, and output streams, and that grid is the `[-0.48, 1.50]` vector above.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from `probabilityLeft` in the trial table. The AI detects a new block whenever `probabilityLeft` changes from one trial to the next.

ii. ```python
def compute_trial_number_in_block(prob_left: np.ndarray) -> np.ndarray:
    out = np.zeros(len(prob_left), dtype=np.int16)
    prev = None
    counter = 0
    for i, val in enumerate(prob_left):
        current = None if pd.isna(val) else float(val)
        if i == 0 or current != prev:
            counter = 1
```

iii. The notes explicitly say the block structure is recovered from `probabilityLeft` because the raw trial table has no separate block index.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The AI computes the counter on the unfiltered trial table before applying `keep_mask`, but it counts from 1 instead of 0. The scalar is then repeated across all 100 time bins for each kept trial.

ii. ```python
if i == 0 or current != prev:
    counter = 1
else:
    counter += 1
out[i] = counter
...
trial_number_in_block=trial_number_in_block[keep_mask],
```

```python
np.full(NBINS, prepared.trial_number_in_block[trial_idx], dtype=np.float32)
```

iii. The notes justify computing this before filtering so the number reflects the animal’s real position in the block, but they also document values starting at 1.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. `choice` is taken directly from the trial table’s `choice` column after trial filtering.

ii. ```python
choice_raw=trials.loc[keep_mask, "choice"].to_numpy(),
```

iii. The notes say the raw sign convention was re-checked directly against the trial table.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The AI maps raw `choice == 1` to class `0` (left) and raw `choice == -1` to class `1` (right), rejects any unexpected values after filtering, and repeats the class across all 100 time bins.

ii. ```python
def map_choice(raw_choice: np.ndarray) -> np.ndarray:
    mapped = np.empty(raw_choice.shape[0], dtype=np.int8)
    mapped[raw_choice == 1] = 0
    mapped[raw_choice == -1] = 1
    if not np.all(np.isin(raw_choice, [-1, 1])):
        raise ValueError("Unexpected choice values encountered after filtering.")
    return mapped
```

```python
np.full(NBINS, choice[trial_idx], dtype=np.int8)
```

iii. The notes and README document the same left/right mapping.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It is derived from the trial table’s `probabilityLeft` column after filtering.

ii. ```python
prior_raw=trials.loc[keep_mask, "probabilityLeft"].to_numpy(),
```

iii. The notes consistently describe prior/block state as coming from `probabilityLeft`.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The AI maps `0.2 -> 0`, `0.5 -> 1`, and `0.8 -> 2`, rounds each float to one decimal place before lookup, and repeats the resulting class across all 100 time bins.

ii. ```python
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

```python
np.full(NBINS, prior[trial_idx], dtype=np.int8)
```

iii. The notes and README document the same categorical mapping for prior probability.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. Wheel speed is derived from `_ibl_wheel.timestamps.npy` and `_ibl_wheel.position.npy`. The AI interpolates wheel position to a regular grid, differentiates it, and takes the absolute value of velocity.

ii. ```python
timestamps = np.load(resolve_latest(alf_path, "**/_ibl_wheel.timestamps.npy"))
position = np.load(resolve_latest(alf_path, "**/_ibl_wheel.position.npy"))
interp_pos, interp_times = interpolate_position(timestamps, position, freq=1000)
velocity, _ = velocity_filtered(interp_pos, fs=1000, corner_frequency=20, order=8)
return interp_times.astype(np.float64), np.abs(velocity).astype(np.float32)
```

iii. The notes explicitly say the AI retained the bundled wheel helper functions from `brainbox.behavior.wheel`.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The AI uses `interpolate_position(..., freq=1000)` and `velocity_filtered(..., corner_frequency=20, order=8)` to build wheel velocity, takes `abs`, and linearly interpolates the resulting speed trace into each stimulus-aligned trial window.

ii. ```python
interp_pos, interp_times = interpolate_position(timestamps, position, freq=1000)
velocity, _ = velocity_filtered(interp_pos, fs=1000, corner_frequency=20, order=8)
return interp_times.astype(np.float64), np.abs(velocity).astype(np.float32)
```

```python
x_interp = np.linspace(t_beg + binsize, t_end, n_bins)
y_interp = np.interp(x_interp, curr_times, curr_vals)
```

iii. The notes justify this as keeping the same wheel helper functions while only replacing the failing data-loader layer.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. The AI computes one global pair of tertile thresholds from all kept wheel-speed time points across all sessions, then digitizes every wheel-speed sample into bins `0/1/2`. A fallback handles tied quantiles.

ii. ```python
all_wheel = np.concatenate([ps.wheel_cont.reshape(-1) for ps in prepared_sessions]).astype(np.float32)
wheel_edges = robust_tertile_edges(all_wheel)
```

```python
def digitize_three_bins(values: np.ndarray, edges: tuple[float, float]) -> np.ndarray:
    low, high = edges
    if high <= low:
        return np.zeros(values.shape, dtype=np.int8)
    return np.digitize(values, bins=np.array([low, high], dtype=np.float32), right=False).astype(np.int8)
```

iii. The notes make the justification explicit: session-specific thresholds would make labels inconsistent across sessions, so the AI chose global thresholds instead.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. Wheel speed is aligned to the same stimulus-onset trial window as the neural data and resampled to 100 time steps, but the AI samples it at `[-0.48, ..., 1.50]` rather than at the neural bin centers used by the human reference.

ii. ```python
interval_begs = align_times + time_window[0]
interval_ends = align_times + time_window[1]
...
x_interp = np.linspace(t_beg + binsize, t_end, n_bins)
```

```python
COMMON_RELATIVE_TIMES = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], NBINS, dtype=np.float32)
```

iii. The notes justify common stimulus-onset alignment for all outputs, but they also report the resulting common time range as `[-0.48, 1.50]`.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. Whisker motion energy is derived from either `leftCamera.ROIMotionEnergy.npy` with `_ibl_leftCamera.times.npy` or, if left is unavailable, the corresponding right-camera files. The AI prefers left and falls back to right.

ii. ```python
for view in ("left", "right"):
    me_candidates = sorted(alf_path.glob(f"**/{view}Camera.ROIMotionEnergy.npy"))
    ts_candidates = sorted(alf_path.glob(f"**/_ibl_{view}Camera.times.npy"))
    if not me_candidates or not ts_candidates:
        continue
    motion_energy = np.load(me_candidates[-1])
    timestamps = np.load(ts_candidates[-1])
    ...
    return timestamps.astype(np.float64), motion_energy.astype(np.float32), view
```

iii. The notes and README both say left camera is preferred and right camera is only a fallback.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The AI uses the released motion-energy trace directly, repairs one specific timestamp-length mismatch by trimming extra timestamps from the front, and linearly interpolates the trace into each stimulus-aligned trial window. It does not apply extra filtering or normalization.

ii. ```python
def check_video_timestamps(view: str, video_timestamps: np.ndarray, video_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if video_timestamps.shape[0] > video_data.shape[0]:
        video_timestamps = video_timestamps[-video_data.shape[0]:]
    return video_timestamps, video_data
```

```python
whisker_interp, whisker_mask = interpolate_behavior_per_trial(whisker_times, whisker_motion, align_times)
```

iii. The notes explicitly call out this timestamp repair as a raw-data irregularity fix and otherwise describe motion energy as using the released trace directly.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Like wheel speed, whisker motion energy is discretized with one global pair of tertile thresholds computed over all kept whisker time points across all sessions, then digitized into bins `0/1/2`.

ii. ```python
all_whisker = np.concatenate([ps.whisker_cont.reshape(-1) for ps in prepared_sessions]).astype(np.float32)
whisker_edges = robust_tertile_edges(all_whisker)
```

```python
whisker_bins = digitize_three_bins(prepared.whisker_cont, whisker_edges)
```

iii. The notes explicitly justify global thresholds for dynamic outputs so categories stay comparable across sessions.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Whisker motion energy is aligned to the same stimulus-onset window and resampled to 100 time steps, but those time points are the AI’s shifted `[-0.48, ..., 1.50]` grid rather than the neural bin centers in the human reference.

ii. ```python
whisker_interp, whisker_mask = interpolate_behavior_per_trial(whisker_times, whisker_motion, align_times)
```

```python
x_interp = np.linspace(t_beg + binsize, t_end, n_bins)
```

iii. The notes justify common stimulus-onset alignment for all outputs, but the implemented interpolation grid is the same shifted one used for the time input and wheel output.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI mostly handles missing or irregular data by dropping it. Trials with missing key fields or incomplete wheel/whisker coverage are removed, sessions with missing required streams or fewer than two valid trials are skipped, sessions with no good units are skipped, and spikes whose cluster IDs fall outside the metrics table are masked out. One specific camera irregularity is repaired by trimming extra timestamps from the front if there are more timestamps than motion-energy samples.

ii. ```python
if video_timestamps.shape[0] < video_data.shape[0]:
    ...
    raise ValueError(...)
if video_timestamps.shape[0] > video_data.shape[0]:
    video_timestamps = video_timestamps[-video_data.shape[0]:]
```

```python
except FileNotFoundError:
    return spec.eid, None, "missing_required_stream"
...
if keep_mask.sum() < 2:
    return None
```

```python
valid_spikes = spikes_clusters < n_clusters
if np.all(valid_spikes):
    spike_mask = good_rows[spikes_clusters]
else:
    spike_mask = np.zeros(spikes_clusters.shape[0], dtype=bool)
    spike_mask[valid_spikes] = good_rows[spikes_clusters[valid_spikes]]
```

iii. The notes justify the trimming rule as a fix for an observed raw-data irregularity and justify skipping sessions without whisker streams because whisker motion energy is mandatory for the requested decoder.

## 10-a. What are the most time-consuming steps of the code?

i. The AI’s notes identify pass-2 spike loading and related copying as the main bottleneck, with ALF path resolution and dense spike payload construction also contributing. In the code, the heaviest operations are reading cluster/spike arrays per probe and binning spikes trial by trial.

ii. ```python
metrics = pd.read_parquet(metrics_path, columns=["label"])
spikes_times = np.load(spikes_times_path, mmap_mode="r")
spikes_clusters = np.load(spikes_clusters_path, mmap_mode="r")
```

```python
for i in range(len(align_times)):
    ...
    counts = np.bincount(flat, minlength=n_units * n_bins).reshape(n_units, n_bins)
```

iii. `CONVERSION_NOTES.md` says the 'real bottleneck was spike-stream reload and dtype copying in pass 2'.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The code still has per-trial loops for behavior interpolation, spike binning, and final input/output assembly. The first two are the same broad opportunities the human reference has; the AI also adds a third per-trial assembly loop.

ii. ```python
for i in range(len(align_times)):
    ...
    y_interp = np.interp(x_interp, curr_times, curr_vals)
```

```python
for i in range(len(align_times)):
    ...
    counts = np.bincount(flat, minlength=n_units * n_bins).reshape(n_units, n_bins)
```

```python
for trial_idx in range(len(neural_trials)):
    inp = np.vstack([...]).astype(np.float32)
    out = np.vstack([...]).astype(np.int8)
    session_input.append(inp)
    session_output.append(out)
```

iii. The notes discuss vectorized spike binning as an optimization target, but the final code still leaves these loops in place.

## 10-c. What processing does the code repeat multiple times?

i. The AI repeats several steps across its two-pass design. It counts good units in pass 1, then reloads cluster metrics again in pass 2; it resolves many revisioned file paths repeatedly with `resolve_latest`; and it computes behavior traces in pass 1, then digitizes them later after global thresholds are known.

ii. ```python
def count_good_units(spec: SessionSpec) -> int:
    ...
    metrics = pd.read_parquet(metrics_path, columns=["label"])
    total += int((metrics["label"].to_numpy() >= 1).sum())
```

```python
metrics = pd.read_parquet(metrics_path, columns=["label"])
good_rows = metrics["label"].to_numpy(copy=False) >= 1
```

```python
wheel_interp, wheel_mask = interpolate_behavior_per_trial(...)
...
wheel_bins = digitize_three_bins(prepared.wheel_cont, wheel_edges)
```

iii. The notes explicitly describe the converter as a two-pass system with pass-1 behavior preparation followed by pass-2 spike reloading and payload construction.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several pieces of work are not used by the final decoder dataset: pass-1 `count_good_units` only supports gating and summaries before pass-2 reloads the real spike data; `excluded_session_notes` is created in metadata but never populated; and the optional plotting pipeline plus `matplotlib` import do not contribute to `converted_data.pkl`.

ii. ```python
n_good_units = count_good_units(spec)
if n_good_units == 0:
    return None
```

```python
excluded_session_notes: list[dict[str, Any]] = []
...
"excluded_session_notes": excluded_session_notes,
```

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
...
def make_processing_plot(...):
    ...
```

iii. The notes emphasize plotting and diagnostic summaries as development aids, and they describe the two-pass structure that makes `count_good_units` only a preliminary screen.
