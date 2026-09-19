# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI uses `bwm_release.csv` (the reference code's release roster of 459 sessions) to enumerate sessions. It constructs local file paths from the CSV fields (lab, subject, date, session_number) to find data in the ONE cache directory. Data files (trials, spikes, wheel, whisker) are read directly from disk as `.npy` and `.pqt` files rather than through the ONE API. It checks local file availability before attempting conversion.

ii.
```python
BWM_RELEASE_CSV = ROOT / "code" / "code_zhang2025" / "data" / "bwm_release.csv"
ONE_CACHE_DIR = ROOT / "data" / "one_cache"

def load_release_sessions() -> tuple[pd.DataFrame, pd.DataFrame]:
    bwm_df = pd.read_csv(BWM_RELEASE_CSV, index_col=0)
    sessions_df = (
        bwm_df[["eid", "lab", "subject", "date", "session_number"]]
        .drop_duplicates(subset=["eid"], keep="first")
        .reset_index(drop=True)
    )
```

```python
def release_session_path(row: pd.Series) -> Path:
    return (
        ONE_CACHE_DIR / row["lab"] / "Subjects" / row["subject"]
        / row["date"] / f"{int(row['session_number']):03d}"
    )
```

iii. The AI chose to use `bwm_release.csv` and direct file reads to avoid the overhead and transient network failures of the ONE API, while still using the same canonical session roster as the reference code.

## 1-b. How are the data split into subjects?

i. Subject names come from the `subject` column of `bwm_release.csv`. An `ordered_unique` function builds the subject list from the order sessions were processed. `subject_idx` maps each session to its index in that list.

ii.
```python
subjects = ordered_unique([session.subject for session in processed_sessions])
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
```

iii. Subject identity is already provided in the release CSV, no derivation needed.

## 1-c. How are the data split into sessions?

i. Each row in `bwm_release.csv` (after deduplication by `eid`) represents one session. Sessions are processed individually and assembled into the final dataset.

ii.
```python
sessions_df = (
    bwm_df[["eid", "lab", "subject", "date", "session_number"]]
    .drop_duplicates(subset=["eid"], keep="first")
    .reset_index(drop=True)
)
```

iii. Sessions are already the unit of organization in the release roster.

## 1-d. How are the data split into trials?

i. The trials table (`_ibl_trials.table.pqt`) has one row per trial. It is loaded per session and indexed by trial number.

ii.
```python
def load_trials_and_mask_current(session_path, min_rt=0.08, max_rt=2.0, max_trial_len=10.0, exclude_nochoice=True):
    trials_path = find_latest_file(session_path / "alf", "**/_ibl_trials.table.pqt")
    trials = pd.read_parquet(trials_path).copy()
```

iii. The trials table provides a natural per-trial structure.

## 1-e. How are trials filtered based on quality controls?

i. Trials are excluded if: (1) reaction time (firstMovement_times - stimOn_times) is below 0.08s or above 2.0s; (2) trial length (feedback_times - goCue_times) exceeds 10.0s; (3) any of stimOn_times, choice, feedback_times, probabilityLeft, firstMovement_times, or feedbackType is NaN; (4) choice == 0 (no response). Additionally, trials without valid wheel or whisker data coverage are excluded.

ii.
```python
query_parts = []
if min_rt is not None:
    query_parts.append(f"(firstMovement_times - stimOn_times < {min_rt})")
if max_rt is not None:
    query_parts.append(f"(firstMovement_times - stimOn_times > {max_rt})")
if max_trial_len is not None:
    query_parts.append(f"(feedback_times - goCue_times > {max_trial_len})")
for event in ["stimOn_times", "choice", "feedback_times", "probabilityLeft",
               "firstMovement_times", "feedbackType"]:
    query_parts.append(f"{event}.isnull()")
if exclude_nochoice:
    query_parts.append("(choice == 0)")
```

```python
keep_mask = np.asarray(trials_mask, dtype=bool) & wheel_mask & whisker_mask
```

iii. The AI adopted the reference code's trial mask logic from `load_trials_and_mask`, including the `max_trial_len=10.0` parameter. Additional wheel/whisker coverage checks ensure behavioral data is available for each trial.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `spikes.times.npy` and `spikes.clusters.npy` per probe, plus `clusters.metrics.pqt` for QC labels and `channels.brainLocationIds_ccf_2017.npy` for region assignments.

ii.
```python
spikes = {
    "times": np.load(sort_dir / "spikes.times.npy"),
    "clusters": np.load(sort_dir / "spikes.clusters.npy").astype(np.int32),
}
clusters_labeled = pd.read_parquet(sort_dir / "clusters.metrics.pqt")
```

iii. These are the standard IBL spike sorting outputs.

## 2-b. How is the `neural` data processed?

i. Spikes are filtered by cluster QC (label >= 1), merged across probes using the reference code's `merge_probes` function, then binned into 20ms bins within the [-0.5, 1.5]s trial window around stimulus onset. The output is stored as spike counts (not firing rates). Per-trial arrays have shape (n_neurons, n_bins) after transposing.

ii.
```python
spikes, clusters = load_spiking_data_current(session_path, probe_name=probe_name, qc=1)
spikes, clusters = merge_probes(spikes_list, clusters_list)

# Spike binning
binned_array = get_spike_data_per_interval(
    regspikes, regclu,
    interval_begs=intervals[:, 0], interval_ends=intervals[:, 1],
    interval_len=interval_len, binsize=PARAMS["binsize"],
)
return np.array([x.T for x in binned_array], dtype=np.float32)
```

iii. The AI reused the reference code's `merge_probes` function and replicated its binning logic via `bincount2D`. However, it stores spike counts rather than converting to firing rates.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only clusters with `label >= 1` (well-isolated) are kept. The AI does NOT explicitly exclude clusters whose Beryl-mapped acronym is `void` (outside the brain), unlike the reference.

ii.
```python
if qc is None:
    return spikes, clusters_labeled
iok = clusters_labeled["label"] >= qc
selected_clusters = clusters_labeled[iok].copy()
spike_idx, ib = ismember(spikes["clusters"], selected_clusters.index.to_numpy())
```

iii. The AI applied the same QC threshold as the data paper (label >= 1 for well-isolated neurons) but did not add the void-region exclusion from the reference.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to stimulus onset (`stimOn_times`). The binning intervals are computed as `[stimOn_times - 0.5, stimOn_times + 1.5]` for each trial.

ii.
```python
intervals = np.vstack([
    trials_df[PARAMS["align_time"]] + PARAMS["time_window"][0],
    trials_df[PARAMS["align_time"]] + PARAMS["time_window"][1],
]).T
```

iii. This matches the reference code's alignment parameters.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 20ms bins, producing 100 time bins per trial. No rebinning is applied.

ii.
```python
PARAMS = {
    "interval_len": 2.0,
    "binsize": 0.02,
    "align_time": "stimOn_times",
    "time_window": (-0.5, 1.5),
}
n_bins = int(np.ceil(interval_len / binsize))  # 100
```

iii. Matches the reference code's binsize of 0.02.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. Derived from the alignment parameters (time window and bin size), not from a raw data variable. It is the time axis of the binning grid.

ii.
```python
def build_time_axis() -> np.ndarray:
    start, end = PARAMS["time_window"]
    binsize = PARAMS["binsize"]
    n_bins = int(math.ceil((end - start) / binsize))
    return np.linspace(start + binsize, end, n_bins, dtype=np.float32)
```

iii. The AI constructs the time axis from the binning parameters.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The time axis is computed as `np.linspace(-0.48, 1.5, 100)`, which represents bin END times rather than bin centers. The reference uses bin centers at `EDGES[:-1] + BIN/2`, producing values from -0.49 to 1.49.

ii.
```python
# AI: bin end times
return np.linspace(start + binsize, end, n_bins, dtype=np.float32)
# Produces: [-0.48, -0.4599..., ..., 1.50]

# Reference: bin centers
TIME = EDGES[:-1] + BIN / 2
# Produces: [-0.49, -0.47, ..., 1.49]
```

iii. The AI commented "Match the reference behavior interpolation grid: bin end times." It intentionally chose bin end times to match the reference code's behavior interpolation grid.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The same time axis array is used for both the input and the behavioral interpolation grid. The neural binning uses the same bin boundaries but the time input represents bin end times, which is offset by half a bin (0.01s) from the bin centers.

ii.
```python
input_trials = [
    np.vstack([
        time_axis,
        np.full(time_axis.shape[0], block_num_all[i], dtype=np.float32),
    ]).astype(np.float32, copy=False)
    for i in keep_idx
]
```

iii. The time axis represents bin end times while neural bins are defined by their start/end edges. This creates a slight misalignment (0.01s) between the time input label and the center of the neural bin it describes.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. Derived from `probabilityLeft` in the trials table. Block boundaries are detected where `probabilityLeft` changes.

ii.
```python
def trial_number_in_block(probability_left: np.ndarray) -> np.ndarray:
    probability_left = np.asarray(probability_left, dtype=float)
    out = np.zeros(probability_left.shape[0], dtype=np.float32)
    current = 0
    prev = None
    for idx, value in enumerate(probability_left):
        if idx == 0 or not np.isclose(value, prev):
            current = 1
        else:
            current += 1
        out[idx] = current
        prev = value
    return out
```

iii. Block boundaries are inferred from changes in `probabilityLeft`.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The trial number starts at 1 for the first trial of each block and increments by 1 for each subsequent trial. Block boundaries are identified using `np.isclose` comparison. This is computed on the full (unfiltered) trial table, preserving the true experimental block structure.

ii.
```python
prob_left_all = trials_df["probabilityLeft"].to_numpy(dtype=float)
block_num_all = trial_number_in_block(prob_left_all)
```

The value is then broadcast across time bins:
```python
np.full(time_axis.shape[0], block_num_all[i], dtype=np.float32)
```

iii. The AI starts counting at 1, while the reference uses `groupby(block).cumcount()` which starts at 0. This is an off-by-one difference.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. From the `choice` column of the trials table, where +1 = left, -1 = right, 0 = no response.

ii.
```python
def choice_to_label(choice_values: np.ndarray) -> np.ndarray:
    return (choice_values == -1).astype(np.int64)
```

iii. Standard IBL choice encoding.

## 5-b. What processing is involved in computing `output` *Choice*?

i. Choice is mapped to binary labels: left (+1) -> 0, right (-1) -> 1. No-response trials (0) are excluded by the trial mask. The per-trial label is broadcast across all time bins.

ii.
```python
# Mapping
return (choice_values == -1).astype(np.int64)

# Broadcast across time bins
np.full(T, session.choice_labels[trial_idx], dtype=np.int64),
```

iii. Matches the instruction specification (left = 0, right = 1).

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. From `probabilityLeft` in the trials table, which takes values 0.2, 0.5, or 0.8.

ii.
```python
PRIOR_MAP = {0.2: 0, 0.5: 1, 0.8: 2}

def prior_to_label(probability_left: np.ndarray) -> np.ndarray:
    rounded = np.round(np.asarray(probability_left, dtype=float), 1)
    return np.array([PRIOR_MAP[x] for x in rounded], dtype=np.int64)
```

iii. Direct mapping from the three task prior values.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The three values are mapped to categorical labels: 0.2 -> 0, 0.5 -> 1, 0.8 -> 2. Values are rounded to 1 decimal place for robustness. The label is broadcast across all time bins.

ii.
```python
np.full(T, session.prior_labels[trial_idx], dtype=np.int64),
```

iii. Matches the instruction specification.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. From `_ibl_wheel.timestamps.npy` and `_ibl_wheel.position.npy`.

ii.
```python
wheel_timestamps = np.load(find_latest_file(alf_path, "**/_ibl_wheel.timestamps.npy"))
wheel_position = np.load(find_latest_file(alf_path, "**/_ibl_wheel.position.npy"))
position, times = interpolate_position(wheel_timestamps, wheel_position, freq=1000)
velocity, _ = velocity_filtered(position, fs=1000, corner_frequency=20, order=8)
return {"times": np.asarray(times, dtype=np.float32),
        "values": np.abs(np.asarray(velocity, dtype=np.float32))}
```

iii. Uses the same IBL wheel processing functions as the reference.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Wheel position is interpolated to 1000Hz, velocity is computed with a 20Hz Butterworth low-pass filter (order 8), and speed is the absolute value of velocity. The continuous speed trace is then interpolated onto the trial time grid and discretized into 3 bins.

ii.
```python
position, times = interpolate_position(wheel_timestamps, wheel_position, freq=1000)
velocity, _ = velocity_filtered(position, fs=1000, corner_frequency=20, order=8)
return {"times": ..., "values": np.abs(np.asarray(velocity, dtype=np.float32))}
```

iii. Matches the reference's wheel processing through `SessionLoader.load_wheel()`.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. The AI computes GLOBAL thresholds by concatenating wheel speed values from ALL sessions and computing the 1/3 and 2/3 quantiles. These global thresholds are then applied to all sessions. The reference computes PER-SESSION percentiles.

ii.
```python
def compute_thresholds(processed_sessions):
    wheel_values = np.concatenate(
        [np.concatenate(session.wheel_continuous) for session in processed_sessions]
    )
    q1, q2 = np.quantile(values, [1/3, 2/3])
    thresholds[name] = (float(q1), float(q2))

def discretize(values, thresholds):
    q1, q2 = thresholds
    return np.digitize(values, bins=np.array([q1, q2]), right=False).astype(np.int64)
```

Reference per-session approach:
```python
SPLIT = [100 / 3, 2 * 100 / 3]
def discretize(trace):
    return np.digitize(trace, np.percentile(trace, SPLIT))
```

iii. The AI chose global thresholds for consistency across sessions, but the reference uses per-session percentiles to ensure roughly equal bin sizes within each session.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The wheel trace is interpolated onto the same time grid as the neural data using `scipy.interpolate.interp1d` with linear interpolation and extrapolation. The interpolation grid uses bin end times.

ii.
```python
x_interp = np.linspace(interval_begs[idx] + binsize, interval_ends[idx], n_bins)
fn = interp1d(t_seg, v_seg, kind="linear", fill_value="extrapolate")
y_interp = fn(x_interp)
```

iii. Uses linear interpolation to match the neural binning grid, same approach as the reference.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. From `leftCamera.ROIMotionEnergy.npy` (preferred) or `rightCamera.ROIMotionEnergy.npy` (fallback), with corresponding camera timestamps from `_ibl_leftCamera.times.npy` or `_ibl_rightCamera.times.npy`.

ii.
```python
def align_continuous_behavior(session_path, behavior_name, trials_df):
    if behavior_name == "whisker-motion-energy":
        target = load_target_behavior_current(session_path, "left-whisker-motion-energy")
        if target.get("skip"):
            target = load_target_behavior_current(session_path, "right-whisker-motion-energy")
```

iii. Uses left camera with right camera fallback, matching the reference code.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The motion energy trace is used as-is from the raw data. If the camera times array is longer than the values array, the times are truncated from the start (keeping the last N values). The trace is interpolated onto the trial time grid and discretized into 3 bins.

ii.
```python
times = np.load(find_latest_file(alf_path, "**/_ibl_leftCamera.times.npy"))
values = np.load(find_latest_file(alf_path, "**/leftCamera.ROIMotionEnergy.npy"))
if times.shape[0] > values.shape[0]:
    times = times[-values.shape[0]:]
```

iii. No additional filtering or normalization applied, matching the reference.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Same as wheel speed: GLOBAL thresholds computed from all sessions' whisker motion energy values at the 1/3 and 2/3 quantiles. The reference uses PER-SESSION percentiles.

ii.
```python
whisker_values = np.concatenate(
    [np.concatenate(session.whisker_continuous) for session in processed_sessions]
)
q1, q2 = np.quantile(values, [1/3, 2/3])
```

iii. Same global-vs-per-session difference as wheel speed.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Same approach as wheel speed: linearly interpolated onto the trial time grid (bin end times) using `interp1d`.

ii.
```python
x_interp = np.linspace(interval_begs[idx] + binsize, interval_ends[idx], n_bins)
fn = interp1d(t_seg, v_seg, kind="linear", fill_value="extrapolate")
```

iii. Uses the same alignment approach as all other time-varying variables.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Several layers of handling: (1) NaN checks in trial event columns exclude trials with missing data; (2) wheel/whisker coverage checks exclude trials where behavioral data doesn't span the trial window; (3) sessions with fewer than 2 valid trials are skipped; (4) sessions where no good clusters survive QC are skipped; (5) camera timestamp/value length mismatches are handled by truncating timestamps; (6) failed sessions are retried sequentially, then skipped if they fail again.

ii.
```python
# NaN exclusion
for event in ["stimOn_times", "choice", "feedback_times", "probabilityLeft",
              "firstMovement_times", "feedbackType"]:
    query_parts.append(f"{event}.isnull()")

# Minimum trial count
if keep_idx.size < 2:
    raise RuntimeError(f"Only {keep_idx.size} valid trials after filtering")

# Timestamp length mismatch
if times.shape[0] > values.shape[0]:
    times = times[-values.shape[0]:]
```

iii. The AI's approach is thorough, handling edge cases at multiple levels.

## 10-a. What are the most time-consuming steps of the code?

i. Loading spike sorting data from disk (reading large `.npy` files for spike times and clusters). The AI's benchmarking showed session conversion takes 3-6s per session, with I/O dominating.

ii.
```python
spikes = {
    "times": np.load(sort_dir / "spikes.times.npy"),
    "clusters": np.load(sort_dir / "spikes.clusters.npy").astype(np.int32),
}
```

iii. From CONVERSION_NOTES.md: "the same sample session that took 3.0 s sequentially took 91.4 s inside the 48-worker pool" due to disk contention.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The spike binning loop in `get_spike_data_per_interval` iterates per trial, applying `bincount2D` for each interval. The behavior interpolation loop in `get_behavior_per_interval_current` also iterates per trial. Both could potentially be vectorized.

ii.
```python
for interval_idx, (t_beg, t_end) in enumerate(zip(interval_begs, interval_ends)):
    idxs_t = (times >= t_beg) & (times < t_end)
    ...
    binned_tmp, _, cluster_idxs = bincount2D(...)
```

```python
for idx, (t_seg, v_seg) in enumerate(zip(target_times_list, target_vals_list)):
    fn = interp1d(t_seg, v_seg, kind="linear", fill_value="extrapolate")
    y_interp = fn(x_interp)
```

iii. The per-trial loops are a direct adaptation of the reference code's approach.

## 10-c. What processing does the code repeat multiple times?

i. The AI's `trial_number_in_block` is computed on the full unfiltered trial table, which is appropriate. No obviously redundant processing was identified.

ii. N/A

iii. N/A

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI loads and checks NaN status for `feedbackType` and `feedback_times` columns, and applies a `max_trial_len=10.0` filter based on `feedback_times - goCue_times`. These columns and this filter are not used in the final converted data or by the reference solution. The AI also stores continuous wheel and whisker traces in the `ProcessedSession` dataclass before discretization, which requires extra memory.

ii.
```python
# Extra trial filtering not in reference solution
if max_trial_len is not None:
    query_parts.append(f"(feedback_times - goCue_times > {max_trial_len})")
for event in [..., "feedback_times", ..., "feedbackType"]:
    query_parts.append(f"{event}.isnull()")
```

iii. These additional filters come from the reference code's `load_trials_and_mask` function but are not part of the reference solution's simpler trial filtering.
