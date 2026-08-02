# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from the 459-session `bwm_release.csv` roster (same as the reference code). It reads the CSV to get session metadata (eid, lab, subject, date, session_number, probe info), then for each session it reads local ALF files directly from disk (trials parquet tables, spike numpy arrays, wheel/whisker numpy arrays) rather than using the ONE API remote calls. Sessions are processed either sequentially (sample mode) or in parallel via `ProcessPoolExecutor` (full mode).

ii.
```python
BWM_RELEASE_CSV = ROOT / "code" / "code_zhang2025" / "data" / "bwm_release.csv"

def load_release_sessions() -> tuple[pd.DataFrame, pd.DataFrame]:
    bwm_df = pd.read_csv(BWM_RELEASE_CSV, index_col=0)
    sessions_df = (
        bwm_df[["eid", "lab", "subject", "date", "session_number"]]
        .drop_duplicates(subset=["eid"], keep="first")
        .reset_index(drop=True)
    )
    probe_map = (
        bwm_df[["eid", "pid", "probe_name"]]
        .astype({"pid": str, "probe_name": str})
        .groupby("eid")
        .apply(
            lambda frame: [
                (str(pid), str(probe_name))
                for pid, probe_name in zip(frame["pid"], frame["probe_name"])
            ],
            include_groups=False,
        )
        .to_dict()
    )
    sessions_df["probe_info"] = sessions_df["eid"].map(probe_map)
    return bwm_df, sessions_df
```

iii. The AI justified using `bwm_release.csv` because it is the same roster used by the reference code's `0_data_caching.py`. The AI chose direct local file reads instead of remote ONE API calls for performance, documenting this reduced sample conversion time from 132.5s to 12.1s.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are derived from the `subject` column in `bwm_release.csv`. After processing all sessions, unique subjects are collected in encounter order across the sorted processed sessions, and each session is assigned a `subject_idx` pointing into the global subjects list.

ii.
```python
def assemble_dataset(processed_sessions, thresholds):
    subjects = ordered_unique([session.subject for session in processed_sessions])
    subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
    ...
    subject_idx.append(subject_to_idx[session.subject])
```

iii. The AI noted 135 subjects in the converted data (vs 139 in the public release), attributing the difference to 21 sessions excluded by task-specific validity requirements.

## 1-c. How are the data split into sessions?

i. Sessions are defined as unique eids from `bwm_release.csv`. Each eid maps to a unique combination of subject, date, and session number. Sessions are processed independently and sorted by their original release index before assembly.

ii.
```python
sessions_df = (
    bwm_df[["eid", "lab", "subject", "date", "session_number"]]
    .drop_duplicates(subset=["eid"], keep="first")
    .reset_index(drop=True)
)
...
processed_sessions.sort(key=lambda session: session.release_index)
```

iii. The AI maintained deterministic session ordering using the release index from the CSV roster.

## 1-d. How are the data split into trials?

i. Trials are loaded from the `_ibl_trials.table.pqt` parquet file in each session's ALF directory. Each row in the trials table represents one trial.

ii.
```python
def load_trials_and_mask_current(session_path, min_rt=0.08, max_rt=2.0,
                                  max_trial_len=10.0, exclude_nochoice=True):
    trials_path = find_latest_file(session_path / "alf", "**/_ibl_trials.table.pqt")
    trials = pd.read_parquet(trials_path).copy()
```

iii. The AI used the same trial table source as the reference code's `SessionLoader.load_trials()`, but loaded it directly from disk.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered using multiple criteria: (1) reaction time < 0.08s excluded, (2) reaction time > 2.0s excluded, (3) trial length (feedback_times - goCue_times) > 10.0s excluded, (4) NaN in any of stimOn_times, choice, feedback_times, probabilityLeft, firstMovement_times, feedbackType excluded, (5) no-choice trials (choice == 0) excluded. Additionally, trials where wheel or whisker behavior data could not be aligned are excluded.

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
...
# Then combined with behavior masks:
keep_mask = np.asarray(trials_mask, dtype=bool) & wheel_mask & whisker_mask
```

iii. The AI documented that it reproduced the reference code's trial mask exactly, including `max_trial_len=10.0`, matching `load_trials_and_mask()` in `ibl_data_utils.py`.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `spikes.times.npy` (spike timestamps) and `spikes.clusters.npy` (cluster assignments) from pykilosort spike sorting outputs, plus `clusters.metrics.pqt` (for QC labels), `clusters.channels.npy` and `channels.brainLocationIds_ccf_2017.npy` (for brain region mapping).

ii.
```python
def load_spiking_data_current(session_path, probe_name, qc=None):
    sort_dir = find_latest_file(
        session_path / "alf" / probe_name / "pykilosort", "**/spikes.times.npy"
    ).parent
    spikes = {
        "times": np.load(sort_dir / "spikes.times.npy"),
        "clusters": np.load(sort_dir / "spikes.clusters.npy").astype(np.int32),
    }
    clusters_labeled = pd.read_parquet(sort_dir / "clusters.metrics.pqt")
```

iii. The AI documented that these are the same spike sorting outputs loaded by the reference `SpikeSortingLoader`.

## 2-b. How is the `neural` data processed?

i. Spikes from multiple probes within a session are merged using the reference code's `merge_probes` function. Merged spikes are then binned into per-trial intervals using 20ms bins, aligned to stimulus onset with a [-0.5, 1.5]s window, producing spike count matrices of shape (n_neurons, 100) per trial.

ii.
```python
spikes, clusters = merge_probes(spikes_list, clusters_list)
...
binned_array = get_spike_data_per_interval(
    regspikes, regclu,
    interval_begs=intervals[:, 0], interval_ends=intervals[:, 1],
    interval_len=interval_len, binsize=PARAMS["binsize"],
)
return np.array([x.T for x in binned_array], dtype=np.float32)
```

iii. The AI directly imported `merge_probes` from the reference code and reimplemented the binning logic following the same `bincount2D` approach as the reference `get_spike_data_per_interval`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI filters neurons to only include well-isolated clusters with `label >= 1` (QC threshold). This is applied during spike loading by passing `qc=1` to the loading function.

ii.
```python
def load_spiking_data_current(session_path, probe_name, qc=None):
    ...
    if qc is None:
        return spikes, clusters_labeled
    iok = clusters_labeled["label"] >= qc
    selected_clusters = clusters_labeled[iok].copy()
    spike_idx, ib = ismember(spikes["clusters"], selected_clusters.index.to_numpy())
    ...
# Called with qc=1:
spikes, clusters = load_spiking_data_current(session_path, probe_name=probe_name, qc=1)
```

iii. The AI justified this by noting the reference paper reports 75,708 well-isolated neurons and the reference code stores `good_clusters = label >= 1` in metadata. The AI chose to filter at load time rather than after, noting this as a deviation from the reference code's `prepare_data()` which loads all clusters but stores the QC flag for later use.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to stimulus onset (`stimOn_times`) with a time window of [-0.5, 1.5] seconds, matching both the instructions and the reference code parameters.

ii.
```python
PARAMS = {
    "interval_len": 2.0, "binsize": 0.02,
    "align_time": "stimOn_times", "time_window": (-0.5, 1.5),
}
...
intervals = np.vstack([
    trials_df[PARAMS["align_time"]] + PARAMS["time_window"][0],
    trials_df[PARAMS["align_time"]] + PARAMS["time_window"][1],
]).T
```

iii. The AI documented that this matches `0_data_caching.py` parameters: `align_time='stimOn_times'`, `time_window=(-0.5, 1.5)`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is 20ms (0.02s), producing 100 time bins per trial over the 2-second window. No rebinning is applied; raw spikes are directly binned at this resolution.

ii.
```python
PARAMS = {"binsize": 0.02, ...}
# metadata:
"time_bin_size": 20.0,  # in ms
```

iii. The AI matched the reference code's `binsize=0.02` parameter from `0_data_caching.py`.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is derived from the `stimOn_times` column in the trials table and the fixed time window parameters. The actual input values are a fixed time axis representing bin end times relative to stimulus onset.

ii.
```python
def build_time_axis() -> np.ndarray:
    start, end = PARAMS["time_window"]
    binsize = PARAMS["binsize"]
    n_bins = int(math.ceil((end - start) / binsize))
    return np.linspace(start + binsize, end, n_bins, dtype=np.float32)
```

iii. The AI noted this input is not present in the reference code; it is a decoder input derived from the adopted alignment grid as required by the task instructions.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. A fixed array of 100 values from -0.48 to 1.50 seconds (bin end times) is computed once using `np.linspace(start + binsize, end, n_bins)`. This same array is used for every trial.

ii.
```python
return np.linspace(start + binsize, end, n_bins, dtype=np.float32)
# Range: [-0.48, 1.50]
```

iii. The AI chose bin end times (not bin centers or left edges) as the time axis, matching the reference behavior interpolation grid used in `get_behavior_per_interval`.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The time axis is inherently aligned with neural data because it uses the same binning parameters (20ms bins, [-0.5, 1.5]s window). The same array is replicated for every trial.

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

iii. The time axis is defined from the same parameters that govern neural and behavioral binning, ensuring alignment.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from `probabilityLeft` column in the trials table. Block transitions are detected by changes in the `probabilityLeft` value.

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

iii. The AI computed trial number in block from the original experimental trial order (before dropping invalid trials), then subsetted to retained trials. This preserves the true block progression.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. Starting from the first trial, a counter starts at 1. When `probabilityLeft` changes between consecutive trials (using `np.isclose` for floating-point comparison), the counter resets to 1. The counter increments for each trial with the same `probabilityLeft`. The value is constant across all time bins within a trial.

ii.
```python
prob_left_all = trials_df["probabilityLeft"].to_numpy(dtype=float)
block_num_all = trial_number_in_block(prob_left_all)
...
np.full(time_axis.shape[0], block_num_all[i], dtype=np.float32),
```

iii. The AI documented this as a per-trial variable repeated across time bins, derived before trial filtering to preserve the original block structure.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. Choice is derived from the `choice` column in the trials table (`_ibl_trials.table.pqt`).

ii.
```python
choice_all = trials_df["choice"].to_numpy()
choice_labels = choice_to_label(choice_all[keep_idx])
```

iii. The AI used the same `trials_df['choice']` variable as the reference code's `bin_behaviors` function.

## 5-b. What processing is involved in computing `output` *Choice*?

i. IBL raw choice values are remapped: `+1` (left) becomes `0`, `-1` (right) becomes `1`. No-choice trials (choice == 0) are already excluded by the trial mask. The choice label is constant across all time bins for a trial.

ii.
```python
def choice_to_label(choice_values: np.ndarray) -> np.ndarray:
    choice_values = np.asarray(choice_values)
    # choice == -1 means chose right, choice == +1 means chose left.
    return (choice_values == -1).astype(np.int64)
...
# In output assembly:
np.full(T, session.choice_labels[trial_idx], dtype=np.int64),
```

iii. The AI documented that IBL choice `+1` = left, `-1` = right based on official IBL docs, matching the task instruction mapping of left=0, right=1.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It is derived from the `probabilityLeft` column in the trials table.

ii.
```python
prob_left_all = trials_df["probabilityLeft"].to_numpy(dtype=float)
prior_labels = prior_to_label(prob_left_all[keep_idx])
```

iii. The AI used the same source variable as the reference code's `bin_behaviors` which extracts `trials_df['probabilityLeft']`.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The continuous `probabilityLeft` values (0.2, 0.5, 0.8) are mapped to categorical labels: 0.2 -> 0, 0.5 -> 1, 0.8 -> 2. Values are rounded to 1 decimal place before mapping. The label is constant across all time bins for a trial.

ii.
```python
PRIOR_MAP = {0.2: 0, 0.5: 1, 0.8: 2}

def prior_to_label(probability_left: np.ndarray) -> np.ndarray:
    rounded = np.round(np.asarray(probability_left, dtype=float), 1)
    unknown = sorted(set(rounded.tolist()) - set(PRIOR_MAP))
    if unknown:
        raise ValueError(f"Unexpected probabilityLeft values: {unknown}")
    return np.array([PRIOR_MAP[x] for x in rounded], dtype=np.int64)
```

iii. The AI followed the task instruction mapping exactly: 0.2->0, 0.5->1, 0.8->2.

## 9-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. Wheel speed is derived from `_ibl_wheel.timestamps.npy` and `_ibl_wheel.position.npy` files.

ii.
```python
if target == "wheel-speed":
    wheel_timestamps = np.load(find_latest_file(alf_path, "**/_ibl_wheel.timestamps.npy"))
    wheel_position = np.load(find_latest_file(alf_path, "**/_ibl_wheel.position.npy"))
```

iii. The AI used the same raw wheel data files that the reference code's `SessionLoader.load_wheel()` accesses.

## 9-b. What processing is involved in computing `output` *Wheel speed*?

i. Raw wheel timestamps and position are interpolated to 1000 Hz using `brainbox.behavior.wheel.interpolate_position`. Velocity is computed using `velocity_filtered` with a 20 Hz corner frequency and 8th-order Butterworth filter. Wheel speed is the absolute value of velocity.

ii.
```python
position, times = interpolate_position(wheel_timestamps, wheel_position, freq=1000)
velocity, _ = velocity_filtered(position, fs=1000, corner_frequency=20, order=8)
return {
    "times": np.asarray(times, dtype=np.float32),
    "values": np.abs(np.asarray(velocity, dtype=np.float32)),
}
```

iii. The AI noted this follows the reference code's `load_target_behavior('wheel-speed')` which uses `SessionLoader.load_wheel()` and takes `abs(velocity)`. However, the reference code uses `SessionLoader.load_wheel()` which internally uses Gaussian smoothing for velocity computation, while the AI uses `velocity_filtered` with a Butterworth filter. This is a difference in the velocity computation method.

## 9-c. How is `output` *Wheel speed* thresholded into categories?

i. After all sessions are processed, global tertile thresholds are computed from all retained aligned wheel speed samples across all sessions. Values are discretized into 3 bins (low=0, medium=1, high=2) using `np.digitize` with the two tertile boundaries.

ii.
```python
def compute_thresholds(processed_sessions):
    wheel_values = np.concatenate(
        [np.concatenate(session.wheel_continuous) for session in processed_sessions]
    )
    q1, q2 = np.quantile(wheel_values, [1 / 3, 2 / 3])
    thresholds[name] = (float(q1), float(q2))

def discretize(values, thresholds):
    q1, q2 = thresholds
    return np.digitize(values, bins=np.array([q1, q2]), right=False).astype(np.int64)
```

iii. The AI documented that global tertile thresholds ensure exactly 1/3 of all timepoints fall in each bin, which is confirmed by the reported distribution of [0.3333, 0.3333, 0.3333].

## 9-d. How is `output` *Wheel speed* aligned with the neural data?

i. Wheel speed is aligned to the same stimulus-onset window ([-0.5, 1.5]s) as neural data. The continuous wheel speed signal is linearly interpolated onto the same time grid of 100 bins (bin end times from `interval_beg + binsize` to `interval_end`).

ii.
```python
def get_behavior_per_interval_current(target_times, target_vals, trials_df, allow_nans=True):
    ...
    x_interp = np.linspace(interval_begs[idx] + binsize, interval_ends[idx], n_bins)
    fn = interp1d(t_seg, v_seg, kind="linear", fill_value="extrapolate")
    y_interp = fn(x_interp)
```

iii. The AI matched the reference code's `get_behavior_per_interval` interpolation approach, using the same interpolation grid as neural binning.

## 10-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. Whisker motion energy is derived from `leftCamera.ROIMotionEnergy.npy` (with `_ibl_leftCamera.times.npy` for timestamps), falling back to `rightCamera.ROIMotionEnergy.npy` if left camera data is unavailable.

ii.
```python
if target == "left-whisker-motion-energy":
    times = np.load(find_latest_file(alf_path, "**/_ibl_leftCamera.times.npy"))
    values = np.load(find_latest_file(alf_path, "**/leftCamera.ROIMotionEnergy.npy"))
    if times.shape[0] > values.shape[0]:
        times = times[-values.shape[0]:]
...
def align_continuous_behavior(session_path, behavior_name, trials_df):
    if behavior_name == "whisker-motion-energy":
        target = load_target_behavior_current(session_path, "left-whisker-motion-energy")
        if target.get("skip"):
            target = load_target_behavior_current(session_path, "right-whisker-motion-energy")
```

iii. The AI matched the reference code's left-camera-first, right-camera-fallback approach from `bin_behaviors`.

## 10-b. What processing is involved in computing `output` *Whisker motion energy*?

i. Raw whisker motion energy values are loaded directly (no additional processing like filtering). If camera timestamps are longer than the motion energy data, the timestamps are trimmed from the beginning (keeping the last N timestamps to match). The signal is then linearly interpolated onto the trial-aligned time grid.

ii.
```python
if times.shape[0] > values.shape[0]:
    times = times[-values.shape[0]:]
return {
    "times": np.asarray(times, dtype=np.float32),
    "values": np.asarray(values, dtype=np.float32),
}
```

iii. The AI handled the timestamp/value length mismatch by trimming timestamps from the front, which aligns with the assumption that initial frames may lack motion energy computation.

## 10-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Same approach as wheel speed: global tertile thresholds computed from all retained aligned whisker motion energy samples, then discretized into 3 bins using `np.digitize`.

ii.
```python
# Same compute_thresholds and discretize functions as wheel speed
whisker_values = np.concatenate(
    [np.concatenate(session.whisker_continuous) for session in processed_sessions]
)
q1, q2 = np.quantile(whisker_values, [1 / 3, 2 / 3])
```

iii. The AI used the same discretization approach for both continuous outputs, ensuring consistent treatment.

## 10-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Same as wheel speed: aligned to stimulus-onset window [-0.5, 1.5]s, linearly interpolated to 100 bins matching the neural data time grid.

ii.
```python
# Same get_behavior_per_interval_current function used for both wheel and whisker
whisker_traces, whisker_mask = align_continuous_behavior(
    session_path, "whisker-motion-energy", trials_df
)
```

iii. The AI used the same alignment approach for all continuous behaviors, matching the reference code.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Multiple missing data scenarios are handled: (1) Sessions missing required modalities (spikes, trials, wheel, whisker) are skipped. (2) Trials with NaN in required event columns are excluded by the trial mask. (3) Trials where behavior data cannot be aligned (data not present, starts too late, ends too early, bad interval) are excluded via behavior masks. (4) Camera timestamp/value length mismatches are handled by trimming timestamps. (5) All-zero neural trials (no spikes in the window) are kept as genuine sparse-data cases. (6) Sessions with fewer than 2 valid trials are skipped. (7) Transient errors during parallel processing trigger retries.

ii.
```python
# Behavior alignment validation:
if len(v_seg) == 0: reasons[idx] = "target data not present"
if np.abs(interval_begs[idx] - t_seg[0]) > binsize: reasons[idx] = "target data starts too late"
...
# Session-level:
if keep_idx.size < 2:
    raise RuntimeError(f"Only {keep_idx.size} valid trials after filtering")
```

iii. The AI documented that 21 sessions were excluded because they had 0 valid trials after all filters, and 16 all-zero neural trials across 3 sessions were verified as genuine sparse data rather than bugs.

## 12-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are: (1) Spike binning via `get_spike_data_per_interval` which loops over all trials and bins spikes using `bincount2D`. (2) Loading spike sorting data from disk (numpy files). (3) Behavior interpolation across all trials. The AI benchmarked that per-session processing takes 3-6 seconds, with the full 438-session conversion taking ~34.5 minutes with 12 parallel workers.

ii.
```python
# Timing instrumented in process_one_session:
timing={"trials_s": t_trials - t0, "spikes_s": t_spikes - t_trials,
        "behavior_s": t_behavior - t_spikes, "total_s": t_behavior - t0}
```

iii. The AI documented benchmarks showing 48 workers caused disk contention, with 12 workers being optimal based on empirical testing.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops could be vectorized: (1) The spike binning loop in `get_spike_data_per_interval` iterates over each trial interval sequentially. (2) The `trial_number_in_block` function uses a Python for-loop over all trials. (3) The behavior interpolation in `get_behavior_per_interval_current` loops over each trial. (4) The `prior_to_label` function uses a Python list comprehension with dictionary lookup per element.

ii.
```python
# Trial-by-trial spike binning loop:
for interval_idx, (t_beg, t_end) in enumerate(zip(interval_begs, interval_ends)):
    idxs_t = (times >= t_beg) & (times < t_end)
    ...
# Trial number in block loop:
for idx, value in enumerate(probability_left):
    if idx == 0 or not np.isclose(value, prev):
        current = 1
    ...
```

iii. The AI opted for sequential loops matching the reference code's approach rather than vectorizing, prioritizing correctness over performance.

## 12-c. What processing does the code repeat multiple times?

i. (1) The `find_latest_file` glob search is called multiple times per session for different file types. (2) Brain region mapping is done per-session and then again globally during assembly. (3) The time axis is passed to each session but could be computed once (it is computed once via `build_time_axis()`). (4) Behavior loading creates a new `interp1d` interpolation object for every trial.

ii.
```python
# find_latest_file called multiple times per session:
trials_path = find_latest_file(session_path / "alf", "**/_ibl_trials.table.pqt")
wheel_timestamps = np.load(find_latest_file(alf_path, "**/_ibl_wheel.timestamps.npy"))
```

iii. The AI acknowledged some redundancy but prioritized code clarity and correctness.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. (1) The code loads and stores `cluster_good` (QC labels) in `ProcessedSession` metadata but already filters to QC >= 1 during loading, making this redundant. (2) The code computes and stores per-session timing information that is only used for logging. (3) The code stores `raw_trial_count`, `kept_trial_count`, `skipped_trial_count`, and `kept_trial_indices` per session which are used only for logging and not included in the final pickle. (4) The code loads `reward` and `contrast` information is not loaded (unlike the reference code which loads these), but `cluster_qc` detailed metrics are stored in `meta` dict but not used in the final dataset.

ii.
```python
meta = {
    "cluster_regions": list(clusters["acronym"]),
    "good_clusters": (clusters["label"] >= 1).to_numpy(dtype=np.int8),
    "cluster_qc": {k: np.asarray(v) for k, v in clusters.to_dict("list").items()},
}
```

iii. The AI stored extra metadata for debugging and validation purposes, which is reasonable during development but adds some overhead.
