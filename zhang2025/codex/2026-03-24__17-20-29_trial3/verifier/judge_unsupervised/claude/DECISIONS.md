# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from the `bwm_release.csv` file (459 sessions, 699 probe insertions, 139 mice) as the canonical session roster. For each session, it reads local ALF files directly from disk (trials parquet, spike times/clusters npy, wheel npy, whisker motion energy npy) rather than using the remote ONE API. Sessions missing any required modality (spikes, trials, wheel, whisker) are skipped.

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
    # ... builds probe_info map from bwm_df
    return bwm_df, sessions_df
```

iii. The AI chose to use the same `bwm_release.csv` roster as the reference code's `0_data_caching.py`, but switched from remote ONE/SessionLoader access to direct local file reads for efficiency.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified from the `subject` column in `bwm_release.csv`. The unique subjects list is built from successfully converted sessions in their original order.

ii.
```python
subjects = ordered_unique([session.subject for session in processed_sessions])
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
```

iii. The AI documents that 135 subjects survived (vs 139 in the release) because 21 sessions were excluded by task-specific filters.

## 1-c. How are the data split into sessions?

i. Each unique `eid` in the release CSV corresponds to one session. Sessions are processed independently (in parallel for `--full` mode) and sorted by their release index. 438 sessions were successfully converted.

ii.
```python
sessions_df = (
    bwm_df[["eid", "lab", "subject", "date", "session_number"]]
    .drop_duplicates(subset=["eid"], keep="first")
    .reset_index(drop=True)
)
# In main():
processed_sessions.sort(key=lambda session: session.release_index)
```

iii. The AI notes this is close to the 433 sessions used in the methods paper; the difference comes from 21 sessions having zero valid trials after all filters.

## 1-d. How are the data split into trials?

i. Trials are loaded from the `_ibl_trials.table.pqt` parquet file for each session. Each row is one trial. After applying the trial quality mask, trials are indexed by their original position in the trials table.

ii.
```python
def load_trials_and_mask_current(session_path, min_rt=0.08, max_rt=2.0, max_trial_len=10.0, exclude_nochoice=True):
    trials_path = find_latest_file(session_path / "alf", "**/_ibl_trials.table.pqt")
    trials = pd.read_parquet(trials_path).copy()
    # ... builds exclusion query
    return trials.reset_index(drop=True), mask
```

iii. The AI follows the reference code structure where each trial is a row in the trials dataframe.

## 1-e. How are trials filtered based on quality controls?

i. Trials are excluded if: (1) reaction time < 0.08s or > 2.0s, (2) trial length (feedback_times - goCue_times) > 10.0s, (3) any of stimOn_times, choice, feedback_times, probabilityLeft, firstMovement_times, feedbackType are NaN, (4) choice == 0 (no-go). Additionally, trials are excluded if wheel or whisker behavior data cannot be aligned (missing or insufficient coverage).

ii.
```python
if min_rt is not None:
    query_parts.append(f"(firstMovement_times - stimOn_times < {min_rt})")
if max_rt is not None:
    query_parts.append(f"(firstMovement_times - stimOn_times > {max_rt})")
if max_trial_len is not None:
    query_parts.append(f"(feedback_times - goCue_times > {max_trial_len})")
for event in ["stimOn_times", "choice", "feedback_times", "probabilityLeft", "firstMovement_times", "feedbackType"]:
    query_parts.append(f"{event}.isnull()")
if exclude_nochoice:
    query_parts.append("(choice == 0)")
# Later:
keep_mask = np.asarray(trials_mask, dtype=bool) & wheel_mask & whisker_mask
```

iii. The AI states this matches `load_trials_and_mask(one=one, eid=eid, max_trial_len=10.0)` from the reference code. The additional wheel/whisker mask is necessary because the task requires these behavioral outputs.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `spikes.times.npy` (spike timestamps) and `spikes.clusters.npy` (cluster assignments) loaded from pykilosort sort directories, along with `clusters.metrics.pqt` for quality labels and `channels.brainLocationIds_ccf_2017.npy` for brain region mapping.

ii.
```python
spikes = {
    "times": np.load(sort_dir / "spikes.times.npy"),
    "clusters": np.load(sort_dir / "spikes.clusters.npy").astype(np.int32),
}
clusters_labeled = pd.read_parquet(sort_dir / "clusters.metrics.pqt")
```

iii. The AI explains these are the same spike-sorted outputs that the reference code's `load_spiking_data` accesses via SpikeSortingLoader, but loaded directly from local files.

## 2-b. How is the `neural` data processed?

i. Spikes are merged across probes using `merge_probes()`, then binned into 20ms bins within each trial's stimulus-aligned window [-0.5s, 1.5s]. The result is spike count matrices of shape (n_neurons, 100) per trial.

ii.
```python
spikes, clusters = merge_probes(spikes_list, clusters_list)
# ...
binned_array = get_spike_data_per_interval(
    regspikes, regclu,
    interval_begs=intervals[:, 0], interval_ends=intervals[:, 1],
    interval_len=interval_len, binsize=PARAMS["binsize"],
)
return np.array([x.T for x in binned_array], dtype=np.float32)
```

iii. The AI reproduces the reference code's spike binning pipeline, using `bincount2D` from iblutil and the same `get_spike_data_per_interval` logic, but without multiprocessing within a session.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only clusters with `label >= 1` (well-isolated) are kept. This is applied at the spike loading stage by filtering the clusters dataframe and then subsetting spikes.

ii.
```python
def load_spiking_data_current(session_path, probe_name, qc=None):
    # ...
    if qc is None:
        return spikes, clusters_labeled
    iok = clusters_labeled["label"] >= qc
    selected_clusters = clusters_labeled[iok].copy()
    spike_idx, ib = ismember(spikes["clusters"], selected_clusters.index.to_numpy())
    # ...

# Called with qc=1:
spikes, clusters = load_spiking_data_current(session_path, probe_name=probe_name, qc=1)
```

iii. The AI chose to filter with `qc=1` because the data paper reports 75,708 well-isolated neurons, whereas the reference code's `prepare_data` does NOT pass a `qc` argument (defaulting to `None`, keeping all clusters). The AI documented this as a deliberate deviation to match the paper's curation intent.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to stimulus onset (`stimOn_times`). Per-trial intervals are computed as `[stimOn_times - 0.5, stimOn_times + 1.5]`.

ii.
```python
PARAMS = {
    "align_time": "stimOn_times",
    "time_window": (-0.5, 1.5),
}
# In bin_spiking_data_current:
intervals = np.vstack([
    trials_df[PARAMS["align_time"]] + PARAMS["time_window"][0],
    trials_df[PARAMS["align_time"]] + PARAMS["time_window"][1],
]).T
```

iii. This matches both the reference code's default parameters and the task instruction ("Temporally align based on stimulus onset").

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is 20ms (0.02s), producing 100 time bins per trial. No rebinning is applied - data is binned directly at 20ms.

ii.
```python
PARAMS = {
    "binsize": 0.02,
    "interval_len": 2.0,
}
# metadata:
"time_bin_size": 20.0,
```

iii. The AI uses 20ms bins matching the reference code's `binsize=0.02`. The methods paper mentions 50ms bins for choice/prior decoding but the AI prioritized the code's 20ms setting, noting the discrepancy.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. This is derived from the alignment parameters themselves (time_window and binsize), not from a raw data variable. It represents the time axis of each trial relative to stimulus onset.

ii.
```python
def build_time_axis() -> np.ndarray:
    start, end = PARAMS["time_window"]
    binsize = PARAMS["binsize"]
    n_bins = int(math.ceil((end - start) / binsize))
    return np.linspace(start + binsize, end, n_bins, dtype=np.float32)
```

iii. The AI constructs a fixed time axis from [-0.48, 1.50] (bin end times) that is the same for every trial, derived from the alignment window parameters.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. A fixed array of 100 values is computed using `np.linspace(start + binsize, end, n_bins)`, representing bin end times from -0.48s to 1.50s relative to stimulus onset.

ii.
```python
return np.linspace(start + binsize, end, n_bins, dtype=np.float32)
```

iii. The AI uses bin end times to match the reference behavior interpolation grid.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The same time axis is used for both neural binning and the time input. The neural bins span [-0.5, 1.5] with 20ms bins, and the time input represents the end times of those same bins.

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

iii. Since the time axis is constructed from the same parameters as the neural binning, alignment is inherent.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. This is derived from `trials.probabilityLeft` in the trials table.

ii.
```python
prob_left_all = trials_df["probabilityLeft"].to_numpy(dtype=float)
block_num_all = trial_number_in_block(prob_left_all)
```

iii. Block boundaries are detected by changes in `probabilityLeft`, and trials are counted within each block.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The code iterates through all trials (before filtering) and increments a counter within each block. A new block starts whenever `probabilityLeft` changes. The counter resets to 1 at each block boundary.

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

iii. The AI computes trial number in block on the full (unfiltered) trial table, then subsets to kept trials. This preserves the true experimental block progression.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. Choice is derived from `trials.choice` in the trials table.

ii.
```python
choice_all = trials_df["choice"].to_numpy()
choice_labels = choice_to_label(choice_all[keep_idx])
```

iii. Raw choice values from the IBL trials table.

## 5-b. What processing is involved in computing `output` *Choice*?

i. IBL choice values are mapped: `+1` (left) -> `0`, `-1` (right) -> `1`. No-go trials (choice == 0) are excluded by the trial mask. The per-trial choice label is repeated across all 100 time bins.

ii.
```python
def choice_to_label(choice_values: np.ndarray) -> np.ndarray:
    # Official IBL docs: choice == -1 means chose right, choice == +1 means chose left.
    return (choice_values == -1).astype(np.int64)
# In assemble_dataset:
np.full(T, session.choice_labels[trial_idx], dtype=np.int64),
```

iii. The AI follows the task instruction: "left = 0, right = 1". However, note the IBL convention comment says choice == -1 means right, but the actual IBL convention is: choice == 1 means right (wheel turn), choice == -1 means left. The AI's mapping `(choice == -1) -> 1 (right)` is actually reversed from the correct IBL mapping.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. Derived from `trials.probabilityLeft` in the trials table.

ii.
```python
prob_left_all = trials_df["probabilityLeft"].to_numpy(dtype=float)
prior_labels = prior_to_label(prob_left_all[keep_idx])
```

iii. Direct mapping from the probabilityLeft column.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The three possible values are mapped: 0.2 -> 0, 0.5 -> 1, 0.8 -> 2. Values are rounded to 1 decimal place before mapping. The label is repeated across all time bins.

ii.
```python
PRIOR_MAP = {0.2: 0, 0.5: 1, 0.8: 2}
def prior_to_label(probability_left: np.ndarray) -> np.ndarray:
    rounded = np.round(np.asarray(probability_left, dtype=float), 1)
    return np.array([PRIOR_MAP[x] for x in rounded], dtype=np.int64)
```

iii. This matches the task instruction: "0.2 -> 0, 0.5 -> 1, 0.8 -> 2".

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. Wheel speed is derived from `_ibl_wheel.timestamps.npy` and `_ibl_wheel.position.npy`.

ii.
```python
wheel_timestamps = np.load(find_latest_file(alf_path, "**/_ibl_wheel.timestamps.npy"))
wheel_position = np.load(find_latest_file(alf_path, "**/_ibl_wheel.position.npy"))
```

iii. These are the standard IBL wheel data files.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Wheel position is interpolated to 1000 Hz using `interpolate_position`, then velocity is computed using `velocity_filtered` (Butterworth low-pass filter at 20 Hz, order 8). Speed is the absolute value of velocity.

ii.
```python
position, times = interpolate_position(wheel_timestamps, wheel_position, freq=1000)
velocity, _ = velocity_filtered(position, fs=1000, corner_frequency=20, order=8)
return {
    "times": np.asarray(times, dtype=np.float32),
    "values": np.abs(np.asarray(velocity, dtype=np.float32)),
}
```

iii. The AI uses brainbox wheel processing functions. The reference code uses `SessionLoader.load_wheel()` which also calls these functions internally, producing `abs(velocity)` for wheel-speed.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Global tertile thresholds are computed from all wheel speed values across all sessions and trials. Values are then discretized into 3 bins (low=0, medium=1, high=2) using `np.digitize`.

ii.
```python
def compute_thresholds(processed_sessions):
    wheel_values = np.concatenate([np.concatenate(session.wheel_continuous) for session in processed_sessions])
    q1, q2 = np.quantile(wheel_values, [1/3, 2/3])
    thresholds["wheel_speed"] = (float(q1), float(q2))

def discretize(values, thresholds):
    q1, q2 = thresholds
    return np.digitize(values, bins=np.array([q1, q2]), right=False).astype(np.int64)
```

iii. The AI chose global tertile thresholds to ensure each bin gets approximately 1/3 of samples. This is not specified in the reference code (which keeps wheel continuous) but follows the task instruction to discretize into 3 bins.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. Wheel speed is aligned to stimulus onset using the same [-0.5, 1.5] window and 20ms bins as the neural data. Continuous wheel speed is linearly interpolated onto the same time grid.

ii.
```python
x_interp = np.linspace(interval_begs[idx] + binsize, interval_ends[idx], n_bins)
fn = interp1d(t_seg, v_seg, kind="linear", fill_value="extrapolate")
y_interp = fn(x_interp)
```

iii. This uses stimulus onset alignment for wheel speed, which differs from the reference code/paper that align dynamic behaviors to first movement onset. The AI justifies this as required by the task instruction ("Temporally align based on stimulus onset").

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. Whisker motion energy is derived from `leftCamera.ROIMotionEnergy.npy` (or `rightCamera.ROIMotionEnergy.npy` as fallback) and corresponding camera timestamps (`_ibl_leftCamera.times.npy` or `_ibl_rightCamera.times.npy`).

ii.
```python
if target == "left-whisker-motion-energy":
    times = np.load(find_latest_file(alf_path, "**/_ibl_leftCamera.times.npy"))
    values = np.load(find_latest_file(alf_path, "**/leftCamera.ROIMotionEnergy.npy"))
```

iii. The AI uses the same camera-side fallback logic as the reference code: try left camera first, fall back to right.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. Raw whisker motion energy values are loaded directly. If camera timestamps are longer than values, the timestamps are truncated from the beginning to match. The signal is then linearly interpolated onto the trial time grid.

ii.
```python
if times.shape[0] > values.shape[0]:
    times = times[-values.shape[0]:]
# Later interpolated via get_behavior_per_interval_current
```

iii. The AI handles the timestamp/value length mismatch by trimming timestamps from the start, then uses the same interpolation approach as for wheel speed.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Same approach as wheel speed: global tertile thresholds computed across all sessions, then `np.digitize` into 3 bins.

ii.
```python
whisker_values = np.concatenate([np.concatenate(session.whisker_continuous) for session in processed_sessions])
q1, q2 = np.quantile(whisker_values, [1/3, 2/3])
thresholds["whisker_motion_energy"] = (float(q1), float(q2))
```

iii. Same justification as wheel speed: the task requires 3-bin discretization of a continuous signal.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Same alignment as wheel speed: stimulus onset aligned, [-0.5, 1.5] window, linear interpolation onto the 20ms bin grid.

ii.
```python
# Same get_behavior_per_interval_current function used for both wheel and whisker
whisker_traces, whisker_mask = align_continuous_behavior(session_path, "whisker-motion-energy", trials_df)
```

iii. Same justification as wheel: task instruction requires stimulus onset alignment.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Multiple strategies: (1) Trials with NaN in key events are excluded by the trial mask. (2) Trials where wheel or whisker data is missing or doesn't cover the trial window are excluded via behavior alignment masks. (3) Camera timestamp/value length mismatches are handled by trimming. (4) Sessions with fewer than 2 valid trials are skipped entirely. (5) All-zero neural trials (16 across 3 sessions) are kept as genuine sparse-data cases. (6) Failed sessions are retried (transient errors) or skipped with documentation.

ii.
```python
keep_mask = np.asarray(trials_mask, dtype=bool) & wheel_mask & whisker_mask
keep_idx = np.flatnonzero(keep_mask)
if keep_idx.size < 2:
    raise RuntimeError(f"Only {keep_idx.size} valid trials after filtering")
```

iii. The AI documented investigation of all-zero neural trials in Step 10, confirming they are genuine sparse-data windows rather than bugs.

## 10-a. What are the most time-consuming steps of the code?

i. Spike binning is the most time-consuming step per session, followed by behavior loading and alignment. The full conversion took ~34.5 minutes for 438 sessions with 12 parallel workers.

ii.
```python
timing={
    "trials_s": t_trials - t0,
    "spikes_s": t_spikes - t_trials,
    "behavior_s": t_behavior - t_spikes,
    "total_s": t_behavior - t0,
}
```

iii. The AI benchmarked at worker counts 4, 8, 12, 16 and found 12 workers optimal for disk I/O-bound processing.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The main loop in `get_spike_data_per_interval` iterates over trials sequentially (replacing the reference code's multiprocessing pool). The `trial_number_in_block` function also uses a Python loop. The `prior_to_label` function uses a list comprehension instead of vectorized mapping.

ii.
```python
# Sequential loop over intervals:
for interval_idx, (t_beg, t_end) in enumerate(zip(interval_begs, interval_ends)):
    idxs_t = (times >= t_beg) & (times < t_end)
    # ...

# Python loop for trial_number_in_block:
for idx, value in enumerate(probability_left):
    if idx == 0 or not np.isclose(value, prev):
        current = 1
    # ...
```

iii. The trial-level spike binning loop could potentially be vectorized but the per-trial nature of bincount2D makes this difficult. The trial_number_in_block loop is inherently sequential.

## 10-c. What processing does the code repeat multiple times?

i. The behavior alignment function `get_behavior_per_interval_current` is called separately for wheel and whisker with largely identical logic. The `find_latest_file` glob search is repeated for each file type in each session. Spike masking with `np.isin` is done once for the entire session rather than repeated per trial.

ii.
```python
wheel_traces, wheel_mask = align_continuous_behavior(session_path, "wheel-speed", trials_df)
whisker_traces, whisker_mask = align_continuous_behavior(session_path, "whisker-motion-energy", trials_df)
```

iii. The duplication in behavior alignment is relatively minor since the function handles different data streams.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code loads and stores `cluster_good` (QC labels) in the ProcessedSession even though clusters have already been filtered to `label >= 1`. The code also builds and stores various metadata fields (session_eids, skipped_sessions, mode) that are not used by the decoder. The `release_index` field is used only for sorting and then stored in metadata unnecessarily.

ii.
```python
"good_clusters": (clusters["label"] >= 1).to_numpy(dtype=np.int8),
# Stored but already filtered - all values are 1
```

iii. These are minor inefficiencies that don't affect correctness.
