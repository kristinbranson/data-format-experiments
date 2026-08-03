# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent starts from the release roster in `code/code_zhang2025/data/bwm_release.csv`, deduplicates to one row per `eid`, attaches per-session probe metadata, and then loads each session directly from the local ONE cache layout under `data/one_cache/<lab>/Subjects/<subject>/<date>/<session_number>/alf`. Within each session it separately reads the trials parquet, wheel arrays, whisker arrays, and pykilosort spike-sorting outputs. It does not use the reference `prepare_data()` function end-to-end; it reimplements local file readers and only reuses `merge_probes`.

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
    ...
    sessions_df["probe_info"] = sessions_df["eid"].map(probe_map)
    return bwm_df, sessions_df

def release_session_path(row: pd.Series) -> Path:
    return (
        ONE_CACHE_DIR / row["lab"] / "Subjects" / row["subject"]
        / row["date"] / f"{int(row['session_number']):03d}"
    )
```

```python
trials_df, trials_mask = load_trials_and_mask_current(session_path)
neural_dict, meta = load_session_neural(session_path, row["probe_info"])
wheel_traces, wheel_mask = align_continuous_behavior(session_path, "wheel-speed", trials_df)
whisker_traces, whisker_mask = align_continuous_behavior(
    session_path, "whisker-motion-energy", trials_df
)
```

iii. In Step 6 of `CONVERSION_NOTES.md`, the agent says it switched to direct local ALF/parquet/numpy reads because the vendored IBL helpers had dependency/API issues and remote ONE access was too slow and failure-prone. In Step 4 it also states that it still wanted to keep the same release roster and the same reference variables while replacing the access path.

## 1-b. How are the data split into subjects?

i. Subjects are identified from the `subject` column in `bwm_release.csv`. After session processing, the script builds an ordered unique subject list and a `subject_idx` array mapping each converted session back to its subject.

ii.
```python
sessions_df = (
    bwm_df[["eid", "lab", "subject", "date", "session_number"]]
    .drop_duplicates(subset=["eid"], keep="first")
    .reset_index(drop=True)
)
```

```python
subjects = ordered_unique([session.subject for session in processed_sessions])
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
...
subject_idx.append(subject_to_idx[session.subject])
```

iii. Step 5 of `CONVERSION_NOTES.md` explicitly maps `subject` from `bwm_release.csv` into `subjects` and `subject_idx`, with session order preserved from the release roster.

## 1-c. How are the data split into sessions?

i. Each unique `eid` in the release CSV becomes one candidate session. Sessions are processed independently by `process_one_session`, and only successful sessions are kept. The final session order is sorted back to original release order using `release_index`.

ii.
```python
sessions_df = (
    bwm_df[["eid", "lab", "subject", "date", "session_number"]]
    .drop_duplicates(subset=["eid"], keep="first")
    .reset_index(drop=True)
)
...
sessions_df = sessions_df.reset_index(names="release_index")
```

```python
session = process_one_session(row_series, time_axis)
processed_sessions.append(session)
...
processed_sessions.sort(key=lambda session: session.release_index)
```

iii. In Step 4, the agent chose the 459-session `bwm_release.csv` roster as canonical and treated later exclusions as task-specific failures, not as a different definition of session identity.

## 1-d. How are the data split into trials?

i. Trials come directly from rows of the per-session `_ibl_trials.table.pqt` parquet table. Neural and behavior data are converted into one per-trial segment for each row of `trials_df`, then filtered with `keep_idx`; the retained trial list for each session is the subset of rows that pass both trial QC and behavior-alignment checks.

ii.
```python
def load_trials_and_mask_current(...):
    trials_path = find_latest_file(session_path / "alf", "**/_ibl_trials.table.pqt")
    trials = pd.read_parquet(trials_path).copy()
    ...
    return trials.reset_index(drop=True), mask
```

```python
keep_mask = np.asarray(trials_mask, dtype=bool) & wheel_mask & whisker_mask
keep_idx = np.flatnonzero(keep_mask)
...
neural_trials = [binned_spikes[i].T.astype(np.float32, copy=False) for i in keep_idx]
input_trials = [ ... for i in keep_idx ]
```

iii. Step 1 notes describe the reference `load_trials_and_mask`, `bin_spiking_data`, `bin_behaviors`, and `align_spike_behavior` flow; the agent's own code mirrors that logic by creating trial-aligned arrays first and deleting bad trials afterward.

## 1-e. How are trials filtered based on quality controls?

i. A trial is kept only if it passes the trial-table mask and both continuous behavior masks. The trial-table mask excludes trials with reaction times outside `0.08-2.0 s`, trials longer than `10.0 s` from `goCue_times` to `feedback_times`, trials with missing `stimOn_times`, `choice`, `feedback_times`, `probabilityLeft`, `firstMovement_times`, or `feedbackType`, and no-choice trials (`choice == 0`). The final mask also requires valid wheel and whisker interpolation for that trial.

ii.
```python
if min_rt is not None:
    query_parts.append(f"(firstMovement_times - stimOn_times < {min_rt})")
if max_rt is not None:
    query_parts.append(f"(firstMovement_times - stimOn_times > {max_rt})")
if max_trial_len is not None:
    query_parts.append(f"(feedback_times - goCue_times > {max_trial_len})")
for event in [
    "stimOn_times",
    "choice",
    "feedback_times",
    "probabilityLeft",
    "firstMovement_times",
    "feedbackType",
]:
    query_parts.append(f"{event}.isnull()")
if exclude_nochoice:
    query_parts.append("(choice == 0)")
```

```python
keep_mask = np.asarray(trials_mask, dtype=bool) & wheel_mask & whisker_mask
keep_idx = np.flatnonzero(keep_mask)
if keep_idx.size < 2:
    raise RuntimeError(f"Only {keep_idx.size} valid trials after filtering")
```

iii. Step 1 and Step 4 of `CONVERSION_NOTES.md` say the agent intended to reproduce the reference `load_trials_and_mask(..., max_trial_len=10.0)` behavior exactly, then additionally require wheel and whisker alignment because the requested joint decoder needs both outputs on every kept trial.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from spike times and spike cluster identities loaded from `spikes.times.npy` and `spikes.clusters.npy`, plus cluster metadata from `clusters.metrics.pqt`, `clusters.channels.npy`, and `channels.brainLocationIds_ccf_2017.npy` to recover QC labels and anatomical acronyms.

ii.
```python
spikes = {
    "times": np.load(sort_dir / "spikes.times.npy"),
    "clusters": np.load(sort_dir / "spikes.clusters.npy").astype(np.int32),
}
clusters_labeled = pd.read_parquet(sort_dir / "clusters.metrics.pqt")
cluster_channels = np.load(sort_dir / "clusters.channels.npy").astype(np.int64)
channel_region_ids = np.load(sort_dir / "channels.brainLocationIds_ccf_2017.npy").astype(
    np.int64
)
...
clusters_labeled["acronym"] = BRAIN_REGIONS.id2acronym(cluster_region_ids)
```

iii. Step 2 notes describe the same raw files as the ephys inputs, and Step 5 maps `spikes.times`, `spikes.clusters`, and cluster metadata to the target `neural` field.

## 2-b. How is the `neural` data processed?

i. Probe-level spike trains are optionally QC-filtered, merged across probes with `merge_probes`, and then binned into per-trial spike-count matrices using a `stimOn_times + [-0.5, 1.5]` window and `20 ms` bins. The per-trial output is transposed to `(n_neurons, n_timepoints)` for the target format.

ii.
```python
spikes, clusters = merge_probes(spikes_list, clusters_list)
neural_dict = {
    "spike_times": spikes["times"],
    "spike_clusters": spikes["clusters"],
    "cluster_regions": clusters["acronym"].to_numpy(),
}
```

```python
intervals = np.vstack(
    [
        trials_df[PARAMS["align_time"]] + PARAMS["time_window"][0],
        trials_df[PARAMS["align_time"]] + PARAMS["time_window"][1],
    ]
).T
...
binned_array = get_spike_data_per_interval(...)
return np.array([x.T for x in binned_array], dtype=np.float32)
```

iii. In Step 1 the agent documented the reference path `prepare_data -> merge_probes -> bin_spiking_data`, and in Step 5 it explicitly chose to keep the reference code's `stimOn`-aligned `2 s` / `20 ms` representation.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The agent applies cluster QC at load time by keeping only clusters with `label >= 1`. If no such clusters remain for a session, the session is discarded.

ii.
```python
def load_spiking_data_current(..., qc: int | None = None):
    ...
    if qc is None:
        return spikes, clusters_labeled

    iok = clusters_labeled["label"] >= qc
    selected_clusters = clusters_labeled[iok].copy()
    ...
    return selected_spikes, selected_clusters
```

```python
for pid, probe_name in probe_info:
    spikes, clusters = load_spiking_data_current(
        session_path,
        probe_name=probe_name,
        qc=1,
    )
...
if not spikes_list:
    raise RuntimeError("No good clusters after QC filtering")
```

iii. Step 4 and Step 5 of `CONVERSION_NOTES.md` justify this as a paper-aligned choice: the notes say the reference caching code loads all clusters but the papers discuss well-isolated units, so the agent chose `label >= 1` as the closest executable approximation to the paper’s curation intent.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to `stimOn_times`. For each trial, the script bins spikes from `-0.5 s` to `+1.5 s` relative to stimulus onset.

ii.
```python
PARAMS = {
    "interval_len": 2.0,
    "binsize": 0.02,
    "align_time": "stimOn_times",
    "time_window": (-0.5, 1.5),
}
```

```python
intervals = np.vstack(
    [
        trials_df[PARAMS["align_time"]] + PARAMS["time_window"][0],
        trials_df[PARAMS["align_time"]] + PARAMS["time_window"][1],
    ]
).T
```

iii. Step 4 says the agent resolved the paper/code discrepancy by prioritizing the executable reference code's common stimulus-onset alignment because the task explicitly asked for "Temporally align based on stimulus onset."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use `20 ms` bins over a `2 s` window, giving `100` bins per trial. No later temporal rebinning is applied.

ii.
```python
PARAMS = {
    "interval_len": 2.0,
    "binsize": 0.02,
    "align_time": "stimOn_times",
    "time_window": (-0.5, 1.5),
}
```

```python
def build_time_axis() -> np.ndarray:
    start, end = PARAMS["time_window"]
    binsize = PARAMS["binsize"]
    n_bins = int(math.ceil((end - start) / binsize))
    return np.linspace(start + binsize, end, n_bins, dtype=np.float32)
```

iii. The Step 1 notes identify `binsize=0.02` in `0_data_caching.py`, and Step 4 says the agent kept that code-level choice despite a paper text mentioning `50 ms` bins for some analyses.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. This input is not read from a raw stored signal. It is derived from the chosen alignment event `stimOn_times` together with the fixed `time_window` and `binsize`, producing a shared relative-time axis for every trial.

ii.
```python
PARAMS = {
    "binsize": 0.02,
    "align_time": "stimOn_times",
    "time_window": (-0.5, 1.5),
}

def build_time_axis() -> np.ndarray:
    start, end = PARAMS["time_window"]
    binsize = PARAMS["binsize"]
    n_bins = int(math.ceil((end - start) / binsize))
    return np.linspace(start + binsize, end, n_bins, dtype=np.float32)
```

iii. Step 5 explicitly says this input is "not present in reference code" and is instead derived directly from the adopted alignment grid because it was required by the decoder specification.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The script computes a 100-point vector of bin end times from `-0.48` to `1.5` seconds relative to stimulus onset and repeats that same vector for every kept trial.

ii.
```python
time_axis = build_time_axis()
...
input_trials = [
    np.vstack(
        [
            time_axis,
            np.full(time_axis.shape[0], block_num_all[i], dtype=np.float32),
        ]
    ).astype(np.float32, copy=False)
    for i in keep_idx
]
```

iii. Step 5 notes say the agent chose the "same 20 ms trial grid as neural/activity outputs" for this channel so all modalities share one common trial tensor.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. It is exactly co-registered to the neural binning grid: both use the same `PARAMS["align_time"]`, `time_window`, `binsize`, and the same 100-bin trial length.

ii.
```python
time_axis = build_time_axis()
...
neural_trials = [binned_spikes[i].T.astype(np.float32, copy=False) for i in keep_idx]
input_trials = [
    np.vstack([time_axis, np.full(time_axis.shape[0], block_num_all[i], dtype=np.float32)])
    for i in keep_idx
]
```

iii. Step 10 says the agent independently reconstructed the first kept trial and verified the converted input matrix matched the raw stimulus-aligned time axis exactly.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from `trials_df["probabilityLeft"]`, treating each change in prior probability as the start of a new block.

ii.
```python
def trial_number_in_block(probability_left: np.ndarray) -> np.ndarray:
    probability_left = np.asarray(probability_left, dtype=float)
    ...
```

```python
prob_left_all = trials_df["probabilityLeft"].to_numpy(dtype=float)
block_num_all = trial_number_in_block(prob_left_all)
```

iii. Step 5 maps `trials.probabilityLeft` block structure to this input and says the counter should reset whenever `probabilityLeft` changes.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The code scans trials in original session order, starts the counter at `1`, resets to `1` whenever `probabilityLeft` changes, increments otherwise, and only afterward subsets to retained trials. It then repeats the scalar block count across all 100 time bins of each kept trial.

ii.
```python
for idx, value in enumerate(probability_left):
    if idx == 0 or not np.isclose(value, prev):
        current = 1
    else:
        current += 1
    out[idx] = current
    prev = value
```

```python
np.full(time_axis.shape[0], block_num_all[i], dtype=np.float32)
```

iii. The Step 5 "Key Decisions" section says the agent intentionally computed block number before dropping invalid trials so it would preserve the true experimental progression.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. Choice is derived from the raw trial-table column `trials_df["choice"]`.

ii.
```python
choice_all = trials_df["choice"].to_numpy()
...
choice_labels = choice_to_label(choice_all[keep_idx])
```

iii. Step 5 lists `trials.choice` as the source variable for `output[0]`.

## 5-b. What processing is involved in computing `output` *Choice*?

i. Trials with `choice == 0` are already filtered out. The remaining raw IBL choice codes are checked to be only `-1` or `1`; then raw `+1` is interpreted as left and encoded as `0`, while raw `-1` is interpreted as right and encoded as `1`. The scalar label is repeated across all time bins of the trial.

ii.
```python
def choice_to_label(choice_values: np.ndarray) -> np.ndarray:
    choice_values = np.asarray(choice_values)
    if not np.all(np.isin(choice_values, [-1, 1])):
        ...
    # Official IBL docs: choice == -1 means chose right, choice == +1 means chose left.
    return (choice_values == -1).astype(np.int64)
```

```python
np.full(T, session.choice_labels[trial_idx], dtype=np.int64)
```

iii. Step 5 explicitly calls out the semantic remapping from raw IBL wheel-direction coding to the task-required left/right label coding.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It is derived directly from `trials_df["probabilityLeft"]`.

ii.
```python
prob_left_all = trials_df["probabilityLeft"].to_numpy(dtype=float)
...
prior_labels = prior_to_label(prob_left_all[keep_idx])
```

iii. Step 5 lists `probabilityLeft` as the source for both the block-count input and the prior output.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The code rounds values to one decimal place, verifies they are among `0.2`, `0.5`, or `0.8`, maps them to labels `0`, `1`, and `2`, and repeats the label across all time bins in the trial.

ii.
```python
PRIOR_MAP = {
    0.2: 0,
    0.5: 1,
    0.8: 2,
}

def prior_to_label(probability_left: np.ndarray) -> np.ndarray:
    rounded = np.round(np.asarray(probability_left, dtype=float), 1)
    unknown = sorted(set(rounded.tolist()) - set(PRIOR_MAP))
    ...
    return np.array([PRIOR_MAP[x] for x in rounded], dtype=np.int64)
```

```python
np.full(T, session.prior_labels[trial_idx], dtype=np.int64)
```

iii. Step 5 says the agent used `probabilityLeft` directly and encoded the three observed task values as the required categorical prior output.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. Wheel speed is derived from `_ibl_wheel.timestamps.npy` and `_ibl_wheel.position.npy`.

ii.
```python
wheel_timestamps = np.load(find_latest_file(alf_path, "**/_ibl_wheel.timestamps.npy"))
wheel_position = np.load(find_latest_file(alf_path, "**/_ibl_wheel.position.npy"))
```

iii. Step 2 notes identify those wheel arrays as the relevant raw files, and Step 5 maps the wheel trace loaded by the reference behavior loader to this output.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The script interpolates wheel position to a uniform `1000 Hz` grid, computes filtered wheel velocity, takes its absolute value, extracts each trial's `stimOn`-aligned `[-0.5, 1.5] s` segment, and linearly interpolates that segment onto the common 100-bin decoder grid.

ii.
```python
position, times = interpolate_position(wheel_timestamps, wheel_position, freq=1000)
velocity, _ = velocity_filtered(position, fs=1000, corner_frequency=20, order=8)
return {
    "times": np.asarray(times, dtype=np.float32),
    "values": np.abs(np.asarray(velocity, dtype=np.float32)),
}
```

```python
x_interp = np.linspace(interval_begs[idx] + binsize, interval_ends[idx], n_bins)
fn = interp1d(t_seg, v_seg, kind="linear", fill_value="extrapolate")
y_interp = fn(x_interp)
```

iii. Step 1 notes say the reference code uses `abs(wheel velocity)`. Step 6 says the agent reimplemented the local wheel path directly because it wanted the same variables without remote `SessionLoader` overhead.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. After all sessions are processed, the script concatenates all retained wheel-speed samples across all retained trials and computes global tertile cutpoints. Each per-trial wheel trace is then discretized into three bins with `np.digitize`.

ii.
```python
wheel_values = np.concatenate(
    [np.concatenate(session.wheel_continuous) for session in processed_sessions]
)
...
q1, q2 = np.quantile(values, [1 / 3, 2 / 3])
thresholds[name] = (float(q1), float(q2))
```

```python
def discretize(values: np.ndarray, thresholds: tuple[float, float]) -> np.ndarray:
    q1, q2 = thresholds
    return np.digitize(values, bins=np.array([q1, q2]), right=False).astype(np.int64)
```

iii. Step 5 calls this a task-driven deviation: the papers/reference code keep wheel speed continuous, but the decoder instructions required three categories, so the agent chose global tertiles.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. Wheel speed is aligned to the same stimulus-onset event and the same `[-0.5, 1.5]` / `20 ms` grid used for spikes.

ii.
```python
PARAMS = {
    "binsize": 0.02,
    "align_time": "stimOn_times",
    "time_window": (-0.5, 1.5),
}
...
wheel_traces, wheel_mask = align_continuous_behavior(session_path, "wheel-speed", trials_df)
```

```python
align_times = trials_df[PARAMS["align_time"]].to_numpy()
interval_begs = align_times + start
interval_ends = align_times + end
```

iii. Step 4 says this was a deliberate choice to honor the explicit task instruction to align everything to stimulus onset, even though the methods paper also describes first-movement alignment for dynamic behaviors.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. Whisker motion energy is derived from camera timestamps and motion-energy arrays, preferring `_ibl_leftCamera.times.npy` with `leftCamera.ROIMotionEnergy.npy` and falling back to the right camera equivalents if left-camera loading fails.

ii.
```python
if target == "left-whisker-motion-energy":
    times = np.load(find_latest_file(alf_path, "**/_ibl_leftCamera.times.npy"))
    values = np.load(find_latest_file(alf_path, "**/leftCamera.ROIMotionEnergy.npy"))
...
if target == "right-whisker-motion-energy":
    times = np.load(find_latest_file(alf_path, "**/_ibl_rightCamera.times.npy"))
    values = np.load(find_latest_file(alf_path, "**/rightCamera.ROIMotionEnergy.npy"))
```

```python
if behavior_name == "whisker-motion-energy":
    target = load_target_behavior_current(session_path, "left-whisker-motion-energy")
    if target.get("skip"):
        target = load_target_behavior_current(session_path, "right-whisker-motion-energy")
```

iii. Step 1 and Step 5 both state that left whisker motion energy is preferred and right whisker is used as fallback, matching the reference helper behavior.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The script loads the raw motion-energy trace, trims the camera time array from the front if there are extra timestamps, extracts per-trial stimulus-aligned windows, and linearly interpolates each trial onto the common 100-bin grid.

ii.
```python
if times.shape[0] > values.shape[0]:
    times = times[-values.shape[0] :]
return {
    "times": np.asarray(times, dtype=np.float32),
    "values": np.asarray(values, dtype=np.float32),
}
```

```python
idxs_beg = np.searchsorted(target_times, interval_begs, side="right")
idxs_end = np.searchsorted(target_times, interval_ends, side="left")
...
fn = interp1d(t_seg, v_seg, kind="linear", fill_value="extrapolate")
y_interp = fn(x_interp)
```

iii. Step 6 says the agent reimplemented local ALF readers for whisker motion energy to match the reference variable while avoiding remote ONE overhead.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. As with wheel speed, all retained aligned whisker samples are concatenated across the converted dataset and global tertile thresholds are computed, then each trial trace is discretized with `np.digitize`.

ii.
```python
whisker_values = np.concatenate(
    [np.concatenate(session.whisker_continuous) for session in processed_sessions]
)
...
q1, q2 = np.quantile(values, [1 / 3, 2 / 3])
```

```python
discretize(
    session.whisker_continuous[trial_idx],
    thresholds["whisker_motion_energy"],
)
```

iii. Step 5 justifies this the same way as wheel speed: continuous in the reference pipeline, discretized only because the task specification required categorical outputs.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. It is aligned to stimulus onset and interpolated to the same 100-bin `20 ms` grid as the spikes.

ii.
```python
traces, good_mask, _ = get_behavior_per_interval_current(
    target.get("times"),
    target.get("values"),
    trials_df=trials_df,
    allow_nans=True,
)
```

```python
align_times = trials_df[PARAMS["align_time"]].to_numpy()
interval_begs = align_times + start
interval_ends = align_times + end
```

iii. Step 4 says the agent intentionally used one common stimulus-onset grid for all requested variables, even though the methods paper describes a different alignment for dynamic behaviors.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing or malformed data are mostly handled by exclusion. Trials with missing key task events are removed by the trial mask. Behavior trials are marked bad if the requested interval is missing, contains NaNs when NaNs are disallowed, starts too late, or ends too early. Left whisker failures fall back to right whisker. If camera times are longer than motion-energy values, the extra leading timestamps are dropped. Sessions with fewer than two valid trials or no good clusters are skipped. Transient worker failures are retried up to three times.

ii.
```python
for event in [
    "stimOn_times", "choice", "feedback_times",
    "probabilityLeft", "firstMovement_times", "feedbackType",
]:
    query_parts.append(f"{event}.isnull()")
```

```python
if len(v_seg) == 0:
    reasons[idx] = "target data not present"
    continue
...
if np.abs(interval_begs[idx] - t_seg[0]) > binsize:
    reasons[idx] = "target data starts too late"
    continue
if np.abs(interval_ends[idx] - t_seg[-1]) > binsize:
    reasons[idx] = "target data ends too early"
    continue
```

```python
if target.get("skip"):
    target = load_target_behavior_current(session_path, "right-whisker-motion-energy")
...
if keep_idx.size < 2:
    raise RuntimeError(f"Only {keep_idx.size} valid trials after filtering")
```

iii. Step 10 says the remaining all-zero neural trials were checked against raw spikes and kept because they were genuine sparse windows, not conversion bugs. The rest of the notes consistently describe missing-data handling as filtering rather than imputation.

## 10-a. What are the most time-consuming steps of the code?

i. The expensive steps are per-session spike binning across all trials, per-trial wheel and whisker interpolation, and high-volume parallel session I/O during full conversion. The notes also identify disk contention from too many session workers as a practical bottleneck.

ii.
```python
binned_spikes = bin_spiking_data_current(reg_clu_ids, neural_dict, trials_df)
wheel_traces, wheel_mask = align_continuous_behavior(session_path, "wheel-speed", trials_df)
whisker_traces, whisker_mask = align_continuous_behavior(
    session_path, "whisker-motion-energy", trials_df
)
```

```python
with ProcessPoolExecutor(max_workers=args.session_workers) as executor:
    future_map = {
        executor.submit(process_one_session_worker, row_dict, time_axis): row_dict
        for row_dict in records
    }
```

iii. Step 6 and Step 9 explicitly discuss spike/behavior loading and full-run worker contention as the dominant runtime costs, and the script records `trials_s`, `spikes_s`, and `behavior_s` for each session.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Several Python loops remain: trial-by-trial spike binning in `get_spike_data_per_interval`, trial-by-trial behavior interpolation in `get_behavior_per_interval_current`, the sequential scan in `trial_number_in_block`, and the per-trial output assembly/discretization loop in `assemble_dataset`.

ii.
```python
for interval_idx, (t_beg, t_end) in enumerate(zip(interval_begs, interval_ends)):
    idxs_t = (times >= t_beg) & (times < t_end)
    ...
```

```python
for idx, (t_seg, v_seg) in enumerate(zip(target_times_list, target_vals_list)):
    ...
```

```python
for trial_idx in range(session.kept_trial_count):
    T = session.neural[trial_idx].shape[1]
    outputs_session.append(
        np.vstack([...])
    )
```

iii. The agent notes some performance work in Step 6, but the final script still reimplements reference-like logic with explicit Python loops rather than trying to fully vectorize interval extraction or discretized output assembly.

## 10-c. What processing does the code repeat multiple times?

i. It repeatedly interpolates behavior one trial at a time for both wheel and whisker, repeatedly creates full-length repeated arrays for static per-trial labels (`choice`, `prior`, block number), and scans every trial twice for dynamic outputs: once to store continuous traces and again to discretize them during final assembly.

ii.
```python
wheel_aligned = [np.asarray(wheel_traces[i], dtype=np.float32) for i in keep_idx]
whisker_aligned = [np.asarray(whisker_traces[i], dtype=np.float32) for i in keep_idx]
...
thresholds = compute_thresholds(processed_sessions)
...
discretize(
    session.wheel_continuous[trial_idx],
    thresholds["wheel_speed"],
)
```

```python
np.full(time_axis.shape[0], block_num_all[i], dtype=np.float32)
np.full(T, session.choice_labels[trial_idx], dtype=np.int64)
np.full(T, session.prior_labels[trial_idx], dtype=np.int64)
```

iii. Step 6 describes the pipeline as two-stage by design: first gather continuous wheel/whisker traces, then compute global thresholds and assemble the final categorical dataset.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script stores `cluster_good` metadata even though all retained clusters were already QC-filtered, retains full continuous wheel and whisker traces only to later replace them with discretized outputs, and expands static variables (`choice`, `prior`, and block number) into full-length time series even though they are constant within each trial. It also constructs detailed skip-session metadata and timing fields that are useful for auditing but not for the decoder itself.

ii.
```python
meta = {
    "cluster_regions": list(clusters["acronym"]),
    "good_clusters": (clusters["label"] >= 1).to_numpy(dtype=np.int8),
    "cluster_qc": {k: np.asarray(v) for k, v in clusters.to_dict("list").items()},
}
```

```python
wheel_continuous=wheel_aligned,
whisker_continuous=whisker_aligned,
...
np.full(T, session.choice_labels[trial_idx], dtype=np.int64),
np.full(T, session.prior_labels[trial_idx], dtype=np.int64),
```

iii. The notes frame most of this as pragmatic rather than accidental: continuous traces are kept until thresholding is known, and the repeated time-series form was chosen to make all trial tensors uniform for the downstream decoder.
