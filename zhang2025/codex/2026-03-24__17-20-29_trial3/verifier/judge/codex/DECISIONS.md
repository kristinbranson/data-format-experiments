# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent does not use the ONE API as the primary loader. It loads a deterministic session roster from `code/code_zhang2025/data/bwm_release.csv`, derives each session's local cache path under `data/one_cache`, and then reads ALF/parquet/numpy files directly for trials, behavior, and spikes. Probe membership is taken from the release CSV.

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
        ONE_CACHE_DIR
        / row["lab"]
        / "Subjects"
        / row["subject"]
        / row["date"]
        / f"{int(row['session_number']):03d}"
    )
```

iii. In `CONVERSION_NOTES.md`, the agent says it chose a deterministic roster from `bwm_release.csv` and "direct local ALF/parquet/numpy readers" to avoid remote metadata resolution, dependency mismatches, and startup latency from ONE/SessionLoader.

## 1-b. How are the data split into subjects?

i. Subjects come directly from the `subject` column in the release roster. After successful conversion, the agent builds `subjects` by preserving first appearance order and builds `subject_idx` from each processed session's subject.

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
```

iii. The notes justify this as using the canonical release roster metadata rather than deriving subject IDs from paths.

## 1-c. How are the data split into sessions?

i. A session is one unique `eid` from `bwm_release.csv`. The agent drops duplicate probe rows to one row per `eid`, processes sessions one by one, and finally sorts successful results back into release order using `release_index`.

ii.
```python
sessions_df = (
    bwm_df[["eid", "lab", "subject", "date", "session_number"]]
    .drop_duplicates(subset=["eid"], keep="first")
    .reset_index(drop=True)
)
```

```python
processed_sessions.sort(key=lambda session: session.release_index)
```

iii. The notes describe this as a deterministic session roster matching the reference release CSV.

## 1-d. How are the data split into trials?

i. Trials are read from the session trials parquet table, treated as one row per trial, and all per-trial neural/input/output arrays are indexed from that table. Retained trials are selected by `keep_idx`.

ii.
```python
trials_path = find_latest_file(session_path / "alf", "**/_ibl_trials.table.pqt")
trials = pd.read_parquet(trials_path).copy()
```

```python
keep_idx = np.flatnonzero(keep_mask)
neural_trials = [binned_spikes[i].T.astype(np.float32, copy=False) for i in keep_idx]
```

iii. No separate justification is given beyond following the ALF trials table structure.

## 1-e. How are trials filtered based on quality controls?

i. The agent first builds a trial mask from reaction time, maximum trial length, missing required event fields, and no-choice trials. It then intersects that with wheel and whisker alignment masks, and drops sessions with fewer than two retained trials.

ii.
```python
if min_rt is not None:
    query_parts.append(f"(firstMovement_times - stimOn_times < {min_rt})")
if max_rt is not None:
    query_parts.append(f"(firstMovement_times - stimOn_times > {max_rt})")
if max_trial_len is not None:
    query_parts.append(f"(feedback_times - goCue_times > {max_trial_len})")
...
if exclude_nochoice:
    query_parts.append("(choice == 0)")
```

```python
keep_mask = np.asarray(trials_mask, dtype=bool) & wheel_mask & whisker_mask
keep_idx = np.flatnonzero(keep_mask)
if keep_idx.size < 2:
    raise RuntimeError(f"Only {keep_idx.size} valid trials after filtering")
```

iii. `CONVERSION_NOTES.md` says this reproduces the reference helper's trial mask, including `max_trial_len=10.0`, and also requires valid wheel and whisker coverage for the decoder task.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural array is ultimately built from spike times and spike cluster IDs. Cluster metadata and channel brain-location IDs are also loaded to assign QC labels and region acronyms.

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
```

iii. The notes say this matches electrophysiology reference processing: spike times plus spike-sorted cluster metadata, not imaging-style fluorescence processing.

## 2-b. How is the `neural` data processed?

i. Per probe, the agent loads spikes and clusters, filters to QC-passing clusters, merges probes with `merge_probes`, bins spikes into stimulus-aligned 20 ms bins, and stores each trial as `(n_neurons, n_timepoints)`. It keeps spike counts, not Hz firing rates.

ii.
```python
spikes, clusters = load_spiking_data_current(
    session_path,
    probe_name=probe_name,
    qc=1,
)
...
spikes, clusters = merge_probes(spikes_list, clusters_list)
```

```python
binned_tmp, _, cluster_idxs = bincount2D(
    times_curr, clust_curr, xbin=binsize, xlim=[t_beg, t_end]
)
...
return np.array([x.T for x in binned_array], dtype=np.float32)
```

iii. The notes justify the merged-probe, stimulus-onset-aligned 2 s / 20 ms grid as matching the executable reference code. The metadata text explicitly describes the neural data as "spike counts."

## 2-c. How is the `neural` data filtered based on quality controls?

i. The agent keeps only clusters with `label >= 1` by passing `qc=1` into the spike loader. It does not apply an explicit `void` region exclusion before keeping neurons.

ii.
```python
iok = clusters_labeled["label"] >= qc
selected_clusters = clusters_labeled[iok].copy()
spike_idx, ib = ismember(spikes["clusters"], selected_clusters.index.to_numpy())
...
return selected_spikes, selected_clusters
```

```python
spikes, clusters = load_spiking_data_current(
    session_path,
    probe_name=probe_name,
    qc=1,
)
```

iii. The notes justify `label >= 1` as paper-aligned well-isolated neuron curation. They separately note a Beryl remap later, but they do not say they excluded `void` neurons at load time.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The neural data are aligned to `stimOn_times`. For every trial, the binning interval is `[stimOn_times - 0.5, stimOn_times + 1.5]`.

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

iii. The notes say the task explicitly required stimulus-onset alignment, so the agent followed the reference code's `stimOn_times` alignment rather than target-specific alignments from the paper.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 20 ms bins over a 2 s window, for 100 bins per trial. Raw spikes are binned once; there is no extra rebinning stage.

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
n_bins = int(np.ceil(interval_len / binsize))
```

iii. The notes explicitly state that the agent preserved the reference code's 2 s / 20 ms grid.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is not read from a raw column directly. The agent derives it from the chosen stimulus-aligned time grid defined by `time_window`, `binsize`, and the use of `stimOn_times` as the alignment event.

ii.
```python
def build_time_axis() -> np.ndarray:
    start, end = PARAMS["time_window"]
    binsize = PARAMS["binsize"]
    n_bins = int(math.ceil((end - start) / binsize))
    return np.linspace(start + binsize, end, n_bins, dtype=np.float32)
```

iii. The notes describe this as a derived common bin axis relative to `stimOn_times`, added because the decoder task required an explicit time input.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The agent computes a fixed 100-sample vector of bin end times from `-0.48` to `1.5` s and reuses it for every retained trial.

ii.
```python
def build_time_axis() -> np.ndarray:
    start, end = PARAMS["time_window"]
    binsize = PARAMS["binsize"]
    n_bins = int(math.ceil((end - start) / binsize))
    # Match the reference behavior interpolation grid: bin end times.
    return np.linspace(start + binsize, end, n_bins, dtype=np.float32)
```

iii. The inline comment gives the justification: the agent wanted the input channel to match its behavior interpolation grid, so it used bin end times.

## 3-c. How is `input` *Time since stimulus onset* aligned with the neural data?

i. The same precomputed `time_axis` is inserted into every trial input matrix, while neural and behavior data are also built on the same `stimOn_times` window and 20 ms bin count.

ii.
```python
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

```python
x_interp = np.linspace(interval_begs[idx] + binsize, interval_ends[idx], n_bins)
```

iii. The notes justify this as a uniform common stimulus-onset-aligned grid for neural, inputs, and outputs.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from `trials_df["probabilityLeft"]`, with a new block declared whenever `probabilityLeft` changes.

ii.
```python
def trial_number_in_block(probability_left: np.ndarray) -> np.ndarray:
    probability_left = np.asarray(probability_left, dtype=float)
    ...
    for idx, value in enumerate(probability_left):
        if idx == 0 or not np.isclose(value, prev):
            current = 1
```

iii. The notes say block identity is recovered from `probabilityLeft` because the trial table does not provide a separate block counter.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The agent loops over all session trials in original order, resets the counter when `probabilityLeft` changes, counts trials starting at 1, and then subsets those values to retained trials.

ii.
```python
current = 0
prev = None
for idx, value in enumerate(probability_left):
    if idx == 0 or not np.isclose(value, prev):
        current = 1
    else:
        current += 1
    out[idx] = current
```

```python
prob_left_all = trials_df["probabilityLeft"].to_numpy(dtype=float)
block_num_all = trial_number_in_block(prob_left_all)
```

iii. `CONVERSION_NOTES.md` says this is computed on the original trial order before filtering so the block position reflects the animal's real progression through the session.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. Choice is derived from the raw `choice` column in the trials table.

ii.
```python
choice_all = trials_df["choice"].to_numpy()
choice_labels = choice_to_label(choice_all[keep_idx])
```

iii. The notes describe this as using the filtered trials table directly and then remapping to the requested left/right labels.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The agent first filters away `choice == 0` trials, then maps `+1` to left (`0`) and `-1` to right (`1`) with a boolean comparison.

ii.
```python
if not np.all(np.isin(choice_values, [-1, 1])):
    bad = np.unique(choice_values[~np.isin(choice_values, [-1, 1])])
    raise ValueError(f"Unexpected choice values after filtering: {bad.tolist()}")
return (choice_values == -1).astype(np.int64)
```

iii. The comment in `choice_to_label` cites the IBL convention and explains the left/right remapping.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It is derived from the raw `probabilityLeft` trial variable.

ii.
```python
prob_left_all = trials_df["probabilityLeft"].to_numpy(dtype=float)
prior_labels = prior_to_label(prob_left_all[keep_idx])
```

iii. The notes say the prior output uses `probabilityLeft` directly because it already encodes the block prior.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The agent rounds `probabilityLeft` to one decimal place, validates that only `0.2`, `0.5`, and `0.8` remain, and maps them to `0`, `1`, and `2`.

ii.
```python
PRIOR_MAP = {
    0.2: 0,
    0.5: 1,
    0.8: 2,
}

def prior_to_label(probability_left: np.ndarray) -> np.ndarray:
    rounded = np.round(np.asarray(probability_left, dtype=float), 1)
    ...
    return np.array([PRIOR_MAP[x] for x in rounded], dtype=np.int64)
```

iii. The notes justify this as the task-specified categorical encoding of the three block priors.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. Wheel speed is derived from raw wheel timestamps and wheel position arrays.

ii.
```python
wheel_timestamps = np.load(find_latest_file(alf_path, "**/_ibl_wheel.timestamps.npy"))
wheel_position = np.load(find_latest_file(alf_path, "**/_ibl_wheel.position.npy"))
```

iii. The notes say this matches the reference code's wheel loading path, but implemented with direct local reads instead of `SessionLoader`.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The agent interpolates wheel position to 1000 Hz, computes filtered velocity, takes absolute value to get speed, linearly interpolates that speed into each trial's stimulus-aligned 100-bin window, and keeps the continuous traces until the later discretization step.

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
fn = interp1d(t_seg, v_seg, kind="linear", fill_value="extrapolate")
y_interp = fn(x_interp)
```

iii. The notes justify this as matching the IBL wheel-processing functions used by the reference code before applying the task-required discretization.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. The agent computes two global tertile thresholds across all retained wheel samples from all converted sessions, then bins each per-trial wheel trace with `np.digitize` into `low`, `medium`, and `high`.

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

iii. `CONVERSION_NOTES.md` explicitly says the agent chose global tertiles "consistently across all sessions" as a task-driven discretization rule.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. Wheel speed is aligned on the same `stimOn_times + [-0.5, 1.5]` window used for the neural data, and interpolated onto the same 100-bin grid.

ii.
```python
align_times = trials_df[PARAMS["align_time"]].to_numpy()
interval_begs = align_times + start
interval_ends = align_times + end
...
x_interp = np.linspace(interval_begs[idx] + binsize, interval_ends[idx], n_bins)
```

```python
wheel_traces, wheel_mask = align_continuous_behavior(session_path, "wheel-speed", trials_df)
```

iii. The notes justify stimulus-onset alignment for wheel as required by the decoder task, even though the paper sometimes uses movement-aligned behavior analyses.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It is derived from `leftCamera.ROIMotionEnergy.npy` plus `_ibl_leftCamera.times.npy`, with fallback to the right camera equivalents if left-camera whisker motion energy is unavailable.

ii.
```python
if target == "left-whisker-motion-energy":
    times = np.load(find_latest_file(alf_path, "**/_ibl_leftCamera.times.npy"))
    values = np.load(find_latest_file(alf_path, "**/leftCamera.ROIMotionEnergy.npy"))
```

```python
if behavior_name == "whisker-motion-energy":
    target = load_target_behavior_current(session_path, "left-whisker-motion-energy")
    if target.get("skip"):
        target = load_target_behavior_current(session_path, "right-whisker-motion-energy")
```

iii. The notes say this left-then-right fallback matches the reference logic.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The agent uses the released motion-energy trace directly, trims extra camera timestamps if needed so times and values have equal length, and linearly interpolates the trace into each trial's 100-bin stimulus-aligned window.

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
fn = interp1d(t_seg, v_seg, kind="linear", fill_value="extrapolate")
y_interp = fn(x_interp)
```

iii. The notes justify this as preserving the reference raw whisker signal and only adding the alignment/discretization required by the decoder dataset.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. The agent uses the same global-tertile strategy as for wheel speed: compute thresholds over all retained whisker samples across sessions, then digitize each trial trace into three bins.

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

iii. `CONVERSION_NOTES.md` says this was chosen as a consistent dataset-wide discretization rule after alignment.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Whisker motion energy is aligned to the same stimulus-onset trial windows and interpolated onto the same 100-bin grid as the neural data.

ii.
```python
whisker_traces, whisker_mask = align_continuous_behavior(
    session_path, "whisker-motion-energy", trials_df
)
```

```python
align_times = trials_df[PARAMS["align_time"]].to_numpy()
interval_begs = align_times + start
interval_ends = align_times + end
```

iii. The notes explicitly say wheel and whisker were put onto the common stimulus-onset grid because the user requested one simultaneously aligned dataset.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing or unusable data are mostly handled by dropping trials or sessions. Behavior-loading failures return `skip`, wheel/whisker windows that do not cover a trial mark that trial bad, camera time/value length mismatches are partly repaired by trimming, sessions with no good clusters or fewer than two valid trials are skipped, and transient full-session failures are retried.

ii.
```python
except BaseException as exc:  # noqa: BLE001
    return {"times": None, "values": None, "skip": True, "error": str(exc)}
```

```python
if times.shape[0] > values.shape[0]:
    times = times[-values.shape[0] :]
...
if keep_idx.size < 2:
    raise RuntimeError(f"Only {keep_idx.size} valid trials after filtering")
```

iii. The notes justify this as robustness for the cached local dataset: drop trials/sessions with missing aligned data, and repair the specific camera timestamp overhang case rather than failing the whole conversion.

## 10-a. What are the most time-consuming steps of the code?

i. The most expensive work is session-level data loading and per-session spike processing, especially reading spike arrays from disk. The notes also say overly parallel full runs caused severe disk contention.

ii.
```python
spikes = {
    "times": np.load(sort_dir / "spikes.times.npy"),
    "clusters": np.load(sort_dir / "spikes.clusters.npy").astype(np.int32),
}
```

```python
with ProcessPoolExecutor(max_workers=args.session_workers) as executor:
    future_map = {
        executor.submit(process_one_session_worker, row_dict, time_axis): row_dict
        for row_dict in records
    }
```

iii. `CONVERSION_NOTES.md` explicitly says remote loading was slow, local spike reads dominate, and high worker counts caused disk contention.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The code still has several per-trial loops that could be vectorized: spike binning over intervals, behavior interpolation over intervals, block counting over trials, and output assembly over trials.

ii.
```python
for interval_idx, (t_beg, t_end) in enumerate(zip(interval_begs, interval_ends)):
    idxs_t = (times >= t_beg) & (times < t_end)
```

```python
for idx, (t_seg, v_seg) in enumerate(zip(target_times_list, target_vals_list)):
    ...
for idx, value in enumerate(probability_left):
    ...
for trial_idx in range(session.kept_trial_count):
```

iii. The agent does not give a separate justification here; the loops remain in place for clarity and because its optimization effort focused on switching to local file reads and session-level parallelism.

## 10-c. What processing does the code repeat multiple times?

i. The full-conversion path can process a session twice if the worker pass fails and the sequential retry pass reruns it. The code also recomputes discretized wheel/whisker bins when assembling outputs after having already built and stored continuous aligned traces.

ii.
```python
except Exception as exc:  # noqa: BLE001
    retry_records.append(row_dict)
...
for idx, row_dict in enumerate(retry_records, start=1):
    ...
    session = process_one_session(row_series, time_axis)
```

```python
outputs_session.append(
    np.vstack(
        [
            ...,
            discretize(session.wheel_continuous[trial_idx], thresholds["wheel_speed"]),
            discretize(
                session.whisker_continuous[trial_idx],
                thresholds["whisker_motion_energy"],
            ),
        ]
    )
)
```

iii. The notes justify the retry path as robustness to transient worker failures; they do not present a separate efficiency justification for the repeated discretization during assembly.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The agent computes and stores several pieces of intermediate metadata that are not written into the final dataset: `cluster_good`, `cluster_qc`, `kept_trial_indices`, `skipped_trial_count`, and per-session timing. It also loads `bwm_df` separately even though only `sessions_df` is used downstream.

ii.
```python
meta = {
    "cluster_regions": list(clusters["acronym"]),
    "good_clusters": (clusters["label"] >= 1).to_numpy(dtype=np.int8),
    "cluster_qc": {k: np.asarray(v) for k, v in clusters.to_dict("list").items()},
}
```

```python
return ProcessedSession(
    ...
    cluster_good=np.asarray(meta["good_clusters"], dtype=np.int8),
    ...
    kept_trial_indices=keep_idx.astype(np.int64),
    skipped_trial_count=int(len(trials_df) - len(keep_idx)),
    timing={...},
)
```

```python
return {
    "neural": neural,
    "input": input_data,
    "output": output_data,
    ...
    "metadata": metadata,
}
```

iii. The notes frame most of this as debugging/sanity-check support, not as required downstream data.
