# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads the canonical session roster from `code/code_zhang2025/data/bwm_release.csv`, deduplicates it by `eid`, attaches per-session probe metadata, and then processes each session by reading local ALF/pykilosort files under `data/one_cache/<lab>/Subjects/<subject>/<date>/<session_number>/`. In full mode it submits all session rows to a `ProcessPoolExecutor` and keeps only sessions that convert successfully.

ii.
```python
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
        ONE_CACHE_DIR
        / row["lab"]
        / "Subjects"
        / row["subject"]
        / row["date"]
        / f"{int(row['session_number']):03d}"
    )
```

iii. In `CONVERSION_NOTES.md`, the agent says the 459-session `bwm_release.csv` used by the reference code should be the canonical roster, and that direct local ALF/parquet/numpy reads were chosen to match the reference variables without depending on slow remote ONE metadata calls.

## 1-b. How are the data split into subjects?

i. Subjects are split by the `subject` field from the release roster. After conversion, the dataset creates an ordered unique list of subject IDs and stores one `subject_idx` per retained session.

ii.
```python
subjects = ordered_unique([session.subject for session in processed_sessions])
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
...
subject_idx.append(subject_to_idx[session.subject])
```

iii. The notes state that `subject` from `bwm_release.csv` should populate `subjects` and `subject_idx`, with ordering matching the converted session order.

## 1-c. How are the data split into sessions?

i. Sessions are split by unique `eid` rows in the release roster. Each session is identified by `eid`, `lab`, `subject`, `date`, and `session_number`, processed independently, and then sorted back to release order via `release_index`.

ii.
```python
sessions_df = (
    bwm_df[["eid", "lab", "subject", "date", "session_number"]]
    .drop_duplicates(subset=["eid"], keep="first")
    .reset_index(drop=True)
)
...
sessions_df = sessions_df.reset_index(names="release_index")
...
processed_sessions.sort(key=lambda session: session.release_index)
```

iii. The notes explicitly call the 459-session roster the base session definition and say session order should be deterministic and fixed by the release table.

## 1-d. How are the data split into trials?

i. Trials are split by rows of the session’s `_ibl_trials.table.pqt` table. Each row defines one trial; per-trial neural and behavior intervals are built by aligning each row’s `stimOn_times` with the common `[-0.5, 1.5]` window, and only kept trials are emitted into the final lists.

ii.
```python
trials_path = find_latest_file(session_path / "alf", "**/_ibl_trials.table.pqt")
trials = pd.read_parquet(trials_path).copy()
...
intervals = np.vstack(
    [
        trials_df[PARAMS["align_time"]] + PARAMS["time_window"][0],
        trials_df[PARAMS["align_time"]] + PARAMS["time_window"][1],
    ]
).T
...
keep_idx = np.flatnonzero(keep_mask)
```

iii. The notes say the converter should use the raw trials table and preserve the reference code’s per-row trial structure before subsetting invalid trials.

## 1-e. How are trials filtered based on quality controls?

i. The agent reproduces the reference trial mask: drop trials with reaction time outside `0.08` to `2.0` s, duration above `10.0` s, NaNs in required event columns, or `choice == 0`. It then further requires successful wheel and whisker alignment for the same trial, and skips sessions with fewer than 2 valid trials.

ii.
```python
def load_trials_and_mask_current(...):
    if min_rt is not None:
        query_parts.append(f"(firstMovement_times - stimOn_times < {min_rt})")
    if max_rt is not None:
        query_parts.append(f"(firstMovement_times - stimOn_times > {max_rt})")
    if max_trial_len is not None:
        query_parts.append(f"(feedback_times - goCue_times > {max_trial_len})")
    for event in [
        "stimOn_times", "choice", "feedback_times",
        "probabilityLeft", "firstMovement_times", "feedbackType",
    ]:
        query_parts.append(f"{event}.isnull()")
    if exclude_nochoice:
        query_parts.append("(choice == 0)")
    ...
    mask = ~trials.eval(query).to_numpy()

keep_mask = np.asarray(trials_mask, dtype=bool) & wheel_mask & whisker_mask
if keep_idx.size < 2:
    raise RuntimeError(f"Only {keep_idx.size} valid trials after filtering")
```

iii. The notes say trial filtering should match `load_trials_and_mask(..., max_trial_len=10.0)` exactly, and then enforce the same final neural/behavior alignment constraint used by `align_spike_behavior`.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from per-probe spike times and spike cluster IDs in `spikes.times.npy` and `spikes.clusters.npy`, plus cluster metadata from `clusters.metrics.pqt`, `clusters.channels.npy`, and `channels.brainLocationIds_ccf_2017.npy` to recover cluster labels and anatomical acronyms.

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
clusters_labeled["acronym"] = BRAIN_REGIONS.id2acronym(cluster_region_ids)
```

iii. The notes frame this as the local-file equivalent of the reference `load_spiking_data` path, using the same underlying spike-sorting outputs while avoiding live ONE access.

## 2-b. How is the `neural` data processed?

i. The agent loads each probe, optionally QC-filters clusters, merges all probes within a session, bins spikes per trial over the common aligned window with `bincount2D`, and stores each kept trial as a `(n_neurons, n_timepoints)` float32 spike-count matrix.

ii.
```python
for pid, probe_name in probe_info:
    spikes, clusters = load_spiking_data_current(
        session_path,
        probe_name=probe_name,
        qc=1,
    )
    ...
spikes, clusters = merge_probes(spikes_list, clusters_list)
...
binned_spikes = bin_spiking_data_current(reg_clu_ids, neural_dict, trials_df)
...
neural_trials = [binned_spikes[i].T.astype(np.float32, copy=False) for i in keep_idx]
```

iii. The notes say this should mirror the reference processing chain `load_spiking_data` -> `merge_probes` -> `bin_spiking_data`, with only the target-format transpose to `(n_neurons, 100)` added at the end.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The agent filters neural data to clusters with `label >= 1` before binning. Sessions with no surviving clusters are skipped, and the good-cluster labels are also stored in metadata.

ii.
```python
if qc is None:
    return spikes, clusters_labeled

iok = clusters_labeled["label"] >= qc
selected_clusters = clusters_labeled[iok].copy()
...
selected_spikes["clusters"] = selected_clusters.index.to_numpy()[ib].astype(np.int32)
...
if not spikes_list:
    raise RuntimeError("No good clusters after QC filtering")
```

iii. In the notes, the agent explicitly justifies this as a paper-informed choice: `prepare_data()` itself does not pass QC, but the paper’s analyses use well-isolated neurons, so the converter applies `label >= 1` to stay near the reported 75,708-neuron scale.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial’s neural data are aligned to `stimOn_times`, as required by the task instructions, using a fixed `[-0.5, 1.5]` second window around stimulus onset.

ii.
```python
PARAMS = {
    "interval_len": 2.0,
    "binsize": 0.02,
    "align_time": "stimOn_times",
    "time_window": (-0.5, 1.5),
}
...
intervals = np.vstack(
    [
        trials_df[PARAMS["align_time"]] + PARAMS["time_window"][0],
        trials_df[PARAMS["align_time"]] + PARAMS["time_window"][1],
    ]
).T
```

iii. The notes call this the main resolved paper-vs-task discrepancy: the executable reference code already uses a common `stimOn`-aligned trial grid, and the task explicitly says to align based on stimulus onset.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 20 ms bins over a 2 s window, yielding 100 time bins per trial. Raw spikes are binned directly into that grid; no extra rebinning stage is applied after spike counting.

ii.
```python
PARAMS = {
    "interval_len": 2.0,
    "binsize": 0.02,
    "align_time": "stimOn_times",
    "time_window": (-0.5, 1.5),
}
...
n_bins = int(np.ceil(interval_len / binsize))
```

iii. The notes state that the agent chose the reference code’s 2 s / 20 ms representation as the common dataset format, despite the methods text mentioning 50 ms bins for some targets.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. The time-since-stimulus input is not read from a raw array. It is derived from the chosen alignment parameters and trial event `stimOn_times`; conceptually it is the relative time axis for the bins in each `stimOn`-aligned trial window.

ii.
```python
def build_time_axis() -> np.ndarray:
    start, end = PARAMS["time_window"]
    binsize = PARAMS["binsize"]
    n_bins = int(math.ceil((end - start) / binsize))
    return np.linspace(start + binsize, end, n_bins, dtype=np.float32)
```

iii. The notes say this channel is a derived decoder input, not something present in the reference cache, so it is built directly from the adopted common trial grid.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The agent computes a fixed float32 vector of bin end times from `-0.48` s to `1.5` s using `np.linspace(start + binsize, end, n_bins)`. That same vector is reused for every kept trial.

ii.
```python
return np.linspace(start + binsize, end, n_bins, dtype=np.float32)
...
np.vstack(
    [
        time_axis,
        np.full(time_axis.shape[0], block_num_all[i], dtype=np.float32),
    ]
)
```

iii. The notes justify the use of bin end times as matching the reference behavior-interpolation grid rather than creating a separate midpoint or left-edge convention.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. It is perfectly aligned by construction: the time axis has the same number of bins as the neural data and is repeated per trial after the same `keep_idx` filtering.

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

iii. The notes state that all inputs and outputs should share the same 100-bin `stimOn`-aligned grid as the neural tensor.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from the trial table’s `probabilityLeft` column, which encodes the current block prior.

ii.
```python
prob_left_all = trials_df["probabilityLeft"].to_numpy(dtype=float)
block_num_all = trial_number_in_block(prob_left_all)
```

iii. The notes say `trial_number_in_block` should come from the block structure in `trials.probabilityLeft`, since that variable directly marks block changes.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The agent scans trials in their original order, resets the counter to 1 whenever `probabilityLeft` changes, increments otherwise, then subsets to retained trials and repeats the scalar across all time bins in that trial.

ii.
```python
def trial_number_in_block(probability_left: np.ndarray) -> np.ndarray:
    ...
    for idx, value in enumerate(probability_left):
        if idx == 0 or not np.isclose(value, prev):
            current = 1
        else:
            current += 1
        out[idx] = current
        prev = value
    return out
```

iii. The notes explicitly justify computing block number before filtering so the converted trial keeps its true experimental block position instead of being renumbered after exclusions.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. `choice` comes directly from the trials table column `choice`.

ii.
```python
choice_all = trials_df["choice"].to_numpy()
...
choice_labels = choice_to_label(choice_all[keep_idx])
```

iii. The notes identify `trials_df['choice']` as the reference variable for the choice target.

## 5-b. What processing is involved in computing `output` *Choice*?

i. After filtering out no-choice trials upstream, the agent validates that only `-1` and `1` remain, maps raw `+1` to left (`0`) and raw `-1` to right (`1`), and then repeats that per-trial label across every time bin of the output tensor.

ii.
```python
def choice_to_label(choice_values: np.ndarray) -> np.ndarray:
    choice_values = np.asarray(choice_values)
    if not np.all(np.isin(choice_values, [-1, 1])):
        ...
    return (choice_values == -1).astype(np.int64)
...
np.full(T, session.choice_labels[trial_idx], dtype=np.int64)
```

iii. The notes say this mapping follows the IBL coding convention while converting it into the task’s requested left=`0`, right=`1` labels.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It is derived from the trial table’s `probabilityLeft` column.

ii.
```python
prob_left_all = trials_df["probabilityLeft"].to_numpy(dtype=float)
...
prior_labels = prior_to_label(prob_left_all[keep_idx])
```

iii. The notes identify `probabilityLeft` as the direct prior variable used both by the paper and by the task specification.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The agent rounds `probabilityLeft` to one decimal place, verifies only `0.2`, `0.5`, and `0.8` occur, maps them to class IDs `0`, `1`, and `2`, and repeats the chosen class across each trial’s time bins.

ii.
```python
PRIOR_MAP = {
    0.2: 0,
    0.5: 1,
    0.8: 2,
}
...
rounded = np.round(np.asarray(probability_left, dtype=float), 1)
return np.array([PRIOR_MAP[x] for x in rounded], dtype=np.int64)
```

iii. The notes say this is a direct encoding of the task’s requested prior classes.

## 9-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. Wheel speed is derived from `_ibl_wheel.timestamps.npy` and `_ibl_wheel.position.npy`.

ii.
```python
wheel_timestamps = np.load(find_latest_file(alf_path, "**/_ibl_wheel.timestamps.npy"))
wheel_position = np.load(find_latest_file(alf_path, "**/_ibl_wheel.position.npy"))
```

iii. The notes describe this as the local-file version of the reference `load_target_behavior('wheel-speed')` path.

## 9-b. What processing is involved in computing `output` *Wheel speed*?

i. The agent interpolates wheel position to a 1000 Hz time base, computes filtered velocity, takes its absolute value to get speed, then linearly interpolates the continuous speed trace into each trial’s common aligned bin grid.

ii.
```python
position, times = interpolate_position(wheel_timestamps, wheel_position, freq=1000)
velocity, _ = velocity_filtered(position, fs=1000, corner_frequency=20, order=8)
return {
    "times": np.asarray(times, dtype=np.float32),
    "values": np.abs(np.asarray(velocity, dtype=np.float32)),
}
...
traces, good_mask, _ = get_behavior_per_interval_current(...)
```

iii. The notes say wheel speed should match the reference behavior loader semantics (`abs(wheel velocity)`) before any task-driven discretization is applied.

## 9-c. How is `output` *Wheel speed* thresholded into categories?

i. The agent concatenates all retained aligned wheel-speed samples across all converted sessions, computes the global one-third and two-third quantiles, and applies `np.digitize` so each bin becomes `0`, `1`, or `2` (`low`, `medium`, `high`).

ii.
```python
wheel_values = np.concatenate(
    [np.concatenate(session.wheel_continuous) for session in processed_sessions]
)
...
q1, q2 = np.quantile(values, [1 / 3, 2 / 3])
...
return np.digitize(values, bins=np.array([q1, q2]), right=False).astype(np.int64)
```

iii. The notes explicitly justify this as a task-driven deviation: the reference data are continuous, so the converter uses global tertiles to satisfy the required 3-bin categorical output.

## 9-d. How is `output` *Wheel speed* aligned with the neural data?

i. Wheel speed is segmented and linearly interpolated onto the exact same `stimOn_times`-aligned `[-0.5, 1.5]` / 20 ms grid as the neural data, then only trials passing the shared `keep_mask` are retained.

ii.
```python
wheel_traces, wheel_mask = align_continuous_behavior(session_path, "wheel-speed", trials_df)
...
keep_mask = np.asarray(trials_mask, dtype=bool) & wheel_mask & whisker_mask
...
wheel_aligned = [np.asarray(wheel_traces[i], dtype=np.float32) for i in keep_idx]
```

iii. The notes call this another deliberate task-driven choice: the paper aligns dynamic behaviors differently, but the task asked for one stimulus-onset-aligned dataset.

## 10-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. Whisker motion energy is derived from camera timestamps (`_ibl_leftCamera.times.npy` or `_ibl_rightCamera.times.npy`) and ROI motion-energy traces (`leftCamera.ROIMotionEnergy.npy` or `rightCamera.ROIMotionEnergy.npy`).

ii.
```python
times = np.load(find_latest_file(alf_path, "**/_ibl_leftCamera.times.npy"))
values = np.load(find_latest_file(alf_path, "**/leftCamera.ROIMotionEnergy.npy"))
...
times = np.load(find_latest_file(alf_path, "**/_ibl_rightCamera.times.npy"))
values = np.load(find_latest_file(alf_path, "**/rightCamera.ROIMotionEnergy.npy"))
```

iii. The notes describe this as the local-file equivalent of the reference whisker-motion loader, with left-camera preference and right-camera fallback.

## 10-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The agent loads left whisker motion energy when available and otherwise falls back to right whisker motion energy. If camera timestamps are longer than the motion-energy array, it trims the timestamps to match; then it linearly interpolates the continuous trace onto each trial’s aligned bin grid.

ii.
```python
if times.shape[0] < values.shape[0]:
    raise ValueError("Camera times are shorter than video data for leftCamera.")
if times.shape[0] > values.shape[0]:
    times = times[-values.shape[0] :]
...
if behavior_name == "whisker-motion-energy":
    target = load_target_behavior_current(session_path, "left-whisker-motion-energy")
    if target.get("skip"):
        target = load_target_behavior_current(session_path, "right-whisker-motion-energy")
```

iii. The notes justify the left-then-right fallback as matching the reference code, and the timestamp trimming as a pragmatic fix for minor camera-length mismatches in local ALF files.

## 10-c. How is `output` *Whisker motion energy* thresholded into categories?

i. It is thresholded the same way as wheel speed: global tertiles over all retained aligned whisker samples define the `low`, `medium`, and `high` classes.

ii.
```python
whisker_values = np.concatenate(
    [np.concatenate(session.whisker_continuous) for session in processed_sessions]
)
...
thresholds[name] = (float(q1), float(q2))
...
discretize(
    session.whisker_continuous[trial_idx],
    thresholds["whisker_motion_energy"],
)
```

iii. The notes again describe this as task-required discretization applied only after reproducing the continuous loading and alignment.

## 10-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. The whisker trace is aligned identically to neural and wheel data: per trial, it is interpolated into the same `stimOn_times`-aligned 2 s window with 20 ms bins, and then filtered by the shared `keep_mask`.

ii.
```python
whisker_traces, whisker_mask = align_continuous_behavior(
    session_path, "whisker-motion-energy", trials_df
)
...
whisker_aligned = [np.asarray(whisker_traces[i], dtype=np.float32) for i in keep_idx]
```

iii. The notes state this is the same single-grid alignment choice applied to all requested decoder outputs.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing or malformed behavior loads are converted into `skip` dictionaries; misaligned intervals are marked bad if samples are absent, NaN, start too late, or end too early; timestamp/value length mismatches for camera traces are trimmed when possible; transient session errors are retried up to three times; and sessions with too few valid trials or no good clusters are skipped.

ii.
```python
except BaseException as exc:  # noqa: BLE001
    return {"times": None, "values": None, "skip": True, "error": str(exc)}
...
if len(v_seg) == 0:
    reasons[idx] = "target data not present"
    continue
...
if np.abs(interval_begs[idx] - t_seg[0]) > binsize:
    reasons[idx] = "target data starts too late"
    continue
...
for attempt in range(1, 4):
    try:
        ...
    except Exception as exc:
        if attempt >= 3 or not is_likely_transient_error(exc):
            raise
```

iii. The notes describe this as a robustness layer around the reference logic: keep the valid aligned subset, record skips, and do not try to impute missing neural or behavioral data.

## 12-a. What are the most time-consuming steps of the code?

i. The dominant costs are session-wise spike loading/binning, wheel and whisker alignment by interpolation across every trial, and the full-session parallel conversion pass. Threshold computation also requires concatenating every retained continuous behavior sample across all sessions.

ii.
```python
for interval_idx, (t_beg, t_end) in enumerate(zip(interval_begs, interval_ends)):
    ...
    binned_tmp, _, cluster_idxs = bincount2D(...)
...
for idx, (t_seg, v_seg) in enumerate(zip(target_times_list, target_vals_list)):
    ...
    fn = interp1d(t_seg, v_seg, kind="linear", fill_value="extrapolate")
...
with ProcessPoolExecutor(max_workers=args.session_workers) as executor:
```

iii. The notes identify session-wise loading/alignment and global threshold computation as the heavy parts, which is why the agent added session-level parallelism for `--full`.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. Several explicit Python loops could be vectorized or batched: the `trial_number_in_block` loop, per-interval spike binning in `get_spike_data_per_interval`, per-interval/per-column interpolation in `get_behavior_per_interval_current`, and the per-trial output assembly/discretization loop in `assemble_dataset`.

ii.
```python
for idx, value in enumerate(probability_left):
    ...

for interval_idx, (t_beg, t_end) in enumerate(zip(interval_begs, interval_ends)):
    ...

for idx, (t_seg, v_seg) in enumerate(zip(target_times_list, target_vals_list)):
    ...

for trial_idx in range(session.kept_trial_count):
    ...
```

iii. The notes mention the scale of the full conversion and imply these loops are the main reason parallelism was needed.

## 12-c. What processing does the code repeat multiple times?

i. The code repeatedly scans the filesystem with `find_latest_file`, separately aligns continuous behaviors per session and then concatenates them again to compute global thresholds, and then re-traverses those same per-trial traces during final discretization. It also performs repeated transposes of the binned spike arrays to move between reference and target tensor layouts.

ii.
```python
sort_dir = find_latest_file(
    session_path / "alf" / probe_name / "pykilosort",
    "**/spikes.times.npy",
).parent
...
wheel_values = np.concatenate(
    [np.concatenate(session.wheel_continuous) for session in processed_sessions]
)
...
discretize(
    session.wheel_continuous[trial_idx],
    thresholds["wheel_speed"],
)
```

iii. The notes describe a two-stage behavior workflow: keep continuous traces first so global thresholds can be estimated, then revisit the traces to discretize them.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script defines `build_one()` but never uses it; loads `bwm_df` even though only `sessions_df` drives conversion; stores intermediate per-session metadata such as `cluster_good`, timing, raw/kept trial counts, and kept-trial indices that are not written into the final decoder dataset; and retains continuous wheel/whisker traces for threshold estimation even though downstream analyses only consume the discretized outputs.

ii.
```python
def build_one() -> ONE:
    return ONE(...)
...
bwm_df, sessions_df = load_release_sessions()
...
class ProcessedSession:
    ...
    cluster_good: np.ndarray
    raw_trial_count: int
    kept_trial_count: int
    kept_trial_indices: np.ndarray
    skipped_trial_count: int
    timing: dict[str, float]
```

iii. The notes show these extra fields were mainly used for validation, logging, and plotting rather than for the final `converted_data.pkl` structure.
