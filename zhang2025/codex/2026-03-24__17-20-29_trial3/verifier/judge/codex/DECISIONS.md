# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does not load data through `ONE.search()` plus loader objects. It first reads the release roster from `code/code_zhang2025/data/bwm_release.csv`, uses that table to get `eid`, `lab`, `subject`, `date`, `session_number`, and per-probe info, then constructs a local session path under `data/one_cache/.../alf/`. Trials, wheel, whisker, spikes, cluster metrics, and channel-region arrays are then opened directly from local parquet/NumPy files with `find_latest_file`, `pd.read_parquet`, and `np.load`.

ii. Code snippets:
```python
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

```python
trials_path = find_latest_file(session_path / "alf", "**/_ibl_trials.table.pqt")
trials = pd.read_parquet(trials_path).copy()
```

iii. In `CONVERSION_NOTES.md`, the AI says it chose a deterministic session roster from `bwm_release.csv` and switched to direct local ALF/parquet/NumPy reads because remote ONE/SessionLoader access was slow and fragile, while the needed files were already cached locally.

## 1-b. How are the data split into subjects?

i. Subject identity comes directly from the `subject` column of the release CSV. After session processing, the final `subjects` list is built with `ordered_unique` over converted sessions in release order, and `subject_idx` maps each session to that subject list.

ii. Code snippets:
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

iii. The notes say the release roster is deterministic and already contains subject IDs, so there is no need to derive subjects from paths or remote metadata.

## 1-c. How are the data split into sessions?

i. Sessions are defined by unique `eid` rows in `bwm_release.csv`. The AI drops duplicate `eid`s, attaches probe info per `eid`, then processes one session at a time. Sessions that fail conversion or end up with fewer than two valid trials are skipped from the final dataset.

ii. Code snippets:
```python
sessions_df = (
    bwm_df[["eid", "lab", "subject", "date", "session_number"]]
    .drop_duplicates(subset=["eid"], keep="first")
    .reset_index(drop=True)
)
```

```python
for future in as_completed(future_map):
    ...
    try:
        session = future.result()
        processed_sessions.append(session)
    except Exception as exc:
        retry_records.append(row_dict)
```

```python
if keep_idx.size < 2:
    raise RuntimeError(f"Only {keep_idx.size} valid trials after filtering")
```

iii. The notes justify this as using the 459-session release roster from the reference code as the canonical starting point, then letting task-specific validity checks determine the final converted subset.

## 1-d. How are the data split into trials?

i. Trials are the rows of `_ibl_trials.table.pqt`. The AI processes all rows of the trial table, computes trial-aligned neural and behavior windows for each row, and then uses `keep_idx` to choose the retained trial rows. Each kept row becomes one per-trial array in `neural`, `input`, and `output`.

ii. Code snippets:
```python
trials_path = find_latest_file(session_path / "alf", "**/_ibl_trials.table.pqt")
trials = pd.read_parquet(trials_path).copy()
```

```python
keep_mask = np.asarray(trials_mask, dtype=bool) & wheel_mask & whisker_mask
keep_idx = np.flatnonzero(keep_mask)
```

```python
neural_trials = [binned_spikes[i].T.astype(np.float32, copy=False) for i in keep_idx]
```

iii. The notes describe the trial table as already being one row per trial and say the AI kept that structure, only subsetting after filtering and modality-coverage checks.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies a trial-table mask and a behavior-coverage mask. The trial-table mask rejects trials with reaction time outside `0.08` to `2.0` s, trial duration `feedback_times - goCue_times > 10.0`, missing `stimOn_times`, `choice`, `feedback_times`, `probabilityLeft`, `firstMovement_times`, or `feedbackType`, and no-choice trials (`choice == 0`). It then requires both wheel and whisker traces to cover the requested trial window, and drops whole sessions with fewer than two surviving trials.

ii. Code snippets:
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

iii. The notes say the AI intended to reproduce the reference trial mask from the Zhang code, while also enforcing valid wheel and whisker alignment windows for the requested outputs.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The final neural matrices are built from `spikes.times.npy` and `spikes.clusters.npy`. The AI also reads `clusters.metrics.pqt`, `clusters.channels.npy`, and `channels.brainLocationIds_ccf_2017.npy` to filter units and assign anatomical labels, but the binned neural activity itself comes from spike times and spike cluster IDs.

ii. Code snippets:
```python
spikes = {
    "times": np.load(sort_dir / "spikes.times.npy"),
    "clusters": np.load(sort_dir / "spikes.clusters.npy").astype(np.int32),
}
clusters_labeled = pd.read_parquet(sort_dir / "clusters.metrics.pqt")
```

```python
cluster_channels = np.load(sort_dir / "clusters.channels.npy").astype(np.int64)
channel_region_ids = np.load(sort_dir / "channels.brainLocationIds_ccf_2017.npy").astype(
    np.int64
)
```

iii. The notes say the conversion uses the same underlying ALF spike-sorting outputs as the reference code, but reads them directly from local files instead of through the loader wrappers.

## 2-b. How is the `neural` data processed?

i. The AI filters clusters, merges probes within a session, bins spikes into `[-0.5, 1.5]` s trial windows with `20` ms bins, and stores each trial as a `(n_neurons, 100)` matrix. The code keeps spike counts; it does not divide by the bin width to convert to firing rates in Hz.

ii. Code snippets:
```python
spikes, clusters = merge_probes(spikes_list, clusters_list)
neural_dict = {
    "spike_times": spikes["times"],
    "spike_clusters": spikes["clusters"],
    "cluster_regions": clusters["acronym"].to_numpy(),
}
```

```python
binned_array = get_spike_data_per_interval(
    regspikes,
    regclu,
    interval_begs=intervals[:, 0],
    interval_ends=intervals[:, 1],
    interval_len=interval_len,
    binsize=PARAMS["binsize"],
)
return np.array([x.T for x in binned_array], dtype=np.float32)
```

```python
neural_trials = [binned_spikes[i].T.astype(np.float32, copy=False) for i in keep_idx]
```

iii. The notes say the AI reused “reference-style spike binning logic” but deliberately exported “well-isolated spike counts” in the requested trial format.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI keeps only clusters with `label >= 1`, i.e. well-isolated units by the IBL QC label. It filters spikes down to those clusters before probe merging, and skips sessions that have no surviving clusters after this QC.

ii. Code snippets:
```python
iok = clusters_labeled["label"] >= qc
selected_clusters = clusters_labeled[iok].copy()
spike_idx, ib = ismember(spikes["clusters"], selected_clusters.index.to_numpy())
```

```python
spikes, clusters = load_spiking_data_current(
    session_path,
    probe_name=probe_name,
    qc=1,
)
...
if not spikes_list:
    raise RuntimeError("No good clusters after QC filtering")
```

iii. The notes repeatedly justify this as matching the paper’s “well-isolated neurons” criterion and bringing the converted neuron counts close to the paper’s 75,708-unit scale.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI aligns neural data to stimulus onset by defining each trial window as `stimOn_times + (-0.5, 1.5)` and binning spikes whose absolute timestamps fall inside that interval. It does not explicitly subtract onset from each spike time; alignment is done through the interval boundaries.

ii. Code snippets:
```python
intervals = np.vstack(
    [
        trials_df[PARAMS["align_time"]] + PARAMS["time_window"][0],
        trials_df[PARAMS["align_time"]] + PARAMS["time_window"][1],
    ]
).T
```

```python
PARAMS = {
    "interval_len": 2.0,
    "binsize": 0.02,
    "align_time": "stimOn_times",
    "time_window": (-0.5, 1.5),
}
```

iii. The notes say the user requested common stimulus-onset alignment, so the AI used the reference code’s `stimOn_times`-aligned `[-0.5, 1.5]` trial grid for neural and behavioral streams.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI uses `20` ms bins over a 2 s interval, producing `100` time bins per trial. No further temporal rebinning or smoothing is applied to neural activity.

ii. Code snippets:
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

iii. The notes say the AI followed the executable reference grid of a 2 s trial window with 20 ms bins.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. The AI does not derive this input from a raw time-series column. It defines a fixed relative time axis from the chosen stimulus-onset alignment scheme: `stimOn_times` supplies the alignment event, while `PARAMS["time_window"]` and `PARAMS["binsize"]` define the values stored in the input channel.

ii. Code snippets:
```python
def build_time_axis() -> np.ndarray:
    start, end = PARAMS["time_window"]
    binsize = PARAMS["binsize"]
    n_bins = int(math.ceil((end - start) / binsize))
    return np.linspace(start + binsize, end, n_bins, dtype=np.float32)
```

```python
PARAMS = {
    "interval_len": 2.0,
    "binsize": 0.02,
    "align_time": "stimOn_times",
    "time_window": (-0.5, 1.5),
}
```

iii. The notes say this channel was not directly present in the reference code and had to be constructed from the adopted alignment grid.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The AI creates one fixed 100-sample vector using `np.linspace(start + binsize, end, n_bins)`, so the saved values are bin end times from `-0.48` s to `1.50` s in 20 ms steps. That same vector is reused for every trial.

ii. Code snippets:
```python
def build_time_axis() -> np.ndarray:
    start, end = PARAMS["time_window"]
    binsize = PARAMS["binsize"]
    n_bins = int(math.ceil((end - start) / binsize))
    # Match the reference behavior interpolation grid: bin end times.
    return np.linspace(start + binsize, end, n_bins, dtype=np.float32)
```

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

iii. The notes justify this as matching the AI’s chosen behavior interpolation grid and explicitly report the resulting sample range as `[-0.48, 1.50]`.

## 3-c. How is `input` *Time since stimulus onset* aligned with the neural data?

i. The AI intends this input to share the same 100-bin trial grid as the neural matrices: spikes are binned in stimulus-onset windows and the time input stores the corresponding bin-end times for those windows. The time vector is repeated identically for every retained trial.

ii. Code snippets:
```python
time_axis = build_time_axis()
```

```python
neural_trials = [binned_spikes[i].T.astype(np.float32, copy=False) for i in keep_idx]
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

iii. The notes say the time channel uses the same common stimulus-aligned 2 s / 20 ms grid as the neural and behavioral data.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. The AI derives trial number in block from `trials_df["probabilityLeft"]`, treating changes in `probabilityLeft` as block boundaries.

ii. Code snippets:
```python
def trial_number_in_block(probability_left: np.ndarray) -> np.ndarray:
    probability_left = np.asarray(probability_left, dtype=float)
```

```python
prob_left_all = trials_df["probabilityLeft"].to_numpy(dtype=float)
block_num_all = trial_number_in_block(prob_left_all)
```

iii. The notes say the block count should come from `probabilityLeft` changes because the trials table does not carry an explicit block index.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The AI scans the full trial order, resets the counter whenever `probabilityLeft` changes, and counts trials within the block as `1, 2, 3, ...`. It computes this before filtering trials, then subsets the precomputed values to `keep_idx` and repeats the chosen value across all time bins of each kept trial.

ii. Code snippets:
```python
current = 0
prev = None
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

iii. The notes justify computing this before filtering so the value reflects the animal’s original position in the block rather than the position after excluded trials are removed.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. The choice output is derived from the `choice` column of the trials table.

ii. Code snippets:
```python
choice_all = trials_df["choice"].to_numpy()
choice_labels = choice_to_label(choice_all[keep_idx])
```

iii. The notes say the conversion uses the filtered trials table directly for `choice`.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The AI first removes no-choice trials in the trial mask, then validates that remaining values are only `-1` and `1`. It maps raw IBL `-1` to decoded label `1` (right) and raw `+1` to decoded label `0` (left), then repeats that label across all 100 bins of the kept trial.

ii. Code snippets:
```python
if not np.all(np.isin(choice_values, [-1, 1])):
    bad = np.unique(choice_values[~np.isin(choice_values, [-1, 1])])
    raise ValueError(f"Unexpected choice values after filtering: {bad.tolist()}")
return (choice_values == -1).astype(np.int64)
```

```python
np.full(T, session.choice_labels[trial_idx], dtype=np.int64)
```

iii. The notes explicitly justify the mapping as converting raw IBL wheel-direction labels into the requested left/right categorical output.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. The prior output is derived from the `probabilityLeft` column of the trials table.

ii. Code snippets:
```python
prob_left_all = trials_df["probabilityLeft"].to_numpy(dtype=float)
prior_labels = prior_to_label(prob_left_all[keep_idx])
```

iii. The notes say the conversion uses `probabilityLeft` directly as the requested prior variable.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The AI rounds the raw values to one decimal place, validates that only `0.2`, `0.5`, and `0.8` remain, maps them to labels `0`, `1`, and `2`, and repeats the chosen label across all time bins of each kept trial.

ii. Code snippets:
```python
rounded = np.round(np.asarray(probability_left, dtype=float), 1)
unknown = sorted(set(rounded.tolist()) - set(PRIOR_MAP))
if unknown:
    raise ValueError(f"Unexpected probabilityLeft values: {unknown}")
return np.array([PRIOR_MAP[x] for x in rounded], dtype=np.int64)
```

```python
np.full(T, session.prior_labels[trial_idx], dtype=np.int64)
```

iii. The notes say this direct categorical remapping is required by the decoder task specification.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. The wheel-speed output is derived from `_ibl_wheel.timestamps.npy` and `_ibl_wheel.position.npy`.

ii. Code snippets:
```python
wheel_timestamps = np.load(find_latest_file(alf_path, "**/_ibl_wheel.timestamps.npy"))
wheel_position = np.load(find_latest_file(alf_path, "**/_ibl_wheel.position.npy"))
```

iii. The notes say the conversion matches the reference behavior source variables for wheel.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The AI interpolates wheel position to 1000 Hz, computes filtered velocity with `velocity_filtered`, takes absolute value to get speed, extracts the stimulus-aligned `[-0.5, 1.5]` trial windows, and linearly interpolates each trial onto the common 100-bin grid.

ii. Code snippets:
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

iii. The notes say this reproduces the reference wheel-processing logic while staying on a shared stimulus-onset grid.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. The AI computes one pair of global thresholds from all retained wheel-speed samples across all converted sessions and trials, using the 1/3 and 2/3 quantiles. It then discretizes each aligned wheel trace with `np.digitize` into `low`, `medium`, and `high`.

ii. Code snippets:
```python
wheel_values = np.concatenate(
    [np.concatenate(session.wheel_continuous) for session in processed_sessions]
)
q1, q2 = np.quantile(values, [1 / 3, 2 / 3])
thresholds[name] = (float(q1), float(q2))
```

```python
return np.digitize(values, bins=np.array([q1, q2]), right=False).astype(np.int64)
```

iii. The notes explicitly call this a “global tertile” decision, justified as applying one consistent three-bin discretization rule across the full converted dataset.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. Wheel speed is aligned to the same `stimOn_times + (-0.5, 1.5)` trial windows used for neural data. Within each kept trial, the wheel trace is resampled at the same 100 bin-end time points that the AI uses for its common trial grid.

ii. Code snippets:
```python
align_times = trials_df[PARAMS["align_time"]].to_numpy()
interval_begs = align_times + start
interval_ends = align_times + end
```

```python
x_interp = np.linspace(interval_begs[idx] + binsize, interval_ends[idx], n_bins)
```

iii. The notes say the dynamic outputs were intentionally forced onto the same stimulus-onset-aligned grid as neural activity.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. The whisker output comes from side-camera timestamps and ROI motion-energy arrays: `_ibl_leftCamera.times.npy` plus `leftCamera.ROIMotionEnergy.npy` when available, otherwise `_ibl_rightCamera.times.npy` plus `rightCamera.ROIMotionEnergy.npy`.

ii. Code snippets:
```python
if behavior_name == "whisker-motion-energy":
    target = load_target_behavior_current(session_path, "left-whisker-motion-energy")
    if target.get("skip"):
        target = load_target_behavior_current(session_path, "right-whisker-motion-energy")
```

```python
times = np.load(find_latest_file(alf_path, "**/_ibl_leftCamera.times.npy"))
values = np.load(find_latest_file(alf_path, "**/leftCamera.ROIMotionEnergy.npy"))
```

iii. The notes justify this as matching the reference logic of preferring the left whisker camera and falling back to the right camera when needed.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The AI uses the released motion-energy trace as the value stream, but first fixes a specific data issue: if the camera time array is longer than the motion-energy array, it truncates the time array from the end to match lengths. It then extracts stimulus-aligned trial windows and linearly interpolates each trial onto the common 100-bin grid without additional filtering or normalization.

ii. Code snippets:
```python
if times.shape[0] < values.shape[0]:
    raise ValueError("Camera times are shorter than video data for leftCamera.")
if times.shape[0] > values.shape[0]:
    times = times[-values.shape[0] :]
```

```python
fn = interp1d(t_seg, v_seg, kind="linear", fill_value="extrapolate")
y_interp = fn(x_interp)
```

iii. The notes say the goal was to use the same raw motion-energy variable as the reference while handling minor cache inconsistencies robustly.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Like wheel speed, whisker motion energy is globally thresholded: the AI concatenates all retained whisker samples across sessions, computes the 1/3 and 2/3 quantiles, and digitizes every time point into three categories.

ii. Code snippets:
```python
whisker_values = np.concatenate(
    [np.concatenate(session.whisker_continuous) for session in processed_sessions]
)
q1, q2 = np.quantile(values, [1 / 3, 2 / 3])
thresholds[name] = (float(q1), float(q2))
```

```python
discretize(
    session.whisker_continuous[trial_idx],
    thresholds["whisker_motion_energy"],
)
```

iii. The notes explicitly describe whisker discretization as using global tertile thresholds over all retained aligned samples.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. The whisker trace is aligned to the same stimulus-onset-centered trial window as the neural data and resampled onto the same common 100-bin grid that the AI uses for time-varying outputs.

ii. Code snippets:
```python
align_times = trials_df[PARAMS["align_time"]].to_numpy()
interval_begs = align_times + start
interval_ends = align_times + end
```

```python
x_interp = np.linspace(interval_begs[idx] + binsize, interval_ends[idx], n_bins)
```

iii. The notes say the user required stimulus-onset alignment for all requested outputs, so whisker motion energy was put on the same trial grid as neural data.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles missing or malformed data by dropping trials or skipping sessions. Missing wheel/whisker streams return `skip` information and produce false entries in the behavior-validity mask. Trial windows are rejected if the target data start too late, end too early, or have no samples. If camera times are longer than motion-energy values, the times are truncated to match. If there are no good clusters or fewer than two valid trials, the session is skipped. Worker failures are retried up to three times, with retries reserved mainly for transient errors.

ii. Code snippets:
```python
if target_times is None or target_vals is None:
    return [None] * n_intervals, np.zeros(n_intervals, dtype=bool), ["missing"] * n_intervals
...
if np.abs(interval_begs[idx] - t_seg[0]) > binsize:
    reasons[idx] = "target data starts too late"
    continue
if np.abs(interval_ends[idx] - t_seg[-1]) > binsize:
    reasons[idx] = "target data ends too early"
    continue
```

```python
if times.shape[0] > values.shape[0]:
    times = times[-values.shape[0] :]
```

```python
if keep_idx.size < 2:
    raise RuntimeError(f"Only {keep_idx.size} valid trials after filtering")
```

iii. The notes describe these choices as robustness measures for local cache inconsistencies and missing modalities, while preserving only trials/sessions that have full usable windows for the requested decoder outputs.

## 10-a. What are the most time-consuming steps of the code?

i. The AI identifies spike loading/binning and heavy local disk I/O as the dominant costs. Its notes specifically say remote ONE access was too slow, that local-file reading plus session-level parallelism was used instead, and that high worker counts created severe disk contention during the full run.

ii. Code snippets:
```python
spikes = {
    "times": np.load(sort_dir / "spikes.times.npy"),
    "clusters": np.load(sort_dir / "spikes.clusters.npy").astype(np.int32),
}
```

```python
binned_spikes = bin_spiking_data_current(reg_clu_ids, neural_dict, trials_df)
```

iii. `CONVERSION_NOTES.md` says the expensive parts were repeated local spike-file reads and binning, with 48-way parallelism turning into a disk-contention bottleneck.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI code has several obvious Python loops that could have been vectorized or batched better: the per-trial spike-binning loop, the per-trial behavior interpolation loop, the explicit loop for `trial_number_in_block`, and the per-trial output assembly loop.

ii. Code snippets:
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
for idx, value in enumerate(probability_left):
    ...
```

iii. The notes discuss speedups elsewhere, but the structure of the final code itself shows these loops remain in place despite the vectorization opportunity.

## 10-c. What processing does the code repeat multiple times?

i. The AI repeats several steps: it repeatedly glob-searches for “latest” files for each modality, performs one pass over all sessions to collect continuous wheel/whisker traces and then a second global pass to compute discretization thresholds, and reruns failed sessions sequentially after the parallel pass.

ii. Code snippets:
```python
def find_latest_file(base_dir: Path, pattern: str) -> Path:
    matches = sorted(base_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No files matched {pattern} under {base_dir}")
    return matches[-1]
```

```python
thresholds = compute_thresholds(processed_sessions)
```

```python
if retry_records:
    ...
    session = process_one_session(row_series, time_axis)
```

iii. The notes explicitly describe the conversion as a “two-stage conversion” and also document the sequential retry pass after worker failures.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI bins spikes and aligns wheel/whisker traces for all trial rows before applying `keep_idx`, so rejected trials are processed and then discarded. It also stores `cluster_qc`, `cluster_good`, and the continuous wheel/whisker traces in `ProcessedSession` even though the final output only keeps region IDs and discretized outputs. It additionally returns `bwm_df` from `load_release_sessions()` even though only `sessions_df` is used downstream.

ii. Code snippets:
```python
binned_spikes = bin_spiking_data_current(reg_clu_ids, neural_dict, trials_df)
wheel_traces, wheel_mask = align_continuous_behavior(session_path, "wheel-speed", trials_df)
whisker_traces, whisker_mask = align_continuous_behavior(
    session_path, "whisker-motion-energy", trials_df
)
...
keep_idx = np.flatnonzero(keep_mask)
```

```python
meta = {
    "cluster_regions": list(clusters["acronym"]),
    "good_clusters": (clusters["label"] >= 1).to_numpy(dtype=np.int8),
    "cluster_qc": {k: np.asarray(v) for k, v in clusters.to_dict("list").items()},
}
```

```python
bwm_df, sessions_df = load_release_sessions()
```

iii. The notes emphasize robustness and validation rather than minimal work, and they describe the need to keep continuous traces around long enough to compute global thresholds and generate optional plots, even though those intermediates are not preserved in the final dataset.
