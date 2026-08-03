# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from the `bwm_release.csv` file shipped with the reference code, which lists 459 sessions with their eids, pids, probe names, subjects, labs, dates, and session numbers. It then constructs local ALF file paths from these metadata fields and reads trials tables, spike sorting outputs, wheel data, and camera motion energy directly from disk via numpy/pandas, rather than using the ONE API.

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
```

```python
def release_session_path(row: pd.Series) -> Path:
    return (
        ONE_CACHE_DIR / row["lab"] / "Subjects" / row["subject"]
        / row["date"] / f"{int(row['session_number']):03d}"
    )
```

iii. The AI chose to use `bwm_release.csv` as the canonical session roster because it is the same roster used by the reference code's `0_data_caching.py`. Direct local file reads were chosen over ONE API calls to avoid network latency and transient failures.

## 1-b. How are the data split into subjects?

i. Subjects are identified from the `subject` column of `bwm_release.csv`. An ordered unique list of subjects is built from the successfully processed sessions, and each session is assigned a `subject_idx` into that list.

ii.
```python
subjects = ordered_unique([session.subject for session in processed_sessions])
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
```

iii. The subject identity comes directly from the release CSV metadata.

## 1-c. How are the data split into sessions?

i. Each row in the release CSV (after deduplication by eid) is one session. Sessions are processed independently, either sequentially (sample mode) or in parallel (full mode).

ii.
```python
sessions_df = (
    bwm_df[["eid", "lab", "subject", "date", "session_number"]]
    .drop_duplicates(subset=["eid"], keep="first")
    .reset_index(drop=True)
)
```

iii. Sessions are already the unit of organization in the release CSV.

## 1-d. How are the data split into trials?

i. Each session's trials table (`_ibl_trials.table.pqt`) has one row per trial. The trial indices that survive filtering are used to slice the binned neural data and aligned behavioral traces.

ii.
```python
trials_path = find_latest_file(session_path / "alf", "**/_ibl_trials.table.pqt")
trials = pd.read_parquet(trials_path).copy()
```

iii. The trials table naturally provides one row per trial.

## 1-e. How are trials filtered based on quality controls?

i. Several filters are applied: (1) reaction time between 0.08s and 2.0s, (2) trial length from goCue to feedback <= 10.0s, (3) exclude no-choice trials (choice == 0), (4) exclude trials with NaN in stimOn_times, choice, feedback_times, probabilityLeft, firstMovement_times, or feedbackType, (5) wheel and whisker behavioral traces must cover the trial window.

ii.
```python
def load_trials_and_mask_current(session_path, min_rt=0.08, max_rt=2.0, max_trial_len=10.0, exclude_nochoice=True):
    query_parts = []
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
```

```python
keep_mask = np.asarray(trials_mask, dtype=bool) & wheel_mask & whisker_mask
```

iii. These filters are taken from the reference code's `load_trials_and_mask` function, including `max_trial_len=10.0`.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `spikes.times.npy` and `spikes.clusters.npy` from the pykilosort spike sorting output for each probe, plus `clusters.metrics.pqt` for QC labels, `clusters.channels.npy` and `channels.brainLocationIds_ccf_2017.npy` for region mapping.

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

i. Spikes are filtered by QC label >= 1, merged across probes using the reference code's `merge_probes` function, then binned into 20ms bins over the 2s trial window using `bincount2D`. The result is stored as raw spike counts (not divided by bin size), with shape (n_neurons, n_timepoints) per trial.

ii.
```python
def get_spike_data_per_interval(times, clusters, interval_begs, interval_ends, interval_len, binsize):
    ...
    binned_tmp, _, cluster_idxs = bincount2D(times_curr, clust_curr, xbin=binsize, xlim=[t_beg, t_end])
    ...
    binned_spikes[interval_idx, target_indices, :] = binned_tmp[:, :n_bins]
    return binned_spikes
```

```python
neural_trials = [binned_spikes[i].T.astype(np.float32, copy=False) for i in keep_idx]
```

iii. The AI used `bincount2D` from `iblutil.numerical` and the reference code's `merge_probes`. Note that spike counts are NOT divided by bin size (no conversion to Hz).

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only clusters with `label >= 1` in `clusters.metrics.pqt` are kept. This is the same QC threshold used to identify well-isolated neurons in the data paper.

ii.
```python
iok = clusters_labeled["label"] >= qc  # qc=1
selected_clusters = clusters_labeled[iok].copy()
spike_idx, ib = ismember(spikes["clusters"], selected_clusters.index.to_numpy())
selected_spikes = {k: v[spike_idx] for k, v in spikes.items()}
```

iii. The AI documented this matches the paper's well-isolated neuron criteria of 75,708 units.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial's spike window is defined as `[stimOn_times + (-0.5), stimOn_times + (1.5)]`. Spikes within this window are binned into 20ms bins.

ii.
```python
intervals = np.vstack([
    trials_df[PARAMS["align_time"]] + PARAMS["time_window"][0],
    trials_df[PARAMS["align_time"]] + PARAMS["time_window"][1],
]).T
```

```python
idxs_t = (times >= t_beg) & (times < t_end)
```

iii. Alignment to stimulus onset matches the reference code parameters.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 20ms bins, 100 bins per trial. The time axis is defined as bin end times using `np.linspace(start + binsize, end, n_bins)`, giving values from -0.48 to 1.50.

ii.
```python
PARAMS = {
    "interval_len": 2.0,
    "binsize": 0.02,
    "time_window": (-0.5, 1.5),
}

def build_time_axis() -> np.ndarray:
    start, end = PARAMS["time_window"]
    binsize = PARAMS["binsize"]
    n_bins = int(math.ceil((end - start) / binsize))
    return np.linspace(start + binsize, end, n_bins, dtype=np.float32)
```

iii. The 20ms bin size matches the reference code's `binsize=0.02`.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is derived from the parameters `time_window=(-0.5, 1.5)` and `binsize=0.02`. It is a synthetic time axis, not from any raw data variable.

ii.
```python
def build_time_axis() -> np.ndarray:
    start, end = PARAMS["time_window"]
    binsize = PARAMS["binsize"]
    n_bins = int(math.ceil((end - start) / binsize))
    return np.linspace(start + binsize, end, n_bins, dtype=np.float32)
```

iii. The time axis is defined by the conversion parameters.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The time axis is computed as 100 evenly spaced bin end times from -0.48 to 1.50 seconds relative to stimulus onset.

ii.
```python
return np.linspace(start + binsize, end, n_bins, dtype=np.float32)
```

iii. This represents bin END times rather than bin centers.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The same time axis is used for both the input and the behavioral interpolation. Neural spike counts are binned into the same 20ms bins over the same window, so they share the same temporal grid.

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

iii. The time axis and neural bins share the same window and bin size.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. Derived from `probabilityLeft` in the trials table. A block boundary is detected wherever `probabilityLeft` changes value.

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

i. The counter starts at 1 for the first trial of each block and increments by 1 for each subsequent trial in the same block. This is computed on ALL trials (before filtering), then the kept trial indices are used to select the corresponding values.

ii.
```python
prob_left_all = trials_df["probabilityLeft"].to_numpy(dtype=float)
block_num_all = trial_number_in_block(prob_left_all)
...
np.full(time_axis.shape[0], block_num_all[i], dtype=np.float32)
```

iii. Computing on all trials before filtering preserves the animal's true position in the block.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. From `trials.choice` in the trials table, which takes values +1 (left), -1 (right), and 0 (no-go).

ii.
```python
choice_all = trials_df["choice"].to_numpy()
choice_labels = choice_to_label(choice_all[keep_idx])
```

iii. Standard IBL choice encoding.

## 5-b. What processing is involved in computing `output` *Choice*?

i. No-go trials (choice == 0) are excluded by the trial filter. The remaining values are mapped: +1 (left) -> 0, -1 (right) -> 1.

ii.
```python
def choice_to_label(choice_values: np.ndarray) -> np.ndarray:
    return (choice_values == -1).astype(np.int64)
```

iii. This matches the instruction: "left = 0, right = 1".

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. From `trials.probabilityLeft` in the trials table, which takes values 0.2, 0.5, and 0.8.

ii.
```python
PRIOR_MAP = {0.2: 0, 0.5: 1, 0.8: 2}

def prior_to_label(probability_left: np.ndarray) -> np.ndarray:
    rounded = np.round(np.asarray(probability_left, dtype=float), 1)
    return np.array([PRIOR_MAP[x] for x in rounded], dtype=np.int64)
```

iii. Matches the instruction: "0.2 -> 0, 0.5 -> 1, 0.8 -> 2".

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Values are rounded to 1 decimal place and mapped to integers 0, 1, 2. Trials with unexpected values are rejected.

ii.
```python
rounded = np.round(np.asarray(probability_left, dtype=float), 1)
unknown = sorted(set(rounded.tolist()) - set(PRIOR_MAP))
if unknown:
    raise ValueError(f"Unexpected probabilityLeft values: {unknown}")
return np.array([PRIOR_MAP[x] for x in rounded], dtype=np.int64)
```

iii. Straightforward categorical encoding.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. From `_ibl_wheel.timestamps.npy` and `_ibl_wheel.position.npy`, which are the raw wheel encoder readings.

ii.
```python
wheel_timestamps = np.load(find_latest_file(alf_path, "**/_ibl_wheel.timestamps.npy"))
wheel_position = np.load(find_latest_file(alf_path, "**/_ibl_wheel.position.npy"))
position, times = interpolate_position(wheel_timestamps, wheel_position, freq=1000)
velocity, _ = velocity_filtered(position, fs=1000, corner_frequency=20, order=8)
return {"times": times, "values": np.abs(velocity)}
```

iii. The wheel data is loaded from raw files and processed using IBL's standard functions.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The wheel position is interpolated to 1000 Hz, filtered with a 20 Hz Butterworth low-pass filter (order 8), differentiated to velocity, and the absolute value taken as speed. The speed trace is then interpolated onto the trial time grid using `scipy.interpolate.interp1d` with linear interpolation and extrapolation.

ii.
```python
position, times = interpolate_position(wheel_timestamps, wheel_position, freq=1000)
velocity, _ = velocity_filtered(position, fs=1000, corner_frequency=20, order=8)
return {"times": times, "values": np.abs(velocity)}
```

```python
x_interp = np.linspace(interval_begs[idx] + binsize, interval_ends[idx], n_bins)
fn = interp1d(t_seg, v_seg, kind="linear", fill_value="extrapolate")
y_interp = fn(x_interp)
```

iii. Uses the same IBL wheel processing functions as the reference.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. After all sessions are processed, GLOBAL thresholds are computed at the 1/3 and 2/3 quantiles of ALL wheel speed values across ALL sessions. These thresholds are then used to discretize each trial's wheel speed into 3 bins (0=low, 1=medium, 2=high).

ii.
```python
def compute_thresholds(processed_sessions):
    wheel_values = np.concatenate([np.concatenate(session.wheel_continuous) for session in processed_sessions])
    q1, q2 = np.quantile(values, [1/3, 2/3])
    thresholds[name] = (float(q1), float(q2))

def discretize(values, thresholds):
    q1, q2 = thresholds
    return np.digitize(values, bins=np.array([q1, q2]), right=False).astype(np.int64)
```

iii. The AI chose global thresholds to ensure consistent discretization across sessions.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The wheel speed trace is interpolated onto the same bin-end time grid as neural data using `interp1d`, evaluated at `np.linspace(interval_beg + binsize, interval_end, n_bins)`.

ii.
```python
x_interp = np.linspace(interval_begs[idx] + binsize, interval_ends[idx], n_bins)
fn = interp1d(t_seg, v_seg, kind="linear", fill_value="extrapolate")
y_interp = fn(x_interp)
```

iii. Uses the same stimulus-onset-aligned window and bin count as the neural data.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. From `leftCamera.ROIMotionEnergy.npy` (preferred) or `rightCamera.ROIMotionEnergy.npy` (fallback), with corresponding camera timestamps `_ibl_leftCamera.times.npy` or `_ibl_rightCamera.times.npy`.

ii.
```python
def align_continuous_behavior(session_path, behavior_name, trials_df):
    if behavior_name == "whisker-motion-energy":
        target = load_target_behavior_current(session_path, "left-whisker-motion-energy")
        if target.get("skip"):
            target = load_target_behavior_current(session_path, "right-whisker-motion-energy")
```

iii. Left camera preferred, right as fallback, matching the reference code.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The raw motion energy trace is used as-is (no additional filtering). If camera times are longer than values, the times are trimmed from the start to match. The trace is interpolated onto the trial time grid using `interp1d`.

ii.
```python
times = np.load(find_latest_file(alf_path, "**/_ibl_leftCamera.times.npy"))
values = np.load(find_latest_file(alf_path, "**/leftCamera.ROIMotionEnergy.npy"))
if times.shape[0] > values.shape[0]:
    times = times[-values.shape[0]:]
```

iii. No additional filtering beyond interpolation to the trial grid.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Same as wheel speed: GLOBAL thresholds at 1/3 and 2/3 quantiles across all sessions, then `np.digitize` into 3 bins.

ii.
```python
whisker_values = np.concatenate([np.concatenate(session.whisker_continuous) for session in processed_sessions])
q1, q2 = np.quantile(values, [1/3, 2/3])
```

iii. Same global discretization approach as wheel speed.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Same as wheel speed: interpolated onto bin-end time grid aligned to stimulus onset.

ii.
```python
x_interp = np.linspace(interval_begs[idx] + binsize, interval_ends[idx], n_bins)
fn = interp1d(t_seg, v_seg, kind="linear", fill_value="extrapolate")
```

iii. Same alignment method as wheel speed.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Multiple strategies: (1) Trials with NaN in key fields are excluded by the trial mask. (2) Sessions where all trials are filtered out (< 2 valid trials) raise a RuntimeError and are skipped. (3) Wheel/whisker coverage is checked per trial. (4) Camera time/value length mismatches are handled by trimming times. (5) Failed sessions are retried once sequentially, then skipped with documentation. (6) Probes with no clusters after QC are skipped.

ii.
```python
if keep_idx.size < 2:
    raise RuntimeError(f"Only {keep_idx.size} valid trials after filtering")
```

```python
if times.shape[0] > values.shape[0]:
    times = times[-values.shape[0]:]
```

```python
if len(clusters) == 0:
    continue
```

iii. The AI documented that 21 sessions were excluded because they had 0 valid trials after filtering.

## 10-a. What are the most time-consuming steps of the code?

i. Loading spike sorting data from disk is the most expensive per-session step. The AI documented that the full conversion with 12 parallel workers took 34.5 minutes for 438 sessions.

ii.
```python
spikes = {
    "times": np.load(sort_dir / "spikes.times.npy"),
    "clusters": np.load(sort_dir / "spikes.clusters.npy").astype(np.int32),
}
```

iii. The AI benchmarked different worker counts and found 12 workers optimal, balancing parallelism against disk contention.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial spike binning loop in `get_spike_data_per_interval` iterates over each trial interval individually, calling `bincount2D` once per trial. This could be vectorized by offsetting spike times and using a single bincount call. Similarly, the per-trial behavior interpolation loop in `get_behavior_per_interval_current` iterates per trial.

ii.
```python
for interval_idx, (t_beg, t_end) in enumerate(zip(interval_begs, interval_ends)):
    idxs_t = (times >= t_beg) & (times < t_end)
    ...
    binned_tmp, _, cluster_idxs = bincount2D(times_curr, clust_curr, xbin=binsize, xlim=[t_beg, t_end])
```

```python
for idx, (t_seg, v_seg) in enumerate(zip(target_times_list, target_vals_list)):
    ...
    fn = interp1d(t_seg, v_seg, kind="linear", fill_value="extrapolate")
    y_interp = fn(x_interp)
```

iii. The AI noted this in CONVERSION_NOTES.md but prioritized correctness over further optimization.

## 10-c. What processing does the code repeat multiple times?

i. The code bins ALL trials' neural data before filtering, then only keeps the filtered subset. Similarly, the behavioral alignment is done for all trials before filtering. This means processing is done on trials that are subsequently discarded.

ii.
```python
binned_spikes = bin_spiking_data_current(reg_clu_ids, neural_dict, trials_df)  # all trials
...
keep_mask = np.asarray(trials_mask, dtype=bool) & wheel_mask & whisker_mask
neural_trials = [binned_spikes[i].T.astype(np.float32, copy=False) for i in keep_idx]  # subset
```

iii. The AI processes all trials first, then filters, meaning discarded trials were processed unnecessarily.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code bins neural data for ALL trials (including those that will be filtered out by wheel/whisker coverage or trial quality masks). It also creates `interp1d` interpolation objects for trials that may later be discarded. Additionally, the code stores `cluster_good` metadata which is redundant since QC filtering was already applied.

ii.
```python
binned_spikes = bin_spiking_data_current(reg_clu_ids, neural_dict, trials_df)  # bins ALL trials
...
neural_trials = [binned_spikes[i].T ... for i in keep_idx]  # only keeps subset
```

iii. Processing all trials before filtering is wasteful but simpler to implement.
