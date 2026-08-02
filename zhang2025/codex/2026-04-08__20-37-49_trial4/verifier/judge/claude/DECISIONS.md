# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads a CSV release file (`bwm_release.csv`) to get the list of all 459 sessions and their associated probe insertions. It then uses the ONE API to locate or download individual ALF datasets (trials table, spike data, wheel data, whisker motion energy) per session, resolving revisioned paths. Data are loaded from a read-only local cache with fallback to a writable cache and ONE downloads.

ii.
```python
RELEASE_CSV = ROOT / "code" / "code_zhang2025" / "data" / "bwm_release.csv"

def load_release_sessions():
    bwm = pd.read_csv(RELEASE_CSV, index_col=0)
    session_rows = (
        bwm.drop_duplicates("eid")[["eid", "subject", "lab", "date", "session_number"]]
        .reset_index(drop=True)
    )
    probe_rows = {
        eid: grp[["pid", "probe_name"]].reset_index(drop=True)
        for eid, grp in bwm.groupby("eid", sort=False)
    }
    return session_rows, probe_rows
```

iii. The AI documented that the BWM release CSV is the canonical session list (459 sessions, 699 probes) and used it as the entry point, consistent with the reference code's use of the same CSV file.

## 1-b. How are the data split into subjects?

i. Subjects are identified from the `subject` column in the release CSV. A unique subject list is built during dataset assembly, and each session is assigned a `subject_idx` pointing into that list.

ii.
```python
subjects = []
subject_to_idx = {}
for rec in session_records:
    if rec.subject not in subject_to_idx:
        subject_to_idx[rec.subject] = len(subjects)
        subjects.append(rec.subject)
    subject_idx.append(subject_to_idx[rec.subject])
```

iii. The AI noted 139 subjects in the release, with 136 retained after 15 sessions were dropped. Subject assignment follows the release metadata directly.

## 1-c. How are the data split into sessions?

i. Each row in the release CSV (deduplicated by `eid`) defines one session. Sessions are processed individually (in parallel in full mode), with each session producing a `SessionRecord` containing neural, behavioral, and metadata for that session.

ii.
```python
session_rows, probe_rows = load_release_sessions()
# ...
for idx, row in enumerate(session_rows.itertuples(index=False)):
    # Each row is one session, processed independently
```

iii. The AI documented that 459 release sessions are the input, with 444 retained after filtering for whisker data availability and valid trial counts.

## 1-d. How are the data split into trials?

i. Trials are loaded from the ALF trials table (`_ibl_trials.table.pqt`) per session. After building a trial quality mask and computing behavioral coverage masks, valid trial indices are extracted and each valid trial produces one entry in the session's neural/input/output lists.

ii.
```python
trials_df = load_trials_table(one, row)
trial_mask = build_trial_mask(trials_df)
# ...
combined_mask = trial_mask.to_numpy(dtype=bool) & wheel_mask & whisker_mask
valid_idx = np.flatnonzero(combined_mask)
for idx in valid_idx:
    neural_trial = compact_neural_trial(binned_spikes[idx].T)
    # ...
```

iii. The AI described splitting by iterating over valid trial indices within each session, after applying quality and coverage filters.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered using a `build_trial_mask` function that excludes trials with: (1) NaN in key event columns (stimOn_times, choice, feedback_times, probabilityLeft, firstMovement_times, feedbackType), (2) reaction time outside [0.08, 2.0] seconds, (3) trial length (feedback_times - goCue_times) > 10 seconds, (4) no-choice trials (choice == 0). Additionally, trials lacking full wheel or whisker coverage over the [-0.5, 1.5] s window are excluded.

ii.
```python
def build_trial_mask(trials_df, min_rt=0.08, max_rt=2.0, nan_exclude="default",
                     min_trial_len=None, max_trial_len=10.0,
                     exclude_unbiased=False, exclude_nochoice=True):
    if nan_exclude == "default":
        nan_exclude = ["stimOn_times", "choice", "feedback_times",
                       "probabilityLeft", "firstMovement_times", "feedbackType"]
    # ...
    combined_mask = trial_mask.to_numpy(dtype=bool) & wheel_mask & whisker_mask
```

iii. The AI documented these filters as matching the reference code's `load_trials_and_mask` with `max_trial_len=10.0` and `exclude_nochoice=True`, plus additional coverage requirements for behavioral streams.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `spikes.times.npy` (spike timestamps) and `spikes.clusters.npy` (cluster assignments) from each probe's pykilosort output directory, along with `clusters.metrics.pqt` for quality labels, `clusters.channels.npy` for channel assignments, and `channels.brainLocationIds_ccf_2017.npy` for brain region mapping.

ii.
```python
def load_probe_data(one, row, probe_name, brain_regions):
    collection = f"alf/{probe_name}/pykilosort"
    times_path = ensure_dataset_path(one, row, f"{collection}/#*/spikes.times.npy", ...)
    clu_path = ensure_dataset_path(one, row, f"{collection}/#*/spikes.clusters.npy", ...)
    # ...
    spike_times = np.load(times_path).astype(np.float32)
    spike_clusters = np.load(clu_path).astype(np.int32)
```

iii. The AI noted these are the standard IBL spike sorting outputs from Kilosort, consistent with the reference code's use of `SpikeSortingLoader`.

## 2-b. How is the `neural` data processed?

i. Spikes from multiple probes within a session are merged (cluster IDs offset and re-indexed, times sorted). The merged spikes are then binned into fixed-width time bins (20 ms) over a 2-second window aligned to stimulus onset, producing a (n_neurons, 100) count matrix per trial. Counts are stored as uint8.

ii.
```python
def bin_spiking_data(spikes, trials_df):
    intervals = np.vstack([
        trials_df[ALIGN_EVENT].to_numpy() + TIME_WINDOW[0],
        trials_df[ALIGN_EVENT].to_numpy() + TIME_WINDOW[1],
    ]).T
    binned_array, cluster_ids = get_spike_data_per_interval(
        spikes["times"], spikes["clusters"],
        interval_begs=intervals[:, 0], interval_ends=intervals[:, 1],
        interval_len=TIME_WINDOW[1] - TIME_WINDOW[0], binsize=BIN_SIZE)
    binned_trials = np.asarray([x.T for x in binned_array], dtype=np.float32)
    return binned_trials, cluster_ids
```

iii. The AI documented using `bincount2D` for spike binning, matching the reference code's approach. Storage as uint8 was for memory efficiency.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Clusters are filtered to well-isolated units only using `label >= 1` from `clusters.metrics.pqt`. Spikes from clusters that fail this threshold are excluded before binning. Probes with zero good units are skipped rather than failing the entire session.

ii.
```python
good_mask = clusters_df["label"].fillna(0).to_numpy() >= 1
if "cluster_id" in clusters_df.columns:
    good_cluster_ids = clusters_df.loc[good_mask, "cluster_id"].to_numpy(dtype=np.int64)
else:
    good_cluster_ids = np.flatnonzero(good_mask).astype(np.int64)
spike_keep = np.isin(spike_clusters, good_cluster_ids)
spike_times = spike_times[spike_keep]
spike_clusters = spike_clusters[spike_keep]
```

iii. The AI justified this by noting that `label >= 1` on the raw release data produces exactly 75,708 units, matching the paper's stated count of well-isolated neurons. The reference code's `prepare_data` loads all clusters (qc=None) but stores `good_clusters` metadata for potential downstream filtering.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to stimulus onset (`stimOn_times`). For each trial, spikes are extracted from a window of [stimOn_times - 0.5, stimOn_times + 1.5] seconds and binned into 100 bins of 20 ms each.

ii.
```python
ALIGN_EVENT = "stimOn_times"
TIME_WINDOW = (-0.5, 1.5)
BIN_SIZE = 0.02

intervals = np.vstack([
    trials_df[ALIGN_EVENT].to_numpy() + TIME_WINDOW[0],
    trials_df[ALIGN_EVENT].to_numpy() + TIME_WINDOW[1],
]).T
```

iii. The AI documented that stimulus onset alignment matches the reference code's `align_time='stimOn_times'` and the instructions' requirement.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is 20 ms (0.02 seconds), producing 100 bins per trial over the 2-second window. No additional rebinning is applied after the initial spike count computation.

ii.
```python
BIN_SIZE = 0.02
N_BINS = int(np.ceil((TIME_WINDOW[1] - TIME_WINDOW[0]) / BIN_SIZE))  # = 100
```

iii. The AI documented 20 ms bins matching the reference code's `binsize=0.02` parameter.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is derived from the fixed time grid computed from the alignment window parameters (TIME_WINDOW and BIN_SIZE), not from any per-trial raw variable. The grid represents bin-end times relative to stimulus onset.

ii.
```python
TIME_GRID = np.linspace(TIME_WINDOW[0] + BIN_SIZE, TIME_WINDOW[1], N_BINS, dtype=np.float32)
```

iii. The AI described this as a fixed time grid from approximately -0.48 to 1.5 seconds, identical for every trial.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. A fixed linspace grid is computed once from the window parameters. It represents the right edges of each 20 ms time bin, starting at -0.48 s and ending at 1.5 s. This is broadcast identically to every trial.

ii.
```python
TIME_GRID = np.linspace(TIME_WINDOW[0] + BIN_SIZE, TIME_WINDOW[1], N_BINS, dtype=np.float32)
# ...
input_trial = np.vstack([
    TIME_GRID,
    np.full(N_BINS, trial_num, dtype=np.float32),
]).astype(np.float16)
```

iii. The AI noted the range is [-0.48, 1.5] because the grid stores bin-end times.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The time grid uses the same bin structure as the neural data (same TIME_WINDOW, BIN_SIZE, N_BINS), so the i-th time value corresponds to the i-th neural bin. Both share the same temporal alignment to stimulus onset.

ii.
```python
# Same N_BINS=100 used for both neural and input
TIME_GRID = np.linspace(TIME_WINDOW[0] + BIN_SIZE, TIME_WINDOW[1], N_BINS, dtype=np.float32)
```

iii. Alignment is inherent from using the same grid parameters.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from the `probabilityLeft` column in the trials table, which indicates the prior probability of a left stimulus for each trial.

ii.
```python
block_trial_number = compute_trial_number_in_block(trials_df["probabilityLeft"].to_numpy())
```

iii. The AI documented that block boundaries are detected by changes in `probabilityLeft`.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The `compute_trial_number_in_block` function iterates through `probabilityLeft` values sequentially. A counter starts at 1 when a new block begins (when `probabilityLeft` changes value or follows a NaN). NaN values in `probabilityLeft` reset the counter. The resulting per-trial number is computed on the full (unfiltered) trial table, then indexed by the valid trial indices.

ii.
```python
def compute_trial_number_in_block(probability_left: np.ndarray) -> np.ndarray:
    out = np.zeros(len(probability_left), dtype=np.float32)
    prev = None
    count = 0
    for idx, value in enumerate(probability_left):
        if np.isnan(value):
            out[idx] = np.nan
            prev = np.nan
            count = 0
            continue
        if prev is None or np.isnan(prev) or not np.isclose(prev, value):
            count = 1
        else:
            count += 1
        out[idx] = count
        prev = value
    return out
```

iii. The AI documented this as detecting block transitions via changes in `probabilityLeft`, consistent with the task structure where blocks are defined by fixed probability values.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. Choice is derived from the `choice` column in the trials table, where +1 indicates left choice and -1 indicates right choice.

ii.
```python
choice_code = map_choice(trials_df.iloc[idx]["choice"])
```

iii. The AI noted the IBL convention: +1 = left, -1 = right, with no-choice (0) trials excluded.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The raw choice values are mapped to binary codes: +1 (left) -> 0, -1 (right) -> 1. This code is then broadcast across all 100 time bins to create a time-varying output of shape (N_BINS,). No-choice trials (choice == 0) are excluded during trial filtering.

ii.
```python
def map_choice(choice_value: float) -> int:
    if np.isclose(choice_value, 1.0):
        return 0
    if np.isclose(choice_value, -1.0):
        return 1
    raise ValueError(f"Unexpected choice value {choice_value}")
# ...
np.full(N_BINS, choice_code, dtype=np.uint8)
```

iii. The AI documented this as matching the instruction spec: "left = 0, right = 1".

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. Prior probability of left is derived from the `probabilityLeft` column in the trials table.

ii.
```python
prior_code = map_prior(trials_df.iloc[idx]["probabilityLeft"])
```

iii. The AI documented the three possible values: 0.2, 0.5, 0.8.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The raw `probabilityLeft` values are mapped to categorical codes: 0.2 -> 0, 0.5 -> 1, 0.8 -> 2. This code is broadcast across all time bins. Sessions exclude unbiased trials only if `exclude_unbiased=True` (default is False), so 0.5 trials are included.

ii.
```python
def map_prior(prob_left: float) -> int:
    if np.isclose(prob_left, 0.2):
        return 0
    if np.isclose(prob_left, 0.5):
        return 1
    if np.isclose(prob_left, 0.8):
        return 2
    raise ValueError(f"Unexpected probabilityLeft value {prob_left}")
# ...
np.full(N_BINS, prior_code, dtype=np.uint8)
```

iii. The AI documented this as matching the instruction spec: "0.2 -> 0, 0.5 -> 1, 0.8 -> 2".

## 9-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. Wheel speed is derived from `_ibl_wheel.timestamps.npy` and `_ibl_wheel.position.npy`, which provide raw wheel position over time.

ii.
```python
def load_wheel_speed(one, row):
    ts_path = ensure_dataset_path_any(one, row,
        ["alf/_ibl_wheel.timestamps.npy", "alf/#*/_ibl_wheel.timestamps.npy"], ...)
    pos_path = ensure_dataset_path_any(one, row,
        ["alf/_ibl_wheel.position.npy", "alf/#*/_ibl_wheel.position.npy"], ...)
    timestamps = np.load(ts_path)
    position = np.load(pos_path)
```

iii. The AI documented using the raw wheel position data, which is then processed into speed.

## 9-b. What processing is involved in computing `output` *Wheel speed*?

i. Raw wheel position is interpolated to 1000 Hz using `brainbox.behavior.wheel.interpolate_position`, then velocity is computed using `velocity_filtered` (Gaussian-smoothed derivative). Speed is the absolute value of velocity. The continuous speed trace is then interpolated to the trial-aligned 20 ms time grid using linear interpolation.

ii.
```python
pos_interp, t_interp = interpolate_position(timestamps, position, freq=1000)
vel, _ = velocity_filtered(pos_interp, 1000)
return {"times": t_interp.astype(np.float32), "values": np.abs(vel).astype(np.float32)}
```

iii. The AI documented using brainbox wheel processing functions, which matches the general approach of the reference code (SessionLoader.load_wheel internally uses similar processing). The reference code uses `SessionLoader.load_wheel()` which provides interpolated wheel with velocity computed via Gaussian smoothing.

## 9-c. How is `output` *Wheel speed* thresholded into categories?

i. Continuous wheel speed values from all valid trials across all sessions are collected, and global tertile edges (1/3 and 2/3 quantiles) are computed. Each time bin's speed is then discretized into 3 bins (0=low, 1=mid, 2=high) using `np.digitize`.

ii.
```python
wheel_all = np.concatenate([np.concatenate(rec.wheel_values) for rec in session_records]).astype(np.float32)
wheel_edges = safe_quantile_edges(wheel_all)
# ...
def discretize(values, edges):
    return np.digitize(values, bins=edges, right=False).astype(np.uint8)
```

iii. The AI documented using global tertile bins to ensure consistent semantics across sessions, stored in metadata.

## 9-d. How is `output` *Wheel speed* aligned with the neural data?

i. Wheel speed is aligned by interpolating the continuous speed trace onto the same stimulus-onset-aligned time grid as the neural data. The interpolation uses `np.linspace(interval_beg + BIN_SIZE, interval_end, N_BINS)`, which matches the neural binning grid.

ii.
```python
x_interp = np.linspace(interval_begs[interval_idx] + BIN_SIZE, interval_ends[interval_idx], n_bins)
y_interp = interp1d(seg_t, seg_v, kind="linear", fill_value="extrapolate")(x_interp)
```

iii. The AI documented using the same temporal alignment as neural data (stimulus onset, [-0.5, 1.5] s window, 20 ms bins).

## 10-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. Whisker motion energy is derived from `leftCamera.ROIMotionEnergy.npy` (or `rightCamera.ROIMotionEnergy.npy` as fallback) and the corresponding camera timestamps (`_ibl_leftCamera.times.npy` or `_ibl_rightCamera.times.npy`).

ii.
```python
def load_whisker_motion_energy(one, row):
    sides = [
        ("left", "_ibl_leftCamera.times.npy", "leftCamera.ROIMotionEnergy.npy"),
        ("right", "_ibl_rightCamera.times.npy", "rightCamera.ROIMotionEnergy.npy"),
    ]
    for side, times_name, energy_name in sides:
        try:
            times_path = ensure_dataset_path_any(one, row, [...], times_name, "alf")
            energy_path = ensure_dataset_path(one, row, f"alf/#*/{energy_name}", energy_name, "alf")
            times = np.load(times_path).astype(np.float32)
            values = np.load(energy_path).astype(np.float32)
            return {"times": times, "values": values}, side
        except Exception:
            continue
```

iii. The AI documented preferring left camera with right camera fallback, matching the reference code's `bin_behaviors` logic.

## 10-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The raw whisker motion energy values are loaded directly (no additional computation needed beyond what was pre-computed in the IBL pipeline). The continuous trace is then interpolated to the trial-aligned 20 ms time grid using linear interpolation, same as wheel speed.

ii.
```python
whisker_values, whisker_mask = get_behavior_per_interval(
    whisker["times"], whisker["values"], trials_df, allow_nans=False)
```

iii. The AI documented using the same interpolation approach as wheel speed, consistent with the reference code.

## 10-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Same approach as wheel speed: global tertile edges are computed across all valid trials and sessions, then `np.digitize` assigns each time bin to one of 3 categories (0=low, 1=mid, 2=high).

ii.
```python
whisker_all = np.concatenate([np.concatenate(rec.whisker_values) for rec in session_records]).astype(np.float32)
whisker_edges = safe_quantile_edges(whisker_all)
# ...
discretize(whisker_vals, whisker_edges)
```

iii. The AI documented global tertile binning for consistent semantics across sessions.

## 10-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Same as wheel speed: whisker motion energy is interpolated onto the stimulus-onset-aligned time grid matching the neural data bins.

ii.
```python
# Same get_behavior_per_interval function as wheel
x_interp = np.linspace(interval_begs[interval_idx] + BIN_SIZE, interval_ends[interval_idx], n_bins)
y_interp = interp1d(seg_t, seg_v, kind="linear", fill_value="extrapolate")(x_interp)
```

iii. The AI documented using the same temporal alignment approach for all behavioral variables.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Several strategies are used: (1) Trials with NaN in key event columns are excluded via the trial mask. (2) Trials lacking full behavioral coverage (wheel or whisker data not spanning the trial window) are excluded via coverage masks. (3) Sessions missing whisker motion energy entirely (14 sessions) are skipped. (4) One session with zero valid trials after filtering is skipped. (5) Probes with zero good-quality clusters are skipped without failing the entire session. (6) NaN values in `probabilityLeft` reset the block trial counter. (7) Missing `cluster_id` columns are handled by falling back to positional indexing.

ii.
```python
# Coverage check in get_behavior_per_interval:
if len(seg_v) == 0:
    good = False
elif np.abs(interval_begs[interval_idx] - seg_t[0]) > BIN_SIZE:
    good = False
elif np.abs(interval_ends[interval_idx] - seg_t[-1]) > BIN_SIZE:
    good = False
# ...
# Probe skip:
if spikes is None or clusters is None:
    print(f"  skipping probe {probe_row.probe_name}: no well-isolated clusters")
    continue
```

iii. The AI documented each category of missing data and the handling strategy, noting that 15 sessions were dropped (14 for missing whisker data, 1 for zero valid trials).

## 12-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are: (1) Loading and merging probe data (spike sorting files) per session, (2) Binning spikes into trial windows using `bincount2D`, (3) Computing wheel velocity from interpolated position. The AI reported full cache fill took ~429 seconds for 444 sessions using 4 parallel workers.

ii.
```python
# Parallel processing for full mode:
with ProcessPoolExecutor(max_workers=max_workers) as pool:
    futures = {}
    for idx, row in enumerate(session_rows.itertuples(index=False)):
        futures[pool.submit(process_session_worker, ...)] = idx
```

iii. The AI documented timing information and implemented parallel processing and session caching to mitigate bottlenecks.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops could potentially be vectorized: (1) The `get_spike_data_per_interval` function loops over trials to bin spikes one trial at a time. (2) The `get_behavior_per_interval` function loops over trials for interpolation. (3) The `compute_trial_number_in_block` function uses a sequential loop. (4) The per-trial output assembly in `process_session` loops over valid indices.

ii.
```python
# Trial-by-trial spike binning loop:
for interval_idx, (ib, ie, t_beg, t_end) in enumerate(zip(idxs_beg, idxs_end, ...)):
    times_curr = times[ib:ie]
    clust_curr = clusters[ib:ie]
    # ...
    binned_tmp, _, cluster_idxs = bincount2D(times_curr, clust_curr, ...)

# Trial-by-trial behavior interpolation loop:
for interval_idx, (ib, ie) in enumerate(zip(idxs_beg, idxs_end)):
    # ...
    y_interp = interp1d(seg_t, seg_v, kind="linear", fill_value="extrapolate")(x_interp)
```

iii. The AI noted these loops but chose to use parallel processing at the session level rather than vectorizing within-session loops.

## 12-c. What processing does the code repeat multiple times?

i. (1) The `build_one()` function creates a new ONE instance for each worker process. (2) `BrainRegions()` is instantiated per worker. (3) The release CSV is read once in main but each worker also needs its own ONE connection. (4) When using `--from-session-cache`, session records are loaded from disk after having been written during processing, duplicating I/O.

ii.
```python
def process_session_worker(row_dict, probe_records, with_diagnostic, record_cache_dir=None):
    one = build_one()
    brain_regions = BrainRegions()
    # ...
```

iii. The AI documented the caching strategy to avoid repeated raw data loading across retries.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. (1) Spike binning is performed for ALL trials (including filtered ones) before the trial mask is applied; only valid trials are kept. (2) Behavioral interpolation is similarly done for all trials before masking. (3) Diagnostic data is collected for up to 2 sessions even when `--show-processing` is not used. (4) The `cluster_depths` array is loaded but not used in the final output. (5) The full raw wheel and whisker time series are loaded into diagnostic records even though only one trial's worth is plotted.

ii.
```python
# Binning ALL trials before filtering:
binned_spikes, cluster_ids = bin_spiking_data(spikes, trials_df)  # All trials
# ...
combined_mask = trial_mask.to_numpy(dtype=bool) & wheel_mask & whisker_mask
valid_idx = np.flatnonzero(combined_mask)  # Then filter
```

iii. The AI documented this as a design choice for simplicity, noting that binning all trials first and then filtering is consistent with the reference code's approach.
