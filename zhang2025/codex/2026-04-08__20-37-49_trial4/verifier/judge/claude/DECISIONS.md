# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads a CSV file (`bwm_release.csv`) from the reference code directory to enumerate all 459 sessions and their probe insertions. It does not use the ONE API search to discover sessions. Instead, it parses the CSV for session eids, subjects, labs, dates, session numbers, and probe info. Data files are then located by constructing file paths from the session metadata and globbing for matching files in both a readonly cache and a writable cache directory. If files are not found locally, it attempts to download them via ONE.

ii.
```python
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

```python
def locate_dataset(row, relative_glob: str):
    rel = session_rel_path(row)
    for root in (READONLY_CACHE, WRITABLE_CACHE):
        base = root / rel
        matches = sorted(base.glob(relative_glob))
        if matches:
            return matches[-1]
    return None
```

iii. The AI chose to use the release CSV rather than the ONE API search because it provides a definitive list of sessions and probes in the release. The CONVERSION_NOTES document this as using the "459-session release list as canonical."

## 1-b. How are the data split into subjects?

i. Subjects are identified from the `subject` column of the release CSV. When building the final dataset, subjects are collected in insertion order (not sorted) into a list, and each session gets an index into that list.

ii.
```python
for rec in session_records:
    if rec.subject not in subject_to_idx:
        subject_to_idx[rec.subject] = len(subjects)
        subjects.append(rec.subject)
    subject_idx.append(subject_to_idx[rec.subject])
```

iii. Subject identity comes directly from the release metadata. No parsing of paths or filenames is needed.

## 1-c. How are the data split into sessions?

i. Each row in the release CSV corresponds to a probe insertion; sessions are identified by unique `eid`. The code deduplicates by eid to get one row per session and iterates over those.

ii.
```python
session_rows = (
    bwm.drop_duplicates("eid")[["eid", "subject", "lab", "date", "session_number"]]
    .reset_index(drop=True)
)
```

iii. Sessions are already uniquely identified by eid in the release.

## 1-d. How are the data split into trials?

i. The trials table (`_ibl_trials.table.pqt`) is loaded per session, with one row per trial. Each trial is processed individually.

ii.
```python
def load_trials_table(one: ONE, row):
    path = ensure_dataset_path(one, row, "alf/#*/_ibl_trials.table.pqt", "_ibl_trials.table.pqt", "alf")
    trials_df = pd.read_parquet(path)
```

iii. The trials table naturally has one row per trial.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies a trial mask that excludes: (1) trials with reaction time outside [0.08, 2.0]s, (2) no-choice trials (choice == 0), (3) trials with NaN in key columns (stimOn_times, choice, feedback_times, probabilityLeft, firstMovement_times, feedbackType), (4) trials with feedback_times - goCue_times > 10s, and (5) trials without full wheel and whisker motion energy coverage in the [-0.5, 1.5]s window.

ii.
```python
def build_trial_mask(
    trials_df: pd.DataFrame,
    min_rt=0.08, max_rt=2.0,
    nan_exclude="default",
    min_trial_len=None, max_trial_len=10.0,
    exclude_unbiased=False, exclude_nochoice=True,
):
    if nan_exclude == "default":
        nan_exclude = [
            "stimOn_times", "choice", "feedback_times",
            "probabilityLeft", "firstMovement_times", "feedbackType",
        ]
    ...
    if max_trial_len is not None:
        query += f" | (feedback_times - goCue_times > {max_trial_len})"
    ...
    if exclude_nochoice:
        query += " | (choice == 0)"
    return ~trials_df.eval(query)
```

```python
combined_mask = trial_mask.to_numpy(dtype=bool) & wheel_mask & whisker_mask
```

iii. The AI documents these filters as matching "the reference missing-event, RT, and max-trial-length logic" from ibl_data_utils.py. The max_trial_len=10.0 filter and the NaN check on goCue_times/feedbackType go beyond what the reference solution applies.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI loads `spikes.times.npy` and `spikes.clusters.npy` per probe, along with `clusters.metrics.pqt` for quality labels, `clusters.channels.npy` for channel assignments, and `channels.brainLocationIds_ccf_2017.npy` for brain region IDs.

ii.
```python
spike_times = np.load(times_path).astype(np.float32)
spike_clusters = np.load(clu_path).astype(np.int32)
cluster_metrics = pd.read_parquet(cluster_metrics_path)
```

iii. These are the standard spike sorting outputs from the IBL pipeline.

## 2-b. How is the `neural` data processed?

i. Spikes are counted into 20ms bins over the [-0.5, 1.5]s window using `bincount2D`. The counts are stored directly as uint8 spike counts, NOT converted to firing rates. When a session has multiple probes, their clusters are merged (renumbered continuously).

ii.
```python
binned_tmp, _, cluster_idxs = bincount2D(times_curr, clust_curr, xbin=binsize, xlim=[t_beg, t_end])
binned_spikes[interval_idx, idxs_tmp, :] = binned_tmp[:, :n_bins]
```

```python
neural_trial = compact_neural_trial(binned_spikes[idx].T)
# compact_neural_trial converts to uint8
```

iii. The AI chose to store spike counts rather than firing rates to save memory, noting that "the trainer converts them during training." The CONVERSION_NOTES mention that "Compact storage (uint8 neural, float16 inputs/continuous traces) to reduce converted_data.pkl to 3.118 GB."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only clusters with `label >= 1` are kept. Clusters with zero good units on a probe cause that probe to be skipped. However, `void` brain regions (channels placed outside the brain by histology) are NOT filtered out.

ii.
```python
good_mask = clusters_df["label"].fillna(0).to_numpy() >= 1
...
spike_keep = np.isin(spike_clusters, good_cluster_ids)
spike_times = spike_times[spike_keep]
spike_clusters = spike_clusters[spike_keep]
```

iii. The AI documents using `label >= 1` because "raw release reproduces the paper's 75,708 well-isolated-unit count exactly under this filter." The AI does not mention void filtering in its code despite the CONVERSION_NOTES referencing it.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial's spike window is defined as [stimOn_times + TIME_WINDOW[0], stimOn_times + TIME_WINDOW[1]] = [stimOn_times - 0.5, stimOn_times + 1.5]. The `bincount2D` function bins spikes within this absolute time window.

ii.
```python
intervals = np.vstack([
    trials_df[ALIGN_EVENT].to_numpy() + TIME_WINDOW[0],
    trials_df[ALIGN_EVENT].to_numpy() + TIME_WINDOW[1],
]).T
```

iii. Alignment is done by defining absolute start/end times for each trial window around stimulus onset.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The bin size is 20ms, producing 100 bins per trial over the 2s window. No rebinning or smoothing is applied.

ii.
```python
BIN_SIZE = 0.02
N_BINS = int(np.ceil((TIME_WINDOW[1] - TIME_WINDOW[0]) / BIN_SIZE))
```

iii. Matches the reference code's 20ms bins.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is derived from the time window parameters and bin size, not from any raw data variable. It is a deterministic time grid.

ii.
```python
TIME_GRID = np.linspace(TIME_WINDOW[0] + BIN_SIZE, TIME_WINDOW[1], N_BINS, dtype=np.float32)
```

iii. The time grid is defined by the decoding window parameters.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The AI computes the time grid as `np.linspace(-0.48, 1.5, 100)`, which gives the RIGHT EDGES of each bin, not the bin centers. The first value is -0.48 (not -0.49 as it would be for bin centers).

ii.
```python
TIME_GRID = np.linspace(TIME_WINDOW[0] + BIN_SIZE, TIME_WINDOW[1], N_BINS, dtype=np.float32)
# This gives: [-0.48, -0.46, ..., 1.48, 1.5]
```

iii. The AI's CONVERSION_NOTES state "Range is [-0.47998, 1.5] because the grid stores bin-end times." This is a deliberate choice to use right edges rather than bin centers.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The same TIME_GRID is used as the interpolation target for behavioral variables. However, the neural data is binned using `bincount2D` which counts spikes in bins starting from `interval_begs`, while TIME_GRID represents the right edges of those bins. So the time input values represent right edges while the neural bins count spikes within each [left_edge, right_edge) interval.

ii.
```python
input_trial = np.vstack([
    TIME_GRID,
    np.full(N_BINS, trial_num, dtype=np.float32),
]).astype(np.float16)
```

iii. The AI uses the same time grid for both neural binning alignment and the time input.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. Derived from the `probabilityLeft` column of the trials table, which is constant within a block.

ii.
```python
def compute_trial_number_in_block(probability_left: np.ndarray) -> np.ndarray:
    ...
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

iii. Block boundaries are inferred from changes in `probabilityLeft`.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The AI counts trials within each block starting from 1 (not 0). The count is computed on the full unfiltered trial table, so filtered-out trials still advance the counter. The value is then broadcast to all 100 time bins.

ii.
```python
count = 1  # starts at 1 for new block
...
count += 1
out[idx] = count
```

```python
input_trial = np.vstack([
    TIME_GRID,
    np.full(N_BINS, trial_num, dtype=np.float32),
]).astype(np.float16)
```

iii. The count starts at 1 rather than 0, which differs from the reference's `cumcount()` (which starts at 0).

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. From the `choice` column of the trials table, which has values +1 (left), -1 (right), and 0 (no response).

ii.
```python
def map_choice(choice_value: float) -> int:
    if np.isclose(choice_value, 1.0):
        return 0
    if np.isclose(choice_value, -1.0):
        return 1
    raise ValueError(f"Unexpected choice value {choice_value}")
```

iii. Standard IBL convention mapping.

## 5-b. What processing is involved in computing `output` *Choice*?

i. Choice is recoded: +1 (left) -> 0, -1 (right) -> 1. No-choice trials are excluded by the trial mask. The value is broadcast to all 100 time bins.

ii.
```python
np.full(N_BINS, choice_code, dtype=np.uint8),
```

iii. Simple recoding as specified in the instructions.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. From the `probabilityLeft` column of the trials table.

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
```

iii. Direct mapping as specified in the instructions.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Recoded from continuous values: 0.2 -> 0, 0.5 -> 1, 0.8 -> 2. Broadcast to all 100 time bins.

ii.
```python
np.full(N_BINS, prior_code, dtype=np.uint8),
```

iii. Simple recoding as specified.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. From `_ibl_wheel.position.npy` and `_ibl_wheel.timestamps.npy`.

ii.
```python
def load_wheel_speed(one: ONE, row):
    ...
    timestamps = np.load(ts_path)
    position = np.load(pos_path)
    pos_interp, t_interp = interpolate_position(timestamps, position, freq=1000)
    vel, _ = velocity_filtered(pos_interp, 1000)
    return {"times": t_interp.astype(np.float32), "values": np.abs(vel).astype(np.float32)}
```

iii. Uses the same brainbox functions as the reference for wheel processing.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The wheel position is interpolated to 1000 Hz, differentiated with a Butterworth low-pass filter to get velocity, and the absolute value is taken to get speed. The speed trace is then interpolated onto the trial time grid using `interp1d` with extrapolation. Finally, the continuous speed values are discretized into 3 bins using GLOBAL tertile edges (33rd and 67th percentiles computed across ALL sessions' concatenated wheel speed values).

ii.
```python
pos_interp, t_interp = interpolate_position(timestamps, position, freq=1000)
vel, _ = velocity_filtered(pos_interp, 1000)
return {"times": t_interp, "values": np.abs(vel)}
```

```python
x_interp = np.linspace(interval_begs[interval_idx] + BIN_SIZE, interval_ends[interval_idx], n_bins)
y_interp = interp1d(seg_t, seg_v, kind="linear", fill_value="extrapolate")(x_interp)
```

```python
wheel_all = np.concatenate([np.concatenate(rec.wheel_values) for rec in session_records]).astype(np.float32)
wheel_edges = safe_quantile_edges(wheel_all)
```

iii. The AI chose global discretization edges rather than per-session edges "so wheel and whisker bins have consistent semantics across sessions."

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. The AI uses GLOBAL tertile edges: the 33rd and 67th percentiles are computed over the concatenated wheel speed values from ALL sessions. Values are then digitized into 3 bins using these global edges.

ii.
```python
def safe_quantile_edges(values: np.ndarray):
    q1, q2 = np.quantile(values, [1.0 / 3.0, 2.0 / 3.0])
    ...
    return np.asarray([q1, q2], dtype=np.float32)

def discretize(values: np.ndarray, edges: np.ndarray):
    return np.digitize(values, bins=edges, right=False).astype(np.uint8)
```

iii. Global edges ensure consistent bin semantics across sessions.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The wheel speed is interpolated onto the same time grid (TIME_GRID = right edges of bins) as the neural data, so they share the same temporal axis.

ii.
```python
x_interp = np.linspace(interval_begs[interval_idx] + BIN_SIZE, interval_ends[interval_idx], n_bins)
y_interp = interp1d(seg_t, seg_v, kind="linear", fill_value="extrapolate")(x_interp)
```

iii. Same time grid used for all variables ensures alignment.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. From `leftCamera.ROIMotionEnergy.npy` (preferred) or `rightCamera.ROIMotionEnergy.npy` (fallback) with corresponding camera timestamps.

ii.
```python
def load_whisker_motion_energy(one: ONE, row):
    sides = [
        ("left", "_ibl_leftCamera.times.npy", "leftCamera.ROIMotionEnergy.npy"),
        ("right", "_ibl_rightCamera.times.npy", "rightCamera.ROIMotionEnergy.npy"),
    ]
    ...
    for side, times_name, energy_name in sides:
        try:
            ...
            return {"times": times, "values": values}, side
```

iii. Left camera preferred, right as fallback, matching reference behavior.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The raw motion energy trace is used as-is (no filtering or normalization). It is interpolated onto the trial time grid using `interp1d` with extrapolation, then discretized into 3 bins using GLOBAL tertile edges.

ii.
```python
whisker_values, whisker_mask = get_behavior_per_interval(
    whisker["times"], whisker["values"], trials_df, allow_nans=False
)
```

```python
whisker_all = np.concatenate([np.concatenate(rec.whisker_values) for rec in session_records]).astype(np.float32)
whisker_edges = safe_quantile_edges(whisker_all)
```

iii. Same global discretization approach as wheel speed.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Same approach as wheel speed: global tertile edges computed across all sessions.

ii.
```python
whisker_edges = safe_quantile_edges(whisker_all)
discretize(whisker_vals, whisker_edges)
```

iii. Global edges for consistent semantics across sessions.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Interpolated onto the same TIME_GRID as neural data.

ii.
```python
x_interp = np.linspace(interval_begs[interval_idx] + BIN_SIZE, interval_ends[interval_idx], n_bins)
y_interp = interp1d(seg_t, seg_v, kind="linear", fill_value="extrapolate")(x_interp)
```

iii. Same time grid ensures temporal alignment.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Multiple strategies: (1) Probes with zero good clusters are skipped rather than failing the session. (2) Sessions with < 2 valid trials after filtering are dropped. (3) Trials without full wheel or whisker coverage are excluded. (4) Sessions missing whisker motion energy entirely are dropped (15 sessions). (5) NaN values in key trial columns cause trial exclusion.

ii.
```python
if spikes is None or clusters is None:
    print(f"  skipping probe {probe_row.probe_name}: no well-isolated clusters", flush=True)
    continue
```

```python
if len(valid_idx) < 2:
    raise RuntimeError(f"Need at least 2 valid trials after filtering, found {len(valid_idx)}")
```

iii. The AI handles missing data by dropping affected trials or sessions.

## 10-a. What are the most time-consuming steps of the code?

i. Loading spike sorting data from disk (hundreds of MB per probe) and processing sessions in parallel using ProcessPoolExecutor. The full conversion from cached session records takes ~16s; the initial cache fill took ~429s.

ii.
```python
with ProcessPoolExecutor(max_workers=max_workers) as pool:
    ...
```

iii. The AI implemented session-level caching to avoid repeated expensive I/O.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The `get_spike_data_per_interval` function loops over trials to bin spikes. The `get_behavior_per_interval` function loops over trials to interpolate behavior. The `compute_trial_number_in_block` function loops over all trials to compute block positions. These per-trial loops could potentially be vectorized.

ii.
```python
for interval_idx, (ib, ie, t_beg, t_end) in enumerate(zip(idxs_beg, idxs_end, interval_begs, interval_ends)):
    ...
```

```python
for interval_idx, (ib, ie) in enumerate(zip(idxs_beg, idxs_end)):
    ...
```

iii. The AI addressed performance through parallelism rather than vectorization.

## 10-c. What processing does the code repeat multiple times?

i. The code bins spikes for ALL trials in the trials table (including filtered-out trials) and then selects only valid trials afterward. This means spike binning is done for trials that are ultimately discarded. Similarly, behavioral interpolation is done for all trials before the combined mask is applied.

ii.
```python
binned_spikes, cluster_ids = bin_spiking_data(spikes, trials_df)  # all trials
...
combined_mask = trial_mask.to_numpy(dtype=bool) & wheel_mask & whisker_mask
valid_idx = np.flatnonzero(combined_mask)  # then filter
```

iii. This is wasteful but doesn't affect correctness.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code processes spike data for all trials (including those that will be filtered out), which is discarded. It also stores extensive diagnostic data for plotting that is not used in the final dataset. The session caching infrastructure adds complexity that is only useful during development.

ii.
```python
binned_spikes, cluster_ids = bin_spiking_data(spikes, trials_df)  # bins ALL trials
# ... only valid_idx trials are kept
```

iii. Processing all trials before filtering is simpler but wastes computation.
