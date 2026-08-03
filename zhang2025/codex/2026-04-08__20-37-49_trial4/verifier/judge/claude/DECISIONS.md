# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads the list of sessions from a CSV file (`bwm_release.csv`) bundled with the reference code, rather than using the ONE API's `search()` against the release tag. It then locates data files on disk by constructing paths from the session metadata (lab, subject, date, session number) and resolving ALF-revisioned directories with glob patterns. For each session, trials, wheel, whisker, and spike data are loaded directly from `.npy`/`.pqt` files. If files are not found locally, it attempts to download them via ONE.

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

iii. The AI noted that the reference helper stack (SessionLoader, SpikeSortingLoader) depended on unavailable optional packages in the execution environment, so it implemented direct ALF file loading instead. The CSV-based session list enumerates all 459 sessions in the release.

## 1-b. How are the data split into subjects?

i. Subjects are identified from the `subject` column in the release CSV. During dataset assembly, unique subjects are collected in encounter order and each session is mapped to its subject index.

ii.
```python
if rec.subject not in subject_to_idx:
    subject_to_idx[rec.subject] = len(subjects)
    subjects.append(rec.subject)
subject_idx.append(subject_to_idx[rec.subject])
```

iii. The subject field is available directly from the release metadata. The AI documents 136 subjects in the final dataset (3 lost from sessions that were skipped).

## 1-c. How are the data split into sessions?

i. Each row in the release CSV is one session identified by `eid`. Sessions are processed independently (in parallel for full mode) and assembled in the original CSV order.

ii.
```python
session_rows, probe_rows = load_release_sessions()
# ...
for row in session_rows.itertuples(index=False):
    rec = process_session(one, row, probe_rows[row.eid], brain_regions, ...)
```

iii. The CSV directly lists sessions as rows, so no splitting is needed.

## 1-d. How are the data split into trials?

i. Each session's trials table (`_ibl_trials.table.pqt`) has one row per trial. The AI reads this table and each row becomes one trial.

ii.
```python
trials_df = load_trials_table(one, row)
# ...
path = ensure_dataset_path(one, row, "alf/#*/_ibl_trials.table.pqt", "_ibl_trials.table.pqt", "alf")
trials_df = pd.read_parquet(path)
```

iii. The trials table is already organized with one row per trial.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies multiple filters via `build_trial_mask`: reaction time must be in [0.08, 2.0] s, no-choice trials (choice==0) are excluded, trials with NaN in key event columns are excluded (stimOn_times, choice, feedback_times, probabilityLeft, firstMovement_times, feedbackType), and trials with `feedback_times - goCue_times > 10 s` are excluded. Additionally, wheel and whisker coverage masks filter out trials where the behavioral trace doesn't span the full trial window.

ii.
```python
def build_trial_mask(trials_df, min_rt=0.08, max_rt=2.0, nan_exclude="default",
                     min_trial_len=None, max_trial_len=10.0, exclude_unbiased=False,
                     exclude_nochoice=True):
    if nan_exclude == "default":
        nan_exclude = ["stimOn_times", "choice", "feedback_times",
                       "probabilityLeft", "firstMovement_times", "feedbackType"]
    # ...
    if max_trial_len is not None:
        query += f" | (feedback_times - goCue_times > {max_trial_len})"
    # ...
```

```python
combined_mask = trial_mask.to_numpy(dtype=bool) & wheel_mask & whisker_mask
```

iii. The AI's CONVERSION_NOTES.md states this follows the reference code's `load_trials_and_mask` function, which includes the max trial length filter and NaN exclusions for multiple event columns.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `spikes.times.npy` and `spikes.clusters.npy` from each probe's pykilosort directory, plus `clusters.metrics.pqt` for quality labels and `channels.brainLocationIds_ccf_2017.npy` for region assignments.

ii.
```python
times_path = ensure_dataset_path(one, row, f"{collection}/#*/spikes.times.npy", ...)
clu_path = ensure_dataset_path(one, row, f"{collection}/#*/spikes.clusters.npy", ...)
spike_times = np.load(times_path).astype(np.float32)
spike_clusters = np.load(clu_path).astype(np.int32)
```

iii. These are the standard spike sorting outputs from the IBL pipeline.

## 2-b. How is the `neural` data processed?

i. Spikes are binned into 20 ms bins over the [-0.5, 1.5] s window using `bincount2D` from `iblutil.numerical`. Multi-probe sessions have their spikes merged (renumbered clusters, sorted by time). The result is stored as **raw spike counts** (uint8), NOT firing rates.

ii.
```python
binned_tmp, _, cluster_idxs = bincount2D(times_curr, clust_curr, xbin=binsize, xlim=[t_beg, t_end])
# ...
binned_spikes[interval_idx, idxs_tmp, :] = binned_tmp[:, :n_bins]
# ...
neural_trial = compact_neural_trial(binned_spikes[idx].T)
# compact_neural_trial converts to uint8
```

iii. The AI stored raw spike counts rather than firing rates (counts/bin_width) to keep the file size small. The CONVERSION_NOTES state the trainer converts these during training.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only clusters with `label >= 1` in `clusters.metrics.pqt` are retained. This matches the paper's "75,708 well-isolated neurons" criterion. Probes with zero good clusters are skipped.

ii.
```python
good_mask = clusters_df["label"].fillna(0).to_numpy() >= 1
# ...
spike_keep = np.isin(spike_clusters, good_cluster_ids)
spike_times = spike_times[spike_keep]
spike_clusters = spike_clusters[spike_keep]
```

iii. The AI verified that `label >= 1` reproduces the paper's 75,708 count exactly from the raw release data.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trial intervals are defined as `[stimOn_times + T_START, stimOn_times + T_STOP]` = `[stimOn_times - 0.5, stimOn_times + 1.5]`. Spikes within each interval are binned relative to the interval start using `bincount2D`.

ii.
```python
intervals = np.vstack([
    trials_df[ALIGN_EVENT].to_numpy() + TIME_WINDOW[0],
    trials_df[ALIGN_EVENT].to_numpy() + TIME_WINDOW[1],
]).T
binned_array, cluster_ids = get_spike_data_per_interval(
    spikes["times"], spikes["clusters"],
    interval_begs=intervals[:, 0], interval_ends=intervals[:, 1], ...)
```

iii. Alignment is to stimulus onset (`stimOn_times`), matching the reference code.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The bin size is 20 ms, producing 100 bins over the 2 s window. No rebinning is applied. However, the AI's time grid uses **bin end times** rather than bin centers.

ii.
```python
BIN_SIZE = 0.02
N_BINS = int(np.ceil((TIME_WINDOW[1] - TIME_WINDOW[0]) / BIN_SIZE))  # 100
TIME_GRID = np.linspace(TIME_WINDOW[0] + BIN_SIZE, TIME_WINDOW[1], N_BINS, dtype=np.float32)
# = [-0.48, -0.46, ..., 1.48, 1.50]
```

iii. The AI's CONVERSION_NOTES confirms 20 ms bins and 100 bins per trial, matching the reference.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. From the fixed time grid defined by `TIME_GRID`, which represents evenly spaced time points relative to stimulus onset. The grid is derived from the alignment event `stimOn_times` and the window parameters.

ii.
```python
TIME_GRID = np.linspace(TIME_WINDOW[0] + BIN_SIZE, TIME_WINDOW[1], N_BINS, dtype=np.float32)
```

iii. The time input is a deterministic grid, not derived from raw data variables.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The time grid is computed as `np.linspace(-0.48, 1.5, 100)`, which produces bin **end times** (right edges) rather than bin centers. The reference uses bin centers: `EDGES[:-1] + BIN/2` = `[-0.49, -0.47, ..., 1.49]`. This results in a systematic 10 ms offset.

ii.
```python
# AI:
TIME_GRID = np.linspace(TIME_WINDOW[0] + BIN_SIZE, TIME_WINDOW[1], N_BINS, dtype=np.float32)
# [-0.48, -0.46, ..., 1.48, 1.50]

# Reference:
EDGES = T_START + BIN * np.arange(N_BINS + 1)
TIME = EDGES[:-1] + BIN / 2  # [-0.49, -0.47, ..., 1.49]
```

iii. The AI's CONVERSION_NOTES document the range as `[-0.47998, 1.5]` (the float32 representation of -0.48).

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The same `TIME_GRID` is used for interpolating behavioral traces and constructing the time input, but the neural data is binned by `bincount2D` with `xlim=[t_beg, t_end]` which may use different internal bin edges. The behavioral traces are interpolated at bin end times while neural spikes are binned starting from `t_beg`.

ii.
```python
# Neural binning:
binned_tmp, _, cluster_idxs = bincount2D(times_curr, clust_curr, xbin=binsize, xlim=[t_beg, t_end])

# Behavior interpolation:
x_interp = np.linspace(interval_begs[interval_idx] + BIN_SIZE, interval_ends[interval_idx], n_bins)
y_interp = interp1d(seg_t, seg_v, kind="linear", fill_value="extrapolate")(x_interp)
```

iii. The AI uses the same grid for behavior and the time input, which is internally consistent.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. From `probabilityLeft` in the trials table. A change in `probabilityLeft` value signals a new block boundary.

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

iii. The CONVERSION_NOTES state that block structure is recovered from changes in `probabilityLeft`, computed on the full trial order before filtering.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The AI counts trials within each block starting from **1** (1-indexed). The reference starts from **0** (0-indexed, using `cumcount()`). The AI also handles NaN values in `probabilityLeft` by setting the trial number to NaN and resetting the counter.

ii.
```python
# AI: starts at 1
if prev is None or np.isnan(prev) or not np.isclose(prev, value):
    count = 1
else:
    count += 1

# Reference: starts at 0
block = (trials.probabilityLeft != trials.probabilityLeft.shift()).cumsum()
trial_in_block = trials.groupby(block).cumcount()  # 0-indexed
```

iii. The AI's approach is computed before trial filtering, preserving the animal's true position in the block.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. From the `choice` column of the trials table, which takes values +1 (left), -1 (right), or 0 (no response).

ii.
```python
def map_choice(choice_value: float) -> int:
    if np.isclose(choice_value, 1.0):
        return 0
    if np.isclose(choice_value, -1.0):
        return 1
    raise ValueError(f"Unexpected choice value {choice_value}")
```

iii. No-choice trials (choice==0) are excluded by the trial mask.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The mapping is +1 -> 0 (left) and -1 -> 1 (right), matching the reference and instructions. The value is broadcast to all 100 time bins as a per-trial constant.

ii.
```python
output_trial = np.vstack([
    np.full(N_BINS, choice_code, dtype=np.uint8),
    # ...
])
```

iii. Same mapping as the reference.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. From the `probabilityLeft` column of the trials table, which takes values 0.2, 0.5, or 0.8.

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

iii. Matches the instructions' mapping exactly.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Direct categorical mapping: 0.2 -> 0, 0.5 -> 1, 0.8 -> 2. Broadcast to all time bins.

ii.
```python
np.full(N_BINS, prior_code, dtype=np.uint8),
```

iii. Same as the reference.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. From `_ibl_wheel.timestamps.npy` and `_ibl_wheel.position.npy`. The AI loads these raw files directly and computes velocity using `interpolate_position` and `velocity_filtered` from `brainbox.behavior.wheel`.

ii.
```python
def load_wheel_speed(one, row):
    timestamps = np.load(ts_path)
    position = np.load(pos_path)
    pos_interp, t_interp = interpolate_position(timestamps, position, freq=1000)
    vel, _ = velocity_filtered(pos_interp, 1000)
    return {"times": t_interp.astype(np.float32), "values": np.abs(vel).astype(np.float32)}
```

iii. The AI uses the same underlying processing functions as `SessionLoader.load_wheel()`.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Position is interpolated to 1000 Hz, velocity is computed with a low-pass Butterworth filter, and the absolute value gives speed. The speed trace is then interpolated onto the trial time grid using `scipy.interpolate.interp1d` with linear interpolation and extrapolation.

ii.
```python
x_interp = np.linspace(interval_begs[interval_idx] + BIN_SIZE, interval_ends[interval_idx], n_bins)
y_interp = interp1d(seg_t, seg_v, kind="linear", fill_value="extrapolate")(x_interp)
```

iii. The underlying wheel processing matches the reference. The interpolation target times differ slightly (bin end times vs. bin centers).

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. The AI computes **global** tertile edges across ALL sessions combined, then applies `np.digitize` to categorize into 3 bins. The reference uses **per-session** percentiles (33rd and 67th).

ii.
```python
# Global edges computed across all sessions:
wheel_all = np.concatenate([np.concatenate(rec.wheel_values) for rec in session_records]).astype(np.float32)
wheel_edges = safe_quantile_edges(wheel_all)

# Applied per session:
discretize(wheel_vals, wheel_edges)

def safe_quantile_edges(values):
    q1, q2 = np.quantile(values, [1.0 / 3.0, 2.0 / 3.0])
    # ...
```

iii. The CONVERSION_NOTES document "global tertile edges so wheel and whisker bins have consistent semantics across sessions." The reference uses per-session percentiles.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The wheel trace is interpolated onto the same `TIME_GRID` (bin end times) used for the time input. This grid is consistent with the behavioral data but uses bin end times rather than the bin centers used by the reference for neural binning.

ii.
```python
x_interp = np.linspace(interval_begs[interval_idx] + BIN_SIZE, interval_ends[interval_idx], n_bins)
```

iii. Internally consistent alignment between behavior and inputs.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. From `leftCamera.ROIMotionEnergy.npy` (preferred) or `rightCamera.ROIMotionEnergy.npy` (fallback), with corresponding `_ibl_<side>Camera.times.npy`.

ii.
```python
sides = [
    ("left", "_ibl_leftCamera.times.npy", "leftCamera.ROIMotionEnergy.npy"),
    ("right", "_ibl_rightCamera.times.npy", "rightCamera.ROIMotionEnergy.npy"),
]
for side, times_name, energy_name in sides:
    # try to load each in order
```

iii. Left camera is preferred with right as fallback, matching the reference.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The raw motion energy trace is loaded directly (no additional filtering or normalization) and interpolated onto the trial time grid using `scipy.interpolate.interp1d`.

ii.
```python
times = np.load(times_path).astype(np.float32)
values = np.load(energy_path).astype(np.float32)
# Then interpolated via get_behavior_per_interval
```

iii. Same as wheel: direct interpolation to the time grid, no additional processing.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Same as wheel speed: **global** tertile edges across all sessions, then `np.digitize` into 3 bins. The reference uses **per-session** percentiles.

ii.
```python
whisker_all = np.concatenate([np.concatenate(rec.whisker_values) for rec in session_records]).astype(np.float32)
whisker_edges = safe_quantile_edges(whisker_all)
# ...
discretize(whisker_vals, whisker_edges)
```

iii. Same global-edge approach as for wheel speed.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Same as wheel: interpolated onto `TIME_GRID` (bin end times).

ii.
```python
x_interp = np.linspace(interval_begs[interval_idx] + BIN_SIZE, interval_ends[interval_idx], n_bins)
y_interp = interp1d(seg_t, seg_v, kind="linear", fill_value="extrapolate")(x_interp)
```

iii. Internally consistent alignment.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Multiple layers of handling: (1) NaN values in key trial columns cause trial exclusion via the trial mask. (2) Trials without full wheel or whisker coverage are excluded via per-trial masks. (3) Probes with zero good clusters are skipped (only the probe, not the session). (4) Sessions with fewer than 2 valid trials or no probes with good units raise exceptions and are skipped. (5) NaN in `probabilityLeft` resets the block trial counter.

ii.
```python
# NaN exclusion in trial mask:
for event in nan_exclude:
    query += f" | {event}.isnull()"

# Probe with no good units:
if good_cluster_ids.size == 0:
    return None, None

# Session with too few trials:
if len(valid_idx) < 2:
    raise RuntimeError(f"Need at least 2 valid trials after filtering, found {len(valid_idx)}")
```

iii. The AI documents handling NaN-valued `probabilityLeft` and missing whisker data specifically.

## 10-a. What are the most time-consuming steps of the code?

i. Loading spike sorting data from disk (large `.npy` files with spike times and clusters), followed by spike binning across all trials. The AI's conversion_full_out.txt shows the full conversion from cached session records took 15.75s, but the initial cache fill took ~429s.

ii.
```python
spike_times = np.load(times_path).astype(np.float32)
spike_clusters = np.load(clu_path).astype(np.int32)
```

iii. The AI implemented session-level caching and parallel processing to amortize the I/O cost.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The `get_spike_data_per_interval` function loops over trials one at a time, calling `bincount2D` per trial. The `get_behavior_per_interval` function similarly loops per trial for interpolation. The `compute_trial_number_in_block` function uses a Python-level for loop over all trials.

ii.
```python
# Per-trial spike binning loop:
for interval_idx, (ib, ie, t_beg, t_end) in enumerate(zip(idxs_beg, idxs_end, ...)):
    binned_tmp, _, cluster_idxs = bincount2D(times_curr, clust_curr, ...)

# Per-trial behavior interpolation loop:
for interval_idx, (ib, ie) in enumerate(zip(idxs_beg, idxs_end)):
    y_interp = interp1d(seg_t, seg_v, ...)(x_interp)

# Per-trial block number computation:
for idx, value in enumerate(probability_left):
    # ...
```

iii. The spike binning and behavior interpolation loops mirror the reference code's structure.

## 10-c. What processing does the code repeat multiple times?

i. The code processes spike binning and behavior interpolation for ALL trials (before filtering), then selects only the valid trials. This means binning and interpolation are computed for trials that will be discarded. In contrast, the reference first determines valid trials, then only processes those.

ii.
```python
# All trials are binned:
binned_spikes, cluster_ids = bin_spiking_data(spikes, trials_df)  # all trials
wheel_values, wheel_mask = get_behavior_per_interval(wheel["times"], wheel["values"], trials_df)  # all trials
whisker_values, whisker_mask = get_behavior_per_interval(whisker["times"], whisker["values"], trials_df)  # all trials

# Then only valid ones are selected:
combined_mask = trial_mask.to_numpy(dtype=bool) & wheel_mask & whisker_mask
valid_idx = np.flatnonzero(combined_mask)
for idx in valid_idx:
    neural_trial = compact_neural_trial(binned_spikes[idx].T)
```

iii. This design is intentional to determine wheel/whisker coverage masks, but it wastes computation on neural binning for invalid trials.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code bins neural data for ALL trials (including filtered ones), then discards those for invalid trials. It also computes and stores diagnostic data per session when `--show-processing` is enabled. The double transpose of neural arrays (`.T` in `bin_spiking_data`, then `.T` again in `process_session`) is redundant.

ii.
```python
# Neural binning for all trials (many discarded):
binned_spikes, cluster_ids = bin_spiking_data(spikes, trials_df)

# Double transpose:
# In bin_spiking_data:
binned_trials = np.asarray([x.T for x in binned_array], dtype=np.float32)
# In process_session:
neural_trial = compact_neural_trial(binned_spikes[idx].T)
```

iii. The wasted computation on filtered trials is the main efficiency concern.
