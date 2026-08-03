# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI discovers sessions from `bwm_release.csv`, iterates over each unique `eid`, resolves the local path as `data/one_cache/<lab>/Subjects/<subject>/<date>/<session_number>/alf/`, and loads trials, wheel, whisker motion energy, and spike data per probe. Multiple probes within a session are merged. All data are loaded from local ALF files (numpy, parquet) rather than using the ONE API.

ii.
```python
BWM_RELEASE_CSV = Path("/app/code/code_zhang2025/data/bwm_release.csv")
ONE_CACHE_DIR = Path("/app/data/one_cache")
# ...
release_df = pd.read_csv(BWM_RELEASE_CSV)
session_rows = []
for eid, group in release_df.groupby("eid", sort=False):
    first = group.iloc[0]
    session_rows.append({
        "eid": eid,
        "subject": first["subject"],
        "lab": first["lab"],
        "date": first["date"],
        "session_number": int(first["session_number"]),
        "probe_names": list(group["probe_name"]),
    })
```

iii. The agent chose this approach because the reference code (`0_data_caching.py`) also loads from `bwm_release.csv` and iterates over sessions. The agent bypassed the ONE API and loaded files directly from the local cache to avoid network dependency.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are derived from the `subject` column of `bwm_release.csv`. Each session row has an associated subject name. After conversion, unique subjects are collected in insertion order and a `subject_idx` array maps each session to its subject.

ii.
```python
subjects = ordered_unique(session["subject"] for session in kept_sessions)
subject_lookup = {subject: idx for idx, subject in enumerate(subjects)}
# ...
"subject_idx": np.asarray(
    [subject_lookup[session["subject"]] for session in kept_sessions],
    dtype=np.int64,
),
```

iii. The agent identified that subjects come from the CSV metadata, consistent with the reference code's use of `bwm_df.subject`.

## 1-c. How are the data split into sessions?

i. Each unique `eid` in `bwm_release.csv` constitutes one session. The agent groups the release CSV by `eid` to get the list of sessions and their associated probes. Sessions that fail to load any required data stream (trials, wheel, whisker, neural) are dropped.

ii.
```python
for eid, group in release_df.groupby("eid", sort=False):
    # ... build session_rows with eid, subject, lab, date, session_number, probe_names
```

iii. This matches the reference code which also iterates over eids from `bwm_release.csv`.

## 1-d. How are the data split into trials?

i. For each session, the trials table is loaded from `_ibl_trials.table.pqt`. Trials are iterated over row by row, and each trial that passes the quality mask and has valid behavioral data is retained as a separate element in the session's trial list.

ii.
```python
trials_df = load_trials_table(alf_path)
trial_mask = compute_trial_mask(trials_df)
# ...
for trial_idx, trial_row in trials_df.iterrows():
    if not trial_mask[trial_idx]:
        continue
    trial_start = float(trial_row["stimOn_times"] + OFF_START_S)
    trial_end = float(trial_row["stimOn_times"] + OFF_END_S)
    # ... process and append trial data
```

iii. The agent followed the reference approach of loading the trials table and applying a mask, then extracting per-trial windows aligned to `stimOn_times`.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered by requiring finite values for 7 columns (`stimOn_times`, `choice`, `feedback_times`, `probabilityLeft`, `firstMovement_times`, `feedbackType`, `goCue_times`), excluding no-choice trials (`choice != 0`), constraining reaction time to [0.08, 2.0] s, and constraining trial length (`feedback_times - goCue_times <= 10.0` s). Additionally, trials where wheel or whisker interpolation fails, or where no spikes are present, are excluded.

ii.
```python
def compute_trial_mask(trials_df: pd.DataFrame):
    required = [
        "stimOn_times", "choice", "feedback_times", "probabilityLeft",
        "firstMovement_times", "feedbackType", "goCue_times",
    ]
    mask = np.ones(len(trials_df), dtype=bool)
    for column in required:
        mask &= np.isfinite(trials_df[column].to_numpy())
    rt = trials_df["firstMovement_times"].to_numpy() - trials_df["stimOn_times"].to_numpy()
    trial_len = trials_df["feedback_times"].to_numpy() - trials_df["goCue_times"].to_numpy()
    mask &= rt >= MIN_RT_S
    mask &= rt <= MAX_RT_S
    mask &= trial_len <= MAX_TRIAL_LEN_S
    mask &= trials_df["choice"].to_numpy() != 0
    return mask
```

iii. The agent combined the reference `load_trials_and_mask` default `nan_exclude` list with the `max_trial_len=10.0` parameter from `prepare_data`. The agent added `goCue_times` to the nan_exclude list (not in the reference default), which is a slight deviation but defensible since `goCue_times` is needed for trial length computation.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `spikes.times.npy`, `spikes.clusters.npy`, `clusters.metrics.pqt`, `clusters.channels.npy`, and `channels.brainLocationIds_ccf_2017.npy` under each probe's `pykilosort` spike sorting directory.

ii.
```python
def load_probe_good_units(probe_alf_path: Path, brain_regions: BrainRegions):
    sorter_base = probe_alf_path / "pykilosort"
    metrics_path = latest_revision_file(sorter_base, "#*/clusters.metrics.pqt")
    spike_times_path = latest_revision_file(sorter_base, "#*/spikes.times.npy")
    spike_clusters_path = latest_revision_file(sorter_base, "#*/spikes.clusters.npy")
    cluster_channels_path = latest_revision_file(sorter_base, "#*/clusters.channels.npy")
    channel_regions_path = latest_revision_file(sorter_base, "#*/channels.brainLocationIds_ccf_2017.npy")
```

iii. The agent identified these as the standard IBL ALF spike sorting output files.

## 2-b. How is the `neural` data processed?

i. Spike times are binned into 20 ms bins over a 2 s window ([-0.5, 1.5] s relative to stimulus onset) per trial, producing spike count matrices of shape `(n_clusters, 100)`. Multiple probes are merged by concatenating clusters with offset indices, then sorting by spike time. Data is stored as `float16`.

ii.
```python
def bin_spikes_for_trial(spike_times, spike_clusters, n_clusters, trial_start, binsize=BIN_SIZE_S, n_bins=N_BINS):
    trial_end = trial_start + binsize * n_bins
    start_idx = np.searchsorted(spike_times, trial_start, side="left")
    end_idx = np.searchsorted(spike_times, trial_end, side="left")
    # ...
    bin_idx = np.floor((times - trial_start) / binsize).astype(np.int64)
    flat_idx = clusters[valid] * n_bins + bin_idx[valid]
    counts = np.bincount(flat_idx, minlength=n_clusters * n_bins).reshape(n_clusters, n_bins)
    return counts.astype(np.float16)
```

iii. The agent noted that the reference code uses `bincount2D` from iblutil and bins similarly with 20 ms bins.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only clusters with `label >= 1` in `clusters.metrics.pqt` are retained. Additionally, clusters assigned to `root` or `void` brain regions are excluded. Trials where no spikes from any retained neuron are present across the full 2 s window are dropped.

ii.
```python
metrics = pd.read_parquet(metrics_path, columns=["label"])
# ...
keep_mask = metrics["label"].to_numpy() >= 1.0
# ...
keep_mask &= ~np.isin(cluster_acronyms, ["void", "root"])
# ...
if not np.any(neural_trial):
    continue
```

iii. The agent justified `label >= 1` by citing the data paper's description of well-isolated neurons (108 per probe average). However, the reference code's `prepare_data` calls `load_spiking_data` with `qc=None` (default), loading ALL clusters, and stores `good_clusters` metadata separately for potential downstream filtering.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to stimulus onset (`stimOn_times`). The trial window starts at `stimOn_times - 0.5` s and ends at `stimOn_times + 1.5` s. Spikes within this window are binned.

ii.
```python
trial_start = float(trial_row["stimOn_times"] + OFF_START_S)  # OFF_START_S = -0.5
trial_end = float(trial_row["stimOn_times"] + OFF_END_S)      # OFF_END_S = 1.5
neural_trial = bin_spikes_for_trial(merged["spike_times"], merged["spike_clusters"], n_clusters, trial_start)
```

iii. The agent matched the reference `params['align_time'] = 'stimOn_times'` and `params['time_window'] = (-.5, 1.5)`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The bin size is 20 ms, producing 100 time bins over the 2 s window. No temporal rebinning is applied; spikes are directly binned at 20 ms resolution.

ii.
```python
BIN_SIZE_S = 0.02
OFF_START_S = -0.5
OFF_END_S = 1.5
N_BINS = int(round((OFF_END_S - OFF_START_S) / BIN_SIZE_S))  # = 100
```

iii. Matches the reference `params['binsize'] = 0.02` and `params['interval_len'] = 2`.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is derived from the `stimOn_times` column of the trials table and the bin structure constants (OFF_START_S, OFF_END_S, BIN_SIZE_S, N_BINS).

ii.
```python
time_since_stim = np.linspace(
    OFF_START_S + BIN_SIZE_S,
    OFF_END_S,
    N_BINS,
    dtype=np.float32,
)
```

iii. The agent computed time since stimulus onset as right-edge bin times from -0.48 s to 1.50 s.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. A linearly-spaced array of 100 values from `OFF_START_S + BIN_SIZE_S` (-0.48) to `OFF_END_S` (1.5) is created. These represent the right edges of each 20 ms bin, consistent with how the reference `get_behavior_per_interval` computes interpolation points.

ii.
```python
time_since_stim = np.linspace(OFF_START_S + BIN_SIZE_S, OFF_END_S, N_BINS, dtype=np.float32)
```

iii. This is the same formula used in `get_behavior_per_interval`: `x_interp = np.linspace(interval_begs[interval_idx] + binsize, interval_ends[interval_idx], n_bins)`.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The time values are identical for every trial (same linspace), and correspond to the right edges of the same bins used for spike binning. Since the neural bins and the time input share the same window and bin count, they are aligned by construction.

ii.
```python
input_trial = np.vstack([
    time_since_stim,
    np.full(N_BINS, trial_number_in_block[trial_idx], dtype=np.float32),
]).astype(np.float32)
```

iii. The agent ensured alignment by using the same N_BINS and time window constants for both neural and input data.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from the `probabilityLeft` column of the trials table.

ii.
```python
trial_number_in_block = compute_trial_number_in_block(trials_df["probabilityLeft"].to_numpy())
```

iii. Block boundaries are detected by changes in `probabilityLeft`.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The function iterates through all trials (before masking) and counts consecutive trials with the same `probabilityLeft` value. The count resets to 1 when the value changes, and increments otherwise. The result is 1-indexed.

ii.
```python
def compute_trial_number_in_block(probability_left):
    values = np.asarray(probability_left, dtype=np.float64)
    numbers = np.zeros(values.shape[0], dtype=np.float32)
    count = 0
    previous = np.nan
    for idx, value in enumerate(values):
        same_block = np.isfinite(value) and np.isfinite(previous) and np.isclose(value, previous)
        count = count + 1 if same_block else 1
        numbers[idx] = float(count)
        previous = value
    return numbers
```

iii. The agent computed trial number in block on the original (unmasked) trial order, then indexed into it for retained trials. This is consistent per-trial and repeated across time bins.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. Derived from the `choice` column of the trials table.

ii.
```python
choice_class = choice_to_class(float(trial_row["choice"]))
```

iii. The agent noted that IBL `choice` uses 1 for left and -1 for right.

## 5-b. What processing is involved in computing `output` *Choice*?

i. IBL choice value 1 (left) is mapped to class 0, and -1 (right) is mapped to class 1. The result is a per-trial scalar replicated across all 100 time bins.

ii.
```python
def choice_to_class(value):
    if np.isclose(value, 1.0):
        return 0  # left
    if np.isclose(value, -1.0):
        return 1  # right
    raise ValueError(f"Unexpected choice value: {value}")
```

iii. The mapping follows the instruction: "left = 0, right = 1".

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. Derived from the `probabilityLeft` column of the trials table.

ii.
```python
prior_class = probability_left_to_class(float(trial_row["probabilityLeft"]))
```

iii. The agent used the explicit values from the instruction.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The `probabilityLeft` value is mapped to a categorical: 0.2 -> 0, 0.5 -> 1, 0.8 -> 2. The result is a per-trial scalar replicated across all 100 time bins.

ii.
```python
def probability_left_to_class(value):
    mapping = {0.2: 0, 0.5: 1, 0.8: 2}
    for key, encoded in mapping.items():
        if np.isclose(value, key):
            return encoded
    raise ValueError(f"Unexpected probabilityLeft value: {value}")
```

iii. Matches the instruction: "0.2 -> 0, 0.5 -> 1, 0.8 -> 2".

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. Derived from `_ibl_wheel.timestamps.npy` and `_ibl_wheel.position.npy` in the ALF directory.

ii.
```python
def load_wheel_speed(alf_path: Path):
    timestamps_path = latest_revision_file(alf_path, ["_ibl_wheel.timestamps.npy", "#*/_ibl_wheel.timestamps.npy"])
    position_path = latest_revision_file(alf_path, ["_ibl_wheel.position.npy", "#*/_ibl_wheel.position.npy"])
    timestamps = np.load(timestamps_path).astype(np.float64)
    position = np.load(position_path).astype(np.float64)
```

iii. The agent used the same raw wheel data files that `SessionLoader.load_wheel()` loads internally.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Wheel position is interpolated to 1000 Hz using linear interpolation. A Butterworth low-pass filter (order 8, corner frequency 20 Hz) is applied, then velocity is computed as the first difference scaled by the sampling rate. Wheel speed is the absolute value of this velocity.

ii.
```python
position_interp, time_interp = interpolate_position(timestamps, position, freq=WHEEL_FS)
velocity, _ = velocity_filtered(position_interp, fs=WHEEL_FS)
speed = np.abs(velocity).astype(np.float32)
```

iii. This matches `SessionLoader.load_wheel()` which calls the same `interpolate_position` and `velocity_filtered` functions with the same default parameters, and the reference `load_target_behavior('wheel-speed')` which takes `np.abs(velocity)`.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. All retained wheel speed time bins across all sessions are concatenated, then global tertile thresholds (33rd and 67th percentiles) are computed. Each time bin is then discretized into 3 categories (0=low, 1=medium, 2=high) using `np.digitize`.

ii.
```python
wheel_flat = np.concatenate(wheel_continuous_all)
_, wheel_edges = discretize_three_bins(wheel_flat)
# ...
wheel_bins = np.digitize(wheel_trial, [wheel_edges[0], wheel_edges[1]], right=False).astype(np.int8)
```

iii. The instruction says "Wheel speed discretized into 3 bins, time-varying". The agent used global tertiles for thresholding.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. Per-trial wheel speed is interpolated onto the same 100-point time grid as the neural data (right edges of 20 ms bins from trial_start to trial_end), using `np.interp` for linear interpolation. Coverage checks ensure the behavioral data spans the trial window.

ii.
```python
def interpolate_behavior_trial(sample_times, sample_values, interval_start, interval_end, ...):
    x_interp = np.linspace(interval_start + binsize, interval_end, n_bins, dtype=np.float64)
    y_interp = np.interp(x_interp, trial_times, trial_values).astype(np.float32)
    return y_interp
```

iii. The agent used the same interpolation approach as `get_behavior_per_interval` in the reference code, with the same `x_interp = np.linspace(interval_beg + binsize, interval_end, n_bins)` formula.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. Derived from `leftCamera.ROIMotionEnergy.npy` (preferred) or `rightCamera.ROIMotionEnergy.npy` (fallback), along with the corresponding `_ibl_leftCamera.times.npy` or `_ibl_rightCamera.times.npy`.

ii.
```python
def load_motion_energy(alf_path: Path):
    for view in ("left", "right"):
        me_path = latest_revision_file(alf_path, f"#*/{view}Camera.ROIMotionEnergy.npy")
        times_path = latest_revision_file(alf_path,
            [f"_ibl_{view}Camera.times.npy", f"#*/_ibl_{view}Camera.times.npy"])
        motion_energy = np.load(me_path).astype(np.float32)
        timestamps = np.load(times_path).astype(np.float64)
```

iii. Matches reference `bin_behaviors` which tries left camera first, falls back to right.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. If timestamps are longer than motion energy data, the extra timestamps at the start are trimmed (matching `_check_video_timestamps`). The raw motion energy values are then used directly (no smoothing or normalization applied).

ii.
```python
if timestamps.shape[0] > motion_energy.shape[0]:
    timestamps = timestamps[-motion_energy.shape[0]:]
```

iii. This matches the reference `_check_video_timestamps` which trims the beginning of timestamps for pre-GPIO sessions.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Same approach as wheel speed: global tertiles across all retained whisker motion energy time bins are computed, then `np.digitize` assigns each time bin to one of 3 categories.

ii.
```python
whisker_flat = np.concatenate(whisker_continuous_all)
_, whisker_edges = discretize_three_bins(whisker_flat)
# ...
whisker_bins = np.digitize(whisker_trial, [whisker_edges[0], whisker_edges[1]], right=False).astype(np.int8)
```

iii. The instruction says "Whisker motion energy discretized into 3 bins, time-varying".

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Identical to wheel speed alignment: per-trial motion energy is linearly interpolated onto the same 100-point time grid using `interpolate_behavior_trial`, which uses `np.interp` with the same linspace formula.

ii.
```python
whisker_trial = interpolate_behavior_trial(whisker_times, whisker_me, trial_start, trial_end)
```

iii. Same approach as reference `get_behavior_per_interval`.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The agent handles missing data at multiple levels:
- Sessions with missing files (wheel, whisker, spike sorting) are dropped with error logging.
- Trials with NaN in required columns are excluded by the trial mask.
- Trials where behavioral interpolation fails (data too sparse, doesn't cover window) return `None` and are skipped.
- Trials with zero spikes in all neurons are skipped.
- Sessions with fewer than 2 valid trials are dropped.
- Camera timestamp length mismatches are handled by trimming.

ii.
```python
except Exception as exc:
    dropped_sessions.append((session["eid"], f"error:{type(exc).__name__}:{exc}"))
# ...
if wheel_trial is None or whisker_trial is None:
    continue
if not np.any(neural_trial):
    continue
if len(session_neural) < 2:
    dropped_sessions.append((session["eid"], "fewer_than_two_trials"))
    continue
```

iii. The agent adopted a robust approach, catching exceptions at the session level and skipping problematic trials individually.

## 10-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are:
1. Loading spike data from disk for each probe (numpy file I/O for potentially large spike arrays).
2. Binning spikes per trial (searchsorted + bincount for each of ~400 trials per session).
3. Interpolating wheel and whisker data per trial.
4. The overall loop over 459 sessions is sequential.

ii.
```python
for session_idx, session in enumerate(session_rows, start=1):
    # ... load all data, process all trials
    for trial_idx, trial_row in trials_df.iterrows():
        neural_trial = bin_spikes_for_trial(...)
        wheel_trial = interpolate_behavior_trial(...)
        whisker_trial = interpolate_behavior_trial(...)
```

iii. The agent did not use multiprocessing, unlike the reference code which uses `multiprocessing.Pool` for spike binning and behavior interpolation.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Key loops that could be vectorized:
1. The per-trial spike binning loop could be replaced with a vectorized approach using 2D bin counting across all trials simultaneously.
2. The `compute_trial_number_in_block` function uses a Python for-loop that could be vectorized with `np.diff` and `np.cumsum`.
3. The per-trial behavior interpolation could be batched.

ii.
```python
for trial_idx, trial_row in trials_df.iterrows():
    # ... individual trial processing
    neural_trial = bin_spikes_for_trial(...)
```

iii. The agent chose simplicity and correctness over performance optimization.

## 10-c. What processing does the code repeat multiple times?

i. The following processing is repeated:
1. Wheel speed and whisker ME are interpolated per trial individually (could be batched).
2. The `latest_revision_file` glob search is called for each file type, repeating filesystem traversals.
3. The discretization (`np.digitize`) of wheel and whisker data is done per trial in the final assembly loop, after already having the continuous values stored.

ii.
```python
# Continuous values stored during trial loop
wheel_continuous_all.append(wheel_trial)
whisker_continuous_all.append(whisker_trial)
# ... later discretized again in assembly loop
wheel_bins = np.digitize(wheel_trial, [wheel_edges[0], wheel_edges[1]], right=False)
```

iii. The two-pass approach (first collect continuous, then discretize) is necessary because the global tertile thresholds can only be computed after all data is collected.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Potentially unnecessary processing includes:
1. Computing wheel acceleration (returned by `velocity_filtered` but never used).
2. Loading and processing ALL sessions even when `--session-limit` could be smaller.
3. The `build_sample_subset` function re-maps subject and region indices, which is extra work only needed for the sample output.
4. The motion view tracking (`motion_views`) is only used for logging statistics.
5. The extensive metadata and statistics collection that goes into `conversion_stats_full.json`.

ii.
```python
def velocity_filtered(...):
    vel = np.insert(np.diff(signal.sosfiltfilt(sos, pos)), 0, 0.0) * fs
    acc = np.insert(np.diff(vel), 0, 0.0) * fs  # acceleration computed but not used
    return vel, acc
```

iii. The agent computed acceleration to match the reference `velocity_filtered` function signature, but only uses velocity (converted to speed).
