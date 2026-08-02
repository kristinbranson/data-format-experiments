# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads the session manifest from `bwm_release.csv`, iterates over unique session IDs (eids), and resolves local file paths using the pattern `data/one_cache/<lab>/Subjects/<subject>/<date>/<session_number>/alf/`. For each session, it loads the trials table (parquet), wheel data (numpy), whisker motion energy (numpy), and spike sorting data from each probe's `pykilosort` subdirectory. This mirrors the reference code's use of `bwm_release.csv` and `ONE` API to discover sessions.

ii.
```python
BWM_RELEASE_CSV = Path("/app/code/code_zhang2025/data/bwm_release.csv")
ONE_CACHE_DIR = Path("/app/data/one_cache")

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

iii. From CONVERSION_NOTES.md: "Sessions were discovered from `code/code_zhang2025/data/bwm_release.csv`. Session paths were resolved locally."

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified from the `subject` field in `bwm_release.csv`. An ordered unique list of all subject names from kept sessions is built and stored in `data['subjects']`. Each session is assigned a `subject_idx` pointing into this list.

ii.
```python
subjects = ordered_unique(session["subject"] for session in kept_sessions)
subject_lookup = {subject: idx for idx, subject in enumerate(subjects)}
```

iii. No explicit justification given; follows directly from the reference data format.

## 1-c. How are the data split into sessions?

i. Sessions are defined by unique `eid` values in `bwm_release.csv`. Each eid corresponds to one recording session. Multiple probes within the same session are merged (not treated as separate sessions), matching the reference code.

ii.
```python
for eid, group in release_df.groupby("eid", sort=False):
    ...
    session_rows.append({"eid": eid, ...})
```

iii. From CONVERSION_NOTES.md: "Probes from the same session were merged before decoding, matching the reference code and the paper description that probes within a session are not treated independently."

## 1-d. How are the data split into trials?

i. Trials are loaded from the `_ibl_trials.table.pqt` parquet file for each session. Each row of the trials DataFrame represents one trial. The AI iterates through trials, applying a mask to select valid ones, and creates per-trial arrays.

ii.
```python
trials_df = load_trials_table(alf_path)
trial_mask = compute_trial_mask(trials_df)
...
for trial_idx, trial_row in trials_df.iterrows():
    if not trial_mask[trial_idx]:
        continue
```

iii. From CONVERSION_NOTES.md: Trial filtering rules are documented under "Trial filtering."

## 1-e. How are trials filtered based on quality controls?

i. Trials are excluded if:
- Any of these fields are non-finite: `stimOn_times`, `choice`, `feedback_times`, `probabilityLeft`, `firstMovement_times`, `feedbackType`, `goCue_times`
- `choice == 0` (no response)
- Reaction time (firstMovement_times - stimOn_times) outside [0.08, 2.0] seconds
- Trial length (feedback_times - goCue_times) > 10.0 seconds
- Wheel or whisker motion energy data is unavailable for the trial interval
- Neural activity is entirely zero across all neurons in the trial window

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

Additional per-trial checks:
```python
if wheel_trial is None or whisker_trial is None:
    continue
if not np.any(neural_trial):
    continue
```

iii. From CONVERSION_NOTES.md: "This follows the masking logic in `load_trials_and_mask(...)` plus the `prepare_data(...)` call used by the reference code."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `spikes.times.npy` (spike timestamps) and `spikes.clusters.npy` (cluster assignments) in each probe's `pykilosort` directory. Cluster quality is determined from `clusters.metrics.pqt`, and region assignment from `clusters.channels.npy` and `channels.brainLocationIds_ccf_2017.npy`.

ii.
```python
spike_times_path = latest_revision_file(sorter_base, "#*/spikes.times.npy")
spike_clusters_path = latest_revision_file(sorter_base, "#*/spikes.clusters.npy")
metrics_path = latest_revision_file(sorter_base, "#*/clusters.metrics.pqt")
cluster_channels_path = latest_revision_file(sorter_base, "#*/clusters.channels.npy")
channel_regions_path = latest_revision_file(sorter_base, "#*/channels.brainLocationIds_ccf_2017.npy")
```

iii. From CONVERSION_NOTES.md: "Spikes were loaded from `alf/<probe>/pykilosort/#...#/spikes.times.npy` and `spikes.clusters.npy`."

## 2-b. How is the `neural` data processed?

i. Spike times are binned into 20ms non-overlapping bins within the trial window [-0.5s, +1.5s] relative to stimulus onset. Multiple probes within a session are merged. The result is a (n_neurons, 100) spike count matrix per trial, stored as float16.

ii.
```python
def bin_spikes_for_trial(spike_times, spike_clusters, n_clusters, trial_start, binsize=BIN_SIZE_S, n_bins=N_BINS):
    trial_end = trial_start + binsize * n_bins
    start_idx = np.searchsorted(spike_times, trial_start, side="left")
    end_idx = np.searchsorted(spike_times, trial_end, side="left")
    ...
    bin_idx = np.floor((times - trial_start) / binsize).astype(np.int64)
    ...
    flat_idx = clusters[valid] * n_bins + bin_idx[valid]
    counts = np.bincount(flat_idx, minlength=n_clusters * n_bins).reshape(n_clusters, n_bins)
    return counts.astype(np.float16)
```

iii. From CONVERSION_NOTES.md: "Bin size: 20 ms. Number of bins per trial: 100."

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI applies two filters:
1. Only clusters with `label >= 1` in `clusters.metrics.pqt` are retained (well-isolated neurons).
2. Clusters assigned to "root" or "void" brain regions are excluded.

ii.
```python
metrics = pd.read_parquet(metrics_path, columns=["label"])
keep_mask = metrics["label"].to_numpy() >= 1.0
...
keep_mask &= ~np.isin(cluster_acronyms, ["void", "root"])
```

iii. From CONVERSION_NOTES.md: "Only well-isolated units with `label >= 1` were retained. Units assigned to `root` or `void` were excluded. ... The data paper explicitly states that final analyses retain well-isolated neurons."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to stimulus onset (`stimOn_times`). The trial window starts at stimOn_times - 0.5s and ends at stimOn_times + 1.5s.

ii.
```python
trial_start = float(trial_row["stimOn_times"] + OFF_START_S)  # OFF_START_S = -0.5
trial_end = float(trial_row["stimOn_times"] + OFF_END_S)      # OFF_END_S = 1.5
```

iii. From CONVERSION_NOTES.md: "Alignment event: `stimOn_times`. Window: `[-0.5 s, +1.5 s]`. This matches the defaults used in `0_data_caching.py`."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is 20ms (0.02s) producing 100 time bins per trial. No temporal rebinning is applied; the 20ms bins are used directly for all variables.

ii.
```python
BIN_SIZE_S = 0.02
N_BINS = int(round((OFF_END_S - OFF_START_S) / BIN_SIZE_S))  # = 100
```

iii. From CONVERSION_NOTES.md: "Bin size: 20 ms. Number of bins per trial: 100." This matches the reference code's `params['binsize'] = 0.02`.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. This is not derived from raw data variables. It is a synthetic time axis computed from the alignment parameters (OFF_START_S, OFF_END_S, BIN_SIZE_S).

ii.
```python
time_since_stim = np.linspace(
    OFF_START_S + BIN_SIZE_S,  # -0.48
    OFF_END_S,                  # 1.5
    N_BINS,                     # 100
    dtype=np.float32,
)
```

iii. From CONVERSION_NOTES.md: "`time_since_stimulus_onset_s` — time-varying, one value per 20 ms bin, represented at the right edge of each bin, from -0.48 s to 1.50 s."

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. A linear ramp is created using `np.linspace` from -0.48s to 1.5s with 100 points. The values represent the right edge of each 20ms time bin relative to stimulus onset. This is the same convention used in the reference code's `get_behavior_per_interval`.

ii.
```python
time_since_stim = np.linspace(
    OFF_START_S + BIN_SIZE_S, OFF_END_S, N_BINS, dtype=np.float32
)
```

iii. The right-edge convention matches the reference code: `x_interp = np.linspace(interval_begs[interval_idx] + binsize, interval_ends[interval_idx], n_bins)`.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The time since stimulus onset array has the same number of time bins (100) as the neural data, with each value corresponding to the right edge of the matching neural time bin. They share the same temporal grid.

ii.
```python
input_trial = np.vstack([
    time_since_stim,
    np.full(N_BINS, trial_number_in_block[trial_idx], dtype=np.float32),
]).astype(np.float32)
```

iii. No explicit justification needed; it is constructed to match the neural bins by design.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. Derived from the `probabilityLeft` column of the trials table. Consecutive trials with the same `probabilityLeft` value are considered part of the same block.

ii.
```python
trial_number_in_block = compute_trial_number_in_block(trials_df["probabilityLeft"].to_numpy())
```

iii. From CONVERSION_NOTES.md: "1-based count of the trial index within the current `probabilityLeft` block."

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The function iterates through all trials (before masking), tracking the current block. When `probabilityLeft` changes, the counter resets to 1; otherwise it increments. The resulting value for each retained trial is repeated across all 100 time bins (since this is a per-trial variable).

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

iii. From CONVERSION_NOTES.md: "computed on the original session trial order before masking."

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. Derived from the `choice` column of the trials table (`_ibl_trials.table.pqt`).

ii.
```python
choice_class = choice_to_class(float(trial_row["choice"]))
```

iii. No additional justification needed.

## 5-b. What processing is involved in computing `output` *Choice*?

i. IBL choice values are mapped to binary classes: choice=1 (left) maps to class 0, choice=-1 (right) maps to class 1. The value is repeated across all 100 time bins.

ii.
```python
def choice_to_class(value):
    if np.isclose(value, 1.0):
        return 0  # left
    if np.isclose(value, -1.0):
        return 1  # right
    raise ValueError(f"Unexpected choice value: {value}")
```

```python
output_trial = np.vstack([
    np.full(N_BINS, choice_class, dtype=np.int8),
    ...
])
```

iii. From CONVERSION_NOTES.md: "mapped from IBL `choice`: 1 -> left -> 0, -1 -> right -> 1."

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. Derived from the `probabilityLeft` column of the trials table.

ii.
```python
prior_class = probability_left_to_class(float(trial_row["probabilityLeft"]))
```

iii. No additional justification needed.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The raw `probabilityLeft` values (0.2, 0.5, 0.8) are mapped to ordinal classes (0, 1, 2). The value is repeated across all 100 time bins.

ii.
```python
def probability_left_to_class(value):
    mapping = {0.2: 0, 0.5: 1, 0.8: 2}
    for key, encoded in mapping.items():
        if np.isclose(value, key):
            return encoded
    raise ValueError(f"Unexpected probabilityLeft value: {value}")
```

iii. From CONVERSION_NOTES.md: "prior_probability_left: 0.2 -> 0, 0.5 -> 1, 0.8 -> 2."

## 9-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. Derived from `_ibl_wheel.timestamps.npy` (wheel encoder timestamps) and `_ibl_wheel.position.npy` (wheel encoder positions).

ii.
```python
timestamps_path = latest_revision_file(alf_path, ["_ibl_wheel.timestamps.npy", "#*/_ibl_wheel.timestamps.npy"])
position_path = latest_revision_file(alf_path, ["_ibl_wheel.position.npy", "#*/_ibl_wheel.position.npy"])
timestamps = np.load(timestamps_path).astype(np.float64)
position = np.load(position_path).astype(np.float64)
```

iii. From CONVERSION_NOTES.md: "Wheel timestamps and position were loaded from `_ibl_wheel.timestamps.npy` and `_ibl_wheel.position.npy`."

## 9-b. What processing is involved in computing `output` *Wheel speed*?

i. The processing chain is:
1. Wheel position is linearly interpolated to 1000 Hz uniform sampling
2. Velocity is computed using a Butterworth low-pass filter (order 8, 20 Hz corner frequency) applied via `sosfiltfilt`, then differentiated
3. Speed is the absolute value of velocity

ii.
```python
def interpolate_position(re_ts, re_pos, freq=WHEEL_FS):
    t = np.arange(re_ts[0], re_ts[-1], 1.0 / freq, dtype=np.float64)
    ...
    yinterp = np.interp(t, re_ts, re_pos)
    return yinterp, t

def velocity_filtered(pos, fs=WHEEL_FS, corner_frequency=WHEEL_FILTER_CORNER_HZ, order=WHEEL_FILTER_ORDER):
    sos = signal.butter(N=order, Wn=corner_frequency / fs * 2.0, btype="lowpass", output="sos")
    vel = np.insert(np.diff(signal.sosfiltfilt(sos, pos)), 0, 0.0) * fs
    ...
    return vel, acc

speed = np.abs(velocity).astype(np.float32)
```

iii. From CONVERSION_NOTES.md: "Velocity was computed with the same Butterworth low-pass filtering used by `SessionLoader.load_wheel(...)`: cutoff 20 Hz, order 8."

## 9-c. How is `output` *Wheel speed* thresholded into categories?

i. Wheel speed values across all retained trials and time bins are pooled, and global tertile boundaries (33rd and 67th percentiles) are computed. Each time bin's speed value is then assigned to one of three bins (0=low, 1=medium, 2=high) using `np.digitize`.

ii.
```python
def discretize_three_bins(values):
    values = np.asarray(values, dtype=np.float64)
    q1, q2 = np.quantile(values, [1.0 / 3.0, 2.0 / 3.0])
    ...
    bins = np.digitize(values, [q1, q2], right=False).astype(np.int8)
    return bins, (float(q1), float(q2))

wheel_flat = np.concatenate(wheel_continuous_all)
_, wheel_edges = discretize_three_bins(wheel_flat)
...
wheel_bins = np.digitize(wheel_trial, [wheel_edges[0], wheel_edges[1]], right=False).astype(np.int8)
```

iii. From CONVERSION_NOTES.md: "wheel_speed_bin — global tertiles across all retained wheel-speed time bins."

## 9-d. How is `output` *Wheel speed* aligned with the neural data?

i. Session-wide wheel speed (at 1kHz) is linearly interpolated onto the 20ms neural time bins within each trial's window. The interpolation targets are at the right edges of the bins (same as neural data), using `np.linspace(interval_start + binsize, interval_end, n_bins)`.

ii.
```python
def interpolate_behavior_trial(sample_times, sample_values, interval_start, interval_end, binsize=BIN_SIZE_S, n_bins=N_BINS):
    ...
    x_interp = np.linspace(interval_start + binsize, interval_end, n_bins, dtype=np.float64)
    y_interp = np.interp(x_interp, trial_times, trial_values).astype(np.float32)
    ...
    return y_interp
```

iii. From CONVERSION_NOTES.md: "Trial values were resampled onto the 20 ms decoder bins using linear interpolation, with the same interval coverage checks as `get_behavior_per_interval(...)`."

## 10-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. Derived from `leftCamera.ROIMotionEnergy.npy` (or `rightCamera.ROIMotionEnergy.npy` as fallback) and corresponding camera timestamps (`_ibl_leftCamera.times.npy` or `_ibl_rightCamera.times.npy`).

ii.
```python
def load_motion_energy(alf_path: Path):
    for view in ("left", "right"):
        me_path = latest_revision_file(alf_path, f"#*/{view}Camera.ROIMotionEnergy.npy")
        ...
        times_path = latest_revision_file(alf_path, [f"_ibl_{view}Camera.times.npy", f"#*/_ibl_{view}Camera.times.npy"])
        ...
        motion_energy = np.load(me_path).astype(np.float32)
        timestamps = np.load(times_path).astype(np.float64)
```

iii. From CONVERSION_NOTES.md: "The left camera whisker motion energy was used when available; otherwise the right camera was used."

## 10-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The raw motion energy values are loaded directly (no filtering or smoothing). If timestamps are longer than the motion energy array, the excess timestamps at the beginning are trimmed (matching `_check_video_timestamps`). The values are then linearly interpolated onto the 20ms time bins per trial.

ii.
```python
if timestamps.shape[0] > motion_energy.shape[0]:
    timestamps = timestamps[-motion_energy.shape[0]:]
```

iii. From CONVERSION_NOTES.md: "If timestamps were longer than the motion-energy trace, the extra timestamps at the start were trimmed, matching `_check_video_timestamps(...)`."

## 10-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Same approach as wheel speed: global tertiles across all retained whisker motion energy time bins, then `np.digitize` to assign 0/1/2 bins.

ii.
```python
whisker_flat = np.concatenate(whisker_continuous_all)
_, whisker_edges = discretize_three_bins(whisker_flat)
...
whisker_bins = np.digitize(whisker_trial, [whisker_edges[0], whisker_edges[1]], right=False).astype(np.int8)
```

iii. From CONVERSION_NOTES.md: "whisker_motion_energy_bin — global tertiles across all retained whisker-motion-energy time bins."

## 10-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Same as wheel speed: linearly interpolated onto the 20ms neural time bins within each trial's window, using right-edge bin positions.

ii.
```python
whisker_trial = interpolate_behavior_trial(whisker_times, whisker_me, trial_start, trial_end)
```

iii. Same `interpolate_behavior_trial` function used for both wheel and whisker.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Several strategies are used:
- Trials with missing/non-finite required fields are excluded via the trial mask
- Trials where wheel or whisker data cannot be interpolated (missing coverage) are dropped
- Trials with zero neural activity across all neurons are dropped
- Sessions with no good units, or fewer than 2 retained trials, are dropped entirely
- Sessions where required files are missing raise exceptions caught by a try/except, causing the session to be dropped
- Camera timestamp length mismatches are handled by trimming (timestamps > motion energy) or raising errors (timestamps < motion energy)

ii.
```python
except Exception as exc:
    dropped_sessions.append((session["eid"], f"error:{type(exc).__name__}:{exc}"))
```

```python
if wheel_trial is None or whisker_trial is None:
    continue
if not np.any(neural_trial):
    continue
if len(session_neural) < 2:
    dropped_sessions.append((session["eid"], "fewer_than_two_trials"))
    continue
```

iii. From CONVERSION_NOTES.md: Drop reasons are documented. Top reasons: "missing left/right whisker motion energy: 14, missing wheel timestamps or wheel position: 5, fewer than two retained trials: 1."

## 12-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are:
1. **Spike binning per trial** (`bin_spikes_for_trial`): Called once per trial per session, involving `searchsorted`, array slicing, and `bincount` on large spike arrays.
2. **Wheel and whisker interpolation per trial** (`interpolate_behavior_trial`): Called twice per trial (once for wheel, once for whisker).
3. **Loading and processing wheel data** (`load_wheel_speed`): Position interpolation to 1kHz and Butterworth filtering on the full session-length signal.

ii.
```python
for trial_idx, trial_row in trials_df.iterrows():
    ...
    wheel_trial = interpolate_behavior_trial(wheel_times, wheel_speed, trial_start, trial_end)
    whisker_trial = interpolate_behavior_trial(whisker_times, whisker_me, trial_start, trial_end)
    neural_trial = bin_spikes_for_trial(...)
```

iii. No explicit discussion of performance in CONVERSION_NOTES.md.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops could be vectorized:
1. **Per-trial spike binning loop**: The inner trial loop calls `bin_spikes_for_trial` sequentially. This could be vectorized by computing bin indices for all trials at once using vectorized `searchsorted` and fancy indexing.
2. **`compute_trial_number_in_block`**: Uses a sequential Python `for` loop. Could be vectorized using `np.diff` on `probabilityLeft` to detect block boundaries, then `np.cumsum` on segment lengths.
3. **Per-trial behavior interpolation**: Each trial calls `np.interp` individually; could be batched.

ii.
```python
def compute_trial_number_in_block(probability_left):
    ...
    for idx, value in enumerate(values):
        same_block = np.isfinite(value) and np.isfinite(previous) and np.isclose(value, previous)
        count = count + 1 if same_block else 1
        ...
```

iii. No explicit discussion of vectorization in CONVERSION_NOTES.md.

## 12-c. What processing does the code repeat multiple times?

i.
1. **`discretize_three_bins` computation vs per-trial digitization**: The global tertile edges are computed once, but `np.digitize` is called per trial in a separate loop over sessions/trials, rather than being batched.
2. **`latest_revision_file` glob matching**: Called multiple times per probe/session with overlapping directory searches.
3. **`np.searchsorted` for spike times**: Called per trial in `bin_spikes_for_trial` on the full session spike array, though the search ranges overlap between consecutive trials.

ii.
```python
# First pass: collect continuous values
wheel_continuous_all.append(wheel_trial)
whisker_continuous_all.append(whisker_trial)

# Second pass: discretize per trial
for static_values, wheel_trial, whisker_trial in zip(...):
    wheel_bins = np.digitize(wheel_trial, [wheel_edges[0], wheel_edges[1]], right=False)
    whisker_bins = np.digitize(whisker_trial, [whisker_edges[0], whisker_edges[1]], right=False)
```

iii. No discussion in CONVERSION_NOTES.md.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i.
1. **Acceleration computation**: `velocity_filtered` computes both velocity and acceleration, but only velocity (via speed = abs(velocity)) is used. The acceleration array is discarded.
2. **float16 neural storage**: Spike counts are stored as float16 to save space, but downstream decoder code will likely convert to float32, so the precision reduction serves only as a storage optimization.
3. **Sorting merged spike times**: After merging probes, spike times are sorted (`np.argsort`), which is necessary for `searchsorted` but involves sorting all spikes even though only per-trial slices are used.

ii.
```python
def velocity_filtered(pos, ...):
    ...
    vel = np.insert(np.diff(signal.sosfiltfilt(sos, pos)), 0, 0.0) * fs
    acc = np.insert(np.diff(vel), 0, 0.0) * fs  # acc is computed but never used
    return vel, acc

# Only velocity is used:
velocity, _ = velocity_filtered(position_interp, fs=WHEEL_FS)
speed = np.abs(velocity).astype(np.float32)
```

iii. No discussion in CONVERSION_NOTES.md.
