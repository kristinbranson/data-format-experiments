# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads sessions from `bwm_release.csv` (a CSV file listing all sessions in the brain-wide map release) rather than using the ONE API. It constructs file paths directly as `data/one_cache/<lab>/Subjects/<subject>/<date>/<session_number>/alf/` and opens raw `.npy`, `.pqt` files with numpy/pandas. Probes, trials, wheel, and motion energy are each loaded by dedicated functions that glob for the latest revision file.

ii.
```python
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

```python
session_path = (
    ONE_CACHE_DIR / session["lab"] / "Subjects" / session["subject"]
    / session["date"] / f"{session['session_number']:03d}"
)
alf_path = session_path / "alf"
```

iii. The AI chose to bypass the ONE API and read files directly from the local cache, using `bwm_release.csv` as the session index. The CONVERSION_NOTES state: "Sessions were discovered from `code/code_zhang2025/data/bwm_release.csv`."

## 1-b. How are the data split into subjects?

i. The subject name comes from the `subject` column of `bwm_release.csv`. Subjects are collected in order of appearance using `ordered_unique`, and `subject_idx` maps each session to its position in that list.

ii.
```python
subjects = ordered_unique(session["subject"] for session in kept_sessions)
subject_lookup = {subject: idx for idx, subject in enumerate(subjects)}
```

iii. The subject identity is directly available in the CSV. No parsing or derivation is needed.

## 1-c. How are the data split into sessions?

i. Each unique `eid` in `bwm_release.csv` defines one session. The CSV is grouped by `eid`, and each group yields one session entry with its associated probe names.

ii.
```python
for eid, group in release_df.groupby("eid", sort=False):
    ...
    session_rows.append({...})
```

iii. Sessions are the natural unit of the release CSV, so no splitting logic is needed.

## 1-d. How are the data split into trials?

i. The trials table (`_ibl_trials.table.pqt`) has one row per trial, so each row is one trial.

ii.
```python
trials_df = load_trials_table(alf_path)
...
for trial_idx, trial_row in trials_df.iterrows():
```

iii. The parquet trials table is already organized as one row per trial.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies several filters: (1) all of 7 columns (`stimOn_times`, `choice`, `feedback_times`, `probabilityLeft`, `firstMovement_times`, `feedbackType`, `goCue_times`) must be finite; (2) `choice != 0`; (3) reaction time between 0.08 s and 2.0 s; (4) trial length (`feedback_times - goCue_times`) must be <= 10 s; (5) wheel and whisker motion energy must have sufficient coverage for interpolation. Trials where the behavioral interpolation returns None (insufficient coverage) are also dropped. Additionally, trials with zero spikes across all neurons are dropped.

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

```python
if wheel_trial is None or whisker_trial is None:
    continue
...
if not np.any(neural_trial):
    continue
```

iii. The CONVERSION_NOTES state this follows "the masking logic in `load_trials_and_mask(...)` plus the `prepare_data(...)` call used by the reference code."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Two arrays per probe: `spikes.times.npy` (spike timestamps) and `spikes.clusters.npy` (cluster assignment per spike), loaded from the pykilosort subdirectory. Cluster quality comes from `clusters.metrics.pqt`, and anatomical location from `clusters.channels.npy` and `channels.brainLocationIds_ccf_2017.npy`.

ii.
```python
spike_times = np.load(spike_times_path).astype(np.float64)
spike_clusters = np.load(spike_clusters_path).astype(np.int64)
cluster_channels = np.load(cluster_channels_path).astype(np.int64)
channel_region_ids = np.load(channel_regions_path).astype(np.int64)
```

iii. The same spike sorting files are used as in the reference, just loaded directly rather than through the SpikeSortingLoader API.

## 2-b. How is the `neural` data processed?

i. Spikes are counted into 20 ms bins over the 2 s trial window (100 bins). The counts are stored as float16. **The counts are NOT divided by the bin width**, so the neural data represents raw spike counts per bin, not firing rates in Hz. When a session has multiple probes, their units are merged (concatenated with offset cluster indices) and sorted by spike time. Region labels are derived from `channels.brainLocationIds_ccf_2017.npy` via `BrainRegions.id2acronym()` without applying the Beryl mapping.

ii.
```python
def bin_spikes_for_trial(spike_times, spike_clusters, n_clusters, trial_start, ...):
    ...
    flat_idx = clusters[valid] * n_bins + bin_idx[valid]
    counts = np.bincount(flat_idx, minlength=n_clusters * n_bins).reshape(n_clusters, n_bins)
    return counts.astype(np.float16)
```

```python
cluster_acronyms = region_ids_to_acronyms(cluster_region_ids, brain_regions)
# uses brain_regions.id2acronym(region_ids) -- no Beryl mapping
```

iii. The CONVERSION_NOTES describe the neural data as "dense float16 spike-count matrices." No mention of converting to firing rates.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Clusters with `label >= 1` in `clusters.metrics.pqt` are kept. Additionally, clusters assigned to "root" or "void" brain regions are excluded. Trials with zero spikes across all retained neurons are dropped.

ii.
```python
keep_mask = metrics["label"].to_numpy() >= 1.0
...
keep_mask &= ~np.isin(cluster_acronyms, ["void", "root"])
```

```python
if not np.any(neural_trial):
    continue
```

iii. The CONVERSION_NOTES state: "Only well-isolated units with `label >= 1` were retained. Units assigned to `root` or `void` were excluded."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial's bin window starts at `stimOn_times + OFF_START_S` (-0.5 s) and the spikes are binned relative to that start time. This effectively aligns each trial to stimulus onset.

ii.
```python
trial_start = float(trial_row["stimOn_times"] + OFF_START_S)
trial_end = float(trial_row["stimOn_times"] + OFF_END_S)
neural_trial = bin_spikes_for_trial(merged["spike_times"], merged["spike_clusters"],
                                     n_clusters, trial_start)
```

```python
def bin_spikes_for_trial(..., trial_start, ...):
    trial_end = trial_start + binsize * n_bins
    ...
    bin_idx = np.floor((times - trial_start) / binsize).astype(np.int64)
```

iii. Alignment is achieved by defining the trial window relative to `stimOn_times`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The bin size is 20 ms, producing 100 bins over the 2 s window. No rebinning or interpolation is applied to the neural data.

ii.
```python
BIN_SIZE_S = 0.02
OFF_START_S = -0.5
OFF_END_S = 1.5
N_BINS = int(round((OFF_END_S - OFF_START_S) / BIN_SIZE_S))
```

iii. Matches the reference code's parameters.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. From `stimOn_times` in the trials table, which defines the alignment event. The time input is constructed as a linearly spaced array from the trial window parameters.

ii.
```python
time_since_stim = np.linspace(
    OFF_START_S + BIN_SIZE_S,
    OFF_END_S,
    N_BINS,
    dtype=np.float32,
)
```

iii. The time values are derived from the window parameters, not from any raw data variable.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The AI computes time values as `np.linspace(-0.48, 1.5, 100)`, which produces the **right edges** of each bin rather than the bin centers. This gives values [-0.48, -0.46, ..., 1.48, 1.50].

ii.
```python
time_since_stim = np.linspace(
    OFF_START_S + BIN_SIZE_S,
    OFF_END_S,
    N_BINS,
    dtype=np.float32,
)
```

iii. The CONVERSION_NOTES describe this as "represented at the right edge of each bin, from `-0.48 s` to `1.50 s`."

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The neural data bins spikes starting from `trial_start = stimOn_times - 0.5`, using left-edge-based binning (`np.floor((times - trial_start) / binsize)`). The time input uses right edges. This means the time values are shifted by half a bin (0.01 s) relative to the neural bin centers.

ii.
```python
# Neural binning uses left edges:
bin_idx = np.floor((times - trial_start) / binsize).astype(np.int64)

# Time input uses right edges:
time_since_stim = np.linspace(OFF_START_S + BIN_SIZE_S, OFF_END_S, N_BINS, ...)
```

iii. The AI explicitly chose right edges as documented in CONVERSION_NOTES.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. From `probabilityLeft` in the trials table. A change in `probabilityLeft` marks the start of a new block.

ii.
```python
trial_number_in_block = compute_trial_number_in_block(trials_df["probabilityLeft"].to_numpy())
```

iii. The block structure is inferred from changes in `probabilityLeft`, as there is no explicit block identifier in the trials table.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The trial number is computed as a **1-based** count within each block. A new block starts whenever `probabilityLeft` changes value (or is NaN). The count is computed on the full trial order before any filtering.

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

iii. The CONVERSION_NOTES state: "1-based count of the trial index within the current `probabilityLeft` block, computed on the original session trial order before masking."

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. The `choice` column of the trials table, which is +1 (left), -1 (right), or 0 (no response).

ii.
```python
choice_class = choice_to_class(float(trial_row["choice"]))
```

iii. Direct mapping from the trials table column.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The value is mapped: +1 (left) -> 0, -1 (right) -> 1. No-response trials (choice = 0) are excluded by the trial mask.

ii.
```python
def choice_to_class(value):
    if np.isclose(value, 1.0):
        return 0
    if np.isclose(value, -1.0):
        return 1
    raise ValueError(f"Unexpected choice value: {value}")
```

iii. Standard recoding matching the instructions.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. The `probabilityLeft` column of the trials table, which takes values 0.2, 0.5, or 0.8.

ii.
```python
prior_class = probability_left_to_class(float(trial_row["probabilityLeft"]))
```

iii. Direct mapping from the trials table column.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The value is mapped: 0.2 -> 0, 0.5 -> 1, 0.8 -> 2, matching the instructions.

ii.
```python
def probability_left_to_class(value):
    mapping = {0.2: 0, 0.5: 1, 0.8: 2}
    for key, encoded in mapping.items():
        if np.isclose(value, key):
            return encoded
    raise ValueError(f"Unexpected probabilityLeft value: {value}")
```

iii. Follows the mapping specified in the instructions.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. `_ibl_wheel.timestamps.npy` and `_ibl_wheel.position.npy`, loaded directly from the ALF directory.

ii.
```python
timestamps = np.load(timestamps_path).astype(np.float64)
position = np.load(position_path).astype(np.float64)
```

iii. Same source data as the reference.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The wheel position is interpolated to 1000 Hz, then velocity is computed using a Butterworth low-pass filter (order 8, corner frequency 20 Hz) via `scipy.signal.sosfiltfilt`. Speed is the absolute value of velocity. The speed trace is then interpolated onto the trial time grid using `np.interp`, and finally discretized into 3 bins using global tertiles.

ii.
```python
position_interp, time_interp = interpolate_position(timestamps, position, freq=WHEEL_FS)
velocity, _ = velocity_filtered(position_interp, fs=WHEEL_FS)
speed = np.abs(velocity).astype(np.float32)
```

```python
def velocity_filtered(pos, fs=WHEEL_FS, corner_frequency=WHEEL_FILTER_CORNER_HZ, order=WHEEL_FILTER_ORDER):
    sos = signal.butter(N=order, Wn=corner_frequency / fs * 2.0, btype="lowpass", output="sos")
    vel = np.insert(np.diff(signal.sosfiltfilt(sos, pos)), 0, 0.0) * fs
    ...
```

iii. The AI manually reimplemented the SessionLoader wheel processing. The CONVERSION_NOTES confirm the same filter parameters.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. The AI computes **global tertiles** across all retained wheel speed time bins from all sessions, then applies those thresholds to each trial. This differs from the reference which uses per-session percentiles.

ii.
```python
wheel_flat = np.concatenate(wheel_continuous_all)
_, wheel_edges = discretize_three_bins(wheel_flat)
...
wheel_bins = np.digitize(wheel_trial, [wheel_edges[0], wheel_edges[1]], right=False).astype(np.int8)
```

```python
def discretize_three_bins(values):
    values = np.asarray(values, dtype=np.float64)
    q1, q2 = np.quantile(values, [1.0 / 3.0, 2.0 / 3.0])
    ...
```

iii. The CONVERSION_NOTES state: "global tertiles across all retained wheel-speed time bins."

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The wheel speed is interpolated onto the trial time grid using `np.linspace(interval_start + binsize, interval_end, n_bins)`, which produces right edges. This is shifted by half a bin from the neural bin centers.

ii.
```python
x_interp = np.linspace(interval_start + binsize, interval_end, n_bins, dtype=np.float64)
y_interp = np.interp(x_interp, trial_times, trial_values).astype(np.float32)
```

iii. The AI uses the same right-edge grid for behavioral interpolation as for the time input.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. `<side>Camera.ROIMotionEnergy.npy` and `_ibl_<side>Camera.times.npy`, with left camera preferred over right.

ii.
```python
for view in ("left", "right"):
    me_path = latest_revision_file(alf_path, f"#*/{view}Camera.ROIMotionEnergy.npy")
    ...
    motion_energy = np.load(me_path).astype(np.float32)
    timestamps = np.load(times_path).astype(np.float64)
```

iii. Same source data and camera preference as the reference.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The motion energy trace is used as-is (no filtering or normalization). It is interpolated onto the trial time grid at right edges, then discretized into 3 bins using global tertiles. When timestamps are longer than the motion energy array, extra timestamps at the start are trimmed.

ii.
```python
if timestamps.shape[0] > motion_energy.shape[0]:
    timestamps = timestamps[-motion_energy.shape[0]:]
```

```python
whisker_trial = interpolate_behavior_trial(whisker_times, whisker_me, trial_start, trial_end)
```

iii. The CONVERSION_NOTES state: "If timestamps were longer than the motion-energy trace, the extra timestamps at the start were trimmed."

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Same as wheel speed: **global tertiles** across all retained whisker motion energy time bins from all sessions.

ii.
```python
whisker_flat = np.concatenate(whisker_continuous_all)
_, whisker_edges = discretize_three_bins(whisker_flat)
...
whisker_bins = np.digitize(whisker_trial, [whisker_edges[0], whisker_edges[1]], right=False).astype(np.int8)
```

iii. The CONVERSION_NOTES state: "global tertiles across all retained whisker-motion-energy time bins."

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Same as wheel speed: interpolated onto right edges rather than bin centers, shifted by half a bin from the neural data.

ii.
```python
x_interp = np.linspace(interval_start + binsize, interval_end, n_bins, dtype=np.float64)
y_interp = np.interp(x_interp, trial_times, trial_values).astype(np.float32)
```

iii. Uses the same time grid as all other behavioral variables.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Several strategies: (1) Sessions with missing required files raise exceptions and are caught and recorded as dropped. (2) Probes with no good units return None and are skipped. (3) Trials where wheel or whisker interpolation fails (returns None) are dropped. (4) Trials with zero neural activity are dropped. (5) Sessions with fewer than 2 retained trials are dropped. (6) Camera timestamp/motion energy length mismatches are handled by trimming extra timestamps.

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
```

```python
if len(session_neural) < 2:
    dropped_sessions.append((session["eid"], "fewer_than_two_trials"))
    continue
```

iii. The approach is conservative: drop anything that is missing or problematic rather than imputing.

## 10-a. What are the most time-consuming steps of the code?

i. Loading spike sorting files from disk (large `.npy` arrays of spike times and cluster assignments), and the per-trial loop that bins spikes and interpolates behavioral traces. The code processes sessions sequentially (no parallelism).

ii.
```python
spike_times = np.load(spike_times_path).astype(np.float64)
spike_clusters = np.load(spike_clusters_path).astype(np.int64)
```

```python
for trial_idx, trial_row in trials_df.iterrows():
    ...
    neural_trial = bin_spikes_for_trial(...)
    wheel_trial = interpolate_behavior_trial(...)
    whisker_trial = interpolate_behavior_trial(...)
```

iii. File I/O dominates, followed by per-trial processing.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop iterating over `trials_df.iterrows()` performs spike binning and behavioral interpolation for each trial individually. The spike binning could be vectorized across trials (e.g., by offsetting spike indices by trial). The behavioral interpolation loop could also potentially be batched.

ii.
```python
for trial_idx, trial_row in trials_df.iterrows():
    ...
    neural_trial = bin_spikes_for_trial(...)
    wheel_trial = interpolate_behavior_trial(...)
    whisker_trial = interpolate_behavior_trial(...)
```

iii. Using `iterrows()` is particularly slow as it converts each row to a Series.

## 10-c. What processing does the code repeat multiple times?

i. The code collects all wheel and whisker continuous values during the main processing loop (appending to `wheel_continuous_all` and `whisker_continuous_all`), computes global tertiles after all sessions are processed, then iterates over all sessions again to apply the discretization thresholds. This requires a second pass over all behavioral data.

ii.
```python
# First pass: collect all continuous values
wheel_continuous_all.append(wheel_trial)
whisker_continuous_all.append(whisker_trial)
...
# Compute global edges
wheel_flat = np.concatenate(wheel_continuous_all)
_, wheel_edges = discretize_three_bins(wheel_flat)
...
# Second pass: apply discretization
for session in kept_sessions:
    ...
    wheel_bins = np.digitize(wheel_trial, [wheel_edges[0], wheel_edges[1]], ...)
```

iii. The two-pass approach is a consequence of using global tertiles instead of per-session percentiles.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The trial mask checks finiteness of `feedback_times`, `feedbackType`, and `goCue_times`, and computes `trial_len = feedback_times - goCue_times` with a 10 s threshold. These columns and the trial length check are not used by the decoder or required by the reference processing. The code also computes acceleration in `velocity_filtered` which is never used.

ii.
```python
required = ["stimOn_times", "choice", "feedback_times", "probabilityLeft",
            "firstMovement_times", "feedbackType", "goCue_times"]
...
trial_len = trials_df["feedback_times"].to_numpy() - trials_df["goCue_times"].to_numpy()
mask &= trial_len <= MAX_TRIAL_LEN_S
```

```python
acc = np.insert(np.diff(vel), 0, 0.0) * fs
return vel, acc  # acc is never used
```

iii. These appear to be extra quality checks from the reference code's `prepare_data()` function that go beyond what the decoder requires.
