# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads `bwm_release.csv` from the reference code repository to enumerate all sessions and their probes. It then constructs filesystem paths directly to the ONE cache directory structure (`one_cache/<lab>/Subjects/<subject>/<date>/<session_number>/alf/`) rather than using the ONE API. Data files (trials table, wheel, camera, spikes) are loaded by searching the ALF directory for the latest revision file matching glob patterns. No DATALIMIT_SUBSET.csv filtering is applied.

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
    ONE_CACHE_DIR
    / session["lab"]
    / "Subjects"
    / session["subject"]
    / session["date"]
    / f"{session['session_number']:03d}"
)
alf_path = session_path / "alf"
```

iii. The agent examined the local ONE cache layout and the `bwm_release.csv` file to understand session enumeration. It decided to load files directly from the filesystem with revision resolution rather than using the ONE API, reasoning that it needed to work offline against the cached data.

## 1-b. How are the data split into subjects?

i. Subject names come from the `subject` column in `bwm_release.csv`. After conversion, unique subjects are collected in insertion order and `subject_idx` maps each session to its subject index.

ii.
```python
subjects = ordered_unique(session["subject"] for session in kept_sessions)
subject_lookup = {subject: idx for idx, subject in enumerate(subjects)}
```

iii. The subject name is directly available in the release CSV, so no parsing or derivation is needed.

## 1-c. How are the data split into sessions?

i. Each unique `eid` in `bwm_release.csv` defines a session. The groupby on `eid` produces one entry per session with its probe names.

ii.
```python
for eid, group in release_df.groupby("eid", sort=False):
    ...
```

iii. Sessions are the natural unit in the release CSV, so no splitting logic is needed.

## 1-d. How are the data split into trials?

i. The trials table (`_ibl_trials.table.pqt`) has one row per trial. The code iterates over rows using `iterrows()`.

ii.
```python
trials_df = load_trials_table(alf_path)
...
for trial_idx, trial_row in trials_df.iterrows():
```

iii. The trials table is already one row per trial, so no further splitting is required.

## 1-e. How are trials filtered based on quality controls?

i. Multiple filters are applied: (1) Several required columns must be finite (`stimOn_times`, `choice`, `feedback_times`, `probabilityLeft`, `firstMovement_times`, `feedbackType`, `goCue_times`). (2) Reaction time must be between 80 ms and 2 s. (3) Trial length (`feedback_times - goCue_times`) must be <= 10 s. (4) Choice must not be 0 (no response). (5) Wheel and whisker motion energy must cover the trial window. (6) Neural data must have at least one spike in the trial window.

ii.
```python
def compute_trial_mask(trials_df: pd.DataFrame):
    required = [
        "stimOn_times", "choice", "feedback_times",
        "probabilityLeft", "firstMovement_times",
        "feedbackType", "goCue_times",
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
if not np.any(neural_trial):
    continue
```

iii. The agent chose these filters based on the reference code's trial masking logic and the data paper's quality controls. The additional trial length filter and finiteness checks on extra columns go beyond the reference.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Two arrays per probe: `spikes.times.npy` and `spikes.clusters.npy`, loaded from the pykilosort subdirectory. The cluster metrics table (`clusters.metrics.pqt`) provides quality labels, and `clusters.channels.npy` with `channels.brainLocationIds_ccf_2017.npy` provide anatomical locations.

ii.
```python
spike_times = np.load(spike_times_path).astype(np.float64)
spike_clusters = np.load(spike_clusters_path).astype(np.int64)
```

iii. The agent identified spikes.times and spikes.clusters as the primary neural data, following the reference code's loading pattern.

## 2-b. How is the `neural` data processed?

i. Spikes are counted into 20 ms bins over a 2 s trial window (-0.5 to 1.5 s around stimulus onset), giving 100 bins per trial. The result is stored as raw spike counts in float16 format. **No division by bin width is applied**, so the data is spike counts, not firing rates. When a session has multiple probes, their units are merged into one population.

ii.
```python
def bin_spikes_for_trial(spike_times, spike_clusters, n_clusters, trial_start, binsize=BIN_SIZE_S, n_bins=N_BINS):
    ...
    bin_idx = np.floor((times - trial_start) / binsize).astype(np.int64)
    ...
    flat_idx = clusters[valid] * n_bins + bin_idx[valid]
    counts = np.bincount(flat_idx, minlength=n_clusters * n_bins).reshape(n_clusters, n_bins)
    return counts.astype(np.float16)
```

iii. The agent followed the reference code's binning approach but did not convert counts to firing rates (Hz). The trajectory shows the agent was focused on matching the binning logic but missed the rate conversion step.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only clusters with `label >= 1` in `clusters.metrics.pqt` are kept. Additionally, clusters in both `void` AND `root` brain regions are excluded. The brain region is determined from `channels.brainLocationIds_ccf_2017.npy` mapped through `BrainRegions.id2acronym()` (Allen CCF acronyms, not Beryl mapping).

ii.
```python
keep_mask = metrics["label"].to_numpy() >= 1.0
...
cluster_acronyms = region_ids_to_acronyms(cluster_region_ids, brain_regions)
keep_mask &= ~np.isin(cluster_acronyms, ["void", "root"])
```

iii. The agent investigated the quality control criteria in the data paper and decided to use `label >= 1` for well-isolated neurons. It also excluded `root` regions in addition to `void`, and used Allen CCF acronyms rather than the Beryl mapping.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial's spike window starts at `stimOn_times + OFF_START_S` (-0.5 s before stimulus onset) and ends at `stimOn_times + OFF_END_S` (1.5 s after). Spikes are binned relative to this absolute start time.

ii.
```python
trial_start = float(trial_row["stimOn_times"] + OFF_START_S)
trial_end = float(trial_row["stimOn_times"] + OFF_END_S)
neural_trial = bin_spikes_for_trial(
    merged["spike_times"], merged["spike_clusters"],
    n_clusters, trial_start,
)
```

iii. The alignment is equivalent to the reference: binning relative to `trial_start = stimOn + T_START` is mathematically the same as subtracting stimulus onset then binning from `T_START`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is 20 ms, producing 100 bins over the 2 s window. No rebinning or smoothing is applied.

ii.
```python
BIN_SIZE_S = 0.02
OFF_START_S = -0.5
OFF_END_S = 1.5
N_BINS = int(round((OFF_END_S - OFF_START_S) / BIN_SIZE_S))
```

iii. The agent matched the reference code's 20 ms bin size and 100-bin count.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is derived from the trial's `stimOn_times` and the window parameters. The time values are computed as evenly spaced points within the trial window.

ii.
```python
time_since_stim = np.linspace(
    OFF_START_S + BIN_SIZE_S,
    OFF_END_S,
    N_BINS,
    dtype=np.float32,
)
```

iii. The agent constructed a time vector representing time since stimulus onset across the trial window.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The time values are computed using `np.linspace` from `OFF_START_S + BIN_SIZE_S` (-0.48) to `OFF_END_S` (1.5), producing 100 evenly spaced values. These correspond to the right edges of the bins rather than the bin centers.

ii.
```python
time_since_stim = np.linspace(
    OFF_START_S + BIN_SIZE_S,
    OFF_END_S,
    N_BINS,
    dtype=np.float32,
)
```

iii. The agent intended these to be bin-aligned time values but computed right edges instead of centers.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. Both use the same number of bins (100) and the same trial window, so they are dimensionally aligned. However, the time input values represent bin right edges (-0.48 to 1.5) while the neural data is binned starting from bin left edges (-0.5 to 1.48), creating a slight misalignment in the time labels.

ii.
```python
# Neural: bins from trial_start = stimOn + OFF_START_S
bin_idx = np.floor((times - trial_start) / binsize).astype(np.int64)

# Input time: right edges
time_since_stim = np.linspace(OFF_START_S + BIN_SIZE_S, OFF_END_S, N_BINS)
```

iii. The agent used the same 100-bin structure for both neural and input data but did not explicitly verify that the time grid matches bin centers.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. From `probabilityLeft` in the trials table. A change in `probabilityLeft` value indicates a new block boundary.

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

iii. The agent identified that block boundaries must be inferred from changes in `probabilityLeft`, as no explicit block identifier exists in the trials table.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. A loop iterates through trials, incrementing a counter when `probabilityLeft` stays the same, resetting to 1 when it changes. The count is 1-based (first trial in a block is 1). The computation is done on the full (unfiltered) trials table, then indexed per trial.

ii.
```python
count = count + 1 if same_block else 1
numbers[idx] = float(count)
```

iii. The agent chose a 1-based count where the first trial in a block has number 1. This differs from the reference which uses 0-based counting via `pandas.groupby.cumcount()`.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. From the `choice` column in the trials table, which is +1 (left), -1 (right), or 0 (no response).

ii.
```python
def choice_to_class(value):
    if np.isclose(value, 1.0):
        return 0
    if np.isclose(value, -1.0):
        return 1
    raise ValueError(f"Unexpected choice value: {value}")
```

iii. The agent correctly identified the IBL choice encoding convention.

## 5-b. What processing is involved in computing `output` *Choice*?

i. Choice is recoded from +1/-1 to 0/1 (left=0, right=1). No-response trials (choice=0) are excluded by the trial mask. The value is broadcast to all 100 time bins.

ii.
```python
output_trial = np.vstack([
    np.full(N_BINS, choice_class, dtype=np.int8),
    ...
])
```

iii. Straightforward recoding matching the instruction specification.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. From the `probabilityLeft` column in the trials table, which takes values 0.2, 0.5, and 0.8.

ii.
```python
def probability_left_to_class(value):
    mapping = {0.2: 0, 0.5: 1, 0.8: 2}
    for key, encoded in mapping.items():
        if np.isclose(value, key):
            return encoded
    raise ValueError(f"Unexpected probabilityLeft value: {value}")
```

iii. Direct mapping as specified in the instructions.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The three values are recoded to 0, 1, 2 using `np.isclose` comparison and broadcast to all time bins.

ii. Same as 6-a code snippet.

iii. No additional processing beyond the mapping.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. From `_ibl_wheel.timestamps.npy` and `_ibl_wheel.position.npy`, loaded directly from the filesystem.

ii.
```python
timestamps = np.load(timestamps_path).astype(np.float64)
position = np.load(position_path).astype(np.float64)
```

iii. The agent identified the raw wheel files as the source.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The wheel position is interpolated onto a 1000 Hz grid, then an 8th-order 20 Hz Butterworth low-pass filter is applied via `sosfiltfilt`, and the velocity is computed by differentiation. Speed is the absolute value of this velocity. The AI reimplemented this processing manually rather than using `SessionLoader`.

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
```

iii. The agent examined the `SessionLoader.load_wheel` source code and reimplemented the same interpolation and filtering pipeline.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. The wheel speed is discretized into 3 bins using **global** tertile thresholds computed across ALL retained sessions/trials/time bins, not per-session. The thresholds are at the 1/3 and 2/3 quantiles of the pooled data.

ii.
```python
wheel_flat = np.concatenate(wheel_continuous_all)
_, wheel_edges = discretize_three_bins(wheel_flat)
...
def discretize_three_bins(values):
    q1, q2 = np.quantile(values, [1.0 / 3.0, 2.0 / 3.0])
    bins = np.digitize(values, [q1, q2], right=False).astype(np.int8)
    return bins, (float(q1), float(q2))
```

iii. The agent chose global tertiles to ensure consistent bin edges across all sessions. This differs from the reference which uses per-session percentiles.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The wheel speed is interpolated onto 100 time points using `np.linspace(interval_start + binsize, interval_end, N_BINS)`, which produces right bin edges rather than bin centers. This is the same grid used for the time input.

ii.
```python
x_interp = np.linspace(interval_start + binsize, interval_end, n_bins, dtype=np.float64)
y_interp = np.interp(x_interp, trial_times, trial_values).astype(np.float32)
```

iii. The agent used the same time grid for all behavioral variables to ensure internal consistency, though this grid uses bin right edges rather than centers.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. From `<side>Camera.ROIMotionEnergy.npy` and `_ibl_<side>Camera.times.npy`, with left camera preferred over right.

ii.
```python
def load_motion_energy(alf_path: Path):
    for view in ("left", "right"):
        me_path = latest_revision_file(alf_path, f"#*/{view}Camera.ROIMotionEnergy.npy")
        ...
        times_path = latest_revision_file(alf_path, [...])
        ...
        return timestamps, motion_energy, view
```

iii. The agent followed the reference's camera selection logic (left preferred).

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The raw motion energy trace is used as-is with no additional filtering or normalization. When camera timestamps are longer than the motion energy array, the timestamps are truncated from the beginning (taking the last N timestamps). It is then interpolated onto the trial time grid and discretized.

ii.
```python
if timestamps.shape[0] > motion_energy.shape[0]:
    timestamps = timestamps[-motion_energy.shape[0]:]
```

iii. The agent handled the timestamp/motion-energy length mismatch by taking trailing timestamps, following IBL conventions.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Same as wheel speed: **global** tertile thresholds computed across all retained sessions/trials/time bins.

ii.
```python
whisker_flat = np.concatenate(whisker_continuous_all)
_, whisker_edges = discretize_three_bins(whisker_flat)
...
whisker_bins = np.digitize(whisker_trial, [whisker_edges[0], whisker_edges[1]], right=False).astype(np.int8)
```

iii. The agent applied the same global discretization approach as for wheel speed.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Same as wheel speed: interpolated onto bin right edges using `np.linspace(interval_start + binsize, interval_end, N_BINS)`.

ii.
```python
x_interp = np.linspace(interval_start + binsize, interval_end, n_bins, dtype=np.float64)
y_interp = np.interp(x_interp, trial_times, trial_values).astype(np.float32)
```

iii. Same alignment approach as wheel speed, using right edges.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Multiple levels of error handling: (1) Required trial columns must be finite or the trial is dropped. (2) Wheel/whisker data must cover the trial window or the trial is dropped. (3) Neural trials with zero spikes are dropped. (4) Sessions with fewer than 2 retained trials are dropped. (5) Sessions where probe loading fails (no good units) are dropped. (6) Any exception during session processing drops the entire session with an error log.

ii.
```python
if wheel_trial is None or whisker_trial is None:
    continue
if not np.any(neural_trial):
    continue
if len(session_neural) < 2:
    dropped_sessions.append((session["eid"], "fewer_than_two_trials"))
    continue
```

iii. The agent applied conservative error handling, preferring to drop problematic data rather than risk corrupted entries.

## 10-a. What are the most time-consuming steps of the code?

i. Loading spike sorting data from disk (spike times and clusters arrays) is the most expensive I/O operation. The session-by-session loop with per-trial iteration is the main computational cost.

ii.
```python
spike_times = np.load(spike_times_path).astype(np.float64)
spike_clusters = np.load(spike_clusters_path).astype(np.int64)
```

iii. The agent's trajectory shows the full conversion taking several minutes, with most time spent on I/O.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in the main conversion iterates using `iterrows()` and processes one trial at a time. The spike binning, behavioral interpolation, and output construction could potentially be vectorized across trials.

ii.
```python
for trial_idx, trial_row in trials_df.iterrows():
    if not trial_mask[trial_idx]:
        continue
    ...
    neural_trial = bin_spikes_for_trial(...)
    wheel_trial = interpolate_behavior_trial(...)
    whisker_trial = interpolate_behavior_trial(...)
```

iii. The per-trial loop is straightforward but not vectorized. `iterrows()` is particularly slow for pandas DataFrames.

## 10-c. What processing does the code repeat multiple times?

i. The `compute_trial_number_in_block` function is called once per session, which is efficient. However, the `discretize_three_bins` function is effectively computed twice: once to get the edges from the global pool, and then `np.digitize` is called again per-trial in the output assembly loop.

ii.
```python
_, wheel_edges = discretize_three_bins(wheel_flat)
...
# Later, per trial:
wheel_bins = np.digitize(wheel_trial, [wheel_edges[0], wheel_edges[1]], right=False)
```

iii. The two-pass discretization (first global edges, then per-trial application) is necessary for the global thresholding approach.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code loads `feedbackType` and `goCue_times` columns and checks them for finiteness as part of trial filtering, but these variables are not used in any decoder input or output. The `MAX_TRIAL_LEN_S` filter based on `feedback_times - goCue_times` is additional processing not required by the reference. The `build_sample_subset` function creates a sample dataset that may not be needed.

ii.
```python
required = [
    "stimOn_times", "choice", "feedback_times",
    "probabilityLeft", "firstMovement_times",
    "feedbackType", "goCue_times",
]
trial_len = trials_df["feedback_times"].to_numpy() - trials_df["goCue_times"].to_numpy()
mask &= trial_len <= MAX_TRIAL_LEN_S
```

iii. These additional filters are extra processing beyond what the reference requires.
