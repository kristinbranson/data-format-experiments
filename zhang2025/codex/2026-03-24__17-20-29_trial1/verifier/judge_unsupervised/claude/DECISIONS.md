# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from a local IBL ONE cache. It reads a session manifest parquet file (`sessions.pqt`) from one of three possible cache directories (`2025_Q3_IBL_et_al_BWM`, `Brainwidemap`, or `2022_Q4_IBL_et_al_BWM`). For each session in the manifest, it constructs a path to the session directory (`<lab>/Subjects/<subject>/<date>/<number>/`) and checks whether it exists locally. Sessions without local data are skipped. Each session is then processed independently: trial tables, spike data (from all probes), wheel data, and whisker motion energy data are loaded from the `alf/` subdirectory of each session.

ii.
```python
MANIFEST_FILES = [
    DATA_ROOT / "2025_Q3_IBL_et_al_BWM" / "sessions.pqt",
    DATA_ROOT / "Brainwidemap" / "sessions.pqt",
    DATA_ROOT / "2022_Q4_IBL_et_al_BWM" / "sessions.pqt",
]

def load_session_manifest() -> pd.DataFrame:
    for manifest in MANIFEST_FILES:
        if manifest.exists():
            return pd.read_parquet(manifest)
    raise FileNotFoundError("No session manifests found in data cache")

def resolve_session_specs() -> tuple[list[SessionSpec], list[str]]:
    manifest = load_session_manifest()
    specs: list[SessionSpec] = []
    missing: list[str] = []
    for eid, row in manifest.iterrows():
        session_path = (
            DATA_ROOT / row["lab"] / "Subjects" / row["subject"]
            / str(row["date"]) / f"{int(row['number']):03d}"
        )
        if not session_path.exists():
            missing.append(eid)
            continue
        specs.append(SessionSpec(...))
    return specs, missing
```

iii. The AI documented that the `2025_Q3_IBL_et_al_BWM/sessions.pqt` manifest lists 459 sessions from the canonical BWM release, matching the data paper's count. It chose this manifest as the primary session index, treating missing local data as expected given the cache may not be complete.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are derived from the session manifest's `subject` column. The AI collects all unique subject names across processed sessions and creates a sorted list. Each session is assigned a `subject_idx` pointing into this list.

ii.
```python
subjects = sorted({s.subject for s in sessions})
subject_to_idx = {s: i for i, s in enumerate(subjects)}
data["subjects"] = subjects
data["subject_idx"] = np.array([subject_to_idx[s.subject] for s in sessions], dtype=np.int16)
```

iii. The AI noted that 139 subjects exist in the full manifest (matching the data paper), but only 135 subjects remain after dropping 21 sessions that lack required data modalities. The 4 dropped subjects had all their sessions fall into unusable categories.

## 1-c. How are the data split into sessions?

i. Each session is identified by its unique `eid` from the manifest and a corresponding file path. Sessions are processed independently, with each session contributing one entry to the `neural`, `input`, and `output` lists. Sessions are skipped if they lack required data (spikes, wheel, whisker motion energy) or have fewer than 2 valid trials.

ii.
```python
def process_session(spec: SessionSpec, br: BrainRegions) -> ProcessedSession | None:
    ...
    if n_clusters_good == 0:
        print(f"[skip] {spec.eid}: no good clusters after QC")
        return None
    try:
        wheel_times, wheel_speed = load_wheel_speed(spec.session_path)
        whisk_times, whisk_values, whisk_source = load_whisker_motion_energy(spec.session_path)
    except FileNotFoundError as exc:
        print(f"[skip] {spec.eid}: {exc}")
        return None
    ...
    if len(masked_trials) < 2:
        print(f"[skip] {spec.eid}: fewer than 2 trials after trial mask")
        return None
```

iii. The AI documented that 438 of 459 sessions survived, with 21 dropped due to missing wheel files (6), missing whisker motion energy (14), or fewer than 2 trials (1).

## 1-d. How are the data split into trials?

i. Trials are loaded from the `_ibl_trials.table.pqt` parquet file for each session. Each row represents one trial. After applying quality filters (trial mask), trials are further filtered by behavioral data availability (wheel and whisker alignment success) and neural data validity (non-zero spike counts). Each surviving trial becomes one entry in the session's trial list.

ii.
```python
def load_trials_table(session_path: Path) -> pd.DataFrame:
    trial_file = pick_one_file(session_path / "alf", "_ibl_trials.table.pqt")
    ...
    return pd.read_parquet(trial_file)

# In process_session:
trials = load_trials_table(spec.session_path)
trial_mask = compute_trial_mask(trials)
masked_trials = trials.loc[trial_mask].reset_index(drop=False)
# ... then further filtered by combined_mask (wheel, whisker, neural validity)
```

iii. The AI described trials as rows in the parquet table, with each trial having associated timing events, choices, and behavioral variables.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered using multiple criteria: (1) required event columns must not be NaN (`stimOn_times`, `choice`, `feedback_times`, `probabilityLeft`, `firstMovement_times`, `feedbackType`); (2) reaction time (`firstMovement_times - stimOn_times`) must be between 0.08 and 2.0 seconds; (3) trial duration (`feedback_times - goCue_times`) must be <= 10.0 seconds; (4) `choice != 0` (no-go trials excluded). Additionally, trials without valid behavioral interpolation (wheel or whisker) or with all-zero neural activity are dropped.

ii.
```python
def compute_trial_mask(trials: pd.DataFrame) -> np.ndarray:
    required = ["stimOn_times", "choice", "feedback_times", "probabilityLeft",
                 "firstMovement_times", "feedbackType"]
    mask = np.ones(len(trials), dtype=bool)
    rt = trials["firstMovement_times"] - trials["stimOn_times"]
    mask &= rt >= TRIAL_MASK_RT[0]  # 0.08
    mask &= rt <= TRIAL_MASK_RT[1]  # 2.0
    mask &= (trials["feedback_times"] - trials["goCue_times"]) <= MAX_TRIAL_LEN  # 10.0
    mask &= trials["choice"] != 0
    for col in required:
        mask &= trials[col].notna().to_numpy()
    return mask

# Additional filtering in process_session:
neural_mask = np.array([np.any(trial) for trial in neural_trials], dtype=bool)
combined_mask = wheel_mask & whisk_mask & neural_mask
```

iii. The AI justified this by referencing the `load_trials_and_mask()` function in the Zhang 2025 reference code, which uses the same filters. The all-zero neural trial filter was added after initial verification revealed 16 warnings.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `spikes.times.npy` and `spikes.clusters.npy` files in the `pykilosort/` subdirectory of each probe. Cluster quality is determined from `clusters.metrics.pqt` (the `label` column). Brain region assignments use `clusters.channels.npy` and `channels.brainLocationIds_ccf_2017.npy`.

ii.
```python
def load_probe_spikes_and_regions(probe_path, br, label_threshold=GOOD_CLUSTER_LABEL):
    spikes_times_file = pick_one_file(pykilo_path, "spikes.times.npy")
    spikes_clusters_file = pick_one_file(pykilo_path, "spikes.clusters.npy")
    metrics_file = pick_one_file(pykilo_path, "clusters.metrics.pqt")
    ...
    spikes_times = np.load(spikes_times_file, mmap_mode="r")
    spikes_clusters = np.load(spikes_clusters_file, mmap_mode="r")
```

iii. The AI noted these are standard IBL spike-sorted outputs from Kilosort, consistent with the reference code's `load_spiking_data` function.

## 2-b. How is the `neural` data processed?

i. Spikes from all probes in a session are merged, with cluster IDs reindexed to be unique across probes. Only clusters with `label >= 1` are kept. Spike times are sorted. Spikes are then binned into 20ms time bins within a [-0.5, 1.5]s window around stimulus onset for each trial, producing a (n_neurons, 100) matrix per trial. The binning uses `bincount2D` from `iblutil.numerical`.

ii.
```python
def load_session_spikes(session_path, br, label_threshold=GOOD_CLUSTER_LABEL):
    # merge probes, reindex clusters, sort by time
    ...
    spike_times = np.concatenate(merged_times)
    spike_clusters = np.concatenate(merged_clusters)
    cluster_regions = np.concatenate(merged_regions)
    order = np.argsort(spike_times, kind="stable")
    ...

def bin_spikes_for_trials(...):
    intervals = np.c_[align_times + window[0], align_times + window[1]]
    ...
    counts, _, cluster_idx = bincount2D(
        spike_times[idx0:idx1], spike_clusters[idx0:idx1],
        xbin=binsize, xlim=[start, end],
    )
    trial_counts[cluster_idx, : counts.shape[1]] = counts.astype(np.float16)
```

iii. The AI documented that this approach matches the reference code's `merge_probes` and `bin_spiking_data` functions, using the same binning parameters (20ms bins, [-0.5, 1.5]s window).

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only clusters with `clusters.metrics.label >= 1` are kept. This is the IBL's "good cluster" quality criterion. Clusters failing this threshold are excluded before any spike binning occurs. Additionally, trials where all neural bins are zero are excluded from the final output.

ii.
```python
GOOD_CLUSTER_LABEL = 1

def load_probe_spikes_and_regions(probe_path, br, label_threshold=GOOD_CLUSTER_LABEL):
    metrics = pd.read_parquet(metrics_file, columns=["label"])
    cluster_labels = metrics["label"].to_numpy()
    good_mask = cluster_labels >= label_threshold
    selected_cluster_ids = np.flatnonzero(good_mask)
    ...
    spike_keep = good_mask[spikes_clusters]
    ...

# In process_session:
neural_mask = np.array([np.any(trial) for trial in neural_trials], dtype=bool)
```

iii. The AI noted that the reference code (Zhang 2025) loads all clusters and stores `good_clusters = label >= 1` as metadata but doesn't explicitly filter. The AI chose to apply the filter to match the data paper's 75,708 well-isolated neuron count and keep the output tractable.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to stimulus onset (`stimOn_times`). For each trial, spikes within the window `[stimOn_times - 0.5, stimOn_times + 1.5]` are collected and binned.

ii.
```python
TIME_WINDOW = (-0.5, 1.5)
# In process_session:
align_times = masked_trials["stimOn_times"].to_numpy(dtype=np.float64)
neural_trials = bin_spikes_for_trials(
    spike_times, spike_clusters, n_clusters=n_clusters_good, align_times=align_times,
)
```

iii. The AI justified this by noting the instructions explicitly say "Temporally align based on stimulus onset" and the reference code uses `align_time='stimOn_times'` with `time_window=(-0.5, 1.5)`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is 20ms (0.02s), producing 100 time bins per trial over the 2-second window. No rebinning is applied -- spike counts are directly binned at 20ms resolution from the raw spike times. This is stored as `time_bin_size: 20.0` (in ms) in metadata.

ii.
```python
BINSIZE_S = 0.02
N_BINS = int(round((TIME_WINDOW[1] - TIME_WINDOW[0]) / BINSIZE_S))  # = 100
```

iii. The AI noted that 20ms bins match the reference code's `binsize=0.02` and the generic model description (`T = 100` over 2s). The method paper's mention of 50ms bins for choice/prior was considered inconsistent with the executable code and was not followed.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. The time-since-stimulus-onset input is not derived from any raw data variable. It is a constructed time grid based on the alignment window parameters.

ii.
```python
def make_time_input() -> np.ndarray:
    return np.linspace(TIME_WINDOW[0] + BINSIZE_S, TIME_WINDOW[1], N_BINS, dtype=np.float32)
```

iii. The AI described this as a signed time grid from -0.48 to 1.50 seconds in 100 bins, matching the right-edge sample times of the behavioral interpolation grid.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The time input is created using `np.linspace(-0.48, 1.5, 100)`. This produces evenly spaced time points representing the right edges of each 20ms bin, starting from `TIME_WINDOW[0] + BINSIZE_S = -0.5 + 0.02 = -0.48` to `TIME_WINDOW[1] = 1.5`. The same array is used for every trial.

ii.
```python
def make_time_input() -> np.ndarray:
    return np.linspace(TIME_WINDOW[0] + BINSIZE_S, TIME_WINDOW[1], N_BINS, dtype=np.float32)
# N_BINS = 100, TIME_WINDOW = (-0.5, 1.5), BINSIZE_S = 0.02
```

iii. The AI stated this matches the behavioral interpolation grid used in the reference code.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. Both the time input and the neural data use the same 100-bin temporal grid aligned to stimulus onset. The time input array is identical for all trials, representing the time offsets of each bin from stimulus onset.

ii.
```python
# In process_session:
time_input = make_time_input()
for block_num, choice_val, prior_val in zip(block_vals, choice_vals, prior_vals):
    input_trial = np.vstack([
        time_input,
        np.full(N_BINS, block_num, dtype=np.float32),
    ]).astype(np.float32)
    inputs.append(input_trial)
```

iii. The AI noted the time input is a deterministic grid that matches the neural binning window, ensuring temporal alignment.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. Trial number in block is derived from the `probabilityLeft` column in the trials table. It counts consecutive trials with the same `probabilityLeft` value.

ii.
```python
def compute_trial_number_in_block(prob_left: np.ndarray) -> np.ndarray:
    counters = np.zeros(len(prob_left), dtype=np.float32)
    if len(prob_left) == 0:
        return counters
    count = 1
    counters[0] = count
    for i in range(1, len(prob_left)):
        if prob_left[i] == prob_left[i - 1]:
            count += 1
        else:
            count = 1
        counters[i] = count
    return counters
```

iii. The AI justified computing this on the unfiltered trial table, noting that "excluded trials should not renumber the latent block progression."

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The block trial number is computed on the full (unmasked) trial sequence by iterating through `probabilityLeft` values. A counter starts at 1 and increments for each consecutive trial with the same `probabilityLeft`; it resets to 1 when the value changes. The resulting count is then looked up for each kept trial using the original trial index and repeated across all 100 time bins.

ii.
```python
# In process_session:
block_trial_number = compute_trial_number_in_block(trials["probabilityLeft"].to_numpy())
# ... after filtering:
block_vals = block_trial_number[masked_keep["index"].to_numpy()]
# ... then repeated across bins:
np.full(N_BINS, block_num, dtype=np.float32)
```

iii. The AI documented this as a deliberate design choice to preserve the correct block counting even when some trials within a block are excluded.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. Choice is derived from the `choice` column in the trials table (`_ibl_trials.table.pqt`).

ii.
```python
choice_vals = map_choice_to_binary(masked_keep["choice"].to_numpy(dtype=np.float64))
```

iii. The AI documented that IBL encodes choices as -1 (right), +1 (left), and 0 (no-go), referencing both the raw data and the IBL documentation.

## 5-b. What processing is involved in computing `output` *Choice*?

i. After excluding no-go trials (`choice == 0`) via the trial mask, the remaining values are remapped: `choice == 1` (left) maps to 0, `choice == -1` (right) maps to 1. The binary value is then repeated across all 100 time bins for the trial.

ii.
```python
def map_choice_to_binary(choice_values: np.ndarray) -> np.ndarray:
    mapped = np.full(choice_values.shape, -1, dtype=np.int16)
    mapped[choice_values == 1] = 0   # left
    mapped[choice_values == -1] = 1  # right
    if np.any(mapped < 0):
        raise ValueError("Unexpected choice values after masking")
    return mapped

# Repeated across bins:
choice.append(np.full(N_BINS, choice_val, dtype=np.int16))
```

iii. The AI justified this mapping using the instructions: "left = 0, right = 1".

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. Prior probability of left is derived from the `probabilityLeft` column in the trials table.

ii.
```python
prior_vals = map_prior_to_categorical(masked_keep["probabilityLeft"].to_numpy(dtype=np.float64))
```

iii. The AI documented that `probabilityLeft` contains values from {0.2, 0.5, 0.8} representing the block-wise prior probability of the stimulus appearing on the left.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The raw `probabilityLeft` values are mapped to categorical indices: 0.2 -> 0, 0.5 -> 1, 0.8 -> 2. The categorical value is repeated across all 100 time bins.

ii.
```python
def map_prior_to_categorical(prob_left: np.ndarray) -> np.ndarray:
    mapped = np.full(prob_left.shape, -1, dtype=np.int16)
    mapped[np.isclose(prob_left, 0.2)] = 0
    mapped[np.isclose(prob_left, 0.5)] = 1
    mapped[np.isclose(prob_left, 0.8)] = 2
    if np.any(mapped < 0):
        vals = np.unique(prob_left[mapped < 0])
        raise ValueError(f"Unexpected probabilityLeft values: {vals}")
    return mapped

# Repeated across bins:
prior.append(np.full(N_BINS, prior_val, dtype=np.int16))
```

iii. The AI noted this matches the instructions exactly: "0.2 -> 0, 0.5 -> 1, 0.8 -> 2".

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. Wheel speed is derived from `_ibl_wheel.position.npy` (wheel rotary encoder position) and `_ibl_wheel.timestamps.npy` (timestamps) in the session's `alf/` directory.

ii.
```python
def load_wheel_speed(session_path: Path) -> tuple[np.ndarray, np.ndarray]:
    wheel_pos_file = pick_one_file(session_path / "alf", "_ibl_wheel.position.npy")
    wheel_ts_file = pick_one_file(session_path / "alf", "_ibl_wheel.timestamps.npy")
    ...
    pos = np.asarray(np.load(wheel_pos_file), dtype=np.float64)
    ts = np.asarray(np.load(wheel_ts_file), dtype=np.float64)
```

iii. The AI documented these as the standard IBL wheel data files.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Wheel position is interpolated to 1000 Hz using `interpolate_position` from the bundled `brainbox.behavior.wheel` module. A filtered velocity is then computed using `velocity_filtered` at 1000 Hz. The absolute value of velocity is taken to get speed. This continuous speed signal is then interpolated onto the trial-aligned 20ms grid using linear interpolation.

ii.
```python
pos_interp, ts_interp = interpolate_position(ts, pos, freq=1000)
vel, _ = velocity_filtered(pos_interp, 1000)
return ts_interp, np.abs(vel)
```

iii. The AI referenced the `load_target_behavior('wheel-speed')` function in the reference code as the basis for this processing.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Wheel speed is discretized into 3 bins using global tertile edges computed across all valid wheel speed samples from all sessions. The 1/3 and 2/3 quantiles of the pooled data define the two edges. `np.digitize` assigns each value to bin 0 (low), 1 (medium), or 2 (high).

ii.
```python
def compute_tertile_edges(values: Iterable[np.ndarray]) -> tuple[float, float]:
    concat = np.concatenate(flat_values)
    q1, q2 = np.quantile(concat, [1 / 3, 2 / 3])
    ...
    return float(q1), float(q2)

def discretize_three_bins(values: np.ndarray, edges: tuple[float, float]) -> np.ndarray:
    return np.digitize(values, bins=np.array(edges, dtype=np.float32), right=False).astype(np.int16)

# In main:
wheel_edges = compute_tertile_edges(
    trial_values for session in processed for trial_values in session.wheel_cont
)
```

iii. The AI chose global tertiles to ensure balanced class distributions across sessions, noting the instructions say "discretized into 3 bins" without specifying the method.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. Wheel speed is interpolated onto the same stimulus-onset-aligned 20ms grid used for neural data. The interpolation uses `interp1d` with linear interpolation and extrapolation for edge cases. Trials where behavioral data doesn't adequately cover the trial window (gap > one bin size at either end) are excluded.

ii.
```python
def interpolate_behavior_trials(target_times, target_values, align_times, ...):
    x_rel = np.linspace(window[0] + binsize, window[1], n_bins)
    ...
    interp = interp1d(ts, vals, kind="linear", fill_value="extrapolate")
    outputs.append(interp(align_time + x_rel).astype(np.float32))
```

iii. The AI documented that behavior alignment matches the neural alignment event (stimulus onset) as required by the instructions.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. Whisker motion energy is derived from `leftCamera.ROIMotionEnergy.npy` (or `rightCamera.ROIMotionEnergy.npy` as fallback) and the corresponding camera timestamps (`_ibl_leftCamera.times.npy` or `_ibl_rightCamera.times.npy`).

ii.
```python
def load_whisker_motion_energy(session_path: Path) -> tuple[np.ndarray, np.ndarray, str]:
    for camera in ("left", "right"):
        me_file = pick_one_file(session_path / "alf", f"{camera}Camera.ROIMotionEnergy.npy")
        times_file = pick_one_file(session_path / "alf", f"*{camera}Camera.times.npy")
        ...
        return times, values, camera
    raise FileNotFoundError(...)
```

iii. The AI documented the left-then-right fallback order as matching the reference code's `bin_behaviors()` function.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The raw whisker motion energy values are loaded directly (no additional computation like filtering or normalization). If the motion energy array and timestamps have different lengths, both are truncated to the shorter length. The values are then interpolated onto the trial-aligned 20ms grid using linear interpolation.

ii.
```python
values = np.asarray(np.load(me_file), dtype=np.float64)
times = np.asarray(np.load(times_file), dtype=np.float64)
if len(values) != len(times):
    n = min(len(values), len(times))
    values = values[:n]
    times = times[:n]
return times, values, camera
```

iii. The AI noted whisker motion energy is pre-computed in the IBL data pipeline as "the mean across pixels of the absolute value of the difference between adjacent frames."

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Same approach as wheel speed: global tertile edges are computed across all valid whisker motion energy samples from all sessions, then `np.digitize` assigns values to bins 0, 1, or 2.

ii.
```python
whisker_edges = compute_tertile_edges(
    trial_values for session in processed for trial_values in session.whisker_cont
)
# Then discretized per-trial:
discretize_three_bins(whisk_cont, whisker_edges)
```

iii. Same justification as wheel speed -- global tertiles for balanced categories.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Identical to wheel speed alignment: interpolated onto the stimulus-onset-aligned 20ms grid using `interpolate_behavior_trials`.

ii.
```python
whisk_trials, whisk_mask = interpolate_behavior_trials(whisk_times, whisk_values, align_times)
```

iii. The AI documented that all behavioral streams use the same alignment as neural data, consistent with the instructions.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Several strategies are used: (1) Sessions missing any required data modality (spikes, wheel, whisker) are skipped entirely. (2) Trials with NaN in required event columns are excluded by the trial mask. (3) Whisker motion energy length mismatches between timestamps and values are handled by truncation. (4) Wheel timestamps in 2D format are averaged. (5) Behavioral interpolation failures (insufficient data coverage) result in trial exclusion. (6) All-zero neural trials are excluded. (7) Versioned file paths are handled by selecting the latest version.

ii.
```python
# Whisker length mismatch:
if len(values) != len(times):
    n = min(len(values), len(times))
    values = values[:n]; times = times[:n]

# Wheel timestamp format:
if ts.ndim == 2 and ts.shape[1] == 2:
    ts = ts.mean(axis=1)

# Behavioral validity check:
if np.abs(start - ts[0]) > binsize or np.abs(end - ts[-1]) > binsize:
    outputs.append(None)  # trial excluded

# Version resolution:
def version_key(path: Path) -> tuple[int, str]:
    ...
matches.sort(key=lambda p: (version_key(p), str(p)))
return matches[-1]
```

iii. The AI documented these edge cases in CONVERSION_NOTES.md under the edge-case review section, noting robust handling was needed due to the complex IBL data cache structure.

## 10-a. What are the most time-consuming steps of the code?

i. The AI identified trial-by-trial spike binning (`bin_spikes_for_trials`) as the main bottleneck, as it involves loading large spike arrays and performing bincount operations for each trial. Session-level I/O (loading spike times, cluster data) is also significant.

ii.
```python
# The hot path - per-trial binning loop:
for idx0, idx1, (start, end) in zip(idx_starts, idx_ends, intervals):
    trial_counts = np.zeros((n_clusters, n_bins), dtype=np.float16)
    if idx1 > idx0:
        counts, _, cluster_idx = bincount2D(...)
```

iii. The AI documented this in Step 7 of CONVERSION_NOTES.md, noting sample conversion took ~13.1s per session and full conversion took ~14 minutes with 8 worker threads.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial-by-trial spike binning loop (iterating over each trial to call `bincount2D`) could potentially be vectorized. The behavioral interpolation loop (iterating over each trial to create interpolation objects) could also be vectorized. The `compute_trial_number_in_block` function uses a Python loop that could be replaced with numpy operations.

ii.
```python
# Trial loop in bin_spikes_for_trials (lines 321-333)
for idx0, idx1, (start, end) in zip(idx_starts, idx_ends, intervals):
    ...

# Trial loop in interpolate_behavior_trials (lines 385-398)
for i, align_time in enumerate(align_times):
    ...

# Python loop in compute_trial_number_in_block (lines 185-193)
for i in range(1, len(prob_left)):
    ...
```

iii. The AI noted that `bincount2D` spike binning is the main hot path but relied on existing vectorized functions within it rather than fully vectorizing the outer loop.

## 10-c. What processing does the code repeat multiple times?

i. `BrainRegions()` is instantiated once per session when using threaded parallelism (via `process_session_worker`), rather than being shared. The time input array `make_time_input()` is called once per session but produces the same array each time. Version resolution (`pick_one_file`) is called separately for each data file, each time scanning the directory tree.

ii.
```python
def process_session_worker(spec: SessionSpec) -> ProcessedSession | None:
    return process_session(spec, BrainRegions())  # new instance per session

# make_time_input called per session:
time_input = make_time_input()
```

iii. The AI did not explicitly address this redundancy in CONVERSION_NOTES.md.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code stores neural data as `float16` spike counts (raw counts, not firing rates or z-scored values). The downstream decoder may expect standardized neural data. The code also stores continuous wheel and whisker values in `ProcessedSession` objects that are only used for computing tertile edges and then discretized -- the continuous values are not saved to the final output. The `cluster_ids` variable in `bin_spikes_for_trials` is computed but never used.

ii.
```python
# Unused variable:
cluster_ids = np.arange(n_clusters, dtype=np.int32)  # line 316, never referenced

# Continuous values stored but only used for discretization:
wheel_cont: list[np.ndarray]  # stored in ProcessedSession, used for edges then discretized
whisker_cont: list[np.ndarray]  # same
```

iii. The AI did not discuss unnecessary processing in CONVERSION_NOTES.md, though it documented the discretization approach.
