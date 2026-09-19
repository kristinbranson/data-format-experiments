# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads a session manifest parquet file (`sessions.pqt`) from the local ONE cache to discover sessions, then resolves each session to a filesystem path (`DATA_ROOT / lab / Subjects / subject / date / number`). It does NOT use the ONE API to load data. Instead, it directly reads files from the resolved paths using `np.load`, `pd.read_parquet`, etc. It searches for the manifest across three possible locations (`2025_Q3_IBL_et_al_BWM`, `Brainwidemap`, `2022_Q4_IBL_et_al_BWM`).

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

def resolve_session_specs() -> tuple[list[SessionSpec], list[str]]:
    manifest = load_session_manifest()
    specs: list[SessionSpec] = []
    for eid, row in manifest.iterrows():
        session_path = (
            DATA_ROOT / row["lab"] / "Subjects" / row["subject"]
            / str(row["date"]) / f"{int(row['number']):03d}"
        )
        if not session_path.exists():
            missing.append(eid)
            continue
        specs.append(SessionSpec(...))
```

iii. The AI chose to bypass the ONE API and load files directly from the local cache. This is documented in CONVERSION_NOTES.md Step 2, where the AI describes the data structure as a local ONE cache with session data stored by lab/subject/date/number.

## 1-b. How are the data split into subjects?

i. Subjects are identified from the `subject` field of the session manifest parquet file. Each `SessionSpec` carries the subject name, and after processing, unique subjects are extracted and sorted to build the `subjects` list.

ii.
```python
specs.append(
    SessionSpec(
        eid=eid,
        lab=str(row["lab"]),
        subject=str(row["subject"]),
        ...
    )
)

# In build_data_dict:
subjects = sorted({s.subject for s in sessions})
subject_to_idx = {s: i for i, s in enumerate(subjects)}
```

iii. The manifest provides subject identity directly as a field.

## 1-c. How are the data split into sessions?

i. Each row in the session manifest corresponds to one session. The AI iterates over all rows, resolves each to a local path, and processes them independently.

ii.
```python
for eid, row in manifest.iterrows():
    session_path = (
        DATA_ROOT / row["lab"] / "Subjects" / row["subject"]
        / str(row["date"]) / f"{int(row['number']):03d}"
    )
```

iii. Sessions are the natural unit of the manifest; no splitting is needed.

## 1-d. How are the data split into trials?

i. The trials table (`_ibl_trials.table.pqt`) has one row per trial. No splitting is needed.

ii.
```python
def load_trials_table(session_path: Path) -> pd.DataFrame:
    trial_file = pick_one_file(session_path / "alf", "_ibl_trials.table.pqt")
    return pd.read_parquet(trial_file)
```

iii. The trials table is already one row per trial.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies four filtering criteria: (1) reaction time between 0.08 and 2.0 s, (2) choice != 0 (no-response excluded), (3) required columns are not NaN (`stimOn_times`, `choice`, `feedback_times`, `probabilityLeft`, `firstMovement_times`, `feedbackType`), and (4) trial duration (`feedback_times - goCue_times`) <= 10 s. Additionally, trials are dropped if wheel or whisker data don't cover the trial window, or if the neural window has all-zero spike counts.

ii.
```python
def compute_trial_mask(trials: pd.DataFrame) -> np.ndarray:
    mask = np.ones(len(trials), dtype=bool)
    rt = trials["firstMovement_times"] - trials["stimOn_times"]
    mask &= rt >= TRIAL_MASK_RT[0]
    mask &= rt <= TRIAL_MASK_RT[1]
    mask &= (trials["feedback_times"] - trials["goCue_times"]) <= MAX_TRIAL_LEN
    mask &= trials["choice"] != 0
    for col in required:
        mask &= trials[col].notna().to_numpy()
    return mask

# Later:
neural_mask = np.array([np.any(trial) for trial in neural_trials], dtype=bool)
combined_mask = wheel_mask & whisk_mask & neural_mask
```

iii. The AI documented that these filters match the reference code's `load_trials_and_mask()` function, and added the all-zero neural trial filter after full verification revealed warnings.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `spikes.times.npy` and `spikes.clusters.npy` from each probe's pykilosort directory, plus `clusters.metrics.pqt` for QC labels and `clusters.channels.npy` / `channels.brainLocationIds_ccf_2017.npy` for region assignment.

ii.
```python
spikes_times_file = pick_one_file(pykilo_path, "spikes.times.npy")
spikes_clusters_file = pick_one_file(pykilo_path, "spikes.clusters.npy")
metrics_file = pick_one_file(pykilo_path, "clusters.metrics.pqt")
```

iii. The AI loads spike times and cluster assignments directly from numpy files.

## 2-b. How is the `neural` data processed?

i. Spikes are counted into 20 ms bins over the [-0.5, 1.5] s trial window using `bincount2D`. The result is stored as spike counts (NOT divided by bin width), so the values are raw counts, not firing rates. Data is stored as float16.

ii.
```python
def bin_spikes_for_trials(...):
    ...
    counts, _, cluster_idx = bincount2D(
        spike_times[idx0:idx1],
        spike_clusters[idx0:idx1],
        xbin=binsize,
        xlim=[start, end],
    )
    if counts.size:
        counts = counts[:, :n_bins]
        trial_counts[cluster_idx, : counts.shape[1]] = counts.astype(np.float16)
    results.append(trial_counts)

# In build_data_dict:
data["neural"].append([trial.astype(np.float16) for trial in session.neural])
```

iii. The AI used `bincount2D` from `iblutil.numerical` for efficiency. The CONVERSION_NOTES.md Step 6 mentions "raw binned spike counts, not standardized z-scores." Probes are merged within each session.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only clusters with `clusters.metrics.label >= 1` are kept. The AI does NOT filter by brain region (no void/root filtering). It uses Allen atlas acronyms via `br.id2acronym()` rather than Beryl mapping.

ii.
```python
metrics = pd.read_parquet(metrics_file, columns=["label"])
cluster_labels = metrics["label"].to_numpy()
good_mask = cluster_labels >= label_threshold  # label_threshold = GOOD_CLUSTER_LABEL = 1

# Region assignment:
cluster_region_ids[valid_channel] = channel_ids[cluster_channels[valid_channel]]
cluster_regions = br.id2acronym(cluster_region_ids)
```

iii. The AI documented in CONVERSION_NOTES.md Step 4 that it chose `label >= 1` to match the data paper's 75,708 well-isolated neuron count.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial is aligned to `stimOn_times`. The spike binning function creates intervals `[stimOn_time + window[0], stimOn_time + window[1]]` and bins spikes within those intervals.

ii.
```python
align_times = masked_trials["stimOn_times"].to_numpy(dtype=np.float64)
intervals = np.c_[align_times + window[0], align_times + window[1]]
idx_starts = np.searchsorted(spike_times, intervals[:, 0], side="left")
idx_ends = np.searchsorted(spike_times, intervals[:, 1], side="left")
```

iii. Stimulus onset alignment is explicitly required by the instructions and matches the reference code.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 20 ms bins over a 2 s window, giving 100 time bins per trial. No rebinning or interpolation is applied to neural data.

ii.
```python
BINSIZE_S = 0.02
N_BINS = int(round((TIME_WINDOW[1] - TIME_WINDOW[0]) / BINSIZE_S))  # 100
```

iii. Matches the reference code's `binsize=0.02`.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is a synthetic variable computed from the time window and bin size parameters, not from any raw data variable. It represents the time grid of the neural bins.

ii.
```python
def make_time_input() -> np.ndarray:
    return np.linspace(TIME_WINDOW[0] + BINSIZE_S, TIME_WINDOW[1], N_BINS, dtype=np.float32)
```

iii. The AI computes this as a fixed grid that is the same for every trial.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The AI uses `np.linspace(-0.48, 1.5, 100)` to produce 100 evenly-spaced values from -0.48 to 1.50. These are the RIGHT EDGES of the bins, not the bin centers.

ii.
```python
return np.linspace(TIME_WINDOW[0] + BINSIZE_S, TIME_WINDOW[1], N_BINS, dtype=np.float32)
# Produces: [-0.48, -0.46, ..., 1.48, 1.50]
```

iii. The AI documented this as matching the behavior interpolation grid (right edges).

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The time input uses the same bin edges as the behavior interpolation. However, it uses right bin edges while the neural binning uses `bincount2D` with `xlim=[start, end]`, so there may be a slight misalignment between the time labels and the actual bin boundaries.

ii.
```python
# Time input uses right edges:
x_rel = np.linspace(window[0] + binsize, window[1], n_bins)

# Neural binning uses bincount2D with xlim:
counts, _, cluster_idx = bincount2D(
    spike_times[idx0:idx1], spike_clusters[idx0:idx1],
    xbin=binsize, xlim=[start, end],
)
```

iii. The AI's approach aligns the behavior interpolation with the time input, but the neural binning via `bincount2D` may use a slightly different bin grid definition.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. From `probabilityLeft` in the trials table. A block boundary is detected wherever the value changes.

ii.
```python
def compute_trial_number_in_block(prob_left: np.ndarray) -> np.ndarray:
    counters = np.zeros(len(prob_left), dtype=np.float32)
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

iii. The AI documented that block boundaries are detected from changes in `probabilityLeft`.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The AI counts trials within each block starting from 1 (not 0). The count is computed on the FULL unfiltered trial table before trial masking, so excluded trials still advance the counter. This is computed before the trial mask is applied.

ii.
```python
block_trial_number = compute_trial_number_in_block(trials["probabilityLeft"].to_numpy())
# First trial in block = 1, second = 2, etc.

# Applied after masking:
block_vals = block_trial_number[masked_keep["index"].to_numpy()]
```

iii. The AI documented in CONVERSION_NOTES.md Step 5 decision 8: "Compute trial number in block on the original unfiltered trial table."

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. From `trials.choice` in the trials table. Values +1 (left), -1 (right), 0 (no-go).

ii.
```python
def map_choice_to_binary(choice_values: np.ndarray) -> np.ndarray:
    mapped = np.full(choice_values.shape, -1, dtype=np.int16)
    mapped[choice_values == 1] = 0   # left
    mapped[choice_values == -1] = 1  # right
    return mapped
```

iii. The AI correctly identifies the IBL choice convention.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The raw choice values are mapped: +1 (left) -> 0, -1 (right) -> 1. Trials with choice == 0 are excluded by the trial mask. The mapped value is repeated across all 100 time bins.

ii.
```python
choice_vals = map_choice_to_binary(masked_keep["choice"].to_numpy(dtype=np.float64))
choice.append(np.full(N_BINS, choice_val, dtype=np.int16))
```

iii. Matches the decoder task specification.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. From `trials.probabilityLeft` in the trials table, which takes values 0.2, 0.5, or 0.8.

ii.
```python
def map_prior_to_categorical(prob_left: np.ndarray) -> np.ndarray:
    mapped = np.full(prob_left.shape, -1, dtype=np.int16)
    mapped[np.isclose(prob_left, 0.2)] = 0
    mapped[np.isclose(prob_left, 0.5)] = 1
    mapped[np.isclose(prob_left, 0.8)] = 2
    return mapped
```

iii. Matches the decoder task specification.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The three values are mapped to categorical integers: 0.2 -> 0, 0.5 -> 1, 0.8 -> 2. The mapped value is repeated across all 100 time bins.

ii.
```python
prior_vals = map_prior_to_categorical(masked_keep["probabilityLeft"].to_numpy(dtype=np.float64))
prior.append(np.full(N_BINS, prior_val, dtype=np.int16))
```

iii. Follows the instructions exactly.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. From `_ibl_wheel.position.npy` and `_ibl_wheel.timestamps.npy`. The raw position is interpolated to 1000 Hz, then velocity is computed with a Butterworth filter, and speed is the absolute value.

ii.
```python
def load_wheel_speed(session_path: Path) -> tuple[np.ndarray, np.ndarray]:
    pos = np.asarray(np.load(wheel_pos_file), dtype=np.float64)
    ts = np.asarray(np.load(wheel_ts_file), dtype=np.float64)
    pos_interp, ts_interp = interpolate_position(ts, pos, freq=1000)
    vel, _ = velocity_filtered(pos_interp, 1000)
    return ts_interp, np.abs(vel)
```

iii. The AI reuses wheel processing functions from the bundled ibllib code.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The wheel position is interpolated to 1000 Hz, filtered with `velocity_filtered` to get velocity, and the absolute value gives speed. This speed is then interpolated onto the trial time grid using `scipy.interpolate.interp1d` with linear interpolation. Finally, it is discretized into 3 bins using GLOBAL tertile edges computed across ALL sessions.

ii.
```python
# Global tertile computation:
wheel_edges = compute_tertile_edges(
    trial_values for session in processed for trial_values in session.wheel_cont
)

def compute_tertile_edges(values):
    concat = np.concatenate(flat_values)
    q1, q2 = np.quantile(concat, [1 / 3, 2 / 3])
    return float(q1), float(q2)

def discretize_three_bins(values, edges):
    return np.digitize(values, bins=np.array(edges), right=False)
```

iii. The AI chose global tertiles for discretization rather than per-session percentiles.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. The wheel speed is discretized into 3 bins (low/medium/high) using dataset-wide tertile edges. The edges are the 33rd and 67th percentiles computed across ALL valid wheel speed values from ALL sessions.

ii.
```python
wheel_edges = compute_tertile_edges(
    trial_values for session in processed for trial_values in session.wheel_cont
)
discretize_three_bins(wheel_cont, wheel_edges)
```

iii. The AI documented in CONVERSION_NOTES.md Step 5 decision 7: "Discretize wheel speed and whisker motion energy using global tertiles."

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The wheel speed trace is interpolated onto the same time grid as the behavior data using `interp1d`. The interpolation grid uses right bin edges: `np.linspace(window[0] + binsize, window[1], n_bins)`.

ii.
```python
def interpolate_behavior_trials(...):
    x_rel = np.linspace(window[0] + binsize, window[1], n_bins)
    interp = interp1d(ts, vals, kind="linear", fill_value="extrapolate")
    outputs.append(interp(align_time + x_rel).astype(np.float32))
```

iii. Uses the same time grid as the time input, ensuring consistency between behavior outputs and the time input.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. From `leftCamera.ROIMotionEnergy.npy` (preferred) or `rightCamera.ROIMotionEnergy.npy` (fallback), with corresponding camera times files.

ii.
```python
def load_whisker_motion_energy(session_path: Path):
    for camera in ("left", "right"):
        me_file = pick_one_file(session_path / "alf", f"{camera}Camera.ROIMotionEnergy.npy")
        times_file = pick_one_file(session_path / "alf", f"*{camera}Camera.times.npy")
```

iii. Left camera preferred, right as fallback, matching the reference code.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The raw motion energy trace is loaded directly (no additional filtering). It is interpolated onto the trial time grid using `interp1d` with linear interpolation. Then it is discretized into 3 bins using GLOBAL tertile edges across all sessions.

ii.
```python
whisk_trials, whisk_mask = interpolate_behavior_trials(whisk_times, whisk_values, align_times)

whisker_edges = compute_tertile_edges(
    trial_values for session in processed for trial_values in session.whisker_cont
)
discretize_three_bins(whisk_cont, whisker_edges)
```

iii. Same approach as wheel speed.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Same as wheel speed: dataset-wide tertile edges at the 33rd and 67th percentiles across all sessions.

ii. Same as 7-c, but with `whisker_edges`.

iii. Uses global tertiles for consistency across sessions.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Same interpolation approach as wheel speed: `interp1d` onto the right-edge time grid.

ii. Same as 7-d, but using whisker times and values.

iii. Same alignment strategy as all behavior data.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Several levels of handling: (1) Sessions missing required files (wheel, whisker, spikes) are skipped. (2) If whisker motion energy array length doesn't match camera times, both are truncated to the shorter length. (3) Trials with missing required event columns are excluded by the trial mask. (4) Trials where behavior data doesn't cover the window are excluded. (5) Trials with all-zero neural activity are excluded. (6) Sessions with fewer than 2 surviving trials are skipped.

ii.
```python
# Length mismatch handling:
if len(values) != len(times):
    n = min(len(values), len(times))
    values = values[:n]
    times = times[:n]

# All-zero neural filter:
neural_mask = np.array([np.any(trial) for trial in neural_trials], dtype=bool)

# Session skip:
if combined_mask.sum() < 2:
    print(f"[skip] {spec.eid}: fewer than 2 trials after behavior/neural alignment")
    return None
```

iii. The AI documented these handling strategies across Steps 9 and 10 of CONVERSION_NOTES.md.

## 10-a. What are the most time-consuming steps of the code?

i. The AI identified spike loading and binning as the main bottleneck. Spike arrays are memory-mapped (`mmap_mode="r"`) to avoid full array loads. The AI also mentions that threaded parallelism was added to reduce wall time.

ii.
```python
spikes_times = np.load(spikes_times_file, mmap_mode="r")
spikes_clusters = np.load(spikes_clusters_file, mmap_mode="r")
```

iii. CONVERSION_NOTES.md Step 6 documents spike array loading as the hot path.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The `compute_trial_number_in_block` function uses a Python for-loop over all trials. The `interpolate_behavior_trials` function loops per trial to interpolate behavior. The `bin_spikes_for_trials` function loops per trial to bin spikes.

ii.
```python
# Trial number in block loop:
for i in range(1, len(prob_left)):
    if prob_left[i] == prob_left[i - 1]:
        count += 1
    else:
        count = 1
    counters[i] = count

# Behavior interpolation loop:
for i, align_time in enumerate(align_times):
    interp = interp1d(ts, vals, kind="linear", fill_value="extrapolate")
    outputs.append(interp(align_time + x_rel).astype(np.float32))

# Spike binning loop:
for idx0, idx1, (start, end) in zip(idx_starts, idx_ends, intervals):
    counts, _, cluster_idx = bincount2D(...)
```

iii. The AI addressed this partially through `bincount2D` for spike binning but retained per-trial loops for interpolation and spike binning.

## 10-c. What processing does the code repeat multiple times?

i. The `BrainRegions()` object is created once per worker call (`process_session_worker` creates a new `BrainRegions()` for every session). This involves loading atlas data from disk each time.

ii.
```python
def process_session_worker(spec: SessionSpec) -> ProcessedSession | None:
    return process_session(spec, BrainRegions())
```

iii. Not explicitly documented by the AI.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI creates a new `interp1d` interpolator object per trial for behavior data, which is somewhat wasteful. The AI also stores continuous wheel and whisker data separately before discretizing them in the final assembly step, meaning the continuous data is held in memory across all sessions before discretization.

ii.
```python
# Stores continuous data per session:
wheel_cont=[np.asarray(x, dtype=np.float32) for x in wheel_keep],
whisker_cont=[np.asarray(x, dtype=np.float32) for x in whisk_keep],

# Then discretizes later in build_data_dict:
discretize_three_bins(wheel_cont, wheel_edges)
```

iii. This is a consequence of computing global tertile edges, which requires seeing all data before discretizing.
