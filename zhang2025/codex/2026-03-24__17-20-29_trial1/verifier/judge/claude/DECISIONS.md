# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads session metadata from a parquet manifest file (`sessions.pqt`) found in the local ONE cache directory structure. It constructs filesystem paths from the manifest fields (lab, subject, date, session number) and reads data files directly from disk using `np.load` and `pd.read_parquet`, rather than using the ONE API. A `pick_one_file` helper resolves versioned ALF directories by selecting the latest revision.

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

iii. The AI chose to bypass the ONE API and read files directly from the local cache. This was documented in CONVERSION_NOTES.md where it noted the data structure of the local ONE cache and built path resolution logic to handle versioned ALF directories.

## 1-b. How are the data split into subjects?

i. The subject name is extracted from the manifest row for each session. A sorted list of unique subjects is built from the processed sessions, and `subject_idx` maps each session to its index in that list.

ii.
```python
subjects = sorted({s.subject for s in sessions})
subject_to_idx = {s: i for i, s in enumerate(subjects)}
"subject_idx": np.array([subject_to_idx[s.subject] for s in sessions], dtype=np.int16),
```

iii. The subject comes from the manifest's `subject` field, which identifies the mouse.

## 1-c. How are the data split into sessions?

i. Each row in the manifest represents one session, identified by its `eid`. The AI iterates over manifest rows to construct `SessionSpec` objects, each representing one session.

ii.
```python
for eid, row in manifest.iterrows():
    session_path = (DATA_ROOT / row["lab"] / "Subjects" / row["subject"]
                    / str(row["date"]) / f"{int(row['number']):03d}")
    ...
    specs.append(SessionSpec(eid=eid, lab=str(row["lab"]), subject=str(row["subject"]), ...))
```

iii. The manifest already has one row per session, so no splitting is needed.

## 1-d. How are the data split into trials?

i. The trials table (`_ibl_trials.table.pqt`) has one row per trial. The AI reads this parquet file for each session and uses it directly.

ii.
```python
def load_trials_table(session_path: Path) -> pd.DataFrame:
    trial_file = pick_one_file(session_path / "alf", "_ibl_trials.table.pqt")
    ...
    return pd.read_parquet(trial_file)
```

iii. The trials table already contains one row per trial, so no further splitting is needed.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered based on: (1) reaction time between 0.08 and 2.0 s, (2) trial duration `feedback_times - goCue_times <= 10 s`, (3) `choice != 0`, (4) required columns not NaN (`stimOn_times`, `choice`, `feedback_times`, `probabilityLeft`, `firstMovement_times`, `feedbackType`). Additionally, after behavior interpolation, trials where the wheel or whisker data doesn't cover the trial window are dropped, and trials with all-zero neural activity are dropped.

ii.
```python
def compute_trial_mask(trials: pd.DataFrame) -> np.ndarray:
    required = ["stimOn_times", "choice", "feedback_times", "probabilityLeft",
                "firstMovement_times", "feedbackType"]
    mask = np.ones(len(trials), dtype=bool)
    rt = trials["firstMovement_times"] - trials["stimOn_times"]
    mask &= rt >= TRIAL_MASK_RT[0]
    mask &= rt <= TRIAL_MASK_RT[1]
    mask &= (trials["feedback_times"] - trials["goCue_times"]) <= MAX_TRIAL_LEN
    mask &= trials["choice"] != 0
    for col in required:
        mask &= trials[col].notna().to_numpy()
    return mask
```

```python
neural_mask = np.array([np.any(trial) for trial in neural_trials], dtype=bool)
combined_mask = wheel_mask & whisk_mask & neural_mask
```

iii. CONVERSION_NOTES.md documents using the reference code trial mask from `load_trials_and_mask()` which includes RT range, no-choice removal, max trial duration, and missing event exclusions. The all-zero neural trial filter was added after Step 10 review when verification warnings appeared.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `spikes.times.npy` and `spikes.clusters.npy` from each probe's pykilosort directory, along with `clusters.metrics.pqt` for QC filtering, `clusters.channels.npy` and `channels.brainLocationIds_ccf_2017.npy` for region mapping.

ii.
```python
spikes_times_file = pick_one_file(pykilo_path, "spikes.times.npy")
spikes_clusters_file = pick_one_file(pykilo_path, "spikes.clusters.npy")
metrics_file = pick_one_file(pykilo_path, "clusters.metrics.pqt")
```

iii. The AI identified these files in Step 2 (Dataset Exploration) as the spike sorting outputs under `alf/probeXX/pykilosort/`.

## 2-b. How is the `neural` data processed?

i. Spikes are binned into 20 ms bins over the [-0.5, 1.5] s window using `bincount2D`. The result is stored as raw spike counts in float16, NOT converted to firing rate. When a session has multiple probes, clusters are merged with an offset so they are numbered continuously.

ii.
```python
trial_counts = np.zeros((n_clusters, n_bins), dtype=np.float16)
if idx1 > idx0:
    counts, _, cluster_idx = bincount2D(
        spike_times[idx0:idx1], spike_clusters[idx0:idx1],
        xbin=binsize, xlim=[start, end])
    if counts.size:
        counts = counts[:, :n_bins]
        trial_counts[cluster_idx, : counts.shape[1]] = counts.astype(np.float16)
```

```python
data["neural"].append([trial.astype(np.float16) for trial in session.neural])
```

iii. The AI's CONVERSION_NOTES.md states neural values are "raw binned spike counts, not standardized z-scores." However, the reference code divides by bin width to get firing rates in Hz. The AI also uses float16 rather than float32.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only clusters with `clusters.metrics.label >= 1` are kept. This is the "well-isolated neuron" criterion from the data paper (75,708 of 621,733 units).

ii.
```python
GOOD_CLUSTER_LABEL = 1
...
metrics = pd.read_parquet(metrics_file, columns=["label"])
cluster_labels = metrics["label"].to_numpy()
good_mask = cluster_labels >= label_threshold
selected_cluster_ids = np.flatnonzero(good_mask)
...
spike_keep = good_mask[spikes_clusters]
```

iii. CONVERSION_NOTES.md documents this as matching the data paper's well-isolated neuron count.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial's spikes are binned relative to the stimulus onset time (`stimOn_times`). The window is [-0.5, 1.5] s. The `bincount2D` function is called with `xlim=[stimOn + window[0], stimOn + window[1]]`.

ii.
```python
align_times = masked_trials["stimOn_times"].to_numpy(dtype=np.float64)
neural_trials = bin_spikes_for_trials(spike_times, spike_clusters, n_clusters=n_clusters_good, align_times=align_times)
...
intervals = np.c_[align_times + window[0], align_times + window[1]]
```

iii. Documented in CONVERSION_NOTES.md Step 4: "use the executable repository settings of 2 s windows, 20 ms bins, and stimulus-onset alignment."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 20 ms bins (0.02 s), producing 100 time bins per trial over the 2 s window. No rebinning is applied.

ii.
```python
BINSIZE_S = 0.02
N_BINS = int(round((TIME_WINDOW[1] - TIME_WINDOW[0]) / BINSIZE_S))
```

iii. Matches the reference code's `binsize=0.02`.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is a constructed time grid based on the window parameters, not derived from any raw data variable directly. It represents the time axis of the bins.

ii.
```python
def make_time_input() -> np.ndarray:
    return np.linspace(TIME_WINDOW[0] + BINSIZE_S, TIME_WINDOW[1], N_BINS, dtype=np.float32)
```

iii. This is a computed variable representing the time axis of the trial.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The AI uses `np.linspace(-0.48, 1.5, 100)`, which produces bin right edges rather than bin centers. This gives values starting at -0.48, -0.4598..., etc.

ii.
```python
x_rel = np.linspace(window[0] + binsize, window[1], n_bins)
# = np.linspace(-0.48, 1.5, 100)
```

iii. The AI's CONVERSION_NOTES.md confirms the time grid as `[-0.48, ..., 1.50]`. This differs from the reference solution which uses bin centers: `EDGES[:-1] + BIN/2 = [-0.49, -0.47, ..., 1.49]`.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The time input uses the same bin grid as the behavior interpolation (`np.linspace(-0.48, 1.5, 100)`). However, note that the neural spike binning uses `bincount2D` with `xlim=[start, end]` which defines a different bin grid (bin edges from -0.5 to 1.5 in 0.02 steps). The time input grid is aligned with the behavior but may not be perfectly aligned with the neural bin grid.

ii.
```python
# Time input and behavior use:
x_rel = np.linspace(window[0] + binsize, window[1], n_bins)  # right edges

# Neural binning uses bincount2D with xlim=[start, end]
# which creates its own bin grid
```

iii. The AI intended these to be aligned but the time input represents right bin edges while the neural bins from `bincount2D` have their own edge/center convention.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. From `probabilityLeft` in the trials table. Block boundaries are detected when `probabilityLeft` changes value.

ii.
```python
block_trial_number = compute_trial_number_in_block(trials["probabilityLeft"].to_numpy())
```

iii. Documented in CONVERSION_NOTES.md as "Block counter resets whenever `probabilityLeft` changes."

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The AI counts from 1 (not 0). The first trial in a block gets count=1, and it increments for each subsequent trial with the same `probabilityLeft`. The count is computed on the full unfiltered trial table before any trial masking.

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

iii. CONVERSION_NOTES.md: "Compute trial number in block on the original unfiltered trial table: Excluded trials should not renumber the latent block progression."

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. From `trials.choice`, which is +1 (left), -1 (right), or 0 (no-go).

ii.
```python
choice_vals = map_choice_to_binary(masked_keep["choice"].to_numpy(dtype=np.float64))
```

iii. Matches the standard IBL convention.

## 5-b. What processing is involved in computing `output` *Choice*?

i. Mapped to binary: `choice == 1 -> 0` (left), `choice == -1 -> 1` (right). Trials with `choice == 0` are excluded. The value is repeated across all 100 time bins.

ii.
```python
def map_choice_to_binary(choice_values: np.ndarray) -> np.ndarray:
    mapped = np.full(choice_values.shape, -1, dtype=np.int16)
    mapped[choice_values == 1] = 0   # left
    mapped[choice_values == -1] = 1  # right
    if np.any(mapped < 0):
        raise ValueError("Unexpected choice values after masking")
    return mapped
```

iii. Follows the instruction specification: left = 0, right = 1.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. From `trials.probabilityLeft`, which takes values 0.2, 0.5, and 0.8.

ii.
```python
prior_vals = map_prior_to_categorical(masked_keep["probabilityLeft"].to_numpy(dtype=np.float64))
```

iii. Standard IBL block probability variable.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Mapped to categorical: 0.2 -> 0, 0.5 -> 1, 0.8 -> 2, using `np.isclose` for floating-point comparison. The value is repeated across all 100 time bins.

ii.
```python
def map_prior_to_categorical(prob_left: np.ndarray) -> np.ndarray:
    mapped = np.full(prob_left.shape, -1, dtype=np.int16)
    mapped[np.isclose(prob_left, 0.2)] = 0
    mapped[np.isclose(prob_left, 0.5)] = 1
    mapped[np.isclose(prob_left, 0.8)] = 2
    ...
    return mapped
```

iii. Follows the instruction specification.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. From `_ibl_wheel.position.npy` and `_ibl_wheel.timestamps.npy`. The position is interpolated to 1000 Hz and differentiated with a Butterworth filter to get velocity, then the absolute value gives speed.

ii.
```python
def load_wheel_speed(session_path: Path) -> tuple[np.ndarray, np.ndarray]:
    ...
    pos_interp, ts_interp = interpolate_position(ts, pos, freq=1000)
    vel, _ = velocity_filtered(pos_interp, 1000)
    return ts_interp, np.abs(vel)
```

iii. Uses the bundled `brainbox.behavior.wheel` functions from the reference code directory, producing the same wheel processing pipeline.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Wheel position is interpolated to 1000 Hz, velocity is computed with a Butterworth low-pass filter, absolute value gives speed. The speed trace is then interpolated onto the trial time grid using `scipy.interpolate.interp1d` (linear interpolation with extrapolation). Finally it is discretized into 3 bins.

ii.
```python
pos_interp, ts_interp = interpolate_position(ts, pos, freq=1000)
vel, _ = velocity_filtered(pos_interp, 1000)
return ts_interp, np.abs(vel)
...
interp = interp1d(ts, vals, kind="linear", fill_value="extrapolate")
outputs.append(interp(align_time + x_rel).astype(np.float32))
```

iii. The wheel processing follows the reference code's `SessionLoader` logic via the same underlying functions.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. The AI uses **global tertiles** across all sessions: it collects all continuous wheel speed values from all sessions, computes the 1/3 and 2/3 quantiles, then discretizes all values using those two edges. This differs from the reference which uses per-session percentiles.

ii.
```python
wheel_edges = compute_tertile_edges(
    trial_values for session in processed for trial_values in session.wheel_cont
)

def compute_tertile_edges(values: Iterable[np.ndarray]) -> tuple[float, float]:
    ...
    concat = np.concatenate(flat_values)
    q1, q2 = np.quantile(concat, [1 / 3, 2 / 3])
    ...

def discretize_three_bins(values: np.ndarray, edges: tuple[float, float]) -> np.ndarray:
    return np.digitize(values, bins=np.array(edges, dtype=np.float32), right=False).astype(np.int16)
```

iii. CONVERSION_NOTES.md: "Discretize wheel speed and whisker motion energy using global tertiles: Global edges preserve a common categorical meaning across sessions."

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The wheel speed is interpolated onto the same time grid as the behavior: `np.linspace(-0.48, 1.5, 100)`, which are right bin edges relative to stimulus onset. This is the same grid used for the time input.

ii.
```python
x_rel = np.linspace(window[0] + binsize, window[1], n_bins)
interp = interp1d(ts, vals, kind="linear", fill_value="extrapolate")
outputs.append(interp(align_time + x_rel).astype(np.float32))
```

iii. Aligned to the same stimulus onset event and time grid as all other variables.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. From `{left,right}Camera.ROIMotionEnergy.npy` and `_ibl_{left,right}Camera.times.npy`. Left camera is preferred, with right as fallback.

ii.
```python
def load_whisker_motion_energy(session_path: Path) -> tuple[np.ndarray, np.ndarray, str]:
    for camera in ("left", "right"):
        me_file = pick_one_file(session_path / "alf", f"{camera}Camera.ROIMotionEnergy.npy")
        times_file = pick_one_file(session_path / "alf", f"*{camera}Camera.times.npy")
        ...
        return times, values, camera
```

iii. Left-then-right fallback matches the reference code.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The raw motion energy values are loaded as-is (no filtering or normalization). They are interpolated onto the trial time grid using `scipy.interpolate.interp1d`. Then discretized into 3 bins using global tertiles.

ii.
```python
whisk_trials, whisk_mask = interpolate_behavior_trials(whisk_times, whisk_values, align_times)
```

iii. No additional processing beyond interpolation and discretization.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Same as wheel speed: global tertiles across all sessions, using the 1/3 and 2/3 quantiles of all whisker motion energy values.

ii.
```python
whisker_edges = compute_tertile_edges(
    trial_values for session in processed for trial_values in session.whisker_cont
)
```

iii. CONVERSION_NOTES.md: "Bins will be dataset-wide tertiles over valid aligned samples."

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Same as wheel speed: interpolated onto `np.linspace(-0.48, 1.5, 100)` relative to stimulus onset.

ii.
```python
interp = interp1d(ts, vals, kind="linear", fill_value="extrapolate")
outputs.append(interp(align_time + x_rel).astype(np.float32))
```

iii. Aligned to stimulus onset, same grid as all other variables.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Multiple layers: (1) Sessions missing required files (wheel, whisker, spikes) are skipped. (2) Trials with NaN in required columns are excluded. (3) Trials where behavior doesn't cover the full window are excluded. (4) Trials with all-zero neural activity are excluded. (5) Sessions with fewer than 2 surviving trials are skipped. (6) When whisker times and values have mismatched lengths, the shorter length is used.

ii.
```python
if len(values) != len(times):
    n = min(len(values), len(times))
    values = values[:n]
    times = times[:n]
...
neural_mask = np.array([np.any(trial) for trial in neural_trials], dtype=bool)
combined_mask = wheel_mask & whisk_mask & neural_mask
if combined_mask.sum() < 2:
    print(f"[skip] {spec.eid}: fewer than 2 trials after behavior/neural alignment")
    return None
```

iii. CONVERSION_NOTES.md documents the all-zero neural trial filter as added after Step 10 review.

## 10-a. What are the most time-consuming steps of the code?

i. Loading spike sorting data from disk (memory-mapped reads of large spike arrays) and binning spikes into trial-aligned bins. The CONVERSION_NOTES.md reports ~13.1 s mean per session.

ii.
```python
spikes_times = np.load(spikes_times_file, mmap_mode="r")
spikes_clusters = np.load(spikes_clusters_file, mmap_mode="r")
```

iii. CONVERSION_NOTES.md: "Trial-by-trial spike binning is the main hot path."

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The spike binning loop iterates per trial, calling `bincount2D` for each trial individually. The behavior interpolation loop also iterates per trial, creating a separate `interp1d` object for each trial.

ii.
```python
for idx0, idx1, (start, end) in zip(idx_starts, idx_ends, intervals):
    trial_counts = np.zeros((n_clusters, n_bins), dtype=np.float16)
    ...

for i, align_time in enumerate(align_times):
    ...
    interp = interp1d(ts, vals, kind="linear", fill_value="extrapolate")
    outputs.append(interp(align_time + x_rel).astype(np.float32))
```

iii. Both loops could potentially be vectorized, though the per-trial slicing makes this nontrivial.

## 10-c. What processing does the code repeat multiple times?

i. A `BrainRegions()` object is created once per session in the threaded worker (`process_session_worker`), which is redundant. The behavior interpolation function creates a new `interp1d` object per trial rather than reusing a single interpolator.

ii.
```python
def process_session_worker(spec: SessionSpec) -> ProcessedSession | None:
    return process_session(spec, BrainRegions())
```

iii. Not explicitly documented as a concern.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code stores continuous wheel and whisker values (`wheel_cont`, `whisker_cont`) in `ProcessedSession` and only discretizes them later in `build_data_dict`. This means the continuous values are held in memory across all sessions before being discretized. The code also computes `n_clusters_total` (total clusters before QC) which is only used for logging.

ii.
```python
wheel_cont=[np.asarray(x, dtype=np.float32) for x in wheel_keep],
whisker_cont=[np.asarray(x, dtype=np.float32) for x in whisk_keep],
```

iii. The continuous values are needed temporarily for computing global tertile edges, but they consume memory across all sessions.
