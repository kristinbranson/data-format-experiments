# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads session specs from the BWM release CSV (`bwm_release.csv`) rather than using `one.search()`. It creates a local ONE client in `mode="local"` only to resolve eids to filesystem paths via `one.eid2path()`. After that, all data files (trials, spikes, wheel, camera) are loaded directly from disk using `np.load()` and `pd.read_parquet()`, not through the ONE API loaders (`SessionLoader`, `SpikeSortingLoader`). When `DATALIMIT_SUBSET.csv` exists, sessions are restricted to the eids listed there.

ii.
```python
one = ONE(mode="local", cache_dir=CACHE_DIR, silent=True)
One.load_cache(one, CACHE_DIR / "Brainwidemap")

release = pd.read_csv(BWM_TABLE, index_col=0)
if SUBSET_TABLE.exists():
    subset = pd.read_csv(SUBSET_TABLE)
    selected = set(subset["eid"].astype(str))
    release = release[release["eid"].astype(str).isin(selected)]

for eid, rows in release.groupby("eid", sort=False):
    session_path = Path(one.eid2path(str(eid)))
    specs.append(SessionSpec(eid=str(eid), subject=str(rows.iloc[0]["subject"]),
                             session_path=session_path, probes=tuple(rows["probe_name"].astype(str))))
```

iii. The agent noted: "The staged cache is the full Brain-Wide Map release, not a toy subset." It used the release CSV as its session index, stating the "methods paper's reported 433 usable sessions is therefore a downstream availability/alignment subset, not a different release."

## 1-b. How are the data split into subjects?

i. The subject name is extracted from the `subject` column of the BWM release CSV per session. During assembly, subjects are globally sorted alphabetically and mapped to integer indices.

ii.
```python
subjects = sorted({payload["subject"] for payload in payloads})
subject_to_idx = {subject: i for i, subject in enumerate(subjects)}
```

iii. The release table already contains a subject column, so no parsing or derivation is needed.

## 1-c. How are the data split into sessions?

i. Sessions are defined by grouping the BWM release table by `eid`. Each unique `eid` forms one session. The probes within a session are listed in the release table and stored in the `SessionSpec`.

ii.
```python
for eid, rows in release.groupby("eid", sort=False):
    session_path = Path(one.eid2path(str(eid)))
    specs.append(SessionSpec(eid=str(eid), subject=str(rows.iloc[0]["subject"]),
                             session_path=session_path, probes=tuple(rows["probe_name"].astype(str))))
```

iii. Each eid in the release table is already a unique session identifier.

## 1-d. How are the data split into trials?

i. The trials table (`_ibl_trials.table.pqt`) has one row per trial. The split is inherent in the data.

ii.
```python
trials = pd.read_parquet(table)
```

iii. No decision needed; the trials table is already one row per trial.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies several filters in `_paper_trial_mask()`: (1) Required columns (`stimOn_times`, `choice`, `feedback_times`, `probabilityLeft`, `firstMovement_times`, `feedbackType`) must be non-NaN; (2) `choice != 0` (excludes no-go trials); (3) Reaction time between 0.08 s and 2.0 s; (4) Trial duration (feedback_times - goCue_times) <= 10 s. Additionally, trials must have valid wheel and whisker camera coverage for the full [-0.5, 1.5) s window.

ii.
```python
def _paper_trial_mask(trials: pd.DataFrame) -> np.ndarray:
    reaction_time = trials["firstMovement_times"] - trials["stimOn_times"]
    trial_duration = trials["feedback_times"] - trials["goCue_times"]
    return (
        trials[needed].notna().all(axis=1).to_numpy()
        & (trials["choice"].to_numpy() != 0)
        & (reaction_time.to_numpy() >= 0.08)
        & (reaction_time.to_numpy() <= 2.0)
        & (trial_duration.to_numpy() <= 10.0)
    )
```
```python
keep = paper_mask & wheel_good & motion_good
```

iii. The agent stated it "applies the paper's trial exclusions (missing events, no-choice, 80 ms-2 s first-movement latency, and <=10 s trial duration)."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Two arrays per probe: `spikes.times.npy` and `spikes.clusters.npy`, loaded directly from disk. Cluster-to-channel mapping (`clusters.channels.npy`) and atlas IDs (`channels.brainLocationIds_ccf_2017.npy`) are used for brain region assignment.

ii.
```python
times_path, clusters_path, cluster_channels_path, atlas_ids_path = _spike_files(spec.session_path, probe)
times = np.load(times_path, mmap_mode="r")
clusters = np.load(clusters_path, mmap_mode="r")
```

iii. The agent loads spike times and cluster assignments directly from the pykilosort output files.

## 2-b. How is the `neural` data processed?

i. Spikes are counted into non-overlapping 20 ms bins over a 2 s window [-0.5, 1.5) s around stimulus onset, giving 100 bins per trial. Raw spike counts are stored (not converted to firing rates). When a session has multiple probes, their clusters are concatenated. No smoothing is applied.

ii.
```python
spike_bins = np.floor((np.asarray(times[lo:hi]) - interval_begins[trial]) / BIN_SIZE).astype(np.int64)
cluster_ids = np.asarray(clusters[lo:hi], dtype=np.int64)
flat = cluster_ids[valid] * N_BINS + spike_bins[valid]
binned[trial].flat[:] = np.bincount(flat, minlength=len(used) * N_BINS).astype(np.float32, copy=False)
```
```python
neural = np.concatenate(probe_bins, axis=1)
```

iii. The agent stated: "The reference code combines all probes within a session, uses 20 ms spike-count bins over a 2 s stimulus-aligned window."

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI does NOT filter clusters by quality label. ALL Kilosort 2.5 clusters are included, regardless of their QC label. The only filtering is the remapping step that handles rare missing cluster IDs. Brain regions are mapped using the Beryl atlas, but `void` or `root` regions are NOT excluded.

ii.
```python
used = np.unique(clusters)
# ... all clusters in the spike train are used
counts, used = _bin_probe(times_path, clusters_path, interval_begins, len(cluster_channels))
cluster_atlas_ids = channel_atlas_ids[cluster_channels[used]]
regions = atlas.id2acronym(cluster_atlas_ids, mapping="Beryl")
```

iii. The metadata states: `"neuron_filter": "all Kilosort clusters, matching the methods-paper decoder pipeline"`. The agent justified this by saying the reference decoder pipeline (`prepare_data` / `load_spiking_data`) does not apply a QC filter.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The trial window begins at `stimulus_times[trial] + OFF_START` (-0.5 s before stimulus onset). Spikes are binned relative to this interval start, so bin 0 corresponds to -0.5 s from stimulus onset.

ii.
```python
interval_begins = stimulus_times[keep] + OFF_START
spike_bins = np.floor((np.asarray(times[lo:hi]) - interval_begins[trial]) / BIN_SIZE).astype(np.int64)
```

iii. The agent aligns to stimulus onset as required by the instructions.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The bin size is 20 ms, producing 100 bins for the 2 s window. No rebinning or interpolation is applied to the neural data.

ii.
```python
OFF_START = -0.5
OFF_END = 1.5
BIN_SIZE = 0.02
N_BINS = 100
```

iii. The agent follows the reference code's `binsize: 0.02` setting.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is a synthetic variable constructed from the bin edges, not derived from any raw data variable. It represents the time coordinate of each bin relative to stimulus onset.

ii.
```python
relative_time = (OFF_START + np.arange(N_BINS, dtype=np.float32) * BIN_SIZE).astype(np.float32)
inputs[:, 0, :] = relative_time
```

iii. The agent constructs the time coordinate as the left edge of each 20 ms bin.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. No processing of raw data. The variable is defined as the left edge of each 20 ms bin: values from -0.5 to +1.48 in 0.02 s steps.

ii.
```python
relative_time = (OFF_START + np.arange(N_BINS, dtype=np.float32) * BIN_SIZE).astype(np.float32)
```

iii. The metadata notes: `"time_coordinate_convention": "left edge of each 20 ms neural bin"`.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The time coordinate uses the left edge of each bin, which corresponds to the start of the interval each neural bin covers. The neural spikes are binned with `np.floor((spike_time - interval_begin) / BIN_SIZE)`, so the bin index matches the left-edge convention.

ii.
```python
relative_time = (OFF_START + np.arange(N_BINS, dtype=np.float32) * BIN_SIZE).astype(np.float32)
# Neural bins use the same OFF_START and BIN_SIZE:
spike_bins = np.floor((np.asarray(times[lo:hi]) - interval_begins[trial]) / BIN_SIZE).astype(np.int64)
```

iii. Both share the same time grid defined by OFF_START and BIN_SIZE.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. From `probabilityLeft` in the trials table. A change in `probabilityLeft` between consecutive trials marks a new block boundary.

ii.
```python
def _trial_number_in_block(probability_left: np.ndarray) -> np.ndarray:
    out = np.zeros(len(probability_left), dtype=np.float32)
    for trial in range(1, len(probability_left)):
        out[trial] = (
            out[trial - 1] + 1
            if probability_left[trial] == probability_left[trial - 1]
            else 0
        )
    return out
```

iii. The trials table carries no block identifier, so blocks are recovered from consecutive `probabilityLeft` values.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The trial number is computed as the zero-based run-length position within each contiguous block of the same `probabilityLeft` value. The computation is done on the full unfiltered trial sequence before any quality filtering, so a dropped trial still advances the count. The per-trial scalar is then broadcast across all 100 time bins.

ii.
```python
trial_in_block = _trial_number_in_block(probability_left)
inputs[:, 1, :] = trial_in_block[keep, None]
```

iii. The metadata states: `"trial_number_in_block_convention": "zero-based run-length within the full unfiltered probabilityLeft sequence"`.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. From the `choice` column of the trials table. IBL uses +1 for left, -1 for right, and 0 for no response.

ii.
```python
raw_choice = trials["choice"].to_numpy()
choice = (raw_choice == -1).astype(np.int64)
```

iii. The agent noted the IBL convention: -1 is rightward, +1 is leftward.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The IBL choice values are remapped: left (+1) becomes 0, right (-1) becomes 1. No-response trials (choice=0) are excluded by the trial mask. The per-trial value is broadcast across all 100 time bins.

ii.
```python
choice = (raw_choice == -1).astype(np.int64)
outputs[:, 0, :] = choice[keep, None]
```

iii. The instructions specify left=0, right=1.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. From the `probabilityLeft` column of the trials table, which takes values 0.2, 0.5, or 0.8.

ii.
```python
probability_left = trials["probabilityLeft"].to_numpy(dtype=np.float64)
prior_lookup = {0.2: 0, 0.5: 1, 0.8: 2}
for value, label in prior_lookup.items():
    prior[np.isclose(probability_left, value)] = label
```

iii. The three values and mapping are specified in the instructions.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The three prior values are mapped to categorical labels: 0.2 -> 0, 0.5 -> 1, 0.8 -> 2. An error is raised if any retained trial has a value outside this set. The per-trial value is broadcast across all time bins.

ii.
```python
prior = np.full(len(trials), -1, dtype=np.int64)
for value, label in prior_lookup.items():
    prior[np.isclose(probability_left, value)] = label
if np.any(prior[keep] < 0):
    raise ValueError("probabilityLeft contains a value outside {0.2, 0.5, 0.8}")
outputs[:, 1, :] = prior[keep, None]
```

iii. N/A

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. From `_ibl_wheel.position.npy` and `_ibl_wheel.timestamps.npy`, loaded directly from disk.

ii.
```python
position = np.load(position_path)
timestamps = np.load(timestamps_path)
```

iii. The wheel position and timestamps are the raw inputs to velocity computation.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The wheel position is interpolated to 1 kHz, then velocity is computed with a 20 Hz Butterworth low-pass filter (order 8) via `velocity_filtered`. The absolute value gives speed. This is then interpolated onto 100 time bins per trial using `scipy.interpolate.interp1d` with linear interpolation and extrapolation.

ii.
```python
position_1khz, times_1khz = interpolate_position(timestamps, position, freq=1000)
velocity, _ = velocity_filtered(position_1khz, fs=1000, corner_frequency=20, order=8)
return _interpolate_trials(times_1khz, np.abs(velocity), stimulus_times)
```

iii. The agent uses the same brainbox functions as `SessionLoader.load_wheel()` does internally.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Within-session tertiles: the 1/3 and 2/3 quantiles of all wheel speed values (across all retained trials and time bins in the session) define two thresholds. `np.digitize` maps values to 0 (low), 1 (medium), or 2 (high).

ii.
```python
def _three_bins(values: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
    q1, q2 = np.quantile(values, [1/3, 2/3])
    labels = np.digitize(values, [q1, q2], right=False).astype(np.int64)
    return labels, (float(q1), float(q2))
```

iii. The agent justified: "The behavioral classes will use within-session tertiles pooled over retained time bins; that preserves three comparable low/medium/high states despite large camera- and rig-dependent scale differences."

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The wheel speed is interpolated to the same time bins as the neural data. The target times are computed as `np.linspace(begins + BIN_SIZE, ends, N_BINS)`, i.e. 100 evenly spaced points from `stim_onset + OFF_START + BIN_SIZE` to `stim_onset + OFF_END`.

ii.
```python
target_times = np.linspace(begins[trial] + BIN_SIZE, ends[trial], N_BINS, dtype=np.float64)
values[trial] = interp1d(times, vals, kind="linear", fill_value="extrapolate")(target_times)
```

iii. The wheel is on the same session clock as the spikes. Interpolation to the bin grid provides alignment.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. The ROI motion energy from the side camera: `leftCamera.ROIMotionEnergy.npy` (preferred) or `rightCamera.ROIMotionEnergy.npy` as fallback, with corresponding `_ibl_<side>Camera.times.npy`.

ii.
```python
for view in ("left", "right"):
    energy_path = _required_file(
        session_path.glob(f"alf/**/{view}Camera.ROIMotionEnergy.npy"), ...)
    times_path = _required_file(
        session_path.glob(f"alf/**/_ibl_{view}Camera.times.npy"), ...)
    energy = np.load(energy_path)
    times = np.load(times_path)
```

iii. Left camera is preferred with right as fallback, matching the reference code logic.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The released motion energy trace is used as-is with no additional filtering or normalization. Camera timestamps are adjusted when they are longer than the energy array (extra initial timestamps are trimmed from the start). The trace is interpolated onto the 100 bin time grid using `interp1d` with linear interpolation.

ii.
```python
if len(times) > len(energy):
    times = times[-len(energy):]
values, good = _interpolate_trials(times, energy, stimulus_times)
```

iii. The agent noted this timestamp adjustment matches `SessionLoader._check_video_timestamps`.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Same within-session tertile approach as wheel speed: 1/3 and 2/3 quantiles define thresholds, `np.digitize` maps to 0/1/2.

ii.
```python
motion_labels, motion_edges = _three_bins(motion)
```

iii. Same justification as wheel speed -- within-session tertiles for comparable classes.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Same `_interpolate_trials()` method as wheel speed -- linear interpolation to 100 time bins per trial aligned to [-0.5, 1.5) s from stimulus onset.

ii.
```python
values, good = _interpolate_trials(times, energy, stimulus_times)
```

iii. The camera frame times are on the same session clock as the neural data.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Multiple levels: (1) `_paper_trial_mask` drops trials with NaN in required columns; (2) `_interpolate_trials` marks trials as bad if behavioral stream doesn't cover the full window or interpolated values are non-finite; (3) Camera timestamps are adjusted when longer than energy array; (4) Sessions with fewer than 2 valid trials are skipped; (5) Failed sessions are caught via exception handling and logged to `failures.json`.

ii.
```python
# Camera timestamp adjustment
if len(times) > len(energy):
    times = times[-len(energy):]

# Session-level skip
if len(keep_idx) < 2:
    raise ValueError(f"only {len(keep_idx)} aligned valid trials")

# Parallel error handling
except Exception as exc:
    failures[spec.eid] = repr(exc)
```

iii. The agent noted: "I'm retaining [all-zero neural trials] because the supplied reference binning code produces zero-filled windows and specifies no neural-coverage exclusion."

## 10-a. What are the most time-consuming steps of the code?

i. Loading and binning spike data from disk. The agent uses memory-mapped spike loading (`mmap_mode="r"`) to avoid loading full spike trains into memory. Per-session checkpointing avoids re-processing on restart.

ii.
```python
times = np.load(times_path, mmap_mode="r")
clusters = np.load(clusters_path, mmap_mode="r")
```

iii. The agent stated: "about 108 GB resident, 16 conversion workers active."

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Two loops: (1) `_trial_number_in_block` iterates trial by trial to compute run-length positions -- this could be vectorized with cumsum-based block detection (as the reference does). (2) `_interpolate_trials` loops over trials for interpolation -- this could potentially be vectorized but the per-trial interpolation windows differ.

ii.
```python
# Trial-by-trial loop for block number
for trial in range(1, len(probability_left)):
    out[trial] = out[trial - 1] + 1 if probability_left[trial] == probability_left[trial - 1] else 0

# Trial-by-trial loop for interpolation
for trial, (beg_idx, end_idx) in enumerate(zip(idx_beg, idx_end)):
    ...
```

iii. The loop for trial_number_in_block is a simple sequential dependency that could be vectorized with pandas cumsum/cumcount. The interpolation loop is harder to vectorize due to variable-length windows.

## 10-c. What processing does the code repeat multiple times?

i. The `_interpolate_trials` function is called separately for wheel speed and whisker motion energy, each time performing the same pattern of `searchsorted` + per-trial interpolation. The pattern is the same but applied to different data streams, so it's a reusable function rather than truly redundant computation.

ii.
```python
wheel, wheel_good = _load_wheel_speed(spec.session_path, stimulus_times)
motion, motion_good, camera_view = _load_whisker_energy(spec.session_path, stimulus_times)
```

iii. N/A

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes and stores detailed session info metadata (tertile edges, elapsed time, trial counts, etc.) that is not used by the decoder. It also writes a `failures.json` file. Additionally, the checkpointing system stores intermediate per-session pickles that are deleted after assembly.

ii.
```python
"session_info": {
    "wheel_speed_tertile_edges": list(wheel_edges),
    "whisker_motion_energy_tertile_edges": list(motion_edges),
    "elapsed_seconds": float(time.time() - started),
    ...
}
```

iii. This extra metadata is informational and doesn't affect decoder performance.
