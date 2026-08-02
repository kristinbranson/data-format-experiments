# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from local ONE cache files. It reads a session manifest parquet file (`sessions.pqt`) from one of three possible cache directories (`2025_Q3_IBL_et_al_BWM`, `Brainwidemap`, `2022_Q4_IBL_et_al_BWM`), iterating over each row to construct file paths to each session's data directory. For each session, it loads the trials table, spike sorting outputs (per probe), wheel data, and whisker motion energy files directly from the filesystem using numpy/pandas, rather than through the ONE API.

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

iii. The AI justified this by noting that the data was available locally in a ONE cache structure, so it could bypass the ONE API and load files directly. It used the `2025_Q3_IBL_et_al_BWM/sessions.pqt` manifest containing 459 sessions matching the data paper's release count. Sessions without local data directories are skipped.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified from the `subject` column of the session manifest. Each session has a subject field, and the final dataset builds a sorted list of unique subjects. Each session is mapped to a subject index.

ii.
```python
# In resolve_session_specs:
subject=str(row["subject"]),

# In build_data_dict:
subjects = sorted({s.subject for s in sessions})
subject_to_idx = {s: i for i, s in enumerate(subjects)}
data = {
    'subjects': subjects,
    'subject_idx': np.array([subject_to_idx[s.subject] for s in sessions], dtype=np.int16),
}
```

iii. The AI followed the session manifest metadata, which associates each session (eid) with a subject name. This matches how the reference code identifies subjects via `bwm_df.subject`.

## 1-c. How are the data split into sessions?

i. Each row in the session manifest corresponds to a unique session identified by its `eid`. The AI processes each session independently, loading its neural, trial, and behavioral data from the session's directory path. Sessions that lack required data files (spikes, wheel, whisker) are skipped.

ii.
```python
for spec in specs:
    session = process_session_worker(spec)
    if session is not None:
        processed.append(session)
```

iii. The session structure matches the reference code's per-eid processing loop in `0_data_caching.py`. Sessions missing required modalities (6 missing wheel, 14 missing whisker, 1 with too few trials) are skipped, resulting in 438 sessions from 459 in the manifest.

## 1-d. How are the data split into trials?

i. For each session, the AI loads the trial table parquet file (`_ibl_trials.table.pqt`) from the session's `alf/` directory. Each row in the table is a trial. Trials are then filtered by a quality mask before being processed.

ii.
```python
def load_trials_table(session_path: Path) -> pd.DataFrame:
    trial_file = pick_one_file(session_path / "alf", "_ibl_trials.table.pqt")
    if trial_file is None:
        raise FileNotFoundError(f"Missing trial table for {session_path}")
    return pd.read_parquet(trial_file)
```

iii. The AI loads trial data from the standard IBL ONE cache file structure. The reference code loads trials via `SessionLoader.load_trials()`, which accesses the same underlying data.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies a trial mask with the following criteria:
- Required columns must not be NaN: `stimOn_times`, `choice`, `feedback_times`, `probabilityLeft`, `firstMovement_times`, `feedbackType`
- Reaction time (firstMovement_times - stimOn_times) must be in [0.08, 2.0] seconds
- choice must not equal 0
- Trial duration (feedback_times - goCue_times) must be <= 10.0 seconds
- Additionally, after behavioral alignment, trials with missing wheel/whisker interpolation or all-zero neural windows are excluded.

ii.
```python
def compute_trial_mask(trials: pd.DataFrame) -> np.ndarray:
    required = [
        "stimOn_times", "choice", "feedback_times",
        "probabilityLeft", "firstMovement_times", "feedbackType",
    ]
    mask = np.ones(len(trials), dtype=bool)
    rt = trials["firstMovement_times"] - trials["stimOn_times"]
    mask &= rt >= TRIAL_MASK_RT[0]  # 0.08
    mask &= rt <= TRIAL_MASK_RT[1]  # 2.0
    mask &= (trials["feedback_times"] - trials["goCue_times"]) <= MAX_TRIAL_LEN  # 10.0
    mask &= trials["choice"] != 0
    for col in required:
        mask &= trials[col].notna().to_numpy()
    return mask
```

iii. The AI documented that these filters match the reference code's `load_trials_and_mask()` defaults: `min_rt=0.08`, `max_rt=2.0`, `exclude_nochoice=True`, default NaN exclusion list, and `max_trial_len=10.0` from `prepare_data()`.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from `spikes.times.npy` and `spikes.clusters.npy` files in each probe's `pykilosort/` directory. Cluster quality labels come from `clusters.metrics.pqt`, channel-to-region mappings from `clusters.channels.npy` and `channels.brainLocationIds_ccf_2017.npy`.

ii.
```python
spikes_times_file = pick_one_file(pykilo_path, "spikes.times.npy")
spikes_clusters_file = pick_one_file(pykilo_path, "spikes.clusters.npy")
metrics_file = pick_one_file(pykilo_path, "clusters.metrics.pqt")
clusters_channels_file = pick_one_file(pykilo_path, "clusters.channels.npy")
channels_ids_file = pick_one_file(pykilo_path, "channels.brainLocationIds_ccf_2017.npy")
```

iii. These are the standard IBL spike sorting output files, equivalent to what `SpikeSortingLoader.load_spike_sorting()` returns in the reference code.

## 2-b. How is the `neural` data processed?

i. For each session: (1) spikes from all probes are loaded and merged, with cluster IDs remapped to avoid duplicates across probes; (2) merged spikes are sorted by time; (3) spikes are binned into 20ms bins over a [-0.5, 1.5]s window relative to stimOn_times using `bincount2D`; (4) the result is a per-trial matrix of shape (n_neurons, 100) stored as float16.

ii.
```python
def bin_spikes_for_trials(...):
    intervals = np.c_[align_times + window[0], align_times + window[1]]
    idx_starts = np.searchsorted(spike_times, intervals[:, 0], side="left")
    idx_ends = np.searchsorted(spike_times, intervals[:, 1], side="left")
    for idx0, idx1, (start, end) in zip(idx_starts, idx_ends, intervals):
        trial_counts = np.zeros((n_clusters, n_bins), dtype=np.float16)
        if idx1 > idx0:
            counts, _, cluster_idx = bincount2D(
                spike_times[idx0:idx1], spike_clusters[idx0:idx1],
                xbin=binsize, xlim=[start, end],
            )
            if counts.size:
                counts = counts[:, :n_bins]
                trial_counts[cluster_idx, : counts.shape[1]] = counts.astype(np.float16)
        results.append(trial_counts)
```

iii. The AI's processing matches the reference code's `merge_probes()` + `bin_spiking_data()` + `get_spike_data_per_interval()` pipeline conceptually, using the same `bincount2D` function and the same parameters (`binsize=0.02`, `time_window=(-0.5, 1.5)`).

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI filters neurons by keeping only clusters with `clusters.metrics.label >= 1`. This corresponds to the "well-isolated neurons" criterion from the data paper, yielding approximately 75,708 neurons across the full release. Additionally, all-zero neural trial windows are excluded after binning.

ii.
```python
GOOD_CLUSTER_LABEL = 1

metrics = pd.read_parquet(metrics_file, columns=["label"])
cluster_labels = metrics["label"].to_numpy()
good_mask = cluster_labels >= label_threshold
selected_cluster_ids = np.flatnonzero(good_mask)
```

iii. The AI justified this choice by noting it matches the data paper's definition of well-isolated neurons (75,708 out of 621,733 units) and keeps the dense pickle file tractable. The AI explicitly acknowledged that the Zhang 2025 reference code loads all clusters (qc=None) but stores a `good_clusters` metadata flag.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to stimulus onset (`stimOn_times`). For each trial, a window of [-0.5, 1.5] seconds relative to `stimOn_times` is extracted and binned into 20ms bins.

ii.
```python
TIME_WINDOW = (-0.5, 1.5)
align_times = masked_trials["stimOn_times"].to_numpy(dtype=np.float64)
neural_trials = bin_spikes_for_trials(
    spike_times, spike_clusters, n_clusters=n_clusters_good,
    align_times=align_times,
)
```

iii. This matches both the reference code parameters (`align_time='stimOn_times'`, `time_window=(-.5, 1.5)`) and the instruction "Temporally align based on stimulus onset".

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is 20ms (0.02s), producing 100 time bins per trial over the 2-second window. No temporal rebinning is applied; spikes are directly binned at 20ms resolution from the raw spike times.

ii.
```python
BINSIZE_S = 0.02
N_BINS = int(round((TIME_WINDOW[1] - TIME_WINDOW[0]) / BINSIZE_S))  # 100
```

iii. The AI chose 20ms to match the reference code's `binsize=0.02` parameter and the methods paper's generic model description ("T = 100" over 2s). The AI noted the methods paper has an inconsistency where data processing text mentions 50ms bins for choice/prior, but chose 20ms to match the executable code.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. This input is not derived from a raw data variable. It is a computed time grid based on the alignment window parameters (TIME_WINDOW and BINSIZE_S).

ii.
```python
def make_time_input() -> np.ndarray:
    return np.linspace(TIME_WINDOW[0] + BINSIZE_S, TIME_WINDOW[1], N_BINS, dtype=np.float32)
    # = np.linspace(-0.48, 1.5, 100)
```

iii. The AI constructed the time grid to match the reference code's behavior interpolation grid: `np.linspace(interval_begs + binsize, interval_ends, n_bins)`. This represents the right-edge (or center-right) time of each bin.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. A fixed array of 100 evenly-spaced values from -0.48 to 1.5 seconds is generated using `np.linspace`. This array is the same for every trial and represents relative time from stimulus onset at each bin position. It is repeated identically as one row of the 2D input array for each trial.

ii.
```python
time_input = make_time_input()  # shape (100,)
input_trial = np.vstack([
    time_input,
    np.full(N_BINS, block_num, dtype=np.float32),
]).astype(np.float32)
inputs.append(input_trial)
```

iii. The time grid starts at -0.48s (not -0.5s) because the reference code's interpolation pattern uses `interval_begs + binsize` as the first sample point. This aligns the time input with the behavioral interpolation grid.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The time input shares the same temporal grid as the neural binning (100 bins of 20ms each over [-0.5, 1.5]s). Both the neural data and the time input have 100 time points per trial, so they are inherently aligned.

ii. (Same `make_time_input()` and `bin_spikes_for_trials()` both use `TIME_WINDOW` and `BINSIZE_S`.)

iii. The AI ensures alignment by using the same window and bin parameters for all data streams.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. Trial number in block is derived from the `probabilityLeft` column of the raw trials table. The AI detects block boundaries by looking for changes in `probabilityLeft` values.

ii.
```python
block_trial_number = compute_trial_number_in_block(trials["probabilityLeft"].to_numpy())
```

iii. The instructions specify "Trial number in block" as a decoder input. The AI computed this on the unfiltered trial table so that excluded trials don't renumber the block progression, which is a reasonable choice.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. A sequential counter starts at 1 for each block and increments for each consecutive trial with the same `probabilityLeft` value. When `probabilityLeft` changes, the counter resets to 1. The counter is computed on ALL trials (before filtering), then the values for kept trials are looked up using the original trial indices.

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

# Later, index into unfiltered block numbers for kept trials:
block_vals = block_trial_number[masked_keep["index"].to_numpy()]
```

iii. The AI noted that block counter should be computed on the original unfiltered trial table so that excluded trials don't renumber the latent block progression. The value is then broadcast across all 100 time bins as a constant per-trial input.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. Choice is derived from the `choice` column of the trials table (`_ibl_trials.table.pqt`).

ii.
```python
choice_vals = map_choice_to_binary(masked_keep["choice"].to_numpy(dtype=np.float64))
```

iii. This directly corresponds to the reference code's `choice = trials_df['choice'].to_numpy()` in `bin_behaviors()`.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The raw IBL choice values (1 for left, -1 for right) are mapped to the target format (0 for left, 1 for right). Trials with choice == 0 (no response) have already been excluded by the trial mask. The mapped value is repeated across all 100 time bins as a constant per-trial output.

ii.
```python
def map_choice_to_binary(choice_values: np.ndarray) -> np.ndarray:
    mapped = np.full(choice_values.shape, -1, dtype=np.int16)
    mapped[choice_values == 1] = 0   # left
    mapped[choice_values == -1] = 1  # right
    if np.any(mapped < 0):
        raise ValueError("Unexpected choice values after masking")
    return mapped

# Per trial:
choice.append(np.full(N_BINS, choice_val, dtype=np.int16))
```

iii. The mapping follows the decoder task instructions: "Choice, binary, per-trial, left = 0, right = 1". The IBL convention is choice=1 for left and choice=-1 for right.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. Prior probability of left is derived from the `probabilityLeft` column of the trials table.

ii.
```python
prior_vals = map_prior_to_categorical(masked_keep["probabilityLeft"].to_numpy(dtype=np.float64))
```

iii. This matches the reference code's `block = trials_df['probabilityLeft'].to_numpy()` in `bin_behaviors()`.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The raw `probabilityLeft` values {0.2, 0.5, 0.8} are mapped to categorical indices {0, 1, 2} respectively. The mapped value is repeated across all 100 time bins as a constant per-trial output.

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

# Per trial:
prior.append(np.full(N_BINS, prior_val, dtype=np.int16))
```

iii. The mapping follows the decoder task instructions: "Prior probability of left, per-trial, 0.2 -> 0, 0.5 -> 1, 0.8 -> 2".

## 9-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. Wheel speed is derived from `_ibl_wheel.position.npy` (wheel position) and `_ibl_wheel.timestamps.npy` (wheel timestamps).

ii.
```python
def load_wheel_speed(session_path: Path) -> tuple[np.ndarray, np.ndarray]:
    wheel_pos_file = pick_one_file(session_path / "alf", "_ibl_wheel.position.npy")
    wheel_ts_file = pick_one_file(session_path / "alf", "_ibl_wheel.timestamps.npy")
    pos = np.asarray(np.load(wheel_pos_file), dtype=np.float64)
    ts = np.asarray(np.load(wheel_ts_file), dtype=np.float64)
```

iii. The AI loads raw wheel position and timestamps, which are the standard IBL wheel data files. The reference code accesses the same underlying data via `SessionLoader.load_wheel()`.

## 9-b. What processing is involved in computing `output` *Wheel speed*?

i. The raw wheel position is first interpolated to a uniform 1000 Hz sampling rate using `brainbox.behavior.wheel.interpolate_position()`. Then velocity is computed using `brainbox.behavior.wheel.velocity_filtered()` (which applies a Gaussian-smoothed numerical derivative). The absolute value of velocity gives speed. This continuous speed signal is then linearly interpolated onto the trial-aligned 20ms bin grid using `scipy.interpolate.interp1d`.

ii.
```python
from brainbox.behavior.wheel import interpolate_position, velocity_filtered

pos_interp, ts_interp = interpolate_position(ts, pos, freq=1000)
vel, _ = velocity_filtered(pos_interp, 1000)
return ts_interp, np.abs(vel)
```

iii. The AI imported the same `brainbox.behavior.wheel` functions that the reference code's `SessionLoader.load_wheel()` uses internally. The reference code uses `np.abs(sess_loader.wheel['velocity'].to_numpy())` for wheel-speed.

## 9-c. How is `output` *Wheel speed* thresholded into categories?

i. Wheel speed is discretized into 3 categories using global tertile edges computed across ALL valid wheel speed values from ALL sessions. The 33rd and 67th percentiles of the pooled wheel speed data define two thresholds, and `np.digitize` maps continuous values to categories {0, 1, 2} (low, medium, high).

ii.
```python
def compute_tertile_edges(values: Iterable[np.ndarray]) -> tuple[float, float]:
    concat = np.concatenate(flat_values)
    q1, q2 = np.quantile(concat, [1 / 3, 2 / 3])
    return float(q1), float(q2)

def discretize_three_bins(values: np.ndarray, edges: tuple[float, float]) -> np.ndarray:
    return np.digitize(values, bins=np.array(edges, dtype=np.float32), right=False).astype(np.int16)

wheel_edges = compute_tertile_edges(
    trial_values for session in processed for trial_values in session.wheel_cont
)
```

iii. The AI chose global tertiles to ensure balanced class distributions across the dataset (approximately 1/3 each). The instructions specify "Wheel speed discretized into 3 bins" without specifying the method, so this is a reasonable approach.

## 9-d. How is `output` *Wheel speed* aligned with the neural data?

i. Wheel speed is aligned to stimulus onset (stimOn_times), using the same [-0.5, 1.5]s window and 20ms bin grid as the neural data. The continuous wheel speed signal is interpolated onto the bin centers of this window for each trial.

ii.
```python
wheel_trials, wheel_mask = interpolate_behavior_trials(wheel_times, wheel_speed, align_times)

# In interpolate_behavior_trials:
x_rel = np.linspace(window[0] + binsize, window[1], n_bins)
interp = interp1d(ts, vals, kind="linear", fill_value="extrapolate")
outputs.append(interp(align_time + x_rel).astype(np.float32))
```

iii. The AI aligned wheel speed to stimulus onset to match the neural alignment, consistent with the instruction "Temporally align based on stimulus onset" and the reference code's use of `stimOn_times` alignment for all data streams in `0_data_caching.py`.

## 10-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. Whisker motion energy is derived from `leftCamera.ROIMotionEnergy.npy` (left camera motion energy) and `_ibl_leftCamera.times.npy` (camera timestamps). If left camera data is unavailable, it falls back to right camera equivalents.

ii.
```python
def load_whisker_motion_energy(session_path: Path) -> tuple[np.ndarray, np.ndarray, str]:
    for camera in ("left", "right"):
        me_file = pick_one_file(session_path / "alf", f"{camera}Camera.ROIMotionEnergy.npy")
        times_file = pick_one_file(session_path / "alf", f"*{camera}Camera.times.npy")
        if me_file is None or times_file is None:
            continue
        values = np.asarray(np.load(me_file), dtype=np.float64)
        times = np.asarray(np.load(times_file), dtype=np.float64)
        return times, values, camera
    raise FileNotFoundError(...)
```

iii. The left-first-then-right fallback matches the reference code's `bin_behaviors()` which tries `left-whisker-motion-energy` first and falls back to `right-whisker-motion-energy`.

## 10-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The raw whisker motion energy values and their timestamps are loaded directly (no further signal processing). The values are then linearly interpolated onto the trial-aligned 20ms bin grid using `scipy.interpolate.interp1d`, identical to the wheel speed interpolation.

ii.
```python
whisk_trials, whisk_mask = interpolate_behavior_trials(whisk_times, whisk_values, align_times)
```

iii. The reference code similarly loads whisker motion energy values directly and interpolates them onto the bin grid via `get_behavior_per_interval()`.

## 10-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Identical approach to wheel speed: global tertile edges (33rd and 67th percentiles) computed across ALL valid whisker motion energy values from ALL sessions, then `np.digitize` maps to categories {0, 1, 2}.

ii.
```python
whisker_edges = compute_tertile_edges(
    trial_values for session in processed for trial_values in session.whisker_cont
)
# Then in build_data_dict:
discretize_three_bins(whisk_cont, whisker_edges)
```

iii. Same rationale as wheel speed -- the instructions specify 3 bins without a specific method, and tertiles ensure balanced classes.

## 10-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Whisker motion energy is aligned to stimulus onset (stimOn_times), using the same [-0.5, 1.5]s window and 20ms grid as neural and wheel data.

ii. (Same `interpolate_behavior_trials()` call with `align_times` from `stimOn_times`.)

iii. Consistent with the instruction "Temporally align based on stimulus onset" and the reference code's universal `stimOn_times` alignment.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several categories of data issues:
- **Missing session files**: Sessions lacking spike sorting, wheel, or whisker files are skipped entirely.
- **Missing trial events**: Trials with NaN in required columns are excluded by the trial mask.
- **Length mismatches**: When whisker ME values and timestamps have different lengths, both are truncated to the shorter length.
- **Behavioral interpolation failures**: Trials where the behavior signal doesn't cover the full trial window (gap > binsize at boundaries) are excluded.
- **All-zero neural windows**: Trials where no spikes occur in any neuron during the trial window are excluded (added after finding 16 such trials in the full run).
- **Versioned files**: The `pick_one_file()` function handles IBL's versioned file structure (e.g., `#2024-05-06#/` revision folders) by selecting the latest revision.

ii.
```python
# Length mismatch handling:
if len(values) != len(times):
    n = min(len(values), len(times))
    values = values[:n]
    times = times[:n]

# All-zero neural trial filtering:
neural_mask = np.array([np.any(trial) for trial in neural_trials], dtype=bool)
combined_mask = wheel_mask & whisk_mask & neural_mask

# Versioned file resolution:
def version_key(path: Path) -> tuple[int, str]:
    revision = ""
    for part in path.parts:
        if part.startswith("#") and part.endswith("#"):
            revision = part.strip("#")
    is_versioned = 1 if revision else 0
    return (is_versioned, revision)
```

iii. The AI documented these edge cases in CONVERSION_NOTES.md Steps 10 and 12, noting that 16 all-zero neural trial warnings were resolved by adding the neural_mask filter.

## 12-a. What are the most time-consuming steps of the code?

i. The AI identified trial-by-trial spike binning (`bin_spikes_for_trials()`) as the main computational bottleneck. This involves iterating over each trial, extracting the relevant spike window, and calling `bincount2D` for each trial. Loading spike data from disk (memory-mapped) and behavioral interpolation are secondary costs.

ii.
```python
# The hot loop:
for idx0, idx1, (start, end) in zip(idx_starts, idx_ends, intervals):
    trial_counts = np.zeros((n_clusters, n_bins), dtype=np.float16)
    if idx1 > idx0:
        counts, _, cluster_idx = bincount2D(...)
```

iii. The AI estimated ~13s per session and ~12.5 minutes total for 459 sessions with 8 workers. The full conversion completed in ~14 minutes.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. Two main loops could potentially be vectorized:
- **`bin_spikes_for_trials()`**: The per-trial loop over `bincount2D` calls could potentially be replaced with a single vectorized operation across all trials, though this is constrained by `bincount2D`'s per-interval API.
- **`compute_trial_number_in_block()`**: The sequential Python for-loop counting block trial numbers could be vectorized using `np.diff` to find block boundaries and cumulative sums.

ii.
```python
# compute_trial_number_in_block -- sequential loop:
for i in range(1, len(prob_left)):
    if prob_left[i] == prob_left[i - 1]:
        count += 1
    else:
        count = 1
    counters[i] = count
```

iii. The AI noted that spike binning is the primary bottleneck and added `searchsorted`-based pre-windowing and `bincount2D` to speed it up, along with thread-level parallelism across sessions.

## 12-c. What processing does the code repeat multiple times?

i. The AI instantiates `BrainRegions()` once per session via the `process_session_worker()` function (which calls `BrainRegions()` inside each thread). This loads atlas data repeatedly. In a serial implementation this would be done once, but in the threaded implementation each worker creates its own instance.

ii.
```python
def process_session_worker(spec: SessionSpec) -> ProcessedSession | None:
    return process_session(spec, BrainRegions())
```

iii. The AI acknowledged this inefficiency in the threading model. Each of the 8 worker threads creates a new `BrainRegions()` object, though the overhead is small compared to spike binning.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes and stores continuous (float32) wheel speed and whisker motion energy values per trial (`wheel_cont`, `whisker_cont`), then discretizes them into 3 bins for the output. The continuous values are not included in the final pickle -- only the discretized categories appear in `output`. Thus the continuous behavioral traces are computed but ultimately discarded.

ii.
```python
# Continuous values stored temporarily:
wheel_cont=[np.asarray(x, dtype=np.float32) for x in wheel_keep],
whisker_cont=[np.asarray(x, dtype=np.float32) for x in whisk_keep],

# Only discretized values go into final output:
output_trial = np.vstack([
    choice, prior,
    discretize_three_bins(wheel_cont, wheel_edges),
    discretize_three_bins(whisk_cont, whisker_edges),
]).astype(np.int16)
```

iii. The continuous values are necessary intermediates for computing the global tertile edges, but they represent extra memory usage during processing. Additionally, the code computes processing summary plots (via `--show-processing`) which are not used by the decoder.
