# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from the frozen BWM release CSV (`bwm_release.csv`) to identify the 459 release sessions and 699 probe insertions. For each session, it resolves data files under `data/one_cache/<lab>/Subjects/<subject>/<date>/<number>/alf/`. It uses a two-pass approach: pass 1 loads trials, wheel, and whisker behavioral data; pass 2 loads spike data and bins it. Direct ALF file loading is used instead of the ONE API because the API dependencies were unavailable in the environment.

ii.
```python
RELEASE_CSV = Path("code/code_zhang2025/data/bwm_release.csv")
DATA_ROOT = Path("data/one_cache")

def load_release_sessions() -> list[SessionSpec]:
    bwm = pd.read_csv(RELEASE_CSV, index_col=0)
    grouped = bwm.groupby("eid", sort=False)
    sessions: list[SessionSpec] = []
    for eid, df in grouped:
        row = df.iloc[0]
        probe_names = tuple(df["probe_name"].tolist())
        sessions.append(
            SessionSpec(
                eid=eid,
                subject=str(row["subject"]),
                lab=str(row["lab"]),
                date=str(row["date"]),
                session_number=int(row["session_number"]),
                probe_names=probe_names,
            )
        )
    return sessions
```

iii. The AI justified using `bwm_release.csv` as the authoritative session list because it matches both the data paper's 459 sessions / 699 insertions and the reference code's data source. Direct ALF loading was used as a workaround for ONE/SessionLoader import failures in the environment.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified from the `subject` column of `bwm_release.csv`. A sorted list of unique subject names is created, and each session is assigned an index into this list.

ii.
```python
subject_names = sorted({ps.spec.subject for ps in prepared_sessions})
subject_to_idx = {subject: i for i, subject in enumerate(subject_names)}
subject_idx_list.append(subject_to_idx[prepared.spec.subject])
```

iii. The AI's approach yields 135 unique subjects in the final dataset (from 139 in the release) because 4 subjects had all their sessions excluded due to missing whisker motion energy data. This is documented in CONVERSION_NOTES.md.

## 1-c. How are the data split into sessions?

i. Sessions are defined by unique `eid` values from the BWM release CSV. Each session may contain 1 or 2 probes, which are merged. The final dataset contains 438 sessions (of 459 in the release), with 21 excluded due to missing whisker data or insufficient valid trials.

ii.
```python
sessions = load_release_sessions()  # 459 sessions from CSV
# Sessions excluded in pass 1:
if prepared is None:
    excluded.append((eid, error or "unknown"))
else:
    prepared_sessions.append(prepared)
```

iii. The AI documented that 20 sessions were excluded for missing whisker motion energy streams and 1 for having no good units or fewer than 2 valid trials after filtering.

## 1-d. How are the data split into trials?

i. Within each session, trials are loaded from the `_ibl_trials.table.pqt` parquet file. Trials are split by row index in this table. Each trial corresponds to one behavioral trial in the experiment.

ii.
```python
def load_trials_table(session_path: Path) -> pd.DataFrame:
    path = resolve_latest(session_path / "alf", "**/_ibl_trials.table.pqt")
    return pd.read_parquet(path)
```

iii. Standard IBL trial structure: each row in the trials table is one trial.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered using a multi-criteria mask that checks for: non-null values in 6 key event columns, reaction time between 0.08-2.0s, trial duration <= 10s, and exclusion of no-choice trials (choice != 0). Additionally, trials are excluded if wheel or whisker behavioral data cannot be interpolated within the trial's time window.

ii.
```python
def compute_trial_mask(trials: pd.DataFrame) -> np.ndarray:
    rt = trials["firstMovement_times"] - trials["stimOn_times"]
    mask = (
        ~trials["stimOn_times"].isnull()
        & ~trials["choice"].isnull()
        & ~trials["feedback_times"].isnull()
        & ~trials["probabilityLeft"].isnull()
        & ~trials["firstMovement_times"].isnull()
        & ~trials["feedbackType"].isnull()
        & (rt >= 0.08)
        & (rt <= 2.0)
        & ((trials["feedback_times"] - trials["goCue_times"]) <= 10.0)
        & (trials["choice"] != 0)
    )
    return mask.to_numpy(dtype=bool)

# Additional behavioral coverage mask:
keep_mask = trial_mask & wheel_mask & whisker_mask
```

iii. The AI justified these filters by referencing the reference code's `load_trials_and_mask` function (which uses the same 6 non-null fields, RT range 0.08-2.0s, and no-choice exclusion) and the `prepare_data` function (which passes `max_trial_len=10.0`). The behavioral coverage mask ensures no trials have missing or insufficient wheel/whisker data in the aligned window.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `spikes.times.npy` and `spikes.clusters.npy` for each probe, filtered by `clusters.metrics.pqt` (keeping only clusters with `label >= 1`). Brain region assignments come from `clusters.channels.npy` and `channels.brainLocationIds_ccf_2017.npy`.

ii.
```python
spikes_times = np.load(spikes_times_path, mmap_mode="r")
spikes_clusters = np.load(spikes_clusters_path, mmap_mode="r")
metrics = pd.read_parquet(metrics_path, columns=["label"])
good_rows = metrics["label"].to_numpy(copy=False) >= 1
```

iii. The AI chose to filter to `label >= 1` (well-isolated neurons) rather than using all clusters, justifying this by matching the data paper's 75,708 well-isolated neuron count.

## 2-b. How is the `neural` data processed?

i. Spikes from multiple probes are merged within each session (re-indexing cluster IDs and sorting by time). Merged spikes are then binned into 20ms bins within a [-0.5, 1.5]s window around stimulus onset, producing spike count matrices of shape (n_neurons, 100) per trial. Data is stored as float16.

ii.
```python
def bin_spikes_by_trial(spike_times, spike_clusters, align_times, n_units,
                        binsize=BINSIZE, time_window=TIME_WINDOW):
    interval_begs = align_times + time_window[0]
    interval_ends = align_times + time_window[1]
    n_bins = int(np.ceil((time_window[1] - time_window[0]) / binsize))
    start_idx = np.searchsorted(spike_times, interval_begs, side="left")
    end_idx = np.searchsorted(spike_times, interval_ends, side="left")
    out = []
    for i in range(len(align_times)):
        rel = spike_times[i0:i1] - interval_begs[i]
        bin_idx = np.floor(rel / binsize).astype(np.int64)
        valid = (bin_idx >= 0) & (bin_idx < n_bins)
        flat = spike_clusters[i0:i1][valid] * n_bins + bin_idx[valid]
        counts = np.bincount(flat, minlength=n_units * n_bins).reshape(n_units, n_bins)
        out.append(counts.astype(np.float16))
    return out
```

iii. The AI documented that spike counts (not firing rates) are used, matching the reference code's `get_spike_data_per_interval`. No smoothing is applied.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only clusters with `label >= 1` in `clusters.metrics.pqt` are retained. This filters from ~621K total units to ~75K well-isolated neurons across the release. No additional neuron-level filtering (e.g., minimum firing rate) is applied.

ii.
```python
metrics = pd.read_parquet(metrics_path, columns=["label"])
good_rows = metrics["label"].to_numpy(copy=False) >= 1
if not np.any(good_rows):
    continue
```

iii. The AI justified this filter by noting it exactly reproduces the data paper's 75,708 well-isolated neuron count. The AI acknowledged the reference code loads all clusters (`qc=None`) but argued the `label >= 1` filter is more appropriate for the dense decoder format.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to `stimOn_times` (stimulus onset). For each trial, spikes are extracted from `stimOn_times - 0.5` to `stimOn_times + 1.5` seconds and binned into 100 bins of 20ms each.

ii.
```python
ALIGN_EVENT = "stimOn_times"
TIME_WINDOW = (-0.5, 1.5)
BINSIZE = 0.02
NBINS = int(round((TIME_WINDOW[1] - TIME_WINDOW[0]) / BINSIZE))  # 100

kept_align_times = prepared.trials.loc[prepared.keep_mask, ALIGN_EVENT].to_numpy(dtype=np.float64)
neural_trials = bin_spikes_by_trial(
    spike_times=spike_times, spike_clusters=spike_clusters,
    align_times=kept_align_times, n_units=cluster_regions.shape[0],
)
```

iii. The AI documented that this matches the reference code's `align_time='stimOn_times'` and `time_window=(-0.5, 1.5)` parameters from `0_data_caching.py`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is 20ms (0.02s), yielding 100 time bins per trial over the 2-second window. No temporal rebinning is applied beyond the initial spike binning.

ii.
```python
BINSIZE = 0.02
NBINS = int(round((TIME_WINDOW[1] - TIME_WINDOW[0]) / BINSIZE))  # 100
```

iii. The AI noted this matches the reference code's `binsize=0.02` and is consistent with the method paper's description of 20ms bins for choice and dynamic behavior decoding.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. The time since stimulus onset is derived from the alignment parameters (`TIME_WINDOW` and `BINSIZE`), not from any per-trial raw variable. It is a fixed vector of 100 time points relative to stimulus onset.

ii.
```python
COMMON_RELATIVE_TIMES = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], NBINS, dtype=np.float32)
# = linspace(-0.48, 1.50, 100)
```

iii. The AI constructed this as a common time vector shared across all trials, matching the reference code's `linspace(interval_beg + binsize, interval_end, n_bins)` interpolation grid.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. A fixed vector of 100 linearly spaced values from -0.48 to 1.50 seconds is computed once and reused for every trial. These represent bin centers offset by one binsize from the window start.

ii.
```python
COMMON_RELATIVE_TIMES = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], NBINS, dtype=np.float32)
# Results in: [-0.48, -0.46, -0.44, ..., 1.48, 1.50]
```

iii. The AI used the same `linspace(start + binsize, end, n_bins)` formula as the reference code's behavior interpolation grid, ensuring temporal consistency between neural and behavioral data.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The time since stimulus onset vector uses the same temporal grid as the neural spike bins and behavioral interpolation, so it is inherently aligned. Each element corresponds to one 20ms bin in the neural data.

ii.
```python
inp = np.vstack([
    COMMON_RELATIVE_TIMES,
    np.full(NBINS, prepared.trial_number_in_block[trial_idx], dtype=np.float32),
]).astype(np.float32)
```

iii. The AI documented that using a common grid for all data streams ensures no temporal misalignment.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. Trial number in block is derived from the `probabilityLeft` column of the trials table. Block boundaries are detected whenever `probabilityLeft` changes value.

ii.
```python
trial_number_in_block = compute_trial_number_in_block(trials["probabilityLeft"].to_numpy())
```

iii. The AI identified block changes from transitions in `probabilityLeft` (0.2, 0.5, or 0.8), counting the position within each block starting from 1.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. A sequential counter is maintained over the unfiltered trial table. The counter resets to 1 whenever `probabilityLeft` changes value (indicating a new block), and increments by 1 for each subsequent trial in the same block. The computation is done on the full trial table before trial filtering, then subsetted to kept trials. The value is repeated across all 100 time bins.

ii.
```python
def compute_trial_number_in_block(prob_left: np.ndarray) -> np.ndarray:
    out = np.zeros(len(prob_left), dtype=np.int16)
    prev = None
    counter = 0
    for i, val in enumerate(prob_left):
        current = None if pd.isna(val) else float(val)
        if i == 0 or current != prev:
            counter = 1
        else:
            counter += 1
        out[i] = counter
        prev = current
    return out
```

iii. The AI justified computing block numbers on the unfiltered trial table to preserve actual behavioral position within blocks rather than renumbering after trial exclusion.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. Choice is derived from the `choice` column of `_ibl_trials.table.pqt`, which contains values 1 (left), -1 (right), and 0 (no-go, excluded by trial filter).

ii.
```python
choice_raw = trials.loc[keep_mask, "choice"].to_numpy()
```

iii. Documented as standard IBL trial table variable.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The raw choice values are remapped: left (1) -> 0, right (-1) -> 1. The mapped value is a per-trial scalar repeated across all 100 time bins.

ii.
```python
def map_choice(raw_choice: np.ndarray) -> np.ndarray:
    mapped = np.empty(raw_choice.shape[0], dtype=np.int8)
    mapped[raw_choice == 1] = 0    # left
    mapped[raw_choice == -1] = 1   # right
    if not np.all(np.isin(raw_choice, [-1, 1])):
        raise ValueError("Unexpected choice values encountered after filtering.")
    return mapped

# In output construction:
np.full(NBINS, choice[trial_idx], dtype=np.int8)
```

iii. The mapping follows the decoder task specification: "left = 0, right = 1".

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. Prior probability of left is derived from the `probabilityLeft` column of `_ibl_trials.table.pqt`, which contains values 0.2, 0.5, or 0.8.

ii.
```python
prior_raw = trials.loc[keep_mask, "probabilityLeft"].to_numpy()
```

iii. Standard IBL trial table variable representing the block prior.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The raw `probabilityLeft` values are mapped to categorical indices: 0.2 -> 0, 0.5 -> 1, 0.8 -> 2. The mapped value is repeated across all 100 time bins.

ii.
```python
def map_prior(raw_prior: np.ndarray) -> np.ndarray:
    out = np.empty(raw_prior.shape[0], dtype=np.int8)
    mapper = {0.2: 0, 0.5: 1, 0.8: 2}
    for i, val in enumerate(raw_prior):
        key = round(float(val), 1)
        if key not in mapper:
            raise ValueError(f"Unexpected probabilityLeft value {val}")
        out[i] = mapper[key]
    return out

# In output construction:
np.full(NBINS, prior[trial_idx], dtype=np.int8)
```

iii. The mapping follows the decoder task specification: "0.2 -> 0, 0.5 -> 1, 0.8 -> 2".

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. Wheel speed is derived from `_ibl_wheel.timestamps.npy` and `_ibl_wheel.position.npy`.

ii.
```python
timestamps = np.load(resolve_latest(alf_path, "**/_ibl_wheel.timestamps.npy"))
position = np.load(resolve_latest(alf_path, "**/_ibl_wheel.position.npy"))
```

iii. These are the standard IBL wheel data files.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Wheel position is interpolated to 1000 Hz using `brainbox.behavior.wheel.interpolate_position`, then velocity is computed using a low-pass Butterworth filter via `velocity_filtered` (corner frequency 20 Hz, order 8). Speed is the absolute value of velocity. The continuous speed signal is then linearly interpolated per trial onto the 100-bin stimulus-aligned grid.

ii.
```python
def load_wheel_speed(session_path: Path) -> tuple[np.ndarray, np.ndarray]:
    alf_path = session_path / "alf"
    timestamps = np.load(resolve_latest(alf_path, "**/_ibl_wheel.timestamps.npy"))
    position = np.load(resolve_latest(alf_path, "**/_ibl_wheel.position.npy"))
    interp_pos, interp_times = interpolate_position(timestamps, position, freq=1000)
    velocity, _ = velocity_filtered(interp_pos, fs=1000, corner_frequency=20, order=8)
    return interp_times.astype(np.float64), np.abs(velocity).astype(np.float32)
```

iii. The AI reused the `brainbox.behavior.wheel` functions from the bundled ibllib code, matching the reference code's `SessionLoader.load_wheel()` which calls the same functions.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Wheel speed is discretized into 3 bins (low=0, medium=1, high=2) using global tertile edges computed across all kept time points from all included sessions. The edges are the 1/3 and 2/3 quantiles of the pooled wheel speed values.

ii.
```python
all_wheel = np.concatenate([ps.wheel_cont.reshape(-1) for ps in prepared_sessions]).astype(np.float32)
wheel_edges = robust_tertile_edges(all_wheel)

def robust_tertile_edges(values):
    finite = np.asarray(values, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    q1, q2 = np.quantile(finite, [1/3, 2/3])
    ...
    return float(q1), float(q2)

def digitize_three_bins(values, edges):
    low, high = edges
    return np.digitize(values, bins=np.array([low, high]), right=False).astype(np.int8)
```

iii. The AI chose global (cross-session) tertile binning to ensure consistent category definitions across sessions, noting that session-specific thresholds would make class labels inconsistent. This yields exactly 1/3 in each bin by construction.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. Wheel speed is aligned to stimulus onset (same as neural data), using the same [-0.5, 1.5]s window and 100-bin interpolation grid. Per-trial interpolation uses `np.interp` on `linspace(interval_beg + binsize, interval_end, n_bins)`.

ii.
```python
wheel_interp, wheel_mask = interpolate_behavior_per_trial(wheel_times, wheel_speed, align_times)

# Inside interpolate_behavior_per_trial:
x_interp = np.linspace(t_beg + binsize, t_end, n_bins)
y_interp = np.interp(x_interp, curr_times, curr_vals)
```

iii. The AI aligned wheel speed to stimulus onset as specified in the task instructions, noting this differs from the method paper which aligns dynamic behaviors to first movement onset.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. Whisker motion energy is derived from `leftCamera.ROIMotionEnergy.npy` (preferred) or `rightCamera.ROIMotionEnergy.npy` (fallback), along with the corresponding camera timestamps `_ibl_leftCamera.times.npy` or `_ibl_rightCamera.times.npy`.

ii.
```python
def load_whisker_motion_energy(session_path: Path) -> tuple[np.ndarray, np.ndarray, str]:
    alf_path = session_path / "alf"
    for view in ("left", "right"):
        me_candidates = sorted(alf_path.glob(f"**/{view}Camera.ROIMotionEnergy.npy"))
        ts_candidates = sorted(alf_path.glob(f"**/_ibl_{view}Camera.times.npy"))
        if not me_candidates or not ts_candidates:
            continue
        motion_energy = np.load(me_candidates[-1])
        timestamps = np.load(ts_candidates[-1])
        timestamps, motion_energy = check_video_timestamps(view, timestamps, motion_energy)
        return timestamps.astype(np.float64), motion_energy.astype(np.float32), view
    raise FileNotFoundError(...)
```

iii. The AI followed the reference code pattern of preferring left camera and falling back to right.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The raw motion energy values are loaded directly from the precomputed `.npy` files. Camera timestamps that are longer than the motion energy array are trimmed from the front (a known data irregularity). The continuous signal is then linearly interpolated per trial onto the 100-bin stimulus-aligned grid.

ii.
```python
def check_video_timestamps(view, video_timestamps, video_data):
    if video_timestamps.shape[0] > video_data.shape[0]:
        video_timestamps = video_timestamps[-video_data.shape[0]:]
    return video_timestamps, video_data
```

iii. The AI documented this front-trimming repair as matching the data architecture convention and the reference code's handling.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Same approach as wheel speed: global tertile edges computed across all kept time points from all included sessions, yielding 3 bins (low=0, medium=1, high=2).

ii.
```python
all_whisker = np.concatenate([ps.whisker_cont.reshape(-1) for ps in prepared_sessions]).astype(np.float32)
whisker_edges = robust_tertile_edges(all_whisker)
whisker_bins = digitize_three_bins(prepared.whisker_cont, whisker_edges)
```

iii. Same justification as wheel speed: global tertile binning for cross-session consistency.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Same approach as wheel speed: aligned to stimulus onset using the same [-0.5, 1.5]s window and 100-bin interpolation grid.

ii.
```python
whisker_interp, whisker_mask = interpolate_behavior_per_trial(whisker_times, whisker_motion, align_times)
```

iii. Same justification as wheel speed: stimulus-onset alignment per task instructions.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Several strategies handle data irregularities:
- Camera timestamps longer than motion energy arrays are trimmed from the front.
- Trials with NaN in key event columns are excluded by the trial mask.
- Trials where behavioral data doesn't cover the full interpolation window (within one binsize tolerance) are excluded.
- Sessions missing whisker motion energy entirely are excluded.
- Sessions with zero good units or fewer than 2 valid trials are excluded.
- Zero-spike trials (16 across 3 sessions) are preserved as all-zero neural matrices rather than excluded.
- Spike cluster indices exceeding the cluster count are handled with bounds checking.

ii.
```python
# Camera timestamp repair:
if video_timestamps.shape[0] > video_data.shape[0]:
    video_timestamps = video_timestamps[-video_data.shape[0]:]

# Behavioral coverage check:
if abs(t_beg - curr_times[0]) > binsize: continue
if abs(t_end - curr_times[-1]) > binsize: continue

# Spike cluster bounds check:
valid_spikes = spikes_clusters < n_clusters
if np.all(valid_spikes):
    spike_mask = good_rows[spikes_clusters]
else:
    spike_mask = np.zeros(spikes_clusters.shape[0], dtype=bool)
    spike_mask[valid_spikes] = good_rows[spikes_clusters[valid_spikes]]
```

iii. The AI investigated the 16 zero-spike trial warnings from the verifier and confirmed they correspond to genuine zero-spike windows in the raw data. These trials were preserved to avoid inventing additional curation rules not in the reference code.

## 10-a. What are the most time-consuming steps of the code?

i. The AI identified spike loading and binning in pass 2 as the main bottleneck. Initial full-run estimates projected ~20 minutes, which was optimized to ~10 minutes through memory-mapped loading, no-copy dtype handling, and increased parallelism.

ii. From CONVERSION_NOTES: "Initial full-run profiling showed the real bottleneck was spike-stream reload and dtype copying in pass 2, not the trial-binning code."

iii. The AI documented timing measurements: pass 1 took ~144s, pass 2 build took ~456s, total ~10 minutes for the full conversion.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial spike binning loop (`bin_spikes_by_trial`) iterates over trials sequentially. The `compute_trial_number_in_block` function uses a Python for-loop. The `map_prior` function also uses a Python for-loop.

ii.
```python
# Per-trial spike binning loop:
for i in range(len(align_times)):
    ...
    flat = spike_clusters[i0:i1][valid] * n_bins + bin_idx[valid]
    counts = np.bincount(flat, minlength=n_units * n_bins).reshape(n_units, n_bins)
    out.append(counts.astype(np.float16))

# Trial number in block loop:
for i, val in enumerate(prob_left):
    current = None if pd.isna(val) else float(val)
    if i == 0 or current != prev:
        counter = 1
    else:
        counter += 1
    out[i] = counter

# Behavior interpolation loop:
for i in range(len(align_times)):
    x_interp = np.linspace(t_beg + binsize, t_end, n_bins)
    y_interp = np.interp(x_interp, curr_times, curr_vals)
```

iii. The AI noted that within each trial the binning is vectorized (`np.searchsorted`, `np.bincount`), but the outer trial loop could potentially be batched. The `compute_trial_number_in_block` loop is inherently sequential due to its stateful counter logic.

## 10-c. What processing does the code repeat multiple times?

i. The two-pass architecture means behavioral data (trials, wheel, whisker) is loaded and processed in pass 1 for all sessions, and then in pass 2 the session data structures are reused. However, spike data is only loaded once in pass 2. The `BrainRegions()` atlas object is instantiated once per session in `build_session_payload` rather than once globally.

ii.
```python
# BrainRegions instantiated per session:
def build_session_payload(prepared, wheel_edges, whisker_edges):
    session_start = time.time()
    brain_regions = BrainRegions()  # Created each time
```

iii. The AI documented that behavioral data is loaded only once in pass 1 and reused, but the `BrainRegions()` instantiation is repeated per session.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several processing steps produce data that may not be used downstream:
- Per-trial static outputs (choice, prior) are repeated across all 100 time bins, creating 100x redundancy for per-trial values.
- The `whisker_source` (left vs right camera) metadata is tracked per session but not used in the decoder.
- Detailed exclusion notes and metadata are stored but not used by the decoder.
- The `robust_tertile_edges` function includes a fallback for degenerate edge cases (all-same values) that is unlikely to be triggered.

ii.
```python
# Choice and prior repeated across all time bins:
np.full(NBINS, choice[trial_idx], dtype=np.int8)
np.full(NBINS, prior[trial_idx], dtype=np.int8)
```

iii. The AI documented that repeating static outputs across time bins was done for shape consistency with time-varying outputs, as required by the target format specification.
