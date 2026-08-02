# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads data from a local IBL ONE-style cache under `/app/data/one_cache`. It first reads a session manifest parquet file from one of three possible release directories, then builds a filesystem path for each session from `lab`, `subject`, `date`, and `number`. Within each session it directly loads ALF/parquet/numpy files for trials, spikes, wheel, and whisker data rather than using the ONE API.

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
            DATA_ROOT
            / row["lab"]
            / "Subjects"
            / row["subject"]
            / str(row["date"])
            / f"{int(row['number']):03d}"
        )
```

iii. In `CONVERSION_NOTES.md`, the agent says the local cache already contains the needed release data, so it chose direct file loading as a local equivalent of the reference ONE-based loader. It also justified using the `2025_Q3_IBL_et_al_BWM` manifest because it matched the 459-session release discussed in the notes.

## 1-b. How are the data split into subjects?

i. Subjects are split using the manifest/session metadata field `subject`. The final export builds a sorted unique subject list and stores one subject index per kept session.

ii.
```python
subject=str(row["subject"]),

subjects = sorted({s.subject for s in sessions})
subject_to_idx = {s: i for i, s in enumerate(subjects)}

"subjects": subjects,
"subject_idx": np.array([subject_to_idx[s.subject] for s in sessions], dtype=np.int16),
```

iii. The notes describe subject identity as session metadata from the release manifest, matching how the reference pipeline groups sessions by subject.

## 1-c. How are the data split into sessions?

i. The agent treats each manifest row / `eid` as a session. Each session is processed independently by `process_session`, and sessions missing required assets or enough valid trials are skipped.

ii.
```python
@dataclass
class SessionSpec:
    eid: str
    lab: str
    subject: str
    date: str
    session_number: int
    session_path: Path

def process_session(spec: SessionSpec, br: BrainRegions) -> ProcessedSession | None:
    ...

for spec in specs:
    session = process_session_worker(spec)
    if session is not None:
        processed.append(session)
```

iii. In the notes the agent explicitly maps this to the reference code’s per-`eid` preprocessing loop in `0_data_caching.py`.

## 1-d. How are the data split into trials?

i. Trials are loaded from each session’s `_ibl_trials.table.pqt` file. Each row in that parquet table is treated as one trial, then later masked and aligned.

ii.
```python
def load_trials_table(session_path: Path) -> pd.DataFrame:
    trial_file = pick_one_file(session_path / "alf", "_ibl_trials.table.pqt")
    if trial_file is None:
        raise FileNotFoundError(f"Missing trial table for {session_path}")
    return pd.read_parquet(trial_file)
```

iii. The notes say this mirrors the reference code’s `SessionLoader.load_trials()` usage, just reading the underlying cached parquet directly.

## 1-e. How are trials filtered based on quality controls?

i. The agent applies a trial mask requiring non-missing critical trial events, reaction time in `[0.08, 2.0]` s, nonzero choice, and trial duration at most 10 s. After that, it further removes masked trials whose aligned wheel interpolation fails, whose aligned whisker interpolation fails, or whose neural matrix is all zeros.

ii.
```python
def compute_trial_mask(trials: pd.DataFrame) -> np.ndarray:
    required = [
        "stimOn_times",
        "choice",
        "feedback_times",
        "probabilityLeft",
        "firstMovement_times",
        "feedbackType",
    ]
    mask = np.ones(len(trials), dtype=bool)
    rt = trials["firstMovement_times"] - trials["stimOn_times"]
    mask &= rt >= TRIAL_MASK_RT[0]
    mask &= rt <= TRIAL_MASK_RT[1]
    mask &= (trials["feedback_times"] - trials["goCue_times"]) <= MAX_TRIAL_LEN
    mask &= trials["choice"] != 0
    for col in required:
        mask &= trials[col].notna().to_numpy()
    return mask

wheel_trials, wheel_mask = interpolate_behavior_trials(wheel_times, wheel_speed, align_times)
whisk_trials, whisk_mask = interpolate_behavior_trials(whisk_times, whisk_values, align_times)
neural_mask = np.array([np.any(trial) for trial in neural_trials], dtype=bool)
combined_mask = wheel_mask & whisk_mask & neural_mask
```

iii. `CONVERSION_NOTES.md` ties the first mask to `load_trials_and_mask()` plus `max_trial_len=10.0` from the reference code. The trajectory also shows a later explicit justification for the extra `neural_mask`: the agent found 16 all-zero neural trials and decided to drop them.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from probe-level spike sorting outputs: `spikes.times.npy`, `spikes.clusters.npy`, `clusters.metrics.pqt`, `clusters.channels.npy`, and `channels.brainLocationIds_ccf_2017.npy`.

ii.
```python
spikes_times_file = pick_one_file(pykilo_path, "spikes.times.npy")
spikes_clusters_file = pick_one_file(pykilo_path, "spikes.clusters.npy")
metrics_file = pick_one_file(pykilo_path, "clusters.metrics.pqt")
clusters_channels_file = pick_one_file(pykilo_path, "clusters.channels.npy")
channels_ids_file = pick_one_file(pykilo_path, "channels.brainLocationIds_ccf_2017.npy")
```

iii. The notes explicitly connect these files to the same underlying spike-sorting assets used by the reference `SpikeSortingLoader`.

## 2-b. How is the `neural` data processed?

i. The agent merges all probes in a session, remaps cluster IDs to avoid collisions across probes, sorts spikes by time, then bins spikes per trial into 20 ms bins over a 2 s window. It stores each trial as a `(n_neurons, 100)` matrix of binned spike counts.

ii.
```python
for probe_path in probe_paths:
    times, clusters, regions, total_here, good_here = load_probe_spikes_and_regions(
        probe_path, br, label_threshold=label_threshold
    )
    ...
    merged_clusters.append(clusters + cluster_offset)
    ...

order = np.argsort(spike_times, kind="stable")
spike_times = spike_times[order]
spike_clusters = spike_clusters[order]

def bin_spikes_for_trials(...):
    intervals = np.c_[align_times + window[0], align_times + window[1]]
    ...
    for idx0, idx1, (start, end) in zip(idx_starts, idx_ends, intervals):
        trial_counts = np.zeros((n_clusters, n_bins), dtype=np.float16)
        if idx1 > idx0:
            counts, _, cluster_idx = bincount2D(
                spike_times[idx0:idx1],
                spike_clusters[idx0:idx1],
                xbin=binsize,
                xlim=[start, end],
            )
```

iii. The notes say this was chosen to match the reference `merge_probes()` and `bin_spiking_data()`/`get_spike_data_per_interval()` pipeline, using the same alignment window and `bincount2D`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The agent keeps only clusters with `clusters.metrics.label >= 1`, then drops trials whose binned neural matrices are entirely zero.

ii.
```python
GOOD_CLUSTER_LABEL = 1

metrics = pd.read_parquet(metrics_file, columns=["label"])
cluster_labels = metrics["label"].to_numpy()
good_mask = cluster_labels >= label_threshold
selected_cluster_ids = np.flatnonzero(good_mask)

neural_mask = np.array([np.any(trial) for trial in neural_trials], dtype=bool)
combined_mask = wheel_mask & whisk_mask & neural_mask
```

iii. In the notes the agent justifies `label >= 1` by matching the paper’s “well-isolated neurons” count and by keeping the exported pickle tractable. The trajectory separately justifies the all-zero-trial drop as a cleanup after observing warning-producing empty windows.

## 2-d. How is the per-trial `neural` data aligned to the event described in the instructions?

i. Neural data is aligned to `stimOn_times`, using a trial window of `[-0.5, 1.5]` seconds relative to stimulus onset.

ii.
```python
TIME_WINDOW = (-0.5, 1.5)

align_times = masked_trials["stimOn_times"].to_numpy(dtype=np.float64)
neural_trials = bin_spikes_for_trials(
    spike_times,
    spike_clusters,
    n_clusters=n_clusters_good,
    align_times=align_times,
)
```

iii. The notes say this follows both the user instruction “Temporally align based on stimulus onset” and the reference caching parameters `align_time='stimOn_times'`, `time_window=(-.5, 1.5)`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The agent uses 20 ms bins. Over a 2 s window this gives 100 bins per trial. The neural data is directly binned from raw spike times at that resolution; there is no later rebinning step.

ii.
```python
BINSIZE_S = 0.02
N_BINS = int(round((TIME_WINDOW[1] - TIME_WINDOW[0]) / BINSIZE_S))
```

iii. The notes justify 20 ms by reference to the executable Zhang code and the method-paper description with `T = 100`.

## 3-a. What variables in the raw data is `input` Time since stimulus onset derived from?

i. It is not derived from a raw file column. The agent synthesizes it from the chosen alignment window and bin size.

ii.
```python
def make_time_input() -> np.ndarray:
    return np.linspace(TIME_WINDOW[0] + BINSIZE_S, TIME_WINDOW[1], N_BINS, dtype=np.float32)
```

iii. The notes say this was chosen to match the reference behavior interpolation grid rather than a raw data field.

## 3-b. What processing is involved in computing `input` Time since stimulus onset?

i. The agent creates a fixed length-100 vector from `-0.48` to `1.5` seconds and repeats it for every trial as the first row of the trial input matrix.

ii.
```python
time_input = make_time_input()
input_trial = np.vstack(
    [
        time_input,
        np.full(N_BINS, block_num, dtype=np.float32),
    ]
).astype(np.float32)
```

iii. In the notes the agent explains that starting at `window[0] + binsize` was intended to mirror the reference `get_behavior_per_interval()` interpolation points.

## 3-c. How is the `input` Time since stimulus onset aligned with the neural data?

i. It uses the same 100-bin grid, the same 20 ms spacing, and the same stimulus-onset-relative trial window as the neural matrices, so it is aligned by construction.

ii.
```python
TIME_WINDOW = (-0.5, 1.5)
BINSIZE_S = 0.02
...
align_times = masked_trials["stimOn_times"].to_numpy(dtype=np.float64)
...
time_input = make_time_input()
```

iii. The notes repeatedly emphasize a single common stimulus-onset-aligned grid for neural and behavior variables.

## 4-a. What variables in the raw data is `input` Trial number in block derived from?

i. It is derived from the raw `probabilityLeft` trial column. Block boundaries are inferred whenever `probabilityLeft` changes.

ii.
```python
block_trial_number = compute_trial_number_in_block(trials["probabilityLeft"].to_numpy())
```

iii. The notes describe this as an additional task-required variable derived from block identity in the trial table.

## 4-b. What processing is involved in computing `input` Trial number in block?

i. The agent runs a sequential counter over the full unfiltered trial sequence, increments while `probabilityLeft` stays the same, and resets to 1 when it changes. It then looks up the original counter value for each kept trial and repeats that scalar across all 100 bins.

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

block_vals = block_trial_number[masked_keep["index"].to_numpy()]
```

iii. The notes explicitly justify computing it on the original unfiltered trial table so exclusions do not renumber the latent block progression.

## 5-a. What variables in the raw data is `output` Choice derived from?

i. Choice comes from the raw trials-table column `choice`.

ii.
```python
choice_vals = map_choice_to_binary(masked_keep["choice"].to_numpy(dtype=np.float64))
```

iii. The notes map this directly to the reference `bin_behaviors()` use of `trials_df['choice']`.

## 5-b. What processing is involved in computing `output` Choice?

i. After no-choice trials are filtered out, the agent maps raw IBL choice codes `1 -> 0` (left) and `-1 -> 1` (right), then repeats the categorical value across all 100 bins of a trial.

ii.
```python
def map_choice_to_binary(choice_values: np.ndarray) -> np.ndarray:
    mapped = np.full(choice_values.shape, -1, dtype=np.int16)
    mapped[choice_values == 1] = 0   # left
    mapped[choice_values == -1] = 1  # right
    if np.any(mapped < 0):
        raise ValueError("Unexpected choice values after masking")
    return mapped

choice.append(np.full(N_BINS, choice_val, dtype=np.int16))
```

iii. The notes justify this as the task-required remapping from the raw IBL sign convention to the requested categorical encoding.

## 6-a. What variables in the raw data is `output` Prior probability of left derived from?

i. It is derived from the raw trials-table column `probabilityLeft`.

ii.
```python
prior_vals = map_prior_to_categorical(masked_keep["probabilityLeft"].to_numpy(dtype=np.float64))
```

iii. The notes match this to the reference code’s `block = trials_df['probabilityLeft']`.

## 6-b. What processing is involved in computing `output` Prior probability of left?

i. The agent maps `0.2 -> 0`, `0.5 -> 1`, and `0.8 -> 2`, then repeats the resulting category across all time bins within a trial.

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
```

iii. The notes justify this as a direct response to the task’s requested categorical coding.

## 9-a. What variables in the raw data is `output` Wheel speed derived from?

i. Wheel speed is derived from `_ibl_wheel.position.npy` and `_ibl_wheel.timestamps.npy`.

ii.
```python
def load_wheel_speed(session_path: Path) -> tuple[np.ndarray, np.ndarray]:
    wheel_pos_file = pick_one_file(session_path / "alf", "_ibl_wheel.position.npy")
    wheel_ts_file = pick_one_file(session_path / "alf", "_ibl_wheel.timestamps.npy")
    ...
    pos = np.asarray(np.load(wheel_pos_file), dtype=np.float64)
    ts = np.asarray(np.load(wheel_ts_file), dtype=np.float64)
```

iii. The notes say this reproduces the same underlying wheel source used by `SessionLoader.load_wheel()` in the reference code.

## 9-b. What processing is involved in computing `output` Wheel speed?

i. The agent interpolates wheel position to 1000 Hz, computes filtered velocity, takes the absolute value to get speed, and then linearly interpolates that continuous speed onto each trial’s aligned 20 ms grid.

ii.
```python
pos_interp, ts_interp = interpolate_position(ts, pos, freq=1000)
vel, _ = velocity_filtered(pos_interp, 1000)
return ts_interp, np.abs(vel)

interp = interp1d(ts, vals, kind="linear", fill_value="extrapolate")
outputs.append(interp(align_time + x_rel).astype(np.float32))
```

iii. In the notes the agent says it intentionally reused the same `brainbox.behavior.wheel` functions as the reference stack.

## 9-c. How is `output` Wheel speed thresholded into categories?

i. The agent computes two global wheel-speed thresholds from pooled valid values across all sessions, using the 33rd and 67th percentiles. It then maps each aligned value into bins `{0,1,2}` with `np.digitize`.

ii.
```python
def compute_tertile_edges(values: Iterable[np.ndarray]) -> tuple[float, float]:
    flat_values = [np.asarray(v, dtype=np.float64).ravel() for v in values if len(v)]
    concat = np.concatenate(flat_values)
    q1, q2 = np.quantile(concat, [1 / 3, 2 / 3])
    ...

def discretize_three_bins(values: np.ndarray, edges: tuple[float, float]) -> np.ndarray:
    return np.digitize(values, bins=np.array(edges, dtype=np.float32), right=False).astype(np.int16)
```

iii. The notes explicitly justify tertiles as a pragmatic way to get roughly balanced three-class targets because the instructions required 3 bins but did not specify how to choose thresholds.

## 9-d. How is `output` Wheel speed aligned with the neural data?

i. It is aligned to `stimOn_times` using the same `[-0.5, 1.5]` window and same 100-bin 20 ms grid as the neural data.

ii.
```python
wheel_trials, wheel_mask = interpolate_behavior_trials(wheel_times, wheel_speed, align_times)

def interpolate_behavior_trials(...):
    x_rel = np.linspace(window[0] + binsize, window[1], n_bins)
```

iii. The notes say this followed the explicit task instruction to align everything to stimulus onset and the reference caching script’s global `stimOn_times` alignment.

## 10-a. What variables in the raw data is `output` Whisker motion energy derived from?

i. Whisker motion energy is derived from `leftCamera.ROIMotionEnergy.npy` and matching camera timestamps, with fallback to the right camera if the left camera stream is unavailable.

ii.
```python
def load_whisker_motion_energy(session_path: Path) -> tuple[np.ndarray, np.ndarray, str]:
    for camera in ("left", "right"):
        me_file = pick_one_file(session_path / "alf", f"{camera}Camera.ROIMotionEnergy.npy")
        times_file = pick_one_file(session_path / "alf", f"*{camera}Camera.times.npy")
        ...
        return times, values, camera
```

iii. The notes explicitly cite the reference `bin_behaviors()` left-then-right fallback and adopt it unchanged.

## 10-b. What processing is involved in computing `output` Whisker motion energy?

i. The raw motion-energy values are loaded directly, length-matched to timestamps if necessary by truncation, and linearly interpolated onto each trial’s aligned 20 ms grid.

ii.
```python
if len(values) != len(times):
    n = min(len(values), len(times))
    values = values[:n]
    times = times[:n]

whisk_trials, whisk_mask = interpolate_behavior_trials(whisk_times, whisk_values, align_times)
```

iii. The notes describe this as matching the reference interpolation logic after loading the appropriate whisker motion-energy stream.

## 10-c. How is `output` Whisker motion energy thresholded into categories?

i. It uses the same global-tertile scheme as wheel speed: pooled valid values across all sessions, 33rd/67th percentile thresholds, and `np.digitize` into three classes.

ii.
```python
whisker_edges = compute_tertile_edges(
    trial_values for session in processed for trial_values in session.whisker_cont
)
...
discretize_three_bins(whisk_cont, whisker_edges)
```

iii. The notes give the same justification as for wheel speed: the task required 3 bins, so the agent chose global tertiles for balanced categories.

## 10-d. How is `output` Whisker motion energy aligned with the neural data?

i. It is aligned to `stimOn_times` with the same common `[-0.5, 1.5]` / 20 ms trial grid as the neural and wheel signals.

ii.
```python
whisk_trials, whisk_mask = interpolate_behavior_trials(whisk_times, whisk_values, align_times)
```

iii. The notes say the agent intentionally forced all exported variables onto one stimulus-onset-aligned grid.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The agent handles missing or messy data by skipping sessions missing essential files, excluding trials with missing critical events, rejecting aligned behavior windows with insufficient coverage, truncating whisker arrays if values/timestamps differ in length, selecting the latest versioned ALF file when multiple revisions exist, and dropping all-zero neural windows.

ii.
```python
def pick_one_file(base: Path, pattern: str) -> Path | None:
    matches = list(base.rglob(pattern))
    if not matches:
        return None
    matches.sort(key=lambda p: (version_key(p), str(p)))
    return matches[-1]

if len(values) != len(times):
    n = min(len(values), len(times))
    values = values[:n]
    times = times[:n]

if np.abs(start - ts[0]) > binsize:
    outputs.append(None)
    continue
if np.abs(end - ts[-1]) > binsize:
    outputs.append(None)
    continue
```

iii. The notes document these as pragmatic fixes for the local cache. The trajectory specifically calls out the all-zero neural trial removal as a late correction after observing verifier warnings.

## 12-a. What are the most time-consuming steps of the code?

i. The agent’s own notes identify per-trial spike binning as the dominant cost. That is the loop over aligned spike windows calling `bincount2D` once per trial. Session-wide loading and behavior interpolation are secondary.

ii.
```python
for idx0, idx1, (start, end) in zip(idx_starts, idx_ends, intervals):
    trial_counts = np.zeros((n_clusters, n_bins), dtype=np.float16)
    if idx1 > idx0:
        counts, _, cluster_idx = bincount2D(
            spike_times[idx0:idx1],
            spike_clusters[idx0:idx1],
            xbin=binsize,
            xlim=[start, end],
        )
```

iii. `CONVERSION_NOTES.md` explicitly flags trial-by-trial spike binning as the main bottleneck and reports total runtime consistent with that choice.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The most obvious non-vectorized loops are the per-trial spike-binning loop and the sequential block-counter loop. The behavior interpolation loop also processes trials one-by-one after precomputing search bounds.

ii.
```python
for i in range(1, len(prob_left)):
    if prob_left[i] == prob_left[i - 1]:
        count += 1
    else:
        count = 1
    counters[i] = count

for i, align_time in enumerate(align_times):
    ...
    interp = interp1d(ts, vals, kind="linear", fill_value="extrapolate")
    outputs.append(interp(align_time + x_rel).astype(np.float32))
```

iii. The notes mention the spike loop and block-counter loop as clear vectorization opportunities, with spike binning singled out as the important one.

## 12-c. What processing does the code repeat multiple times?

i. The code repeatedly constructs per-session `BrainRegions()` objects in worker calls, repeatedly recomputes trial-wise interpolation for wheel and whisker separately, and repeats the same fixed time grid and repeated per-trial broadcast operations for static variables.

ii.
```python
def process_session_worker(spec: SessionSpec) -> ProcessedSession | None:
    return process_session(spec, BrainRegions())

wheel_trials, wheel_mask = interpolate_behavior_trials(wheel_times, wheel_speed, align_times)
whisk_trials, whisk_mask = interpolate_behavior_trials(whisk_times, whisk_values, align_times)

for block_num, choice_val, prior_val in zip(block_vals, choice_vals, prior_vals, strict=True):
    input_trial = np.vstack(
        [
            time_input,
            np.full(N_BINS, block_num, dtype=np.float32),
        ]
    ).astype(np.float32)
```

iii. The notes specifically mention per-session `BrainRegions()` construction as repeated work and discuss repeated per-trial construction of constant vectors as part of the export formatting.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code keeps continuous aligned wheel and whisker traces in memory even though the final export only stores their discretized 3-bin versions. It also has optional processing-summary plotting that is not used by downstream decoding.

ii.
```python
return ProcessedSession(
    ...
    wheel_cont=[np.asarray(x, dtype=np.float32) for x in wheel_keep],
    whisker_cont=[np.asarray(x, dtype=np.float32) for x in whisk_keep],
    ...
)

output_trial = np.vstack(
    [
        choice,
        prior,
        discretize_three_bins(wheel_cont, wheel_edges),
        discretize_three_bins(whisk_cont, whisker_edges),
    ]
).astype(np.int16)
```

iii. The notes justify keeping the continuous traces temporarily because the discretization thresholds are computed globally afterward, but they do not survive into the final pickle.
