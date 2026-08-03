# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does not use the ONE API. It loads a local parquet session manifest from `data/one_cache`, constructs filesystem paths for each session, then reads the trial table and modality files directly from the cache tree with `pandas.read_parquet` and `numpy.load`.

ii.
```python
def load_session_manifest() -> pd.DataFrame:
    for manifest in MANIFEST_FILES:
        if manifest.exists():
            return pd.read_parquet(manifest)
```

```python
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

```python
def load_trials_table(session_path: Path) -> pd.DataFrame:
    trial_file = pick_one_file(session_path / "alf", "_ibl_trials.table.pqt")
    return pd.read_parquet(trial_file)
```

iii. In `CONVERSION_NOTES.md` Step 4 and Step 6, the AI says it chose the local `2025_Q3_IBL_et_al_BWM/sessions.pqt` manifest as the canonical release index and implemented direct local-cache loading because the cache contained versioned ALF files that needed robust path resolution.

## 1-b. How are the data split into subjects?

i. Subjects are taken directly from the manifest row field `row["subject"]`. The script stores the subject in each `SessionSpec`, then later builds `subjects` as the sorted unique subject names and `subject_idx` as one index per kept session.

ii.
```python
specs.append(
    SessionSpec(
        eid=eid,
        lab=str(row["lab"]),
        subject=str(row["subject"]),
        date=str(row["date"]),
        session_number=int(row["number"]),
        session_path=session_path,
    )
)
```

```python
subjects = sorted({s.subject for s in sessions})
subject_to_idx = {s: i for i, s in enumerate(subjects)}
...
"subject_idx": np.array([subject_to_idx[s.subject] for s in sessions], dtype=np.int16),
```

iii. In `CONVERSION_NOTES.md` Step 5, the AI says subject metadata should come from session metadata and then be assembled deterministically into `subjects` and `subject_idx`.

## 1-c. How are the data split into sessions?

i. The manifest is treated as one row per session. Each row is converted into one `SessionSpec`, and each `SessionSpec` is processed independently into one exported session.

ii.
```python
for eid, row in manifest.iterrows():
    ...
    specs.append(
        SessionSpec(
            eid=eid,
            ...
            session_path=session_path,
        )
    )
```

```python
if num_workers == 1:
    for spec in specs:
        session = process_session_worker(spec)
else:
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for session in executor.map(process_session_worker, specs):
```

iii. In `CONVERSION_NOTES.md` Step 4, the AI says the unit of analysis should be the session and that probes should be merged within a session rather than treated as separate recordings.

## 1-d. How are the data split into trials?

i. Trials are taken directly from rows of the session trial table. The script loads `_ibl_trials.table.pqt`, filters rows with a boolean mask, and then uses the remaining rows as the per-session trial list.

ii.
```python
trials = load_trials_table(spec.session_path)
raw_n_trials = len(trials)
trial_mask = compute_trial_mask(trials)
...
masked_trials = trials.loc[trial_mask].reset_index(drop=False)
```

iii. No extra justification beyond the trial-table structure appears in the notes. The AI treats the native trials table as already being trial-split.

## 1-e. How are trials filtered based on quality controls?

i. Trial filtering happens in two stages. First, `compute_trial_mask()` keeps trials with required event columns present, reaction time in `[0.08, 2.0]`, `choice != 0`, and `feedback_times - goCue_times <= 10.0`. Second, after interpolation/binning, the script drops trials lacking wheel coverage, lacking whisker coverage, or having all-zero neural matrices.

ii.
```python
rt = trials["firstMovement_times"] - trials["stimOn_times"]
mask &= rt >= TRIAL_MASK_RT[0]
mask &= rt <= TRIAL_MASK_RT[1]
mask &= (trials["feedback_times"] - trials["goCue_times"]) <= MAX_TRIAL_LEN
mask &= trials["choice"] != 0
for col in required:
    mask &= trials[col].notna().to_numpy()
```

```python
wheel_trials, wheel_mask = interpolate_behavior_trials(wheel_times, wheel_speed, align_times)
whisk_trials, whisk_mask = interpolate_behavior_trials(whisk_times, whisk_values, align_times)
neural_mask = np.array([np.any(trial) for trial in neural_trials], dtype=bool)
combined_mask = wheel_mask & whisk_mask & neural_mask
```

iii. In `CONVERSION_NOTES.md` Steps 4, 5, 6, and 10, the AI says it wanted to follow the Zhang trial mask, add the reference-code `max_trial_len=10.0` rule, require all required modalities, and explicitly remove all-zero neural windows after verification warnings exposed them.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural matrix is derived from `spikes.times.npy` and `spikes.clusters.npy` for each probe. Cluster metadata from `clusters.metrics.pqt`, `clusters.channels.npy`, and `channels.brainLocationIds_ccf_2017.npy` are used for QC and region labels, not for the spike counts themselves.

ii.
```python
spikes_times_file = pick_one_file(pykilo_path, "spikes.times.npy")
spikes_clusters_file = pick_one_file(pykilo_path, "spikes.clusters.npy")
metrics_file = pick_one_file(pykilo_path, "clusters.metrics.pqt")
clusters_channels_file = pick_one_file(pykilo_path, "clusters.channels.npy")
channels_ids_file = pick_one_file(pykilo_path, "channels.brainLocationIds_ccf_2017.npy")
```

```python
selected_times = np.asarray(spikes_times[spike_keep], dtype=np.float64)
selected_clusters = remap[np.asarray(spikes_clusters[spike_keep], dtype=np.int64)]
```

iii. In `CONVERSION_NOTES.md` Step 5, the AI states that `neural` should come from `spikes.times` and `spikes.clusters`, with QC and region metadata coming from the cluster/channel side tables.

## 2-b. How is the `neural` data processed?

i. The AI merges probes within a session, keeps QC-passing clusters, sorts spikes by time, then bins spike counts into 20 ms bins over `[-0.5, 1.5]` seconds around each trial’s `stimOn_times`. The exported neural arrays are spike counts stored as `float16`; the code does not convert counts to firing rates.

ii.
```python
merged_clusters.append(clusters + cluster_offset)
...
order = np.argsort(spike_times, kind="stable")
spike_times = spike_times[order]
spike_clusters = spike_clusters[order]
```

```python
counts, _, cluster_idx = bincount2D(
    spike_times[idx0:idx1],
    spike_clusters[idx0:idx1],
    xbin=binsize,
    xlim=[start, end],
)
...
trial_counts[cluster_idx, : counts.shape[1]] = counts.astype(np.float16)
```

```python
data["neural"].append([trial.astype(np.float16) for trial in session.neural])
```

iii. In `CONVERSION_NOTES.md` Steps 4, 5, and 6, the AI says it chose 20 ms stimulus-onset-aligned spike-count matrices because that matched the executable Zhang pipeline. It also explicitly notes in Step 5 that it planned to export raw binned spike counts rather than standardized values.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Clusters are filtered by `clusters.metrics.label >= 1`. Only spikes from those clusters are kept; sessions with zero surviving good clusters are skipped.

ii.
```python
metrics = pd.read_parquet(metrics_file, columns=["label"])
cluster_labels = metrics["label"].to_numpy()
good_mask = cluster_labels >= label_threshold
...
spike_keep = good_mask[spikes_clusters]
```

```python
if n_clusters_good == 0:
    print(f"[skip] {spec.eid}: no good clusters after QC")
    return None
```

iii. In `CONVERSION_NOTES.md` Steps 4, 5, 6, and 10, the AI repeatedly justifies `label >= 1` as matching the data paper’s well-isolated-neuron count and as a practical way to reduce the export size.

## 2-d. How is the per-trial `neural` data aligned to the event described in the instructions?

i. Neural data are aligned to `stimOn_times`. For each kept trial, the code defines the absolute interval `[stimOn_times - 0.5, stimOn_times + 1.5]` and bins spikes within that interval.

ii.
```python
align_times = masked_trials["stimOn_times"].to_numpy(dtype=np.float64)
neural_trials = bin_spikes_for_trials(
    spike_times,
    spike_clusters,
    n_clusters=n_clusters_good,
    align_times=align_times,
)
```

```python
intervals = np.c_[align_times + window[0], align_times + window[1]]
...
counts, _, cluster_idx = bincount2D(
    spike_times[idx0:idx1],
    spike_clusters[idx0:idx1],
    xbin=binsize,
    xlim=[start, end],
)
```

iii. In `CONVERSION_NOTES.md` Step 4, the AI says it chose stimulus-onset alignment because the executable reference code used `stimOn_times` and the user instruction explicitly required stimulus-onset alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 20 ms bins over a 2-second window, giving 100 bins per trial. No additional temporal rebinning is applied beyond that fixed binning/interpolation step.

ii.
```python
TIME_WINDOW = (-0.5, 1.5)
BINSIZE_S = 0.02
N_BINS = int(round((TIME_WINDOW[1] - TIME_WINDOW[0]) / BINSIZE_S))
```

```python
counts, _, cluster_idx = bincount2D(
    ...,
    xbin=binsize,
    xlim=[start, end],
)
```

iii. In `CONVERSION_NOTES.md` Steps 1, 4, and 5, the AI says it followed the reference code’s `time_window=(-0.5, 1.5)` and `binsize=0.02` settings.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. The actual `time_since_stimulus_onset` vector is synthetic: it is generated from the global constants `TIME_WINDOW`, `BINSIZE_S`, and `N_BINS`. It is implicitly tied to raw `stimOn_times` because trials are aligned with those timestamps, but the values themselves are not read from a raw data column.

ii.
```python
def make_time_input() -> np.ndarray:
    return np.linspace(TIME_WINDOW[0] + BINSIZE_S, TIME_WINDOW[1], N_BINS, dtype=np.float32)
```

```python
align_times = masked_trials["stimOn_times"].to_numpy(dtype=np.float64)
```

iii. In `CONVERSION_NOTES.md` Step 5, the AI says this input should be the common aligned time grid and explicitly planned values like `[-0.48, -0.46, ..., 1.50]` to match its interpolation grid.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The script creates a fixed length-100 vector with `np.linspace(-0.48, 1.5, 100)` and repeats that same vector for every trial. It does not compute bin centers; it uses the right-edge style grid that also drives its behavior interpolation.

ii.
```python
def make_time_input() -> np.ndarray:
    return np.linspace(TIME_WINDOW[0] + BINSIZE_S, TIME_WINDOW[1], N_BINS, dtype=np.float32)
```

```python
input_trial = np.vstack(
    [
        time_input,
        np.full(N_BINS, block_num, dtype=np.float32),
    ]
).astype(np.float32)
```

iii. In `CONVERSION_NOTES.md` Step 5, the AI justifies this as matching the behavior interpolation grid it chose, not as matching the reference code’s bin-center definition.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The AI intends it to be the shared trial grid: it uses one fixed vector per trial and aligns neural, wheel, and whisker data to the same stimulus-onset window. In practice the time vector corresponds to `[-0.48, ..., 1.50]`, while neural binning is defined by window edges.

ii.
```python
align_times = masked_trials["stimOn_times"].to_numpy(dtype=np.float64)
...
time_input = make_time_input()
```

```python
x_rel = np.linspace(window[0] + binsize, window[1], n_bins)
...
outputs.append(interp(align_time + x_rel).astype(np.float32))
```

iii. In `CONVERSION_NOTES.md` Step 10, the AI reports a sanity check against `np.linspace(-0.48, 1.5, 100)` and treats that as the aligned decoder grid.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from the raw `probabilityLeft` trial sequence. The code computes a counter that resets whenever `probabilityLeft` changes.

ii.
```python
def compute_trial_number_in_block(prob_left: np.ndarray) -> np.ndarray:
    counters = np.zeros(len(prob_left), dtype=np.float32)
    ...
    for i in range(1, len(prob_left)):
        if prob_left[i] == prob_left[i - 1]:
            count += 1
        else:
            count = 1
```

iii. In `CONVERSION_NOTES.md` Step 5, the AI says block identity must be recovered from `probabilityLeft` because there is no explicit block index.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The AI computes the count on the full unfiltered trial sequence, resets when `probabilityLeft` changes, starts counting at `1`, then looks up those values for the kept trials and broadcasts each value across the 100 time bins of that trial.

ii.
```python
count = 1
counters[0] = count
for i in range(1, len(prob_left)):
    if prob_left[i] == prob_left[i - 1]:
        count += 1
    else:
        count = 1
    counters[i] = count
```

```python
block_vals = block_trial_number[masked_keep["index"].to_numpy()]
...
np.full(N_BINS, block_num, dtype=np.float32)
```

iii. In `CONVERSION_NOTES.md` Step 5, the AI explicitly justifies computing block counts before filtering so dropped trials still advance the latent block progression.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. Choice comes directly from the trial table column `choice`.

ii.
```python
choice_vals = map_choice_to_binary(masked_keep["choice"].to_numpy(dtype=np.float64))
```

```python
def map_choice_to_binary(choice_values: np.ndarray) -> np.ndarray:
    mapped[choice_values == 1] = 0   # left
    mapped[choice_values == -1] = 1  # right
```

iii. In `CONVERSION_NOTES.md` Steps 4 and 5, the AI says it follows the raw IBL choice coding and then remaps it to the user-required left/right categories.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The script filters out `choice == 0` trials earlier, then remaps `choice == 1` to `0` (left) and `choice == -1` to `1` (right). It repeats the categorical value across all 100 bins of the trial.

ii.
```python
mask &= trials["choice"] != 0
```

```python
mapped[choice_values == 1] = 0
mapped[choice_values == -1] = 1
...
choice.append(np.full(N_BINS, choice_val, dtype=np.int16))
```

iii. In `CONVERSION_NOTES.md` Step 5, the AI says this remapping was required by the decoder task specification.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. Prior comes directly from the trial table column `probabilityLeft`.

ii.
```python
prior_vals = map_prior_to_categorical(masked_keep["probabilityLeft"].to_numpy(dtype=np.float64))
```

```python
mapped[np.isclose(prob_left, 0.2)] = 0
mapped[np.isclose(prob_left, 0.5)] = 1
mapped[np.isclose(prob_left, 0.8)] = 2
```

iii. In `CONVERSION_NOTES.md` Steps 4 and 5, the AI says it follows the task’s requested categorical mapping for block prior.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The code remaps `0.2 -> 0`, `0.5 -> 1`, and `0.8 -> 2`, verifies that no other values remain, and repeats the result across all 100 bins of the trial.

ii.
```python
def map_prior_to_categorical(prob_left: np.ndarray) -> np.ndarray:
    mapped = np.full(prob_left.shape, -1, dtype=np.int16)
    mapped[np.isclose(prob_left, 0.2)] = 0
    mapped[np.isclose(prob_left, 0.5)] = 1
    mapped[np.isclose(prob_left, 0.8)] = 2
```

```python
prior.append(np.full(N_BINS, prior_val, dtype=np.int16))
```

iii. In `CONVERSION_NOTES.md` Step 5, the AI says this categorical remapping was required by the user’s decoder-output specification.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. Wheel speed is derived from `_ibl_wheel.position.npy` and `_ibl_wheel.timestamps.npy`.

ii.
```python
wheel_pos_file = pick_one_file(session_path / "alf", "_ibl_wheel.position.npy")
wheel_ts_file = pick_one_file(session_path / "alf", "_ibl_wheel.timestamps.npy")
...
pos = np.asarray(np.load(wheel_pos_file), dtype=np.float64)
ts = np.asarray(np.load(wheel_ts_file), dtype=np.float64)
```

iii. In `CONVERSION_NOTES.md` Steps 5 and 6, the AI says it followed the reference wheel-loading path using the bundled wheel utilities from `ibllib`.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The code interpolates wheel position to 1000 Hz with `interpolate_position`, differentiates and filters it with `velocity_filtered`, takes the absolute value to make speed, then linearly interpolates that speed trace onto each trial’s common 20 ms stimulus-onset-aligned grid.

ii.
```python
pos_interp, ts_interp = interpolate_position(ts, pos, freq=1000)
vel, _ = velocity_filtered(pos_interp, 1000)
return ts_interp, np.abs(vel)
```

```python
interp = interp1d(ts, vals, kind="linear", fill_value="extrapolate")
outputs.append(interp(align_time + x_rel).astype(np.float32))
```

iii. In `CONVERSION_NOTES.md` Step 6, the AI says it intentionally reused the wheel utilities from the bundled `ibllib` tree to match the reference wheel preprocessing.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Wheel speed is discretized into three bins using global tertile edges computed across all valid wheel samples from all kept sessions and trials, not session-specific thresholds.

ii.
```python
wheel_edges = compute_tertile_edges(
    trial_values for session in processed for trial_values in session.wheel_cont
)
```

```python
def compute_tertile_edges(values: Iterable[np.ndarray]) -> tuple[float, float]:
    ...
    q1, q2 = np.quantile(concat, [1 / 3, 2 / 3])
```

```python
discretize_three_bins(wheel_cont, wheel_edges)
```

iii. In `CONVERSION_NOTES.md` Step 5, the AI explicitly justifies global tertiles as giving “a common categorical meaning across sessions.”

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. Wheel speed is aligned by interpolating onto the same 100-point stimulus-onset-aligned trial grid used for the exported time input. Trials whose wheel data do not cover the full window within one bin are dropped.

ii.
```python
idx_beg = np.searchsorted(target_times, align_times + window[0], side="right")
idx_end = np.searchsorted(target_times, align_times + window[1], side="left")
...
if np.abs(start - ts[0]) > binsize or np.abs(end - ts[-1]) > binsize:
    outputs.append(None)
    continue
interp = interp1d(ts, vals, kind="linear", fill_value="extrapolate")
outputs.append(interp(align_time + x_rel).astype(np.float32))
```

iii. In `CONVERSION_NOTES.md` Steps 4, 5, and 6, the AI says it used the common stimulus-onset grid for all streams and required full modality coverage for retained trials.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. Whisker motion energy is derived from `leftCamera.ROIMotionEnergy.npy` or, if that is absent, `rightCamera.ROIMotionEnergy.npy`, together with the corresponding `*Camera.times.npy` file.

ii.
```python
for camera in ("left", "right"):
    me_file = pick_one_file(session_path / "alf", f"{camera}Camera.ROIMotionEnergy.npy")
    times_file = pick_one_file(session_path / "alf", f"*{camera}Camera.times.npy")
```

iii. In `CONVERSION_NOTES.md` Steps 4, 5, and 10, the AI says it deliberately used left-camera whisker motion energy first and right-camera fallback second to match the reference logic.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The AI uses the released motion-energy values as-is, optionally truncates values and times to the shorter length if they disagree, then linearly interpolates the trace onto the common 20 ms stimulus-onset-aligned grid for each trial.

ii.
```python
values = np.asarray(np.load(me_file), dtype=np.float64)
times = np.asarray(np.load(times_file), dtype=np.float64)
if len(values) != len(times):
    n = min(len(values), len(times))
    values = values[:n]
    times = times[:n]
```

```python
interp = interp1d(ts, vals, kind="linear", fill_value="extrapolate")
outputs.append(interp(align_time + x_rel).astype(np.float32))
```

iii. In `CONVERSION_NOTES.md` Step 6, the AI describes whisker processing as raw motion-energy loading plus interpolation onto the shared grid; no additional filtering or normalization is justified there.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Whisker motion energy is discretized into three bins using global tertile edges computed across all valid whisker samples from all kept sessions and trials, not session-specific thresholds.

ii.
```python
whisker_edges = compute_tertile_edges(
    trial_values for session in processed for trial_values in session.whisker_cont
)
```

```python
def discretize_three_bins(values: np.ndarray, edges: tuple[float, float]) -> np.ndarray:
    return np.digitize(values, bins=np.array(edges, dtype=np.float32), right=False).astype(np.int16)
```

iii. In `CONVERSION_NOTES.md` Step 5, the AI explicitly says it chose global tertiles for whisker motion energy for the same cross-session-consistency reason as wheel speed.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Whisker motion energy is aligned by interpolating it onto the same 100-point stimulus-onset-aligned trial grid used for wheel and the time input. Trials without adequate whisker coverage are dropped.

ii.
```python
whisk_trials, whisk_mask = interpolate_behavior_trials(whisk_times, whisk_values, align_times)
combined_mask = wheel_mask & whisk_mask & neural_mask
```

```python
x_rel = np.linspace(window[0] + binsize, window[1], n_bins)
...
outputs.append(interp(align_time + x_rel).astype(np.float32))
```

iii. In `CONVERSION_NOTES.md` Steps 4, 5, and 6, the AI says all decoder streams should share one stimulus-onset-aligned grid and that whisker availability is required for a kept trial.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing or malformed data are mostly handled by dropping or skipping them. The script skips sessions missing local files or lacking good clusters, drops trials with missing required events, drops trials without wheel/whisker coverage, drops all-zero neural trials, skips sessions with fewer than two surviving trials, and truncates whisker arrays to the shorter of values/times if their lengths differ.

ii.
```python
if not session_path.exists():
    missing.append(eid)
    continue
```

```python
if len(values) != len(times):
    n = min(len(values), len(times))
    values = values[:n]
    times = times[:n]
```

```python
if n_clusters_good == 0:
    ...
if len(masked_trials) < 2:
    ...
if combined_mask.sum() < 2:
    ...
```

iii. In `CONVERSION_NOTES.md` Steps 6, 9, and 10, the AI says it preferred skipping unusable sessions and dropping unusable trials, and it added explicit all-zero-neural filtering after the verifier warned about those cases.

## 10-a. What are the most time-consuming steps of the code?

i. The AI identifies session processing, especially trial-by-trial spike binning, as the main hot path. It also notes that direct full-dataset processing would be too slow without session-level parallelism.

ii.
```python
for idx0, idx1, (start, end) in zip(idx_starts, idx_ends, intervals):
    trial_counts = np.zeros((n_clusters, n_bins), dtype=np.float16)
    if idx1 > idx0:
        counts, _, cluster_idx = bincount2D(...)
```

```python
with ThreadPoolExecutor(max_workers=num_workers) as executor:
    for session in executor.map(process_session_worker, specs):
```

iii. In `CONVERSION_NOTES.md` Steps 6 and 7, the AI explicitly says “Trial-by-trial spike binning is the main hot path” and motivates thread-level parallelism as a speed fix.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The obvious vectorization candidates are the per-trial loops in spike binning, behavior interpolation, and final per-trial array assembly.

ii.
```python
for idx0, idx1, (start, end) in zip(idx_starts, idx_ends, intervals):
    ...
    results.append(trial_counts)
```

```python
for i, align_time in enumerate(align_times):
    ...
    outputs.append(interp(align_time + x_rel).astype(np.float32))
```

```python
for block_num, choice_val, prior_val in zip(block_vals, choice_vals, prior_vals, strict=True):
    ...
    inputs.append(input_trial)
    choice.append(np.full(N_BINS, choice_val, dtype=np.int16))
```

iii. The notes do not give a deeper justification here beyond calling the trial-by-trial spike path the main bottleneck and adding session-level parallelism instead of further vectorizing those loops.

## 10-c. What processing does the code repeat multiple times?

i. The code repeats several small operations: repeated recursive file searches via `pick_one_file`, repeated per-trial interpolation calls, repeated per-trial spike binning loops, and repeated discretization of wheel/whisker traces when building outputs. It also recomputes trial-level constant arrays inside Python loops.

ii.
```python
def pick_one_file(base: Path, pattern: str) -> Path | None:
    matches = list(base.rglob(pattern))
```

```python
wheel_disc = discretize_three_bins(wheel_cont, wheel_edges)
whisk_disc = discretize_three_bins(whisk_cont, whisker_edges)
```

```python
output_trial = np.vstack(
    [
        choice,
        prior,
        discretize_three_bins(wheel_cont, wheel_edges),
        discretize_three_bins(whisk_cont, whisker_edges),
    ]
).astype(np.int16)
```

iii. I did not find an explicit justification for this repeated work in `CONVERSION_NOTES.md`. The notes focus on higher-level speedups rather than defending these repeated local operations.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code keeps continuous `wheel_cont` and `whisker_cont` traces in memory even though the final dataset only stores discretized versions. It also carries bookkeeping fields such as `kept_trial_indices`, `whisker_source`, `n_clusters_total`, and `raw_n_trials` that are used only for logs/metadata construction. Optional plotting also recomputes discretized traces purely for inspection.

ii.
```python
return ProcessedSession(
    ...
    wheel_cont=[np.asarray(x, dtype=np.float32) for x in wheel_keep],
    whisker_cont=[np.asarray(x, dtype=np.float32) for x in whisk_keep],
    kept_trial_indices=masked_keep["index"].to_numpy(dtype=np.int32),
    whisker_source=whisk_source,
    n_clusters_total=n_clusters_total,
    raw_n_trials=raw_n_trials,
)
```

```python
output_trial = np.vstack(
    [
        choice,
        prior,
        discretize_three_bins(wheel_cont, wheel_edges),
        discretize_three_bins(whisk_cont, whisker_edges),
    ]
).astype(np.int16)
```

iii. No explicit justification appears in the notes beyond the need to compute global tertile edges and optional processing plots.
