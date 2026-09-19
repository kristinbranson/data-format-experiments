# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent does not use the `ONE` API or IBL loaders. It reads a local cache manifest parquet file, constructs session paths directly from manifest metadata, recursively finds versioned ALF files on disk, and then loads tables with `pandas.read_parquet()` and arrays with `numpy.load()`.

ii. ```python
def load_session_manifest() -> pd.DataFrame:
    for manifest in MANIFEST_FILES:
        if manifest.exists():
            return pd.read_parquet(manifest)
```

```python
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
def pick_one_file(base: Path, pattern: str) -> Path | None:
    matches = list(base.rglob(pattern))
    ...
    return matches[-1]
```

iii. In `CONVERSION_NOTES.md` and the trajectory, the agent says the data are an IBL ONE cache but that `SessionLoader` dependencies were not installed cleanly, so it switched to direct local-file loading while reusing only lightweight wheel utilities from bundled code.

## 1-b. How are the data split into subjects?

i. Subjects are taken from the manifest row for each session. After session processing, the converted dataset builds `subjects` as sorted unique subject names from kept sessions and `subject_idx` as an index into that list.

ii. ```python
specs.append(
    SessionSpec(
        eid=eid,
        lab=str(row["lab"]),
        subject=str(row["subject"]),
        ...
    )
)
```

```python
subjects = sorted({s.subject for s in sessions})
subject_to_idx = {s: i for i, s in enumerate(subjects)}
...
"subject_idx": np.array([subject_to_idx[s.subject] for s in sessions], dtype=np.int16),
```

iii. The notes say the manifest gives stable subject IDs directly, so no subject parsing from filenames was needed.

## 1-c. How are the data split into sessions?

i. Sessions are the unit of the manifest. Each manifest row becomes one `SessionSpec` if its session directory exists, and each surviving `SessionSpec` is processed independently.

ii. ```python
for eid, row in manifest.iterrows():
    session_path = ...
    if not session_path.exists():
        missing.append(eid)
        continue
    specs.append(SessionSpec(...))
```

```python
for future in futures:
    session = future.result()
    if session is not None:
        sessions.append(session)
```

iii. The agent’s notes say the release manifest is the authoritative session index, so nothing extra is needed to split a session further.

## 1-d. How are the data split into trials?

i. Trials come from rows of the session trial table `_ibl_trials.table.pqt`. The agent loads the whole table, applies a boolean mask, then treats each remaining row as one trial aligned on that row’s `stimOn_times`.

ii. ```python
def load_trials_table(session_path: Path) -> pd.DataFrame:
    trial_file = pick_one_file(session_path / "alf", "_ibl_trials.table.pqt")
    return pd.read_parquet(trial_file)
```

```python
masked_trials = trials.loc[trial_mask].reset_index(drop=False)
align_times = masked_trials["stimOn_times"].to_numpy(dtype=np.float64)
```

iii. The notes describe the trials table as the canonical per-trial structure already present in the raw release.

## 1-e. How are trials filtered based on quality controls?

i. The agent filters trials with a session-level mask requiring non-missing `stimOn_times`, `choice`, `feedback_times`, `probabilityLeft`, `firstMovement_times`, and `feedbackType`; reaction time between 0.08 and 2.0 s; `choice != 0`; and `feedback_times - goCue_times <= 10 s`. After that it further drops trials whose wheel or whisker traces do not cover the full aligned window, and it drops trials whose neural window is entirely zero.

ii. ```python
required = [
    "stimOn_times",
    "choice",
    "feedback_times",
    "probabilityLeft",
    "firstMovement_times",
    "feedbackType",
]
...
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

iii. The notes say this was chosen to mirror the Zhang repository’s `load_trials_and_mask()` logic, plus later full-window alignment checks and the Step 10 fix that removed all-zero neural trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural matrix is derived from `spikes.times.npy` and `spikes.clusters.npy` from each probe. Cluster metrics and channel location arrays are used only to decide which clusters to keep and what brain-region labels to attach.

ii. ```python
spikes_times_file = pick_one_file(pykilo_path, "spikes.times.npy")
spikes_clusters_file = pick_one_file(pykilo_path, "spikes.clusters.npy")
metrics_file = pick_one_file(pykilo_path, "clusters.metrics.pqt")
clusters_channels_file = pick_one_file(pykilo_path, "clusters.channels.npy")
channels_ids_file = pick_one_file(pykilo_path, "channels.brainLocationIds_ccf_2017.npy")
```

iii. The notes explicitly map `spikes.times` and `spikes.clusters` to `neural`, with QC metadata used for neuron inclusion and region labels.

## 2-b. How is the `neural` data processed?

i. The agent merges probes within a session, renumbers clusters across probes, sorts spikes by time, and bins spike counts into a 2 s stimulus-aligned window with 20 ms bins. Unlike the human reference, it stores raw spike counts as `float16`; it does not divide by bin width to convert them to firing rate in Hz.

ii. ```python
merged_clusters.append(clusters + cluster_offset)
...
order = np.argsort(spike_times, kind="stable")
spike_times = spike_times[order]
spike_clusters = spike_clusters[order]
```

```python
trial_counts = np.zeros((n_clusters, n_bins), dtype=np.float16)
...
counts, _, cluster_idx = bincount2D(
    spike_times[idx0:idx1],
    spike_clusters[idx0:idx1],
    xbin=binsize,
    xlim=[start, end],
)
...
trial_counts[cluster_idx, : counts.shape[1]] = counts.astype(np.float16)
```

iii. The notes say the export would use “raw binned spike counts” and focus on a decoder-ready dense pickle, rather than the reference’s firing-rate representation.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Clusters are filtered only by `clusters.metrics.label >= 1`. The code does not remove `void` units and does not remap cluster acronyms to Beryl before saving brain-region labels; it uses raw atlas acronyms from channel brain-location IDs.

ii. ```python
metrics = pd.read_parquet(metrics_file, columns=["label"])
cluster_labels = metrics["label"].to_numpy()
good_mask = cluster_labels >= label_threshold
selected_cluster_ids = np.flatnonzero(good_mask)
```

```python
cluster_region_ids[valid_channel] = channel_ids[cluster_channels[valid_channel]]
cluster_regions = br.id2acronym(cluster_region_ids)
```

iii. The notes justify `label >= 1` as matching the paper’s well-isolated neuron count and keeping the export tractable. The notes also say raw acronyms were preserved to avoid collapsing atlas levels.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial is aligned to `stimOn_times`, using a fixed window from `-0.5` s to `+1.5` s around stimulus onset.

ii. ```python
TIME_WINDOW = (-0.5, 1.5)
...
align_times = masked_trials["stimOn_times"].to_numpy(dtype=np.float64)
intervals = np.c_[align_times + window[0], align_times + window[1]]
```

iii. The notes say the agent chose stimulus-onset alignment because that is the user’s explicit instruction and also matches the executable Zhang preprocessing path.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 20 ms bins over a 2 s window, giving 100 time bins. No further temporal rebinning is applied.

ii. ```python
BINSIZE_S = 0.02
N_BINS = int(round((TIME_WINDOW[1] - TIME_WINDOW[0]) / BINSIZE_S))
```

iii. The notes justify 20 ms as matching the executable reference code and the main decoder description (`T = 100` over 2 s).

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. The input is conceptually derived from the `stimOn_times` alignment event, but the stored per-trial values are generated from fixed constants rather than recalculated from each raw trial row.

ii. ```python
align_times = masked_trials["stimOn_times"].to_numpy(dtype=np.float64)
...
def make_time_input() -> np.ndarray:
    return np.linspace(TIME_WINDOW[0] + BINSIZE_S, TIME_WINDOW[1], N_BINS, dtype=np.float32)
```

iii. The notes say this variable is a constructed time grid for the aligned window, with the agent intentionally using a shared stimulus-onset-aligned grid for every trial.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The code creates a synthetic 100-point time grid from `-0.48` s to `1.5` s with `np.linspace`, then repeats that same vector for every trial.

ii. ```python
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

iii. The notes say the agent deliberately used `[-0.48, -0.46, ..., 1.50]` to match its behavior interpolation grid.

## 3-c. How is `input` *Time since stimulus onset* aligned with the neural data?

i. The agent intends the time input to be the same grid used for the aligned behavior traces and to correspond to the neural trial bins. In practice, the neural bins are defined by the `[-0.5, 1.5]` window and 20 ms width, while the saved time vector is the right-edge-like grid `[-0.48, ..., 1.5]`, not the bin centers used in the human reference.

ii. ```python
intervals = np.c_[align_times + window[0], align_times + window[1]]
...
counts, _, cluster_idx = bincount2D(..., xbin=binsize, xlim=[start, end])
```

```python
x_rel = np.linspace(window[0] + binsize, window[1], n_bins)
...
outputs.append(interp(align_time + x_rel).astype(np.float32))
```

iii. The notes justify using one common stimulus-onset-aligned grid for neural, behavior, and time input so that all streams share a single time axis.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from the raw `probabilityLeft` sequence in the trial table.

ii. ```python
def compute_trial_number_in_block(prob_left: np.ndarray) -> np.ndarray:
```

```python
block_trial_number = compute_trial_number_in_block(trials["probabilityLeft"].to_numpy())
```

iii. The notes say the block identity is not stored directly, so it is reconstructed from changes in `probabilityLeft`.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The code scans the unfiltered `probabilityLeft` sequence and increments a counter while the block value stays the same, resetting when it changes. The count is one-based, not zero-based, and kept trials inherit their original block count after filtering.

ii. ```python
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
```

iii. The notes justify computing it before filtering so dropped trials still advance the latent within-block position.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. Choice is derived from the raw `choice` column in the trials table.

ii. ```python
choice_vals = map_choice_to_binary(masked_keep["choice"].to_numpy(dtype=np.float64))
```

iii. The notes say the sign convention was spot-checked from the raw data before finalizing the mapping.

## 5-b. What processing is involved in computing `output` *Choice*?

i. Raw `choice == 1` is mapped to `0` for left and `choice == -1` is mapped to `1` for right. The value is then repeated across all 100 time bins of that trial.

ii. ```python
mapped[choice_values == 1] = 0   # left
mapped[choice_values == -1] = 1  # right
```

```python
choice.append(np.full(N_BINS, choice_val, dtype=np.int16))
```

iii. The notes say this was chosen to satisfy the decoder task’s explicit `left = 0, right = 1` requirement.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It is derived from the raw `probabilityLeft` column in the trials table.

ii. ```python
prior_vals = map_prior_to_categorical(masked_keep["probabilityLeft"].to_numpy(dtype=np.float64))
```

iii. The notes identify `probabilityLeft` as the released block-prior variable.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The code maps `0.2 -> 0`, `0.5 -> 1`, and `0.8 -> 2`, then repeats that category across all 100 bins in the trial.

ii. ```python
mapped[np.isclose(prob_left, 0.2)] = 0
mapped[np.isclose(prob_left, 0.5)] = 1
mapped[np.isclose(prob_left, 0.8)] = 2
```

```python
prior.append(np.full(N_BINS, prior_val, dtype=np.int16))
```

iii. The notes say this mapping follows the user’s requested categorical coding.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. Wheel speed is derived from `_ibl_wheel.position.npy` and `_ibl_wheel.timestamps.npy`.

ii. ```python
wheel_pos_file = pick_one_file(session_path / "alf", "_ibl_wheel.position.npy")
wheel_ts_file = pick_one_file(session_path / "alf", "_ibl_wheel.timestamps.npy")
```

iii. The notes say the wheel stream should follow IBL’s wheel-processing utilities rather than inventing a new speed definition.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The agent loads wheel position and timestamps, averages two-column timestamps if present, interpolates position to 1000 Hz, computes filtered velocity with `velocity_filtered`, takes the absolute value to get speed, and linearly interpolates that speed onto each trial’s aligned 100-bin grid.

ii. ```python
if ts.ndim == 2 and ts.shape[1] == 2:
    ts = ts.mean(axis=1)
pos_interp, ts_interp = interpolate_position(ts, pos, freq=1000)
vel, _ = velocity_filtered(pos_interp, 1000)
return ts_interp, np.abs(vel)
```

```python
interp = interp1d(ts, vals, kind="linear", fill_value="extrapolate")
outputs.append(interp(align_time + x_rel).astype(np.float32))
```

iii. The notes say the agent reused bundled IBL wheel utilities so the velocity definition matched reference logic even though loading was done from files directly.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. It is thresholded into three categories using dataset-wide tertile edges computed from all kept wheel samples across all sessions, not session-by-session percentiles.

ii. ```python
def compute_tertile_edges(values: Iterable[np.ndarray]) -> tuple[float, float]:
    ...
    q1, q2 = np.quantile(concat, [1 / 3, 2 / 3])
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

iii. The notes explicitly justify global tertiles as giving common category meaning across sessions.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The wheel trace is aligned to `stimOn_times` and interpolated onto the same 100-step relative grid the agent uses for time input and whisker output: `[-0.48, -0.46, ..., 1.50]` s relative to stimulus onset.

ii. ```python
x_rel = np.linspace(window[0] + binsize, window[1], n_bins)
...
wheel_trials, wheel_mask = interpolate_behavior_trials(wheel_times, wheel_speed, align_times)
```

iii. The notes say all decoder streams should share one stimulus-onset-aligned grid because the task requested a single alignment event.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It is derived from `<side>Camera.ROIMotionEnergy.npy` and the matching camera times file, with the left camera preferred and the right camera used as fallback.

ii. ```python
for camera in ("left", "right"):
    me_file = pick_one_file(session_path / "alf", f"{camera}Camera.ROIMotionEnergy.npy")
    times_file = pick_one_file(session_path / "alf", f"*{camera}Camera.times.npy")
    ...
    return times, values, camera
```

iii. The notes say this left-then-right fallback was chosen to match the reference code and maximize session retention.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The released motion-energy values are used directly, except that mismatched value/time arrays are truncated to a common minimum length. Non-finite samples are removed, and the trace is linearly interpolated onto the aligned 100-bin trial grid without further filtering or normalization.

ii. ```python
values = np.asarray(np.load(me_file), dtype=np.float64)
times = np.asarray(np.load(times_file), dtype=np.float64)
if len(values) != len(times):
    n = min(len(values), len(times))
    values = values[:n]
    times = times[:n]
```

```python
valid_source = np.isfinite(target_times) & np.isfinite(target_values)
target_times = target_times[valid_source]
target_values = target_values[valid_source]
...
interp = interp1d(ts, vals, kind="linear", fill_value="extrapolate")
```

iii. The notes describe whisker motion energy as a released continuous trace that should be aligned and then discretized, not otherwise transformed.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Like wheel speed, whisker motion energy is thresholded with one pair of global tertile edges computed across all kept sessions and trials.

ii. ```python
q1, q2 = np.quantile(concat, [1 / 3, 2 / 3])
```

```python
discretize_three_bins(whisk_cont, whisker_edges)
```

iii. The notes justify this as a global discretization scheme with shared semantics across sessions.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. The whisker trace is aligned to `stimOn_times` and interpolated onto the same `[-0.48, ..., 1.50]` relative grid used for wheel and the time input.

ii. ```python
whisk_trials, whisk_mask = interpolate_behavior_trials(whisk_times, whisk_values, align_times)
...
x_rel = np.linspace(window[0] + binsize, window[1], n_bins)
```

iii. The notes say the user-required common stimulus-onset alignment overrode the paper prose that sometimes used movement alignment for dynamic behaviors.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing sessions are skipped if their directory is absent. Missing probe assets or missing wheel/whisker files cause the session to be skipped. Mismatched whisker array lengths are silently truncated to the shorter length. Non-finite behavioral samples are removed. Trials with missing required events, incomplete behavioral coverage, or all-zero neural windows are dropped. Sessions with no good clusters or fewer than two kept trials are skipped.

ii. ```python
if not session_path.exists():
    missing.append(eid)
    continue
```

```python
if any(x is None for x in required):
    raise FileNotFoundError(...)
...
except FileNotFoundError as exc:
    print(f"[skip] {spec.eid}: {exc}")
    return None
```

```python
if len(values) != len(times):
    n = min(len(values), len(times))
    values = values[:n]
    times = times[:n]
```

iii. The notes and Step 10 review say the agent preferred skipping malformed partial data over emitting partial sessions, and it added the all-zero-neural-window drop after validation warnings.

## 10-a. What are the most time-consuming steps of the code?

i. The dominant cost is loading and materializing large spike arrays from disk, followed by per-trial spike binning for each session. Behavioral interpolation is a smaller but repeated per-trial cost.

ii. ```python
spikes_times = np.load(spikes_times_file, mmap_mode="r")
spikes_clusters = np.load(spikes_clusters_file, mmap_mode="r")
```

```python
for idx0, idx1, (start, end) in zip(idx_starts, idx_ends, intervals):
    trial_counts = np.zeros((n_clusters, n_bins), dtype=np.float16)
    ...
```

iii. The notes say spike I/O was expected to dominate runtime, and the full run log shows some high-unit sessions taking tens of seconds each.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The clearest candidates are the per-trial spike-binning loop, the per-trial behavior interpolation loop, and the Python loop used to compute trial number within block. The per-trial assembly loop that builds decoder inputs and outputs could also be reduced.

ii. ```python
for idx0, idx1, (start, end) in zip(idx_starts, idx_ends, intervals):
    ...
```

```python
for i, align_time in enumerate(align_times):
    ...
```

```python
for i in range(1, len(prob_left)):
    ...
```

iii. The agent did not present a strong efficiency justification here; the code favors straightforward Python loops over a more vectorized implementation.

## 10-c. What processing does the code repeat multiple times?

i. The code repeatedly performs recursive filesystem searches with `pick_one_file()` for each needed asset instead of resolving a session’s files once. It also computes continuous wheel and whisker traces for every kept trial, keeps them in memory, and only later performs the final discretization in a separate pass during dataset assembly.

ii. ```python
def pick_one_file(base: Path, pattern: str) -> Path | None:
    matches = list(base.rglob(pattern))
```

```python
wheel_trials, wheel_mask = interpolate_behavior_trials(...)
whisk_trials, whisk_mask = interpolate_behavior_trials(...)
...
output_trial = np.vstack(
    [
        choice,
        prior,
        discretize_three_bins(wheel_cont, wheel_edges),
        discretize_three_bins(whisk_cont, whisker_edges),
    ]
)
```

iii. The notes justify keeping continuous traces until global tertile edges are known, but they do not justify the repeated recursive file scans.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It computes and stores full continuous wheel and whisker traces in each `ProcessedSession`, even though the final pickle only saves the discretized three-bin outputs. It also carries bookkeeping like `kept_trial_indices`, `n_clusters_total`, and `whisker_source` through processing without saving most of it to the final dataset.

ii. ```python
return ProcessedSession(
    ...
    wheel_cont=[np.asarray(x, dtype=np.float32) for x in wheel_keep],
    whisker_cont=[np.asarray(x, dtype=np.float32) for x in whisk_keep],
    ...
    kept_trial_indices=masked_keep["index"].to_numpy(dtype=np.int32),
    whisker_source=whisk_source,
    n_clusters_total=n_clusters_total,
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

iii. The notes say the continuous traces were kept so global discretization edges could be computed later, but those continuous traces are discarded from the exported dataset.
