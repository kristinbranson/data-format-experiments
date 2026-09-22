# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI discovers every `/app/data/sub-*/*.nwb` file in sorted order. It first opens every file with `h5py` to inventory subjects and regions, then opens each file again to convert it. NWB groups for units, trials, behavioral events, tongue tracking, and subject metadata are read directly.

ii.
```python
paths = sorted(glob.glob(str(DATA_ROOT / "sub-*" / "*.nwb")))
subjects, brain_regions = discover_inventory(paths)
for path in paths:
    result = convert_session(path, subject_lookup, region_lookup, ...)
```
```python
with h5py.File(path, "r") as nwb:
    units = nwb["units"]
    trial_table = nwb["intervals/trials"]
```

iii. The notes say there are 174 NWB files in 28 subject directories and one file per session. Sorted paths provide deterministic ordering; direct HDF5 reads were chosen for speed and checked against the NWB layout.

## 1-b. How are the data split into subjects?

i. Subject IDs are read from `general/subject/subject_id`, checked against the `sub-<id>` directory, globally sorted, and converted to integer indices for sessions.

ii.
```python
subject = decode_scalar(nwb["general/subject/subject_id"][()])
folder_subject = Path(path).parent.name.removeprefix("sub-")
if subject != folder_subject:
    raise ValueError(...)
subjects.add(subject)
```
```python
subject_lookup = {name: i for i, name in enumerate(subjects)}
"subject_idx": np.asarray([x["subject_idx"] for x in converted], dtype=np.int32),
```

iii. The NWB subject field is treated as canonical, with the folder comparison serving as a consistency check. The notes report 28 mice.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. Sessions remain in lexicographic path order; the only file omitted is the session with no classifier-curated units.

ii.
```python
def convert_session(path: str, ...):
    with h5py.File(path, "r") as nwb:
        ...
        if len(good_indices) == 0:
            return None
```

iii. The data inventory established that the release contains one recording session per NWB file. The notes reconcile 174 files with 173 usable sessions.

## 1-d. How are the data split into trials?

i. Trial rows come from `intervals/trials`. The AI assumes the recorded neural trials are the leading rows, using the width of `units/is_good_trials`, and takes corresponding leading go cues and trial starts. Retained session arrays are finally sliced into lists of per-trial arrays.

ii.
```python
n_trials_recorded = stability.shape[1]
go_all = nwb["acquisition/BehavioralEvents/go_start_times/timestamps"][:n_trials_recorded]
trial_starts = trial_table["start_time"][:n_trials_recorded]
```
```python
"neural": [rates[i] for i in range(n_trials)],
"input": [inputs[i] for i in range(n_trials)],
"output": [outputs[i] for i in range(n_trials)],
```

iii. The notes state that `is_good_trials` and observation-interval lengths identify the recorded leading trials. Shape and count checks are used to detect disagreement.

## 1-e. How are trials filtered based on quality controls?

i. The AI keeps only leading recorded trials for which every retained unit is marked good in `is_good_trials` and a finite tone exists. It then drops any four-second trial window with no spike from any retained neuron. It keeps photostimulation, early-lick, ignore, auto-water, and free-water trials, and requires at least two trials per session.

ii.
```python
stable_mask = np.all(stability, axis=0)
coverage_mask = np.isfinite(tone_all)
keep = stable_mask & coverage_mask
```
```python
neural_present = np.any(rates != 0, axis=(1, 2))
rates = rates[neural_present]
```

iii. The AI argues the paper's regular-trial mask conflicts with required decoder classes. It uses all-unit stability as published QC and treats entirely silent population windows as missing ephys caused by erroneous terminal observation intervals. The notes explicitly say water trials were retained.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from ragged `units/spike_times` for units whose `units/classification` equals `good`, plus go-cue timestamps used to position trial bins.

ii.
```python
classification = decode_array(units["classification"][:])
good_indices = np.flatnonzero(classification == "good")
spike_ends = units["spike_times_index"][:]
spike_values = units["spike_times"]
absolute_edges = go[:, None] + BIN_EDGES[None, :]
```

iii. The notes identify spike times as the available neural representation and the classifier label as the exported verdict of the spike-sorting QC procedure.

## 2-b. How is the `neural` data processed?

i. For each curated unit, spikes are counted in half-open, non-overlapping bins using `searchsorted`; counts are divided by 0.05 seconds to yield float32 firing rates in Hz. There is no smoothing or normalization.

ii.
```python
edge_positions = np.searchsorted(spikes, absolute_edges.ravel(), side="left")
counts = np.diff(edge_positions.reshape(n_trials, N_TIME + 1), axis=1)
rates[:, out_unit, :] = counts / BIN_WIDTH_S
```

iii. This was chosen to match the reference histogram's count-to-Hz behavior while obeying the task's required 50-ms bins. The notes emphasize exact 20-Hz quantization checks.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only `classification == "good"` units are retained. The AI additionally requires all retained units to be stable on a trial; a session with no good units is skipped.

ii.
```python
good_indices = np.flatnonzero(classification == "good")
if len(good_indices) == 0:
    return None
stability = stability_all[good_indices]
stable_mask = np.all(stability, axis=0)
```

iii. The AI treats `classification` as the final published classifier QC and deliberately avoids recreating scalar thresholds or applying a firing-rate cutoff.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Absolute go-cue time is added to fixed relative edges from -2.5 to +1.5 seconds, and absolute spike times are binned on those edges.

ii.
```python
BIN_EDGES = np.arange(OFF_START, OFF_END + BIN_WIDTH_S / 2, BIN_WIDTH_S)
absolute_edges = go[:, None] + BIN_EDGES[None, :]
```

iii. The AI notes that spikes and behavioral events share the NWB session clock, so no clock correction or interpolation is needed.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The output has 80 non-overlapping 50-ms bins from -2.5 to +1.5 seconds. Raw spikes are binned directly; no subsequent rebinning is applied.

ii.
```python
BIN_WIDTH_S = 0.050
BIN_EDGES = np.arange(OFF_START, OFF_END + BIN_WIDTH_S / 2, BIN_WIDTH_S)
N_TIME = len(BIN_CENTERS)
```

iii. The window and resolution come directly from the decoder instructions, overriding the different sliding windows used by the method paper.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It uses trial start times, go-cue timestamps, and `sample_start_times`. The AI selects the first sample event at or after each trial start, provided it occurs before go.

ii.
```python
indices = np.searchsorted(sample_starts, trial_starts, side="left")
tone = sample_starts[indices]
valid = (tone >= trial_starts - 1e-9) & (tone <= go_times + 1e-9)
```

iii. The AI says this avoids positional pairing and explicitly documents its definition as the first sample event between trial start and go.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. For each neural bin center in absolute time, the selected tone time is subtracted, producing elapsed seconds from tone onset.

ii.
```python
centers_all = go_all[:, None] + BIN_CENTERS[None, :]
inputs[:, 0, :] = centers - tone[:, None]
```

iii. The notes describe this as a continuous time-varying trace increasing by 0.05 seconds per bin.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. The input is evaluated at the exact absolute centers of the same go-aligned bins used for spike counts.

ii.
```python
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2
centers_all = go_all[:, None] + BIN_CENTERS[None, :]
inputs[:, 0, :] = centers - tone[:, None]
```

iii. Shared go-aligned bin centers guarantee matching time indices.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. It is derived from trial-table `start_time`, `photostim_onset`, and `photostim_duration`; onset/duration strings including `N/A` are parsed to floats or NaN.

ii.
```python
onset = numeric_or_nan(trial_table["photostim_onset"][:n_trials_recorded])[keep_idx]
duration = numeric_or_nan(trial_table["photostim_duration"][:n_trials_recorded])[keep_idx]
stim_start = kept_trial_starts + onset
stim_stop = stim_start + duration
```

iii. The AI notes onset is relative to trial start and must be put onto the session-absolute clock.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. Each bin is assigned 1 if its center is within the half-open stimulation interval and 0 otherwise. NaN onset/duration naturally yields all zeros.

ii.
```python
inputs[:, 1, :] = (
    np.isfinite(stim_start[:, None]) & np.isfinite(stim_stop[:, None])
    & (centers >= stim_start[:, None]) & (centers < stim_stop[:, None])
).astype(np.float32)
```

iii. The time-varying binary representation follows the requested decoder input and the same half-open convention as neural bins.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Absolute stimulation bounds are compared with the absolute centers of the go-aligned neural bins.

ii.
```python
centers = centers_all[keep]
stim_start = kept_trial_starts + onset
inputs[:, 1, :] = (centers >= stim_start[:, None]) & (centers < stim_stop[:, None])
```

iii. Both are on the NWB session clock, so no additional offset is applied.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is inferred from trial-table `outcome` and `trial_instruction`: hit means instructed side, miss means the opposite side, and ignore means no lick.

ii.
```python
outcomes = decode_array(trial_table["outcome"][:n_trials_recorded])[keep_idx]
instructions = decode_array(trial_table["trial_instruction"][:n_trials_recorded])[keep_idx]
choices = choice_codes(outcomes, instructions)
```

iii. The notes say actual direction is not directly stored but is fully determined by these two columns.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Left, right, and no lick are coded 0, 1, and 2. The per-trial code is broadcast over all 80 bins.

ii.
```python
code = np.full(len(outcomes), 2, dtype=np.int8)
code[hit & (instructions == "left")] = 0
code[hit & (instructions == "right")] = 1
code[miss & (instructions == "right")] = 0
code[miss & (instructions == "left")] = 1
outputs[:, 0, :] = choices[:, None]
```

iii. Broadcasting allows scalar trial outputs and time-varying tongue output to share one `(4, 80)` array.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. It comes directly from the trial table's `outcome` strings.

ii.
```python
outcomes = decode_array(trial_table["outcome"][:n_trials_recorded])[keep_idx]
```

iii. The source categories already match ignore, miss, and hit.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The categories are mapped in the requested order to 0, 1, and 2 and broadcast over time.

ii.
```python
OUTCOME_VALUES = ["ignore", "miss", "hit"]
outcome_lookup = {name: i for i, name in enumerate(OUTCOME_VALUES)}
outcome_codes = np.asarray([outcome_lookup[x] for x in outcomes], dtype=np.int8)
outputs[:, 1, :] = outcome_codes[:, None]
```

iii. Fixed integer coding is required for categorical decoder targets.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. It comes directly from the trial-table `early_lick` labels.

ii.
```python
early_labels = decode_array(trial_table["early_lick"][:n_trials_recorded])[keep_idx]
```

iii. The field explicitly records `no early` or `early`, so no lick-event reconstruction is needed.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. `no early` is 0 and `early` is 1; the code is validated and broadcast over 80 bins.

ii.
```python
if not set(np.unique(early_labels)).issubset({"no early", "early"}):
    raise ValueError(...)
early_codes = (early_labels == "early").astype(np.int8)
outputs[:, 2, :] = early_codes[:, None]
```

iii. This preserves the requested per-trial binary label in the common output tensor.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. It uses timestamps and columns 1 and 2 (y position and likelihood) from `Camera0_side_TongueTracking`.

ii.
```python
tongue_group = nwb["acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking"]
tongue_t = tongue_group["timestamps"][:]
tongue_data = tongue_group["data"][:]
tongue_y = tongue_data[:, 1]
tongue_likelihood = tongue_data[:, 2]
```

iii. The notes identify the side camera as the specified tongue stream and likelihood as the visibility indicator.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Frames are visible only at likelihood at least 0.9. Session percentiles are computed from all visible raw frame y values. At each 50-ms bin center, the latest preceding camera frame is sampled if no more than 10 ms old; otherwise the bin is not visible. No within-bin averaging is performed.

ii.
```python
TONGUE_LIKELIHOOD_THRESHOLD = 0.9
session_visible = np.isfinite(tongue_y) & np.isfinite(tongue_likelihood) \
    & (tongue_likelihood >= TONGUE_LIKELIHOOD_THRESHOLD)
q40, q60 = np.percentile(tongue_y[session_visible], [40, 60])
frame_idx_all = np.searchsorted(tongue_t, centers_all, side="right") - 1
frame_is_current_all = (frame_lag >= -1e-9) & (frame_lag <= 0.010)
```

iii. The AI cites the method code's preceding-frame alignment and says 10 ms tolerates roughly three nominal camera frames. It excludes low-likelihood coordinates as tracking artifacts and maps absent/noncontemporaneous frames to not visible.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Per-session 40th and 60th percentiles of visible raw frames define classes: y below q40 is 0, q40 through q60 inclusive is 1, above q60 is 2, and invisible/missing is 3.

ii.
```python
tongue_codes = np.full((n_trials, N_TIME), 3, dtype=np.int8)
tongue_codes[sampled_visible & (sampled_y < q40)] = 0
tongue_codes[sampled_visible & (sampled_y >= q40) & (sampled_y <= q60)] = 1
tongue_codes[sampled_visible & (sampled_y > q60)] = 2
```

iii. The percentile scope is per session as requested; invisible samples are excluded from percentile estimation and given the explicit fourth class.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The most recent preceding camera frame is sampled at each absolute go-aligned neural bin center, subject to a 10-ms freshness limit.

ii.
```python
centers_all = go_all[:, None] + BIN_CENTERS[None, :]
frame_idx_all = np.searchsorted(tongue_t, centers_all, side="right") - 1
frame_lag = centers_all - tongue_t[safe_frame_idx]
```

iii. The AI argues camera and neural timestamps share the same clock and that preceding-frame sampling follows the reference video alignment utility. Gaps become `not visible` rather than causing trial deletion.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The all-NaN/no-good-unit session is skipped; malformed shapes, mismatched counts, unknown labels, or missing anatomy raise errors. Optional numeric `N/A` becomes NaN. Noncontemporaneous or low-confidence video becomes tongue class 3. Trials marked stable but having an entirely silent population window are dropped as missing ephys.

ii.
```python
try:
    out[i] = float(decode_scalar(value))
except (TypeError, ValueError):
    pass
```
```python
if len(good_indices) == 0:
    return None
neural_present = np.any(rates != 0, axis=(1, 2))
tongue_codes = np.full((n_trials, N_TIME), 3, dtype=np.int8)
```

iii. The notes describe iterative audits of missing video and off-by-one observation intervals. They favor explicit missing categories for video, exclusion for absent neural recording, and hard failures for structural inconsistencies.

## 10-a. What are the most time-consuming steps of the code?

i. The major costs are reading the files twice (inventory and conversion), loading large spike and camera arrays, per-unit spike binning, accumulating the roughly 11-GiB output, and pickle serialization. The full run took about 177 seconds.

ii.
```python
subjects, brain_regions = discover_inventory(paths)
for path in paths:
    result = convert_session(...)
for out_unit, source_unit in enumerate(good_indices):
    edge_positions = np.searchsorted(spikes, absolute_edges.ravel(), side="left")
pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The notes identify naive unit-by-trial loops and repeated HDF5 access as potential bottlenecks, then report vectorizing trials within each unit and reading each unit slice once.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The remaining important loop is over good units; each iteration bins all trials at once. Metadata coding also uses small Python comprehensions. Per-unit vectorization is difficult because spike trains are ragged, though specialized global event/bin assignment could reduce Python iteration.

ii.
```python
for out_unit, source_unit in enumerate(good_indices):
    spikes = spike_values[start:stop]
    edge_positions = np.searchsorted(spikes, absolute_edges.ravel(), side="left")
```
```python
outcome_codes = np.asarray([outcome_lookup[x] for x in outcomes], dtype=np.int8)
brain_idx = np.asarray([region_lookup[x] for x in annotations], dtype=np.int32)
```

iii. The AI says vectorizing over all trials avoids about 37 million Python unit/trial iterations. It retained the ragged per-unit loop as the practical compromise.

## 10-c. What processing does the code repeat multiple times?

i. Every NWB is opened once during `discover_inventory` and again during conversion; classification and annotations are read in both passes. Some trial/video masks and statistics are computed for reporting after related masks were already built. Validation also revisits representative shapes.

ii.
```python
def discover_inventory(paths):
    for path in paths:
        with h5py.File(path, "r") as nwb:
            classification = decode_array(nwb["units/classification"][:])
            annotations = decode_array(nwb["units/anno_name"][:])[good]
```
```python
with h5py.File(path, "r") as nwb:
    classification = decode_array(units["classification"][:])
    annotations = decode_array(units["anno_name"][:])[good_indices]
```

iii. The notes characterize conversion itself as single-pass per session, but the separate global inventory is required to establish stable subject and region vocabularies before session indices are emitted.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. `video_has_data`, auto/free-water counts, extensive `session_stats`, optional plotting intermediates, and inventory consistency checks are not decoder features; most are retained only in metadata or diagnostics. `n_trials_window_coverage` simply duplicates the current trial count. The all-file inventory also includes vocabularies beyond sample-mode converted sessions.

ii.
```python
video_has_data = np.any(frame_is_current_all, axis=1)
n_trials_window_coverage = n_trials
auto_water = trial_table["auto_water"][:n_trials_recorded][keep_idx]
free_water = trial_table["free_water"][:n_trials_recorded][keep_idx]
```

iii. The AI justifies these computations as sanity checks, provenance, processing plots, and reconciliation with source statistics. They do not affect decoder training, and plotting occurs only when explicitly requested.
