# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent globs every `/app/data/sub-*/*.nwb`, sorts the paths, pre-scans each with `h5py` to exclude sessions with no classifier-good units, and then streams each usable NWB session through `convert_session`.

ii.
```python
files = sorted(glob.glob(str(DATA_DIR / "sub-*" / "*.nwb")))
for path in files:
    with h5py.File(path, "r") as nwb:
        classification = nwb["units/classification"][()]
...
for number, path in enumerate(files, 1):
    result = convert_session(path)
```

iii. The notes justify direct HDF5 access as efficient and lossless, sorting as deterministic, and excluding the one unlabeled session because it has no curated units. They report 174 source NWBs and 173 usable sessions.

## 1-b. How are the data split into subjects?

i. Subject IDs are parsed from each NWB filename, unique IDs are sorted, and each session receives an integer `subject_idx`.

ii.
```python
def subject_from_path(path: str) -> str:
    return Path(path).name.split("_")[0].removeprefix("sub-")
subjects = sorted({subject_from_path(path) for path in files})
subject_idx.append(subject_lookup[result.subject])
```

iii. The agent treats the DANDI `sub-<id>` filename component as the canonical mouse identifier and reports 28 mice.

## 1-c. How are the data split into sessions?

i. One NWB file is treated as one session. Sessions remain in sorted-path order and the filename prefix before `_behavior` becomes the session ID.

ii.
```python
def session_id_from_path(path: str) -> str:
    return Path(path).name.split("_behavior")[0]
for number, path in enumerate(files, 1):
    result = convert_session(path)
```

iii. The notes state that the source layout is one NWB per experimental session, so no inferred grouping is necessary.

## 1-d. How are the data split into trials?

i. Trial rows come from `intervals/trials`. The ephys-backed contiguous trial block is mapped to those rows by matching each retained unit's observation-interval start times to trial-table `start_time`; selected row indices are then used for events and labels.

ii.
```python
trials = nwb["intervals/trials"]
trial_starts_table = np.asarray(trials["start_time"], dtype=np.float64)
ephys_trial_idx = np.searchsorted(trial_starts_table, obs_intervals[:, 0])
candidate_trial_idx = ephys_trial_idx[np.flatnonzero(valid_prefix)]
go_times = go_all[selected_trial_idx]
```

iii. The agent found that one session's neural block begins at raw trial 125, so assuming a prefix would misattach behavior. Exact `obs_intervals` mapping was adopted and independently spot-checked.

## 1-e. How are trials filtered based on quality controls?

i. The agent keeps only ephys-backed trials valid for every retained good unit, removes both auto-water and free-water trials, then removes any remaining trial whose entire curated population has zero spikes in the four-second window. Sessions must retain at least two trials. Early-lick, ignore, and stimulation trials are retained.

ii.
```python
valid_prefix = np.all(good_trial_matrix, axis=0)
candidate_trial_idx = ephys_trial_idx[np.flatnonzero(valid_prefix)]
water_trial = auto_water | free_water
selected_trial_idx = candidate_trial_idx[~water_trial]
zero_spike_trials = ~np.any(neural_rates > 0, axis=(1, 2))
```

iii. The notes call the all-unit mask conservative QC, regard water trials as behaviorally irregular, and retain early/ignore/stimulation trials because they define requested variables. Two all-zero edge trials were interpreted as missing coverage rather than silence.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural values come from ragged `units/spike_times` and `spike_times_index`, restricted by `units/classification == b"good"`; go-cue timestamps define trial windows.

ii.
```python
unit_indices = np.flatnonzero(classification == b"good").astype(np.int64)
neural_rates = bin_selected_units(
    nwb["units/spike_times"], np.asarray(nwb["units/spike_times_index"]), unit_indices, go_times
)
```

iii. The classifier label is described as the paper's spike-sorting QC verdict and is preferred over older permissive unit labels.

## 2-b. How is the `neural` data processed?

i. For each curated unit, spikes are assigned to non-overlapping trial windows and 50-ms bins with `searchsorted` and `bincount`; counts are divided by 0.05 to produce float32 firing rates in Hz. There is no smoothing or normalization.

ii.
```python
bin_idx = np.floor((rel - OFF_START) / BIN_SIZE_S).astype(np.int64)
flat_idx = trial_idx[good_bin] * N_TIME + bin_idx[good_bin]
counts = np.bincount(flat_idx, minlength=n_trials * N_TIME)
rates[:, out_unit, :] = counts.reshape(n_trials, N_TIME) / BIN_SIZE_S
```

iii. The agent says this matches the reference histogram/rate operation while using the task-mandated bin width.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units explicitly classified `good` are retained; sessions without one are excluded. Curated units must also have a nonempty CCF annotation. Trial-level `is_good_trials` is intersected across all retained units.

ii.
```python
unit_indices = np.flatnonzero(classification == b"good").astype(np.int64)
if any((not x) or x.lower() == "nan" for x in region_names):
    raise ValueError(...)
valid_prefix = np.all(good_trial_matrix, axis=0)
```

iii. The notes attribute `classification` to the published QC classifier and treat `is_good_trials` as a manual validity annotation for unstable periods.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each global spike time is assigned relative to its trial's `go_start_times` timestamp; only offsets in `[-2.5, 1.5)` are retained.

ii.
```python
window_starts = go_times + OFF_START
trial_idx = np.searchsorted(window_starts, spikes, side="right") - 1
rel = spikes - go_times[trial_idx]
inside = (rel >= OFF_START) & (rel < OFF_END)
```

iii. The agent notes that spikes and events share the NWB session clock, so subtraction by the go timestamp gives direct alignment without interpolation.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The output has 80 adjacent, non-overlapping 50-ms bins over `[-2.5, 1.5)` seconds. Raw spike timestamps are newly histogrammed into these bins; no later rebinning occurs.

ii.
```python
OFF_START, OFF_END, BIN_SIZE_S = -2.5, 1.5, 0.050
BIN_EDGES = np.linspace(OFF_START, OFF_END, 81, dtype=np.float64)
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2
```

iii. This grid directly follows the decoder instructions and replaces the paper analysis's other window/stride choices.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It uses `sample_start_times`, trial-table `start_time`, go-cue times, and the common bin centers. The agent chooses the first sample event at or after trial start.

ii.
```python
sample_starts = np.asarray(nwb["acquisition/BehavioralEvents/sample_start_times/timestamps"])
first_sample_idx = np.searchsorted(sample_starts, trial_starts, side="left")
tone_onsets = sample_starts[first_sample_idx]
```

iii. The agent argues that “tone onset” means the first tone in the behavioral trial and intentionally preserves long elapsed times on early-lick replay trials.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The selected tone timestamp is subtracted from every global 50-ms bin center, yielding a float32 continuous ramp.

ii.
```python
centers_global = go_times[:, None] + BIN_CENTERS[None, :]
elapsed_from_tone = (centers_global - tone_onsets[:, None]).astype(np.float32)
```

iii. The notes say no transformation beyond elapsed seconds is needed and that consecutive values should increase by exactly 0.05 s.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is evaluated at the same go-cue-relative bin centers as the spike-rate bins.

ii.
```python
centers_global = go_times[:, None] + BIN_CENTERS[None, :]
inputs = np.stack((elapsed_from_tone, stim_on), axis=1)
```

iii. Shared global timestamps and one common center grid are the stated alignment justification.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. It is derived from global `photostim_start_times` and `photostim_stop_times` event timestamps.

ii.
```python
stim_starts = np.asarray(nwb["acquisition/BehavioralEvents/photostim_start_times/timestamps"])
stim_stops = np.asarray(nwb["acquisition/BehavioralEvents/photostim_stop_times/timestamps"])
```

iii. The agent prefers native start/stop event streams because they already share the global clock and directly express light-on intervals.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. For every bin center, the most recent start interval is located and the value is 1 when the center is in the half-open `[start, stop)` interval, otherwise 0.

ii.
```python
interval_idx = np.searchsorted(starts, flat, side="right") - 1
on = (interval_idx >= 0) & (flat >= starts[safe_idx]) & (flat < stops[safe_idx])
return on.reshape(centers_global.shape).astype(np.float32)
```

iii. This implements the requested time-varying binary state and retains stimulated trials.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Global stimulation intervals are sampled at the same go-relative global centers used for neural bins.

ii.
```python
centers_global = go_times[:, None] + BIN_CENTERS[None, :]
stim_on = photostim_state(centers_global, stim_starts, stim_stops)
```

iii. The common NWB clock and common center array are cited as guaranteeing temporal correspondence.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is inferred from trial-table `trial_instruction` and `outcome`: hit means instructed side, miss means opposite side, and ignore means no lick.

ii.
```python
outcomes = np.asarray(trials["outcome"])[selected_trial_idx]
instructions = np.asarray(trials["trial_instruction"])[selected_trial_idx]
outputs[:, 0, :] = map_choice(instructions, outcomes)[:, None]
```

iii. The notes explain that there is no direct choice column but instruction crossed with correctness uniquely determines the response class.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Values are encoded left=0, right=1, no lick=2 and repeated over all 80 time bins.

ii.
```python
if outcome == "ignore": result[i] = 2
elif outcome == "hit": result[i] = 0 if instruction == "left" else 1
elif outcome == "miss": result[i] = 1 if instruction == "left" else 0
```

iii. Repetition permits fixed and time-varying outputs to share one dense `(4, 80)` array.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. It comes directly from the trial-table `outcome` field for retained trial rows.

ii.
```python
outcomes = np.asarray(trials["outcome"])[selected_trial_idx]
```

iii. The source already contains exactly the requested three categories.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Strings are mapped ignore=0, miss=1, hit=2 and repeated across time.

ii.
```python
code = {"ignore": 0, "miss": 1, "hit": 2}
outputs[:, 1, :] = map_outcome(outcomes)[:, None]
```

iii. The fixed codebook mirrors the requested category order; repetition gives a uniform output tensor.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. It comes directly from the retained rows of trial-table `early_lick`.

ii.
```python
early = np.asarray(trials["early_lick"])[selected_trial_idx]
```

iii. The source explicitly supplies the trial flag, so event reconstruction is unnecessary.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. `no early` maps to 0 and `early` to 1, repeated across all time bins.

ii.
```python
code = {"no early": 0, "early": 1}
outputs[:, 2, :] = map_early(early)[:, None]
```

iii. The agent retains these trials because early lick is a requested decoder target.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. It uses timestamps and the x, y, and DeepLabCut likelihood columns of `Camera0_side_TongueTracking`.

ii.
```python
tracking = nwb["acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking"]
tracking_ts = np.asarray(tracking["timestamps"], dtype=np.float64)
tracking_data = np.asarray(tracking["data"], dtype=np.float64)
```

iii. This is the native side-camera tongue track; likelihood is used to distinguish visible tongue from finite but unreliable coordinates.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Frames require likelihood at least 0.9. Within visible consecutive frames, 2-D speed outliers above mean plus five standard deviations are identified and their x/y coordinates interpolated. Session percentiles are then computed over visible cleaned frame-level y values. At each bin center, the last preceding frame is sampled if no older than 5.1 ms; otherwise class 3 is emitted.

ii.
```python
visible = finite & (likelihood >= DLC_VISIBLE_THRESHOLD)
velocity_threshold = float(velocity_values.mean() + 5.0 * velocity_values.std())
xy[outlier, dim] = np.interp(...)
q40, q60 = np.percentile(clean_y[visible], [40.0, 60.0])
idx = np.searchsorted(timestamps, flat, side="right") - 1
near = (idx >= 0) & (age >= -1e-9) & (age <= 0.0051)
```

iii. The agent says high-confidence gating avoids treating retracted/occluded coordinates as real, five-SD interpolation follows the movement paper, and center sampling follows its marker alignment while preserving the required not-visible class.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Per-session 40th and 60th percentiles of cleaned, visible raw-frame y values define: 0 below q40, 1 from q40 through q60, 2 above q60, and 3 for invisible or absent-near-center frames.

ii.
```python
categories = np.full(flat.shape, 3, dtype=np.int8)
categories[is_visible & (y < q40)] = 0
categories[is_visible & (y >= q40) & (y <= q60)] = 1
categories[is_visible & (y > q60)] = 2
```

iii. The 40/60 split and session scope come from the instructions; the explicit fourth class represents low-confidence/missing frames.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Each global neural-bin center selects the immediately preceding camera frame, provided it is within 5.1 ms.

ii.
```python
flat = centers_global.ravel()
idx = np.searchsorted(timestamps, flat, side="right") - 1
age = flat - timestamps[safe_idx]
near = (idx >= 0) & (age >= -1e-9) & (age <= 0.0051)
```

iii. The agent treats one nominal video-frame interval as acceptable and relies on the shared global clock for alignment.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. A session with absent classifier labels is excluded; observation intervals correct behavioral/ephys trial offsets; trials invalid for any retained unit, water trials, and two all-zero coverage edges are dropped. Missing or low-confidence tongue frames become class 3. Structural mismatches, missing annotations, nonfinite arrays, and too-few trials raise errors rather than being imputed.

ii.
```python
if n_good: usable.append(path)
else: excluded.append(...)
if selected_trial_idx.size < 2: raise ValueError(...)
categories = np.full(flat.shape, 3, dtype=np.int8)
if not np.all(np.isfinite(neural_rates)) ...: raise AssertionError(...)
```

iii. The notes distinguish absent neural coverage, which should be excluded, from legitimate tongue nonvisibility, which has an explicit output category. They document iterative fixes for a non-prefix session and two coverage-edge trials.

## 10-a. What are the most time-consuming steps of the code?

i. Session conversion—especially selected-unit spike loading/binning and tongue-array loading/processing—is the dominant compute work; accumulating and serializing the roughly 10.8-GiB pickle is also substantial. The full run reports 130.42 s summed session work and 14.56 s serialization.

ii.
```python
for out_unit, raw_unit in enumerate(unit_indices):
    spikes = np.asarray(spike_times_ds[lo:hi], dtype=np.float64)
...
tracking_data = np.asarray(tracking["data"], dtype=np.float64)
pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The notes attribute speed to streaming HDF5 sessions, vectorized spike assignment within each unit, compact dtypes, and avoiding waveform/raw uncurated-spike loading.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The explicit per-unit neural loop and Python loop in `map_choice` are candidates. Trial/bin spike assignment inside each unit is already vectorized; mapping choice could trivially use array operations. File/session loops are natural because files and ragged spike trains differ in size.

ii.
```python
for out_unit, raw_unit in enumerate(unit_indices):
    ...
for i, (instruction_raw, outcome_raw) in enumerate(zip(instructions, outcomes)):
    ...
```

iii. The agent emphasizes that it removed per-trial histogram loops with `searchsorted`/`bincount`; it does not claim the remaining small categorical loop is performance-critical.

## 10-c. What processing does the code repeat multiple times?

i. Every NWB is opened in `discover_usable_files`, reopened in a separate region-set pass, and reopened again in `convert_session`; classification and region labels are consequently read multiple times. Full validation then loops through every trial array after conversion.

ii.
```python
for path in files:
    with h5py.File(path, "r") as nwb:  # discovery
...
for path in files:
    with h5py.File(path, "r") as nwb:  # region collection
...
with h5py.File(path, "r") as nwb:      # conversion
```

iii. The notes prioritize deterministic global subject/region vocabularies and thorough validation, but do not explicitly acknowledge this repeated I/O.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. `convert_session` always constructs a large `plot_payload` containing references to neural, input, output, raw/cleaned tracking, and auxiliary indices, even when `--show-processing` is false; only optional diagnostic plots consume it. It also computes/stores extensive provenance and cleaning summaries not used by decoder training, although these are useful for auditability.

ii.
```python
plot_payload = {
    "inputs": inputs, "outputs": outputs, "neural": neural_rates,
    "tracking_y_raw": tracking_data[:, 1], "tracking_y_clean": clean_y, ...
}
return SessionResult(..., plot_payload=plot_payload)
```

iii. The agent intentionally created plots and rich metadata for sanity checks. The payload mostly holds existing arrays as references, but constructing ancillary arrays such as frame indices is unnecessary for a normal full conversion and is discarded with each `SessionResult`.
