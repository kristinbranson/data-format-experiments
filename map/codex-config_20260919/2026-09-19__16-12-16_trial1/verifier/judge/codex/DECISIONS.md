# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads all session files with a single glob over `/app/data/sub-*/*.nwb`, sorts the paths deterministically, and opens each file once with `h5py`. Within each file it reads NWB/HDF5 groups directly rather than using `pynwb`.

ii.
```python
paths = sorted(glob.glob(str(DATA_ROOT / "sub-*" / "*.nwb")))
...
for path in paths:
    result = convert_session(path, make_plot=make_plot)
```

```python
with h5py.File(path, "r") as nwb:
    good_units, annotations = get_good_units(nwb)
    selection = map_observed_trials(nwb, good_units)
```

iii. In `CONVERSION_NOTES.md`, the agent says it chose direct HDF5/NWB access and a one-session-at-a-time pass for speed and memory control, while still using the same published NWB files and session layout as the source dataset.

## 1-b. How are the data split into subjects?

i. Subjects are taken from the parent directory name such as `sub-440956`, then deduplicated and sorted at assembly time. The code cross-checks that the NWB subject field agrees with the folder naming convention.

ii.
```python
subject_id = Path(path).parent.name
source_subject = decode_scalar(nwb["general/subject/subject_id"][()])
if source_subject not in {subject_id, subject_id.removeprefix("sub-")}:
    raise ValueError(
        f"Path subject {subject_id} disagrees with NWB subject {source_subject}"
    )
```

```python
subjects = sorted({x["subject"] for x in converted_sessions})
subject_to_idx = {name: idx for idx, name in enumerate(subjects)}
...
"subject_idx": np.asarray(
    [subject_to_idx[x["subject"]] for x in converted_sessions],
    dtype=np.int64,
),
```

iii. The notes say the file-path subject and NWB subject were cross-checked for consistency, and sorted unique subject IDs were used to keep output ordering reproducible.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. Session order follows the sorted file list, and per-session results are appended directly to the top-level lists.

ii.
```python
paths = sorted(glob.glob(str(DATA_ROOT / "sub-*" / "*.nwb")))
...
for path in paths:
    result = convert_session(path, make_plot=make_plot)
    if result is None:
        continue
    converted_sessions.append(result)
```

```python
session_id = Path(path).stem.replace("_behavior+ecephys+ogen", "").replace(
    "_behavior+ecephys", ""
)
```

iii. The notes state that the dandiset layout already provides one session per file, so no further grouping is needed; deterministic sorting preserves stable session order.

## 1-d. How are the data split into trials?

i. Trials are defined from the NWB trials table (`intervals/trials`) and linked to one go cue per row. For observed subsets, the code maps `obs_intervals` start/stop pairs back onto exact trial-table rows.

ii.
```python
trial_group = nwb["intervals/trials"]
starts = trial_group["start_time"][:]
stops = trial_group["stop_time"][:]
...
go_times = nwb["acquisition/BehavioralEvents/go_start_times/timestamps"][:]
if len(go_times) != n_original:
    raise ValueError("Go-cue count does not match trial-table length")
```

```python
for j, (obs_start, obs_stop) in enumerate(common_obs):
    matches = np.flatnonzero(
        np.isclose(starts, obs_start, atol=1e-8, rtol=0)
        & np.isclose(stops, obs_stop, atol=1e-8, rtol=0)
    )
    ...
    observed_trial_indices[j] = matches[0]
```

iii. The notes justify this by saying go cues are one-per-trial, while some other event streams can repeat within a trial, so the trials table is the authoritative source and `obs_intervals` is only used to map observed subsets back onto those rows.

## 1-e. How are trials filtered based on quality controls?

i. The agent applies several neural-validity filters. It keeps only trials listed in the common good-unit `obs_intervals`, then only columns where every retained good unit is marked true in `units/is_good_trials`, then only trials whose full `[-2.5, +1.5]` window lies inside the common recording span, and finally removes trials whose binned neural data are population-all-zero. It does not explicitly filter `free_water` trials.

ii.
```python
all_units_valid = np.all(unit_trial_validity[good_units, :], axis=0)
after_unit_qc = observed_trial_indices[all_units_valid]
...
full_window = (
    (go_times[after_unit_qc] + TIME_EDGES[0] >= recording_start - 1e-9)
    & (go_times[after_unit_qc] + TIME_EDGES[-1] <= recording_stop + 1e-9)
)
selected = after_unit_qc[full_window]
```

```python
neural_data_present = np.any(rates > 0, axis=(1, 2))
if n_all_zero_neural_trials:
    trial_indices = trial_indices[neural_data_present]
    inputs = inputs[neural_data_present]
    rates = rates[neural_data_present]
```

iii. The notes explicitly justify these added filters as honoring NWB valid-period fields and removing truncated recordings that would otherwise look like genuine 0 Hz activity. They also say early/ignore/stimulated trials were deliberately retained because those are required outputs or inputs for the decoder.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from `units/spike_times` plus `units/spike_times_index`, with go-cue timestamps used to define the bin edges in absolute session time.

ii.
```python
spike_index = nwb["units/spike_times_index"][:].astype(np.int64)
previous = np.r_[0, spike_index[:-1]]
all_spikes = nwb["units/spike_times"][:]
...
go_times = nwb["acquisition/BehavioralEvents/go_start_times/timestamps"][:]
absolute_edges = go_times[trial_indices, None] + TIME_EDGES[None, :]
```

iii. The notes say NWB spike times are the native neural representation and already share the same absolute session clock as behavioral event timestamps.

## 2-b. How is the `neural` data processed?

i. The code bins raw spike timestamps into 80 adjacent 50 ms bins per trial and divides the counts by 0.05 s to produce firing rates in Hz. No smoothing, normalization, or baseline subtraction is applied.

ii.
```python
rates = np.empty(
    (len(trial_indices), len(good_units), len(TIME_CENTERS)), dtype=np.float32
)
for out_unit, source_unit in enumerate(good_units):
    spikes = all_spikes[previous[source_unit] : spike_index[source_unit]]
    edge_positions = np.searchsorted(spikes, absolute_edges, side="left")
    rates[:, out_unit, :] = np.diff(edge_positions, axis=1) / BIN_WIDTH_S
```

iii. The notes say this uses the requested decoder binning grid and preserves the reference half-open histogram logic while reading each session’s spike buffer only once.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units whose NWB `classification` equals `"good"` are retained. If a session has no such units, it is dropped.

ii.
```python
classifications = decode_array(nwb["units/classification"])
good_indices = np.flatnonzero(classifications == "good")
...
if len(good_units) == 0:
    print(f"SKIP {session_id}: no classifier-good units", flush=True)
    return None
```

iii. The notes say the agent followed the classifier-based QC described in the spike-sorting white paper and deliberately avoided replacing it with ad hoc metric thresholds.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural activity is aligned to the go cue by adding the fixed relative bin edges to each trial’s go-cue timestamp to create absolute bin edges for that trial.

ii.
```python
TIME_EDGES = np.linspace(-2.5, 1.5, 81, dtype=np.float64)
...
go_times = nwb["acquisition/BehavioralEvents/go_start_times/timestamps"][:]
absolute_edges = go_times[trial_indices, None] + TIME_EDGES[None, :]
```

iii. The notes say the NWB dataset uses a continuous absolute clock, so no separate synchronization step is needed beyond defining go-centered windows.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data uses 80 non-overlapping 50 ms bins covering `[-2.5, +1.5]` s relative to the go cue. There is no additional temporal rebinning beyond this one binning step.

ii.
```python
TIME_EDGES = np.linspace(-2.5, 1.5, 81, dtype=np.float64)
TIME_CENTERS = (TIME_EDGES[:-1] + TIME_EDGES[1:]) / 2
BIN_WIDTH_S = 0.05
```

iii. The notes say the task instructions override the paper’s other binning schemes, so the converter uses this fixed 50 ms grid everywhere.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It is derived from `sample_start_times/timestamps` and `go_start_times/timestamps`. For each retained trial, the code uses the last sample/tone onset before the trial’s go cue.

ii.
```python
sample_starts = nwb[
    "acquisition/BehavioralEvents/sample_start_times/timestamps"
][:]
go_times = nwb["acquisition/BehavioralEvents/go_start_times/timestamps"][:]
...
positions = np.searchsorted(sample_starts, selected_go, side="left") - 1
tones = sample_starts[positions]
```

iii. The notes justify this by saying early licks can replay sample epochs, so the final sample onset preceding the aligned go cue is the relevant instruction tone.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The code first computes the absolute time of every decoder bin center for each trial, then subtracts that trial’s selected tone onset, yielding a continuous signed time-from-tone signal.

ii.
```python
absolute_centers = go_times[trial_indices, None] + TIME_CENTERS[None, :]
time_from_tone = absolute_centers - tone_onsets[:, None]
```

iii. The notes describe this as a unit-slope ramp in seconds that crosses zero at the selected tone onset.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is sampled on exactly the same 80 go-centered bin centers used for the neural data.

ii.
```python
absolute_centers = go_times[trial_indices, None] + TIME_CENTERS[None, :]
time_from_tone = absolute_centers - tone_onsets[:, None]
```

iii. The notes say the tone-time input and neural rates share the same trial-by-trial go-aligned time grid.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. Photostimulation is derived from the absolute event streams `photostim_start_times/timestamps` and `photostim_stop_times/timestamps`, rather than from the string-valued trial-table onset and duration columns.

ii.
```python
event_root = nwb["acquisition/BehavioralEvents"]
photo_starts = event_root["photostim_start_times/timestamps"][:]
photo_stops = event_root["photostim_stop_times/timestamps"][:]
```

iii. The notes say the agent preferred the real paired event intervals, then cross-checked that the number of non-`N/A` trial-table photostim rows matched the event count.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. The code marks a bin center as 1 if more photostim starts than stops have occurred by that absolute time, and 0 otherwise. This yields a binary time-varying laser-state input.

ii.
```python
started = np.searchsorted(photo_starts, absolute_centers, side="right")
stopped = np.searchsorted(photo_stops, absolute_centers, side="right")
photostim_on = (started > stopped).astype(np.float32)
```

iii. The notes justify this as evaluating the true paired laser intervals directly on the decoder time grid instead of inferring the state from trial-level metadata.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. It is evaluated at the same absolute bin centers used for neural activity, so the laser state is aligned to the identical go-centered grid.

ii.
```python
absolute_centers = go_times[trial_indices, None] + TIME_CENTERS[None, :]
started = np.searchsorted(photo_starts, absolute_centers, side="right")
stopped = np.searchsorted(photo_stops, absolute_centers, side="right")
```

iii. The notes say all streams already share the NWB absolute clock, so alignment reduces to sampling the photostim intervals at those neural bin centers.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is not read from a single stored variable. It is derived from the trial-table fields `trial_instruction` and `outcome`.

ii.
```python
instruction = decode_array(trials["trial_instruction"])[trial_indices]
outcome_text = decode_array(trials["outcome"])[trial_indices]
```

iii. The notes say this follows the task semantics used in the papers: hit means the animal licked the instructed side, miss means the opposite side, and ignore means no lick choice.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The code encodes left as 0, right as 1, and no lick as 2. It derives those labels from `instruction × outcome` and then broadcasts the per-trial label across all 80 bins.

ii.
```python
actual_choice = np.full(len(trial_indices), 2, dtype=np.int64)
hit = outcome_text == "hit"
miss = outcome_text == "miss"
actual_choice[hit & (instruction == "left")] = 0
actual_choice[hit & (instruction == "right")] = 1
actual_choice[miss & (instruction == "left")] = 1
actual_choice[miss & (instruction == "right")] = 0
...
outputs[:, 0, :] = actual_choice[:, None]
```

iii. The notes say outcome, not raw lick timestamps, is treated as authoritative for choice because rare lick-event conflicts can occur.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is taken directly from the trial-table `outcome` column.

ii.
```python
outcome_text = decode_array(trials["outcome"])[trial_indices]
```

iii. The notes say the NWB trials table already contains the target categories `ignore`, `miss`, and `hit`.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The code maps `ignore`, `miss`, and `hit` to integer codes 0, 1, and 2, then repeats the per-trial code across all 80 bins.

ii.
```python
outcome_map = {name: idx for idx, name in enumerate(OUTCOME_VALUES)}
outcomes = np.asarray([outcome_map[x] for x in outcome_text], dtype=np.int64)
...
outputs[:, 1, :] = outcomes[:, None]
```

iii. The notes describe this as direct categorical encoding with no extra derivation.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Early lick is taken directly from the trial-table `early_lick` column.

ii.
```python
early_text = decode_array(trials["early_lick"])[trial_indices]
```

iii. The notes say early-lick trials were intentionally retained because early licking is itself a required decoder output.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The code converts `"early"` to 1 and `"no early"` to 0, then broadcasts that label across all 80 bins.

ii.
```python
early = (early_text == "early").astype(np.int64)
...
outputs[:, 2, :] = early[:, None]
```

iii. The notes say the per-trial early-lick label is kept constant across time so it can live in the same `(4, 80)` output array as the time-varying tongue signal.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. Tongue y-position is derived from the side-camera tongue-tracking series `Camera0_side_TongueTracking`, specifically the y coordinate, timestamps, and likelihood columns.

ii.
```python
tracking = nwb[
    "acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking"
]
timestamps = tracking["timestamps"][:]
marker = tracking["data"][:]
...
y = marker[:, 1].astype(np.float64, copy=True)
likelihood = marker[:, 2].astype(np.float64, copy=False)
```

iii. The notes say this is the side-camera marker stream used by the reference tracking pipeline, with y as the quantity to discretize and likelihood used for visibility.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The agent uses a different tongue-processing pipeline from the reference converter. It keeps frames with likelihood at least 0.9, repairs high-speed outlier frames by interpolation using a five-standard-deviation velocity rule, computes session percentile thresholds on visible y values, samples the nearest camera frame to each decoder bin center, and assigns class 3 to uncovered or low-confidence bins.

ii.
```python
visible = (
    np.isfinite(x)
    & np.isfinite(y)
    & np.isfinite(likelihood)
    & (likelihood >= TONGUE_LIKELIHOOD_THRESHOLD)
)
...
cutoff = np.mean(reference_speeds) + 5 * np.std(reference_speeds)
...
y[target] = np.interp(
    timestamps[target], timestamps[interpolation_basis], y[interpolation_basis]
)
```

```python
right = np.searchsorted(timestamps, flat_centers, side="left")
...
sampled_visible = covered & visible[nearest]
sampled_y = y[nearest]
classes = np.full(len(flat_centers), 3, dtype=np.int64)
```

iii. The notes justify this by appealing to the method paper’s tracking preprocessing: likelihood is sharply bimodal, outlier repair is scientifically motivated, and nearest-frame sampling preserves an explicit “not visible” class instead of imputing hidden tongue positions.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. The code uses per-session 40th and 60th percentiles of the visible corrected y values. Class 0 is below `p40`, class 1 is `p40` through `p60` inclusive, class 2 is above `p60`, and class 3 is not visible.

ii.
```python
p40, p60 = np.percentile(y[visible], [40, 60])
...
classes[sampled_visible & (sampled_y < p40)] = 0
classes[
    sampled_visible & (sampled_y >= p40) & (sampled_y <= p60)
] = 1
classes[sampled_visible & (sampled_y > p60)] = 2
```

iii. The notes say this was intended to implement the instruction’s per-session percentile discretization while keeping low-confidence or missing samples as the explicit class-3 category.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Instead of averaging all camera frames within each 50 ms neural bin, the code samples the nearest camera frame to each neural bin center, subject to a coverage check based on 1.5 frame intervals.

ii.
```python
flat_centers = absolute_centers.ravel()
right = np.searchsorted(timestamps, flat_centers, side="left")
...
nearest = np.where(choose_left, left_clipped, right_clipped)
median_dt = float(np.median(np.diff(timestamps)))
covered = (
    (flat_centers >= timestamps[0])
    & (flat_centers <= timestamps[-1])
    & (np.abs(timestamps[nearest] - flat_centers) <= 1.5 * median_dt)
)
```

iii. The notes justify this as aligning video and neural data on the shared absolute clock while respecting the task’s required “not visible” category for missing coverage.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The agent mainly handles bad or missing data by dropping invalid sessions or trials and by using explicit class-3 tongue labels when the tongue is not visible. It also raises hard errors for structural mismatches such as inconsistent event counts or malformed tracking arrays.

ii.
```python
if np.any(annotations == ""):
    raise ValueError("A classifier-good unit has an empty CCF annotation")
...
if len(selected) < 2:
    raise ValueError(f"Only {len(selected)} fully valid trials remain")
```

```python
classes = np.full(len(flat_centers), 3, dtype=np.int64)
...
if len(photo_starts) != len(photo_stops):
    raise ValueError("Photostimulation start/stop event counts differ")
```

iii. The notes specifically mention three recurring edge cases: ragged `is_good_trials` arrays that had to be mapped through `obs_intervals`, truncated all-zero neural windows that were removed, and video bins without valid coverage that become tongue class 3.

## 10-a. What are the most time-consuming steps of the code?

i. The agent identifies spike I/O and spike binning as the dominant costs. It explicitly says that random per-unit HDF5 reads and Python trial/bin loops would have been too slow, and that the dense neural payload is the main scaling bottleneck. Full conversion plus writing also spends nontrivial time on pickle output.

ii.
```python
# One sequential read is substantially faster than thousands of HDF5 reads.
all_spikes = nwb["units/spike_times"][:]
...
for out_unit, source_unit in enumerate(good_units):
    spikes = all_spikes[previous[source_unit] : spike_index[source_unit]]
    edge_positions = np.searchsorted(spikes, absolute_edges, side="left")
```

iii. The notes say the expensive steps are the large sequential spike read, the per-unit `searchsorted` binning, and the large final float32 neural payload and pickle write.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The largest remaining nontrivial Python loop is the per-unit loop in `bin_spikes`. There is also a loop over observed intervals in `map_observed_trials`, plus some small loops in output assembly and summary statistics. The tongue path itself is mostly vectorized.

ii.
```python
for out_unit, source_unit in enumerate(good_units):
    spikes = all_spikes[previous[source_unit] : spike_index[source_unit]]
    edge_positions = np.searchsorted(spikes, absolute_edges, side="left")
    rates[:, out_unit, :] = np.diff(edge_positions, axis=1) / BIN_WIDTH_S
```

```python
for j, (obs_start, obs_stop) in enumerate(common_obs):
    matches = np.flatnonzero(
        np.isclose(starts, obs_start, atol=1e-8, rtol=0)
        & np.isclose(stops, obs_stop, atol=1e-8, rtol=0)
    )
```

iii. The notes say the agent intentionally vectorized event sampling, tracking sampling, and categorical construction already, leaving the ragged-by-unit spike loop as the main unavoidable loop.

## 10-c. What processing does the code repeat multiple times?

i. The code repeats some lightweight reads and traversals, mostly because functionality is split across helpers. For example, go-cue timestamps are loaded in several functions, trial strings are decoded again when outputs are built, and summary code makes another pass over converted inputs and outputs. It does not repeat the expensive spike binning itself.

ii.
```python
go_times = nwb["acquisition/BehavioralEvents/go_start_times/timestamps"][:]
```

This appears in `map_observed_trials`, `final_tone_onsets`, `construct_inputs`, and `bin_spikes`.

```python
for session_inputs, session_outputs in zip(data["input"], data["output"]):
    for x, y in zip(session_inputs, session_outputs):
        ...
```

iii. The notes emphasize that the expensive computations are done once per session, but the modular structure still causes repeated small dataset reads and post-conversion summary passes.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script performs a fair amount of diagnostic and provenance work that is not needed for downstream decoding arrays: optional processing plots, timing/summary prints, extensive per-session metadata, photostim-event count audits, and multiple shape/range checks. These are useful for validation but are not consumed by the decoder inputs or outputs.

ii.
```python
if make_plot:
    plot_processing(
        session_id,
        nwb,
        trial_indices,
        good_units,
        rates,
        inputs,
        outputs,
        absolute_centers,
        tongue,
    )
```

```python
info = {
    "session_id": session_id,
    "source_file": os.path.relpath(path, "/app"),
    ...
    "photostimulation_event_count": int(n_photo_events),
    "retained_choice_counts": np.bincount(labels["choice"], minlength=3).tolist(),
}
```

iii. The notes explicitly frame these as validation and documentation features added to satisfy the conversion workflow, not as data used by downstream analyses themselves.
