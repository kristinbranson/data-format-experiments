# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads the dataset by globbing all NWB files under `/app/data/sub-*/*.nwb`. It then does a first pass over every file with `h5py` to build the global subject and brain-region vocabularies, and a second pass over each file to convert that session. Within each file it reads HDF5 groups directly rather than using `pynwb`.

ii.
```python
paths = sorted(glob.glob(str(DATA_ROOT / "sub-*" / "*.nwb")))
...
subjects, brain_regions = discover_inventory(paths)
...
for path in paths:
    result = convert_session(
        path,
        subject_lookup,
        region_lookup,
        show_processing=args.show_processing and len(converted) < 2,
    )
```

```python
with h5py.File(path, "r") as nwb:
    units = nwb["units"]
    trial_table = nwb["intervals/trials"]
    go_all = nwb["acquisition/BehavioralEvents/go_start_times/timestamps"][:n_trials_recorded]
```

iii. In `CONVERSION_NOTES.md`, the AI describes the dataset as 174 NWB files grouped by subject directories, one file per recording session, and treats direct HDF5 reads as sufficient because all needed arrays are already present in the NWB files.

## 1-b. How are the data split into subjects?

i. Subjects are split by the NWB `general/subject/subject_id` field. The AI checks that this matches the enclosing `sub-<id>` directory, builds a globally sorted subject list, and stores one integer `subject_idx` per converted session.

ii.
```python
subject = decode_scalar(nwb["general/subject/subject_id"][()])
folder_subject = Path(path).parent.name.removeprefix("sub-")
if subject != folder_subject:
    raise ValueError(f"Subject mismatch in {path}: {subject} vs {folder_subject}")
subjects.add(subject)
```

```python
"subjects": subjects,
"subject_idx": np.asarray([x["subject_idx"] for x in converted], dtype=np.int32),
```

iii. The notes say the folder and NWB IDs should be asserted consistent and that the dataset contains 28 mice, so the AI uses the NWB subject ID as the canonical mouse identifier.

## 1-c. How are the data split into sessions?

i. The AI treats one NWB file as one session. Sessions are processed in lexicographic path order, and the session ID written to metadata comes from the filename stem before `_behavior`.

ii.
```python
def session_id_from_path(path: str) -> str:
    name = Path(path).name
    return name.split("_behavior")[0]
...
for path in paths:
    result = convert_session(path, ...)
```

```python
"session_info": [x["session_info"] for x in converted],
```

iii. `CONVERSION_NOTES.md` states that there are 174 NWB files and that one file is one recording session, so no extra grouping step is inferred.

## 1-d. How are the data split into trials?

i. The AI uses the NWB trial table, but truncates it to the number of recorded trials implied by `units/is_good_trials.shape[1]`. Go-cue timestamps are sliced to the same leading set of trials, and later filters are applied on that recorded-trial subset.

ii.
```python
trial_table = nwb["intervals/trials"]
n_trials_table = len(trial_table["id"])
stability_all = units["is_good_trials"][:]
stability = stability_all[good_indices]
n_trials_recorded = stability.shape[1]
...
go_all = nwb["acquisition/BehavioralEvents/go_start_times/timestamps"][:n_trials_recorded]
trial_starts = trial_table["start_time"][:n_trials_recorded]
```

iii. The notes justify this by saying some sessions contain trailing behavioral trials after ephys coverage ends, and that `is_good_trials.shape[1]` plus `obs_intervals` identify the recorded neural-trial extent.

## 1-e. How are trials filtered based on quality controls?

i. The AI keeps only trials that are marked stable for **all** retained good units, have a finite tone assignment, and later show at least one nonzero spike somewhere in the 4 s population tensor. It does not use the human reference rule of `obs_intervals` plus `free_water == 0` as the actual filter, though it checks `obs_intervals` lengths for consistency.

ii.
```python
stability_all = units["is_good_trials"][:]
stability = stability_all[good_indices]
...
stable_mask = np.all(stability, axis=0)
...
coverage_mask = np.isfinite(tone_all)
keep = stable_mask & coverage_mask
keep_idx = np.flatnonzero(keep)
```

```python
neural_present = np.any(rates != 0, axis=(1, 2))
if not np.all(neural_present):
    ...
    rates = rates[neural_present]
    keep_idx = keep_idx[neural_present]
```

iii. The notes say the paper’s behavioral exclusions should not be copied because early lick, ignore, and photostimulation are required targets, and they explicitly choose to retain only trials for which all curated units are stable. Later notes also justify dropping “all-zero population” trials as off-by-one spike-coverage errors.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural activity is derived from `units/spike_times` and `units/spike_times_index` for units whose `classification == "good"`, with go-cue timestamps from `BehavioralEvents/go_start_times` used to place the analysis window.

ii.
```python
classification = decode_array(units["classification"][:])
good_indices = np.flatnonzero(classification == "good")
...
go_all = nwb["acquisition/BehavioralEvents/go_start_times/timestamps"][:n_trials_recorded]
...
spike_ends = units["spike_times_index"][:]
spike_values = units["spike_times"]
```

iii. The notes identify classifier-curated units as the authoritative QC population and describe all timestamps as sharing the same absolute session clock.

## 2-b. How is the `neural` data processed?

i. The AI bins spikes into non-overlapping 50 ms bins from -2.5 s to +1.5 s around the go cue, using `np.searchsorted` at every bin edge for each retained unit, and divides spike counts by bin width to convert them to Hz. No smoothing or baseline subtraction is applied.

ii.
```python
BIN_WIDTH_S = 0.050
BIN_EDGES = np.arange(OFF_START, OFF_END + BIN_WIDTH_S / 2, BIN_WIDTH_S)
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2
```

```python
absolute_edges = go[:, None] + BIN_EDGES[None, :]
rates = np.empty((n_trials, n_units_good, N_TIME), dtype=np.float32)
for out_unit, source_unit in enumerate(good_indices):
    ...
    edge_positions = np.searchsorted(spikes, absolute_edges.ravel(), side="left")
    counts = np.diff(edge_positions.reshape(n_trials, N_TIME + 1), axis=1)
    rates[:, out_unit, :] = counts / BIN_WIDTH_S
```

iii. `CONVERSION_NOTES.md` says the decoder specification overrides the paper’s 40 ms / 3.4 ms scheme, so the AI intentionally uses 80 non-overlapping 50 ms bins and validates the resulting 20 Hz quantization.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI retains only units whose NWB `classification` equals `"good"`. A session with zero such units is skipped. It also requires those retained units to have nonempty anatomy labels during the inventory pass.

ii.
```python
classification = decode_array(units["classification"][:])
good_indices = np.flatnonzero(classification == "good")
if len(good_indices) == 0:
    print(f"SKIP {sid}: no classifier-curated units", flush=True)
    return None
```

```python
annotations = decode_array(nwb["units/anno_name"][:])[good]
if np.any((annotations == "") | (annotations == "nan")):
    raise ValueError(f"Curated unit without anatomy in {path}")
```

iii. The notes repeatedly cite the QC white paper and argue that the classifier verdict, not ad hoc thresholds on scalar QC metrics, is the published inclusion rule.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI aligns neural data to go-cue onset by adding a fixed set of relative bin edges to each trial’s absolute go-cue time. Spikes are already on the same absolute session clock, so no extra offset correction is applied.

ii.
```python
OFF_START = -2.5
OFF_END = 1.5
BIN_EDGES = np.arange(OFF_START, OFF_END + BIN_WIDTH_S / 2, BIN_WIDTH_S)
...
go = go_all[keep]
absolute_edges = go[:, None] + BIN_EDGES[None, :]
```

iii. The notes say all clocks in the NWB files are absolute session time, so alignment is just subtraction of each trial’s go timestamp.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data has 50 ms resolution, yielding 80 bins per trial over the 4 s window. The AI does not rebin from a precomputed neural signal; it bins raw spikes directly at that resolution.

ii.
```python
BIN_WIDTH_S = 0.050
BIN_EDGES = np.arange(OFF_START, OFF_END + BIN_WIDTH_S / 2, BIN_WIDTH_S)
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2
N_TIME = len(BIN_CENTERS)
```

iii. The notes say this 50 ms grid is chosen because the decoder task explicitly asks for it, even though the reference paper’s downstream analyses used different sliding windows.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. The AI derives this input from `BehavioralEvents/sample_start_times`, `intervals/trials/start_time`, and `BehavioralEvents/go_start_times`. It assigns one tone to each retained trial using the first sample event that falls between trial start and go cue.

ii.
```python
sample_starts = nwb["acquisition/BehavioralEvents/sample_start_times/timestamps"][:]
tone_all = first_tone_onsets(trial_starts, go_all, sample_starts)
```

```python
def first_tone_onsets(
    trial_starts: np.ndarray, go_times: np.ndarray, sample_starts: np.ndarray
) -> np.ndarray:
    indices = np.searchsorted(sample_starts, trial_starts, side="left")
    ...
    tone = sample_starts[indices]
```

iii. The notes justify this by saying “tone onset” should mean the trial’s initial instruction onset, and that interval matching is safer than positional pairing when early licks cause epoch replays.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. For each trial, the AI computes time from tone onset as absolute bin center time minus the chosen tone time. Because it chooses the first sample event within the trial, early-lick replay trials get elapsed times measured from the first, not last, sample onset.

ii.
```python
centers_all = go_all[:, None] + BIN_CENTERS[None, :]
...
inputs = np.empty((n_trials, 2, N_TIME), dtype=np.float32)
inputs[:, 0, :] = centers - tone[:, None]
```

iii. The notes explicitly state the decision: use the first `sample_start_times` event between trial start and go because it reflects the initial instruction onset.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. The time-from-tone input is evaluated at the same bin centers used for the neural activity. Neural bins are defined relative to go cue, and the tone-derived input is simply those same absolute bin-center timestamps re-expressed relative to tone onset.

ii.
```python
centers_all = go_all[:, None] + BIN_CENTERS[None, :]
...
inputs[:, 0, :] = centers - tone[:, None]
```

iii. The notes describe this as using one shared go-aligned grid so there is no separate resampling step between neural and input streams.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. Photostimulation is derived from the trial table’s `photostim_onset`, `photostim_duration`, and `start_time` columns.

ii.
```python
onset = numeric_or_nan(trial_table["photostim_onset"][:n_trials_recorded])[keep_idx]
duration = numeric_or_nan(trial_table["photostim_duration"][:n_trials_recorded])[keep_idx]
kept_trial_starts = trial_starts[keep_idx]
stim_start = kept_trial_starts + onset
stim_stop = stim_start + duration
```

iii. The notes say they intentionally use the actual event interval from the NWB trial metadata rather than assuming the nominal stimulation timing described in the methods.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. The AI converts onset and duration strings to floats, computes absolute stimulation start and stop times for each trial, and marks a bin as 1 if its center falls within `[stim_start, stim_stop)`. Trials with missing onset or duration become all-zero because the finite checks fail.

ii.
```python
def numeric_or_nan(values) -> np.ndarray:
    out = np.full(len(values), np.nan, dtype=np.float64)
    for i, value in enumerate(values):
        try:
            out[i] = float(decode_scalar(value))
        except (TypeError, ValueError):
            pass
    return out
```

```python
inputs[:, 1, :] = (
    np.isfinite(stim_start[:, None])
    & np.isfinite(stim_stop[:, None])
    & (centers >= stim_start[:, None])
    & (centers < stim_stop[:, None])
).astype(np.float32)
```

iii. The notes describe this as preserving the actual trial-by-trial stimulation interval and representing it as a time-varying binary series.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The AI computes photostimulation on the same absolute bin-center timestamps used for neural activity. Because both are expressed on the session clock, no further alignment correction is needed.

ii.
```python
centers_all = go_all[:, None] + BIN_CENTERS[None, :]
...
stim_start = kept_trial_starts + onset
stim_stop = stim_start + duration
inputs[:, 1, :] = (... & (centers >= stim_start[:, None]) & (centers < stim_stop[:, None]))
```

iii. The notes say the trial-relative photostimulation interval is placed directly on the go-aligned time axis.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is not read from a dedicated field. The AI derives it from trial-table `outcome` and `trial_instruction`.

ii.
```python
outcomes = decode_array(trial_table["outcome"][:n_trials_recorded])[keep_idx]
instructions = decode_array(trial_table["trial_instruction"][:n_trials_recorded])[keep_idx]
choices = choice_codes(outcomes, instructions)
```

iii. The notes say this matches the reported task logic: hit means the instructed direction was chosen, miss means the opposite direction, and ignore means no lick.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The AI maps `left -> 0`, `right -> 1`, and `no lick -> 2`, then broadcasts the per-trial code across all 80 bins.

ii.
```python
code = np.full(len(outcomes), 2, dtype=np.int8)  # ignore -> no lick
hit = outcomes == "hit"
miss = outcomes == "miss"
code[hit & (instructions == "left")] = 0
code[hit & (instructions == "right")] = 1
code[miss & (instructions == "right")] = 0
code[miss & (instructions == "left")] = 1
```

```python
outputs[:, 0, :] = choices[:, None]
```

iii. The notes explain that this preserves per-trial semantics while keeping all outputs in a shared `(4, 80)` array.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived directly from the trial-table `outcome` column.

ii.
```python
outcomes = decode_array(trial_table["outcome"][:n_trials_recorded])[keep_idx]
```

iii. The notes treat these three labels as already matching the requested decoder categories.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The AI maps `ignore -> 0`, `miss -> 1`, and `hit -> 2`, then repeats that value across all time bins for the trial.

ii.
```python
outcome_lookup = {name: i for i, name in enumerate(OUTCOME_VALUES)}
outcome_codes = np.asarray([outcome_lookup[x] for x in outcomes], dtype=np.int8)
...
outputs[:, 1, :] = outcome_codes[:, None]
```

iii. The notes describe outcome as a per-trial categorical label broadcast through time so all outputs share one tensor shape.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Early lick is derived directly from the trial-table `early_lick` column.

ii.
```python
early_labels = decode_array(trial_table["early_lick"][:n_trials_recorded])[keep_idx]
```

iii. The notes identify this as the NWB field already encoding the required binary label.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The AI encodes `no early -> 0` and `early -> 1`, then repeats the per-trial value across all bins.

ii.
```python
if not set(np.unique(early_labels)).issubset({"no early", "early"}):
    raise ValueError(f"Unknown early-lick labels in {path}: {np.unique(early_labels)}")
early_codes = (early_labels == "early").astype(np.int8)
...
outputs[:, 2, :] = early_codes[:, None]
```

iii. The notes say this mirrors the requested per-trial categorical output while keeping a common output tensor shape.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. The AI derives tongue y-position from `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, using the series timestamps, the y coordinate column, and the likelihood column.

ii.
```python
tongue_group = nwb["acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking"]
tongue_t = tongue_group["timestamps"][:]
tongue_data = tongue_group["data"][:]
...
tongue_y = tongue_data[:, 1]
tongue_likelihood = tongue_data[:, 2]
```

iii. The notes say Camera0 side-view tracking is the canonical tongue source and that x, y, and likelihood are present in every session.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The AI first marks frames as visible when `likelihood >= 0.9`. It computes session `q40` and `q60` directly from all visible frame-level y values, not from 50 ms bin means. Then, for each decoder bin center, it samples the nearest preceding camera frame if that frame is no more than 10 ms old; visible sampled values are thresholded into classes 0/1/2 and everything else becomes class 3.

ii.
```python
TONGUE_LIKELIHOOD_THRESHOLD = 0.9
...
session_visible = (
    np.isfinite(tongue_y)
    & np.isfinite(tongue_likelihood)
    & (tongue_likelihood >= TONGUE_LIKELIHOOD_THRESHOLD)
)
q40, q60 = np.percentile(tongue_y[session_visible], [40, 60])
```

```python
frame_idx_all = np.searchsorted(tongue_t, centers_all, side="right") - 1
safe_frame_idx = np.clip(frame_idx_all, 0, len(tongue_t) - 1)
frame_lag = centers_all - tongue_t[safe_frame_idx]
frame_is_current_all = (frame_lag >= -1e-9) & (frame_lag <= 0.010)
```

```python
sampled_y = tongue_y[frame_idx]
sampled_likelihood = tongue_likelihood[frame_idx]
sampled_visible = (
    np.isfinite(sampled_y)
    & np.isfinite(sampled_likelihood)
    & (sampled_likelihood >= TONGUE_LIKELIHOOD_THRESHOLD)
    & frame_is_current
)
tongue_codes = np.full((n_trials, N_TIME), 3, dtype=np.int8)
```

iii. The notes justify two departures from the reference: using a “standard DLC” visibility threshold of 0.9 because the likelihood distribution is strongly bimodal, and using the nearest preceding frame at each bin center because the reference Sherlock marker-alignment code samples preceding video frames rather than averaging within windows.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. The AI uses session-wide `q40` and `q60` computed from visible frame-level y values. It assigns class 0 for `< q40`, class 1 for `q40 <= y <= q60`, class 2 for `> q60`, and class 3 for not visible or noncontemporaneous samples.

ii.
```python
q40, q60 = np.percentile(tongue_y[session_visible], [40, 60])
...
tongue_codes = np.full((n_trials, N_TIME), 3, dtype=np.int8)
tongue_codes[sampled_visible & (sampled_y < q40)] = 0
tongue_codes[sampled_visible & (sampled_y >= q40) & (sampled_y <= q60)] = 1
tongue_codes[sampled_visible & (sampled_y > q60)] = 2
```

iii. The notes explicitly state the boundary convention `<q40`, `q40<=y<=q60`, `>q60`, and explain that invisible frames should remain in the requested “not visible” class rather than be imputed.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The AI aligns tongue output by evaluating one sampled camera frame per decoder bin center. It finds the most recent frame at or before each bin center, accepts it only if it is within 10 ms, and otherwise marks that timepoint as not visible.

ii.
```python
centers_all = go_all[:, None] + BIN_CENTERS[None, :]
frame_idx_all = np.searchsorted(tongue_t, centers_all, side="right") - 1
safe_frame_idx = np.clip(frame_idx_all, 0, len(tongue_t) - 1)
frame_lag = centers_all - tongue_t[safe_frame_idx]
frame_is_current_all = (frame_lag >= -1e-9) & (frame_lag <= 0.010)
```

iii. The notes justify this with the method paper’s marker-alignment approach and by saying that missing or stale video should map to class 3 instead of causing trial deletion.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several missing-data cases explicitly. Non-numeric photostimulation entries become NaN, missing or stale tongue frames become class 3, sessions with no classifier-curated units are skipped, and all-zero population windows beyond actual spike coverage are excluded. It also validates subject-ID consistency and anatomical labels for curated units.

ii.
```python
def numeric_or_nan(values) -> np.ndarray:
    out = np.full(len(values), np.nan, dtype=np.float64)
    ...
```

```python
if len(good_indices) == 0:
    print(f"SKIP {sid}: no classifier-curated units", flush=True)
    return None
...
frame_is_current_all = (frame_lag >= -1e-9) & (frame_lag <= 0.010)
...
tongue_codes = np.full((n_trials, N_TIME), 3, dtype=np.int8)
```

```python
neural_present = np.any(rates != 0, axis=(1, 2))
if not np.all(neural_present):
    ...
    rates = rates[neural_present]
```

iii. The notes describe missing/noncontemporaneous video as a legitimate “not visible” state, while trials or sessions without usable neural data are dropped so the converter does not fabricate zeros as if they were real recordings.

## 10-a. What are the most time-consuming steps of the code?

i. The AI identifies full-session spike binning and HDF5 I/O as the dominant costs, especially per-unit spike slicing plus `searchsorted` across all trial edges. The notes also call out the intrinsic cost of serializing an approximately 11 GB pickle.

ii.
```python
rates = np.empty((n_trials, n_units_good, N_TIME), dtype=np.float32)
for out_unit, source_unit in enumerate(good_indices):
    start = 0 if source_unit == 0 else int(spike_ends[source_unit - 1])
    stop = int(spike_ends[source_unit])
    spikes = spike_values[start:stop]
    edge_positions = np.searchsorted(spikes, absolute_edges.ravel(), side="left")
    counts = np.diff(edge_positions.reshape(n_trials, N_TIME + 1), axis=1)
    rates[:, out_unit, :] = counts / BIN_WIDTH_S
```

iii. In Step 6 and Step 9 of `CONVERSION_NOTES.md`, the AI says naive unit-by-trial loops and repeated HDF5 access would dominate runtime, estimates the full neural payload at about 11 GB, and reports total full-conversion runtime.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI already vectorizes most of the trial/time work. The remaining obvious Python loop is over units during spike binning, which it keeps because spike trains are ragged. Unlike the human reference, it does not keep a per-trial tongue-binning loop; tongue alignment is vectorized with one `searchsorted` over all trial/bin centers.

ii.
```python
absolute_edges = go[:, None] + BIN_EDGES[None, :]
...
for out_unit, source_unit in enumerate(good_indices):
    ...
    edge_positions = np.searchsorted(spikes, absolute_edges.ravel(), side="left")
```

```python
frame_idx_all = np.searchsorted(tongue_t, centers_all, side="right") - 1
safe_frame_idx = np.clip(frame_idx_all, 0, len(tongue_t) - 1)
frame_lag = centers_all - tongue_t[safe_frame_idx]
```

iii. The notes explicitly say they added vectorized all-trial `searchsorted` per unit to avoid tens of millions of Python unit/trial loops, and they frame the remaining per-unit ragged loop as the unavoidable bottleneck.

## 10-c. What processing does the code repeat multiple times?

i. The AI repeats a full-file inventory pass before conversion. That means every NWB file is opened once in `discover_inventory` and again in `convert_session`, rereading subject IDs, classification labels, and anatomy labels. Within a session it otherwise computes each converted tensor once.

ii.
```python
def discover_inventory(paths: list[str]) -> tuple[list[str], list[str]]:
    for path in paths:
        with h5py.File(path, "r") as nwb:
            subject = decode_scalar(nwb["general/subject/subject_id"][()])
            classification = decode_array(nwb["units/classification"][:])
            ...
```

```python
for path in paths:
    result = convert_session(
        path,
        subject_lookup,
        region_lookup,
        ...
    )
```

iii. The notes justify the inventory pass because the target format needs global `subjects` and `brain_regions` lists up front, but this still duplicates some file I/O relative to a strict single-pass assembly.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI does several extra computations that are not needed for the final decoder inputs/outputs themselves: a full inventory pass over all files, optional processing plots, `video_has_data` counts and other session-summary statistics, and raw/free-water bookkeeping that is only stored in metadata. It also computes fine-grained region vocabularies from full `anno_name` strings rather than the coarser pooled regions used in the human reference.

ii.
```python
subjects, brain_regions = discover_inventory(paths)
...
video_has_data = np.any(frame_is_current_all, axis=1)
...
session_stats = {
    ...
    "n_trials_with_any_video": int(np.sum(video_has_data[keep_idx])),
    "auto_water_trials_converted": int(np.sum(auto_water != 0)),
    "free_water_trials_converted": int(np.sum(free_water != 0)),
}
...
if show_processing:
    make_processing_plot(...)
```

iii. The notes frame these as sanity checks and documentation aids rather than core conversion outputs, so they are intentional overhead rather than accidental bugs.
