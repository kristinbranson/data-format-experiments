# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI finds every `/app/data/sub-*/*.nwb` file, sorts the paths, and processes each file as one session with `pynwb.NWBHDF5IO`. It reads the NWB subject, trials table, units table, behavioral events, and behavioral time series.

ii.
```python
files = sorted(DATA_DIR.glob("sub-*/*.nwb"))
for path in files:
    session = convert_session(path, ...)
```
```python
with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
    nwb = io.read()
    units = nwb.units
    trials_all = nwb.trials.to_dataframe()
```

iii. The AI states that the release contains one NWB per session and that PyNWB is required. Sorting makes processing deterministic. It reports finding 174 NWBs and retaining 173 sessions after unit QC.

## 1-b. How are the data split into subjects?

i. Each session obtains its subject from `nwb.subject.subject_id`. After conversion, unique subject strings are sorted and every session receives an integer index into that list.

ii.
```python
"subject": str(nwb.subject.subject_id),
subjects = sorted({s["subject"] for s in converted})
subject_lookup = {name: i for i, name in enumerate(subjects)}
"subject_idx": np.asarray([subject_lookup[s["subject"]] for s in converted], dtype=np.int16),
```

iii. The NWB subject field is treated as the canonical mouse identifier. The notes validate 28 unique subjects.

## 1-c. How are the data split into sessions?

i. One NWB file is treated as one session. Output session order is sorted path order, and `nwb.identifier` is retained as the session ID.

ii.
```python
files = sorted(DATA_DIR.glob("sub-*/*.nwb"))
"session_id": nwb.identifier,
"neural": [s["neural"] for s in converted],
```

iii. The AI justifies this from the dataset layout and reports 173 usable sessions: 174 files minus one with no classifier-good units.

## 1-d. How are the data split into trials?

i. NWB trial-table rows define trials. Event timestamps are associated with each row's start/stop interval; the AI requires exactly one go event and at least one sample event, taking the last sample event at or before go.

ii.
```python
trials_all = nwb.trials.to_dataframe()
go_by_trial = _events_by_trial(_events(nwb, "go_start_times"), starts, stops)
sample_by_trial = _events_by_trial(_events(nwb, "sample_start_times"), starts, stops)
if not all(len(x) == 1 for x in go_by_trial):
    raise ValueError(...)
```

iii. The notes explain that early licks can replay the sample epoch, so multiple sample events may occur, whereas each trial should have one go cue.

## 1-e. How are trials filtered based on quality controls?

i. The AI initially keeps all trial rows, bins the selected population, and then removes a trial if every classifier-good unit has zero spikes throughout the requested four-second window. It retains early-lick, ignore, stimulation, auto-water, and free-water trials. Sessions with fewer than two surviving trials are dropped.

ii.
```python
neural_present = np.any(rates != 0, axis=(1, 2))
if n_excluded_no_neural:
    trials = trials.loc[neural_present].copy()
    inputs = inputs[neural_present]
    outputs = outputs[neural_present]
    rates = rates[neural_present]
if len(trials) < 2:
    return None
```

iii. The AI found behavioral trials extending beyond ephys recording. It rejected an all-unit observation-interval intersection as too strict and chose population spike presence as an empirical recording-presence test. This leaves 90,859 trials. It intentionally keeps behavioral categories needed as decoder outputs.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from `units['spike_times']` for units whose `classification` is `good`, using behavioral `go_start_times` to define absolute bin edges.

ii.
```python
classifications = np.asarray(units["classification"][:]).astype(str)
good_indices = np.flatnonzero(classifications == "good")
spikes = _unit_spike_times(units, int(unit_idx))
absolute_edges = go_times[:, None] + BIN_EDGES[None, :]
```

iii. The notes identify extracellular spike times as the raw neural signal and the regional classifier verdict as the appropriate published QC.

## 2-b. How is the `neural` data processed?

i. For each retained unit, the AI uses `searchsorted` at all trial bin edges, differences cumulative positions to obtain spike counts, and divides by 0.05 s to produce unsmoothed firing rates in Hz.

ii.
```python
positions = np.searchsorted(spikes, flattened, side="left")
counts = np.diff(positions.reshape(len(go_times), N_TIME + 1), axis=1)
rates[:, out_idx, :] = counts.astype(np.float32) / BIN_WIDTH
```

iii. The AI says this matches the reference half-open histogram and rate conversion, with the task-required 50-ms bins replacing paper-specific windows.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units labeled exactly `classification == 'good'` are retained. A session with no such units is skipped; selected units must also have nonempty `anno_name` values.

ii.
```python
good_indices = np.flatnonzero(classifications == "good")
if len(good_indices) == 0:
    return None
annotations = annotations_all[good_indices]
if np.any(annotations == ""):
    raise ValueError(...)
```

iii. The notes connect this field to the Chen et al. classifier QC and validate 69,453 good session-units, close to the paper-version count.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI adds fixed offsets from -2.5 to +1.5 s to each trial's absolute go-cue time and bins absolute spike times on those edges.

ii.
```python
go_times = np.asarray([x[0] for x in go_by_trial])
absolute_edges = go_times[:, None] + BIN_EDGES[None, :]
```

iii. It argues that event, video, and spike timestamps share the NWB session clock, so no additional offset or interpolation is required.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The output has 80 non-overlapping 50-ms bins over `[-2.5, 1.5)` relative to go. Raw spikes are newly histogrammed into those bins; no smoothing or further rebinning is applied.

ii.
```python
BIN_WIDTH = 0.050
BIN_EDGES = np.arange(OFF_START, OFF_END + BIN_WIDTH / 2, BIN_WIDTH)
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2
```

iii. This directly follows the decoder specification and yields centers from -2.475 to +1.475 s.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It is derived from `sample_start_times`, trial start/stop times, and `go_start_times`. The last sample onset no later than go is selected for each trial.

ii.
```python
sample_by_trial = _events_by_trial(_events(nwb, "sample_start_times"), starts, stops)
tone_times = np.asarray([x[x <= g][-1] for x, g in zip(sample_by_trial, go_times)])
```

iii. The AI explains that early-lick trials replay the sample epoch, making the final pre-go sample the completed tone relevant to that trial.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. Each go-aligned bin center is converted to an absolute timestamp, then the trial's selected tone timestamp is subtracted. Values are stored as float32 seconds.

ii.
```python
absolute_centers = go_times[:, None] + BIN_CENTERS[None, :]
inputs[:, 0, :] = (absolute_centers - tone_times[:, None]).astype(np.float32)
```

iii. The AI follows the explicit request for a continuous time-varying input, despite a generic instruction elsewhere suggesting binary encodings for event times.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is evaluated at exactly the same go-aligned 50-ms bin centers as the neural bins.

ii.
```python
absolute_centers = go_times[:, None] + BIN_CENTERS[None, :]
inputs[:, 0, :] = absolute_centers - tone_times[:, None]
```

iii. Shared absolute timestamps and the common bin-center grid guarantee corresponding input and neural time indices.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. The AI uses `BehavioralEvents/photostim_start_times` and `photostim_stop_times`, associated with trial start/stop intervals.

ii.
```python
stim_starts_by_trial = _events_by_trial(_events(nwb, "photostim_start_times"), starts, stops)
stim_stops_by_trial = _events_by_trial(_events(nwb, "photostim_stop_times"), starts, stops)
```

iii. The event pairs provide direct absolute laser-on and laser-off times; the AI also checks that starts and stops are paired.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. The input begins as zero and is set to one wherever a bin center falls in any half-open stimulation interval `[onset, offset)`.

ii.
```python
inputs[:, 1, :] = 0
for i, (onsets, offsets) in enumerate(zip(stim_starts_by_trial, stim_stops_by_trial)):
    for onset, offset in zip(onsets, offsets):
        active = ((absolute_centers[i] >= onset) & (absolute_centers[i] < offset))
        inputs[i, 1, active] = 1.0
```

iii. This represents the requested instantaneous binary state, handles multiple intervals, and uses a documented half-open convention.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Absolute laser intervals are tested at the same absolute go-aligned bin centers used for neural time bins.

ii.
```python
absolute_centers = go_times[:, None] + BIN_CENTERS[None, :]
active = ((absolute_centers[i] >= onset) & (absolute_centers[i] < offset))
```

iii. All streams share the NWB session clock, so direct timestamp comparison supplies the alignment.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is derived from trial-table `trial_instruction` and `outcome`: ignore means no lick, hit means the instructed side, and miss means the opposite side.

ii.
```python
choice = np.asarray([
    _classify_choice(inst, outcome)
    for inst, outcome in zip(instructions, outcomes_text)
], dtype=np.int8)
```

iii. The AI notes that actual choice is not stored directly but is determined by instruction and correctness/outcome under this two-choice task.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. It maps left/right/no lick to 0/1/2 and repeats the per-trial class across all 80 time bins.

ii.
```python
if outcome == "ignore": return 2
if outcome == "hit": return 0 if instruction == "left" else 1
if outcome == "miss": return 1 if instruction == "left" else 0
outputs[:, 0, :] = choice[:, None]
```

iii. Repetition lets the four outputs share a uniform `(4, 80)` array while preserving the per-trial semantics.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. It comes directly from the NWB trials-table `outcome` column.

ii.
```python
outcomes_text = trials["outcome"].astype(str).to_numpy()
```

iii. The stored values already correspond to the requested ignore, miss, and hit categories.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Strings are mapped to `ignore=0`, `miss=1`, `hit=2`, then repeated across all bins.

ii.
```python
outcome_map = {"ignore": 0, "miss": 1, "hit": 2}
outcome = np.asarray([outcome_map[x] for x in outcomes_text], dtype=np.int8)
outputs[:, 1, :] = outcome[:, None]
```

iii. The mapping follows the requested output ordering; repetition provides a common time-shaped output representation.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. It comes directly from the trials-table `early_lick` column.

ii.
```python
early_text = trials["early_lick"].astype(str).to_numpy()
```

iii. The trial table explicitly provides this behavioral flag.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The AI maps `no early` to 0 and `early` to 1 and repeats the trial label across all 80 bins.

ii.
```python
early_map = {"no early": 0, "early": 1}
early = np.asarray([early_map[x] for x in early_text], dtype=np.int8)
outputs[:, 2, :] = early[:, None]
```

iii. This follows the requested no/yes ordering and uniform output-array layout.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. It uses `Camera0_side_TongueTracking`: timestamps, data column 1 for y-position, and column 2 for tracking likelihood.

ii.
```python
tongue = behavior.time_series["Camera0_side_TongueTracking"]
tongue_times = np.asarray(tongue.timestamps[:], dtype=np.float64)
tongue_data = np.asarray(tongue.data[:], dtype=np.float64)
tongue_y = tongue_data[:, 1]
tongue_likelihood = tongue_data[:, 2]
```

iii. The notes identify this as the side-camera DLC stream containing the requested tongue coordinate and visibility confidence.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Frames are considered visible when y and likelihood are finite and likelihood is at least 0.9. The AI computes session thresholds from all visible raw frames. For each neural bin center, it selects the nearest camera frame and assigns its y value to a class, or class 3 if that frame is not visible.

ii.
```python
DLC_VISIBLE_THRESHOLD = 0.9
session_visible = np.isfinite(tongue_y) & np.isfinite(tongue_likelihood) & (
    tongue_likelihood >= DLC_VISIBLE_THRESHOLD)
q40, q60 = np.quantile(tongue_y[session_visible], [0.4, 0.6])
nearest = _nearest_indices(tongue_times, absolute_centers.ravel())
```

iii. The AI says the confidence threshold identifies visible tongue frames and that vectorized nearest-frame matching gives one camera observation per neural bin. It chose raw visible frames, not per-bin means, as the percentile population.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Per-session 40th and 60th quantiles of visible raw-frame y values define classes: `<q40` is 0, `q40..q60` inclusive is 1, `>q60` is 2, and low-confidence/nonfinite nearest frames are 3.

ii.
```python
tongue_class = np.full((len(trials), N_TIME), 3, dtype=np.int8)
tongue_class[visible & (matched_y < q40)] = 0
tongue_class[visible & (matched_y >= q40) & (matched_y <= q60)] = 1
tongue_class[visible & (matched_y > q60)] = 2
```

iii. The AI follows the requested per-session 40/60 split and reserves class 3 for not-visible samples.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The camera sample nearest to each absolute neural-bin center is selected; no within-bin averaging is performed.

ii.
```python
nearest = _nearest_indices(tongue_times, absolute_centers.ravel()).reshape(
    len(trials), N_TIME)
matched_y = tongue_y[nearest]
```

iii. The AI justifies nearest-timestamp matching because video and neural data share an absolute clock and the vectorized operation is fast.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. A session with no good units is skipped; a good unit missing annotation raises an error; malformed go/sample or unpaired laser events raise errors; trials with an entirely zero population window are removed; nonfinite or low-confidence tongue samples become `not visible`; insufficient session-visible tongue data raises an error.

ii.
```python
if len(good_indices) == 0: return None
if np.any(annotations == ""): raise ValueError(...)
neural_present = np.any(rates != 0, axis=(1, 2))
tongue_class = np.full((len(trials), N_TIME), 3, dtype=np.int8)
```

iii. The notes distinguish absent neural recording from valid zero-valued activity and avoid imputing uncertain tongue positions. They emphasize assertions and fail-fast checks for malformed event data.

## 10-a. What are the most time-consuming steps of the code?

i. NWB I/O, loading ragged spikes/video, per-unit spike binning, retaining the large neural arrays, and serializing the roughly 11-GiB pickle dominate. Optional plot generation adds work only when requested.

ii.
```python
for out_idx, unit_idx in enumerate(good_indices):
    spikes = _unit_spike_times(units, int(unit_idx))
    positions = np.searchsorted(spikes, flattened, side="left")
with args.outpicklefile.open("wb") as stream:
    pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The AI reports a 223.43-second full conversion and specifically identifies naive spike counting and excess retained raw arrays as the performance risks.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. A per-unit loop remains because spike vectors are ragged. Trial/bin spike operations are already vectorized. Photostimulation retains nested trial/interval loops, and dataset validation loops over every trial; these could be further vectorized or consolidated. Session conversion remains sequential.

ii.
```python
for out_idx, unit_idx in enumerate(good_indices): ...
for i, (onsets, offsets) in enumerate(zip(stim_starts_by_trial, stim_stops_by_trial)):
    for onset, offset in zip(onsets, offsets): ...
for s in range(ns):
    for neural, inp, out in zip(...): ...
```

iii. The notes highlight the important optimization: one `searchsorted` per unit over all trial edges and one vectorized nearest-camera lookup. They consider the remaining ragged-unit loop natural.

## 10-c. What processing does the code repeat multiple times?

i. Trial interval association is separately repeated for go, sample, photostim-start, and photostim-stop event vectors. The output is traversed again for validation and summary counts after conversion. Event names also cause repeated acquisition lookups.

ii.
```python
go_by_trial = _events_by_trial(_events(nwb, "go_start_times"), starts, stops)
sample_by_trial = _events_by_trial(_events(nwb, "sample_start_times"), starts, stops)
stim_starts_by_trial = _events_by_trial(...)
stim_stops_by_trial = _events_by_trial(...)
validate_converted(data)
print_summary(data)
```

iii. The AI characterizes the core conversion as a single pass per file and reuses global bin grids, although validation and reporting intentionally make extra passes over converted arrays.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. `starts_all` and `stops_all` are computed but never used. `_any_observation_mask` is defined but never called. When plotting is enabled, six-panel diagnostic figures are generated but do not affect the pickle. Several metadata diagnostics and repeated full-data validation/summary scans are useful for auditing but not decoder inputs.

ii.
```python
starts_all = trials_all["start_time"].to_numpy(np.float64)
stops_all = trials_all["stop_time"].to_numpy(np.float64)
def _any_observation_mask(...):
    ...
if show_processing:
    _plot_processing(...)
```

iii. The notes justify plots, metadata, validation, and summaries as sanity checks. The unused variables/helper appear to be remnants of the abandoned observation-interval filtering approach and have no stated downstream purpose.
