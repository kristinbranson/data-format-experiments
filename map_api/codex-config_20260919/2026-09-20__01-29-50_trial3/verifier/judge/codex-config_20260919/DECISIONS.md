# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent finds every `sub-*/*.nwb` file, sorts the paths, and opens each session once with `pynwb.NWBHDF5IO`. It reads trials, units, behavioral events, and tracking from each NWB and accumulates retained sessions before pickling.

ii.
```python
files = sorted(DATA_DIR.glob("sub-*/*.nwb"))
for i, path in enumerate(files):
    session, session_audit = convert_session(path, ...)
```
```python
with NWBHDF5IO(str(path), mode="r", load_namespaces=True) as io:
    nwb = io.read()
    trials = nwb.trials.to_dataframe()
```

iii. The notes say this obeys the mandatory `pynwb` constraint, treats the 174 NWBs as sessions, and omits only the session with no classifier-good units, yielding the expected 173 sessions and 28 subjects.

## 1-b. How are the data split into subjects?

i. Each retained session gets its subject from `nwb.subject.subject_id`; unique IDs are sorted globally and each session receives an integer lookup index.

ii.
```python
subject = str(nwb.subject.subject_id)
subjects = sorted({x["subject"] for x in converted})
subject_idx = np.asarray([subject_lookup[x["subject"]] for x in converted], dtype=np.int32)
```

iii. The agent treats the NWB subject field as authoritative and reports 28 subjects, matching the release and paper.

## 1-c. How are the data split into sessions?

i. One NWB file is treated as one session. Retained sessions remain in sorted file order and store `nwb.identifier` and source path in metadata.

ii.
```python
files = sorted(DATA_DIR.glob("sub-*/*.nwb"))
session_id = str(nwb.identifier)
```

iii. The notes identify the NWB boundary as the session boundary and report 173 retained sessions after dropping the one without good units.

## 1-d. How are the data split into trials?

i. Rows of `nwb.trials` define trials. Go events must have the same count; retained row indices are used consistently for all streams, and final arrays are converted into per-trial lists.

ii.
```python
trials = nwb.trials.to_dataframe()
go_all = np.asarray(events["go_start_times"].timestamps[:], dtype=np.float64)
if len(go_all) != len(trials):
    raise ValueError(...)
```
```python
for j in range(len(trial_inds)):
    neural.append(rates[:, j, :])
```

iii. The trials table is the dataset's native trial definition; count checks and common indices are used to prevent stream misalignment.

## 1-e. How are trials filtered based on quality controls?

i. For every retained unit, the agent maps insertion-local `is_good_trials` through that unit's `obs_intervals` to the full trial table, then retains only the intersection valid for every unit. It additionally removes any retained trial whose entire four-second population activity is zero. It deliberately does not directly remove early, ignore, miss, stimulation, auto-water, or free-water trials.

ii.
```python
unit_valid = np.stack([
    full_unit_trial_mask(nwb.units, i, trial_starts, trial_stops) for i in unit_inds
])
trial_mask = np.all(unit_valid, axis=0)
```
```python
recorded_trial = np.any(rates != 0, axis=(0, 2))
rates = rates[:, recorded_trial, :]
```

iii. The agent argues that intersecting validity preserves a constant neuron set and that a wholly silent population indicates an export/recording gap. Requested behavioral classes are retained for decoder coverage. The notes report 1,569 interval-mask removals and 2,423 population-zero removals.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from each classifier-good unit's `units["spike_times"]`, using behavioral `go_start_times` to position trial bin edges.

ii.
```python
spikes = np.asarray(nwb.units["spike_times"][unit_i], dtype=np.float64)
absolute_edges = go[:, None] + EDGES[None, :]
```

iii. The notes state that spike timestamps are the appropriate native neural representation and all NWB timestamps share the session clock.

## 2-b. How is the `neural` data processed?

i. For each good unit, `searchsorted` counts spikes between all consecutive absolute edges; counts are divided by 0.05 seconds to obtain spikes/s. No smoothing, normalization, or baseline subtraction is used.

ii.
```python
edge_indices = np.searchsorted(spikes, absolute_edges, side="left")
rates[out_i] = np.diff(edge_indices, axis=1) / BIN_SIZE_S
```

iii. The agent says this matches the reference's half-open-window histogram/rate calculation while efficiently vectorizing over trials.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units whose NWB `classification` string equals `"good"` are retained; sessions with none are dropped. No individual metric thresholds or older `unit_quality` label are used.

ii.
```python
classifications = np.asarray(nwb.units["classification"][:]).astype(str)
unit_inds = np.flatnonzero(classifications == "good")
if len(unit_inds) == 0:
    return None, audit
```

iii. The notes identify this as the paper's region-specific 15-metric classifier and reject ad hoc single-metric thresholds. The retained 69,453 units match the release labels.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial's go timestamp is added to the fixed relative edge grid, and absolute spike timestamps are binned against those edges.

ii.
```python
go = go_all[trial_inds]
absolute_edges = go[:, None] + EDGES[None, :]
```

iii. Because spikes and events use one absolute NWB clock, the agent says no offset correction or interpolation is required.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The code forms 80 non-overlapping 50-ms bins over `[-2.5, 1.5)` seconds relative to go onset. Raw spike timestamps are binned once; no later rebinning occurs.

ii.
```python
BIN_SIZE_S = 0.050
EDGES = np.linspace(OFF_START, OFF_END, 81, dtype=np.float64)
CENTERS = (EDGES[:-1] + EDGES[1:]) / 2
```

iii. This directly implements the requested resolution and window, with an explicit left-closed/right-open convention.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It uses behavioral `sample_start_times`, trial start times, and go-cue times. The last sample/tone onset at or before each go cue is chosen and checked to fall within the trial.

ii.
```python
sample_times = np.asarray(events["sample_start_times"].timestamps[:], dtype=np.float64)
tone_all = last_sample_before_go(sample_times, trial_starts, go_all)
```

iii. The agent explains that early licking can replay an epoch, so the final completed sample before go is the relevant tone.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. At every neural-bin center, the absolute time is computed and the selected tone timestamp is subtracted.

ii.
```python
tone_time = go[:, None] + CENTERS[None, :] - tone[:, None]
```

iii. This produces a continuous elapsed-seconds signal without additional transformations.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is evaluated at `go + CENTERS`, the centers of the exact bins whose edges were used for neural rates.

ii.
```python
absolute_edges = go[:, None] + EDGES[None, :]
tone_time = go[:, None] + CENTERS[None, :] - tone[:, None]
```

iii. The shared go-relative grid ensures one input value per neural time bin.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. It uses `BehavioralEvents/photostim_start_times` and `photostim_stop_times` rather than the corresponding trial-table onset/duration fields.

ii.
```python
laser_starts = np.asarray(events["photostim_start_times"].timestamps[:], dtype=np.float64)
laser_stops = np.asarray(events["photostim_stop_times"].timestamps[:], dtype=np.float64)
```

iii. The agent regards absolute event timestamps as authoritative and validates matched start/stop counts.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A binary value is set to one for every trial/bin having any positive-duration overlap with any session laser interval.

ii.
```python
for onset, offset in zip(starts, stops):
    result[np.logical_and(left < offset, right > onset)] = 1.0
```

iii. The agent says this vectorizes each comparatively small set of laser intervals over all trial bins and represents whether stimulation is on at each time point.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Absolute laser intervals are compared with the same absolute bin edges (`go + EDGES`) used to count spikes.

ii.
```python
absolute_edges = go[:, None] + EDGES[None, :]
laser = photostim_bins(laser_starts, laser_stops, absolute_edges)
```

iii. The common clock and exact edge grid provide direct alignment without resampling.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is derived from trial-table `trial_instruction` and `outcome`: hit means instructed side, miss means the opposite side, and ignore means no lick. Raw left/right lick events are also read only for an audit.

ii.
```python
choices = np.asarray([derive_choice(a, b) for a, b in zip(instructions, outcomes_str)])
```
```python
if outcome == "ignore": return 2
return instructed if outcome == "hit" else 1 - instructed
```

iii. The agent says the two-port task makes choice deterministic from instruction and outcome; response-window lick events independently matched 99.69% and were not used as labels.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The result is coded `0=left`, `1=right`, `2=no lick`, stored as `int8`, and broadcast across all 80 time bins.

ii.
```python
out[0] = choices[j]
```
```python
["left", "right", "no lick"]
```

iii. Broadcasting gives a uniform `(4,80)` output representation for a per-trial categorical label.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. It comes directly from the trials-table `outcome` column.

ii.
```python
outcomes_str = tr.outcome.astype(str).to_numpy()
```

iii. The raw column already supplies exactly the requested categories.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Strings map to `ignore=0`, `miss=1`, and `hit=2`; the per-trial code is broadcast across time.

ii.
```python
outcome_map = {"ignore": 0, "miss": 1, "hit": 2}
out[1] = outcomes[j]
```

iii. The ordering follows the requested output values and uniform output shape.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. It comes directly from the trials-table `early_lick` column.

ii.
```python
early_str = tr.early_lick.astype(str).to_numpy()
```

iii. The trials table explicitly provides the requested flag.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. `"no early"` maps to 0 and `"early"` to 1; the code is broadcast across all bins.

ii.
```python
early_map = {"no early": 0, "early": 1}
out[2] = early[j]
```

iii. The agent uses a compact categorical representation consistent with the output schema.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. It uses timestamps and columns 1 (y) and 2 (tracking likelihood) from `Camera0_side_TongueTracking`.

ii.
```python
tracking = nwb.acquisition["BehavioralTimeSeries"].time_series["Camera0_side_TongueTracking"]
raw_y, likelihood = track_data[:, 1], track_data[:, 2]
```

iii. The notes identify this as the paper's side-camera DeepLabCut tongue stream.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Frames with finite y/likelihood and likelihood at least 0.9 define visible data. Session q40/q60 are computed from all such raw frame y values. For each neural-bin center, the nearest camera frame is selected and its y/likelihood is classified; no within-bin averaging is done.

ii.
```python
visible = np.isfinite(raw_y) & np.isfinite(likelihood) & (likelihood >= 0.9)
q40, q60 = np.quantile(raw_y[visible], [0.4, 0.6])
track_idx = nearest_indices(track_t, query.ravel()).reshape(query.shape)
```

iii. The agent calls 0.9 a conservative visibility threshold, computes percentiles over all visible session frames as it interprets the request, and uses nearest-frame lookup to synchronize video with bin centers.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Visible nearest-frame y is categorized as 0 below q40, 1 from q40 through q60 inclusive, and 2 above q60; invisible samples remain 3.

ii.
```python
tongue_class = np.full(query.shape, 3, dtype=np.int8)
tongue_class[tongue_visible & (y < q40)] = 0
tongue_class[tongue_visible & (y >= q40) & (y <= q60)] = 1
tongue_class[tongue_visible & (y > q60)] = 2
```

iii. This explicitly implements the requested boundary categories and preserves non-visible tongue as a fourth category.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The code queries camera data at each absolute neural-bin center (`go + CENTERS`) and chooses the temporally nearest frame.

ii.
```python
query = go[:, None] + CENTERS[None, :]
track_idx = nearest_indices(track_t, query.ravel()).reshape(query.shape)
```

iii. The agent justifies nearest-frame lookup because event, spike, and camera timestamps share one clock.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The code raises errors for inconsistent event counts, missing sample onset, unmappable observation intervals, or no visible tongue. It drops sessions without good units, removes intersection-invalid or all-population-zero trials, and assigns tongue class 3 to low-confidence samples. It also checks finite values for percentile estimation but not again when classifying nearest samples.

ii.
```python
if len(laser_starts) != len(laser_stops): raise ValueError(...)
if len(unit_inds) == 0: return None, audit
recorded_trial = np.any(rates != 0, axis=(0, 2))
tongue_class = np.full(query.shape, 3, dtype=np.int8)
```

iii. The notes describe these as raw-data edge cases discovered during sample validation: insertion-local masks were mapped exactly and apparent unrecorded gaps were excluded rather than represented as physiological silence.

## 10-a. What are the most time-consuming steps of the code?

i. NWB loading and per-unit neural binning dominate session conversion; accumulating and serializing the 11.793-GB result is also costly. Full conversion took 3.19 minutes.

ii.
```python
for out_i, unit_i in enumerate(unit_inds):
    spikes = np.asarray(nwb.units["spike_times"][unit_i], dtype=np.float64)
    edge_indices = np.searchsorted(spikes, absolute_edges, side="left")
```

iii. The notes estimate roughly 1.25 seconds/session and explain that the required nested-list pickle creates an unavoidable multi-GB in-memory/serialization workload.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The remaining per-unit spike loop cannot trivially combine ragged spike trains but already vectorizes all trials. The per-unit validity-mask construction, per-laser-interval loop, response-lick audit loop, and final per-trial packaging loop could potentially be reduced or vectorized; the latter mostly constructs required list objects.

ii.
```python
for out_i, unit_i in enumerate(unit_inds): ...
for onset, offset in zip(starts, stops): ...
lick_choices = np.asarray([first_response_lick_choice(...) for g in go])
for j in range(len(trial_inds)): ...
```

iii. The agent specifically highlights replacing a unit-by-trial histogram approach with one `searchsorted` per unit across all trials and vectorizing laser/tracking lookup; it regards remaining loops as small, ragged, or required for the nested output.

## 10-c. What processing does the code repeat multiple times?

i. It repeatedly reads per-unit ragged `spike_times`, `is_good_trials`, and `obs_intervals`; it also constructs per-trial arrays in a loop after session-level arrays already exist. Under optional plotting, already-derived values are rendered for the first two sessions. The fixed time grid itself is computed only once.

ii.
```python
full_unit_trial_mask(nwb.units, i, ...) for i in unit_inds
spikes = np.asarray(nwb.units["spike_times"][unit_i], ...)
for j in range(len(trial_inds)):
    neural.append(...); inputs.append(...); outputs.append(out)
```

iii. The notes characterize conversion as a single pass with no recomputation of scientific quantities; the repetitions above are structural iteration over units/trials rather than repeated analytical passes.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It reconstructs choice a second time from raw lick events solely for audit counters; those `lick_choices` are not stored or used as labels. Timing/ETA/audit bookkeeping and optional diagnostic plots likewise do not enter decoder data.

ii.
```python
lick_choices = np.asarray([
    first_response_lick_choice(left_licks, right_licks, g) for g in go
])
audit["choice_lick_matches"] += int(np.sum(choices == lick_choices))
```

iii. The agent intentionally uses this discarded computation as an independent sanity check, reporting 99.69% agreement and retaining instruction/outcome as authoritative.
