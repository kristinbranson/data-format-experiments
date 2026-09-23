# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent sorts every `/app/data/sub-*/*.nwb` path, treats each file as a session, and opens it once with PyNWB. Full mode processes all files; sample mode stops after two usable sessions.

ii.
```python
def nwb_files() -> list[Path]:
    return sorted(DATA_DIR.glob("sub-*/*.nwb"))
...
for path in files:
    session, diagnostics = process_session(path, region_to_idx)
...
with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
    nwb = io.read()
```

iii. The notes say there are 174 NWBs, one session per file, and that PyNWB is required. Sorting provides deterministic subject/session order, and opening each file once limits repeated I/O.

## 1-b. How are the data split into subjects?

i. The subject is read from `nwb.subject.subject_id`. Subjects receive an index on first encounter in the sorted file traversal; each retained session stores that index.

ii.
```python
"subject": str(nwb.subject.subject_id),
...
if subject not in subject_to_idx:
    subject_to_idx[subject] = len(subjects)
    subjects.append(subject)
subject_idx.append(subject_to_idx[subject])
```

iii. The agent regarded the NWB subject field as canonical and verified that the conversion produced the expected 28 subjects.

## 1-c. How are the data split into sessions?

i. Each NWB file is one session. Retained sessions are appended in sorted path order and identified by `nwb.identifier` in metadata. A file with no classifier-good units is skipped.

ii.
```python
session = {
    ...
    "info": {"session_id": str(nwb.identifier), ...},
}
...
neural.append(session["neural"])
```

iii. Dataset exploration found that the file boundary is the native session boundary. Excluding the sole zero-good-unit file yielded the paper's 173 usable sessions.

## 1-d. How are the data split into trials?

i. Trial rows come from `nwb.trials`, but the agent keeps only the first `N` rows, where `N` is the common length of `is_good_trials` for good units. It similarly takes the first `N` go cues, then removes population-all-zero windows.

ii.
```python
trials_df_all = nwb.trials[:]
recorded_lengths = np.asarray(
    [len(nwb.units["is_good_trials"][int(i)]) for i in good], dtype=int
)
n_recorded_trials = int(recorded_lengths[0])
trials_df = trials_df_all.iloc[:n_recorded_trials].copy()
go_times = go_times_all[:n_recorded_trials]
```

iii. The agent found eight files whose behavior tables outlasted the unit validity vectors and interpreted the vector length as the ephys trial count. It then treated all-zero population windows as absent neural data.

## 1-e. How are trials filtered based on quality controls?

i. Behavioral classes (early lick, ignore, stimulation) are retained. Trials beyond the inferred recorded length and any requested window with no spike from any curated unit are removed; sessions with fewer than two remaining trials error. There is no explicit `free_water` or per-value `is_good_trials` filter.

ii.
```python
neural_valid = np.any(rates != 0, axis=(1, 2))
if excluded_all_zero:
    rates = rates[neural_valid]
    trials_df = trials_df.iloc[np.flatnonzero(neural_valid)].copy()
...
if n_trials < 2:
    raise ValueError(...)
```

iii. Required target classes made the papers' regular-trial exclusion inappropriate. The notes justify the additional removals as periods after acquisition stopped and report 2,576 population-all-zero removals, leaving 90,734 trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural values derive from `units.spike_times` for units whose `units.classification` equals `good`, and from behavioral go-cue timestamps used to define absolute bin edges.

ii.
```python
classifications = as_string_array(nwb.units["classification"])
good = np.flatnonzero(classifications == "good")
...
spike_column = units["spike_times"]
absolute_edges = go_times[:, None] + EDGES_REL[None, :]
```

iii. The notes identify classifier output as the released region-specific spike-sorting QC and spike times as the relevant electrophysiology signal.

## 2-b. How is the `neural` data processed?

i. For each good unit, `searchsorted` counts spikes in all half-open trial bins, differences cumulative indices, and divides counts by 0.05 s to obtain float32 firing rates in Hz. There is no smoothing or normalization.

ii.
```python
for out_idx, unit_idx in enumerate(good_indices):
    spikes = np.asarray(spike_column[int(unit_idx)], dtype=np.float64)
    edge_positions = np.searchsorted(spikes, absolute_edges, side="left")
    rates[:, out_idx, :] = np.diff(edge_positions, axis=1) / BIN_SIZE_S
```

iii. This preserves the reference's half-open histogram/rate convention while the task-required 50-ms nonoverlapping bins override the papers' analysis-specific sliding bins.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only `classification == "good"` units are retained. A zero-good-unit session is skipped; curated units must also have nonempty CCF annotations. No additional firing-rate or metric threshold is imposed.

ii.
```python
good = np.flatnonzero(classifications == "good")
if len(good) == 0:
    return None, {"file": path.name, "skip_reason": "zero curated units"}
...
if np.any(annotations == ""):
    raise ValueError(...)
```

iii. The classifier verdict matches the released QC described in the white paper. The notes reject the movement paper's 2-Hz threshold as specific to a different regression analysis.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial's absolute go time is added to a shared grid spanning -2.5 to +1.5 seconds; absolute spike times are counted within those edges.

ii.
```python
absolute_edges = go_times[:, None] + EDGES_REL[None, :]
edge_positions = np.searchsorted(spikes, absolute_edges, side="left")
```

iii. The agent states that spikes and behavioral events share the NWB session clock, so applying go-relative offsets gives the same alignment as the reference's already go-relative exports.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The output has 80 nonoverlapping 50-ms bins over four seconds. Raw spike timestamps are histogrammed directly; no later resampling occurs.

ii.
```python
BIN_SIZE_S = 0.050
OFF_START = -2.5
OFF_END = 1.5
EDGES_REL = np.arange(OFF_START, OFF_END + BIN_SIZE_S / 2, BIN_SIZE_S)
CENTERS_REL = EDGES_REL[:-1] + BIN_SIZE_S / 2
```

iii. These values exactly follow the decoder task and intentionally replace the papers' other analysis bin widths/strides.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It uses `BehavioralEvents/sample_start_times`, trial `start_time`, go-cue timestamps, and the neural-bin centers. The selected tone is the first sample event at or after trial start and no later than go.

ii.
```python
sample_events = np.asarray(events["sample_start_times"].timestamps[:], dtype=np.float64)
tone_onsets = first_event_per_trial(sample_events, trial_starts, go_times, "tone")
```

iii. The notes call this the "first tone/sample onset within each native trial" and use native trial bounds to avoid assigning an event from another trial.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. `first_event_per_trial` uses `searchsorted` to select and validate the tone. The value at every neural bin is the absolute bin-center timestamp minus that tone onset, cast to float32.

ii.
```python
indices = np.searchsorted(events, starts, side="left")
selected = events[indices]
...
tone_time = (centers_abs - tone_onsets[:, None]).astype(np.float32)
```

iii. The agent intended a continuous seconds-since-tone variable, as specifically requested, rather than the generic binary onset representation.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is evaluated at the exact absolute centers of the same go-aligned bins used for spike counts.

ii.
```python
centers_abs = go_times[:, None] + CENTERS_REL[None, :]
tone_time = (centers_abs - tone_onsets[:, None]).astype(np.float32)
```

iii. Shared bin centers make each input sample correspond to the matching neural column.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. It uses trials-table `photostim_onset`, `photostim_duration`, and `start_time`, plus absolute neural-bin centers.

ii.
```python
onset_raw = np.asarray(trials_df.photostim_onset).astype(str)
duration_raw = np.asarray(trials_df.photostim_duration).astype(str)
trial_start = np.asarray(trials_df.start_time, dtype=np.float64)
```

iii. The notes say these native task timing fields preserve stimulated trials as the requested decoder input.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. Non-`N/A` onset strings are converted to absolute times by adding trial start. A bin center receives 1 if it is in `[onset, onset + duration)`, otherwise 0; the result is float32.

ii.
```python
active = onset_raw != "N/A"
onset[active] = np.asarray(onset_raw[active], dtype=float) + trial_start[active]
...
return (active[:, None] & (centers_abs >= onset[:, None])
        & (centers_abs < onset[:, None] + duration[:, None])).astype(np.float32)
```

iii. The half-open interval matches the temporal-bin convention and creates the required binary time-varying state.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Absolute stimulation intervals are tested at the exact centers of the go-aligned neural bins.

ii.
```python
centers_abs = go_times[:, None] + CENTERS_REL[None, :]
photostim = trial_photostimulation(trials_df, centers_abs)
```

iii. Because task, go, and spike timestamps share the session clock, no interpolation or offset correction is needed.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is inferred from trials-table `trial_instruction` and `outcome`; lick-event streams are read only for a diagnostic check.

ii.
```python
instruction = np.asarray(trials_df.trial_instruction).astype(str)
outcome_text = np.asarray(trials_df.outcome).astype(str)
choice = derive_choice(instruction, outcome_text)
```

iii. In the two-choice task, hit means the instructed side, miss the opposite side, and ignore no lick. The agent reports 99.69% agreement with first post-go lick events.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Left/right instruction becomes 0/1, miss flips it, and ignore becomes 2. The per-trial label is repeated across all 80 bins.

ii.
```python
instructed = np.where(instruction == "left", 0, 1)
choice = instructed.copy()
choice[outcome == "miss"] = 1 - choice[outcome == "miss"]
choice[outcome == "ignore"] = 2
...
output_cube[:, 0, :] = choice[:, None]
```

iii. The encoding follows the requested left/right/no-lick order; broadcasting keeps a rectangular output alongside time-varying tongue position.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. It comes directly from the trials-table `outcome` strings.

ii.
```python
outcome_text = np.asarray(trials_df.outcome).astype(str)
```

iii. NWB already provides exactly the requested ignore, miss, and hit categories.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Strings map to ignore=0, miss=1, hit=2, are stored as uint8, and are repeated across time.

ii.
```python
outcome_map = {"ignore": 0, "miss": 1, "hit": 2}
outcome = np.asarray([outcome_map[x] for x in outcome_text], dtype=np.uint8)
output_cube[:, 1, :] = outcome[:, None]
```

iii. This follows the specified category order and the agent's common rectangular-output design.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. It comes from the trials-table `early_lick` strings.

ii.
```python
early_text = np.asarray(trials_df.early_lick).astype(str)
```

iii. The NWB trial table already records the requested flag, so no event reconstruction is needed.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Equality to `early` produces 1 and every other value 0; the uint8 per-trial label is repeated across bins.

ii.
```python
early = (early_text == "early").astype(np.uint8)
output_cube[:, 2, :] = early[:, None]
```

iii. This implements the requested no/yes coding while retaining early-lick trials as necessary prediction targets.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. It uses timestamp, y (column 1), and tracking likelihood (column 2) from `Camera0_side_TongueTracking`.

ii.
```python
ts = nwb.acquisition["BehavioralTimeSeries"].time_series[
    "Camera0_side_TongueTracking"
]
timestamps = np.asarray(ts.timestamps[:], dtype=np.float64)
tracking = np.asarray(ts.data[:], dtype=np.float64)
y_all = tracking[:, 1]
likelihood_all = tracking[:, 2]
```

iii. The notes identify this side-camera stream as the reference tongue marker and use likelihood to preserve the required not-visible class.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Native samples with finite y/likelihood and likelihood at least 0.9 define visibility and session percentiles. For each bin center, the last video frame at or before the center is selected; low-confidence/nonfinite samples become class 3.

ii.
```python
visible_all = (np.isfinite(y_all) & np.isfinite(likelihood_all)
               & (likelihood_all >= TONGUE_LIKELIHOOD_CUTOFF))
q40, q60 = np.percentile(y_all[visible_all], [40, 60])
frame_idx = np.searchsorted(timestamps, centers_abs, side="right") - 1
```

iii. The agent cites the reference marker alignment's last-sample convention. It chose likelihood 0.9 because scores were strongly bimodal and rejected occlusion imputation because the task explicitly requires not-visible.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Per-session 40th and 60th percentiles are calculated from all visible native-frame y values. Visible values below q40 are 0, q40 through q60 inclusive are 1, above q60 are 2, and invisible values are 3.

ii.
```python
category = np.full(centers_abs.shape, 3, dtype=np.uint8)
category[visible & (y < q40)] = 0
category[visible & (y >= q40) & (y <= q60)] = 1
category[visible & (y > q60)] = 2
```

iii. The per-session percentile scope and four category meanings follow the task. The notes justify excluding invisible frames so occlusion noise does not determine thresholds.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. For every absolute neural-bin center, `searchsorted(..., side="right") - 1` selects the immediately preceding camera frame; its y/visibility becomes that bin's output.

ii.
```python
frame_idx = np.searchsorted(timestamps, centers_abs, side="right") - 1
safe_idx = np.clip(frame_idx, 0, len(timestamps) - 1)
y = y_all[safe_idx]
likelihood = likelihood_all[safe_idx]
```

iii. The streams share an absolute session clock, and the agent states this reproduces the reference's last-frame alignment convention.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. A session with no curated units is skipped; inconsistent unit validity-vector lengths, too few events, missing tones, empty annotations, or fewer than two visible tongue samples raise errors. Behavior rows beyond inferred ephys length and population-all-zero windows are removed. Invalid/low-confidence tongue samples become category 3 rather than being imputed.

ii.
```python
if np.any(recorded_lengths != recorded_lengths[0]):
    raise ValueError(...)
...
neural_valid = np.any(rates != 0, axis=(1, 2))
...
category = np.full(centers_abs.shape, 3, dtype=np.uint8)
```

iii. The agent distinguishes absent neural recordings, which it excludes, from legitimate absent visibility, which it represents explicitly. Strict assertions are intended to prevent silent misalignment.

## 10-a. What are the most time-consuming steps of the code?

i. The agent identifies materializing the neuron × trial × 80 firing-rate arrays, per-unit spike searches, NWB I/O, and final serialization as the main costs. Full conversion took 178.7 seconds and produced an 11.837-GB pickle.

ii.
```python
for out_idx, unit_idx in enumerate(good_indices):
    spikes = np.asarray(spike_column[int(unit_idx)], dtype=np.float64)
    edge_positions = np.searchsorted(spikes, absolute_edges, side="left")
```

iii. The notes say costs scale with the necessary neural payload and already ran well below the time budget.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The unit loop remains because spike trains are ragged, although every trial edge is searched at once. The choice diagnostic loops over trials and could be vectorized; the annotation-to-region loop could also be replaced by a bulk unique/inverse mapping.

ii.
```python
for out_idx, unit_idx in enumerate(good_indices): ...
for i, go in enumerate(go_times): ...
for i, label in enumerate(annotations): ...
```

iii. The agent explicitly describes the per-unit loop as the remaining unavoidable ragged-data loop and says behavioral alignment is otherwise vectorized. It did not discuss the two small bookkeeping/diagnostic loops as material bottlenecks.

## 10-c. What processing does the code repeat multiple times?

i. It does not reopen sessions or recompute output cubes, but each good unit's ragged spike vector is read separately, and the diagnostic re-searches left/right event arrays once per trial. Sample/full runs and independent validation repeat conversion-derived checks outside the core full run.

ii.
```python
spikes = np.asarray(spike_column[int(unit_idx)], dtype=np.float64)
...
for i, go in enumerate(go_times):
    li = np.searchsorted(left, go, side="left")
    ri = np.searchsorted(right, go, side="left")
```

iii. The notes emphasize that each NWB is opened once during a conversion and that cubes are computed once. Repeated checks were retained for validation rather than production necessity.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. `event_choice_check` scans lick events solely to generate metadata diagnostics; per-session timing, tongue diagnostics, and verbose `session_info` are not decoder features. Optional plotting also stacks arrays and renders figures but only when requested.

ii.
```python
choice_check = event_choice_check(nwb, trials_df, go_times, choice)
...
"tongue_diagnostics": tongue_diag,
"choice_event_check": choice_check,
...
if show_processing and len(neural) <= 2:
    plot_processing(session, plot_path)
```

iii. The agent justifies these as sanity checks: choice-event agreement, class occupancy, timing, and representative processing plots helped validate the conversion. They are not needed by decoder training, but the diagnostics are retained in metadata rather than silently discarded.
