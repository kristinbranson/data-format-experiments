# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads all session files with a sorted glob over `/app/data/sub-*/*.nwb`. Each NWB file is opened once with `pynwb.NWBHDF5IO`, then the session-level objects are read from that file: `nwb.units`, `nwb.trials`, `BehavioralEvents`, and `BehavioralTimeSeries`.

ii.
```python
files = sorted(DATA_DIR.glob("sub-*/*.nwb"))
```

```python
with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
    nwb = io.read()
    units = nwb.units
    trials_all = nwb.trials.to_dataframe()
```

iii. In `CONVERSION_NOTES.md` Step 2 and Step 5, the agent justifies this as the native published layout: one NWB per session under subject folders, with all inspection done through PyNWB as required.

## 1-b. How are the data split into subjects (mice)?

i. Each session’s subject is taken from `nwb.subject.subject_id`. After converting sessions, the agent builds `subjects` as the sorted unique set of these IDs and `subject_idx` as the per-session index into that list.

ii.
```python
"subject": str(nwb.subject.subject_id),
```

```python
subjects = sorted({s["subject"] for s in converted})
subject_lookup = {name: i for i, name in enumerate(subjects)}
...
"subject_idx": np.asarray([subject_lookup[s["subject"]] for s in converted], dtype=np.int16),
```

iii. Step 5 of `CONVERSION_NOTES.md` says `subject.subject_id` is the direct NWB-to-target mapping, and Step 2 records that the dataset contains 28 subjects.

## 1-c. How are the data split into sessions?

i. The agent treats each NWB file as one session. Session order follows the sorted file list, and each retained session carries `nwb.identifier` as `session_id`.

ii.
```python
files = sorted(DATA_DIR.glob("sub-*/*.nwb"))
```

```python
"session_id": nwb.identifier,
```

iii. In Step 4 and Step 5 of `CONVERSION_NOTES.md`, the agent states that each retained session is one NWB file and that the published archive already uses that boundary.

## 1-d. How are the data split into trials?

i. Trials come from `nwb.trials.to_dataframe()`, but the agent does not rely only on event-vector position. It uses each trial’s `start_time` and `stop_time` to assign `go_start_times`, `sample_start_times`, and photostim events to individual trials with `_events_by_trial`. It then requires exactly one go event and at least one sample event per trial.

ii.
```python
trials_all = nwb.trials.to_dataframe()
starts = trials["start_time"].to_numpy(np.float64)
stops = trials["stop_time"].to_numpy(np.float64)
go_by_trial = _events_by_trial(_events(nwb, "go_start_times"), starts, stops)
sample_by_trial = _events_by_trial(_events(nwb, "sample_start_times"), starts, stops)
if not all(len(x) == 1 for x in go_by_trial):
    ...
if not all(len(x) >= 1 for x in sample_by_trial):
    ...
```

iii. Step 2 of `CONVERSION_NOTES.md` says event-list counts can exceed trial counts because of repeated state-machine events, so trial association should be done from the trials table intervals rather than by simple positional matching, except after confirming one-go-per-trial.

## 1-e. How are trials filtered based on quality controls?

i. The agent does not use `obs_intervals` or `free_water` directly. It keeps all trials in sessions with classifier-good units, bins the spikes, then drops trials whose go-aligned extraction window has zero spikes across the entire retained neural population. A session is dropped if fewer than two such trials remain.

ii.
```python
rates = _bin_spikes(units, good_indices, go_times)
...
neural_present = np.any(rates != 0, axis=(1, 2))
n_excluded_no_neural = int((~neural_present).sum())
if n_excluded_no_neural:
    trials = trials.loc[neural_present].copy()
    go_times = go_times[neural_present]
    tone_times = tone_times[neural_present]
    inputs = inputs[neural_present]
    outputs = outputs[neural_present]
    rates = rates[neural_present]
if len(trials) < 2:
    ...
```

iii. The code comment and Step 10 of `CONVERSION_NOTES.md` justify this as a response to sessions where behavior continued after recording stopped. The agent says intersecting all units’ observation intervals was too strict, so it resolved the issue by excluding only windows with no spikes from any classifier-good unit.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from `units["spike_times"]` for units whose `classification` is `"good"`, with `go_start_times` used to place the per-trial bin edges.

ii.
```python
classifications = np.asarray(units["classification"][:]).astype(str)
good_indices = np.flatnonzero(classifications == "good")
...
go_times = np.asarray([x[0] for x in go_by_trial])
...
spikes = _unit_spike_times(units, int(unit_idx))
```

iii. Step 5 of `CONVERSION_NOTES.md` maps `units.spike_times` to `neural` and says the decoder window should be defined relative to the trial’s go cue.

## 2-b. How is the `neural` data processed?

i. Spike times are converted to 50 ms firing rates in Hz. For each good unit, the agent builds absolute trial-by-bin edges from the go times, flattens them, uses `np.searchsorted` to get cumulative spike counts, differences adjacent counts to get per-bin spike counts, and divides by `BIN_WIDTH`. No smoothing, baseline subtraction, or normalization is applied.

ii.
```python
absolute_edges = go_times[:, None] + BIN_EDGES[None, :]
flattened = absolute_edges.ravel()
...
positions = np.searchsorted(spikes, flattened, side="left")
counts = np.diff(positions.reshape(len(go_times), N_TIME + 1), axis=1)
rates[:, out_idx, :] = counts.astype(np.float32) / BIN_WIDTH
```

iii. Step 1 and Step 5 of `CONVERSION_NOTES.md` say the agent followed the reference half-open histogram logic and the decoder-specific 50 ms binning override.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `units["classification"] == "good"` are kept. If a session has no such units it is skipped. The agent also requires every kept unit to have a nonempty `anno_name`.

ii.
```python
classifications = np.asarray(units["classification"][:]).astype(str)
good_indices = np.flatnonzero(classifications == "good")
if len(good_indices) == 0:
    print(f"SKIP {path.name}: no classifier-good units", flush=True)
    return None
annotations_all = np.asarray(units["anno_name"][:]).astype(str)
annotations = annotations_all[good_indices]
if np.any(annotations == ""):
    raise ValueError(f"{path.name}: classifier-good unit lacks CCF annotation")
```

iii. Step 4 and Step 5 of `CONVERSION_NOTES.md` say the agent chose the archive’s explicit classifier QC verdict and deliberately did not invent additional metric thresholds or use the method paper’s analysis-specific low-rate neuron filter.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The neural window is aligned to go-cue onset. For each trial, the agent gets the absolute go time and adds the fixed relative bin edges `[-2.5, 1.5]` s to obtain absolute bin edges for spike counting.

ii.
```python
go_times = np.asarray([x[0] for x in go_by_trial])
...
absolute_edges = go_times[:, None] + BIN_EDGES[None, :]
```

iii. Step 5 and Step 10 of `CONVERSION_NOTES.md` say all NWB streams are on the same absolute session clock, so alignment is done by subtracting or adding the trial’s go timestamp consistently across streams.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use non-overlapping 50 ms bins spanning -2.5 s to +1.5 s around go cue onset, for 80 time points per trial. Neural data are directly binned onto that grid; there is no second-stage temporal rebinning.

ii.
```python
BIN_WIDTH = 0.050
OFF_START = -2.5
OFF_END = 1.5
BIN_EDGES = np.arange(OFF_START, OFF_END + BIN_WIDTH / 2, BIN_WIDTH)
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2
N_TIME = len(BIN_CENTERS)
```

iii. Step 1 and Step 5 of `CONVERSION_NOTES.md` explicitly note that the reference papers used other analysis-specific windows, but the decoder task overrides them with 50 ms bins over `[-2.5, 1.5]`.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It is derived from `sample_start_times` and `go_start_times`, assigned to trials via the trials table intervals. For each trial the agent chooses the last sample/tone event at or before the go cue.

ii.
```python
sample_by_trial = _events_by_trial(_events(nwb, "sample_start_times"), starts, stops)
...
go_times = np.asarray([x[0] for x in go_by_trial])
tone_times = np.asarray([x[x <= g][-1] for x, g in zip(sample_by_trial, go_times)])
```

iii. Step 5 of `CONVERSION_NOTES.md` justifies taking the last tone because early-lick trials can replay the sample epoch, so the final onset is the relevant one before go.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The agent computes the absolute center time of each neural bin, subtracts the trial’s tone onset, and stores the result as a continuous float32 time series.

ii.
```python
absolute_centers = go_times[:, None] + BIN_CENTERS[None, :]
inputs = np.empty((len(trials), 2, N_TIME), dtype=np.float32)
inputs[:, 0, :] = (absolute_centers - tone_times[:, None]).astype(np.float32)
```

iii. In Step 5, the agent says it is following the decoder task literally: the input is continuous “time from tone onset in seconds,” not a binary event indicator.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is evaluated at the exact same bin centers used for the go-aligned neural bins: `absolute_centers = go_times + BIN_CENTERS`.

ii.
```python
absolute_centers = go_times[:, None] + BIN_CENTERS[None, :]
inputs[:, 0, :] = (absolute_centers - tone_times[:, None]).astype(np.float32)
```

iii. Step 5 of `CONVERSION_NOTES.md` lists the same bin-center grid for both neural activity and this input, so no extra alignment step is applied.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. The agent derives photostimulation from `BehavioralEvents/photostim_start_times` and `BehavioralEvents/photostim_stop_times`, assigned to trials using the trial start/stop intervals.

ii.
```python
stim_starts_by_trial = _events_by_trial(
    _events(nwb, "photostim_start_times"), starts, stops
)
stim_stops_by_trial = _events_by_trial(
    _events(nwb, "photostim_stop_times"), starts, stops
)
```

iii. Step 5 of `CONVERSION_NOTES.md` says the agent chose paired start/stop events inside each trial interval as the photostimulation source and used their timing as a sanity check against the paper’s delay-period stimulation description.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. The agent makes photostimulation a binary time-varying input: for each trial and each stimulation interval, bins whose centers fall in `[onset, offset)` are set to 1, otherwise 0.

ii.
```python
inputs[:, 1, :] = 0
for i, (onsets, offsets) in enumerate(zip(stim_starts_by_trial, stim_stops_by_trial)):
    for onset, offset in zip(onsets, offsets):
        active = ((absolute_centers[i] >= onset) &
                  (absolute_centers[i] < offset))
        inputs[i, 1, active] = 1.0
```

iii. The agent’s metadata says `photostim_definition` is “bin center in [photostim_start, photostim_stop),” and Step 5 frames this as the required binary per-time-point decoder input.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Alignment is done on the session’s absolute clock. The neural bins are represented by absolute bin centers, and photostimulation is turned on wherever those same absolute centers fall between the trial’s absolute photostim onset and stop times.

ii.
```python
absolute_centers = go_times[:, None] + BIN_CENTERS[None, :]
...
active = ((absolute_centers[i] >= onset) &
          (absolute_centers[i] < offset))
```

iii. Step 10 of `CONVERSION_NOTES.md` says the agent aligned all streams by the same absolute go-based time axis rather than by separate offsets per modality.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is derived from the per-trial fields `trial_instruction` and `outcome`. There is no stored choice column in the NWB file.

ii.
```python
outcomes_text = trials["outcome"].astype(str).to_numpy()
instructions = trials["trial_instruction"].astype(str).to_numpy()
choice = np.asarray([
    _classify_choice(inst, outcome)
    for inst, outcome in zip(instructions, outcomes_text)
], dtype=np.int8)
```

iii. Step 5 of `CONVERSION_NOTES.md` says the agent preferred this task-defined derivation because it gives a reliable no-lick class for `ignore` trials and agreed overwhelmingly with response-period lick events in its audit.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The agent maps choice to integers `0=left`, `1=right`, `2=no lick` using `_classify_choice`, then repeats the per-trial label across all 80 time bins in output row 0.

ii.
```python
def _classify_choice(instruction: str, outcome: str) -> int:
    if outcome == "ignore":
        return 2
    if outcome == "hit":
        return 0 if instruction == "left" else 1
    if outcome == "miss":
        return 1 if instruction == "left" else 0
```

```python
outputs = np.empty((len(trials), 4, N_TIME), dtype=np.int8)
outputs[:, 0, :] = choice[:, None]
```

iii. Step 5 says per-trial outputs were repeated across time so all outputs would share one `(4, 80)` representation for the validator and decoder.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome comes directly from the trial-table column `outcome`.

ii.
```python
outcomes_text = trials["outcome"].astype(str).to_numpy()
```

iii. The agent’s Step 5 mapping treats this as a direct category transfer because the NWB trials table already uses the requested `ignore`, `miss`, and `hit` labels.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The strings are mapped to `0=ignore`, `1=miss`, `2=hit` and then repeated across all 80 bins in output row 1.

ii.
```python
outcome_map = {"ignore": 0, "miss": 1, "hit": 2}
outcome = np.asarray([outcome_map[x] for x in outcomes_text], dtype=np.int8)
outputs[:, 1, :] = outcome[:, None]
```

iii. Step 5 says outcome is a per-trial categorical target, so it is stored as a trial-constant row inside the shared time-by-output array.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Early lick is taken directly from the trial-table column `early_lick`.

ii.
```python
early_text = trials["early_lick"].astype(str).to_numpy()
```

iii. The agent’s Step 5 mapping records this as a direct field transfer because the NWB trials table already encodes the requested categories.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The strings are mapped to `0=no`, `1=yes` via `{"no early": 0, "early": 1}` and repeated across all 80 bins in output row 2.

ii.
```python
early_map = {"no early": 0, "early": 1}
early = np.asarray([early_map[x] for x in early_text], dtype=np.int8)
outputs[:, 2, :] = early[:, None]
```

iii. As in Step 5, the agent justified repeating it across time because it is a trial-level label stored in the same output tensor as the time-varying tongue label.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. Tongue y-position is derived from `BehavioralTimeSeries/Camera0_side_TongueTracking`: `timestamps`, column 1 of `data` for `tongue_y`, and column 2 for the tracking likelihood.

ii.
```python
behavior = nwb.acquisition["BehavioralTimeSeries"]
tongue = behavior.time_series["Camera0_side_TongueTracking"]
tongue_times = np.asarray(tongue.timestamps[:], dtype=np.float64)
tongue_data = np.asarray(tongue.data[:], dtype=np.float64)
tongue_y = tongue_data[:, 1]
tongue_likelihood = tongue_data[:, 2]
```

iii. Step 2 and Step 5 of `CONVERSION_NOTES.md` identify Camera0 side tongue tracking as the consistent session-wide tongue stream and map its y coordinate to the requested output.

## 8-b. How is `output` *Tongue y-position* processed?

i. The agent marks a frame as visible when `tongue_likelihood >= 0.9` and both y and likelihood are finite. It computes `q40` and `q60` from all visible raw session frames, not from 50 ms session-bin means. For each neural bin center it picks the nearest camera frame, reads that frame’s y and likelihood, and assigns a discrete tongue class. If the nearest frame is not visible, the output remains class 3 (`not visible`).

ii.
```python
DLC_VISIBLE_THRESHOLD = 0.9
...
session_visible = np.isfinite(tongue_y) & np.isfinite(tongue_likelihood) & (
    tongue_likelihood >= DLC_VISIBLE_THRESHOLD
)
q40, q60 = np.quantile(tongue_y[session_visible], [0.4, 0.6])
nearest = _nearest_indices(tongue_times, absolute_centers.ravel()).reshape(
    len(trials), N_TIME
)
matched_y = tongue_y[nearest]
matched_likelihood = tongue_likelihood[nearest]
visible = np.isfinite(matched_y) & np.isfinite(matched_likelihood) & (
    matched_likelihood >= DLC_VISIBLE_THRESHOLD
)
```

iii. Step 5 of `CONVERSION_NOTES.md` explicitly justifies this choice as “nearest frame” discretization with a `0.9` confidence threshold, arguing that the confidence is sharply bimodal and that keeping invisibility as class 3 is preferable to imputation for the requested decoder target.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. The agent uses session-wide `q40` and `q60` percentiles of visible raw `tongue_y` values. Visible samples with `y < q40` are class 0, `q40 <= y <= q60` are class 1, `y > q60` are class 2, and non-visible bins are class 3.

ii.
```python
q40, q60 = np.quantile(tongue_y[session_visible], [0.4, 0.6])
...
tongue_class = np.full((len(trials), N_TIME), 3, dtype=np.int8)
tongue_class[visible & (matched_y < q40)] = 0
tongue_class[visible & (matched_y >= q40) & (matched_y <= q60)] = 1
tongue_class[visible & (matched_y > q60)] = 2
```

iii. Step 5 of `CONVERSION_NOTES.md` says the threshold population is “all finite session frames with likelihood >= 0.9” and describes the requested percentile discretization plus the explicit not-visible category.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Instead of averaging all video frames inside each 50 ms bin, the agent aligns tongue output by taking the nearest camera frame to each neural bin center. That nearest-frame index is computed on the absolute session clock and reshaped to trial x time.

ii.
```python
absolute_centers = go_times[:, None] + BIN_CENTERS[None, :]
nearest = _nearest_indices(tongue_times, absolute_centers.ravel()).reshape(
    len(trials), N_TIME
)
matched_y = tongue_y[nearest]
matched_likelihood = tongue_likelihood[nearest]
```

iii. Step 5 of `CONVERSION_NOTES.md` explicitly lists “nearest timestamp to each neural bin center” as the tongue-frame matching rule.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The agent handles anomalies by either dropping or hard-failing. Sessions with no classifier-good units are skipped. Trials without exactly one go event or without any sample event raise errors. Trials with unpaired photostim events raise errors. Sessions with too few visible tongue samples raise errors. Trials with no spikes anywhere in the go-aligned window are removed after neural binning. Missing or low-confidence tongue observations are represented as class 3 rather than imputed.

ii.
```python
good_indices = np.flatnonzero(classifications == "good")
if len(good_indices) == 0:
    ...
if not all(len(x) == 1 for x in go_by_trial):
    raise ValueError(...)
if not all(len(x) >= 1 for x in sample_by_trial):
    raise ValueError(...)
for i, (on, off) in enumerate(zip(stim_starts_by_trial, stim_stops_by_trial)):
    if len(on) != len(off):
        raise ValueError(...)
if session_visible.sum() < 2:
    raise ValueError(...)
...
tongue_class = np.full((len(trials), N_TIME), 3, dtype=np.int8)
...
neural_present = np.any(rates != 0, axis=(1, 2))
```

iii. Step 10 of `CONVERSION_NOTES.md` says the agent preferred excluding or erroring on structurally invalid data rather than fabricating values, while preserving tongue invisibility as an explicit decoder class.

## 10-a. What are the most time-consuming steps of the code?

i. The heaviest work is session I/O and spike binning: opening NWB files, reading per-unit spike trains and full tongue-tracking arrays, looping over all good units in `_bin_spikes`, and finally serializing the large pickle. Optional plotting also adds overhead when enabled.

ii.
```python
with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
    nwb = io.read()
```

```python
for out_idx, unit_idx in enumerate(good_indices):
    spikes = _unit_spike_times(units, int(unit_idx))
    ...
```

```python
tongue_data = np.asarray(tongue.data[:], dtype=np.float64)
...
pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. Step 6 and Step 9 of `CONVERSION_NOTES.md` explicitly call out NWB reading, spike binning, tongue array loading, and pickle writing as the main runtime costs of the full conversion.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The largest remaining explicit loop is over good units in `_bin_spikes`, and there is also a nested loop over trials and photostimulation intervals when building the binary laser input. Trial packaging into Python lists is also loop-based. The expensive trial dimension of spike binning has already been vectorized by flattening all trial/bin edges.

ii.
```python
for out_idx, unit_idx in enumerate(good_indices):
    spikes = _unit_spike_times(units, int(unit_idx))
    ...
```

```python
for i, (onsets, offsets) in enumerate(zip(stim_starts_by_trial, stim_stops_by_trial)):
    for onset, offset in zip(onsets, offsets):
        active = ((absolute_centers[i] >= onset) &
                  (absolute_centers[i] < offset))
        inputs[i, 1, active] = 1.0
```

iii. Step 6 says the agent deliberately vectorized the costly trial-by-bin work, using one `searchsorted` per unit over all trial/bin edges and one vectorized nearest-frame lookup for the tongue output, leaving only the less avoidable or smaller loops in Python.

## 10-c. What processing does the code repeat multiple times?

i. There is no second pass over the full dataset, but within each session the code repeats several similar passes over trial timing: it associates events to trials separately for go, sample, photostim start, and photostim stop; then it makes separate passes to build inputs, outputs, and the post-binning `neural_present` filter.

ii.
```python
go_by_trial = _events_by_trial(_events(nwb, "go_start_times"), starts, stops)
sample_by_trial = _events_by_trial(_events(nwb, "sample_start_times"), starts, stops)
stim_starts_by_trial = _events_by_trial(
    _events(nwb, "photostim_start_times"), starts, stops
)
stim_stops_by_trial = _events_by_trial(
    _events(nwb, "photostim_stop_times"), starts, stops
)
```

```python
inputs[:, 0, :] = ...
...
outputs[:, 3, :] = tongue_class
...
neural_present = np.any(rates != 0, axis=(1, 2))
```

iii. The agent did not give a strong explicit justification for this repetition beyond preferring a single clear per-session pipeline. Step 6 and Step 10 do say it opens each NWB only once and avoids a second whole-dataset pass.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The core converted arrays are all retained, but the script does extra diagnostic work that the decoder does not use: optional processing plots, printed summary counts, detailed per-session `info` bookkeeping, and validation checks before writing the pickle. There is also an unused `_any_observation_mask` helper.

ii.
```python
def _plot_processing(...):
    ...
```

```python
def validate_converted(data: dict) -> None:
    ...
```

```python
def print_summary(data: dict) -> None:
    ...
```

```python
def _any_observation_mask(units, good_indices: np.ndarray,
                          starts: np.ndarray, stops: np.ndarray) -> np.ndarray:
    ...
```

iii. Step 7, Step 10, and Step 11 of `CONVERSION_NOTES.md` justify these as sanity checks, sample diagnostics, and validation aids rather than as features required by the downstream decoder.
