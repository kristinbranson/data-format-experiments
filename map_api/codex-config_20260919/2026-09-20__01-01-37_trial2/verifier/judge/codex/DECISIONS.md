# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads all data by globbing `sub-*/*.nwb` under `/app/data`, sorting the paths, and opening each NWB file once with `pynwb.NWBHDF5IO`. Within each file it reads `nwb.units`, `nwb.trials`, `BehavioralEvents`, and `BehavioralTimeSeries` as needed during `process_session`.

ii.
```python
def nwb_files() -> list[Path]:
    """Return source sessions in deterministic subject/session order."""
    return sorted(DATA_DIR.glob("sub-*/*.nwb"))
```

```python
with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
    nwb = io.read()
    classifications = as_string_array(nwb.units["classification"])
    ...
    trials_df_all = nwb.trials[:]
    events = nwb.acquisition["BehavioralEvents"].time_series
```

iii. In `CONVERSION_NOTES.md` the agent says the archive is organized as one NWB per session and that all inspection/loading should use PyNWB only. It also says the sorted file list gives deterministic subject/session order.

## 1-b. How are the data split into subjects (mice)?

i. Each session is assigned to a subject using `nwb.subject.subject_id`. During assembly, the script builds `subjects` and `subject_idx` incrementally in first-seen order across the sorted session files.

ii.
```python
subject = session["subject"]
if subject not in subject_to_idx:
    subject_to_idx[subject] = len(subjects)
    subjects.append(subject)
...
"subjects": subjects,
"subject_idx": np.asarray(subject_idx, dtype=np.int32),
```

```python
"subject": str(nwb.subject.subject_id),
```

iii. In the mapping notes the agent identifies `subject.subject_id` as the source variable for `subjects` and `subject_idx`, and treats one NWB file as one session belonging to that animal.

## 1-c. How are the data split into sessions?

i. The AI treats each NWB file as one session. Session order follows the sorted NWB path list, and per-session metadata stores `nwb.identifier`.

ii.
```python
for path in files:
    ...
    session, diagnostics = process_session(path, region_to_idx)
```

```python
"info": {
    "session_id": str(nwb.identifier),
    "source_file": str(path),
    ...
},
```

iii. In `CONVERSION_NOTES.md` the agent states that `/app/data` contains one NWB per recording session and that this file boundary should be used as the session boundary.

## 1-d. How are the data split into trials?

i. The AI does not use all rows of `nwb.trials` directly. Instead, it defines the usable trial set as the first `n_recorded_trials` rows of the trials table, where `n_recorded_trials` is taken from the common length of `units["is_good_trials"]` across curated units. The go-cue array is truncated to the same length.

ii.
```python
trials_df_all = nwb.trials[:]
recorded_lengths = np.asarray(
    [len(nwb.units["is_good_trials"][int(i)]) for i in good], dtype=int
)
...
n_recorded_trials = int(recorded_lengths[0])
...
trials_df = trials_df_all.iloc[:n_recorded_trials].copy()
...
go_times_all = np.asarray(events["go_start_times"].timestamps[:], dtype=np.float64)
...
go_times = go_times_all[:n_recorded_trials]
```

iii. The justification in Step 7 of `CONVERSION_NOTES.md` is that sample validation exposed trailing behavioral-table rows with no spikes in eight NWBs, so the agent decided to use the per-unit `is_good_trials` length as the source ephys trial count.

## 1-e. How are trials filtered based on quality controls?

i. The AI does not apply the reference trial filters based on `obs_intervals` and `free_water`. Instead, after truncating to `n_recorded_trials`, it bins spikes for all remaining trials and then drops any trial whose entire population firing-rate tensor is all zeros. It also drops sessions with zero curated units and sessions with fewer than two surviving trials.

ii.
```python
good = np.flatnonzero(classifications == "good")
if len(good) == 0:
    return None, {"file": path.name, "skip_reason": "zero curated units"}
```

```python
rates = bin_spikes(nwb.units, good, go_times)
neural_valid = np.any(rates != 0, axis=(1, 2))
excluded_all_zero = int((~neural_valid).sum())
if excluded_all_zero:
    rates = rates[neural_valid]
    trials_df = trials_df.iloc[np.flatnonzero(neural_valid)].copy()
    go_times = go_times[neural_valid]
    tone_onsets = tone_onsets[neural_valid]
    centers_abs = centers_abs[neural_valid]
...
if n_trials < 2:
    raise ValueError(f"{path.name}: fewer than two trials with neural data")
```

iii. The notes say the agent deliberately retained requested classes such as early-lick, ignore, and stimulation trials, then used `is_good_trials` length plus removal of “population-all-zero windows” to exclude periods with no neural data.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `nwb.units["spike_times"]` for units whose `classification` is `"good"`, using `BehavioralEvents/go_start_times` to place the trial-aligned bin edges.

ii.
```python
classifications = as_string_array(nwb.units["classification"])
good = np.flatnonzero(classifications == "good")
...
go_times_all = np.asarray(events["go_start_times"].timestamps[:], dtype=np.float64)
...
spike_column = units["spike_times"]
for out_idx, unit_idx in enumerate(good_indices):
    spikes = np.asarray(spike_column[int(unit_idx)], dtype=np.float64)
```

iii. In the notes the agent identifies the released classifier-QC label as the unit filter and the absolute spike times plus go-cue timestamps as the raw sources needed for go-aligned firing rates.

## 2-b. How is the `neural` data processed?

i. The AI converts spike times into firing rates by building absolute bin edges around each trial’s go cue, counting spikes in half-open 50 ms bins with `np.searchsorted`, differencing adjacent cumulative counts, and dividing by `0.05` to obtain Hz.

ii.
```python
absolute_edges = go_times[:, None] + EDGES_REL[None, :]
rates = np.empty((len(go_times), len(good_indices), N_TIME), dtype=np.float32)
...
edge_positions = np.searchsorted(spikes, absolute_edges, side="left")
rates[:, out_idx, :] = np.diff(edge_positions, axis=1) / BIN_SIZE_S
```

iii. Step 1 and Step 5 of the notes say the task-required 50 ms nonoverlapping bins should preserve the reference half-open spike-counting convention while overriding the paper’s different bin width/stride.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI keeps only units with `units.classification == "good"`. If a session has zero such units it is skipped. It also raises an error if any curated unit lacks an anatomical `anno_name`.

ii.
```python
classifications = as_string_array(nwb.units["classification"])
good = np.flatnonzero(classifications == "good")
if len(good) == 0:
    return None, {"file": path.name, "skip_reason": "zero curated units"}
```

```python
annotations = as_string_array(nwb.units["anno_name"])[good]
if np.any(annotations == ""):
    raise ValueError(f"{path.name}: curated unit without CCF annotation")
```

iii. The notes explicitly say classifier QC should be reproduced from NWB quality fields and identify `classification == "good"` as the retained neuron set.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural activity is aligned to go-cue onset by adding a fixed relative edge grid spanning `-2.5` to `+1.5` seconds to each trial’s absolute go-cue time.

ii.
```python
OFF_START = -2.5
OFF_END = 1.5
EDGES_REL = np.arange(OFF_START, OFF_END + BIN_SIZE_S / 2, BIN_SIZE_S)
```

```python
absolute_edges = go_times[:, None] + EDGES_REL[None, :]
```

iii. The agent’s notes repeatedly state that all NWB timestamps are on one absolute session clock, so go-cue alignment can be done by placing the shared relative bin grid on each trial’s go cue.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data uses 50 ms bins, yielding 80 time bins from `-2.5` to `+1.5` seconds relative to go cue. No further rebinning is applied.

ii.
```python
BIN_SIZE_S = 0.050
OFF_START = -2.5
OFF_END = 1.5
EDGES_REL = np.arange(OFF_START, OFF_END + BIN_SIZE_S / 2, BIN_SIZE_S)
CENTERS_REL = EDGES_REL[:-1] + BIN_SIZE_S / 2
N_TIME = len(CENTERS_REL)
```

iii. The notes say the decoder instructions override the paper’s original 40 ms / 3.4 ms processing, so the agent uses exactly the mandated nonoverlapping 50 ms bins.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. The AI derives this input from `BehavioralEvents/sample_start_times`, the trial start times from `nwb.trials`, and the go-cue times. It selects the first sample/tone event between each trial start and that trial’s go cue.

ii.
```python
def first_event_per_trial(
    events: np.ndarray, starts: np.ndarray, upper_bounds: np.ndarray, name: str
) -> np.ndarray:
    """Select the first event in each [trial start, upper bound] interval."""
    indices = np.searchsorted(events, starts, side="left")
    ...
    return selected
```

```python
trial_starts = np.asarray(trials_df.start_time, dtype=np.float64)
sample_events = np.asarray(events["sample_start_times"].timestamps[:], dtype=np.float64)
tone_onsets = first_event_per_trial(sample_events, trial_starts, go_times, "tone")
```

iii. In the mapping table in `CONVERSION_NOTES.md`, the agent explicitly writes “Select first tone/sample onset within each native trial,” and the metadata stores `tone_onset_definition` as `"first sample_start event within the native trial"`.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. After choosing one tone onset per trial, the AI computes the input value at each time bin as absolute bin-center time minus that trial’s tone onset, producing a continuous time-varying signal.

ii.
```python
centers_abs = go_times[:, None] + CENTERS_REL[None, :]
...
tone_time = (centers_abs - tone_onsets[:, None]).astype(np.float32)
...
input_cube = np.stack([tone_time, photostim], axis=1)
```

iii. The notes justify a continuous representation because the task explicitly asks for “Time from tone onset in seconds” as a decoder input rather than a binary onset indicator.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is evaluated on exactly the same 80 bin centers used for the neural data. The code first computes the absolute bin centers from the go cue, then subtracts tone onset from those same centers.

ii.
```python
CENTERS_REL = EDGES_REL[:-1] + BIN_SIZE_S / 2
...
centers_abs = go_times[:, None] + CENTERS_REL[None, :]
tone_time = (centers_abs - tone_onsets[:, None]).astype(np.float32)
```

iii. The notes say all streams share the NWB session clock, so the tone-relative input should be put directly onto the same go-aligned neural bin centers.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. Photostimulation is derived from the trial-table columns `photostim_onset`, `photostim_duration`, and `start_time`.

ii.
```python
onset_raw = np.asarray(trials_df.photostim_onset).astype(str)
duration_raw = np.asarray(trials_df.photostim_duration).astype(str)
trial_start = np.asarray(trials_df.start_time, dtype=np.float64)
```

iii. The notes identify trial photostimulation onset/duration as the source variables for the photostimulation input and say stimulation should be preserved rather than filtered out.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. The AI converts non-`"N/A"` onset and duration strings to floats, adds onset to trial start to get absolute stimulation onset, and returns a binary time series that is `1` when a neural-bin center falls in `[stim_on, stim_off)`.

ii.
```python
onset = np.full(len(trials_df), np.nan)
duration = np.full(len(trials_df), np.nan)
active = onset_raw != "N/A"
onset[active] = np.asarray(onset_raw[active], dtype=float) + trial_start[active]
duration[active] = np.asarray(duration_raw[active], dtype=float)
return (
    active[:, None]
    & (centers_abs >= onset[:, None])
    & (centers_abs < onset[:, None] + duration[:, None])
).astype(np.float32)
```

iii. In the notes the agent says photostimulation should be represented as a time-varying binary input aligned to the same decoder bins.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The AI aligns photostimulation by evaluating it at the same absolute bin centers used for the neural data. It does not convert to a separate relative axis; it compares absolute stimulation times to `centers_abs`.

ii.
```python
centers_abs = go_times[:, None] + CENTERS_REL[None, :]
...
return (
    active[:, None]
    & (centers_abs >= onset[:, None])
    & (centers_abs < onset[:, None] + duration[:, None])
).astype(np.float32)
```

iii. The justification in the notes is that all timestamps share the same NWB clock, so direct comparison at common absolute bin centers is sufficient.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. The AI derives choice from the per-trial fields `trial_instruction` and `outcome`. It also computes a diagnostic check against left/right lick event times, but that diagnostic is not used to define the output.

ii.
```python
instruction = np.asarray(trials_df.trial_instruction).astype(str)
outcome_text = np.asarray(trials_df.outcome).astype(str)
choice = derive_choice(instruction, outcome_text)
```

```python
def derive_choice(instruction: np.ndarray, outcome: np.ndarray) -> np.ndarray:
    """Map task result to actual lick choice: left=0, right=1, no lick=2."""
    instructed = np.where(instruction == "left", 0, 1)
    choice = instructed.copy()
    choice[outcome == "miss"] = 1 - choice[outcome == "miss"]
    choice[outcome == "ignore"] = 2
    return choice.astype(np.uint8)
```

iii. The notes say choice is not stored directly in the trials table and should be inferred deterministically from instructed side plus outcome; the later `event_choice_check` is only a validation step.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The AI maps choice to categorical codes `0=left`, `1=right`, `2=no lick`, then repeats the per-trial value across all 80 time bins.

ii.
```python
output_cube = np.empty((n_trials, 4, N_TIME), dtype=np.uint8)
output_cube[:, 0, :] = choice[:, None]
```

```python
"output_values": [
    ["left", "right", "no lick"],
    ...
],
```

iii. The notes justify repeating per-trial outputs across time so all output variables share one rectangular `(n_output, n_timepoints)` representation.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is taken directly from `trials_df.outcome`.

ii.
```python
outcome_text = np.asarray(trials_df.outcome).astype(str)
```

iii. The notes say the NWB trials table already stores the exact requested outcome categories.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The AI maps outcome strings to `0=ignore`, `1=miss`, `2=hit`, then repeats the per-trial code across time bins.

ii.
```python
outcome_map = {"ignore": 0, "miss": 1, "hit": 2}
outcome = np.asarray([outcome_map[x] for x in outcome_text], dtype=np.uint8)
...
output_cube[:, 1, :] = outcome[:, None]
```

iii. The notes say the requested categorical outputs should all share the same time axis, so per-trial variables are broadcast across the 80 bins.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Early lick is taken directly from `trials_df.early_lick`.

ii.
```python
early_text = np.asarray(trials_df.early_lick).astype(str)
```

iii. The notes identify the native early-lick flag as the source and say these trials must be retained because early lick is a required decoder output.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The AI encodes `early` as `1` and anything else as `0`, then repeats the value across all 80 bins.

ii.
```python
early = (early_text == "early").astype(np.uint8)
...
output_cube[:, 2, :] = early[:, None]
```

iii. The notes justify keeping early-lick trials and storing the result as a per-trial categorical output broadcast over time.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. Tongue y-position is derived from `BehavioralTimeSeries/Camera0_side_TongueTracking`. The script uses `timestamps`, column 1 of `data` as `y`, and column 2 as tracking likelihood.

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

iii. The notes identify side-camera tongue tracking as the source series for the tongue output and describe using likelihood to determine visibility.

## 8-b. How is `output` *Tongue y-position* processed?

i. The AI keeps frames with finite values and `likelihood >= 0.9`, computes session-wide 40th and 60th percentiles from the raw visible `y` values, then for every neural-bin center takes the last video frame at or before that center and discretizes its `y` value. Invisible bins become class `3`.

ii.
```python
TONGUE_LIKELIHOOD_CUTOFF = 0.9
...
visible_all = (
    np.isfinite(y_all)
    & np.isfinite(likelihood_all)
    & (likelihood_all >= TONGUE_LIKELIHOOD_CUTOFF)
)
...
q40, q60 = np.percentile(y_all[visible_all], [40, 60])
```

```python
frame_idx = np.searchsorted(timestamps, centers_abs, side="right") - 1
...
category = np.full(centers_abs.shape, 3, dtype=np.uint8)
category[visible & (y < q40)] = 0
category[visible & (y >= q40) & (y <= q60)] = 1
category[visible & (y > q60)] = 2
```

iii. In the notes the agent argues for “reference last-sample alignment for visible y” and for preserving occlusion as class 3 rather than imputing it away. It also says the likelihood distribution is bimodal and therefore uses a high visibility cutoff.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. The thresholds are the session-level 40th and 60th percentiles of raw visible `y` samples, not of 50 ms bin means. Category `0` is below `q40`, category `1` is between `q40` and `q60` inclusive, category `2` is above `q60`, and category `3` is “not visible.”

ii.
```python
q40, q60 = np.percentile(y_all[visible_all], [40, 60])
...
category = np.full(centers_abs.shape, 3, dtype=np.uint8)
category[visible & (y < q40)] = 0
category[visible & (y >= q40) & (y <= q60)] = 1
category[visible & (y > q60)] = 2
```

iii. The mapping notes explicitly say the session thresholds should come from “all native visible y values,” and the metadata stores only those two visible-y percentile cutoffs.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Alignment is performed at the neural-bin centers, not over the full 50 ms neural bins. For each bin center, the AI picks the last camera frame at or before that time.

ii.
```python
centers_abs = go_times[:, None] + CENTERS_REL[None, :]
...
frame_idx = np.searchsorted(timestamps, centers_abs, side="right") - 1
valid_idx = (frame_idx >= 0) & (frame_idx < len(timestamps))
safe_idx = np.clip(frame_idx, 0, len(timestamps) - 1)
y = y_all[safe_idx]
likelihood = likelihood_all[safe_idx]
```

iii. The notes say this follows the “last sample in each interval” convention from the reference video-alignment code, and the metadata records tongue alignment as `"last video frame at or before each neural-bin center"`.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles some missing-data cases by exclusion and others by hard failure. Sessions with zero curated units are skipped. Trials with all-zero neural data are removed. Invisible tongue samples become class `3`. But missing tone events, inconsistent `is_good_trials` lengths, missing curated-unit annotations, and sessions with fewer than two visible tongue samples raise `ValueError` instead of being repaired or tolerated.

ii.
```python
if np.any(indices >= len(events)):
    raise ValueError(f"Missing {name} event after a trial start")
...
if np.any(recorded_lengths != recorded_lengths[0]):
    raise ValueError(f"{path.name}: inconsistent recorded-trial counts across units")
...
if np.any(annotations == ""):
    raise ValueError(f"{path.name}: curated unit without CCF annotation")
...
if visible_all.sum() < 2:
    raise ValueError("Fewer than two visible tongue samples in session")
```

```python
if len(good) == 0:
    return None, {"file": path.name, "skip_reason": "zero curated units"}
...
category = np.full(centers_abs.shape, 3, dtype=np.uint8)
```

iii. The notes explain the session skip for zero curated units and the explicit “not visible” tongue class, but the code itself is stricter than the notes in several places because it converts several data issues into exceptions.

## 10-a. What are the most time-consuming steps of the code?

i. The AI identifies full-session firing-rate construction as the dominant cost: opening each NWB once, reading spike arrays and tongue arrays, and looping over curated units to run `searchsorted`. The notes also mention serialization of the large pickle.

ii.
```python
rates = np.empty((len(go_times), len(good_indices), N_TIME), dtype=np.float32)
spike_column = units["spike_times"]
for out_idx, unit_idx in enumerate(good_indices):
    spikes = np.asarray(spike_column[int(unit_idx)], dtype=np.float64)
    edge_positions = np.searchsorted(spikes, absolute_edges, side="left")
    rates[:, out_idx, :] = np.diff(edge_positions, axis=1) / BIN_SIZE_S
```

iii. Step 6 and Step 9 of `CONVERSION_NOTES.md` say the dominant unavoidable cost is materializing neuron-by-trial-by-time firing rates and that the full conversion finished in about 179 seconds.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The main remaining vectorizable loops are the per-unit loop in `bin_spikes`, the per-trial loop in `event_choice_check`, and the per-unit loop that maps anatomical labels to indices. Optional plotting also loops over representative trials.

ii.
```python
for out_idx, unit_idx in enumerate(good_indices):
    spikes = np.asarray(spike_column[int(unit_idx)], dtype=np.float64)
    edge_positions = np.searchsorted(spikes, absolute_edges, side="left")
    rates[:, out_idx, :] = np.diff(edge_positions, axis=1) / BIN_SIZE_S
```

```python
for i, go in enumerate(go_times):
    ...
```

```python
for i, label in enumerate(annotations):
    if label not in region_to_idx:
        region_to_idx[label] = len(region_to_idx)
    region_idx[i] = region_to_idx[label]
```

iii. The notes say per-unit spike counting is the dominant unavoidable loop and that vectorization was already added where possible. The remaining small loops were apparently left for clarity or diagnostics.

## 10-c. What processing does the code repeat multiple times?

i. The AI repeats some nonessential processing. After deriving choice from `trial_instruction` and `outcome`, it rescans left/right lick-event streams in `event_choice_check` to compare against the derived choice. Optional `plot_processing` also restacks and traverses arrays that were already built for conversion.

ii.
```python
choice = derive_choice(instruction, outcome_text)
...
choice_check = event_choice_check(nwb, trials_df, go_times, choice)
```

```python
def plot_processing(session: dict, output_path: Path) -> None:
    output_array = np.stack(session["output"])
    input_array = np.stack(session["input"])
    ...
```

iii. The justification comes from the notes’ emphasis on sanity checks: the agent deliberately added diagnostic recomputation to validate derived outputs and intermediate processing.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI performs extra diagnostic work that is not needed for downstream decoder training: `event_choice_check`, tongue visibility/class-count diagnostics, processing-time bookkeeping, optional plotting imports and plotting code, and large per-session metadata summaries. These do not affect the actual `neural`, `input`, or `output` arrays used by the decoder.

ii.
```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
```

```python
choice_check = event_choice_check(nwb, trials_df, go_times, choice)
...
"tongue_diagnostics": tongue_diag,
"choice_event_check": choice_check,
...
session["info"]["processing_seconds"] = time.perf_counter() - started
```

```python
if show_processing and len(neural) <= 2:
    plot_path = Path(f"/app/processing_{session['info']['session_id']}.png")
    plot_processing(session, plot_path)
```

iii. The notes explicitly frame these as sanity checks and validation aids rather than core conversion steps, so they are intentional extra work beyond the minimum needed for the decoder dataset.
