# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI globs all NWB files matching `sub-*/*.nwb` under the data directory, sorts them, and processes each file sequentially with `pynwb.NWBHDF5IO`. Within each file, it reads `nwb.units`, `nwb.trials.to_dataframe()`, and `nwb.acquisition['BehavioralEvents']` / `nwb.acquisition['BehavioralTimeSeries']`.

ii.
```python
files = sorted(DATA_DIR.glob("sub-*/*.nwb"))
...
for path in files:
    session = convert_session(path, ...)
```

```python
with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
    nwb = io.read()
    units = nwb.units
    ...
    trials_all = nwb.trials.to_dataframe()
```

iii. The AI noted that NWB is the published format and `pynwb` is its standard reader. One file per session, so globbing gives the complete dataset. The AI documented 174 NWB files and 28 subjects matching `dandiset.yaml`.

## 1-b. How are the data split into subjects?

i. Each NWB file's `nwb.subject.subject_id` provides the animal identifier. The AI collects all unique subject IDs across sessions, sorts them, and builds `subject_idx` as an index into that sorted list.

ii.
```python
"subject": str(nwb.subject.subject_id),
...
subjects = sorted({s["subject"] for s in converted})
subject_lookup = {name: i for i, name in enumerate(subjects)}
...
"subject_idx": np.asarray([subject_lookup[s["subject"]] for s in converted], dtype=np.int16),
```

iii. The AI noted that `subject_id` is the canonical animal identifier in the NWB file and that this yields 28 subjects.

## 1-c. How are the data split into sessions?

i. One NWB file equals one session. Each session is identified by `nwb.identifier`. The AI processes all 174 files but drops sessions with no classifier-good units (1 session) or fewer than 2 trials with neural data, yielding 173 sessions.

ii.
```python
files = sorted(DATA_DIR.glob("sub-*/*.nwb"))
...
"session_id": nwb.identifier,
```

iii. The AI documented that the dandiset stores one session per file, so the file boundary is the session boundary.

## 1-d. How are the data split into trials?

i. Trials come from the NWB trials table (`nwb.trials.to_dataframe()`). The AI verifies that each trial has exactly one go cue by associating `go_start_times` events with trial intervals.

ii.
```python
trials_all = nwb.trials.to_dataframe()
...
go_by_trial = _events_by_trial(_events(nwb, "go_start_times"), starts, stops)
...
if not all(len(x) == 1 for x in go_by_trial):
    bad = [i for i, x in enumerate(go_by_trial) if len(x) != 1]
    raise ValueError(...)
go_times = np.asarray([x[0] for x in go_by_trial])
```

iii. The AI noted that trials are clearly defined by the trials table, and verified a one-to-one mapping between go cues and trials.

## 1-e. How are trials filtered based on quality controls?

i. The AI does NOT use `obs_intervals` or `free_water` filtering. Instead, it bins all spikes for all trials first, then excludes trials where no spike was recorded across the entire classifier-good population in the trial window (`neural_present = np.any(rates != 0, axis=(1, 2))`). Sessions with fewer than 2 remaining trials are dropped. Free-water and auto-water trials are retained.

ii.
```python
neural_present = np.any(rates != 0, axis=(1, 2))
n_excluded_no_neural = int((~neural_present).sum())
if n_excluded_no_neural:
    trials = trials.loc[neural_present].copy()
    go_times = go_times[neural_present]
    ...
    rates = rates[neural_present]
if len(trials) < 2:
    ...
    return None
```

iii. The AI's CONVERSION_NOTES.md documents that an initial observation-interval approach was too restrictive because per-unit accepted intervals differ. The final approach bins all classifier-good units then excludes only windows with zero spikes across the entire population. Free-water trials are intentionally retained because their task-table labels are defined.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `units['spike_times']` for classifier-good units (`units['classification'] == 'good'`), aligned to go cue times from `BehavioralEvents/go_start_times`.

ii.
```python
classifications = np.asarray(units["classification"][:]).astype(str)
good_indices = np.flatnonzero(classifications == "good")
...
spikes = _unit_spike_times(units, int(unit_idx))
# which reads: np.asarray(units["spike_times"][unit_index], dtype=np.float64)
```

iii. The AI documented that `spike_times` is the only neural representation in the file and that firing rates are computed from it directly.

## 2-b. How is the `neural` data processed?

i. Spike times are binned into 80 non-overlapping 50ms bins spanning [-2.5, +1.5) relative to the go cue. For each good unit, absolute bin edges are computed for all trials, flattened, and `np.searchsorted` gives running spike counts at each edge. Differencing gives counts per bin, divided by `BIN_WIDTH` (0.05) to get firing rates in Hz. No smoothing, normalization, or baseline subtraction is applied.

ii.
```python
absolute_edges = go_times[:, None] + BIN_EDGES[None, :]
flattened = absolute_edges.ravel()
...
positions = np.searchsorted(spikes, flattened, side="left")
counts = np.diff(positions.reshape(len(go_times), N_TIME + 1), axis=1)
rates[:, out_idx, :] = counts.astype(np.float32) / BIN_WIDTH
```

iii. The AI documented this matches the reference code's `sliding_histogram(..., rate=True)` logic.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `classification == 'good'` are kept. No metric thresholds are applied. A session with no good units is dropped entirely. This retains 69,453 of 272,227 units.

ii.
```python
classifications = np.asarray(units["classification"][:]).astype(str)
good_indices = np.flatnonzero(classifications == "good")
if len(good_indices) == 0:
    print(f"SKIP {path.name}: no classifier-good units", flush=True)
    return None
```

iii. The AI documented that `classification` is the verdict of the spike-sorting QC classifier. One session has all NaN classifications and is dropped, giving 173 sessions.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spike times and go cue times are on the same session-absolute clock. Bin edges relative to the go cue are added to each trial's go-cue time to give absolute time windows, and spikes are binned against those edges.

ii.
```python
absolute_edges = go_times[:, None] + BIN_EDGES[None, :]
flattened = absolute_edges.ravel()
...
positions = np.searchsorted(spikes, flattened, side="left")
```

iii. The AI noted all NWB timestamps share one global clock, so no resampling or interpolation is needed.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 50ms non-overlapping bins, 80 bins spanning [-2.5, +1.5) seconds relative to the go cue. The bin grid is defined once as 81 edges.

ii.
```python
BIN_WIDTH = 0.050
OFF_START = -2.5
OFF_END = 1.5
BIN_EDGES = np.arange(OFF_START, OFF_END + BIN_WIDTH / 2, BIN_WIDTH)
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2
N_TIME = len(BIN_CENTERS)
```

iii. The window and 50ms bin width are set by the instructions.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. From `sample_start_times` (tone onsets) in `BehavioralEvents`, together with the go cue of each trial. The tone for a trial is the last `sample_start_times` event before the go cue within that trial's interval.

ii.
```python
sample_by_trial = _events_by_trial(_events(nwb, "sample_start_times"), starts, stops)
...
tone_times = np.asarray([x[x <= g][-1] for x, g in zip(sample_by_trial, go_times)])
```

iii. The AI noted that an early lick replays the sample epoch, so a trial can carry more than one tone; the last one before the go cue is used.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. For each bin, the absolute bin center time minus the tone onset time gives the time from tone onset. This is computed as `(go + bin_center_offset) - tone_onset`, which equals `bin_center_offset + (go - tone)`.

ii.
```python
absolute_centers = go_times[:, None] + BIN_CENTERS[None, :]
inputs[:, 0, :] = (absolute_centers - tone_times[:, None]).astype(np.float32)
```

iii. No additional processing beyond the subtraction.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. Both the neural data and the time-from-tone input use the same bin centers defined relative to the go cue, so they are inherently aligned.

ii.
```python
absolute_centers = go_times[:, None] + BIN_CENTERS[None, :]
inputs[:, 0, :] = (absolute_centers - tone_times[:, None]).astype(np.float32)
```

iii. Both share the same go-cue-relative bin grid.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. From `photostim_start_times` and `photostim_stop_times` in `BehavioralEvents`, associated with trials by trial intervals.

ii.
```python
stim_starts_by_trial = _events_by_trial(
    _events(nwb, "photostim_start_times"), starts, stops
)
stim_stops_by_trial = _events_by_trial(
    _events(nwb, "photostim_stop_times"), starts, stops
)
```

iii. The AI uses the event timestamps from `BehavioralEvents` rather than the `photostim_onset`/`photostim_duration` columns in the trials table.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A bin is 1 where its center falls between any photostim start and stop event for that trial, 0 otherwise.

ii.
```python
inputs[:, 1, :] = 0
for i, (onsets, offsets) in enumerate(zip(stim_starts_by_trial, stim_stops_by_trial)):
    for onset, offset in zip(onsets, offsets):
        active = ((absolute_centers[i] >= onset) &
                  (absolute_centers[i] < offset))
        inputs[i, 1, active] = 1.0
```

iii. The AI documented that photostim is a binary time series with the light on/off status at each bin center.

## 4-c. How is `input` *Photostimulation* aligned with the neural data?

i. The photostim onset/offset times are absolute timestamps on the same session clock as the go cue and neural data. Bin centers are computed as absolute times (`go + offset`), and the comparison is done in absolute time.

ii.
```python
absolute_centers = go_times[:, None] + BIN_CENTERS[None, :]
...
active = ((absolute_centers[i] >= onset) & (absolute_centers[i] < offset))
```

iii. Same session-absolute clock for all streams.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is derived from `trial_instruction` (left/right) and `outcome` (hit/miss/ignore) in the trials table. There is no direct choice column.

ii.
```python
def _classify_choice(instruction: str, outcome: str) -> int:
    if outcome == "ignore":
        return 2
    if outcome == "hit":
        return 0 if instruction == "left" else 1
    if outcome == "miss":
        return 1 if instruction == "left" else 0
    raise ValueError(...)
...
choice = np.asarray([
    _classify_choice(inst, outcome)
    for inst, outcome in zip(instructions, outcomes_text)
], dtype=np.int8)
```

iii. The AI documented that choice is fully determined by the instructed side and the outcome.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Coded as 0=left, 1=right, 2=no lick. Hit means animal licked the instructed side, miss means opposite side, ignore means no lick. The per-trial value is repeated across all 80 time bins.

ii.
```python
outputs[:, 0, :] = choice[:, None]
```

iii. Same logic as the reference.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Directly from the `outcome` column of the trials table (strings `'ignore'`, `'miss'`, `'hit'`).

ii.
```python
outcome_map = {"ignore": 0, "miss": 1, "hit": 2}
outcomes_text = trials["outcome"].astype(str).to_numpy()
outcome = np.asarray([outcome_map[x] for x in outcomes_text], dtype=np.int8)
```

iii. Direct categorical mapping.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Mapped to 0=ignore, 1=miss, 2=hit. Per-trial value repeated across all 80 bins.

ii.
```python
outputs[:, 1, :] = outcome[:, None]
```

iii. Straightforward mapping.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. From the `early_lick` column of the trials table (strings `'no early'` and `'early'`).

ii.
```python
early_map = {"no early": 0, "early": 1}
early_text = trials["early_lick"].astype(str).to_numpy()
early = np.asarray([early_map[x] for x in early_text], dtype=np.int8)
```

iii. Direct from the trials table.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Mapped to 0=no, 1=yes. Per-trial value repeated across all 80 bins.

ii.
```python
outputs[:, 2, :] = early[:, None]
```

iii. Straightforward mapping.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, which has `(n_frames, 3)` data corresponding to `tongue_x`, `tongue_y`, `tongue_likelihood`, with matching timestamps.

ii.
```python
tongue = behavior.time_series["Camera0_side_TongueTracking"]
tongue_times = np.asarray(tongue.timestamps[:], dtype=np.float64)
tongue_data = np.asarray(tongue.data[:], dtype=np.float64)
tongue_y = tongue_data[:, 1]
tongue_likelihood = tongue_data[:, 2]
```

iii. This is the only tongue measurement in the file, present in all sessions.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The AI uses a **nearest-frame** approach: for each bin center, it finds the nearest camera frame by timestamp. It then checks visibility using DLC likelihood >= 0.9. Visible frames have their y-value discretized against session-wide 40th/60th percentiles computed from all visible raw frames (not binned means). Non-visible bins get class 3.

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
tongue_class = np.full((len(trials), N_TIME), 3, dtype=np.int8)
tongue_class[visible & (matched_y < q40)] = 0
tongue_class[visible & (matched_y >= q40) & (matched_y <= q60)] = 1
tongue_class[visible & (matched_y > q60)] = 2
```

iii. The AI chose 0.9 as the DLC likelihood threshold, arguing that the likelihood is sharply bimodal near 0/1, making 0.9 a stable visibility threshold. Percentiles are computed on raw visible frames rather than binned means. Nearest-frame matching is used instead of averaging frames within each bin.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Session-wide 40th and 60th percentiles of all visible raw tongue y-values are computed. Each bin's nearest frame y-value is compared: < 40th -> 0, 40th to 60th -> 1, > 60th -> 2, not visible -> 3.

ii.
```python
q40, q60 = np.quantile(tongue_y[session_visible], [0.4, 0.6])
...
tongue_class[visible & (matched_y < q40)] = 0
tongue_class[visible & (matched_y >= q40) & (matched_y <= q60)] = 1
tongue_class[visible & (matched_y > q60)] = 2
```

iii. The boundary conditions use strict `<` for the 40th percentile and strict `>` for the 60th percentile, meaning exactly-at-threshold values fall into class 1. This differs from the reference which uses `np.digitize`.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. For each neural bin center (absolute time), the nearest camera frame by timestamp is found using `_nearest_indices` (a vectorized searchsorted-based nearest-neighbor lookup).

ii.
```python
nearest = _nearest_indices(tongue_times, absolute_centers.ravel()).reshape(
    len(trials), N_TIME
)
matched_y = tongue_y[nearest]
```

iii. The AI documented that camera timestamps share the global clock with spikes and events, so the same bin grid is applied.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Three cases:
- **Session with no good units**: The one session with all-NaN classifications is dropped because `.astype(str)` converts NaN to `'nan'` which doesn't match `'good'`.
- **Trials without neural data**: Trials where no spike occurred across any good unit in the entire window are excluded after binning.
- **Tongue not visible**: Frames with DLC likelihood < 0.9 are treated as not visible; bins matched to such frames become class 3.

ii.
```python
classifications = np.asarray(units["classification"][:]).astype(str)
good_indices = np.flatnonzero(classifications == "good")
if len(good_indices) == 0:
    return None
...
neural_present = np.any(rates != 0, axis=(1, 2))
...
tongue_class = np.full((len(trials), N_TIME), 3, dtype=np.int8)
```

iii. The AI documented handling each of these cases in CONVERSION_NOTES.md.

## 10-a. What are the most time-consuming steps of the code?

i. Reading each NWB file dominates. The full conversion takes ~223 seconds for 174 sessions. Within each session, the main costs are reading the spike times buffer and the tongue tracking array, then the per-unit searchsorted loop. Pickle writing is also significant for the ~11 GiB output.

ii. N/A (timing is logged at runtime)

iii. Documented in CONVERSION_NOTES.md: sample conversion was ~1.1 s/session mean, with full conversion safely under 15 minutes.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Two loops remain: the per-unit spike binning loop (one `searchsorted` per unit over all trial bin edges), and the per-trial photostim loop. The per-unit loop cannot be easily vectorized because each unit has a different number of spikes (ragged arrays). The photostim loop iterates over trials and their events.

ii.
```python
for out_idx, unit_idx in enumerate(good_indices):
    spikes = _unit_spike_times(units, int(unit_idx))
    ...
    positions = np.searchsorted(spikes, flattened, side="left")
```

```python
for i, (onsets, offsets) in enumerate(zip(stim_starts_by_trial, stim_stops_by_trial)):
    for onset, offset in zip(onsets, offsets):
        active = (...)
        inputs[i, 1, active] = 1.0
```

iii. The per-trial photostim loop could be vectorized (the reference does this with a single vectorized comparison). The spike loop is inherently per-unit due to ragged spike arrays.

## 10-c. What processing does the code repeat multiple times?

i. The AI reads spike times per-unit using the PyNWB API (`units["spike_times"][unit_index]`), which may re-read from disk for each unit. The reference reads the entire spike_times buffer once and slices it. The AI also reads video data once per session, which is efficient.

ii.
```python
def _unit_spike_times(units, unit_index: int) -> np.ndarray:
    return np.asarray(units["spike_times"][unit_index], dtype=np.float64)
```
vs reference:
```python
offs = np.asarray(units['spike_times'].data)
allst = np.asarray(units['spike_times'].target.data)
starts = np.concatenate([[0], offs[:-1]])
```

iii. The per-unit spike reading is less efficient than the reference's bulk read, but functionally equivalent.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes the `_any_observation_mask` function but never calls it in the final code (it was part of an earlier approach that was abandoned). The AI also bins spikes for ALL trials before filtering out zero-spike trials, which means it computes rates for trials that are subsequently discarded.

ii.
```python
def _any_observation_mask(units, good_indices, starts, stops):
    ...  # defined but not called

# All trials are binned first:
rates = _bin_spikes(units, good_indices, go_times)
# Then zero-spike trials are removed:
neural_present = np.any(rates != 0, axis=(1, 2))
```

iii. The unused `_any_observation_mask` function is dead code. Binning before filtering is slightly wasteful but produces correct results.
