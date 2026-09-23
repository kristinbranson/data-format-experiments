# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI globs all NWB files under `data/sub-*/` and processes each with `pynwb.NWBHDF5IO`. Each session file is opened once, and trials, units, and behavioral events are accessed via the NWB API (`nwb.trials`, `nwb.units`, `nwb.acquisition`).

ii.
```python
files = sorted(DATA_DIR.glob("sub-*/*.nwb"))
...
for i, path in enumerate(files):
    session, session_audit = convert_session(path, ...)
```

```python
with NWBHDF5IO(str(path), mode="r", load_namespaces=True) as io:
    nwb = io.read()
    trials = nwb.trials.to_dataframe()
    classifications = np.asarray(nwb.units["classification"][:]).astype(str)
```

iii. From CONVERSION_NOTES.md Step 2: "All content inspection used `pynwb.NWBHDF5IO(..., load_namespaces=True)`; `h5py` was not used." The AI identified 174 NWB files across 28 subject directories, matching the dandiset metadata.

## 1-b. How are the data split into subjects?

i. Each NWB file's `nwb.subject.subject_id` provides the subject identifier. The AI collects all unique subject IDs, sorts them, and maps each session to an index.

ii.
```python
subject = str(nwb.subject.subject_id)
...
subjects = sorted({x["subject"] for x in converted})
subject_lookup = {x: i for i, x in enumerate(subjects)}
```

iii. From CONVERSION_NOTES.md Step 5: "28 subjects expected." The AI uses the NWB-native subject_id field.

## 1-c. How are the data split into sessions?

i. One NWB file equals one session. The session is identified by `nwb.identifier`. Sessions are processed in sorted file order.

ii.
```python
files = sorted(DATA_DIR.glob("sub-*/*.nwb"))
...
session_id = str(nwb.identifier)
```

iii. CONVERSION_NOTES.md Step 2: "174 NWB 2.x files under 28 sub-<id>/ directories. Each file is one behavior+ecephys session."

## 1-d. How are the data split into trials?

i. Trials come from `nwb.trials.to_dataframe()`, one row per trial. The AI verifies that go-cue event count matches the trial count.

ii.
```python
trials = nwb.trials.to_dataframe()
...
go_all = np.asarray(events["go_start_times"].timestamps[:], dtype=np.float64)
if len(go_all) != len(trials):
    raise ValueError(f"{path.name}: go cue/trial count mismatch")
```

iii. The AI validates the one-to-one correspondence between trials and go cues as a sanity check.

## 1-e. How are trials filtered based on quality controls?

i. The AI uses per-unit `is_good_trials` masks expanded via `obs_intervals` to the full trial table. A trial is kept only if ALL good units have `is_good_trials == True` for that trial. Additionally, trials with all-zero neural activity across the entire population are excluded. Free water trials are NOT explicitly filtered. Sessions with fewer than 2 valid trials raise an error.

ii.
```python
unit_valid = np.stack([
    full_unit_trial_mask(nwb.units, i, trial_starts, trial_stops) for i in unit_inds
])
trial_mask = np.all(unit_valid, axis=0)
trial_inds = np.flatnonzero(trial_mask)
...
# Exclude all-zero neural trials
recorded_trial = np.any(rates != 0, axis=(0, 2))
if np.any(~recorded_trial):
    trial_inds = trial_inds[recorded_trial]
    ...
```

iii. From CONVERSION_NOTES.md Step 5: "Keep trials for which every retained unit's `is_good_trials` is true. This excludes invalid recording periods while preserving a constant neuron set." And Step 10: "1,569 rejected by insertion-local good-trial masks; 2,423 all-population-zero trials rejected as recording gaps; 90,378 retained." The AI does not filter free_water trials, noting in metadata `free_water_trials_included`.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `units/spike_times` for units with `classification == 'good'`, and `go_start_times` for alignment.

ii.
```python
spikes = np.asarray(nwb.units["spike_times"][unit_i], dtype=np.float64)
edge_indices = np.searchsorted(spikes, absolute_edges, side="left")
rates[out_i] = np.diff(edge_indices, axis=1) / BIN_SIZE_S
```

iii. CONVERSION_NOTES.md Step 5: "For each go-aligned trial, histogram absolute spikes into 80 half-open 50-ms bins."

## 2-b. How is the `neural` data processed?

i. Spike times are binned into 50ms non-overlapping bins using `np.searchsorted` on absolute time edges around each go cue. Spike counts are divided by bin width (0.05 s) to convert to firing rates in Hz. No smoothing or normalization is applied.

ii.
```python
absolute_edges = go[:, None] + EDGES[None, :]
...
for out_i, unit_i in enumerate(unit_inds):
    spikes = np.asarray(nwb.units["spike_times"][unit_i], dtype=np.float64)
    edge_indices = np.searchsorted(spikes, absolute_edges, side="left")
    rates[out_i] = np.diff(edge_indices, axis=1) / BIN_SIZE_S
```

iii. CONVERSION_NOTES.md Step 5: "Same spike-count/rate principle; task-mandated bins."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `classification == 'good'` are retained. No individual QC metric thresholds are applied. Sessions with zero good units are excluded.

ii.
```python
classifications = np.asarray(nwb.units["classification"][:]).astype(str)
unit_inds = np.flatnonzero(classifications == "good")
if len(unit_inds) == 0:
    audit["sessions_zero_good_units"] += 1
    return None, audit
```

iii. CONVERSION_NOTES.md Step 5: "Use supplied multimetric classifier label exactly (`classification == good`); no individual QC thresholds because the QC white paper explicitly rejects that strategy."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. All NWB timestamps share a session-absolute clock. The go-cue time for each trial defines the center of the trial window, and bin edges are computed as offsets from the go cue.

ii.
```python
go_all = np.asarray(events["go_start_times"].timestamps[:], dtype=np.float64)
...
absolute_edges = go[:, None] + EDGES[None, :]
```

iii. CONVERSION_NOTES.md Step 4: "align raw spikes and timestamped behavior directly to each NWB go onset."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 50 ms non-overlapping bins, 80 bins spanning [-2.5, +1.5) seconds relative to go cue. Bin edges are defined with `np.linspace`.

ii.
```python
BIN_SIZE_S = 0.050
OFF_START = -2.5
OFF_END = 1.5
EDGES = np.linspace(OFF_START, OFF_END, 81, dtype=np.float64)
CENTERS = (EDGES[:-1] + EDGES[1:]) / 2
```

iii. CONVERSION_NOTES.md Step 4: "Required non-overlapping 50-ms bins supersede paper analysis bins."

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. From `sample_start_times` (tone onset events) and go cue times. The last sample onset at or before each trial's go cue is taken as the tone for that trial.

ii.
```python
sample_times = np.asarray(events["sample_start_times"].timestamps[:], dtype=np.float64)
tone_all = last_sample_before_go(sample_times, trial_starts, go_all)
```

```python
def last_sample_before_go(sample_times, trial_starts, go_times):
    inds = np.searchsorted(sample_times, go_times, side="right") - 1
    ...
    return sample_times[inds]
```

iii. CONVERSION_NOTES.md Step 5: "Continuous seconds since completed tone/sample onset."

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. Computed as `go + bin_center - tone_onset` for each trial and bin, giving seconds since tone onset at each time point.

ii.
```python
tone_time = go[:, None] + CENTERS[None, :] - tone[:, None]
```

iii. The AI computes elapsed time from tone onset for each bin center, producing a continuous time-varying input.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. The same bin centers (CENTERS) are used for both neural binning and tone-time computation, ensuring alignment. Both are defined relative to the go cue.

ii.
```python
CENTERS = (EDGES[:-1] + EDGES[1:]) / 2
...
tone_time = go[:, None] + CENTERS[None, :] - tone[:, None]
```

iii. The bin grid is shared between neural and input streams.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. From `photostim_start_times` and `photostim_stop_times` in `BehavioralEvents` (session-level laser event timestamps), not from the per-trial `photostim_onset`/`photostim_duration` in the trials table.

ii.
```python
laser_starts = np.asarray(events["photostim_start_times"].timestamps[:], dtype=np.float64)
laser_stops = np.asarray(events["photostim_stop_times"].timestamps[:], dtype=np.float64)
```

iii. CONVERSION_NOTES.md Step 5: "Binary 1 where a 50-ms bin overlaps laser-on interval, otherwise 0." The AI chose session-level event timestamps rather than per-trial fields.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A bin is marked 1 if it has any positive-duration overlap with any laser-on interval (onset, offset), and 0 otherwise. This is computed by checking whether each bin's left edge < laser offset AND bin's right edge > laser onset.

ii.
```python
def photostim_bins(starts, stops, absolute_edges):
    n_trials = absolute_edges.shape[0]
    result = np.zeros((n_trials, len(CENTERS)), dtype=np.float32)
    left, right = absolute_edges[:, :-1], absolute_edges[:, 1:]
    for onset, offset in zip(starts, stops):
        result[np.logical_and(left < offset, right > onset)] = 1.0
    return result
```

iii. The AI uses a bin-overlap criterion (any part of the bin overlaps laser-on), rather than the reference's bin-center criterion.

## 4-c. How is `input` *Photostimulation* aligned with the neural data?

i. The absolute bin edges (go + EDGES) used for neural binning are reused to define the bin intervals for laser overlap testing.

ii.
```python
absolute_edges = go[:, None] + EDGES[None, :]
...
laser = photostim_bins(laser_starts, laser_stops, absolute_edges)
```

iii. The same go-cue-relative edge grid ensures alignment.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Derived from `trial_instruction` (left/right) and `outcome` (hit/miss/ignore) in the trials table. Hit means instructed side, miss means opposite side, ignore means no lick.

ii.
```python
def derive_choice(instruction, outcome):
    if outcome == "ignore":
        return 2
    instructed = 0 if instruction == "left" else 1
    return instructed if outcome == "hit" else 1 - instructed
```

iii. CONVERSION_NOTES.md Step 5: "In a two-port task, outcome deterministically maps instruction to actual choice."

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Choice is coded as 0=left, 1=right, 2=no lick. Per-trial value is broadcast across all 80 time bins. The AI also independently verifies against first response-window lick events.

ii.
```python
choices = np.asarray([derive_choice(a, b) for a, b in zip(instructions, outcomes_str)], dtype=np.int8)
...
out[0] = choices[j]  # broadcast over bins
```

iii. CONVERSION_NOTES.md Step 10: "Choice matched first response-window lick on 90,096/90,378 trials (99.69%)."

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Directly from `trials.outcome` column, which contains 'ignore', 'miss', 'hit'.

ii.
```python
outcomes_str = tr.outcome.astype(str).to_numpy()
outcome_map = {"ignore": 0, "miss": 1, "hit": 2}
outcomes = np.asarray([outcome_map[x] for x in outcomes_str], dtype=np.int8)
```

iii. The trials table stores outcome explicitly.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Mapped to integers: ignore=0, miss=1, hit=2. Broadcast across all 80 time bins per trial.

ii.
```python
outcome_map = {"ignore": 0, "miss": 1, "hit": 2}
outcomes = np.asarray([outcome_map[x] for x in outcomes_str], dtype=np.int8)
...
out[1] = outcomes[j]
```

iii. Direct categorical encoding matching the instructions.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. From `trials.early_lick` column, which contains 'no early' and 'early'.

ii.
```python
early_str = tr.early_lick.astype(str).to_numpy()
early_map = {"no early": 0, "early": 1}
early = np.asarray([early_map[x] for x in early_str], dtype=np.int8)
```

iii. The trials table stores early lick status explicitly.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Mapped to integers: no early=0, early=1. Broadcast across all 80 time bins.

ii.
```python
early_map = {"no early": 0, "early": 1}
early = np.asarray([early_map[x] for x in early_str], dtype=np.int8)
...
out[2] = early[j]
```

iii. Direct categorical encoding matching the instructions.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, columns 1 (tongue_y) and 2 (tongue_likelihood), with matching timestamps.

ii.
```python
tracking = nwb.acquisition["BehavioralTimeSeries"].time_series["Camera0_side_TongueTracking"]
track_t = np.asarray(tracking.timestamps[:], dtype=np.float64)
track_data = np.asarray(tracking.data[:], dtype=np.float64)
raw_y, likelihood = track_data[:, 1], track_data[:, 2]
```

iii. CONVERSION_NOTES.md Step 5: "Side-camera tongue y, likelihood <0.9 -> 3 not visible."

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Frames with likelihood < 0.9 are marked as not visible. The 40th and 60th percentiles are computed from raw visible frames across the whole session. For each trial bin, the nearest video frame to the bin center is looked up, and if visible, its y-value is classified against the percentile thresholds.

ii.
```python
TONGUE_VISIBLE_LIKELIHOOD = 0.9
...
visible = np.isfinite(raw_y) & np.isfinite(likelihood) & (likelihood >= TONGUE_VISIBLE_LIKELIHOOD)
q40, q60 = np.quantile(raw_y[visible], [0.4, 0.6])
query = go[:, None] + CENTERS[None, :]
track_idx = nearest_indices(track_t, query.ravel()).reshape(query.shape)
y = raw_y[track_idx]
tongue_visible = likelihood[track_idx] >= TONGUE_VISIBLE_LIKELIHOOD
tongue_class = np.full(query.shape, 3, dtype=np.int8)
tongue_class[tongue_visible & (y < q40)] = 0
tongue_class[tongue_visible & (y >= q40) & (y <= q60)] = 1
tongue_class[tongue_visible & (y > q60)] = 2
```

iii. CONVERSION_NOTES.md Step 5: "DLC likelihood is strongly bimodal near 0/1; 0.9 is a conservative standard visibility threshold. Percentiles computed before trial selection over all visible session frames."

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Class 0: y < 40th percentile. Class 1: 40th <= y <= 60th percentile. Class 2: y > 60th percentile. Class 3: not visible (likelihood < 0.9 or no frame).

ii.
```python
tongue_class[tongue_visible & (y < q40)] = 0
tongue_class[tongue_visible & (y >= q40) & (y <= q60)] = 1
tongue_class[tongue_visible & (y > q60)] = 2
```

iii. The boundary conditions place values exactly at the 60th percentile in class 1 (between classes), while the instructions specify class 1 as "40th to 60th percentile" and class 2 as "> 60th percentile", so this is consistent with the instructions.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. For each bin center, the nearest video frame (by timestamp) is looked up using `nearest_indices`. This single-frame lookup replaces averaging multiple frames within the bin.

ii.
```python
def nearest_indices(timestamps, query):
    right = np.searchsorted(timestamps, query, side="left")
    right = np.clip(right, 0, len(timestamps) - 1)
    left = np.maximum(right - 1, 0)
    choose_left = np.abs(query - timestamps[left]) <= np.abs(timestamps[right] - query)
    return np.where(choose_left, left, right)

query = go[:, None] + CENTERS[None, :]
track_idx = nearest_indices(track_t, query.ravel()).reshape(query.shape)
```

iii. CONVERSION_NOTES.md Step 5: "Nearest video sample at each bin center." The camera runs at ~294 Hz, so ~15 frames per 50ms bin; this approach uses only the single nearest frame rather than averaging all frames in the bin.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Three cases handled: (1) Sessions with zero classifier-good units (NaN classifications) are dropped. (2) Trials where not all good units have `is_good_trials == True` are excluded, plus all-zero neural trials are excluded. (3) Tongue frames with low likelihood are assigned class 3 "not visible".

ii.
```python
classifications = np.asarray(nwb.units["classification"][:]).astype(str)
unit_inds = np.flatnonzero(classifications == "good")
if len(unit_inds) == 0:
    return None, audit
...
trial_mask = np.all(unit_valid, axis=0)
...
recorded_trial = np.any(rates != 0, axis=(0, 2))
...
tongue_class = np.full(query.shape, 3, dtype=np.int8)  # default not visible
```

iii. CONVERSION_NOTES.md Step 10: "Fixed insertion-local validity vectors; excluded 2,423 trials with no recorded population spikes despite nominal intervals; dropped the sole zero-good-unit session."

## 10-a. What are the most time-consuming steps of the code?

i. Reading each NWB file and extracting spike times dominates. The full conversion takes ~3.19 minutes for 174 sessions. Serialization of the ~11.8 GB pickle is also significant.

ii. N/A (timing reported in CONVERSION_NOTES.md)

iii. CONVERSION_NOTES.md Step 9: "Full conversion took 3.19 min."

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Two main loops remain: (1) Per-unit loop for spike binning (one `searchsorted` per unit over all trials). (2) Per-trial loop for assembling output arrays. The per-unit loop cannot be fully vectorized due to ragged spike arrays.

ii.
```python
for out_i, unit_i in enumerate(unit_inds):
    spikes = np.asarray(nwb.units["spike_times"][unit_i], dtype=np.float64)
    edge_indices = np.searchsorted(spikes, absolute_edges, side="left")
    rates[out_i] = np.diff(edge_indices, axis=1) / BIN_SIZE_S
```

iii. The per-unit loop is inherent to ragged spike storage. The per-trial output assembly loop is simple and not a bottleneck.

## 10-c. What processing does the code repeat multiple times?

i. Nothing significant is recomputed. Each NWB file is opened once and all quantities derived in a single pass. Bin edges (EDGES, CENTERS) are computed once at module level.

ii. N/A

iii. The conversion is a single pass over files.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes `first_response_lick_choice` from left/right lick event timestamps to verify the instruction+outcome derived choice. This lick-based choice is used only for auditing (counting matches/mismatches) and is not included in the output data.

ii.
```python
left_licks = np.asarray(events["left_lick_times"].timestamps[:], dtype=np.float64)
right_licks = np.asarray(events["right_lick_times"].timestamps[:], dtype=np.float64)
lick_choices = np.asarray([
    first_response_lick_choice(left_licks, right_licks, g) for g in go
], dtype=np.int8)
audit["choice_lick_matches"] += int(np.sum(choices == lick_choices))
audit["choice_lick_mismatches"] += int(np.sum(choices != lick_choices))
```

iii. This is a validation/sanity check, not a processing error, but it does add unnecessary computation to the conversion pipeline.
