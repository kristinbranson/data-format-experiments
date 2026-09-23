# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads all session files by globbing `/app/data/sub-*/*.nwb`, sorting the paths, and then processing each NWB file with `pynwb.NWBHDF5IO`. Within each session it reads the trials table, unit table, behavioral event streams, and behavioral time series from the NWB object.

ii. 
```python
files = sorted(DATA_DIR.glob("sub-*/*.nwb"))
...
with NWBHDF5IO(str(path), mode="r", load_namespaces=True) as io:
    nwb = io.read()
    trials = nwb.trials.to_dataframe()
    classifications = np.asarray(nwb.units["classification"][:]).astype(str)
    events = nwb.acquisition["BehavioralEvents"].time_series
    tracking = nwb.acquisition["BehavioralTimeSeries"].time_series["Camera0_side_TongueTracking"]
```

iii. In `CONVERSION_NOTES.md`, the AI says the dataset is “one `dandiset.yaml` plus 174 NWB 2.x files under 28 `sub-<id>/` directories” and that all content inspection used `pynwb.NWBHDF5IO(..., load_namespaces=True)` rather than `h5py`.

## 1-b. How are the data split into subjects?

i. The AI uses `nwb.subject.subject_id` as the per-session subject identifier, then builds a sorted unique `subjects` list and a per-session `subject_idx` lookup.

ii. 
```python
subject = str(nwb.subject.subject_id)
...
subjects = sorted({x["subject"] for x in converted})
subject_lookup = {x: i for i, x in enumerate(subjects)}
...
"subject_idx": np.asarray([subject_lookup[x["subject"]] for x in converted], dtype=np.int32),
```

iii. The notes justify this as using NWB metadata directly and expecting 28 subjects from the release.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. The session order is the sorted file order, and session metadata are taken from `nwb.identifier` plus bookkeeping stored in `session["info"]`.

ii. 
```python
files = sorted(DATA_DIR.glob("sub-*/*.nwb"))
...
session_id = str(nwb.identifier)
session = {
    ...
    "info": {
        "session_id": session_id,
        "source_file": str(path),
        ...
    },
}
```

iii. The notes state “one session per file” and explicitly decide to process all 174 NWBs, then omit the single session with zero classifier-good neurons.

## 1-d. How are the data split into trials?

i. The AI starts from the NWB trials table, checks that the number of go-cue events matches the number of trial rows, and then indexes into that table with a filtered set of trial indices.

ii. 
```python
trials = nwb.trials.to_dataframe()
...
go_all = np.asarray(events["go_start_times"].timestamps[:], dtype=np.float64)
if len(go_all) != len(trials):
    raise ValueError(f"{path.name}: go cue/trial count mismatch")
...
trial_inds = np.flatnonzero(trial_mask)
tr = trials.iloc[trial_inds]
```

iii. The notes treat `nwb.trials` as the canonical trial definition and use go-cue count matching as a consistency check.

## 1-e. How are trials filtered based on quality controls?

i. The AI keeps only trials for which every retained good unit has `is_good_trials == True` after expanding each unit’s insertion-local `obs_intervals` to session trial indices. It then removes any remaining trial whose binned neural activity is zero for all retained units across the full 4 s window. It does not explicitly exclude `free_water` trials.

ii. 
```python
unit_valid = np.stack([
    full_unit_trial_mask(nwb.units, i, trial_starts, trial_stops) for i in unit_inds
])
trial_mask = np.all(unit_valid, axis=0)
trial_inds = np.flatnonzero(trial_mask)
...
recorded_trial = np.any(rates != 0, axis=(0, 2))
if np.any(~recorded_trial):
    trial_inds = trial_inds[recorded_trial]
    ...
```

iii. The main justification in the notes is: “Keep trials for which every retained unit’s `is_good_trials` is true. This excludes invalid recording periods while preserving a constant neuron set.” The notes also say early/ignore/miss/stimulation/auto-water/free-water trials were intentionally retained because the decoder task needs those classes. After debugging, the AI added the extra all-zero-trial exclusion as an export edge-case fix.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural matrix is derived from `units["spike_times"]` for units whose `classification` is `"good"`, using go-cue timestamps to define trial-aligned bin edges.

ii. 
```python
classifications = np.asarray(nwb.units["classification"][:]).astype(str)
unit_inds = np.flatnonzero(classifications == "good")
...
go_all = np.asarray(events["go_start_times"].timestamps[:], dtype=np.float64)
...
spikes = np.asarray(nwb.units["spike_times"][unit_i], dtype=np.float64)
```

iii. The notes say to use the supplied multimetric classifier label exactly and not re-threshold QC metrics.

## 2-b. How is the `neural` data processed?

i. For each retained unit, the AI bins session-absolute spike times into non-overlapping 50 ms bins spanning `[-2.5, 1.5)` seconds around each trial’s go cue, then divides counts by bin width to convert to firing rates in spikes/s.

ii. 
```python
BIN_SIZE_S = 0.050
EDGES = np.linspace(OFF_START, OFF_END, 81, dtype=np.float64)
...
absolute_edges = go[:, None] + EDGES[None, :]
rates = np.empty((len(unit_inds), len(trial_inds), len(CENTERS)), dtype=np.float32)
for out_i, unit_i in enumerate(unit_inds):
    spikes = np.asarray(nwb.units["spike_times"][unit_i], dtype=np.float64)
    edge_indices = np.searchsorted(spikes, absolute_edges, side="left")
    rates[out_i] = np.diff(edge_indices, axis=1) / BIN_SIZE_S
```

iii. The notes describe this as matching the reference code’s spike-count-to-rate logic while replacing the paper’s 40 ms / 3.4 ms representation with the task-mandated 50 ms bins.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `classification == "good"` are kept. Sessions with zero such units are dropped.

ii. 
```python
classifications = np.asarray(nwb.units["classification"][:]).astype(str)
unit_inds = np.flatnonzero(classifications == "good")
if len(unit_inds) == 0:
    audit["sessions_zero_good_units"] += 1
    return None, audit
```

iii. The notes cite the QC white paper and explicitly reject replacing the classifier verdict with hand-chosen thresholds on individual quality metrics.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI aligns neural data to go cue onset by adding a fixed vector of relative bin edges to each trial’s absolute go-cue time. Spike times are already in the same absolute session clock, so no separate offset correction is applied.

ii. 
```python
go = go_all[trial_inds]
absolute_edges = go[:, None] + EDGES[None, :]
...
edge_indices = np.searchsorted(spikes, absolute_edges, side="left")
```

iii. The notes state that the released NWB timestamps are already on a common session clock, so raw spikes and behavior can be aligned directly to each go cue.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data use 80 non-overlapping bins of width 50 ms over a 4 s window from `-2.5` to `+1.5` seconds relative to go cue. There is no secondary rebinning or smoothing.

ii. 
```python
BIN_SIZE_S = 0.050
OFF_START = -2.5
OFF_END = 1.5
EDGES = np.linspace(OFF_START, OFF_END, 81, dtype=np.float64)
CENTERS = (EDGES[:-1] + EDGES[1:]) / 2
```

iii. The notes say this is a deliberate override of the movement-paper binning because the decoder task explicitly requires 50 ms bins.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It is derived from `BehavioralEvents/sample_start_times` and `BehavioralEvents/go_start_times`, using the last sample/tone onset before each trial’s go cue.

ii. 
```python
sample_times = np.asarray(events["sample_start_times"].timestamps[:], dtype=np.float64)
tone_all = last_sample_before_go(sample_times, trial_starts, go_all)
...
go = go_all[trial_inds]
tone = tone_all[trial_inds]
```

iii. The notes justify the “last sample before go” rule because early licks can replay the sample epoch multiple times within a trial.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The AI computes the absolute time at each go-centered bin center and subtracts the selected tone onset, producing a continuous time-since-tone value for each of the 80 bins.

ii. 
```python
tone_time = go[:, None] + CENTERS[None, :] - tone[:, None]
...
inputs.append(np.ascontiguousarray(np.vstack((tone_time[j], laser[j])), dtype=np.float32))
```

iii. The notes describe this as “continuous seconds since completed tone/sample onset.”

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is sampled on the exact same 80 go-centered bin centers as the neural firing rates, so there is one time-from-tone value per neural time bin.

ii. 
```python
CENTERS = (EDGES[:-1] + EDGES[1:]) / 2
...
absolute_edges = go[:, None] + EDGES[None, :]
tone_time = go[:, None] + CENTERS[None, :] - tone[:, None]
```

iii. The notes explicitly frame both neural and input timelines around the same go-centered bin grid.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. The AI derives photostimulation from the absolute event streams `BehavioralEvents/photostim_start_times` and `BehavioralEvents/photostim_stop_times`, not from `trials.photostim_onset` and `trials.photostim_duration`.

ii. 
```python
laser_starts = np.asarray(events["photostim_start_times"].timestamps[:], dtype=np.float64)
laser_stops = np.asarray(events["photostim_stop_times"].timestamps[:], dtype=np.float64)
if len(laser_starts) != len(laser_stops):
    raise ValueError(f"{path.name}: photostim start/stop mismatch")
```

iii. In the notes, the AI says raw spikes and timestamped behavior should be aligned directly on the NWB session clock, and it maps `photostim_start_times/stop_times` directly to the decoder input.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. The AI creates a binary time series by marking a bin as `1` whenever its time interval has positive-duration overlap with any photostimulation interval in the session; otherwise the bin is `0`.

ii. 
```python
def photostim_bins(starts: np.ndarray, stops: np.ndarray,
                   absolute_edges: np.ndarray) -> np.ndarray:
    result = np.zeros((n_trials, len(CENTERS)), dtype=np.float32)
    left, right = absolute_edges[:, :-1], absolute_edges[:, 1:]
    for onset, offset in zip(starts, stops):
        result[np.logical_and(left < offset, right > onset)] = 1.0
    return result
```

iii. The notes describe this as “Binary 1 where a 50-ms bin overlaps laser-on interval, otherwise 0.”

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Photostimulation is aligned using the same absolute bin edges used for neural binning. The AI computes go-centered absolute bin intervals and tests those intervals against absolute laser start/stop times.

ii. 
```python
absolute_edges = go[:, None] + EDGES[None, :]
laser = photostim_bins(laser_starts, laser_stops, absolute_edges)
```

iii. The AI’s notes justify this by treating both spikes and stimulation as event streams on the same absolute session timeline.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. The AI derives choice from the trials-table columns `trial_instruction` and `outcome`. It also computes a lick-event-based audit value from `left_lick_times` and `right_lick_times`, but that audit value is not what gets stored as the decoder output.

ii. 
```python
instructions = tr.trial_instruction.astype(str).to_numpy()
outcomes_str = tr.outcome.astype(str).to_numpy()
choices = np.asarray([derive_choice(a, b) for a, b in zip(instructions, outcomes_str)], dtype=np.int8)
...
left_licks = np.asarray(events["left_lick_times"].timestamps[:], dtype=np.float64)
right_licks = np.asarray(events["right_lick_times"].timestamps[:], dtype=np.float64)
lick_choices = np.asarray([
    first_response_lick_choice(left_licks, right_licks, g) for g in go
], dtype=np.int8)
```

iii. The notes say the trial table contains instruction and outcome but not explicit choice, and that choice is deterministic from those fields in a two-port task; lick times are only an independent sanity check.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The AI maps each trial to `0=left`, `1=right`, or `2=no lick`, then broadcasts that per-trial class across all 80 time bins.

ii. 
```python
def derive_choice(instruction: str, outcome: str) -> int:
    if outcome == "ignore":
        return 2
    instructed = 0 if instruction == "left" else 1
    return instructed if outcome == "hit" else 1 - instructed
...
out = np.empty((4, len(CENTERS)), dtype=np.int8)
out[0] = choices[j]
```

iii. The notes justify the mapping as hit=instructed side, miss=opposite side, ignore=no lick, with repetition across time so outputs share a uniform `(4, 80)` shape.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is taken directly from the trials-table `outcome` column.

ii. 
```python
outcomes_str = tr.outcome.astype(str).to_numpy()
```

iii. The notes treat this as a direct categorical field already present in the NWB trial table.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The strings `ignore`, `miss`, and `hit` are mapped to integer codes `0`, `1`, and `2`, then repeated across all 80 time bins.

ii. 
```python
outcome_map = {"ignore": 0, "miss": 1, "hit": 2}
outcomes = np.asarray([outcome_map[x] for x in outcomes_str], dtype=np.int8)
...
out[1] = outcomes[j]
```

iii. The notes say the output is broadcast across time to keep the full output array time-aligned and rectangular.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Early lick is taken directly from the trials-table `early_lick` column.

ii. 
```python
early_str = tr.early_lick.astype(str).to_numpy()
```

iii. The notes describe this as an explicit per-trial flag already present in the trial table.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The AI maps `no early` to `0` and `early` to `1`, then repeats that class across all 80 time bins.

ii. 
```python
early_map = {"no early": 0, "early": 1}
early = np.asarray([early_map[x] for x in early_str], dtype=np.int8)
...
out[2] = early[j]
```

iii. The notes again justify repetition across time as a uniform decoder-output representation.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. Tongue y-position is derived from `BehavioralTimeSeries/Camera0_side_TongueTracking`: column 1 of the tracking data is `y`, and column 2 is the tracking likelihood used as the visibility indicator.

ii. 
```python
tracking = nwb.acquisition["BehavioralTimeSeries"].time_series["Camera0_side_TongueTracking"]
track_t = np.asarray(tracking.timestamps[:], dtype=np.float64)
track_data = np.asarray(tracking.data[:], dtype=np.float64)
raw_y, likelihood = track_data[:, 1], track_data[:, 2]
```

iii. The notes say the side-camera tongue tracking stream is present in every file and is the source for this output.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The AI marks frames as visible only when `likelihood >= 0.9`, computes session-wide 40th and 60th percentiles of `raw_y` over those visible frames, then assigns each neural bin the nearest camera frame in time and discretizes its `y` value using those session thresholds. No within-bin averaging is performed.

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
```

iii. The notes justify this as using a conservative 0.9 visibility threshold because the DLC likelihood is strongly bimodal, and as computing percentiles over all visible session frames “as requested.”

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. The AI uses four classes: `0` for visible frames with `y < q40`, `1` for visible frames with `q40 <= y <= q60`, `2` for visible frames with `y > q60`, and `3` for bins deemed not visible.

ii. 
```python
tongue_class = np.full(query.shape, 3, dtype=np.int8)
tongue_class[tongue_visible & (y < q40)] = 0
tongue_class[tongue_visible & (y >= q40) & (y <= q60)] = 1
tongue_class[tongue_visible & (y > q60)] = 2
```

iii. The notes explicitly list this boundary convention and the dedicated hidden/occluded class.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Alignment is done by querying the nearest camera timestamp to each go-centered neural bin center. The AI does not average all frames inside each 50 ms neural bin.

ii. 
```python
query = go[:, None] + CENTERS[None, :]
track_idx = nearest_indices(track_t, query.ravel()).reshape(query.shape)
y = raw_y[track_idx]
```

iii. The notes describe this as “Nearest video sample at each bin center.”

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several edge cases by either dropping data or raising errors. Sessions with zero good units are dropped. Unit-local `is_good_trials` vectors are expanded through `obs_intervals` and must map exactly to session trials, otherwise conversion errors out. Trials with an all-zero population over the whole neural window are excluded. Tongue bins are marked “not visible” if the nearest sampled frame fails the likelihood threshold, but a session with no visible tongue frames raises an error.

ii. 
```python
if len(unit_inds) == 0:
    return None, audit
...
if len(local_good) != len(intervals):
    raise ValueError(f"unit {unit_i}: is_good_trials/obs_intervals mismatch")
...
recorded_trial = np.any(rates != 0, axis=(0, 2))
if np.any(~recorded_trial):
    ...
...
if not np.any(visible):
    raise ValueError(f"{path.name}: no visible tongue frames")
```

iii. The notes justify the all-zero-trial exclusion as an NWB export edge case rather than real physiology and say the exact `obs_intervals` mapping was added after discovering that `is_good_trials` is insertion-local.

## 10-a. What are the most time-consuming steps of the code?

i. The AI identifies NWB I/O, loading large spike-time and tracking arrays, the per-unit spike binning loop, and final serialization as the dominant costs.

ii. 
```python
for out_i, unit_i in enumerate(unit_inds):
    spikes = np.asarray(nwb.units["spike_times"][unit_i], dtype=np.float64)
    edge_indices = np.searchsorted(spikes, absolute_edges, side="left")
    rates[out_i] = np.diff(edge_indices, axis=1) / BIN_SIZE_S
...
with args.outpicklefile.open("wb") as f:
    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. In `CONVERSION_NOTES.md`, the AI says the main costs are “pulling the `spike_times` buffer,” reading the tongue tracking array, the per-unit `searchsorted` loop, and writing the large pickle.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The code still contains several Python loops: the per-unit `full_unit_trial_mask` expansion, the per-unit spike binning loop, the loop over photostim intervals, the per-trial lick-choice audit, and the final per-trial list assembly. The AI already vectorized the binning over trials/time within each unit and vectorized nearest-frame lookup for tongue alignment.

ii. 
```python
unit_valid = np.stack([
    full_unit_trial_mask(nwb.units, i, trial_starts, trial_stops) for i in unit_inds
])
...
for out_i, unit_i in enumerate(unit_inds):
    ...
...
for onset, offset in zip(starts, stops):
    result[np.logical_and(left < offset, right > onset)] = 1.0
...
for j in range(len(trial_inds)):
    neural.append(...)
    inputs.append(...)
    outputs.append(out)
```

iii. The notes emphasize the vectorizations that were added, especially using one `searchsorted` per unit over all trial edges and vectorizing tracking lookup, but they do not claim the code is fully loop-free.

## 10-c. What processing does the code repeat multiple times?

i. The core conversion is mostly single-pass, but some quantities are intentionally recomputed for validation or packaging. Choice is derived once from instruction/outcome and independently recomputed from lick events for an audit. Trial outputs are also repackaged into per-trial arrays in the final loop even after session-level arrays (`rates`, `tone_time`, `laser`, `choices`, etc.) already exist.

ii. 
```python
choices = np.asarray([derive_choice(a, b) for a, b in zip(instructions, outcomes_str)], dtype=np.int8)
...
lick_choices = np.asarray([
    first_response_lick_choice(left_licks, right_licks, g) for g in go
], dtype=np.int8)
...
for j in range(len(trial_inds)):
    neural.append(np.ascontiguousarray(rates[:, j, :], dtype=np.float32))
    inputs.append(np.ascontiguousarray(np.vstack((tone_time[j], laser[j])), dtype=np.float32))
```

iii. The notes frame the lick-based choice computation as an “independent check,” not as the authoritative choice label.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The conversion script does extra validation and diagnostic work that is not needed by downstream decoder training: it computes a lick-event-based choice audit, tracks audit counters, optionally renders processing plots, and stores extra metadata such as auto/free-water counts and source file paths. None of that changes the neural/input/output tensors.

ii. 
```python
lick_choices = np.asarray([
    first_response_lick_choice(left_licks, right_licks, g) for g in go
], dtype=np.int8)
audit["choice_lick_matches"] += int(np.sum(choices == lick_choices))
audit["choice_lick_mismatches"] += int(np.sum(choices != lick_choices))
...
if show_processing:
    plot_processing(session_id, {...})
...
"info": {
    "source_file": str(path),
    ...
    "auto_water_trials_included": int(tr.auto_water.sum()),
    "free_water_trials_included": int(tr.free_water.sum()),
},
```

iii. The notes justify these extras as sanity checks and validation support rather than as part of the final decoder representation.
