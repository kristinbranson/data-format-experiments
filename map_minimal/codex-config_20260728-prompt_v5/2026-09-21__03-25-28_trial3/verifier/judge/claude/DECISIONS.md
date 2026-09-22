# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The dataset is distributed as one NWB file per session under `data/sub-<subject_id>/`. All sessions are found with `Path(data_dir).glob("sub-*/*.nwb")`, sorted, and each file is opened with `pynwb.NWBHDF5IO` and processed via `convert_session()`. Subjects, trials, and units are read from within each file (`nwb.subject`, `nwb.trials`, `nwb.units`, `nwb.acquisition`).

ii.
```python
NWB_GLOB = "sub-*/*.nwb"
session_paths = sorted(Path(data_dir).glob(NWB_GLOB))
for session_idx, path in enumerate(session_paths, start=1):
    session_data = convert_session(path, brain_region_to_idx)
```

```python
with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
    nwb = io.read()
    trials = nwb.trials.to_dataframe()
    units_df = nwb.units.to_dataframe()
```

iii. The agent explored the data directory structure and found 174 NWB files. It chose `pynwb` as the standard reader for NWB files. From the trajectory (step 9): "I have the dataset shape now: 173 NWB sessions." The agent also ran `find /app/data -maxdepth 3 | sort` to discover the file layout.

## 1-b. How are the data split into subjects?

i. Each NWB file records its animal in `nwb.subject.subject_id`. That value is read for every session, and at assembly time, subjects are collected into a list with a dictionary mapping subject IDs to indices. Subjects are added in the order they are first encountered (not sorted).

ii.
```python
subject = str(nwb.subject.subject_id)
# In convert_dataset:
if subject not in subject_to_idx:
    subject_to_idx[subject] = len(subjects)
    subjects.append(subject)
data["subject_idx"].append(subject_to_idx[subject])
```

iii. The agent used `nwb.subject.subject_id` (numeric strings like `'440956'`) as subject identifiers. From step 18, the agent discovered the subject field by inspecting the NWB metadata.

## 1-c. How are the data split into sessions?

i. One NWB file equals one session, so no splitting is needed. Each session is identified by the file stem (e.g., `sub-440956_ses-20190207T120657_...`). Session order follows the sorted file list. The session ID stored is `path.stem` rather than `nwb.identifier`.

ii.
```python
session_paths = sorted(Path(data_dir).glob(NWB_GLOB))
# ...
session_info = {
    "session_id": path.stem,
    ...
}
```

iii. The agent recognized from the data directory structure that each NWB file corresponds to one session. From step 9: "I have the dataset shape now: 173 NWB sessions."

## 1-d. How are the data split into trials?

i. Trials come from the NWB trials table (`nwb.trials.to_dataframe()`), one row per behavioral trial. Go cue times are extracted from `BehavioralEvents/go_start_times`.

ii.
```python
trials = nwb.trials.to_dataframe()
go_times = np.asarray(
    nwb.acquisition["BehavioralEvents"].time_series["go_start_times"].timestamps[:],
    dtype=np.float64,
)
go_times_kept = go_times[keep_trials]
```

iii. The agent explored the trial table structure and found columns including `start_time`, `stop_time`, `trial_instruction`, `early_lick`, `outcome`, etc. (step 20).

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered in three stages: (1) trials outside `obs_intervals` of the first good unit are excluded (matching both start and stop times with tolerance), (2) `auto_water` and `free_water` trials are excluded, (3) after computing neural data, trials where all neural bins are zero are removed. A session is dropped if fewer than 2 trials survive.

ii.
```python
recorded_trials = recorded_trial_mask_from_obs_intervals(trials, first_good_unit["obs_intervals"])
keep_trials = (
    recorded_trials
    & (trials["auto_water"].to_numpy() == 0)
    & (trials["free_water"].to_numpy() == 0)
)
# ...
# After neural computation:
nonzero_trial_mask = np.array([np.any(trial != 0) for trial in neural_trials], dtype=bool)
```

```python
def recorded_trial_mask_from_obs_intervals(trials, obs_intervals, atol=1e-4):
    obs_intervals = np.asarray(obs_intervals, dtype=np.float64)
    trial_starts = trials["start_time"].to_numpy(dtype=np.float64)
    trial_stops = trials["stop_time"].to_numpy(dtype=np.float64)
    keep = np.zeros(len(trials), dtype=bool)
    for obs_start, obs_stop in obs_intervals:
        keep |= np.isclose(trial_starts, obs_start, atol=atol) & np.isclose(trial_stops, obs_stop, atol=atol)
    return keep
```

iii. The agent initially did not filter by `obs_intervals`, resulting in many zero-neural trials (step 95). It then discovered the issue (step 100): "Some NWB files contain the full behavior session, but the units' obs_intervals only cover the recorded subset of trials." The `auto_water` filter was added alongside `free_water` (step 67). The nonzero neural trial filter was added as a final cleanup (step 110-112).

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `units["spike_times"]` for each good unit, and `go_start_times` timestamps for alignment.

ii.
```python
good_units = units_df.loc[keep_mask]
for unit_idx, spike_times in enumerate(good_units["spike_times"].to_list()):
    spikes = np.asarray(spike_times, dtype=np.float64)
```

iii. From step 67: "...using classification == 'good' units, go-cue alignment, 50 ms bins..."

## 2-b. How is the `neural` data processed?

i. Spike times are binned into 50 ms bins spanning -2.5 s to +1.5 s relative to the go cue (80 bins). For each unit, bin edges are computed as absolute times, `np.searchsorted` gives counts at edges, and differencing gives spike counts per bin. Counts are divided by bin width (0.05 s) to give firing rates in Hz. Data is stored as float16.

ii.
```python
abs_edges = go_times_kept[:, None] + BIN_EDGES_REL_S[None, :]
for unit_idx, spike_times in enumerate(good_units["spike_times"].to_list()):
    spikes = np.asarray(spike_times, dtype=np.float64)
    edge_idx = np.searchsorted(spikes, abs_edges.ravel(), side="left").reshape(ntrials, -1)
    counts = np.diff(edge_idx, axis=1)
    neural[:, unit_idx, :] = (counts / BIN_SIZE_S).astype(np.float16, copy=False)
```

iii. From step 128: "Alignment/binning: go-cue aligned, [-2.5, 1.5) s, 50 ms bins, firing rates from spike counts / 0.05."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `classification == 'good'` are kept. The classification column is cast to string and compared. A session with no good units is dropped entirely.

ii.
```python
good_unit_mask = units_df["classification"].astype(str).to_numpy() == "good"
if good_unit_mask.sum() == 0:
    return None
```

iii. From step 62: "I've ruled out two potential over-filters: the published session-level unit QC is the classification == 'good' label, and the newer per-unit is_good_trials mask is only occasionally nontrivial and doesn't appear in the reference analysis code." Step 78: "the one NWB session with zero published-good units explains why the paper reports 173 analyzable sessions out of 174 files."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The bin edges are computed relative to the go cue for each trial. Spike times and go cue times are on the same session-absolute clock, so no additional alignment is needed.

ii.
```python
abs_edges = go_times_kept[:, None] + BIN_EDGES_REL_S[None, :]
```

iii. The agent understood that all NWB times share a common clock. The bin grid is defined once as offsets from the go cue.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The bin size is 50 ms, with 80 bins spanning -2.5 s to +1.5 s relative to the go cue. No rebinning is applied; spikes are directly binned at this resolution.

ii.
```python
BIN_SIZE_S = 0.05
WINDOW_START_S = -2.5
WINDOW_END_S = 1.5

def make_time_grid():
    bin_edges = np.arange(WINDOW_START_S, WINDOW_END_S + BIN_SIZE_S / 2, BIN_SIZE_S)
    bin_centers = bin_edges[:-1] + BIN_SIZE_S / 2
    return bin_edges, bin_centers
```

iii. These parameters directly follow the instructions which specify 50 ms bins, -2.5 s to +1.5 s, aligned to go cue onset.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. Derived from `delay_start_times` (from `BehavioralEvents`), not `sample_start_times`. The agent finds the last delay event per trial, then subtracts 0.65 s (the sample epoch duration) to infer tone onset. If no delay event is found for a trial, it falls back to `go_time - 1.85 s`.

ii.
```python
delay_times = np.asarray(
    nwb.acquisition["BehavioralEvents"].time_series["delay_start_times"].timestamps[:],
    dtype=np.float64,
)
delay_last = last_event_per_trial(
    delay_times,
    trials_kept["start_time"].to_numpy(dtype=np.float64),
    trials_kept["stop_time"].to_numpy(dtype=np.float64),
)
tone_onset = np.where(np.isnan(delay_last), go_times_kept - 1.85, delay_last - 0.65)
```

iii. From step 48: "I found a helpful simplification: on cleanly mapped trials the final sample-epoch onset sits exactly 1.85 s before the go cue, including early-lick replays where the last replayed sample is the valid one." The agent verified that the last sample start is always 1.85s before the go cue, and chose to use delay_start_times minus the sample duration (0.65s) as the derivation method.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. For each bin, the value is `bin_center_absolute_time - tone_onset_absolute_time`, where the bin center is `go_time + bin_center_relative` and tone onset is derived from delay times as above.

ii.
```python
time_from_tone = (go_times_kept[:, None] + BIN_CENTERS_REL_S[None, :]) - tone_onset[:, None]
```

iii. The result is a continuous time-varying signal (seconds since tone onset) for each bin in each trial.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. Both use the same go-cue-relative bin grid (`BIN_CENTERS_REL_S`), so they are inherently aligned. The bin centers are the same for neural data and the tone onset input.

ii.
```python
# Same bin grid used for both:
abs_edges = go_times_kept[:, None] + BIN_EDGES_REL_S[None, :]  # neural
time_from_tone = (go_times_kept[:, None] + BIN_CENTERS_REL_S[None, :]) - tone_onset[:, None]  # input
```

iii. The alignment is guaranteed by construction since both use the same time grid.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. From `photostim_onset` and `photostim_duration` in the trials table, with `start_time` and the go cue used to place them on the trial's time axis.

ii.
```python
onset = trial["photostim_onset"]
duration = trial["photostim_duration"]
onset_abs = float(trial["start_time"]) + float(onset)
offset_abs = onset_abs + float(duration)
onset_rel = onset_abs - go_times_kept[i]
offset_rel = offset_abs - go_times_kept[i]
```

iii. The agent inspected photostim columns in the trials table (step 54) and found non-NaN values for stimulated trials.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A binary time-varying signal: 1 where the bin center falls between onset and offset of stimulation, 0 elsewhere. Non-stimulated trials (where onset or duration is `"N/A"`) remain all zeros.

ii.
```python
def build_photostim_input(trials_kept, go_times_kept):
    photostim = np.zeros((len(trials_kept), len(BIN_CENTERS_REL_S)), dtype=np.float32)
    for i, (_, trial) in enumerate(trials_kept.iterrows()):
        onset = trial["photostim_onset"]
        duration = trial["photostim_duration"]
        if onset == "N/A" or duration == "N/A":
            continue
        onset_abs = float(trial["start_time"]) + float(onset)
        offset_abs = onset_abs + float(duration)
        onset_rel = onset_abs - go_times_kept[i]
        offset_rel = offset_abs - go_times_kept[i]
        photostim[i] = ((BIN_CENTERS_REL_S >= onset_rel) & (BIN_CENTERS_REL_S < offset_rel)).astype(np.float32)
    return photostim
```

iii. The agent chose a binary time-varying representation as required by the instructions.

## 4-c. How is `input` *Photostimulation* aligned with the neural data?

i. The onset and offset are expressed relative to the go cue, and the bin centers are also relative to the go cue, so comparison is direct.

ii.
```python
onset_rel = onset_abs - go_times_kept[i]
offset_rel = offset_abs - go_times_kept[i]
photostim[i] = ((BIN_CENTERS_REL_S >= onset_rel) & (BIN_CENTERS_REL_S < offset_rel)).astype(np.float32)
```

iii. Same bin grid ensures alignment.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Derived from `trial_instruction` (left/right) and `outcome` (hit/miss/ignore) in the trials table, since choice is not stored directly.

ii.
```python
def choice_from_instruction_and_outcome(instruction, outcome):
    if outcome == "ignore":
        return 2
    if instruction == "left":
        return 0 if outcome == "hit" else 1
    if instruction == "right":
        return 1 if outcome == "hit" else 0
```

iii. The agent recognized that choice must be inferred: a hit means the animal licked the instructed side, a miss means the opposite side, and ignore means no lick.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Coded as 0 (left), 1 (right), 2 (no lick). The per-trial value is broadcast across all 80 time bins.

ii.
```python
choice = choice_from_instruction_and_outcome(
    str(trial["trial_instruction"]),
    str(trial["outcome"]),
)
output_trial = np.vstack([
    np.full(len(BIN_CENTERS_REL_S), choice, dtype=np.int8),
    ...
])
```

iii. Matches the instruction's categories: left, right, no lick.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Directly from the `outcome` column of the trials table, which holds `'ignore'`, `'miss'`, and `'hit'`.

ii.
```python
def outcome_to_int(outcome):
    mapping = {"ignore": 0, "miss": 1, "hit": 2}
    return mapping[outcome]
```

iii. The trials table stores outcome explicitly with exactly the three categories required.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Mapped to integers: 0 (ignore), 1 (miss), 2 (hit). Broadcast across all 80 time bins.

ii.
```python
outcome = outcome_to_int(str(trial["outcome"]))
np.full(len(BIN_CENTERS_REL_S), outcome, dtype=np.int8),
```

iii. Direct mapping from string to integer code.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. From the `early_lick` column of the trials table, which holds `'no early'` and `'early'`.

ii.
```python
def early_lick_to_int(value):
    mapping = {"no early": 0, "early": 1}
    return mapping[value]
```

iii. The trials table flags early licking explicitly.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Mapped to 0 (no) and 1 (yes). Broadcast across all 80 time bins.

ii.
```python
early = early_lick_to_int(str(trial["early_lick"]))
np.full(len(BIN_CENTERS_REL_S), early, dtype=np.int8),
```

iii. Direct mapping from string to integer code.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, with columns (x, y, likelihood) and matching timestamps.

ii.
```python
tongue_ts = nwb.acquisition["BehavioralTimeSeries"].time_series["Camera0_side_TongueTracking"]
tracking = np.asarray(tongue_ts.data[:], dtype=np.float32)
timestamps = np.asarray(tongue_ts.timestamps[:], dtype=np.float64)
y_all = tracking[:, 1]
likelihood_all = tracking[:, 2]
```

iii. The agent identified this as the only tongue measurement in the file (step 27).

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Frames with `likelihood < 0.9` are considered invisible. Session-wide 40th and 60th percentiles are computed on **raw visible frames** (not bin means). The per-trial discretization then uses a nearest-frame approach: for each bin center, the closest preceding camera frame is found via `searchsorted`, and if that frame is visible, it is classified by the percentile edges.

ii.
```python
TONGUE_VISIBILITY_THRESHOLD = 0.9

visible_all = likelihood_all >= TONGUE_VISIBILITY_THRESHOLD
visible_y = y_all[visible_all]
p40, p60 = np.percentile(visible_y, [40, 60])

abs_centers = go_times_kept[:, None] + BIN_CENTERS_REL_S[None, :]
frame_idx = np.searchsorted(timestamps, abs_centers.ravel(), side="right") - 1
frame_idx = frame_idx.reshape(abs_centers.shape)

tongue_state = np.full(abs_centers.shape, 3, dtype=np.int8)
valid = (frame_idx >= 0) & (frame_idx < len(timestamps))
if np.any(valid):
    y = y_all[frame_idx[valid]]
    likelihood = likelihood_all[frame_idx[valid]]
    visible = likelihood >= TONGUE_VISIBILITY_THRESHOLD
    state[visible & (y < p40)] = 0
    state[visible & (y >= p40) & (y <= p60)] = 1
    state[visible & (y > p60)] = 2
```

iii. From step 128: "tongue state from side-camera tongue tracking using DLC confidence >= 0.9 for visibility and per-session visible-frame y percentiles."

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Categories are:
- 0: y < 40th percentile
- 1: 40th to 60th percentile (inclusive on both ends)
- 2: y > 60th percentile
- 3: not visible (likelihood < 0.9)

The percentiles are computed on raw visible frames across the whole session, not on binned means.

ii.
```python
state[visible & (y < p40)] = 0
state[visible & (y >= p40) & (y <= p60)] = 1
state[visible & (y > p60)] = 2
```

iii. Follows the instruction's discretization scheme (< 40th, 40th-60th, > 60th, not visible).

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. For each bin center (absolute time = go_time + bin_center_relative), the nearest preceding camera frame is found. This is a nearest-frame lookup rather than averaging frames within a bin.

ii.
```python
abs_centers = go_times_kept[:, None] + BIN_CENTERS_REL_S[None, :]
frame_idx = np.searchsorted(timestamps, abs_centers.ravel(), side="right") - 1
```

iii. The camera timestamps share the same clock as spikes and go cues. The agent uses `searchsorted` to find the closest frame to each bin center.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Three cases:
- **Session with no good units**: `classification` is cast to string (NaN becomes `'nan'`), so sessions with no `'good'` units are dropped.
- **Trials without spike data**: excluded via `obs_intervals` matching (both start and stop times with tolerance) and `auto_water`/`free_water` filters.
- **Zero-neural trials**: after computing neural data, trials where all values are zero are explicitly removed.
- **Tongue not visible**: frames with low confidence get category 3 ("not visible").

ii.
```python
good_unit_mask = units_df["classification"].astype(str).to_numpy() == "good"
if good_unit_mask.sum() == 0:
    return None

# obs_intervals filtering
keep |= np.isclose(trial_starts, obs_start, atol=atol) & np.isclose(trial_stops, obs_stop, atol=atol)

# Zero neural removal
nonzero_trial_mask = np.array([np.any(trial != 0) for trial in neural_trials], dtype=bool)
```

iii. From step 100: "Some NWB files contain the full behavior session, but the units' obs_intervals only cover the recorded subset of trials." The zero-neural filter was added as a safety net (step 110-112).

## 10-a. What are the most time-consuming steps of the code?

i. Reading each NWB file and converting units to a dataframe (`units_df = nwb.units.to_dataframe()`) dominates, as this materializes all spike times into memory. The per-unit searchsorted loop and the per-trial photostim loop are also significant. The agent ran the full conversion multiple times (at least 3 runs due to iterative bug fixes).

ii.
```python
units_df = nwb.units.to_dataframe()
# per-unit loop:
for unit_idx, spike_times in enumerate(good_units["spike_times"].to_list()):
    spikes = np.asarray(spike_times, dtype=np.float64)
    edge_idx = np.searchsorted(spikes, abs_edges.ravel(), side="left").reshape(ntrials, -1)
```

iii. The agent monitored throughput (step 88) and found it stable enough not to optimize.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops remain:
- Per-unit loop in `build_neural_trials` (one `searchsorted` per unit) - same as reference, inherent to ragged spike data.
- Per-trial loop in `build_photostim_input` iterating over `trials_kept.iterrows()` - could be vectorized with array operations.
- Per-trial loop in `build_tongue_states` is already partially vectorized but uses a per-frame lookup approach.
- Per-trial loop assembling inputs/outputs in `convert_session` - could be vectorized with array broadcasting.

ii.
```python
# Photostim loop (could be vectorized):
for i, (_, trial) in enumerate(trials_kept.iterrows()):
    onset = trial["photostim_onset"]
    # ...

# Output assembly loop (could be vectorized):
for i, (_, trial) in enumerate(trials_kept.iterrows()):
    input_trial = np.vstack([time_from_tone[i], photostim[i]]).astype(np.float32, copy=False)
    # ...
```

iii. No explicit discussion of vectorization in the trajectory.

## 10-c. What processing does the code repeat multiple times?

i. The agent ran the full 174-session conversion at least 3 times due to iterative bug fixes (adding obs_intervals filtering, then adding nonzero-neural filtering). Within the code itself, the `units_df.to_dataframe()` call loads ALL units including non-good ones, which is more data than needed.

ii.
```python
units_df = nwb.units.to_dataframe()  # loads all units, not just good ones
good_unit_mask = units_df["classification"].astype(str).to_numpy() == "good"
```

iii. From step 85: "The conversion is scaling as expected."

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Two items:
- `units_df = nwb.units.to_dataframe()` loads all units into a DataFrame (including spike times for non-good units), even though only good units are used.
- The `obs_intervals` matching compares both start and stop times with `np.isclose`, looping over each interval pair, which is more work than needed if start times alone are sufficient.
- `delay_start_times` is loaded to compute tone onset, but `sample_start_times` would give the tone onset directly.

ii.
```python
units_df = nwb.units.to_dataframe()  # loads ALL units
# Only good ones are used later:
good_units = units_df.loc[keep_mask]
```

iii. No explicit discussion of unnecessary processing in the trajectory.
