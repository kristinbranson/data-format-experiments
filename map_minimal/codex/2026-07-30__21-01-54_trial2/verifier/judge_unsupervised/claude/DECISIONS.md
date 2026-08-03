# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads all NWB files matching the glob pattern `data/sub-*/*.nwb` under the data directory. Each NWB file is opened with `h5py` (not `pynwb`) and read directly from HDF5 paths. 174 NWB files are found and processed sequentially in sorted order. Trial data comes from `intervals/trials`, neural data from `units`, behavioral events from `acquisition/BehavioralEvents`, and tongue tracking from `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`.

ii.
```python
nwb_paths = sorted(data_dir.glob("sub-*/*.nwb"))
# ...
for session_number, path in enumerate(nwb_paths, start=1):
    with h5py.File(path, "r") as f:
        trials = f["intervals/trials"]
        # ... reads trial_starts, trial_stops, trial_instruction, outcomes, early_lick,
        # auto_water, free_water, photostim_onset, photostim_duration, photostim_power
        be = f["acquisition/BehavioralEvents"]
        go_starts = np.asarray(be["go_start_times"]["timestamps"], dtype=np.float64)
        sample_starts = np.asarray(be["sample_start_times"]["timestamps"], dtype=np.float64)
        tongue_group = f["acquisition/BehavioralTimeSeries"]["Camera0_side_TongueTracking"]
        units = f["units"]
```

iii. The agent chose `h5py` over `pynwb` for speed after initially exploring with `pynwb`. The agent noted: "I'm switching to faster HDF5-level inspection so I can find the exact inclusion rule rather than guessing which session to drop."

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified by extracting `subject_id` from each NWB file's `general/subject` group. A running map (`subject_to_idx`) tracks unique subjects and assigns each session a subject index. 28 unique subjects were found.

ii.
```python
subject = _decode_scalar(f["general"]["subject"]["subject_id"][()])
subject = str(subject)
if subject not in subject_to_idx:
    subject_to_idx[subject] = len(subjects)
    subjects.append(subject)
subject_idx.append(subject_to_idx[subject])
```

iii. Subject IDs are read directly from the NWB metadata. The agent confirmed 28 subjects by cross-referencing with the number of `sub-*` directories.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. Sessions are processed in sorted file path order. Sessions with zero good units are skipped (1 session dropped), and sessions with fewer than 2 kept trials after filtering are also skipped (though none triggered this). Final count: 173 sessions.

ii.
```python
nwb_paths = sorted(data_dir.glob("sub-*/*.nwb"))
# ...
if len(good_unit_indices) == 0:
    continue
# ...
if n_keep_trials < 2:
    continue
```

iii. The agent discovered that the NWB release contains 174 files but the paper references 173 sessions. One file (`sub-440958_ses-20190216T162508`) has zero good units, naturally resolving the discrepancy.

## 1-d. How are the data split into trials?

i. Trials are taken from each session's `intervals/trials` table. The trial count is determined by `len(trials["id"])`. Go cue timestamps from `BehavioralEvents/go_start_times` are validated to match the trial count.

ii.
```python
trials = f["intervals/trials"]
n_trials = len(trials["id"])
trial_starts = np.asarray(trials["start_time"], dtype=np.float64)
trial_stops = np.asarray(trials["stop_time"], dtype=np.float64)
# ...
go_starts = np.asarray(be["go_start_times"]["timestamps"], dtype=np.float64)
if len(go_starts) != n_trials:
    raise RuntimeError(f"{path.name}: expected {n_trials} go cues, found {len(go_starts)}")
```

iii. The agent validates that go cue count matches trial count as a sanity check, raising an error if they differ.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered in three stages:
1. Exclude `auto_water == 1` and `free_water == 1` trials.
2. Restrict to trials where all good units have valid ephys recording, using the intersection of `units/is_good_trials` (logical AND across all good units) and `units/obs_intervals` (mapping recording intervals back to trial indices).
3. Post-hoc removal of trials with all-zero neural activity across all neurons.

Early lick, ignore, and photostimulation trials are NOT excluded.

ii.
```python
trial_keep = np.ones(n_trials, dtype=bool)
if exclude_auto_free:
    trial_keep &= auto_water == 0
    trial_keep &= free_water == 0

# is_good_trials filtering
is_good_trials = np.asarray(units["is_good_trials"][good_unit_indices]).astype(bool)
# ... obs_intervals mapping logic ...
trial_keep &= recorded_trial_mask

# Post-hoc all-zero removal
nonzero_trial_mask = np.any(session_neural != 0, axis=(1, 2))
session_neural = session_neural[nonzero_trial_mask]
```

iii. The agent discovered all-zero neural trials during verification and traced the issue to trials extending beyond the valid ephys interval. It used `is_good_trials` and `obs_intervals` to properly restrict trials, then added a final all-zero check for 2 residual edge cases. The agent noted: "Early/ignore/photostim trials are retained because they are required by the requested decoder outputs and inputs."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `units/spike_times` (spike timestamps for each unit) and `units/spike_times_index` (index array for per-unit spike time boundaries). Only units where `units/classification == "good"` are included.

ii.
```python
classification = _decode_str_array(units["classification"]).reshape(-1)
good_unit_indices = np.flatnonzero(classification == "good")
# ...
def _extract_unit_spikes(units_group, unit_idx):
    spike_times = units_group["spike_times"]
    spike_index = units_group["spike_times_index"]
    end = int(spike_index[unit_idx])
    start = 0 if unit_idx == 0 else int(spike_index[unit_idx - 1])
    return np.asarray(spike_times[start:end], dtype=np.float64)
```

iii. The agent confirmed from the NWB structure that `classification` (not `unit_quality`) is the correct field for the post-QC classifier labels described in the spike sorting QC paper.

## 2-b. How is the `neural` data processed?

i. Spike times are binned into non-overlapping 50 ms bins aligned to go cue onset, spanning [-2.5 s, +1.5 s] (80 bins). Spike counts per bin are divided by bin width (0.05 s) to convert to firing rates in Hz.

ii.
```python
BIN_SIZE_S = 0.05
BIN_EDGES_REL = np.arange(WINDOW_START_S, WINDOW_END_S + 1e-9, BIN_SIZE_S, dtype=np.float64)
# ...
abs_edges = go_starts[trial_keep, None] + BIN_EDGES_REL[None, :]
for pos, unit_idx in enumerate(good_unit_indices):
    unit_spikes = _extract_unit_spikes(units, int(unit_idx))
    spike_bins = np.searchsorted(unit_spikes, abs_edges, side="left")
    spike_counts = np.diff(spike_bins, axis=1).astype(np.float32) / np.float32(BIN_SIZE_S)
    session_neural[:, pos, :] = spike_counts
```

iii. The agent used `np.searchsorted` on pre-sorted spike times against absolute bin edges for efficient binning, then divided by bin width to convert counts to rates.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `classification == "good"` are included. There is no additional filtering based on firing rate thresholds, ISI violations, or other quality metrics. Brain region labels come from `units/anno_name`.

ii.
```python
classification = _decode_str_array(units["classification"]).reshape(-1)
good_unit_indices = np.flatnonzero(classification == "good")
```

iii. The agent relied on the NWB classifier output as the sole unit quality filter, consistent with the spike sorting QC paper's approach of pre-classifying units.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to go cue onset. For each trial, absolute bin edges are computed as `go_cue_time + relative_bin_edges`, where relative bin edges span [-2.5, +1.5] s.

ii.
```python
abs_edges = go_starts[trial_keep, None] + BIN_EDGES_REL[None, :]
```

iii. This matches the instruction to "Temporally align based on Go cue onset" and "Extract 2.5 s before to 1.5 s after the go cue for each trial."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is 50 ms (0.05 s), producing 80 bins per trial over the 4 s window. No temporal rebinning is applied; spikes are directly binned at the target resolution.

ii.
```python
BIN_SIZE_S = 0.05
BIN_EDGES_REL = np.arange(WINDOW_START_S, WINDOW_END_S + 1e-9, BIN_SIZE_S, dtype=np.float64)
N_BINS = len(BIN_CENTERS_REL)  # = 80
```

iii. This directly matches the instruction to "Use 50-ms-width bins for computing firing rates."

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. Derived from `acquisition/BehavioralEvents/sample_start_times/timestamps` (sample/tone onset times) and `acquisition/BehavioralEvents/go_start_times/timestamps` (go cue times).

ii.
```python
sample_starts = np.asarray(be["sample_start_times"]["timestamps"], dtype=np.float64)
tone_onsets = _resolve_sample_onsets(trial_starts, go_starts, sample_starts)
```

iii. The agent identified `sample_start_times` as the tone onset events and noted that they can contain replayed epochs after early licks, requiring careful resolution.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. For each trial, tone onset is resolved as the last `sample_start_times` timestamp occurring before that trial's go cue, with a fallback constrained to the trial's start-to-go interval. The input is then computed as `bin_center_time - tone_onset_time` (time elapsed since tone onset at each bin center).

ii.
```python
def _resolve_sample_onsets(trial_starts, go_starts, sample_starts):
    sample_idx = np.searchsorted(sample_starts, go_starts, side="right") - 1
    valid = sample_idx >= 0
    sample_onsets = np.full(go_starts.shape, np.nan, dtype=np.float64)
    sample_onsets[valid] = sample_starts[sample_idx[valid]]
    invalid = np.isnan(sample_onsets) | (sample_onsets < (trial_starts - 1e-9))
    # fallback logic for edge cases...

# Per-trial computation:
tone_rel = kept_tone_onsets[local_idx] - go_starts[trial_idx]
time_from_tone = (BIN_CENTERS_REL - tone_rel).astype(np.float32)
```

iii. The agent noted: "In these NWB files, `sample_start_times` can contain replayed sample epochs after early licks, so naive one-to-one trial matching is wrong."

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. The time from tone onset is computed at each bin center of the neural time bins. Since both neural bins and the time-from-tone are computed relative to the go cue, they are inherently aligned.

ii.
```python
tone_rel = kept_tone_onsets[local_idx] - go_starts[trial_idx]
time_from_tone = (BIN_CENTERS_REL - tone_rel).astype(np.float32)
input_trial = np.vstack([time_from_tone, stim_on]).astype(np.float32)
```

iii. Both inputs share the same 80-bin temporal grid as the neural data.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. Derived from `intervals/trials/photostim_onset` and `intervals/trials/photostim_duration`. These are stored as strings ("N/A" for non-stim trials) and parsed to floats with NaN for missing values.

ii.
```python
photostim_onset = _parse_optional_float_array(trials["photostim_onset"])
photostim_duration = _parse_optional_float_array(trials["photostim_duration"])
```

iii. The agent used the trial-level photostimulation fields from the NWB trial table.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. `photostim_onset` is interpreted as relative to trial start time. The absolute photostim interval is `[trial_start + photostim_onset, trial_start + photostim_onset + photostim_duration)`. A binary indicator is set to 1 for each time bin whose center falls within this interval.

ii.
```python
if np.isfinite(kept_photostim_onset[local_idx]) and np.isfinite(kept_photostim_duration[local_idx]):
    stim_start_abs = kept_trial_starts[local_idx] + kept_photostim_onset[local_idx]
    stim_stop_abs = stim_start_abs + kept_photostim_duration[local_idx]
    stim_on = ((abs_centers[local_idx] >= stim_start_abs) & (abs_centers[local_idx] < stim_stop_abs)).astype(np.float32)
```

iii. The agent stated: "`photostim_onset` is interpreted relative to trial start, consistent with the NWB event timing."

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Photostimulation is evaluated at bin centers of the neural time grid, producing one binary value per neural time bin.

ii.
```python
stim_on = ((abs_centers[local_idx] >= stim_start_abs) & (abs_centers[local_idx] < stim_stop_abs)).astype(np.float32)
```

iii. Same temporal grid as neural data ensures alignment.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Derived from `intervals/trials/trial_instruction`, which indicates the instructed lick direction ("left" or "right").

ii.
```python
CHOICE_MAP = {"left": 0, "right": 1}
choice_value = CHOICE_MAP[str(kept_trial_instruction[local_idx])]
```

iii. The agent noted: "The reference code uses the paper's left/right trial label (`trial_type` in the original code path), which corresponds to `trial_instruction` in the NWB release."

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The string value is mapped to an integer: left=0, right=1. This per-trial value is replicated across all 80 time bins.

ii.
```python
np.full(N_BINS, choice_value, dtype=np.int16)
```

iii. The value is stored time-varying (replicated) to match the output format requirement that outputs should be time-varying "if at all possible."

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Derived from `intervals/trials/outcome`, which contains the trial outcome strings ("ignore", "miss", "hit").

ii.
```python
OUTCOME_MAP = {"ignore": 0, "miss": 1, "hit": 2}
outcome_value = OUTCOME_MAP[str(kept_outcomes[local_idx])]
```

iii. Direct mapping from NWB trial outcome field.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The string value is mapped to an integer: ignore=0, miss=1, hit=2. This per-trial value is replicated across all 80 time bins.

ii.
```python
np.full(N_BINS, outcome_value, dtype=np.int16)
```

iii. Same replication approach as choice.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Derived from `intervals/trials/early_lick`, which contains "early" or "no early".

ii.
```python
EARLY_LICK_MAP = {"no early": 0, "early": 1}
early_value = EARLY_LICK_MAP[str(kept_early[local_idx])]
```

iii. Direct mapping from NWB early_lick field.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The string value is mapped to an integer: "no early"=0, "early"=1. This per-trial value is replicated across all 80 time bins.

ii.
```python
np.full(N_BINS, early_value, dtype=np.int16)
```

iii. Same replication approach as other per-trial outputs.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. Derived from `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, specifically the y-coordinate (column index 1 of the data array) and its timestamps.

ii.
```python
tongue_group = f["acquisition/BehavioralTimeSeries"]["Camera0_side_TongueTracking"]
tongue_timestamps = np.asarray(tongue_group["timestamps"], dtype=np.float64)
tongue_y_all = np.asarray(tongue_group["data"][:, 1], dtype=np.float32)
```

iii. The agent identified the side-camera tongue tracking data as the source for tongue y-position.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. For each 50 ms neural bin, the tongue y-value is sampled as the last camera frame whose timestamp falls within that bin (sample-and-hold/most-recent-sample approach). This is done by `_bin_tongue_y` using `np.searchsorted`.

ii.
```python
def _bin_tongue_y(timestamps, y_values, go_time):
    bin_starts = go_time + BIN_EDGES_REL[:-1]
    bin_ends = go_time + BIN_EDGES_REL[1:]
    idx = np.searchsorted(timestamps, bin_ends, side="left") - 1
    valid = idx >= 0
    if np.any(valid):
        valid_idx = idx[valid]
        valid[valid] &= timestamps[valid_idx] >= bin_starts[valid]
    sampled = np.zeros(N_BINS, dtype=np.float32)
    if np.any(valid):
        sampled[valid] = y_values[idx[valid]].astype(np.float32)
    return sampled
```

iii. The agent stated: "The reference alignment code uses the most recent marker sample within each time window rather than interpolation."

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Per-session discretization uses the 40th and 60th percentiles of the full session's tongue-y distribution (all camera frames, not just kept trials):
- 0: value < 40th percentile (or exactly 0/no data)
- 1: value >= 40th percentile and <= 60th percentile
- 2: value > 60th percentile

ii.
```python
tongue_q40, tongue_q60 = np.percentile(tongue_y_all, [40, 60])
# ...
tongue_cat = np.zeros(N_BINS, dtype=np.int16)
tongue_cat[tongue_y_trial >= tongue_q40] = 1
tongue_cat[tongue_y_trial > tongue_q60] = 2
```

iii. The agent computed percentiles from the full session's tongue-y distribution as specified in the instructions. The boundary conditions use `>=` for the 40th percentile and `>` for the 60th percentile.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Tongue y-position is sampled at the neural time bin grid (go-cue-aligned 50 ms bins), making it inherently aligned with neural data.

ii.
```python
tongue_y_trial = _bin_tongue_y(tongue_timestamps, tongue_y_all, go_starts[trial_idx])
```

iii. The `_bin_tongue_y` function uses the same go-cue-aligned bin edges as the neural data.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Several types of missing/problematic data are handled:
- **Sessions with zero good units:** Dropped (1 session).
- **Trials outside ephys recording range:** Filtered via `is_good_trials` and `obs_intervals`.
- **Residual all-zero neural trials:** Pruned post-hoc (2 trials removed).
- **Missing photostimulation data:** "N/A" values parsed as NaN; result in all-zero photostim vectors.
- **Missing tongue data:** Bins with no camera sample default to 0.
- **Sessions with < 2 trials:** Would be skipped (none triggered).

ii.
```python
# Missing photostim
if value in ("N/A", "", None):
    continue  # stays NaN

# All-zero neural
nonzero_trial_mask = np.any(session_neural != 0, axis=(1, 2))
session_neural = session_neural[nonzero_trial_mask]

# Missing tongue data
sampled = np.zeros(N_BINS, dtype=np.float32)  # default zero
```

iii. The agent documented these edge cases in CONVERSION_NOTES.md and iteratively fixed them after verification exposed issues.

## 10-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is the spike binning loop, which iterates over every good unit in every session and performs `np.searchsorted` on the full spike train against the trial-by-bin edge matrix. Loading the full tongue tracking data and computing session-wide percentiles for each session is also expensive due to large array reads.

ii.
```python
for pos, unit_idx in enumerate(good_unit_indices):
    unit_spikes = _extract_unit_spikes(units, int(unit_idx))
    spike_bins = np.searchsorted(unit_spikes, abs_edges, side="left")
    spike_counts = np.diff(spike_bins, axis=1).astype(np.float32) / np.float32(BIN_SIZE_S)
    session_neural[:, pos, :] = spike_counts
```

iii. The agent did not explicitly discuss time-consuming steps in the trajectory but chose vectorized operations where possible.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Two main loops could potentially be vectorized:
1. The per-unit spike binning loop (line 260-264) iterates over every good unit individually. This could potentially be parallelized or batched.
2. The per-trial input/output computation loop (line 274-304) constructs each trial's inputs/outputs individually.

ii.
```python
# Per-unit loop
for pos, unit_idx in enumerate(good_unit_indices):
    unit_spikes = _extract_unit_spikes(units, int(unit_idx))
    # ...

# Per-trial loop
for local_idx, trial_idx in enumerate(keep_trial_indices):
    tone_rel = kept_tone_onsets[local_idx] - go_starts[trial_idx]
    time_from_tone = (BIN_CENTERS_REL - tone_rel).astype(np.float32)
    # ...
```

iii. The per-unit loop is difficult to vectorize because each unit has a different number of spikes (ragged arrays). The per-trial loop could partially be vectorized for the input computations.

## 10-c. What processing does the code repeat multiple times?

i. The code does not significantly repeat processing. Each session is processed once. The `_resolve_sample_onsets` function includes a fallback path that re-searches for sample onsets, but this only triggers for edge cases (invalid initial matches). The `_bin_tongue_y` function is called once per trial, each time working with the full session's tongue data.

ii.
```python
# _bin_tongue_y called per trial with full session data
for local_idx, trial_idx in enumerate(keep_trial_indices):
    tongue_y_trial = _bin_tongue_y(tongue_timestamps, tongue_y_all, go_starts[trial_idx])
```

iii. The agent did not discuss repeated processing.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several pieces of unnecessary processing exist:
1. `photostim_power` is loaded but never used in the conversion.
2. Per-trial outputs (choice, outcome, early_lick) are replicated across all 80 time bins, creating 80x more data than needed for per-trial variables. The decoder likely collapses these back to per-trial values.
3. The `tongue_q40` and `tongue_q60` percentiles are computed from the entire session's tongue data, including time periods outside the trial windows.
4. Summary statistics are computed per-session (e.g., `mean_trial_duration_s`, `mean_tone_onset_rel_go_s`) and stored in a JSON file, but are not part of the converted dataset.

ii.
```python
photostim_power = _parse_optional_float_array(trials["photostim_power"])  # never used

# Per-trial values replicated across bins
np.full(N_BINS, choice_value, dtype=np.int16)
np.full(N_BINS, outcome_value, dtype=np.int16)
np.full(N_BINS, early_value, dtype=np.int16)
```

iii. The agent did not discuss unnecessary processing in the trajectory.
