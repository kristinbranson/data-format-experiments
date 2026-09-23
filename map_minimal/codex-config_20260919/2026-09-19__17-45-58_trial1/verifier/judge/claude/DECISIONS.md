# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The dataset is distributed as one NWB file per session under `data/sub-<subject_id>/`. All sessions are found by globbing `sub-*/*.nwb`, sorted alphabetically. Each file is opened with `h5py` (not `pynwb`) and fields are read directly from the HDF5 structure: `units/classification`, `units/spike_times`, `units/is_good_trials`, `intervals/trials/*`, `acquisition/BehavioralEvents/*`, and `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/*`.

ii.
```python
paths = sorted(data_dir.glob("sub-*/*.nwb"))
...
for index, path in enumerate(paths, start=1):
    session = _convert_session(path)
```

```python
with h5py.File(path, "r") as nwb:
    classification = _decode_array(nwb["units/classification"])
    ...
    all_go_times = nwb["acquisition/BehavioralEvents/go_start_times/timestamps"][:]
```

iii. The agent initially explored NWB files with pynwb to understand the structure, but used h5py in the final script for direct HDF5 access. The agent confirmed 174 files across 28 subjects from scanning all NWB files.

## 1-b. How are the data split into subjects?

i. The subject ID is extracted from the directory name of each NWB file by stripping the `sub-` prefix (e.g., `sub-440956` becomes `440956`). A sorted unique list of subject IDs is assembled, and `subject_idx` maps each session to its position in that list.

ii.
```python
subject_id = path.parent.name.removeprefix("sub-")
```

```python
subjects = sorted({session["subject"] for session in converted_sessions})
subject_lookup = {name: i for i, name in enumerate(subjects)}
```

iii. The agent used the directory name rather than `nwb.subject.subject_id` (which gives the same numeric ID). The agent confirmed 28 unique subjects.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session, so no additional splitting is needed. The session order follows the sorted glob of file paths. Sessions with no good units are skipped entirely.

ii.
```python
paths = sorted(data_dir.glob("sub-*/*.nwb"))
...
session = _convert_session(path)
if session is None:
    skipped_sessions.append(path.name)
```

iii. The agent confirmed that one session had no classifier-labeled good units and was skipped, leaving 173 sessions.

## 1-d. How are the data split into trials?

i. Trials come from the NWB trials table (`intervals/trials`). The number of go-cue events (`go_start_times/timestamps`) is verified to match the number of trial rows. Trials are then filtered through multiple quality-control steps before being retained.

ii.
```python
all_go_times = nwb["acquisition/BehavioralEvents/go_start_times/timestamps"][:].astype(np.float64)
n_source_trials = len(nwb["intervals/trials/id"])
if len(all_go_times) != n_source_trials:
    raise ValueError(...)
```

iii. The agent verified that go-cue counts match trial counts in all 174 sessions, establishing a one-to-one mapping between trials and go cues.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered using four criteria:
1. **is_good_trials intersection**: Only trials where *all* retained good units have `is_good_trials == True` are kept. This uses the intersection across all good units.
2. **obs_intervals mapping**: For 8 sessions where the behavioral table extends beyond ephys acquisition, `obs_intervals` is used to map the compact `is_good_trials` mask back to source trial indices.
3. **auto_water and free_water exclusion**: Both auto-water and free-water trials are excluded, following the source repository's `get_regular_trial_mask`.
4. **Population-wide zero-spike dropout**: Trials where all units have zero spikes are excluded as acquisition dropouts.

ii.
```python
good_trial_matrix = nwb["units/is_good_trials"][good_rows, :]
...
common_good_mask = np.all(good_trial_matrix, axis=0)
common_good_idx = ephys_source_idx[common_good_mask]
auto_water_all = nwb["intervals/trials/auto_water"][:].astype(bool)
free_water_all = nwb["intervals/trials/free_water"][:].astype(bool)
water_mask = auto_water_all[common_good_idx] | free_water_all[common_good_idx]
valid_trial_idx = common_good_idx[~water_mask]
...
population_recorded = np.any(rates != 0, axis=(1, 2))
if n_population_dropouts:
    rates = rates[population_recorded]
```

iii. The agent's reasoning evolved over multiple iterations. It initially kept all trials, then added `is_good_trials` filtering after observing that some trials had no neural coverage. It added auto_water/free_water exclusion after finding the source repository's `get_regular_trial_mask` function. The population dropout filter was added last after finding one trial with zero spikes from all clusters despite being marked valid.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `units/spike_times` (the flat spike time array) and `units/spike_times_index` (the per-unit end offsets into that array). Only units with `classification == 'good'` contribute. Go-cue times from `BehavioralEvents/go_start_times` define the trial windows.

ii.
```python
all_spikes = nwb["units/spike_times"][:]
spike_ends = nwb["units/spike_times_index"][:]
rates = _bin_good_units(all_spikes, spike_ends, good_rows, go_times)
```

iii. The agent noted that spike_times is the only neural representation in the NWB files and that firing rates must be computed from raw spike times.

## 2-b. How is the `neural` data processed?

i. Spike times are binned into 80 non-overlapping 50 ms bins spanning -2.5 s to +1.5 s relative to the go cue, then divided by bin width (0.05 s) to produce firing rates in Hz. For each good unit, spikes are assigned to trials via `searchsorted` on window start times, then to time bins via floor division. A small epsilon (1e-10) stabilizes decimal timestamps near bin edges. `np.bincount` on flat (trial * N_BINS + time_bin) indices gives counts.

ii.
```python
def _bin_good_units(all_spikes, spike_ends, good_rows, go_times):
    ...
    window_starts = go_times + OFF_START
    window_stops = go_times + OFF_END
    for out_unit, unit_row in enumerate(good_rows):
        ...
        trial = np.searchsorted(window_starts, spikes, side="right") - 1
        ...
        time_bin = np.floor(
            (spikes - window_starts[trial]) / BIN_SIZE_S + 1e-10
        ).astype(np.int64)
        ...
        flat_bin = trial[valid] * N_BINS + time_bin[valid]
        counts = np.bincount(flat_bin, minlength=n_trials * N_BINS)
        rates[:, out_unit, :] = counts.reshape(n_trials, N_BINS) / BIN_SIZE_S
    return rates
```

iii. The agent noted that bins are left-closed, right-open, producing 80 bins centered from -2.475 to +1.475 s.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `units/classification == 'good'` are kept. No additional quality metrics or thresholds are applied. Sessions with no good units are skipped. The agent confirmed this retains 69,453 units across 173 sessions.

ii.
```python
classification = _decode_array(nwb["units/classification"])
good_rows = np.flatnonzero(classification == "good")
if good_rows.size == 0:
    return None
```

iii. The agent confirmed from the source code and QC paper that `classification == 'good'` corresponds to the published region-specific QC classifier labels. It also noted that good units are the ones with anatomical `anno_name` values.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. All NWB timestamps are on a common session-absolute clock. Bin edges are defined relative to the go cue: `go_times + OFF_START` to `go_times + OFF_END`. Spikes are assigned to trials and bins by their position relative to these window boundaries. No interpolation or offset correction is needed.

ii.
```python
window_starts = go_times + OFF_START
window_stops = go_times + OFF_END
...
trial = np.searchsorted(window_starts, spikes, side="right") - 1
...
time_bin = np.floor(
    (spikes - window_starts[trial]) / BIN_SIZE_S + 1e-10
).astype(np.int64)
```

iii. The agent confirmed that all timestamps share a global clock, so alignment to the go cue only requires computing window boundaries from go-cue times.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The bin width is 50 ms (0.05 s), producing 80 non-overlapping bins spanning -2.5 s to +1.5 s relative to the go cue. Bin edges are defined once as 81 values and reused for every trial and session.

ii.
```python
OFF_START = -2.5
OFF_END = 1.5
BIN_SIZE_S = 0.05
N_BINS = int(round((OFF_END - OFF_START) / BIN_SIZE_S))
BIN_EDGES = OFF_START + np.arange(N_BINS + 1, dtype=np.float64) * BIN_SIZE_S
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2
```

iii. The 50 ms bin width and -2.5 to +1.5 s window are directly specified in the instructions. No rebinning is applied since spikes are binned from raw spike times.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. From `sample_start_times/timestamps` (tone onset events) and go-cue times. The last sample onset before each go cue is taken as the trial's tone onset, since early licks can cause the sample epoch to replay.

ii.
```python
sample_starts = nwb["acquisition/BehavioralEvents/sample_start_times/timestamps"][:]
tone_onset = _tone_onsets(go_times, sample_starts)
```

```python
def _tone_onsets(go_times, sample_starts):
    idx = np.searchsorted(sample_starts, go_times, side="right") - 1
    return sample_starts[idx]
```

iii. The agent noted that replayed sample epochs produce extra tone onsets, so the last one before each go cue is the relevant one.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. For each trial, time from tone onset at each bin center is computed as `go_time + bin_center - tone_onset`. This gives a continuous, time-varying signal in seconds.

ii.
```python
time_from_tone = (
    go_times[trial] + BIN_CENTERS - tone_onset[trial]
).astype(np.float32)
```

iii. Since bin centers are defined relative to the go cue, adding the go-to-tone gap converts them to time from tone onset.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. The same bin centers used for neural data are used to compute time from tone onset, so alignment is inherent in the shared time grid. Both use `BIN_CENTERS` relative to the go cue.

ii.
```python
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2
...
time_from_tone = (go_times[trial] + BIN_CENTERS - tone_onset[trial]).astype(np.float32)
```

iii. The shared bin-center grid ensures time from tone onset and neural firing rates are aligned at the same timepoints.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. From `intervals/trials/photostim_onset` and `intervals/trials/photostim_duration` in the trials table, with `start_time` used to convert to absolute times. Non-stimulated trials have `'N/A'` for onset/duration.

ii.
```python
stim_onset_raw = nwb["intervals/trials/photostim_onset"][:][valid_trial_idx]
stim_duration_raw = nwb["intervals/trials/photostim_duration"][:][valid_trial_idx]
stim_onset = np.asarray([_optional_float(x) for x in stim_onset_raw])
stim_duration = np.asarray([_optional_float(x) for x in stim_duration_raw])
```

iii. The agent verified that `photostim_onset` is stored as a string relative to trial start by checking timestamps against known values.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. The photostim onset is converted to an absolute time by adding it to the trial start time. A binary (0/1) time series is produced: a bin center is 1 if it falls between the absolute onset and offset of stimulation, 0 otherwise. Non-stimulated trials (NaN onset/duration) remain all zeros.

ii.
```python
if np.isfinite(stim_onset[trial]) and np.isfinite(stim_duration[trial]):
    photo_start = trial_start[trial] + stim_onset[trial]
    photo_stop = photo_start + stim_duration[trial]
    absolute_centers = go_times[trial] + BIN_CENTERS
    photo_on[(absolute_centers >= photo_start) & (absolute_centers < photo_stop)] = 1.0
```

iii. The agent explicitly checks for finiteness to handle non-stimulated trials. The comparison uses absolute times (bin centers converted to session-absolute time) against absolute stim onset/offset.

## 4-c. How is `input` *Photostimulation* aligned with the neural data?

i. Bin centers are converted to absolute times (`go_times[trial] + BIN_CENTERS`) and compared against the absolute photostim onset/offset, ensuring alignment with the same time grid used for neural data.

ii.
```python
absolute_centers = go_times[trial] + BIN_CENTERS
photo_on[(absolute_centers >= photo_start) & (absolute_centers < photo_stop)] = 1.0
```

iii. Using absolute times for comparison ensures consistency with the go-cue-aligned neural bins.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. From `intervals/trials/trial_instruction` (left/right) and `intervals/trials/outcome` (hit/miss/ignore). Choice is not stored directly.

ii.
```python
instruction = _decode_array(nwb["intervals/trials/trial_instruction"])[valid_trial_idx]
outcome_text = _decode_array(nwb["intervals/trials/outcome"])[valid_trial_idx]
```

iii. The agent's docstring states: "Define choice from the authoritative instruction/outcome trial labels: a hit has the instructed choice, a miss the opposite choice, and an ignored trial has no lick. This avoids rare missing/mis-timestamped lick events in the NWB."

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Choice is derived as: hit = instructed side (left=0, right=1), miss = opposite side, ignore = no lick (2). The value is per-trial, broadcast across all 80 time bins.

ii.
```python
def _trial_choice(instruction, outcome):
    if outcome == "ignore":
        return 2
    if outcome == "hit":
        return 0 if instruction == "left" else 1
    if outcome == "miss":
        return 1 if instruction == "left" else 0

trial_output[0, :] = _trial_choice(instruction[trial], outcome_text[trial])
```

iii. Output values are `["left", "right", "no lick"]`.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Directly from the `outcome` column of the trials table, which contains strings `'ignore'`, `'miss'`, and `'hit'`.

ii.
```python
outcome_text = _decode_array(nwb["intervals/trials/outcome"])[valid_trial_idx]
```

iii. The trials table stores outcome explicitly with the three required categories.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The three strings are mapped to integers: ignore=0, miss=1, hit=2. The value is per-trial, broadcast across all 80 time bins.

ii.
```python
outcome_map = {"ignore": 0, "miss": 1, "hit": 2}
trial_output[1, :] = outcome_map[outcome_text[trial]]
```

iii. Output values are `["ignore", "miss", "hit"]`.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. From the `early_lick` column of the trials table, which contains strings `'no early'` and `'early'`.

ii.
```python
early_text = _decode_array(nwb["intervals/trials/early_lick"])[valid_trial_idx]
```

iii. The trials table flags early licking explicitly.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Binary encoding: `'early'` maps to 1, everything else to 0. The value is per-trial, broadcast across all 80 bins.

ii.
```python
trial_output[2, :] = 1 if early_text[trial] == "early" else 0
```

iii. Output values are `["no", "yes"]`.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, which contains `(n_frames, 3)` data: tongue_x, tongue_y, tongue_likelihood, with matching timestamps.

ii.
```python
tongue_path = "acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking"
tongue_data = nwb[f"{tongue_path}/data"][:]
camera_times = nwb[f"{tongue_path}/timestamps"][:]
```

iii. This is the only tongue measurement in the NWB files.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Frames with DLC likelihood < 0.9 are marked as not visible. Session-level 40th and 60th percentiles are computed over all visible raw frames (not bin means). For each trial, tongue y is sampled at each neural bin center using the preceding camera frame (via `searchsorted` with `side="right"` minus 1). Frames older than 20 ms are treated as gaps. Categories: 0 = below 40th pct, 1 = 40th-60th pct, 2 = above 60th pct, 3 = not visible.

ii.
```python
TONGUE_LIKELIHOOD_THRESHOLD = 0.9
...
visible_session = np.isfinite(y) & np.isfinite(likelihood) & (likelihood >= TONGUE_LIKELIHOOD_THRESHOLD)
q40, q60 = np.percentile(y[visible_session], [40, 60])

target_times = go_times[:, None] + BIN_CENTERS[None, :]
frame = np.searchsorted(camera_times, target_times, side="right") - 1
valid_frame &= (target_times - camera_times[frame]) <= 0.020
...
categories[visible & (sampled_y < q40)] = 0
categories[visible & (sampled_y >= q40) & (sampled_y <= q60)] = 1
categories[visible & (sampled_y > q60)] = 2
```

iii. The agent used a 0.9 likelihood threshold (standard DLC convention for high-confidence detections), computed percentiles on all visible frames across the session, and sampled at bin centers using the preceding camera frame with a 20ms staleness guard.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Using session-level 40th and 60th percentiles of visible frames' y-positions:
- 0: y < 40th percentile
- 1: 40th percentile <= y <= 60th percentile
- 2: y > 60th percentile
- 3: not visible (likelihood < 0.9 or camera gap > 20ms)

ii.
```python
categories[visible & (sampled_y < q40)] = 0
categories[visible & (sampled_y >= q40) & (sampled_y <= q60)] = 1
categories[visible & (sampled_y > q60)] = 2
```

iii. The boundary conditions differ slightly from the reference: category 1 uses `>=` and `<=` (inclusive on both ends), while the reference uses strict inequalities at the edges.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Tongue y is sampled at each neural bin center time (`go_times + BIN_CENTERS`), ensuring the tongue output has the same 80 timepoints as the neural data. The preceding camera frame is used for each bin center, with a 20ms freshness check.

ii.
```python
target_times = go_times[:, None] + BIN_CENTERS[None, :]
frame = np.searchsorted(camera_times, target_times, side="right") - 1
valid_frame &= (target_times - camera_times[frame]) <= 0.020
```

iii. The agent chose to sample at bin centers (point sampling) rather than averaging frames within each bin.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases:
- **Sessions with no good units**: Skipped entirely (returned `None`).
- **Trials outside neural coverage**: Filtered via `is_good_trials` intersection across all good units, with `obs_intervals` mapping for sessions where behavioral table extends beyond ephys.
- **Population-wide zero spikes**: Excluded as acquisition dropouts.
- **Free-water and auto-water trials**: Excluded (no spikes and not valid trial types).
- **Tongue not visible**: Assigned category 3. Camera gaps > 20ms also marked not visible.
- **NaN photostim values**: Treated as no stimulation (all zeros).
- **NaN classification values**: Decoded as empty string, so not matched as 'good'.

ii.
```python
if good_rows.size == 0:
    return None
...
population_recorded = np.any(rates != 0, axis=(1, 2))
...
valid_frame &= (target_times - camera_times[frame]) <= 0.020
```

iii. The agent progressively discovered and addressed edge cases through multiple iterations of testing and debugging.

## 10-a. What are the most time-consuming steps of the code?

i. Reading NWB files dominates, particularly loading the `spike_times` buffer and tongue tracking arrays. The full conversion of 174 sessions completed in approximately 1.5 minutes. Per-unit spike binning with `searchsorted` and `bincount` is the main computational step within each session.

ii. N/A

iii. The agent checked memory (1 TiB total) and disk space availability before running, and the conversion completed efficiently.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-unit loop in `_bin_good_units` iterates over each good unit, performing vectorized spike-to-trial and spike-to-bin assignments within each unit. The per-trial loop in the output construction iterates over trials to build input/output arrays.

ii.
```python
for out_unit, unit_row in enumerate(good_rows):
    ...
    trial = np.searchsorted(window_starts, spikes, side="right") - 1
    ...
    counts = np.bincount(flat_bin, minlength=n_trials * N_BINS)
```

```python
for trial in range(n_trials):
    time_from_tone = (go_times[trial] + BIN_CENTERS - tone_onset[trial]).astype(np.float32)
    ...
```

iii. The per-unit loop is inherent to the ragged spike storage. The per-trial input/output loop could be vectorized but is not a performance bottleneck compared to I/O and spike binning.

## 10-c. What processing does the code repeat multiple times?

i. No significant repeated processing. Each NWB file is opened once, and derived quantities are computed once per session. The bin grid is defined once at module level.

ii. N/A

iii. The conversion is a single pass over the files.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code stores detailed session metadata (source file, trial counts at various filtering stages, tongue percentile values) that are informational but not used by the decoder. The per-trial loop for constructing inputs/outputs could be viewed as doing unnecessary per-trial work when vectorized operations would suffice, but no data is computed and then discarded.

ii.
```python
"session_info": {
    "source_file": str(path.relative_to(path.parents[1])),
    "n_source_trials": int(n_source_trials),
    "n_ephys_mask_trials": int(n_ephys_trials),
    ...
}
```

iii. The extra metadata is retained for debugging and transparency but has no downstream use.
