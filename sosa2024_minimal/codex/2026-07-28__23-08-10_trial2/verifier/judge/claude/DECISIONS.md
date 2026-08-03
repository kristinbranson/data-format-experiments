# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads all NWB files by globbing `sub-*/sub-*_behavior+ophys.nwb` under the data root directory using `h5py` (not `pynwb`). It reads subject IDs from the HDF5 metadata (`general/subject/subject_id`), session numbers from filenames, and scene names from the NWB `identifier` field. All 152 NWB files across 11 subjects are processed.

ii.
```python
def list_nwb_files(data_root: Path) -> list[SessionMeta]:
    session_meta: list[SessionMeta] = []
    for file_path in sorted(data_root.glob("sub-*/sub-*_behavior+ophys.nwb")):
        with h5py.File(file_path, "r") as f:
            subject = decode_scalar(f["general/subject/subject_id"][()])
            identifier = decode_scalar(f["identifier"][()])
        parts = identifier.strip("/").split("/")
        date = parts[-2]
        scene = parts[-1]
        session_num = int(re.search(r"ses-(\d+)", file_path.name).group(1))
        session_meta.append(SessionMeta(...))
    session_meta.sort(key=lambda s: (int(s.subject[1:]), s.session_num))
    return session_meta
```

iii. The AI documented that the `data/` directory contains 152 NWB sessions (not 154), with 2 missing sessions for m11. The AI uses `h5py` directly rather than `pynwb` for loading.

## 1-b. How are the data split into subjects?

i. Subjects are identified from the HDF5 metadata field `general/subject/subject_id` within each NWB file. Unique subjects are collected and sorted numerically.

ii.
```python
subjects = sorted({meta.subject for meta in session_meta}, key=lambda s: int(s[1:]))
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
```

iii. Subject IDs are read directly from the NWB metadata rather than parsed from directory names.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. Session numbers are parsed from the filename pattern `ses-(\d+)`.

ii.
```python
session_num = int(re.search(r"ses-(\d+)", file_path.name).group(1))
```

iii. Each NWB file is a single session recording.

## 1-d. How are the data split into trials?

i. Trial boundaries are defined by `trial_start` (>0 marks start) and `teleport` (>0 marks end). The kept range for each trial is `[trial_start_idx : teleport_idx)`, excluding the teleport sample.

ii.
```python
trial_start_idx = np.where(trial_start_series > 0)[0]
trial_stop_idx = np.where(teleport_series > 0)[0]
...
start = trial_start_idx[trial_idx]
stop = trial_stop_idx[trial_idx]
```

iii. This matches the paper's trial structure. The AI's CONVERSION_NOTES state that trials are aligned to trial start and the teleport sample is excluded.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered using a lick-sensor error criterion: a trial is dropped if more than 30% of its frame samples have `lick count > 2`. This resulted in 81 dropped trials out of 12216.

ii.
```python
def bad_lick_trial_mask(lick_counts, start_idx, stop_idx):
    mask = np.zeros(len(start_idx), dtype=bool)
    for trial_idx, (start, stop) in enumerate(zip(start_idx, stop_idx)):
        if np.mean(lick_counts[start:stop] > 2.0) > 0.3:
            mask[trial_idx] = True
    return mask
...
lick_error_mask = bad_lick_trial_mask(lick_counts, trial_start_idx, trial_stop_idx)
kept_trial_indices = np.where(~lick_error_mask)[0]
```

iii. The AI states this matches the paper's lick-sensor error criterion described in the methods.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `Fluorescence` and `Neuropil` data in the NWB ophys processing module, NOT from the pre-computed `Deconvolved` field. The AI recomputes the full neural processing pipeline from raw fluorescence.

ii.
```python
fluorescence_ds = ophys["Fluorescence"][plane_key]["data"]
neuropil_ds = ophys["Neuropil"][plane_key]["data"]
...
fluorescence = fluorescence_ds[:session_len, block_local].T.astype(np.float32)
neuropil = neuropil_ds[:session_len, block_local].T.astype(np.float32)
dff, events = compute_dff_and_events(fluorescence, neuropil, ...)
```

iii. The AI's CONVERSION_NOTES explain: "The saved neural activity is trial-wise OASIS event activity recomputed from Suite2p fluorescence and neuropil, rather than taking the NWB Deconvolved array directly." The paper repo's preprocessing logic was treated as the operational reference.

## 2-b. How is the `neural` data processed?

i. The AI implements a multi-step neural processing pipeline:
1. Keep only curated Suite2p ROIs (`iscell[:, 0] == 1`)
2. Neuropil subtraction with coefficient 0.7
3. Per-trial maximin dF/F baseline (Gaussian smooth sigma=15, min filter 300, max filter 300)
4. Compute dF/F = (F - baseline) / abs(baseline)
5. Smooth dF/F with Gaussian sigma=2
6. OASIS deconvolution with tau=0.7

ii.
```python
def compute_dff_and_events(fluorescence, neuropil, start_idx, stop_idx, frame_rate_hz,
                           neu_coef=0.7, tau=0.7):
    f -= neu_coef * f_neu
    ...
    baseline[:, sl] = nansmooth_2d(f[:, sl], sigma=15)
    baseline[:, sl] = minimum_filter1d(baseline[:, sl], size=300, axis=1, mode="nearest")
    baseline[:, sl] = maximum_filter1d(baseline[:, sl], size=300, axis=1, mode="nearest")
    dff[:, valid] = (f[:, valid] - baseline[:, valid]) / np.abs(baseline[:, valid])
    ...
    dff[:, sl] = nansmooth_2d(dff[:, sl], sigma=2)
    events[:, sl] = dcnv.oasis(dff[:, sl], 2000, tau, frame_rate_hz)
    return dff, events
```

iii. The AI followed the paper repo's preprocessing logic (suite2p-style processing) as the operational reference for neural signal processing.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two-stage filtering: (1) `iscell[:, 0] == 1` to keep curated Suite2p ROIs, and (2) putative interneurons excluded when Pearson correlation between dF/F and running speed exceeds 0.5.

ii.
```python
plane_iscell = iscell[plane_slice, 0] == 1
...
block_corr = np.array([
    np.corrcoef(trace, speed_valid)[0, 1]
    if np.nanstd(trace) > 0 and np.nanstd(speed_valid) > 0
    else np.nan
    for trace in dff_valid
])
block_interneuron = np.isfinite(block_corr) & (block_corr > 0.5)
keep_block = ~block_interneuron
```

iii. The AI states this follows the paper's description of excluding speed-correlated putative interneurons. The interneuron exclusion fraction was 0.360% +/- 0.615%.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to trial start. Each trial's neural data spans from `trial_start_idx` to `teleport_idx` (exclusive). No additional temporal shifting is needed since the alignment event IS the trial start.

ii.
```python
start = trial_start_idx[trial_idx]
stop = trial_stop_idx[trial_idx]
kept_events[:, start:stop]
```

iii. The instructions specify alignment to trial start, which is the natural start of the data slice.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No temporal rebinning is applied. The time bin size is derived from the median inter-frame interval of the behavior timestamps, giving approximately 64.48 ms (~15.51 Hz). This is stored in `metadata["time_bin_size"]` as the mean across sessions.

ii.
```python
def infer_frame_rate(position_timestamps):
    diffs = np.diff(position_timestamps.astype(np.float64))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    return float(1.0 / np.median(diffs))
...
mean_frame_rate = float(np.mean([stats["frame_rate_hz"] for stats in session_stats]))
dataset["metadata"]["time_bin_size"] = 1000.0 / mean_frame_rate
```

iii. Frame rate is inferred from timestamps rather than NWB metadata, which matters for multi-plane sessions.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Derived from the `position` behavior timestamps (via `behavior["position"]["timestamps"]`).

ii.
```python
timestamps = behavior["position"]["timestamps"][:].astype(np.float64)
...
trial_times = (timestamps[start:stop] - timestamps[start]).astype(np.float32)
```

iii. The behavior timestamps are used directly.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The timestamp at trial start is subtracted from all timestamps within the trial to get time relative to trial start.

ii.
```python
trial_times = (timestamps[start:stop] - timestamps[start]).astype(np.float32)
```

iii. Simple subtraction of the first timestamp in the trial.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Neural and behavioral data share the same time indexing (same frame indices), so no additional alignment is needed.

ii.
```python
start = trial_start_idx[trial_idx]
stop = trial_stop_idx[trial_idx]
trial_times = (timestamps[start:stop] - timestamps[start]).astype(np.float32)
# Neural data uses same start:stop indices
kept_events[:, start:stop]
```

iii. Both neural and behavioral data are indexed by the same frame indices.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Derived from the session scene name (parsed from NWB `identifier` field), NOT from the `environment` behavior time series. The `environment` stream is used only as a cross-check.

ii.
```python
scene = parts[-1]  # e.g., "Env1_LocationA"
trial_labels = parse_scene(session_meta.scene, len(trial_start_idx))
...
np.full(trial_times.shape, float(label.env), dtype=np.float32)
```

iii. Scene-derived environment was cross-checked against the behavior `environment` stream on every kept trial.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The scene name is parsed using regex patterns to extract the environment number. `Env1` maps to 0, `Env2` maps to 1. For switch sessions (e.g., `Env1_A_to_Env2_B`), the environment changes at trial 30.

ii.
```python
def parse_scene(scene, n_trials, switch_trial_count=30):
    match = SCENE_LOCATION_RE.match(scene)
    if match:
        env = int(match.group("env")) - 1
        ...
    match = SCENE_ENV_SWITCH_RE.match(scene)
    if match:
        env0 = int(match.group("env0")) - 1
        env1 = int(match.group("env1")) - 1
        return [TrialLabel(env=env0 if trial_idx < switch_trial_count else env1, ...)
                for trial_idx in range(n_trials)]
```

iii. The switch point of trial 30 matches the paper/task structure.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Derived from the NWB `trial number` behavior time series, sampled at the trial start frame.

ii.
```python
trial_number_series = behavior["trial number"]["data"][:].astype(np.float32)
...
np.full(trial_times.shape, trial_number_series[start], dtype=np.float32)
```

iii. The AI's CONVERSION_NOTES state: "`trial_number` is copied from the aligned NWB `trial number` stream at trial start."

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The value of `trial number` at the trial start frame index is taken and broadcast across all timepoints in the trial.

ii.
```python
np.full(trial_times.shape, trial_number_series[start], dtype=np.float32)
```

iii. No additional processing; it takes the stored trial number value at trial start.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from the `Reward` behavior timestamps and trial boundary timestamps.

ii.
```python
reward_timestamps = behavior["Reward"]["timestamps"][:].astype(np.float64)
...
reward_outcomes = reward_outcome_per_trial(reward_timestamps, timestamps[trial_start_idx], timestamps[trial_stop_idx])
prev_reward_outcomes = np.concatenate([[0], reward_outcomes[:-1]]).astype(np.float32)
```

iii. Reward outcomes are computed per-trial from reward event timestamps, then shifted by one trial.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For each trial, check if any reward timestamp falls within the trial's time window (between trial start and teleport times). The outcomes array is then shifted by one position (prepended with 0 for the first trial) to get previous trial outcome.

ii.
```python
def reward_outcome_per_trial(reward_timestamps, start_times, stop_times):
    outcomes = np.zeros(len(start_times), dtype=np.int8)
    for trial_idx, (start_time, stop_time) in enumerate(zip(start_times, stop_times)):
        while reward_idx < len(reward_timestamps) and reward_timestamps[reward_idx] < start_time:
            reward_idx += 1
        probe_idx = reward_idx
        while probe_idx < len(reward_timestamps) and reward_timestamps[probe_idx] < stop_time:
            outcomes[trial_idx] = 1
            probe_idx += 1
    return outcomes
...
prev_reward_outcomes = np.concatenate([[0], reward_outcomes[:-1]])
```

iii. The AI notes that reward outcome and previous-trial outcome were computed before removing lick-error trials, so a kept trial still uses the actual previous trial's reward outcome.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from the `position` behavior time series and reward zone bounds from `ZONE_TO_BOUNDS_CM` (hard-coded as A=[80,130], B=[200,250], C=[320,370]). The reward zone label comes from the scene name parsing.

ii.
```python
ZONE_TO_BOUNDS_CM = {"A": (80.0, 130.0), "B": (200.0, 250.0), "C": (320.0, 370.0)}
...
trial_pos = np.clip(position[start:stop], 0.0, 450.0).astype(np.float32)
zone_bounds = ZONE_TO_BOUNDS_CM[label.zone]
discretize_distance_to_zone(trial_pos, zone_bounds)
```

iii. The zone boundaries match the paper's definition.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance is computed: negative before zone (position - zone_start), zero inside zone, positive after zone (position - zone_stop). Position is clipped to [0, 450] before computation.

ii.
```python
def discretize_distance_to_zone(position_cm, zone_bounds):
    zone_start, zone_stop = zone_bounds
    signed_distance = np.where(
        position_cm < zone_start,
        position_cm - zone_start,
        np.where(position_cm > zone_stop, position_cm - zone_stop, 0.0),
    )
```

iii. Standard signed distance computation.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Discretized into 7 categories using explicit conditional assignments matching the instruction bins:
- 0: < -50 cm
- 1: -50 to -10 cm
- 2: -10 to 0 cm
- 3: 0 cm (in zone)
- 4: 0 to 10 cm
- 5: 10 to 50 cm
- 6: > 50 cm

ii.
```python
bins = np.zeros_like(position_cm, dtype=np.int8)
bins[signed_distance < -50.0] = 0
bins[(signed_distance >= -50.0) & (signed_distance <= -10.0)] = 1
bins[(signed_distance > -10.0) & (signed_distance < 0.0)] = 2
bins[signed_distance == 0.0] = 3
bins[(signed_distance > 0.0) & (signed_distance <= 10.0)] = 4
bins[(signed_distance > 10.0) & (signed_distance <= 50.0)] = 5
bins[signed_distance > 50.0] = 6
```

iii. The bin boundaries match the instructions.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Same frame indices as neural data (`start:stop` from trial boundaries), no additional alignment needed.

ii.
```python
start = trial_start_idx[trial_idx]
stop = trial_stop_idx[trial_idx]
trial_pos = np.clip(position[start:stop], 0.0, 450.0)
```

iii. Both neural and behavioral data use the same frame indexing.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from the `position` behavior time series.

ii.
```python
position = behavior["position"]["data"][:].astype(np.float32)
...
trial_pos = np.clip(position[start:stop], 0.0, 450.0).astype(np.float32)
```

iii. Position is read directly from the NWB file.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Position is clipped to [0, 450] cm range. Then discretized into 5 equal bins of 90 cm each using `floor(position / 90)`.

ii.
```python
def discretize_absolute_position(position_cm):
    clipped = np.clip(position_cm, 0.0, 449.999999)
    bins = np.floor(clipped / 90.0).astype(np.int8)
    bins[bins > 4] = 4
    return bins
```

iii. The corridor spans 0-450 cm; 5 bins of 90 cm each covers this range.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Discretized into 5 bins of 90 cm each: [0-90), [90-180), [180-270), [270-360), [360-450].

ii.
```python
clipped = np.clip(position_cm, 0.0, 449.999999)
bins = np.floor(clipped / 90.0).astype(np.int8)
bins[bins > 4] = 4
```

iii. The output_values labels confirm: `["0_to_90cm", "90_to_180cm", "180_to_270cm", "270_to_360cm", "360_to_450cm"]`.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same frame indices as neural data, no additional alignment needed.

ii. Same `start:stop` indexing as neural data.

iii. Aligned by shared frame indices.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from the `lick` behavior time series.

ii.
```python
lick_counts = behavior["lick"]["data"][:].astype(np.float32)
...
trial_lick = (lick_counts[start:stop] > 0).astype(np.int8)
```

iii. The lick variable records lick counts at each timepoint.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Binarized: any positive lick count mapped to 1, otherwise 0.

ii.
```python
trial_lick = (lick_counts[start:stop] > 0).astype(np.int8)
```

iii. The instructions specify binary output (no/yes).

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same frame indices as neural data, no additional alignment needed.

ii. Same `start:stop` indexing.

iii. Aligned by shared frame indices.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from the session scene name (parsed from NWB `identifier` field), NOT from the `reward_zone` behavior time series. The scene name encodes which reward zone is active and when switches occur.

ii.
```python
SCENE_LOCATION_RE = re.compile(r"^Env(?P<env>\d+)_Location(?P<zone>[ABC])$")
SCENE_SWITCH_RE = re.compile(r"^Env(?P<env>\d+)_Location(?P<zone0>[ABC])_to_(?P<zone1>[ABC])$")
...
trial_labels = parse_scene(session_meta.scene, len(trial_start_idx))
zone_idx = ZONE_TO_INDEX[label.zone]
```

iii. The scene name provides a deterministic mapping to reward zone labels. For switch sessions, the switch occurs at trial 30.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Scene name is parsed with regex to extract zone labels. For switch sessions, trials before index 30 get zone0, trials from 30 onward get zone1. The zone label is mapped to an integer (A=0, B=1, C=2) and broadcast across all timepoints in the trial.

ii.
```python
def parse_scene(scene, n_trials, switch_trial_count=30):
    match = SCENE_SWITCH_RE.match(scene)
    if match:
        zone0 = match.group("zone0")
        zone1 = match.group("zone1")
        return [TrialLabel(env=env, zone=zone0 if trial_idx < switch_trial_count else zone1)
                for trial_idx in range(n_trials)]
```

iii. The switch point of 30 trials matches the paper structure.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from the `Reward` behavior timestamps.

ii.
```python
reward_timestamps = behavior["Reward"]["timestamps"][:].astype(np.float64)
reward_outcomes = reward_outcome_per_trial(reward_timestamps, timestamps[trial_start_idx], timestamps[trial_stop_idx])
```

iii. Reward event timestamps are matched to trial boundaries.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, check if any reward timestamp falls within the trial's time window. Binary 0/1 per trial, broadcast across timepoints.

ii.
```python
def reward_outcome_per_trial(reward_timestamps, start_times, stop_times):
    outcomes = np.zeros(len(start_times), dtype=np.int8)
    for trial_idx, (start_time, stop_time) in enumerate(zip(start_times, stop_times)):
        ...
        while probe_idx < len(reward_timestamps) and reward_timestamps[probe_idx] < stop_time:
            outcomes[trial_idx] = 1
            probe_idx += 1
    return outcomes
...
np.full(trial_times.shape, reward_outcomes[trial_idx], dtype=np.int8)
```

iii. The code efficiently sweeps through sorted reward timestamps.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several data issues are handled:
- **Stream length mismatches**: All behavioral and neural streams are truncated to the minimum common length across all streams in the session.
- **Lick-sensor errors**: Trials with >30% of frames having lick count >2 are excluded.
- **Missing sessions**: 2 sessions (m11 ses-01 and ses-02) are noted as absent from the data directory.
- **Environment cross-check**: An assertion verifies that the scene-derived environment matches the behavior `environment` stream.

ii.
```python
session_len = min(len(position), len(speed), len(lick_counts), ..., *plane_lengths)
position = position[:session_len]
...
lick_error_mask = bad_lick_trial_mask(lick_counts, trial_start_idx, trial_stop_idx)
kept_trial_indices = np.where(~lick_error_mask)[0]
```

iii. The AI documents these as defensive measures found during data exploration.

## 13-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are:
1. **Recomputing dF/F and OASIS deconvolution** from raw fluorescence for every neuron in every session
2. **Loading NWB files** with h5py (I/O bound)
3. **Computing interneuron correlations** (speed-dF/F correlation for each neuron)
4. **Saving the pickle file**

ii. N/A

iii. The neural reprocessing is by far the most expensive step since it involves Gaussian smoothing, min/max filtering, and OASIS deconvolution for all neurons.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The interneuron correlation computation loops over individual neurons to compute `np.corrcoef`. The `reward_outcome_per_trial` function uses a sequential sweep that could be vectorized with `np.searchsorted`. The Viterbi-like loop in `parse_scene` is per-trial but is simple enough that it's not a bottleneck.

ii.
```python
block_corr = np.array([
    np.corrcoef(trace, speed_valid)[0, 1]
    if np.nanstd(trace) > 0 and np.nanstd(speed_valid) > 0
    else np.nan
    for trace in dff_valid
])
```

iii. These loops are relatively minor compared to the OASIS deconvolution time.

## 13-c. What processing does the code repeat multiple times?

i. The code processes each session only once (unlike the reference which has separate survey and conversion passes). However, the full dataset is loaded twice if both full and sample datasets are built sequentially (sample is created from the already-built full pickle). The `parse_scene` function is called once per session.

ii. N/A

iii. The sample dataset is efficiently derived from the full dataset without reprocessing.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The dF/F computation is performed but only the OASIS events are kept in the final dataset. The full fluorescence and neuropil arrays are loaded into memory even though only the deconvolved events are saved. Interneuron detection requires computing dF/F for neurons that may be discarded. The `dff` return value from `compute_dff_and_events` is used only for interneuron correlation and then discarded.

ii.
```python
dff, events = compute_dff_and_events(...)
...
block_corr = np.array([np.corrcoef(trace, speed_valid)[0, 1] ... for trace in dff_valid])
block_interneuron = np.isfinite(block_corr) & (block_corr > 0.5)
keep_block = ~block_interneuron
kept_events = events[keep_block]
```

iii. The dF/F is a necessary intermediate for both OASIS deconvolution and interneuron detection, so this is not truly unnecessary, just intermediate.
