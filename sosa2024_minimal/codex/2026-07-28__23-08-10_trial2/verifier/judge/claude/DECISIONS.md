# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. All NWB files are discovered by globbing `data/sub-*/sub-*_behavior+ophys.nwb`. Each file is opened with `h5py` (not `pynwb`). The subject ID and session scene name are extracted from NWB metadata fields (`general/subject/subject_id` and `identifier`). All 152 NWB files across 11 subjects are processed.

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

iii. From CONVERSION_NOTES: "The supplied data/ directory contains 152 NWB sessions, not 154. The missing two sessions are m11 ses-01 and ses-02." The agent verified against the paper's expected count.

## 1-b. How are the data split into subjects?

i. Subjects are identified by reading `general/subject/subject_id` from each NWB file. Unique subject IDs are sorted numerically (by the number after 'm'). 11 subjects total.

ii.
```python
subjects = sorted({meta.subject for meta in session_meta}, key=lambda s: int(s[1:]))
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
```

iii. The agent read the subject ID directly from NWB metadata rather than parsing directory names.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. The session number is parsed from the filename (`ses-XX`). Sessions are sorted by (subject, session_num).

ii.
```python
session_num = int(re.search(r"ses-(\d+)", file_path.name).group(1))
```

iii. From CONVERSION_NOTES: "The final full dataset therefore contains 152 sessions across 11 subjects."

## 1-d. How are the data split into trials?

i. Trial boundaries are identified using `trial_start` (where > 0) for start indices and `teleport` (where > 0) for stop indices. Data between start and stop is extracted per trial.

ii.
```python
trial_start_idx = np.where(trial_start_series > 0)[0]
trial_stop_idx = np.where(teleport_series > 0)[0]
# ...
for out_idx, trial_idx in enumerate(kept_trial_indices):
    start = trial_start_idx[trial_idx]
    stop = trial_stop_idx[trial_idx]
    # neural, input, output sliced as [..., start:stop]
```

iii. From CONVERSION_NOTES: "For each trial, the kept sample range is [trial_start_idx : teleport_idx). The teleport sample itself is excluded."

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered based on a lick-sensor error criterion: a trial is dropped if more than 30% of its frame samples have lick count > 2. This follows the paper's described exclusion criterion. 81 trials were removed across the full dataset.

ii.
```python
def bad_lick_trial_mask(lick_counts, start_idx, stop_idx):
    mask = np.zeros(len(start_idx), dtype=bool)
    for trial_idx, (start, stop) in enumerate(zip(start_idx, stop_idx)):
        if np.mean(lick_counts[start:stop] > 2.0) > 0.3:
            mask[trial_idx] = True
    return mask
# ...
lick_error_mask = bad_lick_trial_mask(lick_counts, trial_start_idx, trial_stop_idx)
kept_trial_indices = np.where(~lick_error_mask)[0]
```

iii. From CONVERSION_NOTES: "This matches the paper's lick-sensor error criterion." The agent explored multiple thresholds (0.3, 0.35, 0.5) before settling on 0.3.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `Fluorescence` and `Neuropil` fields in the NWB ophys processing module. The agent recomputes dF/F and OASIS deconvolution from these raw signals, rather than using the stored `Deconvolved` data.

ii.
```python
fluorescence_ds = ophys["Fluorescence"][plane_key]["data"]
neuropil_ds = ophys["Neuropil"][plane_key]["data"]
# ...
fluorescence = fluorescence_ds[:session_len, block_local].T.astype(np.float32)
neuropil = neuropil_ds[:session_len, block_local].T.astype(np.float32)
dff, events = compute_dff_and_events(
    fluorescence=fluorescence,
    neuropil=neuropil,
    start_idx=trial_start_idx,
    stop_idx=trial_stop_idx,
    frame_rate_hz=frame_rate_hz,
)
```

iii. From CONVERSION_NOTES: "The saved neural activity is trial-wise OASIS event activity recomputed from Suite2p fluorescence and neuropil, rather than taking the NWB Deconvolved array directly." The agent's reasoning was to ensure paper-matched neural processing.

## 2-b. How is the `neural` data processed?

i. The agent implements a full dF/F and deconvolution pipeline:
1. Neuropil subtraction with coefficient 0.7
2. Per-trial maximin baseline (Gaussian smooth sigma=15, min filter 300, max filter 300)
3. dF/F = (F - baseline) / |baseline|
4. Gaussian smoothing of dF/F with sigma=2
5. OASIS deconvolution with tau=0.7
6. Cells from multiple planes are concatenated.

ii.
```python
def compute_dff_and_events(fluorescence, neuropil, start_idx, stop_idx,
                           frame_rate_hz, neu_coef=0.7, tau=0.7):
    f -= neu_coef * f_neu
    # per-trial baseline
    baseline[:, sl] = nansmooth_2d(f[:, sl], sigma=15)
    baseline[:, sl] = minimum_filter1d(baseline[:, sl], size=300, axis=1, mode="nearest")
    baseline[:, sl] = maximum_filter1d(baseline[:, sl], size=300, axis=1, mode="nearest")
    dff[:, valid] = (f[:, valid] - baseline[:, valid]) / np.abs(baseline[:, valid])
    # smoothing and deconvolution
    dff[:, sl] = nansmooth_2d(dff[:, sl], sigma=2)
    events[:, sl] = dcnv.oasis(dff[:, sl], 2000, tau, frame_rate_hz)
    return dff, events
```

iii. From CONVERSION_NOTES: the agent explicitly lists the processing steps as matching "the paper repo's maximin baseline per trial."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters are applied:
1. Only curated Suite2p ROIs with `iscell[:, 0] == 1` are kept.
2. Putative interneurons are excluded when dF/F-speed Pearson correlation exceeds 0.5.

ii.
```python
plane_iscell = iscell[plane_slice, 0] == 1
# ...
block_corr = np.array([
    np.corrcoef(trace, speed_valid)[0, 1]
    if np.nanstd(trace) > 0 and np.nanstd(speed_valid) > 0
    else np.nan
    for trace in dff_valid
])
block_interneuron = np.isfinite(block_corr) & (block_corr > 0.5)
keep_block = ~block_interneuron
```

iii. From CONVERSION_NOTES: "Putative interneurons were excluded when Pearson correlation between dF/F and running speed exceeded 0.5... This follows the paper's description of excluding speed-correlated putative interneurons."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Data is aligned to trial start by slicing neural data between `trial_start_idx` and `trial_stop_idx` for each trial. No additional temporal shifting is needed.

ii.
```python
start = trial_start_idx[trial_idx]
stop = trial_stop_idx[trial_idx]
# ...
kept_events[:, start:stop]
```

iii. From CONVERSION_NOTES: "Trials are aligned to trial start."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning is applied. The native frame rate (~15.5 Hz) is preserved. The time bin size is computed as `1000.0 / mean_frame_rate` = ~64.48 ms. Frame rate is inferred from median diff of behavior timestamps.

ii.
```python
def infer_frame_rate(position_timestamps):
    diffs = np.diff(position_timestamps.astype(np.float64))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    return float(1.0 / np.median(diffs))
# ...
dataset["metadata"]["time_bin_size"] = 1000.0 / mean_frame_rate
```

iii. From CONVERSION_NOTES: "Frame rate is inferred from the behavior timestamps, not trusted from the NWB starting_time/rate metadata. The final mean frame rate was 15.507813 Hz, so metadata['time_bin_size'] is 64.4836 ms."

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Derived from the `position` behavior timestamps in the NWB file.

ii.
```python
timestamps = behavior["position"]["timestamps"][:].astype(np.float64)
# ...
trial_times = (timestamps[start:stop] - timestamps[start]).astype(np.float32)
```

iii. The agent used the position timestamps, which are shared across all behavior time series.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The timestamp at trial start is subtracted from all timestamps within the trial.

ii.
```python
trial_times = (timestamps[start:stop] - timestamps[start]).astype(np.float32)
```

iii. Standard approach to get time relative to trial onset.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Neural and behavioral data share the same sample indices. The same `start:stop` range is used for both, so they are inherently aligned.

ii.
```python
start = trial_start_idx[trial_idx]
stop = trial_stop_idx[trial_idx]
trial_times = (timestamps[start:stop] - timestamps[start]).astype(np.float32)
# neural uses same start:stop
kept_events[:, start:stop]
```

iii. All streams are truncated to a common session length before trial extraction.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Derived from the session scene name stored in the NWB `identifier` field (e.g., "Env1_LocationA"). Cross-validated against the `environment` behavior time series.

ii.
```python
scene = parts[-1]  # from NWB identifier
# ...
trial_labels = parse_scene(session_meta.scene, len(trial_start_idx))
# ...
label = trial_labels[trial_idx]
np.full(trial_times.shape, float(label.env), dtype=np.float32)
# cross-check:
env_in_trial = env_timeseries[start:stop]
env_mode = int(round(float(np.median(env_in_trial))))
if env_mode != label.env:
    raise RuntimeError("Environment mismatch...")
```

iii. From CONVERSION_NOTES: "environment_type is 0 for ENV1, 1 for ENV2." The scene-derived environment is cross-checked on every kept trial.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The scene name is parsed via regex to extract the environment number. For switch sessions (e.g., "Env1_A_to_Env2_B"), the environment switches at trial 30. The environment number is converted from 1-indexed to 0-indexed.

ii.
```python
SCENE_LOCATION_RE = re.compile(r"^Env(?P<env>\d+)_Location(?P<zone>[ABC])$")
SCENE_ENV_SWITCH_RE = re.compile(r"^Env(?P<env0>\d+)_(?P<zone0>[ABC])_to_Env(?P<env1>\d+)_(?P<zone1>[ABC])$")
# ...
def parse_scene(scene, n_trials, switch_trial_count=30):
    match = SCENE_LOCATION_RE.match(scene)
    if match:
        env = int(match.group("env")) - 1
        return [TrialLabel(env=env, zone=zone) for _ in range(n_trials)]
```

iii. The agent identified three scene name patterns and handles each. The 30-trial switch point matches the paper's task structure.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Derived from the `trial number` behavior time series in the NWB file, reading the value at the trial start frame.

ii.
```python
trial_number_series = behavior["trial number"]["data"][:].astype(np.float32)
# ...
np.full(trial_times.shape, trial_number_series[start], dtype=np.float32)
```

iii. The agent used the NWB-stored trial number rather than a sequential loop counter.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The trial number value at the start frame of each trial is read and broadcast across all timepoints in the trial. No further processing.

ii.
```python
np.full(trial_times.shape, trial_number_series[start], dtype=np.float32)
```

iii. The value is taken directly from the NWB behavioral stream.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from the `Reward` behavior time series timestamps.

ii.
```python
reward_timestamps = behavior["Reward"]["timestamps"][:].astype(np.float64)
```

iii. Reward delivery times are recorded with separate timestamps from the behavior sampling rate.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For each trial, reward outcome is determined by checking if any reward timestamp falls within the trial's time range. Previous trial outcome is then the reward outcome of the preceding trial, with the first trial set to 0. Reward outcomes are computed before lick-error trial exclusion.

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
# ...
prev_reward_outcomes = np.concatenate([[0], reward_outcomes[:-1]]).astype(np.float32)
```

iii. From CONVERSION_NOTES: "Reward outcome and previous-trial outcome were computed before removing these trials, so a kept trial still uses the actual previous trial's reward outcome even if that previous trial was later excluded."

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from the `position` behavior time series and the reward zone identity for the current trial. The reward zone identity comes from parsing the session scene name (not from the `reward_zone` behavior variable).

ii.
```python
trial_pos = np.clip(position[start:stop], 0.0, 450.0).astype(np.float32)
label = trial_labels[trial_idx]
zone_bounds = ZONE_TO_BOUNDS_CM[label.zone]
discretize_distance_to_zone(trial_pos, zone_bounds)
```

iii. The zone boundaries are defined in `ZONE_TO_BOUNDS_CM` matching the paper's reward zone locations: A=(80,130), B=(200,250), C=(320,370).

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance is computed from position to the nearest edge of the reward zone. Distance is 0 when inside the zone, negative when before it, positive when past it. Position is clipped to [0, 450] before computation.

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

iii. This matches the paper's concept of distance relative to the reward zone.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The continuous signed distance is discretized into 7 bins using explicit conditional logic:
- 0: < -50 cm
- 1: -50 to -10 cm
- 2: -10 to 0 cm (exclusive)
- 3: 0 cm (in reward zone)
- 4: >0 to 10 cm
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

iii. The bin edges match the instruction specifications.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Same `start:stop` indices used for both position and neural data within each trial. No additional alignment needed.

ii.
```python
start = trial_start_idx[trial_idx]
stop = trial_stop_idx[trial_idx]
trial_pos = np.clip(position[start:stop], 0.0, 450.0)
# neural also uses start:stop
```

iii. All streams share the same sample indexing.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from the `position` behavior time series.

ii.
```python
position = behavior["position"]["data"][:].astype(np.float32)
# ...
trial_pos = np.clip(position[start:stop], 0.0, 450.0).astype(np.float32)
```

iii. The position variable records the animal's location in the VR corridor in cm.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Position is clipped to [0, 450] cm before discretization.

ii.
```python
trial_pos = np.clip(position[start:stop], 0.0, 450.0).astype(np.float32)
```

iii. Clipping prevents out-of-range values.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Position is discretized into 5 bins of 90 cm each by dividing clipped position by 90 and flooring:
- 0: 0-90 cm
- 1: 90-180 cm
- 2: 180-270 cm
- 3: 270-360 cm
- 4: 360-450 cm

ii.
```python
def discretize_absolute_position(position_cm):
    clipped = np.clip(position_cm, 0.0, 449.999999)
    bins = np.floor(clipped / 90.0).astype(np.int8)
    bins[bins > 4] = 4
    return bins
```

iii. The corridor is 450 cm, so 5 equal bins of 90 cm each. The output_values are labeled `["0_to_90cm", "90_to_180cm", "180_to_270cm", "270_to_360cm", "360_to_450cm"]`.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same `start:stop` indices as neural data. No additional alignment needed.

ii. Same indexing as described in 7-d.

iii. All streams share sample indexing.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from the `lick` behavior time series.

ii.
```python
lick_counts = behavior["lick"]["data"][:].astype(np.float32)
```

iii. The lick variable records lick counts at each timepoint.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Binarized: any positive lick count is mapped to 1, otherwise 0.

ii.
```python
trial_lick = (lick_counts[start:stop] > 0).astype(np.int8)
```

iii. The instructions specify binary output (no/yes).

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same `start:stop` indices as neural data. No additional alignment needed.

ii. Same indexing as neural data.

iii. All streams share sample indexing.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from the session scene name in the NWB `identifier` field, not from the `reward_zone` behavior variable. Scene names encode which zone is active (e.g., "Env1_LocationA").

ii.
```python
scene = parts[-1]  # from NWB identifier
trial_labels = parse_scene(session_meta.scene, len(trial_start_idx))
# ...
zone_idx = ZONE_TO_INDEX[label.zone]  # {'A': 0, 'B': 1, 'C': 2}
np.full(trial_times.shape, zone_idx, dtype=np.int8)
```

iii. From CONVERSION_NOTES: "Trial environment and reward-zone labels are derived from the session scene name, matching the task structure in the paper."

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The scene name is parsed with regex to extract the zone letter (A, B, or C). For switch sessions, the zone changes at trial 30. The letter is mapped to an index (A=0, B=1, C=2) and broadcast across all timepoints in the trial.

ii.
```python
def parse_scene(scene, n_trials, switch_trial_count=30):
    match = SCENE_SWITCH_RE.match(scene)
    if match:
        zone0 = match.group("zone0")
        zone1 = match.group("zone1")
        return [
            TrialLabel(env=env, zone=zone0 if trial_idx < switch_trial_count else zone1)
            for trial_idx in range(n_trials)
        ]
```

iii. From CONVERSION_NOTES: "For switch sessions, the switch point is trial 30, matching the paper/task structure."

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from the `Reward` behavior time series timestamps.

ii.
```python
reward_timestamps = behavior["Reward"]["timestamps"][:].astype(np.float64)
```

iii. Reward delivery events are recorded with separate timestamps.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, check if any reward timestamp falls between the trial start and stop times. Output is 1 if rewarded, 0 if not. Value is constant across all timepoints in the trial.

ii.
```python
def reward_outcome_per_trial(reward_timestamps, start_times, stop_times):
    outcomes = np.zeros(len(start_times), dtype=np.int8)
    reward_idx = 0
    for trial_idx, (start_time, stop_time) in enumerate(zip(start_times, stop_times)):
        while reward_idx < len(reward_timestamps) and reward_timestamps[reward_idx] < start_time:
            reward_idx += 1
        probe_idx = reward_idx
        while probe_idx < len(reward_timestamps) and reward_timestamps[probe_idx] < stop_time:
            outcomes[trial_idx] = 1
            probe_idx += 1
    return outcomes
# ...
np.full(trial_times.shape, reward_outcomes[trial_idx], dtype=np.int8)
```

iii. The reward_outcome_per_trial function uses a linear scan through sorted reward timestamps for efficiency.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **Stream length mismatch**: All behavioral and neural streams are truncated to the minimum common length across position, speed, lick, environment, trial number, trial_start, teleport, timestamps, and all fluorescence planes.
- **Lick-sensor errors**: Trials with >30% anomalous lick frames are excluded.
- **NaN handling in neural processing**: The dF/F computation uses NaN-aware smoothing (`nansmooth_2d`) and only processes valid (in-trial) segments.

ii.
```python
session_len = min(
    len(position), len(speed), len(lick_counts), len(env_timeseries),
    len(trial_number_series), len(trial_start_series), len(teleport_series),
    len(timestamps), *plane_lengths,
)
position = position[:session_len]
# ... all streams truncated to session_len
```

iii. From CONVERSION_NOTES: "A few of these sessions have behavior and fluorescence streams that differ by one frame. All streams in a session were truncated to the common minimum length."

## 13-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are:
1. **Neural reprocessing**: Computing dF/F baseline (Gaussian smoothing, min/max filters) and OASIS deconvolution for every neuron in every session. This is the dominant cost.
2. **Loading NWB files**: Reading large fluorescence and neuropil arrays from disk.
3. **Interneuron correlation computation**: Computing Pearson correlation between each neuron's dF/F and running speed.

ii. The `compute_dff_and_events` function contains the heaviest computation: Gaussian smoothing, min/max filtering, and OASIS deconvolution, applied per-block per-plane.

iii. The agent's approach of recomputing neural activity from raw fluorescence rather than using stored Deconvolved data adds substantial computation time.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops could be vectorized:
1. The per-neuron correlation computation (list comprehension over `dff_valid` traces) could use vectorized correlation.
2. The `reward_outcome_per_trial` function iterates through trials and reward timestamps sequentially; `np.searchsorted` could replace this.
3. The `bad_lick_trial_mask` function loops over trials; vectorized operations with reshaping or `np.add.reduceat` could replace it.

ii.
```python
block_corr = np.array([
    np.corrcoef(trace, speed_valid)[0, 1]
    if np.nanstd(trace) > 0 and np.nanstd(speed_valid) > 0
    else np.nan
    for trace in dff_valid
])
```

iii. The per-neuron loop is the most obvious candidate, as correlation can be computed for all neurons simultaneously using matrix operations.

## 13-c. What processing does the code repeat multiple times?

i. The code processes sessions in a single pass (no separate survey step), so NWB files are loaded only once. However, the block-wise processing of neurons means that trial slicing (`start:stop`) is repeated for each block of neurons within a session. The `nansmooth_2d` function is called multiple times on overlapping trial segments (once for baseline, once for dF/F smoothing).

ii. N/A

iii. The block-wise approach is necessary for memory management but introduces repeated trial-level operations.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The full dF/F computation is performed but only the OASIS-deconvolved events are kept as the final neural data. The dF/F values themselves are used only for interneuron correlation checking and are then discarded. Additionally, the `speed` output is computed and included in the converted data, which matches the instructions but adds processing overhead.

ii.
```python
dff, events = compute_dff_and_events(...)
# dff is used only for interneuron correlation, then discarded
# only events is kept in the final dataset
```

iii. The dF/F is an intermediate product required for the processing pipeline, so computing it is necessary even though it's not saved.
