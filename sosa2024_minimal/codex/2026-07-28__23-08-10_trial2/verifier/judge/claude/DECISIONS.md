# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI uses `h5py` to directly read NWB files (rather than `pynwb`). It discovers all NWB files by globbing `sub-*/sub-*_behavior+ophys.nwb` under the data root. For each file it reads the subject ID from `general/subject/subject_id` and the session number from the filename. It also extracts the scene name and date from the NWB `identifier` field. Sessions are sorted by (subject number, session number).

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

iii. The agent explored the NWB file structure in detail, confirmed behavior and ophys data were present in each file, and chose `h5py` for direct and memory-efficient access. It confirmed 10 mice and ~14 sessions each.

## 1-b. How are the data split into subjects?

i. Subjects are identified from the `general/subject/subject_id` field in each NWB file. A unique sorted list of subjects is built from all session metadata.

ii.
```python
subjects = sorted({meta.subject for meta in session_meta}, key=lambda s: int(s[1:]))
```

iii. The agent read subject IDs directly from the NWB metadata rather than parsing directory names.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. The session number is parsed from the filename pattern `ses-(\d+)`.

ii.
```python
session_num = int(re.search(r"ses-(\d+)", file_path.name).group(1))
```

iii. The agent confirmed one NWB file per session during data exploration.

## 1-d. How are the data split into trials?

i. Trial starts are found where `trial_start_series > 0`. Trial ends are found where `teleport_series > 0`.

ii.
```python
trial_start_idx = np.where(trial_start_series > 0)[0]
trial_stop_idx = np.where(teleport_series > 0)[0]
```

iii. The agent explored the behavioral time series and determined that `trial_start` and `teleport` signals mark trial boundaries.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered based on a "lick error" criterion: if more than 30% of frames in a trial have lick counts > 2, the trial is removed. This is described as matching the paper's lick-sensor error criterion.

ii.
```python
def bad_lick_trial_mask(lick_counts, start_idx, stop_idx):
    mask = np.zeros(len(start_idx), dtype=bool)
    for trial_idx, (start, stop) in enumerate(zip(start_idx, stop_idx)):
        if np.mean(lick_counts[start:stop] > 2.0) > 0.3:
            mask[trial_idx] = True
    return mask
```

iii. The agent identified this from the paper's methods section describing lick-sensor error trials and implemented it as a quality filter.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the raw `Fluorescence` and `Neuropil` traces in the NWB ophys processing module, NOT from the stored `Deconvolved` field. The AI recomputes dF/F and deconvolved events from scratch.

ii.
```python
fluorescence_ds = ophys["Fluorescence"][plane_key]["data"]
neuropil_ds = ophys["Neuropil"][plane_key]["data"]
...
dff, events = compute_dff_and_events(
    fluorescence=fluorescence, neuropil=neuropil, ...)
```

iii. The agent investigated the NWB contents and determined that the stored `Deconvolved` field is suite2p's own deconvolution, not the signal the paper actually analyses. It therefore recomputed the paper's processing pipeline.

## 2-b. How is the `neural` data processed?

i. The AI reimplements the paper's dF/F pipeline: subtract 0.7 * neuropil, add back per-trial mean neuropil, compute maximin baseline (Gaussian smooth sigma=15, then 300-sample min filter, then 300-sample max filter), compute dF/F = (F - baseline) / |baseline|, smooth with 2-sample Gaussian, deconvolve with OASIS (tau=0.7, frame_rate_hz). However, the AI does NOT implement the `keep_teleports` logic — it always restricts the baseline window to the trial (lap) period only, never allowing it to span the inter-trial teleport period.

ii.
```python
def compute_dff_and_events(fluorescence, neuropil, start_idx, stop_idx, frame_rate_hz,
                           neu_coef=0.7, tau=0.7):
    ...
    for start, stop in zip(start_idx, stop_idx):
        f[:, start:stop] = fluorescence[:, start:stop]
        f_neu[:, start:stop] = neuropil[:, start:stop]
    f -= neu_coef * f_neu
    ...
    for start, stop in zip(start_idx, stop_idx):
        f[:, sl] = f[:, sl] + neu_coef * np.nanmean(f_neu[:, sl], axis=1, keepdims=True)
        baseline[:, sl] = nansmooth_2d(f[:, sl], sigma=15)
        baseline[:, sl] = minimum_filter1d(baseline[:, sl], size=300, axis=1, mode="nearest")
        baseline[:, sl] = maximum_filter1d(baseline[:, sl], size=300, axis=1, mode="nearest")
    dff[:, valid] = (f[:, valid] - baseline[:, valid]) / np.abs(baseline[:, valid])
    for start, stop in zip(start_idx, stop_idx):
        dff[:, sl] = nansmooth_2d(dff[:, sl], sigma=2)
        events[:, sl] = dcnv.oasis(dff[:, sl], 2000, tau, frame_rate_hz)
```

iii. The agent traced the paper's preprocessing code and replicated its steps. It noted the processing matched the Methods description. However, it did not implement the `keep_teleports` distinction based on `teleport_metadata.py`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters: (1) Only `iscell` ROIs (suite2p manual curation) are kept. (2) Putative interneurons (dF/F-speed correlation > 0.5) are excluded.

ii.
```python
plane_iscell = iscell[plane_slice, 0] == 1
...
block_corr = np.array([np.corrcoef(trace, speed_valid)[0, 1] ...])
block_interneuron = np.isfinite(block_corr) & (block_corr > 0.5)
keep_block = ~block_interneuron
```

iii. The agent identified both filters from the paper's methods and code.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Data is aligned to trial start. Since neural and behavioral data share the same time indices, slicing by trial start/stop indices provides the alignment. No additional processing needed.

ii.
```python
start = trial_start_idx[trial_idx]
stop = trial_stop_idx[trial_idx]
# neural_trial = events[:, start:stop]
```

iii. Alignment is inherent in the data structure.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning is applied. The time bin size is computed as the mean frame rate across sessions: `1000.0 / mean_frame_rate`. The frame rate is inferred from the median of timestamp differences.

ii.
```python
def infer_frame_rate(position_timestamps):
    diffs = np.diff(position_timestamps.astype(np.float64))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    return float(1.0 / np.median(diffs))
...
dataset["metadata"]["time_bin_size"] = 1000.0 / mean_frame_rate
```

iii. The agent kept the native sampling rate without rebinning.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Derived from the `position` timestamps in the behavior data.

ii.
```python
timestamps = behavior["position"]["timestamps"][:].astype(np.float64)
...
trial_times = (timestamps[start:stop] - timestamps[start]).astype(np.float32)
```

iii. The agent used position timestamps since all behavioral time series share the same timestamps.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The first timestamp of the trial is subtracted from all timestamps in that trial.

ii.
```python
trial_times = (timestamps[start:stop] - timestamps[start]).astype(np.float32)
```

iii. Standard approach to compute time relative to trial start.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Neural and behavioral data use the same time indices, so alignment is inherent.

ii. Same `start:stop` slicing for both neural and behavioral data.

iii. The agent confirmed timestamps are aligned.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Primarily derived from the `scene` string parsed from the NWB `identifier` field, using regex patterns. The scene string encodes the environment number (Env1 or Env2). It is cross-validated against the `environment` behavior time series.

ii.
```python
SCENE_LOCATION_RE = re.compile(r"^Env(?P<env>\d+)_Location(?P<zone>[ABC])$")
...
def parse_scene(scene, n_trials, switch_trial_count=30):
    match = SCENE_LOCATION_RE.match(scene)
    if match:
        env = int(match.group("env")) - 1
        ...
```
Cross-validation:
```python
env_in_trial = env_timeseries[start:stop]
env_in_trial = env_in_trial[env_in_trial >= 0]
if env_in_trial.size:
    env_mode = int(round(float(np.median(env_in_trial))))
    if env_mode != label.env:
        raise RuntimeError("Environment mismatch...")
```

iii. The agent parsed the scene metadata to determine environment type, with validation against behavior data.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The environment is converted to 0-indexed (Env1 -> 0, Env2 -> 1) and broadcast as a constant across all timepoints in the trial. For switch sessions, the first 30 trials use the pre-switch environment and the rest use the post-switch environment.

ii.
```python
trial_input = np.vstack([
    ...
    np.full(trial_times.shape, float(label.env), dtype=np.float32),
    ...
])
```

iii. The agent derived this from the scene string parsing logic.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Derived from the `trial number` behavior time series in the NWB file, specifically the value at the trial start index.

ii.
```python
trial_number_series = behavior["trial number"]["data"][:].astype(np.float32)
...
np.full(trial_times.shape, trial_number_series[start], dtype=np.float32),
```

iii. The agent used the stored trial number from the NWB behavior data.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The trial number value at the start of each trial is broadcast as a constant across all timepoints in that trial. No additional processing.

ii.
```python
np.full(trial_times.shape, trial_number_series[start], dtype=np.float32),
```

iii. The value is taken directly from the stored time series.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from the `Reward` timestamps in the behavior data.

ii.
```python
reward_timestamps = behavior["Reward"]["timestamps"][:].astype(np.float64)
...
def reward_outcome_per_trial(reward_timestamps, start_times, stop_times):
    outcomes = np.zeros(len(start_times), dtype=np.int8)
    ...
    while probe_idx < len(reward_timestamps) and reward_timestamps[probe_idx] < stop_time:
        outcomes[trial_idx] = 1
        probe_idx += 1
    return outcomes
```

iii. The agent used reward event timestamps to determine which trials were rewarded.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For each trial, determine if any reward event timestamp fell within its time range. The previous trial's outcome is then shifted: `prev_reward_outcomes = np.concatenate([[0], reward_outcomes[:-1]])`. First trial gets 0.

ii.
```python
reward_outcomes = reward_outcome_per_trial(reward_timestamps, timestamps[trial_start_idx], timestamps[trial_stop_idx])
prev_reward_outcomes = np.concatenate([[0], reward_outcomes[:-1]]).astype(np.float32)
```

iii. Standard shift-by-one approach to get previous trial outcome.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from the `position` behavior time series and reward zone boundaries. The reward zone label for each trial comes from the `parse_scene()` function which parses the NWB identifier scene string (e.g., `Env1_LocationA` gives zone A). Zone boundaries are hardcoded: A=[80,130], B=[200,250], C=[320,370].

ii.
```python
ZONE_TO_BOUNDS_CM = {
    "A": (80.0, 130.0),
    "B": (200.0, 250.0),
    "C": (320.0, 370.0),
}
...
label = trial_labels[trial_idx]
zone_bounds = ZONE_TO_BOUNDS_CM[label.zone]
```

iii. The agent derived reward zone labels from the scene metadata string rather than from the behavioral `reward_zone` time series.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance is computed: negative if before zone start, positive if past zone end, 0 if inside. Position is clipped to [0, 450] before computing distance.

ii.
```python
def discretize_distance_to_zone(position_cm, zone_bounds):
    zone_start, zone_stop = zone_bounds
    signed_distance = np.where(
        position_cm < zone_start,
        position_cm - zone_start,
        np.where(position_cm > zone_stop, position_cm - zone_stop, 0.0),
    )
    bins = np.zeros_like(position_cm, dtype=np.int8)
    bins[signed_distance < -50.0] = 0
    bins[(signed_distance >= -50.0) & (signed_distance <= -10.0)] = 1
    bins[(signed_distance > -10.0) & (signed_distance < 0.0)] = 2
    bins[signed_distance == 0.0] = 3
    bins[(signed_distance > 0.0) & (signed_distance <= 10.0)] = 4
    bins[(signed_distance > 10.0) & (signed_distance <= 50.0)] = 5
    bins[signed_distance > 50.0] = 6
    return bins
```

iii. The agent implemented the distance computation and discretization based on the instructions' bin specifications.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Discretized into 7 bins using explicit conditional logic matching the instruction specifications. Boundary conditions: -50 uses `<=` (inclusive in bin 1), -10 uses `>` for bin 2 (exclusive at -10), 0 is its own bin (bin 3), +10 uses `<=` (inclusive in bin 4), +50 uses `<=` (inclusive in bin 5).

ii. See code in 7-b above.

iii. The bin edges follow the instruction specifications.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Same time indices as neural data within each trial. Position is clipped to [0, 450] before computing.

ii.
```python
trial_pos = np.clip(position[start:stop], 0.0, 450.0).astype(np.float32)
```

iii. Alignment is inherent since the same start:stop indices are used.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from the `position` behavior time series.

ii.
```python
position = behavior["position"]["data"][:].astype(np.float32)
...
trial_pos = np.clip(position[start:stop], 0.0, 450.0).astype(np.float32)
```

iii. Position is the animal's location in the VR corridor.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Position is clipped to [0, 450] range, then discretized using `floor(clipped / 90)` to get 5 equal bins of 90 cm each.

ii.
```python
def discretize_absolute_position(position_cm):
    clipped = np.clip(position_cm, 0.0, 449.999999)
    bins = np.floor(clipped / 90.0).astype(np.int8)
    bins[bins > 4] = 4
    return bins
```

iii. The agent used arithmetic binning rather than `np.digitize`.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. 5 bins of 90 cm each: [0,90), [90,180), [180,270), [270,360), [360,450]. Position is clipped to [0, 449.999999] first.

ii. See code in 8-b.

iii. The bins correspond to equal-sized 90 cm segments spanning the 450 cm track.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same time indices as neural data — no additional alignment needed.

ii. Same `start:stop` slicing.

iii. Alignment is inherent.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from the `lick` behavior time series.

ii.
```python
lick_counts = behavior["lick"]["data"][:].astype(np.float32)
...
trial_lick = (lick_counts[start:stop] > 0).astype(np.int8)
```

iii. The lick variable records lick events at each timepoint.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Binarized: any positive lick value is mapped to 1, otherwise 0.

ii.
```python
trial_lick = (lick_counts[start:stop] > 0).astype(np.int8)
```

iii. Matches the instruction specification of binary lick output.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same time indices — no additional alignment needed.

ii. Same `start:stop` slicing.

iii. Alignment is inherent.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from the `scene` string in the NWB `identifier` field, parsed with regex. For switch sessions (e.g., `Env1_LocationA_to_B`), the first 30 trials use zone A and subsequent trials use zone B.

ii.
```python
def parse_scene(scene, n_trials, switch_trial_count=30):
    match = SCENE_SWITCH_RE.match(scene)
    if match:
        env = int(match.group("env")) - 1
        zone0 = match.group("zone0")
        zone1 = match.group("zone1")
        return [TrialLabel(env=env, zone=zone0 if trial_idx < switch_trial_count else zone1)
                for trial_idx in range(n_trials)]
```

iii. The agent determined that the scene string encodes the reward zone information, including switches. The switch trial count of 30 is hardcoded.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Zone label (A, B, or C) is mapped to index (0, 1, 2) and broadcast as a constant per trial.

ii.
```python
zone_idx = ZONE_TO_INDEX[label.zone]
...
np.full(trial_times.shape, zone_idx, dtype=np.int8),
```

iii. Direct mapping from zone label to integer index.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from the `Reward` timestamps in the behavior data.

ii.
```python
reward_timestamps = behavior["Reward"]["timestamps"][:].astype(np.float64)
reward_outcomes = reward_outcome_per_trial(reward_timestamps, timestamps[trial_start_idx], timestamps[trial_stop_idx])
```

iii. Reward events are identified by their timestamps falling within trial boundaries.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Binary: 1 if any reward timestamp falls within the trial's time range, 0 otherwise. Constant per trial.

ii.
```python
def reward_outcome_per_trial(reward_timestamps, start_times, stop_times):
    outcomes = np.zeros(len(start_times), dtype=np.int8)
    ...
    while probe_idx < len(reward_timestamps) and reward_timestamps[probe_idx] < stop_time:
        outcomes[trial_idx] = 1
        probe_idx += 1
    return outcomes
```

iii. Uses a sweep-line approach through sorted reward timestamps.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **Neural/behavior length mismatch**: All data arrays are cropped to the minimum session length across all behavioral and ophys streams.
- **Lick error trials**: Trials with excessive lick sensor noise (>30% frames with lick > 2) are removed.
- **Missing neurons**: If no neurons are kept for a trial, a RuntimeError is raised.

ii.
```python
session_len = min(len(position), len(speed), len(lick_counts), ..., *plane_lengths)
position = position[:session_len]
speed = speed[:session_len]
...
```

iii. The agent handled length mismatches defensively by cropping to the minimum.

## 13-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are:
1. Reading NWB files with `h5py` (I/O bound)
2. Computing dF/F and deconvolved events (`compute_dff_and_events`) — involves Gaussian smoothing, min/max filters, and OASIS deconvolution per plane block
3. Computing speed-dF/F correlations for interneuron filtering

ii. N/A

iii. The agent monitored memory usage during conversion and noted the process was memory-safe at ~2.3 GB RSS thanks to the blockwise approach.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The `reward_outcome_per_trial` function uses a Python loop over trials that could potentially be vectorized with `np.searchsorted`. The `bad_lick_trial_mask` function similarly loops over trials. The interneuron correlation computation loops over cells.

ii. N/A

iii. These loops are over relatively small numbers of trials/cells, so the performance impact is minor.

## 13-c. What processing does the code repeat multiple times?

i. The code processes all data in a single pass per session — it does not have a separate survey step that re-reads NWB files. This is more efficient than the reference solution which loads NWB files twice (once for survey, once for conversion).

ii. N/A

iii. The agent designed a single-pass approach.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The `dff` (delta-F/F) array is computed but only the `events` (deconvolved) array is used for the final neural data. The dff is needed for interneuron filtering (correlation with speed), but the full dff array itself is not saved. Additionally, the agent computes a `make_sample_dataset` that creates a subset pickle file not required by the task.

ii.
```python
dff, events = compute_dff_and_events(...)
# dff is used for interneuron filtering then discarded
```

iii. The dff is a necessary intermediate for computing events and filtering interneurons.
