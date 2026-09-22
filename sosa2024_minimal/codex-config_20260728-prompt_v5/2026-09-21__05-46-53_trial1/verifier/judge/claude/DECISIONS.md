# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI finds all NWB files by globbing `sub-*/sub-*_behavior+ophys.nwb` under the data root directory. Each NWB file is opened with `pynwb.NWBHDF5IO`. Subject IDs are extracted from the NWB `subject.subject_id` field (by opening each file). All behavioral time series and neural (deconvolved) data are read from each NWB file.

ii.
```python
session_paths = sorted(DATA_ROOT.glob("sub-*/sub-*_behavior+ophys.nwb"))
...
for path in session_paths:
    with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
        subject = io.read().subject.subject_id
```

iii. From trajectory step 47: "The dataset is fairly clean globally: 152 imaging sessions across 11 mice, identical imaging-rate bins (~64.48 ms), and matched trial_start/teleport counts in every session." The AI explored the directory structure and NWB schema extensively before writing the converter.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified by reading the `subject.subject_id` field from each NWB file. Unique subjects are collected in order of first appearance. A `subject_to_idx` mapping is built.

ii.
```python
subjects = []
for path in session_paths:
    with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
        subject = io.read().subject.subject_id
    if subject not in subjects:
        subjects.append(subject)
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
```

iii. The AI opened each NWB file to read the subject ID rather than parsing directory names. This approach is more robust to naming inconsistencies.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. Sessions are processed in sorted file order.

ii.
```python
session_paths = sorted(DATA_ROOT.glob("sub-*/sub-*_behavior+ophys.nwb"))
...
for path in session_paths:
    session = convert_session(path, subject_to_idx)
```

iii. From trajectory step 30: "I now know the NWB files already include per-frame trial_start, teleport, trial number, environment, reward_zone, lick, speed, and deconvolved activity." Each file is treated as one session.

## 1-d. How are the data split into trials?

i. Trials are identified by finding indices where `trial_start > 0` and where `teleport > 0`. Each trial spans from `trial_start[i]` (inclusive) to `teleport[i]` (exclusive).

ii.
```python
trial_start = np.flatnonzero(np.asarray(behavior["trial_start"].data[:]) > 0)
teleport = np.flatnonzero(np.asarray(behavior["teleport"].data[:]) > 0)
...
for trial_idx, (start, stop) in enumerate(zip(trial_start, teleport)):
    ...
    neural_trial = deconv[start:stop].T
```

iii. From trajectory step 62: "Trial segmentation is clear now: the valid lap is trial_start inclusive to teleport exclusive." The AI verified that the number of trial starts equals the number of teleports in every session.

## 1-e. How are trials filtered based on quality controls?

i. Trials are dropped if more than 35% of frames have lick values greater than 2 (a lick sensor error heuristic). Sessions with fewer than 2 remaining trials are also excluded.

ii.
```python
LICK_ERROR_THRESHOLD = 0.35
...
trial_lick = lick[start:stop]
if np.mean(trial_lick > 2) > LICK_ERROR_THRESHOLD:
    drop_trial[trial_idx] = True
...
if session["n_trials_kept"] < 2:
    continue
```

iii. From trajectory step 65: "I have the main conversion decisions now: curated Suite2p cells only, trials from trial_start to teleport, and probably dropping the small number of lick-corrupted trials rather than fabricating lick labels." The AI identified 69 lick-corrupted trials across the dataset.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI uses the NWB's pre-stored `Deconvolved` data interface. This is suite2p's own deconvolution of raw fluorescence, NOT the paper's custom dF/F + OASIS deconvolution pipeline.

ii.
```python
def load_curated_deconvolved_activity(ophys_interfaces):
    deconv_name = next(name for name in ophys_interfaces.keys() if "Deconvolved" in name)
    deconv_iface = ophys_interfaces[deconv_name]
    ...
    for plane_name in sorted(deconv_iface.roi_response_series.keys()):
        rrs = deconv_iface.roi_response_series[plane_name]
        ...
        parts.append(plane_data[:, keep_cells[roi_indices]])
    return np.concatenate(parts, axis=1), keep_cells
```

iii. From trajectory step 25: "The NWB files are already in the useful form: behavior streams are aligned frame-by-frame to imaging, and deconvolved fluorescence has the same number of samples." The AI treated the NWB Deconvolved field as the correct neural signal, without recognizing that the paper computes its own deconvolved events from raw fluorescence and neuropil using a custom pipeline.

## 2-b. How is the `neural` data processed?

i. No additional processing is applied to the deconvolved data beyond filtering to curated cells (`iscell`) and temporal rebinning (summing 16 frames per bin). The AI does NOT replicate the paper's processing pipeline (neuropil subtraction, maximin baseline, dF/F computation, Gaussian smoothing, OASIS deconvolution with paper-specific parameters).

ii.
```python
deconv, keep_cells = load_curated_deconvolved_activity(ophys)
...
neural_binned[:, bin_idx] = neural_trial[:, sl].sum(axis=1, dtype=np.float32)
```

iii. The AI assumed the NWB's deconvolved field was sufficient. From trajectory step 76: "I've finished the design pass. I'm writing convert_data.py now with the concrete rules we recovered: curated iscell ROIs, trial-start to teleport segmentation..."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only the `iscell` filter is applied: cells where `iscell[:,0] == 1` in the ROI table are kept. Putative interneurons are NOT filtered out.

ii.
```python
def get_iscell_mask(plane_segmentation):
    iscell = np.asarray(plane_segmentation.to_dataframe()["iscell"].tolist())
    if iscell.ndim == 1:
        keep = iscell.astype(float) > 0
    else:
        keep = iscell[:, 0].astype(float) > 0
    return keep
```

iii. From trajectory step 55: "The ROI table does include Suite2p iscell, and the deconvolved matrices still contain non-cell ROIs, so I will filter to curated cells (iscell[:,0] == 1) to match the repository's session objects." The AI did not implement the paper's interneuron filtering step.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Data is aligned to trial start by extracting data from `trial_start` to `teleport` indices. No additional alignment shift is needed since the trial start IS the alignment event.

ii.
```python
neural_trial = deconv[start:stop].T
time_trial_s = (timestamps[start:stop] - timestamps[start]).astype(np.float32)
```

iii. The instructions say "Temporally align based on start of the trial." The AI correctly aligns to trial start.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI applies temporal rebinning: 16 imaging frames are aggregated into one time bin. At the native ~15.5 Hz rate, this produces bins of approximately 1.03 seconds. Neural data is summed within each bin; behavioral variables are either averaged (position, speed), left-edge sampled (time), or any-detected (lick).

ii.
```python
BIN_FRAMES = 16
...
def bin_trial(neural_trial, time_trial_s, pos_trial_cm, speed_trial_cm_s, lick_trial, zone_label):
    nframes = neural_trial.shape[1]
    nbins = math.ceil(nframes / BIN_FRAMES)
    ...
    neural_binned[:, bin_idx] = neural_trial[:, sl].sum(axis=1, dtype=np.float32)
    time_binned[bin_idx] = time_trial_s[start]
    pos_binned[bin_idx] = np.mean(pos_trial_cm[sl], dtype=np.float64)
    speed_binned[bin_idx] = np.mean(speed_trial_cm_s[sl], dtype=np.float64)
    lick_binned[bin_idx] = int(np.any(lick_trial[sl] > 0))
```

iii. From trajectory step 69: "The remaining tradeoff is dataset size versus fidelity. At native imaging resolution the full dataset is too large for this shared decoder to train realistically, so I'm measuring trial durations and neuron counts to pick a temporal bin that still respects the frame-aligned source data."

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Derived from the `position` behavior timestamps (which are the same across all behavior streams).

ii.
```python
timestamps = np.asarray(behavior["position"].timestamps[:], dtype=np.float64)
...
time_trial_s = (timestamps[start:stop] - timestamps[start]).astype(np.float32)
```

iii. The AI used the timestamps associated with the position time series.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The first timestamp in the trial is subtracted from all timestamps, then the values are rebinned by taking the left edge of each 16-frame bin.

ii.
```python
time_trial_s = (timestamps[start:stop] - timestamps[start]).astype(np.float32)
...
time_binned[bin_idx] = time_trial_s[start]  # left edge of bin
```

iii. From trajectory step 106: "One small semantic fix is still worth making: the first time bin currently starts around 0.5 s because I stored bin means. I'm switching time_from_trial_start to the left edge of each bin so the aligned trial truly starts at 0."

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Neural and behavioral data share the same frame indexing in the NWB files. After rebinning, the time values correspond to the same bins as the neural data.

ii.
```python
# Both use the same start:stop indices
neural_trial = deconv[start:stop].T
time_trial_s = (timestamps[start:stop] - timestamps[start]).astype(np.float32)
# Then both go through the same binning
```

iii. The AI verified that behavior and neural data have aligned timestamps at the same sampling rate.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Derived from the `environment` behavior time series.

ii.
```python
env = np.asarray(behavior["environment"].data[:], dtype=np.float32)
...
env_trial = env[start:stop]
env_trial = env_trial[env_trial >= 0]
env_value = float(np.round(np.nanmedian(env_trial))) if len(env_trial) else float(parse_scene(scene)[0][-1] == "2")
```

iii. The AI uses the median of valid environment values within the trial. Falls back to parsing the scene name if no valid values exist.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. Within each trial, negative environment values are filtered out, then the median of remaining values is rounded to get 0 or 1. If no valid values exist, the scene name is parsed as fallback. The value is constant across all timepoints in the trial.

ii.
```python
env_trial = env[start:stop]
env_trial = env_trial[env_trial >= 0]
env_value = float(np.round(np.nanmedian(env_trial))) if len(env_trial) else float(parse_scene(scene)[0][-1] == "2")
...
np.full(ntime, env_value, dtype=np.float32),
```

iii. The AI handled edge cases where the environment variable might have invalid values.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Derived from the `trial number` behavior time series in the NWB file.

ii.
```python
trial_number = np.asarray(behavior["trial number"].data[:], dtype=np.float32)
...
trialnum_trial = trial_number[start:stop]
trialnum_trial = trialnum_trial[trialnum_trial >= 0]
trialnum_value = float(np.round(np.nanmedian(trialnum_trial))) if len(trialnum_trial) else float(trial_idx)
```

iii. The AI chose to use the stored trial number from the NWB file rather than the loop counter.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. Within each trial, negative values are filtered out, then the median of remaining values is rounded. Falls back to the loop index if no valid values. The value is constant across all timepoints.

ii.
```python
trialnum_trial = trial_number[start:stop]
trialnum_trial = trialnum_trial[trialnum_trial >= 0]
trialnum_value = float(np.round(np.nanmedian(trialnum_trial))) if len(trialnum_trial) else float(trial_idx)
...
np.full(ntime, trialnum_value, dtype=np.float32),
```

iii. The AI used the NWB-stored trial number, taking care to handle potential negative/invalid values.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from the `Reward` behavior time series timestamps.

ii.
```python
reward_timestamps = np.asarray(behavior["Reward"].timestamps[:], dtype=np.float64)
...
trial_reward_ts = reward_timestamps[
    (reward_timestamps >= timestamps[start]) & (reward_timestamps < timestamps[stop])
]
reward_outcomes[trial_idx] = int(len(trial_reward_ts) > 0)
```

iii. The AI checks whether any reward event timestamps fall within each trial's time window.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For each trial, check whether any reward timestamps fall within the trial's time window. The previous trial's outcome is then shifted: `prev_reward_outcomes[1:] = reward_outcomes[:-1]`, with the first trial set to 0.

ii.
```python
reward_outcomes = np.zeros(ntrials, dtype=np.int16)
for trial_idx, (start, stop) in enumerate(zip(trial_start, teleport)):
    trial_reward_ts = reward_timestamps[
        (reward_timestamps >= timestamps[start]) & (reward_timestamps < timestamps[stop])
    ]
    reward_outcomes[trial_idx] = int(len(trial_reward_ts) > 0)
prev_reward_outcomes = np.zeros(ntrials, dtype=np.int16)
prev_reward_outcomes[1:] = reward_outcomes[:-1]
```

iii. The AI uses a vectorized shift to assign previous trial outcomes, with the first trial defaulting to 0.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from the `position` behavior time series and the reward zone label for each trial. Reward zone labels are determined from the `reward_zone` behavior time series (when active) or from the session scene name (as fallback).

ii.
```python
position = np.asarray(behavior["position"].data[:], dtype=np.float32)
reward_zone = np.asarray(behavior["reward_zone"].data[:], dtype=np.float32)
...
rz_mask = reward_zone[start:stop] > 0
if np.any(rz_mask):
    zone_labels.append(zone_from_position(float(np.nanmedian(position[start:stop][rz_mask]))))
else:
    zone_labels.append(expected_zone_labels[trial_idx])
```

iii. The AI uses median position when `reward_zone` is active to find the closest known reward zone center. When no reward zone is active, it falls back to the expected zone from the session scene name and trial index.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. For each timepoint, compute signed distance from position to reward zone edges: negative if before zone, positive if after, 0 if inside.

ii.
```python
def reward_zone_distance(position_cm, zone_label):
    start_cm, end_cm = RZ_BOUNDS[zone_label]
    dist = np.zeros(position_cm.shape, dtype=np.float32)
    before = position_cm < start_cm
    after = position_cm > end_cm
    dist[before] = position_cm[before] - start_cm
    dist[after] = position_cm[after] - end_cm
    return dist
```

iii. This matches the standard signed distance computation used in the reference.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Discretized into 7 categories using explicit conditional logic matching the instruction bins.

ii.
```python
def discretize_distance(distance_cm):
    out = np.full(distance_cm.shape, 6, dtype=np.int16)
    out[distance_cm < -50.0] = 0
    out[(distance_cm >= -50.0) & (distance_cm < -10.0)] = 1
    out[(distance_cm >= -10.0) & (distance_cm < 0.0)] = 2
    out[distance_cm == 0.0] = 3
    out[(distance_cm > 0.0) & (distance_cm <= 10.0)] = 4
    out[(distance_cm > 10.0) & (distance_cm <= 50.0)] = 5
    out[distance_cm > 50.0] = 6
    return out
```

iii. The bin edges match the instructions.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position data uses the same frame indices as neural data within each trial. After rebinning (averaging position within 16-frame bins), the discretized distance is computed on the binned positions.

ii.
```python
pos_binned[bin_idx] = np.mean(pos_trial_cm[sl], dtype=np.float64)
...
distance_cm = reward_zone_distance(pos_binned, zone_label)
```

iii. Alignment is maintained by using the same binning scheme for all data streams.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from the `position` behavior time series.

ii.
```python
position = np.asarray(behavior["position"].data[:], dtype=np.float32)
...
pos_trial_cm = position[start:stop]
```

iii. Direct use of the position variable from the NWB file.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Position is averaged within each 16-frame bin, then discretized into 5 bins.

ii.
```python
pos_binned[bin_idx] = np.mean(pos_trial_cm[sl], dtype=np.float64)
```

iii. The rebinning averages position values within each temporal bin.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Discretized into 5 bins with edges at 90, 180, 270, 360 cm.

ii.
```python
def discretize_position(position_cm):
    out = np.zeros(position_cm.shape, dtype=np.int16)
    out[(position_cm >= 90.0) & (position_cm < 180.0)] = 1
    out[(position_cm >= 180.0) & (position_cm < 270.0)] = 2
    out[(position_cm >= 270.0) & (position_cm < 360.0)] = 3
    out[position_cm >= 360.0] = 4
    return out
```

iii. Five equal-sized bins spanning the 450 cm track.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same binning as neural data (16-frame bins with averaging).

ii. Same as 7-d: `pos_binned[bin_idx] = np.mean(pos_trial_cm[sl], dtype=np.float64)`

iii. Aligned through shared binning scheme.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from the `lick` behavior time series.

ii.
```python
lick = np.asarray(behavior["lick"].data[:], dtype=np.float32)
...
lick_trial = lick[start:stop]
```

iii. Direct use of the lick variable from the NWB file.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Within each 16-frame bin, lick is binarized: 1 if any frame has lick > 0, else 0.

ii.
```python
lick_binned[bin_idx] = int(np.any(lick_trial[sl] > 0))
```

iii. Binary detection within each temporal bin.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same 16-frame binning as neural data.

ii. Same binning scheme as all other variables.

iii. Aligned through shared binning.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from `reward_zone` and `position` behavior time series. When `reward_zone > 0`, the median position is used to find the closest known reward zone center. Otherwise, the expected zone is parsed from the session scene name.

ii.
```python
rz_mask = reward_zone[start:stop] > 0
if np.any(rz_mask):
    zone_labels.append(zone_from_position(float(np.nanmedian(position[start:stop][rz_mask]))))
else:
    zone_labels.append(expected_zone_labels[trial_idx])
```

```python
def zone_from_position(position_cm):
    best_zone = min(RZ_CENTERS, key=lambda zone: abs(position_cm - RZ_CENTERS[zone]))
    return best_zone
```

iii. The AI uses a nearest-center approach to assign zones, with scene-based fallback.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. For trials with active reward zone, the median position during `reward_zone > 0` is computed and matched to the nearest reward zone center (A, B, or C). For trials without active reward zone, the expected zone is derived from the session scene name (which encodes environment and zone transitions) and the trial index relative to the switch trial (trial 30).

ii.
```python
def parse_scene(scene):
    match = re.fullmatch(r"(Env[12])_Location([ABC])", scene)
    ...
def expected_trial_labels(scene, ntrials):
    envs, zones = parse_scene(scene)
    if len(envs) == 1:
        env_by_trial = [envs[0]] * ntrials
        zone_by_trial = [zones[0]] * ntrials
    else:
        split = min(SWITCH_TRIAL, ntrials)
        env_by_trial = [envs[0]] * split + [envs[1]] * (ntrials - split)
        zone_by_trial = [zones[0]] * split + [zones[1]] * (ntrials - split)
    return env_by_trial, zone_by_trial
```

iii. The AI parsed the session scene format to handle reward zone switches during the session.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from the `Reward` behavior time series timestamps.

ii.
```python
reward_timestamps = np.asarray(behavior["Reward"].timestamps[:], dtype=np.float64)
...
trial_reward_ts = reward_timestamps[
    (reward_timestamps >= timestamps[start]) & (reward_timestamps < timestamps[stop])
]
reward_outcomes[trial_idx] = int(len(trial_reward_ts) > 0)
```

iii. Same reward detection as used for previous trial outcome.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, check if any reward timestamps fall within the trial's time window. Binary: 1 if rewarded, 0 if not. Constant across all timepoints in the trial.

ii.
```python
reward_outcomes[trial_idx] = int(len(trial_reward_ts) > 0)
...
np.full(ntime, reward_outcome_value, dtype=np.int16),
```

iii. Straightforward binary reward detection per trial.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several handling strategies:
- **Lick sensor errors**: Trials where >35% of frames have lick > 2 are dropped entirely.
- **Missing reward zone**: When `reward_zone` is never active in a trial, the expected zone from the scene name is used as fallback.
- **Missing environment/trial number**: Negative values are filtered out; median of remaining values used; falls back to scene parsing or loop index.
- **Sessions with too few trials**: Sessions with <2 kept trials are excluded.

ii.
```python
if np.mean(trial_lick > 2) > LICK_ERROR_THRESHOLD:
    drop_trial[trial_idx] = True
...
if session["n_trials_kept"] < 2:
    continue
...
env_trial = env_trial[env_trial >= 0]
env_value = float(np.round(np.nanmedian(env_trial))) if len(env_trial) else float(parse_scene(scene)[0][-1] == "2")
```

iii. From trajectory step 65: "probably dropping the small number of lick-corrupted trials rather than fabricating lick labels."

## 13-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are:
1. **Loading NWB files** twice: once to extract subject IDs, once to convert data
2. **Reading large neural arrays** from each NWB file (deconvolved data)
3. **Temporal rebinning** within each trial (though this is relatively fast)
4. **Saving the pickle file**

ii. N/A

iii. The AI noted in step 82 that "most of that time is spent streaming all 152 NWB files."

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial binning loop in `bin_trial` iterates over each bin sequentially. This could be vectorized using `np.reshape` or similar operations. The per-trial loop in `convert_session` could potentially be partially vectorized for operations like discretization.

ii.
```python
for bin_idx in range(nbins):
    start = bin_idx * BIN_FRAMES
    stop = min((bin_idx + 1) * BIN_FRAMES, nframes)
    sl = slice(start, stop)
    neural_binned[:, bin_idx] = neural_trial[:, sl].sum(axis=1, dtype=np.float32)
    ...
```

iii. The binning loop processes each bin individually, which could be replaced with array reshaping operations.

## 13-c. What processing does the code repeat multiple times?

i. The code opens each NWB file twice: once during subject ID collection (the initial loop to build `subjects` list) and once during actual conversion. The subject ID extraction loop reads every file just to get subject IDs.

ii.
```python
# First pass: get subjects
for path in session_paths:
    with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
        subject = io.read().subject.subject_id
...
# Second pass: convert
for path in session_paths:
    session = convert_session(path, subject_to_idx)
```

iii. Opening NWB files is expensive I/O; doing it twice for each file is wasteful.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The temporal rebinning (16x downsampling) discards temporal resolution that the native data provides. While the AI justified this as necessary for training, the reference solution shows the decoder can work at native resolution. The AI also computes `speed` as an output, which matches the instructions but adds processing for speed discretization and binning.

ii.
```python
BIN_FRAMES = 16
...
neural_binned[:, bin_idx] = neural_trial[:, sl].sum(axis=1, dtype=np.float32)
```

iii. The rebinning was motivated by dataset size concerns but may not have been necessary.
