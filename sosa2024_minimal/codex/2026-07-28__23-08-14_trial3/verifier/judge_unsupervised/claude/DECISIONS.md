# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads NWB files from the data directory by globbing for `sub-*/sub-*_behavior+ophys.nwb`. Each NWB file represents one session and contains both behavioral timeseries and optical physiology data. The AI opens each file with h5py, reads behavioral channels from `processing/behavior/BehavioralTimeSeries/` and neural data from `processing/ophys/`. All 152 NWB files across 11 subjects are processed sequentially.

ii.
```python
def nwb_files(data_dir: str):
    paths = sorted(Path(data_dir).glob("sub-*/sub-*_behavior+ophys.nwb"))
    if not paths:
        raise FileNotFoundError(f"No NWB files found under {data_dir}")
    return paths
```
```python
for idx, path in enumerate(files, start=1):
    record = convert_session(path, lick_error_threshold=args.lick_error_threshold)
    session_records.append(record)
```

iii. The AI noted that NWB files are the standard data format for this dataset. Each file is one session for one subject. The glob pattern matches the directory structure under `/app/data/`.

## 1-b. How are the data split into subjects (mice)?

i. Subject identity is parsed from the parent directory name of each NWB file (e.g., `sub-m3` -> `m3`). A sorted list of unique subjects is built, and each session is mapped to its subject via `subject_idx`.

ii.
```python
def parse_subject_and_day(path: Path):
    subject = path.parent.name.replace("sub-", "")
    match = re.search(r"ses-(\d+)", path.name)
    ...
    return subject, int(match.group(1))
```
```python
subjects = sorted({record["subject"] for record in session_records})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
```

iii. The AI identified that subject identity is encoded in the directory structure (e.g., `sub-m3/`) and parsed it accordingly. This recovers 11 unique subjects consistent with the paper.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. Session day number is parsed from the filename (e.g., `ses-01` -> day 1). Sessions are sorted by file path, which groups them by subject and then by day.

ii.
```python
paths = sorted(Path(data_dir).glob("sub-*/sub-*_behavior+ophys.nwb"))
```
```python
subject, day = parse_subject_and_day(path)
```

iii. The AI noted that the NWB data structure uses one file per session. 152 sessions total were recovered, with m11 having 12 sessions (starting day 3) and all others having 14 sessions.

## 1-d. How are the data split into trials?

i. Trials are reconstructed from framewise signals: `trial_start` marks the beginning and `teleport` marks the end of each trial. The AI uses `np.flatnonzero(signal > 0)` on both channels to find frame indices. Each trial spans from trial_start (inclusive) to teleport (exclusive).

ii.
```python
trial_start_signal = beh["trial_start"]["data"][()]
teleport_signal = beh["teleport"]["data"][()]
trial_starts = np.flatnonzero(trial_start_signal > 0)
teleports = np.flatnonzero(teleport_signal > 0)
```
```python
for trial_idx, (start, stop) in enumerate(zip(trial_starts, teleports)):
    ...
    T = int(stop - start)
```

iii. The AI noted that the NWB files do not expose a ready-made trials table, so trials must be reconstructed from framewise channels. The CONVERSION_NOTES state: "Each trial uses frames from trial_start inclusive to teleport exclusive. This matches the reference preprocessing logic."

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered by three criteria: (1) lick sensor error — trials where more than 35% of frames have lick count > 2 are dropped; (2) zero-length trials (T <= 0) are dropped; (3) trials with unresolvable zone or environment labels are dropped.

ii.
```python
def drop_lick_error_trials(lick, trial_starts, teleports, threshold):
    ...
    for idx, (start, stop) in enumerate(zip(trial_starts, teleports)):
        lick_trial = lick[start:stop]
        frac = float(np.mean(lick_trial > 2))
        error_fraction[idx] = frac
        if frac > threshold:
            drop_mask[idx] = True
    return drop_mask, error_fraction
```
```python
if drop_mask[trial_idx]:
    dropped_trials.append(trial_idx)
    continue
T = int(stop - start)
if T <= 0:
    dropped_trials.append(trial_idx)
    continue
zone_label = zone_schedule["filled"][trial_idx]
env_label = env_schedule["filled"][trial_idx]
if zone_label is None or env_label is None:
    dropped_trials.append(trial_idx)
    continue
```

iii. The AI referenced the `correct_lick_sensor_error()` function from the reference code, which checks if a threshold fraction of frames have cumulative lick count > 2. The AI used a threshold of 0.35 (35%), noting it is "commonly used for day-level behavior analysis" in the reference code. The methods text states >30% as the threshold. The AI reports 69 trials dropped out of 12,216 total (~0.57%).

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the deconvolved calcium event time series at `processing/ophys/Deconvolved/plane0/data` in the NWB files.

ii.
```python
deconv = f["processing/ophys/Deconvolved/plane0/data"]
```

iii. The AI noted that the paper states analyses used deconvolved calcium event time series. The AI chose the `Deconvolved` series rather than raw fluorescence or dF/F.

## 2-b. How is the `neural` data processed?

i. The deconvolved data is sliced per trial (trial_start to teleport frames), and only columns corresponding to selected ROIs (filtered by iscell) are retained. The data is transposed to (n_neurons, n_timepoints) format and cast to float32. No additional temporal smoothing, rebinning, or normalization is applied.

ii.
```python
neural = deconv[start:stop, selected_indices].T.astype(np.float32)
```

iii. The AI stated: "No extra temporal smoothing or rebinning was added. The recovered bin size is constant across all sessions: 64.48362720402656 ms." This matches the paper's ~15.5 Hz imaging frame rate.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI filters neurons using two criteria: (1) only ROIs referenced by the `Fluorescence/plane0/rois` index are considered, and (2) only those ROIs where the `iscell` flag from `ImageSegmentation/PlaneSegmentation` is true are kept. The AI does NOT apply the additional interneuron exclusion (speed-correlation >0.5) described in the methods.

ii.
```python
rois = f["processing/ophys/Fluorescence/plane0/rois"][()]
iscell = f["processing/ophys/ImageSegmentation/PlaneSegmentation/iscell"][()][:, 0].astype(bool)
selected_mask = iscell[rois]
selected_indices = np.flatnonzero(selected_mask)
```

iii. The AI justified using only iscell by noting that neuron counts (155-1780) fell within the paper's reported range (155-2172). The CONVERSION_NOTES explain that for multi-plane mice m17 and m18, only the published plane0 response series is used, as segmentation tables contain rows from both planes but only plane0 data is available in NWB.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to trial start. Each trial's neural data spans from the `trial_start` frame index (inclusive) to the `teleport` frame index (exclusive), so the first time bin is the trial start frame.

ii.
```python
neural = deconv[start:stop, selected_indices].T.astype(np.float32)
```
(where `start` is the trial_start frame index and `stop` is the teleport frame index)

iii. The instructions specify "Temporally align based on start of the trial." The AI aligns by indexing from the trial_start frame, making t=0 the trial start.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is the native imaging frame rate, approximately 64.48 ms (~15.5 Hz). No temporal rebinning is applied.

ii.
```python
dt_sec = float(np.median(np.diff(times)))
```
```python
"time_bin_size": float(np.median(time_bin_sizes_ms)),
```

iii. The AI noted: "No extra temporal smoothing or rebinning was added. The recovered bin size is constant across all sessions: 64.48362720402656 ms." This matches the paper's reported imaging frame rate.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Derived from the `position/timestamps` array in the NWB behavioral timeseries.

ii.
```python
times = beh["position"]["timestamps"][()].astype(np.float64)
```

iii. The timestamps are shared across all behavioral channels (all sampled at the same imaging frame rate).

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. For each trial, the timestamps of frames within the trial are extracted and the trial start time is subtracted to produce time relative to trial start, in seconds.

ii.
```python
time_trial = (times[start:stop] - times[start]).astype(np.float32)
```

iii. This produces a time vector starting at 0.0 at trial start and increasing by ~0.0645 seconds per frame.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. The time vector uses the exact same frame indices (start:stop) as the neural data, so they are inherently aligned frame-by-frame.

ii.
```python
time_trial = (times[start:stop] - times[start]).astype(np.float32)
neural = deconv[start:stop, selected_indices].T.astype(np.float32)
```

iii. Both use the same trial_start to teleport frame window, ensuring perfect temporal alignment.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Derived from the `environment/data` channel in the NWB behavioral timeseries.

ii.
```python
environment = beh["environment"]["data"][()].astype(np.float32)
```

iii. The environment channel encodes ENV1 as 0 and ENV2 as 1.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. For each trial, the environment values within the trial window are extracted, negative values are filtered out, and the median is taken (rounded to integer). A schedule inference algorithm fills in missing values using a pre/post 30-trial switch rule. The filled value (0 or 1) is broadcast as a constant across all time bins in the trial.

ii.
```python
for start, stop in zip(trial_starts, teleports):
    env_trial = environment[start:stop]
    env_trial = env_trial[env_trial >= 0]
    observed_env.append(int(np.rint(np.nanmedian(env_trial))) if env_trial.size else None)
env_schedule = infer_constant_or_switch_schedule(observed_env)
```
```python
np.full(T, env_label, dtype=np.float32),
```

iii. The AI uses the 30-trial switch boundary from the reference code to fill in any missing environment observations.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Derived from the `trial number/data` channel in the NWB behavioral timeseries.

ii.
```python
trial_number = beh["trial number"]["data"][()].astype(np.float32)
```

iii. The trial number is recorded as a framewise signal in the NWB files.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. For each trial, the trial number values within the trial window are extracted, negative values are filtered out, and the median is taken. If no valid values exist, the trial index is used as fallback. The scalar value is broadcast across all time bins.

ii.
```python
trial_number_values = trial_number[start:stop]
trial_number_values = trial_number_values[trial_number_values >= 0]
trial_number_scalar = (
    float(np.nanmedian(trial_number_values))
    if trial_number_values.size
    else float(trial_idx)
)
```
```python
np.full(T, trial_number_scalar, dtype=np.float32),
```

iii. The AI reads the trial number from the recorded channel rather than assuming sequential indexing, to handle potential gaps or reordering.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from the `Reward/timestamps` channel in the NWB behavioral timeseries, which records the times of reward delivery events.

ii.
```python
reward_times = beh["Reward"]["timestamps"][()].astype(np.float64)
```

iii. Reward times are used to determine whether each trial received a reward, and the previous trial's outcome becomes the input.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For each trial, the code checks whether any reward timestamps fall within the trial's time window (trial_start to teleport). This produces a per-trial binary reward outcome (0=no, 1=yes). The previous trial outcome is then the reward outcome of the preceding trial, with the first trial in each session assigned 0.

ii.
```python
def reward_outcomes_per_trial(reward_times, times, trial_starts, teleports):
    outcomes = []
    for start, stop in zip(trial_starts, teleports):
        has_reward = np.any((reward_times >= times[start]) & (reward_times < times[stop]))
        outcomes.append(int(has_reward))
    return outcomes
```
```python
previous_reward_outcomes = [0] + reward_outcomes[:-1]
```

iii. The AI defines "previous trial outcome" as the reward outcome of the immediately preceding trial within the same session, with the first trial defaulting to 0 (no reward).

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from two sources: (1) `position/data` (animal's position on the track), and (2) the inferred reward zone identity for each trial (from the `reward_zone/data` channel and reward timestamps), which determines which zone boundaries to use.

ii.
```python
pos = beh["position"]["data"][()].astype(np.float32)
```
```python
zone_bounds_cm = ZONE_BOUNDS_CM[zone_label]
dist_trial = signed_distance_to_zone(pos_trial, zone_bounds_cm)
```

iii. The AI computes signed distance from the animal's position to the nearest edge of the trial's reward zone.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. The signed distance is computed as: (1) if position < zone_start, distance = position - zone_start (negative); (2) if position > zone_end, distance = position - zone_end (positive); (3) if inside the zone, distance = 0. The reward zone boundaries are defined as A: 80-130 cm, B: 200-250 cm, C: 320-370 cm.

ii.
```python
def signed_distance_to_zone(pos_cm, zone_bounds_cm):
    zone_start, zone_end = zone_bounds_cm
    dist = np.zeros_like(pos_cm, dtype=np.float32)
    before = pos_cm < zone_start
    after = pos_cm > zone_end
    dist[before] = pos_cm[before] - zone_start
    dist[after] = pos_cm[after] - zone_end
    return dist
```

iii. This produces a signed distance where negative means before the zone, zero means inside, and positive means past the zone.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The continuous signed distance is binned into 7 categories matching the instruction specification.

ii.
```python
def bin_distance_to_zone(dist_cm):
    out = np.zeros(dist_cm.shape, dtype=np.int64)
    out[dist_cm < -50.0] = 0
    out[(dist_cm >= -50.0) & (dist_cm < -10.0)] = 1
    out[(dist_cm >= -10.0) & (dist_cm < 0.0)] = 2
    out[dist_cm == 0.0] = 3
    out[(dist_cm > 0.0) & (dist_cm <= 10.0)] = 4
    out[(dist_cm > 10.0) & (dist_cm <= 50.0)] = 5
    out[dist_cm > 50.0] = 6
    return out
```

iii. The 7 bins match the instruction specification exactly.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. The distance is computed from the same position frames (trial_start to teleport) used for neural data, so alignment is frame-by-frame.

ii.
```python
pos_trial = np.clip(pos[start:stop], 0.0, 450.0).astype(np.float32)
dist_trial = signed_distance_to_zone(pos_trial, zone_bounds_cm)
```

iii. Same frame indices as neural data ensure temporal alignment.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from `position/data` in the NWB behavioral timeseries.

ii.
```python
pos = beh["position"]["data"][()].astype(np.float32)
pos_trial = np.clip(pos[start:stop], 0.0, 450.0).astype(np.float32)
```

iii. Position is the animal's location on the 450 cm virtual linear track.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Position is clipped to [0, 450] cm range and then discretized into 5 equal-sized bins of 90 cm each.

ii.
```python
def bin_absolute_position(pos_cm):
    pos_clipped = np.clip(pos_cm, 0.0, 450.0 - 1e-6)
    return np.minimum((pos_clipped / 90.0).astype(np.int64), 4)
```

iii. The 5 bins span 0-90, 90-180, 180-270, 270-360, and 360-450 cm, matching the instruction to discretize into "5 equal-sized bins."

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Position is divided by 90 cm and truncated to integer to produce 5 bins (0-4). Values at exactly 450 cm are clamped to bin 4.

ii.
```python
pos_clipped = np.clip(pos_cm, 0.0, 450.0 - 1e-6)
return np.minimum((pos_clipped / 90.0).astype(np.int64), 4)
```

iii. Output values are labeled: "0 to <90 cm", "90 to <180 cm", "180 to <270 cm", "270 to <360 cm", "360 to 450 cm".

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Position uses the same frame indices (trial_start to teleport) as neural data, providing frame-by-frame alignment.

ii.
```python
pos_trial = np.clip(pos[start:stop], 0.0, 450.0).astype(np.float32)
```

iii. Same trial window as neural data.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from the `lick/data` channel in the NWB behavioral timeseries.

ii.
```python
lick = beh["lick"]["data"][()].astype(np.float32)
```

iii. The lick channel records cumulative lick counts per frame.

## 9-b. What processing is involved in computing `output` *Lick*?

i. The lick signal is binarized: any frame with lick > 0 is set to 1, otherwise 0.

ii.
```python
lick_trial = (lick[start:stop] > 0).astype(np.int64)
```

iii. The instruction specifies lick as binary (0=no, 1=yes). The AI converts the cumulative lick count to a binary indicator.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Lick uses the same frame indices (trial_start to teleport) as neural data.

ii.
```python
lick_trial = (lick[start:stop] > 0).astype(np.int64)
```

iii. Same trial window ensures alignment.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from the `reward_zone/data` channel, `position/data`, and `Reward/timestamps`. The reward zone signal indicates when the animal is in the reward zone, and position is used to determine which zone (A, B, or C).

ii.
```python
reward_zone_signal = beh["reward_zone"]["data"][()].astype(np.float32)
```
```python
def infer_trial_zone_positions(pos, reward_zone_signal, times, reward_times, trial_starts, teleports):
    ...
    zone_frames = np.flatnonzero(zone_signal_trial > 0)
    zone_position_cm = np.nan
    if zone_frames.size > 0:
        zone_position_cm = float(np.nanmedian(pos_trial[zone_frames]))
    else:
        in_trial_reward = reward_times[...]
        ...
```

iii. The AI infers the zone by finding the median position when the reward_zone signal is active, then maps it to the nearest zone center (A=105, B=225, C=345).

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. For each trial: (1) find frames where reward_zone > 0 and take median position; (2) if no reward_zone frames, use position at reward delivery time; (3) map to nearest zone center to get label A/B/C. Missing labels are filled using a pre/post 30-trial switch schedule. The zone label is encoded as 0=A, 1=B, 2=C and broadcast across all time bins.

ii.
```python
def infer_zone_from_position(zone_position_cm):
    ...
    return min(ZONE_CENTERS_CM, key=lambda label: abs(zone_position_cm - ZONE_CENTERS_CM[label]))

zone_schedule = infer_constant_or_switch_schedule(observed_zone_labels)
```
```python
np.full(T, ZONE_TO_INDEX[zone_label], dtype=np.int64),
```

iii. The 30-trial switch boundary matches the reference code's approach to handling reward zone switches mid-session.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from `Reward/timestamps` in the NWB behavioral timeseries.

ii.
```python
reward_times = beh["Reward"]["timestamps"][()].astype(np.float64)
```

iii. Reward timestamps record when reward was delivered during the session.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, the code checks if any reward timestamp falls within the trial's time window. If yes, reward outcome = 1; otherwise = 0. The value is broadcast across all time bins.

ii.
```python
def reward_outcomes_per_trial(reward_times, times, trial_starts, teleports):
    outcomes = []
    for start, stop in zip(trial_starts, teleports):
        has_reward = np.any((reward_times >= times[start]) & (reward_times < times[stop]))
        outcomes.append(int(has_reward))
    return outcomes
```
```python
np.full(T, reward_outcomes[trial_idx], dtype=np.int64),
```

iii. Simple binary determination based on whether a reward event occurred during the trial.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several strategies are used: (1) Lick sensor error trials (>35% frames with lick>2) are dropped entirely. (2) Missing reward zone labels are filled via the 30-trial switch schedule inference. (3) Missing environment labels are filled the same way. (4) Negative trial number values are filtered before taking median. (5) Trials with unresolvable zone or environment labels (both None) are dropped. (6) Position is clipped to [0, 450] range. (7) Trial start/teleport count mismatches raise an error.

ii.
```python
if trial_starts.size != teleports.size:
    raise ValueError(...)
```
```python
trial_number_values = trial_number_values[trial_number_values >= 0]
```
```python
if zone_label is None or env_label is None:
    dropped_trials.append(trial_idx)
    continue
```

iii. The AI documented these decisions in CONVERSION_NOTES.md, noting that dropping lick-error trials is the main quality filter. The schedule-filling approach handles omission trials where zone/environment may not be directly observed.

## 13-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is reading the deconvolved neural data from NWB files. For each trial, a large HDF5 dataset is sliced (`deconv[start:stop, selected_indices]`). This involves random-access reads from HDF5 files for each of the ~80 trials per session across 152 sessions.

ii.
```python
neural = deconv[start:stop, selected_indices].T.astype(np.float32)
```

iii. HDF5 column-wise slicing with a non-contiguous index array (selected_indices) is inherently slow due to the storage layout.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `convert_session` iterates over all trials sequentially, performing slicing, binning, and stacking operations that could partially be vectorized. The `infer_trial_zone_positions` and `reward_outcomes_per_trial` functions also loop over trials when vectorized approaches using `np.searchsorted` or boolean indexing on the full arrays could be more efficient.

ii.
```python
for trial_idx, (start, stop) in enumerate(zip(trial_starts, teleports)):
    ...
    pos_trial = np.clip(pos[start:stop], 0.0, 450.0)
    speed_trial = speed[start:stop]
    lick_trial = (lick[start:stop] > 0).astype(np.int64)
    ...
```

iii. The trial-by-trial loop structure is straightforward but prevents batch processing of behavioral channels across the full session.

## 13-c. What processing does the code repeat multiple times?

i. The code reads the full behavioral timeseries arrays (position, speed, lick, etc.) once per session but then slices them repeatedly per trial. The `infer_trial_zone_positions` function processes the same position and reward_zone arrays that are also sliced again in the main trial loop. Environment inference similarly processes the environment array once in a pre-loop and again implicitly when building inputs.

ii.
```python
# First pass: infer zone positions
observed_zone_labels, observed_zone_positions = infer_trial_zone_positions(pos=pos, ...)
# Second pass: per-trial extraction
for trial_idx, (start, stop) in enumerate(zip(trial_starts, teleports)):
    pos_trial = np.clip(pos[start:stop], ...)
```

iii. The zone/environment inference requires a first pass to determine the schedule before the main trial-extraction loop, so some repeated access is structurally necessary.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes and stores extensive session metadata (observed zone positions, all reward outcomes, lick error fractions per trial, plane indices, etc.) that is not used by the decoder. The schedule inference algorithm (`infer_constant_or_switch_schedule`) computes detailed pre/post/overall mode statistics that are only used for metadata logging. The `summarize_dataset` and `print_subject_schedules` functions produce console output that doesn't affect the saved data.

ii.
```python
session_info["observed_zone_positions_cm"] = observed_zone_positions
session_info["observed_zone_labels"] = observed_zone_labels
session_info["lick_error_fraction_per_trial"] = lick_error_fraction.tolist()
session_info["selected_plane_indices"] = sorted(np.unique(plane_idx).astype(int).tolist())
```

iii. While this metadata is useful for validation and debugging, it adds to the pickle file size and processing time without being consumed by the decoder training pipeline.
