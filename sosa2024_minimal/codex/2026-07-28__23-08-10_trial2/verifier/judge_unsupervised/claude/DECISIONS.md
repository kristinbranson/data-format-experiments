# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads NWB files from `/app/data/sub-*/sub-*_behavior+ophys.nwb` using `h5py`. It scans all subdirectories matching the `sub-*` pattern, opens each NWB file, extracts subject ID and session metadata (scene, date, session number) from the file's internal fields, then processes each session individually via `process_session()`. Behavioral time series (position, speed, lick, environment, trial number, trial_start, teleport, reward timestamps) and optical physiology data (fluorescence, neuropil, ROI segmentation) are read from the HDF5 groups.

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

iii. The agent inspected the NWB file structure using h5py and pynwb, confirmed the data layout, then built a loader that reads directly from HDF5 groups. The agent noted 152 sessions (2 missing from m11) out of the expected 154.

## 1-b. How are the data split into subjects (mice)?

i. Each NWB file contains a `general/subject/subject_id` field encoding the mouse ID (e.g., "m3", "m11"). The agent extracts unique subject IDs, sorts them numerically, and builds a `subject_to_idx` mapping. Each session is assigned a `subject_idx` based on which mouse it belongs to.

ii.
```python
subjects = sorted({meta.subject for meta in session_meta}, key=lambda s: int(s[1:]))
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
# ...
dataset["subject_idx"].append(subject_to_idx[meta.subject])
```

iii. From CONVERSION_NOTES: "The final full dataset therefore contains 152 sessions across 11 subjects."

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. Sessions are sorted by (subject_number, session_number). Each session is processed independently and appended to the dataset lists.

ii.
```python
for meta in session_meta:
    session_data, stats = process_session(meta, block_size=block_size)
    dataset["neural"].append(session_data["neural"])
    dataset["input"].append(session_data["input"])
    dataset["output"].append(session_data["output"])
```

iii. The agent's trajectory shows bulk scanning of all 152 NWB files to confirm session/trial counts before writing the converter.

## 1-d. How are the data split into trials?

i. Trials are defined by the `trial_start` and `teleport` binary time series in the NWB behavioral data. Trial start indices are frames where `trial_start > 0`, and trial end indices are frames where `teleport > 0`. Each trial spans `[trial_start_idx, teleport_idx)` — the teleport frame is excluded.

ii.
```python
trial_start_idx = np.where(trial_start_series > 0)[0]
trial_stop_idx = np.where(teleport_series > 0)[0]
# ...
start = trial_start_idx[trial_idx]
stop = trial_stop_idx[trial_idx]
# Neural: kept_events[:, start:stop]
# Behavioral: position[start:stop], speed[start:stop], etc.
```

iii. From CONVERSION_NOTES: "For each trial, the kept sample range is `[trial_start_idx : teleport_idx)`. The teleport sample itself is excluded." This matches the reference code's trial segmentation approach.

## 1-e. How are trials filtered based on quality controls?

i. Trials are excluded if more than 30% of their frame samples have a cumulative lick count > 2, matching the paper's lick-sensor error criterion. The filtering is applied after computing reward outcomes (so previous-trial outcomes reflect the actual sequence even if a trial is later removed).

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

iii. From CONVERSION_NOTES: "Trials were dropped when >30% of frame samples had lick counts >2, matching the paper's lick-sensor error criterion." The agent obtained 81 removed trials, matching the paper's reported number exactly.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the raw fluorescence (`Fluorescence`) and neuropil fluorescence (`Neuropil`) time series stored per plane in the NWB ophys processing module, along with ROI curation metadata (`iscell` from `ImageSegmentation/PlaneSegmentation`).

ii.
```python
fluorescence_ds = ophys["Fluorescence"][plane_key]["data"]
neuropil_ds = ophys["Neuropil"][plane_key]["data"]
iscell = imgseg["iscell"][:]
```

iii. From CONVERSION_NOTES: "The saved neural activity is trial-wise OASIS event activity recomputed from Suite2p fluorescence and neuropil, rather than taking the NWB `Deconvolved` array directly."

## 2-b. How is the `neural` data processed?

i. The processing pipeline replicates the paper's procedure:
1. Keep only curated ROIs with `iscell[:, 0] == 1`
2. Neuropil subtraction with coefficient 0.7, adding back per-trial neuropil mean
3. Per-trial maximin baseline: Gaussian smooth (sigma=15), minimum filter (size=300), maximum filter (size=300)
4. dF/F = (F - baseline) / abs(baseline)
5. Gaussian smoothing of dF/F with sigma=2 samples
6. OASIS deconvolution with tau=0.7 and batch size 2000

ii.
```python
def compute_dff_and_events(fluorescence, neuropil, start_idx, stop_idx, frame_rate_hz,
                           neu_coef=0.7, tau=0.7):
    f -= neu_coef * f_neu
    for start, stop in zip(start_idx, stop_idx):
        f[:, sl] = f[:, sl] + neu_coef * np.nanmean(f_neu[:, sl], axis=1, keepdims=True)
        baseline[:, sl] = nansmooth_2d(f[:, sl], sigma=15)
        baseline[:, sl] = minimum_filter1d(baseline[:, sl], size=300, axis=1, mode="nearest")
        baseline[:, sl] = maximum_filter1d(baseline[:, sl], size=300, axis=1, mode="nearest")
    dff[:, valid] = (f[:, valid] - baseline[:, valid]) / np.abs(baseline[:, valid])
    for start, stop in zip(start_idx, stop_idx):
        dff[:, sl] = nansmooth_2d(dff[:, sl], sigma=2)
        events[:, sl] = dcnv.oasis(dff[:, sl], 2000, tau, frame_rate_hz)
    return dff, events
```

iii. From CONVERSION_NOTES: The agent documented each step matching the paper's description. The processing uses the same parameters (neu_coef=0.7, maximin with sigma=15/300/300, Gaussian sigma=2, OASIS with tau=0.7) as the reference code.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filtering steps:
1. ROI curation: Only cells with `iscell[:, 0] == 1` are kept (Suite2p manual curation).
2. Interneuron exclusion: Cells with Pearson correlation > 0.5 between their dF/F trace and running speed are excluded as putative interneurons.

ii.
```python
plane_iscell = iscell[plane_slice, 0] == 1
curated_plane_indices = np.where(plane_iscell)[0]
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

iii. From CONVERSION_NOTES: "Putative interneurons were excluded when Pearson correlation between dF/F and running speed exceeded 0.5... This follows the paper's description." The agent reports 0.360% ± 0.615% exclusion, close to the paper's 0.42% ± 0.85%.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to trial start. Each trial's neural data spans from the trial start frame to the teleport frame (exclusive). The first time bin of each trial corresponds to the trial start event.

ii.
```python
start = trial_start_idx[trial_idx]
stop = trial_stop_idx[trial_idx]
session_trial_blocks[out_idx].append(
    kept_events[:, start:stop].astype(np.float16, copy=False)
)
```

iii. From CONVERSION_NOTES: "Trials are aligned to trial start." The metadata specifies `"temporal_alignment_event": "trial start"` and `"off_start": 0.0`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is the native imaging frame rate of ~15.5 Hz, yielding a time bin of ~64.48 ms. No temporal rebinning is applied. The frame rate is inferred from the median inter-timestamp interval of the behavioral timestamps.

ii.
```python
def infer_frame_rate(position_timestamps):
    diffs = np.diff(position_timestamps.astype(np.float64))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    return float(1.0 / np.median(diffs))
# ...
mean_frame_rate = float(np.mean([stats["frame_rate_hz"] for stats in session_stats]))
dataset["metadata"]["time_bin_size"] = 1000.0 / mean_frame_rate
```

iii. From CONVERSION_NOTES: "Frame rate: 15.507813 ± 0.000000 Hz, so metadata['time_bin_size'] is 64.4836 ms." The std of 0 indicates all sessions have the same effective frame rate.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Derived from the `position` timestamps in the behavioral time series. Specifically, the timestamps array corresponding to the position data.

ii.
```python
timestamps = behavior["position"]["timestamps"][:].astype(np.float64)
# ...
trial_times = (timestamps[start:stop] - timestamps[start]).astype(np.float32)
```

iii. The agent uses the NWB position timestamps, subtracting the timestamp at trial start to get relative time.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. For each trial, the timestamps within the trial window are extracted and the trial start timestamp is subtracted, yielding time in seconds from trial onset. No further processing or scaling.

ii.
```python
trial_times = (timestamps[start:stop] - timestamps[start]).astype(np.float32)
```

iii. Straightforward subtraction. The result is a continuous time vector starting at 0 for each trial.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Both use the same frame indices (`start:stop` from `trial_start_idx` and `trial_stop_idx`), so they are inherently aligned frame-by-frame.

ii.
```python
# Neural: kept_events[:, start:stop]
# Input time: timestamps[start:stop] - timestamps[start]
trial_input = np.vstack([trial_times, ...])
```

iii. Same indexing ensures temporal alignment.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Primarily derived from the session `scene` name (e.g., "Env1_LocationA", "Env2_B_to_Env1_A") parsed via regex. Cross-checked against the `environment` behavioral time series in the NWB file.

ii.
```python
scene = parts[-1]  # from NWB identifier
# ...
trial_labels = parse_scene(session_meta.scene, len(trial_start_idx))
# ...
np.full(trial_times.shape, float(label.env), dtype=np.float32)
```

iii. From CONVERSION_NOTES: "Trial environment and reward-zone labels are derived from the session scene name... The scene-derived environment was cross-checked against the aligned behavior environment stream on every kept trial."

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The scene name is parsed with regex patterns to extract environment number. Environment numbers are 0-indexed (Env1 → 0, Env2 → 1). For switch sessions, the first 30 trials use the pre-switch environment, subsequent trials use the post-switch environment. The value is constant across all timepoints within a trial.

ii.
```python
def parse_scene(scene, n_trials, switch_trial_count=30):
    match = SCENE_LOCATION_RE.match(scene)
    if match:
        env = int(match.group("env")) - 1
        return [TrialLabel(env=env, zone=zone) for _ in range(n_trials)]
    # ... handles switch patterns similarly
```

iii. The agent validates by comparing scene-derived environment against the NWB `environment` time series, raising an error on mismatch.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Derived from the `trial number` behavioral time series in the NWB file.

ii.
```python
trial_number_series = behavior["trial number"]["data"][:].astype(np.float32)
# ...
np.full(trial_times.shape, trial_number_series[start], dtype=np.float32)
```

iii. The agent reads the trial number time series and samples its value at the trial start frame.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The value of the `trial number` time series at the trial start frame is extracted and broadcast across all timepoints in the trial. No further processing.

ii.
```python
np.full(trial_times.shape, trial_number_series[start], dtype=np.float32)
```

iii. Simple extraction and broadcasting.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from the `Reward` timestamps in the behavioral time series, combined with the trial start/stop times.

ii.
```python
reward_timestamps = behavior["Reward"]["timestamps"][:].astype(np.float64)
reward_outcomes = reward_outcome_per_trial(reward_timestamps, timestamps[trial_start_idx], timestamps[trial_stop_idx])
prev_reward_outcomes = np.concatenate([[0], reward_outcomes[:-1]]).astype(np.float32)
```

iii. From CONVERSION_NOTES: "previous_trial_outcome is 0 for the first trial of a session." Reward outcomes are computed before trial exclusion, so previous trial outcome reflects the actual previous trial regardless of whether it was later excluded.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. First, reward outcome per trial is computed by checking if any reward timestamp falls within `[trial_start_time, trial_stop_time)`. Then, the previous trial's outcome is obtained by shifting the array by 1 position, with the first trial getting 0. The value is broadcast across all timepoints in the trial.

ii.
```python
def reward_outcome_per_trial(reward_timestamps, start_times, stop_times):
    outcomes = np.zeros(len(start_times), dtype=np.int8)
    for trial_idx, (start_time, stop_time) in enumerate(zip(start_times, stop_times)):
        # Check if any reward timestamp falls within [start, stop)
        ...
    return outcomes
prev_reward_outcomes = np.concatenate([[0], reward_outcomes[:-1]]).astype(np.float32)
```

iii. The shift-by-one approach correctly captures "previous trial outcome" as specified in the instructions.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from the `position` behavioral time series and the reward zone boundaries (determined from the session scene name). Zone boundaries are hardcoded: A=(80, 130), B=(200, 250), C=(320, 370) in cm.

ii.
```python
ZONE_TO_BOUNDS_CM = {
    "A": (80.0, 130.0),
    "B": (200.0, 250.0),
    "C": (320.0, 370.0),
}
# ...
trial_pos = np.clip(position[start:stop], 0.0, 450.0).astype(np.float32)
zone_bounds = ZONE_TO_BOUNDS_CM[label.zone]
discretize_distance_to_zone(trial_pos, zone_bounds)
```

iii. Zone bounds match the paper's description of 50 cm reward zones at specific track locations.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance is computed as: before zone → `position - zone_start` (negative); inside zone → 0; after zone → `position - zone_stop` (positive). Position is clipped to [0, 450] cm first.

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

iii. This computes the minimum signed distance to the nearest edge of the reward zone, with 0 inside the zone.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Seven bins as specified in the instructions: 0 (< -50), 1 (-50 to -10), 2 (-10 to <0), 3 (0, in zone), 4 (>0 to 10), 5 (10 to 50), 6 (>50).

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

iii. Matches the 7-category discretization specified in the instructions.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Same frame indices (`start:stop`) are used for both neural and position data, ensuring frame-by-frame alignment.

ii.
```python
trial_pos = np.clip(position[start:stop], 0.0, 450.0)
# neural: kept_events[:, start:stop]
```

iii. Inherent alignment through shared indexing.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from the `position` behavioral time series in the NWB file.

ii.
```python
position = behavior["position"]["data"][:].astype(np.float32)
trial_pos = np.clip(position[start:stop], 0.0, 450.0).astype(np.float32)
```

iii. The raw position data in cm along the 450 cm corridor.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Position is clipped to [0, 450] cm (matching the 450 cm corridor length), then discretized.

ii.
```python
trial_pos = np.clip(position[start:stop], 0.0, 450.0).astype(np.float32)
discretize_absolute_position(trial_pos)
```

iii. Minimal processing — just clipping to corridor bounds.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Five equal bins of 90 cm each: 0 (0-90), 1 (90-180), 2 (180-270), 3 (270-360), 4 (360-450).

ii.
```python
def discretize_absolute_position(position_cm):
    clipped = np.clip(position_cm, 0.0, 449.999999)
    bins = np.floor(clipped / 90.0).astype(np.int8)
    bins[bins > 4] = 4
    return bins
```

iii. Matches the instructions: "Absolute position in corridor, discretized into 5 equal-sized bins."

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same frame indices (`start:stop`) ensure alignment with neural data.

ii.
```python
trial_pos = np.clip(position[start:stop], 0.0, 450.0)
```

iii. Inherent alignment through shared indexing.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from the `lick` behavioral time series in the NWB file, which contains cumulative lick counts per frame.

ii.
```python
lick_counts = behavior["lick"]["data"][:].astype(np.float32)
```

iii. The agent verified the lick data structure by inspecting unique values and distribution across sessions.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Binary thresholding: lick count > 0 → 1 (lick), otherwise → 0 (no lick).

ii.
```python
trial_lick = (lick_counts[start:stop] > 0).astype(np.int8)
```

iii. Matches the instructions: "Lick, time-varying. 0 = no, 1 = yes."

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same frame indices (`start:stop`) ensure alignment.

ii.
```python
trial_lick = (lick_counts[start:stop] > 0).astype(np.int8)
```

iii. Inherent alignment through shared indexing.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from the session scene name, which encodes the reward zone (A, B, or C). For switch sessions, the zone changes at trial 30.

ii.
```python
ZONE_TO_INDEX = {"A": 0, "B": 1, "C": 2}
# ...
label = trial_labels[trial_idx]  # from parse_scene()
zone_idx = ZONE_TO_INDEX[label.zone]
np.full(trial_times.shape, zone_idx, dtype=np.int8)
```

iii. From CONVERSION_NOTES: "Trial-level label A/B/C. Repeated across timepoints in the trial."

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The scene name is parsed to extract the zone letter (A/B/C), mapped to indices (0/1/2), and broadcast across all timepoints in the trial. For switch sessions, the first 30 trials get the pre-switch zone, subsequent trials get the post-switch zone.

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

iii. The 30-trial switch point is hardcoded based on the experimental protocol.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from the `Reward` timestamps in the NWB behavioral time series, combined with trial start/stop times.

ii.
```python
reward_timestamps = behavior["Reward"]["timestamps"][:].astype(np.float64)
reward_outcomes = reward_outcome_per_trial(reward_timestamps, timestamps[trial_start_idx], timestamps[trial_stop_idx])
```

iii. The agent checks whether any reward delivery timestamp falls within each trial's time window.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, a binary check: if any reward timestamp falls within `[trial_start_time, trial_stop_time)`, the outcome is 1 (rewarded), otherwise 0 (omitted). The value is broadcast across all timepoints in the trial.

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
np.full(trial_times.shape, reward_outcomes[trial_idx], dtype=np.int8)
```

iii. From CONVERSION_NOTES: "Trial-level label from reward timestamps falling between trial start and teleport."

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several strategies:
1. Multi-plane length mismatch: All behavioral and fluorescence streams are truncated to the minimum common length across all streams, preventing index-out-of-bounds errors.
2. Lick sensor errors: Trials with >30% of frames having lick count > 2 are excluded entirely.
3. NaN handling in neural processing: `nansmooth_2d` handles NaN values during Gaussian smoothing by normalizing by valid-sample weights. The `compute_dff_and_events` function uses NaN-filled arrays and only populates trial segments.
4. Missing sessions: The 2 missing m11 sessions (ses-01, ses-02) are simply absent from the dataset; no imputation is attempted.

ii.
```python
session_len = min(len(position), len(speed), len(lick_counts), ..., *plane_lengths)
position = position[:session_len]
# ... all streams truncated to session_len
```

iii. From CONVERSION_NOTES: "Some sessions contain plane0 and plane1. A few of these sessions have behavior and fluorescence streams that differ by one frame. All streams in a session were truncated to the common minimum length."

## 13-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are:
1. Reading fluorescence and neuropil data from NWB files (disk I/O for large arrays)
2. Computing dF/F and OASIS deconvolution per block of neurons (`compute_dff_and_events`)
3. The per-cell speed correlation computation for interneuron detection (involves `np.corrcoef` in a loop)
4. Serialization of the full dataset with pickle (4.5 GB output)

ii.
```python
for block_start in range(0, len(curated_plane_indices), block_size):
    fluorescence = fluorescence_ds[:session_len, block_local].T.astype(np.float32)
    neuropil = neuropil_ds[:session_len, block_local].T.astype(np.float32)
    dff, events = compute_dff_and_events(...)
    block_corr = np.array([np.corrcoef(trace, speed_valid)[0, 1] ... for trace in dff_valid])
```

iii. The agent's trajectory shows the full conversion took many minutes, with memory usage monitored at ~2.3 GB RSS.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops could be vectorized:
1. The per-cell speed correlation loop (lines 334-340): Could compute all correlations at once with a vectorized correlation matrix.
2. The `bad_lick_trial_mask` loop: Could vectorize using segment-level array operations.
3. The `reward_outcome_per_trial` loop: Could use `np.searchsorted` instead of manual index scanning.

ii.
```python
# Per-cell correlation loop:
block_corr = np.array([
    np.corrcoef(trace, speed_valid)[0, 1]
    if np.nanstd(trace) > 0 and np.nanstd(speed_valid) > 0
    else np.nan
    for trace in dff_valid
])
```

iii. The agent chose explicit loops for clarity, processing neurons in blocks of 256 to manage memory.

## 13-c. What processing does the code repeat multiple times?

i. The code iterates over trial segments multiple times within `compute_dff_and_events`:
1. First loop: Copy trial data into NaN-filled arrays (lines 162-164)
2. Second set of loops: Add back neuropil mean, compute baseline, smooth, apply min/max filters (lines 170-175)
3. Third loop: Smooth dF/F and run OASIS (lines 181-184)

Additionally, the nansmooth operation is called both for baseline computation (sigma=15) and for dF/F smoothing (sigma=2), each iterating over trial segments.

ii.
```python
# Loop 1: populate trial data
for start, stop in zip(start_idx, stop_idx):
    f[:, start:stop] = fluorescence[:, start:stop]
# Loop 2: baseline
for start, stop in zip(start_idx, stop_idx):
    f[:, sl] += neu_coef * np.nanmean(f_neu[:, sl], ...)
    baseline[:, sl] = nansmooth_2d(f[:, sl], sigma=15)
    baseline[:, sl] = minimum_filter1d(...)
    baseline[:, sl] = maximum_filter1d(...)
# Loop 3: smooth + deconvolve
for start, stop in zip(start_idx, stop_idx):
    dff[:, sl] = nansmooth_2d(dff[:, sl], sigma=2)
    events[:, sl] = dcnv.oasis(dff[:, sl], ...)
```

iii. The per-trial loop structure mirrors the reference code, which also processes trials individually for baseline and deconvolution.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes full dF/F traces (`dff`) as an intermediate step, but only the deconvolved events (`events`) are saved in the final dataset. The dF/F is used for interneuron detection (speed correlation) but is otherwise discarded.

ii.
```python
dff, events = compute_dff_and_events(...)
# dF/F is used only for:
block_corr = np.array([np.corrcoef(trace, speed_valid)[0, 1] ... for trace in dff_valid])
# Only events are kept:
kept_events = events[keep_block]
```

iii. The dF/F computation is necessary for interneuron detection and as a prerequisite for OASIS deconvolution, so it's not truly unnecessary — but it is discarded after use. The full dF/F arrays are allocated per block and garbage-collected.
