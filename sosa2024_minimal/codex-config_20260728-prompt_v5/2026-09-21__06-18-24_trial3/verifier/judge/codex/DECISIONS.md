# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI gathers every NWB file under `/app/data/sub-*`, sorts them by subject and session parsed from the filename, and converts each file as one session. It loads the NWB contents directly with `h5py` rather than `pynwb`, and only reads the datasets it needs from the HDF5 tree.

ii.
```python
def build_dataset():
    paths = sort_session_paths(glob.glob(str(DATA_ROOT / "sub-*" / "sub-*_behavior+ophys.nwb")))
    session_results = []

    for path_idx, path in enumerate(paths, start=1):
        converted = convert_session(path)
```

```python
def convert_session(path):
    with h5py.File(path, "r") as f:
        scene = f["identifier"][()].decode().split("/")[-1]
        subject = f["general/subject/subject_id"][()].decode()
        session_id = f["general/session_id"][()].decode()
```

iii. In the trajectory, the AI says the NWB release already contains the needed behavior, fluorescence, deconvolved activity, and segmentation tables, so it decided to read the NWB files directly instead of reconstructing the original `sess` objects (steps 13, 17, 48, 75). It also notes that multi-plane sessions required special per-plane loading logic (step 90).

## 1-b. How are the data split into subjects?

i. Subjects are identified from the per-session NWB metadata and then deduplicated and sorted at dataset-build time. The input file layout also groups files by subject directory.

ii.
```python
with h5py.File(path, "r") as f:
    subject = f["general/subject/subject_id"][()].decode()
```

```python
subjects = sorted({session["subject"] for session in session_results}, key=lambda s: int(s[1:]))
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
```

iii. The trajectory says the plan was to use session order from filenames and then keep the released subject identities from the NWB metadata (step 48). No more detailed subject-splitting rationale is recorded.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as exactly one session. Sessions are sorted by subject number and session number extracted from the filename.

ii.
```python
def sort_session_paths(paths):
    def key(path_str):
        name = Path(path_str).name
        match = re.search(r"sub-m(\d+)_ses-(\d+)_", name)
        return int(match.group(1)), int(match.group(2))

    return sorted(paths, key=key)
```

```python
for path_idx, path in enumerate(paths, start=1):
    converted = convert_session(path)
```

iii. The trajectory repeatedly refers to “one file per session” and to session order coming from the filenames (steps 17, 48, 75).

## 1-d. How are the data split into trials?

i. Trials are split by pairing each positive `trial_start` sample with the next positive `teleport` sample. A trial is the half-open interval `[start, stop)`, so it includes on-track frames and excludes the teleport frame.

ii.
```python
def pair_trial_segments(trial_start_signal, teleport_signal):
    starts = np.flatnonzero(trial_start_signal > 0)
    teleports = np.flatnonzero(teleport_signal > 0)
    segments = []
    teleport_ptr = 0

    for start in starts:
        while teleport_ptr < len(teleports) and teleports[teleport_ptr] <= start:
            teleport_ptr += 1
        if teleport_ptr >= len(teleports):
            break
        stop = teleports[teleport_ptr]
        teleport_ptr += 1
        if stop - start >= 1:
            segments.append((int(start), int(stop)))
```

iii. The trajectory explicitly says the NWB `trials` table is unused and trial segmentation will come from the frame-aligned `trial_start` and `teleport` streams, with care taken not to include teleport samples or drop the first valid frame (steps 21, 36, 75, 112).

## 1-e. How are trials filtered based on quality controls?

i. The AI drops entire trials flagged as lick-sensor-error trials and also drops trials that are too short to yield at least one 8-frame bin. After conversion, it skips any session with fewer than two usable trials.

ii.
```python
LICK_ERROR_THRESHOLD = 0.35
...
lick_error = bool(np.mean(lick_counts[start:stop] > 2.0) > LICK_ERROR_THRESHOLD)
```

```python
if info["lick_error"]:
    lick_error_trials += 1
    continue

n_frames = stop - start
n_bins = n_frames // BIN_FRAMES
if n_bins < 1:
    continue
```

```python
if len(converted["neural"]) < 2:
    continue
```

iii. The trajectory says the agent intentionally “drops lick-sensor-error trials” and chose binning partly to keep the dataset tractable for training (steps 63, 102, 112). It does not record a separate justification for the “at least one bin” filter beyond that.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The final saved `neural` signal comes from the NWB `processing/ophys/Deconvolved` series after curated-cell selection and interneuron exclusion. The raw `Fluorescence` and `Neuropil` series are only used to compute the speed-correlation interneuron filter, not as the saved signal.

ii.
```python
fluorescence = load_roi_response_matrix(
    f,
    "processing/ophys/Fluorescence",
    curated_cell_idx,
    plane_idx_all,
).T
neuropil = load_roi_response_matrix(
    f,
    "processing/ophys/Neuropil",
    curated_cell_idx,
    plane_idx_all,
).T
is_interneuron, speed_corr = find_putative_interneurons(
    fluorescence,
    neuropil,
    speed,
    trial_segments,
)
```

```python
deconvolved = load_roi_response_matrix(
    f,
    "processing/ophys/Deconvolved",
    curated_cell_idx,
    plane_idx_all,
).T
deconvolved = deconvolved[keep_cells]
```

iii. In the trajectory, the AI explicitly frames a decision about whether to trust the released `Deconvolved` signal or reproduce event extraction, then the final summary confirms that it kept the “Suite2p deconvolved calcium activity” as the source signal (steps 28, 75, 112).

## 2-b. How is the `neural` data processed?

i. The saved neural data are mean-binned deconvolved traces. The AI does not recompute the paper’s final event signal from `Fluorescence` and `Neuropil`; it only bins the stored `Deconvolved` traces in non-overlapping 8-frame windows and converts them to `float16`. It separately computes a dF/F-like signal only for interneuron detection.

ii.
```python
BIN_FRAMES = 8
...
def bin_2d_mean(arr, bin_frames):
    n_bins = arr.shape[1] // bin_frames
    if n_bins == 0:
        return None
    trimmed = arr[:, : n_bins * bin_frames]
    return trimmed.reshape(arr.shape[0], n_bins, bin_frames).mean(axis=2)
```

```python
neural = bin_2d_mean(deconvolved[:, start:stop_trimmed], BIN_FRAMES)
...
neural = np.nan_to_num(neural, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float16)
```

iii. The trajectory says the agent wanted a “tractable” saved dataset and chose fixed-width temporal bins so decoder training would be feasible (steps 63, 75, 87, 106, 112). There is no trajectory statement defending the choice to use the stored `Deconvolved` series beyond that it was already present in the NWB release.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neural data are filtered in two stages: first keep only curated `iscell` ROIs, then exclude cells whose dF/F-like trace has Pearson correlation greater than 0.5 with running speed. The dF/F-like signal used for that filter is computed trial-by-trial from `Fluorescence` and `Neuropil` with neuropil subtraction, maximin-like baseline estimation, and smoothing.

ii.
```python
iscell = np.asarray(
    f["processing/ophys/ImageSegmentation/PlaneSegmentation/iscell"][:],
    dtype=np.float32,
)
curated_cell_idx = np.flatnonzero(iscell[:, 0] > 0.5)
```

```python
signal = f_trial - NEUROPIL_COEF * fneu_trial
signal = signal + NEUROPIL_COEF * np.nanmean(fneu_trial, axis=1, keepdims=True)

baseline = smooth_ignore_nan(signal, sigma=15, axis=1)
baseline = minimum_filter1d(baseline, size=300, axis=1, mode="nearest")
baseline = maximum_filter1d(baseline, size=300, axis=1, mode="nearest")

dff = (signal - baseline) / np.abs(baseline)
...
return correlation > INTERNEURON_SPEED_CORR_THRESHOLD, correlation
```

iii. The trajectory says the agent wanted to reproduce the paper’s manual-cell curation plus putative-interneuron exclusion if it was practical to do so, and that it prototyped the filter before committing to it (steps 40, 52, 75, 112).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to trial start simply by slicing each trial from `trial_start` to immediately before `teleport`, then binning within that slice. No additional realignment is performed.

ii.
```python
trial_segments = pair_trial_segments(trial_start_signal, teleport_signal)
...
for trial_idx, info in enumerate(trial_info):
    start = info["start"]
    stop = info["stop"]
    ...
    neural = bin_2d_mean(deconvolved[:, start:stop_trimmed], BIN_FRAMES)
```

iii. The trajectory explicitly says the converter “segments trials from `trial_start` to the frame before `teleport`” and uses that as the trial-aligned representation (steps 75, 112).

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI rebins all streams into fixed 8-frame windows. With the hard-coded frame rate of 15.5078125 Hz, the saved time bin size is about 0.516 s, or 516 ms. Yes, temporal rebinning is applied everywhere.

ii.
```python
FRAME_RATE_HZ = 15.5078125
BIN_FRAMES = 8
TIME_BIN_SIZE_S = BIN_FRAMES / FRAME_RATE_HZ
```

```python
"time_bin_size": float(TIME_BIN_SIZE_S * 1000.0),
"bin_frames": BIN_FRAMES,
"frame_rate_hz": FRAME_RATE_HZ,
```

iii. The trajectory states that the full 15.5 Hz stream was considered too large to save and train on directly, so the agent chose 8-frame binning “to keep the dataset trainable” (steps 63, 75, 87, 102, 106, 112).

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from the behavior frame timestamps, specifically `position/timestamps`, which the AI uses as the canonical frame-time vector for the session.

ii.
```python
frame_times = np.asarray(behavior["position/timestamps"][:], dtype=np.float64)
...
time_from_start = (
    frame_times[start:stop_trimmed:BIN_FRAMES] - frame_times[start]
).astype(np.float32)
```

iii. The trajectory does not call out this specific timestamp choice. The closest recorded rationale is that the NWB release already contains frame-aligned behavior streams and exact trial boundaries (steps 21, 28).

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. For each kept trial, the AI samples one timestamp every 8 frames, subtracts the first trial timestamp, and stores that binned vector as the time input.

ii.
```python
time_from_start = (
    frame_times[start:stop_trimmed:BIN_FRAMES] - frame_times[start]
).astype(np.float32)
```

iii. The trajectory justification is indirect: the agent chose fixed-width 8-frame bins for tractability, so the time input was also reduced to that binned resolution (steps 63, 75, 112).

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. The time input is aligned by using the same trial boundaries and the same 8-frame binning schedule as the saved neural traces. The timestamp for each neural bin is the first frame timestamp in that bin.

ii.
```python
stop_trimmed = start + n_bins * BIN_FRAMES
neural = bin_2d_mean(deconvolved[:, start:stop_trimmed], BIN_FRAMES)
...
time_from_start = (
    frame_times[start:stop_trimmed:BIN_FRAMES] - frame_times[start]
).astype(np.float32)
```

iii. The trajectory says the agent wanted all streams to share the same trial-aligned, fixed-width bins after rebinning (steps 63, 75, 112).

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It is derived primarily from `environment/data`. If all values in a trial slice are missing or invalid, the AI falls back to the environment implied by the session `identifier` scene string.

ii.
```python
environment = np.asarray(behavior["environment/data"][:], dtype=np.float32)
...
env_slice = environment[start:stop]
env_slice = env_slice[np.isfinite(env_slice) & (env_slice >= 0)]
if env_slice.size:
    env_value = int(np.rint(np.nanmedian(env_slice)))
else:
    env_value = expected_envs[trial_idx]
```

iii. The trajectory notes that the scene identifier encodes the task condition and that environment schedule was one of the things needed from the NWB release (steps 21, 28, 75). It does not record a separate environment-specific argument.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The AI reduces the per-frame environment signal to one scalar per trial by taking the rounded median of valid values, then repeats that value across all time bins in the trial.

ii.
```python
if env_slice.size:
    env_value = int(np.rint(np.nanmedian(env_slice)))
...
input_trial = np.vstack(
    [
        time_from_start,
        np.full(n_bins, info["environment"], dtype=np.float32),
```

iii. No explicit environment-processing justification appears in the trajectory. The code implies the AI assumed environment is a per-trial constant once trial boundaries are known.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. The AI derives trial number from the frame-aligned `trial number/data` stream, taking the rounded median within each trial. If a trial has no valid values, it falls back to the loop index.

ii.
```python
trial_number_signal = np.asarray(behavior["trial number/data"][:], dtype=np.float32)
...
trialnum_slice = trial_number_signal[start:stop]
trialnum_slice = trialnum_slice[np.isfinite(trialnum_slice) & (trialnum_slice >= 0)]
if trialnum_slice.size:
    trial_number = int(np.rint(np.nanmedian(trialnum_slice)))
else:
    trial_number = trial_idx
```

iii. The trajectory does not give a dedicated justification for this choice. It only says the agent would use the frame-aligned behavior streams for trial segmentation and trial metadata (step 21).

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The AI collapses the within-trial `trial number` samples to a single rounded median integer, then repeats that constant value over all time bins in the trial.

ii.
```python
if trialnum_slice.size:
    trial_number = int(np.rint(np.nanmedian(trialnum_slice)))
...
np.full(n_bins, info["trial_number"], dtype=np.float32),
```

iii. No explicit trial-number-processing justification is recorded in the trajectory.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from `Reward/timestamps`, compared against each trial’s start and stop times to determine whether the previous trial contained any reward delivery.

ii.
```python
reward_timestamps = np.asarray(behavior["Reward/timestamps"][:], dtype=np.float64)
...
reward_outcome = int(np.any((reward_timestamps >= frame_times[start]) & (reward_timestamps < frame_times[stop])))
reward_outcomes.append(reward_outcome)
```

iii. The trajectory notes that the frame-aligned behavior streams and reward timestamps provide exact per-trial reward information needed for decoder inputs and outputs (step 28).

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. The AI first computes a binary current-trial reward outcome for every trial. Then, when building each trial’s input matrix, it uses the previous trial’s reward outcome as a scalar and repeats it across all bins of the current trial. The first trial gets 0.

ii.
```python
reward_outcome = int(np.any((reward_timestamps >= frame_times[start]) & (reward_timestamps < frame_times[stop])))
reward_outcomes.append(reward_outcome)
...
prev_outcome = reward_outcomes[trial_idx - 1] if trial_idx > 0 else 0
...
np.full(n_bins, prev_outcome, dtype=np.float32),
```

iii. There is no separate explicit justification in the trajectory beyond the general intent to use trial-aligned rewards as task variables (steps 28, 75).

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It is derived from the animal’s position (`position/data`) and the reward-zone identity inferred from the NWB `identifier` scene string. For switch sessions, the AI assumes the reward zone changes after 30 trials based on the parsed scene.

ii.
```python
scene = f["identifier"][()].decode().split("/")[-1]
...
zone_labels, expected_envs = expected_trial_labels(scene, len(trial_segments))
...
zone_bounds = REWARD_ZONES[info["zone_label"]]
distance_binned = reward_distance_cm(pos_binned.astype(np.float32), zone_bounds)
```

```python
def expected_trial_labels(scene, n_trials):
    info = parse_scene(scene)
    if info["switch_trial"] is None:
        zones = [info["zone_before"]] * n_trials
    ...
    zones = [info["zone_before"]] * pre_n + [info["zone_after"]] * post_n
```

iii. The trajectory says the agent discovered that `identifier` encodes the original scene name and then explicitly chose “scene-based reward-zone schedules” for conversion (steps 28, 75, 87, 112).

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. After position is binned, the AI computes signed distance to the nearest edge of the trial’s assigned reward zone: negative before the zone, 0 inside it, positive after it.

ii.
```python
def reward_distance_cm(position_cm, zone_bounds):
    start_cm, end_cm = zone_bounds
    distance = np.zeros_like(position_cm, dtype=np.float32)
    before = position_cm < start_cm
    after = position_cm > end_cm
    distance[before] = position_cm[before] - start_cm
    distance[after] = position_cm[after] - end_cm
    return distance
```

iii. The trajectory does not justify this formula separately; it only records the upstream choice to recover reward-zone identity from the scene metadata (steps 28, 75).

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The continuous distance is thresholded into 7 categories with the exact breakpoints from the task instructions.

ii.
```python
def discretize_reward_distance(distance_cm):
    out = np.empty(distance_cm.shape, dtype=np.uint8)
    out[distance_cm < -50.0] = 0
    out[(distance_cm >= -50.0) & (distance_cm < -10.0)] = 1
    out[(distance_cm >= -10.0) & (distance_cm < 0.0)] = 2
    out[distance_cm == 0.0] = 3
    out[(distance_cm > 0.0) & (distance_cm <= 10.0)] = 4
    out[(distance_cm > 10.0) & (distance_cm <= 50.0)] = 5
    out[distance_cm > 50.0] = 6
```

iii. No distinct trajectory justification is recorded; this appears to be a straightforward implementation of the requested bins.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. It is aligned by using the same trial slice and the same 8-frame bins as the neural data. The distance is computed from the binned position vector corresponding to those same bins.

ii.
```python
neural = bin_2d_mean(deconvolved[:, start:stop_trimmed], BIN_FRAMES)
pos_binned = bin_1d_mean(position[start:stop_trimmed], BIN_FRAMES)
...
distance_binned = reward_distance_cm(pos_binned.astype(np.float32), zone_bounds)
```

iii. The trajectory justification is the general shared-bin design chosen for tractability and decoder compatibility (steps 63, 75, 112).

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived directly from `position/data`.

ii.
```python
position = np.asarray(behavior["position/data"][:], dtype=np.float32)
...
pos_binned = bin_1d_mean(position[start:stop_trimmed], BIN_FRAMES)
```

iii. The trajectory identifies the frame-aligned behavior series as the source for position and other trial-varying outputs (steps 21, 28).

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The AI averages position in non-overlapping 8-frame windows, clips the result to the physical track bounds `[0, 450]`, and then discretizes it.

ii.
```python
pos_binned = bin_1d_mean(position[start:stop_trimmed], BIN_FRAMES)
...
pos_binned = np.clip(pos_binned, 0.0, TRACK_LENGTH_CM)
```

iii. The trajectory gives only the general binning justification: the full frame stream was considered too large, so all time-varying streams were reduced into shared 8-frame bins (steps 63, 75, 112).

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. The AI places position into 5 bins: `<90`, `90-180`, `180-270`, `270-360`, and `>=360` cm.

ii.
```python
def discretize_position(position_cm):
    out = np.empty(position_cm.shape, dtype=np.uint8)
    out[position_cm < 90.0] = 0
    out[(position_cm >= 90.0) & (position_cm < 180.0)] = 1
    out[(position_cm >= 180.0) & (position_cm < 270.0)] = 2
    out[(position_cm >= 270.0) & (position_cm < 360.0)] = 3
    out[position_cm >= 360.0] = 4
```

iii. No separate justification is recorded; this is a direct implementation of the requested bin edges.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. It is aligned by binning the position samples over the same trial slice and 8-frame windows used for the neural data.

ii.
```python
neural = bin_2d_mean(deconvolved[:, start:stop_trimmed], BIN_FRAMES)
pos_binned = bin_1d_mean(position[start:stop_trimmed], BIN_FRAMES)
```

iii. The trajectory justifies this through the shared-bin trial-aligned design chosen for all streams (steps 63, 75, 112).

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from the `lick/data` behavior series.

ii.
```python
lick_counts = np.asarray(behavior["lick/data"][:], dtype=np.float32)
```

iii. The trajectory mentions licks as one of the exact frame-aligned behavior streams available in the NWB release (step 28).

## 9-b. What processing is involved in computing `output` *Lick*?

i. The AI treats lick counts above 1 as 1, max-bins them over each 8-frame window, and then converts the result to binary 0/1. Entire trials can be dropped beforehand if the lick stream looks erroneous.

ii.
```python
lick_trial = lick_counts[start:stop_trimmed].copy()
lick_trial[lick_trial > 1.0] = 1.0
lick_binned = bin_1d_max(lick_trial, BIN_FRAMES)
lick_binned = (lick_binned > 0).astype(np.uint8)
```

```python
lick_error = bool(np.mean(lick_counts[start:stop] > 2.0) > LICK_ERROR_THRESHOLD)
```

iii. The trajectory explicitly says the agent decided to drop lick-sensor-error trials and to bin all streams into fixed 8-frame windows (steps 63, 75, 102, 112).

## 9-c. How is `output` *Lick* aligned with the neural data?

i. It is aligned by taking the same trial slice and the same 8-frame windows used for the neural data, then marking each neural bin as lick/no-lick depending on whether any lick occurred in that bin.

ii.
```python
neural = bin_2d_mean(deconvolved[:, start:stop_trimmed], BIN_FRAMES)
...
lick_binned = bin_1d_max(lick_trial, BIN_FRAMES)
```

iii. The trajectory justification is again the shared-bin trial-aligned representation used throughout the converter (steps 63, 75, 112).

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. The AI derives reward-zone location from the NWB `identifier` scene string, not from the `reward_zone` behavior time series.

ii.
```python
scene = f["identifier"][()].decode().split("/")[-1]
...
zone_labels, expected_envs = expected_trial_labels(scene, len(trial_segments))
```

iii. The trajectory explicitly says the agent found that `identifier` encodes the original scene name and then chose scene-based reward-zone schedules (steps 28, 75, 112).

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The AI parses the scene string with regexes, determines the pre-switch and post-switch reward zones and environments, assumes switch sessions change after trial 30, and then assigns a single zone label to every trial. That label is repeated across all time bins in the saved output matrix.

ii.
```python
def parse_scene(scene):
    match = re.fullmatch(r"(Env[12])_Location([ABC])", scene)
    ...
    match = re.fullmatch(r"(Env[12])_Location([ABC])_to_([ABC])", scene)
    ...
    match = re.fullmatch(r"(Env[12])_([ABC])_to_(Env[12])_([ABC])", scene)
```

```python
SWITCH_TRIAL = 30
...
np.full(n_bins, ord(info["zone_label"]) - ord("A"), dtype=np.uint8),
```

iii. The trajectory states both pieces of the rationale: the `identifier` field exposes the scene, and scene-based schedules were chosen as the explicit reward-zone logic (steps 28, 75, 87, 112).

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It is derived from `Reward/timestamps`, compared against each trial’s frame-time interval.

ii.
```python
reward_timestamps = np.asarray(behavior["Reward/timestamps"][:], dtype=np.float64)
...
reward_outcome = int(np.any((reward_timestamps >= frame_times[start]) & (reward_timestamps < frame_times[stop])))
```

iii. The trajectory notes that reward timestamps are one of the key released behavior signals used to build outputs (step 28).

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, the AI emits 1 if any reward timestamp falls within that trial’s time interval and 0 otherwise. The scalar outcome is repeated across all saved bins of that trial.

ii.
```python
reward_outcome = int(np.any((reward_timestamps >= frame_times[start]) & (reward_timestamps < frame_times[stop])))
...
np.full(n_bins, info["reward_outcome"], dtype=np.uint8),
```

iii. No more detailed reward-outcome justification appears in the trajectory beyond the general plan to use the frame-aligned trial structure and reward timestamps (steps 21, 28, 75).

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several cases in code:
- Multi-plane indexing mismatches are handled by loading each plane separately and reinserting traces into pooled ROI order.
- Missing or invalid environment values fall back to the scene-implied environment.
- Non-finite neural and speed values are replaced with 0 after binning.
- Position is clipped into `[0, 450]`.
- Trials with likely lick-sensor errors are dropped.
- Trials too short to produce one 8-frame bin are dropped.
- Sessions with fewer than two usable trials are dropped.

ii.
```python
def load_roi_response_matrix(h5file, base_path, cell_idx, plane_idx_all):
    ...
    for plane_id in plane_ids:
        ...
        plane_data = np.asarray(
            h5file[f"{base_path}/plane{plane_id}/data"][:, local_idx],
            dtype=np.float32,
        )
```

```python
if env_slice.size:
    env_value = int(np.rint(np.nanmedian(env_slice)))
else:
    env_value = expected_envs[trial_idx]
...
neural = np.nan_to_num(neural, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float16)
pos_binned = np.clip(pos_binned, 0.0, TRACK_LENGTH_CM)
speed_binned = np.nan_to_num(speed_binned, nan=0.0, posinf=0.0, neginf=0.0)
```

iii. The trajectory explicitly records the multi-plane indexing bug and its fix (step 90), the deliberate lick-error filtering (steps 102, 112), and the tractability rationale for dropping too-short binned trials via the 8-frame design (steps 63, 75). It does not mention separate justifications for NaN replacement or position clipping.

## 13-a. What are the most time-consuming steps of the code?

i. The most expensive parts of the AI code are likely: loading large per-plane fluorescence, neuropil, and deconvolved matrices from every NWB file; running the per-trial dF/F-like preprocessing inside `find_putative_interneurons`; and then binning all kept trials.

ii.
```python
fluorescence = load_roi_response_matrix(...)
neuropil = load_roi_response_matrix(...)
is_interneuron, speed_corr = find_putative_interneurons(...)
...
deconvolved = load_roi_response_matrix(...)
```

```python
for start, stop in trial_segments:
    ...
    baseline = smooth_ignore_nan(signal, sigma=15, axis=1)
    baseline = minimum_filter1d(baseline, size=300, axis=1, mode="nearest")
    baseline = maximum_filter1d(baseline, size=300, axis=1, mode="nearest")
```

iii. The trajectory supports this at a high level: it treats whole-session loading, multipane handling, and tractability of the full frame-rate dataset as the main engineering constraints (steps 63, 90, 102, 106). It does not include explicit profiling.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial loop inside `find_putative_interneurons` and the trial loop that constructs `trial_info`, `neural_trials`, `input_trials`, and `output_trials` are the clearest vectorization targets. The per-plane loading loop could also potentially be batched if the storage layout were changed.

ii.
```python
for start, stop in trial_segments:
    ...
```

```python
for trial_idx, (start, stop) in enumerate(trial_segments):
    ...
for trial_idx, info in enumerate(trial_info):
    ...
```

iii. No explicit vectorization discussion appears in the trajectory. This is inferred from the code structure the AI wrote.

## 13-c. What processing does the code repeat multiple times?

i. The code repeats several expensive steps:
- It loads `Fluorescence` and `Neuropil` to compute the interneuron filter, then separately loads `Deconvolved` for the actual saved neural signal.
- It computes a dF/F-like preprocessing pipeline solely for interneuron detection and then discards those traces.
- It parses and inspects each trial once to build `trial_info`, then loops over the trials again to build the saved outputs.

ii.
```python
fluorescence = load_roi_response_matrix(...)
neuropil = load_roi_response_matrix(...)
is_interneuron, speed_corr = find_putative_interneurons(...)
...
deconvolved = load_roi_response_matrix(...)
```

```python
for trial_idx, (start, stop) in enumerate(trial_segments):
    ...
    trial_info.append(...)

for trial_idx, info in enumerate(trial_info):
    ...
```

iii. The trajectory directly mentions the tradeoff between practicality and extra processing around the interneuron filter and the full-frame dataset size (steps 52, 63, 75). It does not explicitly call out the repeated trial loop.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code does several pieces of work that are not used by downstream decoder analyses:
- It computes full dF/F-like traces only to classify interneurons, then discards those traces.
- It keeps diagnostic metadata such as `environment_mismatch_trials`, `putative_interneuron_speed_corr_range`, per-plane cell counts, and reward-zone trial counts, which are not used by the decoder.
- It counts environment mismatches against scene expectations but does not act on them.

ii.
```python
is_interneuron, speed_corr = find_putative_interneurons(...)
...
session_metadata = {
    ...
    "putative_interneuron_speed_corr_range": [
        float(np.min(speed_corr)) if speed_corr.size else 0.0,
        float(np.max(speed_corr)) if speed_corr.size else 0.0,
    ],
    "environment_mismatch_trials": env_mismatch_trials,
    "reward_zone_trial_counts_kept": dict(zone_counter),
    "plane_counts_curated": dict(plane_counts_curated),
    "plane_counts_kept": dict(plane_counts_kept),
}
```

iii. The trajectory shows the AI using some of this work as engineering validation rather than as final decoder features, especially around interneuron filtering and data-integrity checks (steps 52, 90, 102, 112).
