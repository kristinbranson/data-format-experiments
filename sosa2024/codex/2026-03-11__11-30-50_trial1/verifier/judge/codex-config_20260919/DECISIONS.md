# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent recursively discovers every NWB session matching `data/sub-*/sub-*_behavior+ophys.nwb` (unless `--sample` is requested), sorts the paths, and loads each session directly with `h5py`. The full run found 152 sessions from 11 subjects.

ii. ```python
files = sorted(Path("data").glob("sub-*/sub-*_behavior+ophys.nwb"))
if sample:
    return files[:2]
return files
...
with h5py.File(path, "r") as handle:
```

iii. The notes justify direct HDF5 access as faster than NWB object loading and report that the discovered 152 sessions agree with the planned 154 minus m11's two missing early sessions.

## 1-b. How are the data split into subjects?

i. The subject ID is read from each NWB file, and a first-seen mapping assigns each unique subject an index. Sessions retain that index in `subject_idx`.

ii. ```python
subject = decode_h5_scalar(handle["general/subject/subject_id"])
...
if subject not in subject_to_idx:
    subject_to_idx[subject] = len(data["subjects"])
    data["subjects"].append(subject)
subject_idx.append(subject_to_idx[subject])
```

iii. The notes confirm that the 11 `sub-*` directories correspond to the paper's 11-mouse switch cohort.

## 1-c. How are the data split into sessions?

i. Each NWB file is one session and becomes one element in the session-level lists. Files with fewer than two usable trials after QC would be skipped.

ii. ```python
for session_number, path in enumerate(session_files):
    session_data, stats = load_session(path=path, show_processing=...)
    if len(session_data["neural_trials"]) < 2:
        continue
    data["neural"].append(session_data["neural_trials"])
```

iii. The filename structure, NWB session metadata, and the observed 152 files support this interpretation.

## 1-d. How are the data split into trials?

i. Trial starts are frames where `trial_start > 0.5`. For each start, the first later frame where `teleport > 0.5` is the exclusive stop. Teleport frames are excluded.

ii. ```python
starts = np.flatnonzero(trial_start > 0.5)
teleports = np.flatnonzero(teleport > 0.5)
...
stop = teleports[tp_ptr]
if stop > start:
    trials.append((int(start), int(stop)))
...
neural_trial = deconvolved[start:stop].T
```

iii. The notes say this matches the reference semantics of a lap from trial start through teleport onset and avoids teleport contamination.

## 1-e. How are trials filtered based on quality controls?

i. Empty/one-frame trials are removed. Trials are also dropped when more than 35% of lick samples exceed 2; sessions with fewer than two retained trials are skipped. Reward omissions and low-speed frames are retained.

ii. ```python
def is_bad_lick_trial(lick_trial):
    if lick_trial.size == 0:
        return True
    return float(np.mean(lick_trial > 2)) > LICK_ERROR_FRAC
...
if pos_trial.size < 2:
    continue
if is_bad_lick_trial(lick_trial):
    continue
```

iii. The agent invokes the paper/reference lick-sensor QC and explains that whole bad-lick trials are removed because lick is a required output and missing labels are undesirable. It deliberately retains low-speed samples because speed is itself a target class.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes directly from the NWB `processing/ophys/Deconvolved` plane datasets, plus `ImageSegmentation/PlaneSegmentation/{iscell,planeIdx}` for ROI selection and plane assembly. It does not use raw `Fluorescence` or `Neuropil`.

ii. ```python
iscell = segmentation["iscell"][:]
plane_idx = segmentation["planeIdx"][:].astype(np.int16)
curated_idx = np.flatnonzero(iscell[:, 0] > 0.5)
deconv_group = handle["processing/ophys/Deconvolved"]
```

iii. The notes call NWB `Deconvolved` the closest native equivalent of the paper's `events`, while acknowledging that recomputing from fluorescence/neuropil was an alternative needing validation.

## 2-b. How is the `neural` data processed?

i. The agent assembles one or multiple planes, selects curated ROI columns, casts values to `float16`, slices each trial, and transposes from time-by-neuron to neuron-by-time. It does not recompute dF/F, smooth, baseline-correct, or deconvolve.

ii. ```python
deconvolved = np.asarray(deconv_group[plane_keys[0]]["data"][:, curated_idx], dtype=np.float16)
...
all_deconvolved[:, cols] = plane_data
deconvolved = all_deconvolved[:, curated_idx]
...
neural_trial = deconvolved[start:stop].T
```

iii. The justification is speed and memory efficiency and the assumption that exported deconvolution maps to reference events. The notes nevertheless document that the original pipeline computes trial-specific dF/F and OASIS events.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only Suite2p ROIs with `iscell[:,0] > 0.5` are retained. No speed-correlation putative-interneuron filter is applied.

ii. ```python
curated_idx = np.flatnonzero(iscell[:, 0] > 0.5)
...
deconvolved = all_deconvolved[:, curated_idx]
```

iii. The agent says this matches manual Suite2p curation, but flags the paper's additional interneuron exclusion as an unresolved consistency check. The final implementation never adds it.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural arrays are sliced beginning exactly at the reconstructed trial-start frame, so column zero is aligned to trial start; the stop is teleport onset.

ii. ```python
for trial_idx, (start, stop) in enumerate(trials):
    neural_trial = deconvolved[start:stop].T
```

iii. The notes and metadata identify `trial start` as the temporal alignment event and validate that converted time begins at zero.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Native imaging-frame resolution is retained without rebinning: 15.5078125 Hz, or about 64.48 ms per sample.

ii. ```python
FRAME_RATE_HZ = 15.5078125
TIME_BIN_MS = 1000.0 / FRAME_RATE_HZ
...
"time_bin_size": TIME_BIN_MS,
```

iii. The agent states that neural and behavioral signals are already synchronized at the imaging frame rate and that no temporal resampling is needed.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from the `position/timestamps` behavior timestamps and reconstructed trial boundaries.

ii. ```python
position_t = behavior["position/timestamps"][:].astype(np.float64)
...
time_trial = position_t[start:stop] - position_t[start]
```

iii. The notes treat all behavior streams as imaging-frame aligned and choose position timestamps as the common clock.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The timestamp at the first frame of each trial is subtracted from every timestamp in that trial, then the result is cast to `float32`.

ii. ```python
time_trial = position_t[start:stop] - position_t[start]
...
time_trial.astype(np.float32)
```

iii. This directly implements elapsed seconds relative to trial start.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. The same `[start:stop]` indices are used for time and neural arrays, producing equal-length columns with time zero at neural column zero.

ii. ```python
time_trial = position_t[start:stop] - position_t[start]
neural_trial = deconvolved[start:stop].T
n_time = neural_trial.shape[1]
```

iii. The agent's processing plots and sanity checks found no visible temporal offset.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Although raw `environment/data` is loaded, the saved value is derived from the NWB `identifier` scene string parsed into pre/post environments, together with trial index for switch sessions.

ii. ```python
identifier = decode_h5_scalar(handle["identifier"])
scene_info = parse_scene(identifier)
...
env_name, zone_name = zone_for_trial(scene_info, trial_idx)
env_code = float(ENV_TO_INT[env_name])
```

iii. The notes say scene metadata is cross-checked against the native 0/1 environment signal and reliably represents ENV1/ENV2, including environment switches.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. Scene names are parsed with regular expressions. For switch scenes, trial indices 30 and above use the post-switch environment. ENV1 maps to 0 and ENV2 to 1, repeated across all frames.

ii. ```python
if scene_info["switch"] and trial_idx >= SWITCH_TRIAL:
    return scene_info["post_env"], scene_info["post_zone"]
...
np.full(n_time, env_code, dtype=np.float32)
```

iii. The fixed switch-after-30 rule comes from the paper and reference code.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. It is the zero-based loop index of trials reconstructed from raw `trial_start` and `teleport`, not the NWB `trial number` stream.

ii. ```python
for trial_idx, (start, stop) in enumerate(trials):
...
np.full(n_time, float(trial_idx), dtype=np.float32)
```

iii. The agent considers reconstructed order more reliable around invalid/teleport samples and consistent with the reference trial-index variable.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No transformation beyond zero-based enumeration; the scalar index is repeated over every frame in the trial.

ii. ```python
np.full(n_time, float(trial_idx), dtype=np.float32)
```

iii. Repetition gives the uniform `(variables, time)` representation expected by the decoder.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It derives from raw `Reward/timestamps`, behavior `position/timestamps`, and reconstructed trial time intervals.

ii. ```python
reward_times = behavior["Reward/timestamps"][:].astype(np.float64)
...
reward_outcomes = reward_outcomes_from_timestamps(reward_times, position_t, trials)
```

iii. The notes say actual reward delivery, rather than planned/autoreward metadata, matches the requested rewarded-versus-omitted outcome.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. A trial outcome is one if a reward timestamp lies in `[start_time, stop_time)`. Each trial gets the preceding raw trial's outcome; the first gets zero. The value is repeated over time.

ii. ```python
outcomes[trial_idx] = int(reward_ptr < len(reward_times) and reward_times[reward_ptr] < stop_t)
...
prev_reward = int(reward_outcomes[trial_idx - 1]) if trial_idx > 0 else 0
```

iii. This implements the requested binary previous outcome. Importantly, after a bad-lick trial is dropped, the previous outcome still refers to the actual preceding experimental trial.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It uses raw `position/data` and active reward-zone coordinates inferred from the NWB identifier plus switch trial index.

ii. ```python
position = behavior["position/data"][:].astype(np.float32)
...
env_name, zone_name = zone_for_trial(scene_info, trial_idx)
zone_start, zone_end = ZONE_COORDS_CM[zone_name]
```

iii. The agent argues that NWB `reward_zone` values are occupancy-like rather than A/B/C labels and that scene metadata mirrors the paper's explicit task configuration.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance is position minus the near boundary before the zone, zero inside it, and position minus the far boundary after it; this is immediately discretized.

ii. ```python
distance = np.where(
    position_cm < zone_start,
    position_cm - zone_start,
    np.where(position_cm > zone_end, position_cm - zone_end, 0.0),
)
```

iii. This represents distance to the nearest point in the active 50-cm reward zone, with sign indicating before/after.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Seven explicit boolean ranges implement the requested cutoffs: `<-50`, `[-50,-10)`, `[-10,0)`, exactly zero, `(0,10]`, `(10,50]`, and `>50`.

ii. ```python
bins[distance < -50.0] = 0
bins[(distance >= -50.0) & (distance < -10.0)] = 1
bins[(distance >= -10.0) & (distance < 0.0)] = 2
bins[distance == 0.0] = 3
bins[(distance > 0.0) & (distance <= 10.0)] = 4
bins[(distance > 10.0) & (distance <= 50.0)] = 5
```

iii. The boundaries are a direct reading of the decoder specification.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position and neural data use the identical `[start:stop]` frame slice; the resulting distance categories therefore correspond column-for-column.

ii. ```python
pos_trial = position[start:stop]
neural_trial = deconvolved[start:stop].T
dist_bin = discretize_distance_to_zone(pos_trial, zone_start, zone_end)
```

iii. The agent relies on the NWB's already aligned behavior/imaging frame grid and visually checked the alignment.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It comes directly from the raw behavior `position/data` stream.

ii. ```python
position = behavior["position/data"][:].astype(np.float32)
...
pos_trial = position[start:stop]
```

iii. This stream records corridor position in centimeters.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Positions are clipped into `[0,450)` and discretized using five equal 90-cm bins.

ii. ```python
clipped = np.clip(position_cm, TRACK_START_CM, np.nextafter(TRACK_END_CM, TRACK_START_CM))
edges = np.linspace(TRACK_START_CM, TRACK_END_CM, 6)
return np.digitize(clipped, edges[1:-1], right=False).astype(np.int16)
```

iii. Clipping prevents small out-of-track numerical excursions from creating invalid classes while honoring the 450-cm track.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Internal edges 90, 180, 270, and 360 cm yield categories 0 through 4; edge values enter the higher bin.

ii. ```python
edges = np.linspace(TRACK_START_CM, TRACK_END_CM, 6)
np.digitize(clipped, edges[1:-1], right=False)
```

iii. These are the five equal bins required by the instructions.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. It is sliced with the same start and stop frames as neural activity.

ii. ```python
pos_trial = position[start:stop]
neural_trial = deconvolved[start:stop].T
```

iii. Both are on the same imaging-frame grid.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from `processing/behavior/BehavioralTimeSeries/lick/data`.

ii. ```python
lick = behavior["lick/data"][:].astype(np.float32)
...
lick_trial = lick[start:stop]
```

iii. The notes identify this as the aligned lick stream used by the reference pipeline.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Values are rounded, clipped to 0/1, and cast to integer. Beforeward, trials with more than 35% of samples above 2 are discarded as sensor failures.

ii. ```python
return np.clip(np.rint(lick_trial), 0, 1).astype(np.int16)
...
return float(np.mean(lick_trial > 2)) > LICK_ERROR_FRAC
```

iii. The agent follows the reference's binary lick correction and opts to discard corrupted trials instead of emitting NaN labels.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Lick and neural arrays share the same `[start:stop]` indices and thus the same time columns.

ii. ```python
lick_trial = lick[start:stop]
neural_trial = deconvolved[start:stop].T
```

iii. The NWB behavior streams are already synchronized to imaging frames.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It is derived from the NWB `identifier` scene string and reconstructed trial index, not directly from `reward_zone/data`.

ii. ```python
scene_info = parse_scene(identifier)
...
env_name, zone_name = zone_for_trial(scene_info, trial_idx)
```

iii. The agent found that raw `reward_zone` is not a simple A/B/C code, while scene identifiers explicitly encode initial and switched locations.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Regexes parse fixed-zone, zone-switch, and environment-plus-zone-switch scene formats. Switch sessions change at zero-based trial 30. A/B/C map to 0/1/2 and are repeated over trial frames.

ii. ```python
if scene_info["switch"] and trial_idx >= SWITCH_TRIAL:
    return scene_info["post_env"], scene_info["post_zone"]
...
zone_code = int(ZONE_TO_INT[zone_name])
np.full(n_time, zone_code, dtype=np.int16)
```

iii. The switch timing and zone coordinates come from the paper/reference code; the notes report scene-parsing sanity checks.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It uses raw `Reward/timestamps`, `position/timestamps`, and reconstructed trial intervals.

ii. ```python
reward_times = behavior["Reward/timestamps"][:].astype(np.float64)
...
reward_outcomes_from_timestamps(reward_times, position_t, trials)
```

iii. Actual timestamped reward delivery directly distinguishes rewarded from omission trials.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. A monotonic pointer scans reward events; a trial is 1 if an event occurs before its stop time after advancing past earlier events, otherwise 0. The result is repeated over frames.

ii. ```python
while reward_ptr < len(reward_times) and reward_times[reward_ptr] < start_t:
    reward_ptr += 1
outcomes[trial_idx] = int(reward_ptr < len(reward_times) and reward_times[reward_ptr] < stop_t)
...
np.full(n_time, reward_code, dtype=np.int16)
```

iii. This is an efficient implementation of presence/absence of reward within a trial.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The code decodes scalar bytes, rejects unknown scene formats, checks multi-plane ROI mapping and raises on mismatch, skips unmatched/degenerate trial boundaries, drops corrupted lick trials, clips position and lick edge cases, and skips sessions with fewer than two usable trials. It does not explicitly check equal lengths/timestamps among every behavior and neural stream, crop mismatches, or handle NaNs.

ii. ```python
if plane_data.shape[1] != cols.size:
    raise ValueError(...)
...
if tp_ptr >= len(teleports):
    break
...
if pos_trial.size < 2:
    continue
```

iii. The notes emphasize defensive scene/plane handling and trial QC but assume the exported NWB streams are already aligned and complete.

## 13-a. What are the most time-consuming steps of the code?

i. The dominant steps are reading each session's full deconvolved matrix, assembling multi-plane arrays, slicing/storing all trials, and serializing the large pickle. The full conversion took about 182 seconds.

ii. ```python
all_deconvolved = np.empty((n_frames, n_rois), dtype=np.float16)
...
plane_data = np.asarray(deconv_group[plane_key]["data"], dtype=np.float16)
...
pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The notes explicitly identify full-session neural loading as the main remaining cost and use direct `h5py`, curated-column reads, and `float16` to reduce I/O/memory.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. Trial reconstruction, reward-outcome assignment, per-trial conversion, and multi-plane assembly are explicit loops. Reward assignment could be vectorized with `searchsorted`; some full-session discretizations and trial-constant construction could be computed before slicing, though variable trial lengths limit the benefit.

ii. ```python
for start in starts:
...
for trial_idx, (start, stop) in enumerate(trials):
...
for plane_key in plane_keys:
```

iii. The agent did not claim these were vectorized; it prioritized sequential per-session processing to bound memory.

## 13-c. What processing does the code repeat multiple times?

i. For every trial it allocates repeated constant arrays for environment, trial number, previous outcome, zone, and reward. It also slices the same frame interval independently for neural, position, speed, lick, and time. With `--show-processing`, selected trial data are additionally plotted.

ii. ```python
np.full(n_time, env_code, dtype=np.float32)
np.full(n_time, float(trial_idx), dtype=np.float32)
np.full(n_time, float(prev_reward), dtype=np.float32)
...
np.full(n_time, zone_code, dtype=np.int16)
np.full(n_time, reward_code, dtype=np.int16)
```

iii. Repeating constants is intentional to meet a uniform time-varying decoder representation; the notes favor simplicity and decoder compatibility.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It loads raw `environment/data` but never uses it, creates optional plotting-only raw examples, and returns/stores detailed session stats that decoder training does not require. It also computes `session_stats` for reporting after embedding related stats in metadata.

ii. ```python
environment = behavior["environment/data"][:].astype(np.float32)
...
raw_examples = []
...
return data, session_stats
```

iii. Environment loading was intended for cross-checking, while plots and statistics support validation and documentation rather than decoder training.
