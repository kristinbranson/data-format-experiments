# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent recursively discovers every matching NWB session under `/app/data`, sorts the paths, and opens each file directly with `h5py`. It loads behavior, event, ROI, and segmentation arrays from fixed NWB paths and converts every discovered session.

ii.
```python
paths = sorted(Path(data_dir).glob("sub-*/sub-*_behavior+ophys.nwb"))
...
with h5py.File(path, "r") as f:
    beh = f["processing/behavior/BehavioralTimeSeries"]
...
for path in files:
    session_records.append(convert_session(path, ...))
```

iii. The trajectory says it found 11 mice and ultimately 152 NWB sessions. It chose direct HDF5 access after inspecting the real NWB schema, both for speed and because the required fields were available at stable paths.

## 1-b. How are the data split into subjects?

i. A subject ID is parsed from the parent directory (`sub-...`). Unique IDs are sorted, and each session gets an integer `subject_idx`.

ii.
```python
subject = path.parent.name.replace("sub-", "")
subjects = sorted({record["subject"] for record in session_records})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
```

iii. The agent found that the directory organization reliably encoded the 11 mouse IDs and used that organization rather than inferring identity from behavior.

## 1-c. How are the data split into sessions?

i. Each NWB file is one session. The day is parsed from `ses-NN` in its filename, and one session record is appended per file.

ii.
```python
match = re.search(r"ses-(\d+)", path.name)
return subject, int(match.group(1))
...
record = convert_session(path, ...)
session_records.append(record)
```

iii. The trajectory reports that the file structure and session identifiers matched the paper’s session organization; the final conversion contained 152 sessions.

## 1-d. How are the data split into trials?

i. Trial starts are all positive samples in `trial_start`; trial ends are all positive samples in `teleport`. A trial includes `start` and excludes `stop`.

ii.
```python
trial_starts = np.flatnonzero(trial_start_signal > 0)
teleports = np.flatnonzero(teleport_signal > 0)
...
for trial_idx, (start, stop) in enumerate(zip(trial_starts, teleports)):
    neural = deconv[start:stop, selected_indices].T
```

iii. The agent inspected framewise boundary channels and the paper code, concluding that `trial_start` to teleport-exclusive reproduced the on-track interval and excluded the teleport sample.

## 1-e. How are trials filtered based on quality controls?

i. Trials are dropped when more than 35% of frames have raw lick count above 2. Nonpositive intervals and trials lacking an inferred zone or environment are also dropped. There is no reference-style minimum-50-frame filter.

ii.
```python
frac = float(np.mean(lick_trial > 2))
if frac > threshold:
    drop_mask[idx] = True
...
if T <= 0 or zone_label is None or env_label is None:
    continue
```

iii. The agent cited the paper repository’s `correct_lick_sensor_error` logic and selected its commonly used 0.35 threshold. Because dense decoder arrays cannot retain invalid lick values as NaNs, it chose to remove entire affected trials (69 trials).

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes directly from the NWB `processing/ophys/Deconvolved/plane0/data` response series. The linked `Fluorescence/plane0/rois` vector and segmentation `iscell` field select columns.

ii.
```python
deconv = f["processing/ophys/Deconvolved/plane0/data"]
rois = f["processing/ophys/Fluorescence/plane0/rois"][()]
iscell = f["processing/ophys/ImageSegmentation/PlaneSegmentation/iscell"][()][:, 0].astype(bool)
```

iii. The agent reasoned that the paper analyzed deconvolved calcium events and treated the published deconvolved response series as the safest already-produced event signal. It explicitly noted that only plane0 was linked in the NWB files for the multi-plane mice.

## 2-b. How is the `neural` data processed?

i. No dF/F, neuropil subtraction, smoothing, or OASIS deconvolution is recomputed. The stored event matrix is filtered to selected columns, sliced by trial, transposed to neuron-by-time, and cast to `float32`.

ii.
```python
selected_indices = np.flatnonzero(iscell[rois])
neural = deconv[start:stop, selected_indices].T.astype(np.float32)
```

iii. The trajectory shows the agent investigated recomputation and interneuron exclusion, then deliberately favored the simpler published `Deconvolved` series, believing it would closely match reference statistics.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only linked ROIs whose Suite2p/manual `iscell` flag is true are retained. No putative-interneuron exclusion based on dF/F–speed correlation is applied, and only the linked plane0 response series is used.

ii.
```python
selected_mask = iscell[rois]
selected_indices = np.flatnonzero(selected_mask)
...
"brain_region_idx": np.zeros(int(selected_mask.sum()), dtype=np.int64)
```

iii. The agent regarded `iscell` as the reliable curation field. It investigated the speed-correlation exclusion but omitted it for simplicity, and avoided unlinked second-plane segmentation rows because they could not be aligned to a stored response matrix.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural samples are cut from the `trial_start` index inclusive to the teleport index exclusive. Thus time column zero is the trial-start frame.

ii.
```python
for trial_idx, (start, stop) in enumerate(zip(trial_starts, teleports)):
    neural = deconv[start:stop, selected_indices].T.astype(np.float32)
```

iii. The agent confirmed that neural and behavior arrays are frame-aligned and therefore used common indices without interpolation.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning is applied. The behavior timestamp interval is used, and the dataset metadata stores the median across sessions: approximately 64.4836 ms.

ii.
```python
dt_sec = float(np.median(np.diff(times)))
...
"time_bin_size": float(np.median(time_bin_sizes_ms))
```

iii. The agent found a constant imaging-aligned rate across sessions and preserved native frame resolution to match the paper’s frame-aligned analyses.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from the `position` behavior series’ timestamps.

ii.
```python
times = beh["position"]["timestamps"][()].astype(np.float64)
```

iii. The agent observed that the behavior streams share the same frame timestamps and chose the position timestamps as the common clock.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The timestamp at the trial start is subtracted from every timestamp in the trial, then values are cast to `float32`.

ii.
```python
time_trial = (times[start:stop] - times[start]).astype(np.float32)
```

iii. This directly implements time relative to the requested alignment event.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It uses the identical `[start:stop]` frame slice and therefore has exactly one value per neural column.

ii.
```python
time_trial = times[start:stop] - times[start]
neural = deconv[start:stop, selected_indices].T
```

iii. The agent’s schema inspection indicated behavior and neural response arrays already share frame indexing, so it performed no resampling.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It is derived from the framewise `environment` behavior series.

ii.
```python
environment = beh["environment"]["data"][()].astype(np.float32)
```

iii. The agent identified this channel as the direct ENV1/ENV2 indicator.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. For each trial, negative values are discarded and the rounded median becomes its observed environment. Missing/noisy observations are replaced by a dominant schedule: one overall mode, or pre-/post-trial-30 modes if they differ. The resulting scalar is repeated across time.

ii.
```python
env_trial = env_trial[env_trial >= 0]
observed_env.append(int(np.rint(np.nanmedian(env_trial))) if env_trial.size else None)
env_schedule = infer_constant_or_switch_schedule(observed_env)
np.full(T, env_label, dtype=np.float32)
```

iii. The agent used the paper’s known 30-trial switch structure to recover day-8 environment switches and fill missing observations robustly.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. It is primarily derived from the framewise `trial number` behavior series; the loop index is a fallback if no nonnegative samples exist.

ii.
```python
trial_number = beh["trial number"]["data"][()].astype(np.float32)
trial_number_values = trial_number[start:stop]
```

iii. The agent inspected the stored trial-number field and chose its per-trial central value, with a safe sequential fallback.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. Negative samples are removed, the median is taken, and that scalar is repeated at every timepoint. If all samples are invalid, the zero-based `trial_idx` is used.

ii.
```python
trial_number_values = trial_number_values[trial_number_values >= 0]
trial_number_scalar = float(np.nanmedian(trial_number_values)) if trial_number_values.size else float(trial_idx)
np.full(T, trial_number_scalar, dtype=np.float32)
```

iii. The median was intended to tolerate occasional invalid/noisy samples while preserving the NWB’s stated trial number.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from the `Reward` event timestamps, the position timestamps, and reconstructed trial boundaries.

ii.
```python
reward_times = beh["Reward"]["timestamps"][()].astype(np.float64)
reward_outcomes = reward_outcomes_per_trial(reward_times, times, trial_starts, teleports)
```

iii. The agent recognized that Reward is event-timestamped rather than a framewise aligned channel and compared events directly with trial time intervals.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. A trial is rewarded if any event falls in its half-open time interval. Outcomes are shifted by one trial; the first trial is assigned zero. The scalar is repeated across time.

ii.
```python
has_reward = np.any((reward_times >= times[start]) & (reward_times < times[stop]))
previous_reward_outcomes = [0] + reward_outcomes[:-1]
```

iii. This directly follows the requested omitted=0/rewarded=1 definition and supplies a defined value for the session’s first trial.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It is derived from position, framewise `reward_zone`, Reward timestamps as a fallback, and hard-coded paper zone bounds A=80–130, B=200–250, C=320–370 cm.

ii.
```python
ZONE_BOUNDS_CM = {"A": (80.0, 130.0), "B": (200.0, 250.0), "C": (320.0, 370.0)}
zone_frames = np.flatnonzero(zone_signal_trial > 0)
zone_position_cm = float(np.nanmedian(pos_trial[zone_frames]))
```

iii. The agent used known task geometry and inferred a trial’s zone from where the zone signal was active; reward position supplies evidence when that signal is absent.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Observed zone positions are mapped to the closest zone center and filled using a constant or fixed trial-30 switch schedule. Position is clipped to 0–450 cm. Signed distance is position minus the near edge before the zone, zero inside it, and position minus the far edge after it.

ii.
```python
return min(ZONE_CENTERS_CM, key=lambda label: abs(zone_position_cm - ZONE_CENTERS_CM[label]))
...
dist[before] = pos_cm[before] - zone_start
dist[after] = pos_cm[after] - zone_end
```

iii. The agent considered this robust to omission trials while enforcing the paper’s known switch design.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Seven explicit masks implement: below -50; [-50,-10); [-10,0); exactly 0; (0,10]; (10,50]; above 50.

ii.
```python
out[dist_cm < -50.0] = 0
out[(dist_cm >= -50.0) & (dist_cm < -10.0)] = 1
out[(dist_cm >= -10.0) & (dist_cm < 0.0)] = 2
out[dist_cm == 0.0] = 3
out[(dist_cm > 0.0) & (dist_cm <= 10.0)] = 4
out[(dist_cm > 10.0) & (dist_cm <= 50.0)] = 5
out[dist_cm > 50.0] = 6
```

iii. The boundaries were chosen to implement the decoder specification literally, including a distinct exact-zero/in-zone class.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position and neural data use the same trial frame slice; distance is computed for every sliced position sample.

ii.
```python
pos_trial = pos[start:stop]
neural = deconv[start:stop, selected_indices].T
dist_trial = signed_distance_to_zone(pos_trial, zone_bounds_cm)
```

iii. The agent found the streams were already frame-aligned, so common slicing was sufficient.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived from the `position` behavior time series.

ii.
```python
pos = beh["position"]["data"][()].astype(np.float32)
```

iii. This is the direct VR corridor coordinate in centimeters.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Per-trial position is clipped to [0,450] and then discretized; no interpolation or smoothing is performed.

ii.
```python
pos_trial = np.clip(pos[start:stop], 0.0, 450.0).astype(np.float32)
pos_clipped = np.clip(pos_cm, 0.0, 450.0 - 1e-6)
```

iii. The agent treated slight out-of-range values as boundary noise and forced them into the specified 450-cm track.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Clipped position is divided by 90 cm and integer-cast, capped at category 4.

ii.
```python
return np.minimum((pos_clipped / 90.0).astype(np.int64), 4)
```

iii. Five equal bins across 450 cm are each 90 cm wide.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Both use the same `[start:stop]` trial frame indices.

ii.
```python
pos_trial = pos[start:stop]
neural = deconv[start:stop, selected_indices].T
```

iii. No alignment transform was needed because the NWB arrays share frame indexing.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It comes from the framewise `lick` behavior series.

ii.
```python
lick = beh["lick"]["data"][()].astype(np.float32)
```

iii. The agent identified this as the direct lick-count channel.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Any positive count becomes 1; all other values become 0. Trials with excessive values above 2 are filtered beforehand.

ii.
```python
lick_trial = (lick[start:stop] > 0).astype(np.int64)
```

iii. Binarization implements the requested no/yes output, while the separate trial filter addresses a known lick-sensor failure mode.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Lick and neural arrays are sliced with identical trial boundaries.

ii.
```python
lick_trial = lick[start:stop]
neural = deconv[start:stop, selected_indices].T
```

iii. The agent verified common frame alignment in the NWB schema.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It uses position and `reward_zone`, with Reward timestamps as fallback evidence, plus the known A/B/C geometry.

ii.
```python
observed_zone_labels, observed_zone_positions = infer_trial_zone_positions(
    pos, reward_zone_signal, times, reward_times, trial_starts, teleports)
```

iii. The zone channel is only directly informative on some frames/trials, so the agent combined it with position and known task geometry.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Median observed zone position is assigned to its nearest zone center. Missing observations are filled by a dominant constant or pre/post-trial-30 schedule. A/B/C map to 0/1/2 and the scalar is repeated across the trial.

ii.
```python
zone_schedule = infer_constant_or_switch_schedule(observed_zone_labels)
ZONE_TO_INDEX = {"A": 0, "B": 1, "C": 2}
np.full(T, ZONE_TO_INDEX[zone_label], dtype=np.int64)
```

iii. The agent used the experiment’s fixed switch design to avoid noisy per-frame labels and fill omission trials.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It is derived from `Reward` timestamps, behavior timestamps, and reconstructed trial intervals.

ii.
```python
reward_times = beh["Reward"]["timestamps"][()].astype(np.float64)
reward_outcomes_per_trial(reward_times, times, trial_starts, teleports)
```

iii. The Reward series is event-based, so the agent used temporal membership rather than its data array.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Outcome is 1 if at least one reward timestamp lies between trial start (inclusive) and teleport (exclusive), otherwise 0; it is repeated across the trial.

ii.
```python
has_reward = np.any((reward_times >= times[start]) & (reward_times < times[stop]))
np.full(T, reward_outcomes[trial_idx], dtype=np.int64)
```

iii. This implements the requested trial-level binary reward outcome without unnecessary frame-event alignment.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The code errors on unequal counts of starts and teleports; drops nonpositive trials; fills missing zone/environment labels from dominant fixed schedules; falls back from missing zone activity to reward position; falls back from invalid trial-number samples to loop index; filters lick-error trials; and raises if no valid trial remains. It does not crop neural/behavior length mismatches or explicitly verify all timestamps.

ii.
```python
if trial_starts.size != teleports.size:
    raise ValueError(...)
...
trial_number_scalar = ... if trial_number_values.size else float(trial_idx)
...
if not neural_trials:
    raise ValueError(...)
```

iii. The agent’s choices emphasize deterministic recovery of missing trial context and strict failure for structural inconsistencies. Its notes describe schedule filling and trial dropping as necessary for dense decoder-compatible arrays.

## 13-a. What are the most time-consuming steps of the code?

i. Reading large NWB datasets session-by-session, slicing/copying neural matrices for every trial, serializing the full pickle, and the separate decoder validation/training runs dominate. Reward-zone/environment inference is comparatively small.

ii.
```python
with h5py.File(path, "r") as f:
    neural = deconv[start:stop, selected_indices].T.astype(np.float32)
...
pickle.dump(payload, f)
```

iii. The trajectory repeatedly notes that full NWB-stack reads were slow and later used direct HDF5 access; full conversion and decoder training were long-running validation steps.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. Reward outcomes, lick-error fractions, zone observations, environment medians, and per-trial assembly are Python loops. The first four could be partly vectorized using trial labels/reductions, while variable-length output assembly naturally remains a loop.

ii.
```python
for start, stop in zip(trial_starts, teleports): ...
for idx, (start, stop) in enumerate(zip(trial_starts, teleports)): ...
for trial_idx, (start, stop) in enumerate(zip(trial_starts, teleports)): ...
```

iii. The agent did not explicitly optimize these loops; its implementation prioritizes transparent variable-length trial construction.

## 13-c. What processing does the code repeat multiple times?

i. It makes several separate passes over the same trial boundaries to infer zones, environments, rewards, lick errors, and finally construct arrays. Position is clipped twice, and all selected data are traversed again in dataset summarization. It also builds both full and sample datasets from already converted records.

ii.
```python
infer_trial_zone_positions(...)
reward_outcomes_per_trial(...)
drop_lick_error_trials(...)
for trial_idx, (start, stop) in enumerate(...): ...
```

iii. The trajectory does not justify the repeated passes specifically; they arise from separating each operation into readable helper functions and from producing both required/full and optional sample artifacts.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It computes and stores extensive diagnostics (`observed_zone_positions`, raw observed labels/environments, per-trial lick-error fractions, schedules, selected plane IDs), constructs and saves an optional sample dataset, prints summaries/schedules, and reads `planeIdx` only for metadata. These do not feed decoder arrays.

ii.
```python
session_info["observed_zone_positions_cm"] = observed_zone_positions
session_info["lick_error_fraction_per_trial"] = lick_error_fraction.tolist()
sample_data = build_dataset(sample_records)
print_subject_schedules(full_data)
```

iii. The agent intentionally created these diagnostics and the sample dataset for sanity checks, documentation, and faster validator/decoder testing, not for the downstream full-data decoder itself.
