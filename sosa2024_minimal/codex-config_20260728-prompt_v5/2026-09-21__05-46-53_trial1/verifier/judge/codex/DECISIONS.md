# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads all NWB files under `/app/data/sub-*/sub-*_behavior+ophys.nwb`, then opens each file with `pynwb.NWBHDF5IO`. Within each session it reads behavior streams from `nwb.processing["behavior"].data_interfaces["BehavioralTimeSeries"].time_series` and optical physiology streams from `nwb.processing["ophys"].data_interfaces`.

ii. 
```python
session_paths = sorted(DATA_ROOT.glob("sub-*/sub-*_behavior+ophys.nwb"))
...
with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
    nwb = io.read()
    behavior = nwb.processing["behavior"].data_interfaces["BehavioralTimeSeries"].time_series
    ophys = nwb.processing["ophys"].data_interfaces
```

iii. The trajectory says the dataset is “NWB-based” and later that the agent would “build the converter directly from NWB” rather than recreate older session objects. In its final summary it says it “stream[s] all 152 NWB files.”

## 1-b. How are the data split into subjects?

i. Subjects are identified from `nwb.subject.subject_id`. The script accumulates unique subject IDs while iterating over all session files, and stores a `subject_to_idx` map.

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

iii. The trajectory does not give a separate justification for subject parsing, but step 47 reports “152 imaging sessions across 11 mice,” showing that the agent intentionally used per-file subject IDs to recover all mice.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. The top-level dataset is a session list built by iterating over `session_paths`, converting each file independently, and appending its session-level lists.

ii. 
```python
for path in session_paths:
    session = convert_session(path, subject_to_idx)
    if session["n_trials_kept"] < 2:
        continue

    data["neural"].append(session["neural"])
    data["input"].append(session["input"])
    data["output"].append(session["output"])
```

iii. The trajectory repeatedly discusses “152 sessions” and refers to “each session” as the unit being streamed and converted, so the agent’s intended session split was one NWB file per session.

## 1-d. How are the data split into trials?

i. Trials are split using the behavior streams `trial_start` and `teleport`. The code takes indices where `trial_start > 0` and indices where `teleport > 0`, then zips those arrays and treats each trial window as `[trial_start, teleport)`, i.e. trial start inclusive and teleport exclusive.

ii. 
```python
trial_start = np.flatnonzero(np.asarray(behavior["trial_start"].data[:]) > 0)
teleport = np.flatnonzero(np.asarray(behavior["teleport"].data[:]) > 0)
...
for trial_idx, (start, stop) in enumerate(zip(trial_start, teleport)):
    neural_trial = deconv[start:stop].T
```

iii. The trajectory explicitly says in step 62 that “the valid lap is `trial_start` inclusive to `teleport` exclusive,” and the final summary repeats that exact rule.

## 1-e. How are trials filtered based on quality controls?

i. The AI drops trials if more than 35% of the frames have `lick > 2`, interpreting those as lick-sensor-corrupted laps. It also drops any whole session that ends up with fewer than two kept trials.

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

iii. The trajectory says the agent was “checking how often the lick-sensor error rule would flag entire trials,” then concluded it would “probably [drop] the small number of lick-corrupted trials rather than fabricating lick labels.” The final summary says 69 trials were dropped for this reason.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derives `neural` from the NWB deconvolved activity interface, not from raw fluorescence and neuropil. It reads the deconvolved ROI response matrices and filters them with the Suite2p `iscell` curation mask from the segmentation table.

ii. 
```python
def load_curated_deconvolved_activity(ophys_interfaces):
    deconv_name = next(name for name in ophys_interfaces.keys() if "Deconvolved" in name)
    deconv_iface = ophys_interfaces[deconv_name]
    plane_seg = ophys_interfaces["ImageSegmentation"].plane_segmentations["PlaneSegmentation"]
    keep_cells = get_iscell_mask(plane_seg)
```

iii. The trajectory says the NWB files “already include ... deconvolved activity,” and later that the agent would use “curated `iscell` ROIs” and “deconvolved events” directly from NWB.

## 2-b. How is the `neural` data processed?

i. The AI concatenates the stored deconvolved matrices across planes after applying the `iscell` mask, slices them into trials, transposes them to neuron-by-time, and then rebins them by summing over fixed 16-frame windows. It does not recreate the paper’s fluorescence-to-dF/F-to-OASIS pipeline.

ii. 
```python
parts.append(plane_data[:, keep_cells[roi_indices]])
...
return np.concatenate(parts, axis=1), keep_cells
...
neural_trial = deconv[start:stop].T
...
neural_binned[:, bin_idx] = neural_trial[:, sl].sum(axis=1, dtype=np.float32)
```

iii. The trajectory first notes that the NWB files already contain “deconvolved events,” then says native resolution would make the shared decoder too large, so it chose “fixed multi-frame temporal bins for a tractable decoder dataset.” The final summary states that all streams were binned into “fixed 16-frame windows.”

## 2-c. How is the `neural` data filtered based on quality controls?

i. The only neural quality-control filter is Suite2p’s curated `iscell` flag. The script does not apply the paper’s putative interneuron exclusion or any additional cell-level curation.

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

iii. Step 55 says the ROI table includes Suite2p `iscell`, and the agent decided to “filter to curated cells (`iscell[:,0] == 1`) to match the repository’s session objects.” There is no trajectory evidence of an interneuron filter.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The neural data are aligned to trial start by cutting each trial directly from the trial-start index to the teleport index, then binning from the first frame of that slice. The first time bin begins at the left edge of the trial.

ii. 
```python
neural_trial = deconv[start:stop].T
...
start = bin_idx * BIN_FRAMES
stop = min((bin_idx + 1) * BIN_FRAMES, nframes)
...
time_binned[bin_idx] = time_trial_s[start]
```

iii. The trajectory says the data are segmented “from `trial_start` inclusive to `teleport` exclusive,” and later notes a fix so that “the aligned trial truly starts at 0” by storing the left edge of each time bin.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 16 imaging frames per bin. The script sets `BIN_FRAMES = 16` and records a metadata time-bin size of about 1031.74 ms, so the native imaging frames are temporally rebinned into roughly 1.03 s bins.

ii. 
```python
BIN_FRAMES = 16
...
nbins = math.ceil(nframes / BIN_FRAMES)
...
"time_bin_size": float(BIN_FRAMES * 1000.0 * 0.06448362720402656),
"time_bin_size_frames": BIN_FRAMES,
```

iii. The trajectory says native-resolution data were too large for the shared decoder, so the agent chose “fixed multi-frame temporal bins.” The final summary explicitly reports “16-frame windows at the imaging rate (about 1.03 s per bin).”

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. This input is derived from `behavior["position"].timestamps`, which are used as the session timestamps for trial slicing and time offsets.

ii. 
```python
timestamps = np.asarray(behavior["position"].timestamps[:], dtype=np.float64)
...
time_trial_s = (timestamps[start:stop] - timestamps[start]).astype(np.float32)
```

iii. The trajectory does not justify using `position.timestamps` specifically, but step 25 says the behavior streams are already “aligned frame-by-frame to imaging,” which is the implicit reason for using one behavior timestamp stream directly.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The AI subtracts the trial’s first timestamp to make time relative to trial start. After that, it rebins time together with the other modalities and stores the left edge of each 16-frame bin.

ii. 
```python
time_trial_s = (timestamps[start:stop] - timestamps[start]).astype(np.float32)
...
time_binned[bin_idx] = time_trial_s[start]
```

iii. Step 106 says the agent changed the implementation so “the first time bin currently starts around 0.5 s” would be fixed by using “the left edge of each bin so the aligned trial truly starts at 0.”

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Time is aligned by using the same trial slice and the same 16-frame bins as the neural data. Each bin’s time is taken from the same frame block whose neural activity is summed.

ii. 
```python
binned = bin_trial(
    neural_trial=neural_trial,
    time_trial_s=time_trial_s,
    pos_trial_cm=pos_trial_cm,
    speed_trial_cm_s=speed_trial_cm_s,
    lick_trial=lick_trial,
    zone_label=zone_labels[trial_idx],
)
...
input_trial = np.vstack([binned["time"], ...])
```

iii. The trajectory repeatedly states that the NWB behavior streams are already aligned frame-by-frame with imaging, and the 16-frame binning was intentionally applied “to all streams.”

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. The environment input is derived primarily from the `environment` behavior time series. If a trial’s `environment` values are unavailable after filtering negative entries, the script falls back to parsing the session `scene` string.

ii. 
```python
env = np.asarray(behavior["environment"].data[:], dtype=np.float32)
...
env_trial = env[start:stop]
env_trial = env_trial[env_trial >= 0]
env_value = float(np.round(np.nanmedian(env_trial))) if len(env_trial) else float(parse_scene(scene)[0][-1] == "2")
```

iii. The trajectory says the NWB files already contain `environment`, but also that the agent wanted a “scene parser” that “covers every case,” which explains the fallback to `scene`.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The AI removes negative values, takes the rounded median over the trial, and repeats that single per-trial environment label across all time bins. The scene string is only used as a fallback.

ii. 
```python
env_trial = env[start:stop]
env_trial = env_trial[env_trial >= 0]
env_value = float(np.round(np.nanmedian(env_trial))) if len(env_trial) else float(parse_scene(scene)[0][-1] == "2")
...
np.full(ntime, env_value, dtype=np.float32)
```

iii. The trajectory does not include a dedicated justification for median-rounding the environment stream, but it consistently frames environment as a per-trial label rather than a genuinely time-varying signal.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is derived from the stored `behavior["trial number"]` stream, with a fallback to the loop index if no nonnegative values exist in the trial.

ii. 
```python
trial_number = np.asarray(behavior["trial number"].data[:], dtype=np.float32)
...
trialnum_trial = trial_number[start:stop]
trialnum_trial = trialnum_trial[trialnum_trial >= 0]
trialnum_value = float(np.round(np.nanmedian(trialnum_trial))) if len(trialnum_trial) else float(trial_idx)
```

iii. The trajectory does not separately justify this choice. The only visible justification is implicit: the agent saw that the NWB files already contain per-frame `trial number` and therefore used that stored field.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The script filters out negative values, takes the rounded median trial number within the trial, and repeats that scalar across all time bins. If nothing remains, it falls back to the current `trial_idx`.

ii. 
```python
trialnum_trial = trial_number[start:stop]
trialnum_trial = trialnum_trial[trialnum_trial >= 0]
trialnum_value = float(np.round(np.nanmedian(trialnum_trial))) if len(trialnum_trial) else float(trial_idx)
...
np.full(ntime, trialnum_value, dtype=np.float32)
```

iii. There is no explicit trajectory justification for median-rounding the stored trial numbers.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Previous trial outcome is derived from `behavior["Reward"].timestamps`, combined with the per-trial `[start, stop)` windows defined by `trial_start` and `teleport`.

ii. 
```python
reward_timestamps = np.asarray(behavior["Reward"].timestamps[:], dtype=np.float64)
...
trial_reward_ts = reward_timestamps[
    (reward_timestamps >= timestamps[start]) & (reward_timestamps < timestamps[stop])
]
reward_outcomes[trial_idx] = int(len(trial_reward_ts) > 0)
```

iii. Step 62 says the agent was checking how many unrewarded laps were “true random omissions” versus “zone active but no reward” laps, which is the trajectory evidence that reward timestamps were the basis for trial outcomes.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. The code first computes a binary `reward_outcomes` array for each trial by asking whether any reward timestamp falls inside that trial. It then shifts that array by one trial, sets the first trial to 0, and repeats the previous-trial label across all time bins in the current trial.

ii. 
```python
reward_outcomes = np.zeros(ntrials, dtype=np.int16)
...
reward_outcomes[trial_idx] = int(len(trial_reward_ts) > 0)
...
prev_reward_outcomes = np.zeros(ntrials, dtype=np.int16)
prev_reward_outcomes[1:] = reward_outcomes[:-1]
...
np.full(ntime, float(prev_reward_outcomes[trial_idx]), dtype=np.float32)
```

iii. The trajectory’s omission-analysis step shows the agent was explicitly reasoning about rewarded versus unrewarded previous laps. No separate justification beyond that is recorded.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Distance to reward zone is derived from `position` and a per-trial reward-zone label. That label is inferred from the `reward_zone` behavior stream when it is active, by taking the median position inside the active zone and mapping it to the nearest of zones A/B/C. If the `reward_zone` stream is inactive for the trial, the code falls back to scene-derived expected zone labels.

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

iii. The trajectory says the agent wanted “scene-based reward-zone fallback plus observed zone detection when `reward_zone` is active.” Step 65 says it listed unique scene strings so the parser would cover every case.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. After the trial is temporally rebinned, the AI computes the signed distance from the binned mean position to the nearest edge of the reward zone. Distance is 0 inside the zone, negative before the zone, and positive after it.

ii. 
```python
pos_binned[bin_idx] = np.mean(pos_trial_cm[sl], dtype=np.float64)
...
def reward_zone_distance(position_cm, zone_label):
    start_cm, end_cm = RZ_BOUNDS[zone_label]
    dist = np.zeros(position_cm.shape, dtype=np.float32)
    before = position_cm < start_cm
    after = position_cm > end_cm
    dist[before] = position_cm[before] - start_cm
    dist[after] = position_cm[after] - end_cm
    return dist
```

iii. The trajectory justification is indirect: the agent chose fixed 16-frame bins for tractability, so all downstream behavioral outputs, including distance to reward zone, were computed on those binned streams.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The AI uses seven manually coded threshold regions matching the requested bins: `< -50`, `[-50, -10)`, `[-10, 0)`, `0`, `(0, 10]`, `(10, 50]`, and `> 50`.

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
```

iii. The trajectory does not separately justify the thresholds; they follow the task specification directly.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. The distance signal is aligned to neural data by using the same per-trial frame slice and then the same 16-frame bins used for neural aggregation. Distance is computed from the binned position trace returned by `bin_trial`.

ii. 
```python
binned = bin_trial(
    neural_trial=neural_trial,
    time_trial_s=time_trial_s,
    pos_trial_cm=pos_trial_cm,
    speed_trial_cm_s=speed_trial_cm_s,
    lick_trial=lick_trial,
    zone_label=zone_labels[trial_idx],
)
...
output_trial = np.vstack([binned["distance_bin"], ...])
```

iii. The trajectory justification is the same as for 3-c: the agent believed the NWB streams were already frame-aligned and then intentionally binned all streams together.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Absolute position is derived directly from `behavior["position"].data`.

ii. 
```python
position = np.asarray(behavior["position"].data[:], dtype=np.float32)
...
pos_trial_cm = position[start:stop]
```

iii. The trajectory refers to “position” as one of the already aligned per-frame behavior streams and treats it as a direct behavioral variable from the NWB files.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The AI averages position within each 16-frame bin, then discretizes those binned mean positions. It does not discretize the native per-frame position samples directly.

ii. 
```python
pos_binned[bin_idx] = np.mean(pos_trial_cm[sl], dtype=np.float64)
...
"position_bin": discretize_position(pos_binned),
```

iii. The trajectory’s justification is again the global binning choice for decoder tractability rather than a position-specific rationale.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Position is thresholded into five bins with edges at 90, 180, 270, and 360 cm, with everything below 90 in bin 0 and everything at or above 360 in bin 4.

ii. 
```python
def discretize_position(position_cm):
    out = np.zeros(position_cm.shape, dtype=np.int16)
    out[(position_cm >= 90.0) & (position_cm < 180.0)] = 1
    out[(position_cm >= 180.0) & (position_cm < 270.0)] = 2
    out[(position_cm >= 270.0) & (position_cm < 360.0)] = 3
    out[position_cm >= 360.0] = 4
```

iii. The trajectory does not provide a separate justification; these thresholds match the task specification.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Position is aligned by slicing the same trial window as the neural data and then binning those samples with the same 16-frame bin boundaries used for neural aggregation.

ii. 
```python
neural_trial = deconv[start:stop].T
pos_trial_cm = position[start:stop]
...
pos_binned[bin_idx] = np.mean(pos_trial_cm[sl], dtype=np.float64)
```

iii. The trajectory explicitly says the behavior streams are already aligned frame-by-frame to imaging, and the final implementation bins them jointly.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Lick is derived from `behavior["lick"].data`.

ii. 
```python
lick = np.asarray(behavior["lick"].data[:], dtype=np.float32)
...
lick_trial = lick[start:stop]
```

iii. The trajectory names `lick` as one of the per-frame behavior streams already present in the NWB files.

## 9-b. What processing is involved in computing `output` *Lick*?

i. The AI converts lick into a binary per-bin label by setting each 16-frame bin to 1 if any frame in that bin has `lick > 0`, else 0. Separately, it drops whole trials if too many frames have `lick > 2`.

ii. 
```python
if np.mean(trial_lick > 2) > LICK_ERROR_THRESHOLD:
    drop_trial[trial_idx] = True
...
lick_binned[bin_idx] = int(np.any(lick_trial[sl] > 0))
```

iii. The trajectory explicitly justifies dropping “lick-corrupted trials rather than fabricating lick labels.” The binarization itself is not separately justified beyond being required by the task.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Lick is aligned by taking the same `[start:stop)` trial slice as the neural data and then applying the same 16-frame bins.

ii. 
```python
lick_trial = lick[start:stop]
...
lick_binned[bin_idx] = int(np.any(lick_trial[sl] > 0))
...
output_trial = np.vstack([..., binned["lick_bin"], ...])
```

iii. The trajectory justification is the same shared frame alignment and common binning of all streams.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Reward-zone location is derived from the `reward_zone` and `position` behavior streams, with fallback to the session `scene` string when no reward-zone-active samples are found in a trial.

ii. 
```python
reward_zone = np.asarray(behavior["reward_zone"].data[:], dtype=np.float32)
position = np.asarray(behavior["position"].data[:], dtype=np.float32)
scene = nwb.identifier.split("/")[-1]
...
if np.any(rz_mask):
    zone_labels.append(zone_from_position(float(np.nanmedian(position[start:stop][rz_mask]))))
else:
    zone_labels.append(expected_zone_labels[trial_idx])
```

iii. The trajectory’s explicit justification is that the agent wanted “scene-based reward-zone fallback plus observed zone detection when `reward_zone` is active.”

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. When `reward_zone > 0` within a trial, the AI takes the median position of those active samples and assigns the closest of zones A, B, or C. If no active samples exist, it predicts the zone from `scene`, using a hard-coded switch after trial 30 for two-phase scenes. The chosen label is then converted to 0/1/2 and repeated across time bins.

ii. 
```python
def expected_trial_labels(scene, ntrials):
    ...
    split = min(SWITCH_TRIAL, ntrials)
...
if np.any(rz_mask):
    zone_labels.append(zone_from_position(float(np.nanmedian(position[start:stop][rz_mask]))))
else:
    zone_labels.append(expected_zone_labels[trial_idx])
...
reward_zone_value = {"A": 0, "B": 1, "C": 2}[zone_labels[trial_idx]]
```

iii. Step 65 says the agent listed scene strings so the reward-zone parser would cover all cases. The final summary again highlights “scene-based reward-zone fallback.”

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Reward outcome is derived from the reward event timestamps in `behavior["Reward"].timestamps`, evaluated against each trial’s time window.

ii. 
```python
reward_timestamps = np.asarray(behavior["Reward"].timestamps[:], dtype=np.float64)
...
trial_reward_ts = reward_timestamps[
    (reward_timestamps >= timestamps[start]) & (reward_timestamps < timestamps[stop])
]
reward_outcomes[trial_idx] = int(len(trial_reward_ts) > 0)
```

iii. The trajectory explicitly discusses rewarded versus omitted laps in step 62, so reward timestamps were the basis for this output.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. The code marks a trial as rewarded if any reward timestamp falls within that trial’s `[start, stop)` window. The per-trial binary value is then repeated across all time bins in the output array.

ii. 
```python
reward_outcomes[trial_idx] = int(len(trial_reward_ts) > 0)
...
reward_outcome_value = int(reward_outcomes[trial_idx])
...
np.full(ntime, reward_outcome_value, dtype=np.int16)
```

iii. The trajectory justifies this indirectly through its rewarded-versus-omission analysis; no more detailed reward-outcome justification is recorded.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles a few specific issues rather than applying a broad missing-data policy. It raises an error if `trial_start` and `teleport` counts differ; it falls back to scene-derived environment labels if `environment` is missing/negative within a trial; it falls back to `trial_idx` if `trial number` is missing/negative; it falls back to scene-derived reward zones if `reward_zone` is inactive; and it has explicit multipane ROI-index handling for deconvolved matrices.

ii. 
```python
if len(trial_start) != len(teleport):
    raise ValueError(f"{path.name}: trial starts and teleports do not match")
...
env_value = float(np.round(np.nanmedian(env_trial))) if len(env_trial) else ...
...
trialnum_value = float(np.round(np.nanmedian(trialnum_trial))) if len(trialnum_trial) else float(trial_idx)
...
if np.any(rz_mask):
    ...
else:
    zone_labels.append(expected_zone_labels[trial_idx])
...
roi_indices = np.asarray(rrs.rois.data[:], dtype=int)
parts.append(plane_data[:, keep_cells[roi_indices]])
```

iii. The trajectory says the agent checked multipane layout after hitting a schema difference, then patched the loader to use shared ROI-table indices correctly. For missing reward-zone information it explicitly wanted scene-based fallback. There is no evidence of more general missing-data handling.

## 13-a. What are the most time-consuming steps of the code?

i. The slowest parts are reading all NWB files, loading large behavior and deconvolved matrices, looping over all trials in every session, binning each trial in `bin_trial`, and finally serializing the full dataset. In practice the agent also spent substantial time rerunning conversion and validation/training.

ii. 
```python
for path in session_paths:
    session = convert_session(path, subject_to_idx)
...
for trial_idx, (start, stop) in enumerate(zip(trial_start, teleport)):
    ...
    binned = bin_trial(...)
...
with open(OUTPUT_PATH, "wb") as f:
    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The trajectory says “most of that time is spent streaming all 152 NWB files and building the trial lists,” then later mentions official verification and full decoder training runs after conversion.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The clearest vectorization candidates are the inner per-bin loop in `bin_trial`, the first per-trial loop that computes reward outcomes, zone labels, and lick-corruption flags, and the second per-trial loop that slices and stacks arrays. Some of these loops exist because trials have variable duration, but the binning logic in particular could be rewritten more vectorially.

ii. 
```python
for bin_idx in range(nbins):
    start = bin_idx * BIN_FRAMES
    stop = min((bin_idx + 1) * BIN_FRAMES, nframes)
    ...

for trial_idx, (start, stop) in enumerate(zip(trial_start, teleport)):
    ...

for trial_idx, (start, stop) in enumerate(zip(trial_start, teleport)):
    ...
```

iii. The trajectory does not discuss vectorization directly. The code structure itself shows the repeated scalar loops.

## 13-c. What processing does the code repeat multiple times?

i. The code repeats a few operations. It opens every NWB file once in `main` just to recover `subject_id`, then opens it again in `convert_session` to do the real conversion. Within each session it makes one per-trial pass to compute `reward_outcomes`, `zone_labels`, and `drop_trial`, then a second per-trial pass to actually extract and bin trial data. It also reparses `scene` inside `expected_trial_labels` and again in the environment fallback.

ii. 
```python
for path in session_paths:
    with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
        subject = io.read().subject.subject_id
...
for trial_idx, (start, stop) in enumerate(zip(trial_start, teleport)):
    ...
for trial_idx, (start, stop) in enumerate(zip(trial_start, teleport)):
    ...
env_value = ... else float(parse_scene(scene)[0][-1] == "2")
```

iii. The trajectory does not explicitly call these repetitions out, but it does mention long runtimes from repeatedly streaming NWB files and rebuilding trial lists.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script contains some unnecessary work. `TRACK_LENGTH_CM` is defined but never used. `expected_trial_labels` computes both `env_by_trial` and `zone_by_trial`, but only the zone labels are actually used. The decoder-facing pipeline also stores large session metadata blocks and raw/kept ROI counts that are not consumed by downstream decoder training.

ii. 
```python
TRACK_LENGTH_CM = 450.0
...
def expected_trial_labels(scene, ntrials):
    envs, zones = parse_scene(scene)
    env_by_trial = []
    zone_by_trial = []
    ...
    return env_by_trial, zone_by_trial
...
_, expected_zone_labels = expected_trial_labels(scene, ntrials)
...
"session_info": [],
...
"raw_roi_count": session["raw_roi_count"],
"kept_roi_count": session["kept_roi_count"],
```

iii. The trajectory does not explicitly justify these extra pieces. The closest related statement is that the agent wanted the scene parser to “cover every case,” which likely explains why it computed both environment and zone label sequences even though only one sequence was later used.
