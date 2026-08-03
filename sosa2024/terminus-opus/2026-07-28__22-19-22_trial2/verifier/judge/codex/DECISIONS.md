# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads every NWB file matching `data/sub-*/*.nwb`, then processes each file as one session with `h5py`. It does not use `pynwb`; it reads NWB groups and datasets directly.

ii.
```python
all_files = sorted(glob.glob('data/sub-*/*.nwb'))
...
with h5py.File(nwb_path, 'r') as f:
    behav = f['processing/behavior/BehavioralTimeSeries']
    ...
    planes = sorted(f['processing/ophys/Deconvolved'].keys())
```

iii. `CONVERSION_NOTES.md` Step 2 says the NWB files live at `data/sub-{mouse}/sub-{mouse}_ses-{session}_behavior+ophys.nwb` and reports 11 subjects / 152 sessions. The trajectory shows the agent explicitly listing all subject folders and then choosing to process all matching `.nwb` files.

## 1-b. How are the data split into subjects?

i. Subjects are inferred from the NWB filename prefix, e.g. `sub-m11` from `sub-m11_ses-03_behavior+ophys.nwb`. A unique subject list is built as files are processed.

ii.
```python
subj_name = os.path.basename(nwb_path).split('_')[0]  # e.g., sub-m11
...
if subj_name not in subject_list:
    subject_list.append(subj_name)
subj_idx = subject_list.index(subj_name)
```

iii. `CONVERSION_NOTES.md` Step 2 lists the 11 subject IDs and describes the folder naming convention. The code follows that naming scheme directly.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. Session identity is read from `general/session_id` and each processed file becomes one entry in `neural`, `input`, and `output`.

ii.
```python
with h5py.File(nwb_path, 'r') as f:
    sess_id = f['general/session_id'][()].decode()
...
result = process_session(nwb_path, show_processing=show, session_idx=file_idx)
...
all_neural.append(neural_trials)
all_input.append(input_trials)
all_output.append(output_trials)
```

iii. The notes say there are 152 session files total, and the trajectory repeatedly treats one NWB file as one session during exploration and conversion.

## 1-d. How are the data split into trials?

i. Trial starts are all samples where `trial_start > 0`; trial ends are all samples where `teleport > 0`. The agent pairs those indices by order and uses each `[start:stop]` slice as one trial.

ii.
```python
trial_starts = np.where(trial_start_signal > 0)[0]
teleports = np.where(teleport_signal > 0)[0]
...
for i in range(n_trials):
    start = trial_starts[i]
    stop = teleports[i]
```

iii. `CONVERSION_NOTES.md` Step 5 says “Trial boundaries: trial_start to teleport signals.” The trajectory also states that `trial_start`/`teleport` matches the reference session object’s `trial_start_inds` / `teleport_inds`.

## 1-e. How are trials filtered based on quality controls?

i. The agent does not apply the reference solution’s `<50`-timepoint trial filter. It only skips degenerate trials with `stop <= start`, truncates mismatched start/end counts to the shorter length, and later drops whole sessions with fewer than 2 remaining trials.

ii.
```python
if len(teleports) != n_trials:
    n_trials = min(n_trials, len(teleports))
    trial_starts = trial_starts[:n_trials]
    teleports = teleports[:n_trials]
...
if stop <= start:
    continue
...
if len(neural_trials) < 2:
    print(f"  SKIPPING: only {len(neural_trials)} trials")
    continue
```

iii. The notes justify trial boundaries and length-mismatch truncation, but do not document any minimum-trial-length filter. This omission appears to be an implementation choice rather than a documented curation rule.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from the deconvolved calcium-event data in `processing/ophys/Deconvolved/{plane}/data`, together with `iscell` labels from `processing/ophys/ImageSegmentation/PlaneSegmentation/iscell`.

ii.
```python
iscell = f['processing/ophys/ImageSegmentation/PlaneSegmentation/iscell'][()]
planes = sorted(f['processing/ophys/Deconvolved'].keys())
for plane in planes:
    deconv_planes.append(f[f'processing/ophys/Deconvolved/{plane}/data'][()])
deconv_data = np.concatenate(deconv_planes, axis=1)
```

iii. `CONVERSION_NOTES.md` Step 1 identifies neural data as “Deconvolved events,” and the trajectory repeatedly states that deconvolved events are the decoder input.

## 2-b. How is the `neural` data processed?

i. The agent concatenates deconvolved planes across imaging planes, crops neural/behavior streams to a common minimum length if needed, filters to `iscell`, then applies an additional interneuron exclusion by correlating deconvolved activity with speed and dropping cells with `r > 0.5`.

ii.
```python
deconv_data = np.concatenate(deconv_planes, axis=1)
...
if n_behav != n_neural:
    min_len = min(n_behav, n_neural)
    ...
    deconv_data = deconv_data[:min_len, :]
...
cell_mask = iscell[:, 0] == 1
...
r = np.corrcoef(valid_neural[both_valid], valid_speed[both_valid])[0, 1]
...
cell_mask = cell_mask & ~interneuron_mask
neural_data = deconv_data[:, cell_indices]
```

iii. Step 5 of the notes explicitly says neural data use “iscell filter + interneuron exclusion,” and Step 4 says the agent intentionally used the paper’s `r > 0.5` threshold even though it only had deconvolved events rather than recomputed dF/F.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are first filtered by Suite2p `iscell`, then further filtered as putative interneurons if the correlation between deconvolved activity and clipped speed exceeds `0.5` over `trial_number >= 0` timepoints.

ii.
```python
cell_mask = iscell[:, 0] == 1
...
valid_mask = trial_number >= 0
...
interneuron_mask = speed_corr > 0.5
n_interneurons = int(np.sum(cell_mask & interneuron_mask))
cell_mask = cell_mask & ~interneuron_mask
```

iii. The notes frame this as matching the paper’s interneuron exclusion rule, with the caveat that the implementation uses deconvolved events as an approximation to dF/F.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to trial start by slicing the same trial index range `[trial_start:teleport]` used for all other variables; the first neural time bin of each trial is therefore the trial-start-aligned first frame.

ii.
```python
start = trial_starts[i]
stop = teleports[i]
trial_neural = neural_data[start:stop, :].T.astype(np.float32)
```

iii. The notes and trajectory both say the temporal alignment event is `trial_start`, so the agent treated trial segmentation itself as the alignment mechanism.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data keep the raw frame resolution. The agent estimates the time bin from the median difference of behavioral timestamps and does not rebin or resample.

ii.
```python
dt = np.median(np.diff(timestamps))
...
'time_bin_size': dt_ms,
```

iii. `CONVERSION_NOTES.md` Step 2 reports a sampling rate of about 15.5 Hz (64.48 ms), and Step 10 says the agent used raw imaging frames with no additional temporal binning.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from the `position` timestamps array, not from the NWB `trial number` timestamps used in the human reference solution.

ii.
```python
timestamps = behav['position/timestamps'][()]
...
time_from_start = (timestamps[start:stop] - timestamps[start]).astype(np.float32)
```

iii. The trajectory shows the agent concluding that behavioral streams share the same frame timestamps, so using one behavioral timestamp array was sufficient.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. For each trial, it subtracts the first timestamp of that trial from every timestamp in that trial.

ii.
```python
time_from_start = (timestamps[start:stop] - timestamps[start]).astype(np.float32)
```

iii. The notes map this variable as “timestamps - trial_start_time” and describe it as a time-varying float input.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It uses the exact same `[start:stop]` indices as the neural slice for each trial, after any global crop to the minimum neural/behavior length.

ii.
```python
if n_behav != n_neural:
    ...
    timestamps = timestamps[:min_len]
    deconv_data = deconv_data[:min_len, :]
...
trial_neural = neural_data[start:stop, :].T.astype(np.float32)
time_from_start = (timestamps[start:stop] - timestamps[start]).astype(np.float32)
```

iii. Step 10 of the notes says the time input matched the raw timestamps in spot checks and that trial counts matched after conversion.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. In the implemented converter, environment type is derived from the NWB `identifier` / scene string and the hard-coded switch trial, not from the `environment` time series that the script also loads. This conflicts with the notes, which describe the source as the environment signal.

ii.
```python
identifier = f['identifier'][()].decode()
scene_info = parse_scene(identifier)
...
environment = behav['environment/data'][()]
...
env_per_trial = get_env_per_trial(scene_info, n_trials)
```

iii. The notes’ mapping table says “environment signal,” but the trajectory and final code emphasize scene parsing plus `CHANGE_TRIAL = 30`, justified by the experiment design and the session identifiers.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The agent maps `Env1 -> 0` and `Env2 -> 1`, assigns a per-trial environment from parsed scene metadata, switches it after trial 30 only for environment-switch sessions, then broadcasts the result across all time bins in that trial.

ii.
```python
def get_env_per_trial(scene_info, n_trials, change_trial=CHANGE_TRIAL):
    env_map = {'Env1': 0, 'Env2': 1}
    ...
    if is_env_switch and i >= change_trial:
        env_vals.append(env_map.get(env_after, 0))
...
input_arr = np.array([
    time_from_start,
    np.full(n_tp, env_per_trial[i], dtype=np.float32),
```

iii. `CONVERSION_NOTES.md` Step 3 records “Switch trial = 30,” and the trajectory shows the agent treating scene metadata as authoritative for environment switching.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is not taken from the NWB `trial number` series. It is derived from the Python trial loop index after trials are segmented by `trial_start` and `teleport`.

ii.
```python
for i in range(n_trials):
    ...
    np.full(n_tp, i, dtype=np.float32),  # trial number within session
```

iii. The notes explicitly say the target variable is “trial index” and the trajectory repeatedly mentions preferring segmented trials over the raw `trial number` field.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The within-session trial index `i` is broadcast across every time bin in that trial.

ii.
```python
np.full(n_tp, i, dtype=np.float32)
```

iii. The notes map trial number as a per-trial broadcast variable with no extra processing.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from reward event timestamps in `Reward/timestamps`, converted into a per-trial rewarded/not-rewarded label.

ii.
```python
reward_timestamps = behav['Reward/timestamps'][()]
...
reward_in_trial = np.any(
    (reward_timestamps >= trial_start_time) &
    (reward_timestamps <= trial_end_time)
)
is_rewarded[i] = int(reward_in_trial)
```

iii. The notes map previous outcome to “previous trial reward,” and the trajectory describes reward outcome as coming from reward events rather than the `Reward/data` values themselves.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. The agent first marks each trial as rewarded if any reward timestamp falls within that trial’s time window. It then shifts that label by one trial so trial `i` gets the outcome of trial `i-1`; trial 0 is set to 0.

ii.
```python
prev_outcome = np.zeros(n_trials, dtype=int)
for i in range(1, n_trials):
    prev_outcome[i] = is_rewarded[i - 1]
...
np.full(n_tp, prev_outcome[i], dtype=np.float32),
```

iii. This matches the notes’ “previous trial reward” description, and the trajectory explicitly checks that the first trial is assigned 0.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Distance is derived from per-trial position plus reward-zone coordinates inferred from scene metadata (`identifier`) and `CHANGE_TRIAL = 30`. The loaded `reward_zone` time series is not used to generate the final reward-zone identity.

ii.
```python
position = behav['position/data'][()]
reward_zone_signal = behav['reward_zone/data'][()]
...
scene_info = parse_scene(identifier)
rz_labels, rz_coords = get_reward_zone_per_trial(scene_info, n_trials)
...
rz_start, rz_end = rz_coords[i]
trial_dist = distance_to_reward_zone(trial_pos, rz_start, rz_end)
```

iii. `CONVERSION_NOTES.md` maps reward-zone location to “scene name,” and the trajectory shows the agent deciding that parsed scene labels plus a 30-trial switch matched the NWB data well enough to use directly.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. For each timepoint, the agent computes signed distance to the current trial’s reward-zone interval: negative before the zone, zero inside it, positive after it.

ii.
```python
def distance_to_reward_zone(position, rz_start, rz_end):
    dist = np.zeros_like(position, dtype=float)
    before_mask = position < rz_start
    in_mask = (position >= rz_start) & (position <= rz_end)
    after_mask = position > rz_end
    dist[before_mask] = position[before_mask] - rz_start
    dist[in_mask] = 0.0
    dist[after_mask] = position[after_mask] - rz_end
```

iii. The notes describe this output as “position - reward zone” with 7 bins, and the trajectory’s final review says the agent intentionally used signed distance to the nearest zone edge.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The continuous signed distance is manually bucketed into 7 categories matching the instruction bins.

ii.
```python
bins[dist < -50] = 0
bins[(dist >= -50) & (dist < -10)] = 1
bins[(dist >= -10) & (dist < 0)] = 2
bins[dist == 0] = 3
bins[(dist > 0) & (dist <= 10)] = 4
bins[(dist > 10) & (dist <= 50)] = 5
bins[dist > 50] = 6
```

iii. The notes’ mapping table specifies “7 bins,” and the trajectory’s final review explicitly mentions matching the instruction’s distance categories.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. It is computed from the same per-trial `[start:stop]` position slice used alongside the per-trial neural slice, so it is aligned frame-by-frame within trial.

ii.
```python
trial_neural = neural_data[start:stop, :].T.astype(np.float32)
trial_pos = position[start:stop].astype(np.float32)
trial_dist = distance_to_reward_zone(trial_pos, rz_start, rz_end)
```

iii. The notes say all decoder variables are built from the aligned trial slices, and the trajectory’s spot checks report that raw position and converted outputs matched within a sampled trial.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Absolute position is derived directly from the `position` behavioral time series.

ii.
```python
position = behav['position/data'][()]
...
trial_pos = position[start:stop].astype(np.float32)
```

iii. The notes map position directly to `absolute_position`.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The agent clips position to `[0, 450]` cm, then discretizes it into 5 equal-width bins over the 450 cm track, i.e. 90 cm per bin.

ii.
```python
trial_pos = position[start:stop].astype(np.float32)
trial_pos = np.clip(trial_pos, TRACK_START, TRACK_LENGTH)
...
bin_edges = np.linspace(TRACK_START, TRACK_LENGTH, n_bins + 1)
bins = np.digitize(position, bin_edges[1:-1])
```

iii. `CONVERSION_NOTES.md` Step 5 says this output uses “5 equal bins (90cm each),” and Step 3 records the track length as 450 cm.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. The 5 categories are created by `np.digitize` over equally spaced edges from 0 to 450 cm, with values clipped to stay in bin indices `0..4`.

ii.
```python
bin_edges = np.linspace(TRACK_START, TRACK_LENGTH, n_bins + 1)
bins = np.digitize(position, bin_edges[1:-1])  # 0 to n_bins-1
bins = np.clip(bins, 0, n_bins - 1)
```

iii. The notes explicitly justify these as equal-sized corridor bins.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. The position slice and neural slice use the same trial boundaries and the same time-bin count for each trial.

ii.
```python
trial_neural = neural_data[start:stop, :].T.astype(np.float32)
trial_pos = position[start:stop].astype(np.float32)
```

iii. The notes’ validation section says spot checks on raw position versus converted output were successful.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Lick is derived from the `lick` behavioral time series.

ii.
```python
lick = behav['lick/data'][()]
...
trial_lick = lick_binary[start:stop].astype(np.float32)
```

iii. The notes map lick directly from the behavioral lick signal.

## 9-b. What processing is involved in computing `output` *Lick*?

i. The agent first performs per-trial lick-sensor error correction: if more than 35% of samples in a trial have lick values `> 2`, it replaces that trial’s lick samples with `NaN`. It then binarizes by capping positive values at 1 and finally converts `NaN` to 0 before producing the output.

ii.
```python
for i in range(n_trials):
    ...
    frac_high = np.sum(trial_licks > 2) / len(trial_licks)
    if frac_high > 0.35:
        lick_corrected[start:stop] = np.nan
...
lick_binary[lick_binary > 1] = 1
lick_binary[np.isnan(lick_binary)] = 0
...
lick_disc = (trial_lick > 0).astype(int)
```

iii. The notes say the agent followed the reference code’s lick-sensor correction threshold (`0.35`) rather than the paper text’s `>30%`, and the trajectory’s final review explicitly revisits that choice.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Lick uses the same `[start:stop]` trial slice as neural data, so it is frame-aligned within trial.

ii.
```python
trial_neural = neural_data[start:stop, :].T.astype(np.float32)
trial_lick = lick_binary[start:stop].astype(np.float32)
```

iii. The notes say all behavioral streams were sampled at the same frame rate and aligned to the neural bins.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Reward-zone location is derived from the NWB `identifier` scene string, not from the `reward_zone` signal values. The script parses the scene and uses a hard-coded switch at trial 30 when applicable.

ii.
```python
identifier = f['identifier'][()].decode()
scene_info = parse_scene(identifier)
...
rz_labels, rz_coords = get_reward_zone_per_trial(scene_info, n_trials)
```

iii. The notes’ mapping table explicitly lists “scene name” as the source for reward-zone location, and the trajectory says the agent verified the parsed reward-zone assignments against session behavior.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The scene string is parsed into pre-switch and post-switch reward-zone letters (`A/B/C`). For switch sessions, trials `>= 30` use the post-switch zone. The chosen letter is then mapped to `0/1/2`.

ii.
```python
if is_switch and i >= change_trial:
    label = rz_after
else:
    label = rz_before
...
rz_label_map = {'A': 0, 'B': 1, 'C': 2}
rz_loc = rz_label_map[rz_labels[i]]
```

iii. `CONVERSION_NOTES.md` Step 3 records the 30 warm-up trials, and Step 9 claims the scene-based reward-zone assignment matched the dataset statistics.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Reward outcome is derived from `Reward/timestamps`; the script checks whether any reward timestamp falls inside each trial window.

ii.
```python
reward_timestamps = behav['Reward/timestamps'][()]
...
reward_in_trial = np.any(
    (reward_timestamps >= trial_start_time) &
    (reward_timestamps <= trial_end_time)
)
```

iii. The notes map reward outcome directly from reward events and report an omission rate close to the paper.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, reward outcome is `1` if any reward timestamp falls between the trial’s start and end timestamps, else `0`. That scalar is then broadcast across the time bins of the trial.

ii.
```python
is_rewarded[i] = int(reward_in_trial)
...
output_arr = np.array([
    ...
    np.full(n_tp, reward_out, dtype=int),
], dtype=int)
```

iii. The notes describe reward outcome as a per-trial binary output and cite a full-dataset omission rate sanity check.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The agent uses several local heuristics: crop neural and behavioral arrays to the same minimum length; truncate trial starts/ends to the shorter count if mismatched; skip empty/non-positive-length trials; mark lick-error trials as `NaN` then convert them to no-lick (`0`); clip negative speed to `0`; and clip position to the track range `[0, 450]`.

ii.
```python
if n_behav != n_neural:
    min_len = min(n_behav, n_neural)
    ...
if len(teleports) != n_trials:
    n_trials = min(n_trials, len(teleports))
...
if stop <= start:
    continue
...
if frac_high > 0.35:
    lick_corrected[start:stop] = np.nan
...
lick_binary[np.isnan(lick_binary)] = 0
trial_speed = np.clip(trial_speed, 0, None)
trial_pos = np.clip(trial_pos, TRACK_START, TRACK_LENGTH)
```

iii. The notes explicitly justify length-mismatch truncation, lick correction, and keeping all speeds for the decoder; the remaining clipping behavior is implied by the code rather than defended in the notes.

## 13-a. What are the most time-consuming steps of the code?

i. The likely bottlenecks are loading large NWB arrays, concatenating multi-plane deconvolved data, computing per-cell speed correlations for interneuron exclusion, iterating over trials multiple times, and saving the large pickle output.

ii.
```python
with h5py.File(nwb_path, 'r') as f:
    ...
for plane in planes:
    deconv_planes.append(...)
...
for c in range(deconv_data.shape[1]):
    ...
for i in range(n_trials):
    ...
with open(args.output, 'wb') as f:
    pickle.dump(data, f, protocol=4)
```

iii. The notes report about `~1.3s` per session and `~200s` total; the rest of this answer is inferred from the visible loops and I/O-heavy structure of the converter.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-cell correlation loop for interneuron exclusion could be vectorized; the per-trial reward detection loop, lick-correction loop, and data-construction loop could also be reduced or fused; `get_reward_zone_per_trial` and `get_env_per_trial` are simple loops over trial count that could be vectorized or replaced with array construction.

ii.
```python
for c in range(deconv_data.shape[1]):
    ...
for i in range(n_trials):
    ...
for i in range(n_trials):
    ...
for i in range(n_trials):
    ...
```

iii. The notes mention runtime estimates but do not discuss vectorization in detail; this judgment mostly comes from reading the implementation.

## 13-c. What processing does the code repeat multiple times?

i. The converter walks the same trial boundaries multiple times: once to detect rewarded trials, again to perform lick correction, again to construct per-trial arrays, and additional times in plotting mode. It also separately constructs reward-zone labels and environment labels from the same scene metadata.

ii.
```python
for i in range(n_trials):
    ...  # reward detection
...
for i in range(n_trials):
    ...  # lick correction
...
for i in range(n_trials):
    ...  # build neural/input/output
```

iii. This is not called out in the notes, but it is clear from the code structure.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script loads `reward_zone_signal` and `environment` but does not use them to construct the final reward-zone or environment outputs. It also computes and stores some metadata-only counts, includes several unused constants/imports (`REWARD_ZONE_CENTERS`, `SAMPLING_RATE`, `defaultdict`, `sys`), and keeps a large plotting pathway that is irrelevant to the final pickle.

ii.
```python
reward_zone_signal = behav['reward_zone/data'][()]
environment = behav['environment/data'][()]
...
env_per_trial = get_env_per_trial(scene_info, n_trials)
rz_labels, rz_coords = get_reward_zone_per_trial(scene_info, n_trials)
...
REWARD_ZONE_CENTERS = {k: (v[0] + v[1]) / 2 for k, v in REWARD_ZONE_DICT.items()}
SAMPLING_RATE = 15.5
from collections import defaultdict
import sys
```

iii. This is an inference from the code itself; the notes do not discuss these redundancies.
