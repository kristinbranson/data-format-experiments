# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent treats each NWB file as one session, finds all files with `glob.glob('data/sub-*/*.nwb')`, and processes them one by one with `h5py`. Within each session it loads behavioral streams from `processing/behavior/BehavioralTimeSeries` and neural data from `processing/ophys/Deconvolved/{plane}/data`, concatenating planes for multi-plane sessions.

ii. ```python
all_files = sorted(glob.glob('data/sub-*/*.nwb'))

with h5py.File(nwb_path, 'r') as f:
    behav = f['processing/behavior/BehavioralTimeSeries']
    position = behav['position/data'][()]
    speed = behav['speed/data'][()]
    lick = behav['lick/data'][()]
    trial_start_signal = behav['trial_start/data'][()]
    teleport_signal = behav['teleport/data'][()]
    timestamps = behav['position/timestamps'][()]

    planes = sorted(f['processing/ophys/Deconvolved'].keys())
    deconv_planes = []
    for plane in planes:
        deconv_planes.append(f[f'processing/ophys/Deconvolved/{plane}/data'][()])
    deconv_data = np.concatenate(deconv_planes, axis=1)
```

iii. In `CONVERSION_NOTES.md`, Step 2 says the NWB files are the primary data units, and Step 1 maps the reference `sess.timeseries['events']` and behavioral fields onto the NWB structure. The trajectory also shows the agent explicitly exploring those NWB groups before implementing the loader.

## 1-b. How are the data split into subjects?

i. Subjects are inferred from the NWB filename prefix such as `sub-m11`, then deduplicated into `subject_list`. Each processed session stores an index into that list.

ii. ```python
subj_name = os.path.basename(nwb_path).split('_')[0]

if subj_name not in subject_list:
    subject_list.append(subj_name)
subj_idx = subject_list.index(subj_name)
all_subject_idx.append(subj_idx)
```

iii. The notes say the dataset is organized as `data/sub-{mouse}/...`, and the agent’s Step 2 exploration enumerated the 11 subject folders. That directory structure appears to be the basis for this decision.

## 1-c. How are the data split into sessions?

i. Each `.nwb` file is treated as one session. The output lists `neural`, `input`, and `output` each get one top-level entry per file/session.

ii. ```python
all_files = sorted(glob.glob('data/sub-*/*.nwb'))

for file_idx, nwb_path in enumerate(files_to_process):
    result = process_session(nwb_path, show_processing=show, session_idx=file_idx)
    neural_trials, input_trials, output_trials, subj_name, n_cells, session_info = result
    all_neural.append(neural_trials)
    all_input.append(input_trials)
    all_output.append(output_trials)
```

iii. `CONVERSION_NOTES.md` Step 2 describes the NWB layout as one file per subject/session, and the trajectory shows the agent counting 152 NWB files and equating that to 152 sessions.

## 1-d. How are the data split into trials?

i. Trials are defined by the binary `trial_start` and `teleport` signals. The agent finds indices where those signals are positive and treats each `start:stop` interval as one trial.

ii. ```python
trial_starts = np.where(trial_start_signal > 0)[0]
teleports = np.where(teleport_signal > 0)[0]

for i in range(n_trials):
    start = trial_starts[i]
    stop = teleports[i]
    if stop <= start:
        continue
    trial_neural = neural_data[start:stop, :].T.astype(np.float32)
```

iii. In Step 1 of the notes, the agent identified `sess.trial_start_inds` and `sess.teleport_inds` from the reference code as the trial boundaries. The trajectory then mapped those to `trial_start/data` and `teleport/data` in NWB.

## 1-e. How are trials filtered based on quality controls?

i. The agent does very little trial-level filtering. It truncates mismatched `trial_start`/`teleport` lists to the shorter length, skips any interval with `stop <= start`, and skips entire sessions with fewer than two valid trials. It does not drop lick-error trials; instead it zero-fills their lick output later.

ii. ```python
if len(teleports) != n_trials:
    n_trials = min(n_trials, len(teleports))
    trial_starts = trial_starts[:n_trials]
    teleports = teleports[:n_trials]

for i in range(n_trials):
    start = trial_starts[i]
    stop = teleports[i]
    if stop <= start:
        continue

if len(neural_trials) < 2:
    print(f"  SKIPPING: only {len(neural_trials)} trials")
    continue
```

iii. `CONVERSION_NOTES.md` Step 5 lists “length mismatch” and “lick sensor error” handling but does not describe any global trial rejection. The trajectory indicates the agent chose to preserve trials for the decoder rather than mask them out as the reference code often did.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The final `neural` matrices come from `processing/ophys/Deconvolved/{plane}/data`, with `iscell` used for cell selection and behavioral `speed` plus `trial number` used only for the interneuron-correlation filter.

ii. ```python
iscell = f['processing/ophys/ImageSegmentation/PlaneSegmentation/iscell'][()]
planes = sorted(f['processing/ophys/Deconvolved'].keys())
deconv_planes = []
for plane in planes:
    deconv_planes.append(f[f'processing/ophys/Deconvolved/{plane}/data'][()])
deconv_data = np.concatenate(deconv_planes, axis=1)

speed = behav['speed/data'][()]
trial_number = behav['trial number/data'][()]
```

iii. Step 1 of the notes explicitly maps reference `sess.timeseries['events']` to the NWB deconvolved data and notes that the original paper derived these events from dF/F with OASIS before the NWB export.

## 2-b. How is the `neural` data processed?

i. The agent concatenates all deconvolved imaging planes, truncates to the minimum behavior/neural length if they disagree, filters to selected cells, and slices the result trial by trial. It does not recompute dF/F or deconvolution from fluorescence.

ii. ```python
deconv_data = np.concatenate(deconv_planes, axis=1)

if n_behav != n_neural:
    min_len = min(n_behav, n_neural)
    deconv_data = deconv_data[:min_len, :]

cell_indices = np.where(cell_mask)[0]
neural_data = deconv_data[:, cell_indices]

trial_neural = neural_data[start:stop, :].T.astype(np.float32)
```

iii. In the trajectory, the agent concluded that the NWB already contains deconvolved events and therefore chose not to recompute dF/F or rerun OASIS. `CONVERSION_NOTES.md` Step 5 repeats that choice.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The agent first keeps only `iscell[:, 0] == 1` ROIs, then applies an approximate interneuron filter by correlating deconvolved events, not dF/F, with speed during `trial_number >= 0` samples and excluding cells with correlation `> 0.5`.

ii. ```python
cell_mask = iscell[:, 0] == 1

speed_valid = speed.copy()
speed_valid[speed_valid < 0] = 0
valid_mask = trial_number >= 0

for c in range(deconv_data.shape[1]):
    if cell_mask[c]:
        valid_neural = deconv_data[valid_mask, c]
        valid_speed = speed_valid[valid_mask]
        both_valid = ~np.isnan(valid_neural) & ~np.isnan(valid_speed)
        if np.sum(both_valid) > 100:
            r = np.corrcoef(valid_neural[both_valid], valid_speed[both_valid])[0, 1]
            speed_corr[c] = r if not np.isnan(r) else 0

interneuron_mask = speed_corr > 0.5
cell_mask = cell_mask & ~interneuron_mask
```

iii. The notes say the agent wanted to follow the paper’s `r > 0.5` interneuron rule but considered full dF/F recomputation “complex,” so it substituted deconvolved-event correlation as a “reasonable approximation.” The trajectory shows this was a deliberate approximation.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural activity is aligned to trial start by extracting the exact same `start:stop` frame interval used to define the trial, and `time_from_start` is computed from the same first frame.

ii. ```python
start = trial_starts[i]
stop = teleports[i]

trial_neural = neural_data[start:stop, :].T.astype(np.float32)
time_from_start = (timestamps[start:stop] - timestamps[start]).astype(np.float32)
```

iii. The task instructions explicitly required alignment to trial start, and `CONVERSION_NOTES.md` Step 5 lists “trial boundaries: trial_start to teleport signals” as the chosen alignment scheme.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data keeps the original imaging-frame resolution. The bin size is the median spacing of the behavior/imaging timestamps, about 64.5 ms, and no temporal rebinning is applied.

ii. ```python
dt = np.median(np.diff(timestamps))
...
dt_ms = session_infos[0]['dt'] * 1000
...
'time_bin_size': dt_ms,
```

iii. The notes repeatedly state that the sampling rate is about 15.5 Hz and that the converter uses raw imaging frames with no additional temporal binning.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from `position/timestamps` plus the trial boundaries from `trial_start/data` and `teleport/data`.

ii. ```python
trial_start_signal = behav['trial_start/data'][()]
teleport_signal = behav['teleport/data'][()]
timestamps = behav['position/timestamps'][()]
...
time_from_start = (timestamps[start:stop] - timestamps[start]).astype(np.float32)
```

iii. The trajectory shows the agent inspected the shared timestamp arrays in NWB and concluded they are aligned to the imaging frames, which motivated using them directly.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. For each trial, the agent slices the per-frame timestamp array from trial start to teleport and subtracts the timestamp at the first frame, producing a zero-based continuous time vector in seconds.

ii. ```python
start = trial_starts[i]
stop = teleports[i]
time_from_start = (timestamps[start:stop] - timestamps[start]).astype(np.float32)
```

iii. `CONVERSION_NOTES.md` Step 5 maps this variable directly to “timestamps - trial_start_time,” and the trajectory shows the agent validated the timestamp spacing empirically.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It is frame-aligned: both the neural matrix and the time vector use the same `start:stop` slice, so each neural column corresponds to one timestamp entry.

ii. ```python
n_tp = stop - start
trial_neural = neural_data[start:stop, :].T.astype(np.float32)
time_from_start = (timestamps[start:stop] - timestamps[start]).astype(np.float32)

input_arr = np.array([
    time_from_start,
    np.full(n_tp, env_per_trial[i], dtype=np.float32),
    np.full(n_tp, i, dtype=np.float32),
    np.full(n_tp, prev_outcome[i], dtype=np.float32),
], dtype=np.float32)
```

iii. The notes describe this input as time-varying and aligned to trial start, and the code implements it with the same trial slice used for neural activity.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. In the converter, environment type is not taken from the NWB `environment/data` stream. It is derived from the session `identifier` string by parsing the scene name, then applying the session’s switch structure.

ii. ```python
identifier = f['identifier'][()].decode()
scene_info = parse_scene(identifier)
...
env_per_trial = get_env_per_trial(scene_info, n_trials)
```

iii. The trajectory shows the agent inspected `environment/data` but ultimately decided that the scene string plus the known switch trial were sufficient. `CONVERSION_NOTES.md` Step 5 lists “scene name” as the practical source of environment identity.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The scene parser identifies `Env1` or `Env2` before and after any switch. `get_env_per_trial` then assigns 0 for Env1 and 1 for Env2, switching from the pre-switch to the post-switch environment at trial 30 only for environment-switch sessions, and finally broadcasts the per-trial value across all timepoints.

ii. ```python
def get_env_per_trial(scene_info, n_trials, change_trial=CHANGE_TRIAL):
    env_map = {'Env1': 0, 'Env2': 1}
    env_vals = []
    for i in range(n_trials):
        if is_env_switch and i >= change_trial:
            env_vals.append(env_map.get(env_after, 0))
        else:
            env_vals.append(env_map.get(env_before, 0))

input_arr = np.array([
    time_from_start,
    np.full(n_tp, env_per_trial[i], dtype=np.float32),
    ...
], dtype=np.float32)
```

iii. The notes say the agent used the code’s `change_trial=30` convention and the trajectory shows it verified that environment-switch sessions do in fact change around trial 30.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. The final trial number input is not taken from the NWB `trial number/data` field. It is derived from the trial loop index `i`, i.e. the order of trial intervals obtained from `trial_start` and `teleport`.

ii. ```python
for i in range(n_trials):
    start = trial_starts[i]
    stop = teleports[i]
    ...
    input_arr = np.array([
        time_from_start,
        np.full(n_tp, env_per_trial[i], dtype=np.float32),
        np.full(n_tp, i, dtype=np.float32),
        np.full(n_tp, prev_outcome[i], dtype=np.float32),
    ], dtype=np.float32)
```

iii. The trajectory shows the agent inspected `trial number/data`, but Step 1 of the notes also records that the reference `get_timeseries_data()` writes `trial_ids` from the loop index rather than copying a raw trial-number channel.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The agent assigns each trial its zero-based within-session index and broadcasts that scalar across all frames in the trial.

ii. ```python
np.full(n_tp, i, dtype=np.float32)  # trial number within session
```

iii. `CONVERSION_NOTES.md` Step 5 explicitly says “trial index” for this mapping, which is consistent with the reference code behavior the agent documented earlier.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived indirectly from the reward event timestamps in `processing/behavior/BehavioralTimeSeries/Reward/timestamps` together with the trial boundaries.

ii. ```python
reward_timestamps = behav['Reward/timestamps'][()]
...
trial_start_time = timestamps[start]
trial_end_time = timestamps[stop - 1] if stop > start else timestamps[start]
reward_in_trial = np.any(
    (reward_timestamps >= trial_start_time) &
    (reward_timestamps <= trial_end_time)
)
is_rewarded[i] = int(reward_in_trial)
```

iii. The notes say reward outcome comes from reward events, and the trajectory shows the agent inspected omission rates and reward event counts before finalizing this logic.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. The agent first computes `is_rewarded` for every trial, then shifts that array by one trial so trial `i` gets the outcome of trial `i-1`; the first trial is assigned 0.

ii. ```python
prev_outcome = np.zeros(n_trials, dtype=int)
for i in range(1, n_trials):
    prev_outcome[i] = is_rewarded[i - 1]
```

iii. `CONVERSION_NOTES.md` Step 5 describes this input as “previous trial reward, 0=omitted, 1=rewarded,” and the first-trial fallback of 0 appears to be the agent’s own practical choice.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It is derived from the per-frame position trace plus the per-trial reward-zone coordinates inferred from the parsed scene identity and switch trial.

ii. ```python
position = behav['position/data'][()]
scene_info = parse_scene(identifier)
rz_labels, rz_coords = get_reward_zone_per_trial(scene_info, n_trials)
...
rz_start, rz_end = rz_coords[i]
trial_dist = distance_to_reward_zone(trial_pos, rz_start, rz_end)
```

iii. The notes say reward-zone labels and coordinates come from the scene structure, matching the reference `get_reward_zones()` logic the agent extracted from `behavior.py`.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. The position trace is first clipped into `[0, 450]` cm. The agent then computes signed distance to the nearest point in the reward-zone interval: negative before the zone, 0 inside it, positive after it.

ii. ```python
trial_pos = position[start:stop].astype(np.float32)
trial_pos = np.clip(trial_pos, TRACK_START, TRACK_LENGTH)

def distance_to_reward_zone(position, rz_start, rz_end):
    dist = np.zeros_like(position, dtype=float)
    before_mask = position < rz_start
    in_mask = (position >= rz_start) & (position <= rz_end)
    after_mask = position > rz_end
    dist[before_mask] = position[before_mask] - rz_start
    dist[in_mask] = 0.0
    dist[after_mask] = position[after_mask] - rz_end
    return dist
```

iii. `CONVERSION_NOTES.md` Step 5 records this as “position - reward zone,” and the code comments state the intended interpretation explicitly.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The signed distance is mapped into the seven decoder bins specified in the task.

ii. ```python
bins[dist < -50] = 0
bins[(dist >= -50) & (dist < -10)] = 1
bins[(dist >= -10) & (dist < 0)] = 2
bins[dist == 0] = 3
bins[(dist > 0) & (dist <= 10)] = 4
bins[(dist > 10) & (dist <= 50)] = 5
bins[dist > 50] = 6
```

iii. This matches the task specification exactly. The notes present the same seven-bin mapping in Step 5.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. It is computed from the same per-trial `start:stop` position slice used to define the neural trial matrix, so one distance label exists for every neural frame.

ii. ```python
trial_neural = neural_data[start:stop, :].T.astype(np.float32)
trial_pos = position[start:stop].astype(np.float32)
trial_dist = distance_to_reward_zone(trial_pos, rz_start, rz_end)
dist_disc = discretize_distance(trial_dist)
```

iii. The notes emphasize trial-start temporal alignment for all time-varying variables, and this output follows that shared frame grid.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It comes directly from the NWB `position/data` behavioral trace.

ii. ```python
position = behav['position/data'][()]
...
trial_pos = position[start:stop].astype(np.float32)
```

iii. Step 1 of the notes maps reference `sess.vr_data['pos']` onto the NWB position channel, which is the variable used here.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The per-trial position is clipped to the valid 0 to 450 cm track range and then discretized.

ii. ```python
trial_pos = position[start:stop].astype(np.float32)
trial_pos = np.clip(trial_pos, TRACK_START, TRACK_LENGTH)
pos_disc = discretize_position(trial_pos)
```

iii. The trajectory shows the agent inspected the raw position range and found off-track values such as `-500` during inter-trial periods, which motivated the clipping step.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. The 450 cm track is divided into five equal bins of 90 cm each using `np.digitize`.

ii. ```python
def discretize_position(position, n_bins=5):
    bin_edges = np.linspace(TRACK_START, TRACK_LENGTH, n_bins + 1)
    bins = np.digitize(position, bin_edges[1:-1])
    bins = np.clip(bins, 0, n_bins - 1)
    return bins
```

iii. `CONVERSION_NOTES.md` Step 5 states “5 equal bins (90 cm each),” which came directly from the decoder task instructions.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Position is sliced with the same trial start and teleport indices as the neural data, so the discretized position vector is frame-aligned with the neural matrix.

ii. ```python
trial_neural = neural_data[start:stop, :].T.astype(np.float32)
trial_pos = position[start:stop].astype(np.float32)
pos_disc = discretize_position(trial_pos)
```

iii. The notes say all decoder variables are aligned on the trial-start frame grid; absolute position follows that same pattern.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It comes from the NWB `lick/data` time series.

ii. ```python
lick = behav['lick/data'][()]
```

iii. The notes identify reference `sess.timeseries['licks']` as the source, and the trajectory shows the agent verified that NWB `lick/data` is a cumulative lick-count stream.

## 9-b. What processing is involved in computing `output` *Lick*?

i. The agent applies a lick-sensor-error heuristic: if more than 35% of frames in a trial have cumulative lick count greater than 2, it marks that trial’s lick values as `NaN`. It then caps all positive lick counts above 1 to 1 and finally converts any `NaN` values to 0, so corrupted lick traces become “no lick” in the exported data.

ii. ```python
lick_corrected = lick.copy()
for i in range(n_trials):
    start = trial_starts[i]
    stop = teleports[i]
    trial_licks = lick_corrected[start:stop]
    if len(trial_licks) > 0:
        frac_high = np.sum(trial_licks > 2) / len(trial_licks)
        if frac_high > 0.35:
            lick_corrected[start:stop] = np.nan

lick_binary = lick_corrected.copy()
lick_binary[lick_binary > 1] = 1
lick_binary[np.isnan(lick_binary)] = 0
```

iii. The notes justify using the code’s 0.35 threshold instead of the paper’s “>30%” wording, and Step 5 explicitly says the agent decided to set erroneous lick data to 0 for the decoder.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. The lick output is the corrected binary lick vector sliced over the same `start:stop` frame interval as the neural trial matrix, then re-thresholded as integers.

ii. ```python
trial_lick = lick_binary[start:stop].astype(np.float32)
lick_disc = (trial_lick > 0).astype(int)
trial_neural = neural_data[start:stop, :].T.astype(np.float32)
```

iii. The notes describe lick as a time-varying output on the imaging-frame grid, which is exactly how the code constructs it.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It is derived from the NWB `identifier` string, specifically the encoded scene name such as `Env1_LocationB_to_A`, not from the `reward_zone/data` signal itself.

ii. ```python
identifier = f['identifier'][()].decode()
scene_info = parse_scene(identifier)
rz_labels, rz_coords = get_reward_zone_per_trial(scene_info, n_trials)
```

iii. The trajectory shows the agent compared the scene string against the reward-zone positions in the NWB stream and concluded that the scene metadata is the cleanest way to recover the active reward zone, matching the reference `get_reward_zones()` logic.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The parser extracts the pre-switch and post-switch zone labels from the scene name, applies the `change_trial=30` rule for switch sessions, and maps `A/B/C` to categorical values `0/1/2`. That per-trial category is then broadcast across all frames in the trial.

ii. ```python
rz_labels, rz_coords = get_reward_zone_per_trial(scene_info, n_trials)
...
rz_label_map = {'A': 0, 'B': 1, 'C': 2}
rz_loc = rz_label_map[rz_labels[i]]
...
np.full(n_tp, rz_loc, dtype=int)
```

iii. `CONVERSION_NOTES.md` Step 5 says this output comes from “scene name,” and the trajectory shows the agent verified the trial-30 switch empirically against reward-zone positions.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Reward outcome is derived from the NWB reward-event timestamps together with trial boundaries and per-frame timestamps.

ii. ```python
reward_timestamps = behav['Reward/timestamps'][()]
timestamps = behav['position/timestamps'][()]
...
trial_start_time = timestamps[start]
trial_end_time = timestamps[stop - 1] if stop > start else timestamps[start]
reward_in_trial = np.any(
    (reward_timestamps >= trial_start_time) &
    (reward_timestamps <= trial_end_time)
)
is_rewarded[i] = int(reward_in_trial)
```

iii. The notes map reward outcome to “reward events,” and the agent checked reward-event counts against omission rates during exploration.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, the agent checks whether any reward timestamp falls between the trial’s first and last behavioral timestamp. If yes, the trial is labeled 1; otherwise 0. That scalar is then broadcast across the trial’s timepoints.

ii. ```python
reward_in_trial = np.any(
    (reward_timestamps >= trial_start_time) &
    (reward_timestamps <= trial_end_time)
)
is_rewarded[i] = int(reward_in_trial)
...
reward_out = is_rewarded[i]
np.full(n_tp, reward_out, dtype=int)
```

iii. `CONVERSION_NOTES.md` Step 5 describes reward outcome as binary per trial, and this windowed event-detection rule is how the code operationalizes it.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The agent truncates neural and behavioral streams to the shorter common length when they differ, truncates unmatched `trial_start` and `teleport` arrays to the shorter count, skips degenerate trials, clips negative speed to 0, clips position to `[0, 450]`, ignores NaNs during the interneuron-correlation calculation, and converts lick-error NaNs to 0 in the exported output.

ii. ```python
if n_behav != n_neural:
    min_len = min(n_behav, n_neural)
    ...
    deconv_data = deconv_data[:min_len, :]

if len(teleports) != n_trials:
    n_trials = min(n_trials, len(teleports))
    trial_starts = trial_starts[:n_trials]
    teleports = teleports[:n_trials]

trial_speed = speed[start:stop].astype(np.float32)
trial_speed = np.clip(trial_speed, 0, None)
trial_pos = np.clip(trial_pos, TRACK_START, TRACK_LENGTH)

both_valid = ~np.isnan(valid_neural) & ~np.isnan(valid_speed)
...
lick_binary[np.isnan(lick_binary)] = 0
```

iii. The notes explicitly call out “length mismatch,” “lick sensor error,” and “no speed filtering” as handling decisions. The trajectory shows these were pragmatic fixes after exploring imperfections in the NWB files.

## 13-a. What are the most time-consuming steps of the code?

i. The likely bottlenecks are reading large NWB arrays from disk, concatenating multi-plane deconvolved data, the per-cell speed-correlation loop for interneuron exclusion, the per-trial construction of neural/input/output arrays, and serializing the final multi-gigabyte pickle.

ii. ```python
for plane in planes:
    deconv_planes.append(f[f'processing/ophys/Deconvolved/{plane}/data'][()])
deconv_data = np.concatenate(deconv_planes, axis=1)

for c in range(deconv_data.shape[1]):
    if cell_mask[c]:
        ...
        r = np.corrcoef(valid_neural[both_valid], valid_speed[both_valid])[0, 1]

for i in range(n_trials):
    ...
    neural_trials.append(trial_neural)
    input_trials.append(input_arr)
    output_trials.append(output_arr)

with open(args.output, 'wb') as f:
    pickle.dump(data, f, protocol=4)
```

iii. `CONVERSION_NOTES.md` Step 6 reports about 1.3 s per session and ~200 s total, which is consistent with I/O-heavy processing plus Python loops over cells and trials.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The biggest vectorization candidates are the per-cell correlation loop, the per-trial reward detection loop, the per-trial lick-error loop, and parts of the per-trial feature construction loop.

ii. ```python
for c in range(deconv_data.shape[1]):
    ...

for i in range(n_trials):
    trial_start_time = timestamps[start]
    trial_end_time = timestamps[stop - 1] ...
    reward_in_trial = np.any(...)

for i in range(n_trials):
    trial_licks = lick_corrected[start:stop]
    ...

for i in range(n_trials):
    ...
    input_arr = np.array([...])
    output_arr = np.array([...])
```

iii. These loops are all explicit in `convert_data.py`, and none of them depend on complex Python objects. They are the obvious candidates for batched or vectorized computation.

## 13-c. What processing does the code repeat multiple times?

i. The code makes multiple passes over the same trial list: once to determine rewards, once to correct lick errors, once to build `prev_outcome`, and once to assemble trial-level arrays. It also re-traverses the entire converted dataset at the end to compute output distributions for logging.

ii. ```python
for i in range(n_trials):
    ... is_rewarded[i] = int(reward_in_trial)

for i in range(n_trials):
    ... lick_corrected[start:stop] = np.nan

for i in range(1, n_trials):
    prev_outcome[i] = is_rewarded[i - 1]

for i in range(n_trials):
    ... neural_trials.append(trial_neural)

for out_idx, out_name in enumerate(data['output_names']):
    for sess_outputs in all_output:
        for trial_output in sess_outputs:
            all_vals.extend(trial_output[out_idx].tolist())
```

iii. This repetition follows from the agent building the converter incrementally rather than organizing all per-trial computations into one shared pass. The notes emphasize correctness and validation over efficiency.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several loaded or computed items are not used for the exported decoder dataset: `reward_zone_signal`, `environment`, and most of `trial_number` are loaded but not used to define final outputs; `REWARD_ZONE_CENTERS`, `SAMPLING_RATE`, and `all_subjects` are unused; the plotting path and end-of-run output-distribution summaries are diagnostic only; and the converter computes summary metadata that the downstream decoder does not consume directly.

ii. ```python
reward_zone_signal = behav['reward_zone/data'][()]
environment = behav['environment/data'][()]
trial_number = behav['trial number/data'][()]

REWARD_ZONE_CENTERS = {k: (v[0] + v[1]) / 2 for k, v in REWARD_ZONE_DICT.items()}
SAMPLING_RATE = 15.5
all_subjects = []

if show_processing:
    plot_processing(...)

for out_idx, out_name in enumerate(data['output_names']):
    ...
    print(f"  {out_name}:")
```

iii. The notes present several of these as sanity checks or convenience values rather than essential conversion steps. They help document or debug the run but are not required by the downstream decoder input format.
