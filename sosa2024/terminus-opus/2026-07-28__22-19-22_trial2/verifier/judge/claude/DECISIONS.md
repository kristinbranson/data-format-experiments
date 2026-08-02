# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI finds all NWB files by globbing `data/sub-*/*.nwb`, sorts them, and processes each file as one session. Data is loaded using `h5py` (not `pynwb`). Each file is opened and behavioral data, neural data, and metadata are extracted directly from HDF5 paths.

ii.
```python
all_files = sorted(glob.glob('data/sub-*/*.nwb'))
...
for file_idx, nwb_path in enumerate(files_to_process):
    result = process_session(nwb_path, show_processing=show, session_idx=file_idx)
```

Inside `process_session`:
```python
with h5py.File(nwb_path, 'r') as f:
    behav = f['processing/behavior/BehavioralTimeSeries']
    position = behav['position/data'][()]
    speed = behav['speed/data'][()]
    lick = behav['lick/data'][()]
    ...
    planes = sorted(f['processing/ophys/Deconvolved'].keys())
    deconv_planes = []
    for plane in planes:
        deconv_planes.append(f[f'processing/ophys/Deconvolved/{plane}/data'][()])
    deconv_data = np.concatenate(deconv_planes, axis=1)
```

iii. The AI's CONVERSION_NOTES state it found 152 NWB files across 11 subjects, consistent with the paper. Using h5py rather than pynwb is an implementation choice for the same underlying HDF5 file format.

## 1-b. How are the data split into subjects?

i. Subjects are identified from the NWB file basename (e.g., `sub-m11` from the filename). Unique subject names are collected during processing and tracked in `subject_list`.

ii.
```python
subj_name = os.path.basename(nwb_path).split('_')[0]  # e.g., sub-m11
...
if subj_name not in subject_list:
    subject_list.append(subj_name)
subj_idx = subject_list.index(subj_name)
```

iii. The filenames follow the pattern `sub-{mouse}_ses-{session}_behavior+ophys.nwb`, so parsing the first underscore-separated token gives the subject ID. This produces 11 unique subjects matching the paper.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. All NWB files are sorted and processed sequentially. Sessions with fewer than 2 valid trials are skipped.

ii.
```python
all_files = sorted(glob.glob('data/sub-*/*.nwb'))
...
for file_idx, nwb_path in enumerate(files_to_process):
    ...
    if len(neural_trials) < 2:
        print(f"  SKIPPING: only {len(neural_trials)} trials")
        continue
```

iii. The one-file-per-session structure is documented in the CONVERSION_NOTES and consistent with the NWB data organization.

## 1-d. How are the data split into trials?

i. Trial boundaries are identified using `trial_start` and `teleport` behavior signals. Trial starts are indices where `trial_start_signal > 0`. Trial ends are indices where `teleport_signal > 0`. The AI uses `np.where(teleport_signal > 0)[0]` to find all positive teleport indices, rather than detecting teleport onset transitions.

ii.
```python
trial_starts = np.where(trial_start_signal > 0)[0]
teleports = np.where(teleport_signal > 0)[0]
n_trials = len(trial_starts)

if len(teleports) != n_trials:
    n_trials = min(n_trials, len(teleports))
    trial_starts = trial_starts[:n_trials]
    teleports = teleports[:n_trials]
```

Then for each trial:
```python
start = trial_starts[i]
stop = teleports[i]
n_tp = stop - start
trial_neural = neural_data[start:stop, :].T.astype(np.float32)
```

iii. The AI notes in CONVERSION_NOTES that trial boundaries come from trial_start and teleport signals. However, it uses `np.where(teleport > 0)` instead of detecting teleport onset transitions, which could give incorrect results if the teleport signal is positive for multiple consecutive samples.

## 1-e. How are trials filtered based on quality controls?

i. Trials are only filtered if `stop <= start` (zero or negative length). There is no minimum timepoint threshold like the reference's 50-timepoint cutoff. Sessions with fewer than 2 valid trials are skipped entirely.

ii.
```python
for i in range(n_trials):
    start = trial_starts[i]
    stop = teleports[i]
    if stop <= start:
        continue
```

iii. The AI does not document or implement a minimum trial length filter beyond zero-length trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is from the `Deconvolved` field under `processing/ophys/Deconvolved/`, read via h5py. Multiple planes are concatenated along the neuron axis.

ii.
```python
planes = sorted(f['processing/ophys/Deconvolved'].keys())
deconv_planes = []
for plane in planes:
    deconv_planes.append(f[f'processing/ophys/Deconvolved/{plane}/data'][()])
deconv_data = np.concatenate(deconv_planes, axis=1)
```

iii. CONVERSION_NOTES: "Neural: `sess.timeseries['events']` (deconvolved calcium events)" — consistent with the paper's use of deconvolved data.

## 2-b. How is the `neural` data processed?

i. The neural data is filtered by `iscell` classification and then by an interneuron exclusion step. The AI correlates each cell's deconvolved activity with speed and excludes cells with Pearson r > 0.5 (putative interneurons). Multiple planes are concatenated.

ii.
```python
iscell = f['processing/ophys/ImageSegmentation/PlaneSegmentation/iscell'][()]
cell_mask = iscell[:, 0] == 1

# Interneuron exclusion
speed_corr = np.zeros(deconv_data.shape[1])
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

iii. The AI notes the paper describes excluding putative interneurons with speed correlation > 0.5 and implements this filtering step.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two-stage filtering: (1) `iscell` from Suite2p classification, and (2) interneuron exclusion based on speed correlation > 0.5. The `iscell` array is loaded from a single `PlaneSegmentation` table rather than per-plane.

ii.
```python
iscell = f['processing/ophys/ImageSegmentation/PlaneSegmentation/iscell'][()]
cell_mask = iscell[:, 0] == 1
...
interneuron_mask = speed_corr > 0.5
cell_mask = cell_mask & ~interneuron_mask
n_cells_final = int(np.sum(cell_mask))
cell_indices = np.where(cell_mask)[0]
neural_data = deconv_data[:, cell_indices]
```

iii. CONVERSION_NOTES: "Neuron curation: Suite2p iscell + interneuron exclusion (speed corr > 0.5)". The AI chose the paper's threshold of 0.5 over the code's default of 0.3.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Data is aligned to trial start. The trial starts at index `trial_starts[i]` and ends at `teleports[i]`. No additional temporal shifting is applied since the alignment event is the trial start itself.

ii.
```python
start = trial_starts[i]
stop = teleports[i]
trial_neural = neural_data[start:stop, :].T.astype(np.float32)
```

iii. The instructions say "Temporally align based on start of the trial." Since the data is sliced from trial_start, no additional alignment is needed.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No temporal rebinning is applied. The native imaging frame rate (~15.5 Hz, ~64.48 ms per bin) is used directly. The time bin size is computed from the median difference of behavioral timestamps.

ii.
```python
dt = np.median(np.diff(timestamps))
...
dt_ms = session_infos[0]['dt'] * 1000  # convert to ms
```

iii. CONVERSION_NOTES: "Using raw imaging frames (~64.48 ms), no additional temporal binning."

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Derived from the `position/timestamps` array in the behavioral time series.

ii.
```python
timestamps = behav['position/timestamps'][()]
...
time_from_start = (timestamps[start:stop] - timestamps[start]).astype(np.float32)
```

iii. The AI uses position timestamps as the reference time base for all behavioral data.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The trial's start timestamp is subtracted from all timestamps within the trial to get time relative to trial start.

ii.
```python
time_from_start = (timestamps[start:stop] - timestamps[start]).astype(np.float32)
```

iii. Standard approach — subtract the first timestamp of the trial.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Neural and behavioral data share the same time indices (same array indexing from `start` to `stop`). After handling length mismatches, both are truncated to the same length.

ii.
```python
n_behav = len(position)
n_neural = deconv_data.shape[0]
if n_behav != n_neural:
    min_len = min(n_behav, n_neural)
    ...
    timestamps = timestamps[:min_len]
    deconv_data = deconv_data[:min_len, :]
```

iii. The AI verifies and handles neural/behavioral length mismatches by truncation.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. The environment type is parsed from the NWB file's `identifier` field (the scene name), NOT from the `environment` behavior time series directly.

ii.
```python
identifier = f['identifier'][()].decode()
scene_info = parse_scene(identifier)
...
env_per_trial = get_env_per_trial(scene_info, n_trials)
```

In `parse_scene`:
```python
def parse_scene(identifier):
    scene = identifier.split('/')[-1]
    if '_to_Env' in scene:
        env_before = before_part.split('_')[0]
        env_after = after_part.split('_')[0]
        return env_before, env_after, ...
    ...
```

iii. The AI chose to derive environment from the session identifier metadata rather than from the per-timepoint `environment` behavioral signal. The AI's approach gives the same result if the metadata is consistent with the recorded signal.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The scene name is parsed to determine if there is an environment switch. For switch sessions, environment changes at trial 30 (the default `CHANGE_TRIAL`). Environment is mapped as Env1=0, Env2=1.

ii.
```python
def get_env_per_trial(scene_info, n_trials, change_trial=CHANGE_TRIAL):
    env_map = {'Env1': 0, 'Env2': 1}
    for i in range(n_trials):
        if is_env_switch and i >= change_trial:
            env_vals.append(env_map.get(env_after, 0))
        else:
            env_vals.append(env_map.get(env_before, 0))
    return env_vals
```

iii. The AI notes: "Switch trial: 30" from the paper's "30 warm-up trials."

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is the loop index `i` within the session, which is the sequential trial index (0-based).

ii.
```python
for i in range(n_trials):
    ...
    input_arr = np.array([
        ...
        np.full(n_tp, i, dtype=np.float32),  # trial number within session
        ...
    ])
```

iii. Trial number is simply the within-session sequential index.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No processing — the loop index is directly used as the trial number. The value is constant across all timepoints in the trial (broadcast via `np.full`).

ii.
```python
np.full(n_tp, i, dtype=np.float32)  # trial number within session
```

iii. Straightforward sequential numbering.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from `Reward/timestamps` in the behavioral time series. Reward event timestamps are compared against each trial's time window to determine if a reward occurred.

ii.
```python
reward_timestamps = behav['Reward/timestamps'][()]
...
is_rewarded = np.zeros(n_trials, dtype=int)
for i in range(n_trials):
    start = trial_starts[i]
    stop = teleports[i]
    trial_start_time = timestamps[start]
    trial_end_time = timestamps[stop - 1] if stop > start else timestamps[start]
    reward_in_trial = np.any(
        (reward_timestamps >= trial_start_time) &
        (reward_timestamps <= trial_end_time)
    )
    is_rewarded[i] = int(reward_in_trial)
```

iii. The AI uses the raw reward event timestamps and checks whether any fall within each trial's time window.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For each trial, the previous trial's reward outcome is used. For the first trial, set to 0.

ii.
```python
prev_outcome = np.zeros(n_trials, dtype=int)
for i in range(1, n_trials):
    prev_outcome[i] = is_rewarded[i - 1]
...
input_arr = np.array([
    ...
    np.full(n_tp, prev_outcome[i], dtype=np.float32),
])
```

iii. The value is constant across all timepoints in the trial.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from the `position` behavioral time series and the reward zone coordinates. The reward zone label (A, B, or C) is determined by parsing the NWB identifier/scene name (not from the `reward_zone` behavior signal).

ii.
```python
rz_labels, rz_coords = get_reward_zone_per_trial(scene_info, n_trials)
...
rz_start, rz_end = rz_coords[i]
trial_dist = distance_to_reward_zone(trial_pos, rz_start, rz_end)
```

iii. The AI derives reward zone from metadata rather than from the behavior signal, using the known zone coordinates from the paper.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance from position to the nearest edge of the assigned reward zone. Zero inside the zone, negative before, positive after. Position is first clipped to [0, 450].

ii.
```python
def distance_to_reward_zone(position, rz_start, rz_end):
    dist = np.zeros_like(position, dtype=float)
    before_mask = position < rz_start
    dist[before_mask] = position[before_mask] - rz_start
    in_mask = (position >= rz_start) & (position <= rz_end)
    dist[in_mask] = 0.0
    after_mask = position > rz_end
    dist[after_mask] = position[after_mask] - rz_end
    return dist
```

Position clipping:
```python
trial_pos = np.clip(trial_pos, TRACK_START, TRACK_LENGTH)
```

iii. The distance computation matches the reference paper's concept of reward-relative distance.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Discretized into 7 bins using explicit conditional assignments matching the instruction's bin specification.

ii.
```python
def discretize_distance(dist):
    bins = np.zeros(len(dist), dtype=int)
    bins[dist < -50] = 0
    bins[(dist >= -50) & (dist < -10)] = 1
    bins[(dist >= -10) & (dist < 0)] = 2
    bins[dist == 0] = 3
    bins[(dist > 0) & (dist <= 10)] = 4
    bins[(dist > 10) & (dist <= 50)] = 5
    bins[dist > 50] = 6
    return bins
```

iii. The bin edges match the instructions.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Same time indices as neural data — both are sliced from `start` to `stop` within each trial.

ii.
```python
trial_pos = position[start:stop].astype(np.float32)
trial_dist = distance_to_reward_zone(trial_pos, rz_start, rz_end)
```

iii. Neural and behavioral data share the same indexing after length mismatch handling.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from the `position` behavioral time series.

ii.
```python
position = behav['position/data'][()]
...
trial_pos = position[start:stop].astype(np.float32)
trial_pos = np.clip(trial_pos, TRACK_START, TRACK_LENGTH)
```

iii. Direct extraction of the VR corridor position.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Position is clipped to [0, 450] cm range before discretization.

ii.
```python
trial_pos = np.clip(trial_pos, TRACK_START, TRACK_LENGTH)
pos_disc = discretize_position(trial_pos)
```

iii. The AI clips position to valid track bounds before discretizing.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Discretized into 5 equal-sized bins using `np.linspace(0, 450, 6)` = [0, 90, 180, 270, 360, 450], giving 90 cm bins.

ii.
```python
def discretize_position(position, n_bins=5):
    bin_edges = np.linspace(TRACK_START, TRACK_LENGTH, n_bins + 1)
    bins = np.digitize(position, bin_edges[1:-1])
    bins = np.clip(bins, 0, n_bins - 1)
    return bins
```

iii. The AI uses "5 equal-sized bins" spanning the track length 0-450 cm. This produces bins of width 90 cm: [0-90), [90-180), [180-270), [270-360), [360-450].

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same time indices as neural data within each trial.

ii.
```python
trial_pos = position[start:stop].astype(np.float32)
```

iii. Same indexing as neural data.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from the `lick` behavioral time series.

ii.
```python
lick = behav['lick/data'][()]
```

iii. Direct loading of the lick behavioral signal.

## 9-b. What processing is involved in computing `output` *Lick*?

i. The AI applies lick sensor error correction (following reference code's `correct_lick_sensor_error` function): if >35% of samples in a trial have lick count > 2, the entire trial's lick data is set to NaN (then treated as 0). Lick values > 1 are capped to 1. The result is binarized (> 0 = lick).

ii.
```python
# Lick sensor error correction
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
...
lick_disc = (trial_lick > 0).astype(int)
```

iii. CONVERSION_NOTES: "Lick sensor error: >35% samples with cumulative lick >2 -> NaN" and "Licks capped at 1 (binary)."

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same time indices as neural data within each trial.

ii.
```python
trial_lick = lick_binary[start:stop].astype(np.float32)
```

iii. Same indexing as neural data.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from the NWB file's `identifier` field, which encodes the scene name (e.g., "Env1_LocationC" or "Env1_C_to_Env2_B"). The scene name is parsed to determine reward zone labels for pre- and post-switch trials.

ii.
```python
identifier = f['identifier'][()].decode()
scene_info = parse_scene(identifier)
rz_labels, rz_coords = get_reward_zone_per_trial(scene_info, n_trials)
...
rz_label_map = {'A': 0, 'B': 1, 'C': 2}
rz_loc = rz_label_map[rz_labels[i]]
```

iii. The AI uses metadata from the NWB identifier rather than inferring from the `reward_zone` behavior signal. The scene name encodes the reward zone location(s) and switch information.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The scene name is parsed to extract pre- and post-switch reward zone labels. For switch sessions, the reward zone changes at trial 30 (`CHANGE_TRIAL`). Labels are mapped to integers: A=0, B=1, C=2.

ii.
```python
def get_reward_zone_per_trial(scene_info, n_trials, change_trial=CHANGE_TRIAL):
    for i in range(n_trials):
        if is_switch and i >= change_trial:
            label = rz_after
        else:
            label = rz_before
        rz_labels.append(label)
        rz_coords.append(REWARD_ZONE_DICT[label])
    return rz_labels, rz_coords
```

iii. The switch at trial 30 matches the paper's "30 warm-up trials."

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from the `Reward/timestamps` behavioral time series.

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

iii. Reward events have separate timestamps from the main behavioral sampling.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, check if any reward event timestamp falls within the trial's time window [trial_start_time, trial_end_time]. Binary: 1 if rewarded, 0 if not.

ii.
```python
is_rewarded = np.zeros(n_trials, dtype=int)
for i in range(n_trials):
    start = trial_starts[i]
    stop = teleports[i]
    trial_start_time = timestamps[start]
    trial_end_time = timestamps[stop - 1] if stop > start else timestamps[start]
    reward_in_trial = np.any(
        (reward_timestamps >= trial_start_time) &
        (reward_timestamps <= trial_end_time)
    )
    is_rewarded[i] = int(reward_in_trial)
...
output_arr[5] = np.full(n_tp, reward_out, dtype=int)
```

iii. Per-trial binary output, constant across timepoints.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases handled:
- **Neural/behavioral length mismatch**: Truncate both to the minimum length.
- **Trial start/teleport count mismatch**: Truncate to the minimum count with a warning.
- **Zero-length trials**: Skipped if `stop <= start`.
- **Sessions with <2 trials**: Skipped entirely.
- **Lick sensor errors**: Trials with >35% high lick values set to NaN then 0.
- **Negative speeds**: Clipped to 0.
- **Position out of range**: Clipped to [0, 450].

ii.
```python
if n_behav != n_neural:
    min_len = min(n_behav, n_neural)
    ...
if len(teleports) != n_trials:
    n_trials = min(n_trials, len(teleports))
if stop <= start:
    continue
if len(neural_trials) < 2:
    continue
trial_pos = np.clip(trial_pos, TRACK_START, TRACK_LENGTH)
trial_speed = np.clip(trial_speed, 0, None)
```

iii. The AI handles edge cases defensively with truncation and clipping.

## 13-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are:
1. Loading NWB files with h5py (I/O bound)
2. Interneuron exclusion — computing per-cell speed correlations in a loop over all cells
3. The per-trial loop building input/output arrays

ii. From CONVERSION_NOTES: "Processing time: ~1.3s per session, ~200s total"

iii. The AI reports timing in the output: "load=X.Xs, total=X.Xs" per session.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The interneuron exclusion loop iterates over each cell individually to compute correlations:
```python
for c in range(deconv_data.shape[1]):
    if cell_mask[c]:
        ...
        r = np.corrcoef(valid_neural[both_valid], valid_speed[both_valid])[0, 1]
```
This could be vectorized as a matrix correlation. The per-trial loop for building input/output arrays could also be partially vectorized.

ii. See code above.

iii. The per-cell loop is the most obvious candidate for vectorization.

## 13-c. What processing does the code repeat multiple times?

i. The AI processes each NWB file only once (unlike the reference which has separate survey and conversion passes). However, within `process_session`, it iterates over trials twice: once for `is_rewarded` computation and once for building trial data. The lick sensor error correction also loops over trials separately.

ii.
```python
# First loop: is_rewarded
for i in range(n_trials):
    ...
    is_rewarded[i] = int(reward_in_trial)

# Second loop: lick correction
for i in range(n_trials):
    ...
    if frac_high > 0.35:
        lick_corrected[start:stop] = np.nan

# Third loop: build trial data
for i in range(n_trials):
    ...
    neural_trials.append(trial_neural)
```

iii. Three separate loops over trials could be merged into one.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The interneuron exclusion step (computing per-cell speed correlations) is additional processing not present in the reference solution. While it follows the paper, the reference human solution did not implement it, and it adds significant computation time. The position and speed clipping are also additional steps not in the reference.

ii.
```python
# Interneuron exclusion adds ~0.5s per session
speed_corr = np.zeros(deconv_data.shape[1])
for c in range(deconv_data.shape[1]):
    ...
```

iii. Whether this is "unnecessary" depends on perspective — the paper describes it, but the reference solution omits it.
