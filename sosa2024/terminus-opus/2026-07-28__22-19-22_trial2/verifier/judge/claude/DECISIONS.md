# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI uses `h5py` to open NWB files directly (not `pynwb`). It finds all NWB files via `glob.glob('data/sub-*/*.nwb')`, sorting them alphabetically. Each file is opened with `h5py.File(nwb_path, 'r')` and behavioral/neural data are read from the HDF5 groups directly.

ii.
```python
all_files = sorted(glob.glob('data/sub-*/*.nwb'))
...
with h5py.File(nwb_path, 'r') as f:
    identifier = f['identifier'][()].decode()
    behav = f['processing/behavior/BehavioralTimeSeries']
    position = behav['position/data'][()]
    ...
    iscell = f['processing/ophys/ImageSegmentation/PlaneSegmentation/iscell'][()]
    planes = sorted(f['processing/ophys/Deconvolved'].keys())
    deconv_planes = []
    for plane in planes:
        deconv_planes.append(f[f'processing/ophys/Deconvolved/{plane}/data'][()])
    deconv_data = np.concatenate(deconv_planes, axis=1)
```

iii. The AI chose h5py for direct HDF5 access rather than pynwb. The glob pattern finds all NWB files across subject directories.

## 1-b. How are the data split into subjects?

i. Subjects are derived from the NWB file names by parsing the `sub-{mouse}` prefix. A list of unique subjects is built as files are processed.

ii.
```python
subj_name = os.path.basename(nwb_path).split('_')[0]  # e.g., sub-m11
...
if subj_name not in subject_list:
    subject_list.append(subj_name)
subj_idx = subject_list.index(subj_name)
```

iii. Subject names are extracted from filenames. The full `sub-m11` string is stored as the subject name (whereas the reference stores just `m11`).

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. Sessions are discovered via the glob pattern and processed individually.

ii.
```python
all_files = sorted(glob.glob('data/sub-*/*.nwb'))
...
for file_idx, nwb_path in enumerate(files_to_process):
    result = process_session(nwb_path, ...)
```

iii. One NWB file per session is the natural data organization.

## 1-d. How are the data split into trials?

i. Trial starts are found where `trial_start_signal > 0`. Trial ends are found where `teleport_signal > 0`. Data between these indices forms a trial.

ii.
```python
trial_starts = np.where(trial_start_signal > 0)[0]
teleports = np.where(teleport_signal > 0)[0]
n_trials = len(trial_starts)

if len(teleports) != n_trials:
    n_trials = min(n_trials, len(teleports))
    trial_starts = trial_starts[:n_trials]
    teleports = teleports[:n_trials]
...
for i in range(n_trials):
    start = trial_starts[i]
    stop = teleports[i]
    n_tp = stop - start
    trial_neural = neural_data[start:stop, :].T
```

iii. Uses trial_start and teleport signals from behavior data.

## 1-e. How are trials filtered based on quality controls?

i. Sessions with fewer than 2 trials are skipped. Trials with `stop <= start` are skipped. No minimum trial length filter is applied.

ii.
```python
if len(neural_trials) < 2:
    print(f"  SKIPPING: only {len(neural_trials)} trials")
    continue
...
if stop <= start:
    continue
```

iii. The AI checks for degenerate trials but does not apply a minimum timepoint threshold.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the NWB `Deconvolved` field (`processing/ophys/Deconvolved/plane{N}/data`), which is suite2p's own deconvolution of raw fluorescence. This is NOT the same signal the paper analyzes.

ii.
```python
planes = sorted(f['processing/ophys/Deconvolved'].keys())
deconv_planes = []
for plane in planes:
    deconv_planes.append(f[f'processing/ophys/Deconvolved/{plane}/data'][()])
deconv_data = np.concatenate(deconv_planes, axis=1)
```

iii. The AI identified the `Deconvolved` field and used it as the neural signal. The CONVERSION_NOTES identify this as "deconvolved calcium events" and reference the paper's use of deconvolved events (sess.timeseries['events']), but the AI did not realize these are different signals.

## 2-b. How is the `neural` data processed?

i. No processing is applied to the neural data beyond filtering by `iscell` and interneuron exclusion. The raw `Deconvolved` values are used directly. No dF/F computation, no neuropil subtraction, no baseline correction, no smoothing, and no OASIS deconvolution is performed.

ii.
```python
cell_mask = iscell[:, 0] == 1
...
neural_data = deconv_data[:, cell_indices]
...
trial_neural = neural_data[start:stop, :].T.astype(np.float32)
```

iii. The AI treated the `Deconvolved` field as a pre-processed signal ready for use. The paper's actual pipeline computes dF/F from raw Fluorescence and Neuropil traces (subtracting 0.7*Fneu, maximin baseline, Gaussian smoothing, OASIS deconvolution with tau=0.7).

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters are applied: (1) suite2p's `iscell` classification, and (2) putative interneuron exclusion based on correlation between deconvolved events and speed (r > 0.5).

ii.
```python
cell_mask = iscell[:, 0] == 1
...
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

iii. The paper specifies interneuron exclusion with r > 0.5 correlation with speed. The AI applies this but correlates deconvolved events with speed rather than dF/F with speed (as the paper describes).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Data is aligned to trial start. The trial is extracted from `trial_starts[i]` to `teleports[i]`, so the first timepoint corresponds to trial start. No additional alignment is needed.

ii.
```python
trial_neural = neural_data[start:stop, :].T.astype(np.float32)
```

iii. Since the data is sliced starting from the trial start index, the alignment is implicit.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The native sampling rate is used (~15.5 Hz, ~64.48 ms per bin). No temporal rebinning is applied. The time bin size is computed from the median of timestamp differences.

ii.
```python
dt = np.median(np.diff(timestamps))
...
dt_ms = session_infos[0]['dt'] * 1000  # convert to ms
```

iii. No rebinning is necessary as the neural and behavioral data share the same sampling rate.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Derived from the `position/timestamps` behavioral timestamps.

ii.
```python
timestamps = behav['position/timestamps'][()]
...
time_from_start = (timestamps[start:stop] - timestamps[start]).astype(np.float32)
```

iii. The position timestamps are used as the common timebase.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The timestamp at trial start is subtracted from all timestamps within the trial.

ii.
```python
time_from_start = (timestamps[start:stop] - timestamps[start]).astype(np.float32)
```

iii. Standard approach.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Neural and behavioral data share the same time indices within the NWB file, so alignment is implicit through shared indexing.

ii.
```python
trial_neural = neural_data[start:stop, :].T.astype(np.float32)
time_from_start = (timestamps[start:stop] - timestamps[start]).astype(np.float32)
```

iii. Same index range is used for both neural and behavioral data.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Environment type is derived from the NWB `identifier` field (scene name), NOT from the `environment` behavioral time series (which is loaded but not used for this purpose).

ii.
```python
identifier = f['identifier'][()].decode()
scene_info = parse_scene(identifier)
...
env_per_trial = get_env_per_trial(scene_info, n_trials)
```

iii. The AI parses the scene name (e.g., "Env1_C_to_Env2_B") to determine environment type, assuming switches happen at trial 30.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The scene name is parsed to determine whether an environment switch occurs. If so, trials before trial 30 get the "before" environment and trials >= 30 get the "after" environment. Environment is mapped as Env1=0, Env2=1 and broadcast across all timepoints.

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

iii. The AI used the scene name because it contains environment information. The hardcoded switch at trial 30 comes from the paper's "30 warm-up trials" description.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is the within-session loop index (0, 1, 2, ...).

ii.
```python
np.full(n_tp, i, dtype=np.float32),  # trial number within session
```

iii. Sequential index within the session.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No processing beyond using the loop index. The value is constant across all timepoints within a trial.

ii.
```python
np.full(n_tp, i, dtype=np.float32),  # trial number within session
```

iii. Simple sequential indexing.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from the `Reward/timestamps` behavioral time series. Reward events are matched to trial time windows.

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

iii. Reward events have their own timestamps which are compared to trial time windows.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For each trial, check whether the previous trial was rewarded. First trial gets 0.

ii.
```python
prev_outcome = np.zeros(n_trials, dtype=int)
for i in range(1, n_trials):
    prev_outcome[i] = is_rewarded[i - 1]
```

iii. Standard lookback to previous trial.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from the `position` behavioral time series and reward zone coordinates. The reward zone assignment comes from parsing the NWB `identifier` (scene name), not from the `reward_zone` behavioral signal.

ii.
```python
trial_pos = position[start:stop].astype(np.float32)
trial_pos = np.clip(trial_pos, TRACK_START, TRACK_LENGTH)  # clip to [0, 450]
rz_start, rz_end = rz_coords[i]
trial_dist = distance_to_reward_zone(trial_pos, rz_start, rz_end)
```

iii. The AI uses scene-name-derived reward zone boundaries with position data.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. For each timepoint, signed distance is computed: negative if before the zone, 0 if inside, positive if after. Position is first clipped to [0, 450].

ii.
```python
def distance_to_reward_zone(position, rz_start, rz_end):
    dist = np.zeros_like(position, dtype=float)
    before_mask = position < rz_start
    dist[before_mask] = position[before_mask] - rz_start
    after_mask = position > rz_end
    dist[after_mask] = position[after_mask] - rz_end
    return dist
```

iii. Standard signed distance computation.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Discretized into 7 bins using conditional comparisons.

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

iii. The bin edges match the instructions. Minor edge-case differences with the reference's `np.digitize` approach at exact boundary values (e.g., dist==10 goes to bin 4 in AI vs bin 5 in reference, dist==50 goes to bin 5 in AI vs bin 6 in reference).

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Same index range is used for neural and behavioral data within each trial.

ii.
```python
trial_pos = position[start:stop].astype(np.float32)
trial_neural = neural_data[start:stop, :].T.astype(np.float32)
```

iii. Implicit alignment through shared indexing.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from the `position` behavioral time series.

ii.
```python
trial_pos = position[start:stop].astype(np.float32)
trial_pos = np.clip(trial_pos, TRACK_START, TRACK_LENGTH)
```

iii. Direct use of position data with clipping.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Position is clipped to [0, 450] before discretization.

ii.
```python
trial_pos = np.clip(trial_pos, TRACK_START, TRACK_LENGTH)
pos_disc = discretize_position(trial_pos)
```

iii. Clipping ensures position values stay within track bounds.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Discretized into 5 equal-sized bins of 90 cm each using `np.digitize` with edges at [90, 180, 270, 360] (derived from `np.linspace(0, 450, 6)`), clipped to [0, 4].

ii.
```python
def discretize_position(position, n_bins=5):
    bin_edges = np.linspace(TRACK_START, TRACK_LENGTH, n_bins + 1)
    bins = np.digitize(position, bin_edges[1:-1])
    bins = np.clip(bins, 0, n_bins - 1)
    return bins
```

iii. Five 90 cm bins spanning the 450 cm track, matching the instructions.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same index range as neural data within each trial.

ii. Same as 7-d.

iii. Implicit alignment through shared indexing.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from the `lick` behavioral time series.

ii.
```python
lick = behav['lick/data'][()]
```

iii. Direct use of lick data.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Lick sensor error correction is applied first: if >35% of samples in a trial have cumulative lick count >2, the entire trial's lick data is set to NaN, then converted to 0. Remaining licks >1 are capped at 1. Finally, lick is binarized (>0 = 1).

ii.
```python
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

iii. The AI identified the lick sensor error correction from the reference code (`correct_lick_sensor_error` in behavior.py) and applied it. The reference conversion code does NOT apply this correction.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same index range as neural data within each trial.

ii. Same as 7-d.

iii. Implicit alignment through shared indexing.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from the NWB `identifier` field (scene name) by parsing the reward zone letter from the scene description.

ii.
```python
scene_info = parse_scene(identifier)
rz_labels, rz_coords = get_reward_zone_per_trial(scene_info, n_trials)
...
rz_label_map = {'A': 0, 'B': 1, 'C': 2}
rz_loc = rz_label_map[rz_labels[i]]
```

iii. The AI parsed scene names like "Env1_LocationA_to_B" to determine reward zone, assuming switches at trial 30.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Scene name is parsed to extract reward zone letter(s). For switch sessions, trials before trial 30 get the "before" zone and trials >= 30 get the "after" zone. Zone letters are mapped to integers: A=0, B=1, C=2.

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

iii. Hardcoded switch at trial 30 based on paper description.

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

iii. Reward events are matched to trial time windows.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, check if any reward event timestamp falls within the trial's time window. Binary output: 1 if rewarded, 0 if not.

ii.
```python
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

iii. Standard approach matching reward timestamps to trial windows.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **Neural/behavior length mismatch**: Data is truncated to the minimum of neural and behavioral lengths.
- **Trial start/teleport count mismatch**: The minimum count is used.
- **Degenerate trials** (stop <= start): Skipped.
- **Sessions with < 2 trials**: Skipped entirely.
- **Lick sensor errors**: Corrected via the 35% threshold method.
- **Negative speeds**: Clipped to 0.
- **Out-of-range positions**: Clipped to [0, 450].

ii.
```python
if n_behav != n_neural:
    min_len = min(n_behav, n_neural)
    position = position[:min_len]
    ...
    deconv_data = deconv_data[:min_len, :]

if len(teleports) != n_trials:
    n_trials = min(n_trials, len(teleports))
    ...

if stop <= start:
    continue
```

iii. Defensive checks discovered during data exploration.

## 13-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are:
1. Loading NWB files with h5py and reading large arrays (neural data)
2. Computing speed correlations for interneuron exclusion (loops over all cells)

ii. N/A

iii. The code runs at ~1.3s per session, ~200s total, which is efficient since no dF/F computation is performed.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The interneuron exclusion loop iterates over each cell computing correlations, which could be vectorized with matrix operations. The reward detection loop over trials could also be vectorized.

ii.
```python
for c in range(deconv_data.shape[1]):
    if cell_mask[c]:
        ...
        r = np.corrcoef(valid_neural[both_valid], valid_speed[both_valid])[0, 1]
```

iii. The per-cell correlation loop is the natural structure but could be vectorized.

## 13-c. What processing does the code repeat multiple times?

i. No significant repeated processing. The AI processes each session once. Unlike the reference, there is no separate survey step that loads all files before conversion.

ii. N/A

iii. The AI's approach is streamlined in this regard.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The lick sensor error correction is applied but is not part of the reference conversion pipeline. Position clipping and speed clipping are applied but the reference does not clip these values.

ii.
```python
trial_pos = np.clip(trial_pos, TRACK_START, TRACK_LENGTH)
trial_speed = np.clip(trial_speed, 0, None)
```

iii. These are extra processing steps not present in the reference.
