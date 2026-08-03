# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI finds all NWB files by globbing `data/sub-*/*.nwb` and processes each file as a session. Data is loaded using `h5py` directly (not `pynwb`). Each NWB file contains behavioral and neural data for one session.

ii.
```python
all_files = sorted(glob.glob('data/sub-*/*.nwb'))
...
with h5py.File(nwb_path, 'r') as f:
    identifier = f['identifier'][()].decode()
    behav = f['processing/behavior/BehavioralTimeSeries']
    position = behav['position/data'][()]
    speed = behav['speed/data'][()]
    lick = behav['lick/data'][()]
    ...
    deconv_data = np.concatenate(deconv_planes, axis=1)
```

iii. The AI found 152 NWB files across 11 subjects, matching the paper's count. Using `h5py` instead of `pynwb` is functionally equivalent for reading data.

## 1-b. How are the data split into subjects?

i. Subjects are identified by parsing the NWB filename prefix (e.g., `sub-m11`). Unique subjects are tracked in a list as sessions are processed.

ii.
```python
subj_name = os.path.basename(nwb_path).split('_')[0]  # e.g., sub-m11
...
if subj_name not in subject_list:
    subject_list.append(subj_name)
subj_idx = subject_list.index(subj_name)
```

iii. The AI correctly identifies 11 subjects from the filenames.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. All NWB files are found via glob and processed individually.

ii.
```python
all_files = sorted(glob.glob('data/sub-*/*.nwb'))
for file_idx, nwb_path in enumerate(files_to_process):
    result = process_session(nwb_path, ...)
```

iii. File naming convention makes this straightforward. The AI found 152 sessions total.

## 1-d. How are the data split into trials?

i. Trial boundaries are determined by finding indices where `trial_start_signal > 0` (trial starts) and `teleport_signal > 0` (trial ends). Each trial spans from `trial_starts[i]` to `teleports[i]`.

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
    trial_neural = neural_data[start:stop, :].T
```

iii. The AI uses `np.where(teleport_signal > 0)` to find ALL timepoints where teleport is positive, rather than detecting the rising edge (onset) of the teleport signal. If the teleport signal is high for multiple consecutive timepoints, this would produce far more indices than trial count, leading to mismatched trial boundaries. The reference uses rising edge detection: `(teleport[1:] > 0) & (teleport[:-1] <= 0)`.

## 1-e. How are trials filtered based on quality controls?

i. Sessions with fewer than 2 trials are skipped. Trials where `stop <= start` are also skipped. No minimum timepoint threshold is applied.

ii.
```python
if len(neural_trials) < 2:
    print(f"  SKIPPING: only {len(neural_trials)} trials")
    continue
...
if stop <= start:
    continue
```

iii. The AI does not filter short trials (unlike the reference which filters trials with < 50 timepoints).

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from the `Deconvolved` data containers under `processing/ophys/Deconvolved/`, with planes concatenated for multi-plane sessions.

ii.
```python
planes = sorted(f['processing/ophys/Deconvolved'].keys())
deconv_planes = []
for plane in planes:
    deconv_planes.append(f[f'processing/ophys/Deconvolved/{plane}/data'][()])
deconv_data = np.concatenate(deconv_planes, axis=1)
```

iii. The paper specifies training decoders on deconvolved calcium events. Multi-plane sessions (m17, m18) are handled by concatenating planes.

## 2-b. How is the `neural` data processed?

i. After loading and concatenating planes, cells are filtered by `iscell` and then an additional interneuron exclusion step is applied (cells with speed correlation > 0.5 are removed).

ii.
```python
iscell = f['processing/ophys/ImageSegmentation/PlaneSegmentation/iscell'][()]
cell_mask = iscell[:, 0] == 1
...
speed_corr = np.zeros(deconv_data.shape[1])
for c in range(deconv_data.shape[1]):
    if cell_mask[c]:
        r = np.corrcoef(valid_neural[both_valid], valid_speed[both_valid])[0, 1]
        speed_corr[c] = r if not np.isnan(r) else 0
interneuron_mask = speed_corr > 0.5
cell_mask = cell_mask & ~interneuron_mask
neural_data = deconv_data[:, cell_indices]
```

iii. The paper mentions excluding putative interneurons based on speed-dF/F correlation > 0.5. The AI implements this using deconvolved events correlated with speed as an approximation.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two-stage filtering: (1) `iscell` flag from Suite2p, and (2) putative interneuron exclusion via speed correlation > 0.5.

ii. See 2-b code snippets.

iii. The AI loads `iscell` from a single `ImageSegmentation/PlaneSegmentation/iscell` path, which is a single concatenated array. The reference loads iscell per-plane from each ROI response series. For multi-plane sessions, this difference could cause incorrect filtering if the iscell dimensions don't match.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Data is aligned to trial start. Neural data is sliced from `trial_starts[i]` to `teleports[i]`, starting at trial onset.

ii.
```python
trial_neural = neural_data[start:stop, :].T.astype(np.float32)
```

iii. The instructions specify alignment to trial start, so no offset is needed.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The native imaging rate (~64.48 ms per frame, ~15.5 Hz) is preserved. No temporal rebinning is applied. The time bin size is computed from the median of timestamp differences.

ii.
```python
dt = np.median(np.diff(timestamps))
...
dt_ms = session_infos[0]['dt'] * 1000  # convert to ms
```

iii. The AI uses the first session's dt for the metadata. The reference computes the modal time bin size across all sessions.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Derived from the `position/timestamps` array in the behavioral data.

ii.
```python
timestamps = behav['position/timestamps'][()]
...
time_from_start = (timestamps[start:stop] - timestamps[start]).astype(np.float32)
```

iii. The timestamps associated with position are used (any behavioral time series would give the same result since they share timestamps).

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The trial start timestamp is subtracted from all timepoints within the trial.

ii.
```python
time_from_start = (timestamps[start:stop] - timestamps[start]).astype(np.float32)
```

iii. Straightforward subtraction to get time relative to trial onset.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Same time indices are used for both neural and behavioral data. A length mismatch check truncates both to the minimum length.

ii.
```python
if n_behav != n_neural:
    min_len = min(n_behav, n_neural)
    ...
    timestamps = timestamps[:min_len]
    deconv_data = deconv_data[:min_len, :]
```

iii. Neural and behavioral data share the same sampling rate and indices.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. The AI derives environment type from parsing the NWB file's `identifier` string (scene name), NOT from the `environment` behavior time series. The scene name encodes environment info (e.g., "Env1_C_to_Env2_B").

ii.
```python
identifier = f['identifier'][()].decode()
scene_info = parse_scene(identifier)
...
env_per_trial = get_env_per_trial(scene_info, n_trials)
```

iii. The AI parses the scene name to determine environment type, with a fixed switch at trial 30.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The `parse_scene()` function extracts environment info from the identifier string. For switch sessions, trials before trial 30 get the "before" environment, trials >= 30 get the "after" environment.

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

iii. Uses a hardcoded switch trial of 30 (from "30 warm-up trials" in the paper). This is fragile compared to reading the actual environment signal from the data.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is the within-session trial index `i` from the processing loop.

ii.
```python
for i in range(n_trials):
    ...
    np.full(n_tp, i, dtype=np.float32),  # trial number within session
```

iii. Simple sequential numbering within each session.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No processing beyond assigning the loop index. The value is constant across all timepoints within a trial.

ii.
```python
np.full(n_tp, i, dtype=np.float32),  # trial number within session
```

iii. Matches the reference approach.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from `Reward/timestamps` in the behavioral data. Reward timestamps are compared against trial time windows to determine if each trial was rewarded.

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

iii. The AI checks reward timestamps against trial time windows.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For trial i, the previous trial outcome is `is_rewarded[i-1]`. For the first trial, it is set to 0.

ii.
```python
prev_outcome = np.zeros(n_trials, dtype=int)
for i in range(1, n_trials):
    prev_outcome[i] = is_rewarded[i - 1]
...
np.full(n_tp, prev_outcome[i], dtype=np.float32),
```

iii. Standard previous-trial lookup. Matches the reference approach logically.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from the `position` behavior time series and reward zone boundaries. The reward zone for each trial comes from parsing the NWB identifier scene name (not from the `reward_zone` behavior signal).

ii.
```python
rz_labels, rz_coords = get_reward_zone_per_trial(scene_info, n_trials)
...
rz_start, rz_end = rz_coords[i]
trial_dist = distance_to_reward_zone(trial_pos, rz_start, rz_end)
```

iii. Reward zone assignment is based on metadata parsing rather than data-driven Viterbi approach.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance: negative before zone, 0 inside zone, positive after zone. Computed using the `distance_to_reward_zone()` function.

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
    return dist
```

iii. Logic matches the reference's `compute_distance_to_reward_zone()`.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Discretized into 7 bins using explicit conditional logic in `discretize_distance()`.

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

iii. Bin boundaries match the instructions. The bin for "0 cm" uses exact equality (`dist == 0`), while the reference uses a small epsilon range (`0` to `1e-6`). Both capture points inside the reward zone where distance is exactly 0.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Same time indices as neural data within each trial. No additional alignment needed.

ii.
```python
trial_pos = position[start:stop].astype(np.float32)
trial_dist = distance_to_reward_zone(trial_pos, rz_start, rz_end)
```

iii. All data uses the same `start:stop` slice.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from the `position` behavior time series.

ii.
```python
position = behav['position/data'][()]
...
trial_pos = position[start:stop].astype(np.float32)
trial_pos = np.clip(trial_pos, TRACK_START, TRACK_LENGTH)
```

iii. Position is clipped to [0, 450] before discretization. The reference does not clip.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Position is clipped to [0, 450], then discretized into 5 bins using `np.linspace(0, 450, 6)` = [0, 90, 180, 270, 360, 450].

ii.
```python
def discretize_position(position, n_bins=5):
    bin_edges = np.linspace(TRACK_START, TRACK_LENGTH, n_bins + 1)
    bins = np.digitize(position, bin_edges[1:-1])
    bins = np.clip(bins, 0, n_bins - 1)
    return bins
```

iii. The bin edges [0, 90, 180, 270, 360, 450] are different from the reference's `[-inf, 50, 150, 250, 350, inf]`. Both produce 5 bins but with substantially different boundaries.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. 5 bins of 90 cm each: [0-90), [90-180), [180-270), [270-360), [360-450].

ii.
```python
bin_edges = np.linspace(TRACK_START, TRACK_LENGTH, n_bins + 1)  # [0, 90, 180, 270, 360, 450]
bins = np.digitize(position, bin_edges[1:-1])  # uses inner edges [90, 180, 270, 360]
bins = np.clip(bins, 0, n_bins - 1)
```

iii. The reference uses bin edges `[-inf, 50, 150, 250, 350, inf]` giving bins: <50, 50-150, 150-250, 250-350, >350 (100 cm wide). The AI's bins are 90 cm wide and centered differently. Neither matches the instruction "5 equal-sized bins" perfectly, but the reference's bins (100 cm) are more consistent with the position range observed in the data (-50 to 450, roughly 500 cm).

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same time indices as neural data within each trial.

ii.
```python
trial_pos = position[start:stop].astype(np.float32)
```

iii. Same `start:stop` slice used for all data streams.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from the `lick` behavior time series.

ii.
```python
lick = behav['lick/data'][()]
```

iii. Direct read from the NWB file.

## 9-b. What processing is involved in computing `output` *Lick*?

i. The AI applies lick sensor error correction following the reference code: if >35% of samples in a trial have cumulative lick count > 2, set the entire trial's lick data to NaN. Lick values > 1 are capped to 1 (binary). NaN values are treated as 0.

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
```

iii. The AI added lick sensor error correction from the reference paper's analysis code. The reference conversion code does not apply this correction, simply thresholding at > 0.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same time indices as neural data within each trial.

ii.
```python
trial_lick = lick_binary[start:stop].astype(np.float32)
```

iii. Same `start:stop` slice.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from parsing the NWB file's `identifier` string (scene name). The scene name encodes the reward zone location (e.g., "Env1_LocationC" or "Env1_C_to_Env2_B").

ii.
```python
identifier = f['identifier'][()].decode()
scene_info = parse_scene(identifier)
rz_labels, rz_coords = get_reward_zone_per_trial(scene_info, n_trials)
```

iii. The AI does NOT use the `reward_zone` behavior time series from the NWB file. Instead, it parses metadata from the file identifier. The reference uses the `reward_zone` behavior signal combined with position data and a Viterbi algorithm to assign zones.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The `parse_scene()` function extracts the reward zone letter from the identifier. For switch sessions, trials before trial 30 get the "before" zone, trials >= 30 get the "after" zone. The zone letter is mapped to an integer: A=0, B=1, C=2.

ii.
```python
def get_reward_zone_per_trial(scene_info, n_trials, change_trial=CHANGE_TRIAL):
    for i in range(n_trials):
        if is_switch and i >= change_trial:
            label = rz_after
        else:
            label = rz_before
        rz_labels.append(label)
...
rz_label_map = {'A': 0, 'B': 1, 'C': 2}
rz_loc = rz_label_map[rz_labels[i]]
```

iii. The hardcoded switch at trial 30 could be inaccurate if the actual switch trial varies. The reference's Viterbi approach is more robust.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from `Reward/timestamps` in the behavioral data.

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

iii. Checks whether any reward event timestamp falls within the trial's time window.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, the reward outcome is 1 if any reward timestamp falls within [trial_start_time, trial_end_time], 0 otherwise. The value is constant across all timepoints in the trial.

ii.
```python
output_arr = np.array([
    ...
    np.full(n_tp, reward_out, dtype=int),
], dtype=int)
```

iii. Straightforward per-trial binary classification.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases handled:
- **Neural/behavior length mismatch**: Truncate both to minimum length.
- **Teleport/trial-start count mismatch**: Truncate to minimum count with a warning.
- **Empty trials**: Trials where `stop <= start` are skipped.
- **Lick sensor errors**: Trials with >35% high lick counts get NaN licks (set to 0).
- **Negative speeds**: Clipped to 0.
- **Position out of range**: Clipped to [0, 450].

ii.
```python
if n_behav != n_neural:
    min_len = min(n_behav, n_neural)
    ...
if len(teleports) != n_trials:
    n_trials = min(n_trials, len(teleports))
trial_pos = np.clip(trial_pos, TRACK_START, TRACK_LENGTH)
trial_speed = np.clip(trial_speed, 0, None)
```

iii. The AI takes a defensive approach, clipping and truncating to handle edge cases. The reference handles neural/behavior mismatches similarly but does not clip position or speed.

## 13-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading NWB files with `h5py` and reading large neural data arrays. The AI reports ~1.3s per session, ~200s total for all 152 sessions.

ii. N/A (timing is printed during execution)

iii. Using `h5py` directly is likely faster than `pynwb` since it avoids the overhead of NWB object construction.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-cell speed correlation loop for interneuron exclusion iterates over all cells:
```python
for c in range(deconv_data.shape[1]):
    if cell_mask[c]:
        r = np.corrcoef(valid_neural[both_valid], valid_speed[both_valid])[0, 1]
```
This could be vectorized using matrix operations. The per-trial loop for building input/output arrays is also sequential but hard to vectorize due to variable trial lengths.

ii. See above.

iii. The interneuron exclusion loop is an extra step not in the reference, adding computation.

## 13-c. What processing does the code repeat multiple times?

i. The code does not have a separate survey step that re-reads files. It processes each NWB file once. However, it loads the full neural data array (`deconv_data`) before filtering by `iscell`, which means reading more data than needed.

ii. N/A

iii. The single-pass approach is more efficient than the reference's survey + conversion two-pass approach.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The interneuron exclusion step (speed correlation computation for all cells) is extra processing not present in the reference. The lick sensor error correction is also additional processing. Speed clipping and position clipping are unnecessary if discretization bins handle out-of-range values.

ii.
```python
speed_corr = np.zeros(deconv_data.shape[1])
for c in range(deconv_data.shape[1]):
    ...
```

iii. These steps add computational overhead without clear benefit for the decoder task.
