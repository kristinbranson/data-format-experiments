# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads all NWB files from `/app/data/sub-*/` using `h5py` (not `pynwb`). It finds all `.nwb` files with `glob.glob`, sorts them, and processes each one in a single pass through `process_session()`.

ii.
```python
def get_all_nwb_files():
    """Get all NWB file paths, sorted by subject and session."""
    files = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', '*.nwb')))
    return files
```
```python
with h5py.File(nwb_path, 'r') as f:
    identifier = f['identifier'][()].decode() ...
    subject_id = f['general/subject/subject_id'][()].decode() ...
    # ... reads all fields via h5py paths
```

iii. The AI chose h5py for direct file access rather than the pynwb library. Both can read NWB files. The AI processes all files in a single pass rather than the reference's two-pass approach (survey then convert).

## 1-b. How are the data split into subjects?

i. Subjects are identified from the `general/subject/subject_id` field within each NWB file. Unique subjects are accumulated as files are processed.

ii.
```python
subject_id = f['general/subject/subject_id'][()].decode() ...
# ...
subj = result['subject']
if subj not in unique_subjects:
    unique_subjects.append(subj)
```

iii. Subject identifiers come directly from the NWB metadata rather than from directory names.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. The session ID is extracted from the NWB `general/session_id` field.

ii.
```python
session_id = f['general/session_id'][()].decode() ...
```

iii. Same approach as reference: one NWB file = one session.

## 1-d. How are the data split into trials?

i. Trial boundaries are determined by finding all indices where `trial_start_data > 0` (trial starts) and all indices where `teleport_data > 0` (trial ends). If the number of starts and teleport indices don't match, they are trimmed to the minimum length.

ii.
```python
trial_start_inds = np.where(trial_start_data > 0)[0]
teleport_inds = np.where(teleport_data > 0)[0]
n_trials = len(trial_start_inds)

if len(teleport_inds) != n_trials:
    min_len = min(len(trial_start_inds), len(teleport_inds))
    trial_start_inds = trial_start_inds[:min_len]
    teleport_inds = teleport_inds[:min_len]
    n_trials = min_len
```

iii. The AI uses all positive values of the teleport signal rather than detecting the rising edge as the reference does. If teleport stays high for multiple frames, `np.where(teleport_data > 0)` would return many more indices than trials, and the trimming logic would produce incorrect trial boundaries.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies three filters: (1) trials where `te <= ts` are skipped, (2) trials with fewer than 2 valid timepoints after speed filtering are skipped, and (3) a speed threshold of >= 2 cm/s is applied, removing all timepoints where speed is below this threshold from each trial.

ii.
```python
if te <= ts:
    continue
# ...
speed_mask = trial_speed >= SPEED_THRESHOLD
neural_nan_mask = ~np.isnan(trial_neural[:, 0])
valid_mask = speed_mask & neural_nan_mask
if valid_mask.sum() < 2:
    continue
# Apply mask
trial_pos_valid = trial_pos[valid_mask]
trial_neural_valid = trial_neural[valid_mask, :]
```

iii. The AI documented applying the speed threshold based on the paper's mention of excluding activity when the animal was moving at < 2 cm/s. However, the reference conversion code does NOT apply speed filtering -- the paper applies it in specific downstream analyses but the data is stored unfiltered.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI uses the NWB's pre-stored `Deconvolved` data (`processing/ophys/Deconvolved/plane*/data`). It loads deconvolved events from all planes and concatenates them.

ii.
```python
deconv_group = f['processing/ophys/Deconvolved']
plane_keys = sorted([k for k in deconv_group.keys() if k.startswith('plane')])
plane_data = [deconv_group[pk]['data'][:] for pk in plane_keys]
deconv_all = np.concatenate(plane_data, axis=1)  # (T, total_ROIs)
deconv = deconv_all[:, cell_mask]  # (T, N_cells)
```

iii. The AI's CONVERSION_NOTES.md states: "NWB deconvolved data is ready to use" and "We will use the deconvolved events as a proxy, or skip this step." The AI identified but did not implement the paper's own dF/F pipeline.

## 2-b. How is the `neural` data processed?

i. No processing is applied. The AI loads the pre-computed Deconvolved data from the NWB file, applies the iscell mask, and uses it directly. NaN values are replaced with 0.

ii.
```python
deconv = deconv_all[:, cell_mask]  # apply iscell mask
# ...
trial_neural_valid = np.nan_to_num(trial_neural_valid, nan=0.0)
neural_matrix = trial_neural_valid.T.astype(np.float32)
```

iii. The AI chose to use the NWB's stored deconvolved data rather than recomputing the paper's dF/F pipeline (neuropil subtraction with coefficient 0.7, maximin baseline with 300-sample window, Gaussian smoothing with sigma 2, OASIS deconvolution with tau 0.7). The AI noted in CONVERSION_NOTES.md that "dFF is NOT in the NWB" and the "NWB has Deconvolved data already."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only the `iscell` filter is applied. The AI explicitly chose to skip the interneuron exclusion step (filtering cells whose dF/F correlates with speed at r > 0.5).

ii.
```python
iscell = f['processing/ophys/ImageSegmentation/PlaneSegmentation/iscell'][:, 0]
cell_mask = iscell == 1
n_neurons = int(cell_mask.sum())
deconv = deconv_all[:, cell_mask]
```

iii. The AI documented this decision: "Interneuron exclusion: Skip this filtering step. It affects <0.5% of cells and requires dFF computation which is not in the NWB." However, the paper's Methods describe this as a standard step, and the reference solution implements it.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to trial start by extracting the data between trial_start and teleport indices. No additional alignment is needed since data is already at the imaging frame rate.

ii.
```python
trial_neural = deconv[ts:te, :]
```

iii. The alignment is to trial start as specified in the instructions.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI uses the native imaging rate (~15.5 Hz, ~64.5 ms per frame). No temporal rebinning is applied.

ii.
```python
imaging_rate = float(f['acquisition/TwoPhotonSeries/imaging_plane/imaging_rate'][()])
# ...
'time_bin_size': 64.5,  # ms (approximate, ~1/15.5 Hz)
```

iii. The time bin size is hardcoded as 64.5 ms in metadata rather than computed from the actual rate of each session. The reference computes it dynamically per session as `nplanes/rate*1000`.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Derived from the `position` timestamps (`bts['position/timestamps'][:]`).

ii.
```python
timestamps = bts['position/timestamps'][:]
# ...
trial_timestamps = timestamps[ts:te]
time_from_start = trial_times_valid - trial_timestamps[0]
```

iii. All behavioral time series share the same timestamps, so using position timestamps is equivalent to the reference's use of trial_number timestamps.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The first timestamp of the trial is subtracted from all timestamps in the trial. Note that because speed-filtered timepoints are removed, the time-from-start values may have gaps.

ii.
```python
time_from_start = trial_times_valid - trial_timestamps[0]
```

iii. The subtraction is standard. However, the reference starts time from `timestamps_curr[0]` (the first frame of the trial), while the AI starts from `trial_timestamps[0]` which is the same. The speed filtering means some intermediate timepoints are missing.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Both share the same frame indices. After speed filtering, the same valid_mask is applied to both neural and behavioral data.

ii.
```python
trial_times_valid = trial_timestamps[valid_mask]
trial_neural_valid = trial_neural[valid_mask, :]
```

iii. The speed mask ensures neural and input data have the same timepoints.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. The AI derives environment type from the NWB `identifier` field, which contains the scene name (e.g., `Env1_LocationA_to_B`). The scene name is parsed to determine which environment is active.

ii.
```python
identifier = f['identifier'][()].decode() ...
scene = parse_scene(identifier)
# ...
env_per_trial = get_environment_from_scene(scene, n_trials)
```
```python
def get_environment_from_scene(scene, n_trials, change_trial=SWITCH_TRIAL):
    env = np.zeros(n_trials, dtype=np.int64)
    if '_to_Env' in scene:
        parts = scene.split('_to_')
        env1_num = int(parts[0][3])
        env2_num = int(parts[1][3])
        env[:change_trial] = env1_num - 1
        env[change_trial:] = env2_num - 1
    elif scene.startswith('Env1'):
        env[:] = 0
    elif scene.startswith('Env2'):
        env[:] = 1
```

iii. The AI parses the scene name rather than reading the `environment` behavior time series directly. This relies on correct parsing of all scene name formats and the assumption that environment switches always happen at trial 30.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. Scene name string parsing, including handling of switch days where the environment changes at trial 30. The value is per-trial.

ii. See 4-a code snippets.

iii. The reference reads the `environment` field directly from behavior data, which already contains the correct per-timepoint value, requiring no parsing logic.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is the loop index `i` over trials within each session (0-indexed).

ii.
```python
for i in range(n_trials):
    # ...
    trial_num = float(i)
    input_data[2, :] = trial_num
```

iii. Same approach as reference: sequential 0-indexed trial count.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No processing -- the loop index is assigned directly. The value is constant across all timepoints within a trial.

ii.
```python
trial_num = float(i)
input_data[2, :] = trial_num
```

iii. Matches reference approach.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from `Reward/timestamps` in the behavioral time series. Reward event timestamps are compared against trial time boundaries.

ii.
```python
reward_timestamps = bts['Reward/timestamps'][:]
# ...
for i in range(n_trials):
    trial_time_start = timestamps[ts]
    trial_time_end = timestamps[te] if te < len(timestamps) else timestamps[-1]
    rewards_in_trial = np.sum((reward_timestamps >= trial_time_start) &
                               (reward_timestamps <= trial_time_end))
    reward_per_trial[i] = 1 if rewards_in_trial > 0 else 0
```

iii. Uses reward timestamps directly to find rewards in trial windows.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For each trial, compute whether any reward event timestamp falls within the previous trial's time window. First trial defaults to 0.

ii.
```python
prev_outcome = np.zeros(n_trials, dtype=np.int64)
prev_outcome[1:] = reward_per_trial[:-1]
```

iii. The vectorized shift approach is equivalent to the reference's per-trial loop checking the previous trial.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from the `position` behavior time series and the reward zone location determined by parsing the scene name from the NWB identifier.

ii.
```python
rz_coords, rz_labels = get_reward_zones_from_scene(scene, n_trials)
# ...
rz_start = rz_coords[i, 0]
rz_end = rz_coords[i, 1]
dist = compute_distance_to_reward_zone(trial_pos_valid, rz_start, rz_end)
```

iii. The reward zone boundaries match the reference (`A=[80,130], B=[200,250], C=[320,370]`). The AI derives zone identity from the scene name rather than from the `reward_zone` behavior data.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. For each timepoint, compute signed distance from position to the nearest edge of the reward zone. Distance is 0 inside the zone, negative before, positive after.

ii.
```python
def compute_distance_to_reward_zone(position, rz_start, rz_end):
    distance = np.zeros_like(position)
    before = position < rz_start
    after = position > rz_end
    inside = ~before & ~after
    distance[before] = position[before] - rz_start
    distance[after] = position[after] - rz_end
    distance[inside] = 0.0
    return distance
```

iii. Same logic as reference.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Discretized into 7 bins using explicit conditional assignments.

ii.
```python
def discretize_distance(distance):
    out = np.zeros(len(distance), dtype=np.int64)
    out[distance < -50] = 0
    out[(distance >= -50) & (distance < -10)] = 1
    out[(distance >= -10) & (distance < 0)] = 2
    out[distance == 0] = 3  # in zone
    out[(distance > 0) & (distance <= 10)] = 4
    out[(distance > 10) & (distance <= 50)] = 5
    out[distance > 50] = 6
    return out
```

iii. Bin edges match the instructions. Minor boundary differences from reference (e.g., `<= 10` vs `< 10` at some edges due to reference using `np.digitize` with `1e-6` boundary), but functionally equivalent for continuous position data.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Same frame indices are used after speed-mask filtering.

ii.
```python
trial_pos_valid = trial_pos[valid_mask]
trial_neural_valid = trial_neural[valid_mask, :]
```

iii. Both are indexed by the same valid_mask.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from the `position` behavior time series.

ii.
```python
position = bts['position/data'][:]
# ...
trial_pos = position[ts:te]
```

iii. Same source as reference.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. No processing beyond extracting the per-trial slice, applying speed mask, and discretizing.

ii.
```python
trial_pos_valid = trial_pos[valid_mask]
pos_disc = discretize_position(trial_pos_valid)
```

iii. Same as reference (except speed filtering).

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Discretized into 5 equal bins of 90 cm spanning 0-450 cm.

ii.
```python
def discretize_position(position):
    out = np.zeros(len(position), dtype=np.int64)
    out[position < 90] = 0
    out[(position >= 90) & (position < 180)] = 1
    out[(position >= 180) & (position < 270)] = 2
    out[(position >= 270) & (position < 360)] = 3
    out[position >= 360] = 4
    return out
```

iii. Matches the instruction's bin specification.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same frame indices via valid_mask.

ii. Same as 7-d.

iii. Both neural and position data share the same speed-filtered indices.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from the `lick` behavior time series.

ii.
```python
lick_raw = bts['lick/data'][:]
```

iii. Same source as reference.

## 9-b. What processing is involved in computing `output` *Lick*?

i. The AI applies lick sensor error correction before binarizing. For each trial, if more than 35% of frames have cumulative lick count > 2, all licks in that trial are set to NaN (then to 0). Remaining positive lick values are binarized to 1.

ii.
```python
lick_corrected = lick_raw.copy()
for i in range(n_trials):
    ts = trial_start_inds[i]
    te = teleport_inds[i]
    trial_licks = lick_corrected[ts:te]
    if len(trial_licks) > 0:
        frac_error = np.sum(trial_licks > 2) / len(trial_licks)
        if frac_error > LICK_ERROR_THRESHOLD:
            lick_corrected[ts:te] = np.nan
# Binarize licks
lick_binary = np.zeros_like(lick_corrected)
lick_binary[lick_corrected > 0] = 1
lick_binary[np.isnan(lick_corrected)] = 0
```

iii. The AI found the lick error correction in the reference code (`correct_lick_sensor_error` in `behavior.py`) and applied it. The reference conversion solution does NOT apply this correction, only binarizing with `(licks_curr > 0).astype(int)`.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same speed-filtered indices as other variables.

ii.
```python
trial_lick_valid = trial_lick[valid_mask]
```

iii. Aligned via the common valid_mask.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from the NWB `identifier` field containing the scene name. The scene name is parsed to determine reward zone labels (A, B, or C) per trial.

ii.
```python
scene = parse_scene(identifier)
rz_coords, rz_labels = get_reward_zones_from_scene(scene, n_trials)
# ...
rz_label = REWARD_ZONE_LABEL_MAP[rz_labels[i]]
```

iii. The AI determined that the scene name encodes the reward zone configuration, including switches.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Complex scene name parsing handles multiple formats: single-location (e.g., `Env1_LocationA`), within-environment switch (e.g., `Env1_LocationA_to_B`), and cross-environment switch (e.g., `Env1_B_to_Env2_C`). For switch sessions, the first 30 trials get one zone and remaining trials get the other.

ii.
```python
def get_reward_zones_from_scene(scene, n_trials, change_trial=SWITCH_TRIAL):
    # ... parses various scene name formats
    if '_to_' in scene ...:
        rz_coords[:change_trial] = REWARD_ZONE_DICT[loc1]
        rz_labels[:change_trial] = loc1
        rz_coords[change_trial:] = REWARD_ZONE_DICT[loc2]
        rz_labels[change_trial:] = loc2
```

iii. The reference uses a data-driven Viterbi algorithm on the `reward_zone` behavior time series positions instead of parsing the scene name. The scene-based approach is more fragile but produces equivalent results when parsing is correct.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from the `Reward` time series timestamps.

ii.
```python
reward_timestamps = bts['Reward/timestamps'][:]
# ...
rewards_in_trial = np.sum((reward_timestamps >= trial_time_start) &
                           (reward_timestamps <= trial_time_end))
reward_per_trial[i] = 1 if rewards_in_trial > 0 else 0
```

iii. Same source as reference (Reward timestamps).

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, check if any reward timestamp falls within the trial's time window. Binary per-trial output.

ii.
```python
reward_per_trial[i] = 1 if rewards_in_trial > 0 else 0
# ...
output_data[5, :] = reward_out  # broadcast per-trial
```

iii. Equivalent logic to reference.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases:
- NaN in neural data: replaced with 0 (`np.nan_to_num`)
- Speed below threshold: timepoints removed from trial
- Lick sensor errors: detected and set to 0
- Mismatched trial starts/teleports: trimmed to minimum count
- Empty or very short trials (< 2 valid timepoints): skipped

ii.
```python
trial_neural_valid = np.nan_to_num(trial_neural_valid, nan=0.0)
# ...
if valid_mask.sum() < 2:
    continue
```

iii. The AI is more aggressive in data cleaning than the reference, which primarily handles neural/behavior length mismatches and short trials (< 50 timepoints).

## 13-a. What are the most time-consuming steps of the code?

i. Loading NWB files with h5py (I/O bound), particularly loading the large deconvolved neural data arrays.

ii. N/A (single-pass design)

iii. The AI's single-pass approach avoids the reference's two-pass overhead (survey + convert), but doesn't compute dF/F which is computationally expensive.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop for determining reward delivery (`for i in range(n_trials)`) iterates over trials sequentially. The `previous_trial_outcome` computation is already vectorized with `prev_outcome[1:] = reward_per_trial[:-1]`.

ii. N/A

iii. The code is reasonably efficient with most operations vectorized within trials.

## 13-c. What processing does the code repeat multiple times?

i. The code processes each NWB file in a single pass, so there is no repeated loading. However, some behavioral arrays (e.g., timestamps, position) are loaded and indexed multiple times within `process_session`.

ii. N/A

iii. The single-pass design is efficient.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The speed threshold filtering removes timepoints that the reference conversion retains. This changes the trial structure and is arguably unnecessary since the decoder could learn speed information itself.

ii.
```python
speed_mask = trial_speed >= SPEED_THRESHOLD
valid_mask = speed_mask & neural_nan_mask
```

iii. The speed filtering was applied based on the paper's analysis methodology but is not required for the decoder conversion task.
