# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI finds all NWB files by globbing `data/sub-*/*.nwb`. It uses `h5py` (not `pynwb`) to read each file directly. All behavioral time series, neural data (fluorescence, neuropil, deconvolved), and metadata (iscell, planeIdx, imaging_rate, identifier) are loaded per session.

ii.
```python
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
nwb_files = sorted(glob.glob(os.path.join(data_dir, 'sub-*', '*.nwb')))
...
with h5py.File(nwb_path, 'r') as f:
    subject_id = f['general/subject/subject_id'][()].decode()
    session_id = f['general/session_id'][()].decode()
    identifier = f['identifier'][()].decode()
    imaging_rate = f['general/optophysiology/ImagingPlane/imaging_rate'][()]
    bts = f['processing/behavior/BehavioralTimeSeries']
    position = bts['position/data'][:]
    speed = bts['speed/data'][:]
    ...
    ophys = f['processing/ophys']
    deconv_data = ophys[f'Deconvolved/{plane_name}/data'][:]
    F_data = ophys[f'Fluorescence/{plane_name}/data'][:]
    Fneu_data = ophys[f'Neuropil/{plane_name}/data'][:]
```

iii. The AI chose h5py for direct HDF5 access rather than pynwb. The CONVERSION_NOTES document that 11 subjects and 152 sessions were found matching the paper. The glob pattern captures all NWB files across all subject directories.

## 1-b. How are the data split into subjects?

i. Subjects are identified from the `general/subject/subject_id` field within each NWB file. Unique subjects are collected across all sessions and sorted.

ii.
```python
subjects_set = set()
for nwb_path in nwb_files:
    result = process_session(nwb_path, ...)
    if result is not None:
        subjects_set.add(result['subject_id'])
subjects = sorted(subjects_set)
```

iii. Each NWB file contains the subject ID in its metadata, so subjects are discovered as files are processed. The final count (11) matches the paper.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. All NWB files are processed independently, each producing one session's worth of data.

ii.
```python
nwb_files = sorted(glob.glob(os.path.join(data_dir, 'sub-*', '*.nwb')))
for nwb_path in nwb_files:
    result = process_session(nwb_path, ...)
    if result is not None:
        all_sessions.append(result)
```

iii. The NWB file naming convention (`sub-{mouse}_ses-{session}_behavior+ophys.nwb`) makes each file a distinct session. The total of 152 sessions matches expectations.

## 1-d. How are the data split into trials?

i. Trial boundaries are identified using `trial_start_flag` (trial start) and `teleport_flag` (trial end). The AI uses `np.where(flag > 0)` for both, then takes `n_trials = min(len(starts), len(ends))`. Data is extracted as `s:e` slices.

ii.
```python
trial_start_inds = np.where(trial_start_flag > 0)[0]
teleport_inds = np.where(teleport_flag > 0)[0]
n_trials = min(len(trial_start_inds), len(teleport_inds))
trial_start_inds = trial_start_inds[:n_trials]
teleport_inds = teleport_inds[:n_trials]
...
for i in range(n_trials):
    s = trial_start_inds[i]
    e = teleport_inds[i]
    trial_pos = position[s:e]
    ...
```

iii. The AI noted in the trajectory that `trial_start` and `teleport` are used rather than `trial_number` because the latter did not agree with `trial_start`. The approach uses all positive teleport values rather than detecting teleport onset transitions.

## 1-e. How are trials filtered based on quality controls?

i. Trials with `e <= s` or `(e - s) < 2` timepoints are skipped. These are trivially short or degenerate trials.

ii.
```python
if e <= s or (e - s) < 2:
    trial_start_time = pos_timestamps[s] if s < len(pos_timestamps) else 0
    trial_end_time = pos_timestamps[min(e, len(pos_timestamps) - 1)]
    was_rewarded = detect_reward_in_trial(reward_timestamps, trial_start_time, trial_end_time)
    prev_trial_rewarded = int(was_rewarded)
    continue
```

iii. When a trial is skipped, the previous trial reward outcome is still updated to maintain correct tracking for the next valid trial.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `Deconvolved` events stored in the NWB file under `processing/ophys/Deconvolved/plane{N}/data`.

ii.
```python
for pi, plane_name in enumerate(fluor_planes):
    deconv_data = ophys[f'Deconvolved/{plane_name}/data'][:]
    ...
deconv_all = np.concatenate(deconv_list, axis=1)
```

iii. The AI notes in CONVERSION_NOTES that "the reference decoder (Fig3) uses deconvolved events, NOT dF/F". The deconvolved events are pre-computed in the NWB files via suite2p's OASIS algorithm.

## 2-b. How is the `neural` data processed?

i. Deconvolved data from multiple planes are concatenated. Additionally, the AI computes dF/F from raw fluorescence and neuropil fluorescence (with neuropil subtraction coefficient 0.7, maximin baseline, Gaussian smoothing) for the purpose of interneuron detection. The final neural data uses deconvolved events, not dF/F.

ii.
```python
# Concatenate across planes
deconv_all = np.concatenate(deconv_list, axis=1)
F_all = np.concatenate(F_list, axis=1)
Fneu_all = np.concatenate(Fneu_list, axis=1)
...
# dF/F computed per trial for interneuron detection
F_corr = F_trial - NEUROPIL_COEF * Fneu_trial  # 0.7 coefficient
smoothed = gaussian_filter1d(F_corr, sigma=15, axis=1)
baseline = minimum_filter1d(smoothed, size=min(baseline_window, n_t), axis=1)
baseline = maximum_filter1d(baseline, size=min(baseline_window, n_t), axis=1)
dff = (F_corr - baseline) / abs_baseline
dff = gaussian_filter1d(dff, sigma=DFF_SMOOTH_SIGMA, axis=1)
```

iii. The dF/F computation mirrors the reference code's `preprocessing.dff()` function. The CONVERSION_NOTES document the parameters: neuropil coefficient 0.7, baseline window 300 samples, Gaussian smoothing sigma=2.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters are applied:
1. **iscell filter**: Only ROIs with `iscell[:, 0] == 1` are kept.
2. **Interneuron exclusion**: Cells whose dF/F has Pearson correlation > 0.5 with speed are excluded.

ii.
```python
cell_mask = iscell[:, 0].astype(bool)
...
is_interneuron = detect_interneurons(dff_full, speed)
final_cell_mask = ~is_interneuron
neural_data = deconv_cells[final_cell_mask]
```

```python
def detect_interneurons(dff_all, speed_all, threshold=0.5):
    speed_centered = speed_valid - speed_mean
    dff_centered = dff_valid - dff_mean
    corr = dff_centered @ speed_centered / (dff_std * speed_std + 1e-10)
    is_interneuron = corr > threshold
    return is_interneuron
```

iii. The paper states "Pearson corr >0.5 with speed" removes ~0.42% of cells. The CONVERSION_NOTES document this as matching the reference code's interneuron exclusion criteria.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Data is aligned to trial start by extracting the neural data from `trial_start_inds[i]` to `teleport_inds[i]` for each trial. No additional temporal shifting is needed since the alignment event is the trial start itself.

ii.
```python
trial_neural = neural_data[:, s:e]  # (n_cells, n_timepoints)
```

iii. The instructions specify "Temporally align based on start of the trial." Since data extraction begins at the trial start index, alignment is inherent.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is the native imaging frame period: `1000.0 / effective_rate` ms, where effective_rate is ~15.5 Hz (giving ~64.5 ms bins). For multi-plane animals (m17, m18), the stored rate is ~31 Hz but divided by 2 planes. No temporal rebinning is applied.

ii.
```python
n_planes = len(fluor_planes)
effective_rate = imaging_rate if n_planes == 1 else imaging_rate / n_planes
time_bin_ms = 1000.0 / effective_rate
```

iii. The paper states imaging at ~15.5 Hz. The CONVERSION_NOTES confirm consistent bin sizes across sessions.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Derived from the frame count within the trial and the effective imaging rate. Not directly from timestamps.

ii.
```python
time_from_start = (np.arange(n_t) / effective_rate).astype(np.float32)
```

iii. The AI computes idealized time values from the constant imaging rate rather than using the stored behavioral timestamps.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. For each trial of length `n_t` frames, time is computed as `frame_index / effective_rate`. This gives uniformly spaced time values starting at 0.

ii.
```python
time_from_start = (np.arange(n_t) / effective_rate).astype(np.float32)
input_arr[0, :] = time_from_start
```

iii. Since the imaging rate is constant, this produces the same result as subtracting the first timestamp from all timestamps within a trial.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Neural and behavioral data share the same time indices within each trial (both use `s:e` slicing), so they are inherently aligned. The time_from_start is computed for the same number of timepoints as the neural data.

ii.
```python
n_t = e - s
trial_neural = neural_data[:, s:e]
time_from_start = (np.arange(n_t) / effective_rate).astype(np.float32)
```

iii. Both neural and behavioral data come from the same frame indices, so no additional alignment is needed.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Derived from the `environment` behavioral time series in the NWB file.

ii.
```python
environment = bts['environment/data'][:]
...
trial_env = environment[s:e]
env_val = np.median(trial_env[trial_env >= 0]) if np.any(trial_env >= 0) else 0.0
env_type = np.float32(env_val)
```

iii. The environment variable records the VR environment type (0 or 1) at each timepoint.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The median of non-negative environment values within the trial is taken. This collapses the per-timepoint values to a single per-trial scalar. The value is then broadcast across all timepoints.

ii.
```python
env_val = np.median(trial_env[trial_env >= 0]) if np.any(trial_env >= 0) else 0.0
env_type = np.float32(env_val)
input_arr[1, :] = env_type  # broadcast
```

iii. Using the median provides robustness against any transient values at trial boundaries.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Derived from the loop counter `i` over trials within each session.

ii.
```python
for i in range(n_trials):
    ...
    trial_number = np.float32(i)
    input_arr[2, :] = trial_number
```

iii. The 0-indexed trial counter within a session is used as the trial number, not the `trial number` behavioral time series.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No processing beyond assigning the loop index. The value is constant across all timepoints within a trial (broadcast).

ii.
```python
trial_number = np.float32(i)
input_arr[2, :] = trial_number  # broadcast
```

iii. Simple sequential indexing.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from the `Reward/timestamps` behavioral time series. Reward timestamps are compared against trial boundaries to determine if a reward was delivered.

ii.
```python
reward_timestamps = bts['Reward/timestamps'][:]
...
def detect_reward_in_trial(reward_timestamps, trial_start_time, trial_end_time):
    if len(reward_timestamps) == 0:
        return False
    return np.any((reward_timestamps >= trial_start_time) & (reward_timestamps <= trial_end_time))
```

iii. The reward timestamps have their own time base separate from the behavioral sampling rate.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For each trial, the reward outcome of the *previous* trial is checked. For the first trial, it defaults to 0 (no reward). The code tracks `prev_trial_rewarded` across the trial loop, updating it after each trial. Even skipped trials update this tracker.

ii.
```python
prev_trial_rewarded = 0  # For the first trial
for i in range(n_trials):
    ...
    if e <= s or (e - s) < 2:
        was_rewarded = detect_reward_in_trial(reward_timestamps, trial_start_time, trial_end_time)
        prev_trial_rewarded = int(was_rewarded)
        continue
    ...
    prev_outcome = np.float32(prev_trial_rewarded)
    input_arr[3, :] = prev_outcome  # broadcast
    ...
    prev_trial_rewarded = int(was_rewarded)
```

iii. The previous trial outcome is correctly tracked even when trials are skipped.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from:
1. The `position` behavioral time series (animal's current position)
2. The reward zone coordinates, which are determined from parsing the NWB `identifier` field (scene name) and a hardcoded `change_trial=30` for the reward zone switch point.

ii.
```python
scene = get_scene_from_identifier(identifier)
rz_coords, rz_labels = get_reward_zones(scene, n_trials)
...
rz_start, rz_end = rz_coords[i]
dist_to_rz = compute_distance_to_reward_zone(trial_pos, rz_start, rz_end)
```

Reward zone mapping from scene name:
```python
def get_reward_zones(scene, n_trials, change_trial=DEFAULT_CHANGE_TRIAL):
    if 'Location' in scene and '_to' not in scene:
        loc = scene.split('Location')[-1]
        rz_coords[:] = REWARD_ZONE_DICT[loc]
        rz_labels[:] = loc
    elif 'A_to' in scene and scene[-1] == 'B':
        rz_coords[:change_trial] = REWARD_ZONE_DICT['A']
        rz_labels[:change_trial] = 'A'
        rz_coords[change_trial:] = REWARD_ZONE_DICT['B']
        rz_labels[change_trial:] = 'B'
    ...
```

iii. The AI mirrors the reference code's `behavior.get_reward_zones()` function, parsing scene names to determine reward zone coordinates. This is a deterministic approach based on experimental design rather than data-driven inference.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance is computed: negative when before the reward zone, 0 when inside, positive when past it.

ii.
```python
def compute_distance_to_reward_zone(position, rz_start, rz_end):
    dist = np.zeros_like(position)
    before = position < rz_start
    inside = (position >= rz_start) & (position <= rz_end)
    after = position > rz_end
    dist[before] = position[before] - rz_start  # negative
    dist[inside] = 0.0
    dist[after] = position[after] - rz_end  # positive
    return dist
```

iii. This matches the concept of signed distance to the reward zone as described in the paper.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Discretized into 7 bins using explicit conditional assignments:
- 0: < -50 cm
- 1: [-50, -10) cm
- 2: [-10, 0) cm
- 3: 0 cm (inside reward zone)
- 4: (0, 10] cm
- 5: (10, 50] cm
- 6: > 50 cm

ii.
```python
def discretize_distance_to_rz(dist):
    out = np.zeros(len(dist), dtype=np.int64)
    out[dist < -50] = 0
    out[(dist >= -50) & (dist < -10)] = 1
    out[(dist >= -10) & (dist < 0)] = 2
    out[dist == 0] = 3
    out[(dist > 0) & (dist <= 10)] = 4
    out[(dist > 10) & (dist <= 50)] = 5
    out[dist > 50] = 6
    return out
```

iii. The bin edges match the instructions. The explicit conditional approach produces equivalent results to the reference's `np.digitize` approach.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position and neural data use the same `s:e` indices within each trial, so no additional alignment is needed.

ii.
```python
trial_pos = position[s:e]
trial_neural = neural_data[:, s:e]
dist_to_rz = compute_distance_to_reward_zone(trial_pos, rz_start, rz_end)
```

iii. Both data streams share the same frame indexing.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from the `position` behavioral time series.

ii.
```python
position = bts['position/data'][:]
...
trial_pos = position[s:e]
pos_binned = discretize_position(trial_pos)
```

iii. The position variable records the animal's location in the VR corridor in cm.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. No processing beyond extracting the per-trial slice and discretizing.

ii.
```python
trial_pos = position[s:e]
pos_binned = discretize_position(trial_pos)
```

iii. Raw position values are used directly.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Discretized into 5 equal bins using `np.linspace(0, 450, 6)` giving bin edges `[0, 90, 180, 270, 360, 450]`, then clipped to `[0, 4]`.

ii.
```python
def discretize_position(position, n_bins=POSITION_BINS):
    bin_edges = np.linspace(0, TRACK_LENGTH, n_bins + 1)  # [0, 90, 180, 270, 360, 450]
    binned = np.digitize(position, bin_edges) - 1
    binned = np.clip(binned, 0, n_bins - 1)
    return binned
```

iii. The AI interprets "5 equal-sized bins" as dividing the 450 cm track into 5 bins of 90 cm each. Positions outside [0, 450] are clipped to the edge bins.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same frame indices as neural data within each trial.

ii.
```python
trial_pos = position[s:e]
trial_neural = neural_data[:, s:e]
```

iii. Inherent alignment from shared indexing.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from the `lick` behavioral time series.

ii.
```python
lick_raw = bts['lick/data'][:]
...
trial_lick = lick[s:e].copy()
```

iii. The lick variable records lick events at each timepoint.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Two processing steps:
1. **Lick sensor error correction**: If >35% of samples in a trial have lick count > 2, all lick values for that trial are set to 0.
2. **Binarization**: Lick values > 1 are capped at 1, then any positive value is mapped to 1.

ii.
```python
if np.sum(trial_lick > 2) / n_t > LICK_ERROR_FRACTION_THRESHOLD:
    trial_lick[:] = 0  # set to 0 instead of NaN for decoder output
trial_lick[trial_lick > 1] = 1
trial_lick = (trial_lick > 0).astype(np.float32)
```

iii. The lick sensor error correction mirrors the reference code's approach documented in the paper. The CONVERSION_NOTES cite a threshold of 35% (from the reference code, vs 30% mentioned in the paper text).

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same frame indices as neural data within each trial.

ii.
```python
trial_lick = lick[s:e].copy()
trial_neural = neural_data[:, s:e]
```

iii. Inherent alignment from shared indexing.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from the NWB `identifier` field, which contains the scene name (e.g., `Env1_LocationB_to_A`). The scene name is parsed to determine which reward zone(s) are active and when they switch (at trial 30).

ii.
```python
identifier = f['identifier'][()].decode()
scene = get_scene_from_identifier(identifier)
rz_coords, rz_labels = get_reward_zones(scene, n_trials)
...
rz_loc = rz_label_to_idx(rz_labels[i])  # A=0, B=1, C=2
output_arr[4, :] = rz_loc
```

iii. This mirrors the reference code's `behavior.get_reward_zones()` function which uses scene name parsing and a default change trial of 30.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Scene name is parsed to identify reward zone transitions. For switch sessions (e.g., `Env1_LocationA_to_B`), trials before `change_trial=30` get the first zone label, trials after get the second. The zone label is converted to an index (A=0, B=1, C=2).

ii.
```python
def get_reward_zones(scene, n_trials, change_trial=DEFAULT_CHANGE_TRIAL):
    if 'A_to' in scene and scene[-1] == 'B':
        rz_coords[:change_trial] = REWARD_ZONE_DICT['A']
        rz_labels[:change_trial] = 'A'
        rz_coords[change_trial:] = REWARD_ZONE_DICT['B']
        rz_labels[change_trial:] = 'B'
    ...
```

iii. The AI correctly handles both single-location sessions and switch sessions, including cross-environment switches (e.g., `Env1_A_to_Env2_B`).

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from the `Reward/timestamps` behavioral time series. These timestamps indicate when reward was delivered.

ii.
```python
reward_timestamps = bts['Reward/timestamps'][:]
...
was_rewarded = detect_reward_in_trial(reward_timestamps, trial_start_time, trial_end_time)
```

iii. The reward time series has its own timestamps separate from the behavioral sampling rate.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, checks if any reward timestamp falls within the trial's time window `[trial_start_time, trial_end_time]`. Returns binary 0/1. The value is constant across all timepoints in the trial.

ii.
```python
def detect_reward_in_trial(reward_timestamps, trial_start_time, trial_end_time):
    if len(reward_timestamps) == 0:
        return False
    return np.any((reward_timestamps >= trial_start_time) & (reward_timestamps <= trial_end_time))
...
trial_start_time = trial_timestamps[0]
trial_end_time = trial_timestamps[-1]
was_rewarded = detect_reward_in_trial(reward_timestamps, trial_start_time, trial_end_time)
output_arr[5, :] = reward_outcome  # broadcast
```

iii. The CONVERSION_NOTES show ~15% omission rate, matching the paper's ~15% reward omission.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **Neural/behavior length mismatch**: If neural and behavioral data have different lengths (common in multi-plane recordings), data is cropped to the minimum.
- **Very short trials**: Trials with `e <= s` or `(e - s) < 2` are skipped.
- **Lick sensor errors**: Trials with >35% of samples having lick > 2 have licks zeroed out.
- **Sessions with too few cells**: Sessions with < 2 cells after filtering are skipped.
- **Sessions with too few trials**: Sessions with < 2 trials are skipped.

ii.
```python
n_timepoints_total = min(n_timepoints_neural, n_timepoints_behav)
if n_timepoints_neural != n_timepoints_behav:
    deconv_all = deconv_all[:n_timepoints_total]
    position = position[:n_timepoints_total]
    ...
if e <= s or (e - s) < 2:
    continue
if n_final_cells < 2:
    return None
```

iii. These defensive checks were developed during data exploration and validation.

## 13-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are:
1. **Loading NWB files** via h5py and reading large arrays (fluorescence, neuropil, deconvolved)
2. **Computing dF/F for interneuron detection** — requires processing raw fluorescence per trial
3. **Interneuron detection** — speed-dFF correlation computation (originally slow, vectorized for performance)

ii. N/A

iii. The trajectory shows the AI optimized interneuron detection from 18.2s to 7.1s for 2 sessions by vectorizing the Pearson correlation computation.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `process_session` iterates sequentially over trials to extract data, compute inputs/outputs. The dF/F computation loop over trials could potentially be vectorized but is constrained by per-trial baseline computation. The interneuron detection was already vectorized.

ii. N/A

iii. The variable trial lengths make full vectorization difficult. The AI prioritized vectorizing the interneuron detection as the main bottleneck.

## 13-c. What processing does the code repeat multiple times?

i. Each NWB file is loaded once (unlike a survey-then-convert approach that would load each file twice). However, the fluorescence data (F, Fneu) is loaded for every session even though it's only needed for interneuron detection, and the deconvolved data is separately loaded alongside it.

ii. N/A

iii. The AI processes each file in a single pass, which is more efficient than a two-pass survey + conversion approach.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes dF/F from raw fluorescence and neuropil data solely for interneuron detection. The dF/F values themselves are discarded after interneuron detection; only the deconvolved events are kept as neural data. This adds significant computation (per-trial maximin baseline, Gaussian smoothing) that the reference solution avoids entirely by not performing interneuron exclusion.

ii.
```python
# dF/F computed only for interneuron detection, then discarded
dff_full = np.full_like(F_cells, np.nan)
for i in range(n_trials):
    dff_full[:, s:e] = compute_dff_trial(F_cells[:, s:e], Fneu_cells[:, s:e])
is_interneuron = detect_interneurons(dff_full, speed)
# dff_full is never used again
```

iii. The dF/F computation is the main additional processing burden compared to the reference.
